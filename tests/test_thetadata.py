"""ThetaData adapter: parsing, normalization, and pull orchestration — all
offline via an injected http_get. Live behavior is exercised by the CLI."""

import urllib.error
from datetime import date

import polars as pl
import pytest

from xsp_research.ingestion.base import CHAIN_SCHEMA, validate_chain_frame
from xsp_research.ingestion.thetadata import (
    ThetaDataClient,
    ThetaDataError,
    implied_underlying_from_quotes,
    normalize_snapshot,
    parse_open_interest_csv,
    parse_option_quote_csv,
    pull_chain_history,
)


def parity_chain_csv(spot: float, expiration: str = "2024-06-05") -> str:
    """Chain CSV whose C/P mids satisfy parity exactly (r=0): C - P = S - K."""
    lines = ["symbol,expiration,strike,right,bid,ask"]
    for k in range(int(spot) - 5, int(spot) + 6):
        c = max(spot - k, 0.0) + 1.50  # intrinsic + common extrinsic
        p = c - (spot - k)  # parity with r=0
        lines.append(f"SPY,{expiration},{k}.000,call,{c - 0.05:.2f},{c + 0.05:.2f}")
        lines.append(f"SPY,{expiration},{k}.000,put,{p - 0.05:.2f},{p + 0.05:.2f}")
    return "\n".join(lines)


QUOTE_CSV = """symbol,expiration,strike,right,timestamp,bid_size,bid,ask_size,ask
SPY,2024-07-19,550.000,call,2024-06-18T15:30:00-04:00,120,4.10,95,4.15
SPY,2024-07-19,555.000,call,2024-06-18T15:30:00-04:00,80,2.05,60,2.08
SPY,2024-07-19,550.000,put,2024-06-18T15:30:00-04:00,150,5.20,110,5.25
"""

QUOTE_CSV_YYYYMMDD = """symbol,expiration,strike,right,bid,ask
SPY,20240719,550.000,C,4.10,4.15
"""

OI_CSV = """symbol,expiration,strike,right,open_interest
SPY,2024-07-19,550.000,call,15000
SPY,2024-07-19,555.000,call,9000
SPY,2024-07-19,550.000,put,22000
"""

STOCK_CSV = """symbol,timestamp,bid_size,bid,ask_size,ask
SPY,2024-06-18T15:30:00-04:00,500,547.10,400,547.12
"""

SESSION = date(2024, 6, 18)


class TestParsing:
    def test_quote_csv_parsed(self):
        df = parse_option_quote_csv(QUOTE_CSV)
        assert df.height == 3
        assert set(df["option_type"].to_list()) == {"C", "P"}
        assert df["expiration"][0] == date(2024, 7, 19)
        assert df["strike"][0] == pytest.approx(550.0)
        assert df["bid_size"][0] == 120

    def test_quote_csv_alt_formats(self):
        df = parse_option_quote_csv(QUOTE_CSV_YYYYMMDD)
        assert df["expiration"][0] == date(2024, 7, 19)
        assert df["option_type"][0] == "C"
        assert df["bid_size"][0] is None  # sizes absent -> null, not fabricated

    def test_empty_response_is_empty_frame(self):
        assert parse_option_quote_csv("").is_empty()

    def test_unexpected_columns_raise(self):
        with pytest.raises(ThetaDataError, match="missing"):
            parse_option_quote_csv("foo,bar\n1,2\n")

    def test_open_interest_parsed(self):
        df = parse_open_interest_csv(OI_CSV)
        assert df.height == 3
        assert df["open_interest"].to_list() == [15000, 9000, 22000]


class TestNormalization:
    def test_canonical_schema_with_oi_join(self):
        chain = normalize_snapshot(
            parse_option_quote_csv(QUOTE_CSV),
            parse_open_interest_csv(OI_CSV),
            underlying_price=547.11,
            session=SESSION,
            snapshot_et="15:30:00",
            root="SPY",
        )
        assert chain.columns == list(CHAIN_SCHEMA)
        validate_chain_frame(chain)
        assert chain.height == 3
        assert (chain["underlying_price"] == 547.11).all()
        row = chain.filter((pl.col("strike") == 550.0) & (pl.col("option_type") == "C")).row(
            0, named=True
        )
        assert row["open_interest"] == 15000
        assert row["volume"] is None  # quotes carry no volume; never invented

    def test_snapshot_timestamp_is_et_converted_to_utc(self):
        chain = normalize_snapshot(
            parse_option_quote_csv(QUOTE_CSV), pl.DataFrame(), 547.11, SESSION, "15:30:00", "SPY"
        )
        ts = chain["ts"][0]
        # June (EDT, UTC-4): 15:30 ET -> 19:30 UTC.
        assert (ts.hour, ts.minute) == (19, 30)

    def test_missing_oi_yields_nulls(self):
        chain = normalize_snapshot(
            parse_option_quote_csv(QUOTE_CSV), pl.DataFrame(), 547.11, SESSION, "15:30:00", "SPY"
        )
        assert chain["open_interest"].null_count() == chain.height

    def test_empty_quotes_yield_typed_empty_frame(self):
        chain = normalize_snapshot(
            pl.DataFrame(), pl.DataFrame(), 547.11, SESSION, "15:30:00", "SPY"
        )
        assert chain.is_empty() and chain.columns == list(CHAIN_SCHEMA)


class TestImpliedUnderlying:
    def test_recovers_spot_from_parity(self):
        quotes = parse_option_quote_csv(parity_chain_csv(spot=547.30))
        implied = implied_underlying_from_quotes(quotes, date(2024, 6, 3), rate=0.0)
        assert implied == pytest.approx(547.30, abs=0.02)

    def test_rate_correction_applied(self):
        # Same quotes; with r>0 the implied spot shifts by K(1-e^{-rT}).
        quotes = parse_option_quote_csv(parity_chain_csv(spot=550.0))
        no_rate = implied_underlying_from_quotes(quotes, date(2024, 6, 3), rate=0.0)
        with_rate = implied_underlying_from_quotes(quotes, date(2024, 6, 3), rate=0.05)
        assert with_rate < no_rate  # discounting lowers K*e^{-rT}

    def test_too_few_pairs_returns_none(self):
        quotes = parse_option_quote_csv(QUOTE_CSV)  # single C/P pair
        assert implied_underlying_from_quotes(quotes, date(2024, 6, 3)) is None

    def test_empty_returns_none(self):
        assert implied_underlying_from_quotes(pl.DataFrame(), date(2024, 6, 3)) is None


def make_fake_http(responses: dict[str, str]):
    """Route by endpoint path substring; record requested URLs."""
    calls: list[str] = []

    def fake_get(url: str) -> str:
        calls.append(url)
        for key, text in responses.items():
            if key in url:
                return text
        raise AssertionError(f"unexpected URL {url}")

    return fake_get, calls


class TestClient:
    def test_snapshot_request_shape(self):
        fake, calls = make_fake_http({"/v3/option/history/quote": QUOTE_CSV})
        client = ThetaDataClient(base_url="http://test", http_get=fake)
        df = client.option_quote_snapshot("SPY", SESSION, "15:30:00", max_dte=70)
        assert df.height == 3
        url = calls[0]
        for fragment in (
            "symbol=SPY",
            "expiration=%2A",
            "right=both",
            "date=20240618",
            "interval=1m",
            "start_time=15%3A30%3A00",
            "end_time=15%3A31%3A00",
            "max_dte=70",
        ):
            assert fragment in url, url

    def test_stock_mid(self):
        fake, _ = make_fake_http({"/v3/stock/history/quote": STOCK_CSV})
        client = ThetaDataClient(base_url="http://test", http_get=fake)
        assert client.stock_quote_mid("SPY", SESSION) == pytest.approx(547.11)

    def test_transient_timeout_retried_then_succeeds(self):
        """Terminal reconnect stalls must not kill a multi-hour pull."""
        attempts: list[int] = []

        def flaky_get(url: str) -> str:
            attempts.append(1)
            if len(attempts) < 3:
                raise TimeoutError("timed out")
            return QUOTE_CSV

        client = ThetaDataClient(base_url="http://test", http_get=flaky_get, retry_wait_s=0.0)
        df = client.option_quote_snapshot("SPY", SESSION)
        assert df.height == 3
        assert len(attempts) == 3

    def test_http_error_never_retried(self):
        """403 (subscription tier) is a definitive answer, not a transient."""
        attempts: list[int] = []

        def forbidden_get(url: str) -> str:
            attempts.append(1)
            raise urllib.error.HTTPError(url, 403, "Forbidden", None, None)  # type: ignore[arg-type]

        client = ThetaDataClient(base_url="http://test", http_get=forbidden_get, retry_wait_s=0.0)
        with pytest.raises(ThetaDataError, match="403"):
            client.option_quote_snapshot("SPY", SESSION)
        assert len(attempts) == 1

    def test_persistent_timeout_exhausts_retries(self):
        attempts: list[int] = []

        def dead_get(url: str) -> str:
            attempts.append(1)
            raise TimeoutError("timed out")

        # base_url port 1 → the terminal_running probe fails fast too.
        client = ThetaDataClient(base_url="http://127.0.0.1:1", http_get=dead_get, retry_wait_s=0.0)
        with pytest.raises(ThetaDataError):
            client.option_quote_snapshot("SPY", SESSION)
        assert len(attempts) == 4


class TestPullOrchestration:
    def _client(self):
        fake, calls = make_fake_http(
            {
                "/v3/option/history/quote": QUOTE_CSV,
                "/v3/option/history/open_interest": OI_CSV,
                "/v3/stock/history/quote": STOCK_CSV,
            }
        )
        return ThetaDataClient(base_url="http://test", http_get=fake), calls

    def test_pull_writes_monthly_files_and_manifest(self, tmp_path):
        client, _ = self._client()
        manifest = pull_chain_history(
            client,
            "SPY",
            date(2024, 6, 3),
            date(2024, 7, 5),
            tmp_path,
            progress=lambda m: None,
        )
        assert set(manifest["months_written"]) == {"2024-06", "2024-07"}
        assert manifest["session_errors"] == {}
        june = pl.read_parquet(tmp_path / "chain_2024-06.parquet")
        validate_chain_frame(june)
        assert (tmp_path / "pull_manifest.json").exists()

    def test_resumable_skips_existing_months(self, tmp_path):
        client, calls = self._client()
        pull_chain_history(
            client,
            "SPY",
            date(2024, 6, 3),
            date(2024, 6, 28),
            tmp_path,
            progress=lambda m: None,
        )
        n_first = len(calls)
        manifest = pull_chain_history(
            client,
            "SPY",
            date(2024, 6, 3),
            date(2024, 7, 5),
            tmp_path,
            progress=lambda m: None,
        )
        assert "2024-06" in manifest["months_skipped_existing"]
        # Second run only fetched July sessions.
        assert len(calls) - n_first < n_first

    def test_no_stock_quote_and_too_few_pairs_records_error(self, tmp_path):
        """QUOTE_CSV has only one C/P pair: parity fallback must refuse to
        guess and the session must be recorded as an error."""
        fake, _ = make_fake_http(
            {
                "/v3/option/history/quote": QUOTE_CSV,
                "/v3/option/history/open_interest": OI_CSV,
                "/v3/stock/history/quote": "",  # no underlying data
            }
        )
        client = ThetaDataClient(base_url="http://test", http_get=fake)
        manifest = pull_chain_history(
            client,
            "SPY",
            date(2024, 6, 3),
            date(2024, 6, 4),
            tmp_path,
            progress=lambda m: None,
            rate_lookup=lambda d: 0.0,
        )
        assert manifest["months_written"] == {}
        assert all("underlying" in e for e in manifest["session_errors"].values())

    def test_stock_403_falls_back_to_parity(self, tmp_path):
        def fake(url: str) -> str:
            if "/v3/stock/history/quote" in url:
                raise OSError("HTTP Error 403: Forbidden")
            if "/v3/option/history/quote" in url:
                return parity_chain_csv(spot=550.0)
            raise AssertionError(url)

        # terminal_running() probes a real URL on failure paths; base_url is
        # unreachable so the client wraps the 403 as ThetaDataError("...403...").
        client = ThetaDataClient(base_url="http://127.0.0.1:1", http_get=fake)
        manifest = pull_chain_history(
            client,
            "SPY",
            date(2024, 6, 3),
            date(2024, 6, 3),
            tmp_path,
            with_open_interest=False,
            progress=lambda m: None,
            rate_lookup=lambda d: 0.0,
        )
        assert manifest["underlying_price_methods"]["parity_implied"] == 1
        chain = pl.read_parquet(tmp_path / "chain_2024-06.parquet")
        assert chain["underlying_price"][0] == pytest.approx(550.0, abs=0.05)

    def test_holidays_recorded_as_empty_sessions(self, tmp_path):
        fake, _ = make_fake_http(
            {
                "/v3/option/history/quote": "",  # holiday: no rows
                "/v3/stock/history/quote": STOCK_CSV,
            }
        )
        client = ThetaDataClient(base_url="http://test", http_get=fake)
        manifest = pull_chain_history(
            client,
            "SPY",
            date(2024, 6, 3),
            date(2024, 6, 4),
            tmp_path,
            with_open_interest=False,
            progress=lambda m: None,
        )
        assert len(manifest["empty_sessions"]) == 2


class TestPartialMonthHealing:
    def test_partial_month_repulled_on_resume(self, tmp_path):
        """A month file with far fewer sessions than weekdays (interrupted
        pull) must NOT be skipped by the resume."""
        fake, calls = make_fake_http(
            {
                "/v3/option/history/quote": QUOTE_CSV,
                "/v3/option/history/open_interest": OI_CSV,
                "/v3/stock/history/quote": STOCK_CSV,
            }
        )
        client = ThetaDataClient(base_url="http://test", http_get=fake)
        # Seed a partial June file: one session only.
        partial = normalize_snapshot(
            parse_option_quote_csv(QUOTE_CSV),
            parse_open_interest_csv(OI_CSV),
            547.11,
            date(2024, 6, 3),
            "15:30:00",
            "SPY",
        )
        partial.write_parquet(tmp_path / "chain_2024-06.parquet")

        manifest = pull_chain_history(
            client,
            "SPY",
            date(2024, 6, 3),
            date(2024, 6, 28),
            tmp_path,
            progress=lambda m: None,
        )
        assert "2024-06" not in manifest["months_skipped_existing"]
        assert "2024-06" in manifest["months_written"]
        healed = pl.read_parquet(tmp_path / "chain_2024-06.parquet")
        assert healed["ts"].dt.date().n_unique() == 20  # all June weekdays

    def test_complete_month_still_skipped(self, tmp_path):
        fake, calls = make_fake_http(
            {
                "/v3/option/history/quote": QUOTE_CSV,
                "/v3/option/history/open_interest": OI_CSV,
                "/v3/stock/history/quote": STOCK_CSV,
            }
        )
        client = ThetaDataClient(base_url="http://test", http_get=fake)
        pull_chain_history(
            client,
            "SPY",
            date(2024, 6, 3),
            date(2024, 7, 5),
            tmp_path,
            progress=lambda m: None,
        )
        n_before = len(calls)
        manifest = pull_chain_history(
            client,
            "SPY",
            date(2024, 6, 3),
            date(2024, 7, 5),
            tmp_path,
            progress=lambda m: None,
        )
        assert "2024-06" in manifest["months_skipped_existing"]
        # Only the current end month (July) is re-pulled; June cost zero requests.
        assert len(calls) < n_before * 2
