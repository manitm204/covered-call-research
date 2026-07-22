"""Aux-data ingestion: parsers, scaling, manifests, bundle loading.

All tests are offline: parsers consume inline CSV text and ingestion uses an
injected fake fetcher. The real network pull happens only via `xsp ingest-aux`.
"""

import json
from datetime import date

import polars as pl
import pytest

from xsp_research.ingestion.aux_data import (
    DEFAULT_UNIVERSE,
    SECTOR_SYMBOLS,
    AuxSeriesSpec,
    ingest_aux_bundle,
    load_aux_bundle,
    parse_cboe_index_csv,
    parse_fred_csv,
    parse_yahoo_chart_json,
    source_url,
)

FRED_CSV = """DATE,VIXCLS
2023-01-03,22.90
2023-01-04,22.01
2023-01-05,.
2023-01-06,21.13
"""

FRED_CSV_NEW_HEADER = """observation_date,DTB4WK
2023-01-03,4.30
2023-01-04,4.32
"""

YAHOO_JSON = """{"chart":{"result":[{"meta":{"symbol":"SPY"},
"timestamp":[1672756200,1672842600],
"indicators":{"quote":[{"close":[380.82,383.76]}],
"adjclose":[{"adjclose":[375.10,378.00]}]}}],"error":null}}"""

YAHOO_JSON_NO_ADJ = """{"chart":{"result":[{"meta":{"symbol":"^GSPC"},
"timestamp":[1672756200,1672842600],
"indicators":{"quote":[{"close":[3824.14,3852.97]}]}}],"error":null}}"""

YAHOO_ERROR_JSON = """{"chart":{"result":null,"error":{"code":"Not Found"}}}"""

CBOE_CSV = """DATE,OPEN,HIGH,LOW,CLOSE
2023-01-03,23.09,23.76,22.55,22.71
2023-01-04,22.87,23.11,21.80,22.01
"""

CBOE_CSV_SINGLE_VALUE = """DATE,VVIX
03/06/2023,91.730000
03/07/2023,89.210000
"""


class TestParsers:
    def test_fred_parses_and_drops_missing(self):
        df = parse_fred_csv(FRED_CSV)
        assert df.height == 3  # '.' row dropped
        assert df["date"][0] == date(2023, 1, 3)
        assert df["close"][0] == pytest.approx(22.90)

    def test_fred_new_header_variant(self):
        df = parse_fred_csv(FRED_CSV_NEW_HEADER)
        assert df.height == 2
        assert df["close"][1] == pytest.approx(4.32)

    def test_yahoo_prefers_adjusted_close(self):
        df = parse_yahoo_chart_json(YAHOO_JSON)
        assert df.height == 2
        assert df["close"].to_list() == pytest.approx([375.10, 378.00])
        assert df["date"][0] == date(2023, 1, 3)

    def test_yahoo_falls_back_to_raw_close(self):
        df = parse_yahoo_chart_json(YAHOO_JSON_NO_ADJ)
        assert df["close"].to_list() == pytest.approx([3824.14, 3852.97])

    def test_yahoo_error_raises(self):
        with pytest.raises(ValueError, match="Yahoo chart error"):
            parse_yahoo_chart_json(YAHOO_ERROR_JSON)

    def test_cboe_ohlc_layout(self):
        df = parse_cboe_index_csv(CBOE_CSV)
        assert df.height == 2
        assert df["close"][1] == pytest.approx(22.01)

    def test_cboe_single_value_layout_with_us_dates(self):
        df = parse_cboe_index_csv(CBOE_CSV_SINGLE_VALUE)
        assert df.height == 2
        assert df["date"][0] == date(2023, 3, 6)  # 03/06/2023 parsed as US format
        assert df["close"][0] == pytest.approx(91.73)

    def test_bad_shape_raises(self):
        with pytest.raises(ValueError):
            parse_cboe_index_csv("foo\n1\n")


class TestSourceUrls:
    def test_known_sources(self):
        fred = AuxSeriesSpec("VIX", "fred", "VIXCLS", "")
        yahoo = AuxSeriesSpec("SPY", "yahoo", "SPY", "")
        cboe = AuxSeriesSpec("VVIX", "cboe_index", "VVIX", "")
        assert "fredgraph.csv?id=VIXCLS" in source_url(fred)
        assert "query1.finance.yahoo.com" in source_url(yahoo)
        assert "VVIX_History.csv" in source_url(cboe)

    def test_unknown_source_raises(self):
        with pytest.raises(ValueError):
            source_url(AuxSeriesSpec("X", "nope", "x", ""))


def fake_fetcher(spec: AuxSeriesSpec) -> pl.DataFrame:
    """Deterministic fake series; applies spec.scale like the real fetcher."""
    df = pl.DataFrame(
        {
            "date": [date(2022, 12, 30), date(2023, 1, 3), date(2023, 1, 4)],
            "close": [100.0, 101.0, 102.0],
        }
    )
    if spec.scale != 1.0:
        df = df.with_columns((pl.col("close") * spec.scale).alias("close"))
    return df


class TestIngestBundle:
    def test_writes_parquet_and_manifest(self, tmp_path):
        manifest = ingest_aux_bundle(
            tmp_path / "aux",
            DEFAULT_UNIVERSE,
            start=date(2023, 1, 1),
            manifest_dir=tmp_path / "manifests",
            fetcher=fake_fetcher,
        )
        assert manifest["errors"] == {}
        assert len(manifest["series"]) == len(DEFAULT_UNIVERSE)
        # start filter applied
        assert manifest["series"]["SPY"]["date_min"] == "2023-01-03"
        # scaling recorded and applied: UNDERLYING = SPX/10
        und = pl.read_parquet(tmp_path / "aux" / "UNDERLYING.parquet")
        spx = pl.read_parquet(tmp_path / "aux" / "SPX.parquet")
        assert und["close"].to_list() == pytest.approx([c / 10 for c in spx["close"].to_list()])
        # rate stored as decimal (percent * 0.01)
        rate = pl.read_parquet(tmp_path / "aux" / "RATE_3M.parquet")
        assert rate["close"][0] == pytest.approx(1.01)  # fake 101.0 * 0.01
        on_disk = json.loads((tmp_path / "manifests" / "aux_bundle.manifest.json").read_text())
        assert on_disk["series"]["VIX"]["sha256"]

    def test_rates_file_written_for_backtester(self, tmp_path):
        ingest_aux_bundle(
            tmp_path / "aux",
            DEFAULT_UNIVERSE,
            manifest_dir=tmp_path / "manifests",
            fetcher=fake_fetcher,
        )
        rates = pl.read_parquet(tmp_path / "rates" / "tbill_4w.parquet")
        assert rates.columns == ["date", "rate"]

    def test_partial_failure_recorded_not_fatal(self, tmp_path):
        def flaky(spec):
            if spec.symbol == "VVIX":
                raise ValueError("host unreachable")
            return fake_fetcher(spec)

        manifest = ingest_aux_bundle(
            tmp_path / "aux",
            DEFAULT_UNIVERSE,
            manifest_dir=tmp_path / "manifests",
            fetcher=flaky,
        )
        assert "VVIX" in manifest["errors"]
        assert "VIX" in manifest["series"]  # others unaffected
        assert not (tmp_path / "aux" / "VVIX.parquet").exists()  # nothing fabricated


class TestLoadBundle:
    def test_round_trip_into_market_data_bundle(self, tmp_path):
        ingest_aux_bundle(
            tmp_path / "aux",
            DEFAULT_UNIVERSE,
            manifest_dir=tmp_path / "manifests",
            fetcher=fake_fetcher,
        )
        bundle = load_aux_bundle(tmp_path / "aux")
        assert bundle.has("UNDERLYING", "VIX", "SPY", "RSP", "RATE_3M")
        assert bundle.sector_symbols == SECTOR_SYMBOLS
        assert bundle.calendar()  # UNDERLYING present and non-empty

    def test_missing_underlying_raises(self, tmp_path):
        (tmp_path / "empty").mkdir()
        with pytest.raises(ValueError, match="UNDERLYING"):
            load_aux_bundle(tmp_path / "empty")
