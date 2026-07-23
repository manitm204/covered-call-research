"""ThetaData v3 adapter: terminal management, bulk snapshot pulls, normalization.

Access model: ThetaData runs a local *Theta Terminal* (Java 21+) that
authenticates with the API key from `.env` and serves a REST API on
localhost. This module can download/launch the terminal, then pulls one
15:30 ET quote snapshot per session for every option contract of a symbol
(bulk `expiration=*`), joins daily open interest and the underlying quote,
and normalizes into the canonical chain schema with per-month Parquet files
and manifests.

Design rules:
- The HTTP layer is injectable; all parsing/normalization is unit-tested
  offline against canned responses.
- Sessions with no data (holidays) are skipped and recorded — never invented.
- A session whose underlying quote cannot be fetched is recorded as an error
  for that session; we never guess the underlying price.
- Pulls are resumable: completed months are skipped unless --refresh.

Licensing: downloaded data is for personal research use, never committed
(data/ is git-ignored) and never redistributed.
"""

from __future__ import annotations

import json
import subprocess
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import polars as pl

from xsp_research.env import thetadata_api_key, thetadata_base_url
from xsp_research.ingestion.base import CHAIN_SCHEMA, validate_chain_frame

ET = ZoneInfo("America/New_York")
UTC = ZoneInfo("UTC")

TERMINAL_JAR_URL = "https://downloads.thetadata.us/ThetaTerminalv3.jar"
DEFAULT_JAR_DIR = Path.home() / ".local" / "share" / "xsp-research"


class ThetaDataError(RuntimeError):
    pass


class TerminalNotRunningError(ThetaDataError):
    def __init__(self, base_url: str) -> None:
        super().__init__(
            f"Theta Terminal is not responding at {base_url}.\n"
            "Start it with:  xsp terminal-start\n"
            "or manually:    java -jar ThetaTerminalv3.jar --api-key <key>\n"
            f"(jar download: {TERMINAL_JAR_URL}; requires Java 21+)"
        )


# ------------------------------------------------------------------ terminal
def download_terminal_jar(jar_dir: Path = DEFAULT_JAR_DIR) -> Path:
    jar_dir.mkdir(parents=True, exist_ok=True)
    jar = jar_dir / "ThetaTerminalv3.jar"
    if not jar.exists():
        urllib.request.urlretrieve(TERMINAL_JAR_URL, jar)  # noqa: S310 - fixed https URL
    return jar


def launch_terminal(
    jar: Path | None = None,
    api_key: str | None = None,
    log_path: Path | None = None,
    wait_s: float = 90.0,
    base_url: str | None = None,
) -> subprocess.Popen:
    """Start the terminal detached and wait until its REST port answers."""
    api_key = api_key or thetadata_api_key()
    if not api_key:
        raise ThetaDataError("THETADATA_API_KEY is not set (see .env)")
    jar = jar or download_terminal_jar()
    log_path = log_path or (jar.parent / "theta_terminal.log")
    log = open(log_path, "ab")  # noqa: SIM115 - handed to the child process
    proc = subprocess.Popen(
        ["java", "-jar", str(jar), "--api-key", api_key],
        cwd=jar.parent,
        stdout=log,
        stderr=subprocess.STDOUT,
    )
    base = base_url or thetadata_base_url()
    deadline = time.monotonic() + wait_s
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            raise ThetaDataError(
                f"Theta Terminal exited immediately (code {proc.returncode}); see {log_path}"
            )
        if terminal_running(base):
            return proc
        time.sleep(2.0)
    raise ThetaDataError(f"Theta Terminal did not answer within {wait_s:.0f}s; see {log_path}")


def terminal_running(base_url: str | None = None) -> bool:
    base = base_url or thetadata_base_url()
    try:
        req = urllib.request.Request(f"{base}/v3/option/list/expirations?symbol=SPY")
        with urllib.request.urlopen(req, timeout=5) as resp:
            return resp.status == 200
    except Exception:
        return False


# ------------------------------------------------------------------ client
HttpGet = Callable[[str], str]


def _default_http_get(url: str) -> str:
    req = urllib.request.Request(url)
    with urllib.request.urlopen(req, timeout=300) as resp:
        body = resp.read().decode("utf-8", errors="replace")
        next_page = resp.headers.get("Next-Page")
    # Follow ThetaData pagination when present.
    while next_page and next_page.lower() != "null":
        req = urllib.request.Request(next_page)
        with urllib.request.urlopen(req, timeout=300) as resp:
            page = resp.read().decode("utf-8", errors="replace")
            next_page = resp.headers.get("Next-Page")
        # Drop the duplicated CSV header on continuation pages.
        body += "\n" + page.split("\n", 1)[1] if "\n" in page else page
    return body


@dataclass
class ThetaDataClient:
    base_url: str = field(default_factory=thetadata_base_url)
    http_get: HttpGet = _default_http_get

    retry_attempts: int = 4
    retry_wait_s: float = 15.0

    def _get(self, path: str, **params: Any) -> str:
        query = urllib.parse.urlencode({k: v for k, v in params.items() if v is not None})
        url = f"{self.base_url}{path}?{query}"
        last_exc: Exception | None = None
        for attempt in range(self.retry_attempts):
            try:
                return self.http_get(url)
            except Exception as exc:
                # An HTTP status error means the terminal IS answering (e.g. 403 =
                # subscription tier) — never retried. Connection-level failures
                # (timeouts, refused) are usually the terminal's periodic
                # upstream reconnect, so wait and retry before giving up.
                http_level = isinstance(exc, urllib.error.HTTPError) or "HTTP Error" in str(exc)
                if http_level:
                    raise ThetaDataError(f"request failed: {path}: {exc}") from exc
                last_exc = exc
                if attempt < self.retry_attempts - 1:
                    time.sleep(self.retry_wait_s)
        if not terminal_running(self.base_url):
            raise TerminalNotRunningError(self.base_url) from last_exc
        raise ThetaDataError(f"request failed: {path}: {last_exc}") from last_exc

    def option_quote_snapshot(
        self,
        symbol: str,
        session: date,
        snapshot_et: str = "15:30:00",
        max_dte: int = 70,
    ) -> pl.DataFrame:
        """All contracts' NBBO in the snapshot minute (empty frame on holidays)."""
        text = self._get(
            "/v3/option/history/quote",
            symbol=symbol,
            expiration="*",
            strike="*",
            right="both",
            date=session.strftime("%Y%m%d"),
            interval="1m",
            start_time=snapshot_et,
            end_time=_plus_minute(snapshot_et),
            max_dte=max_dte,
            format="csv",
        )
        return parse_option_quote_csv(text)

    def option_open_interest(self, symbol: str, session: date, max_dte: int = 70) -> pl.DataFrame:
        text = self._get(
            "/v3/option/history/open_interest",
            symbol=symbol,
            expiration="*",
            strike="*",
            right="both",
            date=session.strftime("%Y%m%d"),
            max_dte=max_dte,
            format="csv",
        )
        return parse_open_interest_csv(text)

    def stock_quote_mid(
        self, symbol: str, session: date, snapshot_et: str = "15:30:00"
    ) -> float | None:
        text = self._get(
            "/v3/stock/history/quote",
            symbol=symbol,
            date=session.strftime("%Y%m%d"),
            interval="1m",
            start_time=snapshot_et,
            end_time=_plus_minute(snapshot_et),
            format="csv",
        )
        df = _read_csv(text)
        if df.is_empty():
            return None
        cols = {c.lower(): c for c in df.columns}
        if "bid" not in cols or "ask" not in cols:
            return None
        row = df.row(0, named=True)
        bid, ask = float(row[cols["bid"]]), float(row[cols["ask"]])
        if ask <= 0 or bid < 0:
            return None
        return 0.5 * (bid + ask)


def _plus_minute(hhmmss: str) -> str:
    t = datetime.strptime(hhmmss, "%H:%M:%S")
    return (t + timedelta(minutes=1)).strftime("%H:%M:%S")


# ------------------------------------------------------- underlying fallback
MIN_PARITY_PAIRS = 5


def implied_underlying_from_quotes(
    quotes: pl.DataFrame, session: date, rate: float = 0.0
) -> float | None:
    """Spot implied by put-call parity at the snapshot: S = C - P + K*e^(-rT).

    Used when the subscription tier lacks intraday stock quotes. Uses the
    NEAREST expiration only (SPY expires most weekdays, so DTE is typically
    0-3), which keeps both the rate correction and any early-exercise or
    dividend distortion to cents. Returns the median across two-sided pairs;
    None when fewer than MIN_PARITY_PAIRS usable pairs exist. Sessions that
    fall inside an ex-dividend window can be biased by roughly the dividend
    PV — accepted and documented (the dividend calendar work will tighten
    this).
    """
    if quotes.is_empty():
        return None
    two_sided = quotes.filter((pl.col("bid") > 0) & (pl.col("ask") > pl.col("bid")))
    expiries = sorted(e for e in two_sided["expiration"].unique().to_list() if e >= session)
    if not expiries:
        return None
    import math

    nearest = expiries[0]
    t_years = (nearest - session).days / 365.0
    df_ = math.exp(-rate * t_years)
    sl = two_sided.filter(pl.col("expiration") == nearest)
    calls = sl.filter(pl.col("option_type") == "C").select(
        "strike", ((pl.col("bid") + pl.col("ask")) / 2).alias("c_mid")
    )
    puts = sl.filter(pl.col("option_type") == "P").select(
        "strike", ((pl.col("bid") + pl.col("ask")) / 2).alias("p_mid")
    )
    pairs = calls.join(puts, on="strike", how="inner").with_columns(
        (pl.col("c_mid") - pl.col("p_mid") + pl.col("strike") * df_).alias("implied")
    )
    if pairs.height < MIN_PARITY_PAIRS:
        return None
    # Restrict to strikes near the preliminary estimate (parity is tightest
    # near ATM; deep wings add noise from wide quotes).
    prelim = float(pairs["implied"].median())
    near = pairs.filter((pl.col("strike") / prelim - 1.0).abs() < 0.03)
    if near.height >= MIN_PARITY_PAIRS:
        pairs = near
    return float(pairs["implied"].median())


def default_rate_lookup(
    rates_path: str | Path = "data/normalized/rates/tbill_4w.parquet",
) -> Callable[[date], float]:
    """Rate series for the parity discounting; flat 0 if the file is absent
    (costs only cents at the nearest-expiry horizon)."""
    p = Path(rates_path)
    if not p.exists():
        return lambda _d: 0.0
    df = pl.read_parquet(p).sort("date")
    dates = df["date"].to_list()
    values = df["rate"].to_list()
    from bisect import bisect_right

    def lookup(d: date) -> float:
        i = bisect_right(dates, d)
        return float(values[i - 1]) if i > 0 else 0.0

    return lookup


# ------------------------------------------------------------------ parsing
def _read_csv(text: str) -> pl.DataFrame:
    text = text.strip()
    if not text or "\n" not in text and "," not in text:
        return pl.DataFrame()
    try:
        return pl.read_csv(text.encode(), infer_schema_length=10_000, ignore_errors=True)
    except Exception:
        return pl.DataFrame()


_RIGHT_MAP = {"call": "C", "put": "P", "c": "C", "p": "P"}


def _require(df: pl.DataFrame, names: list[str], what: str) -> dict[str, str]:
    cols = {c.lower(): c for c in df.columns}
    missing = [n for n in names if n not in cols]
    if missing:
        raise ThetaDataError(f"unexpected {what} response columns {df.columns}; missing {missing}")
    return cols


def parse_option_quote_csv(text: str) -> pl.DataFrame:
    """Raw bulk quote CSV -> (expiration, strike, right, bid, ask, sizes)."""
    df = _read_csv(text)
    if df.is_empty():
        return pl.DataFrame()
    cols = _require(df, ["expiration", "strike", "right", "bid", "ask"], "option quote")
    out = df.select(
        pl.col(cols["expiration"]).cast(pl.Utf8).alias("expiration_raw"),
        pl.col(cols["strike"]).cast(pl.Float64, strict=False).alias("strike"),
        pl.col(cols["right"]).cast(pl.Utf8).str.to_lowercase().alias("right_raw"),
        pl.col(cols["bid"]).cast(pl.Float64, strict=False).alias("bid"),
        pl.col(cols["ask"]).cast(pl.Float64, strict=False).alias("ask"),
        (
            pl.col(cols["bid_size"]).cast(pl.Int64, strict=False)
            if "bid_size" in cols
            else pl.lit(None, dtype=pl.Int64)
        ).alias("bid_size"),
        (
            pl.col(cols["ask_size"]).cast(pl.Int64, strict=False)
            if "ask_size" in cols
            else pl.lit(None, dtype=pl.Int64)
        ).alias("ask_size"),
    )
    return out.with_columns(
        pl.coalesce(
            pl.col("expiration_raw").str.to_date(format="%Y-%m-%d", strict=False),
            pl.col("expiration_raw").str.to_date(format="%Y%m%d", strict=False),
        ).alias("expiration"),
        pl.col("right_raw").replace_strict(_RIGHT_MAP, default=None).alias("option_type"),
    ).drop("expiration_raw", "right_raw")


def parse_open_interest_csv(text: str) -> pl.DataFrame:
    df = _read_csv(text)
    if df.is_empty():
        return pl.DataFrame()
    cols = _require(df, ["expiration", "strike", "right", "open_interest"], "open interest")
    return (
        df.select(
            pl.col(cols["expiration"]).cast(pl.Utf8).alias("expiration_raw"),
            pl.col(cols["strike"]).cast(pl.Float64, strict=False).alias("strike"),
            pl.col(cols["right"]).cast(pl.Utf8).str.to_lowercase().alias("right_raw"),
            pl.col(cols["open_interest"]).cast(pl.Int64, strict=False).alias("open_interest"),
        )
        .with_columns(
            pl.coalesce(
                pl.col("expiration_raw").str.to_date(format="%Y-%m-%d", strict=False),
                pl.col("expiration_raw").str.to_date(format="%Y%m%d", strict=False),
            ).alias("expiration"),
            pl.col("right_raw").replace_strict(_RIGHT_MAP, default=None).alias("option_type"),
        )
        .drop("expiration_raw", "right_raw")
    )


def normalize_snapshot(
    quotes: pl.DataFrame,
    open_interest: pl.DataFrame,
    underlying_price: float,
    session: date,
    snapshot_et: str,
    root: str,
) -> pl.DataFrame:
    """Join quotes + OI into one canonical-schema snapshot frame."""
    if quotes.is_empty():
        return pl.DataFrame(schema=dict(CHAIN_SCHEMA))
    hh, mm, ss = (int(x) for x in snapshot_et.split(":"))
    ts = datetime(session.year, session.month, session.day, hh, mm, ss, tzinfo=ET).astimezone(UTC)
    out = quotes.drop_nulls(subset=["expiration", "option_type", "strike"]).with_columns(
        pl.lit(ts).dt.cast_time_unit("us").alias("ts"),
        pl.lit(root).alias("root"),
        pl.lit(None, dtype=pl.Int64).alias("volume"),  # quotes carry no volume
        pl.lit(float(underlying_price)).alias("underlying_price"),
    )
    if not open_interest.is_empty():
        out = out.join(
            open_interest.select(["expiration", "strike", "option_type", "open_interest"]),
            on=["expiration", "strike", "option_type"],
            how="left",
        )
    else:
        out = out.with_columns(pl.lit(None, dtype=pl.Int64).alias("open_interest"))
    out = (
        out.filter((pl.col("bid") >= 0) & (pl.col("ask") >= 0))
        .unique(subset=["expiration", "strike", "option_type"], keep="first")
        .select(list(CHAIN_SCHEMA))
        .sort(["expiration", "strike", "option_type"])
    )
    validate_chain_frame(out)
    return out


# ------------------------------------------------------------------ bulk pull
def _month_key(d: date) -> str:
    return f"{d.year:04d}-{d.month:02d}"


def _month_file_complete(path: Path, n_weekdays: int) -> bool:
    """A month file interrupted mid-pull (network errors) must not be treated
    as done: require sessions >= weekdays minus a holiday margin."""
    try:
        have = pl.read_parquet(path, columns=["ts"])["ts"].dt.date().n_unique()
    except Exception:
        return False  # unreadable file: re-pull it
    return have >= n_weekdays - 4


def _weekdays(start: date, end: date) -> list[date]:
    return [
        start + timedelta(days=i)
        for i in range((end - start).days + 1)
        if (start + timedelta(days=i)).weekday() < 5
    ]


def pull_chain_history(
    client: ThetaDataClient,
    symbol: str,
    start: date,
    end: date,
    out_dir: str | Path,
    snapshot_et: str = "15:30:00",
    max_dte: int = 70,
    with_open_interest: bool = True,
    refresh: bool = False,
    progress: Callable[[str], None] = print,
    rate_lookup: Callable[[date], float] | None = None,
) -> dict[str, Any]:
    """Pull one snapshot per session, written as chain_YYYY-MM.parquet files.

    Underlying price per session: intraday stock quote when the subscription
    allows it, else put-call-parity implied from the option quotes themselves
    (method counts recorded in the manifest). Resumable: existing month files
    are skipped unless refresh=True (the current partial month is always
    re-pulled).
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rate_lookup = rate_lookup or default_rate_lookup()
    sessions_by_month: dict[str, list[date]] = {}
    for d in _weekdays(start, end):
        sessions_by_month.setdefault(_month_key(d), []).append(d)

    written: dict[str, Any] = {}
    skipped_months: list[str] = []
    empty_sessions: list[str] = []
    errors: dict[str, str] = {}
    underlying_methods = {"stock_quote": 0, "parity_implied": 0}
    stock_quote_available = True  # flips off after the first 403 to save requests

    for month, sessions in sorted(sessions_by_month.items()):
        path = out / f"chain_{month}.parquet"
        is_current_month = month == _month_key(end)
        if (
            path.exists()
            and not refresh
            and not is_current_month
            and _month_file_complete(path, len(sessions))
        ):
            skipped_months.append(month)
            continue
        frames: list[pl.DataFrame] = []
        for session in sessions:
            try:
                quotes = client.option_quote_snapshot(symbol, session, snapshot_et, max_dte)
                if quotes.is_empty():
                    empty_sessions.append(str(session))  # holiday / no data
                    continue
                spot = None
                if stock_quote_available:
                    try:
                        spot = client.stock_quote_mid(symbol, session, snapshot_et)
                    except ThetaDataError as exc:
                        if "403" in str(exc):
                            stock_quote_available = False  # tier lacks intraday stock
                        else:
                            raise
                if spot is not None:
                    underlying_methods["stock_quote"] += 1
                else:
                    spot = implied_underlying_from_quotes(quotes, session, rate_lookup(session))
                    if spot is None:
                        errors[str(session)] = (
                            "no underlying price (stock quote unavailable and parity "
                            "fallback lacked usable pairs); session skipped"
                        )
                        continue
                    underlying_methods["parity_implied"] += 1
                oi = (
                    client.option_open_interest(symbol, session, max_dte)
                    if with_open_interest
                    else pl.DataFrame()
                )
                frames.append(normalize_snapshot(quotes, oi, spot, session, snapshot_et, symbol))
            except TerminalNotRunningError:
                raise
            except ThetaDataError as exc:
                errors[str(session)] = str(exc)
        if frames:
            month_df = pl.concat(frames)
            month_df.write_parquet(path)
            written[month] = {"sessions": len(frames), "rows": month_df.height}
            progress(f"{month}: {len(frames)} sessions, {month_df.height} rows")
        elif sessions:
            progress(f"{month}: no data")

    manifest = {
        "symbol": symbol,
        "snapshot_et": snapshot_et,
        "max_dte": max_dte,
        "start": str(start),
        "end": str(end),
        "months_written": written,
        "months_skipped_existing": skipped_months,
        "empty_sessions": empty_sessions,
        "session_errors": errors,
        "underlying_price_methods": underlying_methods,
        "source": "thetadata_v3_terminal",
        "license_note": "personal research use; never committed or redistributed",
    }
    (out / "pull_manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


def pull_underlying_eod(
    client: ThetaDataClient, symbol: str, start: date, end: date, out_path: str | Path
) -> pl.DataFrame:
    """UNADJUSTED daily closes from the EOD endpoint (works on the free stock
    tier) — the settlement-price series for the backtester's UnderlyingProvider.
    Pulled in yearly chunks (the endpoint limits multi-day ranges)."""
    frames: list[pl.DataFrame] = []
    chunk_start = start
    while chunk_start <= end:
        chunk_end = min(date(chunk_start.year, 12, 31), end)
        text = client._get(
            "/v3/stock/history/eod",
            symbol=symbol,
            start_date=chunk_start.strftime("%Y%m%d"),
            end_date=chunk_end.strftime("%Y%m%d"),
            format="csv",
        )
        df = _read_csv(text)
        if not df.is_empty():
            cols = {c.lower(): c for c in df.columns}
            if "created" not in cols or "close" not in cols:
                raise ThetaDataError(f"unexpected EOD response columns: {df.columns}")
            frames.append(
                df.select(
                    pl.col(cols["created"])
                    .cast(pl.Utf8)
                    .str.slice(0, 10)
                    .str.to_date()
                    .alias("date"),
                    pl.col(cols["close"]).cast(pl.Float64, strict=False).alias("close"),
                ).drop_nulls()
            )
        chunk_start = date(chunk_start.year + 1, 1, 1)
    if not frames:
        raise ThetaDataError("EOD endpoint returned no data")
    out = pl.concat(frames).unique(subset="date", keep="last").sort("date")
    p = Path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    out.write_parquet(p)
    return out
