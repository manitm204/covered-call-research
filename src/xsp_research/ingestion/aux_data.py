"""Free auxiliary-data ingestion: FRED, Yahoo, and Cboe index histories.

Downloads the research bundle's non-options legs — ETFs, VIX family, rates,
and the SPX-derived XSP underlying level — normalizes each to the canonical
(date, close) schema, and writes per-series Parquet + a manifest with source
URL, coverage, and file hash.

Sources and licensing:
- FRED (fredgraph.csv, no API key): public series; cite FRED.
- Yahoo Finance chart API (JSON): ETF/index daily closes for personal research
  use. ETF series use ADJUSTED closes (dividends folded in) so return-based
  features measure total return; the SPX index level is a price index.
  (Stooq was the original plan but now sits behind a JavaScript challenge.)
- Cboe index histories (public CSVs): VIX9D/VVIX not available on FRED.

Unit conventions:
- ETF/index series: closing price/level as published.
- VIX family: index points (feature code divides by 100 where vol is needed).
- RATE_3M: FRED DTB4WK is a percent discount yield -> stored as a decimal
  (0.052 = 5.2%). It is a PROXY for the risk-free rate, not a term-matched
  cont-comp rate; documented in PLAN.md assumption list.
- UNDERLYING: XSP is defined by Cboe as 1/10th of SPX, so the underlying level
  is ingested as SPX/10. This is the index definition, not an approximation —
  but it is an INDEX level, not a tradable price.

Parsers are separated from fetchers so tests never touch the network.
"""

from __future__ import annotations

import hashlib
import json
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import polars as pl

TIMEOUT_S = 30
# Browser-like UA: Yahoo's chart API rejects default urllib user agents.
_UA = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) xsp-research/0.1 personal research"}


@dataclass(frozen=True, slots=True)
class AuxSeriesSpec:
    symbol: str  # bundle symbol (MarketDataBundle key)
    source: str  # "fred" | "yahoo" | "cboe_index"
    source_id: str  # series id / ticker / cboe index name
    description: str
    scale: float = 1.0  # applied to close (e.g. 0.1 for SPX->XSP, 0.01 for pct->decimal)


SECTOR_SYMBOLS: tuple[str, ...] = (
    "XLK",
    "XLF",
    "XLE",
    "XLV",
    "XLI",
    "XLP",
    "XLY",
    "XLU",
    "XLB",
)

DEFAULT_UNIVERSE: tuple[AuxSeriesSpec, ...] = (
    # Underlying: XSP level = SPX / 10 by index definition.
    AuxSeriesSpec("UNDERLYING", "yahoo", "^GSPC", "XSP level (SPX/10 by definition)", 0.1),
    AuxSeriesSpec("SPX", "yahoo", "^GSPC", "S&P 500 index close"),
    AuxSeriesSpec("SPY", "yahoo", "SPY", "SPDR S&P 500 ETF (adjusted close)"),
    AuxSeriesSpec("RSP", "yahoo", "RSP", "Invesco equal-weight S&P 500 ETF (adjusted close)"),
    AuxSeriesSpec("QQQ", "yahoo", "QQQ", "Invesco Nasdaq-100 ETF (adjusted close)"),
    AuxSeriesSpec("IWM", "yahoo", "IWM", "iShares Russell 2000 ETF (adjusted close)"),
    AuxSeriesSpec("HYG", "yahoo", "HYG", "iShares high-yield bond ETF (adjusted close)"),
    AuxSeriesSpec("LQD", "yahoo", "LQD", "iShares inv-grade bond ETF (adjusted close)"),
    AuxSeriesSpec("GLD", "yahoo", "GLD", "SPDR gold ETF (adjusted close)"),
    AuxSeriesSpec("USO", "yahoo", "USO", "US oil fund (adjusted close)"),
    AuxSeriesSpec("UUP", "yahoo", "UUP", "Invesco dollar index fund (adjusted close)"),
    AuxSeriesSpec("DBC", "yahoo", "DBC", "Invesco broad commodity fund (adjusted close)"),
    *(
        AuxSeriesSpec(s, "yahoo", s, f"SPDR sector ETF {s} (adjusted close)")
        for s in SECTOR_SYMBOLS
    ),
    AuxSeriesSpec("VIX", "cboe_index", "VIX", "CBOE VIX close (points)"),
    AuxSeriesSpec("VIX9D", "cboe_index", "VIX9D", "CBOE 9-day VIX close (points)"),
    AuxSeriesSpec("VVIX", "cboe_index", "VVIX", "CBOE VVIX close (points)"),
    AuxSeriesSpec("RATE_3M", "fred", "DTB4WK", "4-week T-bill discount yield (decimal)", 0.01),
    AuxSeriesSpec("SOFR", "fred", "SOFR", "SOFR overnight rate (decimal)", 0.01),
)


# ------------------------------------------------------------------ parsers
def parse_fred_csv(text: str) -> pl.DataFrame:
    """FRED fredgraph.csv: header (DATE|observation_date, <SERIES>); '.' = missing."""
    df = pl.read_csv(text.encode(), infer_schema_length=0)
    if df.width < 2:
        raise ValueError("unexpected FRED CSV shape")
    date_col, value_col = df.columns[0], df.columns[1]
    return (
        df.select(
            pl.col(date_col).str.to_date().alias("date"),
            pl.col(value_col).replace(".", None).cast(pl.Float64, strict=False).alias("close"),
        )
        .drop_nulls()
        .sort("date")
    )


def parse_yahoo_chart_json(text: str) -> pl.DataFrame:
    """Yahoo v8 chart JSON: adjusted close preferred, raw close as fallback."""
    payload = json.loads(text)
    result = payload.get("chart", {}).get("result")
    if not result:
        raise ValueError(f"Yahoo chart error: {payload.get('chart', {}).get('error')}")
    r = result[0]
    timestamps = r.get("timestamp")
    if not timestamps:
        raise ValueError("Yahoo chart response has no timestamps")
    closes = r["indicators"]["quote"][0]["close"]
    adj = r["indicators"].get("adjclose")
    if adj and adj[0].get("adjclose"):
        closes = adj[0]["adjclose"]
    rows = [
        (datetime.fromtimestamp(ts, tz=UTC).date(), float(c))
        for ts, c in zip(timestamps, closes, strict=True)
        if c is not None
    ]
    return (
        pl.DataFrame(rows, schema={"date": pl.Date, "close": pl.Float64}, orient="row")
        .unique(subset="date", keep="last")
        .sort("date")
    )


def _parse_flexible_dates(col: pl.Expr) -> pl.Expr:
    """Cboe files mix ISO (2023-01-03) and US (01/03/2023) date formats."""
    return pl.coalesce(
        col.str.to_date(format="%Y-%m-%d", strict=False),
        col.str.to_date(format="%m/%d/%Y", strict=False),
    )


def parse_cboe_index_csv(text: str) -> pl.DataFrame:
    """Cboe index history CSV. Two known layouts:
    DATE,OPEN,HIGH,LOW,CLOSE  and  DATE,<INDEXNAME> (single value column)."""
    df = pl.read_csv(text.encode(), infer_schema_length=0)
    cols = {c.strip().lower(): c for c in df.columns}
    if "date" not in cols or df.width < 2:
        raise ValueError(f"unexpected Cboe CSV columns: {df.columns}")
    value_col = cols.get("close", df.columns[-1])
    return (
        df.select(
            _parse_flexible_dates(pl.col(cols["date"])).alias("date"),
            pl.col(value_col).cast(pl.Float64, strict=False).alias("close"),
        )
        .drop_nulls()
        .sort("date")
    )


# ------------------------------------------------------------------ fetchers
def _http_get(url: str, retries: int = 3) -> str:
    """GET with small exponential backoff (public endpoints time out transiently)."""
    import time

    last: Exception | None = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers=_UA)
            with urllib.request.urlopen(req, timeout=TIMEOUT_S) as resp:
                return resp.read().decode("utf-8", errors="replace")
        except Exception as exc:  # noqa: BLE001 - retried, then re-raised
            last = exc
            if attempt < retries - 1:
                time.sleep(2.0 * (attempt + 1))
    raise last  # type: ignore[misc]


# Fixed epoch window for Yahoo: `range=max` silently coerces long ranges to
# MONTHLY bars; explicit period1/period2 keeps daily granularity.
_YAHOO_PERIOD1 = 820454400  # 1996-01-01 UTC
_YAHOO_PERIOD2 = 4102444800  # 2100-01-01 UTC


def source_url(spec: AuxSeriesSpec) -> str:
    if spec.source == "fred":
        return f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={spec.source_id}"
    if spec.source == "yahoo":
        return (
            "https://query1.finance.yahoo.com/v8/finance/chart/"
            f"{urllib.parse.quote(spec.source_id)}"
            f"?period1={_YAHOO_PERIOD1}&period2={_YAHOO_PERIOD2}&interval=1d"
        )
    if spec.source == "cboe_index":
        return (
            f"https://cdn.cboe.com/api/global/us_indices/daily_prices/{spec.source_id}_History.csv"
        )
    raise ValueError(f"unknown source {spec.source}")


def fetch_series(spec: AuxSeriesSpec) -> pl.DataFrame:
    text = _http_get(source_url(spec))
    if spec.source == "fred":
        df = parse_fred_csv(text)
    elif spec.source == "yahoo":
        df = parse_yahoo_chart_json(text)
    else:
        df = parse_cboe_index_csv(text)
    if df.is_empty():
        raise ValueError(f"{spec.symbol}: source returned no rows")
    if spec.scale != 1.0:
        df = df.with_columns((pl.col("close") * spec.scale).alias("close"))
    return df


# ------------------------------------------------------------------ ingestion
def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def ingest_aux_bundle(
    out_dir: str | Path,
    specs: tuple[AuxSeriesSpec, ...] = DEFAULT_UNIVERSE,
    start: date | None = None,
    manifest_dir: str | Path = "data/manifests",
    fetcher=fetch_series,
    only: tuple[str, ...] | None = None,
) -> dict[str, Any]:
    """Download every series, write <SYMBOL>.parquet + one bundle manifest.

    Failures are recorded per series and never fabricate data; the returned
    summary lists exactly what was and wasn't obtained. `fetcher` is
    injectable for tests. `only` restricts the run to named symbols (for
    retrying flaky public sources); results merge into the existing manifest
    so previously-fetched series keep their provenance.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    manifest_path = Path(manifest_dir) / "aux_bundle.manifest.json"
    series_meta: dict[str, Any] = {}
    errors: dict[str, str] = {}
    if only is not None and manifest_path.exists():
        previous = json.loads(manifest_path.read_text())
        series_meta = dict(previous.get("series", {}))
        errors = dict(previous.get("errors", {}))
        specs = tuple(s for s in specs if s.symbol in only)
    for spec in specs:
        try:
            df = fetcher(spec)
            if start is not None:
                df = df.filter(pl.col("date") >= start)
            if df.is_empty():
                raise ValueError("no rows after start-date filter")
            path = out / f"{spec.symbol}.parquet"
            df.write_parquet(path)
            series_meta[spec.symbol] = {
                "source": spec.source,
                "source_id": spec.source_id,
                "source_url": source_url(spec),
                "description": spec.description,
                "scale": spec.scale,
                "rows": df.height,
                "date_min": str(df["date"].min()),
                "date_max": str(df["date"].max()),
                "sha256": _sha256(path),
            }
            errors.pop(spec.symbol, None)  # a retry that succeeds clears its error
        except Exception as exc:  # record and continue: partial bundles are usable
            errors[spec.symbol] = f"{type(exc).__name__}: {exc}"

    # Rates file for the backtester's SeriesRatesProvider (date, rate).
    if "RATE_3M" in series_meta:
        rates = pl.read_parquet(out / "RATE_3M.parquet").rename({"close": "rate"})
        rates_dir = out.parent / "rates"
        rates_dir.mkdir(parents=True, exist_ok=True)
        rates.write_parquet(rates_dir / "tbill_4w.parquet")
        series_meta["RATE_3M"]["also_written"] = str(rates_dir / "tbill_4w.parquet")

    manifest = {
        "created_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "out_dir": str(out),
        "start_filter": str(start) if start else None,
        "sector_symbols": [s for s in SECTOR_SYMBOLS if s in series_meta],
        "series": series_meta,
        "errors": errors,
        "license_note": (
            "FRED public series; Yahoo Finance chart API (personal research use); "
            "Cboe public index histories. Do not redistribute; not committed to git."
        ),
    }
    mdir = Path(manifest_dir)
    mdir.mkdir(parents=True, exist_ok=True)
    (mdir / "aux_bundle.manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


def load_aux_bundle(bundle_dir: str | Path):
    """Load an ingested aux directory into a MarketDataBundle (sectors auto-detected)."""
    from xsp_research.features.registry import MarketDataBundle

    d = Path(bundle_dir)
    series = {p.stem: pl.read_parquet(p).select(["date", "close"]) for p in d.glob("*.parquet")}
    if "UNDERLYING" not in series:
        raise ValueError(f"{d} has no UNDERLYING.parquet; run `xsp ingest-aux` first")
    sectors = tuple(s for s in SECTOR_SYMBOLS if s in series)
    return MarketDataBundle(series=series, sector_symbols=sectors)
