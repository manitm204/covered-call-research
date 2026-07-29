"""FMP daily price history pull (adjusted + unadjusted + dividends).

Used once to build the 2000-2017 signal pre-validation dataset. Key comes from the
environment / .env via xsp_research.env; never logged.
"""

from __future__ import annotations

import json
import os
import time
import urllib.parse
import urllib.request
from pathlib import Path

import pandas as pd

from xsp_research.env import load_dotenv

BASE = "https://financialmodelingprep.com"


def _get(path: str, params: dict) -> dict | list:
    load_dotenv()
    key = os.environ.get("FMP_API_KEY", "").strip()
    if not key:
        raise RuntimeError("FMP_API_KEY not set")
    params = dict(params, apikey=key)
    url = f"{BASE}{path}?{urllib.parse.urlencode(params)}"
    for attempt in range(4):
        try:
            with urllib.request.urlopen(url, timeout=60) as r:
                return json.loads(r.read().decode())
        except Exception:
            if attempt == 3:
                raise
            time.sleep(2 * (attempt + 1))
    raise RuntimeError("unreachable")


def _pull_chunked(path: str, symbol: str, start: str, end: str) -> pd.DataFrame:
    """Stable endpoints cap the span per request; pull in 4-year chunks."""
    frames = []
    lo = pd.Timestamp(start)
    hard_end = pd.Timestamp(end)
    while lo <= hard_end:
        hi = min(lo + pd.DateOffset(years=4) - pd.Timedelta(days=1), hard_end)
        out = _get(path, {"symbol": symbol, "from": str(lo.date()), "to": str(hi.date())})
        if isinstance(out, list) and out:
            frames.append(pd.DataFrame(out))
        lo = hi + pd.Timedelta(days=1)
    if not frames:
        raise RuntimeError(f"no history returned for {symbol}")
    df = pd.concat(frames, ignore_index=True)
    df["date"] = pd.to_datetime(df["date"])
    return df.drop_duplicates("date").sort_values("date").reset_index(drop=True)


def pull_daily_history(symbol: str, start: str, end: str) -> pd.DataFrame:
    """Daily bars: unadjusted OHLC + dividend-adjusted close, ascending by date."""
    raw = _pull_chunked("/stable/historical-price-eod/full", symbol, start, end)
    adj = _pull_chunked("/stable/historical-price-eod/dividend-adjusted", symbol, start, end)
    adj_col = "adjClose" if "adjClose" in adj.columns else "close"
    adj = adj[["date", adj_col]].rename(columns={adj_col: "adjClose"})
    df = raw.merge(adj, on="date", how="left")
    cols = ["date", "open", "high", "low", "close", "adjClose", "volume"]
    return df[[c for c in cols if c in df.columns]]


def pull_dividends(symbol: str) -> pd.DataFrame:
    out = _get("/stable/dividends", {"symbol": symbol, "limit": 1000})
    df = pd.DataFrame(out if isinstance(out, list) else [])
    if df.empty:
        return pd.DataFrame(columns=["ex_date", "amount"])
    df["ex_date"] = pd.to_datetime(df["date"])
    df = df.rename(columns={"dividend": "amount"})[["ex_date", "amount"]]
    return df.sort_values("ex_date").reset_index(drop=True)


def main(out_dir: str = "data/normalized/prices_long") -> None:
    outp = Path(out_dir)
    outp.mkdir(parents=True, exist_ok=True)
    manifest: dict = {"source": "FMP historical-price-full", "pulled": []}
    for sym in ["SPY", "QQQ", "IWM"]:
        px = pull_daily_history(sym, "1999-01-01", "2026-07-29")
        dv = pull_dividends(sym)
        px.to_parquet(outp / f"{sym}.parquet", index=False)
        dv.to_parquet(outp / f"{sym}_dividends.parquet", index=False)
        manifest["pulled"].append(
            {
                "symbol": sym,
                "rows": len(px),
                "start": str(px.date.min().date()),
                "end": str(px.date.max().date()),
                "dividend_rows": len(dv),
            }
        )
        print(sym, len(px), px.date.min().date(), "->", px.date.max().date(), f"({len(dv)} divs)")
    (outp / "manifest.json").write_text(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
