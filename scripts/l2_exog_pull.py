"""Cycle-2 exogenous data pull (FMP /stable/) -> data/normalized/exog/.

Series and their look-ahead handling (documented per research discipline):
- treasury curve (daily): observable same day -> lag 1 session at feature time.
- economic indicators (monthly obs dates, NOT release dates): lagged 45 calendar
  days at feature time to clear any release-lag ambiguity.
- sector P/E snapshots (daily by date param): pulled weekly (Fridays) 2017->now;
  observable same day -> lag 1 session.
- earnings calendar (per-symbol actual vs estimate, dated by report date):
  aggregates computed over trailing windows ending 2 days before feature date
  (reports after the close / timing ambiguity).
- COT ES report (weekly, Tuesday data released Friday): lagged 5 calendar days.
"""

from __future__ import annotations

import json
import time
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

import sys
sys.path.insert(0, "src")
from level2_research.fmp import _get  # noqa: E402

OUT = Path("data/normalized/exog")
OUT.mkdir(parents=True, exist_ok=True)
manifest = {}


def save(name: str, df: pd.DataFrame):
    df.to_parquet(OUT / f"{name}.parquet", index=False)
    manifest[name] = dict(rows=len(df), cols=list(df.columns)[:12],
                          start=str(df["date"].min())[:10], end=str(df["date"].max())[:10])
    print(f"{name}: {len(df)} rows {manifest[name]['start']} -> {manifest[name]['end']}", flush=True)


def pull_treasury():
    frames = []
    lo = pd.Timestamp("2000-01-01")
    while lo < pd.Timestamp("2026-07-29"):
        hi = min(lo + pd.DateOffset(months=3), pd.Timestamp("2026-07-29"))
        out = _get("/stable/treasury-rates", {"from": str(lo.date()), "to": str(hi.date())})
        if isinstance(out, list) and out:
            frames.append(pd.DataFrame(out))
        lo = hi + pd.Timedelta(days=1)
        time.sleep(0.15)
    df = pd.concat(frames, ignore_index=True).drop_duplicates("date")
    df["date"] = pd.to_datetime(df["date"])
    save("treasury_curve", df.sort_values("date"))


def pull_econ():
    for name in ["CPI", "unemploymentRate", "consumerSentiment", "retailSales",
                 "industrialProductionTotalIndex", "federalFunds"]:
        try:
            out = _get("/stable/economic-indicators",
                       {"name": name, "from": "1999-01-01", "to": "2026-07-29"})
            df = pd.DataFrame(out)
            if len(df):
                df["date"] = pd.to_datetime(df["date"])
                save(f"econ_{name}", df.sort_values("date"))
        except Exception as e:
            print(f"econ_{name} FAIL {e}", flush=True)
        time.sleep(0.2)


def pull_sector_pe():
    rows = []
    d = date(2017, 1, 6)
    while d <= date(2026, 7, 24):
        try:
            out = _get("/stable/sector-pe-snapshot", {"date": d.isoformat(), "exchange": "NYSE"})
            if isinstance(out, list):
                rows.extend(out)
        except Exception:
            pass
        d += timedelta(days=7)
        time.sleep(0.12)
    df = pd.DataFrame(rows)
    if len(df):
        df["date"] = pd.to_datetime(df["date"])
        save("sector_pe_weekly", df)


def pull_earnings_calendar():
    frames = []
    lo = pd.Timestamp("2016-06-01")
    while lo < pd.Timestamp("2026-07-29"):
        hi = min(lo + pd.DateOffset(months=1) - pd.Timedelta(days=1), pd.Timestamp("2026-07-29"))
        try:
            out = _get("/stable/earnings-calendar", {"from": str(lo.date()), "to": str(hi.date())})
            if isinstance(out, list) and out:
                frames.append(pd.DataFrame(out))
        except Exception as e:
            print(f"earnings {lo.date()} FAIL {e}", flush=True)
        lo = hi + pd.Timedelta(days=1)
        time.sleep(0.15)
    df = pd.concat(frames, ignore_index=True)
    df["date"] = pd.to_datetime(df["date"])
    save("earnings_calendar", df)


def pull_cot():
    frames = []
    lo = pd.Timestamp("2010-01-01")
    while lo < pd.Timestamp("2026-07-29"):
        hi = min(lo + pd.DateOffset(months=6), pd.Timestamp("2026-07-29"))
        try:
            out = _get("/stable/commitment-of-traders-report",
                       {"symbol": "ES", "from": str(lo.date()), "to": str(hi.date())})
            if isinstance(out, list) and out:
                frames.append(pd.DataFrame(out))
        except Exception as e:
            print(f"cot {lo.date()} FAIL {e}", flush=True)
        lo = hi + pd.Timedelta(days=1)
        time.sleep(0.2)
    if frames:
        df = pd.concat(frames, ignore_index=True)
        df["date"] = pd.to_datetime(df["date"])
        save("cot_es", df)


def pull_sp500_members():
    out = _get("/stable/sp500-constituent", {})
    df = pd.DataFrame(out)
    df["date"] = pd.Timestamp.today().normalize()
    save("sp500_constituents_current", df)


if __name__ == "__main__":
    pull_treasury()
    pull_econ()
    pull_sp500_members()
    pull_cot()
    pull_sector_pe()
    pull_earnings_calendar()
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print("DONE")
