"""Pull quarterly income statements for current S&P-500 members (FMP /stable/).

Point-in-time discipline: keep `filingDate`; downstream features only use data
whose filingDate is strictly before the feature date. Survivorship caveat:
membership is TODAY's list (FMP's historical-constituent endpoint is not on this
plan); mitigation = use only cross-SECTOR relative aggregates, and treat pre-2018
results as directional, not gate-passing, evidence.

Output: data/normalized/exog/sp500_income_q.parquet
"""

from __future__ import annotations

import sys
import time

import pandas as pd

sys.path.insert(0, "src")
from level2_research.fmp import _get  # noqa: E402

KEEP = ["symbol", "date", "filingDate", "revenue", "grossProfit", "operatingIncome",
        "netIncome", "operatingExpenses", "period", "fiscalYear"]


def main():
    members = pd.read_parquet("data/normalized/exog/sp500_constituents_current.parquet")
    frames, fails = [], []
    for i, sym in enumerate(members.symbol):
        try:
            out = _get("/stable/income-statement",
                       {"symbol": sym, "period": "quarter", "limit": 110})
            if isinstance(out, list) and out:
                df = pd.DataFrame(out)
                frames.append(df[[c for c in KEEP if c in df.columns]])
        except Exception as e:
            fails.append((sym, str(e)[:60]))
        if i % 50 == 0:
            print(f"{i}/{len(members)} pulled, {len(fails)} fails", flush=True)
        time.sleep(0.12)
    all_df = pd.concat(frames, ignore_index=True)
    all_df["date"] = pd.to_datetime(all_df["date"])
    all_df["filingDate"] = pd.to_datetime(all_df["filingDate"])
    all_df = all_df.merge(members[["symbol", "sector"]], on="symbol", how="left")
    all_df.to_parquet("data/normalized/exog/sp500_income_q.parquet", index=False)
    print(f"DONE: {len(all_df)} statement-quarters, {all_df.symbol.nunique()} symbols, "
          f"{len(fails)} fails: {fails[:5]}")


if __name__ == "__main__":
    main()
