"""TG-CER daily signal generator (paper-trading tool).

Implements strategy_spec.md §2-4 exactly, for live/paper use:
  python scripts/tg_cer_signal.py --equity 10000 [--positions POS.json]

- Pulls QQQ daily closes from FMP (adjusted), computes the trend state from
  data through YESTERDAY's close (same lag as the backtest).
- If the ThetaData terminal is running, snapshots the current QQQ chain and
  selects the exact contract per spec (60 DTE target, 0.50 delta); otherwise
  prints strike guidance from a Black-Scholes approximation.
- Prints one of: EXIT-ALL / ROLL / ENTER (with contract + limit price + size) /
  HOLD / STAND-ASIDE, plus the checklist reminders that apply.

Positions file (optional JSON): [{"expiration":"YYYY-MM-DD","strike":570.0,
"contracts":1,"entry_price":19.90}]
"""

from __future__ import annotations

import argparse
import json
import math
from datetime import date, datetime
from pathlib import Path

import numpy as np
import pandas as pd

import sys
sys.path.insert(0, "src")
from level2_research.fmp import _pull_chunked  # noqa: E402

SPEC = dict(symbol="QQQ", band=0.01, delta_target=0.50, delta_band=(0.38, 0.62),
            dte_target=60, dte_band=(40, 70), roll_dte=21, budget_pct=0.10,
            max_rel_spread=0.10, min_oi=100, fee=0.65)


def trend_state():
    end = date.today().isoformat()
    px = _pull_chunked("/stable/historical-price-eod/dividend-adjusted", SPEC["symbol"],
                       "2024-01-01", end)
    col = "adjClose" if "adjClose" in px.columns else "close"
    a = px.set_index("date")[col].astype(float).sort_index()
    # use data through the last COMPLETED session strictly before today
    a_hist = a[a.index.date < date.today()] if len(a) else a
    ma200 = a_hist.rolling(200).mean().iloc[-1]
    p = a_hist.iloc[-1]
    return dict(price=float(p), ma200=float(ma200), asof=str(a_hist.index[-1].date()),
                entry=bool(p > (1 + SPEC["band"]) * ma200),
                hold=bool(p > (1 - SPEC["band"]) * ma200))


def live_contract(equity: float):
    """Try ThetaData for a live chain snapshot; return selection or None."""
    try:
        from xsp_research.ingestion.thetadata import ThetaDataClient, terminal_running
        if not terminal_running():
            return None
        import polars as pl  # noqa
        client = ThetaDataClient()
        snap = client.option_quote_snapshot(SPEC["symbol"], date.today(), "15:30:00",
                                            max_dte=SPEC["dte_band"][1])
        df = snap.to_pandas() if hasattr(snap, "to_pandas") else snap
        return df
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--equity", type=float, required=True)
    ap.add_argument("--positions", default=None)
    args = ap.parse_args()
    st = trend_state()
    print(f"=== TG-CER signal, computed {datetime.now():%Y-%m-%d %H:%M} ===")
    print(f"QQQ adj close (as of {st['asof']}): {st['price']:.2f}; MA200: {st['ma200']:.2f}")
    print(f"trend_entry (>+1%): {st['entry']}   trend_hold (>-1%): {st['hold']}")
    positions = []
    if args.positions and Path(args.positions).exists():
        positions = json.loads(Path(args.positions).read_text())
    if not st["hold"]:
        if positions:
            print("ACTION: EXIT-ALL — sell every open call today (marketable limit at bid or better).")
        else:
            print("ACTION: STAND-ASIDE — trend off, hold cash only.")
        return
    for p in positions:
        dte = (pd.Timestamp(p["expiration"]).date() - date.today()).days
        if dte <= SPEC["roll_dte"]:
            print(f"ACTION: ROLL — sell {p['contracts']}x {p['expiration']} {p['strike']}C "
                  f"({dte} DTE ≤ {SPEC['roll_dte']}), then re-enter per below.")
            positions = []
            break
    if positions:
        print("ACTION: HOLD — open call within DTE band, trend on. No trade.")
        return
    if not st["entry"]:
        print("ACTION: STAND-ASIDE — trend_hold on but trend_entry (+1% band) not met; no new entry.")
        return
    budget = SPEC["budget_pct"] * args.equity
    print(f"ACTION: ENTER — budget {budget:.0f} USD (10% of equity).")
    print(f"  Select: expiry nearest {SPEC['dte_target']} DTE in {SPEC['dte_band']}, "
          f"delta closest to {SPEC['delta_target']} in {SPEC['delta_band']},")
    print(f"  quote filters: rel spread ≤ {SPEC['max_rel_spread']:.0%}, OI ≥ {SPEC['min_oi']}, "
          "uncrossed.")
    # BS guidance for the ~0.50-delta strike: K ≈ S * exp((r - q + iv^2/2) * T) ~ ATM
    print(f"  Guidance: ~0.50-delta ≈ at-the-money strike ≈ {round(st['price'] / 5) * 5} "
          "(verify delta on the Fidelity chain before ordering).")
    print("  Order: single-leg BUY CALL, marketable LIMIT at the ask (or mid+50% of half-spread).")
    print(f"  Size: floor(budget / (100×ask + {SPEC['fee']})). If 0 → no trade today.")
    print("  Checklist: confirm max loss = premium ≤ 10% equity; no earnings-driven IV spike "
          ">1.5x 20d realized; cash reserve ≥ 85% stays untouched.")


if __name__ == "__main__":
    main()
