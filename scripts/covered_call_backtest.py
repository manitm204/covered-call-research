"""Covered-call research: BuyHold vs naive monthly covered calls vs a
signal-gated covered call (skip writing through momentum/rebound "rip" regimes)
for SPY, QQQ, IWM. New work (2026-09) — not part of the frozen TG-CER spec.

Usage:
    python scripts/covered_call_backtest.py [--scenario base]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from level2_research.engine import Engine
from level2_research.market import ChainStore, DailyData
from level2_research.metrics import equity_metrics, trade_metrics, yearly_returns
from level2_research.signals import build_signals
from level2_research.strategies import BuyHoldShares, CoveredCallStrategy

SYMBOLS = ["SPY", "QQQ", "IWM"]
OUT = Path("results/covered_call")
START, END = None, None  # full available history (2018-08 .. latest)

# Per-ticker rule legs, derived from scripts/covered_call_signal_research.py
# (90% block-bootstrap CI, 2005-2026 weekly sample). See results/covered_call/
# signal_research.json "cutoffs" and "forest" for the underlying numbers.
#
# Breach gate (Rule 1): RSI<40 and trend84<0% clear the bar on all three
# funds; price-below-MA200 only clears it on SPY/QQQ (IWM's CI crosses zero,
# so its rule drops that leg).
BREACH_RULE = {
    "SPY": dict(use_ma200_leg=True),
    "QQQ": dict(use_ma200_leg=True),
    "IWM": dict(use_ma200_leg=False),
}

# Forward-return gate (Rule 2): built from every signal whose *continuous*
# rank IC clears a 90% CI against plain forward return, on that fund,
# excluding absorption_shift/vol_rank_252 as near-duplicates of vol_own and
# sector_corr_60. Thresholds are tercile cutoffs from that fund's own weekly
# distribution. Caveat, logged here rather than silently: RSI-lo-tercile and
# sector-corr-hi-tercile also clear a *bucket-level* 90% CI test on all three
# funds, but trend84<0% and vol-hi-tercile do not (SPY/QQQ) — the continuous
# IC is real, the specific binary cutoff tested here is weaker evidence. See
# "Threshold sensitivity" in the report for the full breakdown.
FWDRET_RULE = {
    "SPY": dict(rsi_thresh=50.0, sector_corr_thresh=0.677, trend84_thresh=0.0,
               vol_thresh=19.6, use_ma200_leg=False),
    "QQQ": dict(rsi_thresh=48.2, sector_corr_thresh=0.677, trend84_thresh=0.0,
               vol_thresh=23.2, use_ma200_leg=False),
    "IWM": dict(rsi_thresh=47.5, sector_corr_thresh=0.677, trend84_thresh=0.0,
               vol_thresh=25.2, use_ma200_leg=True),
}


def make(name: str, sig: pd.DataFrame, symbol: str, target_delta: float):
    if name == "buyhold":
        return BuyHoldShares(lots=1)
    if name == "naive_cc":
        return CoveredCallStrategy(signals=sig, lots=1, gated=False, target_delta=target_delta)
    if name == "gated_cc":
        return CoveredCallStrategy(signals=sig, lots=1, gated=True, target_delta=target_delta,
                                   **BREACH_RULE[symbol])
    if name == "fwdret_cc":
        r = FWDRET_RULE[symbol]
        return CoveredCallStrategy(signals=sig, lots=1, gated=True, target_delta=target_delta,
                                   gate_mode="fwd_ret", fwdret_sector_corr_thresh=r["sector_corr_thresh"],
                                   fwdret_vol_thresh=r["vol_thresh"], fwdret_rsi_thresh=r["rsi_thresh"],
                                   fwdret_trend84_thresh=r["trend84_thresh"],
                                   fwdret_use_ma200_leg=r["use_ma200_leg"])
    raise ValueError(name)


def run_one(symbol: str, strat_name: str, scenario: str, target_delta: float = 0.25,
           out: Path = OUT) -> dict:
    store = ChainStore(symbol)
    daily = DailyData.load(symbol)
    sig = build_signals(symbol)
    strat = make(strat_name, sig, symbol, target_delta)
    # Fund exactly one 100-share lot (+2% headroom) so every variant starts
    # fully invested and equity/drawdown reflect a pure single-lot position —
    # not a lot diluted inside a large cash-heavy account.
    sessions = store.sessions()
    px0 = float(store.chain(sessions[0])["underlying_price"].iloc[0])
    initial_cash = round(px0 * 100 * 1.02, 2)
    eng = Engine(store, daily, strat, scenario=scenario, initial_cash=initial_cash)
    res = eng.run(start=START, end=END)
    em = equity_metrics(res.equity, rf=daily.tbill)
    tm = trade_metrics(res.trades)
    yr = yearly_returns(res.equity)

    d = out / symbol / strat_name
    d.mkdir(parents=True, exist_ok=True)
    res.equity.to_parquet(d / "equity.parquet")
    if len(res.trades):
        res.trades.to_parquet(d / "trades.parquet")
    if len(res.events):
        res.events.to_parquet(d / "events.parquet")

    n_calls_written = int((res.trades.tag == "cc_write").sum()) if len(res.trades) else 0
    n_called_away = int((res.trades.close_how == "called_away").sum()) if "close_how" in getattr(res.trades, "columns", []) else 0
    (d / "summary.json").write_text(json.dumps(dict(
        symbol=symbol, strategy=strat_name, scenario=scenario, target_delta=target_delta,
        metrics=em, trade_metrics=tm, yearly=yr.to_dict(),
        n_calls_written=n_calls_written, n_called_away=n_called_away,
    ), indent=2, default=str))
    print(f"{symbol:5s} {strat_name:10s} CAGR={em['cagr']:+.3%}  Sharpe={em['sharpe']:.2f}  "
          f"maxDD={em['max_drawdown']:.1%}  calls_written={n_calls_written}  called_away={n_called_away}")
    return dict(symbol=symbol, strategy=strat_name, **em, n_calls_written=n_calls_written,
               n_called_away=n_called_away)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", default="base")
    ap.add_argument("--delta", type=float, default=0.25)
    ap.add_argument("--out", default=None, help="override output root (default results/covered_call)")
    args = ap.parse_args()
    out = Path(args.out) if args.out else OUT
    rows = []
    for symbol in SYMBOLS:
        for strat_name in ["buyhold", "naive_cc", "gated_cc", "fwdret_cc"]:
            rows.append(run_one(symbol, strat_name, args.scenario, target_delta=args.delta, out=out))
    pd.DataFrame(rows).to_csv(out / f"summary_{args.scenario}.csv", index=False)


if __name__ == "__main__":
    main()
