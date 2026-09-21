"""Reinvesting covered-call variant: own 200 shares, write calls on 100 of
them (100 always uncovered as a buffer), sweep every dollar that comes in —
premium, assignment proceeds, dividends, interest — straight back into more
whole shares. Naive (every month) and rip-risk-gated versions, vs a
DRIP buy-and-hold baseline that reinvests the same way but never sells calls.
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
from level2_research.strategies import DripBuyHoldShares, ReinvestingCoveredCallStrategy

SYMBOLS = ["SPY", "QQQ", "IWM"]
OUT = Path("results/covered_call_reinvest")
INITIAL_SHARES = 200


def make(name: str, sig: pd.DataFrame, target_delta: float):
    if name == "buyhold_drip":
        return DripBuyHoldShares(initial_shares=INITIAL_SHARES)
    if name == "naive_cc_reinvest":
        return ReinvestingCoveredCallStrategy(signals=sig, initial_shares=INITIAL_SHARES, gated=False,
                                              target_delta=target_delta)
    if name == "gated_cc_reinvest":
        return ReinvestingCoveredCallStrategy(signals=sig, initial_shares=INITIAL_SHARES, gated=True,
                                              target_delta=target_delta)
    raise ValueError(name)


def run_one(symbol: str, strat_name: str, scenario: str, target_delta: float = 0.25,
           out: Path = OUT) -> dict:
    store = ChainStore(symbol)
    daily = DailyData.load(symbol)
    sig = build_signals(symbol)
    strat = make(strat_name, sig, target_delta)
    sessions = store.sessions()
    px0 = float(store.chain(sessions[0])["underlying_price"].iloc[0])
    initial_cash = round(px0 * INITIAL_SHARES * 1.01, 2)  # 1% headroom for fees/rounding
    eng = Engine(store, daily, strat, scenario=scenario, initial_cash=initial_cash)
    res = eng.run()
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
    final_shares = int(res.equity.iloc[-1]["shares"])
    (d / "summary.json").write_text(json.dumps(dict(
        symbol=symbol, strategy=strat_name, scenario=scenario, target_delta=target_delta,
        metrics=em, trade_metrics=tm, yearly=yr.to_dict(),
        n_calls_written=n_calls_written, n_called_away=n_called_away,
        initial_shares=INITIAL_SHARES, final_shares=final_shares,
    ), indent=2, default=str))
    print(f"{symbol:5s} {strat_name:18s} CAGR={em['cagr']:+.3%}  Sharpe={em['sharpe']:.2f}  "
          f"maxDD={em['max_drawdown']:.1%}  calls={n_calls_written}  called_away={n_called_away}  "
          f"shares {INITIAL_SHARES}->{final_shares}")
    return dict(symbol=symbol, strategy=strat_name, **em, n_calls_written=n_calls_written,
               n_called_away=n_called_away, final_shares=final_shares)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", default="base")
    ap.add_argument("--delta", type=float, default=0.25)
    ap.add_argument("--out", default=None, help="override output root (default results/covered_call_reinvest)")
    args = ap.parse_args()
    out = Path(args.out) if args.out else OUT
    rows = []
    for symbol in SYMBOLS:
        for strat_name in ["buyhold_drip", "naive_cc_reinvest", "gated_cc_reinvest"]:
            rows.append(run_one(symbol, strat_name, args.scenario, target_delta=args.delta, out=out))
    pd.DataFrame(rows).to_csv(out / f"summary_{args.scenario}.csv", index=False)


if __name__ == "__main__":
    main()
