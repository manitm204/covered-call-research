"""Reinvesting covered-call backtest using the 2026-09 threshold-sweep rule
(see scripts/covered_call_threshold_sweep.py + chat research), not the older
rip-risk board. 200 shares, 100 always covered, every dollar in (premium,
assignment proceeds, dividends, interest) swept back into more whole shares.

Three variants per fund: buy & hold (DRIP), naive monthly covered call
(reinvesting), and the sweep-rule-gated covered call (reinvesting).

Outputs results/covered_call_sweep_rule/<symbol>/<strategy>/{equity,trades,
events}.parquet + summary.json, plus a combined results/covered_call_sweep_rule/
summary_base.csv and a compact JSON for the artifact.
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
OUT = Path("results/covered_call_sweep_rule")
INITIAL_SHARES = 200

# per-ticker rule from this session's threshold sweep (results/covered_call/
# threshold_sweep.json): each leg's own decile cutoff, 90% CI excludes the
# unconditional baseline (SPY RSI) or is the largest, most consistent
# cross-fund effect (sector corr on SPY/QQQ; RSI tercile on IWM, since its
# market-wide leg doesn't replicate).
GATE_PARAMS = {
    "SPY": dict(rsi_thresh=35.0, trend84_thresh=None, use_ma200_leg=False, sector_corr_thresh=0.35),
    "QQQ": dict(rsi_thresh=None, trend84_thresh=-0.08, use_ma200_leg=False, sector_corr_thresh=0.35),
    "IWM": dict(rsi_thresh=46.0, trend84_thresh=None, use_ma200_leg=False, sector_corr_thresh=None),
}


def make(name: str, sig: pd.DataFrame, symbol: str, target_delta: float):
    if name == "buyhold_drip":
        return DripBuyHoldShares(initial_shares=INITIAL_SHARES)
    if name == "naive_cc_reinvest":
        return ReinvestingCoveredCallStrategy(signals=sig, initial_shares=INITIAL_SHARES, gated=False,
                                              target_delta=target_delta)
    if name == "sweep_gated_cc_reinvest":
        return ReinvestingCoveredCallStrategy(signals=sig, initial_shares=INITIAL_SHARES, gated=True,
                                              target_delta=target_delta, **GATE_PARAMS[symbol])
    raise ValueError(name)


def run_one(symbol: str, strat_name: str, scenario: str, target_delta: float, out: Path) -> dict:
    store = ChainStore(symbol)
    daily = DailyData.load(symbol)
    sig = build_signals(symbol)
    strat = make(strat_name, sig, symbol, target_delta)
    sessions = store.sessions()
    px0 = float(store.chain(sessions[0])["underlying_price"].iloc[0])
    initial_cash = round(px0 * INITIAL_SHARES * 1.01, 2)
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
    print(f"{symbol:5s} {strat_name:24s} CAGR={em['cagr']:+.3%}  Sharpe={em['sharpe']:.2f}  "
          f"Sortino={em.get('sortino')}  maxDD={em['max_drawdown']:.1%}  calls={n_calls_written}  "
          f"called_away={n_called_away}  shares {INITIAL_SHARES}->{final_shares}")
    return dict(symbol=symbol, strategy=strat_name, **em, n_calls_written=n_calls_written,
               n_called_away=n_called_away, final_shares=final_shares)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", default="base")
    ap.add_argument("--delta", type=float, default=0.25)
    args = ap.parse_args()
    out = OUT
    rows = []
    for symbol in SYMBOLS:
        for strat_name in ["buyhold_drip", "naive_cc_reinvest", "sweep_gated_cc_reinvest"]:
            rows.append(run_one(symbol, strat_name, args.scenario, target_delta=args.delta, out=out))
    pd.DataFrame(rows).to_csv(out / f"summary_{args.scenario}.csv", index=False)

    # compact bundle for the artifact: metrics table + daily equity curves
    bundle = {"symbols": SYMBOLS,
              "strategies": ["buyhold_drip", "naive_cc_reinvest", "sweep_gated_cc_reinvest"],
              "gate_params": GATE_PARAMS, "metrics": {}, "equity_curves": {}}
    for symbol in SYMBOLS:
        bundle["metrics"][symbol] = {}
        bundle["equity_curves"][symbol] = {}
        for strat_name in bundle["strategies"]:
            summ = json.loads((out / symbol / strat_name / "summary.json").read_text())
            bundle["metrics"][symbol][strat_name] = dict(
                metrics=summ["metrics"], n_calls_written=summ["n_calls_written"],
                n_called_away=summ["n_called_away"], final_shares=summ["final_shares"],
            )
            eq = pd.read_parquet(out / symbol / strat_name / "equity.parquet")
            eq["session"] = pd.to_datetime(eq["session"]).dt.strftime("%Y-%m-%d")
            # monthly-thinned points keep the artifact small
            eq_m = eq.iloc[::5][["session", "equity"]]
            bundle["equity_curves"][symbol][strat_name] = list(
                zip(eq_m["session"], eq_m["equity"].round(2)))
    Path(out / "artifact_bundle.json").write_text(json.dumps(bundle, default=str))
    print("wrote", out / "artifact_bundle.json")


if __name__ == "__main__":
    main()
