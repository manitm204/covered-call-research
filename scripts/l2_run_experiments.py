"""Phase-3 coarse experiment runner (TRAIN period only: 2018-08-01 .. 2022-12-31).

Usage:
    python scripts/l2_run_experiments.py H1 SPY [--scenario base] [--start ... --end ...]
    python scripts/l2_run_experiments.py --all       # coarse grid, all hypotheses

Every run appends to experiment_log.csv and writes an equity/trades parquet under
results/level2/runs/<run_id>/.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import date
from pathlib import Path

import pandas as pd

sys.path.insert(0, "scripts")
from l2_common import append_experiment_log  # noqa: E402

from level2_research.engine import Engine  # noqa: E402
from level2_research.market import ChainStore, DailyData  # noqa: E402
from level2_research.metrics import (block_bootstrap_ci, capital_utilization,  # noqa: E402
                                     equity_metrics, trade_metrics, yearly_returns)
from level2_research.signals import build_signals  # noqa: E402
from level2_research.strategies import (AlwaysAtmCallBenchmark, H1RebounCalls,  # noqa: E402
                                        H2TrendCalls, H4TrendPuts, H5Wheel)

TRAIN = (date(2018, 8, 1), date(2022, 12, 31))
VALID = (date(2023, 1, 1), date(2024, 12, 31))
HOLDOUT = (date(2025, 1, 1), date(2026, 7, 22))  # single use, guarded elsewhere

OUT = Path("results/level2/runs")


def make_strategy(hyp: str, symbol: str, params: dict):
    sig = build_signals(symbol) if hyp in ("H1", "H2", "H4") else None
    if hyp == "H1":
        return H1RebounCalls(signals=sig, **params)
    if hyp == "H2":
        return H2TrendCalls(signals=sig, **params)
    if hyp == "H4":
        return H4TrendPuts(signals=sig, **params)
    if hyp == "BENCH":
        return AlwaysAtmCallBenchmark(**params)
    if hyp == "H5":
        return H5Wheel(**params)
    raise ValueError(hyp)


def run_one(hyp: str, symbol: str, scenario: str, params: dict,
            start: date, end: date, partition: str) -> dict:
    store = ChainStore(symbol)
    daily = DailyData.load(symbol)
    strat = make_strategy(hyp, symbol, params)
    eng = Engine(store, daily, strat, scenario=scenario, initial_cash=10_000.0)
    res = eng.run(start=start, end=end)
    em = equity_metrics(res.equity, rf=daily.tbill)
    tm = trade_metrics(res.trades)
    cfg = json.dumps(params, sort_keys=True, default=str)
    run_id = f"{hyp}_{symbol}_{scenario}_" + hashlib.md5(
        (cfg + partition).encode()).hexdigest()[:8]
    d = OUT / run_id
    d.mkdir(parents=True, exist_ok=True)
    res.equity.to_parquet(d / "equity.parquet")
    if len(res.trades):
        res.trades.to_parquet(d / "trades.parquet")
    if len(res.events):
        res.events.to_parquet(d / "events.parquet")
    (d / "config.json").write_text(json.dumps(dict(
        hypothesis=hyp, symbol=symbol, scenario=scenario, params=params,
        partition=partition, start=str(start), end=str(end),
        metrics=em, trade_metrics=tm), indent=2, default=str))
    ci = (float("nan"), float("nan"))
    if tm.get("n_trades", 0) >= 8:
        pnl = res.trades.pnl.dropna().values
        ci = block_bootstrap_ci(pnl, block=3)
    append_experiment_log(
        hypothesis=hyp, partition=partition, underlying=symbol,
        config=cfg[:180], scenario=scenario, n=tm.get("n_trades", 0),
        metric="ev_per_trade", value=tm.get("ev_per_trade", ""),
        ci_lo=round(ci[0], 2) if ci[0] == ci[0] else "",
        ci_hi=round(ci[1], 2) if ci[1] == ci[1] else "",
        verdict="", notes=f"run_id={run_id} cagr={em['cagr']} sharpe={em['sharpe']} "
                          f"maxdd={em['max_drawdown']} final={em['final_equity']}")
    yr = yearly_returns(res.equity)
    return dict(run_id=run_id, hyp=hyp, symbol=symbol, scenario=scenario, params=params,
                **{k: em[k] for k in ("cagr", "ann_vol", "sharpe", "max_drawdown", "final_equity")},
                **tm, util=capital_utilization(res.equity),
                yearly=yr.to_dict())


COARSE = {
    "H1": [dict(dd_thresh=t, confirm_ret5=0.02, target_delta=d, dte_target=dt,
                dte_lo=dt - 15, dte_hi=dt + 20, hold_days=h, budget_pct=0.10)
           for t in (0.12, 0.15) for d in (0.35, 0.50) for dt, h in ((45, 40), (60, 60))],
    "H2": [dict(target_delta=d, dte_target=60, dte_lo=40, dte_hi=70, roll_dte=21,
                budget_pct=b)
           for d in (0.35, 0.50, 0.65, 0.80) for b in (0.06, 0.10, 0.15)],
    "H4": [dict(target_delta=-0.60, dte_target=60, dte_lo=40, dte_hi=70, roll_dte=21,
                budget_pct=0.10)],
    "BENCH": [dict(budget_pct=0.10)],
}

# H5 wheel runs on the sub-$100 ETFs (separate driver: --wheel)
WHEEL = [dict(target_delta=d, dte_target=35, dte_lo=25, dte_hi=50, profit_take=pt,
              manage_dte=7, budget_frac=0.80)
         for d in (-0.20, -0.25, -0.30) for pt in (0.5, 1.1)]  # pt=1.1 -> never takes profit
WHEEL_SYMS = ["XLF", "SLV", "EWZ"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("hyp", nargs="?", default=None)
    ap.add_argument("symbol", nargs="?", default=None)
    ap.add_argument("--scenario", default="base")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--wheel", action="store_true")
    ap.add_argument("--partition", default="train", choices=["train", "valid", "holdout"])
    args = ap.parse_args()
    span = {"train": TRAIN, "valid": VALID, "holdout": HOLDOUT}[args.partition]
    if args.partition == "holdout":
        guard = Path("results/level2/HOLDOUT_UNLOCKED")
        if not guard.exists():
            print("HOLDOUT is locked. Create results/level2/HOLDOUT_UNLOCKED after "
                  "freezing the spec (see research_plan.md §7 G5).")
            return 2
    rows = []
    if args.all or args.wheel:
        from concurrent.futures import ProcessPoolExecutor, as_completed
        if args.wheel:
            jobs = [("H5", symbol, "base", params)
                    for symbol in WHEEL_SYMS for params in WHEEL]
        else:
            jobs = [(hyp, symbol, "base", params)
                    for hyp, grids in COARSE.items()
                    for symbol in ["SPY", "QQQ", "IWM"]
                    for params in grids]
        with ProcessPoolExecutor(max_workers=10) as ex:
            futs = {ex.submit(run_one, h, s, sc, p, *span, args.partition): (h, s, p)
                    for h, s, sc, p in jobs}
            for f in as_completed(futs):
                h, s, p = futs[f]
                try:
                    r = f.result()
                    rows.append(r)
                    print(f"{r['run_id']}: cagr={r['cagr']:.3f} sharpe={r['sharpe']} "
                          f"dd={r['max_drawdown']:.3f} trades={r.get('n_trades')} "
                          f"ev={r.get('ev_per_trade')}", flush=True)
                except Exception as e:
                    print(f"FAIL {h} {s} {p}: {e}", flush=True)
        stem = "wheel" if args.wheel else "coarse"
        pd.DataFrame(rows).to_csv(f"results/level2/{stem}_{args.partition}_summary.csv", index=False)
    else:
        params = COARSE[args.hyp][0]
        r = run_one(args.hyp, args.symbol, args.scenario, params, *span, args.partition)
        print(json.dumps(r, indent=2, default=str))


if __name__ == "__main__":
    main()
