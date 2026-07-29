"""Robustness suite for the H2 finalist (TRAIN ONLY — validation stays untouched).

Blocks:
  A. parameter neighborhood: delta x roll_dte x dte_target x budget
  B. hysteresis band {0, 1, 2}% and entry delay {0, 1, 2} sessions
  C. execution stress (conservative fills already run; here: 'stress' scenario)
  D. outlier dependence: drop best trade / best year (analytic, from ledger)
  E. regime windows: 2018Q4, COVID crash, 2020 recovery, 2021 bull, 2022 bear
  F. block-bootstrap CIs on per-trade P&L and daily returns

Usage: python scripts/l2_robustness.py [SYMBOL]
Writes results/level2/robustness_<sym>.csv and appends to experiment_log.csv.
"""

from __future__ import annotations

import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import date

import numpy as np
import pandas as pd

sys.path.insert(0, "scripts")
from l2_run_experiments import TRAIN, run_one  # noqa: E402

BASELINE = dict(target_delta=0.5, dte_target=60, dte_lo=40, dte_hi=70,
                roll_dte=21, budget_pct=0.10, band=0.01, extra_lag=0)


def neighborhood(sym: str) -> list[dict]:
    out = []
    jobs = []
    for d in (0.40, 0.50, 0.60):
        for roll in (14, 21, 30):
            for dte in (45, 60):
                p = dict(BASELINE, target_delta=d, roll_dte=roll, dte_target=dte,
                         dte_lo=dte - 20, dte_hi=dte + 10)
                jobs.append(("A_neighborhood", p, "base"))
    for band in (0.0, 0.02):
        jobs.append(("B_band", dict(BASELINE, band=band), "base"))
    for lag in (1, 2):
        jobs.append(("B_lag", dict(BASELINE, extra_lag=lag), "base"))
    for b in (0.06, 0.08, 0.12):
        jobs.append(("A_budget", dict(BASELINE, budget_pct=b), "base"))
    jobs.append(("C_stress", dict(BASELINE), "stress"))
    with ProcessPoolExecutor(max_workers=10) as ex:
        futs = {ex.submit(run_one, "H2", sym, sc, p, *TRAIN, "train"): (blk, p, sc)
                for blk, p, sc in jobs}
        for f in as_completed(futs):
            blk, p, sc = futs[f]
            try:
                r = f.result()
                out.append(dict(block=blk, scenario=sc,
                                delta=p["target_delta"], roll=p["roll_dte"],
                                dte=p["dte_target"], budget=p["budget_pct"],
                                band=p.get("band"), lag=p.get("extra_lag"),
                                cagr=r["cagr"], sharpe=r["sharpe"],
                                maxdd=r["max_drawdown"], n=r.get("n_trades"),
                                ev=r.get("ev_per_trade"), run_id=r["run_id"]))
                print(f"{blk} d={p['target_delta']} roll={p['roll_dte']} dte={p['dte_target']} "
                      f"b={p['budget_pct']} band={p.get('band')} lag={p.get('extra_lag')} "
                      f"{sc}: cagr={r['cagr']:+.3f} sharpe={r['sharpe']:+.2f} "
                      f"dd={r['max_drawdown']:.3f}", flush=True)
            except Exception as e:
                print("FAIL", blk, p, e, flush=True)
    return out


def outlier_and_regime(sym: str, run_id: str):
    import glob
    import json
    d = f"results/level2/runs/{run_id}"
    eq = pd.read_parquet(f"{d}/equity.parquet")
    tr = pd.read_parquet(f"{d}/trades.parquet")
    eq["session"] = pd.to_datetime(eq.session)
    e = eq.set_index("session").equity
    rets = e.pct_change().dropna()
    pnl = tr.pnl.dropna()
    res = {}
    res["total_pnl"] = pnl.sum()
    res["pnl_wo_best_trade"] = pnl.sum() - pnl.max()
    res["pnl_wo_2best"] = pnl.sum() - pnl.nlargest(2).sum()
    tr["year"] = pd.to_datetime(tr.close_session).dt.year
    yr = tr.groupby("year").pnl.sum()
    res["pnl_wo_best_year"] = pnl.sum() - yr.max()
    res["best_year"] = int(yr.idxmax())
    windows = {
        "2018Q4_selloff": ("2018-09-20", "2018-12-31"),
        "covid_crash": ("2020-02-19", "2020-03-23"),
        "covid_recovery": ("2020-03-24", "2020-12-31"),
        "bull_2021": ("2021-01-01", "2021-12-31"),
        "bear_2022": ("2022-01-01", "2022-12-31"),
    }
    for name, (a, b) in windows.items():
        w = e.loc[a:b]
        if len(w) > 2:
            res[f"ret_{name}"] = round(float(w.iloc[-1] / w.iloc[0] - 1), 4)
    from level2_research.metrics import block_bootstrap_ci
    lo, hi = block_bootstrap_ci(pnl.values, block=3)
    res["trade_ev_ci95"] = (round(lo, 1), round(hi, 1))
    lo, hi = block_bootstrap_ci(rets.values, block=21,
                                stat=lambda x: x.mean() / (x.std() + 1e-12) * np.sqrt(252))
    res["sharpe_ci95"] = (round(lo, 2), round(hi, 2))
    return res


if __name__ == "__main__":
    sym = sys.argv[1] if len(sys.argv) > 1 else "QQQ"
    rows = neighborhood(sym)
    df = pd.DataFrame(rows)
    df.to_csv(f"results/level2/robustness_{sym}.csv", index=False)
    base_row = df[(df.block == "A_neighborhood") & (df.delta == 0.5) & (df.roll == 21)
                  & (df.dte == 60)]
    rid = base_row.run_id.iloc[0]
    extra = outlier_and_regime(sym, rid)
    print("\n--- outlier/regime on baseline", rid)
    for k, v in extra.items():
        print(f"{k}: {v}")
    pd.Series(extra).to_json(f"results/level2/robustness_{sym}_outlier.json")
