"""Cycle-2 IC screen: Spearman rank IC of every cycle-2 feature vs forward
targets, computed on TRAIN (2018-08..2022-12), with sign-stability check on
VALIDATION (2023-24). Holdout untouched.

Targets per underlying (QQQ, SPY): fwd21 return; fwd21 left-tail (<-5%);
fwd21 realized vol. Moving-block bootstrap CI90 on the train IC (block 42d).
Output: results/level2/ic_screen.csv + significant rows logged.
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd
from scipy.stats import rankdata

sys.path.insert(0, "scripts")
from l2_common import append_experiment_log  # noqa: E402

RNG = np.random.default_rng(7)
TRAIN = ("2018-08-01", "2022-12-31")
VALID = ("2023-01-01", "2024-12-31")


def spearman(a, b):
    return np.corrcoef(rankdata(a), rankdata(b))[0, 1]


def ic_ci(x, y, block=42, n_boot=1000):
    n = len(x)
    k = max(1, int(np.ceil(n / block)))
    vals = []
    for _ in range(n_boot):
        st = RNG.integers(0, max(1, n - block + 1), size=k)
        idx = np.concatenate([np.arange(s, s + block) for s in st])[:n]
        vals.append(spearman(x[idx], y[idx]))
    return float(np.quantile(vals, 0.05)), float(np.quantile(vals, 0.95))


def main():
    F = pd.read_parquet("data/features/cycle2_features.parquet")
    F = F.drop(columns=[c for c in ("date",) if c in F.columns])
    rows = []
    for sym in ("QQQ", "SPY"):
        px = pd.read_parquet(f"data/normalized/prices_long/{sym}.parquet")
        px["date"] = pd.to_datetime(px.date)
        a = px.set_index("date").adjClose.sort_index()
        r1 = a.pct_change()
        tgt = pd.DataFrame(index=a.index)
        tgt["fwd21_ret"] = a.shift(-21) / a - 1
        tgt["fwd21_tail"] = (tgt.fwd21_ret < -0.05).astype(float)
        tgt["fwd21_rv"] = r1.shift(-21).rolling(21).std().shift(-20) * np.sqrt(252)
        for feat in F.columns:
            f = F[feat]
            for tname in tgt.columns:
                d = pd.DataFrame(dict(f=f, t=tgt[tname])).dropna()
                tr = d.loc[TRAIN[0]:TRAIN[1]]
                va = d.loc[VALID[0]:VALID[1]]
                if len(tr) < 300 or tr.f.nunique() < 20:
                    continue
                ic_tr = spearman(tr.f.values, tr.t.values)
                lo, hi = ic_ci(tr.f.values, tr.t.values)
                ic_va = spearman(va.f.values, va.t.values) if len(va) > 200 else np.nan
                sig = (lo > 0 and ic_tr > 0) or (hi < 0 and ic_tr < 0)
                stable = sig and ic_va == ic_va and np.sign(ic_va) == np.sign(ic_tr) \
                    and abs(ic_va) > 0.03
                rows.append(dict(symbol=sym, feature=feat, target=tname, n=len(tr),
                                 ic_train=round(ic_tr, 3), ci_lo=round(lo, 3),
                                 ci_hi=round(hi, 3), ic_valid=round(ic_va, 3) if ic_va == ic_va else None,
                                 significant=sig, stable=bool(stable)))
    out = pd.DataFrame(rows)
    out.to_csv("results/level2/ic_screen.csv", index=False)
    stable = out[out.stable].sort_values("ic_train", key=abs, ascending=False)
    print(f"{len(out)} feature-target pairs; {out.significant.sum()} train-significant; "
          f"{len(stable)} sign-stable into validation")
    print(stable.head(25).to_string(index=False))
    append_experiment_log(hypothesis="IC_SCREEN", partition="train+valid", underlying="QQQ/SPY",
                          config="57 features x 3 targets x 2 syms", scenario="n/a",
                          n=len(out), metric="n_stable", value=len(stable),
                          notes="full table results/level2/ic_screen.csv; multiple-testing: "
                                f"expect ~{int(len(out)*0.10)} false sig at 10% level")


if __name__ == "__main__":
    main()
