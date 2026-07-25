"""EXPLORATORY QQQ signal ablation (overfit — hypothesis generation only).

Mirrors the SPY signal study on the QQQ 414-trade panel:
  A. 10 candidate signals: on/off means, subperiod consistency, frequency
  B. phi correlation matrix (redundancy clusters) -> saved for heatmap
  C. all-pairs AND means on the panel (which couples work together)
  D. engine cap-2 verification: mandatory-signal + k-of-rest voting variants,
     top pairs, top triples
"""

from __future__ import annotations

import itertools
import json
import sys

import numpy as np
import polars as pl

sys.path.insert(0, "scripts")
from qqq_postmortem import run  # noqa: E402
from preregistered_qqq_test import FEATURES, SECTOR_CORR  # noqa: E402
from phaseA_aggression_frontier import episode_boot_ci, episode_ids  # noqa: E402
from xsp_research.evaluation.robustness import bootstrap_mean_ci  # noqa: E402

SIGNALS = {
    "trend120":    ("ret_120d", ">", 0.15),
    "trend60":     ("ret_60d", ">", 0.08),
    "ma200_10":    ("dist_ma200", ">", 0.10),
    "ma50_up":     ("ma50_slope_20d", ">", 0.02),
    "beta_hi":     ("qqq_beta_spy_60", ">", 1.30),
    "absorb_fall": ("absorption_chg_20d", "<", 0.0),
    "seccorr_lo":  ("sector_avg_corr_20", "<", 0.45),
    "iv_rich":     ("iv_minus_rv20", ">", 0.0),
    "iwm_decor":   ("corr_spy_iwm_20", "<", 0.80),
    "vxn_lo":      ("vix_level", "<", 25.0),
}

panel = pl.read_parquet("results/qqq_prereg/regime_scan_panel.parquet")
pnl = panel["pnl"].to_numpy()
yr = panel["entry_date"].dt.year().to_numpy()
sig = {}
for name, (col, op, thr) in SIGNALS.items():
    v = panel[col].to_numpy()
    sig[name] = (v < thr) if op == "<" else (v > thr)
names = list(SIGNALS)

print(f"=== A. PER-SIGNAL (panel n={len(pnl)}, mean ${pnl.mean():.1f}) ===")
print(f"{'signal':12s} {'freq':>5s} | {'n_on':>4s} {'mean_on':>8s} {'win':>4s} "
      f"{'18-20':>7s} {'21-23':>7s} {'24-26':>7s} | {'mean_off':>8s}")
for s in names:
    m = sig[s]
    subs = []
    for a, b in [(2018, 2020), (2021, 2023), (2024, 2026)]:
        mm = m & (yr >= a) & (yr <= b)
        subs.append(pnl[mm].mean() if mm.sum() >= 5 else float("nan"))
    print(f"{s:12s} {m.mean():5.2f} | {m.sum():4d} {pnl[m].mean():8.1f} "
          f"{(pnl[m] > 0).mean():4.2f} {subs[0]:7.1f} {subs[1]:7.1f} {subs[2]:7.1f} "
          f"| {pnl[~m].mean():8.1f}")

M = np.array([sig[s] for s in names], dtype=float)
C = np.corrcoef(M)
print("\n=== B. PHI CORRELATION (redundancy) ===")
print("            " + " ".join(f"{s[:7]:>8s}" for s in names))
for i, s in enumerate(names):
    print(f"{s:12s}" + " ".join(f"{C[i, j]:8.2f}" for j in range(len(names))))
json.dump({"names": names, "matrix": C.tolist()},
          open("results/qqq_prereg/signal_phi_matrix.json", "w"))

print("\n=== C. ALL PAIRS (AND), panel means, n>=60, sorted ===")
pairs = []
for a, b in itertools.combinations(names, 2):
    m = sig[a] & sig[b]
    if m.sum() >= 60:
        pairs.append((a, b, int(m.sum()), pnl[m].mean(), (pnl[m] > 0).mean()))
for a, b, n, mu, w in sorted(pairs, key=lambda x: -x[3])[:15]:
    print(f"  {a:12s}+{b:12s} n={n:3d} mean=${mu:7.1f} win={w:.2f}")

# ---- D. engine verification
feats = pl.read_parquet(FEATURES).join(pl.read_parquet(SECTOR_CORR), on="date", how="left")
rows = {r["date"]: r for r in feats.iter_rows(named=True)}


def sval(r, name):
    col, op, thr = SIGNALS[name]
    v = r.get(col)
    if v is None:
        raise TypeError
    return (v < thr) if op == "<" else (v > thr)


def make(must=(), vote_pool=(), k=None):
    def gate(s):
        r = rows.get(s)
        if r is None:
            return False, "no row"
        try:
            if not all(sval(r, m) for m in must):
                return False, "must off"
            if k is not None and sum(sval(r, v) for v in vote_pool) < k:
                return False, "votes"
            return True, "ok"
        except TypeError:
            return False, "na"
    return gate


CORE4 = ["absorb_fall", "seccorr_lo", "iv_rich", "iwm_decor"]
combos: dict[str, tuple] = {}
for must in ["trend120", "trend60", "ma200_10", "beta_hi"]:
    combos[f"{must} + >=2of4 core"] = ((must,), CORE4, 2)
for k in (1, 2, 3):
    combos[f"trend120 + >={k}of4 core"] = (("trend120",), CORE4, k)
TOP = ["trend120", "trend60", "ma200_10", "beta_hi", "absorb_fall", "seccorr_lo"]
for a, b in itertools.combinations(TOP, 2):
    combos[f"pair {a}+{b}"] = ((a, b), (), None)
combos["triple trend120+absorb+seccorr"] = (("trend120", "absorb_fall", "seccorr_lo"), (), None)
combos["triple trend120+ma200+absorb"] = (("trend120", "ma200_10", "absorb_fall"), (), None)
combos["triple ma200+absorb+seccorr"] = (("ma200_10", "absorb_fall", "seccorr_lo"), (), None)
combos["triple trend120+beta+absorb"] = (("trend120", "beta_hi", "absorb_fall"), (), None)

print(f"\n=== D. ENGINE CAP-2 (base fills) — {len(combos)} combos ===")
print(f"{'combo':38s} {'n':>3s} {'mean$':>8s} {'tradeCI':>18s} {'epCI':>18s} "
      f"{'win':>4s} {'worst':>6s} {'tot':>6s}")
results = []
for name, (must, pool, k) in combos.items():
    t = run(make(must, pool, k), 2)
    if not t.height:
        print(f"{name:38s}   0")
        continue
    per = t["pnl"].to_numpy()
    ci = bootstrap_mean_ci(per, n_boot=2000, block=5, seed=7)
    eids = episode_ids(t["entry_date"].to_numpy())
    lo, hi, _ = episode_boot_ci(per, eids)
    results.append({"combo": name, "n": t.height, "mean": float(per.mean()),
                    "ci": [ci["ci_low"], ci["ci_high"]], "ep_ci": [lo, hi],
                    "win": float((per > 0).mean()), "worst": float(per.min()),
                    "total": float(per.sum())})
    print(f"{name:38s} {t.height:3d} {per.mean():8.2f} "
          f"[{ci['ci_low']:7.2f},{ci['ci_high']:7.2f}] [{lo:7.2f},{hi:7.2f}] "
          f"{(per > 0).mean():4.2f} {per.min():6.0f} {per.sum():6.0f}")
json.dump(results, open("results/qqq_prereg/signal_ablation_engine.json", "w"), indent=2)
