"""EXPLORATORY QQQ regime scan (deliberately overfit — hypothesis generation only).

Question: does QQQ's bear-call-spread profitability condition on the SAME
regimes as SPY (just weaker), or on completely different ones?

1. Unfiltered QQQ panel: weekly, 0.15d, $8->grid, 30 DTE, hold, max 12 open
   (analog of SPY's 417-trade panel; n~414, mean ~-$47).
2. Join all 37 QQQ features (UNDERLYING=NDX, VIX=VXN) + market-wide regime v2.
3. Tercile scan of every numeric feature, ranked by hi-lo spread.
4. Head-to-head: the exact SPY-study signals evaluated on QQQ.
Output: results/qqq_prereg/regime_scan_panel.parquet + printed report.
"""

from __future__ import annotations

import sys

import numpy as np
import polars as pl

sys.path.insert(0, "scripts")
from qqq_postmortem import run  # noqa: E402  (reuses engine wiring)
from preregistered_qqq_test import FEATURES, SECTOR_CORR  # noqa: E402

t = run(None, 12)
print(f"panel: n={t.height} mean=${t['pnl'].mean():.2f} win={(t['pnl'] > 0).mean():.2f}")

feats = pl.read_parquet(FEATURES).join(
    pl.read_parquet(SECTOR_CORR), on="date", how="left"
)
panel = t.join(feats, left_on="entry_date", right_on="date", how="left")
panel.write_parquet("results/qqq_prereg/regime_scan_panel.parquet")
pnl = panel["pnl"].to_numpy()
yr = panel["entry_date"].dt.year().to_numpy()

skip = {"pnl", "trade_id", "qty", "width", "short_strike", "long_strike",
        "entry_credit", "exit_price", "dte_at_entry", "realized_gross",
        "realized_net", "fees", "max_loss_dollars", "return_on_max_risk",
        "pct_credit_captured", "mfe_dollars", "mae_dollars", "days_in_trade",
        "short_delta_at_entry", "short_iv_at_entry"}
numeric = [c for c, d in zip(panel.columns, panel.dtypes)
           if d in (pl.Float64, pl.Float32) and c not in skip]

rows = []
for c in numeric:
    v = panel[c].to_numpy()
    ok = np.isfinite(v)
    if ok.sum() < 200:
        continue
    q1, q2 = np.nanquantile(v[ok], [1 / 3, 2 / 3])
    if q1 == q2:
        continue
    lo, mid, hi = pnl[ok & (v <= q1)], pnl[ok & (v > q1) & (v <= q2)], pnl[ok & (v > q2)]
    rows.append({"feature": c, "lo": lo.mean(), "mid": mid.mean(), "hi": hi.mean(),
                 "spread": hi.mean() - lo.mean(),
                 "n": int(ok.sum())})
sc = pl.DataFrame(rows).sort(pl.col("spread").abs(), descending=True)
print(f"\n=== TERCILE SCAN, top 20 of {sc.height} by |hi-lo| (panel mean ${pnl.mean():.1f}) ===")
print(f"{'feature':26s} {'lo':>8s} {'mid':>8s} {'hi':>8s} {'hi-lo':>8s}")
for r in sc.head(20).iter_rows(named=True):
    print(f"{r['feature']:26s} {r['lo']:8.1f} {r['mid']:8.1f} {r['hi']:8.1f} {r['spread']:8.1f}")

# ---- the exact SPY-study signals, evaluated on QQQ
SIGNALS = {
    "rsi70 (SPY: +$23 on)": pl.col("rsi_14") > 70,
    "iv>rv (SPY: off -$118)": pl.col("iv_minus_rv20") > 0,
    "vxn<25 (SPY vix<25)": pl.col("vix_level") < 25,
    "vxn<20 (SPY vix<20)": pl.col("vix_level") < 20,
    "absorb_fall (SPY +$16)": pl.col("absorption_chg_20d") < 0,
    "seccorr<.45 (SPY +$11)": pl.col("sector_avg_corr_20") < 0.45,
    "above_ma200 (SPY off -$102)": pl.col("dist_ma200") > 0,
    "breadth_up (SPY noise)": pl.col("rspspy_ema5_20") > 0,
    "ret20d>0 (SPY good)": pl.col("ret_20d") > 0,
    "vixfall (vix_chg_20d<0)": pl.col("vix_chg_20d") < 0,
}
print("\n=== SPY-STUDY SIGNALS ON QQQ (on-mean | off-mean | freq) ===")
for name, e in SIGNALS.items():
    m = panel.select(e.alias("s"))["s"].to_numpy()
    ok = m != None  # noqa: E711
    mm = m.astype(bool) & ok
    on, off = pnl[mm], pnl[ok & ~mm]
    print(f"{name:28s} on n={len(on):3d} ${on.mean():7.1f} | "
          f"off n={len(off):3d} ${off.mean():7.1f} | freq {mm.mean():.2f}")

# ---- yearly to see if any period was sellable at all
print("\n=== YEARLY UNFILTERED ===")
for y in sorted(set(yr)):
    p = pnl[yr == y]
    print(f"  {y}: n={len(p):3d} mean=${p.mean():7.1f} win={(p > 0).mean():.2f}")
