"""EXPLORATORY IWM regime scan (deliberately overfit — hypothesis generation only).

IWM is already burned as confirmation data (pre-registered test executed,
verdict WEAK/WEAK). Question here is purely descriptive: does IWM condition
on the same regimes as SPY/QQQ — overbought/extended + healthy internals
(low sector corr, falling absorption) good, stress regimes bad?

1. Unfiltered IWM panel: weekly, 0.15d, $3 wide, 30 DTE, hold, max 12 open
   (analog of the SPY 417-trade / QQQ 414-trade panels).
2. Join RUT/RVX features + market-wide regime v2 features.
3. Tercile scan of every numeric feature, ranked by hi-lo spread.
4. Per-signal table for the 11 candidate gate signals (incl. rsi70).
Output: results/iwm_prereg/regime_scan_panel.parquet + printed report.
"""

from __future__ import annotations

import sys

import numpy as np
import polars as pl

from xsp_research.backtest.american import DividendCalendar
from xsp_research.backtest.engine import BacktestEngine
from xsp_research.config import EXECUTION_SCENARIOS, load_strategy_config
from xsp_research.ingestion.file_provider import ParquetUnderlyingProvider, SeriesRatesProvider

sys.path.insert(0, "scripts")
from full_ablation_sweep import RATES, PreloadedOptionsProvider  # noqa: E402
from preregistered_iwm_test import (  # noqa: E402
    DIVIDENDS, FEATURES, OPTIONS_GLOB, SECTOR_CORR, UNDERLYING,
)

CONFIG = "configs/strategy_iwm_prereg_h1.yaml"

SIGNALS = {
    "rsi70":       ("rsi_14", ">", 70.0),
    "trend120":    ("ret_120d", ">", 0.15),
    "trend60":     ("ret_60d", ">", 0.08),
    "ma200_10":    ("dist_ma200", ">", 0.10),
    "ma50_up":     ("ma50_slope_20d", ">", 0.02),
    "beta_hi":     ("iwm_beta_spy_60", ">", 1.30),
    "absorb_fall": ("absorption_chg_20d", "<", 0.0),
    "seccorr_lo":  ("sector_avg_corr_20", "<", 0.45),
    "iv_rich":     ("iv_minus_rv20", ">", 0.0),
    "iwm_decor":   ("corr_spy_iwm_20", "<", 0.80),
    "rvx_lo":      ("vix_level", "<", 25.0),
}


def build_panel() -> pl.DataFrame:
    cfg0 = load_strategy_config(CONFIG)
    cfg = cfg0.model_copy(update={
        "execution": EXECUTION_SCENARIOS["base"],
        "sizing": cfg0.sizing.model_copy(update={
            "contracts_per_entry": 1, "max_open_positions": 12}),
    })
    options = PreloadedOptionsProvider(OPTIONS_GLOB, "IWM", cfg0.backtest.mark_time_et)
    und = ParquetUnderlyingProvider(UNDERLYING, "IWM")
    rates = SeriesRatesProvider(RATES)
    div = DividendCalendar.from_parquet(DIVIDENDS)
    res = BacktestEngine(
        cfg, options, und, rates, execution_scenario="base",
        entry_gate=None, dividends=div,
    ).run()
    return res.trades_frame().with_columns(
        (pl.col("realized_net") / pl.col("qty")).alias("pnl"),
        pl.col("entry_ts").dt.convert_time_zone("America/New_York")
        .dt.date().alias("entry_date"),
    )


t = build_panel()
print(f"panel: n={t.height} mean=${t['pnl'].mean():.2f} win={(t['pnl'] > 0).mean():.2f} "
      f"total=${t['pnl'].sum():.0f}")

feats = pl.read_parquet(FEATURES).join(
    pl.read_parquet(SECTOR_CORR), on="date", how="left"
)
panel = t.join(feats, left_on="entry_date", right_on="date", how="left")
panel.write_parquet("results/iwm_prereg/regime_scan_panel.parquet")
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
                 "spread": hi.mean() - lo.mean(), "n": int(ok.sum())})
sc = pl.DataFrame(rows).sort(pl.col("spread").abs(), descending=True)
print(f"\n=== TERCILE SCAN, top 20 of {sc.height} by |hi-lo| (panel mean ${pnl.mean():.1f}) ===")
print(f"{'feature':26s} {'lo':>8s} {'mid':>8s} {'hi':>8s} {'hi-lo':>8s}")
for r in sc.head(20).iter_rows(named=True):
    print(f"{r['feature']:26s} {r['lo']:8.1f} {r['mid']:8.1f} {r['hi']:8.1f} {r['spread']:8.1f}")

print(f"\n=== PER-SIGNAL (panel n={len(pnl)}) ===")
print(f"{'signal':12s} {'freq':>5s} | {'n_on':>4s} {'mean_on':>8s} {'win':>4s} "
      f"{'18-20':>7s} {'21-23':>7s} {'24-26':>7s} | {'mean_off':>8s}")
for name, (col, op, thr) in SIGNALS.items():
    v = panel[col].to_numpy()
    m = (v < thr) if op == "<" else (v > thr)
    m = m & np.isfinite(v)
    subs = []
    for a, b in [(2018, 2020), (2021, 2023), (2024, 2026)]:
        mm = m & (yr >= a) & (yr <= b)
        subs.append(pnl[mm].mean() if mm.sum() >= 5 else float("nan"))
    print(f"{name:12s} {m.mean():5.2f} | {m.sum():4d} {pnl[m].mean():8.1f} "
          f"{(pnl[m] > 0).mean():4.2f} {subs[0]:7.1f} {subs[1]:7.1f} {subs[2]:7.1f} "
          f"| {pnl[~m].mean():8.1f}")

print("\n=== YEARLY UNFILTERED ===")
for y in sorted(set(yr)):
    p = pnl[yr == y]
    print(f"  {y}: n={len(p):3d} mean=${p.mean():7.1f} win={(p > 0).mean():.2f}")
