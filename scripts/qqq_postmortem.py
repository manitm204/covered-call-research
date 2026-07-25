"""EXPLORATORY POSTMORTEM of the failed pre-registered QQQ test.

This does NOT alter the NO-GO verdict (results/qqq_prereg/verdict.json).
It regenerates the identical deterministic trade list for diagnosis and asks:
  1. Did the gate provide ANY lift over unfiltered QQQ selling?
  2. Which gate votes carried each trade — do trades admitted by the
     S&P-specific confirmations (sector-corr, absorption) do worse than
     trades with the NDX-native confirmation (VXN>RV) on?
  3. What was the 2026 disaster (yearly mean -$441)?
  4. On shared entry dates, did SPY win while QQQ lost (instrument-specific
     failure) or were the QQQ entry dates different ones entirely?
  5. How big were the NDX moves that killed the losers?
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
from preregistered_qqq_test import (  # noqa: E402
    CONFIG, DIVIDENDS, FEATURES, OPTIONS_GLOB, SECTOR_CORR, UNDERLYING, make_gate,
)

def run(gate, max_open):
    cfg0 = load_strategy_config(CONFIG)
    cfg = cfg0.model_copy(update={
        "execution": EXECUTION_SCENARIOS["base"],
        "sizing": cfg0.sizing.model_copy(update={
            "contracts_per_entry": 1, "max_open_positions": max_open}),
    })
    res = BacktestEngine(
        cfg, OPTIONS, UND, RATESP, execution_scenario="base",
        entry_gate=gate, dividends=DIV,
    ).run()
    t = res.trades_frame()
    return t.with_columns(
        (pl.col("realized_net") / pl.col("qty")).alias("pnl"),
        pl.col("entry_ts").dt.convert_time_zone("America/New_York")
        .dt.date().alias("entry_date"),
    )


feats = pl.read_parquet(FEATURES).join(
    pl.read_parquet(SECTOR_CORR).select(["date", "sector_avg_corr_20"]),
    on="date", how="left",
)
cfg0 = load_strategy_config(CONFIG)
OPTIONS = PreloadedOptionsProvider(OPTIONS_GLOB, "QQQ", cfg0.backtest.mark_time_et)
UND = ParquetUnderlyingProvider(UNDERLYING, "QQQ")
RATESP = SeriesRatesProvider(RATES)
DIV = DividendCalendar.from_parquet(DIVIDENDS)

gated = run(make_gate(feats), 2)
unf = run(None, 12)
print(f"[1] GATE LIFT: gated n={gated.height} mean=${gated['pnl'].mean():.2f} | "
      f"unfiltered n={unf.height} mean=${unf['pnl'].mean():.2f} "
      f"win={(unf['pnl'] > 0).mean():.2f} total=${unf['pnl'].sum():.0f}")

# ---- votes per trade
g = gated.join(feats, left_on="entry_date", right_on="date", how="left").with_columns(
    (pl.col("iv_minus_rv20") > 0).alias("v_iv"),
    (pl.col("sector_avg_corr_20") < 0.45).alias("v_cor"),
    (pl.col("absorption_chg_20d") < 0).alias("v_abs"),
)
print("\n[2] P&L BY VOTE PATTERN (iv=NDX-native, cor/abs=S&P-specific):")
for pat, grp in sorted(g.group_by(["v_iv", "v_cor", "v_abs"]),
                       key=lambda x: -len(x[1])):
    p = grp["pnl"].to_numpy()
    print(f"  iv={int(pat[0])} cor={int(pat[1])} abs={int(pat[2])}: "
          f"n={len(p):2d} mean=${p.mean():8.2f} win={(p > 0).mean():.2f}")
for v, label in [("v_iv", "VXN>RV (NDX-native)"), ("v_cor", "sector-corr<0.45 (S&P)"),
                 ("v_abs", "absorption falling (S&P)")]:
    on = g.filter(pl.col(v))["pnl"]
    off = g.filter(~pl.col(v))["pnl"]
    print(f"  {label:26s}: on n={len(on):2d} ${on.mean():7.2f} | "
          f"off n={len(off):2d} ${off.mean() if len(off) else float('nan'):7.2f}")

# ---- 2026 disaster
print("\n[3] 2026 TRADES:")
for r in g.filter(pl.col("entry_date").dt.year() == 2026).sort("entry_date").iter_rows(named=True):
    print(f"  {r['entry_date']} K={r['short_strike']:g}/{r['long_strike']:g} "
          f"exp={r['expiration']} credit=${r['entry_credit']:.2f} pnl=${r['pnl']:8.2f} "
          f"exit={r['exit_reason']}")

# ---- SPY comparison on shared dates
spy = pl.read_parquet("results/ablation_full/signal_study_panel.parquet").with_columns(
    ((pl.col("rsi_14") > 70)
     & ((pl.col("iv_minus_rv20") > 0).cast(pl.Int8)
        + (pl.col("sector_avg_corr_20") < 0.45).cast(pl.Int8)
        + (pl.col("absorption_chg_20d") < 0).cast(pl.Int8) >= 2)).alias("spy_gated")
)
spy_g = spy.filter(pl.col("spy_gated")).select(
    pl.col("entry_date"), pl.col("pnl").alias("spy_pnl"))
qd = set(g["entry_date"].to_list())
sd = set(spy_g["entry_date"].to_list())
print(f"\n[4] DATE OVERLAP: QQQ gated dates={len(qd)}, SPY gated dates={len(sd)}, "
      f"shared={len(qd & sd)}")
j = g.select(["entry_date", "pnl"]).join(spy_g, on="entry_date", how="inner")
if j.height:
    a, b = j["pnl"].to_numpy(), j["spy_pnl"].to_numpy()
    print(f"  shared dates n={j.height}: QQQ mean=${a.mean():.2f} | SPY mean=${b.mean():.2f} "
          f"| corr={np.corrcoef(a, b)[0, 1]:.2f}")
    both_neg = ((a < 0) & (b < 0)).sum()
    q_only = ((a < 0) & (b >= 0)).sum()
    print(f"  losses: both lost={both_neg}, QQQ-only lost={q_only}, "
          f"SPY-only lost={((a >= 0) & (b < 0)).sum()}")

# ---- underlying move behind losers
closes = {r["date"]: r["close"] for r in pl.read_parquet(UNDERLYING).iter_rows(named=True)}
rows = []
for r in g.iter_rows(named=True):
    s0, s1 = closes.get(r["entry_date"]), closes.get(r["expiration"])
    if s0 and s1:
        rows.append({"pnl": r["pnl"], "ret": s1 / s0 - 1,
                     "breach": s1 > r["short_strike"]})
m = pl.DataFrame(rows)
lose = m.filter(pl.col("pnl") < 0)
winr = m.filter(pl.col("pnl") >= 0)
print(f"\n[5] NDX MOVE ENTRY->EXPIRY: losers n={lose.height} "
      f"avg {lose['ret'].mean():+.1%} (breach rate {lose['breach'].mean():.0%}) | "
      f"winners n={winr.height} avg {winr['ret'].mean():+.1%}")
print(f"  QQQ 30d ret needed to breach 0.15d short strike; "
      f"losing rallies: {sorted([f'{x:+.1%}' for x in lose['ret'].to_list()])}")
