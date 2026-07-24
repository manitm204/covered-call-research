"""Portfolio-level comparison of the final rule vs holding SPY.

Portfolios (daily, 2018-08 -> 2026-07, $100k base):
  TBILL      — 4w T-bill roll (ACT/365 calendar-day accrual)
  SPY        — buy & hold, dividends reinvested on ex-date
  STRAT      — the frozen rule standalone: cap-2 overlay, idle cash in T-bills
               (this is the engine's own ledger: interest on free cash only,
               collateral earns nothing)
  SPY+STRAT  — SPY buy & hold plus the overlay's trading P&L (margin-overlay
               assumption: spread collateral satisfied against the SPY
               position, no cash carve-out)

Stats: CAGR, ann vol, Sharpe & Sortino (excess over T-bill), max drawdown,
beta/alpha vs SPY, correlation, worst day. Also episode economics of the
overlay in isolation.
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
from full_ablation_sweep import (  # noqa: E402
    BASE_CONFIG, DIVIDENDS, OPTIONS_GLOB, RATES, UNDERLYING, PreloadedOptionsProvider,
)
from phaseA_aggression_frontier import make_final_gate  # noqa: E402

EXP_FEATURES = (
    "reports/experiments/spy_regime_rsi70_ivrich-20260723-204901-b6454793/features.parquet"
)
REGIME_V2 = "results/ablation_full/regime_features_v2.parquet"
CAPITAL = 100_000.0


def ann_stats(values: np.ndarray, tbill_ret: np.ndarray, spy_ret: np.ndarray) -> dict:
    ret = np.diff(values) / values[:-1]
    n_yrs = len(ret) / 252.0
    cagr = (values[-1] / values[0]) ** (1 / n_yrs) - 1
    vol = ret.std(ddof=1) * np.sqrt(252)
    excess = ret - tbill_ret
    sharpe = excess.mean() / excess.std(ddof=1) * np.sqrt(252) if excess.std() > 0 else np.nan
    downside = excess[excess < 0]
    sortino = (
        excess.mean() * 252 / (downside.std(ddof=1) * np.sqrt(252))
        if len(downside) > 1 and downside.std() > 0 else np.nan
    )
    peak = np.maximum.accumulate(values)
    max_dd = float(((values - peak) / peak).min())
    var_spy = spy_ret.var(ddof=1)
    beta = float(np.cov(ret, spy_ret, ddof=1)[0, 1] / var_spy) if var_spy > 0 else np.nan
    alpha_ann = (ret.mean() - tbill_ret.mean() - beta * (spy_ret.mean() - tbill_ret.mean())) * 252
    corr = float(np.corrcoef(ret, spy_ret)[0, 1]) if ret.std() > 0 else np.nan
    return {
        "CAGR": cagr, "vol": vol, "Sharpe": sharpe, "Sortino": sortino,
        "maxDD": max_dd, "beta": beta, "alpha_ann": alpha_ann, "corr_spy": corr,
        "worst_day": float(ret.min()), "final": values[-1],
    }


def main() -> int:
    feats = pl.read_parquet(EXP_FEATURES).join(
        pl.read_parquet(REGIME_V2), on="date", how="left"
    )
    gate = make_final_gate(feats)
    base = load_strategy_config(BASE_CONFIG)
    options = PreloadedOptionsProvider(OPTIONS_GLOB, "SPY", base.backtest.mark_time_et)
    und = ParquetUnderlyingProvider(UNDERLYING, "SPY")
    rates = SeriesRatesProvider(RATES)

    cfg = base.model_copy(update={
        "name": "portfolio_comparison",
        "execution": EXECUTION_SCENARIOS["base"],
        "exits": [],
        "selection": base.selection.model_copy(update={
            "short_delta_target": 0.15, "fixed_width": 8.0,
            "target_dte": 30, "min_dte": 25, "max_dte": 35,
        }),
        "sizing": base.sizing.model_copy(update={
            "contracts_per_entry": 1, "max_open_positions": 2,
        }),
    })
    res = BacktestEngine(
        cfg, options, und, rates, execution_scenario="base",
        entry_gate=gate, dividends=DividendCalendar.from_parquet(DIVIDENDS),
    ).run()
    eq = res.equity_curve
    print("equity curve columns:", eq.columns, flush=True)
    date_col = "date" if "date" in eq.columns else eq.columns[0]
    eq = eq.select(pl.col(date_col).alias("date"), pl.col("equity")).sort("date")
    dates = eq["date"].to_list()
    strat_eq = eq["equity"].to_numpy()

    # ---- SPY total return on the same dates
    closes = {r["date"]: r["close"] for r in
              pl.read_parquet(UNDERLYING).iter_rows(named=True)}
    divs = {r["ex_date"]: r["amount"] for r in
            pl.read_parquet(DIVIDENDS).iter_rows(named=True)}
    spy_close = np.array([closes[d] for d in dates])
    shares = CAPITAL / spy_close[0]
    spy_val = np.empty(len(dates))
    for i, d in enumerate(dates):
        if i and d in divs:
            shares *= 1 + divs[d] / spy_close[i]
        spy_val[i] = shares * spy_close[i]

    # ---- T-bill roll (calendar-day ACT/365, same convention as the ledger)
    tb = np.empty(len(dates))
    tb[0] = CAPITAL
    for i in range(1, len(dates)):
        gap = (dates[i] - dates[i - 1]).days
        tb[i] = tb[i - 1] * (1 + rates.rate(dates[i - 1]) * gap / 365.0)

    # ---- overlay trading P&L and the combined portfolio
    # net-of-drag: engine equity minus pure T-bill growth (cash collateral
    # forfeits interest while spreads are open)
    trading_pnl = strat_eq - tb
    # pure trading P&L (no interest opportunity cost): what a margin overlay
    # adds when collateral is satisfied against the SPY position
    interest_cum = res.equity_curve.sort(date_col)["interest_income_cum"].to_numpy()
    trading_pnl_pure = strat_eq - CAPITAL - interest_cum
    combo = spy_val + trading_pnl_pure

    tbill_ret = np.diff(tb) / tb[:-1]
    spy_ret = np.diff(spy_val) / spy_val[:-1]

    print(f"\nrange {dates[0]} -> {dates[-1]} ({len(dates)} sessions), base ${CAPITAL:,.0f}")
    print(f"overlay: {res.trades_frame().height} trades, "
          f"pure trading P&L ${trading_pnl_pure[-1]:,.0f}; "
          f"net of cash-collateral interest drag ${trading_pnl[-1]:,.0f} "
          f"(drag ${trading_pnl_pure[-1] - trading_pnl[-1]:,.0f})")

    rows = {
        "T-bills": tb, "SPY (TR)": spy_val,
        "Strategy standalone": strat_eq, "SPY + overlay": combo,
    }
    hdr = ["portfolio", "CAGR", "vol", "Sharpe", "Sortino", "maxDD",
           "beta", "alpha", "corr", "worstDay", "final $"]
    print(f"\n{hdr[0]:20s}" + "".join(f"{h:>9s}" for h in hdr[1:]))
    out = {}
    for name, v in rows.items():
        s = ann_stats(v, tbill_ret, spy_ret)
        out[name] = s
        print(f"{name:20s}{s['CAGR']:9.2%}{s['vol']:9.2%}{s['Sharpe']:9.2f}"
              f"{s['Sortino']:9.2f}{s['maxDD']:9.2%}{s['beta']:9.3f}"
              f"{s['alpha_ann']:9.2%}{s['corr_spy']:9.2f}{s['worst_day']:9.2%}"
              f"{s['final']:>9,.0f}")

    # ---- overlay in isolation: same dollar P&L on dedicated small capital
    print("\noverlay economics, independent of base capital:")
    yrs = len(spy_ret) / 252.0
    pure_ann = trading_pnl_pure[-1] / yrs
    net_ann = trading_pnl[-1] / yrs
    dd_overlay = float(
        (trading_pnl_pure - np.maximum.accumulate(trading_pnl_pure)).min()
    )
    print(f"  margin overlay (no cash carve-out): ${pure_ann:,.0f}/yr per 2-spread cap")
    print(f"  cash-collateralized (T-bill opportunity cost): ${net_ann:,.0f}/yr")
    print(f"  worst cumulative-P&L drawdown: ${dd_overlay:,.0f}")
    for cap in (5_000, 10_000):
        print(f"  on ${cap:,} dedicated cash: T-bills + {net_ann / cap:.2%}/yr; "
              f"max overlay DD {dd_overlay / cap:.2%}")

    pl.DataFrame({
        "date": dates, "tbill": tb, "spy_tr": spy_val,
        "strategy": strat_eq, "spy_plus_overlay": combo,
        "overlay_pnl_net_of_drag": trading_pnl, "overlay_pnl_pure": trading_pnl_pure,
    }).write_parquet("results/ablation_full/portfolio_curves.parquet")
    print("\nwrote results/ablation_full/portfolio_curves.parquet")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
