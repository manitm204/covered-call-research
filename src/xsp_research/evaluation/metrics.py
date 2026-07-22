"""Trade-level and equity-curve performance metrics.

Definitions are authoritative in docs/PLAN.md section 8. Gross and net figures
are reported separately; interest income is never mixed into trading P&L.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import polars as pl

TRADING_DAYS = 252


def equity_metrics(equity_curve: pl.DataFrame, rf_annual: float = 0.0) -> dict[str, Any]:
    """Metrics from a (date, equity, ...) daily curve."""
    eq = equity_curve["equity"].to_numpy()
    dates = equity_curve["date"].to_list()
    if len(eq) < 2:
        return {"error": "equity curve too short"}
    days = (dates[-1] - dates[0]).days or 1
    total_return = eq[-1] / eq[0] - 1.0
    cagr = (eq[-1] / eq[0]) ** (365.25 / days) - 1.0

    rets = np.diff(eq) / eq[:-1]
    rf_daily = (1.0 + rf_annual) ** (1.0 / TRADING_DAYS) - 1.0
    excess = rets - rf_daily
    vol = float(np.std(rets, ddof=1) * math.sqrt(TRADING_DAYS))
    sharpe = (
        float(np.mean(excess) / np.std(excess, ddof=1) * math.sqrt(TRADING_DAYS))
        if np.std(excess, ddof=1) > 0
        else float("nan")
    )
    downside = excess[excess < 0]
    sortino = (
        float(np.mean(excess) / np.std(downside, ddof=1) * math.sqrt(TRADING_DAYS))
        if len(downside) > 1 and np.std(downside, ddof=1) > 0
        else float("nan")
    )
    running_max = np.maximum.accumulate(eq)
    dd = eq / running_max - 1.0
    max_dd = float(dd.min())
    calmar = cagr / abs(max_dd) if max_dd < 0 else float("nan")

    return {
        "start": str(dates[0]),
        "end": str(dates[-1]),
        "total_return": total_return,
        "cagr": cagr,
        "annualized_vol": vol,
        "sharpe": sharpe,
        "sortino": sortino,
        "max_drawdown": max_dd,
        "calmar": calmar,
        "final_equity": float(eq[-1]),
    }


def tail_metrics(equity_curve: pl.DataFrame) -> dict[str, Any]:
    """Tail and calendar-bucket risk measures from a daily (date, equity) curve."""
    eq = equity_curve["equity"].to_numpy()
    if len(eq) < 20:
        return {"error": "too few observations for tail metrics"}
    rets = np.diff(eq) / eq[:-1]
    var_95 = float(np.quantile(rets, 0.05))
    cvar_95 = float(rets[rets <= var_95].mean()) if (rets <= var_95).any() else var_95
    monthly = (
        equity_curve.with_columns(pl.col("date").dt.strftime("%Y-%m").alias("month"))
        .group_by("month", maintain_order=True)
        .agg(pl.col("equity").last())
        .with_columns((pl.col("equity") / pl.col("equity").shift(1) - 1.0).alias("ret"))
        .drop_nulls("ret")
    )
    worst_day_idx = int(np.argmin(rets))
    return {
        "var_95_daily": var_95,
        "cvar_95_daily": cvar_95,
        "worst_day_return": float(rets.min()),
        "worst_day_date": str(equity_curve["date"][worst_day_idx + 1]),
        "worst_month_return": float(monthly["ret"].min()) if not monthly.is_empty() else None,
        "worst_month": (
            monthly.sort("ret").head(1)["month"][0] if not monthly.is_empty() else None
        ),
    }


def benchmark_relative_metrics(port_rets: np.ndarray, bench_rets: np.ndarray) -> dict[str, Any]:
    """Beta, downside beta, capture ratios, correlation vs a benchmark return series."""
    if len(port_rets) != len(bench_rets) or len(port_rets) < 20:
        return {"error": "return series unaligned or too short"}
    var_b = float(np.var(bench_rets, ddof=1))
    beta = float(np.cov(port_rets, bench_rets, ddof=1)[0, 1] / var_b) if var_b > 0 else None
    down = bench_rets < 0
    up = bench_rets > 0
    downside_beta = None
    if down.sum() >= 10 and np.var(bench_rets[down], ddof=1) > 0:
        downside_beta = float(
            np.cov(port_rets[down], bench_rets[down], ddof=1)[0, 1]
            / np.var(bench_rets[down], ddof=1)
        )
    up_capture = float(port_rets[up].mean() / bench_rets[up].mean()) if up.sum() >= 10 else None
    down_capture = (
        float(port_rets[down].mean() / bench_rets[down].mean()) if down.sum() >= 10 else None
    )
    corr = float(np.corrcoef(port_rets, bench_rets)[0, 1])
    return {
        "beta": beta,
        "downside_beta": downside_beta,
        "upside_capture": up_capture,
        "downside_capture": down_capture,
        "correlation": corr,
    }


def trade_metrics(trades: pl.DataFrame) -> dict[str, Any]:
    """Metrics from the per-trade record frame (realized_net etc.)."""
    if trades.is_empty():
        return {"n_trades": 0}
    net = trades["realized_net"].to_numpy()
    gross = trades["realized_gross"].to_numpy()
    wins, losses = net[net > 0], net[net <= 0]
    gross_profit, gross_loss = float(wins.sum()), float(-losses.sum())
    return {
        "n_trades": len(net),
        "win_rate": float(len(wins) / len(net)),
        "avg_win": float(wins.mean()) if len(wins) else 0.0,
        "avg_loss": float(losses.mean()) if len(losses) else 0.0,
        "profit_factor": gross_profit / gross_loss if gross_loss > 0 else float("inf"),
        "expectancy_net_per_spread": float(net.sum() / trades["qty"].sum()),
        "total_net_pnl": float(net.sum()),
        "total_gross_pnl": float(gross.sum()),
        "total_fees": float(trades["fees"].sum()),
        "fees_pct_of_gross_profit": (
            float(trades["fees"].sum() / gross_profit) if gross_profit > 0 else float("nan")
        ),
        "avg_return_on_max_risk": float(trades["return_on_max_risk"].mean()),
        "avg_pct_credit_captured": float(trades["pct_credit_captured"].mean()),
        "worst_trade_net": float(net.min()),
        "best_trade_net": float(net.max()),
        "avg_days_in_trade": float(trades["days_in_trade"].mean()),
        "avg_mae": float(trades["mae_dollars"].mean()),
        "avg_mfe": float(trades["mfe_dollars"].mean()),
        "exit_reason_counts": dict(trades.group_by("exit_reason").len().iter_rows()),
    }


def summarize(result) -> dict[str, Any]:
    """Combined summary for a BacktestResult, with honest data-source labeling."""
    out: dict[str, Any] = {
        "data_source": result.data_source,
        "execution_scenario": result.execution_scenario,
    }
    if result.data_source == "synthetic":
        out["WARNING"] = (
            "SYNTHETIC DATA - software validation only; numbers are NOT evidence "
            "of real-world strategy performance"
        )
    out["equity"] = equity_metrics(result.equity_curve)
    out["trades"] = trade_metrics(result.trades_frame())
    out["interest_income_total"] = result.ledger.interest_income()
    out["realized_trading_pnl_gross"] = result.ledger.realized_pnl_gross()
    out["total_fees"] = result.ledger.total_fees()
    out["entry_attempts"] = {
        "attempted": len(result.entry_attempts),
        "filled": sum(1 for a in result.entry_attempts if a.filled),
        "rejected": [
            {"session": str(a.session), "reason": a.reason}
            for a in result.entry_attempts
            if not a.filled
        ],
    }
    return out
