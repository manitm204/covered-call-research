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
