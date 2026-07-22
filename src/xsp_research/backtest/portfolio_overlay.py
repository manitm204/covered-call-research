"""Portfolio overlay: stock portfolio alone vs spread alone vs combined.

Capital model (documented, configurable):

- ``margin_overlay`` (default): the account's capital is fully invested in the
  stock portfolio; spread collateral is satisfied by margin against the stock
  holdings. The combined equity is therefore
      stock_equity(t) + cumulative overlay TRADING P&L (net of fees)
  with NO cash interest credited to the overlay — there is no idle cash sleeve.
  This isolates the true incremental effect of the spreads.

- ``carve_out``: a fixed fraction of initial capital is held as a cash+spread
  sleeve (earning the backtest's configured interest) and the remainder holds
  the stock portfolio. This models an investor who explicitly reserves
  collateral cash.

The overlay provides negative delta but its downside benefit is CAPPED at the
spread's max profit — it is not downside protection, and reports produced from
these frames must (and do) surface rally/rebound underperformance explicitly.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import polars as pl

from xsp_research.backtest.engine import BacktestResult
from xsp_research.evaluation.metrics import (
    benchmark_relative_metrics,
    equity_metrics,
    tail_metrics,
)


@dataclass(frozen=True, slots=True)
class OverlayConfig:
    mode: Literal["margin_overlay", "carve_out"] = "margin_overlay"
    carve_out_pct: float = 0.10  # only used in carve_out mode


def stock_returns_from_closes(closes: pl.DataFrame) -> pl.DataFrame:
    """(date, close) -> (date, stock_ret); first day's return is 0."""
    return (
        closes.sort("date")
        .with_columns((pl.col("close") / pl.col("close").shift(1) - 1.0).alias("stock_ret"))
        .with_columns(pl.col("stock_ret").fill_null(0.0))
        .select(["date", "stock_ret"])
    )


def overlay_trading_pnl(result: BacktestResult) -> pl.DataFrame:
    """Daily overlay TRADING P&L (net of fees, excluding cash interest).

    Derived from the engine equity curve: d(equity) - d(interest_cum). The
    first session's P&L is measured against initial capital.
    """
    eq = result.equity_curve
    initial = float(result.config_snapshot["backtest"]["initial_cash"])
    return (
        eq.select(["date", "equity", "interest_income_cum"])
        .with_columns(
            (
                pl.col("equity").diff().fill_null(pl.col("equity").first() - initial)
                - pl.col("interest_income_cum")
                .diff()
                .fill_null(pl.col("interest_income_cum").first())
            ).alias("overlay_pnl")
        )
        .select(["date", "overlay_pnl"])
    )


def build_overlay_frame(
    stock_closes: pl.DataFrame,
    result: BacktestResult,
    cfg: OverlayConfig | None = None,
) -> pl.DataFrame:
    """Join the three portfolios on common sessions.

    Columns: date, stock_equity, overlay_equity (standalone incl. interest),
    combined_equity, stock_ret, combined_ret.
    """
    cfg = cfg if cfg is not None else OverlayConfig()
    initial = float(result.config_snapshot["backtest"]["initial_cash"])
    rets = stock_returns_from_closes(stock_closes)
    pnl = overlay_trading_pnl(result)
    standalone = result.equity_curve.select(["date", pl.col("equity").alias("overlay_equity")])

    joined = (
        rets.join(pnl, on="date", how="inner").join(standalone, on="date", how="inner").sort("date")
    )
    if joined.height < min(pnl.height, rets.height):
        # Inner join dropped sessions: stock and options calendars disagree.
        dropped = max(pnl.height, rets.height) - joined.height
        if dropped > 0.05 * pnl.height:
            raise ValueError(
                f"stock and overlay calendars misaligned: {dropped} sessions dropped; "
                "supply a stock series covering the backtest sessions"
            )

    if cfg.mode == "margin_overlay":
        stock_base, sleeve_base = initial, 0.0
    else:
        sleeve_base = cfg.carve_out_pct * initial
        stock_base = initial - sleeve_base

    growth = (1.0 + joined["stock_ret"].to_numpy()).cumprod()
    stock_equity = stock_base * growth
    overlay_cum = joined["overlay_pnl"].to_numpy().cumsum()

    if cfg.mode == "margin_overlay":
        combined = stock_equity + overlay_cum
    else:
        # Sleeve holds the standalone overlay account scaled to the carve-out size.
        sleeve = joined["overlay_equity"].to_numpy() * (sleeve_base / initial)
        combined = stock_equity + sleeve

    out = joined.with_columns(
        pl.Series(
            "stock_equity", stock_base * growth if cfg.mode == "margin_overlay" else stock_equity
        ),
        pl.Series("combined_equity", combined),
    ).with_columns(
        (pl.col("combined_equity") / pl.col("combined_equity").shift(1) - 1.0)
        .fill_null(0.0)
        .alias("combined_ret")
    )
    return out.select(
        [
            "date",
            "stock_ret",
            "stock_equity",
            "overlay_pnl",
            "overlay_equity",
            "combined_equity",
            "combined_ret",
        ]
    )


def compare_portfolios(frame: pl.DataFrame, result: BacktestResult) -> dict[str, Any]:
    """Metric panel for stock-only, overlay-standalone, and combined portfolios."""

    def curve(col: str) -> pl.DataFrame:
        return frame.select(["date", pl.col(col).alias("equity")])

    stock_rets = frame["stock_ret"].to_numpy()[1:]
    combined_rets = frame["combined_ret"].to_numpy()[1:]

    stock_m = equity_metrics(curve("stock_equity")) | tail_metrics(curve("stock_equity"))
    combined_m = equity_metrics(curve("combined_equity")) | tail_metrics(curve("combined_equity"))
    overlay_m = equity_metrics(curve("overlay_equity")) | tail_metrics(curve("overlay_equity"))

    delta = {}
    for k in ("cagr", "sharpe", "sortino", "max_drawdown", "annualized_vol", "cvar_95_daily"):
        s, c = stock_m.get(k), combined_m.get(k)
        if isinstance(s, (int, float)) and isinstance(c, (int, float)):
            delta[k] = c - s

    # Rally behavior: overlay P&L on the stock's best days (capped-upside drag).
    up_days = frame.filter(pl.col("stock_ret") > 0.01)
    rally_drag = float(up_days["overlay_pnl"].sum()) if not up_days.is_empty() else 0.0

    return {
        "data_source": result.data_source,
        "mode_note": "see portfolio_overlay module docstring for the capital model",
        "stock_only": stock_m,
        "overlay_standalone": overlay_m,
        "combined": combined_m,
        "combined_minus_stock": delta,
        "combined_vs_stock_benchmark": benchmark_relative_metrics(combined_rets, stock_rets),
        "overlay_pnl_on_strong_up_days_total": rally_drag,
        "overlay_trading_pnl_total": float(frame["overlay_pnl"].sum()),
    }
