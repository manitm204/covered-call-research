"""Execution model: fill pricing, transaction costs, and rejections.

Nothing here ever fills at a price better than the quoted market allows under
the configured model. Rejected executions return an ExecutionReport with
filled=False and a reason — the engine records these, it does not retry with
looser assumptions.
"""

from __future__ import annotations

from dataclasses import dataclass

from xsp_research.config import CostConfig, ExecutionConfig, FillModel
from xsp_research.domain import SpreadQuote


@dataclass(frozen=True, slots=True)
class ExecutionReport:
    filled: bool
    price: float | None  # per spread, index points; credit (open) or debit (close)
    fees_dollars: float
    reason: str


def _leg_quality_ok(sq: SpreadQuote, cfg: ExecutionConfig) -> str | None:
    for name, q in (("short", sq.short_quote), ("long", sq.long_quote)):
        if not q.has_valid_market:
            return f"{name} leg market invalid (crossed/locked/one-sided)"
        if q.rel_spread > cfg.max_leg_rel_spread:
            return f"{name} leg rel spread {q.rel_spread:.2f} > max {cfg.max_leg_rel_spread:.2f}"
    return None


def open_spread(
    sq: SpreadQuote, qty: int, cfg: ExecutionConfig, costs: CostConfig
) -> ExecutionReport:
    """Sell to open: credit between natural and mid per the fill model."""
    problem = _leg_quality_ok(sq, cfg)
    if problem:
        return ExecutionReport(False, None, 0.0, problem)

    natural, mid = sq.credit_natural, sq.credit_mid
    if cfg.fill_model is FillModel.NATURAL:
        price = natural
    elif cfg.fill_model is FillModel.MIDPOINT:
        price = mid
    else:
        price = natural + cfg.fill_pct * (mid - natural)
    price -= cfg.slippage_per_spread  # slippage reduces credit received

    if price < cfg.min_credit:
        return ExecutionReport(
            False, None, 0.0, f"credit {price:.2f} below minimum {cfg.min_credit:.2f}"
        )
    fees = costs.per_spread() * qty
    return ExecutionReport(True, round(price, 4), fees, "filled")


def close_spread(
    sq: SpreadQuote, qty: int, cfg: ExecutionConfig, costs: CostConfig
) -> ExecutionReport:
    """Buy to close: debit between natural and mid per the fill model."""
    problem = _leg_quality_ok(sq, cfg)
    if problem:
        return ExecutionReport(False, None, 0.0, problem)

    natural, mid = sq.close_debit_natural, sq.close_debit_mid
    if cfg.fill_model is FillModel.NATURAL:
        price = natural
    elif cfg.fill_model is FillModel.MIDPOINT:
        price = mid
    else:
        price = natural + cfg.fill_pct * (mid - natural)
    price += cfg.slippage_per_spread  # slippage increases debit paid
    price = max(price, 0.0)

    fees = costs.per_spread() * qty
    return ExecutionReport(True, round(price, 4), fees, "filled")


def settlement_fees(qty: int, costs: CostConfig) -> float:
    """Fees on cash settlement at expiration (both legs)."""
    return 2.0 * costs.settlement_fee_per_contract * qty
