"""Prioritized, configurable exit-rule evaluation.

Rules are evaluated in configured order against the *current snapshot only*
(no look-ahead): the same SpreadQuote used for the decision is the one the
execution model prices the exit against.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from xsp_research.config import ExitRuleConfig
from xsp_research.domain import ExitReason, SpreadPosition


@dataclass(frozen=True, slots=True)
class ExitContext:
    position: SpreadPosition
    ts: datetime
    cost_to_close_mid: float  # per spread, index points, clamped to [0, width]
    short_delta: float | None  # model delta of the short leg; None if IV unrecoverable
    dte: int


_KIND_TO_REASON = {
    "profit_target": ExitReason.PROFIT_TARGET,
    "stop_loss": ExitReason.STOP_LOSS,
    "dte": ExitReason.DTE,
    "delta_stop": ExitReason.DELTA_STOP,
}


def evaluate_exit(rules: list[ExitRuleConfig], ctx: ExitContext) -> ExitReason | None:
    """First triggered rule wins (list order = priority)."""
    credit = ctx.position.entry_credit
    for rule in rules:
        if rule.kind == "profit_target":
            # Closed profit fraction = (credit - cost_to_close)/credit
            if ctx.cost_to_close_mid <= (1.0 - rule.profit_target_pct) * credit:
                return ExitReason.PROFIT_TARGET
        elif rule.kind == "stop_loss":
            # Convention (documented): trigger when cost-to-close >= multiple * credit.
            if ctx.cost_to_close_mid >= rule.stop_loss_multiple * credit:
                return ExitReason.STOP_LOSS
        elif rule.kind == "dte":
            if ctx.dte <= rule.dte_threshold:
                return ExitReason.DTE
        elif rule.kind == "delta_stop":
            if ctx.short_delta is not None and ctx.short_delta >= rule.delta_threshold:
                return ExitReason.DELTA_STOP
        else:  # pragma: no cover - config validation prevents this
            raise ValueError(f"unknown exit rule {rule.kind}")
    return None
