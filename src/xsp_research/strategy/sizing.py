"""Position sizing.

Phase 1: fixed contracts per entry, constrained by available buying power
measured against spread MAX LOSS collateral (never premium received).
Risk-budget and model-based sizing arrive in later phases.
"""

from __future__ import annotations

from xsp_research.config import SizingConfig
from xsp_research.domain import BearCallSpread


def size_entry(
    cfg: SizingConfig,
    spread: BearCallSpread,
    buying_power: float,
    open_positions: int,
) -> tuple[int, str]:
    """Return (quantity, reason). quantity=0 means no entry."""
    if open_positions >= cfg.max_open_positions:
        return 0, f"max open positions ({cfg.max_open_positions}) reached"
    collateral_per_spread = spread.width * spread.multiplier
    qty = min(cfg.contracts_per_entry, int(buying_power // collateral_per_spread))
    if qty <= 0:
        return 0, (
            f"insufficient buying power ${buying_power:,.2f} for collateral "
            f"${collateral_per_spread:,.2f}/spread"
        )
    return qty, "ok"
