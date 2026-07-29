"""Single-leg execution model: fills, fees, validity (research_plan.md §3).

Scenarios (buy side shown; sells mirror):
  optimistic:   mid
  base:         mid + 0.5 * half_spread
  conservative: ask + $0.01
Fees per contract per side: base/optimistic $0.70, conservative $1.00
(Fidelity $0.65 commission + regulatory; stress rounds up).
"""

from __future__ import annotations

from dataclasses import dataclass

SCENARIOS = ("optimistic", "base", "conservative", "stress")
FEES = {"optimistic": 0.70, "base": 0.70, "conservative": 1.00, "stress": 2.00}
TICK = 0.01
STRESS_EXTRA_TICKS = 2  # stress: cross the spread plus 3 ticks total, double fees


class InvalidQuote(Exception):
    pass


def quote_ok(bid: float, ask: float, *, for_sell: bool, bid_size: int = 1, ask_size: int = 1) -> bool:
    if ask <= 0 or bid > ask:
        return False
    if for_sell and bid <= 0:
        return False
    if (for_sell and bid_size < 1) or (not for_sell and ask_size < 1):
        return False
    return True


def fill_price(side: str, bid: float, ask: float, scenario: str) -> float:
    """Price per share (not per contract). side: 'buy'|'sell'."""
    if scenario not in SCENARIOS:
        raise ValueError(scenario)
    if bid > ask or ask <= 0:
        raise InvalidQuote(f"bid {bid} > ask {ask}")
    mid = (bid + ask) / 2.0
    half = (ask - bid) / 2.0
    if side == "buy":
        px = {"optimistic": mid, "base": mid + 0.5 * half, "conservative": ask + TICK,
              "stress": ask + TICK * (1 + STRESS_EXTRA_TICKS)}[scenario]
        return max(px, 0.01)
    if side == "sell":
        px = {"optimistic": mid, "base": mid - 0.5 * half, "conservative": max(bid - TICK, 0.0),
              "stress": max(bid - TICK * (1 + STRESS_EXTRA_TICKS), 0.0)}[scenario]
        return max(px, 0.0)
    raise ValueError(side)


@dataclass(frozen=True)
class Fill:
    price: float  # per share
    fee: float  # total dollars for the order

    @property
    def per_contract_cost(self) -> float:
        return self.price * 100.0


def execute(side: str, bid: float, ask: float, contracts: int, scenario: str) -> Fill:
    px = fill_price(side, bid, ask, scenario)
    return Fill(price=px, fee=FEES[scenario] * contracts)
