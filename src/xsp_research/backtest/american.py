"""American-exercise modeling for physically-settled options (SPY).

Why this exists: SPY options are American-style and physically settled. For a
CALL-SPREAD SELLER the dominant American feature is dividend-driven early
assignment: on the session before an ex-dividend date, a short ITM call whose
remaining extrinsic value is below the dividend is rationally exercised by
its holder. Ignoring this understates losses in rally scenarios — exactly
where a bear call spread is already weakest.

Model (deterministic rational-exercise boundary, standard in the literature):

  triggered  <=>  S > K_short  and  extrinsic(short) < dividend
  where extrinsic = max(short_mid - (S - K_short), 0)

P&L of an assignment event, per spread (relative to the pre-event mark):
the short leg is effectively settled at intrinsic, the seller additionally
owes (dividend - extrinsic) [the value transferred to the exerciser], and
the remaining long leg is liquidated at its bid:

  exit_debit = max(S - K_short, 0) + max(div - extrinsic, 0) - long_bid
  (floored at 0)

Limitations (documented, revisit with position-level data if ever material):
- Assignment is modeled as certain when rational; real assignment is
  probabilistic near the boundary. The deterministic rule is the conservative
  choice for a seller.
- The long leg is sold at the same snapshot's bid (no overnight gap risk on
  the assigned shares — we model the unwind as immediate).
- Non-dividend early exercise of calls is never rational (no borrow costs
  modeled) and is ignored.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path

import polars as pl


@dataclass(frozen=True, slots=True)
class DividendCalendar:
    """Ex-dividend dates and per-share amounts for one underlying."""

    symbol: str
    amounts: dict[date, float]

    @classmethod
    def from_parquet(cls, path: str | Path, symbol: str = "SPY") -> DividendCalendar:
        df = pl.read_parquet(path)
        if not {"ex_date", "amount"} <= set(df.columns):
            raise ValueError(f"{path}: dividend file needs (ex_date, amount) columns")
        return cls(
            symbol=symbol,
            amounts=dict(zip(df["ex_date"].to_list(), df["amount"].to_list(), strict=True)),
        )

    def amount_on(self, ex_date: date) -> float | None:
        return self.amounts.get(ex_date)

    def next_ex_date_after(self, d: date) -> date | None:
        later = [x for x in self.amounts if x > d]
        return min(later) if later else None


def short_call_extrinsic(spot: float, short_strike: float, short_mid: float) -> float:
    return max(short_mid - max(spot - short_strike, 0.0), 0.0)


def early_assignment_triggered(
    spot: float, short_strike: float, short_mid: float, dividend: float
) -> bool:
    """Rational-exercise boundary on the eve of an ex-dividend date."""
    if dividend <= 0.0 or spot <= short_strike:
        return False
    return short_call_extrinsic(spot, short_strike, short_mid) < dividend


def assignment_exit_debit(
    spot: float,
    short_strike: float,
    short_mid: float,
    dividend: float,
    long_bid: float,
) -> float:
    """Per-spread cost of the assignment event (see module docstring)."""
    intrinsic = max(spot - short_strike, 0.0)
    penalty = max(dividend - short_call_extrinsic(spot, short_strike, short_mid), 0.0)
    return max(intrinsic + penalty - long_bid, 0.0)
