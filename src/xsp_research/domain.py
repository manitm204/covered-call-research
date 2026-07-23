"""Validated core domain objects.

These are provider-agnostic. All prices are in index points unless a name says
dollars; dollar conversion always goes through ``contract.multiplier``.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from datetime import date, datetime


class OptionType(enum.StrEnum):
    CALL = "C"
    PUT = "P"


class ExerciseStyle(enum.StrEnum):
    EUROPEAN = "european"
    AMERICAN = "american"


class SettlementStyle(enum.StrEnum):
    AM = "AM"  # a.m. cash settlement (standard SPX 3rd Friday) — unsupported
    PM = "PM"  # p.m. cash settlement (XSP, SPXW)
    PHYSICAL = "physical"  # share delivery (SPY): engine force-closes before expiry


class PositionStatus(enum.StrEnum):
    OPEN = "open"
    CLOSED = "closed"  # closed early via trade
    EXPIRED = "expired"  # cash-settled at expiration


class ExitReason(enum.StrEnum):
    EXPIRY_SETTLEMENT = "expiry_settlement"  # cash settlement (European PM)
    EXPIRY_CLOSE = "expiry_close"  # forced close on expiry day (physical settlement)
    EARLY_ASSIGNMENT = "early_assignment"  # dividend-driven assignment (American)
    PROFIT_TARGET = "profit_target"
    STOP_LOSS = "stop_loss"
    DTE = "dte"
    DELTA_STOP = "delta_stop"
    END_OF_BACKTEST = "end_of_backtest"


@dataclass(frozen=True, slots=True)
class OptionContract:
    root: str
    expiration: date
    strike: float
    option_type: OptionType
    multiplier: int = 100
    exercise_style: ExerciseStyle = ExerciseStyle.EUROPEAN
    settlement: SettlementStyle = SettlementStyle.PM

    def __post_init__(self) -> None:
        if self.strike <= 0:
            raise ValueError("strike must be positive")

    @property
    def osi_symbol(self) -> str:
        return (
            f"{self.root}{self.expiration:%y%m%d}"
            f"{self.option_type.value}{int(round(self.strike * 1000)):08d}"
        )


@dataclass(frozen=True, slots=True)
class OptionQuote:
    contract: OptionContract
    ts: datetime
    bid: float
    ask: float
    bid_size: int | None = None
    ask_size: int | None = None
    volume: int | None = None
    open_interest: int | None = None
    underlying_price: float | None = None

    def __post_init__(self) -> None:
        if self.bid < 0 or self.ask < 0:
            raise ValueError(f"negative quote for {self.contract.osi_symbol}")

    @property
    def has_valid_market(self) -> bool:
        """Two-sided, uncrossed, unlocked market with a positive ask."""
        return self.ask > 0 and self.ask > self.bid >= 0

    @property
    def mid(self) -> float:
        return 0.5 * (self.bid + self.ask)

    @property
    def spread_abs(self) -> float:
        return self.ask - self.bid

    @property
    def rel_spread(self) -> float:
        """(ask-bid)/mid; inf when mid is 0 so filters reject it."""
        return self.spread_abs / self.mid if self.mid > 0 else float("inf")


@dataclass(frozen=True, slots=True)
class BearCallSpread:
    """Short the lower-strike call, long the higher-strike call. Same expiry."""

    short_leg: OptionContract
    long_leg: OptionContract

    def __post_init__(self) -> None:
        s, lo = self.short_leg, self.long_leg
        if s.option_type is not OptionType.CALL or lo.option_type is not OptionType.CALL:
            raise ValueError("bear call spread requires two calls")
        if s.expiration != lo.expiration:
            raise ValueError("legs must share an expiration")
        if s.root != lo.root or s.multiplier != lo.multiplier:
            raise ValueError("legs must share root and multiplier")
        if lo.strike <= s.strike:
            raise ValueError("long strike must exceed short strike")

    @property
    def width(self) -> float:
        return self.long_leg.strike - self.short_leg.strike

    @property
    def expiration(self) -> date:
        return self.short_leg.expiration

    @property
    def multiplier(self) -> int:
        return self.short_leg.multiplier

    def settlement_value(self, settlement_price: float) -> float:
        """Intrinsic value owed by the spread seller at expiration, in [0, width]."""
        s = max(settlement_price - self.short_leg.strike, 0.0)
        lo = max(settlement_price - self.long_leg.strike, 0.0)
        return s - lo

    def max_profit_dollars(self, credit: float, qty: int) -> float:
        return credit * self.multiplier * qty

    def max_loss_dollars(self, credit: float, qty: int) -> float:
        return (self.width - credit) * self.multiplier * qty


@dataclass(frozen=True, slots=True)
class SpreadQuote:
    """Contemporaneous quotes for both legs."""

    short_quote: OptionQuote
    long_quote: OptionQuote

    def __post_init__(self) -> None:
        # Legs may have slightly different last-update times within a snapshot,
        # but must come from the same session (guards against stale-day mixing).
        if self.short_quote.ts.date() != self.long_quote.ts.date():
            raise ValueError("leg quotes must be from the same session")

    @property
    def ts(self) -> datetime:
        return max(self.short_quote.ts, self.long_quote.ts)

    @property
    def spread(self) -> BearCallSpread:
        return BearCallSpread(self.short_quote.contract, self.long_quote.contract)

    # --- opening (sell short leg, buy long leg) ---
    @property
    def credit_natural(self) -> float:
        return self.short_quote.bid - self.long_quote.ask

    @property
    def credit_mid(self) -> float:
        return self.short_quote.mid - self.long_quote.mid

    # --- closing (buy short leg back, sell long leg) ---
    @property
    def close_debit_natural(self) -> float:
        return self.short_quote.ask - self.long_quote.bid

    @property
    def close_debit_mid(self) -> float:
        return self.short_quote.mid - self.long_quote.mid


@dataclass(frozen=True, slots=True)
class Fill:
    ts: datetime
    price: float  # per spread, index points; credit for opens, debit for closes
    qty: int
    fees_dollars: float


@dataclass(slots=True)
class SpreadPosition:
    """One open/closed bear call spread position with its audit trail."""

    position_id: str
    spread: BearCallSpread
    qty: int
    entry_fill: Fill
    status: PositionStatus = PositionStatus.OPEN
    exit_fill: Fill | None = None
    exit_reason: ExitReason | None = None
    # Mark history (snapshot-limited; see PLAN.md assumption 2)
    mfe_dollars: float = 0.0  # max favorable excursion of unrealized P&L (gross)
    mae_dollars: float = 0.0  # max adverse excursion (<= 0)
    last_mark: float | None = None  # last cost-to-close per spread (mid)
    marks: list[tuple[datetime, float]] = field(default_factory=list)

    @property
    def entry_credit(self) -> float:
        return self.entry_fill.price

    def record_mark(self, ts: datetime, cost_to_close: float) -> None:
        """Update mark, clamped to the arbitrage-free band [0, width]."""
        m = min(max(cost_to_close, 0.0), self.spread.width)
        self.marks.append((ts, m))
        self.last_mark = m
        unreal = (self.entry_credit - m) * self.spread.multiplier * self.qty
        self.mfe_dollars = max(self.mfe_dollars, unreal)
        self.mae_dollars = min(self.mae_dollars, unreal)

    def unrealized_dollars(self) -> float:
        if self.status is not PositionStatus.OPEN or self.last_mark is None:
            return 0.0
        return (self.entry_credit - self.last_mark) * self.spread.multiplier * self.qty

    def realized_gross_dollars(self) -> float:
        if self.exit_fill is None:
            return 0.0
        return (self.entry_credit - self.exit_fill.price) * self.spread.multiplier * self.qty

    def realized_net_dollars(self) -> float:
        if self.exit_fill is None:
            return 0.0
        return (
            self.realized_gross_dollars()
            - self.entry_fill.fees_dollars
            - self.exit_fill.fees_dollars
        )

    def collateral_dollars(self) -> float:
        """Restricted while open: full width (Reg-T style; see PLAN.md assumption 4)."""
        if self.status is not PositionStatus.OPEN:
            return 0.0
        return self.spread.width * self.spread.multiplier * self.qty

    def days_in_trade(self) -> int:
        if self.exit_fill is None:
            return 0
        return (self.exit_fill.ts.date() - self.entry_fill.ts.date()).days
