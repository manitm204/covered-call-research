"""Double-entry ledger, collateral tracking, and interest accrual.

Accounting model (all amounts in dollars):

  Assets:      CASH, OPTIONS_MTM (negative = liability of open short spreads)
  Income:      REALIZED_PNL, UNREALIZED_PNL, INTEREST_INCOME  (credit-positive)
  Expense:     FEES  (debit-positive)

Every posting is a list of (account, signed_amount) that must sum to zero.
Sign convention: positive = debit (increases assets/expenses), negative =
credit (increases income). Equity = CASH + OPTIONS_MTM, and the trial balance
identity  assets == income - expenses + initial capital  is checked on demand.

Interest accrues on the configured eligible balance using ACT/365 simple
accrual over calendar days between trading sessions (weekends accrue).
"""

from __future__ import annotations

import enum
from dataclasses import dataclass
from datetime import date, datetime

from xsp_research.config import InterestConfig, InterestMode
from xsp_research.ingestion.base import RatesProvider

_TOL = 1e-6


class Account(enum.StrEnum):
    CASH = "cash"
    OPTIONS_MTM = "options_mtm"
    REALIZED_PNL = "realized_pnl"
    UNREALIZED_PNL = "unrealized_pnl"
    INTEREST_INCOME = "interest_income"
    FEES = "fees"
    CAPITAL = "capital"


@dataclass(frozen=True, slots=True)
class LedgerEntry:
    ts: datetime
    memo: str
    postings: tuple[tuple[Account, float], ...]
    trade_id: str | None = None


class Ledger:
    """Balanced double-entry ledger with per-trade attribution."""

    def __init__(self) -> None:
        self._entries: list[LedgerEntry] = []
        self._balances: dict[Account, float] = {a: 0.0 for a in Account}

    def post(
        self,
        ts: datetime,
        memo: str,
        postings: list[tuple[Account, float]],
        trade_id: str | None = None,
    ) -> None:
        total = sum(amount for _, amount in postings)
        if abs(total) > _TOL:
            raise ValueError(f"unbalanced posting ({total:+.6f}): {memo}")
        entry = LedgerEntry(ts=ts, memo=memo, postings=tuple(postings), trade_id=trade_id)
        self._entries.append(entry)
        for account, amount in postings:
            self._balances[account] += amount

    # ------------------------------------------------------------- convenience
    def deposit_capital(self, ts: datetime, amount: float) -> None:
        self.post(ts, "initial capital", [(Account.CASH, amount), (Account.CAPITAL, -amount)])

    def open_spread(self, ts: datetime, trade_id: str, credit_dollars: float, fees: float) -> None:
        """Receive credit; book the short spread at entry value as a liability."""
        self.post(
            ts,
            f"open {trade_id} credit {credit_dollars:.2f}",
            [(Account.CASH, credit_dollars), (Account.OPTIONS_MTM, -credit_dollars)],
            trade_id,
        )
        if fees:
            self.post(
                ts, f"open fees {trade_id}", [(Account.FEES, fees), (Account.CASH, -fees)], trade_id
            )

    def mark_spread(self, ts: datetime, trade_id: str, old_mtm: float, new_mtm: float) -> None:
        """Move the liability from -old to -new cost-to-close; delta hits UNREALIZED."""
        delta = new_mtm - old_mtm  # increase in cost-to-close = loss
        if abs(delta) < _TOL:
            return
        self.post(
            ts,
            f"mark {trade_id} {old_mtm:.2f}->{new_mtm:.2f}",
            [(Account.OPTIONS_MTM, -delta), (Account.UNREALIZED_PNL, delta)],
            trade_id,
        )

    def close_spread(
        self,
        ts: datetime,
        trade_id: str,
        entry_credit_dollars: float,
        current_mtm: float,
        exit_debit_dollars: float,
        fees: float,
        memo: str,
    ) -> None:
        """Pay the closing debit (or settlement), retire the liability, and move
        this trade's cumulative unrealized P&L into realized."""
        # 1. Final mark to the actual exit price.
        self.mark_spread(ts, trade_id, current_mtm, exit_debit_dollars)
        # 2. Pay cash, retire liability.
        postings = [(Account.OPTIONS_MTM, exit_debit_dollars)]
        if exit_debit_dollars:
            postings.append((Account.CASH, -exit_debit_dollars))
        else:
            postings = []
        if postings:
            self.post(ts, f"{memo} {trade_id} debit {exit_debit_dollars:.2f}", postings, trade_id)
        # 3. Reclassify unrealized -> realized for this trade.
        gross = entry_credit_dollars - exit_debit_dollars
        # Reverse the trade's accumulated unrealized credit and credit realized:
        # debit UNREALIZED (+gross), credit REALIZED (-gross).
        self.post(
            ts,
            f"realize {trade_id} gross {gross:+.2f}",
            [(Account.UNREALIZED_PNL, gross), (Account.REALIZED_PNL, -gross)],
            trade_id,
        )
        if fees:
            self.post(
                ts,
                f"close fees {trade_id}",
                [(Account.FEES, fees), (Account.CASH, -fees)],
                trade_id,
            )

    def accrue_interest(self, ts: datetime, amount: float) -> None:
        if abs(amount) < _TOL:
            return
        self.post(
            ts, "interest accrual", [(Account.CASH, amount), (Account.INTEREST_INCOME, -amount)]
        )

    # ------------------------------------------------------------------ views
    def balance(self, account: Account) -> float:
        return self._balances[account]

    @property
    def cash(self) -> float:
        return self._balances[Account.CASH]

    @property
    def equity(self) -> float:
        return self._balances[Account.CASH] + self._balances[Account.OPTIONS_MTM]

    @property
    def entries(self) -> list[LedgerEntry]:
        return list(self._entries)

    def trial_balance_ok(self) -> bool:
        """Assets = capital + income - expenses (all as positive magnitudes)."""
        assets = self._balances[Account.CASH] + self._balances[Account.OPTIONS_MTM]
        capital = -self._balances[Account.CAPITAL]
        income = -(
            self._balances[Account.REALIZED_PNL]
            + self._balances[Account.UNREALIZED_PNL]
            + self._balances[Account.INTEREST_INCOME]
        )
        expenses = self._balances[Account.FEES]
        return abs(assets - (capital + income - expenses)) < 1e-4

    def realized_pnl_gross(self) -> float:
        return -self._balances[Account.REALIZED_PNL]

    def interest_income(self) -> float:
        return -self._balances[Account.INTEREST_INCOME]

    def total_fees(self) -> float:
        return self._balances[Account.FEES]


class InterestAccruer:
    """ACT/365 simple interest on the configured eligible balance."""

    def __init__(self, cfg: InterestConfig, rates: RatesProvider) -> None:
        self._cfg = cfg
        self._rates = rates
        self._last_accrual: date | None = None

    def eligible_balance(self, cash: float, restricted_collateral: float) -> float:
        if self._cfg.mode is InterestMode.NONE:
            return 0.0
        if self._cfg.mode is InterestMode.FREE_CASH_ONLY:
            return max(cash - restricted_collateral, 0.0)
        return max(cash, 0.0)  # FREE_CASH_AND_COLLATERAL

    def accrue(self, as_of: date, cash: float, restricted_collateral: float) -> float:
        """Interest dollars for calendar days since last accrual (0 on first call)."""
        if self._last_accrual is None:
            self._last_accrual = as_of
            return 0.0
        days = (as_of - self._last_accrual).days
        self._last_accrual = as_of
        if days <= 0:
            return 0.0
        if self._cfg.rate_source == "fixed":
            rate = self._cfg.fixed_rate
        else:
            rate = self._rates.rate(as_of)
        rate += self._cfg.spread_bps / 10_000.0
        return self.eligible_balance(cash, restricted_collateral) * rate * days / 365.0
