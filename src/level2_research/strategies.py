"""Strategy implementations for the hypothesis set (research_plan.md §2).

All strategies: whole contracts, premium budgets as % of current equity, forced
exit before expiry week, one position per slot, no averaging down.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date

import pandas as pd

from .engine import (BuyToOpen, Ctx, Order, SellCallToOpen, SellPutToOpen,
                     SellToClose, SellStock)
from .selection import pick_by_delta
from .signals import asof


def _equity_now(ctx: Ctx) -> float:
    # cash + stock marked at spot; open option marks unavailable here cheaply —
    # premium budgets use cash+stock which is conservative for sizing
    return ctx.account.cash + ctx.account.stock.shares * ctx.spot


def _contracts_for_budget(budget: float, price_per_share: float, fee: float) -> int:
    if price_per_share <= 0:
        return 0
    return max(int(budget // (price_per_share * 100 + fee)), 0)


@dataclass
class H1RebounCalls:
    """Buy calls when emerging from a deep drawdown: deep_dd_60 <= -dd_thresh and
    ret5 >= confirm. Time-boxed exit after hold_sessions; 5-DTE hard exit."""

    signals: pd.DataFrame
    dd_thresh: float = 0.15
    confirm_ret5: float = 0.02
    target_delta: float = 0.50
    dte_target: int = 45
    dte_lo: int = 35
    dte_hi: int = 65
    hold_days: int = 60  # calendar days ≈ 42 sessions
    budget_pct: float = 0.05
    exit_dte: int = 5
    cooldown_days: int = 7
    _last_exit: date | None = None

    def on_session(self, ctx: Ctx) -> list[Order]:
        orders: list[Order] = []
        acct = ctx.account
        for pid, p in list(acct.long_options.items()):
            age = (ctx.session - p.entry_session).days
            dte = (p.contract.expiration - ctx.session).days
            if age >= self.hold_days or dte <= self.exit_dte:
                orders.append(SellToClose(pid, tag="time_exit" if age >= self.hold_days else "dte_exit"))
                self._last_exit = ctx.session
        s = asof(self.signals, ctx.session)
        if s is None or acct.long_options:
            return orders
        if self._last_exit is not None and (ctx.session - self._last_exit).days < self.cooldown_days:
            return orders
        if not (s["deep_dd_60"] <= -self.dd_thresh and s["ret5"] >= self.confirm_ret5):
            return orders
        c = pick_by_delta(ctx.chain, ctx.spot, ctx.session, ctx.rate, option_type="C",
                          target_delta=self.target_delta, dte_target=self.dte_target,
                          dte_lo=self.dte_lo, dte_hi=self.dte_hi)
        if c is None:
            return orders
        n = _contracts_for_budget(self.budget_pct * _equity_now(ctx), c.ask, 0.70)
        if n >= 1:
            orders.append(BuyToOpen(c, n, tag="h1_entry"))
        return orders


@dataclass
class H2TrendCalls:
    """Hold a call while the trend filter is on (hysteresis bands); roll at
    roll_dte; sell everything when the filter turns off."""

    signals: pd.DataFrame
    target_delta: float = 0.60
    dte_target: int = 60
    dte_lo: int = 40
    dte_hi: int = 70
    roll_dte: int = 21
    budget_pct: float = 0.08
    max_positions: int = 1
    band: float = 0.01  # hysteresis band around the 200d MA
    extra_lag: int = 0  # additional sessions of signal delay (timing robustness)
    affordable_floor: float | None = None  # e.g. 0.30: walk delta down to fit budget
    use_brake: bool = False  # spec §4 drawdown brake (budget x ctx.brake_mult)

    def on_session(self, ctx: Ctx) -> list[Order]:
        orders: list[Order] = []
        acct = ctx.account
        sig = self.signals
        if self.extra_lag:
            ts = pd.Timestamp(ctx.session)
            prior = sig.index[sig.index < ts]
            if len(prior) <= self.extra_lag:
                return orders
            s = sig.loc[prior[-self.extra_lag]]
        else:
            s = asof(sig, ctx.session)
        if s is None or s[["px", "ma200"]].isna().any():
            return orders
        trend_on_entry = bool(s["px"] > s["ma200"] * (1 + self.band))
        trend_on_hold = bool(s["px"] > s["ma200"] * (1 - self.band))
        if not trend_on_hold:
            for pid in list(acct.long_options):
                orders.append(SellToClose(pid, tag="trend_off"))
            return orders
        for pid, p in list(acct.long_options.items()):
            dte = (p.contract.expiration - ctx.session).days
            if dte <= self.roll_dte:
                orders.append(SellToClose(pid, tag="roll"))
        held_after = [pid for pid in acct.long_options
                      if (acct.long_options[pid].contract.expiration - ctx.session).days > self.roll_dte]
        if len(held_after) >= self.max_positions or not trend_on_entry:
            return orders
        budget = self.budget_pct * _equity_now(ctx)
        if self.use_brake:
            budget *= ctx.brake_mult
            if budget <= 0:
                return orders
        if self.affordable_floor is not None:
            from .selection import pick_affordable_call
            c = pick_affordable_call(ctx.chain, ctx.spot, ctx.session, ctx.rate,
                                     budget=budget, fee=0.70,
                                     delta_hi=self.target_delta + 0.12,
                                     delta_lo=self.affordable_floor,
                                     dte_target=self.dte_target, dte_lo=self.dte_lo,
                                     dte_hi=self.dte_hi, prefer_delta=self.target_delta)
        else:
            c = pick_by_delta(ctx.chain, ctx.spot, ctx.session, ctx.rate, option_type="C",
                              target_delta=self.target_delta, dte_target=self.dte_target,
                              dte_lo=self.dte_lo, dte_hi=self.dte_hi)
        if c is None:
            return orders
        n = _contracts_for_budget(budget, c.ask, 0.70)
        if n >= 1:
            orders.append(BuyToOpen(c, n, tag="h2_entry"))
        return orders


@dataclass
class AlwaysAtmCallBenchmark:
    """Unoptimized family benchmark: roll a ~30 DTE ATM call monthly, fixed budget."""

    target_delta: float = 0.50
    dte_target: int = 30
    dte_lo: int = 21
    dte_hi: int = 45
    roll_dte: int = 7
    budget_pct: float = 0.05

    def on_session(self, ctx: Ctx) -> list[Order]:
        orders: list[Order] = []
        acct = ctx.account
        for pid, p in list(acct.long_options.items()):
            if (p.contract.expiration - ctx.session).days <= self.roll_dte:
                orders.append(SellToClose(pid, tag="roll"))
        live = [pid for pid, p in acct.long_options.items()
                if (p.contract.expiration - ctx.session).days > self.roll_dte]
        if live:
            return orders
        c = pick_by_delta(ctx.chain, ctx.spot, ctx.session, ctx.rate, option_type="C",
                          target_delta=self.target_delta, dte_target=self.dte_target,
                          dte_lo=self.dte_lo, dte_hi=self.dte_hi)
        if c is None:
            return orders
        n = _contracts_for_budget(self.budget_pct * _equity_now(ctx), c.ask, 0.70)
        if n >= 1:
            orders.append(BuyToOpen(c, n, tag="bench_entry"))
        return orders


@dataclass
class H4TrendPuts:
    """Control: mirror of H2 with puts while below the 200d MA."""

    signals: pd.DataFrame
    target_delta: float = -0.60
    dte_target: int = 60
    dte_lo: int = 40
    dte_hi: int = 70
    roll_dte: int = 21
    budget_pct: float = 0.08

    def on_session(self, ctx: Ctx) -> list[Order]:
        orders: list[Order] = []
        acct = ctx.account
        s = asof(self.signals, ctx.session)
        if s is None:
            return orders
        below = not bool(s["above_ma200_dn"])
        if not below:
            for pid in list(acct.long_options):
                orders.append(SellToClose(pid, tag="trend_off"))
            return orders
        for pid, p in list(acct.long_options.items()):
            if (p.contract.expiration - ctx.session).days <= self.roll_dte:
                orders.append(SellToClose(pid, tag="roll"))
        live = [pid for pid, p in acct.long_options.items()
                if (p.contract.expiration - ctx.session).days > self.roll_dte]
        if live:
            return orders
        c = pick_by_delta(ctx.chain, ctx.spot, ctx.session, ctx.rate, option_type="P",
                          target_delta=self.target_delta, dte_target=self.dte_target,
                          dte_lo=self.dte_lo, dte_hi=self.dte_hi)
        if c is None:
            return orders
        n = _contracts_for_budget(self.budget_pct * _equity_now(ctx), c.ask, 0.70)
        if n >= 1:
            orders.append(BuyToOpen(c, n, tag="h4_entry"))
        return orders


@dataclass
class H5Wheel:
    """Cash-secured put wheel: sell ~25Δ 30-45 DTE put; close at profit_take of
    credit or tend to assignment; when assigned, sell ~25Δ covered calls at or
    above basis; when called away, restart. One underlying per account slot."""

    target_delta: float = -0.25
    dte_target: int = 35
    dte_lo: int = 25
    dte_hi: int = 50
    profit_take: float = 0.5  # buy back when mid <= (1-pt)*credit... implemented vs entry credit
    manage_dte: int = 7  # buy back short with dte <= this if OTM-cheap, else let assign
    cc_min_strike_rel_basis: float = 1.0  # covered call strike >= basis
    budget_frac: float = 0.60  # max collateral as fraction of equity
    earnings_dates: frozenset = frozenset()  # skip new shorts spanning a report

    def _spans_earnings(self, session, expiration) -> bool:
        return any(session < d <= expiration for d in self.earnings_dates)

    def on_session(self, ctx: Ctx) -> list[Order]:
        from .engine import BuyToCloseShort  # local import to avoid cycle noise
        orders: list[Order] = []
        acct = ctx.account
        eq = _equity_now(ctx)
        # manage short puts
        for pid, p in list(acct.short_puts.items()):
            from .selection import lookup
            q = lookup(ctx.chain, p.contract.expiration, p.contract.strike, "P")
            if q is None:
                continue
            dte = (p.contract.expiration - ctx.session).days
            if q.mid <= (1 - self.profit_take) * p.credit and q.ask > 0:
                orders.append(BuyToCloseShort(pid, tag="profit_take"))
            elif dte <= self.manage_dte and q.mid <= 0.10 and q.ask > 0:
                orders.append(BuyToCloseShort(pid, tag="cheap_close"))
            # else: ride to expiry; assignment handled by engine
        # covered calls when holding stock
        shares_uncovered = acct.stock.shares - 100 * sum(
            c.contracts for c in acct.short_calls.values())
        if shares_uncovered >= 100:
            c = pick_by_delta(ctx.chain, ctx.spot, ctx.session, ctx.rate, option_type="C",
                              target_delta=0.25, dte_target=self.dte_target,
                              dte_lo=self.dte_lo, dte_hi=self.dte_hi, for_sell=True)
            if c is not None and c.strike >= acct.stock.basis * self.cc_min_strike_rel_basis:
                orders.append(SellCallToOpen(c, shares_uncovered // 100, tag="wheel_cc"))
        # new CSP when flat
        if not acct.short_puts and acct.stock.shares == 0:
            c = pick_by_delta(ctx.chain, ctx.spot, ctx.session, ctx.rate, option_type="P",
                              target_delta=self.target_delta, dte_target=self.dte_target,
                              dte_lo=self.dte_lo, dte_hi=self.dte_hi, for_sell=True)
            if (c is not None and c.strike * 100 <= self.budget_frac * eq
                    and not self._spans_earnings(ctx.session, c.expiration)):
                orders.append(SellPutToOpen(c, 1, tag="wheel_csp"))
        return orders
