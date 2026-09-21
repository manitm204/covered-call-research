"""Strategy implementations for the hypothesis set (research_plan.md §2).

All strategies: whole contracts, premium budgets as % of current equity, forced
exit before expiry week, one position per slot, no averaging down.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date

import pandas as pd

from .engine import (BuyStock, BuyToOpen, Ctx, Order, SellCallToOpen, SellPutToOpen,
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


@dataclass
class BuyHoldShares:
    """Own lots*100 shares permanently; buy once, rebuy immediately if ever
    reduced. No options. The zero-strategy baseline for the covered-call work."""

    lots: int = 1

    def on_session(self, ctx: Ctx) -> list[Order]:
        target = 100 * self.lots
        held = ctx.account.stock.shares
        if held < target:
            return [BuyStock(target - held, tag="init_buy")]
        return []


@dataclass
class CoveredCallStrategy:
    """Own lots*100 shares permanently (rebuy the session after being called
    away — settlement clears the shares first, so the rebuy lands one session
    late; documented simplification, not a strategy edge). Whenever shares are
    fully uncovered, sell one ~target_delta OTM call per lot (dte in
    [dte_lo, dte_hi]); let it ride to expiry/assignment, then repeat.

    gated=False -> naive "sell every month, no matter what" covered call.
    gated=True  -> skip writing this cycle under the "rip-risk veto board"
    (see scripts/covered_call_signal_research.py): a weekly rank-IC study,
    2005-2026, of nine regime signals against three targets that matter for a
    short call specifically (does the underlying breach an OTM strike within
    a month; the continuous upside-tail size; the vol-normalized forward
    return), each with a block-bootstrap 95% CI. The signals whose CI excludes
    zero on breach probability, on *all three* of SPY/QQQ/IWM, describe a
    *weak*, not strong, tape: RSI(14) < 40, 84-session trend < 0%, and price
    below its 200-day MA. Two signals that matter a lot for *plain* forward
    return did NOT clear the bar on breach probability specifically and are
    deliberately excluded here: elevated own vol-index level (predicts bigger
    moves in general, not a disproportionate breach of a vol-scaled strike —
    the option is already priced for that), and market-wide sector
    correlation (correlated markets do have better forward returns, but that
    doesn't translate into more OR fewer strike breaches). `iwm_sector_corr_leg`
    is kept as an optional, off-by-default fourth leg for experimentation; an
    earlier, differently-defined version of this signal (each fund's own
    correlation to the sector average, which is nearly tautological for a
    broad index) looked significant for IWM and does not replicate under the
    corrected market-wide definition.
    """

    signals: pd.DataFrame
    lots: int = 1
    target_delta: float = 0.25
    dte_target: int = 35
    dte_lo: int = 25
    dte_hi: int = 45
    gated: bool = True
    rsi_thresh: float = 40.0
    trend84_thresh: float = 0.0
    use_ma200_leg: bool = True
    # breach-gate leg selection is per-ticker: RSI<40 and trend84<0% clear a
    # 90% CI on all three funds, but price-below-MA200 only clears it on
    # SPY/QQQ — it's dropped for IWM (use_ma200_leg=False there).
    iwm_sector_corr_leg: bool = False
    sector_corr_thresh: float = 0.70
    gate_mode: str = "breach"  # "breach" or "fwd_ret" — see _rip_risk docstring
    # gate_mode="breach" (default): veto on the weak-tape rip-risk board above.
    # gate_mode="fwd_ret": veto instead on the signals with the strongest IC
    # against plain forward return (RSI14, sector_corr_60, trend_84d, and the
    # fund's own vol-index level — see scripts/covered_call_signal_research.py).
    # Thresholds are each signal's own tercile cutoff, 2005-2026 weekly sample,
    # computed per ticker (they differ slightly fund to fund).
    fwdret_sector_corr_thresh: float = 0.677
    fwdret_absorption_thresh: float | None = None  # off by default — near-duplicate of sector_corr/vol_own
    fwdret_vol_thresh: float | None = None  # required when gate_mode="fwd_ret"
    fwdret_rsi_thresh: float | None = None  # veto if RSI14 below this (oversold -> good month coming)
    fwdret_trend84_thresh: float | None = None  # veto if trend84 below this
    fwdret_use_ma200_leg: bool = False  # IWM only — the only fund where px-vs-MA200 clears fwd_ret's IC bar

    def _rip_risk(self, ctx: Ctx) -> bool:
        s = asof(self.signals, ctx.session)
        if s is None:
            return False
        if self.gate_mode == "fwd_ret":
            high_corr = pd.notna(s.get("sector_corr_60")) and s["sector_corr_60"] > self.fwdret_sector_corr_thresh
            rising_absorption = (self.fwdret_absorption_thresh is not None and pd.notna(s.get("absorption_shift"))
                                 and s["absorption_shift"] > self.fwdret_absorption_thresh)
            high_vol = (self.fwdret_vol_thresh is not None and pd.notna(s.get("vol_own"))
                       and s["vol_own"] > self.fwdret_vol_thresh)
            oversold = (self.fwdret_rsi_thresh is not None and pd.notna(s.get("rsi14"))
                       and s["rsi14"] < self.fwdret_rsi_thresh)
            weak_trend = (self.fwdret_trend84_thresh is not None and pd.notna(s.get("ret84"))
                         and s["ret84"] < self.fwdret_trend84_thresh)
            below_ma200 = (self.fwdret_use_ma200_leg and pd.notna(s.get("px")) and pd.notna(s.get("ma200"))
                          and s["px"] < s["ma200"])
            return bool(high_corr or rising_absorption or high_vol or oversold or weak_trend or below_ma200)
        oversold = pd.notna(s.get("rsi14")) and s["rsi14"] < self.rsi_thresh
        weak_trend = pd.notna(s.get("ret84")) and s["ret84"] < self.trend84_thresh
        below_ma200 = (self.use_ma200_leg and pd.notna(s.get("px")) and pd.notna(s.get("ma200"))
                      and s["px"] < s["ma200"])
        veto = bool(oversold or weak_trend or below_ma200)
        if self.iwm_sector_corr_leg and pd.notna(s.get("sector_corr_60")):
            veto = veto or bool(s["sector_corr_60"] > self.sector_corr_thresh)
        return veto

    def on_session(self, ctx: Ctx) -> list[Order]:
        acct = ctx.account
        target = 100 * self.lots
        if acct.stock.shares < target:
            # buy back up to `target`, but never require the full lot in one
            # unaffordable order — after an assignment, deploy whatever cash
            # is on hand into as many whole shares as it affords (even a
            # partial lot), the same fix applied to ReinvestingCoveredCallStrategy.
            # Requiring the exact full lot and doing nothing otherwise was the
            # bug: it left the account sitting in 100% cash indefinitely
            # whenever it was even $1 short of a full rebuy.
            afford = int(acct.available_cash // ctx.spot)
            want = min(target - acct.stock.shares, afford)
            if want > 0:
                return [BuyStock(want, tag="init_buy" if acct.stock.shares == 0 else "reinvest")]
            return []
        uncovered = acct.stock.shares - 100 * sum(c.contracts for c in acct.short_calls.values())
        if uncovered < 100:
            return []
        if self.gated and self._rip_risk(ctx):
            return []
        c = pick_by_delta(ctx.chain, ctx.spot, ctx.session, ctx.rate, option_type="C",
                          target_delta=self.target_delta, dte_target=self.dte_target,
                          dte_lo=self.dte_lo, dte_hi=self.dte_hi, for_sell=True)
        if c is None:
            return []
        return [SellCallToOpen(c, uncovered // 100, tag="cc_write")]


@dataclass
class DripBuyHoldShares:
    """Own initial_shares permanently; every session, sweep all free cash
    (dividends, interest — there are no options here) into more whole shares
    instead of letting it sit idle. The fair 'fully-compounding' buy-and-hold
    baseline for the reinvesting covered-call strategies below."""

    initial_shares: int = 200

    def on_session(self, ctx: Ctx) -> list[Order]:
        acct = ctx.account
        afford = int(acct.available_cash // ctx.spot)
        if afford > 0:
            return [BuyStock(afford, tag="init_buy" if acct.stock.shares == 0 else "reinvest")]
        return []


@dataclass
class ReinvestingCoveredCallStrategy:
    """Own initial_shares (default 200) permanently; ALWAYS keep exactly
    covered_lots*100 shares (default 100) covered by one short call, leaving
    the rest (initially 100, growing over time) permanently uncovered as a
    buffer. Every session, sweep all free cash — option premium, assignment
    proceeds, dividends, interest, all alike, none of it ever sits idle —
    into as many additional whole shares as it affords. This is a compounding
    variant of CoveredCallStrategy: the position grows over time instead of
    staying pinned at one lot.

    gated=False -> naive "sell every month, no matter what" covered call.
    gated=True  -> same rip-risk veto board as CoveredCallStrategy (RSI < 40,
    84-session trend < 0%, price below its 200-day MA — see
    scripts/covered_call_signal_research.py).
    """

    signals: pd.DataFrame
    initial_shares: int = 200
    covered_lots: int = 1
    target_delta: float = 0.25
    dte_target: int = 35
    dte_lo: int = 25
    dte_hi: int = 45
    gated: bool = True
    rsi_thresh: float | None = 40.0
    trend84_thresh: float | None = 0.0
    use_ma200_leg: bool = True
    # sector-decorrelation leg from the 2026-09 threshold-sweep rule: veto if
    # the market-wide 60d sector correlation drops BELOW this (fragmented tape
    # -> elevated breach risk on SPY/QQQ; doesn't replicate on IWM, left off
    # there by leaving sector_corr_thresh=None).
    sector_corr_thresh: float | None = None

    def _rip_risk(self, ctx: Ctx) -> bool:
        s = asof(self.signals, ctx.session)
        if s is None:
            return False
        oversold = self.rsi_thresh is not None and pd.notna(s.get("rsi14")) and s["rsi14"] < self.rsi_thresh
        weak_trend = self.trend84_thresh is not None and pd.notna(s.get("ret84")) and s["ret84"] < self.trend84_thresh
        below_ma200 = (self.use_ma200_leg and pd.notna(s.get("px")) and pd.notna(s.get("ma200"))
                      and s["px"] < s["ma200"])
        fragmented = (self.sector_corr_thresh is not None and pd.notna(s.get("sector_corr_60"))
                     and s["sector_corr_60"] < self.sector_corr_thresh)
        return bool(oversold or weak_trend or below_ma200 or fragmented)

    def on_session(self, ctx: Ctx) -> list[Order]:
        acct = ctx.account
        orders: list[Order] = []
        # always sweep whatever cash is on hand into as many whole shares as
        # it affords — never wait to re-accumulate the full initial_shares in
        # one lump sum (that's the bug this replaced: after an assignment it
        # tried to buy ALL the way back to 200 shares in a single order, that
        # order was unaffordable, and it silently did nothing, forever)
        afford = int(acct.available_cash // ctx.spot)
        if afford > 0:
            orders.append(BuyStock(afford, tag="init_buy" if acct.stock.shares == 0 else "reinvest"))
        covered_target = self.covered_lots * 100
        if not acct.short_calls and acct.stock.shares >= covered_target:
            if not (self.gated and self._rip_risk(ctx)):
                c = pick_by_delta(ctx.chain, ctx.spot, ctx.session, ctx.rate, option_type="C",
                                  target_delta=self.target_delta, dte_target=self.dte_target,
                                  dte_lo=self.dte_lo, dte_hi=self.dte_hi, for_sell=True)
                if c is not None:
                    orders.append(SellCallToOpen(c, self.covered_lots, tag="cc_write"))
        return orders
