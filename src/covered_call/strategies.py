"""Covered-call strategy implementations used in the paper.

Four strategies: a zero-strategy buy-and-hold baseline (fixed lot and a
fully-reinvesting variant), and a covered-call writer (fixed lot and a
fully-reinvesting variant) that can either sell every cycle unconditionally
("naive") or skip a cycle when a regime-signal gate is active ("gated").
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .engine import BuyStock, Ctx, Order, SellCallToOpen
from .selection import pick_by_delta
from .signals import asof


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
    below its 200-day MA.
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
    iwm_sector_corr_leg: bool = False
    sector_corr_thresh: float = 0.70

    def _rip_risk(self, ctx: Ctx) -> bool:
        s = asof(self.signals, ctx.session)
        if s is None:
            return False
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
    instead of letting it sit idle. The fully-compounding buy-and-hold
    baseline used in the paper."""

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
    into as many additional whole shares as it affords. This is the
    compounding variant used for the paper's main results (Section 4, 8).

    gated=False -> naive "sell every month, no matter what" covered call.
    gated=True  -> the per-fund rule from the paper (Table 4): a trend leg
    plus one fund-specific second leg.
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
    # sector-decorrelation leg: veto if the market-wide 60d sector correlation
    # drops BELOW this (fragmented tape -> elevated breach risk).
    sector_corr_thresh: float | None = None
    # rising-absorption leg: veto if absorption_shift rises ABOVE this
    # (market structure consolidating -> elevated breach risk).
    absorption_shift_thresh: float | None = None
    # percentage-threshold MA200 leg: veto if price is more than this far
    # below/above its 200-day MA (distinct from use_ma200_leg's plain
    # px < ma200 boolean; lets the MA200 leg be tuned to a specific cutoff).
    ma200_pct_thresh: float | None = None

    def _rip_risk(self, ctx: Ctx) -> bool:
        s = asof(self.signals, ctx.session)
        if s is None:
            return False
        oversold = self.rsi_thresh is not None and pd.notna(s.get("rsi14")) and s["rsi14"] < self.rsi_thresh
        weak_trend = self.trend84_thresh is not None and pd.notna(s.get("ret84")) and s["ret84"] < self.trend84_thresh
        below_ma200 = (self.use_ma200_leg and pd.notna(s.get("px")) and pd.notna(s.get("ma200"))
                      and s["px"] < s["ma200"])
        below_ma200_pct = (self.ma200_pct_thresh is not None and pd.notna(s.get("px")) and pd.notna(s.get("ma200"))
                          and (s["px"] / s["ma200"] - 1.0) < self.ma200_pct_thresh)
        fragmented = (self.sector_corr_thresh is not None and pd.notna(s.get("sector_corr_60"))
                     and s["sector_corr_60"] < self.sector_corr_thresh)
        rising_absorption = (self.absorption_shift_thresh is not None and pd.notna(s.get("absorption_shift"))
                             and s["absorption_shift"] > self.absorption_shift_thresh)
        return bool(oversold or weak_trend or below_ma200 or below_ma200_pct or fragmented or rising_absorption)

    def on_session(self, ctx: Ctx) -> list[Order]:
        acct = ctx.account
        orders: list[Order] = []
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
