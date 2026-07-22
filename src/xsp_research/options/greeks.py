"""Quote-level and spread-level Greek computation.

IV is always model-derived from executable quote mids (vendor Greeks are never
used for selection; see PLAN.md assumption 6).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

from xsp_research.domain import OptionQuote, SpreadQuote
from xsp_research.options.black_scholes import Greeks, bs_greeks, year_fraction
from xsp_research.options.implied_vol import implied_vol


@dataclass(frozen=True, slots=True)
class QuoteAnalytics:
    iv: float
    greeks: Greeks


def analyze_quote(
    quote: OptionQuote, spot: float, r: float, q: float, as_of: date
) -> QuoteAnalytics | None:
    """IV from the quote mid, then analytic Greeks. None if IV unrecoverable."""
    t = year_fraction(as_of, quote.contract.expiration)
    if not quote.has_valid_market:
        return None
    iv = implied_vol(quote.mid, spot, quote.contract.strike, t, r, q, quote.contract.option_type)
    if iv is None:
        return None
    g = bs_greeks(spot, quote.contract.strike, t, iv, r, q, quote.contract.option_type)
    return QuoteAnalytics(iv=iv, greeks=g)


@dataclass(frozen=True, slots=True)
class SpreadGreeks:
    """Net Greeks of one SHORT bear call spread (short low strike, long high strike)."""

    delta: float
    gamma: float
    vega: float
    theta: float
    short_leg_delta: float  # raw long-call delta of the short leg (for delta stops)


def spread_greeks(
    sq: SpreadQuote, spot: float, r: float, q: float, as_of: date
) -> SpreadGreeks | None:
    short_a = analyze_quote(sq.short_quote, spot, r, q, as_of)
    long_a = analyze_quote(sq.long_quote, spot, r, q, as_of)
    if short_a is None or long_a is None:
        return None
    s, lo = short_a.greeks, long_a.greeks
    return SpreadGreeks(
        delta=-s.delta + lo.delta,
        gamma=-s.gamma + lo.gamma,
        vega=-s.vega + lo.vega,
        theta=-s.theta + lo.theta,
        short_leg_delta=s.delta,
    )
