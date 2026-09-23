"""Robust implied-volatility inversion.

Uses bracketed Brent root-finding on [VOL_MIN, VOL_MAX] after checking
no-arbitrage price bounds. Returns None (never a garbage number) when the
price is outside arbitrage bounds or the root is not bracketed — callers must
treat None as "no usable IV" and record it in validation reports.
"""

from __future__ import annotations

import math

from scipy.optimize import brentq

from covered_call.domain import OptionType
from covered_call.options.black_scholes import bs_price, forward_price

VOL_MIN = 1e-4
VOL_MAX = 5.0


def no_arb_bounds(
    spot: float, strike: float, t: float, r: float, q: float, option_type: OptionType
) -> tuple[float, float]:
    """(lower, upper) European price bounds under continuous carry."""
    disc_r = math.exp(-r * t)
    f = forward_price(spot, r, q, t)
    if option_type is OptionType.CALL:
        lower = max(disc_r * (f - strike), 0.0)
        upper = disc_r * f
    else:
        lower = max(disc_r * (strike - f), 0.0)
        upper = disc_r * strike
    return lower, upper


def implied_vol(
    price: float,
    spot: float,
    strike: float,
    t: float,
    r: float,
    q: float,
    option_type: OptionType,
) -> float | None:
    """Invert Black-Scholes for volatility; None if unrecoverable."""
    if t <= 0.0 or price <= 0.0 or spot <= 0.0 or strike <= 0.0:
        return None
    lower, upper = no_arb_bounds(spot, strike, t, r, q, option_type)
    tol = 1e-12
    if price <= lower + tol or price >= upper - tol:
        return None

    def objective(vol: float) -> float:
        return bs_price(spot, strike, t, vol, r, q, option_type) - price

    f_lo, f_hi = objective(VOL_MIN), objective(VOL_MAX)
    if f_lo * f_hi > 0:
        return None
    return float(brentq(objective, VOL_MIN, VOL_MAX, xtol=1e-10, maxiter=200))
