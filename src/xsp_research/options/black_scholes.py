"""Black-Scholes / Black-76 pricing and analytic Greeks for European index options.

Conventions:
- ``t`` is time to expiration in years (ACT/365 by callers).
- ``r`` and ``q`` are continuously compounded annual rates (risk-free, carry/dividend).
- Vega is per unit of volatility (multiply by 0.01 for per-vol-point).
- Theta is per year (divide by 365 for per-calendar-day).
- Degenerate inputs (t<=0 or vol<=0) return discounted intrinsic on the forward,
  which is the correct no-time-value limit for European options.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from xsp_research.domain import OptionType

_SQRT_2PI = math.sqrt(2.0 * math.pi)


def _norm_pdf(x: float) -> float:
    return math.exp(-0.5 * x * x) / _SQRT_2PI


def _norm_cdf(x: float) -> float:
    return 0.5 * math.erfc(-x / math.sqrt(2.0))


def forward_price(spot: float, r: float, q: float, t: float) -> float:
    """Index forward under continuous carry: F = S * exp((r - q) t)."""
    return spot * math.exp((r - q) * t)


def _d1_d2(f: float, k: float, t: float, vol: float) -> tuple[float, float]:
    sig_sqrt_t = vol * math.sqrt(t)
    d1 = (math.log(f / k) + 0.5 * vol * vol * t) / sig_sqrt_t
    return d1, d1 - sig_sqrt_t


def black76_price(
    forward: float,
    strike: float,
    t: float,
    vol: float,
    discount_factor: float,
    option_type: OptionType,
) -> float:
    """Black-76: price off the forward. The natural form for index options."""
    if t <= 0.0 or vol <= 0.0:
        intrinsic = (
            max(forward - strike, 0.0)
            if option_type is OptionType.CALL
            else max(strike - forward, 0.0)
        )
        return discount_factor * intrinsic
    d1, d2 = _d1_d2(forward, strike, t, vol)
    if option_type is OptionType.CALL:
        return discount_factor * (forward * _norm_cdf(d1) - strike * _norm_cdf(d2))
    return discount_factor * (strike * _norm_cdf(-d2) - forward * _norm_cdf(-d1))


def bs_price(
    spot: float,
    strike: float,
    t: float,
    vol: float,
    r: float,
    q: float,
    option_type: OptionType,
) -> float:
    """Black-Scholes-Merton on spot with continuous dividend yield q."""
    f = forward_price(spot, r, q, t) if t > 0 else spot
    return black76_price(f, strike, t, vol, math.exp(-r * max(t, 0.0)), option_type)


@dataclass(frozen=True, slots=True)
class Greeks:
    delta: float
    gamma: float
    vega: float  # per unit vol
    theta: float  # per year
    rho: float  # per unit rate

    def per_day_theta(self) -> float:
        return self.theta / 365.0


def bs_greeks(
    spot: float,
    strike: float,
    t: float,
    vol: float,
    r: float,
    q: float,
    option_type: OptionType,
) -> Greeks:
    """Analytic spot Greeks under Black-Scholes-Merton."""
    if t <= 0.0 or vol <= 0.0:
        # Limit case: delta is a step function; other Greeks vanish.
        f = forward_price(spot, r, q, max(t, 0.0))
        itm = f > strike if option_type is OptionType.CALL else f < strike
        sign = 1.0 if option_type is OptionType.CALL else -1.0
        return Greeks(delta=sign if itm else 0.0, gamma=0.0, vega=0.0, theta=0.0, rho=0.0)

    f = forward_price(spot, r, q, t)
    d1, d2 = _d1_d2(f, strike, t, vol)
    sqrt_t = math.sqrt(t)
    disc_r = math.exp(-r * t)
    disc_q = math.exp(-q * t)
    pdf_d1 = _norm_pdf(d1)

    gamma = disc_q * pdf_d1 / (spot * vol * sqrt_t)
    vega = spot * disc_q * pdf_d1 * sqrt_t

    if option_type is OptionType.CALL:
        delta = disc_q * _norm_cdf(d1)
        theta = (
            -spot * disc_q * pdf_d1 * vol / (2.0 * sqrt_t)
            - r * strike * disc_r * _norm_cdf(d2)
            + q * spot * disc_q * _norm_cdf(d1)
        )
        rho = strike * t * disc_r * _norm_cdf(d2)
    else:
        delta = -disc_q * _norm_cdf(-d1)
        theta = (
            -spot * disc_q * pdf_d1 * vol / (2.0 * sqrt_t)
            + r * strike * disc_r * _norm_cdf(-d2)
            - q * spot * disc_q * _norm_cdf(-d1)
        )
        rho = -strike * t * disc_r * _norm_cdf(-d2)

    return Greeks(delta=delta, gamma=gamma, vega=vega, theta=theta, rho=rho)


def year_fraction(start_date, end_date) -> float:
    """ACT/365 fixed day count between two dates."""
    return max((end_date - start_date).days, 0) / 365.0
