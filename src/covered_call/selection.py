"""Deterministic contract selection from a session snapshot.

Delta is computed locally (Black-Scholes on the snapshot mid via implied vol) —
never from vendor fields — so the value used for selection is exactly what was
knowable at the snapshot.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date

import pandas as pd

from .domain import OptionType
from .options.black_scholes import bs_greeks, year_fraction
from .options.implied_vol import implied_vol

# Liquidity gates (audit §3): reject wide/crossed/empty quotes.
MAX_REL_SPREAD = 0.10
MIN_OI = 100


@dataclass(frozen=True)
class Contract:
    expiration: date
    strike: float
    option_type: str  # 'C' | 'P'
    bid: float
    ask: float
    mid: float
    dte: int
    delta: float | None
    iv: float | None
    open_interest: float


def _delta_iv(row, spot: float, session: date, rate: float, q: float = 0.013):
    t = year_fraction(session, row.expiration)
    if t <= 0 or row.mid <= 0:
        return None, None
    ot = OptionType(row.option_type)
    try:
        iv = implied_vol(row.mid, spot, row.strike, t, rate, q, ot)
    except Exception:
        return None, None
    if iv is None or not (0.005 < iv < 5.0):
        return None, None
    g = bs_greeks(spot, row.strike, t, iv, rate, q, ot)
    return float(g.delta), float(iv)


def eligible(chain: pd.DataFrame, option_type: str, dte_lo: int, dte_hi: int,
             for_sell: bool = False) -> pd.DataFrame:
    c = chain[
        (chain.option_type == option_type)
        & (chain.dte >= dte_lo)
        & (chain.dte <= dte_hi)
        & (chain.ask > 0)
        & (chain.bid <= chain.ask)
        & (chain.rel_spread <= MAX_REL_SPREAD)
        & (chain.open_interest.fillna(0) >= MIN_OI)
    ]
    if for_sell:
        c = c[c.bid > 0]
    return c


def _nearest_expiry_pool(c: pd.DataFrame, dte_target: int) -> pd.DataFrame:
    if not len(c):
        return c
    best_exp = c.loc[(c.dte - dte_target).abs().idxmin(), "expiration"]
    return c[c.expiration == best_exp]


def pick_by_delta(chain: pd.DataFrame, spot: float, session: date, rate: float, *,
                  option_type: str, target_delta: float, dte_target: int,
                  dte_lo: int, dte_hi: int, for_sell: bool = False) -> Contract | None:
    """Nearest listed expiry to dte_target within [lo,hi], then the strike whose
    locally-computed delta is closest to target. Deterministic; ties -> lower strike."""
    c = eligible(chain, option_type, dte_lo, dte_hi, for_sell=for_sell)
    c = _nearest_expiry_pool(c, dte_target)
    # plausible strike window for |delta| in [0.15, 0.85] — cuts IV work ~8x
    c = c[(c.strike >= 0.75 * spot) & (c.strike <= 1.18 * spot) & (c.mid >= 0.05)]
    if not len(c):
        return None
    rows = []
    for row in c.sort_values("strike").itertuples():
        d, iv = _delta_iv(row, spot, session, rate)
        if d is None:
            continue
        rows.append((abs(abs(d) - abs(target_delta)), row, d, iv))
    if not rows:
        return None
    rows.sort(key=lambda x: (x[0], x[1].strike))
    _, row, d, iv = rows[0]
    if abs(abs(d) - abs(target_delta)) > 0.12:  # nothing anywhere near target
        return None
    return Contract(row.expiration, float(row.strike), option_type, float(row.bid),
                    float(row.ask), float(row.mid), int(row.dte), d, iv,
                    float(row.open_interest or 0))


def pick_affordable_call(chain: pd.DataFrame, spot: float, session: date, rate: float, *,
                         budget: float, fee: float, delta_hi: float, delta_lo: float,
                         dte_target: int, dte_lo: int, dte_hi: int,
                         prefer_delta: float) -> Contract | None:
    """Highest-delta call within [delta_lo, delta_hi] whose ask-side cost fits the
    budget; among equally affordable, the one nearest prefer_delta from above.
    Deterministic. Used by the small-account variant where the preferred delta's
    premium can exceed the per-position budget."""
    c = eligible(chain, "C", dte_lo, dte_hi)
    c = _nearest_expiry_pool(c, dte_target)
    c = c[(c.strike >= 0.75 * spot) & (c.strike <= 1.18 * spot) & (c.mid >= 0.05)]
    if not len(c):
        return None
    cands = []
    for row in c.sort_values("strike").itertuples():
        d, iv = _delta_iv(row, spot, session, rate)
        if d is None or not (delta_lo - 1e-9 <= d <= delta_hi + 1e-9):
            continue
        if row.ask * 100 + fee > budget:
            continue
        cands.append((d, row, iv))
    if not cands:
        return None
    cands.sort(key=lambda x: (-x[0], x[1].strike))  # highest delta first
    d, row, iv = cands[0]
    return Contract(row.expiration, float(row.strike), "C", float(row.bid), float(row.ask),
                    float(row.mid), int(row.dte), d, iv, float(row.open_interest or 0))


def lookup(chain: pd.DataFrame, expiration: date, strike: float, option_type: str) -> Contract | None:
    c = chain[(chain.expiration == expiration) & (chain.strike == strike)
              & (chain.option_type == option_type)]
    if not len(c):
        return None
    r = c.iloc[0]
    return Contract(r.expiration, float(r.strike), option_type, float(r.bid), float(r.ask),
                    float(r.mid), int(r.dte), None, None, float(r.open_interest or 0))
