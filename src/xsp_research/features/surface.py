"""Family F: options-surface features, computed from the entry-time snapshot.

Unlike daily features these are timestamp-exact (lag 0): they derive from the
same chain snapshot the selection and execution decisions use, so no future
information is involved. They are captured per trade by the experiment
runner's entry hook and keyed by trade_id.
"""

from __future__ import annotations

from datetime import date

import polars as pl

from xsp_research.domain import OptionType
from xsp_research.options.black_scholes import year_fraction
from xsp_research.options.greeks import spread_greeks
from xsp_research.options.implied_vol import implied_vol
from xsp_research.strategy.candidate_selection import SelectionResult

SURFACE_FEATURE_NAMES = (
    "surf_atm_iv",
    "surf_iv_25d_minus_atm",
    "surf_wing_slope",
    "surf_curv_short",
    "surf_credit_over_width",
    "surf_credit_over_max_loss",
    "surf_credit_over_exp_move",
    "surf_bidask_over_credit",
    "surf_net_delta",
    "surf_net_gamma",
    "surf_net_vega",
    "surf_net_theta",
    "surf_short_delta",
    "surf_short_oi",
    "surf_dte",
)


def _call_iv_by_strike(
    chain: pl.DataFrame, expiry: date, spot: float, r: float, q: float, as_of: date
) -> dict[float, float]:
    t = year_fraction(as_of, expiry)
    calls = chain.filter(
        (pl.col("expiration") == expiry) & (pl.col("option_type") == "C") & (pl.col("ask") > 0)
    )
    out: dict[float, float] = {}
    for row in calls.iter_rows(named=True):
        mid = 0.5 * (row["bid"] + row["ask"])
        iv = implied_vol(mid, spot, row["strike"], t, r, q, OptionType.CALL)
        if iv is not None:
            out[row["strike"]] = iv
    return out


def _nearest(strikes: list[float], target: float) -> float | None:
    return min(strikes, key=lambda k: abs(k - target)) if strikes else None


def compute_surface_features(
    chain: pl.DataFrame,
    sel: SelectionResult,
    spot: float,
    r: float,
    q: float,
    as_of: date,
) -> dict[str, float | None]:
    """Surface features for a selected spread; None where not computable."""
    out: dict[str, float | None] = dict.fromkeys(SURFACE_FEATURE_NAMES)
    if not sel.selected:
        return out
    sq = sel.spread_quote
    spread = sq.spread
    expiry = spread.expiration
    t = year_fraction(as_of, expiry)
    ivs = _call_iv_by_strike(chain, expiry, spot, r, q, as_of)
    strikes = sorted(ivs)

    atm_k = _nearest(strikes, spot)
    atm_iv = ivs.get(atm_k) if atm_k is not None else None
    out["surf_atm_iv"] = atm_iv

    # 25-delta call: strike whose model delta is nearest 0.25 (approximate via
    # the IV grid: pick by strike distance to the delta-25 strike from sel meta
    # is unavailable, so search directly).
    if atm_iv is not None and t > 0:
        from xsp_research.options.black_scholes import bs_greeks

        deltas = {k: bs_greeks(spot, k, t, ivs[k], r, q, OptionType.CALL).delta for k in strikes}
        k25 = min(deltas, key=lambda k: abs(deltas[k] - 0.25))
        out["surf_iv_25d_minus_atm"] = ivs[k25] - atm_iv

    ks, kl = spread.short_leg.strike, spread.long_leg.strike
    if ks in ivs and kl in ivs and kl > ks:
        out["surf_wing_slope"] = (ivs[kl] - ivs[ks]) / (kl - ks)
    below = [k for k in strikes if k < ks]
    above = [k for k in strikes if k > ks]
    if below and above and ks in ivs:
        kb, ka = below[-1], above[0]
        # second difference scaled by average step (local curvature proxy)
        step = (ka - kb) / 2.0
        out["surf_curv_short"] = (ivs[ka] - 2 * ivs[ks] + ivs[kb]) / (step**2) if step > 0 else None

    credit = sq.credit_mid
    width = spread.width
    if credit > 0:
        out["surf_credit_over_width"] = credit / width
        if width > credit:
            out["surf_credit_over_max_loss"] = credit / (width - credit)
        if atm_iv is not None and t > 0:
            exp_move = spot * atm_iv * (t**0.5)
            if exp_move > 0:
                out["surf_credit_over_exp_move"] = credit / exp_move
        out["surf_bidask_over_credit"] = (
            sq.short_quote.spread_abs + sq.long_quote.spread_abs
        ) / credit

    greeks = spread_greeks(sq, spot, r, q, as_of)
    if greeks is not None:
        out["surf_net_delta"] = greeks.delta
        out["surf_net_gamma"] = greeks.gamma
        out["surf_net_vega"] = greeks.vega
        out["surf_net_theta"] = greeks.theta

    out["surf_short_delta"] = sel.short_delta
    oi = sq.short_quote.open_interest
    out["surf_short_oi"] = float(oi) if oi is not None else None
    out["surf_dte"] = float(sel.dte) if sel.dte is not None else None
    return out
