"""Family C: breadth and concentration features (SPY vs RSP).

Constituent-level breadth (% above MAs, advance/decline) requires member data
the bundle does not carry yet; the SPY/RSP complex is the honest subset
available from ETF closes alone.
"""

from __future__ import annotations

import polars as pl

from xsp_research.features import helpers as h
from xsp_research.features.registry import FeatureFamily, MarketDataBundle, feature

F = FeatureFamily.BREADTH


@feature("spy_rsp_ret20_diff", F, "SPY 20-session return minus RSP (cap>eq-weight)", ("SPY", "RSP"))
def spy_rsp_ret20_diff(b: MarketDataBundle) -> pl.DataFrame:
    return h.ret_diff(b.get("SPY"), b.get("RSP"), 20)


@feature("spy_rsp_ret60_diff", F, "SPY 60-session return minus RSP", ("SPY", "RSP"))
def spy_rsp_ret60_diff(b: MarketDataBundle) -> pl.DataFrame:
    return h.ret_diff(b.get("SPY"), b.get("RSP"), 60)


@feature(
    "spy_rsp_ratio_z60",
    F,
    "z-score of SPY/RSP ratio over trailing 60 sessions (concentration trend)",
    ("SPY", "RSP"),
)
def spy_rsp_ratio_z60(b: MarketDataBundle) -> pl.DataFrame:
    r = h.ratio(b.get("SPY"), b.get("RSP")).rename({"value": "close"})
    return h.rolling_zscore(r, 60)


@feature(
    "eqw_capw_vol_ratio_20",
    F,
    "RSP 20-session realized vol / SPY 20-session realized vol",
    ("SPY", "RSP"),
)
def eqw_capw_vol_ratio_20(b: MarketDataBundle) -> pl.DataFrame:
    rsp = h.realized_vol(b.get("RSP"), 20).rename({"value": "a"})
    spy = h.realized_vol(b.get("SPY"), 20).rename({"value": "b"})
    return rsp.join(spy, on="date", how="inner").select(
        pl.col("date"), (pl.col("a") / pl.col("b")).alias("value")
    )
