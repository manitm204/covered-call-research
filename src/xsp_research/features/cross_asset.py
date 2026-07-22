"""Family D: cross-asset signals (relative strength, credit, rates).

MOVE/dollar/commodity features register the same way once those series are
licensed and present in the bundle; absent sources are skipped with an
explicit reason, never zero-filled.
"""

from __future__ import annotations

import polars as pl

from xsp_research.features import helpers as h
from xsp_research.features.registry import FeatureFamily, MarketDataBundle, feature

F = FeatureFamily.CROSS_ASSET


@feature("qqq_spy_ret20_diff", F, "QQQ 20-session return minus SPY", ("QQQ", "SPY"))
def qqq_spy_ret20_diff(b: MarketDataBundle) -> pl.DataFrame:
    return h.ret_diff(b.get("QQQ"), b.get("SPY"), 20)


@feature("iwm_spy_ret20_diff", F, "IWM 20-session return minus SPY", ("IWM", "SPY"))
def iwm_spy_ret20_diff(b: MarketDataBundle) -> pl.DataFrame:
    return h.ret_diff(b.get("IWM"), b.get("SPY"), 20)


@feature(
    "hyg_lqd_ret20_diff",
    F,
    "HYG 20-session return minus LQD (credit risk appetite)",
    ("HYG", "LQD"),
)
def hyg_lqd_ret20_diff(b: MarketDataBundle) -> pl.DataFrame:
    return h.ret_diff(b.get("HYG"), b.get("LQD"), 20)


@feature(
    "rate_3m_level",
    F,
    "3-month rate level (RATE_3M series close)",
    ("RATE_3M",),
    availability="uses only rates observable at the session (no revised data)",
)
def rate_3m_level(b: MarketDataBundle) -> pl.DataFrame:
    return h.value_frame(b.get("RATE_3M"), pl.col("close"))


@feature("rate_3m_chg_20d", F, "20-session change in the 3-month rate", ("RATE_3M",))
def rate_3m_chg_20d(b: MarketDataBundle) -> pl.DataFrame:
    return h.value_frame(b.get("RATE_3M"), pl.col("close") - pl.col("close").shift(20))


@feature(
    "spy_lqd_corr_60",
    F,
    "60-session rolling correlation of SPY and LQD log returns (bond-equity link)",
    ("SPY", "LQD"),
)
def spy_lqd_corr_60(b: MarketDataBundle) -> pl.DataFrame:
    return h.rolling_return_corr(b.get("SPY"), b.get("LQD"), 60)
