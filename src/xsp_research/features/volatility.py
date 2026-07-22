"""Family A: volatility-regime features.

Sources: VIX (level index), optional VIX9D/VVIX, UNDERLYING closes.
All are hypotheses to be tested, not assumed predictors.
Overnight-gap features require OHLC data and are deferred until the bundle
carries opens (documented gap, not silently approximated).
"""

from __future__ import annotations

import polars as pl

from xsp_research.features import helpers as h
from xsp_research.features.registry import FeatureFamily, MarketDataBundle, feature

F = FeatureFamily.VOLATILITY


@feature("vix_level", F, "VIX close level", ("VIX",))
def vix_level(b: MarketDataBundle) -> pl.DataFrame:
    return h.value_frame(b.get("VIX"), pl.col("close"))


@feature(
    "vix_pct_252",
    F,
    "trailing 252-session percentile of VIX close",
    ("VIX",),
    availability="needs 252 sessions of VIX history",
)
def vix_pct_252(b: MarketDataBundle) -> pl.DataFrame:
    return h.trailing_percentile(b.get("VIX"), 252)


@feature("vix_chg_5d", F, "5-session change in VIX close (points)", ("VIX",))
def vix_chg_5d(b: MarketDataBundle) -> pl.DataFrame:
    return h.value_frame(b.get("VIX"), pl.col("close") - pl.col("close").shift(5))


@feature("vix_chg_20d", F, "20-session change in VIX close (points)", ("VIX",))
def vix_chg_20d(b: MarketDataBundle) -> pl.DataFrame:
    return h.value_frame(b.get("VIX"), pl.col("close") - pl.col("close").shift(20))


@feature("vix9d_vix_ratio", F, "VIX9D / VIX (term-structure inversion gauge)", ("VIX9D", "VIX"))
def vix9d_vix_ratio(b: MarketDataBundle) -> pl.DataFrame:
    return h.ratio(b.get("VIX9D"), b.get("VIX"))


@feature("vvix_level", F, "VVIX close level (vol-of-vol)", ("VVIX",))
def vvix_level(b: MarketDataBundle) -> pl.DataFrame:
    return h.value_frame(b.get("VVIX"), pl.col("close"))


@feature("rv_5", F, "5-session annualized realized vol of UNDERLYING", ("UNDERLYING",))
def rv_5(b: MarketDataBundle) -> pl.DataFrame:
    return h.realized_vol(b.get("UNDERLYING"), 5)


@feature("rv_10", F, "10-session annualized realized vol of UNDERLYING", ("UNDERLYING",))
def rv_10(b: MarketDataBundle) -> pl.DataFrame:
    return h.realized_vol(b.get("UNDERLYING"), 10)


@feature("rv_20", F, "20-session annualized realized vol of UNDERLYING", ("UNDERLYING",))
def rv_20(b: MarketDataBundle) -> pl.DataFrame:
    return h.realized_vol(b.get("UNDERLYING"), 20)


@feature("rv_60", F, "60-session annualized realized vol of UNDERLYING", ("UNDERLYING",))
def rv_60(b: MarketDataBundle) -> pl.DataFrame:
    return h.realized_vol(b.get("UNDERLYING"), 60)


@feature(
    "iv_minus_rv20",
    F,
    "VIX/100 minus 20-session realized vol (variance-risk-premium proxy)",
    ("VIX", "UNDERLYING"),
)
def iv_minus_rv20(b: MarketDataBundle) -> pl.DataFrame:
    vix = h.value_frame(b.get("VIX"), pl.col("close") / 100.0).rename({"value": "iv"})
    rv = h.realized_vol(b.get("UNDERLYING"), 20).rename({"value": "rv"})
    return vix.join(rv, on="date", how="inner").select(
        pl.col("date"), (pl.col("iv") - pl.col("rv")).alias("value")
    )


@feature("vol_of_vol_20", F, "20-session annualized vol of VIX log changes", ("VIX",))
def vol_of_vol_20(b: MarketDataBundle) -> pl.DataFrame:
    return h.realized_vol(b.get("VIX"), 20)


@feature(
    "expected_move_30d",
    F,
    "UNDERLYING * VIX/100 * sqrt(30/365): 30-day expected move in points",
    ("VIX", "UNDERLYING"),
)
def expected_move_30d(b: MarketDataBundle) -> pl.DataFrame:
    vix = h.value_frame(b.get("VIX"), pl.col("close")).rename({"value": "vix"})
    und = h.value_frame(b.get("UNDERLYING"), pl.col("close")).rename({"value": "spot"})
    return vix.join(und, on="date", how="inner").select(
        pl.col("date"),
        (pl.col("spot") * pl.col("vix") / 100.0 * (30.0 / 365.0) ** 0.5).alias("value"),
    )
