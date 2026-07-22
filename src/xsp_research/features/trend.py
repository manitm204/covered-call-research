"""Family B: trend and momentum features on the UNDERLYING.

Overnight-vs-intraday decomposition needs opens (OHLC); deferred until the
bundle carries them.
"""

from __future__ import annotations

import polars as pl

from xsp_research.features import helpers as h
from xsp_research.features.registry import FeatureFamily, MarketDataBundle, feature

F = FeatureFamily.TREND


def _und(b: MarketDataBundle) -> pl.DataFrame:
    return b.get("UNDERLYING")


for _n in (1, 5, 20, 60, 120):

    def _make(n: int):
        @feature(f"ret_{n}d", F, f"{n}-session simple return of UNDERLYING", ("UNDERLYING",))
        def _ret(b: MarketDataBundle, n: int = n) -> pl.DataFrame:
            return h.pct_change(_und(b), n)

        return _ret

    _make(_n)


@feature("dist_ma50", F, "close / 50-session MA - 1", ("UNDERLYING",))
def dist_ma50(b: MarketDataBundle) -> pl.DataFrame:
    return h.value_frame(
        _und(b), pl.col("close") / pl.col("close").rolling_mean(window_size=50) - 1.0
    )


@feature("dist_ma200", F, "close / 200-session MA - 1", ("UNDERLYING",))
def dist_ma200(b: MarketDataBundle) -> pl.DataFrame:
    return h.value_frame(
        _und(b), pl.col("close") / pl.col("close").rolling_mean(window_size=200) - 1.0
    )


@feature("ma50_slope_20d", F, "20-session change of the 50-session MA (relative)", ("UNDERLYING",))
def ma50_slope_20d(b: MarketDataBundle) -> pl.DataFrame:
    ma = pl.col("close").rolling_mean(window_size=50)
    return h.value_frame(_und(b), ma / ma.shift(20) - 1.0)


@feature("rsi_14", F, "14-session RSI (simple-mean variant)", ("UNDERLYING",))
def rsi_14(b: MarketDataBundle) -> pl.DataFrame:
    return h.rsi(_und(b), 14)


@feature("dist_from_ath", F, "close / running all-time high - 1 (within sample)", ("UNDERLYING",))
def dist_from_ath(b: MarketDataBundle) -> pl.DataFrame:
    return h.value_frame(_und(b), pl.col("close") / pl.col("close").cum_max() - 1.0)


@feature(
    "rebound_from_20d_low",
    F,
    "close / 20-session rolling low - 1 (rebound-risk gauge for call spreads)",
    ("UNDERLYING",),
)
def rebound_from_20d_low(b: MarketDataBundle) -> pl.DataFrame:
    return h.value_frame(
        _und(b), pl.col("close") / pl.col("close").rolling_min(window_size=20) - 1.0
    )


@feature(
    "ret_accel_5_20",
    F,
    "5-session return minus 20-session return/4 (acceleration)",
    ("UNDERLYING",),
)
def ret_accel_5_20(b: MarketDataBundle) -> pl.DataFrame:
    r5 = h.pct_change(_und(b), 5).rename({"value": "r5"})
    r20 = h.pct_change(_und(b), 20).rename({"value": "r20"})
    return r5.join(r20, on="date", how="inner").select(
        pl.col("date"), (pl.col("r5") - pl.col("r20") / 4.0).alias("value")
    )
