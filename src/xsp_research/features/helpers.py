"""Shared building blocks for daily feature computations.

All helpers preserve the prefix property: the value at date d depends only on
observations at dates <= d (rolling/expanding windows, never centered or
full-sample statistics).
"""

from __future__ import annotations

import math

import numpy as np
import polars as pl

ANN = math.sqrt(252.0)


def value_frame(df: pl.DataFrame, expr: pl.Expr) -> pl.DataFrame:
    return df.sort("date").select(pl.col("date"), expr.alias("value"))


def pct_change(df: pl.DataFrame, sessions: int) -> pl.DataFrame:
    return value_frame(df, pl.col("close") / pl.col("close").shift(sessions) - 1.0)


def log_returns(df: pl.DataFrame) -> pl.DataFrame:
    return value_frame(df, (pl.col("close") / pl.col("close").shift(1)).log())


def realized_vol(df: pl.DataFrame, window: int) -> pl.DataFrame:
    rets = (pl.col("close") / pl.col("close").shift(1)).log()
    return value_frame(df, rets.rolling_std(window_size=window) * ANN)


def rolling_zscore(df: pl.DataFrame, window: int) -> pl.DataFrame:
    x = pl.col("close")
    mu = x.rolling_mean(window_size=window)
    sd = x.rolling_std(window_size=window)
    return value_frame(df, (x - mu) / sd)


def trailing_percentile(df: pl.DataFrame, window: int) -> pl.DataFrame:
    """Fraction of the trailing window (inclusive) at or below today's value."""
    s = df.sort("date")
    x = s["close"].to_numpy()
    out = np.full(len(x), np.nan)
    for i in range(window - 1, len(x)):
        w = x[i - window + 1 : i + 1]
        out[i] = float(np.mean(w <= x[i]))
    # Warm-up rows are null (not NaN) so missing-value policies apply cleanly.
    return pl.DataFrame({"date": s["date"], "value": out}).with_columns(
        pl.col("value").fill_nan(None)
    )


def ratio(df_a: pl.DataFrame, df_b: pl.DataFrame) -> pl.DataFrame:
    j = df_a.sort("date").join(df_b.sort("date"), on="date", how="inner", suffix="_b")
    return j.select(pl.col("date"), (pl.col("close") / pl.col("close_b")).alias("value"))


def ret_diff(df_a: pl.DataFrame, df_b: pl.DataFrame, sessions: int) -> pl.DataFrame:
    a = pct_change(df_a, sessions).rename({"value": "a"})
    b = pct_change(df_b, sessions).rename({"value": "b"})
    return a.join(b, on="date", how="inner").select(
        pl.col("date"), (pl.col("a") - pl.col("b")).alias("value")
    )


def rolling_return_corr(df_a: pl.DataFrame, df_b: pl.DataFrame, window: int) -> pl.DataFrame:
    a = log_returns(df_a).rename({"value": "a"})
    b = log_returns(df_b).rename({"value": "b"})
    j = a.join(b, on="date", how="inner").sort("date")
    return j.select(
        pl.col("date"),
        pl.rolling_corr(pl.col("a"), pl.col("b"), window_size=window).alias("value"),
    )


def rsi(df: pl.DataFrame, window: int = 14) -> pl.DataFrame:
    delta = pl.col("close").diff()
    gain = delta.clip(lower_bound=0.0).rolling_mean(window_size=window)
    loss = (-delta).clip(lower_bound=0.0).rolling_mean(window_size=window)
    value = pl.when(loss == 0.0).then(100.0).otherwise(100.0 - 100.0 / (1.0 + gain / loss))
    return value_frame(df, value)
