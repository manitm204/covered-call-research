"""Family E: PCA/eigenvalue regime features on sector returns.

Each value at data-date d is computed from a trailing ``WINDOW``-session
correlation matrix of sector daily returns ending at d — strictly rolling, so
the prefix property holds by construction (verified by leakage tests anyway).
"""

from __future__ import annotations

from datetime import date

import numpy as np
import polars as pl

from xsp_research.features.registry import (
    SECTORS_SENTINEL,
    FeatureFamily,
    MarketDataBundle,
    feature,
)

F = FeatureFamily.PCA_REGIME
WINDOW = 120


def _sector_returns(bundle: MarketDataBundle) -> tuple[list[date], np.ndarray]:
    frames = [bundle.get(s).rename({"close": s}) for s in bundle.sector_symbols]
    joined = frames[0]
    for f in frames[1:]:
        joined = joined.join(f, on="date", how="inner")
    joined = joined.sort("date")
    closes = joined.select(list(bundle.sector_symbols)).to_numpy()
    rets = closes[1:] / closes[:-1] - 1.0
    return joined["date"].to_list()[1:], rets


def _pca_stats(bundle: MarketDataBundle, window: int = WINDOW) -> pl.DataFrame:
    """(date, ar1, ncomp80, pc1_dispersion) per rolling window."""
    dates, rets = _sector_returns(bundle)
    rows: list[tuple[date, float, int, float]] = []
    for i in range(window - 1, len(dates)):
        x = rets[i - window + 1 : i + 1]
        sd = x.std(axis=0, ddof=1)
        if (sd < 1e-12).any():
            continue  # degenerate window (a flat series): no stable correlation
        corr = np.corrcoef(x.T)
        eig, vec = np.linalg.eigh(corr)
        eig = np.clip(eig[::-1], 0.0, None)
        total = eig.sum()
        cum = np.cumsum(eig) / total
        pc1 = vec[:, -1]
        rows.append(
            (
                dates[i],
                float(eig[0] / total),
                int(np.searchsorted(cum, 0.8) + 1),
                float(np.std(np.abs(pc1))),
            )
        )
    return pl.DataFrame(
        rows,
        schema={"date": pl.Date, "ar1": pl.Float64, "ncomp80": pl.Int64, "disp": pl.Float64},
        orient="row",
    )


@feature(
    "absorption_ratio",
    F,
    f"share of sector-return correlation explained by the first eigenvalue "
    f"({WINDOW}-session rolling window)",
    (SECTORS_SENTINEL,),
    availability=f"needs {WINDOW}+ sessions of sector history",
)
def absorption_ratio(b: MarketDataBundle) -> pl.DataFrame:
    return _pca_stats(b).select(pl.col("date"), pl.col("ar1").alias("value"))


@feature(
    "absorption_chg_20d",
    F,
    "20-session change in absorption ratio",
    (SECTORS_SENTINEL,),
)
def absorption_chg_20d(b: MarketDataBundle) -> pl.DataFrame:
    s = _pca_stats(b)
    return s.select(pl.col("date"), (pl.col("ar1") - pl.col("ar1").shift(20)).alias("value"))


@feature(
    "ncomp_80pct",
    F,
    "number of eigen-components explaining 80% of sector correlation variance",
    (SECTORS_SENTINEL,),
)
def ncomp_80pct(b: MarketDataBundle) -> pl.DataFrame:
    return _pca_stats(b).select(pl.col("date"), pl.col("ncomp80").cast(pl.Float64).alias("value"))


@feature(
    "pc1_loading_dispersion",
    F,
    "std of |first-eigenvector loadings| (uniformity of the market factor)",
    (SECTORS_SENTINEL,),
)
def pc1_loading_dispersion(b: MarketDataBundle) -> pl.DataFrame:
    return _pca_stats(b).select(pl.col("date"), pl.col("disp").alias("value"))
