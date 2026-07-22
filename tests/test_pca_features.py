"""PCA regime features: known-structure sanity checks."""

from datetime import date, timedelta

import numpy as np
import polars as pl

from xsp_research.features import MarketDataBundle, build_features


def weekday_dates(start: date, n: int) -> list[date]:
    out, d = [], start
    while len(out) < n:
        if d.weekday() < 5:
            out.append(d)
        d += timedelta(days=1)
    return out


def sector_bundle(rets_by_sector: dict[str, np.ndarray]) -> MarketDataBundle:
    n = len(next(iter(rets_by_sector.values())))
    dates = weekday_dates(date(2022, 1, 3), n + 1)
    series = {"UNDERLYING": pl.DataFrame({"date": dates, "close": [100.0] * (n + 1)})}
    for sym, rets in rets_by_sector.items():
        closes = 100.0 * np.cumprod(np.concatenate(([1.0], 1.0 + rets)))
        series[sym] = pl.DataFrame({"date": dates, "close": closes})
    return MarketDataBundle(series=series, sector_symbols=tuple(rets_by_sector))


N = 200
K = 8


def test_perfectly_correlated_sectors_absorption_near_one():
    rng = np.random.default_rng(3)
    common = rng.normal(0, 0.01, N)
    bundle = sector_bundle({f"S{i}": common * (1 + 0.1 * i) for i in range(K)})
    built = build_features(bundle, names=["absorption_ratio", "ncomp_80pct"])
    ar = [v for v in built.frame["absorption_ratio"].to_list() if v is not None]
    nc = [v for v in built.frame["ncomp_80pct"].to_list() if v is not None]
    assert ar and min(ar) > 0.999  # one factor explains everything
    assert nc and max(nc) == 1.0


def test_independent_sectors_absorption_near_uniform():
    rng = np.random.default_rng(4)
    bundle = sector_bundle({f"S{i}": rng.normal(0, 0.01, N) for i in range(K)})
    built = build_features(bundle, names=["absorption_ratio", "ncomp_80pct"])
    ar = [v for v in built.frame["absorption_ratio"].to_list() if v is not None]
    nc = [v for v in built.frame["ncomp_80pct"].to_list() if v is not None]
    # With 8 independent series, first eigenvalue share ~1/8 (sampling noise up).
    assert ar and 1.0 / K <= np.mean(ar) < 0.35
    assert nc and np.mean(nc) >= 5.0


def test_absorption_bounded():
    rng = np.random.default_rng(5)
    common = rng.normal(0, 0.008, N)
    bundle = sector_bundle({f"S{i}": 0.7 * common + rng.normal(0, 0.006, N) for i in range(K)})
    built = build_features(bundle, names=["absorption_ratio"])
    ar = [v for v in built.frame["absorption_ratio"].to_list() if v is not None]
    assert all(1.0 / K - 1e-9 <= v <= 1.0 + 1e-9 for v in ar)


def test_requires_at_least_two_sectors():
    rng = np.random.default_rng(6)
    bundle = sector_bundle({"S0": rng.normal(0, 0.01, N)})
    built = build_features(bundle, names=["absorption_ratio"])
    assert "absorption_ratio" in built.skipped
