"""Leakage validation: every registered daily feature must be prefix-consistent,
and the checker itself must catch a deliberately leaky feature."""

from datetime import date

import polars as pl
import pytest

from xsp_research.features import REGISTRY, FeatureFamily, build_features
from xsp_research.features.leakage import (
    assert_no_future_dates,
    check_all,
    prefix_consistency_violations,
    sample_cutoffs,
)
from xsp_research.features.registry import FeatureSpec
from xsp_research.ingestion.synthetic import SyntheticConfig, SyntheticMarket, research_bundle


@pytest.fixture(scope="module")
def bundle():
    market = SyntheticMarket(SyntheticConfig(start=date(2023, 1, 2), end=date(2024, 6, 28)))
    return research_bundle(market)


def test_all_registered_features_prefix_consistent(bundle):
    """The core leakage guarantee: removing future data never changes past values."""
    violations = check_all(list(REGISTRY.values()), bundle, n_samples=3)
    assert violations == {}, violations


def test_checker_catches_global_statistic_leak(bundle):
    """A feature normalized by the FULL-SAMPLE mean leaks the future; the
    checker must flag it (validates the test, not just the features)."""

    def leaky(b):
        df = b.get("UNDERLYING")
        full_mean = df["close"].mean()  # <- uses future observations
        return df.select(pl.col("date"), (pl.col("close") / full_mean).alias("value"))

    spec = FeatureSpec(
        name="leaky_global_mean",
        family=FeatureFamily.TREND,
        definition="close / full-sample mean (deliberately leaky)",
        source_symbols=("UNDERLYING",),
        compute=leaky,
    )
    cutoffs = sample_cutoffs(bundle, 3)
    violations = prefix_consistency_violations(spec, bundle, cutoffs)
    assert violations, "checker failed to detect a global-statistic leak"


def test_feature_dates_stay_on_calendar(bundle):
    built = build_features(bundle, families=["volatility", "trend"])
    assert_no_future_dates(built.frame, bundle.calendar())


def test_truncated_bundle_really_truncates(bundle):
    cutoff = bundle.calendar()[100]
    t = bundle.truncated(cutoff)
    for sym in t.series:
        assert t.get(sym)["date"].max() <= cutoff
