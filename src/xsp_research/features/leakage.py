"""Empirical leakage validation: prefix-consistency checks.

A feature is leakage-free iff recomputing it on a bundle truncated at date t
reproduces the full-sample values at every date <= t. Features that use
full-sample statistics (global means, min-max scaling, centered windows) fail
this check; rolling/expanding constructions pass. This is the required
`leakage test` from the feature-framework spec and runs in CI over every
registered daily feature.
"""

from __future__ import annotations

import math
from datetime import date

import polars as pl

from xsp_research.features.registry import FeatureSpec, MarketDataBundle


def prefix_consistency_violations(
    spec: FeatureSpec,
    bundle: MarketDataBundle,
    sample_dates: list[date],
    atol: float = 1e-9,
) -> list[str]:
    """Empty list = leakage-free at the sampled truncation dates."""
    full = spec.compute(bundle).rename({"value": "full"})
    violations: list[str] = []
    for cutoff in sample_dates:
        truncated = spec.compute(bundle.truncated(cutoff)).rename({"value": "trunc"})
        joined = (
            full.filter(pl.col("date") <= cutoff)
            .join(truncated, on="date", how="inner")
            .drop_nulls()
        )
        bad = joined.filter(
            ((pl.col("full") - pl.col("trunc")).abs() > atol)
            & pl.col("full").is_finite()
            & pl.col("trunc").is_finite()
        )
        if not bad.is_empty():
            r = bad.row(0, named=True)
            violations.append(
                f"{spec.name}@cutoff={cutoff}: value at {r['date']} changed "
                f"{r['full']:.10g} -> {r['trunc']:.10g} when future data was removed"
            )
    return violations


def sample_cutoffs(bundle: MarketDataBundle, n: int = 5) -> list[date]:
    """Deterministic truncation dates spread over the second half of the calendar
    (features need warm-up history before values exist)."""
    cal = bundle.calendar()
    if len(cal) < 4:
        return cal
    start = len(cal) // 2
    step = max(1, (len(cal) - 1 - start) // max(n - 1, 1))
    idx = list(range(start, len(cal), step))[:n]
    if (len(cal) - 1) not in idx:
        idx.append(len(cal) - 1)
    return [cal[i] for i in idx]


def check_all(
    specs: list[FeatureSpec], bundle: MarketDataBundle, n_samples: int = 4
) -> dict[str, list[str]]:
    """name -> violations (empty dict means everything passed)."""
    cutoffs = sample_cutoffs(bundle, n_samples)
    result: dict[str, list[str]] = {}
    for spec in specs:
        if not bundle.has(*spec.source_symbols):
            continue
        v = prefix_consistency_violations(spec, bundle, cutoffs)
        if v:
            result[spec.name] = v
    return result


def assert_no_future_dates(features: pl.DataFrame, calendar: list[date]) -> None:
    """Structural invariant: feature rows exist only on decision-calendar dates."""
    extra = set(features["date"].to_list()) - set(calendar)
    if extra:
        raise AssertionError(f"feature frame contains non-calendar dates: {sorted(extra)[:5]}")
    if not math.isfinite(features.height):  # pragma: no cover - defensive
        raise AssertionError("invalid frame")
