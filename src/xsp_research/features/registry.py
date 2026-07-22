"""Feature registry: metadata-first feature definitions with enforced lagging.

Every feature is registered with:
  name, family, definition, source symbols, required lag (in sessions),
  missing-value policy, availability note, and a compute function.

Timestamp convention (critical, documented once here):
  A feature's compute function receives a MarketDataBundle and returns a frame
  of (date, value) where the value at data-date d uses ONLY information from
  sessions <= d. The builder then shifts each series by `required_lag_sessions`
  on the decision calendar, so the feature value seen on decision date t comes
  from data through session t - lag. The default lag is 1 session because the
  engine makes decisions at ~15:30 ET, BEFORE that session's official close is
  known — daily-close-derived features must therefore come from the prior
  session. Leakage is checked two ways: structurally (the shift) and
  empirically (prefix-consistency tests in features/leakage.py).
"""

from __future__ import annotations

import enum
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Any

import polars as pl


class FeatureFamily(enum.StrEnum):
    VOLATILITY = "volatility"
    TREND = "trend"
    BREADTH = "breadth"
    CROSS_ASSET = "cross_asset"
    PCA_REGIME = "pca_regime"
    SURFACE = "surface"  # snapshot-based; computed at entry, not by the builder


SECTORS_SENTINEL = "__SECTORS__"


@dataclass(slots=True)
class MarketDataBundle:
    """Daily research inputs: one (date, close) frame per symbol.

    ``UNDERLYING`` is the decision calendar's primary series. Sector ETFs used
    by PCA features are listed in ``sector_symbols`` and must each have a
    series entry.
    """

    series: dict[str, pl.DataFrame]
    sector_symbols: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        for sym, df in self.series.items():
            if set(df.columns) < {"date", "close"}:
                raise ValueError(f"series {sym} must have (date, close) columns")
        missing = [s for s in self.sector_symbols if s not in self.series]
        if missing:
            raise ValueError(f"sector symbols missing from series: {missing}")

    def get(self, symbol: str) -> pl.DataFrame:
        return self.series[symbol].select(["date", "close"]).sort("date")

    def has(self, *symbols: str) -> bool:
        return all(
            (s == SECTORS_SENTINEL and len(self.sector_symbols) >= 2) or s in self.series
            for s in symbols
        )

    def calendar(self) -> list[date]:
        return self.get("UNDERLYING")["date"].to_list()

    def truncated(self, cutoff: date) -> MarketDataBundle:
        """Bundle containing only observations at or before ``cutoff``.

        Used by leakage checks: recomputing a feature on the truncated bundle
        must reproduce the full-sample values at dates <= cutoff.
        """
        return MarketDataBundle(
            series={s: df.filter(pl.col("date") <= cutoff) for s, df in self.series.items()},
            sector_symbols=self.sector_symbols,
        )


ComputeFn = Callable[[MarketDataBundle], pl.DataFrame]  # -> (date, value)


@dataclass(frozen=True, slots=True)
class FeatureSpec:
    name: str
    family: FeatureFamily
    definition: str
    source_symbols: tuple[str, ...]
    compute: ComputeFn
    required_lag_sessions: int = 1
    missing_policy: str = "asof_5"  # "exact" | "asof" | "asof_N" (N-day as-of tolerance)
    availability_note: str = ""

    def to_manifest(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "family": self.family.value,
            "definition": self.definition,
            "source_symbols": list(self.source_symbols),
            "required_lag_sessions": self.required_lag_sessions,
            "missing_policy": self.missing_policy,
            "availability_note": self.availability_note,
        }


REGISTRY: dict[str, FeatureSpec] = {}


def feature(
    name: str,
    family: FeatureFamily,
    definition: str,
    sources: tuple[str, ...],
    lag: int = 1,
    missing: str = "asof_5",
    availability: str = "",
) -> Callable[[ComputeFn], ComputeFn]:
    """Decorator registering a compute function as a feature."""

    def deco(fn: ComputeFn) -> ComputeFn:
        if name in REGISTRY:
            raise ValueError(f"duplicate feature name: {name}")
        if lag < 1 and family is not FeatureFamily.SURFACE:
            raise ValueError(
                f"{name}: daily features require lag >= 1 (close unknown at decision time)"
            )
        REGISTRY[name] = FeatureSpec(
            name=name,
            family=family,
            definition=definition,
            source_symbols=tuple(sources),
            compute=fn,
            required_lag_sessions=lag,
            missing_policy=missing,
            availability_note=availability,
        )
        return fn

    return deco


def select_specs(
    families: list[str] | None = None, names: list[str] | None = None
) -> list[FeatureSpec]:
    specs = list(REGISTRY.values())
    if families is not None:
        fams = {FeatureFamily(f) for f in families}
        specs = [s for s in specs if s.family in fams]
    if names is not None:
        wanted = set(names)
        specs += [REGISTRY[n] for n in wanted - {s.name for s in specs} if n in REGISTRY]
        specs = [s for s in specs if s.name in wanted or (families and s.family in fams)]
    return sorted(specs, key=lambda s: s.name)


@dataclass(slots=True)
class BuildResult:
    frame: pl.DataFrame  # (date, <feature columns>) on the decision calendar
    used: list[FeatureSpec]
    skipped: dict[str, str] = field(default_factory=dict)  # name -> reason

    def manifest(self) -> dict[str, Any]:
        return {
            "features": [s.to_manifest() for s in self.used],
            "skipped": self.skipped,
            "n_rows": self.frame.height,
            "date_range": [str(self.frame["date"].min()), str(self.frame["date"].max())],
        }


def _align(cal: pl.DataFrame, raw: pl.DataFrame, policy: str) -> pl.Series:
    raw = raw.sort("date")
    if policy == "exact":
        joined = cal.join(raw, on="date", how="left")
    else:
        tolerance = None
        if policy.startswith("asof_"):
            tolerance = timedelta(days=int(policy.split("_")[1]))
        joined = cal.join_asof(raw, on="date", strategy="backward", tolerance=tolerance)
    return joined["value"]


def build_features(
    bundle: MarketDataBundle,
    families: list[str] | None = None,
    names: list[str] | None = None,
) -> BuildResult:
    """Compute selected daily features on the bundle's decision calendar.

    Surface-family specs are excluded (they are snapshot-based and computed at
    entry time by the experiment runner's hook).
    """
    specs = [s for s in select_specs(families, names) if s.family is not FeatureFamily.SURFACE]
    cal = pl.DataFrame({"date": bundle.calendar()}).sort("date")
    out = cal
    used: list[FeatureSpec] = []
    skipped: dict[str, str] = {}
    for spec in specs:
        if not bundle.has(*spec.source_symbols):
            skipped[spec.name] = f"missing sources {list(spec.source_symbols)}"
            continue
        raw = spec.compute(bundle)
        if set(raw.columns) != {"date", "value"}:
            raise ValueError(f"feature {spec.name} compute must return (date, value)")
        aligned = _align(cal, raw, spec.missing_policy)
        out = out.with_columns(aligned.shift(spec.required_lag_sessions).alias(spec.name))
        used.append(spec)
    return BuildResult(frame=out, used=used, skipped=skipped)
