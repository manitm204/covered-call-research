"""Daily feature families: hand-computed values, lag enforcement, availability."""

from datetime import date, timedelta

import polars as pl
import pytest

from xsp_research.features import REGISTRY, FeatureFamily, MarketDataBundle, build_features
from xsp_research.features.registry import FeatureSpec, feature
from xsp_research.ingestion.synthetic import SyntheticConfig, SyntheticMarket, research_bundle


def weekday_dates(start: date, n: int) -> list[date]:
    out, d = [], start
    while len(out) < n:
        if d.weekday() < 5:
            out.append(d)
        d += timedelta(days=1)
    return out


def geometric_bundle(n: int = 300, daily: float = 1.001) -> MarketDataBundle:
    dates = weekday_dates(date(2023, 1, 2), n)
    closes = [100.0 * daily**i for i in range(n)]
    frame = pl.DataFrame({"date": dates, "close": closes})
    return MarketDataBundle(series={"UNDERLYING": frame})


@pytest.fixture(scope="module")
def synth_bundle():
    market = SyntheticMarket(SyntheticConfig(start=date(2023, 1, 2), end=date(2024, 6, 28)))
    return research_bundle(market)


class TestRegistryMetadata:
    def test_every_feature_fully_documented(self):
        for spec in REGISTRY.values():
            assert spec.definition, spec.name
            assert spec.source_symbols, spec.name
            assert spec.required_lag_sessions >= 1, f"{spec.name}: daily lag must be >= 1"

    def test_families_populated(self):
        fams = {s.family for s in REGISTRY.values()}
        assert {
            FeatureFamily.VOLATILITY,
            FeatureFamily.TREND,
            FeatureFamily.BREADTH,
            FeatureFamily.CROSS_ASSET,
            FeatureFamily.PCA_REGIME,
        } <= fams


class TestLagEnforcement:
    def test_decision_date_uses_prior_session_value(self):
        """With lag=1, the feature value on decision date t must equal the raw
        computed value at session t-1 — verified with an identity feature."""
        bundle = geometric_bundle(20)
        name = "test_identity_close"
        if name not in REGISTRY:

            @feature(name, FeatureFamily.TREND, "close itself (test)", ("UNDERLYING",))
            def _identity(b: MarketDataBundle) -> pl.DataFrame:
                return b.get("UNDERLYING").rename({"close": "value"})

        try:
            built = build_features(bundle, names=[name])
            closes = bundle.get("UNDERLYING")
            # row i of features corresponds to calendar[i]; value must be close[i-1]
            feats = built.frame[name].to_list()
            raw = closes["close"].to_list()
            assert feats[0] is None
            assert feats[1:] == pytest.approx(raw[:-1])
        finally:
            REGISTRY.pop(name, None)

    def test_lag_zero_daily_feature_rejected(self):
        with pytest.raises(ValueError, match="lag >= 1"):

            @feature("bad_lag0", FeatureFamily.TREND, "x", ("UNDERLYING",), lag=0)
            def _bad(b):  # pragma: no cover
                return None

        REGISTRY.pop("bad_lag0", None)


class TestHandComputedValues:
    def test_ret_5d_on_geometric_series(self):
        bundle = geometric_bundle(30)
        built = build_features(bundle, names=["ret_5d"])
        # value at decision t = return through t-1 = 1.001^5 - 1
        val = built.frame["ret_5d"][10]
        assert val == pytest.approx(1.001**5 - 1.0, rel=1e-9)

    def test_dist_from_ath_zero_on_monotone_series(self):
        bundle = geometric_bundle(30)
        built = build_features(bundle, names=["dist_from_ath"])
        assert built.frame["dist_from_ath"][-1] == pytest.approx(0.0, abs=1e-12)

    def test_rsi_100_on_monotone_up(self):
        bundle = geometric_bundle(40)
        built = build_features(bundle, names=["rsi_14"])
        assert built.frame["rsi_14"][-1] == pytest.approx(100.0)

    def test_rv_positive_and_annualized_scale(self, synth_bundle):
        built = build_features(synth_bundle, names=["rv_20"])
        vals = [v for v in built.frame["rv_20"].to_list() if v is not None]
        assert all(v >= 0 for v in vals)
        assert 0.01 < sum(vals) / len(vals) < 1.0  # plausible annualized equity vol


class TestSyntheticBundleBuild:
    def test_all_families_build_without_skips(self, synth_bundle):
        built = build_features(synth_bundle)
        assert built.skipped == {}
        assert built.frame.height == len(synth_bundle.calendar())

    def test_percentile_bounded(self, synth_bundle):
        built = build_features(synth_bundle, names=["vix_pct_252"])
        vals = [v for v in built.frame["vix_pct_252"].to_list() if v is not None]
        assert vals and all(0.0 <= v <= 1.0 for v in vals)

    def test_manifest_metadata(self, synth_bundle):
        built = build_features(synth_bundle, families=["volatility"])
        manifest = built.manifest()
        names = {f["name"] for f in manifest["features"]}
        assert "vix_level" in names
        assert all(f["required_lag_sessions"] >= 1 for f in manifest["features"])


class TestAvailability:
    def test_missing_source_skipped_with_reason(self):
        bundle = geometric_bundle(50)  # no VIX series
        built = build_features(bundle, names=["vix_level", "ret_5d"])
        assert "vix_level" in built.skipped
        assert "VIX" in built.skipped["vix_level"]
        assert "ret_5d" in built.frame.columns

    def test_spec_is_frozen(self):
        spec = next(iter(REGISTRY.values()))
        assert isinstance(spec, FeatureSpec)
        with pytest.raises(AttributeError):
            spec.name = "x"
