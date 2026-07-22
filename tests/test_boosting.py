"""Nested-tuned boosting: leakage-safe inner splits, admission comparison."""

from datetime import date, timedelta

import numpy as np

from tests.test_models_eval import research_dataset  # noqa: F401 (shared fixture)
from xsp_research.models.boosting import (
    DEFAULT_GRID,
    XGBClassifierModel,
    XGBRegressorModel,
    _inner_split,
    nested_boosting_walkforward,
)
from xsp_research.models.walkforward import WalkForwardConfig

WF = WalkForwardConfig(n_folds=3, embargo_days=5, min_train_size=10)


class TestXGBWrappers:
    def test_classifier_learns_separable(self):
        rng = np.random.default_rng(0)
        x = rng.normal(0, 1, (400, 3))
        y = (x[:, 0] > 0).astype(float)
        m = XGBClassifierModel(max_depth=2, n_estimators=50, learning_rate=0.1)
        m.fit(x, y)
        p = m.predict_proba1(x)
        assert np.mean((p > 0.5) == (y == 1)) > 0.95

    def test_classifier_handles_nan_natively(self):
        rng = np.random.default_rng(1)
        x = rng.normal(0, 1, (200, 3))
        x[::5, 1] = np.nan
        y = (x[:, 0] > 0).astype(float)
        m = XGBClassifierModel(max_depth=2, n_estimators=50, learning_rate=0.1)
        m.fit(x, y)
        assert np.all(np.isfinite(m.predict_proba1(x)))

    def test_single_class_degrades_to_base_rate(self):
        m = XGBClassifierModel(max_depth=2, n_estimators=50, learning_rate=0.1)
        x = np.zeros((20, 2))
        m.fit(x, np.ones(20))
        assert np.allclose(m.predict_proba1(x), 1.0)

    def test_regressor_fits(self):
        rng = np.random.default_rng(2)
        x = rng.normal(0, 1, (300, 2))
        y = 3.0 * x[:, 0] + rng.normal(0, 0.1, 300)
        m = XGBRegressorModel(max_depth=3, n_estimators=150, learning_rate=0.1)
        m.fit(x, y)
        pred = m.predict(x)
        assert np.corrcoef(pred, y)[0, 1] > 0.9


class TestInnerSplit:
    def test_inner_split_is_chronological_and_purged(self):
        n = 60
        entries = [date(2022, 1, 3) + timedelta(days=7 * i) for i in range(n)]
        ends = [e + timedelta(days=30) for e in entries]
        train_idx = np.arange(n)
        split = _inner_split(train_idx, entries, ends, inner_frac=0.25, embargo_days=5)
        assert split is not None
        inner_train, inner_val = split
        val_start = min(entries[i] for i in inner_val)
        # Purge: every inner-training label window ends before the embargoed start.
        for i in inner_train:
            assert ends[i] < val_start - timedelta(days=4)
        # Chronology: all validation entries after all training entries.
        assert max(entries[i] for i in inner_train) < val_start

    def test_too_small_returns_none(self):
        entries = [date(2022, 1, 3) + timedelta(days=7 * i) for i in range(8)]
        ends = [e + timedelta(days=30) for e in entries]
        assert _inner_split(np.arange(8), entries, ends, 0.25, 5) is None


class TestNestedWalkforward:
    def test_end_to_end_structure(self, research_dataset):  # noqa: F811
        from xsp_research.models.evaluate import feature_family_columns

        fams = feature_family_columns(research_dataset)
        cols = sorted({c for v in fams.values() for c in v})
        out = nested_boosting_walkforward(research_dataset, cols, "label_expire_itm", WF)
        assert "error" not in out, out
        assert len(out["chosen_params_per_fold"]) == out["n_folds"]
        for c in out["chosen_params_per_fold"]:
            assert {k: v for k, v in c.items() if k != "fold"} in DEFAULT_GRID
        assert "base_rate" in out["benchmarks"] and "logistic" in out["benchmarks"]
        adm = out["admission"]
        assert adm["metric"] == "brier"
        assert isinstance(adm["boosting_beats_benchmarks"], bool)

    def test_regression_target(self, research_dataset):  # noqa: F811
        from xsp_research.models.evaluate import BASELINE_FEATURES

        out = nested_boosting_walkforward(
            research_dataset, list(BASELINE_FEATURES), "label_net_pnl", WF
        )
        assert "error" not in out, out
        assert out["admission"]["metric"] == "mae"
        assert "train_mean" in out["benchmarks"]

    def test_deterministic(self, research_dataset):  # noqa: F811
        from xsp_research.models.evaluate import BASELINE_FEATURES

        cols = list(BASELINE_FEATURES)
        a = nested_boosting_walkforward(research_dataset, cols, "label_expire_itm", WF)
        b = nested_boosting_walkforward(research_dataset, cols, "label_expire_itm", WF)
        assert a["boosting"]["brier"] == b["boosting"]["brier"]
        assert a["chosen_params_per_fold"] == b["chosen_params_per_fold"]

    def test_on_noise_boosting_should_not_beat_base_rate(self, research_dataset):  # noqa: F811
        """Synthetic GBM has no signal: the admission rule should (almost
        always) reject boosting. This guards the guardrail."""
        from xsp_research.models.evaluate import feature_family_columns

        fams = feature_family_columns(research_dataset)
        cols = sorted({c for v in fams.values() for c in v})
        out = nested_boosting_walkforward(research_dataset, cols, "label_expire_itm", WF)
        assert out["admission"]["boosting_beats_benchmarks"] is False


def test_param_stability_summary():
    from xsp_research.models.boosting import _param_stability

    chosen = [
        {"fold": 1, "max_depth": 2, "n_estimators": 50},
        {"fold": 2, "max_depth": 2, "n_estimators": 150},
    ]
    stab = _param_stability(chosen)
    assert stab["max_depth"]["n_unique"] == 1
    assert stab["n_estimators"]["n_unique"] == 2
