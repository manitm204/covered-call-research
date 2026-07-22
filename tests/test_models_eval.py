"""Benchmark models + end-to-end walk-forward evaluation and ablations."""

from datetime import date

import numpy as np
import polars as pl
import pytest

from tests.test_engine import make_config
from xsp_research.backtest.engine import BacktestEngine
from xsp_research.config import BacktestConfig, EntryFrequency
from xsp_research.experiments.runner import ExperimentConfig, build_research_dataset
from xsp_research.features import build_features
from xsp_research.ingestion.synthetic import SyntheticConfig, SyntheticMarket, research_bundle
from xsp_research.models.evaluate import (
    BASELINE_FEATURES,
    feature_family_columns,
    run_family_ablation,
    run_walkforward,
)
from xsp_research.models.pnl_model import regression_model_factories
from xsp_research.models.probability_model import probability_model_factories
from xsp_research.models.walkforward import WalkForwardConfig

START, END = date(2023, 1, 2), date(2024, 12, 31)


@pytest.fixture(scope="module")
def research_dataset():
    """Weekly hold-to-expiry synthetic run -> research dataset with labels."""
    market = SyntheticMarket(SyntheticConfig(start=START, end=END, strike_pct_range=0.12))
    bundle = research_bundle(market)
    cfg = make_config(
        exits=[],
        backtest=BacktestConfig(
            start=START, end=END, initial_cash=100_000.0, entry_frequency=EntryFrequency.WEEKLY
        ),
    )
    cfg = cfg.model_copy(update={"sizing": cfg.sizing.model_copy(update={"max_open_positions": 6})})

    from xsp_research.features.surface import compute_surface_features

    def hook(session, snap_ts, chain, sel, spot, rate):
        return compute_surface_features(chain, sel, spot, rate, 0.0, session)

    result = BacktestEngine(cfg, market, market, market, "base", entry_feature_hook=hook).run()
    built = build_features(bundle)
    dataset = build_research_dataset(result, built.frame, market)
    assert dataset.height >= 60  # weekly entries over 2 years
    return dataset


WF = WalkForwardConfig(n_folds=3, embargo_days=5, min_train_size=10)


class TestBenchmarkModels:
    def test_base_rate_predicts_constant(self):
        m = probability_model_factories()["base_rate"]()
        y = np.array([1, 0, 0, 1, 1], dtype=float)
        m.fit(np.zeros((5, 2)), y)
        assert np.allclose(m.predict_proba1(np.zeros((3, 2))), 0.6)

    def test_logistic_learns_separable_data(self):
        rng = np.random.default_rng(0)
        x = rng.normal(0, 1, (300, 3))
        y = (x[:, 0] > 0).astype(float)
        m = probability_model_factories()["logistic"]()
        m.fit(x, y)
        p = m.predict_proba1(x)
        assert np.mean((p > 0.5) == (y == 1)) > 0.95

    def test_models_handle_nans_via_imputation(self):
        rng = np.random.default_rng(1)
        x = rng.normal(0, 1, (100, 3))
        x[::7, 1] = np.nan
        y = (x[:, 0] > 0).astype(float)
        for name, factory in probability_model_factories().items():
            m = factory()
            m.fit(x, y)
            p = m.predict_proba1(x)
            assert np.all(np.isfinite(p)), name

    def test_degenerate_single_class_fold_degrades_to_base_rate(self):
        m = probability_model_factories()["logistic"]()
        x = np.random.default_rng(2).normal(0, 1, (20, 2))
        m.fit(x, np.zeros(20))
        assert np.allclose(m.predict_proba1(x), 0.0)
        m.fit(x, np.ones(20))
        assert np.allclose(m.predict_proba1(x), 1.0)

    def test_regression_train_mean(self):
        m = regression_model_factories()["train_mean"]()
        m.fit(np.zeros((4, 1)), np.array([1.0, 2.0, 3.0, 4.0]))
        assert np.allclose(m.predict(np.zeros((2, 1))), 2.5)


class TestWalkforwardEvaluation:
    def test_classification_run_structure(self, research_dataset):
        fams = feature_family_columns(research_dataset)
        cols = sorted({c for v in fams.values() for c in v})
        out = run_walkforward(research_dataset, cols, "label_expire_itm", WF)
        assert "error" not in out, out
        assert set(out["models"]) == set(probability_model_factories())
        for rep in out["models"].values():
            assert 0.0 <= rep["brier"] <= 1.0
            assert rep["pnl_by_predicted_decile"]
            assert "always_trade_mean_pnl" in rep

    def test_regression_run_structure(self, research_dataset):
        out = run_walkforward(research_dataset, list(BASELINE_FEATURES), "label_net_pnl", WF)
        assert "error" not in out, out
        assert set(out["models"]) == set(regression_model_factories())
        for rep in out["models"].values():
            assert rep["mae"] >= 0 and rep["rmse"] >= rep["mae"] * 0.99

    def test_final_test_reduces_samples(self, research_dataset):
        cols = list(BASELINE_FEATURES)
        full = run_walkforward(research_dataset, cols, "label_expire_itm", WF)
        held = run_walkforward(
            research_dataset,
            cols,
            "label_expire_itm",
            WalkForwardConfig(
                n_folds=3, embargo_days=5, min_train_size=10, final_test_start=date(2024, 7, 1)
            ),
        )
        if "error" not in held and "error" not in full:
            pooled_full = sum(f["val_n"] for f in full["folds"])
            pooled_held = sum(f["val_n"] for f in held["folds"])
            assert pooled_held < pooled_full

    def test_deterministic(self, research_dataset):
        cols = list(BASELINE_FEATURES)
        a = run_walkforward(research_dataset, cols, "label_expire_itm", WF)
        b = run_walkforward(research_dataset, cols, "label_expire_itm", WF)
        assert a["models"]["logistic"]["brier"] == b["models"]["logistic"]["brier"]


class TestAblation:
    def test_required_grid_present(self, research_dataset):
        out = run_family_ablation(research_dataset, "label_expire_itm", WF)
        sets = out["sets"]
        assert "baseline_only" in sets and "all_features" in sets
        fams = feature_family_columns(research_dataset)
        for fam in fams:
            assert f"baseline_plus_{fam}" in sets
            assert f"all_minus_{fam}" in sets
        ok = [s for s in sets.values() if "error" not in s]
        assert ok and all("brier" in s for s in ok)

    def test_unknown_model_rejected(self, research_dataset):
        with pytest.raises(ValueError, match="unknown model"):
            run_family_ablation(research_dataset, "label_expire_itm", WF, model_name="nope")


class TestRunnerIntegration:
    def test_experiment_dataset_contains_labels(self, tmp_path):
        """run_experiment now embeds label_* columns in research datasets."""
        import yaml

        from xsp_research.config import load_strategy_config
        from xsp_research.experiments.runner import run_experiment

        market = SyntheticMarket(
            SyntheticConfig(start=START, end=date(2023, 9, 29), strike_pct_range=0.12)
        )
        bundle = research_bundle(market)
        raw = load_strategy_config("configs/strategy_hold_to_expiry.yaml").model_dump(mode="json")
        raw["backtest"]["start"] = str(START)
        raw["backtest"]["end"] = str(date(2023, 9, 29))
        cfg_path = tmp_path / "s.yaml"
        cfg_path.write_text(yaml.safe_dump(raw))
        exp = ExperimentConfig(
            name="label_gen_test", strategy_config=str(cfg_path), scenarios=["base"]
        )
        run = run_experiment(exp, market, market, market, bundle, out_root=tmp_path)
        assert run.errors == []
        ds = pl.read_parquet(run.out_dir / "base" / "research_dataset.parquet")
        assert "label_expire_itm" in ds.columns
        assert "label_end" in ds.columns
        assert ds["label_expire_itm"].null_count() < ds.height  # labels populated
