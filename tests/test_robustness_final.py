"""Robustness bootstraps, delta/width grid, and the final-test usage guard."""

from datetime import date

import numpy as np
import pytest

from tests.test_engine import make_config
from tests.test_models_eval import research_dataset  # noqa: F401 (shared fixture)
from xsp_research.backtest.engine import BacktestEngine
from xsp_research.config import BacktestConfig
from xsp_research.evaluation.final_test import FinalTestReuseError, run_final_test
from xsp_research.evaluation.robustness import (
    bootstrap_mean_ci,
    bootstrap_sharpe_ci,
    delta_width_grid,
    result_robustness,
)
from xsp_research.ingestion.synthetic import SyntheticConfig, SyntheticMarket
from xsp_research.models.probability_model import probability_model_factories
from xsp_research.models.walkforward import WalkForwardConfig


class TestBootstrapCI:
    def test_constant_series_collapses(self):
        ci = bootstrap_mean_ci(np.full(50, 3.0), n_boot=200)
        assert ci["mean"] == ci["ci_low"] == ci["ci_high"] == 3.0

    def test_mean_within_ci_and_ordered(self):
        rng = np.random.default_rng(0)
        vals = rng.normal(5.0, 2.0, 200)
        ci = bootstrap_mean_ci(vals, n_boot=500, block=5)
        assert ci["ci_low"] <= ci["mean"] <= ci["ci_high"]
        assert ci["ci_low"] < ci["ci_high"]

    def test_deterministic_with_seed(self):
        vals = np.random.default_rng(1).normal(0, 1, 100)
        a = bootstrap_mean_ci(vals, n_boot=300, seed=42)
        b = bootstrap_mean_ci(vals, n_boot=300, seed=42)
        assert a == b

    def test_empty_input(self):
        assert "error" in bootstrap_mean_ci([])

    def test_sharpe_ci_structure(self):
        rets = np.random.default_rng(2).normal(0.0005, 0.01, 500)
        ci = bootstrap_sharpe_ci(rets, n_boot=300)
        assert ci["ci_low"] <= ci["sharpe"] <= ci["ci_high"]

    def test_sharpe_too_few_obs(self):
        assert "error" in bootstrap_sharpe_ci(np.ones(10))


class TestResultRobustness:
    def test_panel_on_synthetic_run(self):
        start, end = date(2023, 1, 2), date(2023, 9, 29)
        market = SyntheticMarket(SyntheticConfig(start=start, end=end, strike_pct_range=0.12))
        result = BacktestEngine(make_config(), market, market, market, "base").run()
        panel = result_robustness(result, n_boot=300)
        assert panel["data_source"] == "synthetic"
        assert "expectancy_per_spread_ci" in panel
        assert isinstance(panel["expectancy_includes_zero"], bool)


class TestDeltaWidthGrid:
    def test_grid_runs_all_combos(self):
        start, end = date(2023, 1, 2), date(2023, 6, 30)
        market = SyntheticMarket(SyntheticConfig(start=start, end=end, strike_pct_range=0.15))
        cfg = make_config(backtest=BacktestConfig(start=start, end=end, initial_cash=100_000.0))
        grid = delta_width_grid(cfg, market, market, market, deltas=(0.15, 0.25), widths=(2.0, 5.0))
        assert grid.height == 4
        assert set(zip(grid["short_delta"].to_list(), grid["width"].to_list(), strict=True)) == {
            (0.15, 2.0),
            (0.15, 5.0),
            (0.25, 2.0),
            (0.25, 5.0),
        }
        assert (grid["data_source"] == "synthetic").all()
        # Structural invariant: no trade can lose more than the spread width
        # (x$100) plus round-turn fees, whatever the config.
        for row in grid.iter_rows(named=True):
            if row["n_trades"] > 0:
                assert row["worst_trade_net"] >= -(row["width"] * 100.0 + 20.0)


class TestFinalTestGuard:
    FEATURES = ["surf_short_delta", "surf_dte", "surf_credit_over_width"]

    def _cfg(self):
        return WalkForwardConfig(
            n_folds=3, embargo_days=5, min_train_size=10, final_test_start=date(2024, 7, 1)
        )

    def test_runs_once_then_blocks(self, research_dataset, tmp_path):  # noqa: F811
        usage = tmp_path / "usage.json"
        factory = probability_model_factories()["logistic"]
        out = run_final_test(
            research_dataset,
            self.FEATURES,
            "label_expire_itm",
            self._cfg(),
            factory,
            usage_file=usage,
        )
        assert "final_test" in out and out["usage_record"]["n_test"] > 0
        assert usage.exists()
        with pytest.raises(FinalTestReuseError, match="already evaluated"):
            run_final_test(
                research_dataset,
                self.FEATURES,
                "label_expire_itm",
                self._cfg(),
                factory,
                usage_file=usage,
            )

    def test_force_records_reuse(self, research_dataset, tmp_path):  # noqa: F811
        import json

        usage = tmp_path / "usage.json"
        factory = probability_model_factories()["base_rate"]
        run_final_test(
            research_dataset,
            self.FEATURES,
            "label_expire_itm",
            self._cfg(),
            factory,
            usage_file=usage,
        )
        out = run_final_test(
            research_dataset,
            self.FEATURES,
            "label_expire_itm",
            self._cfg(),
            factory,
            usage_file=usage,
            force=True,
        )
        assert out["usage_record"]["forced_reuse"] is True
        records = json.loads(usage.read_text())
        assert len(records) == 2

    def test_different_labels_tracked_separately(self, research_dataset, tmp_path):  # noqa: F811
        usage = tmp_path / "usage.json"
        run_final_test(
            research_dataset,
            self.FEATURES,
            "label_expire_itm",
            self._cfg(),
            probability_model_factories()["base_rate"],
            usage_file=usage,
        )
        # A different label is a fresh final test, not a reuse.
        from xsp_research.models.pnl_model import regression_model_factories

        out = run_final_test(
            research_dataset,
            self.FEATURES,
            "label_net_pnl",
            self._cfg(),
            regression_model_factories()["train_mean"],
            usage_file=usage,
        )
        assert out["usage_record"]["forced_reuse"] is False

    def test_requires_final_test_start(self, research_dataset, tmp_path):  # noqa: F811
        with pytest.raises(ValueError, match="final_test_start"):
            run_final_test(
                research_dataset,
                self.FEATURES,
                "label_expire_itm",
                WalkForwardConfig(n_folds=3),
                probability_model_factories()["base_rate"],
                usage_file=tmp_path / "u.json",
            )
