"""Calibration and regression diagnostics: hand-computed examples."""

import numpy as np
import pytest

from xsp_research.models.calibration import (
    brier_score,
    calibration_slope_intercept,
    classification_report,
    expected_calibration_error,
    log_loss,
    outcome_by_predicted_decile,
    regression_report,
    reliability_table,
)


class TestBrierAndLogLoss:
    def test_brier_hand_computed(self):
        y = np.array([1.0, 0.0, 1.0, 0.0])
        p = np.array([0.8, 0.3, 0.6, 0.1])
        # ((0.2)^2 + (0.3)^2 + (0.4)^2 + (0.1)^2) / 4 = 0.30/4 = 0.075
        assert brier_score(y, p) == pytest.approx(0.075)

    def test_perfect_predictions(self):
        y = np.array([1.0, 0.0])
        assert brier_score(y, y) == 0.0
        assert log_loss(y, y) == pytest.approx(0.0, abs=1e-10)

    def test_log_loss_hand_computed(self):
        y = np.array([1.0])
        p = np.array([0.5])
        assert log_loss(y, p) == pytest.approx(np.log(2.0))

    def test_log_loss_clips_extremes(self):
        y = np.array([1.0])
        p = np.array([0.0])  # would be infinite without clipping
        assert np.isfinite(log_loss(y, p))


class TestCalibrationDiagnostics:
    def test_reliability_table_bins(self):
        rng = np.random.default_rng(0)
        p = rng.uniform(0, 1, 5000)
        y = (rng.uniform(0, 1, 5000) < p).astype(float)  # perfectly calibrated
        table = reliability_table(y, p, 10)
        assert len(table) == 10
        for row in table:
            assert row["observed_rate"] == pytest.approx(row["mean_predicted"], abs=0.08)

    def test_ece_near_zero_when_calibrated(self):
        rng = np.random.default_rng(1)
        p = rng.uniform(0, 1, 20000)
        y = (rng.uniform(0, 1, 20000) < p).astype(float)
        assert expected_calibration_error(y, p) < 0.02

    def test_ece_large_when_miscalibrated(self):
        rng = np.random.default_rng(2)
        y = (rng.uniform(0, 1, 5000) < 0.5).astype(float)
        p = np.full(5000, 0.9)  # confidently wrong
        assert expected_calibration_error(y, p) > 0.3

    def test_slope_near_one_when_calibrated(self):
        rng = np.random.default_rng(3)
        p = rng.uniform(0.05, 0.95, 20000)
        y = (rng.uniform(0, 1, 20000) < p).astype(float)
        cal = calibration_slope_intercept(y, p)
        assert cal["slope"] == pytest.approx(1.0, abs=0.1)
        assert cal["intercept"] == pytest.approx(0.0, abs=0.1)

    def test_degenerate_inputs_return_none(self):
        y = np.array([1.0, 0.0, 1.0])
        p = np.full(3, 0.5)  # constant predictions: slope unidentifiable
        cal = calibration_slope_intercept(y, p)
        assert cal["slope"] is None

    def test_classification_report_shape(self):
        rng = np.random.default_rng(4)
        p = rng.uniform(0, 1, 500)
        y = (rng.uniform(0, 1, 500) < p).astype(float)
        rep = classification_report(y, p)
        for key in ("brier", "log_loss", "ece", "calibration", "reliability", "base_rate"):
            assert key in rep


class TestRegressionDiagnostics:
    def test_hand_computed_mae_rmse(self):
        y = np.array([1.0, 2.0, 3.0])
        pred = np.array([2.0, 2.0, 5.0])
        rep = regression_report(y, pred)
        assert rep["mae"] == pytest.approx(1.0)
        assert rep["rmse"] == pytest.approx(np.sqrt((1 + 0 + 4) / 3))

    def test_rank_corr_perfect(self):
        y = np.array([1.0, 3.0, 2.0, 5.0, 4.0])
        rep = regression_report(y, y * 10)
        assert rep["rank_corr_spearman"] == pytest.approx(1.0)

    def test_constant_prediction_rank_corr_none(self):
        rep = regression_report(np.array([1.0, 2.0, 3.0]), np.full(3, 2.0))
        assert rep["rank_corr_spearman"] is None


class TestDecileTable:
    def test_ordering_and_totals(self):
        pred = np.arange(100, dtype=float)
        outcome = np.arange(100, dtype=float)  # outcome rises with prediction
        table = outcome_by_predicted_decile(pred, outcome, 10)
        assert len(table) == 10
        means = [row["mean_outcome"] for row in table]
        assert means == sorted(means)
        assert sum(row["total_outcome"] for row in table) == pytest.approx(outcome.sum())

    def test_small_samples_reduce_bins(self):
        pred = np.arange(6, dtype=float)
        table = outcome_by_predicted_decile(pred, pred, 10)
        assert 2 <= len(table) <= 3
