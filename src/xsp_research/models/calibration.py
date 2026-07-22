"""Probability-calibration and regression-quality diagnostics.

Implements the required metrics: Brier score, log loss, reliability tables,
calibration slope/intercept (logistic recalibration), expected calibration
error, and for regression MAE/RMSE/rank-correlation plus realized-outcome
tables by predicted decile.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy import stats

_EPS = 1e-12


def brier_score(y: np.ndarray, p: np.ndarray) -> float:
    return float(np.mean((p - y) ** 2))


def log_loss(y: np.ndarray, p: np.ndarray) -> float:
    p = np.clip(p, _EPS, 1.0 - _EPS)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def reliability_table(y: np.ndarray, p: np.ndarray, n_bins: int = 10) -> list[dict[str, Any]]:
    """Equal-width probability bins: predicted mean vs observed rate per bin."""
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    out = []
    for lo, hi in zip(edges[:-1], edges[1:], strict=True):
        mask = (p >= lo) & (p < hi if hi < 1.0 else p <= hi)
        if not mask.any():
            continue
        out.append(
            {
                "bin": f"[{lo:.1f},{hi:.1f})",
                "n": int(mask.sum()),
                "mean_predicted": float(p[mask].mean()),
                "observed_rate": float(y[mask].mean()),
            }
        )
    return out


def expected_calibration_error(y: np.ndarray, p: np.ndarray, n_bins: int = 10) -> float:
    table = reliability_table(y, p, n_bins)
    n = len(y)
    return float(
        sum(row["n"] / n * abs(row["mean_predicted"] - row["observed_rate"]) for row in table)
    )


def calibration_slope_intercept(y: np.ndarray, p: np.ndarray) -> dict[str, float | None]:
    """Logistic recalibration: fit y ~ logit(p). Slope 1 / intercept 0 = calibrated.

    Degenerate inputs (constant p, single-class y) return None values rather
    than fabricated numbers.
    """
    p = np.clip(p, _EPS, 1.0 - _EPS)
    logit = np.log(p / (1.0 - p))
    if np.ptp(logit) < 1e-12 or len(np.unique(y)) < 2:
        return {"slope": None, "intercept": None}
    from sklearn.linear_model import LogisticRegression

    lr = LogisticRegression(C=1e6, max_iter=2000)
    lr.fit(logit.reshape(-1, 1), y)
    return {"slope": float(lr.coef_[0][0]), "intercept": float(lr.intercept_[0])}


def classification_report(y: np.ndarray, p: np.ndarray, n_bins: int = 10) -> dict[str, Any]:
    return {
        "n": int(len(y)),
        "base_rate": float(np.mean(y)),
        "mean_predicted": float(np.mean(p)),
        "brier": brier_score(y, p),
        "log_loss": log_loss(y, p),
        "ece": expected_calibration_error(y, p, n_bins),
        "calibration": calibration_slope_intercept(y, p),
        "reliability": reliability_table(y, p, n_bins),
    }


def regression_report(y: np.ndarray, pred: np.ndarray) -> dict[str, Any]:
    err = pred - y
    if np.ptp(pred) < 1e-12:
        rank_corr = None  # constant predictions have no ranking information
    else:
        rank_corr = float(stats.spearmanr(pred, y).statistic)
    return {
        "n": int(len(y)),
        "mae": float(np.mean(np.abs(err))),
        "rmse": float(np.sqrt(np.mean(err**2))),
        "rank_corr_spearman": rank_corr,
        "mean_actual": float(np.mean(y)),
        "mean_predicted": float(np.mean(pred)),
    }


def outcome_by_predicted_decile(
    pred: np.ndarray, outcome: np.ndarray, n_bins: int = 10
) -> list[dict[str, Any]]:
    """Realized outcome (e.g. net P&L) grouped by predicted-value decile.
    This is the 'does ranking by the model select better trades' view."""
    if len(pred) < n_bins:
        n_bins = max(2, len(pred) // 2)
    order = np.argsort(pred)
    bins = np.array_split(order, n_bins)
    out = []
    for i, idx in enumerate(bins):
        if len(idx) == 0:
            continue
        out.append(
            {
                "decile": i + 1,  # 1 = lowest predicted value
                "n": int(len(idx)),
                "mean_predicted": float(pred[idx].mean()),
                "mean_outcome": float(outcome[idx].mean()),
                "total_outcome": float(outcome[idx].sum()),
            }
        )
    return out
