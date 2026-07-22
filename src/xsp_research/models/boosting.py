"""Gradient boosting with leakage-safe NESTED hyperparameter tuning.

Admission rule (research brief section 13): boosting is only adopted if it
beats the simpler benchmarks on untouched walk-forward results after costs —
this module therefore always reports the base-rate and logistic benchmarks on
the identical folds next to the tuned booster.

Nesting: for each OUTER fold, hyperparameters are chosen on an INNER
chronological split of that fold's training data only — the inner validation
block is the latest slice of training entries, and inner-training samples
whose label windows reach into it are purged (same embargo as the outer
split). The outer validation data never influences the choice. The small
deterministic grid keeps the search reproducible; Optuna could replace the
grid inside `_select_params` without touching the nesting structure.
"""

from __future__ import annotations

from datetime import date, timedelta
from typing import Any

import numpy as np
import polars as pl
import xgboost as xgb

from xsp_research.models.calibration import (
    classification_report,
    outcome_by_predicted_decile,
    regression_report,
)
from xsp_research.models.evaluate import CLASSIFICATION_LABELS, _prepare
from xsp_research.models.pnl_model import TrainMeanModel, regression_model_factories
from xsp_research.models.probability_model import BaseRateModel, probability_model_factories
from xsp_research.models.walkforward import (
    WalkForwardConfig,
    verify_no_leakage,
    walk_forward_splits,
)

# Ordered simplest-first: ties in inner score resolve to the simpler setting.
DEFAULT_GRID: list[dict[str, Any]] = [
    {"max_depth": d, "n_estimators": n, "learning_rate": lr}
    for d in (2, 3)
    for n in (50, 150)
    for lr in (0.1, 0.05)
]

_COMMON = {
    "subsample": 0.9,
    "colsample_bytree": 0.9,
    "min_child_weight": 5,
    "random_state": 7,
    "n_jobs": 1,
    "verbosity": 0,
}


class XGBClassifierModel:
    """XGBoost classifier (handles NaN natively; no imputation needed)."""

    def __init__(self, **params: Any) -> None:
        self._params = {**_COMMON, **params}
        self._model: xgb.XGBClassifier | None = None
        self._constant: float | None = None

    def fit(self, x: np.ndarray, y: np.ndarray) -> None:
        if len(np.unique(y)) < 2:
            self._constant = float(np.mean(y))
            return
        self._constant = None
        self._model = xgb.XGBClassifier(**self._params)
        self._model.fit(x, y)

    def predict_proba1(self, x: np.ndarray) -> np.ndarray:
        if self._constant is not None:
            return np.full(len(x), self._constant)
        return self._model.predict_proba(x)[:, 1]


class XGBRegressorModel:
    def __init__(self, **params: Any) -> None:
        self._params = {**_COMMON, **params}
        self._model: xgb.XGBRegressor | None = None

    def fit(self, x: np.ndarray, y: np.ndarray) -> None:
        self._model = xgb.XGBRegressor(**self._params)
        self._model.fit(x, y)

    def predict(self, x: np.ndarray) -> np.ndarray:
        return self._model.predict(x).astype(float)


def _inner_split(
    train_idx: np.ndarray,
    entries: list[date],
    ends: list[date],
    inner_frac: float,
    embargo_days: int,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Chronological purged inner split within an outer training set."""
    order = sorted(train_idx, key=lambda i: entries[i])
    n_inner = max(int(len(order) * inner_frac), 5)
    if len(order) < n_inner + 10:
        return None
    inner_val = np.array(order[-n_inner:])
    inner_val_start = min(entries[i] for i in inner_val)
    cutoff = inner_val_start - timedelta(days=embargo_days)
    inner_train = np.array([i for i in order[:-n_inner] if ends[i] < cutoff])
    if len(inner_train) < 10:
        return None
    return inner_train, inner_val


def _inner_score(y_true: np.ndarray, pred: np.ndarray, kind: str) -> float:
    """Lower is better: Brier for classification, MAE for regression."""
    if kind == "classification":
        return float(np.mean((pred - y_true) ** 2))
    return float(np.mean(np.abs(pred - y_true)))


def _select_params(
    x: np.ndarray,
    y: np.ndarray,
    inner_train: np.ndarray,
    inner_val: np.ndarray,
    grid: list[dict[str, Any]],
    kind: str,
) -> dict[str, Any]:
    best_params, best_score = grid[0], np.inf
    for params in grid:
        model = (
            XGBClassifierModel(**params)
            if kind == "classification"
            else XGBRegressorModel(**params)
        )
        model.fit(x[inner_train], y[inner_train])
        pred = (
            model.predict_proba1(x[inner_val])
            if kind == "classification"
            else model.predict(x[inner_val])
        )
        score = _inner_score(y[inner_val], pred, kind)
        if score < best_score - 1e-12:  # strict improvement: ties keep the simpler
            best_params, best_score = params, score
    return best_params


def nested_boosting_walkforward(
    dataset: pl.DataFrame,
    feature_cols: list[str],
    label_col: str,
    wf_cfg: WalkForwardConfig,
    grid: list[dict[str, Any]] | None = None,
    inner_frac: float = 0.25,
) -> dict[str, Any]:
    """Outer purged walk-forward; per-fold nested tuning; benchmark comparison."""
    grid = grid or DEFAULT_GRID
    kind = "classification" if label_col in CLASSIFICATION_LABELS else "regression"
    x, y, entries, ends, pnl = _prepare(dataset, feature_cols, label_col)
    folds = walk_forward_splits(entries, ends, wf_cfg)
    if not folds:
        return {"label": label_col, "error": "not enough samples for walk-forward folds"}
    verify_no_leakage(folds, entries, ends)

    chosen: list[dict[str, Any]] = []
    idx_all: list[int] = []
    pred_all: list[float] = []
    for fold in folds:
        split = _inner_split(fold.train_idx, entries, ends, inner_frac, wf_cfg.embargo_days)
        if split is None:
            params = grid[0]  # too little data to tune: simplest setting
        else:
            params = _select_params(x, y, *split, grid, kind)
        chosen.append({"fold": fold.fold_id, **params})
        model = (
            XGBClassifierModel(**params)
            if kind == "classification"
            else XGBRegressorModel(**params)
        )
        model.fit(x[fold.train_idx], y[fold.train_idx])
        pred = (
            model.predict_proba1(x[fold.val_idx])
            if kind == "classification"
            else model.predict(x[fold.val_idx])
        )
        idx_all.extend(fold.val_idx.tolist())
        pred_all.extend(np.asarray(pred, dtype=float).tolist())

    idx = np.array(idx_all, dtype=int)
    pred = np.array(pred_all)
    report = (
        classification_report(y[idx], pred)
        if kind == "classification"
        else regression_report(y[idx], pred)
    )
    report["pnl_by_predicted_decile"] = outcome_by_predicted_decile(pred, pnl[idx], 5)

    # Benchmarks on identical folds (the admission bar).
    benchmarks: dict[str, dict[str, Any]] = {}
    bench_factories = (
        {"base_rate": BaseRateModel, "logistic": probability_model_factories()["logistic"]}
        if kind == "classification"
        else {"train_mean": TrainMeanModel, "ridge": regression_model_factories()["ridge"]}
    )
    for name, factory in bench_factories.items():
        b_idx: list[int] = []
        b_pred: list[float] = []
        for fold in folds:
            m = factory()
            m.fit(x[fold.train_idx], y[fold.train_idx])
            p = (
                m.predict_proba1(x[fold.val_idx])
                if kind == "classification"
                else m.predict(x[fold.val_idx])
            )
            b_idx.extend(fold.val_idx.tolist())
            b_pred.extend(np.asarray(p, dtype=float).tolist())
        bi, bp = np.array(b_idx, dtype=int), np.array(b_pred)
        benchmarks[name] = (
            classification_report(y[bi], bp)
            if kind == "classification"
            else regression_report(y[bi], bp)
        )

    primary = "brier" if kind == "classification" else "mae"
    bar = min(b[primary] for b in benchmarks.values())
    return {
        "label": label_col,
        "kind": kind,
        "n_samples": len(y),
        "n_folds": len(folds),
        "chosen_params_per_fold": chosen,
        "param_stability": _param_stability(chosen),
        "boosting": report,
        "benchmarks": benchmarks,
        "admission": {
            "metric": primary,
            "boosting": report[primary],
            "best_benchmark": bar,
            "boosting_beats_benchmarks": bool(report[primary] < bar),
            "rule": "adopt boosting ONLY if it beats every simpler benchmark on this metric",
        },
    }


def _param_stability(chosen: list[dict[str, Any]]) -> dict[str, Any]:
    """How consistent were the tuned settings across folds (instability is a
    red flag for overfitting the inner splits)."""
    keys = [k for k in chosen[0] if k != "fold"]
    return {
        k: {"values": [c[k] for c in chosen], "n_unique": len({c[k] for c in chosen})} for k in keys
    }
