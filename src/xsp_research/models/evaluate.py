"""Walk-forward model evaluation and feature-family ablations.

Consumes the trade-level research dataset (features + label_* columns), runs
purged/embargoed walk-forward for each benchmark model, and reports pooled
out-of-fold diagnostics. Every model is compared against the always-trade
baseline via realized-P&L-by-predicted-decile tables — a model only matters if
ranking by it separates good trades from bad ones after costs.

No conclusion from a single run of this module is treated as statistically
significant; it produces evidence for comparison, not verdicts.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import polars as pl

from xsp_research.features.registry import REGISTRY, FeatureFamily
from xsp_research.features.surface import SURFACE_FEATURE_NAMES
from xsp_research.models.calibration import (
    classification_report,
    outcome_by_predicted_decile,
    regression_report,
)
from xsp_research.models.pnl_model import RegressorFactory, regression_model_factories
from xsp_research.models.probability_model import ModelFactory, probability_model_factories
from xsp_research.models.walkforward import (
    Fold,
    WalkForwardConfig,
    verify_no_leakage,
    walk_forward_splits,
)

# Trade-structure-only baseline: what you know from the spread itself, no
# market state. Every feature family must add value beyond this to matter.
BASELINE_FEATURES = ["surf_short_delta", "surf_dte", "surf_credit_over_width"]

CLASSIFICATION_LABELS = ("label_expire_itm", "label_touch")
REGRESSION_LABELS = ("label_net_pnl", "label_mae", "label_pct_credit")


def feature_family_columns(dataset: pl.DataFrame) -> dict[str, list[str]]:
    """Feature columns present in the dataset, grouped by family."""
    fams: dict[str, list[str]] = {}
    for spec in REGISTRY.values():
        if spec.family is FeatureFamily.SURFACE:
            continue
        if spec.name in dataset.columns:
            fams.setdefault(spec.family.value, []).append(spec.name)
    surface = [c for c in SURFACE_FEATURE_NAMES if c in dataset.columns]
    if surface:
        fams["surface"] = surface
    return {k: sorted(v) for k, v in fams.items()}


def _prepare(
    dataset: pl.DataFrame, feature_cols: list[str], label_col: str
) -> tuple[np.ndarray, np.ndarray, list, list, np.ndarray]:
    """Drop null-label rows; return X, y, entry dates, label ends, pnl per spread."""
    needed = ["entry_ts", "label_end", label_col, "label_net_pnl", "qty"]
    df = dataset.drop_nulls(subset=[label_col]).sort("entry_ts")
    missing = [c for c in needed if c not in dataset.columns]
    if missing:
        raise ValueError(f"research dataset missing columns: {missing}")
    x = df.select(feature_cols).to_numpy().astype(float)
    y = df[label_col].to_numpy().astype(float)
    entries = [ts.date() for ts in df["entry_ts"].to_list()]
    ends = df["label_end"].to_list()
    pnl = df["label_net_pnl"].to_numpy().astype(float)
    return x, y, entries, ends, pnl


def _oof_predictions(
    x: np.ndarray,
    y: np.ndarray,
    folds: list[Fold],
    factory,
    kind: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Pooled out-of-fold predictions (indices, predictions)."""
    idx_all: list[int] = []
    pred_all: list[float] = []
    for fold in folds:
        model = factory()
        model.fit(x[fold.train_idx], y[fold.train_idx])
        pred = (
            model.predict_proba1(x[fold.val_idx])
            if kind == "classification"
            else model.predict(x[fold.val_idx])
        )
        idx_all.extend(fold.val_idx.tolist())
        pred_all.extend(np.asarray(pred, dtype=float).tolist())
    return np.array(idx_all, dtype=int), np.array(pred_all)


def run_walkforward(
    dataset: pl.DataFrame,
    feature_cols: list[str],
    label_col: str,
    wf_cfg: WalkForwardConfig,
    model_factories: dict[str, ModelFactory] | dict[str, RegressorFactory] | None = None,
) -> dict[str, Any]:
    """Walk-forward evaluation of every benchmark model on one label."""
    kind = "classification" if label_col in CLASSIFICATION_LABELS else "regression"
    if model_factories is None:
        model_factories = (
            probability_model_factories()
            if kind == "classification"
            else regression_model_factories()
        )
    x, y, entries, ends, pnl = _prepare(dataset, feature_cols, label_col)
    folds = walk_forward_splits(entries, ends, wf_cfg)
    if not folds:
        return {
            "label": label_col,
            "kind": kind,
            "error": "not enough samples/time-span for walk-forward folds",
            "n_samples": len(y),
        }
    verify_no_leakage(folds, entries, ends)

    out: dict[str, Any] = {
        "label": label_col,
        "kind": kind,
        "n_samples": len(y),
        "n_features": len(feature_cols),
        "n_folds": len(folds),
        "folds": [
            {
                "fold": f.fold_id,
                "train_n": len(f.train_idx),
                "val_n": len(f.val_idx),
                "val_window": [str(f.val_start), str(f.val_end)],
            }
            for f in folds
        ],
        "models": {},
    }
    for name, factory in model_factories.items():
        idx, pred = _oof_predictions(x, y, folds, factory, kind)
        report = (
            classification_report(y[idx], pred)
            if kind == "classification"
            else regression_report(y[idx], pred)
        )
        # Economic value view: realized per-spread net P&L by predicted decile.
        report["pnl_by_predicted_decile"] = outcome_by_predicted_decile(pred, pnl[idx], 5)
        report["always_trade_mean_pnl"] = float(pnl[idx].mean())
        out["models"][name] = report
    return out


def run_family_ablation(
    dataset: pl.DataFrame,
    label_col: str,
    wf_cfg: WalkForwardConfig,
    model_name: str = "logistic",
) -> dict[str, Any]:
    """Required ablation grid: baseline / each family alone (with baseline) /
    all / all-except-each-family. One model, identical folds; only the feature
    set varies."""
    fams = feature_family_columns(dataset)
    kind = "classification" if label_col in CLASSIFICATION_LABELS else "regression"
    factories = (
        probability_model_factories() if kind == "classification" else regression_model_factories()
    )
    if model_name not in factories:
        raise ValueError(f"unknown model {model_name}; have {sorted(factories)}")
    factory = {model_name: factories[model_name]}

    baseline = [c for c in BASELINE_FEATURES if c in dataset.columns]
    all_cols = sorted({c for cols in fams.values() for c in cols} | set(baseline))

    feature_sets: dict[str, list[str]] = {"baseline_only": baseline, "all_features": all_cols}
    for fam, cols in fams.items():
        feature_sets[f"baseline_plus_{fam}"] = sorted(set(baseline) | set(cols))
        feature_sets[f"all_minus_{fam}"] = sorted(set(all_cols) - set(cols))

    results: dict[str, Any] = {"label": label_col, "model": model_name, "sets": {}}
    for set_name, cols in feature_sets.items():
        if not cols:
            results["sets"][set_name] = {"error": "no features available"}
            continue
        wf = run_walkforward(dataset, cols, label_col, wf_cfg, model_factories=factory)
        if "error" in wf:
            results["sets"][set_name] = {"error": wf["error"]}
            continue
        report = wf["models"][model_name]
        key_metrics = (
            {"brier": report["brier"], "log_loss": report["log_loss"], "ece": report["ece"]}
            if kind == "classification"
            else {
                "mae": report["mae"],
                "rmse": report["rmse"],
                "rank_corr": report["rank_corr_spearman"],
            }
        )
        results["sets"][set_name] = {"n_features": len(cols), **key_metrics}
    return results
