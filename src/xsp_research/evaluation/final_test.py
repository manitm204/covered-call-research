"""Single-use final-test evaluation with an operational usage guard.

The final untouched period is the last honest estimate of real-world
performance; every additional evaluation against it converts it into another
validation set. This module makes that cost explicit and auditable:

- Usage is recorded in a JSON file (default `reports/final_test_usage.json`)
  with timestamp, label, dataset fingerprint, and git commit.
- A second evaluation for the same (label, final_test_start) refuses to run
  unless `force=True`, and forced reruns are recorded as reuse — the file is
  the audit trail reviewers should demand to see.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from xsp_research.models.calibration import (
    classification_report,
    outcome_by_predicted_decile,
    regression_report,
)
from xsp_research.models.evaluate import CLASSIFICATION_LABELS, _prepare
from xsp_research.models.walkforward import WalkForwardConfig, final_test_indices

DEFAULT_USAGE_FILE = "reports/final_test_usage.json"


class FinalTestReuseError(RuntimeError):
    """Raised when the final test period would be evaluated a second time."""


def _dataset_fingerprint(dataset: pl.DataFrame) -> str:
    ids = ",".join(sorted(dataset["trade_id"].to_list())) if "trade_id" in dataset.columns else ""
    return hashlib.sha256(f"{dataset.height}:{ids}".encode()).hexdigest()[:16]


def _git_commit() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=10
        )
        return out.stdout.strip() if out.returncode == 0 else None
    except OSError:
        return None


def _load_usage(path: Path) -> list[dict[str, Any]]:
    if path.exists():
        return json.loads(path.read_text())
    return []


def run_final_test(
    dataset: pl.DataFrame,
    feature_cols: list[str],
    label_col: str,
    wf_cfg: WalkForwardConfig,
    model_factory,
    usage_file: str | Path = DEFAULT_USAGE_FILE,
    force: bool = False,
) -> dict[str, Any]:
    """Train on everything before the final period (purged), evaluate ONCE on it.

    `model_factory` builds the frozen model choice (selected earlier via
    walk-forward — do not shop for models here; that is exactly the reuse this
    guard exists to prevent).
    """
    if wf_cfg.final_test_start is None:
        raise ValueError("wf_cfg.final_test_start must be set for a final test")
    usage_path = Path(usage_file)
    usage = _load_usage(usage_path)
    key = {"label": label_col, "final_test_start": str(wf_cfg.final_test_start)}
    prior = [
        u
        for u in usage
        if u["label"] == key["label"] and u["final_test_start"] == key["final_test_start"]
    ]
    if prior and not force:
        raise FinalTestReuseError(
            f"final test for {key} already evaluated on {prior[0]['run_ts_utc']} "
            f"(recorded in {usage_path}). Re-running invalidates it; pass force=True "
            "only with that understanding — the reuse will be recorded."
        )

    kind = "classification" if label_col in CLASSIFICATION_LABELS else "regression"
    x, y, entries, ends, pnl = _prepare(dataset, feature_cols, label_col)
    test_idx = final_test_indices(entries, ends, wf_cfg)
    if len(test_idx) == 0:
        raise ValueError("no samples in the final test period")
    from datetime import timedelta

    cutoff = wf_cfg.final_test_start - timedelta(days=wf_cfg.embargo_days)
    train_idx = np.array([i for i, le in enumerate(ends) if le < cutoff], dtype=int)
    if len(train_idx) < wf_cfg.min_train_size:
        raise ValueError("insufficient training data before the final test period")

    model = model_factory()
    model.fit(x[train_idx], y[train_idx])
    pred = (
        model.predict_proba1(x[test_idx])
        if kind == "classification"
        else model.predict(x[test_idx])
    )
    pred = np.asarray(pred, dtype=float)
    report = (
        classification_report(y[test_idx], pred)
        if kind == "classification"
        else regression_report(y[test_idx], pred)
    )
    report["pnl_by_predicted_decile"] = outcome_by_predicted_decile(pred, pnl[test_idx], 5)

    record = {
        **key,
        "run_ts_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "n_train": int(len(train_idx)),
        "n_test": int(len(test_idx)),
        "dataset_fingerprint": _dataset_fingerprint(dataset),
        "git_commit": _git_commit(),
        "forced_reuse": bool(prior),
    }
    usage.append(record)
    usage_path.parent.mkdir(parents=True, exist_ok=True)
    usage_path.write_text(json.dumps(usage, indent=2))

    return {
        "final_test": report,
        "usage_record": record,
        "WARNING": (
            "This period has now been consumed. Any further evaluation against it "
            "is in-sample by construction."
        ),
    }
