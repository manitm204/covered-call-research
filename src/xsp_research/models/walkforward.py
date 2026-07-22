"""Purged, embargoed, forward-chaining walk-forward splits.

Design (research brief sections 13/14):
- Validation folds are contiguous, non-overlapping, strictly chronological
  windows. Training data for a fold is everything BEFORE the fold whose label
  window (entry -> label_end) finishes at least ``embargo_days`` before the
  fold starts: purging removes samples whose 30-day labels would straddle the
  boundary; the embargo adds a safety buffer for serial correlation.
- There is never training data from after a validation fold (forward-chaining
  only, expanding window).
- Samples on/after ``final_test_start`` are excluded from every fold. The
  final test period is used once, explicitly, at the end of the research
  program — `final_test_indices` exists for that single call and the caller is
  expected to record its use.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta

import numpy as np


@dataclass(frozen=True, slots=True)
class WalkForwardConfig:
    n_folds: int = 4
    embargo_days: int = 5  # calendar days between last train label_end and fold start
    min_train_size: int = 15  # folds with fewer eligible training samples are dropped
    final_test_start: date | None = None


@dataclass(frozen=True, slots=True)
class Fold:
    fold_id: int
    train_idx: np.ndarray
    val_idx: np.ndarray
    val_start: date
    val_end: date


def walk_forward_splits(
    entry_dates: list[date],
    label_ends: list[date],
    cfg: WalkForwardConfig,
) -> list[Fold]:
    """Build folds over samples given their entry dates and label windows.

    Indices refer to positions in the input lists (dataset row order).
    """
    if len(entry_dates) != len(label_ends):
        raise ValueError("entry_dates and label_ends must align")
    entries = np.array(entry_dates)
    ends = np.array(label_ends)
    for e, le in zip(entry_dates, label_ends, strict=True):
        if le < e:
            raise ValueError(f"label_end {le} before entry {e}")

    in_scope = np.ones(len(entries), dtype=bool)
    if cfg.final_test_start is not None:
        # A sample belongs to the final test if ANY part of its label window
        # reaches into the final-test period.
        in_scope = np.array([le < cfg.final_test_start for le in label_ends])
    scope_idx = np.where(in_scope)[0]
    if len(scope_idx) < 2 * cfg.n_folds:
        return []

    scoped_entries = entries[scope_idx]
    t0, t1 = scoped_entries.min(), scoped_entries.max()
    total_days = (t1 - t0).days
    if total_days < cfg.n_folds + 1:
        return []
    # Segment 0 is the training warm-up; segments 1..n_folds are validation windows.
    seg_len = total_days / (cfg.n_folds + 1)
    folds: list[Fold] = []
    for k in range(1, cfg.n_folds + 1):
        val_start = t0 + timedelta(days=round(k * seg_len) + (1 if k > 0 else 0))
        val_end = t0 + timedelta(days=round((k + 1) * seg_len))
        if k == cfg.n_folds:
            val_end = t1
        val_mask = in_scope & (entries >= val_start) & (entries <= val_end)
        purge_cutoff = val_start - timedelta(days=cfg.embargo_days)
        train_mask = in_scope & (ends < purge_cutoff)
        train_idx = np.where(train_mask)[0]
        val_idx = np.where(val_mask)[0]
        if len(train_idx) < cfg.min_train_size or len(val_idx) == 0:
            continue
        folds.append(
            Fold(
                fold_id=k,
                train_idx=train_idx,
                val_idx=val_idx,
                val_start=val_start,
                val_end=val_end,
            )
        )
    return folds


def final_test_indices(
    entry_dates: list[date], label_ends: list[date], cfg: WalkForwardConfig
) -> np.ndarray:
    """Indices of the untouched final-test samples. Call once, at the end,
    and record that it was used (repeated evaluation invalidates it)."""
    if cfg.final_test_start is None:
        return np.array([], dtype=int)
    return np.array([i for i, le in enumerate(label_ends) if le >= cfg.final_test_start], dtype=int)


def verify_no_leakage(folds: list[Fold], entry_dates: list[date], label_ends: list[date]) -> None:
    """Invariant check (also used by tests): raises on any purge/embargo breach."""
    entries = np.array(entry_dates)
    ends = np.array(label_ends)
    for f in folds:
        if len(set(f.train_idx) & set(f.val_idx)):
            raise AssertionError(f"fold {f.fold_id}: train/val overlap")
        if ends[f.train_idx].size and ends[f.train_idx].max() >= f.val_start:
            raise AssertionError(f"fold {f.fold_id}: training label window reaches validation")
        if entries[f.train_idx].size and entries[f.train_idx].min() > f.val_end:
            raise AssertionError(f"fold {f.fold_id}: training data after validation window")
