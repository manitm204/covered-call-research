"""Walk-forward splits: purge, embargo, final-test isolation, chronology."""

from datetime import date, timedelta

import pytest

from xsp_research.models.walkforward import (
    Fold,
    WalkForwardConfig,
    final_test_indices,
    verify_no_leakage,
    walk_forward_splits,
)


def make_samples(n: int = 120, horizon_days: int = 30):
    """Weekly entries over ~2.3 years, 30-day label windows."""
    entries = [date(2022, 1, 3) + timedelta(days=7 * i) for i in range(n)]
    ends = [e + timedelta(days=horizon_days) for e in entries]
    return entries, ends


CFG = WalkForwardConfig(n_folds=4, embargo_days=5, min_train_size=10)


class TestSplitStructure:
    def test_folds_created(self):
        entries, ends = make_samples()
        folds = walk_forward_splits(entries, ends, CFG)
        assert 1 <= len(folds) <= 4

    def test_no_leakage_invariant(self):
        entries, ends = make_samples()
        folds = walk_forward_splits(entries, ends, CFG)
        verify_no_leakage(folds, entries, ends)  # must not raise

    def test_purge_removes_overlapping_labels(self):
        """No training sample's label window may reach the validation start."""
        entries, ends = make_samples()
        folds = walk_forward_splits(entries, ends, CFG)
        for f in folds:
            for i in f.train_idx:
                assert ends[i] < f.val_start - timedelta(days=CFG.embargo_days - 1)

    def test_validation_windows_chronological_and_disjoint(self):
        entries, ends = make_samples()
        folds = walk_forward_splits(entries, ends, CFG)
        for a, b in zip(folds, folds[1:], strict=False):
            assert a.val_end < b.val_start

    def test_each_sample_in_at_most_one_validation_fold(self):
        entries, ends = make_samples()
        folds = walk_forward_splits(entries, ends, CFG)
        seen: set[int] = set()
        for f in folds:
            overlap = seen & set(f.val_idx.tolist())
            assert not overlap
            seen |= set(f.val_idx.tolist())

    def test_training_expands_over_folds(self):
        entries, ends = make_samples()
        folds = walk_forward_splits(entries, ends, CFG)
        sizes = [len(f.train_idx) for f in folds]
        assert sizes == sorted(sizes)


class TestFinalTestIsolation:
    def test_final_test_never_in_folds(self):
        entries, ends = make_samples()
        cutoff = date(2023, 9, 1)
        cfg = WalkForwardConfig(n_folds=3, embargo_days=5, final_test_start=cutoff)
        folds = walk_forward_splits(entries, ends, cfg)
        test_idx = set(final_test_indices(entries, ends, cfg).tolist())
        assert test_idx
        for f in folds:
            assert not (set(f.train_idx.tolist()) | set(f.val_idx.tolist())) & test_idx

    def test_label_window_reaching_test_period_is_excluded(self):
        """A sample entered before the cutoff whose label ends after it belongs
        to neither training nor validation."""
        entries, ends = make_samples()
        cutoff = date(2023, 9, 1)
        cfg = WalkForwardConfig(n_folds=3, embargo_days=5, final_test_start=cutoff)
        folds = walk_forward_splits(entries, ends, cfg)
        straddlers = {
            i for i, (e, le) in enumerate(zip(entries, ends, strict=True)) if e < cutoff <= le
        }
        assert straddlers  # the fixture must actually contain straddlers
        for f in folds:
            assert not (set(f.train_idx.tolist()) | set(f.val_idx.tolist())) & straddlers


class TestDegenerateInputs:
    def test_too_few_samples_returns_empty(self):
        entries, ends = make_samples(5)
        assert walk_forward_splits(entries, ends, CFG) == []

    def test_label_before_entry_rejected(self):
        with pytest.raises(ValueError, match="label_end"):
            walk_forward_splits([date(2023, 1, 2)], [date(2022, 12, 30)], CFG)

    def test_verify_catches_bad_fold(self):
        import numpy as np

        entries, ends = make_samples(20)
        bad = Fold(
            fold_id=1,
            train_idx=np.array([19]),  # future sample in training
            val_idx=np.array([0]),
            val_start=entries[0],
            val_end=entries[0],
        )
        with pytest.raises(AssertionError):
            verify_no_leakage([bad], entries, ends)
