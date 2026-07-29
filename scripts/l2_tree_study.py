"""Cycle-2 interpretable-tree study.

Shallow decision trees (depth 2 and 3, min_leaf 10% of samples) on the cycle-2
features vs QQQ forward-21d up/down, evaluated with PURGED walk-forward CV on
train+validation (2018-08..2024-12; 5 chronological folds, 21-session purge gap
on each side of the test fold). The tree must beat the base rate out-of-fold to
be taken seriously; either way its splits are printed and interpreted.

Anti-leak notes: features are point-in-time lagged (see l2_exog_features);
labels overlap 21 days -> purge removes label-overlap contamination at fold
boundaries; no shuffling anywhere.
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss
from sklearn.tree import DecisionTreeClassifier, export_text

sys.path.insert(0, "scripts")
from l2_common import append_experiment_log  # noqa: E402

SPAN = ("2018-08-01", "2024-12-31")
H = 21


def load():
    F = pd.read_parquet("data/features/cycle2_features.parquet")
    F = F.drop(columns=[c for c in ("date",) if c in F.columns])
    px = pd.read_parquet("data/normalized/prices_long/QQQ.parquet")
    px["date"] = pd.to_datetime(px.date)
    a = px.set_index("date").adjClose.sort_index()
    y = ((a.shift(-H) / a - 1) > 0).astype(int)
    df = F.join(y.rename("label")).loc[SPAN[0]:SPAN[1]]
    df = df.dropna(subset=["label"])
    # drop features with <90% coverage in span, median-impute the rest per-fold
    keep = [c for c in df.columns if c != "label" and df[c].notna().mean() > 0.9]
    return df[keep + ["label"]], keep


def purged_folds(n, k=5, purge=H):
    fold = n // k
    for i in range(k):
        te = np.arange(i * fold, min((i + 1) * fold, n))
        tr = np.array([j for j in range(n) if j < te[0] - purge or j > te[-1] + purge])
        yield tr, te


def main():
    df, feats = load()
    X, y = df[feats].values, df.label.values
    n = len(df)
    print(f"{n} sessions, {len(feats)} features, base rate P(up)={y.mean():.3f}")
    results = {}
    for depth in (2, 3):
        accs, briers, base_briers = [], [], []
        for tr, te in purged_folds(n):
            med = np.nanmedian(X[tr], axis=0)
            Xtr = np.where(np.isnan(X[tr]), med, X[tr])
            Xte = np.where(np.isnan(X[te]), med, X[te])
            m = DecisionTreeClassifier(max_depth=depth, min_samples_leaf=int(0.10 * len(tr)),
                                       random_state=0)
            m.fit(Xtr, y[tr])
            p = m.predict_proba(Xte)[:, 1]
            accs.append(((p > 0.5) == y[te]).mean())
            briers.append(brier_score_loss(y[te], p))
            base = np.full(len(te), y[tr].mean())
            base_briers.append(brier_score_loss(y[te], base))
        results[depth] = dict(oof_acc=np.mean(accs), oof_brier=np.mean(briers),
                              base_brier=np.mean(base_briers))
        print(f"depth {depth}: OOF acc {np.mean(accs):.3f} vs base {max(y.mean(),1-y.mean()):.3f}; "
              f"OOF Brier {np.mean(briers):.4f} vs base-rate Brier {np.mean(base_briers):.4f} "
              f"-> {'BEATS' if np.mean(briers) < np.mean(base_briers) else 'DOES NOT BEAT'} base")
        append_experiment_log(hypothesis="TREE", partition="train+valid_purgedCV",
                              underlying="QQQ", config=f"depth{depth}_minleaf10pct",
                              scenario="n/a", n=n, metric="oof_brier_minus_base",
                              value=round(np.mean(briers) - np.mean(base_briers), 4),
                              verdict="PASS" if np.mean(briers) < np.mean(base_briers) else "fail",
                              notes=f"oof_acc={np.mean(accs):.3f}")
    # full-span tree for INTERPRETATION only (never for performance claims)
    med = np.nanmedian(X, axis=0)
    Xf = np.where(np.isnan(X), med, X)
    m = DecisionTreeClassifier(max_depth=3, min_samples_leaf=int(0.10 * n), random_state=0)
    m.fit(Xf, y)
    print("\n--- full-span depth-3 tree (interpretation only, in-sample) ---")
    print(export_text(m, feature_names=feats, decimals=3))
    imp = pd.Series(m.feature_importances_, index=feats)
    print("importances:", imp[imp > 0].sort_values(ascending=False).round(3).to_dict())


if __name__ == "__main__":
    main()
