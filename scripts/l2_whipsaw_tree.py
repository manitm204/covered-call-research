"""Cycle-3 whipsaw-prediction tree.

Motivation: TG-CER's only proven failure mode is chop around the 200d MA
(2025). Question the tree answers: do feature UNIONS exist that flag
"the next 21 sessions will whipsaw the +/-1% MA200 band"?

Label (QQQ): next 21 sessions contain >= 2 crossings of the +/-1% hysteresis
band state (i.e. at least one full exit-and-reenter round trip).
Method: depth-2/3 trees, purged 5-fold walk-forward on 2018-08..2024-12,
compared against the base rate. Any promising split is then re-tested
OUT-OF-ERA on 2000-2017 prices (whipsaw label computable from prices alone;
feature availability permitting).
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss
from sklearn.tree import DecisionTreeClassifier, export_text

sys.path.insert(0, "scripts")
from l2_common import append_experiment_log  # noqa: E402

H = 21


def band_state(a: pd.Series) -> pd.Series:
    ma = a.rolling(200).mean()
    state = pd.Series(index=a.index, dtype=float)
    cur = np.nan
    for t in a.index:
        p, m = a.loc[t], ma.loc[t]
        if np.isnan(m):
            continue
        if np.isnan(cur):
            cur = 1.0 if p > m else 0.0
        elif cur == 1.0 and p < 0.99 * m:
            cur = 0.0
        elif cur == 0.0 and p > 1.01 * m:
            cur = 1.0
        state.loc[t] = cur
    return state


def whipsaw_label(a: pd.Series) -> pd.Series:
    st = band_state(a)
    chg = (st != st.shift(1)) & st.notna() & st.shift(1).notna()
    fwd_crossings = pd.Series(
        [chg.iloc[i + 1: i + 1 + H].sum() if i + 1 + H <= len(chg) else np.nan
         for i in range(len(chg))], index=chg.index)
    return (fwd_crossings >= 2).astype(float).where(fwd_crossings.notna())


def load_panel():
    px = pd.read_parquet("data/normalized/prices_long/QQQ.parquet")
    px["date"] = pd.to_datetime(px.date)
    a = px.set_index("date").adjClose.sort_index()
    y = whipsaw_label(a)
    F = pd.read_parquet("data/features/cycle2_features.parquet")
    F = F.drop(columns=[c for c in ("date",) if c in F.columns])
    df = F.join(y.rename("label")).loc["2018-08-01":"2024-12-31"].dropna(subset=["label"])
    keep = [c for c in df.columns if c != "label" and df[c].notna().mean() > 0.9]
    return df[keep + ["label"]], keep, a


def purged_folds(n, k=5, purge=H):
    fold = n // k
    for i in range(k):
        te = np.arange(i * fold, min((i + 1) * fold, n))
        tr = np.array([j for j in range(n) if j < te[0] - purge or j > te[-1] + purge])
        yield tr, te


def main():
    df, feats, a = load_panel()
    X, y = df[feats].values, df.label.values
    n = len(df)
    print(f"{n} sessions; whipsaw base rate {y.mean():.3f}")
    for depth in (2, 3):
        briers, base_b, accs = [], [], []
        for tr, te in purged_folds(n):
            med = np.nanmedian(X[tr], axis=0)
            Xtr = np.where(np.isnan(X[tr]), med, X[tr])
            Xte = np.where(np.isnan(X[te]), med, X[te])
            m = DecisionTreeClassifier(max_depth=depth,
                                       min_samples_leaf=int(0.10 * len(tr)), random_state=0)
            m.fit(Xtr, y[tr])
            p = m.predict_proba(Xte)[:, 1] if m.n_classes_ > 1 else np.full(len(te), y[tr].mean())
            briers.append(brier_score_loss(y[te], p))
            base_b.append(brier_score_loss(y[te], np.full(len(te), y[tr].mean())))
            accs.append(((p > 0.5) == y[te]).mean())
        verdict = "PASS" if np.mean(briers) < np.mean(base_b) else "fail"
        print(f"depth {depth}: OOF Brier {np.mean(briers):.4f} vs base {np.mean(base_b):.4f} "
              f"acc {np.mean(accs):.3f} -> {verdict}")
        append_experiment_log(hypothesis="H16_whipsaw_tree", partition="train+valid_purgedCV",
                              underlying="QQQ", config=f"depth{depth}", scenario="n/a", n=n,
                              metric="oof_brier_minus_base",
                              value=round(np.mean(briers) - np.mean(base_b), 4),
                              verdict=verdict, notes=f"whipsaw base rate {y.mean():.3f}")
    med = np.nanmedian(X, axis=0)
    m = DecisionTreeClassifier(max_depth=3, min_samples_leaf=int(0.10 * n), random_state=0)
    m.fit(np.where(np.isnan(X), med, X), y)
    print("\n--- full-span tree (interpretation only) ---")
    print(export_text(m, feature_names=feats, decimals=3))
    imp = pd.Series(m.feature_importances_, index=feats)
    print("importances:", imp[imp > 0].sort_values(ascending=False).round(3).to_dict())


if __name__ == "__main__":
    main()
