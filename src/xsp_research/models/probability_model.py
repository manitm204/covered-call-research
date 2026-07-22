"""Benchmark probability models, in mandated order of complexity.

1. Historical base rate (the bar every model must clear)
2. Logistic regression
3. Regularized logistic (L1)
4. Shallow decision tree

All sklearn models run inside a Pipeline whose imputer/scaler are fit ONLY on
the training data of each fold — no global preprocessing, no leakage. More
complex models (forests, boosting) belong to Phase 5 and are only admitted if
they beat these benchmarks on untouched walk-forward results after costs.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Protocol

import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier


class ProbabilityModel(Protocol):
    def fit(self, x: np.ndarray, y: np.ndarray) -> None: ...
    def predict_proba1(self, x: np.ndarray) -> np.ndarray: ...


class BaseRateModel:
    """Predicts the training-set positive rate for every sample."""

    def __init__(self) -> None:
        self._rate: float | None = None

    def fit(self, x: np.ndarray, y: np.ndarray) -> None:
        self._rate = float(np.mean(y))

    def predict_proba1(self, x: np.ndarray) -> np.ndarray:
        if self._rate is None:
            raise RuntimeError("fit first")
        return np.full(len(x), self._rate)


class SklearnBinaryModel:
    """Imputer + scaler + estimator pipeline; fold-local fitting only.

    A training fold containing a single class (possible for rare-event labels
    like deep-OTM expiration) cannot support a discriminative fit; the model
    degrades to the training base rate for that fold instead of crashing —
    equivalent to admitting it learned nothing there.
    """

    def __init__(self, estimator) -> None:
        self._pipe = Pipeline(
            [
                ("impute", SimpleImputer(strategy="median", keep_empty_features=True)),
                ("scale", StandardScaler()),
                ("model", estimator),
            ]
        )
        self._constant: float | None = None

    def fit(self, x: np.ndarray, y: np.ndarray) -> None:
        if len(np.unique(y)) < 2:
            self._constant = float(np.mean(y))
            return
        self._constant = None
        self._pipe.fit(x, y)

    def predict_proba1(self, x: np.ndarray) -> np.ndarray:
        if self._constant is not None:
            return np.full(len(x), self._constant)
        classes = list(self._pipe.classes_)
        return self._pipe.predict_proba(x)[:, classes.index(1)]


ModelFactory = Callable[[], ProbabilityModel]


def probability_model_factories(seed: int = 7) -> dict[str, ModelFactory]:
    """Benchmark suite in complexity order."""
    return {
        "base_rate": BaseRateModel,
        "logistic": lambda: SklearnBinaryModel(
            LogisticRegression(C=1.0, max_iter=2000, random_state=seed)
        ),
        "logistic_l1": lambda: SklearnBinaryModel(
            LogisticRegression(
                C=0.5, penalty="l1", solver="liblinear", max_iter=2000, random_state=seed
            )
        ),
        "tree_shallow": lambda: SklearnBinaryModel(
            DecisionTreeClassifier(max_depth=3, min_samples_leaf=10, random_state=seed)
        ),
    }
