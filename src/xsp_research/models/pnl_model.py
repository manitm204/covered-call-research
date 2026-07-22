"""Benchmark regression models for expected-P&L and adverse-excursion targets.

Same discipline as the probability suite: the training-mean predictor is the
bar; fold-local preprocessing; complexity added only when it survives
walk-forward comparison after costs.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Protocol

import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor


class RegressionModel(Protocol):
    def fit(self, x: np.ndarray, y: np.ndarray) -> None: ...
    def predict(self, x: np.ndarray) -> np.ndarray: ...


class TrainMeanModel:
    def __init__(self) -> None:
        self._mean: float | None = None

    def fit(self, x: np.ndarray, y: np.ndarray) -> None:
        self._mean = float(np.mean(y))

    def predict(self, x: np.ndarray) -> np.ndarray:
        if self._mean is None:
            raise RuntimeError("fit first")
        return np.full(len(x), self._mean)


class SklearnRegressionModel:
    def __init__(self, estimator) -> None:
        self._pipe = Pipeline(
            [
                ("impute", SimpleImputer(strategy="median", keep_empty_features=True)),
                ("scale", StandardScaler()),
                ("model", estimator),
            ]
        )

    def fit(self, x: np.ndarray, y: np.ndarray) -> None:
        self._pipe.fit(x, y)

    def predict(self, x: np.ndarray) -> np.ndarray:
        return self._pipe.predict(x)


RegressorFactory = Callable[[], RegressionModel]


def regression_model_factories(seed: int = 7) -> dict[str, RegressorFactory]:
    return {
        "train_mean": TrainMeanModel,
        "ridge": lambda: SklearnRegressionModel(Ridge(alpha=1.0, random_state=seed)),
        "tree_shallow": lambda: SklearnRegressionModel(
            DecisionTreeRegressor(max_depth=3, min_samples_leaf=10, random_state=seed)
        ),
    }
