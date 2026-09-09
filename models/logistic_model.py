"""
Logistic regression probability model (spec Part 17): "Use ML only when
justified. Start with Logistic Regression... Use probability calibration."

Missing features (None -> insufficient history) are median-imputed using
medians computed from the TRAINING set only — never from data being
predicted on, which would leak information from evaluation/live data back
into a value the model treats as "known."
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler



@dataclass
class FitResult:
    n_train: int
    class_balance: float   # fraction of label==1 in training data
    feature_medians: list[float] = field(default_factory=list)


class LogisticProbabilityModel:
    """
    Usage:
        model = LogisticProbabilityModel()
        fit_result = model.fit(training_examples)
        p_up = model.predict_proba(feature_vector)
    """

    def __init__(self, calibrate: bool = True, cv: int = 3):
        self._calibrate = calibrate
        self._cv = cv
        self._pipeline = None
        self._feature_medians: np.ndarray | None = None

    def fit(self, examples: list[tuple[list[float | None], int]]) -> FitResult:
        if len(examples) < 10:
            raise ValueError("need at least 10 training examples")

        X_raw = np.array([ex[0] for ex in examples], dtype=object)
        y = np.array([ex[1] for ex in examples], dtype=int)

        # Median-impute per column, using ONLY this training data.
        medians = np.zeros(X_raw.shape[1])
        X = np.zeros(X_raw.shape, dtype=float)
        for col in range(X_raw.shape[1]):
            col_values = [v for v in X_raw[:, col] if v is not None]
            median = float(np.median(col_values)) if col_values else 0.0
            medians[col] = median
            X[:, col] = [float(v) if v is not None else median for v in X_raw[:, col]]

        self._feature_medians = medians

        base = LogisticRegression(max_iter=1000)
        if self._calibrate:
            classifier = CalibratedClassifierCV(base, method="sigmoid", cv=min(self._cv, self._effective_cv(y)))
        else:
            classifier = base

        self._pipeline = make_pipeline(StandardScaler(), classifier)
        self._pipeline.fit(X, y)

        return FitResult(
            n_train=len(examples),
            class_balance=float(y.mean()),
            feature_medians=medians.tolist(),
        )

    @staticmethod
    def _effective_cv(y: np.ndarray) -> int:
        """CalibratedClassifierCV needs at least 2 examples of the minority class per fold."""
        counts = np.bincount(y)
        min_class_count = counts.min() if len(counts) > 1 else 1
        return max(2, min(3, min_class_count))

    def predict_proba(self, feature_vector: list[float | None]) -> float:
        """Returns calibrated P(price higher after `duration_candles`), i.e. P(CALL would win)."""
        if self._pipeline is None or self._feature_medians is None:
            raise RuntimeError("model has not been fit yet")
        x = np.array(
            [float(v) if v is not None else self._feature_medians[i] for i, v in enumerate(feature_vector)],
            dtype=float,
        ).reshape(1, -1)
        return float(self._pipeline.predict_proba(x)[0, 1])
