"""
Evaluation metrics for the probability model (spec Part 17). Deliberately
does NOT include plain accuracy as a first-class metric — the spec is
explicit that "the trading objective is expected value after payout
economics and risk," not classification accuracy.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.metrics import brier_score_loss, roc_auc_score


def brier_score(y_true: list[int], y_prob: list[float]) -> float:
    """Lower is better. 0.25 is what a constant p=0.5 model scores against a 50/50 true distribution."""
    return float(brier_score_loss(y_true, y_prob))


def roc_auc(y_true: list[int], y_prob: list[float]) -> float | None:
    """None (not 0.5) when the test set is single-class — AUC is undefined there, not "no better than chance"."""
    if len(set(y_true)) < 2:
        return None
    return float(roc_auc_score(y_true, y_prob))


@dataclass
class CalibrationBin:
    bin_low: float
    bin_high: float
    predicted_mean: float | None
    actual_rate: float | None
    count: int


def calibration_curve(y_true: list[int], y_prob: list[float], n_bins: int = 10) -> list[CalibrationBin]:
    """
    Reliability diagram data: for well-calibrated probabilities, actual_rate
    should track predicted_mean closely in every bin with enough samples.
    A bin with count=0 has predicted_mean/actual_rate=None rather than a
    misleading 0.0 — an empty bin says nothing about calibration there.
    """
    y_true_arr = np.asarray(y_true)
    y_prob_arr = np.asarray(y_prob)
    edges = np.linspace(0, 1, n_bins + 1)
    bins = []
    for i in range(n_bins):
        low, high = edges[i], edges[i + 1]
        if i == n_bins - 1:
            mask = (y_prob_arr >= low) & (y_prob_arr <= high)
        else:
            mask = (y_prob_arr >= low) & (y_prob_arr < high)
        count = int(mask.sum())
        if count == 0:
            bins.append(CalibrationBin(bin_low=float(low), bin_high=float(high), predicted_mean=None, actual_rate=None, count=0))
        else:
            bins.append(
                CalibrationBin(
                    bin_low=float(low),
                    bin_high=float(high),
                    predicted_mean=float(y_prob_arr[mask].mean()),
                    actual_rate=float(y_true_arr[mask].mean()),
                    count=count,
                )
            )
    return bins


def calibration_error(bins: list[CalibrationBin]) -> float | None:
    """Expected Calibration Error (ECE): sample-weighted mean |predicted - actual| across non-empty bins."""
    non_empty = [b for b in bins if b.count > 0]
    if not non_empty:
        return None
    total = sum(b.count for b in non_empty)
    return sum(b.count * abs(b.predicted_mean - b.actual_rate) for b in non_empty) / total
