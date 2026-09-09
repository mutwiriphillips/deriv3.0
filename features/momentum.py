"""MOMENTUM features (spec Part 9): RSI, ROC, stochastic, momentum, acceleration."""
from __future__ import annotations

import numpy as np

from app.features.types import CandleSeries


def rsi(series: CandleSeries, period: int = 14) -> float | None:
    """Wilder's RSI. Returns None with insufficient history."""
    closes = series.closes
    if len(closes) < period + 1:
        return None
    deltas = np.diff(np.asarray(closes, dtype=float))
    gains = np.where(deltas > 0, deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)

    avg_gain = gains[:period].mean()
    avg_loss = losses[:period].mean()
    for g, l in zip(gains[period:], losses[period:]):
        avg_gain = (avg_gain * (period - 1) + g) / period
        avg_loss = (avg_loss * (period - 1) + l) / period

    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return float(100 - (100 / (1 + rs)))


def roc(series: CandleSeries, period: int = 10) -> float | None:
    """Rate of change over `period` candles, as a fraction (not percent)."""
    closes = series.closes
    if len(closes) <= period or closes[-period - 1] == 0:
        return None
    return (closes[-1] - closes[-period - 1]) / closes[-period - 1]


def stochastic_k(series: CandleSeries, period: int = 14) -> float | None:
    """%K of the stochastic oscillator, in [0, 100]."""
    if len(series) < period:
        return None
    highs = series.highs[-period:]
    lows = series.lows[-period:]
    highest, lowest = max(highs), min(lows)
    if highest == lowest:
        return 50.0  # flat range — neither overbought nor oversold
    return float((series.closes[-1] - lowest) / (highest - lowest) * 100)


def momentum(series: CandleSeries, period: int = 10) -> float | None:
    """Raw price difference N candles back (unnormalized, unlike ROC)."""
    closes = series.closes
    if len(closes) <= period:
        return None
    return closes[-1] - closes[-period - 1]
