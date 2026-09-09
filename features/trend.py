"""TREND features (spec Part 9): EMA 9/21/50/200, slopes, MA separation."""
from __future__ import annotations

import numpy as np

from app.features.types import CandleSeries

EMA_PERIODS = (9, 21, 50, 200)


def ema(values: list[float], period: int) -> float | None:
    """Standard EMA over `values`, seeded with an SMA of the first `period` values.
    Returns None if there isn't enough history — callers must not silently
    treat that as zero, since zero is a valid (if unlikely) EMA value."""
    if len(values) < period:
        return None
    arr = np.asarray(values, dtype=float)
    alpha = 2.0 / (period + 1)
    ema_val = arr[:period].mean()
    for price in arr[period:]:
        ema_val = alpha * price + (1 - alpha) * ema_val
    return float(ema_val)


def all_emas(series: CandleSeries) -> dict[str, float | None]:
    return {f"ema_{p}": ema(series.closes, p) for p in EMA_PERIODS}


def ema_slope(series: CandleSeries, period: int, lookback: int = 5) -> float | None:
    """Change in EMA over the last `lookback` candles, normalized by price level."""
    if len(series.closes) < period + lookback:
        return None
    ema_now = ema(series.closes, period)
    ema_before = ema(series.closes[:-lookback], period)
    if ema_now is None or ema_before is None or ema_before == 0:
        return None
    return (ema_now - ema_before) / ema_before


def ma_separation(series: CandleSeries, fast: int = 9, slow: int = 50) -> float | None:
    """Normalized gap between a fast and slow EMA — a simple trend-strength proxy."""
    ema_fast = ema(series.closes, fast)
    ema_slow = ema(series.closes, slow)
    if ema_fast is None or ema_slow is None or ema_slow == 0:
        return None
    return (ema_fast - ema_slow) / ema_slow
