"""PRICE features (spec Part 9). All operate on CandleSeries.closes only."""
from __future__ import annotations

import numpy as np

from app.features.types import CandleSeries


def returns(series: CandleSeries) -> float:
    """Simple return over the last candle. 0.0 if fewer than 2 candles."""
    c = series.closes
    if len(c) < 2 or c[-2] == 0:
        return 0.0
    return (c[-1] - c[-2]) / c[-2]


def log_returns(series: CandleSeries) -> float:
    c = series.closes
    if len(c) < 2 or c[-2] <= 0 or c[-1] <= 0:
        return 0.0
    return float(np.log(c[-1] / c[-2]))


def rolling_returns(series: CandleSeries, window: int) -> float:
    """Return over the last `window` candles (not just one)."""
    c = series.closes
    if len(c) <= window or c[-window - 1] == 0:
        return 0.0
    return (c[-1] - c[-window - 1]) / c[-window - 1]


def price_acceleration(series: CandleSeries) -> float:
    """Change in return, i.e. the discrete second derivative of price."""
    c = series.closes
    if len(c) < 3:
        return 0.0
    r1 = (c[-1] - c[-2]) / c[-2] if c[-2] else 0.0
    r2 = (c[-2] - c[-3]) / c[-3] if c[-3] else 0.0
    return r1 - r2


def distance_from_recent_high(series: CandleSeries, window: int) -> float:
    """(current close - rolling max high) / rolling max high, over the last `window` candles."""
    highs = series.highs[-window:]
    if not highs:
        return 0.0
    recent_high = max(highs)
    if recent_high == 0:
        return 0.0
    return (series.closes[-1] - recent_high) / recent_high


def distance_from_recent_low(series: CandleSeries, window: int) -> float:
    lows = series.lows[-window:]
    if not lows:
        return 0.0
    recent_low = min(lows)
    if recent_low == 0:
        return 0.0
    return (series.closes[-1] - recent_low) / recent_low
