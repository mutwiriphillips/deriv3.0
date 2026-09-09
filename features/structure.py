"""STRUCTURE features (spec Part 9)."""
from __future__ import annotations

from app.features.types import CandleSeries


def is_higher_high(series: CandleSeries, lookback: int = 5) -> bool:
    if len(series) < lookback + 1:
        return False
    return series.highs[-1] > max(series.highs[-lookback - 1:-1])


def is_lower_low(series: CandleSeries, lookback: int = 5) -> bool:
    if len(series) < lookback + 1:
        return False
    return series.lows[-1] < min(series.lows[-lookback - 1:-1])


def is_higher_low(series: CandleSeries, lookback: int = 5) -> bool:
    if len(series) < lookback + 1:
        return False
    return series.lows[-1] > min(series.lows[-lookback - 1:-1])


def is_lower_high(series: CandleSeries, lookback: int = 5) -> bool:
    if len(series) < lookback + 1:
        return False
    return series.highs[-1] < max(series.highs[-lookback - 1:-1])


def breakout_proximity(series: CandleSeries, lookback: int = 20) -> float | None:
    """
    How close current price sits to the recent range boundary it's approaching,
    as a fraction of the range width. 0 = at the boundary, 1 = at the opposite
    boundary (i.e. far from breaking out). None if there isn't a real range yet.
    """
    if len(series) < lookback:
        return None
    highs, lows = series.highs[-lookback:], series.lows[-lookback:]
    range_high, range_low = max(highs), min(lows)
    width = range_high - range_low
    if width == 0:
        return None
    price = series.closes[-1]
    dist_to_high = range_high - price
    dist_to_low = price - range_low
    return float(min(dist_to_high, dist_to_low) / width)


def range_compression(series: CandleSeries, short_window: int = 10, long_window: int = 50) -> float | None:
    """
    Ratio of recent range width to longer-term range width. < 1 means the
    market is compressing (a common precursor to a breakout regime).
    """
    if len(series) < long_window:
        return None
    short_highs, short_lows = series.highs[-short_window:], series.lows[-short_window:]
    long_highs, long_lows = series.highs[-long_window:], series.lows[-long_window:]
    long_width = max(long_highs) - min(long_lows)
    if long_width == 0:
        return None
    short_width = max(short_highs) - min(short_lows)
    return float(short_width / long_width)
