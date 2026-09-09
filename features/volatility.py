"""VOLATILITY features (spec Part 9)."""
from __future__ import annotations

import numpy as np

from app.features.types import CandleSeries


def true_ranges(series: CandleSeries) -> list[float]:
    """True range per candle (needs the prior close, so the first candle is skipped)."""
    trs = []
    for i in range(1, len(series)):
        h, l, prev_c = series.highs[i], series.lows[i], series.closes[i - 1]
        trs.append(max(h - l, abs(h - prev_c), abs(l - prev_c)))
    return trs


def atr(series: CandleSeries, period: int = 14) -> float | None:
    trs = true_ranges(series)
    if len(trs) < period:
        return None
    return float(np.mean(trs[-period:]))


def rolling_std(series: CandleSeries, period: int = 20) -> float | None:
    closes = series.closes[-period:]
    if len(closes) < period:
        return None
    return float(np.std(closes, ddof=0))


def realized_volatility(series: CandleSeries, period: int = 20) -> float | None:
    """Std of log returns over the window — the standard "realized vol" definition."""
    closes = series.closes[-(period + 1):]
    if len(closes) < period + 1:
        return None
    log_rets = np.diff(np.log(np.asarray(closes, dtype=float)))
    return float(np.std(log_rets, ddof=0))


def bollinger_band_width(series: CandleSeries, period: int = 20, num_std: float = 2.0) -> float | None:
    closes = series.closes[-period:]
    if len(closes) < period:
        return None
    mean = np.mean(closes)
    std = np.std(closes, ddof=0)
    if mean == 0:
        return None
    upper, lower = mean + num_std * std, mean - num_std * std
    return float((upper - lower) / mean)


def volatility_percentile(series: CandleSeries, lookback: int = 100, period: int = 20) -> float | None:
    """
    Where the current rolling std sits within its own recent history (0-100).
    Needs `lookback + period` candles; returns None below that.
    """
    if len(series) < lookback + period:
        return None
    closes = series.closes
    window_stds = [
        float(np.std(closes[i - period:i], ddof=0))
        for i in range(len(closes) - lookback, len(closes) + 1)
        if i >= period
    ]
    if not window_stds:
        return None
    current = window_stds[-1]
    rank = sum(1 for s in window_stds if s <= current) / len(window_stds)
    return float(rank * 100)
