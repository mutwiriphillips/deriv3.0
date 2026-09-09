"""
Regime classification (spec Part 11). First implementation is deterministic
rules over already-computed features, exactly as the spec directs — an ML/
statistical classifier can be compared against this later, never replacing
it silently.
"""
from __future__ import annotations

from app.markets.context import Regime


def classify_regime(
    ma_separation: float | None,
    ema_slope_fast: float | None,
    volatility_percentile: float | None,
    range_compression: float | None,
    breakout_proximity: float | None,
    trend_threshold: float = 0.001,
    high_vol_percentile: float = 80.0,
    low_vol_percentile: float = 20.0,
    compression_threshold: float = 0.5,
    breakout_proximity_threshold: float = 0.1,
) -> Regime:
    """
    Every input may be None (insufficient history) — classify_regime always
    returns UNKNOWN rather than guessing when it doesn't have enough to say.
    Order of checks matters: volatility extremes are checked before trend,
    since a HIGH_VOLATILITY regime overrides a nominal trend read.
    """
    if volatility_percentile is None or ma_separation is None:
        return Regime.UNKNOWN

    if volatility_percentile >= high_vol_percentile:
        return Regime.HIGH_VOLATILITY
    if volatility_percentile <= low_vol_percentile:
        return Regime.LOW_VOLATILITY

    if (
        range_compression is not None
        and breakout_proximity is not None
        and range_compression <= compression_threshold
        and breakout_proximity <= breakout_proximity_threshold
    ):
        return Regime.BREAKOUT

    if ma_separation >= trend_threshold and (ema_slope_fast is None or ema_slope_fast > 0):
        return Regime.TREND_UP
    if ma_separation <= -trend_threshold and (ema_slope_fast is None or ema_slope_fast < 0):
        return Regime.TREND_DOWN

    return Regime.RANGE
