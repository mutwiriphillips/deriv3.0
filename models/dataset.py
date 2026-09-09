"""
Builds labeled (features, outcome) examples for training a probability
model. Uses the exact same walk as backtest/simulator.py — features computed
from candles[:i+1] only — so a model trained here never saw the future
during training, matching Part 18's no-look-ahead requirement.

Label convention: 1 if price is higher `duration_candles` candles later
(a "CALL would have won"), 0 otherwise (including ties — matches the same
settlement assumption flagged in backtest/simulator.py).
"""
from __future__ import annotations

from app.features.engine import compute_features
from app.features.types import CandleSeries

# Fixed, ordered list of numeric feature keys used as the model's input
# vector. Booleans are cast to 0/1; anything not in this list (e.g. the
# string-valued "session", "tick_run_direction") is excluded here — a
# categorical encoder is a reasonable future addition, not done in this
# first pass per Part 16 (start simple, prove it beats the baseline first).
FEATURE_KEYS = [
    "returns", "log_returns", "rolling_returns_10", "price_acceleration",
    "distance_from_recent_high_20", "distance_from_recent_low_20",
    "ema_9", "ema_21", "ema_50", "ema_200", "ema_slope_9", "ma_separation",
    "rsi_14", "roc_10", "stochastic_k_14", "momentum_10",
    "atr_14", "rolling_std_20", "realized_volatility_20", "bollinger_width_20", "volatility_percentile",
    "higher_high", "lower_low", "higher_low", "lower_high",
    "breakout_proximity", "range_compression",
]


def features_to_vector(features: dict) -> list[float | None]:
    """None entries mean 'insufficient history' — the model layer decides how to impute, not this function."""
    vector = []
    for key in FEATURE_KEYS:
        val = features.get(key)
        if isinstance(val, bool):
            val = 1.0 if val else 0.0
        vector.append(val)
    return vector


def build_training_examples(
    timestamps: list[int],
    opens: list[float],
    highs: list[float],
    lows: list[float],
    closes: list[float],
    duration_candles: int,
    warmup: int = 60,
    symbol: str = "unknown",
    duration_s: int = 60,
) -> list[tuple[list[float | None], int]]:
    """Returns (feature_vector, label) pairs, one per walkable candle index."""
    n = len(closes)
    examples: list[tuple[list[float | None], int]] = []
    last_index = n - 1 - duration_candles

    for i in range(warmup, last_index + 1):
        series = CandleSeries(
            symbol=symbol,
            duration_s=duration_s,
            timestamps=timestamps[: i + 1],
            opens=opens[: i + 1],
            highs=highs[: i + 1],
            lows=lows[: i + 1],
            closes=closes[: i + 1],
        )
        features = compute_features(series)
        vector = features_to_vector(features)

        future_close = closes[i + duration_candles]
        label = 1 if future_close > closes[i] else 0
        examples.append((vector, label))

    return examples
