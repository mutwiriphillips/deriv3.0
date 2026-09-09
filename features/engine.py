"""
Feature engine (spec Part 9): the single place that turns a CandleSeries (and
optionally recent ticks) — both already sliced to "now" — into the
`signal_features` dict and the summary fields (trend, momentum, volatility,
regime) that populate a MarketContext.

No function anywhere in app/features/ accepts a symbol+timestamp and looks up
its own history; everything is handed a pre-sliced series. That's what makes
the no-look-ahead guarantee (spec Part 18) structural rather than a discipline
someone has to remember to uphold.
"""
from __future__ import annotations

from datetime import datetime, timezone

from app.features import momentum as mom
from app.features import price as prc
from app.features import structure as struct_
from app.features import trend as trend_
from app.features import volatility as vol
from app.features.regime import classify_regime
from app.features.time_features import time_features
from app.features.tick_features import consecutive_direction_run, reversal_frequency, tick_imbalance
from app.features.types import CandleSeries
from app.markets.context import MarketContext, MarketType


def compute_features(series: CandleSeries, ticks: list[tuple[int, float]] | None = None) -> dict:
    """Every candidate feature from spec Part 9, keyed by name. None where insufficient history."""
    features: dict = {
        # PRICE
        "returns": prc.returns(series),
        "log_returns": prc.log_returns(series),
        "rolling_returns_10": prc.rolling_returns(series, 10),
        "price_acceleration": prc.price_acceleration(series),
        "distance_from_recent_high_20": prc.distance_from_recent_high(series, 20),
        "distance_from_recent_low_20": prc.distance_from_recent_low(series, 20),
        # TREND
        **trend_.all_emas(series),
        "ema_slope_9": trend_.ema_slope(series, 9),
        "ma_separation": trend_.ma_separation(series),
        # MOMENTUM
        "rsi_14": mom.rsi(series),
        "roc_10": mom.roc(series),
        "stochastic_k_14": mom.stochastic_k(series),
        "momentum_10": mom.momentum(series),
        # VOLATILITY
        "atr_14": vol.atr(series),
        "rolling_std_20": vol.rolling_std(series),
        "realized_volatility_20": vol.realized_volatility(series),
        "bollinger_width_20": vol.bollinger_band_width(series),
        "volatility_percentile": vol.volatility_percentile(series),
        # STRUCTURE
        "higher_high": struct_.is_higher_high(series),
        "lower_low": struct_.is_lower_low(series),
        "higher_low": struct_.is_higher_low(series),
        "lower_high": struct_.is_lower_high(series),
        "breakout_proximity": struct_.breakout_proximity(series),
        "range_compression": struct_.range_compression(series),
    }

    if series.timestamps:
        features.update(time_features(series.timestamps[-1]))

    if ticks:
        direction, run_len = consecutive_direction_run(ticks)
        features["tick_run_direction"] = direction
        features["tick_run_length"] = run_len
        features["tick_imbalance"] = tick_imbalance(ticks)
        features["reversal_frequency"] = reversal_frequency(ticks)

    return features


def build_market_context(
    symbol: str,
    market_type: MarketType,
    series: CandleSeries,
    available_contracts: list[str],
    ticks: list[tuple[int, float]] | None = None,
    payout: dict[str, float] | None = None,
    data_quality_score: int = 0,
) -> MarketContext:
    """
    Assembles a MarketContext strictly from `series` (and optional `ticks`),
    both assumed already truncated to the instant this context represents.
    """
    features = compute_features(series, ticks)

    regime = classify_regime(
        ma_separation=features.get("ma_separation"),
        ema_slope_fast=features.get("ema_slope_9"),
        volatility_percentile=features.get("volatility_percentile"),
        range_compression=features.get("range_compression"),
        breakout_proximity=features.get("breakout_proximity"),
    )

    last_ts = series.timestamps[-1] if series.timestamps else int(datetime.now(timezone.utc).timestamp())

    return MarketContext(
        symbol=symbol,
        market_type=market_type,
        timestamp=datetime.fromtimestamp(last_ts, tz=timezone.utc),
        price=series.closes[-1] if series.closes else 0.0,
        recent_ticks=[p for _, p in (ticks or [])],
        candles={series.duration_s: [
            {"open": o, "high": h, "low": l, "close": c, "timestamp": t}
            for o, h, l, c, t in zip(series.opens, series.highs, series.lows, series.closes, series.timestamps)
        ]},
        volatility=features.get("realized_volatility_20") or 0.0,
        trend=features.get("ma_separation") or 0.0,
        momentum=features.get("rsi_14") or 0.0,
        regime=regime,
        available_contracts=available_contracts,
        payout=payout,
        signal_features=features,
        data_quality_score=data_quality_score,
    )
