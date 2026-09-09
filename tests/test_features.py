import numpy as np
import pytest

from app.features import momentum as mom
from app.features import price as prc
from app.features import structure as struct_
from app.features import trend as trend_
from app.features import volatility as vol
from app.features.engine import build_market_context, compute_features
from app.features.regime import classify_regime
from app.features.tick_features import consecutive_direction_run, tick_imbalance
from app.features.time_features import classify_session, Session
from app.features.types import CandleSeries
from app.markets.context import MarketType, Regime


def make_series(closes, highs=None, lows=None, opens=None, start_ts=1_700_000_000, step=60):
    n = len(closes)
    return CandleSeries(
        symbol="frxEURUSD",
        duration_s=step,
        timestamps=[start_ts + i * step for i in range(n)],
        opens=opens or closes,
        highs=highs or [c + 0.001 for c in closes],
        lows=lows or [c - 0.001 for c in closes],
        closes=closes,
    )


# --- types.py ---

def test_candle_series_rejects_mismatched_lengths():
    with pytest.raises(ValueError):
        CandleSeries(symbol="x", duration_s=60, timestamps=[1, 2], opens=[1], highs=[1, 2], lows=[1, 2], closes=[1, 2])


def test_candle_series_tail():
    s = make_series([1, 2, 3, 4, 5])
    t = s.tail(2)
    assert t.closes == [4, 5]
    assert len(t) == 2


# --- price.py ---

def test_returns_basic():
    s = make_series([1.0, 1.1])
    assert prc.returns(s) == pytest.approx(0.1)


def test_rolling_returns():
    s = make_series([1.0, 1.0, 1.0, 1.2])
    assert prc.rolling_returns(s, 3) == pytest.approx(0.2)


def test_distance_from_recent_high():
    s = make_series([1.0, 1.2, 1.1], highs=[1.0, 1.2, 1.1])
    d = prc.distance_from_recent_high(s, 3)
    assert d == pytest.approx((1.1 - 1.2) / 1.2)


# --- trend.py ---

def test_ema_matches_manual_calc():
    values = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    period = 3
    # manual EMA: seed = mean(first 3) = 2, then apply alpha=0.5 for remaining
    alpha = 2 / (period + 1)
    expected = sum(values[:period]) / period
    for v in values[period:]:
        expected = alpha * v + (1 - alpha) * expected
    assert trend_.ema(values, period) == pytest.approx(expected)


def test_ema_insufficient_history_returns_none():
    assert trend_.ema([1, 2], 5) is None


def test_ma_separation_positive_for_uptrend():
    # Strong uptrend: fast EMA should sit above slow EMA -> positive separation
    closes = list(np.linspace(1.0, 2.0, 60))
    s = make_series(closes)
    sep = trend_.ma_separation(s, fast=5, slow=20)
    assert sep is not None and sep > 0


# --- momentum.py ---

def test_rsi_all_gains_is_100():
    closes = list(range(1, 20))  # strictly increasing
    s = make_series([float(c) for c in closes])
    r = mom.rsi(s, period=14)
    assert r == pytest.approx(100.0)


def test_rsi_all_losses_is_0():
    closes = list(range(20, 1, -1))  # strictly decreasing
    s = make_series([float(c) for c in closes])
    r = mom.rsi(s, period=14)
    assert r == pytest.approx(0.0)


def test_stochastic_k_flat_range_is_50():
    s = make_series([1.0] * 20, highs=[1.0] * 20, lows=[1.0] * 20)
    assert mom.stochastic_k(s, period=14) == pytest.approx(50.0)


# --- volatility.py ---

def test_atr_needs_enough_history():
    s = make_series([1.0, 1.0, 1.0])
    assert vol.atr(s, period=14) is None


def test_bollinger_width_zero_for_flat_series():
    s = make_series([1.0] * 25)
    width = vol.bollinger_band_width(s, period=20)
    assert width == pytest.approx(0.0)


# --- structure.py ---

def test_higher_high_detected():
    closes = [1.0] * 5 + [1.5]
    highs = [1.0] * 5 + [1.5]
    s = make_series(closes, highs=highs)
    assert struct_.is_higher_high(s, lookback=5) is True


def test_breakout_proximity_none_without_range():
    s = make_series([1.0, 1.0])
    assert struct_.breakout_proximity(s, lookback=20) is None


# --- tick_features.py ---

def test_consecutive_direction_run():
    ticks = [(1, 1.0), (2, 1.01), (3, 1.02), (4, 1.03)]
    direction, run = consecutive_direction_run(ticks)
    assert direction == "UP"
    assert run == 3


def test_tick_imbalance_all_up():
    ticks = [(i, float(i)) for i in range(5)]
    assert tick_imbalance(ticks) == pytest.approx(1.0)


# --- time_features.py ---

def test_classify_session_overlap():
    # 14:00 UTC on an arbitrary day -> London/NY overlap
    import datetime
    epoch = int(datetime.datetime(2026, 1, 5, 14, 0, tzinfo=datetime.timezone.utc).timestamp())
    assert classify_session(epoch) == Session.LONDON_NY_OVERLAP


# --- regime.py ---

def test_regime_unknown_without_enough_data():
    assert classify_regime(None, None, None, None, None) == Regime.UNKNOWN


def test_regime_high_volatility_overrides_trend():
    r = classify_regime(
        ma_separation=0.05,  # would otherwise read as strong TREND_UP
        ema_slope_fast=0.01,
        volatility_percentile=95.0,
        range_compression=None,
        breakout_proximity=None,
    )
    assert r == Regime.HIGH_VOLATILITY


def test_regime_trend_up():
    r = classify_regime(
        ma_separation=0.01,
        ema_slope_fast=0.001,
        volatility_percentile=50.0,
        range_compression=None,
        breakout_proximity=None,
    )
    assert r == Regime.TREND_UP


def test_regime_range_when_flat():
    r = classify_regime(
        ma_separation=0.0,
        ema_slope_fast=0.0,
        volatility_percentile=50.0,
        range_compression=None,
        breakout_proximity=None,
    )
    assert r == Regime.RANGE


# --- engine.py ---

def test_compute_features_returns_all_expected_keys():
    s = make_series(list(np.linspace(1.0, 1.1, 60)))
    features = compute_features(s)
    for key in ("returns", "ema_9", "rsi_14", "atr_14", "higher_high", "hour", "session"):
        assert key in features


def test_build_market_context_uses_only_given_series():
    """
    The structural no-look-ahead guarantee: build_market_context has no way
    to reach data beyond `series` because it never receives a symbol+time to
    look anything up with — only the pre-sliced series itself.
    """
    closes = list(np.linspace(1.0, 1.05, 60))
    full_series = make_series(closes)
    truncated_series = make_series(closes[:30])

    ctx_full = build_market_context("frxEURUSD", MarketType.FOREX, full_series, ["CALL", "PUT"])
    ctx_truncated = build_market_context("frxEURUSD", MarketType.FOREX, truncated_series, ["CALL", "PUT"])

    # Different input series -> different price/timestamp; proves no hidden global lookup happened
    assert ctx_full.price != ctx_truncated.price
    assert ctx_full.timestamp != ctx_truncated.timestamp
    assert ctx_full.regime in Regime
    assert ctx_truncated.data_quality_score == 0
