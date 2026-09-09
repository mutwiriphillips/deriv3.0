import pytest

from app.strategies.base import Direction
from app.strategies.baselines import (
    MajorityClassStrategy,
    RandomStrategy,
    SimpleMomentumStrategy,
    SimpleTrendStrategy,
)
from app.markets.context import MarketContext, MarketType, Regime


def make_ctx(regime=Regime.RANGE, data_quality_score=3, signal_features=None):
    from datetime import datetime, timezone
    return MarketContext(
        symbol="frxEURUSD",
        market_type=MarketType.FOREX,
        timestamp=datetime.now(timezone.utc),
        price=1.085,
        recent_ticks=[],
        candles={},
        volatility=0.001,
        trend=0.0,
        momentum=50.0,
        regime=regime,
        available_contracts=["CALL", "PUT"],
        signal_features=signal_features or {},
        data_quality_score=data_quality_score,
    )


# --- shared data-quality gate ---

def test_gate_blocks_low_quality_data_for_any_strategy():
    ctx = make_ctx(data_quality_score=0)
    for strategy in (RandomStrategy(seed=1), SimpleMomentumStrategy(), SimpleTrendStrategy()):
        result = strategy.evaluate(ctx)
        assert result.direction == Direction.NO_TRADE
        assert "DATA_QUALITY" in result.reasons[0]


# --- RandomStrategy ---

def test_random_strategy_picks_call_or_put_above_quality_threshold():
    ctx = make_ctx(data_quality_score=3)
    result = RandomStrategy(seed=42).evaluate(ctx)
    assert result.direction in (Direction.CALL, Direction.PUT)
    assert result.model_probability == pytest.approx(0.5)
    assert result.confidence < 0.2  # a coin flip should never claim much confidence


def test_random_strategy_is_deterministic_given_a_seed():
    ctx = make_ctx(data_quality_score=3)
    r1 = RandomStrategy(seed=7).evaluate(ctx)
    r2 = RandomStrategy(seed=7).evaluate(ctx)
    assert r1.direction == r2.direction


# --- MajorityClassStrategy ---

def test_majority_class_fit_picks_the_more_common_direction():
    outcomes = [Direction.CALL] * 60 + [Direction.PUT] * 40
    strategy = MajorityClassStrategy.fit(outcomes)
    assert strategy.majority_direction == Direction.CALL
    assert strategy.majority_probability == pytest.approx(0.6)
    assert strategy.n_samples == 100


def test_majority_class_fit_rejects_empty_history():
    with pytest.raises(ValueError):
        MajorityClassStrategy.fit([])


def test_majority_class_confidence_scales_with_sample_size_but_is_capped():
    small = MajorityClassStrategy.fit([Direction.CALL] * 6 + [Direction.PUT] * 4)
    large = MajorityClassStrategy.fit([Direction.CALL] * 6000 + [Direction.PUT] * 4000)
    ctx = make_ctx(data_quality_score=3)
    r_small = small.evaluate(ctx)
    r_large = large.evaluate(ctx)
    assert r_small.confidence < r_large.confidence
    assert r_large.confidence <= 0.6  # capped per Part 15 — big samples still don't earn full trust from a simple model


# --- SimpleMomentumStrategy ---

def test_momentum_strategy_calls_on_positive_roc():
    ctx = make_ctx(signal_features={"roc_10": 0.01}, data_quality_score=3)
    result = SimpleMomentumStrategy().evaluate(ctx)
    assert result.direction == Direction.CALL
    assert result.model_probability > 0.5


def test_momentum_strategy_puts_on_negative_roc():
    ctx = make_ctx(signal_features={"roc_10": -0.01}, data_quality_score=3)
    result = SimpleMomentumStrategy().evaluate(ctx)
    assert result.direction == Direction.PUT


def test_momentum_strategy_no_trade_when_roc_missing():
    ctx = make_ctx(signal_features={}, data_quality_score=3)
    result = SimpleMomentumStrategy().evaluate(ctx)
    assert result.direction == Direction.NO_TRADE


def test_momentum_probability_never_reaches_max_cap():
    # Even an absurdly large ROC should not push probability to/past 0.85
    ctx = make_ctx(signal_features={"roc_10": 100.0}, data_quality_score=3)
    result = SimpleMomentumStrategy().evaluate(ctx)
    assert result.model_probability < 0.85


# --- SimpleTrendStrategy ---

def test_trend_strategy_no_trade_outside_trend_regimes():
    ctx = make_ctx(regime=Regime.RANGE, signal_features={"ma_separation": 0.02}, data_quality_score=3)
    result = SimpleTrendStrategy().evaluate(ctx)
    assert result.direction == Direction.NO_TRADE
    assert "regime_not_allowed" in result.reasons[0]


def test_trend_strategy_calls_in_trend_up_regime():
    ctx = make_ctx(regime=Regime.TREND_UP, signal_features={"ma_separation": 0.02}, data_quality_score=3)
    result = SimpleTrendStrategy().evaluate(ctx)
    assert result.direction == Direction.CALL


def test_trend_strategy_puts_in_trend_down_regime():
    ctx = make_ctx(regime=Regime.TREND_DOWN, signal_features={"ma_separation": -0.02}, data_quality_score=3)
    result = SimpleTrendStrategy().evaluate(ctx)
    assert result.direction == Direction.PUT


def test_trend_strategy_no_trade_when_feature_missing():
    ctx = make_ctx(regime=Regime.TREND_UP, signal_features={}, data_quality_score=3)
    result = SimpleTrendStrategy().evaluate(ctx)
    assert result.direction == Direction.NO_TRADE
    assert "insufficient_history" in result.reasons[0]
