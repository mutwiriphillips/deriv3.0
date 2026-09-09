import numpy as np
import pytest

from app.backtest.simulator import BacktestResult, TradeRecord, run_backtest
from app.markets.context import MarketType, Regime
from app.strategies.base import Direction, TradeCandidate
from app.strategies.baselines import SimpleTrendStrategy


class AlwaysCallStrategy:
    """Deterministic dummy: always predicts CALL with a fixed high probability."""
    strategy_id = "test_always_call"
    allowed_regimes = set(Regime)

    def evaluate(self, ctx):
        return TradeCandidate(direction=Direction.CALL, model_probability=0.9, confidence=1.0, signal_score=0.4, reasons=["test"])


class AlwaysNoTradeStrategy:
    strategy_id = "test_never_trades"
    allowed_regimes = set(Regime)

    def evaluate(self, ctx):
        return TradeCandidate(direction=Direction.NO_TRADE, model_probability=0.5, confidence=0.0, signal_score=0.0, reasons=["test"])


def make_price_series(n, kind="up", seed=42):
    if kind == "up":
        rng = np.random.default_rng(seed)
        steps = rng.normal(loc=0.0008, scale=0.003, size=n)
        closes = list(1.0 + np.cumsum(steps))
    elif kind == "strict_up":
        closes = list(np.linspace(1.0, 1.5, n))
    elif kind == "flat":
        closes = [1.0 + 0.0001 * ((-1) ** i) for i in range(n)]  # tiny noise, no real trend
    else:
        raise ValueError(kind)
    timestamps = [1_700_000_000 + i * 60 for i in range(n)]
    highs = [c + 0.0015 for c in closes]
    lows = [c - 0.0015 for c in closes]
    opens = closes
    return timestamps, opens, highs, lows, closes


# --- BacktestResult math, independent of run_backtest's walk logic ---

def make_trade(profit_loss, result, stake=1.0):
    return TradeRecord(
        entry_index=0, entry_timestamp=0, direction=Direction.CALL, model_probability=0.6,
        payout_ratio=0.8, stake=stake, probability_edge=0.05, expected_value=0.05,
        exit_index=1, exit_timestamp=60, entry_price=1.0, exit_price=1.01,
        result=result, profit_loss=profit_loss,
    )


def test_backtest_result_basic_stats():
    r = BacktestResult()
    r.trades = [make_trade(0.8, "WIN"), make_trade(-1.0, "LOSS"), make_trade(0.8, "WIN")]
    assert r.total_trades == 3
    assert r.wins == 2
    assert r.losses == 1
    assert r.win_rate == pytest.approx(2 / 3)
    assert r.total_profit_loss == pytest.approx(0.6)


def test_backtest_result_profit_factor_undefined_with_no_losses():
    r = BacktestResult()
    r.trades = [make_trade(0.8, "WIN"), make_trade(0.8, "WIN")]
    assert r.profit_factor is None


def test_backtest_result_profit_factor_normal_case():
    r = BacktestResult()
    r.trades = [make_trade(0.8, "WIN"), make_trade(-1.0, "LOSS")]
    assert r.profit_factor == pytest.approx(0.8 / 1.0)


def test_backtest_result_max_drawdown():
    r = BacktestResult()
    # equity path: 0 -> 1.0 -> 0.5 -> -0.5 -> 2.0 ; peak 1.0 then trough -0.5 => drawdown 1.5
    r.trades = [make_trade(1.0, "WIN"), make_trade(-0.5, "LOSS"), make_trade(-1.0, "LOSS"), make_trade(2.5, "WIN")]
    assert r.max_drawdown == pytest.approx(1.5)


def test_backtest_result_streaks():
    r = BacktestResult()
    r.trades = [
        make_trade(1, "WIN"), make_trade(1, "WIN"), make_trade(-1, "LOSS"),
        make_trade(-1, "LOSS"), make_trade(-1, "LOSS"), make_trade(1, "WIN"),
    ]
    assert r.longest_winning_streak == 2
    assert r.longest_losing_streak == 3


# --- run_backtest end-to-end ---

def test_run_backtest_all_wins_on_monotonic_uptrend():
    timestamps, opens, highs, lows, closes = make_price_series(300, kind="strict_up")
    result = run_backtest(
        symbol="frxEURUSD", market_type=MarketType.FOREX, duration_s=60,
        timestamps=timestamps, opens=opens, highs=highs, lows=lows, closes=closes,
        strategy=AlwaysCallStrategy(), duration_candles=1, payout_ratio=0.8, stake=1.0,
        warmup=120,
    )
    assert result.total_trades > 0
    assert result.win_rate == pytest.approx(1.0)  # monotonic uptrend -> every CALL wins
    assert result.max_drawdown == pytest.approx(0.0)


def test_run_backtest_never_trades_yields_zero_trades():
    timestamps, opens, highs, lows, closes = make_price_series(200, kind="strict_up")
    result = run_backtest(
        symbol="frxEURUSD", market_type=MarketType.FOREX, duration_s=60,
        timestamps=timestamps, opens=opens, highs=highs, lows=lows, closes=closes,
        strategy=AlwaysNoTradeStrategy(), duration_candles=1, payout_ratio=0.8,
        warmup=120,
    )
    assert result.total_trades == 0
    assert result.no_trade_count > 0


def test_run_backtest_edge_gate_blocks_low_edge_trades():
    timestamps, opens, highs, lows, closes = make_price_series(300, kind="strict_up")
    # AlwaysCallStrategy gives model_probability=0.9 -> at payout 0.8, edge is huge and positive;
    # setting an impossibly high min_probability_edge should block every trade.
    result = run_backtest(
        symbol="frxEURUSD", market_type=MarketType.FOREX, duration_s=60,
        timestamps=timestamps, opens=opens, highs=highs, lows=lows, closes=closes,
        strategy=AlwaysCallStrategy(), duration_candles=1, payout_ratio=0.8,
        min_probability_edge=0.99, warmup=120,
    )
    assert result.total_trades == 0


def test_run_backtest_respects_strategy_allowed_regimes():
    # A flat/noisy series should mostly classify as RANGE, not TREND_UP/DOWN,
    # so SimpleTrendStrategy (which only trades in trend regimes) should take
    # few or no trades compared to a real trend series.
    timestamps, opens, highs, lows, closes = make_price_series(300, kind="flat")
    flat_result = run_backtest(
        symbol="frxEURUSD", market_type=MarketType.FOREX, duration_s=60,
        timestamps=timestamps, opens=opens, highs=highs, lows=lows, closes=closes,
        strategy=SimpleTrendStrategy(), duration_candles=1, payout_ratio=0.8,
        warmup=120,
    )

    timestamps2, opens2, highs2, lows2, closes2 = make_price_series(300, kind="up", seed=7)
    trend_result = run_backtest(
        symbol="frxEURUSD", market_type=MarketType.FOREX, duration_s=60,
        timestamps=timestamps2, opens=opens2, highs=highs2, lows=lows2, closes=closes2,
        strategy=SimpleTrendStrategy(), duration_candles=1, payout_ratio=0.8,
        warmup=120,
    )
    assert trend_result.total_trades > flat_result.total_trades


def test_run_backtest_no_out_of_range_access():
    """duration_candles near the end of the series must not cause an index error."""
    timestamps, opens, highs, lows, closes = make_price_series(130, kind="strict_up")
    result = run_backtest(
        symbol="frxEURUSD", market_type=MarketType.FOREX, duration_s=60,
        timestamps=timestamps, opens=opens, highs=highs, lows=lows, closes=closes,
        strategy=AlwaysCallStrategy(), duration_candles=5, payout_ratio=0.8,
        warmup=120,
    )
    # Just needs to complete without raising; only a handful of candles are even walkable.
    assert result.total_trades >= 0
