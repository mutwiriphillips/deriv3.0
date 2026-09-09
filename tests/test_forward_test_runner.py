import numpy as np
from unittest.mock import AsyncMock

import pytest

from app.execution.order_manager import OrderManager, OrderManagerConfig
from app.markets.context import MarketType
from app.monitoring.session_stats import SessionStats
from app.orchestration.forward_test_runner import ForwardTestRunner
from app.risk.exposure import ExposureManager
from app.risk.governor import RiskGovernor
from app.risk.overtrading import CooldownTracker, DuplicateSignalGuard, RateLimiter
from app.strategies.base import Direction, TradeCandidate
from app.features.types import CandleSeries


class AlwaysCallStrategy:
    strategy_id = "test_always_call"

    def evaluate(self, ctx):
        return TradeCandidate(direction=Direction.CALL, model_probability=0.65, confidence=0.5, signal_score=0.15, reasons=["x"])


class AlwaysNoTradeStrategy:
    strategy_id = "test_never"

    def evaluate(self, ctx):
        return TradeCandidate(direction=Direction.NO_TRADE, model_probability=0.5, confidence=0.0, signal_score=0.0, reasons=["x"])


def make_series(n=200, seed=1):
    rng = np.random.default_rng(seed)
    closes = list(1.0 + np.cumsum(rng.normal(0, 0.002, n)))
    timestamps = [1_700_000_000 + i * 60 for i in range(n)]
    highs = [c + 0.001 for c in closes]
    lows = [c - 0.001 for c in closes]
    return CandleSeries(symbol="frxEURUSD", duration_s=60, timestamps=timestamps, opens=closes, highs=highs, lows=lows, closes=closes)


def make_runner(strategy):
    config = OrderManagerConfig(
        currency="USD", risk_per_trade=0.01, max_stake=100.0,
        min_probability_edge=0.0, min_expected_value=0.0, max_latency_ms=5000.0,
    )
    risk_governor = RiskGovernor(starting_balance=1000, max_daily_loss=1000, max_drawdown=0.9, max_consecutive_losses=100)
    order_manager = OrderManager(
        config=config, risk_governor=risk_governor,
        exposure_manager=ExposureManager(2.0), duplicate_guard=DuplicateSignalGuard(),
        cooldown_tracker=CooldownTracker(60), rate_limiter=RateLimiter(10, 50),
    )
    session_stats = SessionStats()
    public_client = AsyncMock()
    public_client.proposal = AsyncMock(return_value={"id": "prop-1", "ask_price": "10.00", "payout": "18.00"})
    auth_client = AsyncMock()
    auth_client.buy = AsyncMock(return_value={"contract_id": 999, "buy_price": 10.0, "payout": 18.0})

    runner = ForwardTestRunner(
        symbol="frxEURUSD", market_type=MarketType.FOREX, strategy=strategy, model_version="v1",
        order_manager=order_manager, risk_governor=risk_governor, session_stats=session_stats,
        public_client=public_client, auth_client=auth_client,
        duration_candles=1, duration_s=60,
    )
    return runner, session_stats, risk_governor, auth_client, public_client


@pytest.mark.asyncio
async def test_run_once_no_trade_case_produces_dashboard_with_reason():
    runner, session_stats, risk_governor, auth_client, public_client = make_runner(AlwaysNoTradeStrategy())
    series = make_series()

    state = await runner.run_once(series, signal_id="sig-1")

    assert state.no_trade_reason == "NO_TRADE_SIGNAL_INVALID"
    assert state.trades_today == 0
    public_client.proposal.assert_not_called()
    auth_client.buy.assert_not_called()


@pytest.mark.asyncio
async def test_run_once_trade_case_settles_and_updates_everything():
    runner, session_stats, risk_governor, auth_client, public_client = make_runner(AlwaysCallStrategy())
    series = make_series()
    auth_client.proposal_open_contract = AsyncMock(return_value={"is_sold": 1, "status": "won", "profit": 8.0})

    balance_before = risk_governor.current_balance
    state = await runner.run_once(series, signal_id="sig-1")

    assert state.no_trade_reason is None
    assert state.trades_today == 1
    assert state.win_rate == pytest.approx(1.0)
    assert risk_governor.current_balance == pytest.approx(balance_before + 8.0)
    assert session_stats.session_pnl == pytest.approx(8.0)
    public_client.proposal.assert_called_once()
    auth_client.buy.assert_called_once()


@pytest.mark.asyncio
async def test_run_once_trade_case_with_loss_settlement():
    runner, session_stats, risk_governor, auth_client, public_client = make_runner(AlwaysCallStrategy())
    series = make_series()
    auth_client.proposal_open_contract = AsyncMock(return_value={"is_sold": 1, "status": "lost", "profit": -10.0})

    balance_before = risk_governor.current_balance
    state = await runner.run_once(series, signal_id="sig-1")

    assert state.trades_today == 1
    assert state.win_rate == pytest.approx(0.0)
    assert risk_governor.current_balance == pytest.approx(balance_before - 10.0)
    assert risk_governor.consecutive_losses == 1


@pytest.mark.asyncio
async def test_run_once_dashboard_reflects_current_regime_and_symbol():
    runner, *_ = make_runner(AlwaysNoTradeStrategy())
    series = make_series()
    state = await runner.run_once(series, signal_id="sig-1")
    assert state.current_symbol == "frxEURUSD"
    assert state.current_regime is not None


@pytest.mark.asyncio
async def test_run_once_invokes_on_settlement_callback_with_correct_data():
    runner, session_stats, risk_governor, auth_client, public_client = make_runner(AlwaysCallStrategy())
    series = make_series()
    auth_client.proposal_open_contract = AsyncMock(return_value={"is_sold": 1, "status": "won", "profit": 8.0})

    calls = []
    runner.on_settlement = lambda symbol, direction, result, profit, stake, latency: calls.append(
        (symbol, direction, result, profit, stake, latency)
    )

    await runner.run_once(series, signal_id="sig-1")

    assert len(calls) == 1
    symbol, direction, result, profit, stake, latency = calls[0]
    assert symbol == "frxEURUSD"
    assert direction == "CALL"
    assert result == "WIN"
    assert profit == pytest.approx(8.0)
    assert stake == pytest.approx(10.0)


@pytest.mark.asyncio
async def test_run_once_no_trade_never_invokes_on_settlement():
    runner, session_stats, risk_governor, auth_client, public_client = make_runner(AlwaysNoTradeStrategy())
    series = make_series()

    calls = []
    runner.on_settlement = lambda *args: calls.append(args)

    await runner.run_once(series, signal_id="sig-1")

    assert calls == []


@pytest.mark.asyncio
async def test_run_once_invokes_on_trade_placed_before_settlement_known():
    runner, session_stats, risk_governor, auth_client, public_client = make_runner(AlwaysCallStrategy())
    series = make_series()
    auth_client.proposal_open_contract = AsyncMock(return_value={"is_sold": 1, "status": "won", "profit": 8.0})

    order_of_calls = []
    runner.on_trade_placed = lambda symbol, direction, stake, payout: order_of_calls.append(("placed", symbol, direction, stake, payout))
    runner.on_settlement = lambda *args: order_of_calls.append(("settled", *args))

    await runner.run_once(series, signal_id="sig-1")

    assert len(order_of_calls) == 2
    assert order_of_calls[0][0] == "placed"   # placed fires before settled
    assert order_of_calls[1][0] == "settled"
    assert order_of_calls[0][1] == "frxEURUSD"
    assert order_of_calls[0][2] == "CALL"
    assert order_of_calls[0][3] == pytest.approx(10.0)   # stake
