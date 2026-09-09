from unittest.mock import AsyncMock, patch

import pytest

from app.execution.order_manager import OrderManager, OrderManagerConfig
from app.risk.exposure import ExposureManager
from app.risk.governor import RiskGovernor
from app.risk.overtrading import CooldownTracker, DuplicateSignalGuard, RateLimiter
from app.strategies.base import Direction, TradeCandidate


def make_manager(**overrides):
    config = OrderManagerConfig(
        currency="USD", risk_per_trade=0.01, max_stake=100.0,
        min_probability_edge=0.0, min_expected_value=0.0, max_latency_ms=5000.0,
    )
    for k, v in overrides.items():
        setattr(config, k, v)
    return OrderManager(
        config=config,
        risk_governor=RiskGovernor(starting_balance=1000, max_daily_loss=1000, max_drawdown=0.9, max_consecutive_losses=100),
        exposure_manager=ExposureManager(max_exposure_per_currency=2.0),
        duplicate_guard=DuplicateSignalGuard(),
        cooldown_tracker=CooldownTracker(cooldown_seconds=60),
        rate_limiter=RateLimiter(max_trades_per_hour=10, max_trades_per_day=50),
    )


def make_candidate(direction=Direction.CALL, probability=0.65):
    return TradeCandidate(direction=direction, model_probability=probability, confidence=0.5, signal_score=0.15, reasons=["test"])


def make_clients(proposal_response=None, buy_response=None):
    public_client = AsyncMock()
    public_client.proposal = AsyncMock(return_value=proposal_response or {"id": "prop-1", "ask_price": "10.00", "payout": "18.00"})
    auth_client = AsyncMock()
    auth_client.buy = AsyncMock(return_value=buy_response or {"contract_id": 12345, "buy_price": 10.0, "payout": 18.0})
    return public_client, auth_client


@pytest.mark.asyncio
async def test_no_trade_candidate_short_circuits_without_api_calls():
    manager = make_manager()
    public_client, auth_client = make_clients()
    candidate = TradeCandidate(direction=Direction.NO_TRADE, model_probability=0.5, confidence=0.0, signal_score=0.0, reasons=["x"])

    outcome = await manager.attempt_trade("sig-1", "frxEURUSD", candidate, 60, "s", 1000, public_client, auth_client, now=0.0)

    assert outcome.status == "NO_TRADE"
    assert "SIGNAL_INVALID" in outcome.reason
    public_client.proposal.assert_not_called()
    auth_client.buy.assert_not_called()


@pytest.mark.asyncio
async def test_duplicate_signal_is_never_traded_twice():
    manager = make_manager()
    public_client, auth_client = make_clients()
    candidate = make_candidate()

    first = await manager.attempt_trade("sig-1", "frxEURUSD", candidate, 60, "s", 1000, public_client, auth_client, now=0.0)
    assert first.status == "TRADE"
    assert auth_client.buy.call_count == 1

    second = await manager.attempt_trade("sig-1", "frxEURUSD", candidate, 60, "s", 1000, public_client, auth_client, now=1000.0)
    assert second.status == "ALREADY_HANDLED"
    assert auth_client.buy.call_count == 1  # not called again


@pytest.mark.asyncio
async def test_risk_governor_emergency_stop_blocks_trade():
    manager = make_manager()
    manager.risk_governor.record_trade_result(-999999)  # force emergency stop via massive drawdown
    public_client, auth_client = make_clients()
    candidate = make_candidate()

    outcome = await manager.attempt_trade("sig-1", "frxEURUSD", candidate, 60, "s", 1000, public_client, auth_client, now=0.0)
    assert outcome.status == "NO_TRADE"
    assert outcome.reason == "NO_TRADE_RISK_LIMIT"
    public_client.proposal.assert_not_called()


@pytest.mark.asyncio
async def test_cooldown_blocks_rapid_repeat_trades_on_same_symbol():
    manager = make_manager()
    public_client, auth_client = make_clients()

    c1 = make_candidate()
    r1 = await manager.attempt_trade("sig-1", "frxEURUSD", c1, 60, "s", 1000, public_client, auth_client, now=0.0)
    assert r1.status == "TRADE"

    c2 = make_candidate()
    r2 = await manager.attempt_trade("sig-2", "frxEURUSD", c2, 60, "s", 1000, public_client, auth_client, now=10.0)
    assert r2.status == "NO_TRADE"
    assert r2.reason == "NO_TRADE_COOLDOWN"


@pytest.mark.asyncio
async def test_rate_limiter_blocks_beyond_hourly_cap():
    manager = make_manager()
    manager.rate_limiter = RateLimiter(max_trades_per_hour=1, max_trades_per_day=50)
    public_client, auth_client = make_clients()

    r1 = await manager.attempt_trade("sig-1", "frxEURUSD", make_candidate(), 60, "s", 1000, public_client, auth_client, now=0.0)
    assert r1.status == "TRADE"

    r2 = await manager.attempt_trade("sig-2", "frxGBPUSD", make_candidate(), 60, "s", 1000, public_client, auth_client, now=5.0)
    assert r2.status == "NO_TRADE"
    assert r2.reason == "NO_TRADE_RATE_LIMIT_EXCEEDED"


@pytest.mark.asyncio
async def test_exposure_manager_blocks_correlated_trade():
    manager = make_manager()
    manager.exposure_manager = ExposureManager(max_exposure_per_currency=1.5)
    public_client, auth_client = make_clients()

    r1 = await manager.attempt_trade("sig-1", "frxEURUSD", make_candidate(Direction.CALL), 60, "s", 1000, public_client, auth_client, now=0.0)
    assert r1.status == "TRADE"

    r2 = await manager.attempt_trade("sig-2", "frxGBPUSD", make_candidate(Direction.CALL), 60, "s", 1000, public_client, auth_client, now=100.0)
    assert r2.status == "NO_TRADE"
    assert r2.reason == "NO_TRADE_CORRELATED_EXPOSURE"


@pytest.mark.asyncio
async def test_proposal_missing_pricing_fields_is_rejected():
    manager = make_manager()
    public_client, auth_client = make_clients(proposal_response={"id": "prop-1"})  # only guaranteed field
    outcome = await manager.attempt_trade("sig-1", "frxEURUSD", make_candidate(), 60, "s", 1000, public_client, auth_client, now=0.0)
    assert outcome.status == "NO_TRADE"
    assert outcome.reason == "NO_TRADE_SIGNAL_INVALID"
    auth_client.buy.assert_not_called()


@pytest.mark.asyncio
async def test_low_edge_blocks_trade():
    manager = make_manager(min_probability_edge=0.5)  # impossibly high requirement
    public_client, auth_client = make_clients()
    outcome = await manager.attempt_trade("sig-1", "frxEURUSD", make_candidate(probability=0.55), 60, "s", 1000, public_client, auth_client, now=0.0)
    assert outcome.status == "NO_TRADE"
    assert outcome.reason == "NO_TRADE_LOW_EDGE"
    auth_client.buy.assert_not_called()


@pytest.mark.asyncio
async def test_negative_ev_blocks_trade():
    manager = make_manager(min_expected_value=1000.0)  # impossibly high requirement
    public_client, auth_client = make_clients()
    outcome = await manager.attempt_trade("sig-1", "frxEURUSD", make_candidate(), 60, "s", 1000, public_client, auth_client, now=0.0)
    assert outcome.status == "NO_TRADE"
    assert outcome.reason == "NO_TRADE_NEGATIVE_EV"
    auth_client.buy.assert_not_called()


@pytest.mark.asyncio
async def test_high_latency_blocks_trade():
    manager = make_manager(max_latency_ms=1.0)  # 1ms is unrealistically strict
    public_client, auth_client = make_clients()

    with patch("app.execution.order_manager.time.monotonic", side_effect=[0.0, 5.0, 5.001]):
        outcome = await manager.attempt_trade("sig-1", "frxEURUSD", make_candidate(), 60, "s", 1000, public_client, auth_client, now=0.0)
    assert outcome.status == "NO_TRADE"
    assert outcome.reason == "NO_TRADE_HIGH_LATENCY"
    auth_client.buy.assert_not_called()


@pytest.mark.asyncio
async def test_successful_trade_returns_full_outcome_and_updates_all_trackers():
    manager = make_manager()
    public_client, auth_client = make_clients()

    outcome = await manager.attempt_trade("sig-1", "frxEURUSD", make_candidate(), 60, "s", 1000, public_client, auth_client, now=0.0)

    assert outcome.status == "TRADE"
    assert outcome.contract_id == "12345"
    assert outcome.buy_price == pytest.approx(10.0)
    assert outcome.payout == pytest.approx(18.0)
    assert outcome.total_execution_latency_ms is not None and outcome.total_execution_latency_ms >= 0

    # Every tracker touched by a successful trade should now reflect it.
    assert manager.duplicate_guard.has_already_traded("sig-1") is True
    assert manager.cooldown_tracker.is_satisfied("frxEURUSD", now=1.0) is False
    assert manager.exposure_manager.current_exposure() != {}


@pytest.mark.asyncio
async def test_proposal_called_with_direction_as_contract_type():
    manager = make_manager()
    public_client, auth_client = make_clients()
    await manager.attempt_trade("sig-1", "frxEURUSD", make_candidate(Direction.PUT), 60, "s", 1000, public_client, auth_client, now=0.0)
    call_kwargs = public_client.proposal.call_args.kwargs
    assert call_kwargs["contract_type"] == "PUT"
    assert call_kwargs["underlying_symbol"] == "frxEURUSD"
