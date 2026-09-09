import numpy as np
import pytest
from unittest.mock import AsyncMock

from app.analytics.trade_advisor import get_trade_advice
from app.features.types import CandleSeries
from app.strategies.base import Direction, TradeCandidate


def make_series(n=200, seed=1):
    rng = np.random.default_rng(seed)
    closes = list(1.0 + np.cumsum(rng.normal(0.0005, 0.002, n)))
    timestamps = [1_700_000_000 + i * 60 for i in range(n)]
    highs = [c + 0.001 for c in closes]
    lows = [c - 0.001 for c in closes]
    return CandleSeries(symbol="frxEURUSD", duration_s=60, timestamps=timestamps, opens=closes, highs=highs, lows=lows, closes=closes)


class AlwaysCallStrategy:
    strategy_id = "test_always_call"

    def evaluate(self, ctx):
        return TradeCandidate(direction=Direction.CALL, model_probability=0.65, confidence=0.6, signal_score=0.15, reasons=["strong trend"])


class AlwaysNoTradeStrategy:
    strategy_id = "test_never"

    def evaluate(self, ctx):
        return TradeCandidate(direction=Direction.NO_TRADE, model_probability=0.5, confidence=0.0, signal_score=0.0, reasons=["nothing here"])


def make_mock_client(proposal_side_effect=None, proposal_return=None):
    client = AsyncMock()
    if proposal_side_effect is not None:
        client.proposal = AsyncMock(side_effect=proposal_side_effect)
    else:
        client.proposal = AsyncMock(return_value=proposal_return or {"id": "p1", "ask_price": "10.00", "payout": "18.00"})
    return client


@pytest.mark.asyncio
async def test_no_trade_when_strategy_has_no_signal():
    client = make_mock_client()
    advice = await get_trade_advice("frxEURUSD", AlwaysNoTradeStrategy(), make_series(), client, candidate_durations_s=[60])
    assert advice.action == "NO_TRADE"
    assert advice.direction is None
    client.proposal.assert_not_called()   # no point checking proposals with no directional signal


@pytest.mark.asyncio
async def test_recommends_trade_when_a_duration_clears_the_bar():
    client = make_mock_client(proposal_return={"id": "p1", "ask_price": "10.00", "payout": "18.00"})
    advice = await get_trade_advice(
        "frxEURUSD", AlwaysCallStrategy(), make_series(), client,
        candidate_durations_s=[60], min_probability_edge=0.0, min_expected_value=0.0,
    )
    assert advice.action == "TRADE"
    assert advice.direction == "CALL"
    assert advice.recommended_duration_s == 60
    assert advice.probability_edge is not None and advice.probability_edge > 0


@pytest.mark.asyncio
async def test_picks_the_best_ev_duration_among_several():
    responses = {
        30: {"id": "p1", "ask_price": "10.00", "payout": "12.00"},   # low payout ratio (0.2)
        60: {"id": "p2", "ask_price": "10.00", "payout": "18.00"},   # higher payout ratio (0.8) -> better EV
        120: {"id": "p3", "ask_price": "10.00", "payout": "14.00"},  # medium (0.4)
    }

    async def side_effect(contract_type, underlying_symbol, currency, amount, duration, duration_unit="s", basis="stake", barrier=None):
        return responses[duration]

    client = make_mock_client(proposal_side_effect=side_effect)
    advice = await get_trade_advice(
        "frxEURUSD", AlwaysCallStrategy(), make_series(), client,
        candidate_durations_s=[30, 60, 120], min_probability_edge=0.0, min_expected_value=0.0,
    )
    assert advice.action == "TRADE"
    assert advice.recommended_duration_s == 60   # the 0.8 payout ratio wins
    assert len(advice.duration_candidates) == 3


@pytest.mark.asyncio
async def test_no_trade_when_no_duration_clears_the_edge_bar():
    # Low payout ratio (0.1) means break-even is very high (~90.9%); a 65% model probability can't clear it.
    client = make_mock_client(proposal_return={"id": "p1", "ask_price": "10.00", "payout": "11.00"})
    advice = await get_trade_advice(
        "frxEURUSD", AlwaysCallStrategy(), make_series(), client,
        candidate_durations_s=[60], min_probability_edge=0.10, min_expected_value=0.0,
    )
    assert advice.action == "NO_TRADE"
    assert advice.direction == "CALL"   # direction was identified, just nothing qualified
    assert advice.duration_candidates[0].qualifies is False


@pytest.mark.asyncio
async def test_handles_a_failed_proposal_request_gracefully():
    async def side_effect(**kwargs):
        raise RuntimeError("network error")

    client = make_mock_client(proposal_side_effect=side_effect)
    advice = await get_trade_advice(
        "frxEURUSD", AlwaysCallStrategy(), make_series(), client, candidate_durations_s=[60],
    )
    assert advice.action == "NO_TRADE"
    assert advice.duration_candidates[0].reason is not None
    assert "failed" in advice.duration_candidates[0].reason


@pytest.mark.asyncio
async def test_handles_proposal_missing_pricing_fields():
    client = make_mock_client(proposal_return={"id": "p1"})   # only guaranteed field
    advice = await get_trade_advice(
        "frxEURUSD", AlwaysCallStrategy(), make_series(), client, candidate_durations_s=[60],
    )
    assert advice.action == "NO_TRADE"
    assert "missing pricing" in advice.duration_candidates[0].reason


@pytest.mark.asyncio
async def test_suggested_stake_computed_when_balance_given():
    client = make_mock_client(proposal_return={"id": "p1", "ask_price": "10.00", "payout": "18.00"})
    advice = await get_trade_advice(
        "frxEURUSD", AlwaysCallStrategy(), make_series(), client, candidate_durations_s=[60],
        account_balance=1000.0, risk_per_trade=0.02, max_stake=50.0,
    )
    assert advice.suggested_stake == pytest.approx(20.0)   # 2% of 1000


@pytest.mark.asyncio
async def test_no_suggested_stake_when_balance_not_given():
    client = make_mock_client(proposal_return={"id": "p1", "ask_price": "10.00", "payout": "18.00"})
    advice = await get_trade_advice(
        "frxEURUSD", AlwaysCallStrategy(), make_series(), client, candidate_durations_s=[60],
    )
    assert advice.suggested_stake is None


@pytest.mark.asyncio
async def test_regime_and_reasoning_present_on_trade_advice():
    client = make_mock_client(proposal_return={"id": "p1", "ask_price": "10.00", "payout": "18.00"})
    advice = await get_trade_advice("frxEURUSD", AlwaysCallStrategy(), make_series(), client, candidate_durations_s=[60])
    assert advice.regime is not None
    assert len(advice.reasoning) > 0
