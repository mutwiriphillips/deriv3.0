from unittest.mock import AsyncMock

import pytest

from app.execution.contract_monitor import (
    ContractSettlementTimeout,
    get_settlement_snapshot,
    poll_until_settled,
)


def make_client(responses):
    client = AsyncMock()
    client.proposal_open_contract = AsyncMock(side_effect=responses)
    return client


@pytest.mark.asyncio
async def test_get_settlement_snapshot_open_contract():
    client = make_client([{"is_sold": 0, "status": "open", "contract_id": "1"}])
    snapshot = await get_settlement_snapshot(client, "1")
    assert snapshot.is_sold is False
    assert snapshot.result is None


@pytest.mark.asyncio
async def test_get_settlement_snapshot_won():
    client = make_client([{"is_sold": 1, "status": "won", "profit": 8.0, "payout": 18.0}])
    snapshot = await get_settlement_snapshot(client, "1")
    assert snapshot.is_sold is True
    assert snapshot.result == "WIN"
    assert snapshot.profit == pytest.approx(8.0)


@pytest.mark.asyncio
async def test_get_settlement_snapshot_lost():
    client = make_client([{"is_sold": 1, "status": "lost", "profit": -10.0}])
    snapshot = await get_settlement_snapshot(client, "1")
    assert snapshot.result == "LOSS"
    assert snapshot.profit == pytest.approx(-10.0)


@pytest.mark.asyncio
async def test_poll_until_settled_returns_once_sold():
    client = make_client([
        {"is_sold": 0, "status": "open"},
        {"is_sold": 0, "status": "open"},
        {"is_sold": 1, "status": "won", "profit": 8.0},
    ])
    sleeps = []

    async def fake_sleep(s):
        sleeps.append(s)

    result = await poll_until_settled(client, "1", poll_interval_s=1.0, timeout_s=10.0, sleep_fn=fake_sleep)
    assert result.result == "WIN"
    assert len(sleeps) == 2  # slept between the two "open" polls, not after the final settled one
    assert client.proposal_open_contract.call_count == 3


@pytest.mark.asyncio
async def test_poll_until_settled_times_out():
    client = make_client([{"is_sold": 0, "status": "open"}] * 100)

    async def fake_sleep(s):
        return None

    with pytest.raises(ContractSettlementTimeout):
        await poll_until_settled(client, "1", poll_interval_s=1.0, timeout_s=3.0, sleep_fn=fake_sleep)
