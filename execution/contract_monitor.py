"""
Contract settlement monitor. Polls proposal_open_contract until a contract
is sold/expired, so callers (the forward-test loop, eventually live trading)
can learn WIN/LOSS and feed it to RiskGovernor and SessionStats. Polling
rather than a persistent subscription for now — simpler to reason about and
test; a subscribe:1 streaming version is a reasonable future upgrade.
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass

from app.execution.auth_client import DerivAuthenticatedClient


@dataclass
class SettlementResult:
    is_sold: bool
    result: str | None    # "WIN" or "LOSS", None while still open
    profit: float | None
    payout: float | None
    raw: dict


class ContractSettlementTimeout(RuntimeError):
    pass


async def get_settlement_snapshot(client: DerivAuthenticatedClient, contract_id: str) -> SettlementResult:
    data = await client.proposal_open_contract(contract_id)
    is_sold = bool(data.get("is_sold"))
    status = data.get("status")
    result = None
    if is_sold and status in ("won", "lost"):
        result = "WIN" if status == "won" else "LOSS"
    profit = data.get("profit")
    return SettlementResult(
        is_sold=is_sold,
        result=result,
        profit=float(profit) if profit is not None else None,
        payout=float(data["payout"]) if data.get("payout") is not None else None,
        raw=data,
    )


async def poll_until_settled(
    client: DerivAuthenticatedClient,
    contract_id: str,
    poll_interval_s: float = 2.0,
    timeout_s: float = 300.0,
    sleep_fn=asyncio.sleep,
) -> SettlementResult:
    """
    `sleep_fn` is injectable so tests don't have to actually wait — a real
    caller just uses the default asyncio.sleep.
    """
    elapsed = 0.0
    while elapsed <= timeout_s:
        snapshot = await get_settlement_snapshot(client, contract_id)
        if snapshot.is_sold:
            return snapshot
        await sleep_fn(poll_interval_s)
        elapsed += poll_interval_s
    raise ContractSettlementTimeout(f"contract {contract_id} not settled within {timeout_s}s")
