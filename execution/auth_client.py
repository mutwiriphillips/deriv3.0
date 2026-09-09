"""
Authenticated Deriv WS client — connects to the OTP-embedded demo/real URL
from rest_client.get_ws_url_for_account() rather than the public endpoint.
No separate `authorize` call is needed: per the current docs, the OTP is
already baked into the connection URL itself (this is a real architectural
difference from the Legacy pattern, where you'd send {"authorize": token}
as the first message after connecting).
"""
from __future__ import annotations

from app.markets.ws_client import _DerivWSConnection


class DerivAuthenticatedClient(_DerivWSConnection):
    """
    Usage:
        ws_url = await get_ws_url_for_account(account_id)   # from rest_client.py
        async with DerivAuthenticatedClient(ws_url) as client:
            result = await client.buy(proposal_id, price)
    """

    async def buy(self, proposal_id: str, price: float) -> dict:
        """
        Confirmed against the current docs exactly: {"buy": proposal_id,
        "price": price} is the full New-API request (loginid is confirmed
        removed). Response `buy` object is now a required field — safe to
        access without a None-check, per the docs.
        """
        data = await self._request({"buy": proposal_id, "price": price, "subscribe": 0})
        return data["buy"]

    async def proposal_open_contract(self, contract_id: str) -> dict:
        """
        One-shot open-contract lookup (subscribe=0). `contract_id`,
        `contract_type`, and `currency` are confirmed-guaranteed fields in
        the New API; `is_sold`, `status` ("open"/"won"/"lost"), and `profit`
        are documented and widely relied upon in practice but weren't
        explicitly re-confirmed as unchanged in the migration comparison —
        worth a sanity check against a real response the first time this runs.
        """
        data = await self._request({"proposal_open_contract": 1, "contract_id": contract_id, "subscribe": 0})
        return data["proposal_open_contract"]
