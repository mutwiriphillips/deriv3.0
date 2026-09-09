"""
Thin async wrapper around Deriv's public WebSocket endpoint. Public because
active_symbols and contracts_for require no auth (see docs/data/*). Trading
and account endpoints need the OTP-authenticated demo/real URLs from
rest_client.get_ws_url_for_account and belong in execution/, not here.
"""
from __future__ import annotations

import asyncio
import itertools
import json
from typing import Any

import websockets

from app.config.settings import settings

_req_id_counter = itertools.count(1)


class _DerivWSConnection:
    """
    Shared connect/request machinery. Split out so execution/'s authenticated
    client (which connects to the OTP demo/real URL instead of the public
    one) doesn't duplicate this logic.
    """

    def __init__(self, url: str):
        self._url = url
        self._ws: websockets.WebSocketClientProtocol | None = None

    async def __aenter__(self) -> "_DerivWSConnection":
        self._ws = await websockets.connect(self._url, ping_interval=20, ping_timeout=20)
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        if self._ws is not None:
            await self._ws.close()

    async def _request(self, payload: dict[str, Any], timeout: float = 10.0) -> dict[str, Any]:
        if self._ws is None:
            raise RuntimeError(f"{type(self).__name__} used outside `async with`")
        req_id = next(_req_id_counter)
        payload = {**payload, "req_id": req_id}
        await self._ws.send(json.dumps(payload))

        # The connection is shared, so skip messages that answer a different req_id
        # (relevant once this client is reused for concurrent calls/subscriptions).
        async with asyncio.timeout(timeout):
            while True:
                raw = await self._ws.recv()
                data = json.loads(raw)
                if data.get("error"):
                    raise DerivApiError(data["error"], request=payload)
                if data.get("echo_req", {}).get("req_id") == req_id:
                    return data


class DerivPublicClient(_DerivWSConnection):
    """
    Usage:
        async with DerivPublicClient() as client:
            symbols = await client.active_symbols()
            contracts = await client.contracts_for("frxEURUSD")
    """

    def __init__(self, url: str | None = None):
        super().__init__(url or settings.deriv_ws_public_url)

    async def active_symbols(self, mode: str = "brief", contract_type: list[str] | None = None) -> list[dict]:
        """New-API request shape: only `active_symbols` and optional `contract_type` remain."""
        payload: dict[str, Any] = {"active_symbols": mode}
        if contract_type:
            payload["contract_type"] = contract_type
        data = await self._request(payload)
        return data["active_symbols"]

    async def contracts_for(self, symbol: str) -> dict:
        """New-API request shape: just {"contracts_for": symbol} — no currency/landing_company filters."""
        data = await self._request({"contracts_for": symbol})
        return data["contracts_for"]

    async def proposal(
        self,
        contract_type: str,
        underlying_symbol: str,
        currency: str,
        amount: float,
        duration: int,
        duration_unit: str = "s",
        basis: str = "stake",
        barrier: str | None = None,
    ) -> dict:
        """
        Confirmed from the current docs: `proposal`, `contract_type`, `currency`,
        `underlying_symbol` (renamed from `symbol`), `amount` are all real
        New-API fields, and `loginid`/`barrier_range`/`product_type`/`date_start`/
        `trade_risk_profile`/`trading_period_start` are confirmed REMOVED.
        `duration`, `duration_unit`, and `basis` were not shown in the
        (abbreviated) migration-guide example but are near-certainly still
        required to specify a contract at all — this is a reasonable
        inference, not a confirmed-from-docs field name, and should be
        verified against a real response before being trusted further.
        Returns the raw `proposal` object (`id` is the only guaranteed field
        in the New API — everything else, including `ask_price`/`payout`,
        may be absent and must be checked before use).
        """
        payload: dict[str, Any] = {
            "proposal": 1,
            "contract_type": contract_type,
            "underlying_symbol": underlying_symbol,
            "currency": currency,
            "amount": amount,
            "basis": basis,
            "duration": duration,
            "duration_unit": duration_unit,
            "subscribe": 0,
        }
        if barrier is not None:
            payload["barrier"] = barrier
        data = await self._request(payload)
        return data["proposal"]

    async def ticks_history(
        self,
        symbol: str,
        style: str = "ticks",           # "ticks" or "candles"
        granularity: int = 60,          # only used when style="candles"; any positive int per current docs
        count: int | None = None,
        start: str | None = None,
        end: str = "latest",
        adjust_start_time: int = 0,
    ) -> dict:
        """
        One-shot historical fetch. Returns the raw `history` dict
        ({"prices": [...], "times": [...]}) for style="ticks", or the raw
        `candles` list for style="candles" — caller decides what to do with it.

        IMPORTANT — confirmed against a live error, not just docs: sending
        `subscribe`/`adjust_start_time` on a style="candles" request gets
        rejected with InputValidationFailed naming exactly those two fields.
        The docs' example request showing subscribe:0/adjust_start_time:0
        was for style="ticks" only; candle requests apparently don't accept
        those fields at all (evidence: the reported error field names,
        the fact this is the only ticks_history call path ever exercised in
        production, and that the API's own migration example never showed
        a style="candles" one-time-request variant including them). Treated
        here as confirmed rather than a guess, but still worth a second look
        if a future response ever contradicts it.
        """
        payload: dict[str, Any] = {
            "ticks_history": symbol,
            "style": style,
            "end": end,
        }
        if style == "ticks":
            payload["subscribe"] = 0
            payload["adjust_start_time"] = adjust_start_time
        if style == "candles":
            payload["granularity"] = granularity
        if count is not None:
            payload["count"] = count
        if start is not None:
            payload["start"] = start
        data = await self._request(payload)
        return data.get("candles") if style == "candles" else data.get("history", {})

    async def stream_ticks(self, symbol: str):
        """
        Async generator yielding raw tick dicts ({"epoch", "quote", "symbol", ...})
        as they arrive. Caller is responsible for breaking out of the loop
        (e.g. via `break`) — this does not auto-unsubscribe (see subscription/forget
        for that, not yet implemented here).
        """
        if self._ws is None:
            raise RuntimeError("DerivPublicClient used outside `async with`")
        req_id = next(_req_id_counter)
        payload = {"ticks": symbol, "subscribe": 1, "req_id": req_id}
        await self._ws.send(json.dumps(payload))
        while True:
            raw = await self._ws.recv()
            data = json.loads(raw)
            if data.get("error"):
                raise DerivApiError(data["error"], request=payload)
            if data.get("msg_type") == "tick":
                yield data["tick"]


class DerivApiError(RuntimeError):
    def __init__(self, error: dict[str, Any], request: dict[str, Any] | None = None):
        request_str = f" | request sent: {json.dumps(request)}" if request is not None else ""
        super().__init__(f"Deriv API error [{error.get('code')}]: {error.get('message')}{request_str}")
        self.code = error.get("code")
        self.raw = error
        self.request = request
