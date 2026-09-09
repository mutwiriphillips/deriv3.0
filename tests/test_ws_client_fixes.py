import json

import pytest

from app.markets.ws_client import DerivApiError, DerivPublicClient


class FakeWebSocket:
    """
    Minimal fake matching what _DerivWSConnection._request needs: send()
    records the payload, recv() replays a canned response with echo_req
    dynamically filled in to match whatever req_id was actually sent --
    the real API always echoes the caller's req_id, and _req_id_counter is
    a shared module-level counter across the whole test session, so a
    hardcoded req_id in the canned response would only work by coincidence.
    """

    def __init__(self, responses):
        self.sent_payloads = []
        self._responses = list(responses)

    async def send(self, raw):
        self.sent_payloads.append(json.loads(raw))

    async def recv(self):
        response = self._responses.pop(0)
        if "echo_req" in response and self.sent_payloads:
            response = {**response, "echo_req": {"req_id": self.sent_payloads[-1]["req_id"]}}
        return json.dumps(response)

    async def close(self):
        pass


@pytest.mark.asyncio
async def test_ticks_history_candles_style_omits_subscribe_and_adjust_start_time():
    """
    This is the exact bug reported from a live Render deploy: the API
    rejected these two fields on a style="candles" request. Confirms the
    fix -- they must not be sent at all for candles.
    """
    client = DerivPublicClient()
    fake_ws = FakeWebSocket([{"echo_req": {"req_id": 1}, "candles": [{"epoch": 100, "open": 1, "high": 1, "low": 1, "close": 1}]}])
    client._ws = fake_ws

    await client.ticks_history("frxEURUSD", style="candles", granularity=60, count=10)

    sent = fake_ws.sent_payloads[0]
    assert "subscribe" not in sent
    assert "adjust_start_time" not in sent
    assert sent["style"] == "candles"
    assert sent["granularity"] == 60


@pytest.mark.asyncio
async def test_ticks_history_ticks_style_still_includes_subscribe_and_adjust_start_time():
    """The docs' own example DOES show these fields for style="ticks" -- only candles should omit them."""
    client = DerivPublicClient()
    fake_ws = FakeWebSocket([{"echo_req": {"req_id": 1}, "history": {"prices": [1.0], "times": [100]}}])
    client._ws = fake_ws

    await client.ticks_history("frxEURUSD", style="ticks", count=10)

    sent = fake_ws.sent_payloads[0]
    assert sent["subscribe"] == 0
    assert "adjust_start_time" in sent


@pytest.mark.asyncio
async def test_deriv_api_error_includes_the_request_that_caused_it():
    """
    Closes the visibility gap that slowed down diagnosing the live bug:
    the exception message itself must show the exact outgoing payload,
    not just the server's complaint about which fields were bad.
    """
    client = DerivPublicClient()
    fake_ws = FakeWebSocket([
        {"echo_req": {"req_id": 1}, "error": {"code": "InputValidationFailed", "message": "Input validation failed: subscribe, adjust_start_time"}},
    ])
    client._ws = fake_ws

    with pytest.raises(DerivApiError) as exc_info:
        await client.ticks_history("frxEURUSD", style="candles")

    message = str(exc_info.value)
    assert "InputValidationFailed" in message
    assert "request sent" in message
    assert "candles" in message   # the actual payload is in the message, not just the error code
    assert exc_info.value.request is not None
    assert exc_info.value.request["style"] == "candles"
