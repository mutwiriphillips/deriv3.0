from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.markets.rest_client import DerivRestError, get_accounts, get_ws_url_for_account


def make_mock_response(status_code, json_data, text=""):
    resp = MagicMock()
    resp.status_code = status_code
    resp.json.return_value = json_data
    resp.text = text
    return resp


def patch_settings_credentials():
    return patch.multiple(
        "app.markets.rest_client.settings",
        deriv_api_token="test-token",
        deriv_app_id="test-app-id",
        deriv_rest_base_url="https://api.derivws.com",
    )


@pytest.mark.asyncio
async def test_get_accounts_unwraps_the_data_envelope():
    """
    Confirmed against Deriv's OpenAPI spec: response is {"data": [...], "meta": {...}}.
    This is exactly the bug reported from production -- accounts must come
    from response["data"], not the raw response itself.
    """
    envelope = {
        "data": [
            {"account_id": "DOT001", "account_type": "demo", "currency": "USD"},
            {"account_id": "DOT002", "account_type": "real", "currency": "USD"},
        ],
        "meta": {"endpoint": "/accounts", "method": "GET", "timing": 10},
    }
    mock_client = AsyncMock()
    mock_client.get = AsyncMock(return_value=make_mock_response(200, envelope))
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=None)

    with patch_settings_credentials(), patch("app.markets.rest_client.httpx.AsyncClient", return_value=mock_client):
        accounts = await get_accounts()

    assert isinstance(accounts, list)
    assert len(accounts) == 2
    assert accounts[0]["account_id"] == "DOT001"  # a real dict, not a string key from the envelope


@pytest.mark.asyncio
async def test_get_accounts_raises_deriv_rest_error_on_4xx():
    error_body = {"errors": [{"status": 401, "code": "Unauthorized", "message": "Invalid or missing authentication"}]}
    mock_client = AsyncMock()
    mock_client.get = AsyncMock(return_value=make_mock_response(401, error_body))
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=None)

    with patch_settings_credentials(), patch("app.markets.rest_client.httpx.AsyncClient", return_value=mock_client):
        with pytest.raises(DerivRestError) as exc_info:
            await get_accounts()

    assert exc_info.value.status == 401
    assert exc_info.value.code == "Unauthorized"


@pytest.mark.asyncio
async def test_get_ws_url_for_account_reads_data_url_directly():
    """Confirmed against the OpenAPI spec: {"data": {"url": "..."}, "meta": {...}}."""
    envelope = {"data": {"url": "wss://api.derivws.com/trading/v1/options/ws/demo?otp=abc123"}, "meta": {}}
    mock_client = AsyncMock()
    mock_client.post = AsyncMock(return_value=make_mock_response(200, envelope))
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=None)

    with patch_settings_credentials(), patch("app.markets.rest_client.httpx.AsyncClient", return_value=mock_client):
        ws_url = await get_ws_url_for_account("DOT001")

    assert ws_url == "wss://api.derivws.com/trading/v1/options/ws/demo?otp=abc123"


@pytest.mark.asyncio
async def test_get_ws_url_for_account_raises_on_error():
    error_body = {"errors": [{"status": 400, "code": "InvalidAccount", "message": "Account not found"}]}
    mock_client = AsyncMock()
    mock_client.post = AsyncMock(return_value=make_mock_response(400, error_body))
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=None)

    with patch_settings_credentials(), patch("app.markets.rest_client.httpx.AsyncClient", return_value=mock_client):
        with pytest.raises(DerivRestError):
            await get_ws_url_for_account("bad-id")
