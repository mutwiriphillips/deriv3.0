"""
Deriv REST client — account setup and OTP minting only.

Per the current docs (developers.deriv.com/docs/intro/api-overview):
  1. REST: get_accounts (which demo/real accounts exist under this token)
  2. REST: mint an OTP for the account you want to trade on
  3. WebSocket: connect using the OTP-embedded URL returned by step 2

This module never touches trading logic — it only turns a PAT into a
ready-to-use WebSocket URL.
"""
from __future__ import annotations

import httpx

from app.config.settings import settings


class DerivRestError(RuntimeError):
    def __init__(self, status: int, code: str, message: str):
        super().__init__(f"Deriv REST error {status} [{code}]: {message}")
        self.status = status
        self.code = code


def _headers() -> dict[str, str]:
    if not settings.deriv_api_token:
        raise RuntimeError("DERIV_API_TOKEN is not set — see .env.example")
    headers = {"Authorization": f"Bearer {settings.deriv_api_token}"}
    # Required for PAT auth; omitting it returns 401 per the docs.
    if settings.deriv_app_id:
        headers["Deriv-App-ID"] = settings.deriv_app_id
    return headers


async def get_accounts() -> list[dict]:
    """
    Returns all Options trading accounts (demo + real) visible to this token.
    Confirmed against Deriv's published OpenAPI spec
    (developers.deriv.com/data/production-openapi.json): the response is
    {"data": [OptionsAccount, ...], "meta": {...}} — NOT a bare list. Every
    prior version of this function returned the raw envelope, which made
    calling code iterate over the envelope's own keys (["data", "meta"])
    instead of actual account objects.
    """
    url = f"{settings.deriv_rest_base_url}/trading/v1/options/accounts"
    async with httpx.AsyncClient(timeout=10.0) as client:
        resp = await client.get(url, headers=_headers())
    _raise_for_error(resp)
    return resp.json()["data"]


async def get_ws_url_for_account(account_id: str) -> str:
    """
    Mints an OTP for `account_id` and returns the ready-to-use WebSocket URL
    (already includes ?otp=... per the docs — do not append your own token).
    Confirmed against the published OpenAPI spec: response shape is exactly
    {"data": {"url": "..."}, "meta": {...}}, with "url" a required field —
    safe to index directly rather than guessing among several possible key
    names.
    """
    url = f"{settings.deriv_rest_base_url}/trading/v1/options/accounts/{account_id}/otp"
    async with httpx.AsyncClient(timeout=10.0) as client:
        resp = await client.post(url, headers=_headers())
    _raise_for_error(resp)
    return resp.json()["data"]["url"]


def _raise_for_error(resp: "httpx.Response") -> None:
    if resp.status_code >= 400:
        try:
            err = resp.json()["errors"][0]
            raise DerivRestError(resp.status_code, err.get("code", "?"), err.get("message", "?"))
        except (KeyError, IndexError, ValueError):
            raise DerivRestError(resp.status_code, "?", resp.text)
