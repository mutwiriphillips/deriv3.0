import sqlite3
from unittest.mock import AsyncMock, patch

import pytest

from app.markets.discovery import (
    RegistryEntry,
    classify_market,
    discover_markets,
    upsert_registry,
)
from app.markets.context import MarketType


def test_classify_market_known_values():
    assert classify_market("forex") == MarketType.FOREX
    assert classify_market("synthetic_index") == MarketType.SYNTHETIC
    assert classify_market("commodities") == MarketType.COMMODITY


def test_classify_market_unknown_falls_back_to_other():
    assert classify_market("some_future_market_type") == MarketType.OTHER
    assert classify_market(None) == MarketType.OTHER


# Response shapes below match the current (New API) docs as of Sep 2026 —
# developers.deriv.com/comparison/active-symbols/ and .../contracts-for/
MOCK_ACTIVE_SYMBOLS = [
    {
        "underlying_symbol": "frxEURUSD",
        "underlying_symbol_name": "EUR/USD",
        "market": "forex",
        "submarket": "major_pairs",
        "exchange_is_open": 1,
        "is_trading_suspended": 0,
    },
    {
        "underlying_symbol": "1HZ100V",
        "underlying_symbol_name": "Volatility 100 (1s) Index",
        "market": "synthetic_index",
        "submarket": "random_index",
        "exchange_is_open": 1,
        "is_trading_suspended": 0,
    },
]

MOCK_CONTRACTS_FOR = {
    "available": [
        {"contract_type": "CALL", "market": "forex", "submarket": "major_pairs"},
        {"contract_type": "PUT", "market": "forex", "submarket": "major_pairs"},
    ],
    "hit_count": 2,
}


@pytest.mark.asyncio
async def test_discover_markets_builds_registry_entries():
    with patch("app.markets.discovery.DerivPublicClient") as MockClient:
        instance = MockClient.return_value
        instance.__aenter__ = AsyncMock(return_value=instance)
        instance.__aexit__ = AsyncMock(return_value=None)
        instance.active_symbols = AsyncMock(return_value=MOCK_ACTIVE_SYMBOLS)
        instance.contracts_for = AsyncMock(return_value=MOCK_CONTRACTS_FOR)

        entries = await discover_markets()

    assert len(entries) == 2
    eurusd = next(e for e in entries if e.symbol == "frxEURUSD")
    assert eurusd.market == MarketType.FOREX
    assert eurusd.available_contracts == ["CALL", "PUT"]
    assert eurusd.status == "ACTIVE"

    vol100 = next(e for e in entries if e.symbol == "1HZ100V")
    assert vol100.market == MarketType.SYNTHETIC


def test_upsert_registry_writes_and_updates(tmp_path):
    db_path = str(tmp_path / "test.db")
    conn = sqlite3.connect(db_path)
    with open("app/storage/schema.sql") as f:
        conn.executescript(f.read())
    conn.close()

    entry = RegistryEntry(
        symbol="frxEURUSD",
        display_name="EUR/USD",
        market=MarketType.FOREX,
        submarket="major_pairs",
        exchange="FOREX",
        available_contracts=["CALL", "PUT"],
        last_updated="2026-09-09T00:00:00+00:00",
    )
    upsert_registry(db_path, [entry])

    conn = sqlite3.connect(db_path)
    row = conn.execute("SELECT symbol, market, status FROM markets WHERE symbol=?", ("frxEURUSD",)).fetchone()
    assert row == ("frxEURUSD", "FOREX", "ACTIVE")

    # Re-running with a status change should UPDATE, not duplicate.
    entry.status = "SUSPENDED"
    upsert_registry(db_path, [entry])
    rows = conn.execute("SELECT status FROM markets WHERE symbol=?", ("frxEURUSD",)).fetchall()
    assert len(rows) == 1
    assert rows[0][0] == "SUSPENDED"
    conn.close()
