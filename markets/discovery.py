"""
Market discovery engine (spec Part 3). Builds and refreshes the MarketRegistry
purely from what the Deriv API reports right now — no hardcoded symbol list.
"""
from __future__ import annotations

import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, timezone

from app.markets.context import MarketType
from app.markets.ws_client import DerivPublicClient


@dataclass
class RegistryEntry:
    symbol: str                    # underlying_symbol in the New API
    display_name: str              # underlying_symbol_name in the New API (may be absent -> falls back to symbol)
    market: MarketType
    submarket: str | None
    exchange: str | None
    available_contracts: list[str] = field(default_factory=list)
    available_durations: list[str] = field(default_factory=list)
    trading_hours: dict | None = None
    tick_frequency: float | None = None
    status: str = "ACTIVE"
    last_updated: str = ""


def classify_market(raw_market: str | None) -> MarketType:
    """
    Deriv's `market` field values seen in the docs: "forex", "synthetic_index",
    "indices", "commodities", "cryptocurrency", "basket_index", etc.
    Anything not recognized falls into OTHER rather than raising, per the
    spec's "do not assume every market behaves the same way."
    """
    if not raw_market:
        return MarketType.OTHER
    key = raw_market.lower()
    if key == "forex":
        return MarketType.FOREX
    if key in ("synthetic_index", "synthetic", "basket_index"):
        return MarketType.SYNTHETIC
    if key == "commodities":
        return MarketType.COMMODITY
    return MarketType.OTHER


async def discover_markets(contract_type_filter: list[str] | None = None) -> list[RegistryEntry]:
    """
    Full discovery pass:
      1. active_symbols -> every currently tradable underlying
      2. contracts_for(symbol) per symbol -> available contract types for that symbol
    Returns RegistryEntry objects ready to upsert into the `markets` table.
    """
    now = datetime.now(timezone.utc).isoformat()
    entries: list[RegistryEntry] = []

    async with DerivPublicClient() as client:
        symbols = await client.active_symbols(mode="full", contract_type=contract_type_filter)

        for s in symbols:
            symbol = s.get("underlying_symbol") or s.get("symbol")  # New vs Legacy fallback
            if not symbol:
                continue
            display_name = s.get("underlying_symbol_name") or s.get("display_name") or symbol

            try:
                contracts = await client.contracts_for(symbol)
                available = contracts.get("available", [])
                contract_types = sorted({c["contract_type"] for c in available if "contract_type" in c})
            except Exception:
                # A symbol that fails contracts_for is still registered, just marked
                # with no known contracts — the filter pipeline will reject it via
                # NO_TRADE_CONTRACT_UNAVAILABLE rather than crashing discovery.
                contract_types = []

            entries.append(
                RegistryEntry(
                    symbol=symbol,
                    display_name=display_name,
                    market=classify_market(s.get("market")),
                    submarket=s.get("submarket"),
                    exchange=s.get("exchange_name"),
                    available_contracts=contract_types,
                    status="ACTIVE" if s.get("is_trading_suspended", 0) == 0 else "SUSPENDED",
                    last_updated=now,
                )
            )

    return entries


def upsert_registry(db_path: str, entries: list[RegistryEntry]) -> None:
    """Writes/updates the `markets` table (app/storage/schema.sql) from a discovery pass."""
    conn = sqlite3.connect(db_path)
    try:
        for e in entries:
            conn.execute(
                """
                INSERT INTO markets
                    (symbol, display_name, market, submarket, exchange,
                     available_contracts, available_durations, trading_hours,
                     tick_frequency, status, last_updated)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(symbol) DO UPDATE SET
                    display_name=excluded.display_name,
                    market=excluded.market,
                    submarket=excluded.submarket,
                    exchange=excluded.exchange,
                    available_contracts=excluded.available_contracts,
                    status=excluded.status,
                    last_updated=excluded.last_updated
                """,
                (
                    e.symbol,
                    e.display_name,
                    e.market.value,
                    e.submarket,
                    e.exchange,
                    _json(e.available_contracts),
                    _json(e.available_durations),
                    _json(e.trading_hours),
                    e.tick_frequency,
                    e.status,
                    e.last_updated,
                ),
            )
        conn.commit()
    finally:
        conn.close()


def _json(value) -> str:
    import json
    return json.dumps(value)
