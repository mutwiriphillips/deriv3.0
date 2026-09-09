"""
Tick ingestion engine (spec Part 6). Wraps DerivPublicClient.stream_ticks with
the quality checks from data/quality.py and writes every tick (including
rejected ones — reasons matter for debugging) to the `ticks` table.
"""
from __future__ import annotations

import sqlite3
import time
from datetime import datetime, timezone

from app.data.quality import SymbolStreamState


class TickIngestor:
    def __init__(self, db_path: str, stale_after_s: float = 30.0):
        self._db_path = db_path
        self._states: dict[str, SymbolStreamState] = {}
        self._stale_after_s = stale_after_s

    def _state_for(self, symbol: str) -> SymbolStreamState:
        if symbol not in self._states:
            self._states[symbol] = SymbolStreamState(stale_after_s=self._stale_after_s)
        return self._states[symbol]

    def ingest_tick(self, symbol: str, raw_tick: dict, received_monotonic: float | None = None) -> int:
        """
        Evaluates and stores one raw tick dict (as returned by
        DerivPublicClient.stream_ticks / ticks_history). Returns the
        DataQualityScore assigned. Never raises on a bad tick — a bad tick
        is data, not an exception.
        """
        received_monotonic = time.monotonic() if received_monotonic is None else received_monotonic
        epoch = raw_tick.get("epoch")
        price = raw_tick.get("quote")

        result = self._state_for(symbol).evaluate(epoch, price, received_monotonic)

        conn = sqlite3.connect(self._db_path)
        try:
            direction = None
            conn.execute(
                """
                INSERT INTO ticks
                    (symbol, timestamp, price, tick_direction, source, received_at, latency_ms, quality_score)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    symbol,
                    _epoch_to_iso(epoch) if epoch else None,
                    price,
                    direction,
                    "deriv_ws",
                    datetime.now(timezone.utc).isoformat(),
                    None,
                    int(result.score),
                ),
            )
            conn.commit()
        finally:
            conn.close()

        return int(result.score)


def _epoch_to_iso(epoch: int) -> str:
    return datetime.fromtimestamp(epoch, tz=timezone.utc).isoformat()
