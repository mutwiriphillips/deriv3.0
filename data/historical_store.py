"""
Local historical-data store (spec Part 6): "Do not repeatedly request the
same historical data unnecessarily." Before fetching, check what candle range
is already stored for (symbol, duration_s) and only request what's missing.
"""
from __future__ import annotations

import sqlite3

from app.data.candles import Candle
from app.markets.ws_client import DerivApiError, DerivPublicClient


def get_stored_range(db_path: str, symbol: str, duration_s: int) -> tuple[int | None, int | None]:
    """Returns (earliest_timestamp, latest_timestamp) already stored, or (None, None)."""
    conn = sqlite3.connect(db_path)
    try:
        row = conn.execute(
            "SELECT MIN(timestamp), MAX(timestamp) FROM candles WHERE symbol=? AND duration_s=?",
            (symbol, duration_s),
        ).fetchone()
        return row if row else (None, None)
    finally:
        conn.close()


def store_candles(db_path: str, candles: list[Candle]) -> int:
    """Inserts candles, skipping any (symbol, duration_s, timestamp) already present. Returns rows inserted."""
    if not candles:
        return 0
    conn = sqlite3.connect(db_path)
    inserted = 0
    try:
        for c in candles:
            cur = conn.execute(
                """
                INSERT OR IGNORE INTO candles
                    (symbol, duration_s, timestamp, open, high, low, close, volume, tick_count)
                VALUES (?, ?, ?, ?, ?, ?, ?, NULL, ?)
                """,
                (c.symbol, c.duration_s, c.timestamp, c.open, c.high, c.low, c.close, c.tick_count),
            )
            inserted += cur.rowcount
        conn.commit()
    finally:
        conn.close()
    return inserted


async def fetch_and_store_candles(
    db_path: str,
    symbol: str,
    duration_s: int,
    count: int = 5000,
    force_refetch: bool = False,
) -> int:
    """
    Fetches candle history for `symbol`/`duration_s` and stores whatever isn't
    already present. If we already have a stored range and `force_refetch` is
    False, this only asks Deriv for data since our latest stored candle
    (per the spec's "don't repeatedly request the same historical data").
    """
    earliest, latest = get_stored_range(db_path, symbol, duration_s)

    start = None
    if latest is not None and not force_refetch:
        start = str(latest + duration_s)  # only ask for what comes after what we have

    async with DerivPublicClient() as client:
        try:
            raw_candles = await client.ticks_history(
                symbol,
                style="candles",
                granularity=duration_s,
                count=count,
                start=start,
                end="latest",
            )
        except DerivApiError as e:
            if e.code == "InvalidStartEnd" and start is not None:
                # Confirmed against a real production error: this happens
                # whenever `start` (last stored candle + duration_s) lands at
                # or after the server's actual latest completed candle --
                # i.e. no new candle has closed since our last poll yet. With
                # a poll interval matching the candle duration, this is an
                # expected, recurring condition, not a real error: there's
                # simply nothing new to fetch on this tick.
                return 0
            raise

    candles = [
        Candle(
            symbol=symbol,
            duration_s=duration_s,
            timestamp=int(rc["epoch"]),
            open=float(rc["open"]),
            high=float(rc["high"]),
            low=float(rc["low"]),
            close=float(rc["close"]),
            tick_count=0,  # not provided by ticks_history candles; left unknown rather than guessed
        )
        for rc in (raw_candles or [])
    ]
    return store_candles(db_path, candles)
