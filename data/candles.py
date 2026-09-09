"""
Candle aggregation (spec Part 6/8). Pure functions only — given a list of
(epoch, price) ticks already known up to some point in time, bucket them into
OHLC candles of a fixed duration. No function here ever looks past the last
tick it was given, so callers control the no-look-ahead guarantee simply by
never handing future ticks in.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Candle:
    symbol: str
    duration_s: int
    timestamp: int      # candle open epoch
    open: float
    high: float
    low: float
    close: float
    tick_count: int


def aggregate_ticks_to_candles(symbol: str, ticks: list[tuple[int, float]], duration_s: int) -> list[Candle]:
    """
    `ticks` must be (epoch, price) pairs, already sorted ascending by epoch —
    out-of-order or duplicate ticks should be filtered by data/quality.py
    before they ever reach here.
    """
    if not ticks:
        return []

    buckets: dict[int, list[float]] = {}
    bucket_order: list[int] = []
    for epoch, price in ticks:
        bucket_start = (epoch // duration_s) * duration_s
        if bucket_start not in buckets:
            buckets[bucket_start] = []
            bucket_order.append(bucket_start)
        buckets[bucket_start].append(price)

    candles: list[Candle] = []
    for bucket_start in bucket_order:
        prices = buckets[bucket_start]
        candles.append(
            Candle(
                symbol=symbol,
                duration_s=duration_s,
                timestamp=bucket_start,
                open=prices[0],
                high=max(prices),
                low=min(prices),
                close=prices[-1],
                tick_count=len(prices),
            )
        )
    return candles
