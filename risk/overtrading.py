"""
Overtrading protection (spec Part 28) and duplicate-trade protection
(spec Part 29). All time-based checks take `now` as an explicit parameter
rather than reading a real clock internally — makes this deterministically
testable and keeps the module free of hidden global state.
"""
from __future__ import annotations

from collections import deque


class CooldownTracker:
    """Per-key (symbol, strategy_id, or 'global') minimum time between trades."""

    def __init__(self, cooldown_seconds: float):
        self.cooldown_seconds = cooldown_seconds
        self._last_trade_time: dict[str, float] = {}

    def is_satisfied(self, key: str, now: float) -> bool:
        last = self._last_trade_time.get(key)
        return last is None or (now - last) >= self.cooldown_seconds

    def record_trade(self, key: str, now: float) -> None:
        self._last_trade_time[key] = now


class RateLimiter:
    """Sliding-window trade-count limits per hour and per day."""

    def __init__(self, max_trades_per_hour: int, max_trades_per_day: int):
        self.max_trades_per_hour = max_trades_per_hour
        self.max_trades_per_day = max_trades_per_day
        self._trade_times: deque[float] = deque()

    def _prune(self, now: float) -> None:
        cutoff = now - 86400
        while self._trade_times and self._trade_times[0] < cutoff:
            self._trade_times.popleft()

    def is_within_limits(self, now: float) -> bool:
        self._prune(now)
        last_hour_count = sum(1 for t in self._trade_times if now - t < 3600)
        last_day_count = len(self._trade_times)
        return last_hour_count < self.max_trades_per_hour and last_day_count < self.max_trades_per_day

    def record_trade(self, now: float) -> None:
        self._trade_times.append(now)


class DuplicateSignalGuard:
    """
    Ensures a given signal_id only ever produces one trade — the spec's
    "before executing: check whether the signal already produced a trade"
    and the idempotency requirement around network retries after a
    purchase request whose response was lost.
    """

    def __init__(self):
        self._traded_signal_ids: set[str] = set()

    def has_already_traded(self, signal_id: str) -> bool:
        return signal_id in self._traded_signal_ids

    def mark_traded(self, signal_id: str) -> None:
        self._traded_signal_ids.add(signal_id)
