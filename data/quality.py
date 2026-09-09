"""
Data quality scoring (spec Part 7). A tick can't influence a trading decision
below the configured threshold (app.config.settings.min_data_quality).

Score scale: 0=unusable, 1=questionable, 2=acceptable, 3=high quality.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum


class DataQualityScore(IntEnum):
    UNUSABLE = 0
    QUESTIONABLE = 1
    ACCEPTABLE = 2
    HIGH_QUALITY = 3


@dataclass
class TickCheckResult:
    score: DataQualityScore
    is_duplicate: bool
    is_out_of_order: bool
    is_stale: bool
    reasons: list[str]


class SymbolStreamState:
    """
    Per-symbol rolling state needed to evaluate each incoming tick. One
    instance per symbol; the caller (TickIngestor) owns the dict of these.
    """

    def __init__(self, stale_after_s: float = 30.0):
        self.last_epoch: int | None = None
        self.last_received_monotonic: float | None = None
        self.stale_after_s = stale_after_s

    def evaluate(self, epoch: int, price: float, received_monotonic: float) -> TickCheckResult:
        reasons: list[str] = []
        is_duplicate = False
        is_out_of_order = False
        is_stale = False

        # --- price validity ---
        if price is None or price <= 0 or price != price:  # NaN check via self-inequality
            reasons.append("invalid_price")

        # --- timestamp validity ---
        if epoch is None or epoch <= 0:
            reasons.append("invalid_timestamp")

        # --- duplicate / out-of-order (only meaningful once we've seen a prior tick) ---
        if self.last_epoch is not None:
            if epoch == self.last_epoch:
                is_duplicate = True
                reasons.append("duplicate_tick")
            elif epoch < self.last_epoch:
                is_out_of_order = True
                reasons.append("out_of_order_tick")

        # --- staleness (wall-clock gap since the last tick we received) ---
        if self.last_received_monotonic is not None:
            gap = received_monotonic - self.last_received_monotonic
            if gap > self.stale_after_s:
                is_stale = True
                reasons.append("stale_gap")

        # Only advance state on ticks that are at least chronologically valid —
        # a duplicate or out-of-order tick must not corrupt "last known good" state.
        if not reasons or reasons == ["stale_gap"]:
            if self.last_epoch is None or epoch > self.last_epoch:
                self.last_epoch = epoch
            self.last_received_monotonic = received_monotonic

        if "invalid_price" in reasons or "invalid_timestamp" in reasons:
            score = DataQualityScore.UNUSABLE
        elif is_duplicate or is_out_of_order:
            score = DataQualityScore.QUESTIONABLE
        elif is_stale:
            score = DataQualityScore.ACCEPTABLE
        else:
            score = DataQualityScore.HIGH_QUALITY

        return TickCheckResult(
            score=score,
            is_duplicate=is_duplicate,
            is_out_of_order=is_out_of_order,
            is_stale=is_stale,
            reasons=reasons,
        )
