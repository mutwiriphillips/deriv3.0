"""
Shared behavior every strategy — baseline or otherwise — must respect:
"The strategy must not trade when data quality is below the configured
threshold" (spec Part 7). Concrete strategies implement `_evaluate` for the
happy path; `evaluate` wraps it with the gate so no strategy can forget it.
"""
from __future__ import annotations

from app.config.settings import settings
from app.markets.context import MarketContext
from app.monitoring.reasons import NoTradeReason
from app.strategies.base import Direction, TradeCandidate


class GatedStrategy:
    def evaluate(self, ctx: MarketContext) -> TradeCandidate:
        if ctx.data_quality_score < settings.min_data_quality:
            return TradeCandidate(
                direction=Direction.NO_TRADE,
                model_probability=0.0,
                confidence=0.0,
                signal_score=0.0,
                reasons=[NoTradeReason.DATA_QUALITY_BELOW_THRESHOLD.value],
            )
        return self._evaluate(ctx)

    def _evaluate(self, ctx: MarketContext) -> TradeCandidate:
        raise NotImplementedError
