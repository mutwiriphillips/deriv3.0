"""
Every strategy — Forex, Synthetic, or Generic — implements this same interface
against a MarketContext. Strategies output a probability distribution over
{CALL, PUT, NO_TRADE}, never a raw indicator rule and never a stake or contract
decision (that belongs to risk/ and execution/).
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Protocol

from app.markets.context import MarketContext, Regime


class Direction(str, Enum):
    CALL = "CALL"
    PUT = "PUT"
    NO_TRADE = "NO_TRADE"


@dataclass(frozen=True)
class TradeCandidate:
    direction: Direction
    model_probability: float   # calibrated P(direction is correct), in [0, 1]
    confidence: float          # 0-1, reflects sample size / model uncertainty, NOT the same as probability
    signal_score: float
    reasons: list[str]         # human-readable justification, for the trade journal


class Strategy(Protocol):
    strategy_id: str
    allowed_regimes: set[Regime]
    allowed_symbols: set[str] | None      # None = no restriction (rare; usually set explicitly per §22)
    allowed_durations_s: set[int]

    def evaluate(self, ctx: MarketContext) -> TradeCandidate:
        """
        Pure function of `ctx` only. Must not consult any information timestamped
        after ctx.timestamp. Must return Direction.NO_TRADE (not raise) when the
        strategy has nothing to say about this context.
        """
        ...
