"""
Normalized market context passed to every strategy, regardless of market family.

No strategy may depend on Forex-specific fields directly; anything market-specific
belongs in `extra`, and a strategy must explicitly opt in to reading it.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Optional


class MarketType(str, Enum):
    FOREX = "FOREX"
    SYNTHETIC = "SYNTHETIC"
    COMMODITY = "COMMODITY"
    OTHER = "OTHER"


class Regime(str, Enum):
    TREND_UP = "TREND_UP"
    TREND_DOWN = "TREND_DOWN"
    RANGE = "RANGE"
    HIGH_VOLATILITY = "HIGH_VOLATILITY"
    LOW_VOLATILITY = "LOW_VOLATILITY"
    BREAKOUT = "BREAKOUT"
    MEAN_REVERSION = "MEAN_REVERSION"
    UNKNOWN = "UNKNOWN"


@dataclass(frozen=True)
class MarketContext:
    symbol: str
    market_type: MarketType
    timestamp: datetime            # the instant this context is valid for — nothing after it may be used
    price: float
    recent_ticks: list[float]      # must already be sliced to [:timestamp], no look-ahead
    candles: dict[int, list[dict]] # keyed by duration_s -> list of OHLC dicts, oldest first, sliced to timestamp
    volatility: float
    trend: float
    momentum: float
    regime: Regime
    available_contracts: list[str]
    payout: Optional[dict[str, float]] = None   # contract_type -> payout ratio, from a live proposal
    signal_features: dict[str, float] = field(default_factory=dict)
    data_quality_score: int = 0    # 0=unusable .. 3=high quality; strategies should refuse below threshold
    extra: dict[str, Any] = field(default_factory=dict)  # market-specific escape hatch
