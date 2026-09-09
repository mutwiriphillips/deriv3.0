"""
CandleSeries is the only way feature functions see price history. It is
always a slice already truncated to "everything known at time t" — feature
functions never take a raw DB handle or a symbol+timestamp to look up
themselves, so there is no way for a feature to accidentally read the future.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CandleSeries:
    symbol: str
    duration_s: int
    timestamps: list[int]   # ascending, epoch seconds — timestamps[-1] is "now" for this series
    opens: list[float]
    highs: list[float]
    lows: list[float]
    closes: list[float]

    def __post_init__(self):
        n = len(self.closes)
        for name in ("timestamps", "opens", "highs", "lows"):
            if len(getattr(self, name)) != n:
                raise ValueError(f"CandleSeries field '{name}' length does not match closes")

    def __len__(self) -> int:
        return len(self.closes)

    def tail(self, n: int) -> "CandleSeries":
        """Last n candles, still respecting the same no-look-ahead invariant."""
        return CandleSeries(
            symbol=self.symbol,
            duration_s=self.duration_s,
            timestamps=self.timestamps[-n:],
            opens=self.opens[-n:],
            highs=self.highs[-n:],
            lows=self.lows[-n:],
            closes=self.closes[-n:],
        )
