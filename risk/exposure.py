"""
Correlated exposure manager (spec Part 27): "EURUSD CALL, GBPUSD CALL may
represent similar USD exposure... group correlated positions... if exposure
exceeds configured limits: NO TRADE."

Exposure is counted in simple directional units (one open position = ±1 to
each of its two currencies) — the spec doesn't specify a weighting scheme
beyond "group correlated positions," so this is a reasonable first pass, not
a claim that unit-weighting is the only valid approach.
"""
from __future__ import annotations

from app.strategies.base import Direction


def parse_forex_symbol(symbol: str) -> tuple[str, str] | None:
    """
    Deriv Forex symbols are typically "frx" + a 6-letter base/quote pair
    (e.g. "frxEURUSD" -> ("EUR", "USD")). Returns None for anything that
    doesn't match this shape — synthetic indices etc. simply have no
    currency exposure to track.
    """
    core = symbol[3:] if symbol.startswith("frx") else symbol
    if len(core) != 6 or not core.isalpha():
        return None
    return core[:3].upper(), core[3:].upper()


def currency_exposure_for_trade(symbol: str, direction: Direction) -> dict[str, float]:
    """CALL = long base currency, short quote currency (betting base strengthens vs quote); PUT is the reverse."""
    parsed = parse_forex_symbol(symbol)
    if parsed is None:
        return {}
    base, quote = parsed
    sign = 1.0 if direction == Direction.CALL else -1.0
    return {base: sign, quote: -sign}


class ExposureManager:
    def __init__(self, max_exposure_per_currency: float = 2.0):
        self.max_exposure_per_currency = max_exposure_per_currency
        self._positions: dict[str, dict[str, float]] = {}   # position_id -> currency exposure contributed

    def current_exposure(self) -> dict[str, float]:
        totals: dict[str, float] = {}
        for contribution in self._positions.values():
            for currency, amount in contribution.items():
                totals[currency] = totals.get(currency, 0.0) + amount
        return totals

    def would_exceed_limit(self, symbol: str, direction: Direction) -> bool:
        candidate = currency_exposure_for_trade(symbol, direction)
        if not candidate:
            return False  # nothing to check for non-forex symbols
        current = self.current_exposure()
        for currency, amount in candidate.items():
            projected = current.get(currency, 0.0) + amount
            if abs(projected) > self.max_exposure_per_currency:
                return True
        return False

    def open_position(self, position_id: str, symbol: str, direction: Direction) -> None:
        self._positions[position_id] = currency_exposure_for_trade(symbol, direction)

    def close_position(self, position_id: str) -> None:
        self._positions.pop(position_id, None)
