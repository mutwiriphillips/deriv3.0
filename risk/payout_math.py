"""
Binary payout mathematics (spec Parts 13-14). This is deliberately the one
place break-even probability and expected value get computed — the
backtester, and later risk/execution, both import from here rather than
recomputing it inline.
"""
from __future__ import annotations

from dataclasses import dataclass


def break_even_probability(payout_ratio: float) -> float:
    """
    payout_ratio = net profit per unit staked if the contract wins (e.g. 0.80
    means winning returns 1.80x stake, i.e. 0.80 profit per 1.00 staked).
    break_even_probability = 1 / (1 + payout_ratio) — the spec's own example:
    an 80% payout needs P(win) >= 1/1.8 = 55.56% just to break even.
    """
    if payout_ratio <= 0:
        raise ValueError("payout_ratio must be positive")
    return 1.0 / (1.0 + payout_ratio)


def expected_value(model_probability: float, payout_ratio: float, stake: float = 1.0) -> float:
    """EV = P(win) * net_win - P(loss) * stake, per unit stake by default."""
    net_win = stake * payout_ratio
    return model_probability * net_win - (1 - model_probability) * stake


def probability_edge(model_probability: float, payout_ratio: float) -> float:
    return model_probability - break_even_probability(payout_ratio)


@dataclass(frozen=True)
class TradeEconomics:
    model_probability: float
    payout_ratio: float
    stake: float
    break_even_probability: float
    probability_edge: float
    expected_value: float


def evaluate_economics(model_probability: float, payout_ratio: float, stake: float = 1.0) -> TradeEconomics:
    be = break_even_probability(payout_ratio)
    return TradeEconomics(
        model_probability=model_probability,
        payout_ratio=payout_ratio,
        stake=stake,
        break_even_probability=be,
        probability_edge=model_probability - be,
        expected_value=expected_value(model_probability, payout_ratio, stake),
    )
