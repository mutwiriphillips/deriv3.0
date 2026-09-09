"""
Stake sizing (spec Part 25): "Never use Martingale. Never increase stake
merely because the previous trade lost." Fixed-fractional only — this
function has no memory of past trades at all, which is itself the guarantee
against loss-chasing: there's nothing here a losing streak could feed into.
"""
from __future__ import annotations


def calculate_stake(balance: float, risk_per_trade: float, max_stake: float) -> float:
    if balance <= 0:
        return 0.0
    if risk_per_trade <= 0:
        raise ValueError("risk_per_trade must be positive")
    stake = balance * risk_per_trade
    return min(stake, max_stake)
