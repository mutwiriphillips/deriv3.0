"""
Statistical evaluation (spec Part 49): "Do not call a strategy profitable
based on 20/30/50 trades unless there is an extraordinary reason... Account
for multiple testing when evaluating many symbols, durations, indicators,
and strategies."

No scipy dependency — the normal-approximation p-value uses math.erf, which
is exact for this purpose and keeps this module free of a heavy dependency
for a fairly simple calculation.
"""
from __future__ import annotations

import math
from dataclasses import dataclass


def wilson_interval(wins: int, n: int, confidence: float = 0.95) -> tuple[float, float]:
    """
    Wilson score interval for a binomial proportion — more reliable than the
    naive normal approximation at small n or extreme win rates (both of
    which come up constantly when evaluating a new strategy).
    """
    if n == 0:
        return (0.0, 1.0)
    z = _z_for_confidence(confidence)
    phat = wins / n
    denom = 1 + z**2 / n
    center = phat + z**2 / (2 * n)
    margin = z * math.sqrt((phat * (1 - phat) + z**2 / (4 * n)) / n)
    low = (center - margin) / denom
    high = (center + margin) / denom
    return (max(0.0, low), min(1.0, high))


def _z_for_confidence(confidence: float) -> float:
    # Inverse-normal for the handful of confidence levels this system actually
    # uses; avoids pulling in scipy.stats.norm.ppf for one number.
    table = {0.90: 1.645, 0.95: 1.96, 0.99: 2.576}
    if confidence in table:
        return table[confidence]
    raise ValueError(f"unsupported confidence level {confidence}; add it to the lookup table")


def _normal_cdf(z: float) -> float:
    return 0.5 * (1 + math.erf(z / math.sqrt(2)))


def z_test_vs_break_even(wins: int, n: int, break_even_probability: float) -> tuple[float, float]:
    """
    One-sided z-test: is the observed win rate significantly ABOVE break-even?
    Returns (z_score, p_value). A strategy that isn't beating break-even
    doesn't need a two-sided test — we only ever care about "better", never
    "different in either direction."
    """
    if n == 0:
        return (0.0, 1.0)
    phat = wins / n
    se = math.sqrt(break_even_probability * (1 - break_even_probability) / n)
    if se == 0:
        return (float("inf") if phat > break_even_probability else 0.0, 0.0 if phat > break_even_probability else 1.0)
    z = (phat - break_even_probability) / se
    p_value = 1 - _normal_cdf(z)
    return (z, p_value)


def bonferroni_alpha(alpha: float, n_hypotheses: int) -> float:
    """
    Corrected significance threshold when testing `n_hypotheses` things at
    once (many symbols × durations × strategies). Conservative but simple —
    per spec Part 49's point that more hypotheses tested means more false
    edges found by chance unless this is accounted for.
    """
    if n_hypotheses < 1:
        raise ValueError("n_hypotheses must be >= 1")
    return alpha / n_hypotheses


@dataclass
class StatisticalEvaluation:
    n_trades: int
    win_rate: float
    wilson_ci_low: float
    wilson_ci_high: float
    break_even_probability: float
    z_score: float
    p_value: float
    alpha: float
    is_significant: bool
    min_sample_size: int
    meets_minimum_sample: bool

    @property
    def is_reliable(self) -> bool:
        """Both conditions must hold — significance without enough samples is exactly what Part 49 warns against."""
        return self.is_significant and self.meets_minimum_sample


def evaluate_significance(
    wins: int,
    n: int,
    break_even_probability: float,
    alpha: float = 0.05,
    min_sample_size: int = 100,
) -> StatisticalEvaluation:
    win_rate = wins / n if n else 0.0
    ci_low, ci_high = wilson_interval(wins, n)
    z, p = z_test_vs_break_even(wins, n, break_even_probability)
    return StatisticalEvaluation(
        n_trades=n,
        win_rate=win_rate,
        wilson_ci_low=ci_low,
        wilson_ci_high=ci_high,
        break_even_probability=break_even_probability,
        z_score=z,
        p_value=p,
        alpha=alpha,
        is_significant=p < alpha,
        min_sample_size=min_sample_size,
        meets_minimum_sample=n >= min_sample_size,
    )


def evaluate_backtest_significance(result, break_even_probability: float, alpha: float = 0.05, min_sample_size: int = 100) -> StatisticalEvaluation:
    """Convenience wrapper taking a backtest.simulator.BacktestResult directly."""
    return evaluate_significance(result.wins, result.total_trades, break_even_probability, alpha, min_sample_size)
