"""
Baseline strategies (spec Part 16). Deliberately unsophisticated — these
exist to be beaten, not traded. Every probability estimate here is an honest
heuristic, not a calibrated model; calibration only enters with the ML
strategies in Part 17.
"""
from __future__ import annotations

import math
import random
from collections import Counter

from app.markets.context import Regime
from app.strategies.base import Direction, TradeCandidate
from app.strategies.gated import GatedStrategy


def _magnitude_to_probability(magnitude: float, scale: float, min_prob: float = 0.5, max_prob: float = 0.85) -> float:
    """
    Monotonic squash of an unbounded signal magnitude into a probability in
    (min_prob, max_prob). Never reaches max_prob exactly — a baseline should
    never claim near-certainty.
    """
    squashed = min(1 - math.exp(-abs(magnitude) * scale), 0.999)
    return min_prob + (max_prob - min_prob) * squashed


class RandomStrategy(GatedStrategy):
    """Coin-flip direction. The floor every other strategy must clear."""

    strategy_id = "baseline_random"
    allowed_regimes = set(Regime)
    allowed_symbols = None
    allowed_durations_s: set[int] = set()  # unrestricted; duration is chosen at the StrategyRun level

    def __init__(self, seed: int | None = None):
        self._rng = random.Random(seed)

    def _evaluate(self, ctx) -> TradeCandidate:
        direction = self._rng.choice([Direction.CALL, Direction.PUT])
        return TradeCandidate(
            direction=direction,
            model_probability=0.5,
            confidence=0.1,   # deliberately low — a coin flip has no real edge to be confident about
            signal_score=0.0,
            reasons=["baseline_random_coin_flip"],
        )


class MajorityClassStrategy(GatedStrategy):
    """
    Always predicts whichever direction won more often historically. Must be
    `fit()` on real outcomes before use — there is no sensible default.
    """

    strategy_id = "baseline_majority_class"
    allowed_regimes = set(Regime)
    allowed_symbols = None
    allowed_durations_s: set[int] = set()

    def __init__(self, majority_direction: Direction, majority_probability: float, n_samples: int):
        self.majority_direction = majority_direction
        self.majority_probability = majority_probability
        self.n_samples = n_samples

    @classmethod
    def fit(cls, historical_outcomes: list[Direction]) -> "MajorityClassStrategy":
        if not historical_outcomes:
            raise ValueError("MajorityClassStrategy.fit requires at least one historical outcome")
        counts = Counter(historical_outcomes)
        majority_direction, majority_count = counts.most_common(1)[0]
        return cls(
            majority_direction=majority_direction,
            majority_probability=majority_count / len(historical_outcomes),
            n_samples=len(historical_outcomes),
        )

    def _evaluate(self, ctx) -> TradeCandidate:
        # Confidence grows with sample size but is capped — per Part 15, even
        # a large sample doesn't earn full confidence from a model this simple.
        confidence = min(0.6, self.n_samples / 1000)
        return TradeCandidate(
            direction=self.majority_direction,
            model_probability=self.majority_probability,
            confidence=confidence,
            signal_score=self.majority_probability - 0.5,
            reasons=[f"baseline_majority_class_n={self.n_samples}"],
        )


class SimpleMomentumStrategy(GatedStrategy):
    """CALL if recent ROC is positive, PUT if negative, NO_TRADE if flat/unknown."""

    strategy_id = "baseline_simple_momentum"
    allowed_regimes = set(Regime) - {Regime.UNKNOWN}
    allowed_symbols = None
    allowed_durations_s: set[int] = set()

    def __init__(self, roc_key: str = "roc_10", scale: float = 50.0):
        self.roc_key = roc_key
        self.scale = scale

    def _evaluate(self, ctx) -> TradeCandidate:
        roc = ctx.signal_features.get(self.roc_key)
        if roc is None or roc == 0:
            return TradeCandidate(
                direction=Direction.NO_TRADE,
                model_probability=0.5,
                confidence=0.0,
                signal_score=0.0,
                reasons=["baseline_momentum_flat_or_unknown"],
            )
        direction = Direction.CALL if roc > 0 else Direction.PUT
        probability = _magnitude_to_probability(roc, self.scale)
        return TradeCandidate(
            direction=direction,
            model_probability=probability,
            confidence=0.3,  # fixed, modest — this is a heuristic, not a fitted model
            signal_score=probability - 0.5,
            reasons=[f"baseline_momentum_roc={roc:.5f}"],
        )


class SimpleTrendStrategy(GatedStrategy):
    """
    CALL/PUT only in a TREND_UP/TREND_DOWN regime, using EMA separation's
    sign and magnitude. NO_TRADE in any other regime — this baseline
    deliberately does not guess outside its own stated conditions.
    """

    strategy_id = "baseline_simple_trend"
    allowed_regimes = {Regime.TREND_UP, Regime.TREND_DOWN}
    allowed_symbols = None
    allowed_durations_s: set[int] = set()

    def __init__(self, ma_separation_key: str = "ma_separation", scale: float = 30.0):
        self.ma_separation_key = ma_separation_key
        self.scale = scale

    def _evaluate(self, ctx) -> TradeCandidate:
        if ctx.regime not in self.allowed_regimes:
            return TradeCandidate(
                direction=Direction.NO_TRADE,
                model_probability=0.5,
                confidence=0.0,
                signal_score=0.0,
                reasons=[f"baseline_trend_regime_not_allowed:{ctx.regime.value}"],
            )
        sep = ctx.signal_features.get(self.ma_separation_key)
        if sep is None:
            return TradeCandidate(
                direction=Direction.NO_TRADE,
                model_probability=0.5,
                confidence=0.0,
                signal_score=0.0,
                reasons=["baseline_trend_insufficient_history"],
            )
        direction = Direction.CALL if ctx.regime == Regime.TREND_UP else Direction.PUT
        probability = _magnitude_to_probability(sep, self.scale)
        return TradeCandidate(
            direction=direction,
            model_probability=probability,
            confidence=0.35,
            signal_score=probability - 0.5,
            reasons=[f"baseline_trend_regime={ctx.regime.value}_sep={sep:.5f}"],
        )
