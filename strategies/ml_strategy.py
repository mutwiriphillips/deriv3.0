"""
Wraps a fitted LogisticProbabilityModel as a Strategy, so it can be run
through the exact same backtester/filter-pipeline path as the baselines —
no special-cased evaluation path for "the ML one."
"""
from __future__ import annotations

from app.markets.context import Regime
from app.models.dataset import features_to_vector
from app.models.logistic_model import LogisticProbabilityModel
from app.strategies.base import Direction, TradeCandidate
from app.strategies.gated import GatedStrategy


class ProbabilityModelStrategy(GatedStrategy):
    strategy_id = "logistic_probability_model"
    allowed_regimes = set(Regime) - {Regime.UNKNOWN}
    allowed_symbols = None
    allowed_durations_s: set[int] = set()

    def __init__(self, model: LogisticProbabilityModel, n_train: int, decision_threshold: float = 0.5):
        self.model = model
        self.n_train = n_train
        self.decision_threshold = decision_threshold

    def _evaluate(self, ctx) -> TradeCandidate:
        vector = features_to_vector(ctx.signal_features)
        if all(v is None for v in vector):
            return TradeCandidate(
                direction=Direction.NO_TRADE,
                model_probability=0.5,
                confidence=0.0,
                signal_score=0.0,
                reasons=["ml_model_no_features_available"],
            )

        p_up = self.model.predict_proba(vector)
        if p_up > self.decision_threshold:
            direction = Direction.CALL
            probability = p_up
        elif p_up < self.decision_threshold:
            direction = Direction.PUT
            probability = 1 - p_up
        else:
            return TradeCandidate(
                direction=Direction.NO_TRADE,
                model_probability=0.5,
                confidence=0.0,
                signal_score=0.0,
                reasons=["ml_model_exactly_at_decision_threshold"],
            )

        # Confidence reflects training sample size per Part 15, capped —
        # calibration makes the probability itself trustworthy, but a model
        # trained on a small sample still shouldn't claim high confidence.
        confidence = min(0.8, self.n_train / 5000)

        return TradeCandidate(
            direction=direction,
            model_probability=probability,
            confidence=confidence,
            signal_score=probability - 0.5,
            reasons=[f"ml_model_p_up={p_up:.4f}_n_train={self.n_train}"],
        )
