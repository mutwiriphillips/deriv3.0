import numpy as np
import pytest

from app.models.dataset import FEATURE_KEYS, build_training_examples, features_to_vector
from app.models.logistic_model import LogisticProbabilityModel
from app.models.evaluation import brier_score, calibration_curve, calibration_error, roc_auc
from app.models.baseline_comparison import compare_to_baselines
from app.strategies.ml_strategy import ProbabilityModelStrategy
from app.markets.context import MarketContext, MarketType, Regime
from app.strategies.base import Direction


# --- dataset.py ---

def test_features_to_vector_casts_booleans_and_preserves_none():
    features = {"returns": 0.01, "higher_high": True, "lower_low": False, "rsi_14": None}
    vector = features_to_vector(features)
    idx = {k: i for i, k in enumerate(FEATURE_KEYS)}
    assert vector[idx["returns"]] == 0.01
    assert vector[idx["higher_high"]] == 1.0
    assert vector[idx["lower_low"]] == 0.0
    assert vector[idx["rsi_14"]] is None


def test_build_training_examples_labels_match_future_direction():
    n = 200
    closes = list(1.0 + np.cumsum(np.full(n, 0.001)))  # strictly increasing
    timestamps = [1_700_000_000 + i * 60 for i in range(n)]
    highs = [c + 0.0005 for c in closes]
    lows = [c - 0.0005 for c in closes]
    opens = closes

    examples = build_training_examples(timestamps, opens, highs, lows, closes, duration_candles=1, warmup=60)
    labels = [label for _, label in examples]
    assert all(label == 1 for label in labels)  # strictly increasing series -> every future close is higher


def test_build_training_examples_produces_no_out_of_range_index():
    n = 80
    closes = list(np.linspace(1.0, 1.1, n))
    timestamps = [1_700_000_000 + i * 60 for i in range(n)]
    highs = [c + 0.0005 for c in closes]
    lows = [c - 0.0005 for c in closes]
    examples = build_training_examples(timestamps, closes, highs, lows, closes, duration_candles=5, warmup=60)
    assert len(examples) >= 0  # just must not raise


# --- logistic_model.py ---

def make_separable_examples(n=500, seed=0):
    """Synthetic problem where feature 0 ('returns') perfectly determines the label -- an easy sanity check."""
    rng = np.random.default_rng(seed)
    examples = []
    for _ in range(n):
        vector = [None] * len(FEATURE_KEYS)
        returns_val = rng.normal(0, 1)
        vector[FEATURE_KEYS.index("returns")] = returns_val
        vector[FEATURE_KEYS.index("rsi_14")] = rng.uniform(0, 100)  # noise feature
        label = 1 if returns_val > 0 else 0
        examples.append((vector, label))
    return examples


def test_model_fit_rejects_too_few_examples():
    model = LogisticProbabilityModel()
    with pytest.raises(ValueError):
        model.fit([([1.0] * len(FEATURE_KEYS), 1)] * 5)


def test_model_learns_a_separable_relationship():
    examples = make_separable_examples(n=500, seed=1)
    train, test = examples[:400], examples[400:]

    model = LogisticProbabilityModel(calibrate=True)
    fit_result = model.fit(train)
    assert fit_result.n_train == 400

    y_true = [label for _, label in test]
    y_prob = [model.predict_proba(vec) for vec, _ in test]
    auc = roc_auc(y_true, y_prob)
    assert auc > 0.85  # should learn this easy relationship well out-of-sample


def test_model_predict_before_fit_raises():
    model = LogisticProbabilityModel()
    with pytest.raises(RuntimeError):
        model.predict_proba([None] * len(FEATURE_KEYS))


def test_model_imputes_missing_features_at_predict_time():
    examples = make_separable_examples(n=300, seed=2)
    model = LogisticProbabilityModel()
    model.fit(examples)
    # An all-None vector should not raise -- gets fully imputed from training medians
    p = model.predict_proba([None] * len(FEATURE_KEYS))
    assert 0.0 <= p <= 1.0


# --- evaluation.py ---

def test_brier_score_perfect_predictions_is_zero():
    assert brier_score([1, 0, 1, 0], [1.0, 0.0, 1.0, 0.0]) == pytest.approx(0.0)


def test_brier_score_worst_predictions_is_one():
    assert brier_score([1, 0], [0.0, 1.0]) == pytest.approx(1.0)


def test_brier_score_constant_half_on_balanced_labels():
    assert brier_score([1, 0, 1, 0], [0.5, 0.5, 0.5, 0.5]) == pytest.approx(0.25)


def test_roc_auc_none_for_single_class():
    assert roc_auc([1, 1, 1], [0.6, 0.7, 0.8]) is None


def test_calibration_curve_well_calibrated_case():
    # 100 predictions at p=0.7, with actual positive rate close to 0.7
    rng = np.random.default_rng(3)
    y_prob = [0.7] * 100
    y_true = [1 if rng.random() < 0.7 else 0 for _ in range(100)]
    bins = calibration_curve(y_true, y_prob, n_bins=10)
    populated = [b for b in bins if b.count > 0]
    assert len(populated) == 1
    assert populated[0].predicted_mean == pytest.approx(0.7)
    assert populated[0].actual_rate == pytest.approx(0.7, abs=0.15)


def test_calibration_error_low_when_well_calibrated():
    y_prob = [0.5] * 200
    y_true = [1, 0] * 100  # exactly 50% positive rate
    bins = calibration_curve(y_true, y_prob, n_bins=10)
    ece = calibration_error(bins)
    assert ece == pytest.approx(0.0, abs=0.01)


def test_calibration_error_none_when_no_bins_populated():
    assert calibration_error([]) is None


# --- ml_strategy.py ---

def make_ctx(regime=Regime.RANGE, data_quality_score=3, signal_features=None):
    from datetime import datetime, timezone
    return MarketContext(
        symbol="frxEURUSD", market_type=MarketType.FOREX, timestamp=datetime.now(timezone.utc),
        price=1.085, recent_ticks=[], candles={}, volatility=0.001, trend=0.0, momentum=50.0,
        regime=regime, available_contracts=["CALL", "PUT"],
        signal_features=signal_features or {}, data_quality_score=data_quality_score,
    )


def test_ml_strategy_respects_data_quality_gate():
    examples = make_separable_examples(n=100, seed=4)
    model = LogisticProbabilityModel()
    model.fit(examples)
    strategy = ProbabilityModelStrategy(model, n_train=100)

    ctx = make_ctx(data_quality_score=0, signal_features={"returns": 1.0})
    result = strategy.evaluate(ctx)
    assert result.direction == Direction.NO_TRADE
    assert "DATA_QUALITY" in result.reasons[0]


def test_ml_strategy_direction_follows_model_prediction():
    examples = make_separable_examples(n=500, seed=5)
    model = LogisticProbabilityModel()
    model.fit(examples)
    strategy = ProbabilityModelStrategy(model, n_train=500)

    ctx_up = make_ctx(signal_features={"returns": 3.0})
    ctx_down = make_ctx(signal_features={"returns": -3.0})
    result_up = strategy.evaluate(ctx_up)
    result_down = strategy.evaluate(ctx_down)
    assert result_up.direction == Direction.CALL
    assert result_down.direction == Direction.PUT


def test_ml_strategy_no_trade_when_no_features_at_all():
    examples = make_separable_examples(n=100, seed=6)
    model = LogisticProbabilityModel()
    model.fit(examples)
    strategy = ProbabilityModelStrategy(model, n_train=100)
    ctx = make_ctx(signal_features={})
    result = strategy.evaluate(ctx)
    assert result.direction == Direction.NO_TRADE


# --- baseline_comparison.py ---

class FakeBacktestResult:
    def __init__(self, ev, wins, total):
        self.realized_ev_per_stake = ev
        self.wins = wins
        self.total_trades = total


def test_compare_to_baselines_approved_when_candidate_beats_all_and_is_significant():
    candidate = FakeBacktestResult(ev=0.10, wins=600, total=1000)
    baselines = {"random": FakeBacktestResult(ev=0.0, wins=500, total=1000)}
    report = compare_to_baselines(candidate, baselines, break_even_probability=0.5556, min_sample_size=100)
    assert report.approved is True


def test_compare_to_baselines_rejected_when_candidate_underperforms_a_baseline():
    candidate = FakeBacktestResult(ev=0.01, wins=560, total=1000)
    baselines = {"strong_baseline": FakeBacktestResult(ev=0.05, wins=590, total=1000)}
    report = compare_to_baselines(candidate, baselines, break_even_probability=0.5556, min_sample_size=100)
    assert report.approved is False
    assert any("did not beat baseline" in r for r in report.reasons)


def test_compare_to_baselines_rejected_when_sample_too_small_even_if_it_beats_baselines():
    candidate = FakeBacktestResult(ev=0.20, wins=18, total=20)  # tiny sample, Part 49 trap
    baselines = {"random": FakeBacktestResult(ev=0.0, wins=10, total=20)}
    report = compare_to_baselines(candidate, baselines, break_even_probability=0.5556, min_sample_size=100)
    assert report.approved is False
    assert report.candidate_is_statistically_reliable is False
