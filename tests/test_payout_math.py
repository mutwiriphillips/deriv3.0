import pytest

from app.risk.payout_math import (
    break_even_probability,
    evaluate_economics,
    expected_value,
    probability_edge,
)


def test_break_even_matches_spec_worked_example():
    # Spec Part 13: 80% payout -> break-even ~= 55.56%
    assert break_even_probability(0.80) == pytest.approx(0.5556, abs=0.0001)


def test_break_even_50_50_payout():
    assert break_even_probability(1.0) == pytest.approx(0.5)


def test_break_even_rejects_non_positive_payout():
    with pytest.raises(ValueError):
        break_even_probability(0.0)
    with pytest.raises(ValueError):
        break_even_probability(-0.1)


def test_expected_value_positive_when_edge_exists():
    ev = expected_value(model_probability=0.60, payout_ratio=0.80, stake=10.0)
    # 0.60 * 8.0 - 0.40 * 10.0 = 4.8 - 4.0 = 0.8
    assert ev == pytest.approx(0.8)


def test_expected_value_negative_below_break_even():
    # Spec's own point: 53% is NOT enough at 80% payout even though > 50%
    ev = expected_value(model_probability=0.53, payout_ratio=0.80, stake=1.0)
    assert ev < 0


def test_probability_edge_sign_matches_ev_sign_direction():
    edge_positive = probability_edge(0.60, 0.80)
    edge_negative = probability_edge(0.53, 0.80)
    assert edge_positive > 0
    assert edge_negative < 0


def test_evaluate_economics_bundles_everything_consistently():
    econ = evaluate_economics(model_probability=0.60, payout_ratio=0.80, stake=10.0)
    assert econ.break_even_probability == pytest.approx(break_even_probability(0.80))
    assert econ.probability_edge == pytest.approx(0.60 - econ.break_even_probability)
    assert econ.expected_value == pytest.approx(expected_value(0.60, 0.80, 10.0))
