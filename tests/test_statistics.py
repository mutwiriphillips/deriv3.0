import pytest

from app.backtest.statistics import (
    bonferroni_alpha,
    evaluate_significance,
    wilson_interval,
    z_test_vs_break_even,
)


# --- wilson_interval ---

def test_wilson_interval_contains_observed_rate():
    low, high = wilson_interval(wins=60, n=100)
    assert low < 0.6 < high


def test_wilson_interval_narrows_with_more_samples():
    low_small, high_small = wilson_interval(wins=6, n=10)
    low_large, high_large = wilson_interval(wins=600, n=1000)
    assert (high_large - low_large) < (high_small - low_small)


def test_wilson_interval_empty_sample_is_maximally_uncertain():
    assert wilson_interval(0, 0) == (0.0, 1.0)


def test_wilson_interval_bounds_stay_in_unit_range():
    low, high = wilson_interval(wins=1, n=1)
    assert 0.0 <= low <= high <= 1.0


# --- z_test_vs_break_even, checked against known z-table p-values ---

def test_p_value_matches_known_z_table_value_at_z_1_645():
    # z = 1.645 -> one-sided p ~= 0.05 (standard textbook value)
    wins, n = 0, 0
    # construct wins/n and break_even to produce z close to 1.645
    break_even = 0.5
    n = 1000
    se = (break_even * (1 - break_even) / n) ** 0.5
    target_z = 1.645
    phat = break_even + target_z * se
    wins = round(phat * n)
    z, p = z_test_vs_break_even(wins, n, break_even)
    assert z == pytest.approx(1.645, abs=0.01)
    assert p == pytest.approx(0.05, abs=0.005)


def test_p_value_matches_known_z_table_value_at_z_1_96():
    break_even = 0.5
    n = 1000
    se = (break_even * (1 - break_even) / n) ** 0.5
    target_z = 1.96
    phat = break_even + target_z * se
    wins = round(phat * n)
    z, p = z_test_vs_break_even(wins, n, break_even)
    assert p == pytest.approx(0.025, abs=0.005)


def test_z_test_not_significant_when_below_break_even():
    z, p = z_test_vs_break_even(wins=40, n=100, break_even_probability=0.5556)
    assert z < 0
    assert p > 0.5


def test_z_test_zero_trades_is_never_significant():
    z, p = z_test_vs_break_even(wins=0, n=0, break_even_probability=0.5)
    assert p == 1.0


# --- bonferroni_alpha ---

def test_bonferroni_alpha_basic():
    assert bonferroni_alpha(0.05, 10) == pytest.approx(0.005)


def test_bonferroni_alpha_rejects_invalid_n():
    with pytest.raises(ValueError):
        bonferroni_alpha(0.05, 0)


# --- evaluate_significance / the Part 49 "20 trades isn't enough" point ---

def test_significant_but_insufficient_sample_is_flagged_unreliable():
    # A small sample can look statistically significant by chance but still
    # shouldn't be trusted per spec Part 49 -- is_reliable must catch this.
    evaluation = evaluate_significance(wins=18, n=20, break_even_probability=0.5556, min_sample_size=100)
    assert evaluation.n_trades == 20
    assert evaluation.meets_minimum_sample is False
    assert evaluation.is_reliable is False  # regardless of whether is_significant happens to be True


def test_large_sample_with_real_edge_is_reliable():
    # 600/1000 wins at break-even 0.5556 is a real, large-sample edge.
    evaluation = evaluate_significance(wins=600, n=1000, break_even_probability=0.5556, min_sample_size=100)
    assert evaluation.is_significant is True
    assert evaluation.meets_minimum_sample is True
    assert evaluation.is_reliable is True


def test_large_sample_at_break_even_is_not_significant():
    evaluation = evaluate_significance(wins=556, n=1000, break_even_probability=0.5556, min_sample_size=100)
    assert evaluation.is_significant is False
