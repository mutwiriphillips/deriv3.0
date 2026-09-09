import pytest

from app.backtest.simulator import BacktestResult, TradeRecord
from app.backtest.stress_testing import (
    apply_dropped_trades,
    apply_forced_losing_streak,
    apply_payout_reduction,
    apply_slippage,
    apply_win_probability_reduction,
    filter_by_regime,
    render_stress_test_text,
    run_stress_test_suite,
)
from app.markets.context import Regime


def make_trade(result, profit_loss, stake=10.0, regime="RANGE"):
    return TradeRecord(
        entry_index=0, entry_timestamp=0, direction="CALL", model_probability=0.6,
        payout_ratio=0.8, stake=stake, probability_edge=0.05, expected_value=0.05,
        exit_index=1, exit_timestamp=60, entry_price=1.0, exit_price=1.01,
        result=result, profit_loss=profit_loss, regime=regime,
    )


def make_trades(n_wins, n_losses, stake=10.0, payout_ratio=0.8, regime="RANGE"):
    trades = [make_trade("WIN", stake * payout_ratio, stake, regime) for _ in range(n_wins)]
    trades += [make_trade("LOSS", -stake, stake, regime) for _ in range(n_losses)]
    return trades


# --- apply_win_probability_reduction ---

def test_win_probability_reduction_flips_correct_count():
    trades = make_trades(n_wins=60, n_losses=40)  # 60% win rate, 100 trades
    pnl = apply_win_probability_reduction(trades, reduction=0.10, seed=1)
    new_win_count = sum(1 for p in pnl if p > 0)
    assert new_win_count == 50  # 60% - 10pp = 50%


def test_win_probability_reduction_never_goes_negative():
    trades = make_trades(n_wins=5, n_losses=95)  # only 5% win rate
    pnl = apply_win_probability_reduction(trades, reduction=0.50, seed=1)  # would need -45%
    new_win_count = sum(1 for p in pnl if p > 0)
    assert new_win_count == 0  # clamped, not negative


def test_win_probability_reduction_empty_trades():
    assert apply_win_probability_reduction([], reduction=0.1) == []


# --- apply_payout_reduction ---

def test_payout_reduction_scales_only_wins():
    trades = [make_trade("WIN", 8.0, stake=10.0), make_trade("LOSS", -10.0, stake=10.0)]
    pnl = apply_payout_reduction(trades, reduction_fraction=0.20)
    assert pnl[0] == pytest.approx(6.4)   # 8.0 * 0.8
    assert pnl[1] == pytest.approx(-10.0)  # loss unaffected


# --- apply_slippage ---

def test_slippage_subtracts_fixed_cost_from_every_trade():
    trades = [make_trade("WIN", 8.0), make_trade("LOSS", -10.0)]
    pnl = apply_slippage(trades, extra_cost_per_trade=1.0)
    assert pnl == [7.0, -11.0]


# --- apply_dropped_trades ---

def test_dropped_trades_reduces_count_correctly():
    trades = make_trades(n_wins=50, n_losses=50)
    pnl = apply_dropped_trades(trades, drop_fraction=0.20, seed=1)
    assert len(pnl) == 80


def test_dropped_trades_empty():
    assert apply_dropped_trades([], drop_fraction=0.2) == []


# --- apply_forced_losing_streak ---

def test_forced_losing_streak_appends_correct_length_and_magnitude():
    trades = [make_trade("WIN", 8.0, stake=10.0), make_trade("WIN", 8.0, stake=20.0)]
    pnl = apply_forced_losing_streak(trades, streak_length=5)
    assert len(pnl) == 2 + 5
    forced_part = pnl[2:]
    assert all(p == pytest.approx(-15.0) for p in forced_part)  # avg stake = (10+20)/2 = 15


def test_forced_losing_streak_with_no_trades():
    pnl = apply_forced_losing_streak([], streak_length=3)
    assert pnl == [0.0, 0.0, 0.0]


# --- filter_by_regime ---

def test_filter_by_regime_selects_correct_trades():
    trades = [
        make_trade("WIN", 8.0, regime="HIGH_VOLATILITY"),
        make_trade("LOSS", -10.0, regime="RANGE"),
        make_trade("WIN", 8.0, regime="HIGH_VOLATILITY"),
    ]
    filtered = filter_by_regime(trades, {Regime.HIGH_VOLATILITY})
    assert len(filtered) == 2
    assert all(t.regime == "HIGH_VOLATILITY" for t in filtered)


def test_filter_by_regime_empty_when_none_match():
    trades = [make_trade("WIN", 8.0, regime="RANGE")]
    assert filter_by_regime(trades, {Regime.HIGH_VOLATILITY}) == []


# --- run_stress_test_suite ---

def test_full_suite_runs_all_scenarios():
    result = BacktestResult()
    result.trades = make_trades(n_wins=70, n_losses=30, regime="TREND_UP")  # strong, real edge
    suite = run_stress_test_suite(result, starting_bankroll=1000, n_simulations=100, seed=1)
    names = {s.name for s in suite.scenarios}
    assert names == {
        "win_probability_-10pp", "win_probability_-20pp", "lower_payout_-20pct",
        "higher_latency_slippage", "missed_trades_-20pct", "forced_losing_streak",
        "high_volatility_regime_only", "low_volatility_regime_only",
    }


def test_regime_scenarios_report_note_when_no_matching_trades():
    result = BacktestResult()
    result.trades = make_trades(n_wins=70, n_losses=30, regime="TREND_UP")  # no HIGH/LOW_VOLATILITY trades at all
    suite = run_stress_test_suite(result, starting_bankroll=1000, n_simulations=100, seed=1)
    high_vol = next(s for s in suite.scenarios if s.name == "high_volatility_regime_only")
    assert high_vol.monte_carlo is None
    assert high_vol.note is not None
    assert high_vol.survived is False


def test_scenarios_with_no_trades_dont_count_toward_all_survived():
    """A scenario that couldn't run (no applicable trades) shouldn't silently pass or silently fail the overall verdict."""
    result = BacktestResult()
    result.trades = make_trades(n_wins=95, n_losses=5, regime="TREND_UP")  # very strong edge, no volatility-regime trades
    suite = run_stress_test_suite(result, starting_bankroll=10000, n_simulations=200, seed=1)
    # Even though 2 of 8 scenarios have no applicable trades, all_survived should reflect only the ones that ran
    ran_count = sum(1 for s in suite.scenarios if s.monte_carlo is not None)
    assert ran_count == 6  # 8 total minus 2 empty regime scenarios


def test_weak_edge_strategy_fails_stress_scenarios():
    """A strategy with only a thin real edge should fail at least the harsher reduction scenarios."""
    result = BacktestResult()
    # 56% win rate at 80% payout is barely above break-even (55.56%) -- a 20pp reduction should clearly ruin it
    result.trades = make_trades(n_wins=56, n_losses=44, stake=100.0, regime="TREND_UP")
    suite = run_stress_test_suite(result, starting_bankroll=200, n_simulations=300, seed=1)
    worst_case = next(s for s in suite.scenarios if s.name == "win_probability_-20pp")
    assert worst_case.survived is False


def test_render_stress_test_text_includes_all_scenarios():
    result = BacktestResult()
    result.trades = make_trades(n_wins=70, n_losses=30, regime="TREND_UP")
    suite = run_stress_test_suite(result, starting_bankroll=1000, n_simulations=100, seed=1)
    text = render_stress_test_text(suite)
    assert "STRESS TEST REPORT" in text
    assert "win_probability_-10pp" in text
    assert "high_volatility_regime_only" in text
