import numpy as np
import pytest

from app.backtest.walk_forward import make_walk_forward_windows, run_walk_forward, WalkForwardReport
from app.backtest.monte_carlo import run_monte_carlo
from app.markets.context import MarketType
from app.strategies.baselines import RandomStrategy


# --- make_walk_forward_windows ---

def test_windows_are_chronological_and_non_overlapping_across_windows():
    windows = make_walk_forward_windows(n_candles=1000, n_windows=4, train_fraction=0.7)
    assert len(windows) == 4
    # each window's own train comes before its own test
    for train_range, test_range in windows:
        assert train_range[0] < train_range[1] <= test_range[0] < test_range[1]
    # windows themselves are chronological and don't overlap each other
    for i in range(len(windows) - 1):
        assert windows[i][1][1] <= windows[i + 1][0][0]


def test_windows_cover_the_full_range():
    windows = make_walk_forward_windows(n_candles=1000, n_windows=4)
    assert windows[0][0][0] == 0
    assert windows[-1][1][1] == 1000


def test_windows_rejects_too_few_candles():
    with pytest.raises(ValueError):
        make_walk_forward_windows(n_candles=3, n_windows=10)


# --- run_walk_forward ---

def test_run_walk_forward_produces_one_result_per_window():
    rng = np.random.default_rng(1)
    n = 600
    closes = list(1.0 + np.cumsum(rng.normal(0, 0.002, n)))
    timestamps = [1_700_000_000 + i * 60 for i in range(n)]
    highs = [c + 0.001 for c in closes]
    lows = [c - 0.001 for c in closes]
    opens = closes

    report = run_walk_forward(
        symbol="frxEURUSD", market_type=MarketType.FOREX, duration_s=60,
        timestamps=timestamps, opens=opens, highs=highs, lows=lows, closes=closes,
        strategy_factory=lambda train_range: RandomStrategy(seed=1),
        duration_candles=1, payout_ratio=0.8, n_windows=3, warmup=60,
    )
    assert len(report.windows) == 3
    assert len(report.performance_per_window) == 3


def test_stability_score_is_high_for_consistent_performance():
    report = WalkForwardReport()

    class FakeResult:
        def __init__(self, ev):
            self._ev = ev
        @property
        def realized_ev_per_stake(self):
            return self._ev

    from app.backtest.walk_forward import WalkForwardWindow
    report.windows = [
        WalkForwardWindow(0, (0, 10), (10, 20), FakeResult(0.05)),
        WalkForwardWindow(1, (20, 30), (30, 40), FakeResult(0.051)),
        WalkForwardWindow(2, (40, 50), (50, 60), FakeResult(0.049)),
    ]
    assert report.stability_score > 0.9


def test_stability_score_is_low_for_inconsistent_performance():
    report = WalkForwardReport()

    class FakeResult:
        def __init__(self, ev):
            self._ev = ev
        @property
        def realized_ev_per_stake(self):
            return self._ev

    from app.backtest.walk_forward import WalkForwardWindow
    report.windows = [
        WalkForwardWindow(0, (0, 10), (10, 20), FakeResult(0.5)),
        WalkForwardWindow(1, (20, 30), (30, 40), FakeResult(-0.5)),
        WalkForwardWindow(2, (40, 50), (50, 60), FakeResult(0.3)),
    ]
    assert report.stability_score < 0.5


# --- monte_carlo ---

def test_monte_carlo_with_all_positive_trades_has_zero_ruin():
    trades = [1.0, 2.0, 0.5, 1.5]
    report = run_monte_carlo(trades, starting_bankroll=100.0, n_simulations=200, seed=1)
    assert report.probability_of_ruin == pytest.approx(0.0)
    assert report.expected_ending_bankroll > 100.0


def test_monte_carlo_detects_ruin_risk_with_catastrophic_losses():
    trades = [1.0] * 9 + [-1000.0]  # rare catastrophic loss
    report = run_monte_carlo(trades, starting_bankroll=50.0, n_simulations=500, seed=2)
    assert report.probability_of_ruin > 0.0


def test_monte_carlo_is_deterministic_given_a_seed():
    trades = [1.0, -1.0, 2.0, -0.5]
    r1 = run_monte_carlo(trades, starting_bankroll=100.0, n_simulations=100, seed=99)
    r2 = run_monte_carlo(trades, starting_bankroll=100.0, n_simulations=100, seed=99)
    assert r1.max_drawdowns == r2.max_drawdowns


def test_monte_carlo_empty_trades_returns_empty_report():
    report = run_monte_carlo([], starting_bankroll=100.0, n_simulations=50)
    assert report.probability_of_ruin == 0.0
    assert report.expected_ending_bankroll == 100.0


def test_drawdown_percentile_monotonic():
    trades = [1.0, -2.0, 3.0, -1.0, 0.5]
    report = run_monte_carlo(trades, starting_bankroll=100.0, n_simulations=300, seed=3)
    p50 = report.drawdown_percentile(50)
    p95 = report.drawdown_percentile(95)
    assert p95 >= p50
