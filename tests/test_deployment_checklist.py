from app.analytics.deployment_checklist import (
    CheckStatus,
    generate_deployment_checklist,
    render_deployment_checklist_text,
)
from app.analytics.model_comparison_report import ModelComparisonReport
from app.analytics.real_data_report import RealDataReport
from app.backtest.statistics import StatisticalEvaluation
from app.backtest.stress_testing import StressScenarioResult, StressTestSuite
from app.backtest.walk_forward import WalkForwardReport
from app.config.settings import Settings
from app.models.baseline_comparison import ModelApprovalReport


def make_settings(**overrides):
    s = Settings()
    for k, v in overrides.items():
        setattr(s, k, v)
    return s


def make_backtest_result(ev=0.05, drawdown=10.0):
    # A drawdown property is computed from trades' equity curve; easiest to
    # monkeypatch it directly via a tiny subclass-free trick: set trades such
    # that max_drawdown naturally comes out to the desired value isn't
    # trivial, so we just check the property directly in the report builder
    # via a stand-in object with the same attributes the checklist reads.
    class FakeResult:
        realized_ev_per_stake = ev
        max_drawdown = drawdown
    return FakeResult()


def make_significance(meets_minimum_sample=True, n_trades=200, min_sample_size=100):
    return StatisticalEvaluation(
        n_trades=n_trades, win_rate=0.6, wilson_ci_low=0.55, wilson_ci_high=0.65,
        break_even_probability=0.5556, z_score=3.0, p_value=0.001, alpha=0.05,
        is_significant=True, min_sample_size=min_sample_size, meets_minimum_sample=meets_minimum_sample,
    )


def make_real_data_report(ev=0.05, drawdown=10.0, meets_minimum_sample=True, stability=0.8, strategy_id="simple_trend"):
    return _make_real_data_report_with_stability(ev, drawdown, meets_minimum_sample, stability, strategy_id)


def _make_real_data_report_with_stability(ev, drawdown, meets_minimum_sample, stability, strategy_id):
    report = RealDataReport(
        symbol="frxEURUSD", duration_s=60, n_candles=1000, strategy_id=strategy_id,
        backtest_result=make_backtest_result(ev, drawdown),
        walk_forward_report=WalkForwardReport(),
        significance=make_significance(meets_minimum_sample=meets_minimum_sample),
    )
    # stability_score is a computed property on WalkForwardReport based on .windows;
    # simplest path here is a stand-in with the same property surface.
    class FakeWFReport:
        def __init__(self, score):
            self._score = score
            self.windows = [1, 2, 3, 4] if score is not None else []
        @property
        def stability_score(self):
            return self._score
    report.walk_forward_report = FakeWFReport(stability)
    return report


def make_model_comparison_report(oos_ev=0.03, calibration_error=0.05, approved=True):
    candidate = make_backtest_result(ev=oos_ev, drawdown=5.0)
    candidate.total_trades = 150
    return ModelComparisonReport(
        symbol="frxEURUSD", n_candles=1000, n_train_examples=500,
        candidate_result=candidate,
        approval=_make_approval(approved),
        calibration_error=calibration_error,
    )


def _make_approval(approved):
    approval = ModelApprovalReport()
    approval.candidate_is_statistically_reliable = True
    if not approved:
        approval.reasons.append("did not beat baseline 'simple_trend'")
        approval.comparisons = []
    return approval


def make_stress_suite(all_survived=True):
    suite = StressTestSuite()
    from app.backtest.monte_carlo import MonteCarloReport
    mc = MonteCarloReport(n_simulations=100, starting_bankroll=1000, max_drawdowns=[10.0], ending_bankrolls=[1050.0], ruin_count=0 if all_survived else 50)
    suite.scenarios.append(StressScenarioResult(name="test_scenario", description="d", n_trades=100, monte_carlo=mc, survived=all_survived))
    return suite


BASE_SETTINGS = make_settings(
    risk_per_trade=0.01, max_stake=10.0, max_daily_loss=20.0, max_drawdown=0.15, max_consecutive_losses=5, max_latency_ms=800.0,
)


def test_all_automated_items_pass_produces_approved_only_if_truly_everything_passes():
    """Even with every automatable item passing, the 3 NOT_AUTOMATED items must block overall approval."""
    real = make_real_data_report()
    model = make_model_comparison_report()
    stress = make_stress_suite()
    checklist = generate_deployment_checklist(real, model, stress, BASE_SETTINGS, starting_bankroll=1000.0)
    assert checklist.all_passed is False  # NOT_AUTOMATED items (7, 9, 10) always block
    statuses = {i.name: i.status for i in checklist.items}
    assert statuses["demo_forward_test_passed"] == CheckStatus.NOT_AUTOMATED
    assert statuses["api_reliability_acceptable"] == CheckStatus.NOT_AUTOMATED
    assert statuses["execution_latency_acceptable"] == CheckStatus.NOT_AUTOMATED


def test_insufficient_sample_fails_that_item():
    real = make_real_data_report(meets_minimum_sample=False)
    checklist = generate_deployment_checklist(real, None, None, BASE_SETTINGS, starting_bankroll=1000.0)
    item = next(i for i in checklist.items if i.name == "sufficient_historical_sample")
    assert item.status == CheckStatus.FAIL


def test_negative_ev_fails_that_item():
    real = make_real_data_report(ev=-0.02)
    checklist = generate_deployment_checklist(real, None, None, BASE_SETTINGS, starting_bankroll=1000.0)
    item = next(i for i in checklist.items if i.name == "positive_expected_value")
    assert item.status == CheckStatus.FAIL


def test_excessive_drawdown_fails_that_item():
    real = make_real_data_report(drawdown=500.0)  # 50% of 1000 bankroll, limit is 15%
    checklist = generate_deployment_checklist(real, None, None, BASE_SETTINGS, starting_bankroll=1000.0)
    item = next(i for i in checklist.items if i.name == "acceptable_drawdown")
    assert item.status == CheckStatus.FAIL


def test_acceptable_drawdown_passes_within_limit():
    real = make_real_data_report(drawdown=50.0)  # 5% of 1000, limit 15%
    checklist = generate_deployment_checklist(real, None, None, BASE_SETTINGS, starting_bankroll=1000.0)
    item = next(i for i in checklist.items if i.name == "acceptable_drawdown")
    assert item.status == CheckStatus.PASS


def test_no_model_comparison_report_marks_two_items_not_automated():
    real = make_real_data_report()
    checklist = generate_deployment_checklist(real, None, None, BASE_SETTINGS, starting_bankroll=1000.0)
    statuses = {i.name: i.status for i in checklist.items}
    assert statuses["profitable_out_of_sample"] == CheckStatus.NOT_AUTOMATED
    assert statuses["probability_calibration_acceptable"] == CheckStatus.NOT_AUTOMATED


def test_poor_calibration_fails_that_item():
    real = make_real_data_report()
    model = make_model_comparison_report(calibration_error=0.5)  # way above default 0.15 limit
    checklist = generate_deployment_checklist(real, model, None, BASE_SETTINGS, starting_bankroll=1000.0)
    item = next(i for i in checklist.items if i.name == "probability_calibration_acceptable")
    assert item.status == CheckStatus.FAIL


def test_low_walk_forward_stability_fails_that_item():
    real = make_real_data_report(stability=0.1)  # below default 0.5 threshold
    checklist = generate_deployment_checklist(real, None, None, BASE_SETTINGS, starting_bankroll=1000.0)
    item = next(i for i in checklist.items if i.name == "walk_forward_validation_passed")
    assert item.status == CheckStatus.FAIL


def test_stress_suite_not_survived_fails_that_item():
    real = make_real_data_report()
    stress = make_stress_suite(all_survived=False)
    checklist = generate_deployment_checklist(real, None, stress, BASE_SETTINGS, starting_bankroll=1000.0)
    item = next(i for i in checklist.items if i.name == "monte_carlo_risk_acceptable")
    assert item.status == CheckStatus.FAIL


def test_insane_risk_config_fails_risk_limits_item():
    bad_settings = make_settings(risk_per_trade=0.5, max_stake=10.0, max_daily_loss=20.0, max_drawdown=0.15, max_consecutive_losses=5)  # 50% risk per trade is not sane
    real = make_real_data_report()
    checklist = generate_deployment_checklist(real, None, None, bad_settings, starting_bankroll=1000.0)
    item = next(i for i in checklist.items if i.name == "risk_limits_configured")
    assert item.status == CheckStatus.FAIL


def test_emergency_stop_and_duplicate_protection_always_pass_as_static_facts():
    real = make_real_data_report()
    checklist = generate_deployment_checklist(real, None, None, BASE_SETTINGS, starting_bankroll=1000.0)
    statuses = {i.name: i.status for i in checklist.items}
    assert statuses["emergency_stop_tested"] == CheckStatus.PASS
    assert statuses["duplicate_trade_protection_tested"] == CheckStatus.PASS


def test_render_text_shows_not_approved_when_any_item_fails():
    real = make_real_data_report(ev=-0.01)
    checklist = generate_deployment_checklist(real, None, None, BASE_SETTINGS, starting_bankroll=1000.0)
    text = render_deployment_checklist_text(checklist)
    assert "NOT APPROVED" in text
    assert "positive_expected_value" in text


def test_all_items_present_regardless_of_pass_fail():
    real = make_real_data_report()
    checklist = generate_deployment_checklist(real, None, None, BASE_SETTINGS, starting_bankroll=1000.0)
    names = {i.name for i in checklist.items}
    expected = {
        "sufficient_historical_sample", "positive_expected_value", "acceptable_drawdown",
        "profitable_out_of_sample", "walk_forward_validation_passed", "monte_carlo_risk_acceptable",
        "demo_forward_test_passed", "probability_calibration_acceptable", "api_reliability_acceptable",
        "execution_latency_acceptable", "risk_limits_configured", "emergency_stop_tested",
        "duplicate_trade_protection_tested", "model_version_locked",
    }
    assert names == expected


def test_lifetime_stats_automates_demo_forward_test_when_provided():
    from app.monitoring.trade_log import LifetimeStats

    real = make_real_data_report()
    good_stats = LifetimeStats(trade_count=150, wins=90, win_rate=0.6, total_pnl=250.0, latencies_ms=[100.0] * 150)
    checklist = generate_deployment_checklist(real, None, None, BASE_SETTINGS, starting_bankroll=1000.0, lifetime_stats=good_stats)
    item = next(i for i in checklist.items if i.name == "demo_forward_test_passed")
    assert item.status == CheckStatus.PASS


def test_lifetime_stats_fails_demo_forward_test_with_too_few_trades():
    from app.monitoring.trade_log import LifetimeStats

    real = make_real_data_report()
    thin_stats = LifetimeStats(trade_count=10, wins=8, win_rate=0.8, total_pnl=50.0, latencies_ms=[100.0] * 10)
    checklist = generate_deployment_checklist(real, None, None, BASE_SETTINGS, starting_bankroll=1000.0, lifetime_stats=thin_stats)
    item = next(i for i in checklist.items if i.name == "demo_forward_test_passed")
    assert item.status == CheckStatus.FAIL


def test_lifetime_stats_fails_demo_forward_test_with_negative_pnl():
    from app.monitoring.trade_log import LifetimeStats

    real = make_real_data_report()
    losing_stats = LifetimeStats(trade_count=150, wins=50, win_rate=0.33, total_pnl=-100.0, latencies_ms=[100.0] * 150)
    checklist = generate_deployment_checklist(real, None, None, BASE_SETTINGS, starting_bankroll=1000.0, lifetime_stats=losing_stats)
    item = next(i for i in checklist.items if i.name == "demo_forward_test_passed")
    assert item.status == CheckStatus.FAIL


def test_lifetime_stats_automates_latency_check():
    from app.monitoring.trade_log import LifetimeStats

    real = make_real_data_report()
    fast_stats = LifetimeStats(trade_count=150, wins=90, win_rate=0.6, total_pnl=100.0, latencies_ms=[100.0] * 150)
    checklist = generate_deployment_checklist(real, None, None, BASE_SETTINGS, starting_bankroll=1000.0, lifetime_stats=fast_stats)
    item = next(i for i in checklist.items if i.name == "execution_latency_acceptable")
    assert item.status == CheckStatus.PASS


def test_lifetime_stats_fails_latency_check_when_p95_exceeds_limit():
    from app.monitoring.trade_log import LifetimeStats

    real = make_real_data_report()
    # BASE_SETTINGS has max_latency_ms=800; make p95 clearly exceed that
    slow_latencies = [100.0] * 90 + [2000.0] * 10  # p95 will land in the 2000ms tail
    slow_stats = LifetimeStats(trade_count=100, wins=60, win_rate=0.6, total_pnl=100.0, latencies_ms=slow_latencies)
    checklist = generate_deployment_checklist(real, None, None, BASE_SETTINGS, starting_bankroll=1000.0, lifetime_stats=slow_stats)
    item = next(i for i in checklist.items if i.name == "execution_latency_acceptable")
    assert item.status == CheckStatus.FAIL


def test_lifetime_stats_with_no_latency_samples_stays_not_automated():
    from app.monitoring.trade_log import LifetimeStats

    real = make_real_data_report()
    no_latency_stats = LifetimeStats(trade_count=150, wins=90, win_rate=0.6, total_pnl=100.0, latencies_ms=[])
    checklist = generate_deployment_checklist(real, None, None, BASE_SETTINGS, starting_bankroll=1000.0, lifetime_stats=no_latency_stats)
    item = next(i for i in checklist.items if i.name == "execution_latency_acceptable")
    assert item.status == CheckStatus.NOT_AUTOMATED


def test_reliability_stats_passes_with_few_errors():
    from app.monitoring.system_events_log import ReliabilityStats

    real = make_real_data_report()
    good_reliability = ReliabilityStats(lookback_hours=24, error_count=1, total_events=50)
    checklist = generate_deployment_checklist(real, None, None, BASE_SETTINGS, starting_bankroll=1000.0, reliability_stats=good_reliability)
    item = next(i for i in checklist.items if i.name == "api_reliability_acceptable")
    assert item.status == CheckStatus.PASS


def test_reliability_stats_fails_with_too_many_errors():
    from app.monitoring.system_events_log import ReliabilityStats

    real = make_real_data_report()
    bad_reliability = ReliabilityStats(lookback_hours=24, error_count=10, total_events=50)
    checklist = generate_deployment_checklist(real, None, None, BASE_SETTINGS, starting_bankroll=1000.0, reliability_stats=bad_reliability)
    item = next(i for i in checklist.items if i.name == "api_reliability_acceptable")
    assert item.status == CheckStatus.FAIL


def test_reliability_stats_none_stays_not_automated():
    real = make_real_data_report()
    checklist = generate_deployment_checklist(real, None, None, BASE_SETTINGS, starting_bankroll=1000.0)
    item = next(i for i in checklist.items if i.name == "api_reliability_acceptable")
    assert item.status == CheckStatus.NOT_AUTOMATED


def test_all_gaps_closed_together_can_reach_full_pass():
    """With every optional report supplied and genuinely good numbers, all_passed can finally be True."""
    from app.monitoring.trade_log import LifetimeStats
    from app.monitoring.system_events_log import ReliabilityStats

    real = make_real_data_report(ev=0.1, drawdown=10.0, meets_minimum_sample=True, stability=0.9)
    model = make_model_comparison_report(oos_ev=0.05, calibration_error=0.02)
    stress = make_stress_suite(all_survived=True)
    lifetime = LifetimeStats(trade_count=200, wins=130, win_rate=0.65, total_pnl=500.0, latencies_ms=[100.0] * 200)
    reliability = ReliabilityStats(lookback_hours=24, error_count=0, total_events=200)

    checklist = generate_deployment_checklist(
        real, model, stress, BASE_SETTINGS, starting_bankroll=1000.0,
        lifetime_stats=lifetime, reliability_stats=reliability,
    )
    assert checklist.all_passed is True
