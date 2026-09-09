import pytest

from app.monitoring.session_stats import SessionStats
from app.monitoring.dashboard import build_dashboard_state, render_dashboard_text
from app.risk.governor import RiskGovernor


# --- SessionStats ---

def test_session_stats_accumulates_wins_and_losses():
    stats = SessionStats()
    stats.record_settlement("WIN", 8.0)
    stats.record_settlement("LOSS", -10.0)
    stats.record_settlement("WIN", 8.0)
    assert stats.trades_today == 3
    assert stats.wins == 2
    assert stats.losses == 1
    assert stats.session_pnl == pytest.approx(6.0)
    assert stats.win_rate == pytest.approx(2 / 3)


def test_session_stats_losing_streak_tracking():
    stats = SessionStats()
    stats.record_settlement("LOSS", -1)
    stats.record_settlement("LOSS", -1)
    stats.record_settlement("LOSS", -1)
    assert stats.current_losing_streak == 3
    assert stats.longest_losing_streak_today == 3
    stats.record_settlement("WIN", 1)
    assert stats.current_losing_streak == 0
    assert stats.longest_losing_streak_today == 3  # longest today is preserved after the streak breaks


def test_session_stats_win_rate_none_with_no_trades():
    stats = SessionStats()
    assert stats.win_rate is None


def test_session_stats_reset_daily_clears_everything():
    stats = SessionStats()
    stats.record_settlement("WIN", 5.0)
    stats.reset_daily()
    assert stats.trades_today == 0
    assert stats.session_pnl == 0.0
    assert stats.win_rate is None


# --- dashboard.py ---

def test_dashboard_bot_status_normal():
    gov = RiskGovernor(starting_balance=1000, max_daily_loss=1000, max_drawdown=0.9, max_consecutive_losses=100)
    stats = SessionStats()
    state = build_dashboard_state(gov, stats, api_status="CONNECTED", active_strategy="baseline_trend", model_version="v1")
    assert state.bot_status == "RUNNING"
    assert state.account_balance == pytest.approx(1000)


def test_dashboard_bot_status_emergency_stop():
    gov = RiskGovernor(starting_balance=1000, max_daily_loss=10000, max_drawdown=0.1, max_consecutive_losses=100)
    gov.record_trade_result(-150.0)  # triggers emergency stop, see risk engine tests
    stats = SessionStats()
    state = build_dashboard_state(gov, stats, api_status="CONNECTED", active_strategy="x", model_version="v1")
    assert state.bot_status == "EMERGENCY_STOP"


def test_dashboard_pulls_session_stats_correctly():
    gov = RiskGovernor(starting_balance=1000, max_daily_loss=1000, max_drawdown=0.9, max_consecutive_losses=100)
    stats = SessionStats()
    stats.record_settlement("WIN", 8.0)
    stats.record_settlement("LOSS", -10.0)
    state = build_dashboard_state(gov, stats, api_status="CONNECTED", active_strategy="x", model_version="v1")
    assert state.trades_today == 2
    assert state.win_rate == pytest.approx(0.5)
    assert state.session_pnl == pytest.approx(-2.0)


def test_render_dashboard_text_includes_all_key_fields():
    gov = RiskGovernor(starting_balance=1000, max_daily_loss=1000, max_drawdown=0.9, max_consecutive_losses=100)
    stats = SessionStats()
    state = build_dashboard_state(
        gov, stats, api_status="CONNECTED", active_strategy="baseline_trend", model_version="v1",
        current_symbol="frxEURUSD", current_regime="TREND_UP",
        signal_probability=0.62, break_even_probability=0.5556, edge=0.0644, expected_value=0.8,
        stake=10.0, payout=18.0, latency_ms=120.5,
    )
    text = render_dashboard_text(state)
    for label in (
        "BOT STATUS", "API STATUS", "ACCOUNT BALANCE", "DAILY P/L", "SESSION P/L", "DRAWDOWN",
        "ACTIVE STRATEGY", "MODEL VERSION", "CURRENT SYMBOL", "CURRENT REGIME",
        "SIGNAL PROBABILITY", "BREAK-EVEN PROBABILITY", "EDGE", "EXPECTED VALUE",
        "STAKE", "PAYOUT", "LATENCY", "TRADES TODAY", "WIN RATE", "LOSING STREAK",
    ):
        assert label in text
    assert "frxEURUSD" in text
    assert "TREND_UP" in text


def test_render_dashboard_text_handles_missing_fields_gracefully():
    gov = RiskGovernor(starting_balance=1000, max_daily_loss=1000, max_drawdown=0.9, max_consecutive_losses=100)
    stats = SessionStats()
    state = build_dashboard_state(gov, stats, api_status="DISCONNECTED", active_strategy="x", model_version="v1")
    text = render_dashboard_text(state)
    assert "N/A" in text
    assert "CURRENT SYMBOL: N/A" in text


def test_render_dashboard_text_includes_no_trade_reason_only_when_present():
    gov = RiskGovernor(starting_balance=1000, max_daily_loss=1000, max_drawdown=0.9, max_consecutive_losses=100)
    stats = SessionStats()

    state_without = build_dashboard_state(gov, stats, api_status="CONNECTED", active_strategy="x", model_version="v1")
    assert "NO-TRADE REASON" not in render_dashboard_text(state_without)

    state_with = build_dashboard_state(
        gov, stats, api_status="CONNECTED", active_strategy="x", model_version="v1",
        no_trade_reason="NO_TRADE_LOW_EDGE",
    )
    assert "NO-TRADE REASON: NO_TRADE_LOW_EDGE" in render_dashboard_text(state_with)
