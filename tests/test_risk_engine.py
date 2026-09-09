import pytest

from app.risk.stake_sizing import calculate_stake
from app.risk.governor import RiskGovernor, RiskState
from app.risk.exposure import ExposureManager, currency_exposure_for_trade, parse_forex_symbol
from app.risk.overtrading import CooldownTracker, DuplicateSignalGuard, RateLimiter
from app.strategies.base import Direction


# --- stake_sizing.py ---

def test_calculate_stake_basic():
    assert calculate_stake(balance=1000, risk_per_trade=0.01, max_stake=100) == pytest.approx(10.0)


def test_calculate_stake_clamped_to_max():
    assert calculate_stake(balance=100000, risk_per_trade=0.01, max_stake=50) == pytest.approx(50.0)


def test_calculate_stake_zero_or_negative_balance_is_zero():
    assert calculate_stake(balance=0, risk_per_trade=0.01, max_stake=100) == 0.0
    assert calculate_stake(balance=-50, risk_per_trade=0.01, max_stake=100) == 0.0


def test_calculate_stake_rejects_non_positive_risk():
    with pytest.raises(ValueError):
        calculate_stake(balance=1000, risk_per_trade=0.0, max_stake=100)


# --- governor.py ---

def test_governor_starts_normal():
    gov = RiskGovernor(starting_balance=1000, max_daily_loss=100, max_drawdown=0.2, max_consecutive_losses=5)
    assert gov.state == RiskState.NORMAL
    assert gov.risk_multiplier == pytest.approx(1.0)
    assert gov.can_trade() is True


def test_governor_never_increases_risk_after_a_loss():
    """The core invariant: risk_multiplier after any loss must be <= what it was before."""
    gov = RiskGovernor(starting_balance=1000, max_daily_loss=1000, max_drawdown=0.9, max_consecutive_losses=100)
    multiplier_before = gov.risk_multiplier
    for _ in range(10):
        gov.record_trade_result(-10.0)
        assert gov.risk_multiplier <= multiplier_before
        multiplier_before = gov.risk_multiplier


def test_governor_drawdown_triggers_emergency_stop():
    gov = RiskGovernor(starting_balance=1000, max_daily_loss=10000, max_drawdown=0.1, max_consecutive_losses=100)
    gov.record_trade_result(-150.0)  # 15% drawdown from peak 1000 -> exceeds 10% limit
    assert gov.is_emergency_stopped is True
    assert gov.state == RiskState.EMERGENCY_STOP
    assert gov.risk_multiplier == 0.0
    assert gov.can_trade() is False


def test_governor_emergency_stop_persists_until_manual_reset():
    gov = RiskGovernor(starting_balance=1000, max_daily_loss=10000, max_drawdown=0.1, max_consecutive_losses=100)
    gov.record_trade_result(-150.0)
    assert gov.is_emergency_stopped is True

    # A subsequent WIN must not clear it on its own -- only manual_reset() can.
    gov.record_trade_result(500.0)
    assert gov.is_emergency_stopped is True

    gov.manual_reset()
    assert gov.is_emergency_stopped is False


def test_governor_consecutive_losses_reset_on_a_win():
    gov = RiskGovernor(starting_balance=1000, max_daily_loss=10000, max_drawdown=0.9, max_consecutive_losses=5)
    gov.record_trade_result(-1.0)
    gov.record_trade_result(-1.0)
    assert gov.consecutive_losses == 2
    gov.record_trade_result(5.0)
    assert gov.consecutive_losses == 0


def test_governor_start_new_day_resets_daily_loss_but_not_drawdown():
    gov = RiskGovernor(starting_balance=1000, max_daily_loss=50, max_drawdown=0.9, max_consecutive_losses=100)
    gov.record_trade_result(-40.0)
    assert gov.daily_loss == pytest.approx(40.0)
    peak_before = gov.peak_balance

    gov.start_new_day()

    assert gov.daily_loss == pytest.approx(0.0)   # daily reference point moved to today's balance
    assert gov.peak_balance == peak_before         # drawdown tracking is NOT a calendar-day concept
    assert gov.drawdown_fraction == pytest.approx((peak_before - gov.current_balance) / peak_before)


def test_governor_daily_loss_limit_logs_event():
    gov = RiskGovernor(starting_balance=1000, max_daily_loss=50, max_drawdown=0.9, max_consecutive_losses=100)
    gov.record_trade_result(-60.0)
    assert any(e.event_type == "DAILY_LIMIT_REACHED" for e in gov.events)


# --- exposure.py ---

def test_parse_forex_symbol():
    assert parse_forex_symbol("frxEURUSD") == ("EUR", "USD")
    assert parse_forex_symbol("EURUSD") == ("EUR", "USD")
    assert parse_forex_symbol("1HZ100V") is None


def test_currency_exposure_for_call_and_put_are_opposite():
    call_exp = currency_exposure_for_trade("frxEURUSD", Direction.CALL)
    put_exp = currency_exposure_for_trade("frxEURUSD", Direction.PUT)
    assert call_exp == {"EUR": 1.0, "USD": -1.0}
    assert put_exp == {"EUR": -1.0, "USD": 1.0}


def test_currency_exposure_empty_for_non_forex_symbol():
    assert currency_exposure_for_trade("1HZ100V", Direction.CALL) == {}


def test_exposure_manager_blocks_correlated_usd_exposure():
    manager = ExposureManager(max_exposure_per_currency=1.5)
    manager.open_position("pos1", "frxEURUSD", Direction.CALL)   # USD: -1
    # A second CALL on GBPUSD would push USD exposure to -2, exceeding 1.5
    assert manager.would_exceed_limit("frxGBPUSD", Direction.CALL) is True


def test_exposure_manager_allows_uncorrelated_trade():
    manager = ExposureManager(max_exposure_per_currency=1.5)
    manager.open_position("pos1", "frxEURUSD", Direction.CALL)   # EUR: +1, USD: -1
    # A CALL on USDJPY: USD +1, JPY -1 -- offsets the existing short-USD exposure back toward 0
    assert manager.would_exceed_limit("frxUSDJPY", Direction.CALL) is False


def test_exposure_manager_close_position_frees_up_exposure():
    manager = ExposureManager(max_exposure_per_currency=1.0)
    manager.open_position("pos1", "frxEURUSD", Direction.CALL)
    assert manager.would_exceed_limit("frxGBPUSD", Direction.CALL) is True
    manager.close_position("pos1")
    assert manager.would_exceed_limit("frxGBPUSD", Direction.CALL) is False


# --- overtrading.py ---

def test_cooldown_tracker_blocks_within_window():
    tracker = CooldownTracker(cooldown_seconds=60)
    tracker.record_trade("frxEURUSD", now=1000)
    assert tracker.is_satisfied("frxEURUSD", now=1030) is False
    assert tracker.is_satisfied("frxEURUSD", now=1061) is True


def test_cooldown_tracker_is_per_key():
    tracker = CooldownTracker(cooldown_seconds=60)
    tracker.record_trade("frxEURUSD", now=1000)
    assert tracker.is_satisfied("frxGBPUSD", now=1001) is True  # different symbol, unaffected


def test_rate_limiter_blocks_beyond_hourly_limit():
    limiter = RateLimiter(max_trades_per_hour=2, max_trades_per_day=100)
    limiter.record_trade(now=0)
    limiter.record_trade(now=10)
    assert limiter.is_within_limits(now=20) is False
    assert limiter.is_within_limits(now=4000) is True  # more than an hour later


def test_rate_limiter_blocks_beyond_daily_limit():
    limiter = RateLimiter(max_trades_per_hour=100, max_trades_per_day=2)
    limiter.record_trade(now=0)
    limiter.record_trade(now=5000)
    assert limiter.is_within_limits(now=6000) is False


def test_duplicate_signal_guard():
    guard = DuplicateSignalGuard()
    assert guard.has_already_traded("sig-1") is False
    guard.mark_traded("sig-1")
    assert guard.has_already_traded("sig-1") is True
    assert guard.has_already_traded("sig-2") is False
