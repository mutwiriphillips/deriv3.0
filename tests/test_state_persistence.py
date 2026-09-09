import sqlite3

import pytest

from app.monitoring.session_stats import SessionStats
from app.monitoring.state_persistence import (
    load_daily_state,
    restore_state_if_present,
    save_daily_state,
)
from app.risk.governor import RiskGovernor


def make_db(tmp_path):
    db_path = str(tmp_path / "test.db")
    conn = sqlite3.connect(db_path)
    with open("app/storage/schema.sql") as f:
        conn.executescript(f.read())
    conn.close()
    return db_path


def test_load_daily_state_missing_day_returns_none(tmp_path):
    db_path = make_db(tmp_path)
    assert load_daily_state(db_path, "2026-01-01") is None


def test_save_and_load_round_trip(tmp_path):
    db_path = make_db(tmp_path)
    gov = RiskGovernor(starting_balance=1000, max_daily_loss=100, max_drawdown=0.2, max_consecutive_losses=5)
    gov.record_trade_result(-20.0)
    stats = SessionStats()
    stats.record_settlement("LOSS", -20.0)

    save_daily_state(db_path, "2026-03-15", gov, stats)
    snapshot = load_daily_state(db_path, "2026-03-15")

    assert snapshot.current_balance == pytest.approx(980.0)
    assert snapshot.consecutive_losses == 1
    assert snapshot.trades_today == 1
    assert snapshot.losses == 1
    assert snapshot.session_pnl == pytest.approx(-20.0)
    assert snapshot.emergency_stopped is False


def test_save_twice_same_day_updates_in_place_not_duplicates(tmp_path):
    db_path = make_db(tmp_path)
    gov = RiskGovernor(starting_balance=1000, max_daily_loss=1000, max_drawdown=0.9, max_consecutive_losses=100)
    stats = SessionStats()

    save_daily_state(db_path, "2026-03-15", gov, stats)
    gov.record_trade_result(50.0)
    stats.record_settlement("WIN", 50.0)
    save_daily_state(db_path, "2026-03-15", gov, stats)

    conn = sqlite3.connect(db_path)
    count = conn.execute("SELECT COUNT(*) FROM daily_state WHERE trade_date=?", ("2026-03-15",)).fetchone()[0]
    conn.close()
    assert count == 1

    snapshot = load_daily_state(db_path, "2026-03-15")
    assert snapshot.current_balance == pytest.approx(1050.0)
    assert snapshot.trades_today == 1


def test_restore_state_if_present_applies_snapshot_onto_fresh_instances(tmp_path):
    db_path = make_db(tmp_path)
    original_gov = RiskGovernor(starting_balance=1000, max_daily_loss=1000, max_drawdown=0.9, max_consecutive_losses=100)
    original_gov.record_trade_result(-30.0)
    original_gov.record_trade_result(-10.0)
    original_stats = SessionStats()
    original_stats.record_settlement("LOSS", -30.0)
    original_stats.record_settlement("LOSS", -10.0)
    save_daily_state(db_path, "2026-03-15", original_gov, original_stats)

    # Simulate a restart: brand new instances, config from settings as normal
    fresh_gov = RiskGovernor(starting_balance=1000, max_daily_loss=1000, max_drawdown=0.9, max_consecutive_losses=100)
    fresh_stats = SessionStats()

    restored = restore_state_if_present(db_path, "2026-03-15", fresh_gov, fresh_stats)

    assert restored is True
    assert fresh_gov.current_balance == pytest.approx(960.0)
    assert fresh_gov.consecutive_losses == 2
    assert fresh_stats.trades_today == 2
    assert fresh_stats.current_losing_streak == 2


def test_restore_state_if_present_returns_false_for_a_new_day():
    """The core daily-reset mechanism: no row for today means a genuinely fresh start, not an error."""
    import tempfile
    import os
    db_path = tempfile.mktemp(suffix=".db")
    conn = sqlite3.connect(db_path)
    with open("app/storage/schema.sql") as f:
        conn.executescript(f.read())
    conn.close()

    gov = RiskGovernor(starting_balance=1000, max_daily_loss=1000, max_drawdown=0.9, max_consecutive_losses=100)
    stats = SessionStats()
    # Save state for a past day, then ask for a different (new) day
    save_daily_state(db_path, "2026-03-14", gov, stats)
    restored = restore_state_if_present(db_path, "2026-03-15", gov, stats)

    assert restored is False
    os.remove(db_path)


def test_restore_preserves_emergency_stop_across_restart(tmp_path):
    db_path = make_db(tmp_path)
    original_gov = RiskGovernor(starting_balance=1000, max_daily_loss=10000, max_drawdown=0.1, max_consecutive_losses=100)
    original_gov.record_trade_result(-150.0)  # triggers emergency stop
    assert original_gov.is_emergency_stopped is True
    save_daily_state(db_path, "2026-03-15", original_gov, SessionStats())

    fresh_gov = RiskGovernor(starting_balance=1000, max_daily_loss=10000, max_drawdown=0.1, max_consecutive_losses=100)
    restore_state_if_present(db_path, "2026-03-15", fresh_gov, SessionStats())

    # Without this restore, a restart would silently un-stop a bot that was
    # emergency-stopped for a real reason -- exactly the risk this closes.
    assert fresh_gov.is_emergency_stopped is True
