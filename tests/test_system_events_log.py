import sqlite3
from datetime import datetime, timedelta, timezone

from app.monitoring.system_events_log import get_reliability_stats


def make_db(tmp_path):
    db_path = str(tmp_path / "test.db")
    conn = sqlite3.connect(db_path)
    with open("app/storage/schema.sql") as f:
        conn.executescript(f.read())
    conn.close()
    return db_path


def insert_event(db_path, event_type, hours_ago=0.0):
    ts = (datetime.now(timezone.utc) - timedelta(hours=hours_ago)).isoformat()
    conn = sqlite3.connect(db_path)
    conn.execute("INSERT INTO system_events (timestamp, event_type, details) VALUES (?, ?, ?)", (ts, event_type, "{}"))
    conn.commit()
    conn.close()


def test_empty_log_has_zero_errors_and_zero_total():
    import tempfile, os
    db_path = tempfile.mktemp(suffix=".db")
    conn = sqlite3.connect(db_path)
    with open("app/storage/schema.sql") as f:
        conn.executescript(f.read())
    conn.close()
    stats = get_reliability_stats(db_path)
    assert stats.error_count == 0
    assert stats.total_events == 0
    assert stats.has_data is False
    os.remove(db_path)


def test_counts_error_events_within_window(tmp_path):
    db_path = make_db(tmp_path)
    insert_event(db_path, "API_DISCONNECTED", hours_ago=1)
    insert_event(db_path, "TRADE_EXECUTED", hours_ago=2)
    insert_event(db_path, "ABNORMAL_LATENCY", hours_ago=3)

    stats = get_reliability_stats(db_path, lookback_hours=24)
    assert stats.total_events == 3
    assert stats.error_count == 2  # only the two error types
    assert stats.has_data is True


def test_excludes_events_outside_lookback_window(tmp_path):
    db_path = make_db(tmp_path)
    insert_event(db_path, "API_DISCONNECTED", hours_ago=48)  # outside a 24h window
    insert_event(db_path, "TRADE_EXECUTED", hours_ago=1)

    stats = get_reliability_stats(db_path, lookback_hours=24)
    assert stats.total_events == 1
    assert stats.error_count == 0
