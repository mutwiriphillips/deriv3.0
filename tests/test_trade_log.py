import sqlite3

import pytest

from app.monitoring.trade_log import get_latest_trade_id, get_lifetime_stats, record_trade


def make_db(tmp_path):
    db_path = str(tmp_path / "test.db")
    conn = sqlite3.connect(db_path)
    with open("app/storage/schema.sql") as f:
        conn.executescript(f.read())
    conn.close()
    return db_path


def test_lifetime_stats_empty_log(tmp_path):
    db_path = make_db(tmp_path)
    stats = get_lifetime_stats(db_path)
    assert stats.trade_count == 0
    assert stats.win_rate is None
    assert stats.total_pnl == 0
    assert stats.latency_percentile(95) is None


def test_record_and_read_back_trades(tmp_path):
    db_path = make_db(tmp_path)
    record_trade(db_path, "frxEURUSD", "CALL", "WIN", 8.0, 10.0, 120.0)
    record_trade(db_path, "frxEURUSD", "PUT", "LOSS", -10.0, 10.0, 150.0)
    record_trade(db_path, "frxEURUSD", "CALL", "WIN", 8.0, 10.0, 90.0)

    stats = get_lifetime_stats(db_path)
    assert stats.trade_count == 3
    assert stats.wins == 2
    assert stats.win_rate == pytest.approx(2 / 3)
    assert stats.total_pnl == pytest.approx(6.0)
    assert sorted(stats.latencies_ms) == [90.0, 120.0, 150.0]


def test_stats_filtered_by_symbol(tmp_path):
    db_path = make_db(tmp_path)
    record_trade(db_path, "frxEURUSD", "CALL", "WIN", 8.0, 10.0, 100.0)
    record_trade(db_path, "frxGBPUSD", "CALL", "LOSS", -10.0, 10.0, 200.0)

    eurusd_stats = get_lifetime_stats(db_path, symbol="frxEURUSD")
    assert eurusd_stats.trade_count == 1
    assert eurusd_stats.wins == 1


def test_latency_percentile_computation(tmp_path):
    db_path = make_db(tmp_path)
    for lat in [100, 200, 300, 400, 500]:
        record_trade(db_path, "frxEURUSD", "CALL", "WIN", 1.0, 10.0, float(lat))
    stats = get_lifetime_stats(db_path)
    assert stats.latency_percentile(50) == 300.0
    assert stats.latency_percentile(100) == 500.0
    assert stats.latency_percentile(0) == 100.0


def test_record_trade_handles_none_latency(tmp_path):
    db_path = make_db(tmp_path)
    record_trade(db_path, "frxEURUSD", "CALL", "WIN", 8.0, 10.0, None)
    stats = get_lifetime_stats(db_path)
    assert stats.trade_count == 1
    assert stats.latencies_ms == []  # None latency excluded, not treated as 0


def test_get_latest_trade_id_empty_log(tmp_path):
    db_path = make_db(tmp_path)
    assert get_latest_trade_id(db_path) == 0


def test_get_latest_trade_id_increases_with_each_trade(tmp_path):
    db_path = make_db(tmp_path)
    record_trade(db_path, "frxEURUSD", "CALL", "WIN", 8.0, 10.0, 100.0)
    id1 = get_latest_trade_id(db_path)
    record_trade(db_path, "frxEURUSD", "CALL", "WIN", 8.0, 10.0, 100.0)
    id2 = get_latest_trade_id(db_path)
    assert id2 > id1


def test_since_id_filters_out_trades_before_the_snapshot(tmp_path):
    db_path = make_db(tmp_path)
    record_trade(db_path, "frxEURUSD", "CALL", "WIN", 8.0, 10.0, 100.0)   # before snapshot
    snapshot_id = get_latest_trade_id(db_path)
    record_trade(db_path, "frxEURUSD", "CALL", "WIN", 8.0, 10.0, 100.0)   # after snapshot
    record_trade(db_path, "frxEURUSD", "CALL", "LOSS", -10.0, 10.0, 100.0)  # after snapshot

    all_stats = get_lifetime_stats(db_path)
    since_stats = get_lifetime_stats(db_path, since_id=snapshot_id)

    assert all_stats.trade_count == 3
    assert since_stats.trade_count == 2
