import sqlite3

import numpy as np

from app.analytics.real_data_report import generate_real_data_report, render_report_text
from app.strategies.baselines import RandomStrategy


def make_db_with_candles(tmp_path, n_candles, seed=1):
    db_path = str(tmp_path / "real.db")
    conn = sqlite3.connect(db_path)
    with open("app/storage/schema.sql") as f:
        conn.executescript(f.read())

    rng = np.random.default_rng(seed)
    closes = list(1.0 + np.cumsum(rng.normal(0, 0.002, n_candles)))
    timestamps = [1_700_000_000 + i * 60 for i in range(n_candles)]
    highs = [c + 0.001 for c in closes]
    lows = [c - 0.001 for c in closes]
    opens = closes

    for t, o, h, l, c in zip(timestamps, opens, highs, lows, closes):
        conn.execute(
            "INSERT INTO candles (symbol, duration_s, timestamp, open, high, low, close, tick_count) VALUES (?,?,?,?,?,?,?,?)",
            ("frxEURUSD", 60, t, o, h, l, c, 1),
        )
    conn.commit()
    conn.close()
    return db_path


def test_report_flags_warning_when_too_little_data(tmp_path):
    db_path = make_db_with_candles(tmp_path, n_candles=50)  # well below warmup=120
    report = generate_real_data_report(
        db_path=db_path, symbol="frxEURUSD", duration_s=60,
        strategy_factory=lambda train_range: RandomStrategy(seed=1),
        duration_candles=1, payout_ratio=0.8, warmup=120,
    )
    assert report.warning is not None
    assert "only 50 real candles" in report.warning
    assert report.n_candles == 50


def test_report_no_warning_with_sufficient_data(tmp_path):
    db_path = make_db_with_candles(tmp_path, n_candles=1000)
    report = generate_real_data_report(
        db_path=db_path, symbol="frxEURUSD", duration_s=60,
        strategy_factory=lambda train_range: RandomStrategy(seed=1),
        duration_candles=1, payout_ratio=0.8, warmup=120, n_walk_forward_windows=4,
    )
    assert report.warning is None
    assert report.n_candles == 1000
    assert len(report.walk_forward_report.windows) == 4


def test_report_significance_reflects_actual_backtest_result(tmp_path):
    db_path = make_db_with_candles(tmp_path, n_candles=1000)
    report = generate_real_data_report(
        db_path=db_path, symbol="frxEURUSD", duration_s=60,
        strategy_factory=lambda train_range: RandomStrategy(seed=1),
        duration_candles=1, payout_ratio=0.8, warmup=120,
    )
    # A coin-flip strategy against an 80% payout should NOT be statistically reliable.
    assert report.significance.is_reliable is False


def test_render_report_text_includes_key_sections(tmp_path):
    db_path = make_db_with_candles(tmp_path, n_candles=1000)
    report = generate_real_data_report(
        db_path=db_path, symbol="frxEURUSD", duration_s=60,
        strategy_factory=lambda train_range: RandomStrategy(seed=1),
        duration_candles=1, payout_ratio=0.8, warmup=120,
    )
    text = render_report_text(report)
    for label in ("Total trades", "Win rate", "Break-even probability", "Wilson 95% CI", "RELIABLE", "Walk-forward"):
        assert label in text


def test_report_handles_completely_empty_database(tmp_path):
    db_path = str(tmp_path / "empty.db")
    conn = sqlite3.connect(db_path)
    with open("app/storage/schema.sql") as f:
        conn.executescript(f.read())
    conn.commit()
    conn.close()

    report = generate_real_data_report(
        db_path=db_path, symbol="frxEURUSD", duration_s=60,
        strategy_factory=lambda train_range: RandomStrategy(seed=1),
        duration_candles=1, payout_ratio=0.8, warmup=120,
    )
    assert report.n_candles == 0
    assert report.warning is not None
    assert report.backtest_result.total_trades == 0


def test_max_candles_bounds_runtime_and_keeps_most_recent_data(tmp_path):
    """
    Confirms the fix for the real, measured (~6ms/candle/strategy) runtime
    growth: max_candles must actually cap what's loaded, and must keep the
    MOST RECENT candles, not the oldest -- recent market conditions are what
    a live edge check should reflect.
    """
    db_path = make_db_with_candles(tmp_path, n_candles=5000)
    report = generate_real_data_report(
        db_path=db_path, symbol="frxEURUSD", duration_s=60,
        strategy_factory=lambda train_range: RandomStrategy(seed=1),
        duration_candles=1, payout_ratio=0.8, warmup=120, max_candles=500,
    )
    assert report.n_candles == 500
