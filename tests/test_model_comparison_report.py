import sqlite3

import numpy as np

from app.analytics.model_comparison_report import (
    generate_model_comparison_report,
    render_model_comparison_text,
)


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

    for t, o, h, l in zip(timestamps, closes, highs, lows):
        conn.execute(
            "INSERT INTO candles (symbol, duration_s, timestamp, open, high, low, close, tick_count) VALUES (?,?,?,?,?,?,?,?)",
            ("frxEURUSD", 60, t, o, h, l, o, 1),
        )
    conn.commit()
    conn.close()
    return db_path


def test_report_warns_when_too_little_training_data(tmp_path):
    db_path = make_db_with_candles(tmp_path, n_candles=100)  # tiny -- far below what's needed
    report = generate_model_comparison_report(
        db_path=db_path, symbol="frxEURUSD", duration_s=60,
        duration_candles=1, payout_ratio=0.8, warmup=120,
    )
    assert report.candidate_result is None
    assert report.warning is not None
    assert "training examples" in report.warning


def test_report_runs_full_comparison_with_enough_data(tmp_path):
    db_path = make_db_with_candles(tmp_path, n_candles=600)
    report = generate_model_comparison_report(
        db_path=db_path, symbol="frxEURUSD", duration_s=60,
        duration_candles=1, payout_ratio=0.8, warmup=120,
    )
    assert report.candidate_result is not None
    assert set(report.baseline_results.keys()) == {"random", "simple_momentum", "simple_trend"}
    assert report.approval is not None
    # random noise data should NOT produce a statistically reliable edge
    assert report.approval.candidate_is_statistically_reliable is False


def test_report_flags_preliminary_when_candidate_trades_too_few_times(tmp_path):
    db_path = make_db_with_candles(tmp_path, n_candles=600)
    report = generate_model_comparison_report(
        db_path=db_path, symbol="frxEURUSD", duration_s=60,
        duration_candles=1, payout_ratio=0.8, warmup=120, min_sample_size=100000,  # impossible to reach
    )
    assert report.warning is not None
    assert "preliminary" in report.warning


def test_report_computes_calibration_error_on_test_region(tmp_path):
    db_path = make_db_with_candles(tmp_path, n_candles=600)
    report = generate_model_comparison_report(
        db_path=db_path, symbol="frxEURUSD", duration_s=60,
        duration_candles=1, payout_ratio=0.8, warmup=120,
    )
    assert report.calibration_error is not None
    assert 0.0 <= report.calibration_error <= 1.0


def test_approval_requires_beating_every_baseline():
    """
    Direct unit check on the comparison logic itself (not the real-data
    pipeline): if the candidate's EV is worse than any baseline, it must not
    be approved, regardless of statistical reliability.
    """
    from app.models.baseline_comparison import compare_to_baselines

    class Fake:
        def __init__(self, ev, wins, total):
            self.realized_ev_per_stake = ev
            self.wins = wins
            self.total_trades = total

    candidate = Fake(ev=0.01, wins=600, total=1000)
    baselines = {"simple_trend": Fake(ev=0.05, wins=620, total=1000)}
    approval = compare_to_baselines(candidate, baselines, break_even_probability=0.5556)
    assert approval.approved is False


def test_render_model_comparison_text_includes_all_strategies(tmp_path):
    db_path = make_db_with_candles(tmp_path, n_candles=600)
    report = generate_model_comparison_report(
        db_path=db_path, symbol="frxEURUSD", duration_s=60,
        duration_candles=1, payout_ratio=0.8, warmup=120,
    )
    text = render_model_comparison_text(report)
    assert "CANDIDATE" in text
    assert "random" in text
    assert "simple_momentum" in text
    assert "simple_trend" in text
    assert "APPROVED" in text


def test_render_handles_insufficient_data_gracefully(tmp_path):
    db_path = make_db_with_candles(tmp_path, n_candles=50)
    report = generate_model_comparison_report(
        db_path=db_path, symbol="frxEURUSD", duration_s=60,
        duration_candles=1, payout_ratio=0.8, warmup=120,
    )
    text = render_model_comparison_text(report)
    assert "No comparison run" in text
