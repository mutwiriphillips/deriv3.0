"""
Real-data backtest report (spec Stage A of the live-trading roadmap).

Every other validation built so far (backtest/simulator.py, walk_forward.py,
statistics.py) has only ever been exercised against synthetic test data.
This module runs that exact same infrastructure against the REAL candles
accumulated in the production database — the actual answer to "does this
strategy have a statistically real edge," not a hunch from a handful of
live trades.
"""
from __future__ import annotations

import sqlite3
from dataclasses import dataclass

from app.backtest.simulator import BacktestResult, run_backtest
from app.backtest.statistics import StatisticalEvaluation, evaluate_backtest_significance
from app.backtest.walk_forward import WalkForwardReport, run_walk_forward
from app.markets.context import MarketType
from app.risk.payout_math import break_even_probability


@dataclass
class RealDataReport:
    symbol: str
    duration_s: int
    n_candles: int
    strategy_id: str
    backtest_result: BacktestResult
    walk_forward_report: WalkForwardReport
    significance: StatisticalEvaluation
    warning: str | None = None   # e.g. "not enough candles yet" — report is still returned, just flagged


def load_candles_from_db(
    db_path: str, symbol: str, duration_s: int, limit: int | None = None
) -> tuple[list[int], list[float], list[float], list[float], list[float]]:
    """
    Returns (timestamps, opens, highs, lows, closes), ascending by time.

    `limit`, when given, returns only the most recent `limit` candles rather
    than all accumulated history. Feature computation costs roughly 6ms per
    candle per strategy evaluated (measured directly, not estimated) — with
    no limit, report runtime grows linearly with however much history has
    accumulated, and would eventually exceed a typical HTTP request timeout
    (e.g. Render's) right as months of real data made the report most
    useful to check. A bounded recent window keeps runtime predictable at
    the cost of not using very old history, which is a reasonable trade for
    a report about current edge anyway — market conditions from months ago
    are of limited relevance regardless.
    """
    conn = sqlite3.connect(db_path)
    try:
        if limit is not None:
            rows = conn.execute(
                "SELECT timestamp, open, high, low, close FROM candles WHERE symbol=? AND duration_s=? ORDER BY timestamp DESC LIMIT ?",
                (symbol, duration_s, limit),
            ).fetchall()
            rows = list(reversed(rows))
        else:
            rows = conn.execute(
                "SELECT timestamp, open, high, low, close FROM candles WHERE symbol=? AND duration_s=? ORDER BY timestamp ASC",
                (symbol, duration_s),
            ).fetchall()
    finally:
        conn.close()
    return (
        [r[0] for r in rows],
        [r[1] for r in rows],
        [r[2] for r in rows],
        [r[3] for r in rows],
        [r[4] for r in rows],
    )


def generate_real_data_report(
    db_path: str,
    symbol: str,
    duration_s: int,
    strategy_factory,
    duration_candles: int,
    payout_ratio: float,
    market_type: MarketType = MarketType.FOREX,
    warmup: int = 120,
    n_walk_forward_windows: int = 4,
    min_probability_edge: float = 0.0,
    min_expected_value: float = 0.0,
    alpha: float = 0.05,
    min_sample_size: int = 100,
    max_candles: int | None = 3000,
) -> RealDataReport:
    """
    `strategy_factory(train_range)` matches walk_forward.run_walk_forward's
    contract — a fresh strategy instance per window, given that window's
    train-range indices (stateless baselines can just ignore the argument).

    `max_candles` bounds runtime to the most recent N candles regardless of
    how much history has accumulated — see load_candles_from_db's docstring
    for why. Pass None to use everything (only advisable for a one-off
    offline analysis, not a live HTTP endpoint).
    """
    timestamps, opens, highs, lows, closes = load_candles_from_db(db_path, symbol, duration_s, limit=max_candles)
    n = len(closes)

    warning = None
    min_needed = warmup + duration_candles + 1
    if n < min_needed:
        warning = f"only {n} real candles accumulated so far (need at least {min_needed}) — results below are not yet meaningful"

    # Even with too little data, run everything with what exists rather than
    # raising — an early, clearly-flagged partial report is more useful than
    # nothing, and the warning field makes the caller's obligation explicit.
    strategy = strategy_factory((0, n))
    backtest_result = run_backtest(
        symbol=symbol, market_type=market_type, duration_s=duration_s,
        timestamps=timestamps, opens=opens, highs=highs, lows=lows, closes=closes,
        strategy=strategy, duration_candles=duration_candles, payout_ratio=payout_ratio,
        min_probability_edge=min_probability_edge, min_expected_value=min_expected_value,
        warmup=warmup,
    )

    if n >= min_needed * n_walk_forward_windows:
        walk_forward_report = run_walk_forward(
            symbol=symbol, market_type=market_type, duration_s=duration_s,
            timestamps=timestamps, opens=opens, highs=highs, lows=lows, closes=closes,
            strategy_factory=strategy_factory, duration_candles=duration_candles,
            payout_ratio=payout_ratio, n_windows=n_walk_forward_windows, warmup=warmup,
            min_probability_edge=min_probability_edge, min_expected_value=min_expected_value,
        )
    else:
        walk_forward_report = WalkForwardReport()  # empty — not enough data to form even one window per side
        note = "not enough candles for walk-forward windows yet"
        warning = f"{warning}; {note}" if warning else note

    significance = evaluate_backtest_significance(
        backtest_result, break_even_probability(payout_ratio), alpha=alpha, min_sample_size=min_sample_size
    )

    return RealDataReport(
        symbol=symbol, duration_s=duration_s, n_candles=n, strategy_id=getattr(strategy, "strategy_id", "unknown"),
        backtest_result=backtest_result, walk_forward_report=walk_forward_report,
        significance=significance, warning=warning,
    )


def render_report_text(report: RealDataReport) -> str:
    r = report.backtest_result
    lines = [
        f"REAL-DATA BACKTEST REPORT — {report.symbol} ({report.duration_s}s candles)",
        f"Strategy: {report.strategy_id}",
        f"Candles available: {report.n_candles}",
    ]
    if report.warning:
        lines.append(f"WARNING: {report.warning}")
    lines += [
        "",
        f"Total trades: {r.total_trades}",
        f"No-trade count: {r.no_trade_count}",
        f"Win rate: {r.win_rate if r.win_rate is not None else 'N/A'}",
        f"Realized EV per stake: {r.realized_ev_per_stake if r.realized_ev_per_stake is not None else 'N/A'}",
        f"Profit factor: {r.profit_factor if r.profit_factor is not None else 'N/A (no losses or no trades)'}",
        f"Max drawdown: {r.max_drawdown}",
        f"Longest losing streak: {r.longest_losing_streak}",
        "",
        f"Break-even probability: {report.significance.break_even_probability:.4f}",
        f"Wilson 95% CI on win rate: [{report.significance.wilson_ci_low:.4f}, {report.significance.wilson_ci_high:.4f}]",
        f"Statistically significant vs break-even: {report.significance.is_significant} (p={report.significance.p_value:.4f})",
        f"Meets minimum sample size ({report.significance.min_sample_size}): {report.significance.meets_minimum_sample}",
        f"RELIABLE (significant AND enough samples): {report.significance.is_reliable}",
        "",
        f"Walk-forward windows: {len(report.walk_forward_report.windows)}",
        f"Walk-forward stability score: {report.walk_forward_report.stability_score}",
        f"Walk-forward worst window EV: {report.walk_forward_report.worst_window}",
        f"Walk-forward best window EV: {report.walk_forward_report.best_window}",
    ]
    return "\n".join(lines)
