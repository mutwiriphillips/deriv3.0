"""
Model comparison report (spec Stage B / Part 16 / Part 35). Fits the
logistic regression model on REAL accumulated data for the first time —
every prior fit in this codebase used synthetic data — and puts it through
the exact same baseline_comparison.py gate built in Phase 11, now with a
real verdict instead of a synthetic one.

Chronological train/test split only (spec Part 18/19: no look-ahead, never
shuffled). The model is fit on the train region only; every strategy
(candidate and every baseline) is then backtested on the SAME held-out test
region, which is what makes the comparison fair.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from app.analytics.real_data_report import load_candles_from_db
from app.backtest.simulator import BacktestResult, run_backtest
from app.markets.context import MarketType
from app.models.baseline_comparison import ModelApprovalReport, compare_to_baselines
from app.models.dataset import build_training_examples
from app.models.evaluation import calibration_curve, calibration_error
from app.models.logistic_model import LogisticProbabilityModel
from app.risk.payout_math import break_even_probability
from app.strategies.baselines import RandomStrategy, SimpleMomentumStrategy, SimpleTrendStrategy
from app.strategies.ml_strategy import ProbabilityModelStrategy

_MIN_TRAIN_EXAMPLES = 50   # LogisticProbabilityModel technically accepts >=10, but 50 is a saner floor for anything meaningful


@dataclass
class ModelComparisonReport:
    symbol: str
    n_candles: int
    n_train_examples: int
    candidate_result: BacktestResult | None
    baseline_results: dict[str, BacktestResult] = field(default_factory=dict)
    approval: ModelApprovalReport | None = None
    calibration_error: float | None = None   # Expected Calibration Error on the held-out test region; None if not computed
    warning: str | None = None


def generate_model_comparison_report(
    db_path: str,
    symbol: str,
    duration_s: int,
    duration_candles: int,
    payout_ratio: float,
    market_type: MarketType = MarketType.FOREX,
    train_fraction: float = 0.7,
    warmup: int = 120,
    min_probability_edge: float = 0.0,
    min_expected_value: float = 0.0,
    alpha: float = 0.05,
    min_sample_size: int = 100,
    max_candles: int | None = 3000,
) -> ModelComparisonReport:
    """`max_candles` bounds runtime the same way as real_data_report.py's identically-named parameter — see its docstring."""
    timestamps, opens, highs, lows, closes = load_candles_from_db(db_path, symbol, duration_s, limit=max_candles)
    n = len(closes)
    train_end = int(n * train_fraction)

    training_examples = build_training_examples(
        timestamps[:train_end], opens[:train_end], highs[:train_end], lows[:train_end], closes[:train_end],
        duration_candles=duration_candles, warmup=warmup, symbol=symbol, duration_s=duration_s,
    )

    if len(training_examples) < _MIN_TRAIN_EXAMPLES:
        return ModelComparisonReport(
            symbol=symbol, n_candles=n, n_train_examples=len(training_examples),
            candidate_result=None,
            warning=(
                f"only {len(training_examples)} training examples available (need at least {_MIN_TRAIN_EXAMPLES}) "
                f"from {n} total candles ({train_end} in the train split) — cannot fit a model yet"
            ),
        )

    model = LogisticProbabilityModel()
    fit_result = model.fit(training_examples)
    candidate_strategy = ProbabilityModelStrategy(model, n_train=fit_result.n_train)

    def _test_backtest(strategy) -> BacktestResult:
        # Full series is passed so features have real history to compute
        # from, but `warmup=train_end` means no trade is ever entered before
        # the test region starts -- this is the same technique
        # backtest/walk_forward.py already uses for window isolation.
        return run_backtest(
            symbol=symbol, market_type=market_type, duration_s=duration_s,
            timestamps=timestamps, opens=opens, highs=highs, lows=lows, closes=closes,
            strategy=strategy, duration_candles=duration_candles, payout_ratio=payout_ratio,
            min_probability_edge=min_probability_edge, min_expected_value=min_expected_value,
            warmup=max(warmup, train_end),
        )

    candidate_result = _test_backtest(candidate_strategy)
    baseline_results = {
        "random": _test_backtest(RandomStrategy(seed=1)),
        "simple_momentum": _test_backtest(SimpleMomentumStrategy()),
        "simple_trend": _test_backtest(SimpleTrendStrategy()),
    }

    approval = compare_to_baselines(
        candidate_result, baseline_results, break_even_probability(payout_ratio), alpha, min_sample_size
    )

    # Calibration (spec Part 17) on the SAME held-out test region -- never
    # the training data, which would trivially look well-calibrated.
    test_examples = build_training_examples(
        timestamps[train_end:], opens[train_end:], highs[train_end:], lows[train_end:], closes[train_end:],
        duration_candles=duration_candles, warmup=0, symbol=symbol, duration_s=duration_s,
    )
    candidate_calibration_error = None
    if test_examples:
        y_true = [label for _, label in test_examples]
        y_prob = [model.predict_proba(vec) for vec, _ in test_examples]
        candidate_calibration_error = calibration_error(calibration_curve(y_true, y_prob))

    warning = None
    if candidate_result.total_trades < min_sample_size:
        warning = (
            f"candidate only traded {candidate_result.total_trades} times on the test region "
            f"(need {min_sample_size} for a reliable verdict) — treat this result as preliminary"
        )

    return ModelComparisonReport(
        symbol=symbol, n_candles=n, n_train_examples=len(training_examples),
        candidate_result=candidate_result, baseline_results=baseline_results,
        approval=approval, calibration_error=candidate_calibration_error, warning=warning,
    )


def render_model_comparison_text(report: ModelComparisonReport) -> str:
    lines = [
        f"CHAMPION/CHALLENGER REPORT — {report.symbol}",
        f"Total candles: {report.n_candles}",
        f"Training examples: {report.n_train_examples}",
    ]
    if report.warning:
        lines.append(f"WARNING: {report.warning}")
    if report.candidate_result is None:
        lines.append("No comparison run — insufficient training data.")
        return "\n".join(lines)

    c = report.candidate_result
    lines += [
        "",
        f"CANDIDATE (logistic regression) — trades: {c.total_trades}, win_rate: {c.win_rate}, "
        f"realized_ev: {c.realized_ev_per_stake}",
    ]
    for name, r in report.baseline_results.items():
        lines.append(f"BASELINE '{name}' — trades: {r.total_trades}, win_rate: {r.win_rate}, realized_ev: {r.realized_ev_per_stake}")

    lines += [
        "",
        f"Statistically reliable: {report.approval.candidate_is_statistically_reliable}",
        f"APPROVED (beats every baseline AND statistically reliable): {report.approval.approved}",
        f"Calibration error (ECE) on test region: {report.calibration_error}",
    ]
    if report.approval.reasons:
        lines.append("Reasons not approved:")
        lines.extend(f"  - {r}" for r in report.approval.reasons)
    return "\n".join(lines)
