"""
Baseline comparison (spec Part 16): "The complex model must beat the
baseline out-of-sample. If it does not: DO NOT USE THE COMPLEX MODEL."
This is a hard gate, not a suggestion — approved=False means the calling
code should not proceed to use the candidate strategy live or in further
validation stages.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from app.backtest.statistics import evaluate_backtest_significance


@dataclass
class BaselineComparisonResult:
    baseline_name: str
    baseline_ev: float | None
    candidate_ev: float | None
    candidate_beats_baseline: bool


@dataclass
class ModelApprovalReport:
    comparisons: list[BaselineComparisonResult] = field(default_factory=list)
    candidate_is_statistically_reliable: bool = False
    reasons: list[str] = field(default_factory=list)

    @property
    def approved(self) -> bool:
        """Every gate must pass — beating baselines AND being statistically reliable itself."""
        return self.candidate_is_statistically_reliable and all(c.candidate_beats_baseline for c in self.comparisons)


def compare_to_baselines(
    candidate_result,
    baseline_results: dict[str, "object"],
    break_even_probability: float,
    alpha: float = 0.05,
    min_sample_size: int = 100,
) -> ModelApprovalReport:
    """
    `candidate_result` and each value in `baseline_results` are
    backtest.simulator.BacktestResult objects — ideally all produced on the
    SAME out-of-sample data, or this comparison isn't meaningful.
    """
    report = ModelApprovalReport()

    candidate_ev = candidate_result.realized_ev_per_stake
    for name, baseline_result in baseline_results.items():
        baseline_ev = baseline_result.realized_ev_per_stake
        beats = candidate_ev is not None and (baseline_ev is None or candidate_ev > baseline_ev)
        report.comparisons.append(
            BaselineComparisonResult(
                baseline_name=name,
                baseline_ev=baseline_ev,
                candidate_ev=candidate_ev,
                candidate_beats_baseline=beats,
            )
        )
        if not beats:
            report.reasons.append(f"did not beat baseline '{name}' (candidate_ev={candidate_ev}, baseline_ev={baseline_ev})")

    significance = evaluate_backtest_significance(candidate_result, break_even_probability, alpha, min_sample_size)
    report.candidate_is_statistically_reliable = significance.is_reliable
    if not significance.is_reliable:
        report.reasons.append(
            f"candidate not statistically reliable (n={significance.n_trades}, "
            f"p={significance.p_value:.4f}, meets_min_sample={significance.meets_minimum_sample})"
        )

    return report
