"""
Stress testing (spec Part 38): "The strategy should remain survivable under
adverse conditions." Each scenario transforms a real backtest's trade
sequence into a stressed P&L list, then reuses the Monte Carlo engine
(Phase 8) to answer the only question that matters: does this survive, or
does it ruin the account.

Two of the spec's named scenarios — volatile markets, low-volatility
markets — are implemented as genuine regime-filtered analysis using
TradeRecord.regime (added this phase specifically to make this real,
rather than faked with an arbitrary volatility multiplier that wouldn't
reflect anything the strategy actually experienced).
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field

from app.backtest.monte_carlo import MonteCarloReport, run_monte_carlo
from app.backtest.simulator import BacktestResult, TradeRecord
from app.markets.context import Regime


@dataclass
class StressScenarioResult:
    name: str
    description: str
    n_trades: int
    monte_carlo: MonteCarloReport | None
    survived: bool
    note: str | None = None   # e.g. "no trades in this regime yet" when a scenario can't run


@dataclass
class StressTestSuite:
    scenarios: list[StressScenarioResult] = field(default_factory=list)

    @property
    def all_survived(self) -> bool:
        """Only counts scenarios that actually ran — a scenario with no applicable trades is neither a pass nor a fail."""
        ran = [s for s in self.scenarios if s.monte_carlo is not None]
        return len(ran) > 0 and all(s.survived for s in ran)


def _pnl_list(trades: list[TradeRecord]) -> list[float]:
    return [t.profit_loss for t in trades]


def apply_win_probability_reduction(trades: list[TradeRecord], reduction: float, seed: int | None = None) -> list[float]:
    """
    Flips enough WIN trades to LOSS (chosen at random) to reduce the
    empirical win rate by `reduction` percentage points (e.g. 0.10 = 10
    points lower). If the current win rate is already below the reduction
    amount, flips all wins to losses rather than going negative.
    """
    if not trades:
        return []
    rng = random.Random(seed)
    wins = [t for t in trades if t.result == "WIN"]
    losses = [t for t in trades if t.result == "LOSS"]
    n_to_flip = min(len(wins), round(reduction * len(trades)))
    flip_set = set(rng.sample(range(len(wins)), n_to_flip)) if n_to_flip else set()

    pnl = []
    for idx, t in enumerate(wins):
        if idx in flip_set:
            pnl.append(-t.stake)   # now a loss
        else:
            pnl.append(t.profit_loss)
    pnl.extend(t.profit_loss for t in losses)
    return pnl


def apply_payout_reduction(trades: list[TradeRecord], reduction_fraction: float) -> list[float]:
    """Scales down the profit on every WIN by `reduction_fraction` (e.g. 0.20 = 20% lower payout). Losses are unaffected."""
    return [t.profit_loss * (1 - reduction_fraction) if t.result == "WIN" else t.profit_loss for t in trades]


def apply_slippage(trades: list[TradeRecord], extra_cost_per_trade: float) -> list[float]:
    """
    Models higher latency / execution slippage as a fixed extra cost
    subtracted from every trade's outcome — a simplification (real slippage
    would depend on the specific price move missed), stated plainly as a
    modeling choice rather than a measured market fact.
    """
    return [t.profit_loss - extra_cost_per_trade for t in trades]


def apply_dropped_trades(trades: list[TradeRecord], drop_fraction: float, seed: int | None = None) -> list[float]:
    """Models missed trades / execution failures by removing a random fraction of trades entirely, as if never taken."""
    if not trades:
        return []
    rng = random.Random(seed)
    n_keep = max(0, len(trades) - round(drop_fraction * len(trades)))
    kept = rng.sample(trades, n_keep) if n_keep < len(trades) else list(trades)
    return [t.profit_loss for t in kept]


def apply_forced_losing_streak(trades: list[TradeRecord], streak_length: int) -> list[float]:
    """Appends an artificial run of `streak_length` losses (using the average stake) to the real sequence — an explicit worst-case addition, not a resample."""
    pnl = _pnl_list(trades)
    if not trades:
        return [0.0] * streak_length
    avg_stake = sum(t.stake for t in trades) / len(trades)
    return pnl + [-avg_stake] * streak_length


def filter_by_regime(trades: list[TradeRecord], regimes: set[Regime]) -> list[TradeRecord]:
    regime_values = {r.value for r in regimes}
    return [t for t in trades if t.regime in regime_values]


def _run_scenario(name: str, description: str, pnl: list[float], starting_bankroll: float, n_simulations: int, ruin_threshold: float, seed: int | None) -> StressScenarioResult:
    if not pnl:
        return StressScenarioResult(name=name, description=description, n_trades=0, monte_carlo=None, survived=False, note="no trades available for this scenario")
    mc = run_monte_carlo(pnl, starting_bankroll, n_simulations=n_simulations, seed=seed)
    return StressScenarioResult(
        name=name, description=description, n_trades=len(pnl), monte_carlo=mc,
        survived=mc.probability_of_ruin <= ruin_threshold,
    )


def run_stress_test_suite(
    backtest_result: BacktestResult,
    starting_bankroll: float,
    n_simulations: int = 500,
    ruin_probability_threshold: float = 0.05,
    seed: int | None = None,
) -> StressTestSuite:
    """
    Runs every named scenario from spec Part 38 against a real backtest
    result. `ruin_probability_threshold` is the bar for "survivable" — 5%
    default, i.e. no more than a 1-in-20 chance of hitting zero across the
    Monte Carlo resamples.
    """
    trades = backtest_result.trades
    suite = StressTestSuite()

    suite.scenarios.append(_run_scenario(
        "win_probability_-10pp", "10 percentage point reduction in win rate",
        apply_win_probability_reduction(trades, 0.10, seed=seed), starting_bankroll, n_simulations, ruin_probability_threshold, seed,
    ))
    suite.scenarios.append(_run_scenario(
        "win_probability_-20pp", "20 percentage point reduction in win rate",
        apply_win_probability_reduction(trades, 0.20, seed=seed), starting_bankroll, n_simulations, ruin_probability_threshold, seed,
    ))
    suite.scenarios.append(_run_scenario(
        "lower_payout_-20pct", "20% lower payout on every win",
        apply_payout_reduction(trades, 0.20), starting_bankroll, n_simulations, ruin_probability_threshold, seed,
    ))
    suite.scenarios.append(_run_scenario(
        "higher_latency_slippage", "fixed extra cost per trade modeling execution slippage",
        apply_slippage(trades, extra_cost_per_trade=sum(t.stake for t in trades) / len(trades) * 0.05 if trades else 0.0),
        starting_bankroll, n_simulations, ruin_probability_threshold, seed,
    ))
    suite.scenarios.append(_run_scenario(
        "missed_trades_-20pct", "20% of trades never executed (missed opportunities / execution failures)",
        apply_dropped_trades(trades, 0.20, seed=seed), starting_bankroll, n_simulations, ruin_probability_threshold, seed,
    ))
    suite.scenarios.append(_run_scenario(
        "forced_losing_streak", "an artificial run of 10 consecutive losses appended to the real sequence",
        apply_forced_losing_streak(trades, streak_length=10), starting_bankroll, n_simulations, ruin_probability_threshold, seed,
    ))
    suite.scenarios.append(_run_scenario(
        "high_volatility_regime_only", "performance using only trades entered during HIGH_VOLATILITY",
        _pnl_list(filter_by_regime(trades, {Regime.HIGH_VOLATILITY})), starting_bankroll, n_simulations, ruin_probability_threshold, seed,
    ))
    suite.scenarios.append(_run_scenario(
        "low_volatility_regime_only", "performance using only trades entered during LOW_VOLATILITY",
        _pnl_list(filter_by_regime(trades, {Regime.LOW_VOLATILITY})), starting_bankroll, n_simulations, ruin_probability_threshold, seed,
    ))

    return suite


def render_stress_test_text(suite: StressTestSuite) -> str:
    lines = ["STRESS TEST REPORT", f"Overall survived (all applicable scenarios): {suite.all_survived}", ""]
    for s in suite.scenarios:
        lines.append(f"[{'PASS' if s.survived else 'FAIL' if s.monte_carlo else 'N/A'}] {s.name} — {s.description}")
        if s.note:
            lines.append(f"    {s.note}")
        elif s.monte_carlo:
            lines.append(
                f"    n_trades={s.n_trades} probability_of_ruin={s.monte_carlo.probability_of_ruin:.4f} "
                f"expected_ending_bankroll={s.monte_carlo.expected_ending_bankroll:.2f} "
                f"p95_drawdown={s.monte_carlo.drawdown_percentile(95):.2f}"
            )
    return "\n".join(lines)
