"""
Monte Carlo analysis (spec Part 37): "Do not assume historical order is the
only possible sequence." Resamples the realized trade P&L sequence with
replacement to estimate the drawdown distribution and probability of ruin,
rather than trusting the one sequence that actually happened.
"""
from __future__ import annotations

import random
from dataclasses import dataclass


@dataclass
class MonteCarloReport:
    n_simulations: int
    starting_bankroll: float
    max_drawdowns: list[float]
    ending_bankrolls: list[float]
    ruin_count: int   # simulations where bankroll hit <= 0 at some point

    @property
    def probability_of_ruin(self) -> float:
        return self.ruin_count / self.n_simulations if self.n_simulations else 0.0

    def drawdown_percentile(self, p: float) -> float:
        """p in [0, 100]. Uses nearest-rank; fine for the sample sizes this runs at (hundreds-thousands)."""
        if not self.max_drawdowns:
            return 0.0
        sorted_dd = sorted(self.max_drawdowns)
        idx = min(len(sorted_dd) - 1, max(0, int(round(p / 100 * (len(sorted_dd) - 1)))))
        return sorted_dd[idx]

    @property
    def expected_ending_bankroll(self) -> float:
        return sum(self.ending_bankrolls) / len(self.ending_bankrolls) if self.ending_bankrolls else self.starting_bankroll


def run_monte_carlo(
    trade_profit_losses: list[float],
    starting_bankroll: float,
    n_simulations: int = 1000,
    seed: int | None = None,
) -> MonteCarloReport:
    """
    Resamples `trade_profit_losses` (the actual per-trade P&L from a
    backtest) WITH REPLACEMENT, `n_simulations` times, each simulation
    replaying as many trades as were in the original sequence. This tests
    "what if the same edge existed but the wins/losses fell in a different
    order (or repeated some trades disproportionately)" — not "what if the
    edge itself were different."
    """
    if not trade_profit_losses:
        return MonteCarloReport(
            n_simulations=n_simulations, starting_bankroll=starting_bankroll,
            max_drawdowns=[], ending_bankrolls=[], ruin_count=0,
        )

    rng = random.Random(seed)
    n_trades = len(trade_profit_losses)
    max_drawdowns: list[float] = []
    ending_bankrolls: list[float] = []
    ruin_count = 0

    for _ in range(n_simulations):
        bankroll = starting_bankroll
        peak = starting_bankroll
        max_dd = 0.0
        ruined = False
        for _ in range(n_trades):
            pnl = rng.choice(trade_profit_losses)
            bankroll += pnl
            peak = max(peak, bankroll)
            max_dd = max(max_dd, peak - bankroll)
            if bankroll <= 0:
                ruined = True
        max_drawdowns.append(max_dd)
        ending_bankrolls.append(bankroll)
        if ruined:
            ruin_count += 1

    return MonteCarloReport(
        n_simulations=n_simulations,
        starting_bankroll=starting_bankroll,
        max_drawdowns=max_drawdowns,
        ending_bankrolls=ending_bankrolls,
        ruin_count=ruin_count,
    )
