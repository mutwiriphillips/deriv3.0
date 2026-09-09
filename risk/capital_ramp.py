"""
Micro-live capital ramp (spec Part 52): "Do not go directly from demo to
full intended capital. Use stages: DEMO -> MICRO_LIVE -> SMALL_LIVE ->
NORMAL_LIVE. Advance only when performance remains within expected
statistical bounds. If performance deteriorates: return to DEMO or
MICRO_LIVE."

Same asymmetry as RiskGovernor throughout this codebase: advancing stages
is a DELIBERATE action (call .advance() explicitly — meant to sit behind an
authenticated control endpoint, never automatic), while demoting is safe to
do automatically the moment performance deteriorates, since reducing risk
never needs a human to bless it.

Scope note: this module is genuinely usable infrastructure, but nothing in
this codebase currently runs a LIVE trading loop to gate — scripts/run_forward_test.py
refuses to run at all when LIVE_TRADING is true, by design (demo-first).
This ramp is ready for whenever a live execution path exists; it currently
has nothing live to actually throttle.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from app.backtest.statistics import evaluate_significance
from app.monitoring.trade_log import LifetimeStats

_STAGE_ORDER = ["DEMO", "MICRO_LIVE", "SMALL_LIVE", "NORMAL_LIVE"]

# Illustrative ramp percentages -- the spec specifies the STAGES but not exact
# fractions; these are a reasonable, conservative default, not a spec-mandated number.
_STAGE_STAKE_FRACTION = {
    "DEMO": 1.0,          # not real money; fraction is moot but kept for completeness
    "MICRO_LIVE": 0.05,
    "SMALL_LIVE": 0.25,
    "NORMAL_LIVE": 1.0,
}


class CapitalStage(str, Enum):
    DEMO = "DEMO"
    MICRO_LIVE = "MICRO_LIVE"
    SMALL_LIVE = "SMALL_LIVE"
    NORMAL_LIVE = "NORMAL_LIVE"


@dataclass
class AdvancementEligibility:
    eligible: bool
    reason: str


@dataclass
class CapitalRampManager:
    stage: CapitalStage = CapitalStage.DEMO
    min_trades_to_advance: int = 50
    alpha: float = 0.05
    stage_entry_trade_id: int = field(default=0)   # trade_log id snapshot from when this stage began

    def stake_fraction(self) -> float:
        return _STAGE_STAKE_FRACTION[self.stage.value]

    def max_stake_for_stage(self, base_max_stake: float) -> float:
        return base_max_stake * self.stake_fraction()

    def evaluate_advancement_eligibility(self, stage_stats: LifetimeStats, break_even_probability: float) -> AdvancementEligibility:
        """
        Read-only check against performance SINCE this stage began (caller
        must supply stage_stats already scoped via
        trade_log.get_lifetime_stats(..., since_id=self.stage_entry_trade_id)).
        Never advances anything itself.
        """
        if self.stage == CapitalStage.NORMAL_LIVE:
            return AdvancementEligibility(False, "already at the highest stage")

        if stage_stats.trade_count < self.min_trades_to_advance:
            return AdvancementEligibility(
                False, f"only {stage_stats.trade_count} trades at this stage, need {self.min_trades_to_advance}"
            )
        if stage_stats.total_pnl <= 0:
            return AdvancementEligibility(False, f"total P&L at this stage is not positive ({stage_stats.total_pnl})")

        sig = evaluate_significance(
            wins=stage_stats.wins, n=stage_stats.trade_count, break_even_probability=break_even_probability,
            alpha=self.alpha, min_sample_size=self.min_trades_to_advance,
        )
        if not sig.is_reliable:
            return AdvancementEligibility(False, f"performance not statistically reliable at this stage (p={sig.p_value:.4f})")

        return AdvancementEligibility(True, "eligible to advance")

    def advance(self, current_latest_trade_id: int) -> None:
        """Deliberate action only — call this from an authenticated control endpoint, never automatically."""
        idx = _STAGE_ORDER.index(self.stage.value)
        if idx >= len(_STAGE_ORDER) - 1:
            raise ValueError("already at the highest stage")
        self.stage = CapitalStage(_STAGE_ORDER[idx + 1])
        self.stage_entry_trade_id = current_latest_trade_id

    def demote(self, current_latest_trade_id: int, reason: str) -> None:
        """
        Automatic on deterioration — reducing risk never needs a human to
        bless it, matching RiskGovernor's own asymmetry elsewhere in this
        codebase. A no-op if already at DEMO (nothing lower to fall back to).
        """
        idx = _STAGE_ORDER.index(self.stage.value)
        if idx <= 0:
            return
        self.stage = CapitalStage(_STAGE_ORDER[idx - 1])
        self.stage_entry_trade_id = current_latest_trade_id

    def status(self) -> dict:
        return {
            "stage": self.stage.value,
            "stake_fraction": self.stake_fraction(),
            "stage_entry_trade_id": self.stage_entry_trade_id,
            "min_trades_to_advance": self.min_trades_to_advance,
        }
