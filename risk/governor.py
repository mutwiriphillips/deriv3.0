"""
Risk governor (spec Parts 26 + 33). Two things live here because they're the
same invariant applied at two timescales: risk exposure can only ever be
held steady or reduced by what's happened so far — never increased because
of losses, and an emergency stop, once triggered, persists until a human
explicitly resets it (spec is explicit: "The stop should persist until
manually reset").
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class RiskState(str, Enum):
    NORMAL = "NORMAL"                    # 100% of configured risk
    REDUCED_CONFIDENCE = "REDUCED_CONFIDENCE"   # 50%
    DRAWDOWN_WARNING = "DRAWDOWN_WARNING"       # 25%
    MAJOR_ANOMALY = "MAJOR_ANOMALY"             # 0%
    EMERGENCY_STOP = "EMERGENCY_STOP"           # 0%, persists until manual reset


_STATE_MULTIPLIERS = {
    RiskState.NORMAL: 1.0,
    RiskState.REDUCED_CONFIDENCE: 0.5,
    RiskState.DRAWDOWN_WARNING: 0.25,
    RiskState.MAJOR_ANOMALY: 0.0,
    RiskState.EMERGENCY_STOP: 0.0,
}


@dataclass
class RiskEvent:
    event_type: str
    details: dict


class RiskGovernor:
    def __init__(
        self,
        starting_balance: float,
        max_daily_loss: float,
        max_drawdown: float,          # fraction, e.g. 0.15 = 15% of peak balance
        max_consecutive_losses: int,
        drawdown_warning_threshold: float = 0.5,   # fraction OF max_drawdown that triggers a warning state
    ):
        self.starting_balance = starting_balance
        self.peak_balance = starting_balance
        self.current_balance = starting_balance
        self.max_daily_loss = max_daily_loss
        self.max_drawdown = max_drawdown
        self.max_consecutive_losses = max_consecutive_losses
        self.drawdown_warning_threshold = drawdown_warning_threshold

        self.consecutive_losses = 0
        self._emergency_stopped = False
        self.events: list[RiskEvent] = []

    @property
    def daily_loss(self) -> float:
        return max(0.0, self.starting_balance - self.current_balance)

    @property
    def drawdown_fraction(self) -> float:
        if self.peak_balance <= 0:
            return 0.0
        return max(0.0, (self.peak_balance - self.current_balance) / self.peak_balance)

    @property
    def is_emergency_stopped(self) -> bool:
        return self._emergency_stopped

    def record_trade_result(self, profit_loss: float) -> None:
        self.current_balance += profit_loss
        self.peak_balance = max(self.peak_balance, self.current_balance)

        if profit_loss < 0:
            self.consecutive_losses += 1
        else:
            self.consecutive_losses = 0

        if self.daily_loss >= self.max_daily_loss:
            self._log_event("DAILY_LIMIT_REACHED", {"daily_loss": self.daily_loss})

        if self.drawdown_fraction >= self.max_drawdown:
            self._trigger_emergency_stop("DRAWDOWN_LIMIT_REACHED", {"drawdown_fraction": self.drawdown_fraction})

        if self.consecutive_losses >= self.max_consecutive_losses:
            self._log_event("CONSECUTIVE_LOSS_LIMIT", {"consecutive_losses": self.consecutive_losses})

    def _trigger_emergency_stop(self, reason: str, details: dict) -> None:
        self._emergency_stopped = True
        self._log_event("EMERGENCY_STOP", {**details, "reason": reason})

    def manual_reset(self) -> None:
        """The only way out of EMERGENCY_STOP — spec Part 33: persists until manually reset."""
        self._emergency_stopped = False
        self._log_event("MANUAL_RESET", {})

    def load_state(self, current_balance: float, peak_balance: float, consecutive_losses: int, emergency_stopped: bool) -> None:
        """
        Restores persisted state after a restart (see monitoring/state_persistence.py).
        Config — the limits themselves — always comes from construction/settings,
        never from persisted state, so a config change takes effect immediately
        on restart rather than being silently overridden by an old snapshot.
        """
        self.current_balance = current_balance
        self.peak_balance = peak_balance
        self.consecutive_losses = consecutive_losses
        self._emergency_stopped = emergency_stopped

    def start_new_day(self) -> None:
        """
        Resets the daily-loss reference point at calendar-day rollover.
        Deliberately does NOT reset peak_balance/drawdown — Part 33's drawdown
        is measured from the all-time peak, not a calendar-day concept, so a
        losing streak that started yesterday still counts toward drawdown today.
        """
        self.starting_balance = self.current_balance

    def _log_event(self, event_type: str, details: dict) -> None:
        self.events.append(RiskEvent(event_type=event_type, details=details))

    @property
    def state(self) -> RiskState:
        if self._emergency_stopped:
            return RiskState.EMERGENCY_STOP
        if self.daily_loss >= self.max_daily_loss:
            return RiskState.MAJOR_ANOMALY
        if self.consecutive_losses >= self.max_consecutive_losses:
            return RiskState.MAJOR_ANOMALY
        if self.drawdown_fraction >= self.max_drawdown * self.drawdown_warning_threshold:
            return RiskState.DRAWDOWN_WARNING
        if self.consecutive_losses >= max(1, self.max_consecutive_losses // 2):
            return RiskState.REDUCED_CONFIDENCE
        return RiskState.NORMAL

    @property
    def risk_multiplier(self) -> float:
        """Applied to the configured risk_per_trade before stake sizing. Never > what NORMAL allows."""
        return _STATE_MULTIPLIERS[self.state]

    def can_trade(self) -> bool:
        return self.risk_multiplier > 0.0
