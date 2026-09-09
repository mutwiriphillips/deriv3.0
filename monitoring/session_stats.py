"""
Session statistics for the dashboard (spec Part 39: TRADES TODAY, WIN RATE,
LOSING STREAK). Deliberately separate from RiskGovernor — that module
decides whether to keep trading; this one just reports what happened. Fed by
settlement events, which aren't wired up yet (no proposal_open_contract
listener exists as of Phase 13) — this is ready to receive them once that's
built.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class SessionStats:
    trades_today: int = 0
    wins: int = 0
    losses: int = 0
    session_pnl: float = 0.0
    current_losing_streak: int = 0
    longest_losing_streak_today: int = 0

    def record_settlement(self, result: str, profit_loss: float) -> None:
        self.trades_today += 1
        self.session_pnl += profit_loss
        if result == "WIN":
            self.wins += 1
            self.current_losing_streak = 0
        else:
            self.losses += 1
            self.current_losing_streak += 1
            self.longest_losing_streak_today = max(self.longest_losing_streak_today, self.current_losing_streak)

    @property
    def win_rate(self) -> float | None:
        return self.wins / self.trades_today if self.trades_today else None

    def reset_daily(self) -> None:
        """Called at the start of a new trading day — spec's daily risk governor resets alongside this."""
        self.trades_today = 0
        self.wins = 0
        self.losses = 0
        self.session_pnl = 0.0
        self.current_losing_streak = 0
        self.longest_losing_streak_today = 0
