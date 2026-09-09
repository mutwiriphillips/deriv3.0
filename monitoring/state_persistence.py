"""
Daily state persistence — mitigates the exact gap flagged in the Render
deployment write-up: RiskGovernor and SessionStats previously lived only in
process memory, so a restart silently forgot today's P/L and loss streak.

Writes to the SAME database already used for candles (app/storage/schema.sql),
so on Render this rides the persistent disk that's already provisioned —
no new infrastructure, just a new table and two functions.

One row per calendar date is the whole mechanism for the daily reset: a new
day has no row yet, so load_daily_state returns None and the caller starts
fresh, exactly matching the "reset_daily" semantics SessionStats already had.
"""
from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone

from app.monitoring.session_stats import SessionStats
from app.risk.governor import RiskGovernor


def today_utc_date_str() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


@dataclass
class DailyStateSnapshot:
    trade_date: str
    current_balance: float
    peak_balance: float
    consecutive_losses: int
    emergency_stopped: bool
    trades_today: int
    wins: int
    losses: int
    session_pnl: float
    current_losing_streak: int
    longest_losing_streak_today: int


def save_daily_state(db_path: str, trade_date: str, risk_governor: RiskGovernor, session_stats: SessionStats) -> None:
    conn = sqlite3.connect(db_path)
    try:
        conn.execute(
            """
            INSERT INTO daily_state
                (trade_date, current_balance, peak_balance, consecutive_losses, emergency_stopped,
                 trades_today, wins, losses, session_pnl, current_losing_streak, longest_losing_streak_today,
                 updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(trade_date) DO UPDATE SET
                current_balance=excluded.current_balance,
                peak_balance=excluded.peak_balance,
                consecutive_losses=excluded.consecutive_losses,
                emergency_stopped=excluded.emergency_stopped,
                trades_today=excluded.trades_today,
                wins=excluded.wins,
                losses=excluded.losses,
                session_pnl=excluded.session_pnl,
                current_losing_streak=excluded.current_losing_streak,
                longest_losing_streak_today=excluded.longest_losing_streak_today,
                updated_at=excluded.updated_at
            """,
            (
                trade_date,
                risk_governor.current_balance,
                risk_governor.peak_balance,
                risk_governor.consecutive_losses,
                risk_governor.is_emergency_stopped,
                session_stats.trades_today,
                session_stats.wins,
                session_stats.losses,
                session_stats.session_pnl,
                session_stats.current_losing_streak,
                session_stats.longest_losing_streak_today,
                datetime.now(timezone.utc).isoformat(),
            ),
        )
        conn.commit()
    finally:
        conn.close()


def load_daily_state(db_path: str, trade_date: str) -> DailyStateSnapshot | None:
    conn = sqlite3.connect(db_path)
    try:
        row = conn.execute(
            """
            SELECT trade_date, current_balance, peak_balance, consecutive_losses, emergency_stopped,
                   trades_today, wins, losses, session_pnl, current_losing_streak, longest_losing_streak_today
            FROM daily_state WHERE trade_date = ?
            """,
            (trade_date,),
        ).fetchone()
    finally:
        conn.close()
    if row is None:
        return None
    return DailyStateSnapshot(
        trade_date=row[0],
        current_balance=row[1],
        peak_balance=row[2],
        consecutive_losses=row[3],
        emergency_stopped=bool(row[4]),
        trades_today=row[5],
        wins=row[6],
        losses=row[7],
        session_pnl=row[8],
        current_losing_streak=row[9],
        longest_losing_streak_today=row[10],
    )


def restore_state_if_present(db_path: str, trade_date: str, risk_governor: RiskGovernor, session_stats: SessionStats) -> bool:
    """
    Applies a saved snapshot onto already-constructed risk_governor/session_stats
    instances, in place. Returns True if a snapshot was found and applied,
    False if today has no saved state yet (fresh start — not an error).
    """
    snapshot = load_daily_state(db_path, trade_date)
    if snapshot is None:
        return False

    risk_governor.load_state(
        current_balance=snapshot.current_balance,
        peak_balance=snapshot.peak_balance,
        consecutive_losses=snapshot.consecutive_losses,
        emergency_stopped=snapshot.emergency_stopped,
    )
    session_stats.trades_today = snapshot.trades_today
    session_stats.wins = snapshot.wins
    session_stats.losses = snapshot.losses
    session_stats.session_pnl = snapshot.session_pnl
    session_stats.current_losing_streak = snapshot.current_losing_streak
    session_stats.longest_losing_streak_today = snapshot.longest_losing_streak_today
    return True
