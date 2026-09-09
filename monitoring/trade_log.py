"""
Lifetime trade log (closes two of the three NOT_AUTOMATED items in
deployment_checklist.py: demo_forward_test_passed and
execution_latency_acceptable). Deliberately separate from SessionStats
(daily) and daily_state (keyed per calendar day) — this table is never
reset, so it's the first place in the codebase that can answer "how has
this actually performed over weeks," not just "today."
"""
from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone


def record_trade(
    db_path: str,
    symbol: str,
    direction: str,
    result: str,
    profit_loss: float,
    stake: float,
    total_execution_latency_ms: float | None,
) -> None:
    conn = sqlite3.connect(db_path)
    try:
        conn.execute(
            """
            INSERT INTO trade_log
                (timestamp, symbol, direction, result, profit_loss, stake, total_execution_latency_ms)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (datetime.now(timezone.utc).isoformat(), symbol, direction, result, profit_loss, stake, total_execution_latency_ms),
        )
        conn.commit()
    finally:
        conn.close()


@dataclass
class LifetimeStats:
    trade_count: int
    wins: int
    win_rate: float | None
    total_pnl: float
    latencies_ms: list[float]

    def latency_percentile(self, p: float) -> float | None:
        """p in [0, 100]. Nearest-rank; fine for the sample sizes this runs at."""
        if not self.latencies_ms:
            return None
        sorted_lat = sorted(self.latencies_ms)
        idx = min(len(sorted_lat) - 1, max(0, int(round(p / 100 * (len(sorted_lat) - 1)))))
        return sorted_lat[idx]


def get_latest_trade_id(db_path: str) -> int:
    """Returns the highest trade_log id so far, or 0 if empty. Used to snapshot 'as of now' for stage-scoped stats."""
    conn = sqlite3.connect(db_path)
    try:
        row = conn.execute("SELECT MAX(id) FROM trade_log").fetchone()
    finally:
        conn.close()
    return row[0] or 0


def get_lifetime_stats(db_path: str, symbol: str | None = None, since_id: int | None = None) -> LifetimeStats:
    conn = sqlite3.connect(db_path)
    try:
        clauses = []
        params: list = []
        if symbol is not None:
            clauses.append("symbol=?")
            params.append(symbol)
        if since_id is not None:
            clauses.append("id > ?")
            params.append(since_id)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        rows = conn.execute(f"SELECT result, profit_loss, total_execution_latency_ms FROM trade_log {where}", params).fetchall()
    finally:
        conn.close()

    trade_count = len(rows)
    wins = sum(1 for r in rows if r[0] == "WIN")
    total_pnl = sum(r[1] for r in rows)
    latencies = [r[2] for r in rows if r[2] is not None]

    return LifetimeStats(
        trade_count=trade_count,
        wins=wins,
        win_rate=(wins / trade_count) if trade_count else None,
        total_pnl=total_pnl,
        latencies_ms=latencies,
    )
