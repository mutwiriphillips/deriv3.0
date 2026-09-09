"""
Reads the system_events table (populated by app/alerts/notifier.py's
AlertManager) to compute an API reliability metric — closes the last
remaining NOT_AUTOMATED item in deployment_checklist.py.
"""
from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

_ERROR_EVENT_TYPES = {"API_DISCONNECTED", "ABNORMAL_LATENCY"}


@dataclass
class ReliabilityStats:
    lookback_hours: float
    error_count: int
    total_events: int

    @property
    def has_data(self) -> bool:
        return self.total_events > 0


def get_reliability_stats(db_path: str, lookback_hours: float = 24.0) -> ReliabilityStats:
    cutoff = (datetime.now(timezone.utc) - timedelta(hours=lookback_hours)).isoformat()
    conn = sqlite3.connect(db_path)
    try:
        total = conn.execute("SELECT COUNT(*) FROM system_events WHERE timestamp >= ?", (cutoff,)).fetchone()[0]
        placeholders = ",".join("?" * len(_ERROR_EVENT_TYPES))
        errors = conn.execute(
            f"SELECT COUNT(*) FROM system_events WHERE timestamp >= ? AND event_type IN ({placeholders})",
            (cutoff, *_ERROR_EVENT_TYPES),
        ).fetchone()[0]
    finally:
        conn.close()
    return ReliabilityStats(lookback_hours=lookback_hours, error_count=errors, total_events=total)
