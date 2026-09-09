"""
Alerts (spec Part 54): "Make the notification layer modular so
Telegram/email/etc. can be added later." Every event from the spec's list is
covered as an AlertType. Two sinks ship here: console (always on) and a
generic webhook (opt-in, since it needs a URL configured).

Every alert is ALSO persisted to the `system_events` table — this is
deliberate double duty: it's both the notification record AND the data
source that finally lets deployment_checklist.py's api_reliability_acceptable
item stop being NOT_AUTOMATED (see monitoring/system_events_log.py).
"""
from __future__ import annotations

import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Protocol

import httpx


class AlertType(str, Enum):
    TRADE_EXECUTED = "TRADE_EXECUTED"
    TRADE_SETTLED = "TRADE_SETTLED"
    DAILY_LIMIT_REACHED = "DAILY_LIMIT_REACHED"
    DRAWDOWN_LIMIT_REACHED = "DRAWDOWN_LIMIT_REACHED"
    CONSECUTIVE_LOSS_LIMIT = "CONSECUTIVE_LOSS_LIMIT"
    STRATEGY_PAUSED = "STRATEGY_PAUSED"
    MODEL_PAUSED = "MODEL_PAUSED"
    API_DISCONNECTED = "API_DISCONNECTED"
    API_RECONNECTED = "API_RECONNECTED"
    ABNORMAL_LATENCY = "ABNORMAL_LATENCY"
    EMERGENCY_STOP = "EMERGENCY_STOP"
    LIVE_MODE_ENABLED = "LIVE_MODE_ENABLED"


@dataclass
class Alert:
    type: AlertType
    message: str
    details: dict = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class NotificationSink(Protocol):
    def send(self, alert: Alert) -> None: ...


class ConsoleSink:
    """Always-on baseline sink — just print()s. Never fails, never needs configuration."""

    def send(self, alert: Alert) -> None:
        print(f"[ALERT] {alert.timestamp} {alert.type.value}: {alert.message} {alert.details or ''}")


class WebhookSink:
    """
    Generic HTTP POST sink — works for Slack/Discord-style incoming
    webhooks, or any custom endpoint, without hardcoding to one provider.
    A delivery failure here is logged, never raised — an alerting problem
    must never crash the trading loop that's trying to report through it.
    """

    def __init__(self, url: str, timeout: float = 5.0):
        self.url = url
        self.timeout = timeout

    def send(self, alert: Alert) -> None:
        payload = {"type": alert.type.value, "message": alert.message, "details": alert.details, "timestamp": alert.timestamp}
        try:
            httpx.post(self.url, json=payload, timeout=self.timeout)
        except Exception as e:
            print(f"[ALERT] WebhookSink delivery failed (non-fatal): {e!r}")


class AlertManager:
    def __init__(self, sinks: list[NotificationSink] | None = None, db_path: str | None = None):
        self.sinks = sinks if sinks is not None else [ConsoleSink()]
        self.db_path = db_path

    def notify(self, alert_type: AlertType, message: str, details: dict | None = None) -> None:
        alert = Alert(type=alert_type, message=message, details=details or {})
        for sink in self.sinks:
            try:
                sink.send(alert)
            except Exception as e:
                # A broken sink must never take down the others or the caller.
                print(f"[ALERT] sink {type(sink).__name__} raised (non-fatal): {e!r}")
        if self.db_path is not None:
            self._persist(alert)

    def _persist(self, alert: Alert) -> None:
        import json
        conn = sqlite3.connect(self.db_path)
        try:
            conn.execute(
                "INSERT INTO system_events (timestamp, event_type, details) VALUES (?, ?, ?)",
                (alert.timestamp, alert.type.value, json.dumps({"message": alert.message, **alert.details})),
            )
            conn.commit()
        finally:
            conn.close()
