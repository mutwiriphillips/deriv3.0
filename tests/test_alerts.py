import json
import sqlite3

from app.alerts.notifier import Alert, AlertManager, AlertType, ConsoleSink, WebhookSink


def make_db(tmp_path):
    db_path = str(tmp_path / "test.db")
    conn = sqlite3.connect(db_path)
    with open("app/storage/schema.sql") as f:
        conn.executescript(f.read())
    conn.close()
    return db_path


class RecordingSink:
    def __init__(self):
        self.received = []

    def send(self, alert: Alert) -> None:
        self.received.append(alert)


class BrokenSink:
    def send(self, alert: Alert) -> None:
        raise RuntimeError("sink is broken")


def test_console_sink_does_not_raise(capsys):
    sink = ConsoleSink()
    sink.send(Alert(type=AlertType.TRADE_EXECUTED, message="test"))
    captured = capsys.readouterr()
    assert "TRADE_EXECUTED" in captured.out


def test_alert_manager_delivers_to_all_sinks():
    sink1, sink2 = RecordingSink(), RecordingSink()
    manager = AlertManager(sinks=[sink1, sink2])
    manager.notify(AlertType.EMERGENCY_STOP, "drawdown exceeded")
    assert len(sink1.received) == 1
    assert len(sink2.received) == 1
    assert sink1.received[0].type == AlertType.EMERGENCY_STOP


def test_broken_sink_does_not_prevent_delivery_to_other_sinks():
    good_sink = RecordingSink()
    manager = AlertManager(sinks=[BrokenSink(), good_sink])
    manager.notify(AlertType.API_DISCONNECTED, "connection lost")   # must not raise
    assert len(good_sink.received) == 1


def test_alert_manager_persists_to_system_events_when_db_path_given(tmp_path):
    db_path = make_db(tmp_path)
    manager = AlertManager(sinks=[RecordingSink()], db_path=db_path)
    manager.notify(AlertType.DRAWDOWN_LIMIT_REACHED, "drawdown at 16%", details={"drawdown_fraction": 0.16})

    conn = sqlite3.connect(db_path)
    row = conn.execute("SELECT event_type, details FROM system_events").fetchone()
    conn.close()

    assert row[0] == "DRAWDOWN_LIMIT_REACHED"
    details = json.loads(row[1])
    assert details["message"] == "drawdown at 16%"
    assert details["drawdown_fraction"] == 0.16


def test_alert_manager_without_db_path_does_not_persist():
    manager = AlertManager(sinks=[RecordingSink()], db_path=None)
    manager.notify(AlertType.TRADE_EXECUTED, "test")  # must not raise despite no db_path


def test_webhook_sink_failure_is_caught_not_raised():
    sink = WebhookSink(url="http://localhost:1/nonexistent-port-should-fail")
    sink.send(Alert(type=AlertType.API_DISCONNECTED, message="test"))  # must not raise


def test_alert_default_timestamp_is_set():
    alert = Alert(type=AlertType.TRADE_EXECUTED, message="test")
    assert alert.timestamp is not None
    assert "T" in alert.timestamp  # ISO format
