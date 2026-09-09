from datetime import datetime

from fastapi.testclient import TestClient

from app.api.server import app, latest_state, LatestStateHolder
from app.monitoring.dashboard import DashboardState


def test_health_returns_ok():
    client = TestClient(app)
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_root_serves_the_advisor_frontend():
    client = TestClient(app)
    response = client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert "Trade Advisor" in response.text


def test_status_before_any_tick_says_starting():
    latest_state.state = None
    latest_state.updated_at = None
    client = TestClient(app)
    response = client.get("/status")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "starting"


def test_status_after_a_tick_returns_dashboard_json():
    state = DashboardState(
        bot_status="RUNNING", account_balance=1000.0, daily_pnl=0.0, session_pnl=0.0,
        drawdown_fraction=0.0, active_strategy="baseline_trend", model_version="v1",
        api_status="CONNECTED", current_symbol="frxEURUSD", current_regime="TREND_UP",
        trades_today=3, win_rate=0.67, current_losing_streak=0,
    )
    latest_state.update(state)

    client = TestClient(app)
    response = client.get("/status")
    body = response.json()

    assert body["status"] == "running"
    assert body["dashboard"]["bot_status"] == "RUNNING"
    assert body["dashboard"]["current_symbol"] == "frxEURUSD"
    assert body["dashboard"]["trades_today"] == 3
    assert body["updated_at"] is not None


def test_backtest_report_endpoint_rejects_unknown_strategy():
    client = TestClient(app)
    response = client.get("/backtest-report?strategy=not_a_real_strategy")
    assert response.status_code == 200
    assert "error" in response.json()


def test_backtest_report_endpoint_runs_against_a_real_db(tmp_path, monkeypatch):
    import sqlite3
    import numpy as np

    db_path = str(tmp_path / "test.db")
    conn = sqlite3.connect(db_path)
    with open("app/storage/schema.sql") as f:
        conn.executescript(f.read())
    rng = np.random.default_rng(1)
    closes = list(1.0 + np.cumsum(rng.normal(0, 0.002, 200)))
    for i, c in enumerate(closes):
        t = 1_700_000_000 + i * 60
        conn.execute(
            "INSERT INTO candles (symbol, duration_s, timestamp, open, high, low, close, tick_count) VALUES (?,?,?,?,?,?,?,?)",
            ("frxEURUSD", 60, t, c, c + 0.001, c - 0.001, c, 1),
        )
    conn.commit()
    conn.close()

    monkeypatch.setattr("app.api.server.settings.db_path", db_path)

    client = TestClient(app)
    response = client.get("/backtest-report?strategy=random")
    body = response.json()

    assert response.status_code == 200
    assert body["symbol"] == "frxEURUSD"
    assert "text_report" in body
    assert "Total trades" in body["text_report"]


def test_model_comparison_report_endpoint_with_insufficient_data(tmp_path, monkeypatch):
    import sqlite3

    db_path = str(tmp_path / "test.db")
    conn = sqlite3.connect(db_path)
    with open("app/storage/schema.sql") as f:
        conn.executescript(f.read())
    conn.commit()
    conn.close()

    monkeypatch.setattr("app.api.server.settings.db_path", db_path)

    client = TestClient(app)
    response = client.get("/model-comparison-report")
    body = response.json()

    assert response.status_code == 200
    assert body["n_candles"] == 0
    assert body["warning"] is not None
    assert "text_report" in body


def test_stress_test_report_endpoint_with_no_trades(tmp_path, monkeypatch):
    import sqlite3

    db_path = str(tmp_path / "test.db")
    conn = sqlite3.connect(db_path)
    with open("app/storage/schema.sql") as f:
        conn.executescript(f.read())
    conn.commit()
    conn.close()

    monkeypatch.setattr("app.api.server.settings.db_path", db_path)

    client = TestClient(app)
    response = client.get("/stress-test-report")
    body = response.json()

    assert response.status_code == 200
    assert "warning" in body


def test_stress_test_report_endpoint_rejects_unknown_strategy():
    client = TestClient(app)
    response = client.get("/stress-test-report?strategy=not_real")
    assert "error" in response.json()


def test_deployment_checklist_endpoint_with_no_data(tmp_path, monkeypatch):
    import sqlite3

    db_path = str(tmp_path / "test.db")
    conn = sqlite3.connect(db_path)
    with open("app/storage/schema.sql") as f:
        conn.executescript(f.read())
    conn.commit()
    conn.close()

    monkeypatch.setattr("app.api.server.settings.db_path", db_path)

    client = TestClient(app)
    response = client.get("/deployment-checklist")
    body = response.json()

    assert response.status_code == 200
    assert body["all_passed"] is False   # can never pass with zero data
    assert len(body["items"]) == 14
    assert "text_report" in body


def test_trade_advice_endpoint_with_no_local_data(tmp_path, monkeypatch):
    import sqlite3
    db_path = str(tmp_path / "test.db")
    conn = sqlite3.connect(db_path)
    with open("app/storage/schema.sql") as f:
        conn.executescript(f.read())
    conn.close()
    monkeypatch.setattr("app.api.server.settings.db_path", db_path)

    client = TestClient(app)
    response = client.get("/trade-advice")
    body = response.json()
    assert "error" in body
    assert "not enough warmup history" in body["error"]


def test_trade_advice_endpoint_rejects_unknown_strategy():
    client = TestClient(app)
    response = client.get("/trade-advice?strategy=not_real")
    assert "error" in response.json()


def test_trade_advice_endpoint_rejects_malformed_durations():
    client = TestClient(app)
    response = client.get("/trade-advice?durations=abc,def")
    assert "error" in response.json()


def test_trade_advice_endpoint_returns_advice_with_enough_data(tmp_path, monkeypatch):
    import sqlite3
    import numpy as np
    from unittest.mock import AsyncMock, patch

    db_path = str(tmp_path / "test.db")
    conn = sqlite3.connect(db_path)
    with open("app/storage/schema.sql") as f:
        conn.executescript(f.read())
    rng = np.random.default_rng(1)
    closes = list(1.0 + np.cumsum(rng.normal(0.0005, 0.002, 200)))
    for i, c in enumerate(closes):
        t = 1_700_000_000 + i * 60
        conn.execute(
            "INSERT INTO candles (symbol, duration_s, timestamp, open, high, low, close, tick_count) VALUES (?,?,?,?,?,?,?,?)",
            ("frxEURUSD", 60, t, c, c + 0.001, c - 0.001, c, 1),
        )
    conn.commit()
    conn.close()
    monkeypatch.setattr("app.api.server.settings.db_path", db_path)

    mock_client = AsyncMock()
    mock_client.proposal = AsyncMock(return_value={"id": "p1", "ask_price": "10.00", "payout": "18.00"})
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=None)

    with patch("app.api.server.DerivPublicClient", return_value=mock_client):
        client = TestClient(app)
        response = client.get("/trade-advice?durations=60,120")

    body = response.json()
    assert response.status_code == 200
    assert body["action"] in ("TRADE", "NO_TRADE")
    assert "duration_candidates" in body
    assert body["symbol"] == "frxEURUSD"


def test_latest_state_holder_updates_timestamp():
    holder = LatestStateHolder()
    assert holder.updated_at is None
    state = DashboardState(
        bot_status="RUNNING", account_balance=1000.0, daily_pnl=0.0, session_pnl=0.0,
        drawdown_fraction=0.0, active_strategy="x", model_version="v1", api_status="CONNECTED",
    )
    holder.update(state)
    assert holder.updated_at is not None
    parsed = datetime.fromisoformat(holder.updated_at)
    assert parsed.tzinfo is not None
