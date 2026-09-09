from fastapi.testclient import TestClient

from app.api.server import app, bot_controller


def test_strategies_endpoint_is_read_only_and_unauthenticated():
    client = TestClient(app)
    response = client.get("/strategies")
    assert response.status_code == 200
    body = response.json()
    assert "available_strategies" in body
    assert "active_strategy" in body


def test_control_disabled_by_default_returns_401(monkeypatch):
    monkeypatch.setattr("app.api.server.settings.control_api_key", "")
    client = TestClient(app)
    response = client.post("/control/stop")
    assert response.status_code == 401
    assert "disabled" in response.json()["detail"]


def test_control_rejects_wrong_key(monkeypatch):
    monkeypatch.setattr("app.api.server.settings.control_api_key", "correct-key")
    client = TestClient(app)
    response = client.post("/control/stop", headers={"X-API-Key": "wrong-key"})
    assert response.status_code == 401


def test_control_accepts_correct_key_and_actually_stops(monkeypatch):
    monkeypatch.setattr("app.api.server.settings.control_api_key", "correct-key")
    bot_controller.start()  # ensure known starting state
    client = TestClient(app)

    response = client.post("/control/stop", headers={"X-API-Key": "correct-key"})
    assert response.status_code == 200
    assert response.json()["running"] is False
    assert bot_controller.running is False  # the actual shared controller changed, not just the response


def test_control_start_after_stop(monkeypatch):
    monkeypatch.setattr("app.api.server.settings.control_api_key", "correct-key")
    bot_controller.stop()
    client = TestClient(app)
    response = client.post("/control/start", headers={"X-API-Key": "correct-key"})
    assert response.status_code == 200
    assert response.json()["running"] is True


def test_control_select_strategy_changes_active_strategy(monkeypatch):
    monkeypatch.setattr("app.api.server.settings.control_api_key", "correct-key")
    client = TestClient(app)
    response = client.post("/control/select-strategy?name=random", headers={"X-API-Key": "correct-key"})
    assert response.status_code == 200
    assert response.json()["active_strategy"] == "random"
    assert bot_controller.active_strategy_name == "random"


def test_control_select_unknown_strategy_returns_400(monkeypatch):
    monkeypatch.setattr("app.api.server.settings.control_api_key", "correct-key")
    client = TestClient(app)
    response = client.post("/control/select-strategy?name=not_real", headers={"X-API-Key": "correct-key"})
    assert response.status_code == 400


def test_capital_ramp_status_is_read_only_and_unauthenticated(tmp_path, monkeypatch):
    import sqlite3
    db_path = str(tmp_path / "test.db")
    conn = sqlite3.connect(db_path)
    with open("app/storage/schema.sql") as f:
        conn.executescript(f.read())
    conn.close()
    monkeypatch.setattr("app.api.server.settings.db_path", db_path)

    client = TestClient(app)
    response = client.get("/capital-ramp/status")
    assert response.status_code == 200
    body = response.json()
    assert "stage" in body
    assert "trades_at_current_stage" in body


def test_capital_ramp_advance_requires_auth(tmp_path, monkeypatch):
    monkeypatch.setattr("app.api.server.settings.control_api_key", "")
    client = TestClient(app)
    response = client.post("/control/capital-ramp/advance")
    assert response.status_code == 401


def test_capital_ramp_advance_refuses_when_not_eligible(tmp_path, monkeypatch):
    import sqlite3
    from app.api.server import capital_ramp_manager
    from app.risk.capital_ramp import CapitalStage

    db_path = str(tmp_path / "test.db")
    conn = sqlite3.connect(db_path)
    with open("app/storage/schema.sql") as f:
        conn.executescript(f.read())
    conn.close()

    capital_ramp_manager.stage = CapitalStage.DEMO
    capital_ramp_manager.stage_entry_trade_id = 0
    monkeypatch.setattr("app.api.server.settings.db_path", db_path)
    monkeypatch.setattr("app.api.server.settings.control_api_key", "correct-key")

    client = TestClient(app)
    response = client.post("/control/capital-ramp/advance", headers={"X-API-Key": "correct-key"})
    assert response.status_code == 400
    assert "not eligible" in response.json()["detail"]
    assert capital_ramp_manager.stage == CapitalStage.DEMO  # unchanged


def test_capital_ramp_advance_succeeds_when_eligible(tmp_path, monkeypatch):
    import sqlite3
    from app.api.server import capital_ramp_manager
    from app.risk.capital_ramp import CapitalStage
    from app.monitoring.trade_log import record_trade

    db_path = str(tmp_path / "test.db")
    conn = sqlite3.connect(db_path)
    with open("app/storage/schema.sql") as f:
        conn.executescript(f.read())
    conn.close()

    # Strong, reliable performance: well above break-even at an 80% payout, plenty of trades.
    for _ in range(65):
        record_trade(db_path, "frxEURUSD", "CALL", "WIN", 8.0, 10.0, 100.0)
    for _ in range(35):
        record_trade(db_path, "frxEURUSD", "CALL", "LOSS", -10.0, 10.0, 100.0)

    capital_ramp_manager.stage = CapitalStage.DEMO
    capital_ramp_manager.stage_entry_trade_id = 0
    capital_ramp_manager.min_trades_to_advance = 50
    monkeypatch.setattr("app.api.server.settings.db_path", db_path)
    monkeypatch.setattr("app.api.server.settings.control_api_key", "correct-key")

    client = TestClient(app)
    response = client.post("/control/capital-ramp/advance", headers={"X-API-Key": "correct-key"})
    assert response.status_code == 200
    assert response.json()["stage"] == "MICRO_LIVE"
    assert capital_ramp_manager.stage == CapitalStage.MICRO_LIVE
