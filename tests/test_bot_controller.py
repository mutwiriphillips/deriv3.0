import pytest

from app.control.bot_controller import BotController


def make_controller(default="simple_trend"):
    return BotController(
        default_strategy=default,
        strategy_factories={
            "random": lambda: "RANDOM_INSTANCE",
            "simple_trend": lambda: "TREND_INSTANCE",
        },
    )


def test_controller_starts_running_with_default_strategy():
    controller = make_controller()
    assert controller.running is True
    assert controller.active_strategy_name == "simple_trend"


def test_controller_rejects_unknown_default_strategy():
    with pytest.raises(ValueError):
        BotController(default_strategy="not_real", strategy_factories={"random": lambda: None})


def test_controller_stop_and_start():
    controller = make_controller()
    controller.stop()
    assert controller.running is False
    controller.start()
    assert controller.running is True


def test_controller_select_strategy_changes_active_and_get_active_returns_new_instance():
    controller = make_controller()
    controller.select_strategy("random")
    assert controller.active_strategy_name == "random"
    assert controller.get_active_strategy() == "RANDOM_INSTANCE"


def test_controller_select_unknown_strategy_raises_and_does_not_change_state():
    controller = make_controller()
    with pytest.raises(ValueError):
        controller.select_strategy("not_real")
    assert controller.active_strategy_name == "simple_trend"  # unchanged


def test_controller_status_reflects_current_state():
    controller = make_controller()
    controller.stop()
    controller.select_strategy("random")
    status = controller.status()
    assert status == {
        "running": False,
        "active_strategy": "random",
        "available_strategies": ["random", "simple_trend"],
    }


def test_get_active_strategy_calls_factory_fresh_each_time():
    """Each call should invoke the factory again, not cache a stale instance -- matters for stateful strategies."""
    calls = []
    controller = BotController(default_strategy="x", strategy_factories={"x": lambda: calls.append(1) or object()})
    controller.get_active_strategy()
    controller.get_active_strategy()
    assert len(calls) == 2
