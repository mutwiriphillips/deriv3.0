"""
Bot controller — the actual "agent" surface. Everything built before this
point (dashboard, reports) was read-only monitoring; this is the first
piece that lets something external actually change what the bot is doing
while it runs, rather than restarting the process with different hardcoded
constants.

Deliberately minimal: start/stop/select-strategy only. Nothing here can
override LIVE_TRADING or the risk governor's limits — those stay
config-driven (settings.py) and code-enforced (RiskGovernor), not
controllable via this surface, on purpose. A UI that could flip live
trading on remotely would defeat the entire demo-first design this system
was built around.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable


@dataclass
class BotController:
    default_strategy: str
    strategy_factories: dict[str, Callable[[], object]]
    running: bool = True
    active_strategy_name: str = field(init=False)

    def __post_init__(self):
        if self.default_strategy not in self.strategy_factories:
            raise ValueError(f"default_strategy '{self.default_strategy}' not in strategy_factories")
        self.active_strategy_name = self.default_strategy

    def get_active_strategy(self):
        return self.strategy_factories[self.active_strategy_name]()

    def start(self) -> None:
        self.running = True

    def stop(self) -> None:
        self.running = False

    def select_strategy(self, name: str) -> None:
        if name not in self.strategy_factories:
            raise ValueError(f"unknown strategy '{name}', choose one of {list(self.strategy_factories)}")
        self.active_strategy_name = name

    def status(self) -> dict:
        return {
            "running": self.running,
            "active_strategy": self.active_strategy_name,
            "available_strategies": list(self.strategy_factories),
        }
