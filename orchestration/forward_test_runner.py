"""
Forward-test orchestration (spec Part 20/56 Phase 15). `ForwardTestRunner`
is deliberately given already-built collaborators (strategy, order manager,
risk governor, session stats, clients) rather than constructing them itself
— composition over a monolithic setup function, and it makes every piece
independently swappable/mockable, same as everywhere else in this codebase.

This does NOT fetch its own candle data — `run_once` takes a CandleSeries
the caller already has (from data/historical_store.py + data/tick_ingestion.py
built in Phase 4). That keeps this module about orchestration logic, not
data-fetching mechanics, which are already built and tested elsewhere.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from app.execution.contract_monitor import poll_until_settled
from app.execution.order_manager import OrderManager
from app.features.engine import build_market_context
from app.features.types import CandleSeries
from app.markets.context import MarketType
from app.monitoring.dashboard import DashboardState, build_dashboard_state
from app.monitoring.session_stats import SessionStats
from app.risk.governor import RiskGovernor
from app.strategies.base import Direction


@dataclass
class ForwardTestRunner:
    symbol: str
    market_type: MarketType
    strategy: object            # anything satisfying the Strategy protocol
    model_version: str
    order_manager: OrderManager
    risk_governor: RiskGovernor
    session_stats: SessionStats
    public_client: object       # DerivPublicClient, kept loosely typed here to stay test-friendly
    auth_client: object         # DerivAuthenticatedClient
    duration_candles: int
    duration_s: int
    duration_unit: str = "s"
    available_contracts: tuple[str, ...] = ("CALL", "PUT")
    # Called with (symbol, direction, result, profit_loss, stake, latency_ms) on every
    # settlement. Optional and DB-agnostic on purpose — this module still doesn't touch
    # a database directly; the caller (scripts/run_forward_test.py) owns persistence.
    on_settlement: Callable[[str, str, str, float, float, float | None], None] | None = None
    # Called with (symbol, direction, stake, payout) the moment an order is placed,
    # BEFORE settlement is known — spec Part 54 treats TRADE_EXECUTED and
    # TRADE_SETTLED as distinct events, so this is a separate hook from on_settlement.
    on_trade_placed: Callable[[str, str, float, float], None] | None = None

    async def run_once(self, series: CandleSeries, signal_id: str) -> DashboardState:
        """
        One full tick against `series` (already sliced to "now" by the
        caller — same no-look-ahead contract as everywhere else features are
        computed). Returns the resulting dashboard snapshot; the caller
        decides what to do with it (print, log, both).
        """
        ctx = build_market_context(
            symbol=self.symbol,
            market_type=self.market_type,
            series=series,
            available_contracts=list(self.available_contracts),
            data_quality_score=3,   # forward-test data is assumed ingested via Phase 4's quality-scored pipeline
        )

        candidate = self.strategy.evaluate(ctx)

        outcome = await self.order_manager.attempt_trade(
            signal_id=signal_id,
            symbol=self.symbol,
            candidate=candidate,
            duration=self.duration_candles,
            duration_unit=self.duration_unit,
            balance=self.risk_governor.current_balance,
            public_client=self.public_client,
            auth_client=self.auth_client,
        )

        no_trade_reason = outcome.reason if outcome.status == "NO_TRADE" else None
        stake = None
        payout = None

        if outcome.status == "TRADE":
            stake = outcome.buy_price
            payout = outcome.payout
            if self.on_trade_placed is not None:
                self.on_trade_placed(self.symbol, candidate.direction.value, stake, payout)
            settlement = await poll_until_settled(self.auth_client, outcome.contract_id)
            if settlement.result is not None and settlement.profit is not None:
                self.risk_governor.record_trade_result(settlement.profit)
                self.session_stats.record_settlement(settlement.result, settlement.profit)
                if self.on_settlement is not None:
                    self.on_settlement(
                        self.symbol, candidate.direction.value, settlement.result,
                        settlement.profit, stake, outcome.total_execution_latency_ms,
                    )

        econ_probability = candidate.model_probability if candidate.direction != Direction.NO_TRADE else None

        return build_dashboard_state(
            risk_governor=self.risk_governor,
            session_stats=self.session_stats,
            api_status="CONNECTED",
            active_strategy=getattr(self.strategy, "strategy_id", type(self.strategy).__name__),
            model_version=self.model_version,
            current_symbol=self.symbol,
            current_regime=ctx.regime.value,
            signal_probability=econ_probability,
            stake=stake,
            payout=payout,
            latency_ms=outcome.total_execution_latency_ms,
            no_trade_reason=no_trade_reason,
        )
