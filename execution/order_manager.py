"""
Order manager (ties together Phase 1's Strategy output, Phase 12's risk
engine, and this phase's proposal/buy calls). This is the first place in the
codebase where a trade candidate can actually become a real order — every
gate here exists because skipping it would violate an explicit spec
requirement, not because it seemed like good practice in general.
"""
from __future__ import annotations

import time
from dataclasses import dataclass

from app.execution.auth_client import DerivAuthenticatedClient
from app.markets.ws_client import DerivPublicClient
from app.monitoring.reasons import NoTradeReason
from app.risk.exposure import ExposureManager
from app.risk.governor import RiskGovernor
from app.risk.overtrading import CooldownTracker, DuplicateSignalGuard, RateLimiter
from app.risk.payout_math import evaluate_economics
from app.risk.stake_sizing import calculate_stake
from app.strategies.base import Direction, TradeCandidate


@dataclass
class ExecutionOutcome:
    status: str                    # "TRADE", "NO_TRADE", "ALREADY_HANDLED"
    reason: str | None = None      # a NoTradeReason value, when status == "NO_TRADE"
    contract_id: str | None = None
    buy_price: float | None = None
    payout: float | None = None
    signal_to_proposal_latency_ms: float | None = None
    proposal_to_execution_latency_ms: float | None = None
    total_execution_latency_ms: float | None = None


@dataclass
class OrderManagerConfig:
    currency: str
    risk_per_trade: float
    max_stake: float
    min_probability_edge: float
    min_expected_value: float
    max_latency_ms: float
    cooldown_seconds: float = 60.0


class OrderManager:
    def __init__(
        self,
        config: OrderManagerConfig,
        risk_governor: RiskGovernor,
        exposure_manager: ExposureManager,
        duplicate_guard: DuplicateSignalGuard,
        cooldown_tracker: CooldownTracker,
        rate_limiter: RateLimiter,
    ):
        self.config = config
        self.risk_governor = risk_governor
        self.exposure_manager = exposure_manager
        self.duplicate_guard = duplicate_guard
        self.cooldown_tracker = cooldown_tracker
        self.rate_limiter = rate_limiter

    def _no_trade(self, reason: NoTradeReason) -> ExecutionOutcome:
        return ExecutionOutcome(status="NO_TRADE", reason=reason.value)

    async def attempt_trade(
        self,
        signal_id: str,
        symbol: str,
        candidate: TradeCandidate,
        duration: int,
        duration_unit: str,
        balance: float,
        public_client: DerivPublicClient,
        auth_client: DerivAuthenticatedClient,
        now: float | None = None,
    ) -> ExecutionOutcome:
        now = time.monotonic() if now is None else now
        signal_perf_time = time.monotonic()   # always the real clock -- latency math must never use the simulatable `now`

        if candidate.direction == Direction.NO_TRADE:
            return self._no_trade(NoTradeReason.SIGNAL_INVALID)

        if self.duplicate_guard.has_already_traded(signal_id):
            return ExecutionOutcome(status="ALREADY_HANDLED")

        if not self.risk_governor.can_trade():
            return self._no_trade(NoTradeReason.RISK_LIMIT)

        if not self.cooldown_tracker.is_satisfied(symbol, now):
            return self._no_trade(NoTradeReason.COOLDOWN)

        if not self.rate_limiter.is_within_limits(now):
            return self._no_trade(NoTradeReason.RATE_LIMIT_EXCEEDED)

        if self.exposure_manager.would_exceed_limit(symbol, candidate.direction):
            return self._no_trade(NoTradeReason.CORRELATED_EXPOSURE)

        effective_risk = self.config.risk_per_trade * self.risk_governor.risk_multiplier
        stake = calculate_stake(balance, effective_risk, self.config.max_stake)
        if stake <= 0:
            return self._no_trade(NoTradeReason.RISK_LIMIT)

        signal_time = signal_perf_time
        proposal = await public_client.proposal(
            contract_type=candidate.direction.value,
            underlying_symbol=symbol,
            currency=self.config.currency,
            amount=stake,
            duration=duration,
            duration_unit=duration_unit,
        )
        proposal_time = time.monotonic()

        if "ask_price" not in proposal or "payout" not in proposal:
            # New API only guarantees `id` — a proposal missing pricing fields
            # can't be evaluated, so we don't guess at fallback values.
            return self._no_trade(NoTradeReason.SIGNAL_INVALID)

        ask_price = float(proposal["ask_price"])
        payout = float(proposal["payout"])
        payout_ratio = (payout - ask_price) / ask_price if ask_price else 0.0

        econ = evaluate_economics(candidate.model_probability, payout_ratio, ask_price)
        if econ.probability_edge < self.config.min_probability_edge:
            return self._no_trade(NoTradeReason.LOW_EDGE)
        if econ.expected_value < self.config.min_expected_value:
            return self._no_trade(NoTradeReason.NEGATIVE_EV)

        signal_to_proposal_ms = (proposal_time - signal_time) * 1000
        if signal_to_proposal_ms > self.config.max_latency_ms:
            return self._no_trade(NoTradeReason.HIGH_LATENCY)

        # Mark BEFORE the buy call, not after — if the connection drops right
        # after we send this and we never see the response, we must not
        # blindly retry (spec Part 29). A caller that reconnects should check
        # portfolio/proposal_open_contract for this signal_id's outcome
        # rather than calling attempt_trade again for the same signal_id.
        self.duplicate_guard.mark_traded(signal_id)

        buy_result = await auth_client.buy(proposal["id"], ask_price)
        execution_time = time.monotonic()

        self.cooldown_tracker.record_trade(symbol, now)
        self.rate_limiter.record_trade(now)
        contract_id = str(buy_result.get("contract_id", ""))
        self.exposure_manager.open_position(contract_id or signal_id, symbol, candidate.direction)

        return ExecutionOutcome(
            status="TRADE",
            contract_id=contract_id,
            buy_price=float(buy_result.get("buy_price", ask_price)),
            payout=float(buy_result.get("payout", payout)),
            signal_to_proposal_latency_ms=signal_to_proposal_ms,
            proposal_to_execution_latency_ms=(execution_time - proposal_time) * 1000,
            total_execution_latency_ms=(execution_time - signal_time) * 1000,
        )
