"""
Trade advisor: "given the current market, what trade should I make, and for
what duration?" Reuses everything already built — features, the strategy
registry, and risk/payout_math's economics — rather than inventing a
separate signal path. This is ADVICE only: nothing here places an order.

Duration IS treated as a real parameter to check (spec Part 23), not a fixed
assumption — the strategy determines direction and probability once (a
directional view doesn't change with contract length), but the ACTUAL
payout differs per duration, so multiple live proposals are checked to find
which duration currently offers a genuine edge, if any.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from app.features.engine import build_market_context
from app.features.types import CandleSeries
from app.markets.context import MarketType
from app.markets.ws_client import DerivPublicClient
from app.risk.payout_math import evaluate_economics
from app.risk.stake_sizing import calculate_stake
from app.strategies.base import Direction


@dataclass
class DurationCandidate:
    duration_s: int
    payout_ratio: float | None
    break_even_probability: float | None
    probability_edge: float | None
    expected_value: float | None
    qualifies: bool
    reason: str | None = None   # why it doesn't qualify, when it doesn't


@dataclass
class TradeAdvice:
    symbol: str
    regime: str
    action: str                 # "TRADE" or "NO_TRADE"
    direction: str | None = None
    model_probability: float | None = None
    confidence: float | None = None
    recommended_duration_s: int | None = None
    recommended_payout_ratio: float | None = None
    probability_edge: float | None = None
    expected_value: float | None = None
    suggested_stake: float | None = None
    reasoning: list[str] = field(default_factory=list)
    duration_candidates: list[DurationCandidate] = field(default_factory=list)


async def get_trade_advice(
    symbol: str,
    strategy,
    series: CandleSeries,
    public_client: DerivPublicClient,
    candidate_durations_s: list[int],
    currency: str = "USD",
    min_probability_edge: float = 0.0,
    min_expected_value: float = 0.0,
    account_balance: float | None = None,
    risk_per_trade: float = 0.01,
    max_stake: float = 10.0,
    market_type: MarketType = MarketType.FOREX,
) -> TradeAdvice:
    """
    `series` must already be sliced to "now" by the caller (same
    no-look-ahead contract as everywhere else). `candidate_durations_s` are
    checked as actual contract durations in seconds against LIVE proposals —
    this makes a real network call per duration checked, so keep the list
    reasonably short (3-5 durations is plenty).
    """
    ctx = build_market_context(
        symbol=symbol, market_type=market_type, series=series,
        available_contracts=["CALL", "PUT"], data_quality_score=3,
    )
    candidate = strategy.evaluate(ctx)

    if candidate.direction == Direction.NO_TRADE:
        return TradeAdvice(
            symbol=symbol, regime=ctx.regime.value, action="NO_TRADE",
            reasoning=[f"strategy produced no signal: {candidate.reasons}"],
        )

    duration_results: list[DurationCandidate] = []
    for duration_s in candidate_durations_s:
        try:
            proposal = await public_client.proposal(
                contract_type=candidate.direction.value,
                underlying_symbol=symbol,
                currency=currency,
                amount=max_stake,   # ask_price influences payout ratio negligibly for a ratio quote; a fixed probe amount is fine
                duration=duration_s,
                duration_unit="s",
            )
        except Exception as e:
            duration_results.append(DurationCandidate(
                duration_s=duration_s, payout_ratio=None, break_even_probability=None,
                probability_edge=None, expected_value=None, qualifies=False,
                reason=f"proposal request failed: {e!r}",
            ))
            continue

        if "ask_price" not in proposal or "payout" not in proposal:
            duration_results.append(DurationCandidate(
                duration_s=duration_s, payout_ratio=None, break_even_probability=None,
                probability_edge=None, expected_value=None, qualifies=False,
                reason="proposal response missing pricing fields",
            ))
            continue

        ask_price = float(proposal["ask_price"])
        payout = float(proposal["payout"])
        payout_ratio = (payout - ask_price) / ask_price if ask_price else 0.0
        econ = evaluate_economics(candidate.model_probability, payout_ratio, ask_price)

        qualifies = econ.probability_edge >= min_probability_edge and econ.expected_value >= min_expected_value
        reason = None
        if not qualifies:
            if econ.probability_edge < min_probability_edge:
                reason = f"edge {econ.probability_edge:.4f} below minimum {min_probability_edge}"
            else:
                reason = f"EV {econ.expected_value:.4f} below minimum {min_expected_value}"

        duration_results.append(DurationCandidate(
            duration_s=duration_s, payout_ratio=payout_ratio,
            break_even_probability=econ.break_even_probability,
            probability_edge=econ.probability_edge, expected_value=econ.expected_value,
            qualifies=qualifies, reason=reason,
        ))

    qualifying = [d for d in duration_results if d.qualifies]
    if not qualifying:
        return TradeAdvice(
            symbol=symbol, regime=ctx.regime.value, action="NO_TRADE",
            direction=candidate.direction.value, model_probability=candidate.model_probability,
            confidence=candidate.confidence,
            reasoning=[f"direction {candidate.direction.value} identified but no duration currently clears the edge/EV bar"],
            duration_candidates=duration_results,
        )

    best = max(qualifying, key=lambda d: d.expected_value)
    stake = calculate_stake(account_balance, risk_per_trade, max_stake) if account_balance is not None else None

    return TradeAdvice(
        symbol=symbol, regime=ctx.regime.value, action="TRADE",
        direction=candidate.direction.value, model_probability=candidate.model_probability,
        confidence=candidate.confidence,
        recommended_duration_s=best.duration_s, recommended_payout_ratio=best.payout_ratio,
        probability_edge=best.probability_edge, expected_value=best.expected_value,
        suggested_stake=stake,
        reasoning=[
            f"{candidate.direction.value} on {symbol}, regime={ctx.regime.value}",
            f"model probability {candidate.model_probability:.4f}, edge {best.probability_edge:.4f} at {best.duration_s}s duration",
        ] + candidate.reasons,
        duration_candidates=duration_results,
    )
