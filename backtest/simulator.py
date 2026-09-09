"""
Backtest engine (spec Part 36). Walks a candle series chronologically —
never shuffled — building a MarketContext from only what's known "so far" at
each step, running a strategy against it, gating on probability edge + EV,
and resolving the outcome against a future candle close.

Contract semantics assumed here (flagged as an assumption, not invented from
nothing): a CALL wins if the close `duration_candles` candles later is
strictly higher than the entry close; a PUT wins if strictly lower. An exact
tie is scored as a LOSS — this mirrors a "higher/lower" contract, but the
real settlement rule should be confirmed against a live proposal/contract
before this number is trusted for anything beyond research.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from app.features.engine import build_market_context
from app.features.types import CandleSeries
from app.markets.context import MarketType
from app.risk.payout_math import evaluate_economics
from app.strategies.base import Direction


@dataclass
class TradeRecord:
    entry_index: int
    entry_timestamp: int
    direction: Direction
    model_probability: float
    payout_ratio: float
    stake: float
    probability_edge: float
    expected_value: float
    exit_index: int
    exit_timestamp: int
    entry_price: float
    exit_price: float
    result: str          # "WIN" or "LOSS"
    profit_loss: float
    regime: str = "UNKNOWN"   # the market regime at entry — enables regime-filtered analysis (Part 34/38)


@dataclass
class BacktestResult:
    trades: list[TradeRecord] = field(default_factory=list)
    no_trade_count: int = 0

    @property
    def total_trades(self) -> int:
        return len(self.trades)

    @property
    def wins(self) -> int:
        return sum(1 for t in self.trades if t.result == "WIN")

    @property
    def losses(self) -> int:
        return sum(1 for t in self.trades if t.result == "LOSS")

    @property
    def win_rate(self) -> float | None:
        return self.wins / self.total_trades if self.total_trades else None

    @property
    def average_model_ev(self) -> float | None:
        if not self.trades:
            return None
        return sum(t.expected_value for t in self.trades) / len(self.trades)

    @property
    def realized_ev_per_stake(self) -> float | None:
        """Actual average return per trade, as a fraction of stake — the number that matters, not the model's estimate."""
        if not self.trades:
            return None
        return sum(t.profit_loss / t.stake for t in self.trades) / len(self.trades)

    @property
    def total_profit_loss(self) -> float:
        return sum(t.profit_loss for t in self.trades)

    @property
    def profit_factor(self) -> float | None:
        gross_profit = sum(t.profit_loss for t in self.trades if t.profit_loss > 0)
        gross_loss = abs(sum(t.profit_loss for t in self.trades if t.profit_loss < 0))
        if gross_loss == 0:
            return None  # undefined (no losses) rather than a misleading infinity
        return gross_profit / gross_loss

    @property
    def equity_curve(self) -> list[float]:
        curve = [0.0]
        for t in self.trades:
            curve.append(curve[-1] + t.profit_loss)
        return curve

    @property
    def max_drawdown(self) -> float:
        curve = self.equity_curve
        peak = curve[0]
        max_dd = 0.0
        for v in curve:
            peak = max(peak, v)
            max_dd = max(max_dd, peak - v)
        return max_dd

    @property
    def longest_winning_streak(self) -> int:
        return _longest_streak(self.trades, "WIN")

    @property
    def longest_losing_streak(self) -> int:
        return _longest_streak(self.trades, "LOSS")


def _longest_streak(trades: list[TradeRecord], result: str) -> int:
    longest = current = 0
    for t in trades:
        if t.result == result:
            current += 1
            longest = max(longest, current)
        else:
            current = 0
    return longest


def run_backtest(
    symbol: str,
    market_type: MarketType,
    duration_s: int,
    timestamps: list[int],
    opens: list[float],
    highs: list[float],
    lows: list[float],
    closes: list[float],
    strategy,
    duration_candles: int,
    payout_ratio: float,
    stake: float = 1.0,
    min_probability_edge: float = 0.0,
    min_expected_value: float = 0.0,
    warmup: int = 60,
    available_contracts: list[str] | None = None,
) -> BacktestResult:
    """
    `warmup` candles are consumed before the first possible signal (features
    like EMA-200 need history) — those candles never generate a trade
    opportunity, matching a real system that can't trade before it has data.
    `duration_candles` is the contract duration expressed in candles of this
    series (e.g. a 5-minute contract on 1-minute candles is duration_candles=5).
    """
    available_contracts = available_contracts or ["CALL", "PUT"]
    n = len(closes)
    result = BacktestResult()

    last_entry_index_allowing_resolution = n - 1 - duration_candles
    for i in range(warmup, last_entry_index_allowing_resolution + 1):
        series = CandleSeries(
            symbol=symbol,
            duration_s=duration_s,
            timestamps=timestamps[: i + 1],
            opens=opens[: i + 1],
            highs=highs[: i + 1],
            lows=lows[: i + 1],
            closes=closes[: i + 1],
        )
        ctx = build_market_context(
            symbol=symbol,
            market_type=market_type,
            series=series,
            available_contracts=available_contracts,
            payout={"CALL": payout_ratio, "PUT": payout_ratio},
            data_quality_score=3,  # backtest data is assumed clean; live ingestion scores this for real
        )

        if ctx.regime not in getattr(strategy, "allowed_regimes", set()) and hasattr(strategy, "allowed_regimes"):
            result.no_trade_count += 1
            continue

        candidate = strategy.evaluate(ctx)
        if candidate.direction == Direction.NO_TRADE:
            result.no_trade_count += 1
            continue

        econ = evaluate_economics(candidate.model_probability, payout_ratio, stake)
        if econ.probability_edge < min_probability_edge or econ.expected_value < min_expected_value:
            result.no_trade_count += 1
            continue

        exit_index = i + duration_candles
        entry_price = closes[i]
        exit_price = closes[exit_index]

        if candidate.direction == Direction.CALL:
            win = exit_price > entry_price
        else:
            win = exit_price < entry_price

        profit_loss = stake * payout_ratio if win else -stake

        result.trades.append(
            TradeRecord(
                entry_index=i,
                entry_timestamp=timestamps[i],
                direction=candidate.direction,
                model_probability=candidate.model_probability,
                payout_ratio=payout_ratio,
                stake=stake,
                probability_edge=econ.probability_edge,
                expected_value=econ.expected_value,
                exit_index=exit_index,
                exit_timestamp=timestamps[exit_index],
                entry_price=entry_price,
                exit_price=exit_price,
                result="WIN" if win else "LOSS",
                profit_loss=profit_loss,
                regime=ctx.regime.value,
            )
        )

    return result
