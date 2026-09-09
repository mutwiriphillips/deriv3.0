"""
Live dashboard (spec Part 39) and no-trade explanation (spec Part 40).
`build_dashboard_state` pulls together what already exists elsewhere
(RiskGovernor, SessionStats, the current MarketContext/candidate/economics)
into one snapshot; `render_dashboard_text` turns that into the actual
CLI/log output. Every field that isn't currently known renders as "N/A"
rather than a misleading 0 or blank.
"""
from __future__ import annotations

from dataclasses import dataclass

from app.risk.governor import RiskGovernor, RiskState
from app.monitoring.session_stats import SessionStats

_BOT_STATUS_BY_RISK_STATE = {
    RiskState.NORMAL: "RUNNING",
    RiskState.REDUCED_CONFIDENCE: "RUNNING",
    RiskState.DRAWDOWN_WARNING: "RUNNING",
    RiskState.MAJOR_ANOMALY: "PAUSED",
    RiskState.EMERGENCY_STOP: "EMERGENCY_STOP",
}


@dataclass
class DashboardState:
    bot_status: str
    account_balance: float
    daily_pnl: float
    session_pnl: float
    drawdown_fraction: float
    active_strategy: str
    model_version: str
    api_status: str
    current_symbol: str | None = None
    current_regime: str | None = None
    signal_probability: float | None = None
    break_even_probability: float | None = None
    edge: float | None = None
    expected_value: float | None = None
    stake: float | None = None
    payout: float | None = None
    latency_ms: float | None = None
    trades_today: int = 0
    win_rate: float | None = None
    current_losing_streak: int = 0
    no_trade_reason: str | None = None


def build_dashboard_state(
    risk_governor: RiskGovernor,
    session_stats: SessionStats,
    api_status: str,
    active_strategy: str,
    model_version: str,
    current_symbol: str | None = None,
    current_regime: str | None = None,
    signal_probability: float | None = None,
    break_even_probability: float | None = None,
    edge: float | None = None,
    expected_value: float | None = None,
    stake: float | None = None,
    payout: float | None = None,
    latency_ms: float | None = None,
    no_trade_reason: str | None = None,
) -> DashboardState:
    return DashboardState(
        bot_status=_BOT_STATUS_BY_RISK_STATE[risk_governor.state],
        account_balance=risk_governor.current_balance,
        daily_pnl=-risk_governor.daily_loss,   # daily_loss is a non-negative magnitude; P/L sign matters for a dashboard
        session_pnl=session_stats.session_pnl,
        drawdown_fraction=risk_governor.drawdown_fraction,
        active_strategy=active_strategy,
        model_version=model_version,
        api_status=api_status,
        current_symbol=current_symbol,
        current_regime=current_regime,
        signal_probability=signal_probability,
        break_even_probability=break_even_probability,
        edge=edge,
        expected_value=expected_value,
        stake=stake,
        payout=payout,
        latency_ms=latency_ms,
        trades_today=session_stats.trades_today,
        win_rate=session_stats.win_rate,
        current_losing_streak=session_stats.current_losing_streak,
        no_trade_reason=no_trade_reason,
    )


def _fmt(value, suffix: str = "", precision: int = 4) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, float):
        return f"{value:.{precision}f}{suffix}"
    return f"{value}{suffix}"


def render_dashboard_text(state: DashboardState) -> str:
    lines = [
        f"BOT STATUS: {state.bot_status}",
        f"API STATUS: {state.api_status}",
        f"ACCOUNT BALANCE: {_fmt(state.account_balance, precision=2)}",
        f"DAILY P/L: {_fmt(state.daily_pnl, precision=2)}",
        f"SESSION P/L: {_fmt(state.session_pnl, precision=2)}",
        f"DRAWDOWN: {_fmt(state.drawdown_fraction * 100 if state.drawdown_fraction is not None else None, suffix='%', precision=2)}",
        f"ACTIVE STRATEGY: {state.active_strategy}",
        f"MODEL VERSION: {state.model_version}",
        f"CURRENT SYMBOL: {state.current_symbol or 'N/A'}",
        f"CURRENT REGIME: {state.current_regime or 'N/A'}",
        f"SIGNAL PROBABILITY: {_fmt(state.signal_probability)}",
        f"BREAK-EVEN PROBABILITY: {_fmt(state.break_even_probability)}",
        f"EDGE: {_fmt(state.edge)}",
        f"EXPECTED VALUE: {_fmt(state.expected_value)}",
        f"STAKE: {_fmt(state.stake, precision=2)}",
        f"PAYOUT: {_fmt(state.payout, precision=2)}",
        f"LATENCY: {_fmt(state.latency_ms, suffix='ms', precision=1)}",
        f"TRADES TODAY: {state.trades_today}",
        f"WIN RATE: {_fmt(state.win_rate * 100 if state.win_rate is not None else None, suffix='%', precision=1)}",
        f"LOSING STREAK: {state.current_losing_streak}",
    ]
    if state.no_trade_reason:
        lines.append(f"NO-TRADE REASON: {state.no_trade_reason}")
    return "\n".join(lines)
