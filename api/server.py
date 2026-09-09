"""
Minimal HTTP surface (spec Part 1's "FastAPI for optional local API/dashboard").

/health, /status, /strategies, and the report endpoints are read-only, no
auth. /control/* endpoints actually change what the bot is doing (start,
stop, switch strategy) and require the X-API-Key header to match
settings.control_api_key — control is disabled entirely (every /control/*
call returns 401) if that setting is empty, so there's no accidental
unauthenticated control surface from a default config.

Deliberately NOT controllable via this API, on purpose: LIVE_TRADING itself,
and any RiskGovernor limit. Those stay config-driven and code-enforced —
a UI that could flip live trading on remotely would defeat the entire
demo-first design this system was built around.

The trading loop writes into `latest_state` (a tiny in-process holder) and
reads `bot_controller` each tick; this app just reads/writes both. No
shared database, no locking needed beyond what a single asyncio event loop
already gives you for free.
"""
from __future__ import annotations

import asyncio
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from fastapi import FastAPI, Header, HTTPException
from fastapi.responses import FileResponse

from app.analytics.deployment_checklist import generate_deployment_checklist, render_deployment_checklist_text
from app.analytics.model_comparison_report import generate_model_comparison_report, render_model_comparison_text
from app.analytics.real_data_report import generate_real_data_report, load_candles_from_db, render_report_text
from app.analytics.trade_advisor import get_trade_advice
from app.backtest.simulator import run_backtest
from app.backtest.stress_testing import render_stress_test_text, run_stress_test_suite
from app.config.settings import settings
from app.control.bot_controller import BotController
from app.risk.capital_ramp import CapitalRampManager
from app.features.types import CandleSeries
from app.markets.context import MarketType
from app.markets.ws_client import DerivPublicClient
from app.monitoring.dashboard import DashboardState
from app.monitoring.trade_log import get_latest_trade_id, get_lifetime_stats
from app.monitoring.system_events_log import get_reliability_stats
from app.strategies.baselines import RandomStrategy, SimpleMomentumStrategy, SimpleTrendStrategy

app = FastAPI(title="Deriv Trading Bot Status")

_STATIC_DIR = Path(__file__).resolve().parent.parent / "static"


@app.get("/")
async def root():
    """Serves the trade-advisor frontend. Path anchored to this file's own location, not the CWD — see scripts/run_forward_test.py's docstring for why that distinction matters."""
    return FileResponse(_STATIC_DIR / "advisor.html")

_STRATEGY_FACTORIES = {
    "random": lambda train_range: RandomStrategy(seed=1),
    "simple_momentum": lambda train_range: SimpleMomentumStrategy(),
    "simple_trend": lambda train_range: SimpleTrendStrategy(),
}

bot_controller = BotController(
    default_strategy="simple_trend",
    strategy_factories={
        "random": lambda: RandomStrategy(seed=1),
        "simple_momentum": lambda: SimpleMomentumStrategy(),
        "simple_trend": lambda: SimpleTrendStrategy(),
    },
)

capital_ramp_manager = CapitalRampManager()


def verify_api_key(x_api_key: str | None = Header(default=None)) -> None:
    if not settings.control_api_key:
        raise HTTPException(status_code=401, detail="control API is disabled (CONTROL_API_KEY not set)")
    if x_api_key != settings.control_api_key:
        raise HTTPException(status_code=401, detail="invalid or missing X-API-Key header")


class LatestStateHolder:
    """Single mutable slot the trading loop writes to and the API reads from."""

    def __init__(self):
        self.state: DashboardState | None = None
        self.updated_at: str | None = None

    def update(self, state: DashboardState) -> None:
        self.state = state
        self.updated_at = datetime.now(timezone.utc).isoformat()


latest_state = LatestStateHolder()


@app.get("/health")
async def health():
    """Liveness only — deliberately does not say anything about trading state, just that the process is up."""
    return {"status": "ok"}


@app.get("/strategies")
async def strategies():
    """Read-only list of selectable strategies and which one is currently active."""
    return bot_controller.status()


@app.get("/trade-advice")
async def trade_advice(
    symbol: str = "frxEURUSD",
    duration_s: int = 60,
    strategy: str = "simple_trend",
    durations: str = "30,60,120,300",
    account_balance: float | None = None,
):
    """
    "What trade should I make right now, and for what duration?" Uses the
    LATEST locally-stored candles (kept fresh by the running trading loop's
    own polling — this does not fetch its own historical backfill) plus a
    LIVE proposal check per candidate duration, so the recommendation
    reflects the actual current payout, not a backtested assumption.
    This is advice only — nothing here places an order.
    """
    if strategy not in _STRATEGY_FACTORIES:
        return {"error": f"unknown strategy '{strategy}', choose one of {list(_STRATEGY_FACTORIES)}"}
    try:
        candidate_durations = [int(d.strip()) for d in durations.split(",") if d.strip()]
    except ValueError:
        return {"error": "durations must be a comma-separated list of integers (seconds)"}
    if not candidate_durations:
        return {"error": "at least one duration must be provided"}

    timestamps, opens, highs, lows, closes = load_candles_from_db(settings.db_path, symbol, duration_s, limit=300)
    if len(closes) < 120:
        return {"error": f"only {len(closes)} candles available locally for {symbol} — not enough warmup history yet (need at least 120)"}

    series = CandleSeries(symbol=symbol, duration_s=duration_s, timestamps=timestamps, opens=opens, highs=highs, lows=lows, closes=closes)
    strategy_instance = _STRATEGY_FACTORIES[strategy]((0, len(closes)))

    async with DerivPublicClient() as client:
        advice = await get_trade_advice(
            symbol=symbol, strategy=strategy_instance, series=series, public_client=client,
            candidate_durations_s=candidate_durations, currency="USD",
            min_probability_edge=settings.min_probability_edge, min_expected_value=settings.min_expected_value,
            account_balance=account_balance, risk_per_trade=settings.risk_per_trade, max_stake=settings.max_stake,
        )

    return {
        "symbol": advice.symbol,
        "regime": advice.regime,
        "action": advice.action,
        "direction": advice.direction,
        "model_probability": advice.model_probability,
        "confidence": advice.confidence,
        "recommended_duration_s": advice.recommended_duration_s,
        "recommended_payout_ratio": advice.recommended_payout_ratio,
        "probability_edge": advice.probability_edge,
        "expected_value": advice.expected_value,
        "suggested_stake": advice.suggested_stake,
        "reasoning": advice.reasoning,
        "duration_candidates": [
            {
                "duration_s": d.duration_s, "payout_ratio": d.payout_ratio,
                "break_even_probability": d.break_even_probability, "probability_edge": d.probability_edge,
                "expected_value": d.expected_value, "qualifies": d.qualifies, "reason": d.reason,
            }
            for d in advice.duration_candidates
        ],
    }


@app.post("/control/start")
async def control_start(x_api_key: str | None = Header(default=None)):
    verify_api_key(x_api_key)
    bot_controller.start()
    return bot_controller.status()


@app.post("/control/stop")
async def control_stop(x_api_key: str | None = Header(default=None)):
    verify_api_key(x_api_key)
    bot_controller.stop()
    return bot_controller.status()


@app.post("/control/select-strategy")
async def control_select_strategy(name: str, x_api_key: str | None = Header(default=None)):
    verify_api_key(x_api_key)
    try:
        bot_controller.select_strategy(name)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return bot_controller.status()


@app.get("/capital-ramp/status")
async def capital_ramp_status():
    """
    Read-only. Spec Part 52's staged capital ramp. Note: nothing in this
    codebase currently runs a LIVE trading loop for this to actually gate
    (scripts/run_forward_test.py refuses to run when LIVE_TRADING is true,
    by design) — this reports the manager's state honestly either way.
    """
    stage_stats = get_lifetime_stats(settings.db_path, since_id=capital_ramp_manager.stage_entry_trade_id)
    return {
        **capital_ramp_manager.status(),
        "trades_at_current_stage": stage_stats.trade_count,
        "pnl_at_current_stage": stage_stats.total_pnl,
    }


@app.post("/control/capital-ramp/advance")
async def control_capital_ramp_advance(x_api_key: str | None = Header(default=None)):
    """
    Deliberate action only — advancing capital stage is never automatic, by
    design (see capital_ramp.py's docstring). Actually checks eligibility
    first using break-even probability at the configured payout; refuses to
    advance if the real numbers don't support it, even with a valid API key.
    """
    verify_api_key(x_api_key)
    from app.risk.payout_math import break_even_probability

    stage_stats = get_lifetime_stats(settings.db_path, since_id=capital_ramp_manager.stage_entry_trade_id)
    eligibility = capital_ramp_manager.evaluate_advancement_eligibility(stage_stats, break_even_probability(0.8))
    if not eligibility.eligible:
        raise HTTPException(status_code=400, detail=f"not eligible to advance: {eligibility.reason}")

    latest_id = get_latest_trade_id(settings.db_path)
    capital_ramp_manager.advance(latest_id)
    return capital_ramp_manager.status()


@app.get("/status")
async def status():
    """
    Current dashboard snapshot as JSON. Returns a clear "not started yet"
    response rather than a 404/500 if the trading loop hasn't run a single
    tick yet — that's a normal state right after startup, not an error.
    """
    if latest_state.state is None:
        return {"status": "starting", "message": "no tick has completed yet"}
    return {
        "status": "running",
        "updated_at": latest_state.updated_at,
        "dashboard": asdict(latest_state.state),
    }


@app.get("/backtest-report")
async def backtest_report(
    symbol: str = "frxEURUSD",
    duration_s: int = 60,
    duration_candles: int = 1,
    payout_ratio: float = 0.8,
    strategy: str = "simple_trend",
    max_candles: int = 3000,
):
    """
    Runs the backtest/walk-forward/statistics engine against REAL accumulated
    candle data — the actual answer to "does this have edge," not a hunch
    from a handful of live trades. `strategy` selects among the Phase 7
    baselines (random/simple_momentum/simple_trend); this deliberately does
    not expose the ML model here since it hasn't been fit on real data yet
    (see /model-comparison-report for that).

    Offloaded to a thread via asyncio.to_thread: this is CPU-bound work
    (measured at roughly 6ms per candle) that shares this process's single
    event loop with the live trading loop — running it inline would stall
    trade ticks for the duration of the report. `max_candles` bounds this to
    a recent window so runtime stays predictable regardless of how much
    history has accumulated — see real_data_report.py for the measurement
    behind that default.
    """
    if strategy not in _STRATEGY_FACTORIES:
        return {"error": f"unknown strategy '{strategy}', choose one of {list(_STRATEGY_FACTORIES)}"}

    report = await asyncio.to_thread(
        generate_real_data_report,
        db_path=settings.db_path,
        symbol=symbol,
        duration_s=duration_s,
        strategy_factory=_STRATEGY_FACTORIES[strategy],
        duration_candles=duration_candles,
        payout_ratio=payout_ratio,
        market_type=MarketType.FOREX,
        max_candles=max_candles,
    )

    return {
        "symbol": report.symbol,
        "strategy": report.strategy_id,
        "n_candles": report.n_candles,
        "warning": report.warning,
        "total_trades": report.backtest_result.total_trades,
        "win_rate": report.backtest_result.win_rate,
        "realized_ev_per_stake": report.backtest_result.realized_ev_per_stake,
        "max_drawdown": report.backtest_result.max_drawdown,
        "is_statistically_reliable": report.significance.is_reliable,
        "p_value": report.significance.p_value,
        "wilson_ci": [report.significance.wilson_ci_low, report.significance.wilson_ci_high],
        "walk_forward_stability_score": report.walk_forward_report.stability_score,
        "text_report": render_report_text(report),
    }


@app.get("/model-comparison-report")
async def model_comparison_report(
    symbol: str = "frxEURUSD",
    duration_s: int = 60,
    duration_candles: int = 1,
    payout_ratio: float = 0.8,
    max_candles: int = 3000,
):
    """
    Champion/challenger comparison (spec Part 16/35): fits the logistic
    regression model on a chronological train split of REAL accumulated
    data, then tests it out-of-sample against every Phase 7 baseline on the
    same held-out region. `approved` in the response is only True if the
    candidate beats every baseline's EV AND is itself statistically
    reliable — the hard gate from models/baseline_comparison.py, unchanged
    here, just fed real data for the first time.
    """
    report = await asyncio.to_thread(
        generate_model_comparison_report,
        db_path=settings.db_path,
        symbol=symbol,
        duration_s=duration_s,
        duration_candles=duration_candles,
        payout_ratio=payout_ratio,
        market_type=MarketType.FOREX,
        max_candles=max_candles,
    )

    if report.candidate_result is None:
        return {"symbol": report.symbol, "n_candles": report.n_candles, "warning": report.warning, "text_report": render_model_comparison_text(report)}

    return {
        "symbol": report.symbol,
        "n_candles": report.n_candles,
        "n_train_examples": report.n_train_examples,
        "warning": report.warning,
        "candidate": {
            "total_trades": report.candidate_result.total_trades,
            "win_rate": report.candidate_result.win_rate,
            "realized_ev_per_stake": report.candidate_result.realized_ev_per_stake,
        },
        "baselines": {
            name: {"total_trades": r.total_trades, "win_rate": r.win_rate, "realized_ev_per_stake": r.realized_ev_per_stake}
            for name, r in report.baseline_results.items()
        },
        "approved": report.approval.approved,
        "is_statistically_reliable": report.approval.candidate_is_statistically_reliable,
        "reasons": report.approval.reasons,
        "text_report": render_model_comparison_text(report),
    }


def _stress_test_from_real_data(symbol, duration_s, duration_candles, payout_ratio, strategy, max_candles, starting_bankroll, n_simulations):
    """Runs synchronously — called via asyncio.to_thread from the endpoint below, same pattern as the other reports."""
    timestamps, opens, highs, lows, closes = load_candles_from_db(settings.db_path, symbol, duration_s, limit=max_candles)
    backtest_result = run_backtest(
        symbol=symbol, market_type=MarketType.FOREX, duration_s=duration_s,
        timestamps=timestamps, opens=opens, highs=highs, lows=lows, closes=closes,
        strategy=_STRATEGY_FACTORIES[strategy]((0, len(closes))), duration_candles=duration_candles, payout_ratio=payout_ratio,
    )
    if backtest_result.total_trades == 0:
        return len(closes), None
    suite = run_stress_test_suite(backtest_result, starting_bankroll=starting_bankroll, n_simulations=n_simulations, seed=1)
    return len(closes), suite


def _deployment_checklist_sync(symbol, duration_s, duration_candles, payout_ratio, strategy, max_candles, starting_bankroll, n_simulations):
    """Runs synchronously — called via asyncio.to_thread, same pattern as every other report."""
    real = generate_real_data_report(
        db_path=settings.db_path, symbol=symbol, duration_s=duration_s,
        strategy_factory=_STRATEGY_FACTORIES[strategy], duration_candles=duration_candles,
        payout_ratio=payout_ratio, market_type=MarketType.FOREX, max_candles=max_candles,
    )

    model = None
    stress = None
    if real.backtest_result.total_trades > 0:
        model = generate_model_comparison_report(
            db_path=settings.db_path, symbol=symbol, duration_s=duration_s,
            duration_candles=duration_candles, payout_ratio=payout_ratio,
            market_type=MarketType.FOREX, max_candles=max_candles,
        )
        # Reuses the SAME backtest result real_data_report already produced
        # (which now carries regime per trade) rather than re-running a
        # third backtest just for stress testing.
        stress = run_stress_test_suite(real.backtest_result, starting_bankroll=starting_bankroll, n_simulations=n_simulations, seed=1)

    checklist = generate_deployment_checklist(
        real, model, stress, settings, starting_bankroll,
        lifetime_stats=get_lifetime_stats(settings.db_path, symbol=symbol),
        reliability_stats=get_reliability_stats(settings.db_path),
    )
    return checklist


@app.get("/deployment-checklist")
async def deployment_checklist(
    symbol: str = "frxEURUSD",
    duration_s: int = 60,
    duration_candles: int = 1,
    payout_ratio: float = 0.8,
    strategy: str = "simple_trend",
    max_candles: int = 3000,
    starting_bankroll: float = 1000.0,
    n_simulations: int = 500,
):
    """
    Spec Part 51: the literal all-or-nothing gate. Combines Stage A
    (real_data_report), Stage B (model_comparison_report), and Stage C
    (stress_testing) into one verdict. `all_passed` is True only if every
    one of the 14 items is a literal PASS — see deployment_checklist.py's
    docstring for why NOT_AUTOMATED items block just as hard as FAIL does.
    This is the single most expensive endpoint in the whole system (runs
    all three prior stages), hence the same max_candles bound as the others.
    """
    if strategy not in _STRATEGY_FACTORIES:
        return {"error": f"unknown strategy '{strategy}', choose one of {list(_STRATEGY_FACTORIES)}"}

    checklist = await asyncio.to_thread(
        _deployment_checklist_sync, symbol, duration_s, duration_candles, payout_ratio, strategy, max_candles, starting_bankroll, n_simulations,
    )

    return {
        "all_passed": checklist.all_passed,
        "items": [
            {"name": i.name, "description": i.description, "status": i.status.value, "detail": i.detail}
            for i in checklist.items
        ],
        "text_report": render_deployment_checklist_text(checklist),
    }


@app.get("/stress-test-report")
async def stress_test_report(
    symbol: str = "frxEURUSD",
    duration_s: int = 60,
    duration_candles: int = 1,
    payout_ratio: float = 0.8,
    strategy: str = "simple_trend",
    max_candles: int = 3000,
    starting_bankroll: float = 1000.0,
    n_simulations: int = 500,
):
    """
    Spec Part 38: does the strategy remain survivable under adverse
    conditions, using real accumulated trades as the base sequence. Every
    scenario reuses the Monte Carlo engine from Phase 8. Two scenarios
    (high/low volatility regime) may report "no trades available" if the
    strategy hasn't fired in that regime yet — that's an honest gap in the
    sample, not a pass or a fail.
    """
    if strategy not in _STRATEGY_FACTORIES:
        return {"error": f"unknown strategy '{strategy}', choose one of {list(_STRATEGY_FACTORIES)}"}

    n_candles, suite = await asyncio.to_thread(
        _stress_test_from_real_data, symbol, duration_s, duration_candles, payout_ratio, strategy, max_candles, starting_bankroll, n_simulations,
    )

    if suite is None:
        return {"symbol": symbol, "n_candles": n_candles, "warning": "no trades in the backtest yet — cannot run stress scenarios"}

    return {
        "symbol": symbol,
        "n_candles": n_candles,
        "strategy": strategy,
        "all_survived": suite.all_survived,
        "scenarios": [
            {
                "name": s.name,
                "description": s.description,
                "n_trades": s.n_trades,
                "survived": s.survived,
                "note": s.note,
                "probability_of_ruin": s.monte_carlo.probability_of_ruin if s.monte_carlo else None,
                "expected_ending_bankroll": s.monte_carlo.expected_ending_bankroll if s.monte_carlo else None,
            }
            for s in suite.scenarios
        ],
        "text_report": render_stress_test_text(suite),
    }
