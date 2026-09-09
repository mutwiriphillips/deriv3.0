"""
Runnable forward-test loop against a live Deriv DEMO account.

This is the first script in the whole project that actually talks to the
real API rather than a mock. Run it locally (it cannot run inside the
sandbox this was built in — no network access to api.derivws.com there).

WHAT THIS DOES, every poll_interval_s:
  1. Fetches the latest candles for `SYMBOL` (data/historical_store.py)
  2. Builds a MarketContext (features/engine.py)
  3. Runs `STRATEGY` against it
  4. If it trades: places a real DEMO order, polls until settled
  5. Prints the dashboard (monitoring/dashboard.py)

WHAT THIS DELIBERATELY DOES NOT DO:
  - Trade on a real-money account (MODE is hardcoded to DEMO below; there is
    no code path in this script that can reach the real endpoint)
  - Run unattended for long periods without you watching it — this is for
    supervised forward testing (spec Part 20), not a "start and forget" bot

BEFORE RUNNING:
  1. cp .env.example .env, fill in DERIV_API_TOKEN (your DEMO token) and
     DERIV_APP_ID
  2. pip install -r requirements.txt
  3. Confirm you have some historical candles already (or let step 1 below
     pull them - it will, the first time)

WHAT TO WATCH FOR (the flagged assumptions from Phase 13/15 that only a
real response can confirm):
  - Does get_ws_url_for_account() return a usable URL under the key this
    script expects? If not, it will raise KeyError immediately - check
    rest_client.get_ws_url_for_account and fix the field name.
  - Does the printed PROPOSAL RESPONSE (uncomment the debug print below)
    actually include ask_price/payout, and are duration/duration_unit/basis
    accepted? If Deriv returns an error about an unrecognized proposal
    field, that's app/markets/ws_client.py's proposal() to fix.
  - Does SETTLEMENT show up correctly (is_sold/status/profit) after a
    contract expires? If proposal_open_contract's shape differs, that's
    execution/contract_monitor.py to fix.
WHAT'S NEW IN THIS VERSION:
  - Serves a minimal read-only HTTP surface (/health, /status) via uvicorn,
    running concurrently with the trading loop in the same process. This is
    why Render deploys this as a `web` service now, not a `worker` — see
    render.yaml's comments.
  - Persists RiskGovernor/SessionStats state to the same SQLite file used
    for candles after every tick, and restores it on startup. This closes
    the "a restart forgets today's P/L" gap — see monitoring/state_persistence.py.
"""
from __future__ import annotations

import sys
from pathlib import Path

# Must happen before any `from app...` import below. Running this script
# directly (`python scripts/run_forward_test.py`) puts THIS FILE'S directory
# (scripts/) on sys.path, not the project root -- so the `app` package at the
# repo root is invisible unless we add it ourselves. This is exactly the bug
# that broke the first Render deploy (ModuleNotFoundError: No module named
# 'app') even though every local import check passed, because those checks
# used importlib.util.spec_from_file_location with sys.path already patched
# manually in the *test* snippet -- not the actual `python scripts/...`
# invocation Render (and any normal user) actually runs.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import asyncio
import os
import sys
import traceback

import uvicorn

from app.api.server import app as fastapi_app
from app.api.server import bot_controller, latest_state
from app.config.settings import settings
from app.data.historical_store import fetch_and_store_candles
from app.execution.auth_client import DerivAuthenticatedClient
from app.execution.order_manager import OrderManager, OrderManagerConfig
from app.features.types import CandleSeries
from app.markets.context import MarketType
from app.markets.rest_client import get_accounts, get_ws_url_for_account
from app.markets.ws_client import DerivPublicClient
from app.monitoring.dashboard import render_dashboard_text
from app.monitoring.session_stats import SessionStats
from app.monitoring.state_persistence import (
    restore_state_if_present,
    save_daily_state,
    today_utc_date_str,
)
from app.monitoring.trade_log import record_trade
from app.alerts.notifier import AlertManager, AlertType
from app.orchestration.forward_test_runner import ForwardTestRunner
from app.risk.exposure import ExposureManager
from app.risk.governor import RiskGovernor
from app.risk.overtrading import CooldownTracker, DuplicateSignalGuard, RateLimiter
from app.strategies.baselines import SimpleTrendStrategy

# ---- Hardcoded, deliberately conservative defaults for a first supervised run ----
SYMBOL = "frxEURUSD"
DURATION_S = 60              # 1-minute candles
DURATION_CANDLES = 1         # contract resolves 1 candle (i.e. 60s) after entry
POLL_INTERVAL_S = 60.0       # check for a new candle once a minute
STARTING_BALANCE_FOR_GOVERNOR = 1000.0   # the governor's OWN tracking; it does not read your real balance
DB_PATH = settings.db_path

alert_manager = AlertManager(db_path=DB_PATH)   # console sink by default; add a WebhookSink here for Slack/Discord/etc.


async def get_demo_ws_url() -> str:
    accounts = await get_accounts()
    # account_type with enum ["demo", "real"] is confirmed against Deriv's
    # published OpenAPI spec (used identically in the account-creation
    # request body); is_virtual is kept only as a fallback in case a given
    # account object omits account_type for some reason.
    demo_accounts = [
        a for a in accounts
        if a.get("account_type") == "demo" or a.get("is_virtual")
    ]
    if not demo_accounts:
        raise RuntimeError(
            f"No demo account found in get_accounts() response: {accounts}. "
            "Check the actual field name/value Deriv uses to mark an account as demo "
            "and fix this filter."
        )
    account_id = demo_accounts[0].get("account_id") or demo_accounts[0].get("id")
    if account_id is None:
        raise RuntimeError(f"Could not find an account id field in: {demo_accounts[0]}")
    return await get_ws_url_for_account(account_id)


def ensure_db_schema(db_path: str) -> None:
    """
    Applies app/storage/schema.sql before anything touches the database.
    Without this, the very first run crashes with "no such table: candles" —
    every test avoided this because the test helpers always applied the
    schema first; this script originally didn't, and would fail immediately
    on a fresh checkout with no forward_test.db yet. Safe to call every run:
    every CREATE in schema.sql is IF NOT EXISTS.
    """
    import sqlite3
    conn = sqlite3.connect(db_path)
    schema_path = Path(__file__).resolve().parent.parent / "app" / "storage" / "schema.sql"
    try:
        with open(schema_path) as f:
            conn.executescript(f.read())
        conn.commit()
    finally:
        conn.close()


def load_series_from_db(db_path: str, symbol: str, duration_s: int) -> CandleSeries:
    import sqlite3
    conn = sqlite3.connect(db_path)
    rows = conn.execute(
        "SELECT timestamp, open, high, low, close FROM candles WHERE symbol=? AND duration_s=? ORDER BY timestamp ASC",
        (symbol, duration_s),
    ).fetchall()
    conn.close()
    return CandleSeries(
        symbol=symbol,
        duration_s=duration_s,
        timestamps=[r[0] for r in rows],
        opens=[r[1] for r in rows],
        highs=[r[2] for r in rows],
        lows=[r[3] for r in rows],
        closes=[r[4] for r in rows],
    )


async def trading_loop(order_manager: OrderManager, runner: ForwardTestRunner, session_stats: SessionStats, ws_url: str):
    current_trade_date = today_utc_date_str()
    restored = restore_state_if_present(DB_PATH, current_trade_date, order_manager.risk_governor, session_stats)
    print(f"Restored today's saved state: {restored}")

    print(f"Starting forward test on {SYMBOL}. Press Ctrl+C to stop.\n")
    tick_count = 0
    last_alerted_event_count = len(order_manager.risk_governor.events)
    was_emergency_stopped = order_manager.risk_governor.is_emergency_stopped

    # RiskGovernor event_type strings map directly to AlertType values except
    # MANUAL_RESET, which has no spec Part 54 alert equivalent -- skipped below.
    _GOVERNOR_EVENT_TO_ALERT = {
        "DAILY_LIMIT_REACHED": AlertType.DAILY_LIMIT_REACHED,
        "DRAWDOWN_LIMIT_REACHED": AlertType.DRAWDOWN_LIMIT_REACHED,
        "CONSECUTIVE_LOSS_LIMIT": AlertType.CONSECUTIVE_LOSS_LIMIT,
        "EMERGENCY_STOP": AlertType.EMERGENCY_STOP,
    }

    while True:
        tick_count += 1

        if not bot_controller.running:
            print(f"--- tick {tick_count}: PAUSED (bot_controller.running=False) ---")
            await asyncio.sleep(POLL_INTERVAL_S)
            continue

        today = today_utc_date_str()
        if today != current_trade_date:
            print(f"New day detected ({current_trade_date} -> {today}): resetting daily P/L, keeping drawdown history.")
            order_manager.risk_governor.start_new_day()
            session_stats.reset_daily()
            current_trade_date = today

        # Re-fetched every tick, not just at startup, so a strategy switch via
        # POST /control/select-strategy takes effect on the very next tick
        # rather than requiring a restart.
        runner.strategy = bot_controller.get_active_strategy()
        runner.model_version = bot_controller.active_strategy_name

        await fetch_and_store_candles(DB_PATH, SYMBOL, DURATION_S, count=50)
        series = load_series_from_db(DB_PATH, SYMBOL, DURATION_S)

        if len(series) < 120:
            print(f"Only {len(series)} candles so far - waiting for enough warmup history...")
            await asyncio.sleep(POLL_INTERVAL_S)
            continue

        # Fresh connections per tick keep this simple and robust to a dropped
        # socket between ticks; a longer-running version would keep these
        # open and add reconnect/backoff (spec Part 30) instead.
        async with DerivPublicClient() as public_client:
            async with DerivAuthenticatedClient(ws_url) as auth_client:
                runner.public_client = public_client
                runner.auth_client = auth_client
                signal_id = f"{SYMBOL}-{series.timestamps[-1]}-{tick_count}"
                state = await runner.run_once(series, signal_id=signal_id)

        latest_state.update(state)
        save_daily_state(DB_PATH, current_trade_date, order_manager.risk_governor, session_stats)

        # Forward any new RiskGovernor events (drawdown/daily-loss/consecutive-loss
        # triggers) as alerts, without re-alerting on ones already seen.
        new_events = order_manager.risk_governor.events[last_alerted_event_count:]
        for event in new_events:
            alert_type = _GOVERNOR_EVENT_TO_ALERT.get(event.event_type)
            if alert_type is not None:
                alert_manager.notify(alert_type, f"RiskGovernor: {event.event_type}", details=event.details)
        last_alerted_event_count = len(order_manager.risk_governor.events)

        # Emergency stop is also surfaced via bot_status, alerted once on the
        # transition rather than every tick it stays stopped.
        if order_manager.risk_governor.is_emergency_stopped and not was_emergency_stopped:
            alert_manager.notify(AlertType.EMERGENCY_STOP, "Bot has entered EMERGENCY_STOP", details={"balance": order_manager.risk_governor.current_balance})
        was_emergency_stopped = order_manager.risk_governor.is_emergency_stopped

        print(f"--- tick {tick_count} ---")
        print(render_dashboard_text(state))
        print()

        await asyncio.sleep(POLL_INTERVAL_S)


async def initialize_bot_dependencies():
    """
    Retries schema init + candle fetch + WS URL resolution with exponential
    backoff (capped) until they succeed. Isolated from main() specifically
    so a Deriv API hiccup on startup can never prevent the HTTP server from
    binding — this is the actual fix for the 502 Bad Gateway a prior version
    of this script produced: it ran these network calls BEFORE ever
    constructing the uvicorn server, so any hang or failure here meant
    nothing was ever listening on $PORT for Render's health check to hit.
    """
    backoff = 5.0
    had_previously_failed = False
    while True:
        try:
            print("Initializing local database schema (if not already present)...")
            ensure_db_schema(DB_PATH)

            print("Ensuring local candle history exists / is up to date...")
            inserted = await fetch_and_store_candles(DB_PATH, SYMBOL, DURATION_S, count=1000)
            print(f"  inserted {inserted} new candles")

            print("Resolving demo account WebSocket URL...")
            ws_url = await get_demo_ws_url()

            if had_previously_failed:
                alert_manager.notify(AlertType.API_RECONNECTED, "Bot initialization succeeded after prior failure(s)")

            order_manager = OrderManager(
                config=OrderManagerConfig(
                    currency="USD",
                    risk_per_trade=settings.risk_per_trade,
                    max_stake=settings.max_stake,
                    min_probability_edge=settings.min_probability_edge,
                    min_expected_value=settings.min_expected_value,
                    max_latency_ms=settings.max_latency_ms,
                ),
                risk_governor=RiskGovernor(
                    starting_balance=STARTING_BALANCE_FOR_GOVERNOR,
                    max_daily_loss=settings.max_daily_loss,
                    max_drawdown=settings.max_drawdown,
                    max_consecutive_losses=settings.max_consecutive_losses,
                ),
                exposure_manager=ExposureManager(max_exposure_per_currency=2.0),
                duplicate_guard=DuplicateSignalGuard(),
                cooldown_tracker=CooldownTracker(cooldown_seconds=60.0),
                rate_limiter=RateLimiter(
                    max_trades_per_hour=settings.max_trades_per_hour,
                    max_trades_per_day=settings.max_trades_per_day,
                ),
            )
            session_stats = SessionStats()

            def _on_settlement(symbol, direction, result, profit, stake, latency):
                record_trade(DB_PATH, symbol, direction, result, profit, stake, latency)
                alert_manager.notify(
                    AlertType.TRADE_SETTLED, f"{direction} on {symbol}: {result}",
                    details={"profit_loss": profit, "stake": stake},
                )

            runner = ForwardTestRunner(
                symbol=SYMBOL,
                market_type=MarketType.FOREX,
                strategy=SimpleTrendStrategy(),   # deliberately a Phase 7 baseline for the first-ever live run, not the ML model
                model_version="baseline_simple_trend_v1",
                order_manager=order_manager,
                risk_governor=order_manager.risk_governor,
                session_stats=session_stats,
                public_client=None,   # set fresh each loop iteration (see note in trading_loop)
                auth_client=None,
                duration_candles=DURATION_CANDLES,
                duration_s=DURATION_S,
                on_trade_placed=lambda symbol, direction, stake, payout: alert_manager.notify(
                    AlertType.TRADE_EXECUTED, f"{direction} on {symbol}", details={"stake": stake, "payout": payout}
                ),
                on_settlement=_on_settlement,
            )
            return order_manager, runner, session_stats, ws_url

        except Exception as e:
            had_previously_failed = True
            alert_manager.notify(AlertType.API_DISCONNECTED, f"Bot initialization failed: {e!r}", details={"retry_in_s": backoff})
            print(f"Bot initialization failed ({e!r}); retrying in {backoff:.0f}s")
            traceback.print_exc()
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, 300.0)   # cap at 5 minutes between retries


async def bot_main_loop():
    """
    Everything Deriv-API-dependent lives inside this one coroutine, run
    alongside the HTTP server via asyncio.gather in main(). Catches
    literally everything so this coroutine can never raise and take the
    server down with it (asyncio.gather cancels sibling tasks the moment
    one raises) — worst case, the bot loop stalls and logs loudly, but
    /health and /status stay reachable throughout for diagnosis.
    """
    try:
        order_manager, runner, session_stats, ws_url = await initialize_bot_dependencies()
        await trading_loop(order_manager, runner, session_stats, ws_url)
    except Exception as e:
        print(f"FATAL, unrecoverable error in bot_main_loop: {e!r}")
        traceback.print_exc()
        print("The HTTP server will keep running so /health and /status remain reachable; "
              "the trading loop itself has stopped and needs a manual restart to recover.")
        while True:
            await asyncio.sleep(3600)


async def main():
    if settings.live_trading:
        print("LIVE_TRADING is true in .env - refusing to run this script. This script is DEMO-only by design.")
        sys.exit(1)

    port = int(os.environ.get("PORT", 8000))   # Render sets PORT for web services; defaults to 8000 for local runs
    server_config = uvicorn.Config(fastapi_app, host="0.0.0.0", port=port, log_level="warning")
    server = uvicorn.Server(server_config)
    print(f"HTTP status surface listening on :{port} (/health, /status)")

    await asyncio.gather(
        server.serve(),
        bot_main_loop(),
    )


if __name__ == "__main__":
    asyncio.run(main())
