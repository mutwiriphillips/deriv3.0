# Deriv Multi-Market Binary Trading System (Forex-First)

**Status: PHASE 1 — Architecture & Interfaces. No API calls, no trading logic yet.**

## 0. Governing principle

This is an **edge-detection and risk-management system**, not a trade generator.
Default output for every tick evaluated is `NO_TRADE`. A trade is only ever the
*exception* that survives a strict filter pipeline (see §5).

---

## 1. Architecture diagram (text)

```
                        ┌─────────────────────────┐
                        │        Deriv API         │
                        │  (WebSocket + REST)       │
                        └────────────┬─────────────┘
                                     │ ticks / active_symbols / contracts_for
                                     ▼
                        ┌─────────────────────────┐
                        │   markets/ (Market       │
                        │   Registry + Discovery)  │
                        └────────────┬─────────────┘
                                     │ MarketContext (normalized)
                                     ▼
        ┌───────────────┐   ┌───────────────┐   ┌───────────────┐
        │  data/         │──▶│ features/      │──▶│ probability/   │
        │  (ticks,       │   │ (trend, mom,   │   │ (calibrated    │
        │  candles, DQ)  │   │  vol, regime)  │   │  P(win) model) │
        └───────────────┘   └───────────────┘   └───────┬───────┘
                                                          │
                                                          ▼
                                                ┌───────────────────┐
                                                │  strategies/        │
                                                │  (Strategy Registry,│
                                                │   signal_score)     │
                                                └─────────┬───────────┘
                                                           │ TradeCandidate
                                                           ▼
                                                ┌───────────────────┐
                                                │  Trade Filter        │
                                                │  Pipeline (§5)       │
                                                │  → TRADE / NO_TRADE  │
                                                └─────────┬───────────┘
                                        TRADE ─────────────┤ NO_TRADE ──▶ storage/
                                                            ▼               (reason logged)
                                                ┌───────────────────┐
                                                │  risk/               │
                                                │  (stake sizing,      │
                                                │   exposure, limits)  │
                                                └─────────┬───────────┘
                                                           ▼
                                                ┌───────────────────┐
                                                │  execution/          │
                                                │  (proposal/buy,       │
                                                │   idempotency)        │
                                                └─────────┬───────────┘
                                                           ▼
                                                ┌───────────────────┐
                                                │  portfolio/ +        │
                                                │  analytics/          │
                                                │  (P&L, decay         │
                                                │   detection)         │
                                                └─────────┬───────────┘
                                                           ▼
                                                ┌───────────────────┐
                                                │  monitoring/ +       │
                                                │  alerts/ + api/      │
                                                │  (dashboard, kill    │
                                                │   switch)            │
                                                └───────────────────┘

  Offline/parallel path:
  data/ (historical store) ──▶ backtest/ ──▶ walk-forward validation ──▶
  strategy_runs (RESEARCH → BACKTESTING → VALIDATION → DEMO → APPROVED)
```

## 2. Component descriptions

| Module | Responsibility |
|---|---|
| `config/` | Pydantic settings loaded from `.env`; single source of truth for every threshold (`MIN_PROBABILITY_EDGE`, `MAX_DAILY_LOSS`, etc.). Nothing thresholds-related is hardcoded in strategy code. |
| `markets/` | Talks to `active_symbols`, `contracts_for`. Builds/refreshes the `MarketRegistry`. Classifies symbols into FOREX / SYNTHETIC / COMMODITY / OTHER. |
| `data/` | Live tick ingestion, candle aggregation, historical store, data-quality scoring (0–3), stale/duplicate/out-of-order detection. |
| `features/` | Stateless feature functions (trend, momentum, volatility, structure, tick, time, regime) each independently backtestable/toggleable. |
| `probability/` | Calibrated `P(CALL)/P(PUT)/P(NO_TRADE)` model + uncertainty (sample size, CI) per symbol×duration×session×regime. |
| `strategies/` | Strategy Registry; each strategy declares allowed regimes/symbols/durations and produces a `TradeCandidate` (direction, probability, confidence) — never raw indicator rules. |
| `risk/` | Break-even probability from actual payout, edge/EV calculation, fixed-fractional stake sizing, exposure (currency-correlation) manager, daily/session risk governor, kill switch. |
| `execution/` | Proposal/buy calls, idempotent trade IDs, reconnect/backoff, latency measurement. |
| `backtest/` | Simulates binary contract economics candle-by-candle with no look-ahead; walk-forward windows; Monte Carlo; stress tests. |
| `portfolio/` | Open contract tracking, realized/unrealized P&L, drawdown, correlated exposure state. |
| `analytics/` | Rolling win-rate/EV/calibration by symbol×duration×regime; performance-decay detection; champion/challenger comparison. |
| `monitoring/` | Dashboard/CLI state (balance, P/L, current regime, edge, no-trade reason). |
| `alerts/` | Pluggable notification sinks (stub now; Telegram/email later). |
| `storage/` | DB access layer (see §4 schema) — every trade must be reconstructable: "what did the bot know right before this trade?" |
| `api/` | Optional FastAPI surface for the dashboard/local control. |
| `tests/` | Pytest suite, one subpackage per module above. |

## 3. Data-flow summary

1. `markets/` discovers the tradable universe at startup → `MarketRegistry`.
2. `data/` streams/aggregates ticks → candles, tagged with a `DataQualityScore`.
3. `features/` converts (candles, ticks) at time *t* into a `MarketContext` using only information available at *t* (no look-ahead — enforced by only ever slicing history `[:t]`).
4. `strategies/` (per the Strategy Registry's allowed regimes) turn `MarketContext` → `TradeCandidate(direction, model_probability, confidence)`.
5. `risk/` computes `break_even_probability` from the live payout, then `probability_edge` and `expected_value`.
6. The **Trade Filter Pipeline** (§5) evaluates every gate; the vast majority of candidates should terminate in a logged `NO_TRADE_<reason>`.
7. Surviving candidates go through `risk/` stake sizing and exposure checks, then `execution/`.
8. Every decision (trade or no-trade) is persisted so it's reproducible later.

## 4. Database schema (initial DDL)

See `storage/schema.sql`. Tables: `markets, ticks, candles, features, signals, proposals, trades, contracts, strategy_runs, model_versions, risk_events, system_events, performance_snapshots`.

## 5. Trade filter pipeline (order matters — first failure short-circuits to NO_TRADE)

```
symbol_available → contract_available → data_quality_ok → market_open →
payout_acceptable → regime_acceptable → signal_valid → probability_sufficient →
statistical_confidence_sufficient → expected_value_positive → risk_limits_ok →
cooldown_satisfied → no_duplicate_signal → api_healthy → execution_latency_acceptable
→ TRADE
```
Any failure emits one of the `NO_TRADE_*` reason codes (see `app/monitoring/reasons.py`).

## 6. Strategy interface

See `app/strategies/base.py`. Every strategy — Forex, Synthetic, or Generic — implements
the same `Strategy` protocol against `MarketContext`; no market-specific assumptions leak
into the interface itself.

## 7. Risk-management design

- Fixed-fractional stake only (`stake = balance × RISK_PER_TRADE`, clamped to `MAX_STAKE`). No Martingale, ever.
- `RiskGovernor` scales exposure down (never up) under drawdown/uncertainty; can force `EMERGENCY_STOP`.
- `ExposureManager` groups positions by currency (USD/EUR/GBP/JPY/...) to catch correlated Forex exposure.

## 8. Backtesting methodology

- Chronological only — never shuffled. Walk-forward windows (train → validate → test, roll forward).
- Simulates actual binary payout economics per trade, not just directional accuracy.
- Locked final test set, untouched until a strategy version is frozen (data-snoopage discipline).
- Monte Carlo resampling of trade sequences for drawdown/ruin probability; explicit stress tests (§38 of the spec).

## 9. API integration plan

- Phase 2 will pull the *current* Deriv WebSocket schema directly from https://developers.deriv.com/docs/ at
  implementation time (not from memory) for: `authorize`, `active_symbols`, `contracts_for`, `ticks`,
  `ticks_history`, `proposal`, `buy`, `proposal_open_contract`.
- No endpoint, field, or contract type is invented — anything uncertain gets flagged rather than guessed.

## 10. Testing plan

Per-module pytest suites under `tests/`, mirroring the module list in §2, plus integration tests for:
reconnect/backoff, idempotent buy retries, EV/break-even math, drawdown/daily-loss triggers, and the
full filter pipeline (including that a single failing gate blocks the trade).

## 11. Development order (locked, from spec Part 56)

Architecture (this doc) → Deriv API integration → Market discovery → Historical data engine →
Forex data pipeline → Feature engine → Baseline strategies → Backtesting engine → Statistical
evaluation → Walk-forward validation → Probability model → Risk engine → Demo execution →
Monitoring/dashboard → Forward demo testing → Micro-live safety testing → (only then) Live deployment.

**Nothing beyond Phase 1 is implemented yet.** Next phase (Phase 2) starts with the current Deriv
API schema and requires a Deriv account (demo is fine) with an API token to proceed.

## 12. Deployment (Render)

This deploys as a Render **web service** — `scripts/run_forward_test.py` runs the trading loop and
a minimal read-only HTTP surface (`/health`, `/status`, see `app/api/server.py`) concurrently in
the same process, so Render's health checks and external monitoring both work. `/status` returns
the current dashboard snapshot as JSON; there are deliberately no control endpoints (stop trading,
reset emergency stop) exposed over HTTP without real auth — that's a reasonable future addition,
not an oversight.

**Blueprint deploy (recommended):** push this repo to a Git provider, then in the Render
dashboard choose New → Blueprint and point it at the repo. `render.yaml` provisions everything —
build command, start command, every config threshold as an env var, and a persistent disk for
the candle store and daily risk/session state. You'll be prompted to fill in `DERIV_API_TOKEN` and
`DERIV_APP_ID` yourself; they're marked `sync: false` so they're never written into the blueprint file.

**Docker deploy (alternative):** set the service's Runtime to Docker instead — the included
`Dockerfile` builds the same app and runs the same start command.

**State persistence:** `RiskGovernor` and `SessionStats` are saved to the same disk-backed SQLite
file used for candles after every tick (`app/monitoring/state_persistence.py`), keyed by UTC date.
A restart resumes today's P/L, consecutive-loss count, and (importantly) an active emergency-stop
state — it does not silently forget you were stopped for a real reason. A new calendar day has no
saved row yet, which is what triggers the daily reset naturally.

**Known caveats, stated plainly rather than glossed over:**
- Render Disks require a paid instance tier — check current Render pricing before assuming this
  works on a free plan. Without the disk, both candle history and daily state are lost on every
  restart, same as before this was added.
- The `/status` endpoint is read-only and unauthenticated by design — don't add a write/control
  endpoint to this file without adding real authentication first.
- `render.yaml` sets `MODE=DEMO` and `LIVE_TRADING=false` deliberately — the script itself also
  refuses to run at all if `LIVE_TRADING` is true, so switching to live mode requires deliberately
  overriding both the blueprint and the script's own check, not just one dashboard toggle.

## 13. Control API (the "agent" surface)

Everything through Phase 15 was read-only monitoring. `app/control/bot_controller.py`
adds the first real control surface: **start**, **stop**, and **switch strategy** at
runtime, without restarting the process.

**Read-only, no auth:**
- `GET /strategies` — lists selectable strategies and which one is currently active

**Requires the `X-API-Key` header to match `CONTROL_API_KEY`** (control is entirely
disabled — every call returns 401 — if that env var isn't set):
- `POST /control/start`
- `POST /control/stop`
- `POST /control/select-strategy?name=<random|simple_momentum|simple_trend>`

**Deliberately NOT controllable via this API:** `LIVE_TRADING` itself, and any
`RiskGovernor` limit (`MAX_DAILY_LOSS`, `MAX_DRAWDOWN`, etc.). Those stay
config-driven (env vars) and code-enforced, on purpose — a remote "flip live trading
on" lever would defeat the entire demo-first design this system was built around.

A strategy switch takes effect on the very next tick — `scripts/run_forward_test.py`'s
loop re-reads `bot_controller.get_active_strategy()` every iteration rather than
fixing it once at startup.

**Note on scope:** this is a programmatic control API, not a visual strategy builder.
Letting a user construct arbitrary custom logic via drag-and-drop blocks (like Deriv's
own Bot product) would need a separate block-to-strategy interpreter that doesn't
exist here — a reasonable future feature, not something folded into this control API.


"# deriv3.0" 
"# deriv3.0" 
