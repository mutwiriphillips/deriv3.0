-- Phase 1 schema. SQLite-compatible now; portable to PostgreSQL later.
-- Every trade must be reproducible from stored information alone.

CREATE TABLE IF NOT EXISTS markets (
    symbol              TEXT PRIMARY KEY,
    display_name        TEXT,
    market              TEXT,          -- e.g. FOREX, SYNTHETIC, COMMODITY, OTHER
    submarket           TEXT,
    exchange            TEXT,
    available_contracts TEXT,          -- JSON array
    available_durations TEXT,          -- JSON array
    trading_hours       TEXT,          -- JSON
    tick_frequency      REAL,
    status              TEXT,          -- ACTIVE, SUSPENDED, REMOVED
    last_updated        TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS ticks (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol        TEXT NOT NULL REFERENCES markets(symbol),
    timestamp     TEXT NOT NULL,       -- ISO8601, source time
    price         REAL NOT NULL,
    tick_direction TEXT,               -- UP, DOWN, FLAT
    source        TEXT,
    received_at   TEXT NOT NULL,       -- local receipt time
    latency_ms    REAL,
    quality_score INTEGER NOT NULL DEFAULT 0  -- 0=unusable .. 3=high quality
);
CREATE INDEX IF NOT EXISTS idx_ticks_symbol_ts ON ticks(symbol, timestamp);

CREATE TABLE IF NOT EXISTS candles (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol      TEXT NOT NULL REFERENCES markets(symbol),
    duration_s  INTEGER NOT NULL,      -- candle size in seconds
    timestamp   INTEGER NOT NULL,      -- candle open time, epoch seconds
    open        REAL NOT NULL,
    high        REAL NOT NULL,
    low         REAL NOT NULL,
    close       REAL NOT NULL,
    volume      REAL,
    tick_count  INTEGER,
    UNIQUE(symbol, duration_s, timestamp)
);
CREATE INDEX IF NOT EXISTS idx_candles_symbol_dur_ts ON candles(symbol, duration_s, timestamp);

CREATE TABLE IF NOT EXISTS features (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol      TEXT NOT NULL,
    timestamp   TEXT NOT NULL,
    feature_set_version TEXT NOT NULL,
    payload     TEXT NOT NULL          -- JSON blob of computed features at time t (no look-ahead)
);
CREATE INDEX IF NOT EXISTS idx_features_symbol_ts ON features(symbol, timestamp);

CREATE TABLE IF NOT EXISTS model_versions (
    id              TEXT PRIMARY KEY,  -- e.g. "trend_v3"
    description     TEXT,
    created_at      TEXT NOT NULL,
    training_window TEXT,              -- JSON {start,end}
    baseline_beaten BOOLEAN,
    status          TEXT               -- RESEARCH, BACKTESTING, VALIDATION, DEMO, APPROVED, PAUSED, RETIRED
);

CREATE TABLE IF NOT EXISTS strategy_runs (
    id                  TEXT PRIMARY KEY,
    strategy_id         TEXT NOT NULL,
    market_type         TEXT,
    symbols             TEXT,          -- JSON array
    contract_types      TEXT,          -- JSON array
    durations           TEXT,          -- JSON array
    allowed_regimes     TEXT,          -- JSON array
    minimum_probability REAL,
    minimum_edge        REAL,
    minimum_ev          REAL,
    risk_profile        TEXT,
    model_version       TEXT REFERENCES model_versions(id),
    status              TEXT           -- RESEARCH, BACKTESTING, VALIDATION, DEMO, APPROVED, PAUSED, RETIRED
);

CREATE TABLE IF NOT EXISTS signals (
    id                    TEXT PRIMARY KEY,   -- signal_id
    strategy_run_id       TEXT REFERENCES strategy_runs(id),
    symbol                TEXT NOT NULL,
    timestamp             TEXT NOT NULL,
    direction             TEXT,               -- CALL, PUT, NO_TRADE
    model_probability     REAL,
    confidence            REAL,
    signal_score          REAL,
    features_id           INTEGER REFERENCES features(id),
    regime                TEXT,
    result                TEXT                -- TRADE or NO_TRADE
);

CREATE TABLE IF NOT EXISTS proposals (
    id                    TEXT PRIMARY KEY,
    signal_id             TEXT REFERENCES signals(id),
    symbol                TEXT NOT NULL,
    contract_type         TEXT NOT NULL,
    duration_s            INTEGER NOT NULL,
    stake                 REAL NOT NULL,
    payout                REAL NOT NULL,
    break_even_probability REAL NOT NULL,
    probability_edge       REAL NOT NULL,
    expected_value         REAL NOT NULL,
    no_trade_reason        TEXT,               -- populated when result != TRADE
    requested_at           TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS trades (
    id              TEXT PRIMARY KEY,   -- trade_id (internal, idempotency key)
    proposal_id     TEXT REFERENCES proposals(id),
    contract_id     TEXT,               -- Deriv contract_id, once known
    symbol          TEXT NOT NULL,
    direction       TEXT NOT NULL,
    stake           REAL NOT NULL,
    payout          REAL NOT NULL,
    signal_to_proposal_latency_ms   REAL,
    proposal_to_execution_latency_ms REAL,
    total_execution_latency_ms      REAL,
    executed_at     TEXT NOT NULL,
    settled_at      TEXT,
    result           TEXT,               -- WIN, LOSS, PENDING
    profit_loss      REAL
);

CREATE TABLE IF NOT EXISTS contracts (
    contract_id     TEXT PRIMARY KEY,
    trade_id        TEXT REFERENCES trades(id),
    raw_payload     TEXT                -- JSON snapshot of Deriv's contract object
);

CREATE TABLE IF NOT EXISTS risk_events (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp   TEXT NOT NULL,
    event_type  TEXT NOT NULL,          -- DAILY_LIMIT_REACHED, DRAWDOWN_LIMIT_REACHED, EMERGENCY_STOP, ...
    details     TEXT                    -- JSON
);

CREATE TABLE IF NOT EXISTS system_events (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp   TEXT NOT NULL,
    event_type  TEXT NOT NULL,          -- API_DISCONNECTED, API_RECONNECTED, MODEL_PAUSED, ...
    details     TEXT
);

CREATE TABLE IF NOT EXISTS trade_log (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp           TEXT NOT NULL,
    symbol              TEXT NOT NULL,
    direction           TEXT NOT NULL,
    result              TEXT NOT NULL,        -- WIN or LOSS
    profit_loss         REAL NOT NULL,
    stake               REAL NOT NULL,
    total_execution_latency_ms REAL
);

CREATE TABLE IF NOT EXISTS daily_state (
    trade_date          TEXT PRIMARY KEY,   -- "YYYY-MM-DD", UTC. One row per day -- a new day has no row, which IS the daily reset.
    current_balance     REAL NOT NULL,
    peak_balance        REAL NOT NULL,
    consecutive_losses  INTEGER NOT NULL,
    emergency_stopped   BOOLEAN NOT NULL,
    trades_today        INTEGER NOT NULL,
    wins                INTEGER NOT NULL,
    losses              INTEGER NOT NULL,
    session_pnl         REAL NOT NULL,
    current_losing_streak      INTEGER NOT NULL,
    longest_losing_streak_today INTEGER NOT NULL,
    updated_at          TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS performance_snapshots (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp           TEXT NOT NULL,
    strategy_run_id     TEXT REFERENCES strategy_runs(id),
    symbol              TEXT,
    duration_s          INTEGER,
    regime              TEXT,
    session             TEXT,
    trades_count        INTEGER,
    win_rate            REAL,
    average_ev          REAL,
    drawdown            REAL,
    calibration_error   REAL
);
