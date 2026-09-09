"""
Central configuration. Every threshold referenced anywhere in strategies/, risk/,
or execution/ must be read from here — never hardcoded inline.
"""
from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class Mode(str):
    BACKTEST = "BACKTEST"
    DEMO = "DEMO"
    LIVE = "LIVE"


class Settings(BaseSettings):
    # --- mode / safety gates ---
    mode: str = "DEMO"
    live_trading: bool = False
    live_confirmation: bool = False

    # --- Deriv API (current "New API", as of the September 2026 docs) ---
    # NOTE: this is deliberately NOT the old wss://ws.derivws.com/websockets/v3?app_id=...
    # pattern shown in most tutorials/SDKs. Deriv now runs a REST+WebSocket split:
    #   REST base: account setup, get_accounts, and OTP minting for WS auth
    #   WS public: no auth — market data only (active_symbols, contracts_for, ticks)
    #   WS demo/real: OTP-authenticated — trading, balance, portfolio
    deriv_api_token: str = ""        # Personal Access Token (PAT); never logged, never printed
    deriv_app_id: str = ""           # required as Deriv-App-ID header on every REST call when using a PAT
    deriv_rest_base_url: str = "https://api.derivws.com"
    deriv_ws_public_url: str = "wss://api.derivws.com/trading/v1/options/ws/public"
    deriv_ws_demo_url: str = "wss://api.derivws.com/trading/v1/options/ws/demo"   # append ?otp=...
    deriv_ws_real_url: str = "wss://api.derivws.com/trading/v1/options/ws/real"   # append ?otp=...

    # --- market focus ---
    primary_market: str = "FOREX"

    # --- risk ---
    risk_per_trade: float = 0.005
    max_stake: float = 10.0
    max_daily_loss: float = 20.0
    max_drawdown: float = 0.15
    max_consecutive_losses: int = 5
    max_trades_per_hour: int = 4
    max_trades_per_day: int = 20
    max_open_contracts: int = 3

    # --- edge requirements ---
    min_probability_edge: float = 0.03
    min_expected_value: float = 0.0
    min_payout: float = 0.0

    # --- data / execution quality ---
    min_data_quality: int = 2          # 0-3 scale
    max_latency_ms: float = 800.0

    # --- durations to test (seconds) ---
    supported_durations_s: list[int] = [30, 60, 120, 300, 600, 900]

    # --- storage ---
    db_path: str = "forward_test.db"   # override via env for a persistent disk mount path (e.g. on Render)

    # --- control API (start/stop/strategy-switch endpoints) ---
    control_api_key: str = ""   # required header value (X-API-Key) for any /control/* endpoint; empty means control is disabled entirely

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")


settings = Settings()
