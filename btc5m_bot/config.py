from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return int(raw)


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return float(raw)


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_str(name: str, default: str) -> str:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return str(raw)


@dataclass(frozen=True)
class Settings:
    bot_root: Path
    bot_name: str
    bot_mode: str
    series_id: int

    decision_offset_seconds: int
    decision_window_start_seconds: int
    decision_window_end_seconds: int
    poll_interval_seconds: int

    default_cost_cents: float
    fee_fixed_cents_per_contract: float
    fee_bps_per_notional: float
    l1_slippage_buffer_cents: float
    l3_slippage_buffer_cents: float
    min_edge_cents: float
    max_spread_cents: float
    max_overround_cents: float
    l3_slippage_cents: float
    min_l1_depth_for_sizing_up: float
    min_l3_depth_for_sizing_up: float
    side_calib_lookback_days: int
    side_calib_min_samples: int
    side_calib_min_win_rate: float
    side_calib_edge_penalty_cents: float
    high_vol_rv60_threshold: float
    high_vol_min_edge_add_cents: float
    high_vol_max_spread_cents: float
    whatif_lookback_days: int
    execution_latency_seconds: float
    fixed_contracts: int
    sizing_mode: str
    risk_per_trade_pct: float
    max_contracts: int
    max_qty_per_signal: int
    max_trades_per_cycle: int
    max_open_live_positions: int
    paper_bankroll_cents: int
    shadow_enabled: bool

    lookback_days: int
    min_train_rows: int
    retrain_utc_hour: int
    auto_retrain: bool

    paper_window_days: int
    gate_min_labeled: int
    gate_min_executed: int
    gate_min_net_pnl_cents: int
    gate_min_expectancy_cents: float
    gate_max_drawdown_pct: float
    gate_drawdown_window_days: int
    auto_promote_live: bool

    rolling_expectancy_window: int
    max_daily_drawdown_pct: float
    hard_daily_net_stop_cents: int
    model_stuck_window: int
    model_stuck_epsilon: float
    max_sync_stale_seconds: int
    extreme_prob_guard_enabled: bool
    extreme_prob_threshold: float
    extreme_prob_lookback_days: int
    extreme_prob_min_samples: int
    extreme_prob_min_win_rate: float

    sync_interval_seconds: int
    hq_sync_url: str
    hq_settle_url: str
    hq_api_key: str

    market_provider: str
    kalshi_base_url: str
    kalshi_series_ticker: str
    gamma_base_url: str
    clob_base_url: str
    binance_ticker_url: str
    coinbase_ticker_url: str

    tick_retention_days: int
    cleanup_interval_seconds: int
    audit_snapshot_interval_seconds: int
    alert_webhook_url: str

    @property
    def data_dir(self) -> Path:
        return self.bot_root / "data"

    @property
    def model_dir(self) -> Path:
        return self.bot_root / "model"

    @property
    def db_path(self) -> Path:
        return self.data_dir / "bot.db"

    @property
    def status_path(self) -> Path:
        return self.data_dir / "status.json"

    @property
    def sync_state_path(self) -> Path:
        return self.data_dir / "sync_state.json"

    @property
    def model_path(self) -> Path:
        return self.model_dir / "model.joblib"



def load_settings() -> Settings:
    bot_root = Path(os.environ.get("BOT_ROOT", "/root/projects/kalshi-btc-15m-bot"))
    return Settings(
        bot_root=bot_root,
        bot_name=os.environ.get("BOT_NAME", "kalshi-btc-15m-v1"),
        bot_mode=os.environ.get("BOT_MODE", "paper").strip().lower(),
        series_id=_env_int("SERIES_ID", 0),
        decision_offset_seconds=_env_int("DECISION_OFFSET_SECONDS", 180),
        decision_window_start_seconds=_env_int("DECISION_WINDOW_START_SECONDS", 120),
        decision_window_end_seconds=_env_int("DECISION_WINDOW_END_SECONDS", 600),
        poll_interval_seconds=_env_int("POLL_INTERVAL_SECONDS", 1),
        default_cost_cents=_env_float("DEFAULT_COST_CENTS", 1.5),
        fee_fixed_cents_per_contract=_env_float("FEE_FIXED_CENTS_PER_CONTRACT", 0.0),
        fee_bps_per_notional=_env_float("FEE_BPS_PER_NOTIONAL", 0.0),
        l1_slippage_buffer_cents=_env_float("L1_SLIPPAGE_BUFFER_CENTS", 0.05),
        l3_slippage_buffer_cents=_env_float("L3_SLIPPAGE_BUFFER_CENTS", 0.35),
        min_edge_cents=_env_float("MIN_EDGE_CENTS", 1.8),
        max_spread_cents=_env_float("MAX_SPREAD_CENTS", 2.0),
        max_overround_cents=_env_float("MAX_OVERROUND_CENTS", 3.0),
        l3_slippage_cents=_env_float("L3_SLIPPAGE_CENTS", 1.0),
        min_l1_depth_for_sizing_up=_env_float("MIN_L1_DEPTH_FOR_SIZING_UP", 5.0),
        min_l3_depth_for_sizing_up=_env_float("MIN_L3_DEPTH_FOR_SIZING_UP", 15.0),
        side_calib_lookback_days=_env_int("SIDE_CALIB_LOOKBACK_DAYS", 14),
        side_calib_min_samples=_env_int("SIDE_CALIB_MIN_SAMPLES", 40),
        side_calib_min_win_rate=_env_float("SIDE_CALIB_MIN_WIN_RATE", 0.53),
        side_calib_edge_penalty_cents=_env_float("SIDE_CALIB_EDGE_PENALTY_CENTS", 0.5),
        high_vol_rv60_threshold=_env_float("HIGH_VOL_RV60_THRESHOLD", 0.0018),
        high_vol_min_edge_add_cents=_env_float("HIGH_VOL_MIN_EDGE_ADD_CENTS", 0.6),
        high_vol_max_spread_cents=_env_float("HIGH_VOL_MAX_SPREAD_CENTS", 1.5),
        whatif_lookback_days=_env_int("WHATIF_LOOKBACK_DAYS", 14),
        execution_latency_seconds=_env_float("EXECUTION_LATENCY_SECONDS", 1.0),
        fixed_contracts=_env_int("FIXED_CONTRACTS", 1),
        sizing_mode=_env_str("SIZING_MODE", "fixed").strip().lower(),
        risk_per_trade_pct=_env_float("RISK_PER_TRADE_PCT", 1.0),
        max_contracts=_env_int("MAX_CONTRACTS", 50),
        max_qty_per_signal=_env_int("MAX_QTY_PER_SIGNAL", 50),
        max_trades_per_cycle=_env_int("MAX_TRADES_PER_CYCLE", 2),
        max_open_live_positions=_env_int("MAX_OPEN_LIVE_POSITIONS", 1),
        paper_bankroll_cents=_env_int("PAPER_BANKROLL_CENTS", 200000),
        shadow_enabled=_env_bool("SHADOW_ENABLED", True),
        lookback_days=_env_int("LOOKBACK_DAYS", 14),
        min_train_rows=_env_int("MIN_TRAIN_ROWS", 200),
        retrain_utc_hour=_env_int("RETRAIN_UTC_HOUR", 0),
        auto_retrain=_env_bool("AUTO_RETRAIN", True),
        paper_window_days=_env_int("PAPER_WINDOW_DAYS", 7),
        gate_min_labeled=_env_int("GATE_MIN_LABELED", 1000),
        gate_min_executed=_env_int("GATE_MIN_EXECUTED", 300),
        gate_min_net_pnl_cents=_env_int("GATE_MIN_NET_PNL_CENTS", 1),
        gate_min_expectancy_cents=_env_float("GATE_MIN_EXPECTANCY_CENTS", 0.8),
        gate_max_drawdown_pct=_env_float("GATE_MAX_DRAWDOWN_PCT", 10.0),
        gate_drawdown_window_days=_env_int("GATE_DRAWDOWN_WINDOW_DAYS", 7),
        auto_promote_live=_env_bool("AUTO_PROMOTE_LIVE", False),
        rolling_expectancy_window=_env_int("ROLLING_EXPECTANCY_WINDOW", 50),
        max_daily_drawdown_pct=_env_float("MAX_DAILY_DRAWDOWN_PCT", 5.0),
        hard_daily_net_stop_cents=_env_int("HARD_DAILY_NET_STOP_CENTS", -5000),
        model_stuck_window=_env_int("MODEL_STUCK_WINDOW", 20),
        model_stuck_epsilon=_env_float("MODEL_STUCK_EPSILON", 1e-4),
        max_sync_stale_seconds=_env_int("MAX_SYNC_STALE_SECONDS", 300),
        extreme_prob_guard_enabled=_env_bool("EXTREME_PROB_GUARD_ENABLED", True),
        extreme_prob_threshold=_env_float("EXTREME_PROB_THRESHOLD", 0.95),
        extreme_prob_lookback_days=_env_int("EXTREME_PROB_LOOKBACK_DAYS", 14),
        extreme_prob_min_samples=_env_int("EXTREME_PROB_MIN_SAMPLES", 25),
        extreme_prob_min_win_rate=_env_float("EXTREME_PROB_MIN_WIN_RATE", 0.54),
        sync_interval_seconds=_env_int("SYNC_INTERVAL_SECONDS", 60),
        hq_sync_url=os.environ.get("KALSHI_HQ_SYNC_URL", "http://localhost:3000/api/sync"),
        hq_settle_url=os.environ.get("KALSHI_HQ_SETTLE_URL", "http://localhost:3000/api/sync/settle"),
        hq_api_key=os.environ.get("KALSHI_HQ_API_KEY", ""),
        market_provider=_env_str("MARKET_PROVIDER", "kalshi_btc15m").strip().lower(),
        kalshi_base_url=_env_str(
            "KALSHI_BASE_URL", "https://api.elections.kalshi.com/trade-api/v2"
        ).strip().rstrip("/"),
        kalshi_series_ticker=_env_str("KALSHI_SERIES_TICKER", "KXBTC15M").strip().upper(),
        gamma_base_url=_env_str("GAMMA_BASE_URL", "https://gamma-api.polymarket.com").strip().rstrip("/"),
        clob_base_url=_env_str("CLOB_BASE_URL", "https://clob.polymarket.com").strip().rstrip("/"),
        binance_ticker_url=os.environ.get(
            "BINANCE_TICKER_URL",
            "https://api.binance.com/api/v3/ticker/price?symbol=BTCUSDT",
        ),
        coinbase_ticker_url=os.environ.get(
            "COINBASE_TICKER_URL",
            "https://api.exchange.coinbase.com/products/BTC-USD/ticker",
        ),
        tick_retention_days=_env_int("TICK_RETENTION_DAYS", 7),
        cleanup_interval_seconds=_env_int("CLEANUP_INTERVAL_SECONDS", 3600),
        audit_snapshot_interval_seconds=_env_int("AUDIT_SNAPSHOT_INTERVAL_SECONDS", 3600),
        alert_webhook_url=_env_str("ALERT_WEBHOOK_URL", ""),
    )


def ensure_runtime_paths(settings: Settings) -> None:
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    settings.model_dir.mkdir(parents=True, exist_ok=True)
