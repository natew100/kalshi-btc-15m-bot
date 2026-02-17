from datetime import datetime, timezone
from types import SimpleNamespace

from btc15m_bot import db
from btc15m_bot.gating import evaluate_auto_pause, evaluate_go_live_gate


def _settings() -> SimpleNamespace:
    return SimpleNamespace(
        paper_window_days=7,
        gate_drawdown_window_days=7,
        gate_min_labeled=1,
        gate_min_executed=1,
        gate_min_net_pnl_cents=1,
        gate_min_expectancy_cents=0.0,
        gate_max_drawdown_pct=100.0,
        paper_bankroll_cents=200000,
        rolling_expectancy_window=2,
        max_daily_drawdown_pct=100.0,
        hard_daily_net_stop_cents=-999999,
        model_stuck_window=100,
        model_stuck_epsilon=1e-6,
        max_sync_stale_seconds=3600,
    )


def test_go_live_gate_uses_recorded_fees(tmp_path) -> None:
    conn = db.connect_db(tmp_path / "bot.db")
    db.init_db(conn)
    now = datetime.now(timezone.utc).isoformat()
    db.upsert_cycle(
        conn,
        {
            "event_id": "e1",
            "market_id": "m1",
            "slug": "btc-updown-5m-e1",
            "ticker": "btc-updown-5m-e1",
            "start_ts": now,
            "end_ts": now,
        },
    )
    db.resolve_cycle(conn, "e1", 1, now)
    db.insert_trade(
        conn,
        {
            "external_id": "x1",
            "event_id": "e1",
            "ticker": "btc-updown-5m-e1",
            "side": "yes",
            "action": "buy",
            "price_cents": 50,
            "quantity": 1,
            "model_prob": 0.57,
            "market_price": 0.50,
            "edge": 0.02,
            "kelly_fraction": 0.01,
            "confidence": 0.14,
            "status": "settled_win",
            "pnl_cents": 50,
            "fees_cents": 40,
            "reasoning": "t",
            "mode": "paper",
            "created_at": now,
            "settled_at": now,
        },
    )
    out = evaluate_go_live_gate(conn, _settings())
    assert out.net_pnl_cents_after_cost == 10.0
    conn.close()


def test_auto_pause_rolling_expectancy_uses_fees(tmp_path) -> None:
    conn = db.connect_db(tmp_path / "bot.db")
    db.init_db(conn)
    now = datetime.now(timezone.utc).isoformat()
    for i, pnl in enumerate([10, 10], start=1):
        eid = f"e{i}"
        db.upsert_cycle(
            conn,
            {
                "event_id": eid,
                "market_id": eid,
                "slug": eid,
                "ticker": eid,
                "start_ts": now,
                "end_ts": now,
            },
        )
        db.insert_trade(
            conn,
            {
                "external_id": f"x{i}",
                "event_id": eid,
                "ticker": eid,
                "side": "yes",
                "action": "buy",
                "price_cents": 50,
                "quantity": 1,
                "model_prob": 0.57,
                "market_price": 0.50,
                "edge": 0.02,
                "kelly_fraction": 0.01,
                "confidence": 0.14,
                "status": "settled_win",
                "pnl_cents": pnl,
                "fees_cents": 20,
                "reasoning": "t",
                "mode": "paper",
                "created_at": now,
                "settled_at": now,
            },
        )
    paused, reasons = evaluate_auto_pause(conn, _settings(), None)
    assert paused is True
    assert "rolling_50_expectancy_negative" in reasons
    conn.close()
