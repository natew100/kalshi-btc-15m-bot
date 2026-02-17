from datetime import datetime, timezone
from types import SimpleNamespace

from btc15m_bot import db
from btc15m_bot.runner import _dynamic_thresholds, _whatif_metrics


def test_dynamic_thresholds_apply_high_vol_and_side_penalty() -> None:
    settings = SimpleNamespace(
        min_edge_cents=1.8,
        max_spread_cents=2.0,
        high_vol_rv60_threshold=0.0018,
        high_vol_min_edge_add_cents=0.6,
        high_vol_max_spread_cents=1.5,
        side_calib_min_samples=40,
        side_calib_min_win_rate=0.53,
        side_calib_edge_penalty_cents=0.5,
    )
    min_edge, max_spread, tags = _dynamic_thresholds(
        settings,
        rv_60s=0.0021,
        side_samples=60,
        side_win_rate=0.50,
    )
    assert abs(min_edge - 2.9) < 1e-9
    assert abs(max_spread - 1.5) < 1e-9
    assert "high_vol" in tags
    assert "side_calib_penalty" in tags


def test_whatif_metrics_has_stricter_subset(tmp_path) -> None:
    conn = db.connect_db(tmp_path / "bot.db")
    db.init_db(conn)

    now = datetime.now(timezone.utc).isoformat()
    cycle = {
        "event_id": "e1",
        "market_id": "m1",
        "slug": "btc-updown-5m-e1",
        "ticker": "btc-updown-5m-e1",
        "start_ts": now,
        "end_ts": now,
    }
    db.upsert_cycle(conn, cycle)
    db.insert_decision(
        conn,
        {
            "event_id": "e1",
            "trade_external_id": "x1",
            "decided_at": now,
            "model_prob": 0.6,
            "ask_yes": 55.0,
            "ask_no": 46.0,
            "spread": 1.2,
            "ev_yes": 2.5,
            "ev_no": -1.0,
            "chosen_side": "yes",
            "acted": True,
            "reason": "ok",
            "mode": "paper",
            "cost_cents": 1.5,
        },
    )
    db.insert_trade(
        conn,
        {
            "external_id": "x1",
            "event_id": "e1",
            "ticker": "btc-updown-5m-e1",
            "side": "yes",
            "action": "buy",
            "price_cents": 55,
            "quantity": 1,
            "model_prob": 0.6,
            "market_price": 0.55,
            "edge": 2.5,
            "kelly_fraction": 0.01,
            "confidence": 0.2,
            "status": "settled_win",
            "pnl_cents": 45,
            "reasoning": "t",
            "mode": "paper",
            "fees_cents": 2,
            "created_at": now,
        },
    )
    out = _whatif_metrics(conn, mode="paper", lookback_days=14, base_min_edge=1.8, base_max_spread=2.0)
    assert out["base_settled_trades"] >= 1
    assert out["scenarios"]["baseline"]["settled_trades"] >= out["scenarios"]["strict_both"]["settled_trades"]
    conn.close()
