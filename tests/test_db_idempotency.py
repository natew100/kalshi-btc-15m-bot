from datetime import datetime, timezone

import pytest

from btc5m_bot import db


def test_trade_upsert_idempotent(tmp_path) -> None:
    conn = db.connect_db(tmp_path / "bot.db")
    db.init_db(conn)

    cycle = {
        "event_id": "e1",
        "market_id": "m1",
        "slug": "btc-updown-5m-e1",
        "ticker": "btc-updown-5m-e1",
        "start_ts": datetime.now(timezone.utc).isoformat(),
        "end_ts": datetime.now(timezone.utc).isoformat(),
    }
    db.upsert_cycle(conn, cycle)

    trade = {
        "external_id": "x1",
        "event_id": "e1",
        "ticker": "btc-updown-5m-e1",
        "side": "yes",
        "action": "buy",
        "price_cents": 55,
        "quantity": 1,
        "model_prob": 0.57,
        "market_price": 0.55,
        "edge": 0.02,
        "kelly_fraction": 0.01,
        "confidence": 0.14,
        "status": "dry_run",
        "pnl_cents": None,
        "reasoning": "test",
        "mode": "paper",
        "created_at": datetime.now(timezone.utc).isoformat(),
    }

    db.insert_trade(conn, trade)
    db.insert_trade(conn, {**trade, "status": "settled_win", "pnl_cents": 45})

    row = conn.execute("select * from trades where external_id = 'x1'").fetchone()
    assert row is not None
    assert row["status"] == "settled_win"
    assert int(row["pnl_cents"]) == 45

    conn.close()


def test_trade_notional_mismatch_rejected(tmp_path) -> None:
    conn = db.connect_db(tmp_path / "bot.db")
    db.init_db(conn)

    now = datetime.now(timezone.utc).isoformat()
    db.upsert_cycle(
        conn,
        {
            "event_id": "e2",
            "market_id": "m2",
            "slug": "btc-updown-5m-e2",
            "ticker": "btc-updown-5m-e2",
            "start_ts": now,
            "end_ts": now,
        },
    )
    with pytest.raises(ValueError, match="position_notional_mismatch"):
        db.insert_trade(
            conn,
            {
                "external_id": "x2",
                "event_id": "e2",
                "ticker": "btc-updown-5m-e2",
                "side": "yes",
                "action": "buy",
                "price_cents": 50,
                "quantity": 2,
                "model_prob": 0.57,
                "market_price": 0.50,
                "edge": 0.02,
                "kelly_fraction": 0.01,
                "confidence": 0.14,
                "status": "dry_run",
                "pnl_cents": None,
                "reasoning": "test",
                "mode": "paper",
                "position_notional_cents": 99,
                "created_at": now,
            },
        )
    conn.close()
