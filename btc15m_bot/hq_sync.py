from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import requests

from . import db
from .config import ensure_runtime_paths, load_settings


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return {}


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True))


def _trade_to_sync_payload(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "external_id": str(row["external_id"]),
        "ticker": str(row["ticker"]),
        "city": "btc",
        "side": str(row["side"]),
        "action": str(row["action"]),
        "price_cents": int(row["price_cents"]),
        "quantity": int(row["quantity"]),
        "model_prob": float(row["model_prob"]),
        "market_price": float(row["market_price"]),
        "edge": float(row["edge"]),
        "kelly_fraction": float(row["kelly_fraction"]),
        "confidence": float(row["confidence"]),
        "status": str(row["status"]),
        "pnl_cents": int(row["pnl_cents"]) if row["pnl_cents"] is not None else None,
        "balance_before_cents": int(row["balance_before_cents"])
        if row.get("balance_before_cents") is not None
        else None,
        "balance_after_cents": int(row["balance_after_cents"])
        if row.get("balance_after_cents") is not None
        else None,
        "position_notional_cents": int(row["position_notional_cents"])
        if row.get("position_notional_cents") is not None
        else None,
        "max_loss_cents": int(row["max_loss_cents"])
        if row.get("max_loss_cents") is not None
        else None,
        "fees_cents": int(row["fees_cents"]) if row.get("fees_cents") is not None else None,
        "reasoning": row["reasoning"],
        "created_at": str(row["created_at"]),
    }


def push_settlements(
    settle_url: str,
    api_key: str,
    bot_name: str,
    settlements: list[dict[str, Any]],
) -> tuple[bool, str]:
    if not settlements:
        return True, "no settlements"
    if not api_key:
        return False, "KALSHI_HQ_API_KEY missing"

    try:
        resp = requests.post(
            settle_url,
            json={"bot_name": bot_name, "settlements": settlements},
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=20,
        )
        if not resp.ok:
            return False, f"settle sync failed {resp.status_code}: {resp.text[:200]}"
        data = resp.json()
        return True, f"settled updates={data.get('updated', 0)}"
    except requests.RequestException as exc:
        return False, f"settle sync error: {exc}"


def sync_once() -> tuple[bool, str]:
    settings = load_settings()
    ensure_runtime_paths(settings)

    if not settings.hq_api_key:
        return False, "KALSHI_HQ_API_KEY missing"

    conn = db.connect_db(settings.db_path)
    db.init_db(conn)

    try:
        # Periodic reconciliation: even if a previous /settle call was rejected by HQ DB constraints,
        # this re-sends recent settlements (idempotent) so HQ converges.
        since = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0).isoformat()
        settlements = [
            {
                "external_id": str(r["external_id"]),
                "status": str(r["status"]),
                "pnl_cents": int(r["pnl_cents"] or 0),
                "settled_at": str(r["settled_at"]),
                "settlement_source": str(r["settlement_source"] or "unknown"),
            }
            for r in db.fetch_recent_settlements(conn, since_ts=since, limit=50)
        ]
        push_settlements(
            settle_url=settings.hq_settle_url,
            api_key=settings.hq_api_key,
            bot_name=settings.bot_name,
            settlements=settlements,
        )

        rows = db.fetch_unsynced_trades(conn, limit=500)

        # Backfill extended fields (fees/balances/settlement_source) for recent trades.
        # This is necessary when HQ DB migrations are applied after trades were already synced.
        state = _read_json(settings.sync_state_path)
        last_backfill_at = str(state.get("last_trade_backfill_at") or "")
        do_backfill = True
        if last_backfill_at:
            try:
                do_backfill = (datetime.now(timezone.utc) - datetime.fromisoformat(last_backfill_at)).total_seconds() >= 3600
            except ValueError:
                do_backfill = True

        backfill_rows: list[Any] = []
        if do_backfill:
            since_backfill = (datetime.now(timezone.utc) - timedelta(days=2)).isoformat()
            backfill_rows = db.fetch_recent_trades(conn, since_ts=since_backfill, limit=250)
            state["last_trade_backfill_at"] = datetime.now(timezone.utc).isoformat()
            _write_json(settings.sync_state_path, state)

        by_external: dict[str, dict[str, Any]] = {}
        for r in backfill_rows + rows:
            d = dict(r)
            by_external[str(d["external_id"])] = d
        payload_trades = [_trade_to_sync_payload(d) for d in by_external.values()]

        status = _read_json(settings.status_path)
        live_state = status if status else {"bot": {"status": "running", "updated_at": db.utcnow_iso()}}

        # Only push live_state when a new decision has been made since last sync.
        # This prevents HQ from recording every poll as a separate cycle row.
        state = _read_json(settings.sync_state_path)
        last_synced_decision = str(state.get("last_synced_decision_ts") or "")
        current_decision = str(status.get("last_decision_ts") or "")
        has_new_decision = current_decision and current_decision != last_synced_decision

        payload: dict[str, Any] = {

            "bot_name": settings.bot_name,
            "new_trades": payload_trades,
        }
        if has_new_decision or payload_trades:
            payload["live_state"] = live_state
            state["last_synced_decision_ts"] = current_decision
            _write_json(settings.sync_state_path, state)

        resp = requests.post(
            settings.hq_sync_url,
            json=payload,
            headers={"Authorization": f"Bearer {settings.hq_api_key}"},
            timeout=20,
        )
        if not resp.ok:
            return False, f"sync failed {resp.status_code}: {resp.text[:200]}"

        now = datetime.now(timezone.utc).isoformat()
        db.mark_trades_synced(conn, [str(r["external_id"]) for r in rows], now)

        state = _read_json(settings.sync_state_path)
        state.update(
            {
                "last_sync_ok_at": now,
                "last_sync_trade_count": len(rows),
                "last_sync_response": resp.json(),
            }
        )
        _write_json(settings.sync_state_path, state)

        status["sync"] = {
            "last_sync_ok_at": now,
            "queued_trades": 0,
            "last_sync_trade_count": len(rows),
        }
        _write_json(settings.status_path, status)

        return True, f"synced trades={len(rows)}"
    except requests.RequestException as exc:
        return False, f"sync request error: {exc}"
    finally:
        conn.close()


def _loop_forever() -> int:
    settings = load_settings()
    interval = max(5, int(settings.sync_interval_seconds))
    while True:
        ok, msg = sync_once()
        stamp = datetime.now(timezone.utc).isoformat()
        stream = sys.stdout if ok else sys.stderr
        print(f"[{stamp}] {msg}", file=stream)
        time.sleep(interval)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sync BTC 5m bot state/trades to KalshiHQ")
    parser.add_argument("--once", action="store_true", help="Run one sync pass and exit")
    args = parser.parse_args()

    if args.once:
        ok, msg = sync_once()
        print(msg, file=sys.stdout if ok else sys.stderr)
        raise SystemExit(0 if ok else 1)
    raise SystemExit(_loop_forever())
