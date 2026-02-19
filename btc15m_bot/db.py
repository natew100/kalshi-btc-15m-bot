from __future__ import annotations

import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any


def utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def connect_db(db_path: Path) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path), timeout=30)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def init_db(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        create table if not exists cycles (
          event_id text primary key,
          market_id text not null,
          slug text not null,
          ticker text not null,
          start_ts text not null,
          end_ts text not null,
          decision_ts text,
          resolution_ts text,
          label_up integer,
          resolved integer not null default 0,
          created_at text not null default (strftime('%Y-%m-%dT%H:%M:%fZ','now')),
          updated_at text not null default (strftime('%Y-%m-%dT%H:%M:%fZ','now'))
        );

        create table if not exists ticks (
          id integer primary key autoincrement,
          event_id text not null,
          ts text not null,
          best_bid real,
          best_ask real,
          last_trade_price real,
          spread real,
          mid_yes real,
          bid_sz_l1 real,
          ask_sz_l1 real,
          bid_sz_l3 real,
          ask_sz_l3 real,
          imbalance_l1 real,
          imbalance_l3 real,
          spot_price real,
          created_at text not null default (strftime('%Y-%m-%dT%H:%M:%fZ','now')),
          foreign key(event_id) references cycles(event_id) on delete cascade
        );
        create index if not exists idx_ticks_event_ts on ticks(event_id, ts);

        create table if not exists feature_rows (
          event_id text primary key,
          decision_ts text not null,
          label_up integer,
          mid_yes real not null,
          spread real not null,
          imbalance_l1 real not null,
          imbalance_l3 real not null,
          ret_15s real not null,
          ret_30s real not null,
          ret_60s real not null,
          ret_120s real not null default 0.0,
          ret_180s real not null default 0.0,
          ret_300s real not null default 0.0,
          rv_60s real not null,
          rv_180s real not null default 0.0,
          tod_sin real not null,
          tod_cos real not null,
          prev1_up real not null,
          prev2_up real not null,
          prev3_up real not null,
          created_at text not null default (strftime('%Y-%m-%dT%H:%M:%fZ','now')),
          foreign key(event_id) references cycles(event_id) on delete cascade
        );
        create index if not exists idx_feature_rows_decision_ts on feature_rows(decision_ts);

        create table if not exists decisions (
          event_id text primary key,
          trade_external_id text,
          decided_at text not null,
          model_prob real not null,
          ask_yes real not null,
          ask_no real not null,
          spread real not null,
          ev_yes real not null,
          ev_no real not null,
          chosen_side text,
          acted integer not null default 0,
          reason text,
          mode text not null,
          cost_cents real not null,
          created_at text not null default (strftime('%Y-%m-%dT%H:%M:%fZ','now')),
          foreign key(event_id) references cycles(event_id) on delete cascade
        );

        create table if not exists trades (
          external_id text primary key,
          event_id text not null,
          ticker text not null,
          side text not null check (side in ('yes','no')),
          action text not null default 'buy' check (action in ('buy','sell')),
          price_cents integer not null,
          quantity integer not null default 1,
          model_prob real not null,
          market_price real not null,
          edge real not null,
          kelly_fraction real not null,
          confidence real not null,
          status text not null,
          pnl_cents integer,
          reasoning text,
          mode text not null,
          balance_before_cents integer,
          balance_after_cents integer,
          position_notional_cents integer,
          max_loss_cents integer,
          fees_cents integer,
          expected_fee_cents integer,
          realized_slippage_cents integer,
          all_in_cost_cents integer,
          requested_qty integer,
          filled_qty integer,
          fill_tier text,
          fill_reason text,
          fill_slippage_cents real,
          created_at text not null,
          settled_at text,
          settlement_source text,
          synced_at text,
          updated_at text not null default (strftime('%Y-%m-%dT%H:%M:%fZ','now')),
          foreign key(event_id) references cycles(event_id) on delete cascade
        );
        create index if not exists idx_trades_event on trades(event_id);
        create index if not exists idx_trades_status on trades(status);

        create table if not exists spot_ticks (
          ts text primary key,
          price real not null
        );

        create table if not exists model_runs (
          id integer primary key autoincrement,
          started_at text not null,
          finished_at text,
          n_rows integer,
          brier real,
          logloss real,
          auc real,
          status text not null,
          error text
        );
        """
    )
    conn.commit()
    _ensure_trade_columns(conn)


def _table_columns(conn: sqlite3.Connection, table: str) -> set[str]:
    rows = conn.execute(f"pragma table_info({table})").fetchall()
    return {str(r[1]) for r in rows}


def _ensure_trade_columns(conn: sqlite3.Connection) -> None:
    cols = _table_columns(conn, "trades")
    wanted: list[tuple[str, str]] = [
        ("balance_before_cents", "integer"),
        ("balance_after_cents", "integer"),
        ("position_notional_cents", "integer"),
        ("max_loss_cents", "integer"),
        ("fees_cents", "integer"),
        ("expected_fee_cents", "integer"),
        ("realized_slippage_cents", "integer"),
        ("all_in_cost_cents", "integer"),
        ("requested_qty", "integer"),
        ("filled_qty", "integer"),
        ("fill_tier", "text"),
        ("fill_reason", "text"),
        ("fill_slippage_cents", "real"),
    ]
    for name, typ in wanted:
        if name in cols:
            continue
        # SQLite has no ADD COLUMN IF NOT EXISTS; do it manually.
        conn.execute(f"alter table trades add column {name} {typ}")
    conn.commit()


def upsert_cycle(conn: sqlite3.Connection, cycle: dict[str, Any]) -> None:
    now = utcnow_iso()
    conn.execute(
        """
        insert into cycles(event_id, market_id, slug, ticker, start_ts, end_ts, updated_at)
        values(?, ?, ?, ?, ?, ?, ?)
        on conflict(event_id) do update set
          market_id=excluded.market_id,
          slug=excluded.slug,
          ticker=excluded.ticker,
          start_ts=excluded.start_ts,
          end_ts=excluded.end_ts,
          updated_at=excluded.updated_at
        """,
        (
            cycle["event_id"],
            cycle["market_id"],
            cycle["slug"],
            cycle["ticker"],
            cycle["start_ts"],
            cycle["end_ts"],
            now,
        ),
    )
    conn.commit()


def insert_tick(conn: sqlite3.Connection, event_id: str, tick: dict[str, Any]) -> None:
    conn.execute(
        """
        insert into ticks(
          event_id, ts, best_bid, best_ask, last_trade_price, spread, mid_yes,
          bid_sz_l1, ask_sz_l1, bid_sz_l3, ask_sz_l3, imbalance_l1, imbalance_l3,
          spot_price
        ) values (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            event_id,
            tick["ts"],
            tick.get("best_bid"),
            tick.get("best_ask"),
            tick.get("last_trade_price"),
            tick.get("spread"),
            tick.get("mid_yes"),
            tick.get("bid_sz_l1"),
            tick.get("ask_sz_l1"),
            tick.get("bid_sz_l3"),
            tick.get("ask_sz_l3"),
            tick.get("imbalance_l1"),
            tick.get("imbalance_l3"),
            tick.get("spot_price"),
        ),
    )
    conn.commit()


def insert_spot_tick(conn: sqlite3.Connection, ts: str, price: float) -> None:
    conn.execute(
        "insert or replace into spot_ticks(ts, price) values(?, ?)",
        (ts, price),
    )
    conn.commit()


def cycle_already_decided(conn: sqlite3.Connection, event_id: str) -> bool:
    row = conn.execute(
        "select 1 from decisions where event_id = ? limit 1", (event_id,)
    ).fetchone()
    return row is not None


def insert_feature_row(
    conn: sqlite3.Connection,
    event_id: str,
    decision_ts: str,
    features: dict[str, float],
) -> None:
    conn.execute(
        """
        insert into feature_rows(
          event_id, decision_ts, mid_yes, spread, imbalance_l1, imbalance_l3,
          ret_15s, ret_30s, ret_60s, ret_120s, ret_180s, ret_300s,
          rv_60s, rv_180s, tod_sin, tod_cos,
          prev1_up, prev2_up, prev3_up
        ) values(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        on conflict(event_id) do update set
          decision_ts=excluded.decision_ts,
          mid_yes=excluded.mid_yes,
          spread=excluded.spread,
          imbalance_l1=excluded.imbalance_l1,
          imbalance_l3=excluded.imbalance_l3,
          ret_15s=excluded.ret_15s,
          ret_30s=excluded.ret_30s,
          ret_60s=excluded.ret_60s,
          ret_120s=excluded.ret_120s,
          ret_180s=excluded.ret_180s,
          ret_300s=excluded.ret_300s,
          rv_60s=excluded.rv_60s,
          rv_180s=excluded.rv_180s,
          tod_sin=excluded.tod_sin,
          tod_cos=excluded.tod_cos,
          prev1_up=excluded.prev1_up,
          prev2_up=excluded.prev2_up,
          prev3_up=excluded.prev3_up
        """,
        (
            event_id,
            decision_ts,
            features["mid_yes"],
            features["spread"],
            features["imbalance_l1"],
            features["imbalance_l3"],
            features["ret_15s"],
            features["ret_30s"],
            features["ret_60s"],
            features["ret_120s"],
            features["ret_180s"],
            features["ret_300s"],
            features["rv_60s"],
            features["rv_180s"],
            features["tod_sin"],
            features["tod_cos"],
            features["prev1_up"],
            features["prev2_up"],
            features["prev3_up"],
        ),
    )
    conn.commit()


def insert_decision(conn: sqlite3.Connection, payload: dict[str, Any]) -> None:
    conn.execute(
        """
        insert into decisions(
          event_id, trade_external_id, decided_at, model_prob, ask_yes, ask_no,
          spread, ev_yes, ev_no, chosen_side, acted, reason, mode, cost_cents
        ) values(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        on conflict(event_id) do update set
          trade_external_id=excluded.trade_external_id,
          decided_at=excluded.decided_at,
          model_prob=excluded.model_prob,
          ask_yes=excluded.ask_yes,
          ask_no=excluded.ask_no,
          spread=excluded.spread,
          ev_yes=excluded.ev_yes,
          ev_no=excluded.ev_no,
          chosen_side=excluded.chosen_side,
          acted=excluded.acted,
          reason=excluded.reason,
          mode=excluded.mode,
          cost_cents=excluded.cost_cents
        """,
        (
            payload["event_id"],
            payload.get("trade_external_id"),
            payload["decided_at"],
            payload["model_prob"],
            payload["ask_yes"],
            payload["ask_no"],
            payload["spread"],
            payload["ev_yes"],
            payload["ev_no"],
            payload.get("chosen_side"),
            1 if payload.get("acted") else 0,
            payload.get("reason"),
            payload["mode"],
            payload["cost_cents"],
        ),
    )
    conn.commit()


def set_cycle_decision_ts(conn: sqlite3.Connection, event_id: str, decision_ts: str) -> None:
    conn.execute(
        """
        update cycles
        set decision_ts = ?,
            updated_at = ?
        where event_id = ?
        """,
        (decision_ts, utcnow_iso(), event_id),
    )
    conn.commit()


def insert_trade(conn: sqlite3.Connection, trade: dict[str, Any]) -> None:
    qty = int(trade.get("quantity", 1) or 1)
    px = int(trade["price_cents"])
    expected_notional = px * qty
    provided_notional = trade.get("position_notional_cents")
    if provided_notional is not None and int(provided_notional) != expected_notional:
        raise ValueError(
            f"position_notional_mismatch external_id={trade.get('external_id')} "
            f"expected={expected_notional} got={int(provided_notional)}"
        )
    trade["position_notional_cents"] = expected_notional

    now = utcnow_iso()
    conn.execute(
        """
        insert into trades(
          external_id, event_id, ticker, side, action, price_cents, quantity,
          model_prob, market_price, edge, kelly_fraction, confidence, status,
          pnl_cents, reasoning, mode,
          balance_before_cents, balance_after_cents, position_notional_cents, max_loss_cents, fees_cents,
          expected_fee_cents, realized_slippage_cents, all_in_cost_cents,
          requested_qty, filled_qty, fill_tier, fill_reason, fill_slippage_cents,
          created_at, updated_at
        ) values(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        on conflict(external_id) do update set
          status=excluded.status,
          pnl_cents=excluded.pnl_cents,
          reasoning=excluded.reasoning,
          fees_cents=coalesce(excluded.fees_cents, trades.fees_cents),
          expected_fee_cents=coalesce(excluded.expected_fee_cents, trades.expected_fee_cents),
          realized_slippage_cents=coalesce(excluded.realized_slippage_cents, trades.realized_slippage_cents),
          all_in_cost_cents=coalesce(excluded.all_in_cost_cents, trades.all_in_cost_cents),
          balance_before_cents=coalesce(excluded.balance_before_cents, trades.balance_before_cents),
          balance_after_cents=coalesce(excluded.balance_after_cents, trades.balance_after_cents),
          position_notional_cents=coalesce(excluded.position_notional_cents, trades.position_notional_cents),
          max_loss_cents=coalesce(excluded.max_loss_cents, trades.max_loss_cents),
          requested_qty=coalesce(excluded.requested_qty, trades.requested_qty),
          filled_qty=coalesce(excluded.filled_qty, trades.filled_qty),
          fill_tier=coalesce(excluded.fill_tier, trades.fill_tier),
          fill_reason=coalesce(excluded.fill_reason, trades.fill_reason),
          fill_slippage_cents=coalesce(excluded.fill_slippage_cents, trades.fill_slippage_cents),
          updated_at=excluded.updated_at
        """,
        (
            trade["external_id"],
            trade["event_id"],
            trade["ticker"],
            trade["side"],
            trade.get("action", "buy"),
            trade["price_cents"],
            trade.get("quantity", 1),
            trade["model_prob"],
            trade["market_price"],
            trade["edge"],
            trade["kelly_fraction"],
            trade["confidence"],
            trade["status"],
            trade.get("pnl_cents"),
            trade.get("reasoning"),
            trade["mode"],
            trade.get("balance_before_cents"),
            trade.get("balance_after_cents"),
            trade.get("position_notional_cents"),
            trade.get("max_loss_cents"),
            trade.get("fees_cents"),
            trade.get("expected_fee_cents"),
            trade.get("realized_slippage_cents"),
            trade.get("all_in_cost_cents"),
            trade.get("requested_qty"),
            trade.get("filled_qty"),
            trade.get("fill_tier"),
            trade.get("fill_reason"),
            trade.get("fill_slippage_cents"),
            trade["created_at"],
            now,
        ),
    )
    conn.commit()


def trade_count_for_event_modes(conn: sqlite3.Connection, event_id: str, modes: tuple[str, ...]) -> int:
    if not modes:
        return 0
    placeholders = ",".join(["?"] * len(modes))
    row = conn.execute(
        f"""
        select count(*) as c
        from trades
        where event_id = ?
          and mode in ({placeholders})
        """,
        (event_id, *list(modes)),
    ).fetchone()
    return int(row["c"] if row else 0)


def get_trade_for_event(conn: sqlite3.Connection, event_id: str) -> sqlite3.Row | None:
    return conn.execute(
        "select * from trades where event_id = ? order by created_at asc limit 1",
        (event_id,),
    ).fetchone()


def get_trades_for_event(conn: sqlite3.Connection, event_id: str) -> list[sqlite3.Row]:
    return conn.execute(
        "select * from trades where event_id = ? order by created_at asc",
        (event_id,),
    ).fetchall()


def settle_trade(
    conn: sqlite3.Connection,
    external_id: str,
    status: str,
    pnl_cents: int,
    settled_at: str,
    settlement_source: str,
) -> None:
    conn.execute(
        """
        update trades
        set status = ?,
            pnl_cents = ?,
            settled_at = ?,
            settlement_source = ?,
            updated_at = ?
        where external_id = ?
        """,
        (status, pnl_cents, settled_at, settlement_source, utcnow_iso(), external_id),
    )
    conn.commit()


def resolve_cycle(conn: sqlite3.Connection, event_id: str, label_up: int, resolution_ts: str) -> None:
    conn.execute(
        """
        update cycles
        set label_up = ?,
            resolution_ts = ?,
            resolved = 1,
            updated_at = ?
        where event_id = ?
        """,
        (label_up, resolution_ts, utcnow_iso(), event_id),
    )
    conn.execute(
        "update feature_rows set label_up = ? where event_id = ?",
        (label_up, event_id),
    )
    conn.commit()


def recent_outcomes(conn: sqlite3.Connection, before_ts: str, limit: int = 3) -> list[int]:
    rows = conn.execute(
        """
        select label_up
        from cycles
        where resolved = 1 and label_up is not null and decision_ts < ?
        order by decision_ts desc
        limit ?
        """,
        (before_ts, limit),
    ).fetchall()
    return [int(r["label_up"]) for r in rows]


def unresolved_cycles_ended_before(
    conn: sqlite3.Connection, before_ts: str, limit: int = 50
) -> list[sqlite3.Row]:
    return conn.execute(
        """
        select event_id, end_ts
        from cycles
        where resolved = 0 and end_ts < ?
        order by end_ts asc
        limit ?
        """,
        (before_ts, int(limit)),
    ).fetchall()


def recent_ticks_for_event(
    conn: sqlite3.Connection,
    event_id: str,
    window_seconds: int = 180,
) -> list[sqlite3.Row]:
    floor_ts = (datetime.now(timezone.utc) - timedelta(seconds=window_seconds)).isoformat()
    return conn.execute(
        """
        select * from ticks
        where event_id = ? and ts >= ?
        order by ts asc
        """,
        (event_id, floor_ts),
    ).fetchall()


def all_ticks_for_event(conn: sqlite3.Connection, event_id: str) -> list[sqlite3.Row]:
    return conn.execute(
        "select * from ticks where event_id = ? order by ts asc",
        (event_id,),
    ).fetchall()


def fetch_labeled_features(conn: sqlite3.Connection, lookback_days: int) -> list[sqlite3.Row]:
    floor_ts = (datetime.now(timezone.utc) - timedelta(days=lookback_days)).isoformat()
    return conn.execute(
        """
        select fr.*, c.start_ts, c.end_ts
        from feature_rows fr
        join cycles c on c.event_id = fr.event_id
        where fr.label_up is not null and fr.decision_ts >= ?
        order by fr.decision_ts asc
        """,
        (floor_ts,),
    ).fetchall()


def insert_model_run_start(conn: sqlite3.Connection) -> int:
    cur = conn.execute(
        "insert into model_runs(started_at, status) values(?, ?)",
        (utcnow_iso(), "running"),
    )
    conn.commit()
    return int(cur.lastrowid)


def complete_model_run(
    conn: sqlite3.Connection,
    run_id: int,
    status: str,
    n_rows: int | None = None,
    brier: float | None = None,
    logloss: float | None = None,
    auc: float | None = None,
    error: str | None = None,
) -> None:
    conn.execute(
        """
        update model_runs
        set finished_at = ?,
            status = ?,
            n_rows = ?,
            brier = ?,
            logloss = ?,
            auc = ?,
            error = ?
        where id = ?
        """,
        (utcnow_iso(), status, n_rows, brier, logloss, auc, error, run_id),
    )
    conn.commit()


def fetch_unsynced_trades(conn: sqlite3.Connection, limit: int = 500) -> list[sqlite3.Row]:
    return conn.execute(
        """
        select rowid, *
        from trades
        where (synced_at is null or updated_at > synced_at)
          and mode in ('paper', 'live')
        order by updated_at asc
        limit ?
        """,
        (limit,),
    ).fetchall()


def fetch_recent_settlements(
    conn: sqlite3.Connection,
    since_ts: str,
    limit: int = 50,
) -> list[sqlite3.Row]:
    return conn.execute(
        """
        select external_id, status, pnl_cents, settled_at, settlement_source
        from trades
        where mode in ('paper', 'live')
          and status in ('settled_win', 'settled_loss')
          and settled_at is not null
          and settled_at >= ?
        order by settled_at desc
        limit ?
        """,
        (since_ts, int(limit)),
    ).fetchall()


def fetch_recent_trades(
    conn: sqlite3.Connection,
    since_ts: str,
    limit: int = 200,
) -> list[sqlite3.Row]:
    return conn.execute(
        """
        select rowid, *
        from trades
        where mode in ('paper', 'live')
          and created_at >= ?
        order by created_at desc
        limit ?
        """,
        (since_ts, int(limit)),
    ).fetchall()


def cleanup_ticks_older_than(conn: sqlite3.Connection, floor_ts: str) -> int:
    """
    Best-effort retention: tick tables are high cadence and will grow without bound.
    Keep everything else (cycles/trades/feature_rows) for audit/training.
    """
    cur = conn.execute("delete from ticks where ts < ?", (floor_ts,))
    # spot_ticks is currently unused, but may be enabled again later.
    conn.execute("delete from spot_ticks where ts < ?", (floor_ts,))
    conn.commit()
    return int(cur.rowcount or 0)


def mark_trades_synced(conn: sqlite3.Connection, external_ids: list[str], synced_at: str) -> None:
    if not external_ids:
        return
    placeholders = ",".join(["?"] * len(external_ids))
    conn.execute(
        f"update trades set synced_at = ? where external_id in ({placeholders})",
        [synced_at, *external_ids],
    )
    conn.commit()


def count_open_live_positions(conn: sqlite3.Connection) -> int:
    row = conn.execute(
        """
        select count(*) as c
        from trades
        where mode = 'live' and status in ('pending', 'filled')
        """
    ).fetchone()
    return int(row["c"]) if row else 0


def latest_model_run(conn: sqlite3.Connection) -> sqlite3.Row | None:
    return conn.execute(
        "select * from model_runs order by id desc limit 1"
    ).fetchone()


def fetch_model_runs_since(conn: sqlite3.Connection, after_id: int) -> list[sqlite3.Row]:
    return conn.execute(
        "SELECT id, started_at, finished_at, n_rows, brier, logloss, auc, status, error "
        "FROM model_runs WHERE id > ? ORDER BY id",
        (after_id,),
    ).fetchall()


def fetch_market_cycles_since(
    conn: sqlite3.Connection, since_ts: str, limit: int = 2000
) -> list[sqlite3.Row]:
    """Fetch cycles updated since the given timestamp (for HQ sync)."""
    return conn.execute(
        "SELECT event_id, market_id, slug, ticker, start_ts, end_ts, "
        "decision_ts, resolution_ts, label_up, resolved, created_at, updated_at "
        "FROM cycles WHERE updated_at > ? ORDER BY updated_at LIMIT ?",
        (since_ts, limit),
    ).fetchall()
