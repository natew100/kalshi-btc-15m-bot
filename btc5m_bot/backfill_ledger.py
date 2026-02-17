from __future__ import annotations

import math
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from .strategy import estimate_execution_cost

def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _parse_iso(ts: str) -> datetime:
    return datetime.fromisoformat(ts.replace("Z", "+00:00")).astimezone(timezone.utc)


def _fee_cents(default_cost_cents: float, qty: int) -> int:
    # Match runner behavior: ceil(cost_cents * qty).
    return int(math.ceil(float(default_cost_cents) * float(int(qty))))


@dataclass(frozen=True)
class TradeRow:
    rowid: int
    external_id: str
    created_at: str
    settled_at: str | None
    status: str
    price_cents: int
    quantity: int
    pnl_cents: int | None
    fees_cents: int | None


def backfill_mode(
    conn: sqlite3.Connection,
    *,
    mode: str,
    starting_cash_cents: int,
    default_cost_cents: float,
) -> int:
    rows = conn.execute(
        """
        select rowid, external_id, created_at, settled_at, status, price_cents, quantity, pnl_cents, fees_cents
        from trades
        where mode = ?
        order by created_at asc
        """,
        (mode,),
    ).fetchall()

    trades: list[TradeRow] = [
        TradeRow(
            rowid=int(r[0]),
            external_id=str(r[1]),
            created_at=str(r[2]),
            settled_at=str(r[3]) if r[3] is not None else None,
            status=str(r[4]),
            price_cents=int(r[5]),
            quantity=int(r[6]),
            pnl_cents=int(r[7]) if r[7] is not None else None,
            fees_cents=int(r[8]) if r[8] is not None else None,
        )
        for r in rows
    ]

    # Build timeline events: open at created_at, settle at settled_at (if settled).
    events: list[tuple[datetime, str, TradeRow]] = []
    for t in trades:
        events.append((_parse_iso(t.created_at), "open", t))
        if t.status in ("settled_win", "settled_loss") and t.settled_at:
            events.append((_parse_iso(t.settled_at), "settle", t))

    # Ensure deterministic ordering for same-timestamp events.
    # Process settlements before opens at the same timestamp to avoid negative cash artifacts.
    events.sort(key=lambda x: (x[0], 0 if x[1] == "settle" else 1, x[2].external_id))

    cash = int(starting_cash_cents)

    # Track open state for settlement cash-in.
    opened: dict[str, dict[str, Any]] = {}

    updates: dict[int, dict[str, Any]] = {}
    now = _utcnow_iso()

    for ts, kind, t in events:
        if kind == "open":
            qty = int(t.quantity)
            px = int(t.price_cents)
            if t.fees_cents is not None:
                fees = int(t.fees_cents)
                slippage = 0
            else:
                cost = estimate_execution_cost(
                    price_cents=px,
                    quantity=qty,
                    fill_tier="l1",
                    fee_fixed_cents_per_contract=default_cost_cents,
                    fee_bps_per_notional=0.0,
                    l1_slippage_buffer_cents=0.0,
                    l3_slippage_buffer_cents=0.0,
                )
                fees = int(cost.fee_cents)
                slippage = int(cost.slippage_buffer_cents)
            spent = px * qty

            bal_before = cash
            cash = cash - spent - fees
            bal_after = cash

            opened[t.external_id] = {"spent": spent, "qty": qty, "px": px, "fees": fees}
            updates[t.rowid] = {
                "fees_cents": fees,
                "balance_before_cents": bal_before,
                "balance_after_cents": bal_after,
                "position_notional_cents": spent,
                "max_loss_cents": spent,
                "expected_fee_cents": fees,
                "realized_slippage_cents": slippage,
                "all_in_cost_cents": fees + slippage,
                "updated_at": now,
            }
        else:
            # Settle: cash-in is either 0 or 100*qty. We derive it from stored pnl + entry spent.
            o = opened.get(t.external_id)
            if not o:
                continue
            spent = int(o["spent"])
            pnl = int(t.pnl_cents or 0)
            cash_in = spent + pnl  # win: 100*qty; loss: 0
            cash += cash_in

    # Apply updates.
    if not updates:
        return 0

    cur = conn.cursor()
    for rowid, u in updates.items():
        cur.execute(
            """
            update trades
            set fees_cents = ?,
                balance_before_cents = ?,
                balance_after_cents = ?,
                position_notional_cents = ?,
                max_loss_cents = ?,
                expected_fee_cents = ?,
                realized_slippage_cents = ?,
                all_in_cost_cents = ?,
                updated_at = ?
            where rowid = ?
            """,
            (
                u["fees_cents"],
                u["balance_before_cents"],
                u["balance_after_cents"],
                u["position_notional_cents"],
                u["max_loss_cents"],
                u["expected_fee_cents"],
                u["realized_slippage_cents"],
                u["all_in_cost_cents"],
                u["updated_at"],
                rowid,
            ),
        )
    conn.commit()
    return len(updates)


def backfill_all(
    db_path: str,
    *,
    starting_cash_cents: int,
    default_cost_cents: float,
) -> dict[str, int]:
    conn = sqlite3.connect(db_path)
    try:
        conn.execute("pragma journal_mode=wal;")
        updated_paper = backfill_mode(
            conn,
            mode="paper",
            starting_cash_cents=starting_cash_cents,
            default_cost_cents=default_cost_cents,
        )
        updated_shadow = backfill_mode(
            conn,
            mode="shadow",
            starting_cash_cents=starting_cash_cents,
            default_cost_cents=default_cost_cents,
        )
        return {"paper": updated_paper, "shadow": updated_shadow}
    finally:
        conn.close()
