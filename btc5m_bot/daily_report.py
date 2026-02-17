from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

from . import db
from .config import ensure_runtime_paths, load_settings
from .gating import evaluate_go_live_gate, gate_to_dict


def main() -> int:
    settings = load_settings()
    ensure_runtime_paths(settings)

    conn = db.connect_db(settings.db_path)
    db.init_db(conn)

    floor = (datetime.now(timezone.utc) - timedelta(days=1)).isoformat()
    rows = conn.execute(
        """
        select status, pnl_cents, fees_cents
        from trades
        where created_at >= ?
        """,
        (floor,),
    ).fetchall()

    executed = [r for r in rows if r["status"] != "no_fill"]
    settled = [r for r in rows if r["status"] in ("settled_win", "settled_loss")]
    wins = sum(1 for r in settled if r["status"] == "settled_win")

    gross_pnl = sum(int(r["pnl_cents"] or 0) for r in settled)
    actual_fees = sum(int(r["fees_cents"] or 0) for r in executed)
    net_pnl = gross_pnl - actual_fees

    hit_rate = (wins / len(settled)) if settled else 0.0
    net_cents_per_trade = (net_pnl / len(executed)) if executed else 0.0

    gate = evaluate_go_live_gate(conn, settings)

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "window_hours": 24,
        "executed_trades": len(executed),
        "settled_trades": len(settled),
        "hit_rate": hit_rate,
        "gross_pnl_cents": gross_pnl,
        "actual_fees_cents": actual_fees,
        "net_pnl_cents": net_pnl,
        "net_cents_per_trade": net_cents_per_trade,
        "go_live_gate": gate_to_dict(gate),
    }

    print(json.dumps(report, indent=2, sort_keys=True))
    conn.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
