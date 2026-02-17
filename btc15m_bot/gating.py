from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import math
import sqlite3

from .config import Settings


@dataclass
class GateResult:
    passed: bool
    labeled_cycles: int
    executed_trades: int
    settled_trades: int
    net_pnl_cents_after_cost: float
    expectancy_cents: float
    max_drawdown_pct: float
    drawdown_window_days: int
    reasons: list[str]



def _iso_floor(days: int) -> str:
    return (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()


def _max_drawdown_pct(pnls: list[int], starting_bankroll_cents: int) -> float:
    equity = float(starting_bankroll_cents)
    peak = equity
    worst = 0.0
    for pnl in pnls:
        equity += float(pnl)
        peak = max(peak, equity)
        if peak <= 0:
            continue
        dd = max(0.0, (peak - equity) / peak * 100.0)
        worst = max(worst, dd)
    return worst


def evaluate_go_live_gate(conn: sqlite3.Connection, settings: Settings) -> GateResult:
    floor_ts = _iso_floor(settings.paper_window_days)

    labeled_cycles_row = conn.execute(
        """
        select count(*) as c
        from cycles
        where decision_ts >= ? and label_up is not null
        """,
        (floor_ts,),
    ).fetchone()
    labeled_cycles = int(labeled_cycles_row["c"] if labeled_cycles_row else 0)

    executed_row = conn.execute(
        """
        select count(*) as c
        from trades
        where mode = 'paper'
          and created_at >= ?
          and status <> 'no_fill'
        """,
        (floor_ts,),
    ).fetchone()
    executed_trades = int(executed_row["c"] if executed_row else 0)

    settled_rows = conn.execute(
        """
        select pnl_cents, fees_cents
        from trades
        where mode = 'paper'
          and created_at >= ?
          and status in ('settled_win', 'settled_loss')
        order by created_at asc
        """,
        (floor_ts,),
    ).fetchall()

    settled_pnls = [int(r["pnl_cents"] or 0) for r in settled_rows]
    settled_fees = [int(r["fees_cents"] or 0) for r in settled_rows]
    settled_trades = len(settled_pnls)

    total_pnl = float(sum(settled_pnls))
    total_cost = float(sum(settled_fees))
    net_after_cost = total_pnl - total_cost
    expectancy = net_after_cost / settled_trades if settled_trades > 0 else 0.0

    drawdown_window_days = max(1, int(settings.gate_drawdown_window_days or settings.paper_window_days))
    dd_floor_ts = _iso_floor(drawdown_window_days)
    dd_rows = conn.execute(
        """
        select pnl_cents
        from trades
        where mode = 'paper'
          and settled_at >= ?
          and status in ('settled_win', 'settled_loss')
        order by settled_at asc, created_at asc
        """,
        (dd_floor_ts,),
    ).fetchall()
    dd_pnls = [int(r["pnl_cents"] or 0) for r in dd_rows]
    max_dd = _max_drawdown_pct(dd_pnls, settings.paper_bankroll_cents) if dd_pnls else 0.0

    reasons: list[str] = []
    if labeled_cycles < settings.gate_min_labeled:
        reasons.append(f"labeled {labeled_cycles}/{settings.gate_min_labeled}")
    if executed_trades < settings.gate_min_executed:
        reasons.append(f"executed {executed_trades}/{settings.gate_min_executed}")
    if net_after_cost <= settings.gate_min_net_pnl_cents:
        reasons.append(f"net_pnl {net_after_cost:.2f} <= {settings.gate_min_net_pnl_cents}")
    if expectancy < settings.gate_min_expectancy_cents:
        reasons.append(
            f"expectancy {expectancy:.3f} < {settings.gate_min_expectancy_cents:.3f}"
        )
    if max_dd > settings.gate_max_drawdown_pct:
        reasons.append(
            f"drawdown({drawdown_window_days}d) {max_dd:.2f}% > {settings.gate_max_drawdown_pct:.2f}%"
        )

    return GateResult(
        passed=len(reasons) == 0,
        labeled_cycles=labeled_cycles,
        executed_trades=executed_trades,
        settled_trades=settled_trades,
        net_pnl_cents_after_cost=net_after_cost,
        expectancy_cents=expectancy,
        max_drawdown_pct=max_dd,
        drawdown_window_days=drawdown_window_days,
        reasons=reasons,
    )


def evaluate_auto_pause(
    conn: sqlite3.Connection,
    settings: Settings,
    last_sync_ok_at: str | None,
) -> tuple[bool, list[str]]:
    reasons: list[str] = []

    rolling = conn.execute(
        """
        select pnl_cents, fees_cents
        from trades
        where status in ('settled_win', 'settled_loss')
          and mode in ('paper', 'live')
        order by settled_at desc, created_at desc
        limit ?
        """,
        (settings.rolling_expectancy_window,),
    ).fetchall()
    # Only enforce rolling expectancy once the window is fully populated.
    # Otherwise you end up auto-pausing on tiny samples (noise), which defeats paper mode.
    if rolling and len(rolling) >= settings.rolling_expectancy_window:
        values = [int(r["pnl_cents"] or 0) - int(r["fees_cents"] or 0) for r in rolling]
        exp = sum(values) / len(values)
        if exp < 0:
            reasons.append("rolling_50_expectancy_negative")

    day_floor = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
    today_rows = conn.execute(
        """
        select pnl_cents
        from trades
        where status in ('settled_win', 'settled_loss')
          and mode in ('paper', 'live')
          and settled_at >= ?
        order by settled_at asc
        """,
        (day_floor.isoformat(),),
    ).fetchall()
    day_pnls = [int(r["pnl_cents"] or 0) for r in today_rows]
    dd_today = _max_drawdown_pct(day_pnls, settings.paper_bankroll_cents)
    if dd_today > settings.max_daily_drawdown_pct:
        reasons.append("daily_drawdown_limit")

    # Hard stop: if today's net (settled pnl - fees) breaches a floor, pause.
    net_row = conn.execute(
        """
        select
          coalesce(sum(case when status in ('settled_win','settled_loss') then pnl_cents else 0 end), 0) as pnl,
          coalesce(sum(case when status in ('pending','filled','dry_run','settled_win','settled_loss') then fees_cents else 0 end), 0) as fees
        from trades
        where mode in ('paper', 'live')
          and created_at >= ?
        """,
        (day_floor.isoformat(),),
    ).fetchone()
    day_net_after_fees = int(net_row["pnl"] or 0) - int(net_row["fees"] or 0)
    if day_net_after_fees <= int(settings.hard_daily_net_stop_cents):
        reasons.append("daily_net_stop")

    # Hard stop: if model output has been effectively constant, pause to prevent blind firing.
    if int(settings.model_stuck_window) > 1:
        rows = conn.execute(
            """
            select model_prob
            from decisions
            where mode in ('paper', 'live')
              and acted = 1
            order by decided_at desc
            limit ?
            """,
            (int(settings.model_stuck_window),),
        ).fetchall()
        if len(rows) >= int(settings.model_stuck_window):
            vals = [float(r["model_prob"] or 0.0) for r in rows]
            if vals and (max(vals) - min(vals)) <= float(settings.model_stuck_epsilon):
                reasons.append("model_output_stuck")

    if last_sync_ok_at:
        try:
            last_sync = datetime.fromisoformat(last_sync_ok_at.replace("Z", "+00:00")).astimezone(
                timezone.utc
            )
            stale = (datetime.now(timezone.utc) - last_sync).total_seconds()
            if stale > settings.max_sync_stale_seconds:
                reasons.append("sync_stale")
        except ValueError:
            reasons.append("sync_timestamp_invalid")

    return len(reasons) > 0, reasons


def gate_to_dict(gate: GateResult) -> dict:
    return {
        "passed": gate.passed,
        "labeled_cycles": gate.labeled_cycles,
        "executed_trades": gate.executed_trades,
        "settled_trades": gate.settled_trades,
        "net_pnl_cents_after_cost": gate.net_pnl_cents_after_cost,
        "expectancy_cents": gate.expectancy_cents,
        "max_drawdown_pct": gate.max_drawdown_pct,
        "drawdown_window_days": gate.drawdown_window_days,
        "reasons": gate.reasons,
    }
