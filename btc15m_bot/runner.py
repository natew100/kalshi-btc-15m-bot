from __future__ import annotations

import json
import math
import sys
import time
from datetime import datetime, timedelta, timezone
from urllib import request

from . import db
from .config import Settings, ensure_runtime_paths, load_settings
from .features import build_feature_row
from .gating import evaluate_auto_pause, evaluate_go_live_gate, gate_to_dict, is_evening_skip_window
from .hq_sync import _read_json, _write_json, push_settlements
from .modeling import load_model, maybe_train_model, predict_prob, top_feature_contributions, train_model
from .kalshi_markets import KalshiBTCClient
from .settlement import settle_trade_pnl
from .spot import SpotPriceClient
from .strategy import (
    build_trade_payload,
    estimate_execution_cost,
    should_trade,
    simulate_buy_fill,
    size_quantity,
    within_decision_window,
)

VERSION = "kalshi-btc15m-v1"


def _post_alert(settings: Settings, payload: dict) -> None:
    if not settings.alert_webhook_url:
        return
    body = json.dumps(payload).encode("utf-8")
    req = request.Request(
        settings.alert_webhook_url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with request.urlopen(req, timeout=5) as resp:
        _ = resp.read()


def _parse_iso(ts: str) -> datetime:
    return datetime.fromisoformat(ts.replace("Z", "+00:00")).astimezone(timezone.utc)


def _cycle_age_seconds(now: datetime, start_ts: str) -> float:
    return (now - _parse_iso(start_ts)).total_seconds()


def _daily_stats(conn, settings: Settings, modes: tuple[str, ...]) -> dict:
    floor = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
    placeholders = ",".join(["?"] * max(1, len(modes)))
    mode_params = list(modes) if modes else []
    rows = conn.execute(
        f"""
        select status, pnl_cents, fees_cents
        from trades
        where created_at >= ?
          and mode in ({placeholders})
        """,
        (floor.isoformat(), *mode_params),
    ).fetchall()

    wins = sum(1 for r in rows if r["status"] == "settled_win")
    losses = sum(1 for r in rows if r["status"] == "settled_loss")
    filled_like = sum(
        1 for r in rows if r["status"] in ("pending", "filled", "dry_run", "settled_win", "settled_loss")
    )
    settled_pnl = sum(int(r["pnl_cents"] or 0) for r in rows if r["status"] in ("settled_win", "settled_loss"))
    fees = sum(
        int(r["fees_cents"] or 0)
        for r in rows
        if r["status"] in ("pending", "filled", "dry_run", "settled_win", "settled_loss")
    )

    total_settled_pnl = conn.execute(
        f"""
        select coalesce(sum(pnl_cents), 0) as s
        from trades
        where status in ('settled_win', 'settled_loss')
          and mode in ({placeholders})
        """,
        (*mode_params,),
    ).fetchone()
    bankroll = settings.paper_bankroll_cents + int(total_settled_pnl["s"] if total_settled_pnl else 0) - int(fees)

    return {
        "date": floor.date().isoformat(),
        "total_trades": filled_like,
        "wins": wins,
        "losses": losses,
        # Net P&L for the day (after fees).
        "pnl": (settled_pnl - fees) / 100.0,
        "fees": fees / 100.0,
        "circuit_breaker": False,
        "bankroll": bankroll / 100.0,
    }


def _portfolio_kpis(conn, settings: Settings, modes: tuple[str, ...]) -> dict:
    floor = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
    placeholders = ",".join(["?"] * max(1, len(modes)))
    mode_params = list(modes) if modes else []

    rows = conn.execute(
        f"""
        select status, pnl_cents, fees_cents
        from trades
        where created_at >= ?
          and mode in ({placeholders})
        """,
        (floor.isoformat(), *mode_params),
    ).fetchall()

    filled_like = sum(
        1 for r in rows if r["status"] in ("pending", "filled", "dry_run", "settled_win", "settled_loss")
    )
    no_fill = sum(1 for r in rows if r["status"] == "no_fill")
    wins = sum(1 for r in rows if r["status"] == "settled_win")
    losses = sum(1 for r in rows if r["status"] == "settled_loss")
    settled = wins + losses
    pnl_cents = sum(int(r["pnl_cents"] or 0) for r in rows if r["status"] in ("settled_win", "settled_loss"))

    # Net after fees (integer cents). Slippage is already baked into price_cents when L3 fill happens.
    fees_cents = sum(
        int(r["fees_cents"] or 0)
        for r in rows
        if r["status"] in ("pending", "filled", "dry_run", "settled_win", "settled_loss")
    )
    net_after_cost_cents = float(pnl_cents) - float(fees_cents)
    expectancy_cents = (net_after_cost_cents / float(filled_like)) if filled_like > 0 else 0.0
    fill_rate = (float(filled_like) / float(filled_like + no_fill)) if (filled_like + no_fill) > 0 else 0.0

    return {
        "date": floor.date().isoformat(),
        "modes": list(modes),
        "attempted": filled_like + no_fill,
        "filled_like": filled_like,
        "no_fill": no_fill,
        "fill_rate": fill_rate,
        "settled": settled,
        "wins": wins,
        "losses": losses,
        "pnl_cents": pnl_cents,
        "fees_cents": fees_cents,
        "net_after_cost_cents": net_after_cost_cents,
        "expectancy_cents": expectancy_cents,
    }


def _portfolio_cash_state(
    conn,
    settings: Settings,
    modes: tuple[str, ...],
) -> dict[str, int]:
    placeholders = ",".join(["?"] * max(1, len(modes)))
    mode_params = list(modes) if modes else []

    settled_row = conn.execute(
        f"""
        select coalesce(sum(pnl_cents), 0) as s
        from trades
        where status in ('settled_win', 'settled_loss')
          and mode in ({placeholders})
        """,
        (*mode_params,),
    ).fetchone()
    settled_pnl = int(settled_row["s"] if settled_row else 0)

    open_row = conn.execute(
        f"""
        select coalesce(sum(price_cents * quantity), 0) as s
        from trades
        where status in ('pending', 'filled', 'dry_run')
          and mode in ({placeholders})
        """,
        (*mode_params,),
    ).fetchone()
    locked = int(open_row["s"] if open_row else 0)

    fee_row = conn.execute(
        f"""
        select coalesce(sum(fees_cents), 0) as s
        from trades
        where status in ('pending', 'filled', 'dry_run', 'settled_win', 'settled_loss')
          and mode in ({placeholders})
        """,
        (*mode_params,),
    ).fetchone()
    fees_paid = int(fee_row["s"] if fee_row else 0)

    equity = int(settings.paper_bankroll_cents) + int(settled_pnl) - int(fees_paid)
    cash = int(equity) - int(locked)
    return {
        "equity_cents": equity,
        "cash_cents": cash,
        "locked_cents": locked,
        "settled_pnl_cents": settled_pnl,
    }


def _execution_metrics(conn, floor_iso: str, modes: tuple[str, ...]) -> dict:
    placeholders = ",".join(["?"] * max(1, len(modes)))
    mode_params = list(modes) if modes else []
    rows = conn.execute(
        f"""
        select status, quantity, requested_qty, filled_qty, fill_tier, fill_slippage_cents
        from trades
        where created_at >= ?
          and mode in ({placeholders})
        """,
        (floor_iso, *mode_params),
    ).fetchall()

    attempted = 0
    requested_total = 0
    filled_total = 0
    l1 = 0
    l3 = 0
    no_fill = 0
    slippage_sum = 0.0
    slippage_n = 0

    for r in rows:
        status = str(r["status"] or "")
        if status not in ("pending", "filled", "dry_run", "settled_win", "settled_loss", "no_fill"):
            continue
        qty = int(r["quantity"] or 0)
        req = int(r["requested_qty"] or qty)
        fill = int(r["filled_qty"] or (qty if status != "no_fill" else 0))
        attempted += 1
        requested_total += req
        filled_total += fill
        tier = str(r["fill_tier"] or "")
        if tier == "l1":
            l1 += 1
        elif tier == "l3":
            l3 += 1
        if status == "no_fill" or fill <= 0:
            no_fill += 1
        slip = r["fill_slippage_cents"]
        if slip is not None:
            slippage_sum += float(slip)
            slippage_n += 1

    fill_rate = (float(filled_total) / float(requested_total)) if requested_total > 0 else 0.0
    return {
        "attempted_trades": attempted,
        "requested_qty": requested_total,
        "filled_qty": filled_total,
        "fill_rate_qty": fill_rate,
        "no_fill_trades": no_fill,
        "tier_l1": l1,
        "tier_l3": l3,
        "avg_slippage_cents": (slippage_sum / slippage_n) if slippage_n > 0 else 0.0,
    }


def _calibration_metrics(conn, floor_iso: str, modes: tuple[str, ...]) -> dict:
    placeholders = ",".join(["?"] * max(1, len(modes)))
    mode_params = list(modes) if modes else []
    rows = conn.execute(
        f"""
        select side, model_prob, status
        from trades
        where created_at >= ?
          and mode in ({placeholders})
          and status in ('settled_win', 'settled_loss')
        """,
        (floor_iso, *mode_params),
    ).fetchall()

    if not rows:
        return {"settled": 0, "brier": None, "bins": []}

    brier_sum = 0.0
    bins = [{"n": 0, "wins": 0, "pred_sum": 0.0} for _ in range(10)]

    for r in rows:
        side = str(r["side"] or "yes")
        p_up = float(r["model_prob"] or 0.5)
        p_side = p_up if side == "yes" else (1.0 - p_up)
        p_side = max(0.0, min(1.0, p_side))
        y = 1.0 if str(r["status"]) == "settled_win" else 0.0
        brier_sum += (p_side - y) ** 2

        idx = min(9, int(p_side * 10.0))
        bins[idx]["n"] += 1
        bins[idx]["wins"] += int(y)
        bins[idx]["pred_sum"] += p_side

    out_bins: list[dict] = []
    for i, b in enumerate(bins):
        if b["n"] == 0:
            continue
        out_bins.append(
            {
                "lo": round(i / 10.0, 2),
                "hi": round((i + 1) / 10.0, 2),
                "n": int(b["n"]),
                "actual_win_rate": float(b["wins"]) / float(b["n"]),
                "avg_pred": float(b["pred_sum"]) / float(b["n"]),
            }
        )

    return {
        "settled": len(rows),
        "brier": brier_sum / float(len(rows)),
        "bins": out_bins,
    }


def _extreme_prob_guard_stats(
    conn,
    *,
    mode: str,
    side_prob_threshold: float,
    lookback_days: int,
) -> tuple[int, float]:
    floor = (datetime.now(timezone.utc) - timedelta(days=max(1, int(lookback_days)))).isoformat()
    row = conn.execute(
        """
        select
          count(*) as n,
          coalesce(avg(case when status = 'settled_win' then 1.0 else 0.0 end), 0.0) as wr
        from (
          select
            status,
            case
              when side = 'yes' then model_prob
              else (1.0 - model_prob)
            end as side_prob
          from trades
          where mode = ?
            and status in ('settled_win', 'settled_loss')
            and created_at >= ?
        ) t
        where side_prob >= ?
        """,
        (mode, floor, float(side_prob_threshold)),
    ).fetchone()
    if not row:
        return 0, 0.0
    return int(row["n"] or 0), float(row["wr"] or 0.0)


def _side_calibration_stats(
    conn,
    *,
    mode: str,
    side: str,
    lookback_days: int,
) -> tuple[int, float]:
    floor = (datetime.now(timezone.utc) - timedelta(days=max(1, int(lookback_days)))).isoformat()
    row = conn.execute(
        """
        select
          count(*) as n,
          coalesce(avg(case when status = 'settled_win' then 1.0 else 0.0 end), 0.0) as wr
        from trades
        where mode = ?
          and side = ?
          and status in ('settled_win', 'settled_loss')
          and created_at >= ?
        """,
        (mode, side, floor),
    ).fetchone()
    if not row:
        return 0, 0.0
    return int(row["n"] or 0), float(row["wr"] or 0.0)


def _dynamic_thresholds(
    settings: Settings,
    *,
    rv_60s: float,
    side_samples: int,
    side_win_rate: float,
) -> tuple[float, float, list[str]]:
    min_edge = float(settings.min_edge_cents)
    max_spread = float(settings.max_spread_cents)
    tags: list[str] = []

    if float(rv_60s) >= float(settings.high_vol_rv60_threshold):
        min_edge += float(settings.high_vol_min_edge_add_cents)
        max_spread = min(max_spread, float(settings.high_vol_max_spread_cents))
        tags.append("high_vol")

    if side_samples >= int(settings.side_calib_min_samples) and side_win_rate < float(settings.side_calib_min_win_rate):
        min_edge += float(settings.side_calib_edge_penalty_cents)
        tags.append("side_calib_penalty")

    return min_edge, max_spread, tags


def _whatif_metrics(conn, *, mode: str, lookback_days: int, base_min_edge: float, base_max_spread: float) -> dict:
    floor = (datetime.now(timezone.utc) - timedelta(days=max(1, int(lookback_days)))).isoformat()
    rows = conn.execute(
        """
        select
          t.side,
          t.pnl_cents,
          t.fees_cents,
          d.ev_yes,
          d.ev_no,
          d.spread
        from trades t
        join decisions d
          on d.event_id = t.event_id
        where t.mode = ?
          and t.status in ('settled_win', 'settled_loss')
          and t.created_at >= ?
        order by t.created_at desc
        """,
        (mode, floor),
    ).fetchall()
    scenarios = [
        ("baseline", float(base_min_edge), float(base_max_spread)),
        ("strict_edge", float(base_min_edge) + 0.8, float(base_max_spread)),
        ("tight_spread", float(base_min_edge), max(1.0, float(base_max_spread) - 0.5)),
        ("strict_both", float(base_min_edge) + 0.8, max(1.0, float(base_max_spread) - 0.5)),
    ]
    out: dict[str, dict] = {}
    for name, min_edge, max_spread in scenarios:
        settled = 0
        net = 0.0
        for r in rows:
            side = str(r["side"] or "yes")
            chosen_ev = float(r["ev_yes"] or 0.0) if side == "yes" else float(r["ev_no"] or 0.0)
            spread = float(r["spread"] or 0.0)
            if chosen_ev < min_edge or spread > max_spread:
                continue
            settled += 1
            net += float(r["pnl_cents"] or 0) - float(r["fees_cents"] or 0)
        out[name] = {
            "min_edge_cents": min_edge,
            "max_spread_cents": max_spread,
            "settled_trades": settled,
            "net_after_cost_cents": net,
            "expectancy_cents": (net / settled) if settled > 0 else 0.0,
        }
    return {
        "lookback_days": int(lookback_days),
        "base_settled_trades": len(rows),
        "scenarios": out,
    }


def _status_payload(
    settings: Settings,
    mode: str,
    gate,
    paused: bool,
    pause_reasons: list[str],
    active_event_id: str | None,
    cycle_age: float | None,
    last_error: str | None,
    conn,
    *,
    has_model: bool = True,
) -> dict:
    sync_state = _read_json(settings.sync_state_path)
    # "gate" represents the real strategy portfolio for the current mode.
    # Don't mix paper/live stats; it makes monitoring misleading.
    gate_modes = ("paper",) if mode == "paper" else ("live",)
    daily_gate = _daily_stats(conn, settings, modes=gate_modes)
    payload: dict = {
        "daily": daily_gate,
        "portfolios": {
            "gate": _portfolio_kpis(conn, settings, modes=gate_modes),
        },
    }
    payload["portfolios"]["gate"]["cash"] = _portfolio_cash_state(conn, settings, modes=gate_modes)

    day_floor = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0).isoformat()
    payload["diagnostics"] = {
        "execution": {
            "gate": _execution_metrics(conn, day_floor, modes=gate_modes),
        },
        "calibration": {
            "gate": _calibration_metrics(conn, day_floor, modes=gate_modes),
        },
        "whatif": {
            "gate": _whatif_metrics(
                conn,
                mode=mode,
                lookback_days=settings.whatif_lookback_days,
                base_min_edge=settings.min_edge_cents,
                base_max_spread=settings.max_spread_cents,
            )
        },
    }
    side_yes_n, side_yes_wr = _side_calibration_stats(
        conn, mode=mode, side="yes", lookback_days=settings.side_calib_lookback_days
    )
    side_no_n, side_no_wr = _side_calibration_stats(
        conn, mode=mode, side="no", lookback_days=settings.side_calib_lookback_days
    )
    payload["diagnostics"]["calibration"]["side"] = {
        "lookback_days": settings.side_calib_lookback_days,
        "yes": {"n": side_yes_n, "win_rate": side_yes_wr},
        "no": {"n": side_no_n, "win_rate": side_no_wr},
    }

    warnings = [r for r in pause_reasons if str(r).startswith("warn:")]
    if not has_model:
        warnings.insert(0, "no model — collecting training data")
    errors = [r for r in pause_reasons if not str(r).startswith("warn:")]
    data_sources = {
        "market_provider": settings.market_provider,
        "series_id": settings.series_id,
        "kalshi_series_ticker": settings.kalshi_series_ticker,
        "kalshi_base_url": settings.kalshi_base_url,
        "spot_sources": [
            settings.binance_ticker_url,
            settings.coinbase_ticker_url,
        ],
        "decision_offset_seconds": settings.decision_offset_seconds,
        "risk_filters": {
            "max_overround_cents": settings.max_overround_cents,
            "min_l1_depth_for_sizing_up": settings.min_l1_depth_for_sizing_up,
            "min_l3_depth_for_sizing_up": settings.min_l3_depth_for_sizing_up,
            "extreme_prob_threshold": settings.extreme_prob_threshold,
            "extreme_prob_min_samples": settings.extreme_prob_min_samples,
            "extreme_prob_min_win_rate": settings.extreme_prob_min_win_rate,
            "side_calib_min_samples": settings.side_calib_min_samples,
            "side_calib_min_win_rate": settings.side_calib_min_win_rate,
            "side_calib_edge_penalty_cents": settings.side_calib_edge_penalty_cents,
            "high_vol_rv60_threshold": settings.high_vol_rv60_threshold,
            "high_vol_min_edge_add_cents": settings.high_vol_min_edge_add_cents,
            "high_vol_max_spread_cents": settings.high_vol_max_spread_cents,
        },
    }

    return {
        "bot": {
            "status": "paused" if paused else "running",
            "updated_at": db.utcnow_iso(),
            "version": VERSION,
            "mode": mode,
        },
        "status": "paused" if paused else "running",
        "updated_at": db.utcnow_iso(),
        **payload,
        "signals": [] if not active_event_id else [{"event_id": active_event_id, "cycle_age_sec": cycle_age}],
        "errors": ([last_error] if last_error else []) + errors,
        "warnings": warnings,
        "data_sources": data_sources,
        "gate": gate_to_dict(gate),
        "sync": sync_state,
        "evening_skip": is_evening_skip_window(
            datetime.now(timezone.utc),
            start_hour_et=settings.evening_skip_start_hour_et,
            end_hour_et=settings.evening_skip_end_hour_et,
        ),
        "evening_skip_window_et": f"{settings.evening_skip_start_hour_et}:00-{settings.evening_skip_end_hour_et % 24}:00",
    }


def _append_audit_snapshot(settings: Settings, status_payload: dict, now_utc: datetime) -> None:
    # Immutable append-only hourly snapshots for forensic analysis.
    reports_dir = settings.data_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    ndjson_path = reports_dir / "hourly_metrics.ndjson"
    snap = {
        "ts": now_utc.isoformat(),
        "status": status_payload.get("status"),
        "daily": status_payload.get("daily"),
        "gate": status_payload.get("gate"),
        "portfolios": status_payload.get("portfolios"),
        "diagnostics": status_payload.get("diagnostics"),
        "warnings": status_payload.get("warnings"),
        "errors": status_payload.get("errors"),
    }
    with ndjson_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(snap, sort_keys=True) + "\n")


def _iso_z(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _settle_due_cycles(conn, settings: Settings, market_client: KalshiBTCClient) -> None:
    # Gamma can lag a bit; settle only for ended cycles with a small grace.
    before_ts = _iso_z(datetime.now(timezone.utc) - timedelta(seconds=15))
    due = db.unresolved_cycles_ended_before(conn, before_ts=before_ts, limit=5)
    if not due:
        return

    settlements: list[dict] = []
    for row in due:
        event_id = str(row["event_id"])
        label, _event = market_client.event_resolution_label(event_id)
        if label is None:
            continue

        resolution_ts = db.utcnow_iso()
        db.resolve_cycle(conn, event_id, label, resolution_ts)

        for trade in db.get_trades_for_event(conn, event_id):
            if trade["status"] not in ("pending", "filled", "dry_run"):
                continue
            settled_status, pnl = settle_trade_pnl(
                label_up=label,
                side=str(trade["side"]),
                price_cents=int(trade["price_cents"]),
                quantity=int(trade["quantity"]),
            )
            settled_at = db.utcnow_iso()
            db.settle_trade(
                conn,
                external_id=str(trade["external_id"]),
                status=settled_status,
                pnl_cents=pnl,
                settled_at=settled_at,
                settlement_source="kalshi_official",
            )
            if str(trade["mode"]) in ("paper", "live"):
                settlements.append(
                    {
                        "external_id": str(trade["external_id"]),
                        "status": settled_status,
                        "pnl_cents": pnl,
                        "settled_at": settled_at,
                        "settlement_source": "kalshi_official",
                    }
                )

    # Fast-path settlement sync; regular sync will reconcile the full trade state.
    if settlements:
        push_settlements(
            settle_url=settings.hq_settle_url,
            api_key=settings.hq_api_key,
            bot_name=settings.bot_name,
            settlements=settlements,
        )


def run_forever() -> int:
    settings = load_settings()
    ensure_runtime_paths(settings)

    conn = db.connect_db(settings.db_path)
    db.init_db(conn)

    market_client = KalshiBTCClient(settings=settings, timeout=8)
    spot_client = SpotPriceClient(settings=settings, timeout=3)

    last_error: str | None = None
    last_decision_reason: str | None = None
    last_settle_scan_at = datetime.min.replace(tzinfo=timezone.utc)
    last_cleanup_at = datetime.min.replace(tzinfo=timezone.utc)
    last_train_attempt_at: datetime | None = None
    last_audit_snapshot_at = datetime.min.replace(tzinfo=timezone.utc)
    last_pause_signature = ""
    last_real_decision_ts: str | None = None

    ok, message = maybe_train_model(conn, settings)
    if ok:
        print(message)
    model_bundle = load_model(settings.model_path)

    while True:
        loop_started = datetime.now(timezone.utc)
        try:
            # If we don't have a model yet, try periodically so trading can start as soon as data exists.
            if model_bundle is None:
                if last_train_attempt_at is None or (loop_started - last_train_attempt_at) >= timedelta(minutes=10):
                    trained, msg = train_model(conn, settings)
                    if trained:
                        print(msg)
                        model_bundle = load_model(settings.model_path)
                    last_train_attempt_at = loop_started

            if settings.auto_retrain and loop_started.minute == 0 and loop_started.second < settings.poll_interval_seconds:
                retrained, msg = maybe_train_model(conn, settings)
                if retrained:
                    print(msg)
                    model_bundle = load_model(settings.model_path)

            # Resolve/settle older cycles; the "active event" rolls continuously every 5 minutes.
            if (loop_started - last_settle_scan_at).total_seconds() >= 5:
                _settle_due_cycles(conn, settings, market_client)
                last_settle_scan_at = loop_started

            # High-cadence tables need retention or the DB will grow forever.
            if (
                settings.tick_retention_days > 0
                and settings.cleanup_interval_seconds > 0
                and (loop_started - last_cleanup_at).total_seconds() >= settings.cleanup_interval_seconds
            ):
                floor_ts = (loop_started - timedelta(days=settings.tick_retention_days)).isoformat()
                deleted = db.cleanup_ticks_older_than(conn, floor_ts=floor_ts)
                if deleted:
                    print(f"tick retention cleanup deleted={deleted}")
                last_cleanup_at = loop_started

            active_event = market_client.get_active_event(settings.series_id)
            gate = evaluate_go_live_gate(conn, settings)
            sync_state = _read_json(settings.sync_state_path)
            last_sync_ok_at = sync_state.get("last_sync_ok_at")

            mode = settings.bot_mode
            if mode == "paper" and settings.auto_promote_live and gate.passed:
                mode = "live"

            paused, pause_reasons = evaluate_auto_pause(conn, settings, last_sync_ok_at)
            if mode == "paper":
                # Paper mode is R&D. Keep trading so we gather data, but surface risk signals as warnings.
                hard = {"sync_stale", "sync_timestamp_invalid", "daily_net_stop", "model_output_stuck"}
                hard_reasons = [r for r in pause_reasons if r in hard]
                soft_reasons = [r for r in pause_reasons if r not in hard]
                paused = len(hard_reasons) > 0
                pause_reasons = hard_reasons + [f"warn:{r}" for r in soft_reasons]
            pause_signature = "|".join(sorted(pause_reasons)) if pause_reasons else "ok"
            if pause_signature != last_pause_signature:
                try:
                    _post_alert(
                        settings,
                        {
                            "bot": settings.bot_name,
                            "status": "paused" if paused else "running",
                            "reasons": pause_reasons,
                            "at": loop_started.isoformat(),
                        },
                    )
                except Exception as exc:
                    print(f"alert_webhook_error: {exc}", file=sys.stderr)
                last_pause_signature = pause_signature

            if not active_event:
                status = _status_payload(
                    settings=settings,
                    mode=mode,
                    gate=gate,
                    paused=paused,
                    pause_reasons=pause_reasons,
                    active_event_id=None,
                    cycle_age=None,
                    last_error=last_error,
                    conn=conn,
                    has_model=model_bundle is not None,
                )
                _write_json(settings.status_path, status)
                last_error = None
                time.sleep(max(1, settings.poll_interval_seconds))
                continue

            snapshot = market_client.market_snapshot(active_event)
            if snapshot is None:
                last_error = "snapshot_unavailable"
                time.sleep(max(1, settings.poll_interval_seconds))
                continue

            db.upsert_cycle(
                conn,
                {
                    "event_id": snapshot.event_id,
                    "market_id": snapshot.market_id,
                    "slug": snapshot.slug,
                    "ticker": snapshot.ticker,
                    "start_ts": snapshot.start_ts,
                    "end_ts": snapshot.end_ts,
                },
            )

            now_iso = loop_started.isoformat()
            market_open = _parse_iso(snapshot.start_ts)
            cycle_age = _cycle_age_seconds(loop_started, snapshot.start_ts)

            decided = db.cycle_already_decided(conn, snapshot.event_id)

            # After decision is made, skip tick collection — no value in polling
            # the same event repeatedly.  Just wait for settlement.
            if decided:
                time.sleep(max(5, settings.poll_interval_seconds))
                continue

            spot_price = spot_client.blended_btc_price()
            if spot_price:
                db.insert_spot_tick(conn, now_iso, spot_price)

            db.insert_tick(
                conn,
                snapshot.event_id,
                {
                    "ts": now_iso,
                    "best_bid": snapshot.best_bid,
                    "best_ask": snapshot.best_ask,
                    "last_trade_price": snapshot.last_trade_price,
                    "spread": snapshot.spread,
                    "mid_yes": snapshot.mid_yes,
                    "bid_sz_l1": snapshot.bid_sz_l1,
                    "ask_sz_l1": snapshot.ask_sz_l1,
                    "bid_sz_l3": snapshot.bid_sz_l3,
                    "ask_sz_l3": snapshot.ask_sz_l3,
                    "imbalance_l1": snapshot.imbalance_l1,
                    "imbalance_l3": snapshot.imbalance_l3,
                    "spot_price": spot_price,
                },
            )
            if not decided and loop_started >= market_open + timedelta(seconds=settings.decision_offset_seconds):
                ticks = [dict(r) for r in db.all_ticks_for_event(conn, snapshot.event_id)]
                prev = db.recent_outcomes(conn, before_ts=now_iso, limit=3)
                features = build_feature_row(ticks=ticks, decision_ts=now_iso, prev_outcomes_desc=prev)
                db.insert_feature_row(conn, snapshot.event_id, now_iso, features)
                db.set_cycle_decision_ts(conn, snapshot.event_id, now_iso)

                # No model trained yet — collect data only, don't trade on coinflip.
                if model_bundle is None:
                    print(f"no_model event={snapshot.event_id} — collecting data only")
                    db.insert_decision(
                        conn,
                        {
                            "event_id": snapshot.event_id,
                            "trade_external_id": None,
                            "decided_at": now_iso,
                            "model_prob": 0.5,
                            "ask_yes": float(snapshot.best_ask),
                            "ask_no": max(0.0, 100.0 - float(snapshot.best_bid)),
                            "spread": float(snapshot.spread),
                            "ev_yes": 0.0,
                            "ev_no": 0.0,
                            "chosen_side": None,
                            "acted": False,
                            "reason": "no_model",
                            "mode": mode,
                            "cost_cents": 0.0,
                        },
                    )
                    last_real_decision_ts = now_iso
                    last_decision_reason = "no_model"

            if not decided and model_bundle is not None and loop_started >= market_open + timedelta(seconds=settings.decision_offset_seconds):
                prob_up = predict_prob(model_bundle, features)
                top_attr = top_feature_contributions(model_bundle, features, top_k=3)
                attr_text = ",".join([f"{k}:{v:+.3f}" for k, v in top_attr]) if top_attr else "none"
                rv_60s = float(features.get("rv_60s", 0.0))
                qty_probe = 1

                # Execution realism: model decision is made at a fixed offset, but fills occur with latency.
                exec_snapshot = snapshot
                if settings.execution_latency_seconds > 0:
                    time.sleep(float(settings.execution_latency_seconds))
                    latest = market_client.get_event(snapshot.event_id)
                    if latest is not None:
                        refreshed = market_client.market_snapshot(latest)
                        if refreshed is not None:
                            exec_snapshot = refreshed

                ask_yes = float(exec_snapshot.best_ask)
                bid_yes = float(exec_snapshot.best_bid)
                ask_no = (
                    float(exec_snapshot.best_ask_no)
                    if exec_snapshot.best_ask_no is not None
                    else max(0.0, 100.0 - bid_yes)
                )
                bid_no = (
                    float(exec_snapshot.best_bid_no)
                    if exec_snapshot.best_bid_no is not None
                    else max(0.0, 100.0 - ask_yes)
                )

                spread_yes = max(0.0, ask_yes - bid_yes)
                spread_no = max(0.0, ask_no - bid_no)
                overround_cents = (ask_yes + ask_no) - 100.0
                overround_known = exec_snapshot.ask_sz_l1_no is not None and exec_snapshot.ask_sz_l3_no is not None

                fill_yes = simulate_buy_fill(
                    best_ask_cents=ask_yes,
                    quantity=qty_probe,
                    ask_sz_l1=exec_snapshot.ask_sz_l1,
                    ask_sz_l3=exec_snapshot.ask_sz_l3,
                    l3_slippage_cents=settings.l3_slippage_cents,
                )
                fill_no = simulate_buy_fill(
                    best_ask_cents=ask_no,
                    quantity=qty_probe,
                    ask_sz_l1=exec_snapshot.ask_sz_l1_no,
                    ask_sz_l3=exec_snapshot.ask_sz_l3_no,
                    l3_slippage_cents=settings.l3_slippage_cents,
                )

                # Book EV (for logging/debug).
                yes_cost_probe = estimate_execution_cost(
                    price_cents=fill_yes.price_cents,
                    quantity=qty_probe,
                    fill_tier=fill_yes.tier,
                    fee_fixed_cents_per_contract=settings.fee_fixed_cents_per_contract,
                    fee_bps_per_notional=settings.fee_bps_per_notional,
                    l1_slippage_buffer_cents=settings.l1_slippage_buffer_cents,
                    l3_slippage_buffer_cents=settings.l3_slippage_buffer_cents,
                )
                no_cost_probe = estimate_execution_cost(
                    price_cents=fill_no.price_cents,
                    quantity=qty_probe,
                    fill_tier=fill_no.tier,
                    fee_fixed_cents_per_contract=settings.fee_fixed_cents_per_contract,
                    fee_bps_per_notional=settings.fee_bps_per_notional,
                    l1_slippage_buffer_cents=settings.l1_slippage_buffer_cents,
                    l3_slippage_buffer_cents=settings.l3_slippage_buffer_cents,
                )
                book_ev_yes = (prob_up * 100.0) - ask_yes - yes_cost_probe.per_contract_cost_cents
                book_ev_no = ((1.0 - prob_up) * 100.0) - ask_no - no_cost_probe.per_contract_cost_cents

                # Fill EV (for trade selection/gating). If the side can't fill, treat as untradeable.
                fill_px_yes = float(fill_yes.price_cents) if fill_yes.filled else ask_yes
                fill_px_no = float(fill_no.price_cents) if fill_no.filled else ask_no
                fill_ev_yes = (prob_up * 100.0) - fill_px_yes - yes_cost_probe.per_contract_cost_cents
                fill_ev_no = ((1.0 - prob_up) * 100.0) - fill_px_no - no_cost_probe.per_contract_cost_cents

                choice_ev_yes = fill_ev_yes if fill_yes.filled else -1e9
                choice_ev_no = fill_ev_no if fill_no.filled else -1e9

                if choice_ev_yes >= choice_ev_no:
                    chosen_side = "yes"
                    chosen_ev = choice_ev_yes
                    chosen_spread = spread_yes
                    chosen_fill = fill_yes
                    chosen_cost = yes_cost_probe
                else:
                    chosen_side = "no"
                    chosen_ev = choice_ev_no
                    chosen_spread = spread_no
                    chosen_fill = fill_no
                    chosen_cost = no_cost_probe
                chosen_side_prob = prob_up if chosen_side == "yes" else (1.0 - prob_up)
                side_n, side_wr = _side_calibration_stats(
                    conn,
                    mode=mode,
                    side=chosen_side,
                    lookback_days=settings.side_calib_lookback_days,
                )
                dynamic_min_edge, dynamic_max_spread, threshold_tags = _dynamic_thresholds(
                    settings,
                    rv_60s=rv_60s,
                    side_samples=side_n,
                    side_win_rate=side_wr,
                )

                in_window = within_decision_window(
                    now_utc=loop_started,
                    market_open_utc=market_open,
                    start_seconds=settings.decision_window_start_seconds,
                    end_seconds=settings.decision_window_end_seconds,
                )
                trade_ok, reason = should_trade(
                    chosen_ev_cents=chosen_ev,
                    spread_cents=chosen_spread,
                    in_window=in_window,
                    min_edge_cents=dynamic_min_edge,
                    max_spread_cents=dynamic_max_spread,
                    max_edge_cents=settings.max_edge_cents,
                )

                if not chosen_fill.filled:
                    trade_ok = False
                    reason = f"no_fill:{chosen_fill.reason}"

                if (
                    trade_ok
                    and overround_known
                    and settings.max_overround_cents > 0
                    and overround_cents > settings.max_overround_cents
                ):
                    trade_ok = False
                    reason = "overround_too_high"

                if (
                    trade_ok
                    and settings.extreme_prob_guard_enabled
                    and chosen_side_prob >= float(settings.extreme_prob_threshold)
                ):
                    n_extreme, wr_extreme = _extreme_prob_guard_stats(
                        conn,
                        mode=mode,
                        side_prob_threshold=float(settings.extreme_prob_threshold),
                        lookback_days=int(settings.extreme_prob_lookback_days),
                    )
                    if (
                        n_extreme < int(settings.extreme_prob_min_samples)
                        or wr_extreme < float(settings.extreme_prob_min_win_rate)
                    ):
                        trade_ok = False
                        reason = (
                            f"extreme_prob_uncalibrated:n={n_extreme}"
                            f",wr={wr_extreme:.3f}"
                        )

                if paused:
                    trade_ok = False
                    reason = f"auto_pause:{','.join(pause_reasons)}"
                evening_skip = is_evening_skip_window(
                    loop_started,
                    start_hour_et=settings.evening_skip_start_hour_et,
                    end_hour_et=settings.evening_skip_end_hour_et,
                )
                if trade_ok and evening_skip:
                    trade_ok = False
                    reason = "evening_skip"

                if mode == "live" and db.count_open_live_positions(conn) >= settings.max_open_live_positions:
                    trade_ok = False
                    reason = "live_position_cap"

                # Determine size for the gate portfolio (paper) after gating checks.
                qty_gate = None
                if trade_ok:
                    cash_gate = _portfolio_cash_state(conn, settings, modes=(mode,))
                    equity_gate = int(cash_gate["equity_cents"])
                    ref_px = float(fill_px_yes if chosen_side == "yes" else fill_px_no)
                    qty_gate = size_quantity(
                        sizing_mode=settings.sizing_mode,
                        fixed_contracts=settings.fixed_contracts,
                        max_contracts=settings.max_contracts,
                        risk_per_trade_pct=settings.risk_per_trade_pct,
                        equity_cents=equity_gate,
                        price_cents=ref_px,
                    )
                    qty_gate = min(int(qty_gate), int(settings.max_qty_per_signal), int(settings.max_contracts))
                    # Never size beyond observable L3 depth; otherwise we'd "no_fill_sized" constantly.
                    depth_l3 = (
                        exec_snapshot.ask_sz_l3 if chosen_side == "yes" else exec_snapshot.ask_sz_l3_no
                    )
                    if depth_l3 is not None:
                        qty_gate = max(1, min(int(qty_gate), int(float(depth_l3))))
                    depth_l1 = (
                        exec_snapshot.ask_sz_l1 if chosen_side == "yes" else exec_snapshot.ask_sz_l1_no
                    )
                    # Only allow scaling above 1 contract when both L1 and L3 depth are adequate.
                    # If depth is thin, degrade to qty=1 instead of forcing a full no-trade.
                    if int(qty_gate) > 1:
                        l1_ok = depth_l1 is not None and float(depth_l1) >= float(settings.min_l1_depth_for_sizing_up)
                        l3_ok = depth_l3 is not None and float(depth_l3) >= float(settings.min_l3_depth_for_sizing_up)
                        if not (l1_ok and l3_ok):
                            qty_gate = 1

                    # Re-simulate fill at sized quantity since L3 slippage depends on qty.
                    sized_fill = simulate_buy_fill(
                        best_ask_cents=ref_px,
                        quantity=qty_gate,
                        ask_sz_l1=exec_snapshot.ask_sz_l1 if chosen_side == "yes" else exec_snapshot.ask_sz_l1_no,
                        ask_sz_l3=exec_snapshot.ask_sz_l3 if chosen_side == "yes" else exec_snapshot.ask_sz_l3_no,
                        l3_slippage_cents=settings.l3_slippage_cents,
                    )
                    if not sized_fill.filled:
                        trade_ok = False
                        reason = f"no_fill_sized:{sized_fill.reason}"
                    else:
                        sized_px = float(sized_fill.price_cents)
                        # Recompute EV using sized fill price.
                        sized_cost = estimate_execution_cost(
                            price_cents=sized_px,
                            quantity=qty_gate,
                            fill_tier=sized_fill.tier,
                            fee_fixed_cents_per_contract=settings.fee_fixed_cents_per_contract,
                            fee_bps_per_notional=settings.fee_bps_per_notional,
                            l1_slippage_buffer_cents=settings.l1_slippage_buffer_cents,
                            l3_slippage_buffer_cents=settings.l3_slippage_buffer_cents,
                        )
                        chosen_ev = (
                            (prob_up * 100.0) - sized_px - sized_cost.per_contract_cost_cents
                            if chosen_side == "yes"
                            else ((1.0 - prob_up) * 100.0) - sized_px - sized_cost.per_contract_cost_cents
                        )
                        trade_ok, reason = should_trade(
                            chosen_ev_cents=chosen_ev,
                            spread_cents=chosen_spread,
                            in_window=in_window,
                            min_edge_cents=dynamic_min_edge,
                            max_spread_cents=dynamic_max_spread,
                            max_edge_cents=settings.max_edge_cents,
                        )
                        if trade_ok:
                            chosen_fill = sized_fill
                            chosen_cost = sized_cost
                            if chosen_side == "yes":
                                fill_px_yes = sized_px
                            else:
                                fill_px_no = sized_px

                # One line per cycle for auditability in journald.
                print(
                    "decision"
                    f" event={snapshot.event_id}"
                    f" mode={mode}"
                    f" p_up={prob_up:.4f}"
                    f" side={chosen_side}"
                    f" ev={chosen_ev:.3f}"
                    f" qty={(qty_gate if qty_gate is not None else settings.fixed_contracts)}"
                    f" ask_yes={ask_yes:.1f} ask_no={ask_no:.1f}"
                    f" spread_yes={spread_yes:.2f} spread_no={spread_no:.2f}"
                    f" overround={overround_cents:.2f}"
                    f" rv60={rv_60s:.6f}"
                    f" min_edge={dynamic_min_edge:.2f} max_spread={dynamic_max_spread:.2f}"
                    f" side_n={side_n} side_wr={side_wr:.3f}"
                    f" tags={','.join(threshold_tags) if threshold_tags else '-'}"
                    f" fill_yes={fill_yes.tier}:{fill_yes.reason}@{fill_px_yes:.1f}"
                    f" fill_no={fill_no.tier}:{fill_no.reason}@{fill_px_no:.1f}"
                    f" trade_ok={int(trade_ok)} reason={reason}"
                )

                trade_external_id = None
                if trade_ok:
                    if db.trade_count_for_event_modes(conn, snapshot.event_id, (mode,)) > 0:
                        trade_ok = False
                        reason = "cycle_write_lock"
                    elif db.trade_count_for_event_modes(
                        conn, snapshot.event_id, ("paper", "live")
                    ) >= int(settings.max_trades_per_cycle):
                        trade_ok = False
                        reason = "cycle_trade_cap"
                if trade_ok:
                    cash_gate = _portfolio_cash_state(conn, settings, modes=(mode,))
                    qty = min(
                        int(qty_gate if qty_gate is not None else settings.fixed_contracts),
                        int(settings.max_qty_per_signal),
                        int(settings.max_contracts),
                    )
                    max_loss_gate = int(round(float(fill_px_yes if chosen_side == "yes" else fill_px_no))) * qty
                    gate_cost = estimate_execution_cost(
                        price_cents=(fill_px_yes if chosen_side == "yes" else fill_px_no),
                        quantity=qty,
                        fill_tier=chosen_fill.tier,
                        fee_fixed_cents_per_contract=settings.fee_fixed_cents_per_contract,
                        fee_bps_per_notional=settings.fee_bps_per_notional,
                        l1_slippage_buffer_cents=settings.l1_slippage_buffer_cents,
                        l3_slippage_buffer_cents=settings.l3_slippage_buffer_cents,
                    )
                    fees_gate = int(gate_cost.fee_cents)
                    slippage_gate = int(gate_cost.slippage_buffer_cents)
                    if cash_gate["cash_cents"] < max_loss_gate:
                        trade_ok = False
                        reason = "insufficient_cash"
                    else:
                        trade = build_trade_payload(
                            event_id=snapshot.event_id,
                            ticker=snapshot.ticker,
                            side=chosen_side,
                            mode=mode,
                            quantity=qty,
                            model_prob_up=prob_up,
                            ask_yes_cents=fill_px_yes,
                            ask_no_cents=fill_px_no,
                            chosen_ev_cents=chosen_ev,
                            reason=(
                                f"book_ev_yes={book_ev_yes:.3f},book_ev_no={book_ev_no:.3f}"
                                f"|fill_ev_yes={fill_ev_yes:.3f},fill_ev_no={fill_ev_no:.3f}"
                                f"|spread_yes={spread_yes:.3f},spread_no={spread_no:.3f}"
                                f"|overround={overround_cents:.3f},overround_known={int(overround_known)}"
                                f"|fill={chosen_fill.tier}:{chosen_fill.reason}"
                                f"|rv_60s={rv_60s:.6f},min_edge={dynamic_min_edge:.3f},max_spread={dynamic_max_spread:.3f}"
                                f"|side_n={side_n},side_wr={side_wr:.3f},tags={','.join(threshold_tags) if threshold_tags else '-'}"
                                f"|attr={attr_text}"
                            ),
                        )
                        trade.update(
                            {
                                "balance_before_cents": cash_gate["cash_cents"],
                                "balance_after_cents": cash_gate["cash_cents"] - max_loss_gate - fees_gate,
                                "position_notional_cents": max_loss_gate,
                                "max_loss_cents": max_loss_gate,
                                "fees_cents": fees_gate,
                                "expected_fee_cents": fees_gate,
                                "realized_slippage_cents": slippage_gate,
                                "all_in_cost_cents": fees_gate + slippage_gate,
                                "requested_qty": qty,
                                "filled_qty": qty if chosen_fill.filled else 0,
                                "fill_tier": chosen_fill.tier,
                                "fill_reason": chosen_fill.reason,
                                "fill_slippage_cents": max(
                                    0.0,
                                    float(fill_px_yes if chosen_side == "yes" else fill_px_no)
                                    - float(ask_yes if chosen_side == "yes" else ask_no),
                                ),
                            }
                        )
                        trade["reasoning"] = (
                            f"{trade['reasoning']}|fee_cents={fees_gate}"
                            f"|slippage_buffer_cents={slippage_gate}"
                            f"|all_in_cost_cents={fees_gate + slippage_gate}"
                        )
                        if mode == "live":
                            # Live execution adapter is intentionally not coupled into this scaffold.
                            trade["status"] = "no_fill"
                            trade["reasoning"] = f"{trade['reasoning']}|live_adapter_not_configured"
                        db.insert_trade(conn, trade)
                        trade_external_id = trade["external_id"]

                db.insert_decision(
                    conn,
                    {
                        "event_id": snapshot.event_id,
                        "trade_external_id": trade_external_id,
                        "decided_at": now_iso,
                        "model_prob": prob_up,
                        "ask_yes": ask_yes,
                        "ask_no": ask_no,
                        "spread": chosen_spread,
                        "ev_yes": fill_ev_yes,
                        "ev_no": fill_ev_no,
                        "chosen_side": chosen_side if trade_ok else None,
                        "acted": trade_ok,
                        "reason": (
                            f"{reason}"
                            f"|rv_60s={rv_60s:.6f}"
                            f"|min_edge={dynamic_min_edge:.3f},max_spread={dynamic_max_spread:.3f}"
                            f"|side_n={side_n},side_wr={side_wr:.3f}"
                            f"|tags={','.join(threshold_tags) if threshold_tags else '-'}"
                            f"|attr={attr_text}"
                        ),
                        "mode": mode,
                        "cost_cents": chosen_cost.per_contract_cost_cents,
                    },
                )
                last_real_decision_ts = now_iso
                last_decision_reason = reason

            status = _status_payload(
                settings=settings,
                mode=mode,
                gate=gate,
                paused=paused,
                pause_reasons=pause_reasons,
                active_event_id=snapshot.event_id,
                cycle_age=cycle_age,
                last_error=last_error,
                conn=conn,
                has_model=model_bundle is not None,
            )
            if last_real_decision_ts:
                status["last_decision_ts"] = last_real_decision_ts
            if last_decision_reason is not None:
                status["last_decision_reason"] = last_decision_reason
            _write_json(settings.status_path, status)
            if settings.audit_snapshot_interval_seconds > 0:
                if (loop_started - last_audit_snapshot_at).total_seconds() >= settings.audit_snapshot_interval_seconds:
                    _append_audit_snapshot(settings, status, loop_started)
                    last_audit_snapshot_at = loop_started
            last_error = None

        except Exception as exc:
            last_error = str(exc)
            print(f"runner error: {exc}", file=sys.stderr)

        elapsed = (datetime.now(timezone.utc) - loop_started).total_seconds()
        sleep_for = max(0.0, settings.poll_interval_seconds - elapsed)
        time.sleep(sleep_for)


def main() -> int:
    return run_forever()


if __name__ == "__main__":
    raise SystemExit(main())
