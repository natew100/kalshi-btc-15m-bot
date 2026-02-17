from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
import re
import sqlite3
from typing import Iterable


_REASON_KV_RE = re.compile(r"([a-zA-Z0-9_]+)=([^|]+)")


@dataclass(frozen=True)
class ReplayRow:
    event_id: str
    decided_at: str
    model_prob: float
    ask_yes: float
    ask_no: float
    spread: float
    label_up: int
    reason: str


@dataclass(frozen=True)
class ReplayScenario:
    name: str
    base_cost_cents: float
    min_edge_cents: float
    max_spread_cents: float
    max_overround_cents: float
    high_vol_rv60_threshold: float
    high_vol_edge_add_cents: float
    high_vol_max_spread_cents: float
    side_calib_min_samples: int
    side_calib_min_win_rate: float
    side_calib_edge_penalty_cents: float


@dataclass
class ReplayTrade:
    event_id: str
    decided_at: str
    side: str
    pnl_cents_after_cost: float
    rv_60s: float | None
    hour_utc: int


@dataclass
class ReplayMetrics:
    scenario: str
    rows: int
    traded: int
    wins: int
    losses: int
    win_rate: float
    net_after_cost_cents: float
    expectancy_cents: float
    max_drawdown_pct: float
    by_vol: dict[str, dict]
    by_hour_block: dict[str, dict]


def parse_reason_kv(reason: str | None) -> dict[str, str]:
    if not reason:
        return {}
    out: dict[str, str] = {}
    for key, value in _REASON_KV_RE.findall(reason):
        out[key] = value.strip()
    return out


def load_replay_rows(conn: sqlite3.Connection, lookback_days: int = 14) -> list[ReplayRow]:
    floor = (datetime.now(timezone.utc) - timedelta(days=max(1, int(lookback_days)))).isoformat()
    try:
        rows = conn.execute(
            """
            select
              d.event_id,
              d.decided_at,
              d.model_prob,
              d.ask_yes,
              d.ask_no,
              d.spread,
              coalesce(d.reason, '') as reason,
              c.label_up
            from decisions d
            join cycles c on c.event_id = d.event_id
            where d.mode = 'paper'
              and c.label_up is not null
              and d.decided_at >= ?
            order by d.decided_at asc
            """,
            (floor,),
        ).fetchall()
    except sqlite3.OperationalError:
        return []
    out: list[ReplayRow] = []
    for r in rows:
        out.append(
            ReplayRow(
                event_id=str(r["event_id"]),
                decided_at=str(r["decided_at"]),
                model_prob=float(r["model_prob"] or 0.5),
                ask_yes=float(r["ask_yes"] or 0.0),
                ask_no=float(r["ask_no"] or 0.0),
                spread=float(r["spread"] or 0.0),
                label_up=int(r["label_up"]),
                reason=str(r["reason"] or ""),
            )
        )
    return out


def _argmax_side(model_prob: float, ask_yes: float, ask_no: float, cost_cents: float) -> tuple[str, float]:
    ev_yes = (model_prob * 100.0) - ask_yes - cost_cents
    ev_no = ((1.0 - model_prob) * 100.0) - ask_no - cost_cents
    if ev_yes >= ev_no:
        return "yes", ev_yes
    return "no", ev_no


def _pnl_for_trade(side: str, label_up: int, ask_yes: float, ask_no: float, cost_cents: float) -> tuple[bool, float]:
    if side == "yes":
        won = label_up == 1
        gross = (100.0 - ask_yes) if won else (-ask_yes)
    else:
        won = label_up == 0
        gross = (100.0 - ask_no) if won else (-ask_no)
    return won, (gross - cost_cents)


def _max_drawdown_pct(pnls: Iterable[float], start_bankroll_cents: float = 200000.0) -> float:
    equity = float(start_bankroll_cents)
    peak = equity
    worst = 0.0
    for pnl in pnls:
        equity += float(pnl)
        if equity > peak:
            peak = equity
        if peak > 0:
            dd = max(0.0, (peak - equity) / peak * 100.0)
            if dd > worst:
                worst = dd
    return worst


def _bucket_vol(rv: float | None) -> str:
    if rv is None:
        return "unknown"
    if rv < 0.0001:
        return "low"
    if rv < 0.0004:
        return "med"
    return "high"


def _bucket_hour_block(hour_utc: int) -> str:
    if 0 <= hour_utc < 8:
        return "00-07"
    if 8 <= hour_utc < 16:
        return "08-15"
    return "16-23"


def replay_scenario(rows: list[ReplayRow], scenario: ReplayScenario) -> ReplayMetrics:
    side_hist = {
        "yes": {"wins": 0, "n": 0},
        "no": {"wins": 0, "n": 0},
    }

    trades: list[ReplayTrade] = []
    for row in rows:
        parsed = parse_reason_kv(row.reason)
        rv_60s: float | None = None
        if "rv_60s" in parsed:
            try:
                rv_60s = float(parsed["rv_60s"])
            except ValueError:
                rv_60s = None

        side, chosen_ev = _argmax_side(
            model_prob=row.model_prob,
            ask_yes=row.ask_yes,
            ask_no=row.ask_no,
            cost_cents=scenario.base_cost_cents,
        )

        side_n = int(side_hist[side]["n"])
        side_w = int(side_hist[side]["wins"])
        side_wr = (float(side_w) / float(side_n)) if side_n > 0 else 0.0

        min_edge = float(scenario.min_edge_cents)
        max_spread = float(scenario.max_spread_cents)

        if rv_60s is not None and rv_60s >= float(scenario.high_vol_rv60_threshold):
            min_edge += float(scenario.high_vol_edge_add_cents)
            max_spread = min(max_spread, float(scenario.high_vol_max_spread_cents))

        if side_n >= int(scenario.side_calib_min_samples) and side_wr < float(scenario.side_calib_min_win_rate):
            min_edge += float(scenario.side_calib_edge_penalty_cents)

        overround = (float(row.ask_yes) + float(row.ask_no)) - 100.0
        if chosen_ev < min_edge:
            continue
        if float(row.spread) > max_spread:
            continue
        if overround > float(scenario.max_overround_cents):
            continue

        won, pnl_after_cost = _pnl_for_trade(
            side=side,
            label_up=int(row.label_up),
            ask_yes=float(row.ask_yes),
            ask_no=float(row.ask_no),
            cost_cents=float(scenario.base_cost_cents),
        )
        side_hist[side]["n"] += 1
        if won:
            side_hist[side]["wins"] += 1

        hour_utc = datetime.fromisoformat(row.decided_at.replace("Z", "+00:00")).astimezone(timezone.utc).hour
        trades.append(
            ReplayTrade(
                event_id=row.event_id,
                decided_at=row.decided_at,
                side=side,
                pnl_cents_after_cost=pnl_after_cost,
                rv_60s=rv_60s,
                hour_utc=hour_utc,
            )
        )

    wins = sum(1 for t in trades if t.pnl_cents_after_cost > 0)
    losses = sum(1 for t in trades if t.pnl_cents_after_cost < 0)
    net = sum(t.pnl_cents_after_cost for t in trades)
    traded = len(trades)

    by_vol: dict[str, dict] = {}
    by_hour: dict[str, dict] = {}
    for t in trades:
        vol_key = _bucket_vol(t.rv_60s)
        hv = by_vol.setdefault(vol_key, {"n": 0, "net_cents": 0.0})
        hv["n"] += 1
        hv["net_cents"] += t.pnl_cents_after_cost

        hr_key = _bucket_hour_block(t.hour_utc)
        hh = by_hour.setdefault(hr_key, {"n": 0, "net_cents": 0.0})
        hh["n"] += 1
        hh["net_cents"] += t.pnl_cents_after_cost

    for bucket in (by_vol, by_hour):
        for v in bucket.values():
            n = int(v["n"])
            v["expectancy_cents"] = (float(v["net_cents"]) / float(n)) if n > 0 else 0.0

    return ReplayMetrics(
        scenario=scenario.name,
        rows=len(rows),
        traded=traded,
        wins=wins,
        losses=losses,
        win_rate=(float(wins) / float(traded)) if traded > 0 else 0.0,
        net_after_cost_cents=net,
        expectancy_cents=(float(net) / float(traded)) if traded > 0 else 0.0,
        max_drawdown_pct=_max_drawdown_pct([t.pnl_cents_after_cost for t in trades]),
        by_vol=by_vol,
        by_hour_block=by_hour,
    )


def scenario_grid(base: ReplayScenario) -> list[ReplayScenario]:
    out: list[ReplayScenario] = []
    edges = [base.min_edge_cents, base.min_edge_cents + 0.4, base.min_edge_cents + 0.8]
    spreads = [base.max_spread_cents, max(1.0, base.max_spread_cents - 0.5)]
    overrounds = [base.max_overround_cents, max(1.5, base.max_overround_cents - 0.5)]
    for e in edges:
        for s in spreads:
            for o in overrounds:
                out.append(
                    ReplayScenario(
                        **{
                            **asdict(base),
                            "name": f"edge{e:.1f}_spr{s:.1f}_ovr{o:.1f}",
                            "min_edge_cents": float(e),
                            "max_spread_cents": float(s),
                            "max_overround_cents": float(o),
                        }
                    )
                )
    return out


def stress_scenarios(base: ReplayScenario) -> list[ReplayScenario]:
    mults = [0.75, 1.0, 1.25, 1.5]
    out: list[ReplayScenario] = []
    for m in mults:
        out.append(
            ReplayScenario(
                **{
                    **asdict(base),
                    "name": f"cost_x{m:.2f}",
                    "base_cost_cents": float(base.base_cost_cents) * float(m),
                }
            )
        )
    return out
