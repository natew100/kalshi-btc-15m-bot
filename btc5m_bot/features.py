from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Iterable

FEATURE_COLUMNS = [
    "mid_yes",
    "spread",
    "imbalance_l1",
    "imbalance_l3",
    "ret_15s",
    "ret_30s",
    "ret_60s",
    "rv_60s",
    "tod_sin",
    "tod_cos",
    "prev1_up",
    "prev2_up",
    "prev3_up",
]


@dataclass
class TickPoint:
    ts: datetime
    mid_yes: float
    spread: float
    imbalance_l1: float
    imbalance_l3: float
    spot_price: float


def _parse_iso(ts: str) -> datetime:
    return datetime.fromisoformat(ts.replace("Z", "+00:00")).astimezone(timezone.utc)


def _normalize_tick(row: dict) -> TickPoint:
    return TickPoint(
        ts=_parse_iso(str(row["ts"])),
        mid_yes=float(row.get("mid_yes") or 0.5),
        spread=float(row.get("spread") or 0.0),
        imbalance_l1=float(row.get("imbalance_l1") or 0.0),
        imbalance_l3=float(row.get("imbalance_l3") or 0.0),
        spot_price=float(row.get("spot_price") or 0.0),
    )


def _find_price_at_or_before(points: list[TickPoint], target: datetime) -> float | None:
    best: float | None = None
    for p in points:
        if p.ts <= target and p.spot_price > 0:
            best = p.spot_price
        if p.ts > target:
            break
    return best


def _realized_vol(points: list[TickPoint], end_ts: datetime, lookback_sec: int = 60) -> float:
    start_ts = end_ts - timedelta(seconds=lookback_sec)
    window = [p for p in points if start_ts <= p.ts <= end_ts and p.spot_price > 0]
    if len(window) < 2:
        return 0.0

    rets: list[float] = []
    prev = window[0].spot_price
    for p in window[1:]:
        if prev > 0 and p.spot_price > 0:
            rets.append(math.log(p.spot_price / prev))
        prev = p.spot_price

    if len(rets) < 2:
        return 0.0
    mean = sum(rets) / len(rets)
    var = sum((r - mean) ** 2 for r in rets) / (len(rets) - 1)
    return math.sqrt(max(var, 0.0))


def build_feature_row(
    ticks: Iterable[dict],
    decision_ts: str,
    prev_outcomes_desc: list[int],
) -> dict[str, float]:
    points = [_normalize_tick(dict(t)) for t in ticks]
    points.sort(key=lambda p: p.ts)
    if not points:
        raise ValueError("no ticks available for feature generation")

    dt = _parse_iso(decision_ts)

    current = next((p for p in reversed(points) if p.ts <= dt), points[-1])
    p_now = current.spot_price if current.spot_price > 0 else None
    p_15 = _find_price_at_or_before(points, dt - timedelta(seconds=15))
    p_30 = _find_price_at_or_before(points, dt - timedelta(seconds=30))
    p_60 = _find_price_at_or_before(points, dt - timedelta(seconds=60))

    def ret(base: float | None) -> float:
        if p_now is None or base is None or base <= 0:
            return 0.0
        return (p_now / base) - 1.0

    seconds_in_day = dt.hour * 3600 + dt.minute * 60 + dt.second
    angle = (seconds_in_day / 86400.0) * 2 * math.pi

    prev = [0.5, 0.5, 0.5]
    for i, value in enumerate(prev_outcomes_desc[:3]):
        prev[i] = float(value)

    return {
        "mid_yes": float(current.mid_yes),
        "spread": float(current.spread),
        "imbalance_l1": float(current.imbalance_l1),
        "imbalance_l3": float(current.imbalance_l3),
        "ret_15s": ret(p_15),
        "ret_30s": ret(p_30),
        "ret_60s": ret(p_60),
        "rv_60s": _realized_vol(points, dt, lookback_sec=60),
        "tod_sin": math.sin(angle),
        "tod_cos": math.cos(angle),
        "prev1_up": prev[0],
        "prev2_up": prev[1],
        "prev3_up": prev[2],
    }
