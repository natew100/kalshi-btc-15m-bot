#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from btc5m_bot.backtest import (
    ReplayScenario,
    load_replay_rows,
    replay_scenario,
    scenario_grid,
    stress_scenarios,
)
from btc5m_bot.config import load_settings
from btc5m_bot.db import connect_db


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Replay backtest for BTC 5m bot decisions")
    p.add_argument("--db-path", type=str, default="", help="SQLite path (defaults to settings db)")
    p.add_argument("--lookback-days", type=int, default=14)
    p.add_argument(
        "--out",
        type=str,
        default="",
        help="Output JSON path (default: <bot_root>/data/reports/backtest_latest.json)",
    )
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    settings = load_settings()
    default_data_dir = settings.data_dir
    if not default_data_dir.exists():
        default_data_dir = REPO_ROOT / "data"
    db_path = Path(args.db_path) if args.db_path else (settings.db_path if settings.db_path.exists() else (default_data_dir / "bot.db"))
    out_path = Path(args.out) if args.out else (default_data_dir / "reports" / "backtest_latest.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    conn = connect_db(db_path)
    rows = load_replay_rows(conn, lookback_days=args.lookback_days)
    conn.close()

    base = ReplayScenario(
        name="base",
        base_cost_cents=settings.default_cost_cents,
        min_edge_cents=settings.min_edge_cents,
        max_spread_cents=settings.max_spread_cents,
        max_overround_cents=settings.max_overround_cents,
        high_vol_rv60_threshold=settings.high_vol_rv60_threshold,
        high_vol_edge_add_cents=settings.high_vol_min_edge_add_cents,
        high_vol_max_spread_cents=settings.high_vol_max_spread_cents,
        side_calib_min_samples=settings.side_calib_min_samples,
        side_calib_min_win_rate=settings.side_calib_min_win_rate,
        side_calib_edge_penalty_cents=settings.side_calib_edge_penalty_cents,
    )

    base_metrics = replay_scenario(rows, base)
    sweep = [replay_scenario(rows, s) for s in scenario_grid(base)]
    sweep.sort(key=lambda m: (m.expectancy_cents, m.net_after_cost_cents), reverse=True)
    stress = [replay_scenario(rows, s) for s in stress_scenarios(base)]

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "db_path": str(db_path),
        "rows": len(rows),
        "lookback_days": args.lookback_days,
        "base_config": asdict(base),
        "base_metrics": asdict(base_metrics),
        "top_scenarios": [asdict(m) for m in sweep[:10]],
        "stress_costs": [asdict(m) for m in stress],
    }

    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {out_path} rows={len(rows)} top={payload['top_scenarios'][0]['scenario'] if payload['top_scenarios'] else 'none'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
