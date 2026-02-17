from __future__ import annotations

import argparse

from btc5m_bot.backfill_ledger import backfill_all
from btc5m_bot.config import ensure_runtime_paths, load_settings


def main() -> int:
    parser = argparse.ArgumentParser(description="Backfill fees/balances for historical trades")
    parser.add_argument("--db", default=None, help="Path to bot.db (defaults to settings.db_path)")
    args = parser.parse_args()

    settings = load_settings()
    ensure_runtime_paths(settings)
    db_path = args.db or str(settings.db_path)

    res = backfill_all(
        db_path,
        starting_cash_cents=int(settings.paper_bankroll_cents),
        default_cost_cents=float(settings.default_cost_cents),
    )
    print(res)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

