from datetime import datetime, timedelta, timezone

from btc15m_bot.features import build_feature_row


def test_feature_generation_at_open_plus_75_seconds() -> None:
    start = datetime(2026, 2, 13, 20, 0, 0, tzinfo=timezone.utc)
    ticks = []
    for i in range(0, 90):
        ts = start + timedelta(seconds=i)
        ticks.append(
            {
                "ts": ts.isoformat(),
                "mid_yes": 0.50 + i * 0.0005,
                "spread": 1.5,
                "imbalance_l1": 0.1,
                "imbalance_l3": 0.05,
                "spot_price": 100000 + i,
            }
        )

    decision_ts = (start + timedelta(seconds=75)).isoformat()
    row = build_feature_row(ticks=ticks, decision_ts=decision_ts, prev_outcomes_desc=[1, 0, 1])

    assert row["mid_yes"] > 0.5
    assert row["ret_15s"] > 0
    assert row["ret_30s"] > 0
    assert row["ret_60s"] > 0
    assert row["rv_60s"] >= 0
    assert row["prev1_up"] == 1.0
    assert row["prev2_up"] == 0.0
    assert row["prev3_up"] == 1.0
