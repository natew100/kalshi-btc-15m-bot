from btc5m_bot.backtest import (
    ReplayRow,
    ReplayScenario,
    parse_reason_kv,
    replay_scenario,
    scenario_grid,
)


def _base_scenario() -> ReplayScenario:
    return ReplayScenario(
        name="base",
        base_cost_cents=1.5,
        min_edge_cents=1.8,
        max_spread_cents=2.0,
        max_overround_cents=3.0,
        high_vol_rv60_threshold=0.0018,
        high_vol_edge_add_cents=0.6,
        high_vol_max_spread_cents=1.5,
        side_calib_min_samples=40,
        side_calib_min_win_rate=0.53,
        side_calib_edge_penalty_cents=0.5,
    )


def test_parse_reason_kv_extracts_values() -> None:
    parsed = parse_reason_kv("ok|rv_60s=0.000123|min_edge=2.300|max_spread=2.000")
    assert parsed["rv_60s"] == "0.000123"
    assert parsed["min_edge"] == "2.300"


def test_replay_scenario_filters_by_edge_and_spread() -> None:
    rows = [
        ReplayRow(
            event_id="e1",
            decided_at="2026-02-16T00:00:00+00:00",
            model_prob=0.65,
            ask_yes=55.0,
            ask_no=46.0,
            spread=1.0,
            label_up=1,
            reason="rv_60s=0.000050",
        ),
        ReplayRow(
            event_id="e2",
            decided_at="2026-02-16T00:05:00+00:00",
            model_prob=0.52,
            ask_yes=60.0,
            ask_no=41.0,
            spread=3.0,
            label_up=1,
            reason="rv_60s=0.000050",
        ),
    ]
    m = replay_scenario(rows, _base_scenario())
    assert m.rows == 2
    assert m.traded == 1
    assert m.wins + m.losses == 1


def test_scenario_grid_generates_multiple_configs() -> None:
    grid = scenario_grid(_base_scenario())
    assert len(grid) >= 6
    assert any(s.max_spread_cents < 2.0 for s in grid)
