from datetime import datetime, timedelta, timezone

from btc5m_bot.strategy import (
    compute_edge,
    estimate_execution_cost,
    should_trade,
    simulate_buy_fill,
    within_decision_window,
)


def test_ev_side_selection() -> None:
    out = compute_edge(model_prob_up=0.57, ask_yes_cents=54, ask_no_cents=47, cost_cents=1.5)
    assert out.side == "yes"
    assert out.ev_yes > out.ev_no


def test_trade_gate_blocks_wide_spread() -> None:
    allowed, reason = should_trade(
        chosen_ev_cents=2.0,
        spread_cents=3.0,
        in_window=True,
        min_edge_cents=1.8,
        max_spread_cents=2.0,
    )
    assert allowed is False
    assert reason == "spread_too_wide"


def test_decision_window_logic() -> None:
    open_ts = datetime(2026, 2, 13, 20, 0, 0, tzinfo=timezone.utc)
    now = open_ts + timedelta(seconds=75)
    assert within_decision_window(now, open_ts, start_seconds=45, end_seconds=120)


def test_fill_sim_l1() -> None:
    out = simulate_buy_fill(best_ask_cents=55.0, quantity=1, ask_sz_l1=2.0, ask_sz_l3=5.0, l3_slippage_cents=1.0)
    assert out.filled is True
    assert out.tier == "l1"
    assert out.price_cents == 55.0


def test_fill_sim_l3() -> None:
    out = simulate_buy_fill(best_ask_cents=55.0, quantity=1, ask_sz_l1=0.2, ask_sz_l3=2.0, l3_slippage_cents=1.0)
    assert out.filled is True
    assert out.tier == "l3"
    assert out.price_cents == 56.0


def test_fill_sim_no_fill() -> None:
    out = simulate_buy_fill(best_ask_cents=55.0, quantity=1, ask_sz_l1=0.2, ask_sz_l3=0.8, l3_slippage_cents=1.0)
    assert out.filled is False
    assert out.tier == "none"


def test_estimate_execution_cost_l3() -> None:
    out = estimate_execution_cost(
        price_cents=54.0,
        quantity=10,
        fill_tier="l3",
        fee_fixed_cents_per_contract=0.2,
        fee_bps_per_notional=10.0,
        l1_slippage_buffer_cents=0.05,
        l3_slippage_buffer_cents=0.3,
    )
    # fee ~= 2 + 0.54 => ceil(2.54)=3; slippage ceil(3.0)=3
    assert out.fee_cents == 3
    assert out.slippage_buffer_cents == 3
    assert out.all_in_cost_cents == 6
