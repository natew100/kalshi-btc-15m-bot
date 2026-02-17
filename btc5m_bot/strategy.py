from __future__ import annotations

from dataclasses import dataclass
import math
from datetime import datetime, timezone


@dataclass
class EdgeDecision:
    side: str
    ev_yes: float
    ev_no: float
    chosen_ev: float
    ask_yes: float
    ask_no: float


@dataclass
class FillResult:
    filled: bool
    price_cents: float
    tier: str
    reason: str


@dataclass
class ExecutionCost:
    fee_cents: int
    slippage_buffer_cents: int
    all_in_cost_cents: int
    per_contract_cost_cents: float


def simulate_buy_fill(
    best_ask_cents: float,
    quantity: int,
    ask_sz_l1: float | None,
    ask_sz_l3: float | None,
    l3_slippage_cents: float,
) -> FillResult:
    """
    Coarse fill simulator for paper/shadow.

    - qty <= L1 ask size: fill at best ask
    - qty <= L3 cumulative ask size: fill at best ask + slippage
    - else: no fill
    """
    qty = int(quantity)
    if best_ask_cents <= 0 or qty <= 0:
        return FillResult(False, float(best_ask_cents), "none", "invalid_quote_or_qty")

    if ask_sz_l1 is None or ask_sz_l3 is None:
        # Conservative: if we can't observe depth for this side, treat as unfillable.
        return FillResult(False, float(best_ask_cents), "none", "missing_depth")

    # Some books return fractional sizes; still treat qty as shares/contracts.
    if float(ask_sz_l1) >= qty:
        return FillResult(True, float(best_ask_cents), "l1", "filled_l1")

    if float(ask_sz_l3) >= qty:
        px = float(best_ask_cents) + float(l3_slippage_cents)
        # Guard against impossible prices.
        px = min(px, 99.0)
        return FillResult(True, px, "l3", "filled_l3")

    return FillResult(False, float(best_ask_cents), "none", "insufficient_depth")


def size_quantity(
    *,
    sizing_mode: str,
    fixed_contracts: int,
    max_contracts: int,
    risk_per_trade_pct: float,
    equity_cents: int,
    price_cents: float,
) -> int:
    """
    Paper-sizing only.

    - fixed: use FIXED_CONTRACTS
    - risk_pct: target (risk_per_trade_pct% of equity) as max loss; qty ~= budget / entry_price
    """
    mode = (sizing_mode or "fixed").strip().lower()
    max_q = max(1, int(max_contracts))

    if mode == "fixed":
        return max(1, min(max_q, int(fixed_contracts)))

    if mode == "risk_pct":
        px = float(price_cents)
        if px <= 0:
            return max(1, min(max_q, int(fixed_contracts)))
        budget = int(round(float(equity_cents) * (float(risk_per_trade_pct) / 100.0)))
        budget = max(1, budget)
        q = int(budget // int(round(px)))
        return max(1, min(max_q, q))

    # Unknown mode: safe fallback.
    return max(1, min(max_q, int(fixed_contracts)))



def compute_edge(
    model_prob_up: float,
    ask_yes_cents: float,
    ask_no_cents: float,
    cost_cents: float,
) -> EdgeDecision:
    ev_yes = model_prob_up * 100.0 - ask_yes_cents - cost_cents
    ev_no = (1.0 - model_prob_up) * 100.0 - ask_no_cents - cost_cents
    if ev_yes >= ev_no:
        side = "yes"
        chosen_ev = ev_yes
    else:
        side = "no"
        chosen_ev = ev_no

    return EdgeDecision(
        side=side,
        ev_yes=ev_yes,
        ev_no=ev_no,
        chosen_ev=chosen_ev,
        ask_yes=ask_yes_cents,
        ask_no=ask_no_cents,
    )


def within_decision_window(
    now_utc: datetime,
    market_open_utc: datetime,
    start_seconds: int,
    end_seconds: int,
) -> bool:
    age = (now_utc - market_open_utc).total_seconds()
    return start_seconds <= age <= end_seconds


def should_trade(
    chosen_ev_cents: float,
    spread_cents: float,
    in_window: bool,
    min_edge_cents: float,
    max_spread_cents: float,
) -> tuple[bool, str]:
    if not in_window:
        return False, "outside_decision_window"
    if chosen_ev_cents < min_edge_cents:
        return False, "edge_below_threshold"
    if spread_cents > max_spread_cents:
        return False, "spread_too_wide"
    return True, "ok"


def confidence_from_prob(model_prob_up: float) -> float:
    return min(1.0, max(0.0, abs(model_prob_up - 0.5) * 2.0))


def kelly_from_prob(side_prob: float, price_cents: float) -> float:
    # Conservative capped approximation; live sizing is fixed 1 contract.
    p = min(max(side_prob, 1e-4), 1 - 1e-4)
    b = (100.0 - price_cents) / max(price_cents, 1e-4)
    frac = (b * p - (1.0 - p)) / b if b > 0 else 0.0
    return max(0.0, min(0.03, frac))


def build_trade_payload(
    event_id: str,
    ticker: str,
    side: str,
    mode: str,
    quantity: int,
    model_prob_up: float,
    ask_yes_cents: float,
    ask_no_cents: float,
    chosen_ev_cents: float,
    reason: str,
) -> dict:
    now = datetime.now(timezone.utc).isoformat()
    price_cents = ask_yes_cents if side == "yes" else ask_no_cents
    side_prob = model_prob_up if side == "yes" else (1.0 - model_prob_up)
    market_prob = price_cents / 100.0
    edge_prob = side_prob - market_prob

    return {
        # Deterministic id per (event_id, portfolio mode). Makes inserts idempotent.
        "external_id": f"{event_id}:{mode}",
        "event_id": event_id,
        "ticker": ticker,
        "side": side,
        "action": "buy",
        "price_cents": int(round(price_cents)),
        "quantity": int(quantity),
        "model_prob": float(model_prob_up),
        "market_price": float(market_prob),
        "edge": float(edge_prob),
        "kelly_fraction": float(kelly_from_prob(side_prob, price_cents)),
        "confidence": float(confidence_from_prob(model_prob_up)),
        "status": "dry_run" if mode in ("paper", "shadow") else "pending",
        "pnl_cents": None,
        "reasoning": reason,
        "mode": mode,
        "created_at": now,
    }


def estimate_execution_cost(
    *,
    price_cents: float,
    quantity: int,
    fill_tier: str,
    fee_fixed_cents_per_contract: float,
    fee_bps_per_notional: float,
    l1_slippage_buffer_cents: float,
    l3_slippage_buffer_cents: float,
) -> ExecutionCost:
    qty = max(1, int(quantity))
    px = max(0.0, float(price_cents))
    notional = px * float(qty)
    fee = (
        float(fee_fixed_cents_per_contract) * float(qty)
        + (float(fee_bps_per_notional) / 10000.0) * notional
    )

    if fill_tier == "l3":
        slip = float(l3_slippage_buffer_cents) * float(qty)
    elif fill_tier == "l1":
        slip = float(l1_slippage_buffer_cents) * float(qty)
    else:
        slip = 0.0

    fee_i = int(math.ceil(max(0.0, fee)))
    slip_i = int(math.ceil(max(0.0, slip)))
    total = fee_i + slip_i
    return ExecutionCost(
        fee_cents=fee_i,
        slippage_buffer_cents=slip_i,
        all_in_cost_cents=total,
        per_contract_cost_cents=float(total) / float(qty),
    )
