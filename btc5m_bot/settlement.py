from __future__ import annotations


def settle_trade_pnl(label_up: int, side: str, price_cents: int, quantity: int) -> tuple[str, int]:
    won = (label_up == 1 and side == "yes") or (label_up == 0 and side == "no")
    if won:
        pnl = (100 - int(price_cents)) * int(quantity)
        return "settled_win", pnl
    pnl = -int(price_cents) * int(quantity)
    return "settled_loss", pnl
