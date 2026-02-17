from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any

import requests

from .config import Settings


@dataclass
class MarketSnapshot:
    event_id: str
    market_id: str
    slug: str
    ticker: str
    start_ts: str
    end_ts: str
    best_bid: float
    best_ask: float
    best_bid_no: float | None
    best_ask_no: float | None
    last_trade_price: float
    spread: float
    mid_yes: float
    bid_sz_l1: float | None
    ask_sz_l1: float | None
    bid_sz_l3: float | None
    ask_sz_l3: float | None
    bid_sz_l1_no: float | None
    ask_sz_l1_no: float | None
    bid_sz_l3_no: float | None
    ask_sz_l3_no: float | None
    imbalance_l1: float | None
    imbalance_l3: float | None


def _parse_iso(ts: str) -> datetime:
    return datetime.fromisoformat(ts.replace("Z", "+00:00")).astimezone(timezone.utc)


def _iso_z(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _to_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or value == "":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


class KalshiBTCClient:
    def __init__(self, settings: Settings, timeout: int = 10):
        self.settings = settings
        self.timeout = timeout
        self.session = requests.Session()
        self._active_cache_event: dict[str, Any] | None = None
        self._active_cache_until: datetime | None = None
        self._no_active_until: datetime | None = None

    @staticmethod
    def _imbalance(bid: float | None, ask: float | None) -> float | None:
        if bid is None or ask is None:
            return None
        denom = bid + ask
        if denom <= 0:
            return 0.0
        return (bid - ask) / denom

    def _list_markets(self, *, status: str, limit: int = 200) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        cursor: str | None = None
        while True:
            params: dict[str, Any] = {
                "series_ticker": self.settings.kalshi_series_ticker,
                "status": status,
                "limit": limit,
            }
            if cursor:
                params["cursor"] = cursor
            resp = self.session.get(
                f"{self.settings.kalshi_base_url}/markets",
                params=params,
                timeout=self.timeout,
            )
            resp.raise_for_status()
            payload = resp.json()
            markets = payload.get("markets", []) if isinstance(payload, dict) else []
            if isinstance(markets, list):
                out.extend([m for m in markets if isinstance(m, dict)])
            cursor = payload.get("cursor") if isinstance(payload, dict) else None
            if not cursor:
                break
        return out

    def get_event(self, event_id: str) -> dict[str, Any] | None:
        resp = self.session.get(
            f"{self.settings.kalshi_base_url}/markets/{event_id}",
            timeout=self.timeout,
        )
        if resp.status_code != 200:
            return None
        payload = resp.json()
        market = payload.get("market") if isinstance(payload, dict) else None
        return market if isinstance(market, dict) else None

    def get_active_event(self, _unused_series_id: int) -> dict[str, Any] | None:
        now = datetime.now(timezone.utc)
        if self._no_active_until is not None and now < self._no_active_until:
            return None
        if self._active_cache_event is not None and self._active_cache_until and now < self._active_cache_until:
            return self._active_cache_event

        markets = self._list_markets(status="open", limit=200)
        in_window: list[dict[str, Any]] = []
        for m in markets:
            try:
                open_ts = _parse_iso(str(m.get("open_time")))
                close_ts = _parse_iso(str(m.get("close_time")))
            except Exception:
                continue
            if open_ts <= now < close_ts:
                in_window.append(m)
        if not in_window:
            self._no_active_until = now + timedelta(seconds=5)
            return None
        in_window.sort(key=lambda x: str(x.get("close_time", "")))
        chosen = in_window[0]
        close_ts = _parse_iso(str(chosen.get("close_time")))
        self._active_cache_event = chosen
        self._active_cache_until = close_ts + timedelta(seconds=1)
        self._no_active_until = None
        return chosen

    def _orderbook(self, ticker: str) -> tuple[list[tuple[float, float]], list[tuple[float, float]]]:
        try:
            resp = self.session.get(
                f"{self.settings.kalshi_base_url}/markets/{ticker}/orderbook",
                params={"depth": 10},
                timeout=self.timeout,
            )
            if resp.status_code != 200:
                return [], []
            payload = resp.json()
            ob = payload.get("orderbook", {}) if isinstance(payload, dict) else {}
            yes_raw = ob.get("yes", []) if isinstance(ob, dict) else []
            no_raw = ob.get("no", []) if isinstance(ob, dict) else []
            if yes_raw is None:
                yes_raw = []
            if no_raw is None:
                no_raw = []
            yes = [(float(l[0]), float(l[1])) for l in yes_raw if isinstance(l, list) and len(l) >= 2]
            no = [(float(l[0]), float(l[1])) for l in no_raw if isinstance(l, list) and len(l) >= 2]
            return yes, no
        except requests.RequestException:
            return [], []

    @staticmethod
    def _split_book(
        levels: list[tuple[float, float]], bid_price: float, ask_price: float
    ) -> tuple[list[tuple[float, float]], list[tuple[float, float]]]:
        """Split orderbook levels into bid-side and ask-side.

        Prices at or below ``bid_price`` are resting buy orders (bids).
        Prices at or above ``ask_price`` are resting sell orders (asks).
        """
        bids = [(px, sz) for px, sz in levels if px <= bid_price]
        asks = [(px, sz) for px, sz in levels if px >= ask_price]
        bids.sort(key=lambda x: x[0], reverse=True)   # best bid first
        asks.sort(key=lambda x: x[0])                  # best ask first
        return bids, asks

    @staticmethod
    def _implied_ask_depth(
        opposite_levels: list[tuple[float, float]], ask_price: float
    ) -> tuple[float | None, float | None]:
        """Derive ask depth from opposite-side buy orders.

        A NO buy at price P implies a YES sell at (100-P).
        To fill a YES buy at ``ask_price``, NO buys must be >= (100-ask_price).
        Highest opposite bids give the tightest implied asks (best ask first).
        """
        if not opposite_levels:
            return None, None
        threshold = max(0.0, 100.0 - float(ask_price))
        levels = [(px, sz) for px, sz in opposite_levels if px >= threshold]
        if not levels:
            return None, None
        levels.sort(key=lambda x: x[0], reverse=True)
        l1 = float(levels[0][1])
        l3 = float(sum(sz for _px, sz in levels[:3]))
        return l1, l3

    @staticmethod
    def _depth(levels: list[tuple[float, float]]) -> tuple[float | None, float | None]:
        """Return L1 size and L3 cumulative size from sorted levels."""
        if not levels:
            return None, None
        l1 = float(levels[0][1])
        l3 = float(sum(sz for _px, sz in levels[:3]))
        return l1, l3

    def market_snapshot(self, market: dict[str, Any]) -> MarketSnapshot | None:
        ticker = str(market.get("ticker", ""))
        if not ticker:
            return None

        try:
            start_ts = _parse_iso(str(market.get("open_time")))
            end_ts = _parse_iso(str(market.get("close_time")))
        except Exception:
            return None

        yes_bid = _to_float(market.get("yes_bid"), 0.0)
        yes_ask = _to_float(market.get("yes_ask"), 0.0)
        no_bid = _to_float(market.get("no_bid"), max(0.0, 100.0 - yes_ask))
        no_ask = _to_float(market.get("no_ask"), max(0.0, 100.0 - yes_bid))

        if yes_bid <= 0 or yes_ask <= 0:
            return None

        yes_levels, no_levels = self._orderbook(ticker)
        yes_bids, yes_asks = self._split_book(yes_levels, yes_bid, yes_ask)
        no_bids, no_asks = self._split_book(no_levels, no_bid, no_ask)

        bid_l1, bid_l3 = self._depth(yes_bids)
        ask_l1, ask_l3 = self._depth(yes_asks)
        if ask_l1 is None:
            ask_l1, ask_l3 = self._implied_ask_depth(no_levels, yes_ask)

        bid_l1_no, bid_l3_no = self._depth(no_bids)
        ask_l1_no, ask_l3_no = self._depth(no_asks)
        if ask_l1_no is None:
            ask_l1_no, ask_l3_no = self._implied_ask_depth(yes_levels, no_ask)

        spread = max(0.0, yes_ask - yes_bid)
        mid_yes = max(0.0, min(1.0, ((yes_bid + yes_ask) / 2.0) / 100.0))
        last_trade = _to_float(market.get("last_price"), (yes_bid + yes_ask) / 2.0)

        return MarketSnapshot(
            event_id=ticker,
            market_id=ticker,
            slug=ticker.lower(),
            ticker=ticker,
            start_ts=_iso_z(start_ts),
            end_ts=_iso_z(end_ts),
            best_bid=yes_bid,
            best_ask=yes_ask,
            best_bid_no=no_bid,
            best_ask_no=no_ask,
            last_trade_price=last_trade,
            spread=spread,
            mid_yes=mid_yes,
            bid_sz_l1=bid_l1,
            ask_sz_l1=ask_l1,
            bid_sz_l3=bid_l3,
            ask_sz_l3=ask_l3,
            bid_sz_l1_no=bid_l1_no,
            ask_sz_l1_no=ask_l1_no,
            bid_sz_l3_no=bid_l3_no,
            ask_sz_l3_no=ask_l3_no,
            imbalance_l1=self._imbalance(bid_l1, ask_l1),
            imbalance_l3=self._imbalance(bid_l3, ask_l3),
        )

    def event_resolution_label(self, event_id: str) -> tuple[int | None, dict[str, Any] | None]:
        market = self.get_event(event_id)
        if not market:
            return None, None
        result = str(market.get("result", "")).strip().lower()
        if result == "yes":
            return 1, market
        if result == "no":
            return 0, market
        return None, market
