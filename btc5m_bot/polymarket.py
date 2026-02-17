from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any

import requests

from .config import Settings

_EPOCH_SUFFIX_RE = re.compile(r"(\d{9,})$")


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


def _to_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or value == "":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _parse_iso(ts: str) -> datetime:
    return datetime.fromisoformat(ts.replace("Z", "+00:00")).astimezone(timezone.utc)


def _iso_z(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def window_start_from_slug(value: str) -> datetime | None:
    """
    BTC 5m series encodes the underlying 5-minute window start as a Unix epoch suffix:
      btc-updown-5m-1771019100
    Gamma's event.startDate is listing time (often ~24h before), not the 5m window start.
    """
    raw = (value or "").strip()
    if not raw:
        return None
    m = _EPOCH_SUFFIX_RE.search(raw)
    if not m:
        return None
    try:
        epoch = int(m.group(1))
    except ValueError:
        return None
    # Sanity: ignore obviously invalid timestamps.
    if epoch < 1_000_000_000:
        return None
    return datetime.fromtimestamp(epoch, tz=timezone.utc)


def _event_window(event: dict[str, Any]) -> tuple[datetime | None, datetime | None]:
    slug = str(event.get("slug") or "")
    ticker = str(event.get("ticker") or "")
    start = window_start_from_slug(slug) or window_start_from_slug(ticker)

    end: datetime | None = None
    end_raw = event.get("endDate")
    if end_raw:
        try:
            end = _parse_iso(str(end_raw))
        except ValueError:
            end = None

    if start is not None and end is None:
        end = start + timedelta(seconds=300)
    if start is None and end is not None:
        start = end - timedelta(seconds=300)
    return start, end


def _parse_clob_token_ids(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(x) for x in value]
    if isinstance(value, str):
        try:
            decoded = json.loads(value)
            if isinstance(decoded, list):
                return [str(x) for x in decoded]
        except json.JSONDecodeError:
            pass
    return []


def decode_outcome_prices(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(x) for x in value]
    if isinstance(value, str):
        try:
            decoded = json.loads(value)
            if isinstance(decoded, list):
                return [str(x) for x in decoded]
        except json.JSONDecodeError:
            pass
    return []


def label_from_outcome_prices(value: Any) -> int | None:
    prices = decode_outcome_prices(value)
    if len(prices) < 2:
        return None
    p0, p1 = prices[0], prices[1]
    if p0 == "1" and p1 == "0":
        return 1
    if p0 == "0" and p1 == "1":
        return 0
    return None


class PolymarketClient:
    def __init__(self, settings: Settings, timeout: int = 10):
        self.settings = settings
        self.session = requests.Session()
        self.timeout = timeout
        self._active_cache_series_id: int | None = None
        self._active_cache_event: dict[str, Any] | None = None
        self._active_cache_until: datetime | None = None

    def list_open_events(self, series_id: int, limit: int = 300) -> list[dict[str, Any]]:
        resp = self.session.get(
            f"{self.settings.gamma_base_url}/events",
            params={"series_id": str(series_id), "closed": "false", "limit": limit},
            timeout=self.timeout,
        )
        resp.raise_for_status()
        payload = resp.json()
        if not isinstance(payload, list):
            return []
        return payload

    def get_event(self, event_id: str) -> dict[str, Any] | None:
        resp = self.session.get(
            f"{self.settings.gamma_base_url}/events",
            params={"id": event_id},
            timeout=self.timeout,
        )
        resp.raise_for_status()
        payload = resp.json()
        if isinstance(payload, list) and payload:
            return payload[0]
        return None

    def get_active_event(self, series_id: int) -> dict[str, Any] | None:
        now = datetime.now(timezone.utc)
        if (
            self._active_cache_series_id == series_id
            and self._active_cache_event is not None
            and self._active_cache_until is not None
            and now < self._active_cache_until
        ):
            return self._active_cache_event

        events = self.list_open_events(series_id)
        in_window: list[dict[str, Any]] = []

        for e in events:
            start_dt, end_dt = _event_window(e)
            if start_dt is None or end_dt is None:
                continue
            if start_dt <= now < end_dt:
                in_window.append(e)

        if not in_window:
            return None

        in_window.sort(key=lambda e: e.get("endDate", ""))
        chosen = in_window[0]
        _start, end_dt = _event_window(chosen)
        self._active_cache_series_id = series_id
        self._active_cache_event = chosen
        # Re-check active event roughly at boundary; avoid hammering Gamma every second.
        self._active_cache_until = (end_dt + timedelta(seconds=1)) if end_dt else (now + timedelta(seconds=1))
        return chosen

    def _book_summary(
        self, token_id: str
    ) -> tuple[float | None, float | None, float | None, float | None, float | None, float | None]:
        if not token_id:
            return None, None, None, None, None, None
        try:
            resp = self.session.get(
                f"{self.settings.clob_base_url}/book",
                params={"token_id": token_id},
                timeout=self.timeout,
            )
            if resp.status_code != 200:
                return None, None, None, None, None, None
            payload = resp.json()
            bids_raw = payload.get("bids", []) if isinstance(payload, dict) else []
            asks_raw = payload.get("asks", []) if isinstance(payload, dict) else []

            bids: list[tuple[float, float]] = []
            asks: list[tuple[float, float]] = []
            for r in bids_raw:
                price = _to_float(r.get("price"), 0.0)
                size = _to_float(r.get("size"), 0.0)
                if price > 0 and size > 0:
                    bids.append((price, size))
            for r in asks_raw:
                price = _to_float(r.get("price"), 0.0)
                size = _to_float(r.get("size"), 0.0)
                if price > 0 and size > 0:
                    asks.append((price, size))

            if not bids or not asks:
                return None, None, None, None, None, None

            bids.sort(key=lambda x: x[0])  # ascending
            asks.sort(key=lambda x: x[0])  # ascending

            best_bid = bids[-1][0] * 100.0
            best_ask = asks[0][0] * 100.0
            bid_l1 = bids[-1][1]
            ask_l1 = asks[0][1]
            bid_l3 = sum(sz for _p, sz in bids[-3:])
            ask_l3 = sum(sz for _p, sz in asks[:3])
            return best_bid, best_ask, bid_l1, ask_l1, bid_l3, ask_l3
        except requests.RequestException:
            return None, None, None, None, None, None

    @staticmethod
    def _imbalance(bid: float | None, ask: float | None) -> float | None:
        if bid is None or ask is None:
            return None
        denom = bid + ask
        if denom <= 0:
            return 0.0
        return (bid - ask) / denom

    def market_snapshot(self, event: dict[str, Any]) -> MarketSnapshot | None:
        markets = event.get("markets")
        if not isinstance(markets, list) or not markets:
            return None

        window_start, window_end = _event_window(event)
        if window_start is None or window_end is None:
            return None

        market = markets[0]
        token_ids = _parse_clob_token_ids(market.get("clobTokenIds"))
        yes_token_id = token_ids[0] if token_ids else ""
        no_token_id = token_ids[1] if len(token_ids) > 1 else ""

        best_bid, best_ask, bid_l1, ask_l1, bid_l3, ask_l3 = self._book_summary(yes_token_id)
        if best_bid is None or best_ask is None:
            return None

        best_bid_no: float | None = None
        best_ask_no: float | None = None
        bid_l1_no: float | None = None
        ask_l1_no: float | None = None
        bid_l3_no: float | None = None
        ask_l3_no: float | None = None
        no_best_bid, no_best_ask, bid_l1_no, ask_l1_no, bid_l3_no, ask_l3_no = self._book_summary(no_token_id)
        if no_best_bid is not None and no_best_ask is not None:
            best_bid_no, best_ask_no = no_best_bid, no_best_ask
        else:
            # Fallback: in perfectly complementary markets, NO ~= 100 - YES.
            best_bid_no = max(0.0, 100.0 - best_ask)
            best_ask_no = max(0.0, 100.0 - best_bid)
            bid_l1_no = None
            ask_l1_no = None
            bid_l3_no = None
            ask_l3_no = None

        spread = max(0.0, best_ask - best_bid)
        mid_yes = max(0.0, min(1.0, ((best_bid + best_ask) / 2.0) / 100.0))
        last_trade = (best_bid + best_ask) / 2.0

        return MarketSnapshot(
            event_id=str(event.get("id", "")),
            market_id=str(market.get("id", "")),
            slug=str(event.get("slug", event.get("ticker", ""))),
            ticker=str(event.get("ticker", event.get("slug", ""))),
            start_ts=_iso_z(window_start),
            end_ts=_iso_z(window_end),
            best_bid=best_bid,
            best_ask=best_ask,
            best_bid_no=best_bid_no,
            best_ask_no=best_ask_no,
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
        event = self.get_event(event_id)
        if not event:
            return None, None
        markets = event.get("markets")
        if not isinstance(markets, list) or not markets:
            return None, event
        label = label_from_outcome_prices(markets[0].get("outcomePrices"))
        return label, event
