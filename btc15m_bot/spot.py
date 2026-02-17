from __future__ import annotations

import statistics

import requests

from .config import Settings


class SpotPriceClient:
    def __init__(self, settings: Settings, timeout: int = 4):
        self.settings = settings
        self.timeout = timeout
        self.session = requests.Session()

    def _binance(self) -> float | None:
        try:
            resp = self.session.get(self.settings.binance_ticker_url, timeout=self.timeout)
            if resp.status_code != 200:
                return None
            payload = resp.json()
            return float(payload["price"])
        except Exception:
            return None

    def _coinbase(self) -> float | None:
        try:
            resp = self.session.get(self.settings.coinbase_ticker_url, timeout=self.timeout)
            if resp.status_code != 200:
                return None
            payload = resp.json()
            # Coinbase endpoint can return either {price} or nested under data.
            if isinstance(payload, dict) and "price" in payload:
                return float(payload["price"])
            data = payload.get("data", {}) if isinstance(payload, dict) else {}
            if isinstance(data, dict) and "amount" in data:
                return float(data["amount"])
            return None
        except Exception:
            return None

    def blended_btc_price(self) -> float | None:
        values = [x for x in (self._binance(), self._coinbase()) if x is not None]
        if not values:
            return None
        if len(values) == 1:
            return values[0]
        return statistics.median(values)
