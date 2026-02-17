from types import SimpleNamespace

from btc5m_bot.kalshi_markets import KalshiBTCClient


def _settings() -> SimpleNamespace:
    return SimpleNamespace(
        kalshi_base_url="https://api.elections.kalshi.com/trade-api/v2",
        kalshi_series_ticker="KXBTC15M",
    )


def test_market_snapshot_parses_basic_fields() -> None:
    client = KalshiBTCClient(settings=_settings())
    market = {
        "ticker": "KXBTC15M-26FEB162100-00",
        "open_time": "2026-02-17T01:45:00Z",
        "close_time": "2026-02-17T02:00:00Z",
        "yes_bid": 22,
        "yes_ask": 24,
        "no_bid": 76,
        "no_ask": 78,
        "last_price": 25,
    }

    def _fake_orderbook(_ticker: str):
        yes = [(11.0, 100.0), (12.0, 50.0), (13.0, 10.0)]
        no = [(83.0, 90.0), (84.0, 40.0), (85.0, 20.0)]
        return yes, no

    client._orderbook = _fake_orderbook  # type: ignore[method-assign]
    snap = client.market_snapshot(market)
    assert snap is not None
    assert snap.event_id == "KXBTC15M-26FEB162100-00"
    assert snap.best_bid == 22.0
    assert snap.best_ask == 24.0
    assert snap.best_bid_no == 76.0
    assert snap.best_ask_no == 78.0
    assert snap.ask_sz_l1 is not None
    assert snap.ask_sz_l3 is not None


def test_resolution_label_maps_yes_no() -> None:
    client = KalshiBTCClient(settings=_settings())
    client.get_event = lambda _eid: {"result": "yes"}  # type: ignore[method-assign]
    label, event = client.event_resolution_label("X")
    assert label == 1
    assert event is not None

    client.get_event = lambda _eid: {"result": "no"}  # type: ignore[method-assign]
    label, event = client.event_resolution_label("Y")
    assert label == 0
    assert event is not None
