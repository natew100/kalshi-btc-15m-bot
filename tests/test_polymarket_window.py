from datetime import datetime, timezone

from btc15m_bot.polymarket import window_start_from_slug


def test_window_start_from_slug_epoch_suffix() -> None:
    # 1771019100 == 2026-02-13T21:45:00Z
    dt = window_start_from_slug("btc-updown-5m-1771019100")
    assert dt == datetime(2026, 2, 13, 21, 45, 0, tzinfo=timezone.utc)


def test_window_start_returns_none_for_invalid() -> None:
    assert window_start_from_slug("") is None
    assert window_start_from_slug("btc-updown-5m-") is None
    assert window_start_from_slug("btc-updown-5m-abc") is None

