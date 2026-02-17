from btc5m_bot.polymarket import decode_outcome_prices, label_from_outcome_prices


def test_decode_outcome_prices_from_json_string() -> None:
    raw = '["1", "0"]'
    assert decode_outcome_prices(raw) == ["1", "0"]


def test_label_up_from_outcomes() -> None:
    assert label_from_outcome_prices('["1", "0"]') == 1


def test_label_down_from_outcomes() -> None:
    assert label_from_outcome_prices('["0", "1"]') == 0


def test_label_unresolved_returns_none() -> None:
    assert label_from_outcome_prices('["0.52", "0.48"]') is None
