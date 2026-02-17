from dataclasses import dataclass

import numpy as np

from btc15m_bot.modeling import ModelBundle, top_feature_contributions


@dataclass
class _FakeLogistic:
    coef_: np.ndarray


def test_top_feature_contributions_orders_by_magnitude() -> None:
    bundle = ModelBundle(
        logistic=_FakeLogistic(coef_=np.array([[2.0, -1.0, 0.5]], dtype=float)),
        calibrator=None,
        trained_at="",
        feature_columns=["a", "b", "c"],
    )
    row = {"a": 0.3, "b": 2.0, "c": 4.0}
    out = top_feature_contributions(bundle, row, top_k=2)
    # Contributions: a=+0.6, b=-2.0, c=+2.0 => top two are b and c by abs magnitude.
    assert len(out) == 2
    assert out[0][0] in {"b", "c"}
    assert out[1][0] in {"b", "c"}
    assert out[0][0] != out[1][0]
