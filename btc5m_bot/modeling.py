from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np

from . import db
from .config import Settings
from .features import FEATURE_COLUMNS


@dataclass
class ModelBundle:
    logistic: Any
    calibrator: Any
    trained_at: str
    feature_columns: list[str]


def load_model(path: Path) -> ModelBundle | None:
    if not path.exists():
        return None
    payload = joblib.load(path)
    return ModelBundle(
        logistic=payload["logistic"],
        calibrator=payload["calibrator"],
        trained_at=payload.get("trained_at", ""),
        feature_columns=list(payload.get("feature_columns", FEATURE_COLUMNS)),
    )


def predict_prob(bundle: ModelBundle | None, feature_row: dict[str, float]) -> float:
    if bundle is None:
        return 0.5

    x = np.array([[feature_row[col] for col in bundle.feature_columns]], dtype=float)
    raw = float(bundle.logistic.predict_proba(x)[:, 1][0])
    try:
        calibrated = float(bundle.calibrator.predict([raw])[0])
    except Exception:
        calibrated = raw
    return max(0.01, min(0.99, calibrated))


def top_feature_contributions(
    bundle: ModelBundle | None,
    feature_row: dict[str, float],
    top_k: int = 3,
) -> list[tuple[str, float]]:
    """
    Lightweight attribution for linear-logistic model:
    contribution ~= coef_i * x_i.
    """
    if bundle is None:
        return []
    try:
        coefs = np.array(bundle.logistic.coef_[0], dtype=float)
    except Exception:
        return []

    cols = list(bundle.feature_columns)
    if len(cols) != len(coefs):
        return []

    out: list[tuple[str, float]] = []
    for i, col in enumerate(cols):
        x = float(feature_row.get(col, 0.0))
        out.append((col, float(coefs[i]) * x))
    out.sort(key=lambda it: abs(it[1]), reverse=True)
    k = max(1, int(top_k))
    return out[:k]


def _needs_retrain(latest_run: Any, retrain_utc_hour: int) -> bool:
    if latest_run is None:
        return True
    status = str(latest_run["status"]) if "status" in latest_run.keys() else ""
    finished_at = latest_run["finished_at"] if "finished_at" in latest_run.keys() else None
    if not finished_at:
        return True

    finished = datetime.fromisoformat(str(finished_at).replace("Z", "+00:00")).astimezone(
        timezone.utc
    )
    now = datetime.now(timezone.utc)

    # If the last attempt failed or was skipped (commonly due to insufficient rows),
    # retry after a cooldown so the bot can begin training as soon as enough data exists.
    if status in {"skipped", "error"}:
        if (now - finished).total_seconds() >= 1800:
            return True

    if finished.date() < now.date() and now.hour >= retrain_utc_hour:
        return True
    return False


def _parse_iso(ts: str) -> datetime:
    return datetime.fromisoformat(ts.replace("Z", "+00:00")).astimezone(timezone.utc)


def _filter_valid_cycles(rows: list[Any], min_seconds: float = 240.0, max_seconds: float = 1200.0) -> list[Any]:
    """
    Guardrail: filter out obviously invalid cycle windows.
    This supports 5m/15m style cycles while still excluding bad multi-hour spans.
    """
    out: list[Any] = []
    for r in rows:
        try:
            start = _parse_iso(str(r["start_ts"]))
            end = _parse_iso(str(r["end_ts"]))
        except Exception:
            continue
        dur = (end - start).total_seconds()
        if min_seconds <= dur <= max_seconds:
            out.append(r)
    return out


def maybe_train_model(conn, settings: Settings) -> tuple[bool, str]:
    latest = db.latest_model_run(conn)
    if not settings.auto_retrain:
        return False, "auto retrain disabled"
    if not _needs_retrain(latest, settings.retrain_utc_hour):
        return False, "already trained in current window"
    return train_model(conn, settings)


def train_model(conn, settings: Settings) -> tuple[bool, str]:
    run_id = db.insert_model_run_start(conn)
    try:
        rows = db.fetch_labeled_features(conn, lookback_days=settings.lookback_days)
        rows = _filter_valid_cycles(list(rows))
        if len(rows) < settings.min_train_rows:
            msg = f"insufficient rows: {len(rows)} < {settings.min_train_rows}"
            db.complete_model_run(conn, run_id, status="skipped", n_rows=len(rows), error=msg)
            return False, msg

        # Lazy import so runtime can still operate without sklearn until training is needed.
        from sklearn.isotonic import IsotonicRegression
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

        x = np.array([[float(r[c]) for c in FEATURE_COLUMNS] for r in rows], dtype=float)
        y = np.array([int(r["label_up"]) for r in rows], dtype=int)

        split_idx = max(int(len(rows) * 0.8), settings.min_train_rows // 2)
        split_idx = min(split_idx, len(rows) - 1)
        x_train, y_train = x[:split_idx], y[:split_idx]
        x_val, y_val = x[split_idx:], y[split_idx:]

        logistic = LogisticRegression(max_iter=2000, penalty="l2", solver="lbfgs")
        logistic.fit(x_train, y_train)

        p_val = logistic.predict_proba(x_val)[:, 1]
        calibrator = IsotonicRegression(out_of_bounds="clip")
        calibrator.fit(p_val, y_val)
        p_cal = calibrator.predict(p_val)

        brier = float(brier_score_loss(y_val, p_cal))
        ll = float(log_loss(y_val, np.clip(p_cal, 1e-6, 1 - 1e-6)))
        try:
            auc = float(roc_auc_score(y_val, p_cal))
        except ValueError:
            auc = 0.5

        settings.model_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "logistic": logistic,
                "calibrator": calibrator,
                "trained_at": datetime.now(timezone.utc).isoformat(),
                "feature_columns": FEATURE_COLUMNS,
                "metrics": {
                    "brier": brier,
                    "logloss": ll,
                    "auc": auc,
                    "n_rows": len(rows),
                    "train_rows": len(x_train),
                    "val_rows": len(x_val),
                },
            },
            settings.model_path,
        )

        db.complete_model_run(
            conn,
            run_id,
            status="ok",
            n_rows=len(rows),
            brier=brier,
            logloss=ll,
            auc=auc,
        )
        return True, f"trained rows={len(rows)} brier={brier:.4f} auc={auc:.3f}"
    except Exception as exc:
        db.complete_model_run(conn, run_id, status="error", error=str(exc))
        return False, f"training error: {exc}"
