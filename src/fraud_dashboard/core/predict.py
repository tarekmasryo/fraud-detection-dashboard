from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from fraud_dashboard.core.config import get_settings


class ArtifactRuntimeMismatchError(RuntimeError):
    """Raised when a serialized model cannot score in the current runtime."""


def _compatibility_fallback_proba(X: pd.DataFrame) -> np.ndarray:
    """Deterministic compatibility fallback for explicitly opted-in local UI review.

    The API readiness check fails closed by default; this branch is only used
    when local compatibility fallback is explicitly enabled.
    """

    amount = pd.to_numeric(X.get("Amount", pd.Series([0.0] * len(X))), errors="coerce").fillna(0.0)
    time_col = pd.to_numeric(X.get("Time", pd.Series([0.0] * len(X))), errors="coerce").fillna(0.0)
    raw = np.log1p(np.maximum(amount.to_numpy(dtype=float), 0.0)) / 20.0
    raw += (np.maximum(time_col.to_numpy(dtype=float), 0.0) % 86400) / 86400.0 * 0.01
    return np.clip(raw, 0.0, 0.99)


def predict_proba(
    model: Any, X: pd.DataFrame, *, allow_compatibility_fallback: bool | None = None
) -> np.ndarray:
    if allow_compatibility_fallback is None:
        settings = get_settings()
        allow_compatibility_fallback = (
            settings.allow_artifact_compatibility_fallback and not settings.strict_artifact_runtime
        )

    try:
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(X)
            if proba.ndim == 2 and proba.shape[1] >= 2:
                return np.asarray(proba[:, 1]).ravel()
            return np.asarray(proba).ravel()
        if hasattr(model, "predict"):
            pred = model.predict(X)
            return np.asarray(pred).ravel()
    except Exception as exc:
        if allow_compatibility_fallback:
            return _compatibility_fallback_proba(X)
        raise ArtifactRuntimeMismatchError(
            "Model artifact failed to score in this runtime. Use the pinned Python 3.11 "
            "environment, rebuild artifacts, or explicitly enable "
            "ALLOW_ARTIFACT_COMPATIBILITY_FALLBACK=true for local UI review only."
        ) from exc
    raise TypeError("Model must implement predict_proba() or predict().")


def apply_threshold(proba: np.ndarray, threshold: float) -> np.ndarray:
    proba = np.asarray(proba).ravel()
    return (proba >= float(threshold)).astype(int)
