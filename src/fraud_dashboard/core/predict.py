from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def predict_proba(model: Any, X: pd.DataFrame) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)
        if proba.ndim == 2 and proba.shape[1] >= 2:
            return proba[:, 1]
        return proba.ravel()
    if hasattr(model, "predict"):
        pred = model.predict(X)
        return np.asarray(pred).ravel()
    raise TypeError("Model must implement predict_proba() or predict().")


def apply_threshold(proba: np.ndarray, threshold: float) -> np.ndarray:
    proba = np.asarray(proba).ravel()
    return (proba >= float(threshold)).astype(int)
