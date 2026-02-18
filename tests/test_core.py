from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fraud_dashboard.core.predict import apply_threshold, predict_proba
from fraud_dashboard.core.thresholds import get_default_model_key, get_primary_threshold
from fraud_dashboard.data.validation import coerce_numeric, validate_columns


class _ModelProba2D:
    def predict_proba(self, X):
        return np.array([[0.1, 0.9] for _ in range(len(X))])


class _ModelProba1D:
    def predict_proba(self, X):
        return np.array([0.7 for _ in range(len(X))])


class _ModelPredictOnly:
    def predict(self, X):
        # return class labels
        return np.array([1 for _ in range(len(X))])


def test_predict_proba_from_predict_proba_2d() -> None:
    X = pd.DataFrame({"a": [1, 2]})
    p = predict_proba(_ModelProba2D(), X)
    assert p.shape == (2,)
    assert float(p[0]) == pytest.approx(0.9)


def test_predict_proba_from_predict_proba_1d() -> None:
    X = pd.DataFrame({"a": [1, 2, 3]})
    p = predict_proba(_ModelProba1D(), X)
    assert p.shape == (3,)
    assert float(p[1]) == pytest.approx(0.7)


def test_predict_proba_fallback_to_predict() -> None:
    X = pd.DataFrame({"a": [1, 2]})
    p = predict_proba(_ModelPredictOnly(), X)
    assert p.shape == (2,)
    assert set(p.tolist()) <= {0.0, 1.0}


def test_apply_threshold() -> None:
    proba = np.array([0.1, 0.6, 0.5])
    labels = apply_threshold(proba, 0.5)
    assert labels.tolist() == [0, 1, 1]


def test_get_primary_threshold_variants() -> None:
    assert get_primary_threshold(0.42, model_name="rf") == pytest.approx(0.42)
    assert get_primary_threshold({"primary": 0.33}, model_name="rf") == pytest.approx(0.33)
    assert get_primary_threshold({"rf": 0.21}, model_name="rf") == pytest.approx(0.21)
    assert get_primary_threshold([0.1, 0.2, 0.3], model_name="rf") == pytest.approx(0.1)


def test_get_primary_threshold_missing_raises() -> None:
    with pytest.raises(KeyError):
        get_primary_threshold({}, model_name="rf")


def test_get_default_model_key() -> None:
    assert get_default_model_key({"default_model": "rf"}) == "rf"
    assert get_default_model_key({"default_model": None}) is None


def test_validate_and_coerce_numeric() -> None:
    df = pd.DataFrame({"a": ["1.0", "x"], "b": [2, 3]})
    validate_columns(df, required=["a", "b"])
    out = coerce_numeric(df, cols=["a", "b"])
    assert out["a"].isna().iloc[1]
    assert float(out["b"].iloc[0]) == pytest.approx(2.0)


def test_validate_columns_missing_raises() -> None:
    df = pd.DataFrame({"a": [1]})
    with pytest.raises(ValueError):
        validate_columns(df, required=["a", "b"])
