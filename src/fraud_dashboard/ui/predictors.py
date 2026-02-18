from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np
import pandas as pd
import streamlit as st

from fraud_dashboard.core.artifacts import load_bundle, load_thresholds
from fraud_dashboard.core.predict import predict_proba
from fraud_dashboard.ui.api_client import ApiConfig, fetch_metadata, ping_ok, predict_batch


@dataclass(frozen=True)
class BackendInfo:
    kind: str  # "api" or "local"
    details: str


class Predictor(Protocol):
    def info(self) -> BackendInfo: ...

    def available_models(self) -> list[str]: ...

    def thresholds(self) -> dict[str, Any]: ...

    def schema_features(self) -> list[str]: ...

    def predict_proba_batch(
        self, df_features: pd.DataFrame, *, model_key: str
    ) -> tuple[np.ndarray, float]: ...


@st.cache_resource(show_spinner=False)
def _local_bundle() -> dict[str, Any]:
    return load_bundle()


@st.cache_resource(show_spinner=False)
def _local_thresholds() -> dict[str, Any]:
    return load_thresholds()


class LocalPredictor:
    def __init__(self) -> None:
        self._bundle = _local_bundle()

    def info(self) -> BackendInfo:
        return BackendInfo(kind="local", details="Local artifacts (no API required)")

    def available_models(self) -> list[str]:
        return list(self._bundle.get("available_models", []))

    def thresholds(self) -> dict[str, Any]:
        return _local_thresholds()

    def schema_features(self) -> list[str]:
        schema = self._bundle.get("schema", {})
        feats = schema.get("features") or []
        return list(feats)

    def predict_proba_batch(
        self, df_features: pd.DataFrame, *, model_key: str
    ) -> tuple[np.ndarray, float]:
        t0 = time.perf_counter()
        model = self._bundle["models"][model_key]
        probs = predict_proba(model, df_features)
        dt = time.perf_counter() - t0
        return np.asarray(probs, dtype=float), float(dt)


class ApiPredictor:
    def __init__(self, cfg: ApiConfig) -> None:
        self.cfg = cfg
        self._meta: dict[str, Any] | None = None

    def _metadata(self) -> dict[str, Any]:
        if self._meta is None:
            self._meta = fetch_metadata(self.cfg)
        return self._meta

    def info(self) -> BackendInfo:
        return BackendInfo(kind="api", details=f"FastAPI @ {self.cfg.base_url}")

    def available_models(self) -> list[str]:
        return list(self._metadata().get("available_models", []))

    def thresholds(self) -> dict[str, Any]:
        return dict(self._metadata().get("thresholds", {}))

    def schema_features(self) -> list[str]:
        schema = self._metadata().get("schema", {})
        feats = schema.get("features") or []
        return list(feats)

    def predict_proba_batch(
        self, df_features: pd.DataFrame, *, model_key: str
    ) -> tuple[np.ndarray, float]:
        feats = list(df_features.columns)
        records = df_features[feats].to_dict(orient="records")
        t0 = time.perf_counter()
        out = predict_batch(cfg=self.cfg, model=model_key, records=records)
        dt = time.perf_counter() - t0
        probs = [r["proba_fraud"] for r in out.get("results", [])]
        return np.asarray(probs, dtype=float), float(dt)


def resolve_predictor(mode: str, api_url: str) -> tuple[Predictor, str]:
    """Select predictor given mode.

    mode: "Auto", "API", "Local"
    returns: (predictor, status_message)
    """

    api_cfg = ApiConfig(base_url=api_url.rstrip("/"), timeout_s=15.0)

    if mode == "Local":
        return LocalPredictor(), "Using local artifacts."
    if mode == "API":
        if not ping_ok(api_cfg):
            return LocalPredictor(), "API not reachable → fell back to local artifacts."
        return ApiPredictor(api_cfg), "Using FastAPI backend."

    # Auto
    if ping_ok(api_cfg):
        return ApiPredictor(api_cfg), "Auto: using FastAPI backend."
    return LocalPredictor(), "Auto: API not reachable → using local artifacts."


def ui_backend_selector() -> tuple[Predictor, dict[str, Any]]:
    st.sidebar.header("Backend")
    mode = st.sidebar.radio("Inference mode", ["Auto", "API", "Local"], horizontal=True)
    default_url = os.getenv("FRAUD_API_URL", "http://127.0.0.1:8000")
    api_url = st.sidebar.text_input("FastAPI base URL", value=default_url)
    predictor, msg = resolve_predictor(mode, api_url)

    if predictor.info().kind == "api":
        st.sidebar.success("API reachable")
    else:
        st.sidebar.info("Local mode")

    st.sidebar.caption(msg)

    meta = {"mode": mode, "api_url": api_url.rstrip("/")}
    return predictor, meta
