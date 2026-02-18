from __future__ import annotations

import time
from functools import lru_cache
from typing import Any

import pandas as pd
from fastapi import FastAPI, HTTPException, Query, Response
from pydantic import BaseModel, Field

from fraud_dashboard.core.artifacts import load_bundle
from fraud_dashboard.core.predict import apply_threshold, predict_proba
from fraud_dashboard.core.thresholds import get_model_threshold, normalize_model_key, pick_model_key
from fraud_dashboard.data.validation import (
    coerce_numeric,
    validate_columns,
    validate_finite,
    validate_no_nan,
)


class PredictRequest(BaseModel):
    record: dict[str, Any] = Field(..., description="Single record as a feature->value mapping.")
    model: str | None = Field(
        None, description="Model key (e.g., calibrated_rf, xgb). Defaults to repo policy."
    )
    threshold: float | None = Field(None, description="Override decision threshold (0..1).")


class PredictBatchRequest(BaseModel):
    records: list[dict[str, Any]] = Field(..., min_length=1, description="Batch of records.")
    model: str | None = Field(None)
    threshold: float | None = None


class PredictResponse(BaseModel):
    model: str
    threshold: float
    proba_fraud: float
    label: int
    latency_ms: int


class PredictBatchResponse(BaseModel):
    model: str
    threshold: float
    results: list[PredictResponse]
    latency_ms: int


@lru_cache(maxsize=1)
def _bundle():
    return load_bundle()


def _required_features() -> list[str]:
    schema = _bundle().get("schema", {})
    feats = schema.get("features", [])
    if not feats:
        raise RuntimeError("metadata.json missing schema.features")
    return list(feats)


def _make_df(records: list[dict[str, Any]]) -> pd.DataFrame:
    df = pd.DataFrame(records)
    try:
        validate_columns(df, _required_features())
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    df = df[_required_features()]  # enforce order + drop extras
    df = coerce_numeric(df, _required_features())
    try:
        validate_no_nan(df, _required_features())
        validate_finite(df, _required_features())
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return df


def _resolve_policy(model: str | None, threshold: float | None) -> tuple[str, float]:
    b = _bundle()
    models = b["models"]
    thresholds = b["thresholds"]

    req_model = normalize_model_key(model)
    model_key = req_model or pick_model_key(models, thresholds)

    if model_key not in models:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown model '{model_key}'. Available: {sorted(models.keys())}",
        )

    th = float(threshold if threshold is not None else get_model_threshold(thresholds, model_key))
    if not (0.0 <= th <= 1.0):
        raise HTTPException(status_code=500, detail=f"Invalid threshold for '{model_key}': {th}")
    return model_key, th


app = FastAPI(
    title="Fraud Detection API",
    version="1.0.0",
    description="Inference API for the Fraud Detection Dashboard artifacts.",
)


@app.get("/", include_in_schema=False)
def root() -> dict[str, str]:
    """Landing endpoint (polish): points users to the interactive docs."""
    return {
        "message": "Use /docs for the OpenAPI UI. Useful endpoints: /health, /metadata, /predict, /predict/batch."
    }


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/favicon.ico", include_in_schema=False)
def favicon() -> Response:
    # Avoid noisy 404s when browsers hit /favicon.ico.
    return Response(status_code=204)


@app.get("/metadata")
def metadata() -> dict[str, Any]:
    b = _bundle()
    thresholds = b.get("thresholds", {})
    models = b.get("models", {})
    tbm = {}
    for mk in sorted(models.keys()):
        try:
            tbm[mk] = float(get_model_threshold(thresholds, mk))
        except Exception:
            continue

    return {
        "available_models": b.get("available_models", []),
        "default_model": pick_model_key(models, thresholds) if models else None,
        "thresholds": thresholds,
        "thresholds_by_model": tbm,
        "schema": b.get("schema", {}),
        "env": (b.get("metadata") or {}).get("env", {}),
    }


@app.post("/predict", response_model=PredictResponse)
def predict(
    req: PredictRequest,
    model: str | None = Query(
        default=None,
        description="Optional model key via query string. Overrides body.model if provided.",
        examples=["rf", "xgb"],
    ),
    threshold: float | None = Query(
        default=None,
        ge=0.0,
        le=1.0,
        description="Optional threshold override via query string (0..1). Overrides body.threshold if provided.",
    ),
) -> PredictResponse:
    t0 = time.perf_counter()
    model_key, th = _resolve_policy(
        model if model is not None else req.model,
        threshold if threshold is not None else req.threshold,
    )
    df = _make_df([req.record])

    proba = float(predict_proba(_bundle()["models"][model_key], df)[0])
    label = int(apply_threshold([proba], th)[0])

    dt_ms = int((time.perf_counter() - t0) * 1000)
    return PredictResponse(
        model=model_key, threshold=th, proba_fraud=proba, label=label, latency_ms=dt_ms
    )


@app.post("/predict/batch", response_model=PredictBatchResponse)
def predict_batch(
    req: PredictBatchRequest,
    model: str | None = Query(
        default=None,
        description="Optional model key via query string. Overrides body.model if provided.",
        examples=["rf", "xgb"],
    ),
    threshold: float | None = Query(
        default=None,
        ge=0.0,
        le=1.0,
        description="Optional threshold override via query string (0..1). Overrides body.threshold if provided.",
    ),
) -> PredictBatchResponse:
    t0 = time.perf_counter()
    model_key, th = _resolve_policy(
        model if model is not None else req.model,
        threshold if threshold is not None else req.threshold,
    )
    df = _make_df(req.records)

    probs = predict_proba(_bundle()["models"][model_key], df)
    preds = apply_threshold(probs, th)

    results = []
    for p, y in zip(probs.tolist(), preds.tolist(), strict=False):
        results.append(
            PredictResponse(
                model=model_key,
                threshold=th,
                proba_fraud=float(p),
                label=int(y),
                latency_ms=0,
            )
        )

    dt_ms = int((time.perf_counter() - t0) * 1000)
    # Keep per-item latency_ms at 0, aggregate at batch-level (simpler & honest).
    return PredictBatchResponse(model=model_key, threshold=th, results=results, latency_ms=dt_ms)
