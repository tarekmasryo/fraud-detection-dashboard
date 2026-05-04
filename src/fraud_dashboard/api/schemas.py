from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, field_validator

from fraud_dashboard.core.config import get_settings


class PredictRequest(BaseModel):
    record: dict[str, Any] = Field(..., description="Single record as a feature->value mapping.")
    model: str | None = Field(None, description="Model key (e.g., rf, xgb). Defaults to policy.")
    policy: str | None = Field(
        None, description="Named threshold policy from artifacts/policy.json."
    )
    threshold: float | None = Field(None, ge=0.0, le=1.0, description="Override threshold.")


class PredictBatchRequest(BaseModel):
    records: list[dict[str, Any]] = Field(..., min_length=1, description="Batch of records.")
    model: str | None = Field(None)
    policy: str | None = Field(None)
    threshold: float | None = Field(None, ge=0.0, le=1.0)

    @field_validator("records")
    @classmethod
    def _validate_batch_size(cls, value: list[dict[str, Any]]) -> list[dict[str, Any]]:
        limit = get_settings().max_batch_records
        if len(value) > limit:
            raise ValueError(f"Batch size {len(value)} exceeds MAX_BATCH_RECORDS={limit}")
        return value


class LoginRequest(BaseModel):
    username: str
    password: str


class PredictResponse(BaseModel):
    model: str
    threshold: float
    proba_fraud: float
    risk_score: float
    label: int
    decision: str
    review_required: bool
    risk_band: str
    reason_codes: list[str]
    policy: str | None = None
    policy_version: str | None = None
    input_hash: str | None = None
    latency_ms: int
    request_id: str | None = None


class PredictBatchResponse(BaseModel):
    model: str
    threshold: float
    results: list[PredictResponse]
    latency_ms: int
    request_id: str | None = None


class BatchJobCreateResponse(BaseModel):
    job_id: str
    status: str
