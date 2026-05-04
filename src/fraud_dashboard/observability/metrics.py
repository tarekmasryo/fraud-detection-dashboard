from __future__ import annotations

try:
    from prometheus_client import Counter, Histogram, generate_latest
except Exception:  # pragma: no cover
    Counter = Histogram = None  # type: ignore[assignment]
    generate_latest = None  # type: ignore[assignment]


if Counter is not None:
    HTTP_REQUESTS_TOTAL = Counter(
        "fraud_http_requests_total",
        "HTTP requests handled by the API",
        ["method", "path", "status"],
    )
    HTTP_REQUEST_DURATION = Histogram(
        "fraud_http_request_duration_seconds", "HTTP request duration", ["method", "path"]
    )
    PREDICTION_REQUESTS_TOTAL = Counter(
        "fraud_prediction_requests_total", "Prediction requests", ["model", "source"]
    )
    PREDICTION_ERRORS_TOTAL = Counter(
        "fraud_prediction_errors_total", "Prediction errors", ["model", "error_type"]
    )
    PREDICTION_LATENCY = Histogram(
        "fraud_prediction_latency_seconds", "Prediction latency", ["model", "source"]
    )
    HIGH_RISK_TOTAL = Counter(
        "fraud_high_risk_predictions_total", "High-risk predictions", ["model"]
    )
    THRESHOLD_POLICY_USAGE = Counter(
        "fraud_threshold_policy_usage_total", "Threshold policy usage", ["model", "policy"]
    )
    BATCH_JOBS_TOTAL = Counter("fraud_batch_jobs_total", "Batch jobs", ["status"])
else:  # pragma: no cover
    HTTP_REQUESTS_TOTAL = HTTP_REQUEST_DURATION = PREDICTION_REQUESTS_TOTAL = None
    PREDICTION_ERRORS_TOTAL = PREDICTION_LATENCY = HIGH_RISK_TOTAL = None
    THRESHOLD_POLICY_USAGE = BATCH_JOBS_TOTAL = None


def render_metrics() -> bytes:
    if generate_latest is None:
        return b"# prometheus_client not installed\n"
    return generate_latest()
