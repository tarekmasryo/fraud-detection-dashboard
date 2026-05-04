from __future__ import annotations

from fraud_dashboard.api.schemas import PredictBatchRequest
from fraud_dashboard.core.config import get_settings
from fraud_dashboard.observability.metrics import BATCH_JOBS_TOTAL
from fraud_dashboard.platform.store import get_store
from fraud_dashboard.services.scoring import FraudScoringService


def run_batch_job(
    job_id: str,
    req: PredictBatchRequest,
    *,
    scoring_service: FraudScoringService | None = None,
) -> None:
    """Execute a persisted batch job and write its terminal state.

    This function is intentionally framework-neutral so both the FastAPI process
    and the standalone worker can use the same job lifecycle without importing
    route-module internals.
    """

    store = get_store()
    service = scoring_service or FraudScoringService()
    try:
        out = service.score_records(
            req.records,
            model=req.model,
            threshold=req.threshold,
            policy=req.policy,
            source="async_job",
        )
        store.update_job(
            job_id,
            status="completed",
            processed_records=len(out.results),
            failed_records=0,
            result=out.model_dump(),
        )
        if get_settings().prometheus_enabled and BATCH_JOBS_TOTAL is not None:
            BATCH_JOBS_TOTAL.labels(status="completed").inc()
    except Exception as exc:
        store.update_job(
            job_id,
            status="failed",
            processed_records=0,
            failed_records=len(req.records),
            error=f"{type(exc).__name__}: {exc}",
        )
        if get_settings().prometheus_enabled and BATCH_JOBS_TOTAL is not None:
            BATCH_JOBS_TOTAL.labels(status="failed").inc()
