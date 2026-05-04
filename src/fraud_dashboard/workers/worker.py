from __future__ import annotations

import logging
import time

from fraud_dashboard.api.schemas import PredictBatchRequest
from fraud_dashboard.core.config import get_settings
from fraud_dashboard.core.logging import configure_logging
from fraud_dashboard.platform.queue import dequeue_job
from fraud_dashboard.platform.store import get_store
from fraud_dashboard.services.batch_jobs import run_batch_job

configure_logging()
logger = logging.getLogger(__name__)


def _process_job(job: dict) -> None:
    request_payload = job.get("request") or {}
    req = PredictBatchRequest(**request_payload)
    run_batch_job(str(job["id"]), req)


def run_once() -> bool:
    """Process one queued job when available."""
    store = get_store()
    job_id = dequeue_job(timeout_s=get_settings().worker_poll_seconds)
    job = store.claim_job(job_id) if job_id else None
    if job is None:
        job = store.claim_next_job()
    if job is None:
        return False
    logger.info("worker_job_claimed", extra={"job_id": job["id"]})
    _process_job(job)
    return True


def main() -> None:
    settings = get_settings()
    logger.info(
        "Fraud Risk Ops worker started",
        extra={"database_url": settings.database_url, "redis_url": settings.redis_url},
    )
    print("Fraud Risk Ops worker started. Consuming queued batch jobs.", flush=True)
    while True:
        processed = run_once()
        if not processed:
            time.sleep(max(settings.worker_poll_seconds, 1))


if __name__ == "__main__":
    main()
