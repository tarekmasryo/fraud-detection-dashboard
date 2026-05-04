# Trade-offs

## SQLite-first persistence

The repository uses SQLite by default so reviewers can run the full system without a managed database. The schema is platform-oriented and keeps operational records explicit: prediction requests, prediction rows, audit events, jobs, model versions, and threshold policies.

## SQLite-backed queue + Redis wake-up

Docker Compose runs a worker process. Jobs are persisted in SQLite and Redis is used as a lightweight wake-up queue for job ids. SQLite remains the source of truth, which keeps the runtime durable for local technical review without adding unnecessary queue infrastructure. The Redis integration intentionally uses a small buffered RESP client instead of adding `redis-py`; that keeps dependencies compact, but a high-throughput production queue should replace it with a maintained Redis client or a managed job system such as RQ/Celery.

## Environment-controlled auth

The repo includes environment-controlled JWT/API-key gates to demonstrate protected API flows. Protected mode rejects unsafe default secrets and keeps the authentication boundary visible in the API contract.

## Artifact compatibility fallback

The official runtime is Python 3.11 with pinned ML dependencies. `STRICT_ARTIFACT_RUNTIME=false` allows a deterministic compatibility fallback when old serialized sklearn artifacts are executed on newer runtimes. Set `STRICT_ARTIFACT_RUNTIME=true` when you want readiness and prediction to fail closed.

## Docker image size

`xgboost==3.0.2` keeps the shipped artifacts compatible with the runtime. The trade-off is a heavier Linux wheel during container builds. The release prioritizes reproducibility and artifact compatibility over minimal image size.


## Audit log pagination

The audit-log endpoint supports `limit` and `offset`, which is enough for local review and lightweight operator inspection. Cursor pagination, retention windows, and archival storage are intentionally left to downstream production hardening.

## Training metrics boundary

Packaged artifacts are reference artifacts for running the platform. The training script uses separate train, calibration, and holdout-test splits so regenerated artifacts can report honest holdout metrics after calibration and threshold selection.
