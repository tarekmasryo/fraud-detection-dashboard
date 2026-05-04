# Job Lifecycle

## Create

`POST /v1/batch-jobs` validates the batch payload, creates a row in `batch_jobs`, and returns a job id.

## Queue

In Docker Compose, `RUN_JOBS_IN_API=false`. The API pushes the job id into Redis as a wake-up event. SQLite remains the source of truth.

## Claim

The worker receives a Redis job id or polls SQLite for queued jobs. It atomically claims a job by moving it from `queued` to `running`.

## Execute

The worker reconstructs the persisted request payload, scores records with the same policy/model path as the API, writes prediction history and audit logs, then marks the job `completed` or `failed`.

## Inspect

`GET /v1/jobs/{job_id}` returns the current job status, counts, result payload, and error details if any.
