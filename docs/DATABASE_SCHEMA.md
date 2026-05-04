# Database Schema

The local operations store uses SQLite for first-run reproducibility. Connections enable WAL mode and a busy timeout to support the API/worker local-compose topology.

## `prediction_requests`

Tracks each scoring request.

| Column | Purpose |
|---|---|
| `id` | Request id, e.g. `pred_...`. |
| `created_at` | Unix timestamp. |
| `model_key` | Selected model key. |
| `threshold` | Applied decision threshold. |
| `record_count` | Number of scored rows. |
| `latency_ms` | API/model scoring latency. |
| `source` | Request source, e.g. API, legacy, async job. |
| `status` | Request status. |
| `policy_name` | Applied policy name. |
| `policy_version` | Applied policy version. |

## `predictions`

Tracks row-level scoring output.

| Column | Purpose |
|---|---|
| `id` | Row prediction id. |
| `request_id` | Parent request. |
| `row_index` | Row position within the request. |
| `proba_fraud` | Fraud risk score. |
| `label` | Binary threshold label. |
| `decision` | Public decision contract: `approve` or `review`. |
| `reason_codes_json` | JSON list of operational review hints. |
| `input_hash` | Stable hash for audit correlation without storing raw payload. |

## `audit_logs`

Stores user/system actions and metadata.

## `batch_jobs`

Stores asynchronous batch lifecycle state.

## `model_versions`

Stores the active artifact inventory seeded from `artifacts/metadata.json`. This keeps model-version state inspectable through the operational database while the release remains file-artifact based.

| Column | Purpose |
|---|---|
| `id` | Stable local row id. |
| `created_at` | Unix timestamp for first seed. |
| `model_key` | Normalized model key such as `rf` or `xgb`. |
| `artifact_file` | Serialized model artifact filename. |
| `active` | Active model flag. |
| `metadata_json` | Artifact version, training timestamp, pipeline file, checksum metadata. |

## `threshold_policies`

Stores named operating thresholds seeded from `artifacts/policy.json`.

| Column | Purpose |
|---|---|
| `id` | Stable local row id. |
| `created_at` | Unix timestamp for first seed. |
| `policy_name` | Named operating policy. |
| `policy_version` | Policy version from the artifact. |
| `threshold` | Applied review threshold. |
| `model_key` | Normalized model key. |
| `metadata_json` | Policy description and release metadata. |

## Operational summary

`GET /v1/metrics/summary` aggregates prediction requests, predictions, job states, policy usage, model usage, and active model-version counts from this store.

SQLite keeps this release easy to run and inspect while still making operational state explicit. The schema is intentionally shaped around platform records: requests, predictions, jobs, policies, model versions, and audit events.
