# Engineering Notes

This document explains the clean-code and design decisions behind the public reference implementation.

## Design principles

1. **Keep the release reviewable.** Avoid feature stuffing that makes the project look larger but weaker.
2. **Separate score from decision.** The model returns a probability; the policy layer converts it into an operating decision.
3. **Fail closed on artifact readiness.** Runtime mismatch is surfaced clearly instead of hidden behind silent fallback.
4. **Make operational state explicit.** Predictions, jobs, and audit events are persisted rather than only printed to logs.
5. **Keep claims tied to implementation.** The project presents the platform boundaries it actually implements.

## Patterns used

| Pattern | Where | Why |
|---|---|---|
| **Adapter** | `ui/predictors.py` | The UI can talk to the API or local artifacts through the same predictor interface. |
| **Repository** | `platform/store.py` | SQLite persistence is isolated behind an operations store. |
| **Policy object / configuration as contract** | `artifacts/policy.json`, `core/policy.py` | Thresholds and operating policies stay outside model binaries. |
| **Fail-closed readiness gate** | `core/artifact_contract.py` | `/ready` catches broken artifacts/runtime mismatch before serving decisions. |
| **Worker entrypoint** | `workers/worker.py` | Batch processing consumes persisted jobs and uses the shared scoring path. |
| **Explicit response contract** | `api/main.py` | Risk decisions are structured, auditable, and stable enough for review. |

## Why the architecture stays compact

The release keeps the platform small enough to run, inspect, and critique quickly. It includes the operating layers that matter for the use case — API contracts, policies, audit records, jobs, readiness, metrics, UI, and Docker deployment — without adding unrelated product surface area.

## Code-quality boundaries

- Keep domain helpers in `core/`.
- Keep persistence in `platform/`.
- Keep Streamlit-specific code in `ui/`.
- Keep route orchestration in `api/`.
- Avoid adding business claims that the code does not support.
- Prefer explicit failures over silent fallback for model, policy, and runtime issues.
