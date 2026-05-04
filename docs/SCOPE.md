# Release Scope

This release focuses on the engineering system around fraud-risk decisions.

Implemented scope:

- FastAPI inference and platform endpoints.
- Streamlit review console.
- Shared threshold policy governance.
- Fail-closed artifact readiness checks.
- SQLite-backed operational records.
- Audit logs for prediction and job activity.
- Worker-backed batch jobs with Redis wake-up signaling.
- Prometheus metrics and Grafana provisioning.
- Docker Compose deployment for local technical review.

Operational boundaries:

- The repository uses synthetic/local runtime defaults and does not include real customer data.
- The dataset is not redistributed; compatible local CSV input is supported.
- The project does not include payment-network integrations, tenant isolation, billing, or regulatory-compliance workflows.
- `reason_codes` are deterministic operational review hints designed for auditability and review workflow clarity.

These boundaries keep the release focused, runnable, and easy to inspect while preserving the core platform architecture around ML risk decisions.
