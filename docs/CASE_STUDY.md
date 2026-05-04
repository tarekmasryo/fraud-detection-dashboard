# Fraud Risk Ops Platform — Case Study

## Problem

A fraud model is rarely useful as a naked notebook or isolated prediction endpoint. Operators need a workflow that makes model scores reviewable, auditable, and measurable.

## Approach

This repository wraps calibrated model artifacts with a compact operating layer:

- FastAPI scoring routes with schema validation
- policy-driven threshold resolution
- persisted prediction requests and audit logs
- synchronous and asynchronous batch flows
- Streamlit review console for operators
- Prometheus/Grafana monitoring hooks
- explicit docs for architecture, scope, and operations

## Key design choice

The project separates **model score** from **policy decision**:

```text
model -> risk_score
policy -> threshold
decision layer -> approve/review
```

This keeps threshold governance outside the model artifact and makes policy changes easier to inspect.

## Outcome

The result is a production-structured public reference implementation that demonstrates practical ML system design around fraud-risk operations.
