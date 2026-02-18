# Fraud Detection Dashboard — Case Study

## Overview

This project is a **decision-ready** fraud screening experience with two entry points:

- **Streamlit dashboard** (analyst / ops UI): EDA + scoring + thresholds + cost-minded views
- **FastAPI inference service** (integration-ready): `/predict`, `/metadata`, `/predict/batch` using the same artifact bundle

The goal is not “just a model”, but a small operational system:
**validated inputs → calibrated probabilities → policy threshold → action**.

## Problem

Fraud detection is always a **risk + capacity** tradeoff:

- False negatives (missed fraud) are expensive
- False positives waste review capacity and damage UX
- The “best” threshold depends on business costs, review bandwidth, and the fraud base rate

## Solution

1) Train two models (RF + XGBoost) and calibrate probabilities  
2) Store artifacts + metadata (schema + environment versions) under `artifacts/`  
3) Expose a strict inference API with input validation and stable response contracts  
4) Provide a dashboard that makes threshold decisions explicit and measurable

## What makes it production-minded?

- **Clean separation**: UI calls API; model code is not embedded in Streamlit
- **Artifact bundle contracts**: schema and thresholds are loaded from disk, validated, and exposed via `/metadata`
- **Fast feedback**: Ruff + Pytest in CI, plus a Docker build test stage
- **Version checks**: runtime can warn if `scikit-learn` or `xgboost` versions drift from the artifact metadata

## How to run

- Local: see `README.md`
- Docker Compose: `docker compose up --build`

## Testing

```bash
python -m pytest -q
```

## Limitations / Next steps

- Add **monitoring hooks** (latency, failures, drift metrics) for live ops
- Add **model registry** (versioned bundles) + “shadow mode” testing
- Add a more realistic **evaluation report** on a time-safe split (if timestamps exist)
