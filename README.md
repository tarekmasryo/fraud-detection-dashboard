# 💳 Credit Card Fraud Detection Dashboard

[![Powered by Streamlit](https://img.shields.io/badge/Powered%20by-Streamlit-FF4B4B)](https://streamlit.io/)
[![Made by Tarek Masryo](https://img.shields.io/badge/Made%20by-Tarek%20Masryo-blue)](https://github.com/tarekmasryo)

**Live demo:** https://fraud-detection-dashboard-mtwbg9xk6cr6kghzd2nrdg.streamlit.app/

---

## Overview

A production-minded **Streamlit + Plotly** dashboard for **credit card fraud detection** with a **business-aware decision layer** (thresholding + cost trade-offs).

Built on the classic **Credit Card Fraud Dataset** (284,807 transactions, 492 frauds ≈ 0.17%).

### What you can do

- Upload your own transactions CSV or use the built-in dataset
- Choose **Strict / Balanced / Lenient** policies (or set a custom threshold)
- Visualize **Confusion Matrix**, **ROC/PR curves**, and **cost vs threshold**
- Inspect **feature importance** for interpretability
- Review **segmented performance** (e.g., amount bands, time-of-day proxies)

---

## Dashboard Preview

### Data Overview
![Data](assets/data_overview.png)

### Prediction Engine
![Prediction](assets/prediction_engine.png)

### Model Metrics
![Metrics](assets/model_metrics.png)

### Model Insights
![Insights](assets/model_insights.png)

### Data Quality & Segments
![Segments](assets/data_quality.png)

---

## Key Features

- **Models**: Calibrated **RandomForest** and **XGBoost**
- **Decision policies**: Presets + custom thresholding
- **Threshold Finder**: Auto-select by target precision/recall
- **Cost analysis**: Business-aligned FP vs FN costs
- **Visual diagnostics**: Confusion matrix, ROC, PR, cost/threshold curves
- **Interpretability**: Permutation feature importance
- **Data handling**: Basic schema validation + engineered features (`log(Amount)`, business-hours proxy, night proxy)

---

## Run Locally

### 1) Clone

```bash
git clone https://github.com/tarekmasryo/fraud-detection-dashboard.git
cd fraud-detection-dashboard
```

### 2) Create & activate a virtual environment (recommended)

**Windows (PowerShell):**
```bash
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**macOS/Linux:**
```bash
python -m venv .venv
source .venv/bin/activate
```

### 3) Install dependencies

```bash
python -m pip install -U pip
python -m pip install -r requirements.txt
```

### 4) Start the app

```bash
streamlit run streamlit_app.py
```

> `streamlit_app.py` is the Streamlit Cloud entrypoint. It imports the dashboard from `app.py`.

---

## Development (Optional)

```bash
python -m pip install -r requirements-dev.txt
pre-commit run --all-files
python -m ruff check .
python -m ruff format --check .
python -m pytest -q
```

---

## Project Structure

```text
.
├─ streamlit_app.py          # Streamlit Cloud entrypoint
├─ app.py                    # Dashboard implementation
├─ artifacts/                # Pre-trained models/pipelines
├─ tests/                    # Smoke tests
├─ assets/                   # README screenshots
├─ requirements.txt
├─ requirements-dev.txt
└─ .github/workflows/        # CI (if enabled)
```

---

## Notes on Model Artifacts

This repo ships pre-trained artifacts under `artifacts/`.
If you change core ML dependencies (especially **scikit-learn**) without rebuilding the artifacts, model loading may break or become unreliable.

---

## Related Repositories

- 🔍 **Fraud Detection EDA + Baseline Models**: https://github.com/tarekmasryo/creditcard-fraud-detection

---

## Credits

If you use or reference this project, please credit:

> Credit Card Fraud Detection Dashboard — **Tarek Masryo**
