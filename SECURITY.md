# Security Policy

## Reporting a vulnerability

If you believe you found a security issue, please open a private report via GitHub Security Advisories (preferred) or open an issue with **no sensitive details**.

## Dependency updates

This repository ships **pre-trained serialized artifacts** in `artifacts/`. To keep the repo runnable "out of the box", the ML/data stack is pinned on `main`.

- CI runs `pip-audit` in **non-blocking** mode (visibility without breaking builds).
- Dependabot ignores ML/data stack upgrades on `main` (`numpy`, `scipy`, `scikit-learn`, `xgboost`, `joblib`, `pandas`, `streamlit`).

For production use, follow a refresh cycle: upgrade deps → re-export artifacts → rerun tests → release.
