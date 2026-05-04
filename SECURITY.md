# Security Policy

## Reporting a vulnerability

If you believe you found a security issue, please open a private report via GitHub Security Advisories (preferred) or open an issue with **no sensitive details**.

## Dependency updates

This repository ships **pre-trained serialized artifacts** in `artifacts/`. To keep the repo runnable out of the box, the ML/data stack is pinned on `main`.

- CI runs `pip-audit` in **non-blocking** mode for dependency visibility.
- Dependabot is configured for Python and GitHub Actions. It intentionally ignores ML/data stack upgrades on `main` (`numpy`, `scipy`, `scikit-learn`, `xgboost`, `joblib`, `pandas`, `streamlit`) because the serialized artifacts are tied to the pinned runtime stack.

When the ML/data stack changes, refresh the artifacts and rerun the full validation suite before release.

## Protected-mode boundary

The repository includes environment-controlled authentication for local technical review. Do not use the default credentials or API key in any exposed environment. When `REQUIRE_AUTH=true`, the application validates that local placeholder secrets were replaced. The Streamlit console can forward `X-API-Key` or bearer credentials to the API for protected local runs.

The shipped artifacts and synthetic workflows are included to make the platform runnable and inspectable. Real payment-risk decisions require organization-specific data validation, governance approval, monitoring, and review procedures around the deployed environment.
