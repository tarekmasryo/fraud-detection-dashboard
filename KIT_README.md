# App/Dashboard Kit

## What this adds
- ruff config (`pyproject.toml`)
- pre-commit hooks (`.pre-commit-config.yaml`)
- CI pipeline (`.github/workflows/ci.yml`)
- smoke tests (`tests/`)
- common docs (`CONTRIBUTING.md`, `SECURITY.md`)
- optional helpers (`Makefile`, `.editorconfig`, `.gitignore`)
- license templates (`LICENSES/`)

## What you MUST edit after copying
1) Update `Makefile` -> `run` target to point to your entrypoint:
   - Streamlit: `streamlit_app.py` or `app.py`
   - FastAPI: `uvicorn api.main:app --reload`
2) Choose one license template in `LICENSES/` and copy it to `LICENSE`.
3) Ensure `README.md` run commands match your real entrypoint.

## Local commands
```bash
pip install -r requirements.txt
pip install -r requirements-dev.txt
pre-commit install
pre-commit run --all-files
pytest -q
```
