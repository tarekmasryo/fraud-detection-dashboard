# Makefile (Windows / cmd.exe) — one-button workflow
# Usage: make all

SHELL := cmd.exe
.SHELLFLAGS := /C

PY := .venv\Scripts\python.exe

.PHONY: help venv deps install format lint test api ui all clean

help:
	@echo Targets:
	@echo   make venv     - create .venv
	@echo   make deps     - install requirements
	@echo   make install  - pip install -e .
	@echo   make format   - ruff format --check
	@echo   make lint     - ruff check
	@echo   make test     - pytest
	@echo   make api      - run FastAPI (uvicorn)
	@echo   make ui       - run Streamlit UI
	@echo   make all      - deps+checks then launch api+ui
	@echo   make clean    - remove .venv/.pytest_cache/__pycache__

venv:
	@if exist ".venv" (echo .venv already exists) else (py -3.11 -m venv .venv)

deps: venv
	@"$(PY)" -m pip install -U pip setuptools wheel
	@"$(PY)" -m pip install -r requirements.txt -r requirements-dev.txt

install: deps
	@"$(PY)" -m pip install -e .

format: install
	@"$(PY)" -m ruff format --check .

lint: install
	@"$(PY)" -m ruff check .

test: install
	@"$(PY)" -m pytest -q

api: install
	@"$(PY)" -m uvicorn fraud_dashboard.api.main:app --reload --reload-dir src --host 127.0.0.1 --port 8000

ui: install
	@"$(PY)" -m streamlit run streamlit_app.py

all: format lint test
	@echo Launching API + UI in separate terminals...
	@start "API" cmd /k ""$(PY)" -m uvicorn fraud_dashboard.api.main:app --reload --reload-dir src --host 127.0.0.1 --port 8000"
	@start "UI"  cmd /k ""$(PY)" -m streamlit run streamlit_app.py"
	@echo API Docs: http://127.0.0.1:8000/docs
	@echo UI      : http://localhost:8501

clean:
	@if exist ".venv" rmdir /s /q ".venv"
	@if exist ".pytest_cache" rmdir /s /q ".pytest_cache"
	@for /d /r %%d in (__pycache__) do @if exist "%%d" rmdir /s /q "%%d"


.PHONY: rebuild-venv fresh doctor

doctor:
	@echo PYVENV:
	@if exist ".venv\pyvenv.cfg" type ".venv\pyvenv.cfg" else echo (no venv yet)
	@echo PYEXE:
	@if exist ".venv\Scripts\python.exe" (".venv\Scripts\python.exe" -c "import sys; print(sys.executable)") else echo (missing .venv\Scripts\python.exe)

rebuild-venv:
	@if exist ".venv" rmdir /s /q ".venv"
	@py -3.11 -m venv .venv

fresh: rebuild-venv all
