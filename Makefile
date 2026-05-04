.PHONY: help install format lint test test-cov api ui docker-up docker-down clean

PY ?= python

help:
	@echo "Targets:"
	@echo "  make install    Install runtime and dev dependencies"
	@echo "  make format     Check formatting with Ruff"
	@echo "  make lint       Run Ruff lint checks"
	@echo "  make test       Run unit tests"
	@echo "  make test-cov   Run tests with coverage gate"
	@echo "  make api        Run FastAPI locally"
	@echo "  make ui         Run Streamlit locally"
	@echo "  make docker-up  Start Docker Compose stack"
	@echo "  make docker-down Stop Docker Compose stack"
	@echo "  make clean      Remove local test/cache files"

install:
	$(PY) -m pip install -U pip setuptools wheel
	$(PY) -m pip install -r requirements.txt -r requirements-dev.txt
	$(PY) -m pip install -e .

format:
	$(PY) -m ruff format --check .

lint:
	$(PY) -m ruff check .

test:
	$(PY) -m pytest -q

test-cov:
	$(PY) -m pytest -q --cov=src --cov-fail-under=75

api:
	$(PY) -m uvicorn fraud_dashboard.api.main:app --reload --reload-dir src --host 127.0.0.1 --port 8000

ui:
	$(PY) -m streamlit run app.py

docker-up:
	docker compose up --build

docker-down:
	docker compose down -v

clean:
	rm -rf .pytest_cache .ruff_cache .mypy_cache .coverage htmlcov coverage.xml
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
