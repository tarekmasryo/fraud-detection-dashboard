SHELL := /bin/bash

.PHONY: install dev run lint test

install:
	python -m pip install -U pip
	if [ -f requirements.txt ]; then pip install -r requirements.txt; fi
	if [ -f requirements-dev.txt ]; then pip install -r requirements-dev.txt; else pip install ruff pytest pre-commit; fi

dev:
	python -m pip install -U pip
	pip install -r requirements-dev.txt

# Update this if your entrypoint differs
run:
	streamlit run streamlit_app.py

lint:
	ruff check . --fix
	ruff format .

test:
	pytest -q
