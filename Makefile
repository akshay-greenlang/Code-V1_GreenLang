.PHONY: help install dev test lint demo fetch-opa clean

# Variables
OPA_VERSION ?= 0.64.0
PYTHON ?= python3

help:
	@echo "GreenLang Development Commands"
	@echo ""
	@echo "  make install      Install GreenLang"
	@echo "  make dev          Install in development mode with dev dependencies"
	@echo "  make test         Run tests"
	@echo "  make lint         Run linters"
	@echo "  make demo         Run the demo pipeline"
	@echo "  make fetch-opa    Download OPA binary"
	@echo "  make clean        Clean build artifacts"

install:
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install .

dev:
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -e ".[dev]"
	pre-commit install

test:
	pytest tests/ -v

lint:
	black greenlang/ tests/
	isort greenlang/ tests/
	flake8 greenlang/ tests/

demo:
	$(PYTHON) -m greenlang.cli demo run

fetch-opa:
	$(PYTHON) scripts/fetch_opa.py $(OPA_VERSION)

clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info
	rm -rf .pytest_cache/
	rm -rf .coverage
	rm -rf htmlcov/
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete