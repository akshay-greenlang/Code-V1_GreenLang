.PHONY: help install dev test unit integ e2e cov test-all lint demo fetch-opa clean security-scan doctor

# Variables
OPA_VERSION ?= 0.64.0
PYTHON ?= python3

help:
	@echo "GreenLang Development Commands"
	@echo ""
	@echo "  make install         Install GreenLang"
	@echo "  make dev             Install in development mode with dev dependencies"
	@echo "  make test            Run tests with coverage"
	@echo "  make lint            Run linters"
	@echo "  make security-scan   Run security scans (secrets & dependencies)"
	@echo "  make doctor          Check development environment"
	@echo "  make demo            Run the demo pipeline"
	@echo "  make fetch-opa       Download OPA binary"
	@echo "  make clean           Clean build artifacts"

install:
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install .

dev:
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -e ".[dev]"
	pre-commit install

test: ## Run tests with coverage
	coverage run -m pytest
	coverage report -m

unit: ## Run unit tests only
	pytest -q -m "not integration and not e2e"

integ: ## Run integration tests
	pytest -q -m integration

e2e: ## Run end-to-end tests
	pytest -q -m e2e

cov: ## Run all tests with coverage report
	coverage run -m pytest
	coverage report -m
	coverage xml
	@echo "Coverage XML generated: coverage.xml"

test-all: ## Run all tests (unit, integration, e2e)
	pytest -v tests/

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

security-scan: ## Run security scans for secrets and vulnerabilities
	@echo "Running pip-audit for dependency vulnerabilities..."
	-$(PYTHON) -m pip install pip-audit
	$(PYTHON) -m pip_audit --strict --desc
	@echo ""
	@echo "Running TruffleHog for secret scanning..."
	trufflehog filesystem --no-update . || echo "Install TruffleHog: https://github.com/trufflesecurity/trufflehog"

doctor: ## Check development environment and configuration
	@echo "Checking Python version..."
	$(PYTHON) --version
	@echo ""
	@echo "Checking installed packages..."
	$(PYTHON) -m pip list | grep -E "(greenlang|pytest|coverage|pip-audit)" || true
	@echo ""
	@echo "Checking for signing configuration..."
	@if [ -z "$$GL_SIGNING_MODE" ]; then echo "GL_SIGNING_MODE not set (will default to ephemeral in tests)"; else echo "GL_SIGNING_MODE=$$GL_SIGNING_MODE"; fi
	@echo ""
	@echo "Running greenlang doctor..."
	$(PYTHON) -m greenlang.cli doctor || echo "GreenLang doctor not available"