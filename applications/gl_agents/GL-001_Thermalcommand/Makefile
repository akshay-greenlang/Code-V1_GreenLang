# GL-001 ThermalCommand - Development Makefile
# Comprehensive code quality and development commands
# Version: 1.0.0

.PHONY: help install install-dev lint format type-check test test-unit test-integration \
        test-coverage security clean pre-commit quality-check all

# Default target
.DEFAULT_GOAL := help

# Python executable
PYTHON := python
PIP := pip

# Coverage threshold
COVERAGE_THRESHOLD := 85

# =============================================================================
# HELP
# =============================================================================

help: ## Show this help message
	@echo "GL-001 ThermalCommand - Development Commands"
	@echo ""
	@echo "Usage: make <target>"
	@echo ""
	@echo "Targets:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'

# =============================================================================
# INSTALLATION
# =============================================================================

install: ## Install production dependencies
	$(PIP) install -e .

install-dev: ## Install development dependencies
	$(PIP) install -e ".[dev]"
	pre-commit install
	pre-commit install --hook-type commit-msg

install-all: ## Install all optional dependencies
	$(PIP) install -e ".[all]"
	pre-commit install
	pre-commit install --hook-type commit-msg

# =============================================================================
# CODE QUALITY
# =============================================================================

format: ## Format code with Black
	$(PYTHON) -m black --config pyproject.toml .

format-check: ## Check code formatting with Black
	$(PYTHON) -m black --check --diff --config pyproject.toml .

lint: ## Lint code with Ruff
	$(PYTHON) -m ruff check --config pyproject.toml .

lint-fix: ## Lint and fix code with Ruff
	$(PYTHON) -m ruff check --fix --config pyproject.toml .

type-check: ## Run MyPy type checking
	$(PYTHON) -m mypy --config-file pyproject.toml \
		--install-types --non-interactive \
		--exclude 'tests/' --exclude 'deployment/' .

security: ## Run security checks with Bandit
	$(PYTHON) -m bandit -c pyproject.toml -r . \
		--exclude ./tests,./deployment,./.venv -ll

docstring-check: ## Check docstring coverage with Interrogate
	$(PYTHON) -m interrogate --config pyproject.toml --fail-under=80 .

# =============================================================================
# TESTING
# =============================================================================

test: ## Run all tests
	$(PYTHON) -m pytest tests/ -v --tb=short

test-unit: ## Run unit tests only
	$(PYTHON) -m pytest tests/test_unit/ -v --tb=short -m "unit"

test-simulation: ## Run simulation tests
	$(PYTHON) -m pytest tests/test_simulation/ -v --tb=short -m "simulation"

test-integration: ## Run integration tests
	$(PYTHON) -m pytest tests/ -v --tb=short -m "integration"

test-safety: ## Run safety tests
	$(PYTHON) -m pytest tests/test_safety/ -v --tb=short -m "safety"

test-performance: ## Run performance tests
	$(PYTHON) -m pytest tests/ -v --tb=short -m "performance" --timeout=120

test-determinism: ## Run determinism tests
	$(PYTHON) -m pytest tests/ -v --tb=short -m "determinism"

test-quick: ## Run quick smoke tests
	$(PYTHON) -m pytest tests/test_unit/ -x --tb=short -q \
		-m "not slow and not integration and not requires_hardware" \
		--timeout=10

test-coverage: ## Run tests with coverage report
	$(PYTHON) -m pytest tests/ -v \
		--cov=. --cov-config=pyproject.toml \
		--cov-report=term-missing \
		--cov-report=html:htmlcov \
		--cov-report=xml:coverage.xml \
		--cov-fail-under=$(COVERAGE_THRESHOLD) \
		-m "not slow and not integration and not requires_hardware"

test-parallel: ## Run tests in parallel
	$(PYTHON) -m pytest tests/ -n auto -v --tb=short

# =============================================================================
# PRE-COMMIT
# =============================================================================

pre-commit: ## Run pre-commit hooks on all files
	pre-commit run --all-files

pre-commit-update: ## Update pre-commit hooks
	pre-commit autoupdate

# =============================================================================
# QUALITY GATE
# =============================================================================

quality-check: ## Run full quality check (matches CI)
	$(PYTHON) scripts/quality_check.py

quality-check-quick: ## Run quick quality check
	$(PYTHON) scripts/quality_check.py --quick

quality-check-fix: ## Run quality check with auto-fix
	$(PYTHON) scripts/quality_check.py --fix

# Comprehensive quality gate
quality: format-check lint type-check test-coverage security ## Run all quality checks

# =============================================================================
# CLEANUP
# =============================================================================

clean: ## Clean build artifacts
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/
	rm -rf .ruff_cache/
	rm -rf htmlcov/
	rm -f coverage.xml
	rm -f coverage.json
	rm -f .coverage
	rm -f bandit-report.json
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true

clean-all: clean ## Clean all artifacts including virtual environments
	rm -rf .venv/
	rm -rf venv/

# =============================================================================
# DOCUMENTATION
# =============================================================================

docs: ## Build documentation
	$(PYTHON) -m mkdocs build

docs-serve: ## Serve documentation locally
	$(PYTHON) -m mkdocs serve

# =============================================================================
# DEVELOPMENT
# =============================================================================

dev: install-dev ## Set up development environment
	@echo "Development environment ready!"
	@echo "Run 'make help' to see available commands."

ci: quality ## Run CI pipeline locally

# =============================================================================
# COMBINED TARGETS
# =============================================================================

all: clean install-dev quality ## Clean, install, and run all quality checks
