# GreenLang Agent Factory - Makefile
# Production-grade CI/CD automation targets
# Usage: make <target>

.PHONY: help install lint format test coverage build deploy-staging deploy-prod certify golden-tests clean \
	db-upgrade db-downgrade db-downgrade-all db-revision db-revision-empty db-current db-history \
	db-heads db-show db-stamp db-check db-branches db-merge

# Default target
.DEFAULT_GOAL := help

# =============================================================================
# Variables
# =============================================================================
PYTHON := python3.11
PIP := $(PYTHON) -m pip
PYTEST := $(PYTHON) -m pytest
DOCKER := docker
KUBECTL := kubectl
HELM := helm

# Project settings
PROJECT_NAME := greenlang-agent-factory
VERSION := $(shell cat VERSION 2>/dev/null || echo "0.1.0")
GIT_SHA := $(shell git rev-parse --short HEAD 2>/dev/null || echo "unknown")
BUILD_DATE := $(shell date -u +%Y-%m-%dT%H:%M:%SZ)

# Docker settings
REGISTRY := ghcr.io
IMAGE_NAME := $(REGISTRY)/greenlang/$(PROJECT_NAME)
IMAGE_TAG := $(VERSION)-$(GIT_SHA)

# Kubernetes settings
KUBE_NAMESPACE_STAGING := greenlang-staging
KUBE_NAMESPACE_PROD := greenlang-production

# Test settings
COVERAGE_THRESHOLD := 85
GOLDEN_TESTS_DIR := tests/golden

# =============================================================================
# Help
# =============================================================================
help: ## Show this help message
	@echo "GreenLang Agent Factory - CI/CD Automation"
	@echo ""
	@echo "Usage: make <target>"
	@echo ""
	@echo "Targets:"
	@awk 'BEGIN {FS = ":.*##"; printf ""} /^[a-zA-Z_-]+:.*?##/ { printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2 }' $(MAKEFILE_LIST)

# =============================================================================
# Development Setup
# =============================================================================
install: ## Install all dependencies
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt
	$(PIP) install -r requirements-dev.txt
	$(PIP) install -e ".[dev,test]" || $(PIP) install -e .
	pre-commit install

install-dev: install ## Install development dependencies (alias for install)

install-prod: ## Install production dependencies only
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt
	$(PIP) install -e .

setup-hooks: ## Set up pre-commit hooks
	pre-commit install
	pre-commit install --hook-type commit-msg

# =============================================================================
# Code Quality
# =============================================================================
lint: ## Run all linters
	@echo "Running Ruff linter..."
	ruff check . --fix
	@echo ""
	@echo "Running Black formatter check..."
	black --check --diff .
	@echo ""
	@echo "Running isort check..."
	isort --check-only --diff .
	@echo ""
	@echo "Running mypy type checker..."
	mypy . --ignore-missing-imports

lint-ci: ## Run linters in CI mode (no fixes, exit on error)
	ruff check . --output-format=github
	black --check .
	isort --check-only .
	mypy . --ignore-missing-imports --no-error-summary

format: ## Format code with Black and isort
	@echo "Formatting with Black..."
	black .
	@echo ""
	@echo "Sorting imports with isort..."
	isort .
	@echo ""
	@echo "Fixing linting issues with Ruff..."
	ruff check . --fix

# =============================================================================
# Testing
# =============================================================================
test: ## Run all tests
	$(PYTEST) tests/ \
		-v \
		--tb=short \
		-n auto \
		--ignore=tests/integration \
		--ignore=tests/e2e \
		--ignore=tests/golden

test-unit: ## Run unit tests only
	$(PYTEST) tests/unit/ \
		-v \
		--tb=short \
		-n auto

test-integration: ## Run integration tests
	$(PYTEST) tests/integration/ \
		-v \
		--tb=short

test-e2e: ## Run end-to-end tests
	$(PYTEST) tests/e2e/ \
		-v \
		--tb=short

test-golden: ## Run golden tests
	$(PYTEST) $(GOLDEN_TESTS_DIR)/ \
		-v \
		--tb=short

test-all: test test-integration test-e2e test-golden ## Run all test suites

coverage: ## Generate test coverage report
	$(PYTEST) tests/ \
		--cov=. \
		--cov-report=xml \
		--cov-report=html \
		--cov-report=term-missing \
		--cov-fail-under=$(COVERAGE_THRESHOLD) \
		-v \
		--ignore=tests/integration \
		--ignore=tests/e2e \
		--ignore=tests/golden
	@echo ""
	@echo "Coverage report generated in htmlcov/"

coverage-ci: ## Generate coverage for CI (XML output)
	$(PYTEST) tests/ \
		--cov=. \
		--cov-report=xml \
		--cov-report=term \
		--cov-fail-under=$(COVERAGE_THRESHOLD) \
		--ignore=tests/integration \
		--ignore=tests/e2e \
		--ignore=tests/golden

# =============================================================================
# Security
# =============================================================================
security: ## Run security scans
	@echo "Running Bandit security scan..."
	bandit -r . -ll -f txt
	@echo ""
	@echo "Running Safety dependency check..."
	safety check --full-report || true
	@echo ""
	@echo "Running pip-audit..."
	pip-audit --strict || true

security-ci: ## Run security scans for CI
	bandit -r . -f json -o bandit-report.json --severity-level medium || true
	safety check --json > safety-report.json || true
	pip-audit --format json > pip-audit-report.json || true

secrets-scan: ## Scan for secrets in code
	detect-secrets scan --all-files --baseline .secrets.baseline
	detect-secrets audit .secrets.baseline

# =============================================================================
# Docker
# =============================================================================
build: ## Build Docker image
	$(DOCKER) build \
		--tag $(IMAGE_NAME):$(IMAGE_TAG) \
		--tag $(IMAGE_NAME):latest \
		--build-arg VERSION=$(VERSION) \
		--build-arg GIT_SHA=$(GIT_SHA) \
		--build-arg BUILD_DATE=$(BUILD_DATE) \
		.

build-no-cache: ## Build Docker image without cache
	$(DOCKER) build \
		--no-cache \
		--tag $(IMAGE_NAME):$(IMAGE_TAG) \
		--tag $(IMAGE_NAME):latest \
		--build-arg VERSION=$(VERSION) \
		--build-arg GIT_SHA=$(GIT_SHA) \
		--build-arg BUILD_DATE=$(BUILD_DATE) \
		.

push: ## Push Docker image to registry
	$(DOCKER) push $(IMAGE_NAME):$(IMAGE_TAG)
	$(DOCKER) push $(IMAGE_NAME):latest

docker-run: ## Run Docker container locally
	$(DOCKER) run \
		--rm \
		-p 8000:8000 \
		-e ENVIRONMENT=development \
		$(IMAGE_NAME):latest

docker-shell: ## Open shell in Docker container
	$(DOCKER) run \
		--rm \
		-it \
		--entrypoint /bin/bash \
		$(IMAGE_NAME):latest

# =============================================================================
# Kubernetes Deployment
# =============================================================================
deploy-staging: build push ## Deploy to staging environment
	@echo "Deploying to staging..."
	$(KUBECTL) config use-context staging
	$(KUBECTL) set image deployment/greenlang-app \
		greenlang-app=$(IMAGE_NAME):$(IMAGE_TAG) \
		-n $(KUBE_NAMESPACE_STAGING)
	$(KUBECTL) rollout status deployment/greenlang-app \
		-n $(KUBE_NAMESPACE_STAGING) \
		--timeout=300s
	@echo "Staging deployment complete!"

deploy-prod: ## Deploy to production environment (requires confirmation)
	@echo "WARNING: You are about to deploy to PRODUCTION!"
	@echo "Image: $(IMAGE_NAME):$(IMAGE_TAG)"
	@read -p "Type 'yes' to continue: " confirm && [ "$$confirm" = "yes" ] || exit 1
	@echo "Deploying to production..."
	$(KUBECTL) config use-context production
	$(KUBECTL) set image deployment/greenlang-app \
		greenlang-app=$(IMAGE_NAME):$(IMAGE_TAG) \
		-n $(KUBE_NAMESPACE_PROD)
	$(KUBECTL) rollout status deployment/greenlang-app \
		-n $(KUBE_NAMESPACE_PROD) \
		--timeout=600s
	@echo "Production deployment complete!"

rollback-staging: ## Rollback staging deployment
	$(KUBECTL) config use-context staging
	$(KUBECTL) rollout undo deployment/greenlang-app \
		-n $(KUBE_NAMESPACE_STAGING)
	$(KUBECTL) rollout status deployment/greenlang-app \
		-n $(KUBE_NAMESPACE_STAGING)

rollback-prod: ## Rollback production deployment
	@echo "WARNING: Rolling back PRODUCTION deployment!"
	@read -p "Type 'yes' to continue: " confirm && [ "$$confirm" = "yes" ] || exit 1
	$(KUBECTL) config use-context production
	$(KUBECTL) rollout undo deployment/greenlang-app \
		-n $(KUBE_NAMESPACE_PROD)
	$(KUBECTL) rollout status deployment/greenlang-app \
		-n $(KUBE_NAMESPACE_PROD)

status-staging: ## Check staging deployment status
	$(KUBECTL) config use-context staging
	$(KUBECTL) get pods -n $(KUBE_NAMESPACE_STAGING) -l app=greenlang-app
	$(KUBECTL) get svc -n $(KUBE_NAMESPACE_STAGING)

status-prod: ## Check production deployment status
	$(KUBECTL) config use-context production
	$(KUBECTL) get pods -n $(KUBE_NAMESPACE_PROD) -l app=greenlang-app
	$(KUBECTL) get svc -n $(KUBE_NAMESPACE_PROD)

# =============================================================================
# Agent Certification
# =============================================================================
certify: ## Run 12-dimension agent certification
	@echo "Running Agent Certification..."
	$(PYTHON) -m greenlang.certification.runner \
		--all-dimensions \
		--output-format json \
		--output-file certification-report.json
	@echo ""
	@echo "Certification report saved to certification-report.json"

certify-agent: ## Certify specific agent (usage: make certify-agent AGENT=GL-001)
ifndef AGENT
	$(error AGENT is not set. Usage: make certify-agent AGENT=GL-001)
endif
	@echo "Certifying agent: $(AGENT)"
	$(PYTHON) -m greenlang.certification.runner \
		--agent-id $(AGENT) \
		--all-dimensions \
		--output-format json \
		--output-file certification-$(AGENT).json

certify-quick: ## Run quick certification (dimensions 1-3 only)
	$(PYTHON) -m greenlang.certification.runner \
		--dimensions 1,2,3 \
		--output-format text

# =============================================================================
# Golden Tests
# =============================================================================
golden-tests: ## Run golden tests with determinism check
	@echo "Running Golden Tests..."
	$(PYTEST) $(GOLDEN_TESTS_DIR)/ \
		-v \
		--tb=short \
		-x
	@echo ""
	@echo "Running determinism check (10 iterations)..."
	@for i in 1 2 3 4 5 6 7 8 9 10; do \
		echo "Run $$i/10..."; \
		$(PYTEST) $(GOLDEN_TESTS_DIR)/ -q --tb=no > /tmp/golden_run_$$i.txt; \
	done
	@echo "Comparing results..."
	@diff /tmp/golden_run_1.txt /tmp/golden_run_10.txt && echo "PASS: All runs identical" || echo "FAIL: Results varied"

golden-update-baseline: ## Update golden test baselines
	@echo "Updating golden test baselines..."
	$(PYTEST) $(GOLDEN_TESTS_DIR)/ \
		--update-baseline \
		-v

# =============================================================================
# Documentation
# =============================================================================
docs: ## Generate documentation
	$(PYTHON) -m sphinx.cmd.build -b html docs/ docs/_build/html

docs-serve: ## Serve documentation locally
	$(PYTHON) -m http.server 8080 --directory docs/_build/html

# =============================================================================
# Cleanup
# =============================================================================
clean: ## Clean build artifacts
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .eggs/
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/
	rm -rf .ruff_cache/
	rm -rf htmlcov/
	rm -rf .coverage
	rm -rf coverage.xml
	rm -rf *.json
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete

clean-docker: ## Clean Docker images
	$(DOCKER) rmi $(IMAGE_NAME):$(IMAGE_TAG) || true
	$(DOCKER) rmi $(IMAGE_NAME):latest || true
	$(DOCKER) image prune -f

clean-all: clean clean-docker ## Clean everything

# =============================================================================
# Release
# =============================================================================
release-patch: ## Create patch release (0.0.x)
	bump2version patch
	git push --follow-tags

release-minor: ## Create minor release (0.x.0)
	bump2version minor
	git push --follow-tags

release-major: ## Create major release (x.0.0)
	bump2version major
	git push --follow-tags

changelog: ## Generate changelog
	git-changelog -o CHANGELOG.md

# =============================================================================
# Development Utilities
# =============================================================================
run: ## Run application locally
	uvicorn main:app --host 0.0.0.0 --port 8000 --reload

shell: ## Open Python shell with project context
	$(PYTHON) -i -c "from greenlang import *; print('GreenLang Agent Factory Shell')"

check: lint test security ## Run all checks (lint, test, security)

ci: lint-ci coverage-ci security-ci ## Run full CI pipeline locally

pre-commit: ## Run pre-commit on all files
	pre-commit run --all-files

update-deps: ## Update all dependencies
	$(PIP) install --upgrade pip
	$(PIP) install pip-tools
	pip-compile requirements.in --upgrade
	pip-compile requirements-dev.in --upgrade
	$(PIP) install -r requirements.txt -r requirements-dev.txt

# =============================================================================
# Database Migrations (Alembic)
# =============================================================================
ALEMBIC := alembic
BACKEND_DIR := backend

db-upgrade: ## Run all pending database migrations
	cd $(BACKEND_DIR) && $(ALEMBIC) upgrade head

db-downgrade: ## Rollback last database migration
	cd $(BACKEND_DIR) && $(ALEMBIC) downgrade -1

db-downgrade-all: ## Rollback all database migrations (DANGEROUS)
	@echo "WARNING: This will rollback ALL migrations!"
	@read -p "Type 'yes' to continue: " confirm && [ "$$confirm" = "yes" ] || exit 1
	cd $(BACKEND_DIR) && $(ALEMBIC) downgrade base

db-revision: ## Create new migration (usage: make db-revision MSG="description")
ifndef MSG
	$(error MSG is not set. Usage: make db-revision MSG="add user table")
endif
	cd $(BACKEND_DIR) && $(ALEMBIC) revision --autogenerate -m "$(MSG)"

db-revision-empty: ## Create empty migration (usage: make db-revision-empty MSG="description")
ifndef MSG
	$(error MSG is not set. Usage: make db-revision-empty MSG="add custom function")
endif
	cd $(BACKEND_DIR) && $(ALEMBIC) revision -m "$(MSG)"

db-current: ## Show current database revision
	cd $(BACKEND_DIR) && $(ALEMBIC) current

db-history: ## Show migration history
	cd $(BACKEND_DIR) && $(ALEMBIC) history --verbose

db-heads: ## Show current head revisions
	cd $(BACKEND_DIR) && $(ALEMBIC) heads

db-show: ## Show migration SQL without executing (usage: make db-show REV=head)
ifndef REV
	$(eval REV := head)
endif
	cd $(BACKEND_DIR) && $(ALEMBIC) upgrade $(REV) --sql

db-stamp: ## Stamp database with revision without running migrations (usage: make db-stamp REV=head)
ifndef REV
	$(error REV is not set. Usage: make db-stamp REV=head)
endif
	cd $(BACKEND_DIR) && $(ALEMBIC) stamp $(REV)

db-check: ## Check if database is up to date with migrations
	@cd $(BACKEND_DIR) && $(ALEMBIC) check && echo "Database is up to date" || echo "Pending migrations exist"

db-branches: ## Show branch points in migration history
	cd $(BACKEND_DIR) && $(ALEMBIC) branches

db-merge: ## Merge migration branches (usage: make db-merge REVS="rev1 rev2" MSG="merge branches")
ifndef REVS
	$(error REVS is not set. Usage: make db-merge REVS="rev1 rev2" MSG="merge branches")
endif
ifndef MSG
	$(eval MSG := merge migration branches)
endif
	cd $(BACKEND_DIR) && $(ALEMBIC) merge $(REVS) -m "$(MSG)"

# =============================================================================
# Infrastructure
# =============================================================================
tf-init: ## Initialize Terraform
	cd terraform && terraform init

tf-plan: ## Plan Terraform changes
	cd terraform && terraform plan -out=tfplan

tf-apply: ## Apply Terraform changes
	cd terraform && terraform apply tfplan

tf-destroy: ## Destroy Terraform infrastructure (DANGEROUS)
	@echo "WARNING: This will DESTROY all infrastructure!"
	@read -p "Type 'destroy' to continue: " confirm && [ "$$confirm" = "destroy" ] || exit 1
	cd terraform && terraform destroy
