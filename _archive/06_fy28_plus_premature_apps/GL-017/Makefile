# =============================================================================
# GL-017 CONDENSYNC - Makefile
# Build, test, lint, and deployment automation
# =============================================================================

# Configuration
SHELL := /bin/bash
.DEFAULT_GOAL := help

# Application Configuration
APP_NAME := gl-017-condensync
AGENT_ID := GL-017
VERSION := 1.0.0

# Docker Configuration
DOCKER_REGISTRY := registry.greenlang.io
DOCKER_IMAGE := $(DOCKER_REGISTRY)/greenlang/$(APP_NAME)
DOCKER_TAG := $(VERSION)
DOCKERFILE := Dockerfile

# Kubernetes Configuration
KUBE_NAMESPACE := greenlang
KUBE_CONTEXT := greenlang-cluster
DEPLOYMENT_DIR := deployment

# Python Configuration
PYTHON := python3
VENV := .venv
PIP := $(VENV)/bin/pip
PYTEST := $(VENV)/bin/pytest

# Colors for output
RED := \033[0;31m
GREEN := \033[0;32m
YELLOW := \033[0;33m
BLUE := \033[0;34m
NC := \033[0m

# =============================================================================
# Help Target
# =============================================================================
.PHONY: help
help: ## Display this help message
	@echo ""
	@echo "$(BLUE)GL-017 CONDENSYNC - Available Targets$(NC)"
	@echo "========================================"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "$(GREEN)%-20s$(NC) %s\n", $$1, $$2}'
	@echo ""

# =============================================================================
# Development Environment
# =============================================================================
.PHONY: venv
venv: ## Create Python virtual environment
	@echo "$(BLUE)Creating virtual environment...$(NC)"
	$(PYTHON) -m venv $(VENV)
	$(PIP) install --upgrade pip setuptools wheel
	$(PIP) install -r requirements.txt
	@echo "$(GREEN)Virtual environment created successfully$(NC)"

.PHONY: install
install: venv ## Install all dependencies
	@echo "$(BLUE)Installing dependencies...$(NC)"
	$(PIP) install -r requirements.txt
	$(PIP) install -r requirements-dev.txt
	@echo "$(GREEN)Dependencies installed successfully$(NC)"

.PHONY: clean
clean: ## Clean build artifacts and caches
	@echo "$(BLUE)Cleaning build artifacts...$(NC)"
	rm -rf build/ dist/ *.egg-info/
	rm -rf .pytest_cache/ .mypy_cache/ .ruff_cache/
	rm -rf htmlcov/ .coverage coverage.xml
	rm -rf __pycache__ **/__pycache__
	rm -rf .venv/
	@echo "$(GREEN)Cleanup completed$(NC)"

# =============================================================================
# Code Quality
# =============================================================================
.PHONY: lint
lint: ## Run all linters
	@echo "$(BLUE)Running linters...$(NC)"
	$(VENV)/bin/ruff check .
	$(VENV)/bin/black --check .
	$(VENV)/bin/isort --check-only .
	@echo "$(GREEN)Linting passed$(NC)"

.PHONY: format
format: ## Format code with black and isort
	@echo "$(BLUE)Formatting code...$(NC)"
	$(VENV)/bin/black .
	$(VENV)/bin/isort .
	@echo "$(GREEN)Code formatted$(NC)"

.PHONY: typecheck
typecheck: ## Run type checking with mypy
	@echo "$(BLUE)Running type checks...$(NC)"
	$(VENV)/bin/mypy --ignore-missing-imports .
	@echo "$(GREEN)Type checking passed$(NC)"

.PHONY: security
security: ## Run security checks
	@echo "$(BLUE)Running security checks...$(NC)"
	$(VENV)/bin/safety check -r requirements.txt
	$(VENV)/bin/bandit -r . -x tests/
	@echo "$(GREEN)Security checks passed$(NC)"

# =============================================================================
# Testing
# =============================================================================
.PHONY: test
test: ## Run all tests
	@echo "$(BLUE)Running tests...$(NC)"
	$(PYTEST) tests/ -v --tb=short
	@echo "$(GREEN)Tests passed$(NC)"

.PHONY: test-unit
test-unit: ## Run unit tests only
	@echo "$(BLUE)Running unit tests...$(NC)"
	$(PYTEST) tests/unit/ -v --tb=short
	@echo "$(GREEN)Unit tests passed$(NC)"

.PHONY: test-integration
test-integration: ## Run integration tests only
	@echo "$(BLUE)Running integration tests...$(NC)"
	$(PYTEST) tests/integration/ -v --tb=short
	@echo "$(GREEN)Integration tests passed$(NC)"

.PHONY: test-coverage
test-coverage: ## Run tests with coverage report
	@echo "$(BLUE)Running tests with coverage...$(NC)"
	$(PYTEST) tests/ --cov=. --cov-report=html --cov-report=term --cov-report=xml
	@echo "$(GREEN)Coverage report generated$(NC)"

# =============================================================================
# Docker Operations
# =============================================================================
.PHONY: docker-build
docker-build: ## Build Docker image
	@echo "$(BLUE)Building Docker image...$(NC)"
	docker build \
		--tag $(DOCKER_IMAGE):$(DOCKER_TAG) \
		--tag $(DOCKER_IMAGE):latest \
		--build-arg VERSION=$(VERSION) \
		--file $(DOCKERFILE) \
		.
	@echo "$(GREEN)Docker image built: $(DOCKER_IMAGE):$(DOCKER_TAG)$(NC)"

.PHONY: docker-push
docker-push: ## Push Docker image to registry
	@echo "$(BLUE)Pushing Docker image...$(NC)"
	docker push $(DOCKER_IMAGE):$(DOCKER_TAG)
	docker push $(DOCKER_IMAGE):latest
	@echo "$(GREEN)Docker image pushed$(NC)"

.PHONY: docker-run
docker-run: ## Run Docker container locally
	@echo "$(BLUE)Running Docker container...$(NC)"
	docker run -d \
		--name $(APP_NAME) \
		--publish 8017:8017 \
		--publish 9017:9017 \
		--env ENVIRONMENT=development \
		$(DOCKER_IMAGE):$(DOCKER_TAG)
	@echo "$(GREEN)Container started: $(APP_NAME)$(NC)"

.PHONY: docker-stop
docker-stop: ## Stop and remove Docker container
	@echo "$(BLUE)Stopping Docker container...$(NC)"
	docker stop $(APP_NAME) || true
	docker rm $(APP_NAME) || true
	@echo "$(GREEN)Container stopped$(NC)"

.PHONY: docker-logs
docker-logs: ## View Docker container logs
	docker logs -f $(APP_NAME)

.PHONY: docker-shell
docker-shell: ## Open shell in running container
	docker exec -it $(APP_NAME) /bin/bash

.PHONY: docker-scan
docker-scan: ## Scan Docker image for vulnerabilities
	@echo "$(BLUE)Scanning Docker image...$(NC)"
	trivy image $(DOCKER_IMAGE):$(DOCKER_TAG)
	@echo "$(GREEN)Scan completed$(NC)"

# =============================================================================
# Docker Compose Operations
# =============================================================================
.PHONY: compose-up
compose-up: ## Start all services with docker-compose
	@echo "$(BLUE)Starting services...$(NC)"
	docker-compose up -d
	@echo "$(GREEN)Services started$(NC)"

.PHONY: compose-down
compose-down: ## Stop all services with docker-compose
	@echo "$(BLUE)Stopping services...$(NC)"
	docker-compose down -v
	@echo "$(GREEN)Services stopped$(NC)"

.PHONY: compose-logs
compose-logs: ## View docker-compose logs
	docker-compose logs -f

.PHONY: compose-ps
compose-ps: ## List docker-compose services
	docker-compose ps

# =============================================================================
# Kubernetes Operations
# =============================================================================
.PHONY: k8s-apply
k8s-apply: ## Apply Kubernetes manifests
	@echo "$(BLUE)Applying Kubernetes manifests...$(NC)"
	kubectl apply -k $(DEPLOYMENT_DIR)/ -n $(KUBE_NAMESPACE)
	@echo "$(GREEN)Manifests applied$(NC)"

.PHONY: k8s-delete
k8s-delete: ## Delete Kubernetes resources
	@echo "$(YELLOW)Deleting Kubernetes resources...$(NC)"
	kubectl delete -k $(DEPLOYMENT_DIR)/ -n $(KUBE_NAMESPACE) --ignore-not-found
	@echo "$(GREEN)Resources deleted$(NC)"

.PHONY: k8s-status
k8s-status: ## Check Kubernetes deployment status
	@echo "$(BLUE)Deployment Status:$(NC)"
	kubectl get deployment $(APP_NAME) -n $(KUBE_NAMESPACE)
	@echo ""
	@echo "$(BLUE)Pod Status:$(NC)"
	kubectl get pods -l app=$(APP_NAME) -n $(KUBE_NAMESPACE)
	@echo ""
	@echo "$(BLUE)Service Status:$(NC)"
	kubectl get svc -l app=$(APP_NAME) -n $(KUBE_NAMESPACE)

.PHONY: k8s-logs
k8s-logs: ## View Kubernetes pod logs
	kubectl logs -l app=$(APP_NAME) -n $(KUBE_NAMESPACE) -f --tail=100

.PHONY: k8s-rollout
k8s-rollout: ## Trigger rolling update
	@echo "$(BLUE)Triggering rolling update...$(NC)"
	kubectl rollout restart deployment/$(APP_NAME) -n $(KUBE_NAMESPACE)
	kubectl rollout status deployment/$(APP_NAME) -n $(KUBE_NAMESPACE)
	@echo "$(GREEN)Rolling update completed$(NC)"

.PHONY: k8s-rollback
k8s-rollback: ## Rollback to previous deployment
	@echo "$(YELLOW)Rolling back deployment...$(NC)"
	kubectl rollout undo deployment/$(APP_NAME) -n $(KUBE_NAMESPACE)
	kubectl rollout status deployment/$(APP_NAME) -n $(KUBE_NAMESPACE)
	@echo "$(GREEN)Rollback completed$(NC)"

.PHONY: k8s-scale
k8s-scale: ## Scale deployment (usage: make k8s-scale REPLICAS=5)
	@echo "$(BLUE)Scaling deployment to $(REPLICAS) replicas...$(NC)"
	kubectl scale deployment/$(APP_NAME) --replicas=$(REPLICAS) -n $(KUBE_NAMESPACE)
	@echo "$(GREEN)Scaling completed$(NC)"

.PHONY: k8s-exec
k8s-exec: ## Execute shell in pod
	kubectl exec -it $$(kubectl get pod -l app=$(APP_NAME) -n $(KUBE_NAMESPACE) -o jsonpath='{.items[0].metadata.name}') -n $(KUBE_NAMESPACE) -- /bin/bash

# =============================================================================
# CI/CD Helpers
# =============================================================================
.PHONY: ci-validate
ci-validate: lint typecheck test ## Run all CI validations
	@echo "$(GREEN)All CI validations passed$(NC)"

.PHONY: ci-build
ci-build: docker-build docker-scan ## Build and scan Docker image
	@echo "$(GREEN)CI build completed$(NC)"

.PHONY: ci-deploy
ci-deploy: docker-push k8s-apply k8s-status ## Deploy to Kubernetes
	@echo "$(GREEN)CI deployment completed$(NC)"

# =============================================================================
# Version Information
# =============================================================================
.PHONY: version
version: ## Display version information
	@echo "$(BLUE)GL-017 CONDENSYNC$(NC)"
	@echo "Version: $(VERSION)"
	@echo "Image:   $(DOCKER_IMAGE):$(DOCKER_TAG)"
