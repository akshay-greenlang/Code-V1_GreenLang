# GL-016 WATERGUARD - Makefile
# Project: GreenLang Industrial Sustainability Framework
# Agent: Water Treatment Quality Control

.PHONY: help build test lint format validate deploy clean

# Variables
IMAGE_NAME := gl-016-waterguard
IMAGE_TAG := $(shell git describe --tags --always --dirty 2>/dev/null || echo "dev")
REGISTRY := registry.greenlang.io/agents
FULL_IMAGE := $(REGISTRY)/$(IMAGE_NAME):$(IMAGE_TAG)
NAMESPACE := greenlang

# Colors for output
RED := \033[0;31m
GREEN := \033[0;32m
YELLOW := \033[0;33m
NC := \033[0m # No Color

help: ## Show this help message
	@echo "$(GREEN)GL-016 WATERGUARD - Available Commands$(NC)"
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  $(YELLOW)%-20s$(NC) %s\n", $$1, $$2}'
	@echo ""

# ============================================================================
# Development
# ============================================================================

install: ## Install dependencies
	@echo "$(GREEN)Installing dependencies...$(NC)"
	pip install -r requirements.txt
	pre-commit install

venv: ## Create virtual environment
	@echo "$(GREEN)Creating virtual environment...$(NC)"
	python3 -m venv .venv
	@echo "$(YELLOW)Activate with: source .venv/bin/activate$(NC)"

format: ## Format code with black and isort
	@echo "$(GREEN)Formatting code...$(NC)"
	black .
	isort .

lint: ## Lint code with flake8 and mypy
	@echo "$(GREEN)Linting code...$(NC)"
	flake8 .
	mypy .

security: ## Run security checks with bandit
	@echo "$(GREEN)Running security checks...$(NC)"
	bandit -r . -c pyproject.toml

# ============================================================================
# Testing
# ============================================================================

test: ## Run all tests
	@echo "$(GREEN)Running tests...$(NC)"
	pytest

test-unit: ## Run unit tests only
	@echo "$(GREEN)Running unit tests...$(NC)"
	pytest tests/unit -v

test-integration: ## Run integration tests
	@echo "$(GREEN)Running integration tests...$(NC)"
	pytest tests/integration -v

test-e2e: ## Run end-to-end tests
	@echo "$(GREEN)Running E2E tests...$(NC)"
	pytest tests/e2e -v

test-performance: ## Run performance tests
	@echo "$(GREEN)Running performance tests...$(NC)"
	pytest tests/performance -v

test-coverage: ## Run tests with coverage report
	@echo "$(GREEN)Running tests with coverage...$(NC)"
	pytest --cov=. --cov-report=html --cov-report=term

# ============================================================================
# Validation
# ============================================================================

validate: ## Validate all configuration files
	@echo "$(GREEN)Validating configuration...$(NC)"
	python validate_config.py

validate-k8s: ## Validate Kubernetes manifests
	@echo "$(GREEN)Validating Kubernetes manifests...$(NC)"
	kubectl apply --dry-run=client -f deployment/

validate-docker: ## Validate Dockerfile
	@echo "$(GREEN)Validating Dockerfile...$(NC)"
	docker build --check .

# ============================================================================
# Docker
# ============================================================================

build: ## Build Docker image
	@echo "$(GREEN)Building Docker image: $(FULL_IMAGE)$(NC)"
	docker build -t $(IMAGE_NAME):latest -t $(FULL_IMAGE) .

build-no-cache: ## Build Docker image without cache
	@echo "$(GREEN)Building Docker image (no cache): $(FULL_IMAGE)$(NC)"
	docker build --no-cache -t $(IMAGE_NAME):latest -t $(FULL_IMAGE) .

push: ## Push Docker image to registry
	@echo "$(GREEN)Pushing image: $(FULL_IMAGE)$(NC)"
	docker push $(FULL_IMAGE)

scan: ## Scan Docker image for vulnerabilities
	@echo "$(GREEN)Scanning image for vulnerabilities...$(NC)"
	trivy image $(FULL_IMAGE)

run: ## Run Docker container locally
	@echo "$(GREEN)Running container...$(NC)"
	docker run --rm -it -p 8000:8000 -p 9090:9090 \
		--env-file .env \
		$(IMAGE_NAME):latest

# ============================================================================
# Kubernetes
# ============================================================================

deploy: ## Deploy to Kubernetes
	@echo "$(GREEN)Deploying to Kubernetes...$(NC)"
	kubectl apply -f deployment/ -n $(NAMESPACE)

deploy-dev: ## Deploy to development environment
	@echo "$(GREEN)Deploying to development...$(NC)"
	kubectl apply -k deployment/overlays/dev

deploy-staging: ## Deploy to staging environment
	@echo "$(GREEN)Deploying to staging...$(NC)"
	kubectl apply -k deployment/overlays/staging

deploy-prod: ## Deploy to production environment
	@echo "$(YELLOW)Deploying to production...$(NC)"
	@read -p "Are you sure? [y/N] " -n 1 -r; \
	echo; \
	if [[ $$REPLY =~ ^[Yy]$$ ]]; then \
		kubectl apply -k deployment/overlays/production; \
	fi

undeploy: ## Remove deployment from Kubernetes
	@echo "$(RED)Removing deployment...$(NC)"
	kubectl delete -f deployment/ -n $(NAMESPACE)

status: ## Check deployment status
	@echo "$(GREEN)Checking deployment status...$(NC)"
	kubectl get pods,svc,ingress -n $(NAMESPACE) -l app=waterguard

logs: ## View pod logs
	@echo "$(GREEN)Viewing logs...$(NC)"
	kubectl logs -n $(NAMESPACE) -l app=waterguard --tail=100 -f

describe: ## Describe pods
	@echo "$(GREEN)Describing pods...$(NC)"
	kubectl describe pods -n $(NAMESPACE) -l app=waterguard

shell: ## Open shell in pod
	@echo "$(GREEN)Opening shell...$(NC)"
	kubectl exec -it -n $(NAMESPACE) $$(kubectl get pod -n $(NAMESPACE) -l app=waterguard -o jsonpath='{.items[0].metadata.name}') -- /bin/bash

port-forward: ## Port forward to local machine
	@echo "$(GREEN)Port forwarding 8000:8000...$(NC)"
	kubectl port-forward -n $(NAMESPACE) svc/gl-016-waterguard 8000:8000

scale: ## Scale deployment (usage: make scale REPLICAS=5)
	@echo "$(GREEN)Scaling to $(REPLICAS) replicas...$(NC)"
	kubectl scale deployment gl-016-waterguard --replicas=$(REPLICAS) -n $(NAMESPACE)

restart: ## Restart deployment
	@echo "$(GREEN)Restarting deployment...$(NC)"
	kubectl rollout restart deployment/gl-016-waterguard -n $(NAMESPACE)

rollback: ## Rollback deployment
	@echo "$(YELLOW)Rolling back deployment...$(NC)"
	kubectl rollout undo deployment/gl-016-waterguard -n $(NAMESPACE)

# ============================================================================
# CI/CD
# ============================================================================

ci: lint test validate ## Run CI pipeline
	@echo "$(GREEN)CI pipeline completed successfully$(NC)"

pre-commit: format lint test-unit validate ## Run pre-commit checks
	@echo "$(GREEN)Pre-commit checks passed$(NC)"

release: build push ## Build and push release
	@echo "$(GREEN)Release $(IMAGE_TAG) completed$(NC)"

# ============================================================================
# Cleanup
# ============================================================================

clean: ## Clean up build artifacts
	@echo "$(GREEN)Cleaning up...$(NC)"
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "htmlcov" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name ".coverage" -delete
	rm -rf build/ dist/

clean-docker: ## Remove Docker images
	@echo "$(RED)Removing Docker images...$(NC)"
	docker rmi $(IMAGE_NAME):latest $(FULL_IMAGE) 2>/dev/null || true

clean-all: clean clean-docker ## Clean everything
	@echo "$(GREEN)Cleanup complete$(NC)"

# ============================================================================
# Default target
# ============================================================================

.DEFAULT_GOAL := help
