# ============================================================================
# GL-CBAM-APP - Makefile
# ============================================================================
# Convenience commands for development, testing, and deployment
# ============================================================================

.PHONY: help build test deploy clean

# Default target
.DEFAULT_GOAL := help

# Variables
DOCKER_IMAGE := gl-cbam-app
DOCKER_TAG := latest
REGISTRY := ghcr.io/greenlang
KUBE_NAMESPACE := gl-cbam

# ============================================================================
# HELP
# ============================================================================

help: ## Show this help message
	@echo "GL-CBAM-APP - Available Commands"
	@echo "================================="
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

# ============================================================================
# DEVELOPMENT
# ============================================================================

install: ## Install dependencies
	pip install --upgrade pip
	pip install -r requirements.txt

install-dev: ## Install development dependencies
	pip install --upgrade pip
	pip install -r requirements.txt
	pip install pytest pytest-cov ruff bandit mypy safety

test: ## Run tests
	pytest tests/ -v --cov=. --cov-report=term --cov-report=html

test-coverage: ## Run tests with coverage report
	pytest tests/ -v --cov=. --cov-report=term --cov-report=html --cov-report=xml
	@echo "Coverage report: htmlcov/index.html"

lint: ## Run linter
	ruff check .

format: ## Format code
	ruff format .

type-check: ## Run type checking
	mypy . --ignore-missing-imports

security-scan: ## Run security scan
	bandit -r . -f json -o bandit-report.json
	safety check

quality: lint type-check security-scan ## Run all quality checks

# ============================================================================
# DOCKER
# ============================================================================

docker-build: ## Build Docker image
	docker build -t $(DOCKER_IMAGE):$(DOCKER_TAG) .

docker-build-no-cache: ## Build Docker image without cache
	docker build --no-cache -t $(DOCKER_IMAGE):$(DOCKER_TAG) .

docker-run: ## Run Docker container
	docker run -d -p 8000:8000 \
		-e DATABASE_URL=postgresql://cbam_user:password@host.docker.internal:5432/cbam_db \
		--name cbam-api $(DOCKER_IMAGE):$(DOCKER_TAG)

docker-stop: ## Stop Docker container
	docker stop cbam-api || true
	docker rm cbam-api || true

docker-logs: ## View Docker logs
	docker logs -f cbam-api

docker-shell: ## Open shell in container
	docker exec -it cbam-api /bin/bash

docker-clean: ## Remove Docker images
	docker rmi $(DOCKER_IMAGE):$(DOCKER_TAG) || true
	docker system prune -f

# ============================================================================
# DOCKER COMPOSE
# ============================================================================

up: ## Start all services with Docker Compose
	docker-compose up -d

down: ## Stop all services
	docker-compose down

restart: ## Restart all services
	docker-compose restart

logs: ## View logs
	docker-compose logs -f

logs-api: ## View API logs
	docker-compose logs -f api

ps: ## Show running containers
	docker-compose ps

build: ## Build services
	docker-compose build

rebuild: ## Rebuild services without cache
	docker-compose build --no-cache

shell: ## Open shell in API container
	docker-compose exec api /bin/bash

db-shell: ## Open PostgreSQL shell
	docker-compose exec postgres psql -U cbam_user -d cbam_db

redis-cli: ## Open Redis CLI
	docker-compose exec redis redis-cli

# ============================================================================
# KUBERNETES
# ============================================================================

k8s-create-namespace: ## Create Kubernetes namespace
	kubectl create namespace $(KUBE_NAMESPACE) || true

k8s-apply: ## Apply all Kubernetes manifests
	kubectl apply -f k8s/configmap.yaml
	kubectl apply -f k8s/secrets.yaml
	kubectl apply -f k8s/deployment.yaml
	kubectl apply -f k8s/service.yaml
	kubectl apply -f k8s/ingress.yaml

k8s-delete: ## Delete all Kubernetes resources
	kubectl delete -f k8s/ingress.yaml || true
	kubectl delete -f k8s/service.yaml || true
	kubectl delete -f k8s/deployment.yaml || true
	kubectl delete -f k8s/secrets.yaml || true
	kubectl delete -f k8s/configmap.yaml || true

k8s-status: ## Show Kubernetes resources status
	kubectl get all -n $(KUBE_NAMESPACE)

k8s-pods: ## List pods
	kubectl get pods -n $(KUBE_NAMESPACE) -l app=cbam-api

k8s-logs: ## View pod logs
	kubectl logs -f deployment/cbam-api -n $(KUBE_NAMESPACE)

k8s-describe: ## Describe deployment
	kubectl describe deployment/cbam-api -n $(KUBE_NAMESPACE)

k8s-shell: ## Open shell in pod
	kubectl exec -it deployment/cbam-api -n $(KUBE_NAMESPACE) -- /bin/bash

k8s-scale: ## Scale deployment (use REPLICAS=N)
	kubectl scale deployment/cbam-api --replicas=$(REPLICAS) -n $(KUBE_NAMESPACE)

k8s-restart: ## Restart deployment
	kubectl rollout restart deployment/cbam-api -n $(KUBE_NAMESPACE)

k8s-rollback: ## Rollback deployment
	kubectl rollout undo deployment/cbam-api -n $(KUBE_NAMESPACE)

k8s-port-forward: ## Port forward to local (8000)
	kubectl port-forward svc/cbam-api 8000:8000 -n $(KUBE_NAMESPACE)

# ============================================================================
# DEPLOYMENT
# ============================================================================

deploy-staging: ## Deploy to staging
	@echo "Deploying to staging..."
	kubectl set image deployment/cbam-api \
		api=$(REGISTRY)/$(DOCKER_IMAGE):$(DOCKER_TAG) \
		-n gl-cbam-staging
	kubectl rollout status deployment/cbam-api -n gl-cbam-staging

deploy-production: ## Deploy to production
	@echo "Deploying to production..."
	@read -p "Are you sure? [y/N] " -n 1 -r; \
	echo; \
	if [[ $$REPLY =~ ^[Yy]$$ ]]; then \
		kubectl set image deployment/cbam-api \
			api=$(REGISTRY)/$(DOCKER_IMAGE):$(DOCKER_TAG) \
			-n $(KUBE_NAMESPACE); \
		kubectl rollout status deployment/cbam-api -n $(KUBE_NAMESPACE); \
	fi

# ============================================================================
# DATABASE
# ============================================================================

db-backup: ## Backup database
	@echo "Creating database backup..."
	docker-compose exec -T postgres pg_dump -U cbam_user cbam_db | gzip > backup-$$(date +%Y%m%d-%H%M%S).sql.gz
	@echo "Backup created!"

db-restore: ## Restore database (use BACKUP=file.sql.gz)
	@echo "Restoring database from $(BACKUP)..."
	gunzip < $(BACKUP) | docker-compose exec -T postgres psql -U cbam_user cbam_db
	@echo "Restore complete!"

db-migrate: ## Run database migrations
	docker-compose exec api alembic upgrade head

db-reset: ## Reset database (CAUTION!)
	docker-compose exec postgres psql -U cbam_user -d cbam_db -c "DROP SCHEMA public CASCADE; CREATE SCHEMA public;"
	$(MAKE) db-migrate

# ============================================================================
# TESTING
# ============================================================================

test-integration: ## Run integration tests
	pytest tests/test_pipeline_integration.py -v

test-unit: ## Run unit tests
	pytest tests/ -v -m "not integration"

test-smoke: ## Run smoke tests against deployed API
	curl -f http://localhost:8000/health || exit 1
	curl -f http://localhost:8000/docs || exit 1
	@echo "Smoke tests passed!"

# ============================================================================
# CLEANUP
# ============================================================================

clean: ## Clean build artifacts
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name "*.log" -delete
	rm -rf build/ dist/ *.egg-info htmlcov/ .pytest_cache/ .coverage

clean-all: clean docker-clean ## Clean everything
	rm -rf venv/ .venv/ env/

# ============================================================================
# CI/CD
# ============================================================================

ci: quality test ## Run CI checks (quality + tests)

cd: docker-build docker-push ## Build and push Docker image

docker-push: ## Push Docker image to registry
	docker tag $(DOCKER_IMAGE):$(DOCKER_TAG) $(REGISTRY)/$(DOCKER_IMAGE):$(DOCKER_TAG)
	docker push $(REGISTRY)/$(DOCKER_IMAGE):$(DOCKER_TAG)

# ============================================================================
# UTILITIES
# ============================================================================

health: ## Check API health
	@curl -s http://localhost:8000/health | python -m json.tool

version: ## Show application version
	@grep "version" pack.yaml | head -1

env-example: ## Create .env from example
	cp .env.production.example .env
	@echo ".env file created. Please update with your values."

secrets-generate: ## Generate random secrets
	@echo "SECRET_KEY=$$(openssl rand -base64 32)"
	@echo "JWT_SECRET_KEY=$$(openssl rand -base64 32)"
	@echo "POSTGRES_PASSWORD=$$(openssl rand -base64 24)"
	@echo "REDIS_PASSWORD=$$(openssl rand -base64 24)"

# ============================================================================
# MONITORING
# ============================================================================

stats: ## Show Docker stats
	docker stats --no-stream

k8s-metrics: ## Show Kubernetes metrics
	kubectl top pods -n $(KUBE_NAMESPACE)
	kubectl top nodes

# ============================================================================
# END OF MAKEFILE
# ============================================================================
