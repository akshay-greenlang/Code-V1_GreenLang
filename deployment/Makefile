# =============================================================================
# GreenLang Platform - INFRA-001 Deployment Makefile
# =============================================================================
# Usage:
#   make init ENV=prod          # Initialize Terraform
#   make plan ENV=prod          # Plan infrastructure changes
#   make apply ENV=prod         # Apply infrastructure changes
#   make deploy-k8s ENV=prod    # Deploy to Kubernetes
#   make validate ENV=prod      # Run all validation
#   make destroy ENV=prod       # Teardown (use with caution!)
# =============================================================================

# Default environment
ENV ?= dev

# Configuration
SHELL := /bin/bash
.SHELLFLAGS := -eu -o pipefail -c
MAKEFLAGS += --warn-undefined-variables
MAKEFLAGS += --no-builtin-rules

# Directories
ROOT_DIR := $(shell pwd)
TERRAFORM_DIR := $(ROOT_DIR)/terraform/environments/$(ENV)
HELM_DIR := $(ROOT_DIR)/infrastructure/helm/greenlang
K8S_DIR := $(ROOT_DIR)/infrastructure/kubernetes/greenlang

# AWS Configuration
AWS_REGION ?= us-east-1
AWS_PROFILE ?= default

# Terraform Configuration
TF_STATE_BUCKET := greenlang-terraform-state
TF_LOCK_TABLE := greenlang-terraform-locks
TF_VAR_FILE := terraform.tfvars

# Kubernetes Configuration
K8S_NAMESPACE := greenlang
K8S_CONTEXT ?= greenlang-$(ENV)
KUBECONFIG_PATH := /tmp/kubeconfig-$(ENV)

# Helm Configuration
HELM_RELEASE_NAME := greenlang
HELM_VALUES_FILE := values-$(ENV).yaml

# Image Configuration
IMAGE_TAG ?= latest
IMAGE_REGISTRY ?= ghcr.io/greenlang

# Colors for output
RED := \033[0;31m
GREEN := \033[0;32m
YELLOW := \033[1;33m
BLUE := \033[0;34m
NC := \033[0m

# =============================================================================
# Help Target
# =============================================================================
.PHONY: help
help: ## Display this help message
	@echo ""
	@echo "GreenLang Platform - INFRA-001 Deployment"
	@echo "=========================================="
	@echo ""
	@echo "Usage: make <target> ENV=<environment>"
	@echo ""
	@echo "Environments: dev, staging, prod"
	@echo ""
	@echo "Targets:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  $(BLUE)%-20s$(NC) %s\n", $$1, $$2}'
	@echo ""

# =============================================================================
# Pre-flight Checks
# =============================================================================
.PHONY: check-env
check-env: ## Verify environment is set correctly
	@if [ "$(ENV)" != "dev" ] && [ "$(ENV)" != "staging" ] && [ "$(ENV)" != "prod" ]; then \
		echo "$(RED)Error: ENV must be dev, staging, or prod$(NC)"; \
		exit 1; \
	fi
	@echo "$(GREEN)Environment: $(ENV)$(NC)"

.PHONY: check-tools
check-tools: ## Verify required tools are installed
	@echo "$(BLUE)Checking required tools...$(NC)"
	@command -v aws >/dev/null 2>&1 || { echo "$(RED)aws CLI is required$(NC)"; exit 1; }
	@command -v terraform >/dev/null 2>&1 || { echo "$(RED)terraform is required$(NC)"; exit 1; }
	@command -v kubectl >/dev/null 2>&1 || { echo "$(RED)kubectl is required$(NC)"; exit 1; }
	@command -v helm >/dev/null 2>&1 || { echo "$(RED)helm is required$(NC)"; exit 1; }
	@command -v jq >/dev/null 2>&1 || { echo "$(RED)jq is required$(NC)"; exit 1; }
	@echo "$(GREEN)All required tools are installed$(NC)"

.PHONY: check-aws
check-aws: ## Verify AWS credentials and account
	@echo "$(BLUE)Verifying AWS credentials...$(NC)"
	@aws sts get-caller-identity --output table
	@echo "$(GREEN)AWS credentials verified$(NC)"

.PHONY: preflight
preflight: check-env check-tools check-aws ## Run all pre-flight checks
	@echo ""
	@echo "$(GREEN)All pre-flight checks passed!$(NC)"
	@echo ""

# =============================================================================
# Terraform Targets
# =============================================================================
.PHONY: init
init: preflight ## Initialize Terraform backend and workspace
	@echo ""
	@echo "$(BLUE)========================================$(NC)"
	@echo "$(BLUE)Initializing Terraform for $(ENV)$(NC)"
	@echo "$(BLUE)========================================$(NC)"
	@echo ""
	@cd $(TERRAFORM_DIR) && \
		terraform init \
			-backend-config="bucket=$(TF_STATE_BUCKET)" \
			-backend-config="key=environments/$(ENV)/terraform.tfstate" \
			-backend-config="region=$(AWS_REGION)" \
			-backend-config="encrypt=true" \
			-backend-config="dynamodb_table=$(TF_LOCK_TABLE)" \
			-reconfigure
	@cd $(TERRAFORM_DIR) && \
		terraform workspace select $(ENV) 2>/dev/null || terraform workspace new $(ENV)
	@echo ""
	@echo "$(GREEN)Terraform initialized successfully!$(NC)"

.PHONY: validate-tf
validate-tf: ## Validate Terraform configuration
	@echo "$(BLUE)Validating Terraform configuration...$(NC)"
	@cd $(TERRAFORM_DIR) && terraform validate
	@echo "$(GREEN)Terraform configuration is valid$(NC)"

.PHONY: fmt
fmt: ## Format Terraform files
	@echo "$(BLUE)Formatting Terraform files...$(NC)"
	@cd $(ROOT_DIR)/terraform && terraform fmt -recursive
	@echo "$(GREEN)Terraform files formatted$(NC)"

.PHONY: plan
plan: init validate-tf ## Generate and show Terraform execution plan
	@echo ""
	@echo "$(BLUE)========================================$(NC)"
	@echo "$(BLUE)Planning infrastructure changes for $(ENV)$(NC)"
	@echo "$(BLUE)========================================$(NC)"
	@echo ""
	@cd $(TERRAFORM_DIR) && \
		terraform plan \
			-var-file="$(TF_VAR_FILE)" \
			-out=tfplan.binary \
			-detailed-exitcode || true
	@cd $(TERRAFORM_DIR) && \
		terraform show -json tfplan.binary > tfplan.json
	@echo ""
	@echo "$(YELLOW)Plan Summary:$(NC)"
	@cd $(TERRAFORM_DIR) && \
		jq -r '.resource_changes | group_by(.change.actions[0]) | map({action: .[0].change.actions[0], count: length}) | .[] | "\(.action): \(.count)"' tfplan.json
	@echo ""
	@echo "$(GREEN)Plan generated: $(TERRAFORM_DIR)/tfplan.binary$(NC)"

.PHONY: apply
apply: plan ## Apply Terraform changes
	@echo ""
	@echo "$(BLUE)========================================$(NC)"
	@echo "$(BLUE)Applying infrastructure changes for $(ENV)$(NC)"
	@echo "$(BLUE)========================================$(NC)"
	@echo ""
ifeq ($(ENV),prod)
	@echo "$(YELLOW)WARNING: You are about to apply changes to PRODUCTION$(NC)"
	@read -p "Type 'yes' to continue: " confirm && [ "$$confirm" = "yes" ] || exit 1
endif
	@cd $(TERRAFORM_DIR) && \
		terraform apply tfplan.binary
	@echo ""
	@echo "$(GREEN)Infrastructure changes applied successfully!$(NC)"

.PHONY: apply-target
apply-target: init ## Apply specific Terraform module (usage: make apply-target TARGET=module.vpc)
	@if [ -z "$(TARGET)" ]; then \
		echo "$(RED)Error: TARGET is required$(NC)"; \
		echo "Usage: make apply-target TARGET=module.vpc ENV=prod"; \
		exit 1; \
	fi
	@echo "$(BLUE)Applying target: $(TARGET)$(NC)"
	@cd $(TERRAFORM_DIR) && \
		terraform apply \
			-target=$(TARGET) \
			-var-file="$(TF_VAR_FILE)" \
			-auto-approve

.PHONY: output
output: ## Show Terraform outputs
	@cd $(TERRAFORM_DIR) && terraform output -json | jq

# =============================================================================
# Kubernetes / Helm Targets
# =============================================================================
.PHONY: kubeconfig
kubeconfig: ## Update kubeconfig for EKS cluster
	@echo "$(BLUE)Updating kubeconfig for $(ENV)...$(NC)"
	@aws eks update-kubeconfig \
		--name greenlang-$(ENV)-eks \
		--region $(AWS_REGION) \
		--kubeconfig $(KUBECONFIG_PATH) \
		--alias greenlang-$(ENV)
	@export KUBECONFIG=$(KUBECONFIG_PATH) && kubectl cluster-info
	@echo "$(GREEN)Kubeconfig updated: $(KUBECONFIG_PATH)$(NC)"

.PHONY: create-namespaces
create-namespaces: kubeconfig ## Create Kubernetes namespaces
	@echo "$(BLUE)Creating namespaces...$(NC)"
	@export KUBECONFIG=$(KUBECONFIG_PATH) && \
		kubectl apply -f $(K8S_DIR)/base/namespace.yaml
	@echo "$(GREEN)Namespaces created$(NC)"

.PHONY: deploy-addons
deploy-addons: kubeconfig ## Deploy Kubernetes add-ons (cert-manager, ingress, etc.)
	@echo "$(BLUE)Deploying Kubernetes add-ons...$(NC)"
	@export KUBECONFIG=$(KUBECONFIG_PATH) && \
		helm repo add jetstack https://charts.jetstack.io && \
		helm repo add ingress-nginx https://kubernetes.github.io/ingress-nginx && \
		helm repo add external-secrets https://charts.external-secrets.io && \
		helm repo add autoscaler https://kubernetes.github.io/autoscaler && \
		helm repo update
	@echo ""
	@echo "$(YELLOW)Installing cert-manager...$(NC)"
	@export KUBECONFIG=$(KUBECONFIG_PATH) && \
		helm upgrade --install cert-manager jetstack/cert-manager \
			--namespace cert-manager --create-namespace \
			--set installCRDs=true --wait --timeout 10m
	@echo ""
	@echo "$(YELLOW)Installing ingress-nginx...$(NC)"
	@export KUBECONFIG=$(KUBECONFIG_PATH) && \
		helm upgrade --install ingress-nginx ingress-nginx/ingress-nginx \
			--namespace ingress-nginx --create-namespace \
			--set controller.service.type=LoadBalancer \
			--set controller.metrics.enabled=true \
			--set controller.autoscaling.enabled=true \
			--set controller.autoscaling.minReplicas=2 \
			--wait --timeout 10m
	@echo ""
	@echo "$(YELLOW)Installing external-secrets...$(NC)"
	@export KUBECONFIG=$(KUBECONFIG_PATH) && \
		helm upgrade --install external-secrets external-secrets/external-secrets \
			--namespace external-secrets --create-namespace \
			--wait --timeout 10m
	@echo ""
	@echo "$(GREEN)Add-ons deployed successfully!$(NC)"

.PHONY: deploy-monitoring
deploy-monitoring: kubeconfig ## Deploy monitoring stack (Prometheus, Grafana)
	@echo "$(BLUE)Deploying monitoring stack...$(NC)"
	@export KUBECONFIG=$(KUBECONFIG_PATH) && \
		helm repo add prometheus-community https://prometheus-community.github.io/helm-charts && \
		helm repo update
	@export KUBECONFIG=$(KUBECONFIG_PATH) && \
		helm upgrade --install kube-prometheus-stack prometheus-community/kube-prometheus-stack \
			--namespace monitoring --create-namespace \
			--set prometheus.prometheusSpec.retention=30d \
			--set grafana.enabled=true \
			--set grafana.persistence.enabled=true \
			--wait --timeout 15m
	@echo ""
	@echo "$(YELLOW)Deploying custom alerts...$(NC)"
	@export KUBECONFIG=$(KUBECONFIG_PATH) && \
		kubectl apply -f $(ROOT_DIR)/monitoring/alerts-unified.yml -n monitoring || true
	@echo ""
	@echo "$(GREEN)Monitoring stack deployed successfully!$(NC)"

.PHONY: deploy-k8s
deploy-k8s: kubeconfig create-namespaces ## Deploy GreenLang application to Kubernetes
	@echo ""
	@echo "$(BLUE)========================================$(NC)"
	@echo "$(BLUE)Deploying GreenLang to $(ENV)$(NC)"
	@echo "$(BLUE)========================================$(NC)"
	@echo ""
ifeq ($(ENV),prod)
	@echo "$(YELLOW)WARNING: You are deploying to PRODUCTION$(NC)"
	@read -p "Type 'yes' to continue: " confirm && [ "$$confirm" = "yes" ] || exit 1
endif
	@echo "$(YELLOW)Installing/upgrading GreenLang Helm release...$(NC)"
	@export KUBECONFIG=$(KUBECONFIG_PATH) && \
		helm upgrade --install $(HELM_RELEASE_NAME) $(HELM_DIR) \
			--namespace $(K8S_NAMESPACE) \
			-f $(HELM_DIR)/$(HELM_VALUES_FILE) \
			--set global.environment=$(ENV) \
			--set image.tag=$(IMAGE_TAG) \
			--wait --timeout 15m
	@echo ""
	@echo "$(GREEN)GreenLang deployed successfully!$(NC)"
	@echo ""
	@export KUBECONFIG=$(KUBECONFIG_PATH) && \
		kubectl get pods -n $(K8S_NAMESPACE)

.PHONY: rollback-k8s
rollback-k8s: kubeconfig ## Rollback Helm release to previous version
	@echo "$(YELLOW)Rolling back $(HELM_RELEASE_NAME) to previous version...$(NC)"
	@export KUBECONFIG=$(KUBECONFIG_PATH) && \
		helm rollback $(HELM_RELEASE_NAME) -n $(K8S_NAMESPACE) --wait
	@echo "$(GREEN)Rollback completed$(NC)"

.PHONY: helm-history
helm-history: kubeconfig ## Show Helm release history
	@export KUBECONFIG=$(KUBECONFIG_PATH) && \
		helm history $(HELM_RELEASE_NAME) -n $(K8S_NAMESPACE)

# =============================================================================
# Validation Targets
# =============================================================================
.PHONY: validate
validate: validate-infra validate-k8s validate-app ## Run all validation checks
	@echo ""
	@echo "$(GREEN)========================================$(NC)"
	@echo "$(GREEN)All validations passed for $(ENV)!$(NC)"
	@echo "$(GREEN)========================================$(NC)"
	@echo ""

.PHONY: validate-infra
validate-infra: ## Validate infrastructure
	@echo ""
	@echo "$(BLUE)Validating Infrastructure...$(NC)"
	@echo ""
	@echo "Checking VPC..."
	@VPC_ID=$$(cd $(TERRAFORM_DIR) && terraform output -raw vpc_id 2>/dev/null) && \
		aws ec2 describe-vpcs --vpc-ids $$VPC_ID --query 'Vpcs[0].State' --output text | \
		grep -q "available" && echo "$(GREEN)  VPC: available$(NC)" || echo "$(RED)  VPC: not available$(NC)"
	@echo ""
	@echo "Checking EKS..."
	@CLUSTER_NAME=$$(cd $(TERRAFORM_DIR) && terraform output -raw eks_cluster_name 2>/dev/null) && \
		aws eks describe-cluster --name $$CLUSTER_NAME --query 'cluster.status' --output text | \
		grep -q "ACTIVE" && echo "$(GREEN)  EKS: ACTIVE$(NC)" || echo "$(RED)  EKS: not active$(NC)"
	@echo ""
	@echo "Checking RDS..."
	@aws rds describe-db-instances \
		--db-instance-identifier greenlang-$(ENV)-postgres \
		--query 'DBInstances[0].DBInstanceStatus' --output text 2>/dev/null | \
		grep -q "available" && echo "$(GREEN)  RDS: available$(NC)" || echo "$(YELLOW)  RDS: not available or not found$(NC)"
	@echo ""
	@echo "Checking ElastiCache..."
	@aws elasticache describe-replication-groups \
		--replication-group-id greenlang-$(ENV)-redis \
		--query 'ReplicationGroups[0].Status' --output text 2>/dev/null | \
		grep -q "available" && echo "$(GREEN)  ElastiCache: available$(NC)" || echo "$(YELLOW)  ElastiCache: not available or not found$(NC)"

.PHONY: validate-k8s
validate-k8s: kubeconfig ## Validate Kubernetes deployment
	@echo ""
	@echo "$(BLUE)Validating Kubernetes Deployment...$(NC)"
	@echo ""
	@echo "Checking nodes..."
	@export KUBECONFIG=$(KUBECONFIG_PATH) && \
		READY_NODES=$$(kubectl get nodes -o json | jq '[.items[] | select(.status.conditions[] | select(.type=="Ready" and .status=="True"))] | length') && \
		echo "$(GREEN)  Ready nodes: $$READY_NODES$(NC)"
	@echo ""
	@echo "Checking pods in $(K8S_NAMESPACE)..."
	@export KUBECONFIG=$(KUBECONFIG_PATH) && \
		kubectl get pods -n $(K8S_NAMESPACE) --no-headers 2>/dev/null | wc -l | \
		xargs -I {} echo "$(GREEN)  Total pods: {}$(NC)"
	@export KUBECONFIG=$(KUBECONFIG_PATH) && \
		kubectl get pods -n $(K8S_NAMESPACE) --field-selector=status.phase=Running --no-headers 2>/dev/null | wc -l | \
		xargs -I {} echo "$(GREEN)  Running pods: {}$(NC)"

.PHONY: validate-app
validate-app: kubeconfig ## Validate application health
	@echo ""
	@echo "$(BLUE)Validating Application Health...$(NC)"
	@echo ""
	@echo "Checking executor deployment..."
	@export KUBECONFIG=$(KUBECONFIG_PATH) && \
		REPLICAS=$$(kubectl get deployment greenlang-executor -n $(K8S_NAMESPACE) -o jsonpath='{.status.readyReplicas}' 2>/dev/null || echo "0") && \
		echo "$(GREEN)  Executor ready replicas: $$REPLICAS$(NC)"
	@echo ""
	@echo "Checking worker deployment..."
	@export KUBECONFIG=$(KUBECONFIG_PATH) && \
		REPLICAS=$$(kubectl get deployment greenlang-worker -n $(K8S_NAMESPACE) -o jsonpath='{.status.readyReplicas}' 2>/dev/null || echo "0") && \
		echo "$(GREEN)  Worker ready replicas: $$REPLICAS$(NC)"
	@echo ""
	@echo "Running health check..."
	@export KUBECONFIG=$(KUBECONFIG_PATH) && \
		kubectl exec -n $(K8S_NAMESPACE) deployment/greenlang-executor -- \
			curl -sf http://localhost:8080/api/v1/health 2>/dev/null && \
		echo "$(GREEN)  Health check: passed$(NC)" || \
		echo "$(YELLOW)  Health check: not available$(NC)"

.PHONY: smoke-test
smoke-test: kubeconfig ## Run smoke tests
	@echo "$(BLUE)Running smoke tests...$(NC)"
	@export KUBECONFIG=$(KUBECONFIG_PATH) && \
		kubectl run smoke-test-$$(date +%s) \
			--image=curlimages/curl:latest \
			--namespace $(K8S_NAMESPACE) \
			--restart=Never --rm -i --attach \
			--command -- curl -sf http://greenlang-executor:8080/api/v1/health && \
		echo "$(GREEN)Smoke tests passed!$(NC)" || \
		echo "$(RED)Smoke tests failed!$(NC)"

# =============================================================================
# Destroy / Cleanup Targets
# =============================================================================
.PHONY: destroy
destroy: preflight ## Destroy all infrastructure (DANGER!)
	@echo ""
	@echo "$(RED)========================================$(NC)"
	@echo "$(RED)WARNING: DESTRUCTIVE OPERATION$(NC)"
	@echo "$(RED)========================================$(NC)"
	@echo ""
	@echo "This will DESTROY all infrastructure in $(ENV)!"
	@echo ""
ifeq ($(ENV),prod)
	@echo "$(RED)PRODUCTION ENVIRONMENT DETECTED!$(NC)"
	@echo "$(RED)This action requires manual confirmation.$(NC)"
	@read -p "Type 'destroy-production' to continue: " confirm && [ "$$confirm" = "destroy-production" ] || exit 1
else
	@read -p "Type 'yes' to continue: " confirm && [ "$$confirm" = "yes" ] || exit 1
endif
	@echo ""
	@echo "$(YELLOW)Creating final backups before destruction...$(NC)"
	@aws rds create-db-snapshot \
		--db-instance-identifier greenlang-$(ENV)-postgres \
		--db-snapshot-identifier greenlang-$(ENV)-final-$$(date +%Y%m%d%H%M%S) 2>/dev/null || true
	@echo ""
	@echo "$(YELLOW)Destroying Kubernetes resources...$(NC)"
	@export KUBECONFIG=$(KUBECONFIG_PATH) && \
		helm uninstall $(HELM_RELEASE_NAME) -n $(K8S_NAMESPACE) --wait 2>/dev/null || true
	@echo ""
	@echo "$(YELLOW)Destroying infrastructure...$(NC)"
	@cd $(TERRAFORM_DIR) && \
		terraform destroy \
			-var-file="$(TF_VAR_FILE)" \
			-auto-approve
	@echo ""
	@echo "$(GREEN)Destruction complete$(NC)"

.PHONY: destroy-k8s
destroy-k8s: kubeconfig ## Destroy Kubernetes deployment only
	@echo "$(YELLOW)Uninstalling Helm release...$(NC)"
	@export KUBECONFIG=$(KUBECONFIG_PATH) && \
		helm uninstall $(HELM_RELEASE_NAME) -n $(K8S_NAMESPACE) --wait
	@echo "$(GREEN)Kubernetes deployment destroyed$(NC)"

.PHONY: clean
clean: ## Clean local artifacts
	@echo "$(BLUE)Cleaning local artifacts...$(NC)"
	@rm -f $(TERRAFORM_DIR)/tfplan.binary
	@rm -f $(TERRAFORM_DIR)/tfplan.json
	@rm -f $(KUBECONFIG_PATH)
	@echo "$(GREEN)Cleaned$(NC)"

# =============================================================================
# Docker / Local Development Targets
# =============================================================================
.PHONY: docker-up
docker-up: ## Start local Docker Compose environment
	@echo "$(BLUE)Starting local Docker environment...$(NC)"
	@docker compose -f $(ROOT_DIR)/docker-compose-unified.yml up -d
	@echo "$(GREEN)Local environment started$(NC)"

.PHONY: docker-down
docker-down: ## Stop local Docker Compose environment
	@echo "$(BLUE)Stopping local Docker environment...$(NC)"
	@docker compose -f $(ROOT_DIR)/docker-compose-unified.yml down
	@echo "$(GREEN)Local environment stopped$(NC)"

.PHONY: docker-logs
docker-logs: ## Show Docker Compose logs
	@docker compose -f $(ROOT_DIR)/docker-compose-unified.yml logs -f

.PHONY: docker-build
docker-build: ## Build Docker images
	@echo "$(BLUE)Building Docker images...$(NC)"
	@docker build -t $(IMAGE_REGISTRY)/greenlang-executor:$(IMAGE_TAG) -f $(ROOT_DIR)/Dockerfile.runner .
	@docker build -t $(IMAGE_REGISTRY)/greenlang-full:$(IMAGE_TAG) -f $(ROOT_DIR)/Dockerfile.full .
	@echo "$(GREEN)Images built$(NC)"

# =============================================================================
# Utility Targets
# =============================================================================
.PHONY: status
status: kubeconfig ## Show deployment status
	@echo ""
	@echo "$(BLUE)========================================$(NC)"
	@echo "$(BLUE)GreenLang $(ENV) Status$(NC)"
	@echo "$(BLUE)========================================$(NC)"
	@echo ""
	@echo "$(YELLOW)Terraform Outputs:$(NC)"
	@cd $(TERRAFORM_DIR) && terraform output 2>/dev/null || echo "  No outputs available"
	@echo ""
	@echo "$(YELLOW)Kubernetes Status:$(NC)"
	@export KUBECONFIG=$(KUBECONFIG_PATH) && \
		kubectl get all -n $(K8S_NAMESPACE) 2>/dev/null || echo "  Namespace not available"
	@echo ""
	@echo "$(YELLOW)Helm Releases:$(NC)"
	@export KUBECONFIG=$(KUBECONFIG_PATH) && \
		helm list -n $(K8S_NAMESPACE) 2>/dev/null || echo "  No releases"

.PHONY: logs
logs: kubeconfig ## Show application logs
	@export KUBECONFIG=$(KUBECONFIG_PATH) && \
		kubectl logs -l app=greenlang --all-containers -f -n $(K8S_NAMESPACE)

.PHONY: shell
shell: kubeconfig ## Open shell in executor pod
	@export KUBECONFIG=$(KUBECONFIG_PATH) && \
		kubectl exec -it deployment/greenlang-executor -n $(K8S_NAMESPACE) -- /bin/sh

.PHONY: port-forward
port-forward: kubeconfig ## Port-forward to application
	@echo "$(BLUE)Port-forwarding to localhost:8080...$(NC)"
	@export KUBECONFIG=$(KUBECONFIG_PATH) && \
		kubectl port-forward svc/greenlang-executor 8080:8080 -n $(K8S_NAMESPACE)

.PHONY: grafana
grafana: kubeconfig ## Port-forward to Grafana
	@echo "$(BLUE)Port-forwarding Grafana to localhost:3000...$(NC)"
	@export KUBECONFIG=$(KUBECONFIG_PATH) && \
		kubectl port-forward svc/kube-prometheus-stack-grafana 3000:80 -n monitoring

# =============================================================================
# Full Deployment Pipeline
# =============================================================================
.PHONY: deploy-all
deploy-all: ## Run complete deployment pipeline
	@echo ""
	@echo "$(BLUE)========================================$(NC)"
	@echo "$(BLUE)Full Deployment Pipeline for $(ENV)$(NC)"
	@echo "$(BLUE)========================================$(NC)"
	@echo ""
	@$(MAKE) preflight ENV=$(ENV)
	@$(MAKE) init ENV=$(ENV)
	@$(MAKE) apply ENV=$(ENV)
	@$(MAKE) kubeconfig ENV=$(ENV)
	@$(MAKE) deploy-addons ENV=$(ENV)
	@$(MAKE) deploy-monitoring ENV=$(ENV)
	@$(MAKE) deploy-k8s ENV=$(ENV)
	@$(MAKE) validate ENV=$(ENV)
	@echo ""
	@echo "$(GREEN)========================================$(NC)"
	@echo "$(GREEN)Deployment Complete!$(NC)"
	@echo "$(GREEN)========================================$(NC)"

# Default target
.DEFAULT_GOAL := help
