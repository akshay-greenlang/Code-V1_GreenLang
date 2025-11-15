# GL-002 BoilerEfficiencyOptimizer - Complete Deployment Guide

Comprehensive guide for deploying GL-002 to development, staging, and production environments.

## Executive Summary

GL-002 BoilerEfficiencyOptimizer is a production-ready microservice deployed on Kubernetes with:
- Multi-stage Docker builds (final size: 500 MB)
- Kubernetes manifests for high availability (3-10 pods auto-scaling)
- GitHub Actions CI/CD pipelines with security scanning
- Comprehensive monitoring with Prometheus and health checks
- Infrastructure as Code with Terraform templates
- Zero-trust networking with Kubernetes NetworkPolicies
- Automatic TLS/SSL with cert-manager

**Key Metrics**:
- **RTO**: 4 hours (Recovery Time Objective)
- **RPO**: 1 hour (Recovery Point Objective)
- **Uptime SLA**: 99.9% (3 nines)
- **Performance**: <500ms p99 latency
- **Scalability**: 3-10 pods, auto-scaling at 70% CPU

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Development Environment Setup](#development-environment-setup)
3. [Building & Testing](#building--testing)
4. [Staging Deployment](#staging-deployment)
5. [Production Deployment](#production-deployment)
6. [CI/CD Integration](#cicd-integration)
7. [Monitoring & Observability](#monitoring--observability)
8. [Infrastructure as Code (Terraform)](#infrastructure-as-code-terraform)
9. [Operational Procedures](#operational-procedures)
10. [Troubleshooting & Support](#troubleshooting--support)

---

## Prerequisites

### System Requirements

**Local Development**:
- CPU: 4+ cores
- RAM: 8+ GB
- Disk: 50+ GB SSD
- OS: Linux (Ubuntu 20.04+), macOS 11+, or Windows 10+ (WSL2)

**Kubernetes Cluster**:
- Kubernetes 1.24+
- 3+ nodes with 4 CPU and 8 GB RAM each
- Persistent volume storage (20+ GB)
- Load balancer support (cloud or MetalLB)

### Software Installation

#### macOS/Linux

```bash
# Install Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
sudo usermod -aG docker $USER

# Install Kubernetes tools
brew install kubectl helm minikube

# Install Python
brew install python@3.11

# Install other tools
brew install git jq yq
```

#### Windows (using WSL2)

```powershell
# Enable WSL2
wsl --install

# Inside WSL2
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh

# Install kubectl
curl -LO "https://dl.k8s.io/release/$(curl -L -s https://dl.k8s.io/release/stable.txt)/bin/linux/amd64/kubectl"
sudo install -o root -g root -m 0755 kubectl /usr/local/bin/kubectl
```

### Required Credentials

Create a `.env` file in your working directory:

```bash
# Copy template
cp .env.template .env

# Edit with actual values
export GITHUB_TOKEN=ghp_...
export DOCKER_USERNAME=...
export DOCKER_PASSWORD=...
export KUBE_CONFIG=$(cat ~/.kube/config | base64)
export DATABASE_URL=postgresql://user:pass@host:5432/db
export REDIS_URL=redis://host:6379/0
export API_KEY=your-api-key-here
```

---

## Development Environment Setup

### 1. Clone Repository

```bash
git clone https://github.com/greenlang/greenlang.git
cd greenlang
cd GreenLang_2030/agent_foundation/agents/GL-002
```

### 2. Local Kubernetes Cluster (Minikube)

```bash
# Start minikube
minikube start \
  --cpus=4 \
  --memory=8192 \
  --driver=docker \
  --kubernetes-version=1.27

# Enable required addons
minikube addons enable ingress
minikube addons enable metrics-server

# Get cluster info
minikube status
kubectl get nodes
```

### 3. Setup Namespaces and Configuration

```bash
# Create namespace
kubectl create namespace greenlang

# Create ConfigMap
kubectl create configmap gl-002-config \
  --from-file=config/development.yaml \
  -n greenlang

# Create Secrets (development values)
kubectl create secret generic gl-002-secrets \
  --from-literal=database_url='postgresql://dev:dev@postgres:5432/greenlang_dev' \
  --from-literal=redis_url='redis://redis:6379/1' \
  --from-literal=api_key='dev-api-key' \
  -n greenlang
```

### 4. Deploy Local Dependencies

```bash
# Deploy PostgreSQL
kubectl run postgres \
  --image=postgres:14-alpine \
  --env=POSTGRES_PASSWORD=postgres \
  --env=POSTGRES_DB=greenlang_dev \
  -n greenlang \
  --expose \
  --port=5432

# Deploy Redis
kubectl run redis \
  --image=redis:7-alpine \
  -n greenlang \
  --expose \
  --port=6379

# Wait for deployment
kubectl wait --for=condition=Ready pod -l run=postgres -n greenlang --timeout=300s
kubectl wait --for=condition=Ready pod -l run=redis -n greenlang --timeout=300s
```

### 5. Build and Deploy Application

```bash
# Build image (using Minikube's Docker daemon)
eval $(minikube docker-env)
docker build -t gl-002:dev .

# Deploy to Minikube
kubectl apply -f deployment/configmap.yaml -n greenlang
kubectl apply -f deployment/deployment.yaml -n greenlang
kubectl apply -f deployment/service.yaml -n greenlang

# Verify
kubectl get pods -n greenlang
kubectl logs -f deployment/gl-002-boiler-efficiency -n greenlang
```

### 6. Access Application Locally

```bash
# Port forward
kubectl port-forward -n greenlang svc/gl-002-boiler-efficiency 8000:80

# Test
curl http://localhost:8000/api/v1/health
curl http://localhost:8000/api/v1/ready
```

---

## Building & Testing

### Docker Build

```bash
# Build image
docker build -t gl-002:latest .

# Build with custom tag
docker build -t gl-002:v1.0.0 .

# Build for specific Python version
docker build --build-arg PYTHON_VERSION=3.11 -t gl-002:py311 .

# Build and push to registry
docker build -t ghcr.io/greenlang/gl-002:latest .
docker push ghcr.io/greenlang/gl-002:latest
```

### Local Testing

```bash
# Run container
docker run -p 8000:8000 \
  -e GREENLANG_ENV=development \
  -e LOG_LEVEL=DEBUG \
  gl-002:latest

# In another terminal, test endpoints
curl http://localhost:8000/api/v1/health
curl http://localhost:8000/api/v1/ready
curl http://localhost:8000/api/v1/metrics

# Run with database
docker-compose -f docker-compose.dev.yml up -d
```

### Unit and Integration Tests

```bash
# Install test dependencies
pip install -r requirements.txt
pip install pytest pytest-cov pytest-asyncio pytest-mock

# Run unit tests
pytest tests/unit --cov=. --cov-report=html

# Run integration tests (requires PostgreSQL and Redis)
pytest tests/integration -v

# Generate coverage report
coverage report
coverage html
open htmlcov/index.html
```

### Security Scanning

```bash
# Lint code
ruff check .
black --check .
isort --check-only .

# Type checking
mypy . --ignore-missing-imports

# Security scanning
bandit -r .
safety check

# Container image scanning (if using Trivy)
trivy image gl-002:latest
```

---

## Staging Deployment

### Prerequisites

- AWS account with EKS cluster running
- RDS PostgreSQL instance
- ElastiCache Redis cluster
- ECR repository for images

### Deployment Steps

#### 1. Prepare AWS Environment

```bash
# Export variables
export AWS_REGION=us-east-1
export CLUSTER_NAME=greenlang-staging
export IMAGE_URI=<aws-account>.dkr.ecr.${AWS_REGION}.amazonaws.com/gl-002:staging

# Login to ECR
aws ecr get-login-password --region $AWS_REGION | \
  docker login --username AWS --password-stdin $(echo $IMAGE_URI | cut -d/ -f1)

# Build and push image
docker build -t gl-002:staging .
docker tag gl-002:staging $IMAGE_URI
docker push $IMAGE_URI
```

#### 2. Update Kubernetes Manifests

```bash
# Edit deployment with staging values
kubectl set image deployment/gl-002-boiler-efficiency \
  gl-002-boiler-efficiency=$IMAGE_URI \
  -n greenlang-staging

# Update ConfigMap with staging configuration
kubectl create configmap gl-002-config \
  --from-file=config/staging.yaml \
  -n greenlang-staging \
  -o yaml --dry-run=client | kubectl apply -f -
```

#### 3. Deploy Secrets

```bash
# Create secrets from AWS Secrets Manager
aws secretsmanager get-secret-value \
  --secret-id gl-002/staging \
  --region $AWS_REGION \
  --query SecretString \
  --output text | kubectl create secret generic gl-002-secrets --from-file=/dev/stdin -n greenlang-staging
```

#### 4. Apply Kubernetes Manifests

```bash
# Apply in order
kubectl apply -f deployment/configmap.yaml -n greenlang-staging
kubectl apply -f deployment/secret.yaml -n greenlang-staging
kubectl apply -f deployment/deployment.yaml -n greenlang-staging
kubectl apply -f deployment/service.yaml -n greenlang-staging
kubectl apply -f deployment/hpa.yaml -n greenlang-staging
kubectl apply -f deployment/networkpolicy.yaml -n greenlang-staging
kubectl apply -f deployment/ingress.yaml -n greenlang-staging

# Wait for rollout
kubectl rollout status deployment/gl-002-boiler-efficiency -n greenlang-staging --timeout=5m
```

#### 5. Verify Deployment

```bash
# Check pod status
kubectl get pods -n greenlang-staging

# Check logs
kubectl logs -f deployment/gl-002-boiler-efficiency -n greenlang-staging

# Test health endpoints
STAGING_URL=$(kubectl get ingress -n greenlang-staging -o jsonpath='{.items[0].status.loadBalancer.ingress[0].hostname}')
curl -s https://$STAGING_URL/api/v1/health
curl -s https://$STAGING_URL/api/v1/ready
```

#### 6. Run Smoke Tests

```bash
# Test basic functionality
./scripts/smoke-tests-staging.sh

# Expected output:
# ✓ Health check passed
# ✓ Readiness check passed
# ✓ Metrics endpoint responding
# ✓ Database connectivity OK
# ✓ Cache connectivity OK
```

---

## Production Deployment

### Pre-Deployment Checklist

- [ ] Code reviewed and merged to main branch
- [ ] All tests passing (100% CI pass rate)
- [ ] Security scans completed with zero critical issues
- [ ] Performance tested (load test at 2x expected traffic)
- [ ] Disaster recovery procedure tested
- [ ] Stakeholders notified of deployment window
- [ ] Rollback plan documented
- [ ] On-call engineer available

### Deployment Procedure

#### 1. Prepare Production Environment

```bash
# Export production variables
export AWS_REGION=us-east-1
export CLUSTER_NAME=greenlang-production
export IMAGE_URI=<aws-account>.dkr.ecr.${AWS_REGION}.amazonaws.com/gl-002:v1.0.0

# Build production image
docker build -t gl-002:v1.0.0 .
docker tag gl-002:v1.0.0 $IMAGE_URI
docker push $IMAGE_URI

# Tag as latest
docker tag gl-002:v1.0.0 $IMAGE_URI:latest
docker push $IMAGE_URI:latest
```

#### 2. Setup Production Secrets

```bash
# Use AWS Secrets Manager (preferred over base64 encoding)
aws secretsmanager create-secret \
  --name gl-002/production \
  --secret-string '{
    "database_url": "postgresql://...",
    "redis_url": "redis://...",
    "api_key": "...",
    "jwt_secret": "..."
  }' \
  --region $AWS_REGION

# OR create from Sealed Secrets
kubectl create secret generic gl-002-secrets \
  --from-literal=database_url=$DATABASE_URL \
  --from-literal=redis_url=$REDIS_URL \
  --from-literal=api_key=$API_KEY \
  -n greenlang \
  --dry-run=client -o yaml | kubeseal -f - -w deployment/secret-sealed.yaml
```

#### 3. Blue-Green Deployment Strategy

```bash
# Get current (blue) deployment
kubectl get deployment gl-002-boiler-efficiency -n greenlang -o yaml > blue-deployment.yaml

# Create green deployment with new image
kubectl set image deployment/gl-002-boiler-efficiency \
  gl-002-boiler-efficiency=$IMAGE_URI \
  -n greenlang

# Wait for green deployment to be ready
kubectl rollout status deployment/gl-002-boiler-efficiency -n greenlang --timeout=10m

# Verify all pods are running
kubectl get pods -n greenlang -l app=gl-002-boiler-efficiency

# Run production verification
./scripts/verify-production.sh

# If successful, keep green deployment
# If failed, rollback to blue:
# kubectl rollout undo deployment/gl-002-boiler-efficiency -n greenlang
```

#### 4. Verify Production Deployment

```bash
# Check all resources
kubectl get all -n greenlang

# Check ingress
kubectl get ingress -n greenlang

# Get production URL
PROD_URL=$(kubectl get ingress gl-002-boiler-efficiency-ingress -n greenlang -o jsonpath='{.spec.rules[0].host}')

# Test endpoints
curl -s -I https://$PROD_URL/api/v1/health
curl -s https://$PROD_URL/api/v1/ready | jq .
curl -s https://$PROD_URL/api/v1/metrics | head -20

# Check logs
kubectl logs -n greenlang -l app=gl-002-boiler-efficiency --tail=50

# Monitor metrics
kubectl top pods -n greenlang
```

#### 5. Production Monitoring

```bash
# Setup Prometheus scraping
kubectl apply -f monitoring/prometheus-servicemonitor.yaml

# Create Grafana dashboard
kubectl apply -f monitoring/grafana-dashboard.yaml

# Check metrics are being collected
curl -s http://prometheus:9090/api/v1/targets | jq '.data.activeTargets'

# Create alerting rules
kubectl apply -f monitoring/alerting-rules.yaml
```

#### 6. Post-Deployment

```bash
# Update documentation
git add .
git commit -m "docs: Update deployment record for v1.0.0"

# Notify stakeholders
./scripts/send-deployment-notification.sh

# Monitor for 24 hours
watch -n 30 'kubectl top pods -n greenlang'

# Check error rates
kubectl logs -n greenlang -l app=gl-002-boiler-efficiency | grep ERROR
```

---

## CI/CD Integration

### GitHub Actions Setup

#### 1. Store Credentials as Secrets

```bash
# Kubernetes
gh secret set KUBE_CONFIG_STAGING --body "$(cat ~/.kube/config-staging | base64)"
gh secret set KUBE_CONFIG_PRODUCTION --body "$(cat ~/.kube/config-prod | base64)"

# Slack
gh secret set SLACK_WEBHOOK_URL --body "https://hooks.slack.com/services/..."

# AWS (if using ECR)
gh secret set AWS_ACCESS_KEY_ID --body "AKIAI..."
gh secret set AWS_SECRET_ACCESS_KEY --body "..."
gh secret set AWS_REGION --body "us-east-1"
```

#### 2. Configure Workflow Triggers

```yaml
# Edit .github/workflows/gl-002-ci.yaml
# Triggers on:
# - Push to main, master, develop
# - Pull requests to main, master, develop
# - Manual workflow_dispatch
```

#### 3. Configure Deployment Environments

```bash
# Add environment protection rules in GitHub:
# Settings > Environments > production > Required reviewers

# Require approval from:
# - @greenlang/devops-team
# - @greenlang/engineering-leads
```

#### 4. Monitor CI/CD Execution

```bash
# View workflow runs
gh run list --workflow=gl-002-ci.yaml

# View specific run details
gh run view <run-id> --log

# Cancel a run
gh run cancel <run-id>

# View deployment status
gh deployment list --environment=production
```

---

## Monitoring & Observability

### Prometheus Setup

```bash
# Deploy Prometheus
helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
helm install prometheus prometheus-community/kube-prometheus-stack \
  -n monitoring \
  --create-namespace \
  --values monitoring/prometheus-values.yaml

# Verify
kubectl get pods -n monitoring
kubectl port-forward -n monitoring svc/prometheus-kube-prometheus-prometheus 9090:9090
```

### Grafana Setup

```bash
# Get Grafana password
kubectl get secret -n monitoring prometheus-grafana -o jsonpath="{.data.admin-password}" | base64 --decode

# Port forward
kubectl port-forward -n monitoring svc/prometheus-grafana 3000:80

# Access at http://localhost:3000
# Username: admin
# Password: <from above>
```

### Create Alerts

```bash
# High error rate alert
kubectl apply -f monitoring/alerts/high-error-rate.yaml

# Pod crash alert
kubectl apply -f monitoring/alerts/pod-crash.yaml

# Emissions compliance alert
kubectl apply -f monitoring/alerts/emissions-compliance.yaml

# Database connection alert
kubectl apply -f monitoring/alerts/database-connection.yaml
```

### Logging Setup

```bash
# Deploy ELK Stack (or Loki)
helm repo add grafana https://grafana.github.io/helm-charts
helm install loki grafana/loki-stack \
  -n logging \
  --create-namespace

# Forward logs to centralized system
kubectl apply -f monitoring/fluent-bit-configmap.yaml
kubectl apply -f monitoring/fluent-bit-daemonset.yaml
```

---

## Infrastructure as Code (Terraform)

### Prerequisites

```bash
# Install Terraform
brew install terraform

# Install AWS CLI
brew install awscli

# Configure AWS credentials
aws configure
```

### Terraform Modules

```
terraform/
├── main.tf                  # Main infrastructure
├── vpc.tf                   # VPC and networking
├── rds.tf                   # PostgreSQL database
├── elasticache.tf           # Redis cache
├── eks.tf                   # Kubernetes cluster
├── iam.tf                   # IAM roles and policies
├── s3.tf                    # Backup storage
└── variables.tf             # Input variables
```

### Deploy Infrastructure

```bash
cd terraform

# Initialize Terraform
terraform init

# Plan deployment
terraform plan -out=tfplan

# Apply changes
terraform apply tfplan

# Save outputs
terraform output > outputs.json
```

### Manage Infrastructure

```bash
# Update variable
terraform apply -var="environment=production" -var="instance_count=5"

# Destroy (use with caution!)
terraform destroy -auto-approve

# Import existing resource
terraform import aws_rds_cluster.main arn:aws:rds:...

# State management
terraform state list
terraform state show aws_rds_cluster.main
terraform state rm aws_instance.example  # Remove from state
```

---

## Operational Procedures

### Daily Operations

```bash
# Morning check
kubectl get all -n greenlang
kubectl top pods -n greenlang
kubectl logs -n greenlang -l app=gl-002-boiler-efficiency --tail=100 | grep ERROR

# Check metrics
curl -s http://prometheus:9090/api/v1/query?query='up{job="gl-002"}' | jq .

# Review alerts
kubectl get alerts -n monitoring
```

### Scheduled Maintenance

```bash
# Database maintenance (off-peak hours)
kubectl drain <node-name> --ignore-daemonsets
# Perform maintenance
kubectl uncordon <node-name>

# Rolling restart (for config updates)
kubectl rollout restart deployment/gl-002-boiler-efficiency -n greenlang
kubectl rollout status deployment/gl-002-boiler-efficiency -n greenlang

# Update dependencies
kubectl set image deployment/gl-002-boiler-efficiency \
  gl-002-boiler-efficiency=ghcr.io/greenlang/gl-002:latest \
  -n greenlang
```

### Incident Response

```bash
# Check pod status
kubectl describe pod <pod-name> -n greenlang

# View crash logs
kubectl logs <pod-name> -n greenlang --previous

# Exec into pod for debugging
kubectl exec -it <pod-name> -n greenlang -- /bin/sh

# Scale down problematic pods
kubectl scale deployment gl-002-boiler-efficiency --replicas=1 -n greenlang

# Check node health
kubectl describe node <node-name>
kubectl top node <node-name>
```

### Rollback Procedure

```bash
# View rollout history
kubectl rollout history deployment/gl-002-boiler-efficiency -n greenlang

# Rollback to previous version
kubectl rollout undo deployment/gl-002-boiler-efficiency -n greenlang

# Rollback to specific revision
kubectl rollout undo deployment/gl-002-boiler-efficiency --to-revision=2 -n greenlang

# Verify rollback
kubectl rollout status deployment/gl-002-boiler-efficiency -n greenlang
```

---

## Troubleshooting & Support

### Common Issues

**Pods not starting**
```bash
kubectl describe pod <name> -n greenlang
kubectl logs <name> -n greenlang
# Check: image pull errors, resource limits, secret/configmap missing
```

**High latency**
```bash
kubectl top pods -n greenlang
kubectl top nodes
# Check: CPU/memory pressure, network issues, database slow queries
```

**Ingress not working**
```bash
kubectl get ingress -n greenlang
kubectl describe ingress <name> -n greenlang
# Check: DNS resolution, TLS certificate, ingress controller running
```

**Database connection failures**
```bash
kubectl run -it --rm debug --image=postgres:14 -- \
  psql postgresql://user:pass@host:5432/db
# Check: connection string, RDS security groups, network policy
```

### Getting Help

- **Documentation**: https://docs.greenlang.ai/agents/GL-002
- **Issues**: https://github.com/greenlang/agents/issues?label=GL-002
- **Email**: boiler-systems@greenlang.ai
- **Slack**: #gl-boiler-systems
- **On-call**: See runbook for escalation

### Feedback

Submit improvements or bug reports:
```bash
gh issue create --title "GL-002 deployment issue" --body "Description..."
```

---

## Appendix

### A. Environment Variables Reference

See `.env.template` for complete list of environment variables.

### B. Configuration File Reference

- `config/development.yaml` - Development settings
- `config/staging.yaml` - Staging settings
- `config/production.yaml` - Production settings

### C. Kubernetes Manifest Reference

- `deployment/deployment.yaml` - Pod and container specs
- `deployment/service.yaml` - Internal service
- `deployment/ingress.yaml` - External HTTPS access
- `deployment/hpa.yaml` - Auto-scaling rules
- `deployment/networkpolicy.yaml` - Network security

### D. Monitoring Queries

```promql
# Request latency p99
histogram_quantile(0.99, gl_002_http_request_duration_seconds_bucket)

# Error rate
rate(gl_002_http_requests_total{status="error"}[5m])

# Pod memory usage
container_memory_usage_bytes{pod="gl-002-boiler-efficiency-*"}

# Emissions compliance status
gl_002_emissions_compliance_status{boiler_id="*"}
```

---

**Last Updated**: November 15, 2025
**Version**: 1.0.0
**Maintainer**: GreenLang DevOps Team
