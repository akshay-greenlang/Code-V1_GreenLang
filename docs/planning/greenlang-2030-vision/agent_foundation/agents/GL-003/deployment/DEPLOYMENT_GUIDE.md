# GL-003 SteamSystemAnalyzer - Deployment Guide

## Table of Contents
- [Overview](#overview)
- [Prerequisites](#prerequisites)
- [Quick Start](#quick-start)
- [Deployment Environments](#deployment-environments)
- [Deployment Steps](#deployment-steps)
- [Rollback Procedures](#rollback-procedures)
- [Monitoring & Observability](#monitoring--observability)
- [Troubleshooting](#troubleshooting)
- [Security Considerations](#security-considerations)

## Overview

GL-003 SteamSystemAnalyzer is a production-grade AI agent for industrial steam system analysis and optimization. This guide covers complete deployment procedures for Kubernetes environments.

### Architecture Highlights
- **High Availability**: 3+ replicas with pod anti-affinity
- **Auto-scaling**: HPA based on CPU/memory metrics
- **Zero-downtime**: Rolling updates with health checks
- **Security**: Non-root containers, network policies, RBAC
- **Observability**: Prometheus metrics, distributed tracing

## Prerequisites

### Required Tools
```bash
# Kubernetes CLI
kubectl version --client  # >= 1.24

# Kustomize
kustomize version  # >= 4.5

# Docker (for building images)
docker --version  # >= 20.10

# Optional but recommended
helm version  # >= 3.10
kubeval --version  # For manifest validation
trivy --version  # For security scanning
```

### Cluster Requirements
- Kubernetes cluster >= 1.24
- Storage class for persistent volumes
- Ingress controller (nginx recommended)
- Cert-manager for TLS certificates
- Prometheus Operator (optional, for monitoring)

### Access Requirements
- Cluster admin access (for initial setup)
- Container registry access (GCR, ECR, or Docker Hub)
- Database credentials (PostgreSQL)
- Redis connection details

## Quick Start

### 1. Build Docker Image
```bash
cd GreenLang_2030/agent_foundation/agents/GL-003

# Build production image
docker build -f Dockerfile.production \
  --build-arg BUILD_DATE=$(date -u +'%Y-%m-%dT%H:%M:%SZ') \
  --build-arg VCS_REF=$(git rev-parse --short HEAD) \
  --build-arg VERSION=1.0.0 \
  -t greenlang/gl-003:1.0.0 \
  .

# Tag for registry
docker tag greenlang/gl-003:1.0.0 gcr.io/your-project/gl-003:1.0.0

# Push to registry
docker push gcr.io/your-project/gl-003:1.0.0
```

### 2. Configure Secrets
```bash
# Create namespace
kubectl create namespace greenlang

# Create secrets
kubectl create secret generic gl-003-secrets \
  --from-literal=database_url='postgresql://user:pass@host:5432/db' \
  --from-literal=redis_url='redis://host:6379/0' \
  --from-literal=api_key='your-api-key' \
  --from-literal=openai_api_key='sk-your-openai-key' \
  --from-literal=anthropic_api_key='sk-ant-your-anthropic-key' \
  -n greenlang
```

### 3. Deploy Application
```bash
cd deployment

# Validate manifests
./scripts/validate-manifests.sh

# Deploy to development
./scripts/deploy.sh dev

# Deploy to staging
./scripts/deploy.sh staging

# Deploy to production (requires approval)
./scripts/deploy.sh production
```

### 4. Verify Deployment
```bash
# Check pods
kubectl get pods -n greenlang -l app=gl-003-steam-system-analyzer

# Check service
kubectl get svc -n greenlang gl-003-steam-system-analyzer

# Check logs
kubectl logs -f -n greenlang -l app=gl-003-steam-system-analyzer

# Test health endpoint
kubectl port-forward -n greenlang svc/gl-003-steam-system-analyzer 8000:80
curl http://localhost:8000/api/v1/health
```

## Deployment Environments

### Development Environment
- **Namespace**: `greenlang-dev`
- **Replicas**: 1
- **Resources**: Minimal (256Mi RAM, 250m CPU)
- **Auto-deploy**: Enabled on push
- **Purpose**: Testing and development

```bash
# Deploy to dev
kubectl apply -k deployment/kustomize/overlays/dev
```

### Staging Environment
- **Namespace**: `greenlang-staging`
- **Replicas**: 2
- **Resources**: Production-like (512Mi RAM, 500m CPU)
- **Auto-deploy**: Disabled
- **Purpose**: Pre-production validation

```bash
# Deploy to staging
kubectl apply -k deployment/kustomize/overlays/staging
```

### Production Environment
- **Namespace**: `greenlang-prod`
- **Replicas**: 3 (min) to 10 (max)
- **Resources**: Full (1Gi RAM, 1000m CPU)
- **Auto-deploy**: Disabled (manual approval required)
- **Purpose**: Live production workloads

```bash
# Deploy to production
kubectl apply -k deployment/kustomize/overlays/production
```

## Deployment Steps

### Step 1: Pre-Deployment Validation

```bash
# Validate manifests
cd deployment/scripts
./validate-manifests.sh

# Security scan Docker image
trivy image greenlang/gl-003:1.0.0

# Dry-run deployment
kubectl apply -k deployment/kustomize/overlays/production --dry-run=client
```

### Step 2: Deploy Infrastructure

```bash
# Create namespace (if not exists)
kubectl create namespace greenlang-prod

# Apply network policies
kubectl apply -f deployment/networkpolicy.yaml

# Create RBAC
kubectl apply -f deployment/serviceaccount.yaml

# Create resource quotas
kubectl apply -f deployment/resourcequota.yaml
kubectl apply -f deployment/limitrange.yaml
```

### Step 3: Deploy Application

```bash
# Deploy using automated script
./scripts/deploy.sh production

# Or manually with kustomize
kubectl apply -k deployment/kustomize/overlays/production

# Wait for rollout
kubectl rollout status deployment/gl-003-steam-system-analyzer -n greenlang-prod
```

### Step 4: Post-Deployment Verification

```bash
# Check pod status
kubectl get pods -n greenlang-prod -l app=gl-003-steam-system-analyzer

# Check service endpoints
kubectl get endpoints -n greenlang-prod gl-003-steam-system-analyzer

# Check HPA status
kubectl get hpa -n greenlang-prod

# Run health checks
./scripts/health-check.sh
```

### Step 5: Enable Monitoring

```bash
# Apply ServiceMonitor (if using Prometheus Operator)
kubectl apply -f deployment/servicemonitor.yaml

# Verify metrics
kubectl port-forward -n greenlang-prod svc/gl-003-steam-system-analyzer 8001:8001
curl http://localhost:8001/metrics
```

## Rollback Procedures

### Automatic Rollback (Recommended)

```bash
# Rollback to previous revision
./scripts/rollback.sh auto

# Rollback to specific revision
./scripts/rollback.sh auto 3

# Verify rollback
kubectl rollout history deployment/gl-003-steam-system-analyzer -n greenlang-prod
```

### Manual Rollback

```bash
# View rollout history
kubectl rollout history deployment/gl-003-steam-system-analyzer -n greenlang-prod

# Rollback to previous version
kubectl rollout undo deployment/gl-003-steam-system-analyzer -n greenlang-prod

# Rollback to specific revision
kubectl rollout undo deployment/gl-003-steam-system-analyzer \
  -n greenlang-prod \
  --to-revision=2
```

### Restore from Backup

```bash
# Restore from backup file
./scripts/rollback.sh backup /tmp/gl-003-backup-20250117/deployment.yaml
```

## Monitoring & Observability

### Health Checks

```bash
# Liveness check
curl http://gl-003.greenlang.io/api/v1/health

# Readiness check
curl http://gl-003.greenlang.io/api/v1/ready

# Metrics endpoint
curl http://gl-003.greenlang.io/metrics
```

### Logs

```bash
# Stream logs from all pods
kubectl logs -f -n greenlang-prod -l app=gl-003-steam-system-analyzer

# Logs from specific pod
kubectl logs -n greenlang-prod gl-003-steam-system-analyzer-<pod-id>

# Previous container logs (after restart)
kubectl logs -n greenlang-prod gl-003-steam-system-analyzer-<pod-id> --previous
```

### Metrics

```bash
# Prometheus metrics
kubectl port-forward -n greenlang-prod svc/gl-003-steam-system-analyzer 8001:8001
curl http://localhost:8001/metrics

# Key metrics:
# - http_requests_total
# - http_request_duration_seconds
# - steam_analysis_total
# - steam_pressure_psi
# - steam_temperature_fahrenheit
# - steam_efficiency_percent
```

### Grafana Dashboards

Import pre-built dashboard:
```bash
# Dashboard JSON located at:
# deployment/monitoring/grafana/gl-003-dashboard.json

# Or access via Grafana UI:
# https://grafana.greenlang.io/d/gl-003-steam-analyzer
```

## Troubleshooting

### Pods Not Starting

```bash
# Check pod events
kubectl describe pod -n greenlang-prod gl-003-steam-system-analyzer-<pod-id>

# Check logs
kubectl logs -n greenlang-prod gl-003-steam-system-analyzer-<pod-id>

# Common issues:
# - Image pull errors: Check registry credentials
# - Init container failures: Check database/Redis connectivity
# - Resource constraints: Check node capacity
```

### Service Not Accessible

```bash
# Check service
kubectl get svc -n greenlang-prod gl-003-steam-system-analyzer

# Check endpoints
kubectl get endpoints -n greenlang-prod gl-003-steam-system-analyzer

# Check ingress
kubectl get ingress -n greenlang-prod

# Common issues:
# - No endpoints: Pods not ready
# - Ingress issues: Check ingress controller logs
# - Network policy: Verify NetworkPolicy rules
```

### High Memory Usage

```bash
# Check resource usage
kubectl top pods -n greenlang-prod -l app=gl-003-steam-system-analyzer

# Check HPA scaling
kubectl get hpa -n greenlang-prod

# Actions:
# 1. Increase resource limits
# 2. Review memory leaks in application logs
# 3. Enable memory profiling
```

### Database Connection Issues

```bash
# Test database connectivity from pod
kubectl exec -it -n greenlang-prod gl-003-steam-system-analyzer-<pod-id> -- \
  curl -v telnet://postgresql.database.svc.cluster.local:5432

# Check secrets
kubectl get secret gl-003-secrets -n greenlang-prod -o yaml

# Verify NetworkPolicy allows database access
```

## Security Considerations

### Container Security

- **Non-root user**: Runs as UID 1000
- **Read-only filesystem**: Except for `/tmp`, `/app/logs`, `/app/cache`
- **Dropped capabilities**: All capabilities dropped except `NET_BIND_SERVICE`
- **Security scanning**: Trivy and Snyk scans in CI/CD

### Network Security

- **NetworkPolicy**: Restricts ingress/egress traffic
- **TLS encryption**: All external traffic via HTTPS
- **Service mesh**: Optional Istio/Linkerd integration

### Secret Management

- **Kubernetes Secrets**: Base64 encoded (not secure for production)
- **External Secrets Operator**: Sync from AWS Secrets Manager, GCP Secret Manager
- **Sealed Secrets**: Encrypted secrets for GitOps
- **HashiCorp Vault**: Enterprise secret management

### RBAC

- **Service Account**: Minimal permissions (read ConfigMaps, Secrets)
- **Role/RoleBinding**: Namespace-scoped permissions
- **No cluster-admin**: Never use cluster-admin for applications

## CI/CD Integration

### GitHub Actions

Workflows located at:
- `.github/workflows/gl-003-ci.yaml` - Main CI pipeline
- `.github/workflows/gl-003-scheduled.yaml` - Scheduled scans

### Deployment Pipeline

```yaml
1. Lint & Type Check (ruff, mypy, black)
2. Unit & Integration Tests (pytest, 95% coverage)
3. Security Scan (bandit, safety, Trivy)
4. Build Docker Image (multi-stage)
5. Push to Registry (GCR/ECR)
6. Validate Manifests (kubectl, kustomize)
7. Deploy to Dev (automatic)
8. Deploy to Staging (manual approval)
9. Deploy to Production (manual approval + change ticket)
```

## Performance Tuning

### Resource Optimization

```yaml
# Recommended resource settings
resources:
  requests:
    memory: "512Mi"
    cpu: "500m"
  limits:
    memory: "1024Mi"
    cpu: "1000m"
```

### Scaling Configuration

```yaml
# HPA settings
minReplicas: 3
maxReplicas: 10
targetCPUUtilization: 70%
targetMemoryUtilization: 80%
```

### Database Connection Pool

```python
# PostgreSQL pool settings
DB_POOL_SIZE=10
DB_MAX_OVERFLOW=20
DB_POOL_TIMEOUT=30
```

## Maintenance

### Regular Tasks

- **Weekly**: Review logs and metrics
- **Monthly**: Update dependencies
- **Quarterly**: Security audit
- **Annually**: Disaster recovery test

### Upgrade Procedures

```bash
# 1. Test in dev environment
./scripts/deploy.sh dev

# 2. Validate in staging
./scripts/deploy.sh staging

# 3. Schedule maintenance window
# 4. Deploy to production
./scripts/deploy.sh production

# 5. Monitor for 24 hours
# 6. Document changes
```

## Support

- **Documentation**: https://docs.greenlang.ai/agents/GL-003
- **Slack**: #gl-003-alerts
- **PagerDuty**: GL-003-SteamSystem
- **Email**: gl-003-oncall@greenlang.ai

---

**Last Updated**: 2025-11-17
**Version**: 1.0.0
**Maintained By**: GreenLang Platform Team
