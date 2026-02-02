# GL-VCCI Advanced Deployment Guide

## Overview

This guide covers the advanced deployment capabilities of the GL-VCCI Carbon Intelligence Platform, including production-grade Kubernetes configurations, deployment strategies, and automation scripts.

## Table of Contents

1. [Architecture](#architecture)
2. [Prerequisites](#prerequisites)
3. [Quick Start](#quick-start)
4. [Deployment Strategies](#deployment-strategies)
5. [Configuration Management](#configuration-management)
6. [Security](#security)
7. [Monitoring and Observability](#monitoring-and-observability)
8. [Troubleshooting](#troubleshooting)
9. [CI/CD Integration](#cicd-integration)

## Architecture

### Kubernetes Resources

```
GL-VCCI Platform
├── Namespaces (dev, staging, prod)
├── Deployments
│   ├── Backend API (FastAPI)
│   ├── Worker (Celery)
│   ├── Beat (Scheduler)
│   └── Frontend (React)
├── StatefulSets
│   ├── PostgreSQL
│   ├── Redis
│   └── Weaviate
├── Services
├── ConfigMaps
├── Secrets (External Secrets Operator)
├── HPA (Horizontal Pod Autoscaler)
├── PDB (Pod Disruption Budget)
├── Network Policies
└── Istio Service Mesh (optional)
```

### High Availability Features

- **Auto-scaling**: HPA based on CPU, memory, and custom metrics
- **High Availability**: PDB ensures minimum pod availability during updates
- **Zero-downtime deployments**: Rolling, Blue-Green, and Canary strategies
- **Network Security**: Network policies restrict pod-to-pod communication
- **Secrets Management**: External secrets integration (Vault, AWS Secrets Manager)
- **Service Mesh**: Istio for advanced traffic management and security (optional)

## Prerequisites

### Required Tools

```bash
# Kubernetes cluster (v1.26+)
kubectl version --client

# Kustomize (v4.5+)
kustomize version

# Helm (v3.10+)
helm version

# Docker (for building images)
docker --version

# Optional: Istio CLI
istioctl version
```

### Cluster Requirements

- Kubernetes 1.26 or higher
- Minimum 3 worker nodes
- Network plugin supporting NetworkPolicy (Calico, Cilium, Weave)
- Storage class for persistent volumes
- Ingress controller (nginx, traefik)

### Resource Requirements

**Production Environment:**
- CPU: 100 cores total
- Memory: 200 GiB total
- Storage: 500 GiB total

**Staging Environment:**
- CPU: 50 cores total
- Memory: 100 GiB total
- Storage: 250 GiB total

**Development Environment:**
- CPU: 25 cores total
- Memory: 50 GiB total
- Storage: 100 GiB total

## Quick Start

### 1. Clone Repository

```bash
git clone https://github.com/your-org/GL-VCCI-Carbon-APP.git
cd GL-VCCI-Carbon-APP/VCCI-Scope3-Platform
```

### 2. Configure Environment

```bash
# Copy environment template
cp .env.example .env

# Edit configuration
vim .env
```

### 3. Create Namespaces

```bash
kubectl apply -f k8s/namespace.yaml
```

### 4. Setup External Secrets

```bash
# Install External Secrets Operator
helm repo add external-secrets https://charts.external-secrets.io
helm install external-secrets external-secrets/external-secrets \
  -n external-secrets-system --create-namespace

# Configure secret store (Vault example)
kubectl apply -f k8s/external-secrets.yaml
```

### 5. Deploy to Development

```bash
# Build and push images
export VERSION="v2.0.0"
./deployment/scripts/build-images.sh $VERSION

# Deploy using main script
./deployment/scripts/deploy.sh -e dev -v $VERSION
```

### 6. Deploy to Production

```bash
# Deploy with canary strategy
./deployment/scripts/deploy.sh -e prod -v $VERSION -s canary
```

## Deployment Strategies

### 1. Rolling Deployment (Default)

**Use Case**: Standard updates, low-risk changes

**Advantages:**
- Simple and reliable
- Built into Kubernetes
- No additional infrastructure

**How to Deploy:**
```bash
./deployment/scripts/deploy.sh -e prod -v v2.0.0 -s rolling
```

**Configuration:**
```yaml
strategy:
  type: RollingUpdate
  rollingUpdate:
    maxSurge: 1
    maxUnavailable: 0
```

### 2. Blue-Green Deployment

**Use Case**: Zero-downtime, instant rollback capability

**Advantages:**
- Instant cutover
- Easy rollback
- Full production testing before cutover

**How to Deploy:**
```bash
./deployment/scripts/deploy.sh -e prod -v v2.0.0 -s blue-green
```

**Documentation**: See [deployment/strategies/blue-green-deployment.md](./strategies/blue-green-deployment.md)

### 3. Canary Deployment

**Use Case**: High-risk changes, gradual rollout

**Advantages:**
- Limited blast radius
- Early issue detection
- Gradual traffic shift

**How to Deploy:**
```bash
./deployment/scripts/deploy.sh -e prod -v v2.0.0 -s canary
```

**Stages:**
1. 5% traffic to canary (1 pod)
2. 10% traffic to canary (2 pods)
3. 25% traffic to canary (5 pods)
4. 50% traffic to canary (10 pods)
5. 75% traffic to canary (15 pods)
6. 100% traffic to canary (20 pods)

**Documentation**: See [deployment/strategies/canary-deployment.md](./strategies/canary-deployment.md)

## Configuration Management

### Environment-Specific Configurations

Configurations are managed using Kustomize overlays:

```
k8s/
├── base/              # Base configurations
├── overlays/
│   ├── dev/          # Development overrides
│   ├── staging/      # Staging overrides
│   └── prod/         # Production overrides
```

### Customizing Environments

**Production (k8s/overlays/prod/kustomization.yaml):**
```yaml
replicas:
  - name: vcci-backend-api
    count: 5
  - name: vcci-worker
    count: 3

images:
  - name: YOUR_REGISTRY/vcci-backend
    newTag: v2.0.0

configMapGenerator:
  - name: backend-config
    behavior: merge
    literals:
      - APP_ENV=production
      - LOG_LEVEL=info
```

### Resource Allocation

**Backend API:**
- Requests: 500m CPU, 1Gi memory
- Limits: 2 CPU, 4Gi memory

**Worker:**
- Requests: 1 CPU, 2Gi memory
- Limits: 4 CPU, 8Gi memory

**PostgreSQL:**
- Requests: 2 CPU, 4Gi memory
- Limits: 8 CPU, 16Gi memory

## Security

### Network Policies

Network policies implement zero-trust security:

```bash
# Apply network policies
kubectl apply -f k8s/network-policy.yaml
```

**Default policy**: Deny all traffic
**Allowed traffic**:
- Frontend → Backend API
- Backend API → Database, Redis, Weaviate
- Worker → Database, Redis, Weaviate
- Monitoring → All (metrics scraping)

### Secrets Management

#### Using HashiCorp Vault

```bash
# Configure Vault
vault auth enable kubernetes
vault write auth/kubernetes/config \
  kubernetes_host=https://kubernetes.default.svc

# Store secrets
vault kv put secret/vcci/production/postgres \
  username=vcci \
  password=<strong-password>

vault kv put secret/vcci/production/api-keys \
  openai=sk-... \
  anthropic=sk-ant-...
```

#### Using AWS Secrets Manager

```bash
# Create IAM policy
aws iam create-policy \
  --policy-name VCCISecretsManagerPolicy \
  --policy-document file://iam-policy.json

# Create IAM role for service account
eksctl create iamserviceaccount \
  --name vcci-secrets-sa \
  --namespace vcci-production \
  --cluster vcci-cluster \
  --attach-policy-arn arn:aws:iam::ACCOUNT_ID:policy/VCCISecretsManagerPolicy \
  --approve
```

### Pod Security

All pods run with:
- Non-root user (UID 1000)
- Read-only root filesystem
- Dropped capabilities
- seccomp profile

```yaml
securityContext:
  runAsNonRoot: true
  runAsUser: 1000
  readOnlyRootFilesystem: true
  allowPrivilegeEscalation: false
  capabilities:
    drop:
      - ALL
```

## Monitoring and Observability

### Prometheus Metrics

```bash
# Install Prometheus
helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
helm install prometheus prometheus-community/kube-prometheus-stack \
  -n monitoring --create-namespace
```

### Key Metrics

**Application Metrics:**
- `http_requests_total` - Total HTTP requests
- `http_request_duration_seconds` - Request latency
- `celery_tasks_total` - Total Celery tasks
- `celery_task_duration_seconds` - Task duration

**System Metrics:**
- `container_cpu_usage_seconds_total` - CPU usage
- `container_memory_usage_bytes` - Memory usage
- `kube_pod_status_phase` - Pod status

### Grafana Dashboards

Access Grafana:
```bash
kubectl port-forward -n monitoring svc/prometheus-grafana 3000:80
# Default credentials: admin/prom-operator
```

### Distributed Tracing (Istio)

If using Istio service mesh:

```bash
# Access Jaeger UI
istioctl dashboard jaeger

# Access Kiali (Service Graph)
istioctl dashboard kiali
```

## Troubleshooting

### Common Issues

#### 1. Pods not starting

```bash
# Check pod status
kubectl get pods -n vcci-production

# Describe pod
kubectl describe pod <pod-name> -n vcci-production

# Check logs
kubectl logs <pod-name> -n vcci-production

# Check events
kubectl get events -n vcci-production --sort-by='.lastTimestamp'
```

#### 2. Service not accessible

```bash
# Check service
kubectl get svc -n vcci-production

# Check endpoints
kubectl get endpoints vcci-backend-api -n vcci-production

# Test connectivity from pod
kubectl run test-pod --rm -it --image=nicolaka/netshoot -- /bin/bash
curl http://vcci-backend-api:8000/health
```

#### 3. HPA not scaling

```bash
# Check HPA status
kubectl get hpa -n vcci-production

# Describe HPA
kubectl describe hpa vcci-backend-api-hpa -n vcci-production

# Check metrics server
kubectl top nodes
kubectl top pods -n vcci-production
```

#### 4. Database connection errors

```bash
# Test database connectivity
kubectl exec -it <backend-pod> -n vcci-production -- \
  psql $DATABASE_URL -c "SELECT 1"

# Check database pod
kubectl logs vcci-postgres-0 -n vcci-production

# Verify secrets
kubectl get secret postgres-credentials -n vcci-production -o yaml
```

### Rollback Procedure

```bash
# Rollback to previous version
./deployment/scripts/rollback.sh prod

# Rollback to specific revision
./deployment/scripts/rollback.sh prod 3

# Check rollout history
kubectl rollout history deployment/vcci-backend-api -n vcci-production
```

## CI/CD Integration

### GitHub Actions

```yaml
# .github/workflows/deploy.yml
name: Deploy to Production

on:
  push:
    tags:
      - 'v*'

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3

      - name: Configure kubectl
        uses: azure/k8s-set-context@v3
        with:
          kubeconfig: ${{ secrets.KUBE_CONFIG }}

      - name: Deploy
        run: |
          export VERSION=${GITHUB_REF#refs/tags/}
          ./deployment/scripts/deploy.sh -e prod -v $VERSION -s canary
```

### GitLab CI

```yaml
# .gitlab-ci.yml
deploy:production:
  stage: deploy
  image: bitnami/kubectl:latest
  script:
    - ./deployment/scripts/deploy.sh -e prod -v $CI_COMMIT_TAG -s canary
  only:
    - tags
  when: manual
```

## Best Practices

1. **Always use version tags**: Never deploy `latest` tag
2. **Test in staging first**: Deploy to staging before production
3. **Monitor deployments**: Watch metrics for at least 30 minutes
4. **Use canary for risky changes**: ML model updates, schema changes
5. **Keep rollback plan ready**: Test rollback procedure regularly
6. **Document changes**: Update CHANGELOG for each deployment
7. **Backup before deployment**: Take database snapshot
8. **Notify team**: Announce deployments in team channel
9. **Check dependencies**: Ensure external services are available
10. **Review logs**: Check for warnings/errors after deployment

## Support

For issues or questions:
- GitHub Issues: https://github.com/your-org/GL-VCCI-Carbon-APP/issues
- Team Slack: #vcci-deployment
- Email: devops@greenlang.ai

## License

Copyright (c) 2024 GreenLang. All rights reserved.
