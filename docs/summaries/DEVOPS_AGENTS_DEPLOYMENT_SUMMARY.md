# GreenLang Agents - DevOps Deployment Summary

## Overview

This document summarizes the Docker and Kubernetes deployment infrastructure created for the 3 GreenLang agents:

1. **Fuel Emissions Analyzer** (`emissions/fuel_analyzer_v1`)
2. **CBAM Carbon Intensity Calculator** (`cbam/carbon_intensity_v1`)
3. **Building Energy Performance Calculator** (`building/energy_performance_v1`)

## Created Files Summary

### Docker Configuration

| File | Purpose |
|------|---------|
| `docker/base/Dockerfile.base` | Multi-stage base image with Python 3.11, non-root user (UID 1000) |
| `docker/base/requirements-base.txt` | Base dependencies for all agents |
| `generated/fuel_analyzer_agent/Dockerfile` | Fuel Analyzer agent image |
| `generated/fuel_analyzer_agent/requirements.txt` | Fuel Analyzer dependencies |
| `generated/fuel_analyzer_agent/entrypoint.py` | FastAPI entrypoint with health checks |
| `generated/fuel_analyzer_agent/.dockerignore` | Docker build exclusions |
| `generated/carbon_intensity_v1/Dockerfile` | Carbon Intensity agent image |
| `generated/carbon_intensity_v1/requirements.txt` | Carbon Intensity dependencies |
| `generated/carbon_intensity_v1/entrypoint.py` | FastAPI entrypoint with health checks |
| `generated/carbon_intensity_v1/.dockerignore` | Docker build exclusions |
| `generated/energy_performance_v1/Dockerfile` | Energy Performance agent image |
| `generated/energy_performance_v1/requirements.txt` | Energy Performance dependencies |
| `generated/energy_performance_v1/entrypoint.py` | FastAPI entrypoint with health checks |
| `generated/energy_performance_v1/.dockerignore` | Docker build exclusions |
| `docker-compose.agents.yml` | Local development orchestration |

### Kubernetes Manifests

| File | Purpose |
|------|---------|
| `k8s/agents/namespace.yaml` | Namespace, ResourceQuota, LimitRange |
| `k8s/agents/rbac.yaml` | ServiceAccount, Role, RoleBinding, NetworkPolicy |
| `k8s/agents/configmap.yaml` | Shared and agent-specific configuration |
| `k8s/agents/services.yaml` | ClusterIP services for all agents |
| `k8s/agents/deployment-fuel-analyzer.yaml` | Fuel Analyzer Deployment |
| `k8s/agents/deployment-carbon-intensity.yaml` | Carbon Intensity Deployment |
| `k8s/agents/deployment-energy-performance.yaml` | Energy Performance Deployment |
| `k8s/agents/hpa.yaml` | HorizontalPodAutoscaler + PodDisruptionBudget |
| `k8s/agents/kustomization.yaml` | Kustomize configuration |
| `k8s/agents/README.md` | Deployment documentation |

### Scripts

| File | Purpose |
|------|---------|
| `scripts/build-agents.sh` | Build script (Linux/macOS) |
| `scripts/build-agents.ps1` | Build script (Windows PowerShell) |
| `scripts/deploy-agents.sh` | Deployment script (Linux/macOS) |
| `scripts/deploy-agents.ps1` | Deployment script (Windows PowerShell) |

### Monitoring

| File | Purpose |
|------|---------|
| `monitoring/prometheus/prometheus.yml` | Prometheus scrape configuration |

## Docker Configuration Details

### Base Image Features

- **Base**: `python:3.11-slim`
- **Multi-stage build**: Optimized image size (~250MB target)
- **Non-root user**: UID 1000, GID 1000
- **Security**: Read-only filesystem support, minimal packages
- **Health check**: Built-in curl-based health check

### Agent Image Features

- **Multi-stage build**: Build and runtime separation
- **Health endpoints**: `/health`, `/health/live`, `/health/ready`
- **Metrics endpoint**: `/metrics` on port 9090
- **API endpoint**: `/api/v1/execute` on port 8000
- **OpenAPI docs**: `/docs`, `/redoc`

## Kubernetes Configuration Details

### Namespace: `greenlang-dev`

- Pod Security Standards: `restricted`
- Resource Quota: 10 CPU, 10Gi memory
- Limit Range: 50m-2 CPU, 64Mi-4Gi memory per container

### Deployments

| Agent | Replicas | CPU Request | Memory Request | CPU Limit | Memory Limit |
|-------|----------|-------------|----------------|-----------|--------------|
| fuel-analyzer | 2 | 250m | 256Mi | 1000m | 1Gi |
| carbon-intensity | 2 | 250m | 256Mi | 1000m | 1Gi |
| energy-performance | 2 | 250m | 256Mi | 1000m | 1Gi |

### Services

All services are ClusterIP type:
- `fuel-analyzer`: Port 80 -> 8000, Port 9090 -> 9090
- `carbon-intensity`: Port 80 -> 8000, Port 9090 -> 9090
- `energy-performance`: Port 80 -> 8000, Port 9090 -> 9090

### HPA Configuration

| Metric | Target |
|--------|--------|
| CPU Utilization | 70% |
| Memory Utilization | 80% |
| Min Replicas | 2 |
| Max Replicas | 10 |
| Scale Up Stabilization | 60s |
| Scale Down Stabilization | 300s |

### Security Features

- Non-root user (UID 1000)
- Read-only root filesystem
- No privilege escalation
- All capabilities dropped
- Seccomp profile: RuntimeDefault
- Network policies for ingress/egress control

## Quick Start Commands

### Local Development (Docker Compose)

```bash
# Start all agents
docker compose -f docker-compose.agents.yml up -d

# View logs
docker compose -f docker-compose.agents.yml logs -f

# Stop
docker compose -f docker-compose.agents.yml down
```

### Build Images

```bash
# Linux/macOS
./scripts/build-agents.sh v1.0.0 --push --scan

# Windows PowerShell
.\scripts\build-agents.ps1 -Tag "v1.0.0" -Push -Scan
```

### Deploy to Kubernetes

```bash
# Using Kustomize
kubectl apply -k k8s/agents/

# Using script (Linux/macOS)
./scripts/deploy-agents.sh greenlang-dev

# Using script (Windows PowerShell)
.\scripts\deploy-agents.ps1 -Namespace greenlang-dev
```

### Access Agents (Port Forwarding)

```bash
kubectl port-forward svc/fuel-analyzer 8001:80 -n greenlang-dev
kubectl port-forward svc/carbon-intensity 8002:80 -n greenlang-dev
kubectl port-forward svc/energy-performance 8003:80 -n greenlang-dev
```

### Health Checks

```bash
curl http://localhost:8001/health
curl http://localhost:8002/health
curl http://localhost:8003/health
```

## Next Steps

1. **CI/CD Pipeline**: Add GitHub Actions workflow for automated builds
2. **Ingress**: Configure Ingress controller for external access
3. **TLS**: Add cert-manager for automatic TLS certificates
4. **Secrets Management**: Integrate External Secrets Operator
5. **Monitoring**: Deploy Prometheus and Grafana
6. **Logging**: Configure Fluent Bit for log aggregation
7. **Tracing**: Deploy OpenTelemetry Collector

## Image Registry

Images should be pushed to:
- `ghcr.io/greenlang/fuel-analyzer:latest`
- `ghcr.io/greenlang/carbon-intensity:latest`
- `ghcr.io/greenlang/energy-performance:latest`

---

**Created**: 2025-12-03
**Author**: GL-DevOpsEngineer
**Version**: 1.0.0
