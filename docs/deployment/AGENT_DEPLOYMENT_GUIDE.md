# GreenLang Agent Deployment Guide

## Overview

This guide covers the build and deployment process for the 4 GreenLang agents:

| Agent | ID | Description | Priority |
|-------|-----|-------------|----------|
| Fuel Analyzer | `emissions/fuel_analyzer_v1` | GHG emissions calculator using IPCC factors | Standard |
| Carbon Intensity | `cbam/carbon_intensity_v1` | CBAM carbon intensity calculator | Standard |
| Energy Performance | `building/energy_performance_v1` | Building energy performance (BPS) | Standard |
| EUDR Compliance | `regulatory/eudr_compliance_v1` | EU Deforestation Regulation compliance | CRITICAL (Deadline: 2025-12-30) |

## Prerequisites

### Required Tools

```powershell
# Check Docker
docker --version
# Expected: Docker version 29.0.1 or higher

# Check kubectl (for Kubernetes deployment)
kubectl version --client

# Optional: Trivy for security scanning
trivy --version
```

### Directory Structure

```
C:\Users\aksha\Code-V1_GreenLang\
├── docker/
│   └── base/
│       ├── Dockerfile.base          # Base image
│       └── requirements-base.txt    # Base dependencies
├── generated/
│   ├── fuel_analyzer_agent/
│   │   ├── Dockerfile
│   │   ├── requirements.txt
│   │   ├── entrypoint.py
│   │   ├── agent.py
│   │   └── tools.py
│   ├── carbon_intensity_v1/
│   │   └── ...
│   ├── energy_performance_v1/
│   │   └── ...
│   └── eudr_compliance_v1/
│       └── ...
├── k8s/
│   └── agents/
│       ├── kustomization.yaml
│       ├── namespace.yaml
│       ├── rbac.yaml
│       ├── configmap.yaml
│       ├── services.yaml
│       ├── deployment-fuel-analyzer.yaml
│       ├── deployment-carbon-intensity.yaml
│       ├── deployment-energy-performance.yaml
│       ├── deployment-eudr-compliance.yaml
│       └── hpa.yaml
└── scripts/
    ├── build-agents.ps1             # PowerShell build script
    └── build-agents.sh              # Bash build script
```

---

## Build Process

### Option 1: PowerShell Build Script (Recommended for Windows)

```powershell
# Navigate to project root
cd C:\Users\aksha\Code-V1_GreenLang

# Build all 4 agents with local tags
.\scripts\build-agents.ps1 -Local -Verify

# Build with specific tag
.\scripts\build-agents.ps1 -Tag "v1.0.0" -Local -Verify

# Build with security scan
.\scripts\build-agents.ps1 -Tag "v1.0.0" -Local -Scan -Verify

# Build and push to registry
.\scripts\build-agents.ps1 -Tag "v1.0.0" -Push
```

### Option 2: Manual Docker Build Commands

#### Step 1: Build Base Image

```powershell
# Set environment variable for BuildKit
$env:DOCKER_BUILDKIT = "1"

# Build base image
docker build `
    -t greenlang/greenlang-base:latest `
    -f docker/base/Dockerfile.base `
    docker/base/
```

#### Step 2: Build Agent Images

```powershell
# Set working directory
cd C:\Users\aksha\Code-V1_GreenLang

# Fuel Analyzer Agent
docker build `
    -t greenlang/fuel-analyzer:latest `
    -f generated/fuel_analyzer_agent/Dockerfile `
    .

# Carbon Intensity Agent
docker build `
    -t greenlang/carbon-intensity:latest `
    -f generated/carbon_intensity_v1/Dockerfile `
    .

# Energy Performance Agent
docker build `
    -t greenlang/energy-performance:latest `
    -f generated/energy_performance_v1/Dockerfile `
    .

# EUDR Compliance Agent (CRITICAL)
docker build `
    -t greenlang/eudr-compliance:latest `
    -f generated/eudr_compliance_v1/Dockerfile `
    .
```

#### Step 3: Verify Builds

```powershell
# List all GreenLang images
docker images | Select-String "greenlang"

# Check image details
docker image inspect greenlang/fuel-analyzer:latest --format '{{.Config.Labels}}'
```

---

## Image Verification

### Test Images Locally

```powershell
# Test Fuel Analyzer (port 8000)
docker run --rm -d --name test-fuel -p 8000:8000 greenlang/fuel-analyzer:latest
curl http://localhost:8000/health
docker stop test-fuel

# Test Carbon Intensity (port 8001)
docker run --rm -d --name test-carbon -p 8001:8000 greenlang/carbon-intensity:latest
curl http://localhost:8001/health
docker stop test-carbon

# Test Energy Performance (port 8002)
docker run --rm -d --name test-energy -p 8002:8000 greenlang/energy-performance:latest
curl http://localhost:8002/health
docker stop test-energy

# Test EUDR Compliance (port 8003)
docker run --rm -d --name test-eudr -p 8003:8000 greenlang/eudr-compliance:latest
curl http://localhost:8003/health
docker stop test-eudr
```

### Security Scanning

```powershell
# Install Trivy (if not already installed)
# choco install trivy

# Scan for HIGH/CRITICAL vulnerabilities
trivy image --severity HIGH,CRITICAL greenlang/fuel-analyzer:latest
trivy image --severity HIGH,CRITICAL greenlang/carbon-intensity:latest
trivy image --severity HIGH,CRITICAL greenlang/energy-performance:latest
trivy image --severity HIGH,CRITICAL greenlang/eudr-compliance:latest
```

---

## Kubernetes Deployment

### Prerequisites

1. Kubernetes cluster access (kubectl configured)
2. Docker images pushed to registry (or available locally in cluster)
3. ConfigMaps for EUDR reference data (if using EUDR agent)

### Step 1: Apply Kustomization

```powershell
# Apply all K8s manifests using Kustomize
kubectl apply -k k8s/agents/

# Or apply individual manifests
kubectl apply -f k8s/agents/namespace.yaml
kubectl apply -f k8s/agents/rbac.yaml
kubectl apply -f k8s/agents/configmap.yaml
kubectl apply -f k8s/agents/services.yaml
kubectl apply -f k8s/agents/deployment-fuel-analyzer.yaml
kubectl apply -f k8s/agents/deployment-carbon-intensity.yaml
kubectl apply -f k8s/agents/deployment-energy-performance.yaml
kubectl apply -f k8s/agents/deployment-eudr-compliance.yaml
kubectl apply -f k8s/agents/hpa.yaml
```

### Step 2: Verify Deployment

```powershell
# Check namespace
kubectl get namespaces | Select-String "greenlang"

# Check deployments
kubectl get deployments -n greenlang-dev

# Check pods
kubectl get pods -n greenlang-dev -o wide

# Check services
kubectl get services -n greenlang-dev

# Check HPA status
kubectl get hpa -n greenlang-dev
```

### Step 3: Check Pod Health

```powershell
# Get pod logs
kubectl logs -l app=fuel-analyzer -n greenlang-dev --tail=50

# Describe pod for troubleshooting
kubectl describe pod -l app=fuel-analyzer -n greenlang-dev

# Check pod readiness
kubectl get pods -n greenlang-dev -o jsonpath='{range .items[*]}{.metadata.name}: {.status.conditions[?(@.type=="Ready")].status}{"\n"}{end}'
```

---

## Configuration Reference

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `LOG_LEVEL` | `INFO` | Logging level (DEBUG, INFO, WARN, ERROR) |
| `LOG_FORMAT` | `json` | Log format (json, text) |
| `METRICS_PORT` | `9090` | Prometheus metrics port |
| `GREENLANG_ENV` | `development` | Environment (development, staging, production) |
| `SERVER_PORT` | `8000` | HTTP server port |
| `CACHE_TTL_SECONDS` | `3600` | Cache TTL in seconds |

### Resource Limits

| Agent | CPU Request | CPU Limit | Memory Request | Memory Limit |
|-------|-------------|-----------|----------------|--------------|
| Fuel Analyzer | 250m | 1000m | 256Mi | 1Gi |
| Carbon Intensity | 250m | 1000m | 256Mi | 1Gi |
| Energy Performance | 250m | 1000m | 256Mi | 1Gi |
| EUDR Compliance | 500m | 2000m | 512Mi | 2Gi |

### HPA Configuration

| Agent | Min Replicas | Max Replicas | CPU Target | Memory Target |
|-------|--------------|--------------|------------|---------------|
| Fuel Analyzer | 2 | 10 | 70% | 80% |
| Carbon Intensity | 2 | 10 | 70% | 80% |
| Energy Performance | 2 | 10 | 70% | 80% |
| EUDR Compliance | 3 | 15 | 70% | 80% |

---

## Troubleshooting

### Common Issues

#### 1. Image Pull Errors

```powershell
# Check image exists locally
docker images | Select-String "greenlang"

# If using private registry, ensure secrets are configured
kubectl create secret docker-registry regcred `
    --docker-server=ghcr.io `
    --docker-username=<username> `
    --docker-password=<token> `
    -n greenlang-dev
```

#### 2. Pod CrashLoopBackOff

```powershell
# Check pod logs
kubectl logs -l app=fuel-analyzer -n greenlang-dev --previous

# Check events
kubectl get events -n greenlang-dev --sort-by='.lastTimestamp'
```

#### 3. Health Check Failures

```powershell
# Test health endpoint directly
kubectl exec -it <pod-name> -n greenlang-dev -- curl http://localhost:8000/health

# Check probe configuration
kubectl describe deployment fuel-analyzer -n greenlang-dev | Select-String -Pattern "Liveness|Readiness"
```

#### 4. Resource Quota Exceeded

```powershell
# Check resource quota
kubectl describe resourcequota -n greenlang-dev

# Adjust resource requests in deployment if needed
```

### Rollback Procedure

```powershell
# Check rollout history
kubectl rollout history deployment/fuel-analyzer -n greenlang-dev

# Rollback to previous version
kubectl rollout undo deployment/fuel-analyzer -n greenlang-dev

# Rollback to specific revision
kubectl rollout undo deployment/fuel-analyzer -n greenlang-dev --to-revision=1
```

---

## Monitoring

### Prometheus Metrics

All agents expose metrics at `/metrics` on port 9090:

```yaml
# ServiceMonitor example
apiVersion: monitoring.coreos.com/v1
kind: ServiceMonitor
metadata:
  name: greenlang-agents
  namespace: monitoring
spec:
  selector:
    matchLabels:
      app.kubernetes.io/part-of: greenlang-platform
  endpoints:
    - port: metrics
      interval: 30s
```

### Key Metrics

- `agent_requests_total` - Total requests processed
- `agent_request_duration_seconds` - Request latency histogram
- `agent_errors_total` - Error count by type
- `agent_cache_hits_total` - Cache hit/miss ratio

---

## Quick Reference

### Build Commands (PowerShell)

```powershell
# Full local build with verification
.\scripts\build-agents.ps1 -Local -Verify

# Build specific version
.\scripts\build-agents.ps1 -Tag "v1.0.0" -Local

# Build and push to registry
.\scripts\build-agents.ps1 -Tag "v1.0.0" -Push
```

### Deploy Commands

```powershell
# Deploy all agents
kubectl apply -k k8s/agents/

# Check status
kubectl get pods -n greenlang-dev

# View logs
kubectl logs -f -l app.kubernetes.io/part-of=greenlang-platform -n greenlang-dev
```

### Test Endpoints

```
http://localhost:8000/health        # Health check
http://localhost:8000/health/live   # Liveness probe
http://localhost:8000/health/ready  # Readiness probe
http://localhost:9090/metrics       # Prometheus metrics
```

---

## Document Information

- **Version**: 1.0.0
- **Last Updated**: 2025-12-03
- **Author**: GL-DevOpsEngineer
- **Status**: Production Ready
