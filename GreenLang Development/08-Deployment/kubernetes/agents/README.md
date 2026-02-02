# GreenLang Agents - Kubernetes Deployment

## Overview

This directory contains Kubernetes manifests for deploying the 3 GreenLang agents:

1. **Fuel Emissions Analyzer** (`fuel-analyzer`) - Calculates GHG emissions from fuel combustion
2. **CBAM Carbon Intensity Calculator** (`carbon-intensity`) - Calculates carbon intensity for EU CBAM compliance
3. **Building Energy Performance Calculator** (`energy-performance`) - Calculates EUI and BPS compliance

## Directory Structure

```
k8s/agents/
  namespace.yaml         # Namespace, ResourceQuota, LimitRange
  rbac.yaml              # ServiceAccount, Role, RoleBinding, NetworkPolicy
  configmap.yaml         # Shared and agent-specific configuration
  services.yaml          # ClusterIP services for all agents
  deployment-*.yaml      # Deployment for each agent
  hpa.yaml               # HorizontalPodAutoscaler for all agents
  kustomization.yaml     # Kustomize configuration
```

## Prerequisites

- Kubernetes cluster (1.25+)
- kubectl configured with cluster access
- Docker images built and pushed to registry

## Quick Start

### Using Kustomize (Recommended)

```bash
# Preview what will be applied
kubectl apply -k k8s/agents/ --dry-run=client

# Apply all manifests
kubectl apply -k k8s/agents/

# Verify deployment
kubectl get all -n greenlang-dev
```

### Using kubectl directly

```bash
# Apply in order
kubectl apply -f k8s/agents/namespace.yaml
kubectl apply -f k8s/agents/rbac.yaml
kubectl apply -f k8s/agents/configmap.yaml
kubectl apply -f k8s/agents/services.yaml
kubectl apply -f k8s/agents/deployment-fuel-analyzer.yaml
kubectl apply -f k8s/agents/deployment-carbon-intensity.yaml
kubectl apply -f k8s/agents/deployment-energy-performance.yaml
kubectl apply -f k8s/agents/hpa.yaml
```

### Using deployment script

```bash
# Linux/macOS
./scripts/deploy-agents.sh greenlang-dev

# Windows PowerShell
.\scripts\deploy-agents.ps1 -Namespace greenlang-dev
```

## Accessing Agents

### Port Forwarding (Development)

```bash
# Fuel Analyzer
kubectl port-forward svc/fuel-analyzer 8001:80 -n greenlang-dev

# Carbon Intensity
kubectl port-forward svc/carbon-intensity 8002:80 -n greenlang-dev

# Energy Performance
kubectl port-forward svc/energy-performance 8003:80 -n greenlang-dev
```

### Health Check

```bash
curl http://localhost:8001/health
curl http://localhost:8002/health
curl http://localhost:8003/health
```

### API Documentation

```bash
# OpenAPI docs
curl http://localhost:8001/docs
curl http://localhost:8002/docs
curl http://localhost:8003/docs
```

## Configuration

### Environment Variables

Shared configuration is in `configmap.yaml`:

| Variable | Description | Default |
|----------|-------------|---------|
| LOG_LEVEL | Logging level | INFO |
| LOG_FORMAT | Log format (json/text) | json |
| METRICS_PORT | Prometheus metrics port | 9090 |
| TRACING_ENABLED | Enable distributed tracing | true |

### Resource Limits

Each agent has the following resource configuration:

| Resource | Request | Limit |
|----------|---------|-------|
| CPU | 250m | 1000m |
| Memory | 256Mi | 1Gi |

### Scaling

HPA is configured with:
- Min replicas: 2
- Max replicas: 10
- Target CPU utilization: 70%
- Target Memory utilization: 80%

## Monitoring

### Prometheus Metrics

All agents expose metrics at `/metrics` on port 9090.

```bash
kubectl port-forward svc/fuel-analyzer 9091:9090 -n greenlang-dev
curl http://localhost:9091/metrics
```

### Pod Status

```bash
kubectl get pods -n greenlang-dev -w
kubectl describe pod <pod-name> -n greenlang-dev
kubectl logs <pod-name> -n greenlang-dev
```

## Troubleshooting

### Pods not starting

```bash
# Check pod status
kubectl describe pod <pod-name> -n greenlang-dev

# Check events
kubectl get events -n greenlang-dev --sort-by='.lastTimestamp'
```

### Health check failing

```bash
# Check logs
kubectl logs <pod-name> -n greenlang-dev

# Check health endpoint
kubectl exec <pod-name> -n greenlang-dev -- curl -v http://localhost:8000/health
```

### Resource issues

```bash
# Check resource usage
kubectl top pods -n greenlang-dev

# Check resource quota
kubectl describe resourcequota -n greenlang-dev
```

## Cleanup

```bash
# Delete all resources
kubectl delete -k k8s/agents/

# Or delete namespace (deletes everything in it)
kubectl delete namespace greenlang-dev
```
