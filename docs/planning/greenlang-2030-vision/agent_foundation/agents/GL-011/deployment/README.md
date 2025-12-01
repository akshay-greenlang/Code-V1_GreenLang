# GL-011 FUELCRAFT - Kubernetes Deployment Guide

## Overview

GL-011 FUELCRAFT is a fuel mix optimization and emissions management agent that provides real-time fuel optimization recommendations for industrial processes.

**Agent Details:**
- **Agent ID:** GL-011
- **Name:** FUELCRAFT
- **Category:** Fuel Optimization
- **Version:** 1.0.0
- **Tier:** Tier-2 (Production Critical)

## Quick Start

### Prerequisites

- Kubernetes 1.24+
- kubectl configured with cluster access
- Helm 3.x (optional, for Helm deployments)
- kustomize (optional, for environment overlays)

### Deploy to Production

```bash
# Validate manifests first
./scripts/validate-manifests.sh

# Deploy using Kustomize
kubectl apply -k kustomize/overlays/production

# Or deploy using the script
./scripts/deploy.sh production
```

### Deploy to Staging

```bash
./scripts/deploy.sh staging
```

### Deploy to Development

```bash
./scripts/deploy.sh dev
```

## Directory Structure

```
deployment/
|-- deployment.yaml         # Main Kubernetes Deployment
|-- service.yaml            # ClusterIP Service
|-- configmap.yaml          # Configuration data (fuel specs, emission factors)
|-- secret.yaml             # Secrets template (API keys, database credentials)
|-- hpa.yaml                # Horizontal Pod Autoscaler (2-10 replicas)
|-- pdb.yaml                # Pod Disruption Budget (min 2 available)
|-- networkpolicy.yaml      # Network security rules
|-- ingress.yaml            # Ingress controller configuration
|-- serviceaccount.yaml     # RBAC configuration
|-- servicemonitor.yaml     # Prometheus ServiceMonitor
|-- resourcequota.yaml      # Resource limits for namespace
|-- limitrange.yaml         # Default resource constraints
|-- Dockerfile              # Standard Docker image
|-- Dockerfile.production   # Production-optimized Docker image
|-- kustomize/
|   |-- base/
|   |   +-- kustomization.yaml
|   +-- overlays/
|       |-- dev/
|       |-- staging/
|       +-- production/
+-- scripts/
    |-- deploy.sh           # Deployment script
    |-- rollback.sh         # Rollback script
    +-- validate-manifests.sh # Manifest validation
```

## Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `GREENLANG_ENV` | Environment (dev/staging/production) | production |
| `LOG_LEVEL` | Logging level | INFO |
| `DATABASE_URL` | PostgreSQL connection string | (secret) |
| `REDIS_URL` | Redis connection string | (secret) |
| `API_KEY` | API authentication key | (secret) |
| `OPTIMIZATION_TIMEOUT` | Timeout for optimization requests | 300 |
| `ENABLE_EMISSIONS_TRACKING` | Enable emissions tracking | true |
| `ENABLE_COST_OPTIMIZATION` | Enable cost optimization | true |

### Secrets Setup

Before deploying, create the required secrets:

```bash
kubectl create secret generic gl-011-secrets \
  --from-literal=database_url="postgresql://user:pass@host:5432/db" \
  --from-literal=redis_url="redis://:pass@host:6379/0" \
  --from-literal=api_key="your-api-key" \
  --from-literal=jwt_secret="your-jwt-secret" \
  --from-literal=market_data_api_key="your-market-api-key" \
  -n greenlang
```

### Resource Requirements

| Environment | CPU Request | CPU Limit | Memory Request | Memory Limit | Replicas |
|-------------|-------------|-----------|----------------|--------------|----------|
| Development | 250m | 1000m | 512Mi | 2Gi | 1 |
| Staging | 500m | 1500m | 1Gi | 3Gi | 2 |
| Production | 1000m | 2000m | 2Gi | 4Gi | 3 |

## Deployment Operations

### Scaling

The HPA automatically scales between 2-10 replicas based on:
- CPU utilization (target: 70%)
- Memory utilization (target: 80%)

Manual scaling:
```bash
kubectl scale deployment gl-011-fuelcraft --replicas=5 -n greenlang
```

### Rollback

Rollback to previous version:
```bash
./scripts/rollback.sh production
```

Rollback to specific revision:
```bash
./scripts/rollback.sh production 5
```

Emergency rollback (no confirmation):
```bash
EMERGENCY=true ./scripts/rollback.sh production
```

### Health Checks

Check deployment status:
```bash
kubectl get deployment gl-011-fuelcraft -n greenlang
kubectl get pods -l app=gl-011-fuelcraft -n greenlang
```

Check health endpoint:
```bash
kubectl port-forward svc/gl-011-fuelcraft 8080:8080 -n greenlang
curl http://localhost:8080/api/v1/health
```

### Logs

View logs:
```bash
kubectl logs -l app=gl-011-fuelcraft -n greenlang -f
```

View logs for specific pod:
```bash
kubectl logs gl-011-fuelcraft-xxxxx -n greenlang
```

## Monitoring

### Prometheus Metrics

Metrics are exposed on port 9090 at `/api/v1/metrics`.

Key metrics:
- `gl_011_http_requests_total` - Total HTTP requests
- `gl_011_http_request_duration_seconds` - Request latency
- `gl_011_fuel_optimization_total` - Total optimization requests
- `gl_011_fuel_optimization_success_total` - Successful optimizations
- `gl_011_emissions_actual` - Current emissions levels
- `gl_011_fuel_cost_actual` - Current fuel costs

### Alerts

Configured alerts (see servicemonitor.yaml):
- `GL011Down` - Agent is down (critical)
- `GL011HighErrorRate` - Error rate > 5% (critical)
- `GL011FuelOptimizationFailures` - Optimization failures (critical)
- `GL011HighLatency` - p95 latency > 2s (warning)
- `GL011HighCPU` - CPU > 85% (warning)
- `GL011HighMemory` - Memory > 85% (warning)
- `GL011EmissionsExceedance` - Emissions limit exceeded (warning)

### Grafana Dashboards

Import the dashboard JSON from `servicemonitor.yaml` into Grafana for:
- Request rate and latency
- Fuel optimization success rate
- Emissions vs limits
- Cost optimization performance

## Security

### Network Policies

GL-011 is configured with zero-trust networking:
- **Ingress:** Only from ingress controller, Prometheus, and authorized agents
- **Egress:** Only to database, Redis, and external APIs (HTTPS only)

### RBAC

Minimal permissions following least-privilege principle:
- Read ConfigMaps and Secrets (own resources only)
- Read Pods and Services for health checks
- Create Events for logging

### Security Context

All containers run with:
- Non-root user (UID 1000)
- Read-only root filesystem
- No privilege escalation
- Dropped capabilities (except NET_BIND_SERVICE)

## Troubleshooting

### Pod Not Starting

1. Check events:
   ```bash
   kubectl describe pod <pod-name> -n greenlang
   kubectl get events -n greenlang --sort-by='.lastTimestamp'
   ```

2. Check secrets exist:
   ```bash
   kubectl get secret gl-011-secrets -n greenlang
   ```

3. Check resource quota:
   ```bash
   kubectl describe resourcequota -n greenlang
   ```

### Health Check Failing

1. Check logs:
   ```bash
   kubectl logs <pod-name> -n greenlang
   ```

2. Check database connectivity:
   ```bash
   kubectl exec -it <pod-name> -n greenlang -- nc -zv postgresql.database 5432
   ```

### High Latency

1. Check HPA status:
   ```bash
   kubectl get hpa gl-011-fuelcraft-hpa -n greenlang
   ```

2. Check resource usage:
   ```bash
   kubectl top pods -l app=gl-011-fuelcraft -n greenlang
   ```

## Contact

- **Team:** Fuel Optimization Team
- **Slack:** #gl-011-alerts
- **PagerDuty:** GL-011-FUELCRAFT
- **Email:** gl-011-oncall@greenlang.ai
- **Documentation:** https://docs.greenlang.ai/agents/GL-011
