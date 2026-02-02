# GL-017 CONDENSYNC Kubernetes Deployment Guide

## Production Deployment Documentation

**Version:** 1.0.0
**Last Updated:** December 2025
**Target Platform:** Kubernetes 1.28+

---

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Deployment Architecture](#deployment-architecture)
3. [Quick Start](#quick-start)
4. [Detailed Deployment Steps](#detailed-deployment-steps)
5. [Configuration](#configuration)
6. [Monitoring Setup](#monitoring-setup)
7. [Scaling](#scaling)
8. [Troubleshooting](#troubleshooting)
9. [Maintenance](#maintenance)

---

## Prerequisites

### Required Tools

| Tool | Version | Purpose |
|------|---------|---------|
| `kubectl` | 1.28+ | Kubernetes CLI |
| `helm` | 3.12+ | Package manager |
| `kustomize` | 5.0+ | Configuration management |
| `docker` | 24.0+ | Container runtime |

### Cluster Requirements

| Resource | Minimum | Recommended |
|----------|---------|-------------|
| Kubernetes | 1.28+ | 1.29+ |
| Nodes | 3 | 5+ |
| CPU per node | 4 cores | 8 cores |
| Memory per node | 16 GB | 32 GB |
| Storage | 100 GB SSD | 500 GB SSD |

### External Dependencies

| Service | Version | Required |
|---------|---------|----------|
| PostgreSQL | 15+ | Yes |
| Redis | 7+ | Yes |
| InfluxDB | 2.7+ | Yes |
| Kafka | 3.5+ | Yes |
| HashiCorp Vault | 1.15+ | Yes |

### Network Requirements

| Port | Protocol | Purpose |
|------|----------|---------|
| 8017 | TCP | HTTP API |
| 9017 | TCP | Prometheus metrics |
| 4840 | TCP | OPC-UA (outbound) |
| 502 | TCP | Modbus TCP (outbound) |

---

## Deployment Architecture

```
+==============================================================================+
|                      KUBERNETES DEPLOYMENT ARCHITECTURE                       |
+==============================================================================+

  NAMESPACE: greenlang
  +------------------------------------------------------------------------+
  |                                                                        |
  |  INGRESS                                                               |
  |  +------------------------------------------------------------------+ |
  |  | Host: gl-017.greenlang.io                                        | |
  |  | TLS: enabled (cert-manager)                                      | |
  |  | Annotations: nginx.ingress.kubernetes.io/*                       | |
  |  +------------------------------+-----------------------------------+ |
  |                                 |                                     |
  |  SERVICE                        v                                     |
  |  +------------------------------------------------------------------+ |
  |  | Name: gl-017-condensync                                          | |
  |  | Type: ClusterIP                                                  | |
  |  | Ports: 8017 (http), 9017 (metrics)                               | |
  |  +------------------------------+-----------------------------------+ |
  |                                 |                                     |
  |  DEPLOYMENT (replicas: 3)       v                                     |
  |  +------------------------------------------------------------------+ |
  |  |  +----------------+  +----------------+  +----------------+      | |
  |  |  |    POD #1      |  |    POD #2      |  |    POD #3      |      | |
  |  |  | condensync     |  | condensync     |  | condensync     |      | |
  |  |  | :8017, :9017   |  | :8017, :9017   |  | :8017, :9017   |      | |
  |  |  +----------------+  +----------------+  +----------------+      | |
  |  +------------------------------------------------------------------+ |
  |                                 |                                     |
  |  HORIZONTAL POD AUTOSCALER      v                                     |
  |  +------------------------------------------------------------------+ |
  |  | Min: 2 | Max: 10 | CPU Target: 70% | Memory Target: 80%          | |
  |  +------------------------------------------------------------------+ |
  |                                                                        |
  |  CONFIGMAP                      SECRETS                               |
  |  +-------------------------+    +-------------------------+           |
  |  | gl-017-condensync-config|    | gl-017-condensync-secrets|          |
  |  | - Application config    |    | - Database credentials   |          |
  |  | - Feature flags         |    | - API keys               |          |
  |  +-------------------------+    +-------------------------+           |
  |                                                                        |
  |  PVC                            SERVICE ACCOUNT                       |
  |  +-------------------------+    +-------------------------+           |
  |  | gl-017-data-pvc (50Gi)  |    | gl-017-condensync-sa     |          |
  |  | gl-017-models-pvc (10Gi)|    | - RBAC roles             |          |
  |  +-------------------------+    +-------------------------+           |
  |                                                                        |
  +------------------------------------------------------------------------+

+==============================================================================+
```

---

## Quick Start

### Option 1: Kustomize Deployment

```bash
# Clone repository
git clone https://github.com/greenlang/gl-017-condensync.git
cd gl-017-condensync/deployment

# Create namespace
kubectl create namespace greenlang

# Create secrets (edit first!)
cp secret.yaml.example secret.yaml
nano secret.yaml
kubectl apply -f secret.yaml

# Deploy with kustomize
kubectl apply -k .

# Verify deployment
kubectl -n greenlang get pods -l app=gl-017-condensync
kubectl -n greenlang rollout status deployment/gl-017-condensync
```

### Option 2: Helm Chart

```bash
# Add GreenLang Helm repository
helm repo add greenlang https://charts.greenlang.io
helm repo update

# Install with custom values
helm install gl-017-condensync greenlang/condensync \
  --namespace greenlang \
  --create-namespace \
  --values values-production.yaml

# Verify
helm status gl-017-condensync -n greenlang
```

---

## Detailed Deployment Steps

### Step 1: Create Namespace

```bash
kubectl create namespace greenlang
kubectl label namespace greenlang name=greenlang
```

### Step 2: Deploy Secrets

Create the secrets file from template:

```bash
# Copy template
cp secret.yaml.example secret.yaml
```

Edit `secret.yaml` with your credentials:

```yaml
apiVersion: v1
kind: Secret
metadata:
  name: gl-017-condensync-secrets
  namespace: greenlang
type: Opaque
stringData:
  database-url: "postgresql://user:password@postgres:5432/condensync"
  redis-url: "redis://:password@redis:6379/0"
  scada-api-key: "your-scada-api-key"
  influxdb-token: "your-influxdb-token"
  kafka-password: "your-kafka-password"
```

Apply secrets:

```bash
kubectl apply -f secret.yaml
```

### Step 3: Deploy ConfigMap

```bash
kubectl apply -f configmap.yaml
```

Verify configuration:

```bash
kubectl -n greenlang describe configmap gl-017-condensync-config
```

### Step 4: Create Service Account

```bash
kubectl apply -f serviceaccount.yaml
```

### Step 5: Deploy Persistent Volume Claims

```bash
kubectl apply -f pvc.yaml
```

Verify PVCs are bound:

```bash
kubectl -n greenlang get pvc
```

Expected output:

```
NAME                        STATUS   VOLUME   CAPACITY   ACCESS MODES
gl-017-condensync-data-pvc  Bound    pv-xxx   50Gi       RWO
gl-017-condensync-models-pvc Bound   pv-yyy   10Gi       RWO
```

### Step 6: Deploy Network Policy

```bash
kubectl apply -f networkpolicy.yaml
```

### Step 7: Deploy Main Application

```bash
kubectl apply -f deployment.yaml
```

Wait for rollout:

```bash
kubectl -n greenlang rollout status deployment/gl-017-condensync --timeout=300s
```

### Step 8: Deploy Service

```bash
kubectl apply -f service.yaml
```

Verify service:

```bash
kubectl -n greenlang get svc gl-017-condensync
```

### Step 9: Deploy Ingress

```bash
kubectl apply -f ingress.yaml
```

### Step 10: Deploy Horizontal Pod Autoscaler

```bash
kubectl apply -f hpa.yaml
```

Verify HPA:

```bash
kubectl -n greenlang get hpa gl-017-condensync-hpa
```

---

## Configuration

### ConfigMap Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `LOG_LEVEL` | Logging level | `INFO` |
| `METRICS_ENABLED` | Enable Prometheus metrics | `true` |
| `METRICS_PORT` | Metrics port | `9017` |
| `CACHE_TTL_SECONDS` | Redis cache TTL | `300` |
| `DETERMINISTIC_MODE` | Zero-hallucination mode | `true` |
| `MAX_WORKERS` | Parallel workers | `6` |
| `BATCH_SIZE` | Batch processing size | `200` |

### Environment Variables

```yaml
env:
  - name: AGENT_ID
    value: "GL-017"
  - name: AGENT_NAME
    value: "CONDENSYNC"
  - name: ENVIRONMENT
    value: "production"
  - name: DATABASE_URL
    valueFrom:
      secretKeyRef:
        name: gl-017-condensync-secrets
        key: database-url
```

### Resource Requests/Limits

```yaml
resources:
  requests:
    cpu: 250m
    memory: 512Mi
    ephemeral-storage: 100Mi
  limits:
    cpu: 1000m
    memory: 2Gi
    ephemeral-storage: 1Gi
```

---

## Monitoring Setup

### Prometheus ServiceMonitor

Create a ServiceMonitor for automatic scraping:

```yaml
apiVersion: monitoring.coreos.com/v1
kind: ServiceMonitor
metadata:
  name: gl-017-condensync
  namespace: greenlang
  labels:
    app: gl-017-condensync
spec:
  selector:
    matchLabels:
      app: gl-017-condensync
  endpoints:
    - port: metrics
      interval: 30s
      path: /metrics
  namespaceSelector:
    matchNames:
      - greenlang
```

### Grafana Dashboard

Import the pre-built dashboard:

```bash
# Download dashboard JSON
curl -o condensync-dashboard.json \
  https://raw.githubusercontent.com/greenlang/gl-017-condensync/main/monitoring/grafana-dashboard.json

# Import via Grafana API
curl -X POST -H "Content-Type: application/json" \
  -d @condensync-dashboard.json \
  http://grafana:3000/api/dashboards/db
```

### Alert Rules

Apply Prometheus alert rules:

```yaml
apiVersion: monitoring.coreos.com/v1
kind: PrometheusRule
metadata:
  name: gl-017-condensync-alerts
  namespace: greenlang
spec:
  groups:
    - name: condensync-alerts
      rules:
        - alert: CondenSyncHighVacuumDeviation
          expr: gl017_vacuum_deviation_kpa > 3.0
          for: 5m
          labels:
            severity: critical
          annotations:
            summary: "Critical vacuum deviation on {{ $labels.condenser_id }}"

        - alert: CondenSyncLowCleanliness
          expr: gl017_cleanliness_factor_percent < 60
          for: 15m
          labels:
            severity: warning
          annotations:
            summary: "Low cleanliness factor on {{ $labels.condenser_id }}"
```

---

## Scaling

### Manual Scaling

```bash
# Scale to 5 replicas
kubectl -n greenlang scale deployment/gl-017-condensync --replicas=5

# Verify
kubectl -n greenlang get pods -l app=gl-017-condensync
```

### Autoscaling Configuration

The HPA is configured to scale based on:

| Metric | Target | Scale Up | Scale Down |
|--------|--------|----------|------------|
| CPU | 70% | 60s stabilization | 300s stabilization |
| Memory | 80% | 60s stabilization | 300s stabilization |
| Custom (calc/sec) | 100 | 60s stabilization | 300s stabilization |

Modify HPA:

```bash
kubectl -n greenlang edit hpa gl-017-condensync-hpa
```

### Vertical Pod Autoscaler (Optional)

```yaml
apiVersion: autoscaling.k8s.io/v1
kind: VerticalPodAutoscaler
metadata:
  name: gl-017-condensync-vpa
  namespace: greenlang
spec:
  targetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: gl-017-condensync
  updatePolicy:
    updateMode: Auto
  resourcePolicy:
    containerPolicies:
      - containerName: condensync
        minAllowed:
          cpu: 100m
          memory: 256Mi
        maxAllowed:
          cpu: 4000m
          memory: 8Gi
```

---

## Troubleshooting

### Common Issues

#### 1. Pods Not Starting

```bash
# Check pod status
kubectl -n greenlang get pods -l app=gl-017-condensync

# Describe pod for events
kubectl -n greenlang describe pod <pod-name>

# Check logs
kubectl -n greenlang logs <pod-name> --previous
```

**Common causes:**
- Missing secrets
- PVC not bound
- Resource limits too low
- Image pull error

#### 2. Health Check Failures

```bash
# Check endpoints
kubectl -n greenlang get endpoints gl-017-condensync

# Test health endpoints
kubectl -n greenlang port-forward svc/gl-017-condensync 8017:8017
curl http://localhost:8017/health/ready
```

**Common causes:**
- Database connection failed
- Redis not available
- SCADA connection timeout

#### 3. High Memory Usage

```bash
# Check resource usage
kubectl -n greenlang top pods -l app=gl-017-condensync

# Get detailed metrics
kubectl -n greenlang describe pod <pod-name> | grep -A 10 "Containers:"
```

**Resolution:**
- Increase memory limits
- Check for memory leaks
- Reduce cache size

#### 4. SCADA Connection Issues

```bash
# Check network connectivity
kubectl -n greenlang exec -it <pod-name> -- nc -zv scada-server 4840

# Check logs for OPC-UA errors
kubectl -n greenlang logs <pod-name> | grep -i "opc"
```

### Debug Mode

Enable debug logging:

```bash
kubectl -n greenlang set env deployment/gl-017-condensync LOG_LEVEL=DEBUG
```

### Log Analysis

```bash
# Stream logs from all pods
kubectl -n greenlang logs -l app=gl-017-condensync -f

# Search for errors
kubectl -n greenlang logs -l app=gl-017-condensync | grep -i error

# Export logs
kubectl -n greenlang logs -l app=gl-017-condensync > condensync-logs.txt
```

---

## Maintenance

### Rolling Update

```bash
# Update image
kubectl -n greenlang set image deployment/gl-017-condensync \
  condensync=greenlang/gl-017-condensync:1.1.0

# Monitor rollout
kubectl -n greenlang rollout status deployment/gl-017-condensync
```

### Rollback

```bash
# View rollout history
kubectl -n greenlang rollout history deployment/gl-017-condensync

# Rollback to previous version
kubectl -n greenlang rollout undo deployment/gl-017-condensync

# Rollback to specific revision
kubectl -n greenlang rollout undo deployment/gl-017-condensync --to-revision=2
```

### Backup and Restore

```bash
# Backup all resources
kubectl -n greenlang get all -o yaml > backup-all.yaml
kubectl -n greenlang get configmap,secret -o yaml > backup-configs.yaml

# Restore
kubectl apply -f backup-all.yaml
```

### Certificate Renewal

```bash
# Check certificate status
kubectl -n greenlang get certificate

# Force renewal
kubectl -n greenlang delete certificate gl-017-condensync-tls
kubectl apply -f ingress.yaml
```

---

## Deployment Files Reference

| File | Description |
|------|-------------|
| `deployment.yaml` | Main deployment spec |
| `service.yaml` | ClusterIP service |
| `configmap.yaml` | Application configuration |
| `secret.yaml` | Credentials (template) |
| `hpa.yaml` | Horizontal Pod Autoscaler |
| `networkpolicy.yaml` | Network access control |
| `ingress.yaml` | External access |
| `pvc.yaml` | Persistent storage |
| `serviceaccount.yaml` | RBAC configuration |
| `kustomization.yaml` | Kustomize orchestration |

---

## Health Check Endpoints

| Endpoint | Purpose | Expected Response |
|----------|---------|-------------------|
| `/health/live` | Liveness probe | `200 OK` |
| `/health/ready` | Readiness probe | `200 OK` if dependencies ready |
| `/health/startup` | Startup probe | `200 OK` after initialization |

---

## Support

- **Documentation:** https://docs.greenlang.io/agents/GL-017/deployment
- **Issues:** https://github.com/greenlang/gl-017-condensync/issues
- **Slack:** #gl-017-condensync

---

*GL-017 CONDENSYNC Deployment Guide - Version 1.0.0*
