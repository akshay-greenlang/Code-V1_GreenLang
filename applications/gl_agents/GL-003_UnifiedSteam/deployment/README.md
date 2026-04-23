# GL-003 UNIFIEDSTEAM - Kubernetes Deployment

Production-grade Kubernetes deployment manifests for the GL-003 UNIFIEDSTEAM Steam System Optimization Agent.

## Overview

This directory contains comprehensive Kubernetes manifests for deploying the UnifiedSteam application in a production environment with:

- High availability (3 replicas with pod anti-affinity)
- Horizontal pod autoscaling (CPU, memory, custom metrics)
- Zero-trust network security (NetworkPolicy)
- Graceful deployments (rolling updates, PDB)
- Security hardening (non-root, read-only filesystem, dropped capabilities)
- Observability (Prometheus metrics, OpenTelemetry tracing)

## Files

| File | Description |
|------|-------------|
| `deployment.yaml` | Main Deployment with security context, probes, anti-affinity |
| `service.yaml` | ClusterIP, LoadBalancer, and Ingress resources |
| `configmap.yaml` | Application configuration and feature flags |
| `secrets.yaml` | Secret templates (placeholders only) |
| `hpa.yaml` | Horizontal Pod Autoscaler with custom metrics |
| `pdb.yaml` | Pod Disruption Budget (minAvailable: 2) |
| `networkpolicy.yaml` | Ingress/egress network policies |
| `namespace.yaml` | Namespace with resource quotas and limits |
| `Dockerfile.production` | Multi-stage production Dockerfile |
| `kustomization.yaml` | Kustomize configuration |

## Prerequisites

1. **Kubernetes Cluster** (v1.25+)
2. **CNI with NetworkPolicy support** (Calico, Cilium, or Weave Net)
3. **Metrics Server** (for HPA resource metrics)
4. **Prometheus Adapter** (for custom HPA metrics)
5. **cert-manager** (for TLS certificate management)
6. **Ingress Controller** (nginx-ingress recommended)

## Quick Start

### 1. Create Namespace

```bash
kubectl apply -f namespace.yaml
```

### 2. Create Secrets

**IMPORTANT**: Replace placeholder values with real credentials before applying.

```bash
# Edit secrets.yaml with actual values
kubectl apply -f secrets.yaml
```

Or use a secrets management solution:
- HashiCorp Vault with Vault Secrets Operator
- AWS Secrets Manager with External Secrets Operator
- Sealed Secrets for GitOps

### 3. Deploy Application

Using Kustomize:
```bash
kubectl apply -k .
```

Or apply individual files:
```bash
kubectl apply -f configmap.yaml
kubectl apply -f deployment.yaml
kubectl apply -f service.yaml
kubectl apply -f hpa.yaml
kubectl apply -f pdb.yaml
kubectl apply -f networkpolicy.yaml
```

### 4. Verify Deployment

```bash
# Check pods
kubectl get pods -n greenlang-steam

# Check services
kubectl get svc -n greenlang-steam

# Check HPA
kubectl get hpa -n greenlang-steam

# Check PDB
kubectl get pdb -n greenlang-steam

# Check ingress
kubectl get ingress -n greenlang-steam
```

## Docker Image Build

### Build Production Image

```bash
docker build -f Dockerfile.production \
  --build-arg VERSION=1.0.0 \
  --build-arg BUILD_DATE=$(date -u +"%Y-%m-%dT%H:%M:%SZ") \
  --build-arg GIT_COMMIT=$(git rev-parse HEAD) \
  -t greenlang/gl-003-unifiedsteam:1.0.0 \
  ..
```

### Push to Registry

```bash
docker push greenlang/gl-003-unifiedsteam:1.0.0
```

## Configuration

### Environment Variables

Key environment variables are loaded from ConfigMap and Secrets:

| Variable | Source | Description |
|----------|--------|-------------|
| `UNIFIEDSTEAM_ENV` | Inline | Environment (production/staging) |
| `UNIFIEDSTEAM_LOG_LEVEL` | ConfigMap | Log level (DEBUG/INFO/WARNING) |
| `DATABASE_URL` | Secret | PostgreSQL connection string |
| `REDIS_PASSWORD` | Secret | Redis authentication |
| `KAFKA_USERNAME` | Secret | Kafka SASL username |
| `API_SECRET_KEY` | Secret | JWT signing key |

### Feature Flags

Feature flags are configured in `configmap.yaml`:

```yaml
feature-flags: |
  {
    "enable_real_time_optimization": true,
    "enable_predictive_maintenance": true,
    "enable_causal_analysis": true,
    "enable_explainability": true,
    "enable_automatic_control": false
  }
```

## Autoscaling

### Horizontal Pod Autoscaler

Scales based on:
- CPU utilization (target: 70%)
- Memory utilization (target: 80%)
- HTTP requests per second (target: 1000/pod)
- Kafka consumer lag (target: 10000 messages)
- Request latency P95 (target: 200ms)

```bash
# View HPA status
kubectl get hpa unifiedsteam-hpa -n greenlang-steam

# Describe for details
kubectl describe hpa unifiedsteam-hpa -n greenlang-steam
```

### Vertical Pod Autoscaler (Optional)

VPA is configured in recommendation mode by default:

```bash
# View VPA recommendations
kubectl get vpa unifiedsteam-vpa -n greenlang-steam -o yaml
```

## High Availability

### Pod Disruption Budget

Maintains minimum 2 pods available during disruptions:

```bash
# Check PDB status
kubectl get pdb unifiedsteam-pdb -n greenlang-steam

# Test disruption (dry-run)
kubectl drain <node> --dry-run=server
```

### Pod Anti-Affinity

Pods are spread across:
- Different nodes (preferred)
- Different availability zones (required)

## Security

### Network Policies

Default deny-all with explicit allow rules:
- Ingress from nginx-ingress, Prometheus
- Egress to Kafka, Redis, PostgreSQL, OPC-UA

```bash
# Verify network policies
kubectl get networkpolicy -n greenlang-steam

# Test connectivity
kubectl exec -n greenlang-steam <pod> -- nc -zv redis 6379
```

### Pod Security

- Non-root user (UID 1000)
- Read-only root filesystem
- Dropped ALL capabilities
- Seccomp profile: RuntimeDefault

## Monitoring

### Prometheus Metrics

Metrics endpoint: `http://<pod>:9090/metrics`

Key metrics:
- `http_requests_total`
- `http_request_duration_seconds`
- `optimization_queue_depth`
- `steam_trap_status`

### Health Checks

| Probe | Path | Port | Interval |
|-------|------|------|----------|
| Liveness | `/api/v1/health` | 8080 | 10s |
| Readiness | `/api/v1/ready` | 8080 | 5s |
| Startup | `/api/v1/health` | 8080 | 5s |

## Troubleshooting

### Pod Not Starting

```bash
# Check pod events
kubectl describe pod <pod-name> -n greenlang-steam

# Check init containers
kubectl logs <pod-name> -n greenlang-steam -c wait-for-kafka

# Check main container logs
kubectl logs <pod-name> -n greenlang-steam -c unifiedsteam
```

### Network Issues

```bash
# Test DNS
kubectl exec -n greenlang-steam <pod> -- nslookup kafka

# Test connectivity
kubectl exec -n greenlang-steam <pod> -- curl -v http://redis:6379

# Check network policies
kubectl get networkpolicy -n greenlang-steam -o yaml
```

### Resource Issues

```bash
# Check resource usage
kubectl top pods -n greenlang-steam

# Check resource quotas
kubectl describe resourcequota -n greenlang-steam

# Check limit ranges
kubectl describe limitrange -n greenlang-steam
```

## Multi-Environment Deployment

Use Kustomize overlays for different environments:

```
deployment/
  base/
    kustomization.yaml
    deployment.yaml
    ...
  overlays/
    development/
      kustomization.yaml
      patches/
    staging/
      kustomization.yaml
      patches/
    production/
      kustomization.yaml
      patches/
```

Deploy to specific environment:
```bash
kubectl apply -k overlays/production
```

## Rollback

```bash
# View rollout history
kubectl rollout history deployment/unifiedsteam -n greenlang-steam

# Rollback to previous version
kubectl rollout undo deployment/unifiedsteam -n greenlang-steam

# Rollback to specific revision
kubectl rollout undo deployment/unifiedsteam -n greenlang-steam --to-revision=2
```

## Support

- Documentation: https://docs.greenlang.io/agents/gl-003
- Issues: https://github.com/greenlang/gl-003-unifiedsteam/issues
- Team: steam-team@greenlang.io
