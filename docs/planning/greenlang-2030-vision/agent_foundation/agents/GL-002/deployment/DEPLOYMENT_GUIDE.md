# GL-002 BoilerEfficiencyOptimizer - Deployment Guide

## Table of Contents

1. [Overview](#overview)
2. [Prerequisites](#prerequisites)
3. [Architecture](#architecture)
4. [Pre-Deployment Checklist](#pre-deployment-checklist)
5. [Deployment Methods](#deployment-methods)
6. [Environment-Specific Deployment](#environment-specific-deployment)
7. [Verification](#verification)
8. [Rollback Procedures](#rollback-procedures)
9. [Troubleshooting](#troubleshooting)
10. [Monitoring and Observability](#monitoring-and-observability)
11. [Security Best Practices](#security-best-practices)
12. [Maintenance and Updates](#maintenance-and-updates)

---

## Overview

GL-002 BoilerEfficiencyOptimizer is a production-grade Kubernetes application designed for industrial boiler optimization. This guide provides comprehensive instructions for deploying, managing, and troubleshooting GL-002 across different environments (dev, staging, production).

**Key Features:**
- High Availability (HA) with 3+ replicas
- Horizontal Pod Autoscaling (HPA) from 3 to 10 pods
- Zero-downtime rolling updates
- Comprehensive health checks (liveness, readiness, startup)
- Production-grade security (RBAC, NetworkPolicies, SecurityContext)
- Observability (Prometheus metrics, ServiceMonitor)

---

## Prerequisites

### Required Tools

1. **kubectl** (v1.24+)
   ```bash
   kubectl version --client
   ```
   Install: https://kubernetes.io/docs/tasks/tools/

2. **Kubernetes Cluster** (v1.24+)
   - Minimum: 3 nodes (for HA)
   - CPU: 20+ cores total
   - Memory: 40+ GiB total
   - Storage: 500+ GiB

3. **Access Credentials**
   - kubectl config with cluster admin or namespace admin permissions
   - Verify: `kubectl cluster-info`

### Optional Tools

1. **kubeval** (YAML validation)
   ```bash
   kubeval --version
   ```
   Install: https://kubeval.com

2. **kustomize** (configuration management)
   ```bash
   kustomize version
   ```
   Install: https://kustomize.io

3. **helm** (package management - if using Helm charts)
   ```bash
   helm version
   ```
   Install: https://helm.sh

### Infrastructure Dependencies

1. **PostgreSQL Database** (v14+)
   - Host: `postgresql.database.svc.cluster.local:5432`
   - Database: `greenlang`
   - User: `greenlang_admin`
   - Schema: initialized with migrations

2. **Redis Cache** (v7+)
   - Host: `redis.cache.svc.cluster.local:6379`
   - DB: 0 (default)

3. **Container Registry**
   - GCR: `gcr.io/greenlang/gl-002-boiler-efficiency`
   - Image tags: `1.0.0`, `staging-1.0.0-rc.1`, `dev-latest`

4. **DNS and TLS**
   - Production: `api.boiler.greenlang.io`
   - Staging: `staging-api.boiler.greenlang.io`
   - Development: `gl-002-dev.greenlang.local`
   - cert-manager for automatic TLS certificates

---

## Architecture

### Kubernetes Resources

GL-002 deployment consists of 12 manifest files:

| Resource | File | Purpose |
|----------|------|---------|
| Deployment | `deployment.yaml` | Main application workload (3-10 replicas) |
| Service | `service.yaml` | ClusterIP service for internal communication |
| ConfigMap | `configmap.yaml` | Non-sensitive configuration values |
| Secret | `secret.yaml` | Sensitive credentials (database, API keys) |
| HorizontalPodAutoscaler | `hpa.yaml` | Auto-scaling based on CPU/memory |
| NetworkPolicy | `networkpolicy.yaml` | Zero-trust networking rules |
| Ingress | `ingress.yaml` | HTTPS external access with TLS |
| ServiceMonitor | `servicemonitor.yaml` | Prometheus metrics scraping |
| PodDisruptionBudget | `pdb.yaml` | Availability during disruptions |
| ServiceAccount + RBAC | `serviceaccount.yaml` | Least-privilege access |
| ResourceQuota | `resourcequota.yaml` | Namespace resource limits |
| LimitRange | `limitrange.yaml` | Default pod resource constraints |

### Resource Requirements

**Per Pod:**
- CPU Request: 500m (0.5 cores)
- CPU Limit: 1000m (1 core)
- Memory Request: 512Mi
- Memory Limit: 1024Mi (1 GiB)

**Total (3 replicas):**
- CPU: 1.5 cores request, 3 cores limit
- Memory: 1.5 GiB request, 3 GiB limit

**Scaled (10 replicas):**
- CPU: 5 cores request, 10 cores limit
- Memory: 5 GiB request, 10 GiB limit

---

## Pre-Deployment Checklist

### 1. Namespace Preparation

```bash
# Create namespace (if not exists)
kubectl create namespace greenlang

# Label namespace for monitoring
kubectl label namespace greenlang name=greenlang
kubectl label namespace greenlang monitoring=enabled
```

### 2. Secrets Management

**WARNING:** Never commit secrets to version control!

```bash
# Create secrets (replace with actual values)
kubectl create secret generic gl-002-secrets \
  --from-literal=database_url="postgresql://user:password@postgresql.database.svc.cluster.local:5432/greenlang" \
  --from-literal=redis_url="redis://:password@redis.cache.svc.cluster.local:6379/0" \
  --from-literal=api_key="YOUR_API_KEY_HERE" \
  --from-literal=jwt_secret="YOUR_JWT_SECRET_HERE" \
  -n greenlang

# Verify secret
kubectl get secret gl-002-secrets -n greenlang
```

**Recommended:** Use External Secrets Operator or Sealed Secrets for production.

### 3. Database Migration

```bash
# Run database migrations (before deploying)
kubectl run gl-002-migrate \
  --image=gcr.io/greenlang/gl-002-boiler-efficiency:1.0.0 \
  --restart=Never \
  --command -- python -m alembic upgrade head

# Wait for migration to complete
kubectl wait --for=condition=complete job/gl-002-migrate --timeout=300s

# Clean up migration job
kubectl delete pod gl-002-migrate
```

### 4. Validation

```bash
# Run validation script
cd deployment/scripts
chmod +x validate-manifests.sh
./validate-manifests.sh
```

---

## Deployment Methods

### Method 1: Direct Manifest Deployment (Recommended for Beginners)

```bash
# Deploy all manifests in order
cd deployment

# 1. RBAC and limits
kubectl apply -f serviceaccount.yaml -n greenlang
kubectl apply -f resourcequota.yaml -n greenlang
kubectl apply -f limitrange.yaml -n greenlang

# 2. Configuration
kubectl apply -f configmap.yaml -n greenlang
kubectl apply -f secret.yaml -n greenlang

# 3. Core resources
kubectl apply -f service.yaml -n greenlang
kubectl apply -f deployment.yaml -n greenlang

# 4. Scaling and availability
kubectl apply -f hpa.yaml -n greenlang
kubectl apply -f pdb.yaml -n greenlang

# 5. Networking
kubectl apply -f networkpolicy.yaml -n greenlang
kubectl apply -f ingress.yaml -n greenlang

# 6. Monitoring
kubectl apply -f servicemonitor.yaml -n greenlang
```

### Method 2: Automated Script Deployment (Recommended for Production)

```bash
# Deploy to production
cd deployment/scripts
chmod +x deploy.sh
./deploy.sh production

# Deploy to staging
./deploy.sh staging

# Deploy to development
./deploy.sh dev

# Dry-run (validation only)
DRY_RUN=true ./deploy.sh production
```

### Method 3: Kustomize Deployment (Recommended for Multi-Environment)

```bash
# Production deployment
kubectl apply -k deployment/kustomize/overlays/production

# Staging deployment
kubectl apply -k deployment/kustomize/overlays/staging

# Development deployment
kubectl apply -k deployment/kustomize/overlays/dev

# Preview changes (dry-run)
kubectl apply -k deployment/kustomize/overlays/production --dry-run=client
```

---

## Environment-Specific Deployment

### Development Environment

**Characteristics:**
- 1 replica (single instance)
- Lower resource limits
- Debug logging enabled
- Self-signed TLS certificate
- Relaxed security policies

```bash
# Deploy to dev
./scripts/deploy.sh dev

# Access dev environment
kubectl port-forward -n greenlang-dev svc/dev-gl-002-boiler-efficiency 8000:80
# Visit: http://localhost:8000/api/v1/health
```

### Staging Environment

**Characteristics:**
- 2 replicas (HA testing)
- Production-like resources
- Info logging
- Let's Encrypt staging TLS
- Production-like security

```bash
# Deploy to staging
./scripts/deploy.sh staging

# Run smoke tests
kubectl run smoke-test \
  --image=curlimages/curl:latest \
  --rm -it --restart=Never \
  -- curl -f http://staging-gl-002-boiler-efficiency.greenlang-staging/api/v1/health
```

### Production Environment

**Characteristics:**
- 3 replicas (HA)
- Full resource allocation
- Info/Warning logging
- Let's Encrypt production TLS
- Maximum security enforcement
- Enhanced monitoring

```bash
# IMPORTANT: Production deployment requires approval
# 1. Create change ticket
# 2. Get approval from team lead
# 3. Schedule deployment window
# 4. Notify on-call team

# Deploy to production
./scripts/deploy.sh production

# Monitor deployment
kubectl rollout status deployment/gl-002-boiler-efficiency -n greenlang

# Verify health
kubectl get pods -n greenlang -l app=gl-002-boiler-efficiency
```

---

## Verification

### 1. Deployment Status

```bash
# Check deployment
kubectl get deployment gl-002-boiler-efficiency -n greenlang

# Expected output:
# NAME                       READY   UP-TO-DATE   AVAILABLE   AGE
# gl-002-boiler-efficiency   3/3     3            3           5m

# Check pods
kubectl get pods -n greenlang -l app=gl-002-boiler-efficiency

# Expected output:
# NAME                                        READY   STATUS    RESTARTS   AGE
# gl-002-boiler-efficiency-5d4c8f9b7c-abc12   1/1     Running   0          5m
# gl-002-boiler-efficiency-5d4c8f9b7c-def34   1/1     Running   0          5m
# gl-002-boiler-efficiency-5d4c8f9b7c-ghi56   1/1     Running   0          5m
```

### 2. Health Checks

```bash
# Port-forward to access health endpoint
kubectl port-forward -n greenlang svc/gl-002-boiler-efficiency 8000:80

# In another terminal, test health endpoint
curl http://localhost:8000/api/v1/health

# Expected response:
# {
#   "status": "healthy",
#   "timestamp": "2024-11-17T10:30:00Z",
#   "version": "1.0.0"
# }

# Test readiness endpoint
curl http://localhost:8000/api/v1/ready

# Test metrics endpoint
curl http://localhost:8000/api/v1/metrics
```

### 3. Logs

```bash
# View logs from all pods
kubectl logs -n greenlang -l app=gl-002-boiler-efficiency --tail=100 -f

# View logs from specific pod
kubectl logs -n greenlang gl-002-boiler-efficiency-5d4c8f9b7c-abc12 -f

# View init container logs (dependency checks)
kubectl logs -n greenlang gl-002-boiler-efficiency-5d4c8f9b7c-abc12 -c wait-for-db
kubectl logs -n greenlang gl-002-boiler-efficiency-5d4c8f9b7c-abc12 -c wait-for-redis
```

### 4. HPA Status

```bash
# Check HPA
kubectl get hpa gl-002-boiler-efficiency-hpa -n greenlang

# Expected output:
# NAME                            REFERENCE                             TARGETS         MINPODS   MAXPODS   REPLICAS
# gl-002-boiler-efficiency-hpa    Deployment/gl-002-boiler-efficiency   15%/70%,20%/80%  3         10        3

# Describe HPA (detailed metrics)
kubectl describe hpa gl-002-boiler-efficiency-hpa -n greenlang
```

### 5. Ingress and TLS

```bash
# Check Ingress
kubectl get ingress gl-002-boiler-efficiency-ingress -n greenlang

# Test HTTPS endpoint (production)
curl https://api.boiler.greenlang.io/api/v1/health

# Check TLS certificate
kubectl get secret gl-002-tls-prod -n greenlang
kubectl describe certificate gl-002-tls-prod -n greenlang
```

---

## Rollback Procedures

### Quick Rollback (Previous Revision)

```bash
# Rollback to previous revision
./scripts/rollback.sh production

# Or using kubectl directly
kubectl rollout undo deployment/gl-002-boiler-efficiency -n greenlang

# Monitor rollback
kubectl rollout status deployment/gl-002-boiler-efficiency -n greenlang
```

### Rollback to Specific Revision

```bash
# View rollout history
kubectl rollout history deployment/gl-002-boiler-efficiency -n greenlang

# View specific revision details
kubectl rollout history deployment/gl-002-boiler-efficiency -n greenlang --revision=5

# Rollback to revision 5
./scripts/rollback.sh production 5

# Or using kubectl
kubectl rollout undo deployment/gl-002-boiler-efficiency -n greenlang --to-revision=5
```

### Emergency Rollback

```bash
# Emergency rollback (skips confirmations)
EMERGENCY=true ./scripts/rollback.sh production

# Force immediate rollback
kubectl rollout undo deployment/gl-002-boiler-efficiency -n greenlang --force
```

---

## Troubleshooting

### Common Issues

#### 1. Pods Not Starting

**Symptoms:**
- Pods stuck in `Pending` or `ContainerCreating` state

**Diagnosis:**
```bash
kubectl describe pod <pod-name> -n greenlang
kubectl get events -n greenlang --sort-by='.lastTimestamp'
```

**Common Causes:**
- Insufficient cluster resources (CPU/memory)
- Image pull errors (registry authentication)
- PVC binding issues
- Init containers failing (DB/Redis not ready)

**Solutions:**
```bash
# Check resource availability
kubectl describe nodes

# Check image pull secrets
kubectl get pods -n greenlang -o jsonpath='{.items[*].spec.imagePullSecrets}'

# Check init container logs
kubectl logs <pod-name> -n greenlang -c wait-for-db
```

#### 2. Health Check Failures

**Symptoms:**
- Pods restarting frequently
- `CrashLoopBackOff` status

**Diagnosis:**
```bash
kubectl logs <pod-name> -n greenlang --previous
kubectl describe pod <pod-name> -n greenlang
```

**Common Causes:**
- Application startup timeout
- Database connection errors
- Missing environment variables
- Port conflicts

**Solutions:**
```bash
# Check environment variables
kubectl exec -it <pod-name> -n greenlang -- env | grep GREENLANG

# Test database connectivity
kubectl exec -it <pod-name> -n greenlang -- nc -zv postgresql.database.svc.cluster.local 5432

# Increase startup probe timeout (edit deployment.yaml)
```

#### 3. HPA Not Scaling

**Symptoms:**
- HPA shows `<unknown>` for metrics
- Pods not scaling despite high load

**Diagnosis:**
```bash
kubectl describe hpa gl-002-boiler-efficiency-hpa -n greenlang
kubectl top pods -n greenlang -l app=gl-002-boiler-efficiency
```

**Common Causes:**
- Metrics server not installed
- Resource requests not defined
- Metrics API unavailable

**Solutions:**
```bash
# Install metrics-server
kubectl apply -f https://github.com/kubernetes-sigs/metrics-server/releases/latest/download/components.yaml

# Verify metrics-server
kubectl get apiservice v1beta1.metrics.k8s.io -o yaml
```

#### 4. Ingress Not Working

**Symptoms:**
- 404 errors when accessing external URL
- TLS certificate errors

**Diagnosis:**
```bash
kubectl describe ingress gl-002-boiler-efficiency-ingress -n greenlang
kubectl logs -n ingress-nginx -l app.kubernetes.io/name=ingress-nginx
```

**Common Causes:**
- Ingress controller not installed
- DNS not configured
- TLS certificate not ready
- Path routing misconfigured

**Solutions:**
```bash
# Check ingress controller
kubectl get pods -n ingress-nginx

# Check certificate status
kubectl describe certificate gl-002-tls-prod -n greenlang

# Test service internally
kubectl run curl-test --image=curlimages/curl:latest --rm -it --restart=Never \
  -- curl -f http://gl-002-boiler-efficiency.greenlang/api/v1/health
```

---

## Monitoring and Observability

### Prometheus Metrics

GL-002 exposes Prometheus metrics on port 8001:

```bash
# Access metrics endpoint
kubectl port-forward -n greenlang svc/gl-002-metrics 8001:8001
curl http://localhost:8001/metrics
```

**Key Metrics:**
- `gl_002_optimization_requests_total` - Total optimization requests
- `gl_002_optimization_duration_seconds` - Optimization duration histogram
- `gl_002_boiler_efficiency_percent` - Current boiler efficiency
- `gl_002_fuel_consumption_kg_per_hour` - Fuel consumption rate
- `gl_002_emissions_nox_ppm` - NOx emissions

### Grafana Dashboards

Import pre-built dashboards from `monitoring/grafana/`:

1. **Executive Dashboard** - High-level KPIs
2. **Operations Dashboard** - Operational metrics
3. **Agent Dashboard** - Agent performance
4. **Quality Dashboard** - Quality assurance metrics

### Logging

```bash
# Stream logs
kubectl logs -n greenlang -l app=gl-002-boiler-efficiency -f --max-log-requests=10

# Search logs for errors
kubectl logs -n greenlang -l app=gl-002-boiler-efficiency | grep ERROR

# Export logs to file
kubectl logs -n greenlang -l app=gl-002-boiler-efficiency > gl-002-logs.txt
```

### Alerting

Configure alerts in Prometheus AlertManager:

- **Critical:** Pod down for > 1 minute
- **Critical:** High error rate (> 5%)
- **Warning:** High CPU usage (> 80%)
- **Warning:** High memory usage (> 85%)

---

## Security Best Practices

### 1. Secrets Management

- Never commit secrets to Git
- Use External Secrets Operator or Sealed Secrets
- Rotate secrets quarterly
- Use strong, random passwords (32+ characters)

### 2. RBAC

- Use least privilege principle
- ServiceAccount has minimal permissions
- Avoid ClusterRole unless necessary

### 3. Network Policies

- Zero-trust networking enabled
- Only allow required ingress/egress
- Deny all by default

### 4. Pod Security

- runAsNonRoot: true
- readOnlyRootFilesystem: true
- allowPrivilegeEscalation: false
- Drop all capabilities

### 5. Image Security

- Use specific image tags (not `latest`)
- Scan images for vulnerabilities
- Use minimal base images (distroless)

---

## Maintenance and Updates

### Rolling Updates

```bash
# Update image version
kubectl set image deployment/gl-002-boiler-efficiency \
  gl-002-boiler-efficiency=gcr.io/greenlang/gl-002-boiler-efficiency:1.1.0 \
  -n greenlang

# Monitor rollout
kubectl rollout status deployment/gl-002-boiler-efficiency -n greenlang

# Pause rollout (if issues detected)
kubectl rollout pause deployment/gl-002-boiler-efficiency -n greenlang

# Resume rollout
kubectl rollout resume deployment/gl-002-boiler-efficiency -n greenlang
```

### Blue-Green Deployment

1. Deploy new version with different labels
2. Test new version thoroughly
3. Switch Ingress to new version
4. Decommission old version

### Canary Deployment

1. Deploy 1 replica of new version
2. Monitor metrics and errors
3. Gradually increase new version replicas
4. Rollback if issues detected

---

## Support and Escalation

### Contact Information

- **Team:** GreenLang Industrial Optimization
- **Email:** gl-002-oncall@greenlang.ai
- **Slack:** #gl-002-alerts
- **PagerDuty:** GL-002-BoilerEfficiency
- **Documentation:** https://docs.greenlang.ai/agents/GL-002

### Escalation Path

1. **L1:** On-call engineer (PagerDuty)
2. **L2:** Platform team lead
3. **L3:** CTO / VP Engineering

---

## Appendix

### Useful Commands

```bash
# Scale manually
kubectl scale deployment gl-002-boiler-efficiency --replicas=5 -n greenlang

# Update ConfigMap
kubectl edit configmap gl-002-config -n greenlang

# Restart deployment (without changing image)
kubectl rollout restart deployment/gl-002-boiler-efficiency -n greenlang

# Delete all GL-002 resources
kubectl delete all -l app=gl-002-boiler-efficiency -n greenlang

# Backup manifests
kubectl get deployment,service,configmap,secret -n greenlang -o yaml > backup.yaml
```

### Related Documentation

- [Kubernetes Documentation](https://kubernetes.io/docs/)
- [kubectl Cheat Sheet](https://kubernetes.io/docs/reference/kubectl/cheatsheet/)
- [GL-002 Agent Documentation](https://docs.greenlang.ai/agents/GL-002)
- [GreenLang Platform Documentation](https://docs.greenlang.ai/)

---

**Last Updated:** 2025-11-17
**Version:** 1.0.0
**Maintained By:** GreenLang DevOps Team
