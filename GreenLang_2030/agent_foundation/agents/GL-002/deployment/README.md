# GL-002 BoilerEfficiencyOptimizer - Deployment Guide

Production-grade deployment infrastructure for GL-002 BoilerEfficiencyOptimizer with Kubernetes, Docker, CI/CD, and comprehensive monitoring.

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Prerequisites](#prerequisites)
3. [Quick Start](#quick-start)
4. [Docker Build](#docker-build)
5. [Kubernetes Deployment](#kubernetes-deployment)
6. [CI/CD Pipeline](#cicd-pipeline)
7. [Configuration Management](#configuration-management)
8. [Monitoring & Observability](#monitoring--observability)
9. [Health Checks](#health-checks)
10. [Troubleshooting](#troubleshooting)
11. [Scaling](#scaling)
12. [Disaster Recovery](#disaster-recovery)

## Architecture Overview

### High-Level Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                   GreenLang Infrastructure                   │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  ┌──────────────┐     ┌──────────────┐     ┌──────────────┐ │
│  │   Ingress    │────▶│   Service    │────▶│     Pods     │ │
│  │  (NGINX)     │     │ (ClusterIP)  │     │  (3 Replicas)│ │
│  │ (HTTPS/TLS)  │     │              │     │              │ │
│  └──────────────┘     └──────────────┘     └──────────────┘ │
│                              │                       │        │
│                              ▼                       ▼        │
│  ┌──────────────────┐  ┌────────────────────────────────┐   │
│  │  ConfigMap &     │  │  PostgreSQL Database (RDS)     │   │
│  │  Secrets (K8s)   │  │  Redis Cache (ElastiCache)     │   │
│  └──────────────────┘  └────────────────────────────────┘   │
│          │                                                    │
│          ▼                                                    │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  HPA (Horizontal Pod Autoscaler)                     │   │
│  │  - Min: 3 pods, Max: 10 pods                         │   │
│  │  - Scales at 70% CPU, 80% Memory                     │   │
│  └──────────────────────────────────────────────────────┘   │
│                                                               │
└─────────────────────────────────────────────────────────────┘

CI/CD Pipeline:
  GitHub Push/PR ──▶ CI Pipeline ──▶ Security Scans ──▶ Build Docker ──▶ CD Pipeline ──▶ K8s Deploy
```

### Deployment Environments

- **Development**: Local testing, 1 replica, debug logging
- **Staging**: Pre-production, 2 replicas, integrated testing
- **Production**: High availability, 3-10 replicas, full monitoring

## Prerequisites

### Required Tools

- **Docker**: v20.10+
- **Kubernetes**: v1.24+
- **kubectl**: Latest stable version
- **Helm**: v3.0+ (optional, for advanced deployments)
- **Python**: 3.11+
- **Git**: Latest version

### Required Services

- **PostgreSQL**: 14+
- **Redis**: 7.0+
- **Kubernetes Ingress Controller**: NGINX Ingress or similar
- **cert-manager**: For automatic TLS certificate management
- **Prometheus**: For metrics collection
- **Grafana**: For visualization (optional)

### AWS Resources (for cloud deployment)

- **RDS PostgreSQL** instance
- **ElastiCache Redis** cluster
- **S3** bucket for backups
- **ECR** (Elastic Container Registry) for image storage
- **EKS** (Elastic Kubernetes Service) cluster

### Required Credentials

- Docker Registry credentials (GitHub Container Registry, ECR, etc.)
- Kubernetes cluster configuration (kubeconfig)
- Database connection strings
- API keys for external services (SCADA, Fuel Management, etc.)

## Quick Start

### 1. Build Docker Image Locally

```bash
cd C:\Users\aksha\Code-V1_GreenLang\GreenLang_2030\agent_foundation\agents\GL-002

# Build image
docker build -t gl-002-boiler-efficiency:latest .

# Test image locally
docker run -p 8000:8000 \
  -e GREENLANG_ENV=development \
  -e LOG_LEVEL=DEBUG \
  gl-002-boiler-efficiency:latest
```

### 2. Deploy to Kubernetes (Development)

```bash
# Create namespace
kubectl create namespace greenlang

# Create ConfigMap
kubectl apply -f deployment/configmap.yaml

# Create Secrets (template)
kubectl create secret generic gl-002-secrets \
  --from-literal=database_url='postgresql://user:pass@localhost:5432/db' \
  --from-literal=redis_url='redis://localhost:6379/0' \
  --from-literal=api_key='your-api-key' \
  -n greenlang

# Deploy
kubectl apply -f deployment/deployment.yaml
kubectl apply -f deployment/service.yaml
kubectl apply -f deployment/ingress.yaml

# Verify
kubectl get pods -n greenlang
kubectl logs -f deployment/gl-002-boiler-efficiency -n greenlang
```

### 3. Check Health

```bash
# Port forward to test locally
kubectl port-forward -n greenlang svc/gl-002-boiler-efficiency 8000:80

# Health check
curl http://localhost:8000/api/v1/health

# Readiness check
curl http://localhost:8000/api/v1/ready

# Metrics
curl http://localhost:8000/api/v1/metrics
```

## Docker Build

### Multi-Stage Build Explanation

The Dockerfile uses a multi-stage build process:

**Stage 1: Builder**
- Base image: `python:3.11-slim`
- Installs build dependencies (gcc, g++, libpq-dev)
- Creates Python virtual environment
- Installs all Python packages
- Size: ~1.5 GB (discarded in final image)

**Stage 2: Runtime**
- Base image: `python:3.11-slim`
- Installs only runtime dependencies
- Copies virtual environment from builder
- Creates non-root user (boiler, UID 1000)
- Sets up volumes for logs and data
- Final size: ~500 MB

### Build Options

```bash
# Development build (with debug)
docker build -t gl-002:dev --target=builder .

# Production build (optimized)
docker build -t gl-002:latest --build-arg ENVIRONMENT=production .

# Custom build args
docker build \
  -t gl-002:custom \
  --build-arg PYTHON_VERSION=3.11 \
  --build-arg BASE_IMAGE=python:3.11-slim \
  .

# Push to registry
docker tag gl-002:latest ghcr.io/greenlang/gl-002:latest
docker push ghcr.io/greenlang/gl-002:latest
```

### Health Check

The Dockerfile includes a health check:

```dockerfile
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8000/api/v1/health || exit 1
```

This ensures:
- Container restart if unhealthy after 3 failures (90 seconds)
- Startup grace period of 40 seconds
- Timeout of 10 seconds per check

## Kubernetes Deployment

### File Structure

```
deployment/
├── deployment.yaml          # Deployment spec (3 replicas, resource limits)
├── service.yaml             # ClusterIP service for load balancing
├── configmap.yaml           # Non-sensitive configuration
├── secret.yaml              # Template for sensitive data
├── hpa.yaml                 # Horizontal Pod Autoscaler (3-10 pods)
├── networkpolicy.yaml       # Zero-trust network security
├── ingress.yaml             # HTTPS ingress with TLS
└── README.md                # This file
```

### Deploy All Components

```bash
# Create namespace
kubectl create namespace greenlang

# Apply all manifests in order
kubectl apply -f deployment/configmap.yaml -n greenlang
kubectl apply -f deployment/secret.yaml -n greenlang
kubectl apply -f deployment/deployment.yaml -n greenlang
kubectl apply -f deployment/service.yaml -n greenlang
kubectl apply -f deployment/hpa.yaml -n greenlang
kubectl apply -f deployment/networkpolicy.yaml -n greenlang
kubectl apply -f deployment/ingress.yaml -n greenlang

# Verify
kubectl get all -n greenlang
```

### Deployment Manifest Highlights

**Deployment (deployment.yaml)**
- 3 replicas for high availability
- Rolling update strategy (maxSurge: 1, maxUnavailable: 0)
- Resource requests: 512Mi memory, 500m CPU
- Resource limits: 1024Mi memory, 1000m CPU
- Liveness probe (restart if unresponsive)
- Readiness probe (remove from service if not ready)
- Startup probe (allow 150 seconds for startup)
- Anti-affinity to spread pods across nodes

**Service (service.yaml)**
- ClusterIP for internal communication
- Port 80 (HTTP) → 8000 (app port)
- Port 8001 for metrics
- Session affinity for stateful connections

**ConfigMap (configmap.yaml)**
- Non-sensitive configuration values
- Environment-specific settings
- Feature flags
- Default operational parameters

**Secrets (secret.yaml)**
- Template only (never commit actual secrets)
- Use External Secrets Operator or Sealed Secrets in production
- Supports: database URL, Redis URL, API keys, JWT secrets

**HPA (hpa.yaml)**
- Min 3 replicas, max 10 replicas
- Scale up at 70% CPU or 80% memory
- Scale down conservatively (every 5 minutes)
- Prevents pod churn with stabilization windows

**Network Policy (networkpolicy.yaml)**
- Ingress from Ingress controller only
- Ingress from other GL agents (GL-001, GL-003, GL-004, GL-012)
- Ingress from Prometheus for metrics scraping
- Egress to PostgreSQL, Redis, external APIs
- DNS egress to kube-dns

**Ingress (ingress.yaml)**
- HTTPS termination with automatic TLS via cert-manager
- Rate limiting (100 req/sec, 10 concurrent connections)
- CORS configuration
- Security headers (CSP, X-Frame-Options, etc.)
- Optional basic authentication

## CI/CD Pipeline

### GitHub Actions Workflows

#### CI Pipeline (gl-002-ci.yaml)

Runs on every push and pull request:

1. **Lint & Type Check** (5 min)
   - ruff: Code linting
   - black: Code formatting check
   - isort: Import sorting
   - mypy: Type checking

2. **Run Tests** (15 min)
   - Unit tests (pytest)
   - Integration tests (with PostgreSQL, Redis)
   - Code coverage reporting (>75%)

3. **Security Scan** (10 min)
   - bandit: Security vulnerability scanning
   - safety: Dependency vulnerability check
   - SBOM generation (CycloneDX)

4. **Build Docker Image** (15 min)
   - Multi-stage build
   - Push to GitHub Container Registry
   - Cache optimization with BuildKit

#### CD Pipeline (gl-002-cd.yaml)

Runs on merge to main/master:

1. **Determine Environment** (1 min)
   - Production if main/master
   - Staging if other branch
   - Manual override via workflow_dispatch

2. **Build & Push Image** (15 min)
   - Version tagging (v1.0.0-prod, dev-sha)
   - Push to registry with cache

3. **Deploy to Staging** (10 min)
   - kubectl set image (rolling update)
   - Wait for rollout
   - Smoke tests (health, readiness)
   - Slack notification

4. **Manual Approval** (manual)
   - Production deployment requires approval
   - Prevents accidental prod deployments

5. **Deploy to Production** (15 min)
   - Blue-green deployment strategy
   - Verify all pods are running
   - Run production smoke tests
   - Rollback on failure with Slack alert

### CI/CD Configuration

```bash
# Add secrets to GitHub Actions
gh secret set KUBE_CONFIG_PRODUCTION --body "$(cat ~/.kube/config | base64)"
gh secret set SLACK_WEBHOOK_URL --body "https://hooks.slack.com/services/..."
gh secret set GITHUB_TOKEN  # Automatic, for container registry
```

## Configuration Management

### Environment-Specific Files

Three configuration files for different environments:

**development.yaml**
- Debug mode enabled
- Local database (localhost:5432)
- All features enabled
- Mock data for testing
- No external API calls

**staging.yaml**
- Debug mode disabled (but DEBUG logging)
- Staging database
- Production-like setup
- Integrated testing
- All features enabled

**production.yaml**
- Optimized for performance
- Production database with replication
- Redis Sentinel/Cluster for HA
- Rate limiting enabled
- Comprehensive monitoring
- Disaster recovery enabled

### Load Configuration

```python
# Load based on environment
import yaml
from pathlib import Path

env = os.getenv('GREENLANG_ENV', 'development')
config_path = Path(__file__).parent / 'config' / f'{env}.yaml'

with open(config_path) as f:
    config = yaml.safe_load(f)
```

### Override with Environment Variables

```bash
# Environment variables override config file
export GREENLANG_ENV=production
export LOG_LEVEL=DEBUG
export DATABASE_URL=postgresql://...
export API_PORT=8001
```

## Monitoring & Observability

### Health Checks

Three types of health checks:

**Liveness Probe** (/api/v1/health)
- Detects deadlocks, memory leaks
- Restarts pod if unhealthy after 3 failures
- 30-second check interval

**Readiness Probe** (/api/v1/ready)
- Detects slow startup, initialization issues
- Removes from service if not ready
- 5-second check interval

**Startup Probe** (/api/v1/health)
- Allows time for slow application startup
- 150 seconds (30 * 5 seconds) total time

### Metrics

Prometheus metrics available at `/api/v1/metrics`:

```
# HTTP Request Metrics
gl_002_http_requests_total{method, endpoint, status}
gl_002_http_request_duration_seconds{method, endpoint}

# Optimization Metrics
gl_002_optimization_requests_total{strategy, status}
gl_002_optimization_duration_seconds{strategy}
gl_002_optimization_efficiency_improvement_percent{strategy}

# Boiler Operating Metrics
gl_002_boiler_efficiency_percent{boiler_id}
gl_002_boiler_steam_flow_kg_hr{boiler_id}
gl_002_boiler_fuel_flow_kg_hr{boiler_id}

# Emissions Metrics
gl_002_emissions_co2_kg_hr{boiler_id}
gl_002_emissions_nox_ppm{boiler_id}
gl_002_emissions_compliance_status{boiler_id}

# System Metrics
gl_002_system_uptime_seconds
gl_002_system_memory_usage_bytes{type}
gl_002_system_cpu_usage_percent
```

### Prometheus Configuration

```yaml
# prometheus.yaml
scrape_configs:
  - job_name: 'gl-002'
    static_configs:
      - targets: ['gl-002-boiler-efficiency.greenlang.svc.cluster.local:8001']
    metrics_path: '/api/v1/metrics'
    scrape_interval: 30s
    scrape_timeout: 10s
```

### Grafana Dashboards

Pre-built dashboards available for:
- Real-time efficiency monitoring
- Emissions compliance status
- System resource utilization
- Request latency distribution
- Optimization metrics over time

## Health Checks

### Manual Testing

```bash
# Port forward
kubectl port-forward -n greenlang svc/gl-002-boiler-efficiency 8000:80

# Health endpoint
curl -v http://localhost:8000/api/v1/health
# Expected: {"status": "healthy", "components": {...}}

# Readiness endpoint
curl -v http://localhost:8000/api/v1/ready
# Expected: {"ready": true, "checks": {...}}

# Metrics endpoint
curl http://localhost:8000/api/v1/metrics | head -20
```

### Kubernetes Probe Status

```bash
# Check probe status
kubectl describe pod <pod-name> -n greenlang
# Look for "Liveness" and "Readiness" sections

# View logs for probe failures
kubectl logs <pod-name> -n greenlang --previous

# Manually test from pod
kubectl exec -it <pod-name> -n greenlang -- curl http://localhost:8000/api/v1/health
```

## Troubleshooting

### Pod Not Starting

```bash
# Check pod status
kubectl get pods -n greenlang
kubectl describe pod <pod-name> -n greenlang

# View logs
kubectl logs <pod-name> -n greenlang

# Check events
kubectl get events -n greenlang --sort-by='.lastTimestamp'
```

### Probe Failures

```bash
# Liveness probe failures (pod keeps restarting)
kubectl logs <pod-name> -n greenlang --previous
# Check /api/v1/health endpoint

# Readiness probe failures (pod not in service)
kubectl logs <pod-name> -n greenlang
# Check /api/v1/ready endpoint
# Ensure database and cache are accessible

# Increase initial delay if startup is slow
kubectl edit deployment gl-002-boiler-efficiency -n greenlang
# Increase livenessProbe.initialDelaySeconds
```

### High Memory Usage

```bash
# Check memory limits
kubectl top pods -n greenlang
kubectl describe node <node-name>

# Check for memory leaks
kubectl exec -it <pod-name> -n greenlang -- top -b -n 1

# Increase limits if needed
kubectl set resources deployment gl-002-boiler-efficiency \
  --limits=memory=2Gi,cpu=2000m \
  -n greenlang
```

### Database Connection Issues

```bash
# Verify database accessibility
kubectl run -it --rm debug --image=postgres:14 --restart=Never -- \
  psql postgresql://user:pass@db-host:5432/db

# Check connection pool
kubectl exec <pod-name> -n greenlang -- \
  curl -s http://localhost:8000/api/v1/metrics | grep db_connection
```

### Ingress Not Working

```bash
# Check ingress status
kubectl get ingress -n greenlang
kubectl describe ingress gl-002-boiler-efficiency-ingress -n greenlang

# Check NGINX controller logs
kubectl logs -n ingress-nginx -l app.kubernetes.io/name=ingress-nginx

# Test with curl
curl -v -H "Host: api.boiler.greenlang.io" http://ingress-ip/api/v1/health
```

## Scaling

### Manual Scaling

```bash
# Scale to specific number of replicas
kubectl scale deployment gl-002-boiler-efficiency --replicas=5 -n greenlang

# Watch scaling progress
kubectl get deployment gl-002-boiler-efficiency -w -n greenlang
```

### Auto-Scaling

HPA automatically scales based on:
- **CPU**: Scale up at 70% average utilization
- **Memory**: Scale up at 80% average utilization
- **Min**: 3 replicas (high availability)
- **Max**: 10 replicas (cost control)

```bash
# Check HPA status
kubectl get hpa -n greenlang
kubectl describe hpa gl-002-boiler-efficiency-hpa -n greenlang

# Watch scaling in action
kubectl get hpa -w -n greenlang

# Modify HPA settings
kubectl autoscale deployment gl-002-boiler-efficiency \
  --min=2 --max=8 --cpu-percent=80 -n greenlang
```

### Load Testing

```bash
# Generate load to trigger scaling
kubectl run -it --rm load-generator --image=busybox --restart=Never -- \
  /bin/sh -c "while sleep 1; do wget -q -O- http://gl-002-boiler-efficiency:80/api/v1/health; done"

# Watch HPA respond
kubectl get hpa -w -n greenlang
```

## Disaster Recovery

### Backup Strategy

- **Database**: Daily snapshots via RDS automated backups (7-day retention)
- **Application State**: Saved to S3 every hour
- **Configuration**: Stored in Git (never in secrets)
- **Secrets**: Managed by Sealed Secrets or External Secrets Operator

### Recovery Procedures

**RTO**: 4 hours (Recovery Time Objective)
**RPO**: 1 hour (Recovery Point Objective)

### Restore Database

```bash
# List available backups
aws rds describe-db-snapshots --db-instance-identifier gl-002

# Restore from snapshot
aws rds restore-db-instance-from-db-snapshot \
  --db-instance-identifier gl-002-restored \
  --db-snapshot-identifier gl-002-snapshot-time
```

### Restore Kubernetes

```bash
# Backup etcd
ETCDCTL_API=3 etcdctl \
  --endpoints=https://127.0.0.1:2379 \
  --cacert=/etc/kubernetes/pki/etcd/ca.crt \
  --cert=/etc/kubernetes/pki/etcd/server.crt \
  --key=/etc/kubernetes/pki/etcd/server.key \
  snapshot save /backup/etcd-snapshot.db

# Restore from backup
ETCDCTL_API=3 etcdctl \
  snapshot restore /backup/etcd-snapshot.db \
  --data-dir=/var/lib/etcd-restored
```

### Verify Deployment Health

```bash
# Check all resources
kubectl get all -n greenlang

# Verify pod readiness
kubectl rollout status deployment/gl-002-boiler-efficiency -n greenlang

# Test endpoints
curl https://api.boiler.greenlang.io/api/v1/health
curl https://api.boiler.greenlang.io/api/v1/ready
```

## Best Practices

1. **Always use namespaces** for resource isolation
2. **Set resource limits** to prevent resource exhaustion
3. **Use liveness and readiness probes** for automatic healing
4. **Enable horizontal pod autoscaling** for dynamic load handling
5. **Use network policies** to restrict traffic
6. **Enable RBAC** for security
7. **Use Secrets** for sensitive data (never in ConfigMaps)
8. **Enable audit logging** for compliance
9. **Monitor continuously** with Prometheus and Grafana
10. **Test disaster recovery** regularly

## Support & Documentation

- **Full Deployment Guide**: See `DEPLOYMENT_GUIDE.md`
- **Architecture**: See `ARCHITECTURE.md` in GL-002 root directory
- **API Documentation**: `/docs/api/boiler/optimizer`
- **Issue Tracker**: github.com/greenlang/agents/issues?label=GL-002
- **Support Email**: boiler-systems@greenlang.ai
