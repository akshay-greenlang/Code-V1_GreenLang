# GreenLang DevOps Infrastructure

Production-grade CI/CD pipeline, containerization, and Kubernetes deployment infrastructure for GreenLang.

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Quick Start](#quick-start)
- [CI/CD Pipelines](#cicd-pipelines)
- [Docker](#docker)
- [Kubernetes](#kubernetes)
- [Deployment](#deployment)
- [Monitoring & Observability](#monitoring--observability)

## Overview

This infrastructure provides:

- **Multi-stage GitHub Actions CI/CD** - Automated testing, building, and deployment
- **Optimized Docker images** - Multi-stage builds with security hardening
- **Kubernetes manifests** - Production-ready deployments with HPA, health checks
- **Enhanced Makefile** - Developer-friendly commands for all operations
- **Security scanning** - Trivy, Bandit, pip-audit integration

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     GitHub Actions CI/CD                      │
├─────────────────────────────────────────────────────────────┤
│  ci-comprehensive.yml   │  build-docker.yml  │  deploy-k8s.yml│
│  - Lint & Type Check   │  - Multi-platform  │  - Dev/Staging │
│  - Unit Tests          │  - Security Scan   │  - Production  │
│  - Integration Tests   │  - SBOM & Signing  │  - Rollback    │
│  - Security Scanning   │  - Push to GHCR    │  - Blue/Green  │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                      Docker Images                            │
├─────────────────────────────────────────────────────────────┤
│  greenlang:latest      │  greenlang-api:latest                │
│  greenlang-full:latest │  greenlang-runner:latest             │
│  - Python 3.11 slim    │  - Multi-stage build                 │
│  - Non-root user       │  - Optimized layers                  │
│  - Health checks       │  - Security hardened                 │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                   Kubernetes Clusters                         │
├─────────────────────────────────────────────────────────────┤
│  Development           │  Staging            │  Production    │
│  - 2 replicas         │  - 3 replicas       │  - 5 replicas  │
│  - HPA (2-10)         │  - HPA (3-15)       │  - HPA (5-20)  │
│  - Ingress + TLS      │  - Ingress + TLS    │  - Blue/Green  │
│  - ConfigMaps         │  - Secrets Manager  │  - Monitoring  │
└─────────────────────────────────────────────────────────────┘
```

## Quick Start

### Prerequisites

- Python 3.11+
- Docker 24.0+
- kubectl 1.28+
- Access to Kubernetes cluster

### Local Development

```bash
# Install development dependencies
make dev

# Run tests
make test

# Run linting
make lint

# Type checking
make type-check
```

### Docker Build & Run

```bash
# Build all Docker images
make docker-build-all

# Run API container locally
make docker-run-api

# Start full stack with Docker Compose
make docker-compose-up

# View logs
make docker-compose-logs

# Stop stack
make docker-compose-down
```

### Kubernetes Deployment

```bash
# Deploy to dev environment
make deploy-dev

# Check deployment status
make k8s-status

# View logs
make k8s-logs

# Port forward to local
make k8s-port-forward
```

## CI/CD Pipelines

### 1. Comprehensive CI Pipeline (`.github/workflows/ci-comprehensive.yml`)

Runs on every PR and push to main branches.

**Jobs:**
- **Lint & Type Check**: Black, isort, Ruff, mypy, Bandit
- **Unit Tests**: Python 3.10, 3.11, 3.12 matrix
- **Integration Tests**: With PostgreSQL and Redis services
- **Security Scan**: pip-audit, Safety, Trivy
- **Build Validation**: Package build and installation test
- **CI Gate**: Final validation checkpoint

**Usage:**
```bash
# Triggered automatically on PR/push
# Or manually run CI locally:
make ci-test
```

### 2. Docker Build Pipeline (`.github/workflows/build-docker.yml`)

Builds and pushes multi-platform Docker images.

**Features:**
- Multi-platform builds (linux/amd64, linux/arm64)
- SBOM and provenance attestation
- Cosign image signing
- Trivy security scanning
- Layer caching with GitHub Actions cache

**Images Built:**
- `ghcr.io/greenlang/greenlang:latest` - CLI runtime
- `ghcr.io/greenlang/greenlang-api:latest` - FastAPI server
- `ghcr.io/greenlang/greenlang-full:latest` - All features
- `ghcr.io/greenlang/greenlang-runner:latest` - Optimized runner

**Usage:**
```bash
# Triggered on push to main or tags
# Or build locally:
make docker-build-all
make docker-push-all
```

### 3. Kubernetes Deployment Pipeline (`.github/workflows/deploy-k8s.yml`)

Automated deployment to dev, staging, and production.

**Environments:**
- **Development**: Auto-deploy on main branch
- **Staging**: Manual trigger or on release
- **Production**: Manual approval required

**Features:**
- Blue-green deployments for production
- Automatic rollback on failure
- Health check verification
- Smoke tests after deployment

**Usage:**
```bash
# Triggered automatically or manually via GitHub Actions UI
# Or deploy locally:
make deploy-dev
make deploy-staging
make deploy-prod  # Requires confirmation
```

## Docker

### Dockerfile Architecture

**Multi-stage Build:**

1. **Builder Stage** (`Dockerfile.api`)
   - Base: `python:3.11-slim`
   - Installs build dependencies
   - Creates virtual environment
   - Compiles Python packages

2. **Runtime Stage**
   - Base: `python:3.11-slim`
   - Copies virtual environment from builder
   - Runs as non-root user (UID 10001)
   - Minimal attack surface

**Key Features:**
- Layer caching optimization
- Security hardening (non-root, dropped capabilities)
- Health checks
- OCI standard labels
- Tini for proper signal handling

### Available Docker Commands

```bash
# Build
make docker-build              # Build main CLI image
make docker-build-api          # Build API server image
make docker-build-all          # Build all images

# Run
make docker-run                # Run CLI container
make docker-run-api            # Run API container
make docker-compose-up         # Start full stack

# Test
make docker-test               # Test Docker image

# Clean
make docker-stop               # Stop containers
make docker-clean              # Remove images and containers

# Push
make docker-push               # Push main image
make docker-push-all           # Push all images
```

## Kubernetes

### Manifest Structure

```
kubernetes/
├── dev/
│   ├── namespace.yaml         # greenlang-dev namespace
│   ├── configmap.yaml         # Non-sensitive configuration
│   ├── secrets.yaml           # Sensitive credentials (template)
│   ├── deployment.yaml        # API deployment (2 replicas)
│   ├── service.yaml           # ClusterIP service
│   ├── ingress.yaml           # NGINX ingress with TLS
│   └── hpa.yaml              # Horizontal Pod Autoscaler
├── staging/
│   └── ...                    # Staging environment
└── prod/
    └── ...                    # Production environment
```

### Key Features

**Deployment:**
- Rolling updates with zero downtime
- Init containers for dependency checks
- Resource requests and limits
- Security context (non-root, read-only root filesystem)
- Liveness, readiness, and startup probes
- Pod anti-affinity for high availability

**Service:**
- ClusterIP for internal communication
- Session affinity
- Metrics endpoint exposure

**Ingress:**
- TLS/SSL with Let's Encrypt
- CORS configuration
- Rate limiting
- Security headers
- Custom error pages

**HPA (Horizontal Pod Autoscaler):**
- CPU-based scaling (70% threshold)
- Memory-based scaling (80% threshold)
- Min replicas: 2, Max replicas: 10
- Scale-down stabilization: 5 minutes

### Kubernetes Commands

```bash
# Deploy
make k8s-apply                 # Apply all manifests
make k8s-delete                # Delete all resources

# Monitor
make k8s-status                # Show cluster status
make k8s-logs                  # Stream pod logs
make k8s-describe              # Describe deployment

# Manage
make k8s-restart               # Restart deployment
make k8s-port-forward          # Port forward to localhost
make k8s-exec                  # Execute shell in pod

# Rollback
make rollback-dev              # Rollback to previous version
```

## Deployment

### Development Environment

```bash
# Full deployment pipeline
make deploy-dev

# This will:
# 1. Build all Docker images
# 2. Push to container registry
# 3. Apply Kubernetes manifests
# 4. Wait for rollout completion
# 5. Verify health checks
```

### Staging Environment

```bash
# Deploy to staging
make deploy-staging

# Verify deployment
kubectl get pods -n greenlang-staging
curl https://staging.greenlang.io/api/v1/health
```

### Production Environment

```bash
# Deploy to production (with confirmation)
make deploy-prod

# Blue-green deployment process:
# 1. Deploy new version as "green"
# 2. Run smoke tests
# 3. Manual approval required
# 4. Switch traffic to green
# 5. Monitor metrics
```

### Rollback

```bash
# Automatic rollback on failure
# Or manual rollback:
make rollback-dev

# Rollback specific revision:
kubectl rollout undo deployment/greenlang-api --to-revision=2 -n greenlang-dev
```

## Monitoring & Observability

### Metrics Endpoints

- **Health Check**: `GET /api/v1/health`
- **Readiness**: `GET /api/v1/ready`
- **Metrics**: `GET /metrics` (Prometheus format)

### Logging

```bash
# View application logs
make k8s-logs

# View specific pod logs
kubectl logs -f <pod-name> -n greenlang-dev

# View logs from all replicas
kubectl logs -f -l app=greenlang-api -n greenlang-dev
```

### Prometheus Integration

Metrics exposed on port 9090:
- Request rate, duration, errors
- Resource utilization (CPU, memory)
- Custom business metrics

### Grafana Dashboards

Pre-built dashboards for:
- API performance
- Resource utilization
- Error rates
- Request latency

## Security

### Image Scanning

All images are scanned with:
- **Trivy**: Vulnerability scanner
- **Bandit**: Python security linter
- **pip-audit**: Dependency vulnerability checker

### Supply Chain Security

- SBOM (Software Bill of Materials) generation
- Cosign image signing
- SLSA provenance attestation

### Secrets Management

- Kubernetes secrets (encrypted at rest)
- External secrets integration ready
- AWS Secrets Manager / HashiCorp Vault compatible

### Network Policies

- Ingress rules for external traffic
- Egress rules for database/cache access
- Service mesh ready

## Troubleshooting

### Common Issues

**1. Image Pull Errors**
```bash
# Check image exists
docker pull ghcr.io/greenlang/greenlang:latest

# Verify image pull secrets
kubectl get secrets -n greenlang-dev
```

**2. Pod CrashLoopBackOff**
```bash
# Check pod logs
kubectl logs <pod-name> -n greenlang-dev --previous

# Describe pod for events
kubectl describe pod <pod-name> -n greenlang-dev
```

**3. Service Not Reachable**
```bash
# Check service endpoints
kubectl get endpoints -n greenlang-dev

# Port forward for debugging
make k8s-port-forward
```

## Best Practices

1. **Always run tests before deployment**
   ```bash
   make ci-test
   ```

2. **Use semantic versioning for images**
   ```bash
   VERSION=0.3.1 make docker-build-all
   ```

3. **Monitor deployments**
   ```bash
   kubectl rollout status deployment/greenlang-api -n greenlang-dev
   ```

4. **Verify health after deployment**
   ```bash
   make verify-deployment
   ```

5. **Keep secrets out of version control**
   - Use `.env` files locally
   - Use Kubernetes secrets in cluster
   - Use secret management tools in production

## Support

For issues or questions:
- GitHub Issues: https://github.com/greenlang/greenlang/issues
- Documentation: https://greenlang.io/docs
- Discord: https://discord.gg/greenlang

## License

MIT License - see LICENSE file for details
