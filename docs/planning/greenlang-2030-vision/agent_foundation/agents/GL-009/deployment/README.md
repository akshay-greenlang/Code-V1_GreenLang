# GL-009 THERMALIQ ThermalEfficiencyCalculator - DevOps Infrastructure

## Overview

This directory contains complete production-grade DevOps infrastructure for GL-009 THERMALIQ ThermalEfficiencyCalculator, including:

- Multi-stage Docker builds
- Kubernetes deployments with Kustomize
- CI/CD pipelines with GitHub Actions
- Monitoring and observability
- Multi-environment support (dev, staging, production)

## Directory Structure

```
GL-009/
├── Dockerfile                          # Multi-stage production build (184 lines)
├── requirements.txt                    # Python dependencies (194 lines)
├── deployment/
│   ├── kustomize/
│   │   ├── base/                       # Base Kubernetes manifests
│   │   │   ├── deployment.yaml         # Deployment + RBAC (579 lines)
│   │   │   ├── service.yaml            # Services (80 lines)
│   │   │   ├── configmap.yaml          # Configuration (150 lines)
│   │   │   ├── secret.yaml             # External Secrets (100 lines)
│   │   │   ├── hpa.yaml                # Horizontal Pod Autoscaler (60 lines)
│   │   │   ├── pdb.yaml                # PDB + NetworkPolicy + Ingress (80 lines)
│   │   │   └── kustomization.yaml      # Base kustomization
│   │   └── overlays/
│   │       ├── dev/                    # Development environment
│   │       │   └── kustomization.yaml
│   │       ├── staging/                # Staging environment
│   │       │   ├── kustomization.yaml
│   │       │   └── staging-configmap-patch.yaml
│   │       └── production/             # Production environment
│   │           ├── kustomization.yaml
│   │           ├── production-configmap-patch.yaml
│   │           └── production-monitoring.yaml
│   └── README.md                       # This file
└── .github/workflows/
    └── gl-009-ci.yaml                  # CI/CD pipeline (493 lines)
```

## Quick Start

### Local Development

1. Build Docker image:
```bash
cd GL-009
docker build -t gl-009-thermaliq:dev .
```

2. Run locally:
```bash
docker run -p 8000:8000 \
  -e DATABASE_URL=postgresql://user:pass@localhost:5432/db \
  -e REDIS_URL=redis://localhost:6379/0 \
  gl-009-thermaliq:dev
```

### Deploy to Kubernetes

#### Development Environment
```bash
cd deployment/kustomize/overlays/dev
kubectl apply -k .
```

#### Staging Environment
```bash
cd deployment/kustomize/overlays/staging
kubectl apply -k .
```

#### Production Environment
```bash
cd deployment/kustomize/overlays/production
kubectl apply -k .
```

## Dockerfile Architecture

**Multi-stage build with 3 stages:**

1. **Builder Stage**: Compiles scientific computing libraries (NumPy, SciPy) from source
2. **Security Scanner Stage**: Runs Bandit, Safety, pip-audit (optional for CI/CD)
3. **Runtime Stage**: Minimal production image with only runtime dependencies

**Key Features:**
- Python 3.11-slim base
- Non-root user (greenlang:1000)
- Health checks
- Optimized for scientific computing (OpenBLAS, LAPACK)
- 4 exposed ports: 8000 (API), 8001 (Metrics), 8002 (Admin), 8003 (WebSocket)

## Dependencies (requirements.txt)

**Total: 100+ production dependencies including:**

- **Core Framework**: FastAPI, Uvicorn, Pydantic
- **Scientific Computing**: NumPy, SciPy, Pandas, SymPy
- **Thermodynamics**: CoolProp, IAPWS, fluids, ht
- **Industrial Protocols**: pymodbus, asyncua, paho-mqtt, BACnet
- **Monitoring**: Prometheus, OpenTelemetry, Sentry
- **Security**: python-jose, cryptography, passlib
- **Testing**: pytest, pytest-asyncio, pytest-cov

## Kubernetes Deployment

### Base Manifests

#### deployment.yaml (579 lines)
- **Replicas**: 3 (production HA)
- **Rolling Update**: maxSurge=1, maxUnavailable=0
- **Resources**: 1Gi/1CPU request, 2Gi/2CPU limit
- **Probes**: Liveness, readiness, startup
- **Security**: runAsNonRoot, readOnlyRootFilesystem
- **Init Containers**: Database and Redis health checks
- **Affinity**: Pod anti-affinity across nodes/zones
- **RBAC**: ServiceAccount, ClusterRole, ClusterRoleBinding

#### service.yaml (80 lines)
- **Main Service**: ClusterIP on port 80 (API + WebSocket)
- **Metrics Service**: Port 8001 for Prometheus
- **Admin Service**: Port 8002 for internal management
- **Headless Service**: Direct pod-to-pod communication

#### configmap.yaml (150 lines)
Application configuration:
- Environment settings
- Logging configuration (JSON, INFO level)
- Database pool size: 20
- Calculation timeout: 300s
- Max concurrent calculations: 50
- Feature flags (heat recovery, pinch analysis, exergy analysis)
- Industrial protocol configuration (Modbus, OPC UA, MQTT)

#### secret.yaml (100 lines)
External Secrets Operator integration:
- Database credentials (AWS Secrets Manager)
- Redis credentials
- API keys and JWT secrets
- Sentry DSN
- Fallback dev secrets (DO NOT USE IN PRODUCTION)

#### hpa.yaml (60 lines)
Horizontal Pod Autoscaler:
- **Min/Max**: 3-10 replicas
- **Metrics**: CPU (70%), Memory (80%), custom metrics
- **Custom Metrics**:
  - calculation_queue_depth
  - active_calculations
  - http_request_duration_p95_seconds
- **Scaling Behavior**: Aggressive scale-up, conservative scale-down

#### pdb.yaml (80 lines)
- **PodDisruptionBudget**: minAvailable=2
- **NetworkPolicy**: Ingress/egress rules
- **Ingress**: NGINX with TLS, rate limiting, WebSocket support

### Environment Overlays

#### Development (overlays/dev/)
- **Replicas**: 1
- **Resources**: 512Mi/250m request, 1Gi/500m limit
- **Log Level**: DEBUG
- **Secrets**: Static development secrets
- **HPA**: 1-2 replicas

#### Staging (overlays/staging/)
- **Replicas**: 2
- **Resources**: 768Mi/750m request, 1536Mi/1500m limit
- **Log Level**: INFO
- **Domain**: staging.thermaliq.greenlang.ai
- **HPA**: 2-5 replicas
- **Features**: Beta features enabled, enhanced logging

#### Production (overlays/production/)
- **Replicas**: 3-10 (HPA)
- **Resources**: 1Gi/1000m request, 2Gi/2000m limit
- **Log Level**: WARNING
- **Domain**: thermaliq.greenlang.ai
- **HPA**: 3-10 replicas
- **Monitoring**: ServiceMonitor, PrometheusRule, Grafana Dashboard
- **Security**: Required pod anti-affinity, strict health checks

## CI/CD Pipeline (GitHub Actions)

**8 Jobs, ~500 lines:**

### 1. Lint & Code Quality (10 min)
- Ruff linter
- Black code formatting
- isort import sorting
- mypy type checking

### 2. Security Scan (15 min)
- Bandit security scanner
- Safety dependency checker
- pip-audit vulnerability scanning
- Upload security reports

### 3. Unit Tests (20 min)
- PostgreSQL + Redis services
- pytest with coverage
- Upload to Codecov
- JUnit XML reports

### 4. Integration Tests (30 min)
- End-to-end API testing
- Database migrations
- Cache integration

### 5. Build Docker Image (30 min)
- Docker Buildx multi-platform
- Push to GCR
- Trivy vulnerability scanning
- Upload SARIF to GitHub Security

### 6. Deploy to Staging
- Kustomize deployment
- Rollout verification
- Pod health checks

### 7. Smoke Tests (10 min)
- Health endpoint
- Readiness endpoint
- Metrics endpoint
- Sample calculation API test

### 8. Deploy to Production
- Manual approval required
- Blue-green deployment
- HPA verification
- Slack notification

## Monitoring & Observability

### Prometheus Metrics
- HTTP request rate, latency, errors
- Calculation queue depth
- Active calculations
- Database connection pool
- Redis connection status
- Pod CPU/memory usage

### Alerts (PrometheusRule)
- High error rate (>5%)
- High latency (p95 > 5s)
- Pod availability (<2 pods)
- High memory usage (>90%)
- High CPU usage (>1.8 cores)
- Queue depth (>100)
- Database pool exhaustion (>90%)
- Redis connection failures

### Grafana Dashboard
- Request rate
- Error rate
- Response time (p50, p95, p99)
- Active calculations
- Pod resource usage
- Queue depth

## Security

### Container Security
- Non-root user (UID 1000)
- Read-only root filesystem
- Dropped all capabilities (except NET_BIND_SERVICE)
- No privilege escalation
- SeccompProfile: RuntimeDefault

### Network Security
- NetworkPolicy for ingress/egress
- TLS/SSL termination at Ingress
- Rate limiting (100 req/min)
- CORS configuration
- Security headers (X-Frame-Options, X-Content-Type-Options, etc.)

### Secret Management
- External Secrets Operator
- AWS Secrets Manager integration
- Automatic secret rotation (1h refresh)
- No secrets in Git

## Scaling

### Horizontal Scaling (HPA)
- **Development**: 1-2 pods
- **Staging**: 2-5 pods
- **Production**: 3-10 pods
- **Triggers**: CPU, memory, custom metrics

### Vertical Scaling
- **Development**: 512Mi-1Gi, 250m-500m CPU
- **Staging**: 768Mi-1536Mi, 750m-1500m CPU
- **Production**: 1Gi-2Gi, 1-2 CPU cores

### Database Scaling
- **Connection Pool**: 20 connections (production)
- **Pool Overflow**: 10 additional connections
- **Statement Timeout**: 30s

## Troubleshooting

### Check Pod Status
```bash
kubectl get pods -n greenlang-production -l app=gl-009-thermaliq
```

### View Logs
```bash
kubectl logs -n greenlang-production -l app=gl-009-thermaliq --tail=100 -f
```

### Check Resource Usage
```bash
kubectl top pods -n greenlang-production -l app=gl-009-thermaliq
```

### Check HPA Status
```bash
kubectl get hpa -n greenlang-production gl-009-thermaliq-hpa
```

### Exec into Pod
```bash
kubectl exec -it -n greenlang-production <pod-name> -- /bin/bash
```

### Rollback Deployment
```bash
kubectl rollout undo deployment/prod-gl-009-thermaliq -n greenlang-production
```

## Environment Variables

### Required
- `DATABASE_URL`: PostgreSQL connection string
- `REDIS_URL`: Redis connection string
- `API_KEY`: API authentication key
- `JWT_SECRET`: JWT signing secret

### Optional
- `LOG_LEVEL`: Logging level (DEBUG, INFO, WARNING, ERROR)
- `GREENLANG_ENV`: Environment (development, staging, production)
- `SENTRY_DSN`: Sentry error tracking DSN
- `JAEGER_ENDPOINT`: Jaeger tracing endpoint

## Performance Tuning

### Application
- Worker processes: 4
- Uvloop for async I/O
- NumPy threads: 4
- Connection pooling enabled
- Query result caching enabled

### Database
- Pool size: 20
- Pool timeout: 30s
- Pool recycle: 3600s (1h)

### Cache
- TTL: 3600s (1h)
- Max size: 1GB
- Eviction policy: LRU

## License

MIT License - Copyright (c) 2025 GreenLang

## Support

- Documentation: https://docs.greenlang.ai/agents/GL-009
- Issues: https://github.com/greenlang/gl-009/issues
- Email: devops@greenlang.ai
