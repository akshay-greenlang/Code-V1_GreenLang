# GL-010 EMISSIONWATCH EmissionsComplianceAgent - DevOps Infrastructure

## Overview

This directory contains complete production-grade DevOps infrastructure for GL-010 EMISSIONWATCH EmissionsComplianceAgent, including:

- Multi-stage Docker builds
- Kubernetes deployments with Kustomize
- CI/CD pipelines with GitHub Actions
- Monitoring and observability
- Multi-environment support (dev, staging, production)
- Regulatory compliance support (EPA 40 CFR Part 75, EU ETS, ISO 14064)

## Directory Structure

```
GL-010/
|-- Dockerfile                              # Multi-stage production build (~180 lines)
|-- requirements.txt                        # Python dependencies (~200 lines)
|-- deployment/
|   |-- kustomize/
|   |   |-- base/                           # Base Kubernetes manifests
|   |   |   |-- deployment.yaml             # Deployment + RBAC (~580 lines)
|   |   |   |-- service.yaml                # Services (~100 lines)
|   |   |   |-- configmap.yaml              # Configuration (~200 lines)
|   |   |   |-- secret.yaml                 # External Secrets (~180 lines)
|   |   |   |-- hpa.yaml                    # Horizontal Pod Autoscaler (~90 lines)
|   |   |   |-- pdb.yaml                    # PDB + NetworkPolicy + Ingress (~280 lines)
|   |   |   |-- serviceaccount.yaml         # ServiceAccount + RBAC (~150 lines)
|   |   |   |-- kustomization.yaml          # Base kustomization
|   |   |-- overlays/
|   |       |-- dev/                        # Development environment
|   |       |   |-- kustomization.yaml
|   |       |-- staging/                    # Staging environment
|   |       |   |-- kustomization.yaml
|   |       |   |-- staging-configmap-patch.yaml
|   |       |-- production/                 # Production environment
|   |           |-- kustomization.yaml
|   |           |-- production-configmap-patch.yaml
|   |           |-- production-monitoring.yaml
|   |-- README.md                           # This file
|-- .github/workflows/
    |-- gl-010-ci.yaml                      # CI/CD pipeline (~500 lines)
```

## Quick Start

### Local Development

1. Build Docker image:
```bash
cd GL-010
docker build -t gl-010-emissionwatch:dev .
```

2. Run locally with Docker Compose:
```bash
docker-compose up -d
```

3. Or run directly:
```bash
docker run -p 8080:8080 \
  -e DATABASE_URL=postgresql://user:pass@localhost:5432/db \
  -e TIMESCALE_URL=postgresql://user:pass@localhost:5433/timescale \
  -e REDIS_URL=redis://localhost:6379/0 \
  -e USE_CEMS_SIMULATOR=true \
  gl-010-emissionwatch:dev
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

1. **Builder Stage**: Compiles scientific computing libraries and installs dependencies
2. **Security Scanner Stage**: Runs Bandit, Safety, pip-audit (optional for CI/CD)
3. **Runtime Stage**: Minimal production image with only runtime dependencies

**Key Features:**
- Python 3.11-slim base
- Non-root user (greenlang:1000)
- Health checks
- Optimized for scientific computing (OpenBLAS, LAPACK)
- 4 exposed ports: 8080 (API), 9090 (Metrics), 8081 (Admin), 8082 (WebSocket)

## Dependencies (requirements.txt)

**Total: 120+ production dependencies including:**

- **Core Framework**: FastAPI, Uvicorn, Pydantic
- **Scientific Computing**: NumPy, SciPy, Pandas, statsmodels
- **Emissions Calculations**: chempy, pint, uncertainties
- **Industrial Protocols**: pymodbus, asyncua, paho-mqtt, pydnp3
- **Regulatory Reporting**: lxml, reportlab, weasyprint
- **Monitoring**: Prometheus, OpenTelemetry, Sentry
- **Security**: python-jose, cryptography, pyhanko
- **Testing**: pytest, pytest-asyncio, pytest-cov, hypothesis

## Kubernetes Deployment

### Base Manifests

#### deployment.yaml (~580 lines)
- **Replicas**: 3 (production HA)
- **Rolling Update**: maxSurge=1, maxUnavailable=0
- **Resources**: 1Gi/1CPU request, 2Gi/2CPU limit
- **Probes**: Liveness, readiness, startup
- **Security**: runAsNonRoot, readOnlyRootFilesystem
- **Init Containers**: Database, Redis, TimescaleDB health checks
- **Affinity**: Pod anti-affinity across nodes/zones
- **RBAC**: ServiceAccount, ClusterRole, ClusterRoleBinding

#### service.yaml (~100 lines)
- **Main Service**: ClusterIP on port 80 (API + WebSocket)
- **Metrics Service**: Port 9090 for Prometheus
- **Admin Service**: Port 8081 for internal management
- **Headless Service**: Direct pod-to-pod communication
- **CEMS Service**: OPC UA (4840) and Modbus (502)

#### configmap.yaml (~200 lines)
Application configuration:
- Environment settings
- CEMS configuration (polling interval, buffer size)
- Emissions limits by jurisdiction (EPA, EU ETS, CARB)
- Alert thresholds (80%, 90%, 95%)
- Compliance reporting schedules
- Industrial protocol configuration (Modbus, OPC UA, MQTT)
- Data retention policies (7 years for raw data)

#### secret.yaml (~180 lines)
External Secrets Operator integration:
- Database credentials (PostgreSQL, TimescaleDB)
- Redis credentials
- API keys and JWT secrets
- EPA CDX credentials
- EU ETS registry credentials
- Report signing certificates
- Fallback dev secrets (DO NOT USE IN PRODUCTION)

#### hpa.yaml (~90 lines)
Horizontal Pod Autoscaler:
- **Min/Max**: 3-10 replicas
- **Metrics**: CPU (70%), Memory (80%), custom metrics
- **Custom Metrics**:
  - cems_data_points_per_second
  - emissions_calculation_queue_depth
  - active_cems_connections
  - pending_compliance_reports
- **Scaling Behavior**: Aggressive scale-up, conservative scale-down

#### pdb.yaml (~280 lines)
- **PodDisruptionBudget**: minAvailable=2
- **NetworkPolicy**: Ingress/egress rules including CEMS protocols
- **Ingress**: NGINX with TLS, rate limiting, WebSocket support

### Environment Overlays

#### Development (overlays/dev/)
- **Replicas**: 1
- **Resources**: 512Mi/250m request, 1Gi/500m limit
- **Log Level**: DEBUG
- **CEMS**: Simulator mode enabled
- **Secrets**: Static development secrets
- **HPA**: 1-2 replicas
- **Regulatory APIs**: Disabled

#### Staging (overlays/staging/)
- **Replicas**: 2
- **Resources**: 768Mi/750m request, 1536Mi/1500m limit
- **Log Level**: INFO
- **Domain**: staging.emissionwatch.greenlang.ai
- **HPA**: 2-5 replicas
- **Regulatory APIs**: Sandbox environments
- **Features**: Debug endpoints enabled

#### Production (overlays/production/)
- **Replicas**: 3-10 (HPA)
- **Resources**: 1Gi/1000m request, 2Gi/2000m limit
- **Log Level**: WARNING
- **Domain**: emissionwatch.greenlang.ai
- **HPA**: 3-10 replicas
- **Monitoring**: ServiceMonitor, PrometheusRule, Grafana Dashboard
- **Security**: Required pod anti-affinity, strict health checks
- **Regulatory APIs**: Production EPA/EU ETS

## CI/CD Pipeline (GitHub Actions)

**12 Jobs, ~500 lines:**

### 1. Lint & Code Quality (15 min)
- Ruff linter
- Black code formatting
- isort import sorting
- MyPy type checking

### 2. Security Scan (20 min)
- Bandit security scanner
- Safety dependency checker
- pip-audit vulnerability scanning
- Upload security reports

### 3. Unit Tests (30 min)
- PostgreSQL + TimescaleDB + Redis services
- pytest with coverage
- Upload to Codecov
- JUnit XML reports

### 4. Integration Tests (45 min)
- End-to-end API testing
- Database migrations
- Cache integration
- CEMS integration (simulator)

### 5. CEMS Simulator Tests (30 min)
- CEMS data ingestion tests
- Protocol communication tests
- Data quality validation

### 6. Determinism Tests (20 min)
- Emissions calculation reproducibility
- Floating-point consistency
- Hash verification

### 7. Build Docker Image (45 min)
- Docker Buildx multi-platform (amd64, arm64)
- Push to GCR
- Trivy vulnerability scanning
- Upload SARIF to GitHub Security

### 8. Deploy to Staging
- Kustomize deployment
- Rollout verification
- Pod health checks

### 9. Smoke Tests (15 min)
- Health endpoint
- Readiness endpoint
- Metrics endpoint
- CEMS status endpoint
- Compliance API tests

### 10. Deploy to Production
- Manual approval required
- Blue-green deployment
- HPA verification
- Slack notification

### 11. Production Health Check
- Health endpoint verification
- CEMS connectivity check
- EPA CDX connectivity
- EU ETS connectivity

### 12. Rollback (Manual)
- Automatic rollback on failure
- Deployment undo
- Slack notification

## Monitoring & Observability

### Prometheus Metrics
- HTTP request rate, latency, errors
- CEMS data ingestion rate
- Emissions calculation queue depth
- Active CEMS connections
- Pending compliance reports
- Database connection pool
- Redis connection status
- Pod CPU/memory usage

### Alerts (PrometheusRule)

**Availability Alerts:**
- Pod availability (<2 pods) - CRITICAL
- Pod crash looping - WARNING

**Performance Alerts:**
- High error rate (>5%) - WARNING
- High latency (p95 > 5s) - WARNING
- High memory usage (>90%) - WARNING
- High CPU usage (>1.8 cores) - WARNING

**CEMS Alerts:**
- CEMS connection failure - CRITICAL
- CEMS data gap (>15 min) - CRITICAL
- High emissions queue depth - WARNING

**Compliance Alerts:**
- Emissions threshold approaching (80%) - WARNING
- Emissions threshold exceeded (100%) - CRITICAL
- Compliance report submission failure - CRITICAL
- Pending compliance reports accumulating - WARNING

**Database Alerts:**
- Database pool exhaustion (>90%) - WARNING
- TimescaleDB connection failure - CRITICAL
- Redis connection failure - WARNING

### Grafana Dashboard
- CEMS data ingestion rate
- Real-time emissions levels
- Compliance status
- Alert history
- Pod resource usage

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
- Automatic secret rotation (30min-24h refresh)
- No secrets in Git
- Digital signatures for compliance reports

### Compliance Security
- Audit trail for all operations
- Digital signatures for reports
- MFA requirement for submissions
- Approval workflow for regulatory reports

## Scaling

### Horizontal Scaling (HPA)
- **Development**: 1-2 pods
- **Staging**: 2-5 pods
- **Production**: 3-10 pods
- **Triggers**: CPU, memory, CEMS data rate, queue depth

### Vertical Scaling
- **Development**: 512Mi-1Gi, 250m-500m CPU
- **Staging**: 768Mi-1536Mi, 750m-1500m CPU
- **Production**: 1Gi-2Gi, 1-2 CPU cores

### Database Scaling
- **PostgreSQL Connection Pool**: 20 connections (production)
- **TimescaleDB**: Automatic partitioning by time
- **Redis**: Cluster mode for high availability

## Regulatory Compliance

### EPA 40 CFR Part 75
- ECMPS quarterly reporting
- CDX electronic submissions
- DAHS data requirements
- QA/QC procedures

### EU ETS
- Annual emissions reporting
- Verification requirements
- Registry integration
- Allowance tracking

### ISO 14064
- GHG inventory reporting
- Uncertainty analysis
- Verification support

### Data Retention
- Raw emissions data: 7 years (EPA requirement)
- Aggregated data: 10 years
- Compliance reports: 10 years
- Audit logs: 7 years

## Troubleshooting

### Check Pod Status
```bash
kubectl get pods -n greenlang-production -l app=gl-010-emissionwatch
```

### View Logs
```bash
kubectl logs -n greenlang-production -l app=gl-010-emissionwatch --tail=100 -f
```

### Check CEMS Status
```bash
kubectl exec -it -n greenlang-production <pod-name> -- \
  curl http://localhost:8080/api/v1/cems/status
```

### Check Emissions Data
```bash
kubectl exec -it -n greenlang-production <pod-name> -- \
  curl http://localhost:8080/api/v1/emissions/current
```

### Check Resource Usage
```bash
kubectl top pods -n greenlang-production -l app=gl-010-emissionwatch
```

### Check HPA Status
```bash
kubectl get hpa -n greenlang-production gl-010-emissionwatch-hpa
```

### View Compliance Report Queue
```bash
kubectl exec -it -n greenlang-production <pod-name> -- \
  curl http://localhost:8080/api/v1/compliance/reports/pending
```

### Exec into Pod
```bash
kubectl exec -it -n greenlang-production <pod-name> -- /bin/bash
```

### Rollback Deployment
```bash
kubectl rollout undo deployment/prod-gl-010-emissionwatch -n greenlang-production
```

## Environment Variables

### Required
- `DATABASE_URL`: PostgreSQL connection string
- `TIMESCALE_URL`: TimescaleDB connection string
- `REDIS_URL`: Redis connection string
- `API_KEY`: API authentication key
- `JWT_SECRET`: JWT signing secret

### EPA CDX (Production)
- `EPA_CDX_USERNAME`: EPA CDX username
- `EPA_CDX_PASSWORD`: EPA CDX password
- `EPA_CDX_API_KEY`: EPA CDX API key
- `EPA_CDX_PROGRAM_ID`: Facility program ID

### EU ETS (Production)
- `EU_ETS_API_KEY`: EU ETS registry API key
- `EU_ETS_CLIENT_ID`: OAuth client ID
- `EU_ETS_CLIENT_SECRET`: OAuth client secret
- `EU_ETS_OPERATOR_ID`: Operator ID

### Optional
- `LOG_LEVEL`: Logging level (DEBUG, INFO, WARNING, ERROR)
- `GREENLANG_ENV`: Environment (development, staging, production)
- `SENTRY_DSN`: Sentry error tracking DSN
- `JAEGER_ENDPOINT`: Jaeger tracing endpoint
- `USE_CEMS_SIMULATOR`: Enable CEMS simulator (dev only)

## Performance Tuning

### Application
- Worker processes: 4
- Uvloop for async I/O
- NumPy threads: 4
- Connection pooling enabled
- Query result caching enabled

### CEMS Data Processing
- Polling interval: 60 seconds
- Data buffer size: 10,000 readings
- Flush interval: 300 seconds
- Batch processing size: 1,000

### Database
- PostgreSQL pool size: 20
- Pool timeout: 30s
- Pool recycle: 3600s (1h)
- TimescaleDB chunk interval: 1 day

### Cache
- TTL: 300s (5 min for real-time data)
- Max size: 1GB
- Eviction policy: LRU

## CEMS Integration

### Supported Protocols
- **Modbus TCP/RTU**: Port 502
- **OPC UA**: Port 4840
- **MQTT**: Port 1883/8883 (TLS)
- **DNP3**: Port 20000

### Data Quality Checks
- Range validation
- Spike detection
- Gap detection (max 15 min)
- Calibration drift monitoring

### Simulator Mode
For development/testing:
```bash
export USE_CEMS_SIMULATOR=true
export CEMS_SIMULATOR_SCENARIO=normal  # or exceedance, failure, drift
```

## License

MIT License - Copyright (c) 2025 GreenLang

## Support

- Documentation: https://docs.greenlang.ai/agents/GL-010
- Issues: https://github.com/greenlang/gl-010/issues
- Email: devops@greenlang.ai
- Compliance Support: compliance@greenlang.ai
