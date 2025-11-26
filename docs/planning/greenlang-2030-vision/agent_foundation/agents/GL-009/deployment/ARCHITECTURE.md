# GL-009 THERMALIQ - Infrastructure Architecture

## System Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          INTERNET / EXTERNAL USERS                           │
└─────────────────────────────────┬───────────────────────────────────────────┘
                                  │
                                  │ HTTPS (TLS 1.3)
                                  │
                    ┌─────────────▼─────────────┐
                    │   Load Balancer (AWS ALB) │
                    │   - SSL Termination       │
                    │   - DDoS Protection       │
                    └─────────────┬─────────────┘
                                  │
                                  │
                    ┌─────────────▼─────────────┐
                    │  Kubernetes Ingress       │
                    │  (NGINX Ingress)          │
                    │  - Rate Limiting: 200/min │
                    │  - WebSocket Support      │
                    │  - CORS Configuration     │
                    └─────────────┬─────────────┘
                                  │
            ┌─────────────────────┼─────────────────────┐
            │                     │                     │
            │                     │                     │
┌───────────▼───────────┐ ┌──────▼──────┐ ┌───────────▼───────────┐
│ GL-009 Pod 1          │ │ GL-009 Pod 2│ │ GL-009 Pod 3          │
│ ┌───────────────────┐ │ │             │ │ ┌───────────────────┐ │
│ │ ThermalIQ App     │ │ │  (HPA: 3-10)│ │ │ ThermalIQ App     │ │
│ │ - Port 8000: API  │ │ │             │ │ │ - Port 8000: API  │ │
│ │ - Port 8001: Metrics│ │             │ │ │ - Port 8001: Metrics│ │
│ │ - Port 8002: Admin│ │ │             │ │ │ - Port 8002: Admin│ │
│ │ - Port 8003: WS   │ │ │             │ │ │ - Port 8003: WS   │ │
│ └───────────────────┘ │ │             │ │ └───────────────────┘ │
│                       │ │             │ │                       │
│ Resources:            │ │             │ │ Resources:            │
│ - CPU: 1-2 cores      │ │             │ │ - CPU: 1-2 cores      │
│ - Memory: 1-2Gi       │ │             │ │ - Memory: 1-2Gi       │
└───────┬───────────────┘ └─────────────┘ └───────┬───────────────┘
        │                                          │
        └──────────────────┬───────────────────────┘
                           │
        ┌──────────────────┼──────────────────┐
        │                  │                  │
┌───────▼────────┐ ┌──────▼──────┐ ┌─────────▼──────────┐
│ PostgreSQL RDS │ │ Redis Cache │ │ AWS Secrets Manager│
│ - Pool: 20     │ │ - Port 6379 │ │ - Database creds   │
│ - Port 5432    │ │ - TTL: 1h   │ │ - API keys         │
│ - Multi-AZ     │ │ - Cluster   │ │ - JWT secrets      │
└────────────────┘ └─────────────┘ └────────────────────┘
```

## Container Architecture (Dockerfile)

```
┌─────────────────────────────────────────────────────────────────┐
│ Stage 1: BUILDER                                                │
│ FROM: python:3.11-slim                                          │
│ ┌─────────────────────────────────────────────────────────────┐ │
│ │ Install build dependencies:                                 │ │
│ │ - gcc, g++, gfortran                                        │ │
│ │ - libopenblas-dev, liblapack-dev                            │ │
│ │ - libpq-dev, libssl-dev                                     │ │
│ └─────────────────────────────────────────────────────────────┘ │
│ ┌─────────────────────────────────────────────────────────────┐ │
│ │ Create virtual environment: /opt/venv                       │ │
│ │ Install Python dependencies:                                │ │
│ │ - NumPy, SciPy, Pandas (compiled from source)               │ │
│ │ - CoolProp, IAPWS (thermodynamic properties)                │ │
│ │ - FastAPI, Uvicorn (web framework)                          │ │
│ │ - Prometheus, OpenTelemetry (monitoring)                    │ │
│ └─────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│ Stage 2: SECURITY SCANNER (Optional)                            │
│ ┌─────────────────────────────────────────────────────────────┐ │
│ │ Run security scans:                                         │ │
│ │ - Bandit (code security)                                    │ │
│ │ - Safety (dependency vulnerabilities)                       │ │
│ │ - pip-audit (known CVEs)                                    │ │
│ └─────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│ Stage 3: RUNTIME                                                │
│ FROM: python:3.11-slim                                          │
│ ┌─────────────────────────────────────────────────────────────┐ │
│ │ Runtime dependencies only:                                  │ │
│ │ - libopenblas0, liblapack3 (no dev packages)                │ │
│ │ - libpq5, libssl3                                           │ │
│ │ - curl, tini                                                │ │
│ └─────────────────────────────────────────────────────────────┘ │
│ ┌─────────────────────────────────────────────────────────────┐ │
│ │ Copy /opt/venv from builder                                 │ │
│ │ Create non-root user: greenlang (UID 1000)                  │ │
│ │ Copy application code                                       │ │
│ │ Set security context:                                       │ │
│ │ - runAsNonRoot: true                                        │ │
│ │ - readOnlyRootFilesystem: true                              │ │
│ └─────────────────────────────────────────────────────────────┘ │
│                                                                 │
│ EXPOSE: 8000, 8001, 8002, 8003                                  │
│ HEALTHCHECK: GET /api/v1/health every 30s                       │
│ CMD: uvicorn thermaliq.main:app --workers 4                     │
└─────────────────────────────────────────────────────────────────┘
```

## Kubernetes Deployment Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│ Namespace: greenlang-production                                             │
│                                                                             │
│ ┌─────────────────────────────────────────────────────────────────────────┐ │
│ │ Deployment: prod-gl-009-thermaliq                                       │ │
│ │ ┌─────────────────────────────────────────────────────────────────────┐ │ │
│ │ │ ReplicaSet (managed by HPA: 3-10 replicas)                          │ │ │
│ │ │ ┌────────────┐ ┌────────────┐ ┌────────────┐ ┌────────────┐        │ │ │
│ │ │ │  Pod 1     │ │  Pod 2     │ │  Pod 3     │ │  Pod N     │        │ │ │
│ │ │ │            │ │            │ │            │ │  (scaled)  │        │ │ │
│ │ │ │ Init:      │ │ Init:      │ │ Init:      │ │            │        │ │ │
│ │ │ │ - DB check │ │ - DB check │ │ - DB check │ │            │        │ │ │
│ │ │ │ - Redis ✓  │ │ - Redis ✓  │ │ - Redis ✓  │ │            │        │ │ │
│ │ │ │            │ │            │ │            │ │            │        │ │ │
│ │ │ │ Container: │ │ Container: │ │ Container: │ │ Container: │        │ │ │
│ │ │ │ - CPU: 1-2 │ │ - CPU: 1-2 │ │ - CPU: 1-2 │ │ - CPU: 1-2 │        │ │ │
│ │ │ │ - Mem: 1-2G│ │ - Mem: 1-2G│ │ - Mem: 1-2G│ │ - Mem: 1-2G│        │ │ │
│ │ │ │            │ │            │ │            │ │            │        │ │ │
│ │ │ │ Probes:    │ │ Probes:    │ │ Probes:    │ │ Probes:    │        │ │ │
│ │ │ │ - Liveness │ │ - Liveness │ │ - Liveness │ │ - Liveness │        │ │ │
│ │ │ │ - Readiness│ │ - Readiness│ │ - Readiness│ │ - Readiness│        │ │ │
│ │ │ │ - Startup  │ │ - Startup  │ │ - Startup  │ │ - Startup  │        │ │ │
│ │ │ │            │ │            │ │            │ │            │        │ │ │
│ │ │ │ Volumes:   │ │ Volumes:   │ │ Volumes:   │ │ Volumes:   │        │ │ │
│ │ │ │ - logs     │ │ - logs     │ │ - logs     │ │ - logs     │        │ │ │
│ │ │ │ - cache    │ │ - cache    │ │ - cache    │ │ - cache    │        │ │ │
│ │ │ │ - data     │ │ - data     │ │ - data     │ │ - data     │        │ │ │
│ │ │ └────────────┘ └────────────┘ └────────────┘ └────────────┘        │ │ │
│ │ └─────────────────────────────────────────────────────────────────────┘ │ │
│ └─────────────────────────────────────────────────────────────────────────┘ │
│                                                                             │
│ ┌─────────────────────────────────────────────────────────────────────────┐ │
│ │ Services                                                                │ │
│ │ ┌────────────────────┐ ┌──────────────────┐ ┌────────────────────────┐ │ │
│ │ │ Main Service       │ │ Metrics Service  │ │ Admin Service          │ │ │
│ │ │ - Port 80 → 8000   │ │ - Port 8001      │ │ - Port 8002            │ │ │
│ │ │ - ClusterIP        │ │ - ClusterIP      │ │ - ClusterIP            │ │ │
│ │ └────────────────────┘ └──────────────────┘ └────────────────────────┘ │ │
│ └─────────────────────────────────────────────────────────────────────────┘ │
│                                                                             │
│ ┌─────────────────────────────────────────────────────────────────────────┐ │
│ │ HorizontalPodAutoscaler                                                 │ │
│ │ - Min: 3, Max: 10                                                       │ │
│ │ - Metrics: CPU (70%), Memory (80%), Queue Depth, Active Calculations    │ │
│ └─────────────────────────────────────────────────────────────────────────┘ │
│                                                                             │
│ ┌─────────────────────────────────────────────────────────────────────────┐ │
│ │ PodDisruptionBudget                                                     │ │
│ │ - minAvailable: 2 (always keep 2 pods running during disruptions)      │ │
│ └─────────────────────────────────────────────────────────────────────────┘ │
│                                                                             │
│ ┌─────────────────────────────────────────────────────────────────────────┐ │
│ │ NetworkPolicy                                                           │ │
│ │ - Ingress: NGINX Ingress, Prometheus, API Gateway                       │ │
│ │ - Egress: DNS, PostgreSQL, Redis, Jaeger, HTTPS, MQTT, Modbus, OPC UA  │ │
│ └─────────────────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────────┘
```

## CI/CD Pipeline Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│ GitHub Actions Workflow: gl-009-ci.yaml                                     │
│                                                                             │
│ Trigger: Push to main/staging, PR, Manual                                  │
└───────────────────────────────┬─────────────────────────────────────────────┘
                                │
                ┌───────────────┴───────────────┐
                │                               │
        ┌───────▼────────┐            ┌────────▼──────────┐
        │ Job 1: Lint    │            │ Job 2: Security   │
        │ - Ruff         │            │ - Bandit          │
        │ - Black        │            │ - Safety          │
        │ - isort        │            │ - pip-audit       │
        │ - mypy         │            │ - Upload reports  │
        └───────┬────────┘            └────────┬──────────┘
                │                              │
                └───────────────┬──────────────┘
                                │
                        ┌───────▼────────┐
                        │ Job 3: Test    │
                        │ - PostgreSQL   │
                        │ - Redis        │
                        │ - pytest       │
                        │ - Coverage     │
                        │ - Codecov      │
                        └───────┬────────┘
                                │
                    ┌───────────┴───────────┐
                    │                       │
            ┌───────▼────────┐    ┌────────▼──────────┐
            │ Job 4:         │    │ Job 5: Build      │
            │ Integration    │    │ - Docker Buildx   │
            │ - E2E tests    │    │ - Push to GCR     │
            │ - DB migration │    │ - Trivy scan      │
            └───────┬────────┘    └────────┬──────────┘
                    │                      │
                    └──────────┬───────────┘
                               │
                    ┌──────────▼──────────┐
                    │ Job 6: Deploy       │
                    │ Staging             │
                    │ - Kustomize apply   │
                    │ - Rollout wait      │
                    └──────────┬──────────┘
                               │
                    ┌──────────▼──────────┐
                    │ Job 7: Smoke Tests  │
                    │ - Health check      │
                    │ - API test          │
                    └──────────┬──────────┘
                               │
                    ┌──────────▼──────────┐
                    │ Job 8: Deploy       │
                    │ Production          │
                    │ - Manual approval   │
                    │ - Kustomize apply   │
                    │ - Slack notify      │
                    └─────────────────────┘
```

## Monitoring Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│ Observability Stack                                                         │
│                                                                             │
│ ┌─────────────────────────────────────────────────────────────────────────┐ │
│ │ Metrics (Prometheus)                                                    │ │
│ │ ┌────────────────┐                                                      │ │
│ │ │ ServiceMonitor │──► Scrapes /api/v1/metrics every 30s                │ │
│ │ └────────────────┘                                                      │ │
│ │                                                                         │ │
│ │ Metrics Collected:                                                      │ │
│ │ - http_requests_total (counter)                                         │ │
│ │ - http_request_duration_seconds (histogram)                             │ │
│ │ - calculation_queue_depth (gauge)                                       │ │
│ │ - active_calculations (gauge)                                           │ │
│ │ - database_connections_active (gauge)                                   │ │
│ │ - redis_connection_errors_total (counter)                               │ │
│ │ - pod_cpu_usage (gauge)                                                 │ │
│ │ - pod_memory_usage (gauge)                                              │ │
│ └─────────────────────────────────────────────────────────────────────────┘ │
│                                                                             │
│ ┌─────────────────────────────────────────────────────────────────────────┐ │
│ │ Alerts (PrometheusRule)                                                 │ │
│ │ - GL009HighErrorRate (>5% for 5m) → CRITICAL                            │ │
│ │ - GL009HighLatency (p95 >5s for 10m) → WARNING                          │ │
│ │ - GL009PodDown (<2 pods for 5m) → CRITICAL                              │ │
│ │ - GL009HighMemoryUsage (>90% for 10m) → WARNING                         │ │
│ │ - GL009HighCPUUsage (>1.8 cores for 10m) → WARNING                      │ │
│ │ - GL009HighQueueDepth (>100 for 5m) → WARNING                           │ │
│ │ - GL009DatabasePoolExhaustion (>90% for 5m) → CRITICAL                  │ │
│ │ - GL009RedisConnectionFailure (>10 errors for 5m) → CRITICAL            │ │
│ └─────────────────────────────────────────────────────────────────────────┘ │
│                                                                             │
│ ┌─────────────────────────────────────────────────────────────────────────┐ │
│ │ Dashboards (Grafana)                                                    │ │
│ │ - Request Rate (per second)                                             │ │
│ │ - Error Rate (5xx errors)                                               │ │
│ │ - Response Time (p50, p95, p99)                                         │ │
│ │ - Active Calculations                                                   │ │
│ │ - Pod CPU/Memory Usage                                                  │ │
│ │ - Queue Depth                                                           │ │
│ └─────────────────────────────────────────────────────────────────────────┘ │
│                                                                             │
│ ┌─────────────────────────────────────────────────────────────────────────┐ │
│ │ Tracing (Jaeger + OpenTelemetry)                                        │ │
│ │ - Distributed tracing across services                                   │ │
│ │ - Trace sample rate: 10% (production)                                   │ │
│ │ - Spans: HTTP requests, DB queries, Redis operations                    │ │
│ └─────────────────────────────────────────────────────────────────────────┘ │
│                                                                             │
│ ┌─────────────────────────────────────────────────────────────────────────┐ │
│ │ Error Tracking (Sentry)                                                 │ │
│ │ - Application errors and exceptions                                     │ │
│ │ - Performance monitoring                                                │ │
│ │ - User context and breadcrumbs                                          │ │
│ └─────────────────────────────────────────────────────────────────────────┘ │
│                                                                             │
│ ┌─────────────────────────────────────────────────────────────────────────┐ │
│ │ Logs (ELK / Loki)                                                       │ │
│ │ - Structured JSON logs                                                  │ │
│ │ - Log levels: DEBUG, INFO, WARNING, ERROR                               │ │
│ │ - Retention: 7 days (production)                                        │ │
│ └─────────────────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Network Flow

```
External Client
      │
      │ HTTPS (443)
      ▼
AWS Load Balancer (ALB)
      │
      │ TLS Termination
      ▼
Kubernetes Ingress (NGINX)
      │
      │ Rate Limiting: 200 req/min
      │ WebSocket Upgrade
      ▼
Service: gl-009-thermaliq (ClusterIP:80)
      │
      ├─────► Pod 1 (10.0.1.23:8000)
      ├─────► Pod 2 (10.0.1.45:8000)
      └─────► Pod 3 (10.0.1.67:8000)
              │
              ├─────► PostgreSQL RDS (5432)
              ├─────► Redis ElastiCache (6379)
              ├─────► AWS Secrets Manager (HTTPS)
              ├─────► Jaeger Collector (14268)
              └─────► External APIs (443)
```

## Data Flow (Thermal Calculation)

```
Client Request
      │
      ▼
[API Gateway] → Authentication & Rate Limiting
      │
      ▼
[Load Balancer] → Session Affinity
      │
      ▼
[GL-009 Pod] → Receive calculation request
      │
      ├─────► [Redis] → Check cache
      │       │
      │       ├─► Cache Hit → Return cached result
      │       │
      │       └─► Cache Miss
      │             │
      ├─────────────┘
      │
      ▼
[Calculation Engine]
      │
      ├─────► Load thermodynamic properties (CoolProp, IAPWS)
      ├─────► Perform heat transfer calculations (NumPy, SciPy)
      ├─────► Optimize efficiency (SciPy optimization)
      ├─────► Generate visualizations (Plotly, Matplotlib)
      │
      ▼
[Store Result]
      │
      ├─────► [PostgreSQL] → Persist calculation
      ├─────► [Redis] → Cache result (TTL: 1h)
      │
      ▼
[Response] → Return to client
      │
      ├─────► [Prometheus] → Record metrics
      ├─────► [Jaeger] → Trace distributed request
      └─────► [Sentry] → Log errors (if any)
```

## High Availability Strategy

```
┌─────────────────────────────────────────────────────────────────┐
│ Geographic Distribution                                         │
│                                                                 │
│ Region: us-east-1                                               │
│ ┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐   │
│ │ AZ: us-east-1a  │ │ AZ: us-east-1b  │ │ AZ: us-east-1c  │   │
│ │ - GL-009 Pod 1  │ │ - GL-009 Pod 2  │ │ - GL-009 Pod 3  │   │
│ │ - Node 1        │ │ - Node 2        │ │ - Node 3        │   │
│ └─────────────────┘ └─────────────────┘ └─────────────────┘   │
│                                                                 │
│ Pod Anti-Affinity: Ensures pods spread across nodes/zones      │
│ PodDisruptionBudget: minAvailable=2 (always 2 pods running)    │
│ HPA: Auto-scale 3-10 based on CPU/memory/custom metrics        │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│ Database High Availability                                      │
│ - PostgreSQL RDS: Multi-AZ deployment                           │
│ - Automated backups: Daily, 30-day retention                    │
│ - Read replicas: 2 read replicas for scaling                    │
│                                                                 │
│ Cache High Availability                                         │
│ - Redis ElastiCache: Cluster mode                              │
│ - Automatic failover                                            │
│ - Snapshot backups: Daily                                       │
└─────────────────────────────────────────────────────────────────┘
```

## Security Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│ Defense in Depth                                                │
│                                                                 │
│ Layer 1: Network Security                                       │
│ - AWS Security Groups (firewall)                                │
│ - Kubernetes NetworkPolicy (ingress/egress rules)               │
│ - Private subnets for pods                                      │
│                                                                 │
│ Layer 2: Application Security                                   │
│ - Rate limiting (200 req/min)                                   │
│ - API key authentication                                        │
│ - JWT token validation                                          │
│ - Input validation (Pydantic)                                   │
│ - SQL injection protection                                      │
│ - XSS protection                                                │
│                                                                 │
│ Layer 3: Container Security                                     │
│ - Non-root user (UID 1000)                                      │
│ - Read-only root filesystem                                     │
│ - No privilege escalation                                       │
│ - Dropped capabilities (except NET_BIND_SERVICE)                │
│ - Seccomp profile: RuntimeDefault                               │
│                                                                 │
│ Layer 4: Secret Management                                      │
│ - External Secrets Operator                                     │
│ - AWS Secrets Manager (encrypted at rest)                       │
│ - Automatic secret rotation (1h refresh)                        │
│ - No secrets in Git                                             │
│                                                                 │
│ Layer 5: Vulnerability Scanning                                 │
│ - Trivy (container scanning)                                    │
│ - Bandit (code security)                                        │
│ - Safety (dependency vulnerabilities)                           │
│ - pip-audit (known CVEs)                                        │
└─────────────────────────────────────────────────────────────────┘
```

## Disaster Recovery

```
┌─────────────────────────────────────────────────────────────────┐
│ Backup Strategy                                                 │
│ - Database: Daily automated backups (30-day retention)          │
│ - Redis: Daily snapshots                                        │
│ - Application state: Stateless (no local state)                 │
│ - Configuration: Version controlled (Git)                       │
│                                                                 │
│ Recovery Time Objective (RTO): 15 minutes                       │
│ Recovery Point Objective (RPO): 24 hours                        │
│                                                                 │
│ Rollback Strategy                                               │
│ - Kubernetes: kubectl rollout undo                              │
│ - Database: Point-in-time recovery                              │
│ - DNS: Weighted routing for gradual traffic shift              │
└─────────────────────────────────────────────────────────────────┘
```

## Cost Optimization

```
┌─────────────────────────────────────────────────────────────────┐
│ Resource Efficiency                                             │
│ - HPA: Scale down to 3 pods during low traffic                  │
│ - Spot instances: Use for non-production environments           │
│ - Reserved instances: Production database (30-50% savings)      │
│ - S3 lifecycle policies: Archive old calculation data           │
│                                                                 │
│ Monthly Cost Estimate (Production)                              │
│ - EKS cluster: $150                                             │
│ - EC2 instances (3-10 pods): $300-$1000                         │
│ - RDS PostgreSQL (db.t3.medium): $150                           │
│ - ElastiCache Redis (cache.t3.medium): $100                     │
│ - Data transfer: $50                                            │
│ - Load balancer: $25                                            │
│ Total: $775-$1475/month                                         │
└─────────────────────────────────────────────────────────────────┘
```
