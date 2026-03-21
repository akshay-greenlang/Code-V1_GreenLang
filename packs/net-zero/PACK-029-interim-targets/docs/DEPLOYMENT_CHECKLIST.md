# PACK-029 Interim Targets Pack -- Deployment Checklist

**Pack ID:** PACK-029-interim-targets
**Version:** 1.0.0
**Last Updated:** 2026-03-19

---

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Database Migration Guide](#database-migration-guide)
3. [Docker Build Instructions](#docker-build-instructions)
4. [Kubernetes Deployment](#kubernetes-deployment)
5. [Environment Variable Configuration](#environment-variable-configuration)
6. [Health Check Endpoints](#health-check-endpoints)
7. [Monitoring and Alerting Setup](#monitoring-and-alerting-setup)
8. [Troubleshooting Guide](#troubleshooting-guide)
9. [Rollback Procedures](#rollback-procedures)
10. [Production Readiness Checklist](#production-readiness-checklist)

---

## Prerequisites

### Software Requirements

| Component | Minimum Version | Recommended Version | Notes |
|-----------|---------------|-------------------|-------|
| Python | 3.11.0 | 3.12.x | Pydantic v2 required |
| PostgreSQL | 16.0 | 16.2 | TimescaleDB extension required |
| TimescaleDB | 2.13.0 | 2.14.x | Hypertable for time-series monitoring data |
| Redis | 7.0 | 7.2.x | Used for caching and session management |
| Kubernetes | 1.28 | 1.29 | For container orchestration |
| Docker | 24.0 | 25.x | For containerization |
| Helm | 3.13 | 3.14 | For K8s deployment management |

### Platform Migrations Required

| Migration Range | Description | Required |
|----------------|-------------|----------|
| V001-V006 | Core platform tables | Yes |
| V007-V008 | Feature flags + Agent Factory | Yes |
| V009-V010 | Auth + RBAC | Yes |
| V011-V018 | Security components | Yes |
| V019-V020 | Observability | Yes |
| V021-V128 | Agent + App tables | Yes |
| V196-V210 | **PACK-029 specific tables** | **Yes** |

### Hardware Requirements

| Resource | Minimum | Recommended | Production |
|----------|---------|-------------|------------|
| CPU | 2 vCPU | 4 vCPU | 8 vCPU |
| RAM | 4 GB | 8 GB | 16 GB |
| Storage (DB) | 5 GB | 20 GB | 100 GB |
| Storage (App) | 1 GB | 2 GB | 5 GB |
| Network | 100 Mbps | 1 Gbps | 10 Gbps |

---

## Database Migration Guide

### Migration Overview (V196-V210)

| Migration | Description | Tables Created | Indexes |
|-----------|-------------|---------------|---------|
| V196 | Interim targets core tables | `gl_interim_targets`, `gl_interim_milestones` | 15 |
| V197 | Annual pathways | `gl_annual_pathways`, `gl_pathway_points` | 12 |
| V198 | Quarterly monitoring | `gl_quarterly_monitoring`, `gl_quarterly_alerts` | 18 |
| V199 | Annual reviews | `gl_annual_reviews` | 10 |
| V200 | Variance analysis | `gl_variance_analyses`, `gl_lmdi_decompositions` | 16 |
| V201 | Trend extrapolation | `gl_trend_forecasts` | 8 |
| V202 | Corrective actions | `gl_corrective_actions`, `gl_action_initiatives` | 14 |
| V203 | Target recalibration | `gl_recalibration_events` | 8 |
| V204 | SBTi validation | `gl_sbti_validations` | 10 |
| V205 | Carbon budget | `gl_carbon_budgets` | 8 |
| V206 | Alert configuration | `gl_alert_configs`, `gl_alert_history` | 12 |
| V207 | CDP/TCFD exports | `gl_regulatory_exports` | 8 |
| V208 | Audit trail | `gl_pack029_audit_trail` | 10 |
| V209 | Views and materialized views | 3 views | N/A |
| V210 | Performance indexes and constraints | N/A | 111 |

**Total: 15 tables, 3 views, 250+ indexes**

### Applying Migrations

```bash
# Verify current migration state
psql -h $DB_HOST -U $DB_USER -d greenlang -c \
  "SELECT version FROM schema_migrations ORDER BY version DESC LIMIT 5;"

# Apply PACK-029 migrations
cd packs/net-zero/PACK-029-interim-targets/migrations

# Apply each migration in order
for version in $(seq 196 210); do
    echo "Applying V${version}..."
    psql -h $DB_HOST -U $DB_USER -d greenlang -f "V${version}__PACK029_*.sql"
    if [ $? -ne 0 ]; then
        echo "ERROR: Migration V${version} failed!"
        exit 1
    fi
done

# Verify all migrations applied
psql -h $DB_HOST -U $DB_USER -d greenlang -c \
  "SELECT version, description, applied_at FROM schema_migrations WHERE version >= 'V196' ORDER BY version;"
```

### Rollback Migrations

```bash
# Rollback in reverse order
for version in $(seq 210 -1 196); do
    echo "Rolling back V${version}..."
    psql -h $DB_HOST -U $DB_USER -d greenlang -f "V${version}__PACK029_*.down.sql"
done
```

---

## Docker Build Instructions

### Dockerfile

```dockerfile
FROM python:3.12-slim AS base

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "from integrations.health_check import HealthCheck; print(HealthCheck().run().status)"

EXPOSE 8029

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8029", "--workers", "4"]
```

### Build and Push

```bash
# Build
docker build -t greenlang/pack-029-interim-targets:1.0.0 .
docker tag greenlang/pack-029-interim-targets:1.0.0 greenlang/pack-029-interim-targets:latest

# Test locally
docker run -p 8029:8029 \
    -e INTERIM_TARGETS_DB_HOST=host.docker.internal \
    -e INTERIM_TARGETS_REDIS_HOST=host.docker.internal \
    greenlang/pack-029-interim-targets:1.0.0

# Push to registry
docker push greenlang/pack-029-interim-targets:1.0.0
docker push greenlang/pack-029-interim-targets:latest
```

---

## Kubernetes Deployment

### Deployment Manifest

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: pack-029-interim-targets
  namespace: greenlang
  labels:
    app: pack-029
    pack: interim-targets
    version: "1.0.0"
spec:
  replicas: 2
  selector:
    matchLabels:
      app: pack-029
  template:
    metadata:
      labels:
        app: pack-029
        pack: interim-targets
      annotations:
        prometheus.io/scrape: "true"
        prometheus.io/port: "8029"
        prometheus.io/path: "/metrics"
    spec:
      containers:
        - name: pack-029
          image: greenlang/pack-029-interim-targets:1.0.0
          ports:
            - containerPort: 8029
              name: http
          envFrom:
            - configMapRef:
                name: pack-029-config
            - secretRef:
                name: pack-029-secrets
          resources:
            requests:
              cpu: "500m"
              memory: "512Mi"
            limits:
              cpu: "2000m"
              memory: "2Gi"
          livenessProbe:
            httpGet:
              path: /health/live
              port: 8029
            initialDelaySeconds: 10
            periodSeconds: 30
          readinessProbe:
            httpGet:
              path: /health/ready
              port: 8029
            initialDelaySeconds: 5
            periodSeconds: 10
---
apiVersion: v1
kind: Service
metadata:
  name: pack-029-service
  namespace: greenlang
spec:
  selector:
    app: pack-029
  ports:
    - port: 8029
      targetPort: 8029
      name: http
  type: ClusterIP
```

### ConfigMap

```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: pack-029-config
  namespace: greenlang
data:
  INTERIM_TARGETS_DB_HOST: "postgresql.greenlang.svc.cluster.local"
  INTERIM_TARGETS_DB_PORT: "5432"
  INTERIM_TARGETS_DB_NAME: "greenlang"
  INTERIM_TARGETS_REDIS_HOST: "redis.greenlang.svc.cluster.local"
  INTERIM_TARGETS_REDIS_PORT: "6379"
  INTERIM_TARGETS_LOG_LEVEL: "INFO"
  INTERIM_TARGETS_PROVENANCE: "true"
  INTERIM_TARGETS_PACK021_ENABLED: "true"
  INTERIM_TARGETS_PACK028_ENABLED: "true"
```

---

## Environment Variable Configuration

### Required Variables

| Variable | Description | Example |
|----------|-------------|---------|
| `INTERIM_TARGETS_DB_HOST` | PostgreSQL host | `localhost` |
| `INTERIM_TARGETS_DB_PORT` | PostgreSQL port | `5432` |
| `INTERIM_TARGETS_DB_NAME` | Database name | `greenlang` |
| `INTERIM_TARGETS_DB_USER` | Database user | `greenlang_app` |
| `INTERIM_TARGETS_DB_PASSWORD` | Database password | `***` |
| `INTERIM_TARGETS_REDIS_HOST` | Redis host | `localhost` |
| `INTERIM_TARGETS_REDIS_PORT` | Redis port | `6379` |

### Optional Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `INTERIM_TARGETS_LOG_LEVEL` | Log level | `INFO` |
| `INTERIM_TARGETS_PROVENANCE` | SHA-256 hashing | `true` |
| `INTERIM_TARGETS_CACHE_TTL` | Cache TTL (seconds) | `3600` |
| `INTERIM_TARGETS_PACK021_ENABLED` | PACK-021 integration | `true` |
| `INTERIM_TARGETS_PACK028_ENABLED` | PACK-028 integration | `true` |
| `INTERIM_TARGETS_MONITORING_FREQUENCY` | Default frequency | `quarterly` |
| `INTERIM_TARGETS_ALERT_EMAIL_ENABLED` | Email alerting | `false` |
| `INTERIM_TARGETS_ALERT_SLACK_WEBHOOK` | Slack webhook | `` |

---

## Health Check Endpoints

| Endpoint | Method | Purpose | Expected Response |
|----------|--------|---------|-------------------|
| `/health/live` | GET | Liveness probe | `200 {"status": "alive"}` |
| `/health/ready` | GET | Readiness probe | `200 {"status": "ready"}` |
| `/health/full` | GET | Full 20-category check | `200 {"overall_score": 100}` |

### Full Health Check Categories

| Category | Checks |
|----------|--------|
| Database connectivity | PostgreSQL connection, table existence |
| Redis connectivity | Redis ping, key operations |
| Migration status | All V196-V210 applied |
| Engine availability | All 10 engines loadable |
| Workflow availability | All 7 workflows loadable |
| Template availability | All 10 templates loadable |
| Integration connectivity | PACK-021, PACK-028, MRV bridges |
| SBTi thresholds | Threshold data loaded |
| Cache health | Cache hit/miss ratio |
| Disk space | Sufficient storage |

---

## Monitoring and Alerting Setup

### Prometheus Metrics

```yaml
# Prometheus scrape config
- job_name: 'pack-029-interim-targets'
  metrics_path: '/metrics'
  static_configs:
    - targets: ['pack-029-service:8029']
```

### Key Metrics

| Metric | Type | Description |
|--------|------|-------------|
| `pack029_engine_requests_total` | Counter | Total engine requests |
| `pack029_engine_duration_seconds` | Histogram | Engine execution time |
| `pack029_workflow_requests_total` | Counter | Total workflow requests |
| `pack029_monitoring_rag_status` | Gauge | Current RAG status per entity |
| `pack029_alerts_generated_total` | Counter | Total alerts generated |
| `pack029_sbti_validations_total` | Counter | Total SBTi validations |
| `pack029_cache_hit_ratio` | Gauge | Redis cache hit ratio |
| `pack029_health_score` | Gauge | Overall health score |

### Grafana Dashboard

Import the PACK-029 Grafana dashboard from `config/grafana/pack029_dashboard.json`. Key panels:

1. **Engine Performance**: Request rate, latency percentiles, error rate
2. **RAG Status Distribution**: Count of GREEN/AMBER/RED entities
3. **Alert Activity**: Alert generation rate by type and severity
4. **SBTi Compliance**: Validation results over time
5. **Budget Burn Rate**: Carbon budget consumption trend

---

## Troubleshooting Guide

### Common Deployment Issues

#### Migration Fails with "relation already exists"

**Cause:** Migration was partially applied or applied out of order.

**Resolution:**
```sql
-- Check current state
SELECT * FROM schema_migrations WHERE version LIKE 'V19%' OR version LIKE 'V20%';

-- Drop conflicting objects if needed (CAUTION: data loss)
DROP TABLE IF EXISTS gl_interim_targets CASCADE;

-- Re-run migration
psql -f V196__PACK029_interim_targets.sql
```

#### Health Check Shows "Database Not Ready"

**Cause:** PostgreSQL connection parameters incorrect or migrations not applied.

**Resolution:**
```bash
# Verify connection
psql -h $INTERIM_TARGETS_DB_HOST -U $INTERIM_TARGETS_DB_USER -d $INTERIM_TARGETS_DB_NAME -c "SELECT 1;"

# Verify tables exist
psql -c "SELECT table_name FROM information_schema.tables WHERE table_name LIKE 'gl_interim%';"
```

#### Redis Connection Refused

**Cause:** Redis not running or wrong host/port configuration.

**Resolution:**
```bash
# Verify Redis
redis-cli -h $INTERIM_TARGETS_REDIS_HOST -p $INTERIM_TARGETS_REDIS_PORT ping
# Expected: PONG
```

#### Kubernetes Pod CrashLoopBackOff

**Cause:** Missing environment variables, database connectivity, or insufficient resources.

**Resolution:**
```bash
# Check pod logs
kubectl logs -n greenlang deployment/pack-029-interim-targets --tail=100

# Check events
kubectl describe pod -n greenlang -l app=pack-029

# Increase resources if OOMKilled
kubectl set resources deployment/pack-029-interim-targets -n greenlang \
  --limits=memory=4Gi --requests=memory=1Gi
```

---

## Rollback Procedures

### Application Rollback

```bash
# Rollback Kubernetes deployment
kubectl rollout undo deployment/pack-029-interim-targets -n greenlang

# Or rollback to specific revision
kubectl rollout undo deployment/pack-029-interim-targets -n greenlang --to-revision=2

# Verify rollback
kubectl rollout status deployment/pack-029-interim-targets -n greenlang
```

### Database Rollback

```bash
# Apply down migrations in reverse order
for version in $(seq 210 -1 196); do
    psql -h $DB_HOST -U $DB_USER -d greenlang -f \
        "migrations/V${version}__PACK029_*.down.sql"
done
```

### Emergency Rollback (Full)

```bash
# 1. Scale down the deployment
kubectl scale deployment/pack-029-interim-targets -n greenlang --replicas=0

# 2. Roll back database
# (apply down migrations as above)

# 3. Remove configmap and secrets
kubectl delete configmap pack-029-config -n greenlang
kubectl delete secret pack-029-secrets -n greenlang

# 4. Delete the deployment
kubectl delete deployment pack-029-interim-targets -n greenlang
kubectl delete service pack-029-service -n greenlang
```

---

## Production Readiness Checklist

### Pre-Deployment

- [ ] All V001-V128 platform migrations applied
- [ ] PostgreSQL 16+ with TimescaleDB verified
- [ ] Redis 7+ operational and accessible
- [ ] Python 3.11+ runtime available
- [ ] Docker image built and tested locally
- [ ] Environment variables configured
- [ ] TLS certificates configured (if applicable)
- [ ] RBAC roles created for pack-029 permissions

### Deployment

- [ ] V196-V210 migrations applied successfully
- [ ] Docker image pushed to registry
- [ ] Kubernetes manifests applied
- [ ] ConfigMap and Secrets created
- [ ] Health check endpoints responding
- [ ] All 20 health check categories passing
- [ ] Prometheus metrics being scraped
- [ ] Grafana dashboard imported

### Post-Deployment

- [ ] Health check score >= 90/100
- [ ] Sample interim target calculation successful
- [ ] Sample quarterly monitoring successful
- [ ] LMDI variance analysis returning perfect decomposition
- [ ] SBTi validation returning correct results
- [ ] Alert channel connectivity verified (email/Slack/Teams)
- [ ] PACK-021 bridge connectivity verified
- [ ] PACK-028 bridge connectivity verified (if enabled)
- [ ] MRV bridge connectivity verified
- [ ] Performance within targets (p95 < 500ms for engines)
- [ ] Provenance hashing operational
- [ ] Audit trail writing correctly
- [ ] Backup procedures documented and tested
- [ ] Runbook created for on-call team
- [ ] Monitoring alerts configured (PagerDuty/OpsGenie)

---

**End of Deployment Checklist**
