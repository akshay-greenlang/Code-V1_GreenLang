# PACK-027 Enterprise Net Zero Pack -- Deployment Guide

**Pack ID:** PACK-027-enterprise-net-zero
**Version:** 1.0.0
**Date:** 2026-03-19
**Author:** GreenLang Platform Engineering

---

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Infrastructure Requirements](#infrastructure-requirements)
3. [Kubernetes Deployment](#kubernetes-deployment)
4. [Database Setup](#database-setup)
5. [Redis Configuration](#redis-configuration)
6. [Migration Guide](#migration-guide)
7. [OAuth and SSO Configuration](#oauth-and-sso-configuration)
8. [ERP Connector Setup](#erp-connector-setup)
9. [Environment Variables](#environment-variables)
10. [Production Checklist](#production-checklist)
11. [Monitoring and Alerting](#monitoring-and-alerting)
12. [Backup and Recovery](#backup-and-recovery)
13. [Scaling Guide](#scaling-guide)
14. [Upgrade Procedures](#upgrade-procedures)

---

## Prerequisites

### Platform Requirements

| Component | Minimum Version | Status Check |
|-----------|----------------|--------------|
| GreenLang Platform | v1.0.0 | `greenlang --version` |
| Python | 3.11+ | `python --version` |
| PostgreSQL | 16 + TimescaleDB | `psql -c "SELECT version()"` |
| Redis | 7+ | `redis-cli info server` |
| Kubernetes | 1.28+ | `kubectl version` |
| Docker | 24+ | `docker version` |
| Helm | 3.12+ | `helm version` |
| Terraform | 1.6+ (optional) | `terraform version` |

### Platform Migrations

Ensure all platform migrations are applied before deploying PACK-027:

```bash
# Check current migration version
psql -h $DB_HOST -U $DB_USER -d greenlang -c \
  "SELECT version FROM schema_migrations ORDER BY version DESC LIMIT 1;"

# Required: V128 or higher
# Platform migrations V001-V128 must be applied
```

---

## Infrastructure Requirements

### Compute Resources

| Component | CPU | Memory | Storage | Notes |
|-----------|-----|--------|---------|-------|
| Pack API pods (x3) | 2 vCPU each | 4 GB each | - | Auto-scale to 10 pods |
| Worker pods (x2) | 4 vCPU each | 8 GB each | - | Monte Carlo and batch processing |
| PostgreSQL | 4 vCPU | 16 GB | 100 GB SSD | 3-node HA cluster |
| Redis | 2 vCPU | 4 GB | 10 GB | 3-node sentinel cluster |
| S3 / Object Storage | - | - | 50 GB | Workpapers, reports, evidence |

### Network Requirements

| Direction | Destination | Port | Protocol | Purpose |
|-----------|-------------|------|----------|---------|
| Outbound | SAP S/4HANA | 443/8443 | HTTPS | ERP data extraction |
| Outbound | Oracle ERP Cloud | 443 | HTTPS | ERP data extraction |
| Outbound | Workday API | 443 | HTTPS | HCM data extraction |
| Outbound | CDP API | 443 | HTTPS | Questionnaire submission |
| Outbound | SBTi Portal | 443 | HTTPS | Target submission |
| Outbound | Vault | 8200 | HTTPS | Secrets management |
| Inbound | API clients | 443 | HTTPS | Pack API endpoints |
| Internal | PostgreSQL | 5432 | TCP | Database connectivity |
| Internal | Redis | 6379 | TCP | Cache connectivity |

---

## Kubernetes Deployment

### Namespace Setup

```bash
# Create namespace
kubectl create namespace greenlang-packs

# Apply resource quotas
kubectl apply -f - <<EOF
apiVersion: v1
kind: ResourceQuota
metadata:
  name: pack-027-quota
  namespace: greenlang-packs
spec:
  hard:
    requests.cpu: "20"
    requests.memory: 40Gi
    limits.cpu: "40"
    limits.memory: 80Gi
    pods: "20"
EOF
```

### Deployment Manifest

```yaml
# pack-027-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: pack-027-enterprise-net-zero
  namespace: greenlang-packs
  labels:
    app: greenlang
    component: solution-pack
    pack: enterprise-net-zero
    version: "1.0.0"
spec:
  replicas: 3
  selector:
    matchLabels:
      pack: enterprise-net-zero
  template:
    metadata:
      labels:
        app: greenlang
        pack: enterprise-net-zero
      annotations:
        prometheus.io/scrape: "true"
        prometheus.io/port: "8080"
        prometheus.io/path: "/metrics"
    spec:
      serviceAccountName: pack-027-sa
      containers:
      - name: pack-027
        image: greenlang/pack-027-enterprise-net-zero:1.0.0
        ports:
        - containerPort: 8080
          name: http
        - containerPort: 8081
          name: metrics
        resources:
          requests:
            cpu: "2"
            memory: 4Gi
          limits:
            cpu: "4"
            memory: 8Gi
        env:
        - name: ENT_NET_ZERO_LOG_LEVEL
          value: "INFO"
        - name: ENT_NET_ZERO_PROVENANCE
          value: "true"
        - name: ENT_NET_ZERO_MONTE_CARLO_WORKERS
          value: "4"
        - name: ENT_NET_ZERO_DB_HOST
          valueFrom:
            secretKeyRef:
              name: pack-027-db
              key: host
        - name: ENT_NET_ZERO_DB_PORT
          valueFrom:
            secretKeyRef:
              name: pack-027-db
              key: port
        - name: ENT_NET_ZERO_DB_NAME
          valueFrom:
            secretKeyRef:
              name: pack-027-db
              key: database
        - name: ENT_NET_ZERO_DB_USER
          valueFrom:
            secretKeyRef:
              name: pack-027-db
              key: username
        - name: ENT_NET_ZERO_DB_PASSWORD
          valueFrom:
            secretKeyRef:
              name: pack-027-db
              key: password
        - name: ENT_NET_ZERO_REDIS_HOST
          valueFrom:
            secretKeyRef:
              name: pack-027-redis
              key: host
        livenessProbe:
          httpGet:
            path: /health/live
            port: 8080
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /health/ready
            port: 8080
          initialDelaySeconds: 15
          periodSeconds: 5
        volumeMounts:
        - name: emission-factors
          mountPath: /app/data
          readOnly: true
      volumes:
      - name: emission-factors
        configMap:
          name: pack-027-emission-factors
---
apiVersion: v1
kind: Service
metadata:
  name: pack-027-service
  namespace: greenlang-packs
spec:
  selector:
    pack: enterprise-net-zero
  ports:
  - port: 80
    targetPort: 8080
    name: http
  type: ClusterIP
---
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: pack-027-hpa
  namespace: greenlang-packs
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: pack-027-enterprise-net-zero
  minReplicas: 3
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
```

### Worker Deployment

```yaml
# pack-027-worker.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: pack-027-worker
  namespace: greenlang-packs
spec:
  replicas: 2
  selector:
    matchLabels:
      pack: enterprise-net-zero-worker
  template:
    spec:
      containers:
      - name: worker
        image: greenlang/pack-027-enterprise-net-zero:1.0.0
        command: ["python", "-m", "celery", "worker"]
        resources:
          requests:
            cpu: "4"
            memory: 8Gi
          limits:
            cpu: "8"
            memory: 16Gi
        env:
        - name: ENT_NET_ZERO_WORKER_MODE
          value: "true"
        - name: ENT_NET_ZERO_MONTE_CARLO_WORKERS
          value: "8"
```

### Deploying

```bash
# Apply Kubernetes manifests
kubectl apply -f pack-027-deployment.yaml
kubectl apply -f pack-027-worker.yaml

# Verify deployment
kubectl -n greenlang-packs get pods -l pack=enterprise-net-zero
kubectl -n greenlang-packs get svc pack-027-service

# Check pod logs
kubectl -n greenlang-packs logs -l pack=enterprise-net-zero --tail=100
```

---

## Database Setup

### Apply Pack Migrations

```bash
# Apply all 15 PACK-027 specific migrations
for i in $(seq -w 1 15); do
  echo "Applying V083-PACK027-${i}..."
  psql -h $DB_HOST -U $DB_USER -d greenlang -f \
    deployment/migrations/V083-PACK027-${i}.sql
done

# Verify migrations
psql -h $DB_HOST -U $DB_USER -d greenlang -c \
  "SELECT * FROM schema_migrations WHERE version LIKE 'V083-PACK027%' ORDER BY version;"
```

### Enable Row-Level Security

```sql
-- Enable RLS on all pack tables
ALTER TABLE ent_corporate_profiles ENABLE ROW LEVEL SECURITY;
ALTER TABLE ent_entity_hierarchy ENABLE ROW LEVEL SECURITY;
ALTER TABLE ent_baselines ENABLE ROW LEVEL SECURITY;
ALTER TABLE ent_sbti_targets ENABLE ROW LEVEL SECURITY;
ALTER TABLE ent_scenarios ENABLE ROW LEVEL SECURITY;
ALTER TABLE ent_carbon_pricing ENABLE ROW LEVEL SECURITY;
ALTER TABLE ent_avoided_emissions ENABLE ROW LEVEL SECURITY;
ALTER TABLE ent_supply_chain ENABLE ROW LEVEL SECURITY;
ALTER TABLE ent_consolidation ENABLE ROW LEVEL SECURITY;
ALTER TABLE ent_financial_integration ENABLE ROW LEVEL SECURITY;
ALTER TABLE ent_assurance ENABLE ROW LEVEL SECURITY;
ALTER TABLE ent_data_quality ENABLE ROW LEVEL SECURITY;
ALTER TABLE ent_regulatory_filings ENABLE ROW LEVEL SECURITY;
ALTER TABLE ent_erp_connections ENABLE ROW LEVEL SECURITY;
ALTER TABLE ent_audit_trail ENABLE ROW LEVEL SECURITY;

-- Verify RLS
SELECT tablename, rowsecurity FROM pg_tables
WHERE tablename LIKE 'ent_%';
```

### Database Connection Pooling

```yaml
# PgBouncer configuration for PACK-027
[databases]
greenlang = host=postgres-primary port=5432 dbname=greenlang

[pgbouncer]
listen_port = 6432
listen_addr = *
auth_type = md5
pool_mode = transaction
max_client_conn = 200
default_pool_size = 20
min_pool_size = 5
reserve_pool_size = 5
```

---

## Redis Configuration

```bash
# Redis configuration for PACK-027
redis-cli CONFIG SET maxmemory 4gb
redis-cli CONFIG SET maxmemory-policy allkeys-lru

# Create cache key prefix
# PACK-027 uses prefix: "p027:"
# p027:ef:*     -- emission factors (TTL: 24h)
# p027:hier:*   -- entity hierarchy (TTL: 1h)
# p027:calc:*   -- calculation results (TTL: 1h)
# p027:sess:*   -- session data (TTL: 30m)
```

---

## Migration Guide

### Upgrading from PACK-026 (SME)

```python
from integrations.setup_wizard import SetupWizard

wizard = SetupWizard()

# Step 1: Import PACK-026 data
migration = wizard.migrate_from_pack026(
    source_pack="PACK-026-sme-net-zero",
    preserve_baseline=True,
    preserve_targets=True,
)

print(f"Baseline imported: {migration.baseline_imported}")
print(f"Targets imported: {migration.targets_imported}")
print(f"Recalculation needed: {migration.recalculation_needed}")

# Step 2: Recalculate with enterprise methodology
# Spend-based estimates are replaced with activity-based where data permits
# Scope 3 expands from 3 categories to 15
# SBTi pathway changes from SME to Corporate Standard
```

### Upgrading from PACK-022 (Acceleration)

```python
migration = wizard.migrate_from_pack022(
    source_pack="PACK-022-net-zero-acceleration",
    preserve_scenarios=True,
    preserve_suppliers=True,
)
```

---

## OAuth and SSO Configuration

### SAML 2.0 Setup

```yaml
# SSO configuration
auth:
  sso:
    provider: "saml2"
    idp_metadata_url: "https://idp.example.com/metadata"
    sp_entity_id: "greenlang-pack027"
    acs_url: "https://greenlang.example.com/auth/saml/callback"
    certificate_path: "/certs/sp-cert.pem"
    key_path: "/certs/sp-key.pem"
    attribute_mapping:
      email: "urn:oid:1.2.840.113549.1.9.1"
      name: "urn:oid:2.5.4.3"
      groups: "urn:oid:1.3.6.1.4.1.5923.1.5.1.1"
    role_mapping:
      "CN=Sustainability_Admins": "enterprise_admin"
      "CN=Sustainability_Team": "sustainability_manager"
      "CN=CSO_Office": "cso"
      "CN=Finance_Team": "finance_viewer"
      "CN=Audit_Team": "auditor"
      "CN=Board_Members": "board_viewer"
```

### OIDC Setup

```yaml
auth:
  sso:
    provider: "oidc"
    issuer: "https://idp.example.com"
    client_id: "greenlang-pack027"
    client_secret_vault_key: "auth/oidc/client_secret"
    redirect_uri: "https://greenlang.example.com/auth/oidc/callback"
    scopes: ["openid", "profile", "email", "groups"]
    role_claim: "groups"
```

---

## ERP Connector Setup

### SAP S/4HANA

```bash
# Store SAP credentials in Vault
vault kv put secret/erp/sap \
  host="sap.example.com" \
  client="100" \
  username="gl_service_user" \
  password="$(cat /secure/sap_password)"

# Configure SAP connector
export ENT_NET_ZERO_SAP_ENABLED=true
export ENT_NET_ZERO_SAP_VAULT_PATH="secret/erp/sap"
```

### Oracle ERP Cloud

```bash
vault kv put secret/erp/oracle \
  host="oracle.example.com" \
  tenant="org-tenant-id" \
  client_id="gl_integration" \
  client_secret="$(cat /secure/oracle_secret)"

export ENT_NET_ZERO_ORACLE_ENABLED=true
export ENT_NET_ZERO_ORACLE_VAULT_PATH="secret/erp/oracle"
```

### Workday HCM

```bash
vault kv put secret/erp/workday \
  tenant="tenant-id" \
  client_id="gl_integration" \
  client_secret="$(cat /secure/workday_secret)"

export ENT_NET_ZERO_WORKDAY_ENABLED=true
export ENT_NET_ZERO_WORKDAY_VAULT_PATH="secret/erp/workday"
```

---

## Environment Variables

### Complete Environment Variable Reference

```bash
# === DATABASE ===
ENT_NET_ZERO_DB_HOST=localhost
ENT_NET_ZERO_DB_PORT=5432
ENT_NET_ZERO_DB_NAME=greenlang
ENT_NET_ZERO_DB_USER=pack027_user
ENT_NET_ZERO_DB_PASSWORD=<from-vault>
ENT_NET_ZERO_DB_POOL_SIZE=20
ENT_NET_ZERO_DB_MAX_OVERFLOW=10

# === REDIS ===
ENT_NET_ZERO_REDIS_HOST=localhost
ENT_NET_ZERO_REDIS_PORT=6379
ENT_NET_ZERO_REDIS_DB=0
ENT_NET_ZERO_CACHE_TTL=3600

# === APPLICATION ===
ENT_NET_ZERO_LOG_LEVEL=INFO
ENT_NET_ZERO_PROVENANCE=true
ENT_NET_ZERO_PROVENANCE_ALGORITHM=sha256
ENT_NET_ZERO_MONTE_CARLO_WORKERS=4
ENT_NET_ZERO_BATCH_SIZE=20
ENT_NET_ZERO_MEMORY_CEILING_MB=4096
ENT_NET_ZERO_MAX_ENTITIES=500

# === ERP ===
ENT_NET_ZERO_SAP_ENABLED=false
ENT_NET_ZERO_ORACLE_ENABLED=false
ENT_NET_ZERO_WORKDAY_ENABLED=false

# === SECURITY ===
ENT_NET_ZERO_JWT_PUBLIC_KEY_PATH=/certs/jwt-public.pem
ENT_NET_ZERO_VAULT_ADDR=https://vault.example.com:8200
ENT_NET_ZERO_VAULT_TOKEN=<from-k8s-auth>

# === OBSERVABILITY ===
ENT_NET_ZERO_OTEL_ENDPOINT=http://otel-collector:4317
ENT_NET_ZERO_PROMETHEUS_PORT=8081
```

---

## Production Checklist

### Pre-Deployment

- [ ] Platform migrations V001-V128 applied
- [ ] Pack migrations V083-PACK027-001 to V083-PACK027-015 applied
- [ ] RLS enabled on all 15 pack tables
- [ ] Redis cluster healthy (3 nodes)
- [ ] PostgreSQL HA cluster healthy (3 nodes)
- [ ] S3 bucket created for workpapers and reports
- [ ] Vault secrets configured (DB, ERP, JWT)
- [ ] SSL/TLS certificates deployed
- [ ] DNS entries configured
- [ ] Network firewall rules applied
- [ ] SSO/OIDC integration tested

### Deployment

- [ ] Docker image built and pushed to registry
- [ ] Kubernetes manifests applied (deployment + worker + HPA)
- [ ] Pods in Running state (3 API + 2 worker)
- [ ] Liveness and readiness probes passing
- [ ] HPA configured and responding to load

### Post-Deployment

- [ ] Health check returns 100/100
- [ ] All 25 health categories passing
- [ ] API endpoints reachable through ingress
- [ ] MRV agent connectivity verified (30 agents)
- [ ] DATA agent connectivity verified (20 agents)
- [ ] FOUND agent connectivity verified (10 agents)
- [ ] ERP connector tested (if enabled)
- [ ] Prometheus metrics scraping
- [ ] Grafana dashboards configured
- [ ] Alert rules configured
- [ ] Backup schedule configured
- [ ] Disaster recovery tested
- [ ] Load test completed (100 entities, 10,000 MC runs)
- [ ] Security scan completed (no critical/high findings)

---

## Monitoring and Alerting

### Prometheus Metrics

```yaml
# Grafana dashboard: PACK-027 Operations
panels:
  - pack027_baseline_duration_seconds (histogram)
  - pack027_entities_processed_total (counter)
  - pack027_monte_carlo_runs_total (counter)
  - pack027_dq_score (gauge)
  - pack027_cache_hit_ratio (gauge)
  - pack027_erp_extraction_duration (histogram)
  - pack027_api_request_duration_seconds (histogram)
  - pack027_error_total (counter)
```

### Alert Rules

```yaml
groups:
- name: pack-027-alerts
  rules:
  - alert: Pack027BaselineFailure
    expr: increase(pack027_error_total{engine="baseline"}[5m]) > 0
    for: 1m
    labels:
      severity: critical
    annotations:
      summary: "Enterprise baseline calculation failed"

  - alert: Pack027HighMemory
    expr: pack027_memory_usage_bytes / pack027_memory_ceiling_bytes > 0.9
    for: 5m
    labels:
      severity: warning
    annotations:
      summary: "Pack-027 memory usage above 90%"

  - alert: Pack027ERPTimeout
    expr: pack027_erp_extraction_duration{quantile="0.95"} > 3600
    for: 10m
    labels:
      severity: warning
    annotations:
      summary: "ERP extraction taking longer than expected"
```

---

## Backup and Recovery

### Backup Schedule

| Data | Frequency | Retention | Method |
|------|-----------|-----------|--------|
| PostgreSQL (full) | Daily | 30 days | pg_dump + S3 |
| PostgreSQL (WAL) | Continuous | 7 days | WAL archiving to S3 |
| Redis (RDB) | Hourly | 24 hours | Redis BGSAVE |
| S3 (workpapers) | N/A | Versioned | S3 versioning |
| Configuration | On change | 90 days | Git + Vault |

### Recovery Procedures

```bash
# Database point-in-time recovery
pg_restore --host=$DB_HOST --dbname=greenlang --clean /backup/greenlang_20260319.dump

# Redis recovery
redis-cli FLUSHALL
redis-cli --pipe < /backup/redis_20260319.rdb

# Full cluster rebuild
terraform apply -target=module.pack027
kubectl apply -f pack-027-deployment.yaml
```

---

## Scaling Guide

### Horizontal Scaling

```bash
# Scale API pods
kubectl -n greenlang-packs scale deployment pack-027-enterprise-net-zero --replicas=6

# Scale worker pods (for Monte Carlo burst)
kubectl -n greenlang-packs scale deployment pack-027-worker --replicas=4
```

### Vertical Scaling

```bash
# Increase memory for Monte Carlo workloads
kubectl -n greenlang-packs set resources deployment pack-027-worker \
  --limits=cpu=16,memory=32Gi \
  --requests=cpu=8,memory=16Gi
```

---

## Upgrade Procedures

### Pack Version Upgrade

```bash
# 1. Build new image
docker build -t greenlang/pack-027-enterprise-net-zero:1.1.0 .

# 2. Apply any new migrations
psql -h $DB_HOST -U $DB_USER -d greenlang -f deployment/migrations/V083-PACK027-016.sql

# 3. Rolling update
kubectl -n greenlang-packs set image deployment/pack-027-enterprise-net-zero \
  pack-027=greenlang/pack-027-enterprise-net-zero:1.1.0

# 4. Monitor rollout
kubectl -n greenlang-packs rollout status deployment/pack-027-enterprise-net-zero

# 5. Run health check
kubectl -n greenlang-packs exec -it deploy/pack-027-enterprise-net-zero -- \
  python -c "from integrations.health_check import HealthCheck; print(HealthCheck().run())"
```

### Rollback

```bash
# Rollback to previous version
kubectl -n greenlang-packs rollout undo deployment/pack-027-enterprise-net-zero

# Verify rollback
kubectl -n greenlang-packs rollout status deployment/pack-027-enterprise-net-zero
```
