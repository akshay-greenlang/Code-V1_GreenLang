# AGENT-EUDR-001: Deployment Guide

This document covers deployment, configuration, database migration, scaling,
monitoring, and performance tuning for the Supply Chain Mapping Master agent.

---

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Configuration Reference](#configuration-reference)
3. [Database Migration](#database-migration)
4. [Kubernetes Deployment](#kubernetes-deployment)
5. [Docker Deployment](#docker-deployment)
6. [Scaling Recommendations](#scaling-recommendations)
7. [Monitoring Setup](#monitoring-setup)
8. [Performance Tuning](#performance-tuning)
9. [Disaster Recovery](#disaster-recovery)
10. [Operational Runbook](#operational-runbook)

---

## Prerequisites

| Component | Minimum Version | Purpose |
|-----------|----------------|---------|
| Python | 3.11+ | Runtime |
| PostgreSQL | 15+ with TimescaleDB | Persistent storage |
| PostGIS | 3.3+ | Spatial queries |
| pgvector | 0.5+ | Vector similarity (optional) |
| Redis | 7.0+ | Caching, distributed locks |
| Kubernetes | 1.28+ | Container orchestration |
| Docker | 24+ | Container builds |
| Prometheus | 2.45+ | Metrics collection |
| Grafana | 11.0+ | Dashboard visualization |

### Python Dependencies

```
pydantic>=2.0
fastapi>=0.110.0
uvicorn[standard]>=0.27.0
psycopg[binary,pool]>=3.1.0
aioredis>=2.0.0
networkx>=3.2
prometheus_client>=0.19.0
opentelemetry-api>=1.22.0
opentelemetry-sdk>=1.22.0
```

---

## Configuration Reference

All settings use the `GL_EUDR_SCM_` environment variable prefix. The
`SupplyChainMapperConfig` class validates all values at startup and raises
`ValueError` with detailed messages for any constraint violations.

### Connection Settings

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `GL_EUDR_SCM_DATABASE_URL` | string | `postgresql://localhost:5432/greenlang` | PostgreSQL connection URL |
| `GL_EUDR_SCM_REDIS_URL` | string | `redis://localhost:6379/0` | Redis connection URL |
| `GL_EUDR_SCM_POOL_SIZE` | integer | 10 | PostgreSQL connection pool size |
| `GL_EUDR_SCM_CACHE_TTL` | integer | 3600 | Redis cache TTL in seconds |

### Logging

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `GL_EUDR_SCM_LOG_LEVEL` | string | `INFO` | Log level: DEBUG, INFO, WARNING, ERROR, CRITICAL |

### Risk Propagation Weights

All four weights must sum to exactly 1.0.

| Variable | Type | Default | Range | Description |
|----------|------|---------|-------|-------------|
| `GL_EUDR_SCM_RISK_WEIGHT_COUNTRY` | float | 0.30 | 0.0-1.0 | Country-level risk weight |
| `GL_EUDR_SCM_RISK_WEIGHT_COMMODITY` | float | 0.20 | 0.0-1.0 | Commodity-level risk weight |
| `GL_EUDR_SCM_RISK_WEIGHT_SUPPLIER` | float | 0.25 | 0.0-1.0 | Supplier-level risk weight |
| `GL_EUDR_SCM_RISK_WEIGHT_DEFORESTATION` | float | 0.25 | 0.0-1.0 | Deforestation risk weight |

### Graph Capacity Limits

| Variable | Type | Default | Max | Description |
|----------|------|---------|-----|-------------|
| `GL_EUDR_SCM_MAX_NODES_PER_GRAPH` | integer | 100,000 | 10,000,000 | Max nodes before sharding |
| `GL_EUDR_SCM_MAX_EDGES_PER_GRAPH` | integer | 500,000 | 50,000,000 | Max edges per graph |
| `GL_EUDR_SCM_MAX_TIER_DEPTH` | integer | 50 | 1,000 | Max recursive tier depth |

### Performance Targets

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `GL_EUDR_SCM_GRAPH_QUERY_TIMEOUT_MS` | integer | 500 | Query timeout in ms |
| `GL_EUDR_SCM_BATCH_THROUGHPUT_TARGET` | integer | 50,000 | Target custody transfers/min |
| `GL_EUDR_SCM_MEMORY_LIMIT_MB` | integer | 2,048 | Max memory for in-memory graph |

### Risk Classification Thresholds

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `GL_EUDR_SCM_RISK_HIGH_THRESHOLD` | float | 70.0 | Score >= this = HIGH risk |
| `GL_EUDR_SCM_RISK_LOW_THRESHOLD` | float | 30.0 | Score <= this = LOW risk |

Note: `risk_low_threshold` must be strictly less than `risk_high_threshold`.

### Gap Analysis Settings

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `GL_EUDR_SCM_MASS_BALANCE_TOLERANCE` | float | 2.0 | Tolerance % for mass balance checks |
| `GL_EUDR_SCM_STALE_DATA_DAYS` | integer | 365 | Days before data flagged as stale |

### Provenance Tracking

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `GL_EUDR_SCM_ENABLE_PROVENANCE` | boolean | true | Enable SHA-256 provenance chain |
| `GL_EUDR_SCM_GENESIS_HASH` | string | `GL-EUDR-SCM-001-SUPPLY-CHAIN-MAPPER-GENESIS` | Genesis anchor string |

### Metrics and Rate Limiting

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `GL_EUDR_SCM_ENABLE_METRICS` | boolean | true | Enable Prometheus metrics |
| `GL_EUDR_SCM_RATE_LIMIT` | integer | 1,000 | Max API requests per minute |

### Example .env File

```env
# Connections
GL_EUDR_SCM_DATABASE_URL=postgresql://eudr_user:password@postgres.svc:5432/greenlang
GL_EUDR_SCM_REDIS_URL=redis://redis.svc:6379/3
GL_EUDR_SCM_POOL_SIZE=20

# Logging
GL_EUDR_SCM_LOG_LEVEL=INFO

# Risk weights (must sum to 1.0)
GL_EUDR_SCM_RISK_WEIGHT_COUNTRY=0.30
GL_EUDR_SCM_RISK_WEIGHT_COMMODITY=0.20
GL_EUDR_SCM_RISK_WEIGHT_SUPPLIER=0.25
GL_EUDR_SCM_RISK_WEIGHT_DEFORESTATION=0.25

# Risk thresholds
GL_EUDR_SCM_RISK_HIGH_THRESHOLD=70.0
GL_EUDR_SCM_RISK_LOW_THRESHOLD=30.0

# Graph capacity
GL_EUDR_SCM_MAX_NODES_PER_GRAPH=100000
GL_EUDR_SCM_MAX_EDGES_PER_GRAPH=500000
GL_EUDR_SCM_MAX_TIER_DEPTH=50

# Performance
GL_EUDR_SCM_GRAPH_QUERY_TIMEOUT_MS=500
GL_EUDR_SCM_BATCH_THROUGHPUT_TARGET=50000
GL_EUDR_SCM_MEMORY_LIMIT_MB=2048

# Gap analysis
GL_EUDR_SCM_MASS_BALANCE_TOLERANCE=2.0
GL_EUDR_SCM_STALE_DATA_DAYS=365

# Provenance
GL_EUDR_SCM_ENABLE_PROVENANCE=true

# Metrics and rate limit
GL_EUDR_SCM_ENABLE_METRICS=true
GL_EUDR_SCM_RATE_LIMIT=1000
GL_EUDR_SCM_CACHE_TTL=3600
```

---

## Database Migration

The Supply Chain Mapper uses migration **V089** to create its database schema.

### Migration File

```
deployment/migrations/V089__eudr_supply_chain_mapper.sql
```

### Schema Created

The migration creates the `eudr_scm` schema with the following tables:

| Table | Description |
|-------|-------------|
| `eudr_scm.supply_chain_graphs` | Graph metadata (operator, commodity, version) |
| `eudr_scm.supply_chain_nodes` | Actor nodes with attributes and geolocation |
| `eudr_scm.supply_chain_edges` | Custody transfer edges with quantities |
| `eudr_scm.supply_chain_gaps` | Detected compliance gaps |
| `eudr_scm.risk_propagation_results` | Risk score history per node |
| `eudr_scm.graph_snapshots` | Immutable versioned graph snapshots |
| `eudr_scm.provenance_entries` | SHA-256 provenance chain records |
| `eudr_scm.onboarding_invitations` | Supplier onboarding invitations |
| `eudr_scm.onboarding_submissions` | Supplier submission data |
| `eudr_scm.dds_exports` | DDS export records and history |

### Running the Migration

```bash
# Using Flyway
flyway -url=jdbc:postgresql://localhost:5432/greenlang \
       -user=greenlang_admin \
       -password=$DB_PASSWORD \
       migrate

# Using psql directly
psql -h localhost -U greenlang_admin -d greenlang \
     -f deployment/migrations/V089__eudr_supply_chain_mapper.sql

# Verify migration
psql -h localhost -U greenlang_admin -d greenlang \
     -c "SELECT * FROM flyway_schema_history WHERE script LIKE '%V089%';"
```

### PostGIS Extension

Ensure PostGIS is installed for spatial queries:

```sql
CREATE EXTENSION IF NOT EXISTS postgis;
CREATE EXTENSION IF NOT EXISTS postgis_topology;

-- Verify
SELECT PostGIS_Version();
```

---

## Kubernetes Deployment

### Deployment Manifest

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: eudr-supply-chain-mapper
  namespace: greenlang
  labels:
    app: eudr-supply-chain-mapper
    agent: GL-EUDR-SCM-001
    version: "1.0.0"
spec:
  replicas: 3
  selector:
    matchLabels:
      app: eudr-supply-chain-mapper
  template:
    metadata:
      labels:
        app: eudr-supply-chain-mapper
        agent: GL-EUDR-SCM-001
      annotations:
        prometheus.io/scrape: "true"
        prometheus.io/port: "8080"
        prometheus.io/path: "/metrics"
    spec:
      containers:
        - name: eudr-scm
          image: greenlang/eudr-supply-chain-mapper:1.0.0
          ports:
            - containerPort: 8080
              name: http
              protocol: TCP
          env:
            - name: GL_EUDR_SCM_DATABASE_URL
              valueFrom:
                secretKeyRef:
                  name: eudr-scm-secrets
                  key: database-url
            - name: GL_EUDR_SCM_REDIS_URL
              valueFrom:
                secretKeyRef:
                  name: eudr-scm-secrets
                  key: redis-url
            - name: GL_EUDR_SCM_LOG_LEVEL
              value: "INFO"
            - name: GL_EUDR_SCM_POOL_SIZE
              value: "20"
            - name: GL_EUDR_SCM_ENABLE_METRICS
              value: "true"
            - name: GL_EUDR_SCM_ENABLE_PROVENANCE
              value: "true"
          resources:
            requests:
              cpu: 500m
              memory: 1Gi
            limits:
              cpu: 2000m
              memory: 4Gi
          livenessProbe:
            httpGet:
              path: /api/v1/eudr-scm/health
              port: 8080
            initialDelaySeconds: 15
            periodSeconds: 30
            timeoutSeconds: 5
            failureThreshold: 3
          readinessProbe:
            httpGet:
              path: /api/v1/eudr-scm/health
              port: 8080
            initialDelaySeconds: 10
            periodSeconds: 10
            timeoutSeconds: 3
            failureThreshold: 3
          startupProbe:
            httpGet:
              path: /api/v1/eudr-scm/health
              port: 8080
            initialDelaySeconds: 5
            periodSeconds: 5
            failureThreshold: 30
      topologySpreadConstraints:
        - maxSkew: 1
          topologyKey: kubernetes.io/hostname
          whenUnsatisfiable: DoNotSchedule
          labelSelector:
            matchLabels:
              app: eudr-supply-chain-mapper
```

### Service

```yaml
apiVersion: v1
kind: Service
metadata:
  name: eudr-supply-chain-mapper
  namespace: greenlang
  labels:
    app: eudr-supply-chain-mapper
spec:
  type: ClusterIP
  ports:
    - port: 8080
      targetPort: 8080
      protocol: TCP
      name: http
  selector:
    app: eudr-supply-chain-mapper
```

### Horizontal Pod Autoscaler

```yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: eudr-supply-chain-mapper-hpa
  namespace: greenlang
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: eudr-supply-chain-mapper
  minReplicas: 2
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
  behavior:
    scaleUp:
      stabilizationWindowSeconds: 60
      policies:
        - type: Percent
          value: 50
          periodSeconds: 60
    scaleDown:
      stabilizationWindowSeconds: 300
      policies:
        - type: Percent
          value: 25
          periodSeconds: 120
```

### Secrets

```yaml
apiVersion: v1
kind: Secret
metadata:
  name: eudr-scm-secrets
  namespace: greenlang
type: Opaque
data:
  database-url: <base64-encoded-url>
  redis-url: <base64-encoded-url>
```

---

## Docker Deployment

### Dockerfile

```dockerfile
FROM python:3.11-slim AS base

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq-dev \
    libgdal-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY greenlang/ greenlang/

# Non-root user
RUN useradd -r -s /bin/false appuser && chown -R appuser:appuser /app
USER appuser

EXPOSE 8080

CMD ["uvicorn", "greenlang.agents.eudr.supply_chain_mapper.api.router:router", \
     "--host", "0.0.0.0", "--port", "8080", "--workers", "4"]
```

### Docker Compose (Development)

```yaml
version: "3.9"
services:
  eudr-scm:
    build: .
    ports:
      - "8080:8080"
    environment:
      GL_EUDR_SCM_DATABASE_URL: postgresql://postgres:postgres@db:5432/greenlang
      GL_EUDR_SCM_REDIS_URL: redis://redis:6379/0
      GL_EUDR_SCM_LOG_LEVEL: DEBUG
      GL_EUDR_SCM_POOL_SIZE: 5
    depends_on:
      - db
      - redis

  db:
    image: postgis/postgis:15-3.3
    environment:
      POSTGRES_DB: greenlang
      POSTGRES_USER: postgres
      POSTGRES_PASSWORD: postgres
    volumes:
      - pgdata:/var/lib/postgresql/data
    ports:
      - "5432:5432"

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"

volumes:
  pgdata:
```

---

## Scaling Recommendations

### Horizontal Scaling

| Graph Size | Replicas | CPU (per pod) | Memory (per pod) | Pool Size |
|------------|----------|---------------|------------------|-----------|
| < 10,000 nodes | 2 | 500m | 1 Gi | 10 |
| 10,000-50,000 | 3 | 1000m | 2 Gi | 15 |
| 50,000-100,000 | 5 | 2000m | 4 Gi | 20 |
| > 100,000 | 8-10 | 2000m | 8 Gi | 30 |

### Database Scaling

- **Read replicas:** Route read-heavy endpoints (GET /graphs, GET /trace) to replicas
- **Connection pooling:** Use PgBouncer in front of PostgreSQL for connection multiplexing
- **Table partitioning:** Partition `supply_chain_nodes` and `supply_chain_edges` by graph_id for graphs > 100K nodes
- **Spatial indexing:** GIST indexes on geometry columns for PostGIS queries

### Redis Scaling

- **Graph query cache:** TTL of 3600s for graph metadata, 300s for node lookups
- **Distributed locks:** Use Redis SETNX for concurrent graph mutations
- **Memory:** Allocate 1 GB per 100K cached nodes

---

## Monitoring Setup

### Prometheus Metrics

The agent exports 15 Prometheus metrics under the `gl_eudr_scm_` prefix.

**Counter Metrics:**

| Metric | Labels | Description |
|--------|--------|-------------|
| `gl_eudr_scm_graphs_created_total` | -- | Total graphs created |
| `gl_eudr_scm_nodes_added_total` | `node_type` | Nodes added by type |
| `gl_eudr_scm_edges_added_total` | -- | Edges added |
| `gl_eudr_scm_tier_discovery_total` | -- | Tier discovery operations |
| `gl_eudr_scm_trace_operations_total` | `direction` | Trace operations (forward/backward) |
| `gl_eudr_scm_risk_propagations_total` | -- | Risk propagation runs |
| `gl_eudr_scm_gaps_detected_total` | `gap_type`, `severity` | Gaps detected |
| `gl_eudr_scm_gaps_resolved_total` | -- | Gaps resolved |
| `gl_eudr_scm_dds_exports_total` | -- | DDS exports generated |
| `gl_eudr_scm_errors_total` | `operation` | Errors by operation |

**Histogram Metrics:**

| Metric | Labels | Buckets | Description |
|--------|--------|---------|-------------|
| `gl_eudr_scm_processing_duration_seconds` | `operation` | 0.01-60s | Operation duration |
| `gl_eudr_scm_graph_query_duration_seconds` | -- | 0.001-10s | Graph query latency |

**Gauge Metrics:**

| Metric | Description |
|--------|-------------|
| `gl_eudr_scm_active_graphs` | Currently active graphs |
| `gl_eudr_scm_total_nodes` | Total nodes across all graphs |
| `gl_eudr_scm_compliance_readiness_avg` | Average compliance readiness score |

### Prometheus Scrape Configuration

```yaml
scrape_configs:
  - job_name: 'eudr-supply-chain-mapper'
    kubernetes_sd_configs:
      - role: pod
        namespaces:
          names:
            - greenlang
    relabel_configs:
      - source_labels: [__meta_kubernetes_pod_label_app]
        regex: eudr-supply-chain-mapper
        action: keep
      - source_labels: [__meta_kubernetes_pod_annotation_prometheus_io_port]
        target_label: __address__
        regex: (.+)
        replacement: ${1}:8080
```

### Grafana Dashboard

Import the pre-built Grafana dashboard for the Supply Chain Mapper:

**Dashboard Panels:**

1. **Request Rate** -- Requests/sec by endpoint group
2. **Error Rate** -- Error percentage by operation
3. **P99 Latency** -- Response time distribution by endpoint
4. **Graphs Created** -- Graph creation rate over time
5. **Nodes & Edges** -- Total nodes and edges (gauges)
6. **Risk Distribution** -- Pie chart of LOW/STANDARD/HIGH
7. **Gap Detection** -- Gaps detected by type and severity
8. **Compliance Readiness** -- Average score over time
9. **Trace Operations** -- Forward vs backward trace volume
10. **DDS Exports** -- Export generation rate
11. **Memory Usage** -- Per-pod memory consumption
12. **CPU Usage** -- Per-pod CPU utilization
13. **Connection Pool** -- Active/idle database connections
14. **Cache Hit Rate** -- Redis cache hit percentage

### Alerting Rules

```yaml
groups:
  - name: eudr-supply-chain-mapper
    rules:
      - alert: EUDRSCMHighErrorRate
        expr: rate(gl_eudr_scm_errors_total[5m]) > 0.05
        for: 5m
        labels:
          severity: warning
          agent: GL-EUDR-SCM-001
        annotations:
          summary: "EUDR SCM error rate elevated"
          description: "Error rate > 5% for 5 minutes"

      - alert: EUDRSCMHighLatency
        expr: histogram_quantile(0.99, gl_eudr_scm_processing_duration_seconds) > 2.0
        for: 5m
        labels:
          severity: warning
          agent: GL-EUDR-SCM-001
        annotations:
          summary: "EUDR SCM p99 latency > 2s"

      - alert: EUDRSCMLowComplianceReadiness
        expr: gl_eudr_scm_compliance_readiness_avg < 50
        for: 1h
        labels:
          severity: info
          agent: GL-EUDR-SCM-001
        annotations:
          summary: "Average compliance readiness below 50%"

      - alert: EUDRSCMHighRiskNodes
        expr: gl_eudr_scm_gaps_detected_total{severity="critical"} > 0
        for: 0m
        labels:
          severity: critical
          agent: GL-EUDR-SCM-001
        annotations:
          summary: "Critical EUDR compliance gaps detected"
```

---

## Performance Tuning

### Database Indexes

The V089 migration creates the following indexes. Verify they exist:

```sql
-- Graph lookup by operator
CREATE INDEX IF NOT EXISTS idx_scm_graphs_operator
    ON eudr_scm.supply_chain_graphs (operator_id);

-- Node lookup by graph
CREATE INDEX IF NOT EXISTS idx_scm_nodes_graph
    ON eudr_scm.supply_chain_nodes (graph_id);

-- Node lookup by type and country
CREATE INDEX IF NOT EXISTS idx_scm_nodes_type_country
    ON eudr_scm.supply_chain_nodes (node_type, country_code);

-- Edge lookup by graph
CREATE INDEX IF NOT EXISTS idx_scm_edges_graph
    ON eudr_scm.supply_chain_edges (graph_id);

-- Edge lookup by batch number
CREATE INDEX IF NOT EXISTS idx_scm_edges_batch
    ON eudr_scm.supply_chain_edges (batch_number)
    WHERE batch_number IS NOT NULL;

-- Spatial index on node coordinates
CREATE INDEX IF NOT EXISTS idx_scm_nodes_geom
    ON eudr_scm.supply_chain_nodes
    USING GIST (ST_SetSRID(ST_Point(longitude, latitude), 4326))
    WHERE latitude IS NOT NULL AND longitude IS NOT NULL;

-- Gap lookup by severity
CREATE INDEX IF NOT EXISTS idx_scm_gaps_severity
    ON eudr_scm.supply_chain_gaps (severity, is_resolved);
```

### Connection Pool Tuning

```python
# For production with 20 concurrent requests per pod:
GL_EUDR_SCM_POOL_SIZE=20

# PostgreSQL max_connections should be:
# pool_size * num_pods + overhead = 20 * 5 + 20 = 120
```

### Cache Strategy

| Data | TTL | Invalidation |
|------|-----|-------------|
| Graph metadata | 3600s | On graph mutation |
| Node lookup | 300s | On node update |
| Tier distribution | 600s | On discovery |
| Risk summary | 300s | On risk propagation |
| Gap analysis result | 300s | On gap analyze/resolve |

---

## Disaster Recovery

### Backup Strategy

| Component | Frequency | Retention | Method |
|-----------|-----------|-----------|--------|
| PostgreSQL (graphs, nodes, edges) | Daily + WAL streaming | 30 days | pg_dump + continuous archiving |
| Redis (cache) | Not backed up | -- | Reconstructed from PostgreSQL |
| Provenance chain | Daily | Indefinite | pg_dump of provenance_entries |
| Graph snapshots | On creation | 5 years (EUDR Art. 31) | Immutable in PostgreSQL |

### Recovery Procedures

1. **Database restore:** Restore from latest pg_dump + replay WAL
2. **Cache warming:** On startup, the service automatically populates Redis from PostgreSQL
3. **Provenance verification:** Run `verify_provenance_chain()` after restore to confirm chain integrity
4. **Graph consistency check:** Run gap analysis on all active graphs to detect any data corruption

### RTO/RPO Targets

| Metric | Target |
|--------|--------|
| RPO (Recovery Point Objective) | < 1 minute (WAL streaming) |
| RTO (Recovery Time Objective) | < 15 minutes |

---

## Operational Runbook

### Common Operations

**Check service health:**
```bash
curl http://eudr-supply-chain-mapper:8080/api/v1/eudr-scm/health
```

**View active graph count:**
```bash
curl -s http://prometheus:9090/api/v1/query?query=gl_eudr_scm_active_graphs \
  | jq '.data.result[0].value[1]'
```

**Check compliance readiness:**
```bash
curl -s http://prometheus:9090/api/v1/query?query=gl_eudr_scm_compliance_readiness_avg \
  | jq '.data.result[0].value[1]'
```

**Force cache clear:**
```bash
redis-cli -h redis.svc SELECT 3
redis-cli -h redis.svc FLUSHDB
```

### Troubleshooting

| Symptom | Likely Cause | Resolution |
|---------|-------------|------------|
| 500 errors on /graphs | Database connection exhausted | Increase `GL_EUDR_SCM_POOL_SIZE` |
| High latency on trace | Large graph without indexes | Verify PostGIS GIST indexes exist |
| Memory OOM kills | Graph exceeds memory limit | Increase `GL_EUDR_SCM_MEMORY_LIMIT_MB` or shard graph |
| 429 rate limit errors | Client sending too fast | Implement exponential backoff in client |
| Risk weights error | Weights do not sum to 1.0 | Check `GL_EUDR_SCM_RISK_WEIGHT_*` env vars |
| Provenance chain broken | Database restored without WAL | Run `verify_provenance_chain()` and re-anchor |

---

## Related Documentation

- [README.md](README.md) -- Agent overview and quick start
- [API.md](API.md) -- Complete REST API reference
- [INTEGRATION.md](INTEGRATION.md) -- Integration guide with data agents
