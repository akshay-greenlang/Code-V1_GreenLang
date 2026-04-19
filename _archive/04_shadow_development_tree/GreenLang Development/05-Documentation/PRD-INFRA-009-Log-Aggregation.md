# PRD: INFRA-009 - Log Aggregation (Loki Stack)

**Document Version:** 1.0
**Date:** February 4, 2026
**Status:** READY FOR EXECUTION
**Priority:** P1 - HIGH
**Owner:** Infrastructure Team
**Ralphy Task ID:** INFRA-009

---

## Executive Summary

Upgrade the GreenLang Climate OS log aggregation infrastructure from a development-grade monolithic Loki 2.9.2 deployment with filesystem storage and Promtail (EOL March 2026) to a production-ready Grafana Loki 3.x Simple Scalable Deployment (SSD) on AWS S3 with Grafana Alloy for log collection, multi-tenant isolation, structured JSON logging across all 47+ agents, log-based alerting rules, sensitive data redaction, and OpenTelemetry-native observability. This infrastructure supports regulatory compliance data retention for CSRD, CBAM, EUDR, and SB253 with audit log retention up to 365 days.

### Current State
- **Loki 2.9.2**: Monolithic single-replica StatefulSet, filesystem storage (50Gi PVC)
- **Promtail 2.9.2**: DaemonSet with 6 scrape configs (kubernetes-pods, eudr-agent, cbam-agent, api-gateway, audit-logs, systemd-journal)
- **Storage**: Local filesystem only (not production-scalable)
- **Auth**: Disabled (`auth_enabled: false`) - no multi-tenant isolation
- **Ring**: `inmemory` KV store (single instance only)
- **Retention**: Configured but only filesystem-backed compactor
- **Fluent Bit**: Helm template exists in `greenlang-agents` chart with multi-output support
- **OTel Collector**: Configured at `deployment/infrastructure/monitoring/jaeger/otel-collector-config.yaml` with Loki exporter
- **Python Logging**: Custom `StructuredLogger` at `greenlang/monitoring/telemetry/logging.py` using dataclass-based LogEntry with JSON output
- **Grafana Datasources**: Loki datasource already provisioned at `http://loki:3100`
- **Schema**: TSDB v13 already configured (good foundation)
- **Docker Compose**: Full monitoring stack exists for local development

### Target State
- **Loki 3.4+**: Simple Scalable Deployment (write/read/backend targets) with S3 storage
- **Grafana Alloy**: Replaces Promtail (EOL March 2026), unified log/metric/trace collection, OTLP-native
- **S3 Storage**: TSDB v13 with dedicated S3 buckets for chunks and ruler, IRSA authentication
- **Multi-Tenant**: `auth_enabled: true`, per-tenant ingestion limits, per-tenant retention policies
- **Structured Logging**: Production `structlog` framework with JSON output, correlation IDs, sensitive data redaction
- **Log-Based Alerting**: Loki Ruler with alerting rules and recording rules exported to Prometheus
- **Grafana Dashboards**: Log exploration dashboard, Loki operational meta-monitoring dashboard
- **Performance**: Chunk optimization, Memcached/Redis caching, ingestion rate limits per tenant
- **OpenTelemetry**: Native OTLP ingestion via Alloy, log-trace correlation with Tempo
- **Compliance**: 365-day audit log retention, 90-day compliance service logs, 30-day operational logs

---

## Scope

### In Scope
1. **Loki 3.x Helm Chart** - Production SSD deployment with S3 backend
2. **Grafana Alloy** - DaemonSet replacing Promtail with unified telemetry collection
3. **Terraform Module** - S3 buckets for Loki storage + IRSA for secure access
4. **Python Logging Framework** - Production structlog configuration with redaction and correlation
5. **Loki Ruler** - Log-based alerting and recording rules
6. **Grafana Dashboards** - Log exploration + Loki operational monitoring
7. **Multi-Tenant Configuration** - Per-tenant limits, retention, and isolation
8. **K8s Manifests** - Updated namespace, NetworkPolicies, ServiceMonitors
9. **Sensitive Data Redaction** - Pipeline stages for PII/credential filtering
10. **Migration Tooling** - Promtail-to-Alloy config converter, Loki 2.x to 3.x migration

### Out of Scope
- ELK/Elasticsearch stack (Loki chosen for cost, K8s-native, existing Grafana integration)
- Custom Loki plugins or forking
- Log archival to Glacier (handled by S3 lifecycle at infrastructure level)
- Centralized syslog from non-K8s sources
- Real-time log streaming WebSocket API

---

## Architecture

### Component Diagram

```
+-------------------+     +-------------------+     +-------------------+
| 47+ Python Agents |     | FastAPI Services  |     | K8s System Logs   |
| (structlog JSON)  |     | (structlog JSON)  |     | (kubelet, etcd)   |
+--------+----------+     +--------+----------+     +--------+----------+
         |                         |                          |
         v                         v                          v
    stdout/stderr             stdout/stderr              /var/log/pods
         |                         |                          |
         +-------------------------+--------------------------+
                                   |
                    +---------- Grafana Alloy ----------+
                    |  DaemonSet (1 per node)            |
                    |  - K8s SD pod log collection       |
                    |  - JSON pipeline stages            |
                    |  - Sensitive data redaction         |
                    |  - OTLP receiver (4317/4318)       |
                    |  - Prometheus metrics forwarding   |
                    +------+-------+-------+------------+
                           |       |       |
                    logs   |       |       |  traces
                           v       |       v
                    +------+--+    |    +--+------+
                    | Loki    |    |    | Tempo   |
                    | Gateway |    |    | (exist) |
                    | (nginx) |    |    +---------+
                    +----+----+    |
                         |         v
              +----------+---------+----------+
              |          |                     |
        +-----+---+ +---+----+  +--------+---+
        | Write   | | Read   |  | Backend    |
        | (3 rep) | | (2-6)  |  | (2 rep)    |
        | Distrib.| | Query  |  | Compactor  |
        | Ingester| | Frontend|  | Ruler      |
        +---------+ +--------+  | Index GW   |
              |          |       +------+-----+
              |          |              |
              +----------+--------------+
                         |
              +----------v-----------+
              |     AWS S3           |
              | gl-loki-chunks-prod  |
              | gl-loki-ruler-prod   |
              | Lifecycle tiering    |
              +----------------------+
                         |
              +----------v-----------+
              |     Grafana          |
              | + Loki datasource    |
              | + Log dashboards     |
              | + Alertmanager       |
              +----------------------+
```

### Data Flow

1. **Ingestion**: Agents write JSON to stdout -> Alloy DaemonSet collects via K8s SD -> pipeline stages parse/redact -> push to Loki Gateway
2. **Storage**: Gateway routes to Write path -> Distributor hashes to Ingester -> chunks compressed (snappy) -> flushed to S3
3. **Query**: Grafana/LogQL -> Gateway routes to Read path -> Query Frontend splits/caches -> Querier reads from Ingesters + S3
4. **Alerting**: Backend Ruler evaluates LogQL rules -> fires alerts to Alertmanager -> recording rules -> Prometheus remote_write
5. **Retention**: Backend Compactor runs retention policies -> deletes expired chunks from S3

---

## Technical Requirements

### TR-001: Loki 3.x Simple Scalable Deployment

**Helm Chart Configuration** (`deployment/helm/loki/`):
- Chart: `grafana/loki` v6.x
- Deployment mode: `SimpleScalable`
- Targets: write (3 replicas), read (2-6 autoscaled), backend (2 replicas)
- Gateway: nginx reverse proxy (2 replicas)
- Ring: `memberlist` (not `inmemory`)
- Schema: TSDB v13, S3 object store
- Retention: Compactor-based with per-stream policies
- Caching: embedded results cache (100MB), optional Memcached for chunks

**Environment Overlays**:
| Parameter | Dev | Staging | Production |
|-----------|-----|---------|------------|
| Write replicas | 1 | 2 | 3 |
| Read replicas | 1 | 2 | 2-6 (HPA) |
| Backend replicas | 1 | 1 | 2 |
| Gateway replicas | 1 | 1 | 2 |
| S3 storage | local-minio | S3 | S3 |
| Auth enabled | false | true | true |
| Ingestion rate | 5 MB/s | 10 MB/s | 20 MB/s |
| Retention default | 72h | 168h | 720h |

### TR-002: Grafana Alloy DaemonSet

**Replaces Promtail** (EOL March 2026):
- Helm Chart: `grafana/alloy` v1.x
- DaemonSet with tolerations for all nodes
- Pipeline components:
  - `loki.source.kubernetes` - pod log discovery
  - `loki.process` - JSON parsing, label extraction, redaction
  - `loki.write` - push to Loki gateway
  - `otelcol.receiver.otlp` - receive OTLP logs from instrumented apps
  - `prometheus.scrape` - forward metrics (optional)
- Sensitive data redaction pipeline:
  - Email addresses: `[REDACTED_EMAIL]`
  - API keys/tokens: `[REDACTED_KEY]`
  - Credit card numbers: `[REDACTED_CC]`
  - IP addresses in non-access logs: configurable
  - Custom patterns via ConfigMap

**Scrape Configurations** (migrated from Promtail):
1. `kubernetes-pods` - all GreenLang namespace pods
2. `eudr-agent` - EUDR compliance-specific fields
3. `cbam-agent` - CBAM calculation fields
4. `sb253-agent` - SB253 disclosure fields
5. `csrd-agent` - CSRD reporting fields
6. `api-gateway` - Kong access logs with duration metrics
7. `audit-logs` - compliance audit trail (365-day retention label)
8. `systemd-journal` - node system logs

### TR-003: S3 Storage Backend

**Terraform Module** (`deployment/terraform/modules/loki-storage/`):
- S3 buckets:
  - `gl-loki-chunks-{env}` - log chunk storage
  - `gl-loki-ruler-{env}` - ruler rule storage
- Bucket policies: deny public access, enforce TLS, versioning disabled
- Lifecycle rules:
  - 30 days: STANDARD -> STANDARD_IA
  - 90 days: STANDARD_IA -> GLACIER
- IRSA: IAM role for Loki ServiceAccount with S3 permissions
- KMS encryption at rest

### TR-004: Multi-Tenant Logging

- `auth_enabled: true` in production
- Tenant ID via `X-Scope-OrgID` header
- Alloy maps K8s namespaces to tenant IDs
- Per-tenant limits:
  | Tenant | Ingestion Rate | Burst | Streams | Retention |
  |--------|---------------|-------|---------|-----------|
  | greenlang-prod | 20 MB/s | 40 MB | 15000 | 720h |
  | greenlang-staging | 10 MB/s | 20 MB | 10000 | 168h |
  | greenlang-dev | 5 MB/s | 10 MB | 5000 | 72h |
  | compliance-audit | 5 MB/s | 10 MB | 5000 | 8760h |

### TR-005: Python Structured Logging Framework

**Module**: `greenlang/infrastructure/logging/`

Core components:
- `config.py` - Pydantic Settings for logging configuration
- `setup.py` - structlog configuration with JSON renderer
- `middleware.py` - FastAPI middleware for request correlation
- `redaction.py` - Sensitive data redaction processors
- `context.py` - Context variable management for correlation IDs
- `formatters.py` - Log formatters for different environments

**Standard Log Schema** (enforced across all 47+ agents):
```json
{
  "timestamp": "2026-02-04T12:00:00.000Z",
  "level": "INFO",
  "event": "emission_calculated",
  "logger": "greenlang.agents.csrd",
  "service": "csrd-agent",
  "version": "1.2.0",
  "environment": "production",
  "tenant_id": "tenant-001",
  "request_id": "req-abc123",
  "trace_id": "trace-xyz789",
  "span_id": "span-def456",
  "duration_ms": 42.5,
  "data": {}
}
```

### TR-006: Log-Based Alerting (Loki Ruler)

**Alert Rules** (`deployment/monitoring/alerts/loki-log-alerts.yaml`):
- High error rate per service (>0.5 errors/s for 5m)
- Compliance processing failures (CSRD/CBAM/EUDR/SB253)
- Emission calculation timeouts
- Authentication failures (>10 in 5m)
- Audit log gaps (no audit events for 15m)
- Log ingestion failures
- Sensitive data leak detection

**Recording Rules** (`deployment/monitoring/alerts/loki-recording-rules.yaml`):
- `greenlang:log_error_rate:5m` - error rate per service
- `greenlang:log_volume_bytes:1m` - ingestion volume per service
- `greenlang:log_latency_p99:5m` - request duration from access logs

### TR-007: Grafana Dashboards

1. **Log Exploration Dashboard** (`deployment/monitoring/dashboards/log-exploration.json`):
   - Service selector, level filter, time range
   - Log volume over time (stacked by level)
   - Error rate by service
   - Top error messages table
   - Log stream panel with JSON expansion
   - Correlation ID search

2. **Loki Operational Dashboard** (`deployment/monitoring/dashboards/loki-operations.json`):
   - Ingestion rate (bytes/s, lines/s)
   - Query latency (P50, P95, P99)
   - Chunk flush rate and size
   - Compactor progress
   - S3 operation rate and errors
   - Cache hit/miss ratio
   - Per-tenant ingestion breakdown

### TR-008: OpenTelemetry Integration

- Alloy receives OTLP logs via gRPC (4317) and HTTP (4318)
- Loki native OTLP endpoint at `/otlp`
- Structured Metadata stores OTel resource attributes
- Log-trace correlation: `traceId` and `spanId` fields enable Grafana Explore jump from logs to Tempo traces
- Service name auto-assignment from OTel `service.name` resource attribute

### TR-009: Performance Tuning

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| chunk_idle_period | 30m | Maximize chunk fill |
| chunk_target_size | 1572864 (1.5MB) | Optimal S3 object size |
| max_chunk_age | 2h | Bound memory usage |
| chunk_encoding | snappy | Fast compression |
| ingestion_rate_mb | 20 | Production throughput |
| per_stream_rate_limit | 3MB | Prevent hot streams |
| max_label_names_per_series | 15 | Loki 3.x default |
| max_query_parallelism | 16 | TSDB optimal |
| results_cache | 100MB embedded | Fast repeated queries |
| split_queries_by_interval | 30m | Query sharding |

### TR-010: Migration Plan

**Phase 1: Deploy Loki 3.x alongside existing Loki 2.9.2**
- Deploy new Helm chart with S3 backend
- Dual-write from Alloy to both Loki instances
- Validate data parity

**Phase 2: Migrate Promtail to Alloy**
- Deploy Alloy DaemonSet alongside Promtail
- Verify log collection parity
- Remove Promtail DaemonSet

**Phase 3: Cut over**
- Update Grafana datasource to new Loki
- Decommission Loki 2.9.2
- Remove old ConfigMaps and StatefulSet

---

## File Structure

```
# Helm Charts
deployment/helm/loki/Chart.yaml
deployment/helm/loki/values.yaml
deployment/helm/loki/values-dev.yaml
deployment/helm/loki/values-staging.yaml
deployment/helm/loki/values-prod.yaml
deployment/helm/alloy/Chart.yaml
deployment/helm/alloy/values.yaml
deployment/helm/alloy/values-dev.yaml
deployment/helm/alloy/values-staging.yaml
deployment/helm/alloy/values-prod.yaml
deployment/helm/alloy/templates/configmap.yaml
deployment/helm/alloy/templates/daemonset.yaml
deployment/helm/alloy/templates/serviceaccount.yaml
deployment/helm/alloy/templates/clusterrole.yaml
deployment/helm/alloy/templates/service.yaml

# Terraform
deployment/terraform/modules/loki-storage/main.tf
deployment/terraform/modules/loki-storage/variables.tf
deployment/terraform/modules/loki-storage/outputs.tf
deployment/terraform/modules/loki-storage/iam.tf

# Python Logging Framework
greenlang/infrastructure/logging/__init__.py
greenlang/infrastructure/logging/config.py
greenlang/infrastructure/logging/setup.py
greenlang/infrastructure/logging/middleware.py
greenlang/infrastructure/logging/redaction.py
greenlang/infrastructure/logging/context.py
greenlang/infrastructure/logging/formatters.py

# Monitoring
deployment/monitoring/dashboards/log-exploration.json
deployment/monitoring/dashboards/loki-operations.json
deployment/monitoring/alerts/loki-log-alerts.yaml
deployment/monitoring/alerts/loki-recording-rules.yaml

# K8s Manifests
deployment/kubernetes/loki/namespace-update.yaml
deployment/kubernetes/loki/networkpolicy.yaml
deployment/kubernetes/loki/servicemonitor.yaml

# Loki Ruler Rules
deployment/config/loki/rules/greenlang-alerts.yaml
deployment/config/loki/rules/greenlang-recording.yaml

# Tests
tests/unit/test_logging/__init__.py
tests/unit/test_logging/test_setup.py
tests/unit/test_logging/test_redaction.py
tests/unit/test_logging/test_middleware.py
tests/unit/test_logging/test_context.py
tests/unit/test_logging/test_formatters.py

# Documentation
.ralphy/INFRA-009-tasks.md
```

---

## Acceptance Criteria

1. Loki 3.x Helm chart deploys successfully in SSD mode with S3 storage
2. Grafana Alloy collects logs from all GreenLang namespace pods
3. Multi-tenant isolation prevents cross-tenant log access
4. All 47+ agents emit structlog JSON with consistent schema
5. Sensitive data (emails, API keys, CC numbers) redacted before ingestion
6. Log-based alerts fire correctly (error rate, compliance failures)
7. Recording rules export metrics to Prometheus
8. Grafana dashboards display log exploration and Loki operations
9. Audit logs retained for 365 days, compliance logs 90 days, operational 30 days
10. OTLP log ingestion works via Alloy (traces correlated with logs)
11. Query P99 latency < 5s for 24h range queries
12. Ingestion handles 20 MB/s sustained throughput
13. Migration from Promtail to Alloy completed with zero log loss
14. All unit tests pass with 85%+ coverage on Python logging framework

---

## Ralphy Task List

### Phase 1: Infrastructure (Terraform + Helm)
- [ ] Create Terraform module: loki-storage (S3 + IRSA)
- [ ] Create Loki Helm chart values (base + 3 env overlays)
- [ ] Create Alloy Helm chart with templates
- [ ] Create K8s manifests (namespace update, network policy, service monitor)

### Phase 2: Python Logging Framework
- [ ] Create logging config module
- [ ] Create structlog setup module
- [ ] Create FastAPI middleware
- [ ] Create sensitive data redaction
- [ ] Create context variable management
- [ ] Create log formatters

### Phase 3: Alerting & Dashboards
- [ ] Create Loki alerting rules
- [ ] Create Loki recording rules
- [ ] Create log exploration Grafana dashboard
- [ ] Create Loki operational Grafana dashboard

### Phase 4: Loki Ruler Rules
- [ ] Create GreenLang-specific alert rules
- [ ] Create recording rules for Prometheus export

### Phase 5: Tests
- [ ] Create logging setup tests
- [ ] Create redaction tests
- [ ] Create middleware tests
- [ ] Create context tests
- [ ] Create formatter tests

### Phase 6: Documentation
- [ ] Create Ralphy task checklist
- [ ] Update MEMORY.md
