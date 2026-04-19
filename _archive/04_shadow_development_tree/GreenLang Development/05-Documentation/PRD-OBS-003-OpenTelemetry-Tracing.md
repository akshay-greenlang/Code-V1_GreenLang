# PRD-OBS-003: OpenTelemetry Distributed Tracing Platform

**Component**: OBS-003 - OpenTelemetry Distributed Tracing
**Priority**: P1 - High
**Status**: Approved
**Version**: 1.0
**Date**: 2026-02-07
**Author**: GreenLang Platform Team
**Depends On**: OBS-001 (Prometheus), OBS-002 (Grafana), SEC-004 (TLS), SEC-005 (Audit Logging)

---

## 1. Executive Summary

Deploy a production-grade distributed tracing platform for GreenLang Climate OS using OpenTelemetry (OTel) as the unified telemetry standard and Grafana Tempo as the scalable trace backend. This replaces the existing Jaeger all-in-one deployment with a horizontally-scalable, S3-backed architecture capable of handling 10,000+ spans/second with 30-day retention. The platform consolidates 17+ scattered tracing implementations into a unified Python SDK, adds auto-instrumentation for all critical libraries (FastAPI, httpx, psycopg, Redis, Celery), and enables full trace-to-logs and trace-to-metrics correlation through the existing Grafana platform.

## 2. Current State Assessment

### 2.1 Existing Infrastructure

| Component | Location | State | Issues |
|-----------|----------|-------|--------|
| TracingManager | `greenlang/monitoring/telemetry/tracing.py` (583 lines) | Functional | Uses deprecated `opentelemetry.exporter.jaeger.thrift`; no auto-instrumentation |
| OTel Collector Config | `deployment/infrastructure/monitoring/jaeger/otel-collector-config.yaml` (213 lines) | Config only | References Tempo but Tempo not deployed; not in Helm chart |
| Jaeger Config | `deployment/infrastructure/monitoring/jaeger/jaeger-config.yaml` (241 lines) | Active | ES backend config but deployed with Badger; not production-grade |
| Jaeger Deployment | `deployment/infrastructure/monitoring/k8s/jaeger-deployment.yaml` (265 lines) | Active | All-in-one v1.51, single replica, Badger storage, no HA |
| Agent Factory Tracer | `greenlang/infrastructure/agent_factory/telemetry/tracer.py` | Active | Singleton pattern, separate from main TracingManager |
| Framework Tracing | `applications/GL Agents/Framework_GreenLang/observability/tracing.py` (912 lines) | Active | Custom Span/Exporter classes, duplicates OTel functionality |
| Agent-specific tracers | 17+ files across `greenlang/agents/*/tracing.py` | Active | Inconsistent implementations, no correlation |

### 2.2 Key Gaps

1. **No Production Trace Backend**: Jaeger all-in-one with Badger storage cannot scale; no HA, no S3 persistence
2. **No Grafana Tempo**: Referenced in OTel Collector config but never deployed
3. **No Helm Charts**: OTel Collector and Tempo have no Helm packaging
4. **No Terraform Module**: No S3 bucket, KMS, or IAM for trace storage
5. **Scattered Implementations**: 17+ separate tracing modules with inconsistent APIs
6. **No Auto-Instrumentation**: FastAPI, httpx, psycopg, Redis, Celery not auto-instrumented
7. **Deprecated Exporters**: Still using `opentelemetry.exporter.jaeger.thrift` (deprecated Jan 2024)
8. **No Correlation**: No trace-to-logs or trace-to-metrics linking in Grafana
9. **No Tracing Dashboards**: No Grafana dashboards for trace analytics
10. **No Tracing Alerts**: No alerts for trace pipeline health or latency anomalies

## 3. Architecture

### 3.1 High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                    GreenLang Application Layer                       │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐ │
│  │ API Svc  │ │ Agent    │ │ Jobs Svc │ │ Auth Svc │ │ Emissions│ │
│  │ (FastAPI)│ │ Factory  │ │          │ │          │ │ Pipeline │ │
│  └────┬─────┘ └────┬─────┘ └────┬─────┘ └────┬─────┘ └────┬─────┘ │
│       │             │            │             │            │       │
│  ┌────▼─────────────▼────────────▼─────────────▼────────────▼────┐ │
│  │              Unified Tracing SDK (greenlang/infrastructure/    │ │
│  │              tracing_service/) - Auto-Instrumentation          │ │
│  │              OTLP Export, W3C Context Propagation              │ │
│  └──────────────────────────┬────────────────────────────────────┘ │
└─────────────────────────────┼───────────────────────────────────────┘
                              │ OTLP gRPC/HTTP (4317/4318)
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│                  OTel Collector (Gateway Mode)                       │
│  ┌────────────┐ ┌─────────────────┐ ┌────────────────────────────┐ │
│  │ Receivers  │ │ Processors      │ │ Exporters                  │ │
│  │ - OTLP     │ │ - Memory Limiter│ │ - OTLP/Tempo (traces)     │ │
│  │ - Jaeger   │ │ - Resource      │ │ - Prometheus (metrics)     │ │
│  │ - Zipkin   │ │ - Attributes    │ │ - Loki (logs)             │ │
│  │            │ │ - Tail Sampling │ │                            │ │
│  │            │ │ - Filter        │ │                            │ │
│  │            │ │ - Batch         │ │                            │ │
│  └────────────┘ └─────────────────┘ └───────┬────────────────────┘ │
└─────────────────────────────────────────────┼───────────────────────┘
                                              │ OTLP gRPC
                                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    Grafana Tempo (Distributed)                       │
│  ┌────────────┐ ┌──────────┐ ┌──────────┐ ┌──────────────────────┐ │
│  │ Distributor│ │ Ingester │ │ Querier  │ │ Compactor            │ │
│  │ (2 replicas│ │(3 repli- │ │(2 repli- │ │(1 replica)           │ │
│  │  HA)       │ │ cas)     │ │ cas)     │ │                      │ │
│  └────────────┘ └────┬─────┘ └────┬─────┘ └──────────┬───────────┘ │
│                      │            │                   │             │
│                      ▼            ▼                   ▼             │
│               ┌──────────────────────────────────────────┐         │
│               │         S3 Backend Storage               │         │
│               │  (30-day retention, lifecycle tiering)    │         │
│               └──────────────────────────────────────────┘         │
└─────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    Grafana (OBS-002)                                 │
│  ┌───────────────────────────────────────────────────────────────┐ │
│  │ Tempo Datasource → Trace Search, Trace Detail, Service Graph  │ │
│  │ Trace-to-Logs (Loki) → Correlated log lines per span          │ │
│  │ Trace-to-Metrics (Prometheus) → Exemplars on metric panels    │ │
│  └───────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────┘
```

### 3.2 Component Selection

| Component | Technology | Version | Justification |
|-----------|-----------|---------|---------------|
| Trace Backend | Grafana Tempo | 2.7.x | Native Grafana integration, S3 backend, cost-effective, TraceQL |
| Collector | OpenTelemetry Collector | 0.96.x | Vendor-neutral, production-tested, rich processor ecosystem |
| SDK | OpenTelemetry Python | 1.24.x | Industry standard, auto-instrumentation, W3C propagation |
| Auto-Instrumentation | OTel Python contrib | 0.45b0 | FastAPI, httpx, psycopg, redis, celery instrumentors |
| Correlation | Grafana derived fields | N/A | Trace-to-logs via traceID field, exemplars for metrics |

### 3.3 Migration Strategy

Jaeger all-in-one → Grafana Tempo (phased):

1. **Phase 1**: Deploy Tempo alongside Jaeger; OTel Collector dual-writes to both
2. **Phase 2**: Switch Grafana datasource from Jaeger to Tempo; validate trace queries
3. **Phase 3**: Remove Jaeger deployment; update OTel Collector config
4. **Phase 4**: Remove Jaeger receiver from OTel Collector (after all services migrate to OTLP)

## 4. Detailed Component Specifications

### 4.1 Grafana Tempo Helm Chart

**Location**: `deployment/helm/tempo/`

**Files**:
- `Chart.yaml` - Tempo 2.7.x, appVersion 2.7.0
- `values.yaml` - Base configuration (distributed mode)
- `values-dev.yaml` - Single-binary mode, local storage, 1 replica
- `values-staging.yaml` - Distributed mode, S3, 2 replicas
- `values-prod.yaml` - Distributed mode, S3, full HA

**Architecture Mode**:
- **Dev**: `monolithic` (single binary, local filesystem, 7-day retention)
- **Staging/Prod**: `distributed` (distributor, ingester, querier, compactor, metrics-generator)

**Key Configuration**:
```yaml
tempo:
  # Storage backend
  storage:
    trace:
      backend: s3
      s3:
        bucket: gl-${environment}-tempo-traces
        region: eu-west-1
        endpoint: s3.eu-west-1.amazonaws.com
      wal:
        path: /var/tempo/wal
      block:
        version: vParquet4  # Latest columnar format
      blocklist_poll: 5m
      cache: redis  # Memcached alternative

  # Distributor
  distributor:
    receivers:
      otlp:
        protocols:
          grpc:
            endpoint: 0.0.0.0:4317
          http:
            endpoint: 0.0.0.0:4318
      jaeger:
        protocols:
          grpc:
            endpoint: 0.0.0.0:14250
    ring:
      kvstore:
        store: memberlist

  # Ingester
  ingester:
    max_block_duration: 5m
    max_block_bytes: 524288000  # 500MB
    flush_check_period: 10s
    trace_idle_period: 30s
    lifecycler:
      ring:
        replication_factor: 3

  # Querier
  querier:
    max_concurrent_queries: 20
    search:
      query_timeout: 30s
      external_endpoints: []

  # Compactor
  compactor:
    compaction:
      block_retention: 720h  # 30 days
      compacted_block_retention: 1h
      compaction_window: 1h
      max_block_bytes: 107374182400  # 100GB
    ring:
      kvstore:
        store: memberlist

  # Metrics Generator (span metrics → Prometheus)
  metrics_generator:
    storage:
      path: /var/tempo/generator/wal
      remote_write:
        - url: http://gl-prometheus-server:9090/api/v1/write
          send_exemplars: true
    processor:
      span_metrics:
        dimensions:
          - service.name
          - http.method
          - http.status_code
          - gl.tenant_id
          - gl.agent_type
        enable_target_info: true
      service_graphs:
        dimensions:
          - gl.tenant_id
        enable_client_server_prefix: true
        peer_attributes:
          - service.name
          - gl.agent_type

  # Global overrides
  overrides:
    defaults:
      ingestion:
        rate_limit_bytes: 15000000  # 15MB/s
        burst_size_bytes: 20000000  # 20MB
        max_traces_per_user: 0  # Unlimited
      search:
        max_duration: 168h  # 7 days
      metrics_generator:
        ring_size: 1
        processor:
          span_metrics:
            enable_target_info: true
```

**Resource Specifications**:

| Component | Dev | Staging | Prod |
|-----------|-----|---------|------|
| Distributor | N/A (monolithic) | 2 replicas, 256Mi/0.25 CPU | 3 replicas, 512Mi/0.5 CPU |
| Ingester | N/A | 2 replicas, 1Gi/0.5 CPU | 3 replicas, 2Gi/1 CPU, 10Gi PVC |
| Querier | N/A | 2 replicas, 512Mi/0.5 CPU | 2 replicas, 1Gi/0.5 CPU |
| Compactor | N/A | 1 replica, 512Mi/0.5 CPU | 1 replica, 2Gi/1 CPU |
| Metrics Generator | N/A | 1 replica, 256Mi/0.25 CPU | 2 replicas, 512Mi/0.5 CPU |
| Monolithic | 1 replica, 256Mi/0.25 CPU | N/A | N/A |

### 4.2 OpenTelemetry Collector Helm Chart

**Location**: `deployment/helm/otel-collector/`

**Files**:
- `Chart.yaml` - OTel Collector 0.96.x
- `values.yaml` - Base configuration
- `values-dev.yaml` - Minimal, debug exporter enabled
- `values-staging.yaml` - Tail sampling, moderate limits
- `values-prod.yaml` - Full tail sampling, production limits
- `templates/` - Deployment, Service, ConfigMap, ServiceAccount, HPA, PDB, NetworkPolicy, ServiceMonitor

**Deployment Topology**:
- **DaemonSet Agent** (optional, for node-level collection): Lightweight, forwards to Gateway
- **Deployment Gateway** (primary): Central processing, tail sampling, export

**Key Configuration (Gateway)**:
```yaml
otelCollector:
  mode: deployment  # deployment (gateway) or daemonset (agent)
  replicas: 2

  config:
    receivers:
      otlp:
        protocols:
          grpc:
            endpoint: 0.0.0.0:4317
            max_recv_msg_size_mib: 4
          http:
            endpoint: 0.0.0.0:4318
      # Jaeger receiver (migration compatibility)
      jaeger:
        protocols:
          grpc:
            endpoint: 0.0.0.0:14250
          thrift_http:
            endpoint: 0.0.0.0:14268

    processors:
      memory_limiter:
        check_interval: 1s
        limit_mib: 1536
        spike_limit_mib: 512

      resource:
        attributes:
          - key: service.environment
            from_attribute: ""
            value: "${ENVIRONMENT}"
            action: upsert
          - key: service.cluster
            value: "greenlang-${ENVIRONMENT}"
            action: upsert
          - key: service.platform
            value: "greenlang-climate-os"
            action: insert

      attributes/redact:
        actions:
          - key: http.request.header.authorization
            action: delete
          - key: http.request.header.cookie
            action: delete
          - key: db.statement
            action: hash
          - key: http.request.body
            action: delete

      filter/health:
        error_mode: ignore
        traces:
          span:
            - 'attributes["http.target"] == "/health"'
            - 'attributes["http.target"] == "/ready"'
            - 'attributes["http.target"] == "/ping"'
            - 'attributes["http.target"] == "/metrics"'

      tail_sampling:
        decision_wait: 10s
        num_traces: 50000
        expected_new_traces_per_sec: 500
        policies:
          # Always sample errors
          - name: errors
            type: status_code
            status_code:
              status_codes: [ERROR]
          # Always sample slow traces (>2s)
          - name: slow-traces
            type: latency
            latency:
              threshold_ms: 2000
          # 100% sampling for compliance agents
          - name: compliance-agents
            type: string_attribute
            string_attribute:
              key: service.name
              values: [eudr-agent, cbam-agent, sb253-agent, csrd-agent, vcci-agent, taxonomy-agent]
          # 100% sampling for auth failures
          - name: auth-failures
            type: string_attribute
            string_attribute:
              key: gl.auth.result
              values: [denied, failed, revoked]
          # 50% sampling for API services
          - name: api-services
            type: probabilistic
            probabilistic:
              sampling_percentage: 50
          # 10% default for everything else
          - name: default
            type: probabilistic
            probabilistic:
              sampling_percentage: 10

      batch:
        timeout: 2s
        send_batch_size: 1024
        send_batch_max_size: 2048

    exporters:
      otlp/tempo:
        endpoint: tempo-distributor:4317
        tls:
          insecure: true
        retry_on_failure:
          enabled: true
          initial_interval: 5s
          max_interval: 30s
          max_elapsed_time: 300s
      # Jaeger (migration period only)
      otlp/jaeger:
        endpoint: jaeger-collector:4317
        tls:
          insecure: true
      prometheus:
        endpoint: 0.0.0.0:8889
        namespace: gl_otel
        resource_to_telemetry_conversion:
          enabled: true
      loki:
        endpoint: http://loki-gateway:3100/loki/api/v1/push
        labels:
          attributes:
            service.name: "service"
            severity: "level"

    extensions:
      health_check:
        endpoint: 0.0.0.0:13133
      pprof:
        endpoint: 0.0.0.0:1777
      zpages:
        endpoint: 0.0.0.0:55679

    service:
      extensions: [health_check, pprof, zpages]
      pipelines:
        traces:
          receivers: [otlp, jaeger]
          processors: [memory_limiter, resource, attributes/redact, filter/health, tail_sampling, batch]
          exporters: [otlp/tempo]
        metrics:
          receivers: [otlp]
          processors: [memory_limiter, resource, batch]
          exporters: [prometheus]
        logs:
          receivers: [otlp]
          processors: [memory_limiter, resource, batch]
          exporters: [loki]
```

**Resource Specifications**:

| Environment | Replicas | CPU Request | Memory Request | CPU Limit | Memory Limit |
|-------------|----------|-------------|----------------|-----------|--------------|
| Dev | 1 | 100m | 256Mi | 500m | 512Mi |
| Staging | 2 | 250m | 512Mi | 1 | 1Gi |
| Prod | 3 | 500m | 1Gi | 2 | 2Gi |

### 4.3 Terraform Module

**Location**: `deployment/terraform/modules/tempo-storage/`

**Files**:
- `main.tf` - S3 bucket for Tempo blocks, lifecycle policies
- `variables.tf` - Input variables with validation
- `outputs.tf` - Bucket ARN, IRSA role ARN
- `iam.tf` - IRSA role for Tempo service account
- `kms.tf` - KMS key for trace encryption at rest

**S3 Bucket Configuration**:
```hcl
# Bucket: gl-${environment}-tempo-traces
# Lifecycle:
#   - Current: 30 days (Standard)
#   - Transition: 30→90 days (Infrequent Access)
#   - Expiration: 90 days
# Encryption: SSE-KMS with dedicated CMK
# Versioning: Disabled (Tempo manages blocks)
# Object Lock: Disabled (blocks are immutable by design)
```

**IRSA Policy**:
```hcl
# Allow Tempo pods to:
# - s3:GetObject, s3:PutObject, s3:DeleteObject on trace blocks
# - s3:ListBucket for block discovery
# - kms:Encrypt, kms:Decrypt, kms:GenerateDataKey for SSE-KMS
```

**Environment Configs**: `deployment/terraform/environments/{dev,staging,prod}/tempo.tf`

### 4.4 Unified Python Tracing SDK

**Location**: `greenlang/infrastructure/tracing_service/`

**Files**:

| File | Purpose | Lines (est.) |
|------|---------|-------------|
| `__init__.py` | Public API exports | ~50 |
| `config.py` | TracingConfig with environment defaults | ~120 |
| `provider.py` | TracerProvider setup, resource detection, exporter configuration | ~250 |
| `instrumentors.py` | Auto-instrumentation for FastAPI, httpx, psycopg, redis, celery | ~300 |
| `context.py` | W3C TraceContext propagation, baggage, tenant context injection | ~180 |
| `decorators.py` | @trace_operation, @trace_agent, @trace_pipeline decorators | ~200 |
| `span_enrichment.py` | GreenLang-specific span attributes (tenant_id, agent_type, emission_scope) | ~150 |
| `sampling.py` | Parent-based sampler with per-service overrides | ~120 |
| `middleware.py` | FastAPI TracingMiddleware (extract context, create server span, inject headers) | ~180 |
| `metrics_bridge.py` | Span-to-metrics bridge, histogram from trace duration | ~100 |
| `setup.py` | `configure_tracing(app)` one-liner setup function | ~80 |

**Total**: ~11 files, ~1,730 lines estimated

**Key Design Decisions**:

1. **Wraps existing TracingManager**: Does not replace `greenlang/monitoring/telemetry/tracing.py` but provides a higher-level, production-hardened layer on top
2. **OTLP-only export**: No Jaeger Thrift exporter; all traces go via OTLP gRPC to OTel Collector
3. **Auto-instrumentation**: Automatically instruments FastAPI, httpx, psycopg, redis, celery on startup
4. **Tenant context injection**: Every span automatically gets `gl.tenant_id` from request context
5. **GreenLang semantic conventions**: Custom span attributes following `gl.*` namespace convention

**Public API**:
```python
from greenlang.infrastructure.tracing_service import configure_tracing, trace_operation

# One-liner setup in main.py
configure_tracing(app, service_name="api-service")

# Decorator for custom spans
@trace_operation(name="calculate_emissions", attributes={"gl.scope": "scope_1"})
async def calculate_emissions(tenant_id: str, data: dict) -> dict:
    ...

# Manual span creation
from greenlang.infrastructure.tracing_service import get_tracer
tracer = get_tracer(__name__)
with tracer.start_as_current_span("custom-operation") as span:
    span.set_attribute("gl.custom_key", "value")
    ...
```

**Auto-Instrumentation Matrix**:

| Library | Instrumentor | Captured Attributes |
|---------|-------------|-------------------|
| FastAPI | `FastAPIInstrumentor` | http.method, http.route, http.status_code, http.url |
| httpx | `HTTPXClientInstrumentor` | http.method, http.url, http.status_code, peer.service |
| psycopg | `Psycopg2Instrumentor` | db.system, db.name, db.statement (hashed), db.operation |
| redis | `RedisInstrumentor` | db.system=redis, db.operation, net.peer.name |
| celery | `CeleryInstrumentor` | messaging.system, messaging.destination, celery.task_name |
| requests | `RequestsInstrumentor` | http.method, http.url, http.status_code |

**GreenLang Semantic Conventions** (`gl.*` namespace):

| Attribute | Type | Description |
|-----------|------|-------------|
| `gl.tenant_id` | string | Tenant identifier from request context |
| `gl.agent_type` | string | Agent type (e.g., csrd, eudr, cbam) |
| `gl.agent_id` | string | Agent instance UUID |
| `gl.pipeline_id` | string | Pipeline execution UUID |
| `gl.emission_scope` | string | Emission scope (scope_1, scope_2, scope_3) |
| `gl.regulation` | string | Regulation being processed (e.g., CSRD, EUDR) |
| `gl.data_source` | string | Data source identifier |
| `gl.calculation_type` | string | Calculation type (e.g., ghg_protocol, pcf) |
| `gl.environment` | string | Environment (dev, staging, prod) |

### 4.5 Kubernetes Manifests

**Location**: `deployment/kubernetes/otel-collector/`

**Files**:
- `namespace.yaml` - Namespace `monitoring` (shared with Prometheus, Grafana)
- `configmap.yaml` - OTel Collector configuration
- `deployment.yaml` - Gateway deployment with HPA
- `service.yaml` - ClusterIP for OTLP endpoints (4317, 4318)
- `servicemonitor.yaml` - Prometheus scrape config for collector metrics
- `networkpolicy.yaml` - Ingress from application namespaces, egress to Tempo/Prometheus/Loki
- `kustomization.yaml` - Kustomize overlay base

**Location**: `deployment/kubernetes/tempo/`

**Files**:
- `namespace.yaml` - Namespace `monitoring`
- `networkpolicy.yaml` - Ingress from OTel Collector, Grafana; egress to S3
- `servicemonitor.yaml` - Prometheus scrape for Tempo components
- `poddisruptionbudget.yaml` - PDB for ingester (minAvailable: 2)
- `kustomization.yaml`

### 4.6 Monitoring & Alerting

#### 4.6.1 Grafana Dashboards

**Location**: `deployment/monitoring/dashboards/`

| Dashboard | UID | Panels | Description |
|-----------|-----|--------|-------------|
| `tracing-overview.json` | `tracing-overview` | 16 | Service map, latency heatmap, error rates, throughput |
| `tempo-operations.json` | `tempo-operations` | 18 | Tempo distributor/ingester/querier/compactor health |
| `otel-collector.json` | `otel-collector` | 14 | Collector pipeline metrics, queue depth, drop rate |
| `trace-analytics.json` | `trace-analytics` | 12 | TraceQL explorer, span duration histograms, top operations |

**Folder**: `02-Observability` (existing from OBS-002)

#### 4.6.2 Alert Rules

**Location**: `deployment/monitoring/alerts/tracing-alerts.yaml`

| Alert | Severity | Condition |
|-------|----------|-----------|
| `TempoDistributorHighLatency` | warning | P99 ingestion latency > 500ms for 5m |
| `TempoIngesterFlushFailing` | critical | Flush failures > 0 for 10m |
| `TempoCompactorHalted` | critical | Compactor not running for 30m |
| `TempoStorageErrors` | critical | S3 write errors > 0 for 5m |
| `OTelCollectorDroppedSpans` | warning | Dropped spans > 100/min for 5m |
| `OTelCollectorQueueFull` | critical | Exporter queue at capacity for 5m |
| `OTelCollectorHighMemory` | warning | Memory usage > 80% of limit for 10m |
| `OTelCollectorDown` | critical | No collector instances running |
| `TracePipelineEndToEndLatency` | warning | E2E trace ingestion > 30s for 10m |
| `TraceSearchLatencyHigh` | warning | TraceQL search P99 > 10s for 10m |
| `TempoTenantRateLimited` | warning | Tenant rate limited for 5m |
| `NoTracesReceived` | critical | Zero spans received for 15m |

#### 4.6.3 Runbooks

**Location**: `docs/runbooks/`

| Runbook | Description |
|---------|-------------|
| `tempo-ingester-failures.md` | Diagnose flush failures, WAL corruption, OOM |
| `otel-collector-dropped-spans.md` | Pipeline bottlenecks, queue tuning, scaling |
| `trace-search-slow.md` | TraceQL optimization, compactor lag, bloom filters |

### 4.7 Trace-to-Logs & Trace-to-Metrics Correlation

#### Trace-to-Logs (Grafana Derived Fields)

Configure Tempo datasource in Grafana to link traces to Loki logs:

```yaml
# Tempo datasource config
tracesToLogs:
  datasourceUid: loki
  spanStartTimeShift: -1m
  spanEndTimeShift: 1m
  tags:
    - key: service.name
      value: service_name
  filterByTraceID: true
  filterBySpanID: true
  mapTagNamesEnabled: true
  mappedTags:
    - key: gl.tenant_id
      value: tenant_id
```

#### Trace-to-Metrics (Exemplars)

Tempo metrics-generator produces span metrics with exemplars that link back to traces:

```yaml
# Tempo metrics_generator → Prometheus remote_write → exemplars
# Grafana panels can show trace links on metric graphs
tracesToMetrics:
  datasourceUid: thanos
  spanStartTimeShift: -5m
  spanEndTimeShift: 5m
  tags:
    - key: service.name
      value: service
  queries:
    - name: Request rate
      query: "sum(rate(traces_spanmetrics_calls_total{$$__tags}[5m]))"
    - name: Error rate
      query: "sum(rate(traces_spanmetrics_calls_total{$$__tags,status_code=\"STATUS_CODE_ERROR\"}[5m]))"
    - name: Duration (p95)
      query: "histogram_quantile(0.95, sum(rate(traces_spanmetrics_duration_seconds_bucket{$$__tags}[5m])) by (le))"
```

#### Logs-to-Traces

Structured logs include `trace_id` and `span_id` fields (already in SEC-005 audit logging):

```python
# Already implemented in greenlang/infrastructure/logging/context.py
# Logs emitted with trace_id field enable Loki → Tempo linking
structlog.configure(
    processors=[
        # ... existing processors ...
        add_trace_context,  # Adds trace_id, span_id to log entries
    ]
)
```

### 4.8 CI/CD Pipeline

**Location**: `.github/workflows/tracing-ci.yml`

**Jobs**:
1. **lint-otel-config**: Validate OTel Collector config YAML syntax
2. **validate-tempo-config**: Validate Tempo configuration
3. **helm-lint**: Lint Tempo and OTel Collector Helm charts, template with all environments
4. **test-sdk**: Run tracing SDK unit tests
5. **test-integration**: Run integration tests with Tempo test container
6. **schema-validate**: Validate dashboard JSON and alert rule YAML

## 5. Integration Points

### 5.1 With OBS-001 (Prometheus)

- Tempo metrics-generator remote-writes span metrics to Prometheus
- OTel Collector exports pipeline metrics to Prometheus
- ServiceMonitors scrape Tempo and OTel Collector endpoints

### 5.2 With OBS-002 (Grafana)

- Tempo datasource configured via Helm values and k8s-sidecar
- Trace-to-logs and trace-to-metrics derived fields
- 4 new dashboards in `02-Observability` folder
- Service Graph panel in platform-overview dashboard

### 5.3 With SEC-005 (Audit Logging)

- Audit events include `trace_id` for correlation
- Structured logs in Loki have trace context for linking

### 5.4 With Existing Services

- `configure_tracing(app)` called in each FastAPI service's main.py
- Backward-compatible with existing `TracingManager` (wraps, doesn't replace)
- Gradual migration path for 17+ agent-specific tracers

## 6. Testing Requirements

### 6.1 Unit Tests

**Location**: `tests/unit/tracing_service/`

| Test File | Tests (est.) | Coverage |
|-----------|-------------|----------|
| `test_config.py` | 15 | TracingConfig defaults, environment overrides |
| `test_provider.py` | 20 | TracerProvider setup, resource attributes, exporter selection |
| `test_instrumentors.py` | 25 | Auto-instrumentation enable/disable, attribute capture |
| `test_context.py` | 18 | W3C propagation, baggage, tenant injection |
| `test_decorators.py` | 22 | @trace_operation, @trace_agent, error recording |
| `test_span_enrichment.py` | 15 | GreenLang semantic conventions, attribute validation |
| `test_sampling.py` | 12 | Sampling decisions, per-service overrides |
| `test_middleware.py` | 20 | Request tracing, header injection, error handling |
| `test_metrics_bridge.py` | 10 | Span-to-metric conversion |
| `test_setup.py` | 8 | One-liner setup, idempotency |

**Total**: ~165 unit tests

### 6.2 Integration Tests

**Location**: `tests/integration/tracing_service/`

| Test File | Tests (est.) | Coverage |
|-----------|-------------|----------|
| `test_end_to_end.py` | 15 | Full trace flow: SDK → Collector → Tempo → Grafana query |
| `test_correlation.py` | 12 | Trace-to-logs, trace-to-metrics linking |
| `test_sampling.py` | 8 | Tail sampling decisions, compliance agent 100% |
| `conftest.py` | N/A | Tempo test container, OTel Collector test config |

**Total**: ~35 integration tests

### 6.3 Load Tests

**Location**: `tests/load/tracing_service/`

| Test File | Tests (est.) | Coverage |
|-----------|-------------|----------|
| `test_throughput.py` | 6 | 10K spans/sec sustained, P99 < 5ms SDK overhead |
| `test_backpressure.py` | 4 | Collector queue full behavior, graceful degradation |

**Total**: ~10 load tests

### 6.4 Minimum Coverage

- Unit tests: 85%+ line coverage on `greenlang/infrastructure/tracing_service/`
- All auto-instrumentors tested with mock servers
- Integration tests validate end-to-end trace flow

## 7. Success Criteria

| Metric | Target | Measurement |
|--------|--------|-------------|
| SDK instrumentation overhead | < 5ms P99 per request | Load test measurement |
| Span ingestion throughput | 10,000 spans/sec sustained | Tempo metrics |
| Trace query latency (by ID) | < 500ms P95 | Grafana Tempo datasource |
| TraceQL search latency | < 5s P95 for 24h window | Grafana Tempo datasource |
| Trace-to-logs correlation | 100% of traces linkable to logs | Manual verification |
| Auto-instrumented libraries | 6 (FastAPI, httpx, psycopg, redis, celery, requests) | Code review |
| Dashboard count | 4 new dashboards | File count |
| Alert rules | 12 alerts | File review |
| Test count | 200+ (165 unit + 35 integration + 10 load) | pytest count |
| Code coverage | 85%+ | pytest-cov report |

## 8. Non-Goals (v1)

- **eBPF-based auto-instrumentation**: Out of scope; use language-level instrumentors
- **Continuous profiling**: Separate OBS component (Pyroscope integration)
- **Real User Monitoring (RUM)**: Frontend tracing is a separate effort
- **Cross-cluster tracing**: Single EKS cluster only for v1
- **Custom TraceQL alerting**: Use Prometheus alerts on span metrics instead

## 9. Dependencies

| Dependency | Version | Purpose |
|------------|---------|---------|
| `opentelemetry-api` | ~=1.24 | Core OTel API |
| `opentelemetry-sdk` | ~=1.24 | TracerProvider, SpanProcessor |
| `opentelemetry-exporter-otlp-proto-grpc` | ~=1.24 | OTLP gRPC exporter |
| `opentelemetry-instrumentation-fastapi` | ~=0.45b0 | FastAPI auto-instrumentation |
| `opentelemetry-instrumentation-httpx` | ~=0.45b0 | httpx auto-instrumentation |
| `opentelemetry-instrumentation-psycopg2` | ~=0.45b0 | PostgreSQL auto-instrumentation |
| `opentelemetry-instrumentation-redis` | ~=0.45b0 | Redis auto-instrumentation |
| `opentelemetry-instrumentation-celery` | ~=0.45b0 | Celery auto-instrumentation |
| `opentelemetry-instrumentation-requests` | ~=0.45b0 | requests auto-instrumentation |
| `opentelemetry-propagator-b3` | ~=1.24 | B3 propagation (Zipkin compat) |

## 10. Rollout Plan

### Phase 1 (Week 1): Infrastructure
- Deploy Tempo Helm chart (dev first, then staging)
- Deploy OTel Collector Helm chart
- Apply Terraform for S3 bucket + IRSA
- Validate trace ingestion with test spans

### Phase 2 (Week 2): SDK & Integration
- Build unified tracing SDK
- Add `configure_tracing(app)` to API service
- Configure Grafana Tempo datasource
- Deploy tracing dashboards and alerts

### Phase 3 (Week 3): Migration & Validation
- Dual-write to Jaeger + Tempo via OTel Collector
- Validate all traces visible in Tempo
- Remove Jaeger datasource from Grafana
- Remove Jaeger deployment

### Phase 4 (Week 4): Production
- Deploy to production environment
- Monitor trace pipeline health for 1 week
- Enable auto-instrumentation for all services
- Complete runbook documentation

---

**Approved By**: Platform Engineering Team
**Review Date**: 2026-02-07
