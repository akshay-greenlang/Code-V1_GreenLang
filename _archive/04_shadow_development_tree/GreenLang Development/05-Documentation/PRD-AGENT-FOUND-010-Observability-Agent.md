# PRD: AGENT-FOUND-010 - Observability & Telemetry Agent

## 1. Overview

| Field | Value |
|-------|-------|
| **PRD ID** | AGENT-FOUND-010 |
| **Agent ID** | GL-FOUND-X-010 |
| **Component** | Observability & Telemetry Agent (Metrics, Tracing, Logging, Alerting, Dashboards, SLOs) |
| **Category** | Foundations Agent |
| **Priority** | P0 - Critical (observability backbone for all 47+ agents) |
| **Status** | Layer 1 Complete (~1,316 lines), Integration Gap-Fill Required |
| **Author** | GreenLang Platform Team |
| **Date** | February 2026 |

## 2. Problem Statement

GreenLang Climate OS runs 47+ agents producing emission calculations, compliance
assessments, and regulatory reports in a distributed microservices environment.
Without a production-grade observability agent providing a unified SDK:

- **No unified metrics collection**: Each agent implements ad-hoc Prometheus instrumentation
- **No distributed tracing integration**: Request flows across agents are invisible
- **No structured log aggregation**: Logs lack correlation IDs and trace context
- **No programmatic alert management**: Alert rules must be manually configured
- **No health check orchestration**: Service health monitoring is fragmented
- **No dashboard data provisioning**: Grafana dashboards lack programmatic data sources
- **No SLO/SLI tracking integration**: Error budgets and burn rates are not agent-accessible
- **No audit trail**: Observability operations are not recorded for compliance
- **No zero-hallucination verification**: Metric correctness is not provably traceable

## 3. Existing Implementation

### 3.1 Layer 1: Foundation Agent
**File**: `greenlang/agents/foundation/observability_agent.py` (~1,316 lines)
- `ObservabilityAgent` (BaseAgent subclass, AGENT_ID: GL-FOUND-X-009 -- note: ID conflict, should be GL-FOUND-X-010)
- 5 enums: MetricType(4: counter/gauge/histogram/summary), AlertSeverity(4: info/warning/error/critical), AlertStatus(3: firing/resolved/pending), HealthStatus(3: healthy/degraded/unhealthy), TraceStatus(3: unset/ok/error)
- 12 Pydantic models: MetricLabel, MetricDefinition, MetricValue, TraceContext, SpanDefinition, LogEntry, AlertRule, Alert, HealthCheck, DashboardPanel, ObservabilityInput, ObservabilityOutput
- 2 internal dataclasses: MetricSeries, ActiveSpan
- 5 pre-defined platform metrics: agent_execution_duration_seconds, agent_execution_total, agent_errors_total, lineage_completeness_ratio, zero_hallucination_compliance
- Operations: record_metric, increment_counter, set_gauge, observe_histogram, start_span, end_span, log, add_alert_rule, check_alerts, liveness_probe, readiness_probe, get_dashboard_data, export_metrics
- Convenience methods: record_agent_execution, create_trace, log_structured, get_metrics_summary, get_active_alerts_summary
- In-memory storage (no database persistence)
- SHA-256 provenance hashing

### 3.2 OBS Infrastructure (Fully Built)
The observability agent integrates with 5 complete OBS infrastructure components:
- **OBS-001**: Prometheus Metrics Collection (HA, Thanos, Alertmanager)
- **OBS-002**: Grafana Dashboards Platform (11.4 HA, 7-folder hierarchy)
- **OBS-003**: OpenTelemetry Tracing (Tempo 2.7, OTel Collector)
- **OBS-004**: Alerting & Notification Platform (6 channels, lifecycle state machine)
- **OBS-005**: SLO/SLI Definitions & Error Budget (burn rate, error budget engine)

### 3.3 Layer 1 Tests
None found.

## 4. Identified Gaps

### Gap 1: No Integration Module
No `greenlang/observability_agent/` package providing a clean SDK for other agents/services.

### Gap 2: No Prometheus Self-Monitoring Metrics
No `greenlang/observability_agent/metrics.py` following the standard 12-metric pattern for self-monitoring.

### Gap 3: No Service Setup Facade
No `configure_observability_agent(app)` / `get_observability_agent(app)` pattern.

### Gap 4: Foundation Agent Doesn't Delegate
Layer 1 has in-memory storage; doesn't delegate to persistent integration module.

### Gap 5: No REST API Router
No `greenlang/observability_agent/api/router.py` with FastAPI endpoints.

### Gap 6: No K8s Deployment Manifests
No `deployment/kubernetes/observability-agent-service/` manifests.

### Gap 7: No Database Migration
No `V030__observability_agent_service.sql` for persistent metric/trace/log storage.

### Gap 8: No Monitoring
No Grafana dashboard or alert rules for self-monitoring.

### Gap 9: No CI/CD Pipeline
No `.github/workflows/observability-agent-ci.yml`.

### Gap 10: No Operational Runbooks
No `docs/runbooks/` for observability agent operations.

### Gap 11: No SLO/SLI Integration
No integration with OBS-005 SLO/SLI engine for programmatic SLO tracking.

## 5. Architecture (Final State)

### 5.1 Integration Module
```
greenlang/observability_agent/
  __init__.py                  # Public API exports
  config.py                    # ObservabilityAgentConfig with GL_OBSERVABILITY_AGENT_ env prefix
  models.py                    # Pydantic v2 models (re-export + enhance from foundation agent)
  metrics_collector.py         # MetricsCollector: unified Prometheus metric recording
  trace_manager.py             # TraceManager: OTel-compatible distributed tracing
  log_aggregator.py            # LogAggregator: structured log collection with Loki integration
  alert_evaluator.py           # AlertEvaluator: rule evaluation, threshold checking, notification dispatch
  health_checker.py            # HealthChecker: orchestrated liveness/readiness/startup probes
  dashboard_provider.py        # DashboardProvider: Grafana data provisioning and query proxy
  slo_tracker.py               # SLOTracker: SLO/SLI tracking, burn rate calculation, error budget monitoring
  provenance.py                # ProvenanceTracker: SHA-256 hash chain for observability audit
  metrics.py                   # 12 Prometheus self-monitoring metrics
  setup.py                     # ObservabilityAgentService facade, configure/get
  api/
    __init__.py
    router.py                  # FastAPI router (20 endpoints)
```

### 5.2 Database Schema (V030)
```sql
CREATE SCHEMA observability_agent_service;
-- metric_definitions (metric registry with type, labels, description)
-- metric_recordings (hypertable - metric observation records)
-- trace_spans (hypertable - distributed trace span records)
-- log_entries (hypertable - structured log entries with correlation)
-- alert_rules (alert rule definitions with conditions)
-- alert_instances (active and historical alert instances)
-- health_check_results (health check execution records)
-- dashboard_configs (dashboard configuration and provisioning)
-- slo_definitions (SLO/SLI definitions with targets)
-- obs_audit_log (hypertable - observability operation audit trail)
```

### 5.3 Prometheus Self-Monitoring Metrics (12)
| # | Metric | Type | Description |
|---|--------|------|-------------|
| 1 | `gl_obs_metrics_recorded_total` | Counter | Total metric observations recorded |
| 2 | `gl_obs_operation_duration_seconds` | Histogram | Observability operation latency |
| 3 | `gl_obs_spans_created_total` | Counter | Total trace spans created |
| 4 | `gl_obs_spans_active` | Gauge | Currently active trace spans |
| 5 | `gl_obs_logs_ingested_total` | Counter | Total log entries ingested |
| 6 | `gl_obs_alerts_evaluated_total` | Counter | Total alert rule evaluations |
| 7 | `gl_obs_alerts_firing` | Gauge | Currently firing alerts |
| 8 | `gl_obs_health_checks_total` | Counter | Total health checks executed |
| 9 | `gl_obs_health_status` | Gauge | Current health status (1=healthy, 0.5=degraded, 0=unhealthy) |
| 10 | `gl_obs_slo_compliance_ratio` | Gauge | Current SLO compliance ratio by service |
| 11 | `gl_obs_error_budget_remaining` | Gauge | Remaining error budget by SLO |
| 12 | `gl_obs_dashboard_queries_total` | Counter | Total dashboard data queries |

### 5.4 API Endpoints (20)
| Method | Path | Description |
|--------|------|-------------|
| POST | `/v1/metrics/record` | Record a metric observation |
| GET | `/v1/metrics` | List metric definitions |
| GET | `/v1/metrics/{metric_name}` | Get metric details and current value |
| POST | `/v1/metrics/export` | Export metrics in Prometheus format |
| POST | `/v1/traces/spans` | Create/start a trace span |
| PUT | `/v1/traces/spans/{span_id}` | End/update a trace span |
| GET | `/v1/traces/{trace_id}` | Get full trace with all spans |
| POST | `/v1/logs` | Ingest structured log entries |
| GET | `/v1/logs` | Query log entries (with filters) |
| POST | `/v1/alerts/rules` | Create/update alert rule |
| GET | `/v1/alerts/rules` | List alert rules |
| POST | `/v1/alerts/evaluate` | Evaluate all alert rules |
| GET | `/v1/alerts/active` | Get currently firing alerts |
| POST | `/v1/health/check` | Run health check probes |
| GET | `/v1/health/status` | Get aggregated health status |
| GET | `/v1/dashboards/{dashboard_id}` | Get dashboard data |
| POST | `/v1/slos` | Create/update SLO definition |
| GET | `/v1/slos` | List SLO definitions with compliance |
| GET | `/v1/slos/{slo_id}/burn-rate` | Get SLO burn rate analysis |
| GET | `/health` | Service health check |

### 5.5 Key Design Principles
1. **Unified SDK**: Single entry point for all observability operations across 47+ agents
2. **OBS infrastructure bridge**: Programmatic access to Prometheus, Grafana, Tempo, Loki, Alertmanager
3. **Zero-hallucination compliance**: All metrics deterministically calculated, SHA-256 provenance
4. **Multi-signal correlation**: Metrics, traces, and logs linked via correlation IDs and trace context
5. **SLO-driven operations**: Error budget tracking with burn rate alerts for each service
6. **Health orchestration**: Unified liveness/readiness/startup probe management
7. **Tenant isolation**: All observations scoped to tenant_id
8. **Complete audit trail**: Every observability operation logged with SHA-256 provenance chain
9. **Graceful degradation**: Works without Prometheus/OTel client libraries installed
10. **Platform metrics standard**: Pre-defined metric templates for consistent agent instrumentation

## 6. Completion Plan

### Phase 1: Core Integration (Backend Developer)
1. Create `greenlang/observability_agent/__init__.py` - Public API exports
2. Create `greenlang/observability_agent/config.py` - ObservabilityAgentConfig with GL_OBSERVABILITY_AGENT_ env prefix
3. Create `greenlang/observability_agent/models.py` - Pydantic v2 models
4. Create `greenlang/observability_agent/metrics_collector.py` - MetricsCollector with Prometheus integration
5. Create `greenlang/observability_agent/trace_manager.py` - TraceManager with OTel integration
6. Create `greenlang/observability_agent/log_aggregator.py` - LogAggregator with structured logging
7. Create `greenlang/observability_agent/alert_evaluator.py` - AlertEvaluator with rule management
8. Create `greenlang/observability_agent/health_checker.py` - HealthChecker with probe orchestration
9. Create `greenlang/observability_agent/dashboard_provider.py` - DashboardProvider with Grafana data
10. Create `greenlang/observability_agent/slo_tracker.py` - SLOTracker with burn rate calculation
11. Create `greenlang/observability_agent/provenance.py` - ProvenanceTracker
12. Create `greenlang/observability_agent/metrics.py` - 12 Prometheus self-monitoring metrics
13. Create `greenlang/observability_agent/api/router.py` - FastAPI router with 20 endpoints
14. Create `greenlang/observability_agent/setup.py` - ObservabilityAgentService facade

### Phase 2: Infrastructure (DevOps Engineer)
1. Create `deployment/database/migrations/sql/V030__observability_agent_service.sql`
2. Create K8s manifests in `deployment/kubernetes/observability-agent-service/`
3. Create monitoring dashboards and alerts
4. Create CI/CD pipeline
5. Create operational runbooks

### Phase 3: Tests (Test Engineer)
1-16. Create unit, integration, and load tests (600+ tests target)

## 7. Success Criteria
- Integration module provides clean SDK for all observability operations
- All 12 self-monitoring Prometheus metrics instrumented
- Standard GreenLang deployment pattern (K8s, monitoring, CI/CD)
- V030 database migration for persistent observability storage
- 20 REST API endpoints operational
- 600+ tests passing
- Unified metrics/tracing/logging/alerting SDK operational
- SLO/SLI tracking with burn rate analysis
- Health check orchestration across all agents
- Dashboard data provisioning for Grafana
- Complete audit trail for every observability operation
- Integration with all 5 OBS infrastructure components

## 8. Integration Points

### 8.1 Upstream Dependencies
- **OBS-001 Prometheus**: Metric recording and export target
- **OBS-002 Grafana**: Dashboard data provisioning
- **OBS-003 OpenTelemetry**: Distributed tracing integration
- **OBS-004 Alerting**: Alert rule evaluation and notification dispatch
- **OBS-005 SLO/SLI**: Error budget engine and burn rate calculation
- **AGENT-FOUND-006 Access Guard**: Authorization for observability operations
- **AGENT-FOUND-007 Agent Registry**: Discover agents for health monitoring

### 8.2 Downstream Consumers
- **All agents (001-009+)**: Use SDK for metrics, tracing, logging
- **AGENT-FOUND-001 Orchestrator**: DAG execution observability
- **AGENT-FOUND-009 QA Test Harness**: Performance benchmarking integration
- **CI/CD pipelines**: Health check gates in deployment
- **Admin dashboard**: Observability overview visualization
- **Incident response**: Alert management and correlation

### 8.3 Infrastructure Integration
- **PostgreSQL**: Persistent metric/trace/log storage (V030 migration)
- **Redis**: Metric aggregation caching, alert state caching
- **Prometheus**: 12 self-monitoring metrics + platform metric export
- **Grafana**: Observability agent dashboard + data provisioning
- **Alertmanager**: 15 alert rules for observability agent self-monitoring
- **Tempo**: Trace span ingestion
- **Loki**: Structured log forwarding
- **K8s**: Standard deployment with HPA
