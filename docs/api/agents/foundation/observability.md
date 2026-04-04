# AGENT-FOUND-010: Observability & Telemetry Agent API Reference

**Agent ID:** AGENT-FOUND-010
**Service:** Observability & Telemetry Agent
**Status:** Production Ready
**Base Path:** `/api/v1/observability-agent`
**Tag:** `observability-agent`
**Source:** `greenlang/agents/foundation/observability_agent/api/router.py`

The Observability Agent provides a unified telemetry collection and analysis
layer including metric recording, distributed tracing, structured log
aggregation, alert evaluation, health checking, dashboard provisioning, and SLO
tracking with burn-rate analysis.

---

## Endpoint Summary

| # | Method | Path | Summary | Status Codes |
|---|--------|------|---------|--------------|
| 1 | POST | `/v1/metrics/record` | Record a metric observation | 200, 400, 500, 503 |
| 2 | GET | `/v1/metrics` | List metric definitions | 200, 500, 503 |
| 3 | GET | `/v1/metrics/{metric_name}` | Get metric details | 200, 404, 500, 503 |
| 4 | POST | `/v1/metrics/export` | Export metrics (Prometheus) | 200, 400, 500, 503 |
| 5 | POST | `/v1/traces/spans` | Create/start a trace span | 200, 400, 500, 503 |
| 6 | PUT | `/v1/traces/spans/{span_id}` | End/update a trace span | 200, 400, 404, 500, 503 |
| 7 | GET | `/v1/traces/{trace_id}` | Get full trace | 200, 404, 500, 503 |
| 8 | POST | `/v1/logs` | Ingest structured log entry | 200, 400, 500, 503 |
| 9 | GET | `/v1/logs` | Query log entries | 200, 500, 503 |
| 10 | POST | `/v1/alerts/rules` | Create/update alert rule | 200, 400, 500, 503 |
| 11 | GET | `/v1/alerts/rules` | List alert rules | 200, 500, 503 |
| 12 | POST | `/v1/alerts/evaluate` | Evaluate all alert rules | 200, 500, 503 |
| 13 | GET | `/v1/alerts/active` | Get active (firing) alerts | 200, 500, 503 |
| 14 | POST | `/v1/health/check` | Run health check probes | 200, 400, 500, 503 |
| 15 | GET | `/v1/health/status` | Get aggregated health status | 200, 500, 503 |
| 16 | GET | `/v1/dashboards/{dashboard_id}` | Get dashboard data | 200, 404, 500, 503 |
| 17 | POST | `/v1/slos` | Create/update SLO definition | 200, 400, 500, 503 |
| 18 | GET | `/v1/slos` | List SLOs with compliance | 200, 500, 503 |
| 19 | GET | `/v1/slos/{slo_id}/burn-rate` | Get SLO burn rate analysis | 200, 404, 500, 503 |
| 20 | GET | `/health` | Service health check | 200 |

---

## Detailed Endpoints

### Metrics

#### POST /v1/metrics/record -- Record Metric

Record a metric observation with labels and tenant context.

**Request Body:**

```json
{
  "metric_name": "ghg_calculation_duration_ms",
  "value": 245.3,
  "labels": {
    "agent": "gl-mrv-scope1-stationary",
    "scope": "scope1",
    "fuel_type": "natural_gas"
  },
  "tenant_id": "tenant_001",
  "metric_type": "histogram",
  "description": "GHG calculation duration in milliseconds",
  "unit": "ms"
}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `metric_name` | string | Yes | Name of the metric |
| `value` | float | Yes | Observation value |
| `labels` | object | No | Metric labels (key-value pairs) |
| `tenant_id` | string | No | Tenant identifier (default: `default`) |
| `metric_type` | string | No | Type: `counter`, `gauge`, `histogram`, `summary` (default: `gauge`) |
| `description` | string | No | Metric description |
| `unit` | string | No | Metric unit |

**Response (200):**

```json
{
  "metric_name": "ghg_calculation_duration_ms",
  "value": 245.3,
  "recorded_at": "2026-04-04T10:00:00Z",
  "provenance_hash": "sha256:..."
}
```

---

#### GET /v1/metrics -- List Metrics

**Query Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `tenant_id` | string | -- | Filter by tenant |
| `metric_type` | string | -- | Filter by type |
| `limit` | integer | 100 | Max results (1-1000) |
| `offset` | integer | 0 | Result offset |

**Response (200):**

```json
{
  "metrics": [
    {
      "metric_name": "ghg_calculation_duration_ms",
      "metric_type": "histogram",
      "description": "GHG calculation duration in milliseconds",
      "unit": "ms",
      "label_count": 3
    }
  ],
  "total": 1,
  "limit": 100,
  "offset": 0
}
```

---

### Traces

#### POST /v1/traces/spans -- Create Span

Create and start a new trace span for distributed tracing.

**Request Body:**

```json
{
  "operation_name": "calculate_scope1_emissions",
  "trace_id": null,
  "parent_span_id": null,
  "service_name": "mrv-scope1-agent",
  "attributes": {
    "fuel_type": "natural_gas",
    "input_records": 150
  }
}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `operation_name` | string | Yes | Name of the operation |
| `trace_id` | string | No | Trace ID (auto-generated if null) |
| `parent_span_id` | string | No | Parent span for nested spans |
| `service_name` | string | No | Service name (default: `unknown`) |
| `attributes` | object | No | Span attributes |

**Response (200):**

```json
{
  "span_id": "span_abc123",
  "trace_id": "trace_xyz789",
  "operation_name": "calculate_scope1_emissions",
  "service_name": "mrv-scope1-agent",
  "status": "in_progress",
  "started_at": "2026-04-04T10:00:00Z"
}
```

---

#### PUT /v1/traces/spans/{span_id} -- End Span

**Request Body:**

```json
{
  "status": "ok",
  "attributes": {
    "output_records": 150,
    "total_emissions_tCO2e": 3.0
  }
}
```

---

#### GET /v1/traces/{trace_id} -- Get Full Trace

Returns the complete trace tree with all spans, timing, and attributes.

**Response (200):**

```json
{
  "trace_id": "trace_xyz789",
  "root_span": {
    "span_id": "span_abc123",
    "operation_name": "calculate_scope1_emissions",
    "duration_ms": 245.3,
    "status": "ok",
    "children": [ ... ]
  },
  "span_count": 5,
  "duration_ms": 245.3
}
```

---

### Logs

#### POST /v1/logs -- Ingest Log Entry

**Request Body:**

```json
{
  "level": "info",
  "message": "Scope 1 calculation completed for 150 records",
  "agent_id": "gl-mrv-scope1-stationary",
  "tenant_id": "tenant_001",
  "trace_id": "trace_xyz789",
  "span_id": "span_abc123",
  "correlation_id": "run_abc123",
  "attributes": {
    "records_processed": 150,
    "duration_ms": 245.3
  }
}
```

---

#### GET /v1/logs -- Query Logs

**Query Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `level` | string | Filter by level: `debug`, `info`, `warning`, `error`, `critical` |
| `agent_id` | string | Filter by agent ID |
| `tenant_id` | string | Filter by tenant |
| `trace_id` | string | Filter by trace ID |
| `correlation_id` | string | Filter by correlation ID |
| `limit` | integer | Max results (default: 100, max: 1000) |
| `offset` | integer | Result offset |

---

### Alerts

#### POST /v1/alerts/rules -- Create Alert Rule

**Request Body:**

```json
{
  "name": "high_calculation_latency",
  "metric_name": "ghg_calculation_duration_ms",
  "condition": "gt",
  "threshold": 500.0,
  "duration_seconds": 300,
  "severity": "warning",
  "labels": {
    "team": "platform"
  },
  "annotations": {
    "summary": "GHG calculation latency exceeds 500ms",
    "description": "Agent {{ $labels.agent }} has p95 latency above 500ms for 5 minutes",
    "runbook": "https://runbooks.greenlang.io/high-latency"
  }
}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `name` | string | Yes | Alert rule name |
| `metric_name` | string | Yes | Metric to evaluate |
| `condition` | string | Yes | Condition: `gt`, `lt`, `gte`, `lte`, `eq`, `ne` |
| `threshold` | float | Yes | Threshold value |
| `duration_seconds` | integer | No | Duration before firing (default: 60) |
| `severity` | string | No | Severity: `info`, `warning`, `error`, `critical` (default: `warning`) |
| `labels` | object | No | Alert labels |
| `annotations` | object | No | Annotations (summary, description, runbook) |

---

#### GET /v1/alerts/active -- Get Active Alerts

**Query Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `severity` | string | Filter by severity |

**Response (200):**

```json
{
  "alerts": [
    {
      "name": "high_calculation_latency",
      "metric_name": "ghg_calculation_duration_ms",
      "severity": "warning",
      "firing_since": "2026-04-04T09:55:00Z",
      "current_value": 612.5,
      "threshold": 500.0
    }
  ],
  "count": 1
}
```

---

### SLOs

#### POST /v1/slos -- Create SLO

**Request Body:**

```json
{
  "name": "GHG Calculation Availability",
  "description": "99.9% availability for GHG calculation service",
  "service_name": "mrv-scope1-agent",
  "slo_type": "availability",
  "target": 0.999,
  "window_days": 30,
  "burn_rate_thresholds": {
    "fast": 14.4,
    "medium": 6.0,
    "slow": 1.0
  }
}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `name` | string | Yes | SLO name |
| `description` | string | No | SLO description |
| `service_name` | string | Yes | Target service |
| `slo_type` | string | Yes | Type: `availability`, `latency`, `throughput`, `error_rate` |
| `target` | float | Yes | Target ratio, 0.0-1.0 (e.g., 0.999 for 99.9%) |
| `window_days` | integer | No | Rolling window in days (default: 30, max: 365) |
| `burn_rate_thresholds` | object | No | Burn rate thresholds by speed tier |

---

#### GET /v1/slos/{slo_id}/burn-rate -- SLO Burn Rate Analysis

Returns burn rate analysis including error budget consumption rate.

**Response (200):**

```json
{
  "slo_id": "slo_abc123",
  "name": "GHG Calculation Availability",
  "target": 0.999,
  "current_ratio": 0.9995,
  "error_budget_remaining_pct": 50.0,
  "burn_rates": {
    "fast_1h": 0.5,
    "medium_6h": 0.3,
    "slow_24h": 0.1
  },
  "alerting": false,
  "window_days": 30,
  "evaluated_at": "2026-04-04T10:00:00Z"
}
```

---

### Health

#### GET /health -- Service Health Check

Lightweight endpoint for load balancer and orchestrator probes.

**Response (200):**

```json
{
  "status": "healthy",
  "service": "observability-agent"
}
```
