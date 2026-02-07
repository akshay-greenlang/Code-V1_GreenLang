# Metrics Developer Guide

## Overview

This guide explains how to add metrics to GreenLang services, naming conventions, best practices, and how to use the PushGateway for batch jobs.

---

## Table of Contents

1. [Metric Naming Conventions](#metric-naming-conventions)
2. [Adding Metrics to Your Service](#adding-metrics-to-your-service)
3. [Label Best Practices](#label-best-practices)
4. [ServiceMonitor Creation](#servicemonitor-creation)
5. [PushGateway for Batch Jobs](#pushgateway-for-batch-jobs)
6. [Recording Rules](#recording-rules)
7. [Testing Metrics](#testing-metrics)

---

## Metric Naming Conventions

### Prefix

All GreenLang metrics MUST use the `gl_` prefix to avoid conflicts with system metrics.

```
gl_<component>_<metric_name>_<unit>
```

### Examples

```
# Good
gl_api_requests_total
gl_api_request_duration_seconds
gl_agent_executions_total
gl_emission_calculations_total
gl_cache_hits_total

# Bad
requests_total          # Missing gl_ prefix
api_request_duration    # Missing unit suffix
greenlang_api_requests  # Use gl_ not greenlang_
```

### Metric Types

| Type | Suffix | Use Case |
|------|--------|----------|
| Counter | `_total` | Things that only go up (requests, errors) |
| Gauge | (none) | Values that go up and down (queue size, temperature) |
| Histogram | `_bucket`, `_sum`, `_count` | Duration, size distributions |
| Summary | `_sum`, `_count` | Similar to histogram, for quantiles |

### Unit Suffixes

| Unit | Suffix | Example |
|------|--------|---------|
| Seconds | `_seconds` | `gl_api_request_duration_seconds` |
| Bytes | `_bytes` | `gl_memory_usage_bytes` |
| Ratio | `_ratio` | `gl_cache_hit_ratio` |
| Total | `_total` | `gl_requests_total` |
| Info | `_info` | `gl_build_info` |

---

## Adding Metrics to Your Service

### Python (FastAPI)

#### 1. Install Dependencies

```bash
pip install prometheus-client prometheus-fastapi-instrumentator
```

#### 2. Basic Setup

```python
# app/metrics.py
from prometheus_client import Counter, Histogram, Gauge, CollectorRegistry
from prometheus_fastapi_instrumentator import Instrumentator

# Create a custom registry (optional, for isolation)
REGISTRY = CollectorRegistry()

# Define metrics
gl_requests_total = Counter(
    "gl_api_requests_total",
    "Total API requests",
    ["method", "endpoint", "status_code"],
    registry=REGISTRY,
)

gl_request_duration = Histogram(
    "gl_api_request_duration_seconds",
    "API request duration in seconds",
    ["method", "endpoint"],
    buckets=[0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0],
    registry=REGISTRY,
)

gl_active_requests = Gauge(
    "gl_api_active_requests",
    "Number of active requests",
    registry=REGISTRY,
)

gl_emission_calculations = Counter(
    "gl_emission_calculations_total",
    "Total emission calculations performed",
    ["calculation_type", "status"],
    registry=REGISTRY,
)
```

#### 3. FastAPI Integration

```python
# app/main.py
from fastapi import FastAPI, Request
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
from starlette.responses import Response
from starlette.middleware.base import BaseHTTPMiddleware
import time

from app.metrics import (
    gl_requests_total,
    gl_request_duration,
    gl_active_requests,
    REGISTRY,
)

app = FastAPI()


class MetricsMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        gl_active_requests.inc()
        start_time = time.perf_counter()

        try:
            response = await call_next(request)
            status_code = response.status_code
        except Exception as e:
            status_code = 500
            raise
        finally:
            duration = time.perf_counter() - start_time
            gl_active_requests.dec()

            gl_requests_total.labels(
                method=request.method,
                endpoint=request.url.path,
                status_code=str(status_code),
            ).inc()

            gl_request_duration.labels(
                method=request.method,
                endpoint=request.url.path,
            ).observe(duration)

        return response


app.add_middleware(MetricsMiddleware)


@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint."""
    return Response(
        content=generate_latest(REGISTRY),
        media_type=CONTENT_TYPE_LATEST,
    )
```

#### 4. Recording Business Metrics

```python
# In your business logic
from app.metrics import gl_emission_calculations


def calculate_emissions(data: dict) -> float:
    try:
        result = perform_calculation(data)
        gl_emission_calculations.labels(
            calculation_type=data["type"],
            status="success",
        ).inc()
        return result
    except Exception as e:
        gl_emission_calculations.labels(
            calculation_type=data["type"],
            status="error",
        ).inc()
        raise
```

---

## Label Best Practices

### Do

```python
# Limited, known values
labels=["method", "status_code"]  # GET, POST, 200, 404, 500

# Bounded cardinality
labels=["service", "environment"]  # Known services and environments

# Useful for aggregation
labels=["agent_type", "region"]
```

### Don't

```python
# AVOID: High cardinality labels
labels=["request_id"]        # Unique per request = millions of series
labels=["user_id"]           # Many users = high cardinality
labels=["timestamp"]         # Infinite cardinality
labels=["ip_address"]        # Many IPs = high cardinality
labels=["trace_id"]          # Unique per trace
```

### Cardinality Guidelines

| Label Type | Cardinality | Acceptable |
|------------|-------------|------------|
| HTTP Method | 5-10 | Yes |
| Status Code | 10-20 | Yes |
| Service Name | 10-50 | Yes |
| Region | 5-20 | Yes |
| User ID | 10K+ | NO |
| Request ID | Unlimited | NO |
| IP Address | 1K+ | NO |

### Calculating Cardinality

```
Total Series = Metric * (Label1 values * Label2 values * ...)

Example:
- gl_api_requests_total{method, endpoint, status_code}
- 5 methods * 20 endpoints * 10 status codes = 1000 series
```

---

## ServiceMonitor Creation

### Basic ServiceMonitor

```yaml
# deployment/kubernetes/monitoring/servicemonitors/my-service.yaml
apiVersion: monitoring.coreos.com/v1
kind: ServiceMonitor
metadata:
  name: my-service
  namespace: monitoring
  labels:
    release: prometheus  # Required for Prometheus Operator discovery
spec:
  selector:
    matchLabels:
      app: my-service  # Must match your Service labels
  namespaceSelector:
    matchNames:
      - greenlang
  endpoints:
    - port: http  # Port name from your Service
      path: /metrics
      interval: 15s
      scrapeTimeout: 10s
```

### Service Configuration

```yaml
# Your service must have matching labels
apiVersion: v1
kind: Service
metadata:
  name: my-service
  namespace: greenlang
  labels:
    app: my-service  # This must match ServiceMonitor selector
spec:
  ports:
    - name: http  # This must match ServiceMonitor endpoint port
      port: 8080
      targetPort: 8080
    - name: metrics  # Or a separate metrics port
      port: 9090
      targetPort: 9090
  selector:
    app: my-service
```

### With Metric Relabeling

```yaml
apiVersion: monitoring.coreos.com/v1
kind: ServiceMonitor
metadata:
  name: my-service
  namespace: monitoring
spec:
  selector:
    matchLabels:
      app: my-service
  endpoints:
    - port: http
      path: /metrics
      interval: 15s
      # Add custom labels
      relabelings:
        - sourceLabels: [__meta_kubernetes_pod_label_app]
          targetLabel: service
        - sourceLabels: [__meta_kubernetes_namespace]
          targetLabel: namespace
      # Drop unwanted metrics
      metricRelabelings:
        - sourceLabels: [__name__]
          regex: 'go_.*'
          action: drop
        - sourceLabels: [__name__]
          regex: 'gl_.*'
          action: keep
```

---

## PushGateway for Batch Jobs

### When to Use PushGateway

- Batch jobs that run and exit
- CronJobs
- Short-lived processes
- Jobs that cannot be scraped directly

### Python SDK Usage

```python
from greenlang.monitoring.pushgateway import BatchJobMetrics


def my_batch_job():
    """Example batch job with metrics."""
    # Initialize metrics
    metrics = BatchJobMetrics(
        job_name="emission-factor-update",
        pushgateway_url="http://pushgateway.monitoring.svc:9091",
        grouping_key={"instance": "worker-1"},
    )

    # Track job duration
    with metrics.track_duration():
        # Your job logic here
        records = fetch_emission_factors()

        for record in records:
            try:
                process_record(record)
                metrics.record_records(processed=1, record_type="emission_factor")
            except Exception as e:
                metrics.record_error(error_type=type(e).__name__)

    # Push metrics to PushGateway
    metrics.push()


# With context manager (auto-push on exit)
def my_batch_job_v2():
    """Example with context manager."""
    with BatchJobMetrics(
        job_name="data-sync",
        grouping_key={"source": "sap"},
    ) as metrics:
        with metrics.track_duration():
            sync_data()
            metrics.record_records(processed=1000, record_type="sync")

    # Metrics are automatically pushed when exiting the context
```

### Kubernetes CronJob Example

```yaml
apiVersion: batch/v1
kind: CronJob
metadata:
  name: emission-factor-update
  namespace: greenlang
spec:
  schedule: "0 */6 * * *"  # Every 6 hours
  jobTemplate:
    spec:
      template:
        spec:
          containers:
            - name: job
              image: greenlang/ef-updater:latest
              env:
                - name: PUSHGATEWAY_URL
                  value: "http://pushgateway.monitoring.svc:9091"
                - name: JOB_NAME
                  value: "emission-factor-update"
              command:
                - python
                - -m
                - greenlang.jobs.update_emission_factors
          restartPolicy: OnFailure
```

### Cleaning Up Old Metrics

```python
# Delete metrics after job completion (optional)
metrics.delete()

# Or set up automatic cleanup via CronJob
# See: docs/runbooks/batch-job-metrics-stale.md
```

---

## Recording Rules

Recording rules pre-compute expensive queries for faster dashboard loading.

### Creating Recording Rules

```yaml
# deployment/kubernetes/monitoring/prometheus-rules/recording-rules.yaml
apiVersion: monitoring.coreos.com/v1
kind: PrometheusRule
metadata:
  name: greenlang-recording-rules
  namespace: monitoring
  labels:
    release: prometheus
spec:
  groups:
    - name: gl_api_recording_rules
      interval: 1m
      rules:
        # Pre-aggregate request rate by service
        - record: gl:api_request_rate:5m
          expr: sum(rate(gl_api_requests_total[5m])) by (service)

        # Pre-compute latency percentiles
        - record: gl:api_latency_p99:5m
          expr: |
            histogram_quantile(0.99,
              sum(rate(gl_api_request_duration_seconds_bucket[5m])) by (le, service)
            )

        - record: gl:api_latency_p95:5m
          expr: |
            histogram_quantile(0.95,
              sum(rate(gl_api_request_duration_seconds_bucket[5m])) by (le, service)
            )

        # Error rate
        - record: gl:api_error_rate:5m
          expr: |
            sum(rate(gl_api_requests_total{status_code=~"5.."}[5m])) by (service)
            /
            sum(rate(gl_api_requests_total[5m])) by (service)

    - name: gl_agent_recording_rules
      interval: 1m
      rules:
        - record: gl:agent_execution_rate:5m
          expr: sum(rate(gl_agent_executions_total[5m])) by (agent_type)

        - record: gl:agent_success_rate:5m
          expr: |
            sum(rate(gl_agent_executions_total{status="success"}[5m])) by (agent_type)
            /
            sum(rate(gl_agent_executions_total[5m])) by (agent_type)
```

### Apply Recording Rules

```bash
kubectl apply -f deployment/kubernetes/monitoring/prometheus-rules/recording-rules.yaml
```

### Use in Dashboards

```promql
# Instead of computing on-the-fly:
histogram_quantile(0.99, sum(rate(gl_api_request_duration_seconds_bucket[5m])) by (le, service))

# Use the pre-computed recording rule:
gl:api_latency_p99:5m
```

---

## Testing Metrics

### Local Testing

```python
# tests/unit/test_metrics.py
import pytest
from prometheus_client import REGISTRY, Counter, generate_latest


def test_metric_increments():
    """Test that counter increments correctly."""
    from app.metrics import gl_requests_total

    initial = gl_requests_total._value._value

    gl_requests_total.labels(
        method="GET",
        endpoint="/api/v1/test",
        status_code="200",
    ).inc()

    assert gl_requests_total._value._value == initial + 1


def test_metrics_endpoint():
    """Test metrics endpoint returns valid Prometheus format."""
    from app.main import app
    from fastapi.testclient import TestClient

    client = TestClient(app)
    response = client.get("/metrics")

    assert response.status_code == 200
    assert "gl_api_requests_total" in response.text
    assert "# HELP" in response.text
    assert "# TYPE" in response.text
```

### Verify Metrics in Prometheus

```bash
# Port-forward Prometheus
kubectl port-forward -n monitoring svc/prometheus-server 9090:9090

# Check if your metrics are being scraped
curl 'http://localhost:9090/api/v1/query?query=gl_api_requests_total'

# Check target status
curl 'http://localhost:9090/api/v1/targets' | jq '.data.activeTargets[] | select(.labels.job=="my-service")'
```

### Validate PromQL

```bash
# Use promtool to validate expressions
echo 'sum(rate(gl_api_requests_total[5m])) by (service)' | promtool query instant http://localhost:9090

# Validate recording rules
promtool check rules recording-rules.yaml
```

---

## Checklist for New Metrics

Before adding new metrics, review this checklist:

- [ ] Metric uses `gl_` prefix
- [ ] Metric name follows naming conventions
- [ ] Unit suffix is appropriate (`_total`, `_seconds`, `_bytes`)
- [ ] Labels are bounded (no high-cardinality values)
- [ ] Total cardinality is estimated and acceptable (<10K series)
- [ ] Metric is documented in code
- [ ] ServiceMonitor is configured
- [ ] Tests are written
- [ ] Recording rules added for expensive queries
- [ ] Dashboard updated (if applicable)

---

## Related Documentation

- [Prometheus Stack Architecture](../architecture/prometheus-stack.md)
- [Prometheus Operations Guide](../operations/prometheus-operations.md)
- [PushGateway SDK Reference](../../greenlang/monitoring/pushgateway.py)
