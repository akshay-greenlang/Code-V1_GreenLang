# CBAM Importer Copilot - Monitoring & Observability Guide

**Version:** 1.0.0
**Team:** Team A3 - GL-CBAM Monitoring & Observability
**Status:** Production Ready

---

## Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Health Checks](#health-checks)
4. [Logging](#logging)
5. [Metrics](#metrics)
6. [Dashboards](#dashboards)
7. [Alerting](#alerting)
8. [Quick Start](#quick-start)
9. [Production Deployment](#production-deployment)
10. [Troubleshooting](#troubleshooting)
11. [SLA & Performance Targets](#sla--performance-targets)

---

## Overview

The CBAM Importer Copilot monitoring system provides production-grade observability through:

- **Health Checks**: Liveness, readiness, and basic health endpoints
- **Structured Logging**: JSON logs with correlation IDs for distributed tracing
- **Prometheus Metrics**: Comprehensive metrics for performance, errors, and business KPIs
- **Grafana Dashboards**: Real-time visualization of system health and performance
- **Alerting**: Critical alerts for errors, latency, and availability issues

### Key Features

- Zero-configuration defaults for development
- Production-ready with Docker Compose deployment
- Compatible with Kubernetes health probes
- Integration with Sentry for error tracking (optional)
- Log aggregation ready (ELK, Splunk, CloudWatch compatible)

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                   CBAM Importer Copilot                     │
│                                                             │
│  ┌─────────────┐  ┌──────────────┐  ┌──────────────┐     │
│  │   Health    │  │   Logging    │  │   Metrics    │     │
│  │   Checks    │  │  (Structured)│  │ (Prometheus) │     │
│  └──────┬──────┘  └──────┬───────┘  └──────┬───────┘     │
│         │                │                  │              │
└─────────┼────────────────┼──────────────────┼──────────────┘
          │                │                  │
          ▼                ▼                  ▼
   ┌────────────┐   ┌────────────┐   ┌────────────┐
   │ Load       │   │ Log        │   │ Prometheus │
   │ Balancer   │   │ Aggregator │   │ (Scraper)  │
   │ / K8s      │   │ (ELK/etc)  │   └──────┬─────┘
   └────────────┘   └────────────┘          │
                                             ▼
                                      ┌────────────┐
                                      │  Grafana   │
                                      │ Dashboard  │
                                      └────────────┘
                                             │
                                             ▼
                                      ┌────────────┐
                                      │ Alert      │
                                      │ Manager    │
                                      └────────────┘
```

---

## Health Checks

### Endpoints

The application provides three levels of health checks:

#### 1. Basic Health (`/health`)

Returns basic service information. Always returns 200 if service is running.

```bash
curl http://localhost:8000/health
```

**Response:**
```json
{
  "status": "healthy",
  "timestamp": "2025-11-08T10:30:00Z",
  "service": "cbam-importer-copilot",
  "version": "1.0.0",
  "uptime_seconds": 3600,
  "uptime_human": "1h 0m 0s"
}
```

**Use Cases:**
- Uptime monitoring
- Basic availability checks
- Load balancer health checks (if all you need is "is it up?")

#### 2. Readiness Check (`/health/ready`)

Checks if service is ready to accept requests (dependencies available).

```bash
curl http://localhost:8000/health/ready
```

**Response (Healthy):**
```json
{
  "status": "healthy",
  "timestamp": "2025-11-08T10:30:00Z",
  "service": "cbam-importer-copilot",
  "checks": {
    "file_system": {
      "status": "healthy",
      "message": "All required directories accessible",
      "details": {"directories_checked": 3},
      "duration_ms": 2.5
    },
    "reference_data": {
      "status": "healthy",
      "message": "All reference data files present",
      "details": {
        "file_sizes": {
          "cn_codes": 524288,
          "cbam_rules": 16384,
          "emission_factors": 8192
        }
      },
      "duration_ms": 5.2
    },
    "python_dependencies": {
      "status": "healthy",
      "message": "All required Python packages installed",
      "details": {
        "packages": {
          "pandas": "2.0.0",
          "pydantic": "2.0.0",
          "jsonschema": "4.0.0"
        }
      },
      "duration_ms": 10.1
    }
  },
  "duration_ms": 17.8
}
```

**Use Cases:**
- Kubernetes readiness probes
- Load balancer health checks (production)
- Deployment verification

**Kubernetes Configuration:**
```yaml
readinessProbe:
  httpGet:
    path: /health/ready
    port: 8000
  initialDelaySeconds: 10
  periodSeconds: 10
  failureThreshold: 3
```

#### 3. Liveness Check (`/health/live`)

Verifies application is functioning correctly (not deadlocked).

```bash
curl http://localhost:8000/health/live
```

**Use Cases:**
- Kubernetes liveness probes
- Automatic restart on failure

**Kubernetes Configuration:**
```yaml
livenessProbe:
  httpGet:
    path: /health/live
    port: 8000
  initialDelaySeconds: 30
  periodSeconds: 30
  failureThreshold: 5
```

### CLI Testing

Test health checks from command line:

```bash
# Run all health checks
python backend/health.py --check all --json

# Run specific check
python backend/health.py --check ready

# Specify base path
python backend/health.py --check all --base-path /app/cbam
```

---

## Logging

### Configuration

The CBAM application uses structured JSON logging for production environments.

#### Development Configuration

```python
from backend.logging_config import configure_development_logging

# Human-readable logs for development
logger = configure_development_logging()
```

**Output:**
```
2025-11-08 10:30:00 - cbam.pipeline - INFO - Processing shipment S-12345
```

#### Production Configuration

```python
from backend.logging_config import configure_production_logging

# JSON logs for production
logger = configure_production_logging(log_dir="/var/log/cbam")
```

**Output:**
```json
{
  "timestamp": "2025-11-08T10:30:00.123456Z",
  "level": "INFO",
  "logger": "cbam.pipeline",
  "message": "Processing shipment S-12345",
  "service": "cbam-importer-copilot",
  "correlation_id": "550e8400-e29b-41d4-a716-446655440000",
  "thread": {"id": 12345, "name": "MainThread"},
  "process": {"id": 67890, "name": "MainProcess"},
  "source": {
    "file": "/app/cbam_pipeline.py",
    "line": 142,
    "function": "run"
  },
  "context": {
    "shipment_id": "S-12345",
    "quantity": 100,
    "supplier": "ACME Corp"
  }
}
```

### Using Structured Logging

#### Basic Usage

```python
from backend.logging_config import StructuredLogger

logger = StructuredLogger("cbam.pipeline")

# Log with context
logger.info(
    "Processing shipment",
    shipment_id="S-12345",
    quantity=100,
    supplier="ACME Corp"
)
```

#### Operation Tracking

```python
# Automatic timing of operations
with logger.operation("calculate_emissions", shipment_count=1000):
    # ... do work ...
    pass
```

**Output:**
```json
{
  "message": "Starting operation: calculate_emissions",
  "operation": "calculate_emissions",
  "shipment_count": 1000
}
{
  "message": "Completed operation: calculate_emissions",
  "operation": "calculate_emissions",
  "duration_ms": 1234.56,
  "status": "success",
  "shipment_count": 1000
}
```

#### Error Logging

```python
try:
    # ... code that might fail ...
    raise ValueError("Invalid CN code")
except Exception as e:
    logger.error(
        "Validation failed",
        exception=e,
        cn_code="ABC123",
        shipment_id="S-12345"
    )
```

### Correlation IDs

Track requests across distributed systems:

```python
from backend.logging_config import CorrelationContext

# Generate new correlation ID
correlation_id = CorrelationContext.new_correlation_id()

# All logs in this thread will include the correlation ID
logger.info("Request started", user_id="user-123")
logger.info("Nested operation", step="validation")

# Clear when done
CorrelationContext.clear_correlation_id()
```

### Log Files (Production)

When file logging is enabled, logs are written to:

```
/var/log/cbam/
├── cbam-importer.log           # All logs (human-readable)
├── cbam-importer.json.log      # All logs (JSON format)
└── cbam-importer-errors.log    # Errors only
```

**Log Rotation:**
- Max file size: 10 MB
- Backup count: 5
- Total max storage: ~50 MB per log file

---

## Metrics

### Available Metrics

#### Pipeline Metrics

| Metric | Type | Description |
|--------|------|-------------|
| `cbam_pipeline_executions_total` | Counter | Total pipeline executions (labels: status) |
| `cbam_pipeline_duration_seconds` | Histogram | Pipeline duration (labels: stage) |
| `cbam_pipeline_active` | Gauge | Currently active pipelines |
| `cbam_records_processed_total` | Counter | Records processed (labels: stage, status) |
| `cbam_records_per_second` | Gauge | Processing rate (labels: stage) |

#### Agent Metrics

| Metric | Type | Description |
|--------|------|-------------|
| `cbam_agent_executions_total` | Counter | Agent executions (labels: agent, status) |
| `cbam_agent_duration_seconds` | Histogram | Agent execution duration (labels: agent) |
| `cbam_agent_ms_per_record` | Summary | Milliseconds per record (labels: agent) |

#### Validation Metrics

| Metric | Type | Description |
|--------|------|-------------|
| `cbam_validation_results_total` | Counter | Validation results (labels: result) |
| `cbam_validation_errors_by_type` | Counter | Validation errors (labels: error_type) |
| `cbam_validation_duration_seconds` | Histogram | Validation duration |

#### Emissions Metrics

| Metric | Type | Description |
|--------|------|-------------|
| `cbam_emissions_calculated_tco2` | Counter | Total emissions calculated (tCO2) |
| `cbam_emissions_calculation_rate` | Gauge | Calculation rate (tCO2/second) |
| `cbam_calculation_method_total` | Counter | Calculations by method (labels: method) |
| `cbam_emissions_by_cn_code` | Counter | Emissions by CN code (labels: cn_code) |

#### System Metrics

| Metric | Type | Description |
|--------|------|-------------|
| `cbam_memory_usage_bytes` | Gauge | Current memory usage |
| `cbam_cpu_usage_percent` | Gauge | Current CPU usage |
| `cbam_disk_usage_bytes` | Gauge | Disk usage (labels: path) |

### Using Metrics

#### Initialize Metrics

```python
from backend.metrics import CBAMMetrics

metrics = CBAMMetrics()
```

#### Record Pipeline Execution

```python
# Record successful execution
metrics.record_pipeline_execution(
    status="success",
    duration_seconds=45.5,
    stage="total"
)

# Record failure
metrics.record_pipeline_execution(
    status="failed",
    duration_seconds=10.2,
    stage="intake"
)
```

#### Record Agent Execution

```python
metrics.record_agent_execution(
    agent="intake",
    status="success",
    duration_seconds=10.2,
    records_processed=1000
)
```

#### Using Decorators

```python
from backend.metrics import track_execution_time, track_agent_execution

@track_execution_time(metrics, "pipeline")
def run_pipeline():
    # ... pipeline code ...
    pass

@track_agent_execution(metrics, "intake")
def process_shipments():
    # ... agent code ...
    pass
```

#### Export Metrics

```python
from backend.metrics import MetricsExporter

exporter = MetricsExporter(metrics)

# Get Prometheus format
print(exporter.export_text())

# Push to Prometheus Pushgateway (for batch jobs)
exporter.push_to_gateway("http://localhost:9091", job="cbam-batch")
```

### Metrics Endpoint

For FastAPI deployment:

```python
from fastapi import FastAPI
from backend.metrics import CBAMMetrics, create_metrics_endpoint

app = FastAPI()
metrics = CBAMMetrics()

# Add metrics endpoint
metrics_route = create_metrics_endpoint(metrics)
app.add_api_route(**metrics_route)
```

Access metrics at: `http://localhost:8000/metrics`

---

## Dashboards

### Grafana Dashboard

A comprehensive Grafana dashboard is provided at `monitoring/grafana-dashboard.json`.

#### Features

1. **System Health Overview**
   - Pipeline success rate
   - Active pipelines
   - Total emissions calculated
   - Records processed

2. **Pipeline Performance**
   - Duration by stage (p50, p95, p99)
   - Execution rate
   - Throughput trends

3. **Agent Performance**
   - Execution duration by agent
   - Processing speed (ms/record)
   - Agent-specific metrics

4. **Validation & Errors**
   - Validation results distribution
   - Errors by type
   - Exception tracking

5. **Business Metrics**
   - Emissions calculation rate
   - Calculation method distribution
   - CN code distribution

6. **System Resources**
   - Memory usage
   - CPU usage
   - Application info

#### Import Dashboard

**Via UI:**
1. Open Grafana
2. Go to Dashboards → Import
3. Upload `monitoring/grafana-dashboard.json`

**Via API:**
```bash
curl -X POST \
  -H "Content-Type: application/json" \
  -d @monitoring/grafana-dashboard.json \
  http://admin:admin@localhost:3000/api/dashboards/db
```

**Via Provisioning:**
```yaml
# /etc/grafana/provisioning/dashboards/cbam.yml
apiVersion: 1
providers:
  - name: 'CBAM Dashboards'
    folder: 'CBAM'
    type: file
    options:
      path: /var/lib/grafana/dashboards/cbam
```

---

## Alerting

### Alert Rules

Critical production alerts are defined in `monitoring/alerts.yml`.

#### Alert Categories

1. **Availability** (Critical)
   - Service down
   - Health check failures
   - Readiness check failures

2. **Performance** (Warning)
   - High latency
   - Slow agent performance
   - Low throughput

3. **Errors** (Critical/Warning)
   - High error rate
   - Validation failures
   - Exception spikes

4. **Resources** (Warning/Critical)
   - High memory usage
   - High CPU usage
   - Low disk space

5. **SLA** (Critical)
   - Success rate below 99%
   - Latency above 10 minutes

#### Example Alerts

**Service Down (Critical):**
```yaml
- alert: CBAMServiceDown
  expr: up{job="cbam-importer-copilot"} == 0
  for: 2m
  annotations:
    summary: "CBAM service is down"
    impact: "Pipeline executions will fail"
```

**High Error Rate (Critical):**
```yaml
- alert: CBAMHighErrorRate
  expr: |
    (sum(rate(cbam_pipeline_executions_total{status="failed"}[5m]))
     / sum(rate(cbam_pipeline_executions_total[5m]))) > 0.05
  for: 5m
  annotations:
    summary: "Error rate above 5%"
```

### Alertmanager Configuration

```yaml
# alertmanager.yml
route:
  receiver: 'cbam-team'
  group_by: ['alertname', 'severity']

  routes:
    - match:
        severity: critical
      receiver: 'cbam-pagerduty'

    - match:
        severity: warning
      receiver: 'cbam-slack'

receivers:
  - name: 'cbam-team'
    email_configs:
      - to: 'cbam-team@greenlang.com'

  - name: 'cbam-pagerduty'
    pagerduty_configs:
      - service_key: '<your-key>'

  - name: 'cbam-slack'
    slack_configs:
      - api_url: '<webhook-url>'
        channel: '#cbam-alerts'
```

---

## Quick Start

### Development Setup

**1. Install dependencies:**
```bash
pip install -r requirements.txt
```

**2. Configure logging (optional):**
```python
from backend.logging_config import configure_development_logging
configure_development_logging()
```

**3. Run health check:**
```bash
python backend/health.py --check all
```

**4. Test metrics:**
```bash
python backend/metrics.py
```

### Local Monitoring Stack

**1. Create docker-compose.yml:**

```yaml
version: '3.8'

services:
  prometheus:
    image: prom/prometheus:latest
    ports:
      - "9090:9090"
    volumes:
      - ./monitoring/prometheus.yml:/etc/prometheus/prometheus.yml
      - ./monitoring/alerts.yml:/etc/prometheus/alerts.yml
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'

  grafana:
    image: grafana/grafana:latest
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin
    volumes:
      - grafana-storage:/var/lib/grafana

  alertmanager:
    image: prom/alertmanager:latest
    ports:
      - "9093:9093"
    volumes:
      - ./monitoring/alertmanager.yml:/etc/alertmanager/alertmanager.yml

volumes:
  grafana-storage:
```

**2. Start stack:**
```bash
docker-compose up -d
```

**3. Access:**
- Prometheus: http://localhost:9090
- Grafana: http://localhost:3000 (admin/admin)
- Alertmanager: http://localhost:9093

**4. Import dashboard:**
- Go to Grafana → Dashboards → Import
- Upload `monitoring/grafana-dashboard.json`

---

## Production Deployment

### Environment Variables

```bash
# Logging
export CBAM_LOG_LEVEL=INFO
export CBAM_LOG_DIR=/var/log/cbam
export CBAM_ENABLE_JSON_LOGGING=true

# Metrics
export CBAM_ENABLE_METRICS=true
export CBAM_METRICS_PORT=8000

# Health Checks
export CBAM_HEALTH_CHECK_ENABLED=true
```

### Kubernetes Deployment

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: cbam-importer
spec:
  replicas: 3
  template:
    metadata:
      labels:
        app: cbam-importer
      annotations:
        prometheus.io/scrape: "true"
        prometheus.io/port: "8000"
        prometheus.io/path: "/metrics"
    spec:
      containers:
      - name: cbam-importer
        image: cbam-importer:1.0.0
        ports:
        - containerPort: 8000
          name: metrics

        # Health checks
        livenessProbe:
          httpGet:
            path: /health/live
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 30

        readinessProbe:
          httpGet:
            path: /health/ready
            port: 8000
          initialDelaySeconds: 10
          periodSeconds: 10

        # Resources
        resources:
          limits:
            memory: "1Gi"
            cpu: "1000m"
          requests:
            memory: "512Mi"
            cpu: "500m"

        # Environment
        env:
        - name: CBAM_LOG_LEVEL
          value: "INFO"
        - name: CBAM_ENABLE_JSON_LOGGING
          value: "true"
```

### Log Aggregation

#### ELK Stack

```bash
# Filebeat configuration for shipping logs to Elasticsearch
filebeat.inputs:
- type: log
  paths:
    - /var/log/cbam/*.json.log
  json.keys_under_root: true
  json.add_error_key: true

output.elasticsearch:
  hosts: ["elasticsearch:9200"]
  index: "cbam-logs-%{+yyyy.MM.dd}"
```

#### CloudWatch

```python
import watchtower
import logging

logger = logging.getLogger()
logger.addHandler(watchtower.CloudWatchLogHandler(
    log_group='/aws/cbam/importer',
    stream_name='production-{machine}-{strftime:%Y-%m-%d}'
))
```

---

## Troubleshooting

### Common Issues

#### 1. Health Check Failing

**Symptom:** `/health/ready` returns 503

**Diagnosis:**
```bash
curl http://localhost:8000/health/ready | jq '.checks'
```

**Solutions:**
- Check file system permissions
- Verify reference data files exist
- Ensure Python dependencies installed

#### 2. No Metrics in Prometheus

**Symptom:** Prometheus can't scrape metrics

**Diagnosis:**
```bash
# Check if metrics endpoint is accessible
curl http://localhost:8000/metrics

# Check Prometheus targets
curl http://localhost:9090/api/v1/targets
```

**Solutions:**
- Verify application is exposing `/metrics`
- Check Prometheus configuration
- Verify network connectivity
- Check firewall rules

#### 3. High Memory Usage

**Symptom:** Alert `CBAMHighMemoryUsage` firing

**Diagnosis:**
```python
# Check current memory usage
from backend.metrics import get_metrics
metrics = get_metrics()
metrics.update_system_metrics()
```

**Solutions:**
- Reduce batch sizes
- Enable streaming processing
- Increase memory limits
- Check for memory leaks

#### 4. Logs Not in JSON Format

**Symptom:** Logs still in plain text format

**Solution:**
```python
# Ensure JSON logging is enabled
from backend.logging_config import configure_production_logging
configure_production_logging(log_dir="/var/log/cbam")
```

---

## SLA & Performance Targets

### Service Level Objectives (SLOs)

| Metric | Target | Measurement |
|--------|--------|-------------|
| **Availability** | 99.9% | `up{job="cbam"} == 1` |
| **Success Rate** | 99% | Pipeline executions without errors |
| **Latency (p95)** | < 10 minutes | 95% of pipelines complete in < 10 min |
| **Latency (p99)** | < 15 minutes | 99% of pipelines complete in < 15 min |
| **Error Rate** | < 1% | Failed executions / total executions |
| **Throughput** | > 100 records/sec | Average processing rate |

### Performance Baselines

| Stage | Target Duration | Throughput |
|-------|----------------|------------|
| **Intake** | < 2 min | > 166 records/sec |
| **Calculate** | < 5 min | > 33 records/sec |
| **Package** | < 3 min | > 55 records/sec |
| **Total** | < 10 min | > 16 records/sec |

*Based on 10,000 shipment benchmark*

### Resource Limits

| Resource | Limit | Alert Threshold |
|----------|-------|----------------|
| Memory | 1 GB | 500 MB |
| CPU | 100% (1 core) | 80% |
| Disk | 10 GB | 90% full |

---

## Advanced Topics

### Custom Metrics

Add business-specific metrics:

```python
from prometheus_client import Counter, Gauge

# Custom counter
custom_metric = Counter(
    'cbam_custom_operations_total',
    'Custom operations',
    ['operation_type'],
    registry=metrics.registry
)

custom_metric.labels(operation_type='special').inc()
```

### Distributed Tracing

Integrate with OpenTelemetry:

```python
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.jaeger.thrift import JaegerExporter

# Configure tracing
trace.set_tracer_provider(TracerProvider())
jaeger_exporter = JaegerExporter(
    agent_host_name='localhost',
    agent_port=6831,
)
trace.get_tracer_provider().add_span_processor(
    BatchSpanProcessor(jaeger_exporter)
)

tracer = trace.get_tracer(__name__)

# Use in code
with tracer.start_as_current_span("process_shipment"):
    # ... work ...
    pass
```

### Error Tracking (Sentry)

```python
import sentry_sdk

sentry_sdk.init(
    dsn="https://your-sentry-dsn",
    environment="production",
    traces_sample_rate=0.1,
)

# Errors automatically captured
try:
    # ... code ...
    pass
except Exception as e:
    sentry_sdk.capture_exception(e)
```

---

## Support

### Resources

- **Documentation:** https://wiki.greenlang.com/cbam
- **Runbooks:** https://wiki.greenlang.com/cbam/runbooks
- **Slack:** #cbam-support
- **Email:** cbam-team@greenlang.com

### On-Call Escalation

1. **L1 Support:** Slack #cbam-alerts (5 min response)
2. **L2 Engineering:** PagerDuty (15 min response)
3. **L3 Architecture:** Phone (30 min response)

---

**Monitoring implemented by Team A3: GL-CBAM Monitoring & Observability**

Production ready. Zero configuration required for basic functionality.
