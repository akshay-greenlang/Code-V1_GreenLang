# CSRD Reporting Platform - Monitoring & Observability Guide

**Version:** 1.0.0
**Date:** 2025-11-08
**Team:** Team B3 - GL-CSRD Monitoring & Observability
**Status:** Production-Ready

---

## Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Health Checks](#health-checks)
4. [Logging](#logging)
5. [Metrics](#metrics)
6. [Dashboards](#dashboards)
7. [Alerting](#alerting)
8. [Error Tracking](#error-tracking)
9. [ESRS-Specific Monitoring](#esrs-specific-monitoring)
10. [Troubleshooting](#troubleshooting)
11. [Best Practices](#best-practices)

---

## Overview

The CSRD Reporting Platform monitoring and observability infrastructure provides comprehensive visibility into:

- **System Health**: Real-time health checks for all components
- **Performance**: Request latency, throughput, and resource utilization
- **ESRS Compliance**: Data coverage, validation status, and compliance deadlines
- **Data Quality**: Completeness, accuracy, and timeliness metrics
- **Errors**: Centralized error tracking with full context
- **Costs**: LLM API usage and cost tracking

### Key Features

- ✅ Kubernetes-compatible health checks (liveness, readiness, startup)
- ✅ ESRS-specific health endpoints for each standard (E1-E5, S1-S4, G1)
- ✅ Structured JSON logging with ESRS context
- ✅ Prometheus metrics for all operations
- ✅ Grafana dashboards for compliance monitoring
- ✅ Sentry integration for error tracking
- ✅ Compliance deadline alerts
- ✅ Data gap detection and alerting

---

## Architecture

### Monitoring Stack

```
┌─────────────────────────────────────────────────────────────┐
│                    CSRD Platform Application                 │
│                                                               │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │   Health     │  │   Logging    │  │   Metrics    │      │
│  │   Endpoints  │  │   (JSON)     │  │  (Prometheus)│      │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘      │
│         │                  │                  │               │
└─────────┼──────────────────┼──────────────────┼──────────────┘
          │                  │                  │
          │                  │                  │
    ┌─────▼─────┐      ┌────▼────┐       ┌────▼────┐
    │Kubernetes │      │  ELK    │       │Prometheus│
    │  Probes   │      │  Stack  │       │  Server  │
    └───────────┘      └────┬────┘       └────┬────┘
                            │                  │
                       ┌────▼────┐        ┌───▼────┐
                       │ Kibana  │        │Grafana │
                       └─────────┘        └────────┘

    ┌──────────────────────────────────────────┐
    │         Sentry (Error Tracking)          │
    └──────────────────────────────────────────┘
```

### Components

| Component | Purpose | Port |
|-----------|---------|------|
| FastAPI App | Main application with health endpoints | 8000 |
| Prometheus | Metrics collection and storage | 9090 |
| Grafana | Metrics visualization and dashboards | 3000 |
| Alertmanager | Alert routing and management | 9093 |
| ELK Stack | Log aggregation and analysis | 5601 |
| Sentry | Error tracking and performance | - |

---

## Health Checks

### Endpoints

The CSRD Platform provides comprehensive health check endpoints:

#### 1. Basic Health Check

```bash
GET /health
GET /health/
```

**Purpose:** Simple liveness check
**Use Case:** Quick verification that service is running
**Response:**

```json
{
  "status": "healthy",
  "timestamp": "2025-11-08T10:30:00Z",
  "version": "1.0.0"
}
```

#### 2. Liveness Probe

```bash
GET /health/live
```

**Purpose:** Kubernetes liveness probe
**Use Case:** Detect if application is deadlocked or needs restart
**Response:**

```json
{
  "status": "alive",
  "timestamp": "2025-11-08T10:30:00Z",
  "checks": {
    "uptime_seconds": 86400,
    "uptime_human": "1d 0h 0m 0s"
  }
}
```

#### 3. Readiness Probe

```bash
GET /health/ready
```

**Purpose:** Kubernetes readiness probe
**Use Case:** Determine if service can accept traffic
**Checks:**
- Database connectivity
- Cache availability
- Disk space
- Memory availability
- ESRS data availability

**Response:**

```json
{
  "status": "ready",
  "timestamp": "2025-11-08T10:30:00Z",
  "checks": {
    "database": {
      "healthy": true,
      "message": "Database connection OK",
      "response_time_ms": 5,
      "connection_pool": {
        "active": 2,
        "idle": 8,
        "total": 10
      }
    },
    "cache": {
      "healthy": true,
      "message": "Cache connection OK",
      "response_time_ms": 2,
      "memory_usage_mb": 45.2,
      "hit_rate": 0.87
    },
    "disk_space": {
      "healthy": true,
      "message": "Disk usage: 65.0%",
      "total_gb": 100,
      "used_gb": 65,
      "free_gb": 35,
      "percent_used": 65.0,
      "threshold": 85.0
    },
    "memory": {
      "healthy": true,
      "message": "Memory usage: 75.0%",
      "total_gb": 16,
      "available_gb": 4,
      "used_gb": 12,
      "percent_used": 75.0,
      "threshold": 90.0
    },
    "esrs_data": {
      "healthy": true,
      "message": "ESRS data ready",
      "data_points_available": 5432,
      "last_sync": "2025-11-08T08:30:00Z"
    }
  }
}
```

#### 4. Startup Probe

```bash
GET /health/startup
```

**Purpose:** Kubernetes startup probe
**Use Case:** Verify application initialization is complete
**Checks:**
- Database schema loaded
- Agents initialized
- Configuration loaded
- ESRS standards loaded

#### 5. ESRS Health Check

```bash
GET /health/esrs
```

**Purpose:** Comprehensive ESRS/CSRD compliance health check
**Use Case:** Monitor compliance readiness across all standards
**Response:**

```json
{
  "status": "healthy",
  "timestamp": "2025-11-08T10:30:00Z",
  "checks": {
    "ESRS-E1": {
      "status": "healthy",
      "standard": "ESRS-E1",
      "data_point_coverage": 0.96,
      "required_data_points": 150,
      "available_data_points": 144,
      "validation_errors": 0,
      "last_update": "2025-11-08T09:00:00Z",
      "completeness_score": 96.0
    },
    "ESRS-S1": {
      "status": "warning",
      "standard": "ESRS-S1",
      "data_point_coverage": 0.82,
      "required_data_points": 120,
      "available_data_points": 98,
      "validation_errors": 3,
      "last_update": "2025-11-08T08:00:00Z",
      "completeness_score": 82.0
    },
    "compliance_deadlines": {
      "status": "warning",
      "next_deadline": "2025-04-30T00:00:00Z",
      "days_until_deadline": 45,
      "deadline_type": "Annual CSRD Report",
      "message": "45 days until next compliance deadline"
    },
    "data_quality": {
      "status": "healthy",
      "overall_quality_score": 94.5,
      "validation_pass_rate": 0.96,
      "completeness_score": 0.93,
      "accuracy_score": 0.95,
      "timeliness_score": 0.97,
      "issues": {
        "missing_data_points": 12,
        "validation_errors": 5,
        "outdated_records": 3
      }
    }
  }
}
```

#### 6. ESRS Standard-Specific Health

```bash
GET /health/esrs/{standard}
```

**Example:** `GET /health/esrs/E1`

**Purpose:** Detailed health check for specific ESRS standard
**Supported Standards:** E1, E2, E3, E4, E5, S1, S2, S3, S4, G1, ESRS-2

### Kubernetes Integration

```yaml
# deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: csrd-platform
spec:
  template:
    spec:
      containers:
      - name: csrd-app
        livenessProbe:
          httpGet:
            path: /health/live
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
          timeoutSeconds: 5
          failureThreshold: 3

        readinessProbe:
          httpGet:
            path: /health/ready
            port: 8000
          initialDelaySeconds: 10
          periodSeconds: 5
          timeoutSeconds: 3
          failureThreshold: 3

        startupProbe:
          httpGet:
            path: /health/startup
            port: 8000
          initialDelaySeconds: 0
          periodSeconds: 10
          timeoutSeconds: 3
          failureThreshold: 30
```

---

## Logging

### Structured JSON Logging

All logs are formatted as JSON with ESRS/CSRD context for easy parsing and analysis.

#### Log Format

```json
{
  "timestamp": "2025-11-08T10:30:00.123Z",
  "level": "INFO",
  "logger": "csrd.agents.intake",
  "message": "Processing ESRS E1 data",
  "source": {
    "file": "intake_agent.py",
    "line": 145,
    "function": "process_data"
  },
  "request_id": "req-abc123",
  "user_id": "user-456",
  "company_id": "comp-789",
  "esrs_standard": "E1",
  "agent_name": "intake_agent",
  "service": "csrd-reporting-platform",
  "environment": "production",
  "process": {
    "pid": 12345,
    "name": "MainProcess"
  },
  "thread": {
    "id": 67890,
    "name": "Thread-1"
  }
}
```

#### Usage in Code

```python
from backend.logging_config import setup_structured_logging, get_logger, LogContext

# Setup logging
setup_structured_logging(
    log_level='INFO',
    log_file='logs/csrd-app.log',
    enable_json=True
)

# Get logger
logger = get_logger(__name__)

# Log with ESRS context
with LogContext(
    request_id="req-12345",
    company_id="comp-acme",
    esrs_standard="E1"
):
    logger.info("Processing ESRS E1 climate data")
    logger.warning("Missing data points detected", extra={'missing_count': 5})
```

#### Audit Logging

For compliance tracking:

```python
from backend.logging_config import get_audit_logger

audit = get_audit_logger()

# Log data access
audit.log_data_access(
    user_id="user-123",
    company_id="comp-acme",
    esrs_standard="E1",
    action="read",
    data_points=150,
    success=True
)

# Log report generation
audit.log_report_generation(
    user_id="user-123",
    company_id="comp-acme",
    report_type="annual",
    reporting_period="2024",
    esrs_standards=["E1", "E2", "S1"],
    success=True,
    duration_seconds=45.2
)

# Log validation
audit.log_validation(
    company_id="comp-acme",
    esrs_standard="E1",
    validation_type="data_completeness",
    passed=True,
    errors_count=0,
    warnings_count=3
)
```

### Log Levels

| Level | When to Use | Examples |
|-------|-------------|----------|
| DEBUG | Detailed diagnostic information | Variable values, function calls |
| INFO | General informational messages | Request processed, task completed |
| WARNING | Warning messages for non-critical issues | Missing optional data, deprecated features |
| ERROR | Error messages for failures | Validation errors, API failures |
| CRITICAL | Critical errors requiring immediate attention | System failures, data corruption |

---

## Metrics

### Prometheus Metrics

The platform exposes comprehensive Prometheus metrics at `/metrics`.

#### HTTP/API Metrics

```promql
# Total HTTP requests
csrd_http_requests_total{method="GET", endpoint="/api/v1/companies", status_code="200"}

# Request duration (histogram)
csrd_http_request_duration_seconds{method="GET", endpoint="/api/v1/companies"}

# Request size
csrd_http_request_size_bytes{method="POST", endpoint="/api/v1/data"}

# Response size
csrd_http_response_size_bytes{method="GET", endpoint="/api/v1/reports"}
```

#### ESRS Data Coverage Metrics

```promql
# Data point coverage ratio (0-1)
csrd_esrs_data_point_coverage_ratio{esrs_standard="E1", company_id="comp-123"}

# Required data points
csrd_esrs_required_data_points_total{esrs_standard="E1"}

# Available data points
csrd_esrs_available_data_points_total{esrs_standard="E1", company_id="comp-123"}

# Missing data points
csrd_esrs_missing_data_points_total{esrs_standard="E1", company_id="comp-123"}
```

#### ESRS Data Quality Metrics

```promql
# Overall data quality score (0-100)
csrd_esrs_data_quality_score{esrs_standard="E1", company_id="comp-123"}

# Data completeness ratio (0-1)
csrd_esrs_data_completeness_ratio{esrs_standard="E1", company_id="comp-123"}

# Data accuracy ratio (0-1)
csrd_esrs_data_accuracy_ratio{esrs_standard="E1", company_id="comp-123"}

# Data timeliness ratio (0-1)
csrd_esrs_data_timeliness_ratio{esrs_standard="E1", company_id="comp-123"}
```

#### Validation Metrics

```promql
# Validation checks
csrd_validation_checks_total{validation_type="schema", esrs_standard="E1", status="passed"}

# Validation errors
csrd_validation_errors_total{error_type="missing_field", esrs_standard="E1", severity="error"}

# Validation duration
csrd_validation_duration_seconds{validation_type="schema", esrs_standard="E1"}

# Validation rules applied
csrd_validation_rules_applied_total{esrs_standard="E1"}
```

#### Agent Performance Metrics

```promql
# Agent executions
csrd_agent_execution_total{agent_name="intake", operation="process", status="success"}

# Agent execution duration
csrd_agent_execution_duration_seconds{agent_name="intake", operation="process"}

# Agent failures
csrd_agent_execution_failures_total{agent_name="intake", error_type="validation_error"}

# Agent memory usage
csrd_agent_memory_usage_bytes{agent_name="intake"}

# Active agent tasks
csrd_agent_active_tasks{agent_name="intake"}
```

#### Report Generation Metrics

```promql
# Reports generated
csrd_reports_generated_total{report_type="annual", format="pdf", status="success"}

# Report generation duration
csrd_report_generation_duration_seconds{report_type="annual", format="pdf"}

# Report generation failures
csrd_report_generation_failures_total{report_type="annual", error_type="xbrl_error"}

# Report size
csrd_report_size_bytes{report_type="annual", format="pdf"}
```

#### LLM API Metrics

```promql
# LLM API calls
csrd_llm_api_calls_total{provider="openai", model="gpt-4", status="success"}

# LLM API duration
csrd_llm_api_duration_seconds{provider="openai", model="gpt-4"}

# LLM tokens used
csrd_llm_api_tokens_used_total{provider="openai", model="gpt-4", type="input"}

# LLM API cost
csrd_llm_api_cost_usd_total{provider="openai", model="gpt-4"}

# LLM API errors
csrd_llm_api_errors_total{provider="openai", model="gpt-4", error_type="rate_limit"}
```

#### Compliance Metrics

```promql
# Days until deadline
csrd_compliance_deadline_days_remaining{deadline_type="Annual Report", company_id="comp-123"}

# Compliance tasks
csrd_compliance_tasks_total{status="pending", company_id="comp-123"}

# Compliance score (0-100)
csrd_compliance_score{company_id="comp-123"}
```

### Usage in Code

```python
from backend.metrics import (
    record_http_request,
    record_esrs_coverage,
    record_llm_usage,
    MetricsTimer,
    MetricsCounter
)

# Record HTTP request
record_http_request(
    method="GET",
    endpoint="/api/v1/companies",
    status_code=200,
    duration=0.125,
    request_size=256,
    response_size=4096
)

# Record ESRS coverage
record_esrs_coverage(
    esrs_standard="E1",
    company_id="comp-123",
    required_points=150,
    available_points=142
)

# Using timer context manager
with MetricsTimer(
    agent_execution_duration_seconds,
    labels={'agent_name': 'intake', 'operation': 'process'}
):
    # ... do work ...
    pass

# Using counter context manager
with MetricsCounter(
    reports_generated_total,
    labels={'report_type': 'annual', 'format': 'pdf'},
    error_counter=report_generation_failures_total
):
    # ... generate report ...
    pass
```

---

## Dashboards

### Grafana Dashboards

The platform includes a comprehensive Grafana dashboard for CSRD compliance monitoring.

**Dashboard File:** `monitoring/grafana-csrd-dashboard.json`

#### Dashboard Sections

1. **ESRS Compliance Overview**
   - Overall compliance score
   - Days until deadline
   - ESRS E1, S1, G1 coverage gauges
   - Data quality score

2. **ESRS Data Coverage by Standard**
   - Bar gauge showing coverage for all ESRS standards
   - Detailed table with required/available/missing data points

3. **Validation & Data Quality**
   - Validation errors over time
   - Pie chart of validation errors by type

4. **Agent Performance**
   - Agent execution duration (p95)
   - Agent success rate

5. **Report Generation**
   - Reports generated count
   - Report generation duration by type

6. **API Performance**
   - API request rate
   - API latency (p95)

7. **LLM API Usage & Costs**
   - LLM API cost (last 24h)
   - Token usage by provider
   - API call duration

8. **System Health**
   - Service health status
   - Data records processed
   - Authentication success rate
   - Encryption operations

#### Importing Dashboard

```bash
# Using Grafana API
curl -X POST \
  http://localhost:3000/api/dashboards/db \
  -H 'Content-Type: application/json' \
  -H 'Authorization: Bearer <api-key>' \
  -d @monitoring/grafana-csrd-dashboard.json
```

#### Dashboard Variables

- `company_id`: Filter by company
- `esrs_standard`: Filter by ESRS standard(s)

---

## Alerting

### Alert Rules

Alert rules are defined in `monitoring/alerts/alerts-csrd.yml`.

#### Alert Categories

1. **Compliance Deadline Alerts**
   - `ComplianceDeadlineUrgent`: < 7 days
   - `ComplianceDeadlineWarning`: < 30 days
   - `ComplianceDeadlineInfo`: < 60 days

2. **ESRS Data Coverage Alerts**
   - `ESRSDataCoverageCritical`: < 70% coverage
   - `ESRSDataCoverageLow`: < 85% coverage
   - `ESRSMissingDataPoints`: > 50 missing points
   - `ESRSMissingDataPointsCritical`: > 100 missing points

3. **ESRS Data Quality Alerts**
   - `ESRSDataQualityLow`: < 80 quality score
   - `ESRSDataQualityCritical`: < 60 quality score
   - `ESRSDataCompletenessLow`: < 85% complete
   - `ESRSDataAccuracyLow`: < 90% accurate
   - `ESRSDataTimelinessIssue`: < 90% timely

4. **ESRS Validation Alerts**
   - `ESRSValidationErrorsHigh`: > 0.5 errors/sec
   - `ESRSValidationWarningsHigh`: > 1.0 warnings/sec
   - `ESRSValidationRulesNotApplied`: No rules applied

5. **ESRS Standard-Specific Alerts**
   - `ESRSE1EmissionsDataMissing`: E1 climate data gaps
   - `ESRSS1WorkforceDataMissing`: S1 workforce data gaps
   - `ESRSG1GovernanceDataMissing`: G1 governance data gaps

#### Alert Severity Levels

| Severity | Description | Response Time |
|----------|-------------|---------------|
| critical | Immediate compliance impact | < 1 hour |
| warning | Approaching thresholds | < 24 hours |
| info | Awareness only | < 7 days |

#### Alert Routing

Configure Alertmanager to route alerts:

```yaml
# alertmanager.yml
route:
  group_by: ['alertname', 'esrs_standard']
  group_wait: 10s
  group_interval: 10s
  repeat_interval: 12h
  receiver: 'csrd-team'
  routes:
  - match:
      severity: critical
    receiver: 'csrd-oncall'
    continue: true
  - match:
      severity: warning
    receiver: 'csrd-team'

receivers:
- name: 'csrd-oncall'
  pagerduty_configs:
  - service_key: '<pagerduty-key>'
  slack_configs:
  - api_url: '<slack-webhook-url>'
    channel: '#csrd-critical'

- name: 'csrd-team'
  slack_configs:
  - api_url: '<slack-webhook-url>'
    channel: '#csrd-alerts'
```

---

## Error Tracking

### Sentry Integration

The platform uses Sentry for error tracking and performance monitoring.

#### Initialization

```python
from backend.error_tracking import init_sentry

init_sentry(
    dsn="https://your-sentry-dsn@sentry.io/project-id",
    environment="production",
    release="csrd-platform@1.0.0",
    traces_sample_rate=0.1,
    enable_tracing=True
)
```

#### Setting Context

```python
from backend.error_tracking import (
    set_esrs_context,
    set_user_context,
    set_agent_context,
    add_breadcrumb
)

# Set ESRS context
set_esrs_context(
    esrs_standard="E1",
    company_id="comp-123",
    reporting_period="2024",
    data_points=150
)

# Set user context
set_user_context(
    user_id="user-456",
    username="john.doe",
    email="john@example.com",
    company_id="comp-123"
)

# Set agent context
set_agent_context(
    agent_name="intake",
    operation="process_data",
    input_size=5000,
    version="1.0.0"
)

# Add breadcrumb
add_breadcrumb(
    message="Starting data validation",
    category="validation",
    level="info",
    data={"records": 150}
)
```

#### Capturing Errors

```python
from backend.error_tracking import (
    capture_exception,
    capture_message,
    track_validation_error,
    track_compliance_issue
)

# Capture exception
try:
    process_data()
except Exception as e:
    capture_exception(
        exception=e,
        tags={"esrs_standard": "E1"},
        extra={"data_point": "emissions_scope_1"}
    )

# Track validation error
track_validation_error(
    esrs_standard="E1",
    error_type="missing_data",
    error_message="Scope 1 emissions data missing",
    company_id="comp-123",
    data_point="emissions_scope_1"
)

# Track compliance issue
track_compliance_issue(
    issue_type="deadline_approaching",
    description="Annual report due in 15 days",
    company_id="comp-123",
    deadline="2025-04-30",
    severity="warning"
)
```

#### Using Decorators

```python
from backend.error_tracking import monitor_errors, monitor_performance

@monitor_errors(agent_name="intake", operation="process_data")
def process_data(data):
    # ... process data ...
    pass

@monitor_performance(operation_name="data_validation")
def validate_data(data):
    # ... validate data ...
    pass
```

---

## ESRS-Specific Monitoring

### Coverage Monitoring

Track data point coverage for each ESRS standard:

```promql
# E1 - Climate Change
csrd_esrs_data_point_coverage_ratio{esrs_standard="E1"}

# E2 - Pollution
csrd_esrs_data_point_coverage_ratio{esrs_standard="E2"}

# S1 - Own Workforce
csrd_esrs_data_point_coverage_ratio{esrs_standard="S1"}

# G1 - Business Conduct
csrd_esrs_data_point_coverage_ratio{esrs_standard="G1"}
```

### Compliance Deadline Tracking

Monitor days until compliance deadlines:

```promql
# Days until next deadline
csrd_compliance_deadline_days_remaining{deadline_type="Annual Report"}

# Alert when < 30 days
csrd_compliance_deadline_days_remaining < 30
```

### Data Quality Monitoring

Track data quality across dimensions:

```promql
# Overall quality score
csrd_esrs_data_quality_score{esrs_standard="E1"}

# Completeness
csrd_esrs_data_completeness_ratio{esrs_standard="E1"}

# Accuracy
csrd_esrs_data_accuracy_ratio{esrs_standard="E1"}

# Timeliness
csrd_esrs_data_timeliness_ratio{esrs_standard="E1"}
```

---

## Troubleshooting

### Common Issues

#### 1. Health Check Failing

**Symptoms:** `/health/ready` returns 503

**Diagnosis:**
```bash
curl http://localhost:8000/health/ready | jq
```

**Solutions:**
- Check database connectivity
- Verify cache (Redis) is running
- Check disk space (< 85% threshold)
- Check memory usage (< 90% threshold)

#### 2. Missing Metrics

**Symptoms:** Metrics not appearing in Prometheus

**Diagnosis:**
```bash
# Check if metrics endpoint is accessible
curl http://localhost:8000/metrics

# Check Prometheus targets
curl http://localhost:9090/api/v1/targets
```

**Solutions:**
- Verify Prometheus scrape configuration
- Check network connectivity
- Verify service discovery is working

#### 3. Alerts Not Firing

**Symptoms:** Expected alerts not received

**Diagnosis:**
```bash
# Check Prometheus rules
curl http://localhost:9090/api/v1/rules

# Check Alertmanager status
curl http://localhost:9093/api/v2/status
```

**Solutions:**
- Verify alert rules are loaded
- Check Alertmanager configuration
- Verify notification channels (Slack, PagerDuty, etc.)

#### 4. High ESRS Data Gap Alerts

**Symptoms:** Frequent `ESRSMissingDataPoints` alerts

**Diagnosis:**
```bash
# Check missing data points by standard
curl "http://localhost:9090/api/v1/query?query=csrd_esrs_missing_data_points_total"
```

**Solutions:**
- Review data collection processes
- Prioritize data collection for standards with gaps
- Check data source integrations

---

## Best Practices

### 1. Monitoring Configuration

- **Health Checks**: Use all three Kubernetes probes (liveness, readiness, startup)
- **Metrics Retention**: Keep metrics for at least 30 days for compliance auditing
- **Log Retention**: Keep audit logs for 7 years per CSRD requirements
- **Alert Tuning**: Review and tune alert thresholds monthly

### 2. ESRS Compliance Monitoring

- **Daily Reviews**: Review ESRS coverage dashboards daily
- **Weekly Reports**: Generate weekly compliance status reports
- **Deadline Tracking**: Set alerts at 60, 30, and 7 days before deadlines
- **Data Quality**: Monitor data quality scores daily

### 3. Performance Monitoring

- **SLOs**: Define SLOs for key operations:
  - API latency: p95 < 200ms
  - Report generation: p95 < 60s
  - Data validation: p95 < 30s
  - Agent execution: p95 < 120s

- **Capacity Planning**: Monitor resource usage trends
- **Cost Optimization**: Track LLM API costs and optimize usage

### 4. Security Monitoring

- **Authentication**: Monitor authentication failures
- **Encryption**: Track encryption operation success rates
- **Audit Logs**: Review audit logs regularly
- **Access Patterns**: Monitor unusual access patterns

### 5. Incident Response

- **Runbooks**: Maintain runbooks for common issues
- **Escalation**: Define escalation paths for critical alerts
- **Post-Mortems**: Conduct post-mortems for incidents
- **Continuous Improvement**: Update monitoring based on incidents

---

## Quick Reference

### Health Check URLs

```
/health              - Basic health check
/health/live         - Liveness probe
/health/ready        - Readiness probe
/health/startup      - Startup probe
/health/esrs         - ESRS compliance health
/health/esrs/{std}   - Specific ESRS standard health
/metrics             - Prometheus metrics
```

### Key Metrics

```
csrd_esrs_data_point_coverage_ratio       - ESRS data coverage
csrd_compliance_deadline_days_remaining   - Days to deadline
csrd_esrs_data_quality_score              - Data quality score
csrd_validation_errors_total              - Validation errors
csrd_agent_execution_duration_seconds     - Agent performance
csrd_llm_api_cost_usd_total               - LLM API costs
```

### Important Alerts

```
ComplianceDeadlineUrgent        - < 7 days to deadline
ESRSDataCoverageCritical        - < 70% coverage
ESRSMissingDataPointsCritical   - > 100 missing points
ESRSDataQualityCritical         - < 60 quality score
```

---

## Support

For monitoring and observability support:

- **Documentation**: https://docs.greenlang.io/monitoring
- **Runbooks**: https://docs.greenlang.io/runbooks
- **Team**: Team B3 - GL-CSRD Monitoring & Observability
- **Slack**: #csrd-monitoring

---

**End of Monitoring & Observability Guide**
