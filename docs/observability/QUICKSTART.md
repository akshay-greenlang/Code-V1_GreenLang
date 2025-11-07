# GreenLang Observability Quick Start Guide

## Getting Started in 5 Minutes

### Step 1: Start the Monitoring Stack (1 minute)

```bash
cd observability
docker-compose -f docker-compose-monitoring.yml up -d
```

This starts:
- Prometheus (metrics): http://localhost:9090
- Grafana (dashboards): http://localhost:3000
- Jaeger (tracing): http://localhost:16686
- Alertmanager: http://localhost:9093

### Step 2: Initialize Observability in Your App (2 minutes)

```python
from greenlang.observability import (
    configure_logging,
    get_metrics_collector,
    get_tracing_manager,
    get_health_checker,
)

# Configure structured logging
configure_logging(level="INFO", format_json=True)

# Start metrics collection
collector = get_metrics_collector()
collector.start_collection(port=8000)

# Initialize tracing
tracing_manager = get_tracing_manager()

# Start health checks
health_checker = get_health_checker()
health_checker.start_background_checks()

print("Observability initialized!")
```

### Step 3: Add Observability to Your Code (2 minutes)

```python
from greenlang.observability import (
    get_logger,
    LogContext,
    trace_operation,
    track_execution,
)

# Get structured logger
logger = get_logger(__name__, LogContext(component="api"))

@trace_operation("process_data")
@track_execution(pipeline="data_processing", tenant_id="default")
def process_data(data):
    logger.info("Processing started", data_size=len(data))

    try:
        result = transform(data)
        logger.info("Processing completed", result_size=len(result))
        return result
    except Exception as e:
        logger.error("Processing failed", exception=e)
        raise
```

### Step 4: View Your Dashboards

1. Open Grafana: http://localhost:3000
2. Login: admin / greenlang2024
3. Navigate to Dashboards
4. Open "GreenLang System Overview"

You'll see:
- Request rates and latency
- CPU and memory usage
- Error rates
- Cache hit rates
- Active executions

## What You Get

### Metrics (Prometheus)

**Automatic metrics for**:
- Pipeline executions (rate, duration, success/failure)
- API requests (rate, latency, status codes)
- Resource usage (CPU, memory, disk)
- Cache operations (hits, misses)
- Database queries (count, duration)

**Access at**: http://localhost:9090

### Logs (Loki + Grafana)

**Structured JSON logs with**:
- Automatic correlation IDs
- Rich context (tenant, component, operation)
- Error tracking with stack traces
- Log aggregation and search

**Query in Grafana**: http://localhost:3000/explore

### Traces (Jaeger)

**Distributed tracing with**:
- End-to-end request visibility
- Performance bottleneck identification
- Error tracking across services
- Critical path analysis

**Access at**: http://localhost:16686

### Health Checks

**Kubernetes-compatible endpoints**:
- `/health/live` - Liveness probe
- `/health/ready` - Readiness probe
- `/health/startup` - Startup probe

## Sample Outputs

### Metrics Export (Prometheus Format)

```
# HELP gl_pipeline_runs_total Total pipeline runs
# TYPE gl_pipeline_runs_total counter
gl_pipeline_runs_total{pipeline="data_processing",status="success",tenant_id="default"} 142.0

# HELP gl_pipeline_duration_seconds Pipeline execution time
# TYPE gl_pipeline_duration_seconds histogram
gl_pipeline_duration_seconds_bucket{pipeline="data_processing",tenant_id="default",le="0.5"} 95.0
gl_pipeline_duration_seconds_bucket{pipeline="data_processing",tenant_id="default",le="1.0"} 132.0
gl_pipeline_duration_seconds_sum{pipeline="data_processing",tenant_id="default"} 45.6
gl_pipeline_duration_seconds_count{pipeline="data_processing",tenant_id="default"} 142.0
```

### Structured Log Entry (JSON)

```json
{
  "timestamp": "2025-01-15T10:30:45.123456Z",
  "level": "INFO",
  "message": "Processing completed successfully",
  "tenant_id": "customer1",
  "component": "api",
  "operation": "process_data",
  "trace_id": "a1b2c3d4e5f6g7h8",
  "span_id": "1a2b3c4d",
  "data": {
    "request_id": "req-12345",
    "duration_ms": 145,
    "items_processed": 250,
    "status": "success"
  }
}
```

### Health Check Response

```json
{
  "status": "healthy",
  "version": "1.0.0",
  "timestamp": "2025-01-15T10:31:00.000000Z",
  "uptime_seconds": 3600.5,
  "checks": [
    {
      "name": "liveness",
      "status": "healthy",
      "message": "Application is alive",
      "duration_ms": 0.5
    },
    {
      "name": "readiness",
      "status": "healthy",
      "message": "Application is ready",
      "duration_ms": 1.2
    },
    {
      "name": "disk_space",
      "status": "healthy",
      "message": "Disk space OK: 125.5GB free",
      "duration_ms": 2.1,
      "details": {
        "free_gb": 125.5,
        "used_percent": 35.2
      }
    },
    {
      "name": "memory",
      "status": "healthy",
      "message": "Memory usage OK: 45.2%",
      "duration_ms": 1.0,
      "details": {
        "used_percent": 45.2,
        "available_gb": 8.5
      }
    }
  ]
}
```

## Common Queries

### PromQL (Metrics)

```promql
# Error rate
rate(gl_errors_total[5m]) / rate(gl_pipeline_runs_total[5m])

# P95 latency
histogram_quantile(0.95, rate(gl_api_latency_seconds_bucket[5m]))

# Cache hit rate
rate(gl_cache_hits_total[5m]) / (rate(gl_cache_hits_total[5m]) + rate(gl_cache_misses_total[5m]))

# CPU usage
avg(gl_cpu_usage_percent)
```

### LogQL (Logs)

```logql
# All errors
{job="greenlang"} | json | level="ERROR"

# Specific tenant
{job="greenlang"} | json | tenant_id="customer1"

# Search pattern
{job="greenlang"} | json | message =~ ".*timeout.*"

# Count by component
sum by (component) (count_over_time({job="greenlang"} | json [5m]))
```

## Testing Your Setup

### 1. Generate Test Metrics

```python
from greenlang.observability import get_metrics_collector

collector = get_metrics_collector()

# Generate some test data
for i in range(100):
    collector.collect_system_metrics()
    time.sleep(0.1)
```

### 2. Check Metrics in Prometheus

1. Open http://localhost:9090
2. Go to Graph tab
3. Query: `gl_cpu_usage_percent`
4. Click "Execute"
5. See your metrics!

### 3. Generate Test Logs

```python
from greenlang.observability import get_logger, LogContext

logger = get_logger("test", LogContext(component="test"))

logger.info("Test log entry", test_value=123)
logger.warning("Test warning", alert_level="medium")
logger.error("Test error", error_code="E001")
```

### 4. Create Test Trace

```python
from greenlang.observability import trace_operation
import time

@trace_operation("test_operation")
def test_function():
    time.sleep(0.1)
    return "done"

# Execute 10 times
for i in range(10):
    test_function()
```

Then check Jaeger at http://localhost:16686

## Next Steps

1. **Read the Guides**:
   - [Monitoring Guide](monitoring-guide.md) - Full monitoring setup
   - [Metrics Reference](metrics-reference.md) - All available metrics
   - [Logging Guide](logging-guide.md) - Structured logging best practices
   - [Tracing Guide](tracing-guide.md) - Distributed tracing setup
   - [Alerting Guide](alerting-guide.md) - Alert configuration

2. **Customize Dashboards**:
   - Add your custom metrics
   - Create team-specific views
   - Set up alert thresholds

3. **Configure Alerts**:
   - Edit `observability/alerting-rules.yml`
   - Set up notification channels
   - Define SLOs/SLAs

4. **Integrate with Your Infrastructure**:
   - Add to Kubernetes deployments
   - Configure for production
   - Set up long-term storage

## Performance Impact

All observability features are designed for minimal overhead:

| Feature | CPU Overhead | Memory Overhead |
|---------|--------------|-----------------|
| Metrics Collection | < 1% | ~ 50MB |
| Structured Logging | < 2% | ~ 20MB |
| Distributed Tracing (100% sampling) | < 3% | ~ 30MB |
| Health Checks | < 0.1% | ~ 5MB |
| **Total** | **< 5%** | **~ 100MB** |

## Troubleshooting

**Metrics not appearing?**
- Check http://localhost:8000/metrics works
- Verify Prometheus scrape targets at http://localhost:9090/targets

**Can't access Grafana?**
- Check Docker containers: `docker-compose ps`
- Restart if needed: `docker-compose restart grafana`

**Traces not showing?**
- Verify Jaeger is running: http://localhost:16686
- Check tracing is initialized in your app

## Support

- Documentation: `docs/observability/`
- Sample Code: `examples/observability/`
- Tests: `tests/observability/`

---

**You're all set!** Your GreenLang application now has production-grade observability. 🎉
