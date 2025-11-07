# GreenLang Monitoring Guide

## Overview

GreenLang provides comprehensive observability infrastructure including metrics, logging, tracing, and health checks. This guide covers how to effectively monitor your GreenLang deployment.

## Architecture

The GreenLang observability stack consists of:

- **Prometheus**: Metrics collection and storage
- **Grafana**: Visualization and dashboards
- **Loki**: Log aggregation
- **Jaeger**: Distributed tracing
- **Alertmanager**: Alert management and routing

## Quick Start

### 1. Start the Monitoring Stack

```bash
cd observability
docker-compose -f docker-compose-monitoring.yml up -d
```

### 2. Access Dashboards

- **Grafana**: http://localhost:3000 (admin/greenlang2024)
- **Prometheus**: http://localhost:9090
- **Jaeger**: http://localhost:16686
- **Alertmanager**: http://localhost:9093

### 3. Configure GreenLang Application

```python
from greenlang.observability import (
    get_metrics_collector,
    configure_logging,
    get_tracing_manager,
    get_health_checker,
)

# Initialize observability
configure_logging(level="INFO", format_json=True)
collector = get_metrics_collector()
collector.start_collection(port=8000)

# Start tracing
tracing_manager = get_tracing_manager()

# Start health checks
health_checker = get_health_checker()
health_checker.start_background_checks()
```

## Metrics Collection

### Available Metrics

#### Pipeline Metrics
- `gl_pipeline_runs_total`: Total pipeline executions (counter)
- `gl_pipeline_duration_seconds`: Pipeline execution time (histogram)
- `gl_active_executions`: Currently running pipelines (gauge)

#### API Metrics
- `gl_api_requests_total`: Total API requests (counter)
- `gl_api_latency_seconds`: API response time (histogram)

#### Resource Metrics
- `gl_cpu_usage_percent`: CPU usage percentage (gauge)
- `gl_memory_usage_bytes`: Memory usage in bytes (gauge)
- `gl_disk_usage_bytes`: Disk usage in bytes (gauge)

#### Cache Metrics
- `gl_cache_hits_total`: Cache hits (counter)
- `gl_cache_misses_total`: Cache misses (counter)

#### Database Metrics
- `gl_db_queries_total`: Database queries (counter)
- `gl_db_query_duration_seconds`: Query execution time (histogram)
- `gl_db_connections`: Active database connections (gauge)

### Custom Metrics

Add custom metrics to your application:

```python
from greenlang.observability import MetricsCollector, MetricType, CustomMetric

collector = get_metrics_collector()

# Register custom metric
custom_metric = CustomMetric(
    name="my_custom_metric",
    type=MetricType.HISTOGRAM,
    description="Custom operation duration",
    labels=["operation_type"],
    buckets=[0.1, 0.5, 1.0, 5.0],
)
collector.register_custom_metric(custom_metric)

# Record values
collector.record_metric(
    "my_custom_metric",
    value=0.25,
    labels={"operation_type": "process"}
)
```

### Using Decorators

Track execution metrics automatically:

```python
from greenlang.observability import track_execution

@track_execution(pipeline="data_processing", tenant_id="customer1")
def process_data(data):
    # Your processing logic
    return processed_data
```

## Grafana Dashboards

### Pre-built Dashboards

GreenLang includes 4 pre-configured dashboards:

1. **System Overview** (`system-overview.json`)
   - Request rate and latency
   - CPU and memory usage
   - Active executions
   - Error rates
   - Cache hit rates

2. **Agent Performance** (`agent-performance.json`)
   - Agent execution rates
   - Success vs failure rates
   - Duration percentiles
   - Agent-specific errors

3. **API Metrics** (`api-metrics.json`)
   - Request rates by endpoint
   - Latency percentiles
   - HTTP status codes
   - Error rates (4xx, 5xx)

4. **Errors & Alerts** (`errors-alerts.json`)
   - Error rates by component
   - Error distribution
   - Recent errors
   - Active alerts

### Importing Dashboards

Dashboards are automatically provisioned when using the Docker Compose setup. To manually import:

1. Open Grafana (http://localhost:3000)
2. Navigate to Dashboards → Import
3. Upload dashboard JSON file
4. Select Prometheus as data source

### Creating Custom Dashboards

Use PromQL queries in Grafana:

```promql
# Request rate
rate(gl_api_requests_total[5m])

# P95 latency
histogram_quantile(0.95, rate(gl_api_latency_seconds_bucket[5m]))

# Error rate
rate(gl_errors_total[5m]) / rate(gl_pipeline_runs_total[5m])

# Memory usage
gl_memory_usage_bytes / 1024 / 1024 / 1024
```

## Alerting

### Alert Rules

Alerts are defined in `alerting-rules.yml`. Key alerts include:

- **HighErrorRate**: Error rate > 5%
- **CriticalErrorRate**: Error rate > 10%
- **SlowPipelineExecution**: P95 latency > 60s
- **HighCPUUsage**: CPU > 80%
- **LowDiskSpace**: Disk usage > 85%
- **ServiceDown**: Service unreachable

### Alert Routing

Alerts are routed based on severity:

- **Critical**: Immediate notification (PagerDuty, Slack, Email)
- **Warning**: Grouped notifications every 4 hours
- **Info**: Daily digest

### Configuring Notifications

Edit `alertmanager-config.yml`:

```yaml
receivers:
  - name: 'critical-alerts'
    slack_configs:
      - api_url: 'YOUR_SLACK_WEBHOOK_URL'
        channel: '#alerts-critical'
    email_configs:
      - to: 'ops-team@example.com'
    pagerduty_configs:
      - service_key: 'YOUR_PAGERDUTY_KEY'
```

## Health Checks

### Kubernetes Probes

GreenLang provides three health check endpoints:

```python
from greenlang.observability import check_health, CheckType

# Liveness probe
liveness = check_health(CheckType.LIVENESS)
# Returns 200 if service is alive

# Readiness probe
readiness = check_health(CheckType.READINESS)
# Returns 200 if service can accept traffic

# Startup probe
startup = check_health(CheckType.STARTUP)
# Returns 200 if initialization is complete
```

### Kubernetes Configuration

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: greenlang
spec:
  containers:
  - name: greenlang
    image: greenlang:latest
    livenessProbe:
      httpGet:
        path: /health/live
        port: 8000
      initialDelaySeconds: 10
      periodSeconds: 30
    readinessProbe:
      httpGet:
        path: /health/ready
        port: 8000
      initialDelaySeconds: 5
      periodSeconds: 10
    startupProbe:
      httpGet:
        path: /health/startup
        port: 8000
      failureThreshold: 30
      periodSeconds: 10
```

## Troubleshooting

### Common Issues

#### 1. Metrics Not Appearing

**Problem**: Metrics not showing in Prometheus

**Solutions**:
- Check Prometheus scrape targets: http://localhost:9090/targets
- Verify GreenLang metrics endpoint: http://localhost:8000/metrics
- Check network connectivity between services

#### 2. High Memory Usage

**Problem**: Prometheus consuming too much memory

**Solutions**:
- Reduce retention period in `prometheus.yml`
- Decrease scrape interval
- Enable remote storage

#### 3. Missing Logs

**Problem**: Logs not appearing in Loki

**Solutions**:
- Check Promtail configuration
- Verify log file paths
- Check Loki endpoint connectivity

### Debug Mode

Enable debug logging for troubleshooting:

```python
from greenlang.observability import configure_logging

configure_logging(level="DEBUG", format_json=True, log_file="/var/log/greenlang/debug.log")
```

## Best Practices

### 1. Metric Naming

Use consistent metric names:
- Use underscores, not hyphens
- Include units in metric names (e.g., `_seconds`, `_bytes`)
- Use base units (seconds, not milliseconds)

### 2. Label Cardinality

Avoid high-cardinality labels:
- ❌ Don't use: `user_id`, `email`, `ip_address`
- ✅ Use: `tenant_id`, `region`, `environment`

### 3. Dashboard Organization

- Group related metrics together
- Use consistent time ranges
- Add annotations for deployments
- Set appropriate refresh intervals

### 4. Alert Fatigue

Prevent alert fatigue:
- Set appropriate thresholds
- Use alert inhibition rules
- Group related alerts
- Include runbooks in annotations

## Performance Impact

The observability infrastructure is designed for minimal overhead:

- **Metrics Collection**: < 1% CPU overhead
- **Structured Logging**: < 2% CPU overhead
- **Distributed Tracing**: Configurable sampling (default: 100%)
- **Health Checks**: < 0.1% CPU overhead

### Optimizing Performance

1. **Adjust Sampling Rates**
```python
from greenlang.observability import TraceConfig

config = TraceConfig(sampling_rate=0.1)  # Sample 10% of traces
```

2. **Reduce Metric Collection Frequency**
```python
collector = get_metrics_collector()
collector.collection_interval = 120  # Collect every 2 minutes
```

3. **Limit Log Retention**
```python
from greenlang.observability import LogAggregator

aggregator = LogAggregator(max_logs=1000)  # Keep last 1000 logs
```

## Next Steps

- Read the [Metrics Reference](metrics-reference.md) for all available metrics
- Learn about [Distributed Tracing](tracing-guide.md)
- Configure [Alerting Rules](alerting-guide.md)
- Implement [Custom Health Checks](monitoring-guide.md#custom-health-checks)
