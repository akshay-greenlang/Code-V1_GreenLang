# GreenLang Observability - Quick Start Guide

## 5-Minute Setup

### Step 1: Install Dependencies

```bash
pip install -r requirements.txt
```

**requirements.txt:**
```
opentelemetry-api>=1.21.0
opentelemetry-sdk>=1.21.0
opentelemetry-instrumentation-requests>=0.42b0
opentelemetry-exporter-jaeger>=1.21.0
prometheus-client>=0.19.0
elasticsearch>=8.11.0
boto3>=1.34.0
psutil>=5.9.0
```

### Step 2: Start Monitoring Stack

```bash
cd deployment/monitoring
docker-compose up -d
```

Wait 30 seconds for services to start, then verify:
- Prometheus: http://localhost:9090
- Grafana: http://localhost:3000 (login: admin/admin)
- Jaeger: http://localhost:16686

### Step 3: Initialize Observability in Your Agent

```python
from observability import (
    setup_logging,
    setup_tracing,
    setup_metrics,
    PerformanceMonitor
)

# One-time setup at application start
logger = setup_logging(
    app_name="greenlang",
    level="INFO",
    outputs=["console", "elasticsearch"],
    elasticsearch_config={"hosts": ["localhost:9200"]}
)

tracer = setup_tracing(
    service_name="greenlang-agents",
    backends=["jaeger"],
    jaeger_config={"host": "localhost", "port": 6831}
)

metrics = setup_metrics(
    namespace="greenlang",
    subsystem="agents",
    http_port=9090
)

monitor = PerformanceMonitor()
monitor.start_monitoring()
```

### Step 4: Instrument Your Agent

```python
from observability import LogContext

class MyAgent:
    def __init__(self, agent_id):
        self.id = agent_id
        self.type = "calculator"

    def execute_task(self, task):
        # Set context for correlation
        LogContext.set_agent_id(self.id)
        LogContext.set_correlation_id(task.correlation_id)

        # Log start
        logger.info(f"Starting task: {task.name}")

        # Trace execution
        with tracer.trace_agent(self.id, "execute_task"):
            start = time.time()

            try:
                # Your agent logic here
                result = self._do_work(task)

                # Record success metrics
                duration = time.time() - start
                metrics.record_task_completion(
                    agent_type=self.type,
                    task_type=task.name,
                    duration_seconds=duration,
                    success=True
                )

                monitor.record_request(
                    latency_ms=duration * 1000,
                    success=True,
                    operation=task.name
                )

                logger.info(
                    f"Task completed: {task.name}",
                    performance={"duration_seconds": duration}
                )

                return result

            except Exception as e:
                # Record failure
                logger.error(f"Task failed: {task.name}", error=e)

                metrics.record_task_completion(
                    agent_type=self.type,
                    task_type=task.name,
                    duration_seconds=time.time() - start,
                    success=False
                )

                monitor.record_request(
                    latency_ms=(time.time() - start) * 1000,
                    success=False,
                    error_type=type(e).__name__
                )

                raise
```

### Step 5: View Dashboards

1. Open Grafana: http://localhost:3000
2. Login with admin/admin
3. Navigate to Dashboards
4. Import dashboards from `deployment/monitoring/grafana_dashboards/`:
   - executive.json - For business metrics
   - operations.json - For system health
   - agents.json - For agent performance
   - quality.json - For code quality
   - financial.json - For cost tracking

## Common Use Cases

### 1. Track LLM API Calls

```python
with tracer.trace_llm_call(
    model="gpt-4",
    provider="openai",
    tokens_input=100,
    tokens_output=50,
    cost=0.003
):
    response = openai.ChatCompletion.create(...)

metrics.record_llm_call(
    model="gpt-4",
    tokens_input=100,
    tokens_output=50,
    duration_seconds=1.2,
    cost=0.003
)
```

### 2. Monitor Database Operations

```python
with tracer.trace_database("SELECT", "emissions", "postgresql"):
    results = db.query("SELECT * FROM emissions WHERE...")

metrics.set_gauge(
    "database_connections",
    db.pool.size,
    {"database": "greenlang", "status": "active"}
)
```

### 3. Check System Health

```python
from observability import DebugTools

debug_tools = DebugTools()
results = await debug_tools.health_checker.run_checks()

# Get JSON output
health_json = debug_tools.health_checker.to_json()

# Check overall status
status = debug_tools.health_checker.get_overall_status()
# Returns: HealthStatus.HEALTHY, DEGRADED, or UNHEALTHY
```

### 4. Profile Performance Bottlenecks

```python
from observability import Profiler

profiler = Profiler()

# Profile CPU
with profiler.profile_cpu("expensive_calculation"):
    result = expensive_calculation()

# Get results
profile = profiler.get_profile("expensive_calculation")
print(f"Top functions: {profile.top_functions}")

# Profile memory
with profiler.profile_memory("data_processing"):
    process_large_dataset()
```

### 5. Analyze Logs for Errors

```python
from observability import LogAnalyzer
from datetime import timedelta

analyzer = LogAnalyzer()

# Analyze last hour
analysis = analyzer.analyze_logs(time_range=timedelta(hours=1))

print(f"Total errors: {analysis['statistics']['error_count']}")
print(f"Error rate: {analysis['statistics']['error_rate']:.2%}")
print(f"Top errors: {analysis['statistics']['top_errors']}")

# Detect anomalies
anomalies = analyzer.detect_anomalies()
for anomaly in anomalies:
    print(f"{anomaly['type']}: {anomaly['message']}")
```

### 6. Record Custom Metrics

```python
# Counter - for things that only increase
metrics.increment_counter(
    "calculations_performed",
    labels={"calculation_type": "scope3", "scope": "upstream"}
)

# Gauge - for values that go up and down
metrics.set_gauge(
    "active_agents",
    len(agent_pool),
    labels={"type": "calculator", "state": "running"}
)

# Histogram - for distributions (latency, size, etc.)
metrics.observe_histogram(
    "task_completion_time",
    duration_seconds,
    labels={"agent_type": "calculator", "task_type": "emissions"}
)
```

### 7. Create Custom Alerts

Add to `deployment/monitoring/alerting_rules.yaml`:

```yaml
- alert: MyCustomAlert
  expr: |
    greenlang_agents_my_metric > 100
  for: 5m
  labels:
    severity: warning
    team: my-team
  annotations:
    summary: "My metric is too high"
    description: "Current value: {{ $value }}"
```

## Testing Observability

### Generate Test Traffic

```python
import random
import time

# Simulate agent activity
for i in range(100):
    agent = MyAgent(f"agent-{i}")

    # Random success/failure
    success = random.random() > 0.1  # 90% success rate

    # Random latency
    latency_ms = random.gauss(200, 50)  # 200ms mean, 50ms std dev

    monitor.record_request(
        latency_ms=latency_ms,
        success=success,
        operation="test_task"
    )

    metrics.increment_counter(
        "messages_processed",
        labels={"agent_type": "test", "message_type": "task"}
    )

    time.sleep(0.1)

# Check SLA compliance
sla_status = monitor.check_sla_compliance()
for status in sla_status:
    print(f"{status.metric}: {'✓' if status.compliant else '✗'} (target: {status.target}, current: {status.current})")
```

### View Test Results

1. Prometheus: http://localhost:9090/graph
   - Query: `greenlang_agents_messages_processed`
   - See your test metrics

2. Jaeger: http://localhost:16686
   - Search for traces
   - View trace details

3. Grafana: http://localhost:3000
   - Operations dashboard shows real-time metrics
   - Agent dashboard shows agent activity

## Troubleshooting

### Metrics not appearing in Prometheus

```bash
# Check if metrics endpoint is working
curl http://localhost:9090/metrics | grep greenlang

# Check Prometheus targets
# Visit: http://localhost:9090/targets
# Ensure your job is listed and status is UP
```

### No traces in Jaeger

```python
# Enable debug logging
import logging
logging.basicConfig(level=logging.DEBUG)

# Check Jaeger agent connectivity
# Ensure JAEGER_AGENT_HOST environment variable is set
```

### Elasticsearch connection failed

```bash
# Check Elasticsearch is running
curl http://localhost:9200/_cluster/health

# Check logs
docker logs greenlang-elasticsearch
```

### Dashboard showing "No Data"

1. Check time range (top-right in Grafana)
2. Verify Prometheus datasource: Configuration > Data Sources
3. Test metric query in Prometheus first
4. Ensure metrics have been generated (run test traffic)

## Best Practices

1. **Always set correlation IDs** for request tracking
2. **Use structured logging** - avoid string formatting
3. **Keep label cardinality low** (<100 values per label)
4. **Sample traces wisely** - 100% for errors, lower for success
5. **Create actionable alerts** - link to runbooks
6. **Review dashboards weekly** with your team
7. **Monitor the monitors** - check observability stack health
8. **Test in staging first** before production deployment

## Next Steps

1. Customize dashboards for your specific metrics
2. Set up alert notifications (PagerDuty, Slack)
3. Configure long-term storage (Thanos, Cortex)
4. Implement security (authentication, TLS)
5. Create runbooks for each alert
6. Train team on observability tools

## Resources

- Full Documentation: `deployment/monitoring/README.md`
- Architecture Spec: `Agent_Foundation_Architecture.md` (lines 983-1354)
- Dashboard Examples: `deployment/monitoring/grafana_dashboards/`
- Alert Rules: `deployment/monitoring/alerting_rules.yaml`

## Support

Questions? Check:
1. README.md for detailed documentation
2. Inline code comments
3. Example integration in OBSERVABILITY_SUMMARY.md
4. Create GitHub issue for bugs

---

**Happy Monitoring!**