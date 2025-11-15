# GreenLang Agent Foundation - Observability & Monitoring

## Overview

Comprehensive observability infrastructure for production-grade agent systems with OpenTelemetry integration, Prometheus metrics, distributed tracing, and multi-stakeholder Grafana dashboards.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    GreenLang Agents                          │
│  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐       │
│  │ Agent 1 │  │ Agent 2 │  │ Agent 3 │  │ Agent N │       │
│  └────┬────┘  └────┬────┘  └────┬────┘  └────┬────┘       │
│       │            │            │            │              │
│       └────────────┴────────────┴────────────┘              │
│                    │                                         │
└────────────────────┼─────────────────────────────────────────┘
                     │
        ┌────────────┴────────────┐
        │                         │
        ▼                         ▼
┌───────────────┐         ┌──────────────┐
│ OpenTelemetry │         │  Prometheus  │
│   Collector   │         │   Metrics    │
└───────┬───────┘         └──────┬───────┘
        │                        │
        │                        │
┌───────▼───────────────────────▼────────┐
│         Observability Stack            │
├────────────────────────────────────────┤
│ • Jaeger (Distributed Tracing)        │
│ • Elasticsearch (Log Aggregation)     │
│ • Grafana (Visualization)             │
│ • AlertManager (Alerting)             │
└────────────────────────────────────────┘
```

## Components

### 1. Structured Logging (`observability/logging.py`)

**Features:**
- JSON-structured logs with ISO 8601 timestamps
- OpenTelemetry trace context injection
- Multiple outputs: console, file, Elasticsearch, CloudWatch
- Thread-local correlation ID tracking
- Performance metrics in log entries
- Log level filtering (DEBUG, INFO, WARN, ERROR, FATAL)

**Usage:**
```python
from observability import setup_logging, LogContext

# Initialize logger
logger = setup_logging(
    app_name="greenlang",
    level="INFO",
    outputs=["console", "elasticsearch"],
    elasticsearch_config={
        "hosts": ["localhost:9200"],
        "index": "greenlang-logs"
    }
)

# Set correlation context
LogContext.set_correlation_id("req-12345")
LogContext.set_agent_id("agent-abc")

# Log with context
logger.info("Agent started", context={"version": "1.0.0"})
logger.error("Task failed", error=exception, performance={"duration_ms": 234})
```

### 2. Distributed Tracing (`observability/tracing.py`)

**Features:**
- OpenTelemetry standard implementation
- Multiple backends: Jaeger, DataDog, New Relic
- Adaptive sampling (1% base, 100% errors, 10% slow requests)
- Automatic trace context propagation
- Agent lifecycle tracing
- LLM call instrumentation

**Usage:**
```python
from observability import setup_tracing

# Initialize tracing
tracer = setup_tracing(
    service_name="greenlang-agents",
    backends=["jaeger"],
    jaeger_config={
        "host": "localhost",
        "port": 6831
    }
)

# Trace agent operations
with tracer.trace_agent(agent_id="agent-123", operation="execute_task"):
    # ... agent work ...
    pass

# Trace LLM calls
with tracer.trace_llm_call(
    model="gpt-4",
    provider="openai",
    tokens_input=100,
    tokens_output=50,
    cost=0.003
):
    # ... LLM API call ...
    pass
```

### 3. Prometheus Metrics (`observability/metrics.py`)

**Metrics Categories:**

**Application Metrics:**
- `greenlang_agents_agent_count` - Active agent count by type/state
- `greenlang_agents_messages_processed` - Message processing rate
- `greenlang_agents_task_completion_time` - Task latency histogram
- `greenlang_agents_error_rate` - Error rate by type
- `greenlang_agents_memory_usage_bytes` - Memory consumption
- `greenlang_agents_cpu_utilization` - CPU usage percentage

**Business Metrics:**
- `greenlang_agents_calculations_performed` - Calculations by type
- `greenlang_agents_reports_generated` - Reports by format
- `greenlang_agents_compliance_checks` - Compliance validation count
- `greenlang_agents_api_calls` - External API usage
- `greenlang_agents_cache_hit_rate` - Cache efficiency

**Infrastructure Metrics:**
- `greenlang_agents_pod_count` - Kubernetes pod status
- `greenlang_agents_database_connections` - DB pool usage
- `greenlang_agents_queue_depth` - Message queue backlog

**Quality Metrics (12 Dimensions):**
- `greenlang_agents_quality_functional_quality`
- `greenlang_agents_quality_performance_efficiency`
- `greenlang_agents_quality_reliability`
- `greenlang_agents_quality_security`
- ... (and 8 more dimensions)

**Usage:**
```python
from observability import setup_metrics, MetricsCollector

# Initialize metrics
collector = setup_metrics(
    namespace="greenlang",
    subsystem="agents",
    http_port=9090  # Prometheus scrape endpoint
)

# Record metrics
collector.record_agent_metrics(
    agent_id="agent-123",
    agent_type="calculator",
    state="running",
    memory_mb=256,
    cpu_percent=45
)

collector.record_task_completion(
    agent_type="calculator",
    task_type="emissions_calculation",
    duration_seconds=1.23,
    success=True
)
```

### 4. Performance Monitoring (`observability/performance_monitor.py`)

**Features:**
- Real-time latency tracking (P50, P95, P99)
- Error rate monitoring with type classification
- Throughput analysis
- SLA compliance tracking (99.99% availability target)
- Resource utilization monitoring
- Historical metrics retention

**SLA Targets:**
- Availability: 99.99% (four nines)
- Latency P50: 100ms
- Latency P95: 500ms
- Latency P99: 2000ms
- Error Rate: <0.1%

**Usage:**
```python
from observability import PerformanceMonitor

monitor = PerformanceMonitor(name="greenlang_agents")
monitor.start_monitoring(interval_seconds=60)

# Record request
monitor.record_request(
    latency_ms=234,
    success=True,
    operation="calculate_emissions"
)

# Check SLA compliance
sla_status = monitor.check_sla_compliance()
for status in sla_status:
    print(f"{status.metric}: {status.compliant}")

# Generate report
report = monitor.get_performance_report()
```

### 5. Grafana Dashboards (`observability/dashboards.py`)

**Five Stakeholder-Specific Dashboards:**

**Executive Dashboard** (`executive.json`)
- Refresh: 1 minute
- Audience: C-level executives
- Panels:
  - System availability gauge
  - Active agent count
  - Daily throughput
  - Cost trends
  - Business impact (reports generated)
  - Compliance status
  - Quality score

**Operations Dashboard** (`operations.json`)
- Refresh: 10 seconds
- Audience: Operations team
- Panels:
  - Service health matrix
  - Request latency (P50/P95/P99)
  - Error rate by type
  - Throughput trends
  - Resource utilization
  - Active alerts table
  - SLA compliance

**Agent Performance Dashboard** (`agents.json`)
- Refresh: 30 seconds
- Audience: Development team
- Panels:
  - Agent state distribution
  - Inter-agent communication flow
  - Task success rate
  - LLM API usage
  - Token usage and costs
  - Memory usage heatmap
  - Task completion time distribution

**Quality Dashboard** (`quality.json`)
- Refresh: 1 hour
- Audience: Quality team
- Panels:
  - Overall quality score gauge
  - 12-dimension quality framework
  - Test coverage (unit/integration/E2E)
  - Code quality trends
  - Security vulnerabilities
  - Technical debt
  - Compliance status

**Financial Dashboard** (`financial.json`)
- Refresh: 1 hour
- Audience: Finance team
- Panels:
  - Monthly cost breakdown
  - Cost trends
  - Cost per agent/report
  - API efficiency (cache hit rate)
  - Budget vs actual
  - Cost optimization opportunities
  - ROI metrics

### 6. Debugging Tools (`observability/debugging.py`)

**Features:**
- Comprehensive health checks (database, cache, LLM, disk, memory)
- CPU profiling with flame graph generation
- Memory profiling
- Log pattern analysis and anomaly detection
- System diagnostics
- Automated troubleshooting workflows

**Usage:**
```python
from observability import DebugTools, HealthChecker

# Run health checks
health_checker = HealthChecker()
results = await health_checker.run_checks()
print(health_checker.to_json())

# Profile performance
debug_tools = DebugTools()
with debug_tools.profiler.profile_cpu("api_handler"):
    # ... code to profile ...
    pass

# Analyze logs
log_analyzer = debug_tools.log_analyzer
analysis = log_analyzer.analyze_logs(time_range=timedelta(hours=1))
anomalies = log_analyzer.detect_anomalies()

# Run diagnostics
diagnostics = await debug_tools.run_diagnostics()
```

## Prometheus Configuration

**File:** `prometheus.yaml`

**Key Features:**
- 15-second scrape interval
- Kubernetes service discovery
- Node exporter integration
- Database and cache exporters
- Remote write for long-term storage
- Recording and alerting rules

**Scrape Targets:**
- GreenLang agents (port 9090)
- PostgreSQL (postgres-exporter:9187)
- Redis (redis-exporter:9121)
- Kubernetes API server
- Node exporter

## Alerting Rules

**File:** `alerting_rules.yaml`

**Alert Categories:**

**Agent Alerts:**
- HighErrorRate (>1% for 5min) - CRITICAL
- AgentFailures (>5 failures in 5min) - WARNING
- PotentialMemoryLeak - WARNING

**Performance Alerts:**
- HighLatency (P95 >500ms for 10min) - WARNING
- LowThroughput (<100 msg/s for 15min) - WARNING
- SLABreach - CRITICAL

**Resource Alerts:**
- HighCPUUsage (>80% for 10min) - WARNING
- DatabaseConnectionPoolExhausted (>90% for 5min) - CRITICAL
- LowCacheHitRate (<70% for 15min) - WARNING

**Business Alerts:**
- NoReportsGenerated (1 hour) - WARNING
- HighAPICosts (>$100/hour) - WARNING
- ComplianceCheckFailures (>10/hour) - CRITICAL

**Quality Alerts:**
- QualityScoreDrop (<85% for 30min) - WARNING
- TestCoverageDrop (<80% for 1hour) - WARNING

## Deployment

### Local Development (Docker Compose)

```bash
cd deployment/monitoring
docker-compose up -d
```

**Services:**
- Prometheus: http://localhost:9090
- Grafana: http://localhost:3000 (admin/admin)
- Jaeger UI: http://localhost:16686
- AlertManager: http://localhost:9093

### Kubernetes Production

```bash
# Create namespace
kubectl create namespace greenlang-monitoring

# Deploy Prometheus
kubectl apply -f k8s/prometheus.yaml

# Deploy Grafana
kubectl apply -f k8s/grafana.yaml

# Import dashboards
kubectl create configmap grafana-dashboards \
  --from-file=grafana_dashboards/ \
  -n greenlang-monitoring
```

## Integration Example

```python
from observability import (
    setup_logging,
    setup_tracing,
    setup_metrics,
    PerformanceMonitor,
    DebugTools
)

# Initialize observability stack
logger = setup_logging(
    app_name="greenlang",
    level="INFO",
    outputs=["console", "elasticsearch"]
)

tracer = setup_tracing(
    service_name="greenlang-agents",
    backends=["jaeger"]
)

metrics = setup_metrics(
    namespace="greenlang",
    subsystem="agents",
    http_port=9090
)

monitor = PerformanceMonitor()
monitor.start_monitoring()

# Use in agent
class MyAgent:
    def execute_task(self, task):
        # Set correlation context
        from observability import LogContext
        LogContext.set_agent_id(self.id)
        LogContext.set_correlation_id(task.correlation_id)

        # Trace execution
        with tracer.trace_agent(self.id, "execute_task"):
            start = time.time()

            try:
                # Log start
                logger.info(f"Executing task: {task.type}")

                # Do work
                result = self._process_task(task)

                # Record metrics
                duration = time.time() - start
                metrics.record_task_completion(
                    agent_type=self.type,
                    task_type=task.type,
                    duration_seconds=duration,
                    success=True
                )

                monitor.record_request(
                    latency_ms=duration * 1000,
                    success=True,
                    operation=task.type
                )

                logger.info(
                    f"Task completed: {task.type}",
                    performance={"duration_seconds": duration}
                )

                return result

            except Exception as e:
                # Log error
                logger.error(f"Task failed: {task.type}", error=e)

                # Record failure
                metrics.record_task_completion(
                    agent_type=self.type,
                    task_type=task.type,
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

## Monitoring Best Practices

1. **Always set correlation IDs** for request tracking across services
2. **Use structured logging** - avoid string formatting in log messages
3. **Record all critical operations** with metrics
4. **Sample traces adaptively** - 100% for errors, lower for success
5. **Set appropriate SLO targets** based on business requirements
6. **Create actionable alerts** - avoid alert fatigue
7. **Document runbooks** for each alert type
8. **Review dashboards regularly** with stakeholders
9. **Analyze trends** for capacity planning
10. **Test observability** in staging before production

## Troubleshooting

### No metrics appearing in Prometheus
- Check agent is exposing metrics on port 9090: `curl http://localhost:9090/metrics`
- Verify Prometheus scrape config includes your job
- Check Prometheus targets page: http://localhost:9090/targets

### Missing traces in Jaeger
- Verify JAEGER_AGENT_HOST environment variable
- Check sampling rate isn't too low
- Ensure spans are being flushed before process exit

### High cardinality metrics
- Avoid using high-cardinality labels (UUIDs, timestamps)
- Use aggregation for user/request IDs
- Limit label value count (<100 per metric)

### Dashboard not loading
- Check Prometheus datasource connection in Grafana
- Verify metric names match dashboard queries
- Check time range is appropriate

## Support

For issues or questions:
- Documentation: https://docs.greenlang.io/observability
- Slack: #agent-platform-monitoring
- Email: platform-team@greenlang.io