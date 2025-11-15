# Observability Guide

## Monitoring, Logging, and Debugging Agents

Comprehensive guide to observability for GreenLang agents.

---

## Structured Logging

### Implementation

```python
import structlog

# Configure structured logging
structlog.configure(
    processors=[
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.JSONRenderer()
    ]
)

class ObservableAgent(BaseAgent):
    """Agent with structured logging."""

    def __init__(self, config: AgentConfig):
        super().__init__(config)
        self.logger = structlog.get_logger(
            agent_id=self.id,
            agent_name=self.name,
            version=self.version
        )

    async def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Process with logging."""
        self.logger.info(
            "processing_started",
            input_size=len(data),
            timestamp=datetime.utcnow()
        )

        try:
            result = await self._process(data)

            self.logger.info(
                "processing_completed",
                duration_ms=self.get_duration(),
                output_size=len(result)
            )

            return result

        except Exception as e:
            self.logger.error(
                "processing_failed",
                error=str(e),
                error_type=type(e).__name__,
                traceback=traceback.format_exc()
            )
            raise
```

---

## Metrics Collection

### Prometheus Metrics

```python
from prometheus_client import Counter, Histogram, Gauge, Summary

class InstrumentedAgent(BaseAgent):
    """Agent with Prometheus metrics."""

    # Define metrics
    requests_total = Counter(
        'agent_requests_total',
        'Total requests processed',
        ['agent_name', 'status']
    )

    request_duration = Histogram(
        'agent_request_duration_seconds',
        'Request duration in seconds',
        ['agent_name']
    )

    active_requests = Gauge(
        'agent_active_requests',
        'Number of active requests',
        ['agent_name']
    )

    memory_usage = Gauge(
        'agent_memory_bytes',
        'Memory usage in bytes',
        ['agent_name']
    )

    async def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Process with metrics."""
        self.active_requests.labels(agent_name=self.name).inc()

        try:
            with self.request_duration.labels(agent_name=self.name).time():
                result = await self._process(data)

            self.requests_total.labels(
                agent_name=self.name,
                status='success'
            ).inc()

            return result

        except Exception as e:
            self.requests_total.labels(
                agent_name=self.name,
                status='error'
            ).inc()
            raise

        finally:
            self.active_requests.labels(agent_name=self.name).dec()
            self.update_memory_usage()
```

---

## Distributed Tracing

### OpenTelemetry Integration

```python
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.jaeger import JaegerExporter

# Setup tracer
trace.set_tracer_provider(TracerProvider())
tracer = trace.get_tracer(__name__)

# Configure Jaeger exporter
jaeger_exporter = JaegerExporter(
    agent_host_name="localhost",
    agent_port=6831,
)

trace.get_tracer_provider().add_span_processor(
    BatchSpanProcessor(jaeger_exporter)
)

class TracedAgent(BaseAgent):
    """Agent with distributed tracing."""

    async def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Process with tracing."""
        with tracer.start_as_current_span("agent.process") as span:
            # Add attributes
            span.set_attribute("agent.id", self.id)
            span.set_attribute("agent.name", self.name)
            span.set_attribute("input.size", len(data))

            try:
                result = await self._process(data)
                span.set_attribute("status", "success")
                return result

            except Exception as e:
                span.set_attribute("status", "error")
                span.set_attribute("error.type", type(e).__name__)
                span.set_attribute("error.message", str(e))
                raise

    async def _process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Internal processing with nested spans."""
        # LLM call span
        with tracer.start_as_current_span("llm.generate"):
            response = await self.llm.generate(data['prompt'])

        # Database span
        with tracer.start_as_current_span("database.query"):
            db_result = await self.db.query(data['query'])

        return {'response': response, 'data': db_result}
```

---

## Health Checks

### Comprehensive Health Monitoring

```python
class HealthChecker:
    """Comprehensive health checks."""

    async def check(self) -> Dict[str, Any]:
        """Perform all health checks."""
        checks = {
            'agent': await self.check_agent(),
            'database': await self.check_database(),
            'redis': await self.check_redis(),
            'llm': await self.check_llm(),
            'memory': await self.check_memory()
        }

        overall_healthy = all(c['healthy'] for c in checks.values())

        return {
            'healthy': overall_healthy,
            'timestamp': datetime.utcnow(),
            'checks': checks
        }

    async def check_database(self) -> Dict[str, Any]:
        """Check database connectivity."""
        try:
            start = time.time()
            await self.db.execute("SELECT 1")
            latency = (time.time() - start) * 1000

            return {
                'healthy': True,
                'latency_ms': latency
            }
        except Exception as e:
            return {
                'healthy': False,
                'error': str(e)
            }

    async def check_redis(self) -> Dict[str, Any]:
        """Check Redis connectivity."""
        try:
            start = time.time()
            await self.redis.ping()
            latency = (time.time() - start) * 1000

            return {
                'healthy': True,
                'latency_ms': latency
            }
        except Exception as e:
            return {
                'healthy': False,
                'error': str(e)
            }
```

---

## Dashboards

### Grafana Dashboard Configuration

```json
{
  "dashboard": {
    "title": "GreenLang Agent Monitoring",
    "panels": [
      {
        "title": "Request Rate",
        "targets": [
          {
            "expr": "rate(agent_requests_total[5m])"
          }
        ]
      },
      {
        "title": "Error Rate",
        "targets": [
          {
            "expr": "rate(agent_requests_total{status=\"error\"}[5m])"
          }
        ]
      },
      {
        "title": "Latency (p50, p95, p99)",
        "targets": [
          {
            "expr": "histogram_quantile(0.50, agent_request_duration_seconds)"
          },
          {
            "expr": "histogram_quantile(0.95, agent_request_duration_seconds)"
          },
          {
            "expr": "histogram_quantile(0.99, agent_request_duration_seconds)"
          }
        ]
      },
      {
        "title": "Memory Usage",
        "targets": [
          {
            "expr": "agent_memory_bytes"
          }
        ]
      }
    ]
  }
}
```

---

## Alerting

### Alert Rules

```yaml
groups:
  - name: agent_alerts
    interval: 30s
    rules:
      - alert: HighErrorRate
        expr: rate(agent_requests_total{status="error"}[5m]) > 0.05
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "High error rate detected"
          description: "Error rate is {{ $value }} for agent {{ $labels.agent_name }}"

      - alert: HighLatency
        expr: histogram_quantile(0.95, agent_request_duration_seconds) > 2
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "High latency detected"
          description: "P95 latency is {{ $value }}s"

      - alert: AgentDown
        expr: up{job="greenlang-agents"} == 0
        for: 1m
        labels:
          severity: critical
        annotations:
          summary: "Agent is down"
          description: "Agent {{ $labels.instance }} has been down for 1 minute"
```

---

## Debugging

### Debug Mode

```python
class DebuggableAgent(BaseAgent):
    """Agent with debug capabilities."""

    def __init__(self, config: AgentConfig, debug: bool = False):
        super().__init__(config)
        self.debug = debug

        if debug:
            self.enable_debug_mode()

    def enable_debug_mode(self):
        """Enable comprehensive debugging."""
        # Verbose logging
        self.logger.setLevel(logging.DEBUG)

        # Trace all method calls
        self.trace_calls = True

        # Save intermediate results
        self.save_intermediates = True

        # Disable caching
        self.cache_enabled = False

    async def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Process with debug information."""
        if self.debug:
            debug_info = {
                'input': data,
                'intermediates': [],
                'timings': {}
            }

        result = await self._process(data)

        if self.debug:
            debug_info['output'] = result
            debug_info['duration'] = self.get_duration()
            self.save_debug_info(debug_info)

        return result
```

---

**Last Updated**: November 2024
**Version**: 1.0.0