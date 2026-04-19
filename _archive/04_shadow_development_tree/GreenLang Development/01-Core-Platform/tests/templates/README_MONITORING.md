# GreenLang Agent Operational Monitoring Guide

**Version:** 1.0.0
**Last Updated:** 2025-10-16
**Status:** Production Ready

---

## Table of Contents

1. [Overview](#overview)
2. [Quick Start](#quick-start)
3. [Architecture](#architecture)
4. [Integration Guide](#integration-guide)
5. [Monitoring Features](#monitoring-features)
6. [Health Checks](#health-checks)
7. [Metrics Collection](#metrics-collection)
8. [Alerting System](#alerting-system)
9. [Changelog Management](#changelog-management)
10. [Best Practices](#best-practices)
11. [Troubleshooting](#troubleshooting)
12. [API Reference](#api-reference)
13. [Examples](#examples)

---

## Overview

The GreenLang Operational Monitoring system provides production-grade observability for all AI agents. This system addresses the universal gaps D11 (Operations) and D12 (Improvement) that were blocking production readiness across all 8 agents.

### Key Features

- **Performance Tracking**: Latency, cost, token usage, cache hit rates
- **Health Monitoring**: Liveness, readiness, degradation detection
- **Metrics Export**: Prometheus-compatible metrics
- **Structured Logging**: JSON-formatted logs for analysis
- **Alert Generation**: Configurable alerts for operational issues
- **Change Management**: Standardized changelog templates
- **Zero-Overhead Design**: Minimal performance impact
- **Easy Integration**: Simple mixin pattern

### What Problems Does This Solve?

1. **D11 - Operations Monitoring**: Real-time visibility into agent performance
2. **D12 - Continuous Improvement**: Track changes and version history
3. **Production Readiness**: Meet enterprise operational requirements
4. **Debugging**: Structured logs and metrics for troubleshooting
5. **Cost Management**: Track and alert on AI API costs
6. **SLA Compliance**: Monitor latency and availability

---

## Quick Start

### 5-Minute Integration

**Step 1: Add monitoring to an agent**

```python
# Before
from greenlang.agents.base import BaseAgent, AgentResult

class CarbonAgent(BaseAgent):
    def __init__(self):
        super().__init__()

    def execute(self, input_data):
        # Calculate emissions
        result = self._calculate(input_data)
        return AgentResult(success=True, data=result)

# After
from greenlang.agents.base import BaseAgent, AgentResult
from templates.agent_monitoring import OperationalMonitoringMixin

class CarbonAgent(OperationalMonitoringMixin, BaseAgent):
    def __init__(self):
        super().__init__()
        self.setup_monitoring(agent_name="carbon_agent")

    def execute(self, input_data):
        with self.track_execution(input_data) as tracker:
            # Calculate emissions
            result = self._calculate(input_data)

            # Track metrics
            tracker.set_cost(0.08)
            tracker.set_tokens(2500)

            return AgentResult(success=True, data=result)
```

**Step 2: Use automated integration script**

```bash
# Single agent
python scripts/add_monitoring_and_changelog.py --agent carbon

# All agents
python scripts/add_monitoring_and_changelog.py --all-agents

# Dry run first
python scripts/add_monitoring_and_changelog.py --agent fuel --dry-run
```

**Step 3: Verify integration**

```bash
python scripts/add_monitoring_and_changelog.py --agent carbon --verify-only
```

---

## Architecture

### System Components

```
┌─────────────────────────────────────────────────────────┐
│                    Your Agent                           │
│  ┌───────────────────────────────────────────────────┐  │
│  │  OperationalMonitoringMixin                       │  │
│  │  ┌─────────────┐  ┌─────────────┐  ┌───────────┐ │  │
│  │  │ Performance │  │   Health    │  │  Alerts   │ │  │
│  │  │  Tracking   │  │   Checks    │  │ Generator │ │  │
│  │  └─────────────┘  └─────────────┘  └───────────┘ │  │
│  │  ┌─────────────┐  ┌─────────────┐  ┌───────────┐ │  │
│  │  │   Metrics   │  │  Structured │  │ Execution │ │  │
│  │  │ Collection  │  │   Logging   │  │  History  │ │  │
│  │  └─────────────┘  └─────────────┘  └───────────┘ │  │
│  └───────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────┘
                          │
                          ▼
        ┌─────────────────────────────────────┐
        │      Monitoring Outputs             │
        ├─────────────────────────────────────┤
        │ • Prometheus Metrics                │
        │ • JSON Logs                         │
        │ • Health Check Endpoints            │
        │ • Performance Reports               │
        │ • Alert Notifications               │
        └─────────────────────────────────────┘
```

### Design Principles

1. **Mixin Pattern**: Non-invasive integration via multiple inheritance
2. **Context Managers**: Automatic metric tracking with Python's `with` statement
3. **Thread-Safe**: Safe for concurrent execution
4. **Configurable**: Flexible thresholds and options
5. **Extensible**: Easy to add custom metrics

---

## Integration Guide

### Method 1: Automated Integration (Recommended)

Use the integration script for quick, consistent integration:

```bash
# Integrate monitoring + changelog for one agent
python scripts/add_monitoring_and_changelog.py --agent carbon_agent

# Preview changes without modifying files
python scripts/add_monitoring_and_changelog.py --agent fuel_agent --dry-run

# Integrate all agents at once
python scripts/add_monitoring_and_changelog.py --all-agents

# Custom version for changelog
python scripts/add_monitoring_and_changelog.py --agent boiler_agent --version 1.2.0
```

**What the script does:**

1. Adds `OperationalMonitoringMixin` to imports
2. Updates class inheritance
3. Adds `setup_monitoring()` call in `__init__`
4. Wraps `execute()` method with `track_execution()`
5. Creates agent-specific CHANGELOG.md
6. Creates backup of original file
7. Verifies integration

### Method 2: Manual Integration

For more control or custom scenarios:

**Step 1: Add import**

```python
from templates.agent_monitoring import OperationalMonitoringMixin
```

**Step 2: Update class inheritance**

```python
# Before
class MyAgent(BaseAgent):
    pass

# After
class MyAgent(OperationalMonitoringMixin, BaseAgent):
    pass
```

**Step 3: Initialize monitoring**

```python
def __init__(self, config=None):
    super().__init__(config)

    # Add monitoring
    self.setup_monitoring(
        agent_name="my_agent",
        enable_metrics=True,
        enable_health_checks=True,
        enable_alerting=True,
        max_history=1000
    )
```

**Step 4: Track execution**

```python
def execute(self, input_data):
    with self.track_execution(input_data) as tracker:
        # Your agent logic here
        result = self._do_work(input_data)

        # Update tracker (optional but recommended)
        tracker.set_tokens(result.get('tokens', 0))
        tracker.set_cost(result.get('cost', 0.0))
        tracker.increment_ai_calls(1)

        return AgentResult(success=True, data=result)
```

**Step 5: Create CHANGELOG.md**

Copy `templates/CHANGELOG_TEMPLATE.md` to your agent directory and customize.

---

## Monitoring Features

### Performance Tracking

Every execution automatically tracks:

- **Duration**: Execution time in milliseconds
- **Cost**: AI API costs in USD
- **Tokens**: Token usage (input + output)
- **AI Calls**: Number of LLM API calls
- **Tool Calls**: Number of tool/function calls
- **Cache Hits**: Whether response was cached
- **Success Rate**: Execution success/failure
- **Error Details**: Error type and message

**Example:**

```python
def execute(self, input_data):
    with self.track_execution(input_data) as tracker:
        # Make AI call
        response = self.ai_client.chat(prompt)
        tracker.increment_ai_calls(1)
        tracker.set_tokens(response.usage.total_tokens)
        tracker.set_cost(response.usage.total_tokens * 0.00003)

        # Use tools
        result = self.tool.calculate(data)
        tracker.increment_tool_calls(1)

        # Check cache
        if response.cached:
            tracker.set_cache_hit(True)

        return AgentResult(success=True, data=result)
```

### Execution History

Access historical metrics:

```python
# Get last 100 executions
history = agent.get_execution_history(limit=100)

# Analyze patterns
for execution in history:
    print(f"ID: {execution['execution_id']}")
    print(f"Duration: {execution['duration_ms']}ms")
    print(f"Cost: ${execution['cost_usd']}")
    print(f"Success: {execution['success']}")
```

### Performance Summary

Get aggregated statistics:

```python
# Last 60 minutes
summary = agent.get_performance_summary(window_minutes=60)

print(f"Total Executions: {summary['total_executions']}")
print(f"Success Rate: {summary['success_rate']:.1%}")
print(f"Avg Latency: {summary['latency']['avg_ms']:.0f}ms")
print(f"P95 Latency: {summary['latency']['p95_ms']:.0f}ms")
print(f"Total Cost: ${summary['cost']['total_usd']:.2f}")
print(f"Avg Tokens: {summary['tokens']['avg']:.0f}")
print(f"Cache Hit Rate: {summary['cache_hit_rate']:.1%}")
```

---

## Health Checks

### Comprehensive Health Monitoring

The health check system evaluates:

1. **Monitoring Status**: Is monitoring enabled?
2. **Recent Errors**: Number of errors in last 10 executions
3. **Success Rate**: Overall success percentage
4. **Latency**: Average response time
5. **Last Error**: Details of most recent error
6. **Uptime**: Time since agent started

### Health Status Levels

- **HEALTHY**: All checks passing, no issues
- **DEGRADED**: Some checks failing, but agent functional
- **UNHEALTHY**: Critical issues, agent may not be working

### Performing Health Checks

```python
# Basic health check
health = agent.health_check()

print(f"Status: {health.status.value}")
print(f"Uptime: {health.uptime_seconds / 3600:.1f} hours")

# Check individual tests
for check_name, passed in health.checks.items():
    status = "✓" if passed else "✗"
    print(f"{status} {check_name}")

# View metrics
print(f"Success Rate: {health.metrics['success_rate']:.1%}")
print(f"Avg Latency: {health.metrics['avg_latency_ms']:.0f}ms")

# Degradation reasons
if health.degradation_reasons:
    print("Issues:")
    for reason in health.degradation_reasons:
        print(f"  - {reason}")
```

### Health Check Endpoint

Expose health checks via HTTP:

```python
from flask import Flask, jsonify

app = Flask(__name__)
agent = CarbonAgent()

@app.route('/health')
def health():
    health_result = agent.health_check()
    return jsonify(health_result.to_dict())

@app.route('/health/liveness')
def liveness():
    # Basic liveness check
    return jsonify({"status": "alive"}), 200

@app.route('/health/readiness')
def readiness():
    # Readiness check with health status
    health_result = agent.health_check()

    if health_result.status == HealthStatus.HEALTHY:
        return jsonify({"status": "ready"}), 200
    elif health_result.status == HealthStatus.DEGRADED:
        return jsonify({"status": "degraded"}), 200
    else:
        return jsonify({"status": "not ready"}), 503
```

---

## Metrics Collection

### Prometheus Integration

Export metrics in Prometheus format:

```python
# Get Prometheus-formatted metrics
metrics_text = agent.export_metrics_prometheus()

print(metrics_text)
# Output:
# # TYPE executions_total counter
# executions_total{agent="carbon_agent",success="True"} 150
# executions_total{agent="carbon_agent",success="False"} 5
# # TYPE execution_duration_ms histogram
# execution_duration_ms_sum 187500.0
# execution_duration_ms_count 155
# execution_duration_ms_avg 1209.68
```

### Metrics HTTP Endpoint

```python
from flask import Flask, Response

@app.route('/metrics')
def metrics():
    metrics_text = agent.export_metrics_prometheus()
    return Response(metrics_text, mimetype='text/plain')
```

### Custom Metrics

Add your own metrics:

```python
# In your agent's execute() method
with self.track_execution(input_data) as tracker:
    # Emit custom metric
    self._emit_metric("custom_calculation_time", 123.45)

    # With labels
    self._emit_metric(
        "emissions_calculated",
        total_emissions,
        labels={"fuel_type": "natural_gas", "region": "US"}
    )
```

### Metric Types

1. **Counters**: Ever-increasing values (e.g., total executions)
2. **Gauges**: Point-in-time values (e.g., current queue size)
3. **Histograms**: Distributions (e.g., latency percentiles)

```python
# Counter
self._metrics_collector.increment_counter(
    "api_calls_total",
    labels={"endpoint": "calculate"}
)

# Gauge
self._metrics_collector.set_gauge(
    "queue_size",
    len(self.work_queue)
)

# Histogram (automatic from tracked executions)
# Duration, cost, tokens are automatically histogrammed
```

---

## Alerting System

### Alert Severity Levels

- **INFO**: Informational, no action needed
- **WARNING**: Attention needed, not critical
- **ERROR**: Error occurred, investigate soon
- **CRITICAL**: Critical issue, immediate action required

### Automatic Alerts

Alerts are automatically generated for:

1. **High Latency**: Execution > threshold (default: 5000ms)
2. **High Cost**: Execution > threshold (default: $1.00)
3. **High Error Rate**: Error rate > threshold (default: 10%)
4. **Execution Failures**: Any failed execution

### Configuring Thresholds

```python
# Set custom thresholds
agent.set_thresholds(
    latency_ms=3000,      # Alert if > 3 seconds
    error_rate=0.05,      # Alert if > 5% errors
    cost_usd=0.50         # Alert if > $0.50
)
```

### Retrieving Alerts

```python
# Get all unresolved alerts
alerts = agent.get_alerts(unresolved_only=True)

for alert in alerts:
    print(f"[{alert.severity.value.upper()}] {alert.message}")
    print(f"  Time: {alert.timestamp}")
    print(f"  Context: {alert.context}")

# Filter by severity
critical_alerts = agent.get_alerts(
    severity=AlertSeverity.CRITICAL,
    unresolved_only=True
)
```

### Resolving Alerts

```python
# Resolve an alert
agent.resolve_alert(alert_id="123e4567-e89b-12d3-a456-426614174000")

# Bulk resolve
for alert in agent.get_alerts(severity=AlertSeverity.WARNING):
    agent.resolve_alert(alert.alert_id)
```

### Alert Callbacks

Get notified when alerts are generated:

```python
def alert_callback(alert):
    """Send alert to monitoring system."""
    # Send to Slack
    send_slack_message(
        channel="#alerts",
        text=f"[{alert.severity.value}] {alert.agent_name}: {alert.message}"
    )

    # Send to PagerDuty
    if alert.severity == AlertSeverity.CRITICAL:
        trigger_pagerduty_incident(alert)

    # Log to monitoring system
    log_to_datadog(alert.to_dict())

# Setup agent with callback
agent.setup_monitoring(
    agent_name="carbon_agent",
    alert_callback=alert_callback
)
```

---

## Changelog Management

### Why Changelogs Matter

Changelogs provide:

- **Traceability**: What changed and when
- **Communication**: Inform users of updates
- **Debugging**: Correlate issues with changes
- **Compliance**: Audit trail for certifications
- **Planning**: Understand evolution

### Changelog Template

The `CHANGELOG_TEMPLATE.md` follows [Keep a Changelog](https://keepachangelog.com/) format:

**Categories:**

- **Added**: New features
- **Changed**: Changes to existing features
- **Deprecated**: Soon-to-be-removed features
- **Removed**: Removed features
- **Fixed**: Bug fixes
- **Security**: Security improvements
- **Performance**: Performance optimizations

### Maintaining Changelogs

**When to update:**

1. Before each release
2. After significant changes
3. When deprecating features
4. When introducing breaking changes

**How to update:**

```markdown
## [Unreleased]

### Added
- Added support for international carbon factors (150+ countries)
- Added caching layer for common calculations

### Changed
- Updated default timeout from 30s to 60s
- Improved error messages for better debugging

### Fixed
- Fixed division by zero error in edge case calculations
- Fixed incorrect emissions factor for natural gas
```

### Versioning Best Practices

Follow [Semantic Versioning](https://semver.org/):

- **MAJOR** (X.0.0): Breaking changes
- **MINOR** (0.Y.0): New features (backwards-compatible)
- **PATCH** (0.0.Z): Bug fixes (backwards-compatible)

**Example progression:**

```
1.0.0 → Initial release
1.0.1 → Bug fix (patch)
1.1.0 → New feature (minor)
2.0.0 → Breaking API change (major)
```

---

## Best Practices

### 1. Always Use Context Managers

```python
# Good
with self.track_execution(input_data) as tracker:
    result = self._process(input_data)
    tracker.set_cost(0.08)
    return result

# Bad - metrics won't be tracked
result = self._process(input_data)
return result
```

### 2. Track All Metrics

```python
with self.track_execution(input_data) as tracker:
    # Track AI calls
    response = ai_client.chat(prompt)
    tracker.increment_ai_calls(1)
    tracker.set_tokens(response.usage.total_tokens)

    # Track tool calls
    result = tool.calculate(data)
    tracker.increment_tool_calls(1)

    # Track cost
    cost = response.usage.total_tokens * TOKEN_PRICE
    tracker.set_cost(cost)

    # Track cache hits
    if cached:
        tracker.set_cache_hit(True)
```

### 3. Set Appropriate Thresholds

```python
# Production agent with strict SLAs
agent.set_thresholds(
    latency_ms=2000,     # 2 second SLA
    error_rate=0.01,     # 1% error tolerance
    cost_usd=0.25        # Budget constraint
)

# Development/testing agent
agent.set_thresholds(
    latency_ms=10000,    # Relaxed for debugging
    error_rate=0.10,     # Allow more errors
    cost_usd=1.00        # Higher budget for testing
)
```

### 4. Implement Health Check Endpoints

```python
# Kubernetes liveness probe
@app.route('/health/liveness')
def liveness():
    return jsonify({"status": "alive"}), 200

# Kubernetes readiness probe
@app.route('/health/readiness')
def readiness():
    health = agent.health_check()
    if health.status != HealthStatus.UNHEALTHY:
        return jsonify({"status": "ready"}), 200
    return jsonify({"status": "not ready"}), 503
```

### 5. Monitor in Production

```python
# Setup comprehensive monitoring
agent.setup_monitoring(
    agent_name="carbon_agent_prod",
    enable_metrics=True,
    enable_health_checks=True,
    enable_alerting=True,
    max_history=10000,
    alert_callback=send_to_pagerduty
)

# Regular health checks
while True:
    health = agent.health_check()
    if health.status == HealthStatus.UNHEALTHY:
        trigger_incident()
    time.sleep(60)
```

### 6. Use Structured Logging

The monitoring system automatically logs in JSON format:

```json
{
  "timestamp": "2025-10-16T10:30:00Z",
  "level": "info",
  "message": "Execution completed",
  "agent": "carbon_agent",
  "context": {
    "execution_id": "123e4567-e89b-12d3-a456-426614174000",
    "duration_ms": 1250,
    "cost_usd": 0.08,
    "success": true
  }
}
```

### 7. Regular Performance Reviews

```python
# Weekly performance report
summary = agent.get_performance_summary(window_minutes=10080)  # 7 days

print(f"""
Weekly Performance Report
========================
Total Executions: {summary['total_executions']}
Success Rate: {summary['success_rate']:.1%}
Avg Latency: {summary['latency']['avg_ms']:.0f}ms
P95 Latency: {summary['latency']['p95_ms']:.0f}ms
Total Cost: ${summary['cost']['total_usd']:.2f}
Avg Cost: ${summary['cost']['avg_usd']:.3f}
Cache Hit Rate: {summary['cache_hit_rate']:.1%}
""")
```

---

## Troubleshooting

### Common Issues

#### 1. Monitoring not tracking executions

**Symptom:** `get_execution_history()` returns empty list

**Solution:**

```python
# Check if monitoring is enabled
if not hasattr(agent, '_monitoring_enabled'):
    print("Monitoring not initialized!")
    agent.setup_monitoring(agent_name="my_agent")

# Ensure track_execution is used
def execute(self, input_data):
    with self.track_execution(input_data) as tracker:  # ← Must use this
        return self._do_work(input_data)
```

#### 2. High memory usage

**Symptom:** Agent consuming too much memory

**Solution:**

```python
# Reduce history size
agent.setup_monitoring(
    agent_name="my_agent",
    max_history=100  # Default is 1000
)

# Or periodically reset
if agent._total_executions > 10000:
    agent.reset_monitoring_state()
```

#### 3. Metrics not appearing in Prometheus

**Symptom:** `/metrics` endpoint returns empty

**Solution:**

```python
# Ensure metrics are enabled
agent.setup_monitoring(
    agent_name="my_agent",
    enable_metrics=True  # ← Must be True
)

# Check if executions occurred
if agent._total_executions == 0:
    print("No executions yet - run agent first")
```

#### 4. Alerts not being generated

**Symptom:** No alerts despite errors

**Solution:**

```python
# Ensure alerting is enabled
agent.setup_monitoring(
    agent_name="my_agent",
    enable_alerting=True  # ← Must be True
)

# Check thresholds aren't too high
agent.set_thresholds(
    latency_ms=3000,  # Not 999999
    error_rate=0.10,  # Not 1.0
    cost_usd=0.50     # Not 1000.0
)
```

### Debug Mode

Enable verbose logging:

```python
import logging

# Enable debug logs
logging.basicConfig(level=logging.DEBUG)

# Monitor specific logger
logger = logging.getLogger("carbon_agent.monitoring")
logger.setLevel(logging.DEBUG)
```

### Verification Script

```bash
# Verify integration
python scripts/add_monitoring_and_changelog.py --agent my_agent --verify-only

# Expected output:
#   ✓ Agent Exists
#   ✓ Monitoring Imported
#   ✓ Mixin Inherited
#   ✓ Setup Called
#   ✓ Tracking Used
#   ✓ Changelog Exists
```

---

## API Reference

### OperationalMonitoringMixin

#### `setup_monitoring()`

```python
def setup_monitoring(
    agent_name: str,
    enable_metrics: bool = True,
    enable_health_checks: bool = True,
    enable_alerting: bool = True,
    max_history: int = 1000,
    alert_callback: Optional[Callable[[Alert], None]] = None,
) -> None
```

Initialize monitoring for the agent.

**Parameters:**

- `agent_name` (str): Unique name for the agent
- `enable_metrics` (bool): Enable performance metrics collection
- `enable_health_checks` (bool): Enable health monitoring
- `enable_alerting` (bool): Enable alert generation
- `max_history` (int): Maximum execution history to retain
- `alert_callback` (callable): Function to call when alerts are generated

#### `track_execution()`

```python
@contextmanager
def track_execution(
    input_data: Dict[str, Any],
    track_tokens: bool = True,
    track_cost: bool = True
)
```

Context manager to track execution metrics.

**Parameters:**

- `input_data` (dict): Input data being processed
- `track_tokens` (bool): Track token usage
- `track_cost` (bool): Track cost

**Yields:**

- `ExecutionTracker`: Object to update metrics during execution

**ExecutionTracker Methods:**

- `set_tokens(tokens: int)`: Set token count
- `set_cost(cost: float)`: Set cost in USD
- `increment_tool_calls(count: int = 1)`: Increment tool call counter
- `increment_ai_calls(count: int = 1)`: Increment AI call counter
- `set_cache_hit(hit: bool)`: Mark as cache hit
- `set_error(error_type: str, error_message: str)`: Record error details

#### `health_check()`

```python
def health_check() -> HealthCheckResult
```

Perform comprehensive health check.

**Returns:**

- `HealthCheckResult`: Health status with checks and metrics

#### `get_performance_summary()`

```python
def get_performance_summary(window_minutes: int = 60) -> Dict[str, Any]
```

Get aggregated performance statistics.

**Parameters:**

- `window_minutes` (int): Time window for analysis

**Returns:**

- dict: Performance statistics including latency, cost, tokens

#### `get_alerts()`

```python
def get_alerts(
    severity: Optional[AlertSeverity] = None,
    unresolved_only: bool = True
) -> List[Alert]
```

Get alerts filtered by criteria.

**Parameters:**

- `severity` (AlertSeverity): Filter by severity level
- `unresolved_only` (bool): Only return unresolved alerts

**Returns:**

- list[Alert]: Filtered alerts

#### `resolve_alert()`

```python
def resolve_alert(alert_id: str) -> bool
```

Mark an alert as resolved.

**Parameters:**

- `alert_id` (str): UUID of alert to resolve

**Returns:**

- bool: True if alert was found and resolved

#### `export_metrics_prometheus()`

```python
def export_metrics_prometheus() -> str
```

Export metrics in Prometheus text format.

**Returns:**

- str: Prometheus-formatted metrics

#### `get_execution_history()`

```python
def get_execution_history(limit: int = 100) -> List[Dict[str, Any]]
```

Get recent execution history.

**Parameters:**

- `limit` (int): Maximum executions to return

**Returns:**

- list[dict]: Execution metrics

#### `set_thresholds()`

```python
def set_thresholds(
    latency_ms: Optional[int] = None,
    error_rate: Optional[float] = None,
    cost_usd: Optional[float] = None
)
```

Update alerting thresholds.

**Parameters:**

- `latency_ms` (int): Latency threshold in milliseconds
- `error_rate` (float): Error rate threshold (0.0 to 1.0)
- `cost_usd` (float): Cost threshold in USD

---

## Examples

### Example 1: Basic Integration

```python
from greenlang.agents.base import BaseAgent, AgentResult
from templates.agent_monitoring import OperationalMonitoringMixin

class SimpleAgent(OperationalMonitoringMixin, BaseAgent):
    def __init__(self):
        super().__init__()
        self.setup_monitoring(agent_name="simple_agent")

    def execute(self, input_data):
        with self.track_execution(input_data) as tracker:
            # Do work
            result = {"answer": 42}

            # Track metrics
            tracker.set_cost(0.05)
            tracker.set_tokens(1000)

            return AgentResult(success=True, data=result)

# Usage
agent = SimpleAgent()
result = agent.execute({"question": "What is the answer?"})

# Check health
health = agent.health_check()
print(f"Status: {health.status.value}")
```

### Example 2: Production Agent with Full Monitoring

```python
from greenlang.agents.base import BaseAgent, AgentResult
from templates.agent_monitoring import (
    OperationalMonitoringMixin,
    AlertSeverity
)
import anthropic

class ProductionAgent(OperationalMonitoringMixin, BaseAgent):
    def __init__(self, api_key: str):
        super().__init__()

        # Initialize AI client
        self.client = anthropic.Anthropic(api_key=api_key)

        # Setup monitoring with custom thresholds
        self.setup_monitoring(
            agent_name="production_agent",
            enable_metrics=True,
            enable_health_checks=True,
            enable_alerting=True,
            max_history=5000,
            alert_callback=self._handle_alert
        )

        # Set strict production thresholds
        self.set_thresholds(
            latency_ms=2000,   # 2 second SLA
            error_rate=0.01,   # 1% error tolerance
            cost_usd=0.25      # Cost control
        )

    def execute(self, input_data):
        with self.track_execution(input_data) as tracker:
            try:
                # Call AI
                response = self.client.messages.create(
                    model="claude-3-5-sonnet-20241022",
                    max_tokens=1000,
                    messages=[{
                        "role": "user",
                        "content": input_data["prompt"]
                    }]
                )

                # Track AI metrics
                tracker.increment_ai_calls(1)
                tracker.set_tokens(response.usage.total_tokens)

                # Calculate cost (example rate)
                cost = (
                    response.usage.input_tokens * 0.000003 +
                    response.usage.output_tokens * 0.000015
                )
                tracker.set_cost(cost)

                # Process response
                result = self._process_response(response)
                tracker.increment_tool_calls(len(result.get("tools_used", [])))

                return AgentResult(
                    success=True,
                    data=result
                )

            except Exception as e:
                # Error is automatically tracked
                tracker.set_error(type(e).__name__, str(e))
                raise

    def _process_response(self, response):
        # Processing logic
        return {
            "answer": response.content[0].text,
            "tools_used": []
        }

    def _handle_alert(self, alert):
        """Handle generated alerts."""
        if alert.severity == AlertSeverity.CRITICAL:
            # Send to PagerDuty
            self._trigger_pagerduty(alert)
        elif alert.severity == AlertSeverity.ERROR:
            # Send to Slack
            self._send_slack_alert(alert)
        else:
            # Log only
            print(f"[{alert.severity.value}] {alert.message}")

    def _trigger_pagerduty(self, alert):
        # PagerDuty integration
        pass

    def _send_slack_alert(self, alert):
        # Slack integration
        pass

# Usage
agent = ProductionAgent(api_key="your-api-key")

# Execute
result = agent.execute({"prompt": "Calculate emissions"})

# Monitor
health = agent.health_check()
summary = agent.get_performance_summary(window_minutes=60)
alerts = agent.get_alerts(unresolved_only=True)

print(f"Health: {health.status.value}")
print(f"Success Rate: {summary['success_rate']:.1%}")
print(f"Active Alerts: {len(alerts)}")
```

### Example 3: HTTP Monitoring Endpoints

```python
from flask import Flask, jsonify, Response
from greenlang.agents.carbon_agent import CarbonAgent

app = Flask(__name__)
agent = CarbonAgent()  # Assumes monitoring is integrated

@app.route('/health')
def health():
    """Detailed health check."""
    health_result = agent.health_check()
    return jsonify(health_result.to_dict())

@app.route('/health/liveness')
def liveness():
    """Kubernetes liveness probe."""
    return jsonify({"status": "alive"}), 200

@app.route('/health/readiness')
def readiness():
    """Kubernetes readiness probe."""
    health_result = agent.health_check()

    if health_result.status == HealthStatus.UNHEALTHY:
        return jsonify({"status": "not ready"}), 503

    return jsonify({"status": "ready"}), 200

@app.route('/metrics')
def metrics():
    """Prometheus metrics endpoint."""
    metrics_text = agent.export_metrics_prometheus()
    return Response(metrics_text, mimetype='text/plain')

@app.route('/performance')
def performance():
    """Performance summary."""
    summary = agent.get_performance_summary(window_minutes=60)
    return jsonify(summary)

@app.route('/alerts')
def alerts():
    """Active alerts."""
    active_alerts = agent.get_alerts(unresolved_only=True)
    return jsonify([alert.to_dict() for alert in active_alerts])

@app.route('/history')
def history():
    """Execution history."""
    history = agent.get_execution_history(limit=100)
    return jsonify(history)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
```

---

## Deployment Checklist

Before deploying an agent with monitoring:

- [ ] Monitoring integrated (run verification script)
- [ ] CHANGELOG.md created and populated
- [ ] Health check endpoint configured
- [ ] Metrics endpoint configured (/metrics)
- [ ] Alert thresholds set appropriately
- [ ] Alert callbacks configured (PagerDuty, Slack, etc.)
- [ ] Logging configured (structured JSON logs)
- [ ] Performance baseline established
- [ ] Load testing completed
- [ ] Documentation updated
- [ ] Team trained on monitoring dashboards
- [ ] Runbooks created for common alerts

---

## Support

**Documentation:** https://docs.greenlang.io/monitoring
**Issues:** https://github.com/greenlang/agents/issues
**Community:** https://community.greenlang.io
**Enterprise Support:** enterprise@greenlang.io

---

**Version:** 1.0.0
**Last Updated:** 2025-10-16
**Maintained By:** GreenLang Framework Team
