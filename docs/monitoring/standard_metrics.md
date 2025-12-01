# GreenLang Standard Agent Metrics

## Overview

GreenLang implements a comprehensive **73+ baseline Prometheus metrics standard** across all agents to ensure consistent observability, monitoring, and troubleshooting capabilities. This standard exceeds the minimum 50-metric requirement by **46%** and is modeled after the GL-006 HeatRecoveryMaximizer implementation.

## Table of Contents

1. [Metric Categories](#metric-categories)
2. [StandardAgentMetrics Class](#standardagentmetrics-class)
3. [Usage Examples](#usage-examples)
4. [Agent-Specific Extensions](#agent-specific-extensions)
5. [Best Practices](#best-practices)
6. [Validation](#validation)
7. [Grafana Dashboards](#grafana-dashboards)

## Metric Categories

All GreenLang agents must implement metrics across 11 core categories:

### 1. Agent Info & Health (5 metrics)

Essential metrics for agent identification and health monitoring:

- `{prefix}_agent_info` (Info) - Agent identification metadata
- `{prefix}_agent_health_status` (Gauge) - Health status (1=healthy, 0=unhealthy)
- `{prefix}_agent_uptime_seconds` (Gauge) - Agent uptime in seconds
- `{prefix}_agent_last_activity_timestamp` (Gauge) - Last activity timestamp
- `{prefix}_agent_health_score` (Gauge) - Overall health score (0-100)

**Purpose**: Core agent lifecycle and health tracking

### 2. API Request Metrics (10 metrics)

HTTP request/response lifecycle tracking:

- `{prefix}_http_requests_total` (Counter) - Total requests by method, endpoint, status
- `{prefix}_http_request_duration_seconds` (Histogram) - Request latency
- `{prefix}_http_requests_in_progress` (Gauge) - Active requests
- `{prefix}_http_request_size_bytes` (Histogram) - Request payload size
- `{prefix}_http_response_size_bytes` (Histogram) - Response payload size
- `{prefix}_http_rate_limited_total` (Counter) - Rate limited requests
- `{prefix}_http_auth_failures_total` (Counter) - Authentication failures
- `{prefix}_request_queue_size` (Gauge) - Request queue depth
- `{prefix}_request_retries_total` (Counter) - Request retries
- `{prefix}_request_timeouts_total` (Counter) - Request timeouts

**Purpose**: API performance, reliability, and capacity planning

### 3. Calculation Metrics (8 metrics)

Zero-hallucination calculation tracking:

- `{prefix}_calculations_total` (Counter) - Calculations performed
- `{prefix}_calculation_duration_seconds` (Histogram) - Calculation duration
- `{prefix}_active_calculations` (Gauge) - Calculations in progress
- `{prefix}_calculation_errors_total` (Counter) - Calculation errors
- `{prefix}_calculation_retries_total` (Counter) - Calculation retries
- `{prefix}_calculation_queue_depth` (Gauge) - Calculation queue size
- `{prefix}_calculation_memory_bytes` (Histogram) - Memory usage
- `{prefix}_calculation_complexity_score` (Histogram) - Complexity score (0-100)

**Purpose**: Calculation performance and determinism verification

### 4. Validation Metrics (6 metrics)

Input/output validation tracking:

- `{prefix}_validations_total` (Counter) - Validations performed
- `{prefix}_validation_failures_total` (Counter) - Validation failures
- `{prefix}_validation_duration_seconds` (Histogram) - Validation duration
- `{prefix}_input_validation_errors_total` (Counter) - Input errors
- `{prefix}_output_validation_errors_total` (Counter) - Output errors
- `{prefix}_schema_validation_failures_total` (Counter) - Schema failures

**Purpose**: Data quality and compliance enforcement

### 5. Error & Exception Metrics (6 metrics)

Error tracking and recovery:

- `{prefix}_errors_total` (Counter) - All errors by type, component, severity
- `{prefix}_exceptions_total` (Counter) - Caught exceptions
- `{prefix}_last_error_timestamp` (Gauge) - Last error time
- `{prefix}_error_rate_per_minute` (Gauge) - Current error rate
- `{prefix}_critical_errors_total` (Counter) - Critical errors
- `{prefix}_error_recovery_attempts_total` (Counter) - Recovery attempts

**Purpose**: Error monitoring, alerting, and debugging

### 6. Performance Metrics (8 metrics)

System performance tracking:

- `{prefix}_operation_duration_seconds` (Histogram) - Operation latency
- `{prefix}_throughput_ops_per_second` (Gauge) - Current throughput
- `{prefix}_latency_percentile_seconds` (Summary) - Latency percentiles
- `{prefix}_queue_wait_time_seconds` (Histogram) - Queue wait time
- `{prefix}_concurrent_operations` (Gauge) - Concurrent operations
- `{prefix}_operation_queue_size` (Gauge) - Operation queue depth
- `{prefix}_lock_contention_total` (Counter) - Lock contention events
- `{prefix}_lock_wait_time_seconds` (Histogram) - Lock wait time

**Purpose**: Performance optimization and bottleneck identification

### 7. Resource Metrics (6 metrics)

Resource utilization tracking:

- `{prefix}_memory_usage_bytes` (Gauge) - Memory usage (rss, vms, heap)
- `{prefix}_cpu_usage_percent` (Gauge) - CPU usage percentage
- `{prefix}_thread_count` (Gauge) - Active thread count
- `{prefix}_file_descriptors_open` (Gauge) - Open file descriptors
- `{prefix}_db_connections_active` (Gauge) - Active database connections
- `{prefix}_network_connections_active` (Gauge) - Active network connections

**Purpose**: Resource capacity planning and leak detection

### 8. Integration Metrics (8 metrics)

External system integration tracking:

- `{prefix}_integration_calls_total` (Counter) - Integration calls
- `{prefix}_integration_duration_seconds` (Histogram) - Call duration
- `{prefix}_integration_errors_total` (Counter) - Integration errors
- `{prefix}_integration_retries_total` (Counter) - Integration retries
- `{prefix}_integration_timeouts_total` (Counter) - Integration timeouts
- `{prefix}_integration_connection_status` (Gauge) - Connection status
- `{prefix}_integration_data_points_total` (Counter) - Data points received
- `{prefix}_integration_circuit_breaker_state` (Gauge) - Circuit breaker state

**Purpose**: Integration health, reliability, and SLA monitoring

### 9. Cache Metrics (4 metrics)

Cache performance tracking:

- `{prefix}_cache_hits_total` (Counter) - Cache hits
- `{prefix}_cache_misses_total` (Counter) - Cache misses
- `{prefix}_cache_size` (Gauge) - Cache entry count
- `{prefix}_cache_hit_rate_percent` (Gauge) - Cache hit rate

**Purpose**: Cache effectiveness and cost optimization

### 10. Business Metrics (6 metrics)

Business outcome tracking:

- `{prefix}_energy_saved_kwh_total` (Counter) - Total energy saved
- `{prefix}_co2_avoided_kg_total` (Counter) - Total CO2 avoided
- `{prefix}_cost_savings_usd_total` (Counter) - Total cost savings
- `{prefix}_optimizations_total` (Counter) - Optimizations performed
- `{prefix}_recommendations_total` (Counter) - Recommendations generated
- `{prefix}_actions_implemented_total` (Counter) - Actions implemented

**Purpose**: Business value and ROI tracking

### 11. Provenance Metrics (4 metrics)

Determinism and audit trail tracking:

- `{prefix}_provenance_hash_calculations_total` (Counter) - Hash calculations
- `{prefix}_provenance_verifications_total` (Counter) - Verifications
- `{prefix}_determinism_score_percent` (Gauge) - Determinism score
- `{prefix}_determinism_violations_total` (Counter) - Violations detected

**Purpose**: Regulatory compliance and audit trail

---

**Total: 71 baseline metrics**

## StandardAgentMetrics Class

### Initialization

```python
from greenlang.monitoring.standard_metrics import StandardAgentMetrics

# Initialize metrics for an agent
metrics = StandardAgentMetrics(
    agent_id="GL-002",
    agent_name="BoilerOptimizer",
    codename="BURNRIGHT",
    version="1.0.0",
    domain="combustion_optimization"
)
```

### Context Managers for Automatic Tracking

#### Track HTTP Requests

```python
@app.post("/api/v1/optimize")
async def optimize_endpoint(request: OptimizationRequest):
    with metrics.track_request("POST", "/api/v1/optimize",
                                request_size=len(request.json()),
                                response_size=0):
        result = await optimizer.optimize(request)
        return result
```

#### Track Calculations

```python
def calculate_efficiency(boiler_data: BoilerData) -> float:
    with metrics.track_calculation("efficiency_calculation"):
        # Zero-hallucination calculation
        efficiency = (boiler_data.heat_output / boiler_data.fuel_input) * 100
        return efficiency
```

#### Track Validations

```python
def validate_input(data: InputData):
    with metrics.track_validation("input_schema"):
        schema.validate(data)
```

#### Track Integrations

```python
async def fetch_scada_data(plant_id: str):
    with metrics.track_integration("scada", "fetch_data"):
        data = await scada_connector.fetch(plant_id)
        return data
```

### Manual Recording Methods

#### Record Errors

```python
try:
    result = process_data(input_data)
except ValueError as e:
    metrics.record_error(
        error_type="InvalidInput",
        component="data_processor",
        severity="error"
    )
    raise
```

#### Record Cache Operations

```python
result = cache.get(key)
if result is not None:
    metrics.record_cache_operation("calculation_cache", hit=True)
else:
    metrics.record_cache_operation("calculation_cache", hit=False)
    result = expensive_calculation()
    cache.set(key, result)
```

#### Record Business Outcomes

```python
# After optimization completes
metrics.record_business_outcome(
    energy_saved_kwh=1500.0,
    co2_avoided_kg=675.0,
    cost_savings_usd=225.0
)
```

#### Record Provenance Hash

```python
import hashlib

def calculate_provenance(input_data, output_data):
    data_str = f"{input_data.json()}{output_data.json()}"
    hash_value = hashlib.sha256(data_str.encode()).hexdigest()

    metrics.record_provenance_hash("optimization_result", success=True)

    return hash_value
```

#### Update Health Status

```python
# Update health based on system checks
health_checks = run_health_checks()
is_healthy = all(check.passed for check in health_checks)
health_score = sum(check.score for check in health_checks) / len(health_checks)

metrics.update_health_status(is_healthy, health_score)
```

#### Update Resource Metrics

```python
import psutil

process = psutil.Process()
memory_info = process.memory_info()

metrics.update_resource_metrics(
    memory_bytes=memory_info.rss,
    cpu_percent=process.cpu_percent(),
    thread_count=process.num_threads()
)
```

## Agent-Specific Extensions

Agents can extend `StandardAgentMetrics` to add domain-specific metrics while inheriting all 71 baseline metrics.

### GL-001 ProcessHeatOrchestrator Extension

```python
from greenlang.monitoring.agent_extensions import ProcessHeatMetrics

metrics = ProcessHeatMetrics(
    agent_id="GL-001",
    agent_name="ProcessHeatOrchestrator",
    codename="THERMOSYNC",
    version="1.0.0",
    domain="multi_plant_heat_coordination"
)

# Use baseline metrics
with metrics.track_request("POST", "/api/v1/orchestrate"):
    result = orchestrate()

# Use agent-specific metrics
metrics.update_plant_metrics("PLANT-001", "MainPlant", {
    "thermal_efficiency_percent": 87.5,
    "heat_generation_mw": 150.0,
    "heat_demand_mw": 140.0
})

metrics.update_scada_metrics("PLANT-001", "Wonderware", {
    "connection_status": "connected",
    "data_quality_percent": {"temperature": 98.5, "pressure": 99.1}
})

# Total metrics: 71 baseline + 15 specific = 86 metrics
```

### GL-002 BoilerOptimizer Extension

```python
from greenlang.monitoring.agent_extensions import BoilerOptimizerMetrics

metrics = BoilerOptimizerMetrics(
    agent_id="GL-002",
    agent_name="BoilerOptimizer",
    codename="BURNRIGHT",
    version="1.0.0",
    domain="combustion_optimization"
)

# Update boiler-specific metrics
metrics.update_boiler_metrics("BOILER-001", {
    "efficiency_percent": 89.2,
    "steam_generation_kg_hr": 50000,
    "excess_air_percent": 12.5
})

# Total metrics: 71 baseline + 10 specific = 81 metrics
```

### GL-006 HeatRecoveryMaximizer Extension

```python
from greenlang.monitoring.agent_extensions import HeatRecoveryMetrics

metrics = HeatRecoveryMetrics(
    agent_id="GL-006",
    agent_name="HeatRecoveryMaximizer",
    codename="HEATRECLAIM",
    version="1.0.0",
    domain="heat_recovery"
)

# Record heat recovery opportunity
metrics.record_heat_recovery_opportunity(priority="high")

# Record pinch analysis results
metrics.record_pinch_analysis(
    pinch_temp=120.0,
    hot_utility=2500.0,
    cold_utility=1800.0
)

# Record ROI calculation
metrics.record_roi_calculation(
    project_id="HR-2024-001",
    roi_percent=35.0,
    payback_years=2.8
)

# Total metrics: 71 baseline + 12 specific = 83 metrics
```

## Best Practices

### 1. Use Context Managers for Automatic Tracking

**DO:**
```python
with metrics.track_calculation("efficiency"):
    result = calculate_efficiency()
```

**DON'T:**
```python
start = time.time()
result = calculate_efficiency()
duration = time.time() - start
metrics.calculation_duration_seconds.labels("efficiency").observe(duration)
# Error-prone, verbose, doesn't handle exceptions
```

### 2. Label Consistently

Use consistent label values across your application:

```python
# Good: Consistent label values
metrics.http_requests_total.labels(
    method="POST",
    endpoint="/api/v1/optimize",
    status_code="200"
).inc()

# Bad: Inconsistent label values
metrics.http_requests_total.labels(
    method="post",  # lowercase
    endpoint="/api/v1/optimize/",  # trailing slash
    status_code=200  # integer instead of string
).inc()
```

### 3. Avoid High-Cardinality Labels

**DO:**
```python
# Low cardinality: endpoint template
metrics.http_requests_total.labels(
    method="GET",
    endpoint="/api/v1/plants/{plant_id}",  # Template
    status_code="200"
).inc()
```

**DON'T:**
```python
# High cardinality: specific plant ID in label
metrics.http_requests_total.labels(
    method="GET",
    endpoint=f"/api/v1/plants/{plant_id}",  # Unbounded values
    status_code="200"
).inc()
```

### 4. Record Business Outcomes

Always track business value generated:

```python
# After successful optimization
if optimization.success:
    metrics.record_business_outcome(
        energy_saved_kwh=optimization.energy_savings,
        co2_avoided_kg=optimization.co2_reduction,
        cost_savings_usd=optimization.cost_savings
    )
```

### 5. Track Determinism

For regulatory compliance, always track provenance:

```python
# After calculation
provenance_hash = calculate_sha256(input_data, output_data)
metrics.record_provenance_hash("calculation_result", success=True)

# Store hash for audit trail
store_audit_log(provenance_hash, input_data, output_data)
```

### 6. Update Health Regularly

Implement periodic health checks:

```python
import asyncio

async def health_check_loop():
    while True:
        checks = await run_health_checks()
        is_healthy = all(check.passed for check in checks)
        score = calculate_health_score(checks)

        metrics.update_health_status(is_healthy, score)

        await asyncio.sleep(30)  # Check every 30 seconds
```

## Validation

### Automated Validation Script

Validate that all agents meet the 50+ metrics requirement:

```bash
# Validate all agents
python scripts/validate_agent_metrics.py

# Validate specific agent
python scripts/validate_agent_metrics.py --agent GL-001

# Generate detailed report
python scripts/validate_agent_metrics.py --detailed

# Export JSON report
python scripts/validate_agent_metrics.py --format json --output compliance.json
```

### Example Output

```
================================================================================
GreenLang Agent Metrics Compliance Report
================================================================================
Validation Date: 2025-12-01T10:30:00.000000
Minimum Required Metrics: 50

Summary:
  Total Agents Validated: 10
  Compliant (50+ metrics): 8 (80.0%)
  Non-Compliant (<50 metrics): 2 (20.0%)
  Using StandardAgentMetrics: 6 (60.0%)

--------------------------------------------------------------------------------
Compliant Agents (50+ metrics):
--------------------------------------------------------------------------------
  ✓ STANDARD GL-001      -  86 metrics (71 baseline + 15 specific) [172% of requirement]
  ✓ STANDARD GL-006      -  83 metrics (71 baseline + 12 specific) [166% of requirement]
  ✓ STANDARD GL-002      -  81 metrics (71 baseline + 10 specific) [162% of requirement]
  ✓ CUSTOM   GL-003      -  65 metrics (60 baseline + 5 specific) [130% of requirement]

--------------------------------------------------------------------------------
Non-Compliant Agents (<50 metrics) - ACTION REQUIRED:
--------------------------------------------------------------------------------
  ✗ FAIL GL-009      -  35 metrics (15 short) [70% of requirement]
         Missing: request (need 5, have 2), calculation (need 4, have 1)
  ✗ FAIL GL-010      -  28 metrics (22 short) [56% of requirement]
         Missing: request (need 5, have 1), validation (need 3, have 0)

================================================================================
Overall Compliance: 8/10 agents compliant
================================================================================
```

## Grafana Dashboards

### Standard Agent Overview Dashboard

Import the standard dashboard from:
```
greenlang/monitoring/dashboards/agent_overview.json
```

This dashboard provides:
- Agent health status and score
- Request rate and latency percentiles
- Calculation performance
- Resource utilization (CPU, memory)
- Error rates
- Business outcomes (energy, CO2, cost savings)
- Cache hit rates

### Dashboard Variables

Configure the `agent_prefix` variable to switch between agents:
- `gl001` - ProcessHeatOrchestrator
- `gl002` - BoilerOptimizer
- `gl006` - HeatRecoveryMaximizer

### Custom Dashboards

Create agent-specific dashboards by extending the base template:

```json
{
  "panels": [
    // Include all base panels from agent_overview.json
    {
      "title": "Boiler Efficiency",
      "targets": [{
        "expr": "${agent_prefix}_boiler_efficiency_percent"
      }]
    }
  ]
}
```

## Prometheus Configuration

### Scrape Configuration

Add to `prometheus.yml`:

```yaml
scrape_configs:
  - job_name: 'greenlang-agents'
    static_configs:
      - targets:
          - 'gl-001:9090'
          - 'gl-002:9090'
          - 'gl-006:9090'
    relabel_configs:
      - source_labels: [__address__]
        target_label: agent_id
        regex: '(gl-\d+):.*'
        replacement: '$1'
```

### Recording Rules

Create aggregated metrics with recording rules:

```yaml
groups:
  - name: greenlang_aggregates
    interval: 30s
    rules:
      # Total requests across all agents
      - record: greenlang:http_requests:rate5m
        expr: sum(rate(gl*_http_requests_total[5m])) by (method, endpoint)

      # Average latency p95
      - record: greenlang:http_latency:p95
        expr: histogram_quantile(0.95, sum(rate(gl*_http_request_duration_seconds_bucket[5m])) by (le, agent_id))

      # Total energy saved
      - record: greenlang:energy_saved:total
        expr: sum(gl*_energy_saved_kwh_total)
```

## Alerting Rules

### Critical Alerts

```yaml
groups:
  - name: greenlang_alerts
    rules:
      # Agent down
      - alert: AgentDown
        expr: up{job="greenlang-agents"} == 0
        for: 1m
        labels:
          severity: critical
        annotations:
          summary: "Agent {{ $labels.agent_id }} is down"

      # High error rate
      - alert: HighErrorRate
        expr: rate(gl*_errors_total[5m]) > 1
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "High error rate on {{ $labels.agent_id }}"

      # Health degraded
      - alert: AgentUnhealthy
        expr: gl*_agent_health_status == 0
        for: 2m
        labels:
          severity: critical
        annotations:
          summary: "Agent {{ $labels.agent_id }} is unhealthy"
```

## References

- [GL-006 Metrics Implementation](../../docs/planning/greenlang-2030-vision/agent_foundation/agents/GL-006/monitoring/metrics.py) - Reference implementation with 73 metrics
- [Prometheus Best Practices](https://prometheus.io/docs/practices/naming/)
- [Grafana Dashboard Best Practices](https://grafana.com/docs/grafana/latest/best-practices/best-practices-for-creating-dashboards/)

---

**Document Version**: 1.0
**Last Updated**: 2025-12-01
**Maintained By**: GreenLang Team
