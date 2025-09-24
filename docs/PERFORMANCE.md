# GreenLang v0.2.0 Performance Guide

This document outlines performance baselines, Service Level Agreements (SLAs), monitoring strategies, and optimization guidelines for GreenLang v0.2.0 production deployments.

## Table of Contents

- [Performance Baselines](#performance-baselines)
- [Service Level Agreements (SLAs)](#service-level-agreements-slas)
- [Monitoring and Observability](#monitoring-and-observability)
- [Performance Testing](#performance-testing)
- [Optimization Guidelines](#optimization-guidelines)
- [Troubleshooting](#troubleshooting)
- [Capacity Planning](#capacity-planning)

## Performance Baselines

### Test Environment
All baseline measurements were conducted on:
- **CPU**: 4-core x86_64 processor
- **Memory**: 8GB RAM
- **Storage**: SSD storage
- **Python**: 3.10+
- **OS**: Linux/Windows/macOS

### Core Operation Baselines

#### Agent Execution Performance

| Complexity Level | P50 Latency | P95 Latency | P99 Latency | Throughput (ops/sec) | Memory Usage |
|------------------|-------------|-------------|-------------|----------------------|--------------|
| Light            | 5ms         | 15ms        | 25ms        | 100                  | 10MB         |
| Medium           | 25ms        | 75ms        | 150ms       | 50                   | 25MB         |
| Heavy            | 150ms       | 400ms       | 800ms       | 8                    | 75MB         |

#### Pipeline Execution Performance

| Pipeline Type | Steps | P50 Latency | P95 Latency | P99 Latency | Throughput (ops/sec) |
|---------------|-------|-------------|-------------|-------------|----------------------|
| Simple        | 2     | 50ms        | 150ms       | 300ms       | 25                   |
| Medium        | 5     | 200ms       | 500ms       | 1000ms      | 10                   |
| Complex       | 10    | 800ms       | 2000ms      | 4000ms      | 2                    |

#### System Resource Baselines

| Operation             | Memory Peak | Memory Average | CPU Average | I/O Impact |
|-----------------------|-------------|----------------|-------------|------------|
| Pack Loading          | 50MB        | 30MB           | 15%         | Low        |
| Context Creation      | 20MB        | 15MB           | 5%          | Minimal    |
| Memory Intensive      | 200MB       | 150MB          | 25%         | Low        |
| Concurrent (10 users) | 100MB       | 80MB           | 60%         | Medium     |

## Service Level Agreements (SLAs)

### Production SLA Targets

#### Availability
- **Target**: 99.9% uptime (8.77 hours downtime/year)
- **Measurement**: Health check endpoint availability
- **Recovery**: < 5 minutes for planned maintenance, < 15 minutes for incidents

#### Performance
- **Agent Execution P95**: < 100ms for light operations, < 500ms for heavy operations
- **Pipeline Execution P95**: < 200ms for simple pipelines, < 2s for complex pipelines
- **API Response Time P95**: < 250ms
- **Health Check Response**: < 50ms

#### Throughput
- **Agent Operations**: > 50 operations/second sustained
- **Pipeline Executions**: > 20 pipelines/second for simple workflows
- **Concurrent Users**: Support for 100+ concurrent users

#### Resource Limits
- **Memory Usage**: < 1GB per process under normal load
- **CPU Usage**: < 80% average, < 95% peak
- **Disk Usage**: < 80% of allocated storage

### Error Rates
- **Success Rate**: > 99.5%
- **Error Rate**: < 0.5% of all operations
- **Timeout Rate**: < 0.1% of all operations

## Monitoring and Observability

### Key Metrics to Monitor

#### Application Metrics
```python
# Pipeline execution metrics
greenlang_pipeline_executions_total{pipeline_name, status}
greenlang_pipeline_duration_seconds{pipeline_name}

# Agent execution metrics
greenlang_agent_executions_total{agent_name, status}
greenlang_agent_duration_seconds{agent_name}

# Error metrics
greenlang_errors_total{component, error_type}
```

#### System Metrics
```python
# Resource utilization
greenlang_memory_usage_bytes
greenlang_cpu_usage_percent
system_memory_used_percent
system_memory_available_bytes

# Performance metrics
greenlang_response_time_seconds{operation}
greenlang_throughput_ops_per_second{operation_type}
```

### Health Check Endpoints

#### Liveness Probe
```
GET /health/live
```
- **Purpose**: Verify the application is running
- **SLA**: < 10ms response time, 100% availability
- **Action**: Restart container if failing

#### Readiness Probe
```
GET /health/ready
```
- **Purpose**: Verify the application can serve traffic
- **SLA**: < 50ms response time
- **Action**: Remove from load balancer if failing

#### Deep Health Check
```
GET /health
```
- **Purpose**: Comprehensive system health assessment
- **Components Checked**:
  - System resources (memory, CPU, disk)
  - Python runtime status
  - GreenLang core functionality
  - Metrics collector health
  - Custom component checks

### Alerting Rules

#### Critical Alerts (Page Immediately)
```yaml
# High error rate
- alert: HighErrorRate
  expr: rate(greenlang_errors_total[5m]) > 0.1
  labels:
    severity: critical
  annotations:
    summary: "GreenLang error rate > 10%"

# Service down
- alert: ServiceDown
  expr: up{job="greenlang"} == 0
  for: 1m
  labels:
    severity: critical

# High latency
- alert: HighLatency
  expr: histogram_quantile(0.95, greenlang_pipeline_duration_seconds) > 2.0
  for: 2m
  labels:
    severity: critical
```

#### Warning Alerts (Review During Business Hours)
```yaml
# Elevated error rate
- alert: ElevatedErrorRate
  expr: rate(greenlang_errors_total[10m]) > 0.05
  for: 5m
  labels:
    severity: warning

# High memory usage
- alert: HighMemoryUsage
  expr: greenlang_memory_usage_bytes > 800_000_000  # 800MB
  for: 10m
  labels:
    severity: warning

# Degraded performance
- alert: DegradedPerformance
  expr: histogram_quantile(0.95, greenlang_agent_duration_seconds) > 0.5
  for: 5m
  labels:
    severity: warning
```

## Performance Testing

### Running Benchmarks

```bash
# Basic benchmark suite
python -m greenlang.benchmarks.performance_suite --iterations 100

# With load testing
python -m greenlang.benchmarks.performance_suite --load-test --users 10 --requests 100

# Full production test
python -m greenlang.benchmarks.performance_suite --iterations 1000 --load-test --users 50 --requests 200
```

### Continuous Performance Testing

#### Pre-deployment Testing
```bash
# Performance gate - must pass before deployment
./scripts/performance-gate.sh
```

#### Post-deployment Validation
```bash
# Smoke test after deployment
./scripts/performance-smoke-test.sh
```

### Load Testing Scenarios

#### Scenario 1: Normal Load
- **Users**: 10 concurrent
- **Duration**: 5 minutes
- **Operations**: Mixed pipeline execution
- **Expected**: P95 < 500ms, 0% errors

#### Scenario 2: Peak Load
- **Users**: 50 concurrent
- **Duration**: 10 minutes
- **Operations**: Heavy agent processing
- **Expected**: P95 < 1s, < 1% errors

#### Scenario 3: Stress Test
- **Users**: 100 concurrent
- **Duration**: 15 minutes
- **Operations**: Maximum complexity
- **Expected**: Graceful degradation, no crashes

### Performance Regression Detection

Automated regression detection compares current performance against baselines:

```python
# Example regression check
def check_regression(results, baselines):
    regressions = []

    for benchmark, result in results.items():
        baseline = baselines.get(benchmark)
        if not baseline:
            continue

        # P95 latency regression (>20% increase)
        if result.p95 > baseline.p95 * 1.2:
            regressions.append(f"{benchmark}: P95 latency regression")

        # Throughput regression (>20% decrease)
        if result.throughput < baseline.throughput * 0.8:
            regressions.append(f"{benchmark}: throughput regression")

    return regressions
```

## Optimization Guidelines

### Application-Level Optimizations

#### Agent Performance
```python
# Use caching for expensive operations
from functools import lru_cache

class OptimizedAgent(Agent):
    @lru_cache(maxsize=128)
    def expensive_computation(self, input_data):
        # Cache results for repeated inputs
        return compute_result(input_data)
```

#### Pipeline Efficiency
```yaml
# Optimize pipeline structure
pipeline:
  name: "optimized_pipeline"
  steps:
    - name: "parallel_step"
      type: "parallel"  # Run multiple agents concurrently
      agents:
        - "agent_1"
        - "agent_2"
    - name: "aggregation"
      agent: "aggregator"
```

#### Memory Management
```python
# Explicit garbage collection for long-running processes
import gc

class MemoryOptimizedExecutor(Executor):
    def execute_pipeline(self, pipeline, inputs):
        result = super().execute_pipeline(pipeline, inputs)

        # Clean up after heavy operations
        if pipeline.get('memory_intensive'):
            gc.collect()

        return result
```

### System-Level Optimizations

#### Resource Allocation
```yaml
# Docker resource limits
services:
  greenlang:
    image: greenlang:latest
    resources:
      limits:
        memory: 2G
        cpus: '2.0'
      reservations:
        memory: 1G
        cpus: '1.0'
```

#### Concurrency Settings
```python
# Optimize executor settings
executor = Executor(
    backend="local",
    deterministic=False,  # Disable for better performance
    max_workers=4,        # Match CPU cores
    enable_caching=True   # Enable result caching
)
```

### Database and Storage Optimizations

#### Artifact Storage
```python
# Use efficient storage backends
from greenlang.storage import S3Storage, RedisCache

# Configure optimized storage
storage_config = {
    "artifacts": S3Storage(bucket="greenlang-artifacts"),
    "cache": RedisCache(host="redis-cluster"),
    "compression": "gzip"  # Compress large artifacts
}
```

## Troubleshooting

### Performance Issues

#### High Latency
1. **Check system resources**: CPU, memory, disk I/O
2. **Review pipeline complexity**: Simplify or parallelize steps
3. **Examine agent performance**: Profile individual agents
4. **Database queries**: Optimize data access patterns

#### Low Throughput
1. **Increase concurrency**: Adjust worker pool size
2. **Remove bottlenecks**: Identify and optimize slow operations
3. **Scale horizontally**: Deploy additional instances
4. **Optimize I/O**: Use async operations where possible

#### Memory Issues
1. **Check for leaks**: Monitor memory growth over time
2. **Review agent caching**: Ensure caches have size limits
3. **Garbage collection**: Tune GC settings for workload
4. **Resource cleanup**: Ensure proper resource disposal

### Common Performance Anti-Patterns

#### Avoid These Patterns
```python
# ❌ Don't: Synchronous I/O in agents
def slow_agent(data):
    response = requests.get("http://slow-api.com")  # Blocking
    return process(response.json())

# ✅ Do: Asynchronous I/O
async def fast_agent(data):
    async with httpx.AsyncClient() as client:
        response = await client.get("http://api.com")
        return process(response.json())

# ❌ Don't: Unbounded caching
cache = {}  # Will grow indefinitely

# ✅ Do: Bounded caching
from functools import lru_cache
@lru_cache(maxsize=1000)
def cached_operation(input):
    return expensive_computation(input)
```

### Diagnostic Commands

```bash
# System resource usage
htop
iostat -x 1
free -h

# Application metrics
curl http://localhost:9090/metrics | grep greenlang

# Health status
curl http://localhost:8080/health

# Performance profiling
python -m cProfile -s cumtime your_pipeline.py
```

## Capacity Planning

### Sizing Guidelines

#### Minimum Requirements
- **CPU**: 2 cores
- **Memory**: 4GB RAM
- **Storage**: 20GB SSD
- **Network**: 100 Mbps

#### Recommended Production
- **CPU**: 4+ cores
- **Memory**: 8GB+ RAM
- **Storage**: 100GB+ SSD
- **Network**: 1 Gbps

#### High-Scale Deployment
- **CPU**: 8+ cores
- **Memory**: 16GB+ RAM
- **Storage**: 500GB+ SSD with backup
- **Network**: 10 Gbps with redundancy

### Scaling Strategies

#### Vertical Scaling (Scale Up)
```yaml
# Increase resources per instance
resources:
  requests:
    cpu: 2000m
    memory: 4Gi
  limits:
    cpu: 4000m
    memory: 8Gi
```

#### Horizontal Scaling (Scale Out)
```yaml
# Deploy multiple instances
replicas: 5
strategy:
  type: RollingUpdate
  rollingUpdate:
    maxSurge: 1
    maxUnavailable: 0
```

#### Auto-scaling Configuration
```yaml
# Kubernetes HPA
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: greenlang-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: greenlang
  minReplicas: 3
  maxReplicas: 20
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
```

### Capacity Planning Formula

```python
def calculate_capacity(
    expected_rps: float,
    avg_processing_time: float,
    target_cpu_utilization: float = 0.7,
    safety_margin: float = 0.2
) -> dict:
    """
    Calculate required capacity for GreenLang deployment

    Args:
        expected_rps: Expected requests per second
        avg_processing_time: Average processing time per request (seconds)
        target_cpu_utilization: Target CPU utilization (0.0-1.0)
        safety_margin: Safety margin for unexpected load spikes
    """

    # Calculate base capacity needed
    concurrent_requests = expected_rps * avg_processing_time
    required_capacity = concurrent_requests / target_cpu_utilization

    # Add safety margin
    total_capacity = required_capacity * (1 + safety_margin)

    return {
        "concurrent_requests": concurrent_requests,
        "required_instances": max(1, int(total_capacity)),
        "cpu_cores_per_instance": 2,  # Minimum recommended
        "memory_gb_per_instance": 4,  # Minimum recommended
        "total_cpu_cores": max(2, int(total_capacity * 2)),
        "total_memory_gb": max(4, int(total_capacity * 4))
    }

# Example calculation
capacity = calculate_capacity(
    expected_rps=100,
    avg_processing_time=0.1,  # 100ms
    target_cpu_utilization=0.7
)
print(f"Required instances: {capacity['required_instances']}")
print(f"Total CPU cores: {capacity['total_cpu_cores']}")
print(f"Total memory: {capacity['total_memory_gb']}GB")
```

## Performance Monitoring Dashboard

### Key Performance Indicators (KPIs)

1. **Response Time**: P50, P95, P99 latencies
2. **Throughput**: Requests per second, operations per second
3. **Error Rate**: Failed requests percentage
4. **Resource Utilization**: CPU, memory, disk usage
5. **Availability**: Uptime percentage

### Grafana Dashboard Queries

```promql
# Response time percentiles
histogram_quantile(0.95,
  rate(greenlang_pipeline_duration_seconds_bucket[5m])
)

# Throughput
rate(greenlang_pipeline_executions_total[5m])

# Error rate
rate(greenlang_errors_total[5m]) /
rate(greenlang_pipeline_executions_total[5m]) * 100

# Memory usage
greenlang_memory_usage_bytes / 1024 / 1024

# CPU usage
greenlang_cpu_usage_percent
```

---

## Conclusion

This performance guide provides the foundation for operating GreenLang v0.2.0 in production environments. Regular monitoring, testing, and optimization using these guidelines will ensure optimal performance and reliability.

For additional support or questions about performance optimization, please refer to the [GreenLang documentation](../README.md) or contact the development team.

**Last Updated**: September 2024
**Version**: 0.2.0
**Document Version**: 1.0