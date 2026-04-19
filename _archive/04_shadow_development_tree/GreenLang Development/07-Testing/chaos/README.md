# Chaos Engineering Tests for Process Heat Agents

## Overview

This test suite validates the resilience, fault-tolerance, and automatic recovery capabilities of Process Heat agents under various infrastructure failure scenarios. The tests use chaos engineering principles to ensure agents maintain SLOs (Service Level Objectives) even when infrastructure is degraded.

## Test Coverage

### 1. Agent Failover (TestAgentFailover)

**Tests pod failure and backup takeover scenarios:**

- **test_primary_pod_failure_backup_takeover**
  - Kill primary agent pod
  - Verify backup becomes active within 5 seconds
  - Validate 99%+ availability during failover
  - Ensure no request loss

- **test_cascading_pod_failures**
  - Kill 2 out of 3 pods
  - Verify single pod handles traffic
  - Measure graceful degradation (70% throughput)
  - Validate error rate < 1%

### 2. Database Resilience (TestDatabaseResilience)

**Tests graceful degradation when database is unavailable:**

- **test_database_connection_loss_retry**
  - Simulate database connection failure
  - Verify exponential backoff retry logic (3^n pattern)
  - Validate retries at 0.1s, 0.3s, 0.9s (testing timescale)
  - Confirm eventual recovery

- **test_graceful_cache_fallback**
  - Database unavailable 100% of requests
  - Verify agent uses cached emission factors
  - Validate cache accuracy > 98.5%
  - Measure cache hit rate (99%+)

### 3. High Latency and Timeouts (TestHighLatencyAndTimeouts)

**Tests timeout handling under high latency conditions:**

- **test_latency_timeout_handling**
  - Inject 10 second latency
  - Verify requests timeout after 5 seconds
  - Confirm circuit breaker activates on 3 consecutive timeouts
  - Validate automatic retry after cooldown

- **test_request_queuing_under_latency**
  - Inject 2 second latency
  - Submit 100 concurrent requests
  - Verify queue depth < 500 requests
  - Ensure FIFO ordering maintained

### 4. Memory and Resource Pressure (TestMemoryAndResourcePressure)

**Tests behavior under memory and CPU constraints:**

- **test_memory_exhaustion_oom_handling**
  - Apply 800MB memory pressure
  - Verify agent detects memory limit before hitting 85% utilization
  - Confirm graceful shutdown (not crash)
  - Validate recovery after pressure released

- **test_cpu_saturation_performance_degradation**
  - Apply 4-core CPU stress
  - Measure latency increase under load (expect 16x slower)
  - Verify P99 latency < 10 seconds
  - Confirm zero requests dropped

## Test Execution

### Run All Chaos Tests

```bash
pytest tests/chaos/ -v -m chaos
```

### Run Specific Test Categories

```bash
# Agent failover tests
pytest tests/chaos/ -v -m chaos_failover

# Database resilience tests
pytest tests/chaos/ -v -m chaos_database

# Latency/timeout tests
pytest tests/chaos/ -v -m chaos_latency

# Resource pressure tests
pytest tests/chaos/ -v -m chaos_resource
```

### Run with Custom Options

```bash
# Show detailed logging
pytest tests/chaos/chaos_tests.py -v -s --log-cli-level=DEBUG

# Run specific test
pytest tests/chaos/chaos_tests.py::TestAgentFailover::test_primary_pod_failure_backup_takeover -v

# Run with timeout (300 seconds)
pytest tests/chaos/ -v --timeout=300
```

## ChaosTestRunner API

### Chaos Injection Methods

```python
from tests.chaos import ChaosTestRunner, SteadyStateMetrics

runner = ChaosTestRunner(environment="test")

# Inject latency
event = runner.inject_latency(
    service="process_heat_agent",
    delay_ms=10000,  # 10 second latency
    duration_s=30    # For 30 seconds
)

# Inject failures
event = runner.inject_failure(
    service="postgresql",
    error_rate_percent=100.0,  # 100% failure rate
    duration_s=60
)

# Kill pods
events = runner.kill_pod(
    deployment="process-heat-agent",
    count=2  # Kill 2 pods
)

# Network partition
event = runner.network_partition(
    service_a="process_heat_agent",
    service_b="postgresql"
)

# CPU stress
event = runner.cpu_stress(
    deployment="process-heat-agent",
    cores=4,        # Stress 4 CPU cores
    duration_s=60   # For 60 seconds
)

# Memory pressure
event = runner.memory_pressure(
    deployment="process-heat-agent",
    mb=800,         # Allocate 800MB
    duration_s=30   # For 30 seconds
)
```

### Monitoring and Validation

```python
# Get active chaos events
active_chaos = runner.get_active_chaos()

# Stop all chaos
runner.stop_all_chaos()

# Validate steady-state metrics
metrics = {
    "availability_percent": 99.5,
    "latency_ms": 4500,
    "error_rate_percent": 0.5
}

steady_state = SteadyStateMetrics(
    max_latency_ms=5000.0,
    min_availability_percent=99.0,
    max_error_rate_percent=1.0
)

passed, errors = runner.validate_steady_state(metrics, steady_state)
```

## Expected SLOs

During normal operation (without chaos):

| Metric | Target |
|--------|--------|
| Availability | 99.99% |
| P50 Latency | 50ms |
| P99 Latency | 500ms |
| Throughput | 1000+ req/s |
| Error Rate | <0.01% |
| Memory Usage | <200MB |

During chaos injection (degraded SLOs):

| Metric | Minimum Acceptable |
|--------|-------------------|
| Availability | 99.0% |
| P99 Latency | 5000ms |
| Throughput | 100+ req/s |
| Error Rate | <1.0% |
| Memory Usage | <1000MB |

## Test Result Reporting

### JSON Report

```python
from tests.chaos.conftest import ChaosTestReporter

results = [...]  # List of ChaosTestResult objects
ChaosTestReporter.generate_json_report(results, Path("chaos_report.json"))
```

### HTML Report

```python
ChaosTestReporter.generate_html_report(results, Path("chaos_report.html"))
```

## Failure Handling

All tests include automatic rollback on failure:

1. **Detect Failure**: Test validates metrics against SLOs
2. **Log Errors**: All errors logged with timestamps
3. **Cleanup**: `chaos_runner.stop_all_chaos()` called automatically
4. **Report**: Detailed failure report generated

## Integration with CI/CD

### GitHub Actions

```yaml
- name: Run Chaos Tests
  run: |
    pytest tests/chaos/ -v -m chaos \
      --junitxml=chaos-results.xml \
      --html=chaos-report.html

- name: Upload Results
  uses: actions/upload-artifact@v2
  with:
    name: chaos-test-results
    path: chaos-report.html
```

### Pytest Configuration

See `pytest.ini` for:
- Test discovery patterns
- Marker definitions
- Timeout configuration
- Logging setup

## Development Notes

### Adding New Chaos Tests

1. Create test method in appropriate TestClass
2. Mark with `@pytest.mark.chaos` and specific category
3. Use `ChaosTestRunner` to inject chaos
4. Validate metrics using `validate_steady_state()`
5. Ensure cleanup via `stop_all_chaos()`

### Mock Data Generation

```python
@pytest.fixture
def test_metrics():
    return {
        "availability_percent": 99.5,
        "latency_ms": 1200,
        "error_rate_percent": 0.5,
        "throughput_rps": 500
    }
```

## Troubleshooting

### Test Timeout

If tests timeout (default 300s):
- Increase `--timeout` parameter
- Reduce `duration_s` in chaos injection
- Check for hanging threads with `threading.enumerate()`

### Memory Errors

If memory pressure tests fail:
- Verify sufficient available memory on host
- Reduce `mb` parameter in `memory_pressure()`
- Check for memory leaks in agent code

### Latency Tests Failing

If latency injection not working:
- Verify `inject_latency()` called successfully
- Check that mock/patch is active
- Confirm timeout < injected latency

## References

- Chaos-Mesh Documentation: https://chaos-mesh.org/
- Gremlin Chaos Engineering: https://www.gremlin.com/
- Netflix Chaos Engineering: https://netflixtechblog.com/

## Authors

- GreenLang Test Engineering Team
- Date: December 2025
