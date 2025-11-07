# Load Testing Guide

## Overview

GreenLang provides comprehensive load testing capabilities to validate production readiness. The load testing framework simulates real-world concurrent loads and measures system performance under stress.

## Quick Start

### Running Basic Load Tests

```python
from tests.performance.load_testing import LoadTester, LoadPattern

async def main():
    tester = LoadTester()

    # Test 10 concurrent executions
    results = await tester.run_concurrent_load_test(num_concurrent=10)

    # Save results
    tester.save_results_json(results)

asyncio.run(main())
```

### Running from Command Line

```bash
# Run standalone load tests
python tests/performance/load_testing.py

# Run specific test scenarios
pytest tests/performance/test_concurrent_execution.py -v
```

## Load Patterns

The framework supports multiple load patterns:

### 1. Constant Load
Maintains constant RPS throughout the test.

```python
results = await tester.run_load_test(
    pattern=LoadPattern.CONSTANT,
    target_rps=50,
    duration_seconds=60
)
```

### 2. Ramp-Up Load
Gradually increases load from 0 to target RPS.

```python
results = await tester.run_load_test(
    pattern=LoadPattern.RAMP_UP,
    target_rps=100,
    duration_seconds=120
)
```

### 3. Spike Load
Simulates sudden traffic bursts.

```python
results = await tester.run_load_test(
    pattern=LoadPattern.SPIKE,
    target_rps=200,
    duration_seconds=60
)
```

### 4. Step Load
Increases load in steps.

```python
results = await tester.run_load_test(
    pattern=LoadPattern.STEP,
    target_rps=100,
    duration_seconds=90
)
```

## Concurrent Execution Testing

### Small Scale (10 concurrent)
```python
results = await tester.run_concurrent_load_test(num_concurrent=10)
```

### Medium Scale (100 concurrent)
```python
results = await tester.run_concurrent_load_test(num_concurrent=100)
```

### Large Scale (1000+ concurrent)
```python
results = await tester.run_concurrent_load_test(num_concurrent=1000)
```

## Metrics Collected

### Request Metrics
- **Total Requests**: Total number of requests executed
- **Successful Requests**: Requests that completed successfully
- **Failed Requests**: Requests that failed or timed out
- **Error Rate**: Percentage of failed requests

### Latency Metrics (milliseconds)
- **Min**: Minimum request latency
- **Mean**: Average request latency
- **Median (p50)**: 50th percentile latency
- **p95**: 95th percentile latency
- **p99**: 99th percentile latency
- **Max**: Maximum request latency

### Throughput Metrics
- **Target RPS**: Target requests per second
- **Actual RPS**: Achieved requests per second
- **Duration**: Total test duration

### Resource Metrics
- **Peak CPU**: Peak CPU usage during test
- **Avg CPU**: Average CPU usage
- **Peak Memory**: Peak memory consumption (MB)
- **Avg Memory**: Average memory consumption (MB)

## Results Output

### JSON Format
```python
tester.save_results_json(results, "load_test_results.json")
```

Output structure:
```json
{
  "test_name": "concurrent_100",
  "pattern": "concurrent",
  "total_requests": 100,
  "successful_requests": 98,
  "error_rate": 0.02,
  "latency_ms": {
    "min": 145.23,
    "mean": 287.45,
    "median": 265.12,
    "p95": 456.78,
    "p99": 523.45
  },
  "throughput": {
    "target_rps": 100,
    "actual_rps": 45.2
  }
}
```

### CSV Format
```python
tester.save_results_csv(results, "load_test_requests.csv")
```

Contains detailed per-request data:
- request_id
- start_time
- end_time
- duration_ms
- success
- error

## Performance SLOs

### Recommended Service Level Objectives

| Metric | SLO | Rationale |
|--------|-----|-----------|
| p95 Latency | < 500ms | Ensures responsive user experience |
| p99 Latency | < 1000ms | Handles worst-case scenarios |
| Error Rate | < 1% | Maintains high reliability |
| Throughput (10 concurrent) | > 5 RPS | Validates async efficiency |

## Best Practices

### 1. Start Small, Scale Up
```python
# Start with small load
await tester.run_concurrent_load_test(num_concurrent=10)

# Gradually increase
await tester.run_concurrent_load_test(num_concurrent=50)
await tester.run_concurrent_load_test(num_concurrent=100)
```

### 2. Use Realistic Test Data
```python
# Vary inputs to simulate real usage
test_inputs = [
    {"fuel_type": "natural_gas", "amount": 1000},
    {"fuel_type": "diesel", "amount": 500},
    {"fuel_type": "electricity", "amount": 2000},
]
```

### 3. Monitor System Resources
```python
# Enable resource monitoring
from tests.performance.load_testing import ResourceMonitor

monitor = ResourceMonitor()
monitor.start()

# Run tests...

monitor.stop()
print(f"Peak CPU: {monitor.peak_cpu}%")
print(f"Peak Memory: {monitor.peak_memory_mb} MB")
```

### 4. Test Under Different Conditions
- Normal load
- Peak load
- Sustained load
- Burst load

### 5. Establish Baselines
```python
# Create baseline
baseline_results = await tester.run_concurrent_load_test(num_concurrent=100)
tester.save_results_json(baseline_results, "baseline.json")

# Compare against baseline in future tests
```

## Interpreting Results

### Good Performance Indicators
- ✓ p95 latency < 500ms
- ✓ Error rate < 1%
- ✓ Actual RPS close to target RPS
- ✓ Memory usage stable (no growth)
- ✓ CPU usage < 80%

### Warning Signs
- ⚠ p95 latency > 1000ms
- ⚠ Error rate > 5%
- ⚠ Actual RPS << target RPS
- ⚠ Memory growing continuously
- ⚠ CPU pegged at 100%

### Action Items for Poor Performance

**High Latency:**
- Profile CPU bottlenecks
- Check database query performance
- Optimize serialization
- Consider caching

**High Error Rate:**
- Check logs for error patterns
- Verify resource limits (connections, file descriptors)
- Review timeout configurations
- Check for race conditions

**Low Throughput:**
- Verify async implementation
- Check for blocking I/O
- Review connection pooling
- Consider horizontal scaling

## Example Test Suite

```python
import asyncio
from tests.performance.load_testing import LoadTester, LoadPattern

async def run_comprehensive_load_tests():
    """Run comprehensive load test suite."""
    tester = LoadTester()

    # Test 1: Baseline (10 concurrent)
    print("Test 1: Baseline Load (10 concurrent)")
    r1 = await tester.run_concurrent_load_test(num_concurrent=10)
    tester.save_results_json(r1, "test_10_concurrent.json")

    # Test 2: Medium load (50 concurrent)
    print("Test 2: Medium Load (50 concurrent)")
    r2 = await tester.run_concurrent_load_test(num_concurrent=50)
    tester.save_results_json(r2, "test_50_concurrent.json")

    # Test 3: High load (100 concurrent)
    print("Test 3: High Load (100 concurrent)")
    r3 = await tester.run_concurrent_load_test(num_concurrent=100)
    tester.save_results_json(r3, "test_100_concurrent.json")

    # Test 4: Ramp-up pattern
    print("Test 4: Ramp-Up Load Pattern")
    r4 = await tester.run_load_test(
        pattern=LoadPattern.RAMP_UP,
        target_rps=50,
        duration_seconds=60
    )
    tester.save_results_json(r4, "test_ramp_up.json")

    # Test 5: Spike pattern
    print("Test 5: Spike Load Pattern")
    r5 = await tester.run_load_test(
        pattern=LoadPattern.SPIKE,
        target_rps=100,
        duration_seconds=60
    )
    tester.save_results_json(r5, "test_spike.json")

    print("\nAll load tests complete!")
    print(f"Results saved to: {tester.results_dir}")

if __name__ == "__main__":
    asyncio.run(run_comprehensive_load_tests())
```

## Continuous Integration

### GitHub Actions Example

```yaml
name: Load Tests

on:
  pull_request:
    branches: [main]
  schedule:
    - cron: '0 0 * * 0'  # Weekly

jobs:
  load-test:
    runs-on: ubuntu-latest

    steps:
      - uses: actions/checkout@v2

      - name: Set up Python
        uses: actions/setup-python@v2
        with:
          python-version: 3.11

      - name: Install dependencies
        run: |
          pip install -e .
          pip install pytest psutil

      - name: Run load tests
        run: |
          python tests/performance/load_testing.py

      - name: Upload results
        uses: actions/upload-artifact@v2
        with:
          name: load-test-results
          path: tests/performance/results/
```

## Troubleshooting

### Issue: Tests timing out
**Solution**: Increase timeout in agent configuration

### Issue: Connection errors
**Solution**: Check connection pool limits, increase max connections

### Issue: Inconsistent results
**Solution**: Run multiple iterations, take median results

### Issue: Resource exhaustion
**Solution**: Monitor system resources, reduce concurrency level

## References

- [Performance Profiling Guide](profiling-guide.md)
- [Performance Tuning Guide](performance-tuning.md)
- [Benchmark Results](benchmark-results.md)
