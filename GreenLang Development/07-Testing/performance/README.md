# GreenLang Performance Testing

Comprehensive performance and scale testing infrastructure for GreenLang Phase 3.

## Quick Start

### Run All Tests
```bash
python run_performance_tests.py
```

### Run Individual Test Suites

**Load Testing:**
```bash
python tests/performance/load_testing.py
```

**Profiling:**
```bash
python tests/performance/profiling.py
```

**Regression Testing:**
```bash
python tests/performance/regression_tests.py
```

**Pytest Tests:**
```bash
pytest tests/performance/test_concurrent_execution.py -v
pytest tests/performance/test_resource_usage.py -v
```

## Test Suites

### 1. Load Testing (`load_testing.py`)

Test concurrent execution and measure system performance under load.

**Features:**
- Multiple load patterns (CONSTANT, RAMP_UP, SPIKE, STEP)
- Concurrent execution testing (10, 100, 1000+)
- Comprehensive metrics (latency, throughput, error rate)
- Resource monitoring (CPU, memory)
- JSON/CSV results export

**Example:**
```python
from tests.performance.load_testing import LoadTester

async def main():
    tester = LoadTester()

    # Test 10 concurrent executions
    results = await tester.run_concurrent_load_test(num_concurrent=10)
    tester.save_results_json(results)
```

### 2. Performance Profiling (`profiling.py`)

Identify performance bottlenecks and optimize agent execution.

**Features:**
- CPU profiling (cProfile)
- Memory profiling (tracemalloc)
- Bottleneck detection
- Memory leak detection
- Profiling decorators

**Example:**
```python
from tests.performance.profiling import PerformanceProfiler

async def main():
    profiler = PerformanceProfiler()

    # Profile agent execution
    report = await profiler.profile_agent_execution(
        agent_name="FuelAgentAI",
        num_iterations=100
    )
    profiler.print_report(report)
```

### 3. Regression Testing (`regression_tests.py`)

Detect performance regressions and validate SLOs.

**Features:**
- Baseline management
- Automated regression detection
- Performance SLOs validation
- Historical tracking

**Example:**
```python
from tests.performance.regression_tests import RegressionTester

async def main():
    tester = RegressionTester()

    # Create baseline
    await tester.run_regression_tests(update_baseline=True)

    # Test against baseline
    suite = await tester.run_regression_tests(update_baseline=False)
    print(f"All passed: {suite.all_passed}")
```

### 4. Concurrent Execution Tests (`test_concurrent_execution.py`)

pytest-based tests for concurrent execution validation.

**Tests:**
- `test_10_concurrent_executions`
- `test_100_concurrent_executions`
- `test_1000_lightweight_operations`
- `test_thread_safety_shared_agent`
- `test_async_speedup_vs_sequential`

**Run:**
```bash
pytest tests/performance/test_concurrent_execution.py -v -s
```

### 5. Resource Usage Tests (`test_resource_usage.py`)

Monitor and validate resource consumption.

**Tests:**
- `test_cpu_usage_under_load`
- `test_memory_usage_patterns`
- `test_memory_leak_detection`
- `test_thread_count_stability`
- `test_resource_usage_under_stress`

**Run:**
```bash
pytest tests/performance/test_resource_usage.py -v -s
```

## Performance Metrics

### Latency
- **p50 (median)**: Median response time
- **p95**: 95th percentile (SLO: <500ms)
- **p99**: 99th percentile (SLO: <1000ms)

### Throughput
- **RPS**: Requests per second
- **Target**: >5 RPS for 10 concurrent

### Resource Usage
- **CPU**: Peak and average (target: <80%)
- **Memory**: Peak and average (target: <500 MB)

### Reliability
- **Error Rate**: Percentage of failed requests (target: <1%)
- **Success Rate**: Percentage of successful requests

## Test Results

See `PERFORMANCE_TEST_RESULTS.md` for latest results.

**Latest Results (November 2025):**
- ✓ p95 Latency: 208-220ms (target: <500ms)
- ✓ Throughput: 47-223 RPS (target: >5 RPS)
- ✓ Error Rate: 0% (target: <1%)
- ✓ CPU Usage: 25.8% (target: <80%)
- ✓ Memory: 61.7 MB (target: <500 MB)

## Documentation

- **[Load Testing Guide](../../docs/performance/load-testing-guide.md)**: How to run load tests
- **[Profiling Guide](../../docs/performance/profiling-guide.md)**: How to profile performance
- **[Performance Tuning](../../docs/performance/performance-tuning.md)**: Optimization strategies
- **[Benchmark Results](../../docs/performance/benchmark-results.md)**: Latest benchmark results

## Directory Structure

```
tests/performance/
├── README.md                      # This file
├── __init__.py                    # Package init
├── load_testing.py                # Load testing framework
├── profiling.py                   # Performance profiling
├── regression_tests.py            # Regression testing
├── test_concurrent_execution.py   # Concurrent tests (pytest)
├── test_resource_usage.py         # Resource tests (pytest)
└── results/                       # Test results
    ├── test_10_concurrent.json
    ├── test_50_concurrent.json
    ├── fuel_agent_profile.txt
    └── baselines/
        └── single_agent_execution.json
```

## Service Level Objectives (SLOs)

| Metric | SLO | Current | Status |
|--------|-----|---------|--------|
| p95 Latency | < 500ms | 208-220ms | ✓ PASS |
| p99 Latency | < 1000ms | 208-220ms | ✓ PASS |
| Error Rate | < 1% | 0% | ✓ PASS |
| Throughput | > 5 RPS | 47-223 RPS | ✓ PASS |
| CPU Usage | < 80% | 25.8% | ✓ PASS |
| Memory | < 500 MB | 61.7 MB | ✓ PASS |

## Best Practices

### 1. Run Tests Before Deployment
```bash
python run_performance_tests.py
```

### 2. Create Baselines
```bash
python -m tests.performance.regression_tests
# Creates baseline in results/baselines/
```

### 3. Monitor Regressions
```bash
pytest tests/performance/ -v
# Fails if regressions detected
```

### 4. Profile Before Optimizing
```bash
python tests/performance/profiling.py
# Identifies actual bottlenecks
```

### 5. Use Decorators for Profiling
```python
from tests.performance.profiling import profile, memory_profile

@profile
async def my_function():
    # Function is automatically profiled
    pass

@memory_profile
async def memory_intensive():
    # Memory usage is tracked
    pass
```

## Troubleshooting

### Tests Timing Out
- Increase timeout in agent configuration
- Check network connectivity
- Verify API keys are set

### Inconsistent Results
- Run multiple iterations
- Check for background processes
- Use dedicated test environment

### High Memory Usage
- Check for memory leaks (run memory tests)
- Review cache sizes
- Verify cleanup in `__aexit__`

### Low Throughput
- Profile to find bottlenecks
- Check for blocking I/O
- Verify async implementation

## Contributing

When adding new tests:

1. **Follow naming convention**: `test_*.py` for pytest
2. **Add documentation**: Include docstrings
3. **Update baselines**: Run with `update_baseline=True`
4. **Update this README**: Document new tests

## References

- [GreenLang Documentation](../../docs/)
- [Async Agent Base](../../greenlang/agents/async_agent_base.py)
- [Benchmarks](../../benchmarks/)
- [Performance Results](../../PERFORMANCE_TEST_RESULTS.md)

---

**Status:** Production Ready ✓
**Last Updated:** November 2025
