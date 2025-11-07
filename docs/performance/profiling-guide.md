# Performance Profiling Guide

## Overview

GreenLang provides comprehensive profiling tools to identify performance bottlenecks and optimize agent execution. This guide covers CPU profiling, memory profiling, and I/O profiling.

## Quick Start

### Basic Profiling

```python
from tests.performance.profiling import PerformanceProfiler

async def main():
    profiler = PerformanceProfiler()

    # Profile agent execution
    report = await profiler.profile_agent_execution(
        agent_name="FuelAgentAI",
        num_iterations=100
    )

    # Print results
    profiler.print_report(report)

    # Save to file
    profiler.save_report("fuel_agent_profile.txt", report)

asyncio.run(main())
```

### Running from Command Line

```bash
# Run standalone profiling
python tests/performance/profiling.py

# Profile specific agent
pytest tests/performance/test_concurrent_execution.py::test_10_concurrent_executions -v -s
```

## Profiling Types

### 1. CPU Profiling

Identifies CPU bottlenecks using cProfile.

```python
from tests.performance.profiling import PerformanceProfiler

profiler = PerformanceProfiler()

report = await profiler.profile_agent_execution(
    agent_name="FuelAgentAI",
    num_iterations=100,
    enable_cpu=True,
    enable_memory=False,
    enable_io=False
)
```

**CPU Profile Output:**
```
CPU PROFILE:
  Total Calls: 15,234
  Total Time: 12.45s

  Top Functions (by cumulative time):
    1. greenlang/agents/fuel_agent_ai_async.py:execute_impl_async
        Time: 8.234s  Calls: 100
    2. greenlang/llm/openai_client.py:generate_async
        Time: 7.123s  Calls: 100
    3. json.dumps
        Time: 0.345s  Calls: 1,234
```

### 2. Memory Profiling

Tracks memory allocation and identifies memory leaks using tracemalloc.

```python
report = await profiler.profile_agent_execution(
    agent_name="FuelAgentAI",
    num_iterations=100,
    enable_cpu=False,
    enable_memory=True,
    enable_io=False
)
```

**Memory Profile Output:**
```
MEMORY PROFILE:
  Peak Memory: 145.23 MB
  Current Memory: 132.45 MB

  Top Allocations:
    1. 45.23 MB (1,234 blocks)
       greenlang/agents/fuel_agent_ai_async.py:245
    2. 23.45 MB (567 blocks)
       greenlang/llm/openai_client.py:123
```

### 3. I/O Profiling

Measures I/O operations (read/write).

```python
report = await profiler.profile_agent_execution(
    agent_name="FuelAgentAI",
    num_iterations=100,
    enable_cpu=False,
    enable_memory=False,
    enable_io=True
)
```

**I/O Profile Output:**
```
I/O PROFILE:
  Read: 1,234,567 bytes (123 ops)
  Write: 234,567 bytes (45 ops)
  I/O Time: 0.234s
```

### 4. Combined Profiling

Profile all aspects simultaneously.

```python
report = await profiler.profile_agent_execution(
    agent_name="FuelAgentAI",
    num_iterations=100,
    enable_cpu=True,
    enable_memory=True,
    enable_io=True
)
```

## Profiling Decorators

### @profile Decorator

Profile individual functions:

```python
from tests.performance.profiling import profile

@profile
async def my_expensive_function():
    # Function code here
    result = await some_async_operation()
    return result
```

### @memory_profile Decorator

Track memory usage of specific functions:

```python
from tests.performance.profiling import memory_profile

@memory_profile
async def my_memory_intensive_function():
    # Function code here
    large_data = [0] * 1000000
    return process(large_data)
```

## Bottleneck Detection

The profiler automatically identifies bottlenecks:

```python
report = await profiler.profile_agent_execution(
    agent_name="FuelAgentAI",
    num_iterations=100
)

# Bottlenecks are automatically identified
for bottleneck in report.bottlenecks:
    print(f"- {bottleneck}")
```

**Example Output:**
```
BOTTLENECKS IDENTIFIED:
  - CPU: greenlang/llm/openai_client.py:generate_async took 7.12s (100 calls)
  - Memory: 45.23 MB allocated at greenlang/agents/base.py:156
```

## Optimization Recommendations

The profiler provides automatic optimization suggestions:

```python
for recommendation in report.recommendations:
    print(f"{recommendation}")
```

**Example Recommendations:**
```
OPTIMIZATION RECOMMENDATIONS:
  1. Consider caching expensive computations or using async I/O
  2. JSON serialization is a bottleneck - consider using msgpack or protobuf
  3. High memory usage detected - consider streaming or chunking data
```

## Concurrent Execution Profiling

Profile concurrent workloads:

```python
report = await profiler.profile_concurrent_execution(
    num_concurrent=50
)
```

This helps identify:
- Thread-safety issues
- Lock contention
- Resource contention
- Concurrency bottlenecks

## Memory Leak Detection

### Automatic Detection

```python
from tests.performance.profiling import MemoryProfiler

mem_profiler = MemoryProfiler()
mem_profiler.start()

# Run operations
for _ in range(100):
    await agent.run_async(test_input)

mem_profiler.stop()

# Check for leaks
leak_warnings = mem_profiler.check_memory_leaks()
for warning in leak_warnings:
    print(f"WARNING: {warning}")
```

### Manual Inspection

```python
# Print top allocations
mem_profiler.print_top_allocations(top_n=10)
```

**Output:**
```
TOP 10 MEMORY ALLOCATIONS

#1: greenlang/agents/fuel_agent_ai_async.py:156
  Size: 23.45 MB (1,234 blocks)
  Code: self.cache = {}

#2: greenlang/llm/openai_client.py:89
  Size: 12.34 MB (567 blocks)
  Code: response = await self.client.chat.completions.create(...)
```

## Profiling Best Practices

### 1. Profile with Realistic Data

```python
# Use production-like data volumes
test_input = {
    "fuel_type": "natural_gas",
    "amount": 10000,  # Realistic amount
    "unit": "therms",
    "country": "US"
}
```

### 2. Run Multiple Iterations

```python
# Run enough iterations for stable results
report = await profiler.profile_agent_execution(
    num_iterations=100  # Minimum recommended
)
```

### 3. Profile Different Scenarios

```python
# Profile single execution
report1 = await profiler.profile_agent_execution(num_iterations=1)

# Profile sequential execution
report2 = await profiler.profile_agent_execution(num_iterations=100)

# Profile concurrent execution
report3 = await profiler.profile_concurrent_execution(num_concurrent=50)
```

### 4. Compare Before and After

```python
# Before optimization
report_before = await profiler.profile_agent_execution(num_iterations=100)

# ... apply optimizations ...

# After optimization
report_after = await profiler.profile_agent_execution(num_iterations=100)

# Compare
speedup = report_before.cpu_profile.total_time_seconds / report_after.cpu_profile.total_time_seconds
print(f"Speedup: {speedup:.2f}x")
```

### 5. Focus on Top Bottlenecks

```python
# Focus on top 5 functions consuming most time
cpu_profiler.print_stats(top_n=5)
```

## Interpreting Results

### CPU Profile

**Good Signs:**
- ✓ Time spent in your code (not in libraries)
- ✓ No single function dominating
- ✓ Async I/O not blocking

**Warning Signs:**
- ⚠ Single function taking >50% of time
- ⚠ Serialization (json.dumps) taking significant time
- ⚠ Time spent in blocking I/O

### Memory Profile

**Good Signs:**
- ✓ Memory usage stable across iterations
- ✓ Memory released after operations
- ✓ Peak memory < 500MB for single agent

**Warning Signs:**
- ⚠ Memory growing linearly with iterations
- ⚠ Large allocations not being freed
- ⚠ Peak memory > 1GB

### Bottleneck Analysis

**Common Bottlenecks:**

1. **LLM API Calls** (Expected)
   - Solution: Batch requests, use async

2. **JSON Serialization**
   - Solution: Use msgpack or protobuf

3. **Database Queries**
   - Solution: Connection pooling, query optimization

4. **Blocking I/O**
   - Solution: Convert to async I/O

## Advanced Profiling

### Line-by-Line Profiling

```python
import cProfile
import pstats

profiler = cProfile.Profile()
profiler.enable()

# Your code here
await agent.run_async(test_input)

profiler.disable()

# Print stats sorted by cumulative time
stats = pstats.Stats(profiler)
stats.sort_stats('cumulative')
stats.print_stats(20)
```

### Call Graph Generation

```python
# Requires graphviz
import pstats
from gprof2dot import gprof2dot

stats = pstats.Stats(profiler)
stats.dump_stats('profile.pstats')

# Generate call graph
# gprof2dot -f pstats profile.pstats | dot -Tpng -o callgraph.png
```

### Memory Snapshots

```python
import tracemalloc

tracemalloc.start()

# Take snapshot before
snapshot1 = tracemalloc.take_snapshot()

# Run operations
await agent.run_async(test_input)

# Take snapshot after
snapshot2 = tracemalloc.take_snapshot()

# Compare
top_stats = snapshot2.compare_to(snapshot1, 'lineno')
for stat in top_stats[:10]:
    print(stat)
```

## Example: Full Profiling Session

```python
import asyncio
from tests.performance.profiling import PerformanceProfiler

async def full_profiling_session():
    """Complete profiling workflow."""
    profiler = PerformanceProfiler()

    # 1. Profile baseline
    print("Step 1: Profiling baseline...")
    baseline = await profiler.profile_agent_execution(
        agent_name="FuelAgentAI",
        num_iterations=100
    )
    profiler.save_report("baseline_profile.txt", baseline)

    # 2. Profile concurrent execution
    print("Step 2: Profiling concurrent execution...")
    concurrent = await profiler.profile_concurrent_execution(
        num_concurrent=50
    )
    profiler.save_report("concurrent_profile.txt", concurrent)

    # 3. Compare results
    print("\nComparison:")
    print(f"Baseline p95: {baseline.cpu_profile.total_time_seconds:.2f}s")
    print(f"Concurrent overhead: {concurrent.cpu_profile.total_time_seconds / baseline.cpu_profile.total_time_seconds:.2f}x")

    # 4. Check for issues
    if baseline.bottlenecks:
        print("\nBottlenecks found:")
        for bottleneck in baseline.bottlenecks:
            print(f"  - {bottleneck}")

    # 5. Review recommendations
    if baseline.recommendations:
        print("\nRecommendations:")
        for i, rec in enumerate(baseline.recommendations, 1):
            print(f"  {i}. {rec}")

if __name__ == "__main__":
    asyncio.run(full_profiling_session())
```

## Continuous Profiling

### Automated Profiling in CI/CD

```yaml
# .github/workflows/profile.yml
name: Performance Profiling

on:
  pull_request:
    branches: [main]

jobs:
  profile:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2

      - name: Profile performance
        run: python tests/performance/profiling.py

      - name: Upload profile reports
        uses: actions/upload-artifact@v2
        with:
          name: profile-reports
          path: tests/performance/results/*.txt
```

## Troubleshooting

### Issue: Profiling adds significant overhead
**Solution**: Use sampling profiler or reduce sampling frequency

### Issue: Memory profiler shows no growth but RSS increases
**Solution**: Check for external process memory, OS caching

### Issue: Can't identify bottleneck
**Solution**: Use line profiler (line_profiler) for detailed analysis

## References

- [Load Testing Guide](load-testing-guide.md)
- [Performance Tuning Guide](performance-tuning.md)
- [Python cProfile Documentation](https://docs.python.org/3/library/profile.html)
- [tracemalloc Documentation](https://docs.python.org/3/library/tracemalloc.html)
