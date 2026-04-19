# GreenLang Framework Performance Benchmarking

This directory contains comprehensive performance benchmarking tools for the GreenLang framework.

## Overview

The benchmarking suite tests all major framework components to ensure they meet performance targets and identify optimization opportunities.

### Components Benchmarked

1. **Base Agent Performance**
   - Simple agent execution time
   - Framework overhead vs direct function calls
   - Metrics collection overhead
   - Hook execution overhead

2. **Data Processor Performance**
   - Batch processing throughput (records/sec)
   - Sequential vs parallel processing
   - Memory usage during batch processing
   - Different batch sizes (10, 100, 1000, 10000)

3. **Calculator Performance**
   - Calculation execution time
   - Cache hit/miss performance
   - Cache lookup overhead
   - Deterministic execution overhead

4. **Validation Performance**
   - Validation speed (validations/sec)
   - Schema validation performance
   - Business rules performance
   - Complex nested validation

5. **I/O Performance**
   - Read/write speeds for different formats (CSV, JSON)
   - File size impact on performance
   - Streaming vs loading comparison
   - Format comparison (JSON vs CSV)

## Quick Start

### Prerequisites

Ensure you have the GreenLang framework installed:

```bash
pip install -e .
```

Or install required dependencies:

```bash
pip install pydantic jsonschema tqdm
```

### Running Benchmarks

#### Run All Benchmarks

```bash
python benchmarks/framework_performance.py
```

This will:
1. Run all benchmark tests (~2-3 minutes)
2. Display results in console with tables
3. Save detailed report to `benchmarks/results.md`

#### Run from Project Root

```bash
cd /path/to/Code-V1_GreenLang
python benchmarks/framework_performance.py
```

## Understanding Results

### Console Output

The console output provides:

1. **Real-time Progress**: See which benchmark is currently running
2. **Individual Metrics**: Each test shows its results immediately
3. **Summary Table**: Final table with all metrics, targets, and pass/fail status
4. **Overall Summary**: Total tests run, passed, and failed

Example output:

```
================================================================================
BENCHMARKING: Base Agent Performance
================================================================================

[1/4] Testing simple agent execution...
  Mean: 0.1234ms | P95: 0.2345ms

[2/4] Measuring framework overhead...
  Direct function: 0.0100ms
  Agent execution: 0.1234ms
  Overhead: 0.1134ms (1.13%)

...

Metric                                                Value        Unit       Target   Status
--------------------------------------------------------------------------------
Simple Execution Time (mean)                         0.1234          ms         1.00   ✓ PASS
Simple Execution Time (p95)                          0.2345          ms         2.00   ✓ PASS
Framework Overhead                                   1.1340           %         5.00   ✓ PASS
...
```

### Markdown Report

The detailed `results.md` report includes:

1. **Executive Summary**: High-level pass/fail statistics
2. **Detailed Tables**: All metrics with values, targets, and status
3. **Additional Information**: Metadata and context for each benchmark
4. **Performance Targets Reference**: Explanation of each target and rationale
5. **Recommendations**: Specific suggestions for failed tests

## Performance Targets

### Framework Overhead
- **Target:** < 5% overhead
- **Why:** The framework should add minimal overhead compared to direct function calls
- **How it's measured:** Compare agent execution time to direct function call time

### Agent Execution
- **Target:** < 1ms mean execution time
- **Why:** Fast agent invocation for high-throughput scenarios
- **How it's measured:** Benchmark simple agent with 1000 iterations

### Data Processing
- **Target:** > 10,000 records/sec throughput
- **Why:** Efficient batch processing for large datasets
- **How it's measured:** Process batches of different sizes and measure throughput

### Validation
- **Target:** < 10ms validation time
- **Why:** Fast validation without blocking operations
- **How it's measured:** Benchmark schema and business rule validation

### Caching
- **Target:** < 0.1ms cache hit time
- **Why:** Near-instant cache lookups
- **How it's measured:** Benchmark calculator with cached inputs

## Interpreting Results

### ✓ PASS
The metric meets or exceeds the performance target. No action needed.

### ✗ FAIL
The metric does not meet the performance target. Consider:
- Is this a regression from a previous version?
- Is the target too aggressive for the current implementation?
- Are there optimization opportunities?

### Common Issues and Solutions

#### High Framework Overhead (> 5%)

**Symptoms:** Framework overhead metric fails target

**Possible Causes:**
- Too many hooks registered
- Complex preprocessing logic
- Inefficient metrics collection

**Solutions:**
- Disable metrics collection if not needed
- Reduce number of hooks
- Profile code to find bottlenecks

#### Low Data Processing Throughput

**Symptoms:** Throughput < 10,000 records/sec

**Possible Causes:**
- Processing logic is too complex
- Not using parallel processing
- Batch size is not optimal

**Solutions:**
- Increase batch size for larger datasets
- Enable parallel processing with `parallel_workers > 1`
- Optimize `process_record()` implementation
- Disable record-level validation if not needed

#### Slow Validation

**Symptoms:** Validation time > 10ms

**Possible Causes:**
- Complex schema with many nested levels
- Too many business rules
- Inefficient validation logic

**Solutions:**
- Simplify schema where possible
- Cache validation results if validating same data repeatedly
- Optimize business rule checks
- Consider disabling non-critical validators

#### Cache Not Effective

**Symptoms:** Low cache hit rate or slow cache lookups

**Possible Causes:**
- Cache size too small
- Input data varies too much
- Cache key generation is slow

**Solutions:**
- Increase `cache_size` in configuration
- Ensure deterministic inputs for calculations
- Review cache key generation logic

## Advanced Usage

### Running Specific Benchmarks

Modify `framework_performance.py` to run only specific benchmarks:

```python
def main():
    results = []

    # Only run agent benchmarks
    results.append(benchmark_base_agent())
    results.append(benchmark_calculator())

    generate_console_report(results)
    # ...
```

### Customizing Iterations

Adjust the number of iterations for more accurate results:

```python
# In benchmark_function calls
stats = benchmark_function(lambda: agent.run(test_input), iterations=10000)  # More iterations
```

### Adding Custom Benchmarks

Add your own benchmark functions:

```python
def benchmark_my_component() -> BenchmarkResult:
    """Benchmark my custom component."""
    result = BenchmarkResult(
        name="My Component Performance",
        description="Testing my custom component"
    )

    # Your benchmark code here
    stats = benchmark_function(lambda: my_function(), iterations=1000)
    result.add_metric("My Metric", stats['mean_ms'], "ms", target=1.0)

    return result
```

Then add it to `main()`:

```python
results.append(benchmark_my_component())
```

### Profiling for Optimization

For detailed profiling, use Python's built-in profilers:

```python
import cProfile
import pstats

profiler = cProfile.Profile()
profiler.enable()

# Run your benchmark
agent.run(test_input)

profiler.disable()
stats = pstats.Stats(profiler)
stats.sort_stats('cumulative')
stats.print_stats(20)  # Top 20 slowest functions
```

## Continuous Integration

### Automated Benchmarking

Add to your CI/CD pipeline:

```yaml
# .github/workflows/benchmark.yml
name: Performance Benchmarks

on: [push, pull_request]

jobs:
  benchmark:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Set up Python
        uses: actions/setup-python@v2
        with:
          python-version: '3.9'
      - name: Install dependencies
        run: |
          pip install -e .
      - name: Run benchmarks
        run: |
          python benchmarks/framework_performance.py
      - name: Upload results
        uses: actions/upload-artifact@v2
        with:
          name: benchmark-results
          path: benchmarks/results.md
```

### Performance Regression Detection

Compare benchmark results across commits:

```bash
# Run benchmarks on main branch
git checkout main
python benchmarks/framework_performance.py
cp benchmarks/results.md benchmarks/results_main.md

# Run benchmarks on feature branch
git checkout feature-branch
python benchmarks/framework_performance.py

# Compare results
diff benchmarks/results_main.md benchmarks/results.md
```

## Troubleshooting

### Import Errors

**Error:** `ModuleNotFoundError: No module named 'greenlang'`

**Solution:** Install the framework:
```bash
pip install -e .
```

### Memory Errors

**Error:** `MemoryError` when benchmarking large datasets

**Solution:** Reduce batch sizes in the benchmark:
```python
batch_sizes = [10, 100, 1000]  # Instead of [10, 100, 1000, 10000]
```

### Slow Benchmarks

**Issue:** Benchmarks take too long

**Solution:** Reduce iterations:
```python
stats = benchmark_function(func, iterations=100)  # Instead of 1000
```

## Best Practices

1. **Run on dedicated hardware**: For consistent results, run benchmarks on a machine with minimal other processes
2. **Warm up**: The benchmark suite includes warmup iterations to ensure stable measurements
3. **Multiple runs**: Run benchmarks multiple times and compare results for variability
4. **Document baselines**: Keep historical benchmark results to track performance over time
5. **Test realistic scenarios**: Adjust test data to match your actual use cases
6. **Profile before optimizing**: Use profiling tools to identify actual bottlenecks before making changes

## Benchmark Methodology

### Timing Measurement

- Uses `time.perf_counter()` for high-resolution timing
- Includes warmup iterations to prime caches
- Reports mean, median, min, max, standard deviation, P95, and P99
- All times converted to milliseconds for consistency

### Statistics Reported

- **Mean:** Average time across all iterations
- **Median:** Middle value (less affected by outliers)
- **Min/Max:** Best and worst case times
- **StdDev:** Measure of variability
- **P95/P99:** 95th and 99th percentile (worst 5% and 1%)

### Test Data Generation

- Deterministic test data for reproducibility
- Varied data sizes to test scalability
- Realistic data structures matching framework usage

## Contributing

When adding new benchmarks:

1. Follow the existing structure
2. Document performance targets and rationale
3. Include both mean and P95 measurements
4. Add tests to both console and markdown reports
5. Update this README with new benchmark descriptions

## License

Same as GreenLang framework license.

## Support

For issues or questions:
- Open an issue on GitHub
- Check existing benchmark results in `results.md`
- Review framework documentation

---

*Last updated: 2025-10-18*
