# Getting Started with GreenLang Framework Benchmarking

This guide will help you quickly start benchmarking the GreenLang framework.

## Quick Start (3 Steps)

### 1. Ensure Dependencies

Make sure you have the GreenLang framework installed:

```bash
# From the project root
pip install -e .
```

Or install just the required dependencies:

```bash
pip install pydantic jsonschema tqdm
```

### 2. Run Benchmarks

```bash
cd /path/to/Code-V1_GreenLang
python benchmarks/framework_performance.py
```

### 3. View Results

Results will be displayed in the console and saved to:
- `benchmarks/results.md` - Detailed markdown report

## What Gets Benchmarked

The suite tests **5 major components**:

### 1. Base Agent Performance
- ⏱️ Execution time
- 📊 Framework overhead
- 📈 Metrics collection impact
- 🪝 Hook execution overhead

**Expected Results:**
- Simple execution: < 1ms
- Framework overhead: < 5%
- Metrics overhead: < 0.5ms

### 2. Data Processor Performance
- 🔄 Batch processing throughput
- ⚡ Sequential vs parallel comparison
- 💾 Memory usage estimates
- 📏 Different batch sizes (10, 100, 1000, 10000)

**Expected Results:**
- Throughput: > 10,000 records/sec
- Parallel speedup: 2-4x with 4 workers

### 3. Calculator Performance
- 🧮 Calculation execution time
- 🚀 Cache performance
- 🔍 Cache lookup overhead
- ✅ Deterministic execution

**Expected Results:**
- Calculation time: < 1ms
- Cache hit time: < 0.1ms
- Cache speedup: > 10x

### 4. Validation Performance
- ✓ Validation speed
- 📋 Schema validation
- 📏 Business rules
- 🔗 Nested validation

**Expected Results:**
- Simple validation: > 10,000 validations/sec
- Schema validation: < 1ms
- Nested validation: < 5ms

### 5. I/O Performance
- 📁 JSON read/write speeds
- 📊 CSV read/write speeds
- 📈 File size impact
- 🔄 Streaming comparison

**Expected Results:**
- JSON throughput: > 1,000 records/sec
- CSV throughput: > 2,000 records/sec

## Understanding Output

### Console Output Example

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
Framework Overhead                                   1.1340           %         5.00   ✓ PASS
...

================================================================================
SUMMARY
================================================================================
Total Tests: 45
Passed: 45 (100.0%)
Failed: 0 (0.0%)

✓ ALL PERFORMANCE TARGETS MET
```

### Status Indicators

- ✓ **PASS** - Performance meets or exceeds target
- ✗ **FAIL** - Performance below target (needs optimization)
- **-** - No target defined (informational metric)

## Common Issues

### Python Not Found

**Problem:** `Python was not found`

**Solutions:**
1. Install Python from [python.org](https://www.python.org/downloads/)
2. Add Python to your PATH
3. Use the Python executable directly: `C:\path\to\python.exe benchmarks/framework_performance.py`

### Import Errors

**Problem:** `ModuleNotFoundError: No module named 'greenlang'`

**Solution:**
```bash
# Install in development mode
pip install -e .

# Or install from source
python setup.py develop
```

### Missing Dependencies

**Problem:** `ModuleNotFoundError: No module named 'pydantic'`

**Solution:**
```bash
# Install required dependencies
pip install pydantic jsonschema tqdm

# Or install all optional dependencies
pip install -e .[all]
```

### Slow Execution

**Problem:** Benchmarks take too long

**Solution:**
- Reduce iterations in the benchmark script (edit `framework_performance.py`)
- Run on a machine with fewer background processes
- Use a faster CPU

## Customization

### Adjust Iterations

Edit `benchmarks/framework_performance.py`:

```python
# Change from 1000 to 100 for faster execution
stats = benchmark_function(func, iterations=100)  # Instead of 1000
```

### Run Specific Benchmarks

Comment out benchmarks you don't need:

```python
def main():
    results = []

    # Only run these two
    results.append(benchmark_base_agent())
    results.append(benchmark_calculator())

    # Skip these
    # results.append(benchmark_data_processor())
    # results.append(benchmark_validation())
    # results.append(benchmark_io())

    generate_console_report(results)
```

### Change Performance Targets

Edit the `target` parameter in each benchmark:

```python
result.add_metric("My Metric", value, "ms", target=2.0)  # Instead of 1.0
```

## Next Steps

1. **Review Results**: Check `benchmarks/results.md` for detailed analysis
2. **Investigate Failures**: If any tests fail, see recommendations in the report
3. **Track Over Time**: Save results and compare across versions
4. **Optimize**: Use profiling tools to identify bottlenecks

## Advanced Topics

See [README.md](README.md) for:
- Performance targets reference
- CI/CD integration
- Profiling techniques
- Adding custom benchmarks
- Regression detection

## Need Help?

- Check [README.md](README.md) for detailed documentation
- Review `benchmarks/results.md` after running
- Open an issue on GitHub
- Review existing test results for reference

---

**Ready to benchmark?** Just run:

```bash
python benchmarks/framework_performance.py
```

🚀 **Happy benchmarking!**
