# GreenLang Framework Performance Targets

Quick reference for all performance targets validated by the benchmarking suite.

## Base Agent Performance

| Metric | Target | Unit | Rationale |
|--------|--------|------|-----------|
| Simple Execution Time (mean) | < 1.0 | ms | Fast agent invocation for high-throughput scenarios |
| Simple Execution Time (p95) | < 2.0 | ms | Consistent performance under load |
| Framework Overhead | < 5.0 | % | Minimal performance impact for framework features |
| Overhead (absolute) | - | ms | Informational (no target) |
| Metrics Overhead | < 0.5 | ms | Metrics collection should be lightweight |
| Hooks Overhead (2 hooks) | < 0.3 | ms | Hook execution should be fast |

**Why These Targets?**
- Agents are invoked frequently in production
- Overhead should be negligible compared to actual work
- Sub-millisecond execution enables high throughput

## Data Processor Performance

| Metric | Target | Unit | Rationale |
|--------|--------|------|-----------|
| Sequential Throughput (10 records) | - | records/sec | Baseline for small batches |
| Sequential Throughput (100 records) | - | records/sec | Baseline for medium batches |
| Sequential Throughput (1000 records) | - | records/sec | Baseline for large batches |
| Sequential Throughput (10000 records) | > 10,000 | records/sec | Production-scale throughput |
| Parallel Throughput | - | records/sec | Should exceed sequential |
| Parallel Speedup | > 2.0 | x | Parallel processing should provide benefit |
| Memory Usage | - | MB | Informational (no hard limit) |

**Why These Targets?**
- Data processing is often the bottleneck in pipelines
- 10,000+ records/sec allows processing millions of records in minutes
- Parallel speedup validates multi-core utilization

## Calculator Performance

| Metric | Target | Unit | Rationale |
|--------|--------|------|-----------|
| Calculation Time (mean) | < 1.0 | ms | Fast calculations for real-time use |
| Calculation Time (p95) | < 2.0 | ms | Consistent calculation performance |
| Cache Hit Time | < 0.1 | ms | Near-instant cache lookups |
| Cache Miss Time | - | ms | Informational (depends on calculation) |
| Cache Speedup | > 10.0 | x | Cache should provide significant benefit |
| Cache Lookup Overhead | < 0.2 | ms | Cache key generation should be fast |
| Deterministic Execution Time | < 1.5 | ms | Determinism adds minimal overhead |

**Why These Targets?**
- Calculations often repeated with same inputs (cache helps)
- Sub-millisecond cache hits enable real-time applications
- 10x+ cache speedup justifies caching overhead

## Validation Performance

| Metric | Target | Unit | Rationale |
|--------|--------|------|-----------|
| Simple Validation Speed | > 10,000 | validations/sec | High-throughput validation |
| Simple Validation Time | < 0.1 | ms | Fast validation for hot paths |
| Schema Validation Time | < 1.0 | ms | Acceptable for complex schemas |
| Business Rules Time | < 0.5 | ms | Rules should be efficient |
| Nested Validation Time | < 5.0 | ms | Complex validation acceptable |

**Why These Targets?**
- Validation happens on every request/record
- Should not be a bottleneck in data processing
- 10,000+ validations/sec allows validating large datasets quickly

## I/O Performance

| Metric | Target | Unit | Rationale |
|--------|--------|------|-----------|
| JSON Write Speed (1000 records) | - | ms | Baseline for comparison |
| JSON Read Speed (1000 records) | - | ms | Baseline for comparison |
| JSON File Size | - | MB | Informational |
| CSV Write Speed (1000 records) | - | ms | CSV baseline |
| CSV Read Speed (1000 records) | - | ms | CSV baseline |
| CSV File Size | - | MB | Informational |
| JSON Write Throughput (100 records) | - | records/sec | Small batch baseline |
| JSON Read Throughput (100 records) | - | records/sec | Small batch baseline |
| JSON Write Throughput (1000 records) | > 1,000 | records/sec | Medium batch target |
| JSON Read Throughput (1000 records) | > 1,000 | records/sec | Medium batch target |
| JSON Write Throughput (10000 records) | - | records/sec | Large batch (no target) |
| JSON Read Throughput (10000 records) | - | records/sec | Large batch (no target) |
| Full Load Time (5000 records) | - | ms | Informational |
| Chunked Load Time (5000 records) | - | ms | Informational |

**Why These Targets?**
- I/O performance varies by format and size
- 1,000+ records/sec allows processing moderate datasets efficiently
- CSV generally faster than JSON for tabular data

## Target Summary by Category

### Latency Targets
- **Agent Execution**: < 1ms
- **Calculation**: < 1ms
- **Validation**: < 1ms (simple), < 5ms (complex)
- **Cache Hit**: < 0.1ms

### Throughput Targets
- **Data Processing**: > 10,000 records/sec
- **Validation**: > 10,000 validations/sec
- **I/O**: > 1,000 records/sec

### Overhead Targets
- **Framework**: < 5%
- **Metrics**: < 0.5ms
- **Hooks**: < 0.3ms
- **Cache Lookup**: < 0.2ms

### Scalability Targets
- **Parallel Speedup**: > 2x with 4 workers
- **Cache Speedup**: > 10x for repeated calculations

## Understanding Pass/Fail

### ✓ PASS
**The metric meets or exceeds the target.**

Example: Execution time 0.5ms with target < 1.0ms

**Action:** None needed. Performance is acceptable.

### ✗ FAIL
**The metric does not meet the target.**

Example: Execution time 1.5ms with target < 1.0ms

**Action:** Investigate and optimize:
1. Profile the code to find bottlenecks
2. Check for inefficient algorithms
3. Review recent changes for regressions
4. Consider if target needs adjustment

### - (No Target)
**Informational metric without pass/fail.**

Example: Memory usage, file sizes

**Action:** Monitor trends over time, but no immediate action needed.

## Adjusting Targets

Targets can be adjusted based on:

1. **Hardware**: Slower machines may need relaxed targets
2. **Use Case**: Real-time systems need stricter targets
3. **Data Size**: Larger datasets may need different targets
4. **Complexity**: More complex logic may need higher targets

To adjust targets, edit `benchmarks/framework_performance.py`:

```python
result.add_metric("My Metric", value, "ms", target=2.0)  # Adjust this value
```

## Best Practices

1. **Run on consistent hardware** - Same machine for comparable results
2. **Multiple runs** - Run 3+ times and average to reduce variance
3. **Baseline comparison** - Compare against previous versions
4. **Document changes** - Note any target adjustments and why
5. **Regular testing** - Run benchmarks before each release

## CI/CD Integration

Recommended thresholds for automated testing:

- **Green (Pass)**: All targets met
- **Yellow (Warning)**: 1-5 targets missed by < 20%
- **Red (Fail)**: 5+ targets missed or any missed by > 20%

Example GitHub Actions:

```yaml
- name: Run benchmarks
  run: python benchmarks/framework_performance.py

- name: Check results
  run: |
    if grep -q "FAIL" benchmarks/results.md; then
      echo "Performance regression detected"
      exit 1
    fi
```

## Benchmark Methodology

All targets are based on:

- **10-100 warmup iterations** to prime caches
- **1000+ measurement iterations** for statistical accuracy
- **Standard hardware** (modern CPU, 8GB+ RAM, SSD)
- **Typical workloads** (representative data and operations)
- **P95/P99 metrics** to catch tail latencies

## Interpreting Results

### Good Performance
```
Simple Execution Time (mean)    0.5000ms    Target: 1.00ms    ✓ PASS
Framework Overhead              2.5000%     Target: 5.00%     ✓ PASS
```

All metrics under target = Excellent performance

### Acceptable Performance
```
Simple Execution Time (mean)    0.9000ms    Target: 1.00ms    ✓ PASS
Framework Overhead              4.5000%     Target: 5.00%     ✓ PASS
```

Metrics near target = Acceptable, but monitor for regressions

### Poor Performance
```
Simple Execution Time (mean)    1.5000ms    Target: 1.00ms    ✗ FAIL
Framework Overhead              7.5000%     Target: 5.00%     ✗ FAIL
```

Metrics over target = Needs optimization

## Quick Reference

**Run benchmarks:**
```bash
python benchmarks/framework_performance.py
```

**View results:**
```bash
cat benchmarks/results.md
```

**Check if all passed:**
```bash
grep "ALL PERFORMANCE TARGETS MET" benchmarks/results.md
```

**Count failures:**
```bash
grep -c "✗ FAIL" benchmarks/results.md
```

---

*Last updated: 2025-10-18*
*For detailed explanations, see [README.md](README.md)*
