# -*- coding: utf-8 -*-
"""
GreenLang Framework Performance Benchmarking Suite
====================================================

Comprehensive performance testing for all framework components:
- Base Agent Performance
- Data Processor Performance
- Calculator Performance
- Validation Performance
- I/O Performance

Usage:
    python benchmarks/framework_performance.py

Output:
    - Console report with tables and metrics
    - Detailed report saved to benchmarks/results.md
"""

import sys
import time
import json
import csv
import tempfile
from pathlib import Path
from typing import Dict, Any, List, Callable
from dataclasses import dataclass, field
from datetime import datetime
import statistics
from greenlang.determinism import DeterministicClock

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Framework imports
from greenlang.agents.base import BaseAgent, AgentConfig, AgentResult
from greenlang.agents.calculator import BaseCalculator, CalculatorConfig
from greenlang.agents.data_processor import BaseDataProcessor, DataProcessorConfig
from greenlang.validation.framework import ValidationFramework, ValidationResult, ValidationError, ValidationSeverity
from greenlang.validation.schema import SchemaValidator
from greenlang.io.readers import DataReader
from greenlang.io.writers import DataWriter


# =============================================================================
# BENCHMARK RESULT STRUCTURES
# =============================================================================

@dataclass
class BenchmarkMetric:
    """A single benchmark metric."""
    name: str
    value: float
    unit: str
    target: float = None
    passed: bool = None

    def __post_init__(self):
        if self.target is not None and self.passed is None:
            self.passed = self.value <= self.target


@dataclass
class BenchmarkResult:
    """Results from a benchmark test."""
    name: str
    description: str
    metrics: List[BenchmarkMetric] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def add_metric(self, name: str, value: float, unit: str, target: float = None):
        """Add a metric to the results."""
        metric = BenchmarkMetric(name, value, unit, target)
        self.metrics.append(metric)
        return metric

    def get_summary(self) -> str:
        """Get a summary of results."""
        passed = sum(1 for m in self.metrics if m.passed is True)
        failed = sum(1 for m in self.metrics if m.passed is False)
        total = len([m for m in self.metrics if m.passed is not None])
        return f"{self.name}: {passed}/{total} passed"


# =============================================================================
# TEST AGENTS
# =============================================================================

class SimpleAgent(BaseAgent):
    """Simple test agent that does minimal work."""

    def execute(self, input_data: Dict[str, Any]) -> AgentResult:
        """Just return the input."""
        return AgentResult(
            success=True,
            data=input_data
        )


class SimpleCalculator(BaseCalculator):
    """Simple calculator for testing."""

    def calculate(self, inputs: Dict[str, Any]) -> Any:
        """Simple multiplication."""
        return inputs.get('a', 0) * inputs.get('b', 1)

    def validate_calculation_inputs(self, inputs: Dict[str, Any]) -> bool:
        """Validate inputs."""
        return 'a' in inputs and 'b' in inputs


class SimpleDataProcessor(BaseDataProcessor):
    """Simple data processor for testing."""

    def process_record(self, record: Dict[str, Any]) -> Dict[str, Any]:
        """Simple transformation."""
        result = record.copy()
        result['processed'] = True
        result['value'] = record.get('value', 0) * 2
        return result

    def validate_record(self, record: Dict[str, Any]) -> bool:
        """Validate record."""
        return 'value' in record


# =============================================================================
# BENCHMARKING UTILITIES
# =============================================================================

def benchmark_function(func: Callable, iterations: int = 1000, warmup: int = 100) -> Dict[str, float]:
    """
    Benchmark a function with multiple iterations.

    Args:
        func: Function to benchmark (no arguments)
        iterations: Number of iterations
        warmup: Number of warmup iterations

    Returns:
        Dictionary with timing statistics
    """
    # Warmup
    for _ in range(warmup):
        func()

    # Benchmark
    times = []
    for _ in range(iterations):
        start = time.perf_counter()
        func()
        end = time.perf_counter()
        times.append((end - start) * 1000)  # Convert to ms

    return {
        'mean_ms': statistics.mean(times),
        'median_ms': statistics.median(times),
        'min_ms': min(times),
        'max_ms': max(times),
        'stdev_ms': statistics.stdev(times) if len(times) > 1 else 0,
        'p95_ms': sorted(times)[int(len(times) * 0.95)],
        'p99_ms': sorted(times)[int(len(times) * 0.99)]
    }


def create_test_records(count: int) -> List[Dict[str, Any]]:
    """Create test records for benchmarking."""
    return [
        {
            'id': i,
            'value': i * 1.5,
            'name': f'Record {i}',
            'category': f'Category {i % 10}'
        }
        for i in range(count)
    ]


# =============================================================================
# BENCHMARK TESTS
# =============================================================================

def benchmark_base_agent() -> BenchmarkResult:
    """Benchmark base agent performance."""
    print("\n" + "="*80)
    print("BENCHMARKING: Base Agent Performance")
    print("="*80)

    result = BenchmarkResult(
        name="Base Agent Performance",
        description="Testing agent execution overhead, metrics collection, and hooks"
    )

    # 1. Simple agent execution time
    print("\n[1/4] Testing simple agent execution...")
    agent = SimpleAgent()
    test_input = {'test': 'data', 'value': 42}

    stats = benchmark_function(lambda: agent.run(test_input), iterations=1000)
    result.add_metric("Simple Execution Time (mean)", stats['mean_ms'], "ms", target=1.0)
    result.add_metric("Simple Execution Time (p95)", stats['p95_ms'], "ms", target=2.0)
    print(f"  Mean: {stats['mean_ms']:.4f}ms | P95: {stats['p95_ms']:.4f}ms")

    # 2. Overhead vs direct function call
    print("\n[2/4] Measuring framework overhead...")
    def direct_function():
        return test_input

    direct_stats = benchmark_function(direct_function, iterations=1000)
    overhead_ms = stats['mean_ms'] - direct_stats['mean_ms']
    overhead_pct = (overhead_ms / direct_stats['mean_ms']) * 100

    result.add_metric("Framework Overhead", overhead_pct, "%", target=5.0)
    result.add_metric("Overhead (absolute)", overhead_ms, "ms")
    print(f"  Direct function: {direct_stats['mean_ms']:.4f}ms")
    print(f"  Agent execution: {stats['mean_ms']:.4f}ms")
    print(f"  Overhead: {overhead_ms:.4f}ms ({overhead_pct:.2f}%)")

    # 3. Metrics collection overhead
    print("\n[3/4] Testing metrics collection overhead...")
    agent_no_metrics = SimpleAgent(AgentConfig(
        name="NoMetrics",
        description="Test",
        enable_metrics=False
    ))

    no_metrics_stats = benchmark_function(lambda: agent_no_metrics.run(test_input), iterations=1000)
    metrics_overhead_ms = stats['mean_ms'] - no_metrics_stats['mean_ms']

    result.add_metric("Metrics Overhead", metrics_overhead_ms, "ms", target=0.5)
    print(f"  With metrics: {stats['mean_ms']:.4f}ms")
    print(f"  Without metrics: {no_metrics_stats['mean_ms']:.4f}ms")
    print(f"  Metrics overhead: {metrics_overhead_ms:.4f}ms")

    # 4. Hook execution overhead
    print("\n[4/4] Testing hook execution overhead...")
    agent_with_hooks = SimpleAgent()
    hook_counter = {'count': 0}

    def test_hook(agent, data):
        hook_counter['count'] += 1

    agent_with_hooks.add_pre_hook(test_hook)
    agent_with_hooks.add_post_hook(test_hook)

    hooks_stats = benchmark_function(lambda: agent_with_hooks.run(test_input), iterations=1000)
    hooks_overhead_ms = hooks_stats['mean_ms'] - stats['mean_ms']

    result.add_metric("Hooks Overhead (2 hooks)", hooks_overhead_ms, "ms", target=0.3)
    print(f"  Without hooks: {stats['mean_ms']:.4f}ms")
    print(f"  With 2 hooks: {hooks_stats['mean_ms']:.4f}ms")
    print(f"  Hooks overhead: {hooks_overhead_ms:.4f}ms")

    return result


def benchmark_data_processor() -> BenchmarkResult:
    """Benchmark data processor performance."""
    print("\n" + "="*80)
    print("BENCHMARKING: Data Processor Performance")
    print("="*80)

    result = BenchmarkResult(
        name="Data Processor Performance",
        description="Testing batch processing throughput and scalability"
    )

    # Test different batch sizes
    batch_sizes = [10, 100, 1000, 10000]

    for batch_size in batch_sizes:
        print(f"\n[Testing batch size: {batch_size}]")

        # 1. Sequential processing
        print(f"  [1/3] Sequential processing...")
        processor = SimpleDataProcessor(DataProcessorConfig(
            name="TestProcessor",
            description="Test",
            batch_size=batch_size,
            parallel_workers=1,
            enable_progress=False
        ))

        records = create_test_records(batch_size)
        input_data = {'records': records}

        start = time.perf_counter()
        seq_result = processor.run(input_data)
        seq_time = (time.perf_counter() - start) * 1000

        if seq_result.success:
            throughput = batch_size / (seq_time / 1000)  # records/sec
            result.add_metric(f"Sequential Throughput ({batch_size} records)", throughput, "records/sec")
            print(f"    Time: {seq_time:.2f}ms | Throughput: {throughput:.0f} records/sec")

        # 2. Parallel processing (only for larger batches)
        if batch_size >= 100:
            print(f"  [2/3] Parallel processing (4 workers)...")
            parallel_processor = SimpleDataProcessor(DataProcessorConfig(
                name="TestProcessor",
                description="Test",
                batch_size=max(batch_size // 4, 10),
                parallel_workers=4,
                enable_progress=False
            ))

            start = time.perf_counter()
            par_result = parallel_processor.run(input_data)
            par_time = (time.perf_counter() - start) * 1000

            if par_result.success:
                throughput = batch_size / (par_time / 1000)
                speedup = seq_time / par_time
                result.add_metric(f"Parallel Throughput ({batch_size} records)", throughput, "records/sec")
                result.add_metric(f"Parallel Speedup ({batch_size} records)", speedup, "x")
                print(f"    Time: {par_time:.2f}ms | Throughput: {throughput:.0f} records/sec | Speedup: {speedup:.2f}x")

        # 3. Memory-efficient processing (estimate)
        print(f"  [3/3] Memory estimation...")
        import sys
        record_size = sys.getsizeof(str(records[0]))
        total_size_mb = (record_size * batch_size) / (1024 * 1024)
        result.add_metric(f"Memory Usage ({batch_size} records)", total_size_mb, "MB")
        print(f"    Estimated memory: {total_size_mb:.2f} MB")

    return result


def benchmark_calculator() -> BenchmarkResult:
    """Benchmark calculator performance."""
    print("\n" + "="*80)
    print("BENCHMARKING: Calculator Performance")
    print("="*80)

    result = BenchmarkResult(
        name="Calculator Performance",
        description="Testing calculation execution, caching, and determinism"
    )

    # 1. Calculation execution time
    print("\n[1/4] Testing calculation execution...")
    calc = SimpleCalculator()
    test_input = {'inputs': {'a': 42, 'b': 3.14}}

    stats = benchmark_function(lambda: calc.run(test_input), iterations=1000)
    result.add_metric("Calculation Time (mean)", stats['mean_ms'], "ms", target=1.0)
    result.add_metric("Calculation Time (p95)", stats['p95_ms'], "ms", target=2.0)
    print(f"  Mean: {stats['mean_ms']:.4f}ms | P95: {stats['p95_ms']:.4f}ms")

    # 2. Cache hit performance
    print("\n[2/4] Testing cache performance...")
    calc_cached = SimpleCalculator(CalculatorConfig(
        name="CachedCalc",
        description="Test",
        enable_caching=True
    ))

    # Prime the cache
    calc_cached.run(test_input)

    # Benchmark cache hits
    cache_hit_stats = benchmark_function(lambda: calc_cached.run(test_input), iterations=1000)

    # Benchmark cache misses
    cache_miss_inputs = [{'inputs': {'a': i, 'b': i * 2}} for i in range(100)]
    def cache_miss_test():
        calc_cached.clear_cache()
        for inp in cache_miss_inputs[:10]:
            calc_cached.run(inp)

    cache_miss_stats = benchmark_function(cache_miss_test, iterations=100, warmup=10)

    speedup = cache_miss_stats['mean_ms'] / cache_hit_stats['mean_ms']
    result.add_metric("Cache Hit Time", cache_hit_stats['mean_ms'], "ms", target=0.1)
    result.add_metric("Cache Miss Time", cache_miss_stats['mean_ms'] / 10, "ms")  # Per calculation
    result.add_metric("Cache Speedup", speedup, "x")
    print(f"  Cache hit: {cache_hit_stats['mean_ms']:.4f}ms")
    print(f"  Cache miss: {cache_miss_stats['mean_ms'] / 10:.4f}ms (per calc)")
    print(f"  Speedup: {speedup:.2f}x")

    # 3. Cache lookup overhead
    print("\n[3/4] Testing cache lookup overhead...")
    calc_no_cache = SimpleCalculator(CalculatorConfig(
        name="NoCache",
        description="Test",
        enable_caching=False
    ))

    no_cache_stats = benchmark_function(lambda: calc_no_cache.run(test_input), iterations=1000)
    lookup_overhead_ms = stats['mean_ms'] - no_cache_stats['mean_ms']

    result.add_metric("Cache Lookup Overhead", lookup_overhead_ms, "ms", target=0.2)
    print(f"  With caching: {stats['mean_ms']:.4f}ms")
    print(f"  Without caching: {no_cache_stats['mean_ms']:.4f}ms")
    print(f"  Lookup overhead: {lookup_overhead_ms:.4f}ms")

    # 4. Deterministic execution overhead
    print("\n[4/4] Testing deterministic execution...")
    calc_deterministic = SimpleCalculator(CalculatorConfig(
        name="Deterministic",
        description="Test",
        deterministic=True,
        precision=6
    ))

    det_stats = benchmark_function(lambda: calc_deterministic.run(test_input), iterations=1000)
    result.add_metric("Deterministic Execution Time", det_stats['mean_ms'], "ms", target=1.5)

    # Verify determinism
    results = [calc_deterministic.run(test_input).data.get('result') for _ in range(10)]
    is_deterministic = len(set(results)) == 1
    result.metadata['determinism_verified'] = is_deterministic
    print(f"  Execution time: {det_stats['mean_ms']:.4f}ms")
    print(f"  Determinism verified: {is_deterministic}")

    return result


def benchmark_validation() -> BenchmarkResult:
    """Benchmark validation performance."""
    print("\n" + "="*80)
    print("BENCHMARKING: Validation Performance")
    print("="*80)

    result = BenchmarkResult(
        name="Validation Performance",
        description="Testing validation speed and schema performance"
    )

    # 1. Simple validation speed
    print("\n[1/4] Testing simple validation...")
    framework = ValidationFramework()

    def simple_validator(data):
        errors = []
        if not isinstance(data, dict):
            errors.append(ValidationError(
                field="root",
                message="Must be a dict",
                validator="type_check",
                severity=ValidationSeverity.ERROR
            ))
        return ValidationResult(valid=len(errors) == 0, errors=errors)

    framework.add_validator("simple", simple_validator)

    test_data = {'value': 42, 'name': 'test'}
    stats = benchmark_function(lambda: framework.validate(test_data), iterations=1000)

    validations_per_sec = 1000 / (stats['mean_ms'] / 1000)
    result.add_metric("Simple Validation Speed", validations_per_sec, "validations/sec", target=10000)
    result.add_metric("Simple Validation Time", stats['mean_ms'], "ms", target=0.1)
    print(f"  Time: {stats['mean_ms']:.4f}ms | Speed: {validations_per_sec:.0f} validations/sec")

    # 2. Schema validation performance
    print("\n[2/4] Testing schema validation...")
    schema = {
        'type': 'object',
        'properties': {
            'value': {'type': 'number'},
            'name': {'type': 'string'},
            'tags': {'type': 'array', 'items': {'type': 'string'}}
        },
        'required': ['value', 'name']
    }

    schema_validator = SchemaValidator(schema=schema)
    schema_stats = benchmark_function(
        lambda: schema_validator.validate({'value': 42, 'name': 'test', 'tags': ['a', 'b']}),
        iterations=1000
    )

    result.add_metric("Schema Validation Time", schema_stats['mean_ms'], "ms", target=1.0)
    print(f"  Time: {schema_stats['mean_ms']:.4f}ms")

    # 3. Business rules performance
    print("\n[3/4] Testing business rules...")

    def business_rules_validator(data):
        errors = []
        if data.get('value', 0) < 0:
            errors.append(ValidationError(
                field="value",
                message="Value must be non-negative",
                validator="business_rules",
                severity=ValidationSeverity.ERROR
            ))
        if data.get('value', 0) > 100:
            errors.append(ValidationError(
                field="value",
                message="Value exceeds maximum",
                validator="business_rules",
                severity=ValidationSeverity.WARNING
            ))
        return ValidationResult(valid=len([e for e in errors if e.severity == ValidationSeverity.ERROR]) == 0, errors=errors)

    rules_framework = ValidationFramework()
    rules_framework.add_validator("rules", business_rules_validator)

    rules_stats = benchmark_function(lambda: rules_framework.validate({'value': 42}), iterations=1000)
    result.add_metric("Business Rules Time", rules_stats['mean_ms'], "ms", target=0.5)
    print(f"  Time: {rules_stats['mean_ms']:.4f}ms")

    # 4. Complex nested validation
    print("\n[4/4] Testing complex nested validation...")

    nested_schema = {
        'type': 'object',
        'properties': {
            'building': {
                'type': 'object',
                'properties': {
                    'name': {'type': 'string'},
                    'floors': {
                        'type': 'array',
                        'items': {
                            'type': 'object',
                            'properties': {
                                'number': {'type': 'integer'},
                                'area': {'type': 'number'}
                            }
                        }
                    }
                }
            }
        }
    }

    nested_data = {
        'building': {
            'name': 'Test Building',
            'floors': [
                {'number': i, 'area': 1000.0 + i * 10}
                for i in range(10)
            ]
        }
    }

    nested_validator = SchemaValidator(schema=nested_schema)
    nested_stats = benchmark_function(lambda: nested_validator.validate(nested_data), iterations=1000)

    result.add_metric("Nested Validation Time", nested_stats['mean_ms'], "ms", target=5.0)
    print(f"  Time: {nested_stats['mean_ms']:.4f}ms")

    return result


def benchmark_io() -> BenchmarkResult:
    """Benchmark I/O performance."""
    print("\n" + "="*80)
    print("BENCHMARKING: I/O Performance")
    print("="*80)

    result = BenchmarkResult(
        name="I/O Performance",
        description="Testing read/write speeds for different formats"
    )

    # Create test data
    test_records = create_test_records(1000)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # 1. JSON I/O
        print("\n[1/5] Testing JSON I/O...")
        json_file = tmpdir / "test.json"

        # Write
        start = time.perf_counter()
        with open(json_file, 'w') as f:
            json.dump(test_records, f)
        write_time = (time.perf_counter() - start) * 1000
        file_size_mb = json_file.stat().st_size / (1024 * 1024)

        # Read
        reader = DataReader()
        start = time.perf_counter()
        data = reader.read(json_file)
        read_time = (time.perf_counter() - start) * 1000

        result.add_metric("JSON Write Speed (1000 records)", write_time, "ms")
        result.add_metric("JSON Read Speed (1000 records)", read_time, "ms")
        result.add_metric("JSON File Size", file_size_mb, "MB")
        print(f"  Write: {write_time:.2f}ms | Read: {read_time:.2f}ms | Size: {file_size_mb:.2f}MB")

        # 2. CSV I/O
        print("\n[2/5] Testing CSV I/O...")
        csv_file = tmpdir / "test.csv"

        # Write
        start = time.perf_counter()
        with open(csv_file, 'w', newline='') as f:
            if test_records:
                writer = csv.DictWriter(f, fieldnames=test_records[0].keys())
                writer.writeheader()
                writer.writerows(test_records)
        write_time = (time.perf_counter() - start) * 1000
        file_size_mb = csv_file.stat().st_size / (1024 * 1024)

        # Read
        start = time.perf_counter()
        data = reader.read(csv_file)
        read_time = (time.perf_counter() - start) * 1000

        result.add_metric("CSV Write Speed (1000 records)", write_time, "ms")
        result.add_metric("CSV Read Speed (1000 records)", read_time, "ms")
        result.add_metric("CSV File Size", file_size_mb, "MB")
        print(f"  Write: {write_time:.2f}ms | Read: {read_time:.2f}ms | Size: {file_size_mb:.2f}MB")

        # 3. Test different file sizes
        print("\n[3/5] Testing file size impact...")
        for count in [100, 1000, 10000]:
            records = create_test_records(count)
            test_file = tmpdir / f"test_{count}.json"

            # Write
            start = time.perf_counter()
            with open(test_file, 'w') as f:
                json.dump(records, f)
            write_time = (time.perf_counter() - start) * 1000

            # Read
            start = time.perf_counter()
            with open(test_file, 'r') as f:
                data = json.load(f)
            read_time = (time.perf_counter() - start) * 1000

            throughput_write = count / (write_time / 1000)
            throughput_read = count / (read_time / 1000)

            result.add_metric(f"JSON Write Throughput ({count} records)", throughput_write, "records/sec")
            result.add_metric(f"JSON Read Throughput ({count} records)", throughput_read, "records/sec")
            print(f"  {count} records: Write {throughput_write:.0f} rec/sec | Read {throughput_read:.0f} rec/sec")

        # 4. Streaming vs loading (simulated)
        print("\n[4/5] Testing streaming impact...")
        large_records = create_test_records(5000)
        stream_file = tmpdir / "stream.json"

        # Full load
        start = time.perf_counter()
        with open(stream_file, 'w') as f:
            json.dump(large_records, f)
        with open(stream_file, 'r') as f:
            data = json.load(f)
        full_load_time = (time.perf_counter() - start) * 1000

        # Chunked write/read (simulated streaming)
        chunk_size = 1000
        start = time.perf_counter()
        chunks = [large_records[i:i+chunk_size] for i in range(0, len(large_records), chunk_size)]
        for i, chunk in enumerate(chunks):
            chunk_file = tmpdir / f"chunk_{i}.json"
            with open(chunk_file, 'w') as f:
                json.dump(chunk, f)
            with open(chunk_file, 'r') as f:
                _ = json.load(f)
        chunked_time = (time.perf_counter() - start) * 1000

        result.add_metric("Full Load Time (5000 records)", full_load_time, "ms")
        result.add_metric("Chunked Load Time (5000 records)", chunked_time, "ms")
        print(f"  Full load: {full_load_time:.2f}ms | Chunked: {chunked_time:.2f}ms")

        # 5. Format comparison
        print("\n[5/5] Format comparison summary...")
        formats_summary = {
            'JSON': {'read': None, 'write': None, 'size': None},
            'CSV': {'read': None, 'write': None, 'size': None}
        }

        for metric in result.metrics:
            if 'JSON Read Speed (1000' in metric.name:
                formats_summary['JSON']['read'] = metric.value
            elif 'JSON Write Speed (1000' in metric.name:
                formats_summary['JSON']['write'] = metric.value
            elif 'JSON File Size' in metric.name:
                formats_summary['JSON']['size'] = metric.value
            elif 'CSV Read Speed (1000' in metric.name:
                formats_summary['CSV']['read'] = metric.value
            elif 'CSV Write Speed (1000' in metric.name:
                formats_summary['CSV']['write'] = metric.value
            elif 'CSV File Size' in metric.name:
                formats_summary['CSV']['size'] = metric.value

        result.metadata['format_comparison'] = formats_summary
        print("  Format comparison:")
        for fmt, stats in formats_summary.items():
            print(f"    {fmt}: Read {stats['read']:.2f}ms | Write {stats['write']:.2f}ms | Size {stats['size']:.2f}MB")

    return result


# =============================================================================
# REPORTING
# =============================================================================

def generate_ascii_bar_chart(values: List[float], labels: List[str], width: int = 50) -> str:
    """Generate an ASCII bar chart."""
    if not values:
        return ""

    max_val = max(values)
    lines = []

    for label, val in zip(labels, values):
        bar_len = int((val / max_val) * width) if max_val > 0 else 0
        bar = "█" * bar_len
        lines.append(f"  {label:30s} {bar} {val:.2f}")

    return "\n".join(lines)


def generate_console_report(results: List[BenchmarkResult]):
    """Generate console report with tables."""
    print("\n" + "="*80)
    print("PERFORMANCE BENCHMARK REPORT")
    print("="*80)
    print(f"Generated: {DeterministicClock.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)

    for bench_result in results:
        print(f"\n{bench_result.name}")
        print("-" * 80)
        print(f"{bench_result.description}\n")

        # Metrics table
        print(f"{'Metric':<50} {'Value':>12} {'Unit':<10} {'Target':>10} {'Status':>8}")
        print("-" * 80)

        for metric in bench_result.metrics:
            status = ""
            if metric.passed is True:
                status = "✓ PASS"
            elif metric.passed is False:
                status = "✗ FAIL"

            target_str = f"{metric.target:.2f}" if metric.target is not None else "-"
            value_str = f"{metric.value:.4f}" if metric.value < 1000 else f"{metric.value:.0f}"

            print(f"{metric.name:<50} {value_str:>12} {metric.unit:<10} {target_str:>10} {status:>8}")

        print()

    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)

    total_tests = sum(len([m for m in r.metrics if m.passed is not None]) for r in results)
    total_passed = sum(len([m for m in r.metrics if m.passed is True]) for r in results)
    total_failed = sum(len([m for m in r.metrics if m.passed is False]) for r in results)

    print(f"Total Tests: {total_tests}")
    print(f"Passed: {total_passed} ({total_passed/total_tests*100:.1f}%)")
    print(f"Failed: {total_failed} ({total_failed/total_tests*100:.1f}%)")

    if total_failed == 0:
        print("\n✓ ALL PERFORMANCE TARGETS MET")
    else:
        print(f"\n✗ {total_failed} PERFORMANCE TARGETS MISSED")

    print("="*80)


def generate_markdown_report(results: List[BenchmarkResult], output_file: Path):
    """Generate detailed markdown report."""
    lines = [
        "# GreenLang Framework Performance Benchmark Results",
        "",
        f"**Generated:** {DeterministicClock.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## Executive Summary",
        ""
    ]

    # Summary stats
    total_tests = sum(len([m for m in r.metrics if m.passed is not None]) for r in results)
    total_passed = sum(len([m for m in r.metrics if m.passed is True]) for r in results)
    total_failed = sum(len([m for m in r.metrics if m.passed is False]) for r in results)

    lines.extend([
        f"- **Total Tests:** {total_tests}",
        f"- **Passed:** {total_passed} ({total_passed/total_tests*100:.1f}%)",
        f"- **Failed:** {total_failed} ({total_failed/total_tests*100:.1f}%)",
        "",
    ])

    if total_failed == 0:
        lines.append("✅ **All performance targets met!**")
    else:
        lines.append(f"⚠️ **{total_failed} performance targets missed**")

    lines.append("")

    # Detailed results
    for bench_result in results:
        lines.extend([
            f"## {bench_result.name}",
            "",
            bench_result.description,
            "",
            "| Metric | Value | Unit | Target | Status |",
            "|--------|-------|------|--------|--------|"
        ])

        for metric in bench_result.metrics:
            status = ""
            if metric.passed is True:
                status = "✅ PASS"
            elif metric.passed is False:
                status = "❌ FAIL"
            else:
                status = "-"

            target_str = f"{metric.target:.2f}" if metric.target is not None else "-"
            value_str = f"{metric.value:.4f}" if metric.value < 1000 else f"{metric.value:.0f}"

            lines.append(f"| {metric.name} | {value_str} | {metric.unit} | {target_str} | {status} |")

        lines.append("")

        # Add metadata if present
        if bench_result.metadata:
            lines.extend([
                "### Additional Information",
                "",
                "```json",
                json.dumps(bench_result.metadata, indent=2),
                "```",
                ""
            ])

    # Performance targets reference
    lines.extend([
        "## Performance Targets Reference",
        "",
        "### Framework Overhead",
        "- **Target:** < 5% overhead compared to direct function calls",
        "- **Rationale:** Minimal performance impact for framework features",
        "",
        "### Agent Execution",
        "- **Target:** < 1ms mean execution time for simple agents",
        "- **Rationale:** Fast agent invocation for high-throughput scenarios",
        "",
        "### Data Processing",
        "- **Target:** > 10,000 records/sec throughput",
        "- **Rationale:** Efficient batch processing for large datasets",
        "",
        "### Validation",
        "- **Target:** < 10ms validation time",
        "- **Rationale:** Fast validation without blocking operations",
        "",
        "### Caching",
        "- **Target:** < 0.1ms cache hit time",
        "- **Rationale:** Near-instant cache lookups",
        "",
        "## Recommendations",
        ""
    ])

    # Generate recommendations based on results
    recommendations = []

    for bench_result in results:
        for metric in bench_result.metrics:
            if metric.passed is False:
                recommendations.append(
                    f"- **{bench_result.name}:** {metric.name} failed target "
                    f"({metric.value:.2f} {metric.unit} vs target {metric.target:.2f} {metric.unit})"
                )

    if recommendations:
        lines.extend(recommendations)
    else:
        lines.append("✅ All performance targets met. No optimization recommendations at this time.")

    lines.extend([
        "",
        "---",
        "",
        "*This report was generated automatically by the GreenLang Framework Performance Benchmarking Suite.*"
    ])

    # Write to file
    output_file.write_text("\n".join(lines))
    print(f"\n✓ Detailed report saved to: {output_file}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    """Run all benchmarks and generate reports."""
    print("="*80)
    print("GREENLANG FRAMEWORK PERFORMANCE BENCHMARKING SUITE")
    print("="*80)
    print("\nThis will benchmark all major framework components.")
    print("Estimated time: 2-3 minutes\n")

    results = []

    # Run benchmarks
    try:
        results.append(benchmark_base_agent())
        results.append(benchmark_data_processor())
        results.append(benchmark_calculator())
        results.append(benchmark_validation())
        results.append(benchmark_io())
    except Exception as e:
        print(f"\n✗ Benchmark failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

    # Generate reports
    generate_console_report(results)

    # Save markdown report
    output_file = Path(__file__).parent / "results.md"
    generate_markdown_report(results, output_file)

    return 0


if __name__ == "__main__":
    sys.exit(main())
