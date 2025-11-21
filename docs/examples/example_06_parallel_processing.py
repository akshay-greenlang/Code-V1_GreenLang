# -*- coding: utf-8 -*-
"""
Example 06: Parallel Processing

This example demonstrates parallel processing for improved performance.
You'll learn:
- How to enable parallel processing
- How to choose the optimal number of workers
- How to compare sequential vs parallel performance
- How to handle thread safety
"""

from greenlang.agents import BaseDataProcessor, DataProcessorConfig
from typing import Dict, Any
import time
import random
from greenlang.determinism import deterministic_random


class CPUIntensiveProcessor(BaseDataProcessor):
    """
    Simulate CPU-intensive processing that benefits from parallelization.

    This agent demonstrates:
    - Performance comparison: sequential vs parallel
    - Worker configuration
    - Thread safety
    - Performance benchmarking
    """

    def __init__(self, batch_size=100, parallel_workers=1, enable_progress=True):
        config = DataProcessorConfig(
            name="CPUIntensiveProcessor",
            description="Process data with simulated CPU-intensive operations",
            batch_size=batch_size,
            parallel_workers=parallel_workers,
            enable_progress=enable_progress,
            collect_errors=True,
            max_errors=100
        )
        super().__init__(config)

    def process_record(self, record: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a single record with CPU-intensive simulation.

        Args:
            record: Input record with 'value' field

        Returns:
            Processed record with computed result
        """
        # Simulate CPU-intensive work (prime number check)
        value = record['value']
        result = self._is_prime(value)

        # Simulate additional computation
        time.sleep(0.001)  # 1ms delay to simulate work

        return {
            'id': record['id'],
            'value': value,
            'is_prime': result,
            'factors': self._get_factors(value) if not result else [1, value]
        }

    def validate_record(self, record: Dict[str, Any]) -> bool:
        """Validate record has required fields."""
        return 'id' in record and 'value' in record and isinstance(record['value'], int)

    @staticmethod
    def _is_prime(n: int) -> bool:
        """Check if number is prime (CPU-intensive for large numbers)."""
        if n < 2:
            return False
        if n == 2:
            return True
        if n % 2 == 0:
            return False

        for i in range(3, int(n ** 0.5) + 1, 2):
            if n % i == 0:
                return False
        return True

    @staticmethod
    def _get_factors(n: int) -> list:
        """Get all factors of a number."""
        factors = []
        for i in range(1, int(n ** 0.5) + 1):
            if n % i == 0:
                factors.append(i)
                if i != n // i:
                    factors.append(n // i)
        return sorted(factors)


def generate_test_data(size=1000):
    """Generate test dataset."""
    return [
        {'id': i, 'value': deterministic_random().randint(1000, 10000)}
        for i in range(size)
    ]


def run_benchmark(processor, data, label):
    """Run benchmark and return results."""
    print(f"  Running: {label}")

    start_time = time.time()
    result = processor.run({"records": data})
    elapsed_time = time.time() - start_time

    if result.success:
        print(f"    ✓ Completed")
        print(f"    - Records processed: {result.records_processed}")
        print(f"    - Wall clock time: {elapsed_time:.2f}s")
        print(f"    - Agent time: {result.metrics.execution_time_ms:.0f}ms")
        print(f"    - Throughput: {result.records_processed / elapsed_time:.0f} records/sec")

        # Count primes
        primes = sum(1 for r in result.data['records'] if r['is_prime'])
        print(f"    - Prime numbers found: {primes}")

    return {
        'label': label,
        'success': result.success,
        'wall_time': elapsed_time,
        'records_processed': result.records_processed,
        'throughput': result.records_processed / elapsed_time if elapsed_time > 0 else 0
    }


def main():
    """Run the example."""
    print("=" * 60)
    print("Example 06: Parallel Processing")
    print("=" * 60)
    print()

    # Generate test data
    print("Generating test data...")
    test_data = generate_test_data(500)
    print(f"  Generated {len(test_data)} records")
    print()

    # Benchmark results
    benchmark_results = []

    # Test 1: Sequential processing (1 worker)
    print("Test 1: Sequential Processing (1 worker)")
    print("-" * 40)
    processor_seq = CPUIntensiveProcessor(
        batch_size=50,
        parallel_workers=1,
        enable_progress=True
    )
    result1 = run_benchmark(processor_seq, test_data, "Sequential (1 worker)")
    benchmark_results.append(result1)
    print()

    # Test 2: Parallel processing (2 workers)
    print("Test 2: Parallel Processing (2 workers)")
    print("-" * 40)
    processor_par2 = CPUIntensiveProcessor(
        batch_size=50,
        parallel_workers=2,
        enable_progress=True
    )
    result2 = run_benchmark(processor_par2, test_data, "Parallel (2 workers)")
    benchmark_results.append(result2)
    print()

    # Test 3: Parallel processing (4 workers)
    print("Test 3: Parallel Processing (4 workers)")
    print("-" * 40)
    processor_par4 = CPUIntensiveProcessor(
        batch_size=50,
        parallel_workers=4,
        enable_progress=True
    )
    result3 = run_benchmark(processor_par4, test_data, "Parallel (4 workers)")
    benchmark_results.append(result3)
    print()

    # Test 4: Parallel processing (8 workers)
    print("Test 4: Parallel Processing (8 workers)")
    print("-" * 40)
    processor_par8 = CPUIntensiveProcessor(
        batch_size=50,
        parallel_workers=8,
        enable_progress=True
    )
    result4 = run_benchmark(processor_par8, test_data, "Parallel (8 workers)")
    benchmark_results.append(result4)
    print()

    # Performance comparison
    print("Performance Comparison:")
    print("=" * 60)
    print(f"{'Configuration':<30} {'Time (s)':<12} {'Throughput':<15} {'Speedup':<10}")
    print("-" * 60)

    baseline_time = benchmark_results[0]['wall_time']

    for result in benchmark_results:
        speedup = baseline_time / result['wall_time'] if result['wall_time'] > 0 else 0
        print(f"{result['label']:<30} {result['wall_time']:<12.2f} {result['throughput']:<15.0f} {speedup:<10.2f}x")

    print()

    # Optimal configuration
    print("Recommendations:")
    print("-" * 40)
    best_result = max(benchmark_results, key=lambda r: r['throughput'])
    print(f"  Best configuration: {best_result['label']}")
    print(f"  Throughput: {best_result['throughput']:.0f} records/sec")
    print(f"  Speedup over sequential: {baseline_time / best_result['wall_time']:.2f}x")
    print()

    # Guidelines
    print("General Guidelines:")
    print("-" * 40)
    print("  • For CPU-intensive tasks: Use workers = CPU cores")
    print("  • For I/O-intensive tasks: Use workers = 2-4x CPU cores")
    print("  • Start with 4 workers and tune based on profiling")
    print("  • Monitor CPU usage to find optimal configuration")
    print("  • Larger batches = less overhead, but less parallelism")
    print()

    print("=" * 60)
    print("Example complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
