"""
Performance Benchmarks and Batch Processing Tests

This test suite validates:
- Batch processing (10,000+ calculations)
- Database query benchmarks
- API response time monitoring
- Throughput testing
- Memory usage profiling
- Scalability testing

Target: Performance validation at scale
Uses pytest-benchmark for accurate performance measurements
"""

import pytest
import sqlite3
import tempfile
import shutil
from pathlib import Path
import sys
import time
import random
from concurrent.futures import ThreadPoolExecutor
import psutil
import os

sys.path.insert(0, str(Path(__file__).parent.parent))

from greenlang.db.emission_factors_schema import create_database
from greenlang.sdk.emission_factor_client import EmissionFactorClient


# ==================== FIXTURES ====================

@pytest.fixture
def temp_db():
    """Create temporary database for testing."""
    temp_dir = tempfile.mkdtemp()
    db_path = Path(temp_dir) / "test_emission_factors.db"

    create_database(str(db_path))

    yield str(db_path)

    shutil.rmtree(temp_dir)


@pytest.fixture
def benchmark_db(temp_db):
    """Create database with benchmark data."""
    conn = sqlite3.connect(temp_db)
    cursor = conn.cursor()

    # Insert 100 emission factors for benchmarking
    for i in range(100):
        factor_id = f"benchmark_factor_{i:04d}"
        cursor.execute("""
            INSERT INTO emission_factors (
                factor_id, name, category, subcategory,
                emission_factor_value, unit, scope,
                source_org, source_uri, standard,
                last_updated, year_applicable,
                geographic_scope, geography_level,
                data_quality_tier
            ) VALUES (?, ?, 'fuels', 'diesel', ?, 'gallon', 'Scope 1',
                'EPA', 'https://epa.gov', 'GHG Protocol',
                '2024-01-01', 2024,
                'United States', 'Country', 'Tier 1')
        """, (factor_id, f"Benchmark Factor {i}", random.uniform(9.0, 11.0)))

    conn.commit()
    conn.close()

    return temp_db


# ==================== BATCH PROCESSING BENCHMARKS ====================

class TestBatchProcessingBenchmarks:
    """Benchmark batch processing performance."""

    def test_batch_1000_calculations(self, benchmark_db, benchmark):
        """Benchmark 1,000 calculations."""
        client = EmissionFactorClient(db_path=benchmark_db)

        def run_batch():
            results = []
            for i in range(1000):
                factor_id = f"benchmark_factor_{i % 100:04d}"
                result = client.calculate_emissions(
                    factor_id=factor_id,
                    activity_amount=random.uniform(10.0, 1000.0),
                    activity_unit='gallon'
                )
                results.append(result)
            return results

        # Benchmark
        results = benchmark(run_batch)

        assert len(results) == 1000

        client.close()

    def test_batch_10000_calculations(self, benchmark_db):
        """Test 10,000+ calculations performance."""
        client = EmissionFactorClient(db_path=benchmark_db)

        start_time = time.perf_counter()

        results = []
        for i in range(10000):
            factor_id = f"benchmark_factor_{i % 100:04d}"
            result = client.calculate_emissions(
                factor_id=factor_id,
                activity_amount=random.uniform(10.0, 1000.0),
                activity_unit='gallon'
            )
            results.append(result)

        elapsed = time.perf_counter() - start_time

        # Performance metrics
        throughput = 10000 / elapsed  # calculations per second
        avg_time_ms = (elapsed / 10000) * 1000  # ms per calculation

        print(f"\n10,000 Calculations Performance:")
        print(f"  Total Time: {elapsed:.2f}s")
        print(f"  Throughput: {throughput:.0f} calc/sec")
        print(f"  Avg Time: {avg_time_ms:.2f}ms per calculation")

        # Assertions
        assert len(results) == 10000
        assert throughput >= 100, "Should process >= 100 calculations/sec"
        assert avg_time_ms < 10, "Average time should be < 10ms per calculation"

        client.close()

    def test_batch_parallel_processing(self, benchmark_db):
        """Test parallel batch processing with multiple threads."""
        def process_batch(db_path, start_idx, count):
            client = EmissionFactorClient(db_path=db_path)
            results = []

            for i in range(start_idx, start_idx + count):
                factor_id = f"benchmark_factor_{i % 100:04d}"
                result = client.calculate_emissions(
                    factor_id=factor_id,
                    activity_amount=random.uniform(10.0, 1000.0),
                    activity_unit='gallon'
                )
                results.append(result)

            client.close()
            return results

        # Process 10,000 calculations across 10 threads
        num_threads = 10
        batch_size = 1000

        start_time = time.perf_counter()

        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = []

            for i in range(num_threads):
                future = executor.submit(
                    process_batch,
                    benchmark_db,
                    i * batch_size,
                    batch_size
                )
                futures.append(future)

            # Wait for all to complete
            all_results = []
            for future in futures:
                results = future.result()
                all_results.extend(results)

        elapsed = time.perf_counter() - start_time

        # Performance metrics
        total_calculations = num_threads * batch_size
        throughput = total_calculations / elapsed

        print(f"\nParallel Processing Performance:")
        print(f"  Threads: {num_threads}")
        print(f"  Total Calculations: {total_calculations}")
        print(f"  Total Time: {elapsed:.2f}s")
        print(f"  Throughput: {throughput:.0f} calc/sec")

        # Assertions
        assert len(all_results) == total_calculations
        assert throughput >= 500, "Parallel processing should achieve >= 500 calc/sec"


# ==================== DATABASE QUERY BENCHMARKS ====================

class TestDatabaseQueryBenchmarks:
    """Benchmark database query performance."""

    def test_factor_lookup_benchmark(self, benchmark_db, benchmark):
        """Benchmark factor lookup performance."""
        client = EmissionFactorClient(db_path=benchmark_db)

        def lookup_factor():
            factor_id = f"benchmark_factor_{random.randint(0, 99):04d}"
            return client.get_factor(factor_id)

        # Benchmark
        result = benchmark(lookup_factor)

        assert result.factor_id.startswith('benchmark_factor_')

        client.close()

    def test_category_query_benchmark(self, benchmark_db, benchmark):
        """Benchmark category query performance."""
        client = EmissionFactorClient(db_path=benchmark_db)

        def query_category():
            return client.get_by_category('fuels')

        # Benchmark
        results = benchmark(query_category)

        assert len(results) == 100

        client.close()

    def test_search_benchmark(self, benchmark_db, benchmark):
        """Benchmark search query performance."""
        from greenlang.models.emission_factor import FactorSearchCriteria

        client = EmissionFactorClient(db_path=benchmark_db)

        def search_factors():
            criteria = FactorSearchCriteria(
                category='fuels',
                scope='Scope 1'
            )
            return client.search_factors(criteria)

        # Benchmark
        results = benchmark(search_factors)

        assert len(results) > 0

        client.close()

    def test_bulk_factor_retrieval(self, benchmark_db):
        """Test bulk retrieval of multiple factors."""
        client = EmissionFactorClient(db_path=benchmark_db)

        factor_ids = [f"benchmark_factor_{i:04d}" for i in range(100)]

        start_time = time.perf_counter()

        factors = []
        for factor_id in factor_ids:
            factor = client.get_factor(factor_id)
            factors.append(factor)

        elapsed = time.perf_counter() - start_time

        # Performance metrics
        throughput = 100 / elapsed
        avg_time_ms = (elapsed / 100) * 1000

        print(f"\nBulk Factor Retrieval Performance:")
        print(f"  Factors Retrieved: 100")
        print(f"  Total Time: {elapsed:.2f}s")
        print(f"  Throughput: {throughput:.0f} factors/sec")
        print(f"  Avg Time: {avg_time_ms:.2f}ms per factor")

        # Assertions
        assert len(factors) == 100
        assert avg_time_ms < 10, "Should be < 10ms per factor lookup"

        client.close()


# ==================== MEMORY USAGE BENCHMARKS ====================

class TestMemoryUsage:
    """Test memory usage for large-scale operations."""

    def test_memory_usage_large_batch(self, benchmark_db):
        """Test memory usage for processing large batch."""
        process = psutil.Process(os.getpid())

        # Initial memory
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB

        client = EmissionFactorClient(db_path=benchmark_db)

        # Process 10,000 calculations
        results = []
        for i in range(10000):
            factor_id = f"benchmark_factor_{i % 100:04d}"
            result = client.calculate_emissions(
                factor_id=factor_id,
                activity_amount=100.0,
                activity_unit='gallon'
            )
            results.append(result)

        # Final memory
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory

        print(f"\nMemory Usage:")
        print(f"  Initial: {initial_memory:.2f} MB")
        print(f"  Final: {final_memory:.2f} MB")
        print(f"  Increase: {memory_increase:.2f} MB")
        print(f"  Per Calculation: {memory_increase / 10000 * 1024:.2f} KB")

        # Assertions
        assert memory_increase < 500, "Memory increase should be < 500 MB for 10k calculations"

        client.close()

    def test_memory_cleanup_after_batch(self, benchmark_db):
        """Test that memory is released after batch processing."""
        process = psutil.Process(os.getpid())

        initial_memory = process.memory_info().rss / 1024 / 1024

        # Process batch in scope
        client = EmissionFactorClient(db_path=benchmark_db)

        for i in range(5000):
            factor_id = f"benchmark_factor_{i % 100:04d}"
            client.calculate_emissions(
                factor_id=factor_id,
                activity_amount=100.0,
                activity_unit='gallon'
            )

        client.close()

        # Force garbage collection
        import gc
        gc.collect()

        # Check memory after cleanup
        time.sleep(1)
        final_memory = process.memory_info().rss / 1024 / 1024
        memory_retained = final_memory - initial_memory

        print(f"\nMemory Cleanup:")
        print(f"  Initial: {initial_memory:.2f} MB")
        print(f"  Final: {final_memory:.2f} MB")
        print(f"  Retained: {memory_retained:.2f} MB")

        # Memory retention should be minimal
        assert memory_retained < 100, "Should not retain > 100 MB after cleanup"


# ==================== SCALABILITY BENCHMARKS ====================

class TestScalability:
    """Test scalability characteristics."""

    def test_linear_scaling_calculations(self, benchmark_db):
        """Test that calculation time scales linearly."""
        client = EmissionFactorClient(db_path=benchmark_db)

        batch_sizes = [100, 500, 1000, 2000]
        times = []

        for size in batch_sizes:
            start_time = time.perf_counter()

            for i in range(size):
                factor_id = f"benchmark_factor_{i % 100:04d}"
                client.calculate_emissions(
                    factor_id=factor_id,
                    activity_amount=100.0,
                    activity_unit='gallon'
                )

            elapsed = time.perf_counter() - start_time
            times.append(elapsed)

        # Calculate scaling factor
        time_per_calc = [times[i] / batch_sizes[i] for i in range(len(batch_sizes))]

        print(f"\nScaling Analysis:")
        for i in range(len(batch_sizes)):
            print(f"  {batch_sizes[i]} calcs: {times[i]:.2f}s ({time_per_calc[i] * 1000:.2f}ms per calc)")

        # Time per calculation should be relatively constant (linear scaling)
        max_variation = max(time_per_calc) - min(time_per_calc)
        assert max_variation < 0.005, "Scaling should be near-linear"

        client.close()

    def test_database_size_impact(self, temp_db):
        """Test performance impact of database size."""
        # Create databases with different sizes
        sizes = [100, 500, 1000]
        lookup_times = []

        for size in sizes:
            # Populate database
            conn = sqlite3.connect(temp_db)
            cursor = conn.cursor()

            # Clear existing data
            cursor.execute("DELETE FROM emission_factors")

            # Insert factors
            for i in range(size):
                factor_id = f"factor_{i:05d}"
                cursor.execute("""
                    INSERT INTO emission_factors (
                        factor_id, name, category, subcategory,
                        emission_factor_value, unit, scope,
                        source_org, source_uri, standard,
                        last_updated, year_applicable,
                        geographic_scope, data_quality_tier
                    ) VALUES (?, ?, 'fuels', 'diesel', 10.21, 'gallon', 'Scope 1',
                        'EPA', 'https://epa.gov', 'GHG Protocol',
                        '2024-01-01', 2024,
                        'United States', 'Tier 1')
                """, (factor_id, f"Factor {i}"))

            conn.commit()
            conn.close()

            # Benchmark lookup
            client = EmissionFactorClient(db_path=temp_db)

            start_time = time.perf_counter()

            for _ in range(100):
                factor_id = f"factor_{random.randint(0, size - 1):05d}"
                client.get_factor(factor_id)

            elapsed = time.perf_counter() - start_time
            avg_time_ms = (elapsed / 100) * 1000

            lookup_times.append(avg_time_ms)

            client.close()

        print(f"\nDatabase Size Impact:")
        for i in range(len(sizes)):
            print(f"  {sizes[i]} factors: {lookup_times[i]:.2f}ms avg lookup")

        # Lookup time should remain relatively constant with proper indexing
        assert all(t < 10 for t in lookup_times), "All lookups should be < 10ms regardless of DB size"


# ==================== STRESS TESTS ====================

class TestStressConditions:
    """Test performance under stress conditions."""

    def test_sustained_load(self, benchmark_db):
        """Test sustained load over extended period."""
        client = EmissionFactorClient(db_path=benchmark_db)

        duration_seconds = 10
        calculation_count = 0
        start_time = time.perf_counter()

        while (time.perf_counter() - start_time) < duration_seconds:
            factor_id = f"benchmark_factor_{random.randint(0, 99):04d}"
            client.calculate_emissions(
                factor_id=factor_id,
                activity_amount=100.0,
                activity_unit='gallon'
            )
            calculation_count += 1

        elapsed = time.perf_counter() - start_time
        throughput = calculation_count / elapsed

        print(f"\nSustained Load Test:")
        print(f"  Duration: {duration_seconds}s")
        print(f"  Calculations: {calculation_count}")
        print(f"  Throughput: {throughput:.0f} calc/sec")

        # Should maintain consistent throughput
        assert throughput >= 100, "Should maintain >= 100 calc/sec under sustained load"

        client.close()

    def test_rapid_connection_cycling(self, benchmark_db):
        """Test rapid connect/disconnect cycles."""
        start_time = time.perf_counter()

        for _ in range(100):
            client = EmissionFactorClient(db_path=benchmark_db)
            factor = client.get_factor('benchmark_factor_0000')
            assert factor.factor_id == 'benchmark_factor_0000'
            client.close()

        elapsed = time.perf_counter() - start_time

        print(f"\nConnection Cycling:")
        print(f"  Cycles: 100")
        print(f"  Time: {elapsed:.2f}s")
        print(f"  Avg per cycle: {elapsed / 100 * 1000:.2f}ms")

        # Should handle connection cycling efficiently
        assert elapsed < 5, "100 connection cycles should complete in < 5s"


# Run tests
if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short', '--benchmark-only'])
