"""
Oracle Throughput Performance Tests
GL-VCCI Scope 3 Platform

Performance tests validating Oracle connector can extract 100K records/hour.
Tests extraction rate, batch processing, pagination performance, and memory usage.

Version: 1.0.0
Phase: 4 (Weeks 24-26)
Date: 2025-11-06
"""

import pytest
import time
import psutil
from typing import List, Dict, Any
from datetime import datetime

from oracle.client import OracleRESTClient, create_query
from oracle.mappers.po_mapper import PurchaseOrderMapper


# Target: 100K records in 60 minutes (1,667 records/min = 27.8 records/sec)
TARGET_THROUGHPUT_PER_HOUR = 100_000
TARGET_THROUGHPUT_PER_SECOND = TARGET_THROUGHPUT_PER_HOUR / 3600


@pytest.mark.performance
@pytest.mark.slow
@pytest.mark.timeout(3600)  # 1 hour max
@pytest.mark.oracle_sandbox
class TestOracle100KThroughput:
    """Test extraction of 100K records in <1 hour."""

    def test_oracle_100k_throughput(
        self,
        oracle_client: OracleRESTClient,
        performance_metrics
    ):
        """
        Test extraction of 100K records from Oracle in under 1 hour.

        Target: 100K records/hour = 1,667 records/min = 27.8 records/sec
        """
        target_records = 100_000
        batch_size = 1000
        records_extracted = 0

        print(f"\n{'='*60}")
        print(f"Starting Oracle 100K Throughput Test")
        print(f"Target: {target_records:,} records in <3600 seconds")
        print(f"Batch size: {batch_size}")
        print(f"{'='*60}\n")

        start_time = time.time()

        # Extract records in batches
        try:
            for batch_num, batch in enumerate(
                oracle_client.query_paginated("purchase_orders", {"limit": batch_size}),
                start=1
            ):
                batch_start = time.time()
                batch_size_actual = len(batch)
                records_extracted += batch_size_actual
                batch_elapsed = time.time() - batch_start

                # Record metrics
                performance_metrics.record_request(
                    latency_ms=batch_elapsed * 1000,
                    records_count=batch_size_actual
                )
                performance_metrics.record_memory()

                # Progress reporting every 10 batches
                if batch_num % 10 == 0:
                    elapsed = time.time() - start_time
                    current_rate = records_extracted / elapsed if elapsed > 0 else 0
                    estimated_total_time = (target_records / current_rate) if current_rate > 0 else 0

                    print(f"Batch {batch_num}: {records_extracted:,} records | "
                          f"Rate: {current_rate:.1f} rec/s | "
                          f"Est. total: {estimated_total_time/60:.1f} min")

                # Stop when target reached
                if records_extracted >= target_records:
                    break

                # Safety check: stop if taking too long
                if time.time() - start_time > 3600:
                    print("\nTimeout reached (3600s)")
                    break

        finally:
            performance_metrics.finalize()
            elapsed = performance_metrics.duration_seconds
            throughput_per_hour = performance_metrics.throughput_per_hour

            # Print results
            print(f"\n{'='*60}")
            print(f"Oracle 100K Throughput Test Results")
            print(f"{'='*60}")
            print(f"Records extracted: {records_extracted:,}")
            print(f"Time elapsed: {elapsed:.2f} seconds ({elapsed/60:.2f} minutes)")
            print(f"Throughput: {throughput_per_hour:,.0f} records/hour")
            print(f"Average rate: {performance_metrics.throughput_per_second:.2f} records/second")
            print(f"API calls: {performance_metrics.requests_made}")
            print(f"Average latency: {performance_metrics.avg_latency_ms:.2f} ms")
            print(f"P95 latency: {performance_metrics.p95_latency_ms:.2f} ms")
            print(f"P99 latency: {performance_metrics.p99_latency_ms:.2f} ms")
            print(f"Max memory: {performance_metrics.max_memory_mb:.2f} MB")
            print(f"Errors: {performance_metrics.errors}")
            print(f"{'='*60}\n")

            # Assertions
            assert records_extracted >= target_records, \
                f"Did not extract enough records: {records_extracted:,} < {target_records:,}"

            assert throughput_per_hour >= TARGET_THROUGHPUT_PER_HOUR, \
                f"Throughput {throughput_per_hour:,.0f} < target {TARGET_THROUGHPUT_PER_HOUR:,}/hour"

            assert elapsed <= 3600, \
                f"Took {elapsed:.0f}s, expected <3600s"


@pytest.mark.performance
@pytest.mark.oracle_sandbox
class TestOracleBatchPerformance:
    """Test Oracle batch processing performance."""

    def test_batch_extraction_performance(
        self,
        oracle_client: OracleRESTClient,
        performance_metrics
    ):
        """Test extraction performance with different batch sizes."""
        batch_sizes = [100, 500, 1000, 2000]
        results = {}

        for batch_size in batch_sizes:
            records_extracted = 0
            batches_processed = 0
            max_batches = 10  # Test first 10 batches

            start_time = time.time()

            for batch in oracle_client.query_paginated("purchase_orders", {"limit": batch_size}):
                records_extracted += len(batch)
                batches_processed += 1

                if batches_processed >= max_batches:
                    break

            elapsed = time.time() - start_time
            rate = records_extracted / elapsed if elapsed > 0 else 0

            results[batch_size] = {
                "records": records_extracted,
                "batches": batches_processed,
                "elapsed": elapsed,
                "rate": rate
            }

            print(f"\nBatch size {batch_size}: {rate:.1f} records/sec "
                  f"({records_extracted} records in {elapsed:.2f}s)")

        # Larger batch sizes should generally be faster
        assert results[1000]["rate"] > results[100]["rate"] * 0.5, \
            "Batch size optimization not effective"


@pytest.mark.performance
@pytest.mark.oracle_sandbox
class TestOraclePaginationPerformance:
    """Test Oracle pagination performance."""

    def test_pagination_with_links(
        self,
        oracle_client: OracleRESTClient,
        performance_metrics
    ):
        """Test Oracle pagination using links array."""
        batch_size = 1000
        batches_to_test = 20

        start_time = time.time()
        total_records = 0

        for batch_num, batch in enumerate(
            oracle_client.query_paginated("purchase_orders", {"limit": batch_size}),
            start=1
        ):
            total_records += len(batch)

            if batch_num >= batches_to_test:
                break

        elapsed = time.time() - start_time
        avg_time_per_batch = elapsed / batches_to_test

        print(f"\nPagination Performance:")
        print(f"  Batches: {batches_to_test}")
        print(f"  Total records: {total_records:,}")
        print(f"  Total time: {elapsed:.2f}s")
        print(f"  Avg time per batch: {avg_time_per_batch:.3f}s")

        # Pagination overhead should be reasonable
        assert avg_time_per_batch < 5.0, \
            f"Pagination too slow: {avg_time_per_batch:.3f}s per batch"


@pytest.mark.performance
@pytest.mark.oracle_sandbox
class TestOracleMappingPerformance:
    """Test Oracle data mapping performance."""

    def test_mapping_performance(
        self,
        oracle_client: OracleRESTClient,
        performance_metrics
    ):
        """Test performance of data mapping."""
        # Extract raw data
        query = create_query().limit(1000)
        response = oracle_client.get("purchase_orders", query.build())
        raw_records = response.get("items", [])

        # Measure mapping performance
        mapper = PurchaseOrderMapper()
        start_time = time.time()

        mapped_records = []
        for record in raw_records:
            mapped = mapper.map(record)
            mapped_records.append(mapped)

        elapsed = time.time() - start_time
        rate = len(raw_records) / elapsed if elapsed > 0 else 0

        print(f"\nMapping Performance:")
        print(f"  Records: {len(raw_records):,}")
        print(f"  Time: {elapsed:.3f}s")
        print(f"  Rate: {rate:.1f} records/sec")

        # Mapping should be fast (>1000 records/sec)
        assert rate > 1000, \
            f"Mapping too slow: {rate:.1f} records/sec"


@pytest.mark.performance
@pytest.mark.oracle_sandbox
class TestOracleMemoryUsage:
    """Test Oracle memory usage during large extractions."""

    def test_memory_usage_large_extraction(
        self,
        oracle_client: OracleRESTClient,
        memory_monitor
    ):
        """Test memory usage during extraction of large dataset."""
        batch_size = 1000
        batches_to_extract = 50  # 50K records
        memory_samples = []

        print(f"\nMemory Usage Test:")
        print(f"  Initial: {memory_monitor()['current_mb']:.2f} MB")

        for batch_num, batch in enumerate(
            oracle_client.query_paginated("purchase_orders", {"limit": batch_size}),
            start=1
        ):
            # Sample memory every 10 batches
            if batch_num % 10 == 0:
                mem_info = memory_monitor()
                memory_samples.append(mem_info)
                print(f"  Batch {batch_num}: {mem_info['current_mb']:.2f} MB "
                      f"(+{mem_info['delta_mb']:.2f} MB)")

            if batch_num >= batches_to_extract:
                break

        final_memory = memory_monitor()
        print(f"  Final: {final_memory['current_mb']:.2f} MB "
              f"(+{final_memory['delta_mb']:.2f} MB)")

        # Memory growth should be reasonable (<500 MB increase)
        assert final_memory['delta_mb'] < 500, \
            f"Excessive memory growth: {final_memory['delta_mb']:.2f} MB"


@pytest.mark.performance
@pytest.mark.oracle_sandbox
class TestOracleAPIPerformance:
    """Test Oracle REST API performance characteristics."""

    def test_api_response_time(
        self,
        oracle_client: OracleRESTClient,
        performance_metrics
    ):
        """Test Oracle API response time distribution."""
        num_requests = 50
        batch_size = 100

        for i in range(num_requests):
            start = time.time()
            query = create_query().limit(batch_size).offset(i * batch_size)
            oracle_client.get("purchase_orders", query.build())
            elapsed_ms = (time.time() - start) * 1000

            performance_metrics.record_request(elapsed_ms, batch_size)

        # Calculate statistics
        print(f"\nAPI Response Time ({num_requests} requests):")
        print(f"  Mean: {performance_metrics.avg_latency_ms:.2f} ms")
        print(f"  P95: {performance_metrics.p95_latency_ms:.2f} ms")
        print(f"  P99: {performance_metrics.p99_latency_ms:.2f} ms")

        # P95 latency should be reasonable (<2000ms)
        assert performance_metrics.p95_latency_ms < 2000, \
            f"P95 latency too high: {performance_metrics.p95_latency_ms:.2f} ms"


@pytest.mark.performance
@pytest.mark.oracle_sandbox
class TestOracleReliability:
    """Test Oracle extraction reliability."""

    def test_extraction_reliability(
        self,
        oracle_client: OracleRESTClient,
        performance_metrics
    ):
        """Test reliability of extraction over multiple batches."""
        batch_size = 100
        target_batches = 50
        successful_batches = 0
        failed_batches = 0

        for batch_num, batch in enumerate(
            oracle_client.query_paginated("purchase_orders", {"limit": batch_size}),
            start=1
        ):
            try:
                assert len(batch) > 0
                successful_batches += 1
            except Exception as e:
                print(f"Batch {batch_num} failed: {e}")
                failed_batches += 1
                performance_metrics.record_error()

            if batch_num >= target_batches:
                break

        success_rate = successful_batches / target_batches

        print(f"\nReliability Test:")
        print(f"  Total batches: {target_batches}")
        print(f"  Successful: {successful_batches}")
        print(f"  Failed: {failed_batches}")
        print(f"  Success rate: {success_rate*100:.2f}%")

        # Should have high success rate (>95%)
        assert success_rate >= 0.95, \
            f"Success rate too low: {success_rate*100:.2f}%"


@pytest.mark.performance
@pytest.mark.oracle_sandbox
class TestOracleScalability:
    """Test Oracle connector scalability."""

    def test_concurrent_endpoint_extraction(
        self,
        oracle_client: OracleRESTClient
    ):
        """Test concurrent extraction from multiple endpoints."""
        import concurrent.futures

        endpoints = ["purchase_orders", "requisitions"]
        results = {}

        def extract_endpoint(endpoint_name: str):
            """Extract from single endpoint."""
            records = []
            for batch in oracle_client.query_paginated(endpoint_name, {"limit": 100}):
                records.extend(batch)
                if len(records) >= 1000:
                    break
            return endpoint_name, records

        start_time = time.time()

        # Extract concurrently
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(endpoints)) as executor:
            futures = [executor.submit(extract_endpoint, ep) for ep in endpoints]

            for future in concurrent.futures.as_completed(futures):
                endpoint_name, records = future.result()
                results[endpoint_name] = records

        elapsed = time.time() - start_time
        total_records = sum(len(records) for records in results.values())

        print(f"\nConcurrent Extraction:")
        print(f"  Endpoints: {len(endpoints)}")
        print(f"  Total records: {total_records:,}")
        print(f"  Time: {elapsed:.2f}s")
        print(f"  Rate: {total_records/elapsed:.1f} records/sec")

        # Concurrent extraction should work
        assert len(results) == len(endpoints)
        assert total_records > 0
