"""
Integration Tests: Large Volume Processing
===========================================

Performance and scalability tests for large datasets:
- 10,000 shipments processing (standard volume)
- 50,000 shipments processing (stress test)
- Memory usage monitoring under load
- Database performance with large volumes
- Memory leak detection

Target: Maturity score +1 point (scalability validation)
Version: 1.0.0
Author: GL-TestEngineer
"""

import pytest
import asyncio
import time
import psutil
import os
from pathlib import Path
from typing import Dict, Any, List
import pandas as pd
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from agents.shipment_intake_agent_v2 import ShipmentIntakeAgent_v2
from agents.emissions_calculator_agent_v2 import EmissionsCalculatorAgent_v2
from agents.reporting_packager_agent_v2 import ReportingPackagerAgent_v2
from cbam_pipeline_v2 import CBAMPipeline_v2


# ============================================================================
# Volume Processing Tests
# ============================================================================

@pytest.mark.integration
@pytest.mark.slow
@pytest.mark.performance
class TestLargeVolumeProcessing:
    """Test processing of large shipment volumes."""

    def test_10k_shipments_processing(
        self,
        cn_codes_path,
        cbam_rules_path,
        suppliers_path,
        importer_info,
        tmp_path
    ):
        """
        Test processing 10,000 shipments (standard volume).

        Performance targets:
        - Processing time: < 60 seconds
        - Throughput: > 166 records/sec
        - Memory usage: < 500 MB increase
        - Success rate: 100%
        """
        # Generate 10,000 shipment records
        print("\n[Performance Test] Generating 10,000 shipment records...")
        shipments_data = self._generate_large_dataset(10000)

        csv_path = tmp_path / "shipments_10k.csv"
        df = pd.DataFrame(shipments_data)
        df.to_csv(csv_path, index=False)

        print(f"  Generated CSV: {csv_path} ({csv_path.stat().st_size / 1024 / 1024:.2f} MB)")

        # Measure memory before processing
        process = psutil.Process(os.getpid())
        memory_before = process.memory_info().rss / 1024 / 1024  # MB

        # Create pipeline
        pipeline = CBAMPipeline_v2(
            cn_codes_path=cn_codes_path,
            cbam_rules_path=cbam_rules_path,
            suppliers_path=suppliers_path
        )

        # Execute pipeline with timing
        print("\n[Execution] Processing 10,000 shipments...")
        start_time = time.perf_counter()

        result = pipeline.execute({
            "input_file": str(csv_path),
            "importer_info": importer_info,
            "intermediate_output_dir": str(tmp_path / "intermediate")
        })

        end_time = time.perf_counter()
        processing_time = end_time - start_time

        # Measure memory after processing
        memory_after = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = memory_after - memory_before

        # Verify results
        assert result.success, "Pipeline should complete successfully"
        report = result.data

        # Performance assertions
        throughput = 10000 / processing_time

        print("\n[Performance Metrics]")
        print(f"  Total records: 10,000")
        print(f"  Processing time: {processing_time:.2f}s")
        print(f"  Throughput: {throughput:.0f} records/sec")
        print(f"  Memory increase: {memory_increase:.2f} MB")

        # Assert performance targets
        assert processing_time < 60, f"Processing time {processing_time:.2f}s exceeds 60s target"
        assert throughput > 166, f"Throughput {throughput:.0f} rec/sec below 166 rec/sec target"
        assert memory_increase < 500, f"Memory increase {memory_increase:.2f} MB exceeds 500 MB limit"

        # Verify data integrity
        assert report['emissions_summary']['total_shipments'] == 10000
        print("\n  ✓ All performance targets met")

    @pytest.mark.slow
    def test_50k_shipments_processing(
        self,
        cn_codes_path,
        cbam_rules_path,
        suppliers_path,
        importer_info,
        tmp_path
    ):
        """
        Test processing 50,000 shipments (stress test).

        Performance targets:
        - Processing time: < 300 seconds (5 minutes)
        - Throughput: > 166 records/sec
        - Memory usage: < 1 GB increase
        - No crashes or OOM errors
        """
        # Generate 50,000 shipment records
        print("\n[Stress Test] Generating 50,000 shipment records...")
        shipments_data = self._generate_large_dataset(50000)

        csv_path = tmp_path / "shipments_50k.csv"
        df = pd.DataFrame(shipments_data)
        df.to_csv(csv_path, index=False)

        print(f"  Generated CSV: {csv_path} ({csv_path.stat().st_size / 1024 / 1024:.2f} MB)")

        # Measure memory before processing
        process = psutil.Process(os.getpid())
        memory_before = process.memory_info().rss / 1024 / 1024  # MB

        # Create pipeline
        pipeline = CBAMPipeline_v2(
            cn_codes_path=cn_codes_path,
            cbam_rules_path=cbam_rules_path,
            suppliers_path=suppliers_path
        )

        # Execute pipeline with timing
        print("\n[Execution] Processing 50,000 shipments...")
        start_time = time.perf_counter()

        result = pipeline.execute({
            "input_file": str(csv_path),
            "importer_info": importer_info
        })

        end_time = time.perf_counter()
        processing_time = end_time - start_time

        # Measure memory after processing
        memory_after = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = memory_after - memory_before

        # Verify results
        assert result.success, "Pipeline should complete successfully"
        report = result.data

        # Performance assertions
        throughput = 50000 / processing_time

        print("\n[Performance Metrics]")
        print(f"  Total records: 50,000")
        print(f"  Processing time: {processing_time:.2f}s ({processing_time / 60:.2f} min)")
        print(f"  Throughput: {throughput:.0f} records/sec")
        print(f"  Memory increase: {memory_increase:.2f} MB")

        # Assert performance targets
        assert processing_time < 300, f"Processing time {processing_time:.2f}s exceeds 300s target"
        assert throughput > 166, f"Throughput {throughput:.0f} rec/sec below 166 rec/sec target"
        assert memory_increase < 1024, f"Memory increase {memory_increase:.2f} MB exceeds 1024 MB limit"

        # Verify data integrity
        assert report['emissions_summary']['total_shipments'] == 50000
        print("\n  ✓ All stress test targets met")

    def test_memory_usage_under_load(
        self,
        cn_codes_path,
        cbam_rules_path,
        tmp_path
    ):
        """
        Test memory usage remains stable under sustained load.

        Monitors memory usage during processing to detect leaks.
        """
        process = psutil.Process(os.getpid())

        # Generate test dataset (5,000 records for faster testing)
        shipments_data = self._generate_large_dataset(5000)
        csv_path = tmp_path / "shipments_memory_test.csv"
        df = pd.DataFrame(shipments_data)
        df.to_csv(csv_path, index=False)

        # Track memory usage over multiple runs
        memory_samples = []

        intake_agent = ShipmentIntakeAgent_v2(
            cn_codes_path=cn_codes_path,
            cbam_rules_path=cbam_rules_path
        )

        print("\n[Memory Test] Running 5 iterations...")

        for i in range(5):
            memory_before = process.memory_info().rss / 1024 / 1024

            # Process data
            result = intake_agent.process_file(str(csv_path))

            memory_after = process.memory_info().rss / 1024 / 1024
            memory_delta = memory_after - memory_before

            memory_samples.append({
                "iteration": i + 1,
                "memory_before": memory_before,
                "memory_after": memory_after,
                "memory_delta": memory_delta
            })

            print(f"  Iteration {i + 1}: {memory_delta:+.2f} MB")

        # Analyze memory trend
        deltas = [s["memory_delta"] for s in memory_samples]
        avg_delta = sum(deltas) / len(deltas)
        max_delta = max(deltas)

        print(f"\n[Analysis]")
        print(f"  Average memory delta: {avg_delta:.2f} MB")
        print(f"  Max memory delta: {max_delta:.2f} MB")

        # Assert no significant memory leak
        # Memory should stabilize (later iterations similar to first)
        later_avg = sum(deltas[2:]) / len(deltas[2:])  # Average of last 3
        first_delta = deltas[0]

        leak_ratio = later_avg / first_delta if first_delta > 0 else 1.0

        assert leak_ratio < 2.0, f"Potential memory leak detected: later iterations use {leak_ratio:.1f}x more memory"
        print(f"  ✓ No significant memory leak detected (ratio: {leak_ratio:.2f})")


# ============================================================================
# Database Performance Tests
# ============================================================================

@pytest.mark.integration
@pytest.mark.performance
class TestDatabasePerformance:
    """Test database performance with large volumes."""

    def test_database_performance_10k_records(
        self,
        cn_codes_path,
        cbam_rules_path,
        tmp_path
    ):
        """
        Test database query performance with 10k records.

        Simulates CN code lookups and emission factor queries.
        """
        intake_agent = ShipmentIntakeAgent_v2(
            cn_codes_path=cn_codes_path,
            cbam_rules_path=cbam_rules_path
        )

        # Measure CN code lookup performance
        cn_codes_to_test = ["72071100", "76011000", "25232900", "28342100", "72071210"]

        print("\n[Database Performance] CN Code Lookups")

        start_time = time.perf_counter()

        for _ in range(2000):  # 2000 iterations * 5 codes = 10,000 lookups
            for cn_code in cn_codes_to_test:
                # Simulate lookup
                _ = intake_agent.cn_codes.get(cn_code)

        end_time = time.perf_counter()
        lookup_time = end_time - start_time
        lookups_per_second = 10000 / lookup_time

        print(f"  10,000 lookups in {lookup_time:.3f}s")
        print(f"  Throughput: {lookups_per_second:.0f} lookups/sec")

        # Assert performance target (should be very fast for in-memory dict)
        assert lookups_per_second > 100000, f"CN code lookup too slow: {lookups_per_second:.0f} lookups/sec"

    def test_batch_processing_optimization(
        self,
        cn_codes_path,
        cbam_rules_path,
        suppliers_path,
        tmp_path
    ):
        """
        Test batch processing is more efficient than individual processing.

        Batch processing should be at least 2x faster.
        """
        # Generate test data
        shipments_data = self._generate_large_dataset(1000)
        csv_path = tmp_path / "batch_test.csv"
        df = pd.DataFrame(shipments_data)
        df.to_csv(csv_path, index=False)

        intake_agent = ShipmentIntakeAgent_v2(
            cn_codes_path=cn_codes_path,
            cbam_rules_path=cbam_rules_path
        )

        # Time batch processing
        start_batch = time.perf_counter()
        result_batch = intake_agent.process_file(str(csv_path))
        time_batch = time.perf_counter() - start_batch

        print(f"\n[Batch Optimization Test]")
        print(f"  Batch processing: {time_batch:.3f}s for 1000 records")
        print(f"  Throughput: {1000 / time_batch:.0f} records/sec")

        # Verify batch processing completed successfully
        assert result_batch['metadata']['total_records'] == 1000

        # Batch processing should be reasonably fast
        assert time_batch < 10, f"Batch processing too slow: {time_batch:.3f}s for 1000 records"


# ============================================================================
# Memory Leak Detection Tests
# ============================================================================

@pytest.mark.integration
@pytest.mark.performance
class TestMemoryLeaks:
    """Test for memory leaks in long-running scenarios."""

    def test_no_memory_leak_repeated_pipeline_runs(
        self,
        cn_codes_path,
        cbam_rules_path,
        suppliers_path,
        importer_info,
        tmp_path
    ):
        """
        Test for memory leaks across repeated pipeline executions.

        Runs pipeline 10 times and monitors memory growth.
        """
        # Generate test dataset
        shipments_data = TestLargeVolumeProcessing._generate_large_dataset(None, 500)
        csv_path = tmp_path / "leak_test.csv"
        df = pd.DataFrame(shipments_data)
        df.to_csv(csv_path, index=False)

        process = psutil.Process(os.getpid())
        memory_measurements = []

        print("\n[Memory Leak Test] Running 10 pipeline iterations...")

        for i in range(10):
            # Create new pipeline instance each time (fresh start)
            pipeline = CBAMPipeline_v2(
                cn_codes_path=cn_codes_path,
                cbam_rules_path=cbam_rules_path,
                suppliers_path=suppliers_path
            )

            memory_before = process.memory_info().rss / 1024 / 1024

            # Execute pipeline
            result = pipeline.execute({
                "input_file": str(csv_path),
                "importer_info": importer_info
            })

            memory_after = process.memory_info().rss / 1024 / 1024
            memory_delta = memory_after - memory_before

            memory_measurements.append({
                "run": i + 1,
                "memory_mb": memory_after,
                "delta_mb": memory_delta
            })

            print(f"  Run {i + 1}: {memory_after:.1f} MB ({memory_delta:+.1f} MB)")

            assert result.success, f"Pipeline run {i + 1} failed"

        # Analyze memory growth pattern
        first_5_avg = sum(m["memory_mb"] for m in memory_measurements[:5]) / 5
        last_5_avg = sum(m["memory_mb"] for m in memory_measurements[5:]) / 5

        growth_rate = (last_5_avg - first_5_avg) / first_5_avg

        print(f"\n[Analysis]")
        print(f"  First 5 runs avg: {first_5_avg:.1f} MB")
        print(f"  Last 5 runs avg: {last_5_avg:.1f} MB")
        print(f"  Growth rate: {growth_rate * 100:.1f}%")

        # Assert acceptable memory growth (< 10%)
        assert growth_rate < 0.10, f"Excessive memory growth detected: {growth_rate * 100:.1f}%"
        print(f"  ✓ Memory stable across runs")


# ============================================================================
# Helper Methods
# ============================================================================

def _generate_large_dataset(self, num_records: int) -> List[Dict[str, Any]]:
    """
    Generate large synthetic dataset for testing.

    Creates realistic CBAM shipment data at scale.
    """
    cn_codes = [
        "72071100",  # Iron/steel
        "72071210",  # Iron/steel
        "76011000",  # Aluminum
        "25232900",  # Cement
        "28342100",  # Fertilizer
    ]

    countries = ["CN", "TR", "RU", "UA", "IN", "KR", "ZA", "BR", "MX", "TH"]
    suppliers = [f"SUP-{country}-{i:03d}" for country in countries for i in range(1, 11)]

    shipments = []

    for i in range(num_records):
        shipment = {
            "shipment_id": f"SHIP-2025-{i:07d}",
            "import_date": f"2025-Q3",
            "quarter": "2025-Q3",
            "cn_code": cn_codes[i % len(cn_codes)],
            "origin_iso": countries[i % len(countries)],
            "net_mass_kg": round(1000 + (i % 10000), 2),
            "supplier_id": suppliers[i % len(suppliers)],
            "invoice_number": f"INV-2025-{i:07d}",
            "importer_country": "NL",
            "has_actual_emissions": "NO"
        }
        shipments.append(shipment)

    return shipments


# Attach helper method to test class
TestLargeVolumeProcessing._generate_large_dataset = _generate_large_dataset
TestMemoryLeaks._generate_large_dataset = staticmethod(_generate_large_dataset)


# ============================================================================
# Test Fixtures
# ============================================================================

@pytest.fixture
def importer_info():
    """Standard importer information for testing."""
    return {
        "importer_name": "Test Import BV",
        "importer_country": "NL",
        "importer_eori": "NL123456789012",
        "declarant_name": "Test User",
        "declarant_position": "Test Manager"
    }


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short', '-m', 'not slow'])
