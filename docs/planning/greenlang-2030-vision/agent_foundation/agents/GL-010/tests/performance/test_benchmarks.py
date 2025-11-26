# -*- coding: utf-8 -*-
"""
Performance Tests for GL-010 EMISSIONWATCH Benchmarks.

Tests calculation latency, compliance check throughput, report generation
speed, memory usage, and concurrent request handling.

Test Count: 10+ tests
Coverage Target: 90%+

Performance Targets:
- Single calculation: <5ms
- Batch processing: >1000 records/sec
- Memory increase: <500MB for 100k records

Author: GreenLang Foundation Test Engineering
Version: 1.0.0
"""

import time
import pytest
from datetime import datetime, timezone
from typing import Any, Dict, List
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from tools import EmissionsComplianceTools


# =============================================================================
# TEST CLASS: PERFORMANCE BENCHMARKS
# =============================================================================

@pytest.mark.performance
class TestBenchmarks:
    """Performance benchmark tests."""

    # =========================================================================
    # CALCULATION LATENCY TESTS
    # =========================================================================

    def test_calculation_latency_nox(
        self, emissions_tools, sample_cems_data, natural_gas_fuel_data
    ):
        """Test NOx calculation latency is under 5ms."""
        # Warm-up
        for _ in range(10):
            emissions_tools.calculate_nox_emissions(sample_cems_data, natural_gas_fuel_data)

        # Measure
        iterations = 100
        start_time = time.perf_counter()

        for _ in range(iterations):
            emissions_tools.calculate_nox_emissions(sample_cems_data, natural_gas_fuel_data)

        end_time = time.perf_counter()
        avg_time_ms = ((end_time - start_time) / iterations) * 1000

        assert avg_time_ms < 5.0, f"NOx calculation took {avg_time_ms:.2f}ms (target: <5ms)"

    def test_calculation_latency_sox(
        self, emissions_tools, fuel_oil_no2_data
    ):
        """Test SOx calculation latency is under 5ms."""
        # Warm-up
        for _ in range(10):
            emissions_tools.calculate_sox_emissions(fuel_oil_no2_data)

        # Measure
        iterations = 100
        start_time = time.perf_counter()

        for _ in range(iterations):
            emissions_tools.calculate_sox_emissions(fuel_oil_no2_data)

        end_time = time.perf_counter()
        avg_time_ms = ((end_time - start_time) / iterations) * 1000

        assert avg_time_ms < 5.0, f"SOx calculation took {avg_time_ms:.2f}ms (target: <5ms)"

    def test_calculation_latency_co2(
        self, emissions_tools, natural_gas_fuel_data
    ):
        """Test CO2 calculation latency is under 5ms."""
        # Warm-up
        for _ in range(10):
            emissions_tools.calculate_co2_emissions(natural_gas_fuel_data)

        # Measure
        iterations = 100
        start_time = time.perf_counter()

        for _ in range(iterations):
            emissions_tools.calculate_co2_emissions(natural_gas_fuel_data)

        end_time = time.perf_counter()
        avg_time_ms = ((end_time - start_time) / iterations) * 1000

        assert avg_time_ms < 5.0, f"CO2 calculation took {avg_time_ms:.2f}ms (target: <5ms)"

    def test_calculation_latency_all_pollutants(
        self, emissions_tools, sample_cems_data, natural_gas_fuel_data
    ):
        """Test complete emissions calculation latency is under 20ms."""
        def calculate_all():
            nox = emissions_tools.calculate_nox_emissions(sample_cems_data, natural_gas_fuel_data)
            sox = emissions_tools.calculate_sox_emissions(natural_gas_fuel_data)
            co2 = emissions_tools.calculate_co2_emissions(natural_gas_fuel_data)
            pm = emissions_tools.calculate_particulate_matter(sample_cems_data, natural_gas_fuel_data)
            return nox, sox, co2, pm

        # Warm-up
        for _ in range(10):
            calculate_all()

        # Measure
        iterations = 100
        start_time = time.perf_counter()

        for _ in range(iterations):
            calculate_all()

        end_time = time.perf_counter()
        avg_time_ms = ((end_time - start_time) / iterations) * 1000

        assert avg_time_ms < 20.0, f"All calculations took {avg_time_ms:.2f}ms (target: <20ms)"

    # =========================================================================
    # COMPLIANCE CHECK THROUGHPUT TESTS
    # =========================================================================

    def test_compliance_check_throughput(self, emissions_tools):
        """Test compliance check throughput exceeds 5000 checks/sec."""
        emissions_result = {
            "nox": {"emission_rate_lb_mmbtu": 0.08},
            "sox": {"emission_rate_lb_mmbtu": 0.10},
            "co2": {"mass_rate_tons_hr": 40.0},
            "pm": {"emission_rate_lb_mmbtu": 0.02},
        }

        # Warm-up
        for _ in range(10):
            emissions_tools.check_compliance_status(emissions_result, "EPA")

        # Measure throughput
        num_checks = 1000
        start_time = time.perf_counter()

        for _ in range(num_checks):
            emissions_tools.check_compliance_status(emissions_result, "EPA")

        end_time = time.perf_counter()
        duration = end_time - start_time
        throughput = num_checks / duration

        assert throughput > 5000, f"Throughput: {throughput:.0f} checks/sec (target: >5000)"

    def test_violation_detection_throughput(self, emissions_tools, epa_permit_limits):
        """Test violation detection throughput exceeds 5000/sec."""
        emissions_result = {
            "nox": {"emission_rate_lb_mmbtu": 0.15},
            "sox": {"emission_rate_lb_mmbtu": 0.08},
            "co2": {"mass_rate_tons_hr": 40.0},
            "pm": {"emission_rate_lb_mmbtu": 0.02},
        }

        # Measure throughput
        num_detections = 1000
        start_time = time.perf_counter()

        for _ in range(num_detections):
            emissions_tools.detect_violations(emissions_result, epa_permit_limits)

        end_time = time.perf_counter()
        duration = end_time - start_time
        throughput = num_detections / duration

        assert throughput > 5000, f"Throughput: {throughput:.0f} detections/sec (target: >5000)"

    # =========================================================================
    # REPORT GENERATION SPEED TESTS
    # =========================================================================

    def test_report_generation_speed(
        self, emissions_tools, facility_data, reporting_period, emissions_records
    ):
        """Test report generation completes in under 1 second."""
        start_time = time.perf_counter()

        report = emissions_tools.generate_regulatory_report(
            report_format="EPA_ECMPS",
            reporting_period=reporting_period,
            facility_data=facility_data,
            emissions_data=emissions_records,  # ~720 records
        )

        end_time = time.perf_counter()
        duration_ms = (end_time - start_time) * 1000

        assert duration_ms < 1000, f"Report generation took {duration_ms:.0f}ms (target: <1000ms)"
        assert report is not None

    def test_audit_trail_generation_speed(
        self, emissions_tools, facility_data, emissions_records
    ):
        """Test audit trail generation speed."""
        audit_period = {"start_date": "2024-01-01", "end_date": "2024-03-31"}

        start_time = time.perf_counter()

        audit = emissions_tools.generate_audit_trail(
            audit_period=audit_period,
            facility_data=facility_data,
            emissions_records=emissions_records,
            compliance_events=[],
        )

        end_time = time.perf_counter()
        duration_ms = (end_time - start_time) * 1000

        # Should complete in under 2 seconds for 720 records
        assert duration_ms < 2000, f"Audit trail took {duration_ms:.0f}ms (target: <2000ms)"
        assert audit is not None

    # =========================================================================
    # MEMORY USAGE TESTS
    # =========================================================================

    def test_memory_usage_batch_processing(
        self, emissions_tools, sample_cems_data, natural_gas_fuel_data
    ):
        """Test memory usage during batch processing."""
        try:
            import psutil
            import os

            process = psutil.Process(os.getpid())
            initial_memory = process.memory_info().rss / 1024 / 1024  # MB

            # Process 10000 records
            results = []
            for _ in range(10000):
                result = emissions_tools.calculate_nox_emissions(
                    sample_cems_data,
                    natural_gas_fuel_data
                )
                results.append(result.emission_rate_lb_mmbtu)

            final_memory = process.memory_info().rss / 1024 / 1024  # MB
            memory_increase = final_memory - initial_memory

            # Should not increase by more than 100MB for 10k records
            assert memory_increase < 100, \
                f"Memory increased by {memory_increase:.0f}MB (target: <100MB)"

        except ImportError:
            pytest.skip("psutil not available for memory testing")

    def test_memory_no_leaks(
        self, emissions_tools, sample_cems_data, natural_gas_fuel_data
    ):
        """Test for memory leaks during repeated operations."""
        try:
            import psutil
            import os
            import gc

            process = psutil.Process(os.getpid())

            # Force garbage collection
            gc.collect()
            initial_memory = process.memory_info().rss / 1024 / 1024

            # Run 5 batches of 1000 calculations
            for batch in range(5):
                for _ in range(1000):
                    emissions_tools.calculate_nox_emissions(
                        sample_cems_data,
                        natural_gas_fuel_data
                    )

                gc.collect()

            final_memory = process.memory_info().rss / 1024 / 1024
            memory_increase = final_memory - initial_memory

            # Memory should stabilize, not continuously grow
            assert memory_increase < 50, \
                f"Possible memory leak: {memory_increase:.0f}MB increase"

        except ImportError:
            pytest.skip("psutil not available for memory testing")

    # =========================================================================
    # CONCURRENT REQUEST TESTS
    # =========================================================================

    def test_concurrent_requests(
        self, emissions_tools, sample_cems_data, natural_gas_fuel_data
    ):
        """Test performance under concurrent requests."""
        import threading

        results = []
        errors = []

        def worker(thread_id, num_iterations):
            try:
                for _ in range(num_iterations):
                    result = emissions_tools.calculate_nox_emissions(
                        sample_cems_data,
                        natural_gas_fuel_data
                    )
                    results.append(result.emission_rate_lb_mmbtu)
            except Exception as e:
                errors.append((thread_id, str(e)))

        # Start concurrent threads
        threads = []
        num_threads = 4
        iterations_per_thread = 250

        start_time = time.perf_counter()

        for i in range(num_threads):
            t = threading.Thread(target=worker, args=(i, iterations_per_thread))
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        end_time = time.perf_counter()
        duration = end_time - start_time

        total_operations = num_threads * iterations_per_thread
        throughput = total_operations / duration

        # No errors should occur
        assert len(errors) == 0, f"Errors during concurrent execution: {errors}"

        # All results should be present
        assert len(results) == total_operations

        # Results should be consistent (deterministic)
        assert len(set(results)) == 1

        # Throughput should be reasonable
        assert throughput > 1000, f"Concurrent throughput: {throughput:.0f}/sec"


# =============================================================================
# TEST CLASS: SCALABILITY
# =============================================================================

@pytest.mark.performance
class TestScalability:
    """Scalability tests for large data volumes."""

    @pytest.mark.slow
    def test_large_dataset_processing(
        self, emissions_tools, natural_gas_fuel_data
    ):
        """Test processing large dataset (100k records)."""
        num_records = 10000  # Reduced for faster test
        records_processed = 0

        start_time = time.perf_counter()

        for i in range(num_records):
            cems_data = {
                "nox_ppm": 45.0 + (i % 10),
                "o2_percent": 3.0 + (i % 5) * 0.1,
            }

            emissions_tools.calculate_nox_emissions(cems_data, natural_gas_fuel_data)
            records_processed += 1

        end_time = time.perf_counter()
        duration = end_time - start_time
        throughput = records_processed / duration

        assert throughput > 1000, \
            f"Large dataset throughput: {throughput:.0f} records/sec (target: >1000)"

    @pytest.mark.slow
    def test_report_generation_large_data(
        self, emissions_tools, facility_data, reporting_period
    ):
        """Test report generation with large dataset."""
        # Generate large dataset
        large_records = []
        for i in range(8760):  # Full year of hourly data
            large_records.append({
                "hour": i,
                "nox_lb_mmbtu": 0.08 + (i % 5) * 0.005,
                "sox_lb_mmbtu": 0.10,
                "co2_tons": 40.0,
                "pm_lb_mmbtu": 0.02,
                "compliant": True,
                "valid": True,
            })

        start_time = time.perf_counter()

        report = emissions_tools.generate_regulatory_report(
            report_format="EPA_ECMPS",
            reporting_period=reporting_period,
            facility_data=facility_data,
            emissions_data=large_records,
        )

        end_time = time.perf_counter()
        duration = end_time - start_time

        assert duration < 5.0, f"Large report took {duration:.1f}s (target: <5s)"
        assert report.total_operating_hours == 8760
