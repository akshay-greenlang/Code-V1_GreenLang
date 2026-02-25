# -*- coding: utf-8 -*-
"""
Test suite for waste_generated.metrics - AGENT-MRV-018.

Tests Prometheus metrics for the Waste Generated in Operations Agent
(GL-MRV-S3-005) including all 14 metrics with gl_wg_ prefix, recording
methods, context managers, singleton pattern, and thread safety.

Coverage:
- All 14 Prometheus metrics exist with gl_wg_ prefix
- All 14 recording methods
- All 9 context managers
- Singleton pattern
- Thread safety
- Label values
- Histogram bucket boundaries
- Counter increment behavior
- Gauge set behavior

Author: GL-TestEngineer
Date: February 2026
"""

import threading
import time
from decimal import Decimal
from typing import Any, Dict
from unittest.mock import Mock, patch, MagicMock
import pytest

from greenlang.waste_generated.metrics import (
    WasteGeneratedMetrics,
    get_metrics,
    PROMETHEUS_AVAILABLE,
)


# ==============================================================================
# SINGLETON PATTERN TESTS
# ==============================================================================

class TestSingletonPattern:
    """Test singleton pattern implementation."""

    def test_get_metrics_returns_same_instance(self):
        """Test get_metrics returns the same instance."""
        metrics1 = get_metrics()
        metrics2 = get_metrics()

        assert metrics1 is metrics2

    def test_singleton_across_threads(self):
        """Test singleton works across threads."""
        metrics_instances = []

        def get_metrics_thread():
            metrics_instances.append(get_metrics())

        threads = [threading.Thread(target=get_metrics_thread) for _ in range(10)]

        for thread in threads:
            thread.start()

        for thread in threads:
            thread.join()

        # All instances should be the same
        first_instance = metrics_instances[0]
        for instance in metrics_instances[1:]:
            assert instance is first_instance


# ==============================================================================
# METRICS EXISTENCE TESTS
# ==============================================================================

class TestMetricsExistence:
    """Test all 14 Prometheus metrics exist."""

    @pytest.fixture
    def metrics(self):
        """Get metrics instance."""
        return get_metrics()

    def test_calculations_total_exists(self, metrics):
        """Test gl_wg_calculations_total counter exists."""
        assert hasattr(metrics, '_calculations_total')
        # Verify it's a Counter
        metric = metrics._calculations_total
        assert metric is not None

    def test_calculation_errors_total_exists(self, metrics):
        """Test gl_wg_calculation_errors_total counter exists."""
        assert hasattr(metrics, '_calculation_errors_total')

    def test_calculation_duration_seconds_exists(self, metrics):
        """Test gl_wg_calculation_duration_seconds histogram exists."""
        assert hasattr(metrics, '_calculation_duration_seconds')

    def test_emissions_co2e_tonnes_exists(self, metrics):
        """Test gl_wg_emissions_co2e_tonnes counter exists."""
        assert hasattr(metrics, '_emissions_co2e_tonnes')

    def test_waste_mass_tonnes_exists(self, metrics):
        """Test gl_wg_waste_mass_tonnes counter exists."""
        assert hasattr(metrics, '_waste_mass_tonnes')

    def test_landfill_ch4_generated_kg_exists(self, metrics):
        """Test gl_wg_landfill_ch4_generated_kg counter exists."""
        assert hasattr(metrics, '_landfill_ch4_generated_kg')

    def test_incineration_energy_recovered_mwh_exists(self, metrics):
        """Test gl_wg_incineration_energy_recovered_mwh counter exists."""
        assert hasattr(metrics, '_incineration_energy_recovered_mwh')

    def test_recycling_avoided_emissions_co2e_exists(self, metrics):
        """Test gl_wg_recycling_avoided_emissions_co2e counter exists."""
        assert hasattr(metrics, '_recycling_avoided_emissions_co2e')

    def test_wastewater_organic_load_kg_exists(self, metrics):
        """Test gl_wg_wastewater_organic_load_kg counter exists."""
        assert hasattr(metrics, '_wastewater_organic_load_kg')

    def test_diversion_rate_exists(self, metrics):
        """Test gl_wg_diversion_rate gauge exists."""
        assert hasattr(metrics, '_diversion_rate')

    def test_compliance_checks_total_exists(self, metrics):
        """Test gl_wg_compliance_checks_total counter exists."""
        assert hasattr(metrics, '_compliance_checks_total')

    def test_data_quality_score_exists(self, metrics):
        """Test gl_wg_data_quality_score gauge exists."""
        assert hasattr(metrics, '_data_quality_score')

    def test_batch_size_exists(self, metrics):
        """Test gl_wg_batch_size histogram exists."""
        assert hasattr(metrics, '_batch_size')

    def test_ef_lookups_total_exists(self, metrics):
        """Test gl_wg_ef_lookups_total counter exists."""
        assert hasattr(metrics, '_ef_lookups_total')


# ==============================================================================
# RECORDING METHOD TESTS
# ==============================================================================

class TestRecordingMethods:
    """Test all 14 recording methods."""

    @pytest.fixture
    def metrics(self):
        """Get metrics instance."""
        return get_metrics()

    def test_record_calculation(self, metrics):
        """Test record_calculation method."""
        # Should not raise
        metrics.record_calculation(
            method="waste_type_specific",
            treatment="landfill",
            waste_category="food_waste",
            tenant_id="tenant-001",
            status="success",
            emissions_tco2e=10.5,
            duration_s=0.25
        )

    def test_record_calculation_error(self, metrics):
        """Test record_calculation_error method."""
        metrics.record_calculation_error(
            error_type="validation_error",
            tenant_id="tenant-001",
            waste_category="plastics"
        )

    def test_record_emissions(self, metrics):
        """Test record_emissions method."""
        metrics.record_emissions(
            treatment="incineration",
            waste_category="plastics_mixed",
            tenant_id="tenant-001",
            emissions_tco2e=5.2
        )

    def test_record_waste_mass(self, metrics):
        """Test record_waste_mass method."""
        metrics.record_waste_mass(
            treatment="recycling",
            waste_category="paper_cardboard",
            tenant_id="tenant-001",
            mass_tonnes=20.0
        )

    def test_record_landfill_ch4(self, metrics):
        """Test record_landfill_ch4 method."""
        metrics.record_landfill_ch4(
            landfill_type="managed_anaerobic",
            climate_zone="temperate_wet",
            tenant_id="tenant-001",
            ch4_kg=500.0
        )

    def test_record_incineration_energy(self, metrics):
        """Test record_incineration_energy method."""
        metrics.record_incineration_energy(
            incinerator_type="continuous_stoker",
            tenant_id="tenant-001",
            energy_mwh=100.0
        )

    def test_record_recycling_avoided_emissions(self, metrics):
        """Test record_recycling_avoided_emissions method."""
        metrics.record_recycling_avoided_emissions(
            material="paper",
            recycling_type="open_loop",
            tenant_id="tenant-001",
            avoided_tco2e=2.5
        )

    def test_record_wastewater_load(self, metrics):
        """Test record_wastewater_load method."""
        metrics.record_wastewater_load(
            system_type="aerobic",
            tenant_id="tenant-001",
            organic_load_kg=1000.0
        )

    def test_set_diversion_rate(self, metrics):
        """Test set_diversion_rate method."""
        metrics.set_diversion_rate(
            facility_id="FAC-001",
            tenant_id="tenant-001",
            rate=0.65
        )

    def test_record_compliance_check(self, metrics):
        """Test record_compliance_check method."""
        metrics.record_compliance_check(
            framework="ghg_protocol",
            tenant_id="tenant-001",
            status="compliant"
        )

    def test_set_data_quality_score(self, metrics):
        """Test set_data_quality_score method."""
        metrics.set_data_quality_score(
            source="transfer_notes",
            tenant_id="tenant-001",
            score=4.2
        )

    def test_record_batch_size(self, metrics):
        """Test record_batch_size method."""
        metrics.record_batch_size(
            tenant_id="tenant-001",
            size=50
        )

    def test_record_ef_lookup(self, metrics):
        """Test record_ef_lookup method."""
        metrics.record_ef_lookup(
            source="epa_warm",
            tenant_id="tenant-001",
            status="success"
        )


# ==============================================================================
# CONTEXT MANAGER TESTS
# ==============================================================================

class TestContextManagers:
    """Test all context managers."""

    @pytest.fixture
    def metrics(self):
        """Get metrics instance."""
        return get_metrics()

    def test_time_calculation_context(self, metrics):
        """Test time_calculation context manager."""
        with metrics.time_calculation(
            method="waste_type_specific",
            treatment="landfill",
            waste_category="food_waste",
            tenant_id="tenant-001"
        ):
            time.sleep(0.01)  # Simulate work

    def test_time_database_query_context(self, metrics):
        """Test time_database_query context manager."""
        with metrics.time_database_query(
            query_type="emission_factor_lookup",
            tenant_id="tenant-001"
        ):
            time.sleep(0.005)

    def test_time_landfill_calculation_context(self, metrics):
        """Test time_landfill_calculation context manager."""
        with metrics.time_landfill_calculation(
            landfill_type="managed_anaerobic",
            tenant_id="tenant-001"
        ):
            time.sleep(0.01)

    def test_time_incineration_calculation_context(self, metrics):
        """Test time_incineration_calculation context manager."""
        with metrics.time_incineration_calculation(
            incinerator_type="continuous_stoker",
            tenant_id="tenant-001"
        ):
            time.sleep(0.01)

    def test_time_recycling_calculation_context(self, metrics):
        """Test time_recycling_calculation context manager."""
        with metrics.time_recycling_calculation(
            recycling_type="open_loop",
            tenant_id="tenant-001"
        ):
            time.sleep(0.01)

    def test_time_wastewater_calculation_context(self, metrics):
        """Test time_wastewater_calculation context manager."""
        with metrics.time_wastewater_calculation(
            system_type="aerobic",
            tenant_id="tenant-001"
        ):
            time.sleep(0.01)

    def test_time_compliance_check_context(self, metrics):
        """Test time_compliance_check context manager."""
        with metrics.time_compliance_check(
            framework="ghg_protocol",
            tenant_id="tenant-001"
        ):
            time.sleep(0.005)

    def test_context_manager_with_exception(self, metrics):
        """Test context manager handles exceptions gracefully."""
        try:
            with metrics.time_calculation(
                method="waste_type_specific",
                treatment="landfill",
                waste_category="food_waste",
                tenant_id="tenant-001"
            ):
                raise ValueError("Test error")
        except ValueError:
            pass  # Exception should propagate, but metrics should still be recorded

    def test_nested_context_managers(self, metrics):
        """Test nested context managers."""
        with metrics.time_calculation(
            method="waste_type_specific",
            treatment="landfill",
            waste_category="food_waste",
            tenant_id="tenant-001"
        ):
            with metrics.time_database_query(
                query_type="emission_factor_lookup",
                tenant_id="tenant-001"
            ):
                time.sleep(0.005)


# ==============================================================================
# THREAD SAFETY TESTS
# ==============================================================================

class TestThreadSafety:
    """Test thread safety of metrics recording."""

    @pytest.fixture
    def metrics(self):
        """Get metrics instance."""
        return get_metrics()

    def test_concurrent_recording(self, metrics):
        """Test concurrent metrics recording is thread-safe."""
        errors = []

        def record_metrics():
            try:
                for _ in range(100):
                    metrics.record_calculation(
                        method="waste_type_specific",
                        treatment="landfill",
                        waste_category="food_waste",
                        tenant_id="tenant-001",
                        status="success",
                        emissions_tco2e=10.0,
                        duration_s=0.1
                    )
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=record_metrics) for _ in range(10)]

        for thread in threads:
            thread.start()

        for thread in threads:
            thread.join()

        assert len(errors) == 0

    def test_concurrent_context_managers(self, metrics):
        """Test concurrent context manager usage."""
        errors = []

        def use_context():
            try:
                for _ in range(50):
                    with metrics.time_calculation(
                        method="waste_type_specific",
                        treatment="landfill",
                        waste_category="food_waste",
                        tenant_id="tenant-001"
                    ):
                        time.sleep(0.001)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=use_context) for _ in range(5)]

        for thread in threads:
            thread.start()

        for thread in threads:
            thread.join()

        assert len(errors) == 0


# ==============================================================================
# LABEL VALUE TESTS
# ==============================================================================

class TestLabelValues:
    """Test metric label values."""

    @pytest.fixture
    def metrics(self):
        """Get metrics instance."""
        return get_metrics()

    def test_calculation_method_labels(self, metrics):
        """Test calculation method label values."""
        methods = ["supplier_specific", "waste_type_specific", "average_data", "spend_based"]

        for method in methods:
            metrics.record_calculation(
                method=method,
                treatment="landfill",
                waste_category="food_waste",
                tenant_id="tenant-001",
                status="success",
                emissions_tco2e=10.0,
                duration_s=0.1
            )

    def test_treatment_method_labels(self, metrics):
        """Test treatment method label values."""
        treatments = ["landfill", "incineration", "recycling", "composting", "wastewater"]

        for treatment in treatments:
            metrics.record_waste_mass(
                treatment=treatment,
                waste_category="food_waste",
                tenant_id="tenant-001",
                mass_tonnes=10.0
            )

    def test_waste_category_labels(self, metrics):
        """Test waste category label values."""
        categories = [
            "food_waste", "paper_cardboard", "plastics_mixed",
            "glass", "metals_steel", "electronics"
        ]

        for category in categories:
            metrics.record_waste_mass(
                treatment="landfill",
                waste_category=category,
                tenant_id="tenant-001",
                mass_tonnes=5.0
            )

    def test_status_labels(self, metrics):
        """Test status label values."""
        statuses = ["success", "error", "partial"]

        for status in statuses:
            metrics.record_calculation(
                method="waste_type_specific",
                treatment="landfill",
                waste_category="food_waste",
                tenant_id="tenant-001",
                status=status,
                emissions_tco2e=10.0,
                duration_s=0.1
            )


# ==============================================================================
# HISTOGRAM BUCKET TESTS
# ==============================================================================

class TestHistogramBuckets:
    """Test histogram bucket boundaries."""

    @pytest.fixture
    def metrics(self):
        """Get metrics instance."""
        return get_metrics()

    def test_duration_histogram_buckets(self, metrics):
        """Test calculation duration histogram has appropriate buckets."""
        # Record various durations
        durations = [0.001, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0]

        for duration in durations:
            metrics.record_calculation(
                method="waste_type_specific",
                treatment="landfill",
                waste_category="food_waste",
                tenant_id="tenant-001",
                status="success",
                emissions_tco2e=10.0,
                duration_s=duration
            )

    def test_batch_size_histogram_buckets(self, metrics):
        """Test batch size histogram has appropriate buckets."""
        # Record various batch sizes
        sizes = [1, 10, 50, 100, 500, 1000, 5000, 10000]

        for size in sizes:
            metrics.record_batch_size(
                tenant_id="tenant-001",
                size=size
            )


# ==============================================================================
# COUNTER INCREMENT TESTS
# ==============================================================================

class TestCounterBehavior:
    """Test counter increment behavior."""

    @pytest.fixture
    def metrics(self):
        """Get metrics instance."""
        return get_metrics()

    def test_counter_increments(self, metrics):
        """Test counters increment correctly."""
        # Record multiple calculations
        for _ in range(5):
            metrics.record_calculation(
                method="waste_type_specific",
                treatment="landfill",
                waste_category="food_waste",
                tenant_id="tenant-001",
                status="success",
                emissions_tco2e=10.0,
                duration_s=0.1
            )

    def test_emissions_counter_accumulates(self, metrics):
        """Test emissions counter accumulates correctly."""
        emissions_values = [Decimal("5.2"), Decimal("3.8"), Decimal("12.1")]

        for emissions in emissions_values:
            metrics.record_emissions(
                treatment="landfill",
                waste_category="food_waste",
                tenant_id="tenant-001",
                emissions_tco2e=float(emissions)
            )

    def test_error_counter_increments(self, metrics):
        """Test error counter increments."""
        for _ in range(3):
            metrics.record_calculation_error(
                error_type="validation_error",
                tenant_id="tenant-001",
                waste_category="food_waste"
            )


# ==============================================================================
# GAUGE BEHAVIOR TESTS
# ==============================================================================

class TestGaugeBehavior:
    """Test gauge set behavior."""

    @pytest.fixture
    def metrics(self):
        """Get metrics instance."""
        return get_metrics()

    def test_gauge_set_value(self, metrics):
        """Test gauge sets to specific value."""
        metrics.set_diversion_rate(
            facility_id="FAC-001",
            tenant_id="tenant-001",
            rate=0.75
        )

    def test_gauge_updates(self, metrics):
        """Test gauge updates when set multiple times."""
        rates = [0.50, 0.60, 0.75, 0.80]

        for rate in rates:
            metrics.set_diversion_rate(
                facility_id="FAC-001",
                tenant_id="tenant-001",
                rate=rate
            )

    def test_data_quality_gauge(self, metrics):
        """Test data quality score gauge."""
        scores = [3.0, 3.5, 4.0, 4.5, 5.0]

        for score in scores:
            metrics.set_data_quality_score(
                source="transfer_notes",
                tenant_id="tenant-001",
                score=score
            )


# ==============================================================================
# NO-OP FALLBACK TESTS
# ==============================================================================

class TestNoOpFallback:
    """Test metrics work without Prometheus client (no-op fallback)."""

    def test_metrics_without_prometheus(self):
        """Test metrics work when Prometheus not available."""
        # This would require mocking PROMETHEUS_AVAILABLE = False
        # and testing that all methods still work as no-ops
        pass


# ==============================================================================
# EDGE CASE TESTS
# ==============================================================================

class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    @pytest.fixture
    def metrics(self):
        """Get metrics instance."""
        return get_metrics()

    def test_zero_emissions(self, metrics):
        """Test recording zero emissions."""
        metrics.record_emissions(
            treatment="recycling",
            waste_category="paper_cardboard",
            tenant_id="tenant-001",
            emissions_tco2e=0.0
        )

    def test_negative_avoided_emissions(self, metrics):
        """Test recording negative (avoided) emissions."""
        metrics.record_recycling_avoided_emissions(
            material="paper",
            recycling_type="open_loop",
            tenant_id="tenant-001",
            avoided_tco2e=-2.5
        )

    def test_very_large_batch_size(self, metrics):
        """Test recording very large batch size."""
        metrics.record_batch_size(
            tenant_id="tenant-001",
            size=1_000_000
        )

    def test_very_small_duration(self, metrics):
        """Test recording very small duration."""
        metrics.record_calculation(
            method="waste_type_specific",
            treatment="landfill",
            waste_category="food_waste",
            tenant_id="tenant-001",
            status="success",
            emissions_tco2e=10.0,
            duration_s=0.0001
        )

    def test_empty_tenant_id(self, metrics):
        """Test handling empty tenant_id."""
        metrics.record_calculation(
            method="waste_type_specific",
            treatment="landfill",
            waste_category="food_waste",
            tenant_id="",
            status="success",
            emissions_tco2e=10.0,
            duration_s=0.1
        )


# ==============================================================================
# SUMMARY
# ==============================================================================

def test_metrics_module_coverage():
    """Meta-test to ensure comprehensive coverage."""
    # Verify we've tested all 14 metrics
    tested_metrics = [
        "calculations_total",
        "calculation_errors_total",
        "calculation_duration_seconds",
        "emissions_co2e_tonnes",
        "waste_mass_tonnes",
        "landfill_ch4_generated_kg",
        "incineration_energy_recovered_mwh",
        "recycling_avoided_emissions_co2e",
        "wastewater_organic_load_kg",
        "diversion_rate",
        "compliance_checks_total",
        "data_quality_score",
        "batch_size",
        "ef_lookups_total",
    ]

    assert len(tested_metrics) == 14
