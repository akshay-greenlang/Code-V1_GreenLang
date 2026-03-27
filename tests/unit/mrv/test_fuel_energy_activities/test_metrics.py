# -*- coding: utf-8 -*-
"""
Unit tests for AGENT-MRV-016 Fuel & Energy Activities Agent metrics.

Tests metrics collection, Prometheus integration, performance tracking,
and graceful fallback behavior.
"""

import pytest
from decimal import Decimal
from unittest.mock import Mock, patch, MagicMock
import time

from greenlang.agents.mrv.fuel_energy_activities.metrics import (
    FuelEnergyActivitiesMetrics,
    MetricsCollector,
)
from greenlang.agents.mrv.fuel_energy_activities.models import (
    FuelType,
    ActivityType,
    CalculationMethod,
    RegulatoryFramework,
)


# ============================================================================
# METRICS SINGLETON TESTS
# ============================================================================

class TestMetricsSingleton:
    """Test FuelEnergyActivitiesMetrics singleton pattern."""

    def test_singleton(self):
        """Test FuelEnergyActivitiesMetrics implements singleton pattern."""
        metrics1 = FuelEnergyActivitiesMetrics.get_instance()
        metrics2 = FuelEnergyActivitiesMetrics.get_instance()

        assert metrics1 is metrics2

    def test_thread_safety(self):
        """Test FuelEnergyActivitiesMetrics singleton is thread-safe."""
        import threading

        metrics_instances = []

        def get_metrics():
            metrics = FuelEnergyActivitiesMetrics.get_instance()
            metrics_instances.append(metrics)

        threads = [threading.Thread(target=get_metrics) for _ in range(10)]

        for thread in threads:
            thread.start()

        for thread in threads:
            thread.join()

        # All instances should be the same
        assert all(m is metrics_instances[0] for m in metrics_instances)


# ============================================================================
# CALCULATION METRICS TESTS
# ============================================================================

class TestCalculationMetrics:
    """Test calculation-related metrics."""

    def test_record_calculation(self):
        """Test record_calculation() increments counter."""
        metrics = FuelEnergyActivitiesMetrics.get_instance()
        metrics.reset()

        metrics.record_calculation(
            activity_type=ActivityType.ACTIVITY_3A,
            calculation_method=CalculationMethod.FUEL_BASED,
            success=True,
            duration_ms=125.5
        )

        summary = metrics.get_metrics_summary()
        assert summary["total_calculations"] == 1
        assert summary["successful_calculations"] == 1
        assert summary["failed_calculations"] == 0

    def test_record_calculation_failure(self):
        """Test record_calculation() tracks failures."""
        metrics = FuelEnergyActivitiesMetrics.get_instance()
        metrics.reset()

        metrics.record_calculation(
            activity_type=ActivityType.ACTIVITY_3A,
            calculation_method=CalculationMethod.FUEL_BASED,
            success=False,
            duration_ms=50.0
        )

        summary = metrics.get_metrics_summary()
        assert summary["total_calculations"] == 1
        assert summary["successful_calculations"] == 0
        assert summary["failed_calculations"] == 1

    def test_record_calculation_multiple_activities(self):
        """Test record_calculation() tracks different activity types."""
        metrics = FuelEnergyActivitiesMetrics.get_instance()
        metrics.reset()

        metrics.record_calculation(ActivityType.ACTIVITY_3A, CalculationMethod.FUEL_BASED, True, 100.0)
        metrics.record_calculation(ActivityType.ACTIVITY_3B, CalculationMethod.LOCATION_BASED, True, 150.0)
        metrics.record_calculation(ActivityType.ACTIVITY_3C, CalculationMethod.LOCATION_BASED, True, 75.0)

        summary = metrics.get_metrics_summary()
        assert summary["total_calculations"] == 3
        assert summary["calculations_by_activity"]["ACTIVITY_3A"] == 1
        assert summary["calculations_by_activity"]["ACTIVITY_3B"] == 1
        assert summary["calculations_by_activity"]["ACTIVITY_3C"] == 1


# ============================================================================
# FUEL CONSUMPTION METRICS TESTS
# ============================================================================

class TestFuelConsumptionMetrics:
    """Test fuel consumption metrics."""

    def test_record_fuel_consumption(self):
        """Test record_fuel_consumption() tracks fuel usage."""
        metrics = FuelEnergyActivitiesMetrics.get_instance()
        metrics.reset()

        metrics.record_fuel_consumption(
            fuel_type=FuelType.NATURAL_GAS,
            quantity=Decimal("10000.0"),
            unit="m3",
            emissions_tco2e=Decimal("20.5")
        )

        summary = metrics.get_metrics_summary()
        assert summary["total_fuel_records"] == 1
        assert summary["fuel_consumption_by_type"]["NATURAL_GAS"]["quantity"] == Decimal("10000.0")
        assert summary["fuel_consumption_by_type"]["NATURAL_GAS"]["emissions_tco2e"] == Decimal("20.5")

    def test_record_fuel_consumption_multiple_fuels(self):
        """Test record_fuel_consumption() tracks multiple fuel types."""
        metrics = FuelEnergyActivitiesMetrics.get_instance()
        metrics.reset()

        metrics.record_fuel_consumption(FuelType.NATURAL_GAS, Decimal("10000.0"), "m3", Decimal("20.5"))
        metrics.record_fuel_consumption(FuelType.DIESEL, Decimal("5000.0"), "L", Decimal("12.8"))
        metrics.record_fuel_consumption(FuelType.COAL, Decimal("50.0"), "tonnes", Decimal("135.0"))

        summary = metrics.get_metrics_summary()
        assert summary["total_fuel_records"] == 3
        assert len(summary["fuel_consumption_by_type"]) == 3

    def test_record_fuel_consumption_aggregation(self):
        """Test record_fuel_consumption() aggregates same fuel type."""
        metrics = FuelEnergyActivitiesMetrics.get_instance()
        metrics.reset()

        metrics.record_fuel_consumption(FuelType.NATURAL_GAS, Decimal("10000.0"), "m3", Decimal("20.5"))
        metrics.record_fuel_consumption(FuelType.NATURAL_GAS, Decimal("5000.0"), "m3", Decimal("10.25"))

        summary = metrics.get_metrics_summary()
        assert summary["total_fuel_records"] == 2
        assert summary["fuel_consumption_by_type"]["NATURAL_GAS"]["quantity"] == Decimal("15000.0")
        assert summary["fuel_consumption_by_type"]["NATURAL_GAS"]["emissions_tco2e"] == Decimal("30.75")


# ============================================================================
# ELECTRICITY CONSUMPTION METRICS TESTS
# ============================================================================

class TestElectricityConsumptionMetrics:
    """Test electricity consumption metrics."""

    def test_record_electricity_consumption(self):
        """Test record_electricity_consumption() tracks electricity usage."""
        metrics = FuelEnergyActivitiesMetrics.get_instance()
        metrics.reset()

        metrics.record_electricity_consumption(
            quantity=Decimal("100000.0"),
            unit="kWh",
            country="US",
            emissions_tco2e=Decimal("8.2")
        )

        summary = metrics.get_metrics_summary()
        assert summary["total_electricity_records"] == 1
        assert summary["electricity_consumption_by_country"]["US"]["quantity"] == Decimal("100000.0")
        assert summary["electricity_consumption_by_country"]["US"]["emissions_tco2e"] == Decimal("8.2")

    def test_record_electricity_consumption_multiple_countries(self):
        """Test record_electricity_consumption() tracks multiple countries."""
        metrics = FuelEnergyActivitiesMetrics.get_instance()
        metrics.reset()

        metrics.record_electricity_consumption(Decimal("100000.0"), "kWh", "US", Decimal("8.2"))
        metrics.record_electricity_consumption(Decimal("50000.0"), "kWh", "GB", Decimal("3.5"))
        metrics.record_electricity_consumption(Decimal("75000.0"), "MWh", "DE", Decimal("45.0"))

        summary = metrics.get_metrics_summary()
        assert summary["total_electricity_records"] == 3
        assert len(summary["electricity_consumption_by_country"]) == 3


# ============================================================================
# T&D LOSS METRICS TESTS
# ============================================================================

class TestTDLossMetrics:
    """Test T&D loss metrics."""

    def test_record_td_loss(self):
        """Test record_td_loss() tracks transmission losses."""
        metrics = FuelEnergyActivitiesMetrics.get_instance()
        metrics.reset()

        metrics.record_td_loss(
            country="US",
            loss_percentage=Decimal("5.0"),
            loss_quantity_kwh=Decimal("5000.0"),
            emissions_tco2e=Decimal("2.5")
        )

        summary = metrics.get_metrics_summary()
        assert summary["total_td_loss_records"] == 1
        assert summary["td_losses_by_country"]["US"]["loss_percentage"] == Decimal("5.0")
        assert summary["td_losses_by_country"]["US"]["emissions_tco2e"] == Decimal("2.5")

    def test_record_td_loss_multiple_countries(self):
        """Test record_td_loss() tracks multiple countries."""
        metrics = FuelEnergyActivitiesMetrics.get_instance()
        metrics.reset()

        metrics.record_td_loss("US", Decimal("5.0"), Decimal("5000.0"), Decimal("2.5"))
        metrics.record_td_loss("GB", Decimal("7.0"), Decimal("3500.0"), Decimal("1.8"))
        metrics.record_td_loss("DE", Decimal("4.0"), Decimal("3000.0"), Decimal("1.5"))

        summary = metrics.get_metrics_summary()
        assert summary["total_td_loss_records"] == 3
        assert len(summary["td_losses_by_country"]) == 3


# ============================================================================
# EMISSION FACTOR LOOKUP METRICS TESTS
# ============================================================================

class TestEmissionFactorLookupMetrics:
    """Test emission factor lookup metrics."""

    def test_record_wtt_lookup(self):
        """Test record_wtt_lookup() tracks WTT factor lookups."""
        metrics = FuelEnergyActivitiesMetrics.get_instance()
        metrics.reset()

        metrics.record_wtt_lookup(
            fuel_type=FuelType.NATURAL_GAS,
            country="US",
            success=True
        )

        summary = metrics.get_metrics_summary()
        assert summary["wtt_lookups"] == 1
        assert summary["successful_wtt_lookups"] == 1
        assert summary["failed_wtt_lookups"] == 0

    def test_record_wtt_lookup_failure(self):
        """Test record_wtt_lookup() tracks failures."""
        metrics = FuelEnergyActivitiesMetrics.get_instance()
        metrics.reset()

        metrics.record_wtt_lookup(FuelType.NATURAL_GAS, "US", success=False)

        summary = metrics.get_metrics_summary()
        assert summary["wtt_lookups"] == 1
        assert summary["successful_wtt_lookups"] == 0
        assert summary["failed_wtt_lookups"] == 1

    def test_record_upstream_lookup(self):
        """Test record_upstream_lookup() tracks upstream electricity lookups."""
        metrics = FuelEnergyActivitiesMetrics.get_instance()
        metrics.reset()

        metrics.record_upstream_lookup(
            country="US",
            egrid_subregion="NEWE",
            success=True
        )

        summary = metrics.get_metrics_summary()
        assert summary["upstream_lookups"] == 1
        assert summary["successful_upstream_lookups"] == 1


# ============================================================================
# COMPLIANCE CHECK METRICS TESTS
# ============================================================================

class TestComplianceCheckMetrics:
    """Test compliance check metrics."""

    def test_record_compliance_check(self):
        """Test record_compliance_check() tracks compliance checks."""
        metrics = FuelEnergyActivitiesMetrics.get_instance()
        metrics.reset()

        metrics.record_compliance_check(
            framework=RegulatoryFramework.GHG_PROTOCOL,
            compliant=True,
            compliance_score=Decimal("0.95")
        )

        summary = metrics.get_metrics_summary()
        assert summary["compliance_checks"] == 1
        assert summary["compliant_checks"] == 1
        assert summary["non_compliant_checks"] == 0

    def test_record_compliance_check_failure(self):
        """Test record_compliance_check() tracks non-compliance."""
        metrics = FuelEnergyActivitiesMetrics.get_instance()
        metrics.reset()

        metrics.record_compliance_check(
            framework=RegulatoryFramework.ISO_14064,
            compliant=False,
            compliance_score=Decimal("0.65")
        )

        summary = metrics.get_metrics_summary()
        assert summary["compliance_checks"] == 1
        assert summary["compliant_checks"] == 0
        assert summary["non_compliant_checks"] == 1

    def test_record_compliance_check_by_framework(self):
        """Test record_compliance_check() tracks by framework."""
        metrics = FuelEnergyActivitiesMetrics.get_instance()
        metrics.reset()

        metrics.record_compliance_check(RegulatoryFramework.GHG_PROTOCOL, True, Decimal("0.95"))
        metrics.record_compliance_check(RegulatoryFramework.ISO_14064, True, Decimal("0.90"))
        metrics.record_compliance_check(RegulatoryFramework.CSRD, False, Decimal("0.70"))

        summary = metrics.get_metrics_summary()
        assert summary["compliance_checks"] == 3
        assert summary["checks_by_framework"]["GHG_PROTOCOL"]["compliant"] == 1
        assert summary["checks_by_framework"]["ISO_14064"]["compliant"] == 1
        assert summary["checks_by_framework"]["CSRD"]["non_compliant"] == 1


# ============================================================================
# BATCH JOB METRICS TESTS
# ============================================================================

class TestBatchJobMetrics:
    """Test batch job metrics."""

    def test_record_batch_job(self):
        """Test record_batch_job() tracks batch processing."""
        metrics = FuelEnergyActivitiesMetrics.get_instance()
        metrics.reset()

        metrics.record_batch_job(
            batch_id="BATCH-001",
            total_records=100,
            success=True,
            duration_ms=5000.0
        )

        summary = metrics.get_metrics_summary()
        assert summary["batch_jobs"] == 1
        assert summary["successful_batch_jobs"] == 1
        assert summary["total_batch_records"] == 100

    def test_record_batch_job_failure(self):
        """Test record_batch_job() tracks failures."""
        metrics = FuelEnergyActivitiesMetrics.get_instance()
        metrics.reset()

        metrics.record_batch_job("BATCH-002", 50, success=False, duration_ms=2000.0)

        summary = metrics.get_metrics_summary()
        assert summary["batch_jobs"] == 1
        assert summary["successful_batch_jobs"] == 0
        assert summary["failed_batch_jobs"] == 1


# ============================================================================
# ERROR METRICS TESTS
# ============================================================================

class TestErrorMetrics:
    """Test error tracking metrics."""

    def test_record_error(self):
        """Test record_error() tracks errors."""
        metrics = FuelEnergyActivitiesMetrics.get_instance()
        metrics.reset()

        metrics.record_error(
            error_type="ValidationError",
            error_message="Invalid fuel quantity"
        )

        summary = metrics.get_metrics_summary()
        assert summary["total_errors"] == 1
        assert summary["errors_by_type"]["ValidationError"] == 1

    def test_record_error_multiple_types(self):
        """Test record_error() tracks multiple error types."""
        metrics = FuelEnergyActivitiesMetrics.get_instance()
        metrics.reset()

        metrics.record_error("ValidationError", "Invalid fuel quantity")
        metrics.record_error("CalculationError", "Division by zero")
        metrics.record_error("ValidationError", "Missing required field")

        summary = metrics.get_metrics_summary()
        assert summary["total_errors"] == 3
        assert summary["errors_by_type"]["ValidationError"] == 2
        assert summary["errors_by_type"]["CalculationError"] == 1


# ============================================================================
# METRICS SUMMARY TESTS
# ============================================================================

class TestMetricsSummary:
    """Test metrics summary generation."""

    def test_get_metrics_summary(self):
        """Test get_metrics_summary() returns complete summary."""
        metrics = FuelEnergyActivitiesMetrics.get_instance()
        metrics.reset()

        # Record various metrics
        metrics.record_calculation(ActivityType.ACTIVITY_3A, CalculationMethod.FUEL_BASED, True, 100.0)
        metrics.record_fuel_consumption(FuelType.NATURAL_GAS, Decimal("10000.0"), "m3", Decimal("20.5"))
        metrics.record_electricity_consumption(Decimal("100000.0"), "kWh", "US", Decimal("8.2"))

        summary = metrics.get_metrics_summary()

        assert "total_calculations" in summary
        assert "total_fuel_records" in summary
        assert "total_electricity_records" in summary
        assert "total_errors" in summary

    def test_reset(self):
        """Test reset() clears all metrics."""
        metrics = FuelEnergyActivitiesMetrics.get_instance()

        # Record some metrics
        metrics.record_calculation(ActivityType.ACTIVITY_3A, CalculationMethod.FUEL_BASED, True, 100.0)
        metrics.record_fuel_consumption(FuelType.NATURAL_GAS, Decimal("10000.0"), "m3", Decimal("20.5"))

        # Reset
        metrics.reset()

        summary = metrics.get_metrics_summary()
        assert summary["total_calculations"] == 0
        assert summary["total_fuel_records"] == 0


# ============================================================================
# CONTEXT MANAGER TESTS
# ============================================================================

class TestContextManagers:
    """Test context manager utilities."""

    def test_calculation_timer_context(self):
        """Test calculation_timer() context manager."""
        metrics = FuelEnergyActivitiesMetrics.get_instance()
        metrics.reset()

        with metrics.calculation_timer(ActivityType.ACTIVITY_3A, CalculationMethod.FUEL_BASED):
            time.sleep(0.1)  # Simulate calculation

        summary = metrics.get_metrics_summary()
        assert summary["total_calculations"] == 1
        assert summary["successful_calculations"] == 1

    def test_calculation_timer_exception_handling(self):
        """Test calculation_timer() handles exceptions."""
        metrics = FuelEnergyActivitiesMetrics.get_instance()
        metrics.reset()

        with pytest.raises(ValueError):
            with metrics.calculation_timer(ActivityType.ACTIVITY_3A, CalculationMethod.FUEL_BASED):
                raise ValueError("Test error")

        summary = metrics.get_metrics_summary()
        assert summary["total_calculations"] == 1
        assert summary["failed_calculations"] == 1


# ============================================================================
# GRACEFUL FALLBACK TESTS
# ============================================================================

class TestGracefulFallback:
    """Test metrics graceful fallback on errors."""

    def test_graceful_fallback(self):
        """Test metrics don't crash application on errors."""
        metrics = FuelEnergyActivitiesMetrics.get_instance()

        # Should not raise exception even with invalid input
        try:
            metrics.record_calculation(None, None, True, -1.0)
            metrics.record_fuel_consumption(None, None, None, None)
        except Exception as e:
            pytest.fail(f"Metrics should not raise exceptions: {e}")
