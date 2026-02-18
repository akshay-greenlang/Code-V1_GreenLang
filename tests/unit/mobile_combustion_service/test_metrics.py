# -*- coding: utf-8 -*-
"""
Unit tests for Mobile Combustion Prometheus Metrics - AGENT-MRV-003

Tests all 12 Prometheus metric definitions, 12 helper functions,
label correctness, and graceful fallback when prometheus_client is
not installed.

Target: 44+ tests
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from greenlang.mobile_combustion import metrics as metrics_mod
from greenlang.mobile_combustion.metrics import (
    PROMETHEUS_AVAILABLE,
    mc_calculations_total,
    mc_emissions_kg_co2e_total,
    mc_vehicle_lookups_total,
    mc_factor_selections_total,
    mc_fleet_operations_total,
    mc_uncertainty_runs_total,
    mc_compliance_checks_total,
    mc_batch_jobs_total,
    mc_calculation_duration_seconds,
    mc_batch_size,
    mc_active_calculations,
    mc_vehicles_registered,
    record_calculation,
    record_emissions,
    record_vehicle_lookup,
    record_factor_selection,
    record_fleet_operation,
    record_uncertainty,
    record_compliance_check,
    record_batch,
    observe_calculation_duration,
    observe_batch_size,
    set_active_calculations,
    set_vehicles_registered,
)


# =========================================================================
# TestMetricExistence - 12 tests
# =========================================================================


class TestMetricExistence:
    """All 12 Prometheus metric objects should be non-None when prometheus_client is available."""

    @pytest.mark.skipif(not PROMETHEUS_AVAILABLE, reason="prometheus_client not installed")
    def test_mc_calculations_total_exists(self) -> None:
        assert mc_calculations_total is not None

    @pytest.mark.skipif(not PROMETHEUS_AVAILABLE, reason="prometheus_client not installed")
    def test_mc_emissions_kg_co2e_total_exists(self) -> None:
        assert mc_emissions_kg_co2e_total is not None

    @pytest.mark.skipif(not PROMETHEUS_AVAILABLE, reason="prometheus_client not installed")
    def test_mc_vehicle_lookups_total_exists(self) -> None:
        assert mc_vehicle_lookups_total is not None

    @pytest.mark.skipif(not PROMETHEUS_AVAILABLE, reason="prometheus_client not installed")
    def test_mc_factor_selections_total_exists(self) -> None:
        assert mc_factor_selections_total is not None

    @pytest.mark.skipif(not PROMETHEUS_AVAILABLE, reason="prometheus_client not installed")
    def test_mc_fleet_operations_total_exists(self) -> None:
        assert mc_fleet_operations_total is not None

    @pytest.mark.skipif(not PROMETHEUS_AVAILABLE, reason="prometheus_client not installed")
    def test_mc_uncertainty_runs_total_exists(self) -> None:
        assert mc_uncertainty_runs_total is not None

    @pytest.mark.skipif(not PROMETHEUS_AVAILABLE, reason="prometheus_client not installed")
    def test_mc_compliance_checks_total_exists(self) -> None:
        assert mc_compliance_checks_total is not None

    @pytest.mark.skipif(not PROMETHEUS_AVAILABLE, reason="prometheus_client not installed")
    def test_mc_batch_jobs_total_exists(self) -> None:
        assert mc_batch_jobs_total is not None

    @pytest.mark.skipif(not PROMETHEUS_AVAILABLE, reason="prometheus_client not installed")
    def test_mc_calculation_duration_seconds_exists(self) -> None:
        assert mc_calculation_duration_seconds is not None

    @pytest.mark.skipif(not PROMETHEUS_AVAILABLE, reason="prometheus_client not installed")
    def test_mc_batch_size_exists(self) -> None:
        assert mc_batch_size is not None

    @pytest.mark.skipif(not PROMETHEUS_AVAILABLE, reason="prometheus_client not installed")
    def test_mc_active_calculations_exists(self) -> None:
        assert mc_active_calculations is not None

    @pytest.mark.skipif(not PROMETHEUS_AVAILABLE, reason="prometheus_client not installed")
    def test_mc_vehicles_registered_exists(self) -> None:
        assert mc_vehicles_registered is not None


# =========================================================================
# TestRecordFunctions - 12 tests
# =========================================================================


class TestRecordFunctions:
    """Each helper function should execute without error."""

    @pytest.mark.skipif(not PROMETHEUS_AVAILABLE, reason="prometheus_client not installed")
    def test_record_calculation_completed(self) -> None:
        record_calculation("PASSENGER_CAR_GASOLINE", "FUEL_BASED", "completed")

    @pytest.mark.skipif(not PROMETHEUS_AVAILABLE, reason="prometheus_client not installed")
    def test_record_calculation_failed(self) -> None:
        record_calculation("HEAVY_DUTY_TRUCK", "DISTANCE_BASED", "failed")

    @pytest.mark.skipif(not PROMETHEUS_AVAILABLE, reason="prometheus_client not installed")
    def test_record_emissions_co2(self) -> None:
        record_emissions("PASSENGER_CAR_GASOLINE", "GASOLINE", "CO2", 1500.0)

    @pytest.mark.skipif(not PROMETHEUS_AVAILABLE, reason="prometheus_client not installed")
    def test_record_emissions_ch4(self) -> None:
        record_emissions("BUS_DIESEL", "DIESEL", "CH4", 0.5)

    @pytest.mark.skipif(not PROMETHEUS_AVAILABLE, reason="prometheus_client not installed")
    def test_record_vehicle_lookup_epa(self) -> None:
        record_vehicle_lookup("EPA")

    @pytest.mark.skipif(not PROMETHEUS_AVAILABLE, reason="prometheus_client not installed")
    def test_record_factor_selection_fuel_based(self) -> None:
        record_factor_selection("FUEL_BASED", "EPA")

    @pytest.mark.skipif(not PROMETHEUS_AVAILABLE, reason="prometheus_client not installed")
    def test_record_fleet_operation_register(self) -> None:
        record_fleet_operation("register", "PASSENGER_CAR_GASOLINE")

    @pytest.mark.skipif(not PROMETHEUS_AVAILABLE, reason="prometheus_client not installed")
    def test_record_uncertainty_monte_carlo(self) -> None:
        record_uncertainty("monte_carlo")

    @pytest.mark.skipif(not PROMETHEUS_AVAILABLE, reason="prometheus_client not installed")
    def test_record_compliance_check_compliant(self) -> None:
        record_compliance_check("GHG_PROTOCOL", "COMPLIANT")

    @pytest.mark.skipif(not PROMETHEUS_AVAILABLE, reason="prometheus_client not installed")
    def test_record_batch_completed(self) -> None:
        record_batch("completed")

    @pytest.mark.skipif(not PROMETHEUS_AVAILABLE, reason="prometheus_client not installed")
    def test_observe_calculation_duration_single(self) -> None:
        observe_calculation_duration("single_calculation", 0.025)

    @pytest.mark.skipif(not PROMETHEUS_AVAILABLE, reason="prometheus_client not installed")
    def test_observe_batch_size_fuel_based(self) -> None:
        observe_batch_size("FUEL_BASED", 100)

    @pytest.mark.skipif(not PROMETHEUS_AVAILABLE, reason="prometheus_client not installed")
    def test_set_active_calculations_gauge(self) -> None:
        set_active_calculations(5)

    @pytest.mark.skipif(not PROMETHEUS_AVAILABLE, reason="prometheus_client not installed")
    def test_set_vehicles_registered_gauge(self) -> None:
        set_vehicles_registered("PASSENGER_CAR_GASOLINE", 42)


# =========================================================================
# TestMetricLabels - 12 tests
# =========================================================================


class TestMetricLabels:
    """Each labeled metric should have the correct label names."""

    @pytest.mark.skipif(not PROMETHEUS_AVAILABLE, reason="prometheus_client not installed")
    def test_calculations_total_labels(self) -> None:
        assert mc_calculations_total._labelnames == ("vehicle_type", "method", "status")

    @pytest.mark.skipif(not PROMETHEUS_AVAILABLE, reason="prometheus_client not installed")
    def test_emissions_total_labels(self) -> None:
        assert mc_emissions_kg_co2e_total._labelnames == ("vehicle_type", "fuel_type", "gas")

    @pytest.mark.skipif(not PROMETHEUS_AVAILABLE, reason="prometheus_client not installed")
    def test_vehicle_lookups_labels(self) -> None:
        assert mc_vehicle_lookups_total._labelnames == ("source",)

    @pytest.mark.skipif(not PROMETHEUS_AVAILABLE, reason="prometheus_client not installed")
    def test_factor_selections_labels(self) -> None:
        assert mc_factor_selections_total._labelnames == ("method", "source")

    @pytest.mark.skipif(not PROMETHEUS_AVAILABLE, reason="prometheus_client not installed")
    def test_fleet_operations_labels(self) -> None:
        assert mc_fleet_operations_total._labelnames == ("operation_type", "vehicle_type")

    @pytest.mark.skipif(not PROMETHEUS_AVAILABLE, reason="prometheus_client not installed")
    def test_uncertainty_runs_labels(self) -> None:
        assert mc_uncertainty_runs_total._labelnames == ("method",)

    @pytest.mark.skipif(not PROMETHEUS_AVAILABLE, reason="prometheus_client not installed")
    def test_compliance_checks_labels(self) -> None:
        assert mc_compliance_checks_total._labelnames == ("framework", "status")

    @pytest.mark.skipif(not PROMETHEUS_AVAILABLE, reason="prometheus_client not installed")
    def test_batch_jobs_labels(self) -> None:
        assert mc_batch_jobs_total._labelnames == ("status",)

    @pytest.mark.skipif(not PROMETHEUS_AVAILABLE, reason="prometheus_client not installed")
    def test_calculation_duration_labels(self) -> None:
        assert mc_calculation_duration_seconds._labelnames == ("operation",)

    @pytest.mark.skipif(not PROMETHEUS_AVAILABLE, reason="prometheus_client not installed")
    def test_batch_size_labels(self) -> None:
        assert mc_batch_size._labelnames == ("method",)

    @pytest.mark.skipif(not PROMETHEUS_AVAILABLE, reason="prometheus_client not installed")
    def test_active_calculations_no_labels(self) -> None:
        # Gauge without labels exposes an empty labelnames tuple
        assert mc_active_calculations._labelnames == ()

    @pytest.mark.skipif(not PROMETHEUS_AVAILABLE, reason="prometheus_client not installed")
    def test_vehicles_registered_labels(self) -> None:
        assert mc_vehicles_registered._labelnames == ("vehicle_type",)


# =========================================================================
# TestGracefulFallback - 12 tests
# =========================================================================


class TestGracefulFallback:
    """When PROMETHEUS_AVAILABLE is False, all helper functions are no-ops."""

    def _run_with_disabled_prometheus(self, func, *args) -> None:
        """Execute func with PROMETHEUS_AVAILABLE patched to False."""
        with patch.object(metrics_mod, "PROMETHEUS_AVAILABLE", False):
            func(*args)

    def test_record_calculation_noop(self) -> None:
        self._run_with_disabled_prometheus(
            record_calculation, "PASSENGER_CAR_GASOLINE", "FUEL_BASED", "completed"
        )

    def test_record_emissions_noop(self) -> None:
        self._run_with_disabled_prometheus(
            record_emissions, "PASSENGER_CAR_GASOLINE", "GASOLINE", "CO2", 100.0
        )

    def test_record_vehicle_lookup_noop(self) -> None:
        self._run_with_disabled_prometheus(
            record_vehicle_lookup, "EPA"
        )

    def test_record_factor_selection_noop(self) -> None:
        self._run_with_disabled_prometheus(
            record_factor_selection, "FUEL_BASED", "EPA"
        )

    def test_record_fleet_operation_noop(self) -> None:
        self._run_with_disabled_prometheus(
            record_fleet_operation, "register", "PASSENGER_CAR_GASOLINE"
        )

    def test_record_uncertainty_noop(self) -> None:
        self._run_with_disabled_prometheus(
            record_uncertainty, "monte_carlo"
        )

    def test_record_compliance_check_noop(self) -> None:
        self._run_with_disabled_prometheus(
            record_compliance_check, "GHG_PROTOCOL", "COMPLIANT"
        )

    def test_record_batch_noop(self) -> None:
        self._run_with_disabled_prometheus(
            record_batch, "completed"
        )

    def test_observe_calculation_duration_noop(self) -> None:
        self._run_with_disabled_prometheus(
            observe_calculation_duration, "single_calculation", 0.01
        )

    def test_observe_batch_size_noop(self) -> None:
        self._run_with_disabled_prometheus(
            observe_batch_size, "FUEL_BASED", 50
        )

    def test_set_active_calculations_noop(self) -> None:
        self._run_with_disabled_prometheus(
            set_active_calculations, 0
        )

    def test_set_vehicles_registered_noop(self) -> None:
        self._run_with_disabled_prometheus(
            set_vehicles_registered, "HEAVY_DUTY_TRUCK", 10
        )
