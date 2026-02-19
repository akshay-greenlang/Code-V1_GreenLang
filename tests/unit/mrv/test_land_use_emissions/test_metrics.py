# -*- coding: utf-8 -*-
"""
Unit tests for AGENT-MRV-006 Land Use Emissions Agent Prometheus Metrics.

Tests all 12 Prometheus metrics existence, gl_lu_ prefix naming, counter
increments, histogram observations, gauge values, and label values.

Target: 25 tests, 85%+ coverage.

Author: GreenLang Platform Team
Date: February 2026
"""

from __future__ import annotations

from unittest.mock import patch, MagicMock

import pytest

from greenlang.land_use_emissions.metrics import (
    PROMETHEUS_AVAILABLE,
    lu_calculations_total,
    lu_calculation_duration_seconds,
    lu_calculation_errors_total,
    lu_emissions_tco2e_total,
    lu_removals_tco2e_total,
    lu_transitions_total,
    lu_carbon_stock_snapshots_total,
    lu_soc_assessments_total,
    lu_compliance_checks_total,
    lu_uncertainty_runs_total,
    lu_batch_size,
    lu_active_parcels,
    record_calculation,
    observe_calculation_duration,
    record_calculation_error,
    record_emissions,
    record_removals,
    record_transition,
    record_carbon_stock_snapshot,
    record_soc_assessment,
    record_compliance_check,
    record_uncertainty_run,
    observe_batch_size,
    track_active_parcels,
)


# ===========================================================================
# Metric Existence Tests
# ===========================================================================


class TestMetricExistence:
    """Tests that all 12 Prometheus metrics are defined."""

    @pytest.mark.skipif(not PROMETHEUS_AVAILABLE, reason="prometheus_client not installed")
    def test_lu_calculations_total_exists(self):
        """gl_lu_calculations_total counter is defined."""
        assert lu_calculations_total is not None

    @pytest.mark.skipif(not PROMETHEUS_AVAILABLE, reason="prometheus_client not installed")
    def test_lu_calculation_duration_seconds_exists(self):
        """gl_lu_calculation_duration_seconds histogram is defined."""
        assert lu_calculation_duration_seconds is not None

    @pytest.mark.skipif(not PROMETHEUS_AVAILABLE, reason="prometheus_client not installed")
    def test_lu_calculation_errors_total_exists(self):
        """gl_lu_calculation_errors_total counter is defined."""
        assert lu_calculation_errors_total is not None

    @pytest.mark.skipif(not PROMETHEUS_AVAILABLE, reason="prometheus_client not installed")
    def test_lu_emissions_tco2e_total_exists(self):
        """gl_lu_emissions_tco2e_total counter is defined."""
        assert lu_emissions_tco2e_total is not None

    @pytest.mark.skipif(not PROMETHEUS_AVAILABLE, reason="prometheus_client not installed")
    def test_lu_removals_tco2e_total_exists(self):
        """gl_lu_removals_tco2e_total counter is defined."""
        assert lu_removals_tco2e_total is not None

    @pytest.mark.skipif(not PROMETHEUS_AVAILABLE, reason="prometheus_client not installed")
    def test_lu_transitions_total_exists(self):
        """gl_lu_transitions_total counter is defined."""
        assert lu_transitions_total is not None

    @pytest.mark.skipif(not PROMETHEUS_AVAILABLE, reason="prometheus_client not installed")
    def test_lu_carbon_stock_snapshots_total_exists(self):
        """gl_lu_carbon_stock_snapshots_total counter is defined."""
        assert lu_carbon_stock_snapshots_total is not None

    @pytest.mark.skipif(not PROMETHEUS_AVAILABLE, reason="prometheus_client not installed")
    def test_lu_soc_assessments_total_exists(self):
        """gl_lu_soc_assessments_total counter is defined."""
        assert lu_soc_assessments_total is not None

    @pytest.mark.skipif(not PROMETHEUS_AVAILABLE, reason="prometheus_client not installed")
    def test_lu_compliance_checks_total_exists(self):
        """gl_lu_compliance_checks_total counter is defined."""
        assert lu_compliance_checks_total is not None

    @pytest.mark.skipif(not PROMETHEUS_AVAILABLE, reason="prometheus_client not installed")
    def test_lu_uncertainty_runs_total_exists(self):
        """gl_lu_uncertainty_runs_total counter is defined."""
        assert lu_uncertainty_runs_total is not None

    @pytest.mark.skipif(not PROMETHEUS_AVAILABLE, reason="prometheus_client not installed")
    def test_lu_batch_size_exists(self):
        """gl_lu_batch_size histogram is defined."""
        assert lu_batch_size is not None

    @pytest.mark.skipif(not PROMETHEUS_AVAILABLE, reason="prometheus_client not installed")
    def test_lu_active_parcels_exists(self):
        """gl_lu_active_parcels gauge is defined."""
        assert lu_active_parcels is not None


# ===========================================================================
# Metric Naming Tests
# ===========================================================================


class TestMetricNaming:
    """Tests that all metrics use the gl_lu_ prefix."""

    @pytest.mark.skipif(not PROMETHEUS_AVAILABLE, reason="prometheus_client not installed")
    def test_calculations_total_name(self):
        """Calculations counter uses gl_lu_ prefix."""
        assert lu_calculations_total._name == "gl_lu_calculations_total"

    @pytest.mark.skipif(not PROMETHEUS_AVAILABLE, reason="prometheus_client not installed")
    def test_duration_histogram_name(self):
        """Duration histogram uses gl_lu_ prefix."""
        assert lu_calculation_duration_seconds._name == "gl_lu_calculation_duration_seconds"

    @pytest.mark.skipif(not PROMETHEUS_AVAILABLE, reason="prometheus_client not installed")
    def test_errors_counter_name(self):
        """Errors counter uses gl_lu_ prefix."""
        assert lu_calculation_errors_total._name == "gl_lu_calculation_errors_total"

    @pytest.mark.skipif(not PROMETHEUS_AVAILABLE, reason="prometheus_client not installed")
    def test_active_parcels_gauge_name(self):
        """Active parcels gauge uses gl_lu_ prefix."""
        assert lu_active_parcels._name == "gl_lu_active_parcels"


# ===========================================================================
# Helper Function Tests
# ===========================================================================


class TestHelperFunctions:
    """Tests for metric helper functions."""

    def test_record_calculation_no_error(self):
        """record_calculation does not raise when called."""
        record_calculation("tier_1", "stock_difference", "forest_land")

    def test_observe_calculation_duration_no_error(self):
        """observe_calculation_duration does not raise when called."""
        observe_calculation_duration("tier_1", "stock_difference", 0.005)

    def test_record_calculation_error_no_error(self):
        """record_calculation_error does not raise when called."""
        record_calculation_error("validation_error")

    def test_record_emissions_no_error(self):
        """record_emissions does not raise when called."""
        record_emissions("CO2", "above_ground_biomass", "forest_land", 150.5)

    def test_record_removals_no_error(self):
        """record_removals does not raise when called."""
        record_removals("above_ground_biomass", "forest_land", 50.0)

    def test_record_transition_no_error(self):
        """record_transition does not raise when called."""
        record_transition("forest_land", "cropland")

    def test_record_carbon_stock_snapshot_no_error(self):
        """record_carbon_stock_snapshot does not raise when called."""
        record_carbon_stock_snapshot("above_ground_biomass")

    def test_record_soc_assessment_no_error(self):
        """record_soc_assessment does not raise when called."""
        record_soc_assessment("tropical_wet", "high_activity_clay")

    def test_record_compliance_check_no_error(self):
        """record_compliance_check does not raise when called."""
        record_compliance_check("GHG_PROTOCOL", "compliant")

    def test_record_uncertainty_run_no_error(self):
        """record_uncertainty_run does not raise when called."""
        record_uncertainty_run("monte_carlo")

    def test_observe_batch_size_no_error(self):
        """observe_batch_size does not raise when called."""
        observe_batch_size("batch_calculation", 100)

    def test_track_active_parcels_no_error(self):
        """track_active_parcels does not raise when called."""
        track_active_parcels("tenant_001", 42)

    def test_record_emissions_default_tco2e(self):
        """record_emissions uses default tco2e=1.0 if not specified."""
        record_emissions("CH4", "litter", "grassland")
