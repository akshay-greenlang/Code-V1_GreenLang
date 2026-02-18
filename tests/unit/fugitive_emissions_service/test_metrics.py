# -*- coding: utf-8 -*-
"""
Unit tests for AGENT-MRV-005 Fugitive Emissions Agent Prometheus Metrics.

Tests all 12 metric definitions, 12 helper functions, disabled mode
(PROMETHEUS_AVAILABLE=False), label combinations, and edge cases.

Test Classes:
    - TestMetricsCreation           (12 tests)
    - TestHelperFunctions           (12 tests)
    - TestMetricsDisabledMode       (12 tests)
    - TestMetricLabelCombinations   (14 tests)
    - TestMetricsEdgeCases          (6 tests)

Total: 50+ tests.

Author: GreenLang Platform Team
Date: February 2026
PRD: AGENT-MRV-005 Fugitive Emissions (GL-MRV-SCOPE1-005)
"""

from __future__ import annotations

from unittest.mock import patch, MagicMock

import pytest

from greenlang.fugitive_emissions.metrics import (
    PROMETHEUS_AVAILABLE,
    # Metric objects
    fe_calculations_total,
    fe_emissions_kg_co2e_total,
    fe_source_lookups_total,
    fe_factor_selections_total,
    fe_ldar_surveys_total,
    fe_uncertainty_runs_total,
    fe_compliance_checks_total,
    fe_batch_jobs_total,
    fe_calculation_duration_seconds,
    fe_batch_size,
    fe_active_calculations,
    fe_components_registered,
    # Helper functions
    record_calculation,
    record_emissions,
    record_source_lookup,
    record_factor_selection,
    record_ldar_survey,
    record_uncertainty_run,
    record_compliance_check,
    record_batch_job,
    observe_calculation_duration,
    observe_batch_size,
    track_active_calculation,
    set_components_registered,
)


# ==========================================================================
# TestMetricsCreation - 12 tests
# ==========================================================================


class TestMetricsCreation:
    """Verify all 12 metric objects are created (or None when unavailable)."""

    def test_calculations_total_exists(self):
        if PROMETHEUS_AVAILABLE:
            assert fe_calculations_total is not None
        else:
            assert fe_calculations_total is None

    def test_emissions_kg_co2e_total_exists(self):
        if PROMETHEUS_AVAILABLE:
            assert fe_emissions_kg_co2e_total is not None
        else:
            assert fe_emissions_kg_co2e_total is None

    def test_source_lookups_total_exists(self):
        if PROMETHEUS_AVAILABLE:
            assert fe_source_lookups_total is not None
        else:
            assert fe_source_lookups_total is None

    def test_factor_selections_total_exists(self):
        if PROMETHEUS_AVAILABLE:
            assert fe_factor_selections_total is not None
        else:
            assert fe_factor_selections_total is None

    def test_ldar_surveys_total_exists(self):
        if PROMETHEUS_AVAILABLE:
            assert fe_ldar_surveys_total is not None
        else:
            assert fe_ldar_surveys_total is None

    def test_uncertainty_runs_total_exists(self):
        if PROMETHEUS_AVAILABLE:
            assert fe_uncertainty_runs_total is not None
        else:
            assert fe_uncertainty_runs_total is None

    def test_compliance_checks_total_exists(self):
        if PROMETHEUS_AVAILABLE:
            assert fe_compliance_checks_total is not None
        else:
            assert fe_compliance_checks_total is None

    def test_batch_jobs_total_exists(self):
        if PROMETHEUS_AVAILABLE:
            assert fe_batch_jobs_total is not None
        else:
            assert fe_batch_jobs_total is None

    def test_calculation_duration_seconds_exists(self):
        if PROMETHEUS_AVAILABLE:
            assert fe_calculation_duration_seconds is not None
        else:
            assert fe_calculation_duration_seconds is None

    def test_batch_size_exists(self):
        if PROMETHEUS_AVAILABLE:
            assert fe_batch_size is not None
        else:
            assert fe_batch_size is None

    def test_active_calculations_exists(self):
        if PROMETHEUS_AVAILABLE:
            assert fe_active_calculations is not None
        else:
            assert fe_active_calculations is None

    def test_components_registered_exists(self):
        if PROMETHEUS_AVAILABLE:
            assert fe_components_registered is not None
        else:
            assert fe_components_registered is None


# ==========================================================================
# TestHelperFunctions - 12 tests
# ==========================================================================


class TestHelperFunctions:
    """Test that all 12 helper functions execute without exceptions."""

    def test_record_calculation_runs(self):
        # Should not raise regardless of prometheus availability
        record_calculation("valve_gas", "AVERAGE_EMISSION_FACTOR", "completed")

    def test_record_emissions_runs(self):
        record_emissions("valve_gas", "CH4", 120.5)

    def test_record_emissions_default_amount(self):
        record_emissions("pump_seal", "VOC")

    def test_record_source_lookup_runs(self):
        record_source_lookup("EPA")

    def test_record_factor_selection_runs(self):
        record_factor_selection("AVERAGE_EMISSION_FACTOR", "EPA")

    def test_record_ldar_survey_runs(self):
        record_ldar_survey("OGI", "completed")

    def test_record_uncertainty_run_runs(self):
        record_uncertainty_run("monte_carlo")

    def test_record_compliance_check_runs(self):
        record_compliance_check("GHG_PROTOCOL", "COMPLIANT")

    def test_record_batch_job_runs(self):
        record_batch_job("completed")

    def test_observe_calculation_duration_runs(self):
        observe_calculation_duration("single_calculation", 0.005)

    def test_observe_batch_size_runs(self):
        observe_batch_size("AVERAGE_EMISSION_FACTOR", 50)

    def test_track_active_calculation_runs(self):
        track_active_calculation(3)

    def test_set_components_registered_runs(self):
        set_components_registered("valve", 150)


# ==========================================================================
# TestMetricsDisabledMode - 12 tests
# ==========================================================================


class TestMetricsDisabledMode:
    """Test that helpers are no-ops when PROMETHEUS_AVAILABLE is False."""

    def _run_with_prom_disabled(self, func, *args):
        """Helper to run a function with PROMETHEUS_AVAILABLE mocked to False."""
        with patch(
            "greenlang.fugitive_emissions.metrics.PROMETHEUS_AVAILABLE", False
        ):
            # Should return None and not raise
            result = func(*args)
            assert result is None

    def test_record_calculation_disabled(self):
        self._run_with_prom_disabled(
            record_calculation, "valve_gas", "AVERAGE_EMISSION_FACTOR", "completed"
        )

    def test_record_emissions_disabled(self):
        self._run_with_prom_disabled(
            record_emissions, "valve_gas", "CH4", 120.5
        )

    def test_record_source_lookup_disabled(self):
        self._run_with_prom_disabled(record_source_lookup, "EPA")

    def test_record_factor_selection_disabled(self):
        self._run_with_prom_disabled(
            record_factor_selection, "AVERAGE_EMISSION_FACTOR", "EPA"
        )

    def test_record_ldar_survey_disabled(self):
        self._run_with_prom_disabled(record_ldar_survey, "OGI", "completed")

    def test_record_uncertainty_run_disabled(self):
        self._run_with_prom_disabled(record_uncertainty_run, "monte_carlo")

    def test_record_compliance_check_disabled(self):
        self._run_with_prom_disabled(
            record_compliance_check, "GHG_PROTOCOL", "COMPLIANT"
        )

    def test_record_batch_job_disabled(self):
        self._run_with_prom_disabled(record_batch_job, "completed")

    def test_observe_calculation_duration_disabled(self):
        self._run_with_prom_disabled(
            observe_calculation_duration, "single_calculation", 0.005
        )

    def test_observe_batch_size_disabled(self):
        self._run_with_prom_disabled(
            observe_batch_size, "AVERAGE_EMISSION_FACTOR", 50
        )

    def test_track_active_calculation_disabled(self):
        self._run_with_prom_disabled(track_active_calculation, 3)

    def test_set_components_registered_disabled(self):
        self._run_with_prom_disabled(set_components_registered, "valve", 150)


# ==========================================================================
# TestMetricLabelCombinations - 14 tests
# ==========================================================================


class TestMetricLabelCombinations:
    """Test various label value combinations for metrics."""

    @pytest.mark.parametrize("source_type", [
        "wellhead", "separator", "dehydrator",
        "pneumatic_controller_high", "valve_gas",
        "pump_seal", "coal_mine_underground",
        "wastewater_lagoon",
    ])
    def test_record_calculation_source_types(self, source_type):
        record_calculation(source_type, "AVERAGE_EMISSION_FACTOR", "completed")

    @pytest.mark.parametrize("method", [
        "AVERAGE_EMISSION_FACTOR", "SCREENING_RANGES",
        "CORRELATION_EQUATION", "ENGINEERING_ESTIMATE",
        "DIRECT_MEASUREMENT",
    ])
    def test_record_calculation_methods(self, method):
        record_calculation("valve_gas", method, "completed")

    @pytest.mark.parametrize("status", [
        "completed", "failed", "pending", "running",
    ])
    def test_record_calculation_statuses(self, status):
        record_calculation("valve_gas", "AVERAGE_EMISSION_FACTOR", status)

    @pytest.mark.parametrize("gas", ["CH4", "CO2", "N2O", "VOC"])
    def test_record_emissions_gases(self, gas):
        record_emissions("valve_gas", gas, 100.0)

    @pytest.mark.parametrize("source", [
        "EPA", "IPCC", "DEFRA", "EU_ETS", "API", "CUSTOM",
    ])
    def test_record_source_lookup_sources(self, source):
        record_source_lookup(source)

    @pytest.mark.parametrize("survey_type", [
        "OGI", "METHOD_21", "AVO", "HI_FLOW",
    ])
    def test_record_ldar_survey_types(self, survey_type):
        record_ldar_survey(survey_type, "completed")

    @pytest.mark.parametrize("framework", [
        "GHG_PROTOCOL", "ISO_14064", "CSRD_ESRS_E1",
        "EPA_40CFR98", "UK_SECR", "EU_ETS",
    ])
    def test_record_compliance_check_frameworks(self, framework):
        record_compliance_check(framework, "COMPLIANT")

    @pytest.mark.parametrize("component_type", [
        "valve", "pump", "compressor", "pressure_relief_device",
        "connector", "open_ended_line", "sampling_connection",
        "flange", "other",
    ])
    def test_set_components_registered_types(self, component_type):
        set_components_registered(component_type, 50)

    @pytest.mark.parametrize("operation", [
        "single_calculation", "batch_calculation", "factor_lookup",
        "unit_conversion", "gwp_application", "uncertainty_analysis",
        "provenance_hash", "compliance_check",
        "ldar_survey_processing", "component_registration",
    ])
    def test_observe_duration_operations(self, operation):
        observe_calculation_duration(operation, 0.01)

    def test_record_emissions_zero_amount(self):
        record_emissions("valve_gas", "CH4", 0.0)

    def test_record_emissions_large_amount(self):
        record_emissions("valve_gas", "CH4", 1_000_000.0)

    def test_track_active_calculation_zero(self):
        track_active_calculation(0)

    def test_track_active_calculation_large(self):
        track_active_calculation(10000)

    def test_observe_batch_size_one(self):
        observe_batch_size("AVERAGE_EMISSION_FACTOR", 1)


# ==========================================================================
# TestMetricsEdgeCases - 6 tests
# ==========================================================================


class TestMetricsEdgeCases:
    """Test edge cases for metric recording functions."""

    def test_record_emissions_very_small_amount(self):
        """Recording a very small emission amount should not raise."""
        record_emissions("valve_gas", "CH4", 0.000001)

    def test_observe_duration_very_small(self):
        """Recording a sub-millisecond duration should not raise."""
        observe_calculation_duration("single_calculation", 0.0001)

    def test_observe_duration_very_large(self):
        """Recording a large duration should not raise."""
        observe_calculation_duration("batch_calculation", 600.0)

    def test_observe_batch_size_large(self):
        """Recording a large batch size should not raise."""
        observe_batch_size("AVERAGE_EMISSION_FACTOR", 10000)

    def test_multiple_calls_same_labels(self):
        """Calling the same helper with same labels many times should not raise."""
        for _ in range(100):
            record_calculation("valve_gas", "AVERAGE_EMISSION_FACTOR", "completed")

    def test_set_components_registered_zero(self):
        """Setting component count to zero should not raise."""
        set_components_registered("valve", 0)

    def test_record_factor_selection_all_combinations(self):
        """Test all method x source combinations do not raise."""
        methods = [
            "AVERAGE_EMISSION_FACTOR", "SCREENING_RANGES",
            "CORRELATION_EQUATION", "ENGINEERING_ESTIMATE",
            "DIRECT_MEASUREMENT",
        ]
        sources = ["EPA", "IPCC", "DEFRA", "EU_ETS", "API", "CUSTOM"]
        for method in methods:
            for source in sources:
                record_factor_selection(method, source)

    def test_record_ldar_survey_all_status_combinations(self):
        """Test all survey_type x status combinations do not raise."""
        survey_types = ["OGI", "METHOD_21", "AVO", "HI_FLOW"]
        statuses = ["completed", "failed", "pending", "running"]
        for st in survey_types:
            for status in statuses:
                record_ldar_survey(st, status)

    def test_record_batch_job_all_statuses(self):
        """Test all batch job statuses do not raise."""
        for status in ["completed", "failed", "pending", "running"]:
            record_batch_job(status)

    def test_record_uncertainty_run_analytical(self):
        """Test analytical uncertainty method does not raise."""
        record_uncertainty_run("analytical")
