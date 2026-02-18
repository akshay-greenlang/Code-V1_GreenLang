# -*- coding: utf-8 -*-
"""
Unit tests for Refrigerants & F-Gas Agent Prometheus Metrics - AGENT-MRV-002

Tests all 12 Prometheus metrics, 12 helper functions, graceful fallback
behavior when prometheus_client is not installed, and label validation.

Target: 45+ tests.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from greenlang.refrigerants_fgas.metrics import (
    PROMETHEUS_AVAILABLE,
    # Metric objects
    rf_calculations_total,
    rf_emissions_kg_co2e_total,
    rf_refrigerant_lookups_total,
    rf_leak_rate_selections_total,
    rf_equipment_events_total,
    rf_uncertainty_runs_total,
    rf_compliance_checks_total,
    rf_batch_jobs_total,
    rf_calculation_duration_seconds,
    rf_batch_size,
    rf_active_calculations,
    rf_refrigerants_loaded,
    # Helper functions
    record_calculation,
    record_emissions,
    record_refrigerant_lookup,
    record_leak_rate_selection,
    record_equipment_event,
    record_uncertainty,
    record_compliance_check,
    record_batch,
    observe_calculation_duration,
    observe_batch_size,
    set_active_calculations,
    set_refrigerants_loaded,
)


# ===================================================================
# PROMETHEUS_AVAILABLE flag
# ===================================================================


class TestPrometheusAvailable:
    """Test the PROMETHEUS_AVAILABLE flag."""

    def test_prometheus_available_is_bool(self):
        assert isinstance(PROMETHEUS_AVAILABLE, bool)

    def test_prometheus_available_true_when_installed(self):
        """prometheus_client should be installed in dev environment."""
        assert PROMETHEUS_AVAILABLE is True


# ===================================================================
# Metric existence tests
# ===================================================================


class TestMetricExistence:
    """Verify all 12 metric objects are defined and not None."""

    def test_rf_calculations_total_exists(self):
        assert rf_calculations_total is not None

    def test_rf_emissions_kg_co2e_total_exists(self):
        assert rf_emissions_kg_co2e_total is not None

    def test_rf_refrigerant_lookups_total_exists(self):
        assert rf_refrigerant_lookups_total is not None

    def test_rf_leak_rate_selections_total_exists(self):
        assert rf_leak_rate_selections_total is not None

    def test_rf_equipment_events_total_exists(self):
        assert rf_equipment_events_total is not None

    def test_rf_uncertainty_runs_total_exists(self):
        assert rf_uncertainty_runs_total is not None

    def test_rf_compliance_checks_total_exists(self):
        assert rf_compliance_checks_total is not None

    def test_rf_batch_jobs_total_exists(self):
        assert rf_batch_jobs_total is not None

    def test_rf_calculation_duration_seconds_exists(self):
        assert rf_calculation_duration_seconds is not None

    def test_rf_batch_size_exists(self):
        assert rf_batch_size is not None

    def test_rf_active_calculations_exists(self):
        assert rf_active_calculations is not None

    def test_rf_refrigerants_loaded_exists(self):
        assert rf_refrigerants_loaded is not None


# ===================================================================
# Helper function tests - record_calculation
# ===================================================================


class TestRecordCalculation:
    """Test record_calculation helper function."""

    def test_record_calculation_does_not_raise(self):
        record_calculation("EQUIPMENT_BASED", "R_410A", "completed")

    @pytest.mark.parametrize("method", [
        "EQUIPMENT_BASED", "MASS_BALANCE", "SCREENING",
        "DIRECT_MEASUREMENT", "TOP_DOWN",
    ])
    def test_record_calculation_all_methods(self, method: str):
        record_calculation(method, "R_134A", "completed")

    @pytest.mark.parametrize("status", [
        "completed", "failed", "pending", "running", "cancelled",
    ])
    def test_record_calculation_all_statuses(self, status: str):
        record_calculation("EQUIPMENT_BASED", "SF6_GAS", status)

    def test_record_calculation_custom_refrigerant(self):
        record_calculation("SCREENING", "CUSTOM", "completed")


# ===================================================================
# Helper function tests - record_emissions
# ===================================================================


class TestRecordEmissions:
    """Test record_emissions helper function."""

    def test_record_emissions_does_not_raise(self):
        record_emissions("R_410A", "hfc_blend", 2500.0)

    def test_record_emissions_default_amount(self):
        record_emissions("R_134A", "hfc")

    @pytest.mark.parametrize("category", [
        "hfc", "hfc_blend", "hfo", "pfc", "sf6",
        "nf3", "hcfc", "cfc", "natural", "other",
    ])
    def test_record_emissions_all_categories(self, category: str):
        record_emissions("R_32", category, 100.0)

    def test_record_emissions_zero_amount(self):
        record_emissions("R_744", "natural", 0.0)


# ===================================================================
# Helper function tests - record_refrigerant_lookup
# ===================================================================


class TestRecordRefrigerantLookup:
    """Test record_refrigerant_lookup helper."""

    def test_does_not_raise(self):
        record_refrigerant_lookup("AR6")

    @pytest.mark.parametrize("source", [
        "AR4", "AR5", "AR6", "AR6_20YR", "CUSTOM", "database", "cache",
    ])
    def test_all_sources(self, source: str):
        record_refrigerant_lookup(source)


# ===================================================================
# Helper function tests - record_leak_rate_selection
# ===================================================================


class TestRecordLeakRateSelection:
    """Test record_leak_rate_selection helper."""

    def test_does_not_raise(self):
        record_leak_rate_selection("commercial_ac", "operating")

    @pytest.mark.parametrize("lifecycle_stage", [
        "installation", "operating", "end_of_life",
    ])
    def test_all_lifecycle_stages(self, lifecycle_stage: str):
        record_leak_rate_selection("chillers_centrifugal", lifecycle_stage)


# ===================================================================
# Helper function tests - record_equipment_event
# ===================================================================


class TestRecordEquipmentEvent:
    """Test record_equipment_event helper."""

    def test_does_not_raise(self):
        record_equipment_event("commercial_ac", "recharge")

    @pytest.mark.parametrize("event_type", [
        "installation", "recharge", "repair", "recovery",
        "leak_check", "decommissioning", "conversion",
    ])
    def test_all_event_types(self, event_type: str):
        record_equipment_event("switchgear", event_type)


# ===================================================================
# Helper function tests - record_uncertainty
# ===================================================================


class TestRecordUncertainty:
    """Test record_uncertainty helper."""

    def test_does_not_raise(self):
        record_uncertainty("monte_carlo")

    @pytest.mark.parametrize("method", [
        "monte_carlo", "analytical", "tier_default",
    ])
    def test_all_methods(self, method: str):
        record_uncertainty(method)


# ===================================================================
# Helper function tests - record_compliance_check
# ===================================================================


class TestRecordComplianceCheck:
    """Test record_compliance_check helper."""

    def test_does_not_raise(self):
        record_compliance_check("eu_fgas_2024_573", "compliant")

    @pytest.mark.parametrize("status", [
        "compliant", "warning", "non_compliant", "exempted", "not_applicable",
    ])
    def test_all_statuses(self, status: str):
        record_compliance_check("ghg_protocol", status)

    @pytest.mark.parametrize("framework", [
        "ghg_protocol", "iso_14064", "csrd_esrs_e1",
        "epa_40cfr98_dd", "epa_40cfr98_oo", "epa_40cfr98_l",
        "eu_fgas_2024_573", "kigali_amendment", "uk_fgas",
    ])
    def test_all_frameworks(self, framework: str):
        record_compliance_check(framework, "compliant")


# ===================================================================
# Helper function tests - record_batch
# ===================================================================


class TestRecordBatch:
    """Test record_batch helper."""

    def test_does_not_raise(self):
        record_batch("completed")

    @pytest.mark.parametrize("status", [
        "completed", "failed", "pending", "running", "cancelled",
    ])
    def test_all_statuses(self, status: str):
        record_batch(status)


# ===================================================================
# Helper function tests - observe_calculation_duration
# ===================================================================


class TestObserveCalculationDuration:
    """Test observe_calculation_duration helper."""

    def test_does_not_raise(self):
        observe_calculation_duration("single_calculation", 0.005)

    @pytest.mark.parametrize("operation", [
        "single_calculation", "batch_calculation", "refrigerant_lookup",
        "blend_decomposition", "leak_rate_estimation", "gwp_application",
        "uncertainty_analysis", "compliance_check", "provenance_hash",
        "pipeline_run",
    ])
    def test_all_operations(self, operation: str):
        observe_calculation_duration(operation, 0.01)

    def test_zero_duration(self):
        observe_calculation_duration("single_calculation", 0.0)

    def test_large_duration(self):
        observe_calculation_duration("pipeline_run", 30.0)


# ===================================================================
# Helper function tests - observe_batch_size
# ===================================================================


class TestObserveBatchSize:
    """Test observe_batch_size helper."""

    def test_does_not_raise(self):
        observe_batch_size("EQUIPMENT_BASED", 100)

    def test_mixed_method(self):
        observe_batch_size("mixed", 500)

    def test_single_item_batch(self):
        observe_batch_size("SCREENING", 1)


# ===================================================================
# Helper function tests - set_active_calculations
# ===================================================================


class TestSetActiveCalculations:
    """Test set_active_calculations helper."""

    def test_does_not_raise(self):
        set_active_calculations(5)

    def test_zero(self):
        set_active_calculations(0)

    def test_large_value(self):
        set_active_calculations(1000)


# ===================================================================
# Helper function tests - set_refrigerants_loaded
# ===================================================================


class TestSetRefrigerantsLoaded:
    """Test set_refrigerants_loaded helper."""

    def test_does_not_raise(self):
        set_refrigerants_loaded("AR6", 50)

    def test_zero_count(self):
        set_refrigerants_loaded("database", 0)

    @pytest.mark.parametrize("source", ["AR4", "AR5", "AR6", "CUSTOM", "database"])
    def test_all_sources(self, source: str):
        set_refrigerants_loaded(source, 100)


# ===================================================================
# Metric label tests
# ===================================================================


class TestMetricLabels:
    """Verify metric objects have the expected label names."""

    def test_calculations_total_labels(self):
        assert rf_calculations_total._labelnames == ("method", "refrigerant_type", "status")

    def test_emissions_total_labels(self):
        assert rf_emissions_kg_co2e_total._labelnames == ("refrigerant_type", "category")

    def test_lookups_total_labels(self):
        assert rf_refrigerant_lookups_total._labelnames == ("source",)

    def test_leak_rate_selections_labels(self):
        assert rf_leak_rate_selections_total._labelnames == ("equipment_type", "lifecycle_stage")

    def test_equipment_events_labels(self):
        assert rf_equipment_events_total._labelnames == ("equipment_type", "event_type")

    def test_uncertainty_runs_labels(self):
        assert rf_uncertainty_runs_total._labelnames == ("method",)

    def test_compliance_checks_labels(self):
        assert rf_compliance_checks_total._labelnames == ("framework", "status")

    def test_batch_jobs_labels(self):
        assert rf_batch_jobs_total._labelnames == ("status",)

    def test_calculation_duration_labels(self):
        assert rf_calculation_duration_seconds._labelnames == ("operation",)

    def test_batch_size_labels(self):
        assert rf_batch_size._labelnames == ("method",)

    def test_refrigerants_loaded_labels(self):
        assert rf_refrigerants_loaded._labelnames == ("source",)


# ===================================================================
# Graceful fallback tests (when prometheus not available)
# ===================================================================


class TestGracefulFallback:
    """Verify helpers do not raise when PROMETHEUS_AVAILABLE is False."""

    @patch("greenlang.refrigerants_fgas.metrics.PROMETHEUS_AVAILABLE", False)
    def test_record_calculation_noop(self):
        record_calculation("EQUIPMENT_BASED", "R_410A", "completed")

    @patch("greenlang.refrigerants_fgas.metrics.PROMETHEUS_AVAILABLE", False)
    def test_record_emissions_noop(self):
        record_emissions("R_410A", "hfc_blend", 100.0)

    @patch("greenlang.refrigerants_fgas.metrics.PROMETHEUS_AVAILABLE", False)
    def test_record_refrigerant_lookup_noop(self):
        record_refrigerant_lookup("AR6")

    @patch("greenlang.refrigerants_fgas.metrics.PROMETHEUS_AVAILABLE", False)
    def test_record_leak_rate_selection_noop(self):
        record_leak_rate_selection("commercial_ac", "operating")

    @patch("greenlang.refrigerants_fgas.metrics.PROMETHEUS_AVAILABLE", False)
    def test_record_equipment_event_noop(self):
        record_equipment_event("switchgear", "installation")

    @patch("greenlang.refrigerants_fgas.metrics.PROMETHEUS_AVAILABLE", False)
    def test_record_uncertainty_noop(self):
        record_uncertainty("monte_carlo")

    @patch("greenlang.refrigerants_fgas.metrics.PROMETHEUS_AVAILABLE", False)
    def test_record_compliance_check_noop(self):
        record_compliance_check("eu_fgas_2024_573", "compliant")

    @patch("greenlang.refrigerants_fgas.metrics.PROMETHEUS_AVAILABLE", False)
    def test_record_batch_noop(self):
        record_batch("completed")

    @patch("greenlang.refrigerants_fgas.metrics.PROMETHEUS_AVAILABLE", False)
    def test_observe_calculation_duration_noop(self):
        observe_calculation_duration("single_calculation", 0.1)

    @patch("greenlang.refrigerants_fgas.metrics.PROMETHEUS_AVAILABLE", False)
    def test_observe_batch_size_noop(self):
        observe_batch_size("EQUIPMENT_BASED", 50)

    @patch("greenlang.refrigerants_fgas.metrics.PROMETHEUS_AVAILABLE", False)
    def test_set_active_calculations_noop(self):
        set_active_calculations(0)

    @patch("greenlang.refrigerants_fgas.metrics.PROMETHEUS_AVAILABLE", False)
    def test_set_refrigerants_loaded_noop(self):
        set_refrigerants_loaded("AR6", 50)
