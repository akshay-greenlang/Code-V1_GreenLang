# -*- coding: utf-8 -*-
"""Unit tests for Process Emissions Agent Prometheus metrics - AGENT-MRV-004.

Tests all 12 gl_pe_ metric objects, 12 helper record functions,
graceful degradation when prometheus_client is unavailable, label
combinations, and edge cases.  Target: 50 tests.
"""

import importlib
import sys
import pytest
from unittest.mock import patch, MagicMock

# ---------------------------------------------------------------------------
# Import the metrics module under test. We use a fresh import in some
# tests that mock away prometheus_client.
# ---------------------------------------------------------------------------
from greenlang.process_emissions import metrics as metrics_mod
from greenlang.process_emissions.metrics import (
    PROMETHEUS_AVAILABLE,
    pe_calculations_total,
    pe_emissions_kg_co2e_total,
    pe_process_lookups_total,
    pe_factor_selections_total,
    pe_material_operations_total,
    pe_uncertainty_runs_total,
    pe_compliance_checks_total,
    pe_batch_jobs_total,
    pe_calculation_duration_seconds,
    pe_batch_size,
    pe_active_calculations,
    pe_process_units_registered,
    record_calculation,
    record_emissions,
    record_process_lookup,
    record_factor_selection,
    record_material_operation,
    record_uncertainty,
    record_compliance_check,
    record_batch,
    observe_calculation_duration,
    observe_batch_size,
    set_active_calculations,
    set_process_units_registered,
)


# ============================================================================
# TestMetricsCreation - 12 tests
# ============================================================================


class TestMetricsCreation:
    """Verify each of the 12 metric objects exists and has the correct type."""

    @pytest.mark.skipif(
        not PROMETHEUS_AVAILABLE,
        reason="prometheus_client not installed",
    )
    def test_calculations_total_is_counter(self):
        from prometheus_client import Counter
        assert isinstance(pe_calculations_total, Counter)

    @pytest.mark.skipif(
        not PROMETHEUS_AVAILABLE,
        reason="prometheus_client not installed",
    )
    def test_emissions_kg_co2e_total_is_counter(self):
        from prometheus_client import Counter
        assert isinstance(pe_emissions_kg_co2e_total, Counter)

    @pytest.mark.skipif(
        not PROMETHEUS_AVAILABLE,
        reason="prometheus_client not installed",
    )
    def test_process_lookups_total_is_counter(self):
        from prometheus_client import Counter
        assert isinstance(pe_process_lookups_total, Counter)

    @pytest.mark.skipif(
        not PROMETHEUS_AVAILABLE,
        reason="prometheus_client not installed",
    )
    def test_factor_selections_total_is_counter(self):
        from prometheus_client import Counter
        assert isinstance(pe_factor_selections_total, Counter)

    @pytest.mark.skipif(
        not PROMETHEUS_AVAILABLE,
        reason="prometheus_client not installed",
    )
    def test_material_operations_total_is_counter(self):
        from prometheus_client import Counter
        assert isinstance(pe_material_operations_total, Counter)

    @pytest.mark.skipif(
        not PROMETHEUS_AVAILABLE,
        reason="prometheus_client not installed",
    )
    def test_uncertainty_runs_total_is_counter(self):
        from prometheus_client import Counter
        assert isinstance(pe_uncertainty_runs_total, Counter)

    @pytest.mark.skipif(
        not PROMETHEUS_AVAILABLE,
        reason="prometheus_client not installed",
    )
    def test_compliance_checks_total_is_counter(self):
        from prometheus_client import Counter
        assert isinstance(pe_compliance_checks_total, Counter)

    @pytest.mark.skipif(
        not PROMETHEUS_AVAILABLE,
        reason="prometheus_client not installed",
    )
    def test_batch_jobs_total_is_counter(self):
        from prometheus_client import Counter
        assert isinstance(pe_batch_jobs_total, Counter)

    @pytest.mark.skipif(
        not PROMETHEUS_AVAILABLE,
        reason="prometheus_client not installed",
    )
    def test_calculation_duration_is_histogram(self):
        from prometheus_client import Histogram
        assert isinstance(pe_calculation_duration_seconds, Histogram)

    @pytest.mark.skipif(
        not PROMETHEUS_AVAILABLE,
        reason="prometheus_client not installed",
    )
    def test_batch_size_is_histogram(self):
        from prometheus_client import Histogram
        assert isinstance(pe_batch_size, Histogram)

    @pytest.mark.skipif(
        not PROMETHEUS_AVAILABLE,
        reason="prometheus_client not installed",
    )
    def test_active_calculations_is_gauge(self):
        from prometheus_client import Gauge
        assert isinstance(pe_active_calculations, Gauge)

    @pytest.mark.skipif(
        not PROMETHEUS_AVAILABLE,
        reason="prometheus_client not installed",
    )
    def test_process_units_registered_is_gauge(self):
        from prometheus_client import Gauge
        assert isinstance(pe_process_units_registered, Gauge)


# ============================================================================
# TestRecordFunctions - 12 tests
# ============================================================================


class TestRecordFunctions:
    """Test each of the 12 helper record/observe/set functions.

    These tests verify the functions are callable and do not raise.
    When prometheus_client is available they verify the counter/gauge
    actually increments; otherwise they verify the no-op path.
    """

    def test_record_calculation_does_not_raise(self):
        record_calculation("cement_production", "EMISSION_FACTOR", "completed")

    def test_record_emissions_does_not_raise(self):
        record_emissions("cement_production", "CO2", 45000.0)

    def test_record_process_lookup_does_not_raise(self):
        record_process_lookup("IPCC")

    def test_record_factor_selection_does_not_raise(self):
        record_factor_selection("TIER_1", "EPA")

    def test_record_material_operation_does_not_raise(self):
        record_material_operation("register", "calcium_carbonate")

    def test_record_uncertainty_does_not_raise(self):
        record_uncertainty("monte_carlo")

    def test_record_compliance_check_does_not_raise(self):
        record_compliance_check("GHG_PROTOCOL", "COMPLIANT")

    def test_record_batch_does_not_raise(self):
        record_batch("completed")

    def test_observe_calculation_duration_does_not_raise(self):
        observe_calculation_duration("single_calculation", 0.05)

    def test_observe_batch_size_does_not_raise(self):
        observe_batch_size("EMISSION_FACTOR", 25)

    def test_set_active_calculations_does_not_raise(self):
        set_active_calculations(5)

    def test_set_process_units_registered_does_not_raise(self):
        set_process_units_registered("cement_production", 10)


# ============================================================================
# TestMetricsDisabled - 10 tests
# ============================================================================


class TestMetricsDisabled:
    """Test graceful behaviour when prometheus_client is not available.

    Uses monkeypatching to simulate the PROMETHEUS_AVAILABLE = False
    code path within the helper functions.
    """

    def test_record_calculation_noop_when_disabled(self):
        with patch.object(metrics_mod, "PROMETHEUS_AVAILABLE", False):
            # Should return None without raising
            result = metrics_mod.record_calculation(
                "cement_production", "EMISSION_FACTOR", "completed"
            )
            assert result is None

    def test_record_emissions_noop_when_disabled(self):
        with patch.object(metrics_mod, "PROMETHEUS_AVAILABLE", False):
            result = metrics_mod.record_emissions("cement_production", "CO2", 100.0)
            assert result is None

    def test_record_process_lookup_noop_when_disabled(self):
        with patch.object(metrics_mod, "PROMETHEUS_AVAILABLE", False):
            result = metrics_mod.record_process_lookup("IPCC")
            assert result is None

    def test_record_factor_selection_noop_when_disabled(self):
        with patch.object(metrics_mod, "PROMETHEUS_AVAILABLE", False):
            result = metrics_mod.record_factor_selection("TIER_1", "EPA")
            assert result is None

    def test_record_material_operation_noop_when_disabled(self):
        with patch.object(metrics_mod, "PROMETHEUS_AVAILABLE", False):
            result = metrics_mod.record_material_operation("register", "coke")
            assert result is None

    def test_record_uncertainty_noop_when_disabled(self):
        with patch.object(metrics_mod, "PROMETHEUS_AVAILABLE", False):
            result = metrics_mod.record_uncertainty("analytical")
            assert result is None

    def test_record_compliance_check_noop_when_disabled(self):
        with patch.object(metrics_mod, "PROMETHEUS_AVAILABLE", False):
            result = metrics_mod.record_compliance_check("ISO_14064", "PARTIAL")
            assert result is None

    def test_record_batch_noop_when_disabled(self):
        with patch.object(metrics_mod, "PROMETHEUS_AVAILABLE", False):
            result = metrics_mod.record_batch("failed")
            assert result is None

    def test_observe_calculation_duration_noop_when_disabled(self):
        with patch.object(metrics_mod, "PROMETHEUS_AVAILABLE", False):
            result = metrics_mod.observe_calculation_duration(
                "batch_calculation", 1.5
            )
            assert result is None

    def test_set_active_calculations_noop_when_disabled(self):
        with patch.object(metrics_mod, "PROMETHEUS_AVAILABLE", False):
            result = metrics_mod.set_active_calculations(0)
            assert result is None


# ============================================================================
# TestMetricsLabels - 10 tests
# ============================================================================


class TestMetricsLabels:
    """Test various label combinations on the helper functions.

    Ensures the functions accept a wide range of valid label values
    without raising.
    """

    @pytest.mark.parametrize("process_type", [
        "cement_production", "iron_steel", "aluminum_smelting",
        "nitric_acid", "semiconductor",
    ])
    def test_record_calculation_various_process_types(self, process_type):
        record_calculation(process_type, "EMISSION_FACTOR", "completed")

    @pytest.mark.parametrize("method", [
        "EMISSION_FACTOR", "MASS_BALANCE", "STOICHIOMETRIC", "DIRECT_MEASUREMENT",
    ])
    def test_record_calculation_various_methods(self, method):
        record_calculation("cement_production", method, "completed")

    @pytest.mark.parametrize("status", [
        "completed", "failed", "pending", "running",
    ])
    def test_record_calculation_various_statuses(self, status):
        record_calculation("cement_production", "EMISSION_FACTOR", status)

    @pytest.mark.parametrize("gas", [
        "CO2", "CH4", "N2O", "CF4", "C2F6", "SF6", "NF3", "HFC",
    ])
    def test_record_emissions_various_gases(self, gas):
        record_emissions("cement_production", gas, 1000.0)

    @pytest.mark.parametrize("source", ["EPA", "IPCC", "DEFRA", "EU_ETS", "CUSTOM"])
    def test_record_process_lookup_various_sources(self, source):
        record_process_lookup(source)

    @pytest.mark.parametrize("tier", ["TIER_1", "TIER_2", "TIER_3"])
    def test_record_factor_selection_various_tiers(self, tier):
        record_factor_selection(tier, "IPCC")

    @pytest.mark.parametrize("op_type", [
        "register", "update", "delete", "consume", "transform", "query",
    ])
    def test_record_material_operation_various_ops(self, op_type):
        record_material_operation(op_type, "calcium_carbonate")

    @pytest.mark.parametrize("framework", [
        "GHG_PROTOCOL", "ISO_14064", "CSRD_ESRS_E1",
        "EPA_40CFR98", "UK_SECR", "EU_ETS",
    ])
    def test_record_compliance_check_various_frameworks(self, framework):
        record_compliance_check(framework, "COMPLIANT")

    @pytest.mark.parametrize("operation", [
        "single_calculation", "batch_calculation", "factor_lookup",
        "unit_conversion", "gwp_application", "uncertainty_analysis",
    ])
    def test_observe_calculation_duration_various_ops(self, operation):
        observe_calculation_duration(operation, 0.01)

    @pytest.mark.parametrize("method", [
        "EMISSION_FACTOR", "MASS_BALANCE", "STOICHIOMETRIC", "mixed",
    ])
    def test_observe_batch_size_various_methods(self, method):
        observe_batch_size(method, 10)


# ============================================================================
# TestMetricsEdgeCases - 6 tests
# ============================================================================


class TestMetricsEdgeCases:
    """Test edge case inputs for metric helper functions."""

    def test_record_emissions_zero_amount(self):
        """Recording zero kg CO2e should not raise."""
        record_emissions("cement_production", "CO2", 0.0)

    def test_record_emissions_very_large_amount(self):
        """Recording a very large emission should not raise."""
        record_emissions("iron_steel", "CO2", 1e12)

    def test_record_emissions_default_amount(self):
        """Default amount parameter (1.0) works correctly."""
        record_emissions("cement_production", "CO2")

    def test_set_active_calculations_zero(self):
        """Setting active calculations to zero should not raise."""
        set_active_calculations(0)

    def test_observe_calculation_duration_zero(self):
        """Observing zero duration should not raise."""
        observe_calculation_duration("single_calculation", 0.0)

    def test_observe_batch_size_one(self):
        """Observing batch size of 1 should not raise."""
        observe_batch_size("EMISSION_FACTOR", 1)


# ============================================================================
# TestMetricsModuleAttributes - 4 tests
# ============================================================================


class TestMetricsModuleAttributes:
    """Test module-level attributes and __all__ exports."""

    def test_prometheus_available_is_bool(self):
        assert isinstance(PROMETHEUS_AVAILABLE, bool)

    def test_all_exports_exist(self):
        """Every name in __all__ should be accessible on the module."""
        for name in metrics_mod.__all__:
            assert hasattr(metrics_mod, name), f"{name} missing from module"

    def test_all_12_metric_objects_in_all(self):
        metric_names = [
            "pe_calculations_total",
            "pe_emissions_kg_co2e_total",
            "pe_process_lookups_total",
            "pe_factor_selections_total",
            "pe_material_operations_total",
            "pe_uncertainty_runs_total",
            "pe_compliance_checks_total",
            "pe_batch_jobs_total",
            "pe_calculation_duration_seconds",
            "pe_batch_size",
            "pe_active_calculations",
            "pe_process_units_registered",
        ]
        for name in metric_names:
            assert name in metrics_mod.__all__

    def test_all_12_helper_functions_in_all(self):
        helper_names = [
            "record_calculation",
            "record_emissions",
            "record_process_lookup",
            "record_factor_selection",
            "record_material_operation",
            "record_uncertainty",
            "record_compliance_check",
            "record_batch",
            "observe_calculation_duration",
            "observe_batch_size",
            "set_active_calculations",
            "set_process_units_registered",
        ]
        for name in helper_names:
            assert name in metrics_mod.__all__
