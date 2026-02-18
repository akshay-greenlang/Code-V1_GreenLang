# -*- coding: utf-8 -*-
"""
Unit tests for Stationary Combustion Agent Prometheus metrics - AGENT-MRV-001.

Tests all 12 Prometheus metrics and 12 helper functions with 30+ tests.
Verifies metric objects exist, helper functions increment/observe/set
correctly, and graceful fallback when prometheus_client is unavailable.

AGENT-MRV-001: Stationary Combustion Agent (GL-MRV-SCOPE1-001)
"""

from __future__ import annotations

import pytest

from greenlang.stationary_combustion.metrics import (
    PROMETHEUS_AVAILABLE,
    # Metric objects
    sc_calculations_total,
    sc_emissions_kg_co2e_total,
    sc_fuel_lookups_total,
    sc_factor_selections_total,
    sc_equipment_profiles_total,
    sc_uncertainty_runs_total,
    sc_audit_entries_total,
    sc_batch_jobs_total,
    sc_calculation_duration_seconds,
    sc_batch_size,
    sc_active_calculations,
    sc_emission_factors_loaded,
    # Helper functions
    record_calculation,
    record_emissions,
    record_fuel_lookup,
    record_factor_selection,
    record_equipment,
    record_uncertainty,
    record_audit,
    record_batch,
    observe_calculation_duration,
    observe_batch_size,
    set_active_calculations,
    set_factors_loaded,
)


# =============================================================================
# TestPrometheusAvailability
# =============================================================================


class TestPrometheusAvailability:
    """Verify the PROMETHEUS_AVAILABLE flag is a boolean."""

    def test_prometheus_available_is_bool(self):
        assert isinstance(PROMETHEUS_AVAILABLE, bool)

    def test_prometheus_available_true_when_installed(self):
        """When prometheus_client is installed, flag should be True."""
        try:
            import prometheus_client  # noqa: F401
            assert PROMETHEUS_AVAILABLE is True
        except ImportError:
            assert PROMETHEUS_AVAILABLE is False


# =============================================================================
# TestMetricObjects - All 12 metric objects exist and are non-None
# =============================================================================


class TestMetricObjects:
    """Verify all 12 metric objects are defined (non-None when prometheus available)."""

    @pytest.mark.skipif(
        not PROMETHEUS_AVAILABLE,
        reason="prometheus_client not installed",
    )
    def test_sc_calculations_total_exists(self):
        assert sc_calculations_total is not None

    @pytest.mark.skipif(
        not PROMETHEUS_AVAILABLE,
        reason="prometheus_client not installed",
    )
    def test_sc_emissions_kg_co2e_total_exists(self):
        assert sc_emissions_kg_co2e_total is not None

    @pytest.mark.skipif(
        not PROMETHEUS_AVAILABLE,
        reason="prometheus_client not installed",
    )
    def test_sc_fuel_lookups_total_exists(self):
        assert sc_fuel_lookups_total is not None

    @pytest.mark.skipif(
        not PROMETHEUS_AVAILABLE,
        reason="prometheus_client not installed",
    )
    def test_sc_factor_selections_total_exists(self):
        assert sc_factor_selections_total is not None

    @pytest.mark.skipif(
        not PROMETHEUS_AVAILABLE,
        reason="prometheus_client not installed",
    )
    def test_sc_equipment_profiles_total_exists(self):
        assert sc_equipment_profiles_total is not None

    @pytest.mark.skipif(
        not PROMETHEUS_AVAILABLE,
        reason="prometheus_client not installed",
    )
    def test_sc_uncertainty_runs_total_exists(self):
        assert sc_uncertainty_runs_total is not None

    @pytest.mark.skipif(
        not PROMETHEUS_AVAILABLE,
        reason="prometheus_client not installed",
    )
    def test_sc_audit_entries_total_exists(self):
        assert sc_audit_entries_total is not None

    @pytest.mark.skipif(
        not PROMETHEUS_AVAILABLE,
        reason="prometheus_client not installed",
    )
    def test_sc_batch_jobs_total_exists(self):
        assert sc_batch_jobs_total is not None

    @pytest.mark.skipif(
        not PROMETHEUS_AVAILABLE,
        reason="prometheus_client not installed",
    )
    def test_sc_calculation_duration_seconds_exists(self):
        assert sc_calculation_duration_seconds is not None

    @pytest.mark.skipif(
        not PROMETHEUS_AVAILABLE,
        reason="prometheus_client not installed",
    )
    def test_sc_batch_size_exists(self):
        assert sc_batch_size is not None

    @pytest.mark.skipif(
        not PROMETHEUS_AVAILABLE,
        reason="prometheus_client not installed",
    )
    def test_sc_active_calculations_exists(self):
        assert sc_active_calculations is not None

    @pytest.mark.skipif(
        not PROMETHEUS_AVAILABLE,
        reason="prometheus_client not installed",
    )
    def test_sc_emission_factors_loaded_exists(self):
        assert sc_emission_factors_loaded is not None


# =============================================================================
# TestRecordCalculation - Counter increment via record_calculation()
# =============================================================================


class TestRecordCalculation:
    """Test record_calculation helper function."""

    def test_record_calculation_does_not_raise(self):
        """Calling record_calculation should never raise, even without prometheus."""
        record_calculation("natural_gas", "TIER_1", "completed")

    @pytest.mark.skipif(
        not PROMETHEUS_AVAILABLE,
        reason="prometheus_client not installed",
    )
    def test_record_calculation_increments_counter(self):
        """After calling record_calculation, the labelled counter increases."""
        before = sc_calculations_total.labels(
            fuel_type="natural_gas", tier="TIER_1", status="completed"
        )._value.get()
        record_calculation("natural_gas", "TIER_1", "completed")
        after = sc_calculations_total.labels(
            fuel_type="natural_gas", tier="TIER_1", status="completed"
        )._value.get()
        assert after == before + 1

    def test_record_calculation_various_labels(self):
        """Call with different label combinations to verify no errors."""
        record_calculation("diesel", "TIER_2", "failed")
        record_calculation("coal_bituminous", "TIER_3", "pending")
        record_calculation("wood", "TIER_1", "running")


# =============================================================================
# TestRecordEmissions - Counter increment via record_emissions()
# =============================================================================


class TestRecordEmissions:
    """Test record_emissions helper function."""

    def test_record_emissions_does_not_raise(self):
        record_emissions("natural_gas", "CO2", 1500.0)

    @pytest.mark.skipif(
        not PROMETHEUS_AVAILABLE,
        reason="prometheus_client not installed",
    )
    def test_record_emissions_increments_by_amount(self):
        before = sc_emissions_kg_co2e_total.labels(
            fuel_type="diesel", gas="CO2"
        )._value.get()
        record_emissions("diesel", "CO2", 500.0)
        after = sc_emissions_kg_co2e_total.labels(
            fuel_type="diesel", gas="CO2"
        )._value.get()
        assert after == pytest.approx(before + 500.0)

    def test_record_emissions_default_amount(self):
        """Default kg_co2e is 1.0 and should not raise."""
        record_emissions("natural_gas", "CH4")


# =============================================================================
# TestRecordFuelLookup
# =============================================================================


class TestRecordFuelLookup:
    """Test record_fuel_lookup helper function."""

    def test_record_fuel_lookup_does_not_raise(self):
        record_fuel_lookup("EPA")

    @pytest.mark.skipif(
        not PROMETHEUS_AVAILABLE,
        reason="prometheus_client not installed",
    )
    def test_record_fuel_lookup_increments(self):
        before = sc_fuel_lookups_total.labels(source="IPCC")._value.get()
        record_fuel_lookup("IPCC")
        after = sc_fuel_lookups_total.labels(source="IPCC")._value.get()
        assert after == before + 1

    def test_record_fuel_lookup_all_sources(self):
        for source in ("EPA", "IPCC", "DEFRA", "EU_ETS", "CUSTOM"):
            record_fuel_lookup(source)


# =============================================================================
# TestRecordFactorSelection
# =============================================================================


class TestRecordFactorSelection:
    """Test record_factor_selection helper function."""

    def test_record_factor_selection_does_not_raise(self):
        record_factor_selection("TIER_1", "EPA")

    @pytest.mark.skipif(
        not PROMETHEUS_AVAILABLE,
        reason="prometheus_client not installed",
    )
    def test_record_factor_selection_increments(self):
        before = sc_factor_selections_total.labels(
            tier="TIER_2", source="DEFRA"
        )._value.get()
        record_factor_selection("TIER_2", "DEFRA")
        after = sc_factor_selections_total.labels(
            tier="TIER_2", source="DEFRA"
        )._value.get()
        assert after == before + 1


# =============================================================================
# TestRecordEquipment
# =============================================================================


class TestRecordEquipment:
    """Test record_equipment helper function."""

    def test_record_equipment_does_not_raise(self):
        record_equipment("boiler_fire_tube", "register")

    @pytest.mark.skipif(
        not PROMETHEUS_AVAILABLE,
        reason="prometheus_client not installed",
    )
    def test_record_equipment_increments(self):
        before = sc_equipment_profiles_total.labels(
            equipment_type="furnace", action="update"
        )._value.get()
        record_equipment("furnace", "update")
        after = sc_equipment_profiles_total.labels(
            equipment_type="furnace", action="update"
        )._value.get()
        assert after == before + 1


# =============================================================================
# TestRecordUncertainty
# =============================================================================


class TestRecordUncertainty:
    """Test record_uncertainty helper function."""

    def test_record_uncertainty_does_not_raise(self):
        record_uncertainty("TIER_1", "monte_carlo")

    @pytest.mark.skipif(
        not PROMETHEUS_AVAILABLE,
        reason="prometheus_client not installed",
    )
    def test_record_uncertainty_increments(self):
        before = sc_uncertainty_runs_total.labels(
            tier="TIER_1", method="analytical"
        )._value.get()
        record_uncertainty("TIER_1", "analytical")
        after = sc_uncertainty_runs_total.labels(
            tier="TIER_1", method="analytical"
        )._value.get()
        assert after == before + 1


# =============================================================================
# TestRecordAudit
# =============================================================================


class TestRecordAudit:
    """Test record_audit helper function."""

    def test_record_audit_does_not_raise(self):
        record_audit("validate_input")

    @pytest.mark.skipif(
        not PROMETHEUS_AVAILABLE,
        reason="prometheus_client not installed",
    )
    def test_record_audit_increments(self):
        before = sc_audit_entries_total.labels(
            step_name="calculate_energy"
        )._value.get()
        record_audit("calculate_energy")
        after = sc_audit_entries_total.labels(
            step_name="calculate_energy"
        )._value.get()
        assert after == before + 1


# =============================================================================
# TestRecordBatch
# =============================================================================


class TestRecordBatch:
    """Test record_batch helper function."""

    def test_record_batch_does_not_raise(self):
        record_batch("completed")

    @pytest.mark.skipif(
        not PROMETHEUS_AVAILABLE,
        reason="prometheus_client not installed",
    )
    def test_record_batch_increments(self):
        before = sc_batch_jobs_total.labels(status="failed")._value.get()
        record_batch("failed")
        after = sc_batch_jobs_total.labels(status="failed")._value.get()
        assert after == before + 1


# =============================================================================
# TestObserveCalculationDuration - Histogram observation
# =============================================================================


class TestObserveCalculationDuration:
    """Test observe_calculation_duration helper function."""

    def test_observe_calculation_duration_does_not_raise(self):
        observe_calculation_duration("single_calculation", 0.005)

    @pytest.mark.skipif(
        not PROMETHEUS_AVAILABLE,
        reason="prometheus_client not installed",
    )
    def test_observe_calculation_duration_records(self):
        """Observation count should increase after observe call."""
        metric = sc_calculation_duration_seconds.labels(
            operation="factor_lookup"
        )
        before_count = metric._sum.get()
        observe_calculation_duration("factor_lookup", 0.01)
        after_count = metric._sum.get()
        assert after_count >= before_count + 0.01

    def test_observe_various_operations(self):
        for op in (
            "single_calculation", "batch_calculation", "factor_lookup",
            "unit_conversion", "gwp_application", "uncertainty_analysis",
            "provenance_hash", "audit_generation",
        ):
            observe_calculation_duration(op, 0.001)


# =============================================================================
# TestObserveBatchSize - Histogram observation
# =============================================================================


class TestObserveBatchSize:
    """Test observe_batch_size helper function."""

    def test_observe_batch_size_does_not_raise(self):
        observe_batch_size("natural_gas", 100)

    @pytest.mark.skipif(
        not PROMETHEUS_AVAILABLE,
        reason="prometheus_client not installed",
    )
    def test_observe_batch_size_records(self):
        metric = sc_batch_size.labels(fuel_type="mixed")
        before_count = metric._sum.get()
        observe_batch_size("mixed", 50)
        after_count = metric._sum.get()
        assert after_count >= before_count + 50


# =============================================================================
# TestSetActiveCalculations - Gauge set
# =============================================================================


class TestSetActiveCalculations:
    """Test set_active_calculations helper function."""

    def test_set_active_calculations_does_not_raise(self):
        set_active_calculations(5)

    @pytest.mark.skipif(
        not PROMETHEUS_AVAILABLE,
        reason="prometheus_client not installed",
    )
    def test_set_active_calculations_sets_value(self):
        set_active_calculations(42)
        val = sc_active_calculations._value.get()
        assert val == 42.0

    @pytest.mark.skipif(
        not PROMETHEUS_AVAILABLE,
        reason="prometheus_client not installed",
    )
    def test_set_active_calculations_zero(self):
        set_active_calculations(0)
        val = sc_active_calculations._value.get()
        assert val == 0.0


# =============================================================================
# TestSetFactorsLoaded - Gauge set with labels
# =============================================================================


class TestSetFactorsLoaded:
    """Test set_factors_loaded helper function."""

    def test_set_factors_loaded_does_not_raise(self):
        set_factors_loaded("EPA", 100)

    @pytest.mark.skipif(
        not PROMETHEUS_AVAILABLE,
        reason="prometheus_client not installed",
    )
    def test_set_factors_loaded_sets_value(self):
        set_factors_loaded("IPCC", 250)
        val = sc_emission_factors_loaded.labels(source="IPCC")._value.get()
        assert val == 250.0

    @pytest.mark.skipif(
        not PROMETHEUS_AVAILABLE,
        reason="prometheus_client not installed",
    )
    def test_set_factors_loaded_overwrites(self):
        set_factors_loaded("EPA", 100)
        set_factors_loaded("EPA", 200)
        val = sc_emission_factors_loaded.labels(source="EPA")._value.get()
        assert val == 200.0
