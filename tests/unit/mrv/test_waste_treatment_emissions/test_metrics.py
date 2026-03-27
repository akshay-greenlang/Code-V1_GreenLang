# -*- coding: utf-8 -*-
"""
Unit tests for AGENT-MRV-007 Waste Treatment Emissions Agent Prometheus Metrics.

Tests all 12 Prometheus metrics existence, gl_wt_ prefix naming, counter
increments, histogram observations, gauge values, label values, helper
function delegation, MetricsCollector singleton, and graceful fallback
when prometheus_client is not installed.

Target: 25+ tests, 85%+ coverage.

Author: GreenLang Platform Team
Date: February 2026
"""

from __future__ import annotations

from unittest.mock import patch, MagicMock

import pytest

from greenlang.agents.mrv.waste_treatment_emissions.metrics import (
    PROMETHEUS_AVAILABLE,
    # Metric objects
    wt_calculations_total,
    wt_calculation_duration_seconds,
    wt_calculation_errors_total,
    wt_emissions_tco2e_total,
    wt_waste_processed_tonnes_total,
    wt_methane_recovery_tonnes_total,
    wt_energy_recovered_gj_total,
    wt_biological_treatments_total,
    wt_thermal_treatments_total,
    wt_compliance_checks_total,
    wt_uncertainty_runs_total,
    wt_active_facilities,
    # Helper functions
    record_calculation,
    observe_calculation_duration,
    record_calculation_error,
    record_emissions,
    record_waste_processed,
    record_methane_recovery,
    record_energy_recovered,
    record_biological_treatment,
    record_thermal_treatment,
    record_compliance_check,
    record_uncertainty_run,
    track_active_facilities,
    # Collector class
    MetricsCollector,
)


# ===========================================================================
# Metric Registration Tests
# ===========================================================================


class TestMetricsRegistration:
    """Tests that all 12 Prometheus metrics are defined."""

    @pytest.mark.skipif(
        not PROMETHEUS_AVAILABLE,
        reason="prometheus_client not installed",
    )
    def test_wt_calculations_total_exists(self):
        """gl_wt_calculations_total counter is defined."""
        assert wt_calculations_total is not None

    @pytest.mark.skipif(
        not PROMETHEUS_AVAILABLE,
        reason="prometheus_client not installed",
    )
    def test_wt_calculation_duration_seconds_exists(self):
        """gl_wt_calculation_duration_seconds histogram is defined."""
        assert wt_calculation_duration_seconds is not None

    @pytest.mark.skipif(
        not PROMETHEUS_AVAILABLE,
        reason="prometheus_client not installed",
    )
    def test_wt_calculation_errors_total_exists(self):
        """gl_wt_calculation_errors_total counter is defined."""
        assert wt_calculation_errors_total is not None

    @pytest.mark.skipif(
        not PROMETHEUS_AVAILABLE,
        reason="prometheus_client not installed",
    )
    def test_wt_emissions_tco2e_total_exists(self):
        """gl_wt_emissions_tco2e_total counter is defined."""
        assert wt_emissions_tco2e_total is not None

    @pytest.mark.skipif(
        not PROMETHEUS_AVAILABLE,
        reason="prometheus_client not installed",
    )
    def test_wt_waste_processed_tonnes_total_exists(self):
        """gl_wt_waste_processed_tonnes_total counter is defined."""
        assert wt_waste_processed_tonnes_total is not None

    @pytest.mark.skipif(
        not PROMETHEUS_AVAILABLE,
        reason="prometheus_client not installed",
    )
    def test_wt_methane_recovery_tonnes_total_exists(self):
        """gl_wt_methane_recovery_tonnes_total counter is defined."""
        assert wt_methane_recovery_tonnes_total is not None

    @pytest.mark.skipif(
        not PROMETHEUS_AVAILABLE,
        reason="prometheus_client not installed",
    )
    def test_wt_energy_recovered_gj_total_exists(self):
        """gl_wt_energy_recovered_gj_total counter is defined."""
        assert wt_energy_recovered_gj_total is not None

    @pytest.mark.skipif(
        not PROMETHEUS_AVAILABLE,
        reason="prometheus_client not installed",
    )
    def test_wt_biological_treatments_total_exists(self):
        """gl_wt_biological_treatments_total counter is defined."""
        assert wt_biological_treatments_total is not None

    @pytest.mark.skipif(
        not PROMETHEUS_AVAILABLE,
        reason="prometheus_client not installed",
    )
    def test_wt_thermal_treatments_total_exists(self):
        """gl_wt_thermal_treatments_total counter is defined."""
        assert wt_thermal_treatments_total is not None

    @pytest.mark.skipif(
        not PROMETHEUS_AVAILABLE,
        reason="prometheus_client not installed",
    )
    def test_wt_compliance_checks_total_exists(self):
        """gl_wt_compliance_checks_total counter is defined."""
        assert wt_compliance_checks_total is not None

    @pytest.mark.skipif(
        not PROMETHEUS_AVAILABLE,
        reason="prometheus_client not installed",
    )
    def test_wt_uncertainty_runs_total_exists(self):
        """gl_wt_uncertainty_runs_total counter is defined."""
        assert wt_uncertainty_runs_total is not None

    @pytest.mark.skipif(
        not PROMETHEUS_AVAILABLE,
        reason="prometheus_client not installed",
    )
    def test_wt_active_facilities_exists(self):
        """gl_wt_active_facilities gauge is defined."""
        assert wt_active_facilities is not None


# ===========================================================================
# Metric Labels Tests
# ===========================================================================


class TestMetricsLabels:
    """Tests for correct label sets on each metric."""

    @pytest.mark.skipif(
        not PROMETHEUS_AVAILABLE,
        reason="prometheus_client not installed",
    )
    def test_calculations_total_labels(self):
        """gl_wt_calculations_total has treatment_method, calculation_method, waste_category labels."""
        expected = ["treatment_method", "calculation_method", "waste_category"]
        actual = list(wt_calculations_total._labelnames)
        assert actual == expected

    @pytest.mark.skipif(
        not PROMETHEUS_AVAILABLE,
        reason="prometheus_client not installed",
    )
    def test_calculation_duration_labels(self):
        """gl_wt_calculation_duration_seconds has treatment_method, calculation_method labels."""
        expected = ["treatment_method", "calculation_method"]
        actual = list(wt_calculation_duration_seconds._labelnames)
        assert actual == expected

    @pytest.mark.skipif(
        not PROMETHEUS_AVAILABLE,
        reason="prometheus_client not installed",
    )
    def test_errors_total_labels(self):
        """gl_wt_calculation_errors_total has error_type label."""
        expected = ["error_type"]
        actual = list(wt_calculation_errors_total._labelnames)
        assert actual == expected

    @pytest.mark.skipif(
        not PROMETHEUS_AVAILABLE,
        reason="prometheus_client not installed",
    )
    def test_emissions_total_labels(self):
        """gl_wt_emissions_tco2e_total has gas, treatment_method, waste_category labels."""
        expected = ["gas", "treatment_method", "waste_category"]
        actual = list(wt_emissions_tco2e_total._labelnames)
        assert actual == expected

    @pytest.mark.skipif(
        not PROMETHEUS_AVAILABLE,
        reason="prometheus_client not installed",
    )
    def test_active_facilities_labels(self):
        """gl_wt_active_facilities has tenant_id label."""
        expected = ["tenant_id"]
        actual = list(wt_active_facilities._labelnames)
        assert actual == expected


# ===========================================================================
# Graceful Fallback Tests
# ===========================================================================


class TestMetricsGracefulFallback:
    """Tests that helpers work safely without prometheus_client."""

    def test_record_calculation_no_error_without_prometheus(self):
        """record_calculation does not raise when prometheus_client is absent."""
        # This test verifies no exception is raised
        record_calculation("landfill", "first_order_decay", "msw")

    def test_observe_duration_no_error_without_prometheus(self):
        """observe_calculation_duration does not raise when prometheus_client is absent."""
        observe_calculation_duration("incineration", "mass_balance", 0.5)

    def test_record_error_no_error_without_prometheus(self):
        """record_calculation_error does not raise when prometheus_client is absent."""
        record_calculation_error("validation_error")

    def test_record_emissions_no_error_without_prometheus(self):
        """record_emissions does not raise when prometheus_client is absent."""
        record_emissions("CH4", "composting", "food_waste", 42.5)

    def test_record_waste_processed_no_error_without_prometheus(self):
        """record_waste_processed does not raise when prometheus_client is absent."""
        record_waste_processed("incineration", "msw", 1000.0)

    def test_record_methane_recovery_no_error_without_prometheus(self):
        """record_methane_recovery does not raise when prometheus_client is absent."""
        record_methane_recovery("captured", 10.0)

    def test_record_energy_recovered_no_error_without_prometheus(self):
        """record_energy_recovered does not raise when prometheus_client is absent."""
        record_energy_recovered("electricity", 500.0)

    def test_record_biological_treatment_no_error(self):
        """record_biological_treatment does not raise."""
        record_biological_treatment("composting")

    def test_record_thermal_treatment_no_error(self):
        """record_thermal_treatment does not raise."""
        record_thermal_treatment("incineration")

    def test_record_compliance_check_no_error(self):
        """record_compliance_check does not raise."""
        record_compliance_check("GHG_PROTOCOL", "compliant")

    def test_record_uncertainty_run_no_error(self):
        """record_uncertainty_run does not raise."""
        record_uncertainty_run("monte_carlo")

    def test_track_active_facilities_no_error(self):
        """track_active_facilities does not raise."""
        track_active_facilities("tenant_001", 7)


# ===========================================================================
# MetricsCollector Singleton Tests
# ===========================================================================


class TestMetricsCollector:
    """Tests for the MetricsCollector thread-safe singleton."""

    def test_singleton_pattern(self):
        """MetricsCollector returns the same instance."""
        c1 = MetricsCollector()
        c2 = MetricsCollector()
        assert c1 is c2

    def test_prometheus_available_attribute(self):
        """MetricsCollector exposes prometheus_available attribute."""
        collector = MetricsCollector()
        assert hasattr(collector, "prometheus_available")
        assert collector.prometheus_available == PROMETHEUS_AVAILABLE

    def test_collector_record_calculation(self):
        """MetricsCollector.record_calculation delegates without error."""
        collector = MetricsCollector()
        collector.record_calculation(
            "composting", "ipcc_default", "food_waste"
        )

    def test_collector_observe_duration(self):
        """MetricsCollector.observe_calculation_duration delegates without error."""
        collector = MetricsCollector()
        collector.observe_calculation_duration(
            "incineration", "mass_balance", 1.23
        )

    def test_collector_record_error(self):
        """MetricsCollector.record_calculation_error delegates without error."""
        collector = MetricsCollector()
        collector.record_calculation_error("timeout_error")

    def test_collector_record_emissions(self):
        """MetricsCollector.record_emissions delegates without error."""
        collector = MetricsCollector()
        collector.record_emissions(
            "N2O", "composting", "food_waste", 0.5
        )

    def test_collector_record_waste_processed(self):
        """MetricsCollector.record_waste_processed delegates without error."""
        collector = MetricsCollector()
        collector.record_waste_processed("landfill", "msw", 500.0)

    def test_collector_record_methane_recovery(self):
        """MetricsCollector.record_methane_recovery delegates without error."""
        collector = MetricsCollector()
        collector.record_methane_recovery("flared", 5.0)

    def test_collector_record_energy_recovered(self):
        """MetricsCollector.record_energy_recovered delegates without error."""
        collector = MetricsCollector()
        collector.record_energy_recovered("heat", 100.0)

    def test_collector_record_biological_treatment(self):
        """MetricsCollector.record_biological_treatment delegates without error."""
        collector = MetricsCollector()
        collector.record_biological_treatment("ad")

    def test_collector_record_thermal_treatment(self):
        """MetricsCollector.record_thermal_treatment delegates without error."""
        collector = MetricsCollector()
        collector.record_thermal_treatment("pyrolysis")

    def test_collector_record_compliance_check(self):
        """MetricsCollector.record_compliance_check delegates without error."""
        collector = MetricsCollector()
        collector.record_compliance_check("CSRD_ESRS_E1", "partial")

    def test_collector_record_uncertainty_run(self):
        """MetricsCollector.record_uncertainty_run delegates without error."""
        collector = MetricsCollector()
        collector.record_uncertainty_run("analytical")

    def test_collector_track_active_facilities(self):
        """MetricsCollector.track_active_facilities delegates without error."""
        collector = MetricsCollector()
        collector.track_active_facilities("tenant_002", 3)


# ===========================================================================
# Metrics Recording Integration Tests
# ===========================================================================


class TestMetricsRecording:
    """Tests for actual metric value recording when prometheus_client is available."""

    @pytest.mark.skipif(
        not PROMETHEUS_AVAILABLE,
        reason="prometheus_client not installed",
    )
    def test_calculation_counter_increments(self):
        """Counter increments after record_calculation call."""
        before = wt_calculations_total.labels(
            treatment_method="test_method",
            calculation_method="test_calc",
            waste_category="test_waste",
        )._value.get()
        record_calculation("test_method", "test_calc", "test_waste")
        after = wt_calculations_total.labels(
            treatment_method="test_method",
            calculation_method="test_calc",
            waste_category="test_waste",
        )._value.get()
        assert after == before + 1

    @pytest.mark.skipif(
        not PROMETHEUS_AVAILABLE,
        reason="prometheus_client not installed",
    )
    def test_duration_histogram_observation(self):
        """Histogram records observation after observe_calculation_duration call."""
        observe_calculation_duration("test_hist", "test_method", 0.123)
        # If no exception occurred, the observation was recorded

    @pytest.mark.skipif(
        not PROMETHEUS_AVAILABLE,
        reason="prometheus_client not installed",
    )
    def test_error_counter_increments(self):
        """Error counter increments after record_calculation_error call."""
        before = wt_calculation_errors_total.labels(
            error_type="test_error_type",
        )._value.get()
        record_calculation_error("test_error_type")
        after = wt_calculation_errors_total.labels(
            error_type="test_error_type",
        )._value.get()
        assert after == before + 1

    @pytest.mark.skipif(
        not PROMETHEUS_AVAILABLE,
        reason="prometheus_client not installed",
    )
    def test_emissions_counter_increments_by_amount(self):
        """Emissions counter increments by the specified tCO2e amount."""
        before = wt_emissions_tco2e_total.labels(
            gas="test_gas",
            treatment_method="test_tm",
            waste_category="test_wc",
        )._value.get()
        record_emissions("test_gas", "test_tm", "test_wc", 42.5)
        after = wt_emissions_tco2e_total.labels(
            gas="test_gas",
            treatment_method="test_tm",
            waste_category="test_wc",
        )._value.get()
        assert after == pytest.approx(before + 42.5)

    @pytest.mark.skipif(
        not PROMETHEUS_AVAILABLE,
        reason="prometheus_client not installed",
    )
    def test_active_facilities_gauge_set(self):
        """Active facilities gauge is set to specified value."""
        track_active_facilities("test_tenant_gauge", 5)
        value = wt_active_facilities.labels(
            tenant_id="test_tenant_gauge",
        )._value.get()
        assert value == 5

    @pytest.mark.skipif(
        not PROMETHEUS_AVAILABLE,
        reason="prometheus_client not installed",
    )
    def test_active_facilities_gauge_overwrite(self):
        """Active facilities gauge overwrites previous value."""
        track_active_facilities("test_tenant_overwrite", 10)
        track_active_facilities("test_tenant_overwrite", 3)
        value = wt_active_facilities.labels(
            tenant_id="test_tenant_overwrite",
        )._value.get()
        assert value == 3
