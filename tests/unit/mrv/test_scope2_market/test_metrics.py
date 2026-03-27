# -*- coding: utf-8 -*-
"""
Unit tests for AGENT-MRV-010 Scope 2 Market-Based Emissions Agent Metrics.

Tests all 12 Prometheus metrics existence, gl_s2m_ prefix naming,
Scope2MarketMetrics singleton pattern, recording methods, graceful
fallback when prometheus_client is not installed, get_metrics_summary(),
and module-level convenience functions.

Target: 60+ tests, 85%+ coverage of metrics.py.

Author: GreenLang Platform Team
Date: February 2026
"""

from __future__ import annotations

from unittest.mock import patch, MagicMock

import pytest

# ---------------------------------------------------------------------------
# Conditional import guard
# ---------------------------------------------------------------------------

try:
    from greenlang.agents.mrv.scope2_market.metrics import (
        PROMETHEUS_AVAILABLE,
        Scope2MarketMetrics,
        get_metrics,
        record_calculation,
        record_instrument_registered,
        record_instrument_retired,
        set_coverage_percentage,
        record_compliance_check,
        record_uncertainty_run,
        record_error,
        record_dual_report,
        record_residual_mix_lookup,
        set_active_instruments,
        record_emissions,
    )
    METRICS_AVAILABLE = True
except ImportError:
    METRICS_AVAILABLE = False

_SKIP = pytest.mark.skipif(not METRICS_AVAILABLE, reason="metrics not available")
_SKIP_NO_PROM = pytest.mark.skipif(
    METRICS_AVAILABLE and not PROMETHEUS_AVAILABLE if METRICS_AVAILABLE else True,
    reason="prometheus_client not installed",
)


# ===========================================================================
# Singleton Pattern Tests
# ===========================================================================


@_SKIP
class TestMetricsSingleton:
    """Tests for the Scope2MarketMetrics singleton pattern."""

    @pytest.mark.skipif(
        not METRICS_AVAILABLE or not PROMETHEUS_AVAILABLE,
        reason="prometheus_client not installed",
    )
    def test_singleton_returns_same_instance(self):
        """Two instantiations return the same object."""
        from prometheus_client import CollectorRegistry
        registry = CollectorRegistry()
        Scope2MarketMetrics._reset()
        m1 = Scope2MarketMetrics(registry=registry)
        m2 = Scope2MarketMetrics(registry=registry)
        assert m1 is m2

    @pytest.mark.skipif(
        not METRICS_AVAILABLE or not PROMETHEUS_AVAILABLE,
        reason="prometheus_client not installed",
    )
    def test_reset_creates_new_instance(self):
        """After _reset(), a new instance is created."""
        from prometheus_client import CollectorRegistry
        reg1 = CollectorRegistry()
        Scope2MarketMetrics._reset()
        m1 = Scope2MarketMetrics(registry=reg1)
        id1 = id(m1)
        Scope2MarketMetrics._reset()
        reg2 = CollectorRegistry()
        m2 = Scope2MarketMetrics(registry=reg2)
        assert id1 != id(m2)

    @pytest.mark.skipif(
        not METRICS_AVAILABLE or not PROMETHEUS_AVAILABLE,
        reason="prometheus_client not installed",
    )
    def test_prometheus_available_flag(self):
        """prometheus_available attribute reflects library availability."""
        from prometheus_client import CollectorRegistry
        registry = CollectorRegistry()
        Scope2MarketMetrics._reset()
        m = Scope2MarketMetrics(registry=registry)
        assert m.prometheus_available == PROMETHEUS_AVAILABLE

    @pytest.mark.skipif(
        not METRICS_AVAILABLE or not PROMETHEUS_AVAILABLE,
        reason="prometheus_client not installed",
    )
    def test_initialized_flag_set_after_init(self):
        """_initialized is True after singleton construction."""
        from prometheus_client import CollectorRegistry
        registry = CollectorRegistry()
        Scope2MarketMetrics._reset()
        m = Scope2MarketMetrics(registry=registry)
        assert m._initialized is True

    @pytest.mark.skipif(
        not METRICS_AVAILABLE or not PROMETHEUS_AVAILABLE,
        reason="prometheus_client not installed",
    )
    def test_registry_stored(self):
        """Custom registry is stored on the instance."""
        from prometheus_client import CollectorRegistry
        registry = CollectorRegistry()
        Scope2MarketMetrics._reset()
        m = Scope2MarketMetrics(registry=registry)
        assert m._registry is registry


# ===========================================================================
# Metric Existence Tests (require prometheus_client)
# ===========================================================================


@_SKIP
class TestMetricExistence:
    """Tests that all 12 Prometheus metrics are defined when available."""

    @pytest.mark.skipif(
        not METRICS_AVAILABLE or not PROMETHEUS_AVAILABLE,
        reason="prometheus_client not installed",
    )
    def test_calculations_total_exists(self):
        """gl_s2m_calculations_total counter is defined."""
        from prometheus_client import CollectorRegistry
        registry = CollectorRegistry()
        Scope2MarketMetrics._reset()
        m = Scope2MarketMetrics(registry=registry)
        assert m.calculations_total is not None

    @pytest.mark.skipif(
        not METRICS_AVAILABLE or not PROMETHEUS_AVAILABLE,
        reason="prometheus_client not installed",
    )
    def test_calculation_duration_seconds_exists(self):
        """gl_s2m_calculation_duration_seconds histogram is defined."""
        from prometheus_client import CollectorRegistry
        registry = CollectorRegistry()
        Scope2MarketMetrics._reset()
        m = Scope2MarketMetrics(registry=registry)
        assert m.calculation_duration_seconds is not None

    @pytest.mark.skipif(
        not METRICS_AVAILABLE or not PROMETHEUS_AVAILABLE,
        reason="prometheus_client not installed",
    )
    def test_emissions_co2e_tonnes_exists(self):
        """gl_s2m_emissions_co2e_tonnes counter is defined."""
        from prometheus_client import CollectorRegistry
        registry = CollectorRegistry()
        Scope2MarketMetrics._reset()
        m = Scope2MarketMetrics(registry=registry)
        assert m.emissions_co2e_tonnes is not None

    @pytest.mark.skipif(
        not METRICS_AVAILABLE or not PROMETHEUS_AVAILABLE,
        reason="prometheus_client not installed",
    )
    def test_instruments_registered_total_exists(self):
        """gl_s2m_instruments_registered_total counter is defined."""
        from prometheus_client import CollectorRegistry
        registry = CollectorRegistry()
        Scope2MarketMetrics._reset()
        m = Scope2MarketMetrics(registry=registry)
        assert m.instruments_registered_total is not None

    @pytest.mark.skipif(
        not METRICS_AVAILABLE or not PROMETHEUS_AVAILABLE,
        reason="prometheus_client not installed",
    )
    def test_instruments_retired_total_exists(self):
        """gl_s2m_instruments_retired_total counter is defined."""
        from prometheus_client import CollectorRegistry
        registry = CollectorRegistry()
        Scope2MarketMetrics._reset()
        m = Scope2MarketMetrics(registry=registry)
        assert m.instruments_retired_total is not None

    @pytest.mark.skipif(
        not METRICS_AVAILABLE or not PROMETHEUS_AVAILABLE,
        reason="prometheus_client not installed",
    )
    def test_coverage_percentage_exists(self):
        """gl_s2m_coverage_percentage gauge is defined."""
        from prometheus_client import CollectorRegistry
        registry = CollectorRegistry()
        Scope2MarketMetrics._reset()
        m = Scope2MarketMetrics(registry=registry)
        assert m.coverage_percentage is not None

    @pytest.mark.skipif(
        not METRICS_AVAILABLE or not PROMETHEUS_AVAILABLE,
        reason="prometheus_client not installed",
    )
    def test_compliance_checks_total_exists(self):
        """gl_s2m_compliance_checks_total counter is defined."""
        from prometheus_client import CollectorRegistry
        registry = CollectorRegistry()
        Scope2MarketMetrics._reset()
        m = Scope2MarketMetrics(registry=registry)
        assert m.compliance_checks_total is not None

    @pytest.mark.skipif(
        not METRICS_AVAILABLE or not PROMETHEUS_AVAILABLE,
        reason="prometheus_client not installed",
    )
    def test_uncertainty_runs_total_exists(self):
        """gl_s2m_uncertainty_runs_total counter is defined."""
        from prometheus_client import CollectorRegistry
        registry = CollectorRegistry()
        Scope2MarketMetrics._reset()
        m = Scope2MarketMetrics(registry=registry)
        assert m.uncertainty_runs_total is not None

    @pytest.mark.skipif(
        not METRICS_AVAILABLE or not PROMETHEUS_AVAILABLE,
        reason="prometheus_client not installed",
    )
    def test_errors_total_exists(self):
        """gl_s2m_errors_total counter is defined."""
        from prometheus_client import CollectorRegistry
        registry = CollectorRegistry()
        Scope2MarketMetrics._reset()
        m = Scope2MarketMetrics(registry=registry)
        assert m.errors_total is not None

    @pytest.mark.skipif(
        not METRICS_AVAILABLE or not PROMETHEUS_AVAILABLE,
        reason="prometheus_client not installed",
    )
    def test_dual_reports_total_exists(self):
        """gl_s2m_dual_reports_total counter is defined."""
        from prometheus_client import CollectorRegistry
        registry = CollectorRegistry()
        Scope2MarketMetrics._reset()
        m = Scope2MarketMetrics(registry=registry)
        assert m.dual_reports_total is not None

    @pytest.mark.skipif(
        not METRICS_AVAILABLE or not PROMETHEUS_AVAILABLE,
        reason="prometheus_client not installed",
    )
    def test_residual_mix_lookups_total_exists(self):
        """gl_s2m_residual_mix_lookups_total counter is defined."""
        from prometheus_client import CollectorRegistry
        registry = CollectorRegistry()
        Scope2MarketMetrics._reset()
        m = Scope2MarketMetrics(registry=registry)
        assert m.residual_mix_lookups_total is not None

    @pytest.mark.skipif(
        not METRICS_AVAILABLE or not PROMETHEUS_AVAILABLE,
        reason="prometheus_client not installed",
    )
    def test_active_instruments_exists(self):
        """gl_s2m_active_instruments gauge is defined."""
        from prometheus_client import CollectorRegistry
        registry = CollectorRegistry()
        Scope2MarketMetrics._reset()
        m = Scope2MarketMetrics(registry=registry)
        assert m.active_instruments is not None


# ===========================================================================
# Recording Method Tests (no-op when prometheus not available)
# ===========================================================================


@_SKIP
class TestRecordingMethods:
    """Tests for metric recording methods -- must not raise even without prometheus."""

    @pytest.fixture(autouse=True)
    def _create_metrics(self):
        """Create a metrics instance with a custom registry for each test."""
        Scope2MarketMetrics._reset()
        if PROMETHEUS_AVAILABLE:
            from prometheus_client import CollectorRegistry
            registry = CollectorRegistry()
            self.metrics = Scope2MarketMetrics(registry=registry)
        else:
            self.metrics = Scope2MarketMetrics()
        yield
        Scope2MarketMetrics._reset()

    def test_record_calculation_no_error(self):
        """record_calculation does not raise regardless of prometheus."""
        self.metrics.record_calculation("rec", "contractual", 0.05, 150.3)

    def test_record_calculation_different_instruments(self):
        """record_calculation works with various instrument types."""
        self.metrics.record_calculation("ppa", "power_purchase_agreement", 0.065, 45.0)
        self.metrics.record_calculation("go", "contractual", 0.032, 80.2)
        self.metrics.record_calculation("residual_mix", "residual_mix", 0.12, 210.0)

    def test_record_instrument_registered_no_error(self):
        """record_instrument_registered does not raise."""
        self.metrics.record_instrument_registered("eac", "active")

    def test_record_instrument_registered_all_statuses(self):
        """record_instrument_registered accepts all status values."""
        for status in ("active", "pending", "verified", "expired", "cancelled"):
            self.metrics.record_instrument_registered("rec", status)

    def test_record_instrument_retired_no_error(self):
        """record_instrument_retired does not raise."""
        self.metrics.record_instrument_retired("rec")

    def test_record_instrument_retired_various_types(self):
        """record_instrument_retired works with various instrument types."""
        for inst_type in ("rec", "go", "ppa", "i_rec", "rego"):
            self.metrics.record_instrument_retired(inst_type)

    def test_set_coverage_percentage_no_error(self):
        """set_coverage_percentage does not raise."""
        self.metrics.set_coverage_percentage("fac_001", 75.5)

    def test_set_coverage_percentage_boundaries(self):
        """set_coverage_percentage accepts boundary values."""
        self.metrics.set_coverage_percentage("fac_001", 0.0)
        self.metrics.set_coverage_percentage("fac_001", 100.0)

    def test_record_compliance_check_no_error(self):
        """record_compliance_check does not raise."""
        self.metrics.record_compliance_check("GHG_PROTOCOL", "compliant")

    def test_record_compliance_check_all_statuses(self):
        """record_compliance_check accepts all status values."""
        for status in ("compliant", "non_compliant", "partial", "not_assessed"):
            self.metrics.record_compliance_check("ISO_14064", status)

    def test_record_uncertainty_run_no_error(self):
        """record_uncertainty_run does not raise."""
        self.metrics.record_uncertainty_run("monte_carlo")

    def test_record_uncertainty_run_all_methods(self):
        """record_uncertainty_run accepts all method values."""
        for method in ("monte_carlo", "analytical", "error_propagation",
                       "ipcc_default_uncertainty", "bootstrap"):
            self.metrics.record_uncertainty_run(method)

    def test_record_error_no_error(self):
        """record_error does not raise."""
        self.metrics.record_error("validation_error")

    def test_record_error_all_types(self):
        """record_error accepts all error types."""
        for err_type in ("validation_error", "calculation_error",
                         "database_error", "configuration_error",
                         "timeout_error", "instrument_error",
                         "coverage_error", "unit_conversion_error",
                         "unknown_error"):
            self.metrics.record_error(err_type)

    def test_record_dual_report_no_error(self):
        """record_dual_report does not raise."""
        self.metrics.record_dual_report("complete")

    def test_record_dual_report_all_statuses(self):
        """record_dual_report accepts all dual report statuses."""
        for status in ("complete", "partial", "location_only",
                       "market_only", "reconciled"):
            self.metrics.record_dual_report(status)

    def test_record_residual_mix_lookup_no_error(self):
        """record_residual_mix_lookup does not raise."""
        self.metrics.record_residual_mix_lookup("AIB_RESIDUAL_MIX")

    def test_record_residual_mix_lookup_all_sources(self):
        """record_residual_mix_lookup accepts all source values."""
        for source in ("AIB_RESIDUAL_MIX", "GREEN_E", "RE_DISS",
                       "NATIONAL_REGISTRY", "SUPPLIER_SPECIFIC",
                       "IEA", "DEFRA", "CUSTOM"):
            self.metrics.record_residual_mix_lookup(source)

    def test_set_active_instruments_no_error(self):
        """set_active_instruments does not raise."""
        self.metrics.set_active_instruments("rec", 42)

    def test_set_active_instruments_zero(self):
        """set_active_instruments accepts zero count."""
        self.metrics.set_active_instruments("go", 0)

    def test_record_emissions_no_error(self):
        """record_emissions does not raise."""
        self.metrics.record_emissions("rec", "CO2", 150.3)

    def test_record_emissions_all_gases(self):
        """record_emissions accepts all greenhouse gas types."""
        for gas in ("CO2", "CH4", "N2O", "CO2e"):
            self.metrics.record_emissions("ppa", gas, 10.5)


# ===========================================================================
# Metrics Summary Tests
# ===========================================================================


@_SKIP
class TestMetricsSummary:
    """Tests for get_metrics_summary()."""

    @pytest.fixture(autouse=True)
    def _create_metrics(self):
        """Create a metrics instance with a custom registry for each test."""
        Scope2MarketMetrics._reset()
        if PROMETHEUS_AVAILABLE:
            from prometheus_client import CollectorRegistry
            registry = CollectorRegistry()
            self.metrics = Scope2MarketMetrics(registry=registry)
        else:
            self.metrics = Scope2MarketMetrics()
        yield
        Scope2MarketMetrics._reset()

    def test_summary_returns_dict(self):
        """get_metrics_summary() returns a dictionary."""
        summary = self.metrics.get_metrics_summary()
        assert isinstance(summary, dict)

    def test_summary_has_all_12_metric_keys(self):
        """Summary contains all 12 metric keys plus prometheus_available."""
        summary = self.metrics.get_metrics_summary()
        expected_keys = [
            "gl_s2m_calculations_total",
            "gl_s2m_calculation_duration_seconds",
            "gl_s2m_emissions_co2e_tonnes",
            "gl_s2m_instruments_registered_total",
            "gl_s2m_instruments_retired_total",
            "gl_s2m_coverage_percentage",
            "gl_s2m_compliance_checks_total",
            "gl_s2m_uncertainty_runs_total",
            "gl_s2m_errors_total",
            "gl_s2m_dual_reports_total",
            "gl_s2m_residual_mix_lookups_total",
            "gl_s2m_active_instruments",
            "prometheus_available",
        ]
        for key in expected_keys:
            assert key in summary, f"Key '{key}' missing from summary"

    def test_summary_prometheus_available_flag(self):
        """Summary contains correct prometheus_available flag."""
        summary = self.metrics.get_metrics_summary()
        assert summary["prometheus_available"] == PROMETHEUS_AVAILABLE

    def test_summary_has_13_keys(self):
        """Summary contains exactly 13 keys (12 metrics + prometheus_available)."""
        summary = self.metrics.get_metrics_summary()
        assert len(summary) == 13

    @pytest.mark.skipif(
        not METRICS_AVAILABLE or not PROMETHEUS_AVAILABLE,
        reason="prometheus_client not installed",
    )
    def test_summary_values_are_dicts_when_available(self):
        """When prometheus is available, metric values are description dicts."""
        summary = self.metrics.get_metrics_summary()
        for key, value in summary.items():
            if key == "prometheus_available":
                continue
            assert isinstance(value, dict), f"Value for '{key}' is not a dict"
            assert "name" in value, f"'name' missing from value for '{key}'"
            assert "documentation" in value, f"'documentation' missing for '{key}'"


# ===========================================================================
# Graceful Fallback Tests (no prometheus_client)
# ===========================================================================


@_SKIP
class TestGracefulFallback:
    """Tests for graceful fallback when prometheus_client is not installed."""

    def test_noop_metrics_are_none_when_no_prometheus(self):
        """When prometheus is unavailable, all metric attributes are None."""
        Scope2MarketMetrics._reset()
        m = Scope2MarketMetrics.__new__(Scope2MarketMetrics)
        m._initialized = False
        m.prometheus_available = False
        m._registry = None
        m._init_noop_metrics()
        m._initialized = True

        assert m.calculations_total is None
        assert m.calculation_duration_seconds is None
        assert m.emissions_co2e_tonnes is None
        assert m.instruments_registered_total is None
        assert m.instruments_retired_total is None
        assert m.coverage_percentage is None
        assert m.compliance_checks_total is None
        assert m.uncertainty_runs_total is None
        assert m.errors_total is None
        assert m.dual_reports_total is None
        assert m.residual_mix_lookups_total is None
        assert m.active_instruments is None

    def test_noop_summary_returns_none_values(self):
        """When prometheus unavailable, summary has None values."""
        Scope2MarketMetrics._reset()
        m = Scope2MarketMetrics.__new__(Scope2MarketMetrics)
        m._initialized = False
        m.prometheus_available = False
        m._registry = None
        m._init_noop_metrics()
        m._initialized = True

        summary = m.get_metrics_summary()
        assert summary["prometheus_available"] is False
        for key in summary:
            if key != "prometheus_available":
                assert summary[key] is None

    def test_noop_record_calculation_does_not_raise(self):
        """record_calculation does not raise when prometheus unavailable."""
        Scope2MarketMetrics._reset()
        m = Scope2MarketMetrics.__new__(Scope2MarketMetrics)
        m._initialized = False
        m.prometheus_available = False
        m._registry = None
        m._init_noop_metrics()
        m._initialized = True

        # Should not raise
        m.record_calculation("rec", "contractual", 0.05, 100.0)

    def test_noop_record_instrument_registered_does_not_raise(self):
        """record_instrument_registered does not raise when prometheus unavailable."""
        Scope2MarketMetrics._reset()
        m = Scope2MarketMetrics.__new__(Scope2MarketMetrics)
        m._initialized = False
        m.prometheus_available = False
        m._registry = None
        m._init_noop_metrics()
        m._initialized = True

        m.record_instrument_registered("eac", "active")

    def test_noop_set_coverage_percentage_does_not_raise(self):
        """set_coverage_percentage does not raise when prometheus unavailable."""
        Scope2MarketMetrics._reset()
        m = Scope2MarketMetrics.__new__(Scope2MarketMetrics)
        m._initialized = False
        m.prometheus_available = False
        m._registry = None
        m._init_noop_metrics()
        m._initialized = True

        m.set_coverage_percentage("fac_001", 50.0)

    def test_noop_record_dual_report_does_not_raise(self):
        """record_dual_report does not raise when prometheus unavailable."""
        Scope2MarketMetrics._reset()
        m = Scope2MarketMetrics.__new__(Scope2MarketMetrics)
        m._initialized = False
        m.prometheus_available = False
        m._registry = None
        m._init_noop_metrics()
        m._initialized = True

        m.record_dual_report("complete")

    def test_noop_record_residual_mix_lookup_does_not_raise(self):
        """record_residual_mix_lookup does not raise when prometheus unavailable."""
        Scope2MarketMetrics._reset()
        m = Scope2MarketMetrics.__new__(Scope2MarketMetrics)
        m._initialized = False
        m.prometheus_available = False
        m._registry = None
        m._init_noop_metrics()
        m._initialized = True

        m.record_residual_mix_lookup("AIB_RESIDUAL_MIX")

    def test_noop_set_active_instruments_does_not_raise(self):
        """set_active_instruments does not raise when prometheus unavailable."""
        Scope2MarketMetrics._reset()
        m = Scope2MarketMetrics.__new__(Scope2MarketMetrics)
        m._initialized = False
        m.prometheus_available = False
        m._registry = None
        m._init_noop_metrics()
        m._initialized = True

        m.set_active_instruments("rec", 10)

    def test_noop_record_emissions_does_not_raise(self):
        """record_emissions does not raise when prometheus unavailable."""
        Scope2MarketMetrics._reset()
        m = Scope2MarketMetrics.__new__(Scope2MarketMetrics)
        m._initialized = False
        m.prometheus_available = False
        m._registry = None
        m._init_noop_metrics()
        m._initialized = True

        m.record_emissions("go", "CO2", 50.0)


# ===========================================================================
# Module-Level Convenience Function Tests
# ===========================================================================


@_SKIP
class TestModuleLevelFunctions:
    """Tests for module-level convenience functions."""

    @pytest.fixture(autouse=True)
    def _setup_custom_registry(self):
        """Set up a custom registry for module-level function tests.

        The module-level functions (record_calculation, etc.) delegate to
        get_metrics() which returns the singleton. To avoid polluting
        the default prometheus registry, we reset and create with a
        custom registry, then wire the module-level _default_metrics.
        """
        import greenlang.agents.mrv.scope2_market.metrics as _m
        _m.Scope2MarketMetrics._reset()
        _m._default_metrics = None
        if PROMETHEUS_AVAILABLE:
            from prometheus_client import CollectorRegistry
            registry = CollectorRegistry()
            instance = Scope2MarketMetrics(registry=registry)
        else:
            instance = Scope2MarketMetrics()
        _m._default_metrics = instance
        yield
        _m.Scope2MarketMetrics._reset()
        _m._default_metrics = None

    def test_get_metrics_returns_singleton(self):
        """get_metrics() returns a Scope2MarketMetrics instance."""
        m = get_metrics()
        assert isinstance(m, Scope2MarketMetrics)

    def test_module_record_calculation_no_error(self):
        """Module-level record_calculation does not raise."""
        record_calculation("rec", "contractual", 0.05, 150.3)

    def test_module_record_instrument_registered_no_error(self):
        """Module-level record_instrument_registered does not raise."""
        record_instrument_registered("eac", "active")

    def test_module_record_instrument_retired_no_error(self):
        """Module-level record_instrument_retired does not raise."""
        record_instrument_retired("rec")

    def test_module_set_coverage_percentage_no_error(self):
        """Module-level set_coverage_percentage does not raise."""
        set_coverage_percentage("fac_001", 75.0)

    def test_module_record_compliance_check_no_error(self):
        """Module-level record_compliance_check does not raise."""
        record_compliance_check("GHG_PROTOCOL", "compliant")

    def test_module_record_uncertainty_run_no_error(self):
        """Module-level record_uncertainty_run does not raise."""
        record_uncertainty_run("monte_carlo")

    def test_module_record_error_no_error(self):
        """Module-level record_error does not raise."""
        record_error("validation_error")

    def test_module_record_dual_report_no_error(self):
        """Module-level record_dual_report does not raise."""
        record_dual_report("complete")

    def test_module_record_residual_mix_lookup_no_error(self):
        """Module-level record_residual_mix_lookup does not raise."""
        record_residual_mix_lookup("AIB_RESIDUAL_MIX")

    def test_module_set_active_instruments_no_error(self):
        """Module-level set_active_instruments does not raise."""
        set_active_instruments("rec", 42)

    def test_module_record_emissions_no_error(self):
        """Module-level record_emissions does not raise."""
        record_emissions("ppa", "CO2", 100.5)
