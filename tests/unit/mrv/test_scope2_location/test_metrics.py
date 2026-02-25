# -*- coding: utf-8 -*-
"""
Unit tests for AGENT-MRV-009 Scope 2 Location-Based Emissions Agent Metrics.

Tests all 12 Prometheus metrics existence, gl_s2l_ prefix naming,
Scope2LocationMetrics singleton pattern, recording methods, graceful
fallback when prometheus_client is not installed, get_metrics_summary(),
and module-level convenience functions.

Target: 25+ tests, 85%+ coverage of metrics.py.

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
    from greenlang.scope2_location.metrics import (
        PROMETHEUS_AVAILABLE,
        Scope2LocationMetrics,
        get_metrics,
        record_calculation,
        record_electricity_calculation,
        record_steam_heat_cooling,
        record_compliance_check,
        record_uncertainty_run,
        record_error,
        set_active_facilities,
        record_grid_factor_lookup,
        record_td_loss_adjustment,
        record_consumption,
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
    """Tests for the Scope2LocationMetrics singleton pattern."""

    @pytest.mark.skipif(
        not METRICS_AVAILABLE or not PROMETHEUS_AVAILABLE,
        reason="prometheus_client not installed",
    )
    def test_singleton_returns_same_instance(self):
        """Two instantiations return the same object."""
        from prometheus_client import CollectorRegistry
        registry = CollectorRegistry()
        Scope2LocationMetrics._reset()
        m1 = Scope2LocationMetrics(registry=registry)
        m2 = Scope2LocationMetrics(registry=registry)
        assert m1 is m2

    @pytest.mark.skipif(
        not METRICS_AVAILABLE or not PROMETHEUS_AVAILABLE,
        reason="prometheus_client not installed",
    )
    def test_reset_creates_new_instance(self):
        """After _reset(), a new instance is created."""
        from prometheus_client import CollectorRegistry
        reg1 = CollectorRegistry()
        Scope2LocationMetrics._reset()
        m1 = Scope2LocationMetrics(registry=reg1)
        id1 = id(m1)
        Scope2LocationMetrics._reset()
        reg2 = CollectorRegistry()
        m2 = Scope2LocationMetrics(registry=reg2)
        assert id1 != id(m2)

    @pytest.mark.skipif(
        not METRICS_AVAILABLE or not PROMETHEUS_AVAILABLE,
        reason="prometheus_client not installed",
    )
    def test_prometheus_available_flag(self):
        """prometheus_available attribute reflects library availability."""
        from prometheus_client import CollectorRegistry
        registry = CollectorRegistry()
        Scope2LocationMetrics._reset()
        m = Scope2LocationMetrics(registry=registry)
        assert m.prometheus_available == PROMETHEUS_AVAILABLE


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
        """gl_s2l_calculations_total counter is defined."""
        from prometheus_client import CollectorRegistry
        registry = CollectorRegistry()
        Scope2LocationMetrics._reset()
        m = Scope2LocationMetrics(registry=registry)
        assert m.calculations_total is not None

    @pytest.mark.skipif(
        not METRICS_AVAILABLE or not PROMETHEUS_AVAILABLE,
        reason="prometheus_client not installed",
    )
    def test_calculation_duration_seconds_exists(self):
        """gl_s2l_calculation_duration_seconds histogram is defined."""
        from prometheus_client import CollectorRegistry
        registry = CollectorRegistry()
        Scope2LocationMetrics._reset()
        m = Scope2LocationMetrics(registry=registry)
        assert m.calculation_duration_seconds is not None

    @pytest.mark.skipif(
        not METRICS_AVAILABLE or not PROMETHEUS_AVAILABLE,
        reason="prometheus_client not installed",
    )
    def test_emissions_co2e_tonnes_exists(self):
        """gl_s2l_emissions_co2e_tonnes counter is defined."""
        from prometheus_client import CollectorRegistry
        registry = CollectorRegistry()
        Scope2LocationMetrics._reset()
        m = Scope2LocationMetrics(registry=registry)
        assert m.emissions_co2e_tonnes is not None

    @pytest.mark.skipif(
        not METRICS_AVAILABLE or not PROMETHEUS_AVAILABLE,
        reason="prometheus_client not installed",
    )
    def test_consumption_mwh_total_exists(self):
        """gl_s2l_consumption_mwh_total counter is defined."""
        from prometheus_client import CollectorRegistry
        registry = CollectorRegistry()
        Scope2LocationMetrics._reset()
        m = Scope2LocationMetrics(registry=registry)
        assert m.consumption_mwh_total is not None

    @pytest.mark.skipif(
        not METRICS_AVAILABLE or not PROMETHEUS_AVAILABLE,
        reason="prometheus_client not installed",
    )
    def test_electricity_calculations_total_exists(self):
        """gl_s2l_electricity_calculations_total counter is defined."""
        from prometheus_client import CollectorRegistry
        registry = CollectorRegistry()
        Scope2LocationMetrics._reset()
        m = Scope2LocationMetrics(registry=registry)
        assert m.electricity_calculations_total is not None

    @pytest.mark.skipif(
        not METRICS_AVAILABLE or not PROMETHEUS_AVAILABLE,
        reason="prometheus_client not installed",
    )
    def test_steam_heat_cooling_calculations_total_exists(self):
        """gl_s2l_steam_heat_cooling_calculations_total counter is defined."""
        from prometheus_client import CollectorRegistry
        registry = CollectorRegistry()
        Scope2LocationMetrics._reset()
        m = Scope2LocationMetrics(registry=registry)
        assert m.steam_heat_cooling_calculations_total is not None

    @pytest.mark.skipif(
        not METRICS_AVAILABLE or not PROMETHEUS_AVAILABLE,
        reason="prometheus_client not installed",
    )
    def test_compliance_checks_total_exists(self):
        """gl_s2l_compliance_checks_total counter is defined."""
        from prometheus_client import CollectorRegistry
        registry = CollectorRegistry()
        Scope2LocationMetrics._reset()
        m = Scope2LocationMetrics(registry=registry)
        assert m.compliance_checks_total is not None

    @pytest.mark.skipif(
        not METRICS_AVAILABLE or not PROMETHEUS_AVAILABLE,
        reason="prometheus_client not installed",
    )
    def test_uncertainty_runs_total_exists(self):
        """gl_s2l_uncertainty_runs_total counter is defined."""
        from prometheus_client import CollectorRegistry
        registry = CollectorRegistry()
        Scope2LocationMetrics._reset()
        m = Scope2LocationMetrics(registry=registry)
        assert m.uncertainty_runs_total is not None

    @pytest.mark.skipif(
        not METRICS_AVAILABLE or not PROMETHEUS_AVAILABLE,
        reason="prometheus_client not installed",
    )
    def test_errors_total_exists(self):
        """gl_s2l_errors_total counter is defined."""
        from prometheus_client import CollectorRegistry
        registry = CollectorRegistry()
        Scope2LocationMetrics._reset()
        m = Scope2LocationMetrics(registry=registry)
        assert m.errors_total is not None

    @pytest.mark.skipif(
        not METRICS_AVAILABLE or not PROMETHEUS_AVAILABLE,
        reason="prometheus_client not installed",
    )
    def test_active_facilities_exists(self):
        """gl_s2l_active_facilities gauge is defined."""
        from prometheus_client import CollectorRegistry
        registry = CollectorRegistry()
        Scope2LocationMetrics._reset()
        m = Scope2LocationMetrics(registry=registry)
        assert m.active_facilities is not None

    @pytest.mark.skipif(
        not METRICS_AVAILABLE or not PROMETHEUS_AVAILABLE,
        reason="prometheus_client not installed",
    )
    def test_grid_factor_lookups_total_exists(self):
        """gl_s2l_grid_factor_lookups_total counter is defined."""
        from prometheus_client import CollectorRegistry
        registry = CollectorRegistry()
        Scope2LocationMetrics._reset()
        m = Scope2LocationMetrics(registry=registry)
        assert m.grid_factor_lookups_total is not None

    @pytest.mark.skipif(
        not METRICS_AVAILABLE or not PROMETHEUS_AVAILABLE,
        reason="prometheus_client not installed",
    )
    def test_td_loss_adjustments_total_exists(self):
        """gl_s2l_td_loss_adjustments_total counter is defined."""
        from prometheus_client import CollectorRegistry
        registry = CollectorRegistry()
        Scope2LocationMetrics._reset()
        m = Scope2LocationMetrics(registry=registry)
        assert m.td_loss_adjustments_total is not None


# ===========================================================================
# Recording Method Tests (no-op when prometheus not available)
# ===========================================================================


@_SKIP
class TestRecordingMethods:
    """Tests for metric recording methods -- must not raise even without prometheus."""

    @pytest.fixture(autouse=True)
    def _create_metrics(self):
        """Create a metrics instance with a custom registry for each test."""
        Scope2LocationMetrics._reset()
        if PROMETHEUS_AVAILABLE:
            from prometheus_client import CollectorRegistry
            registry = CollectorRegistry()
            self.metrics = Scope2LocationMetrics(registry=registry)
        else:
            self.metrics = Scope2LocationMetrics()
        yield
        Scope2LocationMetrics._reset()

    def test_record_calculation_no_error(self):
        """record_calculation does not raise regardless of prometheus."""
        self.metrics.record_calculation("electricity", "location_based", 0.05, 150.3)

    def test_record_electricity_calculation_no_error(self):
        """record_electricity_calculation does not raise."""
        self.metrics.record_electricity_calculation(0.042)

    def test_record_steam_heat_cooling_no_error(self):
        """record_steam_heat_cooling does not raise."""
        self.metrics.record_steam_heat_cooling("steam", 0.065)

    def test_record_compliance_check_no_error(self):
        """record_compliance_check does not raise."""
        self.metrics.record_compliance_check("GHG_PROTOCOL", "compliant")

    def test_record_uncertainty_run_no_error(self):
        """record_uncertainty_run does not raise."""
        self.metrics.record_uncertainty_run("monte_carlo")

    def test_record_error_no_error(self):
        """record_error does not raise."""
        self.metrics.record_error("validation_error")

    def test_set_active_facilities_no_error(self):
        """set_active_facilities does not raise."""
        self.metrics.set_active_facilities(12)

    def test_record_grid_factor_lookup_no_error(self):
        """record_grid_factor_lookup does not raise."""
        self.metrics.record_grid_factor_lookup("EPA_EGRID")

    def test_record_td_loss_adjustment_no_error(self):
        """record_td_loss_adjustment does not raise."""
        self.metrics.record_td_loss_adjustment()

    def test_record_consumption_no_error(self):
        """record_consumption does not raise."""
        self.metrics.record_consumption("electricity", "office", 1000.0)

    def test_record_emissions_no_error(self):
        """record_emissions does not raise."""
        self.metrics.record_emissions("electricity", "CO2", 150.3)


# ===========================================================================
# Metrics Summary Tests
# ===========================================================================


@_SKIP
class TestMetricsSummary:
    """Tests for get_metrics_summary()."""

    @pytest.fixture(autouse=True)
    def _create_metrics(self):
        """Create a metrics instance with a custom registry for each test."""
        Scope2LocationMetrics._reset()
        if PROMETHEUS_AVAILABLE:
            from prometheus_client import CollectorRegistry
            registry = CollectorRegistry()
            self.metrics = Scope2LocationMetrics(registry=registry)
        else:
            self.metrics = Scope2LocationMetrics()
        yield
        Scope2LocationMetrics._reset()

    def test_summary_returns_dict(self):
        """get_metrics_summary() returns a dictionary."""
        summary = self.metrics.get_metrics_summary()
        assert isinstance(summary, dict)

    def test_summary_has_all_12_metric_keys(self):
        """Summary contains all 12 metric keys plus prometheus_available."""
        summary = self.metrics.get_metrics_summary()
        expected_keys = [
            "gl_s2l_calculations_total",
            "gl_s2l_calculation_duration_seconds",
            "gl_s2l_emissions_co2e_tonnes",
            "gl_s2l_consumption_mwh_total",
            "gl_s2l_electricity_calculations_total",
            "gl_s2l_steam_heat_cooling_calculations_total",
            "gl_s2l_compliance_checks_total",
            "gl_s2l_uncertainty_runs_total",
            "gl_s2l_errors_total",
            "gl_s2l_active_facilities",
            "gl_s2l_grid_factor_lookups_total",
            "gl_s2l_td_loss_adjustments_total",
            "prometheus_available",
        ]
        for key in expected_keys:
            assert key in summary, f"Key '{key}' missing from summary"

    def test_summary_prometheus_available_flag(self):
        """Summary contains correct prometheus_available flag."""
        summary = self.metrics.get_metrics_summary()
        assert summary["prometheus_available"] == PROMETHEUS_AVAILABLE


# ===========================================================================
# Graceful Fallback Tests (no prometheus_client)
# ===========================================================================


@_SKIP
class TestGracefulFallback:
    """Tests for graceful fallback when prometheus_client is not installed."""

    def test_noop_metrics_are_none_when_no_prometheus(self):
        """When prometheus is unavailable, all metric attributes are None."""
        Scope2LocationMetrics._reset()
        m = Scope2LocationMetrics.__new__(Scope2LocationMetrics)
        m._initialized = False
        m.prometheus_available = False
        m._registry = None
        m._init_noop_metrics()
        m._initialized = True

        assert m.calculations_total is None
        assert m.calculation_duration_seconds is None
        assert m.emissions_co2e_tonnes is None
        assert m.consumption_mwh_total is None
        assert m.electricity_calculations_total is None
        assert m.steam_heat_cooling_calculations_total is None
        assert m.compliance_checks_total is None
        assert m.uncertainty_runs_total is None
        assert m.errors_total is None
        assert m.active_facilities is None
        assert m.grid_factor_lookups_total is None
        assert m.td_loss_adjustments_total is None

    def test_noop_summary_returns_none_values(self):
        """When prometheus unavailable, summary has None values."""
        Scope2LocationMetrics._reset()
        m = Scope2LocationMetrics.__new__(Scope2LocationMetrics)
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
        import greenlang.scope2_location.metrics as _m
        _m.Scope2LocationMetrics._reset()
        _m._default_metrics = None
        if PROMETHEUS_AVAILABLE:
            from prometheus_client import CollectorRegistry
            registry = CollectorRegistry()
            instance = Scope2LocationMetrics(registry=registry)
        else:
            instance = Scope2LocationMetrics()
        _m._default_metrics = instance
        yield
        _m.Scope2LocationMetrics._reset()
        _m._default_metrics = None

    def test_get_metrics_returns_singleton(self):
        """get_metrics() returns a Scope2LocationMetrics instance."""
        m = get_metrics()
        assert isinstance(m, Scope2LocationMetrics)

    def test_module_record_calculation_no_error(self):
        """Module-level record_calculation does not raise."""
        record_calculation("electricity", "location_based", 0.05, 150.3)

    def test_module_record_electricity_calculation_no_error(self):
        """Module-level record_electricity_calculation does not raise."""
        record_electricity_calculation(0.042)

    def test_module_record_steam_heat_cooling_no_error(self):
        """Module-level record_steam_heat_cooling does not raise."""
        record_steam_heat_cooling("steam", 0.065)

    def test_module_record_compliance_check_no_error(self):
        """Module-level record_compliance_check does not raise."""
        record_compliance_check("GHG_PROTOCOL", "compliant")

    def test_module_record_uncertainty_run_no_error(self):
        """Module-level record_uncertainty_run does not raise."""
        record_uncertainty_run("monte_carlo")

    def test_module_record_error_no_error(self):
        """Module-level record_error does not raise."""
        record_error("validation_error")

    def test_module_set_active_facilities_no_error(self):
        """Module-level set_active_facilities does not raise."""
        set_active_facilities(5)

    def test_module_record_grid_factor_lookup_no_error(self):
        """Module-level record_grid_factor_lookup does not raise."""
        record_grid_factor_lookup("IEA")

    def test_module_record_td_loss_adjustment_no_error(self):
        """Module-level record_td_loss_adjustment does not raise."""
        record_td_loss_adjustment()

    def test_module_record_consumption_no_error(self):
        """Module-level record_consumption does not raise."""
        record_consumption("electricity", "office", 500.0)

    def test_module_record_emissions_no_error(self):
        """Module-level record_emissions does not raise."""
        record_emissions("electricity", "CO2", 100.5)
