# -*- coding: utf-8 -*-
"""
Unit tests for Steam/Heat Purchase Agent Prometheus metrics - AGENT-MRV-011.

Tests the SteamHeatPurchaseMetrics singleton class with all 12 Prometheus
metrics, recording methods, graceful fallback without prometheus_client,
and module-level convenience functions.

Coverage targets:
- Singleton pattern (same instance, reset creates new)
- All 12 metrics registered and accessible
- Each record_* method executes without error
- reset() clears state
- get_metrics() module function
- Graceful fallback without prometheus_client
- get_metrics_summary() returns expected keys

Test count target: ~65 tests.
"""
from __future__ import annotations

from typing import Any, Dict
from unittest.mock import patch

import pytest

try:
    from greenlang.steam_heat_purchase.metrics import (
        SteamHeatPurchaseMetrics,
        get_metrics,
        reset,
        PROMETHEUS_AVAILABLE,
        # Module-level convenience functions
        record_calculation,
        record_batch,
        record_chp_allocation,
        record_uncertainty,
        record_compliance_check,
        set_active_facilities,
        record_db_lookup,
        record_error,
        record_emissions,
        record_biogenic_emissions,
    )

    METRICS_AVAILABLE = True
except ImportError:
    METRICS_AVAILABLE = False

# Try to import prometheus_client for registry-based testing
try:
    from prometheus_client import CollectorRegistry

    PROM_CLIENT_AVAILABLE = True
except ImportError:
    PROM_CLIENT_AVAILABLE = False

pytestmark = pytest.mark.skipif(
    not METRICS_AVAILABLE,
    reason="greenlang.steam_heat_purchase.metrics not importable",
)


# ===================================================================
# Helpers
# ===================================================================


def _fresh_metrics(registry=None) -> SteamHeatPurchaseMetrics:
    """Create a fresh metrics instance after reset."""
    reset()
    if registry is not None:
        return SteamHeatPurchaseMetrics(registry=registry)
    return SteamHeatPurchaseMetrics()


def _isolated_registry():
    """Create a fresh CollectorRegistry for test isolation."""
    if PROM_CLIENT_AVAILABLE:
        return CollectorRegistry()
    return None


# ===================================================================
# Section 1: Singleton pattern
# ===================================================================


class TestMetricsSingleton:
    """Tests for the SteamHeatPurchaseMetrics singleton pattern."""

    def test_same_instance_returned(self):
        reset()
        m1 = SteamHeatPurchaseMetrics()
        m2 = SteamHeatPurchaseMetrics()
        assert m1 is m2

    def test_reset_creates_new_instance(self):
        m1 = SteamHeatPurchaseMetrics()
        reset()
        m2 = SteamHeatPurchaseMetrics()
        assert m1 is not m2

    def test_get_metrics_returns_singleton(self):
        reset()
        m1 = get_metrics()
        m2 = get_metrics()
        assert m1 is m2

    def test_get_metrics_matches_direct_construction(self):
        reset()
        m1 = SteamHeatPurchaseMetrics()
        m2 = get_metrics()
        assert m1 is m2


# ===================================================================
# Section 2: Metric registration (12 metrics)
# ===================================================================


class TestMetricRegistration:
    """Tests that all 12 Prometheus metrics are registered."""

    @pytest.fixture(autouse=True)
    def _setup_metrics(self):
        """Create a fresh metrics instance for each test."""
        reset()
        self.metrics = SteamHeatPurchaseMetrics()

    def test_has_calculations_total(self):
        assert hasattr(self.metrics, "calculations_total")

    def test_has_calculation_duration_seconds(self):
        assert hasattr(self.metrics, "calculation_duration_seconds")

    def test_has_total_co2e_kg(self):
        assert hasattr(self.metrics, "total_co2e_kg")

    def test_has_biogenic_co2_kg(self):
        assert hasattr(self.metrics, "biogenic_co2_kg")

    def test_has_batch_calculations_total(self):
        assert hasattr(self.metrics, "batch_calculations_total")

    def test_has_batch_size(self):
        assert hasattr(self.metrics, "batch_size")

    def test_has_chp_allocations_total(self):
        assert hasattr(self.metrics, "chp_allocations_total")

    def test_has_uncertainty_analyses_total(self):
        assert hasattr(self.metrics, "uncertainty_analyses_total")

    def test_has_compliance_checks_total(self):
        assert hasattr(self.metrics, "compliance_checks_total")

    def test_has_active_facilities(self):
        assert hasattr(self.metrics, "active_facilities")

    def test_has_database_lookups_total(self):
        assert hasattr(self.metrics, "database_lookups_total")

    def test_has_errors_total(self):
        assert hasattr(self.metrics, "errors_total")

    def test_prometheus_available_attribute(self):
        assert isinstance(self.metrics.prometheus_available, bool)


# ===================================================================
# Section 3: Recording methods (with prometheus_client)
# ===================================================================


@pytest.mark.skipif(
    not PROM_CLIENT_AVAILABLE,
    reason="prometheus_client not installed",
)
class TestRecordingMethods:
    """Tests for each record_* method with a real CollectorRegistry."""

    @pytest.fixture(autouse=True)
    def _setup_metrics(self):
        """Create metrics with an isolated registry."""
        reset()
        self.registry = CollectorRegistry()
        self.metrics = SteamHeatPurchaseMetrics(registry=self.registry)

    def test_record_calculation_success(self):
        self.metrics.record_calculation(
            energy_type="steam",
            method="supplier_specific",
            status="success",
            duration=0.042,
            co2e_kg=150.3,
            biogenic_kg=0.0,
            fuel_type="natural_gas",
            tenant_id="tenant-001",
        )
        # No exception means success

    def test_record_calculation_with_biogenic(self):
        self.metrics.record_calculation(
            energy_type="hot_water",
            method="default_emission_factor",
            status="success",
            duration=0.065,
            co2e_kg=45.0,
            biogenic_kg=12.5,
            fuel_type="biomass",
            tenant_id="tenant-002",
        )

    def test_record_calculation_failure(self):
        self.metrics.record_calculation(
            energy_type="district_heat",
            method="fuel_specific",
            status="failure",
            duration=0.010,
            co2e_kg=0.0,
            biogenic_kg=0.0,
            fuel_type="coal",
            tenant_id="tenant-003",
        )

    def test_record_batch_success(self):
        self.metrics.record_batch(
            status="success",
            size=100,
            tenant_id="tenant-001",
        )

    def test_record_batch_failure(self):
        self.metrics.record_batch(
            status="failure",
            size=50,
            tenant_id="tenant-002",
        )

    def test_record_batch_partial(self):
        self.metrics.record_batch(
            status="partial",
            size=25,
            tenant_id="tenant-003",
        )

    def test_record_chp_allocation_efficiency(self):
        self.metrics.record_chp_allocation(
            method="efficiency",
            fuel_type="natural_gas",
            tenant_id="tenant-001",
        )

    def test_record_chp_allocation_energy(self):
        self.metrics.record_chp_allocation(
            method="energy",
            fuel_type="coal",
            tenant_id="tenant-001",
        )

    def test_record_chp_allocation_exergy(self):
        self.metrics.record_chp_allocation(
            method="exergy",
            fuel_type="biomass",
            tenant_id="tenant-002",
        )

    def test_record_uncertainty_monte_carlo(self):
        self.metrics.record_uncertainty(
            method="monte_carlo",
            tenant_id="tenant-001",
        )

    def test_record_uncertainty_analytical(self):
        self.metrics.record_uncertainty(
            method="analytical",
            tenant_id="tenant-002",
        )

    def test_record_compliance_check_compliant(self):
        self.metrics.record_compliance_check(
            framework="GHG_PROTOCOL",
            status="compliant",
            tenant_id="tenant-001",
        )

    def test_record_compliance_check_non_compliant(self):
        self.metrics.record_compliance_check(
            framework="ISO_14064",
            status="non_compliant",
            tenant_id="tenant-002",
        )

    def test_set_active_facilities(self):
        self.metrics.set_active_facilities(
            facility_type="chp_plant",
            count=12,
            tenant_id="tenant-001",
        )

    def test_set_active_facilities_zero(self):
        self.metrics.set_active_facilities(
            facility_type="district_heating",
            count=0,
            tenant_id="tenant-002",
        )

    def test_record_db_lookup_hit(self):
        self.metrics.record_db_lookup(
            lookup_type="emission_factor",
            status="hit",
        )

    def test_record_db_lookup_miss(self):
        self.metrics.record_db_lookup(
            lookup_type="fuel_property",
            status="miss",
        )

    def test_record_db_lookup_error(self):
        self.metrics.record_db_lookup(
            lookup_type="chp_parameter",
            status="error",
        )

    def test_record_error_validation(self):
        self.metrics.record_error(
            engine="calculator",
            error_type="validation_error",
            tenant_id="tenant-001",
        )

    def test_record_error_database(self):
        self.metrics.record_error(
            engine="database",
            error_type="database_error",
            tenant_id="tenant-002",
        )

    def test_record_emissions_standalone(self):
        self.metrics.record_emissions(
            energy_type="steam",
            fuel_type="natural_gas",
            co2e_kg=500.0,
            tenant_id="tenant-001",
        )

    def test_record_biogenic_emissions_standalone(self):
        self.metrics.record_biogenic_emissions(
            fuel_type="biomass",
            biogenic_kg=100.0,
            tenant_id="tenant-001",
        )


# ===================================================================
# Section 4: Metrics summary
# ===================================================================


class TestMetricsSummary:
    """Tests for the get_metrics_summary() method."""

    def test_summary_returns_dict(self):
        m = _fresh_metrics()
        summary = m.get_metrics_summary()
        assert isinstance(summary, dict)

    def test_summary_has_12_metric_keys(self):
        m = _fresh_metrics()
        summary = m.get_metrics_summary()
        expected_keys = {
            "gl_shp_calculations_total",
            "gl_shp_calculation_duration_seconds",
            "gl_shp_total_co2e_kg",
            "gl_shp_biogenic_co2_kg",
            "gl_shp_batch_calculations_total",
            "gl_shp_batch_size",
            "gl_shp_chp_allocations_total",
            "gl_shp_uncertainty_analyses_total",
            "gl_shp_compliance_checks_total",
            "gl_shp_active_facilities",
            "gl_shp_database_lookups_total",
            "gl_shp_errors_total",
            "prometheus_available",
        }
        assert expected_keys.issubset(set(summary.keys()))

    def test_summary_prometheus_available_key(self):
        m = _fresh_metrics()
        summary = m.get_metrics_summary()
        assert "prometheus_available" in summary
        assert isinstance(summary["prometheus_available"], bool)


# ===================================================================
# Section 5: Module-level convenience functions
# ===================================================================


class TestModuleLevelFunctions:
    """Tests for module-level convenience functions."""

    def test_get_metrics_returns_instance(self):
        reset()
        m = get_metrics()
        assert isinstance(m, SteamHeatPurchaseMetrics)

    def test_reset_clears_instance(self):
        m1 = get_metrics()
        reset()
        m2 = get_metrics()
        assert m1 is not m2

    def test_record_calculation_convenience(self):
        reset()
        # Should not raise
        record_calculation(
            "steam", "supplier_specific", "success",
            0.05, 120.0, 0.0, "natural_gas", "tenant-001",
        )

    def test_record_batch_convenience(self):
        reset()
        record_batch("success", 10, "tenant-001")

    def test_record_chp_allocation_convenience(self):
        reset()
        record_chp_allocation("efficiency", "natural_gas", "tenant-001")

    def test_record_uncertainty_convenience(self):
        reset()
        record_uncertainty("monte_carlo", "tenant-001")

    def test_record_compliance_check_convenience(self):
        reset()
        record_compliance_check("GHG_PROTOCOL", "compliant", "tenant-001")

    def test_set_active_facilities_convenience(self):
        reset()
        set_active_facilities("chp_plant", 5, "tenant-001")

    def test_record_db_lookup_convenience(self):
        reset()
        record_db_lookup("emission_factor", "hit")

    def test_record_error_convenience(self):
        reset()
        record_error("calculator", "validation_error", "tenant-001")

    def test_record_emissions_convenience(self):
        reset()
        record_emissions("steam", "natural_gas", 100.0, "tenant-001")

    def test_record_biogenic_emissions_convenience(self):
        reset()
        record_biogenic_emissions("biomass", 50.0, "tenant-001")


# ===================================================================
# Section 6: Graceful fallback without prometheus_client
# ===================================================================


class TestGracefulFallback:
    """Tests for graceful no-op behavior when prometheus_client is unavailable."""

    def test_noop_record_calculation(self):
        """Simulate no prometheus_client by patching prometheus_available."""
        reset()
        m = SteamHeatPurchaseMetrics()
        orig = m.prometheus_available
        m.prometheus_available = False
        # Should be a silent no-op
        m.record_calculation(
            "steam", "default_emission_factor", "success",
            0.01, 50.0, 0.0, "natural_gas", "t1",
        )
        m.prometheus_available = orig

    def test_noop_record_batch(self):
        reset()
        m = SteamHeatPurchaseMetrics()
        m.prometheus_available = False
        m.record_batch("success", 10, "t1")

    def test_noop_record_chp_allocation(self):
        reset()
        m = SteamHeatPurchaseMetrics()
        m.prometheus_available = False
        m.record_chp_allocation("efficiency", "natural_gas", "t1")

    def test_noop_record_uncertainty(self):
        reset()
        m = SteamHeatPurchaseMetrics()
        m.prometheus_available = False
        m.record_uncertainty("monte_carlo", "t1")

    def test_noop_record_compliance_check(self):
        reset()
        m = SteamHeatPurchaseMetrics()
        m.prometheus_available = False
        m.record_compliance_check("GHG_PROTOCOL", "compliant", "t1")

    def test_noop_set_active_facilities(self):
        reset()
        m = SteamHeatPurchaseMetrics()
        m.prometheus_available = False
        m.set_active_facilities("chp_plant", 5, "t1")

    def test_noop_record_db_lookup(self):
        reset()
        m = SteamHeatPurchaseMetrics()
        m.prometheus_available = False
        m.record_db_lookup("emission_factor", "hit")

    def test_noop_record_error(self):
        reset()
        m = SteamHeatPurchaseMetrics()
        m.prometheus_available = False
        m.record_error("calculator", "validation_error", "t1")

    def test_noop_record_emissions(self):
        reset()
        m = SteamHeatPurchaseMetrics()
        m.prometheus_available = False
        m.record_emissions("steam", "natural_gas", 100.0, "t1")

    def test_noop_record_biogenic_emissions(self):
        reset()
        m = SteamHeatPurchaseMetrics()
        m.prometheus_available = False
        m.record_biogenic_emissions("biomass", 50.0, "t1")

    def test_noop_summary_returns_none_values(self):
        reset()
        m = SteamHeatPurchaseMetrics()
        m.prometheus_available = False
        summary = m.get_metrics_summary()
        assert summary["prometheus_available"] is False
        assert summary["gl_shp_calculations_total"] is None
