# -*- coding: utf-8 -*-
"""
PACK-014 CSRD Retail Pack - Agent Integration Tests
======================================================

Tests for MRV agent routing, DATA agent routing, cross-pack bridges,
and health check coverage. Validates that the integration layer correctly
maps retail operations to the 100+ GreenLang platform agents.

12 tests across 4 test classes.
"""

import importlib.util
import os
import sys

import pytest

# ---------------------------------------------------------------------------
# Module loader
# ---------------------------------------------------------------------------

PACK_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def _load_module(name: str, subdir: str = "integrations"):
    """Load a module from PACK-014 via importlib."""
    path = os.path.join(PACK_ROOT, subdir, f"{name}.py")
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


# Pre-load integration modules
mrv_bridge = _load_module("mrv_retail_bridge")
data_bridge = _load_module("data_retail_bridge")
csrd_bridge = _load_module("csrd_pack_bridge")
eudr_bridge = _load_module("eudr_retail_bridge")
taxonomy_bridge = _load_module("taxonomy_bridge")
health_check = _load_module("health_check")


# ======================================================================
# 1. TestMRVAgentRouting (4 tests)
# ======================================================================


class TestMRVAgentRouting:
    """Tests for MRV agent routing table structure and coverage."""

    def test_routing_table_structure(self):
        """Verify routing table entries have required fields."""
        table = mrv_bridge.MRV_ROUTING_TABLE
        assert isinstance(table, list)
        assert len(table) >= 20

        for route in table:
            assert hasattr(route, "source")
            assert hasattr(route, "mrv_agent_id")
            assert hasattr(route, "mrv_agent_name")
            assert hasattr(route, "scope")
            assert hasattr(route, "module_path")
            assert route.mrv_agent_id.startswith("MRV-")

    def test_scope1_coverage(self):
        """Verify Scope 1 coverage: heating, refrigerant, fleet."""
        table = mrv_bridge.MRV_ROUTING_TABLE
        scope1_routes = [
            r for r in table if r.scope == mrv_bridge.MRVScope.SCOPE_1
        ]
        assert len(scope1_routes) >= 3

        scope1_sources = {r.source for r in scope1_routes}
        # Check heating, refrigerant, fleet are covered
        assert mrv_bridge.EmissionSource.STORE_HEATING in scope1_sources
        assert mrv_bridge.EmissionSource.REFRIGERANT_LEAKAGE in scope1_sources
        assert mrv_bridge.EmissionSource.DELIVERY_FLEET in scope1_sources

    def test_scope2_coverage(self):
        """Verify Scope 2 coverage: location-based, market-based."""
        table = mrv_bridge.MRV_ROUTING_TABLE
        scope2_routes = [
            r for r in table if r.scope == mrv_bridge.MRVScope.SCOPE_2
        ]
        assert len(scope2_routes) >= 2

        scope2_sources = {r.source for r in scope2_routes}
        assert mrv_bridge.EmissionSource.STORE_ELECTRICITY_LOCATION in scope2_sources
        assert mrv_bridge.EmissionSource.STORE_ELECTRICITY_MARKET in scope2_sources

    def test_graceful_degradation(self):
        """Verify bridge handles missing agents gracefully."""
        bridge = mrv_bridge.MRVRetailBridge()
        # Route a request -- should return a result even in stub mode
        result = bridge.route_calculation(
            mrv_bridge.EmissionSource.STORE_HEATING,
            {"fuel_type": "natural_gas", "consumption_kwh": 50000.0},
        )
        assert result is not None
        assert hasattr(result, "success")
        assert hasattr(result, "provenance_hash")


# ======================================================================
# 2. TestDataAgentRouting (3 tests)
# ======================================================================


class TestDataAgentRouting:
    """Tests for DATA agent routing and ERP field mapping."""

    def test_erp_field_map_4_systems(self):
        """Verify ERP field mappings cover all 4 retail ERP systems."""
        mappings = data_bridge.ERP_FIELD_MAPPINGS
        erp_systems = {m.erp_system for m in mappings}
        assert data_bridge.RetailERP.SAP_RETAIL in erp_systems
        assert data_bridge.RetailERP.ORACLE_RETAIL in erp_systems
        assert data_bridge.RetailERP.NETSUITE in erp_systems
        assert data_bridge.RetailERP.DYNAMICS_365 in erp_systems

    def test_routing_table(self):
        """Verify DataRetailBridge has data source routing."""
        bridge = data_bridge.DataRetailBridge()
        assert bridge is not None
        # Check data source enum has expected categories
        sources = list(data_bridge.DataSource)
        source_values = [s.value for s in sources]
        assert "pos_sales" in source_values
        assert "energy_bills" in source_values
        assert "waste_records" in source_values
        assert "supplier_questionnaires" in source_values

    def test_config_defaults(self):
        """Verify default config values for DataBridgeConfig."""
        config = data_bridge.DataBridgeConfig()
        assert config.pack_id == "PACK-014"
        assert config.erp_system == data_bridge.RetailERP.SAP_RETAIL
        assert config.enable_provenance is True
        assert config.enable_quality_profiling is True
        assert config.max_records_per_batch == 10000


# ======================================================================
# 3. TestCrossPackBridges (3 tests)
# ======================================================================


class TestCrossPackBridges:
    """Tests for cross-pack bridge initialization."""

    def test_csrd_bridge_init(self):
        """Verify CSRDPackBridge initializes and has ESRS chapter mapping."""
        bridge = csrd_bridge.CSRDPackBridge()
        assert bridge is not None
        assert bridge.config is not None
        chapters = bridge.config.retail_esrs_chapters
        assert isinstance(chapters, list)
        assert len(chapters) >= 4
        # E1 Climate, E5 Resource, S2 Workers, S4 Consumers
        for expected in ["E1", "E5", "S2", "S4"]:
            assert expected in chapters

    def test_eudr_bridge_init(self):
        """Verify EUDRRetailBridge initializes with commodity mapping."""
        bridge = eudr_bridge.EUDRRetailBridge()
        assert bridge is not None
        assert bridge.config is not None
        # Check PRODUCT_COMMODITY_MAP is populated
        mapping = eudr_bridge.PRODUCT_COMMODITY_MAP
        assert isinstance(mapping, list)
        assert len(mapping) >= 5

    def test_taxonomy_bridge_init(self):
        """Verify TaxonomyBridge initializes with NACE activities."""
        bridge = taxonomy_bridge.TaxonomyBridge()
        assert bridge is not None
        assert bridge.config is not None
        activities = taxonomy_bridge.RETAIL_NACE_ACTIVITIES
        assert isinstance(activities, list)
        assert len(activities) >= 3
        # Check first activity has nace_code
        first = activities[0]
        assert hasattr(first, "nace_code")
        assert first.nace_code.startswith("G") or first.nace_code.startswith("4")


# ======================================================================
# 4. TestHealthCheckCoverage (2 tests)
# ======================================================================


class TestHealthCheckCoverage:
    """Tests for RetailHealthCheck coverage."""

    def test_all_22_categories(self):
        """Verify all 22 health check categories are enumerated."""
        categories = list(health_check.CheckCategory)
        assert len(categories) == 22

        expected_categories = [
            "engines",
            "workflows",
            "templates",
            "integrations",
            "presets",
            "config",
            "manifest",
            "demo",
            "mrv_agents",
            "data_agents",
            "found_agents",
            "eudr_agents",
            "database",
            "cache",
            "api",
            "auth",
            "audit",
            "observability",
            "feature_flags",
            "disk_space",
            "memory",
            "network",
        ]
        cat_values = [c.value for c in categories]
        for expected in expected_categories:
            assert expected in cat_values, f"Missing category: {expected}"

    def test_handlers_callable(self):
        """Verify each category has a callable handler."""
        hc = health_check.RetailHealthCheck()
        handlers = hc._check_handlers
        assert len(handlers) == 22

        for cat, handler in handlers.items():
            assert callable(handler), f"Handler for {cat.value} is not callable"
