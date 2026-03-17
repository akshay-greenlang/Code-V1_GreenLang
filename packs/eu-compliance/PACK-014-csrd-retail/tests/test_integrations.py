# -*- coding: utf-8 -*-
"""
PACK-014 CSRD Retail Pack - Integration Tests
================================================

Tests for the 10 integration bridges: PackOrchestrator, CSRDPackBridge,
MRVRetailBridge, DataRetailBridge, EUDRRetailBridge, CircularEconomyBridge,
SupplyChainBridge, TaxonomyBridge, RetailHealthCheck, and RetailSetupWizard.

20 tests across 10 test classes.
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
pack_orch = _load_module("pack_orchestrator")
csrd_bridge_mod = _load_module("csrd_pack_bridge")
mrv_bridge_mod = _load_module("mrv_retail_bridge")
data_bridge_mod = _load_module("data_retail_bridge")
eudr_bridge_mod = _load_module("eudr_retail_bridge")
circular_bridge_mod = _load_module("circular_economy_bridge")
supply_bridge_mod = _load_module("supply_chain_bridge")
taxonomy_bridge_mod = _load_module("taxonomy_bridge")
health_check_mod = _load_module("health_check")
setup_wizard_mod = _load_module("setup_wizard")


# ======================================================================
# 1. TestPackOrchestrator (3 tests)
# ======================================================================


class TestPackOrchestrator:
    """Tests for RetailPipelineOrchestrator."""

    def test_init(self):
        orch = pack_orch.RetailPipelineOrchestrator()
        assert orch is not None
        assert orch.config is not None
        assert orch.config.pack_id == "PACK-014"

    def test_11_phases(self):
        phases = pack_orch.PHASE_EXECUTION_ORDER
        assert len(phases) == 11

    def test_dependency_graph(self):
        deps = pack_orch.PHASE_DEPENDENCIES
        assert isinstance(deps, dict)
        assert len(deps) == 11
        # Initialization has no dependencies
        init_phase = pack_orch.RetailPipelinePhase.INITIALIZATION
        assert deps[init_phase] == []
        # Reporting depends on benchmarking
        reporting = pack_orch.RetailPipelinePhase.REPORTING
        benchmark = pack_orch.RetailPipelinePhase.BENCHMARKING
        assert benchmark in deps[reporting]


# ======================================================================
# 2. TestCSRDPackBridge (2 tests)
# ======================================================================


class TestCSRDPackBridge:
    """Tests for CSRDPackBridge."""

    def test_init(self):
        bridge = csrd_bridge_mod.CSRDPackBridge()
        assert bridge is not None
        assert bridge.config is not None

    def test_esrs_chapters(self):
        bridge = csrd_bridge_mod.CSRDPackBridge()
        chapters = bridge.config.retail_esrs_chapters
        assert isinstance(chapters, list)
        assert "E1" in chapters
        assert "E5" in chapters
        assert "S2" in chapters
        assert "S4" in chapters


# ======================================================================
# 3. TestMRVRetailBridge (3 tests)
# ======================================================================


class TestMRVRetailBridge:
    """Tests for MRVRetailBridge."""

    def test_init(self):
        bridge = mrv_bridge_mod.MRVRetailBridge()
        assert bridge is not None
        assert bridge.config is not None

    def test_routing_table_20_plus(self):
        bridge = mrv_bridge_mod.MRVRetailBridge()
        table = bridge.get_routing_table()
        assert isinstance(table, list)
        assert len(table) >= 20

    def test_subsector_agents(self):
        """Verify routing table covers multiple scopes."""
        table = mrv_bridge_mod.MRV_ROUTING_TABLE
        scopes = {r.scope for r in table}
        assert mrv_bridge_mod.MRVScope.SCOPE_1 in scopes
        assert mrv_bridge_mod.MRVScope.SCOPE_2 in scopes


# ======================================================================
# 4. TestDataRetailBridge (2 tests)
# ======================================================================


class TestDataRetailBridge:
    """Tests for DataRetailBridge."""

    def test_init(self):
        bridge = data_bridge_mod.DataRetailBridge()
        assert bridge is not None
        assert bridge.config is not None
        assert bridge.config.erp_system.value == "sap_retail"

    def test_erp_field_map(self):
        mappings = data_bridge_mod.ERP_FIELD_MAPPINGS
        assert isinstance(mappings, list)
        assert len(mappings) >= 4
        # Verify all 4 ERP systems present
        erp_systems = {m.erp_system for m in mappings}
        expected = {
            data_bridge_mod.RetailERP.SAP_RETAIL,
            data_bridge_mod.RetailERP.ORACLE_RETAIL,
            data_bridge_mod.RetailERP.NETSUITE,
            data_bridge_mod.RetailERP.DYNAMICS_365,
        }
        assert erp_systems == expected


# ======================================================================
# 5. TestEUDRRetailBridge (2 tests)
# ======================================================================


class TestEUDRRetailBridge:
    """Tests for EUDRRetailBridge."""

    def test_init(self):
        bridge = eudr_bridge_mod.EUDRRetailBridge()
        assert bridge is not None
        assert bridge.config is not None

    def test_commodity_mapping(self):
        """Verify PRODUCT_COMMODITY_MAP exists and has entries."""
        mapping = eudr_bridge_mod.PRODUCT_COMMODITY_MAP
        assert isinstance(mapping, list)
        assert len(mapping) >= 5
        # Check structure
        first = mapping[0]
        assert hasattr(first, "product_category") or hasattr(first, "retail_category")


# ======================================================================
# 6. TestCircularEconomyBridge (2 tests)
# ======================================================================


class TestCircularEconomyBridge:
    """Tests for CircularEconomyBridge."""

    def test_init(self):
        bridge = circular_bridge_mod.CircularEconomyBridge()
        assert bridge is not None
        assert bridge.config is not None

    def test_epr_schemes(self):
        """Verify default EPR schemes are configured."""
        bridge = circular_bridge_mod.CircularEconomyBridge()
        schemes = bridge.config.epr_schemes
        assert isinstance(schemes, list)
        assert len(schemes) >= 2
        scheme_values = [s.value for s in schemes]
        assert "packaging" in scheme_values
        assert "weee" in scheme_values


# ======================================================================
# 7. TestSupplyChainBridge (2 tests)
# ======================================================================


class TestSupplyChainBridge:
    """Tests for SupplyChainBridge."""

    def test_init(self):
        bridge = supply_bridge_mod.SupplyChainBridge()
        assert bridge is not None
        assert bridge.config is not None

    def test_csddd_routing(self):
        """Verify CSDDD configuration is present."""
        bridge = supply_bridge_mod.SupplyChainBridge()
        assert bridge.config.csddd_applicable is True
        assert hasattr(bridge, "_dd_agents")
        assert isinstance(bridge._dd_agents, dict)
        # Should attempt to load agents 021-040
        assert len(bridge._dd_agents) >= 1


# ======================================================================
# 8. TestTaxonomyBridge (2 tests)
# ======================================================================


class TestTaxonomyBridge:
    """Tests for TaxonomyBridge."""

    def test_init(self):
        bridge = taxonomy_bridge_mod.TaxonomyBridge()
        assert bridge is not None
        assert bridge.config is not None

    def test_nace_activities(self):
        """Verify RETAIL_NACE_ACTIVITIES definition."""
        activities = taxonomy_bridge_mod.RETAIL_NACE_ACTIVITIES
        assert isinstance(activities, list)
        assert len(activities) >= 3
        # Check structure has nace_code
        first = activities[0]
        assert hasattr(first, "nace_code")


# ======================================================================
# 9. TestHealthCheck (2 tests)
# ======================================================================


class TestHealthCheck:
    """Tests for RetailHealthCheck."""

    def test_22_categories(self):
        """Verify all 22 check categories are defined."""
        categories = list(health_check_mod.CheckCategory)
        assert len(categories) == 22

    def test_category_handlers(self):
        """Verify health check has a handler for each category."""
        hc = health_check_mod.RetailHealthCheck()
        handlers = hc._check_handlers
        assert isinstance(handlers, dict)
        assert len(handlers) == 22
        # Verify each category has a callable handler
        for cat in health_check_mod.CheckCategory:
            assert cat in handlers
            assert callable(handlers[cat])


# ======================================================================
# 10. TestSetupWizard (2 tests)
# ======================================================================


class TestSetupWizard:
    """Tests for RetailSetupWizard."""

    def test_init(self):
        wizard = setup_wizard_mod.RetailSetupWizard()
        assert wizard is not None
        assert hasattr(wizard, "_step_handlers")

    def test_8_steps(self):
        """Verify all 8 wizard steps are defined."""
        steps = list(setup_wizard_mod.RetailWizardStep)
        assert len(steps) == 8
        expected_steps = [
            "company_profile",
            "store_portfolio",
            "retail_sub_sector",
            "regulatory_scope",
            "emissions_sources",
            "product_categories",
            "supply_chain",
            "reporting_setup",
        ]
        step_values = [s.value for s in steps]
        for expected in expected_steps:
            assert expected in step_values
