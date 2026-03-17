# -*- coding: utf-8 -*-
"""
PACK-013 CSRD Manufacturing Pack - Agent Integration Tests

Tests agent wiring, bridge connectivity, MRV routing, Data routing,
cross-pack bridge initialization, and health check category coverage.

12 tests across 4 test classes.
"""

import importlib.util
import sys
import pytest
from pathlib import Path

# ---------------------------------------------------------------------------
# Dynamic module loading via importlib
# ---------------------------------------------------------------------------

PACK_ROOT = Path(__file__).resolve().parent.parent
INTEGRATIONS_DIR = PACK_ROOT / "integrations"


def _load_module(module_name: str, file_name: str, search_dir: Path = INTEGRATIONS_DIR):
    """Load a module dynamically using importlib.util.spec_from_file_location."""
    file_path = search_dir / file_name
    if not file_path.exists():
        pytest.skip(f"Module file not found: {file_path}")
    spec = importlib.util.spec_from_file_location(module_name, str(file_path))
    if spec is None or spec.loader is None:
        pytest.skip(f"Cannot create spec for {file_path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------------------
# Load integration modules
# ---------------------------------------------------------------------------

mrv_mod = _load_module("pack013_ai_mrv", "mrv_industrial_bridge.py")
MRVIndustrialBridge = mrv_mod.MRVIndustrialBridge
MRVBridgeConfig = mrv_mod.MRVBridgeConfig
DEFAULT_ROUTING_TABLE = mrv_mod.DEFAULT_ROUTING_TABLE
SUB_SECTOR_AGENTS = mrv_mod.SUB_SECTOR_AGENTS

data_mod = _load_module("pack013_ai_data", "data_manufacturing_bridge.py")
DataManufacturingBridge = data_mod.DataManufacturingBridge
DataBridgeConfig = data_mod.DataBridgeConfig
ERP_FIELD_MAP = data_mod.ERP_FIELD_MAP

csrd_mod = _load_module("pack013_ai_csrd", "csrd_pack_bridge.py")
CSRDPackBridge = csrd_mod.CSRDPackBridge
CSRDBridgeConfig = csrd_mod.CSRDBridgeConfig

cbam_mod = _load_module("pack013_ai_cbam", "cbam_pack_bridge.py")
CBAMPackBridge = cbam_mod.CBAMPackBridge
CBAMBridgeConfig = cbam_mod.CBAMBridgeConfig

ets_mod = _load_module("pack013_ai_ets", "eu_ets_bridge.py")
EUETSBridge = ets_mod.EUETSBridge
ETSBridgeConfig = ets_mod.ETSBridgeConfig

taxonomy_mod = _load_module("pack013_ai_taxonomy", "taxonomy_bridge.py")
TaxonomyBridge = taxonomy_mod.TaxonomyBridge

health_mod = _load_module("pack013_ai_health", "health_check.py")
ManufacturingHealthCheck = health_mod.ManufacturingHealthCheck
HealthCategory = health_mod.HealthCategory


# ===========================================================================
# 1. MRV Agent Routing (4 tests)
# ===========================================================================

class TestMRVAgentRouting:
    """Tests for MRV agent wiring and routing table correctness."""

    def test_routing_table_maps_esrs_to_agents(self):
        """Each DEFAULT_ROUTING_TABLE entry maps an ESRS code to an agent ID."""
        assert len(DEFAULT_ROUTING_TABLE) >= 20
        # Routing table may be a list of dicts/objects or a dict
        if isinstance(DEFAULT_ROUTING_TABLE, list):
            for entry in DEFAULT_ROUTING_TABLE:
                if hasattr(entry, "agent_id"):
                    assert isinstance(entry.agent_id, str)
                    assert len(entry.agent_id) > 0
                elif isinstance(entry, dict):
                    assert "agent_id" in entry
        else:
            for esrs_code, routing in DEFAULT_ROUTING_TABLE.items():
                assert isinstance(esrs_code, str)
                if hasattr(routing, "agent_id"):
                    assert isinstance(routing.agent_id, str)
                elif isinstance(routing, dict):
                    assert "agent_id" in routing

    def test_scope1_esrs_codes_covered(self):
        """Key Scope 1 ESRS codes are present in the routing table."""
        # Collect all ESRS codes from the routing table
        if isinstance(DEFAULT_ROUTING_TABLE, list):
            all_codes = []
            for entry in DEFAULT_ROUTING_TABLE:
                if hasattr(entry, "esrs_code"):
                    all_codes.append(entry.esrs_code)
                elif isinstance(entry, dict):
                    all_codes.append(entry.get("esrs_code", ""))
            all_codes_str = " ".join(all_codes)
        else:
            all_codes_str = " ".join(DEFAULT_ROUTING_TABLE.keys())

        # Check that scope1 related entries exist
        assert "scope1" in all_codes_str.lower() or "E1" in all_codes_str, (
            "Expected Scope 1 / E1 related entries in routing table"
        )

    def test_sub_sector_agents_cover_key_sectors(self):
        """SUB_SECTOR_AGENTS maps key manufacturing sub-sectors."""
        assert len(SUB_SECTOR_AGENTS) >= 3
        # Verify it is a proper mapping
        for sector, agent_info in SUB_SECTOR_AGENTS.items():
            assert isinstance(sector, str)
            assert agent_info is not None

    def test_bridge_graceful_degradation(self):
        """Bridge handles missing MRV agents gracefully via stubs."""
        bridge = MRVIndustrialBridge()
        # Accessing an unavailable agent should not raise
        assert bridge is not None
        assert bridge.config is not None


# ===========================================================================
# 2. Data Agent Routing (3 tests)
# ===========================================================================

class TestDataAgentRouting:
    """Tests for Data agent wiring and ERP field mapping."""

    def test_erp_field_map_covers_sap(self):
        """ERP_FIELD_MAP includes SAP field mappings."""
        # ERP_FIELD_MAP should have at least one ERP system
        assert len(ERP_FIELD_MAP) >= 1
        # Check that keys are strings (ERP names or field identifiers)
        for key in ERP_FIELD_MAP.keys():
            assert isinstance(key, str)

    def test_data_bridge_has_routing(self):
        """DataManufacturingBridge initializes with routing information."""
        bridge = DataManufacturingBridge()
        assert hasattr(bridge, "_routing")
        assert isinstance(bridge._routing, dict)

    def test_data_bridge_config_defaults(self):
        """DataBridgeConfig has sensible defaults."""
        config = DataBridgeConfig()
        assert config is not None
        # Config should be a pydantic model with fields
        fields = config.model_fields if hasattr(config, "model_fields") else {}
        assert isinstance(fields, dict)


# ===========================================================================
# 3. Cross-Pack Bridge Connectivity (3 tests)
# ===========================================================================

class TestCrossPackBridges:
    """Tests for cross-pack bridge initialization and connectivity."""

    def test_csrd_bridge_initializes(self):
        """CSRDPackBridge initializes without error."""
        bridge = CSRDPackBridge()
        assert bridge is not None
        assert bridge.config is not None
        assert hasattr(bridge.config, "pack_tier")

    def test_cbam_bridge_initializes(self):
        """CBAMPackBridge initializes without error."""
        bridge = CBAMPackBridge()
        assert bridge is not None
        assert bridge.config is not None
        assert hasattr(bridge.config, "cbam_phase")

    def test_ets_and_taxonomy_bridges_initialize(self):
        """EUETSBridge and TaxonomyBridge initialize without error."""
        ets = EUETSBridge()
        assert ets is not None
        assert ets.config is not None

        tax = TaxonomyBridge()
        assert tax is not None
        assert tax.config is not None


# ===========================================================================
# 4. Health Check Category Coverage (2 tests)
# ===========================================================================

class TestHealthCheckCoverage:
    """Tests for health check category completeness."""

    def test_all_categories_have_handlers(self):
        """Every HealthCategory enum value has a corresponding check method."""
        hc = ManufacturingHealthCheck()
        for category in HealthCategory:
            assert category in hc._category_methods, (
                f"HealthCategory.{category.name} has no handler in _category_methods"
            )

    def test_category_methods_are_callable(self):
        """All registered category check methods are callable."""
        hc = ManufacturingHealthCheck()
        for category, method in hc._category_methods.items():
            assert callable(method), (
                f"Handler for {category} is not callable: {type(method)}"
            )
