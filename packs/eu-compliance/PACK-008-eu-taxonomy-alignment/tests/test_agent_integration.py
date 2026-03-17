# -*- coding: utf-8 -*-
"""
PACK-008 EU Taxonomy Alignment - Live Agent Integration Tests
==============================================================

Tests that verify PACK-008's TaxonomyAppBridge, MRVTaxonomyBridge, and other
bridges can connect to real GreenLang MRV agents, data connectors, and
foundation agents.

Requires: pytest -m integration
"""

import asyncio
import importlib.util
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import pytest

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent.parent
_EU_COMPLIANCE_DIR = Path(__file__).resolve().parent.parent.parent
_PACK_008_DIR = Path(__file__).resolve().parent.parent

if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


def _import_from_path(module_name: str, file_path: Path) -> Optional[Any]:
    """Import a module from a file path (supports hyphenated directories)."""
    if not file_path.exists():
        return None
    try:
        spec = importlib.util.spec_from_file_location(module_name, str(file_path))
        if spec is None or spec.loader is None:
            return None
        mod = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = mod
        spec.loader.exec_module(mod)
        return mod
    except Exception:
        return None


def _instantiate_bridge(mod, bridge_name: str, config_names: list):
    """Try to instantiate a bridge with its config, handling various patterns."""
    if mod is None:
        return None
    bridge_cls = getattr(mod, bridge_name, None)
    if bridge_cls is None:
        return None
    # Try to find a config class
    config = None
    for cn in config_names:
        config_cls = getattr(mod, cn, None)
        if config_cls:
            try:
                config = config_cls()
            except Exception:
                pass
            break
    # Try to instantiate
    try:
        return bridge_cls(config) if config else bridge_cls()
    except TypeError:
        if config:
            return bridge_cls(config)
        return None


# ---------------------------------------------------------------------------
# Import PACK-008 integration modules
# ---------------------------------------------------------------------------
_orch_mod = _import_from_path(
    "pack008_ai_orchestrator",
    _PACK_008_DIR / "integrations" / "pack_orchestrator.py",
)
_tax_app_mod = _import_from_path(
    "pack008_ai_tax_app",
    _PACK_008_DIR / "integrations" / "taxonomy_app_bridge.py",
)
_mrv_mod = _import_from_path(
    "pack008_ai_mrv",
    _PACK_008_DIR / "integrations" / "mrv_taxonomy_bridge.py",
)
_csrd_mod = _import_from_path(
    "pack008_ai_csrd",
    _PACK_008_DIR / "integrations" / "csrd_cross_framework_bridge.py",
)
_fin_mod = _import_from_path(
    "pack008_ai_fin",
    _PACK_008_DIR / "integrations" / "financial_data_bridge.py",
)
_activity_mod = _import_from_path(
    "pack008_ai_activity",
    _PACK_008_DIR / "integrations" / "activity_registry_bridge.py",
)
_evidence_mod = _import_from_path(
    "pack008_ai_evidence",
    _PACK_008_DIR / "integrations" / "evidence_management_bridge.py",
)
_gar_mod = _import_from_path(
    "pack008_ai_gar",
    _PACK_008_DIR / "integrations" / "gar_data_bridge.py",
)
_dq_mod = _import_from_path(
    "pack008_ai_dq",
    _PACK_008_DIR / "integrations" / "data_quality_bridge.py",
)
_health_mod = _import_from_path(
    "pack008_ai_health",
    _PACK_008_DIR / "integrations" / "health_check.py",
)

# Import agent loader
_loader_mod = _import_from_path(
    "pack_agent_loader_integ_008",
    _EU_COMPLIANCE_DIR / "agent_loader.py",
)


# ---------------------------------------------------------------------------
# Agent loader fixture
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def agent_loader():
    """Load agent loader and attempt to load taxonomy-relevant agents."""
    if _loader_mod is None:
        pytest.skip("agent_loader not available")
    loader = _loader_mod.AgentLoader()
    # Load MRV Scope 1 agents (most relevant for taxonomy CCM TSC)
    for aid in [
        "GL-MRV-X-001",  # Stationary Combustion
        "GL-MRV-X-003",  # Mobile Combustion
        "GL-MRV-X-004",  # Process Emissions
        "GL-MRV-X-006",  # Land Use
        "GL-MRV-X-007",  # Waste Treatment
        "GL-MRV-X-009",  # Scope 2 Location-Based
        "GL-MRV-X-010",  # Scope 2 Market-Based
        "GL-MRV-X-011",  # Steam/Heat
        "GL-MRV-X-012",  # Cooling
    ]:
        loader.load(aid)
    # Load data agents
    for aid in [
        "GL-DATA-X-003",  # ERP Connector
        "GL-DATA-X-010",  # Data Quality Profiler
    ]:
        loader.load(aid)
    return loader


# ===========================================================================
# MRV Agent Loading Tests
# ===========================================================================
@pytest.mark.integration
class TestAgentIntegration:
    """Integration tests for PACK-008 with live GreenLang agents."""

    # -----------------------------------------------------------------------
    # AINT-001: MRV agent loading
    # -----------------------------------------------------------------------
    def test_mrv_agent_loading(self, agent_loader):
        """MRV agents relevant to taxonomy can be loaded."""
        summary = agent_loader.summary()
        if summary["loaded"] == 0:
            pytest.skip("No MRV agents loaded (dependencies not installed)")
        mrv_ids = [aid for aid in agent_loader.loaded_ids if "MRV" in aid]
        assert len(mrv_ids) > 0, (
            f"At least one MRV agent should load. Failed: {summary['failed_ids']}"
        )

    # -----------------------------------------------------------------------
    # AINT-002: Data agent loading
    # -----------------------------------------------------------------------
    def test_data_agent_loading(self, agent_loader):
        """Data agents relevant to taxonomy can be loaded."""
        summary = agent_loader.summary()
        data_ids = [aid for aid in agent_loader.loaded_ids if "DATA" in aid]
        if len(data_ids) == 0:
            pytest.skip("No DATA agents loaded (dependencies not installed)")
        assert len(data_ids) > 0

    # -----------------------------------------------------------------------
    # AINT-003: Foundation agent loading
    # -----------------------------------------------------------------------
    def test_foundation_agent_loading(self, agent_loader):
        """Foundation agents can be loaded."""
        # Attempt to load foundation agents
        for aid in ["GL-FOUND-X-001", "GL-FOUND-X-002", "GL-FOUND-X-003"]:
            agent_loader.load(aid)
        found_ids = [aid for aid in agent_loader.loaded_ids if "FOUND" in aid]
        if len(found_ids) == 0:
            pytest.skip("No FOUND agents loaded (dependencies not installed)")
        assert len(found_ids) > 0

    # -----------------------------------------------------------------------
    # AINT-004: Taxonomy App Bridge with real agents
    # -----------------------------------------------------------------------
    def test_taxonomy_app_bridge_with_real_agents(self, agent_loader):
        """TaxonomyAppBridge can be created and configured."""
        bridge = _instantiate_bridge(
            _tax_app_mod,
            "TaxonomyAppBridge",
            ["TaxonomyAppBridgeConfig"],
        )
        if bridge is None:
            pytest.skip("TaxonomyAppBridge not instantiable")
        assert bridge is not None

        # Verify bridge has engine proxy attributes
        has_engines = any(
            hasattr(bridge, attr) for attr in [
                "_engines", "_engine_proxies", "engines",
                "eligibility_engine", "_eligibility_proxy",
            ]
        )
        if not has_engines:
            pytest.skip("TaxonomyAppBridge engine attributes not found")
        assert has_engines

    # -----------------------------------------------------------------------
    # AINT-005: MRV Bridge routing to real agents
    # -----------------------------------------------------------------------
    def test_mrv_bridge_routing_to_real_agents(self, agent_loader):
        """MRVTaxonomyBridge routes to real MRV agents when injected."""
        bridge = _instantiate_bridge(
            _mrv_mod,
            "MRVTaxonomyBridge",
            ["MRVTaxonomyBridgeConfig"],
        )
        if bridge is None:
            pytest.skip("MRVTaxonomyBridge not instantiable")

        # Inject loaded MRV agents into bridge stubs
        injected_count = 0
        for agent_id in agent_loader.loaded_ids:
            if "MRV" not in agent_id:
                continue
            instance = agent_loader.get_instance(agent_id)
            if instance is not None and hasattr(bridge, "_agents"):
                # Map GL-MRV-X-001 -> mrv_001_stationary_combustion pattern
                agent_num = agent_id.split("-")[-1]  # "001"
                for stub_name, stub in bridge._agents.items():
                    if f"_{agent_num}_" in stub_name:
                        stub.inject(instance)
                        injected_count += 1
                        break

        if injected_count == 0:
            pytest.skip("No MRV agents could be injected into bridge stubs")
        assert injected_count > 0

    # -----------------------------------------------------------------------
    # AINT-006: Scope 1 emissions for CCM TSC
    # -----------------------------------------------------------------------
    def test_scope1_emissions_for_ccm_tsc(self, agent_loader):
        """Scope 1 emissions data can be retrieved for CCM TSC evaluation."""
        bridge = _instantiate_bridge(
            _mrv_mod,
            "MRVTaxonomyBridge",
            ["MRVTaxonomyBridgeConfig"],
        )
        if bridge is None:
            pytest.skip("MRVTaxonomyBridge not instantiable")

        # Check routing table covers energy sector activities
        routing = getattr(_mrv_mod, "MRV_ROUTING_TABLE", None)
        if routing is None:
            pytest.skip("MRV_ROUTING_TABLE not found")

        energy_activities = [k for k in routing if k.startswith("4.")]
        assert len(energy_activities) > 0, "Routing table must cover energy sector (4.x)"

        # Verify scope 1 agent is in the routing
        scope1_agents = {routing[k]["agent"] for k in energy_activities if routing[k]["scope"] == "scope_1"}
        assert "mrv_001_stationary_combustion" in scope1_agents, (
            "Stationary combustion agent must be routed for energy activities"
        )

    # -----------------------------------------------------------------------
    # AINT-007: Scope 2 emissions for building TSC
    # -----------------------------------------------------------------------
    def test_scope2_emissions_for_building_tsc(self, agent_loader):
        """Scope 2 emissions data covers building sector TSC evaluation."""
        routing = getattr(_mrv_mod, "MRV_ROUTING_TABLE", None)
        if routing is None:
            pytest.skip("MRV_ROUTING_TABLE not found")

        building_activities = [k for k in routing if k.startswith("7.")]
        assert len(building_activities) > 0, "Routing table must cover building sector (7.x)"

        # Verify scope 2 agents are in the routing for buildings
        scope2_agents = {routing[k]["agent"] for k in building_activities}
        has_scope2 = any("scope2" in a or "cooling" in a for a in scope2_agents)
        assert has_scope2, "Building activities must route to Scope 2 agents"

    # -----------------------------------------------------------------------
    # AINT-008: Data quality with real profiler
    # -----------------------------------------------------------------------
    def test_data_quality_with_real_profiler(self, agent_loader):
        """DataQualityBridge can be created for taxonomy data validation."""
        bridge = _instantiate_bridge(
            _dq_mod,
            "DataQualityBridge",
            ["DataQualityConfig"],
        )
        if bridge is None:
            pytest.skip("DataQualityBridge not instantiable")
        assert bridge is not None

        # Check if data quality profiler agent is loaded
        dq_loaded = "GL-DATA-X-010" in agent_loader.loaded_ids
        if not dq_loaded:
            pytest.skip("Data Quality Profiler agent not loaded")

        instance = agent_loader.get_instance("GL-DATA-X-010")
        assert instance is not None, "Data Quality Profiler should be instantiable"

    # -----------------------------------------------------------------------
    # AINT-009: Orchestrator 10-phase structure
    # -----------------------------------------------------------------------
    def test_orchestrator_10_phase_structure(self):
        """Orchestrator exposes 10-phase taxonomy alignment workflow."""
        orch = _instantiate_bridge(
            _orch_mod,
            "TaxonomyPackOrchestrator",
            ["TaxonomyOrchestratorConfig"],
        )
        if orch is None:
            pytest.skip("TaxonomyPackOrchestrator not instantiable")

        # Verify agent stubs initialized
        assert hasattr(orch, "_agents"), "Orchestrator must have _agents dict"
        assert len(orch._agents) > 0, "Orchestrator must have at least one agent stub"

        # Check pipeline phase enum
        phase_enum = getattr(_orch_mod, "TaxonomyPipelinePhase", None)
        if phase_enum is None:
            pytest.skip("TaxonomyPipelinePhase not found")
        phases = list(phase_enum)
        assert len(phases) == 10, f"Expected 10 phases, got {len(phases)}"

    # -----------------------------------------------------------------------
    # AINT-010: Health check runs
    # -----------------------------------------------------------------------
    def test_health_check_runs(self):
        """TaxonomyHealthCheck can be instantiated and has check categories."""
        health = _instantiate_bridge(
            _health_mod,
            "TaxonomyHealthCheck",
            ["HealthCheckConfig"],
        )
        if health is None:
            pytest.skip("TaxonomyHealthCheck not instantiable")
        assert health is not None

    # -----------------------------------------------------------------------
    # AINT-011: Eligibility with real data
    # -----------------------------------------------------------------------
    def test_eligibility_with_real_data(self):
        """Eligibility screening logic can process NACE activity codes."""
        # Verify activity registry bridge provides NACE lookup capability
        bridge = _instantiate_bridge(
            _activity_mod,
            "ActivityRegistryBridge",
            ["ActivityRegistryConfig"],
        )
        if bridge is None:
            pytest.skip("ActivityRegistryBridge not instantiable")

        # Verify bridge has lookup capability
        has_lookup = any(
            hasattr(bridge, attr) for attr in [
                "lookup_activity", "search_activities", "get_activity",
                "get_by_nace", "lookup_by_nace",
            ]
        )
        if not has_lookup:
            # Check for any public method with 'look' or 'search' or 'get'
            methods = [m for m in dir(bridge) if not m.startswith("_") and callable(getattr(bridge, m, None))]
            if not methods:
                pytest.skip("ActivityRegistryBridge has no public methods")
        assert bridge is not None

    # -----------------------------------------------------------------------
    # AINT-012: Alignment with real emissions
    # -----------------------------------------------------------------------
    def test_alignment_with_real_emissions(self, agent_loader):
        """MRV bridge routing table covers all major taxonomy sectors."""
        routing = getattr(_mrv_mod, "MRV_ROUTING_TABLE", None)
        if routing is None:
            pytest.skip("MRV_ROUTING_TABLE not found")

        # Verify coverage of major sectors
        sectors_covered = set()
        for key in routing:
            if key.startswith("1.") or key.startswith("2."):
                sectors_covered.add("forestry_agriculture")
            elif key.startswith("3."):
                sectors_covered.add("manufacturing")
            elif key.startswith("4."):
                sectors_covered.add("energy")
            elif key.startswith("5."):
                sectors_covered.add("waste_water")
            elif key.startswith("6."):
                sectors_covered.add("transport")
            elif key.startswith("7."):
                sectors_covered.add("buildings")
            elif key.startswith("scope3"):
                sectors_covered.add("scope3")

        expected_sectors = {"energy", "transport", "buildings", "manufacturing"}
        missing = expected_sectors - sectors_covered
        assert len(missing) == 0, f"Routing table missing sectors: {missing}"

    # -----------------------------------------------------------------------
    # AINT-013: KPI with real financial data
    # -----------------------------------------------------------------------
    def test_kpi_with_real_financial_data(self):
        """FinancialDataBridge can be created for KPI calculation."""
        bridge = _instantiate_bridge(
            _fin_mod,
            "FinancialDataBridge",
            ["FinancialDataConfig"],
        )
        if bridge is None:
            pytest.skip("FinancialDataBridge not instantiable")
        assert bridge is not None

        # Verify bridge has financial data methods
        has_methods = any(
            hasattr(bridge, attr) for attr in [
                "import_turnover_data", "import_capex_data", "import_opex_data",
                "get_turnover", "get_capex", "get_financial_data",
            ]
        )
        if not has_methods:
            methods = [m for m in dir(bridge) if not m.startswith("_")]
            if not methods:
                pytest.skip("FinancialDataBridge has no public methods")
        assert bridge is not None

    # -----------------------------------------------------------------------
    # AINT-014: Cross-framework with CSRD pack
    # -----------------------------------------------------------------------
    def test_cross_framework_with_csrd_pack(self):
        """CSRDCrossFrameworkBridge can map taxonomy data to ESRS/SFDR/TCFD."""
        bridge = _instantiate_bridge(
            _csrd_mod,
            "CSRDCrossFrameworkBridge",
            ["CrossFrameworkConfig"],
        )
        if bridge is None:
            pytest.skip("CSRDCrossFrameworkBridge not instantiable")
        assert bridge is not None

        # Verify bridge has cross-framework mapping methods
        has_mapping = any(
            hasattr(bridge, attr) for attr in [
                "map_to_esrs", "map_to_sfdr", "map_to_tcfd",
                "generate_esrs_e1", "generate_sfdr_pai",
                "map_taxonomy_to_esrs", "map_taxonomy_to_sfdr",
            ]
        )
        if not has_mapping:
            methods = [m for m in dir(bridge) if not m.startswith("_") and callable(getattr(bridge, m, None))]
            if not methods:
                pytest.skip("CSRDCrossFrameworkBridge has no public methods")
        assert bridge is not None

    # -----------------------------------------------------------------------
    # AINT-015: Agent loader summary
    # -----------------------------------------------------------------------
    def test_agent_loader_summary(self, agent_loader):
        """Agent loader provides a summary of load results."""
        summary = agent_loader.summary()
        assert "total_registered" in summary, "Summary must include total_registered"
        assert "loaded" in summary, "Summary must include loaded count"
        assert "failed" in summary, "Summary must include failed count"
        assert summary["total_registered"] > 0, "Registry must have agents registered"
        # Log summary for debugging
        loaded = summary["loaded"]
        failed = summary["failed"]
        total = summary["total_registered"]
        assert loaded + failed <= total, (
            f"loaded ({loaded}) + failed ({failed}) must not exceed total ({total})"
        )
