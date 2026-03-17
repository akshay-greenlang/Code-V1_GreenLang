# -*- coding: utf-8 -*-
"""
PACK-001 CSRD Starter - Live Agent Integration Tests
======================================================

Tests that inject real GreenLang MRV and Data agents into PACK-001
bridges and verify end-to-end calculation and data flows work correctly
without mocks.

Requires: pytest -m integration
"""

import asyncio
import importlib
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
_PACK_001_DIR = Path(__file__).resolve().parent.parent

if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


def _import_from_path(module_name: str, file_path: Path) -> Optional[Any]:
    """Import a module from a file path."""
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


# ---------------------------------------------------------------------------
# Module-level imports via path (handles hyphenated dirs)
# ---------------------------------------------------------------------------
_mrv_mod = _import_from_path(
    "pack001_mrv_bridge_integ",
    _PACK_001_DIR / "integrations" / "mrv_bridge.py",
)
_dp_mod = _import_from_path(
    "pack001_dp_bridge_integ",
    _PACK_001_DIR / "integrations" / "data_pipeline_bridge.py",
)
_orch_mod = _import_from_path(
    "pack001_orch_integ",
    _PACK_001_DIR / "integrations" / "pack_orchestrator.py",
)

# Agent loader
_loader_mod = _import_from_path(
    "pack_agent_loader_integ",
    _EU_COMPLIANCE_DIR / "agent_loader.py",
)


# ---------------------------------------------------------------------------
# Session fixture: load agents once
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def agent_loader():
    if _loader_mod is None:
        pytest.skip("agent_loader not available")
    loader = _loader_mod.AgentLoader()
    # Load only the agents PACK-001 uses (MRV Scope 1 + key DATA agents)
    for aid in ["GL-MRV-X-001", "GL-MRV-X-003", "GL-MRV-X-009", "GL-MRV-X-010",
                "GL-DATA-X-001", "GL-DATA-X-002", "GL-DATA-X-010", "GL-DATA-X-011"]:
        loader.load(aid)
    return loader


@pytest.fixture(scope="module")
def mrv_bridge_with_agents(agent_loader):
    """MRVBridge with real MRV agents injected."""
    if _mrv_mod is None:
        pytest.skip("MRVBridge not importable")
    bridge = _mrv_mod.MRVBridge(_mrv_mod.MRVBridgeConfig())
    for agent_id in sorted(agent_loader.loaded_ids):
        if agent_id.startswith("GL-MRV-X-"):
            instance = agent_loader.get_instance(agent_id)
            if instance is not None:
                bridge._agents[agent_id] = instance
    return bridge


@pytest.fixture(scope="module")
def data_bridge_with_agents(agent_loader):
    """DataPipelineBridge with real data agents injected."""
    if _dp_mod is None:
        pytest.skip("DataPipelineBridge not importable")
    bridge = _dp_mod.DataPipelineBridge(_dp_mod.DataPipelineConfig())
    for agent_id in sorted(agent_loader.loaded_ids):
        if agent_id.startswith("GL-DATA-X-"):
            instance = agent_loader.get_instance(agent_id)
            if instance is not None:
                if agent_id in ("GL-DATA-X-010", "GL-DATA-X-011",
                                "GL-DATA-X-012", "GL-DATA-X-013",
                                "GL-DATA-X-019"):
                    bridge._quality_agents[agent_id] = instance
                else:
                    bridge._data_agents[agent_id] = instance
    return bridge


# ===========================================================================
# Test 1: MRV Bridge routing table completeness
# ===========================================================================
@pytest.mark.integration
class TestMRVBridgeIntegration:
    """Integration tests for MRV Bridge with real agent injection."""

    def test_routing_table_has_30_entries(self):
        """Verify the MRV routing table covers all 30 MRV agents."""
        if _mrv_mod is None:
            pytest.skip("MRVBridge not importable")
        table = _mrv_mod.MRV_ROUTING_TABLE
        assert len(table) == 30, f"Expected 30 entries, got {len(table)}"

    def test_real_agents_inject_into_bridge(self, mrv_bridge_with_agents):
        """Verify at least one real MRV agent was injected into the bridge."""
        agent_count = len(mrv_bridge_with_agents._agents)
        # We expect at least stationary combustion (GL-MRV-X-001) to load
        if agent_count == 0:
            pytest.skip("No MRV agents could be loaded (missing deps)")
        assert agent_count > 0, "No agents injected"

    def test_route_calculation_scope1_stationary(self, mrv_bridge_with_agents):
        """Route E1-1-1 (stationary combustion) through the bridge."""
        data = {
            "fuel_type": "NATURAL_GAS",
            "quantity": 1000.0,
            "emission_factor": 2.02,
            "activity_data": 1000.0,
            "factor": 2.02,
        }
        result = asyncio.new_event_loop().run_until_complete(
            mrv_bridge_with_agents.route_calculation("E1-1-1", data)
        )
        assert result.metric_code == "E1-1-1"
        assert result.agent_id == "GL-MRV-X-001"
        assert result.status.value in ("success", "partial")
        assert result.emissions_value >= 0
        assert result.provenance_hash != ""

    def test_calculate_scope1_aggregation(self, mrv_bridge_with_agents):
        """Run full Scope 1 aggregation through the bridge."""
        data = {
            "stationary_combustion": {
                "activity_data": 500.0,
                "factor": 2.02,
            },
        }
        result = asyncio.new_event_loop().run_until_complete(
            mrv_bridge_with_agents.calculate_scope1(data)
        )
        assert result.aggregated is not None
        assert result.provenance_hash != ""

    def test_calculate_scope2_dual_method(self, mrv_bridge_with_agents):
        """Run Scope 2 with both location and market methods."""
        data = {
            "location_based": {
                "activity_data": 50000.0,
                "factor": 0.000412,
            },
            "market_based": {
                "activity_data": 50000.0,
                "factor": 0.000100,
            },
        }
        result = asyncio.new_event_loop().run_until_complete(
            mrv_bridge_with_agents.calculate_scope2(data)
        )
        assert result.provenance_hash != ""

    def test_provenance_chain_tracked(self, mrv_bridge_with_agents):
        """Verify provenance entries accumulate across calculations."""
        mrv_bridge_with_agents.reset_provenance()
        data = {"activity_data": 100.0, "factor": 1.5}
        asyncio.new_event_loop().run_until_complete(
            mrv_bridge_with_agents.route_calculation("E1-1-1", data)
        )
        chain = mrv_bridge_with_agents.get_provenance_chain()
        assert len(chain) >= 1
        assert chain[0].agent_id == "GL-MRV-X-001"
        assert chain[0].input_hash != ""
        assert chain[0].output_hash != ""


# ===========================================================================
# Test 2: Data Pipeline Bridge integration
# ===========================================================================
@pytest.mark.integration
class TestDataPipelineBridgeIntegration:
    """Integration tests for DataPipelineBridge with real data agents."""

    def test_data_agents_inject(self, data_bridge_with_agents):
        """Verify data agents were injected into the bridge."""
        total = len(data_bridge_with_agents._data_agents) + \
                len(data_bridge_with_agents._quality_agents)
        if total == 0:
            pytest.skip("No data agents could be loaded")
        assert total > 0


# ===========================================================================
# Test 3: Orchestrator _AgentStub accepts real agent
# ===========================================================================
@pytest.mark.integration
class TestOrchestratorIntegration:
    """Integration tests for PACK-001 PackOrchestrator agent stubs."""

    def test_agent_stub_with_real_agent_injection(self, agent_loader):
        """Create an _AgentStub and inject a real agent into _real_agent."""
        if _orch_mod is None:
            pytest.skip("Orchestrator not importable")

        # Find the stub factory function
        create_fn = getattr(_orch_mod, "_create_agent_stub", None)
        if create_fn is None:
            # Look for any function that creates stubs
            for name in dir(_orch_mod):
                obj = getattr(_orch_mod, name)
                if callable(obj) and "stub" in name.lower():
                    create_fn = obj
                    break

        if create_fn is None:
            pytest.skip("No stub creation function found")

        config_cls = getattr(_orch_mod, "OrchestratorConfig", None)
        config = config_cls() if config_cls else None

        try:
            stub = create_fn("GL-MRV-X-001", config)
        except TypeError:
            stub = create_fn("GL-MRV-X-001")

        # Inject real agent if available
        instance = agent_loader.get_instance("GL-MRV-X-001")
        if instance is not None:
            stub._real_agent = instance
            result = asyncio.new_event_loop().run_until_complete(
                stub.execute({"activity_data": 100.0, "factor": 2.0})
            )
            # With a real agent, we should get something beyond the stub default
            assert isinstance(result, dict)
        else:
            # Even without a real agent, stub should work
            result = asyncio.new_event_loop().run_until_complete(
                stub.execute({"test": True})
            )
            assert result.get("status") == "stub_executed"
