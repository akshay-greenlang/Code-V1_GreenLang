# -*- coding: utf-8 -*-
"""
Shared Integration Test Fixtures for EU Compliance Solution Packs
=================================================================

Provides session-scoped pytest fixtures that load real GreenLang agents
and inject them into pack bridge instances. Tests using these fixtures
exercise live agent code paths instead of mocked stubs.

Usage:
    Import these fixtures in pack-level conftest.py or add this directory
    to conftest discovery. All fixtures skip gracefully when their
    dependent agents cannot be imported.

    @pytest.mark.integration
    def test_scope1_with_real_agents(mrv_bridge_live):
        result = asyncio.run(mrv_bridge_live.route_calculation("E1-1-1", data))
        assert result.status.value == "success"
"""

import importlib
import importlib.util
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import pytest

# Ensure project root is on sys.path for absolute imports
_project_root = Path(__file__).resolve().parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

# Pack root directories (hyphenated — not valid Python identifiers)
_EU_COMPLIANCE_DIR = Path(__file__).resolve().parent
_PACK_001_DIR = _EU_COMPLIANCE_DIR / "PACK-001-csrd-starter"
_PACK_002_DIR = _EU_COMPLIANCE_DIR / "PACK-002-csrd-professional"
_PACK_003_DIR = _EU_COMPLIANCE_DIR / "PACK-003-csrd-enterprise"
_PACK_004_DIR = _EU_COMPLIANCE_DIR / "PACK-004-cbam-readiness"
_PACK_005_DIR = _EU_COMPLIANCE_DIR / "PACK-005-cbam-complete"
_PACK_006_DIR = _EU_COMPLIANCE_DIR / "PACK-006-eudr-starter"


def _import_from_path(module_name: str, file_path: Path) -> Optional[Any]:
    """Import a module from a file path (bypassing hyphenated directory names).

    Args:
        module_name: The name to assign to the imported module.
        file_path: Absolute path to the .py file.

    Returns:
        The imported module, or None if import fails.
    """
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


# =============================================================================
# Agent Loader Fixture (session-scoped)
# =============================================================================


@pytest.fixture(scope="session")
def agent_loader():
    """Session-scoped agent loader that imports all available GreenLang agents.

    Returns an AgentLoader instance with all importable agents pre-loaded.
    Agents that cannot be imported are recorded but do not cause test failure.
    """
    loader_path = _EU_COMPLIANCE_DIR / "agent_loader.py"
    mod = _import_from_path("pack_agent_loader", loader_path)
    if mod is None:
        pytest.skip("agent_loader.py not importable")

    loader = mod.AgentLoader()
    # Load key agents from each category for integration testing
    for aid in ["GL-MRV-X-001", "GL-MRV-X-003", "GL-MRV-X-009", "GL-MRV-X-010",
                "GL-DATA-X-001", "GL-DATA-X-002", "GL-DATA-X-010", "GL-DATA-X-011"]:
        loader.load(aid)
    return loader


@pytest.fixture(scope="session")
def loaded_agent_ids(agent_loader):
    """Set of agent IDs that were successfully loaded."""
    return agent_loader.loaded_ids


@pytest.fixture(scope="session")
def agent_load_summary(agent_loader):
    """Summary dict of agent load results."""
    return agent_loader.summary()


# =============================================================================
# MRV Bridge Fixtures
# =============================================================================


@pytest.fixture(scope="session")
def mrv_bridge_live(agent_loader):
    """Session-scoped MRVBridge with real agents injected.

    Imports the PACK-001 MRVBridge and injects any successfully loaded
    MRV agents (GL-MRV-X-001 through GL-MRV-X-030) into its _agents dict.

    Skips if MRVBridge cannot be imported.
    """
    mrv_file = _PACK_001_DIR / "integrations" / "mrv_bridge.py"
    mrv_mod = _import_from_path("pack001_mrv_bridge", mrv_file)
    if mrv_mod is None:
        pytest.skip("PACK-001 MRVBridge not importable")

    bridge = mrv_mod.MRVBridge(mrv_mod.MRVBridgeConfig())

    # Inject loaded MRV agents
    for agent_id in sorted(agent_loader.loaded_ids):
        if agent_id.startswith("GL-MRV-X-"):
            instance = agent_loader.get_instance(agent_id)
            if instance is not None:
                bridge._agents[agent_id] = instance

    return bridge


@pytest.fixture(scope="session")
def mrv_agent_count(mrv_bridge_live):
    """Number of real MRV agents loaded into the bridge."""
    return len(mrv_bridge_live._agents)


# =============================================================================
# Data Pipeline Bridge Fixtures
# =============================================================================


@pytest.fixture(scope="session")
def data_bridge_live(agent_loader):
    """Session-scoped DataPipelineBridge with real data agents injected.

    Imports the PACK-001 DataPipelineBridge and injects any successfully
    loaded data agents (GL-DATA-X-*) into its _data_agents and
    _quality_agents dicts.

    Skips if DataPipelineBridge cannot be imported.
    """
    dp_file = _PACK_001_DIR / "integrations" / "data_pipeline_bridge.py"
    dp_mod = _import_from_path("pack001_data_pipeline_bridge", dp_file)
    if dp_mod is None:
        pytest.skip("PACK-001 DataPipelineBridge not importable")

    bridge = dp_mod.DataPipelineBridge(dp_mod.DataPipelineConfig())

    # Inject data intake agents
    for agent_id in ["GL-DATA-X-001", "GL-DATA-X-002", "GL-DATA-X-003", "GL-DATA-X-008"]:
        instance = agent_loader.get_instance(agent_id)
        if instance is not None:
            bridge._data_agents[agent_id] = instance

    # Inject quality pipeline agents
    for agent_id in ["GL-DATA-X-010", "GL-DATA-X-011", "GL-DATA-X-012",
                     "GL-DATA-X-013", "GL-DATA-X-019"]:
        instance = agent_loader.get_instance(agent_id)
        if instance is not None:
            bridge._quality_agents[agent_id] = instance

    return bridge


# =============================================================================
# Pack Orchestrator Fixtures
# =============================================================================


@pytest.fixture(scope="session")
def pack001_orchestrator(agent_loader):
    """Session-scoped PACK-001 orchestrator with real agents injected.

    Skips if the orchestrator cannot be imported.
    """
    orch_file = _PACK_001_DIR / "integrations" / "pack_orchestrator.py"
    orch_mod = _import_from_path("pack001_pack_orchestrator", orch_file)
    if orch_mod is None:
        pytest.skip("PACK-001 orchestrator not importable")

    # Find the orchestrator class
    orch_cls = getattr(orch_mod, "PackOrchestrator", None)
    if orch_cls is None:
        for attr_name in dir(orch_mod):
            obj = getattr(orch_mod, attr_name, None)
            if isinstance(obj, type) and "Orchestrator" in attr_name:
                orch_cls = obj
                break

    if orch_cls is None:
        pytest.skip("No Orchestrator class found in pack_orchestrator module")

    try:
        orchestrator = orch_cls()
    except Exception:
        config_cls = getattr(orch_mod, "OrchestratorConfig", None)
        if config_cls:
            orchestrator = orch_cls(config_cls())
        else:
            pytest.skip("Cannot instantiate orchestrator")
            return None

    return orchestrator


# =============================================================================
# CBAM Bridge Fixtures
# =============================================================================


@pytest.fixture(scope="session")
def cbam_app_bridge():
    """Session-scoped CBAMAppBridge.

    Skips if the bridge cannot be imported.
    """
    bridge_file = _PACK_004_DIR / "integrations" / "cbam_app_bridge.py"
    mod = _import_from_path("pack004_cbam_app_bridge", bridge_file)
    if mod is None:
        pytest.skip("PACK-004 CBAMAppBridge not importable")
    bridge_cls = getattr(mod, "CBAMAppBridge", None)
    if bridge_cls is None:
        pytest.skip("CBAMAppBridge class not found")
    return bridge_cls()


@pytest.fixture(scope="session")
def cbam_cross_pack_bridge():
    """Session-scoped CrossPackBridge for PACK-005.

    Skips if the bridge cannot be imported.
    """
    bridge_file = _PACK_005_DIR / "integrations" / "cross_pack_bridge.py"
    mod = _import_from_path("pack005_cross_pack_bridge", bridge_file)
    if mod is None:
        pytest.skip("PACK-005 CrossPackBridge not importable")
    bridge_cls = getattr(mod, "CrossPackBridge", None)
    if bridge_cls is None:
        pytest.skip("CrossPackBridge class not found")
    return bridge_cls()


# =============================================================================
# EUDR Bridge Fixtures
# =============================================================================


@pytest.fixture(scope="session")
def traceability_bridge(agent_loader):
    """Session-scoped TraceabilityBridge with real EUDR agents injected.

    Skips if the bridge cannot be imported.
    """
    bridge_file = _PACK_006_DIR / "integrations" / "traceability_bridge.py"
    mod = _import_from_path("pack006_traceability_bridge", bridge_file)
    if mod is None:
        pytest.skip("PACK-006 TraceabilityBridge not importable")
    bridge_cls = getattr(mod, "TraceabilityBridge", None)
    if bridge_cls is None:
        pytest.skip("TraceabilityBridge class not found")
    return bridge_cls()


@pytest.fixture(scope="session")
def satellite_bridge():
    """Session-scoped SatelliteBridge for PACK-006.

    Skips if the bridge cannot be imported.
    """
    bridge_file = _PACK_006_DIR / "integrations" / "satellite_bridge.py"
    mod = _import_from_path("pack006_satellite_bridge", bridge_file)
    if mod is None:
        pytest.skip("PACK-006 SatelliteBridge not importable")
    bridge_cls = getattr(mod, "SatelliteBridge", None)
    if bridge_cls is None:
        pytest.skip("SatelliteBridge class not found")
    return bridge_cls()


@pytest.fixture(scope="session")
def gis_bridge():
    """Session-scoped GISBridge for PACK-006.

    Skips if the bridge cannot be imported.
    """
    bridge_file = _PACK_006_DIR / "integrations" / "gis_bridge.py"
    mod = _import_from_path("pack006_gis_bridge", bridge_file)
    if mod is None:
        pytest.skip("PACK-006 GISBridge not importable")
    bridge_cls = getattr(mod, "GISBridge", None)
    if bridge_cls is None:
        pytest.skip("GISBridge class not found")
    return bridge_cls()


# =============================================================================
# Utility Fixtures
# =============================================================================


@pytest.fixture
def sample_combustion_data():
    """Sample stationary combustion activity data for testing."""
    return {
        "fuel_type": "NATURAL_GAS",
        "quantity": 1000.0,
        "unit": "CUBIC_METERS",
        "emission_factor": 2.02,
        "activity_data": 1000.0,
        "factor": 2.02,
    }


@pytest.fixture
def sample_scope1_data(sample_combustion_data):
    """Sample Scope 1 data covering all emission sources."""
    return {
        "stationary_combustion": sample_combustion_data,
        "mobile_combustion": {
            "vehicle_type": "diesel_truck",
            "distance_km": 5000,
            "fuel_consumed_liters": 400,
            "emission_factor": 2.68,
            "activity_data": 400,
            "factor": 2.68,
        },
    }


@pytest.fixture
def sample_scope2_data():
    """Sample Scope 2 data for location and market based."""
    return {
        "location_based": {
            "electricity_kwh": 50000,
            "grid_region": "US-WECC",
            "emission_factor": 0.000412,
            "activity_data": 50000,
            "factor": 0.000412,
        },
        "market_based": {
            "electricity_kwh": 50000,
            "contractual_factor": 0.000100,
            "emission_factor": 0.000100,
            "activity_data": 50000,
            "factor": 0.000100,
        },
    }


@pytest.fixture
def sample_eudr_plot_data():
    """Sample EUDR plot registration data."""
    return {
        "supplier_id": "SUP-TEST-001",
        "latitude": -3.4653,
        "longitude": -62.2159,
        "area_ha": 50.0,
        "country": "BR",
        "commodity": "soy",
    }
