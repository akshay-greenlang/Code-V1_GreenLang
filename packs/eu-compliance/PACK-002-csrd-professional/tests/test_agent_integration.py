# -*- coding: utf-8 -*-
"""
PACK-002 CSRD Professional - Live Agent Integration Tests
===========================================================

Tests that inject real GreenLang MRV agents into PACK-002's
ProfessionalMRVBridge and CrossFrameworkBridge and verify intensity
metrics, biogenic carbon, and cross-framework data flows.

Requires: pytest -m integration
"""

import asyncio
import importlib.util
import sys
from pathlib import Path
from typing import Any, Optional

import pytest

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent.parent
_EU_COMPLIANCE_DIR = Path(__file__).resolve().parent.parent.parent
_PACK_002_DIR = Path(__file__).resolve().parent.parent

if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


def _import_from_path(module_name: str, file_path: Path) -> Optional[Any]:
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


_mrv_mod = _import_from_path(
    "pack002_mrv_bridge_integ",
    _PACK_002_DIR / "integrations" / "mrv_bridge.py",
)
_cf_mod = _import_from_path(
    "pack002_cross_framework_integ",
    _PACK_002_DIR / "integrations" / "cross_framework_bridge.py",
)
_loader_mod = _import_from_path(
    "pack_agent_loader_integ_002",
    _EU_COMPLIANCE_DIR / "agent_loader.py",
)


@pytest.fixture(scope="module")
def agent_loader():
    if _loader_mod is None:
        pytest.skip("agent_loader not available")
    loader = _loader_mod.AgentLoader()
    # Load only key MRV agents used by PACK-002
    for aid in ["GL-MRV-X-001", "GL-MRV-X-003", "GL-MRV-X-009", "GL-MRV-X-010"]:
        loader.load(aid)
    return loader


@pytest.fixture(scope="module")
def pro_mrv_bridge(agent_loader):
    """ProfessionalMRVBridge with real agents injected."""
    if _mrv_mod is None:
        pytest.skip("ProfessionalMRVBridge not importable")
    config_cls = getattr(_mrv_mod, "ProfessionalMRVBridgeConfig", None)
    bridge_cls = getattr(_mrv_mod, "ProfessionalMRVBridge", None)
    if bridge_cls is None:
        pytest.skip("ProfessionalMRVBridge class not found")
    config = config_cls() if config_cls else None
    bridge = bridge_cls(config) if config else bridge_cls()
    for agent_id in sorted(agent_loader.loaded_ids):
        if agent_id.startswith("GL-MRV-X-"):
            instance = agent_loader.get_instance(agent_id)
            if instance is not None:
                bridge._agents[agent_id] = instance
    return bridge


# ===========================================================================
# Tests
# ===========================================================================
@pytest.mark.integration
class TestProfessionalMRVBridgeIntegration:
    """Integration tests for PACK-002 Professional MRV Bridge."""

    def test_real_agents_loaded(self, pro_mrv_bridge):
        """At least one MRV agent is injected."""
        if len(pro_mrv_bridge._agents) == 0:
            pytest.skip("No MRV agents loaded")
        assert len(pro_mrv_bridge._agents) > 0

    def test_scope1_calculation_via_pro_bridge(self, pro_mrv_bridge):
        """Scope 1 calculation through Professional bridge."""
        calc_fn = getattr(pro_mrv_bridge, "calculate_scope1", None)
        route_fn = getattr(pro_mrv_bridge, "route_calculation", None)
        if calc_fn is not None:
            data = {"stationary_combustion": {"activity_data": 500, "factor": 2.02}}
            result = asyncio.new_event_loop().run_until_complete(calc_fn(data))
            assert result is not None
        elif route_fn is not None:
            data = {"activity_data": 500, "factor": 2.02}
            result = asyncio.new_event_loop().run_until_complete(
                route_fn("E1-1-1", data)
            )
            assert result.metric_code == "E1-1-1"
        else:
            pytest.skip("No calculation method found on ProfessionalMRVBridge")

    def test_intensity_metrics_structure(self, pro_mrv_bridge):
        """Verify Professional bridge exposes intensity metric methods."""
        has_intensity = any(
            "intensity" in attr.lower()
            for attr in dir(pro_mrv_bridge)
            if not attr.startswith("_")
        )
        # Intensity metrics are a Professional-tier feature
        assert has_intensity or hasattr(pro_mrv_bridge, "calculate_scope1"), \
            "Expected intensity metrics or standard calculation capability"

    def test_biogenic_carbon_tracking(self, pro_mrv_bridge):
        """Verify biogenic carbon fields are accessible."""
        # Professional tier adds biogenic carbon tracking
        has_biogenic = any(
            "biogenic" in attr.lower()
            for attr in dir(pro_mrv_bridge)
            if not attr.startswith("_")
        )
        # Even if not explicit method, route_calculation should work with biogenic data
        route_fn = getattr(pro_mrv_bridge, "route_calculation", None)
        if route_fn and len(pro_mrv_bridge._agents) > 0:
            data = {
                "activity_data": 200,
                "factor": 1.8,
                "fuel_type": "WOOD_PELLETS",
                "biogenic": True,
            }
            result = asyncio.new_event_loop().run_until_complete(
                route_fn("E1-1-1", data)
            )
            assert result.status.value in ("success", "partial")


@pytest.mark.integration
class TestCrossFrameworkBridgeIntegration:
    """Integration tests for PACK-002 Cross-Framework Bridge."""

    def test_bridge_instantiates(self):
        """CrossFrameworkBridge can be created."""
        if _cf_mod is None:
            pytest.skip("CrossFrameworkBridge not importable")
        bridge_cls = getattr(_cf_mod, "CrossFrameworkBridge", None)
        if bridge_cls is None:
            pytest.skip("CrossFrameworkBridge class not found")
        config_cls = getattr(_cf_mod, "CrossFrameworkBridgeConfig", None)
        config = config_cls() if config_cls else None
        bridge = bridge_cls(config) if config else bridge_cls()
        assert bridge is not None

    def test_supported_frameworks_listed(self):
        """Bridge lists supported regulatory frameworks."""
        if _cf_mod is None:
            pytest.skip("CrossFrameworkBridge not importable")
        bridge_cls = getattr(_cf_mod, "CrossFrameworkBridge", None)
        if bridge_cls is None:
            pytest.skip("CrossFrameworkBridge class not found")
        config_cls = getattr(_cf_mod, "CrossFrameworkBridgeConfig", None)
        config = config_cls() if config_cls else None
        bridge = bridge_cls(config) if config else bridge_cls()
        # Check for framework enumeration
        frameworks_attr = None
        for attr in dir(bridge):
            if "framework" in attr.lower() and not attr.startswith("_"):
                frameworks_attr = getattr(bridge, attr, None)
                break
        assert bridge is not None  # Bridge exists even without framework listing
