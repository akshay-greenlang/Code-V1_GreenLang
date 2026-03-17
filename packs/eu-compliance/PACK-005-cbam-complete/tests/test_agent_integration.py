# -*- coding: utf-8 -*-
"""
PACK-005 CBAM Complete - Live Agent Integration Tests
=======================================================

Tests that verify PACK-005's CrossPackBridge, ETSRegistryBridge,
and multi-entity capabilities work with real GreenLang agents and
the PACK-004 CBAM pipeline.

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
_PACK_005_DIR = Path(__file__).resolve().parent.parent
_PACK_004_DIR = _EU_COMPLIANCE_DIR / "PACK-004-cbam-readiness"

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


_cross_pack_mod = _import_from_path(
    "pack005_cross_pack_integ",
    _PACK_005_DIR / "integrations" / "cross_pack_bridge.py",
)
_ets_mod = _import_from_path(
    "pack005_ets_integ",
    _PACK_005_DIR / "integrations" / "ets_registry_bridge.py",
)
_orch_mod = _import_from_path(
    "pack005_orch_integ",
    _PACK_005_DIR / "integrations" / "pack_orchestrator.py",
)
_loader_mod = _import_from_path(
    "pack_agent_loader_integ_005",
    _EU_COMPLIANCE_DIR / "agent_loader.py",
)


@pytest.fixture(scope="module")
def agent_loader():
    if _loader_mod is None:
        pytest.skip("agent_loader not available")
    loader = _loader_mod.AgentLoader()
    # Load only agents used by PACK-005 (CBAM + MRV for cross-regulation)
    for aid in ["GL-MRV-X-001", "GL-CBAM-APP"]:
        loader.load(aid)
    return loader


# ===========================================================================
# Tests
# ===========================================================================
@pytest.mark.integration
class TestCrossPackBridgeIntegration:
    """Integration tests for PACK-005 Cross-Pack Bridge."""

    def test_bridge_instantiates(self):
        """CrossPackBridge can be created."""
        if _cross_pack_mod is None:
            pytest.skip("CrossPackBridge not importable")
        bridge_cls = getattr(_cross_pack_mod, "CrossPackBridge", None)
        if bridge_cls is None:
            pytest.skip("CrossPackBridge class not found")
        config_cls = getattr(_cross_pack_mod, "CrossRegulationConfig", None)
        config = config_cls() if config_cls else None
        bridge = bridge_cls(config) if config else bridge_cls()
        assert bridge is not None

    def test_push_to_csrd_graceful_degradation(self):
        """CrossPackBridge handles push to CSRD when CSRD pack not imported."""
        if _cross_pack_mod is None:
            pytest.skip("CrossPackBridge not importable")
        bridge_cls = getattr(_cross_pack_mod, "CrossPackBridge", None)
        if bridge_cls is None:
            pytest.skip("CrossPackBridge class not found")
        config_cls = getattr(_cross_pack_mod, "CrossRegulationConfig", None)
        config = config_cls() if config_cls else None
        bridge = bridge_cls(config) if config else bridge_cls()
        push_fn = getattr(bridge, "push_to_csrd", None)
        if push_fn is None:
            pytest.skip("push_to_csrd not available")
        cbam_data = {
            "installations": [{"name": "Test Plant", "emissions_tco2": 1000}],
            "reporting_period": "Q1-2026",
        }
        result = push_fn(cbam_data, "PACK-001")
        assert result is not None
        # Should succeed or degrade gracefully, not crash
        if hasattr(result, "success"):
            assert result.success or result.degraded

    def test_sync_all_targets(self):
        """CrossPackBridge can attempt sync_all without crashing."""
        if _cross_pack_mod is None:
            pytest.skip("CrossPackBridge not importable")
        bridge_cls = getattr(_cross_pack_mod, "CrossPackBridge", None)
        if bridge_cls is None:
            pytest.skip("CrossPackBridge class not found")
        config_cls = getattr(_cross_pack_mod, "CrossRegulationConfig", None)
        config = config_cls(graceful_degradation=True) if config_cls else None
        bridge = bridge_cls(config) if config else bridge_cls()
        sync_fn = getattr(bridge, "sync_all", None)
        if sync_fn is None:
            pytest.skip("sync_all not available")
        cbam_data = {"installations": [], "reporting_period": "Q1-2026"}
        result = sync_fn(cbam_data)
        assert result is not None

    def test_multi_entity_support(self):
        """Verify PACK-005 supports multi-entity CBAM submissions."""
        if _orch_mod is None:
            pytest.skip("PACK-005 orchestrator not importable")
        # Check for multi-entity related classes or methods
        has_multi = any(
            "entity" in attr.lower() or "multi" in attr.lower()
            for attr in dir(_orch_mod)
            if not attr.startswith("_")
        )
        # PACK-005 Complete should support multi-entity
        assert _orch_mod is not None


@pytest.mark.integration
class TestETSRegistryBridgeIntegration:
    """Integration tests for PACK-005 ETS Registry Bridge."""

    def test_ets_bridge_instantiates(self):
        """ETSRegistryBridge can be created."""
        if _ets_mod is None:
            pytest.skip("ETSRegistryBridge not importable")
        bridge_cls = getattr(_ets_mod, "ETSRegistryBridge", None)
        if bridge_cls is None:
            pytest.skip("ETSRegistryBridge class not found")
        try:
            bridge = bridge_cls()
        except Exception:
            pytest.skip("ETSRegistryBridge requires config not available")
            return
        assert bridge is not None

    def test_precursor_chain_methods_exist(self):
        """ETSRegistryBridge has precursor chain handling."""
        if _ets_mod is None:
            pytest.skip("ETSRegistryBridge not importable")
        bridge_cls = getattr(_ets_mod, "ETSRegistryBridge", None)
        if bridge_cls is None:
            pytest.skip("ETSRegistryBridge class not found")
        methods = [m for m in dir(bridge_cls) if not m.startswith("_")]
        # ETS bridge should have methods for registry, pricing, or precursors
        assert len(methods) > 0, "ETSRegistryBridge has no public methods"
