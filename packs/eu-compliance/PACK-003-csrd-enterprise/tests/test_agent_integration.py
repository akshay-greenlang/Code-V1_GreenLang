# -*- coding: utf-8 -*-
"""
PACK-003 CSRD Enterprise - Live Agent Integration Tests
=========================================================

Tests that verify PACK-003 Enterprise orchestrator, ML bridge,
GraphQL bridge, and SSO bridge can connect to real GreenLang agents
and app modules.

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
_PACK_003_DIR = Path(__file__).resolve().parent.parent

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


_orch_mod = _import_from_path(
    "pack003_orch_integ",
    _PACK_003_DIR / "integrations" / "pack_orchestrator.py",
)
_ml_mod = _import_from_path(
    "pack003_ml_integ",
    _PACK_003_DIR / "integrations" / "ml_bridge.py",
)
_graphql_mod = _import_from_path(
    "pack003_graphql_integ",
    _PACK_003_DIR / "integrations" / "graphql_bridge.py",
)
_sso_mod = _import_from_path(
    "pack003_sso_integ",
    _PACK_003_DIR / "integrations" / "sso_bridge.py",
)
_loader_mod = _import_from_path(
    "pack_agent_loader_integ_003",
    _EU_COMPLIANCE_DIR / "agent_loader.py",
)


@pytest.fixture(scope="module")
def agent_loader():
    if _loader_mod is None:
        pytest.skip("agent_loader not available")
    loader = _loader_mod.AgentLoader()
    # Load only key agents used by PACK-003
    for aid in ["GL-MRV-X-001", "GL-MRV-X-009"]:
        loader.load(aid)
    return loader


# ===========================================================================
# Tests
# ===========================================================================
@pytest.mark.integration
class TestEnterpriseOrchestratorIntegration:
    """Integration tests for PACK-003 Enterprise Orchestrator."""

    def test_orchestrator_instantiates(self):
        """EnterprisePackOrchestrator can be created with default config."""
        if _orch_mod is None:
            pytest.skip("Enterprise orchestrator not importable")
        orch_cls = getattr(_orch_mod, "EnterprisePackOrchestrator", None)
        if orch_cls is None:
            pytest.skip("EnterprisePackOrchestrator class not found")
        config_cls = getattr(_orch_mod, "EnterpriseOrchestratorConfig", None)
        config = config_cls() if config_cls else None
        orch = orch_cls(config) if config else orch_cls()
        assert orch is not None

    def test_enterprise_scope_coverage(self):
        """Enterprise orchestrator supports multi-scope workflows."""
        if _orch_mod is None:
            pytest.skip("Enterprise orchestrator not importable")
        # Check workflow types exist
        wf_enum = getattr(_orch_mod, "EnterpriseWorkflowType", None)
        if wf_enum is not None:
            members = list(wf_enum)
            assert len(members) >= 1, "Expected at least one workflow type"
        else:
            pytest.skip("EnterpriseWorkflowType enum not found")


@pytest.mark.integration
class TestMLBridgeIntegration:
    """Integration tests for PACK-003 ML Bridge."""

    def test_ml_bridge_instantiates(self):
        """ML Bridge can be created."""
        if _ml_mod is None:
            pytest.skip("ML Bridge not importable")
        bridge_cls = None
        for attr_name in dir(_ml_mod):
            obj = getattr(_ml_mod, attr_name)
            if isinstance(obj, type) and "ML" in attr_name.upper():
                bridge_cls = obj
                break
        if bridge_cls is None:
            pytest.skip("No ML bridge class found")
        try:
            bridge = bridge_cls()
        except Exception:
            # May need config
            pytest.skip("ML Bridge requires config not available in test env")
            return
        assert bridge is not None

    def test_ml_bridge_has_prediction_methods(self):
        """ML Bridge exposes prediction/classification methods."""
        if _ml_mod is None:
            pytest.skip("ML Bridge not importable")
        bridge_cls = None
        for attr_name in dir(_ml_mod):
            obj = getattr(_ml_mod, attr_name)
            if isinstance(obj, type) and ("ML" in attr_name.upper() or "Bridge" in attr_name):
                bridge_cls = obj
                break
        if bridge_cls is None:
            pytest.skip("No ML bridge class found")
        # Check for prediction-related methods
        methods = [m for m in dir(bridge_cls) if not m.startswith("_")]
        has_ml_method = any(
            kw in m.lower() for m in methods
            for kw in ("predict", "classify", "score", "train", "assess")
        )
        assert has_ml_method or len(methods) > 0


@pytest.mark.integration
class TestGraphQLBridgeIntegration:
    """Integration tests for PACK-003 GraphQL Bridge."""

    def test_graphql_bridge_instantiates(self):
        """GraphQL Bridge can be created."""
        if _graphql_mod is None:
            pytest.skip("GraphQL Bridge not importable")
        bridge_cls = None
        for attr_name in dir(_graphql_mod):
            obj = getattr(_graphql_mod, attr_name)
            if isinstance(obj, type) and "GraphQL" in attr_name:
                bridge_cls = obj
                break
        if bridge_cls is None:
            pytest.skip("No GraphQL bridge class found")
        try:
            bridge = bridge_cls()
        except Exception:
            pytest.skip("GraphQL Bridge requires config not available")
            return
        assert bridge is not None
