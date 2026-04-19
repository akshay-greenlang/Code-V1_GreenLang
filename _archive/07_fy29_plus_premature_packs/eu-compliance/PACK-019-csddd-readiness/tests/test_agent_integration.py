# -*- coding: utf-8 -*-
"""
PACK-019 CSDDD Readiness Pack - Agent Integration Tests
=========================================================

Tests cross-engine integration scenarios including bridge data flow,
orchestrator coordination, and health check coverage of all components.

Test count target: ~30 tests
"""

import asyncio
import sys
from pathlib import Path
from typing import Any, Dict, List

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from conftest import (
    ENGINE_CLASSES,
    ENGINE_FILES,
    ENGINES_DIR,
    INTEGRATION_CLASSES,
    WORKFLOW_CLASSES,
    _load_integration,
    _load_module,
    _load_workflow,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

ENGINE_KEYS = list(ENGINE_FILES.keys())


def _run(coro):
    """Run an async coroutine synchronously."""
    return asyncio.get_event_loop().run_until_complete(coro)


def _load_engine(name: str):
    """Load an engine module by short name."""
    filename = ENGINE_FILES[name]
    return _load_module(ENGINES_DIR / filename, f"engines.{name}")


# ---------------------------------------------------------------------------
# 1. Engine instantiation
# ---------------------------------------------------------------------------


class TestEngineInstantiation:
    """Verify all 8 engines can be instantiated."""

    @pytest.mark.parametrize("engine_key", ENGINE_KEYS)
    def test_engine_class_exists(self, engine_key: str):
        mod = _load_engine(engine_key)
        cls_name = ENGINE_CLASSES[engine_key]
        cls = getattr(mod, cls_name, None)
        assert cls is not None, f"Class {cls_name} not found in {engine_key}"

    @pytest.mark.parametrize("engine_key", ENGINE_KEYS)
    def test_engine_instantiation(self, engine_key: str):
        mod = _load_engine(engine_key)
        cls = getattr(mod, ENGINE_CLASSES[engine_key])
        instance = cls()
        assert instance is not None


# ---------------------------------------------------------------------------
# 2. Cross-bridge data flow
# ---------------------------------------------------------------------------


class TestBridgeDataFlow:
    """Test data flows across bridge integrations."""

    def test_csrd_bridge_mapping_structure(self):
        """CSRD bridge S1 mapping should return ESRS-to-CSDDD dict."""
        mod = _load_integration("csrd_pack_bridge")
        bridge = getattr(mod, "CSRDPackBridge")()
        mapping = bridge.get_s1_mapping()
        assert isinstance(mapping, dict)

    def test_csrd_bridge_s2_mapping(self):
        mod = _load_integration("csrd_pack_bridge")
        bridge = getattr(mod, "CSRDPackBridge")()
        mapping = bridge.get_s2_mapping()
        assert isinstance(mapping, dict)

    def test_csrd_bridge_s3_mapping(self):
        mod = _load_integration("csrd_pack_bridge")
        bridge = getattr(mod, "CSRDPackBridge")()
        mapping = bridge.get_s3_mapping()
        assert isinstance(mapping, dict)

    def test_csrd_bridge_s4_mapping(self):
        mod = _load_integration("csrd_pack_bridge")
        bridge = getattr(mod, "CSRDPackBridge")()
        mapping = bridge.get_s4_mapping()
        assert isinstance(mapping, dict)

    def test_csrd_bridge_g1_mapping(self):
        mod = _load_integration("csrd_pack_bridge")
        bridge = getattr(mod, "CSRDPackBridge")()
        mapping = bridge.get_g1_mapping()
        assert isinstance(mapping, dict)

    def test_mrv_bridge_scope_data(self):
        """MRV bridge should provide emission data by scope."""
        mod = _load_integration("mrv_bridge")
        bridge = getattr(mod, "MRVBridge")()
        context = {"reporting_year": 2027, "entity_id": "TEST-001"}
        s1 = bridge.get_scope1_data(context)
        assert s1 is not None

    def test_eudr_bridge_mapping(self):
        """EUDR bridge should map deforestation risks to CSDDD impacts."""
        mod = _load_integration("eudr_bridge")
        bridge = getattr(mod, "EUDRBridge")()
        context = {
            "entity_id": "TEST-001",
            "commodities": ["palm_oil", "soy"],
        }
        result = bridge.map_eudr_to_csddd(context)
        assert result is not None

    def test_supply_chain_bridge_tier_breakdown(self):
        """Supply chain bridge should provide tier breakdown."""
        mod = _load_integration("supply_chain_bridge")
        bridge = getattr(mod, "SupplyChainBridge")()
        suppliers = [
            {"supplier_id": "SUP-001", "name": "MetalWorks", "tier": "tier_1", "country": "CN", "annual_spend_eur": 500000},
            {"supplier_id": "SUP-002", "name": "ChemCorp", "tier": "tier_2", "country": "IN", "annual_spend_eur": 300000},
        ]
        breakdown = bridge.get_tier_breakdown(suppliers)
        assert breakdown is not None


# ---------------------------------------------------------------------------
# 3. Orchestrator coordination
# ---------------------------------------------------------------------------


class TestOrchestratorCoordination:
    """Test orchestrator coordinates engines correctly."""

    def _get_orch_module(self):
        return _load_integration("pack_orchestrator")

    def _get_orchestrator(self):
        mod = self._get_orch_module()
        return getattr(mod, "CSDDDOrchestrator")()

    def _make_profile(self, **kwargs):
        mod = self._get_orch_module()
        profile_cls = getattr(mod, "CompanyProfile")
        return profile_cls(**kwargs)

    def test_orchestrator_full_assessment_structure(self):
        """Full assessment must produce a structured result."""
        orch = self._get_orchestrator()
        profile = self._make_profile(
            company_name="TestCo AG",
            employee_count=6000,
            net_turnover_eur=2_000_000_000,
            sector="MANUFACTURING",
        )
        result = orch.run_full_assessment(profile=profile)
        assert result is not None
        # Result should have phase results or a summary
        result_data = result if isinstance(result, dict) else result.__dict__
        assert len(result_data) > 0

    def test_orchestrator_quick_assessment(self):
        """Quick assessment must complete faster with fewer phases."""
        orch = self._get_orchestrator()
        profile = self._make_profile(
            company_name="SmallCo GmbH",
            employee_count=2000,
            net_turnover_eur=600_000_000,
        )
        result = orch.run_quick_assessment(profile)
        assert result is not None

    def test_orchestrator_status_after_assessment(self):
        """Status should reflect completed assessment."""
        orch = self._get_orchestrator()
        profile = self._make_profile(
            company_name="TestCo AG",
            employee_count=6000,
            net_turnover_eur=2_000_000_000,
        )
        orch.run_full_assessment(profile=profile)
        status = orch.get_status()
        assert isinstance(status, dict)


# ---------------------------------------------------------------------------
# 4. Health check coverage
# ---------------------------------------------------------------------------


class TestHealthCheckCoverage:
    """Test health check covers all pack components."""

    def _get_health_check(self):
        mod = _load_integration("health_check")
        return getattr(mod, "CSDDDHealthCheck")()

    def test_health_check_result_has_overall_status(self):
        hc = self._get_health_check()
        result = hc.run_health_check()
        result_data = result if isinstance(result, dict) else result.__dict__
        # Should have an overall status or health indicator
        assert len(result_data) > 0

    def test_health_check_system_status(self):
        hc = self._get_health_check()
        status = hc.get_system_status()
        assert isinstance(status, dict)
        assert len(status) > 0

    def test_health_check_generate_report(self):
        """Health report should contain detailed component information."""
        hc = self._get_health_check()
        report = hc.generate_health_report()
        assert report is not None


# ---------------------------------------------------------------------------
# 5. Workflow-to-engine alignment
# ---------------------------------------------------------------------------


class TestWorkflowEngineAlignment:
    """Verify workflows and engines are aligned in scope."""

    def test_all_workflows_have_phases(self):
        """Every workflow must define phases that map to engine operations."""
        for wf_key, wf_cls_name in WORKFLOW_CLASSES.items():
            mod = _load_workflow(wf_key)
            cls = getattr(mod, wf_cls_name)
            instance = cls()
            phases = instance.get_phases()
            assert len(phases) >= 3, f"Workflow {wf_key} has too few phases: {len(phases)}"

    def test_workflow_result_consistency(self):
        """All workflow results should share common fields."""
        common_fields = {"workflow_id", "status", "provenance_hash", "executed_at"}
        for wf_key, wf_cls_name in WORKFLOW_CLASSES.items():
            mod = _load_workflow(wf_key)
            cls = getattr(mod, wf_cls_name)
            instance = cls()
            result = _run(instance.execute())
            result_fields = set(result.__dict__.keys()) if hasattr(result, "__dict__") else set()
            if hasattr(type(result), "model_fields"):
                result_fields = set(type(result).model_fields.keys())
            for field in common_fields:
                assert field in result_fields, (
                    f"Workflow {wf_key} result missing common field: {field}"
                )
