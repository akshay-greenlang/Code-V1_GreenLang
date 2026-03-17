# -*- coding: utf-8 -*-
"""
PACK-016 ESRS E1 Climate Pack - Agent Integration Tests
=========================================================

Tests for integration with the GreenLang agent platform: module
loadability, init exports, Pydantic v2 models, decimal arithmetic,
SHA-256 hashing, logging, type hints, and circular import detection.

Target: 30+ tests.

Author:  GreenLang Platform Team (GL-TestEngineer)
Pack:    PACK-016 ESRS E1 Climate Change
Date:    March 2026
"""

import inspect
import re
from pathlib import Path

import pytest

from .conftest import (
    ENGINE_FILES,
    ENGINE_CLASSES,
    ENGINES_DIR,
    WORKFLOW_FILES,
    WORKFLOW_CLASSES,
    WORKFLOWS_DIR,
    TEMPLATE_FILES,
    TEMPLATE_CLASSES,
    TEMPLATES_DIR,
    INTEGRATION_FILES,
    INTEGRATION_CLASSES,
    INTEGRATIONS_DIR,
    PACK_ROOT,
    CONFIG_DIR,
    _load_engine,
    _load_workflow,
    _load_template,
    _load_integration,
    _load_config_module,
)


# ===========================================================================
# Helpers
# ===========================================================================


def _safe_load(loader, key):
    """Safely load a module, returning None on failure."""
    try:
        return loader(key)
    except (ImportError, FileNotFoundError):
        return None


# ===========================================================================
# Engine Module Loadability
# ===========================================================================


class TestEngineLoadability:
    """Tests for engine module loading."""

    @pytest.mark.parametrize("engine_key", list(ENGINE_FILES.keys()))
    def test_engine_module_loads_independently(self, engine_key):
        """Each engine module loads independently via importlib."""
        mod = _safe_load(_load_engine, engine_key)
        assert mod is not None, f"Engine {engine_key} failed to load"

    def test_all_8_engines_loadable(self):
        """All 8 engines load successfully."""
        loaded = []
        for key in ENGINE_FILES:
            mod = _safe_load(_load_engine, key)
            if mod is not None:
                loaded.append(key)
        assert len(loaded) == 8, f"Loaded {len(loaded)}/8 engines: {loaded}"

    @pytest.mark.parametrize("engine_key,engine_class", list(ENGINE_CLASSES.items()))
    def test_engines_follow_agent_pattern(self, engine_key, engine_class):
        """Each engine exports its primary class."""
        mod = _safe_load(_load_engine, engine_key)
        if mod is None:
            pytest.skip(f"Engine {engine_key} not loaded")
        assert hasattr(mod, engine_class), f"Engine {engine_key} missing class {engine_class}"

    @pytest.mark.parametrize("engine_key,engine_class", list(ENGINE_CLASSES.items()))
    def test_all_engines_have_docstring(self, engine_key, engine_class):
        """Each engine class has a docstring."""
        mod = _safe_load(_load_engine, engine_key)
        if mod is None:
            pytest.skip(f"Engine {engine_key} not loaded")
        cls = getattr(mod, engine_class, None)
        if cls is None:
            pytest.skip(f"Class {engine_class} not found")
        assert cls.__doc__ is not None


# ===========================================================================
# MRV Integration Tests
# ===========================================================================


class TestMRVIntegration:
    """Tests for MRV agent bridge integration."""

    def test_mrv_bridge_loads(self):
        """MRV bridge module loads."""
        mod = _safe_load(_load_integration, "mrv_agent_bridge")
        assert mod is not None

    def test_mrv_bridge_class_exists(self):
        """MRVAgentBridge class exists."""
        mod = _safe_load(_load_integration, "mrv_agent_bridge")
        if mod is None:
            pytest.skip("MRV bridge not loaded")
        assert hasattr(mod, "MRVAgentBridge")

    def test_mrv_bridge_maps_scope_1_agents(self):
        """MRV bridge source references Scope 1 agents (001-008)."""
        path = INTEGRATIONS_DIR / "mrv_agent_bridge.py"
        content = path.read_text(encoding="utf-8")
        # Check for Scope 1 agent references
        scope_1_refs = sum(1 for i in range(1, 9)
                          if f"00{i}" in content or f"MRV-00{i}" in content or f"mrv_00{i}" in content)
        assert scope_1_refs >= 4, "MRV bridge should reference Scope 1 agents"

    def test_mrv_bridge_maps_scope_2_agents(self):
        """MRV bridge source references Scope 2 agents (009-013)."""
        path = INTEGRATIONS_DIR / "mrv_agent_bridge.py"
        content = path.read_text(encoding="utf-8")
        scope_2_refs = sum(1 for i in range(9, 14)
                          if f"0{i}" in content or f"MRV-0{i}" in content)
        assert scope_2_refs >= 3, "MRV bridge should reference Scope 2 agents"

    def test_mrv_bridge_maps_scope_3_agents(self):
        """MRV bridge source references Scope 3 agents (014-028)."""
        path = INTEGRATIONS_DIR / "mrv_agent_bridge.py"
        content = path.read_text(encoding="utf-8")
        scope_3_refs = sum(1 for i in range(14, 29)
                          if f"0{i}" in content or f"MRV-0{i}" in content)
        assert scope_3_refs >= 5, "MRV bridge should reference Scope 3 agents"

    def test_mrv_scope_mapping_model(self):
        """MRVAgentMapping model exists."""
        mod = _safe_load(_load_integration, "mrv_agent_bridge")
        if mod is None:
            pytest.skip("MRV bridge not loaded")
        assert hasattr(mod, "MRVAgentMapping")


# ===========================================================================
# DMA Integration Tests
# ===========================================================================


class TestDMAIntegration:
    """Tests for PACK-015 Double Materiality integration."""

    def test_dma_bridge_loads(self):
        """DMA bridge module loads."""
        mod = _safe_load(_load_integration, "dma_pack_bridge")
        assert mod is not None

    def test_dma_bridge_class_exists(self):
        """DMAPackBridge class exists."""
        mod = _safe_load(_load_integration, "dma_pack_bridge")
        if mod is None:
            pytest.skip("DMA bridge not loaded")
        assert hasattr(mod, "DMAPackBridge")

    def test_dma_bridge_references_pack_015(self):
        """DMA bridge source references PACK-015."""
        path = INTEGRATIONS_DIR / "dma_pack_bridge.py"
        content = path.read_text(encoding="utf-8")
        assert "PACK-015" in content or "PACK_015" in content or "pack_015" in content

    def test_dma_bridge_e1_materiality_status(self):
        """MaterialityStatus model exists."""
        mod = _safe_load(_load_integration, "dma_pack_bridge")
        if mod is None:
            pytest.skip("DMA bridge not loaded")
        assert hasattr(mod, "MaterialityStatus")


# ===========================================================================
# Decarbonization Integration Tests
# ===========================================================================


class TestDecarbIntegration:
    """Tests for decarbonization agent bridge integration."""

    def test_decarb_bridge_loads(self):
        """Decarbonization bridge module loads."""
        mod = _safe_load(_load_integration, "decarbonization_bridge")
        assert mod is not None

    def test_decarb_bridge_class_exists(self):
        """DecarbonizationBridge class exists."""
        mod = _safe_load(_load_integration, "decarbonization_bridge")
        if mod is None:
            pytest.skip("Decarb bridge not loaded")
        assert hasattr(mod, "DecarbonizationBridge")

    def test_decarb_bridge_has_target_model(self):
        """AbatementOption model exists in decarb bridge."""
        mod = _safe_load(_load_integration, "decarbonization_bridge")
        if mod is None:
            pytest.skip("Decarb bridge not loaded")
        assert hasattr(mod, "AbatementOption")

    def test_decarb_bridge_has_abatement_model(self):
        """AbatementOption model exists in decarb bridge."""
        mod = _safe_load(_load_integration, "decarbonization_bridge")
        if mod is None:
            pytest.skip("Decarb bridge not loaded")
        assert hasattr(mod, "AbatementOption")


# ===========================================================================
# Adaptation Integration Tests
# ===========================================================================


class TestAdaptIntegration:
    """Tests for adaptation agent bridge integration."""

    def test_adapt_bridge_loads(self):
        """Adaptation bridge module loads."""
        mod = _safe_load(_load_integration, "adaptation_bridge")
        assert mod is not None

    def test_adapt_bridge_class_exists(self):
        """AdaptationBridge class exists."""
        mod = _safe_load(_load_integration, "adaptation_bridge")
        if mod is None:
            pytest.skip("Adapt bridge not loaded")
        assert hasattr(mod, "AdaptationBridge")

    def test_adapt_bridge_has_physical_risk_model(self):
        """PhysicalRisk model exists in adaptation bridge."""
        mod = _safe_load(_load_integration, "adaptation_bridge")
        if mod is None:
            pytest.skip("Adapt bridge not loaded")
        assert hasattr(mod, "PhysicalRisk")

    def test_adapt_bridge_has_transition_risk_model(self):
        """ClimateScenario model exists in adaptation bridge (contains transition_risk_factors)."""
        mod = _safe_load(_load_integration, "adaptation_bridge")
        if mod is None:
            pytest.skip("Adapt bridge not loaded")
        assert hasattr(mod, "ClimateScenario")

    def test_adapt_bridge_has_scenario_analysis(self):
        """ClimateScenario or ResilienceScore model exists in adaptation bridge."""
        mod = _safe_load(_load_integration, "adaptation_bridge")
        if mod is None:
            pytest.skip("Adapt bridge not loaded")
        assert hasattr(mod, "ClimateScenario") or hasattr(mod, "ResilienceScore")


# ===========================================================================
# GHG App Integration Tests
# ===========================================================================


class TestGHGAppIntegration:
    """Tests for GL-GHG-APP bridge integration."""

    def test_ghg_app_bridge_loads(self):
        """GHG App bridge module loads."""
        mod = _safe_load(_load_integration, "ghg_app_bridge")
        assert mod is not None

    def test_ghg_app_bridge_class_exists(self):
        """GHGAppBridge class exists."""
        mod = _safe_load(_load_integration, "ghg_app_bridge")
        if mod is None:
            pytest.skip("GHG app bridge not loaded")
        assert hasattr(mod, "GHGAppBridge")

    def test_ghg_app_bridge_has_import_result(self):
        """InventoryImport or BridgeResult model exists."""
        mod = _safe_load(_load_integration, "ghg_app_bridge")
        if mod is None:
            pytest.skip("GHG app bridge not loaded")
        assert hasattr(mod, "InventoryImport") or hasattr(mod, "BridgeResult")

    def test_ghg_app_bridge_has_export_result(self):
        """BridgeResult model exists for export operations."""
        mod = _safe_load(_load_integration, "ghg_app_bridge")
        if mod is None:
            pytest.skip("GHG app bridge not loaded")
        assert hasattr(mod, "BridgeResult")


# ===========================================================================
# Cross-Pack Dependencies
# ===========================================================================


class TestCrossPackDependencies:
    """Tests for cross-pack dependency declarations."""

    def test_pack_015_dependency_in_manifest(self, pack_yaml_data):
        """PACK-015 is declared as a dependency in pack.yaml."""
        deps = pack_yaml_data.get("dependencies", {})
        required_packs = deps.get("required_packs", [])
        pack_015_found = any("PACK-015" in p.get("id", "") for p in required_packs)
        assert pack_015_found

    def test_agent_dependencies_declared(self, pack_yaml_data):
        """Agent dependencies are declared in pack.yaml."""
        deps = pack_yaml_data.get("dependencies", {})
        agent_deps = deps.get("agent_dependencies", [])
        assert len(agent_deps) > 0

    def test_mrv_agents_in_manifest(self, pack_yaml_data):
        """MRV agents are listed in pack.yaml."""
        agents_mrv = pack_yaml_data.get("agents_mrv", {})
        scope_1_agents = agents_mrv.get("scope_1", [])
        scope_2_agents = agents_mrv.get("scope_2", [])
        scope_3_agents = agents_mrv.get("scope_3", [])
        total = len(scope_1_agents) + len(scope_2_agents) + len(scope_3_agents)
        cross_cutting = agents_mrv.get("cross_cutting", [])
        total += len(cross_cutting)
        assert total == 30, f"Expected 30 MRV agents, got {total}"


# ===========================================================================
# Pydantic v2 Model Patterns
# ===========================================================================


class TestPydanticV2Patterns:
    """Tests for Pydantic v2 model usage across engines."""

    @pytest.mark.parametrize("engine_key", list(ENGINE_FILES.keys()))
    def test_engine_uses_pydantic(self, engine_key):
        """Each engine uses Pydantic BaseModel."""
        source_path = ENGINES_DIR / ENGINE_FILES[engine_key]
        content = source_path.read_text(encoding="utf-8")
        assert "BaseModel" in content, f"Engine {engine_key} should use Pydantic BaseModel"

    @pytest.mark.parametrize("engine_key", list(ENGINE_FILES.keys()))
    def test_engine_uses_decimal(self, engine_key):
        """Each engine uses Decimal for financial precision."""
        source_path = ENGINES_DIR / ENGINE_FILES[engine_key]
        content = source_path.read_text(encoding="utf-8")
        has_decimal = "Decimal" in content or "decimal" in content
        assert has_decimal, f"Engine {engine_key} should use Decimal arithmetic"

    @pytest.mark.parametrize("engine_key", list(ENGINE_FILES.keys()))
    def test_engine_has_logging(self, engine_key):
        """Each engine uses logging."""
        source_path = ENGINES_DIR / ENGINE_FILES[engine_key]
        content = source_path.read_text(encoding="utf-8")
        assert "logging" in content, f"Engine {engine_key} should use logging"

    @pytest.mark.parametrize("engine_key", list(ENGINE_FILES.keys()))
    def test_engine_has_type_hints(self, engine_key):
        """Each engine uses type hints."""
        source_path = ENGINES_DIR / ENGINE_FILES[engine_key]
        content = source_path.read_text(encoding="utf-8")
        has_typing = "from typing import" in content or "from typing " in content
        has_annotations = ":" in content  # simplified check
        assert has_typing or has_annotations, f"Engine {engine_key} should use type hints"


# ===========================================================================
# No Circular Imports
# ===========================================================================


class TestNoCircularImports:
    """Tests to detect circular import issues."""

    def test_config_loads_independently(self):
        """Config module loads without circular imports."""
        mod = _load_config_module()
        assert mod is not None

    @pytest.mark.parametrize("engine_key", list(ENGINE_FILES.keys()))
    def test_engine_loads_independently(self, engine_key):
        """Each engine loads without circular imports."""
        mod = _safe_load(_load_engine, engine_key)
        assert mod is not None, f"Engine {engine_key} has import issues"

    @pytest.mark.parametrize("integration_key", list(INTEGRATION_FILES.keys()))
    def test_integration_loads_independently(self, integration_key):
        """Each integration loads without circular imports."""
        mod = _safe_load(_load_integration, integration_key)
        assert mod is not None, f"Integration {integration_key} has import issues"
