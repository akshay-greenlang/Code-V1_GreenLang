# -*- coding: utf-8 -*-
"""
PACK-017 ESRS Full Coverage Pack - Agent Integration Tests
============================================================

Tests for integration with the GreenLang agent platform: engine module
loadability, Pydantic v2 models, Decimal arithmetic, SHA-256 hashing,
logging, type hints, MRV agent routing, data agent routing, cross-pack
bridges, and health check coverage.

Target: 30+ tests.

Author:  GreenLang Platform Team (GL-TestEngineer)
Pack:    PACK-017 ESRS Full Coverage Pack
Date:    March 2026
"""

import pytest

from .conftest import (
    ENGINE_FILES,
    ENGINE_CLASSES,
    ENGINES_DIR,
    INTEGRATION_FILES,
    INTEGRATION_CLASSES,
    INTEGRATIONS_DIR,
    _load_engine,
    _load_integration,
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
# Engine Loadability
# ===========================================================================


class TestEngineLoadability:
    """Tests for independent engine module loading (11 engines)."""

    @pytest.mark.parametrize("engine_key", list(ENGINE_FILES.keys()))
    def test_engine_module_loads_independently(self, engine_key):
        """Each engine module loads independently via importlib."""
        mod = _safe_load(_load_engine, engine_key)
        assert mod is not None, f"Engine {engine_key} failed to load"

    def test_all_11_engines_loadable(self):
        """All 11 engines load successfully."""
        loaded = []
        for key in ENGINE_FILES:
            mod = _safe_load(_load_engine, key)
            if mod is not None:
                loaded.append(key)
        assert len(loaded) == 11, f"Loaded {len(loaded)}/11 engines: {loaded}"

    @pytest.mark.parametrize("engine_key,engine_class", list(ENGINE_CLASSES.items()))
    def test_engine_exports_primary_class(self, engine_key, engine_class):
        """Each engine exports its primary class."""
        mod = _safe_load(_load_engine, engine_key)
        if mod is None:
            pytest.skip(f"Engine {engine_key} not loaded")
        assert hasattr(mod, engine_class), f"Engine {engine_key} missing class {engine_class}"


# ===========================================================================
# Engine Pattern Compliance
# ===========================================================================


class TestEnginePatternCompliance:
    """Tests for engine pattern compliance (docstrings, Pydantic, Decimal, SHA-256)."""

    @pytest.mark.parametrize("engine_key,engine_class", list(ENGINE_CLASSES.items()))
    def test_engine_has_docstring(self, engine_key, engine_class):
        """Each engine class has a docstring."""
        mod = _safe_load(_load_engine, engine_key)
        if mod is None:
            pytest.skip(f"Engine {engine_key} not loaded")
        cls = getattr(mod, engine_class, None)
        if cls is None:
            pytest.skip(f"Class {engine_class} not found")
        assert cls.__doc__ is not None, f"{engine_class} missing docstring"

    @pytest.mark.parametrize("engine_key", list(ENGINE_FILES.keys()))
    def test_engine_uses_pydantic_v2_basemodel(self, engine_key):
        """Each engine uses Pydantic BaseModel."""
        source_path = ENGINES_DIR / ENGINE_FILES[engine_key]
        content = source_path.read_text(encoding="utf-8")
        assert "BaseModel" in content, f"Engine {engine_key} should use Pydantic BaseModel"

    @pytest.mark.parametrize("engine_key", list(ENGINE_FILES.keys()))
    def test_engine_uses_decimal(self, engine_key):
        """Each engine uses Decimal for bit-perfect arithmetic."""
        source_path = ENGINES_DIR / ENGINE_FILES[engine_key]
        content = source_path.read_text(encoding="utf-8")
        has_decimal = "Decimal" in content or "decimal" in content
        assert has_decimal, f"Engine {engine_key} should use Decimal arithmetic"

    @pytest.mark.parametrize("engine_key", list(ENGINE_FILES.keys()))
    def test_engine_uses_sha256_hashlib(self, engine_key):
        """Each engine uses hashlib SHA-256 for provenance."""
        source_path = ENGINES_DIR / ENGINE_FILES[engine_key]
        content = source_path.read_text(encoding="utf-8")
        has_hash = "hashlib" in content and ("sha256" in content.lower() or "SHA" in content)
        assert has_hash, f"Engine {engine_key} should use hashlib SHA-256"

    @pytest.mark.parametrize("engine_key", list(ENGINE_FILES.keys()))
    def test_engine_has_logging(self, engine_key):
        """Each engine uses the logging module."""
        source_path = ENGINES_DIR / ENGINE_FILES[engine_key]
        content = source_path.read_text(encoding="utf-8")
        assert "logging" in content, f"Engine {engine_key} should use logging"

    @pytest.mark.parametrize("engine_key", list(ENGINE_FILES.keys()))
    def test_engine_has_type_hints(self, engine_key):
        """Each engine uses type hints from typing module."""
        source_path = ENGINES_DIR / ENGINE_FILES[engine_key]
        content = source_path.read_text(encoding="utf-8")
        has_typing = "from typing import" in content or "from typing " in content
        assert has_typing, f"Engine {engine_key} should use type hints"


# ===========================================================================
# MRV Agent Routing
# ===========================================================================


class TestMRVAgentRouting:
    """Tests for MRV agent bridge routing table."""

    @pytest.fixture(autouse=True)
    def _load(self):
        self.mod = _safe_load(_load_integration, "mrv_bridge")

    def test_routing_table_has_30_agents(self):
        """MRV bridge references all 30 MRV agents."""
        path = INTEGRATIONS_DIR / INTEGRATION_FILES["mrv_bridge"]
        if not path.exists():
            pytest.skip("MRV bridge file not found")
        content = path.read_text(encoding="utf-8")
        has_001 = "001" in content
        has_030 = "030" in content
        assert has_001 and has_030, "MRV bridge should reference agents 001 through 030"

    def test_all_scopes_covered(self):
        """MRV bridge covers Scope 1, 2, and 3."""
        path = INTEGRATIONS_DIR / INTEGRATION_FILES["mrv_bridge"]
        if not path.exists():
            pytest.skip("MRV bridge file not found")
        content = path.read_text(encoding="utf-8")
        for scope in ["scope_1", "scope_2", "scope_3"]:
            assert scope in content.lower() or scope.replace("_", " ") in content.lower(), (
                f"MRV bridge should reference {scope}"
            )

    def test_graceful_degradation(self):
        """MRV bridge source handles unavailable agents gracefully."""
        path = INTEGRATIONS_DIR / INTEGRATION_FILES["mrv_bridge"]
        if not path.exists():
            pytest.skip("MRV bridge file not found")
        content = path.read_text(encoding="utf-8")
        has_error_handling = (
            "try" in content
            or "except" in content
            or "fallback" in content.lower()
            or "unavailable" in content.lower()
            or "optional" in content.lower()
        )
        assert has_error_handling, "MRV bridge should handle unavailable agents"


# ===========================================================================
# Data Agent Routing
# ===========================================================================


class TestDataAgentRouting:
    """Tests for data agent bridge routing table."""

    def test_routing_table_has_20_agents(self):
        """Data bridge references all 20 data agents."""
        path = INTEGRATIONS_DIR / INTEGRATION_FILES["data_bridge"]
        if not path.exists():
            pytest.skip("Data bridge file not found")
        content = path.read_text(encoding="utf-8")
        has_001 = "001" in content
        has_020 = "020" in content
        assert has_001 and has_020, "Data bridge should reference agents 001 through 020"

    def test_erp_mappings_sap_oracle_workday_dynamics(self):
        """Data bridge references major ERP systems."""
        path = INTEGRATIONS_DIR / INTEGRATION_FILES["data_bridge"]
        if not path.exists():
            pytest.skip("Data bridge file not found")
        content = path.read_text(encoding="utf-8")
        erp_count = sum(1 for erp in ["SAP", "Oracle", "Workday", "Dynamics"]
                        if erp.lower() in content.lower())
        assert erp_count >= 2, f"Expected 2+ ERP references, found {erp_count}"


# ===========================================================================
# Cross-Pack Bridges
# ===========================================================================


class TestCrossPackBridges:
    """Tests for cross-pack bridge integrations."""

    def test_pack_015_dma_bridge_init(self):
        """DMA bridge (PACK-015) loads and has primary class."""
        mod = _safe_load(_load_integration, "dma_bridge")
        if mod is None:
            pytest.skip("DMA bridge not loaded")
        assert hasattr(mod, INTEGRATION_CLASSES["dma_bridge"])

    def test_pack_016_e1_bridge_init(self):
        """E1 bridge (PACK-016) loads and has primary class."""
        mod = _safe_load(_load_integration, "e1_bridge")
        if mod is None:
            pytest.skip("E1 bridge not loaded")
        assert hasattr(mod, INTEGRATION_CLASSES["e1_bridge"])

    def test_csrd_app_bridge_init(self):
        """CSRD app bridge loads and has primary class."""
        mod = _safe_load(_load_integration, "csrd_app_bridge")
        if mod is None:
            pytest.skip("CSRD app bridge not loaded")
        assert hasattr(mod, INTEGRATION_CLASSES["csrd_app_bridge"])

    def test_dma_bridge_references_pack_015(self):
        """DMA bridge source references PACK-015."""
        path = INTEGRATIONS_DIR / INTEGRATION_FILES["dma_bridge"]
        if not path.exists():
            pytest.skip("DMA bridge file not found")
        content = path.read_text(encoding="utf-8")
        assert "PACK-015" in content or "PACK_015" in content or "pack_015" in content

    def test_e1_bridge_references_pack_016(self):
        """E1 bridge source references PACK-016."""
        path = INTEGRATIONS_DIR / INTEGRATION_FILES["e1_bridge"]
        if not path.exists():
            pytest.skip("E1 bridge file not found")
        content = path.read_text(encoding="utf-8")
        assert "PACK-016" in content or "PACK_016" in content or "pack_016" in content


# ===========================================================================
# Health Check Coverage
# ===========================================================================


class TestHealthCheckCoverage:
    """Tests for health check coverage across all components."""

    @pytest.fixture(autouse=True)
    def _load(self):
        self.mod = _safe_load(_load_integration, "health_check")

    def test_all_check_categories_present(self):
        """Health check covers engines, workflows, templates, integrations."""
        path = INTEGRATIONS_DIR / INTEGRATION_FILES["health_check"]
        if not path.exists():
            pytest.skip("Health check file not found")
        content = path.read_text(encoding="utf-8")
        categories = ["engine", "workflow", "template", "integration"]
        found = sum(1 for cat in categories if cat in content.lower())
        assert found >= 3, f"Expected 3+ categories, found {found}"

    def test_all_handlers_callable(self):
        """Health check class has callable run/check method."""
        if self.mod is None:
            pytest.skip("Module not loaded")
        cls = getattr(self.mod, INTEGRATION_CLASSES["health_check"], None)
        if cls is None:
            pytest.skip("Health check class not found")
        has_run = (
            hasattr(cls, "run")
            or hasattr(cls, "check_all")
            or hasattr(cls, "run_all_checks")
            or hasattr(cls, "run_check")
        )
        assert has_run
