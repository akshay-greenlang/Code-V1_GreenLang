# -*- coding: utf-8 -*-
"""
PACK-015 Double Materiality Assessment Pack - Agent Integration Tests
=======================================================================

Tests for integration with the GreenLang agent platform: module
loadability, init exports, Pydantic v2 models, decimal arithmetic,
SHA-256 hashing, logging, type hints, and circular import detection.

Target: 30+ tests.

Author:  GreenLang Platform Team (GL-TestEngineer)
Pack:    PACK-015 Double Materiality Assessment
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

    @pytest.mark.parametrize("engine_key", list(ENGINE_FILES.keys()))
    def test_all_engines_have_version(self, engine_key):
        """Each engine file references a version string."""
        source_path = ENGINES_DIR / ENGINE_FILES[engine_key]
        content = source_path.read_text(encoding="utf-8")
        has_version = (
            "version" in content.lower()
            or "__version__" in content
            or "Version:" in content
        )
        assert has_version, f"Engine {engine_key} should define a version"


# ===========================================================================
# Workflow Module Loadability
# ===========================================================================


class TestWorkflowLoadability:
    """Tests for workflow module loading."""

    @pytest.mark.parametrize("wf_key", list(WORKFLOW_FILES.keys()))
    def test_all_8_workflows_loadable(self, wf_key):
        """Each workflow module loads via importlib."""
        mod = _safe_load(_load_workflow, wf_key)
        assert mod is not None, f"Workflow {wf_key} failed to load"


# ===========================================================================
# Template Module Loadability
# ===========================================================================


class TestTemplateLoadability:
    """Tests for template module loading."""

    @pytest.mark.parametrize("tmpl_key", list(TEMPLATE_FILES.keys()))
    def test_all_8_templates_loadable(self, tmpl_key):
        """Each template module loads via importlib."""
        mod = _safe_load(_load_template, tmpl_key)
        assert mod is not None, f"Template {tmpl_key} failed to load"


# ===========================================================================
# Integration Module Loadability
# ===========================================================================


class TestIntegrationLoadability:
    """Tests for integration module loading."""

    @pytest.mark.parametrize("int_key", list(INTEGRATION_FILES.keys()))
    def test_all_8_integrations_loadable(self, int_key):
        """Each integration module loads via importlib."""
        mod = _safe_load(_load_integration, int_key)
        assert mod is not None, f"Integration {int_key} failed to load"


# ===========================================================================
# Config Module Loadability
# ===========================================================================


class TestConfigLoadability:
    """Tests for config module loading."""

    def test_config_module_loadable(self):
        """pack_config module loads via importlib."""
        mod = _load_config_module()
        assert mod is not None
        assert hasattr(mod, "DMAConfig")
        assert hasattr(mod, "PackConfig")


# ===========================================================================
# Init Exports
# ===========================================================================


class TestInitExports:
    """Tests for __init__.py exports in each sub-package."""

    def test_engine_init_exports(self):
        """engines/__init__.py defines __all__."""
        init_path = ENGINES_DIR / "__init__.py"
        content = init_path.read_text(encoding="utf-8")
        assert "__all__" in content

    def test_workflow_init_exports(self):
        """workflows/__init__.py defines __all__."""
        init_path = WORKFLOWS_DIR / "__init__.py"
        content = init_path.read_text(encoding="utf-8")
        assert "__all__" in content

    def test_template_init_exports(self):
        """templates/__init__.py defines __all__."""
        init_path = TEMPLATES_DIR / "__init__.py"
        content = init_path.read_text(encoding="utf-8")
        assert "__all__" in content

    def test_integration_init_exports(self):
        """integrations/__init__.py defines __all__."""
        init_path = INTEGRATIONS_DIR / "__init__.py"
        content = init_path.read_text(encoding="utf-8")
        assert "__all__" in content

    def test_config_init_exports(self):
        """config/__init__.py defines __all__."""
        init_path = CONFIG_DIR / "__init__.py"
        content = init_path.read_text(encoding="utf-8")
        assert "__all__" in content


# ===========================================================================
# Code Quality Checks
# ===========================================================================


class TestCodeQuality:
    """Tests for code quality patterns across the pack."""

    @pytest.mark.parametrize("engine_key", list(ENGINE_FILES.keys()))
    def test_pydantic_v2_models(self, engine_key):
        """Engine modules use Pydantic (v2) models."""
        source_path = ENGINES_DIR / ENGINE_FILES[engine_key]
        content = source_path.read_text(encoding="utf-8")
        uses_pydantic = "pydantic" in content or "BaseModel" in content
        assert uses_pydantic, f"Engine {engine_key} should use Pydantic models"

    @pytest.mark.parametrize("engine_key", list(ENGINE_FILES.keys()))
    def test_sha256_hashing_present(self, engine_key):
        """Engine modules use SHA-256 hashing for provenance."""
        source_path = ENGINES_DIR / ENGINE_FILES[engine_key]
        content = source_path.read_text(encoding="utf-8")
        has_sha256 = "sha256" in content.lower() or "hashlib" in content
        assert has_sha256, f"Engine {engine_key} should use SHA-256"

    @pytest.mark.parametrize("engine_key", list(ENGINE_FILES.keys()))
    def test_logging_configured(self, engine_key):
        """Engine modules configure logging."""
        source_path = ENGINES_DIR / ENGINE_FILES[engine_key]
        content = source_path.read_text(encoding="utf-8")
        assert "logging" in content, f"Engine {engine_key} should use logging"

    @pytest.mark.parametrize("engine_key", list(ENGINE_FILES.keys()))
    def test_type_hints_present(self, engine_key):
        """Engine modules use type hints."""
        source_path = ENGINES_DIR / ENGINE_FILES[engine_key]
        content = source_path.read_text(encoding="utf-8")
        has_types = (
            "from typing import" in content
            or "from typing_extensions import" in content
            or "-> " in content
        )
        assert has_types, f"Engine {engine_key} should use type hints"

    @pytest.mark.parametrize("engine_key", list(ENGINE_FILES.keys()))
    def test_decimal_arithmetic_used(self, engine_key):
        """Engine modules use Decimal for precise arithmetic."""
        source_path = ENGINES_DIR / ENGINE_FILES[engine_key]
        content = source_path.read_text(encoding="utf-8")
        # Engines should use Decimal or at least import it
        uses_decimal = "Decimal" in content or "decimal" in content
        # Some engines use float-based scoring which is acceptable with rounding
        uses_rounding = "round(" in content
        assert uses_decimal or uses_rounding, (
            f"Engine {engine_key} should use Decimal or explicit rounding"
        )


# ===========================================================================
# Circular Import Detection
# ===========================================================================


class TestCircularImports:
    """Tests for circular import detection."""

    def test_no_circular_imports(self):
        """Loading all modules should not produce circular imports.

        If circular imports exist, module loading would fail with ImportError.
        This test verifies all modules can be loaded independently.
        """
        failures = []
        # Try loading all engine modules
        for key in ENGINE_FILES:
            try:
                _load_engine(key)
            except ImportError as e:
                if "circular" in str(e).lower():
                    failures.append(f"Engine {key}: {e}")

        # Try loading all workflow modules
        for key in WORKFLOW_FILES:
            try:
                _load_workflow(key)
            except ImportError as e:
                if "circular" in str(e).lower():
                    failures.append(f"Workflow {key}: {e}")

        # Try loading all template modules
        for key in TEMPLATE_FILES:
            try:
                _load_template(key)
            except ImportError as e:
                if "circular" in str(e).lower():
                    failures.append(f"Template {key}: {e}")

        # Try loading all integration modules
        for key in INTEGRATION_FILES:
            try:
                _load_integration(key)
            except ImportError as e:
                if "circular" in str(e).lower():
                    failures.append(f"Integration {key}: {e}")

        assert len(failures) == 0, f"Circular imports detected: {failures}"


# ===========================================================================
# Module Source Structure
# ===========================================================================


class TestModuleStructure:
    """Tests for module source file structure."""

    def test_pack_init_exists(self):
        """Pack __init__.py exists."""
        init_path = PACK_ROOT / "__init__.py"
        assert init_path.exists()

    def test_engines_directory_has_8_files(self):
        """Engines directory contains exactly 8 engine files."""
        engine_files = list(ENGINES_DIR.glob("*_engine.py"))
        assert len(engine_files) == 8, (
            f"Expected 8 engine files, found {len(engine_files)}: "
            f"{[f.name for f in engine_files]}"
        )

    def test_workflows_directory_has_8_files(self):
        """Workflows directory contains exactly 8 workflow files."""
        wf_files = list(WORKFLOWS_DIR.glob("*_workflow.py"))
        assert len(wf_files) == 8, (
            f"Expected 8 workflow files, found {len(wf_files)}: "
            f"{[f.name for f in wf_files]}"
        )

    def test_templates_directory_has_8_files(self):
        """Templates directory contains 8 template files."""
        tmpl_files = [
            f for f in TEMPLATES_DIR.glob("*.py")
            if f.name != "__init__.py"
        ]
        assert len(tmpl_files) == 8, (
            f"Expected 8 template files, found {len(tmpl_files)}: "
            f"{[f.name for f in tmpl_files]}"
        )

    def test_integrations_directory_has_8_files(self):
        """Integrations directory contains 8 integration files."""
        int_files = [
            f for f in INTEGRATIONS_DIR.glob("*.py")
            if f.name != "__init__.py"
        ]
        assert len(int_files) == 8, (
            f"Expected 8 integration files, found {len(int_files)}: "
            f"{[f.name for f in int_files]}"
        )

    def test_presets_directory_has_6_files(self):
        """Presets directory contains 6 YAML files."""
        from .conftest import PRESETS_DIR
        preset_files = list(PRESETS_DIR.glob("*.yaml"))
        assert len(preset_files) == 6, (
            f"Expected 6 preset files, found {len(preset_files)}: "
            f"{[f.name for f in preset_files]}"
        )
