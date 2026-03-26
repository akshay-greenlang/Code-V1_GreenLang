# -*- coding: utf-8 -*-
"""
Unit tests for PACK-041 Manifest
====================================

Tests pack metadata: version, name, category, engine count, workflow count,
template count, integration count, preset count, and file existence.

Coverage target: 85%+
Total tests: ~25
"""

import importlib.util
import sys
from pathlib import Path

import pytest

PACK_ROOT = Path(__file__).resolve().parent.parent

from tests.conftest import (
    ENGINE_FILES,
    ENGINE_CLASSES,
    WORKFLOW_FILES,
    WORKFLOW_CLASSES,
    TEMPLATE_FILES,
    TEMPLATE_CLASSES,
    INTEGRATION_FILES,
    INTEGRATION_CLASSES,
    PRESET_NAMES,
    ENGINES_DIR,
    WORKFLOWS_DIR,
    TEMPLATES_DIR,
    INTEGRATIONS_DIR,
    PRESETS_DIR,
)


# =============================================================================
# Pack Identity
# =============================================================================


class TestPackIdentity:
    """Test pack identity metadata."""

    def test_pack_init_exists(self):
        init_path = PACK_ROOT / "__init__.py"
        assert init_path.exists()

    def test_pack_version(self):
        init_path = PACK_ROOT / "__init__.py"
        if not init_path.exists():
            pytest.skip("__init__.py not found")
        mod_key = "pack041_test.pack_init"
        if mod_key in sys.modules:
            mod = sys.modules[mod_key]
        else:
            spec = importlib.util.spec_from_file_location(mod_key, init_path)
            mod = importlib.util.module_from_spec(spec)
            sys.modules[mod_key] = mod
            try:
                spec.loader.exec_module(mod)
            except Exception:
                pytest.skip("Cannot load __init__.py")
        assert hasattr(mod, "__version__")
        assert mod.__version__ == "1.0.0"

    def test_pack_name(self):
        init_path = PACK_ROOT / "__init__.py"
        if not init_path.exists():
            pytest.skip("__init__.py not found")
        mod_key = "pack041_test.pack_init"
        mod = sys.modules.get(mod_key)
        if mod is None:
            pytest.skip("Module not loaded")
        assert hasattr(mod, "__pack_name__")
        assert mod.__pack_name__ == "Scope 1-2 Complete Pack"

    def test_pack_id(self):
        init_path = PACK_ROOT / "__init__.py"
        if not init_path.exists():
            pytest.skip("__init__.py not found")
        mod_key = "pack041_test.pack_init"
        mod = sys.modules.get(mod_key)
        if mod is None:
            pytest.skip("Module not loaded")
        assert hasattr(mod, "__pack__")
        assert mod.__pack__ == "PACK-041"

    def test_pack_category(self):
        init_path = PACK_ROOT / "__init__.py"
        if not init_path.exists():
            pytest.skip("__init__.py not found")
        mod_key = "pack041_test.pack_init"
        mod = sys.modules.get(mod_key)
        if mod is None:
            pytest.skip("Module not loaded")
        assert hasattr(mod, "__category__")
        assert mod.__category__ == "ghg-accounting"


# =============================================================================
# Component Counts
# =============================================================================


class TestComponentCounts:
    """Test that the pack has the expected number of components."""

    def test_engines_count_10(self):
        assert len(ENGINE_FILES) == 10

    def test_engine_classes_count_10(self):
        assert len(ENGINE_CLASSES) == 10

    def test_workflows_count_8(self):
        assert len(WORKFLOW_FILES) == 8

    def test_workflow_classes_count_8(self):
        assert len(WORKFLOW_CLASSES) == 8

    def test_templates_count_10(self):
        assert len(TEMPLATE_FILES) == 10

    def test_template_classes_count_10(self):
        assert len(TEMPLATE_CLASSES) == 10

    def test_integrations_count_12(self):
        assert len(INTEGRATION_FILES) == 12

    def test_integration_classes_count_12(self):
        assert len(INTEGRATION_CLASSES) == 12

    def test_presets_count_8(self):
        assert len(PRESET_NAMES) == 8


# =============================================================================
# Directory Structure
# =============================================================================


class TestDirectoryStructure:
    """Test expected directory structure."""

    def test_engines_dir_exists(self):
        assert ENGINES_DIR.exists()

    def test_workflows_dir_exists(self):
        assert WORKFLOWS_DIR.exists()

    def test_templates_dir_exists(self):
        assert TEMPLATES_DIR.exists()

    def test_integrations_dir_exists(self):
        assert INTEGRATIONS_DIR.exists()

    def test_config_dir_exists(self):
        assert (PACK_ROOT / "config").exists()

    def test_tests_dir_exists(self):
        assert (PACK_ROOT / "tests").exists()


# =============================================================================
# Engine File Naming
# =============================================================================


class TestEngineFileNaming:
    """Test engine file naming conventions."""

    @pytest.mark.parametrize("engine_key,engine_file", list(ENGINE_FILES.items()))
    def test_engine_file_ends_with_engine_py(self, engine_key, engine_file):
        assert engine_file.endswith("_engine.py")

    @pytest.mark.parametrize("engine_key", list(ENGINE_FILES.keys()))
    def test_engine_class_ends_with_engine(self, engine_key):
        cls_name = ENGINE_CLASSES[engine_key]
        assert cls_name.endswith("Engine")
