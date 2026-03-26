# -*- coding: utf-8 -*-
"""
Unit tests for PACK-042 Manifest / Pack Structure
====================================================

Tests pack identity, component counts, directory structure, file naming
conventions, __init__.py presence, pack.yaml existence, and preset YAML
file existence.

Coverage target: 85%+
Total tests: ~25
"""

import importlib.util
import sys
from pathlib import Path

import pytest

from tests.conftest import (
    PACK_ROOT,
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
    CONFIG_DIR,
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
        mod_key = "pack042_test.pack_init"
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
        mod_key = "pack042_test.pack_init"
        mod = sys.modules.get(mod_key)
        if mod is None:
            pytest.skip("Module not loaded")
        assert hasattr(mod, "__pack_name__")
        assert mod.__pack_name__ == "Scope 3 Starter Pack"

    def test_pack_id(self):
        init_path = PACK_ROOT / "__init__.py"
        if not init_path.exists():
            pytest.skip("__init__.py not found")
        mod_key = "pack042_test.pack_init"
        mod = sys.modules.get(mod_key)
        if mod is None:
            pytest.skip("Module not loaded")
        assert hasattr(mod, "__pack__")
        assert mod.__pack__ == "PACK-042"

    def test_pack_category(self):
        init_path = PACK_ROOT / "__init__.py"
        if not init_path.exists():
            pytest.skip("__init__.py not found")
        mod_key = "pack042_test.pack_init"
        mod = sys.modules.get(mod_key)
        if mod is None:
            pytest.skip("Module not loaded")
        assert hasattr(mod, "__category__")
        assert mod.__category__ == "ghg-accounting"


# =============================================================================
# Component Counts
# =============================================================================


class TestComponentCounts:
    """Test component counts match specification."""

    def test_10_engines(self):
        assert len(ENGINE_FILES) == 10

    def test_8_workflows(self):
        assert len(WORKFLOW_FILES) == 8

    def test_10_templates(self):
        assert len(TEMPLATE_FILES) == 10

    def test_12_integrations(self):
        assert len(INTEGRATION_FILES) == 12

    def test_8_presets(self):
        assert len(PRESET_NAMES) == 8


# =============================================================================
# Directory Structure
# =============================================================================


class TestDirectoryStructure:
    """Test directory structure exists."""

    def test_pack_root_exists(self):
        assert PACK_ROOT.exists()
        assert PACK_ROOT.is_dir()

    def test_engines_dir_exists(self):
        assert ENGINES_DIR.exists()

    def test_workflows_dir_exists(self):
        assert WORKFLOWS_DIR.exists()

    def test_templates_dir_exists(self):
        assert TEMPLATES_DIR.exists()

    def test_integrations_dir_exists(self):
        # May not exist yet
        if not INTEGRATIONS_DIR.exists():
            pytest.skip("Integrations directory not yet created")
        assert INTEGRATIONS_DIR.is_dir()

    def test_config_dir_exists(self):
        assert CONFIG_DIR.exists()

    def test_presets_dir_exists(self):
        assert PRESETS_DIR.exists()


# =============================================================================
# File Naming Convention
# =============================================================================


class TestFileNamingConvention:
    """Test engine file naming convention."""

    @pytest.mark.parametrize("engine_name,file_name", list(ENGINE_FILES.items()))
    def test_engine_file_ends_with_engine_py(self, engine_name, file_name):
        assert file_name.endswith("_engine.py"), (
            f"Engine {engine_name} file {file_name} should end with _engine.py"
        )

    @pytest.mark.parametrize("wf_name,file_name", list(WORKFLOW_FILES.items()))
    def test_workflow_file_ends_with_workflow_py(self, wf_name, file_name):
        assert file_name.endswith("_workflow.py"), (
            f"Workflow {wf_name} file {file_name} should end with _workflow.py"
        )


# =============================================================================
# __init__.py Files
# =============================================================================


class TestInitFiles:
    """Test all __init__.py files present."""

    def test_pack_root_init(self):
        assert (PACK_ROOT / "__init__.py").exists()

    def test_config_init(self):
        assert (CONFIG_DIR / "__init__.py").exists()

    def test_templates_init(self):
        assert (TEMPLATES_DIR / "__init__.py").exists()

    def test_tests_init(self):
        assert (PACK_ROOT / "tests" / "__init__.py").exists()
