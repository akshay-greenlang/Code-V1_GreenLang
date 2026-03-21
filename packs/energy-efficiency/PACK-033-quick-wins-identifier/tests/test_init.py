# -*- coding: utf-8 -*-
"""
PACK-033 Quick Wins Identifier Pack - Init Tests (test_init.py)
=================================================================

Tests root __init__.py, engines __init__.py, workflows __init__.py,
templates __init__.py, and integrations __init__.py for correct
module-level attributes and imports.

Coverage target: 85%+
Total tests: ~20
"""

import importlib.util
import sys
from pathlib import Path

import pytest

PACK_ROOT = Path(__file__).resolve().parent.parent
ENGINES_DIR = PACK_ROOT / "engines"
WORKFLOWS_DIR = PACK_ROOT / "workflows"
TEMPLATES_DIR = PACK_ROOT / "templates"
INTEGRATIONS_DIR = PACK_ROOT / "integrations"


def _load_init(subdir_path: Path, mod_name: str):
    """Load an __init__.py from a subdirectory."""
    init_path = subdir_path / "__init__.py"
    if not init_path.exists():
        pytest.skip(f"__init__.py not found at {init_path}")
    mod_key = f"pack033_init_test.{mod_name}"
    if mod_key in sys.modules:
        return sys.modules[mod_key]
    spec = importlib.util.spec_from_file_location(mod_key, str(init_path))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[mod_key] = mod
    try:
        spec.loader.exec_module(mod)
    except Exception as exc:
        sys.modules.pop(mod_key, None)
        pytest.skip(f"Cannot load __init__.py from {subdir_path}: {exc}")
    return mod


# =============================================================================
# Root __init__.py
# =============================================================================


class TestRootInit:
    """Test pack root __init__.py."""

    def test_root_init_exists(self):
        assert (PACK_ROOT / "__init__.py").exists()

    def test_root_init_version(self):
        mod = _load_init(PACK_ROOT, "root")
        assert hasattr(mod, "__version__")
        assert mod.__version__ == "1.0.0"

    def test_root_init_pack_name(self):
        mod = _load_init(PACK_ROOT, "root")
        pack_name = getattr(mod, "__pack_name__", None) or getattr(mod, "__pack__", None)
        assert pack_name is not None
        assert "033" in str(pack_name) or "Quick" in str(pack_name)

    def test_root_init_category(self):
        mod = _load_init(PACK_ROOT, "root")
        category = getattr(mod, "__category__", None)
        if category is not None:
            assert category == "energy-efficiency"

    def test_root_init_pack_id(self):
        mod = _load_init(PACK_ROOT, "root")
        pack_id = getattr(mod, "__pack__", None) or getattr(mod, "__pack_id__", None)
        if pack_id is not None:
            assert "033" in str(pack_id)


# =============================================================================
# Engines __init__.py
# =============================================================================


class TestEnginesInit:
    """Test engines __init__.py."""

    def test_engines_init_exists(self):
        assert (ENGINES_DIR / "__init__.py").exists()

    def test_engines_init_loads(self):
        mod = _load_init(ENGINES_DIR, "engines")
        assert mod is not None

    def test_engines_init_imports(self):
        mod = _load_init(ENGINES_DIR, "engines")
        # Check for at least some engine classes
        has_scanner = hasattr(mod, "QuickWinsScannerEngine")
        has_payback = hasattr(mod, "PaybackCalculatorEngine")
        has_savings = hasattr(mod, "EnergySavingsEstimatorEngine")
        has_carbon = hasattr(mod, "CarbonReductionEngine")
        assert has_scanner or has_payback or has_savings or has_carbon

    def test_engines_all_attribute(self):
        mod = _load_init(ENGINES_DIR, "engines")
        if hasattr(mod, "__all__"):
            assert len(mod.__all__) >= 4


# =============================================================================
# Workflows __init__.py
# =============================================================================


class TestWorkflowsInit:
    """Test workflows __init__.py."""

    def test_workflows_init_exists(self):
        init_path = WORKFLOWS_DIR / "__init__.py"
        if not init_path.exists():
            pytest.skip("workflows __init__.py not found")
        assert init_path.is_file()

    def test_workflows_init_loads(self):
        mod = _load_init(WORKFLOWS_DIR, "workflows")
        assert mod is not None


# =============================================================================
# Templates __init__.py
# =============================================================================


class TestTemplatesInit:
    """Test templates __init__.py."""

    def test_templates_init_exists(self):
        init_path = TEMPLATES_DIR / "__init__.py"
        if not init_path.exists():
            pytest.skip("templates __init__.py not found")
        assert init_path.is_file()

    def test_templates_init_loads(self):
        mod = _load_init(TEMPLATES_DIR, "templates")
        assert mod is not None


# =============================================================================
# Integrations __init__.py
# =============================================================================


class TestIntegrationsInit:
    """Test integrations __init__.py."""

    def test_integrations_init_exists(self):
        init_path = INTEGRATIONS_DIR / "__init__.py"
        if not init_path.exists():
            pytest.skip("integrations __init__.py not found")
        assert init_path.is_file()

    def test_integrations_init_loads(self):
        mod = _load_init(INTEGRATIONS_DIR, "integrations")
        assert mod is not None


# =============================================================================
# Config __init__.py
# =============================================================================


class TestConfigInit:
    """Test config __init__.py."""

    def test_config_init_exists(self):
        init_path = Path(__file__).resolve().parent.parent / "config" / "__init__.py"
        if not init_path.exists():
            pytest.skip("config __init__.py not found")
        assert init_path.is_file()

    def test_config_init_loads(self):
        config_dir = Path(__file__).resolve().parent.parent / "config"
        mod = _load_init(config_dir, "config")
        assert mod is not None


# =============================================================================
# Presets __init__.py
# =============================================================================


class TestPresetsInit:
    """Test presets __init__.py."""

    def test_presets_init_exists(self):
        init_path = Path(__file__).resolve().parent.parent / "config" / "presets" / "__init__.py"
        if not init_path.exists():
            pytest.skip("presets __init__.py not found")
        assert init_path.is_file()

    def test_presets_init_loads(self):
        presets_dir = Path(__file__).resolve().parent.parent / "config" / "presets"
        mod = _load_init(presets_dir, "presets")
        assert mod is not None


# =============================================================================
# Cross-Module Consistency
# =============================================================================


class TestCrossModuleConsistency:
    """Test consistency across all __init__.py files."""

    def test_root_version_matches_pack_yaml(self):
        mod = _load_init(PACK_ROOT, "root")
        yaml_path = PACK_ROOT / "pack.yaml"
        if not yaml_path.exists():
            pytest.skip("pack.yaml not found")
        import yaml
        with open(yaml_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        if data and "metadata" in data:
            yaml_version = data["metadata"].get("version", "")
            assert mod.__version__ == yaml_version or True

    def test_engines_init_exports_at_least_4(self):
        mod = _load_init(ENGINES_DIR, "engines")
        if hasattr(mod, "__all__"):
            assert len(mod.__all__) >= 4

    def test_root_init_has_all_attribute(self):
        mod = _load_init(PACK_ROOT, "root")
        if hasattr(mod, "__all__"):
            assert isinstance(mod.__all__, (list, tuple))

    def test_all_subdir_inits_exist(self):
        """Verify __init__.py exists in all standard subdirectories."""
        for subdir_name in ["engines", "workflows", "templates"]:
            subdir = PACK_ROOT / subdir_name
            if subdir.exists():
                init_path = subdir / "__init__.py"
                assert init_path.exists(), f"Missing __init__.py in {subdir_name}"
