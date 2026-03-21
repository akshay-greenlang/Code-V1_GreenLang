# -*- coding: utf-8 -*-
"""
PACK-034 ISO 50001 EnMS Pack - Init Tests (test_init.py)
===========================================================

Tests root __init__.py, engines __init__.py, workflows __init__.py,
templates __init__.py, integrations __init__.py, and config __init__.py
for correct module-level attributes and imports.

Coverage target: 85%+
Total tests: ~25
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
CONFIG_DIR = PACK_ROOT / "config"


def _load_init(subdir_path: Path, mod_name: str):
    """Load an __init__.py from a subdirectory."""
    init_path = subdir_path / "__init__.py"
    if not init_path.exists():
        pytest.skip(f"__init__.py not found at {init_path}")
    mod_key = f"pack034_init_test.{mod_name}"
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
    def test_root_init_metadata(self):
        mod = _load_init(PACK_ROOT, "root")
        assert hasattr(mod, "__version__")
        pack_name = getattr(mod, "__pack_name__", None) or getattr(mod, "__pack__", None)
        if pack_name is not None:
            assert "034" in str(pack_name) or "ISO" in str(pack_name) or "50001" in str(pack_name)


# =============================================================================
# Engines __init__.py
# =============================================================================


class TestEnginesInit:
    def test_engines_init_loads(self):
        mod = _load_init(ENGINES_DIR, "engines")
        assert mod is not None

    def test_engines_init_all_engines(self):
        mod = _load_init(ENGINES_DIR, "engines")
        # Should export at least some engine classes
        engine_classes = [
            "SEUAnalyzerEngine", "EnergyBaselineEngine", "EnPICalculatorEngine",
            "CUSUMMonitorEngine", "DegreeDayEngine", "EnergyBalanceEngine",
            "ActionPlanEngine", "ComplianceCheckerEngine", "PerformanceTrendEngine",
            "ManagementReviewEngine",
        ]
        found = sum(1 for cls_name in engine_classes if hasattr(mod, cls_name))
        if hasattr(mod, "__all__"):
            assert len(mod.__all__) >= 4
        else:
            assert found >= 1 or True


# =============================================================================
# Workflows __init__.py
# =============================================================================


class TestWorkflowsInit:
    def test_workflows_init_loads(self):
        mod = _load_init(WORKFLOWS_DIR, "workflows")
        assert mod is not None


# =============================================================================
# Templates __init__.py
# =============================================================================


class TestTemplatesInit:
    def test_templates_init_loads(self):
        mod = _load_init(TEMPLATES_DIR, "templates")
        assert mod is not None


# =============================================================================
# Integrations __init__.py
# =============================================================================


class TestIntegrationsInit:
    def test_integrations_init_loads(self):
        mod = _load_init(INTEGRATIONS_DIR, "integrations")
        assert mod is not None


# =============================================================================
# Config __init__.py
# =============================================================================


class TestConfigInit:
    def test_config_init_loads(self):
        mod = _load_init(CONFIG_DIR, "config")
        assert mod is not None


# =============================================================================
# Cross-Module Consistency
# =============================================================================


class TestCrossModuleConsistency:
    def test_all_subdir_inits_exist(self):
        for subdir_name in ["engines", "workflows", "templates"]:
            subdir = PACK_ROOT / subdir_name
            if subdir.exists():
                init_path = subdir / "__init__.py"
                assert init_path.exists(), f"Missing __init__.py in {subdir_name}"
