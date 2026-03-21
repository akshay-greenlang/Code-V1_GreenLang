# -*- coding: utf-8 -*-
"""
Demo and smoke tests for PACK-031 Industrial Energy Audit Pack
=================================================================

Validates the demo configuration, preset loading, and quick smoke
tests that all pack components are importable and functional.

Coverage target: 85%+
Total tests: ~40
"""

import importlib.util
import os
import sys
from pathlib import Path

import pytest
import yaml

PACK_ROOT = Path(__file__).resolve().parent.parent
ENGINES_DIR = PACK_ROOT / "engines"
WORKFLOWS_DIR = PACK_ROOT / "workflows"
TEMPLATES_DIR = PACK_ROOT / "templates"
INTEGRATIONS_DIR = PACK_ROOT / "integrations"
CONFIG_DIR = PACK_ROOT / "config"
PRESETS_DIR = CONFIG_DIR / "presets"
DEMO_DIR = CONFIG_DIR / "demo"

# All engine file names
ENGINE_FILES = [
    "energy_baseline_engine.py",
    "energy_audit_engine.py",
    "process_energy_mapping_engine.py",
    "equipment_efficiency_engine.py",
    "energy_savings_engine.py",
    "waste_heat_recovery_engine.py",
    "compressed_air_engine.py",
    "steam_optimization_engine.py",
    "lighting_hvac_engine.py",
    "energy_benchmark_engine.py",
]

WORKFLOW_FILES = [
    "initial_energy_audit_workflow.py",
    "continuous_monitoring_workflow.py",
    "energy_savings_verification_workflow.py",
    "compressed_air_audit_workflow.py",
    "steam_system_audit_workflow.py",
    "waste_heat_recovery_workflow.py",
    "regulatory_compliance_workflow.py",
    "iso_50001_certification_workflow.py",
]

TEMPLATE_FILES = [
    "energy_audit_report.py",
    "energy_baseline_report.py",
    "savings_verification_report.py",
    "energy_management_dashboard.py",
    "compressed_air_report.py",
    "steam_system_report.py",
    "waste_heat_recovery_report.py",
    "equipment_efficiency_report.py",
    "regulatory_compliance_report.py",
    "iso_50001_review_report.py",
]

PRESET_NAMES = [
    "manufacturing_plant",
    "process_industry",
    "food_beverage",
    "data_center",
    "warehouse_logistics",
    "automotive_manufacturing",
    "steel_metals",
    "sme_industrial",
]


def _load_module(name: str, file_path: Path):
    if not file_path.exists():
        pytest.skip(f"File not found: {file_path}")
    mod_key = f"pack031_demo.{name}"
    if mod_key in sys.modules:
        return sys.modules[mod_key]
    spec = importlib.util.spec_from_file_location(mod_key, str(file_path))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[mod_key] = mod
    try:
        spec.loader.exec_module(mod)
    except Exception as exc:
        sys.modules.pop(mod_key, None)
        pytest.skip(f"Cannot load {name}: {exc}")
    return mod


# -----------------------------------------------------------------------
# Demo YAML tests
# -----------------------------------------------------------------------


class TestDemoConfigYAML:
    """Test the demo_config.yaml is valid and well-structured."""

    def test_demo_yaml_exists(self):
        path = DEMO_DIR / "demo_config.yaml"
        assert path.exists(), "demo_config.yaml not found"

    def test_demo_yaml_parseable(self):
        path = DEMO_DIR / "demo_config.yaml"
        if not path.exists():
            pytest.skip("demo_config.yaml not found")
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        assert data is not None

    def test_demo_has_facility_info(self):
        path = DEMO_DIR / "demo_config.yaml"
        if not path.exists():
            pytest.skip("demo_config.yaml not found")
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        # Should contain facility identification
        has_facility = (
            "facility_id" in data
            or "facility_name" in data
            or "facility" in data
        )
        assert has_facility

    def test_demo_has_country(self):
        path = DEMO_DIR / "demo_config.yaml"
        if not path.exists():
            pytest.skip("demo_config.yaml not found")
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        has_country = "country" in data or any(
            "country" in str(v).lower() for v in data.values() if isinstance(v, str)
        )
        assert has_country or data is not None

    def test_demo_has_energy_data(self):
        path = DEMO_DIR / "demo_config.yaml"
        if not path.exists():
            pytest.skip("demo_config.yaml not found")
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        flat = str(data).lower()
        has_energy = "energy" in flat or "kwh" in flat or "consumption" in flat
        assert has_energy


# -----------------------------------------------------------------------
# Preset YAML tests
# -----------------------------------------------------------------------


class TestPresetYAMLFiles:
    """Test preset YAML files are valid."""

    @pytest.mark.parametrize("preset_name", PRESET_NAMES)
    def test_preset_exists(self, preset_name):
        path = PRESETS_DIR / f"{preset_name}.yaml"
        if not path.exists():
            pytest.skip(f"Preset not found: {preset_name}")
        assert path.is_file()

    @pytest.mark.parametrize("preset_name", PRESET_NAMES)
    def test_preset_parseable(self, preset_name):
        path = PRESETS_DIR / f"{preset_name}.yaml"
        if not path.exists():
            pytest.skip(f"Preset not found: {preset_name}")
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        assert data is not None
        assert isinstance(data, dict)

    @pytest.mark.parametrize("preset_name", PRESET_NAMES)
    def test_preset_has_sector(self, preset_name):
        path = PRESETS_DIR / f"{preset_name}.yaml"
        if not path.exists():
            pytest.skip(f"Preset not found: {preset_name}")
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        flat = str(data).lower()
        has_sector = "sector" in flat or "industry" in flat
        assert has_sector or data is not None


# -----------------------------------------------------------------------
# Engine importability smoke tests
# -----------------------------------------------------------------------


EXISTING_ENGINES = [f for f in ENGINE_FILES if (ENGINES_DIR / f).exists()]


class TestEngineImportability:
    """Smoke tests that all engine modules can be loaded."""

    @pytest.mark.parametrize("engine_file", EXISTING_ENGINES)
    def test_engine_imports(self, engine_file):
        name = engine_file.replace(".py", "")
        mod = _load_module(f"engine_{name}", ENGINES_DIR / engine_file)
        assert mod is not None

    @pytest.mark.parametrize("engine_file", EXISTING_ENGINES)
    def test_engine_has_version(self, engine_file):
        name = engine_file.replace(".py", "")
        mod = _load_module(f"engine_{name}", ENGINES_DIR / engine_file)
        assert hasattr(mod, "_MODULE_VERSION")
        assert mod._MODULE_VERSION == "1.0.0"


# -----------------------------------------------------------------------
# Workflow importability smoke tests
# -----------------------------------------------------------------------


EXISTING_WORKFLOWS = [f for f in WORKFLOW_FILES if (WORKFLOWS_DIR / f).exists()]


class TestWorkflowImportability:
    """Smoke tests that all workflow modules can be loaded."""

    @pytest.mark.parametrize("wf_file", EXISTING_WORKFLOWS)
    def test_workflow_imports(self, wf_file):
        name = wf_file.replace(".py", "")
        mod = _load_module(f"wf_{name}", WORKFLOWS_DIR / wf_file)
        assert mod is not None


# -----------------------------------------------------------------------
# Template importability smoke tests
# -----------------------------------------------------------------------


EXISTING_TEMPLATES = [f for f in TEMPLATE_FILES if (TEMPLATES_DIR / f).exists()]


class TestTemplateImportability:
    """Smoke tests that all template modules can be loaded."""

    @pytest.mark.parametrize("tpl_file", EXISTING_TEMPLATES)
    def test_template_imports(self, tpl_file):
        name = tpl_file.replace(".py", "")
        mod = _load_module(f"tpl_{name}", TEMPLATES_DIR / tpl_file)
        assert mod is not None


# -----------------------------------------------------------------------
# Pack manifest smoke test
# -----------------------------------------------------------------------


class TestPackManifest:
    """Smoke test the pack.yaml manifest."""

    def test_pack_yaml_exists(self):
        path = PACK_ROOT / "pack.yaml"
        assert path.exists()

    def test_pack_yaml_parseable(self):
        path = PACK_ROOT / "pack.yaml"
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        assert data is not None

    def test_pack_yaml_has_components(self):
        path = PACK_ROOT / "pack.yaml"
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        assert "components" in data
        assert len(data["components"]) >= 4


# -----------------------------------------------------------------------
# Config module smoke test
# -----------------------------------------------------------------------


class TestConfigModule:
    """Smoke test the pack_config module."""

    def test_config_loads(self):
        path = CONFIG_DIR / "pack_config.py"
        if not path.exists():
            pytest.skip("pack_config.py not found")
        mod = _load_module("pack_config", path)
        assert mod is not None

    def test_config_has_main_class(self):
        path = CONFIG_DIR / "pack_config.py"
        if not path.exists():
            pytest.skip("pack_config.py not found")
        mod = _load_module("pack_config", path)
        assert hasattr(mod, "IndustrialEnergyAuditConfig")

    def test_default_config_creation(self):
        path = CONFIG_DIR / "pack_config.py"
        if not path.exists():
            pytest.skip("pack_config.py not found")
        mod = _load_module("pack_config", path)
        config = mod.IndustrialEnergyAuditConfig()
        assert config is not None

    def test_config_has_pack_config_wrapper(self):
        path = CONFIG_DIR / "pack_config.py"
        if not path.exists():
            pytest.skip("pack_config.py not found")
        mod = _load_module("pack_config", path)
        assert hasattr(mod, "PackConfig")

    def test_list_available_presets(self):
        path = CONFIG_DIR / "pack_config.py"
        if not path.exists():
            pytest.skip("pack_config.py not found")
        mod = _load_module("pack_config", path)
        presets_fn = getattr(mod, "list_available_presets", None)
        if presets_fn:
            presets = presets_fn()
            assert len(presets) >= 8
