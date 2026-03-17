# -*- coding: utf-8 -*-
"""
PACK-011 SFDR Article 9 Pack - Demo / Smoke Tests
====================================================

End-to-end smoke tests verifying that all modules are importable,
pack.yaml is present and well-formed, demo_config.yaml is loadable,
preset configs exist, and all engine/workflow/template/integration
modules can be loaded without error.

Self-contained: does NOT import from conftest.
"""

import importlib
import importlib.util
import os
import sys
from pathlib import Path

import pytest
import yaml

# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------

PACK_ROOT = Path(__file__).resolve().parent.parent
PROJECT_ROOT = PACK_ROOT.parent.parent.parent


def _import_from_path(module_name: str, file_path: str):
    """Import a module from an absolute file path."""
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load spec for {file_path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------------------
# Pack Manifest Tests
# ---------------------------------------------------------------------------


class TestPackYAML:
    """Tests for pack.yaml manifest file."""

    def test_pack_yaml_exists(self):
        """pack.yaml file exists in the pack root."""
        pack_yaml = PACK_ROOT / "pack.yaml"
        assert pack_yaml.is_file(), f"pack.yaml not found at {pack_yaml}"

    def test_pack_yaml_parseable(self):
        """pack.yaml is valid YAML."""
        pack_yaml = PACK_ROOT / "pack.yaml"
        with open(pack_yaml, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        assert isinstance(data, dict)

    def test_pack_yaml_metadata(self):
        """pack.yaml has required metadata fields for PACK-011 Article 9."""
        pack_yaml = PACK_ROOT / "pack.yaml"
        with open(pack_yaml, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        metadata = data.get("metadata", {})
        assert metadata.get("name") == "PACK-011-sfdr-article-9"
        assert metadata.get("version") == "1.0.0"
        assert metadata.get("display_name") == "SFDR Article 9 Pack"
        assert metadata.get("category") == "eu-compliance"

    def test_pack_yaml_components(self):
        """pack.yaml declares 8 engines, 8 workflows, 8 templates, 10 integrations."""
        pack_yaml = PACK_ROOT / "pack.yaml"
        with open(pack_yaml, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        components = data.get("components", {})
        assert len(components.get("engines", [])) == 8
        assert len(components.get("workflows", [])) == 8
        assert len(components.get("templates", [])) == 8
        assert len(components.get("integrations", [])) == 10

    def test_pack_yaml_presets(self):
        """pack.yaml declares 5 product presets for Article 9 fund types."""
        pack_yaml = PACK_ROOT / "pack.yaml"
        with open(pack_yaml, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        presets = data.get("presets", {}).get("product_presets", [])
        assert len(presets) == 5
        preset_ids = [p["id"] for p in presets]
        assert "impact_fund" in preset_ids
        assert "climate_fund" in preset_ids
        assert "social_fund" in preset_ids
        assert "esg_leader_fund" in preset_ids
        assert "transition_fund" in preset_ids

    def test_pack_yaml_pack_summary(self):
        """pack.yaml pack_summary declares correct totals."""
        pack_yaml = PACK_ROOT / "pack.yaml"
        with open(pack_yaml, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        summary = data.get("pack_summary", {})
        assert summary.get("pack_engines") == 8
        assert summary.get("pack_workflows") == 8
        assert summary.get("pack_templates") == 8
        assert summary.get("pack_integrations") == 10
        assert summary.get("total_agents") == 50


# ---------------------------------------------------------------------------
# Demo Config Tests
# ---------------------------------------------------------------------------


class TestDemoConfig:
    """Tests for demo_config.yaml."""

    def test_demo_config_exists(self):
        """demo_config.yaml exists."""
        demo_yaml = PACK_ROOT / "config" / "demo" / "demo_config.yaml"
        assert demo_yaml.is_file(), f"demo_config.yaml not found at {demo_yaml}"

    def test_demo_config_parseable(self):
        """demo_config.yaml is valid YAML."""
        demo_yaml = PACK_ROOT / "config" / "demo" / "demo_config.yaml"
        with open(demo_yaml, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        assert isinstance(data, dict)

    def test_demo_config_has_demo_mode(self):
        """demo_config.yaml has demo_mode_enabled=true."""
        demo_yaml = PACK_ROOT / "config" / "demo" / "demo_config.yaml"
        with open(demo_yaml, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        demo = data.get("demo", {})
        assert demo.get("demo_mode_enabled") is True


# ---------------------------------------------------------------------------
# Preset Config Tests
# ---------------------------------------------------------------------------


class TestPresetConfigs:
    """Tests for preset YAML configuration files."""

    # Product presets (Article 9 fund types)
    PRODUCT_PRESETS = [
        "impact_fund",
        "climate_fund",
        "social_fund",
        "esg_leader_fund",
        "transition_fund",
    ]

    # Entity presets (institution types)
    ENTITY_PRESETS = [
        "asset_manager",
        "insurance",
        "bank",
        "pension_fund",
        "wealth_manager",
    ]

    ALL_PRESETS = PRODUCT_PRESETS + ENTITY_PRESETS

    @pytest.mark.parametrize("preset_name", ALL_PRESETS)
    def test_preset_exists(self, preset_name):
        """Each preset YAML file exists."""
        preset_path = PACK_ROOT / "config" / "presets" / f"{preset_name}.yaml"
        assert preset_path.is_file(), f"Preset not found: {preset_path}"

    @pytest.mark.parametrize("preset_name", ALL_PRESETS)
    def test_preset_parseable(self, preset_name):
        """Each preset YAML file is valid YAML."""
        preset_path = PACK_ROOT / "config" / "presets" / f"{preset_name}.yaml"
        with open(preset_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        assert isinstance(data, dict)


# ---------------------------------------------------------------------------
# Module Import Smoke Tests
# ---------------------------------------------------------------------------


class TestEngineModulesImportable:
    """Verify all 8 engine modules can be imported."""

    ENGINE_FILES = [
        "sustainable_objective_engine.py",
        "enhanced_dnsh_engine.py",
        "full_taxonomy_alignment.py",
        "impact_measurement_engine.py",
        "benchmark_alignment_engine.py",
        "pai_mandatory_engine.py",
        "carbon_trajectory_engine.py",
        "investment_universe_engine.py",
    ]

    @pytest.mark.parametrize("engine_file", ENGINE_FILES)
    def test_engine_importable(self, engine_file):
        """Each engine module can be loaded without error."""
        file_path = str(PACK_ROOT / "engines" / engine_file)
        module_name = f"pack011_smoke_engine_{engine_file.replace('.py', '')}"
        mod = _import_from_path(module_name, file_path)
        assert mod is not None


class TestWorkflowModulesImportable:
    """Verify all 8 workflow modules can be imported."""

    WORKFLOW_FILES = [
        "annex_iii_disclosure.py",
        "annex_v_reporting.py",
        "sustainable_verification.py",
        "impact_reporting.py",
        "benchmark_monitoring.py",
        "pai_mandatory_workflow.py",
        "downgrade_monitoring.py",
        "regulatory_update.py",
    ]

    @pytest.mark.parametrize("wf_file", WORKFLOW_FILES)
    def test_workflow_importable(self, wf_file):
        """Each workflow module can be loaded without error."""
        file_path = str(PACK_ROOT / "workflows" / wf_file)
        module_name = f"pack011_smoke_wf_{wf_file.replace('.py', '')}"
        mod = _import_from_path(module_name, file_path)
        assert mod is not None


class TestTemplateModulesImportable:
    """Verify all 8 template modules can be imported."""

    TEMPLATE_FILES = [
        "annex_iii_precontractual.py",
        "annex_v_periodic.py",
        "impact_report.py",
        "benchmark_methodology.py",
        "sustainable_dashboard.py",
        "pai_mandatory_report.py",
        "carbon_trajectory_report.py",
        "audit_trail_report.py",
    ]

    @pytest.mark.parametrize("tpl_file", TEMPLATE_FILES)
    def test_template_importable(self, tpl_file):
        """Each template module can be loaded without error."""
        file_path = str(PACK_ROOT / "templates" / tpl_file)
        module_name = f"pack011_smoke_tpl_{tpl_file.replace('.py', '')}"
        mod = _import_from_path(module_name, file_path)
        assert mod is not None


class TestIntegrationModulesImportable:
    """Verify all 10 integration modules can be imported."""

    INTEGRATION_FILES = [
        "pack_orchestrator.py",
        "article8_pack_bridge.py",
        "taxonomy_pack_bridge.py",
        "mrv_emissions_bridge.py",
        "benchmark_data_bridge.py",
        "impact_data_bridge.py",
        "eet_data_bridge.py",
        "regulatory_bridge.py",
        "health_check.py",
        "setup_wizard.py",
    ]

    @pytest.mark.parametrize("int_file", INTEGRATION_FILES)
    def test_integration_importable(self, int_file):
        """Each integration module can be loaded without error."""
        file_path = str(PACK_ROOT / "integrations" / int_file)
        module_name = f"pack011_smoke_int_{int_file.replace('.py', '')}"
        mod = _import_from_path(module_name, file_path)
        assert mod is not None


# ---------------------------------------------------------------------------
# Directory Structure Tests
# ---------------------------------------------------------------------------


class TestPackDirectoryStructure:
    """Verify the pack has all expected directories and key files."""

    EXPECTED_DIRS = [
        "engines",
        "workflows",
        "templates",
        "integrations",
        "config",
        "tests",
    ]

    @pytest.mark.parametrize("subdir", EXPECTED_DIRS)
    def test_directory_exists(self, subdir):
        """Each expected subdirectory exists."""
        path = PACK_ROOT / subdir
        assert path.is_dir(), f"Directory not found: {path}"

    def test_engines_init_exists(self):
        """engines/__init__.py exists."""
        assert (PACK_ROOT / "engines" / "__init__.py").is_file()

    def test_workflows_init_exists(self):
        """workflows/__init__.py exists."""
        assert (PACK_ROOT / "workflows" / "__init__.py").is_file()

    def test_templates_init_exists(self):
        """templates/__init__.py exists."""
        assert (PACK_ROOT / "templates" / "__init__.py").is_file()

    def test_integrations_init_exists(self):
        """integrations/__init__.py exists."""
        assert (PACK_ROOT / "integrations" / "__init__.py").is_file()

    def test_config_pack_config_exists(self):
        """config/pack_config.py exists."""
        assert (PACK_ROOT / "config" / "pack_config.py").is_file()

    def test_config_demo_dir_exists(self):
        """config/demo/ directory exists."""
        assert (PACK_ROOT / "config" / "demo").is_dir()

    def test_config_presets_dir_exists(self):
        """config/presets/ directory exists."""
        assert (PACK_ROOT / "config" / "presets").is_dir()
