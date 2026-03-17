# -*- coding: utf-8 -*-
"""
PACK-012 CSRD Financial Service Pack - Demo / Smoke Tests
============================================================

End-to-end smoke tests verifying that all modules are importable,
pack.yaml is present and well-formed, demo_config.yaml is loadable,
preset configs exist, and all engine/workflow/template/integration
modules can be loaded without error.

Self-contained: does NOT import from conftest.

Author: GreenLang QA Team
Version: 1.0.0
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
        """pack.yaml has required metadata fields for PACK-012."""
        pack_yaml = PACK_ROOT / "pack.yaml"
        with open(pack_yaml, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        metadata = data.get("metadata", {})
        assert metadata.get("name") == "PACK-012-csrd-financial-service"
        assert metadata.get("version") == "1.0.0"
        assert metadata.get("display_name") == "CSRD Financial Service Pack"
        assert metadata.get("category") == "eu-compliance"
        assert metadata.get("tier") == "sector-specific"

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
        """pack.yaml declares 6 institution-type presets."""
        pack_yaml = PACK_ROOT / "pack.yaml"
        with open(pack_yaml, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        components = data.get("components", {})
        presets = components.get("presets", [])
        assert len(presets) == 6
        preset_ids = [p["id"] for p in presets]
        assert "bank" in preset_ids
        assert "insurance" in preset_ids
        assert "asset_manager" in preset_ids
        assert "investment_firm" in preset_ids
        assert "pension_fund" in preset_ids
        assert "conglomerate" in preset_ids

    def test_pack_yaml_dependencies_total(self):
        """pack.yaml dependencies declare 72 total agents."""
        pack_yaml = PACK_ROOT / "pack.yaml"
        with open(pack_yaml, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        deps = data.get("dependencies", {})
        assert deps.get("total_agents") == 72


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
        """demo_config.yaml has demo_mode=true."""
        demo_yaml = PACK_ROOT / "config" / "demo" / "demo_config.yaml"
        with open(demo_yaml, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        assert data.get("demo_mode") is True

    def test_demo_config_institution_type(self):
        """demo_config.yaml specifies BANK institution type."""
        demo_yaml = PACK_ROOT / "config" / "demo" / "demo_config.yaml"
        with open(demo_yaml, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        assert data.get("institution_type") == "BANK"

    def test_demo_config_pcaf_section(self):
        """demo_config.yaml has pcaf section with enabled=true."""
        demo_yaml = PACK_ROOT / "config" / "demo" / "demo_config.yaml"
        with open(demo_yaml, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        pcaf = data.get("pcaf", {})
        assert pcaf.get("enabled") is True

    def test_demo_config_loadable_as_pack_config(self):
        """demo_config.yaml can be loaded as a PackConfig instance."""
        demo_yaml = PACK_ROOT / "config" / "demo" / "demo_config.yaml"
        if not demo_yaml.exists():
            pytest.skip("demo_config.yaml not found")

        config_path = PACK_ROOT / "config" / "pack_config.py"
        if not config_path.exists():
            pytest.skip("pack_config.py not found")

        mod = _import_from_path("fs12_demo_test_config", str(config_path))
        pc = mod.PackConfig.from_yaml(demo_yaml)
        assert pc.pack is not None
        assert pc.pack.institution_type.value == "BANK"


# ---------------------------------------------------------------------------
# Preset Config Tests
# ---------------------------------------------------------------------------


class TestPresetConfigs:
    """Tests for preset YAML configuration files."""

    ALL_PRESETS = [
        "bank", "insurance", "asset_manager",
        "investment_firm", "pension_fund", "conglomerate",
    ]

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
        "financed_emissions_engine.py",
        "insurance_underwriting_engine.py",
        "green_asset_ratio_engine.py",
        "btar_calculator_engine.py",
        "climate_risk_scoring_engine.py",
        "fs_double_materiality_engine.py",
        "fs_transition_plan_engine.py",
        "pillar3_esg_engine.py",
    ]

    @pytest.mark.parametrize("engine_file", ENGINE_FILES)
    def test_engine_importable(self, engine_file):
        """Each engine module can be loaded without error."""
        file_path = str(PACK_ROOT / "engines" / engine_file)
        module_name = f"pack012_smoke_engine_{engine_file.replace('.py', '')}"
        mod = _import_from_path(module_name, file_path)
        assert mod is not None


class TestWorkflowModulesImportable:
    """Verify all 8 workflow modules can be imported."""

    WORKFLOW_FILES = [
        "financed_emissions_workflow.py",
        "gar_btar_workflow.py",
        "insurance_emissions_workflow.py",
        "climate_stress_test_workflow.py",
        "fs_materiality_workflow.py",
        "transition_plan_workflow.py",
        "pillar3_reporting_workflow.py",
        "regulatory_integration_workflow.py",
    ]

    @pytest.mark.parametrize("wf_file", WORKFLOW_FILES)
    def test_workflow_importable(self, wf_file):
        """Each workflow module can be loaded without error."""
        file_path = str(PACK_ROOT / "workflows" / wf_file)
        module_name = f"pack012_smoke_wf_{wf_file.replace('.py', '')}"
        mod = _import_from_path(module_name, file_path)
        assert mod is not None


class TestTemplateModulesImportable:
    """Verify all 8 template modules can be imported."""

    TEMPLATE_FILES = [
        "pcaf_report.py",
        "gar_btar_report.py",
        "pillar3_esg_template.py",
        "climate_risk_report.py",
        "fs_esrs_chapter.py",
        "financed_emissions_dashboard.py",
        "insurance_esg_template.py",
        "sbti_fi_report.py",
    ]

    @pytest.mark.parametrize("tpl_file", TEMPLATE_FILES)
    def test_template_importable(self, tpl_file):
        """Each template module can be loaded without error."""
        file_path = str(PACK_ROOT / "templates" / tpl_file)
        module_name = f"pack012_smoke_tpl_{tpl_file.replace('.py', '')}"
        mod = _import_from_path(module_name, file_path)
        assert mod is not None


class TestIntegrationModulesImportable:
    """Verify all 10 integration modules can be imported."""

    INTEGRATION_FILES = [
        "pack_orchestrator.py",
        "csrd_pack_bridge.py",
        "sfdr_pack_bridge.py",
        "taxonomy_pack_bridge.py",
        "mrv_investments_bridge.py",
        "finance_agent_bridge.py",
        "climate_risk_bridge.py",
        "eba_pillar3_bridge.py",
        "health_check.py",
        "setup_wizard.py",
    ]

    @pytest.mark.parametrize("int_file", INTEGRATION_FILES)
    def test_integration_importable(self, int_file):
        """Each integration module can be loaded without error."""
        file_path = str(PACK_ROOT / "integrations" / int_file)
        module_name = f"pack012_smoke_int_{int_file.replace('.py', '')}"
        mod = _import_from_path(module_name, file_path)
        assert mod is not None


class TestConfigModuleImportable:
    """Verify configuration module can be imported."""

    def test_pack_config_importable(self):
        """pack_config.py can be loaded without error."""
        file_path = str(PACK_ROOT / "config" / "pack_config.py")
        module_name = "pack012_smoke_pack_config"
        mod = _import_from_path(module_name, file_path)
        assert mod is not None

    def test_engines_init_importable(self):
        """engines/__init__.py can be loaded without error."""
        file_path = str(PACK_ROOT / "engines" / "__init__.py")
        module_name = "pack012_smoke_engines_init"
        mod = _import_from_path(module_name, file_path)
        assert mod is not None

    def test_engines_init_exports_all_engines(self):
        """engines/__init__.py exports all 8 engine classes."""
        file_path = str(PACK_ROOT / "engines" / "__init__.py")
        module_name = "pack012_smoke_engines_exports"
        mod = _import_from_path(module_name, file_path)
        expected_engines = [
            "FinancedEmissionsEngine",
            "InsuranceUnderwritingEngine",
            "GreenAssetRatioEngine",
            "BTARCalculatorEngine",
            "ClimateRiskScoringEngine",
            "FSDoubleMaterialityEngine",
            "FSTransitionPlanEngine",
            "Pillar3ESGEngine",
        ]
        for engine_name in expected_engines:
            assert hasattr(mod, engine_name), (
                f"engines/__init__.py missing export: {engine_name}"
            )


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
        """Expected subdirectory exists."""
        dir_path = PACK_ROOT / subdir
        assert dir_path.is_dir(), f"Directory not found: {dir_path}"

    def test_presets_directory_exists(self):
        """config/presets directory exists."""
        presets_dir = PACK_ROOT / "config" / "presets"
        assert presets_dir.is_dir(), f"Presets dir not found: {presets_dir}"

    def test_demo_directory_exists(self):
        """config/demo directory exists."""
        demo_dir = PACK_ROOT / "config" / "demo"
        assert demo_dir.is_dir(), f"Demo dir not found: {demo_dir}"
