# -*- coding: utf-8 -*-
"""
Manifest tests for PACK-032 Building Energy Assessment

Tests pack.yaml structure, engine definitions, workflow definitions,
template definitions, integration definitions, and preset loading.

Target: 60+ tests
Author: GL-TestEngineer
"""

import importlib.util
import sys
from pathlib import Path
from typing import Any, Dict

import pytest
import yaml

PACK_ROOT = Path(__file__).resolve().parent.parent
ENGINES_DIR = PACK_ROOT / "engines"
WORKFLOWS_DIR = PACK_ROOT / "workflows"
TEMPLATES_DIR = PACK_ROOT / "templates"
INTEGRATIONS_DIR = PACK_ROOT / "integrations"
CONFIG_DIR = PACK_ROOT / "config"
PRESETS_DIR = CONFIG_DIR / "presets"


@pytest.fixture(scope="module")
def manifest() -> Dict[str, Any]:
    path = PACK_ROOT / "pack.yaml"
    if not path.exists():
        pytest.skip("pack.yaml not found")
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    assert data is not None
    return data


# =========================================================================
# Test Manifest Structure
# =========================================================================


class TestManifestStructure:
    def test_manifest_is_dict(self, manifest):
        assert isinstance(manifest, dict)

    def test_has_metadata(self, manifest):
        assert "metadata" in manifest

    def test_metadata_name(self, manifest):
        assert manifest["metadata"]["name"] == "PACK-032-building-energy-assessment"

    def test_metadata_version(self, manifest):
        assert manifest["metadata"]["version"] == "1.0.0"

    def test_metadata_display_name(self, manifest):
        assert "display_name" in manifest["metadata"]
        assert "Building Energy" in manifest["metadata"]["display_name"]

    def test_metadata_category(self, manifest):
        assert manifest["metadata"]["category"] == "energy-efficiency"

    def test_metadata_tier(self, manifest):
        assert manifest["metadata"]["tier"] in ("starter", "professional", "enterprise")

    def test_metadata_author(self, manifest):
        assert "author" in manifest["metadata"]

    def test_metadata_min_platform_version(self, manifest):
        assert "min_platform_version" in manifest["metadata"]

    def test_metadata_release_date(self, manifest):
        assert "release_date" in manifest["metadata"]

    def test_metadata_tags(self, manifest):
        tags = manifest["metadata"].get("tags", [])
        assert isinstance(tags, list)
        assert len(tags) > 5

    def test_has_regulation(self, manifest):
        assert "regulation" in manifest["metadata"]

    def test_primary_regulation(self, manifest):
        reg = manifest["metadata"]["regulation"]
        assert "primary" in reg
        assert "EPBD" in reg["primary"]["name"] or "Energy Performance" in reg["primary"]["name"]

    def test_secondary_regulations(self, manifest):
        reg = manifest["metadata"]["regulation"]
        assert "secondary" in reg
        assert len(reg["secondary"]) >= 5

    def test_compliance_references(self, manifest):
        refs = manifest["metadata"].get("compliance_references", [])
        assert isinstance(refs, list)


# =========================================================================
# Test Engine Definitions
# =========================================================================


EXPECTED_ENGINES = [
    "building_envelope_engine",
    "epc_rating_engine",
    "hvac_assessment_engine",
    "domestic_hot_water_engine",
    "lighting_assessment_engine",
    "renewable_integration_engine",
    "building_benchmark_engine",
    "retrofit_analysis_engine",
    "indoor_environment_engine",
    "whole_life_carbon_engine",
]


class TestEngineDefinitions:
    @pytest.mark.parametrize("engine_file", EXPECTED_ENGINES)
    def test_engine_file_exists(self, engine_file):
        path = ENGINES_DIR / f"{engine_file}.py"
        assert path.exists(), f"Engine file missing: {engine_file}.py"

    def test_engine_count(self):
        py_files = [f for f in ENGINES_DIR.glob("*.py") if f.name != "__init__.py"]
        assert len(py_files) >= 10

    def test_engines_init_exists(self):
        assert (ENGINES_DIR / "__init__.py").exists()

    def test_manifest_defines_engines(self, manifest):
        engines = manifest.get("engines", [])
        assert isinstance(engines, (list, dict))


# =========================================================================
# Test Workflow Definitions
# =========================================================================


EXPECTED_WORKFLOWS = [
    "initial_building_assessment_workflow",
    "epc_generation_workflow",
    "retrofit_planning_workflow",
    "certification_assessment_workflow",
    "nzeb_readiness_workflow",
    "regulatory_compliance_workflow",
    "continuous_building_monitoring_workflow",
    "tenant_engagement_workflow",
]


class TestWorkflowDefinitions:
    @pytest.mark.parametrize("wf_file", EXPECTED_WORKFLOWS)
    def test_workflow_file_exists(self, wf_file):
        path = WORKFLOWS_DIR / f"{wf_file}.py"
        assert path.exists(), f"Workflow file missing: {wf_file}.py"

    def test_workflow_count(self):
        py_files = [f for f in WORKFLOWS_DIR.glob("*.py") if f.name != "__init__.py"]
        assert len(py_files) >= 8

    def test_workflows_init_exists(self):
        assert (WORKFLOWS_DIR / "__init__.py").exists()


# =========================================================================
# Test Template Definitions
# =========================================================================


EXPECTED_TEMPLATES = [
    "building_assessment_report",
    "epc_report",
    "dec_report",
    "building_benchmark_report",
    "retrofit_recommendation_report",
    "certification_scorecard",
    "regulatory_compliance_report",
    "building_dashboard",
    "tenant_energy_report",
    "whole_life_carbon_report",
]


class TestTemplateDefinitions:
    @pytest.mark.parametrize("tpl_file", EXPECTED_TEMPLATES)
    def test_template_file_exists(self, tpl_file):
        path = TEMPLATES_DIR / f"{tpl_file}.py"
        assert path.exists(), f"Template file missing: {tpl_file}.py"

    def test_template_count(self):
        py_files = [f for f in TEMPLATES_DIR.glob("*.py") if f.name != "__init__.py"]
        assert len(py_files) >= 10

    def test_templates_init_exists(self):
        assert (TEMPLATES_DIR / "__init__.py").exists()


# =========================================================================
# Test Integration Definitions
# =========================================================================


EXPECTED_INTEGRATIONS = [
    "pack_orchestrator",
    "data_building_bridge",
    "mrv_building_bridge",
    "bms_integration_bridge",
    "weather_data_bridge",
    "grid_carbon_bridge",
    "crrem_pathway_bridge",
    "certification_bridge",
    "epbd_compliance_bridge",
    "property_registry_bridge",
    "health_check",
    "setup_wizard",
]


class TestIntegrationDefinitions:
    @pytest.mark.parametrize("int_file", EXPECTED_INTEGRATIONS)
    def test_integration_file_exists(self, int_file):
        path = INTEGRATIONS_DIR / f"{int_file}.py"
        assert path.exists(), f"Integration file missing: {int_file}.py"

    def test_integration_count(self):
        py_files = [f for f in INTEGRATIONS_DIR.glob("*.py") if f.name != "__init__.py"]
        assert len(py_files) >= 12

    def test_integrations_init_exists(self):
        assert (INTEGRATIONS_DIR / "__init__.py").exists()


# =========================================================================
# Test Presets
# =========================================================================


EXPECTED_PRESETS = [
    "commercial_office",
    "retail_building",
    "hotel_hospitality",
    "healthcare_facility",
    "education_building",
    "residential_multifamily",
    "mixed_use_development",
    "public_sector_building",
]


class TestPresets:
    @pytest.mark.parametrize("preset", EXPECTED_PRESETS)
    def test_preset_file_exists(self, preset):
        path = PRESETS_DIR / f"{preset}.yaml"
        assert path.exists(), f"Preset file missing: {preset}.yaml"

    @pytest.mark.parametrize("preset", EXPECTED_PRESETS)
    def test_preset_valid_yaml(self, preset):
        path = PRESETS_DIR / f"{preset}.yaml"
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        assert data is not None

    def test_preset_count(self):
        yaml_files = list(PRESETS_DIR.glob("*.yaml"))
        assert len(yaml_files) >= 8


# =========================================================================
# Test Config Directory
# =========================================================================


class TestConfigDirectory:
    def test_pack_config_exists(self):
        assert (CONFIG_DIR / "pack_config.py").exists()

    def test_config_init_exists(self):
        assert (CONFIG_DIR / "__init__.py").exists()

    def test_demo_dir_exists(self):
        assert (CONFIG_DIR / "demo").exists()

    def test_demo_config_exists(self):
        assert (CONFIG_DIR / "demo" / "demo_config.yaml").exists()


# =========================================================================
# Test Pack Root
# =========================================================================


class TestPackRoot:
    def test_pack_yaml_exists(self):
        assert (PACK_ROOT / "pack.yaml").exists()

    def test_init_exists(self):
        assert (PACK_ROOT / "__init__.py").exists()

    def test_engines_dir(self):
        assert ENGINES_DIR.is_dir()

    def test_workflows_dir(self):
        assert WORKFLOWS_DIR.is_dir()

    def test_templates_dir(self):
        assert TEMPLATES_DIR.is_dir()

    def test_integrations_dir(self):
        assert INTEGRATIONS_DIR.is_dir()

    def test_config_dir(self):
        assert CONFIG_DIR.is_dir()
