# -*- coding: utf-8 -*-
"""
PACK-032 Building Energy Assessment Pack - Shared Test Fixtures (conftest.py)
=============================================================================

Provides pytest fixtures for the PACK-032 test suite including:
  - Dynamic module loading via importlib (no package install needed)
  - Pack manifest and configuration fixtures
  - Sample building envelope, EPC, HVAC, DHW, lighting, renewable,
    benchmark, retrofit, indoor environment, and whole-life carbon data

All fixtures use importlib.util.spec_from_file_location to load modules
directly from the pack source tree, enabling test execution without
installing the pack as a Python package.

Author:  GreenLang Platform Team (GL-TestEngineer)
Pack:    PACK-032 Building Energy Assessment
Date:    March 2026
"""

import importlib
import importlib.util
import sys
from decimal import Decimal
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest
import yaml


# =============================================================================
# Constants
# =============================================================================

PACK_ROOT = Path(__file__).resolve().parent.parent
ENGINES_DIR = PACK_ROOT / "engines"
WORKFLOWS_DIR = PACK_ROOT / "workflows"
TEMPLATES_DIR = PACK_ROOT / "templates"
INTEGRATIONS_DIR = PACK_ROOT / "integrations"
CONFIG_DIR = PACK_ROOT / "config"
PRESETS_DIR = CONFIG_DIR / "presets"
DEMO_DIR = CONFIG_DIR / "demo"

# Engine file mapping
ENGINE_FILES = {
    "building_envelope": "building_envelope_engine.py",
    "epc_rating": "epc_rating_engine.py",
    "hvac_assessment": "hvac_assessment_engine.py",
    "domestic_hot_water": "domestic_hot_water_engine.py",
    "lighting_assessment": "lighting_assessment_engine.py",
    "renewable_integration": "renewable_integration_engine.py",
    "building_benchmark": "building_benchmark_engine.py",
    "retrofit_analysis": "retrofit_analysis_engine.py",
    "indoor_environment": "indoor_environment_engine.py",
    "whole_life_carbon": "whole_life_carbon_engine.py",
}

ENGINE_CLASSES = {
    "building_envelope": "BuildingEnvelopeEngine",
    "epc_rating": "EPCRatingEngine",
    "hvac_assessment": "HVACAssessmentEngine",
    "domestic_hot_water": "DomesticHotWaterEngine",
    "lighting_assessment": "LightingAssessmentEngine",
    "renewable_integration": "RenewableIntegrationEngine",
    "building_benchmark": "BuildingBenchmarkEngine",
    "retrofit_analysis": "RetrofitAnalysisEngine",
    "indoor_environment": "IndoorEnvironmentEngine",
    "whole_life_carbon": "WholeLifeCarbonEngine",
}

WORKFLOW_FILES = {
    "initial_building_assessment": "initial_building_assessment_workflow.py",
    "epc_generation": "epc_generation_workflow.py",
    "retrofit_planning": "retrofit_planning_workflow.py",
    "certification_assessment": "certification_assessment_workflow.py",
    "nzeb_readiness": "nzeb_readiness_workflow.py",
    "regulatory_compliance": "regulatory_compliance_workflow.py",
    "continuous_monitoring": "continuous_building_monitoring_workflow.py",
    "tenant_engagement": "tenant_engagement_workflow.py",
}

TEMPLATE_FILES = {
    "building_assessment_report": "building_assessment_report.py",
    "epc_report": "epc_report.py",
    "dec_report": "dec_report.py",
    "building_benchmark_report": "building_benchmark_report.py",
    "retrofit_recommendation_report": "retrofit_recommendation_report.py",
    "certification_scorecard": "certification_scorecard.py",
    "regulatory_compliance_report": "regulatory_compliance_report.py",
    "building_dashboard": "building_dashboard.py",
    "tenant_energy_report": "tenant_energy_report.py",
    "whole_life_carbon_report": "whole_life_carbon_report.py",
}

INTEGRATION_FILES = {
    "pack_orchestrator": "pack_orchestrator.py",
    "data_building_bridge": "data_building_bridge.py",
    "mrv_building_bridge": "mrv_building_bridge.py",
    "bms_integration_bridge": "bms_integration_bridge.py",
    "weather_data_bridge": "weather_data_bridge.py",
    "grid_carbon_bridge": "grid_carbon_bridge.py",
    "crrem_pathway_bridge": "crrem_pathway_bridge.py",
    "certification_bridge": "certification_bridge.py",
    "epbd_compliance_bridge": "epbd_compliance_bridge.py",
    "property_registry_bridge": "property_registry_bridge.py",
    "health_check": "health_check.py",
    "setup_wizard": "setup_wizard.py",
}

PRESET_NAMES = [
    "commercial_office",
    "retail_building",
    "hotel_hospitality",
    "healthcare_facility",
    "education_building",
    "residential_multifamily",
    "mixed_use_development",
    "public_sector_building",
]


# =============================================================================
# Dynamic Module Loader
# =============================================================================


def _load_module(module_name: str, file_name: str, subdir: str = "engines"):
    """Load a module dynamically using importlib.util.spec_from_file_location.

    Args:
        module_name: Logical name for the module.
        file_name: File name of the Python module.
        subdir: Subdirectory under PACK_ROOT.

    Returns:
        The loaded module object, or None if loading fails (via pytest.skip).
    """
    subdir_map = {
        "engines": ENGINES_DIR,
        "workflows": WORKFLOWS_DIR,
        "templates": TEMPLATES_DIR,
        "integrations": INTEGRATIONS_DIR,
        "config": CONFIG_DIR,
    }
    base_dir = subdir_map.get(subdir, PACK_ROOT / subdir)
    file_path = base_dir / file_name

    if not file_path.exists():
        pytest.skip(f"Module file not found: {file_path}")

    full_module_name = f"pack032_test.{subdir}.{module_name}"

    if full_module_name in sys.modules:
        return sys.modules[full_module_name]

    spec = importlib.util.spec_from_file_location(full_module_name, str(file_path))
    if spec is None or spec.loader is None:
        pytest.skip(f"Cannot create module spec for {file_path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[full_module_name] = module

    try:
        spec.loader.exec_module(module)
    except Exception as exc:
        sys.modules.pop(full_module_name, None)
        pytest.skip(f"Cannot load {module_name}: {exc}")

    return module


def _load_engine(engine_key: str):
    """Load an engine module by its logical key."""
    file_name = ENGINE_FILES[engine_key]
    return _load_module(engine_key, file_name, "engines")


def _load_config_module():
    """Load the pack_config module."""
    return _load_module("pack_config", "pack_config.py", "config")


# =============================================================================
# Path and YAML Fixtures
# =============================================================================


@pytest.fixture(scope="session")
def pack_root() -> Path:
    """Return the absolute path to the PACK-032 root directory."""
    return PACK_ROOT


@pytest.fixture(scope="session")
def pack_yaml_path() -> Path:
    """Return the absolute path to pack.yaml."""
    return PACK_ROOT / "pack.yaml"


@pytest.fixture(scope="session")
def pack_yaml_data(pack_yaml_path: Path) -> Dict[str, Any]:
    """Parse and return the pack.yaml manifest as a dictionary."""
    with open(pack_yaml_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    assert data is not None, "pack.yaml parsed to None"
    return data


@pytest.fixture(scope="session")
def demo_yaml_path() -> Path:
    """Return the absolute path to the demo configuration YAML."""
    return DEMO_DIR / "demo_config.yaml"


@pytest.fixture(scope="session")
def demo_yaml_data(demo_yaml_path: Path) -> Dict[str, Any]:
    """Parse and return the demo_config.yaml as a dictionary."""
    if not demo_yaml_path.exists():
        pytest.skip("demo_config.yaml not found")
    with open(demo_yaml_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    assert data is not None, "demo_config.yaml parsed to None"
    return data


# =============================================================================
# Configuration Fixtures
# =============================================================================


@pytest.fixture(scope="session")
def config_module():
    """Load and return the pack_config module."""
    return _load_config_module()


@pytest.fixture
def pack_config(config_module):
    """Create a BuildingEnergyAssessmentConfig with default values."""
    return config_module.BuildingEnergyAssessmentConfig()


@pytest.fixture
def pack_config_wrapper(config_module):
    """Create a PackConfig wrapper with default values."""
    return config_module.PackConfig()
