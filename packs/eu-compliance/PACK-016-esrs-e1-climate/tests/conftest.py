# -*- coding: utf-8 -*-
"""
PACK-016 ESRS E1 Climate Pack - Shared Test Fixtures (conftest.py)
====================================================================

Provides pytest fixtures for the PACK-016 test suite including:
  - Dynamic module loading via importlib (no package install needed)
  - Pack manifest and configuration fixtures
  - E1-specific sample data (GHG inventory, energy, targets, risks)
  - Demo configuration loading

All fixtures use importlib.util.spec_from_file_location to load modules
directly from the pack source tree, enabling test execution without
installing the pack as a Python package.

Fixture Categories:
  1. Paths and YAML data
  2. Configuration objects
  3. Engine fixtures
  4. Workflow fixtures
  5. Template fixtures
  6. Integration fixtures
  7. E1-specific sample data

Author:  GreenLang Platform Team (GL-TestEngineer)
Pack:    PACK-016 ESRS E1 Climate Change
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

# Engine file mapping: logical name -> file name on disk
ENGINE_FILES = {
    "ghg_inventory": "ghg_inventory_engine.py",
    "energy_mix": "energy_mix_engine.py",
    "transition_plan": "transition_plan_engine.py",
    "climate_target": "climate_target_engine.py",
    "climate_action": "climate_action_engine.py",
    "carbon_credit": "carbon_credit_engine.py",
    "carbon_pricing": "carbon_pricing_engine.py",
    "climate_risk": "climate_risk_engine.py",
}

# Engine class names that should exist in each engine module
ENGINE_CLASSES = {
    "ghg_inventory": "GHGInventoryEngine",
    "energy_mix": "EnergyMixEngine",
    "transition_plan": "TransitionPlanEngine",
    "climate_target": "ClimateTargetEngine",
    "climate_action": "ClimateActionEngine",
    "carbon_credit": "CarbonCreditEngine",
    "carbon_pricing": "CarbonPricingEngine",
    "climate_risk": "ClimateRiskEngine",
}

# Workflow file mapping
WORKFLOW_FILES = {
    "ghg_inventory": "ghg_inventory_workflow.py",
    "energy_assessment": "energy_assessment_workflow.py",
    "transition_plan": "transition_plan_workflow.py",
    "target_setting": "target_setting_workflow.py",
    "climate_actions": "climate_actions_workflow.py",
    "carbon_credits": "carbon_credits_workflow.py",
    "carbon_pricing": "carbon_pricing_workflow.py",
    "climate_risk": "climate_risk_workflow.py",
    "full_e1": "full_e1_workflow.py",
}

# Workflow class names
WORKFLOW_CLASSES = {
    "ghg_inventory": "GHGInventoryWorkflow",
    "energy_assessment": "EnergyAssessmentWorkflow",
    "transition_plan": "TransitionPlanWorkflow",
    "target_setting": "TargetSettingWorkflow",
    "climate_actions": "ClimateActionsWorkflow",
    "carbon_credits": "CarbonCreditsWorkflow",
    "carbon_pricing": "CarbonPricingWorkflow",
    "climate_risk": "ClimateRiskWorkflow",
    "full_e1": "FullE1Workflow",
}

# Template file mapping
TEMPLATE_FILES = {
    "ghg_emissions_report": "ghg_emissions_report.py",
    "energy_mix_report": "energy_mix_report.py",
    "transition_plan_report": "transition_plan_report.py",
    "climate_policy_report": "climate_policy_report.py",
    "climate_actions_report": "climate_actions_report.py",
    "climate_targets_report": "climate_targets_report.py",
    "carbon_credits_report": "carbon_credits_report.py",
    "carbon_pricing_report": "carbon_pricing_report.py",
    "climate_risk_report": "climate_risk_report.py",
}

# Template class names
TEMPLATE_CLASSES = {
    "ghg_emissions_report": "GHGEmissionsReportTemplate",
    "energy_mix_report": "EnergyMixReportTemplate",
    "transition_plan_report": "TransitionPlanReportTemplate",
    "climate_policy_report": "ClimatePolicyReportTemplate",
    "climate_actions_report": "ClimateActionsReportTemplate",
    "climate_targets_report": "ClimateTargetsReportTemplate",
    "carbon_credits_report": "CarbonCreditsReportTemplate",
    "carbon_pricing_report": "CarbonPricingReportTemplate",
    "climate_risk_report": "ClimateRiskReportTemplate",
}

# Integration file mapping
INTEGRATION_FILES = {
    "pack_orchestrator": "pack_orchestrator.py",
    "ghg_app_bridge": "ghg_app_bridge.py",
    "mrv_agent_bridge": "mrv_agent_bridge.py",
    "dma_pack_bridge": "dma_pack_bridge.py",
    "decarbonization_bridge": "decarbonization_bridge.py",
    "adaptation_bridge": "adaptation_bridge.py",
    "health_check": "health_check.py",
    "setup_wizard": "setup_wizard.py",
}

# Integration class names
INTEGRATION_CLASSES = {
    "pack_orchestrator": "E1PackOrchestrator",
    "ghg_app_bridge": "GHGAppBridge",
    "mrv_agent_bridge": "MRVAgentBridge",
    "dma_pack_bridge": "DMAPackBridge",
    "decarbonization_bridge": "DecarbonizationBridge",
    "adaptation_bridge": "AdaptationBridge",
    "health_check": "E1HealthCheck",
    "setup_wizard": "E1SetupWizard",
}

# Preset names
PRESET_NAMES = [
    "power_generation",
    "manufacturing",
    "transport",
    "financial_services",
    "real_estate",
    "multi_sector",
]


# =============================================================================
# Helper: Dynamic Module Loader
# =============================================================================


def _load_module(module_name: str, file_name: str, subdir: str = "engines"):
    """Load a module dynamically using importlib.util.spec_from_file_location.

    This avoids the need to install PACK-016 as a Python package. The module
    is loaded from the pack source tree and added to sys.modules under a
    unique key to prevent collisions.

    Args:
        module_name: Logical name for the module (used as sys.modules key prefix).
        file_name: File name of the Python module (e.g., "ghg_inventory_engine.py").
        subdir: Subdirectory under PACK_ROOT ("engines", "workflows", "templates",
                "integrations", or "config").

    Returns:
        The loaded module object.

    Raises:
        FileNotFoundError: If the module file does not exist.
        ImportError: If the module cannot be loaded.
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
        raise FileNotFoundError(
            f"Module file not found: {file_path}. "
            f"Ensure PACK-016 source files are present."
        )

    # Create a unique module key to avoid collisions
    full_module_name = f"pack016_test.{subdir}.{module_name}"

    # Return cached module if already loaded
    if full_module_name in sys.modules:
        return sys.modules[full_module_name]

    spec = importlib.util.spec_from_file_location(full_module_name, file_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot create module spec for {file_path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[full_module_name] = module

    try:
        spec.loader.exec_module(module)
    except Exception as exc:
        # Remove from sys.modules on failure to allow retry
        sys.modules.pop(full_module_name, None)
        raise ImportError(
            f"Failed to load module {full_module_name} from {file_path}: {exc}"
        ) from exc

    return module


def _load_engine(engine_key: str):
    """Load an engine module by its logical key."""
    file_name = ENGINE_FILES[engine_key]
    return _load_module(engine_key, file_name, "engines")


def _load_workflow(workflow_key: str):
    """Load a workflow module by its logical key."""
    file_name = WORKFLOW_FILES[workflow_key]
    return _load_module(workflow_key, file_name, "workflows")


def _load_template(template_key: str):
    """Load a template module by its logical key."""
    file_name = TEMPLATE_FILES[template_key]
    return _load_module(template_key, file_name, "templates")


def _load_integration(integration_key: str):
    """Load an integration module by its logical key."""
    file_name = INTEGRATION_FILES[integration_key]
    return _load_module(integration_key, file_name, "integrations")


def _load_config_module():
    """Load the pack_config module."""
    return _load_module("pack_config", "pack_config.py", "config")


# =============================================================================
# 1. Path and YAML Fixtures
# =============================================================================


@pytest.fixture(scope="session")
def pack_root() -> Path:
    """Return the absolute path to the PACK-016 root directory."""
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
    with open(demo_yaml_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    assert data is not None, "demo_config.yaml parsed to None"
    return data


# =============================================================================
# 2. Configuration Fixtures
# =============================================================================


@pytest.fixture(scope="session")
def config_module():
    """Load and return the pack_config module."""
    return _load_config_module()


@pytest.fixture
def e1_config(config_module):
    """Create an E1ClimateConfig with default values."""
    return config_module.E1ClimateConfig()


@pytest.fixture
def pack_config(config_module):
    """Create a PackConfig wrapper with default values."""
    return config_module.PackConfig()


@pytest.fixture
def demo_config(config_module, demo_yaml_data):
    """Create an E1ClimateConfig loaded from the demo YAML data."""
    return config_module.E1ClimateConfig(**demo_yaml_data)


@pytest.fixture
def manufacturing_config(config_module):
    """Load the manufacturing preset as a PackConfig."""
    return config_module.PackConfig.from_preset("manufacturing")


@pytest.fixture
def power_generation_config(config_module):
    """Load the power_generation preset as a PackConfig."""
    return config_module.PackConfig.from_preset("power_generation")
