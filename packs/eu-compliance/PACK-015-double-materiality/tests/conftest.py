# -*- coding: utf-8 -*-
"""
PACK-015 Double Materiality Assessment Pack - Shared Test Fixtures (conftest.py)
================================================================================

Provides pytest fixtures for the PACK-015 test suite including:
  - Dynamic module loading via importlib (no package install needed)
  - Pack manifest and configuration fixtures
  - DMA-specific sample data (sustainability matters, IROs, scores, stakeholders)
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
  7. DMA-specific sample data

Author:  GreenLang Platform Team (GL-TestEngineer)
Pack:    PACK-015 Double Materiality Assessment
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
    "impact_materiality": "impact_materiality_engine.py",
    "financial_materiality": "financial_materiality_engine.py",
    "stakeholder_engagement": "stakeholder_engagement_engine.py",
    "iro_identification": "iro_identification_engine.py",
    "materiality_matrix": "materiality_matrix_engine.py",
    "esrs_topic_mapping": "esrs_topic_mapping_engine.py",
    "threshold_scoring": "threshold_scoring_engine.py",
    "dma_report": "dma_report_engine.py",
}

# Engine class names that should exist in each engine module
ENGINE_CLASSES = {
    "impact_materiality": "ImpactMaterialityEngine",
    "financial_materiality": "FinancialMaterialityEngine",
    "stakeholder_engagement": "StakeholderEngagementEngine",
    "iro_identification": "IROIdentificationEngine",
    "materiality_matrix": "MaterialityMatrixEngine",
    "esrs_topic_mapping": "ESRSTopicMappingEngine",
    "threshold_scoring": "ThresholdScoringEngine",
    "dma_report": "DMAReportEngine",
}

# Workflow file mapping
WORKFLOW_FILES = {
    "impact_assessment": "impact_assessment_workflow.py",
    "financial_assessment": "financial_assessment_workflow.py",
    "stakeholder_engagement": "stakeholder_engagement_workflow.py",
    "iro_identification": "iro_identification_workflow.py",
    "materiality_matrix": "materiality_matrix_workflow.py",
    "esrs_mapping": "esrs_mapping_workflow.py",
    "full_dma": "full_dma_workflow.py",
    "dma_update": "dma_update_workflow.py",
}

# Workflow class names
WORKFLOW_CLASSES = {
    "impact_assessment": "ImpactAssessmentWorkflow",
    "financial_assessment": "FinancialAssessmentWorkflow",
    "stakeholder_engagement": "StakeholderEngagementWorkflow",
    "iro_identification": "IROIdentificationWorkflow",
    "materiality_matrix": "MaterialityMatrixWorkflow",
    "esrs_mapping": "ESRSMappingWorkflow",
    "full_dma": "FullDMAWorkflow",
    "dma_update": "DMAUpdateWorkflow",
}

# Template file mapping
TEMPLATE_FILES = {
    "impact_materiality_report": "impact_materiality_report.py",
    "financial_materiality_report": "financial_materiality_report.py",
    "stakeholder_engagement_report": "stakeholder_engagement_report.py",
    "materiality_matrix_report": "materiality_matrix_report.py",
    "iro_register_report": "iro_register_report.py",
    "esrs_disclosure_map": "esrs_disclosure_map.py",
    "dma_executive_summary": "dma_executive_summary.py",
    "dma_audit_report": "dma_audit_report.py",
}

# Template class names
TEMPLATE_CLASSES = {
    "impact_materiality_report": "ImpactMaterialityReportTemplate",
    "financial_materiality_report": "FinancialMaterialityReportTemplate",
    "stakeholder_engagement_report": "StakeholderEngagementReportTemplate",
    "materiality_matrix_report": "MaterialityMatrixReportTemplate",
    "iro_register_report": "IRORegisterReportTemplate",
    "esrs_disclosure_map": "ESRSDisclosureMapTemplate",
    "dma_executive_summary": "DMAExecutiveSummaryTemplate",
    "dma_audit_report": "DMAAuditReportTemplate",
}

# Integration file mapping
INTEGRATION_FILES = {
    "pack_orchestrator": "pack_orchestrator.py",
    "csrd_pack_bridge": "csrd_pack_bridge.py",
    "mrv_materiality_bridge": "mrv_materiality_bridge.py",
    "data_materiality_bridge": "data_materiality_bridge.py",
    "sector_classification_bridge": "sector_classification_bridge.py",
    "regulatory_bridge": "regulatory_bridge.py",
    "health_check": "health_check.py",
    "setup_wizard": "setup_wizard.py",
}

# Integration class names
INTEGRATION_CLASSES = {
    "pack_orchestrator": "DMAPackOrchestrator",
    "csrd_pack_bridge": "CSRDPackBridge",
    "mrv_materiality_bridge": "MRVMaterialityBridge",
    "data_materiality_bridge": "DataMaterialityBridge",
    "sector_classification_bridge": "SectorClassificationBridge",
    "regulatory_bridge": "RegulatoryBridge",
    "health_check": "DMAHealthCheck",
    "setup_wizard": "DMASetupWizard",
}

# Preset names
PRESET_NAMES = [
    "large_enterprise",
    "mid_market",
    "sme",
    "financial_services",
    "manufacturing",
    "multi_sector",
]


# =============================================================================
# Helper: Dynamic Module Loader
# =============================================================================


def _load_module(module_name: str, file_name: str, subdir: str = "engines"):
    """Load a module dynamically using importlib.util.spec_from_file_location.

    This avoids the need to install PACK-015 as a Python package. The module
    is loaded from the pack source tree and added to sys.modules under a
    unique key to prevent collisions.

    Args:
        module_name: Logical name for the module (used as sys.modules key prefix).
        file_name: File name of the Python module (e.g., "impact_materiality_engine.py").
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
            f"Ensure PACK-015 source files are present."
        )

    # Create a unique module key to avoid collisions
    full_module_name = f"pack015_test.{subdir}.{module_name}"

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
    """Return the absolute path to the PACK-015 root directory."""
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
def dma_config(config_module):
    """Create a DMAConfig with default values."""
    return config_module.DMAConfig()


@pytest.fixture
def pack_config(config_module):
    """Create a PackConfig wrapper with default values."""
    return config_module.PackConfig()


@pytest.fixture
def demo_config(config_module, demo_yaml_data):
    """Create a DMAConfig loaded from the demo YAML data."""
    return config_module.DMAConfig(**demo_yaml_data)


@pytest.fixture
def large_enterprise_config(config_module):
    """Load the large_enterprise preset as a PackConfig."""
    return config_module.PackConfig.from_preset("large_enterprise")


@pytest.fixture
def sme_config(config_module):
    """Load the sme preset as a PackConfig."""
    return config_module.PackConfig.from_preset("sme")
