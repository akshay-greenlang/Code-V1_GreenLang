# -*- coding: utf-8 -*-
"""
PACK-017 ESRS Full Coverage Pack - Shared Test Fixtures (conftest.py)
======================================================================

Provides pytest fixtures for the PACK-017 test suite including:
  - Dynamic module loading via importlib (no package install needed)
  - Pack manifest and configuration fixtures
  - Sample data fixtures for all 12 ESRS topical/cross-cutting standards
  - Demo configuration loading

All fixtures use importlib.util.spec_from_file_location to load modules
directly from the pack source tree, enabling test execution without
installing the pack as a Python package.

Fixture Categories:
  1. Paths and YAML data
  2. Configuration objects
  3. Engine module fixtures
  4. Workflow module fixtures
  5. Template module fixtures
  6. Integration module fixtures
  7. ESRS-specific sample data (E2, E3, E4, E5, S1, S2, S3, S4, G1, ESRS2)

Engine Coverage (11 engines):
  - ESRS2 General Disclosures Engine
  - E2 Pollution Engine
  - E3 Water & Marine Resources Engine
  - E4 Biodiversity & Ecosystems Engine
  - E5 Circular Economy Engine
  - S1 Own Workforce Engine
  - S2 Value Chain Workers Engine
  - S3 Affected Communities Engine
  - S4 Consumers & End-Users Engine
  - G1 Business Conduct Engine
  - ESRS Coverage Orchestrator Engine

Author:  GreenLang Platform Team (GL-TestEngineer)
Pack:    PACK-017 ESRS Full Coverage
Date:    March 2026
"""

import importlib
import importlib.util
import sys
from decimal import Decimal
from pathlib import Path
from typing import Any, Dict, List

import pytest

try:
    import yaml
except ImportError:  # pragma: no cover
    yaml = None  # type: ignore[assignment]


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

# ---------------------------------------------------------------------------
# Engine file mapping: logical name -> file name on disk (11 engines)
# ---------------------------------------------------------------------------
ENGINE_FILES = {
    "e2_pollution": "e2_pollution_engine.py",
    "e3_water_marine": "e3_water_marine_engine.py",
    "e4_biodiversity": "e4_biodiversity_engine.py",
    "e5_circular_economy": "e5_circular_economy_engine.py",
    "s1_own_workforce": "s1_own_workforce_engine.py",
    "s2_value_chain_workers": "s2_value_chain_workers_engine.py",
    "s3_affected_communities": "s3_affected_communities_engine.py",
    "s4_consumers": "s4_consumers_engine.py",
    "g1_business_conduct": "g1_business_conduct_engine.py",
    "esrs2_general_disclosures": "esrs2_general_disclosures_engine.py",
    "esrs_coverage_orchestrator": "esrs_coverage_orchestrator_engine.py",
}

# Engine class names expected in each module
ENGINE_CLASSES = {
    "e2_pollution": "PollutionEngine",
    "e3_water_marine": "WaterMarineEngine",
    "e4_biodiversity": "BiodiversityEngine",
    "e5_circular_economy": "CircularEconomyEngine",
    "s1_own_workforce": "OwnWorkforceEngine",
    "s2_value_chain_workers": "ValueChainWorkersEngine",
    "s3_affected_communities": "AffectedCommunitiesEngine",
    "s4_consumers": "ConsumersEngine",
    "g1_business_conduct": "BusinessConductEngine",
    "esrs2_general_disclosures": "GeneralDisclosuresEngine",
    "esrs_coverage_orchestrator": "ESRSCoverageOrchestratorEngine",
}

# ---------------------------------------------------------------------------
# Workflow file mapping (12 workflows)
# ---------------------------------------------------------------------------
WORKFLOW_FILES = {
    "esrs2_general": "esrs2_general_workflow.py",
    "esrs2_governance": "esrs2_governance_workflow.py",
    "e2_pollution": "e2_pollution_workflow.py",
    "e3_water": "e3_water_workflow.py",
    "e4_biodiversity": "e4_biodiversity_workflow.py",
    "e5_circular_economy": "e5_circular_economy_workflow.py",
    "e5_circular": "e5_circular_economy_workflow.py",
    "s1_workforce": "s1_workforce_workflow.py",
    "s2_value_chain": "s2_value_chain_workflow.py",
    "s3_communities": "s3_communities_workflow.py",
    "s4_consumers": "s4_consumers_workflow.py",
    "g1_governance": "g1_governance_workflow.py",
    "full_esrs": "full_esrs_workflow.py",
    "full_coverage": "full_esrs_workflow.py",
}

WORKFLOW_CLASSES = {
    "esrs2_general": "ESRS2GeneralWorkflow",
    "esrs2_governance": "ESRS2GovernanceWorkflow",
    "e2_pollution": "E2PollutionWorkflow",
    "e3_water": "E3WaterWorkflow",
    "e4_biodiversity": "E4BiodiversityWorkflow",
    "e5_circular_economy": "E5CircularWorkflow",
    "e5_circular": "E5CircularWorkflow",
    "s1_workforce": "S1WorkforceWorkflow",
    "s2_value_chain": "S2ValueChainWorkflow",
    "s3_communities": "S3CommunitiesWorkflow",
    "s4_consumers": "S4ConsumersWorkflow",
    "g1_governance": "G1GovernanceWorkflow",
    "full_esrs": "FullESRSWorkflow",
    "full_coverage": "FullESRSWorkflow",
}

# ---------------------------------------------------------------------------
# Template file mapping (12 templates)
# ---------------------------------------------------------------------------
TEMPLATE_FILES = {
    "esrs2_general_report": "esrs2_general_report.py",
    "esrs2_general": "esrs2_general_report.py",
    "e2_pollution_report": "e2_pollution_report.py",
    "e2_pollution": "e2_pollution_report.py",
    "e3_water_report": "e3_water_report.py",
    "e3_water": "e3_water_report.py",
    "e4_biodiversity_report": "e4_biodiversity_report.py",
    "e5_circular_economy_report": "e5_circular_economy_report.py",
    "s1_workforce_report": "s1_workforce_report.py",
    "s2_value_chain_report": "s2_value_chain_report.py",
    "s3_communities_report": "s3_communities_report.py",
    "s4_consumers_report": "s4_consumers_report.py",
    "g1_governance_report": "g1_governance_report.py",
    "esrs_scorecard_report": "esrs_scorecard_report.py",
    "esrs_annual_statement": "esrs_annual_statement.py",
}

TEMPLATE_CLASSES = {
    "esrs2_general_report": "ESRS2GeneralReportTemplate",
    "esrs2_general": "ESRS2GeneralReportTemplate",
    "e2_pollution_report": "E2PollutionReportTemplate",
    "e2_pollution": "E2PollutionReportTemplate",
    "e3_water_report": "E3WaterReportTemplate",
    "e3_water": "E3WaterReportTemplate",
    "e4_biodiversity_report": "E4BiodiversityReportTemplate",
    "e5_circular_economy_report": "E5CircularReport",
    "s1_workforce_report": "S1WorkforceReportTemplate",
    "s2_value_chain_report": "S2ValueChainReportTemplate",
    "s3_communities_report": "S3CommunitiesReportTemplate",
    "s4_consumers_report": "S4ConsumersReportTemplate",
    "g1_governance_report": "G1GovernanceReportTemplate",
    "esrs_scorecard_report": "ESRSScorecard",
    "esrs_annual_statement": "ESRSAnnualStatementTemplate",
}

# ---------------------------------------------------------------------------
# Integration file mapping (10 integrations)
# ---------------------------------------------------------------------------
INTEGRATION_FILES = {
    "pack_orchestrator": "pack_orchestrator.py",
    "e1_pack_bridge": "e1_pack_bridge.py",
    "e1_bridge": "e1_pack_bridge.py",  # Alias
    "dma_pack_bridge": "dma_pack_bridge.py",
    "dma_bridge": "dma_pack_bridge.py",  # Alias
    "csrd_app_bridge": "csrd_app_bridge.py",
    "mrv_agent_bridge": "mrv_agent_bridge.py",
    "mrv_bridge": "mrv_agent_bridge.py",  # Alias
    "data_agent_bridge": "data_agent_bridge.py",
    "data_bridge": "data_agent_bridge.py",  # Alias
    "taxonomy_bridge": "taxonomy_bridge.py",
    "xbrl_tagging_bridge": "xbrl_tagging_bridge.py",
    "xbrl_mapper": "xbrl_tagging_bridge.py",  # Alias
    "health_check": "health_check.py",
    "audit_bridge": "health_check.py",
    "setup_wizard": "setup_wizard.py",
}

INTEGRATION_CLASSES = {
    "pack_orchestrator": "ESRSFullOrchestrator",
    "e1_pack_bridge": "E1PackBridge",
    "e1_bridge": "E1PackBridge",  # Alias
    "dma_pack_bridge": "DMAPackBridge",
    "dma_bridge": "DMAPackBridge",  # Alias
    "csrd_app_bridge": "CSRDAppBridge",
    "mrv_agent_bridge": "MRVAgentBridge",
    "mrv_bridge": "MRVAgentBridge",  # Alias
    "data_agent_bridge": "DataAgentBridge",
    "data_bridge": "DataAgentBridge",  # Alias
    "taxonomy_bridge": "TaxonomyBridge",
    "xbrl_tagging_bridge": "XBRLTaggingBridge",
    "xbrl_mapper": "XBRLTaggingBridge",  # Alias
    "health_check": "ESRSHealthCheck",
    "audit_bridge": "ESRSHealthCheck",
    "setup_wizard": "PackSetupWizard",
}

# Preset names
PRESET_NAMES = [
    "manufacturing",
    "financial_services",
    "energy",
    "retail",
    "technology",
    "multi_sector",
]

# All 12 ESRS standards
ALL_ESRS_STANDARDS = [
    "ESRS_1", "ESRS_2", "E1", "E2", "E3", "E4", "E5",
    "S1", "S2", "S3", "S4", "G1",
]


# =============================================================================
# Helper: Dynamic Module Loader
# =============================================================================


def _load_module(module_name: str, file_name: str, subdir: str = "engines"):
    """Load a module dynamically using importlib.util.spec_from_file_location.

    This avoids the need to install PACK-017 as a Python package.  The module
    is loaded from the pack source tree and added to sys.modules under a
    unique key to prevent collisions.

    Args:
        module_name: Logical name for the module (used as sys.modules key prefix).
        file_name: File name of the Python module (e.g., "e2_pollution_engine.py").
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
            f"Ensure PACK-017 source files are present."
        )

    # Create a unique module key to avoid collisions
    full_module_name = f"pack017_test.{subdir}.{module_name}"

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
    """Return the absolute path to the PACK-017 root directory."""
    return PACK_ROOT


@pytest.fixture(scope="session")
def pack_yaml_path() -> Path:
    """Return the absolute path to pack.yaml."""
    return PACK_ROOT / "pack.yaml"


@pytest.fixture(scope="session")
def pack_yaml_data(pack_yaml_path: Path) -> Dict[str, Any]:
    """Parse and return the pack.yaml manifest as a dictionary."""
    if yaml is None:
        pytest.skip("pyyaml not installed")
    if not pack_yaml_path.exists():
        pytest.skip("pack.yaml not found")
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
    if yaml is None:
        pytest.skip("pyyaml not installed")
    if not demo_yaml_path.exists():
        pytest.skip(f"demo_config.yaml not found at {demo_yaml_path}")
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
def esrs_config(config_module):
    """Create an ESRSFullCoverageConfig with default values."""
    return config_module.ESRSFullCoverageConfig()


@pytest.fixture
def pack_config(config_module):
    """Create a PackConfig wrapper with default values."""
    return config_module.PackConfig()


@pytest.fixture
def demo_config(config_module, demo_yaml_data):
    """Create an ESRSFullCoverageConfig loaded from the demo YAML data."""
    return config_module.ESRSFullCoverageConfig(**demo_yaml_data)


@pytest.fixture
def manufacturing_config(config_module):
    """Load the manufacturing preset as a PackConfig."""
    return config_module.PackConfig.from_preset("manufacturing")


@pytest.fixture
def financial_services_config(config_module):
    """Load the financial_services preset as a PackConfig."""
    return config_module.PackConfig.from_preset("financial_services")


@pytest.fixture
def energy_config(config_module):
    """Load the energy preset as a PackConfig."""
    return config_module.PackConfig.from_preset("energy")


# =============================================================================
# 3. Engine Module Fixtures
# =============================================================================


@pytest.fixture(scope="session")
def e2_pollution_module():
    """Load the E2 Pollution engine module."""
    return _load_engine("e2_pollution")


@pytest.fixture(scope="session")
def e3_water_marine_module():
    """Load the E3 Water & Marine Resources engine module."""
    return _load_engine("e3_water_marine")


@pytest.fixture(scope="session")
def e4_biodiversity_module():
    """Load the E4 Biodiversity engine module."""
    return _load_engine("e4_biodiversity")


@pytest.fixture(scope="session")
def e5_circular_economy_module():
    """Load the E5 Circular Economy engine module."""
    return _load_engine("e5_circular_economy")


@pytest.fixture(scope="session")
def s1_own_workforce_module():
    """Load the S1 Own Workforce engine module."""
    return _load_engine("s1_own_workforce")


@pytest.fixture(scope="session")
def s2_value_chain_workers_module():
    """Load the S2 Value Chain Workers engine module."""
    return _load_engine("s2_value_chain_workers")


@pytest.fixture(scope="session")
def s3_affected_communities_module():
    """Load the S3 Affected Communities engine module."""
    return _load_engine("s3_affected_communities")


@pytest.fixture(scope="session")
def s4_consumers_module():
    """Load the S4 Consumers & End-Users engine module."""
    return _load_engine("s4_consumers")


@pytest.fixture(scope="session")
def g1_business_conduct_module():
    """Load the G1 Business Conduct engine module."""
    return _load_engine("g1_business_conduct")


@pytest.fixture(scope="session")
def esrs2_general_disclosures_module():
    """Load the ESRS2 General Disclosures engine module."""
    return _load_engine("esrs2_general_disclosures")


# =============================================================================
# 4. Workflow Module Fixtures
# =============================================================================


@pytest.fixture(scope="session")
def esrs2_general_workflow_module():
    """Load the ESRS2 General workflow module."""
    return _load_workflow("esrs2_general")


@pytest.fixture(scope="session")
def esrs2_governance_workflow_module():
    """Load the ESRS2 Governance workflow module."""
    return _load_workflow("esrs2_governance")


# =============================================================================
# 5. Template Module Fixtures
# =============================================================================


@pytest.fixture(scope="session")
def esrs2_general_report_module():
    """Load the ESRS2 General report template module."""
    return _load_template("esrs2_general_report")


@pytest.fixture(scope="session")
def e2_pollution_report_module():
    """Load the E2 Pollution report template module."""
    return _load_template("e2_pollution_report")


# =============================================================================
# 6. Integration Module Fixtures
# =============================================================================

# Integration fixtures are loaded on demand via _load_integration() helper.
# Usage: module = _load_integration("pack_orchestrator")


# =============================================================================
# 7. ESRS-Specific Sample Data Fixtures
# =============================================================================


# ---------------------------------------------------------------------------
# 7.1  E2 Pollution Sample Data
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_air_emissions() -> List[Dict[str, Any]]:
    """Sample air pollutant emissions for E2-4 testing.

    Includes NOx, SOx, PM, VOCs, and heavy metals as typical industrial
    emissions to air per ESRS E2-4 para 28.
    """
    return [
        {
            "pollutant": "NOx",
            "cas_number": "10102-44-0",
            "medium": "AIR",
            "quantity_tonnes": Decimal("125.50"),
            "measurement_method": "CEMS",
            "facility": "Plant A - Boiler Stack",
            "reporting_year": 2025,
        },
        {
            "pollutant": "SOx",
            "cas_number": "7446-09-5",
            "medium": "AIR",
            "quantity_tonnes": Decimal("42.30"),
            "measurement_method": "CEMS",
            "facility": "Plant A - Boiler Stack",
            "reporting_year": 2025,
        },
        {
            "pollutant": "PM10",
            "cas_number": None,
            "medium": "AIR",
            "quantity_tonnes": Decimal("18.75"),
            "measurement_method": "ISOKINETIC",
            "facility": "Plant B - Kiln",
            "reporting_year": 2025,
        },
        {
            "pollutant": "VOC",
            "cas_number": None,
            "medium": "AIR",
            "quantity_tonnes": Decimal("67.20"),
            "measurement_method": "FID",
            "facility": "Paint Shop",
            "reporting_year": 2025,
        },
        {
            "pollutant": "Mercury",
            "cas_number": "7439-97-6",
            "medium": "AIR",
            "quantity_tonnes": Decimal("0.012"),
            "measurement_method": "LABORATORY",
            "facility": "Plant A - Stack",
            "reporting_year": 2025,
        },
    ]


@pytest.fixture
def sample_water_discharges_e2() -> List[Dict[str, Any]]:
    """Sample water discharge pollutants for E2-4 testing.

    Includes COD, BOD, heavy metals, and nutrients typical of industrial
    effluent per ESRS E2-4 para 29.
    """
    return [
        {
            "pollutant": "COD",
            "medium": "WATER",
            "quantity_tonnes": Decimal("8.50"),
            "receiving_body": "River Main",
            "treatment_method": "BIOLOGICAL",
            "facility": "WWTP-01",
            "reporting_year": 2025,
        },
        {
            "pollutant": "BOD5",
            "medium": "WATER",
            "quantity_tonnes": Decimal("3.20"),
            "receiving_body": "River Main",
            "treatment_method": "BIOLOGICAL",
            "facility": "WWTP-01",
            "reporting_year": 2025,
        },
        {
            "pollutant": "Total Nitrogen",
            "medium": "WATER",
            "quantity_tonnes": Decimal("1.85"),
            "receiving_body": "River Main",
            "treatment_method": "BIOLOGICAL",
            "facility": "WWTP-01",
            "reporting_year": 2025,
        },
        {
            "pollutant": "Zinc",
            "cas_number": "7440-66-6",
            "medium": "WATER",
            "quantity_tonnes": Decimal("0.045"),
            "receiving_body": "River Main",
            "treatment_method": "CHEMICAL_PRECIPITATION",
            "facility": "WWTP-01",
            "reporting_year": 2025,
        },
    ]


@pytest.fixture
def sample_substances_of_concern() -> List[Dict[str, Any]]:
    """Sample substances of concern and SVHC for E2-5 testing.

    Per ESRS E2-5 para 36-40, substances on the REACH Candidate List
    and REACH Annex XIV are classified as substances of very high concern.
    """
    return [
        {
            "substance_name": "Lead",
            "cas_number": "7439-92-1",
            "is_svhc": True,
            "reach_status": "ANNEX_XIV",
            "quantity_tonnes": Decimal("0.85"),
            "use_in_products": True,
            "substitution_plan": True,
        },
        {
            "substance_name": "Chromium VI",
            "cas_number": "7440-47-3",
            "is_svhc": True,
            "reach_status": "CANDIDATE_LIST",
            "quantity_tonnes": Decimal("0.12"),
            "use_in_products": True,
            "substitution_plan": False,
        },
        {
            "substance_name": "Toluene",
            "cas_number": "108-88-3",
            "is_svhc": False,
            "reach_status": "REGISTERED",
            "quantity_tonnes": Decimal("45.00"),
            "use_in_products": True,
            "substitution_plan": False,
        },
    ]


# ---------------------------------------------------------------------------
# 7.2  E3 Water & Marine Resources Sample Data
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_water_withdrawals() -> List[Dict[str, Any]]:
    """Sample water withdrawal data for E3-4 testing.

    Per ESRS E3-4 para 24-30, water consumption must be disaggregated
    by source, stress level, and purpose.
    """
    return [
        {
            "source": "SURFACE_WATER",
            "volume_megaliters": Decimal("450.00"),
            "water_stress_area": False,
            "facility": "Plant A",
            "purpose": "COOLING",
            "reporting_year": 2025,
        },
        {
            "source": "GROUNDWATER",
            "volume_megaliters": Decimal("120.00"),
            "water_stress_area": True,
            "facility": "Plant B",
            "purpose": "PROCESS",
            "reporting_year": 2025,
        },
        {
            "source": "MUNICIPAL_SUPPLY",
            "volume_megaliters": Decimal("85.00"),
            "water_stress_area": False,
            "facility": "HQ Office",
            "purpose": "SANITARY",
            "reporting_year": 2025,
        },
        {
            "source": "RAINWATER",
            "volume_megaliters": Decimal("15.00"),
            "water_stress_area": False,
            "facility": "Plant A",
            "purpose": "LANDSCAPE",
            "reporting_year": 2025,
        },
    ]


@pytest.fixture
def sample_water_discharges_e3() -> List[Dict[str, Any]]:
    """Sample water discharge data for E3 water balance testing.

    Discharges are disaggregated by destination and treatment level
    per ESRS E3-4.
    """
    return [
        {
            "destination": "SURFACE_WATER",
            "volume_megaliters": Decimal("380.00"),
            "treatment_level": "SECONDARY",
            "facility": "Plant A",
            "meets_quality_standards": True,
            "reporting_year": 2025,
        },
        {
            "destination": "MUNICIPAL_SEWER",
            "volume_megaliters": Decimal("95.00"),
            "treatment_level": "PRIMARY",
            "facility": "Plant B",
            "meets_quality_standards": True,
            "reporting_year": 2025,
        },
        {
            "destination": "OCEAN",
            "volume_megaliters": Decimal("50.00"),
            "treatment_level": "TERTIARY",
            "facility": "Coastal Plant",
            "meets_quality_standards": True,
            "reporting_year": 2025,
        },
    ]


# ---------------------------------------------------------------------------
# 7.3  E4 Biodiversity & Ecosystems Sample Data
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_biodiversity_sites() -> List[Dict[str, Any]]:
    """Sample biodiversity-sensitive site data for E4-5 testing.

    Per ESRS E4-5 para 36-46, undertakings must disclose sites in or
    near biodiversity-sensitive areas and their impact metrics.
    """
    return [
        {
            "site_name": "Factory Rhine",
            "location": {"latitude": 50.1109, "longitude": 8.6821},
            "near_protected_area": True,
            "protected_area_name": "Natura 2000 - DE6017302",
            "distance_km": Decimal("0.8"),
            "sensitivity": "HIGH",
            "land_use_type": "INDUSTRIAL",
            "area_hectares": Decimal("12.5"),
            "mitigation_measures": True,
        },
        {
            "site_name": "Warehouse South",
            "location": {"latitude": 48.1351, "longitude": 11.5820},
            "near_protected_area": False,
            "protected_area_name": None,
            "distance_km": None,
            "sensitivity": "LOW",
            "land_use_type": "COMMERCIAL",
            "area_hectares": Decimal("3.2"),
            "mitigation_measures": False,
        },
        {
            "site_name": "Mine Extraction Alpha",
            "location": {"latitude": 51.3397, "longitude": 12.3731},
            "near_protected_area": True,
            "protected_area_name": "KBA - Leipzig Riparian",
            "distance_km": Decimal("0.2"),
            "sensitivity": "CRITICAL",
            "land_use_type": "EXTRACTIVE",
            "area_hectares": Decimal("85.0"),
            "mitigation_measures": True,
        },
    ]


@pytest.fixture
def sample_land_use_changes() -> List[Dict[str, Any]]:
    """Sample land use change data for E4 impact assessment.

    Tracks conversion of natural habitats per ESRS E4-5.
    """
    return [
        {
            "change_type": "CONVERSION",
            "from_land_use": "AGRICULTURAL",
            "to_land_use": "INDUSTRIAL",
            "area_hectares": Decimal("5.0"),
            "year": 2024,
            "biodiversity_offset": False,
        },
        {
            "change_type": "RESTORATION",
            "from_land_use": "DEGRADED",
            "to_land_use": "WETLAND",
            "area_hectares": Decimal("2.0"),
            "year": 2025,
            "biodiversity_offset": True,
        },
    ]


@pytest.fixture
def sample_species_data() -> List[Dict[str, Any]]:
    """Sample species impact data for E4 disclosures.

    Includes IUCN Red List status and site-level species counts.
    """
    return [
        {
            "species_name": "European Otter",
            "iucn_status": "NEAR_THREATENED",
            "site": "Factory Rhine",
            "population_trend": "STABLE",
            "management_action": True,
        },
        {
            "species_name": "Common Kingfisher",
            "iucn_status": "LEAST_CONCERN",
            "site": "Factory Rhine",
            "population_trend": "INCREASING",
            "management_action": False,
        },
        {
            "species_name": "Sand Lizard",
            "iucn_status": "LEAST_CONCERN",
            "site": "Mine Extraction Alpha",
            "population_trend": "DECLINING",
            "management_action": True,
        },
    ]


# ---------------------------------------------------------------------------
# 7.4  E5 Circular Economy Sample Data
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_material_inflows() -> List[Dict[str, Any]]:
    """Sample material inflow data for E5-4 testing.

    Per ESRS E5-4 para 34-38, undertakings must disclose resource inflows
    including virgin vs. recycled/secondary material proportions.
    """
    return [
        {
            "material": "Steel",
            "total_tonnes": Decimal("5000.00"),
            "virgin_tonnes": Decimal("3500.00"),
            "recycled_tonnes": Decimal("1500.00"),
            "recycled_percentage": Decimal("30.0"),
            "certified_sustainable": False,
            "reporting_year": 2025,
        },
        {
            "material": "Plastics",
            "total_tonnes": Decimal("1200.00"),
            "virgin_tonnes": Decimal("960.00"),
            "recycled_tonnes": Decimal("240.00"),
            "recycled_percentage": Decimal("20.0"),
            "certified_sustainable": False,
            "reporting_year": 2025,
        },
        {
            "material": "Aluminium",
            "total_tonnes": Decimal("800.00"),
            "virgin_tonnes": Decimal("320.00"),
            "recycled_tonnes": Decimal("480.00"),
            "recycled_percentage": Decimal("60.0"),
            "certified_sustainable": True,
            "reporting_year": 2025,
        },
        {
            "material": "Wood",
            "total_tonnes": Decimal("350.00"),
            "virgin_tonnes": Decimal("175.00"),
            "recycled_tonnes": Decimal("175.00"),
            "recycled_percentage": Decimal("50.0"),
            "certified_sustainable": True,
            "reporting_year": 2025,
        },
    ]


@pytest.fixture
def sample_waste_outflows() -> List[Dict[str, Any]]:
    """Sample waste outflow data for E5-5 testing.

    Per ESRS E5-5 para 39-44, waste generation must be reported by
    treatment method (recovery, recycling, incineration, landfill).
    """
    return [
        {
            "waste_type": "NON_HAZARDOUS",
            "treatment": "RECYCLED",
            "quantity_tonnes": Decimal("1800.00"),
            "percentage_of_total": Decimal("45.0"),
            "reporting_year": 2025,
        },
        {
            "waste_type": "NON_HAZARDOUS",
            "treatment": "INCINERATION_WITH_ENERGY_RECOVERY",
            "quantity_tonnes": Decimal("800.00"),
            "percentage_of_total": Decimal("20.0"),
            "reporting_year": 2025,
        },
        {
            "waste_type": "NON_HAZARDOUS",
            "treatment": "LANDFILL",
            "quantity_tonnes": Decimal("600.00"),
            "percentage_of_total": Decimal("15.0"),
            "reporting_year": 2025,
        },
        {
            "waste_type": "HAZARDOUS",
            "treatment": "INCINERATION_WITHOUT_ENERGY_RECOVERY",
            "quantity_tonnes": Decimal("200.00"),
            "percentage_of_total": Decimal("5.0"),
            "reporting_year": 2025,
        },
        {
            "waste_type": "HAZARDOUS",
            "treatment": "RECYCLED",
            "quantity_tonnes": Decimal("120.00"),
            "percentage_of_total": Decimal("3.0"),
            "reporting_year": 2025,
        },
        {
            "waste_type": "NON_HAZARDOUS",
            "treatment": "COMPOSTED",
            "quantity_tonnes": Decimal("480.00"),
            "percentage_of_total": Decimal("12.0"),
            "reporting_year": 2025,
        },
    ]


# ---------------------------------------------------------------------------
# 7.5  S1 Own Workforce Sample Data
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_workforce_employees() -> List[Dict[str, Any]]:
    """Sample employee headcount data for S1-6 testing.

    Per ESRS S1-6 para 50, workforce must be disaggregated by gender,
    employment type, and contract type.
    """
    return [
        {
            "gender": "MALE",
            "contract_type": "PERMANENT",
            "employment_type": "FULL_TIME",
            "headcount": 2500,
            "region": "EU",
            "reporting_year": 2025,
        },
        {
            "gender": "FEMALE",
            "contract_type": "PERMANENT",
            "employment_type": "FULL_TIME",
            "headcount": 1800,
            "region": "EU",
            "reporting_year": 2025,
        },
        {
            "gender": "MALE",
            "contract_type": "TEMPORARY",
            "employment_type": "FULL_TIME",
            "headcount": 300,
            "region": "EU",
            "reporting_year": 2025,
        },
        {
            "gender": "FEMALE",
            "contract_type": "TEMPORARY",
            "employment_type": "PART_TIME",
            "headcount": 150,
            "region": "EU",
            "reporting_year": 2025,
        },
        {
            "gender": "OTHER",
            "contract_type": "PERMANENT",
            "employment_type": "FULL_TIME",
            "headcount": 25,
            "region": "EU",
            "reporting_year": 2025,
        },
        {
            "gender": "MALE",
            "contract_type": "PERMANENT",
            "employment_type": "FULL_TIME",
            "headcount": 400,
            "region": "NON_EU",
            "reporting_year": 2025,
        },
        {
            "gender": "FEMALE",
            "contract_type": "PERMANENT",
            "employment_type": "FULL_TIME",
            "headcount": 350,
            "region": "NON_EU",
            "reporting_year": 2025,
        },
    ]


@pytest.fixture
def sample_training_records() -> List[Dict[str, Any]]:
    """Sample training data for S1-13 testing.

    Per ESRS S1-13, training hours must be disclosed per employee
    category, including topic areas.
    """
    return [
        {
            "category": "MANAGEMENT",
            "topic": "Sustainability Awareness",
            "total_hours": 1200,
            "participants": 150,
            "avg_hours_per_employee": Decimal("8.0"),
            "reporting_year": 2025,
        },
        {
            "category": "TECHNICAL",
            "topic": "Health & Safety",
            "total_hours": 8500,
            "participants": 2000,
            "avg_hours_per_employee": Decimal("4.25"),
            "reporting_year": 2025,
        },
        {
            "category": "ADMINISTRATIVE",
            "topic": "Anti-Corruption Compliance",
            "total_hours": 600,
            "participants": 300,
            "avg_hours_per_employee": Decimal("2.0"),
            "reporting_year": 2025,
        },
    ]


@pytest.fixture
def sample_h_and_s_metrics() -> Dict[str, Any]:
    """Sample health and safety metrics for S1-14 testing.

    Per ESRS S1-14, H&S data follows ILO recording and notification
    of occupational accidents and diseases conventions.
    """
    return {
        "reporting_year": 2025,
        "fatalities": 0,
        "high_consequence_injuries": 2,
        "recordable_injuries": 45,
        "lost_time_injury_frequency_rate": Decimal("3.2"),
        "total_hours_worked": 10_500_000,
        "near_misses_reported": 320,
        "safety_training_hours": 12500,
        "absenteeism_rate_percent": Decimal("3.8"),
        "occupational_diseases": 5,
        "workers_covered_by_h_and_s_system_percent": Decimal("100.0"),
    }


# ---------------------------------------------------------------------------
# 7.6  S2 Value Chain Workers Sample Data
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_value_chain_suppliers() -> List[Dict[str, Any]]:
    """Sample supplier data for S2 value chain worker assessment.

    Per ESRS S2-1 to S2-5, undertakings must disclose policies and actions
    related to workers in their upstream and downstream value chain.
    """
    return [
        {
            "supplier_name": "SteelCo Ltda",
            "country": "BR",
            "tier": 1,
            "worker_count": 5000,
            "risk_category": "HIGH",
            "audit_conducted": True,
            "last_audit_date": "2024-11-15",
            "corrective_actions_open": 3,
        },
        {
            "supplier_name": "ChemParts GmbH",
            "country": "DE",
            "tier": 1,
            "worker_count": 200,
            "risk_category": "LOW",
            "audit_conducted": True,
            "last_audit_date": "2025-01-20",
            "corrective_actions_open": 0,
        },
        {
            "supplier_name": "TextileFab Ltd",
            "country": "BD",
            "tier": 2,
            "worker_count": 12000,
            "risk_category": "HIGH",
            "audit_conducted": False,
            "last_audit_date": None,
            "corrective_actions_open": None,
        },
    ]


@pytest.fixture
def sample_value_chain_risk_assessments() -> List[Dict[str, Any]]:
    """Sample risk assessment records for S2 due diligence.

    Covers forced labour, child labour, living wage, and freedom of
    association risk areas per ESRS S2 and UNGPs.
    """
    return [
        {
            "risk_area": "FORCED_LABOUR",
            "geographic_scope": "South Asia",
            "inherent_risk": "HIGH",
            "residual_risk": "MEDIUM",
            "mitigation_actions": ["Supplier Code of Conduct", "Third-party audits"],
            "reporting_year": 2025,
        },
        {
            "risk_area": "CHILD_LABOUR",
            "geographic_scope": "Sub-Saharan Africa",
            "inherent_risk": "HIGH",
            "residual_risk": "HIGH",
            "mitigation_actions": ["Age verification", "Community programs"],
            "reporting_year": 2025,
        },
        {
            "risk_area": "LIVING_WAGE",
            "geographic_scope": "Southeast Asia",
            "inherent_risk": "MEDIUM",
            "residual_risk": "MEDIUM",
            "mitigation_actions": ["Living wage benchmark"],
            "reporting_year": 2025,
        },
    ]


# ---------------------------------------------------------------------------
# 7.7  S3 Affected Communities Sample Data
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_community_engagements() -> List[Dict[str, Any]]:
    """Sample community engagement records for S3-2 testing.

    Per ESRS S3-2, undertakings must disclose processes for engaging
    with affected communities including indigenous peoples.
    """
    return [
        {
            "community_name": "Vila Nova Settlement",
            "country": "BR",
            "engagement_type": "CONSULTATION",
            "indigenous_peoples": True,
            "fpic_obtained": True,
            "topics_discussed": ["Land use", "Water access", "Employment"],
            "date": "2025-03-10",
            "participants": 85,
        },
        {
            "community_name": "Rhine Valley Residents",
            "country": "DE",
            "engagement_type": "PUBLIC_HEARING",
            "indigenous_peoples": False,
            "fpic_obtained": None,
            "topics_discussed": ["Noise pollution", "Traffic"],
            "date": "2025-02-15",
            "participants": 120,
        },
        {
            "community_name": "Coastal Fishing Communities",
            "country": "ID",
            "engagement_type": "PARTICIPATORY_ASSESSMENT",
            "indigenous_peoples": True,
            "fpic_obtained": False,
            "topics_discussed": ["Marine pollution", "Livelihoods"],
            "date": "2025-01-22",
            "participants": 45,
        },
    ]


@pytest.fixture
def sample_community_grievances() -> List[Dict[str, Any]]:
    """Sample community grievance records for S3-3 testing.

    Per ESRS S3-3, grievance mechanisms must be accessible to
    affected communities with transparent resolution timelines.
    """
    return [
        {
            "grievance_id": "GRV-2025-001",
            "community": "Vila Nova Settlement",
            "category": "ENVIRONMENTAL",
            "description": "Dust from operations affecting crops",
            "date_filed": "2025-01-15",
            "status": "RESOLVED",
            "resolution_days": 45,
        },
        {
            "grievance_id": "GRV-2025-002",
            "community": "Rhine Valley Residents",
            "category": "NOISE",
            "description": "Night-time operations exceeding noise limits",
            "date_filed": "2025-02-20",
            "status": "IN_PROGRESS",
            "resolution_days": None,
        },
        {
            "grievance_id": "GRV-2025-003",
            "community": "Coastal Fishing Communities",
            "category": "WATER_QUALITY",
            "description": "Discharge affecting fish stocks",
            "date_filed": "2025-03-01",
            "status": "OPEN",
            "resolution_days": None,
        },
    ]


# ---------------------------------------------------------------------------
# 7.8  S4 Consumers & End-Users Sample Data
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_consumer_complaints() -> List[Dict[str, Any]]:
    """Sample consumer complaint records for S4 assessment.

    Per ESRS S4-3, undertakings must disclose processes for handling
    consumer complaints and remediation.
    """
    return [
        {
            "complaint_id": "CC-2025-0001",
            "category": "PRODUCT_SAFETY",
            "severity": "HIGH",
            "date_received": "2025-01-10",
            "status": "RESOLVED",
            "resolution_days": 12,
            "product_line": "Electronics",
        },
        {
            "complaint_id": "CC-2025-0002",
            "category": "DATA_PRIVACY",
            "severity": "MEDIUM",
            "date_received": "2025-02-05",
            "status": "IN_PROGRESS",
            "resolution_days": None,
            "product_line": "Digital Services",
        },
        {
            "complaint_id": "CC-2025-0003",
            "category": "MISLEADING_INFORMATION",
            "severity": "LOW",
            "date_received": "2025-02-18",
            "status": "RESOLVED",
            "resolution_days": 5,
            "product_line": "Consumer Goods",
        },
    ]


@pytest.fixture
def sample_product_safety_records() -> List[Dict[str, Any]]:
    """Sample product safety records for S4-4 action tracking.

    Includes product recalls, safety incidents, and RAPEX notifications.
    """
    return [
        {
            "product_name": "Widget Pro 3000",
            "product_category": "Electronics",
            "incident_type": "RECALL",
            "severity": "HIGH",
            "units_affected": 15000,
            "date_identified": "2025-01-05",
            "rapex_notification": True,
            "root_cause": "Battery overheating",
            "corrective_action": "Full product recall and replacement",
        },
        {
            "product_name": "CleanMax Spray",
            "product_category": "Household",
            "incident_type": "SAFETY_ALERT",
            "severity": "MEDIUM",
            "units_affected": 2000,
            "date_identified": "2025-03-01",
            "rapex_notification": False,
            "root_cause": "Labelling error",
            "corrective_action": "Updated labels distributed",
        },
    ]


# ---------------------------------------------------------------------------
# 7.9  G1 Business Conduct Sample Data
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_governance_policies() -> List[Dict[str, Any]]:
    """Sample governance policy data for G1-1 testing.

    Per ESRS G1-1, undertakings must disclose corporate culture,
    business conduct policies, and supplier/business partner standards.
    """
    return [
        {
            "policy_name": "Anti-Corruption and Anti-Bribery Policy",
            "version": "3.0",
            "effective_date": "2024-01-01",
            "review_date": "2025-12-31",
            "scope": "GROUP_WIDE",
            "approved_by": "Board of Directors",
            "training_required": True,
            "whistleblower_channel": True,
        },
        {
            "policy_name": "Code of Business Ethics",
            "version": "2.1",
            "effective_date": "2023-07-01",
            "review_date": "2025-06-30",
            "scope": "GROUP_WIDE",
            "approved_by": "Board of Directors",
            "training_required": True,
            "whistleblower_channel": True,
        },
        {
            "policy_name": "Supplier Code of Conduct",
            "version": "1.5",
            "effective_date": "2024-04-01",
            "review_date": "2026-03-31",
            "scope": "VALUE_CHAIN",
            "approved_by": "Chief Procurement Officer",
            "training_required": False,
            "whistleblower_channel": False,
        },
    ]


@pytest.fixture
def sample_corruption_incidents() -> List[Dict[str, Any]]:
    """Sample corruption incident data for G1-4 testing.

    Per ESRS G1-4, confirmed incidents of corruption or bribery
    must be disclosed including legal proceedings.
    """
    return [
        {
            "incident_id": "CORR-2024-001",
            "type": "BRIBERY_ALLEGATION",
            "date_identified": "2024-06-15",
            "country": "NG",
            "status": "INVESTIGATED_DISMISSED",
            "legal_proceedings": False,
            "employees_disciplined": 0,
            "contracts_terminated": 0,
        },
        {
            "incident_id": "CORR-2025-001",
            "type": "FACILITATION_PAYMENT",
            "date_identified": "2025-01-20",
            "country": "IN",
            "status": "UNDER_INVESTIGATION",
            "legal_proceedings": False,
            "employees_disciplined": 1,
            "contracts_terminated": 0,
        },
    ]


@pytest.fixture
def sample_payment_records() -> List[Dict[str, Any]]:
    """Sample payment practice data for G1-6 testing.

    Per ESRS G1-6, undertakings must disclose payment practices
    including average payment terms and late payment statistics.
    """
    return [
        {
            "supplier_category": "SME",
            "average_payment_days": 42,
            "agreed_payment_terms_days": 30,
            "late_payment_percentage": Decimal("15.0"),
            "total_invoices": 8500,
            "late_invoices": 1275,
            "reporting_year": 2025,
        },
        {
            "supplier_category": "LARGE_ENTERPRISE",
            "average_payment_days": 55,
            "agreed_payment_terms_days": 60,
            "late_payment_percentage": Decimal("5.0"),
            "total_invoices": 2200,
            "late_invoices": 110,
            "reporting_year": 2025,
        },
        {
            "supplier_category": "MICRO_ENTERPRISE",
            "average_payment_days": 28,
            "agreed_payment_terms_days": 30,
            "late_payment_percentage": Decimal("8.0"),
            "total_invoices": 1500,
            "late_invoices": 120,
            "reporting_year": 2025,
        },
    ]


# ---------------------------------------------------------------------------
# 7.10  ESRS 2 General Disclosures / Board Composition Sample Data
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_board_members() -> List[Dict[str, Any]]:
    """Sample board composition data for ESRS 2 GOV-1 testing.

    Per ESRS 2 GOV-1 para 21-27, undertakings must disclose
    governance body composition, diversity, and sustainability expertise.
    """
    return [
        {
            "name": "Dr. Anna Mueller",
            "role": "Chairperson",
            "body_type": "SUPERVISORY_BOARD",
            "gender": "FEMALE",
            "independent": True,
            "sustainability_expertise": True,
            "tenure_years": 6,
            "committee_memberships": ["Audit", "Sustainability"],
        },
        {
            "name": "Thomas Schmidt",
            "role": "Deputy Chair",
            "body_type": "SUPERVISORY_BOARD",
            "gender": "MALE",
            "independent": True,
            "sustainability_expertise": False,
            "tenure_years": 4,
            "committee_memberships": ["Audit", "Remuneration"],
        },
        {
            "name": "Maria Fernandez",
            "role": "Member",
            "body_type": "SUPERVISORY_BOARD",
            "gender": "FEMALE",
            "independent": True,
            "sustainability_expertise": True,
            "tenure_years": 2,
            "committee_memberships": ["Sustainability", "Risk"],
        },
        {
            "name": "Li Wei",
            "role": "CEO",
            "body_type": "MANAGEMENT_BOARD",
            "gender": "MALE",
            "independent": False,
            "sustainability_expertise": True,
            "tenure_years": 8,
            "committee_memberships": [],
        },
        {
            "name": "Sarah Johnson",
            "role": "CFO",
            "body_type": "MANAGEMENT_BOARD",
            "gender": "FEMALE",
            "independent": False,
            "sustainability_expertise": False,
            "tenure_years": 3,
            "committee_memberships": [],
        },
    ]


@pytest.fixture
def sample_board_committees() -> List[Dict[str, Any]]:
    """Sample board committee data for ESRS 2 GOV-1 testing.

    Tracks sustainability oversight responsibilities at committee level.
    """
    return [
        {
            "committee_name": "Sustainability Committee",
            "chair": "Dr. Anna Mueller",
            "members_count": 3,
            "meetings_per_year": 4,
            "sustainability_mandate": True,
            "climate_oversight": True,
            "social_oversight": True,
        },
        {
            "committee_name": "Audit Committee",
            "chair": "Thomas Schmidt",
            "members_count": 4,
            "meetings_per_year": 6,
            "sustainability_mandate": False,
            "climate_oversight": False,
            "social_oversight": False,
        },
        {
            "committee_name": "Risk Committee",
            "chair": "Maria Fernandez",
            "members_count": 3,
            "meetings_per_year": 4,
            "sustainability_mandate": True,
            "climate_oversight": True,
            "social_oversight": False,
        },
    ]
