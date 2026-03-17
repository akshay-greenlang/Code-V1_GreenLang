# -*- coding: utf-8 -*-
"""
PACK-012 CSRD Financial Service Pack - Shared Test Fixtures
=============================================================

Provides reusable pytest fixtures for all PACK-012 test modules including
financed emissions (PCAF), insurance underwriting, Green Asset Ratio (GAR),
Banking Book Taxonomy Alignment Ratio (BTAR), climate risk scoring (NGFS),
financial sector double materiality, transition plan generation, and EBA
Pillar 3 ESG disclosures.

Financial Institution Types:
- BANK: Credit institution subject to CRR/CRD VI (GAR, BTAR, Pillar 3)
- INSURANCE: Insurance/reinsurance undertaking (Solvency II, underwriting)
- ASSET_MANAGER: UCITS management company or AIFM (SFDR, WACI)
- INVESTMENT_FIRM: MiFID II investment firm (product governance, ESG prefs)
- PENSION_FUND: IORP II pension scheme (stewardship, long-horizon)
- CONGLOMERATE: Financial conglomerate (multi-entity, cross-sector)

PCAF Asset Classes (6 core + 4 extended):
- Listed equity and corporate bonds
- Business loans and unlisted equity
- Project finance
- Commercial real estate
- Mortgages
- Motor vehicle loans

Regulatory Context:
- CSRD: Directive (EU) 2022/2464
- ESRS: Delegated Regulation (EU) 2023/2772
- EU Taxonomy: Regulation (EU) 2020/852
- CRR/CRD VI: Article 449a (Pillar 3 ESG)
- SFDR: Regulation (EU) 2019/2088
- Solvency II: Directive 2009/138/EC
- PCAF: Global GHG Accounting Standard v3
- SBTi FI: Financial Institutions Framework v1.1
- EBA ITS: Pillar 3 ESG Disclosures (2022/01, 2024 update)

Author: GreenLang QA Team
Version: 1.0.0
"""

import hashlib
import importlib.util
import json
import re
import sys
import uuid
from datetime import date, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock, Mock

import pytest


# ---------------------------------------------------------------------------
# Dynamic import helper for hyphenated directory names
# ---------------------------------------------------------------------------

def _import_from_path(module_name: str, file_path: Path):
    """Import a module from a file path (supports hyphenated directories).

    Args:
        module_name: The name to assign to the imported module.
        file_path: Absolute path to the Python file to import.

    Returns:
        The loaded module object.

    Raises:
        ImportError: If the spec cannot be loaded from the given path.
    """
    spec = importlib.util.spec_from_file_location(module_name, str(file_path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load spec from {file_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

PACK_ROOT = Path(__file__).resolve().parent.parent
PACK_YAML_PATH = PACK_ROOT / "pack.yaml"
CONFIG_DIR = PACK_ROOT / "config"
PRESETS_DIR = CONFIG_DIR / "presets"
DEMO_DIR = CONFIG_DIR / "demo"
ENGINES_DIR = PACK_ROOT / "engines"
WORKFLOWS_DIR = PACK_ROOT / "workflows"
TEMPLATES_DIR = PACK_ROOT / "templates"
INTEGRATIONS_DIR = PACK_ROOT / "integrations"


# ---------------------------------------------------------------------------
# Constants - Financial institution types and PCAF asset classes
# ---------------------------------------------------------------------------

FINANCIAL_INSTITUTION_TYPES = [
    "BANK", "INSURANCE", "ASSET_MANAGER",
    "INVESTMENT_FIRM", "PENSION_FUND", "CONGLOMERATE",
]

PCAF_ASSET_CLASSES = [
    "LISTED_EQUITY_CORPORATE_BONDS",
    "BUSINESS_LOANS_UNLISTED_EQUITY",
    "PROJECT_FINANCE",
    "COMMERCIAL_REAL_ESTATE",
    "MORTGAGES",
    "MOTOR_VEHICLE_LOANS",
    "SOVEREIGN_BONDS",
    "PRIVATE_EQUITY",
    "GREEN_BONDS",
    "SECURITIZED_PRODUCTS",
]

PCAF_CORE_ASSET_CLASSES = [
    "LISTED_EQUITY_CORPORATE_BONDS",
    "BUSINESS_LOANS_UNLISTED_EQUITY",
    "PROJECT_FINANCE",
    "COMMERCIAL_REAL_ESTATE",
    "MORTGAGES",
    "MOTOR_VEHICLE_LOANS",
]

PCAF_DATA_QUALITY_SCORES = [1, 2, 3, 4, 5]

GAR_SCOPES = ["TURNOVER", "CAPEX", "OPEX"]

NGFS_SCENARIOS = [
    "NET_ZERO_2050", "BELOW_2C", "DELAYED_TRANSITION",
    "NDCS", "DIVERGENT_NET_ZERO", "CURRENT_POLICIES",
]

PILLAR3_TEMPLATES = [
    "TEMPLATE_1", "TEMPLATE_2", "TEMPLATE_3", "TEMPLATE_4",
    "TEMPLATE_5", "TEMPLATE_7", "TEMPLATE_8", "TEMPLATE_9", "TEMPLATE_10",
]

ESRS_TOPICS = ["E1", "E2", "E3", "E4", "E5", "S1", "S2", "S3", "S4", "G1"]

PRESET_NAMES = [
    "bank", "insurance", "asset_manager",
    "investment_firm", "pension_fund", "conglomerate",
]

INSURANCE_LINE_TYPES = [
    "COMMERCIAL_PROPERTY", "COMMERCIAL_CASUALTY",
    "PERSONAL_PROPERTY", "PERSONAL_AUTO",
    "SPECIALTY", "LIFE_HEALTH", "REINSURANCE",
]


# ---------------------------------------------------------------------------
# Component ID lists
# ---------------------------------------------------------------------------

FS_ENGINE_IDS = [
    "financed_emissions",
    "insurance_underwriting",
    "green_asset_ratio",
    "btar_calculator",
    "climate_risk_scoring",
    "fs_double_materiality",
    "fs_transition_plan",
    "pillar3_esg",
]

FS_WORKFLOW_IDS = [
    "financed_emissions_workflow",
    "gar_btar_workflow",
    "insurance_emissions_workflow",
    "climate_stress_test",
    "fs_materiality",
    "transition_plan",
    "pillar3_reporting",
    "regulatory_integration",
]

FS_TEMPLATE_IDS = [
    "pcaf_report",
    "gar_btar_report",
    "pillar3_esg",
    "climate_risk_report",
    "fs_esrs_chapter",
    "financed_emissions_dashboard",
    "insurance_esg",
    "sbti_fi_report",
]

FS_INTEGRATION_IDS = [
    "pack_orchestrator",
    "csrd_pack_bridge",
    "sfdr_pack_bridge",
    "taxonomy_pack_bridge",
    "mrv_investments_bridge",
    "finance_agent_bridge",
    "climate_risk_bridge",
    "eba_pillar3_bridge",
    "health_check",
    "setup_wizard",
]


# ---------------------------------------------------------------------------
# File mapping dictionaries
# ---------------------------------------------------------------------------

ENGINE_FILES = {
    "financed_emissions": "financed_emissions_engine.py",
    "insurance_underwriting": "insurance_underwriting_engine.py",
    "green_asset_ratio": "green_asset_ratio_engine.py",
    "btar_calculator": "btar_calculator_engine.py",
    "climate_risk_scoring": "climate_risk_scoring_engine.py",
    "fs_double_materiality": "fs_double_materiality_engine.py",
    "fs_transition_plan": "fs_transition_plan_engine.py",
    "pillar3_esg": "pillar3_esg_engine.py",
}

WORKFLOW_FILES = {
    "financed_emissions_workflow": "financed_emissions_workflow.py",
    "gar_btar_workflow": "gar_btar_workflow.py",
    "insurance_emissions_workflow": "insurance_emissions_workflow.py",
    "climate_stress_test": "climate_stress_test_workflow.py",
    "fs_materiality": "fs_materiality_workflow.py",
    "transition_plan": "transition_plan_workflow.py",
    "pillar3_reporting": "pillar3_reporting_workflow.py",
    "regulatory_integration": "regulatory_integration_workflow.py",
}

TEMPLATE_FILES = {
    "pcaf_report": "pcaf_report.py",
    "gar_btar_report": "gar_btar_report.py",
    "pillar3_esg": "pillar3_esg_template.py",
    "climate_risk_report": "climate_risk_report.py",
    "fs_esrs_chapter": "fs_esrs_chapter.py",
    "financed_emissions_dashboard": "financed_emissions_dashboard.py",
    "insurance_esg": "insurance_esg_template.py",
    "sbti_fi_report": "sbti_fi_report.py",
}

INTEGRATION_FILES = {
    "pack_orchestrator": "pack_orchestrator.py",
    "csrd_pack_bridge": "csrd_pack_bridge.py",
    "sfdr_pack_bridge": "sfdr_pack_bridge.py",
    "taxonomy_pack_bridge": "taxonomy_pack_bridge.py",
    "mrv_investments_bridge": "mrv_investments_bridge.py",
    "finance_agent_bridge": "finance_agent_bridge.py",
    "climate_risk_bridge": "climate_risk_bridge.py",
    "eba_pillar3_bridge": "eba_pillar3_bridge.py",
    "health_check": "health_check.py",
    "setup_wizard": "setup_wizard.py",
}


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def _compute_hash(data: Any) -> str:
    """Compute SHA-256 hash of arbitrary data for provenance tracking."""
    if isinstance(data, dict):
        serialized = json.dumps(data, sort_keys=True, default=str)
    elif isinstance(data, str):
        serialized = data
    else:
        serialized = str(data)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def assert_provenance_hash(result: Dict[str, Any]) -> None:
    """Verify that a result contains a valid SHA-256 provenance hash."""
    assert "provenance_hash" in result, "Result missing 'provenance_hash' field"
    h = result["provenance_hash"]
    assert isinstance(h, str), f"provenance_hash must be str, got {type(h)}"
    assert len(h) == 64, f"SHA-256 hash must be 64 chars, got {len(h)}"
    assert re.match(r"^[0-9a-f]{64}$", h), f"Invalid hex hash: {h}"


def assert_valid_uuid(value: str) -> None:
    """Verify that a value is a valid UUID4 string."""
    assert isinstance(value, str), f"Expected str, got {type(value)}"
    try:
        parsed = uuid.UUID(value, version=4)
        assert str(parsed) == value.lower(), f"UUID normalization mismatch: {value}"
    except ValueError:
        raise AssertionError(f"Invalid UUID4: {value}")


def _safe_import(module_name: str, file_path: Path):
    """Import a module, returning None if file does not exist or fails."""
    if not file_path.exists():
        return None
    try:
        return _import_from_path(module_name, file_path)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Directory fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def pack_dir() -> Path:
    """Return the absolute path to the PACK-012 root directory."""
    return PACK_ROOT


@pytest.fixture(scope="session")
def config_dir() -> Path:
    """Return the absolute path to the config directory."""
    return CONFIG_DIR


@pytest.fixture(scope="session")
def engines_dir() -> Path:
    """Return the absolute path to the engines directory."""
    return ENGINES_DIR


@pytest.fixture(scope="session")
def workflows_dir() -> Path:
    """Return the absolute path to the workflows directory."""
    return WORKFLOWS_DIR


@pytest.fixture(scope="session")
def templates_dir() -> Path:
    """Return the absolute path to the templates directory."""
    return TEMPLATES_DIR


@pytest.fixture(scope="session")
def integrations_dir() -> Path:
    """Return the absolute path to the integrations directory."""
    return INTEGRATIONS_DIR


# ---------------------------------------------------------------------------
# Pack YAML fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def pack_yaml_path() -> Path:
    """Return the absolute path to pack.yaml."""
    return PACK_YAML_PATH


@pytest.fixture(scope="session")
def pack_yaml_raw(pack_yaml_path) -> str:
    """Return the raw text content of pack.yaml, or empty string if missing."""
    if pack_yaml_path.exists():
        return pack_yaml_path.read_text(encoding="utf-8")
    return ""


@pytest.fixture(scope="session")
def pack_yaml(pack_yaml_raw) -> Dict[str, Any]:
    """Return the parsed pack.yaml as a dictionary, or empty dict if missing."""
    if pack_yaml_raw:
        import yaml
        return yaml.safe_load(pack_yaml_raw) or {}
    return {}


# ---------------------------------------------------------------------------
# Configuration fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def pack_config_module():
    """Return the pack_config module for direct class/enum access.

    Dynamically imports the config module from the hyphenated pack directory.
    """
    return _import_from_path(
        "fs12_pack_config",
        CONFIG_DIR / "pack_config.py",
    )


@pytest.fixture
def fs_config(pack_config_module):
    """Create a default CSRDFinancialServiceConfig instance.

    Returns a CSRDFinancialServiceConfig with bank defaults: PCAF enabled,
    GAR/BTAR enabled, Pillar 3 enabled, climate risk enabled, and all
    ESRS topics assessed for double materiality.
    """
    return pack_config_module.CSRDFinancialServiceConfig()


@pytest.fixture
def pack_config_instance(pack_config_module):
    """Create a default PackConfig instance loaded from defaults."""
    return pack_config_module.PackConfig()


# ---------------------------------------------------------------------------
# Engine module fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def engine_modules() -> Dict[str, Any]:
    """Import all 8 engine modules and return as a dictionary.

    Keys are engine IDs, values are imported module objects (or None if
    the module could not be loaded).
    """
    modules = {}
    for engine_id, filename in ENGINE_FILES.items():
        modules[engine_id] = _safe_import(
            f"fs12_engine_{engine_id}",
            ENGINES_DIR / filename,
        )
    return modules


# ---------------------------------------------------------------------------
# Workflow module fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def workflow_modules() -> Dict[str, Any]:
    """Import all 8 workflow modules and return as a dictionary.

    Keys are workflow IDs, values are imported module objects (or None if
    the module could not be loaded).
    """
    modules = {}
    for workflow_id, filename in WORKFLOW_FILES.items():
        modules[workflow_id] = _safe_import(
            f"fs12_workflow_{workflow_id}",
            WORKFLOWS_DIR / filename,
        )
    return modules


# ---------------------------------------------------------------------------
# Template module fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def template_modules() -> Dict[str, Any]:
    """Import all 8 template modules and return as a dictionary.

    Keys are template IDs, values are imported module objects (or None if
    the module could not be loaded).
    """
    modules = {}
    for template_id, filename in TEMPLATE_FILES.items():
        modules[template_id] = _safe_import(
            f"fs12_template_{template_id}",
            TEMPLATES_DIR / filename,
        )
    return modules


# ---------------------------------------------------------------------------
# Integration module fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def integration_modules() -> Dict[str, Any]:
    """Import all 10 integration modules and return as a dictionary.

    Keys are integration IDs, values are imported module objects (or None if
    the module could not be loaded).
    """
    modules = {}
    for integration_id, filename in INTEGRATION_FILES.items():
        modules[integration_id] = _safe_import(
            f"fs12_integration_{integration_id}",
            INTEGRATIONS_DIR / filename,
        )
    return modules


# ---------------------------------------------------------------------------
# Sample data fixtures - Banking portfolio for PCAF financed emissions
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_banking_portfolio() -> List[Dict[str, Any]]:
    """Create sample banking portfolio data for PCAF financed emissions testing.

    Returns a list of 12 holding dictionaries representing a diversified
    bank lending and securities portfolio with exposure data, NACE sectors,
    and emissions data suitable for PCAF attribution calculations.
    """
    return [
        {
            "counterparty_id": "CP-001", "name": "EuroSteel GmbH",
            "nace_code": "C24.10", "nace_sector": "Manufacture of basic iron and steel",
            "asset_class": "BUSINESS_LOANS_UNLISTED_EQUITY",
            "outstanding_amount_eur": 50_000_000.0,
            "total_equity_debt_eur": 200_000_000.0,
            "scope_1_tco2e": 450_000.0, "scope_2_tco2e": 120_000.0,
            "scope_3_tco2e": 850_000.0, "data_quality_score": 2,
            "country": "DE", "taxonomy_aligned_pct": 5.0,
        },
        {
            "counterparty_id": "CP-002", "name": "Nordic Wind Energy AS",
            "nace_code": "D35.11", "nace_sector": "Production of electricity",
            "asset_class": "LISTED_EQUITY_CORPORATE_BONDS",
            "outstanding_amount_eur": 30_000_000.0,
            "total_equity_debt_eur": 500_000_000.0,
            "scope_1_tco2e": 500.0, "scope_2_tco2e": 200.0,
            "scope_3_tco2e": 15_000.0, "data_quality_score": 1,
            "country": "NO", "taxonomy_aligned_pct": 92.0,
        },
        {
            "counterparty_id": "CP-003", "name": "Mediterranean Cement SA",
            "nace_code": "C23.51", "nace_sector": "Manufacture of cement",
            "asset_class": "BUSINESS_LOANS_UNLISTED_EQUITY",
            "outstanding_amount_eur": 25_000_000.0,
            "total_equity_debt_eur": 150_000_000.0,
            "scope_1_tco2e": 380_000.0, "scope_2_tco2e": 45_000.0,
            "scope_3_tco2e": 200_000.0, "data_quality_score": 3,
            "country": "ES", "taxonomy_aligned_pct": 2.0,
        },
        {
            "counterparty_id": "CP-004", "name": "GreenTech Solar BV",
            "nace_code": "D35.11", "nace_sector": "Production of electricity",
            "asset_class": "PROJECT_FINANCE",
            "outstanding_amount_eur": 40_000_000.0,
            "total_equity_debt_eur": 80_000_000.0,
            "scope_1_tco2e": 100.0, "scope_2_tco2e": 50.0,
            "scope_3_tco2e": 5_000.0, "data_quality_score": 1,
            "country": "NL", "taxonomy_aligned_pct": 98.0,
        },
        {
            "counterparty_id": "CP-005", "name": "EuroProperty REIT",
            "nace_code": "L68.20", "nace_sector": "Renting and operating of own real estate",
            "asset_class": "COMMERCIAL_REAL_ESTATE",
            "outstanding_amount_eur": 80_000_000.0,
            "property_value_eur": 120_000_000.0,
            "scope_1_tco2e": 2_000.0, "scope_2_tco2e": 8_000.0,
            "scope_3_tco2e": 1_500.0, "data_quality_score": 2,
            "country": "FR", "taxonomy_aligned_pct": 35.0, "epc_label": "B",
        },
        {
            "counterparty_id": "MG-001", "name": "Residential Mortgage Pool DE",
            "nace_code": "N/A", "nace_sector": "Household mortgages",
            "asset_class": "MORTGAGES",
            "outstanding_amount_eur": 200_000_000.0,
            "property_value_eur": 300_000_000.0,
            "scope_1_tco2e": 5_000.0, "scope_2_tco2e": 15_000.0,
            "scope_3_tco2e": 0.0, "data_quality_score": 3,
            "country": "DE", "taxonomy_aligned_pct": 20.0, "epc_label": "C",
        },
        {
            "counterparty_id": "MV-001", "name": "Auto Loan Portfolio",
            "nace_code": "N/A", "nace_sector": "Motor vehicle loans",
            "asset_class": "MOTOR_VEHICLE_LOANS",
            "outstanding_amount_eur": 60_000_000.0,
            "vehicle_value_eur": 75_000_000.0,
            "scope_1_tco2e": 18_000.0, "scope_2_tco2e": 500.0,
            "scope_3_tco2e": 3_000.0, "data_quality_score": 4,
            "country": "DE", "taxonomy_aligned_pct": 10.0,
        },
        {
            "counterparty_id": "CP-006", "name": "BioPharmaCorp AG",
            "nace_code": "C21.10", "nace_sector": "Manufacture of basic pharmaceutical products",
            "asset_class": "LISTED_EQUITY_CORPORATE_BONDS",
            "outstanding_amount_eur": 20_000_000.0,
            "total_equity_debt_eur": 800_000_000.0,
            "scope_1_tco2e": 5_000.0, "scope_2_tco2e": 12_000.0,
            "scope_3_tco2e": 60_000.0, "data_quality_score": 1,
            "country": "CH", "taxonomy_aligned_pct": 8.0,
        },
        {
            "counterparty_id": "CP-007", "name": "PetroChemEurope NV",
            "nace_code": "B06.10", "nace_sector": "Extraction of crude petroleum",
            "asset_class": "LISTED_EQUITY_CORPORATE_BONDS",
            "outstanding_amount_eur": 15_000_000.0,
            "total_equity_debt_eur": 1_200_000_000.0,
            "scope_1_tco2e": 2_500_000.0, "scope_2_tco2e": 400_000.0,
            "scope_3_tco2e": 12_000_000.0, "data_quality_score": 1,
            "country": "NL", "taxonomy_aligned_pct": 0.5,
        },
        {
            "counterparty_id": "CP-008", "name": "TransportLogistics SpA",
            "nace_code": "H49.41", "nace_sector": "Freight transport by road",
            "asset_class": "BUSINESS_LOANS_UNLISTED_EQUITY",
            "outstanding_amount_eur": 10_000_000.0,
            "total_equity_debt_eur": 40_000_000.0,
            "scope_1_tco2e": 35_000.0, "scope_2_tco2e": 2_000.0,
            "scope_3_tco2e": 8_000.0, "data_quality_score": 3,
            "country": "IT", "taxonomy_aligned_pct": 3.0,
        },
        {
            "counterparty_id": "CP-009", "name": "AgriFood Holdings plc",
            "nace_code": "A01.11", "nace_sector": "Growing of cereals",
            "asset_class": "BUSINESS_LOANS_UNLISTED_EQUITY",
            "outstanding_amount_eur": 8_000_000.0,
            "total_equity_debt_eur": 30_000_000.0,
            "scope_1_tco2e": 15_000.0, "scope_2_tco2e": 3_000.0,
            "scope_3_tco2e": 45_000.0, "data_quality_score": 4,
            "country": "IE", "taxonomy_aligned_pct": 12.0,
        },
        {
            "counterparty_id": "CP-010", "name": "CleanHydrogen Corp",
            "nace_code": "C20.11", "nace_sector": "Manufacture of industrial gases",
            "asset_class": "PROJECT_FINANCE",
            "outstanding_amount_eur": 35_000_000.0,
            "total_equity_debt_eur": 70_000_000.0,
            "scope_1_tco2e": 800.0, "scope_2_tco2e": 1_200.0,
            "scope_3_tco2e": 4_000.0, "data_quality_score": 2,
            "country": "DE", "taxonomy_aligned_pct": 85.0,
        },
    ]


# ---------------------------------------------------------------------------
# Sample data fixtures - Insurance portfolio
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_insurance_portfolio() -> List[Dict[str, Any]]:
    """Create sample insurance underwriting portfolio data.

    Returns a list of 6 lines of business with policy data for
    insurance-associated emissions calculation per PCAF extension.
    """
    return [
        {
            "line_of_business": "COMMERCIAL_PROPERTY", "policy_count": 1_200,
            "gross_written_premium_eur": 85_000_000.0,
            "net_written_premium_eur": 60_000_000.0,
            "claims_paid_eur": 35_000_000.0, "reinsurance_ceded_pct": 29.4,
            "sector_emission_factor_tco2e_per_eur_m": 42.5,
            "country_mix": {"DE": 0.40, "FR": 0.25, "IT": 0.20, "ES": 0.15},
        },
        {
            "line_of_business": "COMMERCIAL_CASUALTY", "policy_count": 3_500,
            "gross_written_premium_eur": 120_000_000.0,
            "net_written_premium_eur": 95_000_000.0,
            "claims_paid_eur": 50_000_000.0, "reinsurance_ceded_pct": 20.8,
            "sector_emission_factor_tco2e_per_eur_m": 28.0,
            "country_mix": {"DE": 0.35, "NL": 0.30, "BE": 0.20, "AT": 0.15},
        },
        {
            "line_of_business": "PERSONAL_PROPERTY", "policy_count": 50_000,
            "gross_written_premium_eur": 200_000_000.0,
            "net_written_premium_eur": 180_000_000.0,
            "claims_paid_eur": 90_000_000.0, "reinsurance_ceded_pct": 10.0,
            "sector_emission_factor_tco2e_per_eur_m": 15.0,
            "country_mix": {"DE": 0.60, "AT": 0.25, "CH": 0.15},
        },
        {
            "line_of_business": "PERSONAL_AUTO", "policy_count": 80_000,
            "gross_written_premium_eur": 150_000_000.0,
            "net_written_premium_eur": 140_000_000.0,
            "claims_paid_eur": 100_000_000.0, "reinsurance_ceded_pct": 6.7,
            "sector_emission_factor_tco2e_per_eur_m": 55.0,
            "country_mix": {"DE": 0.55, "FR": 0.25, "IT": 0.20},
        },
        {
            "line_of_business": "SPECIALTY", "policy_count": 200,
            "gross_written_premium_eur": 45_000_000.0,
            "net_written_premium_eur": 20_000_000.0,
            "claims_paid_eur": 10_000_000.0, "reinsurance_ceded_pct": 55.6,
            "sector_emission_factor_tco2e_per_eur_m": 65.0,
            "country_mix": {"GB": 0.40, "DE": 0.30, "NL": 0.30},
        },
        {
            "line_of_business": "LIFE_HEALTH", "policy_count": 25_000,
            "gross_written_premium_eur": 300_000_000.0,
            "net_written_premium_eur": 280_000_000.0,
            "claims_paid_eur": 150_000_000.0, "reinsurance_ceded_pct": 6.7,
            "sector_emission_factor_tco2e_per_eur_m": 5.0,
            "country_mix": {"DE": 0.70, "AT": 0.20, "CH": 0.10},
        },
    ]


# ---------------------------------------------------------------------------
# Sample data fixtures - Counterparty data with NACE codes
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_counterparties() -> List[Dict[str, Any]]:
    """Create sample counterparty data with NACE codes for GAR/BTAR testing.

    Returns a list of 8 counterparties with taxonomy alignment data
    for Green Asset Ratio and BTAR calculation testing.
    """
    return [
        {
            "counterparty_id": "NFC-001", "name": "EuroSteel GmbH",
            "type": "NON_FINANCIAL_CORPORATES", "nace_code": "C24.10",
            "exposure_eur": 50_000_000.0, "taxonomy_eligible": True,
            "taxonomy_aligned_turnover_pct": 5.0, "taxonomy_aligned_capex_pct": 15.0,
            "taxonomy_aligned_opex_pct": 8.0, "environmental_objective": "CLIMATE_MITIGATION",
            "substantial_contribution": True, "dnsh_pass": True, "minimum_safeguards": True,
        },
        {
            "counterparty_id": "NFC-002", "name": "Nordic Wind Energy AS",
            "type": "NON_FINANCIAL_CORPORATES", "nace_code": "D35.11",
            "exposure_eur": 30_000_000.0, "taxonomy_eligible": True,
            "taxonomy_aligned_turnover_pct": 92.0, "taxonomy_aligned_capex_pct": 95.0,
            "taxonomy_aligned_opex_pct": 88.0, "environmental_objective": "CLIMATE_MITIGATION",
            "substantial_contribution": True, "dnsh_pass": True, "minimum_safeguards": True,
        },
        {
            "counterparty_id": "FC-001", "name": "EuroBank Subsidiary AG",
            "type": "FINANCIAL_CORPORATES", "nace_code": "K64.19",
            "exposure_eur": 20_000_000.0, "taxonomy_eligible": True,
            "taxonomy_aligned_turnover_pct": 30.0, "taxonomy_aligned_capex_pct": 25.0,
            "taxonomy_aligned_opex_pct": 28.0, "environmental_objective": "CLIMATE_MITIGATION",
            "substantial_contribution": True, "dnsh_pass": True, "minimum_safeguards": True,
        },
        {
            "counterparty_id": "HH-001", "name": "Residential Mortgage Pool",
            "type": "HOUSEHOLDS_MORTGAGES", "nace_code": "N/A",
            "exposure_eur": 200_000_000.0, "taxonomy_eligible": True,
            "taxonomy_aligned_turnover_pct": 20.0, "taxonomy_aligned_capex_pct": 20.0,
            "taxonomy_aligned_opex_pct": 20.0, "environmental_objective": "CLIMATE_MITIGATION",
            "substantial_contribution": True, "dnsh_pass": True, "minimum_safeguards": True,
            "epc_label": "A",
        },
        {
            "counterparty_id": "HH-002", "name": "Motor Vehicle Loan Pool",
            "type": "HOUSEHOLDS_MOTOR_VEHICLE", "nace_code": "N/A",
            "exposure_eur": 60_000_000.0, "taxonomy_eligible": True,
            "taxonomy_aligned_turnover_pct": 15.0, "taxonomy_aligned_capex_pct": 15.0,
            "taxonomy_aligned_opex_pct": 15.0, "environmental_objective": "CLIMATE_MITIGATION",
            "substantial_contribution": True, "dnsh_pass": True, "minimum_safeguards": True,
        },
        {
            "counterparty_id": "LG-001", "name": "City of Munich Green Bond",
            "type": "LOCAL_GOVERNMENTS", "nace_code": "O84.11",
            "exposure_eur": 15_000_000.0, "taxonomy_eligible": True,
            "taxonomy_aligned_turnover_pct": 60.0, "taxonomy_aligned_capex_pct": 65.0,
            "taxonomy_aligned_opex_pct": 55.0, "environmental_objective": "CLIMATE_ADAPTATION",
            "substantial_contribution": True, "dnsh_pass": True, "minimum_safeguards": True,
        },
        {
            "counterparty_id": "NFC-003", "name": "PetroChemEurope NV",
            "type": "NON_FINANCIAL_CORPORATES", "nace_code": "B06.10",
            "exposure_eur": 15_000_000.0, "taxonomy_eligible": False,
            "taxonomy_aligned_turnover_pct": 0.0, "taxonomy_aligned_capex_pct": 2.0,
            "taxonomy_aligned_opex_pct": 0.0, "environmental_objective": None,
            "substantial_contribution": False, "dnsh_pass": False, "minimum_safeguards": True,
        },
        {
            "counterparty_id": "NFC-004", "name": "GreenTech Solar BV",
            "type": "NON_FINANCIAL_CORPORATES", "nace_code": "D35.11",
            "exposure_eur": 40_000_000.0, "taxonomy_eligible": True,
            "taxonomy_aligned_turnover_pct": 98.0, "taxonomy_aligned_capex_pct": 99.0,
            "taxonomy_aligned_opex_pct": 95.0, "environmental_objective": "CLIMATE_MITIGATION",
            "substantial_contribution": True, "dnsh_pass": True, "minimum_safeguards": True,
        },
    ]


# ---------------------------------------------------------------------------
# Climate risk sample data
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_climate_risk_exposures() -> List[Dict[str, Any]]:
    """Create sample exposure data for climate risk scoring.

    Returns exposure records with sector, geography, and collateral
    data for physical and transition risk assessment.
    """
    return [
        {
            "exposure_id": "EXP-001", "counterparty": "EuroSteel GmbH",
            "nace_code": "C24.10", "country": "DE", "region": "Bavaria",
            "exposure_eur": 50_000_000.0, "collateral_type": "INDUSTRIAL",
            "latitude": 48.1351, "longitude": 11.5820,
            "carbon_intensity_tco2e_per_eur_m": 900.0,
        },
        {
            "exposure_id": "EXP-002", "counterparty": "Mediterranean Cement SA",
            "nace_code": "C23.51", "country": "ES", "region": "Andalusia",
            "exposure_eur": 25_000_000.0, "collateral_type": "INDUSTRIAL",
            "latitude": 36.7213, "longitude": -4.4217,
            "carbon_intensity_tco2e_per_eur_m": 1_700.0,
        },
        {
            "exposure_id": "EXP-003", "counterparty": "EuroProperty REIT",
            "nace_code": "L68.20", "country": "NL", "region": "South Holland",
            "exposure_eur": 80_000_000.0, "collateral_type": "COMMERCIAL_REAL_ESTATE",
            "latitude": 52.0705, "longitude": 4.3007,
            "carbon_intensity_tco2e_per_eur_m": 25.0,
        },
    ]
