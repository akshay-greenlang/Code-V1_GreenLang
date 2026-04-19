# -*- coding: utf-8 -*-
"""
PACK-010 SFDR Article 8 Pack - Shared Test Fixtures
=====================================================

Provides reusable pytest fixtures for all PACK-010 test modules including
PAI indicator calculation, taxonomy alignment ratio computation, SFDR DNSH
assessment, good governance verification, E/S characteristics tracking,
sustainable investment classification, portfolio carbon footprint
calculation, EET data management, and all SFDR disclosure workflows.

Regulatory Context:
- SFDR Level 1: Regulation (EU) 2019/2088
- SFDR RTS: Delegated Regulation (EU) 2022/1288
- Taxonomy Disclosures DA: Delegated Regulation (EU) 2021/2178
- ESMA Q&A: SFDR and Taxonomy-related Q&A updates
- EET Standard: European ESG Template v1.1 (FinDatEx)

SFDR Product Classifications:
- Article 6: No sustainability characteristics promoted
- Article 8: Promotes environmental or social characteristics
- Article 8+: Article 8 with sustainable investment proportion
- Article 9: Sustainable investment as objective

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
# Constants - SFDR classifications and indicator categories
# ---------------------------------------------------------------------------

SFDR_CLASSIFICATIONS = ["ARTICLE_6", "ARTICLE_8", "ARTICLE_8_PLUS", "ARTICLE_9"]

PAI_INDICATORS = [
    "PAI_1", "PAI_2", "PAI_3", "PAI_4", "PAI_5", "PAI_6",
    "PAI_7", "PAI_8", "PAI_9", "PAI_10", "PAI_11", "PAI_12",
    "PAI_13", "PAI_14", "PAI_15", "PAI_16", "PAI_17", "PAI_18",
]

DISCLOSURE_TYPES = ["PRE_CONTRACTUAL", "PERIODIC", "WEBSITE"]


# ---------------------------------------------------------------------------
# Component ID lists
# ---------------------------------------------------------------------------

SFDR_ENGINE_IDS = [
    "pai_indicator_calculator",
    "taxonomy_alignment_ratio",
    "sfdr_dnsh_engine",
    "good_governance_engine",
    "esg_characteristics_engine",
    "sustainable_investment_calculator",
    "portfolio_carbon_footprint",
    "eet_data_engine",
]

SFDR_WORKFLOW_IDS = [
    "precontractual_disclosure",
    "periodic_reporting",
    "website_disclosure",
    "pai_statement",
    "portfolio_screening",
    "taxonomy_alignment",
    "compliance_review",
    "regulatory_update",
]

SFDR_TEMPLATE_IDS = [
    "annex_ii_precontractual",
    "annex_iv_periodic",
    "annex_iii_website",
    "pai_statement_template",
    "portfolio_esg_dashboard",
    "taxonomy_alignment_report",
    "executive_summary",
    "audit_trail_report",
]

SFDR_INTEGRATION_IDS = [
    "pack_orchestrator",
    "taxonomy_pack_bridge",
    "mrv_emissions_bridge",
    "investment_screener_bridge",
    "portfolio_data_bridge",
    "eet_data_bridge",
    "regulatory_tracking_bridge",
    "data_quality_bridge",
    "health_check",
    "setup_wizard",
]

SFDR_PRESET_IDS = [
    "asset_manager",
    "insurance",
    "bank",
    "pension_fund",
    "wealth_manager",
]


# ---------------------------------------------------------------------------
# File mapping dictionaries
# ---------------------------------------------------------------------------

ENGINE_FILES = {
    "pai_indicator_calculator": "pai_indicator_calculator.py",
    "taxonomy_alignment_ratio": "taxonomy_alignment_ratio.py",
    "sfdr_dnsh_engine": "sfdr_dnsh_engine.py",
    "good_governance_engine": "good_governance_engine.py",
    "esg_characteristics_engine": "esg_characteristics_engine.py",
    "sustainable_investment_calculator": "sustainable_investment_calculator.py",
    "portfolio_carbon_footprint": "portfolio_carbon_footprint.py",
    "eet_data_engine": "eet_data_engine.py",
}

WORKFLOW_FILES = {
    "precontractual_disclosure": "precontractual_disclosure.py",
    "periodic_reporting": "periodic_reporting.py",
    "website_disclosure": "website_disclosure.py",
    "pai_statement": "pai_statement.py",
    "portfolio_screening": "portfolio_screening.py",
    "taxonomy_alignment": "taxonomy_alignment.py",
    "compliance_review": "compliance_review.py",
    "regulatory_update": "regulatory_update.py",
}

TEMPLATE_FILES = {
    "annex_ii_precontractual": "annex_ii_precontractual.py",
    "annex_iv_periodic": "annex_iv_periodic.py",
    "annex_iii_website": "annex_iii_website.py",
    "pai_statement_template": "pai_statement_template.py",
    "portfolio_esg_dashboard": "portfolio_esg_dashboard.py",
    "taxonomy_alignment_report": "taxonomy_alignment_report.py",
    "executive_summary": "executive_summary.py",
    "audit_trail_report": "audit_trail_report.py",
}

INTEGRATION_FILES = {
    "pack_orchestrator": "pack_orchestrator.py",
    "taxonomy_pack_bridge": "taxonomy_pack_bridge.py",
    "mrv_emissions_bridge": "mrv_emissions_bridge.py",
    "investment_screener_bridge": "investment_screener_bridge.py",
    "portfolio_data_bridge": "portfolio_data_bridge.py",
    "eet_data_bridge": "eet_data_bridge.py",
    "regulatory_tracking_bridge": "regulatory_tracking_bridge.py",
    "data_quality_bridge": "data_quality_bridge.py",
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
    """Return the absolute path to the PACK-010 root directory."""
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
        "sfdr_pack_config",
        CONFIG_DIR / "pack_config.py",
    )


@pytest.fixture
def sfdr_config(pack_config_module):
    """Create a default SFDRArticle8Config instance.

    Returns an SFDRArticle8Config with default values (Article 8 classification,
    all 18 PAI indicators enabled, all sub-configurations at defaults).
    """
    return pack_config_module.SFDRArticle8Config()


@pytest.fixture
def pack_config_instance(pack_config_module):
    """Create a default PackConfig instance loaded from defaults."""
    return pack_config_module.get_default_config()


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
            f"sfdr_engine_{engine_id}",
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
            f"sfdr_workflow_{workflow_id}",
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
            f"sfdr_template_{template_id}",
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
            f"sfdr_integration_{integration_id}",
            INTEGRATIONS_DIR / filename,
        )
    return modules


# ---------------------------------------------------------------------------
# Sample data fixtures - Portfolio holdings
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_portfolio_data() -> List[Dict[str, Any]]:
    """Create sample portfolio holdings data for SFDR Article 8 testing.

    Returns a list of 10 holding dictionaries with ISIN, name, weight,
    sector, country, market_value, esg_rating, scope 1/2/3 emissions,
    revenue, and enterprise_value fields.
    """
    return [
        {
            "isin": "DE0007236101",
            "name": "Siemens AG",
            "weight": 12.5,
            "sector": "Industrials",
            "country": "DE",
            "market_value": 1_250_000.0,
            "esg_rating": "AA",
            "scope_1_emissions": 1_200.0,
            "scope_2_emissions": 850.0,
            "scope_3_emissions": 15_000.0,
            "revenue": 62_000_000.0,
            "enterprise_value": 120_000_000.0,
        },
        {
            "isin": "FR0000120271",
            "name": "TotalEnergies SE",
            "weight": 8.0,
            "sector": "Energy",
            "country": "FR",
            "market_value": 800_000.0,
            "esg_rating": "BBB",
            "scope_1_emissions": 35_000.0,
            "scope_2_emissions": 5_000.0,
            "scope_3_emissions": 300_000.0,
            "revenue": 200_000_000.0,
            "enterprise_value": 150_000_000.0,
        },
        {
            "isin": "NL0010773842",
            "name": "ING Groep NV",
            "weight": 10.0,
            "sector": "Financials",
            "country": "NL",
            "market_value": 1_000_000.0,
            "esg_rating": "A",
            "scope_1_emissions": 50.0,
            "scope_2_emissions": 120.0,
            "scope_3_emissions": 2_500.0,
            "revenue": 18_000_000.0,
            "enterprise_value": 55_000_000.0,
        },
        {
            "isin": "DK0060534915",
            "name": "Novo Nordisk A/S",
            "weight": 15.0,
            "sector": "Health Care",
            "country": "DK",
            "market_value": 1_500_000.0,
            "esg_rating": "AAA",
            "scope_1_emissions": 150.0,
            "scope_2_emissions": 200.0,
            "scope_3_emissions": 3_000.0,
            "revenue": 25_000_000.0,
            "enterprise_value": 400_000_000.0,
        },
        {
            "isin": "ES0144580Y14",
            "name": "Iberdrola SA",
            "weight": 9.5,
            "sector": "Utilities",
            "country": "ES",
            "market_value": 950_000.0,
            "esg_rating": "AA",
            "scope_1_emissions": 18_000.0,
            "scope_2_emissions": 500.0,
            "scope_3_emissions": 8_000.0,
            "revenue": 40_000_000.0,
            "enterprise_value": 90_000_000.0,
        },
        {
            "isin": "SE0000108656",
            "name": "Ericsson AB",
            "weight": 7.0,
            "sector": "Technology",
            "country": "SE",
            "market_value": 700_000.0,
            "esg_rating": "A",
            "scope_1_emissions": 80.0,
            "scope_2_emissions": 350.0,
            "scope_3_emissions": 4_500.0,
            "revenue": 22_000_000.0,
            "enterprise_value": 30_000_000.0,
        },
        {
            "isin": "IT0003128367",
            "name": "Enel SpA",
            "weight": 11.0,
            "sector": "Utilities",
            "country": "IT",
            "market_value": 1_100_000.0,
            "esg_rating": "AA",
            "scope_1_emissions": 45_000.0,
            "scope_2_emissions": 2_000.0,
            "scope_3_emissions": 25_000.0,
            "revenue": 85_000_000.0,
            "enterprise_value": 100_000_000.0,
        },
        {
            "isin": "FI0009000681",
            "name": "Nokia Oyj",
            "weight": 6.0,
            "sector": "Technology",
            "country": "FI",
            "market_value": 600_000.0,
            "esg_rating": "A",
            "scope_1_emissions": 30.0,
            "scope_2_emissions": 180.0,
            "scope_3_emissions": 2_000.0,
            "revenue": 23_000_000.0,
            "enterprise_value": 25_000_000.0,
        },
        {
            "isin": "BE0003810273",
            "name": "Anheuser-Busch InBev SA",
            "weight": 8.5,
            "sector": "Consumer Staples",
            "country": "BE",
            "market_value": 850_000.0,
            "esg_rating": "BBB",
            "scope_1_emissions": 5_200.0,
            "scope_2_emissions": 1_800.0,
            "scope_3_emissions": 35_000.0,
            "revenue": 57_000_000.0,
            "enterprise_value": 180_000_000.0,
        },
        {
            "isin": "AT0000652011",
            "name": "Erste Group Bank AG",
            "weight": 12.5,
            "sector": "Financials",
            "country": "AT",
            "market_value": 1_250_000.0,
            "esg_rating": "A",
            "scope_1_emissions": 25.0,
            "scope_2_emissions": 95.0,
            "scope_3_emissions": 1_800.0,
            "revenue": 8_000_000.0,
            "enterprise_value": 20_000_000.0,
        },
    ]


# ---------------------------------------------------------------------------
# Sample data fixtures - PAI indicator values
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_pai_data() -> Dict[str, Any]:
    """Create sample data for all 18 mandatory PAI indicators.

    Returns a dictionary keyed by PAI indicator ID (1-18) with sample
    values, units, data quality flags, and coverage percentages.
    """
    return {
        1: {"name": "GHG Emissions", "value": 125_000.0, "unit": "tCO2e",
            "data_quality": "REPORTED", "coverage_pct": 85.0},
        2: {"name": "Carbon Footprint", "value": 180.5, "unit": "tCO2e/EUR million invested",
            "data_quality": "REPORTED", "coverage_pct": 82.0},
        3: {"name": "GHG Intensity", "value": 245.3, "unit": "tCO2e/EUR million revenue",
            "data_quality": "REPORTED", "coverage_pct": 80.0},
        4: {"name": "Fossil Fuel Exposure", "value": 8.5, "unit": "%",
            "data_quality": "REPORTED", "coverage_pct": 90.0},
        5: {"name": "Non-Renewable Energy Share", "value": 62.3, "unit": "%",
            "data_quality": "ESTIMATED", "coverage_pct": 70.0},
        6: {"name": "Energy Consumption Intensity", "value": 0.45,
            "unit": "GWh/EUR million revenue", "data_quality": "ESTIMATED",
            "coverage_pct": 65.0},
        7: {"name": "Biodiversity-Sensitive Areas", "value": 3.2, "unit": "%",
            "data_quality": "REPORTED", "coverage_pct": 75.0},
        8: {"name": "Emissions to Water", "value": 450.0, "unit": "tonnes",
            "data_quality": "ESTIMATED", "coverage_pct": 60.0},
        9: {"name": "Hazardous Waste Ratio", "value": 1_200.0, "unit": "tonnes",
            "data_quality": "REPORTED", "coverage_pct": 78.0},
        10: {"name": "UNGC/OECD Violations", "value": 2.5, "unit": "%",
             "data_quality": "REPORTED", "coverage_pct": 92.0},
        11: {"name": "UNGC/OECD Compliance Processes", "value": 5.0, "unit": "%",
             "data_quality": "REPORTED", "coverage_pct": 88.0},
        12: {"name": "Unadjusted Gender Pay Gap", "value": 12.8, "unit": "%",
             "data_quality": "REPORTED", "coverage_pct": 72.0},
        13: {"name": "Board Gender Diversity", "value": 35.2, "unit": "%",
             "data_quality": "REPORTED", "coverage_pct": 95.0},
        14: {"name": "Controversial Weapons Exposure", "value": 0.0, "unit": "%",
             "data_quality": "REPORTED", "coverage_pct": 98.0},
        15: {"name": "Sovereign GHG Intensity", "value": 320.0,
             "unit": "tCO2e/EUR million GDP", "data_quality": "REPORTED",
             "coverage_pct": 100.0},
        16: {"name": "Investee Countries Social Violations", "value": 0, "unit": "count",
             "data_quality": "REPORTED", "coverage_pct": 100.0},
        17: {"name": "Fossil Fuel Exposure Real Estate", "value": 15.0, "unit": "%",
             "data_quality": "ESTIMATED", "coverage_pct": 55.0},
        18: {"name": "Energy Inefficient Real Estate", "value": 22.0, "unit": "%",
             "data_quality": "ESTIMATED", "coverage_pct": 50.0},
    }


# ---------------------------------------------------------------------------
# Sample data fixtures - Governance assessments
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_governance_data() -> List[Dict[str, Any]]:
    """Create sample governance assessment data for 5 companies.

    Returns a list of governance dictionaries with company name, ISIN,
    four governance dimension scores, overall status, and controversy flags.
    """
    return [
        {
            "company_name": "Siemens AG",
            "isin": "DE0007236101",
            "sound_management_score": 85.0,
            "employee_relations_score": 78.0,
            "remuneration_score": 72.0,
            "tax_compliance_score": 90.0,
            "overall_status": "PASS",
            "controversy_count": 0,
            "assessment_date": "2025-09-15",
        },
        {
            "company_name": "TotalEnergies SE",
            "isin": "FR0000120271",
            "sound_management_score": 70.0,
            "employee_relations_score": 55.0,
            "remuneration_score": 65.0,
            "tax_compliance_score": 60.0,
            "overall_status": "PARTIAL",
            "controversy_count": 3,
            "assessment_date": "2025-09-15",
        },
        {
            "company_name": "Novo Nordisk A/S",
            "isin": "DK0060534915",
            "sound_management_score": 92.0,
            "employee_relations_score": 88.0,
            "remuneration_score": 85.0,
            "tax_compliance_score": 95.0,
            "overall_status": "PASS",
            "controversy_count": 0,
            "assessment_date": "2025-09-15",
        },
        {
            "company_name": "Enel SpA",
            "isin": "IT0003128367",
            "sound_management_score": 75.0,
            "employee_relations_score": 68.0,
            "remuneration_score": 70.0,
            "tax_compliance_score": 72.0,
            "overall_status": "PASS",
            "controversy_count": 1,
            "assessment_date": "2025-09-15",
        },
        {
            "company_name": "Anheuser-Busch InBev SA",
            "isin": "BE0003810273",
            "sound_management_score": 62.0,
            "employee_relations_score": 45.0,
            "remuneration_score": 58.0,
            "tax_compliance_score": 50.0,
            "overall_status": "FAIL",
            "controversy_count": 4,
            "assessment_date": "2025-09-15",
        },
    ]


# ---------------------------------------------------------------------------
# Temporary output directory
# ---------------------------------------------------------------------------

@pytest.fixture
def temp_output_dir(tmp_path) -> Path:
    """Create a temporary directory for test outputs."""
    output_dir = tmp_path / "sfdr_article8_output"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir
