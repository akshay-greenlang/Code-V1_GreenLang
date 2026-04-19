# -*- coding: utf-8 -*-
"""
PACK-011 SFDR Article 9 Pack - Shared Test Fixtures
=====================================================

Provides reusable pytest fixtures for all PACK-011 test modules including
sustainable objective verification, enhanced DNSH assessment, full taxonomy
alignment, impact measurement, benchmark alignment (CTB/PAB), mandatory PAI
indicator calculation, carbon trajectory tracking, investment universe
management, and all SFDR Article 9 disclosure workflows.

Article 9 Key Characteristics ("Dark Green"):
- Sustainable investment as product OBJECTIVE (100% SI target, 95% minimum)
- Mandatory PAI consideration (all 18 indicators, Scope 3 included)
- Enhanced DNSH applied to ALL holdings (not just sustainable portion)
- Annex III (pre-contractual) / Annex V (periodic) -- not Annex II / IV
- Benchmark designation mandatory for Art. 9(2) and Art. 9(3)
- Carbon trajectory monitoring for Art. 9(3) products
- Impact measurement and SDG mapping as core requirement
- Downgrade risk monitoring (Article 9 -> Article 8 reclassification)

Regulatory Context:
- SFDR Level 1: Regulation (EU) 2019/2088, Article 9 and Article 2(17)
- SFDR RTS: Delegated Regulation (EU) 2022/1288
- Taxonomy Disclosures DA: Delegated Regulation (EU) 2021/2178
- EU Climate Benchmarks: Regulation (EU) 2019/2089 (CTB/PAB)
- ESMA Q&A: SFDR and Taxonomy-related Q&A updates
- EET Standard: European ESG Template v1.1 (FinDatEx)

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
# Constants - SFDR classifications and Article 9 specifics
# ---------------------------------------------------------------------------

SFDR_CLASSIFICATIONS = ["ARTICLE_6", "ARTICLE_8", "ARTICLE_8_PLUS", "ARTICLE_9"]

ARTICLE9_SUB_TYPES = ["GENERAL_9_1", "INDEX_BASED_9_2", "CARBON_REDUCTION_9_3"]

BENCHMARK_TYPES = ["CTB", "PAB", "CUSTOM"]

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
    "sustainable_objective_engine",
    "enhanced_dnsh_engine",
    "full_taxonomy_alignment",
    "impact_measurement_engine",
    "benchmark_alignment_engine",
    "pai_mandatory_engine",
    "carbon_trajectory_engine",
    "investment_universe_engine",
]

SFDR_WORKFLOW_IDS = [
    "annex_iii_disclosure",
    "annex_v_reporting",
    "sustainable_verification",
    "impact_reporting",
    "benchmark_monitoring",
    "pai_mandatory_workflow",
    "downgrade_monitoring",
    "regulatory_update",
]

SFDR_TEMPLATE_IDS = [
    "annex_iii_precontractual",
    "annex_v_periodic",
    "impact_report",
    "benchmark_methodology",
    "sustainable_dashboard",
    "pai_mandatory_report",
    "carbon_trajectory_report",
    "audit_trail_report",
]

SFDR_INTEGRATION_IDS = [
    "pack_orchestrator",
    "article8_pack_bridge",
    "taxonomy_pack_bridge",
    "mrv_emissions_bridge",
    "benchmark_data_bridge",
    "impact_data_bridge",
    "eet_data_bridge",
    "regulatory_bridge",
    "health_check",
    "setup_wizard",
]

SFDR_PRODUCT_PRESET_IDS = [
    "impact_fund",
    "climate_fund",
    "social_fund",
    "esg_leader_fund",
    "transition_fund",
]

SFDR_ENTITY_PRESET_IDS = [
    "asset_manager",
    "insurance",
    "bank",
    "pension_fund",
    "wealth_manager",
]

# Combined: all 10 preset YAML files (5 product + 5 entity presets on disk)
SFDR_ALL_PRESET_FILES = SFDR_PRODUCT_PRESET_IDS + SFDR_ENTITY_PRESET_IDS


# ---------------------------------------------------------------------------
# File mapping dictionaries
# ---------------------------------------------------------------------------

ENGINE_FILES = {
    "sustainable_objective_engine": "sustainable_objective_engine.py",
    "enhanced_dnsh_engine": "enhanced_dnsh_engine.py",
    "full_taxonomy_alignment": "full_taxonomy_alignment.py",
    "impact_measurement_engine": "impact_measurement_engine.py",
    "benchmark_alignment_engine": "benchmark_alignment_engine.py",
    "pai_mandatory_engine": "pai_mandatory_engine.py",
    "carbon_trajectory_engine": "carbon_trajectory_engine.py",
    "investment_universe_engine": "investment_universe_engine.py",
}

WORKFLOW_FILES = {
    "annex_iii_disclosure": "annex_iii_disclosure.py",
    "annex_v_reporting": "annex_v_reporting.py",
    "sustainable_verification": "sustainable_verification.py",
    "impact_reporting": "impact_reporting.py",
    "benchmark_monitoring": "benchmark_monitoring.py",
    "pai_mandatory_workflow": "pai_mandatory_workflow.py",
    "downgrade_monitoring": "downgrade_monitoring.py",
    "regulatory_update": "regulatory_update.py",
}

TEMPLATE_FILES = {
    "annex_iii_precontractual": "annex_iii_precontractual.py",
    "annex_v_periodic": "annex_v_periodic.py",
    "impact_report": "impact_report.py",
    "benchmark_methodology": "benchmark_methodology.py",
    "sustainable_dashboard": "sustainable_dashboard.py",
    "pai_mandatory_report": "pai_mandatory_report.py",
    "carbon_trajectory_report": "carbon_trajectory_report.py",
    "audit_trail_report": "audit_trail_report.py",
}

INTEGRATION_FILES = {
    "pack_orchestrator": "pack_orchestrator.py",
    "article8_pack_bridge": "article8_pack_bridge.py",
    "taxonomy_pack_bridge": "taxonomy_pack_bridge.py",
    "mrv_emissions_bridge": "mrv_emissions_bridge.py",
    "benchmark_data_bridge": "benchmark_data_bridge.py",
    "impact_data_bridge": "impact_data_bridge.py",
    "eet_data_bridge": "eet_data_bridge.py",
    "regulatory_bridge": "regulatory_bridge.py",
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
    """Return the absolute path to the PACK-011 root directory."""
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
        "sfdr9_pack_config",
        CONFIG_DIR / "pack_config.py",
    )


@pytest.fixture
def sfdr_config(pack_config_module):
    """Create a default SFDRArticle9Config instance.

    Returns an SFDRArticle9Config with Article 9 defaults: ARTICLE_9
    classification, 95% sustainable investment minimum, all 18 PAI
    indicators enabled with mandatory consideration, strict DNSH mode,
    and impact measurement enabled.
    """
    return pack_config_module.SFDRArticle9Config()


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
            f"sfdr9_engine_{engine_id}",
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
            f"sfdr9_workflow_{workflow_id}",
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
            f"sfdr9_template_{template_id}",
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
            f"sfdr9_integration_{integration_id}",
            INTEGRATIONS_DIR / filename,
        )
    return modules


# ---------------------------------------------------------------------------
# Sample data fixtures - Portfolio holdings for Article 9 sustainable funds
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_portfolio_data() -> List[Dict[str, Any]]:
    """Create sample portfolio holdings data for SFDR Article 9 testing.

    Returns a list of 10 holding dictionaries representing an Article 9
    sustainable investment fund.  All holdings (except cash) are selected as
    sustainable investments with high ESG ratings, low carbon intensity, and
    positive impact characteristics -- consistent with a dark-green profile.
    """
    return [
        {
            "isin": "DK0060534915",
            "name": "Novo Nordisk A/S",
            "weight": 14.0,
            "sector": "Health Care",
            "country": "DK",
            "market_value": 1_400_000.0,
            "esg_rating": "AAA",
            "scope_1_emissions": 150.0,
            "scope_2_emissions": 200.0,
            "scope_3_emissions": 3_000.0,
            "revenue": 25_000_000.0,
            "enterprise_value": 400_000_000.0,
            "sustainable_investment": True,
            "si_type": "SOCIAL",
            "sdg_alignment": [3],
            "taxonomy_aligned_pct": 15.0,
        },
        {
            "isin": "DK0061539921",
            "name": "Vestas Wind Systems A/S",
            "weight": 12.0,
            "sector": "Industrials",
            "country": "DK",
            "market_value": 1_200_000.0,
            "esg_rating": "AAA",
            "scope_1_emissions": 80.0,
            "scope_2_emissions": 120.0,
            "scope_3_emissions": 2_500.0,
            "revenue": 15_000_000.0,
            "enterprise_value": 25_000_000.0,
            "sustainable_investment": True,
            "si_type": "TAXONOMY_ALIGNED",
            "sdg_alignment": [7, 13],
            "taxonomy_aligned_pct": 95.0,
        },
        {
            "isin": "ES0144580Y14",
            "name": "Iberdrola SA",
            "weight": 11.0,
            "sector": "Utilities",
            "country": "ES",
            "market_value": 1_100_000.0,
            "esg_rating": "AA",
            "scope_1_emissions": 18_000.0,
            "scope_2_emissions": 500.0,
            "scope_3_emissions": 8_000.0,
            "revenue": 40_000_000.0,
            "enterprise_value": 90_000_000.0,
            "sustainable_investment": True,
            "si_type": "TAXONOMY_ALIGNED",
            "sdg_alignment": [7, 13],
            "taxonomy_aligned_pct": 72.0,
        },
        {
            "isin": "FR0010613471",
            "name": "Schneider Electric SE",
            "weight": 10.0,
            "sector": "Industrials",
            "country": "FR",
            "market_value": 1_000_000.0,
            "esg_rating": "AAA",
            "scope_1_emissions": 200.0,
            "scope_2_emissions": 300.0,
            "scope_3_emissions": 5_000.0,
            "revenue": 30_000_000.0,
            "enterprise_value": 80_000_000.0,
            "sustainable_investment": True,
            "si_type": "OTHER_ENVIRONMENTAL",
            "sdg_alignment": [9, 13],
            "taxonomy_aligned_pct": 48.0,
        },
        {
            "isin": "SE0000115446",
            "name": "Nibe Industrier AB",
            "weight": 9.0,
            "sector": "Industrials",
            "country": "SE",
            "market_value": 900_000.0,
            "esg_rating": "AA",
            "scope_1_emissions": 40.0,
            "scope_2_emissions": 90.0,
            "scope_3_emissions": 1_200.0,
            "revenue": 4_000_000.0,
            "enterprise_value": 12_000_000.0,
            "sustainable_investment": True,
            "si_type": "TAXONOMY_ALIGNED",
            "sdg_alignment": [7, 11],
            "taxonomy_aligned_pct": 88.0,
        },
        {
            "isin": "NL0015000IQ2",
            "name": "Alfen NV",
            "weight": 8.5,
            "sector": "Industrials",
            "country": "NL",
            "market_value": 850_000.0,
            "esg_rating": "A",
            "scope_1_emissions": 10.0,
            "scope_2_emissions": 25.0,
            "scope_3_emissions": 500.0,
            "revenue": 500_000.0,
            "enterprise_value": 2_000_000.0,
            "sustainable_investment": True,
            "si_type": "TAXONOMY_ALIGNED",
            "sdg_alignment": [7, 9],
            "taxonomy_aligned_pct": 92.0,
        },
        {
            "isin": "DE000A0D6554",
            "name": "Nordex SE",
            "weight": 8.0,
            "sector": "Industrials",
            "country": "DE",
            "market_value": 800_000.0,
            "esg_rating": "A",
            "scope_1_emissions": 50.0,
            "scope_2_emissions": 60.0,
            "scope_3_emissions": 1_800.0,
            "revenue": 6_000_000.0,
            "enterprise_value": 8_000_000.0,
            "sustainable_investment": True,
            "si_type": "TAXONOMY_ALIGNED",
            "sdg_alignment": [7, 13],
            "taxonomy_aligned_pct": 91.0,
        },
        {
            "isin": "FI0009000681",
            "name": "Nokia Oyj",
            "weight": 7.5,
            "sector": "Technology",
            "country": "FI",
            "market_value": 750_000.0,
            "esg_rating": "AA",
            "scope_1_emissions": 30.0,
            "scope_2_emissions": 180.0,
            "scope_3_emissions": 2_000.0,
            "revenue": 23_000_000.0,
            "enterprise_value": 25_000_000.0,
            "sustainable_investment": True,
            "si_type": "OTHER_ENVIRONMENTAL",
            "sdg_alignment": [9],
            "taxonomy_aligned_pct": 12.0,
        },
        {
            "isin": "BE0974293251",
            "name": "Anheuser-Busch InBev SA",
            "weight": 7.0,
            "sector": "Consumer Staples",
            "country": "BE",
            "market_value": 700_000.0,
            "esg_rating": "A",
            "scope_1_emissions": 5_200.0,
            "scope_2_emissions": 1_800.0,
            "scope_3_emissions": 35_000.0,
            "revenue": 57_000_000.0,
            "enterprise_value": 180_000_000.0,
            "sustainable_investment": True,
            "si_type": "SOCIAL",
            "sdg_alignment": [6, 12],
            "taxonomy_aligned_pct": 5.0,
        },
        {
            "isin": "CASH_EUR_001",
            "name": "Cash & Equivalents (EUR)",
            "weight": 3.0,
            "sector": "Cash",
            "country": "EU",
            "market_value": 300_000.0,
            "esg_rating": "N/A",
            "scope_1_emissions": 0.0,
            "scope_2_emissions": 0.0,
            "scope_3_emissions": 0.0,
            "revenue": 0.0,
            "enterprise_value": 0.0,
            "sustainable_investment": False,
            "si_type": "CASH_HEDGING",
            "sdg_alignment": [],
            "taxonomy_aligned_pct": 0.0,
        },
    ]


# ---------------------------------------------------------------------------
# Sample data fixtures - PAI indicator values (all 18 mandatory)
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_pai_data() -> Dict[str, Any]:
    """Create sample data for all 18 mandatory PAI indicators.

    Returns a dictionary keyed by PAI indicator ID (1-18) with sample
    values, units, data quality flags, and coverage percentages.
    Article 9 products require mandatory Scope 3 and higher coverage.
    """
    return {
        1: {"name": "GHG Emissions", "value": 85_000.0, "unit": "tCO2e",
            "data_quality": "REPORTED", "coverage_pct": 92.0},
        2: {"name": "Carbon Footprint", "value": 95.2, "unit": "tCO2e/EUR million invested",
            "data_quality": "REPORTED", "coverage_pct": 90.0},
        3: {"name": "GHG Intensity", "value": 125.8, "unit": "tCO2e/EUR million revenue",
            "data_quality": "REPORTED", "coverage_pct": 88.0},
        4: {"name": "Fossil Fuel Exposure", "value": 2.5, "unit": "%",
            "data_quality": "REPORTED", "coverage_pct": 95.0},
        5: {"name": "Non-Renewable Energy Share", "value": 28.3, "unit": "%",
            "data_quality": "REPORTED", "coverage_pct": 85.0},
        6: {"name": "Energy Consumption Intensity", "value": 0.22,
            "unit": "GWh/EUR million revenue", "data_quality": "ESTIMATED",
            "coverage_pct": 78.0},
        7: {"name": "Biodiversity-Sensitive Areas", "value": 1.0, "unit": "%",
            "data_quality": "REPORTED", "coverage_pct": 82.0},
        8: {"name": "Emissions to Water", "value": 120.0, "unit": "tonnes",
            "data_quality": "ESTIMATED", "coverage_pct": 72.0},
        9: {"name": "Hazardous Waste Ratio", "value": 350.0, "unit": "tonnes",
            "data_quality": "REPORTED", "coverage_pct": 80.0},
        10: {"name": "UNGC/OECD Violations", "value": 0.0, "unit": "%",
             "data_quality": "REPORTED", "coverage_pct": 98.0},
        11: {"name": "UNGC/OECD Compliance Processes", "value": 1.0, "unit": "%",
             "data_quality": "REPORTED", "coverage_pct": 95.0},
        12: {"name": "Unadjusted Gender Pay Gap", "value": 8.5, "unit": "%",
             "data_quality": "REPORTED", "coverage_pct": 82.0},
        13: {"name": "Board Gender Diversity", "value": 42.0, "unit": "%",
             "data_quality": "REPORTED", "coverage_pct": 98.0},
        14: {"name": "Controversial Weapons Exposure", "value": 0.0, "unit": "%",
             "data_quality": "REPORTED", "coverage_pct": 100.0},
        15: {"name": "Sovereign GHG Intensity", "value": 180.0,
             "unit": "tCO2e/EUR million GDP", "data_quality": "REPORTED",
             "coverage_pct": 100.0},
        16: {"name": "Investee Countries Social Violations", "value": 0, "unit": "count",
             "data_quality": "REPORTED", "coverage_pct": 100.0},
        17: {"name": "Fossil Fuel Exposure Real Estate", "value": 5.0, "unit": "%",
             "data_quality": "ESTIMATED", "coverage_pct": 60.0},
        18: {"name": "Energy Inefficient Real Estate", "value": 10.0, "unit": "%",
             "data_quality": "ESTIMATED", "coverage_pct": 55.0},
    }


# ---------------------------------------------------------------------------
# Sample data fixtures - Governance assessments
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_governance_data() -> List[Dict[str, Any]]:
    """Create sample governance assessment data for Article 9 holdings.

    Article 9 products have stricter governance thresholds (70 minimum score,
    controversy flag threshold of 2, and all four dimensions must pass).
    All holdings in this fixture pass governance for a dark-green portfolio.
    """
    return [
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
            "company_name": "Vestas Wind Systems A/S",
            "isin": "DK0061539921",
            "sound_management_score": 88.0,
            "employee_relations_score": 82.0,
            "remuneration_score": 80.0,
            "tax_compliance_score": 90.0,
            "overall_status": "PASS",
            "controversy_count": 0,
            "assessment_date": "2025-09-15",
        },
        {
            "company_name": "Iberdrola SA",
            "isin": "ES0144580Y14",
            "sound_management_score": 85.0,
            "employee_relations_score": 78.0,
            "remuneration_score": 75.0,
            "tax_compliance_score": 82.0,
            "overall_status": "PASS",
            "controversy_count": 1,
            "assessment_date": "2025-09-15",
        },
        {
            "company_name": "Schneider Electric SE",
            "isin": "FR0010613471",
            "sound_management_score": 90.0,
            "employee_relations_score": 86.0,
            "remuneration_score": 82.0,
            "tax_compliance_score": 88.0,
            "overall_status": "PASS",
            "controversy_count": 0,
            "assessment_date": "2025-09-15",
        },
        {
            "company_name": "Nibe Industrier AB",
            "isin": "SE0000115446",
            "sound_management_score": 82.0,
            "employee_relations_score": 80.0,
            "remuneration_score": 78.0,
            "tax_compliance_score": 85.0,
            "overall_status": "PASS",
            "controversy_count": 0,
            "assessment_date": "2025-09-15",
        },
    ]


# ---------------------------------------------------------------------------
# Sample data fixtures - Impact measurement data
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_impact_data() -> Dict[str, Any]:
    """Create sample impact measurement data for an Article 9 climate fund.

    Includes impact KPIs, SDG alignment, and theory-of-change metrics
    that are unique to Article 9 products.
    """
    return {
        "product_name": "GreenLang Climate Impact Fund",
        "impact_objective": "CLIMATE_MITIGATION",
        "sdg_targets": [7, 13],
        "kpis": [
            {
                "id": "tonnes_co2_avoided",
                "name": "Tonnes CO2e Avoided",
                "current_value": 45_000.0,
                "target_value": 60_000.0,
                "unit": "tCO2e",
                "attainment_pct": 75.0,
            },
            {
                "id": "renewable_energy_generated_mwh",
                "name": "Renewable Energy Generated",
                "current_value": 120_000.0,
                "target_value": 150_000.0,
                "unit": "MWh",
                "attainment_pct": 80.0,
            },
            {
                "id": "green_revenue_share_pct",
                "name": "Green Revenue Share",
                "current_value": 68.0,
                "target_value": 75.0,
                "unit": "%",
                "attainment_pct": 90.7,
            },
        ],
        "theory_of_change": {
            "input": "Capital deployed to climate solutions companies",
            "activities": "Financing renewable energy, energy efficiency, green technology",
            "output": "Increased clean energy capacity, reduced emissions",
            "outcome": "Accelerated climate transition in portfolio companies",
            "impact": "Contribution to Paris Agreement temperature goals",
        },
    }


# ---------------------------------------------------------------------------
# Sample data fixtures - Benchmark alignment data
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_benchmark_data() -> Dict[str, Any]:
    """Create sample benchmark alignment data for CTB/PAB tracking."""
    return {
        "benchmark_type": "PAB",
        "benchmark_name": "MSCI Europe PAB Index",
        "benchmark_provider": "MSCI",
        "portfolio_carbon_intensity": 85.2,
        "benchmark_carbon_intensity": 92.5,
        "tracking_error_pct": 1.8,
        "yoy_decarbonization_actual_pct": 8.2,
        "yoy_decarbonization_target_pct": 7.0,
        "baseline_year": 2019,
        "baseline_carbon_intensity": 180.0,
        "sector_exclusions_compliant": True,
    }


# ---------------------------------------------------------------------------
# Temporary output directory
# ---------------------------------------------------------------------------

@pytest.fixture
def temp_output_dir(tmp_path) -> Path:
    """Create a temporary directory for test outputs."""
    output_dir = tmp_path / "sfdr_article9_output"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir
