# -*- coding: utf-8 -*-
"""
PACK-008 EU Taxonomy Alignment Pack - Shared Test Fixtures
============================================================

Provides reusable pytest fixtures for all PACK-008 test modules including
taxonomy eligibility screening, substantial contribution assessment, DNSH
evaluation, minimum safeguards verification, KPI calculation, GAR computation,
Article 8 disclosure generation, and cross-framework alignment.

Covers three organization types:
- Non-Financial Undertakings (Turnover/CapEx/OpEx KPIs)
- Financial Institutions (GAR/BTAR, EBA Pillar 3)
- Asset Managers (Fund-level taxonomy ratios, SFDR alignment)

Environmental Objectives:
    1. Climate Change Mitigation (CCM)
    2. Climate Change Adaptation (CCA)
    3. Sustainable Use and Protection of Water and Marine Resources (WTR)
    4. Transition to a Circular Economy (CE)
    5. Pollution Prevention and Control (PPC)
    6. Protection and Restoration of Biodiversity and Ecosystems (BIO)

Author: GreenLang QA Team
Version: 1.0.0
"""

import hashlib
import importlib.util
import json
import re
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
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load spec from {file_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

PACK_ROOT = Path(__file__).resolve().parent.parent
PACK_YAML_PATH = PACK_ROOT / "pack.yaml"
CONFIG_DIR = PACK_ROOT / "config"
PRESETS_DIR = CONFIG_DIR / "presets"
SECTORS_DIR = CONFIG_DIR / "sectors"
DEMO_DIR = CONFIG_DIR / "demo"
ENGINES_DIR = PACK_ROOT / "engines"
WORKFLOWS_DIR = PACK_ROOT / "workflows"
TEMPLATES_DIR = PACK_ROOT / "templates"
INTEGRATIONS_DIR = PACK_ROOT / "integrations"


# ---------------------------------------------------------------------------
# Constants - EU Taxonomy regulation parameters
# ---------------------------------------------------------------------------

ENVIRONMENTAL_OBJECTIVES = ["CCM", "CCA", "WTR", "CE", "PPC", "BIO"]

ENVIRONMENTAL_OBJECTIVE_NAMES = {
    "CCM": "Climate Change Mitigation",
    "CCA": "Climate Change Adaptation",
    "WTR": "Sustainable Use and Protection of Water and Marine Resources",
    "CE": "Transition to a Circular Economy",
    "PPC": "Pollution Prevention and Control",
    "BIO": "Protection and Restoration of Biodiversity and Ecosystems",
}

TAXONOMY_SECTORS = [
    "ENERGY",
    "MANUFACTURING",
    "REAL_ESTATE",
    "TRANSPORT",
    "FORESTRY_AGRICULTURE",
    "FINANCIAL_SERVICES",
    "WATER_SUPPLY",
    "WASTE_MANAGEMENT",
    "ICT",
    "PROFESSIONAL_SERVICES",
]

NACE_SAMPLE_CODES = [
    "A01.11",  # Agriculture - Growing of cereals
    "C20.11",  # Manufacturing - Manufacture of industrial gases
    "C24.10",  # Manufacturing - Manufacture of basic iron and steel
    "D35.11",  # Electricity generation
    "F41.10",  # Construction - Development of building projects
    "H49.10",  # Transport - Passenger rail transport
    "J62.01",  # ICT - Computer programming
    "K64.19",  # Financial services - Other monetary intermediation
    "L68.10",  # Real estate - Buying/selling own real estate
    "M71.12",  # Professional services - Engineering activities
]

NACE_SECTIONS = ["A", "B", "C", "D", "E", "F", "G", "H", "J", "K", "L", "M"]

SIZE_PRESETS = [
    "non_financial_undertaking",
    "financial_institution",
    "asset_manager",
    "large_enterprise",
    "sme_simplified",
]

SECTOR_PRESETS = [
    "energy",
    "manufacturing",
    "real_estate",
    "transport",
    "forestry_agriculture",
    "financial_services",
]

ALIGNMENT_STATUSES = [
    "NOT_SCREENED",
    "ELIGIBLE",
    "NOT_ELIGIBLE",
    "ALIGNED",
    "NOT_ALIGNED",
    "PARTIALLY_ALIGNED",
]

ORGANIZATION_TYPES = [
    "NON_FINANCIAL_UNDERTAKING",
    "FINANCIAL_INSTITUTION",
    "ASSET_MANAGER",
]

KPI_TYPES = ["TURNOVER", "CAPEX", "OPEX"]

DELEGATED_ACT_VERSIONS = [
    "CLIMATE_DA_2021",
    "ENVIRONMENTAL_DA_2023",
    "COMPLEMENTARY_DA_2022",
    "DISCLOSURES_DA_2021",
    "SIMPLIFICATION_DA_2025",
]

EXPOSURE_TYPES = [
    "CORPORATE_LOANS",
    "DEBT_SECURITIES",
    "EQUITY_HOLDINGS",
    "RESIDENTIAL_MORTGAGES",
    "COMMERCIAL_MORTGAGES",
    "PROJECT_FINANCE",
    "INTERBANK_LOANS",
    "SOVEREIGN_EXPOSURES",
]

TAXONOMY_REGULATION_REF = "(EU) 2020/852"
CLIMATE_DA_REF = "(EU) 2021/2139"
ENVIRONMENTAL_DA_REF = "(EU) 2023/2486"
DISCLOSURES_DA_REF = "(EU) 2021/2178"
COMPLEMENTARY_DA_REF = "(EU) 2022/1214"


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
def pack_yaml_data(pack_yaml_raw) -> Dict[str, Any]:
    """Return the parsed pack.yaml as a dictionary, or empty dict if missing."""
    if pack_yaml_raw:
        import yaml
        return yaml.safe_load(pack_yaml_raw) or {}
    return {}


# ---------------------------------------------------------------------------
# Configuration fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def pack_config():
    """Create a TaxonomyAlignmentConfig instance with defaults.

    Dynamically imports the config module from the hyphenated pack directory
    and creates a default TaxonomyAlignmentConfig.
    """
    config_module = _import_from_path(
        "pack_config",
        CONFIG_DIR / "pack_config.py",
    )
    return config_module.TaxonomyAlignmentConfig()


@pytest.fixture
def pack_config_module():
    """Return the pack_config module for direct class/enum access."""
    return _import_from_path(
        "pack_config",
        CONFIG_DIR / "pack_config.py",
    )


@pytest.fixture
def mock_config() -> Dict[str, Any]:
    """Return a configuration dictionary covering all PACK-008 config sections."""
    return {
        "pack_id": "PACK-008-eu-taxonomy-alignment",
        "pack_name": "EU Taxonomy Alignment Pack",
        "tier": "standalone",
        "version": "1.0.0",
        "organization_type": "NON_FINANCIAL_UNDERTAKING",
        "reporting_period": "ANNUAL",
        "reporting_year": 2025,
        "objectives_in_scope": ["CCM", "CCA", "WTR", "CE", "PPC", "BIO"],
        "eligibility": {
            "screening_mode": "HYBRID",
            "nace_version": "NACE_REV2",
            "min_confidence": 0.85,
            "batch_size": 100,
            "include_transitional": True,
            "include_enabling": True,
            "revenue_weighted": True,
            "auto_classify_nace": True,
        },
        "sc_assessment": {
            "evaluation_mode": "STANDARD",
            "threshold_strictness": "STANDARD",
            "evidence_required": True,
            "quantitative_tolerance_pct": 5.0,
            "track_enabling_activities": True,
            "track_transitional_activities": True,
            "require_all_quantitative": True,
            "gap_analysis_on_fail": True,
        },
        "dnsh": {
            "objectives_assessed": ["CCM", "CCA", "WTR", "CE", "PPC", "BIO"],
            "require_all_pass": True,
            "climate_risk_assessment_enabled": True,
            "water_framework_directive_check": True,
            "circular_economy_waste_hierarchy": True,
            "pollution_threshold_check": True,
            "biodiversity_impact_assessment": True,
            "evidence_required": True,
        },
        "minimum_safeguards": {
            "human_rights_check": True,
            "anti_corruption_check": True,
            "taxation_check": True,
            "fair_competition_check": True,
            "assessment_mode": "FULL",
            "require_all_pass": True,
            "grievance_mechanism_required": False,
            "supply_chain_due_diligence": False,
        },
        "kpi": {
            "calculate_turnover": True,
            "calculate_capex": True,
            "calculate_opex": True,
            "double_counting_prevention": True,
            "capex_plan_recognition": True,
            "capex_plan_max_years": 5,
            "eligible_vs_aligned_breakdown": True,
            "activity_level_detail": True,
            "currency": "EUR",
            "rounding_precision": 2,
        },
        "gar": {
            "calculate_stock_gar": True,
            "calculate_flow_gar": True,
            "calculate_btar": True,
            "exposure_types": [
                "CORPORATE_LOANS",
                "DEBT_SECURITIES",
                "EQUITY_HOLDINGS",
                "RESIDENTIAL_MORTGAGES",
                "PROJECT_FINANCE",
            ],
            "epc_integration": True,
            "epc_threshold_rating": "C",
            "de_minimis_threshold": 0.0,
            "counterparty_data_source": "DIRECT",
            "sovereign_exclusion": True,
            "interbank_exclusion": True,
        },
        "reporting": {
            "article8_enabled": True,
            "eba_pillar3_enabled": False,
            "xbrl_tagging": False,
            "nuclear_gas_supplementary": True,
            "yoy_comparison": True,
            "default_format": "PDF",
            "include_methodology_note": True,
            "include_audit_opinion": False,
            "language": "en",
            "timezone": "UTC",
            "cross_framework_targets": [],
        },
        "regulatory": {
            "delegated_act_version": "CLIMATE_DA_2021",
            "active_delegated_acts": [
                "CLIMATE_DA_2021",
                "ENVIRONMENTAL_DA_2023",
                "DISCLOSURES_DA_2021",
            ],
            "track_updates": True,
            "auto_migration": False,
            "update_check_interval_hours": 24,
            "include_complementary_da": True,
            "include_simplification_da": False,
        },
        "tsc": {
            "strict_threshold_compliance": True,
            "tolerance_margin_pct": 0.0,
            "track_criteria_changes": True,
            "gap_identification": True,
            "evidence_linking": True,
            "qualitative_assessment_enabled": True,
        },
        "transition_activity": {
            "enabled": True,
            "best_available_technology_check": True,
            "lock_in_avoidance_check": True,
            "sunset_date_tracking": True,
            "transition_pathway_documentation": True,
        },
        "enabling_activity": {
            "enabled": True,
            "direct_enablement_verification": True,
            "lifecycle_consideration": True,
            "technology_lock_in_check": True,
            "market_distortion_assessment": False,
        },
        "data_quality": {
            "min_quality_score": 0.80,
            "completeness_threshold": 0.90,
            "require_primary_data": False,
            "allow_estimates": True,
            "validation_on_ingestion": True,
        },
        "audit_trail": {
            "retention_years": 5,
            "hash_algorithm": "SHA-256",
            "export_formats": ["JSON", "XML", "PDF"],
            "immutable_log": True,
            "include_provenance_hash": True,
        },
        "demo": {
            "demo_mode_enabled": False,
            "use_synthetic_data": False,
            "mock_erp_responses": False,
            "mock_mrv_data": False,
            "tutorial_mode_enabled": False,
            "sample_activities_count": 10,
        },
    }


# ---------------------------------------------------------------------------
# Sample economic activity fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_activities() -> List[Dict[str, Any]]:
    """Create 5 sample economic activities with NACE codes for testing."""
    return [
        {
            "activity_id": str(uuid.uuid4()),
            "name": "Electricity generation using solar photovoltaic technology",
            "nace_code": "D35.11",
            "taxonomy_id": "CCM-4.1",
            "sector": "ENERGY",
            "description": "Generation of electricity from solar PV installations",
            "eligible_objectives": ["CCM"],
            "is_enabling": True,
            "is_transitional": False,
            "revenue_eur": 15_000_000,
        },
        {
            "activity_id": str(uuid.uuid4()),
            "name": "Manufacture of cement",
            "nace_code": "C23.51",
            "taxonomy_id": "CCM-3.7",
            "sector": "MANUFACTURING",
            "description": "Manufacture of cement clinker and grey/white cement",
            "eligible_objectives": ["CCM"],
            "is_enabling": False,
            "is_transitional": True,
            "revenue_eur": 50_000_000,
        },
        {
            "activity_id": str(uuid.uuid4()),
            "name": "Renovation of existing buildings",
            "nace_code": "F41.20",
            "taxonomy_id": "CCM-7.2",
            "sector": "REAL_ESTATE",
            "description": "Renovation of existing buildings to improve energy performance",
            "eligible_objectives": ["CCM", "CCA", "CE"],
            "is_enabling": False,
            "is_transitional": False,
            "revenue_eur": 8_000_000,
        },
        {
            "activity_id": str(uuid.uuid4()),
            "name": "Passenger rail transport",
            "nace_code": "H49.10",
            "taxonomy_id": "CCM-6.1",
            "sector": "TRANSPORT",
            "description": "Passenger rail transport, interurban",
            "eligible_objectives": ["CCM", "PPC"],
            "is_enabling": True,
            "is_transitional": False,
            "revenue_eur": 25_000_000,
        },
        {
            "activity_id": str(uuid.uuid4()),
            "name": "Water supply and treatment",
            "nace_code": "E36.00",
            "taxonomy_id": "WTR-2.1",
            "sector": "WATER_SUPPLY",
            "description": "Water collection, treatment and supply",
            "eligible_objectives": ["WTR", "CE"],
            "is_enabling": False,
            "is_transitional": False,
            "revenue_eur": 12_000_000,
        },
    ]


@pytest.fixture
def sample_financial_data() -> Dict[str, Any]:
    """Create turnover/CapEx/OpEx data for testing KPI calculation."""
    return {
        "organization_name": "GreenTech Industries GmbH",
        "reporting_year": 2025,
        "currency": "EUR",
        "total_turnover": 110_000_000,
        "total_capex": 22_000_000,
        "total_opex": 35_000_000,
        "activities": [
            {
                "taxonomy_id": "CCM-4.1",
                "name": "Solar PV electricity generation",
                "turnover": 15_000_000,
                "capex": 5_000_000,
                "opex": 3_000_000,
                "alignment_status": "ALIGNED",
            },
            {
                "taxonomy_id": "CCM-3.7",
                "name": "Cement manufacturing",
                "turnover": 50_000_000,
                "capex": 10_000_000,
                "opex": 18_000_000,
                "alignment_status": "ELIGIBLE",
            },
            {
                "taxonomy_id": "CCM-7.2",
                "name": "Building renovation",
                "turnover": 8_000_000,
                "capex": 3_000_000,
                "opex": 4_000_000,
                "alignment_status": "ALIGNED",
            },
            {
                "taxonomy_id": "CCM-6.1",
                "name": "Passenger rail transport",
                "turnover": 25_000_000,
                "capex": 2_000_000,
                "opex": 5_000_000,
                "alignment_status": "ALIGNED",
            },
            {
                "taxonomy_id": None,
                "name": "Administrative services",
                "turnover": 12_000_000,
                "capex": 2_000_000,
                "opex": 5_000_000,
                "alignment_status": "NOT_ELIGIBLE",
            },
        ],
    }


@pytest.fixture
def sample_alignment_results() -> List[Dict[str, Any]]:
    """Create pre-computed alignment results for testing."""
    return [
        {
            "activity_id": str(uuid.uuid4()),
            "taxonomy_id": "CCM-4.1",
            "name": "Solar PV electricity generation",
            "eligibility_status": "ELIGIBLE",
            "sc_result": True,
            "dnsh_result": True,
            "ms_result": True,
            "tsc_result": True,
            "alignment_status": "ALIGNED",
            "contributing_objective": "CCM",
            "provenance_hash": _compute_hash("CCM-4.1-aligned"),
        },
        {
            "activity_id": str(uuid.uuid4()),
            "taxonomy_id": "CCM-3.7",
            "name": "Cement manufacturing",
            "eligibility_status": "ELIGIBLE",
            "sc_result": False,
            "dnsh_result": True,
            "ms_result": True,
            "tsc_result": False,
            "alignment_status": "NOT_ALIGNED",
            "contributing_objective": "CCM",
            "gap_reasons": [
                "SC: emissions threshold exceeded",
                "TSC: CO2 capture not met",
            ],
            "provenance_hash": _compute_hash("CCM-3.7-not-aligned"),
        },
        {
            "activity_id": str(uuid.uuid4()),
            "taxonomy_id": "CCM-7.2",
            "name": "Building renovation",
            "eligibility_status": "ELIGIBLE",
            "sc_result": True,
            "dnsh_result": True,
            "ms_result": True,
            "tsc_result": True,
            "alignment_status": "ALIGNED",
            "contributing_objective": "CCM",
            "provenance_hash": _compute_hash("CCM-7.2-aligned"),
        },
    ]


@pytest.fixture
def sample_exposures() -> List[Dict[str, Any]]:
    """Create GAR exposure data for financial institution testing."""
    return [
        {
            "exposure_id": str(uuid.uuid4()),
            "counterparty_name": "GreenTech Industries GmbH",
            "exposure_type": "CORPORATE_LOANS",
            "gross_carrying_amount": 50_000_000,
            "counterparty_taxonomy_turnover_aligned_pct": 43.6,
            "counterparty_taxonomy_capex_aligned_pct": 45.5,
            "taxonomy_aligned_amount": 21_800_000,
        },
        {
            "exposure_id": str(uuid.uuid4()),
            "counterparty_name": "EcoEnergy AG",
            "exposure_type": "CORPORATE_LOANS",
            "gross_carrying_amount": 30_000_000,
            "counterparty_taxonomy_turnover_aligned_pct": 85.0,
            "counterparty_taxonomy_capex_aligned_pct": 90.0,
            "taxonomy_aligned_amount": 25_500_000,
        },
        {
            "exposure_id": str(uuid.uuid4()),
            "counterparty_name": "Residential Portfolio",
            "exposure_type": "RESIDENTIAL_MORTGAGES",
            "gross_carrying_amount": 200_000_000,
            "epc_rating": "B",
            "taxonomy_aligned_amount": 140_000_000,
        },
        {
            "exposure_id": str(uuid.uuid4()),
            "counterparty_name": "Wind Farm Project",
            "exposure_type": "PROJECT_FINANCE",
            "gross_carrying_amount": 75_000_000,
            "counterparty_taxonomy_turnover_aligned_pct": 100.0,
            "taxonomy_aligned_amount": 75_000_000,
        },
    ]


@pytest.fixture
def sample_tsc_data() -> List[Dict[str, Any]]:
    """Create technical screening criteria evaluation data for testing."""
    return [
        {
            "activity_id": "CCM-4.1",
            "objective": "CCM",
            "criteria": [
                {
                    "criterion_id": "CCM-4.1-SC-1",
                    "type": "quantitative",
                    "description": "GHG emissions intensity below lifecycle threshold",
                    "threshold": 100,
                    "unit": "gCO2e/kWh",
                    "actual_value": 22,
                    "passed": True,
                    "evidence_ref": "PV-lifecycle-assessment-2025.pdf",
                },
            ],
            "overall_pass": True,
        },
        {
            "activity_id": "CCM-3.7",
            "objective": "CCM",
            "criteria": [
                {
                    "criterion_id": "CCM-3.7-SC-1",
                    "type": "quantitative",
                    "description": "Specific GHG emissions from clinker production",
                    "threshold": 0.498,
                    "unit": "tCO2e/t clinker",
                    "actual_value": 0.625,
                    "passed": False,
                    "gap": 0.127,
                    "evidence_ref": "cement-emissions-report-2025.pdf",
                },
                {
                    "criterion_id": "CCM-3.7-SC-2",
                    "type": "qualitative",
                    "description": "Best Available Technology assessment",
                    "assessment": "partial",
                    "passed": False,
                    "notes": "BAT benchmark not achieved for kiln efficiency",
                },
            ],
            "overall_pass": False,
        },
    ]


@pytest.fixture
def sample_dnsh_data() -> List[Dict[str, Any]]:
    """Create DNSH matrix assessment data for testing."""
    return [
        {
            "activity_id": "CCM-4.1",
            "contributing_objective": "CCM",
            "dnsh_assessments": {
                "CCA": {"passed": True, "notes": "Climate risk assessment completed"},
                "WTR": {"passed": True, "notes": "No significant water impact"},
                "CE": {"passed": True, "notes": "Waste management plan in place"},
                "PPC": {"passed": True, "notes": "Below pollution thresholds"},
                "BIO": {"passed": True, "notes": "No protected area impact"},
            },
            "overall_pass": True,
        },
        {
            "activity_id": "CCM-3.7",
            "contributing_objective": "CCM",
            "dnsh_assessments": {
                "CCA": {"passed": True, "notes": "Climate adaptation plan documented"},
                "WTR": {"passed": False, "notes": "Water discharge exceeds WFD limits"},
                "CE": {"passed": True, "notes": "Circular economy measures in place"},
                "PPC": {"passed": True, "notes": "BAT compliance confirmed"},
                "BIO": {"passed": True, "notes": "EIA completed, no significant impact"},
            },
            "overall_pass": False,
        },
    ]


@pytest.fixture
def sample_ms_data() -> List[Dict[str, Any]]:
    """Create minimum safeguards verification data for testing."""
    return [
        {
            "organization_id": str(uuid.uuid4()),
            "organization_name": "GreenTech Industries GmbH",
            "assessments": {
                "human_rights": {
                    "passed": True,
                    "framework": "UNGP",
                    "evidence": "HR due diligence policy v3.2",
                    "last_reviewed": "2025-06-15",
                },
                "anti_corruption": {
                    "passed": True,
                    "framework": "UN Convention against Corruption",
                    "evidence": "Anti-bribery compliance programme",
                    "last_reviewed": "2025-04-01",
                },
                "taxation": {
                    "passed": True,
                    "framework": "EU Tax Good Governance",
                    "evidence": "Country-by-country tax reporting",
                    "last_reviewed": "2025-07-20",
                },
                "fair_competition": {
                    "passed": True,
                    "framework": "EU Competition Law",
                    "evidence": "Competition compliance programme",
                    "last_reviewed": "2025-05-10",
                },
            },
            "overall_pass": True,
        },
    ]


@pytest.fixture
def sample_kpi_data() -> Dict[str, Any]:
    """Create KPI calculation results for testing."""
    return {
        "organization_name": "GreenTech Industries GmbH",
        "reporting_year": 2025,
        "currency": "EUR",
        "kpi_results": {
            "turnover": {
                "total": 110_000_000,
                "eligible": 98_000_000,
                "aligned": 48_000_000,
                "eligible_pct": 89.09,
                "aligned_pct": 43.64,
                "by_objective": {
                    "CCM": {"eligible": 98_000_000, "aligned": 48_000_000},
                    "CCA": {"eligible": 8_000_000, "aligned": 8_000_000},
                },
            },
            "capex": {
                "total": 22_000_000,
                "eligible": 20_000_000,
                "aligned": 10_000_000,
                "eligible_pct": 90.91,
                "aligned_pct": 45.45,
            },
            "opex": {
                "total": 35_000_000,
                "eligible": 30_000_000,
                "aligned": 12_000_000,
                "eligible_pct": 85.71,
                "aligned_pct": 34.29,
            },
        },
        "provenance_hash": _compute_hash("kpi-2025-greentech"),
    }


@pytest.fixture
def sample_gar_data() -> Dict[str, Any]:
    """Create GAR/BTAR calculation data for financial institution testing."""
    return {
        "institution_name": "EcoBank AG",
        "reporting_year": 2025,
        "gar_stock": {
            "total_covered_assets": 5_000_000_000,
            "taxonomy_aligned_assets": 1_250_000_000,
            "gar_ratio_pct": 25.0,
            "by_objective": {
                "CCM": {
                    "aligned": 1_000_000_000,
                    "pct": 20.0,
                },
                "CCA": {
                    "aligned": 250_000_000,
                    "pct": 5.0,
                },
            },
            "by_exposure_type": {
                "CORPORATE_LOANS": {
                    "total": 2_000_000_000,
                    "aligned": 500_000_000,
                },
                "RESIDENTIAL_MORTGAGES": {
                    "total": 2_000_000_000,
                    "aligned": 500_000_000,
                },
                "PROJECT_FINANCE": {
                    "total": 500_000_000,
                    "aligned": 200_000_000,
                },
                "EQUITY_HOLDINGS": {
                    "total": 500_000_000,
                    "aligned": 50_000_000,
                },
            },
        },
        "gar_flow": {
            "new_originations": 800_000_000,
            "taxonomy_aligned_originations": 280_000_000,
            "gar_flow_ratio_pct": 35.0,
        },
        "btar": {
            "banking_book_total": 4_500_000_000,
            "taxonomy_aligned_banking": 1_125_000_000,
            "btar_ratio_pct": 25.0,
        },
        "provenance_hash": _compute_hash("gar-2025-ecobank"),
    }


# ---------------------------------------------------------------------------
# Temporary output directory
# ---------------------------------------------------------------------------

@pytest.fixture
def temp_output_dir(tmp_path) -> Path:
    """Create a temporary directory for test outputs."""
    output_dir = tmp_path / "taxonomy_alignment_output"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir
