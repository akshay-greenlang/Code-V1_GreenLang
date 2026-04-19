# -*- coding: utf-8 -*-
"""
PACK-009 EU Climate Compliance Bundle Pack - Shared Test Fixtures
==================================================================

Provides reusable pytest fixtures for all PACK-009 test modules including
cross-regulation compliance management, data deduplication, gap analysis,
regulatory calendar, consistency checking, compliance scoring, evidence
management, and consolidated reporting.

Covers four EU regulations:
- CSRD: Corporate Sustainability Reporting Directive (EU) 2022/2464
- CBAM: Carbon Border Adjustment Mechanism (EU) 2023/956
- EUDR: EU Deforestation Regulation (EU) 2023/1115
- EU Taxonomy: Regulation (EU) 2020/852

Constituent Packs:
- PACK-001: CSRD Starter Pack
- PACK-004: CBAM Readiness Pack
- PACK-006: EUDR Starter Pack
- PACK-008: EU Taxonomy Alignment Pack

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
DEMO_DIR = CONFIG_DIR / "demo"
ENGINES_DIR = PACK_ROOT / "engines"
WORKFLOWS_DIR = PACK_ROOT / "workflows"
TEMPLATES_DIR = PACK_ROOT / "templates"
INTEGRATIONS_DIR = PACK_ROOT / "integrations"


# ---------------------------------------------------------------------------
# Constants - Regulations and constituent packs
# ---------------------------------------------------------------------------

REGULATIONS = ["CSRD", "CBAM", "EUDR", "TAXONOMY"]

CONSTITUENT_PACKS = {
    "PACK-001": "csrd-starter",
    "PACK-004": "cbam-readiness",
    "PACK-006": "eudr-starter",
    "PACK-008": "eu-taxonomy-alignment",
}

REGULATION_DISPLAY_NAMES = {
    "CSRD": "Corporate Sustainability Reporting Directive",
    "CBAM": "Carbon Border Adjustment Mechanism",
    "EUDR": "EU Deforestation Regulation",
    "TAXONOMY": "EU Taxonomy Regulation",
}

REGULATION_REFERENCES = {
    "CSRD": "Directive (EU) 2022/2464",
    "CBAM": "Regulation (EU) 2023/956",
    "EUDR": "Regulation (EU) 2023/1115",
    "TAXONOMY": "Regulation (EU) 2020/852",
}

BUNDLE_ENGINE_IDS = [
    "cross_framework_data_mapper",
    "data_deduplication",
    "cross_regulation_gap_analyzer",
    "regulatory_calendar",
    "consolidated_metrics",
    "multi_regulation_consistency",
    "bundle_compliance_scoring",
    "cross_regulation_evidence",
]

BUNDLE_WORKFLOW_IDS = [
    "unified_data_collection",
    "cross_regulation_assessment",
    "consolidated_reporting",
    "calendar_management",
    "cross_framework_gap_analysis",
    "bundle_health_check",
    "data_consistency_reconciliation",
    "annual_compliance_review",
]

BUNDLE_TEMPLATE_IDS = [
    "consolidated_dashboard",
    "cross_regulation_data_map",
    "unified_gap_analysis",
    "regulatory_calendar_report",
    "data_consistency_report",
    "bundle_executive_summary",
    "deduplication_savings",
    "multi_regulation_audit_trail",
]

BUNDLE_INTEGRATION_IDS = [
    "bundle_orchestrator",
    "csrd_pack_bridge",
    "cbam_pack_bridge",
    "eudr_pack_bridge",
    "taxonomy_pack_bridge",
    "cross_framework_mapper_bridge",
    "shared_data_pipeline_bridge",
    "consolidated_evidence_bridge",
    "bundle_health_check_integration",
    "setup_wizard",
]

BUNDLE_PRESET_IDS = [
    "enterprise_full",
    "financial_institution",
    "eu_importer",
    "sme_essential",
]

ENGINE_FILES = {
    "cross_framework_data_mapper": "cross_framework_data_mapper.py",
    "data_deduplication": "data_deduplication_engine.py",
    "cross_regulation_gap_analyzer": "cross_regulation_gap_analyzer.py",
    "regulatory_calendar": "regulatory_calendar_engine.py",
    "consolidated_metrics": "consolidated_metrics_engine.py",
    "multi_regulation_consistency": "multi_regulation_consistency_engine.py",
    "bundle_compliance_scoring": "bundle_compliance_scoring_engine.py",
    "cross_regulation_evidence": "cross_regulation_evidence_engine.py",
}

WORKFLOW_FILES = {
    "unified_data_collection": "unified_data_collection.py",
    "cross_regulation_assessment": "cross_regulation_assessment.py",
    "consolidated_reporting": "consolidated_reporting.py",
    "calendar_management": "calendar_management.py",
    "cross_framework_gap_analysis": "cross_framework_gap_analysis.py",
    "bundle_health_check": "bundle_health_check.py",
    "data_consistency_reconciliation": "data_consistency_reconciliation.py",
    "annual_compliance_review": "annual_compliance_review.py",
}

TEMPLATE_FILES = {
    "consolidated_dashboard": "consolidated_dashboard.py",
    "cross_regulation_data_map": "cross_regulation_data_map.py",
    "unified_gap_analysis": "unified_gap_analysis_report.py",
    "regulatory_calendar_report": "regulatory_calendar_report.py",
    "data_consistency_report": "data_consistency_report.py",
    "bundle_executive_summary": "bundle_executive_summary.py",
    "deduplication_savings": "deduplication_savings_report.py",
    "multi_regulation_audit_trail": "multi_regulation_audit_trail.py",
}

INTEGRATION_FILES = {
    "bundle_orchestrator": "pack_orchestrator.py",
    "csrd_pack_bridge": "csrd_pack_bridge.py",
    "cbam_pack_bridge": "cbam_pack_bridge.py",
    "eudr_pack_bridge": "eudr_pack_bridge.py",
    "taxonomy_pack_bridge": "taxonomy_pack_bridge.py",
    "cross_framework_mapper_bridge": "cross_framework_mapper_bridge.py",
    "shared_data_pipeline_bridge": "shared_data_pipeline_bridge.py",
    "consolidated_evidence_bridge": "consolidated_evidence_bridge.py",
    "bundle_health_check_integration": "bundle_health_check.py",
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
    """Return the absolute path to the PACK-009 root directory."""
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
        "pack_config",
        CONFIG_DIR / "pack_config.py",
    )


@pytest.fixture
def bundle_config(pack_config_module):
    """Create a default BundleComplianceConfig instance.

    Returns a BundleComplianceConfig with all default values (enterprise_full
    tier, all four regulations enabled, all engines active).
    """
    return pack_config_module.BundleComplianceConfig()


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
            engine_id,
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
            workflow_id,
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
            template_id,
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
            integration_id,
            INTEGRATIONS_DIR / filename,
        )
    return modules


# ---------------------------------------------------------------------------
# Sample data fixtures - Regulation data
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_regulation_data() -> Dict[str, Dict[str, Any]]:
    """Create sample compliance data for each of the four regulations.

    Returns a dictionary keyed by regulation code containing representative
    data for cross-regulation testing scenarios.
    """
    return {
        "CSRD": {
            "regulation": "CSRD",
            "reference": "Directive (EU) 2022/2464",
            "reporting_year": 2025,
            "compliance_status": "IN_PROGRESS",
            "compliance_score": 72.5,
            "data_quality_score": 0.85,
            "ghg_emissions": {
                "scope_1": 12500.0,
                "scope_2_location": 8300.0,
                "scope_2_market": 7100.0,
                "scope_3": 45000.0,
                "unit": "tCO2e",
            },
            "materiality_topics": [
                "climate_change_mitigation",
                "climate_change_adaptation",
                "biodiversity",
                "circular_economy",
            ],
            "esrs_standards_covered": [
                "ESRS E1", "ESRS E2", "ESRS E4", "ESRS S1", "ESRS G1",
            ],
            "taxonomy_kpis": {
                "turnover_eligible_pct": 68.5,
                "turnover_aligned_pct": 42.3,
                "capex_eligible_pct": 75.0,
                "capex_aligned_pct": 55.0,
            },
            "provenance_hash": _compute_hash("csrd-2025-sample"),
        },
        "CBAM": {
            "regulation": "CBAM",
            "reference": "Regulation (EU) 2023/956",
            "reporting_year": 2025,
            "reporting_quarter": "Q4",
            "compliance_status": "COMPLIANT",
            "compliance_score": 88.0,
            "data_quality_score": 0.90,
            "imported_goods": [
                {
                    "product_category": "cement",
                    "hs_code": "2523.29",
                    "weight_tonnes": 5000.0,
                    "embedded_emissions_tco2e": 4250.0,
                    "origin_country": "TR",
                    "supplier": "Anatolian Cement Ltd",
                },
                {
                    "product_category": "steel",
                    "hs_code": "7208.51",
                    "weight_tonnes": 2000.0,
                    "embedded_emissions_tco2e": 3600.0,
                    "origin_country": "IN",
                    "supplier": "Tata Steel India",
                },
            ],
            "total_embedded_emissions": 7850.0,
            "cbam_certificates_required": 7850,
            "provenance_hash": _compute_hash("cbam-2025-q4-sample"),
        },
        "EUDR": {
            "regulation": "EUDR",
            "reference": "Regulation (EU) 2023/1115",
            "reporting_year": 2025,
            "compliance_status": "PARTIALLY_COMPLIANT",
            "compliance_score": 65.0,
            "data_quality_score": 0.75,
            "commodities": ["soya", "palm_oil", "wood"],
            "supply_chain_entries": [
                {
                    "commodity": "soya",
                    "supplier": "Agro Brazil SA",
                    "country": "BR",
                    "geolocation": {"lat": -12.97, "lon": -38.51},
                    "deforestation_free": True,
                    "risk_level": "STANDARD",
                    "dds_submitted": True,
                },
                {
                    "commodity": "palm_oil",
                    "supplier": "Indo Palm Co",
                    "country": "ID",
                    "geolocation": {"lat": -2.49, "lon": 104.73},
                    "deforestation_free": False,
                    "risk_level": "HIGH",
                    "dds_submitted": False,
                },
            ],
            "total_dds_required": 12,
            "total_dds_submitted": 8,
            "provenance_hash": _compute_hash("eudr-2025-sample"),
        },
        "TAXONOMY": {
            "regulation": "TAXONOMY",
            "reference": "Regulation (EU) 2020/852",
            "reporting_year": 2025,
            "compliance_status": "IN_PROGRESS",
            "compliance_score": 78.0,
            "data_quality_score": 0.88,
            "organization_type": "NON_FINANCIAL_UNDERTAKING",
            "activities_screened": 15,
            "activities_eligible": 12,
            "activities_aligned": 8,
            "kpi_results": {
                "turnover": {
                    "total": 110_000_000,
                    "eligible": 85_000_000,
                    "aligned": 48_000_000,
                    "eligible_pct": 77.27,
                    "aligned_pct": 43.64,
                },
                "capex": {
                    "total": 22_000_000,
                    "eligible": 18_000_000,
                    "aligned": 10_000_000,
                    "eligible_pct": 81.82,
                    "aligned_pct": 45.45,
                },
            },
            "environmental_objectives": ["CCM", "CCA", "WTR"],
            "provenance_hash": _compute_hash("taxonomy-2025-sample"),
        },
    }


# ---------------------------------------------------------------------------
# Sample data fixtures - Evidence items
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_evidence_items() -> List[Dict[str, Any]]:
    """Create sample cross-regulation evidence items.

    Returns a list of evidence dictionaries that can be shared across
    multiple regulations.
    """
    return [
        {
            "evidence_id": str(uuid.uuid4()),
            "title": "Annual GHG Emissions Report 2025",
            "evidence_type": "REPORT",
            "description": "Verified Scope 1/2/3 emissions inventory for FY2025",
            "file_format": "pdf",
            "file_size_mb": 12.5,
            "applicable_regulations": ["CSRD", "CBAM", "TAXONOMY"],
            "status": "VERIFIED",
            "uploaded_date": "2025-12-15",
            "verified_date": "2026-01-10",
            "expiry_date": "2026-12-31",
            "provenance_hash": _compute_hash("ghg-report-2025"),
        },
        {
            "evidence_id": str(uuid.uuid4()),
            "title": "Supply Chain Traceability Certificate",
            "evidence_type": "CERTIFICATE",
            "description": "Third-party verified supply chain mapping for EUDR commodities",
            "file_format": "pdf",
            "file_size_mb": 3.2,
            "applicable_regulations": ["EUDR", "CSRD"],
            "status": "APPROVED",
            "uploaded_date": "2025-11-01",
            "verified_date": "2025-11-20",
            "expiry_date": "2026-11-01",
            "provenance_hash": _compute_hash("supply-chain-cert-2025"),
        },
        {
            "evidence_id": str(uuid.uuid4()),
            "title": "EU Taxonomy Alignment Assessment",
            "evidence_type": "CALCULATION",
            "description": "Taxonomy eligibility and alignment KPI calculation workbook",
            "file_format": "xlsx",
            "file_size_mb": 8.7,
            "applicable_regulations": ["TAXONOMY", "CSRD"],
            "status": "COLLECTED",
            "uploaded_date": "2026-01-05",
            "verified_date": None,
            "expiry_date": "2027-01-05",
            "provenance_hash": _compute_hash("taxonomy-kpi-2025"),
        },
        {
            "evidence_id": str(uuid.uuid4()),
            "title": "CBAM Embedded Emissions Calculation",
            "evidence_type": "CALCULATION",
            "description": "Embedded emissions calculation for imported cement and steel",
            "file_format": "xlsx",
            "file_size_mb": 5.1,
            "applicable_regulations": ["CBAM"],
            "status": "VERIFIED",
            "uploaded_date": "2025-10-01",
            "verified_date": "2025-10-15",
            "expiry_date": "2026-03-31",
            "provenance_hash": _compute_hash("cbam-emissions-2025"),
        },
        {
            "evidence_id": str(uuid.uuid4()),
            "title": "Biodiversity Impact Assessment",
            "evidence_type": "DOCUMENT",
            "description": "Environmental impact assessment covering biodiversity and land use",
            "file_format": "pdf",
            "file_size_mb": 25.0,
            "applicable_regulations": ["EUDR", "TAXONOMY", "CSRD"],
            "status": "APPROVED",
            "uploaded_date": "2025-09-15",
            "verified_date": "2025-10-01",
            "expiry_date": "2026-09-15",
            "provenance_hash": _compute_hash("biodiversity-eia-2025"),
        },
    ]


# ---------------------------------------------------------------------------
# Sample data fixtures - Field mappings
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_field_mappings() -> List[Dict[str, Any]]:
    """Create sample cross-regulation field mapping entries.

    Returns a list of field mapping dictionaries describing how data fields
    are shared across regulation-specific data models.
    """
    return [
        {
            "field_id": "ghg_scope_1_emissions",
            "display_name": "Scope 1 GHG Emissions",
            "category": "GHG_EMISSIONS",
            "unit": "tCO2e",
            "source_regulations": ["CSRD", "CBAM", "TAXONOMY"],
            "csrd_field": "esrs_e1.scope_1_emissions",
            "cbam_field": "embedded_emissions.direct_emissions",
            "taxonomy_field": "ccm_screening.emissions_scope_1",
            "transformation_required": False,
            "deduplication_candidate": True,
        },
        {
            "field_id": "turnover_total",
            "display_name": "Total Turnover (Revenue)",
            "category": "FINANCIAL_DATA",
            "unit": "EUR",
            "source_regulations": ["CSRD", "TAXONOMY"],
            "csrd_field": "financial_data.total_revenue",
            "taxonomy_field": "kpi_data.total_turnover",
            "transformation_required": False,
            "deduplication_candidate": True,
        },
        {
            "field_id": "supply_chain_country_origin",
            "display_name": "Country of Origin",
            "category": "SUPPLY_CHAIN",
            "unit": "ISO-3166-1-alpha-2",
            "source_regulations": ["CBAM", "EUDR", "CSRD"],
            "cbam_field": "import_declaration.country_of_origin",
            "eudr_field": "supply_chain.source_country",
            "csrd_field": "esrs_e4.supply_chain_origin",
            "transformation_required": False,
            "deduplication_candidate": True,
        },
        {
            "field_id": "biodiversity_impact_score",
            "display_name": "Biodiversity Impact Score",
            "category": "BIODIVERSITY",
            "unit": "score_0_100",
            "source_regulations": ["EUDR", "TAXONOMY", "CSRD"],
            "eudr_field": "risk_assessment.biodiversity_score",
            "taxonomy_field": "dnsh.bio_impact_score",
            "csrd_field": "esrs_e4.biodiversity_impact",
            "transformation_required": True,
            "deduplication_candidate": False,
        },
    ]


# ---------------------------------------------------------------------------
# Sample data fixtures - Gap data
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_gap_data() -> List[Dict[str, Any]]:
    """Create sample cross-regulation gap analysis entries.

    Returns a list of gap dictionaries identifying compliance gaps
    across multiple regulations.
    """
    return [
        {
            "gap_id": str(uuid.uuid4()),
            "title": "Scope 3 emissions data incomplete for CSRD and Taxonomy",
            "severity": "CRITICAL",
            "affected_regulations": ["CSRD", "TAXONOMY"],
            "requirement_csrd": "ESRS E1 - Complete Scope 3 emissions disclosure",
            "requirement_taxonomy": "CCM Technical Screening Criteria verification",
            "current_status": "Scope 3 categories 1-8 complete, categories 9-15 missing",
            "remediation_plan": "Engage suppliers for categories 9-15 data collection",
            "effort_estimate_days": 45,
            "cross_regulation_leverage": True,
            "leverage_note": "Completing Scope 3 for CSRD also satisfies Taxonomy CCM data needs",
        },
        {
            "gap_id": str(uuid.uuid4()),
            "title": "CBAM quarterly report Q3 not submitted",
            "severity": "HIGH",
            "affected_regulations": ["CBAM"],
            "requirement_cbam": "Quarterly CBAM report due within 1 month of quarter end",
            "current_status": "Q3 2025 report not yet filed",
            "remediation_plan": "Complete embedded emissions calculation and file Q3 report",
            "effort_estimate_days": 10,
            "cross_regulation_leverage": False,
            "leverage_note": None,
        },
        {
            "gap_id": str(uuid.uuid4()),
            "title": "EUDR due diligence statements incomplete for palm oil suppliers",
            "severity": "HIGH",
            "affected_regulations": ["EUDR", "CSRD"],
            "requirement_eudr": "DDS required for all EUDR commodity placements on EU market",
            "requirement_csrd": "ESRS E4 - Supply chain biodiversity due diligence",
            "current_status": "4 of 12 DDS completed for palm oil supply chain",
            "remediation_plan": "Accelerate supplier engagement for remaining 8 DDS",
            "effort_estimate_days": 30,
            "cross_regulation_leverage": True,
            "leverage_note": "EUDR DDS evidence supports CSRD ESRS E4 biodiversity disclosure",
        },
        {
            "gap_id": str(uuid.uuid4()),
            "title": "Taxonomy DNSH assessment missing for water objective",
            "severity": "MEDIUM",
            "affected_regulations": ["TAXONOMY"],
            "requirement_taxonomy": "DNSH assessment for WTR objective (water and marine resources)",
            "current_status": "DNSH completed for CCM, CCA, CE, PPC, BIO but not WTR",
            "remediation_plan": "Commission water impact assessment for relevant activities",
            "effort_estimate_days": 15,
            "cross_regulation_leverage": False,
            "leverage_note": None,
        },
        {
            "gap_id": str(uuid.uuid4()),
            "title": "Cross-regulation GHG emissions data inconsistency",
            "severity": "MEDIUM",
            "affected_regulations": ["CSRD", "CBAM", "TAXONOMY"],
            "current_status": "Scope 1 emissions differ by 3.2% between CSRD and CBAM submissions",
            "remediation_plan": "Reconcile emission factor sources and calculation methodologies",
            "effort_estimate_days": 5,
            "cross_regulation_leverage": True,
            "leverage_note": "Single source of truth for GHG data eliminates inconsistencies",
        },
    ]


# ---------------------------------------------------------------------------
# Sample data fixtures - Calendar events
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_calendar_events() -> List[Dict[str, Any]]:
    """Create sample unified regulatory calendar events.

    Returns a list of calendar event dictionaries spanning all four
    EU regulations with various event types and deadlines.
    """
    base_year = 2026
    return [
        {
            "event_id": str(uuid.uuid4()),
            "event_type": "FILING_DEADLINE",
            "regulation": "CSRD",
            "title": "CSRD Annual Report Filing Deadline",
            "description": "Annual CSRD sustainability report filing due",
            "deadline_date": f"{base_year}-04-30",
            "lead_time_days": 30,
            "priority": 1,
            "status": "UPCOMING",
            "dependencies": [],
        },
        {
            "event_id": str(uuid.uuid4()),
            "event_type": "FILING_DEADLINE",
            "regulation": "CBAM",
            "title": "CBAM Q1 Quarterly Report",
            "description": "Q1 quarterly CBAM report due within 1 month of quarter end",
            "deadline_date": f"{base_year}-04-30",
            "lead_time_days": 14,
            "priority": 2,
            "status": "UPCOMING",
            "dependencies": [],
        },
        {
            "event_id": str(uuid.uuid4()),
            "event_type": "DATA_COLLECTION",
            "regulation": "EUDR",
            "title": "EUDR Supply Chain Data Collection Window",
            "description": "Quarterly supplier due diligence data collection",
            "deadline_date": f"{base_year}-03-31",
            "lead_time_days": 60,
            "priority": 3,
            "status": "IN_PROGRESS",
            "dependencies": [],
        },
        {
            "event_id": str(uuid.uuid4()),
            "event_type": "FILING_DEADLINE",
            "regulation": "TAXONOMY",
            "title": "EU Taxonomy Annual Disclosure",
            "description": "Annual Article 8 taxonomy alignment disclosure",
            "deadline_date": f"{base_year}-04-30",
            "lead_time_days": 30,
            "priority": 2,
            "status": "UPCOMING",
            "dependencies": [],
        },
        {
            "event_id": str(uuid.uuid4()),
            "event_type": "REVIEW_MILESTONE",
            "regulation": "CSRD",
            "title": "Internal CSRD Pre-Filing Review",
            "description": "Internal review of draft CSRD report before filing",
            "deadline_date": f"{base_year}-04-15",
            "lead_time_days": 14,
            "priority": 1,
            "status": "UPCOMING",
            "dependencies": [],
        },
        {
            "event_id": str(uuid.uuid4()),
            "event_type": "AUDIT_DATE",
            "regulation": "CSRD",
            "title": "External Assurance Audit",
            "description": "Third-party limited assurance engagement for CSRD sustainability report",
            "deadline_date": f"{base_year}-03-15",
            "lead_time_days": 7,
            "priority": 1,
            "status": "UPCOMING",
            "dependencies": [],
        },
        {
            "event_id": str(uuid.uuid4()),
            "event_type": "BOARD_REPORT",
            "regulation": "CSRD",
            "title": "Board Sustainability Report Approval",
            "description": "Board meeting to approve annual sustainability report",
            "deadline_date": f"{base_year}-04-01",
            "lead_time_days": 7,
            "priority": 1,
            "status": "UPCOMING",
            "dependencies": [],
        },
    ]


# ---------------------------------------------------------------------------
# Temporary output directory
# ---------------------------------------------------------------------------

@pytest.fixture
def temp_output_dir(tmp_path) -> Path:
    """Create a temporary directory for test outputs."""
    output_dir = tmp_path / "bundle_compliance_output"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir
