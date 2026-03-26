"""
PACK-048 GHG Assurance Prep Pack - Shared Test Fixtures
=============================================================

Provides shared fixtures for all PACK-048 test modules including
engine instances, sample data, configuration objects, evidence items,
controls, engagement details, emissions data, jurisdiction records,
ISAE 3410 checklists, and helper utilities.

All numeric fixtures use Decimal for regulatory precision.

Author: GreenLang QA Team
Date: March 2026
"""
from __future__ import annotations

import hashlib
import json
import sys
import uuid
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pytest

# ---------------------------------------------------------------------------
# Path setup - ensure PACK-048 root is importable
# ---------------------------------------------------------------------------
PACK_ROOT = Path(__file__).resolve().parent.parent
if str(PACK_ROOT) not in sys.path:
    sys.path.insert(0, str(PACK_ROOT))


# ---------------------------------------------------------------------------
# Decimal comparison helpers
# ---------------------------------------------------------------------------


def decimal_approx(
    actual: Optional[Decimal],
    expected: Decimal,
    tolerance: Decimal = Decimal("0.000001"),
) -> bool:
    """Compare two Decimals within a given tolerance.

    Returns True if |actual - expected| <= tolerance.
    """
    if actual is None:
        return False
    return abs(actual - expected) <= tolerance


def assert_decimal_equal(
    actual: Optional[Decimal],
    expected: Decimal,
    tolerance: Decimal = Decimal("0.000001"),
    msg: str = "",
) -> None:
    """Assert two Decimals are equal within tolerance."""
    assert actual is not None, f"Actual value is None. {msg}"
    diff = abs(actual - expected)
    assert diff <= tolerance, (
        f"Decimal mismatch: actual={actual}, expected={expected}, "
        f"diff={diff}, tolerance={tolerance}. {msg}"
    )


def assert_decimal_gt(
    actual: Optional[Decimal],
    threshold: Decimal,
    msg: str = "",
) -> None:
    """Assert Decimal value is greater than threshold."""
    assert actual is not None, f"Actual value is None. {msg}"
    assert actual > threshold, (
        f"Expected {actual} > {threshold}. {msg}"
    )


def assert_decimal_between(
    actual: Optional[Decimal],
    lo: Decimal,
    hi: Decimal,
    msg: str = "",
) -> None:
    """Assert Decimal value is within [lo, hi]."""
    assert actual is not None, f"Actual value is None. {msg}"
    assert lo <= actual <= hi, (
        f"Expected {lo} <= {actual} <= {hi}. {msg}"
    )


def compute_test_hash(data: Any) -> str:
    """Compute SHA-256 hash from arbitrary data (for determinism testing)."""
    canonical = json.dumps(data, sort_keys=True, default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Configuration Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_config() -> Dict[str, Any]:
    """Create PackConfig-like dict with corporate_general preset defaults."""
    return {
        "pack_id": "PACK-048",
        "pack_name": "GHG Assurance Prep",
        "version": "1.0.0",
        "preset": "corporate_general",
        "organisation_id": "org-test-001",
        "organisation_name": "ACME Corp",
        "sector": "INDUSTRIALS",
        "sector_codes": {
            "gics": "2010",
            "nace": "C25",
            "isic": "C25",
        },
        "reporting_year": 2025,
        "base_year": 2020,
        "revenue_usd_m": Decimal("500"),
        "employees_fte": 2000,
        "scope_boundary": "scope_1_2_3",
        "gwp_version": "AR6",
        "currency": "USD",
        "assurance_level": "limited",
        "assurance_standard": "ISAE_3410",
        "verifier_name": "Big Four Assurance LLP",
        "engagement_year": 2025,
        "materiality_pct": Decimal("5"),
        "performance_materiality_pct": Decimal("75"),
        "clearly_trivial_pct": Decimal("5"),
        "confidence_level": Decimal("0.95"),
        "sampling_method": "MUS",
        "jurisdictions": ["EU_CSRD", "US_SEC", "CA_SB253"],
        "output_formats": ["markdown", "html", "json", "xbrl"],
        "company_size": "LARGE",
        "first_time_engagement": True,
    }


# ---------------------------------------------------------------------------
# Evidence Item Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_evidence_items() -> List[Dict[str, Any]]:
    """Create 30 EvidenceItem objects across S1/S2/S3 with varying quality."""
    items = []
    scopes = ["scope_1", "scope_2_location", "scope_2_market", "scope_3"]
    categories = [
        "source_data", "emission_factor", "calculation", "assumption",
        "methodology", "boundary", "completeness", "control",
        "approval", "external_reference",
    ]
    quality_levels = ["HIGH", "MEDIUM", "LOW", "HIGH", "MEDIUM"]

    for i in range(30):
        scope = scopes[i % len(scopes)]
        category = categories[i % len(categories)]
        quality = quality_levels[i % len(quality_levels)]
        items.append({
            "evidence_id": f"EV-{i + 1:03d}",
            "scope": scope,
            "category": category,
            "title": f"Evidence Item {i + 1} - {category.replace('_', ' ').title()}",
            "description": f"Test evidence for {scope} {category}",
            "source_system": ["ERP", "Spreadsheet", "API", "Manual", "Meter"][i % 5],
            "document_ref": f"DOC-{i + 1:04d}",
            "file_hash": hashlib.sha256(f"evidence_{i}".encode()).hexdigest(),
            "quality_grade": quality,
            "completeness_pct": Decimal(str(70 + (i % 4) * 10)),
            "reporting_period": "FY2025",
            "collected_date": datetime(2025, 1 + (i % 12), 15, tzinfo=timezone.utc).isoformat(),
            "verified": i % 3 == 0,
            "linked_calculation_ids": [f"CALC-{i + 1:04d}"],
        })
    return items


# ---------------------------------------------------------------------------
# Control Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_controls() -> List[Dict[str, Any]]:
    """Create 25 Control objects (DC-01 through IT-05)."""
    control_categories = [
        ("DC", "Data Collection", 5),
        ("CA", "Calculation", 5),
        ("RV", "Review", 5),
        ("RE", "Reporting", 5),
        ("IT", "IT General", 5),
    ]
    controls = []
    idx = 0
    for prefix, category_name, count in control_categories:
        for j in range(1, count + 1):
            idx += 1
            maturity_levels = ["INITIAL", "MANAGED", "DEFINED", "MEASURED", "OPTIMISING"]
            controls.append({
                "control_id": f"{prefix}-{j:02d}",
                "control_name": f"{category_name} Control {j}",
                "category": prefix,
                "category_name": category_name,
                "description": f"Control for {category_name.lower()} process step {j}",
                "control_type": "preventive" if j % 2 == 0 else "detective",
                "frequency": ["continuous", "daily", "weekly", "monthly", "quarterly"][j % 5],
                "owner": f"control_owner_{idx}",
                "design_effective": idx <= 20,
                "operating_effective": idx <= 18,
                "maturity_level": maturity_levels[min(j - 1, 4)],
                "last_tested": datetime(2025, 3, 1, tzinfo=timezone.utc).isoformat(),
                "sample_size": 25,
                "exceptions_found": max(0, j - 3),
                "deficiency_level": "NONE" if j <= 3 else ("DEFICIENCY" if j == 4 else "SIGNIFICANT_DEFICIENCY"),
            })
    return controls


# ---------------------------------------------------------------------------
# Engagement Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_engagement() -> Dict[str, Any]:
    """Create mock engagement with verifier details."""
    return {
        "engagement_id": "ENG-2025-001",
        "verifier_name": "Big Four Assurance LLP",
        "verifier_accreditation": "UKAS",
        "lead_verifier": "Jane Smith, CPA",
        "team_size": 4,
        "assurance_standard": "ISAE_3410",
        "assurance_level": "limited",
        "scope_of_work": ["scope_1", "scope_2_location", "scope_2_market", "scope_3_cat_1"],
        "engagement_start": "2025-04-01",
        "fieldwork_start": "2025-05-15",
        "fieldwork_end": "2025-06-30",
        "report_due": "2025-07-31",
        "status": "IN_PROGRESS",
        "queries_open": 5,
        "queries_closed": 12,
        "findings_count": 3,
        "findings_critical": 0,
        "findings_major": 1,
        "findings_minor": 2,
        "sla_response_days": 5,
        "fee_estimate_usd": Decimal("75000"),
        "fee_actual_usd": Decimal("68000"),
    }


# ---------------------------------------------------------------------------
# Emissions Data Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_emissions_data() -> Dict[str, Any]:
    """Create Scope 1+2+3 emissions for materiality calculation."""
    return {
        "organisation_id": "org-test-001",
        "reporting_year": 2025,
        "scope_1": {
            "total_tco2e": Decimal("5000"),
            "stationary_combustion": Decimal("3200"),
            "mobile_combustion": Decimal("1200"),
            "process_emissions": Decimal("400"),
            "fugitive_emissions": Decimal("200"),
        },
        "scope_2_location": {
            "total_tco2e": Decimal("3000"),
            "purchased_electricity": Decimal("2500"),
            "purchased_steam": Decimal("500"),
        },
        "scope_2_market": {
            "total_tco2e": Decimal("2500"),
            "purchased_electricity": Decimal("2000"),
            "purchased_steam": Decimal("500"),
        },
        "scope_3": {
            "total_tco2e": Decimal("15000"),
            "cat_1_purchased_goods": Decimal("8000"),
            "cat_4_upstream_transport": Decimal("3000"),
            "cat_6_business_travel": Decimal("1500"),
            "cat_7_employee_commuting": Decimal("500"),
            "cat_11_use_of_sold_products": Decimal("2000"),
        },
        "total_all_scopes_tco2e": Decimal("23000"),
        "total_s1_s2_location_tco2e": Decimal("8000"),
        "total_s1_s2_market_tco2e": Decimal("7500"),
        "base_year_total_tco2e": Decimal("25000"),
    }


# ---------------------------------------------------------------------------
# Jurisdiction Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_jurisdictions() -> List[Dict[str, Any]]:
    """Create 12 jurisdiction requirement records."""
    jurisdictions = [
        {
            "jurisdiction_id": "EU_CSRD",
            "name": "EU Corporate Sustainability Reporting Directive",
            "country": "EU",
            "assurance_required": True,
            "assurance_level_2025": "limited",
            "assurance_level_2028": "reasonable",
            "standard": "ISAE_3410",
            "effective_date": "2024-01-01",
            "company_threshold_employees": 250,
            "company_threshold_revenue_eur": Decimal("40000000"),
            "scope_coverage": ["scope_1", "scope_2", "scope_3"],
            "penalties": True,
        },
        {
            "jurisdiction_id": "US_SEC",
            "name": "US SEC Climate Disclosure Rules",
            "country": "US",
            "assurance_required": True,
            "assurance_level_2025": "limited",
            "assurance_level_2028": "reasonable",
            "standard": "SSAE_18",
            "effective_date": "2025-01-01",
            "company_threshold_employees": 0,
            "company_threshold_revenue_eur": Decimal("0"),
            "scope_coverage": ["scope_1", "scope_2"],
            "penalties": True,
        },
        {
            "jurisdiction_id": "CA_SB253",
            "name": "California SB 253 Climate Corporate Data Accountability Act",
            "country": "US",
            "assurance_required": True,
            "assurance_level_2025": "limited",
            "assurance_level_2028": "reasonable",
            "standard": "ISO_14064_3",
            "effective_date": "2026-01-01",
            "company_threshold_employees": 0,
            "company_threshold_revenue_eur": Decimal("1000000000"),
            "scope_coverage": ["scope_1", "scope_2", "scope_3"],
            "penalties": True,
        },
        {
            "jurisdiction_id": "UK_SECR",
            "name": "UK Streamlined Energy and Carbon Reporting",
            "country": "GB",
            "assurance_required": False,
            "assurance_level_2025": "none",
            "assurance_level_2028": "limited",
            "standard": "ISAE_3000",
            "effective_date": "2019-04-01",
            "company_threshold_employees": 250,
            "company_threshold_revenue_eur": Decimal("36000000"),
            "scope_coverage": ["scope_1", "scope_2"],
            "penalties": True,
        },
        {
            "jurisdiction_id": "SG_SGX",
            "name": "Singapore SGX Climate Reporting",
            "country": "SG",
            "assurance_required": True,
            "assurance_level_2025": "limited",
            "assurance_level_2028": "limited",
            "standard": "ISAE_3410",
            "effective_date": "2025-01-01",
            "company_threshold_employees": 0,
            "company_threshold_revenue_eur": Decimal("0"),
            "scope_coverage": ["scope_1", "scope_2"],
            "penalties": True,
        },
        {
            "jurisdiction_id": "JP_SSBJ",
            "name": "Japan Sustainability Standards Board",
            "country": "JP",
            "assurance_required": True,
            "assurance_level_2025": "limited",
            "assurance_level_2028": "reasonable",
            "standard": "ISAE_3410",
            "effective_date": "2025-04-01",
            "company_threshold_employees": 0,
            "company_threshold_revenue_eur": Decimal("0"),
            "scope_coverage": ["scope_1", "scope_2", "scope_3"],
            "penalties": True,
        },
        {
            "jurisdiction_id": "AU_ASRS",
            "name": "Australia Sustainability Reporting Standards",
            "country": "AU",
            "assurance_required": True,
            "assurance_level_2025": "limited",
            "assurance_level_2028": "reasonable",
            "standard": "ISAE_3410",
            "effective_date": "2025-01-01",
            "company_threshold_employees": 100,
            "company_threshold_revenue_eur": Decimal("50000000"),
            "scope_coverage": ["scope_1", "scope_2"],
            "penalties": True,
        },
        {
            "jurisdiction_id": "KR_KSQF",
            "name": "South Korea Sustainability Quality Framework",
            "country": "KR",
            "assurance_required": True,
            "assurance_level_2025": "limited",
            "assurance_level_2028": "limited",
            "standard": "ISAE_3000",
            "effective_date": "2025-01-01",
            "company_threshold_employees": 500,
            "company_threshold_revenue_eur": Decimal("200000000"),
            "scope_coverage": ["scope_1", "scope_2"],
            "penalties": True,
        },
        {
            "jurisdiction_id": "HK_HKEX",
            "name": "Hong Kong HKEX Climate Disclosure",
            "country": "HK",
            "assurance_required": False,
            "assurance_level_2025": "none",
            "assurance_level_2028": "limited",
            "standard": "ISAE_3410",
            "effective_date": "2025-01-01",
            "company_threshold_employees": 0,
            "company_threshold_revenue_eur": Decimal("0"),
            "scope_coverage": ["scope_1", "scope_2"],
            "penalties": False,
        },
        {
            "jurisdiction_id": "BR_CVM",
            "name": "Brazil CVM Climate Reporting",
            "country": "BR",
            "assurance_required": True,
            "assurance_level_2025": "limited",
            "assurance_level_2028": "limited",
            "standard": "ISO_14064_3",
            "effective_date": "2026-01-01",
            "company_threshold_employees": 0,
            "company_threshold_revenue_eur": Decimal("0"),
            "scope_coverage": ["scope_1", "scope_2"],
            "penalties": True,
        },
        {
            "jurisdiction_id": "IN_BRSR",
            "name": "India BRSR Core Assurance",
            "country": "IN",
            "assurance_required": True,
            "assurance_level_2025": "limited",
            "assurance_level_2028": "reasonable",
            "standard": "ISAE_3000",
            "effective_date": "2024-04-01",
            "company_threshold_employees": 0,
            "company_threshold_revenue_eur": Decimal("0"),
            "scope_coverage": ["scope_1", "scope_2"],
            "penalties": True,
        },
        {
            "jurisdiction_id": "CA_CSSB",
            "name": "Canada CSSB Sustainability Standards",
            "country": "CA",
            "assurance_required": True,
            "assurance_level_2025": "none",
            "assurance_level_2028": "limited",
            "standard": "ISAE_3410",
            "effective_date": "2027-01-01",
            "company_threshold_employees": 0,
            "company_threshold_revenue_eur": Decimal("0"),
            "scope_coverage": ["scope_1", "scope_2"],
            "penalties": True,
        },
    ]
    return jurisdictions


# ---------------------------------------------------------------------------
# ISAE 3410 Checklist Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_checklist() -> List[Dict[str, Any]]:
    """Create ISAE 3410 checklist items (80 items across 10 categories)."""
    categories = [
        ("GOV", "Governance & Oversight"),
        ("BND", "Organisational Boundary"),
        ("SRC", "Source Data Management"),
        ("EF", "Emission Factor Selection"),
        ("CAL", "Calculation Methodology"),
        ("QC", "Quality Control"),
        ("DOC", "Documentation & Records"),
        ("REP", "Reporting & Disclosure"),
        ("CHG", "Change Management"),
        ("IT", "IT Systems & Controls"),
    ]
    items = []
    item_idx = 0
    for cat_code, cat_name in categories:
        for j in range(1, 9):  # 8 items per category = 80 total
            item_idx += 1
            is_mandatory = j <= 5
            is_met = item_idx <= 60
            items.append({
                "item_id": f"{cat_code}-{j:02d}",
                "category_code": cat_code,
                "category_name": cat_name,
                "title": f"{cat_name} Requirement {j}",
                "description": f"ISAE 3410 requirement for {cat_name.lower()} item {j}",
                "mandatory": is_mandatory,
                "weight": Decimal("1.5") if is_mandatory else Decimal("1.0"),
                "status": "MET" if is_met else ("PARTIALLY_MET" if item_idx <= 70 else "NOT_MET"),
                "evidence_ref": f"EV-{item_idx:03d}" if is_met else None,
                "notes": f"Assessment note for {cat_code}-{j:02d}",
                "assessor": "QA Team",
                "assessed_date": "2025-03-15",
            })
    return items


# ---------------------------------------------------------------------------
# Engine Config Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def evidence_engine_config() -> Dict[str, Any]:
    """Configuration dict for EvidenceConsolidationEngine."""
    return {
        "evidence_categories": [
            "source_data", "emission_factor", "calculation", "assumption",
            "methodology", "boundary", "completeness", "control",
            "approval", "external_reference",
        ],
        "quality_levels": ["HIGH", "MEDIUM", "LOW"],
        "package_versions": ["DRAFT", "REVIEW", "FINAL"],
        "hash_algorithm": "SHA-256",
        "min_completeness_pct": Decimal("80"),
    }


@pytest.fixture
def readiness_engine_config() -> Dict[str, Any]:
    """Configuration dict for ReadinessAssessmentEngine."""
    return {
        "standard": "ISAE_3410",
        "checklist_items": 80,
        "categories": 10,
        "readiness_thresholds": {
            "ready": Decimal("90"),
            "mostly_ready": Decimal("70"),
            "partially_ready": Decimal("40"),
            "not_ready": Decimal("0"),
        },
        "mandatory_gate_items": True,
    }


@pytest.fixture
def provenance_engine_config() -> Dict[str, Any]:
    """Configuration dict for CalculationProvenanceEngine."""
    return {
        "hash_algorithm": "SHA-256",
        "chain_validation": True,
        "tamper_detection": True,
        "min_chain_length": 1,
    }


@pytest.fixture
def control_engine_config() -> Dict[str, Any]:
    """Configuration dict for ControlTestingEngine."""
    return {
        "control_categories": ["DC", "CA", "RV", "RE", "IT"],
        "controls_per_category": 5,
        "total_controls": 25,
        "deficiency_levels": ["NONE", "DEFICIENCY", "SIGNIFICANT_DEFICIENCY"],
        "maturity_levels": ["INITIAL", "MANAGED", "DEFINED", "MEASURED", "OPTIMISING"],
        "default_sample_size": 25,
    }


@pytest.fixture
def verifier_engine_config() -> Dict[str, Any]:
    """Configuration dict for VerifierCollaborationEngine."""
    return {
        "query_statuses": ["OPEN", "IN_PROGRESS", "RESPONDED", "ACCEPTED", "CLOSED"],
        "finding_types": ["observation", "non_conformity", "opportunity", "recommendation", "exception"],
        "severity_levels": ["INFO", "LOW", "MEDIUM", "HIGH"],
        "default_sla_days": 5,
        "escalation_threshold_days": 10,
    }


@pytest.fixture
def materiality_engine_config() -> Dict[str, Any]:
    """Configuration dict for MaterialityAssessmentEngine."""
    return {
        "overall_pct": Decimal("5"),
        "performance_pct_of_overall": Decimal("75"),
        "clearly_trivial_pct_of_overall": Decimal("5"),
        "scope_specific_enabled": True,
        "qualitative_factors": [
            "regulatory_scrutiny",
            "stakeholder_sensitivity",
            "restatement_history",
            "management_bias",
        ],
    }


@pytest.fixture
def sampling_engine_config() -> Dict[str, Any]:
    """Configuration dict for SamplingPlanEngine."""
    return {
        "method": "MUS",
        "confidence_level_reasonable": Decimal("0.95"),
        "confidence_level_limited": Decimal("0.80"),
        "tolerable_misstatement_pct": Decimal("5"),
        "expected_misstatement_pct": Decimal("1"),
        "high_value_threshold_pct": Decimal("50"),
    }


@pytest.fixture
def regulatory_engine_config() -> Dict[str, Any]:
    """Configuration dict for RegulatoryRequirementEngine."""
    return {
        "jurisdictions_count": 12,
        "default_standard": "ISAE_3410",
        "alert_upcoming_days": 180,
    }


@pytest.fixture
def cost_engine_config() -> Dict[str, Any]:
    """Configuration dict for CostTimelineEngine."""
    return {
        "company_sizes": ["MICRO", "SMALL", "MEDIUM", "LARGE", "ENTERPRISE", "MEGA"],
        "reasonable_multiplier": Decimal("2.5"),
        "multi_jurisdiction_uplift_pct": Decimal("15"),
        "first_time_premium_pct": Decimal("25"),
        "scope_3_complexity_uplift_pct": Decimal("20"),
    }


@pytest.fixture
def reporting_engine_config() -> Dict[str, Any]:
    """Configuration dict for AssuranceReportingEngine."""
    return {
        "output_formats": ["markdown", "html", "json", "xbrl"],
        "include_provenance": True,
        "include_methodology": True,
        "company_name": "ACME Corp",
        "reporting_period": "FY2025",
    }


# ---------------------------------------------------------------------------
# Workflow Input Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_workflow_input(sample_config, sample_evidence_items) -> Dict[str, Any]:
    """Create sample workflow execution input."""
    return {
        "organisation_id": "org-test-001",
        "organisation_name": "ACME Corp",
        "sector": "INDUSTRIALS",
        "reporting_year": 2025,
        "base_year": 2020,
        "evidence_items": sample_evidence_items[:10],
        "config": sample_config,
    }


# ---------------------------------------------------------------------------
# Pipeline / Orchestrator Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_pipeline_config() -> Dict[str, Any]:
    """Create sample pipeline configuration for orchestrator tests."""
    return {
        "company_name": "ACME Corp",
        "reporting_period": "2025",
        "sector": "INDUSTRIALS",
        "scope_boundary": "scope_1_2_3",
        "max_retries": 2,
        "retry_base_delay_s": 0.01,
        "enable_parallel": False,
        "timeout_per_phase_s": 30.0,
        "assurance_level": "limited",
        "assurance_standard": "ISAE_3410",
        "enable_evidence_collection": True,
        "enable_readiness_assessment": True,
        "enable_provenance_verification": True,
        "enable_control_testing": True,
        "enable_verifier_collaboration": True,
        "enable_materiality_sampling": True,
        "enable_regulatory_compliance": True,
    }
