# -*- coding: utf-8 -*-
"""
Shared pytest fixtures for GL-SBTi-APP v1.0 test suite.

Provides reusable fixtures for configuration, organizations, emissions
inventories, targets (near-term, long-term, net-zero, FLAG), pathways,
validation results, Scope 3 screening, FLAG assessments, progress records,
temperature scores, FI portfolios, recalculations, five-year reviews,
and mock engine instances used across all 16 test modules.

Author: GL-TestEngineer
Date: March 2026
"""

import sys
import os
from datetime import date, datetime, timedelta
from decimal import Decimal
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest

# ---------------------------------------------------------------------------
# Path setup -- ensure the SBTi services package is importable
# ---------------------------------------------------------------------------
_SERVICES_DIR = os.path.join(
    os.path.dirname(__file__),
    "..", "..", "..", "..",
    "applications", "GL-SBTi-APP", "SBTi-Target-Platform",
)
_SERVICES_DIR = os.path.normpath(_SERVICES_DIR)
if _SERVICES_DIR not in sys.path:
    sys.path.insert(0, _SERVICES_DIR)


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def _new_id() -> str:
    """Generate a unique UUID string."""
    return str(uuid4())


def _now() -> datetime:
    """Return current UTC timestamp without microseconds."""
    return datetime.utcnow().replace(microsecond=0)


def _sha256(data: str) -> str:
    """Generate SHA-256 hex digest."""
    import hashlib
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


# ============================================================================
# CONFIGURATION FIXTURES
# ============================================================================

@pytest.fixture
def sample_config() -> Dict[str, Any]:
    """Default SBTi application configuration."""
    return {
        "app_name": "GL-SBTi-APP",
        "app_version": "1.0.0",
        "sbti_framework_version": "2.1",
        "reporting_year": 2025,
        "default_ambition": "1.5C",
        "scope3_threshold_pct": 40.0,
        "near_term_min_years": 5,
        "near_term_max_years": 10,
        "long_term_target_year": 2050,
        "scope1_2_coverage_min_pct": 95.0,
        "scope3_coverage_min_pct": 67.0,
        "scope3_long_term_coverage_min_pct": 90.0,
        "flag_threshold_pct": 20.0,
        "recalculation_threshold_pct": 5.0,
        "review_cycle_years": 5,
        "aca_annual_rate_1_5c": 4.2,
        "aca_annual_rate_wb2c": 2.5,
        "flag_sector_annual_rate": 3.03,
        "flag_long_term_reduction_pct": 72.0,
        "fi_coverage_target_year": 2040,
        "fi_coverage_target_pct": 100.0,
        "temperature_max_c": 4.0,
        "temperature_min_c": 1.2,
        "environment": "test",
    }


# ============================================================================
# ORGANIZATION FIXTURES
# ============================================================================

@pytest.fixture
def sample_org_id() -> str:
    """Stable organization ID for cross-fixture referencing."""
    return _new_id()


@pytest.fixture
def sample_organization(sample_org_id) -> Dict[str, Any]:
    """Sample organization with sector classification data."""
    return {
        "id": sample_org_id,
        "tenant_id": _new_id(),
        "name": "Acme Manufacturing Corp",
        "sector": "manufacturing",
        "industry": "Cement Production",
        "isic_code": "2394",
        "nace_code": "C23.51",
        "naics_code": "327310",
        "oecd_member": True,
        "country": "US",
        "region": "North America",
        "reporting_currency": "USD",
        "fiscal_year_end": "12-31",
        "employee_count": 12500,
        "annual_revenue_usd": 4_500_000_000,
        "sbti_status": "committed",
        "commitment_date": date(2024, 3, 15),
        "created_at": _now(),
        "updated_at": _now(),
    }


@pytest.fixture
def financial_organization() -> Dict[str, Any]:
    """Financial services organization for FI-specific tests."""
    return {
        "id": _new_id(),
        "tenant_id": _new_id(),
        "name": "Global Finance Holdings",
        "sector": "financial_services",
        "industry": "Commercial Banking",
        "isic_code": "6419",
        "nace_code": "K64.19",
        "naics_code": "522110",
        "oecd_member": True,
        "country": "UK",
        "region": "Europe",
        "reporting_currency": "GBP",
        "fiscal_year_end": "03-31",
        "sbti_status": "committed",
        "commitment_date": date(2024, 6, 1),
    }


@pytest.fixture
def flag_organization() -> Dict[str, Any]:
    """FLAG-sector organization for forestry/agriculture tests."""
    return {
        "id": _new_id(),
        "tenant_id": _new_id(),
        "name": "Agro Commodities Inc",
        "sector": "food_agriculture",
        "industry": "Agricultural Products",
        "isic_code": "0111",
        "nace_code": "A01.11",
        "naics_code": "111110",
        "oecd_member": False,
        "country": "BR",
        "region": "South America",
        "reporting_currency": "BRL",
        "sbti_status": "committed",
        "commitment_date": date(2024, 9, 1),
    }


# ============================================================================
# EMISSIONS INVENTORY FIXTURES
# ============================================================================

@pytest.fixture
def sample_emissions_inventory(sample_org_id) -> Dict[str, Any]:
    """Sample emissions inventory with all scopes, FLAG, and bioenergy."""
    return {
        "id": _new_id(),
        "org_id": sample_org_id,
        "reporting_year": 2024,
        "base_year": 2020,
        "scope1_tco2e": 50_000.0,
        "scope2_location_tco2e": 30_000.0,
        "scope2_market_tco2e": 25_000.0,
        "scope3_tco2e": 120_000.0,
        "scope3_categories": {
            1: 35_000.0, 2: 5_000.0, 3: 8_000.0, 4: 12_000.0,
            5: 3_000.0, 6: 4_500.0, 7: 2_500.0, 8: 1_500.0,
            9: 15_000.0, 10: 6_000.0, 11: 18_000.0, 12: 4_000.0,
            13: 2_000.0, 14: 1_500.0, 15: 1_500.0,
        },
        "total_s123_tco2e": 195_000.0,
        "scope3_pct_of_total": 61.54,
        "flag_emissions_tco2e": 8_000.0,
        "flag_pct_of_total": 4.1,
        "bioenergy_tco2e": 3_500.0,
        "bioenergy_included": True,
        "excluded_sources": [],
        "exclusion_justification": None,
        "data_quality_score": 3.8,
        "verification_status": "third_party_verified",
        "verification_body": "Bureau Veritas",
        "provenance_hash": _sha256(f"{sample_org_id}_2024_inventory"),
        "created_at": _now(),
    }


@pytest.fixture
def high_scope3_inventory(sample_org_id) -> Dict[str, Any]:
    """Inventory where Scope 3 >= 40% (triggers Scope 3 target requirement)."""
    return {
        "id": _new_id(),
        "org_id": sample_org_id,
        "reporting_year": 2024,
        "scope1_tco2e": 20_000.0,
        "scope2_market_tco2e": 15_000.0,
        "scope3_tco2e": 65_000.0,
        "total_s123_tco2e": 100_000.0,
        "scope3_pct_of_total": 65.0,
    }


@pytest.fixture
def low_scope3_inventory(sample_org_id) -> Dict[str, Any]:
    """Inventory where Scope 3 < 40% (no mandatory Scope 3 target)."""
    return {
        "id": _new_id(),
        "org_id": sample_org_id,
        "reporting_year": 2024,
        "scope1_tco2e": 70_000.0,
        "scope2_market_tco2e": 20_000.0,
        "scope3_tco2e": 10_000.0,
        "total_s123_tco2e": 100_000.0,
        "scope3_pct_of_total": 10.0,
    }


@pytest.fixture
def high_flag_inventory(sample_org_id) -> Dict[str, Any]:
    """Inventory where FLAG >= 20% of total (triggers FLAG pathway)."""
    return {
        "id": _new_id(),
        "org_id": sample_org_id,
        "reporting_year": 2024,
        "scope1_tco2e": 30_000.0,
        "scope2_market_tco2e": 10_000.0,
        "scope3_tco2e": 60_000.0,
        "total_s123_tco2e": 100_000.0,
        "flag_emissions_tco2e": 25_000.0,
        "flag_pct_of_total": 25.0,
    }


# ============================================================================
# TARGET FIXTURES
# ============================================================================

@pytest.fixture
def sample_near_term_target(sample_org_id) -> Dict[str, Any]:
    """Near-term ACA 1.5C target with 95% S1+2 coverage."""
    return {
        "id": _new_id(),
        "org_id": sample_org_id,
        "target_name": "Scope 1+2 Near-Term 1.5C Aligned",
        "target_type": "near_term",
        "scope": "scope_1_2",
        "method": "absolute",
        "ambition_level": "1.5C",
        "base_year": 2020,
        "base_year_emissions_tco2e": 80_000.0,
        "target_year": 2030,
        "reduction_pct": 42.0,
        "boundary_coverage_pct": 95.0,
        "linear_annual_reduction_pct": 4.2,
        "scope3_categories_included": None,
        "pathway_id": None,
        "status": "draft",
        "intensity_metric": None,
        "notes": None,
        "provenance_hash": _sha256(f"{sample_org_id}_nt_target"),
        "created_at": _now(),
        "updated_at": _now(),
    }


@pytest.fixture
def sample_long_term_target(sample_org_id) -> Dict[str, Any]:
    """Long-term target for 90% reduction by 2050."""
    return {
        "id": _new_id(),
        "org_id": sample_org_id,
        "target_name": "Long-Term 90% Reduction by 2050",
        "target_type": "long_term",
        "scope": "all_scopes",
        "method": "absolute",
        "ambition_level": "1.5C",
        "base_year": 2020,
        "base_year_emissions_tco2e": 195_000.0,
        "target_year": 2050,
        "reduction_pct": 90.0,
        "boundary_coverage_pct": 95.0,
        "linear_annual_reduction_pct": 3.0,
        "scope3_categories_included": [1, 2, 3, 4, 5, 6, 7, 9, 11, 12],
        "pathway_id": None,
        "status": "draft",
        "provenance_hash": _sha256(f"{sample_org_id}_lt_target"),
        "created_at": _now(),
        "updated_at": _now(),
    }


@pytest.fixture
def sample_net_zero_target(sample_org_id) -> Dict[str, Any]:
    """Net-zero target with residual neutralization plan."""
    return {
        "id": _new_id(),
        "org_id": sample_org_id,
        "target_name": "Net-Zero by 2050",
        "target_type": "net_zero",
        "scope": "all_scopes",
        "method": "absolute",
        "ambition_level": "1.5C",
        "base_year": 2020,
        "base_year_emissions_tco2e": 195_000.0,
        "target_year": 2050,
        "reduction_pct": 90.0,
        "boundary_coverage_pct": 95.0,
        "linear_annual_reduction_pct": 3.0,
        "residual_emissions_tco2e": 19_500.0,
        "neutralization_strategy": "permanent_carbon_removal",
        "neutralization_mechanisms": ["DACCS", "BECCS", "biochar"],
        "interim_target_id": None,
        "status": "draft",
        "provenance_hash": _sha256(f"{sample_org_id}_nz_target"),
        "created_at": _now(),
        "updated_at": _now(),
    }


@pytest.fixture
def sample_scope3_target(sample_org_id) -> Dict[str, Any]:
    """Scope 3 near-term target with category coverage."""
    return {
        "id": _new_id(),
        "org_id": sample_org_id,
        "target_name": "Scope 3 WB2C Aligned",
        "target_type": "near_term",
        "scope": "scope_3",
        "method": "absolute",
        "ambition_level": "well_below_2C",
        "base_year": 2020,
        "base_year_emissions_tco2e": 120_000.0,
        "target_year": 2030,
        "reduction_pct": 25.0,
        "boundary_coverage_pct": 67.0,
        "linear_annual_reduction_pct": 2.5,
        "scope3_categories_included": [1, 3, 4, 9, 11],
        "status": "draft",
        "provenance_hash": _sha256(f"{sample_org_id}_s3_target"),
        "created_at": _now(),
        "updated_at": _now(),
    }


@pytest.fixture
def sample_flag_target(sample_org_id) -> Dict[str, Any]:
    """FLAG sector target with commodity pathway and deforestation commitment."""
    return {
        "id": _new_id(),
        "org_id": sample_org_id,
        "target_name": "FLAG Sector Target - Commodity Pathway",
        "target_type": "near_term",
        "scope": "scope_1_2",
        "method": "absolute",
        "ambition_level": "1.5C",
        "base_year": 2020,
        "base_year_emissions_tco2e": 25_000.0,
        "target_year": 2030,
        "reduction_pct": 30.3,
        "boundary_coverage_pct": 95.0,
        "linear_annual_reduction_pct": 3.03,
        "is_flag_target": True,
        "flag_pathway_type": "commodity",
        "flag_commodity": "cattle",
        "deforestation_commitment": True,
        "deforestation_target_date": date(2025, 12, 31),
        "removals_target_tco2e": 5_000.0,
        "status": "draft",
        "provenance_hash": _sha256(f"{sample_org_id}_flag_target"),
        "created_at": _now(),
        "updated_at": _now(),
    }


@pytest.fixture
def sample_intensity_target(sample_org_id) -> Dict[str, Any]:
    """Physical intensity target for sector pathway."""
    return {
        "id": _new_id(),
        "org_id": sample_org_id,
        "target_name": "Cement Intensity Reduction",
        "target_type": "near_term",
        "scope": "scope_1_2",
        "method": "intensity_physical",
        "ambition_level": "1.5C",
        "base_year": 2020,
        "base_year_emissions_tco2e": 50_000.0,
        "target_year": 2030,
        "reduction_pct": 35.0,
        "intensity_metric": "tCO2e per tonne cement",
        "base_year_intensity": 0.62,
        "target_intensity": 0.403,
        "status": "draft",
    }


# ============================================================================
# PATHWAY FIXTURES
# ============================================================================

@pytest.fixture
def sample_pathway(sample_org_id) -> Dict[str, Any]:
    """ACA 1.5C decarbonization pathway with milestones."""
    base_emissions = 80_000.0
    annual_rate = 4.2
    milestones = {}
    for year in range(2021, 2031):
        years_elapsed = year - 2020
        reduction = base_emissions * (annual_rate / 100.0) * years_elapsed
        milestones[year] = round(base_emissions - reduction, 1)

    return {
        "id": _new_id(),
        "org_id": sample_org_id,
        "pathway_name": "ACA 1.5C Linear Pathway",
        "pathway_type": "aca",
        "ambition_level": "1.5C",
        "base_year": 2020,
        "target_year": 2030,
        "base_emissions_tco2e": base_emissions,
        "annual_reduction_rate_pct": annual_rate,
        "milestones": milestones,
        "uncertainty_lower_pct": 3.5,
        "uncertainty_upper_pct": 5.0,
        "confidence_level": 0.9,
        "provenance_hash": _sha256(f"{sample_org_id}_pathway"),
        "created_at": _now(),
    }


@pytest.fixture
def sda_pathway(sample_org_id) -> Dict[str, Any]:
    """SDA sector-specific convergence pathway for cement."""
    return {
        "id": _new_id(),
        "org_id": sample_org_id,
        "pathway_name": "SDA Cement Convergence",
        "pathway_type": "sda",
        "sector": "cement",
        "base_year": 2020,
        "target_year": 2050,
        "base_intensity": 0.62,
        "convergence_intensity_2050": 0.10,
        "intensity_unit": "tCO2e per tonne cement",
        "milestones": {
            2025: 0.52, 2030: 0.42, 2035: 0.32,
            2040: 0.22, 2045: 0.15, 2050: 0.10,
        },
        "provenance_hash": _sha256(f"{sample_org_id}_sda_pathway"),
    }


@pytest.fixture
def flag_commodity_pathway() -> Dict[str, Any]:
    """FLAG commodity-specific pathway for cattle."""
    return {
        "id": _new_id(),
        "pathway_name": "FLAG Cattle Commodity Pathway",
        "pathway_type": "flag_commodity",
        "commodity": "cattle",
        "base_year": 2020,
        "target_year": 2030,
        "annual_reduction_rate_pct": 3.03,
        "long_term_reduction_pct": 72.0,
        "long_term_target_year": 2050,
        "includes_removals": True,
        "deforestation_free_date": date(2025, 12, 31),
    }


# ============================================================================
# VALIDATION RESULT FIXTURES
# ============================================================================

@pytest.fixture
def sample_validation_result(sample_org_id) -> Dict[str, Any]:
    """Full SBTi validation result with criteria statuses."""
    return {
        "id": _new_id(),
        "org_id": sample_org_id,
        "validation_date": _now(),
        "sbti_framework_version": "2.1",
        "criteria_results": {
            "C1_org_boundary": {"status": "pass", "details": "Operational control boundary defined"},
            "C2_ghg_gases": {"status": "pass", "details": "All 7 GHGs included"},
            "C3_coverage": {"status": "pass", "details": "95.2% S1+2 coverage"},
            "C4_base_year": {"status": "pass", "details": "Base year 2020 >= 2015"},
            "C5_timeframe": {"status": "pass", "details": "10 year timeframe within 5-10 range"},
            "C6_ambition_s12": {"status": "pass", "details": "4.2%/yr meets 1.5C minimum"},
            "C7_ambition_s3": {"status": "pass", "details": "2.5%/yr meets WB2C minimum"},
            "C8_scope3_trigger": {"status": "pass", "details": "Scope 3 = 61.5% >= 40% threshold"},
            "C9_scope3_coverage": {"status": "pass", "details": "72% S3 coverage >= 67% minimum"},
            "C10_bioenergy": {"status": "pass", "details": "Bioenergy emissions included"},
            "C11_carbon_credits": {"status": "pass", "details": "No carbon credits counted toward target"},
            "C12_avoided_emissions": {"status": "pass", "details": "Avoided emissions reported separately"},
        },
        "overall_status": "pass",
        "pass_count": 12,
        "fail_count": 0,
        "warning_count": 0,
        "total_criteria": 12,
        "readiness_score": 100.0,
        "readiness_level": "ready_for_submission",
        "recommendations": [],
        "provenance_hash": _sha256(f"{sample_org_id}_validation"),
        "created_at": _now(),
    }


@pytest.fixture
def failing_validation_result(sample_org_id) -> Dict[str, Any]:
    """Validation result with failures."""
    return {
        "id": _new_id(),
        "org_id": sample_org_id,
        "validation_date": _now(),
        "criteria_results": {
            "C1_org_boundary": {"status": "pass", "details": "OK"},
            "C2_ghg_gases": {"status": "fail", "details": "NF3 not included"},
            "C3_coverage": {"status": "fail", "details": "85% S1+2 < 95% minimum"},
            "C4_base_year": {"status": "pass", "details": "OK"},
            "C5_timeframe": {"status": "fail", "details": "12 year timeframe exceeds 10 year max"},
            "C6_ambition_s12": {"status": "fail", "details": "3.0%/yr below 4.2% for 1.5C"},
        },
        "overall_status": "fail",
        "pass_count": 2,
        "fail_count": 4,
        "readiness_score": 33.3,
        "readiness_level": "significant_gaps",
    }


# ============================================================================
# SCOPE 3 SCREENING FIXTURES
# ============================================================================

@pytest.fixture
def sample_scope3_screening(sample_org_id) -> Dict[str, Any]:
    """Scope 3 screening result with category breakdown."""
    return {
        "id": _new_id(),
        "org_id": sample_org_id,
        "reporting_year": 2024,
        "total_scope3_tco2e": 120_000.0,
        "total_s123_tco2e": 195_000.0,
        "scope3_pct_of_total": 61.54,
        "trigger_threshold_pct": 40.0,
        "scope3_target_required": True,
        "category_breakdown": {
            1: {"tco2e": 35_000, "pct": 29.2, "data_quality": 3},
            2: {"tco2e": 5_000, "pct": 4.2, "data_quality": 2},
            3: {"tco2e": 8_000, "pct": 6.7, "data_quality": 4},
            4: {"tco2e": 12_000, "pct": 10.0, "data_quality": 3},
            5: {"tco2e": 3_000, "pct": 2.5, "data_quality": 2},
            6: {"tco2e": 4_500, "pct": 3.75, "data_quality": 3},
            7: {"tco2e": 2_500, "pct": 2.1, "data_quality": 2},
            8: {"tco2e": 1_500, "pct": 1.25, "data_quality": 1},
            9: {"tco2e": 15_000, "pct": 12.5, "data_quality": 3},
            10: {"tco2e": 6_000, "pct": 5.0, "data_quality": 2},
            11: {"tco2e": 18_000, "pct": 15.0, "data_quality": 3},
            12: {"tco2e": 4_000, "pct": 3.3, "data_quality": 2},
            13: {"tco2e": 2_000, "pct": 1.7, "data_quality": 1},
            14: {"tco2e": 1_500, "pct": 1.25, "data_quality": 1},
            15: {"tco2e": 1_500, "pct": 1.25, "data_quality": 2},
        },
        "hotspot_categories": [1, 11, 9, 4],
        "min_coverage_pct": 67.0,
        "recommended_categories": [1, 3, 4, 9, 11],
        "recommended_coverage_pct": 73.4,
        "provenance_hash": _sha256(f"{sample_org_id}_s3_screen"),
    }


# ============================================================================
# FLAG ASSESSMENT FIXTURES
# ============================================================================

@pytest.fixture
def sample_flag_assessment(sample_org_id) -> Dict[str, Any]:
    """FLAG assessment for an agricultural organization."""
    return {
        "id": _new_id(),
        "org_id": sample_org_id,
        "reporting_year": 2024,
        "total_emissions_tco2e": 100_000.0,
        "flag_emissions_tco2e": 25_000.0,
        "flag_pct_of_total": 25.0,
        "flag_threshold_pct": 20.0,
        "flag_target_required": True,
        "flag_sector_classification": "food_agriculture",
        "flag_sub_sectors": ["crop_production", "livestock"],
        "commodity_breakdown": {
            "cattle": {"tco2e": 12_000, "pct": 48.0},
            "soy": {"tco2e": 5_000, "pct": 20.0},
            "palm_oil": {"tco2e": 3_000, "pct": 12.0},
            "timber": {"tco2e": 2_500, "pct": 10.0},
            "other": {"tco2e": 2_500, "pct": 10.0},
        },
        "deforestation_commitment": True,
        "deforestation_target_date": date(2025, 12, 31),
        "removals_tco2e": 5_000.0,
        "net_flag_emissions_tco2e": 20_000.0,
        "pathway_type": "commodity",
        "near_term_rate_pct": 3.03,
        "long_term_reduction_pct": 72.0,
        "provenance_hash": _sha256(f"{sample_org_id}_flag_assess"),
    }


@pytest.fixture
def non_flag_assessment(sample_org_id) -> Dict[str, Any]:
    """FLAG assessment for a non-FLAG organization (< 20%)."""
    return {
        "id": _new_id(),
        "org_id": sample_org_id,
        "total_emissions_tco2e": 200_000.0,
        "flag_emissions_tco2e": 10_000.0,
        "flag_pct_of_total": 5.0,
        "flag_threshold_pct": 20.0,
        "flag_target_required": False,
    }


# ============================================================================
# PROGRESS RECORD FIXTURES
# ============================================================================

@pytest.fixture
def sample_progress_record(sample_org_id) -> Dict[str, Any]:
    """Annual progress tracking record."""
    return {
        "id": _new_id(),
        "org_id": sample_org_id,
        "target_id": _new_id(),
        "reporting_year": 2024,
        "actual_emissions_tco2e": 70_000.0,
        "expected_emissions_tco2e": 66_400.0,
        "base_year_emissions_tco2e": 80_000.0,
        "cumulative_reduction_pct": 12.5,
        "expected_reduction_pct": 17.0,
        "variance_pct": -4.5,
        "variance_tco2e": 3_600.0,
        "on_track": False,
        "rag_status": "amber",
        "scope_breakdown": {
            "scope_1": {"actual": 42_000, "expected": 40_000},
            "scope_2": {"actual": 28_000, "expected": 26_400},
        },
        "projection_target_year_tco2e": 52_000.0,
        "projected_achievement_pct": 65.0,
        "notes": "Behind target due to increased production volume",
        "provenance_hash": _sha256(f"{sample_org_id}_progress_2024"),
        "created_at": _now(),
    }


@pytest.fixture
def on_track_progress(sample_org_id) -> Dict[str, Any]:
    """Progress record showing target is on track."""
    return {
        "id": _new_id(),
        "org_id": sample_org_id,
        "target_id": _new_id(),
        "reporting_year": 2024,
        "actual_emissions_tco2e": 63_000.0,
        "expected_emissions_tco2e": 66_400.0,
        "base_year_emissions_tco2e": 80_000.0,
        "cumulative_reduction_pct": 21.25,
        "expected_reduction_pct": 17.0,
        "variance_pct": 4.25,
        "on_track": True,
        "rag_status": "green",
    }


# ============================================================================
# TEMPERATURE SCORE FIXTURES
# ============================================================================

@pytest.fixture
def sample_temperature_score(sample_org_id) -> Dict[str, Any]:
    """Temperature score assessment for an organization."""
    return {
        "id": _new_id(),
        "org_id": sample_org_id,
        "assessment_date": _now(),
        "overall_score_c": 1.8,
        "scope1_score_c": 1.5,
        "scope2_score_c": 1.6,
        "scope3_score_c": 2.2,
        "short_term_score_c": 1.7,
        "long_term_score_c": 1.9,
        "methodology": "SBTi_CDP_temperature_rating",
        "data_quality_score": 3.5,
        "peer_percentile": 25,
        "sector_average_c": 2.8,
        "reduction_to_1_5c_pct": 28.0,
        "provenance_hash": _sha256(f"{sample_org_id}_temp_score"),
    }


@pytest.fixture
def high_temperature_score(sample_org_id) -> Dict[str, Any]:
    """High (poor) temperature score."""
    return {
        "id": _new_id(),
        "org_id": sample_org_id,
        "overall_score_c": 3.5,
        "scope1_score_c": 3.0,
        "scope2_score_c": 3.2,
        "scope3_score_c": 3.8,
        "peer_percentile": 80,
    }


# ============================================================================
# FI PORTFOLIO FIXTURES
# ============================================================================

@pytest.fixture
def sample_fi_portfolio(sample_org_id) -> Dict[str, Any]:
    """Financial institution portfolio with holdings."""
    return {
        "id": _new_id(),
        "org_id": sample_org_id,
        "portfolio_name": "Corporate Lending Portfolio",
        "reporting_year": 2024,
        "total_financed_emissions_tco2e": 5_000_000.0,
        "total_portfolio_value_usd": 50_000_000_000.0,
        "waci": 120.5,
        "waci_unit": "tCO2e per USD million invested",
        "holdings": [
            {
                "holding_id": _new_id(),
                "company_name": "Alpha Steel Corp",
                "asset_class": "corporate_bond",
                "exposure_usd": 500_000_000,
                "financed_emissions_tco2e": 250_000,
                "has_sbti_target": True,
                "target_status": "validated",
                "pcaf_data_quality": 2,
                "attribution_method": "evic",
            },
            {
                "holding_id": _new_id(),
                "company_name": "Beta Energy Inc",
                "asset_class": "listed_equity",
                "exposure_usd": 300_000_000,
                "financed_emissions_tco2e": 180_000,
                "has_sbti_target": False,
                "target_status": None,
                "pcaf_data_quality": 3,
                "attribution_method": "evic",
            },
            {
                "holding_id": _new_id(),
                "company_name": "Gamma Renewables",
                "asset_class": "project_finance",
                "exposure_usd": 200_000_000,
                "financed_emissions_tco2e": 20_000,
                "has_sbti_target": True,
                "target_status": "approved",
                "pcaf_data_quality": 1,
                "attribution_method": "project_attribution",
            },
        ],
        "coverage_with_sbti_pct": 66.67,
        "pcaf_avg_data_quality": 2.0,
        "fi_coverage_path": {
            2025: 50.0, 2030: 70.0, 2035: 85.0, 2040: 100.0,
        },
        "finz_compliant": True,
        "provenance_hash": _sha256(f"{sample_org_id}_fi_portfolio"),
    }


# ============================================================================
# RECALCULATION FIXTURES
# ============================================================================

@pytest.fixture
def sample_recalculation(sample_org_id) -> Dict[str, Any]:
    """Target recalculation record."""
    return {
        "id": _new_id(),
        "org_id": sample_org_id,
        "target_id": _new_id(),
        "trigger_type": "acquisition",
        "trigger_description": "Acquisition of SubCo increased base year emissions by 8%",
        "original_base_year_tco2e": 80_000.0,
        "adjusted_base_year_tco2e": 86_400.0,
        "change_pct": 8.0,
        "recalculation_threshold_pct": 5.0,
        "exceeds_threshold": True,
        "revalidation_required": True,
        "original_target_tco2e": 46_400.0,
        "adjusted_target_tco2e": 50_112.0,
        "methodology_change": False,
        "structural_change": True,
        "audit_trail": [
            {"timestamp": _now().isoformat(), "action": "trigger_identified", "user": "system"},
            {"timestamp": _now().isoformat(), "action": "recalculation_performed", "user": "analyst"},
        ],
        "provenance_hash": _sha256(f"{sample_org_id}_recalc"),
        "created_at": _now(),
    }


@pytest.fixture
def minor_recalculation(sample_org_id) -> Dict[str, Any]:
    """Recalculation below threshold (no revalidation needed)."""
    return {
        "id": _new_id(),
        "org_id": sample_org_id,
        "target_id": _new_id(),
        "trigger_type": "organic_growth",
        "original_base_year_tco2e": 80_000.0,
        "adjusted_base_year_tco2e": 82_000.0,
        "change_pct": 2.5,
        "recalculation_threshold_pct": 5.0,
        "exceeds_threshold": False,
        "revalidation_required": False,
    }


# ============================================================================
# FIVE-YEAR REVIEW FIXTURES
# ============================================================================

@pytest.fixture
def sample_five_year_review(sample_org_id) -> Dict[str, Any]:
    """Five-year review cycle record."""
    validation_date = date(2021, 6, 15)
    trigger_date = date(2026, 6, 15)
    deadline_date = date(2027, 6, 15)

    return {
        "id": _new_id(),
        "org_id": sample_org_id,
        "target_id": _new_id(),
        "original_validation_date": validation_date,
        "review_trigger_date": trigger_date,
        "review_deadline": deadline_date,
        "days_until_trigger": (trigger_date - date.today()).days,
        "days_until_deadline": (deadline_date - date.today()).days,
        "review_status": "upcoming",
        "readiness_score": 75.0,
        "notification_schedule": [
            {"months_before": 12, "sent": True, "sent_date": date(2025, 6, 15)},
            {"months_before": 6, "sent": True, "sent_date": date(2025, 12, 15)},
            {"months_before": 3, "sent": False},
            {"months_before": 1, "sent": False},
        ],
        "review_outcome": None,
        "new_target_id": None,
        "provenance_hash": _sha256(f"{sample_org_id}_5yr_review"),
    }


# ============================================================================
# GAP ASSESSMENT FIXTURES
# ============================================================================

@pytest.fixture
def sample_gap_assessment(sample_org_id) -> Dict[str, Any]:
    """Full gap analysis result."""
    return {
        "id": _new_id(),
        "org_id": sample_org_id,
        "assessment_date": _now(),
        "overall_readiness_score": 65.0,
        "readiness_level": "moderate_gaps",
        "data_gaps": [
            {"category": "scope3_cat8", "gap": "No leased asset emissions data", "severity": "medium"},
            {"category": "scope3_cat13", "gap": "No downstream leased asset data", "severity": "low"},
            {"category": "flag", "gap": "No FLAG emissions breakdown", "severity": "high"},
        ],
        "ambition_gaps": [
            {"area": "scope3_ambition", "current_rate": 2.0, "required_rate": 2.5, "shortfall_pct": 0.5},
        ],
        "process_gaps": [
            {"area": "governance", "gap": "No board-level climate oversight committee", "severity": "medium"},
            {"area": "reporting", "gap": "No annual progress disclosure process", "severity": "high"},
        ],
        "criteria_gaps": {
            "C2_ghg_gases": {"status": "warning", "detail": "NF3 excluded without justification"},
            "C9_scope3_coverage": {"status": "gap", "detail": "62% < 67% minimum coverage"},
        },
        "action_plan": [
            {"priority": 1, "action": "Collect FLAG emissions data", "effort": "high", "timeline_months": 6},
            {"priority": 2, "action": "Expand Scope 3 coverage to 67%+", "effort": "medium", "timeline_months": 3},
            {"priority": 3, "action": "Establish board climate committee", "effort": "low", "timeline_months": 2},
        ],
        "peer_benchmark": {
            "sector": "manufacturing",
            "avg_readiness_score": 55.0,
            "org_percentile": 65,
            "top_quartile_score": 80.0,
        },
        "provenance_hash": _sha256(f"{sample_org_id}_gap"),
    }


# ============================================================================
# SECTOR PATHWAY DATA FIXTURES
# ============================================================================

@pytest.fixture
def sector_pathway_data() -> Dict[str, Dict[str, Any]]:
    """Reference sector-specific pathway parameters."""
    return {
        "power": {
            "unit": "tCO2/MWh",
            "base_2020": 0.48,
            "target_2030": 0.14,
            "target_2050": 0.0,
        },
        "cement": {
            "unit": "tCO2e/tonne",
            "base_2020": 0.62,
            "target_2030": 0.42,
            "target_2050": 0.10,
        },
        "steel": {
            "unit": "tCO2e/tonne steel",
            "iron_ore_base_2020": 2.0,
            "iron_ore_target_2050": 0.05,
            "scrap_base_2020": 0.4,
            "scrap_target_2050": 0.02,
        },
        "buildings": {
            "unit": "kgCO2e/m2",
            "residential_base": 35.0,
            "commercial_base": 50.0,
            "crrem_2030": 25.0,
        },
        "maritime": {
            "unit": "gCO2/dwt-nm",
            "base_2020": 8.5,
            "target_2030": 5.2,
            "target_2050": 0.5,
        },
        "aviation": {
            "unit_passenger": "gCO2/RPK",
            "unit_freight": "gCO2/RTK",
            "passenger_base_2020": 90.0,
            "passenger_target_2050": 15.0,
            "freight_base_2020": 600.0,
            "freight_target_2050": 100.0,
        },
    }


# ============================================================================
# FLAG COMMODITY PATHWAY FIXTURES
# ============================================================================

@pytest.fixture
def flag_commodity_pathways() -> Dict[str, Dict[str, Any]]:
    """All 11 FLAG commodity pathways."""
    return {
        "cattle": {"annual_rate": 3.03, "base_year": 2020, "target_year": 2030},
        "soy": {"annual_rate": 3.03, "base_year": 2020, "target_year": 2030},
        "palm_oil": {"annual_rate": 3.03, "base_year": 2020, "target_year": 2030},
        "timber": {"annual_rate": 3.03, "base_year": 2020, "target_year": 2030},
        "cocoa": {"annual_rate": 3.03, "base_year": 2020, "target_year": 2030},
        "coffee": {"annual_rate": 3.03, "base_year": 2020, "target_year": 2030},
        "rubber": {"annual_rate": 3.03, "base_year": 2020, "target_year": 2030},
        "maize": {"annual_rate": 3.03, "base_year": 2020, "target_year": 2030},
        "rice": {"annual_rate": 3.03, "base_year": 2020, "target_year": 2030},
        "wheat": {"annual_rate": 3.03, "base_year": 2020, "target_year": 2030},
        "sugarcane": {"annual_rate": 3.03, "base_year": 2020, "target_year": 2030},
    }


# ============================================================================
# FRAMEWORK INTEGRATION FIXTURES
# ============================================================================

@pytest.fixture
def framework_alignment_data(sample_org_id) -> Dict[str, Any]:
    """Cross-framework alignment mapping."""
    return {
        "id": _new_id(),
        "org_id": sample_org_id,
        "frameworks": {
            "cdp": {
                "module": "C4",
                "status": "aligned",
                "questions_mapped": ["C4.1a", "C4.1b", "C4.2", "C4.2a", "C4.2b"],
            },
            "tcfd": {
                "module": "MT-c",
                "status": "aligned",
                "disclosures_mapped": ["mt_c_targets", "mt_c_progress"],
            },
            "csrd": {
                "module": "ESRS_E1",
                "status": "partially_aligned",
                "paragraphs_mapped": ["E1-4", "E1-5", "E1-6"],
                "gaps": ["E1-7 energy consumption not linked"],
            },
            "ghg_protocol": {
                "status": "aligned",
                "standards": ["corporate_standard", "scope3_standard"],
            },
            "iso14064": {
                "status": "aligned",
                "parts": ["part_1", "part_3"],
                "verification_linkage": True,
            },
        },
        "unified_status": "partially_aligned",
        "overall_coverage_pct": 85.0,
    }


# ============================================================================
# MOCK ENGINE FIXTURES
# ============================================================================

@pytest.fixture
def target_engine() -> MagicMock:
    """Mock TargetConfigurationEngine."""
    engine = MagicMock()
    engine.create_target = MagicMock()
    engine.validate_timeframe = MagicMock(return_value=True)
    engine.validate_coverage = MagicMock(return_value=True)
    engine.calculate_annual_rate = MagicMock(return_value=4.2)
    engine.transition_status = MagicMock(return_value="submitted")
    engine.check_scope3_requirement = MagicMock(return_value=True)
    return engine


@pytest.fixture
def pathway_engine() -> MagicMock:
    """Mock PathwayCalculatorEngine."""
    engine = MagicMock()
    engine.calculate_aca_pathway = MagicMock()
    engine.calculate_sda_pathway = MagicMock()
    engine.calculate_economic_intensity = MagicMock()
    engine.calculate_physical_intensity = MagicMock()
    engine.calculate_flag_commodity_pathway = MagicMock()
    engine.calculate_flag_sector_pathway = MagicMock()
    engine.compare_pathways = MagicMock()
    engine.calculate_uncertainty_bands = MagicMock()
    engine.generate_milestones = MagicMock()
    return engine


@pytest.fixture
def validation_engine() -> MagicMock:
    """Mock ValidationEngine."""
    engine = MagicMock()
    engine.validate_full = MagicMock()
    engine.check_criterion = MagicMock()
    engine.generate_readiness_report = MagicMock()
    engine.validate_net_zero = MagicMock()
    engine.validate_flag = MagicMock()
    return engine


@pytest.fixture
def scope3_engine() -> MagicMock:
    """Mock Scope3ScreeningEngine."""
    engine = MagicMock()
    engine.assess_trigger = MagicMock()
    engine.breakdown_categories = MagicMock()
    engine.identify_hotspots = MagicMock()
    engine.calculate_coverage = MagicMock()
    engine.recommend_categories = MagicMock()
    engine.assess_data_quality = MagicMock()
    return engine


@pytest.fixture
def flag_engine() -> MagicMock:
    """Mock FLAGAssessmentEngine."""
    engine = MagicMock()
    engine.assess_trigger = MagicMock()
    engine.classify_sector = MagicMock()
    engine.calculate_commodity_pathway = MagicMock()
    engine.calculate_sector_pathway = MagicMock()
    engine.validate_deforestation_commitment = MagicMock()
    engine.calculate_long_term_flag = MagicMock()
    engine.split_flag_emissions = MagicMock()
    return engine


@pytest.fixture
def sector_engine() -> MagicMock:
    """Mock SectorPathwayEngine."""
    engine = MagicMock()
    engine.detect_sector = MagicMock()
    engine.calculate_sector_pathway = MagicMock()
    engine.blend_multi_sector = MagicMock()
    engine.get_convergence_value = MagicMock()
    return engine


@pytest.fixture
def progress_engine() -> MagicMock:
    """Mock ProgressTrackingEngine."""
    engine = MagicMock()
    engine.record_progress = MagicMock()
    engine.calculate_variance = MagicMock()
    engine.determine_rag_status = MagicMock()
    engine.project_achievement = MagicMock()
    engine.fetch_mrv_data = MagicMock()
    return engine


@pytest.fixture
def temperature_engine() -> MagicMock:
    """Mock TemperatureScoringEngine."""
    engine = MagicMock()
    engine.score_company = MagicMock()
    engine.score_scope = MagicMock()
    engine.score_portfolio = MagicMock()
    engine.rank_peers = MagicMock()
    engine.map_reduction_to_temperature = MagicMock()
    return engine


@pytest.fixture
def recalculation_engine() -> MagicMock:
    """Mock RecalculationEngine."""
    engine = MagicMock()
    engine.check_threshold = MagicMock()
    engine.create_recalculation = MagicMock()
    engine.assess_revalidation = MagicMock()
    engine.model_ma_impact = MagicMock()
    engine.update_target = MagicMock()
    return engine


@pytest.fixture
def review_engine() -> MagicMock:
    """Mock FiveYearReviewEngine."""
    engine = MagicMock()
    engine.calculate_trigger_date = MagicMock()
    engine.calculate_deadline = MagicMock()
    engine.assess_readiness = MagicMock()
    engine.generate_notifications = MagicMock()
    engine.record_outcome = MagicMock()
    engine.list_upcoming = MagicMock()
    return engine


@pytest.fixture
def fi_engine() -> MagicMock:
    """Mock FinancialInstitutionsEngine."""
    engine = MagicMock()
    engine.create_portfolio = MagicMock()
    engine.calculate_coverage = MagicMock()
    engine.calculate_financed_emissions = MagicMock()
    engine.calculate_waci = MagicMock()
    engine.assess_pcaf_quality = MagicMock()
    engine.project_coverage_path = MagicMock()
    engine.validate_finz = MagicMock()
    return engine


@pytest.fixture
def framework_engine() -> MagicMock:
    """Mock FrameworkIntegrationEngine."""
    engine = MagicMock()
    engine.map_cdp = MagicMock()
    engine.map_tcfd = MagicMock()
    engine.map_csrd = MagicMock()
    engine.map_ghg_protocol = MagicMock()
    engine.map_iso14064 = MagicMock()
    engine.generate_unified_status = MagicMock()
    return engine


@pytest.fixture
def reporting_engine() -> MagicMock:
    """Mock ReportingEngine."""
    engine = MagicMock()
    engine.generate_submission_form = MagicMock()
    engine.generate_progress_report = MagicMock()
    engine.generate_validation_report = MagicMock()
    engine.export_pdf = MagicMock()
    engine.export_excel = MagicMock()
    engine.export_json = MagicMock()
    engine.export_xml = MagicMock()
    engine.generate_executive_summary = MagicMock()
    return engine


@pytest.fixture
def gap_engine() -> MagicMock:
    """Mock GapAnalysisEngine."""
    engine = MagicMock()
    engine.run_full_analysis = MagicMock()
    engine.identify_data_gaps = MagicMock()
    engine.identify_ambition_gaps = MagicMock()
    engine.identify_process_gaps = MagicMock()
    engine.generate_action_plan = MagicMock()
    engine.benchmark_peers = MagicMock()
    return engine


# ============================================================================
# MOCK DATABASE SESSION FIXTURES
# ============================================================================

@pytest.fixture
def mock_db_session():
    """Mock async database session for integration-style tests."""
    session = MagicMock()
    session.execute = AsyncMock(return_value=MagicMock())
    session.commit = AsyncMock()
    session.rollback = AsyncMock()
    session.close = AsyncMock()
    session.refresh = AsyncMock()
    return session


@pytest.fixture
def mock_db_results():
    """Mock database result set."""
    result = MagicMock()
    result.scalars = MagicMock(return_value=MagicMock())
    result.scalars.return_value.all = MagicMock(return_value=[])
    result.scalars.return_value.first = MagicMock(return_value=None)
    return result


# ============================================================================
# MOCK API CLIENT FIXTURES
# ============================================================================

@pytest.fixture
def mock_client():
    """Mock FastAPI test client for route testing."""
    client = MagicMock()
    client.get = MagicMock()
    client.post = MagicMock()
    client.put = MagicMock()
    client.delete = MagicMock()
    client.patch = MagicMock()
    return client


@pytest.fixture
def auth_headers():
    """Authentication headers for protected endpoints."""
    return {"Authorization": "Bearer test-sbti-jwt-token"}
