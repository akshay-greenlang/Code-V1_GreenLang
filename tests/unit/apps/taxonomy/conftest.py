# -*- coding: utf-8 -*-
"""
Shared pytest fixtures for GL-Taxonomy-APP v1.0 test suite.

Provides reusable fixtures for configuration, organizations (non-financial
and financial), economic activity catalogs, NACE mappings, eligibility
screening results, Substantial Contribution (SC) assessments, Technical
Screening Criteria (TSC) evaluations, DNSH assessments, climate risk
assessments, Minimum Safeguard assessments, KPI calculation data, CapEx
plans, Green Asset Ratio (GAR) data, financial exposures, alignment
results, portfolio alignments, report fixtures, evidence items, regulatory
versions, data quality scores, gap assessments, and mock engine instances
used across all 9 test modules.

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
# Path setup -- ensure the Taxonomy services package is importable
# ---------------------------------------------------------------------------
_SERVICES_DIR = os.path.join(
    os.path.dirname(__file__),
    "..", "..", "..", "..",
    "applications", "GL-Taxonomy-APP", "EU-Taxonomy-Platform",
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
    """Default Taxonomy application configuration."""
    return {
        "app_name": "GL-Taxonomy-APP",
        "app_version": "1.0.0",
        "taxonomy_regulation_version": "2020/852",
        "climate_delegated_act_version": "2021/2139",
        "environmental_delegated_act_version": "2023/2486",
        "reporting_year": 2025,
        "reporting_period": "FY2025",
        "environmental_objectives": [
            "climate_mitigation",
            "climate_adaptation",
            "water_marine",
            "circular_economy",
            "pollution_prevention",
            "biodiversity",
        ],
        "kpi_types": ["turnover", "capex", "opex"],
        "de_minimis_threshold_pct": 5.0,
        "sc_confidence_threshold": 80.0,
        "dnsh_requires_all_pass": True,
        "ms_requires_all_four_topics": True,
        "gar_stock_enabled": True,
        "gar_flow_enabled": True,
        "epc_alignment_threshold": "B",
        "auto_loan_co2_threshold_gkm": 50.0,
        "capex_plan_max_years": 10,
        "data_quality_min_grade": "C",
        "supported_formats": ["pdf", "excel", "csv", "xbrl"],
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
def sample_org(sample_org_id) -> Dict[str, Any]:
    """Sample non-financial organization for taxonomy alignment."""
    return {
        "id": sample_org_id,
        "tenant_id": _new_id(),
        "name": "EuroManufacturing GmbH",
        "entity_type": "non_financial",
        "sector": "manufacturing",
        "country": "DE",
        "lei": "529900HNOAA1KXQJUQ27",
        "nfrd_reporting": True,
        "csrd_reporting": True,
        "employee_count": 8500,
        "annual_revenue": Decimal("2500000000.00"),
        "total_assets": Decimal("4200000000.00"),
        "settings": {
            "default_period": "FY2025",
            "base_currency": "EUR",
            "taxonomy_version": "2024",
        },
        "created_at": _now(),
        "updated_at": _now(),
    }


@pytest.fixture
def sample_financial_institution() -> Dict[str, Any]:
    """Financial institution (credit institution) for GAR/BTAR tests."""
    return {
        "id": _new_id(),
        "tenant_id": _new_id(),
        "name": "GreenBank Europe AG",
        "entity_type": "financial",
        "sector": "banking",
        "country": "NL",
        "lei": "724500VKKSH9QOLTFR71",
        "nfrd_reporting": True,
        "csrd_reporting": True,
        "employee_count": 12000,
        "annual_revenue": Decimal("8500000000.00"),
        "total_assets": Decimal("350000000000.00"),
        "settings": {
            "gar_enabled": True,
            "btar_enabled": True,
            "pillar3_reporting": True,
        },
        "created_at": _now(),
        "updated_at": _now(),
    }


@pytest.fixture
def sample_insurance_entity() -> Dict[str, Any]:
    """Insurance undertaking for taxonomy tests."""
    return {
        "id": _new_id(),
        "tenant_id": _new_id(),
        "name": "EuroInsure S.A.",
        "entity_type": "insurance",
        "sector": "insurance",
        "country": "FR",
        "lei": "969500LYSYS0E800SQ95",
        "nfrd_reporting": True,
        "csrd_reporting": True,
        "total_assets": Decimal("125000000000.00"),
        "settings": {"investment_kpi_enabled": True},
    }


@pytest.fixture
def sample_asset_manager() -> Dict[str, Any]:
    """Asset management company for taxonomy tests."""
    return {
        "id": _new_id(),
        "tenant_id": _new_id(),
        "name": "Sustainable Capital Partners",
        "entity_type": "asset_manager",
        "sector": "asset_management",
        "country": "LU",
        "lei": "549300TRUWO2CD2G5692",
        "nfrd_reporting": True,
        "csrd_reporting": True,
        "total_assets": Decimal("45000000000.00"),
    }


# ============================================================================
# ECONOMIC ACTIVITY FIXTURES
# ============================================================================

@pytest.fixture
def sample_activities() -> List[Dict[str, Any]]:
    """Representative EU Taxonomy economic activities (13 entries)."""
    return [
        {
            "id": _new_id(),
            "activity_code": "CCM_4.1",
            "nace_codes": ["D35.11"],
            "sector": "energy",
            "name": "Electricity generation using solar photovoltaic technology",
            "description": "Construction and operation of electricity generation facilities using solar PV.",
            "objectives": ["climate_mitigation"],
            "activity_type": "own_performance",
            "delegated_act": "climate",
            "sc_criteria": {
                "type": "life_cycle_ghg",
                "threshold_gco2e_kwh": 100,
                "methodology": "ISO 14067",
            },
            "dnsh_criteria": {
                "climate_adaptation": {"type": "climate_risk_assessment"},
                "water_marine": {"type": "not_applicable"},
                "circular_economy": {"type": "waste_management_plan"},
                "pollution_prevention": {"type": "hazardous_substance_check"},
                "biodiversity": {"type": "eia_required"},
            },
            "effective_date": date(2022, 1, 1),
            "version": "1.0",
        },
        {
            "id": _new_id(),
            "activity_code": "CCM_3.3",
            "nace_codes": ["C23.51"],
            "sector": "manufacturing",
            "name": "Manufacture of cement",
            "description": "Manufacture of cement clinker, cement or alternative binder.",
            "objectives": ["climate_mitigation"],
            "activity_type": "transitional",
            "delegated_act": "climate",
            "sc_criteria": {
                "type": "specific_emissions",
                "grey_clinker_threshold_tco2e_t": 0.722,
                "cement_threshold_tco2e_t": 0.469,
            },
            "dnsh_criteria": {
                "climate_adaptation": {"type": "climate_risk_assessment"},
                "water_marine": {"type": "water_use_assessment"},
                "circular_economy": {"type": "waste_management_plan"},
                "pollution_prevention": {"type": "bat_compliance"},
                "biodiversity": {"type": "eia_required"},
            },
            "effective_date": date(2022, 1, 1),
            "version": "1.0",
        },
        {
            "id": _new_id(),
            "activity_code": "CCM_3.9",
            "nace_codes": ["C24.10", "C24.20"],
            "sector": "manufacturing",
            "name": "Manufacture of iron and steel",
            "description": "Manufacture of iron, steel, ferro-alloys.",
            "objectives": ["climate_mitigation"],
            "activity_type": "transitional",
            "delegated_act": "climate",
            "sc_criteria": {
                "type": "specific_emissions",
                "hot_metal_threshold_tco2e_t": 1.331,
                "eaf_carbon_threshold_tco2e_t": 0.209,
                "eaf_alloy_threshold_tco2e_t": 0.266,
            },
            "effective_date": date(2022, 1, 1),
            "version": "1.0",
        },
        {
            "id": _new_id(),
            "activity_code": "CCM_6.5",
            "nace_codes": ["H49.10", "H49.20"],
            "sector": "transport",
            "name": "Transport by motorbikes, passenger cars and light commercial vehicles",
            "description": "Purchase, financing, renting, leasing and operation of vehicles.",
            "objectives": ["climate_mitigation"],
            "activity_type": "own_performance",
            "delegated_act": "climate",
            "sc_criteria": {
                "type": "zero_direct_co2",
                "threshold_gco2_km": 0,
                "alternative_threshold_gco2_km": 50,
                "alternative_until_date": "2025-12-31",
            },
            "effective_date": date(2022, 1, 1),
            "version": "1.0",
        },
        {
            "id": _new_id(),
            "activity_code": "CCM_7.1",
            "nace_codes": ["F41.10", "F41.20"],
            "sector": "construction_real_estate",
            "name": "Construction of new buildings",
            "description": "Construction of new buildings, major renovations.",
            "objectives": ["climate_mitigation"],
            "activity_type": "own_performance",
            "delegated_act": "climate",
            "sc_criteria": {
                "type": "nzeb_minus_10",
                "energy_performance": "NZEB - 10%",
                "airtightness_required": True,
                "thermal_integrity_required": True,
                "life_cycle_gwp_required": True,
            },
            "effective_date": date(2022, 1, 1),
            "version": "1.0",
        },
        {
            "id": _new_id(),
            "activity_code": "CCM_7.7",
            "nace_codes": ["L68.20"],
            "sector": "construction_real_estate",
            "name": "Acquisition and ownership of buildings",
            "description": "Buying real property and exercising ownership of that property.",
            "objectives": ["climate_mitigation"],
            "activity_type": "own_performance",
            "delegated_act": "climate",
            "sc_criteria": {
                "type": "epc_class_a_or_top15",
                "epc_threshold": "A",
                "alternative": "top 15% national building stock",
            },
            "effective_date": date(2022, 1, 1),
            "version": "1.0",
        },
        {
            "id": _new_id(),
            "activity_code": "CCA_9.1",
            "nace_codes": ["K65.11", "K65.12"],
            "sector": "financial_services",
            "name": "Non-life insurance: underwriting of climate-related perils",
            "description": "Insurance and reinsurance of climate-related perils.",
            "objectives": ["climate_adaptation"],
            "activity_type": "enabling",
            "delegated_act": "climate",
            "sc_criteria": {
                "type": "climate_risk_integration",
                "nat_cat_modelling_required": True,
                "pricing_risk_signals": True,
            },
            "effective_date": date(2022, 1, 1),
            "version": "1.0",
        },
        {
            "id": _new_id(),
            "activity_code": "WTR_2.1",
            "nace_codes": ["E36.00"],
            "sector": "water_supply",
            "name": "Water supply",
            "description": "Construction, extension and operation of water supply systems.",
            "objectives": ["water_marine"],
            "activity_type": "own_performance",
            "delegated_act": "environmental",
            "sc_criteria": {
                "type": "leakage_rate",
                "infrastructure_leakage_index_threshold": 1.5,
                "energy_efficiency_required": True,
            },
            "effective_date": date(2024, 1, 1),
            "version": "1.0",
        },
        {
            "id": _new_id(),
            "activity_code": "CE_1.2",
            "nace_codes": ["C20.16"],
            "sector": "manufacturing",
            "name": "Manufacture of plastics in primary form",
            "description": "Manufacture of primary-form plastics from renewable feedstock.",
            "objectives": ["circular_economy"],
            "activity_type": "own_performance",
            "delegated_act": "environmental",
            "sc_criteria": {
                "type": "renewable_feedstock",
                "min_renewable_content_pct": 90,
                "recyclability_required": True,
            },
            "effective_date": date(2024, 1, 1),
            "version": "1.0",
        },
        {
            "id": _new_id(),
            "activity_code": "PPC_1.1",
            "nace_codes": ["E38.11"],
            "sector": "waste_management",
            "name": "Collection and transport of non-hazardous waste",
            "description": "Separate collection and transport of non-hazardous waste.",
            "objectives": ["pollution_prevention"],
            "activity_type": "own_performance",
            "delegated_act": "environmental",
            "sc_criteria": {
                "type": "separate_collection",
                "separate_collection_required": True,
                "no_mixing_allowed": True,
            },
            "effective_date": date(2024, 1, 1),
            "version": "1.0",
        },
        {
            "id": _new_id(),
            "activity_code": "BIO_1.1",
            "nace_codes": ["A02.10"],
            "sector": "forestry",
            "name": "Forest management",
            "description": "Forest management including afforestation and reforestation.",
            "objectives": ["biodiversity"],
            "activity_type": "own_performance",
            "delegated_act": "environmental",
            "sc_criteria": {
                "type": "forest_management_plan",
                "sustainable_certification_required": True,
                "no_deforestation": True,
            },
            "effective_date": date(2024, 1, 1),
            "version": "1.0",
        },
        {
            "id": _new_id(),
            "activity_code": "CCM_4.3",
            "nace_codes": ["D35.11"],
            "sector": "energy",
            "name": "Electricity generation from wind power",
            "description": "Construction and operation of electricity generation from wind.",
            "objectives": ["climate_mitigation"],
            "activity_type": "own_performance",
            "delegated_act": "climate",
            "sc_criteria": {
                "type": "life_cycle_ghg",
                "threshold_gco2e_kwh": 100,
            },
            "effective_date": date(2022, 1, 1),
            "version": "1.0",
        },
        {
            "id": _new_id(),
            "activity_code": "CCM_8.1",
            "nace_codes": ["J61", "J62"],
            "sector": "ict",
            "name": "Data processing, hosting and related activities",
            "description": "Storage, manipulation, management of data.",
            "objectives": ["climate_mitigation"],
            "activity_type": "enabling",
            "delegated_act": "climate",
            "sc_criteria": {
                "type": "energy_efficiency",
                "pue_threshold": 1.5,
                "eu_code_of_conduct": True,
            },
            "effective_date": date(2022, 1, 1),
            "version": "1.0",
        },
    ]


@pytest.fixture
def single_activity(sample_activities) -> Dict[str, Any]:
    """Single activity (solar PV) for focused tests."""
    return sample_activities[0]


@pytest.fixture
def transitional_activity(sample_activities) -> Dict[str, Any]:
    """Transitional activity (cement) for SC type tests."""
    return sample_activities[1]


@pytest.fixture
def enabling_activity(sample_activities) -> Dict[str, Any]:
    """Enabling activity (data centres) for SC type tests."""
    return sample_activities[12]


# ============================================================================
# NACE MAPPING FIXTURES
# ============================================================================

@pytest.fixture
def sample_nace_mappings() -> List[Dict[str, Any]]:
    """NACE hierarchy with taxonomy activity mappings."""
    return [
        {
            "id": _new_id(),
            "nace_code": "D",
            "nace_description": "Electricity, gas, steam and air conditioning supply",
            "nace_level": 1,
            "parent_code": None,
            "taxonomy_activities": [],
        },
        {
            "id": _new_id(),
            "nace_code": "D35",
            "nace_description": "Electricity, gas, steam and air conditioning supply",
            "nace_level": 2,
            "parent_code": "D",
            "taxonomy_activities": [],
        },
        {
            "id": _new_id(),
            "nace_code": "D35.1",
            "nace_description": "Electric power generation, transmission and distribution",
            "nace_level": 3,
            "parent_code": "D35",
            "taxonomy_activities": [],
        },
        {
            "id": _new_id(),
            "nace_code": "D35.11",
            "nace_description": "Production of electricity",
            "nace_level": 4,
            "parent_code": "D35.1",
            "taxonomy_activities": ["CCM_4.1", "CCM_4.3", "CCM_4.5", "CCM_4.7", "CCM_4.8"],
        },
        {
            "id": _new_id(),
            "nace_code": "C23.51",
            "nace_description": "Manufacture of cement",
            "nace_level": 4,
            "parent_code": "C23.5",
            "taxonomy_activities": ["CCM_3.3"],
        },
        {
            "id": _new_id(),
            "nace_code": "C24.10",
            "nace_description": "Manufacture of basic iron and steel and of ferro-alloys",
            "nace_level": 4,
            "parent_code": "C24.1",
            "taxonomy_activities": ["CCM_3.9"],
        },
        {
            "id": _new_id(),
            "nace_code": "F41.10",
            "nace_description": "Development of building projects",
            "nace_level": 4,
            "parent_code": "F41.1",
            "taxonomy_activities": ["CCM_7.1"],
        },
        {
            "id": _new_id(),
            "nace_code": "L68.20",
            "nace_description": "Renting and operating of own or leased real estate",
            "nace_level": 4,
            "parent_code": "L68.2",
            "taxonomy_activities": ["CCM_7.7"],
        },
    ]


# ============================================================================
# ELIGIBILITY SCREENING FIXTURES
# ============================================================================

@pytest.fixture
def sample_screening(sample_org_id) -> Dict[str, Any]:
    """Completed eligibility screening with mixed results."""
    return {
        "id": _new_id(),
        "org_id": sample_org_id,
        "period": "FY2025",
        "screening_date": _now(),
        "total_activities": 15,
        "eligible_count": 10,
        "not_eligible_count": 4,
        "de_minimis_excluded": 1,
        "status": "completed",
        "metadata": {
            "screened_by": "system",
            "delegated_act_version": "2021/2139",
        },
        "created_at": _now(),
        "updated_at": _now(),
    }


@pytest.fixture
def sample_screening_results(sample_org_id) -> List[Dict[str, Any]]:
    """Per-activity screening results."""
    screening_id = _new_id()
    return [
        {
            "id": _new_id(),
            "screening_id": screening_id,
            "activity_code": "CCM_4.1",
            "eligible": True,
            "objectives": ["climate_mitigation"],
            "delegated_act": "climate",
            "confidence": Decimal("95.00"),
            "de_minimis": False,
            "notes": "Solar PV generation clearly eligible",
        },
        {
            "id": _new_id(),
            "screening_id": screening_id,
            "activity_code": "CCM_3.3",
            "eligible": True,
            "objectives": ["climate_mitigation"],
            "delegated_act": "climate",
            "confidence": Decimal("90.00"),
            "de_minimis": False,
            "notes": "Cement manufacturing - transitional activity",
        },
        {
            "id": _new_id(),
            "screening_id": screening_id,
            "activity_code": "CCM_7.1",
            "eligible": True,
            "objectives": ["climate_mitigation"],
            "delegated_act": "climate",
            "confidence": Decimal("88.00"),
            "de_minimis": False,
            "notes": None,
        },
        {
            "id": _new_id(),
            "screening_id": screening_id,
            "activity_code": "NON_ELIGIBLE_1",
            "eligible": False,
            "objectives": [],
            "delegated_act": None,
            "confidence": Decimal("15.00"),
            "de_minimis": False,
            "notes": "No matching taxonomy activity found",
        },
        {
            "id": _new_id(),
            "screening_id": screening_id,
            "activity_code": "DE_MINIMIS_1",
            "eligible": False,
            "objectives": ["climate_mitigation"],
            "delegated_act": "climate",
            "confidence": Decimal("70.00"),
            "de_minimis": True,
            "notes": "Below de minimis threshold",
        },
    ]


# ============================================================================
# SUBSTANTIAL CONTRIBUTION FIXTURES
# ============================================================================

@pytest.fixture
def sample_sc_assessment(sample_org_id) -> Dict[str, Any]:
    """SC assessment for solar PV (climate mitigation, own performance)."""
    return {
        "id": _new_id(),
        "org_id": sample_org_id,
        "activity_code": "CCM_4.1",
        "objective": "climate_mitigation",
        "assessment_date": _now(),
        "status": "completed",
        "sc_type": "own_performance",
        "overall_pass": True,
        "threshold_checks": {
            "life_cycle_ghg": {
                "threshold_gco2e_kwh": 100,
                "actual_gco2e_kwh": 25.4,
                "pass": True,
                "methodology": "ISO 14067",
            },
        },
        "evidence_items": [
            {"type": "certification", "ref": "ISO14067-2023-001", "verified": True},
            {"type": "report", "ref": "LCA-SOLAR-2025-Q1", "verified": True},
        ],
        "assessor": "Environmental Compliance Team",
        "notes": "Life-cycle GHG well below 100 gCO2e/kWh threshold",
        "provenance_hash": _sha256("sc_assessment_solar_pv"),
        "created_at": _now(),
        "updated_at": _now(),
    }


@pytest.fixture
def cement_sc_assessment(sample_org_id) -> Dict[str, Any]:
    """SC assessment for cement (transitional activity)."""
    return {
        "id": _new_id(),
        "org_id": sample_org_id,
        "activity_code": "CCM_3.3",
        "objective": "climate_mitigation",
        "assessment_date": _now(),
        "status": "completed",
        "sc_type": "transitional",
        "overall_pass": True,
        "threshold_checks": {
            "grey_clinker_emissions": {
                "threshold_tco2e_t": 0.722,
                "actual_tco2e_t": 0.685,
                "pass": True,
            },
            "cement_emissions": {
                "threshold_tco2e_t": 0.469,
                "actual_tco2e_t": 0.445,
                "pass": True,
            },
        },
        "assessor": "Industrial Process Specialist",
        "notes": "Transitional: below sector threshold, no lock-in",
    }


@pytest.fixture
def steel_sc_assessment(sample_org_id) -> Dict[str, Any]:
    """SC assessment for steel (failing threshold)."""
    return {
        "id": _new_id(),
        "org_id": sample_org_id,
        "activity_code": "CCM_3.9",
        "objective": "climate_mitigation",
        "assessment_date": _now(),
        "status": "completed",
        "sc_type": "transitional",
        "overall_pass": False,
        "threshold_checks": {
            "hot_metal_emissions": {
                "threshold_tco2e_t": 1.331,
                "actual_tco2e_t": 1.520,
                "pass": False,
            },
        },
        "notes": "Exceeds hot metal threshold by 14.2%",
    }


@pytest.fixture
def sample_tsc_evaluations() -> List[Dict[str, Any]]:
    """TSC evaluation records for solar PV SC assessment."""
    assessment_id = _new_id()
    return [
        {
            "id": _new_id(),
            "assessment_id": assessment_id,
            "criterion_id": "CCM_4.1_SC_1",
            "description": "Life-cycle GHG emissions below 100 gCO2e/kWh",
            "threshold_value": Decimal("100.0000"),
            "actual_value": Decimal("25.4000"),
            "unit": "gCO2e/kWh",
            "pass_result": True,
            "evidence_ref": "LCA-SOLAR-2025-Q1",
        },
        {
            "id": _new_id(),
            "assessment_id": assessment_id,
            "criterion_id": "CCM_4.1_SC_2",
            "description": "Power plant complies with IEC 61215 / IEC 61730",
            "threshold_value": None,
            "actual_value": None,
            "unit": None,
            "pass_result": True,
            "evidence_ref": "CERT-IEC61215-2024",
        },
    ]


# ============================================================================
# DNSH ASSESSMENT FIXTURES
# ============================================================================

@pytest.fixture
def sample_dnsh_assessment(sample_org_id) -> Dict[str, Any]:
    """Full DNSH assessment for solar PV (SC = climate mitigation)."""
    return {
        "id": _new_id(),
        "org_id": sample_org_id,
        "activity_code": "CCM_4.1",
        "sc_objective": "climate_mitigation",
        "assessment_date": _now(),
        "overall_pass": True,
        "objective_results": {
            "climate_adaptation": {"status": "pass", "criteria_met": True},
            "water_marine": {"status": "not_applicable"},
            "circular_economy": {"status": "pass", "criteria_met": True},
            "pollution_prevention": {"status": "pass", "criteria_met": True},
            "biodiversity": {"status": "pass", "criteria_met": True},
        },
        "evidence_items": [
            {"type": "report", "ref": "CRA-SOLAR-2025", "objective": "climate_adaptation"},
            {"type": "document", "ref": "WMP-SOLAR-2025", "objective": "circular_economy"},
        ],
        "status": "completed",
        "notes": "All applicable DNSH criteria met",
        "provenance_hash": _sha256("dnsh_solar_pv"),
        "created_at": _now(),
        "updated_at": _now(),
    }


@pytest.fixture
def failing_dnsh_assessment(sample_org_id) -> Dict[str, Any]:
    """DNSH assessment with a failing objective (pollution prevention)."""
    return {
        "id": _new_id(),
        "org_id": sample_org_id,
        "activity_code": "CCM_3.3",
        "sc_objective": "climate_mitigation",
        "assessment_date": _now(),
        "overall_pass": False,
        "objective_results": {
            "climate_adaptation": {"status": "pass"},
            "water_marine": {"status": "pass"},
            "circular_economy": {"status": "pass"},
            "pollution_prevention": {"status": "fail", "reason": "BAT non-compliance"},
            "biodiversity": {"status": "pass"},
        },
        "status": "completed",
        "notes": "Pollution prevention DNSH failed: BAT requirements not met",
    }


@pytest.fixture
def sample_dnsh_objective_results() -> List[Dict[str, Any]]:
    """Per-objective DNSH results for solar PV."""
    assessment_id = _new_id()
    return [
        {
            "id": _new_id(),
            "assessment_id": assessment_id,
            "objective": "climate_adaptation",
            "status": "pass",
            "criteria_checks": {
                "climate_risk_assessment": {"completed": True, "risks_identified": 2, "adaptations_planned": 2},
            },
            "evidence_items": [{"type": "report", "ref": "CRA-2025-001"}],
            "notes": "Physical risks assessed per Appendix A",
        },
        {
            "id": _new_id(),
            "assessment_id": assessment_id,
            "objective": "water_marine",
            "status": "not_applicable",
            "criteria_checks": {},
            "evidence_items": [],
            "notes": "Solar PV: no water DNSH criteria apply",
        },
        {
            "id": _new_id(),
            "assessment_id": assessment_id,
            "objective": "circular_economy",
            "status": "pass",
            "criteria_checks": {
                "waste_management_plan": {"exists": True, "recycling_target_met": True},
                "durability": {"lifetime_years": 25, "meets_standard": True},
            },
            "evidence_items": [{"type": "document", "ref": "WMP-2025-001"}],
        },
        {
            "id": _new_id(),
            "assessment_id": assessment_id,
            "objective": "pollution_prevention",
            "status": "pass",
            "criteria_checks": {
                "hazardous_substances": {"rohs_compliant": True, "reach_compliant": True},
            },
            "evidence_items": [{"type": "certification", "ref": "ROHS-CERT-2025"}],
        },
        {
            "id": _new_id(),
            "assessment_id": assessment_id,
            "objective": "biodiversity",
            "status": "pass",
            "criteria_checks": {
                "eia_completed": True,
                "protected_area_check": True,
                "no_conversion_of_high_biodiversity": True,
            },
            "evidence_items": [{"type": "report", "ref": "EIA-SOLAR-2025"}],
        },
    ]


# ============================================================================
# CLIMATE RISK ASSESSMENT FIXTURES
# ============================================================================

@pytest.fixture
def sample_climate_risk_assessment(sample_org_id) -> Dict[str, Any]:
    """Climate risk assessment for DNSH climate adaptation."""
    return {
        "id": _new_id(),
        "org_id": sample_org_id,
        "activity_code": "CCM_4.1",
        "location": "Bavaria, Germany",
        "time_horizon": "long_term",
        "physical_risks": {
            "chronic": [
                {"hazard": "temperature_increase", "severity": "medium", "likelihood": "high"},
                {"hazard": "drought", "severity": "low", "likelihood": "medium"},
            ],
            "acute": [
                {"hazard": "heatwave", "severity": "medium", "likelihood": "medium"},
                {"hazard": "flooding", "severity": "low", "likelihood": "low"},
            ],
        },
        "adaptation_solutions": [
            {"risk": "temperature_increase", "solution": "Enhanced cooling systems", "cost_eur": 250000},
            {"risk": "heatwave", "solution": "Temperature-resistant panel coatings", "cost_eur": 180000},
        ],
        "residual_risks": {
            "temperature_increase": {"residual_severity": "low", "acceptable": True},
            "heatwave": {"residual_severity": "low", "acceptable": True},
        },
        "overall_status": "managed",
        "created_at": _now(),
    }


# ============================================================================
# MINIMUM SAFEGUARD FIXTURES
# ============================================================================

@pytest.fixture
def sample_safeguard_assessment(sample_org_id) -> Dict[str, Any]:
    """Minimum safeguard assessment passing all four topics."""
    return {
        "id": _new_id(),
        "org_id": sample_org_id,
        "assessment_date": _now(),
        "overall_pass": True,
        "topics": {
            "human_rights": {"procedural": True, "outcome": True, "overall": True},
            "anti_corruption": {"procedural": True, "outcome": True, "overall": True},
            "taxation": {"procedural": True, "outcome": True, "overall": True},
            "fair_competition": {"procedural": True, "outcome": True, "overall": True},
        },
        "evidence_items": [
            {"type": "declaration", "ref": "HR-POLICY-2025", "topic": "human_rights"},
            {"type": "certification", "ref": "ISO37001-2025", "topic": "anti_corruption"},
            {"type": "audit", "ref": "TAX-AUDIT-2025", "topic": "taxation"},
            {"type": "declaration", "ref": "COMP-POLICY-2025", "topic": "fair_competition"},
        ],
        "status": "completed",
        "notes": "All four minimum safeguard topics passed",
        "provenance_hash": _sha256("safeguard_pass"),
        "created_at": _now(),
        "updated_at": _now(),
    }


@pytest.fixture
def failing_safeguard_assessment(sample_org_id) -> Dict[str, Any]:
    """Safeguard assessment with adverse finding on anti-corruption."""
    return {
        "id": _new_id(),
        "org_id": sample_org_id,
        "assessment_date": _now(),
        "overall_pass": False,
        "topics": {
            "human_rights": {"procedural": True, "outcome": True, "overall": True},
            "anti_corruption": {"procedural": True, "outcome": False, "overall": False},
            "taxation": {"procedural": True, "outcome": True, "overall": True},
            "fair_competition": {"procedural": True, "outcome": True, "overall": True},
        },
        "status": "completed",
        "notes": "Anti-corruption: adverse finding in regulatory proceeding",
    }


@pytest.fixture
def sample_safeguard_topic_results() -> List[Dict[str, Any]]:
    """Per-topic safeguard results."""
    assessment_id = _new_id()
    return [
        {
            "id": _new_id(),
            "assessment_id": assessment_id,
            "topic": "human_rights",
            "procedural_pass": True,
            "outcome_pass": True,
            "overall_pass": True,
            "checks": {
                "ungp_due_diligence": True,
                "ilo_core_conventions": True,
                "human_rights_policy": True,
                "grievance_mechanism": True,
                "adverse_impact_assessment": True,
            },
            "evidence_items": [{"type": "declaration", "ref": "HR-POLICY-2025"}],
            "notes": "Full UNGP due diligence process in place",
        },
        {
            "id": _new_id(),
            "assessment_id": assessment_id,
            "topic": "anti_corruption",
            "procedural_pass": True,
            "outcome_pass": True,
            "overall_pass": True,
            "checks": {
                "anti_bribery_policy": True,
                "compliance_programme": True,
                "whistleblower_mechanism": True,
                "no_regulatory_sanctions": True,
                "iso37001_certified": True,
            },
            "evidence_items": [{"type": "certification", "ref": "ISO37001-2025"}],
        },
        {
            "id": _new_id(),
            "assessment_id": assessment_id,
            "topic": "taxation",
            "procedural_pass": True,
            "outcome_pass": True,
            "overall_pass": True,
            "checks": {
                "tax_governance_framework": True,
                "country_by_country_reporting": True,
                "transfer_pricing_compliance": True,
                "no_aggressive_tax_planning": True,
            },
            "evidence_items": [{"type": "audit", "ref": "TAX-AUDIT-2025"}],
        },
        {
            "id": _new_id(),
            "assessment_id": assessment_id,
            "topic": "fair_competition",
            "procedural_pass": True,
            "outcome_pass": True,
            "overall_pass": True,
            "checks": {
                "competition_compliance_policy": True,
                "training_programme": True,
                "no_antitrust_proceedings": True,
                "no_cartel_findings": True,
            },
            "evidence_items": [{"type": "declaration", "ref": "COMP-POLICY-2025"}],
        },
    ]


# ============================================================================
# KPI CALCULATION FIXTURES
# ============================================================================

@pytest.fixture
def sample_kpi_data(sample_org_id) -> Dict[str, Any]:
    """Turnover KPI calculation data."""
    return {
        "id": _new_id(),
        "org_id": sample_org_id,
        "period": "FY2025",
        "kpi_type": "turnover",
        "calculation_date": _now(),
        "eligible_amount": Decimal("1750000000.00"),
        "aligned_amount": Decimal("1050000000.00"),
        "total_amount": Decimal("2500000000.00"),
        "kpi_percentage": Decimal("42.0000"),
        "objective_breakdown": {
            "climate_mitigation": {
                "eligible": 1500000000,
                "aligned": 950000000,
                "percentage": 38.0,
            },
            "climate_adaptation": {
                "eligible": 250000000,
                "aligned": 100000000,
                "percentage": 4.0,
            },
        },
        "status": "calculated",
        "metadata": {
            "double_counting_check": True,
            "intercompany_eliminated": True,
        },
        "provenance_hash": _sha256("kpi_turnover_fy2025"),
        "created_at": _now(),
        "updated_at": _now(),
    }


@pytest.fixture
def sample_capex_kpi(sample_org_id) -> Dict[str, Any]:
    """CapEx KPI calculation data."""
    return {
        "id": _new_id(),
        "org_id": sample_org_id,
        "period": "FY2025",
        "kpi_type": "capex",
        "calculation_date": _now(),
        "eligible_amount": Decimal("420000000.00"),
        "aligned_amount": Decimal("315000000.00"),
        "total_amount": Decimal("600000000.00"),
        "kpi_percentage": Decimal("52.5000"),
        "objective_breakdown": {
            "climate_mitigation": {
                "eligible": 380000000,
                "aligned": 290000000,
                "percentage": 48.33,
            },
            "circular_economy": {
                "eligible": 40000000,
                "aligned": 25000000,
                "percentage": 4.17,
            },
        },
        "status": "calculated",
    }


@pytest.fixture
def sample_opex_kpi(sample_org_id) -> Dict[str, Any]:
    """OpEx KPI calculation data."""
    return {
        "id": _new_id(),
        "org_id": sample_org_id,
        "period": "FY2025",
        "kpi_type": "opex",
        "calculation_date": _now(),
        "eligible_amount": Decimal("85000000.00"),
        "aligned_amount": Decimal("51000000.00"),
        "total_amount": Decimal("200000000.00"),
        "kpi_percentage": Decimal("25.5000"),
        "objective_breakdown": {
            "climate_mitigation": {"eligible": 85000000, "aligned": 51000000},
        },
        "status": "calculated",
    }


@pytest.fixture
def sample_activity_financials(sample_org_id) -> List[Dict[str, Any]]:
    """Activity-level financial data."""
    return [
        {
            "id": _new_id(),
            "org_id": sample_org_id,
            "activity_code": "CCM_4.1",
            "period": "FY2025",
            "turnover": Decimal("800000000.00"),
            "capex": Decimal("200000000.00"),
            "opex": Decimal("40000000.00"),
            "eligible": True,
            "aligned": True,
            "objective": "climate_mitigation",
            "notes": "Solar PV operations",
        },
        {
            "id": _new_id(),
            "org_id": sample_org_id,
            "activity_code": "CCM_3.3",
            "period": "FY2025",
            "turnover": Decimal("500000000.00"),
            "capex": Decimal("120000000.00"),
            "opex": Decimal("25000000.00"),
            "eligible": True,
            "aligned": True,
            "objective": "climate_mitigation",
            "notes": "Cement operations - transitional",
        },
        {
            "id": _new_id(),
            "org_id": sample_org_id,
            "activity_code": "NON_ELIGIBLE_1",
            "period": "FY2025",
            "turnover": Decimal("400000000.00"),
            "capex": Decimal("80000000.00"),
            "opex": Decimal("60000000.00"),
            "eligible": False,
            "aligned": False,
            "objective": None,
            "notes": "Non-eligible activity",
        },
    ]


@pytest.fixture
def sample_capex_plan(sample_org_id) -> Dict[str, Any]:
    """CapEx plan for transitional CapEx reporting."""
    return {
        "id": _new_id(),
        "org_id": sample_org_id,
        "activity_code": "CCM_3.3",
        "start_year": 2025,
        "end_year": 2030,
        "planned_amounts": {
            "2025": 30000000,
            "2026": 45000000,
            "2027": 50000000,
            "2028": 40000000,
            "2029": 35000000,
            "2030": 20000000,
        },
        "actual_amounts": {
            "2025": 28000000,
        },
        "management_approved": True,
        "approved_date": date(2025, 1, 15),
        "status": "active",
        "notes": "Cement plant modernization for emissions reduction",
        "created_at": _now(),
        "updated_at": _now(),
    }


# ============================================================================
# GAR CALCULATION FIXTURES
# ============================================================================

@pytest.fixture
def sample_gar_data() -> Dict[str, Any]:
    """GAR stock calculation for a financial institution."""
    inst_id = _new_id()
    return {
        "id": _new_id(),
        "institution_id": inst_id,
        "period": "FY2025",
        "gar_type": "stock",
        "calculation_date": _now(),
        "aligned_assets": Decimal("52500000000.00"),
        "covered_assets": Decimal("280000000000.00"),
        "gar_percentage": Decimal("18.7500"),
        "sector_breakdown": {
            "energy": {"aligned": 15000000000, "covered": 45000000000, "pct": 33.33},
            "manufacturing": {"aligned": 12000000000, "covered": 80000000000, "pct": 15.00},
            "construction": {"aligned": 18000000000, "covered": 95000000000, "pct": 18.95},
            "transport": {"aligned": 5000000000, "covered": 35000000000, "pct": 14.29},
            "ict": {"aligned": 2500000000, "covered": 25000000000, "pct": 10.00},
        },
        "exposure_breakdown": {
            "corporate_loan": {"aligned": 30000000000, "covered": 150000000000},
            "retail_mortgage": {"aligned": 15000000000, "covered": 80000000000},
            "auto_loan": {"aligned": 2500000000, "covered": 20000000000},
            "project_finance": {"aligned": 5000000000, "covered": 30000000000},
        },
        "status": "calculated",
        "provenance_hash": _sha256("gar_stock_fy2025"),
        "created_at": _now(),
        "updated_at": _now(),
    }


@pytest.fixture
def sample_gar_flow() -> Dict[str, Any]:
    """GAR flow calculation (new originations)."""
    return {
        "id": _new_id(),
        "institution_id": _new_id(),
        "period": "FY2025",
        "gar_type": "flow",
        "calculation_date": _now(),
        "aligned_assets": Decimal("8500000000.00"),
        "covered_assets": Decimal("35000000000.00"),
        "gar_percentage": Decimal("24.2857"),
        "sector_breakdown": {
            "energy": {"aligned": 3000000000, "covered": 8000000000, "pct": 37.50},
            "construction": {"aligned": 4000000000, "covered": 15000000000, "pct": 26.67},
        },
        "status": "calculated",
    }


@pytest.fixture
def sample_btar_data() -> Dict[str, Any]:
    """Banking-book Taxonomy Alignment Ratio (BTAR) calculation."""
    return {
        "id": _new_id(),
        "institution_id": _new_id(),
        "period": "FY2025",
        "gar_type": "stock",
        "calculation_date": _now(),
        "aligned_assets": Decimal("48000000000.00"),
        "covered_assets": Decimal("320000000000.00"),
        "gar_percentage": Decimal("15.0000"),
        "status": "calculated",
        "metadata": {"ratio_type": "btar"},
    }


# ============================================================================
# EXPOSURE FIXTURES
# ============================================================================

@pytest.fixture
def sample_exposures() -> List[Dict[str, Any]]:
    """Financial institution exposure records."""
    inst_id = _new_id()
    return [
        {
            "id": _new_id(),
            "institution_id": inst_id,
            "portfolio_id": _new_id(),
            "counterparty_name": "SolarTech Europe GmbH",
            "nace_code": "D35.11",
            "exposure_type": "corporate_loan",
            "exposure_amount": Decimal("50000000.00"),
            "currency": "EUR",
            "epc_rating": None,
            "co2_gkm": None,
            "taxonomy_aligned": True,
            "alignment_pct": Decimal("95.00"),
            "reporting_date": date(2025, 12, 31),
        },
        {
            "id": _new_id(),
            "institution_id": inst_id,
            "portfolio_id": _new_id(),
            "counterparty_name": "Residential Portfolio - Tranche A",
            "nace_code": "L68.20",
            "exposure_type": "retail_mortgage",
            "exposure_amount": Decimal("200000000.00"),
            "currency": "EUR",
            "epc_rating": "A",
            "co2_gkm": None,
            "taxonomy_aligned": True,
            "alignment_pct": Decimal("100.00"),
            "reporting_date": date(2025, 12, 31),
        },
        {
            "id": _new_id(),
            "institution_id": inst_id,
            "portfolio_id": _new_id(),
            "counterparty_name": "Residential Portfolio - Tranche B",
            "nace_code": "L68.20",
            "exposure_type": "retail_mortgage",
            "exposure_amount": Decimal("150000000.00"),
            "currency": "EUR",
            "epc_rating": "B",
            "co2_gkm": None,
            "taxonomy_aligned": True,
            "alignment_pct": Decimal("100.00"),
            "reporting_date": date(2025, 12, 31),
        },
        {
            "id": _new_id(),
            "institution_id": inst_id,
            "portfolio_id": _new_id(),
            "counterparty_name": "Residential Portfolio - Tranche C",
            "nace_code": "L68.20",
            "exposure_type": "retail_mortgage",
            "exposure_amount": Decimal("300000000.00"),
            "currency": "EUR",
            "epc_rating": "D",
            "co2_gkm": None,
            "taxonomy_aligned": False,
            "alignment_pct": Decimal("0.00"),
            "reporting_date": date(2025, 12, 31),
        },
        {
            "id": _new_id(),
            "institution_id": inst_id,
            "portfolio_id": _new_id(),
            "counterparty_name": "EV Fleet Financing - Batch 1",
            "nace_code": "H49.10",
            "exposure_type": "auto_loan",
            "exposure_amount": Decimal("25000000.00"),
            "currency": "EUR",
            "epc_rating": None,
            "co2_gkm": Decimal("0.00"),
            "taxonomy_aligned": True,
            "alignment_pct": Decimal("100.00"),
            "reporting_date": date(2025, 12, 31),
        },
        {
            "id": _new_id(),
            "institution_id": inst_id,
            "portfolio_id": _new_id(),
            "counterparty_name": "ICE Vehicle Financing - Batch 2",
            "nace_code": "H49.10",
            "exposure_type": "auto_loan",
            "exposure_amount": Decimal("40000000.00"),
            "currency": "EUR",
            "epc_rating": None,
            "co2_gkm": Decimal("120.50"),
            "taxonomy_aligned": False,
            "alignment_pct": Decimal("0.00"),
            "reporting_date": date(2025, 12, 31),
        },
        {
            "id": _new_id(),
            "institution_id": inst_id,
            "portfolio_id": _new_id(),
            "counterparty_name": "WindPark North Sea Project",
            "nace_code": "D35.11",
            "exposure_type": "project_finance",
            "exposure_amount": Decimal("100000000.00"),
            "currency": "EUR",
            "epc_rating": None,
            "co2_gkm": None,
            "taxonomy_aligned": True,
            "alignment_pct": Decimal("100.00"),
            "reporting_date": date(2025, 12, 31),
        },
        {
            "id": _new_id(),
            "institution_id": inst_id,
            "portfolio_id": _new_id(),
            "counterparty_name": "EU Green Bond Fund",
            "nace_code": None,
            "exposure_type": "green_bond",
            "exposure_amount": Decimal("75000000.00"),
            "currency": "EUR",
            "epc_rating": None,
            "co2_gkm": None,
            "taxonomy_aligned": True,
            "alignment_pct": Decimal("88.00"),
            "reporting_date": date(2025, 12, 31),
        },
    ]


# ============================================================================
# ALIGNMENT RESULT FIXTURES
# ============================================================================

@pytest.fixture
def sample_alignment_result(sample_org_id) -> Dict[str, Any]:
    """Full 4-step alignment result (all pass)."""
    return {
        "id": _new_id(),
        "org_id": sample_org_id,
        "activity_code": "CCM_4.1",
        "period": "FY2025",
        "alignment_date": _now(),
        "eligible": True,
        "sc_pass": True,
        "dnsh_pass": True,
        "ms_pass": True,
        "aligned": True,
        "sc_objective": "climate_mitigation",
        "alignment_details": {
            "eligibility": {"status": "eligible", "confidence": 95.0},
            "sc": {"status": "pass", "type": "own_performance", "objective": "climate_mitigation"},
            "dnsh": {"status": "pass", "objectives_checked": 5, "objectives_passed": 5},
            "ms": {"status": "pass", "topics_checked": 4, "topics_passed": 4},
        },
        "provenance_hash": _sha256("alignment_ccm_4_1_fy2025"),
        "created_at": _now(),
        "updated_at": _now(),
    }


@pytest.fixture
def partial_alignment_result(sample_org_id) -> Dict[str, Any]:
    """Alignment result where DNSH fails."""
    return {
        "id": _new_id(),
        "org_id": sample_org_id,
        "activity_code": "CCM_3.3",
        "period": "FY2025",
        "alignment_date": _now(),
        "eligible": True,
        "sc_pass": True,
        "dnsh_pass": False,
        "ms_pass": True,
        "aligned": False,
        "sc_objective": "climate_mitigation",
        "alignment_details": {
            "eligibility": {"status": "eligible"},
            "sc": {"status": "pass"},
            "dnsh": {"status": "fail", "failing_objective": "pollution_prevention"},
            "ms": {"status": "pass"},
        },
    }


@pytest.fixture
def sample_portfolio_alignment(sample_org_id) -> Dict[str, Any]:
    """Portfolio-level alignment summary."""
    return {
        "id": _new_id(),
        "org_id": sample_org_id,
        "period": "FY2025",
        "total_activities": 15,
        "eligible_count": 10,
        "aligned_count": 7,
        "alignment_percentage": Decimal("46.6667"),
        "kpi_summary": {
            "turnover": {"eligible_pct": 70.0, "aligned_pct": 42.0},
            "capex": {"eligible_pct": 70.0, "aligned_pct": 52.5},
            "opex": {"eligible_pct": 42.5, "aligned_pct": 25.5},
        },
        "sector_breakdown": {
            "energy": {"total": 3, "eligible": 3, "aligned": 3},
            "manufacturing": {"total": 5, "eligible": 4, "aligned": 2},
            "construction": {"total": 4, "eligible": 2, "aligned": 1},
            "transport": {"total": 2, "eligible": 1, "aligned": 1},
            "ict": {"total": 1, "eligible": 0, "aligned": 0},
        },
        "provenance_hash": _sha256("portfolio_fy2025"),
        "created_at": _now(),
        "updated_at": _now(),
    }


# ============================================================================
# REPORT FIXTURES
# ============================================================================

@pytest.fixture
def sample_report(sample_org_id) -> Dict[str, Any]:
    """Generated Article 8 turnover report."""
    return {
        "id": _new_id(),
        "org_id": sample_org_id,
        "period": "FY2025",
        "template": "article_8_turnover",
        "format": "excel",
        "status": "generated",
        "generated_at": _now(),
        "download_url": "https://storage.greenlang.io/reports/taxonomy/art8_turnover_fy2025.xlsx",
        "content": {
            "summary": {
                "total_turnover": 2500000000,
                "eligible_turnover": 1750000000,
                "aligned_turnover": 1050000000,
                "eligible_pct": 70.0,
                "aligned_pct": 42.0,
            },
        },
        "metadata": {
            "generated_by": "system",
            "taxonomy_version": "2024",
        },
        "created_at": _now(),
        "updated_at": _now(),
    }


@pytest.fixture
def sample_eba_report() -> Dict[str, Any]:
    """EBA Pillar 3 template 7 report for financial institution."""
    return {
        "id": _new_id(),
        "org_id": _new_id(),
        "period": "FY2025",
        "template": "eba_template_7",
        "format": "excel",
        "status": "generated",
        "generated_at": _now(),
        "download_url": "https://storage.greenlang.io/reports/taxonomy/eba_t7_fy2025.xlsx",
        "content": {
            "gar_stock": {"percentage": 18.75, "aligned": 52500000000, "covered": 280000000000},
            "gar_flow": {"percentage": 24.29, "aligned": 8500000000, "covered": 35000000000},
        },
    }


# ============================================================================
# EVIDENCE FIXTURES
# ============================================================================

@pytest.fixture
def sample_evidence(sample_org_id) -> List[Dict[str, Any]]:
    """Evidence items across assessment types."""
    return [
        {
            "id": _new_id(),
            "org_id": sample_org_id,
            "assessment_type": "sc",
            "assessment_id": _new_id(),
            "evidence_type": "certification",
            "description": "ISO 14067 Carbon Footprint Certificate for solar PV panels",
            "document_ref": "ISO14067-2023-001",
            "uploaded_at": _now(),
            "verified": True,
            "verified_by": "Bureau Veritas",
            "verified_at": _now(),
        },
        {
            "id": _new_id(),
            "org_id": sample_org_id,
            "assessment_type": "dnsh",
            "assessment_id": _new_id(),
            "evidence_type": "report",
            "description": "Climate Risk Assessment per Taxonomy Appendix A",
            "document_ref": "CRA-SOLAR-2025",
            "uploaded_at": _now(),
            "verified": True,
            "verified_by": "Internal Audit",
            "verified_at": _now(),
        },
        {
            "id": _new_id(),
            "org_id": sample_org_id,
            "assessment_type": "safeguard",
            "assessment_id": _new_id(),
            "evidence_type": "declaration",
            "description": "Human Rights Due Diligence Statement per UNGP",
            "document_ref": "HR-DD-STATEMENT-2025",
            "uploaded_at": _now(),
            "verified": False,
            "verified_by": None,
            "verified_at": None,
        },
        {
            "id": _new_id(),
            "org_id": sample_org_id,
            "assessment_type": "dq",
            "assessment_id": _new_id(),
            "evidence_type": "data_extract",
            "description": "ERP data extract for turnover allocation",
            "document_ref": "ERP-EXTRACT-2025-Q4",
            "uploaded_at": _now(),
            "verified": True,
            "verified_by": "Finance Controller",
            "verified_at": _now(),
        },
    ]


# ============================================================================
# REGULATORY VERSION FIXTURES
# ============================================================================

@pytest.fixture
def sample_regulatory_versions() -> List[Dict[str, Any]]:
    """EU Taxonomy delegated act versions."""
    return [
        {
            "id": _new_id(),
            "delegated_act": "climate",
            "version_number": "1.0",
            "effective_date": date(2022, 1, 1),
            "amendment_details": {},
            "activities_affected": [],
            "status": "superseded",
        },
        {
            "id": _new_id(),
            "delegated_act": "climate",
            "version_number": "2.0",
            "effective_date": date(2024, 1, 1),
            "amendment_details": {
                "changes": ["Updated CCM_3.3 thresholds", "Added CCM_4.31 nuclear"],
            },
            "activities_affected": ["CCM_3.3", "CCM_4.31"],
            "status": "active",
        },
        {
            "id": _new_id(),
            "delegated_act": "environmental",
            "version_number": "1.0",
            "effective_date": date(2024, 1, 1),
            "amendment_details": {},
            "activities_affected": [],
            "status": "active",
        },
        {
            "id": _new_id(),
            "delegated_act": "complementary",
            "version_number": "1.0",
            "effective_date": date(2023, 1, 1),
            "amendment_details": {
                "scope": "Nuclear and gas activities",
            },
            "activities_affected": ["CCM_4.26", "CCM_4.27", "CCM_4.28", "CCM_4.29", "CCM_4.30", "CCM_4.31"],
            "status": "active",
        },
    ]


# ============================================================================
# DATA QUALITY FIXTURES
# ============================================================================

@pytest.fixture
def sample_data_quality(sample_org_id) -> Dict[str, Any]:
    """Data quality assessment result."""
    return {
        "id": _new_id(),
        "org_id": sample_org_id,
        "period": "FY2025",
        "assessment_date": _now(),
        "overall_score": Decimal("78.50"),
        "grade": "B+",
        "dimensions": {
            "completeness": {"score": 85.0, "description": "15% activities missing financial data"},
            "accuracy": {"score": 90.0, "description": "ERP-sourced data, minor rounding differences"},
            "timeliness": {"score": 70.0, "description": "Some data from prior quarter"},
            "consistency": {"score": 75.0, "description": "Cross-entity allocation method varies"},
            "traceability": {"score": 72.0, "description": "Evidence links incomplete for 3 activities"},
        },
        "improvement_actions": [
            {"action": "Complete financial mapping for 2 missing activities", "priority": "high"},
            {"action": "Standardize allocation methodology across entities", "priority": "medium"},
            {"action": "Automate ERP data feeds for Q4 closing", "priority": "medium"},
        ],
        "provenance_hash": _sha256("dq_fy2025"),
        "created_at": _now(),
        "updated_at": _now(),
    }


# ============================================================================
# GAP ASSESSMENT FIXTURES
# ============================================================================

@pytest.fixture
def sample_gap_assessment(sample_org_id) -> Dict[str, Any]:
    """Gap analysis result."""
    return {
        "id": _new_id(),
        "org_id": sample_org_id,
        "period": "FY2025",
        "assessment_date": _now(),
        "total_gaps": 8,
        "high_priority": 3,
        "gap_categories": {
            "sc": {"count": 2, "items": ["Steel SC threshold exceeded", "Missing LCA for CCM_4.3"]},
            "dnsh": {"count": 2, "items": ["BAT compliance gap for cement", "EIA pending for solar farm"]},
            "safeguard": {"count": 0},
            "data": {"count": 3, "items": ["Missing OpEx allocation", "Incomplete NACE mapping", "No CapEx plan for CCM_3.9"]},
            "regulatory": {"count": 1, "items": ["Environmental DA criteria not yet integrated"]},
        },
        "action_items": [
            {"priority": 1, "action": "Complete BAT compliance for cement DNSH", "effort": "high", "deadline": "2025-09-30"},
            {"priority": 2, "action": "Commission LCA for wind power activity", "effort": "medium", "deadline": "2025-06-30"},
            {"priority": 3, "action": "Resolve OpEx allocation methodology", "effort": "low", "deadline": "2025-04-30"},
        ],
        "status": "completed",
        "provenance_hash": _sha256("gap_fy2025"),
        "created_at": _now(),
        "updated_at": _now(),
    }


@pytest.fixture
def sample_gap_items() -> List[Dict[str, Any]]:
    """Individual gap items."""
    assessment_id = _new_id()
    return [
        {
            "id": _new_id(),
            "assessment_id": assessment_id,
            "category": "sc",
            "description": "Steel activity CCM_3.9 exceeds hot metal emissions threshold",
            "priority": "critical",
            "current_status": "1.520 tCO2e/t actual",
            "target_status": "1.331 tCO2e/t threshold",
            "action_required": "Implement scrap-based EAF process or carbon capture",
            "deadline": date(2026, 6, 30),
            "assigned_to": "Process Engineering Team",
            "status": "open",
        },
        {
            "id": _new_id(),
            "assessment_id": assessment_id,
            "category": "dnsh",
            "description": "Cement plant BAT compliance not demonstrated for pollution prevention",
            "priority": "high",
            "current_status": "Non-compliant",
            "target_status": "BAT-AEL compliant",
            "action_required": "Install BAT-compliant emissions abatement system",
            "deadline": date(2025, 12, 31),
            "assigned_to": "Environmental Operations",
            "status": "in_progress",
        },
        {
            "id": _new_id(),
            "assessment_id": assessment_id,
            "category": "data",
            "description": "OpEx allocation not completed for 3 activities",
            "priority": "high",
            "current_status": "Missing",
            "target_status": "Complete allocation per activity",
            "action_required": "Work with Finance to allocate OpEx by activity code",
            "deadline": date(2025, 4, 30),
            "assigned_to": "Finance Team",
            "status": "open",
        },
        {
            "id": _new_id(),
            "assessment_id": assessment_id,
            "category": "regulatory",
            "description": "Environmental Delegated Act criteria not yet integrated",
            "priority": "medium",
            "current_status": "Climate DA only",
            "target_status": "Both Climate and Environmental DA",
            "action_required": "Import Environmental DA criteria into assessment engine",
            "deadline": date(2025, 6, 30),
            "assigned_to": "Platform Engineering",
            "status": "open",
        },
        {
            "id": _new_id(),
            "assessment_id": assessment_id,
            "category": "safeguard",
            "description": "Anti-corruption ISO 37001 certification renewal due",
            "priority": "low",
            "current_status": "Expiring 2025-12-31",
            "target_status": "Renewed certification",
            "action_required": "Schedule ISO 37001 recertification audit",
            "deadline": date(2025, 10, 31),
            "assigned_to": "Compliance Team",
            "status": "open",
        },
    ]


# ============================================================================
# MOCK ENGINE FIXTURES
# ============================================================================

@pytest.fixture
def eligibility_engine() -> MagicMock:
    """Mock EligibilityScreeningEngine."""
    engine = MagicMock()
    engine.screen_activities = MagicMock()
    engine.lookup_nace = MagicMock()
    engine.check_de_minimis = MagicMock(return_value=False)
    engine.get_eligible_objectives = MagicMock(return_value=["climate_mitigation"])
    engine.batch_screen = MagicMock()
    engine.search_activity_catalog = MagicMock()
    engine.get_sector_breakdown = MagicMock()
    return engine


@pytest.fixture
def sc_engine() -> MagicMock:
    """Mock SubstantialContributionEngine."""
    engine = MagicMock()
    engine.assess = MagicMock()
    engine.evaluate_threshold = MagicMock()
    engine.classify_activity_type = MagicMock(return_value="own_performance")
    engine.check_enabling_criteria = MagicMock(return_value=True)
    engine.check_transitional_criteria = MagicMock(return_value=True)
    engine.record_evidence = MagicMock()
    engine.batch_assess = MagicMock()
    return engine


@pytest.fixture
def dnsh_engine() -> MagicMock:
    """Mock DNSHAssessmentEngine."""
    engine = MagicMock()
    engine.assess = MagicMock()
    engine.evaluate_objective = MagicMock()
    engine.assess_climate_risk = MagicMock()
    engine.check_water_dnsh = MagicMock(return_value=True)
    engine.check_circular_dnsh = MagicMock(return_value=True)
    engine.check_pollution_dnsh = MagicMock(return_value=True)
    engine.check_biodiversity_dnsh = MagicMock(return_value=True)
    engine.batch_assess = MagicMock()
    return engine


@pytest.fixture
def safeguard_engine() -> MagicMock:
    """Mock MinimumSafeguardsEngine."""
    engine = MagicMock()
    engine.assess = MagicMock()
    engine.check_topic = MagicMock()
    engine.check_human_rights = MagicMock(return_value=True)
    engine.check_anti_corruption = MagicMock(return_value=True)
    engine.check_taxation = MagicMock(return_value=True)
    engine.check_fair_competition = MagicMock(return_value=True)
    engine.record_adverse_finding = MagicMock()
    return engine


@pytest.fixture
def kpi_engine() -> MagicMock:
    """Mock KPICalculationEngine."""
    engine = MagicMock()
    engine.calculate = MagicMock()
    engine.calculate_turnover_kpi = MagicMock()
    engine.calculate_capex_kpi = MagicMock()
    engine.calculate_opex_kpi = MagicMock()
    engine.check_double_counting = MagicMock(return_value=False)
    engine.register_capex_plan = MagicMock()
    engine.get_objective_breakdown = MagicMock()
    return engine


@pytest.fixture
def gar_engine() -> MagicMock:
    """Mock GARCalculationEngine."""
    engine = MagicMock()
    engine.calculate_gar = MagicMock()
    engine.calculate_btar = MagicMock()
    engine.classify_exposure = MagicMock()
    engine.check_mortgage_alignment = MagicMock()
    engine.check_auto_loan_alignment = MagicMock()
    engine.calculate_flow_gar = MagicMock()
    engine.get_sector_breakdown = MagicMock()
    engine.get_exposure_breakdown = MagicMock()
    return engine


@pytest.fixture
def alignment_engine() -> MagicMock:
    """Mock AlignmentEngine."""
    engine = MagicMock()
    engine.run_full_alignment = MagicMock()
    engine.aggregate_portfolio = MagicMock()
    engine.get_alignment_progress = MagicMock()
    engine.get_dashboard_data = MagicMock()
    engine.compare_periods = MagicMock()
    return engine


@pytest.fixture
def reporting_engine() -> MagicMock:
    """Mock ReportingEngine."""
    engine = MagicMock()
    engine.generate_article_8 = MagicMock()
    engine.generate_eba_template = MagicMock()
    engine.generate_gar_summary = MagicMock()
    engine.export_pdf = MagicMock()
    engine.export_excel = MagicMock()
    engine.export_csv = MagicMock()
    engine.export_xbrl = MagicMock()
    engine.generate_qualitative_disclosure = MagicMock()
    engine.generate_executive_summary = MagicMock()
    return engine


@pytest.fixture
def data_quality_engine() -> MagicMock:
    """Mock DataQualityEngine."""
    engine = MagicMock()
    engine.assess = MagicMock()
    engine.score_completeness = MagicMock(return_value=85.0)
    engine.score_accuracy = MagicMock(return_value=90.0)
    engine.score_timeliness = MagicMock(return_value=70.0)
    engine.calculate_grade = MagicMock(return_value="B+")
    engine.generate_improvement_actions = MagicMock()
    return engine


@pytest.fixture
def gap_engine() -> MagicMock:
    """Mock GapAnalysisEngine."""
    engine = MagicMock()
    engine.run_full_analysis = MagicMock()
    engine.identify_sc_gaps = MagicMock()
    engine.identify_dnsh_gaps = MagicMock()
    engine.identify_data_gaps = MagicMock()
    engine.identify_regulatory_gaps = MagicMock()
    engine.generate_action_plan = MagicMock()
    engine.prioritize_gaps = MagicMock()
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
    return {"Authorization": "Bearer test-taxonomy-jwt-token"}


# ============================================================================
# DATA QUALITY ASSESSMENT FIXTURES (for test_data_quality.py)
# ============================================================================

@pytest.fixture
def sample_dq_assessment(sample_org_id) -> Dict[str, Any]:
    """Full 5-dimension data quality assessment."""
    return {
        "org_id": sample_org_id,
        "assessed_at": _now(),
        "overall_score": 78.5,
        "grade": "B",
        "dimensions": {
            "completeness": {
                "score": 85.0, "total_fields": 120,
                "populated_fields": 102,
                "missing_fields": ["opex_allocation", "nace_sub_code"],
            },
            "accuracy": {
                "score": 90.0, "validation_checks": 50,
                "checks_passed": 45,
            },
            "coverage": {
                "score": 70.0, "activities_covered": 10,
                "activities_total": 15, "objectives_assessed": 4,
            },
            "consistency": {
                "score": 75.0, "cross_ref_checks": 30,
                "discrepancies_found": 3,
            },
            "timeliness": {
                "score": 72.0, "avg_data_age_days": 45,
                "stale_records": 5,
            },
        },
        "evidence": [
            {"source": "ERP data extract", "dimension": "accuracy",
             "collected_at": _now()},
            {"source": "EPC registry", "dimension": "completeness",
             "collected_at": _now()},
        ],
        "improvement_plan": [
            {"priority": "high", "dimension": "coverage",
             "action": "Map remaining 5 activities to taxonomy codes"},
            {"priority": "medium", "dimension": "timeliness",
             "action": "Automate ERP data feed for quarterly closing"},
            {"priority": "low", "dimension": "consistency",
             "action": "Standardize allocation methodology across entities"},
        ],
        "provenance_hash": _sha256(f"{sample_org_id}_dq_assess"),
    }


@pytest.fixture
def sample_dq_trend() -> Dict[str, Any]:
    """DQ trend data across multiple periods."""
    return {
        "periods": [
            {
                "period": "2024-Q4", "overall_score": 65.0,
                "dimensions": {
                    "completeness": 70.0, "accuracy": 80.0,
                    "coverage": 55.0, "consistency": 60.0,
                    "timeliness": 60.0,
                },
            },
            {
                "period": "2025-Q2", "overall_score": 72.0,
                "dimensions": {
                    "completeness": 78.0, "accuracy": 85.0,
                    "coverage": 62.0, "consistency": 68.0,
                    "timeliness": 67.0,
                },
            },
            {
                "period": "2025-Q4", "overall_score": 78.5,
                "dimensions": {
                    "completeness": 85.0, "accuracy": 90.0,
                    "coverage": 70.0, "consistency": 75.0,
                    "timeliness": 72.0,
                },
            },
        ],
    }


# ============================================================================
# REGULATORY UPDATE FIXTURES (for test_regulatory_update.py)
# ============================================================================

@pytest.fixture
def sample_delegated_act() -> Dict[str, Any]:
    """Delegated act record with version tracking."""
    return {
        "act_id": _new_id(),
        "act_type": "climate",
        "version": "2023/2486",
        "effective_date": date(2024, 1, 1),
        "activities_count": 88,
        "status": "in_force",
        "version_history": [
            {"version": "2021/2139", "effective_date": date(2022, 1, 1)},
            {"version": "2023/2486", "effective_date": date(2024, 1, 1)},
        ],
    }


@pytest.fixture
def sample_tsc_update() -> Dict[str, Any]:
    """TSC update record."""
    return {
        "update_id": _new_id(),
        "activity_code": "CCM_3.3",
        "criteria_changes": [
            {"field": "cement_threshold_tco2e_t", "old_value": "0.498",
             "new_value": "0.469"},
        ],
        "impact_level": "medium",
        "affected_entities_count": 125,
        "effective_date": date(2024, 7, 1),
    }


@pytest.fixture
def sample_omnibus_impact() -> Dict[str, Any]:
    """Omnibus simplification impact assessment."""
    return {
        "assessment_id": _new_id(),
        "simplifications": [
            {"category": "reporting_threshold",
             "description": "SME reporting threshold raised to 1000 employees"},
            {"category": "voluntary_reporting",
             "description": "Voluntary reporting for non-CSRD entities"},
        ],
        "gar_impact": {
            "affected": True,
            "description": "GAR denominator exclusion for SME exposures",
        },
        "effective_date": date(2026, 1, 1),
        "transition_period_months": 24,
    }


@pytest.fixture
def sample_transition_plan() -> Dict[str, Any]:
    """Transition plan for regulatory update."""
    return {
        "plan_id": _new_id(),
        "milestones": [
            {"milestone": "Update TSC database", "target_date": date(2025, 3, 1)},
            {"milestone": "Recalculate affected GARs", "target_date": date(2025, 4, 1)},
            {"milestone": "Update EBA templates", "target_date": date(2025, 5, 1)},
        ],
        "owner": "Regulatory Compliance Team",
        "status": "in_progress",
        "completion_pct": 33.0,
    }


# ============================================================================
# GAP ANALYSIS FIXTURES (for test_gap_analysis.py)
# ============================================================================

@pytest.fixture
def sample_taxonomy_gap(sample_org_id) -> Dict[str, Any]:
    """Taxonomy gap analysis with all gap categories."""
    return {
        "org_id": sample_org_id,
        "assessed_at": _now(),
        "overall_readiness_pct": 62.0,
        "readiness_level": "moderate_gaps",
        "sc_gaps": [
            {"activity_code": "CCM_3.9", "description": "Steel SC threshold exceeded",
             "severity": "critical"},
        ],
        "dnsh_gaps": [
            {"activity_code": "CCM_3.3", "objective": "pollution_prevention",
             "description": "BAT compliance gap for cement plant",
             "severity": "high"},
            {"activity_code": "CCM_4.1", "objective": "biodiversity",
             "description": "EIA pending for new solar farm site",
             "severity": "medium"},
        ],
        "safeguard_gaps": [
            {"topic": "anti_corruption", "requirement": "ISO 37001 recertification due",
             "status": "partial"},
        ],
        "data_gaps": [
            {"field": "opex_allocation", "impact": "blocks_alignment",
             "severity": "high"},
            {"field": "epc_rating_missing", "impact": "reduces_score",
             "severity": "medium"},
        ],
        "action_plan": [
            {"priority": 1, "action": "Complete BAT compliance for cement DNSH",
             "effort": "high", "timeline_weeks": 12, "category": "dnsh"},
            {"priority": 2, "action": "Resolve OpEx allocation methodology",
             "effort": "low", "timeline_weeks": 4, "category": "data_quality"},
            {"priority": 3, "action": "Commission LCA for wind activity",
             "effort": "medium", "timeline_weeks": 8, "category": "sc"},
        ],
        "priority_matrix": {
            "quick_wins": [
                {"action": "Fix OpEx allocation", "impact": "high", "effort": "low"},
            ],
            "strategic_initiatives": [
                {"action": "BAT compliance upgrade", "impact": "critical", "effort": "high"},
            ],
            "fill_ins": [
                {"action": "Schedule EIA", "impact": "medium", "effort": "medium"},
            ],
            "low_priority": [
                {"action": "ISO 37001 renewal", "impact": "low", "effort": "low"},
            ],
        },
        "provenance_hash": _sha256(f"{sample_org_id}_taxonomy_gap"),
    }


# ============================================================================
# PORTFOLIO MANAGEMENT FIXTURES (for test_portfolio_management.py)
# ============================================================================

@pytest.fixture
def sample_taxonomy_portfolio() -> Dict[str, Any]:
    """Portfolio with holdings for portfolio management tests."""
    return {
        "portfolio_id": _new_id(),
        "institution_id": _new_id(),
        "portfolio_name": "FY2025 Corporate Loan Book",
        "reporting_date": date(2025, 12, 31),
        "currency": "EUR",
        "total_exposure": 525_000_000.0,
        "status": "draft",
        "holdings": [
            {
                "holding_id": _new_id(),
                "counterparty_name": "SolarTech Europe GmbH",
                "nace_code": "D35.11",
                "exposure_amount": 50_000_000.0,
                "exposure_type": "corporate_loan",
            },
            {
                "holding_id": _new_id(),
                "counterparty_name": "GreenBuild Construction SA",
                "nace_code": "F41.20",
                "exposure_amount": 75_000_000.0,
                "exposure_type": "corporate_loan",
            },
            {
                "holding_id": _new_id(),
                "counterparty_name": "Residential Mortgage Pool A",
                "nace_code": "L68.20",
                "exposure_amount": 200_000_000.0,
                "exposure_type": "mortgage",
            },
            {
                "holding_id": _new_id(),
                "counterparty_name": "WindPark North Sea SPV",
                "nace_code": "D35.11",
                "exposure_amount": 100_000_000.0,
                "exposure_type": "project_finance",
            },
            {
                "holding_id": _new_id(),
                "counterparty_name": "EV Fleet Leasing GmbH",
                "nace_code": "H49.10",
                "exposure_amount": 100_000_000.0,
                "exposure_type": "auto_loan",
            },
        ],
    }


@pytest.fixture
def sample_upload_result() -> Dict[str, Any]:
    """Upload processing result."""
    return {
        "upload_id": _new_id(),
        "status": "completed",
        "total_records": 500,
        "valid_records": 495,
        "invalid_records": 5,
        "validation_errors": [
            {"row": 42, "field": "nace_code", "message": "Invalid NACE format"},
            {"row": 103, "field": "exposure_amount", "message": "Negative value"},
        ],
        "nace_enrichment_count": 480,
    }


@pytest.fixture
def sample_portfolio_list() -> List[Dict[str, Any]]:
    """List of portfolio summaries."""
    return [
        {
            "portfolio_id": _new_id(),
            "portfolio_name": "FY2025 Corporate Loan Book",
            "total_exposure": 525_000_000.0,
            "holdings_count": 250,
            "status": "draft",
            "reporting_date": date(2025, 12, 31),
        },
        {
            "portfolio_id": _new_id(),
            "portfolio_name": "FY2025 Retail Mortgage Book",
            "total_exposure": 1_200_000_000.0,
            "holdings_count": 15000,
            "status": "submitted",
            "reporting_date": date(2025, 12, 31),
        },
    ]


# ============================================================================
# FI FEATURES FIXTURES (for test_fi_features.py)
# ============================================================================

@pytest.fixture
def sample_gar_result() -> Dict[str, Any]:
    """GAR stock calculation result."""
    return {
        "gar_type": "stock",
        "gar_pct": 18.75,
        "aligned_assets": 52_500_000_000.0,
        "covered_assets": 280_000_000_000.0,
        "excluded_categories": ["sovereign", "central_bank", "trading_book"],
        "sector_breakdown": [
            {"nace_section": "D", "exposure": 45_000_000_000.0,
             "aligned_exposure": 15_000_000_000.0, "sector_gar_pct": 33.33},
            {"nace_section": "C", "exposure": 80_000_000_000.0,
             "aligned_exposure": 12_000_000_000.0, "sector_gar_pct": 15.0},
            {"nace_section": "F", "exposure": 95_000_000_000.0,
             "aligned_exposure": 18_000_000_000.0, "sector_gar_pct": 18.95},
            {"nace_section": "H", "exposure": 35_000_000_000.0,
             "aligned_exposure": 5_000_000_000.0, "sector_gar_pct": 14.29},
            {"nace_section": "J", "exposure": 25_000_000_000.0,
             "aligned_exposure": 2_500_000_000.0, "sector_gar_pct": 10.0},
        ],
    }


@pytest.fixture
def sample_gar_flow_result() -> Dict[str, Any]:
    """GAR flow calculation result."""
    return {
        "gar_type": "flow",
        "gar_flow_pct": 24.29,
        "new_business_volume": 35_000_000_000.0,
        "aligned_new_business": 8_500_000_000.0,
        "period_start": date(2025, 1, 1),
        "period_end": date(2025, 12, 31),
    }


@pytest.fixture
def sample_btar_result() -> Dict[str, Any]:
    """BTAR calculation result."""
    return {
        "btar_pct": 15.0,
        "aligned_assets": 48_000_000_000.0,
        "covered_assets": 320_000_000_000.0,
        "includes_non_nfrd": True,
        "estimation_method": "proxy",
    }


@pytest.fixture
def sample_eba_template() -> Dict[str, Any]:
    """EBA Pillar III template data."""
    return {
        "template_number": 7,
        "eba_version": "3.2",
        "format": "xlsx",
        "columns": [
            "NACE Sector", "Total Exposure", "Eligible Exposure",
            "Aligned Exposure", "GAR %",
        ],
        "rows": [
            {"sector": "D", "total": 45e9, "eligible": 40e9,
             "aligned": 15e9, "gar_pct": 33.33},
            {"sector": "C", "total": 80e9, "eligible": 60e9,
             "aligned": 12e9, "gar_pct": 15.0},
        ],
    }


# ============================================================================
# KPI INTEGRATION FIXTURES (for test_kpi_integration.py)
# ============================================================================

@pytest.fixture
def sample_kpi_result(sample_org_id) -> Dict[str, Any]:
    """Full KPI calculation result with all 3 KPIs."""
    return {
        "org_id": sample_org_id,
        "reporting_period": "FY2025",
        "currency": "EUR",
        "turnover": {
            "total_denominator": 2_500_000_000.0,
            "eligible_amount": 1_750_000_000.0,
            "aligned_amount": 1_050_000_000.0,
            "eligible_not_aligned_amount": 700_000_000.0,
            "eligible_pct": 70.0,
            "aligned_pct": 42.0,
            "ias_reference": "IAS 1.82(a)",
        },
        "capex": {
            "total_denominator": 600_000_000.0,
            "eligible_amount": 420_000_000.0,
            "aligned_amount": 315_000_000.0,
            "eligible_not_aligned_amount": 105_000_000.0,
            "eligible_pct": 70.0,
            "aligned_pct": 52.5,
            "capex_plan_amount": 50_000_000.0,
            "ias_reference": "IAS 16",
        },
        "opex": {
            "total_denominator": 200_000_000.0,
            "eligible_amount": 85_000_000.0,
            "aligned_amount": 51_000_000.0,
            "eligible_not_aligned_amount": 34_000_000.0,
            "eligible_pct": 42.5,
            "aligned_pct": 25.5,
            "ias_reference": "IFRS 16",
            "denominator_includes": ["R&D", "renovation_measures",
                                      "short_term_leases", "maintenance"],
        },
        "by_objective": [
            {"objective": "climate_mitigation",
             "turnover_pct": 38.0, "capex_pct": 48.33, "opex_pct": 22.0},
            {"objective": "climate_adaptation",
             "turnover_pct": 4.0, "capex_pct": 4.17, "opex_pct": 3.5},
        ],
    }


@pytest.fixture
def sample_kpi_comparison() -> Dict[str, Any]:
    """KPI period-over-period comparison."""
    return {
        "current_period": "FY2025",
        "previous_period": "FY2024",
        "turnover_delta_pct": 5.0,
        "turnover_delta_abs": 125_000_000.0,
        "capex_delta_pct": 8.5,
        "capex_delta_abs": 51_000_000.0,
        "opex_delta_pct": -2.0,
        "opex_delta_abs": -4_000_000.0,
    }


# ============================================================================
# ALIGNMENT WORKFLOW FIXTURES (for test_alignment_workflow.py)
# ============================================================================

@pytest.fixture
def sample_alignment_result_workflow(sample_org_id) -> Dict[str, Any]:
    """Full 4-step alignment result for workflow tests."""
    return {
        "alignment_id": _new_id(),
        "activity_code": "CCM_4.1",
        "org_id": sample_org_id,
        "steps": {
            "eligibility": {"eligible": True, "confidence": 95.0},
            "substantial_contribution": {
                "passes": True, "objective": "climate_mitigation",
            },
            "dnsh": {
                "passes": True,
                "assessments": {
                    "climate_adaptation": "pass", "water_resources": "pass",
                    "circular_economy": "pass", "pollution_prevention": "pass",
                    "biodiversity": "pass",
                },
            },
            "minimum_safeguards": {
                "passes": True,
                "topics": {
                    "human_rights": "pass", "anti_corruption": "pass",
                    "taxation": "pass", "fair_competition": "pass",
                },
            },
        },
        "is_aligned": True,
        "alignment_pct": 100.0,
        "provenance_hash": _sha256(f"{sample_org_id}_alignment_workflow"),
    }


@pytest.fixture
def sample_portfolio_alignment_result(sample_org_id) -> Dict[str, Any]:
    """Portfolio alignment result for workflow tests."""
    return {
        "portfolio_id": _new_id(),
        "org_id": sample_org_id,
        "total_exposure": 525_000_000.0,
        "portfolio_eligibility_pct": 85.0,
        "portfolio_alignment_pct": 55.0,
        "holdings_alignment": [
            {"holding_id": _new_id(), "is_eligible": True,
             "is_aligned": True, "alignment_pct": 100.0},
            {"holding_id": _new_id(), "is_eligible": True,
             "is_aligned": False, "alignment_pct": 0.0},
            {"holding_id": _new_id(), "is_eligible": False,
             "is_aligned": False, "alignment_pct": 0.0},
        ],
    }


@pytest.fixture
def sample_alignment_dashboard(sample_org_id) -> Dict[str, Any]:
    """Alignment dashboard data."""
    return {
        "org_id": sample_org_id,
        "total_activities": 50,
        "eligible_activities": 35,
        "aligned_activities": 12,
        "eligibility_pct": 70.0,
        "alignment_pct": 24.0,
        "kpi_summary": {
            "turnover_aligned_pct": 42.0,
            "capex_aligned_pct": 52.5,
            "opex_aligned_pct": 25.5,
        },
        "by_objective": [
            {"objective": "climate_mitigation", "aligned_count": 10},
            {"objective": "climate_adaptation", "aligned_count": 2},
        ],
        "trend": [
            {"period": "FY2024", "alignment_pct": 18.0},
            {"period": "FY2025", "alignment_pct": 24.0},
        ],
        "funnel": {
            "total": 50, "eligible": 35, "sc_pass": 25,
            "dnsh_pass": 18, "ms_pass": 12, "aligned": 12,
        },
        "by_sector": [
            {"nace_section": "D", "sector_name": "Energy",
             "total_activities": 10, "aligned_activities": 7,
             "sector_alignment_pct": 70.0,
             "total_exposure": 45_000_000_000.0,
             "aligned_exposure": 31_500_000_000.0},
            {"nace_section": "C", "sector_name": "Manufacturing",
             "total_activities": 15, "aligned_activities": 3,
             "sector_alignment_pct": 20.0,
             "total_exposure": 80_000_000_000.0,
             "aligned_exposure": 16_000_000_000.0},
        ],
    }


# ============================================================================
# CLIMATE RISK FIXTURES (for test_climate_risk.py)
# ============================================================================

@pytest.fixture
def sample_climate_risk(sample_org_id) -> Dict[str, Any]:
    """Climate risk assessment for DNSH."""
    return {
        "assessment_id": _new_id(),
        "activity_code": "CCM_4.1",
        "org_id": sample_org_id,
        "location": {"country": "DE", "region": "Bavaria",
                      "latitude": 48.14, "longitude": 11.58},
        "scenarios": [
            {"name": "RCP4.5", "year": 2050},
            {"name": "RCP8.5", "year": 2050},
        ],
        "physical_risks": [
            {"risk_type": "chronic", "hazard": "temperature_change",
             "severity": "medium", "likelihood": "likely",
             "return_period_years": None},
            {"risk_type": "chronic", "hazard": "water_stress",
             "severity": "low", "likelihood": "possible",
             "return_period_years": None},
            {"risk_type": "acute", "hazard": "heatwave",
             "severity": "medium", "likelihood": "likely",
             "return_period_years": 5, "potential_impact": "moderate"},
            {"risk_type": "acute", "hazard": "flood",
             "severity": "low", "likelihood": "unlikely",
             "return_period_years": 50, "potential_impact": "minor"},
        ],
        "materiality": "material",
        "adaptation_required": True,
        "adaptation_solutions": [
            {"description": "Enhanced cooling systems for PV inverters",
             "addresses_risk": "temperature_change",
             "estimated_cost": 250_000, "implementation_months": 6,
             "effectiveness": "high"},
            {"description": "Temperature-resistant panel coatings",
             "addresses_risk": "heatwave",
             "estimated_cost": 180_000, "implementation_months": 3,
             "effectiveness": "medium"},
        ],
        "residual_risk_level": "low",
        "dnsh_passes": True,
        "time_horizons": {
            "short": {"risk_level": "low", "years": 10},
            "medium": {"risk_level": "medium", "years": 30},
            "long": {"risk_level": "medium", "years": 50},
        },
        "asset_lifetime_years": 25,
    }


@pytest.fixture
def sample_multi_location_risk() -> Dict[str, Any]:
    """Multi-location climate risk assessment."""
    return {
        "assessment_id": _new_id(),
        "locations": [
            {"latitude": 48.14, "longitude": 11.58, "name": "Bavaria, DE",
             "risk_profile": [
                 {"hazard": "heatwave", "severity": "medium"},
             ]},
            {"latitude": 40.42, "longitude": -3.70, "name": "Madrid, ES",
             "risk_profile": [
                 {"hazard": "drought", "severity": "high"},
                 {"hazard": "heatwave", "severity": "high"},
             ]},
        ],
        "aggregate_risk_level": "high",
        "highest_risk_location": "Madrid, ES",
    }
