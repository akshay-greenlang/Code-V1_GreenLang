# -*- coding: utf-8 -*-
"""
PACK-001 CSRD Starter Pack - Shared Test Fixtures
===================================================

Provides reusable pytest fixtures for all PACK-001 test modules including
sample configuration, company profiles, ESG data, pre-computed results,
and mocked infrastructure.

Author: GreenLang QA Team
Version: 1.0.0
"""

import json
import os
import tempfile
from datetime import date, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest
import yaml


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

PACK_ROOT = Path(__file__).resolve().parent.parent
PACK_YAML_PATH = PACK_ROOT / "pack.yaml"
CONFIG_DIR = PACK_ROOT / "config"
PRESETS_DIR = CONFIG_DIR / "presets"
SECTORS_DIR = CONFIG_DIR / "sectors"
DEMO_DIR = CONFIG_DIR / "demo"
WORKFLOWS_DIR = PACK_ROOT / "workflows"
TEMPLATES_DIR = PACK_ROOT / "templates"
INTEGRATIONS_DIR = PACK_ROOT / "integrations"


# ---------------------------------------------------------------------------
# Pack YAML fixture
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def pack_yaml_path() -> Path:
    """Return the absolute path to pack.yaml."""
    return PACK_YAML_PATH


@pytest.fixture(scope="session")
def pack_yaml_raw(pack_yaml_path) -> str:
    """Return the raw text content of pack.yaml."""
    return pack_yaml_path.read_text(encoding="utf-8")


@pytest.fixture(scope="session")
def pack_yaml(pack_yaml_raw) -> Dict[str, Any]:
    """Return the parsed pack.yaml as a dictionary."""
    return yaml.safe_load(pack_yaml_raw)


# ---------------------------------------------------------------------------
# Pack configuration fixture
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_pack_config() -> Dict[str, Any]:
    """Create a sample PackConfig dictionary matching demo defaults.

    This mirrors the expected output of PackConfig.load() when run
    with the demo_config preset, suitable for fast unit-test execution.
    """
    return {
        "metadata": {
            "name": "csrd-starter",
            "version": "1.0.0",
            "category": "eu-compliance",
            "display_name": "CSRD Starter Pack",
        },
        "size_preset": "mid_market",
        "sector_preset": "manufacturing",
        "reporting_year": 2025,
        "reporting_period_start": "2025-01-01",
        "reporting_period_end": "2025-12-31",
        "esrs_standards": [
            "ESRS_1", "ESRS_2",
            "E1", "E2", "E3", "E4", "E5",
            "S1", "S2", "S3", "S4",
            "G1",
        ],
        "scope3_categories_enabled": [1, 3, 4, 5, 6],
        "xbrl_enabled": True,
        "language": "en",
        "consolidation_approach": "operational_control",
        "performance_targets": {
            "full_report_max_minutes": 30,
            "quarterly_update_max_minutes": 15,
            "materiality_max_minutes": 10,
            "health_check_max_seconds": 30,
        },
    }


# ---------------------------------------------------------------------------
# Company profile fixture
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_company_profile() -> Dict[str, Any]:
    """Create a demo company profile for GreenTech Manufacturing GmbH.

    This fixture represents a mid-market European manufacturing company
    subject to CSRD reporting requirements. All values are realistic
    enough for integration testing but entirely fictional.
    """
    return {
        "company_name": "GreenTech Manufacturing GmbH",
        "legal_entity_id": "DE-HRB-123456",
        "lei": "529900ABCDEFGHIJK012",
        "country": "DE",
        "sector": "manufacturing",
        "nace_code": "C25.1",
        "employees": 1850,
        "revenue_eur": 285_000_000,
        "total_assets_eur": 420_000_000,
        "listed": False,
        "reporting_year": 2025,
        "fiscal_year_end": "2025-12-31",
        "subsidiaries": [
            {
                "name": "GreenTech Poland Sp. z o.o.",
                "country": "PL",
                "employees": 320,
                "ownership_pct": 100.0,
            },
            {
                "name": "GreenTech France SAS",
                "country": "FR",
                "employees": 180,
                "ownership_pct": 100.0,
            },
        ],
        "facilities": [
            {
                "name": "Dusseldorf Main Plant",
                "country": "DE",
                "type": "manufacturing",
                "floor_area_m2": 45_000,
            },
            {
                "name": "Krakow Assembly",
                "country": "PL",
                "type": "assembly",
                "floor_area_m2": 18_000,
            },
            {
                "name": "Lyon Office",
                "country": "FR",
                "type": "office",
                "floor_area_m2": 2_500,
            },
        ],
        "data_sources": ["erp_sap", "excel_manual", "utility_invoices", "questionnaire"],
        "previous_csrd_report": False,
        "auditor": "EY Deutschland",
    }


# ---------------------------------------------------------------------------
# Sample ESG data fixture (50-record subset)
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_esg_data() -> List[Dict[str, Any]]:
    """Create a 50-record subset of ESG data for fast testing.

    Covers a representative mix across Scope 1, 2, and 3 categories
    plus social and governance metrics.  Each record has a consistent
    schema with id, esrs_standard, data_point, value, unit, source,
    quality_score, and period.
    """
    records: List[Dict[str, Any]] = []

    # --- Scope 1: Stationary combustion (10 records) ---
    fuels = [
        ("natural_gas", 1_250_000, "m3", 2.68),
        ("diesel", 85_000, "litres", 2.68),
        ("lpg", 12_000, "litres", 1.51),
        ("heating_oil", 42_000, "litres", 2.54),
        ("coal", 500, "tonnes", 2394.0),
    ]
    for i, (fuel, qty, unit, ef) in enumerate(fuels, start=1):
        records.append({
            "id": f"SC1-STAT-{i:03d}",
            "esrs_standard": "E1",
            "data_point": "E1-6_01",
            "category": "scope1_stationary_combustion",
            "sub_category": fuel,
            "value": qty,
            "unit": unit,
            "emission_factor": ef,
            "source": "erp_sap",
            "quality_score": 0.95,
            "period": "2025-Q1",
            "facility": "Dusseldorf Main Plant",
        })

    # --- Scope 1: Fugitive / refrigerant (5 records) ---
    refrigerants = [
        ("R-410A", 12.5, 2088),
        ("R-134a", 8.2, 1430),
        ("R-407C", 5.0, 1774),
        ("R-32", 3.0, 675),
        ("HFC-125", 1.5, 3500),
    ]
    for i, (ref, kg, gwp) in enumerate(refrigerants, start=1):
        records.append({
            "id": f"SC1-REF-{i:03d}",
            "esrs_standard": "E1",
            "data_point": "E1-6_03",
            "category": "scope1_refrigerants",
            "sub_category": ref,
            "value": kg,
            "unit": "kg",
            "emission_factor": gwp,
            "source": "maintenance_log",
            "quality_score": 0.85,
            "period": "2025-Q1",
            "facility": "Dusseldorf Main Plant",
        })

    # --- Scope 2: Electricity (5 records) ---
    electricity = [
        ("DE", 8_500_000, 0.385),
        ("PL", 2_200_000, 0.681),
        ("FR", 850_000, 0.052),
        ("DE", 7_900_000, 0.385),
        ("PL", 2_400_000, 0.681),
    ]
    for i, (country, kwh, ef) in enumerate(electricity, start=1):
        records.append({
            "id": f"SC2-ELEC-{i:03d}",
            "esrs_standard": "E1",
            "data_point": "E1-6_04",
            "category": "scope2_electricity",
            "sub_category": f"grid_{country}",
            "value": kwh,
            "unit": "kWh",
            "emission_factor": ef,
            "source": "utility_invoices",
            "quality_score": 0.98,
            "period": "2025-Q1" if i <= 3 else "2025-Q2",
            "facility": (
                "Dusseldorf Main Plant" if country == "DE"
                else "Krakow Assembly" if country == "PL"
                else "Lyon Office"
            ),
        })

    # --- Scope 3: Purchased goods (5 records) ---
    goods = [
        ("steel_plate", 4500, "tonnes", 1.85),
        ("aluminum_ingot", 1200, "tonnes", 8.24),
        ("plastics_pe", 800, "tonnes", 2.50),
        ("copper_wire", 350, "tonnes", 3.81),
        ("packaging_cardboard", 200, "tonnes", 0.94),
    ]
    for i, (mat, qty, unit, ef) in enumerate(goods, start=1):
        records.append({
            "id": f"SC3-PG-{i:03d}",
            "esrs_standard": "E1",
            "data_point": "E1-6_06",
            "category": "scope3_cat1_purchased_goods",
            "sub_category": mat,
            "value": qty,
            "unit": unit,
            "emission_factor": ef,
            "source": "erp_sap",
            "quality_score": 0.80,
            "period": "2025-Q1",
            "facility": "Dusseldorf Main Plant",
        })

    # --- Scope 3: Business travel (5 records) ---
    travel = [
        ("short_haul_flight", 120_000, "pkm", 0.255),
        ("long_haul_flight", 450_000, "pkm", 0.195),
        ("rail", 85_000, "pkm", 0.041),
        ("car_rental", 62_000, "km", 0.171),
        ("hotel_nights", 1200, "nights", 20.6),
    ]
    for i, (mode, qty, unit, ef) in enumerate(travel, start=1):
        records.append({
            "id": f"SC3-BT-{i:03d}",
            "esrs_standard": "E1",
            "data_point": "E1-6_08",
            "category": "scope3_cat6_business_travel",
            "sub_category": mode,
            "value": qty,
            "unit": unit,
            "emission_factor": ef,
            "source": "travel_system",
            "quality_score": 0.90,
            "period": "2025-Q1",
            "facility": "all",
        })

    # --- Social: Workforce (10 records) ---
    social_items = [
        ("S1", "S1-6_01", "total_employees", 1850, "headcount"),
        ("S1", "S1-6_02", "female_employees_pct", 38.5, "percent"),
        ("S1", "S1-6_03", "employee_turnover_rate", 12.3, "percent"),
        ("S1", "S1-6_04", "training_hours_per_employee", 24.5, "hours"),
        ("S1", "S1-6_05", "gender_pay_gap", 8.2, "percent"),
        ("S1", "S1-8_01", "work_related_injuries", 7, "count"),
        ("S1", "S1-8_02", "lost_time_injury_rate", 1.8, "per_million_hours"),
        ("S2", "S2-1_01", "supplier_audits_conducted", 42, "count"),
        ("S2", "S2-1_02", "suppliers_with_code_of_conduct", 85.0, "percent"),
        ("S4", "S4-1_01", "customer_satisfaction_score", 78.5, "score"),
    ]
    for i, (std, dp, metric, val, unit) in enumerate(social_items, start=1):
        records.append({
            "id": f"SOC-{i:03d}",
            "esrs_standard": std,
            "data_point": dp,
            "category": "social",
            "sub_category": metric,
            "value": val,
            "unit": unit,
            "emission_factor": None,
            "source": "hr_system",
            "quality_score": 0.92,
            "period": "2025-Q1",
            "facility": "all",
        })

    # --- Governance (5 records) ---
    gov_items = [
        ("G1", "G1-1_01", "board_independence_pct", 66.7, "percent"),
        ("G1", "G1-1_02", "board_gender_diversity_pct", 33.3, "percent"),
        ("G1", "G1-2_01", "anti_corruption_training_pct", 95.0, "percent"),
        ("G1", "G1-3_01", "whistleblower_cases", 2, "count"),
        ("G1", "G1-4_01", "political_contributions_eur", 0, "EUR"),
    ]
    for i, (std, dp, metric, val, unit) in enumerate(gov_items, start=1):
        records.append({
            "id": f"GOV-{i:03d}",
            "esrs_standard": std,
            "data_point": dp,
            "category": "governance",
            "sub_category": metric,
            "value": val,
            "unit": unit,
            "emission_factor": None,
            "source": "governance_report",
            "quality_score": 0.99,
            "period": "2025",
            "facility": "all",
        })

    # --- Scope 1: Mobile combustion (5 records) ---
    vehicles = [
        ("diesel_truck", 45_000, "litres", 2.68),
        ("petrol_car", 12_000, "litres", 2.31),
        ("diesel_van", 8_500, "litres", 2.68),
        ("lpg_forklift", 3_200, "litres", 1.51),
        ("diesel_generator", 2_800, "litres", 2.68),
    ]
    for i, (vehicle, qty, unit, ef) in enumerate(vehicles, start=1):
        records.append({
            "id": f"SC1-MOB-{i:03d}",
            "esrs_standard": "E1",
            "data_point": "E1-6_02",
            "category": "scope1_mobile_combustion",
            "sub_category": vehicle,
            "value": qty,
            "unit": unit,
            "emission_factor": ef,
            "source": "fleet_system",
            "quality_score": 0.90,
            "period": "2025-Q1",
            "facility": "Dusseldorf Main Plant",
        })

    # --- Water / biodiversity / circular economy (5 records) ---
    env_items = [
        ("E2", "E2-4_01", "total_air_pollutants_tonnes", 12.5, "tonnes"),
        ("E3", "E3-4_01", "total_water_withdrawal_m3", 380_000, "m3"),
        ("E3", "E3-4_02", "water_recycled_pct", 42.0, "percent"),
        ("E4", "E4-5_01", "sites_near_biodiversity_areas", 1, "count"),
        ("E5", "E5-5_01", "waste_recycling_rate_pct", 68.0, "percent"),
    ]
    for i, (std, dp, metric, val, unit) in enumerate(env_items, start=1):
        records.append({
            "id": f"ENV-{i:03d}",
            "esrs_standard": std,
            "data_point": dp,
            "category": "environment",
            "sub_category": metric,
            "value": val,
            "unit": unit,
            "emission_factor": None,
            "source": "environmental_report",
            "quality_score": 0.88,
            "period": "2025",
            "facility": "all",
        })

    assert len(records) == 50, f"Expected 50 records, got {len(records)}"
    return records


# ---------------------------------------------------------------------------
# Pre-computed materiality assessment result
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_materiality_result() -> Dict[str, Any]:
    """Pre-computed materiality assessment for GreenTech Manufacturing.

    Contains a double materiality matrix with both impact and financial
    scores for each ESRS topic. Used as a known-good reference for tests
    that validate downstream behaviour given a fixed materiality output.
    """
    return {
        "company": "GreenTech Manufacturing GmbH",
        "assessment_date": "2025-06-15",
        "methodology": "ESRS_1_double_materiality",
        "material_topics": [
            {
                "topic": "E1 - Climate Change",
                "impact_score": 9.2,
                "financial_score": 8.8,
                "is_material": True,
                "rationale": "High process emissions, energy-intensive manufacturing",
            },
            {
                "topic": "E2 - Pollution",
                "impact_score": 7.5,
                "financial_score": 6.0,
                "is_material": True,
                "rationale": "Industrial air emissions, chemical handling",
            },
            {
                "topic": "E3 - Water & Marine Resources",
                "impact_score": 5.8,
                "financial_score": 4.2,
                "is_material": False,
                "rationale": "Moderate water use, no marine operations",
            },
            {
                "topic": "E4 - Biodiversity & Ecosystems",
                "impact_score": 3.5,
                "financial_score": 2.8,
                "is_material": False,
                "rationale": "No direct biodiversity impact from operations",
            },
            {
                "topic": "E5 - Circular Economy",
                "impact_score": 7.0,
                "financial_score": 7.5,
                "is_material": True,
                "rationale": "Significant material waste, recycling opportunity",
            },
            {
                "topic": "S1 - Own Workforce",
                "impact_score": 8.0,
                "financial_score": 7.0,
                "is_material": True,
                "rationale": "Large workforce, health and safety risks",
            },
            {
                "topic": "S2 - Workers in Value Chain",
                "impact_score": 6.5,
                "financial_score": 5.5,
                "is_material": True,
                "rationale": "Complex supply chain across EU and non-EU",
            },
            {
                "topic": "S3 - Affected Communities",
                "impact_score": 3.0,
                "financial_score": 2.5,
                "is_material": False,
                "rationale": "Industrial sites in established zones",
            },
            {
                "topic": "S4 - Consumers & End-Users",
                "impact_score": 4.0,
                "financial_score": 3.5,
                "is_material": False,
                "rationale": "B2B products, limited end-user contact",
            },
            {
                "topic": "G1 - Business Conduct",
                "impact_score": 7.8,
                "financial_score": 8.5,
                "is_material": True,
                "rationale": "Anti-corruption, ethics compliance critical",
            },
        ],
        "materiality_threshold": 6.0,
        "total_topics_assessed": 10,
        "total_material": 6,
        "total_non_material": 4,
        "provenance_hash": "a3f7c2d8e1b5a9c4d6e2f8b7a1c3d5e7f9a2b4c6d8e0f1a3b5c7d9e1f3a5b7",
    }


# ---------------------------------------------------------------------------
# Pre-computed calculation result
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_calculation_result() -> Dict[str, Any]:
    """Pre-computed GHG calculation summary for GreenTech Manufacturing.

    Provides known-good Scope 1, 2, and 3 emission values for testing
    downstream components such as templates, auditor packages, and
    compliance checks.
    """
    return {
        "company": "GreenTech Manufacturing GmbH",
        "reporting_year": 2025,
        "currency": "EUR",
        "scope1": {
            "total_tco2e": 4_285.3,
            "stationary_combustion_tco2e": 3_652.1,
            "mobile_combustion_tco2e": 245.8,
            "process_emissions_tco2e": 312.4,
            "fugitive_emissions_tco2e": 75.0,
            "methodology": "GHG_Protocol_Corporate_Standard",
        },
        "scope2": {
            "location_based_tco2e": 5_124.7,
            "market_based_tco2e": 3_891.2,
            "electricity_tco2e": 4_850.0,
            "steam_heat_tco2e": 274.7,
            "methodology": "GHG_Protocol_Scope2_Guidance",
        },
        "scope3": {
            "total_tco2e": 28_456.9,
            "cat1_purchased_goods_tco2e": 18_230.5,
            "cat3_fuel_energy_tco2e": 1_850.0,
            "cat4_upstream_transport_tco2e": 3_200.4,
            "cat5_waste_generated_tco2e": 980.0,
            "cat6_business_travel_tco2e": 196.0,
            "categories_reported": [1, 3, 4, 5, 6],
            "methodology": "GHG_Protocol_Corporate_Value_Chain",
        },
        "total_tco2e": 37_866.9,
        "intensity_tco2e_per_meur_revenue": 132.9,
        "intensity_tco2e_per_employee": 20.5,
        "data_quality_score": 0.89,
        "provenance_hash": "b4c8d2e6f0a3b7c1d5e9f2a6b0c4d8e1f5a9b3c7d0e4f8a2b6c0d3e7f1a5b9",
        "calculation_timestamp": "2025-07-15T10:30:00Z",
    }


# ---------------------------------------------------------------------------
# Mock agent registry
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_agent_registry() -> MagicMock:
    """Create a mocked agent registry with all pack agents registered.

    Returns a MagicMock that behaves like a VersionedAgentRegistry,
    with pre-configured responses for agent lookup, health checks,
    and capability queries.
    """
    registry = MagicMock()

    # Maintain a lookup of all agents in the pack
    all_agent_ids = (
        # Data agents
        [f"AGENT-DATA-{i:03d}" for i in [1, 2, 3, 8]]
        # Quality agents
        + [f"AGENT-DATA-{i:03d}" for i in [10, 11, 12, 13, 19]]
        # MRV Scope 1
        + [f"AGENT-MRV-{i:03d}" for i in range(1, 9)]
        # MRV Scope 2
        + [f"AGENT-MRV-{i:03d}" for i in range(9, 14)]
        # MRV Scope 3
        + [f"AGENT-MRV-{i:03d}" for i in range(14, 31)]
        # Foundation
        + [f"AGENT-FOUND-{i:03d}" for i in range(1, 11)]
        # App
        + ["GL-CSRD-APP"]
    )

    def is_available(agent_id: str) -> bool:
        return agent_id in all_agent_ids

    def get_agent(agent_id: str) -> MagicMock:
        if agent_id not in all_agent_ids:
            raise KeyError(f"Agent {agent_id} not registered")
        agent_mock = MagicMock()
        agent_mock.agent_id = agent_id
        agent_mock.status = "healthy"
        agent_mock.version = "1.0.0"
        return agent_mock

    registry.is_available.side_effect = is_available
    registry.get_agent.side_effect = get_agent
    registry.list_agents.return_value = all_agent_ids
    registry.health_check.return_value = {aid: "healthy" for aid in all_agent_ids}
    return registry


# ---------------------------------------------------------------------------
# Temporary output directory
# ---------------------------------------------------------------------------

@pytest.fixture
def temp_output_dir(tmp_path) -> Path:
    """Create a temporary directory for test outputs.

    Yields a Path object pointing to a temp directory that is automatically
    cleaned up when the test completes.
    """
    output_dir = tmp_path / "pack_output"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


# ---------------------------------------------------------------------------
# Preset YAML helpers
# ---------------------------------------------------------------------------

LARGE_ENTERPRISE_PRESET = {
    "preset_id": "large_enterprise",
    "display_name": "Large Enterprise",
    "esrs_standards": [
        "ESRS_1", "ESRS_2",
        "E1", "E2", "E3", "E4", "E5",
        "S1", "S2", "S3", "S4",
        "G1",
    ],
    "scope3_categories": list(range(1, 16)),
    "xbrl_mode": "full",
    "consolidation": "multi_subsidiary",
    "languages": ["en", "de", "fr", "es"],
    "audit_package": True,
    "max_data_points": 1082,
    "performance_targets": {
        "full_report_max_minutes": 30,
        "quarterly_update_max_minutes": 15,
    },
}

MID_MARKET_PRESET = {
    "preset_id": "mid_market",
    "display_name": "Mid-Market Company",
    "esrs_standards": [
        "ESRS_1", "ESRS_2",
        "E1", "E2", "E3", "E4", "E5",
        "S1", "S2", "S3", "S4",
        "G1",
    ],
    "scope3_categories": [1, 3, 4, 5, 6],
    "xbrl_mode": "standard",
    "consolidation": "single_entity",
    "languages": ["en"],
    "audit_package": True,
    "max_data_points": 800,
    "performance_targets": {
        "full_report_max_minutes": 30,
        "quarterly_update_max_minutes": 15,
    },
}

SME_PRESET = {
    "preset_id": "sme",
    "display_name": "SME",
    "esrs_standards": ["ESRS_1", "ESRS_2", "E1", "S1", "G1"],
    "scope3_categories": [],
    "xbrl_mode": "basic",
    "consolidation": "single_entity",
    "languages": ["en"],
    "audit_package": False,
    "max_data_points": 200,
    "performance_targets": {
        "full_report_max_minutes": 15,
        "quarterly_update_max_minutes": 10,
    },
}

FIRST_TIME_PRESET = {
    "preset_id": "first_time_reporter",
    "display_name": "First-Time Reporter",
    "esrs_standards": [
        "ESRS_1", "ESRS_2",
        "E1", "E2", "E3", "E4", "E5",
        "S1", "S2", "S3", "S4",
        "G1",
    ],
    "scope3_categories": [1, 6],
    "xbrl_mode": "standard",
    "consolidation": "single_entity",
    "languages": ["en"],
    "audit_package": True,
    "tutorial_mode": True,
    "ai_assist_level": "high",
    "max_data_points": 600,
    "performance_targets": {
        "full_report_max_minutes": 45,
        "quarterly_update_max_minutes": 20,
    },
}

ALL_PRESETS = {
    "large_enterprise": LARGE_ENTERPRISE_PRESET,
    "mid_market": MID_MARKET_PRESET,
    "sme": SME_PRESET,
    "first_time_reporter": FIRST_TIME_PRESET,
}


@pytest.fixture(params=["large_enterprise", "mid_market", "sme", "first_time_reporter"])
def preset_config(request) -> Dict[str, Any]:
    """Parameterized fixture that yields each size preset in turn."""
    return ALL_PRESETS[request.param]


# ---------------------------------------------------------------------------
# Sector preset helpers
# ---------------------------------------------------------------------------

MANUFACTURING_SECTOR = {
    "sector_id": "manufacturing",
    "display_name": "Manufacturing",
    "nace_codes": ["C10-C33"],
    "emission_focus": ["scope1_process", "scope1_fugitive", "scope1_stationary"],
    "scope1_emphasis": True,
    "scope2_emphasis": True,
    "scope3_emphasis": False,
    "key_agents": ["AGENT-MRV-001", "AGENT-MRV-004", "AGENT-MRV-005"],
    "industry_emission_factors": True,
}

FINANCIAL_SERVICES_SECTOR = {
    "sector_id": "financial_services",
    "display_name": "Financial Services",
    "nace_codes": ["K64-K66"],
    "emission_focus": ["scope3_cat15_investments", "scope3_cat6_travel"],
    "scope1_emphasis": False,
    "scope2_emphasis": True,
    "scope3_emphasis": True,
    "key_agents": ["AGENT-MRV-028", "AGENT-MRV-019"],
    "industry_emission_factors": False,
}

TECHNOLOGY_SECTOR = {
    "sector_id": "technology",
    "display_name": "Technology",
    "nace_codes": ["J58-J63"],
    "emission_focus": ["scope2_electricity", "scope3_cat7_commuting"],
    "scope1_emphasis": False,
    "scope2_emphasis": True,
    "scope3_emphasis": True,
    "key_agents": ["AGENT-MRV-009", "AGENT-MRV-010", "AGENT-MRV-020"],
    "industry_emission_factors": False,
}

RETAIL_SECTOR = {
    "sector_id": "retail",
    "display_name": "Retail",
    "nace_codes": ["G45-G47"],
    "emission_focus": ["scope3_cat1_purchased_goods", "scope3_cat4_transport", "scope3_cat9_downstream"],
    "scope1_emphasis": False,
    "scope2_emphasis": True,
    "scope3_emphasis": True,
    "key_agents": ["AGENT-MRV-014", "AGENT-MRV-017", "AGENT-MRV-022"],
    "industry_emission_factors": False,
}

ENERGY_SECTOR = {
    "sector_id": "energy",
    "display_name": "Energy",
    "nace_codes": ["B05-B09", "D35"],
    "emission_focus": ["scope1_stationary", "scope1_process", "scope1_fugitive"],
    "scope1_emphasis": True,
    "scope2_emphasis": False,
    "scope3_emphasis": True,
    "key_agents": ["AGENT-MRV-001", "AGENT-MRV-004", "AGENT-MRV-005"],
    "industry_emission_factors": True,
}

ALL_SECTORS = {
    "manufacturing": MANUFACTURING_SECTOR,
    "financial_services": FINANCIAL_SERVICES_SECTOR,
    "technology": TECHNOLOGY_SECTOR,
    "retail": RETAIL_SECTOR,
    "energy": ENERGY_SECTOR,
}


@pytest.fixture(params=["manufacturing", "financial_services", "technology", "retail", "energy"])
def sector_config(request) -> Dict[str, Any]:
    """Parameterized fixture that yields each sector config in turn."""
    return ALL_SECTORS[request.param]


# ---------------------------------------------------------------------------
# Demo mode helpers
# ---------------------------------------------------------------------------

@pytest.fixture
def demo_config() -> Dict[str, Any]:
    """Demo mode configuration for rapid testing."""
    return {
        "mode": "demo",
        "company_profile": "demo_company_profile.json",
        "esg_data": "demo_esg_data.csv",
        "preset": "mid_market",
        "sector": "manufacturing",
        "skip_external_apis": True,
        "use_cached_emission_factors": True,
        "max_records": 500,
        "output_formats": ["markdown", "json"],
    }


# ---------------------------------------------------------------------------
# Template data helpers
# ---------------------------------------------------------------------------

@pytest.fixture
def template_render_data(
    sample_company_profile,
    sample_calculation_result,
    sample_materiality_result,
) -> Dict[str, Any]:
    """Consolidated data dict suitable for rendering any pack template."""
    return {
        "company": sample_company_profile,
        "calculations": sample_calculation_result,
        "materiality": sample_materiality_result,
        "reporting_year": 2025,
        "generated_at": "2025-07-15T10:30:00Z",
        "pack_version": "1.0.0",
    }


# ---------------------------------------------------------------------------
# ESRS standards reference
# ---------------------------------------------------------------------------

VALID_ESRS_STANDARDS = [
    "ESRS_1", "ESRS_2",
    "E1", "E2", "E3", "E4", "E5",
    "S1", "S2", "S3", "S4",
    "G1",
]

VALID_SCOPE3_CATEGORIES = list(range(1, 16))
