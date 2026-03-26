# -*- coding: utf-8 -*-
"""
PACK-041 Scope 1-2 Complete Pack - Shared Test Fixtures (conftest.py)
=====================================================================

Provides pytest fixtures for the PACK-041 test suite including:
  - Dynamic module loading via importlib (no package install needed)
  - Organization structure with entities and facilities
  - Boundary definitions for all three consolidation approaches
  - Facility-level fuel, electricity, refrigerant, and fleet data
  - Emission factors from DEFRA, IPCC, and EPA sources
  - Scope 1 per-category calculation results (8 categories)
  - Scope 2 location-based and market-based results
  - Contractual instruments (PPAs, RECs, GOs)
  - Consolidated inventory data
  - Base-year and multi-year trend data
  - GWP values for AR4, AR5, and AR6
  - Pack configuration from presets

Fixture Categories:
  1.  Paths and dynamic module loading
  2.  Organization and boundary fixtures
  3.  Facility emission data
  4.  Emission factor databases
  5.  Scope 1 category results
  6.  Scope 2 results and instruments
  7.  Consolidated inventory
  8.  Base year and trend data
  9.  GWP value tables
  10. Pack configuration and presets

Author:  GreenLang Platform Team (GL-TestEngineer)
Pack:    PACK-041 Scope 1-2 Complete
Date:    March 2026
"""

import hashlib
import importlib
import importlib.util
import json
import math
import random
import sys
from datetime import date, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest


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

# Engine file mapping: logical name -> file name on disk
ENGINE_FILES = {
    "organizational_boundary": "organizational_boundary_engine.py",
    "source_completeness": "source_completeness_engine.py",
    "emission_factor_manager": "emission_factor_manager_engine.py",
    "scope1_consolidation": "scope1_consolidation_engine.py",
    "scope2_consolidation": "scope2_consolidation_engine.py",
    "uncertainty_aggregation": "uncertainty_aggregation_engine.py",
    "base_year_recalculation": "base_year_recalculation_engine.py",
    "trend_analysis": "trend_analysis_engine.py",
    "compliance_mapping": "compliance_mapping_engine.py",
    "inventory_reporting": "inventory_reporting_engine.py",
}

ENGINE_CLASSES = {
    "organizational_boundary": "OrganizationalBoundaryEngine",
    "source_completeness": "SourceCompletenessEngine",
    "emission_factor_manager": "EmissionFactorManagerEngine",
    "scope1_consolidation": "Scope1ConsolidationEngine",
    "scope2_consolidation": "Scope2ConsolidationEngine",
    "uncertainty_aggregation": "UncertaintyAggregationEngine",
    "base_year_recalculation": "BaseYearRecalculationEngine",
    "trend_analysis": "TrendAnalysisEngine",
    "compliance_mapping": "ComplianceMappingEngine",
    "inventory_reporting": "InventoryReportingEngine",
}

WORKFLOW_FILES = {
    "boundary_definition": "boundary_definition_workflow.py",
    "data_collection": "data_collection_workflow.py",
    "scope1_calculation": "scope1_calculation_workflow.py",
    "scope2_calculation": "scope2_calculation_workflow.py",
    "inventory_consolidation": "inventory_consolidation_workflow.py",
    "verification_preparation": "verification_preparation_workflow.py",
    "disclosure_generation": "disclosure_generation_workflow.py",
    "full_inventory": "full_inventory_workflow.py",
}

WORKFLOW_CLASSES = {
    "boundary_definition": "BoundaryDefinitionWorkflow",
    "data_collection": "DataCollectionWorkflow",
    "scope1_calculation": "Scope1CalculationWorkflow",
    "scope2_calculation": "Scope2CalculationWorkflow",
    "inventory_consolidation": "InventoryConsolidationWorkflow",
    "verification_preparation": "VerificationPreparationWorkflow",
    "disclosure_generation": "DisclosureGenerationWorkflow",
    "full_inventory": "FullInventoryWorkflow",
}

WORKFLOW_PHASE_COUNTS = {
    "boundary_definition": 4,
    "data_collection": 4,
    "scope1_calculation": 4,
    "scope2_calculation": 4,
    "inventory_consolidation": 4,
    "verification_preparation": 3,
    "disclosure_generation": 3,
    "full_inventory": 8,
}

TEMPLATE_FILES = {
    "executive_summary": "executive_summary_report.py",
    "ghg_inventory_report": "ghg_inventory_report.py",
    "scope1_detail_report": "scope1_detail_report.py",
    "scope2_dual_report": "scope2_dual_report.py",
    "ef_registry_report": "ef_registry_report.py",
    "uncertainty_report": "uncertainty_report.py",
    "verification_package": "verification_package.py",
    "esrs_e1_disclosure": "esrs_e1_disclosure.py",
    "compliance_dashboard": "compliance_dashboard.py",
    "provenance_report": "provenance_report.py",
}

TEMPLATE_CLASSES = {
    "executive_summary": "ExecutiveSummaryTemplate",
    "ghg_inventory_report": "GHGInventoryReportTemplate",
    "scope1_detail_report": "Scope1DetailReportTemplate",
    "scope2_dual_report": "Scope2DualReportTemplate",
    "ef_registry_report": "EFRegistryReportTemplate",
    "uncertainty_report": "UncertaintyReportTemplate",
    "verification_package": "VerificationPackageTemplate",
    "esrs_e1_disclosure": "ESRSE1DisclosureTemplate",
    "compliance_dashboard": "ComplianceDashboardTemplate",
    "provenance_report": "ProvenanceReportTemplate",
}

INTEGRATION_FILES = {
    "pack_orchestrator": "pack_orchestrator.py",
    "mrv_scope1_bridge": "mrv_scope1_bridge.py",
    "mrv_scope2_bridge": "mrv_scope2_bridge.py",
    "data_bridge": "data_bridge.py",
    "foundation_bridge": "foundation_bridge.py",
    "erp_bridge": "erp_bridge.py",
    "pack028_bridge": "pack028_bridge.py",
    "pack029_bridge": "pack029_bridge.py",
    "pack030_bridge": "pack030_bridge.py",
    "health_check": "health_check.py",
    "setup_wizard": "setup_wizard.py",
    "alert_bridge": "alert_bridge.py",
}

INTEGRATION_CLASSES = {
    "pack_orchestrator": "Scope12Orchestrator",
    "mrv_scope1_bridge": "MRVScope1Bridge",
    "mrv_scope2_bridge": "MRVScope2Bridge",
    "data_bridge": "DataBridge",
    "foundation_bridge": "FoundationBridge",
    "erp_bridge": "ERPBridge",
    "pack028_bridge": "Pack028Bridge",
    "pack029_bridge": "Pack029Bridge",
    "pack030_bridge": "Pack030Bridge",
    "health_check": "HealthCheck",
    "setup_wizard": "SetupWizard",
    "alert_bridge": "AlertBridge",
}

PRESET_NAMES = [
    "corporate_office",
    "manufacturing_plant",
    "energy_utility",
    "transport_fleet",
    "agriculture_farm",
    "healthcare_hospital",
    "sme_simplified",
    "multi_site_portfolio",
]


# =============================================================================
# Helper: Dynamic Module Loader
# =============================================================================


def _load_module(module_name: str, file_name: str, subdir: str = "engines"):
    """Load a module dynamically using importlib.util.spec_from_file_location."""
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
            f"Ensure PACK-041 source files are present."
        )

    full_module_name = f"pack041_test.{subdir}.{module_name}"
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
        sys.modules.pop(full_module_name, None)
        raise ImportError(
            f"Failed to load module {full_module_name} from {file_path}: {exc}"
        ) from exc

    return module


def _load_engine(engine_key: str):
    """Load an engine module by its logical key."""
    file_name = ENGINE_FILES[engine_key]
    return _load_module(engine_key, file_name, "engines")


def _load_config_module():
    """Load the pack_config module."""
    return _load_module("pack_config", "pack_config.py", "config")


# =============================================================================
# Helper: Provenance hash utility
# =============================================================================


def compute_provenance_hash(data: Any) -> str:
    """Compute a SHA-256 provenance hash for any JSON-serializable data."""
    serialized = json.dumps(data, sort_keys=True, default=str)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


# =============================================================================
# Helper: Seeded RNG
# =============================================================================

_RNG = random.Random(42)


def _seeded_float(low: float, high: float) -> float:
    return _RNG.uniform(low, high)


def _seeded_int(low: int, high: int) -> int:
    return _RNG.randint(low, high)


# =============================================================================
# 1. Path and YAML Fixtures
# =============================================================================


@pytest.fixture(scope="session")
def pack_root() -> Path:
    """Return the absolute path to the PACK-041 root directory."""
    return PACK_ROOT


@pytest.fixture(scope="session")
def engines_dir() -> Path:
    """Return the absolute path to the engines directory."""
    return ENGINES_DIR


@pytest.fixture(scope="session")
def pack_yaml_path() -> Path:
    """Return the absolute path to pack.yaml."""
    return PACK_ROOT / "pack.yaml"


@pytest.fixture(scope="session")
def pack_yaml_data(pack_yaml_path: Path) -> Dict[str, Any]:
    """Parse and return the pack.yaml manifest as a dictionary."""
    if not pack_yaml_path.exists():
        pytest.skip("pack.yaml not found")
    with open(pack_yaml_path, "r", encoding="utf-8") as f:
        data = __import__("yaml").safe_load(f)
    assert data is not None, "pack.yaml parsed to None"
    return data


# =============================================================================
# 2. Organization and Boundary Fixtures
# =============================================================================


@pytest.fixture
def sample_organization() -> Dict[str, Any]:
    """Organization with 3 entities and 10 facilities for boundary testing."""
    return {
        "org_id": "ORG-ACME-001",
        "org_name": "Acme Global Industries",
        "reporting_year": 2025,
        "base_year": 2019,
        "default_approach": "operational_control",
        "significance_threshold_pct": Decimal("5.0"),
        "entities": [
            {
                "entity_id": "ENT-001",
                "entity_name": "Acme Manufacturing US",
                "entity_type": "wholly_owned",
                "equity_pct": Decimal("100"),
                "has_operational_control": True,
                "has_financial_control": True,
                "country_of_incorporation": "US",
                "sector": "manufacturing",
                "total_scope1_tco2e": Decimal("12500.0"),
                "total_scope2_tco2e": Decimal("8200.0"),
                "is_active": True,
                "facilities": [
                    {
                        "facility_id": "FAC-001",
                        "facility_name": "Houston Plant",
                        "entity_id": "ENT-001",
                        "country": "US",
                        "region": "TX",
                        "sector": "manufacturing",
                        "operational_status": "active",
                        "scope1_emissions_tco2e": Decimal("5000"),
                        "scope2_emissions_tco2e": Decimal("3200"),
                        "employee_count": 450,
                        "floor_area_m2": Decimal("25000"),
                    },
                    {
                        "facility_id": "FAC-002",
                        "facility_name": "Chicago Office",
                        "entity_id": "ENT-001",
                        "country": "US",
                        "region": "IL",
                        "sector": "office",
                        "operational_status": "active",
                        "scope1_emissions_tco2e": Decimal("500"),
                        "scope2_emissions_tco2e": Decimal("2000"),
                        "employee_count": 200,
                        "floor_area_m2": Decimal("5000"),
                    },
                    {
                        "facility_id": "FAC-003",
                        "facility_name": "Detroit Assembly",
                        "entity_id": "ENT-001",
                        "country": "US",
                        "region": "MI",
                        "sector": "manufacturing",
                        "operational_status": "active",
                        "scope1_emissions_tco2e": Decimal("7000"),
                        "scope2_emissions_tco2e": Decimal("3000"),
                        "employee_count": 600,
                        "floor_area_m2": Decimal("45000"),
                    },
                ],
            },
            {
                "entity_id": "ENT-002",
                "entity_name": "Acme Europe GmbH",
                "entity_type": "wholly_owned",
                "equity_pct": Decimal("100"),
                "has_operational_control": True,
                "has_financial_control": True,
                "country_of_incorporation": "DE",
                "sector": "manufacturing",
                "total_scope1_tco2e": Decimal("9800.0"),
                "total_scope2_tco2e": Decimal("6500.0"),
                "is_active": True,
                "facilities": [
                    {
                        "facility_id": "FAC-004",
                        "facility_name": "Frankfurt Plant",
                        "entity_id": "ENT-002",
                        "country": "DE",
                        "region": "HE",
                        "sector": "manufacturing",
                        "operational_status": "active",
                        "scope1_emissions_tco2e": Decimal("6000"),
                        "scope2_emissions_tco2e": Decimal("4000"),
                        "employee_count": 350,
                        "floor_area_m2": Decimal("30000"),
                    },
                    {
                        "facility_id": "FAC-005",
                        "facility_name": "Berlin Office",
                        "entity_id": "ENT-002",
                        "country": "DE",
                        "region": "BE",
                        "sector": "office",
                        "operational_status": "active",
                        "scope1_emissions_tco2e": Decimal("300"),
                        "scope2_emissions_tco2e": Decimal("1500"),
                        "employee_count": 120,
                        "floor_area_m2": Decimal("3000"),
                    },
                    {
                        "facility_id": "FAC-006",
                        "facility_name": "Munich R&D Lab",
                        "entity_id": "ENT-002",
                        "country": "DE",
                        "region": "BY",
                        "sector": "laboratory",
                        "operational_status": "active",
                        "scope1_emissions_tco2e": Decimal("3500"),
                        "scope2_emissions_tco2e": Decimal("1000"),
                        "employee_count": 80,
                        "floor_area_m2": Decimal("8000"),
                    },
                ],
            },
            {
                "entity_id": "ENT-003",
                "entity_name": "Acme-Nippon JV",
                "entity_type": "joint_venture",
                "equity_pct": Decimal("50"),
                "has_operational_control": False,
                "has_financial_control": False,
                "country_of_incorporation": "JP",
                "sector": "manufacturing",
                "total_scope1_tco2e": Decimal("7200.0"),
                "total_scope2_tco2e": Decimal("5100.0"),
                "is_active": True,
                "facilities": [
                    {
                        "facility_id": "FAC-007",
                        "facility_name": "Osaka Plant",
                        "entity_id": "ENT-003",
                        "country": "JP",
                        "region": "Osaka",
                        "sector": "manufacturing",
                        "operational_status": "active",
                        "scope1_emissions_tco2e": Decimal("4500"),
                        "scope2_emissions_tco2e": Decimal("3000"),
                        "employee_count": 280,
                        "floor_area_m2": Decimal("20000"),
                    },
                    {
                        "facility_id": "FAC-008",
                        "facility_name": "Tokyo Office",
                        "entity_id": "ENT-003",
                        "country": "JP",
                        "region": "Tokyo",
                        "sector": "office",
                        "operational_status": "active",
                        "scope1_emissions_tco2e": Decimal("200"),
                        "scope2_emissions_tco2e": Decimal("1100"),
                        "employee_count": 60,
                        "floor_area_m2": Decimal("2000"),
                    },
                    {
                        "facility_id": "FAC-009",
                        "facility_name": "Nagoya Warehouse",
                        "entity_id": "ENT-003",
                        "country": "JP",
                        "region": "Aichi",
                        "sector": "logistics",
                        "operational_status": "active",
                        "scope1_emissions_tco2e": Decimal("2000"),
                        "scope2_emissions_tco2e": Decimal("800"),
                        "employee_count": 40,
                        "floor_area_m2": Decimal("15000"),
                    },
                    {
                        "facility_id": "FAC-010",
                        "facility_name": "Kyoto Pilot",
                        "entity_id": "ENT-003",
                        "country": "JP",
                        "region": "Kyoto",
                        "sector": "laboratory",
                        "operational_status": "idle",
                        "scope1_emissions_tco2e": Decimal("500"),
                        "scope2_emissions_tco2e": Decimal("200"),
                        "employee_count": 10,
                        "floor_area_m2": Decimal("1500"),
                    },
                ],
            },
        ],
    }


@pytest.fixture
def sample_boundary() -> Dict[str, Any]:
    """BoundaryDefinition result with operational control approach."""
    return {
        "boundary_id": "BND-2025-001",
        "org_id": "ORG-ACME-001",
        "org_name": "Acme Global Industries",
        "reporting_year": 2025,
        "base_year": 2019,
        "approach": "operational_control",
        "total_entities": 3,
        "included_entities": 2,
        "excluded_entities": 1,
        "partial_entities": 0,
        "total_scope1_tco2e": Decimal("22300.0"),
        "total_scope2_tco2e": Decimal("14700.0"),
        "total_emissions_tco2e": Decimal("37000.0"),
        "total_facilities": 6,
        "countries_covered": ["DE", "US"],
        "sectors_covered": ["laboratory", "manufacturing", "office"],
    }


# =============================================================================
# 3. Facility Emission Data
# =============================================================================


@pytest.fixture
def sample_facility_data() -> Dict[str, Any]:
    """Comprehensive facility data with fuel, electricity, refrigerant, fleet."""
    return {
        "facility_id": "FAC-001",
        "facility_name": "Houston Plant",
        "reporting_year": 2025,
        "fuel_data": [
            {
                "fuel_type": "natural_gas",
                "quantity": Decimal("1000000"),
                "unit": "m3",
                "activity": "stationary_combustion",
                "source": "Boiler B-101",
            },
            {
                "fuel_type": "diesel",
                "quantity": Decimal("50000"),
                "unit": "litres",
                "activity": "stationary_combustion",
                "source": "Backup Generator G-001",
            },
            {
                "fuel_type": "lpg",
                "quantity": Decimal("15000"),
                "unit": "litres",
                "activity": "stationary_combustion",
                "source": "Process Heater PH-004",
            },
        ],
        "electricity_data": [
            {
                "quantity_mwh": Decimal("10000"),
                "grid_region": "ERCOT",
                "country": "US",
                "supplier": "TXU Energy",
                "tariff": "large_industrial",
            },
        ],
        "refrigerant_data": [
            {
                "refrigerant_type": "R-410A",
                "quantity_kg": Decimal("100"),
                "leak_rate_pct": Decimal("8"),
                "equipment": "HVAC Chiller CH-001",
            },
            {
                "refrigerant_type": "R-134a",
                "quantity_kg": Decimal("25"),
                "leak_rate_pct": Decimal("5"),
                "equipment": "Cold Store CS-001",
            },
        ],
        "fleet_data": [
            {
                "vehicle_type": "heavy_goods_vehicle",
                "fuel_type": "diesel",
                "distance_km": Decimal("500000"),
                "fuel_consumed_litres": Decimal("125000"),
                "vehicle_count": 10,
            },
            {
                "vehicle_type": "passenger_car",
                "fuel_type": "petrol",
                "distance_km": Decimal("200000"),
                "fuel_consumed_litres": Decimal("20000"),
                "vehicle_count": 15,
            },
        ],
        "process_data": [
            {
                "process_type": "cement_clinker",
                "quantity_tonnes": Decimal("5000"),
                "emission_factor_tco2_per_t": Decimal("0.525"),
            },
        ],
        "waste_data": [
            {
                "waste_type": "on_site_incineration",
                "quantity_tonnes": Decimal("200"),
                "co2_factor_tco2_per_t": Decimal("0.91"),
            },
        ],
    }


# =============================================================================
# 4. Emission Factor Databases
# =============================================================================


@pytest.fixture
def sample_emission_factors() -> Dict[str, Any]:
    """Emission factors from DEFRA, IPCC, and EPA for key fuels and grids."""
    return {
        "fuels": {
            "natural_gas": {
                "ipcc_2006": {
                    "co2_kg_per_gj": Decimal("56.1"),
                    "ch4_kg_per_gj": Decimal("0.001"),
                    "n2o_kg_per_gj": Decimal("0.0001"),
                    "net_cv_gj_per_m3": Decimal("0.0364"),
                },
                "defra_2025": {
                    "co2_kg_per_m3": Decimal("2.02"),
                    "ch4_kg_per_m3": Decimal("0.000037"),
                    "n2o_kg_per_m3": Decimal("0.0000036"),
                    "co2e_kg_per_m3": Decimal("2.0216"),
                },
            },
            "diesel": {
                "ipcc_2006": {
                    "co2_kg_per_gj": Decimal("74.1"),
                    "ch4_kg_per_gj": Decimal("0.003"),
                    "n2o_kg_per_gj": Decimal("0.0006"),
                    "net_cv_gj_per_litre": Decimal("0.0360"),
                },
                "defra_2025": {
                    "co2_kg_per_litre": Decimal("2.5121"),
                    "ch4_kg_per_litre": Decimal("0.000108"),
                    "n2o_kg_per_litre": Decimal("0.0000216"),
                    "co2e_kg_per_litre": Decimal("2.5271"),
                },
            },
            "petrol": {
                "defra_2025": {
                    "co2_kg_per_litre": Decimal("2.1610"),
                    "co2e_kg_per_litre": Decimal("2.1944"),
                },
            },
            "lpg": {
                "defra_2025": {
                    "co2_kg_per_litre": Decimal("1.5148"),
                    "co2e_kg_per_litre": Decimal("1.5226"),
                },
            },
            "fuel_oil": {
                "ipcc_2006": {
                    "co2_kg_per_gj": Decimal("77.4"),
                },
            },
            "coal": {
                "ipcc_2006": {
                    "co2_kg_per_gj": Decimal("94.6"),
                },
            },
        },
        "grids": {
            "DE": {
                "location_based_kg_per_kwh": Decimal("0.385"),
                "source": "IEA 2024",
                "year": 2024,
            },
            "US_average": {
                "location_based_kg_per_kwh": Decimal("0.390"),
                "source": "EPA eGRID 2024",
                "year": 2024,
            },
            "US_ERCOT": {
                "location_based_kg_per_kwh": Decimal("0.370"),
                "source": "EPA eGRID 2024",
                "year": 2024,
            },
            "GB": {
                "location_based_kg_per_kwh": Decimal("0.207"),
                "source": "DEFRA 2025",
                "year": 2025,
            },
            "JP": {
                "location_based_kg_per_kwh": Decimal("0.470"),
                "source": "IEA 2024",
                "year": 2024,
            },
            "FR": {
                "location_based_kg_per_kwh": Decimal("0.052"),
                "source": "IEA 2024",
                "year": 2024,
            },
        },
        "refrigerants": {
            "R-410A": {
                "gwp_ar4": Decimal("2088"),
                "gwp_ar5": Decimal("2088"),
                "gwp_ar6": Decimal("2088"),
                "type": "HFC blend",
            },
            "R-134a": {
                "gwp_ar4": Decimal("1430"),
                "gwp_ar5": Decimal("1300"),
                "gwp_ar6": Decimal("1530"),
                "type": "HFC",
            },
            "R-404A": {
                "gwp_ar4": Decimal("3922"),
                "gwp_ar5": Decimal("3922"),
                "gwp_ar6": Decimal("4728"),
                "type": "HFC blend",
            },
            "R-32": {
                "gwp_ar4": Decimal("675"),
                "gwp_ar5": Decimal("677"),
                "gwp_ar6": Decimal("771"),
                "type": "HFC",
            },
            "SF6": {
                "gwp_ar4": Decimal("22800"),
                "gwp_ar5": Decimal("23500"),
                "gwp_ar6": Decimal("25200"),
                "type": "Fully fluorinated",
            },
        },
    }


# =============================================================================
# 5. Scope 1 Category Results
# =============================================================================


@pytest.fixture
def sample_scope1_results() -> Dict[str, Any]:
    """Per-category Scope 1 results from all 8 MRV agents."""
    return {
        "reporting_year": 2025,
        "facility_id": "FAC-001",
        "categories": {
            "stationary_combustion": {
                "agent": "MRV-001",
                "co2_tco2e": Decimal("2345.6"),
                "ch4_tco2e": Decimal("12.3"),
                "n2o_tco2e": Decimal("5.7"),
                "total_tco2e": Decimal("2363.6"),
                "uncertainty_pct": Decimal("7.0"),
                "source_count": 3,
            },
            "mobile_combustion": {
                "agent": "MRV-002",
                "co2_tco2e": Decimal("380.4"),
                "ch4_tco2e": Decimal("1.2"),
                "n2o_tco2e": Decimal("8.5"),
                "total_tco2e": Decimal("390.1"),
                "uncertainty_pct": Decimal("11.0"),
                "source_count": 2,
            },
            "process_emissions": {
                "agent": "MRV-003",
                "co2_tco2e": Decimal("2625.0"),
                "ch4_tco2e": Decimal("0"),
                "n2o_tco2e": Decimal("0"),
                "total_tco2e": Decimal("2625.0"),
                "uncertainty_pct": Decimal("16.0"),
                "source_count": 1,
            },
            "fugitive_emissions": {
                "agent": "MRV-004",
                "co2_tco2e": Decimal("0"),
                "ch4_tco2e": Decimal("45.0"),
                "n2o_tco2e": Decimal("0"),
                "total_tco2e": Decimal("45.0"),
                "uncertainty_pct": Decimal("32.0"),
                "source_count": 5,
            },
            "refrigerant_fgas": {
                "agent": "MRV-005",
                "co2_tco2e": Decimal("0"),
                "hfcs_tco2e": Decimal("208.8"),
                "total_tco2e": Decimal("208.8"),
                "uncertainty_pct": Decimal("22.0"),
                "source_count": 2,
            },
            "land_use": {
                "agent": "MRV-006",
                "co2_tco2e": Decimal("0"),
                "total_tco2e": Decimal("0"),
                "uncertainty_pct": Decimal("0"),
                "source_count": 0,
            },
            "waste_treatment": {
                "agent": "MRV-007",
                "co2_tco2e": Decimal("182.0"),
                "ch4_tco2e": Decimal("12.0"),
                "n2o_tco2e": Decimal("3.5"),
                "total_tco2e": Decimal("197.5"),
                "uncertainty_pct": Decimal("25.0"),
                "source_count": 1,
            },
            "agricultural": {
                "agent": "MRV-008",
                "co2_tco2e": Decimal("0"),
                "total_tco2e": Decimal("0"),
                "uncertainty_pct": Decimal("0"),
                "source_count": 0,
            },
        },
        "total_scope1_tco2e": Decimal("5830.0"),
    }


# =============================================================================
# 6. Scope 2 Results and Instruments
# =============================================================================


@pytest.fixture
def sample_scope2_results() -> Dict[str, Any]:
    """Scope 2 location-based and market-based results."""
    return {
        "reporting_year": 2025,
        "facility_id": "FAC-001",
        "location_based": {
            "electricity_tco2e": Decimal("3700.0"),
            "steam_tco2e": Decimal("0"),
            "cooling_tco2e": Decimal("0"),
            "total_tco2e": Decimal("3700.0"),
            "grid_factor_kg_per_kwh": Decimal("0.370"),
            "electricity_mwh": Decimal("10000"),
            "uncertainty_pct": Decimal("10.0"),
        },
        "market_based": {
            "electricity_tco2e": Decimal("1850.0"),
            "steam_tco2e": Decimal("0"),
            "cooling_tco2e": Decimal("0"),
            "total_tco2e": Decimal("1850.0"),
            "instruments_applied": ["PPA-001", "REC-001"],
            "rec_coverage_mwh": Decimal("5000"),
            "residual_mix_mwh": Decimal("5000"),
            "residual_factor_kg_per_kwh": Decimal("0.370"),
            "uncertainty_pct": Decimal("5.0"),
        },
        "variance_tco2e": Decimal("1850.0"),
        "variance_pct": Decimal("50.0"),
    }


@pytest.fixture
def sample_instruments() -> List[Dict[str, Any]]:
    """Contractual instruments: PPAs, RECs, Guarantees of Origin."""
    return [
        {
            "instrument_id": "PPA-001",
            "type": "power_purchase_agreement",
            "supplier": "SunWind Energy LLC",
            "technology": "solar",
            "capacity_mw": Decimal("5.0"),
            "annual_generation_mwh": Decimal("8000"),
            "allocated_mwh": Decimal("3000"),
            "emission_factor_kg_per_kwh": Decimal("0"),
            "country": "US",
            "grid_region": "ERCOT",
            "start_date": "2024-01-01",
            "end_date": "2039-12-31",
            "additionality": True,
            "tracking_system": "M-RETS",
            "certificate_ids": ["MRETS-2025-001234"],
        },
        {
            "instrument_id": "REC-001",
            "type": "renewable_energy_certificate",
            "supplier": "Green Certificate Corp",
            "technology": "wind",
            "quantity_mwh": Decimal("2000"),
            "vintage_year": 2025,
            "emission_factor_kg_per_kwh": Decimal("0"),
            "country": "US",
            "grid_region": "ERCOT",
            "tracking_system": "M-RETS",
            "certificate_ids": ["MRETS-2025-005678"],
        },
        {
            "instrument_id": "GO-001",
            "type": "guarantee_of_origin",
            "supplier": "WindEurope AG",
            "technology": "wind_offshore",
            "quantity_mwh": Decimal("4000"),
            "vintage_year": 2025,
            "emission_factor_kg_per_kwh": Decimal("0"),
            "country": "DE",
            "tracking_system": "AIB_EECS",
            "certificate_ids": ["AIB-2025-DE-000123"],
        },
    ]


# =============================================================================
# 7. Consolidated Inventory
# =============================================================================


@pytest.fixture
def sample_inventory() -> Dict[str, Any]:
    """Complete consolidated GHG inventory."""
    return {
        "inventory_id": "INV-2025-001",
        "org_id": "ORG-ACME-001",
        "org_name": "Acme Global Industries",
        "reporting_year": 2025,
        "base_year": 2019,
        "approach": "operational_control",
        "scope1": {
            "total_tco2e": Decimal("22300.0"),
            "by_category": {
                "stationary_combustion": Decimal("12800.0"),
                "mobile_combustion": Decimal("2500.0"),
                "process_emissions": Decimal("4200.0"),
                "fugitive_emissions": Decimal("350.0"),
                "refrigerant_fgas": Decimal("1200.0"),
                "land_use": Decimal("0"),
                "waste_treatment": Decimal("1250.0"),
                "agricultural": Decimal("0"),
            },
            "by_gas": {
                "CO2": Decimal("19800.0"),
                "CH4": Decimal("850.0"),
                "N2O": Decimal("450.0"),
                "HFCs": Decimal("1200.0"),
                "PFCs": Decimal("0"),
                "SF6": Decimal("0"),
                "NF3": Decimal("0"),
            },
            "by_facility": {
                "FAC-001": Decimal("5830.0"),
                "FAC-002": Decimal("500.0"),
                "FAC-003": Decimal("7000.0"),
                "FAC-004": Decimal("6000.0"),
                "FAC-005": Decimal("300.0"),
                "FAC-006": Decimal("2670.0"),
            },
        },
        "scope2_location": {
            "total_tco2e": Decimal("14700.0"),
        },
        "scope2_market": {
            "total_tco2e": Decimal("9200.0"),
        },
        "total_scope12_location": Decimal("37000.0"),
        "total_scope12_market": Decimal("31500.0"),
        "uncertainty_pct": Decimal("8.5"),
        "data_quality_score": Decimal("82.5"),
        "completeness_pct": Decimal("96.0"),
    }


# =============================================================================
# 8. Base Year and Trend Data
# =============================================================================


@pytest.fixture
def sample_base_year() -> Dict[str, Any]:
    """BaseYearData for 2019 with full breakdown."""
    return {
        "base_year": 2019,
        "total_scope1_tco2e": Decimal("25000.0"),
        "total_scope2_location_tco2e": Decimal("16000.0"),
        "total_scope2_market_tco2e": Decimal("15000.0"),
        "total_scope12_location_tco2e": Decimal("41000.0"),
        "total_scope12_market_tco2e": Decimal("40000.0"),
        "revenue_million_usd": Decimal("850.0"),
        "employee_count": 2200,
        "floor_area_m2": Decimal("130000"),
        "scope1_intensity_per_revenue": Decimal("29.41"),
        "scope12_intensity_per_fte": Decimal("18.64"),
    }


@pytest.fixture
def sample_yearly_data() -> List[Dict[str, Any]]:
    """Three years of emissions data for trend analysis."""
    return [
        {
            "year": 2023,
            "total_scope1_tco2e": Decimal("24000.0"),
            "total_scope2_location_tco2e": Decimal("15500.0"),
            "total_scope2_market_tco2e": Decimal("12000.0"),
            "revenue_million_usd": Decimal("900.0"),
            "employee_count": 2350,
            "floor_area_m2": Decimal("132000"),
        },
        {
            "year": 2024,
            "total_scope1_tco2e": Decimal("23100.0"),
            "total_scope2_location_tco2e": Decimal("15000.0"),
            "total_scope2_market_tco2e": Decimal("10500.0"),
            "revenue_million_usd": Decimal("920.0"),
            "employee_count": 2400,
            "floor_area_m2": Decimal("135000"),
        },
        {
            "year": 2025,
            "total_scope1_tco2e": Decimal("22300.0"),
            "total_scope2_location_tco2e": Decimal("14700.0"),
            "total_scope2_market_tco2e": Decimal("9200.0"),
            "revenue_million_usd": Decimal("950.0"),
            "employee_count": 2500,
            "floor_area_m2": Decimal("138000"),
        },
    ]


# =============================================================================
# 9. GWP Value Tables
# =============================================================================


@pytest.fixture
def sample_gwp_values() -> Dict[str, Dict[str, Decimal]]:
    """GWP 100-year values for AR4, AR5, and AR6."""
    return {
        "CO2": {"ar4": Decimal("1"), "ar5": Decimal("1"), "ar6": Decimal("1")},
        "CH4": {"ar4": Decimal("25"), "ar5": Decimal("28"), "ar6": Decimal("27.9")},
        "N2O": {"ar4": Decimal("298"), "ar5": Decimal("265"), "ar6": Decimal("273")},
        "SF6": {"ar4": Decimal("22800"), "ar5": Decimal("23500"), "ar6": Decimal("25200")},
        "NF3": {"ar4": Decimal("17200"), "ar5": Decimal("16100"), "ar6": Decimal("17400")},
        "HFC-134a": {"ar4": Decimal("1430"), "ar5": Decimal("1300"), "ar6": Decimal("1530")},
        "HFC-32": {"ar4": Decimal("675"), "ar5": Decimal("677"), "ar6": Decimal("771")},
        "R-410A": {"ar4": Decimal("2088"), "ar5": Decimal("2088"), "ar6": Decimal("2088")},
        "CF4": {"ar4": Decimal("7390"), "ar5": Decimal("6630"), "ar6": Decimal("7380")},
        "C2F6": {"ar4": Decimal("12200"), "ar5": Decimal("11100"), "ar6": Decimal("12400")},
    }


# =============================================================================
# 10. Pack Configuration and Presets
# =============================================================================


@pytest.fixture
def sample_pack_config() -> Dict[str, Any]:
    """PackConfig from corporate_office preset."""
    return {
        "pack_id": "PACK-041",
        "pack_name": "Scope 1-2 Complete Pack",
        "version": "1.0.0",
        "category": "ghg-accounting",
        "environment": "test",
        "default_region": "US",
        "decimal_precision": 4,
        "provenance_enabled": True,
        "multi_tenant_enabled": True,
        "boundary": {
            "default_approach": "operational_control",
            "significance_threshold_pct": Decimal("5.0"),
        },
        "scope1": {
            "enabled_categories": [
                "stationary_combustion",
                "mobile_combustion",
                "process_emissions",
                "fugitive_emissions",
                "refrigerant_fgas",
                "land_use",
                "waste_treatment",
                "agricultural",
            ],
            "gwp_source": "ar6",
            "default_ef_source": "defra_2025",
        },
        "scope2": {
            "dual_reporting": True,
            "default_grid_source": "iea_2024",
            "instrument_hierarchy": [
                "energy_attribute_certificate",
                "power_purchase_agreement",
                "green_tariff",
                "residual_mix",
            ],
        },
        "uncertainty": {
            "method": "analytical",
            "monte_carlo_iterations": 10000,
            "confidence_level": Decimal("0.95"),
            "seed": 42,
        },
        "reporting": {
            "output_formats": ["markdown", "html", "json"],
            "frameworks": [
                "ghg_protocol",
                "iso_14064",
                "esrs_e1",
                "cdp",
                "sbti",
                "sec",
                "sb_253",
            ],
        },
    }


# =============================================================================
# Composite / Convenience Fixtures
# =============================================================================


@pytest.fixture
def full_inventory_context(
    sample_organization,
    sample_boundary,
    sample_facility_data,
    sample_emission_factors,
    sample_scope1_results,
    sample_scope2_results,
    sample_instruments,
    sample_inventory,
    sample_base_year,
    sample_yearly_data,
    sample_gwp_values,
    sample_pack_config,
) -> Dict[str, Any]:
    """Aggregate fixture combining all data for integration/e2e tests."""
    return {
        "organization": sample_organization,
        "boundary": sample_boundary,
        "facility_data": sample_facility_data,
        "emission_factors": sample_emission_factors,
        "scope1_results": sample_scope1_results,
        "scope2_results": sample_scope2_results,
        "instruments": sample_instruments,
        "inventory": sample_inventory,
        "base_year": sample_base_year,
        "yearly_data": sample_yearly_data,
        "gwp_values": sample_gwp_values,
        "pack_config": sample_pack_config,
    }
