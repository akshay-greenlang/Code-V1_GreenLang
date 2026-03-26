# -*- coding: utf-8 -*-
"""
PACK-042 Scope 3 Starter Pack - Shared Test Fixtures (conftest.py)
===================================================================

Provides pytest fixtures for the PACK-042 test suite including:
  - Dynamic module loading via importlib (no package install needed)
  - Organization profiles for 5 sector archetypes
  - Procurement/spend data fixtures (100+ transactions)
  - Category results fixtures (all 15 categories)
  - Supplier data fixtures (20+ suppliers)
  - EEIO emission factor fixtures
  - Screening results fixtures
  - Consolidated Scope 3 inventory fixture
  - Double-counting detection fixtures
  - Hotspot analysis fixtures
  - Data quality assessment fixtures
  - Uncertainty analysis fixtures
  - Compliance assessment fixtures
  - Pack config fixtures (default, manufacturing, SME presets)
  - Empty/minimal input fixtures for edge case testing

Fixture Categories:
  1.  Paths and dynamic module loading
  2.  Organization profile fixtures
  3.  Procurement/spend data fixtures
  4.  Category result fixtures
  5.  Supplier data fixtures
  6.  EEIO emission factor fixtures
  7.  Screening result fixtures
  8.  Consolidated inventory fixture
  9.  Double-counting detection fixtures
  10. Hotspot analysis fixtures
  11. Data quality assessment fixtures
  12. Uncertainty analysis fixtures
  13. Compliance assessment fixtures
  14. Pack configuration fixtures
  15. Edge case fixtures

Author:  GreenLang Platform Team (GL-BackendDeveloper)
Pack:    PACK-042 Scope 3 Starter
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
    "scope3_screening": "scope3_screening_engine.py",
    "spend_classification": "spend_classification_engine.py",
    "category_consolidation": "category_consolidation_engine.py",
    "double_counting": "double_counting_prevention_engine.py",
    "hotspot_analysis": "hotspot_analysis_engine.py",
    "supplier_engagement": "supplier_engagement_engine.py",
    "data_quality": "data_quality_assessment_engine.py",
    "scope3_uncertainty": "scope3_uncertainty_engine.py",
    "scope3_compliance": "scope3_compliance_engine.py",
    "scope3_reporting": "scope3_reporting_engine.py",
}

ENGINE_CLASSES = {
    "scope3_screening": "Scope3ScreeningEngine",
    "spend_classification": "SpendClassificationEngine",
    "category_consolidation": "CategoryConsolidationEngine",
    "double_counting": "DoubleCountingPreventionEngine",
    "hotspot_analysis": "HotspotAnalysisEngine",
    "supplier_engagement": "SupplierEngagementEngine",
    "data_quality": "DataQualityAssessmentEngine",
    "scope3_uncertainty": "Scope3UncertaintyEngine",
    "scope3_compliance": "Scope3ComplianceEngine",
    "scope3_reporting": "Scope3ReportingEngine",
}

WORKFLOW_FILES = {
    "scope3_screening": "scope3_screening_workflow.py",
    "spend_mapping": "spend_mapping_workflow.py",
    "category_calculation": "category_calculation_workflow.py",
    "supplier_engagement": "supplier_engagement_workflow.py",
    "hotspot_prioritization": "hotspot_prioritization_workflow.py",
    "compliance_assessment": "compliance_assessment_workflow.py",
    "report_generation": "report_generation_workflow.py",
    "full_scope3_pipeline": "full_scope3_pipeline_workflow.py",
}

WORKFLOW_CLASSES = {
    "scope3_screening": "Scope3ScreeningWorkflow",
    "spend_mapping": "SpendMappingWorkflow",
    "category_calculation": "CategoryCalculationWorkflow",
    "supplier_engagement": "SupplierEngagementWorkflow",
    "hotspot_prioritization": "HotspotPrioritizationWorkflow",
    "compliance_assessment": "ComplianceAssessmentWorkflow",
    "report_generation": "ReportGenerationWorkflow",
    "full_scope3_pipeline": "FullScope3PipelineWorkflow",
}

WORKFLOW_PHASE_COUNTS = {
    "scope3_screening": 4,
    "spend_mapping": 4,
    "category_calculation": 5,
    "supplier_engagement": 4,
    "hotspot_prioritization": 4,
    "compliance_assessment": 5,
    "report_generation": 4,
    "full_scope3_pipeline": 8,
}

TEMPLATE_FILES = {
    "scope3_inventory_report": "scope3_inventory_report.py",
    "category_deep_dive_report": "category_deep_dive_report.py",
    "scope3_executive_summary": "scope3_executive_summary.py",
    "hotspot_report": "hotspot_report.py",
    "supplier_engagement_report": "supplier_engagement_report.py",
    "data_quality_report": "data_quality_report.py",
    "scope3_compliance_dashboard": "scope3_compliance_dashboard.py",
    "scope3_uncertainty_report": "scope3_uncertainty_report.py",
    "scope3_verification_package": "scope3_verification_package.py",
    "esrs_e1_scope3_disclosure": "esrs_e1_scope3_disclosure.py",
}

TEMPLATE_CLASSES = {
    "scope3_inventory_report": "Scope3InventoryReportTemplate",
    "category_deep_dive_report": "CategoryDeepDiveReportTemplate",
    "scope3_executive_summary": "Scope3ExecutiveSummaryTemplate",
    "hotspot_report": "HotspotReportTemplate",
    "supplier_engagement_report": "SupplierEngagementReportTemplate",
    "data_quality_report": "DataQualityReportTemplate",
    "scope3_compliance_dashboard": "Scope3ComplianceDashboardTemplate",
    "scope3_uncertainty_report": "Scope3UncertaintyReportTemplate",
    "scope3_verification_package": "Scope3VerificationPackageTemplate",
    "esrs_e1_scope3_disclosure": "ESRSE1Scope3DisclosureTemplate",
}

INTEGRATION_FILES = {
    "pack_orchestrator": "pack_orchestrator.py",
    "mrv_scope3_bridge": "mrv_scope3_bridge.py",
    "category_mapper_bridge": "category_mapper_bridge.py",
    "audit_trail_bridge": "audit_trail_bridge.py",
    "data_bridge": "data_bridge.py",
    "foundation_bridge": "foundation_bridge.py",
    "scope12_bridge": "scope12_bridge.py",
    "eeio_factor_bridge": "eeio_factor_bridge.py",
    "erp_connector": "erp_connector.py",
    "health_check": "health_check.py",
    "setup_wizard": "setup_wizard.py",
    "alert_bridge": "alert_bridge.py",
}

INTEGRATION_CLASSES = {
    "pack_orchestrator": "Scope3Orchestrator",
    "mrv_scope3_bridge": "MRVScope3Bridge",
    "category_mapper_bridge": "CategoryMapperBridge",
    "audit_trail_bridge": "AuditTrailBridge",
    "data_bridge": "DataBridge",
    "foundation_bridge": "FoundationBridge",
    "scope12_bridge": "Scope12Bridge",
    "eeio_factor_bridge": "EEIOFactorBridge",
    "erp_connector": "ERPConnector",
    "health_check": "HealthCheck",
    "setup_wizard": "SetupWizard",
    "alert_bridge": "AlertBridge",
}

PRESET_NAMES = [
    "manufacturing",
    "retail",
    "technology",
    "financial_services",
    "food_agriculture",
    "energy_utility",
    "transport_logistics",
    "sme_simplified",
]

# All 15 Scope 3 categories
SCOPE3_CATEGORIES = [
    "CAT_1", "CAT_2", "CAT_3", "CAT_4", "CAT_5",
    "CAT_6", "CAT_7", "CAT_8", "CAT_9", "CAT_10",
    "CAT_11", "CAT_12", "CAT_13", "CAT_14", "CAT_15",
]

SCOPE3_CATEGORY_NAMES = {
    "CAT_1": "Purchased Goods & Services",
    "CAT_2": "Capital Goods",
    "CAT_3": "Fuel- & Energy-Related Activities",
    "CAT_4": "Upstream Transportation & Distribution",
    "CAT_5": "Waste Generated in Operations",
    "CAT_6": "Business Travel",
    "CAT_7": "Employee Commuting",
    "CAT_8": "Upstream Leased Assets",
    "CAT_9": "Downstream Transportation & Distribution",
    "CAT_10": "Processing of Sold Products",
    "CAT_11": "Use of Sold Products",
    "CAT_12": "End-of-Life Treatment of Sold Products",
    "CAT_13": "Downstream Leased Assets",
    "CAT_14": "Franchises",
    "CAT_15": "Investments",
}

UPSTREAM_CATEGORIES = ["CAT_1", "CAT_2", "CAT_3", "CAT_4", "CAT_5", "CAT_6", "CAT_7", "CAT_8"]
DOWNSTREAM_CATEGORIES = ["CAT_9", "CAT_10", "CAT_11", "CAT_12", "CAT_13", "CAT_14", "CAT_15"]

# Double-counting overlap rules (12 rules)
OVERLAP_RULES = [
    "cat1_vs_cat3_energy",
    "cat1_vs_cat4_logistics",
    "cat1_vs_cat2_capex_opex",
    "cat3_vs_scope2_upstream_energy",
    "cat4_vs_cat9_transport_allocation",
    "cat8_vs_scope12_leased_assets",
    "cat13_vs_cat11_downstream_leased",
    "cat14_vs_scope12_franchise",
    "cat1_vs_cat5_packaging_waste",
    "cat10_vs_cat11_processing_use",
    "cat6_vs_cat7_travel_commuting",
    "cat15_vs_cat13_cat14_investment",
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
            f"Ensure PACK-042 source files are present."
        )

    full_module_name = f"pack042_test.{subdir}.{module_name}"
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
    """Return the absolute path to the PACK-042 root directory."""
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
def scope3_categories() -> List[str]:
    """Return the list of all 15 Scope 3 category identifiers."""
    return list(SCOPE3_CATEGORIES)


@pytest.fixture(scope="session")
def scope3_category_names() -> Dict[str, str]:
    """Return category identifier to human-readable name mapping."""
    return dict(SCOPE3_CATEGORY_NAMES)


@pytest.fixture(scope="session")
def upstream_categories() -> List[str]:
    """Return upstream categories (CAT_1 through CAT_8)."""
    return list(UPSTREAM_CATEGORIES)


@pytest.fixture(scope="session")
def downstream_categories() -> List[str]:
    """Return downstream categories (CAT_9 through CAT_15)."""
    return list(DOWNSTREAM_CATEGORIES)


# =============================================================================
# 2. Organization Profile Fixtures
# =============================================================================


@pytest.fixture
def manufacturing_org() -> Dict[str, Any]:
    """Manufacturing company profile for Scope 3 testing."""
    return {
        "org_id": "ORG-MFG-001",
        "org_name": "Acme Manufacturing GmbH",
        "sector": "MANUFACTURING",
        "naics_code": "33",
        "country": "DE",
        "reporting_year": 2025,
        "revenue_meur": Decimal("500.0"),
        "employees_fte": 2500,
        "total_procurement_spend_eur": Decimal("180000000"),
        "number_of_suppliers": 450,
        "scope1_tco2e": Decimal("12000.0"),
        "scope2_location_tco2e": Decimal("8500.0"),
        "scope2_market_tco2e": Decimal("5200.0"),
        "products": ["automotive_parts", "industrial_machinery"],
        "facilities": 6,
        "countries_of_operation": ["DE", "US", "CN"],
    }


@pytest.fixture
def retail_org() -> Dict[str, Any]:
    """Retail company profile for Scope 3 testing."""
    return {
        "org_id": "ORG-RTL-001",
        "org_name": "EcoMart Retail plc",
        "sector": "RETAIL",
        "naics_code": "44",
        "country": "GB",
        "reporting_year": 2025,
        "revenue_meur": Decimal("2500.0"),
        "employees_fte": 15000,
        "total_procurement_spend_eur": Decimal("1500000000"),
        "number_of_suppliers": 3200,
        "scope1_tco2e": Decimal("5000.0"),
        "scope2_location_tco2e": Decimal("18000.0"),
        "scope2_market_tco2e": Decimal("12000.0"),
        "products": ["consumer_goods", "food_beverage", "household"],
        "facilities": 250,
        "countries_of_operation": ["GB", "IE", "FR"],
    }


@pytest.fixture
def technology_org() -> Dict[str, Any]:
    """Technology company profile for Scope 3 testing."""
    return {
        "org_id": "ORG-TECH-001",
        "org_name": "GreenTech Solutions Inc",
        "sector": "TECHNOLOGY",
        "naics_code": "54",
        "country": "US",
        "reporting_year": 2025,
        "revenue_meur": Decimal("800.0"),
        "employees_fte": 5000,
        "total_procurement_spend_eur": Decimal("250000000"),
        "number_of_suppliers": 180,
        "scope1_tco2e": Decimal("1200.0"),
        "scope2_location_tco2e": Decimal("6000.0"),
        "scope2_market_tco2e": Decimal("2500.0"),
        "products": ["software_platform", "hardware_appliances", "cloud_services"],
        "facilities": 8,
        "countries_of_operation": ["US", "DE", "IN"],
    }


@pytest.fixture
def financial_org() -> Dict[str, Any]:
    """Financial services company profile for Scope 3 testing."""
    return {
        "org_id": "ORG-FIN-001",
        "org_name": "Sustainable Capital AG",
        "sector": "FINANCIAL",
        "naics_code": "52",
        "country": "CH",
        "reporting_year": 2025,
        "revenue_meur": Decimal("1200.0"),
        "employees_fte": 3000,
        "total_procurement_spend_eur": Decimal("80000000"),
        "number_of_suppliers": 120,
        "scope1_tco2e": Decimal("500.0"),
        "scope2_location_tco2e": Decimal("2200.0"),
        "scope2_market_tco2e": Decimal("800.0"),
        "products": ["asset_management", "corporate_lending", "insurance"],
        "facilities": 4,
        "countries_of_operation": ["CH", "DE", "LU"],
        "aum_meur": Decimal("45000.0"),
        "financed_emissions_tco2e": Decimal("850000.0"),
    }


@pytest.fixture
def sme_org() -> Dict[str, Any]:
    """Small-medium enterprise profile for simplified Scope 3."""
    return {
        "org_id": "ORG-SME-001",
        "org_name": "GreenWidgets Ltd",
        "sector": "SME",
        "naics_code": "33",
        "country": "NL",
        "reporting_year": 2025,
        "revenue_meur": Decimal("15.0"),
        "employees_fte": 45,
        "total_procurement_spend_eur": Decimal("5000000"),
        "number_of_suppliers": 25,
        "scope1_tco2e": Decimal("120.0"),
        "scope2_location_tco2e": Decimal("85.0"),
        "scope2_market_tco2e": Decimal("60.0"),
        "products": ["custom_widgets"],
        "facilities": 1,
        "countries_of_operation": ["NL"],
    }


# =============================================================================
# 3. Procurement/Spend Data Fixtures
# =============================================================================


@pytest.fixture
def sample_spend_data() -> List[Dict[str, Any]]:
    """100+ procurement transactions across Scope 3 categories."""
    transactions = []
    base_date = date(2025, 1, 15)

    # Category 1 - Purchased Goods (largest, 50 transactions)
    cat1_items = [
        ("Steel coil - grade 304", "basic_metals", "331110", Decimal("2500000")),
        ("Aluminum sheets 2mm", "basic_metals", "331315", Decimal("1800000")),
        ("Polypropylene pellets", "chemicals_pharmaceuticals", "325211", Decimal("1200000")),
        ("Electronic components PCBs", "electronics_optical", "334412", Decimal("950000")),
        ("Copper wire 2.5mm", "fabricated_metals", "331420", Decimal("780000")),
        ("Industrial lubricants", "chemicals_pharmaceuticals", "324191", Decimal("450000")),
        ("Packaging cardboard", "wood_paper_products", "322211", Decimal("380000")),
        ("Safety equipment PPE", "rubber_plastics", "339113", Decimal("220000")),
        ("Bearings and seals", "machinery_equipment", "332991", Decimal("560000")),
        ("Paint and coatings", "chemicals_pharmaceuticals", "325510", Decimal("310000")),
        ("Fasteners and bolts", "fabricated_metals", "332722", Decimal("180000")),
        ("Rubber gaskets", "rubber_plastics", "326291", Decimal("145000")),
        ("Glass panels", "non_metallic_minerals", "327211", Decimal("290000")),
        ("Welding consumables", "fabricated_metals", "332311", Decimal("125000")),
        ("Hydraulic hoses", "rubber_plastics", "326220", Decimal("95000")),
        ("Insulation materials", "non_metallic_minerals", "327993", Decimal("175000")),
        ("Springs and wire forms", "fabricated_metals", "332612", Decimal("110000")),
        ("Adhesives and sealants", "chemicals_pharmaceuticals", "325520", Decimal("85000")),
        ("Labels and printing", "wood_paper_products", "323111", Decimal("65000")),
        ("Cleaning chemicals", "chemicals_pharmaceuticals", "325612", Decimal("42000")),
        ("Office supplies", "wholesale_trade", "424120", Decimal("35000")),
        ("Maintenance parts", "machinery_equipment", "423830", Decimal("280000")),
        ("Cutting tools", "fabricated_metals", "333515", Decimal("190000")),
        ("Forgings", "fabricated_metals", "332111", Decimal("420000")),
        ("Castings", "basic_metals", "331511", Decimal("380000")),
        ("Rubber compounds", "rubber_plastics", "325212", Decimal("215000")),
        ("Specialty chemicals", "chemicals_pharmaceuticals", "325998", Decimal("165000")),
        ("Protective coatings", "chemicals_pharmaceuticals", "325510", Decimal("98000")),
        ("Wiring harnesses", "electronics_optical", "335931", Decimal("270000")),
        ("Sensors and actuators", "electronics_optical", "334519", Decimal("340000")),
        ("Motors and drives", "electrical_equipment", "335312", Decimal("480000")),
        ("Valves and fittings", "fabricated_metals", "332911", Decimal("195000")),
        ("Filters and filtration", "machinery_equipment", "333999", Decimal("88000")),
        ("Thermal insulation", "non_metallic_minerals", "327993", Decimal("72000")),
        ("Compressed gases", "chemicals_pharmaceuticals", "325120", Decimal("55000")),
        ("Abrasives", "non_metallic_minerals", "327910", Decimal("48000")),
        ("Pallets and crates", "wood_paper_products", "321920", Decimal("62000")),
        ("Stretch film", "rubber_plastics", "326112", Decimal("38000")),
        ("IT hardware", "electronics_optical", "334111", Decimal("185000")),
        ("Software licenses", "it_services", "511210", Decimal("120000")),
        ("Consulting services", "management_consulting", "541610", Decimal("350000")),
        ("Legal services", "legal_accounting", "541110", Decimal("95000")),
        ("Audit services", "legal_accounting", "541211", Decimal("65000")),
        ("Insurance premiums", "insurance", "524210", Decimal("180000")),
        ("Facility management", "cleaning_services", "561720", Decimal("240000")),
        ("Catering services", "food_service", "722310", Decimal("85000")),
        ("Security services", "security_services", "561612", Decimal("110000")),
        ("Temporary staffing", "employment_services", "561320", Decimal("280000")),
        ("Marketing services", "advertising_market_research", "541810", Decimal("150000")),
        ("R&D contract work", "scientific_rd", "541712", Decimal("420000")),
    ]
    for i, (desc, sector, naics, amount) in enumerate(cat1_items):
        transactions.append({
            "transaction_id": f"TXN-2025-{i+1:04d}",
            "date": str(base_date + timedelta(days=i * 7)),
            "description": desc,
            "supplier_id": f"SUP-{i+1:03d}",
            "amount_eur": amount,
            "currency": "EUR",
            "naics_code": naics,
            "eeio_sector": sector,
            "scope3_category": "CAT_1",
            "gl_account": f"5{100+i:04d}",
        })

    # Category 2 - Capital Goods (10 transactions)
    cat2_items = [
        ("CNC milling machine", "machinery_equipment", "333517", Decimal("850000")),
        ("Industrial robot arm", "machinery_equipment", "333249", Decimal("420000")),
        ("HVAC system upgrade", "machinery_equipment", "333415", Decimal("280000")),
        ("Fork lift trucks x3", "motor_vehicles", "333924", Decimal("180000")),
        ("Server rack infrastructure", "electronics_optical", "334111", Decimal("320000")),
        ("Quality testing equipment", "electronics_optical", "334516", Decimal("210000")),
        ("Warehouse racking system", "fabricated_metals", "332999", Decimal("150000")),
        ("Solar panel installation", "electrical_equipment", "335999", Decimal("750000")),
        ("Conveyor belt system", "machinery_equipment", "333922", Decimal("390000")),
        ("Compressor system", "machinery_equipment", "333912", Decimal("260000")),
    ]
    for i, (desc, sector, naics, amount) in enumerate(cat2_items):
        transactions.append({
            "transaction_id": f"TXN-2025-CAP-{i+1:03d}",
            "date": str(base_date + timedelta(days=i * 30)),
            "description": desc,
            "supplier_id": f"SUP-CAP-{i+1:03d}",
            "amount_eur": amount,
            "currency": "EUR",
            "naics_code": naics,
            "eeio_sector": sector,
            "scope3_category": "CAT_2",
            "gl_account": f"1{500+i:04d}",
        })

    # Category 4 - Upstream Transport (10 transactions)
    cat4_items = [
        ("Road freight - DE domestic", "land_transport", "484110", Decimal("320000")),
        ("Road freight - EU cross-border", "land_transport", "484121", Decimal("480000")),
        ("Rail freight services", "land_transport", "482111", Decimal("150000")),
        ("Ocean freight Asia-EU", "water_transport", "483111", Decimal("280000")),
        ("Air freight express", "air_transport", "481112", Decimal("95000")),
        ("Courier services", "postal_courier", "492110", Decimal("45000")),
        ("Warehouse logistics", "warehousing_support", "493110", Decimal("120000")),
        ("Cold chain transport", "land_transport", "484220", Decimal("180000")),
        ("Container shipping", "water_transport", "483111", Decimal("210000")),
        ("Last mile delivery", "postal_courier", "492210", Decimal("75000")),
    ]
    for i, (desc, sector, naics, amount) in enumerate(cat4_items):
        transactions.append({
            "transaction_id": f"TXN-2025-LOG-{i+1:03d}",
            "date": str(base_date + timedelta(days=i * 35)),
            "description": desc,
            "supplier_id": f"SUP-LOG-{i+1:03d}",
            "amount_eur": amount,
            "currency": "EUR",
            "naics_code": naics,
            "eeio_sector": sector,
            "scope3_category": "CAT_4",
            "gl_account": f"6{200+i:04d}",
        })

    # Category 5 - Waste (5 transactions)
    cat5_items = [
        ("General waste disposal", "waste_management_remediation", "562111", Decimal("35000")),
        ("Hazardous waste treatment", "waste_management_remediation", "562112", Decimal("85000")),
        ("Metal scrap recycling", "waste_management_remediation", "562920", Decimal("15000")),
        ("Wastewater treatment", "water_supply_sewerage", "221320", Decimal("22000")),
        ("E-waste recycling", "waste_management_remediation", "562920", Decimal("18000")),
    ]
    for i, (desc, sector, naics, amount) in enumerate(cat5_items):
        transactions.append({
            "transaction_id": f"TXN-2025-WST-{i+1:03d}",
            "date": str(base_date + timedelta(days=i * 60)),
            "description": desc,
            "supplier_id": f"SUP-WST-{i+1:03d}",
            "amount_eur": amount,
            "currency": "EUR",
            "naics_code": naics,
            "eeio_sector": sector,
            "scope3_category": "CAT_5",
            "gl_account": f"6{400+i:04d}",
        })

    # Category 6 - Business Travel (10 transactions)
    cat6_items = [
        ("Flights EU short-haul", "air_transport", "481111", Decimal("85000")),
        ("Flights intercontinental", "air_transport", "481111", Decimal("120000")),
        ("Hotel accommodation", "accommodation", "721110", Decimal("65000")),
        ("Train travel domestic", "land_transport", "482111", Decimal("25000")),
        ("Rental cars", "rental_leasing", "532111", Decimal("35000")),
        ("Taxi and ride-hailing", "land_transport", "485310", Decimal("18000")),
        ("Conference attendance", "accommodation", "721110", Decimal("42000")),
        ("Travel agency fees", "management_consulting", "561510", Decimal("12000")),
        ("Per diem expenses", "food_service", "722511", Decimal("28000")),
        ("Parking and tolls", "land_transport", "488490", Decimal("8000")),
    ]
    for i, (desc, sector, naics, amount) in enumerate(cat6_items):
        transactions.append({
            "transaction_id": f"TXN-2025-TRV-{i+1:03d}",
            "date": str(base_date + timedelta(days=i * 30)),
            "description": desc,
            "supplier_id": f"SUP-TRV-{i+1:03d}",
            "amount_eur": amount,
            "currency": "EUR",
            "naics_code": naics,
            "eeio_sector": sector,
            "scope3_category": "CAT_6",
            "gl_account": f"6{600+i:04d}",
        })

    # Category 7 - Employee Commuting (5 transactions, aggregated)
    cat7_items = [
        ("Employee commuting - car", "land_transport", "000000", Decimal("180000")),
        ("Employee commuting - public transport", "land_transport", "000000", Decimal("45000")),
        ("Employee commuting - cycling", "land_transport", "000000", Decimal("5000")),
        ("Remote work energy", "electricity_gas_steam", "000000", Decimal("25000")),
        ("Shuttle bus service", "land_transport", "485510", Decimal("35000")),
    ]
    for i, (desc, sector, naics, amount) in enumerate(cat7_items):
        transactions.append({
            "transaction_id": f"TXN-2025-CMT-{i+1:03d}",
            "date": str(base_date),
            "description": desc,
            "supplier_id": f"SUP-CMT-{i+1:03d}",
            "amount_eur": amount,
            "currency": "EUR",
            "naics_code": naics,
            "eeio_sector": sector,
            "scope3_category": "CAT_7",
            "gl_account": f"6{700+i:04d}",
        })

    # Category 9 - Downstream Transport (5 transactions)
    cat9_items = [
        ("Outbound distribution DE", "land_transport", "484110", Decimal("250000")),
        ("Outbound distribution EU", "land_transport", "484121", Decimal("180000")),
        ("Last-mile delivery partner", "postal_courier", "492210", Decimal("95000")),
        ("Export shipping", "water_transport", "483111", Decimal("130000")),
        ("Returns logistics", "land_transport", "484110", Decimal("45000")),
    ]
    for i, (desc, sector, naics, amount) in enumerate(cat9_items):
        transactions.append({
            "transaction_id": f"TXN-2025-DST-{i+1:03d}",
            "date": str(base_date + timedelta(days=i * 60)),
            "description": desc,
            "supplier_id": f"SUP-DST-{i+1:03d}",
            "amount_eur": amount,
            "currency": "EUR",
            "naics_code": naics,
            "eeio_sector": sector,
            "scope3_category": "CAT_9",
            "gl_account": f"7{100+i:04d}",
        })

    # Additional miscellaneous categories (5 transactions)
    misc_items = [
        ("Cat 3 - WTT natural gas", "electricity_gas_steam", "221210", Decimal("95000"), "CAT_3"),
        ("Cat 8 - Leased office rent", "real_estate", "531110", Decimal("450000"), "CAT_8"),
        ("Cat 11 - Product energy use", "electronics_optical", "335311", Decimal("800000"), "CAT_11"),
        ("Cat 12 - End-of-life estimate", "waste_management_remediation", "562111", Decimal("120000"), "CAT_12"),
        ("Cat 15 - Equity investment", "financial_services", "523110", Decimal("5000000"), "CAT_15"),
    ]
    for i, (desc, sector, naics, amount, cat) in enumerate(misc_items):
        transactions.append({
            "transaction_id": f"TXN-2025-MSC-{i+1:03d}",
            "date": str(base_date + timedelta(days=i * 90)),
            "description": desc,
            "supplier_id": f"SUP-MSC-{i+1:03d}",
            "amount_eur": amount,
            "currency": "EUR",
            "naics_code": naics,
            "eeio_sector": sector,
            "scope3_category": cat,
            "gl_account": f"8{100+i:04d}",
        })

    return transactions


# =============================================================================
# 4. Category Result Fixtures
# =============================================================================


@pytest.fixture
def sample_category_results() -> Dict[str, Any]:
    """Per-category Scope 3 results for all 15 categories."""
    return {
        "reporting_year": 2025,
        "org_id": "ORG-MFG-001",
        "categories": {
            "CAT_1": {
                "total_tco2e": Decimal("28500.0"),
                "methodology": "HYBRID",
                "dqr": Decimal("3.2"),
                "by_gas": {"CO2": Decimal("27800.0"), "CH4": Decimal("450.0"), "N2O": Decimal("250.0")},
                "uncertainty_pct": Decimal("35.0"),
                "source_count": 50,
                "spend_eur": Decimal("15235000"),
            },
            "CAT_2": {
                "total_tco2e": Decimal("4200.0"),
                "methodology": "SPEND_BASED",
                "dqr": Decimal("4.0"),
                "by_gas": {"CO2": Decimal("4100.0"), "CH4": Decimal("60.0"), "N2O": Decimal("40.0")},
                "uncertainty_pct": Decimal("55.0"),
                "source_count": 10,
                "spend_eur": Decimal("3810000"),
            },
            "CAT_3": {
                "total_tco2e": Decimal("3800.0"),
                "methodology": "AVERAGE_DATA",
                "dqr": Decimal("2.5"),
                "by_gas": {"CO2": Decimal("3600.0"), "CH4": Decimal("120.0"), "N2O": Decimal("80.0")},
                "uncertainty_pct": Decimal("25.0"),
                "source_count": 8,
                "spend_eur": Decimal("0"),
            },
            "CAT_4": {
                "total_tco2e": Decimal("5100.0"),
                "methodology": "SPEND_BASED",
                "dqr": Decimal("3.5"),
                "by_gas": {"CO2": Decimal("4950.0"), "CH4": Decimal("80.0"), "N2O": Decimal("70.0")},
                "uncertainty_pct": Decimal("45.0"),
                "source_count": 10,
                "spend_eur": Decimal("1955000"),
            },
            "CAT_5": {
                "total_tco2e": Decimal("850.0"),
                "methodology": "AVERAGE_DATA",
                "dqr": Decimal("3.0"),
                "by_gas": {"CO2": Decimal("680.0"), "CH4": Decimal("120.0"), "N2O": Decimal("50.0")},
                "uncertainty_pct": Decimal("30.0"),
                "source_count": 5,
                "spend_eur": Decimal("175000"),
            },
            "CAT_6": {
                "total_tco2e": Decimal("1200.0"),
                "methodology": "SPEND_BASED",
                "dqr": Decimal("3.8"),
                "by_gas": {"CO2": Decimal("1180.0"), "CH4": Decimal("10.0"), "N2O": Decimal("10.0")},
                "uncertainty_pct": Decimal("40.0"),
                "source_count": 10,
                "spend_eur": Decimal("438000"),
            },
            "CAT_7": {
                "total_tco2e": Decimal("980.0"),
                "methodology": "AVERAGE_DATA",
                "dqr": Decimal("4.2"),
                "by_gas": {"CO2": Decimal("960.0"), "CH4": Decimal("12.0"), "N2O": Decimal("8.0")},
                "uncertainty_pct": Decimal("50.0"),
                "source_count": 5,
                "spend_eur": Decimal("290000"),
            },
            "CAT_8": {
                "total_tco2e": Decimal("350.0"),
                "methodology": "AVERAGE_DATA",
                "dqr": Decimal("4.0"),
                "by_gas": {"CO2": Decimal("340.0"), "CH4": Decimal("5.0"), "N2O": Decimal("5.0")},
                "uncertainty_pct": Decimal("45.0"),
                "source_count": 2,
                "spend_eur": Decimal("450000"),
            },
            "CAT_9": {
                "total_tco2e": Decimal("3200.0"),
                "methodology": "AVERAGE_DATA",
                "dqr": Decimal("3.8"),
                "by_gas": {"CO2": Decimal("3100.0"), "CH4": Decimal("55.0"), "N2O": Decimal("45.0")},
                "uncertainty_pct": Decimal("50.0"),
                "source_count": 5,
                "spend_eur": Decimal("700000"),
            },
            "CAT_10": {
                "total_tco2e": Decimal("2800.0"),
                "methodology": "AVERAGE_DATA",
                "dqr": Decimal("4.2"),
                "by_gas": {"CO2": Decimal("2720.0"), "CH4": Decimal("45.0"), "N2O": Decimal("35.0")},
                "uncertainty_pct": Decimal("60.0"),
                "source_count": 3,
                "spend_eur": Decimal("0"),
            },
            "CAT_11": {
                "total_tco2e": Decimal("8500.0"),
                "methodology": "AVERAGE_DATA",
                "dqr": Decimal("4.0"),
                "by_gas": {"CO2": Decimal("8400.0"), "CH4": Decimal("50.0"), "N2O": Decimal("50.0")},
                "uncertainty_pct": Decimal("65.0"),
                "source_count": 4,
                "spend_eur": Decimal("0"),
            },
            "CAT_12": {
                "total_tco2e": Decimal("1500.0"),
                "methodology": "AVERAGE_DATA",
                "dqr": Decimal("4.5"),
                "by_gas": {"CO2": Decimal("1200.0"), "CH4": Decimal("200.0"), "N2O": Decimal("100.0")},
                "uncertainty_pct": Decimal("70.0"),
                "source_count": 3,
                "spend_eur": Decimal("0"),
            },
            "CAT_13": {
                "total_tco2e": Decimal("0"),
                "methodology": "NOT_APPLICABLE",
                "dqr": Decimal("0"),
                "by_gas": {"CO2": Decimal("0"), "CH4": Decimal("0"), "N2O": Decimal("0")},
                "uncertainty_pct": Decimal("0"),
                "source_count": 0,
                "spend_eur": Decimal("0"),
            },
            "CAT_14": {
                "total_tco2e": Decimal("0"),
                "methodology": "NOT_APPLICABLE",
                "dqr": Decimal("0"),
                "by_gas": {"CO2": Decimal("0"), "CH4": Decimal("0"), "N2O": Decimal("0")},
                "uncertainty_pct": Decimal("0"),
                "source_count": 0,
                "spend_eur": Decimal("0"),
            },
            "CAT_15": {
                "total_tco2e": Decimal("450.0"),
                "methodology": "SPEND_BASED",
                "dqr": Decimal("4.8"),
                "by_gas": {"CO2": Decimal("445.0"), "CH4": Decimal("3.0"), "N2O": Decimal("2.0")},
                "uncertainty_pct": Decimal("80.0"),
                "source_count": 1,
                "spend_eur": Decimal("5000000"),
            },
        },
        "total_scope3_tco2e": Decimal("61430.0"),
    }


# =============================================================================
# 5. Supplier Data Fixtures
# =============================================================================


@pytest.fixture
def sample_supplier_data() -> List[Dict[str, Any]]:
    """20+ suppliers with engagement status and data quality."""
    return [
        {"supplier_id": "SUP-001", "name": "Stahlwerk Nord GmbH", "category": "CAT_1", "spend_eur": Decimal("2500000"), "emissions_tco2e": Decimal("7375"), "engagement_status": "COMPLETED", "data_quality_level": "LEVEL_2", "country": "DE", "sector": "basic_metals", "response_date": "2025-06-15"},
        {"supplier_id": "SUP-002", "name": "AluminiumWerke AG", "category": "CAT_1", "spend_eur": Decimal("1800000"), "emissions_tco2e": Decimal("5310"), "engagement_status": "COMPLETED", "data_quality_level": "LEVEL_2", "country": "DE", "sector": "basic_metals", "response_date": "2025-07-01"},
        {"supplier_id": "SUP-003", "name": "ChemPlast BV", "category": "CAT_1", "spend_eur": Decimal("1200000"), "emissions_tco2e": Decimal("1860"), "engagement_status": "IN_PROGRESS", "data_quality_level": "LEVEL_3", "country": "NL", "sector": "chemicals_pharmaceuticals"},
        {"supplier_id": "SUP-004", "name": "TechComp Taiwan Ltd", "category": "CAT_1", "spend_eur": Decimal("950000"), "emissions_tco2e": Decimal("427"), "engagement_status": "DATA_REQUESTED", "data_quality_level": "LEVEL_4", "country": "TW", "sector": "electronics_optical"},
        {"supplier_id": "SUP-005", "name": "CopperTrade SA", "category": "CAT_1", "spend_eur": Decimal("780000"), "emissions_tco2e": Decimal("858"), "engagement_status": "NOT_STARTED", "data_quality_level": "LEVEL_5", "country": "CL", "sector": "fabricated_metals"},
        {"supplier_id": "SUP-006", "name": "LubeMax Industries", "category": "CAT_1", "spend_eur": Decimal("450000"), "emissions_tco2e": Decimal("697"), "engagement_status": "COMPLETED", "data_quality_level": "LEVEL_1", "country": "DE", "sector": "chemicals_pharmaceuticals", "response_date": "2025-05-20"},
        {"supplier_id": "SUP-007", "name": "PackCorp GmbH", "category": "CAT_1", "spend_eur": Decimal("380000"), "emissions_tco2e": Decimal("372"), "engagement_status": "IN_PROGRESS", "data_quality_level": "LEVEL_3", "country": "DE", "sector": "wood_paper_products"},
        {"supplier_id": "SUP-008", "name": "SafetyFirst PPE Ltd", "category": "CAT_1", "spend_eur": Decimal("220000"), "emissions_tco2e": Decimal("391"), "engagement_status": "OVERDUE", "data_quality_level": "LEVEL_5", "country": "CN", "sector": "rubber_plastics"},
        {"supplier_id": "SUP-009", "name": "BearingTech Japan", "category": "CAT_1", "spend_eur": Decimal("560000"), "emissions_tco2e": Decimal("324"), "engagement_status": "COMPLETED", "data_quality_level": "LEVEL_2", "country": "JP", "sector": "machinery_equipment", "response_date": "2025-08-10"},
        {"supplier_id": "SUP-010", "name": "CoatMaster AG", "category": "CAT_1", "spend_eur": Decimal("310000"), "emissions_tco2e": Decimal("480"), "engagement_status": "NOT_STARTED", "data_quality_level": "LEVEL_5", "country": "DE", "sector": "chemicals_pharmaceuticals"},
        {"supplier_id": "SUP-CAP-001", "name": "MachineWorks GmbH", "category": "CAT_2", "spend_eur": Decimal("850000"), "emissions_tco2e": Decimal("493"), "engagement_status": "COMPLETED", "data_quality_level": "LEVEL_3", "country": "DE", "sector": "machinery_equipment", "response_date": "2025-09-01"},
        {"supplier_id": "SUP-CAP-002", "name": "RoboTech Inc", "category": "CAT_2", "spend_eur": Decimal("420000"), "emissions_tco2e": Decimal("243"), "engagement_status": "DATA_REQUESTED", "data_quality_level": "LEVEL_4", "country": "US", "sector": "machinery_equipment"},
        {"supplier_id": "SUP-LOG-001", "name": "TransEurope Freight", "category": "CAT_4", "spend_eur": Decimal("800000"), "emissions_tco2e": Decimal("1520"), "engagement_status": "COMPLETED", "data_quality_level": "LEVEL_2", "country": "DE", "sector": "land_transport", "response_date": "2025-05-15"},
        {"supplier_id": "SUP-LOG-002", "name": "OceanShip Ltd", "category": "CAT_4", "spend_eur": Decimal("490000"), "emissions_tco2e": Decimal("1274"), "engagement_status": "IN_PROGRESS", "data_quality_level": "LEVEL_3", "country": "GR", "sector": "water_transport"},
        {"supplier_id": "SUP-LOG-003", "name": "AirCargo Express", "category": "CAT_4", "spend_eur": Decimal("95000"), "emissions_tco2e": Decimal("361"), "engagement_status": "NOT_STARTED", "data_quality_level": "LEVEL_5", "country": "AE", "sector": "air_transport"},
        {"supplier_id": "SUP-WST-001", "name": "WasteManagement DE", "category": "CAT_5", "spend_eur": Decimal("120000"), "emissions_tco2e": Decimal("216"), "engagement_status": "COMPLETED", "data_quality_level": "LEVEL_2", "country": "DE", "sector": "waste_management_remediation", "response_date": "2025-04-20"},
        {"supplier_id": "SUP-WST-002", "name": "HazWaste Solutions", "category": "CAT_5", "spend_eur": Decimal("85000"), "emissions_tco2e": Decimal("153"), "engagement_status": "DATA_REQUESTED", "data_quality_level": "LEVEL_4", "country": "DE", "sector": "waste_management_remediation"},
        {"supplier_id": "SUP-TRV-001", "name": "Lufthansa Group", "category": "CAT_6", "spend_eur": Decimal("205000"), "emissions_tco2e": Decimal("779"), "engagement_status": "COMPLETED", "data_quality_level": "LEVEL_1", "country": "DE", "sector": "air_transport", "response_date": "2025-03-15"},
        {"supplier_id": "SUP-TRV-002", "name": "Deutsche Bahn", "category": "CAT_6", "spend_eur": Decimal("25000"), "emissions_tco2e": Decimal("47"), "engagement_status": "COMPLETED", "data_quality_level": "LEVEL_1", "country": "DE", "sector": "land_transport", "response_date": "2025-03-20"},
        {"supplier_id": "SUP-DST-001", "name": "DistributionCo GmbH", "category": "CAT_9", "spend_eur": Decimal("430000"), "emissions_tco2e": Decimal("817"), "engagement_status": "IN_PROGRESS", "data_quality_level": "LEVEL_3", "country": "DE", "sector": "land_transport"},
        {"supplier_id": "SUP-DST-002", "name": "ParcelForce EU", "category": "CAT_9", "spend_eur": Decimal("95000"), "emissions_tco2e": Decimal("66"), "engagement_status": "NOT_STARTED", "data_quality_level": "LEVEL_5", "country": "GB", "sector": "postal_courier"},
    ]


# =============================================================================
# 6. EEIO Emission Factor Fixtures
# =============================================================================


@pytest.fixture
def sample_eeio_factors() -> Dict[str, float]:
    """EEIO emission factors (kgCO2e per EUR) by sector."""
    return {
        "agriculture_forestry": 2.85,
        "mining_quarrying": 1.42,
        "food_beverages_tobacco": 1.65,
        "textiles_leather": 1.20,
        "wood_paper_products": 0.98,
        "chemicals_pharmaceuticals": 1.55,
        "rubber_plastics": 1.78,
        "non_metallic_minerals": 2.40,
        "basic_metals": 2.95,
        "fabricated_metals": 1.10,
        "electronics_optical": 0.45,
        "electrical_equipment": 0.62,
        "machinery_equipment": 0.58,
        "motor_vehicles": 0.72,
        "furniture_other_manufacturing": 0.85,
        "construction": 1.15,
        "wholesale_trade": 0.25,
        "retail_trade": 0.30,
        "land_transport": 1.90,
        "water_transport": 2.60,
        "air_transport": 3.80,
        "warehousing_support": 0.55,
        "postal_courier": 0.70,
        "accommodation": 0.48,
        "food_service": 0.65,
        "telecommunications": 0.22,
        "it_services": 0.18,
        "financial_services": 0.12,
        "insurance": 0.10,
        "real_estate": 0.20,
        "legal_accounting": 0.14,
        "management_consulting": 0.16,
        "scientific_rd": 0.25,
        "advertising_market_research": 0.13,
        "rental_leasing": 0.28,
        "employment_services": 0.11,
        "security_services": 0.17,
        "cleaning_services": 0.35,
        "waste_management_remediation": 1.80,
        "electricity_gas_steam": 3.50,
        "water_supply_sewerage": 0.95,
    }


# =============================================================================
# 7. Screening Result Fixtures
# =============================================================================


@pytest.fixture
def sample_screening_results() -> Dict[str, Any]:
    """Screening-level results for all 15 categories."""
    return {
        "screening_id": "SCR-2025-001",
        "org_id": "ORG-MFG-001",
        "reporting_year": 2025,
        "categories": {
            "CAT_1": {"estimated_tco2e": Decimal("30000"), "relevance": "HIGH", "pct_of_total": Decimal("42.5"), "recommended_tier": "HYBRID"},
            "CAT_2": {"estimated_tco2e": Decimal("4500"), "relevance": "HIGH", "pct_of_total": Decimal("6.4"), "recommended_tier": "SPEND_BASED"},
            "CAT_3": {"estimated_tco2e": Decimal("4000"), "relevance": "HIGH", "pct_of_total": Decimal("5.7"), "recommended_tier": "AVERAGE_DATA"},
            "CAT_4": {"estimated_tco2e": Decimal("5500"), "relevance": "HIGH", "pct_of_total": Decimal("7.8"), "recommended_tier": "SPEND_BASED"},
            "CAT_5": {"estimated_tco2e": Decimal("900"), "relevance": "MEDIUM", "pct_of_total": Decimal("1.3"), "recommended_tier": "AVERAGE_DATA"},
            "CAT_6": {"estimated_tco2e": Decimal("1300"), "relevance": "MEDIUM", "pct_of_total": Decimal("1.8"), "recommended_tier": "SPEND_BASED"},
            "CAT_7": {"estimated_tco2e": Decimal("1000"), "relevance": "MEDIUM", "pct_of_total": Decimal("1.4"), "recommended_tier": "AVERAGE_DATA"},
            "CAT_8": {"estimated_tco2e": Decimal("400"), "relevance": "LOW", "pct_of_total": Decimal("0.6"), "recommended_tier": "AVERAGE_DATA"},
            "CAT_9": {"estimated_tco2e": Decimal("3500"), "relevance": "HIGH", "pct_of_total": Decimal("5.0"), "recommended_tier": "AVERAGE_DATA"},
            "CAT_10": {"estimated_tco2e": Decimal("3000"), "relevance": "HIGH", "pct_of_total": Decimal("4.3"), "recommended_tier": "AVERAGE_DATA"},
            "CAT_11": {"estimated_tco2e": Decimal("9000"), "relevance": "HIGH", "pct_of_total": Decimal("12.8"), "recommended_tier": "AVERAGE_DATA"},
            "CAT_12": {"estimated_tco2e": Decimal("1600"), "relevance": "MEDIUM", "pct_of_total": Decimal("2.3"), "recommended_tier": "AVERAGE_DATA"},
            "CAT_13": {"estimated_tco2e": Decimal("0"), "relevance": "NOT_APPLICABLE", "pct_of_total": Decimal("0"), "recommended_tier": "NOT_APPLICABLE"},
            "CAT_14": {"estimated_tco2e": Decimal("0"), "relevance": "NOT_APPLICABLE", "pct_of_total": Decimal("0"), "recommended_tier": "NOT_APPLICABLE"},
            "CAT_15": {"estimated_tco2e": Decimal("500"), "relevance": "LOW", "pct_of_total": Decimal("0.7"), "recommended_tier": "SPEND_BASED"},
        },
        "total_estimated_tco2e": Decimal("65200"),
        "significant_categories": ["CAT_1", "CAT_2", "CAT_3", "CAT_4", "CAT_9", "CAT_10", "CAT_11"],
        "provenance_hash": "a" * 64,
    }


# =============================================================================
# 8. Consolidated Inventory Fixture
# =============================================================================


@pytest.fixture
def sample_consolidated_inventory(sample_category_results) -> Dict[str, Any]:
    """Complete consolidated Scope 3 inventory."""
    cats = sample_category_results["categories"]
    upstream_total = sum(
        cats[c]["total_tco2e"] for c in UPSTREAM_CATEGORIES
    )
    downstream_total = sum(
        cats[c]["total_tco2e"] for c in DOWNSTREAM_CATEGORIES
    )
    total = upstream_total + downstream_total

    return {
        "inventory_id": "INV-S3-2025-001",
        "org_id": "ORG-MFG-001",
        "reporting_year": 2025,
        "total_scope3_tco2e": total,
        "upstream_tco2e": upstream_total,
        "downstream_tco2e": downstream_total,
        "upstream_pct": float(upstream_total / total * 100) if total else 0,
        "downstream_pct": float(downstream_total / total * 100) if total else 0,
        "scope1_tco2e": Decimal("12000.0"),
        "scope2_market_tco2e": Decimal("5200.0"),
        "total_footprint_tco2e": total + Decimal("12000.0") + Decimal("5200.0"),
        "scope3_pct_of_total": float(total / (total + Decimal("17200.0")) * 100),
        "categories": cats,
        "by_gas": {
            "CO2": sum(cats[c]["by_gas"]["CO2"] for c in SCOPE3_CATEGORIES),
            "CH4": sum(cats[c]["by_gas"]["CH4"] for c in SCOPE3_CATEGORIES),
            "N2O": sum(cats[c]["by_gas"]["N2O"] for c in SCOPE3_CATEGORIES),
        },
        "weighted_dqr": Decimal("3.6"),
        "provenance_hash": compute_provenance_hash({"total": str(total)}),
    }


# =============================================================================
# 9. Double-Counting Detection Fixtures
# =============================================================================


@pytest.fixture
def sample_double_counting_results() -> Dict[str, Any]:
    """Double-counting detection results with overlaps identified."""
    return {
        "assessment_id": "DC-2025-001",
        "overlaps_detected": [
            {
                "rule": "cat1_vs_cat4_logistics",
                "categories": ["CAT_1", "CAT_4"],
                "overlap_tco2e": Decimal("450.0"),
                "allocation_method": "ECONOMIC",
                "adjustment_category": "CAT_4",
                "adjustment_tco2e": Decimal("-450.0"),
                "rationale": "Transport costs included in Cat 1 supplier invoices already cover upstream logistics",
            },
            {
                "rule": "cat3_vs_scope2_upstream_energy",
                "categories": ["CAT_3", "SCOPE_2"],
                "overlap_tco2e": Decimal("0"),
                "allocation_method": "PHYSICAL",
                "adjustment_category": None,
                "adjustment_tco2e": Decimal("0"),
                "rationale": "Cat 3 correctly excludes Scope 2 emissions; WTT and T&D only",
            },
            {
                "rule": "cat1_vs_cat2_capex_opex",
                "categories": ["CAT_1", "CAT_2"],
                "overlap_tco2e": Decimal("120.0"),
                "allocation_method": "ECONOMIC",
                "adjustment_category": "CAT_1",
                "adjustment_tco2e": Decimal("-120.0"),
                "rationale": "Maintenance parts incorrectly classified as OPEX; should be CAPEX",
            },
        ],
        "total_overlap_tco2e": Decimal("570.0"),
        "net_adjustment_tco2e": Decimal("-570.0"),
        "rules_evaluated": 12,
        "rules_with_overlap": 2,
        "allocation_method": "ECONOMIC",
        "conservative_mode": True,
        "provenance_hash": "b" * 64,
    }


# =============================================================================
# 10. Hotspot Analysis Fixtures
# =============================================================================


@pytest.fixture
def sample_hotspot_analysis() -> Dict[str, Any]:
    """Hotspot analysis results with Pareto identification."""
    return {
        "analysis_id": "HOT-2025-001",
        "pareto_categories": [
            {"category": "CAT_1", "tco2e": Decimal("28500"), "cumulative_pct": Decimal("46.4")},
            {"category": "CAT_11", "tco2e": Decimal("8500"), "cumulative_pct": Decimal("60.2")},
            {"category": "CAT_4", "tco2e": Decimal("5100"), "cumulative_pct": Decimal("68.5")},
            {"category": "CAT_2", "tco2e": Decimal("4200"), "cumulative_pct": Decimal("75.4")},
            {"category": "CAT_3", "tco2e": Decimal("3800"), "cumulative_pct": Decimal("81.6")},
        ],
        "pareto_threshold_pct": 80.0,
        "categories_to_80pct": 5,
        "top_suppliers": [
            {"supplier": "Stahlwerk Nord GmbH", "tco2e": Decimal("7375"), "pct_of_cat1": Decimal("25.9")},
            {"supplier": "AluminiumWerke AG", "tco2e": Decimal("5310"), "pct_of_cat1": Decimal("18.6")},
        ],
        "reduction_opportunities": [
            {"action": "Supplier engagement top 10", "potential_reduction_tco2e": Decimal("3500"), "effort": "MEDIUM"},
            {"action": "Switch to recycled steel", "potential_reduction_tco2e": Decimal("2800"), "effort": "HIGH"},
            {"action": "Optimize logistics routes", "potential_reduction_tco2e": Decimal("1200"), "effort": "LOW"},
        ],
        "materiality_matrix": {
            "CAT_1": {"magnitude": "HIGH", "data_quality": "MEDIUM", "reduction_potential": "HIGH"},
            "CAT_11": {"magnitude": "HIGH", "data_quality": "LOW", "reduction_potential": "MEDIUM"},
        },
        "provenance_hash": "c" * 64,
    }


# =============================================================================
# 11. Data Quality Assessment Fixtures
# =============================================================================


@pytest.fixture
def sample_data_quality() -> Dict[str, Any]:
    """Data quality assessment with 5 DQI scores per category."""
    return {
        "assessment_id": "DQ-2025-001",
        "overall_dqr": Decimal("3.6"),
        "categories": {
            "CAT_1": {
                "dqr": Decimal("3.2"),
                "dqi": {
                    "technological_representativeness": Decimal("3.0"),
                    "temporal_representativeness": Decimal("2.5"),
                    "geographical_representativeness": Decimal("3.0"),
                    "completeness": Decimal("3.5"),
                    "reliability": Decimal("4.0"),
                },
                "tier": "HYBRID",
                "level": "LEVEL_3",
            },
            "CAT_2": {
                "dqr": Decimal("4.0"),
                "dqi": {
                    "technological_representativeness": Decimal("4.0"),
                    "temporal_representativeness": Decimal("3.5"),
                    "geographical_representativeness": Decimal("4.0"),
                    "completeness": Decimal("4.5"),
                    "reliability": Decimal("4.0"),
                },
                "tier": "SPEND_BASED",
                "level": "LEVEL_4",
            },
            "CAT_3": {
                "dqr": Decimal("2.5"),
                "dqi": {
                    "technological_representativeness": Decimal("2.0"),
                    "temporal_representativeness": Decimal("2.0"),
                    "geographical_representativeness": Decimal("2.5"),
                    "completeness": Decimal("3.0"),
                    "reliability": Decimal("3.0"),
                },
                "tier": "AVERAGE_DATA",
                "level": "LEVEL_3",
            },
            "CAT_4": {
                "dqr": Decimal("3.5"),
                "dqi": {
                    "technological_representativeness": Decimal("3.5"),
                    "temporal_representativeness": Decimal("3.0"),
                    "geographical_representativeness": Decimal("3.5"),
                    "completeness": Decimal("4.0"),
                    "reliability": Decimal("3.5"),
                },
                "tier": "SPEND_BASED",
                "level": "LEVEL_4",
            },
        },
        "gap_analysis": [
            {"category": "CAT_2", "current_dqr": Decimal("4.0"), "target_dqr": Decimal("3.0"), "gap": Decimal("1.0"), "priority": "HIGH"},
            {"category": "CAT_4", "current_dqr": Decimal("3.5"), "target_dqr": Decimal("3.0"), "gap": Decimal("0.5"), "priority": "MEDIUM"},
        ],
        "improvement_actions": [
            {"action": "Collect primary data from top 10 Cat 1 suppliers", "impact": "HIGH", "effort": "MEDIUM", "expected_dqr_improvement": Decimal("0.8")},
            {"action": "Switch Cat 2 from spend-based to average-data", "impact": "MEDIUM", "effort": "LOW", "expected_dqr_improvement": Decimal("0.5")},
        ],
        "provenance_hash": "d" * 64,
    }


# =============================================================================
# 12. Uncertainty Analysis Fixtures
# =============================================================================


@pytest.fixture
def sample_uncertainty_results() -> Dict[str, Any]:
    """Uncertainty analysis results with Monte Carlo output."""
    return {
        "analysis_id": "UNC-2025-001",
        "method": "MONTE_CARLO",
        "iterations": 10000,
        "seed": 42,
        "total_scope3": {
            "point_estimate_tco2e": Decimal("61430"),
            "lower_bound_tco2e": Decimal("42000"),
            "upper_bound_tco2e": Decimal("89000"),
            "confidence_level": Decimal("0.95"),
            "relative_uncertainty_pct": Decimal("38.3"),
            "std_dev_tco2e": Decimal("12050"),
        },
        "per_category": {
            "CAT_1": {"point": Decimal("28500"), "lower": Decimal("18525"), "upper": Decimal("38475"), "uncertainty_pct": Decimal("35")},
            "CAT_2": {"point": Decimal("4200"), "lower": Decimal("1890"), "upper": Decimal("6510"), "uncertainty_pct": Decimal("55")},
            "CAT_3": {"point": Decimal("3800"), "lower": Decimal("2850"), "upper": Decimal("4750"), "uncertainty_pct": Decimal("25")},
            "CAT_4": {"point": Decimal("5100"), "lower": Decimal("2805"), "upper": Decimal("7395"), "uncertainty_pct": Decimal("45")},
            "CAT_11": {"point": Decimal("8500"), "lower": Decimal("2975"), "upper": Decimal("14025"), "uncertainty_pct": Decimal("65")},
        },
        "sensitivity_ranking": [
            {"category": "CAT_1", "sensitivity_index": Decimal("0.42")},
            {"category": "CAT_11", "sensitivity_index": Decimal("0.28")},
            {"category": "CAT_4", "sensitivity_index": Decimal("0.12")},
        ],
        "tier_upgrade_impact": [
            {"category": "CAT_1", "from_tier": "SPEND_BASED", "to_tier": "SUPPLIER_SPECIFIC", "uncertainty_reduction_pct": Decimal("65")},
            {"category": "CAT_4", "from_tier": "SPEND_BASED", "to_tier": "AVERAGE_DATA", "uncertainty_reduction_pct": Decimal("30")},
        ],
        "provenance_hash": "e" * 64,
    }


# =============================================================================
# 13. Compliance Assessment Fixtures
# =============================================================================


@pytest.fixture
def sample_compliance_results() -> Dict[str, Any]:
    """Multi-framework compliance assessment results."""
    return {
        "assessment_id": "CMP-2025-001",
        "frameworks": {
            "GHG_PROTOCOL": {
                "score_pct": Decimal("85.0"),
                "status": "SUBSTANTIALLY_COMPLIANT",
                "requirements_met": 12,
                "requirements_total": 15,
                "gaps": [
                    {"requirement": "Biogenic CO2 separate disclosure", "status": "GAP", "effort": "LOW"},
                    {"requirement": "All 15 categories screened", "status": "PARTIAL", "effort": "MEDIUM"},
                    {"requirement": "Avoided emissions separate from inventory", "status": "GAP", "effort": "LOW"},
                ],
            },
            "ESRS_E1": {
                "score_pct": Decimal("72.0"),
                "status": "PHASE_IN_COMPLIANT",
                "requirements_met": 8,
                "requirements_total": 12,
                "phase_in_year": 2025,
                "required_categories_2025": ["CAT_1", "CAT_2", "CAT_3"],
                "required_categories_2029": SCOPE3_CATEGORIES,
                "gaps": [
                    {"requirement": "XBRL tagging for E1-6 data points", "status": "GAP", "effort": "MEDIUM"},
                    {"requirement": "Estimation methodology disclosure", "status": "PARTIAL", "effort": "LOW"},
                ],
            },
            "CDP": {
                "score_pct": Decimal("78.0"),
                "status": "B_LEVEL",
                "requirements_met": 10,
                "requirements_total": 13,
                "gaps": [
                    {"requirement": "C6.5 per-category methodology detail", "status": "PARTIAL", "effort": "LOW"},
                    {"requirement": "C6.10 supplier engagement disclosure", "status": "PARTIAL", "effort": "MEDIUM"},
                ],
            },
            "SBTI": {
                "score_pct": Decimal("65.0"),
                "status": "NOT_YET_ALIGNED",
                "requirements_met": 7,
                "requirements_total": 11,
                "gaps": [
                    {"requirement": "67% Scope 3 coverage by category", "status": "PARTIAL", "effort": "HIGH"},
                    {"requirement": "Near-term SBT approved", "status": "GAP", "effort": "HIGH"},
                    {"requirement": "Category-level targets for material categories", "status": "GAP", "effort": "MEDIUM"},
                ],
            },
            "SEC": {
                "score_pct": Decimal("90.0"),
                "status": "SAFE_HARBOUR_COMPLIANT",
                "requirements_met": 9,
                "requirements_total": 10,
                "gaps": [
                    {"requirement": "Board governance of Scope 3 disclosure", "status": "PARTIAL", "effort": "LOW"},
                ],
            },
            "SB_253": {
                "score_pct": Decimal("60.0"),
                "status": "NOT_YET_REQUIRED",
                "requirements_met": 6,
                "requirements_total": 10,
                "applicable_from": 2027,
                "gaps": [
                    {"requirement": "Per-category assurance-ready data", "status": "GAP", "effort": "HIGH"},
                    {"requirement": "Category 1-15 all quantified", "status": "PARTIAL", "effort": "MEDIUM"},
                ],
            },
        },
        "overall_score_pct": Decimal("75.0"),
        "action_plan": [
            {"action": "Complete screening for all 15 categories", "priority": "HIGH", "frameworks_impacted": ["GHG_PROTOCOL", "CDP", "SBTI"]},
            {"action": "Enable XBRL tagging for ESRS E1", "priority": "MEDIUM", "frameworks_impacted": ["ESRS_E1"]},
            {"action": "Submit SBTi target for validation", "priority": "HIGH", "frameworks_impacted": ["SBTI"]},
        ],
        "provenance_hash": "f" * 64,
    }


# =============================================================================
# 14. Pack Configuration Fixtures
# =============================================================================


@pytest.fixture
def sample_pack_config() -> Dict[str, Any]:
    """Default PackConfig settings for testing."""
    return {
        "pack_id": "PACK-042",
        "pack_name": "Scope 3 Starter Pack",
        "version": "1.0.0",
        "category": "ghg-accounting",
        "environment": "test",
        "sector_type": "MANUFACTURING",
        "country": "DE",
        "reporting_year": 2025,
        "revenue_meur": Decimal("500.0"),
        "employees_fte": 2500,
        "total_procurement_spend_eur": Decimal("180000000"),
        "screening": {
            "eeio_model": "EXIOBASE_3",
            "currency": "EUR",
            "significance_threshold_pct": 1.0,
        },
        "spend_classification": {
            "code_system": "NAICS",
            "confidence_threshold": 0.80,
        },
        "double_counting": {
            "rules_enabled": OVERLAP_RULES,
            "allocation_method": "ECONOMIC",
            "conservative_mode": True,
        },
        "uncertainty": {
            "method": "MONTE_CARLO",
            "monte_carlo_iterations": 10000,
            "confidence_level": 0.95,
            "seed": 42,
        },
        "compliance": {
            "target_frameworks": ["GHG_PROTOCOL", "ESRS_E1", "CDP", "SBTI"],
        },
        "reporting": {
            "formats": ["MARKDOWN", "HTML", "JSON"],
            "include_provenance": True,
        },
    }


@pytest.fixture
def manufacturing_pack_config(sample_pack_config) -> Dict[str, Any]:
    """Manufacturing-specific pack config with additional overrides."""
    config = dict(sample_pack_config)
    config["sector_type"] = "MANUFACTURING"
    config["categories_enabled"] = [
        "CAT_1", "CAT_2", "CAT_3", "CAT_4", "CAT_5",
        "CAT_6", "CAT_7", "CAT_9", "CAT_11", "CAT_12",
    ]
    return config


@pytest.fixture
def sme_pack_config(sample_pack_config) -> Dict[str, Any]:
    """SME simplified pack config."""
    config = dict(sample_pack_config)
    config["sector_type"] = "SME"
    config["revenue_meur"] = Decimal("15.0")
    config["employees_fte"] = 45
    config["total_procurement_spend_eur"] = Decimal("5000000")
    config["categories_enabled"] = ["CAT_1", "CAT_3", "CAT_5", "CAT_6", "CAT_7"]
    config["uncertainty"]["monte_carlo_iterations"] = 3000
    return config


# =============================================================================
# 15. Edge Case Fixtures
# =============================================================================


@pytest.fixture
def empty_spend_data() -> List[Dict[str, Any]]:
    """Empty spend data list for edge case testing."""
    return []


@pytest.fixture
def minimal_org() -> Dict[str, Any]:
    """Minimal organization profile with only required fields."""
    return {
        "org_id": "ORG-MIN-001",
        "org_name": "Minimal Corp",
        "sector": "SERVICES",
        "country": "US",
        "reporting_year": 2025,
    }


@pytest.fixture
def single_category_results() -> Dict[str, Any]:
    """Results with only a single category having emissions."""
    return {
        "reporting_year": 2025,
        "org_id": "ORG-SINGLE-001",
        "categories": {
            cat: {
                "total_tco2e": Decimal("50000") if cat == "CAT_1" else Decimal("0"),
                "methodology": "SPEND_BASED" if cat == "CAT_1" else "NOT_APPLICABLE",
                "dqr": Decimal("4.0") if cat == "CAT_1" else Decimal("0"),
                "by_gas": {
                    "CO2": Decimal("49000") if cat == "CAT_1" else Decimal("0"),
                    "CH4": Decimal("600") if cat == "CAT_1" else Decimal("0"),
                    "N2O": Decimal("400") if cat == "CAT_1" else Decimal("0"),
                },
                "uncertainty_pct": Decimal("50") if cat == "CAT_1" else Decimal("0"),
                "source_count": 100 if cat == "CAT_1" else 0,
                "spend_eur": Decimal("20000000") if cat == "CAT_1" else Decimal("0"),
            }
            for cat in SCOPE3_CATEGORIES
        },
        "total_scope3_tco2e": Decimal("50000"),
    }


@pytest.fixture
def zero_revenue_org() -> Dict[str, Any]:
    """Organization profile with zero revenue (pre-revenue startup)."""
    return {
        "org_id": "ORG-ZERO-001",
        "org_name": "PreRevenue StartupCo",
        "sector": "TECHNOLOGY",
        "country": "US",
        "reporting_year": 2025,
        "revenue_meur": Decimal("0"),
        "employees_fte": 12,
        "total_procurement_spend_eur": Decimal("500000"),
        "number_of_suppliers": 15,
    }


# =============================================================================
# Composite / Convenience Fixtures
# =============================================================================


@pytest.fixture
def full_scope3_context(
    manufacturing_org,
    sample_spend_data,
    sample_category_results,
    sample_supplier_data,
    sample_eeio_factors,
    sample_screening_results,
    sample_consolidated_inventory,
    sample_double_counting_results,
    sample_hotspot_analysis,
    sample_data_quality,
    sample_uncertainty_results,
    sample_compliance_results,
    sample_pack_config,
) -> Dict[str, Any]:
    """Aggregate fixture combining all data for integration/e2e tests."""
    return {
        "organization": manufacturing_org,
        "spend_data": sample_spend_data,
        "category_results": sample_category_results,
        "supplier_data": sample_supplier_data,
        "eeio_factors": sample_eeio_factors,
        "screening_results": sample_screening_results,
        "consolidated_inventory": sample_consolidated_inventory,
        "double_counting": sample_double_counting_results,
        "hotspot_analysis": sample_hotspot_analysis,
        "data_quality": sample_data_quality,
        "uncertainty": sample_uncertainty_results,
        "compliance": sample_compliance_results,
        "pack_config": sample_pack_config,
    }
