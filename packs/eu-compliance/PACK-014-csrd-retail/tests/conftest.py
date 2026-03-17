# -*- coding: utf-8 -*-
"""
PACK-014 CSRD Retail & Consumer Goods Pack - Shared Test Fixtures (conftest.py)
================================================================================

Provides pytest fixtures for the PACK-014 test suite including:
  - Dynamic module loading via importlib (no package install needed)
  - Pack manifest and configuration fixtures
  - Sample store, energy, refrigerant, packaging, product, food waste,
    supplier, material flow, and benchmark KPI data
  - Demo configuration loading

All fixtures use importlib.util.spec_from_file_location to load modules
directly from the pack source tree, enabling test execution without
installing the pack as a Python package.

Fixture Categories:
  1. Paths and YAML data
  2. Configuration objects
  3. Store data (grocery, apparel, online warehouse)
  4. Energy consumption data
  5. Refrigerant data
  6. Packaging data
  7. Product data
  8. Food waste data
  9. Supplier data
  10. Material flow data
  11. Benchmark KPI data

Author:  GreenLang Platform Team (GL-TestEngineer)
Pack:    PACK-014 CSRD Retail & Consumer Goods
Date:    March 2026
"""

import importlib
import importlib.util
import sys
from decimal import Decimal
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest
import yaml


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
DEMO_DIR = CONFIG_DIR / "demo"

# Engine file mapping: logical name -> file name on disk
ENGINE_FILES = {
    "store_emissions": "store_emissions_engine.py",
    "retail_scope3": "retail_scope3_engine.py",
    "packaging_compliance": "packaging_compliance_engine.py",
    "product_sustainability": "product_sustainability_engine.py",
    "food_waste": "food_waste_engine.py",
    "supply_chain_due_diligence": "supply_chain_due_diligence_engine.py",
    "retail_circular_economy": "retail_circular_economy_engine.py",
    "retail_benchmark": "retail_benchmark_engine.py",
}

# Engine class names that should exist in each engine module
ENGINE_CLASSES = {
    "store_emissions": "StoreEmissionsEngine",
    "retail_scope3": "RetailScope3Engine",
    "packaging_compliance": "PackagingComplianceEngine",
    "product_sustainability": "ProductSustainabilityEngine",
    "food_waste": "FoodWasteEngine",
    "supply_chain_due_diligence": "SupplyChainDueDiligenceEngine",
    "retail_circular_economy": "RetailCircularEconomyEngine",
    "retail_benchmark": "RetailBenchmarkEngine",
}

# Workflow file mapping
WORKFLOW_FILES = {
    "store_emissions": "store_emissions_workflow.py",
    "supply_chain_assessment": "supply_chain_assessment_workflow.py",
    "packaging_compliance": "packaging_compliance_workflow.py",
    "product_sustainability": "product_sustainability_workflow.py",
    "food_waste_tracking": "food_waste_tracking_workflow.py",
    "circular_economy": "circular_economy_workflow.py",
    "esrs_retail_disclosure": "esrs_retail_disclosure_workflow.py",
    "regulatory_compliance": "regulatory_compliance_workflow.py",
}

# Template file mapping
TEMPLATE_FILES = {
    "store_emissions_report": "store_emissions_report.py",
    "supply_chain_report": "supply_chain_report.py",
    "packaging_compliance_report": "packaging_compliance_report.py",
    "product_sustainability_report": "product_sustainability_report.py",
    "food_waste_report": "food_waste_report.py",
    "circular_economy_report": "circular_economy_report.py",
    "retail_esg_scorecard": "retail_esg_scorecard.py",
    "esrs_retail_disclosure": "esrs_retail_disclosure.py",
}

# Integration file mapping
INTEGRATION_FILES = {
    "pack_orchestrator": "pack_orchestrator.py",
    "csrd_pack_bridge": "csrd_pack_bridge.py",
    "mrv_retail_bridge": "mrv_retail_bridge.py",
    "data_retail_bridge": "data_retail_bridge.py",
    "eudr_retail_bridge": "eudr_retail_bridge.py",
    "circular_economy_bridge": "circular_economy_bridge.py",
    "supply_chain_bridge": "supply_chain_bridge.py",
    "taxonomy_bridge": "taxonomy_bridge.py",
    "health_check": "health_check.py",
    "setup_wizard": "setup_wizard.py",
}

# Preset names
PRESET_NAMES = [
    "grocery_retail",
    "apparel_retail",
    "electronics_retail",
    "general_retail",
    "online_retail",
    "sme_retailer",
]


# =============================================================================
# Helper: Dynamic Module Loader
# =============================================================================


def _load_module(module_name: str, file_name: str, subdir: str = "engines"):
    """Load a module dynamically using importlib.util.spec_from_file_location.

    This avoids the need to install PACK-014 as a Python package. The module
    is loaded from the pack source tree and added to sys.modules under a
    unique key to prevent collisions.

    Args:
        module_name: Logical name for the module (used as sys.modules key prefix).
        file_name: File name of the Python module (e.g., "store_emissions_engine.py").
        subdir: Subdirectory under PACK_ROOT ("engines", "workflows", "templates",
                "integrations", or "config").

    Returns:
        The loaded module object.

    Raises:
        FileNotFoundError: If the module file does not exist.
        ImportError: If the module cannot be loaded.
    """
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
            f"Ensure PACK-014 source files are present."
        )

    # Create a unique module key to avoid collisions
    full_module_name = f"pack014_test.{subdir}.{module_name}"

    # Return cached module if already loaded
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
        # Remove from sys.modules on failure to allow retry
        sys.modules.pop(full_module_name, None)
        raise ImportError(
            f"Failed to load module {full_module_name} from {file_path}: {exc}"
        ) from exc

    return module


def _load_engine(engine_key: str):
    """Load an engine module by its logical key.

    Args:
        engine_key: Engine key from ENGINE_FILES (e.g., "store_emissions").

    Returns:
        The loaded engine module.
    """
    file_name = ENGINE_FILES[engine_key]
    return _load_module(engine_key, file_name, "engines")


def _load_config_module():
    """Load the pack_config module.

    Returns:
        The loaded pack_config module.
    """
    return _load_module("pack_config", "pack_config.py", "config")


# =============================================================================
# 1. Path and YAML Fixtures
# =============================================================================


@pytest.fixture(scope="session")
def pack_root() -> Path:
    """Return the absolute path to the PACK-014 root directory."""
    return PACK_ROOT


@pytest.fixture(scope="session")
def pack_yaml_path() -> Path:
    """Return the absolute path to pack.yaml."""
    return PACK_ROOT / "pack.yaml"


@pytest.fixture(scope="session")
def pack_yaml_data(pack_yaml_path: Path) -> Dict[str, Any]:
    """Parse and return the pack.yaml manifest as a dictionary."""
    with open(pack_yaml_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    assert data is not None, "pack.yaml parsed to None"
    return data


@pytest.fixture(scope="session")
def demo_yaml_path() -> Path:
    """Return the absolute path to the demo configuration YAML."""
    return DEMO_DIR / "demo_config.yaml"


@pytest.fixture(scope="session")
def demo_yaml_data(demo_yaml_path: Path) -> Dict[str, Any]:
    """Parse and return the demo_config.yaml as a dictionary."""
    with open(demo_yaml_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    assert data is not None, "demo_config.yaml parsed to None"
    return data


# =============================================================================
# 2. Configuration Fixtures
# =============================================================================


@pytest.fixture(scope="session")
def config_module():
    """Load and return the pack_config module."""
    return _load_config_module()


@pytest.fixture
def pack_config(config_module):
    """Create a CSRDRetailConfig with default values."""
    return config_module.CSRDRetailConfig()


@pytest.fixture
def demo_config(config_module, demo_yaml_data):
    """Create a CSRDRetailConfig loaded from the demo YAML data.

    Only passes known fields to the constructor, ignoring demo-only
    keys like stores, omnibus_threshold, etc. that are not part of
    the CSRDRetailConfig model.
    """
    known_fields = {
        "company_name", "reporting_year", "tier", "sub_sectors",
        "store_emissions", "scope3", "packaging", "product_sustainability",
        "food_waste", "supply_chain_dd", "circular_economy", "benchmark",
    }
    filtered = {k: v for k, v in demo_yaml_data.items() if k in known_fields}
    return config_module.CSRDRetailConfig(**filtered)


@pytest.fixture
def pack_config_wrapper(config_module):
    """Create a PackConfig wrapper with default values."""
    return config_module.PackConfig()


# =============================================================================
# 3. Store Data Fixtures
# =============================================================================


@pytest.fixture
def sample_grocery_store():
    """Create a sample StoreData for a Berlin grocery store.

    Facility: Berlin Flagship Store (Germany)
    Floor area: 5,000 sqm
    Refrigerant: R404A (500 kg charge, 15% leakage rate)
    Energy: 2,000 MWh electricity + 800 MWh natural gas + 200 MWh district heating
    Fleet: 5 delivery vans, 3 diesel forklifts
    """
    mod = _load_engine("store_emissions")
    return mod.StoreData(
        store_id="STORE-001-DE",
        store_name="Berlin Flagship Store",
        store_type=mod.StoreType.FLAGSHIP,
        country="DE",
        region="Berlin",
        floor_area_sqm=5000.0,
        employees=120,
        operating_hours_per_year=4380,
        energy_consumption=[
            mod.EnergyConsumption(
                source=mod.EnergySource.ELECTRICITY,
                quantity_kwh=2000000.0,
                cost_eur=320000.0,
                renewable_pct=10.0,
            ),
            mod.EnergyConsumption(
                source=mod.EnergySource.NATURAL_GAS,
                quantity_kwh=800000.0,
                cost_eur=56000.0,
            ),
            mod.EnergyConsumption(
                source=mod.EnergySource.DISTRICT_HEATING,
                quantity_kwh=200000.0,
                cost_eur=18000.0,
            ),
        ],
        refrigerants=[
            mod.RefrigerantData(
                refrigerant_type=mod.RefrigerantType.R404A,
                charge_kg=500.0,
                leakage_rate_pct=15.0,
            ),
        ],
        fleet=[
            mod.FleetData(
                vehicle_type=mod.FleetVehicleType.DELIVERY_VAN,
                count=5,
                fuel_consumption_litres=25000.0,
                distance_km=120000.0,
            ),
            mod.FleetData(
                vehicle_type=mod.FleetVehicleType.FORKLIFT_DIESEL,
                count=3,
                fuel_consumption_litres=8000.0,
                distance_km=5000.0,
            ),
        ],
    )


@pytest.fixture
def sample_apparel_store():
    """Create a sample StoreData for a Lyon apparel boutique.

    Facility: Lyon Fashion Boutique (France)
    Floor area: 800 sqm
    No refrigeration, no fleet, standard energy profile.
    """
    mod = _load_engine("store_emissions")
    return mod.StoreData(
        store_id="STORE-002-FR",
        store_name="Lyon Fashion Boutique",
        store_type=mod.StoreType.STANDARD,
        country="FR",
        region="Auvergne-Rhone-Alpes",
        floor_area_sqm=800.0,
        employees=25,
        operating_hours_per_year=3120,
        energy_consumption=[
            mod.EnergyConsumption(
                source=mod.EnergySource.ELECTRICITY,
                quantity_kwh=160000.0,
                cost_eur=25600.0,
                renewable_pct=0.0,
            ),
            mod.EnergyConsumption(
                source=mod.EnergySource.NATURAL_GAS,
                quantity_kwh=80000.0,
                cost_eur=5600.0,
            ),
        ],
        refrigerants=[],
        fleet=[],
    )


@pytest.fixture
def sample_online_warehouse():
    """Create a sample StoreData for an Amsterdam online warehouse.

    Facility: Amsterdam Online Hub (Netherlands)
    Floor area: 15,000 sqm (warehouse)
    Refrigerant: R744 (CO2, low GWP)
    Fleet: 20 electric vans, 10 cargo bikes
    """
    mod = _load_engine("store_emissions")
    return mod.StoreData(
        store_id="STORE-003-NL",
        store_name="Amsterdam Online Hub",
        store_type=mod.StoreType.WAREHOUSE,
        country="NL",
        region="North Holland",
        floor_area_sqm=15000.0,
        employees=200,
        operating_hours_per_year=6570,
        energy_consumption=[
            mod.EnergyConsumption(
                source=mod.EnergySource.ELECTRICITY,
                quantity_kwh=5000000.0,
                cost_eur=750000.0,
                renewable_pct=50.0,
            ),
        ],
        refrigerants=[
            mod.RefrigerantData(
                refrigerant_type=mod.RefrigerantType.R744_CO2,
                charge_kg=200.0,
                leakage_rate_pct=10.0,
            ),
        ],
        fleet=[
            mod.FleetData(
                vehicle_type=mod.FleetVehicleType.ELECTRIC_VAN,
                count=20,
                fuel_consumption_litres=0.0,
                distance_km=400000.0,
            ),
            mod.FleetData(
                vehicle_type=mod.FleetVehicleType.CARGO_BIKE,
                count=10,
                fuel_consumption_litres=0.0,
                distance_km=60000.0,
            ),
        ],
    )


# =============================================================================
# 4. Energy Consumption Data Fixtures
# =============================================================================


@pytest.fixture
def sample_energy_consumption():
    """Create sample energy consumption records (electricity + gas + district heating)."""
    mod = _load_engine("store_emissions")
    return [
        mod.EnergyConsumption(
            source=mod.EnergySource.ELECTRICITY,
            quantity_kwh=2000000.0,
            cost_eur=320000.0,
            renewable_pct=10.0,
        ),
        mod.EnergyConsumption(
            source=mod.EnergySource.NATURAL_GAS,
            quantity_kwh=800000.0,
            cost_eur=56000.0,
        ),
        mod.EnergyConsumption(
            source=mod.EnergySource.DISTRICT_HEATING,
            quantity_kwh=200000.0,
            cost_eur=18000.0,
        ),
    ]


# =============================================================================
# 5. Refrigerant Data Fixtures
# =============================================================================


@pytest.fixture
def sample_refrigerant_data():
    """Create sample refrigerant data (R404A, 500kg charge, 15% leakage)."""
    mod = _load_engine("store_emissions")
    return mod.RefrigerantData(
        refrigerant_type=mod.RefrigerantType.R404A,
        charge_kg=500.0,
        leakage_rate_pct=15.0,
    )


# =============================================================================
# 6. Packaging Data Fixtures
# =============================================================================


@pytest.fixture
def sample_packaging_items():
    """Create sample packaging items (PET bottles, cardboard boxes, plastic bags, glass jars)."""
    mod = _load_engine("packaging_compliance")
    return [
        mod.PackagingItem(
            item_id="PKG-001",
            item_name="PET Water Bottle 500ml",
            material=mod.PackagingMaterial.PET,
            packaging_type=mod.PackagingType.PRIMARY,
            weight_grams=25.0,
            units_placed=5000000,
            recycled_content_pct=35.0,
            is_contact_sensitive=True,
            recyclability_grade=mod.EPRGrade.A,
            has_material_marking=True,
            has_sorting_instructions=True,
        ),
        mod.PackagingItem(
            item_id="PKG-002",
            item_name="Cardboard Shipping Box",
            material=mod.PackagingMaterial.PAPER_BOARD,
            packaging_type=mod.PackagingType.SECONDARY,
            weight_grams=350.0,
            units_placed=2000000,
            recycled_content_pct=80.0,
            is_contact_sensitive=False,
            recyclability_grade=mod.EPRGrade.A,
            has_material_marking=True,
            has_sorting_instructions=True,
        ),
        mod.PackagingItem(
            item_id="PKG-003",
            item_name="Plastic Shopping Bag",
            material=mod.PackagingMaterial.HDPE,
            packaging_type=mod.PackagingType.PRIMARY,
            weight_grams=12.0,
            units_placed=10000000,
            recycled_content_pct=0.0,
            is_contact_sensitive=False,
            recyclability_grade=mod.EPRGrade.D,
            has_material_marking=False,
            has_sorting_instructions=False,
        ),
        mod.PackagingItem(
            item_id="PKG-004",
            item_name="Glass Jam Jar 250ml",
            material=mod.PackagingMaterial.GLASS,
            packaging_type=mod.PackagingType.PRIMARY,
            weight_grams=200.0,
            units_placed=500000,
            recycled_content_pct=60.0,
            is_contact_sensitive=True,
            recyclability_grade=mod.EPRGrade.A,
            has_material_marking=True,
            has_sorting_instructions=True,
        ),
    ]


# =============================================================================
# 7. Product Data Fixtures
# =============================================================================


@pytest.fixture
def sample_products():
    """Create sample product data (cotton t-shirt, smartphone, wooden chair)."""
    mod = _load_engine("product_sustainability")
    return [
        mod.ProductData(
            product_id="PROD-001",
            product_name="Organic Cotton T-Shirt",
            category="apparel",
            dpp_category=mod.DPPCategory.TEXTILES,
            weight_kg=0.2,
            price_eur=29.99,
            country_of_manufacture="BD",
        ),
        mod.ProductData(
            product_id="PROD-002",
            product_name="Smartphone X12",
            category="electronics",
            dpp_category=mod.DPPCategory.ELECTRONICS,
            weight_kg=0.18,
            price_eur=799.99,
            country_of_manufacture="CN",
        ),
        mod.ProductData(
            product_id="PROD-003",
            product_name="Oak Dining Chair",
            category="furniture",
            dpp_category=mod.DPPCategory.FURNITURE,
            weight_kg=8.5,
            price_eur=249.00,
            country_of_manufacture="PL",
        ),
    ]


# =============================================================================
# 8. Food Waste Data Fixtures
# =============================================================================


@pytest.fixture
def sample_food_waste_records():
    """Create sample food waste records (bakery, produce, dairy, meat waste)."""
    mod = _load_engine("food_waste")
    return [
        mod.FoodWasteRecord(
            store_id="STORE-001-DE",
            category=mod.FoodWasteCategory.BAKERY,
            quantity_kg=1200.0,
            destination=mod.WasteDestination.REDISTRIBUTION,
            measurement_method=mod.MeasurementMethod.DIRECT_WEIGHING,
            reporting_period="2025-Q1",
            value_eur=4200.0,
        ),
        mod.FoodWasteRecord(
            store_id="STORE-001-DE",
            category=mod.FoodWasteCategory.PRODUCE,
            quantity_kg=800.0,
            destination=mod.WasteDestination.COMPOSTING,
            measurement_method=mod.MeasurementMethod.DIRECT_WEIGHING,
            reporting_period="2025-Q1",
            value_eur=1760.0,
        ),
        mod.FoodWasteRecord(
            store_id="STORE-001-DE",
            category=mod.FoodWasteCategory.DAIRY,
            quantity_kg=350.0,
            destination=mod.WasteDestination.ANAEROBIC_DIGESTION,
            measurement_method=mod.MeasurementMethod.SCANNING_DATA,
            reporting_period="2025-Q1",
            value_eur=1680.0,
        ),
        mod.FoodWasteRecord(
            store_id="STORE-001-DE",
            category=mod.FoodWasteCategory.MEAT_POULTRY,
            quantity_kg=150.0,
            destination=mod.WasteDestination.INCINERATION_ENERGY,
            measurement_method=mod.MeasurementMethod.DIRECT_WEIGHING,
            reporting_period="2025-Q1",
            value_eur=1275.0,
        ),
    ]


# =============================================================================
# 9. Supplier Data Fixtures
# =============================================================================


@pytest.fixture
def sample_suppliers():
    """Create sample supplier profiles (Tier 1 Bangladesh garment, Tier 1 Colombia coffee, Tier 2 China electronics)."""
    mod = _load_engine("supply_chain_due_diligence")
    return [
        mod.SupplierProfile(
            supplier_id="SUP-001",
            name="Dhaka Garments Ltd",
            country="BD",
            sector="garments_textiles",
            tier=mod.SupplierTier.TIER_1,
            commodities_supplied=[],
            employee_count=5000,
            annual_spend_eur=2500000.0,
            incident_count=2,
            forced_labour_indicators_present=[
                "excessive_overtime",
                "withholding_wages",
            ],
        ),
        mod.SupplierProfile(
            supplier_id="SUP-002",
            name="Colombian Coffee Estates",
            country="CO",
            sector="coffee_production",
            tier=mod.SupplierTier.TIER_1,
            commodities_supplied=[mod.EUDRCommodity.COFFEE],
            employee_count=300,
            annual_spend_eur=800000.0,
            incident_count=0,
            forced_labour_indicators_present=[],
        ),
        mod.SupplierProfile(
            supplier_id="SUP-003",
            name="Shenzhen Electronics Co",
            country="CN",
            sector="electronics",
            tier=mod.SupplierTier.TIER_2,
            commodities_supplied=[],
            employee_count=15000,
            annual_spend_eur=5000000.0,
            incident_count=1,
            forced_labour_indicators_present=[],
        ),
    ]


# =============================================================================
# 10. Material Flow Data Fixtures
# =============================================================================


@pytest.fixture
def sample_material_flows():
    """Create sample material flow data (cardboard, plastic, glass, textile)."""
    mod = _load_engine("retail_circular_economy")
    return [
        mod.MaterialFlow(
            material="cardboard",
            virgin_input_tonnes=500.0,
            recycled_input_tonnes=200.0,
            waste_output_tonnes=680.0,
            recovery_tonnes=600.0,
            product_mass_tonnes=700.0,
            product_lifetime_years=0.5,
            industry_avg_lifetime_years=0.5,
        ),
        mod.MaterialFlow(
            material="plastic",
            virgin_input_tonnes=300.0,
            recycled_input_tonnes=50.0,
            waste_output_tonnes=320.0,
            recovery_tonnes=120.0,
            product_mass_tonnes=350.0,
            product_lifetime_years=1.0,
            industry_avg_lifetime_years=1.0,
        ),
        mod.MaterialFlow(
            material="glass",
            virgin_input_tonnes=100.0,
            recycled_input_tonnes=80.0,
            waste_output_tonnes=170.0,
            recovery_tonnes=150.0,
            product_mass_tonnes=180.0,
            product_lifetime_years=1.0,
            industry_avg_lifetime_years=1.0,
        ),
        mod.MaterialFlow(
            material="textile",
            virgin_input_tonnes=200.0,
            recycled_input_tonnes=30.0,
            waste_output_tonnes=180.0,
            recovery_tonnes=60.0,
            product_mass_tonnes=230.0,
            product_lifetime_years=3.0,
            industry_avg_lifetime_years=2.0,
        ),
    ]


# =============================================================================
# 11. Benchmark KPI Data Fixtures
# =============================================================================


@pytest.fixture
def sample_retail_kpis():
    """Create sample retail KPIs for benchmarking.

    Company: Mid-size grocery retailer (50 stores, EUR 500M revenue)
    """
    mod = _load_engine("retail_benchmark")
    return mod.RetailKPIs(
        facility_id="EURORETAIL-001",
        sub_sector=mod.RetailSubSector.GROCERY,
        store_count=50,
        total_floor_area_sqm=150000.0,
        revenue_eur=500000000.0,
        employees=3500,
        total_emissions_tco2e=250000.0,
        scope1_tco2e=15000.0,
        scope2_tco2e=10000.0,
        scope3_tco2e=225000.0,
        energy_consumption_kwh=80000000.0,
        renewable_energy_pct=35.0,
        waste_diversion_pct=72.0,
        food_waste_pct=2.1,
        packaging_recycled_content_pct=38.0,
        supplier_engagement_pct=55.0,
    )
