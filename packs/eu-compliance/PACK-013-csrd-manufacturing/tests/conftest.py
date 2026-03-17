# -*- coding: utf-8 -*-
"""
PACK-013 CSRD Manufacturing Pack - Shared Test Fixtures (conftest.py)
=====================================================================

Provides pytest fixtures for the PACK-013 test suite including:
  - Dynamic module loading via importlib (no package install needed)
  - Pack manifest and configuration fixtures
  - Sample manufacturing facility data
  - Sample energy, production, material, waste, water, and pollutant data
  - Sample supplier, BOM, and benchmark KPI data
  - Demo configuration loading

All fixtures use importlib.util.spec_from_file_location to load modules
directly from the pack source tree, enabling test execution without
installing the pack as a Python package.

Fixture Categories:
  1. Paths and YAML data
  2. Configuration objects
  3. Facility data (cement, automotive, chemical)
  4. Energy and production data
  5. Product and BOM data
  6. Material flow and waste stream data
  7. Water intake and discharge data
  8. Pollutant emission data
  9. BAT compliance data
  10. Supply chain data
  11. Benchmark KPI data

Author:  GreenLang Platform Team (GL-TestEngineer)
Pack:    PACK-013 CSRD Manufacturing
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
    "process_emissions": "process_emissions_engine.py",
    "energy_intensity": "energy_intensity_engine.py",
    "product_carbon_footprint": "product_carbon_footprint_engine.py",
    "circular_economy": "circular_economy_engine.py",
    "water_pollution": "water_pollution_engine.py",
    "bat_compliance": "bat_compliance_engine.py",
    "supply_chain_emissions": "supply_chain_emissions_engine.py",
    "manufacturing_benchmark": "manufacturing_benchmark_engine.py",
}

# Engine class names that should exist in each engine module
ENGINE_CLASSES = {
    "process_emissions": "ProcessEmissionsEngine",
    "energy_intensity": "EnergyIntensityEngine",
    "product_carbon_footprint": "ProductCarbonFootprintEngine",
    "circular_economy": "CircularEconomyEngine",
    "water_pollution": "WaterPollutionEngine",
    "bat_compliance": "BATComplianceEngine",
    "supply_chain_emissions": "SupplyChainEmissionsEngine",
    "manufacturing_benchmark": "ManufacturingBenchmarkEngine",
}

# Workflow file mapping
WORKFLOW_FILES = {
    "manufacturing_emissions": "manufacturing_emissions_workflow.py",
    "product_pcf": "product_pcf_workflow.py",
    "circular_economy": "circular_economy_workflow.py",
    "bat_compliance": "bat_compliance_workflow.py",
    "supply_chain_assessment": "supply_chain_assessment_workflow.py",
    "esrs_manufacturing": "esrs_manufacturing_workflow.py",
    "decarbonization_roadmap": "decarbonization_roadmap_workflow.py",
    "regulatory_compliance": "regulatory_compliance_workflow.py",
}

# Template file mapping
TEMPLATE_FILES = {
    "process_emissions_report": "process_emissions_report.py",
    "product_pcf_label": "product_pcf_label.py",
    "energy_performance_report": "energy_performance_report.py",
    "circular_economy_report": "circular_economy_report.py",
    "bat_compliance_report": "bat_compliance_report.py",
    "water_pollution_report": "water_pollution_report.py",
    "manufacturing_scorecard": "manufacturing_scorecard.py",
    "decarbonization_roadmap": "decarbonization_roadmap.py",
}

# Integration file mapping
INTEGRATION_FILES = {
    "pack_orchestrator": "pack_orchestrator.py",
    "csrd_pack_bridge": "csrd_pack_bridge.py",
    "cbam_pack_bridge": "cbam_pack_bridge.py",
    "mrv_industrial_bridge": "mrv_industrial_bridge.py",
    "data_manufacturing_bridge": "data_manufacturing_bridge.py",
    "eu_ets_bridge": "eu_ets_bridge.py",
    "taxonomy_bridge": "taxonomy_bridge.py",
    "health_check": "health_check.py",
    "setup_wizard": "setup_wizard.py",
}

# Preset names
PRESET_NAMES = [
    "heavy_industry",
    "discrete_manufacturing",
    "process_manufacturing",
    "light_manufacturing",
    "multi_site",
    "sme_manufacturer",
]


# =============================================================================
# Helper: Dynamic Module Loader
# =============================================================================


def _load_module(module_name: str, file_name: str, subdir: str = "engines"):
    """Load a module dynamically using importlib.util.spec_from_file_location.

    This avoids the need to install PACK-013 as a Python package. The module
    is loaded from the pack source tree and added to sys.modules under a
    unique key to prevent collisions.

    Args:
        module_name: Logical name for the module (used as sys.modules key prefix).
        file_name: File name of the Python module (e.g., "process_emissions_engine.py").
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
            f"Ensure PACK-013 source files are present."
        )

    # Create a unique module key to avoid collisions
    full_module_name = f"pack013_test.{subdir}.{module_name}"

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
        engine_key: Engine key from ENGINE_FILES (e.g., "process_emissions").

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
    """Return the absolute path to the PACK-013 root directory."""
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
    """Create a CSRDManufacturingConfig with default values."""
    return config_module.CSRDManufacturingConfig()


@pytest.fixture
def demo_config(config_module, demo_yaml_data):
    """Create a CSRDManufacturingConfig loaded from the demo YAML."""
    return config_module.CSRDManufacturingConfig(**demo_yaml_data)


@pytest.fixture
def pack_config_wrapper(config_module):
    """Create a PackConfig wrapper with default values."""
    return config_module.PackConfig()


# =============================================================================
# 3. Facility Data Fixtures
# =============================================================================


@pytest.fixture
def sample_facility():
    """Create a sample FacilityData for a cement plant.

    Facility: Rhein-Main Cement Plant (Germany)
    Process: CaCO3 calcination, natural gas-fired kiln
    Production: 500,000 tonnes/year of clinker
    """
    mod = _load_engine("process_emissions")
    return mod.FacilityData(
        facility_id="FAC-TEST-001",
        facility_name="Test Cement Plant",
        sub_sector=mod.ManufacturingSubSector.CEMENT,
        country="DE",
        eu_ets_installation_id="DE-ETS-2024-TEST01",
        production_lines=[
            mod.ProcessLine(
                line_id="KILN-1",
                line_name="Rotary Kiln 1",
                process_type=mod.ProcessType.CALCINATION,
                annual_production_tonnes=500000.0,
                raw_materials=[
                    mod.RawMaterial(
                        material_name="Limestone (CaCO3)",
                        quantity_tonnes=800000.0,
                        co2_factor_per_tonne=0.525,
                        source="IPCC 2006 Vol.3 - Table 2.1",
                    ),
                ],
                fuel_consumption=[
                    mod.FuelConsumption(
                        fuel_type=mod.FuelType.NATURAL_GAS,
                        quantity=15000.0,
                        unit="1000m3",
                    ),
                    mod.FuelConsumption(
                        fuel_type=mod.FuelType.COAL,
                        quantity=50000.0,
                        unit="tonnes",
                    ),
                ],
            ),
        ],
    )


@pytest.fixture
def sample_automotive_facility():
    """Create a sample FacilityData for an automotive assembly plant.

    Facility: Pilsen Automotive Assembly (Czech Republic)
    Process: Painting, welding, casting
    Production: 80,000 vehicles/year
    """
    mod = _load_engine("process_emissions")
    return mod.FacilityData(
        facility_id="FAC-TEST-002",
        facility_name="Test Automotive Assembly",
        sub_sector=mod.ManufacturingSubSector.AUTOMOTIVE,
        country="CZ",
        production_lines=[
            mod.ProcessLine(
                line_id="PAINT-1",
                line_name="Paint Shop Line 1",
                process_type=mod.ProcessType.COMBUSTION,
                annual_production_tonnes=100000.0,
                raw_materials=[
                    mod.RawMaterial(
                        material_name="Paint VOC emissions",
                        quantity_tonnes=80000.0,
                        co2_factor_per_tonne=0.025,
                        source="STM BREF - automotive painting",
                    ),
                ],
                fuel_consumption=[
                    mod.FuelConsumption(
                        fuel_type=mod.FuelType.NATURAL_GAS,
                        quantity=5000.0,
                        unit="1000m3",
                    ),
                ],
            ),
        ],
    )


@pytest.fixture
def sample_chemical_facility():
    """Create a sample FacilityData for a chemical processing plant.

    Facility: Rotterdam Chemical Processing (Netherlands)
    Process: Ammonia synthesis via steam methane reforming
    Production: 200,000 tonnes/year NH3
    """
    mod = _load_engine("process_emissions")
    return mod.FacilityData(
        facility_id="FAC-TEST-003",
        facility_name="Test Chemical Plant",
        sub_sector=mod.ManufacturingSubSector.CHEMICALS,
        country="NL",
        eu_ets_installation_id="NL-ETS-2024-TEST03",
        production_lines=[
            mod.ProcessLine(
                line_id="REACTOR-A",
                line_name="Ammonia Reactor A",
                process_type=mod.ProcessType.SYNTHESIS,
                annual_production_tonnes=200000.0,
                raw_materials=[
                    mod.RawMaterial(
                        material_name="Ammonia synthesis",
                        quantity_tonnes=200000.0,
                        co2_factor_per_tonne=1.600,
                        source="IPCC 2006 Vol.3 - ammonia",
                    ),
                ],
                fuel_consumption=[
                    mod.FuelConsumption(
                        fuel_type=mod.FuelType.NATURAL_GAS,
                        quantity=80000.0,
                        unit="1000m3",
                    ),
                ],
            ),
        ],
    )


# =============================================================================
# 4. Energy and Production Data Fixtures
# =============================================================================


@pytest.fixture
def sample_energy_data():
    """Create a FacilityEnergyData with multiple energy sources.

    Facility: Test Cement Plant with electricity, natural gas, coal, biomass.
    """
    mod = _load_engine("energy_intensity")
    return mod.FacilityEnergyData(
        facility_id="FAC-TEST-001",
        facility_name="Test Cement Plant",
        sub_sector="cement",
        energy_consumption=[
            mod.EnergyConsumptionData(
                source=mod.EnergySource.ELECTRICITY,
                quantity_mwh=45000.0,
                cost_eur=4500000.0,
                renewable_pct=25.0,
                emission_factor_tco2_per_mwh=0.401,
            ),
            mod.EnergyConsumptionData(
                source=mod.EnergySource.NATURAL_GAS,
                quantity_mwh=120000.0,
                cost_eur=6000000.0,
                renewable_pct=0.0,
                emission_factor_tco2_per_mwh=0.202,
            ),
            mod.EnergyConsumptionData(
                source=mod.EnergySource.COAL,
                quantity_mwh=80000.0,
                cost_eur=3200000.0,
                renewable_pct=0.0,
                emission_factor_tco2_per_mwh=0.341,
            ),
            mod.EnergyConsumptionData(
                source=mod.EnergySource.BIOMASS,
                quantity_mwh=15000.0,
                cost_eur=900000.0,
                renewable_pct=100.0,
                emission_factor_tco2_per_mwh=0.0,
            ),
        ],
        production_volumes=[
            mod.ProductionVolumeData(
                product_name="Portland Cement CEM I",
                volume=500000.0,
                unit=mod.ProductionUnit.TONNES,
                period="2025-annual",
            ),
            mod.ProductionVolumeData(
                product_name="Blended Cement CEM II",
                volume=250000.0,
                unit=mod.ProductionUnit.TONNES,
                period="2025-annual",
            ),
        ],
        baseline_sec_mj=3800.0,
    )


@pytest.fixture
def sample_production_data():
    """Create a list of ProductionVolumeData entries."""
    mod = _load_engine("energy_intensity")
    return [
        mod.ProductionVolumeData(
            product_name="Portland Cement CEM I",
            volume=500000.0,
            unit=mod.ProductionUnit.TONNES,
            period="2025-annual",
        ),
        mod.ProductionVolumeData(
            product_name="Blended Cement CEM II",
            volume=250000.0,
            unit=mod.ProductionUnit.TONNES,
            period="2025-annual",
        ),
        mod.ProductionVolumeData(
            product_name="Specialty Cement",
            volume=50000.0,
            unit=mod.ProductionUnit.TONNES,
            period="2025-annual",
        ),
    ]


# =============================================================================
# 5. Product and BOM Data Fixtures
# =============================================================================


@pytest.fixture
def sample_product():
    """Create a ProductData for an automotive component."""
    mod = _load_engine("product_carbon_footprint")
    return mod.ProductData(
        product_id="PROD-AUTO-001",
        product_name="C-Segment Vehicle Assembly",
        functional_unit="1 vehicle",
        annual_production=80000.0,
        product_weight_kg=1450.0,
        product_category="automotive",
    )


@pytest.fixture
def sample_bom():
    """Create a List[BOMComponent] for an automotive component.

    Components: steel body, plastic interior, glass windshield, electronics.
    """
    mod = _load_engine("product_carbon_footprint")
    return [
        mod.BOMComponent(
            component_id="BOM-STEEL-01",
            component_name="Body-in-White Steel",
            material_type="steel_primary",
            quantity_per_unit=650.0,
            unit="kg",
            origin_country="DE",
            recycled_content_pct=25.0,
            data_quality_score=mod.DataQualityLevel.SCORE_2,
            supplier_name="ThyssenKrupp Steel Europe",
        ),
        mod.BOMComponent(
            component_id="BOM-PLASTIC-01",
            component_name="Interior Plastic Components",
            material_type="plastics_pp",
            quantity_per_unit=180.0,
            unit="kg",
            origin_country="DE",
            recycled_content_pct=15.0,
            data_quality_score=mod.DataQualityLevel.SCORE_3,
            supplier_name="BASF Plastics",
        ),
        mod.BOMComponent(
            component_id="BOM-GLASS-01",
            component_name="Windshield and Windows",
            material_type="glass",
            quantity_per_unit=45.0,
            unit="kg",
            origin_country="CZ",
            recycled_content_pct=30.0,
            data_quality_score=mod.DataQualityLevel.SCORE_3,
            supplier_name="AGC Glass Europe",
        ),
        mod.BOMComponent(
            component_id="BOM-ELEC-01",
            component_name="Electronics and Wiring",
            material_type="electronics_pcb",
            quantity_per_unit=35.0,
            unit="kg",
            origin_country="CN",
            recycled_content_pct=5.0,
            data_quality_score=mod.DataQualityLevel.SCORE_4,
            supplier_name="Foxconn Industrial",
        ),
    ]


# =============================================================================
# 6. Material Flow and Waste Stream Data Fixtures
# =============================================================================


@pytest.fixture
def sample_material_flows():
    """Create a List[MaterialFlowData] with virgin and recycled inputs."""
    mod = _load_engine("circular_economy")
    return [
        mod.MaterialFlowData(
            material_name="Steel",
            virgin_input_tonnes=8000.0,
            recycled_input_tonnes=4000.0,
            total_input_tonnes=12000.0,
            pre_consumer_recycled_pct=10.0,
            post_consumer_recycled_pct=23.3,
        ),
        mod.MaterialFlowData(
            material_name="Plastics (PP)",
            virgin_input_tonnes=3000.0,
            recycled_input_tonnes=500.0,
            total_input_tonnes=3500.0,
            pre_consumer_recycled_pct=5.0,
            post_consumer_recycled_pct=9.3,
        ),
        mod.MaterialFlowData(
            material_name="Glass",
            virgin_input_tonnes=1000.0,
            recycled_input_tonnes=800.0,
            total_input_tonnes=1800.0,
            pre_consumer_recycled_pct=0.0,
            post_consumer_recycled_pct=44.4,
        ),
        mod.MaterialFlowData(
            material_name="Aluminium",
            virgin_input_tonnes=500.0,
            recycled_input_tonnes=300.0,
            total_input_tonnes=800.0,
            pre_consumer_recycled_pct=10.0,
            post_consumer_recycled_pct=27.5,
        ),
    ]


@pytest.fixture
def sample_waste_streams():
    """Create a List[WasteStreamData] with metal, plastic, organic, hazardous."""
    mod = _load_engine("circular_economy")
    return [
        mod.WasteStreamData(
            waste_type=mod.WasteType.METAL_SCRAP,
            waste_category=mod.WasteCategory.NON_HAZARDOUS,
            quantity_tonnes=1200.0,
            destination=mod.WasteDestination.RECYCLING,
            recycling_rate_pct=95.0,
        ),
        mod.WasteStreamData(
            waste_type=mod.WasteType.PLASTIC,
            waste_category=mod.WasteCategory.NON_HAZARDOUS,
            quantity_tonnes=450.0,
            destination=mod.WasteDestination.RECYCLING,
            recycling_rate_pct=70.0,
        ),
        mod.WasteStreamData(
            waste_type=mod.WasteType.ORGANIC,
            waste_category=mod.WasteCategory.NON_HAZARDOUS,
            quantity_tonnes=200.0,
            destination=mod.WasteDestination.COMPOSTING,
            recycling_rate_pct=90.0,
        ),
        mod.WasteStreamData(
            waste_type=mod.WasteType.CHEMICAL,
            waste_category=mod.WasteCategory.HAZARDOUS,
            quantity_tonnes=80.0,
            destination=mod.WasteDestination.INCINERATION,
            recycling_rate_pct=0.0,
        ),
    ]


# =============================================================================
# 7. Water Intake and Discharge Data Fixtures
# =============================================================================


@pytest.fixture
def sample_water_intake():
    """Create a List[WaterIntakeData] with surface and groundwater sources."""
    mod = _load_engine("water_pollution")
    return [
        mod.WaterIntakeData(
            source=mod.WaterSource.SURFACE,
            volume_m3=Decimal("250000"),
            quality_grade=mod.QualityGrade.RAW,
            water_stressed_area=False,
        ),
        mod.WaterIntakeData(
            source=mod.WaterSource.GROUNDWATER,
            volume_m3=Decimal("80000"),
            quality_grade=mod.QualityGrade.RAW,
            water_stressed_area=False,
        ),
    ]


@pytest.fixture
def sample_water_discharge():
    """Create a List[WaterDischargeData] with treated wastewater discharge."""
    mod = _load_engine("water_pollution")
    return [
        mod.WaterDischargeData(
            destination=mod.WaterSource.SURFACE,
            volume_m3=Decimal("200000"),
            treatment_level=mod.TreatmentLevel.SECONDARY,
        ),
    ]


# =============================================================================
# 8. Pollutant Emission Data Fixtures
# =============================================================================


@pytest.fixture
def sample_pollutants():
    """Create a List[PollutantEmission] with NOx, SOx, dust, VOC."""
    mod = _load_engine("water_pollution")
    return [
        mod.PollutantEmission(
            pollutant_type=mod.PollutantType.NOX,
            category=mod.PollutantCategory.AIR,
            quantity_tonnes=Decimal("120.5"),
            measurement_method=mod.MeasurementMethod.CONTINUOUS_MONITORING,
        ),
        mod.PollutantEmission(
            pollutant_type=mod.PollutantType.SOX,
            category=mod.PollutantCategory.AIR,
            quantity_tonnes=Decimal("45.2"),
            measurement_method=mod.MeasurementMethod.PERIODIC_MEASUREMENT,
        ),
        mod.PollutantEmission(
            pollutant_type=mod.PollutantType.PM10,
            category=mod.PollutantCategory.AIR,
            quantity_tonnes=Decimal("18.7"),
            measurement_method=mod.MeasurementMethod.CONTINUOUS_MONITORING,
        ),
        mod.PollutantEmission(
            pollutant_type=mod.PollutantType.VOC,
            category=mod.PollutantCategory.AIR,
            quantity_tonnes=Decimal("8.3"),
            measurement_method=mod.MeasurementMethod.ENGINEERING_ESTIMATE,
        ),
    ]


# =============================================================================
# 9. BAT Compliance Data Fixtures
# =============================================================================


@pytest.fixture
def sample_bat_data():
    """Create a FacilityBATData for a cement plant with measured parameters."""
    mod = _load_engine("bat_compliance")
    return mod.FacilityBATData(
        facility_id="FAC-TEST-001",
        facility_name="Test Cement Plant",
        sub_sector="cement",
        applicable_brefs=[mod.BREFDocument.CEMENT_LIME],
        measured_parameters=[
            mod.MeasuredParameter(
                parameter_name="dust_mg_nm3",
                measured_value=15.0,
                unit="mg/Nm3",
                measurement_date="2025-06-15",
                measurement_method="continuous",
            ),
            mod.MeasuredParameter(
                parameter_name="nox_mg_nm3",
                measured_value=350.0,
                unit="mg/Nm3",
                measurement_date="2025-06-15",
                measurement_method="continuous",
            ),
            mod.MeasuredParameter(
                parameter_name="sox_mg_nm3",
                measured_value=40.0,
                unit="mg/Nm3",
                measurement_date="2025-06-15",
                measurement_method="periodic",
            ),
        ],
        current_technologies=["ESP", "SNCR", "limestone_wet_scrubber"],
        ied_permit_date="2021-03-15",
    )


# =============================================================================
# 10. Supply Chain Data Fixtures
# =============================================================================


@pytest.fixture
def sample_suppliers():
    """Create a List[SupplierData] with tier 1 and tier 2 suppliers."""
    mod = _load_engine("supply_chain_emissions")
    return [
        mod.SupplierData(
            supplier_id="SUP-001",
            supplier_name="ThyssenKrupp Steel Europe",
            tier=mod.SupplierTier.TIER_1,
            country="DE",
            nace_sector="C24.1",
            spend_eur=Decimal("15000000"),
            scope3_category=mod.Scope3Category.CAT_1,
            reported_emissions_tco2e=Decimal("32000"),
            calculation_method=mod.CalculationMethod.SUPPLIER_SPECIFIC,
            data_quality_score=mod.DataQualityScore.SCORE_2,
        ),
        mod.SupplierData(
            supplier_id="SUP-002",
            supplier_name="BASF Plastics",
            tier=mod.SupplierTier.TIER_1,
            country="DE",
            nace_sector="C20.16",
            spend_eur=Decimal("5000000"),
            scope3_category=mod.Scope3Category.CAT_1,
            calculation_method=mod.CalculationMethod.SPEND_BASED,
            data_quality_score=mod.DataQualityScore.SCORE_4,
        ),
        mod.SupplierData(
            supplier_id="SUP-003",
            supplier_name="Foxconn Industrial",
            tier=mod.SupplierTier.TIER_2,
            country="CN",
            nace_sector="C26",
            spend_eur=Decimal("8000000"),
            scope3_category=mod.Scope3Category.CAT_1,
            calculation_method=mod.CalculationMethod.SPEND_BASED,
            data_quality_score=mod.DataQualityScore.SCORE_5,
        ),
    ]


@pytest.fixture
def sample_bom_emissions():
    """Create a List[BOMEmissionData] for supply chain emission calculation."""
    mod = _load_engine("supply_chain_emissions")
    return [
        mod.BOMEmissionData(
            component_id="BOM-STEEL-01",
            component_name="Body-in-White Steel",
            material_type="steel_primary",
            quantity_per_product=Decimal("650"),
            origin_country="DE",
            supplier_id="SUP-001",
            recycled_content_pct=25.0,
        ),
        mod.BOMEmissionData(
            component_id="BOM-PLASTIC-01",
            component_name="Interior Plastics",
            material_type="plastics_pp",
            quantity_per_product=Decimal("180"),
            origin_country="DE",
            supplier_id="SUP-002",
            recycled_content_pct=15.0,
        ),
    ]


# =============================================================================
# 11. Benchmark KPI Fixtures
# =============================================================================


@pytest.fixture
def sample_facility_kpis():
    """Create a FacilityKPIs for benchmark testing."""
    mod = _load_engine("manufacturing_benchmark")
    return mod.FacilityKPIs(
        facility_id="FAC-TEST-001",
        facility_name="Test Cement Plant",
        sub_sector="cement",
        emission_intensity_tco2e_per_unit=Decimal("0.820"),
        energy_intensity_mj_per_unit=Decimal("3400"),
        water_intensity_m3_per_unit=Decimal("0.35"),
        waste_intensity_kg_per_unit=Decimal("12.5"),
        circularity_rate_pct=Decimal("35.0"),
        renewable_share_pct=Decimal("15.0"),
        scope3_ratio_pct=Decimal("20.0"),
    )


# =============================================================================
# Utility: Module Loader Fixture (for parametrized tests)
# =============================================================================


@pytest.fixture(scope="session")
def load_module():
    """Provide _load_module as a fixture for parametrized tests."""
    return _load_module


@pytest.fixture(scope="session")
def load_engine():
    """Provide _load_engine as a fixture for parametrized tests."""
    return _load_engine
