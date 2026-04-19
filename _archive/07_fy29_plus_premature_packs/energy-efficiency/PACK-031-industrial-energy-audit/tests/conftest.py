# -*- coding: utf-8 -*-
"""
PACK-031 Industrial Energy Audit Pack - Shared Test Fixtures (conftest.py)
==========================================================================

Provides pytest fixtures for the PACK-031 test suite including:
  - Dynamic module loading via importlib (no package install needed)
  - Pack manifest and configuration fixtures
  - Sample facility, meter, production, weather, equipment, compressed air,
    steam system, waste heat, lighting, and audit finding data

All fixtures use importlib.util.spec_from_file_location to load modules
directly from the pack source tree, enabling test execution without
installing the pack as a Python package.

Fixture Categories:
  1. Paths and YAML data
  2. Configuration objects
  3. Facility data
  4. Meter readings
  5. Production data
  6. Weather data (HDD/CDD)
  7. Equipment list
  8. Compressed air system
  9. Steam system
  10. Waste heat sources
  11. Lighting zones
  12. Audit findings

Author:  GreenLang Platform Team (GL-TestEngineer)
Pack:    PACK-031 Industrial Energy Audit
Date:    March 2026
"""

import importlib
import importlib.util
import math
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
    "energy_baseline": "energy_baseline_engine.py",
    "energy_audit": "energy_audit_engine.py",
    "process_energy_mapping": "process_energy_mapping_engine.py",
    "equipment_efficiency": "equipment_efficiency_engine.py",
    "energy_savings": "energy_savings_engine.py",
    "waste_heat_recovery": "waste_heat_recovery_engine.py",
    "compressed_air": "compressed_air_engine.py",
    "steam_optimization": "steam_optimization_engine.py",
    "lighting_hvac": "lighting_hvac_engine.py",
    "energy_benchmark": "energy_benchmark_engine.py",
}

# Engine class names that should exist in each engine module
ENGINE_CLASSES = {
    "energy_baseline": "EnergyBaselineEngine",
    "energy_audit": "EnergyAuditEngine",
    "process_energy_mapping": "ProcessEnergyMappingEngine",
    "equipment_efficiency": "EquipmentEfficiencyEngine",
    "energy_savings": "EnergySavingsEngine",
    "waste_heat_recovery": "WasteHeatRecoveryEngine",
    "compressed_air": "CompressedAirEngine",
    "steam_optimization": "SteamOptimizationEngine",
    "lighting_hvac": "LightingHVACEngine",
    "energy_benchmark": "EnergyBenchmarkEngine",
}

# Workflow file mapping
WORKFLOW_FILES = {
    "initial_energy_audit": "initial_energy_audit_workflow.py",
    "continuous_monitoring": "continuous_monitoring_workflow.py",
    "energy_savings_verification": "energy_savings_verification_workflow.py",
    "compressed_air_audit": "compressed_air_audit_workflow.py",
    "steam_system_audit": "steam_system_audit_workflow.py",
    "waste_heat_recovery": "waste_heat_recovery_workflow.py",
    "regulatory_compliance": "regulatory_compliance_workflow.py",
    "iso_50001_certification": "iso_50001_certification_workflow.py",
}

# Template file mapping
TEMPLATE_FILES = {
    "energy_audit_report": "energy_audit_report.py",
    "energy_baseline_report": "energy_baseline_report.py",
    "mv_report": "mv_report.py",
    "compressed_air_report": "compressed_air_report.py",
    "steam_system_report": "steam_system_report.py",
    "waste_heat_report": "waste_heat_report.py",
    "energy_dashboard": "energy_management_dashboard.py",
    "regulatory_compliance_report": "regulatory_compliance_report.py",
    "iso_50001_evidence": "iso_50001_evidence.py",
    "benchmarking_report": "benchmarking_report.py",
}

# Integration file mapping
INTEGRATION_FILES = {
    "pack_orchestrator": "pack_orchestrator.py",
    "smart_meter_bridge": "smart_meter_bridge.py",
    "scada_bridge": "scada_bridge.py",
    "bms_bridge": "bms_bridge.py",
    "erp_bridge": "erp_bridge.py",
    "weather_bridge": "weather_bridge.py",
    "mrv_energy_bridge": "mrv_energy_bridge.py",
    "data_energy_bridge": "data_energy_bridge.py",
    "ets_bridge": "ets_bridge.py",
    "health_check": "health_check.py",
    "setup_wizard": "setup_wizard.py",
    "eed_compliance_bridge": "eed_compliance_bridge.py",
}

# Preset names
PRESET_NAMES = [
    "manufacturing_plant",
    "process_industry",
    "food_beverage",
    "data_center",
    "warehouse_logistics",
    "automotive_manufacturing",
    "steel_metals",
    "sme_industrial",
]


# =============================================================================
# Helper: Dynamic Module Loader
# =============================================================================


def _load_module(module_name: str, file_name: str, subdir: str = "engines"):
    """Load a module dynamically using importlib.util.spec_from_file_location.

    This avoids the need to install PACK-031 as a Python package. The module
    is loaded from the pack source tree and added to sys.modules under a
    unique key to prevent collisions.

    Args:
        module_name: Logical name for the module (used as sys.modules key prefix).
        file_name: File name of the Python module (e.g., "energy_baseline_engine.py").
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
            f"Ensure PACK-031 source files are present."
        )

    # Create a unique module key to avoid collisions
    full_module_name = f"pack031_test.{subdir}.{module_name}"

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
        engine_key: Engine key from ENGINE_FILES (e.g., "energy_baseline").

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
    """Return the absolute path to the PACK-031 root directory."""
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
    """Create an IndustrialEnergyAuditConfig with default values."""
    return config_module.IndustrialEnergyAuditConfig()


@pytest.fixture
def demo_config(config_module, demo_yaml_data):
    """Create an IndustrialEnergyAuditConfig loaded from the demo YAML data."""
    return config_module.IndustrialEnergyAuditConfig(**demo_yaml_data)


@pytest.fixture
def pack_config_wrapper(config_module):
    """Create a PackConfig wrapper with default values."""
    return config_module.PackConfig()


# =============================================================================
# 3. Facility Data Fixture
# =============================================================================


@pytest.fixture
def sample_facility_data():
    """Create a sample manufacturing facility with realistic data.

    Facility: Stuttgart Automotive Parts Plant (Germany)
    Floor area: 18,000 m2
    Production: 12,500 tonnes/year of machined automotive parts
    Employees: 420
    Energy: ~14.5 GWh (Electricity 8.2 GWh + Natural Gas 6.3 GWh)
    Operating hours: 5,800 h/year (2 shifts)
    """
    return {
        "facility_id": "FAC-031-DE-001",
        "facility_name": "Stuttgart Automotive Parts Plant",
        "company_name": "Mittelwerk GmbH",
        "country": "DE",
        "region": "Baden-Wuerttemberg",
        "industry_sector": "MANUFACTURING",
        "nace_code": "C28.1",
        "floor_area_m2": 18000.0,
        "production_area_m2": 12500.0,
        "annual_production_tonnes": 12500.0,
        "production_unit_name": "tonne",
        "employees": 420,
        "operating_hours_per_year": 5800,
        "number_of_shifts": 2,
        "annual_electricity_kwh": 8_200_000.0,
        "annual_gas_kwh": 6_300_000.0,
        "total_energy_kwh": 14_500_000.0,
        "electricity_cost_eur": 1_230_000.0,
        "gas_cost_eur": 378_000.0,
        "total_energy_cost_eur": 1_608_000.0,
        "sec_kwh_per_tonne": 1160.0,
    }


# =============================================================================
# 4. Meter Readings Fixture (12 months hourly -> monthly summaries)
# =============================================================================


@pytest.fixture
def sample_meter_readings():
    """Create 12 months of monthly electricity and gas meter readings.

    Based on a typical manufacturing profile: higher summer electricity
    (cooling), higher winter gas (heating). Monthly granularity for
    baseline regression.
    """
    months = [
        "2024-01", "2024-02", "2024-03", "2024-04", "2024-05", "2024-06",
        "2024-07", "2024-08", "2024-09", "2024-10", "2024-11", "2024-12",
    ]
    # Monthly electricity kWh (seasonal variation: summer higher due to cooling)
    electricity = [
        640_000, 620_000, 660_000, 680_000, 720_000, 760_000,
        780_000, 740_000, 700_000, 680_000, 660_000, 660_000,
    ]
    # Monthly gas kWh (seasonal variation: winter higher due to heating)
    gas = [
        680_000, 650_000, 580_000, 480_000, 380_000, 320_000,
        280_000, 300_000, 400_000, 520_000, 620_000, 690_000,
    ]
    records = []
    for i, month in enumerate(months):
        records.append({
            "period": month,
            "electricity_kwh": electricity[i],
            "gas_kwh": gas[i],
            "total_kwh": electricity[i] + gas[i],
            "electricity_cost_eur": electricity[i] * 0.15,
            "gas_cost_eur": gas[i] * 0.06,
        })
    return records


# =============================================================================
# 5. Production Data Fixture
# =============================================================================


@pytest.fixture
def sample_production_data():
    """Create 12 months of production output data.

    Monthly production in tonnes with typical variation (summer dip for
    maintenance shutdown in August).
    """
    months = [
        "2024-01", "2024-02", "2024-03", "2024-04", "2024-05", "2024-06",
        "2024-07", "2024-08", "2024-09", "2024-10", "2024-11", "2024-12",
    ]
    production = [
        1050, 1020, 1080, 1100, 1120, 1080,
        1060, 800, 1100, 1120, 1080, 1040,
    ]
    return [
        {"period": month, "production_tonnes": prod, "product_mix": "standard"}
        for month, prod in zip(months, production)
    ]


# =============================================================================
# 6. Weather Data Fixture (12 months HDD/CDD)
# =============================================================================


@pytest.fixture
def sample_weather_data():
    """Create 12 months of heating and cooling degree-day data.

    Location: Stuttgart, Germany (base temp 18C for HDD, 24C for CDD)
    """
    months = [
        "2024-01", "2024-02", "2024-03", "2024-04", "2024-05", "2024-06",
        "2024-07", "2024-08", "2024-09", "2024-10", "2024-11", "2024-12",
    ]
    hdd = [520, 460, 380, 220, 100, 30, 10, 15, 80, 240, 390, 500]
    cdd = [0, 0, 0, 5, 25, 60, 95, 85, 40, 5, 0, 0]
    return [
        {
            "period": month,
            "hdd_18c": h,
            "cdd_24c": c,
            "avg_temp_c": round(18.0 - (h / 30.0) + (c / 30.0), 1),
        }
        for month, h, c in zip(months, hdd, cdd)
    ]


# =============================================================================
# 7. Equipment List Fixture
# =============================================================================


@pytest.fixture
def sample_equipment_list():
    """Create a sample equipment list: motors, pumps, compressors, boilers.

    Typical manufacturing plant inventory with mix of new and aged equipment.
    """
    return [
        {
            "equipment_id": "MTR-001",
            "type": "motor",
            "description": "CNC Spindle Motor",
            "rated_power_kw": 37.0,
            "efficiency_class": "IE3",
            "age_years": 5,
            "load_factor_pct": 75.0,
            "annual_operating_hours": 5200,
            "annual_energy_kwh": 37.0 * 0.75 * 5200,
        },
        {
            "equipment_id": "MTR-002",
            "type": "motor",
            "description": "Conveyor Drive Motor",
            "rated_power_kw": 15.0,
            "efficiency_class": "IE2",
            "age_years": 18,
            "load_factor_pct": 60.0,
            "annual_operating_hours": 5800,
            "annual_energy_kwh": 15.0 * 0.60 * 5800,
        },
        {
            "equipment_id": "PMP-001",
            "type": "pump",
            "description": "Cooling Water Pump",
            "rated_power_kw": 22.0,
            "pump_efficiency_pct": 65.0,
            "age_years": 12,
            "flow_m3_per_h": 45.0,
            "head_m": 25.0,
            "annual_operating_hours": 5500,
        },
        {
            "equipment_id": "CMP-001",
            "type": "compressor",
            "description": "Rotary Screw Compressor A",
            "rated_power_kw": 90.0,
            "compressor_type": "rotary_screw",
            "pressure_bar": 7.0,
            "free_air_delivery_m3_min": 14.5,
            "specific_power_kw_per_m3_min": 6.2,
            "age_years": 8,
            "annual_operating_hours": 5800,
            "has_vsd": False,
        },
        {
            "equipment_id": "CMP-002",
            "type": "compressor",
            "description": "Rotary Screw Compressor B (VSD)",
            "rated_power_kw": 75.0,
            "compressor_type": "rotary_screw_vsd",
            "pressure_bar": 7.0,
            "free_air_delivery_m3_min": 12.0,
            "specific_power_kw_per_m3_min": 5.8,
            "age_years": 3,
            "annual_operating_hours": 5800,
            "has_vsd": True,
        },
        {
            "equipment_id": "BLR-001",
            "type": "boiler",
            "description": "Natural Gas Fire-Tube Boiler",
            "rated_capacity_kw": 2000.0,
            "fuel_type": "natural_gas",
            "efficiency_pct": 84.0,
            "age_years": 15,
            "steam_pressure_bar": 10.0,
            "annual_operating_hours": 5000,
        },
        {
            "equipment_id": "FAN-001",
            "type": "fan",
            "description": "Process Exhaust Fan",
            "rated_power_kw": 30.0,
            "efficiency_class": "IE2",
            "age_years": 10,
            "flow_m3_per_s": 8.5,
            "pressure_pa": 1200,
            "annual_operating_hours": 5800,
            "has_vsd": False,
        },
        {
            "equipment_id": "CHL-001",
            "type": "chiller",
            "description": "Process Cooling Chiller",
            "rated_capacity_kw": 350.0,
            "cop": 3.8,
            "refrigerant": "R410A",
            "age_years": 7,
            "annual_operating_hours": 4000,
        },
    ]


# =============================================================================
# 8. Compressed Air System Fixture
# =============================================================================


@pytest.fixture
def sample_compressed_air_system():
    """Create a compressed air system with compressors, receivers, distribution.

    Typical manufacturing plant: 2 compressors (1 fixed + 1 VSD), 2 receivers,
    distribution network with 25% leak rate.
    """
    return {
        "system_id": "CA-001",
        "description": "Main Compressed Air System",
        "operating_pressure_bar": 7.0,
        "total_capacity_m3_min": 26.5,
        "compressors": [
            {
                "id": "CMP-001",
                "type": "rotary_screw",
                "rated_power_kw": 90.0,
                "free_air_delivery_m3_min": 14.5,
                "specific_power_kw_per_m3_min": 6.2,
                "control": "load_unload",
                "has_vsd": False,
            },
            {
                "id": "CMP-002",
                "type": "rotary_screw",
                "rated_power_kw": 75.0,
                "free_air_delivery_m3_min": 12.0,
                "specific_power_kw_per_m3_min": 5.8,
                "control": "vsd",
                "has_vsd": True,
            },
        ],
        "receivers": [
            {"id": "RCV-001", "volume_litres": 3000, "pressure_bar": 7.5},
            {"id": "RCV-002", "volume_litres": 2000, "pressure_bar": 7.0},
        ],
        "distribution": {
            "pipe_length_m": 850.0,
            "main_pipe_diameter_mm": 100,
            "pressure_drop_bar": 0.8,
        },
        "leak_survey": {
            "total_leaks_found": 42,
            "estimated_leak_rate_m3_min": 6.6,
            "leak_rate_pct": 25.0,
            "leak_cost_eur_per_year": 28_500.0,
        },
        "annual_energy_kwh": 940_000.0,
        "annual_operating_hours": 5800,
        "system_specific_power_kw_per_m3_min": 7.8,
    }


# =============================================================================
# 9. Steam System Fixture
# =============================================================================


@pytest.fixture
def sample_steam_system():
    """Create a steam system with boilers, traps, distribution.

    Typical manufacturing steam system: 1 fire-tube boiler, 85 traps,
    10% trap failure rate, 65% condensate return.
    """
    return {
        "system_id": "STM-001",
        "description": "Main Steam Distribution System",
        "boilers": [
            {
                "id": "BLR-001",
                "type": "fire_tube",
                "fuel": "natural_gas",
                "rated_capacity_kw": 2000.0,
                "steam_pressure_bar": 10.0,
                "steam_temperature_c": 184.0,
                "feedwater_temperature_c": 80.0,
                "efficiency_pct": 84.0,
                "flue_gas_temperature_c": 220.0,
                "o2_in_flue_pct": 4.5,
                "co2_in_flue_pct": 9.8,
                "blowdown_rate_pct": 8.0,
                "annual_gas_kwh": 5_200_000.0,
            },
        ],
        "distribution": {
            "total_pipe_length_m": 620.0,
            "insulated_length_m": 480.0,
            "uninsulated_length_m": 140.0,
            "average_pipe_diameter_mm": 80,
            "average_insulation_thickness_mm": 50,
        },
        "traps": {
            "total_traps": 85,
            "traps_surveyed": 85,
            "traps_failed": 9,
            "failure_rate_pct": 10.6,
            "steam_loss_kg_per_h": 45.0,
            "annual_steam_loss_cost_eur": 15_200.0,
        },
        "condensate": {
            "condensate_return_rate_pct": 65.0,
            "condensate_temperature_c": 85.0,
            "flash_steam_potential_pct": 12.0,
        },
        "annual_steam_production_tonnes": 8500.0,
        "annual_cost_eur": 312_000.0,
    }


# =============================================================================
# 10. Waste Heat Sources Fixture
# =============================================================================


@pytest.fixture
def sample_waste_heat_sources():
    """Create waste heat sources: flue gas, cooling water, compressed air aftercooler.

    Three common industrial waste heat sources with temperature and flow data.
    """
    return [
        {
            "source_id": "WH-001",
            "description": "Boiler Flue Gas",
            "source_type": "flue_gas",
            "temperature_c": 220.0,
            "flow_rate_kg_per_h": 3500.0,
            "specific_heat_kj_per_kg_k": 1.05,
            "available_heat_kw": 3500 * 1.05 * (220 - 60) / 3600,
            "potential_recovery_pct": 40.0,
        },
        {
            "source_id": "WH-002",
            "description": "Process Cooling Water Return",
            "source_type": "cooling_water",
            "temperature_c": 45.0,
            "flow_rate_kg_per_h": 12000.0,
            "specific_heat_kj_per_kg_k": 4.18,
            "available_heat_kw": 12000 * 4.18 * (45 - 25) / 3600,
            "potential_recovery_pct": 60.0,
        },
        {
            "source_id": "WH-003",
            "description": "Compressed Air Aftercooler",
            "source_type": "compressed_air_heat",
            "temperature_c": 80.0,
            "flow_rate_kg_per_h": 2000.0,
            "specific_heat_kj_per_kg_k": 1.01,
            "available_heat_kw": 2000 * 1.01 * (80 - 30) / 3600,
            "potential_recovery_pct": 94.0,
        },
    ]


# =============================================================================
# 11. Lighting Zones Fixture
# =============================================================================


@pytest.fixture
def sample_lighting_zones():
    """Create lighting zones: warehouse, office, production floor.

    Typical industrial lighting inventory with mix of HID, fluorescent, and LED.
    """
    return [
        {
            "zone_id": "LZ-001",
            "zone_name": "Warehouse High Bay",
            "area_m2": 4000.0,
            "current_fixture_type": "HID_400W",
            "fixture_count": 80,
            "installed_power_kw": 32.0,
            "current_lpd_w_per_m2": 8.0,
            "target_lpd_w_per_m2": 4.5,
            "operating_hours_per_year": 5800,
            "occupancy_sensor": False,
            "daylight_available": True,
            "replacement": "LED_200W",
            "replacement_power_kw": 16.0,
            "savings_kw": 16.0,
            "annual_savings_kwh": 16.0 * 5800,
        },
        {
            "zone_id": "LZ-002",
            "zone_name": "Office Area",
            "area_m2": 1500.0,
            "current_fixture_type": "T8_fluorescent_36W",
            "fixture_count": 120,
            "installed_power_kw": 4.32,
            "current_lpd_w_per_m2": 2.88,
            "target_lpd_w_per_m2": 2.5,
            "operating_hours_per_year": 2500,
            "occupancy_sensor": False,
            "daylight_available": True,
            "replacement": "LED_panel_24W",
            "replacement_power_kw": 2.88,
            "savings_kw": 1.44,
            "annual_savings_kwh": 1.44 * 2500,
        },
        {
            "zone_id": "LZ-003",
            "zone_name": "Production Floor",
            "area_m2": 8000.0,
            "current_fixture_type": "T5_fluorescent_54W",
            "fixture_count": 200,
            "installed_power_kw": 10.8,
            "current_lpd_w_per_m2": 1.35,
            "target_lpd_w_per_m2": 1.2,
            "operating_hours_per_year": 5800,
            "occupancy_sensor": True,
            "daylight_available": False,
            "replacement": "LED_linear_36W",
            "replacement_power_kw": 7.2,
            "savings_kw": 3.6,
            "annual_savings_kwh": 3.6 * 5800,
        },
    ]


# =============================================================================
# 12. Audit Findings Fixture
# =============================================================================


@pytest.fixture
def sample_audit_findings():
    """Create 12 realistic audit findings with savings data.

    Covers compressed air leaks, motor replacement, VSD retrofit, LED upgrade,
    steam trap repair, insulation, heat recovery, and operational measures.
    """
    return [
        {
            "finding_id": "AF-001",
            "category": "compressed_air",
            "title": "Compressed Air Leak Repair Program",
            "description": "42 leaks found during ultrasonic survey totaling 6.6 m3/min",
            "priority": "high",
            "complexity": "low",
            "annual_savings_kwh": 220_000,
            "annual_savings_eur": 33_000,
            "implementation_cost_eur": 8_500,
            "payback_years": 0.26,
            "co2_reduction_tonnes": 92.4,
        },
        {
            "finding_id": "AF-002",
            "category": "motors",
            "title": "Replace IE2 Conveyor Motors with IE4",
            "description": "6 conveyor motors (15 kW) at IE2 class operating >5000 h/yr",
            "priority": "medium",
            "complexity": "medium",
            "annual_savings_kwh": 18_000,
            "annual_savings_eur": 2_700,
            "implementation_cost_eur": 12_000,
            "payback_years": 4.4,
            "co2_reduction_tonnes": 7.6,
        },
        {
            "finding_id": "AF-003",
            "category": "motors",
            "title": "VSD Retrofit on Cooling Water Pump",
            "description": "22 kW pump running at constant speed, 40% throttled",
            "priority": "high",
            "complexity": "medium",
            "annual_savings_kwh": 45_000,
            "annual_savings_eur": 6_750,
            "implementation_cost_eur": 8_000,
            "payback_years": 1.2,
            "co2_reduction_tonnes": 18.9,
        },
        {
            "finding_id": "AF-004",
            "category": "lighting",
            "title": "LED High Bay Retrofit in Warehouse",
            "description": "Replace 80 x 400W HID with LED 200W in warehouse",
            "priority": "high",
            "complexity": "low",
            "annual_savings_kwh": 92_800,
            "annual_savings_eur": 13_920,
            "implementation_cost_eur": 24_000,
            "payback_years": 1.7,
            "co2_reduction_tonnes": 39.0,
        },
        {
            "finding_id": "AF-005",
            "category": "steam",
            "title": "Steam Trap Replacement Program",
            "description": "9 failed traps losing 45 kg/h of steam",
            "priority": "high",
            "complexity": "low",
            "annual_savings_kwh": 165_000,
            "annual_savings_eur": 15_200,
            "implementation_cost_eur": 4_500,
            "payback_years": 0.30,
            "co2_reduction_tonnes": 33.0,
        },
        {
            "finding_id": "AF-006",
            "category": "steam",
            "title": "Pipe Insulation - Uninsulated Sections",
            "description": "140m of uninsulated steam pipes at 10 bar",
            "priority": "medium",
            "complexity": "low",
            "annual_savings_kwh": 85_000,
            "annual_savings_eur": 5_100,
            "implementation_cost_eur": 7_000,
            "payback_years": 1.4,
            "co2_reduction_tonnes": 17.0,
        },
        {
            "finding_id": "AF-007",
            "category": "waste_heat",
            "title": "Boiler Flue Gas Economizer",
            "description": "Install economizer to recover heat from 220C flue gas",
            "priority": "medium",
            "complexity": "high",
            "annual_savings_kwh": 280_000,
            "annual_savings_eur": 16_800,
            "implementation_cost_eur": 45_000,
            "payback_years": 2.7,
            "co2_reduction_tonnes": 56.0,
        },
        {
            "finding_id": "AF-008",
            "category": "compressed_air",
            "title": "Reduce System Pressure from 7.5 to 6.5 bar",
            "description": "Most end-users require only 5.5 bar; reduce generation pressure",
            "priority": "medium",
            "complexity": "low",
            "annual_savings_kwh": 65_000,
            "annual_savings_eur": 9_750,
            "implementation_cost_eur": 2_000,
            "payback_years": 0.21,
            "co2_reduction_tonnes": 27.3,
        },
        {
            "finding_id": "AF-009",
            "category": "hvac",
            "title": "Install Economizer Free Cooling on Chiller",
            "description": "Enable free cooling mode when ambient < 10C (est. 2500 h/yr)",
            "priority": "medium",
            "complexity": "medium",
            "annual_savings_kwh": 55_000,
            "annual_savings_eur": 8_250,
            "implementation_cost_eur": 15_000,
            "payback_years": 1.8,
            "co2_reduction_tonnes": 23.1,
        },
        {
            "finding_id": "AF-010",
            "category": "operational",
            "title": "Compressed Air Shutdown During Non-Production",
            "description": "System runs 24/7 but production only 2 shifts. Shutdown weekends.",
            "priority": "high",
            "complexity": "low",
            "annual_savings_kwh": 120_000,
            "annual_savings_eur": 18_000,
            "implementation_cost_eur": 500,
            "payback_years": 0.03,
            "co2_reduction_tonnes": 50.4,
        },
        {
            "finding_id": "AF-011",
            "category": "waste_heat",
            "title": "Compressor Heat Recovery for Space Heating",
            "description": "Recover 94% of compressor energy as heat for factory space heating",
            "priority": "medium",
            "complexity": "medium",
            "annual_savings_kwh": 140_000,
            "annual_savings_eur": 8_400,
            "implementation_cost_eur": 18_000,
            "payback_years": 2.1,
            "co2_reduction_tonnes": 28.0,
        },
        {
            "finding_id": "AF-012",
            "category": "motors",
            "title": "VSD Retrofit on Process Exhaust Fan",
            "description": "30 kW exhaust fan running at constant speed, demand varies 50-100%",
            "priority": "high",
            "complexity": "medium",
            "annual_savings_kwh": 52_000,
            "annual_savings_eur": 7_800,
            "implementation_cost_eur": 9_500,
            "payback_years": 1.2,
            "co2_reduction_tonnes": 21.8,
        },
    ]
