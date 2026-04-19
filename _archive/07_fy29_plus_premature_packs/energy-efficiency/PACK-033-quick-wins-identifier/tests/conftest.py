# -*- coding: utf-8 -*-
"""
PACK-033 Quick Wins Identifier Pack - Shared Test Fixtures (conftest.py)
=========================================================================

Provides pytest fixtures for the PACK-033 test suite including:
  - Dynamic module loading via importlib (no package install needed)
  - Pack manifest and configuration fixtures
  - Sample facility, equipment, measures, financial params, and energy
    baseline data for comprehensive engine testing

All fixtures use importlib.util.spec_from_file_location to load modules
directly from the pack source tree, enabling test execution without
installing the pack as a Python package.

Fixture Categories:
  1. Paths and YAML data
  2. Configuration objects
  3. Facility data
  4. Equipment survey data
  5. Quick-win measures data
  6. Financial parameters
  7. Energy baseline data

Author:  GreenLang Platform Team (GL-TestEngineer)
Pack:    PACK-033 Quick Wins Identifier
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
    "quick_wins_scanner": "quick_wins_scanner_engine.py",
    "payback_calculator": "payback_calculator_engine.py",
    "energy_savings_estimator": "energy_savings_estimator_engine.py",
    "carbon_reduction": "carbon_reduction_engine.py",
    "implementation_prioritizer": "implementation_prioritizer_engine.py",
    "behavioral_change": "behavioral_change_engine.py",
    "utility_rebate": "utility_rebate_engine.py",
    "quick_wins_reporting": "quick_wins_reporting_engine.py",
}

# Engine class names that should exist in each engine module
ENGINE_CLASSES = {
    "quick_wins_scanner": "QuickWinsScannerEngine",
    "payback_calculator": "PaybackCalculatorEngine",
    "energy_savings_estimator": "EnergySavingsEstimatorEngine",
    "carbon_reduction": "CarbonReductionEngine",
    "implementation_prioritizer": "ImplementationPrioritizerEngine",
    "behavioral_change": "BehavioralChangeEngine",
    "utility_rebate": "UtilityRebateEngine",
    "quick_wins_reporting": "QuickWinsReportingEngine",
}

# Workflow file mapping
WORKFLOW_FILES = {
    "facility_scan": "facility_scan_workflow.py",
    "prioritization": "prioritization_workflow.py",
    "implementation_planning": "implementation_planning_workflow.py",
    "progress_tracking": "progress_tracking_workflow.py",
    "reporting": "reporting_workflow.py",
    "full_assessment": "full_assessment_workflow.py",
}

# Workflow class names
WORKFLOW_CLASSES = {
    "facility_scan": "FacilityScanWorkflow",
    "prioritization": "PrioritizationWorkflow",
    "implementation_planning": "ImplementationPlanningWorkflow",
    "progress_tracking": "ProgressTrackingWorkflow",
    "reporting": "ReportingWorkflow",
    "full_assessment": "FullAssessmentWorkflow",
}

# Template file mapping
TEMPLATE_FILES = {
    "scan_report": "quick_wins_scan_report.py",
    "payback_report": "payback_analysis_report.py",
    "savings_report": "savings_estimate_report.py",
    "carbon_report": "carbon_reduction_report.py",
    "priority_matrix": "priority_matrix_report.py",
    "behavioral_report": "behavioral_change_report.py",
    "rebate_report": "rebate_incentive_report.py",
    "executive_dashboard": "executive_dashboard.py",
}

# Template class names
TEMPLATE_CLASSES = {
    "scan_report": "QuickWinsScanReportTemplate",
    "payback_report": "PaybackAnalysisReportTemplate",
    "savings_report": "SavingsEstimateReportTemplate",
    "carbon_report": "CarbonReductionReportTemplate",
    "priority_matrix": "PriorityMatrixReportTemplate",
    "behavioral_report": "BehavioralChangeReportTemplate",
    "rebate_report": "RebateIncentiveReportTemplate",
    "executive_dashboard": "ExecutiveDashboardTemplate",
}

# Integration file mapping
INTEGRATION_FILES = {
    "pack_orchestrator": "pack_orchestrator.py",
    "mrv_bridge": "mrv_bridge.py",
    "data_bridge": "data_bridge.py",
    "pack031_bridge": "pack031_bridge.py",
    "pack032_bridge": "pack032_bridge.py",
    "utility_rebate_bridge": "utility_rebate_bridge.py",
    "bms_bridge": "bms_bridge.py",
    "weather_bridge": "weather_bridge.py",
    "health_check": "health_check.py",
    "setup_wizard": "setup_wizard.py",
    "alert_bridge": "alert_bridge.py",
}

# Integration class names
INTEGRATION_CLASSES = {
    "pack_orchestrator": "QuickWinsOrchestrator",
    "mrv_bridge": "MRVBridge",
    "data_bridge": "DataBridge",
    "pack031_bridge": "Pack031Bridge",
    "pack032_bridge": "Pack032Bridge",
    "utility_rebate_bridge": "UtilityRebateBridge",
    "bms_bridge": "BMSBridge",
    "weather_bridge": "WeatherBridge",
    "health_check": "HealthCheck",
    "setup_wizard": "SetupWizard",
    "alert_bridge": "AlertBridge",
}

# Preset names
PRESET_NAMES = [
    "office_building",
    "manufacturing_plant",
    "retail_store",
    "warehouse",
    "healthcare_facility",
    "education_campus",
    "data_center",
    "sme_operations",
]


# =============================================================================
# Helper: Dynamic Module Loader
# =============================================================================


def load_engine_module(engine_name: str):
    """Load an engine module dynamically using importlib.

    This avoids the need to install PACK-033 as a Python package. The module
    is loaded from the pack source tree and added to sys.modules under a
    unique key to prevent collisions.

    Args:
        engine_name: Logical name from ENGINE_FILES (e.g., "payback_calculator").

    Returns:
        The loaded module object.

    Raises:
        FileNotFoundError: If the module file does not exist.
        ImportError: If the module cannot be loaded.
    """
    file_name = ENGINE_FILES[engine_name]
    return _load_module(engine_name, file_name, "engines")


def _load_module(module_name: str, file_name: str, subdir: str = "engines"):
    """Load a module dynamically using importlib.util.spec_from_file_location.

    This avoids the need to install PACK-033 as a Python package. The module
    is loaded from the pack source tree and added to sys.modules under a
    unique key to prevent collisions.

    Args:
        module_name: Logical name for the module (used as sys.modules key prefix).
        file_name: File name of the Python module (e.g., "payback_calculator_engine.py").
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
            f"Ensure PACK-033 source files are present."
        )

    # Create a unique module key to avoid collisions
    full_module_name = f"pack033_test.{subdir}.{module_name}"

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
        engine_key: Engine key from ENGINE_FILES (e.g., "payback_calculator").

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
    """Return the absolute path to the PACK-033 root directory."""
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
    if not demo_yaml_path.exists():
        pytest.skip("demo_config.yaml not found")
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
    try:
        return _load_config_module()
    except (FileNotFoundError, ImportError):
        pytest.skip("pack_config module not available")


# =============================================================================
# 3. Facility Data Fixture
# =============================================================================


@pytest.fixture
def sample_facility():
    """Create a sample commercial facility profile for quick-wins scanning.

    Facility: London Office Tower (United Kingdom)
    Floor area: 12,000 m2
    Employees: 350
    Energy: ~2.4 GWh (Electricity 1.8 GWh + Natural Gas 0.6 GWh)
    Operating hours: 3,120 h/year (standard office)
    """
    return {
        "facility_id": "FAC-033-UK-001",
        "facility_name": "London Office Tower",
        "company_name": "GreenTech Solutions Ltd",
        "country": "GB",
        "region": "Greater London",
        "building_type": "OFFICE",
        "floor_area_m2": 12000.0,
        "conditioned_area_m2": 11000.0,
        "floors": 8,
        "year_built": 2005,
        "last_renovation_year": 2018,
        "employees": 350,
        "operating_hours_per_year": 3120,
        "annual_electricity_kwh": 1_800_000.0,
        "annual_gas_kwh": 600_000.0,
        "total_energy_kwh": 2_400_000.0,
        "electricity_cost_eur": 360_000.0,
        "gas_cost_eur": 36_000.0,
        "total_energy_cost_eur": 396_000.0,
        "eui_kwh_per_m2": 200.0,
    }


# =============================================================================
# 4. Equipment Survey Data Fixture
# =============================================================================


@pytest.fixture
def sample_equipment():
    """Create a sample equipment survey for a commercial building.

    Includes HVAC, lighting, plug loads, and controls typical of an
    office building quick-wins assessment.
    """
    return [
        {
            "equipment_id": "HVAC-001",
            "type": "chiller",
            "description": "Centrifugal Chiller - Main",
            "rated_capacity_kw": 500.0,
            "cop": 4.2,
            "age_years": 12,
            "annual_operating_hours": 2800,
            "control_type": "constant_speed",
            "condition": "fair",
        },
        {
            "equipment_id": "HVAC-002",
            "type": "ahu",
            "description": "AHU Floors 1-4",
            "rated_power_kw": 45.0,
            "age_years": 8,
            "annual_operating_hours": 3120,
            "has_vsd": False,
            "has_economizer": False,
            "condition": "good",
        },
        {
            "equipment_id": "LTG-001",
            "type": "lighting",
            "description": "Office T8 Fluorescent",
            "fixture_count": 600,
            "wattage_per_fixture": 36,
            "installed_power_kw": 21.6,
            "annual_operating_hours": 3120,
            "has_occupancy_sensor": False,
            "has_daylight_sensor": False,
            "condition": "operational",
        },
        {
            "equipment_id": "BLR-001",
            "type": "boiler",
            "description": "Gas Condensing Boiler",
            "rated_capacity_kw": 300.0,
            "fuel_type": "natural_gas",
            "efficiency_pct": 89.0,
            "age_years": 10,
            "annual_operating_hours": 2200,
            "condition": "good",
        },
        {
            "equipment_id": "PLG-001",
            "type": "plug_load",
            "description": "IT Equipment & Desktop PCs",
            "estimated_power_kw": 35.0,
            "annual_operating_hours": 3120,
            "has_power_management": False,
            "condition": "operational",
        },
    ]


# =============================================================================
# 5. Quick-Win Measures Data Fixture
# =============================================================================


@pytest.fixture
def sample_measures():
    """Create a list of sample quick-win measures with financial data.

    Covers lighting, HVAC, controls, behavioral, and operational measures
    typical of a commercial office quick-wins assessment.
    """
    return [
        {
            "measure_id": "QW-001",
            "category": "lighting",
            "title": "LED Retrofit - Office Floors",
            "description": "Replace 600 x 36W T8 fluorescent with 18W LED panels",
            "complexity": "low",
            "annual_savings_kwh": 33_696,
            "annual_savings_eur": 6_739,
            "implementation_cost_eur": 12_000,
            "payback_years": 1.8,
            "co2_reduction_tonnes": 14.2,
        },
        {
            "measure_id": "QW-002",
            "category": "controls",
            "title": "Occupancy Sensors - Meeting Rooms",
            "description": "Install occupancy sensors in 40 meeting rooms for lighting + HVAC",
            "complexity": "low",
            "annual_savings_kwh": 18_000,
            "annual_savings_eur": 3_600,
            "implementation_cost_eur": 8_000,
            "payback_years": 2.2,
            "co2_reduction_tonnes": 7.6,
        },
        {
            "measure_id": "QW-003",
            "category": "hvac",
            "title": "AHU VSD Retrofit",
            "description": "Add variable speed drives to AHU supply/return fans",
            "complexity": "medium",
            "annual_savings_kwh": 42_000,
            "annual_savings_eur": 8_400,
            "implementation_cost_eur": 15_000,
            "payback_years": 1.8,
            "co2_reduction_tonnes": 17.6,
        },
        {
            "measure_id": "QW-004",
            "category": "behavioral",
            "title": "Switch-Off Campaign",
            "description": "Employee engagement program for equipment switch-off",
            "complexity": "low",
            "annual_savings_kwh": 25_000,
            "annual_savings_eur": 5_000,
            "implementation_cost_eur": 2_000,
            "payback_years": 0.4,
            "co2_reduction_tonnes": 10.5,
        },
        {
            "measure_id": "QW-005",
            "category": "operational",
            "title": "HVAC Schedule Optimization",
            "description": "Reduce HVAC pre-conditioning from 2h to 30min using BMS",
            "complexity": "low",
            "annual_savings_kwh": 30_000,
            "annual_savings_eur": 6_000,
            "implementation_cost_eur": 500,
            "payback_years": 0.08,
            "co2_reduction_tonnes": 12.6,
        },
    ]


# =============================================================================
# 6. Financial Parameters Fixture
# =============================================================================


@pytest.fixture
def sample_financial_params():
    """Create sample financial parameters for payback analysis.

    Standard UK commercial analysis with 8% discount rate, 3% electricity
    escalation, 10-year analysis period.
    """
    return {
        "discount_rate": Decimal("0.08"),
        "inflation_rate": Decimal("0.025"),
        "electricity_escalation_rate": Decimal("0.03"),
        "gas_escalation_rate": Decimal("0.025"),
        "analysis_period_years": 10,
        "tax_rate": Decimal("0.19"),
        "electricity_price_eur_per_kwh": Decimal("0.20"),
        "gas_price_eur_per_kwh": Decimal("0.06"),
    }


# =============================================================================
# 7. Energy Baseline Data Fixture
# =============================================================================


@pytest.fixture
def sample_energy_baseline():
    """Create 12 months of energy baseline data for an office building.

    Monthly electricity and gas with typical commercial seasonal profile:
    higher summer electricity (cooling), higher winter gas (heating).
    """
    months = [
        "2024-01", "2024-02", "2024-03", "2024-04", "2024-05", "2024-06",
        "2024-07", "2024-08", "2024-09", "2024-10", "2024-11", "2024-12",
    ]
    electricity = [
        140_000, 135_000, 145_000, 150_000, 165_000, 180_000,
        190_000, 185_000, 160_000, 145_000, 140_000, 145_000,
    ]
    gas = [
        80_000, 75_000, 60_000, 40_000, 20_000, 10_000,
        5_000, 5_000, 15_000, 40_000, 65_000, 85_000,
    ]
    records = []
    for i, month in enumerate(months):
        records.append({
            "period": month,
            "electricity_kwh": electricity[i],
            "gas_kwh": gas[i],
            "total_kwh": electricity[i] + gas[i],
            "electricity_cost_eur": electricity[i] * 0.20,
            "gas_cost_eur": gas[i] * 0.06,
        })
    return records
