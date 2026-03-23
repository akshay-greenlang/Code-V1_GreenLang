# -*- coding: utf-8 -*-
"""
PACK-038 Peak Shaving Pack - Shared Test Fixtures (conftest.py)
====================================================================

Provides pytest fixtures for the PACK-038 test suite including:
  - Dynamic module loading via importlib (no package install needed)
  - Pack manifest and configuration fixtures
  - Sample facility profile (Chicago Commercial Building)
  - Interval data (2,880 readings: 30 days x 96 intervals/day)
  - Peak events (12 monthly billing peaks with attribution)
  - Tariff structure (flat + TOU + CP + ratchet + PF penalty)
  - BESS configuration (500 kWh / 250 kW LFP with degradation)
  - Shiftable loads (6 loads: HVAC, EV, thermal, production, water, laundry)
  - CP data (PJM 5CP events with weather correlation)
  - Power factor data (96 interval PF readings, 0.82-0.97 range)
  - Revenue data (annual financial data with demand charge savings)
  - Emission factors (24-hour marginal emission factors)

Fixture Categories:
  1. Paths and YAML data
  2. Configuration objects
  3. Facility profile
  4. Interval data (30 days x 96 intervals)
  5. Peak events (12 monthly)
  6. Tariff structure (multi-component)
  7. BESS configuration (LFP battery)
  8. Shiftable loads (6 loads)
  9. CP data (5CP events)
  10. Power factor data (96 intervals)
  11. Revenue and financial data
  12. Emission factors (24-hour marginal)

Author:  GreenLang Platform Team (GL-TestEngineer)
Pack:    PACK-038 Peak Shaving
Date:    March 2026
"""

import importlib
import importlib.util
import hashlib
import json
import math
import random
import sys
from datetime import datetime, timedelta
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
DEMO_DIR = CONFIG_DIR / "demo"

# Engine file mapping: logical name -> file name on disk
ENGINE_FILES = {
    "load_profile": "load_profile_engine.py",
    "peak_identifier": "peak_identifier_engine.py",
    "demand_charge": "demand_charge_engine.py",
    "bess_sizing": "bess_sizing_engine.py",
    "load_shifting": "load_shifting_engine.py",
    "cp_management": "cp_management_engine.py",
    "ratchet_analysis": "ratchet_analysis_engine.py",
    "power_factor": "power_factor_engine.py",
    "financial": "financial_engine.py",
    "peak_reporting": "peak_reporting_engine.py",
}

# Engine class names that should exist in each engine module
ENGINE_CLASSES = {
    "load_profile": "LoadProfileEngine",
    "peak_identifier": "PeakIdentifierEngine",
    "demand_charge": "DemandChargeEngine",
    "bess_sizing": "BESSSizingEngine",
    "load_shifting": "LoadShiftingEngine",
    "cp_management": "CPManagementEngine",
    "ratchet_analysis": "RatchetAnalysisEngine",
    "power_factor": "PowerFactorEngine",
    "financial": "FinancialEngine",
    "peak_reporting": "PeakReportingEngine",
}

# Workflow file mapping
WORKFLOW_FILES = {
    "load_analysis": "load_analysis_workflow.py",
    "peak_assessment": "peak_assessment_workflow.py",
    "bess_evaluation": "bess_evaluation_workflow.py",
    "shifting_optimization": "shifting_optimization_workflow.py",
    "cp_strategy": "cp_strategy_workflow.py",
    "ratchet_mitigation": "ratchet_mitigation_workflow.py",
    "reporting": "reporting_workflow.py",
    "full_lifecycle": "full_lifecycle_workflow.py",
}

# Workflow class names
WORKFLOW_CLASSES = {
    "load_analysis": "LoadAnalysisWorkflow",
    "peak_assessment": "PeakAssessmentWorkflow",
    "bess_evaluation": "BESSEvaluationWorkflow",
    "shifting_optimization": "ShiftingOptimizationWorkflow",
    "cp_strategy": "CPStrategyWorkflow",
    "ratchet_mitigation": "RatchetMitigationWorkflow",
    "reporting": "ReportingWorkflow",
    "full_lifecycle": "FullLifecycleWorkflow",
}

# Workflow expected phase counts
WORKFLOW_PHASE_COUNTS = {
    "load_analysis": 4,
    "peak_assessment": 5,
    "bess_evaluation": 5,
    "shifting_optimization": 4,
    "cp_strategy": 4,
    "ratchet_mitigation": 4,
    "reporting": 3,
    "full_lifecycle": 8,
}

# Template file mapping
TEMPLATE_FILES = {
    "load_profile_report": "load_profile_report.py",
    "peak_analysis_report": "peak_analysis_report.py",
    "demand_charge_report": "demand_charge_report.py",
    "bess_sizing_report": "bess_sizing_report.py",
    "load_shifting_report": "load_shifting_report.py",
    "cp_management_report": "cp_management_report.py",
    "ratchet_report": "ratchet_report.py",
    "power_factor_report": "power_factor_report.py",
    "financial_report": "financial_report.py",
    "executive_summary": "executive_summary.py",
}

# Template class names
TEMPLATE_CLASSES = {
    "load_profile_report": "LoadProfileReportTemplate",
    "peak_analysis_report": "PeakAnalysisReportTemplate",
    "demand_charge_report": "DemandChargeReportTemplate",
    "bess_sizing_report": "BESSSizingReportTemplate",
    "load_shifting_report": "LoadShiftingReportTemplate",
    "cp_management_report": "CPManagementReportTemplate",
    "ratchet_report": "RatchetReportTemplate",
    "power_factor_report": "PowerFactorReportTemplate",
    "financial_report": "FinancialReportTemplate",
    "executive_summary": "ExecutiveSummaryTemplate",
}

# Integration file mapping
INTEGRATION_FILES = {
    "pack_orchestrator": "pack_orchestrator.py",
    "mrv_bridge": "mrv_bridge.py",
    "data_bridge": "data_bridge.py",
    "metering_bridge": "metering_bridge.py",
    "pack037_bridge": "pack037_bridge.py",
    "iso_rto_bridge": "iso_rto_bridge.py",
    "scada_bridge": "scada_bridge.py",
    "bms_bridge": "bms_bridge.py",
    "bess_bridge": "bess_bridge.py",
    "health_check": "health_check.py",
    "setup_wizard": "setup_wizard.py",
    "alert_bridge": "alert_bridge.py",
}

# Integration class names
INTEGRATION_CLASSES = {
    "pack_orchestrator": "PeakShavingOrchestrator",
    "mrv_bridge": "MRVBridge",
    "data_bridge": "DataBridge",
    "metering_bridge": "MeteringBridge",
    "pack037_bridge": "Pack037Bridge",
    "iso_rto_bridge": "ISORTOBridge",
    "scada_bridge": "SCADABridge",
    "bms_bridge": "BMSBridge",
    "bess_bridge": "BESSBridge",
    "health_check": "HealthCheck",
    "setup_wizard": "SetupWizard",
    "alert_bridge": "AlertBridge",
}

# Preset names
PRESET_NAMES = [
    "commercial_office",
    "industrial_facility",
    "retail_chain",
    "campus",
    "data_center",
    "hospital",
    "warehouse",
    "multi_site_portfolio",
]


# =============================================================================
# Helper: Dynamic Module Loader
# =============================================================================


def _load_module(module_name: str, file_name: str, subdir: str = "engines"):
    """Load a module dynamically using importlib.util.spec_from_file_location.

    This avoids the need to install PACK-038 as a Python package. The module
    is loaded from the pack source tree and added to sys.modules under a
    unique key to prevent collisions.

    Args:
        module_name: Logical name for the module (used as sys.modules key prefix).
        file_name: File name of the Python module.
        subdir: Subdirectory under PACK_ROOT.

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
            f"Ensure PACK-038 source files are present."
        )

    full_module_name = f"pack038_test.{subdir}.{module_name}"

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
# 1. Path and YAML Fixtures
# =============================================================================


@pytest.fixture(scope="session")
def pack_root() -> Path:
    """Return the absolute path to the PACK-038 root directory."""
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
        data = __import__("yaml").safe_load(f)
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


@pytest.fixture
def pack_config():
    """Create a default PACK-038 configuration dictionary."""
    return {
        "pack_id": "PACK-038",
        "pack_name": "Peak Shaving",
        "version": "1.0.0",
        "category": "energy-efficiency",
        "environment": "test",
        "currency": "USD",
        "default_region": "US",
        "default_iso_rto": "PJM",
        "decimal_precision": 4,
        "provenance_enabled": True,
        "demand_interval_minutes": 15,
        "peak_method": "BILLING_PEAK",
        "ratchet_pct": Decimal("0.80"),
        "power_factor_target": Decimal("0.95"),
        "bess_min_soc_pct": Decimal("0.10"),
        "discount_rate": Decimal("0.08"),
        "analysis_horizon_years": 15,
    }


# =============================================================================
# 3. Facility Profile
# =============================================================================


@pytest.fixture
def sample_facility_profile():
    """Create a sample commercial building profile for peak shaving analysis.

    Facility: Chicago Commercial Building
    Peak demand: 2,000 kW
    Annual consumption: 10,000 MWh
    """
    return {
        "facility_id": "FAC-038-US-001",
        "facility_name": "Chicago Commercial Building",
        "facility_type": "COMMERCIAL_OFFICE",
        "address": "233 South Wacker Dr, Chicago, IL 60606",
        "country": "US",
        "state": "IL",
        "iso_rto": "PJM",
        "utility_zone": "ComEd",
        "climate_zone": "5A",
        "floor_area_m2": 38_000,
        "floors": 10,
        "year_built": 2008,
        "occupancy_hours": "07:00-21:00 Mon-Fri",
        "peak_demand_kw": 2000.0,
        "average_demand_kw": 1140.0,
        "minimum_demand_kw": 520.0,
        "load_factor": 0.57,
        "annual_consumption_kwh": 10_000_000,
        "annual_energy_cost_usd": Decimal("950_000.00"),
        "annual_demand_charge_usd": Decimal("192_000.00"),
        "demand_charge_pct_of_bill": Decimal("0.168"),
        "power_factor_avg": 0.91,
        "has_bms": True,
        "bms_system": "Siemens_Desigo",
        "scada_connected": True,
        "has_backup_generation": True,
        "backup_generator_kw": 1200.0,
        "has_energy_storage": False,
        "has_solar": True,
        "solar_capacity_kw": 300.0,
        "has_ev_chargers": True,
        "ev_charger_count": 15,
        "ev_total_power_kw": 225.0,
    }


# =============================================================================
# 4. Interval Data (30 days x 96 intervals = 2,880 readings)
# =============================================================================


@pytest.fixture
def sample_interval_data():
    """Create 15-minute interval data for 30 days (July 2025).

    30 days x 96 intervals = 2,880 data points.
    Simulates a commercial office load profile with summer cooling peaks.
    Seeded random (seed=42) for deterministic output.
    """
    rng = random.Random(42)
    intervals = []
    for day in range(1, 31):
        for interval in range(96):
            hour = interval // 4
            minute = (interval % 4) * 15
            weekday = ((day - 1) + 1) % 7  # July 1, 2025 = Tuesday
            is_workday = weekday < 5

            if not is_workday:
                base = 540.0
                variation = rng.uniform(-40, 40)
            elif 0 <= hour < 6:
                base = 560.0
                variation = rng.uniform(-25, 25)
            elif 6 <= hour < 9:
                ramp_factor = (hour - 6 + minute / 60) / 3.0
                base = 560.0 + ramp_factor * 1100.0
                variation = rng.uniform(-50, 50)
            elif 9 <= hour < 17:
                base = 1650.0 + rng.uniform(-120, 350)
                variation = rng.uniform(-25, 25)
            elif 17 <= hour < 21:
                ramp_factor = 1.0 - (hour - 17 + minute / 60) / 4.0
                base = 560.0 + ramp_factor * 1100.0
                variation = rng.uniform(-40, 40)
            else:
                base = 580.0
                variation = rng.uniform(-25, 25)

            demand_kw = max(0, base + variation)
            temperature_c = round(
                24 + 10 * math.sin(math.pi * (hour - 6) / 12)
                if 6 <= hour <= 18 else 22, 1
            )
            intervals.append({
                "timestamp": f"2025-07-{day:02d}T{hour:02d}:{minute:02d}:00",
                "demand_kw": round(demand_kw, 2),
                "energy_kwh": round(demand_kw * 0.25, 2),
                "temperature_c": temperature_c,
                "power_factor": round(
                    0.88 + rng.uniform(-0.06, 0.09), 3
                ),
            })
    return intervals


# =============================================================================
# 5. Peak Events (12 monthly billing peaks)
# =============================================================================


@pytest.fixture
def sample_peak_events():
    """Create 12 monthly billing peak events with attribution data."""
    rng = random.Random(42)
    peaks = []
    monthly_peaks_kw = [
        1620, 1580, 1540, 1680, 1750, 1920,
        2000, 1980, 1850, 1700, 1600, 1550,
    ]
    attributions = [
        "HVAC_STARTUP", "HVAC_STARTUP", "NORMAL_OPERATIONS", "HVAC_RAMP",
        "COOLING_PEAK", "EXTREME_HEAT", "EXTREME_HEAT", "COOLING_PEAK",
        "COOLING_PEAK", "HVAC_RAMP", "NORMAL_OPERATIONS", "HEATING_STARTUP",
    ]
    for month_idx in range(12):
        month = month_idx + 1
        day = rng.randint(10, 25)
        hour = rng.choice([10, 11, 13, 14, 15])
        peaks.append({
            "peak_id": f"PK-038-{month:02d}",
            "month": f"2025-{month:02d}",
            "date": f"2025-{month:02d}-{day:02d}",
            "time": f"{hour:02d}:15:00",
            "peak_kw": float(monthly_peaks_kw[month_idx]),
            "attribution": attributions[month_idx],
            "temperature_c": round(15 + 15 * math.sin(
                math.pi * (month - 1) / 6), 1),
            "coincident_with_system_peak": month in [6, 7, 8],
            "avoidable": monthly_peaks_kw[month_idx] > 1700,
            "avoidable_kw": max(0, monthly_peaks_kw[month_idx] - 1700),
        })
    return peaks


# =============================================================================
# 6. Tariff Structure (Multi-Component)
# =============================================================================


@pytest.fixture
def sample_tariff_structure():
    """Create a multi-component tariff with flat, TOU, CP, ratchet, and PF penalty."""
    return {
        "tariff_id": "COMED-SC-10",
        "utility": "ComEd",
        "rate_class": "SC-10 General Service",
        "effective_date": "2025-01-01",
        "expiry_date": "2025-12-31",
        "currency": "USD",
        "flat_demand_charge": {
            "enabled": True,
            "rate_usd_per_kw": Decimal("8.50"),
            "applies_to": "MONTHLY_PEAK",
        },
        "tou_demand_charges": {
            "enabled": True,
            "on_peak": {
                "rate_usd_per_kw": Decimal("14.25"),
                "hours": "09:00-21:00 Mon-Fri",
                "months": [6, 7, 8, 9],
            },
            "mid_peak": {
                "rate_usd_per_kw": Decimal("9.50"),
                "hours": "07:00-09:00,21:00-23:00 Mon-Fri",
                "months": [6, 7, 8, 9],
            },
            "off_peak": {
                "rate_usd_per_kw": Decimal("4.75"),
                "hours": "ALL_OTHER",
                "months": [1, 2, 3, 4, 5, 10, 11, 12],
            },
        },
        "coincident_peak_charge": {
            "enabled": True,
            "methodology": "PJM_5CP",
            "rate_usd_per_kw": Decimal("6.80"),
            "icap_tag_kw": 1850.0,
            "annual_charge_usd": Decimal("12580.00"),
        },
        "ratchet_demand": {
            "enabled": True,
            "ratchet_pct": Decimal("0.80"),
            "lookback_months": 12,
            "ratchet_base_kw": 2000.0,
            "effective_minimum_kw": 1600.0,
        },
        "power_factor_penalty": {
            "enabled": True,
            "target_pf": Decimal("0.90"),
            "penalty_method": "KVA_BILLING",
            "penalty_rate_usd_per_kvar": Decimal("0.45"),
        },
        "energy_charges": {
            "on_peak_usd_per_kwh": Decimal("0.0925"),
            "off_peak_usd_per_kwh": Decimal("0.0642"),
            "shoulder_usd_per_kwh": Decimal("0.0780"),
        },
    }


# =============================================================================
# 7. BESS Configuration (500 kWh / 250 kW LFP)
# =============================================================================


@pytest.fixture
def sample_bess_config():
    """Create a 500 kWh / 250 kW LFP battery config with degradation params."""
    return {
        "bess_id": "BESS-038-001",
        "chemistry": "LFP",
        "nameplate_capacity_kwh": 500.0,
        "nameplate_power_kw": 250.0,
        "usable_capacity_kwh": 450.0,
        "min_soc_pct": 10.0,
        "max_soc_pct": 95.0,
        "current_soc_pct": 50.0,
        "round_trip_efficiency_pct": 92.0,
        "inverter_efficiency_pct": 97.5,
        "aux_power_kw": 3.0,
        "max_charge_rate_c": 0.5,
        "max_discharge_rate_c": 0.5,
        "cycle_count": 0,
        "max_cycles": 6000,
        "calendar_life_years": 15,
        "degradation_pct_per_1000_cycles": 2.0,
        "calendar_degradation_pct_per_year": 1.0,
        "thermal_derating_above_c": 35.0,
        "thermal_derating_pct_per_c": 2.0,
        "installed_cost_usd": Decimal("275000.00"),
        "installed_cost_usd_per_kwh": Decimal("550.00"),
        "annual_om_cost_usd": Decimal("5500.00"),
        "warranty_years": 10,
        "warranty_cycles": 4000,
        "warranty_capacity_retention_pct": 70.0,
    }


# =============================================================================
# 8. Shiftable Loads (6 loads)
# =============================================================================


@pytest.fixture
def sample_shiftable_loads():
    """Create 6 shiftable loads for peak shaving load shifting analysis."""
    return [
        {
            "load_id": "SL-001",
            "name": "HVAC Pre-Cooling",
            "category": "HVAC",
            "rated_kw": 450.0,
            "shiftable_kw": 200.0,
            "shift_window_hours": 3.0,
            "earliest_start": "05:00",
            "latest_end": "22:00",
            "min_runtime_hours": 1.0,
            "max_interruptions": 2,
            "comfort_constraint": "ASHRAE_55",
            "temperature_drift_c_per_hour": 0.8,
            "max_temperature_deviation_c": 2.0,
            "rebound_factor": 1.15,
        },
        {
            "load_id": "SL-002",
            "name": "EV Fleet Charging",
            "category": "EV_CHARGING",
            "rated_kw": 225.0,
            "shiftable_kw": 225.0,
            "shift_window_hours": 8.0,
            "earliest_start": "18:00",
            "latest_end": "07:00",
            "min_runtime_hours": 4.0,
            "max_interruptions": 4,
            "comfort_constraint": "NONE",
            "departure_time": "07:00",
            "min_charge_pct": 80.0,
            "rebound_factor": 1.00,
        },
        {
            "load_id": "SL-003",
            "name": "Thermal Ice Storage",
            "category": "THERMAL_STORAGE",
            "rated_kw": 200.0,
            "shiftable_kw": 200.0,
            "shift_window_hours": 8.0,
            "earliest_start": "22:00",
            "latest_end": "06:00",
            "min_runtime_hours": 6.0,
            "max_interruptions": 0,
            "comfort_constraint": "NONE",
            "storage_capacity_kwh_thermal": 2400.0,
            "discharge_rate_kw_thermal": 300.0,
            "rebound_factor": 1.00,
        },
        {
            "load_id": "SL-004",
            "name": "Production Batch Process",
            "category": "PROCESS",
            "rated_kw": 180.0,
            "shiftable_kw": 180.0,
            "shift_window_hours": 4.0,
            "earliest_start": "06:00",
            "latest_end": "20:00",
            "min_runtime_hours": 2.0,
            "max_interruptions": 0,
            "comfort_constraint": "NONE",
            "deadline": "18:00",
            "rebound_factor": 1.00,
        },
        {
            "load_id": "SL-005",
            "name": "Domestic Water Heating",
            "category": "DHW",
            "rated_kw": 80.0,
            "shiftable_kw": 80.0,
            "shift_window_hours": 6.0,
            "earliest_start": "00:00",
            "latest_end": "06:00",
            "min_runtime_hours": 3.0,
            "max_interruptions": 1,
            "comfort_constraint": "TEMPERATURE_MIN_50C",
            "tank_capacity_litres": 2000,
            "standby_loss_c_per_hour": 0.5,
            "rebound_factor": 1.05,
        },
        {
            "load_id": "SL-006",
            "name": "Laundry / Dryer Systems",
            "category": "LAUNDRY",
            "rated_kw": 60.0,
            "shiftable_kw": 60.0,
            "shift_window_hours": 6.0,
            "earliest_start": "20:00",
            "latest_end": "06:00",
            "min_runtime_hours": 2.0,
            "max_interruptions": 1,
            "comfort_constraint": "NONE",
            "rebound_factor": 1.00,
        },
    ]


# =============================================================================
# 9. CP Data (PJM 5CP events with weather correlation)
# =============================================================================


@pytest.fixture
def sample_cp_data():
    """Create PJM 5CP event data with weather correlation for CP management."""
    return {
        "iso_rto": "PJM",
        "methodology": "5CP",
        "planning_year": "2025-2026",
        "cp_events": [
            {
                "cp_number": 1,
                "date": "2025-07-21",
                "hour_ending": 17,
                "system_peak_mw": 152_300,
                "facility_demand_kw": 1920.0,
                "temperature_c": 38.2,
                "humidity_pct": 62,
                "heat_index_c": 44.5,
                "day_ahead_forecast_mw": 150_000,
                "response_achieved_kw": 280.0,
            },
            {
                "cp_number": 2,
                "date": "2025-07-22",
                "hour_ending": 16,
                "system_peak_mw": 150_800,
                "facility_demand_kw": 1880.0,
                "temperature_c": 37.5,
                "humidity_pct": 58,
                "heat_index_c": 43.0,
                "day_ahead_forecast_mw": 149_500,
                "response_achieved_kw": 320.0,
            },
            {
                "cp_number": 3,
                "date": "2025-08-05",
                "hour_ending": 17,
                "system_peak_mw": 148_500,
                "facility_demand_kw": 1850.0,
                "temperature_c": 36.8,
                "humidity_pct": 65,
                "heat_index_c": 42.8,
                "day_ahead_forecast_mw": 147_000,
                "response_achieved_kw": 350.0,
            },
            {
                "cp_number": 4,
                "date": "2025-08-19",
                "hour_ending": 16,
                "system_peak_mw": 146_200,
                "facility_demand_kw": 1800.0,
                "temperature_c": 35.5,
                "humidity_pct": 60,
                "heat_index_c": 41.0,
                "day_ahead_forecast_mw": 145_000,
                "response_achieved_kw": 400.0,
            },
            {
                "cp_number": 5,
                "date": "2025-09-03",
                "hour_ending": 17,
                "system_peak_mw": 143_800,
                "facility_demand_kw": 1780.0,
                "temperature_c": 34.0,
                "humidity_pct": 55,
                "heat_index_c": 38.5,
                "day_ahead_forecast_mw": 142_500,
                "response_achieved_kw": 420.0,
            },
        ],
        "icap_tag_kw": 1850.0,
        "tag_value_usd_per_kw_year": Decimal("6.80"),
        "annual_cp_charge_usd": Decimal("12580.00"),
        "weather_correlation": {
            "r_squared": 0.89,
            "temp_threshold_c": 33.0,
            "humidity_threshold_pct": 55,
            "heat_index_threshold_c": 38.0,
        },
    }


# =============================================================================
# 10. Power Factor Data (96 intervals)
# =============================================================================


@pytest.fixture
def sample_power_factor_data():
    """Create 96 interval PF readings spanning 0.82-0.97 range."""
    rng = random.Random(42)
    readings = []
    for interval in range(96):
        hour = interval // 4
        minute = (interval % 4) * 15
        # PF dips during heavy motor loads mid-day
        if 9 <= hour <= 16:
            base_pf = 0.86 + rng.uniform(-0.04, 0.03)
        elif 6 <= hour <= 8 or 17 <= hour <= 20:
            base_pf = 0.90 + rng.uniform(-0.03, 0.04)
        else:
            base_pf = 0.94 + rng.uniform(-0.02, 0.03)

        pf = max(0.82, min(0.97, round(base_pf, 3)))
        kw = round(1200 + rng.uniform(-200, 600), 1)
        kvar = round(kw * math.tan(math.acos(pf)), 1)
        kva = round(kw / pf, 1)

        readings.append({
            "timestamp": f"2025-07-15T{hour:02d}:{minute:02d}:00",
            "power_factor": pf,
            "kw": kw,
            "kvar": kvar,
            "kva": kva,
            "leading_lagging": "LAGGING",
        })
    return readings


# =============================================================================
# 11. Revenue and Financial Data
# =============================================================================


@pytest.fixture
def sample_revenue_data():
    """Create annual financial data with demand charge savings for peak shaving."""
    return {
        "facility_id": "FAC-038-US-001",
        "analysis_period": "2025",
        "currency": "USD",
        "baseline_demand_charges": {
            "annual_flat_demand_usd": Decimal("204000.00"),
            "annual_tou_demand_usd": Decimal("171000.00"),
            "annual_cp_charge_usd": Decimal("12580.00"),
            "annual_ratchet_impact_usd": Decimal("18400.00"),
            "annual_pf_penalty_usd": Decimal("8640.00"),
            "total_annual_demand_charges_usd": Decimal("414620.00"),
        },
        "peak_shaving_savings": {
            "flat_demand_savings_usd": Decimal("40800.00"),
            "tou_demand_savings_usd": Decimal("34200.00"),
            "cp_tag_reduction_savings_usd": Decimal("3400.00"),
            "ratchet_avoidance_savings_usd": Decimal("18400.00"),
            "pf_correction_savings_usd": Decimal("6480.00"),
            "total_annual_savings_usd": Decimal("103280.00"),
        },
        "bess_investment": {
            "capital_cost_usd": Decimal("275000.00"),
            "annual_om_cost_usd": Decimal("5500.00"),
            "replacement_cost_year_10_usd": Decimal("137500.00"),
            "itc_credit_usd": Decimal("82500.00"),
            "sgip_rebate_usd": Decimal("45000.00"),
            "net_capital_cost_usd": Decimal("147500.00"),
        },
        "pf_correction_investment": {
            "capacitor_bank_cost_usd": Decimal("35000.00"),
            "installation_cost_usd": Decimal("8000.00"),
            "annual_maintenance_usd": Decimal("500.00"),
        },
        "financial_metrics": {
            "simple_payback_years": Decimal("1.51"),
            "npv_15yr_usd": Decimal("685420.00"),
            "irr_pct": Decimal("0.62"),
            "lcoe_reduction_usd_per_kwh": Decimal("0.0085"),
        },
    }


# =============================================================================
# 12. Emission Factors (24-hour marginal)
# =============================================================================


@pytest.fixture
def sample_emission_factors():
    """Create 24-hour marginal emission factors for PJM grid region.

    Includes marginal and average factors varying by hour of day.
    """
    return {
        "grid_region": "PJM",
        "year": 2025,
        "unit": "kg_CO2e_per_MWh",
        "average_annual": Decimal("420.0"),
        "marginal_annual": Decimal("680.0"),
        "marginal_by_hour": {
            0: Decimal("520.0"), 1: Decimal("500.0"), 2: Decimal("490.0"),
            3: Decimal("485.0"), 4: Decimal("495.0"), 5: Decimal("530.0"),
            6: Decimal("580.0"), 7: Decimal("640.0"), 8: Decimal("700.0"),
            9: Decimal("720.0"), 10: Decimal("740.0"), 11: Decimal("760.0"),
            12: Decimal("780.0"), 13: Decimal("800.0"), 14: Decimal("820.0"),
            15: Decimal("830.0"), 16: Decimal("810.0"), 17: Decimal("780.0"),
            18: Decimal("720.0"), 19: Decimal("660.0"), 20: Decimal("600.0"),
            21: Decimal("570.0"), 22: Decimal("550.0"), 23: Decimal("530.0"),
        },
        "marginal_summer_peak": Decimal("850.0"),
        "marginal_winter_peak": Decimal("720.0"),
        "marginal_shoulder": Decimal("620.0"),
        "marginal_off_peak": Decimal("500.0"),
        "sbti_factor_scope2": Decimal("420.0"),
    }
