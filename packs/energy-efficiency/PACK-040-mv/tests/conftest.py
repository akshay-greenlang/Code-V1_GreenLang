# -*- coding: utf-8 -*-
"""
PACK-040 Measurement & Verification (M&V) Pack - Shared Test Fixtures (conftest.py)
=====================================================================================

Provides pytest fixtures for the PACK-040 test suite including:
  - Dynamic module loading via importlib (no package install needed)
  - Pack manifest and configuration fixtures
  - M&V project data with ECMs and measurement boundaries
  - Baseline data (3 years daily energy with weather: temp, HDD, CDD)
  - Regression data (pre-computed OLS/3P/4P/5P/TOWT results)
  - Adjustment data (routine and non-routine adjustments per IPMVP)
  - Savings data (pre/post consumption for avoided/normalized savings)
  - Uncertainty data (meter specs, model errors, sampling for FSU)
  - Weather data (temperature records with HDD/CDD series)
  - Metering data (meter specifications, calibration, sampling protocols)
  - Persistence data (multi-year savings with degradation tracking)
  - IPMVP option data (Option A/B/C/D configurations)

Fixture Categories:
  1. Paths and YAML data
  2. Configuration objects
  3. M&V project data (ECMs, boundaries)
  4. Baseline data (3 years daily with weather)
  5. Regression data (pre-computed model results)
  6. Adjustment data (routine + non-routine)
  7. Savings data (avoided, normalized, cost)
  8. Uncertainty data (measurement, model, sampling)
  9. Weather data (temperature, HDD, CDD)
  10. Metering data (specs, calibration, sampling)
  11. Persistence data (multi-year, degradation)
  12. IPMVP option data (A/B/C/D configs)

Author:  GreenLang Platform Team (GL-TestEngineer)
Pack:    PACK-040 Measurement & Verification
Date:    March 2026
"""

import importlib
import importlib.util
import hashlib
import json
import math
import random
import sys
from datetime import datetime, timedelta, date
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
    "baseline": "baseline_engine.py",
    "adjustment": "adjustment_engine.py",
    "savings": "savings_engine.py",
    "uncertainty": "uncertainty_engine.py",
    "ipmvp_option": "ipmvp_option_engine.py",
    "regression": "regression_engine.py",
    "weather": "weather_engine.py",
    "metering": "metering_engine.py",
    "persistence": "persistence_engine.py",
    "mv_reporting": "mv_reporting_engine.py",
}

# Engine class names that should exist in each engine module
ENGINE_CLASSES = {
    "baseline": "BaselineEngine",
    "adjustment": "AdjustmentEngine",
    "savings": "SavingsEngine",
    "uncertainty": "UncertaintyEngine",
    "ipmvp_option": "IPMVPOptionEngine",
    "regression": "RegressionEngine",
    "weather": "WeatherEngine",
    "metering": "MeteringEngine",
    "persistence": "PersistenceEngine",
    "mv_reporting": "MVReportingEngine",
}

# Workflow file mapping
WORKFLOW_FILES = {
    "baseline_development": "baseline_development_workflow.py",
    "mv_plan": "mv_plan_workflow.py",
    "option_selection": "option_selection_workflow.py",
    "post_installation": "post_installation_workflow.py",
    "savings_verification": "savings_verification_workflow.py",
    "annual_reporting": "annual_reporting_workflow.py",
    "persistence_tracking": "persistence_tracking_workflow.py",
    "full_mv": "full_mv_workflow.py",
}

# Workflow class names
WORKFLOW_CLASSES = {
    "baseline_development": "BaselineDevelopmentWorkflow",
    "mv_plan": "MVPlanWorkflow",
    "option_selection": "OptionSelectionWorkflow",
    "post_installation": "PostInstallationWorkflow",
    "savings_verification": "SavingsVerificationWorkflow",
    "annual_reporting": "AnnualReportingWorkflow",
    "persistence_tracking": "PersistenceTrackingWorkflow",
    "full_mv": "FullMVWorkflow",
}

# Workflow expected phase counts
WORKFLOW_PHASE_COUNTS = {
    "baseline_development": 4,
    "mv_plan": 4,
    "option_selection": 3,
    "post_installation": 4,
    "savings_verification": 4,
    "annual_reporting": 3,
    "persistence_tracking": 3,
    "full_mv": 8,
}

# Template file mapping
TEMPLATE_FILES = {
    "mv_plan_report": "mv_plan_report.py",
    "baseline_report": "baseline_report.py",
    "savings_report": "savings_report.py",
    "uncertainty_report": "uncertainty_report.py",
    "annual_mv_report": "annual_mv_report.py",
    "option_comparison_report": "option_comparison_report.py",
    "metering_plan_report": "metering_plan_report.py",
    "persistence_report": "persistence_report.py",
    "executive_summary_report": "executive_summary_report.py",
    "compliance_report": "compliance_report.py",
}

# Template class names
TEMPLATE_CLASSES = {
    "mv_plan_report": "MVPlanReportTemplate",
    "baseline_report": "BaselineReportTemplate",
    "savings_report": "SavingsReportTemplate",
    "uncertainty_report": "UncertaintyReportTemplate",
    "annual_mv_report": "AnnualMVReportTemplate",
    "option_comparison_report": "OptionComparisonReportTemplate",
    "metering_plan_report": "MeteringPlanReportTemplate",
    "persistence_report": "PersistenceReportTemplate",
    "executive_summary_report": "ExecutiveSummaryReportTemplate",
    "compliance_report": "ComplianceReportTemplate",
}

# Integration file mapping
INTEGRATION_FILES = {
    "pack_orchestrator": "pack_orchestrator.py",
    "mrv_bridge": "mrv_bridge.py",
    "data_bridge": "data_bridge.py",
    "pack031_bridge": "pack031_bridge.py",
    "pack032_bridge": "pack032_bridge.py",
    "pack033_bridge": "pack033_bridge.py",
    "pack039_bridge": "pack039_bridge.py",
    "weather_service_bridge": "weather_service_bridge.py",
    "utility_data_bridge": "utility_data_bridge.py",
    "health_check": "health_check.py",
    "setup_wizard": "setup_wizard.py",
    "alert_bridge": "alert_bridge.py",
}

# Integration class names
INTEGRATION_CLASSES = {
    "pack_orchestrator": "MVOrchestrator",
    "mrv_bridge": "MRVBridge",
    "data_bridge": "DataBridge",
    "pack031_bridge": "Pack031Bridge",
    "pack032_bridge": "Pack032Bridge",
    "pack033_bridge": "Pack033Bridge",
    "pack039_bridge": "Pack039Bridge",
    "weather_service_bridge": "WeatherServiceBridge",
    "utility_data_bridge": "UtilityDataBridge",
    "health_check": "HealthCheck",
    "setup_wizard": "SetupWizard",
    "alert_bridge": "AlertBridge",
}

# Preset names
PRESET_NAMES = [
    "commercial_office",
    "manufacturing",
    "retail_portfolio",
    "hospital",
    "university_campus",
    "government_femp",
    "esco_performance_contract",
    "portfolio_mv",
]


# =============================================================================
# Helper: Dynamic Module Loader
# =============================================================================


def _load_module(module_name: str, file_name: str, subdir: str = "engines"):
    """Load a module dynamically using importlib.util.spec_from_file_location.

    This avoids the need to install PACK-040 as a Python package. The module
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
            f"Ensure PACK-040 source files are present."
        )

    full_module_name = f"pack040_test.{subdir}.{module_name}"

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
# Helper: Seeded data generation
# =============================================================================

_RNG = random.Random(42)


def _seeded_float(low: float, high: float) -> float:
    """Return a seeded random float in [low, high)."""
    return _RNG.uniform(low, high)


def _seeded_int(low: int, high: int) -> int:
    """Return a seeded random int in [low, high]."""
    return _RNG.randint(low, high)


def _seeded_choice(seq):
    """Return a seeded random choice from sequence."""
    return _RNG.choice(seq)


def _generate_daily_series(
    start_date: date,
    num_days: int,
    base_kwh: float,
    seasonal_amplitude: float,
    noise_pct: float = 0.05,
) -> List[Dict[str, Any]]:
    """Generate a deterministic daily energy series with seasonal pattern.

    Uses a sine wave for seasonal variation (peak in summer for cooling-
    dominated buildings) plus seeded random noise.

    Args:
        start_date: First day of series.
        num_days: Number of daily records to generate.
        base_kwh: Mean daily consumption in kWh.
        seasonal_amplitude: Amplitude of seasonal sine wave (kWh).
        noise_pct: Noise as fraction of base_kwh.

    Returns:
        List of dicts with keys: date, kwh, temp_f, hdd_65, cdd_65.
    """
    rng = random.Random(42)
    records = []
    for i in range(num_days):
        d = start_date + timedelta(days=i)
        day_of_year = d.timetuple().tm_yday
        # Seasonal sine: peak around day 200 (mid-July)
        seasonal = seasonal_amplitude * math.sin(
            2 * math.pi * (day_of_year - 80) / 365
        )
        noise = rng.gauss(0, base_kwh * noise_pct)
        kwh = max(0.0, base_kwh + seasonal + noise)

        # Temperature model: average 55F, amplitude 25F, peak in July
        temp_f = 55.0 + 25.0 * math.sin(
            2 * math.pi * (day_of_year - 80) / 365
        ) + rng.gauss(0, 3.0)

        hdd_65 = max(0.0, 65.0 - temp_f)
        cdd_65 = max(0.0, temp_f - 65.0)

        records.append({
            "date": d.isoformat(),
            "kwh": round(kwh, 2),
            "temp_f": round(temp_f, 1),
            "hdd_65": round(hdd_65, 1),
            "cdd_65": round(cdd_65, 1),
        })
    return records


# =============================================================================
# 1. Path and YAML Fixtures
# =============================================================================


@pytest.fixture(scope="session")
def pack_root() -> Path:
    """Return the absolute path to the PACK-040 root directory."""
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
    return CONFIG_DIR / "demo" / "demo_config.yaml"


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
    """Create a default PACK-040 configuration dictionary."""
    return {
        "pack_id": "PACK-040",
        "pack_name": "Measurement & Verification",
        "version": "1.0.0",
        "category": "energy-efficiency",
        "environment": "test",
        "currency": "USD",
        "default_region": "US",
        "decimal_precision": 4,
        "provenance_enabled": True,
        "multi_tenant_enabled": True,
        "ipmvp_default_option": "C",
        "ashrae14_cvrmse_limit_monthly": Decimal("15.0"),
        "ashrae14_cvrmse_limit_daily": Decimal("25.0"),
        "ashrae14_cvrmse_limit_hourly": Decimal("30.0"),
        "ashrae14_nmbe_limit": Decimal("0.5"),
        "ashrae14_r_squared_min": Decimal("0.70"),
        "confidence_level": Decimal("0.90"),
        "savings_precision_target": Decimal("0.50"),
        "hdd_base_temp_f": Decimal("65.0"),
        "cdd_base_temp_f": Decimal("65.0"),
        "baseline_period_months": 12,
        "reporting_period_months": 12,
        "persistence_tracking_years": 10,
        "degradation_model": "LINEAR",
        "metering_accuracy_class": "0.5",
        "uncertainty_method": "ASHRAE14",
    }


# =============================================================================
# 3. M&V Project Data
# =============================================================================


@pytest.fixture
def mv_project_data() -> Dict[str, Any]:
    """Create a sample M&V project with ECMs, boundaries, and schedule.

    Project: Chicago Office HVAC Retrofit
    3 ECMs: chiller replacement, VFD on AHU fans, LED lighting
    Baseline: 2022 calendar year
    Post-installation: 2024 calendar year
    """
    return {
        "project_id": "MV-PRJ-001",
        "project_name": "Chicago Office HVAC Retrofit",
        "facility": {
            "facility_id": "FAC-CHI-001",
            "name": "Chicago Downtown Office",
            "address": "200 N LaSalle St, Chicago, IL 60601",
            "floor_area_sqft": 250000,
            "building_type": "COMMERCIAL_OFFICE",
            "occupancy_hours_per_week": 60,
            "year_built": 1985,
            "climate_zone": "5A",
            "weather_station": "KORD",
        },
        "ecms": [
            {
                "ecm_id": "ECM-001",
                "name": "Chiller Replacement",
                "description": "Replace 2x 500-ton centrifugal chillers with high-eff units",
                "ecm_type": "HVAC",
                "estimated_savings_kwh": 450000,
                "estimated_savings_pct": Decimal("12.5"),
                "cost_usd": Decimal("850000.00"),
                "ipmvp_option": "C",
                "installation_date": "2023-06-15",
                "useful_life_years": 20,
                "interactive_effects": True,
            },
            {
                "ecm_id": "ECM-002",
                "name": "VFD on AHU Fans",
                "description": "Install VFDs on 8 AHU supply/return fans",
                "ecm_type": "MOTORS",
                "estimated_savings_kwh": 180000,
                "estimated_savings_pct": Decimal("5.0"),
                "cost_usd": Decimal("120000.00"),
                "ipmvp_option": "B",
                "installation_date": "2023-08-01",
                "useful_life_years": 15,
                "interactive_effects": True,
            },
            {
                "ecm_id": "ECM-003",
                "name": "LED Lighting Retrofit",
                "description": "Replace T8 fluorescent with LED panels, 2500 fixtures",
                "ecm_type": "LIGHTING",
                "estimated_savings_kwh": 280000,
                "estimated_savings_pct": Decimal("7.8"),
                "cost_usd": Decimal("375000.00"),
                "ipmvp_option": "A",
                "installation_date": "2023-04-01",
                "useful_life_years": 10,
                "interactive_effects": False,
            },
        ],
        "baseline_period": {
            "start_date": "2022-01-01",
            "end_date": "2022-12-31",
            "data_granularity": "DAILY",
            "total_kwh": Decimal("3600000"),
        },
        "reporting_period": {
            "start_date": "2024-01-01",
            "end_date": "2024-12-31",
            "data_granularity": "DAILY",
        },
        "measurement_boundary": "WHOLE_BUILDING",
        "ipmvp_option": "C",
        "standards": ["IPMVP_2022", "ASHRAE_14_2014", "ISO_50015_2014"],
        "contract_term_years": 10,
        "guaranteed_savings_kwh": 800000,
        "guaranteed_savings_pct": Decimal("80.0"),
    }


# =============================================================================
# 4. Baseline Data (3 years daily with weather)
# =============================================================================


@pytest.fixture
def baseline_data() -> Dict[str, Any]:
    """Create 3 years of daily energy data with weather variables.

    Period: 2020-01-01 to 2022-12-31 (1096 days)
    Base consumption: 10,000 kWh/day
    Seasonal amplitude: 3,000 kWh/day (cooling-dominated)
    Weather: outdoor temp, HDD base 65F, CDD base 65F
    """
    start = date(2020, 1, 1)
    num_days = 1096  # 3 years (2020 is leap)
    records = _generate_daily_series(
        start_date=start,
        num_days=num_days,
        base_kwh=10000.0,
        seasonal_amplitude=3000.0,
        noise_pct=0.05,
    )

    # Add production and occupancy drivers
    rng = random.Random(42)
    for i, rec in enumerate(records):
        d = date.fromisoformat(rec["date"])
        weekday = d.weekday()
        # Production: higher on weekdays
        if weekday < 5:
            rec["production_units"] = round(rng.gauss(1000, 50), 0)
            rec["occupancy_pct"] = round(min(1.0, max(0.0, rng.gauss(0.85, 0.05))), 2)
        else:
            rec["production_units"] = round(rng.gauss(200, 30), 0)
            rec["occupancy_pct"] = round(min(1.0, max(0.0, rng.gauss(0.15, 0.05))), 2)

    return {
        "facility_id": "FAC-CHI-001",
        "period": {
            "start_date": "2020-01-01",
            "end_date": "2022-12-31",
            "num_days": num_days,
            "granularity": "DAILY",
        },
        "records": records,
        "summary": {
            "total_kwh": round(sum(r["kwh"] for r in records), 2),
            "avg_daily_kwh": round(
                sum(r["kwh"] for r in records) / len(records), 2
            ),
            "max_daily_kwh": round(max(r["kwh"] for r in records), 2),
            "min_daily_kwh": round(min(r["kwh"] for r in records), 2),
            "total_hdd": round(sum(r["hdd_65"] for r in records), 1),
            "total_cdd": round(sum(r["cdd_65"] for r in records), 1),
        },
    }


# =============================================================================
# 5. Regression Data (pre-computed model results)
# =============================================================================


@pytest.fixture
def regression_data() -> Dict[str, Any]:
    """Pre-computed regression results for validation testing.

    Contains coefficients, statistics, and predictions for OLS, 3P-cooling,
    4P, 5P, and TOWT models fitted to the baseline data.
    """
    return {
        "ols": {
            "model_type": "OLS",
            "intercept": Decimal("5432.10"),
            "coefficients": {
                "hdd_65": Decimal("85.23"),
                "cdd_65": Decimal("142.67"),
            },
            "r_squared": Decimal("0.8742"),
            "adjusted_r_squared": Decimal("0.8735"),
            "cvrmse_pct": Decimal("12.34"),
            "nmbe_pct": Decimal("0.21"),
            "n_observations": 1096,
            "p_values": {
                "intercept": Decimal("0.0000"),
                "hdd_65": Decimal("0.0000"),
                "cdd_65": Decimal("0.0000"),
            },
            "durbin_watson": Decimal("1.87"),
            "f_statistic": Decimal("3789.45"),
            "f_p_value": Decimal("0.0000"),
        },
        "three_p_cooling": {
            "model_type": "3P_COOLING",
            "balance_point_f": Decimal("62.5"),
            "base_load_kwh": Decimal("7250.00"),
            "cooling_slope": Decimal("185.40"),
            "r_squared": Decimal("0.8156"),
            "cvrmse_pct": Decimal("14.22"),
            "nmbe_pct": Decimal("-0.15"),
            "n_observations": 1096,
        },
        "four_p": {
            "model_type": "4P",
            "heating_balance_point_f": Decimal("55.0"),
            "cooling_balance_point_f": Decimal("65.0"),
            "base_load_kwh": Decimal("6800.00"),
            "heating_slope": Decimal("92.15"),
            "cooling_slope": Decimal("178.30"),
            "r_squared": Decimal("0.9012"),
            "cvrmse_pct": Decimal("10.87"),
            "nmbe_pct": Decimal("0.08"),
            "n_observations": 1096,
        },
        "five_p": {
            "model_type": "5P",
            "heating_balance_point_f": Decimal("52.0"),
            "cooling_balance_point_f": Decimal("68.0"),
            "base_load_kwh": Decimal("6500.00"),
            "heating_slope": Decimal("105.80"),
            "cooling_slope": Decimal("195.20"),
            "r_squared": Decimal("0.9234"),
            "cvrmse_pct": Decimal("9.45"),
            "nmbe_pct": Decimal("-0.05"),
            "n_observations": 1096,
        },
        "towt": {
            "model_type": "TOWT",
            "description": "Time-of-Week-and-Temperature model",
            "num_time_segments": 168,
            "temperature_knots": [40.0, 55.0, 65.0, 75.0, 90.0],
            "r_squared": Decimal("0.9456"),
            "cvrmse_pct": Decimal("7.89"),
            "nmbe_pct": Decimal("0.02"),
            "n_observations": 1096,
        },
        "best_model": "5P",
        "selection_criteria": "LOWEST_CVRMSE",
    }


# =============================================================================
# 6. Adjustment Data (routine + non-routine)
# =============================================================================


@pytest.fixture
def adjustment_data() -> Dict[str, Any]:
    """Sample routine and non-routine adjustments per IPMVP methodology.

    Routine adjustments: weather normalization, production changes, occupancy.
    Non-routine adjustments: floor area change, equipment addition, schedule.
    """
    return {
        "project_id": "MV-PRJ-001",
        "reporting_period": "2024",
        "routine_adjustments": [
            {
                "adjustment_id": "RA-001",
                "type": "WEATHER",
                "description": "Weather normalization to TMY3",
                "variable": "outdoor_temperature",
                "baseline_value": Decimal("5420"),
                "reporting_value": Decimal("5180"),
                "adjustment_kwh": Decimal("35200"),
                "method": "REGRESSION_BASED",
            },
            {
                "adjustment_id": "RA-002",
                "type": "PRODUCTION",
                "description": "Production volume normalization",
                "variable": "production_units",
                "baseline_value": Decimal("285000"),
                "reporting_value": Decimal("310000"),
                "adjustment_kwh": Decimal("-42500"),
                "method": "RATIO_BASED",
            },
            {
                "adjustment_id": "RA-003",
                "type": "OCCUPANCY",
                "description": "Occupancy schedule normalization",
                "variable": "occupied_hours",
                "baseline_value": Decimal("3120"),
                "reporting_value": Decimal("2980"),
                "adjustment_kwh": Decimal("18700"),
                "method": "REGRESSION_BASED",
            },
            {
                "adjustment_id": "RA-004",
                "type": "OPERATING_HOURS",
                "description": "Operating hours normalization",
                "variable": "operating_hours",
                "baseline_value": Decimal("4380"),
                "reporting_value": Decimal("4520"),
                "adjustment_kwh": Decimal("-28400"),
                "method": "RATIO_BASED",
            },
        ],
        "non_routine_adjustments": [
            {
                "adjustment_id": "NRA-001",
                "type": "FLOOR_AREA",
                "description": "4th floor expansion (10,000 sqft added)",
                "effective_date": "2024-03-01",
                "adjustment_kwh": Decimal("-85000"),
                "calculation_method": "ENGINEERING_ESTIMATE",
                "documentation": "Floor expansion permit #2024-0312",
            },
            {
                "adjustment_id": "NRA-002",
                "type": "EQUIPMENT",
                "description": "New data center rack (50 kW IT load)",
                "effective_date": "2024-06-15",
                "adjustment_kwh": Decimal("-219000"),
                "calculation_method": "DIRECT_MEASUREMENT",
                "documentation": "Submeter M-DC-001 readings",
            },
            {
                "adjustment_id": "NRA-003",
                "type": "SCHEDULE",
                "description": "Shift from 5-day to 6-day operation",
                "effective_date": "2024-09-01",
                "adjustment_kwh": Decimal("-120000"),
                "calculation_method": "ENGINEERING_ESTIMATE",
                "documentation": "HR schedule change notice #SC-2024-09",
            },
            {
                "adjustment_id": "NRA-004",
                "type": "STATIC_FACTOR",
                "description": "Tariff structure change affecting load shape",
                "effective_date": "2024-01-01",
                "adjustment_kwh": Decimal("0"),
                "calculation_method": "NOT_APPLICABLE",
                "documentation": "Utility tariff rider update 2024",
            },
        ],
        "total_routine_adjustment_kwh": Decimal("-17000"),
        "total_non_routine_adjustment_kwh": Decimal("-424000"),
        "net_adjustment_kwh": Decimal("-441000"),
    }


# =============================================================================
# 7. Savings Data (avoided, normalized, cost)
# =============================================================================


@pytest.fixture
def savings_data() -> Dict[str, Any]:
    """Pre/post consumption data for savings calculations.

    Baseline: 3,600,000 kWh/year
    Reporting: 2,750,000 kWh/year (actual metered)
    Multiple savings types: avoided energy, normalized, cost, demand.
    """
    return {
        "project_id": "MV-PRJ-001",
        "baseline_consumption_kwh": Decimal("3600000"),
        "reporting_consumption_kwh": Decimal("2750000"),
        "adjusted_baseline_kwh": Decimal("3540000"),
        "routine_adjustments_kwh": Decimal("-17000"),
        "non_routine_adjustments_kwh": Decimal("-424000"),
        "avoided_energy_kwh": Decimal("349000"),
        "normalized_savings_kwh": Decimal("850000"),
        "cost_data": {
            "baseline_energy_rate_usd_kwh": Decimal("0.0850"),
            "reporting_energy_rate_usd_kwh": Decimal("0.0920"),
            "baseline_demand_rate_usd_kw": Decimal("12.50"),
            "reporting_demand_rate_usd_kw": Decimal("13.00"),
            "baseline_annual_cost_usd": Decimal("306000.00"),
            "reporting_annual_cost_usd": Decimal("253000.00"),
            "avoided_cost_usd": Decimal("29665.00"),
            "normalized_cost_savings_usd": Decimal("72250.00"),
        },
        "demand_data": {
            "baseline_peak_kw": Decimal("2200"),
            "reporting_peak_kw": Decimal("1850"),
            "demand_savings_kw": Decimal("350"),
            "demand_savings_pct": Decimal("15.9"),
        },
        "cumulative_savings": [
            {"year": 2024, "kwh": Decimal("850000"), "cumulative_kwh": Decimal("850000")},
            {"year": 2025, "kwh": Decimal("835000"), "cumulative_kwh": Decimal("1685000")},
            {"year": 2026, "kwh": Decimal("820000"), "cumulative_kwh": Decimal("2505000")},
        ],
        "annualized_savings_kwh": Decimal("850000"),
        "savings_pct_of_baseline": Decimal("23.6"),
    }


# =============================================================================
# 8. Uncertainty Data (measurement, model, sampling)
# =============================================================================


@pytest.fixture
def uncertainty_data() -> Dict[str, Any]:
    """Meter specifications, model errors, and sampling data for uncertainty.

    Follows ASHRAE Guideline 14 fractional savings uncertainty (FSU)
    calculation methodology with measurement, model, and sampling components.
    """
    return {
        "project_id": "MV-PRJ-001",
        "confidence_level": Decimal("0.90"),
        "t_statistic_90": Decimal("1.645"),
        "t_statistic_95": Decimal("1.960"),
        "t_statistic_99": Decimal("2.576"),
        "measurement_uncertainty": {
            "meters": [
                {
                    "meter_id": "M-MAIN-001",
                    "meter_type": "REVENUE",
                    "accuracy_class": "0.2",
                    "accuracy_pct": Decimal("0.2"),
                    "ct_accuracy_pct": Decimal("0.3"),
                    "pt_accuracy_pct": Decimal("0.1"),
                    "combined_meter_uncertainty_pct": Decimal("0.374"),
                    "last_calibration": "2023-11-15",
                    "calibration_interval_months": 12,
                },
                {
                    "meter_id": "M-HVAC-001",
                    "meter_type": "SUBMETER",
                    "accuracy_class": "0.5",
                    "accuracy_pct": Decimal("0.5"),
                    "ct_accuracy_pct": Decimal("0.5"),
                    "pt_accuracy_pct": Decimal("0.0"),
                    "combined_meter_uncertainty_pct": Decimal("0.707"),
                    "last_calibration": "2024-01-20",
                    "calibration_interval_months": 12,
                },
            ],
            "total_measurement_uncertainty_pct": Decimal("0.82"),
        },
        "model_uncertainty": {
            "cvrmse_pct": Decimal("9.45"),
            "nmbe_pct": Decimal("-0.05"),
            "n_baseline": 365,
            "n_reporting": 365,
            "model_uncertainty_pct": Decimal("4.23"),
            "autocorrelation_factor": Decimal("1.15"),
        },
        "sampling_uncertainty": {
            "population_size": 2500,
            "sample_size": 120,
            "cv_pct": Decimal("45.0"),
            "sampling_uncertainty_pct": Decimal("7.12"),
            "method": "STRATIFIED_RANDOM",
        },
        "combined_fsu": {
            "fsu_pct_90": Decimal("13.45"),
            "fsu_pct_95": Decimal("16.02"),
            "fsu_pct_99": Decimal("21.05"),
            "fsu_absolute_kwh_90": Decimal("114325"),
            "savings_kwh": Decimal("850000"),
            "savings_significant": True,
        },
        "minimum_detectable_savings": {
            "mds_kwh_90": Decimal("152000"),
            "mds_pct_of_baseline_90": Decimal("4.22"),
            "mds_kwh_95": Decimal("181000"),
            "mds_pct_of_baseline_95": Decimal("5.03"),
        },
        "ashrae14_compliance": {
            "cvrmse_pass": True,
            "nmbe_pass": True,
            "r_squared_pass": True,
            "overall_pass": True,
        },
    }


# =============================================================================
# 9. Weather Data (temperature, HDD, CDD)
# =============================================================================


@pytest.fixture
def weather_data() -> Dict[str, Any]:
    """Temperature records with HDD/CDD for baseline and reporting periods.

    Station: KORD (Chicago O'Hare)
    Baseline: 2022 calendar year (365 days)
    Reporting: 2024 calendar year (366 days - leap year)
    Includes TMY3 normal-year data.
    """
    rng = random.Random(42)

    def _make_weather_year(year: int, num_days: int) -> List[Dict]:
        records = []
        for i in range(num_days):
            d = date(year, 1, 1) + timedelta(days=i)
            day_of_year = d.timetuple().tm_yday
            temp_f = 55.0 + 25.0 * math.sin(
                2 * math.pi * (day_of_year - 80) / 365
            ) + rng.gauss(0, 3.0)
            records.append({
                "date": d.isoformat(),
                "temp_avg_f": round(temp_f, 1),
                "temp_max_f": round(temp_f + rng.uniform(5, 15), 1),
                "temp_min_f": round(temp_f - rng.uniform(5, 15), 1),
                "hdd_65": round(max(0.0, 65.0 - temp_f), 1),
                "cdd_65": round(max(0.0, temp_f - 65.0), 1),
                "humidity_pct": round(rng.uniform(30, 85), 1),
                "wind_speed_mph": round(rng.uniform(2, 25), 1),
            })
        return records

    baseline_records = _make_weather_year(2022, 365)
    reporting_records = _make_weather_year(2024, 366)
    tmy_records = _make_weather_year(2000, 365)  # TMY placeholder year

    return {
        "station_id": "KORD",
        "station_name": "Chicago O'Hare International Airport",
        "latitude": 41.9742,
        "longitude": -87.9073,
        "elevation_ft": 672,
        "baseline_weather": {
            "year": 2022,
            "records": baseline_records,
            "total_hdd_65": round(sum(r["hdd_65"] for r in baseline_records), 1),
            "total_cdd_65": round(sum(r["cdd_65"] for r in baseline_records), 1),
            "avg_temp_f": round(
                sum(r["temp_avg_f"] for r in baseline_records) / len(baseline_records),
                1,
            ),
        },
        "reporting_weather": {
            "year": 2024,
            "records": reporting_records,
            "total_hdd_65": round(sum(r["hdd_65"] for r in reporting_records), 1),
            "total_cdd_65": round(sum(r["cdd_65"] for r in reporting_records), 1),
            "avg_temp_f": round(
                sum(r["temp_avg_f"] for r in reporting_records) / len(reporting_records),
                1,
            ),
        },
        "tmy_weather": {
            "source": "TMY3",
            "records": tmy_records,
            "total_hdd_65": round(sum(r["hdd_65"] for r in tmy_records), 1),
            "total_cdd_65": round(sum(r["cdd_65"] for r in tmy_records), 1),
        },
        "balance_point_analysis": {
            "optimal_heating_bp_f": Decimal("55.0"),
            "optimal_cooling_bp_f": Decimal("65.0"),
            "method": "GRID_SEARCH",
            "search_range_f": [40, 80],
            "step_size_f": Decimal("0.5"),
        },
    }


# =============================================================================
# 10. Metering Data (specs, calibration, sampling)
# =============================================================================


@pytest.fixture
def metering_data() -> Dict[str, Any]:
    """Meter specifications, calibration records, and sampling protocols.

    Includes revenue meters, submeters, and portable loggers for IPMVP
    Options A through D.
    """
    return {
        "project_id": "MV-PRJ-001",
        "meters": [
            {
                "meter_id": "M-MAIN-001",
                "name": "Main Revenue Meter",
                "type": "REVENUE",
                "measurement": "ELECTRICITY",
                "accuracy_class": "0.2",
                "accuracy_pct": Decimal("0.2"),
                "make": "Schneider Electric",
                "model": "ION9000",
                "serial_number": "SE-ION9-2021-0042",
                "ct_ratio": "2000:5",
                "ct_accuracy_pct": Decimal("0.3"),
                "installation_date": "2021-06-15",
                "last_calibration": "2023-11-15",
                "next_calibration": "2024-11-15",
                "calibration_interval_months": 12,
                "data_interval_minutes": 15,
                "communication": "Modbus_TCP",
                "ipmvp_option": "C",
            },
            {
                "meter_id": "M-HVAC-001",
                "name": "HVAC Submeter",
                "type": "SUBMETER",
                "measurement": "ELECTRICITY",
                "accuracy_class": "0.5",
                "accuracy_pct": Decimal("0.5"),
                "make": "Dent Instruments",
                "model": "ELITEpro XC",
                "serial_number": "DENT-EPXC-2023-0188",
                "ct_ratio": "600:5",
                "ct_accuracy_pct": Decimal("0.5"),
                "installation_date": "2023-01-10",
                "last_calibration": "2024-01-20",
                "next_calibration": "2025-01-20",
                "calibration_interval_months": 12,
                "data_interval_minutes": 15,
                "communication": "BACnet_IP",
                "ipmvp_option": "B",
            },
            {
                "meter_id": "M-LTG-001",
                "name": "Lighting Logger",
                "type": "PORTABLE_LOGGER",
                "measurement": "ELECTRICITY",
                "accuracy_class": "1.0",
                "accuracy_pct": Decimal("1.0"),
                "make": "HOBO",
                "model": "UX120-006M",
                "serial_number": "HOBO-UX120-2023-0456",
                "ct_ratio": "N/A",
                "ct_accuracy_pct": Decimal("1.5"),
                "installation_date": "2023-03-15",
                "last_calibration": "2023-03-15",
                "next_calibration": "2024-03-15",
                "calibration_interval_months": 12,
                "data_interval_minutes": 1,
                "communication": "USB",
                "ipmvp_option": "A",
            },
        ],
        "calibration_records": [
            {
                "record_id": "CAL-001",
                "meter_id": "M-MAIN-001",
                "calibration_date": "2023-11-15",
                "technician": "John Smith, PE",
                "company": "Calibration Services Inc.",
                "certificate_number": "CS-2023-4567",
                "as_found_error_pct": Decimal("0.15"),
                "as_left_error_pct": Decimal("0.08"),
                "pass_fail": "PASS",
                "next_due": "2024-11-15",
            },
            {
                "record_id": "CAL-002",
                "meter_id": "M-HVAC-001",
                "calibration_date": "2024-01-20",
                "technician": "Jane Doe, CEM",
                "company": "Energy Metering Solutions",
                "certificate_number": "EMS-2024-0892",
                "as_found_error_pct": Decimal("0.42"),
                "as_left_error_pct": Decimal("0.18"),
                "pass_fail": "PASS",
                "next_due": "2025-01-20",
            },
        ],
        "sampling_protocol": {
            "option_a_sampling": {
                "population": "2500 LED fixtures",
                "sample_size": 120,
                "confidence_level": Decimal("90"),
                "precision_pct": Decimal("10"),
                "method": "STRATIFIED_RANDOM",
                "strata": [
                    {"stratum": "Open Office", "count": 1500, "sample": 72},
                    {"stratum": "Private Office", "count": 600, "sample": 29},
                    {"stratum": "Common Area", "count": 400, "sample": 19},
                ],
                "cv_pct": Decimal("45.0"),
            },
            "required_sample_formula": "n = (z * CV / e)^2",
        },
    }


# =============================================================================
# 11. Persistence Data (multi-year, degradation)
# =============================================================================


@pytest.fixture
def persistence_data() -> Dict[str, Any]:
    """Multi-year savings persistence with degradation tracking.

    10-year tracking period with linear degradation at 1.5% per year.
    Includes re-commissioning triggers and guarantee tracking.
    """
    base_savings = Decimal("850000")
    degradation_rate = Decimal("0.015")
    years = []
    cumulative = Decimal("0")
    for yr in range(10):
        factor = Decimal("1.0") - degradation_rate * Decimal(str(yr))
        annual = (base_savings * factor).quantize(Decimal("1"))
        cumulative += annual
        years.append({
            "year": 2024 + yr,
            "year_number": yr + 1,
            "persistence_factor": float(factor),
            "expected_savings_kwh": annual,
            "actual_savings_kwh": (annual * Decimal("0.97")).quantize(Decimal("1"))
            if yr < 3 else None,
            "cumulative_savings_kwh": cumulative,
            "degradation_pct": float(degradation_rate * Decimal(str(yr)) * 100),
            "verified": yr < 3,
        })

    return {
        "project_id": "MV-PRJ-001",
        "contract_term_years": 10,
        "initial_savings_kwh": base_savings,
        "degradation_model": "LINEAR",
        "annual_degradation_rate_pct": Decimal("1.5"),
        "years": years,
        "re_commissioning_triggers": {
            "savings_shortfall_threshold_pct": Decimal("15.0"),
            "consecutive_months_below_target": 3,
            "equipment_fault_count_threshold": 5,
        },
        "guarantee_tracking": {
            "guaranteed_annual_savings_kwh": Decimal("800000"),
            "guaranteed_pct_of_baseline": Decimal("22.2"),
            "shortfall_penalty_rate_usd_kwh": Decimal("0.08"),
            "surplus_sharing_pct": Decimal("25.0"),
            "years_in_compliance": 3,
            "total_shortfall_kwh": Decimal("0"),
            "total_surplus_kwh": Decimal("135000"),
        },
    }


# =============================================================================
# 12. IPMVP Option Data (A/B/C/D configs)
# =============================================================================


@pytest.fixture
def ipmvp_options() -> Dict[str, Any]:
    """IPMVP Options A through D configuration and applicability data.

    Each option includes description, applicability criteria, required
    metering, and example ECM mapping.
    """
    return {
        "option_a": {
            "name": "Retrofit Isolation: Key Parameter Measurement",
            "code": "A",
            "description": (
                "Savings determined by field measurement of key parameters "
                "that define energy use of the affected system. Parameters "
                "not measured are estimated."
            ),
            "applicability": [
                "Single ECM with well-defined parameters",
                "Low interactive effects",
                "Key parameters can be measured",
                "Stipulated values acceptable for remaining parameters",
            ],
            "metering_requirements": {
                "type": "SPOT_OR_SHORT_TERM",
                "duration": "1-2 weeks typical",
                "meters_needed": 1,
                "accuracy_class_min": "1.0",
            },
            "example_ecms": ["LIGHTING", "CONSTANT_SPEED_MOTOR", "INSULATION"],
            "cost_level": "LOW",
            "complexity": "LOW",
            "uncertainty_range_pct": "15-30",
        },
        "option_b": {
            "name": "Retrofit Isolation: All Parameter Measurement",
            "code": "B",
            "description": (
                "Savings determined by field measurement of all parameters "
                "that define energy use of the affected system."
            ),
            "applicability": [
                "Single ECM or isolated systems",
                "All parameters can be measured",
                "Higher accuracy needed than Option A",
                "Continuous metering justified",
            ],
            "metering_requirements": {
                "type": "CONTINUOUS",
                "duration": "Full reporting period",
                "meters_needed": 2,
                "accuracy_class_min": "0.5",
            },
            "example_ecms": ["VFD", "CHILLER", "BOILER", "COMPRESSED_AIR"],
            "cost_level": "MEDIUM",
            "complexity": "MEDIUM",
            "uncertainty_range_pct": "5-15",
        },
        "option_c": {
            "name": "Whole Facility",
            "code": "C",
            "description": (
                "Savings determined by measuring whole-facility energy use "
                "with regression analysis to normalize for independent variables."
            ),
            "applicability": [
                "Multiple interactive ECMs",
                "Whole-building approach preferred",
                "Baseline regression model achievable",
                "Expected savings > 10% of baseline",
            ],
            "metering_requirements": {
                "type": "UTILITY_METER",
                "duration": "12+ months baseline, 12+ months reporting",
                "meters_needed": 1,
                "accuracy_class_min": "0.2",
            },
            "example_ecms": ["HVAC_SYSTEM", "BUILDING_ENVELOPE", "CONTROLS", "MULTIPLE_ECMS"],
            "cost_level": "LOW_TO_MEDIUM",
            "complexity": "MEDIUM_TO_HIGH",
            "uncertainty_range_pct": "5-20",
        },
        "option_d": {
            "name": "Calibrated Simulation",
            "code": "D",
            "description": (
                "Savings determined through simulation of facility energy "
                "use, calibrated to actual utility billing data."
            ),
            "applicability": [
                "New construction or major renovation",
                "No baseline data available",
                "Complex interactive effects",
                "Simulation tools available",
            ],
            "metering_requirements": {
                "type": "UTILITY_METER_PLUS_SPOT",
                "duration": "12+ months for calibration",
                "meters_needed": 3,
                "accuracy_class_min": "0.5",
            },
            "example_ecms": ["NEW_CONSTRUCTION", "MAJOR_RENOVATION", "COMPLEX_RETROFIT"],
            "cost_level": "HIGH",
            "complexity": "HIGH",
            "uncertainty_range_pct": "10-30",
        },
        "selection_matrix": {
            "LIGHTING": "A",
            "CONSTANT_SPEED_MOTOR": "A",
            "VFD": "B",
            "CHILLER": "B",
            "HVAC_SYSTEM": "C",
            "MULTIPLE_ECMS": "C",
            "BUILDING_ENVELOPE": "C",
            "NEW_CONSTRUCTION": "D",
            "MAJOR_RENOVATION": "D",
            "BOILER": "B",
            "COMPRESSED_AIR": "B",
            "CONTROLS": "C",
        },
    }


# =============================================================================
# Composite / convenience fixtures
# =============================================================================


@pytest.fixture
def sample_enpi_data(baseline_data) -> Dict[str, Any]:
    """Simplified EnPI data extracted from baseline for engine tests."""
    records = baseline_data["records"][:365]
    return {
        "periods": [
            {
                "date": r["date"],
                "energy_kwh": r["kwh"],
                "hdd": r["hdd_65"],
                "cdd": r["cdd_65"],
                "production_units": r.get("production_units", 0),
                "occupancy_pct": r.get("occupancy_pct", 0.85),
                "operating_hours": 16 if date.fromisoformat(r["date"]).weekday() < 5 else 8,
                "outdoor_temp_avg_c": round((r["temp_f"] - 32) * 5 / 9, 1),
            }
            for r in records
        ],
        "baseline_total_kwh": round(sum(r["kwh"] for r in records), 2),
        "driver_primary": "hdd",
    }


@pytest.fixture
def full_mv_context(
    mv_project_data,
    baseline_data,
    regression_data,
    adjustment_data,
    savings_data,
    uncertainty_data,
    weather_data,
    metering_data,
    persistence_data,
    ipmvp_options,
) -> Dict[str, Any]:
    """Aggregate fixture combining all M&V data for integration tests."""
    return {
        "project": mv_project_data,
        "baseline": baseline_data,
        "regression": regression_data,
        "adjustments": adjustment_data,
        "savings": savings_data,
        "uncertainty": uncertainty_data,
        "weather": weather_data,
        "metering": metering_data,
        "persistence": persistence_data,
        "ipmvp_options": ipmvp_options,
    }
