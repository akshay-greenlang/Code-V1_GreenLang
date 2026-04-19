# -*- coding: utf-8 -*-
"""
PACK-037 Demand Response Pack - Shared Test Fixtures (conftest.py)
====================================================================

Provides pytest fixtures for the PACK-037 test suite including:
  - Dynamic module loading via importlib (no package install needed)
  - Pack manifest and configuration fixtures
  - Sample facility profile with DR-relevant attributes
  - Load inventory (20+ loads with criticality levels)
  - 15-minute interval data (30 days)
  - DR program definitions (PJM, ERCOT, CAISO, UK)
  - DR event lifecycle data
  - DER asset inventory (battery, solar, EV, thermal, generator)
  - Baseline calculation data
  - Dispatch plan data
  - Revenue and settlement data
  - Emission factor data (marginal and average)

Fixture Categories:
  1. Paths and YAML data
  2. Configuration objects
  3. Facility profile
  4. Load inventory (20+ loads, 5 criticality levels)
  5. 15-minute interval data (30 days = 2880 points)
  6. DR program definitions
  7. DR event data
  8. DER asset inventory
  9. Baseline data (historical days)
  10. Dispatch plan
  11. Revenue and settlement data
  12. Emission factors (marginal and average)

Author:  GreenLang Platform Team (GL-TestEngineer)
Pack:    PACK-037 Demand Response
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
    "load_flexibility": "load_flexibility_engine.py",
    "dr_program": "dr_program_engine.py",
    "baseline": "baseline_engine.py",
    "dispatch_optimizer": "dispatch_optimizer_engine.py",
    "event_manager": "event_manager_engine.py",
    "der_coordinator": "der_coordinator_engine.py",
    "performance_tracker": "performance_tracker_engine.py",
    "revenue_optimizer": "revenue_optimizer_engine.py",
    "carbon_impact": "carbon_impact_engine.py",
    "dr_reporting": "dr_reporting_engine.py",
}

# Engine class names that should exist in each engine module
ENGINE_CLASSES = {
    "load_flexibility": "LoadFlexibilityEngine",
    "dr_program": "DRProgramEngine",
    "baseline": "BaselineEngine",
    "dispatch_optimizer": "DispatchOptimizerEngine",
    "event_manager": "EventManagerEngine",
    "der_coordinator": "DERCoordinatorEngine",
    "performance_tracker": "PerformanceTrackerEngine",
    "revenue_optimizer": "RevenueOptimizerEngine",
    "carbon_impact": "CarbonImpactEngine",
    "dr_reporting": "DRReportingEngine",
}

# Workflow file mapping
WORKFLOW_FILES = {
    "flexibility_assessment": "flexibility_assessment_workflow.py",
    "program_enrollment": "program_enrollment_workflow.py",
    "event_preparation": "event_preparation_workflow.py",
    "event_execution": "event_execution_workflow.py",
    "settlement": "settlement_workflow.py",
    "der_optimization": "der_optimization_workflow.py",
    "reporting": "reporting_workflow.py",
    "full_lifecycle": "full_lifecycle_workflow.py",
}

# Workflow class names
WORKFLOW_CLASSES = {
    "flexibility_assessment": "FlexibilityAssessmentWorkflow",
    "program_enrollment": "ProgramEnrollmentWorkflow",
    "event_preparation": "EventPreparationWorkflow",
    "event_execution": "EventExecutionWorkflow",
    "settlement": "SettlementWorkflow",
    "der_optimization": "DEROptimizationWorkflow",
    "reporting": "ReportingWorkflow",
    "full_lifecycle": "FullLifecycleWorkflow",
}

# Workflow expected phase counts
WORKFLOW_PHASE_COUNTS = {
    "flexibility_assessment": 4,
    "program_enrollment": 5,
    "event_preparation": 4,
    "event_execution": 5,
    "settlement": 4,
    "der_optimization": 4,
    "reporting": 3,
    "full_lifecycle": 8,
}

# Template file mapping
TEMPLATE_FILES = {
    "flexibility_report": "flexibility_report.py",
    "program_comparison_report": "program_comparison_report.py",
    "baseline_report": "baseline_report.py",
    "dispatch_plan_report": "dispatch_plan_report.py",
    "event_log_report": "event_log_report.py",
    "der_status_report": "der_status_report.py",
    "performance_dashboard": "performance_dashboard.py",
    "revenue_report": "revenue_report.py",
    "carbon_impact_report": "carbon_impact_report.py",
    "executive_summary": "executive_summary.py",
}

# Template class names
TEMPLATE_CLASSES = {
    "flexibility_report": "FlexibilityReportTemplate",
    "program_comparison_report": "ProgramComparisonReportTemplate",
    "baseline_report": "BaselineReportTemplate",
    "dispatch_plan_report": "DispatchPlanReportTemplate",
    "event_log_report": "EventLogReportTemplate",
    "der_status_report": "DERStatusReportTemplate",
    "performance_dashboard": "PerformanceDashboardTemplate",
    "revenue_report": "RevenueReportTemplate",
    "carbon_impact_report": "CarbonImpactReportTemplate",
    "executive_summary": "ExecutiveSummaryTemplate",
}

# Integration file mapping
INTEGRATION_FILES = {
    "pack_orchestrator": "pack_orchestrator.py",
    "mrv_bridge": "mrv_bridge.py",
    "data_bridge": "data_bridge.py",
    "grid_signal_bridge": "grid_signal_bridge.py",
    "pack036_bridge": "pack036_bridge.py",
    "iso_rto_bridge": "iso_rto_bridge.py",
    "scada_bridge": "scada_bridge.py",
    "bms_bridge": "bms_bridge.py",
    "der_bridge": "der_bridge.py",
    "health_check": "health_check.py",
    "setup_wizard": "setup_wizard.py",
    "alert_bridge": "alert_bridge.py",
}

# Integration class names
INTEGRATION_CLASSES = {
    "pack_orchestrator": "DemandResponseOrchestrator",
    "mrv_bridge": "MRVBridge",
    "data_bridge": "DataBridge",
    "grid_signal_bridge": "GridSignalBridge",
    "pack036_bridge": "Pack036Bridge",
    "iso_rto_bridge": "ISORTOBridge",
    "scada_bridge": "SCADABridge",
    "bms_bridge": "BMSBridge",
    "der_bridge": "DERBridge",
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

    This avoids the need to install PACK-037 as a Python package. The module
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
            f"Ensure PACK-037 source files are present."
        )

    full_module_name = f"pack037_test.{subdir}.{module_name}"

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
    """Return the absolute path to the PACK-037 root directory."""
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
    """Create a default PACK-037 configuration dictionary."""
    return {
        "pack_id": "PACK-037",
        "pack_name": "Demand Response",
        "version": "1.0.0",
        "category": "energy-efficiency",
        "environment": "test",
        "currency": "USD",
        "default_region": "US",
        "default_iso_rto": "PJM",
        "decimal_precision": 4,
        "provenance_enabled": True,
        "baseline_methodology": "HIGH_4_OF_5",
        "comfort_priority": "BALANCED",
        "max_event_duration_hours": 6,
        "min_notification_minutes": 30,
        "performance_target_pct": Decimal("0.90"),
        "discount_rate": Decimal("0.08"),
    }


# =============================================================================
# 3. Facility Profile
# =============================================================================


@pytest.fixture
def sample_facility_profile():
    """Create a sample commercial facility profile for DR participation.

    Facility: Chicago Commercial Campus
    Peak demand: 2,500 kW
    Annual consumption: 12,500 MWh
    DR-enrolled capacity: 800 kW
    """
    return {
        "facility_id": "FAC-037-US-001",
        "facility_name": "Chicago Commercial Campus",
        "facility_type": "COMMERCIAL_OFFICE",
        "address": "200 West Monroe St, Chicago, IL 60606",
        "country": "US",
        "state": "IL",
        "iso_rto": "PJM",
        "utility_zone": "ComEd",
        "climate_zone": "5A",
        "floor_area_m2": 45_000,
        "floors": 12,
        "year_built": 2010,
        "occupancy_hours": "06:00-22:00 Mon-Fri",
        "peak_demand_kw": 2500.0,
        "average_demand_kw": 1450.0,
        "minimum_demand_kw": 680.0,
        "load_factor": 0.58,
        "annual_consumption_kwh": 12_500_000,
        "annual_energy_cost_usd": Decimal("1_125_000.00"),
        "dr_enrolled_capacity_kw": 800.0,
        "dr_programs": ["PJM_ELR", "PJM_CSP"],
        "bms_system": "Honeywell_Forge",
        "scada_connected": True,
        "has_backup_generation": True,
        "backup_generator_kw": 1500.0,
        "has_energy_storage": True,
        "battery_capacity_kwh": 500.0,
        "battery_power_kw": 250.0,
        "has_solar": True,
        "solar_capacity_kw": 400.0,
        "has_ev_chargers": True,
        "ev_charger_count": 20,
        "ev_total_power_kw": 300.0,
    }


# =============================================================================
# 4. Load Inventory (20+ loads with 5 criticality levels)
# =============================================================================


@pytest.fixture
def sample_load_inventory():
    """Create a load inventory with 24 loads across 5 criticality levels.

    Criticality levels:
      1 - CRITICAL: Life safety, data, emergency (never curtail)
      2 - ESSENTIAL: Core business operations (limited curtailment)
      3 - IMPORTANT: Standard operations (moderate curtailment)
      4 - DEFERRABLE: Flexible timing (high curtailment potential)
      5 - SHEDDABLE: Discretionary (full curtailment OK)
    """
    return [
        # CRITICAL (Level 1) -- never curtail
        {"load_id": "LD-001", "name": "Fire Alarm & Sprinkler System",
         "criticality": 1, "category": "LIFE_SAFETY",
         "rated_kw": 15.0, "typical_kw": 8.0, "flexible_kw": 0.0,
         "min_notification_min": None, "max_curtail_hours": 0,
         "ramp_rate_kw_per_min": None, "rebound_factor": 1.0,
         "comfort_impact": "NONE", "process_impact": "NONE"},
        {"load_id": "LD-002", "name": "Emergency Lighting",
         "criticality": 1, "category": "LIFE_SAFETY",
         "rated_kw": 25.0, "typical_kw": 12.0, "flexible_kw": 0.0,
         "min_notification_min": None, "max_curtail_hours": 0,
         "ramp_rate_kw_per_min": None, "rebound_factor": 1.0,
         "comfort_impact": "NONE", "process_impact": "NONE"},
        {"load_id": "LD-003", "name": "Data Center / Server Room",
         "criticality": 1, "category": "IT_INFRASTRUCTURE",
         "rated_kw": 350.0, "typical_kw": 280.0, "flexible_kw": 0.0,
         "min_notification_min": None, "max_curtail_hours": 0,
         "ramp_rate_kw_per_min": None, "rebound_factor": 1.0,
         "comfort_impact": "NONE", "process_impact": "CRITICAL"},
        {"load_id": "LD-004", "name": "Elevators",
         "criticality": 1, "category": "BUILDING_SYSTEMS",
         "rated_kw": 120.0, "typical_kw": 60.0, "flexible_kw": 0.0,
         "min_notification_min": None, "max_curtail_hours": 0,
         "ramp_rate_kw_per_min": None, "rebound_factor": 1.0,
         "comfort_impact": "HIGH", "process_impact": "NONE"},

        # ESSENTIAL (Level 2) -- limited curtailment
        {"load_id": "LD-005", "name": "HVAC Chiller #1 (Primary)",
         "criticality": 2, "category": "HVAC",
         "rated_kw": 450.0, "typical_kw": 340.0, "flexible_kw": 100.0,
         "min_notification_min": 60, "max_curtail_hours": 2,
         "ramp_rate_kw_per_min": 10.0, "rebound_factor": 1.15,
         "comfort_impact": "MEDIUM", "process_impact": "LOW"},
        {"load_id": "LD-006", "name": "HVAC Chiller #2 (Secondary)",
         "criticality": 2, "category": "HVAC",
         "rated_kw": 350.0, "typical_kw": 260.0, "flexible_kw": 180.0,
         "min_notification_min": 30, "max_curtail_hours": 3,
         "ramp_rate_kw_per_min": 15.0, "rebound_factor": 1.20,
         "comfort_impact": "MEDIUM", "process_impact": "LOW"},
        {"load_id": "LD-007", "name": "HVAC Air Handling Units (AHU 1-4)",
         "criticality": 2, "category": "HVAC",
         "rated_kw": 200.0, "typical_kw": 150.0, "flexible_kw": 60.0,
         "min_notification_min": 30, "max_curtail_hours": 2,
         "ramp_rate_kw_per_min": 8.0, "rebound_factor": 1.10,
         "comfort_impact": "MEDIUM", "process_impact": "NONE"},
        {"load_id": "LD-008", "name": "Kitchen / Cafeteria",
         "criticality": 2, "category": "FOOD_SERVICE",
         "rated_kw": 80.0, "typical_kw": 45.0, "flexible_kw": 20.0,
         "min_notification_min": 60, "max_curtail_hours": 2,
         "ramp_rate_kw_per_min": 5.0, "rebound_factor": 1.05,
         "comfort_impact": "LOW", "process_impact": "MEDIUM"},

        # IMPORTANT (Level 3) -- moderate curtailment
        {"load_id": "LD-009", "name": "Office Lighting (Floors 1-6)",
         "criticality": 3, "category": "LIGHTING",
         "rated_kw": 120.0, "typical_kw": 90.0, "flexible_kw": 45.0,
         "min_notification_min": 15, "max_curtail_hours": 4,
         "ramp_rate_kw_per_min": 30.0, "rebound_factor": 1.00,
         "comfort_impact": "LOW", "process_impact": "LOW"},
        {"load_id": "LD-010", "name": "Office Lighting (Floors 7-12)",
         "criticality": 3, "category": "LIGHTING",
         "rated_kw": 120.0, "typical_kw": 90.0, "flexible_kw": 45.0,
         "min_notification_min": 15, "max_curtail_hours": 4,
         "ramp_rate_kw_per_min": 30.0, "rebound_factor": 1.00,
         "comfort_impact": "LOW", "process_impact": "LOW"},
        {"load_id": "LD-011", "name": "Domestic Hot Water",
         "criticality": 3, "category": "DHW",
         "rated_kw": 60.0, "typical_kw": 35.0, "flexible_kw": 35.0,
         "min_notification_min": 30, "max_curtail_hours": 4,
         "ramp_rate_kw_per_min": 20.0, "rebound_factor": 1.10,
         "comfort_impact": "LOW", "process_impact": "NONE"},
        {"load_id": "LD-012", "name": "Parking Garage Ventilation",
         "criticality": 3, "category": "VENTILATION",
         "rated_kw": 45.0, "typical_kw": 30.0, "flexible_kw": 20.0,
         "min_notification_min": 15, "max_curtail_hours": 4,
         "ramp_rate_kw_per_min": 15.0, "rebound_factor": 1.05,
         "comfort_impact": "NONE", "process_impact": "NONE"},
        {"load_id": "LD-013", "name": "HVAC Variable Air Volume (VAV) Boxes",
         "criticality": 3, "category": "HVAC",
         "rated_kw": 80.0, "typical_kw": 55.0, "flexible_kw": 30.0,
         "min_notification_min": 15, "max_curtail_hours": 3,
         "ramp_rate_kw_per_min": 10.0, "rebound_factor": 1.10,
         "comfort_impact": "MEDIUM", "process_impact": "NONE"},

        # DEFERRABLE (Level 4) -- high curtailment potential
        {"load_id": "LD-014", "name": "EV Chargers (Level 2, 20 units)",
         "criticality": 4, "category": "EV_CHARGING",
         "rated_kw": 300.0, "typical_kw": 180.0, "flexible_kw": 180.0,
         "min_notification_min": 10, "max_curtail_hours": 6,
         "ramp_rate_kw_per_min": 50.0, "rebound_factor": 1.25,
         "comfort_impact": "LOW", "process_impact": "NONE"},
        {"load_id": "LD-015", "name": "Ice Storage Charging",
         "criticality": 4, "category": "THERMAL_STORAGE",
         "rated_kw": 200.0, "typical_kw": 150.0, "flexible_kw": 150.0,
         "min_notification_min": 15, "max_curtail_hours": 8,
         "ramp_rate_kw_per_min": 40.0, "rebound_factor": 1.00,
         "comfort_impact": "NONE", "process_impact": "NONE"},
        {"load_id": "LD-016", "name": "Water Feature Pumps",
         "criticality": 4, "category": "AMENITY",
         "rated_kw": 25.0, "typical_kw": 20.0, "flexible_kw": 20.0,
         "min_notification_min": 5, "max_curtail_hours": 8,
         "ramp_rate_kw_per_min": 25.0, "rebound_factor": 1.00,
         "comfort_impact": "NONE", "process_impact": "NONE"},
        {"load_id": "LD-017", "name": "Pre-Cooling / Pre-Heating",
         "criticality": 4, "category": "HVAC",
         "rated_kw": 180.0, "typical_kw": 100.0, "flexible_kw": 100.0,
         "min_notification_min": 30, "max_curtail_hours": 6,
         "ramp_rate_kw_per_min": 20.0, "rebound_factor": 1.15,
         "comfort_impact": "LOW", "process_impact": "NONE"},
        {"load_id": "LD-018", "name": "Battery Charging System",
         "criticality": 4, "category": "ENERGY_STORAGE",
         "rated_kw": 250.0, "typical_kw": 120.0, "flexible_kw": 120.0,
         "min_notification_min": 5, "max_curtail_hours": 8,
         "ramp_rate_kw_per_min": 50.0, "rebound_factor": 1.00,
         "comfort_impact": "NONE", "process_impact": "NONE"},

        # SHEDDABLE (Level 5) -- full curtailment OK
        {"load_id": "LD-019", "name": "Decorative Exterior Lighting",
         "criticality": 5, "category": "LIGHTING",
         "rated_kw": 35.0, "typical_kw": 30.0, "flexible_kw": 30.0,
         "min_notification_min": 5, "max_curtail_hours": 12,
         "ramp_rate_kw_per_min": 35.0, "rebound_factor": 1.00,
         "comfort_impact": "NONE", "process_impact": "NONE"},
        {"load_id": "LD-020", "name": "Signage and Display Screens",
         "criticality": 5, "category": "SIGNAGE",
         "rated_kw": 20.0, "typical_kw": 18.0, "flexible_kw": 18.0,
         "min_notification_min": 5, "max_curtail_hours": 12,
         "ramp_rate_kw_per_min": 20.0, "rebound_factor": 1.00,
         "comfort_impact": "NONE", "process_impact": "NONE"},
        {"load_id": "LD-021", "name": "Landscape Irrigation Pumps",
         "criticality": 5, "category": "IRRIGATION",
         "rated_kw": 15.0, "typical_kw": 12.0, "flexible_kw": 12.0,
         "min_notification_min": 5, "max_curtail_hours": 24,
         "ramp_rate_kw_per_min": 15.0, "rebound_factor": 1.00,
         "comfort_impact": "NONE", "process_impact": "NONE"},
        {"load_id": "LD-022", "name": "Gym / Fitness Center Equipment",
         "criticality": 5, "category": "AMENITY",
         "rated_kw": 40.0, "typical_kw": 25.0, "flexible_kw": 25.0,
         "min_notification_min": 15, "max_curtail_hours": 8,
         "ramp_rate_kw_per_min": 20.0, "rebound_factor": 1.00,
         "comfort_impact": "LOW", "process_impact": "NONE"},
        {"load_id": "LD-023", "name": "Pool Heating System",
         "criticality": 5, "category": "AMENITY",
         "rated_kw": 50.0, "typical_kw": 40.0, "flexible_kw": 40.0,
         "min_notification_min": 30, "max_curtail_hours": 12,
         "ramp_rate_kw_per_min": 10.0, "rebound_factor": 1.05,
         "comfort_impact": "LOW", "process_impact": "NONE"},
        {"load_id": "LD-024", "name": "Non-Essential HVAC Zones (Lobby, Corridor)",
         "criticality": 5, "category": "HVAC",
         "rated_kw": 60.0, "typical_kw": 40.0, "flexible_kw": 40.0,
         "min_notification_min": 10, "max_curtail_hours": 6,
         "ramp_rate_kw_per_min": 15.0, "rebound_factor": 1.10,
         "comfort_impact": "LOW", "process_impact": "NONE"},
    ]


# =============================================================================
# 5. Interval Data (15-min, 30 days)
# =============================================================================


@pytest.fixture
def sample_interval_data():
    """Create 15-minute interval data for 30 days (June 2025).

    30 days x 96 intervals = 2880 data points.
    Simulates a commercial office load profile with summer cooling peaks.
    Seeded random for deterministic output.
    """
    rng = random.Random(42)
    intervals = []
    for day in range(1, 31):
        for interval in range(96):
            hour = interval // 4
            minute = (interval % 4) * 15
            weekday = ((day - 1) + 6) % 7  # June 1, 2025 = Sunday
            is_workday = weekday < 5

            if not is_workday:
                base = 700.0
                variation = rng.uniform(-50, 50)
            elif 0 <= hour < 6:
                base = 720.0
                variation = rng.uniform(-30, 30)
            elif 6 <= hour < 9:
                ramp_factor = (hour - 6 + minute / 60) / 3.0
                base = 720.0 + ramp_factor * 1300.0
                variation = rng.uniform(-60, 60)
            elif 9 <= hour < 17:
                base = 2000.0 + rng.uniform(-150, 350)
                variation = rng.uniform(-30, 30)
            elif 17 <= hour < 21:
                ramp_factor = 1.0 - (hour - 17 + minute / 60) / 4.0
                base = 720.0 + ramp_factor * 1280.0
                variation = rng.uniform(-50, 50)
            else:
                base = 740.0
                variation = rng.uniform(-30, 30)

            demand_kw = max(0, base + variation)
            intervals.append({
                "timestamp": f"2025-06-{day:02d}T{hour:02d}:{minute:02d}:00",
                "demand_kw": round(demand_kw, 2),
                "energy_kwh": round(demand_kw * 0.25, 2),
                "temperature_c": round(22 + 8 * math.sin(
                    math.pi * (hour - 6) / 12) if 6 <= hour <= 18 else 20, 1),
            })
    return intervals


# =============================================================================
# 6. DR Program Definitions
# =============================================================================


@pytest.fixture
def sample_dr_program():
    """Create a PJM Economic Load Response (ELR) program definition."""
    return {
        "program_id": "PJM-ELR-2025",
        "program_name": "PJM Economic Load Response",
        "iso_rto": "PJM",
        "program_type": "ECONOMIC",
        "season": "SUMMER",
        "season_start": "2025-06-01",
        "season_end": "2025-09-30",
        "min_reduction_kw": 100,
        "max_events_per_season": 10,
        "max_event_hours_per_season": 60,
        "max_event_duration_hours": 6,
        "min_notification_minutes": 60,
        "baseline_methodology": "HIGH_4_OF_5",
        "measurement_interval_minutes": 5,
        "performance_threshold_pct": Decimal("0.75"),
        "penalty_for_underperformance": True,
        "penalty_rate_usd_per_kw": Decimal("50.00"),
        "capacity_payment_usd_per_kw_year": Decimal("40.00"),
        "energy_payment_usd_per_mwh": Decimal("100.00"),
        "availability_bonus_pct": Decimal("0.10"),
        "stacking_allowed": True,
        "stacking_restrictions": ["CANNOT_STACK_WITH_CSP_SAME_HOUR"],
    }


@pytest.fixture
def sample_dr_programs():
    """Create a portfolio of 4 DR program definitions from different ISOs."""
    return [
        {
            "program_id": "PJM-ELR-2025",
            "program_name": "PJM Economic Load Response",
            "iso_rto": "PJM",
            "program_type": "ECONOMIC",
            "min_reduction_kw": 100,
            "capacity_payment_usd_per_kw_year": Decimal("40.00"),
            "energy_payment_usd_per_mwh": Decimal("100.00"),
            "baseline_methodology": "HIGH_4_OF_5",
            "max_events_per_season": 10,
            "max_event_duration_hours": 6,
        },
        {
            "program_id": "PJM-CSP-2025",
            "program_name": "PJM Capacity Service Provider",
            "iso_rto": "PJM",
            "program_type": "CAPACITY",
            "min_reduction_kw": 100,
            "capacity_payment_usd_per_kw_year": Decimal("65.00"),
            "energy_payment_usd_per_mwh": Decimal("0.00"),
            "baseline_methodology": "HIGH_4_OF_5",
            "max_events_per_season": 6,
            "max_event_duration_hours": 6,
        },
        {
            "program_id": "ERCOT-ERS-2025",
            "program_name": "ERCOT Emergency Response Service",
            "iso_rto": "ERCOT",
            "program_type": "EMERGENCY",
            "min_reduction_kw": 100,
            "capacity_payment_usd_per_kw_year": Decimal("55.00"),
            "energy_payment_usd_per_mwh": Decimal("9000.00"),
            "baseline_methodology": "10_OF_10",
            "max_events_per_season": 5,
            "max_event_duration_hours": 4,
        },
        {
            "program_id": "CAISO-PDR-2025",
            "program_name": "CAISO Proxy Demand Resource",
            "iso_rto": "CAISO",
            "program_type": "ECONOMIC",
            "min_reduction_kw": 100,
            "capacity_payment_usd_per_kw_year": Decimal("35.00"),
            "energy_payment_usd_per_mwh": Decimal("150.00"),
            "baseline_methodology": "10_OF_10",
            "max_events_per_season": 15,
            "max_event_duration_hours": 4,
        },
    ]


# =============================================================================
# 7. DR Event Data
# =============================================================================


@pytest.fixture
def sample_dr_event():
    """Create a sample DR event for testing event lifecycle."""
    return {
        "event_id": "EVT-037-001",
        "program_id": "PJM-ELR-2025",
        "facility_id": "FAC-037-US-001",
        "event_type": "ECONOMIC",
        "event_status": "SCHEDULED",
        "notification_time": "2025-07-15T11:00:00",
        "event_start": "2025-07-15T14:00:00",
        "event_end": "2025-07-15T18:00:00",
        "duration_hours": 4,
        "target_reduction_kw": 800.0,
        "baseline_kw": 2400.0,
        "dispatch_plan_id": "DISP-037-001",
        "weather_forecast": {
            "temperature_high_c": 36,
            "humidity_pct": 65,
            "heat_index_c": 42,
        },
        "grid_conditions": {
            "system_load_mw": 145_000,
            "lmp_usd_per_mwh": Decimal("185.00"),
            "reserve_margin_pct": Decimal("3.2"),
        },
    }


@pytest.fixture
def sample_dr_event_results():
    """Create sample DR event performance results."""
    return {
        "event_id": "EVT-037-001",
        "actual_reduction_kw": 780.0,
        "target_reduction_kw": 800.0,
        "performance_ratio": Decimal("0.975"),
        "compliance_status": "PASS",
        "baseline_kw": 2400.0,
        "actual_demand_kw": 1620.0,
        "measurement_intervals": [
            {"time": "14:00", "baseline_kw": 2380, "actual_kw": 1600, "reduction_kw": 780},
            {"time": "14:15", "baseline_kw": 2400, "actual_kw": 1610, "reduction_kw": 790},
            {"time": "14:30", "baseline_kw": 2420, "actual_kw": 1630, "reduction_kw": 790},
            {"time": "14:45", "baseline_kw": 2410, "actual_kw": 1640, "reduction_kw": 770},
            {"time": "15:00", "baseline_kw": 2390, "actual_kw": 1620, "reduction_kw": 770},
            {"time": "15:15", "baseline_kw": 2400, "actual_kw": 1630, "reduction_kw": 770},
            {"time": "15:30", "baseline_kw": 2410, "actual_kw": 1625, "reduction_kw": 785},
            {"time": "15:45", "baseline_kw": 2400, "actual_kw": 1620, "reduction_kw": 780},
            {"time": "16:00", "baseline_kw": 2380, "actual_kw": 1610, "reduction_kw": 770},
            {"time": "16:15", "baseline_kw": 2370, "actual_kw": 1600, "reduction_kw": 770},
            {"time": "16:30", "baseline_kw": 2360, "actual_kw": 1590, "reduction_kw": 770},
            {"time": "16:45", "baseline_kw": 2350, "actual_kw": 1580, "reduction_kw": 770},
            {"time": "17:00", "baseline_kw": 2300, "actual_kw": 1540, "reduction_kw": 760},
            {"time": "17:15", "baseline_kw": 2250, "actual_kw": 1510, "reduction_kw": 740},
            {"time": "17:30", "baseline_kw": 2200, "actual_kw": 1500, "reduction_kw": 700},
            {"time": "17:45", "baseline_kw": 2150, "actual_kw": 1480, "reduction_kw": 670},
        ],
        "rebound_kw": 120.0,
        "rebound_duration_hours": 1.5,
        "revenue_earned_usd": Decimal("2496.00"),
        "penalty_incurred_usd": Decimal("0.00"),
        "net_revenue_usd": Decimal("2496.00"),
    }


# =============================================================================
# 8. DER Asset Inventory
# =============================================================================


@pytest.fixture
def sample_der_assets():
    """Create a DER asset inventory with battery, solar, EV, thermal, generator."""
    return [
        {
            "asset_id": "DER-BAT-001",
            "asset_type": "BATTERY",
            "name": "Li-Ion Battery Storage",
            "capacity_kwh": 500.0,
            "max_power_kw": 250.0,
            "min_soc_pct": 10.0,
            "max_soc_pct": 95.0,
            "current_soc_pct": 85.0,
            "round_trip_efficiency_pct": 92.0,
            "cycle_count": 450,
            "max_cycles": 5000,
            "degradation_pct_per_1000_cycles": 2.5,
            "ramp_rate_kw_per_min": 250.0,
            "availability_pct": 98.0,
            "cost_usd_per_kwh_throughput": Decimal("0.04"),
        },
        {
            "asset_id": "DER-SOL-001",
            "asset_type": "SOLAR",
            "name": "Rooftop Solar Array",
            "capacity_kw": 400.0,
            "current_output_kw": 320.0,
            "capacity_factor_pct": 22.0,
            "inverter_efficiency_pct": 97.5,
            "azimuth_degrees": 180,
            "tilt_degrees": 15,
            "panel_degradation_pct_per_year": 0.5,
            "age_years": 3,
            "availability_pct": 99.0,
        },
        {
            "asset_id": "DER-EV-001",
            "asset_type": "EV_FLEET",
            "name": "EV Charging Station Fleet",
            "charger_count": 20,
            "total_power_kw": 300.0,
            "current_draw_kw": 180.0,
            "deferrable_kw": 180.0,
            "v2g_capable": False,
            "average_vehicle_battery_kwh": 60.0,
            "min_charge_threshold_pct": 30.0,
            "availability_pct": 95.0,
        },
        {
            "asset_id": "DER-TES-001",
            "asset_type": "THERMAL_STORAGE",
            "name": "Ice Storage System",
            "capacity_kwh_thermal": 2400.0,
            "charge_power_kw": 200.0,
            "discharge_power_kw": 300.0,
            "current_charge_pct": 70.0,
            "efficiency_pct": 85.0,
            "charging_hours": "22:00-06:00",
            "availability_pct": 97.0,
        },
        {
            "asset_id": "DER-GEN-001",
            "asset_type": "BACKUP_GENERATOR",
            "name": "Diesel Backup Generator",
            "capacity_kw": 1500.0,
            "fuel_type": "DIESEL",
            "fuel_tank_litres": 3000,
            "fuel_consumption_litres_per_hour": 120.0,
            "startup_time_minutes": 10,
            "emission_factor_kg_co2_per_litre": 2.68,
            "max_run_hours_per_event": 8,
            "annual_run_hour_limit": 200,
            "run_hours_ytd": 35,
            "noise_restriction": True,
            "availability_pct": 95.0,
        },
    ]


# =============================================================================
# 9. Baseline Data (10 historical days)
# =============================================================================


@pytest.fixture
def sample_baseline_data():
    """Create 10 historical days of load data for baseline calculation.

    Simulates 10 similar weekdays (Mon-Fri) in June 2025 for High-4-of-5.
    Each day has 96 intervals (15-min).
    """
    rng = random.Random(99)
    days = []
    for d in range(10):
        day_data = []
        date = f"2025-06-{d + 2:02d}"  # June 2-11 (Mon-Fri weeks)
        for interval in range(96):
            hour = interval // 4
            minute = (interval % 4) * 15
            if 0 <= hour < 6:
                base = 720.0
            elif 6 <= hour < 9:
                ramp = (hour - 6 + minute / 60) / 3.0
                base = 720.0 + ramp * 1300.0
            elif 9 <= hour < 17:
                base = 2000.0 + rng.uniform(-100, 300)
            elif 17 <= hour < 21:
                ramp = 1.0 - (hour - 17 + minute / 60) / 4.0
                base = 720.0 + ramp * 1280.0
            else:
                base = 730.0
            demand_kw = max(0, base + rng.uniform(-40, 40))
            day_data.append({
                "timestamp": f"{date}T{hour:02d}:{minute:02d}:00",
                "demand_kw": round(demand_kw, 2),
            })
        days.append({
            "date": date,
            "weekday": (d % 5),  # 0=Mon through 4=Fri
            "is_event_day": False,
            "temperature_high_c": 30 + rng.uniform(-3, 5),
            "intervals": day_data,
        })
    return days


# =============================================================================
# 10. Dispatch Plan
# =============================================================================


@pytest.fixture
def sample_dispatch_plan():
    """Create a dispatch plan for a 4-hour DR event."""
    return {
        "plan_id": "DISP-037-001",
        "event_id": "EVT-037-001",
        "facility_id": "FAC-037-US-001",
        "target_reduction_kw": 800.0,
        "event_start": "2025-07-15T14:00:00",
        "event_end": "2025-07-15T18:00:00",
        "pre_event_actions": [
            {"time": "13:00", "action": "PRE_COOL", "target_temp_c": 21.5,
             "load_ids": ["LD-005", "LD-006", "LD-017"]},
            {"time": "13:30", "action": "CHARGE_BATTERY", "target_soc_pct": 95,
             "load_ids": ["LD-018"]},
            {"time": "13:45", "action": "NOTIFY_OCCUPANTS",
             "message": "DR event starting at 14:00"},
        ],
        "curtailment_sequence": [
            {"phase": 1, "time": "14:00", "load_ids": ["LD-019", "LD-020", "LD-021",
             "LD-022", "LD-023", "LD-024"],
             "expected_reduction_kw": 165.0, "action": "SHED"},
            {"phase": 2, "time": "14:05", "load_ids": ["LD-014", "LD-015", "LD-016"],
             "expected_reduction_kw": 350.0, "action": "SHED"},
            {"phase": 3, "time": "14:10", "load_ids": ["LD-009", "LD-010", "LD-012",
             "LD-013"],
             "expected_reduction_kw": 140.0, "action": "CURTAIL_50PCT"},
            {"phase": 4, "time": "14:15", "load_ids": ["LD-006", "LD-007"],
             "expected_reduction_kw": 240.0, "action": "CURTAIL"},
        ],
        "der_dispatch": [
            {"asset_id": "DER-BAT-001", "action": "DISCHARGE", "power_kw": 250.0,
             "start": "14:00", "end": "16:00"},
        ],
        "total_planned_reduction_kw": 895.0,
        "reduction_margin_kw": 95.0,
        "comfort_constraints": {
            "max_temperature_rise_c": 2.0,
            "max_lighting_reduction_pct": 50,
            "critical_loads_protected": True,
        },
        "restoration_sequence": [
            {"phase": 1, "time": "18:00", "load_ids": ["LD-006", "LD-007"],
             "action": "RESTORE"},
            {"phase": 2, "time": "18:05", "load_ids": ["LD-009", "LD-010",
             "LD-012", "LD-013"], "action": "RESTORE"},
            {"phase": 3, "time": "18:10", "load_ids": ["LD-014", "LD-015",
             "LD-016"], "action": "RESTORE"},
            {"phase": 4, "time": "18:15", "load_ids": ["LD-019", "LD-020",
             "LD-021", "LD-022", "LD-023", "LD-024"], "action": "RESTORE"},
        ],
    }


# =============================================================================
# 11. Revenue and Settlement Data
# =============================================================================


@pytest.fixture
def sample_revenue_data():
    """Create revenue data for a DR season (June-September 2025)."""
    return {
        "facility_id": "FAC-037-US-001",
        "program_id": "PJM-ELR-2025",
        "season": "SUMMER_2025",
        "enrolled_capacity_kw": 800.0,
        "capacity_payment": {
            "rate_usd_per_kw_year": Decimal("40.00"),
            "pro_rata_months": 4,
            "gross_usd": Decimal("10666.67"),  # 800 * 40 * 4/12
        },
        "energy_payments": [
            {"event_id": "EVT-001", "date": "2025-06-20",
             "reduction_mwh": Decimal("3.120"),
             "rate_usd_per_mwh": Decimal("100.00"),
             "gross_usd": Decimal("312.00")},
            {"event_id": "EVT-002", "date": "2025-07-08",
             "reduction_mwh": Decimal("2.880"),
             "rate_usd_per_mwh": Decimal("185.00"),
             "gross_usd": Decimal("532.80")},
            {"event_id": "EVT-003", "date": "2025-07-15",
             "reduction_mwh": Decimal("3.120"),
             "rate_usd_per_mwh": Decimal("210.00"),
             "gross_usd": Decimal("655.20")},
            {"event_id": "EVT-004", "date": "2025-08-05",
             "reduction_mwh": Decimal("2.400"),
             "rate_usd_per_mwh": Decimal("150.00"),
             "gross_usd": Decimal("360.00")},
            {"event_id": "EVT-005", "date": "2025-08-18",
             "reduction_mwh": Decimal("3.600"),
             "rate_usd_per_mwh": Decimal("250.00"),
             "gross_usd": Decimal("900.00")},
        ],
        "total_energy_payment_usd": Decimal("2760.00"),
        "penalties": [
            {"event_id": "EVT-004", "reason": "UNDER_PERFORMANCE",
             "shortfall_kw": 50.0,
             "penalty_usd": Decimal("100.00")},
        ],
        "total_penalties_usd": Decimal("100.00"),
        "demand_charge_savings": {
            "monthly_savings": [
                {"month": "2025-06", "peak_avoided_kw": 120,
                 "rate_usd_per_kw": Decimal("12.50"),
                 "savings_usd": Decimal("1500.00")},
                {"month": "2025-07", "peak_avoided_kw": 200,
                 "rate_usd_per_kw": Decimal("12.50"),
                 "savings_usd": Decimal("2500.00")},
                {"month": "2025-08", "peak_avoided_kw": 180,
                 "rate_usd_per_kw": Decimal("12.50"),
                 "savings_usd": Decimal("2250.00")},
                {"month": "2025-09", "peak_avoided_kw": 100,
                 "rate_usd_per_kw": Decimal("12.50"),
                 "savings_usd": Decimal("1250.00")},
            ],
            "total_usd": Decimal("7500.00"),
        },
        "ancillary_services_usd": Decimal("1200.00"),
        "availability_bonus_usd": Decimal("1066.67"),
        "gross_revenue_usd": Decimal("23193.34"),
        "net_revenue_usd": Decimal("23093.34"),
        "implementation_cost_usd": Decimal("15000.00"),
        "annual_operating_cost_usd": Decimal("3000.00"),
    }


# =============================================================================
# 12. Emission Factors (Marginal and Average)
# =============================================================================


@pytest.fixture
def sample_emission_factors():
    """Create emission factors for carbon impact calculations.

    Includes marginal and average emission factors for PJM grid region,
    varying by hour of day and season.
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
