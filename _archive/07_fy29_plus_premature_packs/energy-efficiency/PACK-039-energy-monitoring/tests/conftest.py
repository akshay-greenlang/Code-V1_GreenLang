# -*- coding: utf-8 -*-
"""
PACK-039 Energy Monitoring Pack - Shared Test Fixtures (conftest.py)
====================================================================

Provides pytest fixtures for the PACK-039 test suite including:
  - Dynamic module loading via importlib (no package install needed)
  - Pack manifest and configuration fixtures
  - Sample facility profile (Chicago Commercial Building, 20 submeters)
  - Meter registry (20 meters: 1 revenue + 4 check + 15 submeters)
  - Interval data (2,880 readings: 30 days x 96 intervals/day, seeded)
  - Anomaly data (2,880 readings with 5 injected anomalies)
  - EnPI data (12 months with HDD/CDD/production drivers)
  - Tariff structure (multi-component: energy, demand, TOU, ratchet)
  - Tenant accounts (4 tenants with allocation percentages)
  - Budget (annual with 12 monthly periods)
  - Alarm rules (10 rules across 5 categories)
  - Emission factors (24-hour marginal for PJM)

Fixture Categories:
  1. Paths and YAML data
  2. Configuration objects
  3. Facility profile
  4. Meter registry (20 meters)
  5. Interval data (30 days x 96 intervals)
  6. Anomaly data (with 5 injected anomalies)
  7. EnPI data (12 months + drivers)
  8. Tariff structure (multi-component)
  9. Tenant accounts (4 tenants)
  10. Budget (annual, 12 months)
  11. Alarm rules (10 rules, 5 categories)
  12. Emission factors (24-hour marginal)

Author:  GreenLang Platform Team (GL-TestEngineer)
Pack:    PACK-039 Energy Monitoring
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

# Engine file mapping: logical name -> file name on disk
ENGINE_FILES = {
    "meter_registry": "meter_registry_engine.py",
    "data_acquisition": "data_acquisition_engine.py",
    "data_validation": "data_validation_engine.py",
    "anomaly_detection": "anomaly_detection_engine.py",
    "enpi": "enpi_engine.py",
    "cost_allocation": "cost_allocation_engine.py",
    "budget": "budget_engine.py",
    "alarm": "alarm_engine.py",
    "dashboard": "dashboard_engine.py",
    "monitoring_reporting": "monitoring_reporting_engine.py",
}

# Engine class names that should exist in each engine module
ENGINE_CLASSES = {
    "meter_registry": "MeterRegistryEngine",
    "data_acquisition": "DataAcquisitionEngine",
    "data_validation": "DataValidationEngine",
    "anomaly_detection": "AnomalyDetectionEngine",
    "enpi": "EnPIEngine",
    "cost_allocation": "CostAllocationEngine",
    "budget": "BudgetEngine",
    "alarm": "AlarmEngine",
    "dashboard": "DashboardEngine",
    "monitoring_reporting": "MonitoringReportingEngine",
}

# Workflow file mapping
WORKFLOW_FILES = {
    "meter_setup": "meter_setup_workflow.py",
    "data_collection": "data_collection_workflow.py",
    "anomaly_response": "anomaly_response_workflow.py",
    "enpi_tracking": "enpi_tracking_workflow.py",
    "cost_allocation": "cost_allocation_workflow.py",
    "budget_review": "budget_review_workflow.py",
    "reporting": "reporting_workflow.py",
    "full_monitoring": "full_monitoring_workflow.py",
}

# Workflow class names
WORKFLOW_CLASSES = {
    "meter_setup": "MeterSetupWorkflow",
    "data_collection": "DataCollectionWorkflow",
    "anomaly_response": "AnomalyResponseWorkflow",
    "enpi_tracking": "EnPITrackingWorkflow",
    "cost_allocation": "CostAllocationWorkflow",
    "budget_review": "BudgetReviewWorkflow",
    "reporting": "ReportingWorkflow",
    "full_monitoring": "FullMonitoringWorkflow",
}

# Workflow expected phase counts
WORKFLOW_PHASE_COUNTS = {
    "meter_setup": 4,
    "data_collection": 4,
    "anomaly_response": 3,
    "enpi_tracking": 4,
    "cost_allocation": 3,
    "budget_review": 3,
    "reporting": 3,
    "full_monitoring": 8,
}

# Template file mapping
TEMPLATE_FILES = {
    "meter_inventory_report": "meter_inventory_report.py",
    "energy_consumption_report": "energy_consumption_report.py",
    "anomaly_report": "anomaly_report.py",
    "enpi_performance_report": "enpi_performance_report.py",
    "cost_allocation_report": "cost_allocation_report.py",
    "budget_variance_report": "budget_variance_report.py",
    "alarm_summary_report": "alarm_summary_report.py",
    "utility_bill_report": "utility_bill_report.py",
    "executive_summary_report": "executive_summary_report.py",
    "iso50001_compliance_report": "iso50001_compliance_report.py",
}

# Template class names
TEMPLATE_CLASSES = {
    "meter_inventory_report": "MeterInventoryReportTemplate",
    "energy_consumption_report": "EnergyConsumptionReportTemplate",
    "anomaly_report": "AnomalyReportTemplate",
    "enpi_performance_report": "EnPIPerformanceReportTemplate",
    "cost_allocation_report": "CostAllocationReportTemplate",
    "budget_variance_report": "BudgetVarianceReportTemplate",
    "alarm_summary_report": "AlarmSummaryReportTemplate",
    "utility_bill_report": "UtilityBillReportTemplate",
    "executive_summary_report": "ExecutiveSummaryReportTemplate",
    "iso50001_compliance_report": "ISO50001ComplianceReportTemplate",
}

# Integration file mapping
INTEGRATION_FILES = {
    "pack_orchestrator": "pack_orchestrator.py",
    "mrv_bridge": "mrv_bridge.py",
    "data_bridge": "data_bridge.py",
    "meter_protocol_bridge": "meter_protocol_bridge.py",
    "ami_bridge": "ami_bridge.py",
    "bms_bridge": "bms_bridge.py",
    "iot_sensor_bridge": "iot_sensor_bridge.py",
    "pack036_bridge": "pack036_bridge.py",
    "pack038_bridge": "pack038_bridge.py",
    "health_check": "health_check.py",
    "setup_wizard": "setup_wizard.py",
    "alert_bridge": "alert_bridge.py",
}

# Integration class names
INTEGRATION_CLASSES = {
    "pack_orchestrator": "MonitoringOrchestrator",
    "mrv_bridge": "MRVBridge",
    "data_bridge": "DataBridge",
    "meter_protocol_bridge": "MeterProtocolBridge",
    "ami_bridge": "AMIBridge",
    "bms_bridge": "BMSBridge",
    "iot_sensor_bridge": "IoTSensorBridge",
    "pack036_bridge": "Pack036Bridge",
    "pack038_bridge": "Pack038Bridge",
    "health_check": "HealthCheck",
    "setup_wizard": "SetupWizard",
    "alert_bridge": "AlertBridge",
}

# Preset names
PRESET_NAMES = [
    "commercial_office",
    "manufacturing",
    "retail_chain",
    "hospital",
    "university_campus",
    "data_center",
    "multi_tenant",
    "industrial_process",
]


# =============================================================================
# Helper: Dynamic Module Loader
# =============================================================================


def _load_module(module_name: str, file_name: str, subdir: str = "engines"):
    """Load a module dynamically using importlib.util.spec_from_file_location.

    This avoids the need to install PACK-039 as a Python package. The module
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
            f"Ensure PACK-039 source files are present."
        )

    full_module_name = f"pack039_test.{subdir}.{module_name}"

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
    """Return the absolute path to the PACK-039 root directory."""
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
    """Create a default PACK-039 configuration dictionary."""
    return {
        "pack_id": "PACK-039",
        "pack_name": "Energy Monitoring",
        "version": "1.0.0",
        "category": "energy-efficiency",
        "environment": "test",
        "currency": "USD",
        "default_region": "US",
        "decimal_precision": 4,
        "provenance_enabled": True,
        "data_interval_minutes": 15,
        "validation_checks_enabled": True,
        "anomaly_detection_method": "COMBINED",
        "enpi_normalization": "REGRESSION",
        "alarm_standard": "ISA_18_2",
        "multi_tenant_enabled": True,
        "weather_normalization": True,
        "z_score_threshold": Decimal("3.0"),
        "cusum_h_factor": Decimal("5.0"),
        "ewma_lambda": Decimal("0.20"),
        "min_data_quality_score": Decimal("0.85"),
        "budget_variance_alert_pct": Decimal("0.10"),
    }


# =============================================================================
# 3. Facility Profile
# =============================================================================


@pytest.fixture
def sample_facility():
    """Create a sample commercial building profile for energy monitoring.

    Facility: Chicago Commercial Building with 20 submeters
    Peak demand: 2,000 kW
    Annual consumption: 10,000 MWh
    20 submetered end-uses across 4 floors
    """
    return {
        "facility_id": "FAC-039-US-001",
        "facility_name": "Chicago Commercial Building",
        "facility_type": "COMMERCIAL_OFFICE",
        "address": "233 South Wacker Dr, Chicago, IL 60606",
        "country": "US",
        "state": "IL",
        "climate_zone": "5A",
        "floor_area_m2": 38_000,
        "floors": 4,
        "year_built": 2008,
        "occupancy_hours": "07:00-21:00 Mon-Fri",
        "peak_demand_kw": 2000.0,
        "average_demand_kw": 1140.0,
        "minimum_demand_kw": 520.0,
        "load_factor": 0.57,
        "annual_consumption_kwh": 10_000_000,
        "annual_energy_cost_usd": Decimal("950000.00"),
        "meter_count": 20,
        "submeter_count": 15,
        "check_meter_count": 4,
        "revenue_meter_count": 1,
        "has_bms": True,
        "bms_system": "Siemens_Desigo",
        "has_scada": True,
        "has_ami": True,
        "has_iot_sensors": True,
        "iot_sensor_count": 45,
        "tenant_count": 4,
        "iso_50001_certified": True,
        "hdd_base_c": 18.3,
        "cdd_base_c": 10.0,
    }


# =============================================================================
# 4. Meter Registry (20 meters: 1 revenue + 4 check + 15 submeters)
# =============================================================================


@pytest.fixture
def sample_meter_registry():
    """Create a 20-meter registry with hierarchy, calibration, and protocols.

    1 revenue meter (main incomer)
    4 check meters (one per floor)
    15 submeters (end-use: HVAC, lighting, plug loads, elevators, etc.)
    """
    rng = random.Random(42)
    meters = []

    # Revenue meter
    meters.append({
        "meter_id": "MTR-001",
        "meter_type": "REVENUE",
        "name": "Main Incomer",
        "manufacturer": "Schneider",
        "model": "ION9000",
        "serial_number": "SE-ION9K-2024-001",
        "protocol": "MODBUS_TCP",
        "energy_type": "ELECTRICITY",
        "accuracy_class": "0.2S",
        "ct_ratio": "2000:5",
        "pt_ratio": "1:1",
        "channels": ["kW", "kWh", "kVAR", "kVARh", "PF", "V", "A", "Hz"],
        "parent_meter_id": None,
        "floor": "BASEMENT",
        "status": "ACTIVE",
        "commissioned_date": "2024-01-15",
        "last_calibration": "2025-06-01",
        "calibration_due": "2026-06-01",
        "calibration_interval_months": 12,
    })

    # 4 Check meters (per floor)
    for i in range(4):
        meters.append({
            "meter_id": f"MTR-{i + 2:03d}",
            "meter_type": "CHECK",
            "name": f"Floor {i + 1} Main",
            "manufacturer": "Schneider",
            "model": "PM8000",
            "serial_number": f"SE-PM8K-2024-{i + 2:03d}",
            "protocol": "MODBUS_TCP",
            "energy_type": "ELECTRICITY",
            "accuracy_class": "0.5S",
            "ct_ratio": "600:5",
            "pt_ratio": "1:1",
            "channels": ["kW", "kWh", "kVAR", "PF"],
            "parent_meter_id": "MTR-001",
            "floor": f"FLOOR_{i + 1}",
            "status": "ACTIVE",
            "commissioned_date": "2024-01-20",
            "last_calibration": "2025-06-01",
            "calibration_due": "2026-06-01",
            "calibration_interval_months": 12,
        })

    # 15 Submeters (end-use)
    submeter_defs = [
        ("HVAC Chiller Plant", "HVAC", "BACnet", "ELECTRICITY", "BASEMENT"),
        ("HVAC AHU-1", "HVAC", "BACnet", "ELECTRICITY", "FLOOR_1"),
        ("HVAC AHU-2", "HVAC", "BACnet", "ELECTRICITY", "FLOOR_2"),
        ("HVAC AHU-3", "HVAC", "BACnet", "ELECTRICITY", "FLOOR_3"),
        ("HVAC AHU-4", "HVAC", "BACnet", "ELECTRICITY", "FLOOR_4"),
        ("Lighting Floor 1", "LIGHTING", "MODBUS_RTU", "ELECTRICITY", "FLOOR_1"),
        ("Lighting Floor 2", "LIGHTING", "MODBUS_RTU", "ELECTRICITY", "FLOOR_2"),
        ("Plug Loads Floor 1", "PLUG_LOAD", "MODBUS_RTU", "ELECTRICITY", "FLOOR_1"),
        ("Plug Loads Floor 2", "PLUG_LOAD", "MODBUS_RTU", "ELECTRICITY", "FLOOR_2"),
        ("Elevators", "ELEVATOR", "MODBUS_TCP", "ELECTRICITY", "BASEMENT"),
        ("Domestic Hot Water", "DHW", "MQTT", "NATURAL_GAS", "BASEMENT"),
        ("Kitchen Gas", "KITCHEN", "MQTT", "NATURAL_GAS", "FLOOR_1"),
        ("Server Room", "IT", "OPC_UA", "ELECTRICITY", "FLOOR_3"),
        ("EV Chargers", "EV_CHARGING", "OCPP", "ELECTRICITY", "BASEMENT"),
        ("Solar PV Inverter", "GENERATION", "SUNSPEC", "ELECTRICITY", "ROOF"),
    ]
    for idx, (name, category, protocol, energy_type, floor) in enumerate(submeter_defs):
        parent = f"MTR-{(idx // 3) + 2:03d}" if idx < 12 else "MTR-001"
        meters.append({
            "meter_id": f"MTR-{idx + 6:03d}",
            "meter_type": "SUBMETER",
            "name": name,
            "manufacturer": rng.choice(["Schneider", "Siemens", "Dent", "eGauge", "Leviton"]),
            "model": f"Model-{rng.randint(100, 999)}",
            "serial_number": f"SM-{idx + 6:03d}-2024",
            "protocol": protocol,
            "energy_type": energy_type,
            "accuracy_class": rng.choice(["0.5S", "1.0", "1.0S", "2.0"]),
            "ct_ratio": rng.choice(["100:5", "200:5", "400:5"]),
            "pt_ratio": "1:1",
            "channels": ["kW", "kWh"],
            "parent_meter_id": parent,
            "floor": floor,
            "category": category,
            "status": rng.choice(["ACTIVE", "ACTIVE", "ACTIVE", "ACTIVE", "MAINTENANCE"]),
            "commissioned_date": "2024-02-01",
            "last_calibration": "2025-06-15",
            "calibration_due": "2026-06-15",
            "calibration_interval_months": 12,
        })

    return meters


# =============================================================================
# 5. Interval Data (30 days x 96 intervals = 2,880 readings)
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
                "meter_id": "MTR-001",
                "timestamp": f"2025-07-{day:02d}T{hour:02d}:{minute:02d}:00",
                "demand_kw": round(demand_kw, 2),
                "energy_kwh": round(demand_kw * 0.25, 2),
                "temperature_c": temperature_c,
                "power_factor": round(
                    0.88 + rng.uniform(-0.06, 0.09), 3
                ),
                "voltage_v": round(480.0 + rng.uniform(-8, 8), 1),
                "current_a": round(demand_kw / (480.0 * 1.732 * 0.90), 2),
                "quality_flag": "GOOD",
            })
    return intervals


# =============================================================================
# 6. Anomaly Data (2,880 readings with 5 injected anomalies)
# =============================================================================


@pytest.fixture
def sample_anomaly_data():
    """Create interval data with 5 injected anomalies for testing detection.

    Anomalies:
      1. Spike: day 5, hour 14 - demand jumps to 5,000 kW
      2. Dropout: day 10, hour 10 - demand drops to 0 kW
      3. Flatline: day 15, hours 8-12 - demand stuck at 1,200 kW
      4. Negative: day 20, hour 16 - demand reads -150 kW
      5. Drift: day 25, hours 0-23 - gradual 2x drift upward
    """
    rng = random.Random(42)
    intervals = []
    for day in range(1, 31):
        for interval in range(96):
            hour = interval // 4
            minute = (interval % 4) * 15
            weekday = ((day - 1) + 1) % 7
            is_workday = weekday < 5

            if not is_workday:
                base = 540.0
            elif 0 <= hour < 6:
                base = 560.0
            elif 6 <= hour < 9:
                ramp_factor = (hour - 6 + minute / 60) / 3.0
                base = 560.0 + ramp_factor * 1100.0
            elif 9 <= hour < 17:
                base = 1650.0 + rng.uniform(-120, 350)
            elif 17 <= hour < 21:
                ramp_factor = 1.0 - (hour - 17 + minute / 60) / 4.0
                base = 560.0 + ramp_factor * 1100.0
            else:
                base = 580.0

            demand_kw = max(0, base + rng.uniform(-30, 30))

            # Inject anomalies
            anomaly_type = None
            if day == 5 and hour == 14 and minute == 0:
                demand_kw = 5000.0
                anomaly_type = "SPIKE"
            elif day == 10 and hour == 10 and minute == 0:
                demand_kw = 0.0
                anomaly_type = "DROPOUT"
            elif day == 15 and 8 <= hour <= 12:
                demand_kw = 1200.0
                anomaly_type = "FLATLINE"
            elif day == 20 and hour == 16 and minute == 0:
                demand_kw = -150.0
                anomaly_type = "NEGATIVE"
            elif day == 25:
                drift_factor = 1.0 + (hour / 24.0)
                demand_kw = demand_kw * drift_factor
                anomaly_type = "DRIFT"

            intervals.append({
                "meter_id": "MTR-001",
                "timestamp": f"2025-07-{day:02d}T{hour:02d}:{minute:02d}:00",
                "demand_kw": round(demand_kw, 2),
                "energy_kwh": round(demand_kw * 0.25, 2),
                "quality_flag": "SUSPECT" if anomaly_type else "GOOD",
                "injected_anomaly": anomaly_type,
            })
    return intervals


# =============================================================================
# 7. EnPI Data (12 months with HDD/CDD/production)
# =============================================================================


@pytest.fixture
def sample_enpi_data():
    """Create 12 months of EnPI data with relevant variables.

    Includes energy consumption, HDD, CDD, production output, floor area,
    and occupancy for regression-based normalization.
    """
    rng = random.Random(42)
    months = []
    monthly_kwh = [
        850_000, 780_000, 720_000, 700_000, 750_000, 920_000,
        1_050_000, 1_020_000, 880_000, 760_000, 800_000, 870_000,
    ]
    hdd_values = [680, 550, 420, 200, 80, 10, 0, 0, 30, 180, 400, 620]
    cdd_values = [0, 0, 10, 40, 120, 280, 420, 400, 250, 80, 10, 0]
    production_values = [
        950, 920, 980, 1000, 1020, 1050,
        980, 960, 1010, 1030, 990, 940,
    ]

    for month_idx in range(12):
        month = month_idx + 1
        months.append({
            "period": f"2025-{month:02d}",
            "energy_kwh": monthly_kwh[month_idx] + rng.randint(-5000, 5000),
            "energy_cost_usd": Decimal(str(round(
                monthly_kwh[month_idx] * 0.095 + rng.uniform(-500, 500), 2
            ))),
            "hdd": hdd_values[month_idx] + rng.randint(-20, 20),
            "cdd": cdd_values[month_idx] + rng.randint(-10, 10),
            "production_units": production_values[month_idx] + rng.randint(-30, 30),
            "floor_area_m2": 38_000,
            "occupancy_pct": round(0.75 + rng.uniform(-0.10, 0.15), 2),
            "operating_hours": rng.choice([176, 184, 168, 176, 184, 176,
                                           176, 184, 168, 184, 168, 176]),
            "outdoor_temp_avg_c": round(
                -2 + 20 * math.sin(math.pi * (month - 1) / 6)
                + rng.uniform(-2, 2), 1
            ),
        })
    return months


# =============================================================================
# 8. Tariff Structure (Multi-Component)
# =============================================================================


@pytest.fixture
def sample_tariff():
    """Create a multi-component tariff for cost allocation.

    Includes energy charge (TOU), demand charge, ratchet, PF penalty,
    fixed monthly charges, and taxes.
    """
    return {
        "tariff_id": "COMED-SC-10",
        "utility": "ComEd",
        "rate_class": "SC-10 General Service",
        "effective_date": "2025-01-01",
        "expiry_date": "2025-12-31",
        "currency": "USD",
        "energy_charges": {
            "on_peak_usd_per_kwh": Decimal("0.0925"),
            "mid_peak_usd_per_kwh": Decimal("0.0780"),
            "off_peak_usd_per_kwh": Decimal("0.0642"),
            "on_peak_hours": "09:00-21:00 Mon-Fri Jun-Sep",
            "mid_peak_hours": "07:00-09:00,21:00-23:00 Mon-Fri Jun-Sep",
            "off_peak_hours": "ALL_OTHER",
        },
        "demand_charges": {
            "flat_rate_usd_per_kw": Decimal("8.50"),
            "on_peak_rate_usd_per_kw": Decimal("14.25"),
            "off_peak_rate_usd_per_kw": Decimal("4.75"),
        },
        "ratchet": {
            "enabled": True,
            "ratchet_pct": Decimal("0.80"),
            "lookback_months": 12,
        },
        "power_factor_penalty": {
            "target_pf": Decimal("0.90"),
            "penalty_method": "KVA_BILLING",
            "penalty_rate_usd_per_kvar": Decimal("0.45"),
        },
        "fixed_charges": {
            "customer_charge_usd": Decimal("450.00"),
            "meter_charge_usd": Decimal("35.00"),
            "transformer_charge_usd": Decimal("125.00"),
        },
        "taxes_and_riders": {
            "municipal_tax_pct": Decimal("0.05"),
            "state_tax_pct": Decimal("0.03"),
            "renewable_rider_usd_per_kwh": Decimal("0.0015"),
        },
    }


# =============================================================================
# 9. Tenant Accounts (4 tenants)
# =============================================================================


@pytest.fixture
def sample_tenant_accounts():
    """Create 4 tenant accounts with allocation percentages and meter assignments."""
    return [
        {
            "tenant_id": "TNT-001",
            "name": "Acme Corp",
            "floor": "FLOOR_1",
            "area_m2": 9_500,
            "area_pct": Decimal("0.25"),
            "headcount": 120,
            "assigned_meters": ["MTR-006", "MTR-011", "MTR-013"],
            "allocation_method": "METERED",
            "contract_start": "2024-01-01",
            "contract_end": "2026-12-31",
            "billing_cycle": "MONTHLY",
        },
        {
            "tenant_id": "TNT-002",
            "name": "Beta Industries",
            "floor": "FLOOR_2",
            "area_m2": 9_500,
            "area_pct": Decimal("0.25"),
            "headcount": 95,
            "assigned_meters": ["MTR-007", "MTR-012", "MTR-014"],
            "allocation_method": "METERED",
            "contract_start": "2024-03-01",
            "contract_end": "2027-02-28",
            "billing_cycle": "MONTHLY",
        },
        {
            "tenant_id": "TNT-003",
            "name": "Gamma Services",
            "floor": "FLOOR_3",
            "area_m2": 11_400,
            "area_pct": Decimal("0.30"),
            "headcount": 150,
            "assigned_meters": ["MTR-008", "MTR-018"],
            "allocation_method": "METERED",
            "contract_start": "2024-06-01",
            "contract_end": "2027-05-31",
            "billing_cycle": "MONTHLY",
        },
        {
            "tenant_id": "TNT-004",
            "name": "Delta Tech",
            "floor": "FLOOR_4",
            "area_m2": 7_600,
            "area_pct": Decimal("0.20"),
            "headcount": 80,
            "assigned_meters": ["MTR-009", "MTR-010"],
            "allocation_method": "AREA_PROPORTIONAL",
            "contract_start": "2025-01-01",
            "contract_end": "2027-12-31",
            "billing_cycle": "MONTHLY",
        },
    ]


# =============================================================================
# 10. Budget (Annual with 12 monthly periods)
# =============================================================================


@pytest.fixture
def sample_budget():
    """Create an annual energy budget with 12 monthly periods.

    Includes kWh target, cost target, weather assumptions, and variance alerts.
    """
    rng = random.Random(42)
    monthly_targets_kwh = [
        850_000, 780_000, 720_000, 700_000, 750_000, 920_000,
        1_050_000, 1_020_000, 880_000, 760_000, 800_000, 870_000,
    ]
    periods = []
    for month_idx in range(12):
        month = month_idx + 1
        target_kwh = monthly_targets_kwh[month_idx]
        target_cost = round(target_kwh * 0.095, 2)
        periods.append({
            "period": f"2025-{month:02d}",
            "target_kwh": target_kwh,
            "target_cost_usd": Decimal(str(target_cost)),
            "actual_kwh": target_kwh + rng.randint(-40_000, 40_000),
            "actual_cost_usd": Decimal(str(round(
                target_cost + rng.uniform(-4000, 4000), 2
            ))),
            "hdd_assumed": max(0, 680 - month_idx * 60 + rng.randint(-10, 10)),
            "cdd_assumed": max(0, month_idx * 40 - 80 + rng.randint(-10, 10)),
            "weather_normalized_kwh": target_kwh + rng.randint(-15_000, 15_000),
        })

    return {
        "budget_id": "BUD-039-2025",
        "facility_id": "FAC-039-US-001",
        "fiscal_year": "2025",
        "currency": "USD",
        "annual_target_kwh": sum(monthly_targets_kwh),
        "annual_target_cost_usd": Decimal(str(round(
            sum(monthly_targets_kwh) * 0.095, 2
        ))),
        "variance_alert_threshold_pct": Decimal("0.10"),
        "periods": periods,
    }


# =============================================================================
# 11. Alarm Rules (10 rules across 5 categories)
# =============================================================================


@pytest.fixture
def sample_alarm_rules():
    """Create 10 alarm rules across 5 categories per ISA 18.2 lifecycle.

    Categories: DEMAND, CONSUMPTION, POWER_QUALITY, EQUIPMENT, BUDGET
    """
    return [
        {
            "rule_id": "ALM-001",
            "name": "High Demand Alert",
            "category": "DEMAND",
            "severity": "HIGH",
            "condition": "demand_kw > 1800",
            "threshold_value": 1800.0,
            "threshold_unit": "kW",
            "deadband_pct": 5.0,
            "delay_seconds": 60,
            "auto_acknowledge": False,
            "escalation_minutes": 30,
            "notification_channels": ["EMAIL", "SMS"],
        },
        {
            "rule_id": "ALM-002",
            "name": "Peak Demand Warning",
            "category": "DEMAND",
            "severity": "CRITICAL",
            "condition": "demand_kw > 1950",
            "threshold_value": 1950.0,
            "threshold_unit": "kW",
            "deadband_pct": 3.0,
            "delay_seconds": 30,
            "auto_acknowledge": False,
            "escalation_minutes": 15,
            "notification_channels": ["EMAIL", "SMS", "PUSH"],
        },
        {
            "rule_id": "ALM-003",
            "name": "Excessive Consumption",
            "category": "CONSUMPTION",
            "severity": "MEDIUM",
            "condition": "daily_kwh > 35000",
            "threshold_value": 35_000.0,
            "threshold_unit": "kWh",
            "deadband_pct": 5.0,
            "delay_seconds": 0,
            "auto_acknowledge": True,
            "escalation_minutes": 60,
            "notification_channels": ["EMAIL"],
        },
        {
            "rule_id": "ALM-004",
            "name": "Off-Hours Consumption",
            "category": "CONSUMPTION",
            "severity": "LOW",
            "condition": "off_hours_kw > 600",
            "threshold_value": 600.0,
            "threshold_unit": "kW",
            "deadband_pct": 10.0,
            "delay_seconds": 900,
            "auto_acknowledge": True,
            "escalation_minutes": 120,
            "notification_channels": ["EMAIL"],
        },
        {
            "rule_id": "ALM-005",
            "name": "Low Power Factor",
            "category": "POWER_QUALITY",
            "severity": "MEDIUM",
            "condition": "power_factor < 0.90",
            "threshold_value": 0.90,
            "threshold_unit": "PF",
            "deadband_pct": 2.0,
            "delay_seconds": 300,
            "auto_acknowledge": False,
            "escalation_minutes": 60,
            "notification_channels": ["EMAIL"],
        },
        {
            "rule_id": "ALM-006",
            "name": "Voltage Deviation",
            "category": "POWER_QUALITY",
            "severity": "HIGH",
            "condition": "abs(voltage_v - 480) > 24",
            "threshold_value": 24.0,
            "threshold_unit": "V",
            "deadband_pct": 2.0,
            "delay_seconds": 60,
            "auto_acknowledge": False,
            "escalation_minutes": 30,
            "notification_channels": ["EMAIL", "SMS"],
        },
        {
            "rule_id": "ALM-007",
            "name": "Meter Communication Loss",
            "category": "EQUIPMENT",
            "severity": "HIGH",
            "condition": "minutes_since_last_reading > 30",
            "threshold_value": 30.0,
            "threshold_unit": "minutes",
            "deadband_pct": 0.0,
            "delay_seconds": 0,
            "auto_acknowledge": False,
            "escalation_minutes": 15,
            "notification_channels": ["EMAIL", "SMS", "PUSH"],
        },
        {
            "rule_id": "ALM-008",
            "name": "Calibration Overdue",
            "category": "EQUIPMENT",
            "severity": "LOW",
            "condition": "days_past_calibration_due > 0",
            "threshold_value": 0.0,
            "threshold_unit": "days",
            "deadband_pct": 0.0,
            "delay_seconds": 0,
            "auto_acknowledge": True,
            "escalation_minutes": 1440,
            "notification_channels": ["EMAIL"],
        },
        {
            "rule_id": "ALM-009",
            "name": "Budget Overrun",
            "category": "BUDGET",
            "severity": "HIGH",
            "condition": "month_variance_pct > 10",
            "threshold_value": 10.0,
            "threshold_unit": "%",
            "deadband_pct": 2.0,
            "delay_seconds": 0,
            "auto_acknowledge": False,
            "escalation_minutes": 60,
            "notification_channels": ["EMAIL", "PUSH"],
        },
        {
            "rule_id": "ALM-010",
            "name": "Year-End Forecast Overrun",
            "category": "BUDGET",
            "severity": "CRITICAL",
            "condition": "forecast_overrun_pct > 15",
            "threshold_value": 15.0,
            "threshold_unit": "%",
            "deadband_pct": 2.0,
            "delay_seconds": 0,
            "auto_acknowledge": False,
            "escalation_minutes": 30,
            "notification_channels": ["EMAIL", "SMS", "PUSH"],
        },
    ]


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
