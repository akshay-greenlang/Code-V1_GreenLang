# -*- coding: utf-8 -*-
"""
PACK-034 ISO 50001 EnMS Pack - Shared Test Fixtures (conftest.py)
===================================================================

Provides pytest fixtures for the PACK-034 test suite including:
  - Dynamic module loading via importlib (no package install needed)
  - Pack manifest and configuration fixtures
  - Sample facility profile, energy data, baseline data, EnPI
    definitions, CUSUM data, degree day data, energy flows, action
    plans, compliance evidence, and management review data

All fixtures use importlib.util.spec_from_file_location to load modules
directly from the pack source tree, enabling test execution without
installing the pack as a Python package.

Fixture Categories:
  1. Paths and YAML data
  2. Configuration objects
  3. Facility profile data
  4. Energy consumption data
  5. Baseline data
  6. EnPI definitions
  7. CUSUM monitoring data
  8. Degree day data
  9. Energy balance flows
  10. Action plans
  11. Compliance evidence
  12. Management review data
  13. Pack configuration

Author:  GreenLang Platform Team (GL-TestEngineer)
Pack:    PACK-034 ISO 50001 Energy Management System
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
    "seu_analyzer": "seu_analyzer_engine.py",
    "energy_baseline": "energy_baseline_engine.py",
    "enpi_calculator": "enpi_calculator_engine.py",
    "cusum_monitor": "cusum_monitor_engine.py",
    "degree_day": "degree_day_engine.py",
    "energy_balance": "energy_balance_engine.py",
    "action_plan": "action_plan_engine.py",
    "compliance_checker": "compliance_checker_engine.py",
    "performance_trend": "performance_trend_engine.py",
    "management_review": "management_review_engine.py",
}

# Engine class names that should exist in each engine module
ENGINE_CLASSES = {
    "seu_analyzer": "SEUAnalyzerEngine",
    "energy_baseline": "EnergyBaselineEngine",
    "enpi_calculator": "EnPICalculatorEngine",
    "cusum_monitor": "CUSUMMonitorEngine",
    "degree_day": "DegreeDayEngine",
    "energy_balance": "EnergyBalanceEngine",
    "action_plan": "ActionPlanEngine",
    "compliance_checker": "ComplianceCheckerEngine",
    "performance_trend": "PerformanceTrendEngine",
    "management_review": "ManagementReviewEngine",
}

# Workflow file mapping
WORKFLOW_FILES = {
    "energy_review": "energy_review_workflow.py",
    "baseline_establishment": "baseline_establishment_workflow.py",
    "action_plan": "action_plan_workflow.py",
    "operational_control": "operational_control_workflow.py",
    "monitoring": "monitoring_workflow.py",
    "performance_analysis": "performance_analysis_workflow.py",
    "mv_verification": "mv_verification_workflow.py",
    "audit_certification": "audit_certification_workflow.py",
}

# Workflow class names
WORKFLOW_CLASSES = {
    "energy_review": "EnergyReviewWorkflow",
    "baseline_establishment": "BaselineEstablishmentWorkflow",
    "action_plan": "ActionPlanWorkflow",
    "operational_control": "OperationalControlWorkflow",
    "monitoring": "MonitoringWorkflow",
    "performance_analysis": "PerformanceAnalysisWorkflow",
    "mv_verification": "MVVerificationWorkflow",
    "audit_certification": "AuditCertificationWorkflow",
}

# Template file mapping
TEMPLATE_FILES = {
    "energy_policy": "energy_policy_template.py",
    "energy_review_report": "energy_review_report_template.py",
    "enpi_methodology": "enpi_methodology_template.py",
    "action_plan": "action_plan_template.py",
    "operational_control": "operational_control_template.py",
    "performance_report": "performance_report_template.py",
    "internal_audit": "internal_audit_template.py",
    "management_review": "management_review_template.py",
    "corrective_action": "corrective_action_template.py",
    "enms_documentation": "enms_documentation_template.py",
}

# Template class names
TEMPLATE_CLASSES = {
    "energy_policy": "EnergyPolicyTemplate",
    "energy_review_report": "EnergyReviewReportTemplate",
    "enpi_methodology": "EnPIMethodologyTemplate",
    "action_plan": "ActionPlanTemplate",
    "operational_control": "OperationalControlTemplate",
    "performance_report": "PerformanceReportTemplate",
    "internal_audit": "InternalAuditTemplate",
    "management_review": "ManagementReviewTemplate",
    "corrective_action": "CorrectiveActionTemplate",
    "enms_documentation": "EnMSDocumentationTemplate",
}

# Integration file mapping
INTEGRATION_FILES = {
    "pack_orchestrator": "pack_orchestrator.py",
    "mrv_bridge": "mrv_enms_bridge.py",
    "data_bridge": "data_enms_bridge.py",
    "pack031_bridge": "pack031_bridge.py",
    "pack032_bridge": "pack032_bridge.py",
    "pack033_bridge": "pack033_bridge.py",
    "eed_compliance": "eed_compliance_bridge.py",
    "bms_scada": "bms_scada_bridge.py",
    "metering_bridge": "metering_bridge.py",
    "health_check": "health_check.py",
    "setup_wizard": "setup_wizard.py",
    "certification_body_bridge": "certification_body_bridge.py",
}

# Integration class names
INTEGRATION_CLASSES = {
    "pack_orchestrator": "EnMSOrchestrator",
    "mrv_bridge": "MRVEnMSBridge",
    "data_bridge": "DataEnMSBridge",
    "pack031_bridge": "Pack031Bridge",
    "pack032_bridge": "Pack032Bridge",
    "pack033_bridge": "Pack033Bridge",
    "eed_compliance": "EEDComplianceBridge",
    "bms_scada": "BMSSCADABridge",
    "metering_bridge": "MeteringBridge",
    "health_check": "HealthCheck",
    "setup_wizard": "SetupWizard",
    "certification_body_bridge": "CertificationBodyBridge",
}

# Preset names
PRESET_NAMES = [
    "manufacturing_facility",
    "commercial_office",
    "data_center",
    "healthcare_facility",
    "retail_chain",
    "logistics_warehouse",
    "food_processing",
    "sme_multi_site",
]


# =============================================================================
# Helper: Dynamic Module Loader
# =============================================================================


def load_module_from_file(file_path: Path, module_name: str):
    """Load a Python module from a file path using importlib.

    This avoids the need to install PACK-034 as a Python package. The module
    is loaded from the pack source tree and added to sys.modules under a
    unique key to prevent collisions.

    Args:
        file_path: Absolute path to the .py file.
        module_name: Unique key for sys.modules (e.g., "pack034_test.engines.seu").

    Returns:
        The loaded module object.

    Raises:
        FileNotFoundError: If the module file does not exist.
        ImportError: If the module cannot be loaded.
    """
    if not file_path.exists():
        raise FileNotFoundError(
            f"Module file not found: {file_path}. "
            f"Ensure PACK-034 source files are present."
        )

    # Return cached module if already loaded
    if module_name in sys.modules:
        return sys.modules[module_name]

    spec = importlib.util.spec_from_file_location(module_name, str(file_path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot create module spec for {file_path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module

    try:
        spec.loader.exec_module(module)
    except Exception as exc:
        # Remove from sys.modules on failure to allow retry
        sys.modules.pop(module_name, None)
        raise ImportError(
            f"Failed to load module {module_name} from {file_path}: {exc}"
        ) from exc

    return module


def _load_module(module_name: str, file_name: str, subdir: str = "engines"):
    """Load a module dynamically using importlib.util.spec_from_file_location.

    Args:
        module_name: Logical name for the module (used as sys.modules key prefix).
        file_name: File name of the Python module.
        subdir: Subdirectory under PACK_ROOT ("engines", "workflows", "templates",
                "integrations", or "config").

    Returns:
        The loaded module object.
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
    full_module_name = f"pack034_test.{subdir}.{module_name}"
    return load_module_from_file(file_path, full_module_name)


def _load_engine(engine_key: str):
    """Load an engine module by its logical key."""
    file_name = ENGINE_FILES[engine_key]
    return _load_module(engine_key, file_name, "engines")


def _load_config_module():
    """Load the pack_config module."""
    return _load_module("pack_config", "pack_config.py", "config")


# =============================================================================
# 1. Path and YAML Fixtures
# =============================================================================


@pytest.fixture(scope="session")
def pack_root() -> Path:
    """Return the absolute path to the PACK-034 root directory."""
    return PACK_ROOT


@pytest.fixture(scope="session")
def engines_dir() -> Path:
    """Return the absolute path to the engines directory."""
    return ENGINES_DIR


@pytest.fixture(scope="session")
def workflows_dir() -> Path:
    """Return the absolute path to the workflows directory."""
    return WORKFLOWS_DIR


@pytest.fixture(scope="session")
def templates_dir() -> Path:
    """Return the absolute path to the templates directory."""
    return TEMPLATES_DIR


@pytest.fixture(scope="session")
def integrations_dir() -> Path:
    """Return the absolute path to the integrations directory."""
    return INTEGRATIONS_DIR


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
# 3. Facility Profile Fixture
# =============================================================================


@pytest.fixture
def sample_facility_profile():
    """Create a sample manufacturing facility profile for ISO 50001 testing.

    Facility: Rhine Valley Manufacturing Plant (Germany)
    Floor area: 18,000 m2
    Employees: 280
    Total energy: 2,500,000 kWh/year
    SEUs: 5 significant energy uses
    """
    return {
        "facility_id": "FAC-034-DE-001",
        "facility_name": "Rhine Valley Manufacturing Plant",
        "company_name": "EuroTech GmbH",
        "country": "DE",
        "region": "Rhineland-Palatinate",
        "facility_type": "MANUFACTURING",
        "floor_area_m2": 18000.0,
        "conditioned_area_m2": 12000.0,
        "production_area_m2": 14000.0,
        "floors": 2,
        "year_built": 1998,
        "last_renovation_year": 2020,
        "employees": 280,
        "operating_hours_per_year": 5200,
        "shifts_per_day": 2,
        "annual_electricity_kwh": 1_750_000.0,
        "annual_gas_kwh": 750_000.0,
        "total_energy_kwh": 2_500_000.0,
        "electricity_cost_eur": 350_000.0,
        "gas_cost_eur": 52_500.0,
        "total_energy_cost_eur": 402_500.0,
        "eui_kwh_per_m2": 138.9,
        "seu_count": 5,
        "seus": [
            {"name": "Compressed Air System", "energy_kwh": 625_000, "pct": 25.0},
            {"name": "HVAC Heating", "energy_kwh": 500_000, "pct": 20.0},
            {"name": "Production Line 1", "energy_kwh": 450_000, "pct": 18.0},
            {"name": "Lighting", "energy_kwh": 375_000, "pct": 15.0},
            {"name": "Cooling System", "energy_kwh": 300_000, "pct": 12.0},
        ],
        "iso50001_scope": "All operations at Rhine Valley site",
        "enms_boundary": "Site boundary including offices and production",
    }


# =============================================================================
# 4. Energy Consumption Data Fixture
# =============================================================================


@pytest.fixture
def sample_energy_data():
    """Create 12 months of energy consumption data with temperature and production.

    Monthly electricity and gas with seasonal profile, outdoor air
    temperature (degrees C), and production output (tonnes).
    """
    months = [
        "2025-01", "2025-02", "2025-03", "2025-04", "2025-05", "2025-06",
        "2025-07", "2025-08", "2025-09", "2025-10", "2025-11", "2025-12",
    ]
    electricity = [
        155_000, 148_000, 142_000, 138_000, 140_000, 152_000,
        160_000, 158_000, 145_000, 140_000, 150_000, 160_000,
    ]
    gas = [
        95_000, 88_000, 72_000, 48_000, 25_000, 12_000,
        8_000, 8_000, 20_000, 50_000, 78_000, 96_000,
    ]
    temperature_c = [
        2.1, 3.5, 7.8, 12.4, 16.8, 20.5,
        22.3, 21.8, 17.2, 11.6, 6.2, 2.8,
    ]
    production_tonnes = [
        420, 410, 450, 460, 470, 480,
        440, 430, 460, 470, 450, 420,
    ]
    records = []
    for i, month in enumerate(months):
        records.append({
            "period": month,
            "electricity_kwh": electricity[i],
            "gas_kwh": gas[i],
            "total_kwh": electricity[i] + gas[i],
            "electricity_cost_eur": electricity[i] * 0.20,
            "gas_cost_eur": gas[i] * 0.07,
            "outdoor_temp_c": temperature_c[i],
            "production_tonnes": production_tonnes[i],
            "operating_hours": 433,
        })
    return records


# =============================================================================
# 5. Baseline Data Fixture
# =============================================================================


@pytest.fixture
def sample_baseline_data():
    """Create 12 monthly baseline data points with relevant variables.

    Baseline period: Jan-Dec 2024.
    Variables: HDD, CDD, production output.
    """
    return [
        {"month": "2024-01", "energy_kwh": 248_000, "hdd": 520, "cdd": 0, "production": 415},
        {"month": "2024-02", "energy_kwh": 235_000, "hdd": 480, "cdd": 0, "production": 405},
        {"month": "2024-03", "energy_kwh": 215_000, "hdd": 380, "cdd": 0, "production": 445},
        {"month": "2024-04", "energy_kwh": 188_000, "hdd": 220, "cdd": 10, "production": 455},
        {"month": "2024-05", "energy_kwh": 168_000, "hdd": 80, "cdd": 45, "production": 465},
        {"month": "2024-06", "energy_kwh": 165_000, "hdd": 10, "cdd": 120, "production": 475},
        {"month": "2024-07", "energy_kwh": 170_000, "hdd": 0, "cdd": 180, "production": 435},
        {"month": "2024-08", "energy_kwh": 168_000, "hdd": 0, "cdd": 170, "production": 425},
        {"month": "2024-09", "energy_kwh": 167_000, "hdd": 30, "cdd": 80, "production": 455},
        {"month": "2024-10", "energy_kwh": 192_000, "hdd": 200, "cdd": 15, "production": 465},
        {"month": "2024-11", "energy_kwh": 228_000, "hdd": 410, "cdd": 0, "production": 445},
        {"month": "2024-12", "energy_kwh": 256_000, "hdd": 540, "cdd": 0, "production": 410},
    ]


# =============================================================================
# 6. EnPI Definitions Fixture
# =============================================================================


@pytest.fixture
def sample_enpi_definitions():
    """Create 3 EnPI definitions: absolute, intensity, regression-based.

    Covers the three main EnPI types per ISO 50006:2014.
    """
    return [
        {
            "enpi_id": "ENPI-001",
            "name": "Total Site Energy Consumption",
            "type": "ABSOLUTE",
            "unit": "kWh",
            "baseline_value": 2_500_000.0,
            "current_value": 2_375_000.0,
            "target_value": 2_250_000.0,
            "decrease_is_better": True,
        },
        {
            "enpi_id": "ENPI-002",
            "name": "Energy Intensity per Tonne",
            "type": "INTENSITY",
            "unit": "kWh/tonne",
            "normalizing_variable": "production_tonnes",
            "baseline_value": 465.1,
            "current_value": 442.0,
            "target_value": 418.6,
            "decrease_is_better": True,
        },
        {
            "enpi_id": "ENPI-003",
            "name": "Weather-Adjusted Consumption",
            "type": "REGRESSION",
            "unit": "kWh",
            "model_intercept": 120_000.0,
            "model_slope_hdd": 180.0,
            "model_slope_cdd": 95.0,
            "model_r_squared": 0.92,
            "baseline_value": 2_500_000.0,
            "current_value": 2_420_000.0,
            "expected_value": 2_480_000.0,
            "decrease_is_better": True,
        },
    ]


# =============================================================================
# 7. CUSUM Monitoring Data Fixture
# =============================================================================


@pytest.fixture
def sample_cusum_data():
    """Create 24 data points with actual vs expected for CUSUM analysis.

    Simulates 24 months (2 years) of monitoring with a performance
    shift starting at month 13.
    """
    data = []
    for i in range(24):
        month_num = i + 1
        # Baseline-consistent for first 12 months, then improvement
        if month_num <= 12:
            actual = 200_000 + (i % 3) * 5_000
            expected = 200_000 + (i % 3) * 5_000
        else:
            actual = 190_000 + (i % 3) * 5_000
            expected = 200_000 + (i % 3) * 5_000
        data.append({
            "period": f"2024-{month_num:02d}" if month_num <= 12 else f"2025-{month_num - 12:02d}",
            "actual_kwh": actual,
            "expected_kwh": expected,
            "residual_kwh": actual - expected,
        })
    return data


# =============================================================================
# 8. Degree Day Data Fixture
# =============================================================================


@pytest.fixture
def sample_degree_day_data():
    """Create 365 daily temperature readings for degree day calculations.

    Simulates a temperate European climate with sinusoidal temperature
    pattern: winter low ~0C, summer high ~24C.
    """
    import math
    data = []
    for day in range(365):
        # Sinusoidal annual temperature pattern
        mean_temp = 12.0 + 12.0 * math.sin(2 * math.pi * (day - 80) / 365)
        min_temp = mean_temp - 5.0
        max_temp = mean_temp + 5.0
        data.append({
            "day_of_year": day + 1,
            "date": f"2025-{(day // 30) + 1:02d}-{(day % 30) + 1:02d}",
            "mean_temp_c": round(mean_temp, 1),
            "min_temp_c": round(min_temp, 1),
            "max_temp_c": round(max_temp, 1),
        })
    return data


# =============================================================================
# 9. Energy Balance Flows Fixture
# =============================================================================


@pytest.fixture
def sample_energy_flows():
    """Create 10 energy flow entries for energy balance calculations.

    Covers inputs (electricity, gas), internal distribution, and
    end-use consumption with losses.
    """
    return [
        {"flow_id": "EF-001", "type": "INPUT", "source": "Grid Electricity",
         "destination": "Main Switchboard", "energy_kwh": 1_750_000, "fuel": "electricity"},
        {"flow_id": "EF-002", "type": "INPUT", "source": "Natural Gas Supply",
         "destination": "Gas Manifold", "energy_kwh": 750_000, "fuel": "natural_gas"},
        {"flow_id": "EF-003", "type": "DISTRIBUTION", "source": "Main Switchboard",
         "destination": "Production Lines", "energy_kwh": 825_000, "fuel": "electricity"},
        {"flow_id": "EF-004", "type": "DISTRIBUTION", "source": "Main Switchboard",
         "destination": "HVAC System", "energy_kwh": 450_000, "fuel": "electricity"},
        {"flow_id": "EF-005", "type": "DISTRIBUTION", "source": "Main Switchboard",
         "destination": "Compressed Air", "energy_kwh": 375_000, "fuel": "electricity"},
        {"flow_id": "EF-006", "type": "DISTRIBUTION", "source": "Gas Manifold",
         "destination": "Boiler Plant", "energy_kwh": 650_000, "fuel": "natural_gas"},
        {"flow_id": "EF-007", "type": "DISTRIBUTION", "source": "Gas Manifold",
         "destination": "Process Heating", "energy_kwh": 100_000, "fuel": "natural_gas"},
        {"flow_id": "EF-008", "type": "LOSS", "source": "Distribution",
         "destination": "Transformer Losses", "energy_kwh": 35_000, "fuel": "electricity"},
        {"flow_id": "EF-009", "type": "LOSS", "source": "Distribution",
         "destination": "Pipe Losses", "energy_kwh": 25_000, "fuel": "natural_gas"},
        {"flow_id": "EF-010", "type": "OUTPUT", "source": "Boiler Plant",
         "destination": "Space Heating", "energy_kwh": 585_000, "fuel": "heat"},
    ]


# =============================================================================
# 10. Action Plans Fixture
# =============================================================================


@pytest.fixture
def sample_action_plans():
    """Create 3 action plans with items for ISO 50001 objectives.

    Covers compressed air, lighting, and HVAC improvement objectives.
    """
    return [
        {
            "plan_id": "AP-001",
            "objective": "Reduce compressed air energy by 15%",
            "target_savings_kwh": 93_750,
            "responsible": "Maintenance Manager",
            "deadline": "2026-06-30",
            "status": "IN_PROGRESS",
            "items": [
                {"item_id": "AP-001-01", "action": "Fix identified leaks (25 sites)",
                 "savings_kwh": 45_000, "cost_eur": 8_000, "status": "COMPLETED"},
                {"item_id": "AP-001-02", "action": "Install VSD on compressor 2",
                 "savings_kwh": 30_000, "cost_eur": 15_000, "status": "IN_PROGRESS"},
                {"item_id": "AP-001-03", "action": "Reduce system pressure by 0.5 bar",
                 "savings_kwh": 18_750, "cost_eur": 500, "status": "PLANNED"},
            ],
        },
        {
            "plan_id": "AP-002",
            "objective": "LED retrofit all production areas",
            "target_savings_kwh": 112_500,
            "responsible": "Facilities Manager",
            "deadline": "2026-09-30",
            "status": "PLANNED",
            "items": [
                {"item_id": "AP-002-01", "action": "LED retrofit production hall A",
                 "savings_kwh": 56_250, "cost_eur": 25_000, "status": "PLANNED"},
                {"item_id": "AP-002-02", "action": "LED retrofit production hall B",
                 "savings_kwh": 56_250, "cost_eur": 25_000, "status": "PLANNED"},
            ],
        },
        {
            "plan_id": "AP-003",
            "objective": "Optimise boiler controls for heating season",
            "target_savings_kwh": 50_000,
            "responsible": "Energy Manager",
            "deadline": "2026-10-31",
            "status": "PLANNED",
            "items": [
                {"item_id": "AP-003-01", "action": "Install weather compensation",
                 "savings_kwh": 30_000, "cost_eur": 5_000, "status": "PLANNED"},
                {"item_id": "AP-003-02", "action": "Optimise boiler sequencing",
                 "savings_kwh": 20_000, "cost_eur": 2_000, "status": "PLANNED"},
            ],
        },
    ]


# =============================================================================
# 11. Compliance Evidence Fixture
# =============================================================================


@pytest.fixture
def sample_compliance_evidence():
    """Create compliance evidence dictionary for ISO 50001 clauses 4-10.

    Maps each main clause to available evidence items.
    """
    return {
        "clause_4_context": {
            "interested_parties_identified": True,
            "scope_defined": True,
            "boundaries_documented": True,
            "enms_processes_established": True,
        },
        "clause_5_leadership": {
            "energy_policy_approved": True,
            "roles_assigned": True,
            "energy_team_established": True,
            "management_commitment_documented": True,
            "resources_allocated": True,
        },
        "clause_6_planning": {
            "energy_review_completed": True,
            "seus_identified": True,
            "enb_established": True,
            "enpis_defined": True,
            "objectives_set": True,
            "action_plans_documented": True,
            "risks_assessed": True,
        },
        "clause_7_support": {
            "competence_records": True,
            "awareness_training_completed": True,
            "communication_plan": True,
            "documented_information_controlled": True,
        },
        "clause_8_operation": {
            "operational_controls_implemented": True,
            "procurement_criteria_energy": True,
            "design_criteria_energy": True,
        },
        "clause_9_performance": {
            "monitoring_plan_active": True,
            "enpi_tracking_active": True,
            "calibration_records": True,
            "internal_audit_completed": True,
            "management_review_completed": True,
        },
        "clause_10_improvement": {
            "nonconformities_tracked": True,
            "corrective_actions_closed": True,
            "continual_improvement_demonstrated": True,
        },
    }


# =============================================================================
# 12. Management Review Data Fixture
# =============================================================================


@pytest.fixture
def sample_management_review_data():
    """Create management review input data per ISO 50001 Clause 9.3.

    Includes all required review inputs as specified in the standard.
    """
    return {
        "review_id": "MR-2026-Q1",
        "review_date": "2026-03-15",
        "chairperson": "CEO",
        "attendees": [
            "CEO", "Energy Manager", "Maintenance Manager",
            "Production Manager", "Finance Director",
        ],
        "inputs": {
            "previous_actions_status": "All 5 actions from Q4 review completed",
            "energy_policy_review": "Policy remains adequate; minor wording update proposed",
            "enpi_performance": {
                "total_consumption_kwh": 2_375_000,
                "baseline_kwh": 2_500_000,
                "improvement_pct": 5.0,
                "target_pct": 10.0,
            },
            "compliance_status": "Fully compliant; 0 nonconformities open",
            "audit_findings": [
                {"finding": "Minor: calibration overdue on meter M-007", "status": "CLOSED"},
            ],
            "resource_adequacy": "Current staffing adequate; budget approved for Q2",
            "external_changes": "New EED Article 8 requirement effective Jan 2027",
            "improvement_opportunities": [
                "Waste heat recovery from compressors",
                "Solar PV feasibility study",
            ],
        },
    }


# =============================================================================
# 13. Pack Configuration Fixture
# =============================================================================


@pytest.fixture
def sample_pack_config():
    """Create a default pack configuration dictionary for PACK-034.

    Includes ISO 50001 specific settings: scope, SEU thresholds,
    EnPI methodology, compliance strictness, and reporting options.
    """
    return {
        "pack_id": "PACK-034",
        "pack_name": "ISO 50001 Energy Management System",
        "version": "1.0.0",
        "facility_type": "MANUFACTURING",
        "iso_standard": "ISO 50001:2018",
        "scope": "All site operations",
        "seu_threshold_pct": 5.0,
        "pareto_target_pct": 80.0,
        "baseline_period_months": 12,
        "enpi_methodology": "REGRESSION",
        "cusum_decision_interval": 5.0,
        "compliance_strictness": "CERTIFICATION",
        "reporting_currency": "EUR",
        "reporting_frequency": "MONTHLY",
        "degree_day_base_heating_c": 15.5,
        "degree_day_base_cooling_c": 18.0,
        "output_formats": ["markdown", "html", "json"],
        "enable_provenance": True,
    }
