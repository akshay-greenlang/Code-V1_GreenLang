# -*- coding: utf-8 -*-
"""
PACK-036 Utility Analysis Pack - Shared Test Fixtures (conftest.py)
====================================================================

Provides pytest fixtures for the PACK-036 test suite including:
  - Dynamic module loading via importlib (no package install needed)
  - Pack manifest and configuration fixtures
  - Sample utility bill data (electricity, gas, water)
  - Rate structures (TOU, tiered, demand charges)
  - Interval data (15-min for demand analysis)
  - Allocation entities and rules
  - Historical data for forecasting
  - Facility metrics and market prices
  - Weather data with degree days

Fixture Categories:
  1. Paths and YAML data
  2. Configuration objects
  3. Utility bill data (electricity, gas, water)
  4. Bill history (12 months)
  5. Rate structure data (TOU, tiered, demand)
  6. Interval data (15-min)
  7. Demand profile
  8. Allocation entities and rules
  9. Historical data (24 months for forecasting)
  10. Facility metrics
  11. Market prices
  12. Weather data
  13. Monthly consumption with HDD/CDD (24 months)
  14. Pack configuration

Author:  GreenLang Platform Team (GL-TestEngineer)
Pack:    PACK-036 Utility Analysis
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
    "utility_bill_parser": "utility_bill_parser_engine.py",
    "rate_structure_analyzer": "rate_structure_analyzer_engine.py",
    "demand_analysis": "demand_analysis_engine.py",
    "cost_allocation": "cost_allocation_engine.py",
    "budget_forecasting": "budget_forecasting_engine.py",
    "procurement_intelligence": "procurement_intelligence_engine.py",
    "utility_benchmark": "utility_benchmark_engine.py",
    "regulatory_charge_optimizer": "regulatory_charge_optimizer_engine.py",
    "weather_normalization": "weather_normalization_engine.py",
    "utility_reporting": "utility_reporting_engine.py",
}

# Engine class names that should exist in each engine module
ENGINE_CLASSES = {
    "utility_bill_parser": "UtilityBillParserEngine",
    "rate_structure_analyzer": "RateStructureAnalyzerEngine",
    "demand_analysis": "DemandAnalysisEngine",
    "cost_allocation": "CostAllocationEngine",
    "budget_forecasting": "BudgetForecastingEngine",
    "procurement_intelligence": "ProcurementIntelligenceEngine",
    "utility_benchmark": "UtilityBenchmarkEngine",
    "regulatory_charge_optimizer": "RegulatoryChargeOptimizerEngine",
    "weather_normalization": "WeatherNormalizationEngine",
    "utility_reporting": "UtilityReportingEngine",
}

# Workflow file mapping
WORKFLOW_FILES = {
    "bill_audit": "bill_audit_workflow.py",
    "rate_optimization": "rate_optimization_workflow.py",
    "demand_management": "demand_management_workflow.py",
    "cost_allocation": "cost_allocation_workflow.py",
    "budget_planning": "budget_planning_workflow.py",
    "procurement": "procurement_workflow.py",
    "benchmark": "benchmark_workflow.py",
    "full_utility_analysis": "full_utility_analysis_workflow.py",
}

# Workflow class names
WORKFLOW_CLASSES = {
    "bill_audit": "BillAuditWorkflow",
    "rate_optimization": "RateOptimizationWorkflow",
    "demand_management": "DemandManagementWorkflow",
    "cost_allocation": "CostAllocationWorkflow",
    "budget_planning": "BudgetPlanningWorkflow",
    "procurement": "ProcurementWorkflow",
    "benchmark": "BenchmarkWorkflow",
    "full_utility_analysis": "FullUtilityAnalysisWorkflow",
}

# Template file mapping
TEMPLATE_FILES = {
    "bill_audit_report": "bill_audit_report.py",
    "rate_comparison_report": "rate_comparison_report.py",
    "demand_profile_report": "demand_profile_report.py",
    "cost_allocation_report": "cost_allocation_report.py",
    "budget_forecast_report": "budget_forecast_report.py",
    "procurement_strategy_report": "procurement_strategy_report.py",
    "benchmark_report": "benchmark_report.py",
    "regulatory_charge_report": "regulatory_charge_report.py",
    "executive_dashboard": "executive_dashboard.py",
    "utility_savings_report": "utility_savings_report.py",
}

# Template class names
TEMPLATE_CLASSES = {
    "bill_audit_report": "BillAuditReportTemplate",
    "rate_comparison_report": "RateComparisonReportTemplate",
    "demand_profile_report": "DemandProfileReportTemplate",
    "cost_allocation_report": "CostAllocationReportTemplate",
    "budget_forecast_report": "BudgetForecastReportTemplate",
    "procurement_strategy_report": "ProcurementStrategyReportTemplate",
    "benchmark_report": "BenchmarkReportTemplate",
    "regulatory_charge_report": "RegulatoryChargeReportTemplate",
    "executive_dashboard": "ExecutiveDashboardTemplate",
    "utility_savings_report": "UtilitySavingsReportTemplate",
}

# Integration file mapping
INTEGRATION_FILES = {
    "pack_orchestrator": "pack_orchestrator.py",
    "mrv_bridge": "mrv_bridge.py",
    "data_bridge": "data_bridge.py",
    "pack031_bridge": "pack031_bridge.py",
    "pack032_bridge": "pack032_bridge.py",
    "pack033_bridge": "pack033_bridge.py",
    "utility_provider_bridge": "utility_provider_bridge.py",
    "weather_bridge": "weather_bridge.py",
    "market_data_bridge": "market_data_bridge.py",
    "health_check": "health_check.py",
    "setup_wizard": "setup_wizard.py",
    "alert_bridge": "alert_bridge.py",
}

# Integration class names
INTEGRATION_CLASSES = {
    "pack_orchestrator": "UtilityAnalysisOrchestrator",
    "mrv_bridge": "MRVBridge",
    "data_bridge": "DataBridge",
    "pack031_bridge": "Pack031Bridge",
    "pack032_bridge": "Pack032Bridge",
    "pack033_bridge": "Pack033Bridge",
    "utility_provider_bridge": "UtilityProviderBridge",
    "weather_bridge": "WeatherBridge",
    "market_data_bridge": "MarketDataBridge",
    "health_check": "HealthCheck",
    "setup_wizard": "SetupWizard",
    "alert_bridge": "AlertBridge",
}

# Preset names
PRESET_NAMES = [
    "commercial_office",
    "industrial_facility",
    "retail_chain",
    "multi_tenant",
    "hospital",
    "university_campus",
    "data_center",
    "government_building",
]


# =============================================================================
# Helper: Dynamic Module Loader
# =============================================================================


def _load_module(module_name: str, file_name: str, subdir: str = "engines"):
    """Load a module dynamically using importlib.util.spec_from_file_location.

    This avoids the need to install PACK-036 as a Python package. The module
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
            f"Ensure PACK-036 source files are present."
        )

    full_module_name = f"pack036_test.{subdir}.{module_name}"

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
# 1. Path and YAML Fixtures
# =============================================================================


@pytest.fixture(scope="session")
def pack_root() -> Path:
    """Return the absolute path to the PACK-036 root directory."""
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


@pytest.fixture
def pack_config():
    """Create a default PACK-036 configuration dictionary."""
    return {
        "pack_id": "PACK-036",
        "pack_name": "Utility Analysis",
        "version": "1.0.0",
        "category": "energy-efficiency",
        "environment": "test",
        "currency": "EUR",
        "default_region": "EU",
        "decimal_precision": 4,
        "provenance_enabled": True,
        "analysis_period_years": 10,
        "discount_rate": Decimal("0.08"),
    }


# =============================================================================
# 3. Utility Bill Data Fixtures
# =============================================================================


@pytest.fixture
def sample_utility_bill():
    """Create a sample electricity utility bill with line items.

    Facility: Berlin Commercial Office
    Billing period: 2025-01-01 to 2025-01-31
    """
    return {
        "bill_id": "BILL-036-DE-001",
        "account_number": "ACC-DE-8901234",
        "facility_id": "FAC-036-DE-001",
        "facility_name": "Berlin Office Tower",
        "utility_type": "ELECTRICITY",
        "provider": "Vattenfall Europe",
        "period_start": "2025-01-01",
        "period_end": "2025-01-31",
        "billing_days": 31,
        "meter_number": "MTR-DE-5678",
        "read_type": "ACTUAL",
        "previous_read_kwh": 1_200_000,
        "current_read_kwh": 1_350_000,
        "consumption_kwh": 150_000,
        "demand_kw": 480.0,
        "power_factor": 0.92,
        "line_items": [
            {"description": "Energy Charge", "quantity": 150_000, "unit": "kWh",
             "rate": Decimal("0.12"), "amount": Decimal("18000.00")},
            {"description": "Demand Charge", "quantity": 480, "unit": "kW",
             "rate": Decimal("8.50"), "amount": Decimal("4080.00")},
            {"description": "Network Fee", "quantity": 1, "unit": "month",
             "rate": Decimal("1200.00"), "amount": Decimal("1200.00")},
            {"description": "Renewable Surcharge (EEG)", "quantity": 150_000,
             "unit": "kWh", "rate": Decimal("0.0372"), "amount": Decimal("5580.00")},
            {"description": "Electricity Tax", "quantity": 150_000, "unit": "kWh",
             "rate": Decimal("0.02050"), "amount": Decimal("3075.00")},
            {"description": "Concession Fee", "quantity": 150_000, "unit": "kWh",
             "rate": Decimal("0.00011"), "amount": Decimal("16.50")},
            {"description": "VAT (19%)", "quantity": 1, "unit": "lump",
             "rate": Decimal("0.19"), "amount": Decimal("6070.29")},
        ],
        "subtotal_eur": Decimal("31951.50"),
        "tax_eur": Decimal("6070.29"),
        "total_eur": Decimal("38021.79"),
        "currency": "EUR",
    }


@pytest.fixture
def sample_gas_bill():
    """Create a sample natural gas utility bill."""
    return {
        "bill_id": "BILL-036-DE-002",
        "account_number": "ACC-DE-GAS-456",
        "facility_id": "FAC-036-DE-001",
        "facility_name": "Berlin Office Tower",
        "utility_type": "NATURAL_GAS",
        "provider": "GASAG AG",
        "period_start": "2025-01-01",
        "period_end": "2025-01-31",
        "billing_days": 31,
        "meter_number": "MTR-GAS-9012",
        "read_type": "ACTUAL",
        "previous_read_m3": 85_000,
        "current_read_m3": 90_000,
        "consumption_m3": 5_000,
        "consumption_kwh": 55_000,
        "calorific_value_kwh_per_m3": 11.0,
        "line_items": [
            {"description": "Gas Supply", "quantity": 55_000, "unit": "kWh",
             "rate": Decimal("0.055"), "amount": Decimal("3025.00")},
            {"description": "Network Charge", "quantity": 55_000, "unit": "kWh",
             "rate": Decimal("0.015"), "amount": Decimal("825.00")},
            {"description": "Gas Tax", "quantity": 55_000, "unit": "kWh",
             "rate": Decimal("0.0055"), "amount": Decimal("302.50")},
            {"description": "VAT (19%)", "quantity": 1, "unit": "lump",
             "rate": Decimal("0.19"), "amount": Decimal("788.98")},
        ],
        "subtotal_eur": Decimal("4152.50"),
        "tax_eur": Decimal("788.98"),
        "total_eur": Decimal("4941.48"),
        "currency": "EUR",
    }


@pytest.fixture
def sample_water_bill():
    """Create a sample water/sewerage utility bill."""
    return {
        "bill_id": "BILL-036-DE-003",
        "account_number": "ACC-DE-WAT-789",
        "facility_id": "FAC-036-DE-001",
        "facility_name": "Berlin Office Tower",
        "utility_type": "WATER",
        "provider": "Berliner Wasserbetriebe",
        "period_start": "2025-01-01",
        "period_end": "2025-01-31",
        "billing_days": 31,
        "meter_number": "MTR-WAT-3456",
        "read_type": "ACTUAL",
        "consumption_m3": 450,
        "line_items": [
            {"description": "Fresh Water", "quantity": 450, "unit": "m3",
             "rate": Decimal("1.813"), "amount": Decimal("815.85")},
            {"description": "Sewerage", "quantity": 450, "unit": "m3",
             "rate": Decimal("2.558"), "amount": Decimal("1151.10")},
            {"description": "Standing Charge", "quantity": 1, "unit": "month",
             "rate": Decimal("45.00"), "amount": Decimal("45.00")},
        ],
        "subtotal_eur": Decimal("2011.95"),
        "tax_eur": Decimal("0.00"),
        "total_eur": Decimal("2011.95"),
        "currency": "EUR",
    }


# =============================================================================
# 4. Bill History (12 months)
# =============================================================================


@pytest.fixture
def sample_bill_history():
    """Create 12 months of electricity bill history for trend analysis."""
    months = [
        "2024-01", "2024-02", "2024-03", "2024-04", "2024-05", "2024-06",
        "2024-07", "2024-08", "2024-09", "2024-10", "2024-11", "2024-12",
    ]
    consumption_kwh = [
        140_000, 135_000, 145_000, 155_000, 170_000, 195_000,
        210_000, 205_000, 180_000, 155_000, 142_000, 148_000,
    ]
    demand_kw = [
        420, 410, 430, 460, 510, 560,
        590, 580, 530, 470, 425, 440,
    ]
    cost_eur = [
        22_400, 21_600, 23_200, 24_800, 27_200, 31_200,
        33_600, 32_800, 28_800, 24_800, 22_720, 23_680,
    ]
    records = []
    for i, month in enumerate(months):
        records.append({
            "period": month,
            "utility_type": "ELECTRICITY",
            "consumption_kwh": consumption_kwh[i],
            "demand_kw": demand_kw[i],
            "cost_eur": Decimal(str(cost_eur[i])),
            "rate_eur_per_kwh": Decimal(str(round(cost_eur[i] / consumption_kwh[i], 4))),
            "billing_days": 30 if i != 1 else 29,
            "read_type": "ACTUAL",
        })
    return records


# =============================================================================
# 5. Rate Structure Data
# =============================================================================


@pytest.fixture
def sample_rate_structure():
    """Create a TOU rate structure with tiers and demand charges."""
    return {
        "rate_id": "RATE-DE-TOU-001",
        "rate_name": "Commercial TOU with Demand",
        "provider": "Vattenfall Europe",
        "effective_date": "2025-01-01",
        "expiry_date": "2025-12-31",
        "currency": "EUR",
        "rate_type": "TOU",
        "voltage_level": "MEDIUM",
        "energy_charges": {
            "on_peak": {"hours": "08:00-20:00 Mon-Fri",
                        "rate_eur_per_kwh": Decimal("0.1450")},
            "off_peak": {"hours": "20:00-08:00 Mon-Fri, All day Sat-Sun",
                         "rate_eur_per_kwh": Decimal("0.0950")},
        },
        "tiered_charges": [
            {"tier": 1, "from_kwh": 0, "to_kwh": 100_000,
             "rate_eur_per_kwh": Decimal("0.1200")},
            {"tier": 2, "from_kwh": 100_001, "to_kwh": 500_000,
             "rate_eur_per_kwh": Decimal("0.1100")},
            {"tier": 3, "from_kwh": 500_001, "to_kwh": None,
             "rate_eur_per_kwh": Decimal("0.1000")},
        ],
        "demand_charges": {
            "rate_eur_per_kw": Decimal("8.50"),
            "billing_method": "monthly_peak",
            "ratchet_pct": Decimal("0.80"),
            "ratchet_months": 11,
            "minimum_demand_kw": 50,
        },
        "fixed_charges": {
            "monthly_service_eur": Decimal("250.00"),
            "meter_charge_eur": Decimal("45.00"),
        },
        "power_factor_penalty": {
            "threshold": Decimal("0.90"),
            "penalty_pct_per_point": Decimal("0.01"),
        },
        "taxes_and_surcharges": {
            "eeg_surcharge_eur_per_kwh": Decimal("0.0372"),
            "electricity_tax_eur_per_kwh": Decimal("0.02050"),
            "concession_fee_eur_per_kwh": Decimal("0.00011"),
            "vat_pct": Decimal("0.19"),
        },
    }


# =============================================================================
# 6. Interval Data (15-min)
# =============================================================================


@pytest.fixture
def sample_interval_data():
    """Create 15-minute interval data for 1 month (January 2025).

    31 days x 96 intervals = 2976 data points.
    Simulates a commercial office load profile.
    """
    import random
    random.seed(42)
    intervals = []
    for day in range(1, 32):
        for interval in range(96):
            hour = interval // 4
            minute = (interval % 4) * 15
            weekday = ((day - 1) + 2) % 7  # Jan 1, 2025 = Wednesday
            is_workday = weekday < 5

            if not is_workday:
                base = 120.0
                variation = random.uniform(-10, 10)
            elif 0 <= hour < 6:
                base = 150.0
                variation = random.uniform(-15, 15)
            elif 6 <= hour < 9:
                ramp_factor = (hour - 6 + minute / 60) / 3.0
                base = 150.0 + ramp_factor * 300.0
                variation = random.uniform(-20, 20)
            elif 9 <= hour < 17:
                base = 420.0 + random.uniform(-40, 80)
                variation = random.uniform(-10, 10)
            elif 17 <= hour < 21:
                ramp_factor = 1.0 - (hour - 17 + minute / 60) / 4.0
                base = 150.0 + ramp_factor * 270.0
                variation = random.uniform(-15, 15)
            else:
                base = 155.0
                variation = random.uniform(-10, 10)

            demand_kw = max(0, base + variation)
            intervals.append({
                "timestamp": f"2025-01-{day:02d}T{hour:02d}:{minute:02d}:00",
                "demand_kw": round(demand_kw, 2),
                "energy_kwh": round(demand_kw * 0.25, 2),
            })
    return intervals


# =============================================================================
# 7. Demand Profile
# =============================================================================


@pytest.fixture
def sample_demand_profile():
    """Create a monthly demand profile summary for analysis."""
    return {
        "facility_id": "FAC-036-DE-001",
        "period": "2025-01",
        "peak_demand_kw": 520.0,
        "average_demand_kw": 285.0,
        "minimum_demand_kw": 130.0,
        "load_factor": 0.548,
        "peak_timestamp": "2025-01-15T14:30:00",
        "on_peak_consumption_kwh": 95_000,
        "off_peak_consumption_kwh": 55_000,
        "total_consumption_kwh": 150_000,
        "power_factor": 0.92,
        "coincident_peak_kw": 490.0,
        "ratchet_demand_kw": 0.0,
    }


# =============================================================================
# 8. Allocation Entities and Rules
# =============================================================================


@pytest.fixture
def sample_allocation_entities():
    """Create 5 tenant entities for cost allocation testing."""
    return [
        {"entity_id": "TENANT-001", "entity_name": "TechCorp GmbH",
         "floor_area_m2": 3500, "floor_area_pct": Decimal("0.35"),
         "has_submeter": True, "submetered_kwh": 52_500,
         "employees": 120, "operating_hours": 2600},
        {"entity_id": "TENANT-002", "entity_name": "LegalPartners AG",
         "floor_area_m2": 2000, "floor_area_pct": Decimal("0.20"),
         "has_submeter": True, "submetered_kwh": 28_000,
         "employees": 60, "operating_hours": 2400},
        {"entity_id": "TENANT-003", "entity_name": "FinanceHouse SE",
         "floor_area_m2": 2500, "floor_area_pct": Decimal("0.25"),
         "has_submeter": False, "submetered_kwh": None,
         "employees": 85, "operating_hours": 2800},
        {"entity_id": "TENANT-004", "entity_name": "MediaWorks KG",
         "floor_area_m2": 1200, "floor_area_pct": Decimal("0.12"),
         "has_submeter": False, "submetered_kwh": None,
         "employees": 40, "operating_hours": 3000},
        {"entity_id": "TENANT-005", "entity_name": "BuildingServices (Common)",
         "floor_area_m2": 800, "floor_area_pct": Decimal("0.08"),
         "has_submeter": True, "submetered_kwh": 18_000,
         "employees": 10, "operating_hours": 8760},
    ]


@pytest.fixture
def sample_allocation_rules():
    """Create allocation rules for cost distribution."""
    return {
        "method": "HYBRID",
        "energy_allocation": "SUBMETERED_FIRST",
        "demand_allocation": "COINCIDENT_PEAK",
        "common_area_method": "PRO_RATA_AREA",
        "common_area_pct": Decimal("0.08"),
        "minimum_charge_eur": Decimal("100.00"),
        "reconciliation_period": "MONTHLY",
        "true_up_frequency": "QUARTERLY",
    }


# =============================================================================
# 9. Historical Data (24 months for forecasting)
# =============================================================================


@pytest.fixture
def sample_historical_data():
    """Create 24 months of historical utility data for budget forecasting."""
    months_2023 = [
        ("2023-01", 142_000, 22_720), ("2023-02", 130_000, 20_800),
        ("2023-03", 140_000, 22_400), ("2023-04", 150_000, 24_000),
        ("2023-05", 165_000, 26_400), ("2023-06", 188_000, 30_080),
        ("2023-07", 202_000, 32_320), ("2023-08", 198_000, 31_680),
        ("2023-09", 175_000, 28_000), ("2023-10", 150_000, 24_000),
        ("2023-11", 138_000, 22_080), ("2023-12", 145_000, 23_200),
    ]
    months_2024 = [
        ("2024-01", 140_000, 23_800), ("2024-02", 135_000, 22_950),
        ("2024-03", 145_000, 24_650), ("2024-04", 155_000, 26_350),
        ("2024-05", 170_000, 28_900), ("2024-06", 195_000, 33_150),
        ("2024-07", 210_000, 35_700), ("2024-08", 205_000, 34_850),
        ("2024-09", 180_000, 30_600), ("2024-10", 155_000, 26_350),
        ("2024-11", 142_000, 24_140), ("2024-12", 148_000, 25_160),
    ]
    records = []
    for period, kwh, cost in months_2023 + months_2024:
        records.append({
            "period": period,
            "utility_type": "ELECTRICITY",
            "consumption_kwh": kwh,
            "cost_eur": Decimal(str(cost)),
            "rate_eur_per_kwh": Decimal(str(round(cost / kwh, 4))),
        })
    return records


# =============================================================================
# 10. Facility Metrics
# =============================================================================


@pytest.fixture
def sample_facility_metrics():
    """Create facility metrics for benchmarking."""
    return {
        "facility_id": "FAC-036-DE-001",
        "facility_name": "Berlin Office Tower",
        "building_type": "OFFICE",
        "country": "DE",
        "region": "Europe",
        "climate_zone": "4A",
        "floor_area_m2": 10_000,
        "conditioned_area_m2": 9_200,
        "floors": 8,
        "year_built": 2008,
        "last_renovation_year": 2020,
        "employees": 350,
        "operating_hours_per_year": 2600,
        "annual_electricity_kwh": 1_980_000,
        "annual_gas_kwh": 650_000,
        "annual_site_energy_kwh": 2_630_000,
        "annual_source_energy_kwh": 5_594_600,
        "site_eui_kwh_per_m2": 263.0,
        "source_eui_kwh_per_m2": 559.46,
        "energy_star_score": 72,
    }


# =============================================================================
# 11. Market Prices
# =============================================================================


@pytest.fixture
def sample_market_prices():
    """Create wholesale market price data for procurement intelligence."""
    return {
        "market": "EPEX SPOT DE",
        "currency": "EUR",
        "unit": "MWh",
        "spot_prices": [
            {"date": "2025-01-15", "price_eur_per_mwh": Decimal("85.20")},
            {"date": "2025-01-16", "price_eur_per_mwh": Decimal("92.40")},
            {"date": "2025-01-17", "price_eur_per_mwh": Decimal("78.60")},
            {"date": "2025-01-18", "price_eur_per_mwh": Decimal("65.30")},
            {"date": "2025-01-19", "price_eur_per_mwh": Decimal("62.10")},
        ],
        "forward_curve": [
            {"period": "Q2-2025", "price_eur_per_mwh": Decimal("88.50")},
            {"period": "Q3-2025", "price_eur_per_mwh": Decimal("95.20")},
            {"period": "Q4-2025", "price_eur_per_mwh": Decimal("102.80")},
            {"period": "CAL-2026", "price_eur_per_mwh": Decimal("91.60")},
        ],
        "historical_volatility_pct": Decimal("0.32"),
        "var_95_eur_per_mwh": Decimal("18.50"),
        "cvar_95_eur_per_mwh": Decimal("24.80"),
    }


# =============================================================================
# 12. Weather Data
# =============================================================================


@pytest.fixture
def sample_weather_data():
    """Create weather data (monthly temperatures) for normalization."""
    return [
        {"period": "2024-01", "avg_temp_c": 1.2, "hdd_base18": 521, "cdd_base18": 0},
        {"period": "2024-02", "avg_temp_c": 2.8, "hdd_base18": 438, "cdd_base18": 0},
        {"period": "2024-03", "avg_temp_c": 6.5, "hdd_base18": 357, "cdd_base18": 0},
        {"period": "2024-04", "avg_temp_c": 10.8, "hdd_base18": 216, "cdd_base18": 0},
        {"period": "2024-05", "avg_temp_c": 15.2, "hdd_base18": 87, "cdd_base18": 0},
        {"period": "2024-06", "avg_temp_c": 19.5, "hdd_base18": 0, "cdd_base18": 45},
        {"period": "2024-07", "avg_temp_c": 22.1, "hdd_base18": 0, "cdd_base18": 127},
        {"period": "2024-08", "avg_temp_c": 21.3, "hdd_base18": 0, "cdd_base18": 102},
        {"period": "2024-09", "avg_temp_c": 16.8, "hdd_base18": 36, "cdd_base18": 0},
        {"period": "2024-10", "avg_temp_c": 10.2, "hdd_base18": 242, "cdd_base18": 0},
        {"period": "2024-11", "avg_temp_c": 5.1, "hdd_base18": 387, "cdd_base18": 0},
        {"period": "2024-12", "avg_temp_c": 1.8, "hdd_base18": 502, "cdd_base18": 0},
    ]


# =============================================================================
# 13. Monthly Consumption with Weather (24 months)
# =============================================================================


@pytest.fixture
def sample_monthly_consumption_weather():
    """Create 24 months of consumption data with HDD/CDD for weather normalization."""
    data = [
        {"period": "2023-01", "electricity_kwh": 142_000, "gas_kwh": 85_000,
         "hdd": 530, "cdd": 0, "avg_temp_c": 0.8},
        {"period": "2023-02", "electricity_kwh": 130_000, "gas_kwh": 78_000,
         "hdd": 448, "cdd": 0, "avg_temp_c": 2.0},
        {"period": "2023-03", "electricity_kwh": 140_000, "gas_kwh": 60_000,
         "hdd": 350, "cdd": 0, "avg_temp_c": 6.8},
        {"period": "2023-04", "electricity_kwh": 150_000, "gas_kwh": 38_000,
         "hdd": 210, "cdd": 0, "avg_temp_c": 11.0},
        {"period": "2023-05", "electricity_kwh": 165_000, "gas_kwh": 15_000,
         "hdd": 80, "cdd": 0, "avg_temp_c": 15.5},
        {"period": "2023-06", "electricity_kwh": 188_000, "gas_kwh": 5_000,
         "hdd": 0, "cdd": 50, "avg_temp_c": 19.8},
        {"period": "2023-07", "electricity_kwh": 202_000, "gas_kwh": 3_000,
         "hdd": 0, "cdd": 135, "avg_temp_c": 22.5},
        {"period": "2023-08", "electricity_kwh": 198_000, "gas_kwh": 3_000,
         "hdd": 0, "cdd": 110, "avg_temp_c": 21.5},
        {"period": "2023-09", "electricity_kwh": 175_000, "gas_kwh": 12_000,
         "hdd": 40, "cdd": 0, "avg_temp_c": 16.5},
        {"period": "2023-10", "electricity_kwh": 150_000, "gas_kwh": 42_000,
         "hdd": 250, "cdd": 0, "avg_temp_c": 10.0},
        {"period": "2023-11", "electricity_kwh": 138_000, "gas_kwh": 68_000,
         "hdd": 395, "cdd": 0, "avg_temp_c": 4.8},
        {"period": "2023-12", "electricity_kwh": 145_000, "gas_kwh": 82_000,
         "hdd": 510, "cdd": 0, "avg_temp_c": 1.5},
        {"period": "2024-01", "electricity_kwh": 140_000, "gas_kwh": 82_000,
         "hdd": 521, "cdd": 0, "avg_temp_c": 1.2},
        {"period": "2024-02", "electricity_kwh": 135_000, "gas_kwh": 75_000,
         "hdd": 438, "cdd": 0, "avg_temp_c": 2.8},
        {"period": "2024-03", "electricity_kwh": 145_000, "gas_kwh": 58_000,
         "hdd": 357, "cdd": 0, "avg_temp_c": 6.5},
        {"period": "2024-04", "electricity_kwh": 155_000, "gas_kwh": 36_000,
         "hdd": 216, "cdd": 0, "avg_temp_c": 10.8},
        {"period": "2024-05", "electricity_kwh": 170_000, "gas_kwh": 14_000,
         "hdd": 87, "cdd": 0, "avg_temp_c": 15.2},
        {"period": "2024-06", "electricity_kwh": 195_000, "gas_kwh": 4_500,
         "hdd": 0, "cdd": 45, "avg_temp_c": 19.5},
        {"period": "2024-07", "electricity_kwh": 210_000, "gas_kwh": 3_000,
         "hdd": 0, "cdd": 127, "avg_temp_c": 22.1},
        {"period": "2024-08", "electricity_kwh": 205_000, "gas_kwh": 3_000,
         "hdd": 0, "cdd": 102, "avg_temp_c": 21.3},
        {"period": "2024-09", "electricity_kwh": 180_000, "gas_kwh": 11_000,
         "hdd": 36, "cdd": 0, "avg_temp_c": 16.8},
        {"period": "2024-10", "electricity_kwh": 155_000, "gas_kwh": 40_000,
         "hdd": 242, "cdd": 0, "avg_temp_c": 10.2},
        {"period": "2024-11", "electricity_kwh": 142_000, "gas_kwh": 65_000,
         "hdd": 387, "cdd": 0, "avg_temp_c": 5.1},
        {"period": "2024-12", "electricity_kwh": 148_000, "gas_kwh": 80_000,
         "hdd": 502, "cdd": 0, "avg_temp_c": 1.8},
    ]
    return data
