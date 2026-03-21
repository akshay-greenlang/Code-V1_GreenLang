# -*- coding: utf-8 -*-
"""
PACK-035 Energy Benchmark Pack - Shared Test Fixtures (conftest.py)
=====================================================================

Provides pytest fixtures for the PACK-035 test suite including:
  - Dynamic module loading via importlib (no package install needed)
  - Pack manifest and configuration fixtures
  - Sample facility profiles, energy data, weather data, peer groups,
    portfolio data, and benchmark reference data

All fixtures use importlib.util.spec_from_file_location to load modules
directly from the pack source tree, enabling test execution without
installing the pack as a Python package.

Fixture Categories:
  1. Paths and YAML data
  2. Configuration objects
  3. Facility profiles
  4. Energy meter data (12 months)
  5. Weather data (HDD/CDD)
  6. Peer group data
  7. Portfolio data (10 facilities)
  8. Sector benchmark data (CIBSE TM46)
  9. Regression data
  10. Performance rating data

Author:  GreenLang Platform Team (GL-TestEngineer)
Pack:    PACK-035 Energy Benchmark
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
    "eui_calculator": "eui_calculator_engine.py",
    "peer_comparison": "peer_comparison_engine.py",
    "sector_benchmark": "sector_benchmark_engine.py",
    "weather_normalisation": "weather_normalisation_engine.py",
    "energy_performance_gap": "energy_performance_gap_engine.py",
    "portfolio_benchmark": "portfolio_benchmark_engine.py",
    "regression_analysis": "regression_analysis_engine.py",
    "performance_rating": "performance_rating_engine.py",
    "trend_analysis": "trend_analysis_engine.py",
    "benchmark_report": "benchmark_report_engine.py",
}

# Engine class names that should exist in each engine module
ENGINE_CLASSES = {
    "eui_calculator": "EUICalculatorEngine",
    "peer_comparison": "PeerComparisonEngine",
    "sector_benchmark": "SectorBenchmarkEngine",
    "weather_normalisation": "WeatherNormalisationEngine",
    "energy_performance_gap": "EnergyPerformanceGapEngine",
    "portfolio_benchmark": "PortfolioBenchmarkEngine",
    "regression_analysis": "RegressionAnalysisEngine",
    "performance_rating": "PerformanceRatingEngine",
    "trend_analysis": "TrendAnalysisEngine",
    "benchmark_report": "BenchmarkReportEngine",
}

# Workflow file mapping
WORKFLOW_FILES = {
    "initial_benchmark": "initial_benchmark_workflow.py",
    "continuous_monitoring": "continuous_monitoring_workflow.py",
    "peer_comparison": "peer_comparison_workflow.py",
    "portfolio_benchmark": "portfolio_benchmark_workflow.py",
    "performance_gap": "performance_gap_workflow.py",
    "regulatory_compliance": "regulatory_compliance_workflow.py",
    "target_setting": "target_setting_workflow.py",
    "full_assessment": "full_assessment_workflow.py",
}

# Workflow class names
WORKFLOW_CLASSES = {
    "initial_benchmark": "InitialBenchmarkWorkflow",
    "continuous_monitoring": "ContinuousMonitoringWorkflow",
    "peer_comparison": "PeerComparisonWorkflow",
    "portfolio_benchmark": "PortfolioBenchmarkWorkflow",
    "performance_gap": "PerformanceGapWorkflow",
    "regulatory_compliance": "RegulatoryComplianceWorkflow",
    "target_setting": "TargetSettingWorkflow",
    "full_assessment": "FullAssessmentWorkflow",
}

# Expected phase counts per workflow
WORKFLOW_PHASE_COUNTS = {
    "initial_benchmark": 4,
    "continuous_monitoring": 4,
    "peer_comparison": 4,
    "portfolio_benchmark": 5,
    "performance_gap": 4,
    "regulatory_compliance": 4,
    "target_setting": 4,
    "full_assessment": 6,
}

# Template file mapping
TEMPLATE_FILES = {
    "eui_benchmark_report": "eui_benchmark_report.py",
    "peer_comparison_report": "peer_comparison_report.py",
    "sector_benchmark_report": "sector_benchmark_report.py",
    "energy_performance_certificate": "energy_performance_certificate.py",
    "portfolio_dashboard": "portfolio_dashboard.py",
    "gap_analysis_report": "gap_analysis_report.py",
    "target_tracking_report": "target_tracking_report.py",
    "regulatory_compliance_report": "regulatory_compliance_report.py",
    "trend_analysis_report": "trend_analysis_report.py",
    "executive_summary_report": "executive_summary_report.py",
}

# Template class names
TEMPLATE_CLASSES = {
    "eui_benchmark_report": "EUIBenchmarkReportTemplate",
    "peer_comparison_report": "PeerComparisonReportTemplate",
    "sector_benchmark_report": "SectorBenchmarkReportTemplate",
    "energy_performance_certificate": "EnergyPerformanceCertificateTemplate",
    "portfolio_dashboard": "PortfolioDashboardTemplate",
    "gap_analysis_report": "GapAnalysisReportTemplate",
    "target_tracking_report": "TargetTrackingReportTemplate",
    "regulatory_compliance_report": "RegulatoryComplianceReportTemplate",
    "trend_analysis_report": "TrendAnalysisReportTemplate",
    "executive_summary_report": "ExecutiveSummaryReportTemplate",
}

# Integration file mapping
INTEGRATION_FILES = {
    "pack_orchestrator": "pack_orchestrator.py",
    "mrv_benchmark_bridge": "mrv_benchmark_bridge.py",
    "data_benchmark_bridge": "data_benchmark_bridge.py",
    "pack_031_bridge": "pack_031_bridge.py",
    "pack_032_bridge": "pack_032_bridge.py",
    "pack_033_bridge": "pack_033_bridge.py",
    "energy_star_bridge": "energy_star_bridge.py",
    "weather_service_bridge": "weather_service_bridge.py",
    "epc_registry_bridge": "epc_registry_bridge.py",
    "benchmark_database_bridge": "benchmark_database_bridge.py",
    "health_check": "health_check.py",
    "setup_wizard": "setup_wizard.py",
}

# Integration class names
INTEGRATION_CLASSES = {
    "pack_orchestrator": "EnergyBenchmarkOrchestrator",
    "mrv_benchmark_bridge": "MRVBenchmarkBridge",
    "data_benchmark_bridge": "DataBenchmarkBridge",
    "pack_031_bridge": "Pack031Bridge",
    "pack_032_bridge": "Pack032Bridge",
    "pack_033_bridge": "Pack033Bridge",
    "energy_star_bridge": "EnergyStarBridge",
    "weather_service_bridge": "WeatherServiceBridge",
    "epc_registry_bridge": "EPCRegistryBridge",
    "benchmark_database_bridge": "BenchmarkDatabaseBridge",
    "health_check": "HealthCheck",
    "setup_wizard": "SetupWizard",
}

# Preset names
PRESET_NAMES = [
    "commercial_office",
    "industrial_manufacturing",
    "retail_store",
    "warehouse_logistics",
    "healthcare_facility",
    "educational_campus",
    "data_center",
    "multi_site_portfolio",
]


# =============================================================================
# Helper: Dynamic Module Loader
# =============================================================================


def _load_module(module_name: str, file_name: str, subdir: str = "engines"):
    """Load a module dynamically using importlib.util.spec_from_file_location.

    This avoids the need to install PACK-035 as a Python package. The module
    is loaded from the pack source tree and added to sys.modules under a
    unique key to prevent collisions.

    Args:
        module_name: Logical name for the module (used as sys.modules key prefix).
        file_name: File name of the Python module (e.g., "eui_calculator_engine.py").
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
            f"Ensure PACK-035 source files are present."
        )

    # Create a unique module key to avoid collisions
    full_module_name = f"pack035_test.{subdir}.{module_name}"

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
        engine_key: Engine key from ENGINE_FILES (e.g., "eui_calculator").

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
    """Return the absolute path to the PACK-035 root directory."""
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
        pytest.skip("pack_config.py not found or cannot load")


@pytest.fixture
def pack_config(config_module):
    """Create an EnergyBenchmarkConfig with default values."""
    cls = getattr(config_module, "EnergyBenchmarkConfig", None)
    if cls is None:
        pytest.skip("EnergyBenchmarkConfig not found in pack_config")
    return cls()


# =============================================================================
# 3. Facility Profile Fixtures
# =============================================================================


@pytest.fixture
def sample_facility_profile():
    """Create a sample commercial office facility with realistic data.

    Facility: GreenLang HQ Office (Berlin, Germany)
    Floor area: 5,000 m2 GIA
    Building type: Office
    Climate zone: ASHRAE 4A (mixed-humid)
    Year built: 2010
    Occupancy: 200 people, 55 hours/week
    """
    return {
        "facility_id": "FAC-035-DE-001",
        "facility_name": "GreenLang HQ Office",
        "building_type": "office",
        "climate_zone": "4A",
        "country_code": "DE",
        "city": "Berlin",
        "latitude": 52.5200,
        "longitude": 13.4050,
        "gross_internal_area_m2": 5000.0,
        "floor_area_type": "gia",
        "year_built": 2010,
        "typical_occupancy": 200,
        "operating_hours_per_week": 55.0,
    }


@pytest.fixture
def sample_retail_profile():
    """Create a sample retail store facility."""
    return {
        "facility_id": "FAC-035-UK-002",
        "facility_name": "GreenLang Retail Store",
        "building_type": "retail",
        "climate_zone": "4A",
        "country_code": "GB",
        "city": "London",
        "latitude": 51.5074,
        "longitude": -0.1278,
        "gross_internal_area_m2": 2500.0,
        "floor_area_type": "gla",
        "year_built": 2005,
        "typical_occupancy": 30,
        "operating_hours_per_week": 65.0,
    }


@pytest.fixture
def sample_warehouse_profile():
    """Create a sample warehouse facility."""
    return {
        "facility_id": "FAC-035-NL-003",
        "facility_name": "GreenLang Distribution Centre",
        "building_type": "warehouse",
        "climate_zone": "5A",
        "country_code": "NL",
        "city": "Rotterdam",
        "latitude": 51.9244,
        "longitude": 4.4777,
        "gross_internal_area_m2": 15000.0,
        "floor_area_type": "gia",
        "year_built": 2015,
        "typical_occupancy": 50,
        "operating_hours_per_week": 60.0,
    }


# =============================================================================
# 4. Energy Meter Data Fixture (12 months)
# =============================================================================


@pytest.fixture
def sample_energy_data():
    """Create 12 months of electricity and gas consumption for an office.

    Based on a 5,000 m2 office in Berlin with typical seasonal variation:
    - Higher electricity in summer (cooling)
    - Higher gas in winter (heating)
    Annual total: ~497,000 kWh electricity + ~206,000 kWh gas = ~703,000 kWh
    Site EUI: ~703,000 / 5,000 = ~140.6 kWh/m2/yr
    """
    return [
        {"month": "2025-01", "electricity_kwh": 42000, "gas_kwh": 35000, "cost_eur": 10850},
        {"month": "2025-02", "electricity_kwh": 40000, "gas_kwh": 32000, "cost_eur": 10160},
        {"month": "2025-03", "electricity_kwh": 38000, "gas_kwh": 25000, "cost_eur": 8890},
        {"month": "2025-04", "electricity_kwh": 36000, "gas_kwh": 15000, "cost_eur": 7230},
        {"month": "2025-05", "electricity_kwh": 37000, "gas_kwh": 8000, "cost_eur": 6430},
        {"month": "2025-06", "electricity_kwh": 42000, "gas_kwh": 3000, "cost_eur": 6540},
        {"month": "2025-07", "electricity_kwh": 48000, "gas_kwh": 2000, "cost_eur": 7280},
        {"month": "2025-08", "electricity_kwh": 47000, "gas_kwh": 2000, "cost_eur": 7130},
        {"month": "2025-09", "electricity_kwh": 40000, "gas_kwh": 5000, "cost_eur": 6450},
        {"month": "2025-10", "electricity_kwh": 38000, "gas_kwh": 18000, "cost_eur": 7920},
        {"month": "2025-11", "electricity_kwh": 40000, "gas_kwh": 28000, "cost_eur": 9600},
        {"month": "2025-12", "electricity_kwh": 43000, "gas_kwh": 33000, "cost_eur": 10720},
    ]


@pytest.fixture
def sample_energy_data_annual_totals(sample_energy_data):
    """Pre-computed annual totals from sample_energy_data."""
    total_elec = sum(m["electricity_kwh"] for m in sample_energy_data)
    total_gas = sum(m["gas_kwh"] for m in sample_energy_data)
    total_cost = sum(m["cost_eur"] for m in sample_energy_data)
    return {
        "total_electricity_kwh": total_elec,
        "total_gas_kwh": total_gas,
        "total_energy_kwh": total_elec + total_gas,
        "total_cost_eur": total_cost,
    }


# =============================================================================
# 5. Weather Data Fixture (12 months HDD/CDD)
# =============================================================================


@pytest.fixture
def sample_weather_data():
    """Create 12 months of heating and cooling degree-day data.

    Location: Berlin, Germany (base temp 18C for HDD, 18C for CDD)
    Source: Typical Meteorological Year reference
    """
    return [
        {"month": "2025-01", "hdd_18": 520, "cdd_18": 0, "mean_temp_c": 1.2},
        {"month": "2025-02", "hdd_18": 460, "cdd_18": 0, "mean_temp_c": 2.5},
        {"month": "2025-03", "hdd_18": 380, "cdd_18": 0, "mean_temp_c": 5.8},
        {"month": "2025-04", "hdd_18": 220, "cdd_18": 5, "mean_temp_c": 10.5},
        {"month": "2025-05", "hdd_18": 100, "cdd_18": 25, "mean_temp_c": 15.2},
        {"month": "2025-06", "hdd_18": 30, "cdd_18": 60, "mean_temp_c": 18.5},
        {"month": "2025-07", "hdd_18": 10, "cdd_18": 95, "mean_temp_c": 21.3},
        {"month": "2025-08", "hdd_18": 15, "cdd_18": 85, "mean_temp_c": 20.7},
        {"month": "2025-09", "hdd_18": 80, "cdd_18": 40, "mean_temp_c": 16.2},
        {"month": "2025-10", "hdd_18": 240, "cdd_18": 5, "mean_temp_c": 10.0},
        {"month": "2025-11", "hdd_18": 390, "cdd_18": 0, "mean_temp_c": 5.0},
        {"month": "2025-12", "hdd_18": 500, "cdd_18": 0, "mean_temp_c": 1.8},
    ]


# =============================================================================
# 6. Peer Group Data Fixture
# =============================================================================


@pytest.fixture
def sample_peer_group():
    """Create a peer group with 50 office buildings for percentile comparison.

    EUI distribution based on CIBSE TM46 and ENERGY STAR data for offices:
    - Median EUI: ~180 kWh/m2/yr
    - P25: ~130 kWh/m2/yr
    - P75: ~240 kWh/m2/yr
    """
    import random
    random.seed(42)
    peers = []
    for i in range(50):
        eui = random.gauss(180, 50)
        eui = max(60, min(400, eui))
        peers.append({
            "facility_id": f"PEER-{i+1:03d}",
            "facility_name": f"Peer Office {i+1}",
            "building_type": "office",
            "country_code": "DE",
            "gross_floor_area_m2": random.uniform(1000, 20000),
            "eui_kwh_per_m2_yr": round(eui, 1),
            "year_built": random.randint(1970, 2020),
        })
    return peers


# =============================================================================
# 7. Portfolio Data Fixture (10 facilities)
# =============================================================================


@pytest.fixture
def sample_portfolio():
    """Create a portfolio of 10 facilities with mixed building types.

    Covers offices, retail, warehouse, and a data centre to test
    portfolio-level aggregation, ranking, and trend analysis.
    """
    return [
        {
            "facility_id": "PF-001",
            "facility_name": "Berlin HQ Office",
            "building_type": "office",
            "region": "EMEA",
            "country": "DE",
            "business_unit": "Corporate",
            "gross_floor_area_m2": 5000.0,
            "energy_consumption_kwh": 700000.0,
            "eui_kwh_per_m2": 140.0,
            "carbon_emissions_kgco2": 280000.0,
            "energy_cost_eur": 105000.0,
            "reporting_year": 2025,
            "historical_eui": {2022: 155.0, 2023: 150.0, 2024: 145.0, 2025: 140.0},
        },
        {
            "facility_id": "PF-002",
            "facility_name": "London Sales Office",
            "building_type": "office",
            "region": "EMEA",
            "country": "GB",
            "business_unit": "Sales",
            "gross_floor_area_m2": 3000.0,
            "energy_consumption_kwh": 510000.0,
            "eui_kwh_per_m2": 170.0,
            "carbon_emissions_kgco2": 200000.0,
            "energy_cost_eur": 76500.0,
            "reporting_year": 2025,
            "historical_eui": {2022: 185.0, 2023: 180.0, 2024: 175.0, 2025: 170.0},
        },
        {
            "facility_id": "PF-003",
            "facility_name": "Paris Retail Store",
            "building_type": "retail",
            "region": "EMEA",
            "country": "FR",
            "business_unit": "Retail",
            "gross_floor_area_m2": 2000.0,
            "energy_consumption_kwh": 500000.0,
            "eui_kwh_per_m2": 250.0,
            "carbon_emissions_kgco2": 175000.0,
            "energy_cost_eur": 75000.0,
            "reporting_year": 2025,
            "historical_eui": {2023: 270.0, 2024: 260.0, 2025: 250.0},
        },
        {
            "facility_id": "PF-004",
            "facility_name": "Rotterdam Warehouse",
            "building_type": "warehouse",
            "region": "EMEA",
            "country": "NL",
            "business_unit": "Logistics",
            "gross_floor_area_m2": 15000.0,
            "energy_consumption_kwh": 900000.0,
            "eui_kwh_per_m2": 60.0,
            "carbon_emissions_kgco2": 360000.0,
            "energy_cost_eur": 135000.0,
            "reporting_year": 2025,
            "historical_eui": {2022: 68.0, 2023: 65.0, 2024: 63.0, 2025: 60.0},
        },
        {
            "facility_id": "PF-005",
            "facility_name": "Frankfurt Data Centre",
            "building_type": "data_centre",
            "region": "EMEA",
            "country": "DE",
            "business_unit": "Technology",
            "gross_floor_area_m2": 2000.0,
            "energy_consumption_kwh": 3200000.0,
            "eui_kwh_per_m2": 1600.0,
            "carbon_emissions_kgco2": 1280000.0,
            "energy_cost_eur": 480000.0,
            "reporting_year": 2025,
            "historical_eui": {2022: 1750.0, 2023: 1700.0, 2024: 1650.0, 2025: 1600.0},
        },
        {
            "facility_id": "PF-006",
            "facility_name": "Munich R&D Lab",
            "building_type": "laboratory",
            "region": "EMEA",
            "country": "DE",
            "business_unit": "R&D",
            "gross_floor_area_m2": 4000.0,
            "energy_consumption_kwh": 1200000.0,
            "eui_kwh_per_m2": 300.0,
            "carbon_emissions_kgco2": 480000.0,
            "energy_cost_eur": 180000.0,
            "reporting_year": 2025,
            "historical_eui": {2023: 320.0, 2024: 310.0, 2025: 300.0},
        },
        {
            "facility_id": "PF-007",
            "facility_name": "Milan Office",
            "building_type": "office",
            "region": "EMEA",
            "country": "IT",
            "business_unit": "Corporate",
            "gross_floor_area_m2": 2500.0,
            "energy_consumption_kwh": 425000.0,
            "eui_kwh_per_m2": 170.0,
            "carbon_emissions_kgco2": 170000.0,
            "energy_cost_eur": 63750.0,
            "reporting_year": 2025,
            "historical_eui": {2023: 180.0, 2024: 175.0, 2025: 170.0},
        },
        {
            "facility_id": "PF-008",
            "facility_name": "Madrid Retail Store",
            "building_type": "retail",
            "region": "EMEA",
            "country": "ES",
            "business_unit": "Retail",
            "gross_floor_area_m2": 1800.0,
            "energy_consumption_kwh": 468000.0,
            "eui_kwh_per_m2": 260.0,
            "carbon_emissions_kgco2": 163800.0,
            "energy_cost_eur": 70200.0,
            "reporting_year": 2025,
            "historical_eui": {2023: 280.0, 2024: 270.0, 2025: 260.0},
        },
        {
            "facility_id": "PF-009",
            "facility_name": "Warsaw Warehouse",
            "building_type": "warehouse",
            "region": "EMEA",
            "country": "PL",
            "business_unit": "Logistics",
            "gross_floor_area_m2": 12000.0,
            "energy_consumption_kwh": 840000.0,
            "eui_kwh_per_m2": 70.0,
            "carbon_emissions_kgco2": 336000.0,
            "energy_cost_eur": 126000.0,
            "reporting_year": 2025,
            "historical_eui": {2022: 80.0, 2023: 77.0, 2024: 73.0, 2025: 70.0},
        },
        {
            "facility_id": "PF-010",
            "facility_name": "Dublin Office",
            "building_type": "office",
            "region": "EMEA",
            "country": "IE",
            "business_unit": "Sales",
            "gross_floor_area_m2": 1500.0,
            "energy_consumption_kwh": 225000.0,
            "eui_kwh_per_m2": 150.0,
            "carbon_emissions_kgco2": 90000.0,
            "energy_cost_eur": 33750.0,
            "reporting_year": 2025,
            "historical_eui": {2023: 165.0, 2024: 158.0, 2025: 150.0},
        },
    ]


# =============================================================================
# 8. Sector Benchmark Data (CIBSE TM46)
# =============================================================================


@pytest.fixture
def sample_benchmark_data():
    """Create sector benchmark data based on CIBSE TM46:2008.

    Reference values for UK typical and good-practice buildings.
    Source: CIBSE TM46:2008, Table 1 - Benchmarks for display energy certificates.
    """
    return {
        "office_air_conditioned": {
            "typical_eui_kwh_m2": 358,
            "good_practice_eui_kwh_m2": 228,
            "source": "CIBSE TM46:2008 Table 1, Category 1 (air-conditioned office)",
        },
        "office_naturally_ventilated": {
            "typical_eui_kwh_m2": 120,
            "good_practice_eui_kwh_m2": 85,
            "source": "CIBSE TM46:2008 Table 1, Category 2 (naturally ventilated office)",
        },
        "general_retail": {
            "typical_eui_kwh_m2": 345,
            "good_practice_eui_kwh_m2": 210,
            "source": "CIBSE TM46:2008 Table 1, Category 4 (general retail)",
        },
        "warehouse_distribution": {
            "typical_eui_kwh_m2": 120,
            "good_practice_eui_kwh_m2": 90,
            "source": "CIBSE TM46:2008 Table 1, Category 8 (distribution warehouse)",
        },
        "hospital": {
            "typical_eui_kwh_m2": 553,
            "good_practice_eui_kwh_m2": 392,
            "source": "CIBSE TM46:2008 Table 1, Category 16 (hospital, clinical)",
        },
        "primary_school": {
            "typical_eui_kwh_m2": 150,
            "good_practice_eui_kwh_m2": 113,
            "source": "CIBSE TM46:2008 Table 1, Category 11 (primary school)",
        },
        "secondary_school": {
            "typical_eui_kwh_m2": 165,
            "good_practice_eui_kwh_m2": 120,
            "source": "CIBSE TM46:2008 Table 1, Category 12 (secondary school)",
        },
        "hotel": {
            "typical_eui_kwh_m2": 350,
            "good_practice_eui_kwh_m2": 250,
            "source": "CIBSE TM46:2008 Table 1, Category 14 (hotel)",
        },
    }


# =============================================================================
# 9. Regression Data Fixture
# =============================================================================


@pytest.fixture
def sample_regression_data():
    """Create data for regression analysis: energy vs. HDD and CDD.

    12 months of energy consumption correlated with degree days.
    Suitable for 3P (heating) and 5P (heating+cooling) models.
    """
    return [
        {"month": "2025-01", "energy_kwh": 77000, "hdd": 520, "cdd": 0, "production_hours": 176},
        {"month": "2025-02", "energy_kwh": 72000, "hdd": 460, "cdd": 0, "production_hours": 160},
        {"month": "2025-03", "energy_kwh": 63000, "hdd": 380, "cdd": 0, "production_hours": 176},
        {"month": "2025-04", "energy_kwh": 51000, "hdd": 220, "cdd": 5, "production_hours": 168},
        {"month": "2025-05", "energy_kwh": 45000, "hdd": 100, "cdd": 25, "production_hours": 176},
        {"month": "2025-06", "energy_kwh": 45000, "hdd": 30, "cdd": 60, "production_hours": 168},
        {"month": "2025-07", "energy_kwh": 50000, "hdd": 10, "cdd": 95, "production_hours": 176},
        {"month": "2025-08", "energy_kwh": 49000, "hdd": 15, "cdd": 85, "production_hours": 168},
        {"month": "2025-09", "energy_kwh": 45000, "hdd": 80, "cdd": 40, "production_hours": 176},
        {"month": "2025-10", "energy_kwh": 56000, "hdd": 240, "cdd": 5, "production_hours": 184},
        {"month": "2025-11", "energy_kwh": 68000, "hdd": 390, "cdd": 0, "production_hours": 168},
        {"month": "2025-12", "energy_kwh": 76000, "hdd": 500, "cdd": 0, "production_hours": 176},
    ]


# =============================================================================
# 10. Performance Rating Data Fixture
# =============================================================================


@pytest.fixture
def sample_rating_data():
    """Create data for performance rating tests.

    EPC band thresholds for different rating schemes.
    """
    return {
        "epc_thresholds_uk": {
            "A": {"max_eui": 25},
            "B": {"max_eui": 50},
            "C": {"max_eui": 100},
            "D": {"max_eui": 150},
            "E": {"max_eui": 200},
            "F": {"max_eui": 250},
            "G": {"max_eui": 999},
        },
        "energy_star_median_source_eui_office": 142.0,
        "nabers_5_star_threshold": 90.0,
        "crrem_2030_office_target_kgco2_m2": 25.0,
        "crrem_2050_office_target_kgco2_m2": 5.0,
    }


# =============================================================================
# 11. Sample Report Data Fixture
# =============================================================================


@pytest.fixture
def sample_report_data(sample_facility_profile, sample_energy_data_annual_totals):
    """Create sample data for template rendering tests."""
    return {
        "facility_id": sample_facility_profile["facility_id"],
        "facility_name": sample_facility_profile["facility_name"],
        "building_type": sample_facility_profile["building_type"],
        "floor_area_m2": sample_facility_profile["gross_internal_area_m2"],
        "reporting_year": 2025,
        "site_eui_kwh_m2": round(
            sample_energy_data_annual_totals["total_energy_kwh"]
            / sample_facility_profile["gross_internal_area_m2"],
            1,
        ),
        "source_eui_kwh_m2": 280.0,
        "primary_eui_kwh_m2": 250.0,
        "total_energy_kwh": sample_energy_data_annual_totals["total_energy_kwh"],
        "total_cost_eur": sample_energy_data_annual_totals["total_cost_eur"],
        "epc_rating": "C",
        "energy_star_score": 72,
        "peer_percentile": 65.0,
        "peer_count": 50,
        "provenance_hash": "a" * 64,
    }
