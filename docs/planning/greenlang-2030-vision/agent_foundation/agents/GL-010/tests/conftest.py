# -*- coding: utf-8 -*-
"""
GL-010 EMISSIONWATCH Test Suite - Shared Pytest Fixtures.

This module provides comprehensive test fixtures for the EmissionsComplianceAgent
test suite, including sample emissions data, CEMS fixtures, fuel compositions,
regulatory limits, mock connectors, and test configurations.

Coverage Target: 90%+
Test Count Target: 200+ tests

Standards Compliance:
- EPA 40 CFR Parts 60, 75
- EU Industrial Emissions Directive 2010/75/EU
- China MEE GB 13223-2011

Author: GreenLang Foundation Test Engineering
Version: 1.0.0
"""

import asyncio
import hashlib
import json
import os
import sys
import tempfile
import uuid
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest

# Add parent directories to path for imports
TEST_DIR = Path(__file__).parent
AGENT_DIR = TEST_DIR.parent
sys.path.insert(0, str(AGENT_DIR))
sys.path.insert(0, str(AGENT_DIR.parent.parent))

# Import agent modules
try:
    from config import (
        EmissionsComplianceConfig,
        NOxConfig,
        SOxConfig,
        CO2Config,
        PMConfig,
        CEMSConfig,
        AlertConfig,
        ReportingConfig,
        RegulatoryLimitsConfig,
        Jurisdiction,
        PollutantType,
        AlertSeverity,
        ReportFormat,
    )
    from tools import (
        EmissionsComplianceTools,
        NOxEmissionsResult,
        SOxEmissionsResult,
        CO2EmissionsResult,
        PMEmissionsResult,
        ComplianceCheckResult,
        ViolationResult,
        RegulatoryReportResult,
        ExceedancePredictionResult,
        EmissionFactorResult,
        DispersionResult,
        AuditTrailResult,
        FuelAnalysisResult,
        EMISSIONS_TOOL_SCHEMAS,
        AP42_EMISSION_FACTORS,
        F_FACTORS,
        REGULATORY_LIMITS,
        MOLECULAR_WEIGHTS,
    )
    from emissions_compliance_orchestrator import (
        EmissionsComplianceOrchestrator,
        OperationMode,
        ComplianceStatus,
        ValidationStatus,
        DataQualityCode,
        ThreadSafeCache,
        PerformanceMetrics,
        RetryHandler,
        create_orchestrator,
    )
    from calculators.constants import (
        MW,
        F_FACTORS as CONST_F_FACTORS,
        GWP_100,
        O2_REFERENCE,
        AVERAGING_PERIODS,
    )
except ImportError as e:
    # Provide fallback for module import errors during initial setup
    print(f"Warning: Import error during fixture setup: {e}")


# =============================================================================
# PYTEST CONFIGURATION
# =============================================================================

def pytest_configure(config):
    """Configure pytest markers."""
    config.addinivalue_line("markers", "unit: Unit tests")
    config.addinivalue_line("markers", "integration: Integration tests")
    config.addinivalue_line("markers", "e2e: End-to-end tests")
    config.addinivalue_line("markers", "performance: Performance tests")
    config.addinivalue_line("markers", "determinism: Determinism tests")
    config.addinivalue_line("markers", "compliance: Regulatory compliance tests")
    config.addinivalue_line("markers", "slow: Slow tests")
    config.addinivalue_line("markers", "asyncio: Async tests")


# =============================================================================
# EVENT LOOP FIXTURES
# =============================================================================

@pytest.fixture(scope="session")
def event_loop():
    """Create event loop for async tests."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


# =============================================================================
# CONFIGURATION FIXTURES
# =============================================================================

@pytest.fixture
def base_config() -> EmissionsComplianceConfig:
    """Create base configuration for testing."""
    return EmissionsComplianceConfig(
        agent_id="GL-010-TEST",
        codename="EMISSIONWATCH-TEST",
        version="1.0.0-test",
        jurisdiction="EPA",
        deterministic=True,
        llm_temperature=0.0,
        llm_seed=42,
        nox_limit_ppm=50.0,
        sox_limit_ppm=100.0,
        co2_limit_tons_hr=50.0,
        pm_limit_mg_m3=30.0,
        opacity_limit_percent=20.0,
        cache_ttl_seconds=60,
        cache_max_size=100,
        max_retries=1,
        enable_error_recovery=True,
        enable_monitoring=False,
        enable_email_alerts=False,
        enable_sms_alerts=False,
        enable_webhook_alerts=False,
    )


@pytest.fixture
def epa_config(base_config) -> EmissionsComplianceConfig:
    """Create EPA-specific configuration."""
    base_config.jurisdiction = "EPA"
    base_config.nox_limit_ppm = 50.0
    base_config.sox_limit_ppm = 100.0
    return base_config


@pytest.fixture
def eu_ied_config(base_config) -> EmissionsComplianceConfig:
    """Create EU IED-specific configuration."""
    base_config.jurisdiction = "EU_IED"
    base_config.nox_limit_ppm = 100.0
    base_config.sox_limit_ppm = 150.0
    return base_config


@pytest.fixture
def china_mee_config(base_config) -> EmissionsComplianceConfig:
    """Create China MEE-specific configuration."""
    base_config.jurisdiction = "CHINA_MEE"
    base_config.nox_limit_ppm = 50.0
    base_config.sox_limit_ppm = 35.0
    return base_config


@pytest.fixture
def nox_config() -> NOxConfig:
    """Create NOx configuration."""
    return NOxConfig(
        limit_ppm=50.0,
        limit_lb_mmbtu=0.10,
        limit_mg_nm3=100.0,
        reference_o2_percent=3.0,
        averaging_period_minutes=60,
        scr_efficiency_percent=90.0,
        sncr_efficiency_percent=50.0,
        warning_threshold_percent=80.0,
    )


@pytest.fixture
def sox_config() -> SOxConfig:
    """Create SOx configuration."""
    return SOxConfig(
        limit_ppm=100.0,
        limit_lb_mmbtu=0.15,
        limit_mg_nm3=150.0,
        max_fuel_sulfur_percent=0.5,
        fgd_efficiency_percent=95.0,
        averaging_period_minutes=60,
        warning_threshold_percent=80.0,
    )


@pytest.fixture
def cems_config() -> CEMSConfig:
    """Create CEMS configuration."""
    return CEMSConfig(
        sampling_frequency_seconds=15,
        averaging_period_minutes=15,
        min_data_availability_percent=90.0,
        substitute_data_enabled=True,
        daily_calibration_required=True,
        calibration_drift_limit_percent=2.5,
        rata_frequency_days=365,
        nox_analyzer_range_ppm=500.0,
        sox_analyzer_range_ppm=1000.0,
        co2_analyzer_range_percent=20.0,
        o2_analyzer_range_percent=25.0,
        flow_rate_range_scfm=100000.0,
    )


# =============================================================================
# EMISSIONS DATA FIXTURES
# =============================================================================

@pytest.fixture
def sample_cems_data() -> Dict[str, Any]:
    """Create sample CEMS data for testing."""
    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "nox_ppm": 45.0,
        "sox_ppm": 75.0,
        "co2_percent": 10.5,
        "co_ppm": 50.0,
        "o2_percent": 3.0,
        "flow_rate_dscfm": 50000.0,
        "temperature_f": 350.0,
        "pressure_inhg": 29.92,
        "opacity_percent": 5.0,
        "pm_mg_m3": 15.0,
        "quality_code": "valid",
        "expected_hours": 720,
        "valid_hours": 700,
        "last_calibration_date": (datetime.now(timezone.utc) - timedelta(days=1)).isoformat(),
        "last_rata_date": (datetime.now(timezone.utc) - timedelta(days=180)).isoformat(),
    }


@pytest.fixture
def high_nox_cems_data() -> Dict[str, Any]:
    """Create CEMS data with high NOx (violation scenario)."""
    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "nox_ppm": 150.0,  # Above typical limits
        "sox_ppm": 80.0,
        "co2_percent": 12.0,
        "co_ppm": 75.0,
        "o2_percent": 4.0,
        "flow_rate_dscfm": 55000.0,
        "temperature_f": 400.0,
        "pressure_inhg": 29.85,
        "opacity_percent": 12.0,
        "pm_mg_m3": 25.0,
        "quality_code": "valid",
    }


@pytest.fixture
def low_emissions_cems_data() -> Dict[str, Any]:
    """Create CEMS data with low emissions (compliant scenario)."""
    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "nox_ppm": 15.0,
        "sox_ppm": 20.0,
        "co2_percent": 8.0,
        "co_ppm": 25.0,
        "o2_percent": 5.0,
        "flow_rate_dscfm": 45000.0,
        "temperature_f": 300.0,
        "pressure_inhg": 30.00,
        "opacity_percent": 2.0,
        "pm_mg_m3": 5.0,
        "quality_code": "valid",
    }


@pytest.fixture
def missing_data_cems() -> Dict[str, Any]:
    """Create CEMS data with missing fields."""
    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "nox_ppm": 40.0,
        # Missing sox_ppm
        "co2_percent": 10.0,
        "o2_percent": 3.0,
        # Missing flow_rate_dscfm
        "quality_code": "substitute",
    }


@pytest.fixture
def invalid_cems_data() -> Dict[str, Any]:
    """Create invalid CEMS data for error handling tests."""
    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "nox_ppm": -10.0,  # Invalid negative value
        "sox_ppm": 50000.0,  # Unreasonably high
        "co2_percent": 30.0,  # Too high
        "o2_percent": 25.0,  # Above atmospheric
        "flow_rate_dscfm": 0.0,  # Zero flow
        "quality_code": "invalid",
    }


@pytest.fixture
def cems_data_series() -> List[Dict[str, Any]]:
    """Create time series CEMS data for trend analysis."""
    base_time = datetime.now(timezone.utc) - timedelta(hours=24)
    data_points = []

    for i in range(24):
        timestamp = base_time + timedelta(hours=i)
        # Simulate varying emissions with slight upward trend
        data_points.append({
            "timestamp": timestamp.isoformat(),
            "nox_ppm": 30.0 + i * 0.5 + (i % 3) * 2,
            "sox_ppm": 50.0 + i * 0.3,
            "co2_percent": 9.5 + i * 0.05,
            "pm_mg_m3": 10.0 + i * 0.2,
            "o2_percent": 3.0 + (i % 5) * 0.1,
            "flow_rate_dscfm": 50000.0 + i * 100,
            "load_percent": 70 + i,
            "fuel_sulfur_percent": 0.3 + (i % 4) * 0.02,
        })

    return data_points


# =============================================================================
# FUEL DATA FIXTURES
# =============================================================================

@pytest.fixture
def natural_gas_fuel_data() -> Dict[str, Any]:
    """Create natural gas fuel data."""
    return {
        "fuel_type": "natural_gas",
        "heat_input_mmbtu_hr": 100.0,
        "heating_value_btu_lb": 23000.0,
        "flow_rate_scfh": 100000.0,
        "carbon_percent": 75.0,
        "hydrogen_percent": 24.0,
        "sulfur_percent": 0.0006,
        "nitrogen_percent": 0.01,
        "oxygen_percent": 0.5,
        "ash_percent": 0.0,
        "moisture_percent": 0.0,
    }


@pytest.fixture
def fuel_oil_no2_data() -> Dict[str, Any]:
    """Create No. 2 fuel oil data."""
    return {
        "fuel_type": "fuel_oil_no2",
        "heat_input_mmbtu_hr": 80.0,
        "heating_value_btu_lb": 19500.0,
        "flow_rate_gph": 500.0,
        "carbon_percent": 87.0,
        "hydrogen_percent": 12.5,
        "sulfur_percent": 0.5,
        "nitrogen_percent": 0.01,
        "oxygen_percent": 0.1,
        "ash_percent": 0.01,
        "moisture_percent": 0.1,
    }


@pytest.fixture
def coal_bituminous_data() -> Dict[str, Any]:
    """Create bituminous coal fuel data."""
    return {
        "fuel_type": "coal_bituminous",
        "heat_input_mmbtu_hr": 200.0,
        "heating_value_btu_lb": 12000.0,
        "flow_rate_tph": 50.0,
        "carbon_percent": 75.0,
        "hydrogen_percent": 5.0,
        "sulfur_percent": 2.0,
        "nitrogen_percent": 1.5,
        "oxygen_percent": 8.0,
        "ash_percent": 8.0,
        "moisture_percent": 5.0,
    }


@pytest.fixture
def biomass_wood_data() -> Dict[str, Any]:
    """Create biomass wood fuel data."""
    return {
        "fuel_type": "biomass_wood",
        "heat_input_mmbtu_hr": 50.0,
        "heating_value_btu_lb": 8500.0,
        "flow_rate_tph": 20.0,
        "carbon_percent": 50.0,
        "hydrogen_percent": 6.0,
        "sulfur_percent": 0.1,
        "nitrogen_percent": 0.3,
        "oxygen_percent": 42.0,
        "ash_percent": 1.0,
        "moisture_percent": 20.0,
        "biogenic": True,
    }


@pytest.fixture
def multi_fuel_data() -> List[Dict[str, Any]]:
    """Create multi-fuel scenario data."""
    return [
        {
            "fuel_type": "natural_gas",
            "heat_input_mmbtu_hr": 70.0,
            "percent_contribution": 70.0,
        },
        {
            "fuel_type": "fuel_oil_no2",
            "heat_input_mmbtu_hr": 30.0,
            "percent_contribution": 30.0,
        },
    ]


# =============================================================================
# PROCESS PARAMETER FIXTURES
# =============================================================================

@pytest.fixture
def boiler_process_parameters() -> Dict[str, Any]:
    """Create boiler process parameters."""
    return {
        "process_type": "boiler",
        "combustion_temperature_f": 2800.0,
        "excess_air_percent": 15.0,
        "firing_rate_percent": 80.0,
        "steam_output_lb_hr": 100000.0,
        "combustion_efficiency_percent": 99.0,
        "fgd_efficiency_percent": 0.0,
        "scr_efficiency_percent": 0.0,
        "baghouse_efficiency_percent": 99.5,
    }


@pytest.fixture
def gas_turbine_process_parameters() -> Dict[str, Any]:
    """Create gas turbine process parameters."""
    return {
        "process_type": "gas_turbine",
        "combustion_temperature_f": 2400.0,
        "excess_air_percent": 200.0,
        "firing_rate_percent": 85.0,
        "power_output_mw": 50.0,
        "heat_rate_btu_kwh": 9000.0,
        "combustion_efficiency_percent": 99.5,
        "scr_efficiency_percent": 90.0,
    }


@pytest.fixture
def incinerator_process_parameters() -> Dict[str, Any]:
    """Create incinerator process parameters."""
    return {
        "process_type": "incinerator",
        "combustion_temperature_f": 1800.0,
        "excess_air_percent": 100.0,
        "residence_time_seconds": 2.0,
        "combustion_efficiency_percent": 99.99,
        "wet_scrubber_efficiency_percent": 85.0,
        "baghouse_efficiency_percent": 99.9,
    }


# =============================================================================
# REGULATORY LIMITS FIXTURES
# =============================================================================

@pytest.fixture
def epa_permit_limits() -> Dict[str, float]:
    """Create EPA permit limits."""
    return {
        "nox_limit": 0.10,  # lb/MMBtu
        "sox_limit": 0.15,  # lb/MMBtu
        "co2_limit": 50.0,  # tons/hr
        "pm_limit": 0.03,  # lb/MMBtu
        "opacity_limit": 20.0,  # percent
        "co_limit": 0.1,  # lb/MMBtu
    }


@pytest.fixture
def eu_ied_permit_limits() -> Dict[str, float]:
    """Create EU IED permit limits (BAT-AELs)."""
    return {
        "nox_limit": 100.0,  # mg/Nm3
        "sox_limit": 150.0,  # mg/Nm3
        "pm_limit": 5.0,  # mg/Nm3
        "co_limit": 100.0,  # mg/Nm3
        "hg_limit": 0.001,  # mg/Nm3
    }


@pytest.fixture
def china_ultra_low_limits() -> Dict[str, float]:
    """Create China ultra-low emission limits."""
    return {
        "nox_limit": 50.0,  # mg/Nm3
        "sox_limit": 35.0,  # mg/Nm3
        "pm_limit": 10.0,  # mg/Nm3
        "hg_limit": 0.03,  # ug/Nm3
    }


@pytest.fixture
def stringent_permit_limits() -> Dict[str, float]:
    """Create stringent permit limits for testing."""
    return {
        "nox_limit": 0.05,  # lb/MMBtu
        "sox_limit": 0.05,  # lb/MMBtu
        "co2_limit": 25.0,  # tons/hr
        "pm_limit": 0.01,  # lb/MMBtu
    }


# =============================================================================
# COMPONENT FIXTURES
# =============================================================================

@pytest.fixture
def emissions_tools(base_config) -> EmissionsComplianceTools:
    """Create EmissionsComplianceTools instance."""
    return EmissionsComplianceTools(base_config)


@pytest.fixture
def orchestrator(base_config) -> EmissionsComplianceOrchestrator:
    """Create EmissionsComplianceOrchestrator instance."""
    return EmissionsComplianceOrchestrator(base_config)


@pytest.fixture
def thread_safe_cache() -> ThreadSafeCache:
    """Create ThreadSafeCache instance."""
    return ThreadSafeCache(max_size=100, ttl_seconds=60)


@pytest.fixture
def performance_metrics() -> PerformanceMetrics:
    """Create PerformanceMetrics instance."""
    return PerformanceMetrics()


@pytest.fixture
def retry_handler() -> RetryHandler:
    """Create RetryHandler instance."""
    return RetryHandler(
        max_retries=3,
        initial_delay_ms=100,
        max_delay_ms=1000,
        exponential_base=2.0,
    )


# =============================================================================
# MOCK FIXTURES
# =============================================================================

@pytest.fixture
def mock_cems_connector():
    """Create mock CEMS connector."""
    mock = MagicMock()
    mock.connect.return_value = True
    mock.disconnect.return_value = True
    mock.is_connected.return_value = True
    mock.read_data.return_value = {
        "nox_ppm": 45.0,
        "sox_ppm": 75.0,
        "co2_percent": 10.5,
        "o2_percent": 3.0,
        "flow_rate_dscfm": 50000.0,
    }
    mock.read_calibration_status.return_value = {
        "last_calibration": datetime.now(timezone.utc).isoformat(),
        "drift_percent": 0.5,
        "in_control": True,
    }
    return mock


@pytest.fixture
def mock_historian_connector():
    """Create mock process historian connector."""
    mock = MagicMock()
    mock.connect.return_value = True
    mock.query_tag.return_value = [
        {"timestamp": datetime.now(timezone.utc).isoformat(), "value": 45.0}
    ]
    mock.query_tags_batch.return_value = {
        "NOX.PV": [45.0, 46.0, 44.0],
        "SOX.PV": [75.0, 74.0, 76.0],
    }
    return mock


@pytest.fixture
def mock_ecmps_api():
    """Create mock EPA ECMPS API connector."""
    mock = AsyncMock()
    mock.submit_report.return_value = {
        "submission_id": "SUB-12345",
        "status": "accepted",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    mock.get_submission_status.return_value = {
        "submission_id": "SUB-12345",
        "status": "processed",
        "errors": [],
    }
    return mock


@pytest.fixture
def mock_alert_service():
    """Create mock alert service."""
    mock = MagicMock()
    mock.send_email.return_value = True
    mock.send_sms.return_value = True
    mock.send_webhook.return_value = True
    return mock


# =============================================================================
# FACILITY DATA FIXTURES
# =============================================================================

@pytest.fixture
def facility_data() -> Dict[str, Any]:
    """Create facility data for reporting."""
    return {
        "facility_id": "ORIS-12345",
        "facility_name": "Test Power Plant",
        "unit_id": "UNIT-001",
        "unit_type": "Utility Boiler",
        "state": "TX",
        "county": "Harris",
        "latitude": 29.7604,
        "longitude": -95.3698,
        "designated_representative": "John Smith",
        "alternate_representative": "Jane Doe",
        "program_codes": ["ARP", "CAIR"],
        "monitoring_plan_id": "MP-2024-001",
    }


@pytest.fixture
def reporting_period() -> Dict[str, str]:
    """Create reporting period data."""
    now = datetime.now(timezone.utc)
    quarter_start = datetime(now.year, ((now.month - 1) // 3) * 3 + 1, 1, tzinfo=timezone.utc)
    quarter_end = quarter_start + timedelta(days=90) - timedelta(seconds=1)

    return {
        "start_date": quarter_start.strftime("%Y-%m-%d"),
        "end_date": quarter_end.strftime("%Y-%m-%d"),
        "quarter": f"Q{(now.month - 1) // 3 + 1}",
        "year": str(now.year),
        "reporting_type": "quarterly",
    }


# =============================================================================
# EMISSIONS RECORDS FIXTURES
# =============================================================================

@pytest.fixture
def emissions_records() -> List[Dict[str, Any]]:
    """Create emissions records for reporting."""
    records = []
    base_time = datetime.now(timezone.utc) - timedelta(days=30)

    for i in range(720):  # 30 days * 24 hours
        timestamp = base_time + timedelta(hours=i)
        records.append({
            "timestamp": timestamp.isoformat(),
            "hour": timestamp.hour,
            "date": timestamp.strftime("%Y-%m-%d"),
            "nox_lb_mmbtu": 0.08 + (i % 5) * 0.005,
            "nox_ppm": 40.0 + (i % 10),
            "sox_lb_mmbtu": 0.10 + (i % 4) * 0.01,
            "sox_ppm": 60.0 + (i % 15),
            "co2_tons": 40.0 + (i % 8),
            "pm_lb_mmbtu": 0.02 + (i % 3) * 0.002,
            "heat_input_mmbtu": 100.0 + (i % 20),
            "o2_percent": 3.0 + (i % 5) * 0.2,
            "flow_rate_dscfm": 50000 + (i % 10) * 500,
            "compliant": True,
            "valid": True,
            "data_quality_code": "valid",
        })

    # Add some non-compliant records
    for i in [100, 250, 500]:
        records[i]["nox_lb_mmbtu"] = 0.15  # Exceeds limit
        records[i]["compliant"] = False
        records[i]["exceedance"] = True

    return records


@pytest.fixture
def compliance_events() -> List[Dict[str, Any]]:
    """Create compliance events for audit trail."""
    return [
        {
            "event_id": "EVT-001",
            "event_type": "exceedance",
            "timestamp": (datetime.now(timezone.utc) - timedelta(days=15)).isoformat(),
            "pollutant": "NOx",
            "duration_minutes": 45,
            "cause": "Startup condition",
            "corrective_action": "Optimized burner settings",
        },
        {
            "event_id": "EVT-002",
            "event_type": "calibration_failure",
            "timestamp": (datetime.now(timezone.utc) - timedelta(days=10)).isoformat(),
            "analyzer": "NOx CEMS",
            "drift_percent": 3.5,
            "corrective_action": "Recalibrated analyzer",
        },
        {
            "event_id": "EVT-003",
            "event_type": "monitoring_downtime",
            "timestamp": (datetime.now(timezone.utc) - timedelta(days=5)).isoformat(),
            "duration_hours": 4,
            "cause": "Scheduled maintenance",
            "substitute_data_used": True,
        },
    ]


# =============================================================================
# STACK AND METEOROLOGICAL FIXTURES
# =============================================================================

@pytest.fixture
def stack_parameters() -> Dict[str, float]:
    """Create stack parameters for dispersion modeling."""
    return {
        "height_m": 75.0,
        "diameter_m": 3.5,
        "exit_velocity_m_s": 20.0,
        "exit_temperature_k": 450.0,
        "exit_temp_f": 350.0,
    }


@pytest.fixture
def meteorological_data() -> Dict[str, Any]:
    """Create meteorological data for dispersion modeling."""
    return {
        "wind_speed_m_s": 5.0,
        "wind_direction_deg": 225.0,
        "stability_class": "D",
        "ambient_temperature_k": 298.0,
        "mixing_height_m": 1000.0,
        "relative_humidity_percent": 65.0,
    }


# =============================================================================
# EXPECTED RESULTS FIXTURES
# =============================================================================

@pytest.fixture
def expected_nox_result() -> Dict[str, Any]:
    """Expected NOx calculation result for verification."""
    return {
        "concentration_ppm": 45.0,
        "emission_rate_lb_mmbtu_min": 0.06,
        "emission_rate_lb_mmbtu_max": 0.12,
        "thermal_nox_percent_min": 60.0,
        "fuel_nox_percent_max": 40.0,
    }


@pytest.fixture
def expected_sox_result() -> Dict[str, Any]:
    """Expected SOx calculation result for verification."""
    return {
        "so2_so3_ratio_min": 95.0,
        "so2_so3_ratio_max": 99.0,
    }


@pytest.fixture
def expected_co2_result() -> Dict[str, Any]:
    """Expected CO2 calculation result for verification."""
    return {
        "emission_rate_lb_mmbtu": 117.0,  # AP-42 natural gas factor
    }


# =============================================================================
# VALIDATION TEST CASES
# =============================================================================

@pytest.fixture
def known_calculation_test_cases() -> List[Dict[str, Any]]:
    """Known calculation test cases with expected results."""
    return [
        {
            "name": "Natural gas standard conditions",
            "fuel_type": "natural_gas",
            "heat_input_mmbtu_hr": 100.0,
            "nox_ppm": 50.0,
            "o2_percent": 3.0,
            "expected_nox_lb_mmbtu": 0.1,
            "tolerance": 0.02,
        },
        {
            "name": "Coal with high sulfur",
            "fuel_type": "coal_bituminous",
            "heat_input_mmbtu_hr": 200.0,
            "sulfur_percent": 2.0,
            "expected_sox_lb_mmbtu": 1.2,
            "tolerance": 0.3,
        },
        {
            "name": "Gas turbine at 15% O2",
            "fuel_type": "natural_gas",
            "heat_input_mmbtu_hr": 150.0,
            "nox_ppm": 25.0,
            "o2_percent": 15.0,
            "expected_nox_corrected_ppm": 25.0,  # At reference
            "tolerance": 5.0,
        },
    ]


@pytest.fixture
def boundary_test_cases() -> List[Dict[str, Any]]:
    """Boundary condition test cases."""
    return [
        {"name": "Zero emissions", "nox_ppm": 0.0, "expected_valid": True},
        {"name": "Maximum analyzer range", "nox_ppm": 500.0, "expected_valid": True},
        {"name": "Above analyzer range", "nox_ppm": 600.0, "expected_valid": False},
        {"name": "Negative value", "nox_ppm": -10.0, "expected_valid": False},
        {"name": "O2 at 0%", "o2_percent": 0.0, "expected_valid": True},
        {"name": "O2 at 21%", "o2_percent": 21.0, "expected_valid": True},
        {"name": "O2 above atmospheric", "o2_percent": 25.0, "expected_valid": False},
    ]


# =============================================================================
# TEMPORARY DIRECTORY FIXTURE
# =============================================================================

@pytest.fixture
def temp_directory():
    """Create temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def temp_data_file(temp_directory) -> Path:
    """Create temporary data file."""
    data_file = temp_directory / "test_data.json"
    test_data = {
        "emissions": [
            {"timestamp": "2024-01-01T00:00:00Z", "nox_ppm": 45.0},
            {"timestamp": "2024-01-01T01:00:00Z", "nox_ppm": 46.0},
        ]
    }
    with open(data_file, "w") as f:
        json.dump(test_data, f)
    return data_file


# =============================================================================
# DETERMINISM VERIFICATION FIXTURE
# =============================================================================

@pytest.fixture
def determinism_test_inputs() -> List[Dict[str, Any]]:
    """Input data for determinism testing."""
    return [
        {
            "cems_data": {
                "nox_ppm": 45.0,
                "o2_percent": 3.0,
                "flow_rate_dscfm": 50000.0,
            },
            "fuel_data": {
                "fuel_type": "natural_gas",
                "heat_input_mmbtu_hr": 100.0,
            },
        },
        {
            "cems_data": {
                "nox_ppm": 80.0,
                "o2_percent": 5.0,
                "flow_rate_dscfm": 60000.0,
            },
            "fuel_data": {
                "fuel_type": "fuel_oil_no2",
                "heat_input_mmbtu_hr": 80.0,
                "sulfur_percent": 0.5,
            },
        },
    ]


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def generate_provenance_hash(data: Dict[str, Any]) -> str:
    """Generate SHA-256 hash for provenance verification."""
    data_str = json.dumps(data, sort_keys=True, default=str)
    return hashlib.sha256(data_str.encode()).hexdigest()


def assert_within_tolerance(actual: float, expected: float, tolerance: float, message: str = ""):
    """Assert value is within tolerance of expected."""
    assert abs(actual - expected) <= tolerance, (
        f"{message} Expected {expected} +/- {tolerance}, got {actual}"
    )


def assert_deterministic(results: List[Any], message: str = ""):
    """Assert all results are identical (deterministic)."""
    if len(results) < 2:
        return
    first = results[0]
    for i, result in enumerate(results[1:], 2):
        assert result == first, (
            f"{message} Result {i} differs from result 1"
        )


# =============================================================================
# ASYNC TEST HELPERS
# =============================================================================

@pytest.fixture
def async_test_helper():
    """Helper for async test setup."""
    class AsyncTestHelper:
        @staticmethod
        async def run_concurrent(coro_func, inputs: List[Any], concurrency: int = 5):
            """Run coroutine concurrently with multiple inputs."""
            semaphore = asyncio.Semaphore(concurrency)

            async def limited_coro(input_data):
                async with semaphore:
                    return await coro_func(input_data)

            tasks = [limited_coro(inp) for inp in inputs]
            return await asyncio.gather(*tasks)

    return AsyncTestHelper()


# =============================================================================
# CLEANUP FIXTURES
# =============================================================================

@pytest.fixture(autouse=True)
def cleanup_after_test():
    """Clean up after each test."""
    yield
    # Cleanup logic if needed


@pytest.fixture(scope="session", autouse=True)
def cleanup_after_session():
    """Clean up after test session."""
    yield
    # Session cleanup logic if needed
