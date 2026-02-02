# -*- coding: utf-8 -*-
"""
GL-005 Test Fixtures and Configuration
======================================

Shared pytest fixtures for GL-005 test suite.
Provides reusable test data, mock objects, and configuration.

Author: GreenLang Process Heat Team
Version: 1.0.0
"""

import pytest
from datetime import datetime, timezone, timedelta
from typing import List
from unittest.mock import Mock, MagicMock
import random

# Import GL-005 modules
from greenlang.agents.process_heat.gl_005_combustion_diagnostics.config import (
    GL005Config,
    CQIConfig,
    CQIWeights,
    CQIThresholds,
    AnomalyDetectionConfig,
    SPCConfig,
    MLAnomalyConfig,
    FuelCharacterizationConfig,
    MaintenanceAdvisoryConfig,
    FoulingPredictionConfig,
    BurnerWearConfig,
    TrendingConfig,
    ComplianceConfig,
    DiagnosticMode,
    FuelCategory,
    ComplianceFramework,
    MaintenancePriority,
    AnomalyType,
)
from greenlang.agents.process_heat.gl_005_combustion_diagnostics.schemas import (
    FlueGasReading,
    CombustionOperatingData,
    DiagnosticsInput,
    CQIRating,
    AnomalySeverity,
    TrendDirection,
    AnalysisStatus,
)


# =============================================================================
# CONFIGURATION FIXTURES
# =============================================================================

@pytest.fixture
def default_cqi_weights():
    """Default CQI weights configuration."""
    return CQIWeights()


@pytest.fixture
def default_cqi_thresholds():
    """Default CQI thresholds configuration."""
    return CQIThresholds()


@pytest.fixture
def default_cqi_config():
    """Default CQI configuration."""
    return CQIConfig()


@pytest.fixture
def default_spc_config():
    """Default SPC configuration."""
    return SPCConfig()


@pytest.fixture
def default_ml_config():
    """Default ML anomaly detection configuration."""
    return MLAnomalyConfig()


@pytest.fixture
def default_anomaly_config():
    """Default anomaly detection configuration."""
    return AnomalyDetectionConfig()


@pytest.fixture
def default_fuel_config():
    """Default fuel characterization configuration."""
    return FuelCharacterizationConfig()


@pytest.fixture
def default_fouling_config():
    """Default fouling prediction configuration."""
    return FoulingPredictionConfig()


@pytest.fixture
def default_burner_wear_config():
    """Default burner wear configuration."""
    return BurnerWearConfig()


@pytest.fixture
def default_maintenance_config():
    """Default maintenance advisory configuration."""
    return MaintenanceAdvisoryConfig()


@pytest.fixture
def default_trending_config():
    """Default trending configuration."""
    return TrendingConfig()


@pytest.fixture
def default_compliance_config():
    """Default compliance configuration."""
    return ComplianceConfig()


@pytest.fixture
def gl005_config():
    """Complete GL-005 agent configuration."""
    return GL005Config(
        agent_id="GL005-TEST-001",
        equipment_id="BLR-TEST-001",
        equipment_type="boiler",
        equipment_name="Test Boiler",
        mode=DiagnosticMode.REAL_TIME,
        primary_fuel=FuelCategory.NATURAL_GAS,
    )


@pytest.fixture
def high_precision_config():
    """High-precision GL-005 configuration."""
    config = GL005Config(
        agent_id="GL005-HP-001",
        equipment_id="BLR-HP-001",
        equipment_type="boiler",
        mode=DiagnosticMode.REAL_TIME,
        primary_fuel=FuelCategory.NATURAL_GAS,
    )
    config.cqi.thresholds.co_excellent = 25.0
    config.cqi.thresholds.co_good = 50.0
    config.cqi.calculation_interval_s = 30.0
    config.anomaly_detection.spc.sigma_warning = 1.5
    config.anomaly_detection.spc.sigma_control = 2.5
    config.data_poll_interval_s = 2.0
    return config


# =============================================================================
# FLUE GAS READING FIXTURES
# =============================================================================

@pytest.fixture
def optimal_flue_gas_reading():
    """Optimal combustion flue gas reading."""
    return FlueGasReading(
        timestamp=datetime.now(timezone.utc),
        oxygen_pct=3.0,
        co2_pct=10.5,
        co_ppm=25.0,
        nox_ppm=40.0,
        so2_ppm=5.0,
        combustibles_pct=0.05,
        moisture_pct=8.0,
        flue_gas_temp_c=180.0,
        ambient_temp_c=25.0,
        barometric_pressure_kpa=101.325,
        sensor_status="ok",
        data_quality_flag="good",
    )


@pytest.fixture
def high_co_flue_gas_reading():
    """High CO flue gas reading indicating incomplete combustion."""
    return FlueGasReading(
        timestamp=datetime.now(timezone.utc),
        oxygen_pct=2.5,
        co2_pct=11.0,
        co_ppm=500.0,
        nox_ppm=60.0,
        so2_ppm=10.0,
        combustibles_pct=0.3,
        moisture_pct=9.0,
        flue_gas_temp_c=200.0,
        ambient_temp_c=25.0,
        barometric_pressure_kpa=101.325,
        sensor_status="ok",
        data_quality_flag="good",
    )


@pytest.fixture
def high_o2_flue_gas_reading():
    """High O2 flue gas reading indicating excess air."""
    return FlueGasReading(
        timestamp=datetime.now(timezone.utc),
        oxygen_pct=9.0,
        co2_pct=7.5,
        co_ppm=10.0,
        nox_ppm=80.0,
        so2_ppm=5.0,
        combustibles_pct=0.02,
        moisture_pct=6.0,
        flue_gas_temp_c=220.0,
        ambient_temp_c=25.0,
        barometric_pressure_kpa=101.325,
        sensor_status="ok",
        data_quality_flag="good",
    )


@pytest.fixture
def low_o2_flue_gas_reading():
    """Low O2 flue gas reading indicating incomplete combustion risk."""
    return FlueGasReading(
        timestamp=datetime.now(timezone.utc),
        oxygen_pct=1.0,
        co2_pct=12.0,
        co_ppm=800.0,
        nox_ppm=30.0,
        so2_ppm=15.0,
        combustibles_pct=0.8,
        moisture_pct=10.0,
        flue_gas_temp_c=190.0,
        ambient_temp_c=25.0,
        barometric_pressure_kpa=101.325,
        sensor_status="ok",
        data_quality_flag="good",
    )


@pytest.fixture
def high_nox_flue_gas_reading():
    """High NOx flue gas reading."""
    return FlueGasReading(
        timestamp=datetime.now(timezone.utc),
        oxygen_pct=4.0,
        co2_pct=9.5,
        co_ppm=50.0,
        nox_ppm=300.0,
        so2_ppm=5.0,
        combustibles_pct=0.05,
        moisture_pct=8.0,
        flue_gas_temp_c=210.0,
        ambient_temp_c=25.0,
        barometric_pressure_kpa=101.325,
        sensor_status="ok",
        data_quality_flag="good",
    )


@pytest.fixture
def fouling_flue_gas_reading():
    """Flue gas reading indicating fouling (high stack temp)."""
    return FlueGasReading(
        timestamp=datetime.now(timezone.utc),
        oxygen_pct=3.5,
        co2_pct=10.0,
        co_ppm=60.0,
        nox_ppm=55.0,
        so2_ppm=5.0,
        combustibles_pct=0.1,
        moisture_pct=8.0,
        flue_gas_temp_c=320.0,  # High stack temp indicates fouling
        ambient_temp_c=25.0,
        barometric_pressure_kpa=101.325,
        sensor_status="ok",
        data_quality_flag="good",
    )


@pytest.fixture
def bad_sensor_flue_gas_reading():
    """Flue gas reading with bad sensor status."""
    return FlueGasReading(
        timestamp=datetime.now(timezone.utc),
        oxygen_pct=3.0,
        co2_pct=10.5,
        co_ppm=30.0,
        nox_ppm=45.0,
        flue_gas_temp_c=180.0,
        sensor_status="fault",
        data_quality_flag="bad",
    )


# =============================================================================
# OPERATING DATA FIXTURES
# =============================================================================

@pytest.fixture
def normal_operating_data():
    """Normal operating data."""
    return CombustionOperatingData(
        timestamp=datetime.now(timezone.utc),
        firing_rate_pct=75.0,
        steam_flow_kg_h=5000.0,
        heat_output_mw=3.5,
        fuel_flow_rate=350.0,
        fuel_flow_unit="m3/h",
        fuel_type=FuelCategory.NATURAL_GAS,
        fuel_hhv_mj_kg=55.5,
        combustion_air_flow_m3_h=6000.0,
        air_temp_c=30.0,
        air_humidity_pct=60.0,
        burner_status="modulating",
        damper_position_pct=65.0,
        control_mode="auto",
        operating_hours_total=15000.0,
        operating_hours_since_maintenance=500.0,
    )


@pytest.fixture
def high_firing_rate_operating_data():
    """High firing rate operating data."""
    return CombustionOperatingData(
        timestamp=datetime.now(timezone.utc),
        firing_rate_pct=98.0,
        steam_flow_kg_h=7000.0,
        heat_output_mw=4.8,
        fuel_flow_rate=480.0,
        fuel_flow_unit="m3/h",
        fuel_type=FuelCategory.NATURAL_GAS,
        combustion_air_flow_m3_h=8500.0,
        burner_status="high_fire",
        damper_position_pct=95.0,
        control_mode="auto",
        operating_hours_total=18000.0,
        operating_hours_since_maintenance=2000.0,
    )


@pytest.fixture
def worn_burner_operating_data():
    """Operating data for worn burner scenario."""
    return CombustionOperatingData(
        timestamp=datetime.now(timezone.utc),
        firing_rate_pct=70.0,
        steam_flow_kg_h=4500.0,
        heat_output_mw=3.2,
        fuel_flow_rate=320.0,
        fuel_flow_unit="m3/h",
        fuel_type=FuelCategory.NATURAL_GAS,
        combustion_air_flow_m3_h=5500.0,
        burner_status="modulating",
        damper_position_pct=60.0,
        control_mode="auto",
        operating_hours_total=19500.0,  # Near end of burner life
        operating_hours_since_maintenance=3500.0,
    )


# =============================================================================
# DIAGNOSTICS INPUT FIXTURES
# =============================================================================

@pytest.fixture
def diagnostics_input(optimal_flue_gas_reading, normal_operating_data):
    """Complete diagnostics input."""
    return DiagnosticsInput(
        equipment_id="BLR-TEST-001",
        request_id="REQ-TEST-001",
        flue_gas=optimal_flue_gas_reading,
        operating_data=normal_operating_data,
        run_cqi_analysis=True,
        run_anomaly_detection=True,
        run_fuel_characterization=True,
        run_maintenance_prediction=True,
    )


@pytest.fixture
def diagnostics_input_anomalous(high_co_flue_gas_reading, normal_operating_data):
    """Diagnostics input with anomalous flue gas."""
    return DiagnosticsInput(
        equipment_id="BLR-TEST-001",
        request_id="REQ-TEST-002",
        flue_gas=high_co_flue_gas_reading,
        operating_data=normal_operating_data,
        run_cqi_analysis=True,
        run_anomaly_detection=True,
        run_fuel_characterization=True,
        run_maintenance_prediction=True,
    )


# =============================================================================
# HISTORICAL DATA FIXTURES
# =============================================================================

def generate_historical_readings(
    num_readings: int = 100,
    base_o2: float = 3.0,
    base_co: float = 30.0,
    base_nox: float = 45.0,
    noise_level: float = 0.1,
    seed: int = 42,
) -> List[FlueGasReading]:
    """Generate historical flue gas readings for testing."""
    random.seed(seed)
    readings = []
    base_time = datetime.now(timezone.utc) - timedelta(hours=num_readings)

    for i in range(num_readings):
        timestamp = base_time + timedelta(hours=i)
        # Add random noise
        o2 = base_o2 + random.gauss(0, base_o2 * noise_level)
        co = base_co + random.gauss(0, base_co * noise_level)
        nox = base_nox + random.gauss(0, base_nox * noise_level)
        co2 = 11.8 * (20.95 - o2) / 20.95  # Calculated from O2

        # Clamp values to valid ranges
        o2 = max(0.5, min(20.0, o2))
        co = max(0, co)
        nox = max(0, nox)
        co2 = max(0, min(20.0, co2))

        reading = FlueGasReading(
            timestamp=timestamp,
            oxygen_pct=o2,
            co2_pct=co2,
            co_ppm=co,
            nox_ppm=nox,
            flue_gas_temp_c=180.0 + random.gauss(0, 5),
            sensor_status="ok",
            data_quality_flag="good",
        )
        readings.append(reading)

    return readings


@pytest.fixture
def historical_readings_normal() -> List[FlueGasReading]:
    """Normal historical readings for baseline establishment."""
    return generate_historical_readings(
        num_readings=100,
        base_o2=3.0,
        base_co=30.0,
        base_nox=45.0,
        noise_level=0.1,
    )


@pytest.fixture
def historical_readings_trending_up() -> List[FlueGasReading]:
    """Historical readings with upward CO trend."""
    readings = []
    base_time = datetime.now(timezone.utc) - timedelta(hours=100)

    for i in range(100):
        timestamp = base_time + timedelta(hours=i)
        co = 30.0 + (i * 2)  # CO increasing over time
        o2 = 3.0 + random.gauss(0, 0.3)
        co2 = 11.8 * (20.95 - o2) / 20.95

        reading = FlueGasReading(
            timestamp=timestamp,
            oxygen_pct=max(0.5, o2),
            co2_pct=max(0, co2),
            co_ppm=max(0, co),
            nox_ppm=45.0 + random.gauss(0, 5),
            flue_gas_temp_c=180.0 + random.gauss(0, 5),
            sensor_status="ok",
            data_quality_flag="good",
        )
        readings.append(reading)

    return readings


@pytest.fixture
def historical_readings_anomalous() -> List[FlueGasReading]:
    """Historical readings with anomalies."""
    readings = generate_historical_readings(num_readings=100)

    # Inject anomalies at specific points
    anomaly_indices = [25, 50, 75]
    for idx in anomaly_indices:
        readings[idx] = FlueGasReading(
            timestamp=readings[idx].timestamp,
            oxygen_pct=8.0,  # Anomalous high O2
            co2_pct=7.0,
            co_ppm=500.0,  # Anomalous high CO
            nox_ppm=250.0,  # Anomalous high NOx
            flue_gas_temp_c=280.0,
            sensor_status="ok",
            data_quality_flag="good",
        )

    return readings


# =============================================================================
# MOCK FIXTURES
# =============================================================================

@pytest.fixture
def mock_cmms_api():
    """Mock CMMS API for work order testing."""
    mock = MagicMock()
    mock.create_work_order.return_value = {"status": "created", "wo_id": "WO-12345"}
    mock.get_work_order.return_value = {"status": "pending", "wo_id": "WO-12345"}
    return mock


@pytest.fixture
def mock_data_source():
    """Mock data source (GL-018) for integration testing."""
    mock = MagicMock()
    mock.get_latest_reading.return_value = FlueGasReading(
        timestamp=datetime.now(timezone.utc),
        oxygen_pct=3.0,
        co2_pct=10.5,
        co_ppm=30.0,
        nox_ppm=45.0,
        flue_gas_temp_c=180.0,
        sensor_status="ok",
        data_quality_flag="good",
    )
    return mock


# =============================================================================
# PARAMETRIZED TEST DATA
# =============================================================================

# CQI test scenarios: (o2_pct, co_ppm, nox_ppm, expected_rating)
CQI_TEST_SCENARIOS = [
    (3.0, 25.0, 40.0, CQIRating.EXCELLENT),    # Optimal combustion
    (3.0, 80.0, 80.0, CQIRating.GOOD),          # Good combustion
    (5.0, 150.0, 120.0, CQIRating.ACCEPTABLE),  # Acceptable
    (7.0, 350.0, 200.0, CQIRating.POOR),        # Poor combustion
    (9.0, 600.0, 300.0, CQIRating.CRITICAL),    # Critical
]

# Fuel identification scenarios: (co2_pct, o2_pct, expected_fuel)
FUEL_TEST_SCENARIOS = [
    (10.5, 3.0, FuelCategory.NATURAL_GAS),
    (13.0, 3.0, FuelCategory.PROPANE),
    (14.5, 3.0, FuelCategory.FUEL_OIL_2),
]

# Anomaly detection scenarios: (o2_pct, co_ppm, expected_anomaly_type)
ANOMALY_TEST_SCENARIOS = [
    (1.0, 800.0, AnomalyType.LOW_OXYGEN),
    (9.0, 50.0, AnomalyType.EXCESS_OXYGEN),
    (3.0, 500.0, AnomalyType.HIGH_CO),
]


@pytest.fixture(params=CQI_TEST_SCENARIOS)
def cqi_scenario(request):
    """Parametrized CQI test scenarios."""
    o2, co, nox, expected_rating = request.param
    reading = FlueGasReading(
        timestamp=datetime.now(timezone.utc),
        oxygen_pct=o2,
        co2_pct=11.8 * (20.95 - o2) / 20.95,
        co_ppm=co,
        nox_ppm=nox,
        flue_gas_temp_c=180.0,
        sensor_status="ok",
        data_quality_flag="good",
    )
    return reading, expected_rating


@pytest.fixture(params=FUEL_TEST_SCENARIOS)
def fuel_scenario(request):
    """Parametrized fuel identification scenarios."""
    co2, o2, expected_fuel = request.param
    reading = FlueGasReading(
        timestamp=datetime.now(timezone.utc),
        oxygen_pct=o2,
        co2_pct=co2,
        co_ppm=30.0,
        nox_ppm=45.0,
        flue_gas_temp_c=180.0,
        sensor_status="ok",
        data_quality_flag="good",
    )
    return reading, expected_fuel


# =============================================================================
# UTILITY FUNCTIONS FOR TESTS
# =============================================================================

def assert_valid_provenance_hash(hash_value: str):
    """Assert that a provenance hash is valid SHA-256."""
    assert hash_value is not None
    assert len(hash_value) == 64
    assert all(c in '0123456789abcdef' for c in hash_value)


def assert_valid_cqi_score(score: float):
    """Assert that a CQI score is valid."""
    assert 0.0 <= score <= 100.0


def assert_valid_confidence(confidence: float):
    """Assert that a confidence value is valid."""
    assert 0.0 <= confidence <= 1.0
