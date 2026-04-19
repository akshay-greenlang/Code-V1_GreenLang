# -*- coding: utf-8 -*-
"""
GL-008 TRAPCATCHER - Test Fixtures

Shared pytest fixtures for steam trap monitoring tests.
Provides reusable test data, mock configurations, and factory functions.
"""

from datetime import datetime, timezone, timedelta
from decimal import Decimal
from typing import Dict, List, Any, Optional
from unittest.mock import Mock, MagicMock, patch
import uuid
import random

import pytest

# Import configuration classes
from greenlang.agents.process_heat.gl_008_steam_trap_monitor.config import (
    SteamTrapMonitorConfig,
    TrapType,
    TrapApplication,
    FailureMode,
    DiagnosticMethod,
    SensorType,
    AlertSeverity,
    TrapTypeConfig,
    TrapTypeDefaults,
    UltrasonicThresholds,
    TemperatureThresholds,
    DiagnosticThresholds,
    SensorConfig,
    WirelessSensorConfig,
    SurveyConfig,
    EconomicsConfig,
)

# Import schema classes
from greenlang.agents.process_heat.gl_008_steam_trap_monitor.schemas import (
    TrapStatus,
    DiagnosisConfidence,
    TrendDirection,
    MaintenancePriority,
    SurveyStatus,
    SensorReading,
    UltrasonicReading,
    TemperatureReading,
    VisualInspectionReading,
    TrapInfo,
    TrapDiagnosticInput,
    TrapDiagnosticOutput,
    TrapCondition,
    TrapHealthScore,
    SteamLossEstimate,
    MaintenanceRecommendation,
    FailureModeProbability,
    CondensateLoadInput,
    CondensateLoadOutput,
    TrapSurveyInput,
    SurveyRouteOutput,
    RouteStop,
    TrapStatusSummary,
    EconomicAnalysisOutput,
)


# =============================================================================
# TEST DATA GENERATORS
# =============================================================================

class TrapTestDataGenerator:
    """Generator for realistic steam trap test data."""

    def __init__(self, seed: int = 42):
        """Initialize with seed for reproducibility."""
        random.seed(seed)
        self._trap_counter = 0
        self._sensor_counter = 0

    def generate_trap_id(self) -> str:
        """Generate unique trap ID."""
        self._trap_counter += 1
        return f"ST-{self._trap_counter:04d}"

    def generate_sensor_id(self) -> str:
        """Generate unique sensor ID."""
        self._sensor_counter += 1
        return f"SENSOR-{self._sensor_counter:04d}"

    def generate_trap_info(
        self,
        trap_type: Optional[TrapType] = None,
        application: Optional[str] = None,
        pressure_rating: float = 150.0,
    ) -> TrapInfo:
        """Generate TrapInfo test data."""
        if trap_type is None:
            trap_type = random.choice(list(TrapType))

        return TrapInfo(
            trap_id=self.generate_trap_id(),
            tag_number=f"TAG-{random.randint(100, 999)}",
            trap_type=trap_type.value,
            manufacturer=random.choice(["Spirax Sarco", "Armstrong", "TLV", "Velan"]),
            model=f"Model-{random.randint(100, 999)}",
            orifice_size_in=random.choice([0.125, 0.1875, 0.25, 0.375]),
            connection_size_in=random.choice([0.5, 0.75, 1.0]),
            pressure_rating_psig=pressure_rating,
            application=application or random.choice(["drip_leg", "process", "tracer"]),
            location=f"Area-{random.choice(['A', 'B', 'C'])}-{random.randint(1, 100)}",
            area_code=f"AREA-{random.choice(['01', '02', '03'])}",
        )

    def generate_ultrasonic_reading(
        self,
        decibel_level: Optional[float] = None,
        cycling_detected: bool = False,
        continuous_flow: bool = False,
        quality_score: float = 0.95,
    ) -> UltrasonicReading:
        """Generate ultrasonic reading test data."""
        if decibel_level is None:
            decibel_level = random.uniform(30, 90)

        return UltrasonicReading(
            sensor_id=self.generate_sensor_id(),
            timestamp=datetime.now(timezone.utc),
            quality_score=quality_score,
            decibel_level_db=decibel_level,
            frequency_khz=38.0,
            cycling_detected=cycling_detected,
            continuous_flow_detected=continuous_flow,
            background_noise_db=random.uniform(20, 40),
        )

    def generate_temperature_reading(
        self,
        inlet_temp: Optional[float] = None,
        outlet_temp: Optional[float] = None,
        quality_score: float = 0.95,
    ) -> TemperatureReading:
        """Generate temperature reading test data."""
        if inlet_temp is None:
            inlet_temp = random.uniform(300, 400)
        if outlet_temp is None:
            outlet_temp = inlet_temp - random.uniform(10, 50)

        return TemperatureReading(
            sensor_id=self.generate_sensor_id(),
            timestamp=datetime.now(timezone.utc),
            quality_score=quality_score,
            inlet_temp_f=inlet_temp,
            outlet_temp_f=outlet_temp,
            ambient_temp_f=70.0,
        )

    def generate_visual_inspection(
        self,
        steam_discharge: bool = False,
        condensate_visible: bool = True,
        cycling_observed: bool = True,
    ) -> VisualInspectionReading:
        """Generate visual inspection test data."""
        return VisualInspectionReading(
            inspector_id=f"INSPECTOR-{random.randint(1, 10):02d}",
            timestamp=datetime.now(timezone.utc),
            visible_steam_discharge=steam_discharge,
            condensate_visible=condensate_visible,
            trap_cycling_observed=cycling_observed,
            trap_body_condition="good",
            insulation_condition="intact",
            leaks_detected=False,
        )

    def generate_diagnostic_input(
        self,
        trap_status: Optional[TrapStatus] = None,
        trap_type: Optional[TrapType] = None,
        steam_pressure: float = 150.0,
        include_ultrasonic: bool = True,
        include_temperature: bool = True,
        include_visual: bool = False,
    ) -> TrapDiagnosticInput:
        """Generate complete diagnostic input test data."""
        trap_info = self.generate_trap_info(trap_type=trap_type)

        # Adjust readings based on expected status
        ultrasonic_readings = []
        temperature_readings = []
        visual = None

        if include_ultrasonic:
            if trap_status == TrapStatus.FAILED_OPEN:
                reading = self.generate_ultrasonic_reading(
                    decibel_level=95.0,
                    continuous_flow=True,
                )
            elif trap_status == TrapStatus.LEAKING:
                reading = self.generate_ultrasonic_reading(
                    decibel_level=78.0,
                )
            elif trap_status == TrapStatus.FAILED_CLOSED:
                reading = self.generate_ultrasonic_reading(
                    decibel_level=35.0,
                )
            else:  # GOOD
                reading = self.generate_ultrasonic_reading(
                    decibel_level=55.0,
                    cycling_detected=True,
                )
            ultrasonic_readings.append(reading)

        if include_temperature:
            if trap_status == TrapStatus.FAILED_OPEN:
                reading = self.generate_temperature_reading(
                    inlet_temp=366.0,
                    outlet_temp=360.0,  # Low delta T
                )
            elif trap_status == TrapStatus.FAILED_CLOSED:
                reading = self.generate_temperature_reading(
                    inlet_temp=366.0,
                    outlet_temp=150.0,  # High delta T
                )
            else:
                reading = self.generate_temperature_reading(
                    inlet_temp=366.0,
                    outlet_temp=340.0,  # Normal delta T
                )
            temperature_readings.append(reading)

        if include_visual:
            if trap_status == TrapStatus.FAILED_OPEN:
                visual = self.generate_visual_inspection(steam_discharge=True)
            else:
                visual = self.generate_visual_inspection()

        return TrapDiagnosticInput(
            trap_info=trap_info,
            ultrasonic_readings=ultrasonic_readings,
            temperature_readings=temperature_readings,
            visual_inspection=visual,
            steam_pressure_psig=steam_pressure,
            back_pressure_psig=0.0,
        )


# =============================================================================
# CONFIGURATION FIXTURES
# =============================================================================

@pytest.fixture
def economics_config() -> EconomicsConfig:
    """Create test economics configuration."""
    return EconomicsConfig(
        steam_cost_per_mlb=12.50,
        steam_pressure_psig=150.0,
        boiler_efficiency_pct=80.0,
        operating_hours_per_year=8760,
        average_repair_cost_usd=350.0,
        average_replacement_cost_usd=750.0,
        labor_rate_per_hour_usd=75.0,
        co2_factor_lb_per_mmbtu=117.0,
        carbon_cost_per_ton=50.0,
        discount_rate_pct=10.0,
        analysis_period_years=5,
    )


@pytest.fixture
def survey_config() -> SurveyConfig:
    """Create test survey configuration."""
    return SurveyConfig(
        survey_interval_months=12,
        high_priority_interval_months=6,
        max_traps_per_route=50,
        average_time_per_trap_minutes=5.0,
        route_optimization_algorithm="nearest_neighbor",
        technicians_available=2,
        max_hours_per_day=8.0,
    )


@pytest.fixture
def wireless_config() -> WirelessSensorConfig:
    """Create test wireless sensor configuration."""
    return WirelessSensorConfig(
        network_id="WSN-TEST-001",
        protocol="LoRaWAN",
        max_sensors=1000,
        default_reporting_interval_s=300,
        high_priority_interval_s=60,
        battery_low_threshold_pct=20.0,
        signal_low_threshold_dbm=-100.0,
        offline_alert_minutes=30,
    )


@pytest.fixture
def diagnostic_thresholds() -> DiagnosticThresholds:
    """Create test diagnostic thresholds."""
    return DiagnosticThresholds(
        ultrasonic=UltrasonicThresholds(
            good_max_db=70.0,
            leaking_min_db=75.0,
            failed_open_db=85.0,
            cold_max_db=40.0,
        ),
        temperature=TemperatureThresholds(
            good_delta_t_min_f=15.0,
            good_delta_t_max_f=50.0,
            failed_open_delta_t_max_f=10.0,
            failed_closed_delta_t_min_f=100.0,
        ),
        high_confidence_threshold=0.90,
        medium_confidence_threshold=0.70,
        require_multi_method_agreement=True,
    )


@pytest.fixture
def steam_trap_config(
    economics_config: EconomicsConfig,
    survey_config: SurveyConfig,
    wireless_config: WirelessSensorConfig,
    diagnostic_thresholds: DiagnosticThresholds,
) -> SteamTrapMonitorConfig:
    """Create complete test configuration."""
    return SteamTrapMonitorConfig(
        plant_id="TEST-PLANT-001",
        plant_name="Test Plant",
        steam_pressure_psig=150.0,
        diagnostics=diagnostic_thresholds,
        economics=economics_config,
        survey=survey_config,
        wireless=wireless_config,
        safety_factor_startup=3.0,
        safety_factor_operating=2.0,
        audit_enabled=True,
        provenance_tracking=True,
    )


# =============================================================================
# TEST DATA FIXTURES
# =============================================================================

@pytest.fixture
def test_data_generator() -> TrapTestDataGenerator:
    """Create test data generator."""
    return TrapTestDataGenerator(seed=42)


@pytest.fixture
def sample_trap_info(test_data_generator: TrapTestDataGenerator) -> TrapInfo:
    """Create sample trap info."""
    return test_data_generator.generate_trap_info(
        trap_type=TrapType.FLOAT_THERMOSTATIC,
        application="drip_leg",
    )


@pytest.fixture
def sample_ultrasonic_reading_good(test_data_generator: TrapTestDataGenerator) -> UltrasonicReading:
    """Create sample good ultrasonic reading."""
    return test_data_generator.generate_ultrasonic_reading(
        decibel_level=55.0,
        cycling_detected=True,
    )


@pytest.fixture
def sample_ultrasonic_reading_failed(test_data_generator: TrapTestDataGenerator) -> UltrasonicReading:
    """Create sample failed open ultrasonic reading."""
    return test_data_generator.generate_ultrasonic_reading(
        decibel_level=95.0,
        continuous_flow=True,
    )


@pytest.fixture
def sample_temperature_reading_good(test_data_generator: TrapTestDataGenerator) -> TemperatureReading:
    """Create sample good temperature reading."""
    return test_data_generator.generate_temperature_reading(
        inlet_temp=366.0,
        outlet_temp=340.0,
    )


@pytest.fixture
def sample_temperature_reading_failed(test_data_generator: TrapTestDataGenerator) -> TemperatureReading:
    """Create sample failed open temperature reading."""
    return test_data_generator.generate_temperature_reading(
        inlet_temp=366.0,
        outlet_temp=360.0,  # Very low delta T
    )


@pytest.fixture
def sample_diagnostic_input_good(test_data_generator: TrapTestDataGenerator) -> TrapDiagnosticInput:
    """Create sample diagnostic input for good trap."""
    return test_data_generator.generate_diagnostic_input(
        trap_status=TrapStatus.GOOD,
        trap_type=TrapType.FLOAT_THERMOSTATIC,
    )


@pytest.fixture
def sample_diagnostic_input_failed(test_data_generator: TrapTestDataGenerator) -> TrapDiagnosticInput:
    """Create sample diagnostic input for failed trap."""
    return test_data_generator.generate_diagnostic_input(
        trap_status=TrapStatus.FAILED_OPEN,
        trap_type=TrapType.INVERTED_BUCKET,
    )


@pytest.fixture
def sample_condensate_load_input() -> CondensateLoadInput:
    """Create sample condensate load input."""
    return CondensateLoadInput(
        application="drip_leg",
        steam_pressure_psig=150.0,
        pipe_diameter_in=4.0,
        pipe_length_ft=100.0,
        pipe_material="carbon_steel",
        ambient_temperature_f=70.0,
        insulation_thickness_in=2.0,
        insulation_type="calcium_silicate",
        calculate_startup=True,
        calculate_operating=True,
        startup_time_minutes=15.0,
    )


@pytest.fixture
def sample_trap_survey_input() -> TrapSurveyInput:
    """Create sample survey input."""
    trap_ids = [f"ST-{i:04d}" for i in range(1, 26)]
    trap_locations = {
        trap_id: (100.0 + i * 10, 200.0 + i * 5)
        for i, trap_id in enumerate(trap_ids)
    }
    trap_areas = {
        trap_id: f"AREA-{(i % 3) + 1:02d}"
        for i, trap_id in enumerate(trap_ids)
    }

    return TrapSurveyInput(
        plant_id="TEST-PLANT-001",
        trap_ids=trap_ids,
        trap_locations=trap_locations,
        trap_areas=trap_areas,
        max_traps_per_route=50,
        available_hours=8.0,
        minutes_per_trap=5.0,
    )


# =============================================================================
# MOCK FIXTURES
# =============================================================================

@pytest.fixture
def mock_provenance_tracker():
    """Create mock provenance tracker."""
    tracker = Mock()
    tracker.calculate_hash.return_value = "a" * 64
    tracker.verify_hash.return_value = True
    return tracker


@pytest.fixture
def mock_wireless_network():
    """Create mock wireless sensor network."""
    network = Mock()
    network.get_sensors_for_trap.return_value = ["SENSOR-001", "SENSOR-002"]
    network.get_network_status.return_value = {
        "total_sensors": 10,
        "online": 9,
        "offline": 1,
        "health_rate": 0.9,
    }
    return network


# =============================================================================
# UTILITY FIXTURES
# =============================================================================

@pytest.fixture
def fixed_timestamp() -> datetime:
    """Create fixed timestamp for deterministic tests."""
    return datetime(2025, 1, 15, 12, 0, 0, tzinfo=timezone.utc)


@pytest.fixture
def known_emission_factors() -> Dict[str, float]:
    """Known emission factors for validation."""
    return {
        ("natural_gas", "stationary_combustion"): 117.0,  # lb CO2/MMBTU
        ("diesel", "stationary_combustion"): 163.0,
        ("coal", "stationary_combustion"): 205.0,
    }


@pytest.fixture
def known_steam_properties() -> Dict[int, Dict[str, float]]:
    """Known steam properties for validation."""
    return {
        0: {"saturation_temp_f": 212.0, "latent_heat_btu_lb": 970.3},
        100: {"saturation_temp_f": 338.0, "latent_heat_btu_lb": 880.6},
        150: {"saturation_temp_f": 366.0, "latent_heat_btu_lb": 857.0},
        200: {"saturation_temp_f": 388.0, "latent_heat_btu_lb": 837.7},
    }


# =============================================================================
# PARAMETRIZED TEST DATA
# =============================================================================

# Trap types for parametrized tests
TRAP_TYPES = [
    TrapType.FLOAT_THERMOSTATIC,
    TrapType.INVERTED_BUCKET,
    TrapType.THERMOSTATIC,
    TrapType.THERMODYNAMIC,
    TrapType.BIMETALLIC,
]

# Failure statuses for parametrized tests
FAILURE_STATUSES = [
    TrapStatus.FAILED_OPEN,
    TrapStatus.FAILED_CLOSED,
    TrapStatus.LEAKING,
]

# Pressure test values (psig)
PRESSURE_TEST_VALUES = [0, 15, 50, 100, 150, 200, 300]

# Ultrasonic dB levels for different conditions
ULTRASONIC_TEST_CASES = [
    (35.0, TrapStatus.FAILED_CLOSED, "Very low - blocked"),
    (55.0, TrapStatus.GOOD, "Normal cycling"),
    (78.0, TrapStatus.LEAKING, "Elevated - leaking"),
    (95.0, TrapStatus.FAILED_OPEN, "High continuous - blow-through"),
]

# Temperature differential test cases (inlet, outlet, expected_status)
TEMPERATURE_TEST_CASES = [
    (366.0, 340.0, TrapStatus.GOOD, "Normal subcooling"),
    (366.0, 362.0, TrapStatus.FAILED_OPEN, "Very low delta T"),
    (366.0, 150.0, TrapStatus.FAILED_CLOSED, "Very high delta T"),
]
