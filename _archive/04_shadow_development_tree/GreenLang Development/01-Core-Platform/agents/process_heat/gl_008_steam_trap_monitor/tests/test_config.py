# -*- coding: utf-8 -*-
"""
GL-008 TRAPCATCHER - Configuration Module Tests

Unit tests for config.py module including all configuration schemas,
trap type definitions, and threshold configurations.

Target Coverage: 85%+
"""

import pytest
from datetime import datetime
from typing import Dict

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


class TestTrapTypeEnum:
    """Tests for TrapType enumeration."""

    def test_all_trap_types_defined(self):
        """Verify all expected trap types are defined."""
        expected_types = [
            "float_thermostatic",
            "inverted_bucket",
            "thermostatic",
            "thermodynamic",
            "bimetallic",
            "liquid_expansion",
            "orifice",
            "float",
        ]
        actual_types = [t.value for t in TrapType]

        for expected in expected_types:
            assert expected in actual_types, f"Missing trap type: {expected}"

    def test_trap_type_string_values(self):
        """Verify trap types are string enums."""
        assert TrapType.FLOAT_THERMOSTATIC.value == "float_thermostatic"
        assert TrapType.INVERTED_BUCKET.value == "inverted_bucket"
        assert TrapType.THERMODYNAMIC.value == "thermodynamic"

    def test_trap_type_from_string(self):
        """Test creating TrapType from string."""
        trap = TrapType("float_thermostatic")
        assert trap == TrapType.FLOAT_THERMOSTATIC


class TestTrapApplicationEnum:
    """Tests for TrapApplication enumeration."""

    def test_all_applications_defined(self):
        """Verify all expected applications are defined."""
        expected = [
            "drip_leg",
            "process",
            "tracer",
            "unit_heater",
            "heat_exchanger",
            "coil",
            "jacketed_vessel",
            "reboiler",
            "autoclave",
            "sterilizer",
        ]
        actual = [a.value for a in TrapApplication]

        for app in expected:
            assert app in actual, f"Missing application: {app}"


class TestFailureModeEnum:
    """Tests for FailureMode enumeration."""

    def test_failure_modes_defined(self):
        """Verify all failure modes are defined."""
        expected = ["good", "failed_open", "failed_closed", "leaking", "cold", "flooded"]
        actual = [f.value for f in FailureMode]

        for mode in expected:
            assert mode in actual, f"Missing failure mode: {mode}"


class TestUltrasonicThresholds:
    """Tests for UltrasonicThresholds configuration."""

    def test_default_values(self):
        """Test default ultrasonic threshold values."""
        thresholds = UltrasonicThresholds()

        assert thresholds.good_max_db == 70.0
        assert thresholds.leaking_min_db == 75.0
        assert thresholds.failed_open_db == 85.0
        assert thresholds.cold_max_db == 40.0

    def test_custom_values(self):
        """Test custom ultrasonic threshold values."""
        thresholds = UltrasonicThresholds(
            good_max_db=65.0,
            leaking_min_db=72.0,
            failed_open_db=82.0,
        )

        assert thresholds.good_max_db == 65.0
        assert thresholds.leaking_min_db == 72.0
        assert thresholds.failed_open_db == 82.0

    def test_cycling_parameters(self):
        """Test cycling parameters have valid defaults."""
        thresholds = UltrasonicThresholds()

        assert thresholds.min_cycle_period_s > 0
        assert thresholds.max_cycle_period_s > thresholds.min_cycle_period_s
        assert thresholds.continuous_flow_duration_s > 0

    def test_trap_type_adjustments(self):
        """Test trap type specific adjustments."""
        thresholds = UltrasonicThresholds()

        # Thermodynamic traps have higher threshold
        assert thresholds.thermodynamic_good_max_db > thresholds.good_max_db
        assert thresholds.inverted_bucket_cycling_expected is True

    def test_validation_positive_values(self):
        """Test validation ensures positive values."""
        with pytest.raises(ValueError):
            UltrasonicThresholds(good_max_db=-10.0)


class TestTemperatureThresholds:
    """Tests for TemperatureThresholds configuration."""

    def test_default_values(self):
        """Test default temperature threshold values."""
        thresholds = TemperatureThresholds()

        assert thresholds.good_delta_t_min_f == 15.0
        assert thresholds.good_delta_t_max_f == 50.0
        assert thresholds.failed_open_delta_t_max_f == 10.0
        assert thresholds.failed_closed_delta_t_min_f == 100.0

    def test_outlet_temperature_limits(self):
        """Test outlet temperature limit defaults."""
        thresholds = TemperatureThresholds()

        assert thresholds.outlet_max_above_sat_f >= 0
        assert thresholds.outlet_max_below_sat_f > 0

    def test_ambient_threshold(self):
        """Test ambient delta threshold."""
        thresholds = TemperatureThresholds()

        assert thresholds.ambient_delta_threshold_f > 0
        assert thresholds.min_inlet_temp_f > 100  # Steam must be hot


class TestDiagnosticThresholds:
    """Tests for combined DiagnosticThresholds configuration."""

    def test_default_creation(self):
        """Test creating with defaults."""
        thresholds = DiagnosticThresholds()

        assert thresholds.ultrasonic is not None
        assert thresholds.temperature is not None

    def test_confidence_thresholds(self):
        """Test confidence threshold values."""
        thresholds = DiagnosticThresholds()

        assert 0.5 <= thresholds.high_confidence_threshold <= 1.0
        assert 0.3 <= thresholds.medium_confidence_threshold <= 1.0
        assert thresholds.high_confidence_threshold > thresholds.medium_confidence_threshold

    def test_multi_method_agreement(self):
        """Test multi-method agreement settings."""
        thresholds = DiagnosticThresholds()

        assert isinstance(thresholds.require_multi_method_agreement, bool)
        assert 0 <= thresholds.disagreement_confidence_penalty <= 0.5


class TestTrapTypeConfig:
    """Tests for TrapTypeConfig."""

    def test_create_config(self):
        """Test creating trap type config."""
        config = TrapTypeConfig(
            trap_type=TrapType.FLOAT_THERMOSTATIC,
            description="Test trap",
            max_pressure_psig=465,
            max_capacity_lb_hr=30000,
        )

        assert config.trap_type == TrapType.FLOAT_THERMOSTATIC
        assert config.max_pressure_psig == 465

    def test_default_values(self):
        """Test default values are set."""
        config = TrapTypeConfig(trap_type=TrapType.THERMOSTATIC)

        assert config.max_pressure_psig > 0
        assert config.typical_service_life_years > 0
        assert config.maintenance_interval_months > 0


class TestTrapTypeDefaults:
    """Tests for TrapTypeDefaults static configurations."""

    def test_float_thermostatic_defaults(self):
        """Test float thermostatic defaults."""
        config = TrapTypeDefaults.FLOAT_THERMOSTATIC

        assert config.trap_type == TrapType.FLOAT_THERMOSTATIC
        assert config.subcooling_f == 0  # Continuous discharge
        assert config.air_venting_capability == "excellent"
        assert config.waterhammer_susceptible is True

    def test_inverted_bucket_defaults(self):
        """Test inverted bucket defaults."""
        config = TrapTypeDefaults.INVERTED_BUCKET

        assert config.trap_type == TrapType.INVERTED_BUCKET
        assert config.typical_service_life_years == 15  # Longest life
        assert config.dirt_tolerant is True
        assert config.ultrasonic_adjustment_db > 0  # Cycles loudly

    def test_thermostatic_defaults(self):
        """Test balanced pressure thermostatic defaults."""
        config = TrapTypeDefaults.THERMOSTATIC

        assert config.trap_type == TrapType.THERMOSTATIC
        assert config.subcooling_f > 0  # Operates with subcooling
        assert config.air_venting_capability == "excellent"

    def test_thermodynamic_defaults(self):
        """Test thermodynamic disc defaults."""
        config = TrapTypeDefaults.THERMODYNAMIC

        assert config.trap_type == TrapType.THERMODYNAMIC
        assert config.air_venting_capability == "fair"  # Poor air venting
        assert config.ultrasonic_adjustment_db > 5  # Cycles very loudly

    def test_bimetallic_defaults(self):
        """Test bimetallic defaults."""
        config = TrapTypeDefaults.BIMETALLIC

        assert config.trap_type == TrapType.BIMETALLIC
        assert config.subcooling_f >= 30  # High subcooling
        assert config.predominant_failure_mode == FailureMode.FAILED_CLOSED


class TestSensorConfig:
    """Tests for SensorConfig."""

    def test_create_ultrasonic_sensor(self):
        """Test creating ultrasonic sensor config."""
        config = SensorConfig(
            sensor_id="SENSOR-001",
            sensor_type=SensorType.ULTRASONIC_WIRELESS,
            frequency_khz=38.0,
        )

        assert config.sensor_id == "SENSOR-001"
        assert config.sensor_type == SensorType.ULTRASONIC_WIRELESS
        assert config.frequency_khz == 38.0

    def test_wireless_sensor_properties(self):
        """Test wireless sensor properties."""
        config = SensorConfig(
            sensor_id="SENSOR-002",
            sensor_type=SensorType.TEMPERATURE_WIRELESS,
            battery_level_pct=85.0,
            signal_strength_dbm=-75.0,
        )

        assert config.battery_level_pct == 85.0
        assert config.signal_strength_dbm == -75.0

    def test_calibration_dates(self):
        """Test calibration date handling."""
        now = datetime.now()
        config = SensorConfig(
            sensor_id="SENSOR-003",
            sensor_type=SensorType.ULTRASONIC_HANDHELD,
            calibration_date=now,
        )

        assert config.calibration_date == now


class TestWirelessSensorConfig:
    """Tests for WirelessSensorConfig."""

    def test_default_creation(self):
        """Test creating with defaults."""
        config = WirelessSensorConfig()

        assert config.network_id == "WSN-001"
        assert config.protocol == "LoRaWAN"

    def test_custom_protocol(self):
        """Test custom protocol settings."""
        config = WirelessSensorConfig(
            network_id="CUSTOM-NET",
            protocol="WirelessHART",
            gateway_ip="192.168.1.100",
        )

        assert config.protocol == "WirelessHART"
        assert config.gateway_ip == "192.168.1.100"

    def test_alert_thresholds(self):
        """Test alert threshold settings."""
        config = WirelessSensorConfig(
            battery_low_threshold_pct=25.0,
            signal_low_threshold_dbm=-95.0,
            offline_alert_minutes=15,
        )

        assert config.battery_low_threshold_pct == 25.0
        assert config.offline_alert_minutes == 15


class TestSurveyConfig:
    """Tests for SurveyConfig."""

    def test_default_intervals(self):
        """Test default survey intervals."""
        config = SurveyConfig()

        assert config.survey_interval_months == 12  # DOE annual recommendation
        assert config.high_priority_interval_months < config.survey_interval_months

    def test_route_constraints(self):
        """Test route constraints."""
        config = SurveyConfig(
            max_traps_per_route=60,
            average_time_per_trap_minutes=4.0,
        )

        assert config.max_traps_per_route == 60
        assert config.average_time_per_trap_minutes == 4.0

    def test_optimization_algorithm(self):
        """Test optimization algorithm setting."""
        config = SurveyConfig(route_optimization_algorithm="2opt")

        assert config.route_optimization_algorithm == "2opt"

    def test_quality_settings(self):
        """Test quality assurance settings."""
        config = SurveyConfig()

        assert config.random_verification_pct > 0
        assert config.supervisor_review_required is True


class TestEconomicsConfig:
    """Tests for EconomicsConfig."""

    def test_default_steam_costs(self):
        """Test default steam cost values."""
        config = EconomicsConfig()

        assert config.steam_cost_per_mlb > 0
        assert config.boiler_efficiency_pct > 0

    def test_operating_hours(self):
        """Test operating hours constraints."""
        config = EconomicsConfig(operating_hours_per_year=8000)

        assert config.operating_hours_per_year == 8000
        assert config.operating_hours_per_year <= 8760

    def test_repair_costs(self):
        """Test repair cost settings."""
        config = EconomicsConfig(
            average_repair_cost_usd=400.0,
            average_replacement_cost_usd=800.0,
        )

        assert config.average_repair_cost_usd == 400.0
        assert config.average_replacement_cost_usd > config.average_repair_cost_usd

    def test_environmental_factors(self):
        """Test environmental factor settings."""
        config = EconomicsConfig()

        assert config.co2_factor_lb_per_mmbtu > 0
        assert config.carbon_cost_per_ton >= 0

    def test_financial_parameters(self):
        """Test financial analysis parameters."""
        config = EconomicsConfig(
            discount_rate_pct=8.0,
            analysis_period_years=7,
        )

        assert config.discount_rate_pct == 8.0
        assert config.analysis_period_years == 7


class TestSteamTrapMonitorConfig:
    """Tests for main SteamTrapMonitorConfig."""

    def test_minimal_creation(self):
        """Test creating with minimal required fields."""
        config = SteamTrapMonitorConfig(plant_id="TEST-001")

        assert config.plant_id == "TEST-001"
        assert config.steam_pressure_psig == 150.0  # Default

    def test_full_creation(self, steam_trap_config: SteamTrapMonitorConfig):
        """Test full configuration creation."""
        assert steam_trap_config.plant_id == "TEST-PLANT-001"
        assert steam_trap_config.diagnostics is not None
        assert steam_trap_config.economics is not None
        assert steam_trap_config.survey is not None

    def test_trap_type_configurations(self, steam_trap_config: SteamTrapMonitorConfig):
        """Test trap type configurations are populated."""
        trap_types = steam_trap_config.trap_types

        assert "float_thermostatic" in trap_types
        assert "inverted_bucket" in trap_types
        assert "thermodynamic" in trap_types

    def test_get_trap_type_config(self, steam_trap_config: SteamTrapMonitorConfig):
        """Test getting trap type config by name."""
        config = steam_trap_config.get_trap_type_config("float_thermostatic")

        assert config is not None
        assert config.trap_type == TrapType.FLOAT_THERMOSTATIC

    def test_get_trap_type_config_case_insensitive(self, steam_trap_config: SteamTrapMonitorConfig):
        """Test trap type lookup is case insensitive."""
        config1 = steam_trap_config.get_trap_type_config("FLOAT_THERMOSTATIC")
        config2 = steam_trap_config.get_trap_type_config("Float_Thermostatic")

        # Should normalize to lowercase
        assert config1 is not None or config2 is not None

    def test_get_saturation_temperature(self, steam_trap_config: SteamTrapMonitorConfig):
        """Test saturation temperature calculation."""
        # Known values from steam tables
        test_cases = [
            (0, 212.0),    # Atmospheric
            (100, 338.0),  # 100 psig
            (150, 366.0),  # 150 psig
        ]

        for pressure, expected_temp in test_cases:
            temp = steam_trap_config.get_saturation_temperature(pressure)
            assert abs(temp - expected_temp) < 5, f"At {pressure} psig: got {temp}, expected {expected_temp}"

    def test_get_latent_heat(self, steam_trap_config: SteamTrapMonitorConfig):
        """Test latent heat calculation."""
        # Known values from steam tables
        test_cases = [
            (0, 970.3),    # Atmospheric
            (100, 880.6),  # 100 psig
            (150, 857.0),  # 150 psig
        ]

        for pressure, expected_heat in test_cases:
            heat = steam_trap_config.get_latent_heat(pressure)
            assert abs(heat - expected_heat) < 20, f"At {pressure} psig: got {heat}, expected {expected_heat}"

    def test_steam_temperature_calculation(self):
        """Test automatic steam temperature calculation from pressure."""
        config = SteamTrapMonitorConfig(
            plant_id="TEST",
            steam_pressure_psig=150.0,
            steam_temperature_f=None,  # Should be calculated
        )

        # Should calculate saturation temperature
        assert config.steam_temperature_f is not None
        assert 360 < config.steam_temperature_f < 375

    def test_safety_factors(self, steam_trap_config: SteamTrapMonitorConfig):
        """Test DOE safety factor settings."""
        assert steam_trap_config.safety_factor_startup >= 2.0
        assert steam_trap_config.safety_factor_operating >= 1.5
        assert steam_trap_config.safety_factor_startup >= steam_trap_config.safety_factor_operating

    def test_compliance_settings(self, steam_trap_config: SteamTrapMonitorConfig):
        """Test compliance and audit settings."""
        assert steam_trap_config.asme_b16_34_compliance is True
        assert steam_trap_config.audit_enabled is True
        assert steam_trap_config.provenance_tracking is True

    def test_alert_settings(self, steam_trap_config: SteamTrapMonitorConfig):
        """Test alert configuration."""
        assert steam_trap_config.alert_on_failed_open is True
        assert steam_trap_config.alert_on_failed_closed is True
        assert steam_trap_config.alert_steam_loss_threshold_lb_hr > 0


class TestConfigValidation:
    """Tests for configuration validation."""

    def test_invalid_pressure(self):
        """Test invalid pressure value is rejected."""
        with pytest.raises(ValueError):
            SteamTrapMonitorConfig(
                plant_id="TEST",
                steam_pressure_psig=-10.0,  # Invalid
            )

    def test_invalid_safety_factor(self):
        """Test invalid safety factor is rejected."""
        with pytest.raises(ValueError):
            SteamTrapMonitorConfig(
                plant_id="TEST",
                safety_factor_startup=0.5,  # Too low
            )

    def test_invalid_data_retention(self):
        """Test invalid data retention is rejected."""
        with pytest.raises(ValueError):
            SteamTrapMonitorConfig(
                plant_id="TEST",
                data_retention_days=10,  # Below minimum
            )


class TestConfigInterpolation:
    """Tests for steam property interpolation."""

    @pytest.mark.parametrize("pressure,expected_range", [
        (25, (260, 280)),   # Between 15 and 50 psig
        (75, (310, 330)),   # Between 50 and 100 psig
        (175, (375, 385)),  # Between 150 and 200 psig
    ])
    def test_saturation_temperature_interpolation(
        self,
        steam_trap_config: SteamTrapMonitorConfig,
        pressure: float,
        expected_range: tuple,
    ):
        """Test saturation temperature interpolation between known points."""
        temp = steam_trap_config.get_saturation_temperature(pressure)

        assert expected_range[0] < temp < expected_range[1], \
            f"At {pressure} psig: got {temp}, expected between {expected_range}"

    def test_extrapolation_low_pressure(self, steam_trap_config: SteamTrapMonitorConfig):
        """Test behavior at very low pressure."""
        temp = steam_trap_config.get_saturation_temperature(0)

        assert temp == 212.0  # Atmospheric boiling point

    def test_extrapolation_high_pressure(self, steam_trap_config: SteamTrapMonitorConfig):
        """Test behavior at very high pressure."""
        temp = steam_trap_config.get_saturation_temperature(600)

        assert temp > 480  # Should be above 500 psig value
