# -*- coding: utf-8 -*-
"""
GL-013 PredictMaint Agent - Configuration Tests

Tests for configuration schemas and validation.
Validates all Pydantic models, enum values, and default configurations.

Coverage Target: 90%+
"""

import pytest
from datetime import datetime
from typing import Set

from greenlang.agents.process_heat.gl_013_predictive_maintenance.config import (
    AccelerometerConfig,
    AlertSeverity,
    CMMSConfig,
    CMMSType,
    CurrentSensorConfig,
    EquipmentType,
    FailureMode,
    IRCameraConfig,
    MaintenanceStrategy,
    MCSAThresholds,
    MLModelConfig,
    OilSensorConfig,
    OilThresholds,
    PredictiveMaintenanceConfig,
    SensorType,
    TemperatureThresholds,
    VibrationThresholds,
    WeibullConfig,
)


class TestEquipmentTypeEnum:
    """Tests for EquipmentType enumeration."""

    def test_all_equipment_types_exist(self):
        """Verify all expected equipment types are defined."""
        expected_types = {
            "centrifugal_pump",
            "pd_pump",
            "electric_motor",
            "gearbox",
            "compressor",
            "fan",
            "turbine",
            "heat_exchanger",
            "boiler",
            "furnace",
            "conveyor",
            "bearing",
            "valve",
        }

        actual_types = {e.value for e in EquipmentType}
        assert actual_types == expected_types

    def test_equipment_type_string_conversion(self):
        """Test enum string conversion."""
        assert EquipmentType.CENTRIFUGAL_PUMP.value == "centrifugal_pump"
        assert str(EquipmentType.ELECTRIC_MOTOR.value) == "electric_motor"


class TestFailureModeEnum:
    """Tests for FailureMode enumeration."""

    def test_all_failure_modes_exist(self):
        """Verify all expected failure modes are defined."""
        expected_modes = {
            "bearing_wear",
            "bearing_fatigue",
            "imbalance",
            "misalignment",
            "looseness",
            "rotor_bar_break",
            "stator_winding",
            "eccentricity",
            "cavitation",
            "seal_failure",
            "lubrication_failure",
            "fouling",
            "corrosion",
            "fatigue_crack",
            "overheating",
        }

        actual_modes = {f.value for f in FailureMode}
        assert actual_modes == expected_modes

    def test_failure_mode_membership(self):
        """Test failure mode membership checks."""
        assert FailureMode.BEARING_WEAR in FailureMode
        assert FailureMode("imbalance") == FailureMode.IMBALANCE


class TestAlertSeverityEnum:
    """Tests for AlertSeverity enumeration (ISO 10816 zones)."""

    def test_alert_severity_values(self):
        """Verify alert severity values match ISO 10816 zones."""
        assert AlertSeverity.GOOD.value == "good"
        assert AlertSeverity.ACCEPTABLE.value == "acceptable"
        assert AlertSeverity.UNSATISFACTORY.value == "unsatisfactory"
        assert AlertSeverity.UNACCEPTABLE.value == "unacceptable"

    def test_severity_ordering(self):
        """Test severity can be compared via string values."""
        severities = [
            AlertSeverity.GOOD,
            AlertSeverity.ACCEPTABLE,
            AlertSeverity.UNSATISFACTORY,
            AlertSeverity.UNACCEPTABLE,
        ]
        # All should be distinct
        assert len(set(s.value for s in severities)) == 4


class TestAccelerometerConfig:
    """Tests for AccelerometerConfig model."""

    def test_valid_config(self):
        """Test valid accelerometer configuration."""
        config = AccelerometerConfig(
            sensor_id="ACCEL-001",
            location="DE",
            orientation="radial",
            sensitivity_mv_g=100.0,
            sampling_rate_hz=25600,
        )

        assert config.sensor_id == "ACCEL-001"
        assert config.location == "DE"
        assert config.sensitivity_mv_g == 100.0

    def test_default_values(self):
        """Test default configuration values."""
        config = AccelerometerConfig(
            sensor_id="ACCEL-001",
            location="DE",
        )

        assert config.orientation == "radial"
        assert config.sensitivity_mv_g == 100.0
        assert config.sampling_rate_hz == 25600
        assert config.samples_per_measurement == 16384
        assert config.window_type == "hanning"
        assert config.averaging_count == 4

    def test_sampling_rate_bounds(self):
        """Test sampling rate validation bounds."""
        # Valid minimum
        config = AccelerometerConfig(
            sensor_id="ACCEL-001",
            location="DE",
            sampling_rate_hz=1024,
        )
        assert config.sampling_rate_hz == 1024

        # Valid maximum
        config = AccelerometerConfig(
            sensor_id="ACCEL-001",
            location="DE",
            sampling_rate_hz=102400,
        )
        assert config.sampling_rate_hz == 102400

        # Invalid - below minimum
        with pytest.raises(ValueError):
            AccelerometerConfig(
                sensor_id="ACCEL-001",
                location="DE",
                sampling_rate_hz=512,  # Below 1024 minimum
            )

    def test_sensitivity_must_be_positive(self):
        """Test sensitivity must be positive."""
        with pytest.raises(ValueError):
            AccelerometerConfig(
                sensor_id="ACCEL-001",
                location="DE",
                sensitivity_mv_g=-100.0,
            )


class TestOilSensorConfig:
    """Tests for OilSensorConfig model."""

    def test_valid_config(self):
        """Test valid oil sensor configuration."""
        config = OilSensorConfig(
            sensor_id="OIL-001",
            sample_point="Main Sump",
            oil_type="synthetic",
            viscosity_grade="ISO_VG_68",
        )

        assert config.sensor_id == "OIL-001"
        assert config.oil_type == "synthetic"

    def test_default_values(self):
        """Test default values."""
        config = OilSensorConfig(
            sensor_id="OIL-001",
            sample_point="Sump",
        )

        assert config.oil_type == "mineral"
        assert config.viscosity_grade == "ISO_VG_46"
        assert config.online_monitoring is False
        assert config.particle_counter_enabled is True


class TestIRCameraConfig:
    """Tests for IRCameraConfig model."""

    def test_valid_config(self):
        """Test valid IR camera configuration."""
        config = IRCameraConfig(
            camera_id="IR-001",
            resolution=(640, 480),
            emissivity_default=0.90,
        )

        assert config.camera_id == "IR-001"
        assert config.emissivity_default == 0.90

    def test_emissivity_bounds(self):
        """Test emissivity validation bounds."""
        # Valid emissivity
        config = IRCameraConfig(
            camera_id="IR-001",
            emissivity_default=0.95,
        )
        assert config.emissivity_default == 0.95

        # Invalid - below minimum
        with pytest.raises(ValueError):
            IRCameraConfig(
                camera_id="IR-001",
                emissivity_default=0.05,  # Below 0.1 minimum
            )

        # Invalid - above maximum
        with pytest.raises(ValueError):
            IRCameraConfig(
                camera_id="IR-001",
                emissivity_default=1.5,  # Above 1.0 maximum
            )


class TestCurrentSensorConfig:
    """Tests for CurrentSensorConfig model."""

    def test_valid_config(self):
        """Test valid current sensor configuration."""
        config = CurrentSensorConfig(
            sensor_id="CT-001",
            phases=3,
            current_range_a=200.0,
            line_frequency_hz=60.0,
        )

        assert config.phases == 3
        assert config.current_range_a == 200.0

    def test_phase_bounds(self):
        """Test phase count bounds."""
        # Single phase
        config = CurrentSensorConfig(
            sensor_id="CT-001",
            phases=1,
        )
        assert config.phases == 1

        # Invalid - too many phases
        with pytest.raises(ValueError):
            CurrentSensorConfig(
                sensor_id="CT-001",
                phases=4,  # Max is 3
            )


class TestWeibullConfig:
    """Tests for WeibullConfig model."""

    def test_valid_config(self, weibull_config):
        """Test valid Weibull configuration."""
        assert weibull_config.method == "mle"
        assert weibull_config.confidence_level == 0.90
        assert weibull_config.minimum_failures == 3

    def test_default_values(self):
        """Test default values."""
        config = WeibullConfig()

        assert config.method == "mle"
        assert config.confidence_level == 0.90
        assert config.minimum_failures == 3
        assert config.censoring_enabled is True

    def test_confidence_level_bounds(self):
        """Test confidence level bounds."""
        # Valid minimum
        config = WeibullConfig(confidence_level=0.50)
        assert config.confidence_level == 0.50

        # Valid maximum
        config = WeibullConfig(confidence_level=0.99)
        assert config.confidence_level == 0.99

        # Invalid bounds
        with pytest.raises(ValueError):
            WeibullConfig(confidence_level=0.49)

        with pytest.raises(ValueError):
            WeibullConfig(confidence_level=1.0)


class TestMLModelConfig:
    """Tests for MLModelConfig model."""

    def test_valid_config(self, ml_model_config):
        """Test valid ML model configuration."""
        assert ml_model_config.enabled is True
        assert ml_model_config.model_type == "ensemble"
        assert ml_model_config.ensemble_size == 10

    def test_default_values(self):
        """Test default values."""
        config = MLModelConfig()

        assert config.enabled is True
        assert config.model_type == "ensemble"
        assert config.confidence_threshold == 0.80
        assert config.retrain_interval_days == 30

    def test_ensemble_size_bounds(self):
        """Test ensemble size bounds."""
        # Valid bounds
        config = MLModelConfig(ensemble_size=3)
        assert config.ensemble_size == 3

        config = MLModelConfig(ensemble_size=50)
        assert config.ensemble_size == 50

        # Invalid bounds
        with pytest.raises(ValueError):
            MLModelConfig(ensemble_size=2)


class TestVibrationThresholds:
    """Tests for VibrationThresholds model."""

    def test_valid_thresholds(self, vibration_thresholds):
        """Test valid vibration thresholds."""
        assert vibration_thresholds.velocity_good_mm_s == 2.8
        assert vibration_thresholds.velocity_acceptable_mm_s == 4.5
        assert vibration_thresholds.velocity_unsatisfactory_mm_s == 7.1
        assert vibration_thresholds.velocity_unacceptable_mm_s == 11.2

    def test_iso_10816_defaults(self):
        """Test ISO 10816-3 default thresholds."""
        thresholds = VibrationThresholds()

        # ISO 10816-3 Class III defaults
        assert thresholds.velocity_good_mm_s == 2.8
        assert thresholds.velocity_acceptable_mm_s == 4.5
        assert thresholds.velocity_unsatisfactory_mm_s == 7.1
        assert thresholds.velocity_unacceptable_mm_s == 11.2

    def test_threshold_ordering(self, vibration_thresholds):
        """Test thresholds are properly ordered."""
        assert vibration_thresholds.velocity_good_mm_s < \
               vibration_thresholds.velocity_acceptable_mm_s < \
               vibration_thresholds.velocity_unsatisfactory_mm_s < \
               vibration_thresholds.velocity_unacceptable_mm_s


class TestOilThresholds:
    """Tests for OilThresholds model."""

    def test_valid_thresholds(self, oil_thresholds):
        """Test valid oil thresholds."""
        assert oil_thresholds.tan_warning_mg_koh_g == 2.0
        assert oil_thresholds.tan_critical_mg_koh_g == 4.0
        assert oil_thresholds.iron_warning_ppm == 100.0

    def test_warning_less_than_critical(self, oil_thresholds):
        """Test warning thresholds are less than critical."""
        assert oil_thresholds.tan_warning_mg_koh_g < oil_thresholds.tan_critical_mg_koh_g
        assert oil_thresholds.iron_warning_ppm < oil_thresholds.iron_critical_ppm
        assert oil_thresholds.water_warning_ppm < oil_thresholds.water_critical_ppm


class TestTemperatureThresholds:
    """Tests for TemperatureThresholds model."""

    def test_valid_thresholds(self, temperature_thresholds):
        """Test valid temperature thresholds."""
        assert temperature_thresholds.bearing_warning_c == 70.0
        assert temperature_thresholds.bearing_alarm_c == 85.0
        assert temperature_thresholds.bearing_trip_c == 95.0

    def test_threshold_ordering(self, temperature_thresholds):
        """Test temperature thresholds are properly ordered."""
        assert temperature_thresholds.bearing_warning_c < \
               temperature_thresholds.bearing_alarm_c < \
               temperature_thresholds.bearing_trip_c


class TestMCSAThresholds:
    """Tests for MCSAThresholds model."""

    def test_valid_thresholds(self, mcsa_thresholds):
        """Test valid MCSA thresholds."""
        assert mcsa_thresholds.bearing_defect_db == -40.0
        assert mcsa_thresholds.rotor_bar_break_db == -50.0
        assert mcsa_thresholds.current_unbalance_pct == 5.0

    def test_db_thresholds_are_negative(self, mcsa_thresholds):
        """Test dB thresholds are negative (below fundamental)."""
        assert mcsa_thresholds.bearing_defect_db < 0
        assert mcsa_thresholds.rotor_bar_break_db < 0
        assert mcsa_thresholds.eccentricity_db < 0


class TestCMMSConfig:
    """Tests for CMMSConfig model."""

    def test_valid_config(self, cmms_config):
        """Test valid CMMS configuration."""
        assert cmms_config.enabled is True
        assert cmms_config.system_type == CMMSType.SAP_PM

    def test_default_priority_mapping(self):
        """Test default priority mapping."""
        config = CMMSConfig()

        assert "critical" in config.work_order_priority_mapping
        assert "high" in config.work_order_priority_mapping
        assert config.work_order_priority_mapping["critical"] == "1"


class TestPredictiveMaintenanceConfig:
    """Tests for main PredictiveMaintenanceConfig model."""

    def test_valid_config(self, equipment_config):
        """Test valid equipment configuration."""
        assert equipment_config.equipment_id == "PUMP-001"
        assert equipment_config.equipment_type == EquipmentType.CENTRIFUGAL_PUMP
        assert equipment_config.rated_speed_rpm == 1800.0

    def test_required_fields(self):
        """Test required fields validation."""
        # Should raise without equipment_id
        with pytest.raises(ValueError):
            PredictiveMaintenanceConfig(
                equipment_type=EquipmentType.CENTRIFUGAL_PUMP,
            )

    def test_criticality_validation(self):
        """Test criticality level validation."""
        # Valid criticality
        config = PredictiveMaintenanceConfig(
            equipment_id="PUMP-001",
            equipment_type=EquipmentType.CENTRIFUGAL_PUMP,
            criticality="high",
        )
        assert config.criticality == "high"

        # Invalid criticality
        with pytest.raises(ValueError):
            PredictiveMaintenanceConfig(
                equipment_id="PUMP-001",
                equipment_type=EquipmentType.CENTRIFUGAL_PUMP,
                criticality="extreme",  # Not valid
            )

    def test_equipment_type_string_conversion(self):
        """Test equipment type string to enum conversion."""
        config = PredictiveMaintenanceConfig(
            equipment_id="PUMP-001",
            equipment_type="centrifugal_pump",  # String input
        )
        assert config.equipment_type == EquipmentType.CENTRIFUGAL_PUMP

    def test_default_failure_modes(self):
        """Test default monitored failure modes."""
        config = PredictiveMaintenanceConfig(
            equipment_id="PUMP-001",
            equipment_type=EquipmentType.CENTRIFUGAL_PUMP,
        )

        # Should include common failure modes by default
        assert FailureMode.BEARING_WEAR in config.monitored_failure_modes
        assert FailureMode.IMBALANCE in config.monitored_failure_modes

    def test_nested_config_defaults(self):
        """Test nested configuration defaults are created."""
        config = PredictiveMaintenanceConfig(
            equipment_id="PUMP-001",
            equipment_type=EquipmentType.CENTRIFUGAL_PUMP,
        )

        assert config.weibull is not None
        assert config.ml_model is not None
        assert config.vibration_thresholds is not None
        assert config.oil_thresholds is not None

    def test_bearing_frequencies(self, equipment_config):
        """Test bearing frequency configuration."""
        assert equipment_config.bearing_bpfo == 3.56
        assert equipment_config.bearing_bpfi == 5.44
        assert equipment_config.bearing_bsf == 2.32
        assert equipment_config.bearing_ftf == 0.42

    def test_running_hours_non_negative(self):
        """Test running hours must be non-negative."""
        config = PredictiveMaintenanceConfig(
            equipment_id="PUMP-001",
            equipment_type=EquipmentType.CENTRIFUGAL_PUMP,
            running_hours=0.0,
        )
        assert config.running_hours == 0.0

        with pytest.raises(ValueError):
            PredictiveMaintenanceConfig(
                equipment_id="PUMP-001",
                equipment_type=EquipmentType.CENTRIFUGAL_PUMP,
                running_hours=-100.0,
            )

    def test_data_retention_bounds(self):
        """Test data retention bounds."""
        # Valid bounds
        config = PredictiveMaintenanceConfig(
            equipment_id="PUMP-001",
            equipment_type=EquipmentType.CENTRIFUGAL_PUMP,
            data_retention_days=30,
        )
        assert config.data_retention_days == 30

        config = PredictiveMaintenanceConfig(
            equipment_id="PUMP-001",
            equipment_type=EquipmentType.CENTRIFUGAL_PUMP,
            data_retention_days=3650,
        )
        assert config.data_retention_days == 3650

    def test_serialization(self, equipment_config):
        """Test configuration serialization."""
        # To dict
        config_dict = equipment_config.dict()
        assert config_dict["equipment_id"] == "PUMP-001"
        assert config_dict["equipment_type"] == "centrifugal_pump"

        # To JSON
        config_json = equipment_config.json()
        assert "PUMP-001" in config_json


class TestConfigIntegration:
    """Integration tests for configuration modules."""

    def test_full_config_creation(self):
        """Test creating full configuration programmatically."""
        config = PredictiveMaintenanceConfig(
            equipment_id="MOTOR-001",
            equipment_type=EquipmentType.ELECTRIC_MOTOR,
            equipment_tag="M-2001A",
            rated_speed_rpm=3600.0,
            rated_power_kw=250.0,
            number_of_poles=2,
            maintenance_strategy=MaintenanceStrategy.PREDICTIVE,
            criticality="high",
            accelerometers=[
                AccelerometerConfig(
                    sensor_id="ACCEL-DE",
                    location="DE",
                ),
                AccelerometerConfig(
                    sensor_id="ACCEL-NDE",
                    location="NDE",
                ),
            ],
            current_sensors=[
                CurrentSensorConfig(
                    sensor_id="CT-001",
                    phases=3,
                ),
            ],
        )

        assert config.equipment_id == "MOTOR-001"
        assert len(config.accelerometers) == 2
        assert len(config.current_sensors) == 1
        assert config.number_of_poles == 2
