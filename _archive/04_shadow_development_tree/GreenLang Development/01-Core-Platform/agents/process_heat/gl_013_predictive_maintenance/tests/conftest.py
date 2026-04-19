# -*- coding: utf-8 -*-
"""
GL-013 PredictMaint Agent - Test Fixtures (conftest.py)

Shared pytest fixtures for all GL-013 test modules.
Provides standardized test data, configurations, and mock objects.

Usage:
    Fixtures are automatically available to all test modules in this directory.
    Import specific fixtures by parameter name in test functions.
"""

from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List
from unittest.mock import Mock, MagicMock, patch
import math
import pytest

# Import all necessary modules from the GL-013 package
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
    TemperatureThresholds,
    VibrationThresholds,
    WeibullConfig,
)

from greenlang.agents.process_heat.gl_013_predictive_maintenance.schemas import (
    CurrentReading,
    DiagnosisConfidence,
    FailurePrediction,
    HealthStatus,
    MaintenanceRecommendation,
    MCSAResult,
    OilAnalysisReading,
    OilAnalysisResult,
    PredictiveMaintenanceInput,
    PredictiveMaintenanceOutput,
    TemperatureReading,
    ThermalImage,
    ThermographyResult,
    TrendDirection,
    VibrationAnalysisResult,
    VibrationReading,
    WeibullAnalysisResult,
    WorkOrderPriority,
    WorkOrderRequest,
    WorkOrderType,
)

from greenlang.agents.process_heat.gl_013_predictive_maintenance.weibull import (
    FailureData,
    WeibullAnalyzer,
    WeibullParameters,
)

from greenlang.agents.process_heat.gl_013_predictive_maintenance.oil_analysis import (
    OilAnalyzer,
    OilBaseline,
)

from greenlang.agents.process_heat.gl_013_predictive_maintenance.vibration import (
    VibrationAnalyzer,
    BearingGeometry,
)

from greenlang.agents.process_heat.gl_013_predictive_maintenance.thermography import (
    ThermographyAnalyzer,
    ThermalReference,
)

from greenlang.agents.process_heat.gl_013_predictive_maintenance.mcsa import (
    MCSAAnalyzer,
    MotorParameters,
)

from greenlang.agents.process_heat.gl_013_predictive_maintenance.failure_prediction import (
    FailurePredictionEngine,
    FeatureEngineer,
)

from greenlang.agents.process_heat.gl_013_predictive_maintenance.work_order import (
    WorkOrderGenerator,
)


# =============================================================================
# CONFIGURATION FIXTURES
# =============================================================================

@pytest.fixture
def weibull_config() -> WeibullConfig:
    """Weibull analysis configuration for testing."""
    return WeibullConfig(
        method="mle",
        confidence_level=0.90,
        minimum_failures=3,
        censoring_enabled=True,
    )


@pytest.fixture
def vibration_thresholds() -> VibrationThresholds:
    """Vibration alarm thresholds per ISO 10816."""
    return VibrationThresholds(
        velocity_good_mm_s=2.8,
        velocity_acceptable_mm_s=4.5,
        velocity_unsatisfactory_mm_s=7.1,
        velocity_unacceptable_mm_s=11.2,
        acceleration_alarm_g=5.0,
        acceleration_trip_g=10.0,
        trend_increase_pct=25.0,
    )


@pytest.fixture
def oil_thresholds() -> OilThresholds:
    """Oil analysis alarm thresholds."""
    return OilThresholds(
        viscosity_change_pct=10.0,
        tan_warning_mg_koh_g=2.0,
        tan_critical_mg_koh_g=4.0,
        iron_warning_ppm=100.0,
        iron_critical_ppm=200.0,
        copper_warning_ppm=50.0,
        copper_critical_ppm=100.0,
        water_warning_ppm=500.0,
        water_critical_ppm=1000.0,
    )


@pytest.fixture
def temperature_thresholds() -> TemperatureThresholds:
    """Temperature alarm thresholds."""
    return TemperatureThresholds(
        bearing_warning_c=70.0,
        bearing_alarm_c=85.0,
        bearing_trip_c=95.0,
        motor_winding_alarm_c=130.0,
        delta_alarm_c=15.0,
    )


@pytest.fixture
def mcsa_thresholds() -> MCSAThresholds:
    """MCSA alarm thresholds."""
    return MCSAThresholds(
        bearing_defect_db=-40.0,
        rotor_bar_break_db=-50.0,
        eccentricity_db=-45.0,
        current_unbalance_pct=5.0,
    )


@pytest.fixture
def ml_model_config() -> MLModelConfig:
    """ML model configuration for testing."""
    return MLModelConfig(
        enabled=True,
        model_type="ensemble",
        ensemble_size=10,
        confidence_threshold=0.80,
        feature_importance_enabled=True,
        uncertainty_quantification=True,
    )


@pytest.fixture
def cmms_config() -> CMMSConfig:
    """CMMS configuration for testing."""
    return CMMSConfig(
        enabled=True,
        system_type=CMMSType.SAP_PM,
        api_endpoint="https://api.example.com/sap-pm",
        plant_code="1000",
        auto_create_work_orders=False,
    )


@pytest.fixture
def accelerometer_config() -> AccelerometerConfig:
    """Accelerometer sensor configuration."""
    return AccelerometerConfig(
        sensor_id="ACCEL-001",
        location="DE",
        orientation="radial",
        sensitivity_mv_g=100.0,
        sampling_rate_hz=25600,
    )


@pytest.fixture
def equipment_config(
    weibull_config,
    vibration_thresholds,
    oil_thresholds,
    temperature_thresholds,
    mcsa_thresholds,
    ml_model_config,
    cmms_config,
) -> PredictiveMaintenanceConfig:
    """Full equipment configuration for testing."""
    return PredictiveMaintenanceConfig(
        equipment_id="PUMP-001",
        equipment_type=EquipmentType.CENTRIFUGAL_PUMP,
        equipment_tag="P-1001A",
        equipment_description="Cooling Water Pump",
        location="Area 100",
        rated_speed_rpm=1800.0,
        rated_power_kw=100.0,
        number_of_poles=4,
        bearing_bpfo=3.56,
        bearing_bpfi=5.44,
        bearing_bsf=2.32,
        bearing_ftf=0.42,
        maintenance_strategy=MaintenanceStrategy.PREDICTIVE,
        criticality="high",
        running_hours=25000.0,
        weibull=weibull_config,
        ml_model=ml_model_config,
        vibration_thresholds=vibration_thresholds,
        oil_thresholds=oil_thresholds,
        temperature_thresholds=temperature_thresholds,
        mcsa_thresholds=mcsa_thresholds,
        cmms=cmms_config,
    )


# =============================================================================
# SENSOR READING FIXTURES
# =============================================================================

@pytest.fixture
def vibration_reading_healthy() -> VibrationReading:
    """Healthy vibration reading (Zone A)."""
    return VibrationReading(
        sensor_id="ACCEL-001",
        timestamp=datetime.now(timezone.utc),
        location="DE",
        orientation="radial",
        velocity_rms_mm_s=2.0,
        acceleration_rms_g=0.5,
        displacement_um=25.0,
        operating_speed_rpm=1795.0,
        temperature_c=55.0,
    )


@pytest.fixture
def vibration_reading_warning() -> VibrationReading:
    """Warning vibration reading (Zone C)."""
    return VibrationReading(
        sensor_id="ACCEL-001",
        timestamp=datetime.now(timezone.utc),
        location="DE",
        orientation="radial",
        velocity_rms_mm_s=6.5,
        acceleration_rms_g=2.0,
        displacement_um=80.0,
        operating_speed_rpm=1792.0,
        temperature_c=72.0,
    )


@pytest.fixture
def vibration_reading_critical() -> VibrationReading:
    """Critical vibration reading (Zone D)."""
    return VibrationReading(
        sensor_id="ACCEL-001",
        timestamp=datetime.now(timezone.utc),
        location="DE",
        orientation="radial",
        velocity_rms_mm_s=12.0,
        acceleration_rms_g=4.0,
        displacement_um=150.0,
        operating_speed_rpm=1788.0,
        temperature_c=88.0,
    )


@pytest.fixture
def vibration_reading_with_spectrum() -> VibrationReading:
    """Vibration reading with FFT spectrum data."""
    # Generate synthetic spectrum with 1x and 2x peaks
    freq_resolution = 1.0  # Hz
    spectrum_length = 500
    spectrum = [0.1] * spectrum_length

    # Add 1x peak at 30 Hz (1800 RPM)
    spectrum[30] = 1.5
    # Add 2x peak at 60 Hz
    spectrum[60] = 0.8
    # Add bearing defect frequency at ~107 Hz (BPFO at 3.56x)
    spectrum[107] = 0.3

    return VibrationReading(
        sensor_id="ACCEL-001",
        timestamp=datetime.now(timezone.utc),
        location="DE",
        orientation="radial",
        velocity_rms_mm_s=5.5,
        acceleration_rms_g=1.8,
        displacement_um=70.0,
        spectrum=spectrum,
        frequency_resolution_hz=freq_resolution,
        operating_speed_rpm=1800.0,
        temperature_c=65.0,
    )


@pytest.fixture
def oil_reading_healthy() -> OilAnalysisReading:
    """Healthy oil analysis reading."""
    return OilAnalysisReading(
        sample_id="OIL-001",
        timestamp=datetime.now(timezone.utc),
        sample_point="Main Sump",
        viscosity_40c_cst=46.0,
        viscosity_100c_cst=6.8,
        tan_mg_koh_g=0.5,
        water_ppm=100.0,
        particle_count_iso_4406="16/14/11",
        iron_ppm=25.0,
        copper_ppm=10.0,
        chromium_ppm=2.0,
        silicon_ppm=5.0,
    )


@pytest.fixture
def oil_reading_degraded() -> OilAnalysisReading:
    """Degraded oil analysis reading."""
    return OilAnalysisReading(
        sample_id="OIL-002",
        timestamp=datetime.now(timezone.utc),
        sample_point="Main Sump",
        viscosity_40c_cst=52.0,  # 13% increase
        viscosity_100c_cst=7.2,
        tan_mg_koh_g=2.5,  # Above warning
        water_ppm=600.0,  # Above warning
        particle_count_iso_4406="19/17/14",
        iron_ppm=120.0,  # Above warning
        copper_ppm=55.0,
        chromium_ppm=12.0,
        silicon_ppm=25.0,
    )


@pytest.fixture
def oil_reading_critical() -> OilAnalysisReading:
    """Critical oil analysis reading."""
    return OilAnalysisReading(
        sample_id="OIL-003",
        timestamp=datetime.now(timezone.utc),
        sample_point="Main Sump",
        viscosity_40c_cst=60.0,  # 30% increase
        viscosity_100c_cst=7.8,
        tan_mg_koh_g=4.5,  # Above critical
        water_ppm=1200.0,  # Above critical
        particle_count_iso_4406="21/19/16",
        iron_ppm=250.0,  # Above critical
        copper_ppm=110.0,
        chromium_ppm=30.0,
        silicon_ppm=50.0,
    )


@pytest.fixture
def oil_baseline() -> OilBaseline:
    """Baseline oil properties for comparison."""
    return OilBaseline(
        viscosity_40c_cst=46.0,
        viscosity_100c_cst=6.8,
        tan_mg_koh_g=0.3,
        iron_ppm=0.0,
        copper_ppm=0.0,
        chromium_ppm=0.0,
        water_ppm=50.0,
        particle_count_iso_4406="16/14/11",
    )


@pytest.fixture
def thermal_image_healthy() -> ThermalImage:
    """Healthy thermal image."""
    return ThermalImage(
        image_id="THERM-001",
        camera_id="IR-CAM-001",
        timestamp=datetime.now(timezone.utc),
        min_temperature_c=35.0,
        max_temperature_c=55.0,
        avg_temperature_c=45.0,
        hot_spots=[],
        emissivity=0.95,
        ambient_c=25.0,
    )


@pytest.fixture
def thermal_image_warning() -> ThermalImage:
    """Warning thermal image with hot spots."""
    return ThermalImage(
        image_id="THERM-002",
        camera_id="IR-CAM-001",
        timestamp=datetime.now(timezone.utc),
        min_temperature_c=35.0,
        max_temperature_c=85.0,
        avg_temperature_c=55.0,
        hot_spots=[
            {"x": 150, "y": 200, "temperature_c": 85.0, "area_pixels": 50},
        ],
        emissivity=0.95,
        ambient_c=25.0,
    )


@pytest.fixture
def thermal_image_critical() -> ThermalImage:
    """Critical thermal image with multiple hot spots."""
    return ThermalImage(
        image_id="THERM-003",
        camera_id="IR-CAM-001",
        timestamp=datetime.now(timezone.utc),
        min_temperature_c=40.0,
        max_temperature_c=120.0,
        avg_temperature_c=70.0,
        hot_spots=[
            {"x": 150, "y": 200, "temperature_c": 120.0, "area_pixels": 100},
            {"x": 300, "y": 250, "temperature_c": 105.0, "area_pixels": 75},
        ],
        emissivity=0.95,
        ambient_c=30.0,
    )


@pytest.fixture
def current_reading_healthy() -> CurrentReading:
    """Healthy motor current reading."""
    return CurrentReading(
        sensor_id="MCSA-001",
        timestamp=datetime.now(timezone.utc),
        phase_a_rms_a=100.0,
        phase_b_rms_a=99.5,
        phase_c_rms_a=100.5,
        current_unbalance_pct=1.0,
        line_frequency_hz=60.0,
        operating_speed_rpm=1795.0,
    )


@pytest.fixture
def current_reading_with_fault() -> CurrentReading:
    """Motor current reading with rotor bar fault signature."""
    # Generate spectrum with slip frequency sidebands
    freq_resolution = 0.1  # Hz
    spectrum_length = 1000
    spectrum = [0.001] * spectrum_length

    # Fundamental at 60 Hz
    fundamental_idx = int(60 / freq_resolution)
    spectrum[fundamental_idx] = 100.0

    # Slip frequency sidebands for rotor bar fault
    # At slip = 0.028, sideband at 60*(1-2*0.028) = 56.64 Hz
    slip = 0.028
    lower_sideband_idx = int(60 * (1 - 2 * slip) / freq_resolution)
    upper_sideband_idx = int(60 * (1 + 2 * slip) / freq_resolution)

    # -45 dB below fundamental indicates fault
    spectrum[lower_sideband_idx] = 100.0 * 10 ** (-45 / 20)
    spectrum[upper_sideband_idx] = 100.0 * 10 ** (-45 / 20)

    return CurrentReading(
        sensor_id="MCSA-001",
        timestamp=datetime.now(timezone.utc),
        phase_a_rms_a=105.0,
        phase_b_rms_a=103.0,
        phase_c_rms_a=107.0,
        current_unbalance_pct=3.8,
        spectrum_phase_a=spectrum,
        frequency_resolution_hz=freq_resolution,
        line_frequency_hz=60.0,
        operating_speed_rpm=1750.0,
    )


@pytest.fixture
def temperature_reading_healthy() -> TemperatureReading:
    """Healthy temperature reading."""
    return TemperatureReading(
        sensor_id="TEMP-001",
        timestamp=datetime.now(timezone.utc),
        location="Bearing DE",
        temperature_c=55.0,
        ambient_c=25.0,
        delta_c=30.0,
    )


# =============================================================================
# WEIBULL FIXTURES
# =============================================================================

@pytest.fixture
def failure_data_wearout() -> List[FailureData]:
    """Failure data representing wear-out pattern (beta > 2)."""
    return [
        FailureData(time=45000, is_failure=True),
        FailureData(time=48000, is_failure=True),
        FailureData(time=50000, is_failure=True),
        FailureData(time=52000, is_failure=True),
        FailureData(time=55000, is_failure=True),
    ]


@pytest.fixture
def failure_data_random() -> List[FailureData]:
    """Failure data representing random failures (beta ~ 1)."""
    return [
        FailureData(time=10000, is_failure=True),
        FailureData(time=25000, is_failure=True),
        FailureData(time=40000, is_failure=True),
        FailureData(time=55000, is_failure=True),
        FailureData(time=70000, is_failure=True),
    ]


@pytest.fixture
def failure_data_infant_mortality() -> List[FailureData]:
    """Failure data representing infant mortality (beta < 1)."""
    return [
        FailureData(time=100, is_failure=True),
        FailureData(time=500, is_failure=True),
        FailureData(time=2000, is_failure=True),
        FailureData(time=8000, is_failure=True),
        FailureData(time=25000, is_failure=True),
    ]


@pytest.fixture
def failure_data_with_censoring() -> List[FailureData]:
    """Failure data with right-censored observations."""
    return [
        FailureData(time=45000, is_failure=True),
        FailureData(time=48000, is_failure=True),
        FailureData(time=50000, is_failure=False),  # Still running
        FailureData(time=52000, is_failure=True),
        FailureData(time=55000, is_failure=False),  # Still running
        FailureData(time=58000, is_failure=True),
    ]


@pytest.fixture
def weibull_analyzer(weibull_config) -> WeibullAnalyzer:
    """Configured Weibull analyzer for testing."""
    return WeibullAnalyzer(weibull_config)


# =============================================================================
# ANALYZER FIXTURES
# =============================================================================

@pytest.fixture
def oil_analyzer(oil_thresholds, oil_baseline) -> OilAnalyzer:
    """Configured oil analyzer for testing."""
    return OilAnalyzer(oil_thresholds, oil_baseline)


@pytest.fixture
def vibration_analyzer(equipment_config) -> VibrationAnalyzer:
    """Configured vibration analyzer for testing."""
    return VibrationAnalyzer(
        equipment_config,
        equipment_config.vibration_thresholds,
    )


@pytest.fixture
def thermography_analyzer(temperature_thresholds) -> ThermographyAnalyzer:
    """Configured thermography analyzer for testing."""
    return ThermographyAnalyzer(temperature_thresholds)


@pytest.fixture
def mcsa_analyzer(equipment_config) -> MCSAAnalyzer:
    """Configured MCSA analyzer for testing."""
    return MCSAAnalyzer(
        equipment_config,
        equipment_config.mcsa_thresholds,
    )


@pytest.fixture
def motor_parameters() -> MotorParameters:
    """Motor parameters for MCSA testing."""
    return MotorParameters(
        rated_power_kw=100.0,
        rated_voltage_v=480.0,
        rated_current_a=150.0,
        rated_speed_rpm=1770.0,
        synchronous_speed_rpm=1800.0,
        rated_slip=0.0167,
        number_of_poles=4,
        number_of_rotor_bars=28,
    )


@pytest.fixture
def failure_prediction_engine(ml_model_config) -> FailurePredictionEngine:
    """Configured failure prediction engine for testing."""
    return FailurePredictionEngine(ml_model_config)


@pytest.fixture
def feature_engineer() -> FeatureEngineer:
    """Feature engineer for testing."""
    return FeatureEngineer()


@pytest.fixture
def work_order_generator(cmms_config) -> WorkOrderGenerator:
    """Work order generator for testing."""
    return WorkOrderGenerator(cmms_config)


# =============================================================================
# INPUT/OUTPUT FIXTURES
# =============================================================================

@pytest.fixture
def predictive_maintenance_input_healthy(
    vibration_reading_healthy,
    oil_reading_healthy,
    thermal_image_healthy,
    current_reading_healthy,
) -> PredictiveMaintenanceInput:
    """Full healthy input for predictive maintenance."""
    return PredictiveMaintenanceInput(
        equipment_id="PUMP-001",
        timestamp=datetime.now(timezone.utc),
        vibration_readings=[vibration_reading_healthy],
        oil_analysis=oil_reading_healthy,
        thermal_images=[thermal_image_healthy],
        current_readings=[current_reading_healthy],
        operating_speed_rpm=1795.0,
        load_percent=85.0,
        running_hours=25000.0,
    )


@pytest.fixture
def predictive_maintenance_input_warning(
    vibration_reading_warning,
    oil_reading_degraded,
    thermal_image_warning,
) -> PredictiveMaintenanceInput:
    """Warning-level input for predictive maintenance."""
    return PredictiveMaintenanceInput(
        equipment_id="PUMP-001",
        timestamp=datetime.now(timezone.utc),
        vibration_readings=[vibration_reading_warning],
        oil_analysis=oil_reading_degraded,
        thermal_images=[thermal_image_warning],
        current_readings=[],
        operating_speed_rpm=1792.0,
        load_percent=90.0,
        running_hours=35000.0,
    )


@pytest.fixture
def failure_prediction_sample() -> FailurePrediction:
    """Sample failure prediction for testing."""
    return FailurePrediction(
        failure_mode=FailureMode.BEARING_WEAR,
        probability=0.65,
        confidence=0.85,
        time_to_failure_hours=720.0,
        uncertainty_lower_hours=500.0,
        uncertainty_upper_hours=1000.0,
        feature_importance={
            "velocity_rms_normalized": 0.25,
            "bearing_defect_indicator": 0.30,
            "temperature_normalized": 0.15,
        },
        top_contributing_features=[
            "bearing_defect_indicator",
            "velocity_rms_normalized",
            "temperature_normalized",
        ],
        model_id="gl013_failure_pred",
        model_version="1.0.0",
    )


@pytest.fixture
def maintenance_recommendation_sample() -> MaintenanceRecommendation:
    """Sample maintenance recommendation for testing."""
    return MaintenanceRecommendation(
        failure_mode=FailureMode.BEARING_WEAR,
        priority=WorkOrderPriority.HIGH,
        action_type="inspection",
        description="Bearing wear detected. Schedule inspection.",
        deadline_hours=48.0,
        estimated_duration_hours=4.0,
        parts_required=["Bearings", "Seals", "Lubricant"],
        skills_required=["Mechanical", "Vibration analysis"],
        risk_if_delayed="Equipment failure and unplanned shutdown",
    )


# =============================================================================
# FEATURE VECTOR FIXTURES
# =============================================================================

@pytest.fixture
def feature_vector_healthy() -> Dict[str, float]:
    """Healthy feature vector for ML testing."""
    return {
        "velocity_rms_normalized": 0.15,
        "acceleration_rms_normalized": 0.08,
        "bearing_defect_indicator": 0.0,
        "imbalance_indicator": 0.0,
        "misalignment_indicator": 0.0,
        "viscosity_change_pct": 0.02,
        "tan_normalized": 0.12,
        "iron_ppm_normalized": 0.10,
        "water_ppm_normalized": 0.08,
        "temperature_normalized": 0.35,
        "delta_t_normalized": 0.20,
        "rotor_bar_severity_db": 0.0,
        "eccentricity_severity_db": 0.0,
        "current_unbalance_pct": 0.05,
        "running_hours_normalized": 0.50,
        "load_factor": 0.85,
        "health_score_composite": 92.0,
    }


@pytest.fixture
def feature_vector_degraded() -> Dict[str, float]:
    """Degraded feature vector for ML testing."""
    return {
        "velocity_rms_normalized": 0.55,
        "acceleration_rms_normalized": 0.35,
        "bearing_defect_indicator": 1.0,
        "imbalance_indicator": 0.0,
        "misalignment_indicator": 0.0,
        "viscosity_change_pct": 0.15,
        "tan_normalized": 0.50,
        "iron_ppm_normalized": 0.45,
        "water_ppm_normalized": 0.35,
        "temperature_normalized": 0.65,
        "delta_t_normalized": 0.45,
        "rotor_bar_severity_db": 0.25,
        "eccentricity_severity_db": 0.15,
        "current_unbalance_pct": 0.15,
        "running_hours_normalized": 0.75,
        "load_factor": 0.95,
        "health_score_composite": 55.0,
    }


# =============================================================================
# HELPER FIXTURES
# =============================================================================

@pytest.fixture
def mock_provenance_tracker():
    """Mock provenance tracker for testing."""
    tracker = Mock()
    tracker.record_calculation.return_value = Mock(
        provenance_hash="a" * 64,
    )
    return tracker


@pytest.fixture
def mock_audit_logger():
    """Mock audit logger for testing."""
    logger = Mock()
    logger.log_calculation.return_value = None
    return logger


@pytest.fixture
def sample_spectrum() -> List[float]:
    """Sample FFT spectrum for testing."""
    # Generate a 500-point spectrum with realistic peaks
    spectrum = [0.05] * 500

    # 1x at 30 Hz
    spectrum[30] = 2.0
    # 2x at 60 Hz
    spectrum[60] = 0.8
    # BPFO at ~107 Hz
    spectrum[107] = 0.4
    # Noise floor variations
    for i in range(0, 500, 10):
        spectrum[i] += 0.02

    return spectrum


# =============================================================================
# PARAMETRIZED TEST DATA
# =============================================================================

def pytest_generate_tests(metafunc):
    """Generate parametrized tests dynamically."""
    pass  # Implementation for parameterized test generation


# =============================================================================
# BENCHMARK FIXTURES
# =============================================================================

@pytest.fixture
def benchmark_data() -> Dict[str, Any]:
    """Large dataset for benchmark testing."""
    return {
        "vibration_readings": [
            VibrationReading(
                sensor_id=f"ACCEL-{i:03d}",
                timestamp=datetime.now(timezone.utc),
                location="DE",
                orientation="radial",
                velocity_rms_mm_s=2.0 + i * 0.01,
                acceleration_rms_g=0.5 + i * 0.005,
                operating_speed_rpm=1795.0,
            )
            for i in range(100)
        ],
    }
