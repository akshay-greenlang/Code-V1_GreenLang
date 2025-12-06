# -*- coding: utf-8 -*-
"""
GL-013 PredictMaint Agent - Configuration Module

This module defines all configuration schemas for the Predictive Maintenance
agent, including sensor configurations, ML model settings, CMMS integration,
and safety thresholds.

Configuration follows GreenLang patterns with Pydantic validation and
sensible defaults for industrial equipment monitoring.

Example:
    >>> from greenlang.agents.process_heat.gl_013_predictive_maintenance.config import (
    ...     PredictiveMaintenanceConfig
    ... )
    >>> config = PredictiveMaintenanceConfig(
    ...     equipment_id="PUMP-001",
    ...     equipment_type="centrifugal_pump"
    ... )
"""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Set

from pydantic import BaseModel, Field, validator


# =============================================================================
# ENUMS
# =============================================================================

class EquipmentType(str, Enum):
    """Types of equipment supported for predictive maintenance."""
    CENTRIFUGAL_PUMP = "centrifugal_pump"
    POSITIVE_DISPLACEMENT_PUMP = "pd_pump"
    ELECTRIC_MOTOR = "electric_motor"
    GEARBOX = "gearbox"
    COMPRESSOR = "compressor"
    FAN = "fan"
    TURBINE = "turbine"
    HEAT_EXCHANGER = "heat_exchanger"
    BOILER = "boiler"
    FURNACE = "furnace"
    CONVEYOR = "conveyor"
    BEARING = "bearing"
    VALVE = "valve"


class FailureMode(str, Enum):
    """Common equipment failure modes."""
    BEARING_WEAR = "bearing_wear"
    BEARING_FATIGUE = "bearing_fatigue"
    IMBALANCE = "imbalance"
    MISALIGNMENT = "misalignment"
    LOOSENESS = "looseness"
    ROTOR_BAR_BREAK = "rotor_bar_break"
    STATOR_WINDING = "stator_winding"
    ECCENTRICITY = "eccentricity"
    CAVITATION = "cavitation"
    SEAL_FAILURE = "seal_failure"
    LUBRICATION_FAILURE = "lubrication_failure"
    FOULING = "fouling"
    CORROSION = "corrosion"
    FATIGUE_CRACK = "fatigue_crack"
    OVERHEATING = "overheating"


class MaintenanceStrategy(str, Enum):
    """Maintenance strategy types."""
    REACTIVE = "reactive"  # Run to failure
    PREVENTIVE = "preventive"  # Time-based
    PREDICTIVE = "predictive"  # Condition-based
    PRESCRIPTIVE = "prescriptive"  # AI-recommended


class CMMSType(str, Enum):
    """Supported CMMS systems."""
    SAP_PM = "sap_pm"
    IBM_MAXIMO = "ibm_maximo"
    EMAINT = "emaint"
    FIIX = "fiix"
    UPTIMEWORKS = "uptime_works"
    MPULSE = "mpulse"
    CUSTOM_API = "custom_api"


class SensorType(str, Enum):
    """Types of condition monitoring sensors."""
    ACCELEROMETER = "accelerometer"
    VELOCITY_SENSOR = "velocity_sensor"
    PROXIMITY_PROBE = "proximity_probe"
    TEMPERATURE = "temperature"
    IR_CAMERA = "ir_camera"
    CURRENT_CLAMP = "current_clamp"
    OIL_SENSOR = "oil_sensor"
    ULTRASONIC = "ultrasonic"
    PRESSURE = "pressure"
    FLOW = "flow"


class AlertSeverity(str, Enum):
    """Alert severity levels per ISO 10816."""
    GOOD = "good"  # Zone A
    ACCEPTABLE = "acceptable"  # Zone B
    UNSATISFACTORY = "unsatisfactory"  # Zone C
    UNACCEPTABLE = "unacceptable"  # Zone D - Immediate action


# =============================================================================
# SENSOR CONFIGURATION
# =============================================================================

class AccelerometerConfig(BaseModel):
    """Accelerometer sensor configuration."""

    sensor_id: str = Field(..., description="Unique sensor identifier")
    location: str = Field(..., description="Mounting location (DE, NDE, etc.)")
    orientation: str = Field(
        default="radial",
        description="Measurement orientation: radial, axial, tangential"
    )
    sensitivity_mv_g: float = Field(
        default=100.0,
        gt=0,
        description="Sensitivity in mV/g"
    )
    frequency_range_hz: tuple = Field(
        default=(0.5, 10000),
        description="Frequency range (min, max) Hz"
    )
    sampling_rate_hz: int = Field(
        default=25600,
        ge=1024,
        le=102400,
        description="Sampling rate in Hz"
    )
    samples_per_measurement: int = Field(
        default=16384,
        description="Number of samples per measurement"
    )
    window_type: str = Field(
        default="hanning",
        description="FFT window type: hanning, hamming, blackman, flat_top"
    )
    averaging_count: int = Field(
        default=4,
        ge=1,
        le=16,
        description="Number of averages for spectrum"
    )


class OilSensorConfig(BaseModel):
    """Oil analysis sensor configuration."""

    sensor_id: str = Field(..., description="Unique sensor identifier")
    sample_point: str = Field(..., description="Oil sampling location")
    oil_type: str = Field(
        default="mineral",
        description="Oil type: mineral, synthetic, pao"
    )
    viscosity_grade: str = Field(
        default="ISO_VG_46",
        description="ISO viscosity grade"
    )
    online_monitoring: bool = Field(
        default=False,
        description="Online vs offline analysis"
    )
    particle_counter_enabled: bool = Field(
        default=True,
        description="Enable particle counting"
    )
    moisture_sensor_enabled: bool = Field(
        default=True,
        description="Enable moisture monitoring"
    )


class IRCameraConfig(BaseModel):
    """Infrared camera configuration."""

    camera_id: str = Field(..., description="Camera identifier")
    resolution: tuple = Field(
        default=(640, 480),
        description="Image resolution (width, height)"
    )
    temperature_range_c: tuple = Field(
        default=(-20, 500),
        description="Temperature measurement range"
    )
    emissivity_default: float = Field(
        default=0.95,
        ge=0.1,
        le=1.0,
        description="Default surface emissivity"
    )
    alarm_delta_c: float = Field(
        default=15.0,
        gt=0,
        description="Temperature delta for alarm"
    )
    roi_enabled: bool = Field(
        default=True,
        description="Region of Interest enabled"
    )


class CurrentSensorConfig(BaseModel):
    """Motor current sensor configuration for MCSA."""

    sensor_id: str = Field(..., description="Sensor identifier")
    phases: int = Field(
        default=3,
        ge=1,
        le=3,
        description="Number of phases monitored"
    )
    current_range_a: float = Field(
        default=100.0,
        gt=0,
        description="Maximum current range (Amps)"
    )
    sampling_rate_hz: int = Field(
        default=10240,
        ge=2048,
        le=51200,
        description="Sampling rate for current signature"
    )
    line_frequency_hz: float = Field(
        default=60.0,
        description="Power line frequency (50 or 60 Hz)"
    )


# =============================================================================
# WEIBULL CONFIGURATION
# =============================================================================

class WeibullConfig(BaseModel):
    """Weibull analysis configuration."""

    method: str = Field(
        default="mle",
        description="Parameter estimation: mle, rank_regression, median_ranks"
    )
    confidence_level: float = Field(
        default=0.90,
        ge=0.50,
        le=0.99,
        description="Confidence level for intervals"
    )
    minimum_failures: int = Field(
        default=3,
        ge=2,
        description="Minimum failures for analysis"
    )
    censoring_enabled: bool = Field(
        default=True,
        description="Enable right-censored data handling"
    )
    beta_prior: Optional[float] = Field(
        default=None,
        description="Prior for shape parameter (Bayesian)"
    )
    eta_prior: Optional[float] = Field(
        default=None,
        description="Prior for scale parameter (Bayesian)"
    )
    use_historical_data: bool = Field(
        default=True,
        description="Use historical failure data"
    )


# =============================================================================
# ML MODEL CONFIGURATION
# =============================================================================

class MLModelConfig(BaseModel):
    """Machine learning model configuration."""

    enabled: bool = Field(default=True, description="Enable ML predictions")
    model_type: str = Field(
        default="ensemble",
        description="Model type: ensemble, gradient_boosting, random_forest, lstm"
    )
    ensemble_size: int = Field(
        default=10,
        ge=3,
        le=50,
        description="Number of models in ensemble"
    )
    confidence_threshold: float = Field(
        default=0.80,
        ge=0.50,
        le=0.99,
        description="Minimum confidence for predictions"
    )
    feature_importance_enabled: bool = Field(
        default=True,
        description="Calculate SHAP feature importance"
    )
    uncertainty_quantification: bool = Field(
        default=True,
        description="Enable uncertainty bounds"
    )
    online_learning_enabled: bool = Field(
        default=False,
        description="Enable incremental learning"
    )
    retrain_interval_days: int = Field(
        default=30,
        ge=7,
        le=365,
        description="Model retraining interval"
    )
    drift_detection_enabled: bool = Field(
        default=True,
        description="Enable concept drift detection"
    )
    drift_threshold: float = Field(
        default=0.10,
        ge=0.01,
        le=0.50,
        description="Drift detection threshold"
    )


# =============================================================================
# THRESHOLD CONFIGURATION
# =============================================================================

class VibrationThresholds(BaseModel):
    """Vibration alarm thresholds per ISO 10816."""

    # ISO 10816-3 for industrial machines
    velocity_good_mm_s: float = Field(
        default=2.8,
        gt=0,
        description="Good (Zone A) threshold mm/s RMS"
    )
    velocity_acceptable_mm_s: float = Field(
        default=4.5,
        gt=0,
        description="Acceptable (Zone B) threshold mm/s RMS"
    )
    velocity_unsatisfactory_mm_s: float = Field(
        default=7.1,
        gt=0,
        description="Unsatisfactory (Zone C) threshold mm/s RMS"
    )
    velocity_unacceptable_mm_s: float = Field(
        default=11.2,
        gt=0,
        description="Unacceptable (Zone D) threshold mm/s RMS"
    )

    # Acceleration thresholds
    acceleration_alarm_g: float = Field(
        default=5.0,
        gt=0,
        description="High frequency acceleration alarm (g)"
    )
    acceleration_trip_g: float = Field(
        default=10.0,
        gt=0,
        description="High frequency acceleration trip (g)"
    )

    # Trend thresholds
    trend_increase_pct: float = Field(
        default=25.0,
        gt=0,
        description="Percentage increase for trend alarm"
    )


class OilThresholds(BaseModel):
    """Oil analysis alarm thresholds."""

    # Viscosity
    viscosity_change_pct: float = Field(
        default=10.0,
        gt=0,
        description="Viscosity change from baseline (%)"
    )

    # Total Acid Number
    tan_warning_mg_koh_g: float = Field(
        default=2.0,
        gt=0,
        description="TAN warning threshold (mg KOH/g)"
    )
    tan_critical_mg_koh_g: float = Field(
        default=4.0,
        gt=0,
        description="TAN critical threshold (mg KOH/g)"
    )

    # Particle count ISO 4406
    particle_count_warning: str = Field(
        default="18/16/13",
        description="ISO 4406 warning code"
    )
    particle_count_critical: str = Field(
        default="20/18/15",
        description="ISO 4406 critical code"
    )

    # Metals (ppm)
    iron_warning_ppm: float = Field(default=100.0, gt=0)
    iron_critical_ppm: float = Field(default=200.0, gt=0)
    copper_warning_ppm: float = Field(default=50.0, gt=0)
    copper_critical_ppm: float = Field(default=100.0, gt=0)
    chromium_warning_ppm: float = Field(default=10.0, gt=0)
    chromium_critical_ppm: float = Field(default=25.0, gt=0)

    # Water content
    water_warning_ppm: float = Field(default=500.0, gt=0)
    water_critical_ppm: float = Field(default=1000.0, gt=0)


class TemperatureThresholds(BaseModel):
    """Temperature alarm thresholds."""

    bearing_warning_c: float = Field(
        default=70.0,
        description="Bearing temperature warning"
    )
    bearing_alarm_c: float = Field(
        default=85.0,
        description="Bearing temperature alarm"
    )
    bearing_trip_c: float = Field(
        default=95.0,
        description="Bearing temperature trip"
    )
    motor_winding_alarm_c: float = Field(
        default=130.0,
        description="Motor winding temperature alarm"
    )
    delta_alarm_c: float = Field(
        default=15.0,
        gt=0,
        description="Temperature delta alarm"
    )


class MCSAThresholds(BaseModel):
    """Motor Current Signature Analysis thresholds."""

    # Sideband thresholds (dB below fundamental)
    bearing_defect_db: float = Field(
        default=-40.0,
        description="Bearing defect sideband threshold (dB)"
    )
    rotor_bar_break_db: float = Field(
        default=-50.0,
        description="Rotor bar break sideband threshold (dB)"
    )
    eccentricity_db: float = Field(
        default=-45.0,
        description="Eccentricity sideband threshold (dB)"
    )
    current_unbalance_pct: float = Field(
        default=5.0,
        gt=0,
        description="Current unbalance alarm (%)"
    )


# =============================================================================
# CMMS CONFIGURATION
# =============================================================================

class CMMSConfig(BaseModel):
    """CMMS integration configuration."""

    enabled: bool = Field(default=True, description="Enable CMMS integration")
    system_type: CMMSType = Field(
        default=CMMSType.SAP_PM,
        description="CMMS system type"
    )
    api_endpoint: Optional[str] = Field(
        default=None,
        description="CMMS API endpoint URL"
    )
    api_key: Optional[str] = Field(
        default=None,
        description="CMMS API key (use secrets manager)"
    )
    plant_code: Optional[str] = Field(
        default=None,
        description="Plant/facility code"
    )
    auto_create_work_orders: bool = Field(
        default=False,
        description="Automatically create work orders"
    )
    work_order_priority_mapping: Dict[str, str] = Field(
        default_factory=lambda: {
            "critical": "1",
            "high": "2",
            "medium": "3",
            "low": "4"
        },
        description="Map severity to WO priority"
    )
    notification_enabled: bool = Field(
        default=True,
        description="Enable notifications"
    )
    notification_recipients: List[str] = Field(
        default_factory=list,
        description="Notification email recipients"
    )


# =============================================================================
# MAIN CONFIGURATION
# =============================================================================

class PredictiveMaintenanceConfig(BaseModel):
    """
    Main configuration for GL-013 Predictive Maintenance Agent.

    This configuration encompasses all aspects of predictive maintenance
    including sensor setup, ML models, thresholds, and CMMS integration.

    Attributes:
        equipment_id: Unique equipment identifier
        equipment_type: Type of equipment being monitored
        equipment_tag: Plant equipment tag (e.g., P-1001A)
        maintenance_strategy: Current maintenance strategy

    Example:
        >>> config = PredictiveMaintenanceConfig(
        ...     equipment_id="PUMP-001",
        ...     equipment_type=EquipmentType.CENTRIFUGAL_PUMP,
        ...     equipment_tag="P-1001A"
        ... )
    """

    # Equipment identification
    equipment_id: str = Field(..., description="Unique equipment identifier")
    equipment_type: EquipmentType = Field(
        ...,
        description="Equipment type"
    )
    equipment_tag: str = Field(
        default="",
        description="Plant equipment tag"
    )
    equipment_description: str = Field(
        default="",
        description="Equipment description"
    )
    location: str = Field(
        default="",
        description="Physical location"
    )

    # Operating parameters
    rated_speed_rpm: float = Field(
        default=1800.0,
        gt=0,
        description="Rated operating speed (RPM)"
    )
    rated_power_kw: float = Field(
        default=100.0,
        gt=0,
        description="Rated power (kW)"
    )
    number_of_poles: int = Field(
        default=4,
        ge=2,
        le=24,
        description="Motor pole count"
    )
    bearing_bpfo: Optional[float] = Field(
        default=None,
        description="Bearing ball pass frequency outer race"
    )
    bearing_bpfi: Optional[float] = Field(
        default=None,
        description="Bearing ball pass frequency inner race"
    )
    bearing_bsf: Optional[float] = Field(
        default=None,
        description="Bearing ball spin frequency"
    )
    bearing_ftf: Optional[float] = Field(
        default=None,
        description="Bearing fundamental train frequency"
    )

    # Maintenance settings
    maintenance_strategy: MaintenanceStrategy = Field(
        default=MaintenanceStrategy.PREDICTIVE,
        description="Maintenance strategy"
    )
    criticality: str = Field(
        default="medium",
        description="Equipment criticality: high, medium, low"
    )
    mtbf_hours: Optional[float] = Field(
        default=None,
        description="Mean Time Between Failures (hours)"
    )
    mttr_hours: Optional[float] = Field(
        default=None,
        description="Mean Time To Repair (hours)"
    )
    installation_date: Optional[datetime] = Field(
        default=None,
        description="Equipment installation date"
    )
    last_overhaul_date: Optional[datetime] = Field(
        default=None,
        description="Last major overhaul date"
    )
    running_hours: float = Field(
        default=0.0,
        ge=0,
        description="Total running hours"
    )

    # Sensor configurations
    accelerometers: List[AccelerometerConfig] = Field(
        default_factory=list,
        description="Accelerometer configurations"
    )
    oil_sensors: List[OilSensorConfig] = Field(
        default_factory=list,
        description="Oil sensor configurations"
    )
    ir_cameras: List[IRCameraConfig] = Field(
        default_factory=list,
        description="IR camera configurations"
    )
    current_sensors: List[CurrentSensorConfig] = Field(
        default_factory=list,
        description="Current sensor configurations"
    )

    # Analysis configurations
    weibull: WeibullConfig = Field(
        default_factory=WeibullConfig,
        description="Weibull analysis configuration"
    )
    ml_model: MLModelConfig = Field(
        default_factory=MLModelConfig,
        description="ML model configuration"
    )

    # Thresholds
    vibration_thresholds: VibrationThresholds = Field(
        default_factory=VibrationThresholds,
        description="Vibration alarm thresholds"
    )
    oil_thresholds: OilThresholds = Field(
        default_factory=OilThresholds,
        description="Oil analysis thresholds"
    )
    temperature_thresholds: TemperatureThresholds = Field(
        default_factory=TemperatureThresholds,
        description="Temperature thresholds"
    )
    mcsa_thresholds: MCSAThresholds = Field(
        default_factory=MCSAThresholds,
        description="MCSA thresholds"
    )

    # CMMS integration
    cmms: CMMSConfig = Field(
        default_factory=CMMSConfig,
        description="CMMS integration configuration"
    )

    # Monitoring failure modes
    monitored_failure_modes: Set[FailureMode] = Field(
        default_factory=lambda: {
            FailureMode.BEARING_WEAR,
            FailureMode.IMBALANCE,
            FailureMode.MISALIGNMENT,
            FailureMode.LUBRICATION_FAILURE,
        },
        description="Failure modes being monitored"
    )

    # Data retention
    data_retention_days: int = Field(
        default=365,
        ge=30,
        le=3650,
        description="Data retention period"
    )
    trend_history_days: int = Field(
        default=90,
        ge=7,
        le=365,
        description="Trend analysis history"
    )

    # Audit and provenance
    audit_enabled: bool = Field(
        default=True,
        description="Enable audit logging"
    )
    provenance_tracking: bool = Field(
        default=True,
        description="Enable SHA-256 provenance"
    )

    class Config:
        use_enum_values = True

    @validator("criticality")
    def validate_criticality(cls, v: str) -> str:
        """Validate criticality level."""
        valid = {"high", "medium", "low"}
        if v.lower() not in valid:
            raise ValueError(f"Criticality must be one of {valid}")
        return v.lower()

    @validator("equipment_type", pre=True)
    def validate_equipment_type(cls, v):
        """Convert string to enum if needed."""
        if isinstance(v, str):
            return EquipmentType(v)
        return v
