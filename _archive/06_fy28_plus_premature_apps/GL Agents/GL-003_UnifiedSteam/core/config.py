"""
GL-003 UNIFIEDSTEAM SteamSystemOptimizer - Configuration Module

This module defines all configuration schemas for the UNIFIEDSTEAM
SteamSystemOptimizer including IAPWS-IF97 settings, sensor configurations,
safety thresholds, optimization parameters, and operational settings.

All configurations use Pydantic for validation with comprehensive defaults
and documentation following IAPWS-IF97 and ASME PTC 19.11 standards.

Standards Compliance:
    - IAPWS-IF97 (International Association for Properties of Water and Steam)
    - ASME PTC 19.11 (Steam and Water in Industrial Systems)
    - ISO 50001 (Energy Management Systems)
    - IEC 61511 (Functional Safety - Safety Instrumented Systems)
    - GHG Protocol (Scope 1 emissions reporting)
"""

from datetime import timedelta
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Set, Tuple
import os

from pydantic import BaseModel, Field, validator, root_validator


# =============================================================================
# ENUMS
# =============================================================================

class OperatingState(Enum):
    """Steam system operating states."""
    STARTUP = "startup"
    NORMAL = "normal"
    OPTIMIZATION = "optimization"
    MAINTENANCE = "maintenance"
    SHUTDOWN = "shutdown"
    EMERGENCY = "emergency"
    STANDBY = "standby"
    WARMUP = "warmup"


class SteamQuality(Enum):
    """Steam quality classifications per IAPWS-IF97."""
    WET_STEAM = "wet_steam"          # x < 1.0, two-phase region
    SATURATED = "saturated"          # x = 1.0, saturated vapor line
    SUPERHEATED = "superheated"      # T > Tsat at given P
    SUBCOOLED = "subcooled"          # T < Tsat at given P (liquid)
    SUPERCRITICAL = "supercritical"  # P > 22.064 MPa


class OptimizationType(Enum):
    """Types of steam system optimization."""
    DESUPERHEATER = "desuperheater"
    CONDENSATE_RECOVERY = "condensate_recovery"
    TRAP_OPTIMIZATION = "trap_optimization"
    ENTHALPY_BALANCE = "enthalpy_balance"
    COMBINED = "combined"
    HEADER_PRESSURE = "header_pressure"
    BLOWDOWN = "blowdown"
    FLASH_STEAM = "flash_steam"


class DeploymentMode(Enum):
    """Agent deployment modes."""
    ADVISORY = "advisory"      # Recommendations only, no control actions
    CLOSED_LOOP = "closed_loop"  # Automatic control with operator oversight
    SHADOW = "shadow"          # Parallel run comparing to existing control
    MANUAL = "manual"          # Manual mode, data collection only


class SafetyIntegrityLevel(Enum):
    """IEC 61511 Safety Integrity Levels."""
    NONE = 0
    SIL_1 = 1
    SIL_2 = 2
    SIL_3 = 3
    SIL_4 = 4


class SensorType(Enum):
    """Types of sensors in steam systems."""
    PRESSURE = "pressure"
    TEMPERATURE = "temperature"
    FLOW = "flow"
    LEVEL = "level"
    QUALITY = "quality"          # Steam dryness fraction
    ACOUSTIC = "acoustic"        # For trap diagnostics
    CONDUCTIVITY = "conductivity"  # Water chemistry
    PH = "ph"
    VIBRATION = "vibration"


class TrapFailureMode(Enum):
    """Steam trap failure modes."""
    BLOW_THROUGH = "blow_through"    # Stuck open, leaking steam
    BLOCKED = "blocked"              # Stuck closed, condensate backup
    INTERNAL_LEAK = "internal_leak"  # Partial failure
    WORN = "worn"                    # Degraded performance
    HEALTHY = "healthy"              # Operating normally
    UNKNOWN = "unknown"              # Cannot determine


class MaintenancePriority(Enum):
    """Maintenance priority levels."""
    CRITICAL = "critical"    # Immediate attention required
    HIGH = "high"            # Within 24 hours
    MEDIUM = "medium"        # Within 1 week
    LOW = "low"              # Scheduled maintenance
    ROUTINE = "routine"      # Normal inspection cycle


class ConfidenceLevel(Enum):
    """Statistical confidence levels."""
    LOW = "low"              # < 80%
    MEDIUM = "medium"        # 80-90%
    HIGH = "high"            # 90-95%
    VERY_HIGH = "very_high"  # > 95%


# =============================================================================
# IAPWS-IF97 CONFIGURATION
# =============================================================================

class IAPWSIF97Config(BaseModel):
    """Configuration for IAPWS-IF97 thermodynamic calculations."""

    # Region boundaries per IAPWS-IF97
    pressure_min_kpa: float = Field(
        default=0.611657,  # Triple point
        ge=0.0,
        le=100000.0,
        description="Minimum pressure (kPa) - triple point"
    )
    pressure_max_kpa: float = Field(
        default=100000.0,  # 100 MPa
        ge=0.0,
        le=100000.0,
        description="Maximum pressure (kPa)"
    )
    temperature_min_c: float = Field(
        default=0.01,  # Triple point
        ge=-273.15,
        le=2000.0,
        description="Minimum temperature (C)"
    )
    temperature_max_c: float = Field(
        default=800.0,  # Region 2 limit
        ge=-273.15,
        le=2000.0,
        description="Maximum temperature (C)"
    )
    critical_pressure_kpa: float = Field(
        default=22064.0,  # Critical point
        description="Critical pressure (kPa)"
    )
    critical_temperature_c: float = Field(
        default=373.946,  # Critical point
        description="Critical temperature (C)"
    )

    # Calculation settings
    iteration_tolerance: float = Field(
        default=1e-9,
        ge=1e-15,
        le=1e-3,
        description="Convergence tolerance for iterative calculations"
    )
    max_iterations: int = Field(
        default=50,
        ge=5,
        le=1000,
        description="Maximum iterations for convergence"
    )
    use_backward_equations: bool = Field(
        default=True,
        description="Use backward equations for faster (T,P)->properties"
    )

    # Reference state (IIR convention for steam tables)
    reference_enthalpy_kj_kg: float = Field(
        default=0.0,
        description="Reference enthalpy at 0C saturated liquid"
    )
    reference_entropy_kj_kg_k: float = Field(
        default=0.0,
        description="Reference entropy at 0C saturated liquid"
    )


# =============================================================================
# SENSOR CONFIGURATIONS
# =============================================================================

class PressureSensorConfig(BaseModel):
    """Pressure sensor configuration."""

    sensor_id: str = Field(..., description="Unique sensor identifier")
    tag_name: str = Field(..., description="SCADA/DCS tag name")
    unit: str = Field(default="kPa", description="Pressure unit (kPa, bar, psi)")
    range_min: float = Field(default=0.0, description="Minimum measurement range")
    range_max: float = Field(default=5000.0, description="Maximum measurement range (kPa)")
    accuracy_percent: float = Field(
        default=0.25,
        ge=0.01,
        le=5.0,
        description="Sensor accuracy (%FS)"
    )
    span_check_enabled: bool = Field(
        default=True,
        description="Enable span validation"
    )
    redundancy_voting: Optional[str] = Field(
        default=None,
        description="Voting logic if redundant (2oo3, 1oo2)"
    )


class TemperatureSensorConfig(BaseModel):
    """Temperature sensor configuration."""

    sensor_id: str = Field(..., description="Unique sensor identifier")
    tag_name: str = Field(..., description="SCADA/DCS tag name")
    unit: str = Field(default="C", description="Temperature unit (C, F, K)")
    range_min: float = Field(default=0.0, description="Minimum measurement range")
    range_max: float = Field(default=600.0, description="Maximum measurement range (C)")
    sensor_type: str = Field(
        default="RTD_PT100",
        description="Sensor type (RTD_PT100, TC_K, TC_J)"
    )
    accuracy_c: float = Field(
        default=0.3,
        ge=0.01,
        le=5.0,
        description="Sensor accuracy (C)"
    )
    time_constant_s: float = Field(
        default=5.0,
        ge=0.1,
        le=60.0,
        description="Thermal time constant (seconds)"
    )


class FlowSensorConfig(BaseModel):
    """Flow sensor configuration."""

    sensor_id: str = Field(..., description="Unique sensor identifier")
    tag_name: str = Field(..., description="SCADA/DCS tag name")
    unit: str = Field(default="kg/s", description="Flow unit (kg/s, kg/h, t/h)")
    range_min: float = Field(default=0.0, description="Minimum measurement range")
    range_max: float = Field(default=100.0, description="Maximum measurement range")
    meter_type: str = Field(
        default="vortex",
        description="Meter type (vortex, orifice, coriolis, ultrasonic)"
    )
    accuracy_percent: float = Field(
        default=1.0,
        ge=0.1,
        le=5.0,
        description="Meter accuracy (% of reading)"
    )
    turndown_ratio: float = Field(
        default=20.0,
        ge=3.0,
        le=100.0,
        description="Meter turndown ratio"
    )
    density_compensation: bool = Field(
        default=True,
        description="Enable density compensation"
    )
    pressure_tag: Optional[str] = Field(
        default=None,
        description="Associated pressure tag for compensation"
    )
    temperature_tag: Optional[str] = Field(
        default=None,
        description="Associated temperature tag for compensation"
    )


class QualitySensorConfig(BaseModel):
    """Steam quality (dryness fraction) sensor configuration."""

    sensor_id: str = Field(..., description="Unique sensor identifier")
    tag_name: str = Field(..., description="SCADA/DCS tag name")
    measurement_method: str = Field(
        default="throttling_calorimeter",
        description="Method: throttling_calorimeter, separating_calorimeter, tracer"
    )
    range_min: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Minimum dryness fraction"
    )
    range_max: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Maximum dryness fraction"
    )
    accuracy_percent: float = Field(
        default=2.0,
        ge=0.5,
        le=10.0,
        description="Measurement accuracy (%)"
    )


class AcousticSensorConfig(BaseModel):
    """Acoustic sensor configuration for steam trap monitoring."""

    sensor_id: str = Field(..., description="Unique sensor identifier")
    tag_name: str = Field(..., description="SCADA/DCS tag name")
    frequency_range_hz: Tuple[float, float] = Field(
        default=(20.0, 100000.0),
        description="Frequency range (Hz)"
    )
    sampling_rate_hz: float = Field(
        default=44100.0,
        ge=8000.0,
        le=192000.0,
        description="Audio sampling rate (Hz)"
    )
    sensitivity_db: float = Field(
        default=-26.0,
        ge=-60.0,
        le=0.0,
        description="Microphone sensitivity (dB ref 1V/Pa)"
    )
    threshold_db: float = Field(
        default=60.0,
        ge=30.0,
        le=120.0,
        description="Detection threshold (dB)"
    )


# =============================================================================
# SAFETY AND THRESHOLD CONFIGURATIONS
# =============================================================================

class SafetyLimitsConfig(BaseModel):
    """Safety limits and trip setpoints."""

    # Pressure limits
    high_pressure_alarm_kpa: float = Field(
        default=4500.0,
        ge=100.0,
        le=50000.0,
        description="High pressure alarm setpoint (kPa)"
    )
    high_pressure_trip_kpa: float = Field(
        default=5000.0,
        ge=100.0,
        le=50000.0,
        description="High pressure trip setpoint (kPa)"
    )
    low_pressure_alarm_kpa: float = Field(
        default=200.0,
        ge=0.0,
        le=10000.0,
        description="Low pressure alarm setpoint (kPa)"
    )
    low_pressure_trip_kpa: float = Field(
        default=100.0,
        ge=0.0,
        le=10000.0,
        description="Low pressure trip setpoint (kPa)"
    )

    # Temperature limits
    high_temperature_alarm_c: float = Field(
        default=500.0,
        ge=100.0,
        le=800.0,
        description="High temperature alarm setpoint (C)"
    )
    high_temperature_trip_c: float = Field(
        default=550.0,
        ge=100.0,
        le=800.0,
        description="High temperature trip setpoint (C)"
    )
    low_temperature_alarm_c: float = Field(
        default=100.0,
        ge=0.0,
        le=300.0,
        description="Low temperature alarm setpoint (C)"
    )

    # Superheat limits (for desuperheater)
    min_superheat_c: float = Field(
        default=10.0,
        ge=0.0,
        le=50.0,
        description="Minimum superheat above saturation (C)"
    )
    max_superheat_c: float = Field(
        default=150.0,
        ge=20.0,
        le=300.0,
        description="Maximum superheat (C)"
    )

    # Flow limits
    min_flow_kg_s: float = Field(
        default=0.5,
        ge=0.0,
        le=100.0,
        description="Minimum flow for stable operation (kg/s)"
    )
    max_flow_kg_s: float = Field(
        default=200.0,
        ge=1.0,
        le=1000.0,
        description="Maximum flow capacity (kg/s)"
    )

    # Water quality limits
    max_conductivity_us_cm: float = Field(
        default=3000.0,
        ge=100.0,
        le=10000.0,
        description="Maximum boiler water conductivity (uS/cm)"
    )
    max_silica_ppm: float = Field(
        default=150.0,
        ge=1.0,
        le=500.0,
        description="Maximum silica in boiler water (ppm)"
    )
    max_tds_ppm: float = Field(
        default=3500.0,
        ge=100.0,
        le=10000.0,
        description="Maximum total dissolved solids (ppm)"
    )

    @root_validator
    def validate_limits(cls, values):
        """Validate that alarm < trip setpoints."""
        hp_alarm = values.get('high_pressure_alarm_kpa', 4500)
        hp_trip = values.get('high_pressure_trip_kpa', 5000)
        if hp_alarm >= hp_trip:
            raise ValueError("High pressure alarm must be less than trip setpoint")

        ht_alarm = values.get('high_temperature_alarm_c', 500)
        ht_trip = values.get('high_temperature_trip_c', 550)
        if ht_alarm >= ht_trip:
            raise ValueError("High temperature alarm must be less than trip setpoint")

        return values


class ThresholdConfig(BaseModel):
    """Operational thresholds for optimization and alerts."""

    # Efficiency thresholds
    min_acceptable_efficiency_percent: float = Field(
        default=75.0,
        ge=50.0,
        le=95.0,
        description="Minimum acceptable system efficiency (%)"
    )
    efficiency_alert_threshold_percent: float = Field(
        default=80.0,
        ge=50.0,
        le=95.0,
        description="Efficiency alert threshold (%)"
    )

    # Condensate recovery thresholds
    min_condensate_return_percent: float = Field(
        default=50.0,
        ge=0.0,
        le=100.0,
        description="Minimum condensate return ratio (%)"
    )
    target_condensate_return_percent: float = Field(
        default=85.0,
        ge=50.0,
        le=100.0,
        description="Target condensate return ratio (%)"
    )

    # Steam trap thresholds
    trap_failure_probability_threshold: float = Field(
        default=0.3,
        ge=0.0,
        le=1.0,
        description="Failure probability threshold for maintenance"
    )
    trap_loss_rate_threshold_kg_hr: float = Field(
        default=10.0,
        ge=0.1,
        le=100.0,
        description="Steam loss threshold triggering alert (kg/hr)"
    )

    # Energy loss thresholds
    max_flash_loss_percent: float = Field(
        default=15.0,
        ge=0.0,
        le=50.0,
        description="Maximum acceptable flash steam loss (%)"
    )
    max_radiation_loss_percent: float = Field(
        default=3.0,
        ge=0.0,
        le=10.0,
        description="Maximum acceptable radiation loss (%)"
    )

    # Mass balance tolerance
    mass_balance_tolerance_percent: float = Field(
        default=2.0,
        ge=0.1,
        le=10.0,
        description="Acceptable mass balance error (%)"
    )
    energy_balance_tolerance_percent: float = Field(
        default=3.0,
        ge=0.5,
        le=15.0,
        description="Acceptable energy balance error (%)"
    )


# =============================================================================
# SENSOR ARRAY CONFIGURATION
# =============================================================================

class SensorArrayConfig(BaseModel):
    """Complete sensor array configuration for steam system."""

    pressure_sensors: List[PressureSensorConfig] = Field(
        default_factory=list,
        description="Pressure sensors"
    )
    temperature_sensors: List[TemperatureSensorConfig] = Field(
        default_factory=list,
        description="Temperature sensors"
    )
    flow_sensors: List[FlowSensorConfig] = Field(
        default_factory=list,
        description="Flow sensors"
    )
    quality_sensors: List[QualitySensorConfig] = Field(
        default_factory=list,
        description="Steam quality sensors"
    )
    acoustic_sensors: List[AcousticSensorConfig] = Field(
        default_factory=list,
        description="Acoustic sensors for trap monitoring"
    )

    # Data quality settings
    data_validation_enabled: bool = Field(
        default=True,
        description="Enable sensor data validation"
    )
    outlier_detection_enabled: bool = Field(
        default=True,
        description="Enable outlier detection"
    )
    outlier_sigma_threshold: float = Field(
        default=3.0,
        ge=2.0,
        le=5.0,
        description="Standard deviations for outlier detection"
    )
    stale_data_timeout_s: float = Field(
        default=60.0,
        ge=5.0,
        le=600.0,
        description="Data staleness timeout (seconds)"
    )


# =============================================================================
# OPTIMIZATION CONFIGURATION
# =============================================================================

class OptimizationConfig(BaseModel):
    """Optimization algorithm configuration."""

    # Deployment mode
    deployment_mode: DeploymentMode = Field(
        default=DeploymentMode.ADVISORY,
        description="Deployment mode (advisory or closed-loop)"
    )

    # Optimization objectives
    primary_objective: str = Field(
        default="efficiency",
        description="Primary objective (efficiency, cost, emissions)"
    )
    efficiency_weight: float = Field(
        default=0.4,
        ge=0.0,
        le=1.0,
        description="Weight for efficiency in combined objective"
    )
    cost_weight: float = Field(
        default=0.35,
        ge=0.0,
        le=1.0,
        description="Weight for cost in combined objective"
    )
    emissions_weight: float = Field(
        default=0.25,
        ge=0.0,
        le=1.0,
        description="Weight for emissions in combined objective"
    )

    # Optimization algorithm settings
    algorithm: str = Field(
        default="hybrid",
        description="Algorithm (milp, genetic, hybrid, rule_based)"
    )
    max_iterations: int = Field(
        default=1000,
        ge=10,
        le=100000,
        description="Maximum optimization iterations"
    )
    convergence_tolerance: float = Field(
        default=1e-6,
        ge=1e-10,
        le=1e-2,
        description="Convergence tolerance"
    )
    time_limit_s: float = Field(
        default=30.0,
        ge=1.0,
        le=300.0,
        description="Optimization time limit (seconds)"
    )

    # Setpoint change limits
    max_setpoint_change_per_cycle: float = Field(
        default=5.0,
        ge=0.1,
        le=20.0,
        description="Maximum setpoint change per optimization cycle (%)"
    )
    min_time_between_changes_s: float = Field(
        default=300.0,
        ge=60.0,
        le=3600.0,
        description="Minimum time between setpoint changes (seconds)"
    )

    # Stability requirements
    stability_window_s: int = Field(
        default=300,
        ge=60,
        le=1800,
        description="Required stable period before optimization (seconds)"
    )
    exclude_during_transients: bool = Field(
        default=True,
        description="Exclude optimization during transient conditions"
    )
    load_change_exclusion_threshold_percent: float = Field(
        default=5.0,
        ge=1.0,
        le=20.0,
        description="Load change rate threshold for exclusion (%/min)"
    )

    # Operator approval
    require_operator_approval: bool = Field(
        default=True,
        description="Require operator approval for setpoint changes"
    )
    auto_implement_threshold_percent: float = Field(
        default=2.0,
        ge=0.1,
        le=10.0,
        description="Auto-implement threshold for small changes (%)"
    )

    @root_validator
    def validate_weights(cls, values):
        """Validate optimization weights sum to 1.0."""
        eff = values.get('efficiency_weight', 0.4)
        cost = values.get('cost_weight', 0.35)
        emis = values.get('emissions_weight', 0.25)
        total = eff + cost + emis
        if abs(total - 1.0) > 0.01:
            raise ValueError(f"Optimization weights must sum to 1.0, got {total}")
        return values

    class Config:
        use_enum_values = True


# =============================================================================
# EXPLAINABILITY CONFIGURATION
# =============================================================================

class ExplainabilityConfig(BaseModel):
    """Configuration for explainability features."""

    enabled: bool = Field(
        default=True,
        description="Enable explainability generation"
    )

    # SHAP configuration
    shap_enabled: bool = Field(
        default=True,
        description="Enable SHAP explanations"
    )
    shap_sample_size: int = Field(
        default=100,
        ge=10,
        le=1000,
        description="Sample size for SHAP background dataset"
    )
    shap_max_features: int = Field(
        default=10,
        ge=3,
        le=50,
        description="Maximum features to show in SHAP summary"
    )

    # LIME configuration
    lime_enabled: bool = Field(
        default=True,
        description="Enable LIME explanations"
    )
    lime_num_samples: int = Field(
        default=500,
        ge=100,
        le=5000,
        description="Number of samples for LIME"
    )
    lime_num_features: int = Field(
        default=10,
        ge=3,
        le=30,
        description="Number of features for LIME explanation"
    )

    # Physics trace
    include_physics_trace: bool = Field(
        default=True,
        description="Include physics-based calculation trace"
    )
    include_formula_references: bool = Field(
        default=True,
        description="Include formula/equation references"
    )

    # Confidence reporting
    include_confidence_bounds: bool = Field(
        default=True,
        description="Include confidence bounds in explanations"
    )


# =============================================================================
# UNCERTAINTY CONFIGURATION
# =============================================================================

class UncertaintyConfig(BaseModel):
    """Configuration for uncertainty quantification."""

    enabled: bool = Field(
        default=True,
        description="Enable uncertainty quantification"
    )

    # Confidence levels
    default_confidence_level: float = Field(
        default=0.95,
        ge=0.80,
        le=0.99,
        description="Default confidence level for bounds"
    )

    # Propagation method
    propagation_method: str = Field(
        default="monte_carlo",
        description="Uncertainty propagation (monte_carlo, taylor_series, ensemble)"
    )
    monte_carlo_samples: int = Field(
        default=1000,
        ge=100,
        le=100000,
        description="Monte Carlo sample count"
    )

    # Sensor uncertainty defaults
    default_pressure_uncertainty_percent: float = Field(
        default=0.5,
        ge=0.1,
        le=5.0,
        description="Default pressure measurement uncertainty (%)"
    )
    default_temperature_uncertainty_c: float = Field(
        default=1.0,
        ge=0.1,
        le=5.0,
        description="Default temperature measurement uncertainty (C)"
    )
    default_flow_uncertainty_percent: float = Field(
        default=2.0,
        ge=0.5,
        le=10.0,
        description="Default flow measurement uncertainty (%)"
    )

    # Model uncertainty
    model_uncertainty_factor: float = Field(
        default=0.05,
        ge=0.0,
        le=0.5,
        description="Additional model uncertainty factor"
    )


# =============================================================================
# INTEGRATION CONFIGURATION
# =============================================================================

class IntegrationConfig(BaseModel):
    """External system integration configuration."""

    # OPC-UA
    opcua_enabled: bool = Field(
        default=True,
        description="Enable OPC-UA connectivity"
    )
    opcua_endpoint: str = Field(
        default="opc.tcp://localhost:4840",
        description="OPC-UA server endpoint"
    )
    opcua_namespace: str = Field(
        default="urn:greenlang:unifiedsteam",
        description="OPC-UA namespace URI"
    )
    opcua_security_policy: str = Field(
        default="Basic256Sha256",
        description="OPC-UA security policy"
    )

    # MQTT
    mqtt_enabled: bool = Field(
        default=True,
        description="Enable MQTT messaging"
    )
    mqtt_broker: str = Field(
        default="localhost:1883",
        description="MQTT broker address"
    )
    mqtt_topic_prefix: str = Field(
        default="greenlang/steam",
        description="MQTT topic prefix"
    )

    # Kafka
    kafka_enabled: bool = Field(
        default=False,
        description="Enable Kafka event streaming"
    )
    kafka_bootstrap_servers: str = Field(
        default="localhost:9092",
        description="Kafka bootstrap servers"
    )
    kafka_topic_prefix: str = Field(
        default="steam-system",
        description="Kafka topic prefix"
    )

    # Database
    database_url: str = Field(
        default="postgresql://localhost:5432/greenlang_steam",
        description="Database connection URL"
    )
    timeseries_url: Optional[str] = Field(
        default=None,
        description="TimescaleDB/InfluxDB URL for process data"
    )

    # Polling intervals
    process_data_poll_ms: int = Field(
        default=1000,
        ge=100,
        le=60000,
        description="Process data polling interval (ms)"
    )
    optimization_cycle_s: int = Field(
        default=300,
        ge=60,
        le=3600,
        description="Optimization cycle interval (seconds)"
    )


# =============================================================================
# METRICS CONFIGURATION
# =============================================================================

class MetricsConfig(BaseModel):
    """Prometheus metrics configuration."""

    enabled: bool = Field(
        default=True,
        description="Enable metrics collection"
    )
    prefix: str = Field(
        default="greenlang_unifiedsteam",
        description="Metrics name prefix"
    )
    port: int = Field(
        default=9093,
        ge=1024,
        le=65535,
        description="Metrics HTTP port"
    )
    push_gateway_url: Optional[str] = Field(
        default=None,
        description="Prometheus push gateway URL"
    )
    collection_interval_s: float = Field(
        default=15.0,
        ge=1.0,
        le=300.0,
        description="Metrics collection interval"
    )
    include_thermodynamic_metrics: bool = Field(
        default=True,
        description="Include thermodynamic calculation metrics"
    )
    include_optimization_metrics: bool = Field(
        default=True,
        description="Include optimization metrics"
    )
    include_trap_metrics: bool = Field(
        default=True,
        description="Include steam trap diagnostics metrics"
    )
    histogram_buckets: List[float] = Field(
        default=[0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0],
        description="Histogram bucket boundaries"
    )


# =============================================================================
# MAIN CONFIGURATION
# =============================================================================

class SteamSystemConfig(BaseModel):
    """
    Complete GL-003 UNIFIEDSTEAM SteamSystemOptimizer configuration.

    This is the main configuration class aggregating all sub-configurations
    for comprehensive steam system optimization.

    Example:
        >>> config = SteamSystemConfig(
        ...     system_id="STEAM-001",
        ...     name="Main Steam System",
        ...     optimization=OptimizationConfig(
        ...         deployment_mode=DeploymentMode.ADVISORY
        ...     )
        ... )
        >>> optimizer = SteamSystemOrchestrator(config)
    """

    # Identity
    agent_id: str = Field(
        default_factory=lambda: f"GL-003-{os.getpid()}",
        description="Unique agent identifier"
    )
    system_id: str = Field(
        default="STEAM-001",
        description="Steam system identifier"
    )
    name: str = Field(
        default="UNIFIEDSTEAM-Primary",
        description="Human-readable agent name"
    )
    version: str = Field(
        default="1.0.0",
        description="Agent version"
    )

    # Environment
    environment: str = Field(
        default="production",
        description="Environment (development, staging, production)"
    )
    log_level: str = Field(
        default="INFO",
        description="Logging level"
    )

    # IAPWS-IF97 Settings
    iapws: IAPWSIF97Config = Field(
        default_factory=IAPWSIF97Config,
        description="IAPWS-IF97 thermodynamic calculation settings"
    )

    # Sensor Configuration
    sensors: SensorArrayConfig = Field(
        default_factory=SensorArrayConfig,
        description="Sensor array configuration"
    )

    # Safety Configuration
    safety_level: SafetyIntegrityLevel = Field(
        default=SafetyIntegrityLevel.SIL_2,
        description="Safety Integrity Level per IEC 61511"
    )
    safety_limits: SafetyLimitsConfig = Field(
        default_factory=SafetyLimitsConfig,
        description="Safety limits and trip setpoints"
    )

    # Operational Thresholds
    thresholds: ThresholdConfig = Field(
        default_factory=ThresholdConfig,
        description="Operational thresholds"
    )

    # Optimization
    optimization: OptimizationConfig = Field(
        default_factory=OptimizationConfig,
        description="Optimization configuration"
    )

    # Explainability
    explainability: ExplainabilityConfig = Field(
        default_factory=ExplainabilityConfig,
        description="Explainability configuration"
    )

    # Uncertainty
    uncertainty: UncertaintyConfig = Field(
        default_factory=UncertaintyConfig,
        description="Uncertainty quantification configuration"
    )

    # Integration
    integration: IntegrationConfig = Field(
        default_factory=IntegrationConfig,
        description="External system integration"
    )

    # Metrics
    metrics: MetricsConfig = Field(
        default_factory=MetricsConfig,
        description="Metrics configuration"
    )

    # Operational settings
    audit_enabled: bool = Field(
        default=True,
        description="Enable audit logging"
    )
    provenance_tracking: bool = Field(
        default=True,
        description="Enable SHA-256 provenance tracking"
    )
    deterministic_mode: bool = Field(
        default=True,
        description="Enable deterministic calculations (zero-hallucination)"
    )
    random_seed: int = Field(
        default=42,
        ge=0,
        description="Random seed for reproducibility"
    )

    # Steam system specific settings
    design_pressure_kpa: float = Field(
        default=4000.0,
        ge=100.0,
        le=50000.0,
        description="Design pressure of steam system (kPa)"
    )
    design_temperature_c: float = Field(
        default=400.0,
        ge=100.0,
        le=700.0,
        description="Design temperature of steam system (C)"
    )
    design_flow_kg_s: float = Field(
        default=50.0,
        ge=0.1,
        le=500.0,
        description="Design flow rate (kg/s)"
    )
    steam_cost_usd_per_ton: float = Field(
        default=25.0,
        ge=1.0,
        le=200.0,
        description="Steam cost for economic calculations ($/ton)"
    )
    electricity_cost_usd_per_kwh: float = Field(
        default=0.10,
        ge=0.01,
        le=1.0,
        description="Electricity cost ($/kWh)"
    )
    co2_emission_factor_kg_per_mmbtu: float = Field(
        default=53.06,
        ge=0.0,
        le=120.0,
        description="CO2 emission factor (kg/MMBTU) - natural gas default"
    )

    class Config:
        use_enum_values = True

    @validator("environment")
    def validate_environment(cls, v):
        """Validate environment is valid."""
        valid_envs = {"development", "staging", "production"}
        if v.lower() not in valid_envs:
            raise ValueError(f"Environment must be one of: {valid_envs}")
        return v.lower()

    @validator("log_level")
    def validate_log_level(cls, v):
        """Validate log level."""
        valid_levels = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        if v.upper() not in valid_levels:
            raise ValueError(f"Log level must be one of: {valid_levels}")
        return v.upper()

    @classmethod
    def from_environment(cls) -> "SteamSystemConfig":
        """
        Create configuration from environment variables.

        Environment variables follow the pattern:
        GL_UNIFIEDSTEAM_<SECTION>_<KEY>=value

        Example:
            GL_UNIFIEDSTEAM_SAFETY_LEVEL=SIL_2
            GL_UNIFIEDSTEAM_OPTIMIZATION_DEPLOYMENT_MODE=advisory
        """
        config_dict = {}
        prefix = "GL_UNIFIEDSTEAM_"

        for key, value in os.environ.items():
            if key.startswith(prefix):
                config_path = key[len(prefix):].lower().split("_")
                current = config_dict
                for part in config_path[:-1]:
                    current = current.setdefault(part, {})
                current[config_path[-1]] = value

        return cls(**config_dict)

    @classmethod
    def from_yaml(cls, yaml_path: str) -> "SteamSystemConfig":
        """Load configuration from YAML file."""
        import yaml
        with open(yaml_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        return cls(**config_dict)
