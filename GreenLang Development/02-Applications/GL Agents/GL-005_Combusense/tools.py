# -*- coding: utf-8 -*-
"""
GL-005 CombustionControlAgent - Tool Definitions

This module provides deterministic tool definitions for real-time combustion control.
All tools follow zero-hallucination design with physics-based calculations and
complete audit trail support.

Tools are organized by functional category:
1. Data Acquisition (3 tools)
2. Combustion Analysis (3 tools)
3. Control Optimization (3 tools)
4. Command Execution (2 tools)
5. Safety & Audit (2 tools)

Total: 13 tools for comprehensive combustion control automation.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
from enum import Enum

from pydantic import BaseModel, Field, field_validator, ValidationInfo


# ============================================================================
# ENUMERATIONS
# ============================================================================

class FuelType(str, Enum):
    """Supported fuel types"""
    NATURAL_GAS = "natural_gas"
    FUEL_OIL = "fuel_oil"
    COAL = "coal"
    BIOMASS = "biomass"
    PROPANE = "propane"
    HYDROGEN = "hydrogen"


class ControlMode(str, Enum):
    """Control operation modes"""
    AUTOMATIC = "automatic"
    MANUAL = "manual"
    SETPOINT_ONLY = "setpoint_only"
    EMERGENCY = "emergency"


class SafetyStatus(str, Enum):
    """Safety system status"""
    OK = "ok"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY_STOP = "emergency_stop"


class ComplianceStatus(str, Enum):
    """Regulatory compliance status"""
    COMPLIANT = "compliant"
    WARNING = "warning"
    VIOLATION = "violation"


# ============================================================================
# CATEGORY 1: DATA ACQUISITION TOOLS (3 tools)
# ============================================================================

class ReadCombustionDataInput(BaseModel):
    """Input schema for read_combustion_data tool"""
    unit_id: str = Field(..., description="Combustion unit identifier (e.g., BOILER001)")
    data_sources: List[str] = Field(
        ...,
        description="Data sources to query: [dcs, plc, cems, sensors]"
    )
    sampling_rate_hz: int = Field(
        100,
        ge=1,
        le=1000,
        description="Data acquisition rate (1-1000 Hz)"
    )
    timeout_ms: int = Field(
        1000,
        ge=100,
        le=5000,
        description="Read timeout in milliseconds"
    )

    @field_validator('data_sources')
    @classmethod
    def validate_sources(cls, v: List[str]) -> List[str]:
        valid_sources = ['dcs', 'plc', 'cems', 'sensors']
        for source in v:
            if source not in valid_sources:
                raise ValueError(f"Invalid data source: {source}. Must be one of {valid_sources}")
        return v


class ReadCombustionDataOutput(BaseModel):
    """Output schema for read_combustion_data tool"""
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    unit_id: str

    # Fuel parameters
    fuel_flow_rate_kg_hr: float = Field(..., description="Fuel flow rate (kg/hr)")
    fuel_pressure_bar: float = Field(..., description="Fuel pressure (bar)")
    fuel_temperature_c: float = Field(..., description="Fuel temperature (°C)")

    # Air parameters
    air_flow_rate_m3_hr: float = Field(..., description="Combustion air flow (m³/hr)")
    air_damper_position_percent: float = Field(..., ge=0, le=100, description="Air damper position (%)")
    combustion_air_temp_c: float = Field(..., description="Combustion air temperature (°C)")

    # Flue gas parameters
    flue_gas_temperature_c: float = Field(..., description="Flue gas temperature (°C)")
    stack_draft_pa: float = Field(..., description="Stack draft pressure (Pa)")

    # Emissions (from CEMS)
    o2_percent: float = Field(..., ge=0, le=21, description="O₂ in flue gas (%)")
    co2_percent: float = Field(..., ge=0, le=20, description="CO₂ in flue gas (%)")
    co_ppm: float = Field(..., ge=0, description="CO concentration (ppm)")
    nox_ppm: float = Field(..., ge=0, description="NOx concentration (ppm)")
    so2_ppm: Optional[float] = Field(None, ge=0, description="SO₂ concentration (ppm)")

    # Pressure and temperature
    combustion_chamber_pressure_bar: float = Field(..., description="Chamber pressure (bar)")
    combustion_chamber_temp_c: float = Field(..., description="Chamber temperature (°C)")

    # Heat output
    heat_output_mw: float = Field(..., description="Measured heat output (MW)")
    steam_flow_kg_hr: Optional[float] = Field(None, description="Steam flow if boiler (kg/hr)")

    # Data quality
    data_quality_percent: float = Field(..., ge=0, le=100, description="Overall data quality (%)")
    missing_tags: List[str] = Field(default_factory=list, description="Tags with missing data")
    stale_tags: List[str] = Field(default_factory=list, description="Tags with stale data")


class ValidateSensorDataInput(BaseModel):
    """Input schema for validate_sensor_data tool"""
    sensor_data: Dict[str, float] = Field(..., description="Raw sensor readings")
    validation_rules: Dict[str, Dict[str, float]] = Field(
        ...,
        description="Validation rules: {tag: {min, max, rate_of_change_limit}}"
    )
    previous_readings: Optional[Dict[str, float]] = Field(
        None,
        description="Previous readings for rate-of-change validation"
    )


class ValidateSensorDataOutput(BaseModel):
    """Output schema for validate_sensor_data tool"""
    validation_passed: bool
    validated_data: Dict[str, float] = Field(..., description="Validated sensor data")

    # Validation results
    range_violations: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Sensors outside valid range"
    )
    rate_violations: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Sensors with excessive rate of change"
    )
    quality_issues: List[str] = Field(
        default_factory=list,
        description="Data quality warnings"
    )

    # Statistics
    total_sensors: int
    valid_sensors: int
    invalid_sensors: int
    quality_score: float = Field(..., ge=0, le=1, description="Overall quality (0-1)")


class SynchronizeDataStreamsInput(BaseModel):
    """Input schema for synchronize_data_streams tool"""
    dcs_data: Dict[str, Any] = Field(..., description="DCS data stream")
    plc_data: Dict[str, Any] = Field(..., description="PLC data stream")
    cems_data: Dict[str, Any] = Field(..., description="CEMS data stream")
    timestamp_tolerance_ms: int = Field(
        50,
        description="Maximum time difference for synchronization (ms)"
    )


class SynchronizeDataStreamsOutput(BaseModel):
    """Output schema for synchronize_data_streams tool"""
    synchronized: bool
    synchronized_timestamp: datetime
    synchronized_data: Dict[str, Any] = Field(..., description="Merged synchronized data")

    # Synchronization quality
    time_skew_ms: float = Field(..., description="Maximum time skew across sources (ms)")
    sources_aligned: List[str] = Field(..., description="Successfully aligned sources")
    sources_delayed: List[str] = Field(default_factory=list, description="Delayed sources")

    # Data completeness
    completeness_percent: float = Field(..., ge=0, le=100)


# ============================================================================
# CATEGORY 2: COMBUSTION ANALYSIS TOOLS (3 tools)
# ============================================================================

class AnalyzeCombustionEfficiencyInput(BaseModel):
    """Input schema for analyze_combustion_efficiency tool"""
    fuel_type: FuelType
    fuel_flow_rate_kg_hr: float = Field(..., gt=0)
    air_flow_rate_m3_hr: float = Field(..., gt=0)
    flue_gas_temperature_c: float = Field(..., description="Stack temperature")
    ambient_temperature_c: float = Field(25.0, description="Ambient air temperature")
    o2_percent: float = Field(..., ge=0, le=21)
    co_ppm: float = Field(0, ge=0)
    fuel_heating_value_mj_kg: Optional[float] = Field(None, description="LHV (auto-lookup if None)")


class AnalyzeCombustionEfficiencyOutput(BaseModel):
    """Output schema for analyze_combustion_efficiency tool"""
    # Efficiency metrics
    gross_efficiency_percent: float = Field(..., description="Gross combustion efficiency (%)")
    net_efficiency_percent: float = Field(..., description="Net efficiency (%)")
    thermal_efficiency_percent: float = Field(..., description="Thermal efficiency (%)")

    # Air-fuel metrics
    theoretical_air_kg_kg: float = Field(..., description="Stoichiometric air requirement")
    actual_air_kg_kg: float = Field(..., description="Actual air supplied")
    excess_air_percent: float = Field(..., description="Excess air (%)")
    air_fuel_ratio: float = Field(..., description="AFR (kg air / kg fuel)")

    # Heat loss breakdown (ASME PTC 4.1)
    dry_flue_gas_loss_percent: float
    moisture_loss_h2_percent: float = Field(..., description="H₂ combustion moisture")
    moisture_loss_fuel_percent: float = Field(..., description="Fuel moisture")
    incomplete_combustion_loss_percent: float = Field(..., description="CO formation loss")
    radiation_loss_percent: float
    total_losses_percent: float

    # Performance score
    performance_score: float = Field(..., ge=0, le=100, description="Overall score (0-100)")

    # Recommendations
    efficiency_improvement_potential_percent: float
    recommendations: List[str]


class CalculateHeatOutputInput(BaseModel):
    """Input schema for calculate_heat_output tool"""
    fuel_flow_rate_kg_hr: float = Field(..., gt=0)
    fuel_heating_value_mj_kg: float = Field(..., gt=0, description="Lower heating value")
    combustion_efficiency_percent: float = Field(..., ge=0, le=100)
    steam_flow_kg_hr: Optional[float] = Field(None, description="For boilers")
    steam_enthalpy_kj_kg: Optional[float] = Field(None, description="Steam enthalpy")
    feedwater_enthalpy_kj_kg: Optional[float] = Field(None, description="Feedwater enthalpy")


class CalculateHeatOutputOutput(BaseModel):
    """Output schema for calculate_heat_output tool"""
    heat_input_mw: float = Field(..., description="Total heat input (MW)")
    heat_output_mw: float = Field(..., description="Useful heat output (MW)")
    heat_losses_mw: float = Field(..., description="Total heat losses (MW)")

    # Efficiency
    thermal_efficiency_percent: float

    # For steam systems
    evaporation_ratio: Optional[float] = Field(None, description="kg steam / kg fuel")
    specific_fuel_consumption_kg_mwh: float = Field(..., description="SFC (kg/MWh)")


class MonitorFlameStabilityInput(BaseModel):
    """Input schema for monitor_flame_stability tool"""
    flame_signal_strength_percent: float = Field(..., ge=0, le=100)
    flame_temperature_c: float = Field(..., description="Measured flame temperature")
    fuel_type: FuelType
    air_fuel_ratio: float = Field(..., gt=0)
    combustion_chamber_pressure_bar: float


class MonitorFlameStabilityOutput(BaseModel):
    """Output schema for monitor_flame_stability tool"""
    flame_present: bool
    flame_stability: str = Field(..., description="stable, unstable, oscillating, weak")

    # Flame characteristics
    adiabatic_flame_temperature_c: float = Field(..., description="Theoretical max temperature")
    actual_vs_theoretical_ratio: float = Field(..., ge=0, le=1)
    flame_quality_score: float = Field(..., ge=0, le=1, description="Quality index (0-1)")

    # Stability metrics
    oscillation_frequency_hz: Optional[float] = Field(None, description="If oscillating")
    signal_variance: float = Field(..., description="Flame signal variance")

    # Heat release
    estimated_heat_release_rate_mw: float

    # Issues and recommendations
    stability_issues: List[str] = Field(default_factory=list)
    recommendations: List[str] = Field(default_factory=list)


# ============================================================================
# CATEGORY 3: CONTROL OPTIMIZATION TOOLS (3 tools)
# ============================================================================

class OptimizeFuelAirRatioInput(BaseModel):
    """Input schema for optimize_fuel_air_ratio tool"""
    current_fuel_flow_kg_hr: float = Field(..., gt=0)
    current_air_flow_m3_hr: float = Field(..., gt=0)
    current_efficiency_percent: float = Field(..., ge=0, le=100)
    current_nox_ppm: float = Field(..., ge=0)
    current_co_ppm: float = Field(..., ge=0)

    fuel_type: FuelType
    heat_demand_mw: float = Field(..., gt=0, description="Target heat output")

    # Optimization objectives (weights sum to 1.0)
    efficiency_weight: float = Field(0.5, ge=0, le=1)
    emissions_weight: float = Field(0.3, ge=0, le=1)
    stability_weight: float = Field(0.2, ge=0, le=1)

    # Constraints
    max_nox_limit_ppm: float = Field(50.0)
    max_co_limit_ppm: float = Field(100.0)
    min_excess_air_percent: float = Field(5.0)
    max_excess_air_percent: float = Field(25.0)

    @field_validator('efficiency_weight', 'emissions_weight', 'stability_weight')
    @classmethod
    def check_weights_sum(cls, v: float, info: ValidationInfo):
        """Ensure weights sum to approximately 1.0"""
        if info.data.get('efficiency_weight') is not None and info.data.get('emissions_weight') is not None:
            total = info.data['efficiency_weight'] + info.data['emissions_weight'] + v
            if abs(total - 1.0) > 0.01:
                raise ValueError(f"Weights must sum to 1.0, got {total}")
        return v


class OptimizeFuelAirRatioOutput(BaseModel):
    """Output schema for optimize_fuel_air_ratio tool"""
    # Optimal setpoints
    optimal_fuel_flow_kg_hr: float
    optimal_air_flow_m3_hr: float
    optimal_air_fuel_ratio: float
    optimal_excess_air_percent: float

    # Predicted performance
    predicted_efficiency_percent: float
    predicted_nox_ppm: float
    predicted_co_ppm: float
    predicted_heat_output_mw: float

    # Improvements
    efficiency_improvement_percent: float
    nox_reduction_percent: float
    co_reduction_percent: float
    fuel_savings_kg_hr: float
    fuel_savings_usd_hr: float = Field(..., description="Estimated cost savings")

    # Optimization quality
    optimization_score: float = Field(..., ge=0, le=1, description="Multi-objective score")
    convergence_status: str = Field(..., description="converged, max_iterations, constrained")
    iterations: int
    confidence_score: float = Field(..., ge=0, le=1)


class CalculatePIDSetpointsInput(BaseModel):
    """Input schema for calculate_pid_setpoints tool"""
    # Process variables
    current_heat_output_mw: float
    heat_setpoint_mw: float
    current_o2_percent: float
    o2_setpoint_percent: float

    # PID parameters (primary loop: heat output control)
    primary_kp: float = Field(1.2, description="Proportional gain")
    primary_ki: float = Field(0.5, description="Integral gain")
    primary_kd: float = Field(0.1, description="Derivative gain")

    # PID parameters (secondary loop: air-fuel ratio control)
    secondary_kp: float = Field(0.8, description="Proportional gain")
    secondary_ki: float = Field(0.3, description="Integral gain")
    secondary_kd: float = Field(0.05, description="Derivative gain")

    # Control state
    primary_integral_state: float = Field(0, description="Accumulated integral error")
    primary_previous_error: float = Field(0, description="Previous error for derivative")
    secondary_integral_state: float = Field(0)
    secondary_previous_error: float = Field(0)

    # Constraints
    max_fuel_rate_kg_hr: float = Field(..., gt=0)
    min_fuel_rate_kg_hr: float = Field(..., gt=0)
    max_air_damper_percent: float = Field(100, ge=0, le=100)
    min_air_damper_percent: float = Field(0, ge=0, le=100)

    # Control parameters
    sampling_time_sec: float = Field(0.1, description="Control loop cycle time")
    anti_windup_enabled: bool = Field(True, description="Enable integral anti-windup")


class CalculatePIDSetpointsOutput(BaseModel):
    """Output schema for calculate_pid_setpoints tool"""
    # Primary loop (heat output → fuel flow)
    primary_error: float = Field(..., description="Heat setpoint error (MW)")
    primary_control_output: float = Field(..., description="Fuel flow adjustment (kg/hr)")
    primary_proportional_term: float
    primary_integral_term: float
    primary_derivative_term: float
    primary_new_integral_state: float

    # Secondary loop (O₂ → air damper)
    secondary_error: float = Field(..., description="O₂ setpoint error (%)")
    secondary_control_output: float = Field(..., description="Air damper adjustment (%)")
    secondary_proportional_term: float
    secondary_integral_term: float
    secondary_derivative_term: float
    secondary_new_integral_state: float

    # Recommended setpoints
    recommended_fuel_flow_kg_hr: float
    recommended_air_damper_percent: float

    # Control quality
    control_stability_index: float = Field(..., ge=0, le=1, description="Stability (0-1)")
    setpoint_tracking_error_percent: float

    # Anti-windup status
    primary_anti_windup_active: bool
    secondary_anti_windup_active: bool


class AdjustBurnerSettingsInput(BaseModel):
    """Input schema for adjust_burner_settings tool"""
    new_fuel_flow_kg_hr: float = Field(..., gt=0)
    new_air_damper_percent: float = Field(..., ge=0, le=100)

    # Rate limiting
    max_fuel_ramp_rate_kg_hr_per_min: float = Field(100, description="Max fuel change rate")
    max_damper_ramp_rate_percent_per_min: float = Field(10, description="Max damper change rate")

    # Current state
    current_fuel_flow_kg_hr: float = Field(..., gt=0)
    current_air_damper_percent: float = Field(..., ge=0, le=100)

    # Safety
    perform_safety_check: bool = Field(True)
    allow_override: bool = Field(False, description="Override safety interlocks (admin only)")


class AdjustBurnerSettingsOutput(BaseModel):
    """Output schema for adjust_burner_settings tool"""
    adjustment_status: str = Field(..., description="success, rate_limited, safety_blocked, failed")

    # Executed changes
    actual_fuel_flow_kg_hr: float
    actual_air_damper_percent: float

    # Rate limiting
    fuel_rate_limited: bool
    damper_rate_limited: bool
    ramped_fuel_setpoint_kg_hr: float = Field(..., description="Rate-limited setpoint")
    ramped_damper_setpoint_percent: float = Field(..., description="Rate-limited setpoint")

    # Safety
    safety_checks_passed: bool
    safety_violations: List[str] = Field(default_factory=list)

    # Execution
    execution_time_ms: float
    setpoint_error_percent: float = Field(..., description="Deviation from commanded")

    message: str


# ============================================================================
# CATEGORY 4: COMMAND EXECUTION TOOLS (2 tools)
# ============================================================================

class WriteControlCommandsInput(BaseModel):
    """Input schema for write_control_commands tool"""
    unit_id: str

    # Control commands
    fuel_valve_position_percent: Optional[float] = Field(None, ge=0, le=100)
    air_damper_position_percent: Optional[float] = Field(None, ge=0, le=100)
    recirculation_damper_percent: Optional[float] = Field(None, ge=0, le=100)

    # DCS/PLC connection
    dcs_host: str = Field(..., description="DCS IP address")
    dcs_port: int = Field(502, description="Modbus TCP port")
    plc_endpoint: Optional[str] = Field(None, description="OPC UA endpoint")

    # Execution parameters
    write_timeout_ms: int = Field(1000, ge=100, le=5000)
    verify_write: bool = Field(True, description="Read back to verify")
    retry_count: int = Field(3, ge=1, le=5)

    # Safety
    safety_interlock_override: bool = Field(False, description="DANGEROUS: Override safety")


class WriteControlCommandsOutput(BaseModel):
    """Output schema for write_control_commands tool"""
    write_status: str = Field(..., description="success, partial, failed, safety_blocked")

    # Write results
    fuel_valve_write_success: bool
    air_damper_write_success: bool
    recirculation_damper_write_success: bool

    # Verification (if enabled)
    fuel_valve_readback_percent: Optional[float] = None
    air_damper_readback_percent: Optional[float] = None
    recirculation_damper_readback_percent: Optional[float] = None

    # Timing
    total_write_time_ms: float
    retry_attempts: int

    # Errors
    errors: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)


class ValidateSafetyInterlocksInput(BaseModel):
    """Input schema for validate_safety_interlocks tool"""
    unit_id: str

    # Sensor readings for safety checks
    flame_signal_strength_percent: float = Field(..., ge=0, le=100)
    combustion_chamber_pressure_bar: float
    combustion_chamber_temp_c: float
    o2_percent: float = Field(..., ge=0, le=21)
    fuel_pressure_bar: float

    # Safety limits (configured per unit)
    safety_limits: Dict[str, Dict[str, float]] = Field(
        ...,
        description="Safety limits: {parameter: {min, max}}"
    )

    # Interlock logic
    require_flame_present: bool = Field(True)
    require_minimum_o2: bool = Field(True)
    check_pressure_limits: bool = Field(True)
    check_temperature_limits: bool = Field(True)


class ValidateSafetyInterlocksOutput(BaseModel):
    """Output schema for validate_safety_interlocks tool"""
    safety_status: SafetyStatus
    all_interlocks_ok: bool

    # Individual interlock status
    flame_ok: bool
    pressure_ok: bool
    temperature_ok: bool
    o2_ok: bool
    fuel_pressure_ok: bool

    # Violations
    active_violations: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Active safety violations"
    )

    # Actions
    emergency_shutdown_required: bool
    alarm_conditions: List[str] = Field(default_factory=list)

    # Recommendations
    corrective_actions: List[str] = Field(default_factory=list)


# ============================================================================
# CATEGORY 5: SAFETY & AUDIT TOOLS (2 tools)
# ============================================================================

class GenerateControlReportInput(BaseModel):
    """Input schema for generate_control_report tool"""
    job_id: str = Field(..., description="Control job identifier")
    report_type: str = Field(
        ...,
        description="performance, compliance, safety, audit"
    )
    start_time: datetime
    end_time: datetime

    # Report options
    include_charts: bool = Field(True)
    include_raw_data: bool = Field(False)
    format: str = Field("pdf", description="pdf, csv, json, xlsx")

    # Filtering
    min_efficiency_threshold: Optional[float] = Field(None, description="Filter low efficiency periods")
    max_emissions_threshold: Optional[float] = Field(None, description="Filter high emissions periods")

    @field_validator('report_type')
    @classmethod
    def validate_report_type(cls, v: str) -> str:
        valid_types = ['performance', 'compliance', 'safety', 'audit']
        if v not in valid_types:
            raise ValueError(f"report_type must be one of {valid_types}")
        return v


class GenerateControlReportOutput(BaseModel):
    """Output schema for generate_control_report tool"""
    report_id: str
    report_type: str
    generation_timestamp: datetime = Field(default_factory=datetime.utcnow)

    # Report summary
    total_runtime_hours: float
    average_efficiency_percent: float
    average_heat_output_mw: float
    total_fuel_consumed_kg: float
    total_co2_emissions_kg: float

    # Performance metrics
    efficiency_range: Tuple[float, float] = Field(..., description="(min, max) efficiency")
    nox_range: Tuple[float, float] = Field(..., description="(min, max) NOx ppm")
    co_range: Tuple[float, float] = Field(..., description="(min, max) CO ppm")

    # Compliance
    compliance_status: ComplianceStatus
    compliance_violations: List[Dict[str, Any]] = Field(default_factory=list)

    # Safety events
    safety_events_count: int
    emergency_stops_count: int

    # File output
    report_file_path: Optional[str] = Field(None, description="Path to generated report file")
    report_file_size_kb: Optional[float] = None

    # Metadata
    data_points_analyzed: int
    data_completeness_percent: float


class TrackProvenanceInput(BaseModel):
    """Input schema for track_provenance tool"""
    job_id: str
    agent_name: str = Field(..., description="Agent that performed calculation")
    tool_name: str = Field(..., description="Tool that was executed")

    # Calculation details
    inputs: Dict[str, Any] = Field(..., description="All input parameters")
    outputs: Dict[str, Any] = Field(..., description="Calculation results")
    intermediate_steps: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Intermediate calculation steps for audit"
    )

    # Data sources
    data_sources: List[str] = Field(..., description="Source of input data (dcs, plc, cems)")
    data_timestamps: Dict[str, datetime] = Field(..., description="Timestamp per data source")

    # Calculation metadata
    calculation_method: str = Field(..., description="Algorithm/formula used")
    standards_applied: List[str] = Field(
        default_factory=list,
        description="Standards (ASME PTC 4.1, EPA 40 CFR 60)"
    )


class TrackProvenanceOutput(BaseModel):
    """Output schema for track_provenance tool"""
    provenance_id: str = Field(..., description="Unique provenance record ID")
    provenance_hash: str = Field(..., description="SHA-256 hash of entire calculation")

    # Audit trail
    calculation_reproducible: bool = Field(
        True,
        description="Can this calculation be reproduced bit-for-bit?"
    )

    # Hashing details
    inputs_hash: str = Field(..., description="SHA-256 of inputs")
    outputs_hash: str = Field(..., description="SHA-256 of outputs")
    tool_version: str = Field(..., description="Tool version used")

    # Compliance
    audit_trail_complete: bool
    regulatory_compliant: bool

    # Storage
    stored_in_database: bool
    stored_in_object_storage: bool

    # Chain of custody
    previous_provenance_hash: Optional[str] = Field(
        None,
        description="Hash of previous calculation (blockchain-like chain)"
    )

    timestamp: datetime = Field(default_factory=datetime.utcnow)


# ============================================================================
# TOOL REGISTRY
# ============================================================================

TOOL_REGISTRY = {
    # Category 1: Data Acquisition
    "read_combustion_data": {
        "input": ReadCombustionDataInput,
        "output": ReadCombustionDataOutput,
        "description": "Read real-time combustion data from DCS, PLC, CEMS at 100 Hz",
        "category": "data_acquisition",
        "criticality": "high"
    },
    "validate_sensor_data": {
        "input": ValidateSensorDataInput,
        "output": ValidateSensorDataOutput,
        "description": "Validate sensor data quality with range and rate-of-change checks",
        "category": "data_acquisition",
        "criticality": "high"
    },
    "synchronize_data_streams": {
        "input": SynchronizeDataStreamsInput,
        "output": SynchronizeDataStreamsOutput,
        "description": "Synchronize DCS, PLC, CEMS data streams with timestamp alignment",
        "category": "data_acquisition",
        "criticality": "medium"
    },

    # Category 2: Combustion Analysis
    "analyze_combustion_efficiency": {
        "input": AnalyzeCombustionEfficiencyInput,
        "output": AnalyzeCombustionEfficiencyOutput,
        "description": "Calculate combustion efficiency using ASME PTC 4.1 methodology",
        "category": "combustion_analysis",
        "criticality": "high"
    },
    "calculate_heat_output": {
        "input": CalculateHeatOutputInput,
        "output": CalculateHeatOutputOutput,
        "description": "Calculate heat output and thermal efficiency",
        "category": "combustion_analysis",
        "criticality": "high"
    },
    "monitor_flame_stability": {
        "input": MonitorFlameStabilityInput,
        "output": MonitorFlameStabilityOutput,
        "description": "Monitor flame characteristics and stability metrics",
        "category": "combustion_analysis",
        "criticality": "high"
    },

    # Category 3: Control Optimization
    "optimize_fuel_air_ratio": {
        "input": OptimizeFuelAirRatioInput,
        "output": OptimizeFuelAirRatioOutput,
        "description": "Multi-objective optimization of fuel-air ratio for efficiency and emissions",
        "category": "control_optimization",
        "criticality": "high"
    },
    "calculate_pid_setpoints": {
        "input": CalculatePIDSetpointsInput,
        "output": CalculatePIDSetpointsOutput,
        "description": "Calculate PID controller setpoints for cascade control loops",
        "category": "control_optimization",
        "criticality": "high"
    },
    "adjust_burner_settings": {
        "input": AdjustBurnerSettingsInput,
        "output": AdjustBurnerSettingsOutput,
        "description": "Adjust burner fuel and air settings with rate limiting and safety",
        "category": "control_optimization",
        "criticality": "high"
    },

    # Category 4: Command Execution
    "write_control_commands": {
        "input": WriteControlCommandsInput,
        "output": WriteControlCommandsOutput,
        "description": "Write control commands to DCS/PLC with verification and retry",
        "category": "command_execution",
        "criticality": "critical"
    },
    "validate_safety_interlocks": {
        "input": ValidateSafetyInterlocksInput,
        "output": ValidateSafetyInterlocksOutput,
        "description": "Validate safety interlocks before command execution",
        "category": "command_execution",
        "criticality": "critical"
    },

    # Category 5: Safety & Audit
    "generate_control_report": {
        "input": GenerateControlReportInput,
        "output": GenerateControlReportOutput,
        "description": "Generate performance, compliance, or audit reports",
        "category": "safety_audit",
        "criticality": "medium"
    },
    "track_provenance": {
        "input": TrackProvenanceInput,
        "output": TrackProvenanceOutput,
        "description": "Track provenance of all calculations with SHA-256 hashing",
        "category": "safety_audit",
        "criticality": "high"
    }
}


def get_tool_schema(tool_name: str) -> Dict[str, Any]:
    """
    Get the input/output schema for a tool.

    Args:
        tool_name: Name of the tool

    Returns:
        Dictionary with input/output schemas

    Raises:
        ValueError: If tool not found
    """
    if tool_name not in TOOL_REGISTRY:
        raise ValueError(f"Tool '{tool_name}' not found in registry")

    tool_info = TOOL_REGISTRY[tool_name]
    return {
        "input_schema": tool_info["input"].schema(),
        "output_schema": tool_info["output"].schema(),
        "description": tool_info["description"],
        "category": tool_info["category"],
        "criticality": tool_info["criticality"]
    }


def list_tools_by_category(category: str = None) -> List[str]:
    """
    List all tools, optionally filtered by category.

    Args:
        category: Optional category filter (data_acquisition, combustion_analysis, etc.)

    Returns:
        List of tool names
    """
    if category is None:
        return list(TOOL_REGISTRY.keys())

    return [
        name for name, info in TOOL_REGISTRY.items()
        if info["category"] == category
    ]


# Export all schemas for external use
__all__ = [
    # Enums
    "FuelType",
    "ControlMode",
    "SafetyStatus",
    "ComplianceStatus",

    # Data Acquisition Tools
    "ReadCombustionDataInput",
    "ReadCombustionDataOutput",
    "ValidateSensorDataInput",
    "ValidateSensorDataOutput",
    "SynchronizeDataStreamsInput",
    "SynchronizeDataStreamsOutput",

    # Combustion Analysis Tools
    "AnalyzeCombustionEfficiencyInput",
    "AnalyzeCombustionEfficiencyOutput",
    "CalculateHeatOutputInput",
    "CalculateHeatOutputOutput",
    "MonitorFlameStabilityInput",
    "MonitorFlameStabilityOutput",

    # Control Optimization Tools
    "OptimizeFuelAirRatioInput",
    "OptimizeFuelAirRatioOutput",
    "CalculatePIDSetpointsInput",
    "CalculatePIDSetpointsOutput",
    "AdjustBurnerSettingsInput",
    "AdjustBurnerSettingsOutput",

    # Command Execution Tools
    "WriteControlCommandsInput",
    "WriteControlCommandsOutput",
    "ValidateSafetyInterlocksInput",
    "ValidateSafetyInterlocksOutput",

    # Safety & Audit Tools
    "GenerateControlReportInput",
    "GenerateControlReportOutput",
    "TrackProvenanceInput",
    "TrackProvenanceOutput",

    # Utility functions
    "get_tool_schema",
    "list_tools_by_category",
    "TOOL_REGISTRY"
]
