# -*- coding: utf-8 -*-
"""
GL-005 CombustionControlAgent - Tool Definitions

Tool definitions for combustion control agent capabilities.
All tools follow zero-hallucination design with deterministic control algorithms.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator, ValidationInfo


class ReadCombustionStateInput(BaseModel):
    """Input schema for read_combustion_state tool"""
    include_analyzer_data: bool = Field(True, description="Include combustion analyzer measurements")
    include_derived_values: bool = Field(True, description="Include calculated heat output, efficiency")
    timeout_ms: int = Field(50, ge=10, le=1000, description="Read timeout (ms)")


class ReadCombustionStateOutput(BaseModel):
    """Output schema for read_combustion_state tool"""
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    fuel_flow: float = Field(..., description="Fuel flow rate (kg/hr or m3/hr)")
    air_flow: float = Field(..., description="Air flow rate (m3/hr)")
    air_fuel_ratio: float = Field(..., description="Air-fuel ratio")
    furnace_temperature: float = Field(..., description="Furnace temperature (°C)")
    flue_gas_temperature: float = Field(..., description="Flue gas temperature (°C)")
    fuel_pressure: float = Field(..., description="Fuel pressure (kPa)")
    air_pressure: float = Field(..., description="Air pressure (kPa)")
    o2_percent: float = Field(..., description="O2 in flue gas (%)")
    co_ppm: Optional[float] = Field(None, description="CO concentration (ppm)")
    heat_output_kw: Optional[float] = Field(None, description="Heat output (kW)")
    thermal_efficiency: Optional[float] = Field(None, description="Thermal efficiency (%)")
    read_time_ms: float = Field(..., description="Actual read time (ms)")


class AnalyzeStabilityInput(BaseModel):
    """Input schema for analyze_stability tool"""
    window_size: int = Field(60, ge=10, le=1000, description="Number of samples to analyze")
    target_heat_output: float = Field(..., gt=0, description="Target heat output (kW)")
    tolerance_percent: float = Field(2.0, ge=0.1, le=10.0, description="Acceptable tolerance (%)")
    detect_oscillations: bool = Field(True, description="Perform oscillation detection")


class AnalyzeStabilityOutput(BaseModel):
    """Output schema for analyze_stability tool"""
    overall_stability_score: float = Field(..., ge=0, le=100, description="Composite stability (0-100)")
    stability_rating: str = Field(..., description="excellent, good, fair, poor, unstable")
    heat_output_stability_index: float = Field(..., ge=0, le=1)
    heat_output_variance: float = Field(..., description="Variance in heat output (kW²)")
    heat_output_cv: float = Field(..., description="Coefficient of variation (%)")
    furnace_temp_stability: float = Field(..., ge=0, le=1)
    o2_stability: float = Field(..., ge=0, le=1)
    oscillation_detected: bool
    oscillation_frequency_hz: Optional[float] = Field(None)
    oscillation_amplitude: Optional[float] = Field(None)
    recommendations: List[str] = Field(default_factory=list)


class OptimizeFuelAirRatioInput(BaseModel):
    """Input schema for optimize_fuel_air_ratio tool"""
    current_fuel_flow: float = Field(..., gt=0, description="Current fuel flow")
    current_air_flow: float = Field(..., gt=0, description="Current air flow")
    heat_demand_kw: float = Field(..., gt=0, description="Target heat output (kW)")
    current_o2_percent: float = Field(..., ge=0, le=21, description="Current O2 (%)")
    current_efficiency: float = Field(..., ge=0, le=100, description="Current efficiency (%)")
    fuel_type: str = Field(..., description="Fuel type")
    enable_o2_trim: bool = Field(True, description="Enable O2 trim correction")

    @field_validator('fuel_type')
    @classmethod
    def validate_fuel_type(cls, v: str) -> str:
        valid_types = ['natural_gas', 'fuel_oil', 'coal', 'biomass', 'propane', 'lng']
        if v not in valid_types:
            raise ValueError(f"Fuel type must be one of {valid_types}")
        return v


class OptimizeFuelAirRatioOutput(BaseModel):
    """Output schema for optimize_fuel_air_ratio tool"""
    optimal_fuel_flow: float = Field(..., description="Optimal fuel flow")
    optimal_air_flow: float = Field(..., description="Optimal air flow")
    optimal_air_fuel_ratio: float = Field(..., description="Optimal AFR")
    optimal_excess_air_percent: float = Field(..., description="Optimal excess air (%)")
    predicted_heat_output_kw: float = Field(..., description="Predicted heat output")
    predicted_efficiency: float = Field(..., description="Predicted efficiency (%)")
    predicted_o2_percent: float = Field(..., description="Predicted O2 (%)")
    fuel_savings_percent: float = Field(..., description="Fuel savings vs current (%)")
    o2_trim_correction: float = Field(..., description="O2 trim adjustment applied")


class CalculatePIDControlInput(BaseModel):
    """Input schema for calculate_pid_control tool"""
    setpoint: float = Field(..., description="Control setpoint")
    process_variable: float = Field(..., description="Current measured value")
    kp: float = Field(..., description="Proportional gain")
    ki: float = Field(..., description="Integral gain")
    kd: float = Field(..., description="Derivative gain")
    sample_time_s: float = Field(..., gt=0, description="Sample time (seconds)")
    output_min: float = Field(..., description="Minimum output limit")
    output_max: float = Field(..., description="Maximum output limit")
    reset_integral: bool = Field(False, description="Reset integral term")


class CalculatePIDControlOutput(BaseModel):
    """Output schema for calculate_pid_control tool"""
    control_output: float = Field(..., description="PID control output")
    proportional_term: float = Field(..., description="P term contribution")
    integral_term: float = Field(..., description="I term contribution")
    derivative_term: float = Field(..., description="D term contribution")
    error: float = Field(..., description="Current error (SP - PV)")
    output_saturated: bool = Field(..., description="Output hit limits")
    integral_windup_prevented: bool = Field(..., description="Anti-windup active")


class AdjustBurnerSettingsInput(BaseModel):
    """Input schema for adjust_burner_settings tool"""
    fuel_flow_setpoint: float = Field(..., gt=0, description="Fuel flow setpoint")
    air_flow_setpoint: float = Field(..., gt=0, description="Air flow setpoint")
    ramp_rate_percent_per_sec: float = Field(5.0, ge=0.1, le=50.0, description="Ramp rate (%/s)")
    verify_safety_interlocks: bool = Field(True, description="Check interlocks before adjustment")
    write_to_backup_plc: bool = Field(True, description="Also write to backup PLC")


class AdjustBurnerSettingsOutput(BaseModel):
    """Output schema for adjust_burner_settings tool"""
    success: bool
    fuel_setpoint_written: float = Field(..., description="Actual fuel setpoint written")
    air_setpoint_written: float = Field(..., description="Actual air setpoint written")
    fuel_valve_position: float = Field(..., ge=0, le=100, description="Fuel valve position (%)")
    air_damper_position: float = Field(..., ge=0, le=100, description="Air damper position (%)")
    write_time_ms: float = Field(..., description="Time to write setpoints (ms)")
    safety_interlocks_ok: bool
    message: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class CheckSafetyInterlocksInput(BaseModel):
    """Input schema for check_safety_interlocks tool"""
    check_dcs: bool = Field(True, description="Check DCS interlocks")
    check_plc: bool = Field(True, description="Check PLC interlocks")
    fail_safe_mode: bool = Field(True, description="Use most restrictive interlock status")


class CheckSafetyInterlocksOutput(BaseModel):
    """Output schema for check_safety_interlocks tool"""
    all_interlocks_satisfied: bool
    flame_present: bool
    fuel_pressure_ok: bool
    air_pressure_ok: bool
    furnace_temp_ok: bool
    furnace_pressure_ok: bool
    purge_complete: bool
    emergency_stop_clear: bool
    high_fire_lockout_clear: bool
    low_fire_lockout_clear: bool
    failed_interlocks: List[str] = Field(default_factory=list)
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class CalculateHeatOutputInput(BaseModel):
    """Input schema for calculate_heat_output tool"""
    fuel_flow: float = Field(..., gt=0, description="Fuel flow rate")
    fuel_type: str
    fuel_lhv_mj_per_kg: float = Field(..., gt=0, description="Lower heating value (MJ/kg)")
    efficiency_percent: float = Field(..., ge=0, le=100, description="Combustion efficiency (%)")


class CalculateHeatOutputOutput(BaseModel):
    """Output schema for calculate_heat_output tool"""
    heat_output_kw: float = Field(..., description="Heat output (kW)")
    heat_output_mw: float = Field(..., description="Heat output (MW)")
    heat_output_btu_per_hr: float = Field(..., description="Heat output (BTU/hr)")
    fuel_energy_input_kw: float = Field(..., description="Total fuel energy input (kW)")
    heat_loss_kw: float = Field(..., description="Heat losses (kW)")
    thermal_efficiency_check: float = Field(..., description="Efficiency verification (%)")


class MonitorControlPerformanceInput(BaseModel):
    """Input schema for monitor_control_performance tool"""
    time_window_seconds: int = Field(300, ge=60, le=3600, description="Time window for analysis")
    include_pid_tuning_recommendations: bool = Field(True, description="Include PID tuning suggestions")


class MonitorControlPerformanceOutput(BaseModel):
    """Output schema for monitor_control_performance tool"""
    avg_cycle_time_ms: float = Field(..., description="Average control cycle time (ms)")
    max_cycle_time_ms: float = Field(..., description="Maximum cycle time (ms)")
    cycles_exceeding_target: int = Field(..., description="Cycles slower than target")
    control_errors: int = Field(..., description="Number of control errors")
    avg_settling_time_s: float = Field(..., description="Average settling time (s)")
    avg_overshoot_percent: float = Field(..., description="Average overshoot (%)")
    avg_steady_state_error: float = Field(..., description="Average steady-state error")
    control_quality_score: float = Field(..., ge=0, le=100, description="Overall control quality (0-100)")
    pid_tuning_recommendations: Optional[Dict[str, Any]] = Field(None)


class PredictFlameStabilityInput(BaseModel):
    """Input schema for predict_flame_stability tool"""
    air_fuel_ratio: float = Field(..., gt=0)
    fuel_pressure: float = Field(..., gt=0, description="Fuel pressure (kPa)")
    air_pressure: float = Field(..., gt=0, description="Air pressure (kPa)")
    flame_temperature: Optional[float] = Field(None, description="Flame temperature (°C)")
    fuel_type: str


class PredictFlameStabilityOutput(BaseModel):
    """Output schema for predict_flame_stability tool"""
    flame_stability_predicted: str = Field(..., description="stable, marginal, unstable")
    stability_confidence: float = Field(..., ge=0, le=1, description="Prediction confidence")
    flammability_margin: float = Field(..., description="Distance from flammability limits")
    blowoff_risk: str = Field(..., description="low, medium, high")
    flashback_risk: str = Field(..., description="low, medium, high")
    combustion_quality: str = Field(..., description="excellent, good, fair, poor")
    recommendations: List[str]


class DetectCombustionAnomaliesInput(BaseModel):
    """Input schema for detect_combustion_anomalies tool"""
    current_state: Dict[str, float] = Field(..., description="Current combustion parameters")
    baseline_state: Dict[str, float] = Field(..., description="Normal operating baseline")
    sensitivity: float = Field(1.0, ge=0.1, le=5.0, description="Anomaly detection sensitivity")


class DetectCombustionAnomaliesOutput(BaseModel):
    """Output schema for detect_combustion_anomalies tool"""
    anomalies_detected: bool
    anomaly_count: int
    anomaly_details: List[Dict[str, Any]] = Field(default_factory=list)
    severity: str = Field(..., description="none, minor, moderate, severe, critical")
    probable_causes: List[str]
    recommended_actions: List[str]
    requires_immediate_action: bool


class TunePIDParametersInput(BaseModel):
    """Input schema for tune_pid_parameters tool"""
    process_variable_history: List[float] = Field(..., min_items=20, description="Recent PV values")
    setpoint_history: List[float] = Field(..., min_items=20, description="Recent SP values")
    current_kp: float
    current_ki: float
    current_kd: float
    tuning_method: str = Field("ziegler_nichols", description="Tuning method to use")
    aggressive_tuning: bool = Field(False, description="Use aggressive (fast) tuning")

    @field_validator('tuning_method')
    @classmethod
    def validate_tuning_method(cls, v: str) -> str:
        valid_methods = ['ziegler_nichols', 'cohen_coon', 'lambda_tuning', 'imc', 'auto']
        if v not in valid_methods:
            raise ValueError(f"Tuning method must be one of {valid_methods}")
        return v


class TunePIDParametersOutput(BaseModel):
    """Output schema for tune_pid_parameters tool"""
    recommended_kp: float
    recommended_ki: float
    recommended_kd: float
    tuning_method_used: str
    expected_settling_time_s: float
    expected_overshoot_percent: float
    expected_rise_time_s: float
    stability_margin_db: float = Field(..., description="Gain margin (dB)")
    phase_margin_deg: float = Field(..., description="Phase margin (degrees)")
    improvement_expected_percent: float


class CalculateO2TrimCorrectionInput(BaseModel):
    """Input schema for calculate_o2_trim_correction tool"""
    current_o2_percent: float = Field(..., ge=0, le=21)
    target_o2_percent: float = Field(..., ge=0, le=21)
    current_air_flow: float = Field(..., gt=0)
    trim_kp: float = Field(..., description="O2 trim proportional gain")
    trim_ki: float = Field(..., description="O2 trim integral gain")
    max_trim_adjustment: float = Field(..., gt=0, description="Max trim correction")


class CalculateO2TrimCorrectionOutput(BaseModel):
    """Output schema for calculate_o2_trim_correction tool"""
    trim_correction: float = Field(..., description="Air flow correction (m3/hr)")
    trim_correction_percent: float = Field(..., description="Correction as % of current flow")
    new_air_flow_setpoint: float = Field(..., description="Adjusted air flow setpoint")
    o2_error: float = Field(..., description="Current O2 error")
    within_trim_limits: bool = Field(..., description="Correction within max limits")


# Tool registry
TOOL_REGISTRY = {
    'read_combustion_state': {
        'name': 'read_combustion_state',
        'description': 'Read real-time combustion state from DCS/PLC/analyzers with <50ms target latency',
        'input_schema': ReadCombustionStateInput,
        'output_schema': ReadCombustionStateOutput,
        'deterministic': False,  # Real-time sensor data
        'zero_hallucination': True
    },
    'analyze_stability': {
        'name': 'analyze_stability',
        'description': 'Analyze combustion stability from time-series data with oscillation detection',
        'input_schema': AnalyzeStabilityInput,
        'output_schema': AnalyzeStabilityOutput,
        'deterministic': True,
        'zero_hallucination': True
    },
    'optimize_fuel_air_ratio': {
        'name': 'optimize_fuel_air_ratio',
        'description': 'Optimize fuel and air flows for target heat output with O2 trim',
        'input_schema': OptimizeFuelAirRatioInput,
        'output_schema': OptimizeFuelAirRatioOutput,
        'deterministic': True,
        'zero_hallucination': True
    },
    'calculate_pid_control': {
        'name': 'calculate_pid_control',
        'description': 'Calculate PID control output with anti-windup and output limiting',
        'input_schema': CalculatePIDControlInput,
        'output_schema': CalculatePIDControlOutput,
        'deterministic': True,
        'zero_hallucination': True
    },
    'adjust_burner_settings': {
        'name': 'adjust_burner_settings',
        'description': 'Write fuel and air setpoints to DCS/PLC with safety verification',
        'input_schema': AdjustBurnerSettingsInput,
        'output_schema': AdjustBurnerSettingsOutput,
        'deterministic': False,  # Interacts with physical equipment
        'zero_hallucination': True
    },
    'check_safety_interlocks': {
        'name': 'check_safety_interlocks',
        'description': 'Verify all safety interlocks are satisfied before control actions',
        'input_schema': CheckSafetyInterlocksInput,
        'output_schema': CheckSafetyInterlocksOutput,
        'deterministic': False,  # Real-time interlock status
        'zero_hallucination': True
    },
    'calculate_heat_output': {
        'name': 'calculate_heat_output',
        'description': 'Calculate heat output from fuel flow and efficiency',
        'input_schema': CalculateHeatOutputInput,
        'output_schema': CalculateHeatOutputOutput,
        'deterministic': True,
        'zero_hallucination': True
    },
    'monitor_control_performance': {
        'name': 'monitor_control_performance',
        'description': 'Monitor control loop performance and provide tuning recommendations',
        'input_schema': MonitorControlPerformanceInput,
        'output_schema': MonitorControlPerformanceOutput,
        'deterministic': True,
        'zero_hallucination': True
    },
    'predict_flame_stability': {
        'name': 'predict_flame_stability',
        'description': 'Predict flame stability based on operating conditions',
        'input_schema': PredictFlameStabilityInput,
        'output_schema': PredictFlameStabilityOutput,
        'deterministic': True,
        'zero_hallucination': True
    },
    'detect_combustion_anomalies': {
        'name': 'detect_combustion_anomalies',
        'description': 'Detect anomalies in combustion process compared to baseline',
        'input_schema': DetectCombustionAnomaliesInput,
        'output_schema': DetectCombustionAnomaliesOutput,
        'deterministic': True,
        'zero_hallucination': True
    },
    'tune_pid_parameters': {
        'name': 'tune_pid_parameters',
        'description': 'Auto-tune PID parameters using process response data',
        'input_schema': TunePIDParametersInput,
        'output_schema': TunePIDParametersOutput,
        'deterministic': True,
        'zero_hallucination': True
    },
    'calculate_o2_trim_correction': {
        'name': 'calculate_o2_trim_correction',
        'description': 'Calculate O2 trim correction for air flow optimization',
        'input_schema': CalculateO2TrimCorrectionInput,
        'output_schema': CalculateO2TrimCorrectionOutput,
        'deterministic': True,
        'zero_hallucination': True
    }
}


def get_tool(tool_name: str) -> Dict[str, Any]:
    """
    Get tool definition by name

    Args:
        tool_name: Name of the tool

    Returns:
        Tool definition dict

    Raises:
        KeyError: If tool not found
    """
    if tool_name not in TOOL_REGISTRY:
        raise KeyError(f"Tool '{tool_name}' not found in registry")

    return TOOL_REGISTRY[tool_name]


def list_tools() -> List[str]:
    """Get list of all available tool names"""
    return list(TOOL_REGISTRY.keys())


def get_tool_schema(tool_name: str) -> Dict[str, Any]:
    """
    Get JSON schema for a tool

    Args:
        tool_name: Name of the tool

    Returns:
        Dict with input and output schemas
    """
    tool = get_tool(tool_name)

    return {
        'name': tool['name'],
        'description': tool['description'],
        'input_schema': tool['input_schema'].schema(),
        'output_schema': tool['output_schema'].schema(),
        'deterministic': tool['deterministic'],
        'zero_hallucination': tool['zero_hallucination']
    }


def validate_tool_input(tool_name: str, input_data: Dict[str, Any]) -> Any:
    """
    Validate input data for a tool

    Args:
        tool_name: Name of the tool
        input_data: Input data to validate

    Returns:
        Validated input model instance

    Raises:
        ValidationError: If input is invalid
    """
    tool = get_tool(tool_name)
    return tool['input_schema'](**input_data)


def validate_tool_output(tool_name: str, output_data: Dict[str, Any]) -> Any:
    """
    Validate output data for a tool

    Args:
        tool_name: Name of the tool
        output_data: Output data to validate

    Returns:
        Validated output model instance

    Raises:
        ValidationError: If output is invalid
    """
    tool = get_tool(tool_name)
    return tool['output_schema'](**output_data)


def get_all_schemas() -> Dict[str, Dict[str, Any]]:
    """
    Get all tool schemas

    Returns:
        Dict mapping tool names to their schemas
    """
    return {name: get_tool_schema(name) for name in list_tools()}
