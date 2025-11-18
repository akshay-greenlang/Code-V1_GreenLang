"""
GL-004 BurnerOptimizationAgent - Tool Definitions

Tool definitions for burner optimization agent capabilities.
All tools follow zero-hallucination design with physics-based calculations.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, validator


class AnalyzeCombustionEfficiencyInput(BaseModel):
    """Input schema for analyze_combustion_efficiency tool"""
    fuel_type: str = Field(..., description="Fuel type (natural_gas, fuel_oil, coal, biomass)")
    fuel_flow_rate: float = Field(..., gt=0, description="Fuel flow rate (kg/hr or m3/hr)")
    air_flow_rate: float = Field(..., gt=0, description="Combustion air flow rate (m3/hr)")
    flue_gas_temperature: float = Field(..., description="Flue gas temperature (°C)")
    ambient_temperature: float = Field(..., description="Ambient air temperature (°C)")
    o2_level: float = Field(..., ge=0, le=21, description="O2 in flue gas (%)")
    co_ppm: Optional[float] = Field(0, ge=0, description="CO concentration (ppm)")

    @validator('fuel_type')
    def validate_fuel_type(cls, v: str) -> str:
        valid_types = ['natural_gas', 'fuel_oil', 'coal', 'biomass', 'propane', 'butane']
        if v not in valid_types:
            raise ValueError(f"Fuel type must be one of {valid_types}")
        return v


class AnalyzeCombustionEfficiencyOutput(BaseModel):
    """Output schema for analyze_combustion_efficiency tool"""
    gross_efficiency: float = Field(..., description="Gross combustion efficiency (%)")
    net_efficiency: float = Field(..., description="Net combustion efficiency (%)")
    dry_flue_gas_loss: float = Field(..., description="Heat loss in dry flue gas (%)")
    moisture_loss: float = Field(..., description="Heat loss from moisture (%)")
    incomplete_combustion_loss: float = Field(..., description="Loss from incomplete combustion (%)")
    radiation_loss: float = Field(..., description="Radiation and convection loss (%)")
    total_losses: float = Field(..., description="Total heat losses (%)")
    excess_air_percent: float = Field(..., description="Excess air percentage (%)")
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class OptimizeAirFuelRatioInput(BaseModel):
    """Input schema for optimize_air_fuel_ratio tool"""
    current_fuel_flow: float = Field(..., gt=0, description="Current fuel flow (kg/hr or m3/hr)")
    current_air_flow: float = Field(..., gt=0, description="Current air flow (m3/hr)")
    current_efficiency: float = Field(..., ge=0, le=100, description="Current efficiency (%)")
    current_nox_ppm: float = Field(..., ge=0, description="Current NOx (ppm)")
    current_co_ppm: float = Field(..., ge=0, description="Current CO (ppm)")
    fuel_type: str
    target_efficiency: float = Field(90.0, description="Target efficiency (%)")
    max_nox_limit: float = Field(50.0, description="Maximum NOx limit (ppm)")
    max_co_limit: float = Field(100.0, description="Maximum CO limit (ppm)")


class OptimizeAirFuelRatioOutput(BaseModel):
    """Output schema for optimize_air_fuel_ratio tool"""
    optimal_fuel_flow: float = Field(..., description="Optimal fuel flow (kg/hr or m3/hr)")
    optimal_air_flow: float = Field(..., description="Optimal air flow (m3/hr)")
    optimal_air_fuel_ratio: float = Field(..., description="Optimal AFR")
    optimal_excess_air: float = Field(..., description="Optimal excess air (%)")
    predicted_efficiency: float = Field(..., description="Predicted efficiency (%)")
    predicted_nox: float = Field(..., description="Predicted NOx (ppm)")
    predicted_co: float = Field(..., description="Predicted CO (ppm)")
    efficiency_improvement: float = Field(..., description="Efficiency gain (%)")
    fuel_savings: float = Field(..., description="Fuel savings (kg/hr)")
    convergence_status: str
    iterations: int
    confidence_score: float = Field(..., ge=0, le=1)


class MonitorFlameCharacteristicsInput(BaseModel):
    """Input schema for monitor_flame_characteristics tool"""
    flame_temperature: float = Field(..., description="Measured flame temperature (°C)")
    flame_signal_strength: float = Field(..., ge=0, le=100, description="Flame signal %")
    fuel_type: str
    air_fuel_ratio: float = Field(..., gt=0)


class MonitorFlameCharacteristicsOutput(BaseModel):
    """Output schema for monitor_flame_characteristics tool"""
    flame_present: bool
    flame_quality: str = Field(..., description="excellent, good, fair, poor")
    adiabatic_flame_temperature: float = Field(..., description="Theoretical max temp (°C)")
    actual_vs_theoretical_ratio: float
    flame_stability: str = Field(..., description="stable, unstable, oscillating")
    heat_release_rate: float = Field(..., description="kW or BTU/hr")
    recommendations: List[str]


class AdjustBurnerSettingsInput(BaseModel):
    """Input schema for adjust_burner_settings tool"""
    new_fuel_flow: float = Field(..., gt=0, description="New fuel flow setpoint")
    new_air_flow: float = Field(..., gt=0, description="New air flow setpoint")
    ramp_rate: float = Field(5.0, description="Change rate (%/min)")
    safety_check: bool = Field(True, description="Perform safety checks before adjustment")


class AdjustBurnerSettingsOutput(BaseModel):
    """Output schema for adjust_burner_settings tool"""
    success: bool
    actual_fuel_flow: float
    actual_air_flow: float
    setpoint_error: float = Field(..., description="Deviation from setpoint (%)")
    safety_interlocks_ok: bool
    message: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class MeasureEmissionsLevelsInput(BaseModel):
    """Input schema for measure_emissions_levels tool"""
    measurement_duration_seconds: int = Field(60, ge=10, le=600)
    reference_o2: float = Field(3.0, description="Reference O2 for correction (%)")


class MeasureEmissionsLevelsOutput(BaseModel):
    """Output schema for measure_emissions_levels tool"""
    nox_ppm: float = Field(..., description="NOx at reference O2")
    co_ppm: float = Field(..., description="CO at reference O2")
    co2_percent: float = Field(..., description="CO2 percentage")
    so2_ppm: Optional[float] = Field(None, description="SO2 if measured")
    o2_percent: float
    nox_mg_per_nm3: float = Field(..., description="NOx (mg/Nm³)")
    co_mg_per_nm3: float = Field(..., description="CO (mg/Nm³)")
    epa_compliance: bool = Field(..., description="Meets EPA limits")
    eu_ied_compliance: bool = Field(..., description="Meets EU IED limits")
    measurement_quality: str = Field(..., description="good, fair, poor")


class CalculateStoichiometricRatioInput(BaseModel):
    """Input schema for calculate_stoichiometric_ratio tool"""
    fuel_type: str
    fuel_composition: Dict[str, float] = Field(..., description="C, H, O, N, S percentages")
    temperature: float = Field(25.0, description="Reference temperature (°C)")
    pressure: float = Field(101.325, description="Reference pressure (kPa)")


class CalculateStoichiometricRatioOutput(BaseModel):
    """Output schema for calculate_stoichiometric_ratio tool"""
    stoichiometric_air_fuel_ratio: float = Field(..., description="Theoretical AFR (kg air / kg fuel)")
    stoichiometric_air_volume: float = Field(..., description="Theoretical air (m³/kg fuel)")
    theoretical_co2_percent: float
    theoretical_h2o_percent: float
    theoretical_n2_percent: float
    flue_gas_volume_per_kg_fuel: float = Field(..., description="m³/kg fuel")
    lower_heating_value: float = Field(..., description="LHV (MJ/kg)")
    higher_heating_value: float = Field(..., description="HHV (MJ/kg)")


class DetectIncompleteCombustionInput(BaseModel):
    """Input schema for detect_incomplete_combustion tool"""
    co_ppm: float = Field(..., ge=0)
    o2_percent: float = Field(..., ge=0, le=21)
    smoke_opacity: Optional[float] = Field(None, ge=0, le=100, description="Smoke opacity %")
    unburned_hydrocarbons_ppm: Optional[float] = Field(None, ge=0)


class DetectIncompleteCombustionOutput(BaseModel):
    """Output schema for detect_incomplete_combustion tool"""
    incomplete_combustion_detected: bool
    severity: str = Field(..., description="none, minor, moderate, severe")
    estimated_efficiency_loss: float = Field(..., description="Efficiency loss (%)")
    probable_causes: List[str]
    recommended_actions: List[str]
    economic_impact_per_hour: float = Field(..., description="$ lost per hour")


class OptimizeExcessAirInput(BaseModel):
    """Input schema for optimize_excess_air tool"""
    current_excess_air: float = Field(..., description="Current excess air (%)")
    current_o2: float = Field(..., ge=0, le=21)
    burner_type: str = Field(..., description="premix, non-premix, staged, low_nox")
    fuel_type: str
    desired_outcome: str = Field(..., description="max_efficiency, min_emissions, balanced")


class OptimizeExcessAirOutput(BaseModel):
    """Output schema for optimize_excess_air tool"""
    optimal_excess_air: float = Field(..., description="Optimal excess air (%)")
    optimal_o2: float = Field(..., description="Optimal O2 (%)")
    predicted_efficiency_gain: float
    predicted_nox_reduction: float = Field(..., description="NOx reduction (%)")
    safety_margin: float = Field(..., description="Safety margin from minimum O2 (%)")
    implementation_risk: str = Field(..., description="low, medium, high")


class PredictNOxFormationInput(BaseModel):
    """Input schema for predict_nox_formation tool"""
    flame_temperature: float = Field(..., description="Peak flame temperature (°C)")
    excess_air: float = Field(..., description="Excess air (%)")
    residence_time_ms: float = Field(..., description="Combustion zone residence time (ms)")
    fuel_nitrogen_percent: float = Field(0, description="Nitrogen in fuel (%)")
    burner_type: str


class PredictNOxFormationOutput(BaseModel):
    """Output schema for predict_nox_formation tool"""
    thermal_nox_ppm: float = Field(..., description="Thermal NOx contribution")
    fuel_nox_ppm: float = Field(..., description="Fuel NOx contribution")
    prompt_nox_ppm: float = Field(..., description="Prompt NOx contribution")
    total_nox_ppm: float
    nox_formation_mechanism: str = Field(..., description="Dominant mechanism")
    reduction_strategies: List[str]


class TuneControlParametersInput(BaseModel):
    """Input schema for tune_control_parameters tool"""
    control_loop: str = Field(..., description="fuel_flow, air_flow, o2_trim")
    current_kp: float = Field(..., description="Current proportional gain")
    current_ki: float = Field(..., description="Current integral gain")
    current_kd: float = Field(..., description="Current derivative gain")
    process_variable_trend: List[float] = Field(..., description="Recent PV values")
    setpoint_changes: List[float] = Field(..., description="Recent SP changes")


class TuneControlParametersOutput(BaseModel):
    """Output schema for tune_control_parameters tool"""
    recommended_kp: float
    recommended_ki: float
    recommended_kd: float
    tuning_method: str = Field(..., description="ziegler_nichols, cohen_coon, lambda, imc")
    expected_settling_time_seconds: float
    expected_overshoot_percent: float
    stability_margin: float
    implementation_notes: str


# Tool registry
TOOL_REGISTRY = {
    'analyze_combustion_efficiency': {
        'name': 'analyze_combustion_efficiency',
        'description': 'Analyze combustion efficiency and heat losses using ASME PTC 4.1 methodology',
        'input_schema': AnalyzeCombustionEfficiencyInput,
        'output_schema': AnalyzeCombustionEfficiencyOutput,
        'deterministic': True,
        'zero_hallucination': True
    },
    'optimize_air_fuel_ratio': {
        'name': 'optimize_air_fuel_ratio',
        'description': 'Multi-objective optimization of air-fuel ratio for efficiency and emissions',
        'input_schema': OptimizeAirFuelRatioInput,
        'output_schema': OptimizeAirFuelRatioOutput,
        'deterministic': True,
        'zero_hallucination': True
    },
    'monitor_flame_characteristics': {
        'name': 'monitor_flame_characteristics',
        'description': 'Monitor and analyze flame quality, temperature, and stability',
        'input_schema': MonitorFlameCharacteristicsInput,
        'output_schema': MonitorFlameCharacteristicsOutput,
        'deterministic': True,
        'zero_hallucination': True
    },
    'adjust_burner_settings': {
        'name': 'adjust_burner_settings',
        'description': 'Safely adjust burner fuel and air flow setpoints with interlocks',
        'input_schema': AdjustBurnerSettingsInput,
        'output_schema': AdjustBurnerSettingsOutput,
        'deterministic': False,  # Interacts with physical equipment
        'zero_hallucination': True
    },
    'measure_emissions_levels': {
        'name': 'measure_emissions_levels',
        'description': 'Measure NOx, CO, CO2, SO2 emissions with compliance checking',
        'input_schema': MeasureEmissionsLevelsInput,
        'output_schema': MeasureEmissionsLevelsOutput,
        'deterministic': False,  # Real-time measurements
        'zero_hallucination': True
    },
    'calculate_stoichiometric_ratio': {
        'name': 'calculate_stoichiometric_ratio',
        'description': 'Calculate theoretical air requirement and combustion products',
        'input_schema': CalculateStoichiometricRatioInput,
        'output_schema': CalculateStoichiometricRatioOutput,
        'deterministic': True,
        'zero_hallucination': True
    },
    'detect_incomplete_combustion': {
        'name': 'detect_incomplete_combustion',
        'description': 'Detect incomplete combustion from CO, smoke, and UHC measurements',
        'input_schema': DetectIncompleteCombustionInput,
        'output_schema': DetectIncompleteCombustionOutput,
        'deterministic': True,
        'zero_hallucination': True
    },
    'optimize_excess_air': {
        'name': 'optimize_excess_air',
        'description': 'Optimize excess air for efficiency, emissions, or balanced operation',
        'input_schema': OptimizeExcessAirInput,
        'output_schema': OptimizeExcessAirOutput,
        'deterministic': True,
        'zero_hallucination': True
    },
    'predict_nox_formation': {
        'name': 'predict_nox_formation',
        'description': 'Predict NOx formation from thermal, fuel, and prompt mechanisms',
        'input_schema': PredictNOxFormationInput,
        'output_schema': PredictNOxFormationOutput,
        'deterministic': True,
        'zero_hallucination': True
    },
    'tune_control_parameters': {
        'name': 'tune_control_parameters',
        'description': 'Tune PID control parameters for fuel, air, or O2 trim loops',
        'input_schema': TuneControlParametersInput,
        'output_schema': TuneControlParametersOutput,
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
