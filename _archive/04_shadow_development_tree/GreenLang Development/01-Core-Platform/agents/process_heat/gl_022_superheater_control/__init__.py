"""
GL-022 SUPERHEATER CONTROL AGENT

This module provides comprehensive superheater temperature control with spray
water optimization, PID tuning, safety validation, and ML-based explainability.

Features:
    - Spray water setpoint optimization using energy balance calculations
    - PID control parameter tuning with Lambda method
    - Temperature prediction with uncertainty quantification
    - Safety constraint validation per IEC 61511 / ASME standards
    - SHAP/LIME explainability hooks for control decisions
    - Natural language summaries via LLM integration
    - Zero-hallucination deterministic calculations
    - SHA-256 provenance tracking for audit compliance
    - Async support with deterministic guarantees
    - IAPWS-IF97 steam property calculations
    - Tube metal temperature protection per API 530

Standards References:
    - IAPWS-IF97 Steam Properties
    - ASME Boiler and Pressure Vessel Code
    - IEC 61511 Safety Instrumented Systems
    - ISA-5.1 Instrumentation Symbols and Identification
    - NFPA 85 Boiler and Combustion Systems Hazards Code
    - API 530 Calculation of Heater-tube Thickness

Modules:
    controller: Main SuperheaterController class and calculators
    schemas: Pydantic data models for all agent I/O
    safety: Safety validation and compliance checking
    calculators: Calculation engines (planned)
    integrations: External system integrations (planned)

Example Usage:
    >>> from greenlang.agents.process_heat.gl_022_superheater_control import (
    ...     SuperheaterController,
    ...     SuperheaterControllerConfig,
    ...     SuperheaterControllerInput,
    ...     create_default_controller_config,
    ... )
    >>>
    >>> # Create configuration
    >>> config = create_default_controller_config()
    >>>
    >>> # Initialize controller
    >>> controller = SuperheaterController(config)
    >>>
    >>> # Process control (see controller.py for full example)
    >>> result = controller.process(input_data)
    >>> print(f"Spray setpoint: {result.spray_setpoint.target_flow_kg_s} kg/s")

Author: GreenLang Engineering Team
Version: 1.0.0
License: Proprietary
"""

# =============================================================================
# VERSION INFO
# =============================================================================

__version__ = "1.0.0"
__author__ = "GreenLang Engineering"
__agent_id__ = "GL-022"
__agent_name__ = "Superheater Control Agent"

# =============================================================================
# CONTROLLER EXPORTS
# =============================================================================

from .controller import (
    # Main controller class
    SuperheaterController,

    # Configuration classes (with Controller prefix to avoid conflicts)
    SuperheaterControlConfig as SuperheaterControllerConfig,
    PIDConfig,
    SafetyConfig as ControllerSafetyConfig,
    ExplainabilityConfig,
    create_default_config as create_default_controller_config,

    # Input/Output models (with Controller prefix to avoid conflicts)
    SuperheaterControlInput as SuperheaterControllerInput,
    SuperheaterControlOutput as SuperheaterControllerOutput,

    # Steam and process condition models (with Controller prefix)
    SteamConditions as ControllerSteamConditions,
    SprayWaterConditions,
    ProcessDemand,

    # Output component models
    SpraySetpoint,
    PIDOutput,
    TemperaturePrediction,
    SafetyAssessment,
    ExplainabilityReport,
    SHAPExplanation,
    UncertaintyQuantification,

    # Enums from controller (with Controller prefix to avoid conflicts)
    ControlAction as ControllerControlAction,
    SafetyStatus as ControllerSafetyStatus,
    ControlMode as ControllerControlMode,
    PredictionConfidence,

    # Calculator classes
    SteamPropertyCalculator,
    SprayWaterCalculator,
    PIDController,
    TemperaturePredictor,
    SafetyValidator as ControllerSafetyValidator,
    ExplainabilityEngine,
)

# =============================================================================
# SCHEMAS EXPORTS
# =============================================================================

from .schemas import (
    # Enums from schemas
    ControlMode,
    SprayValveStatus,
    SafetyStatus as SchemasSafetyStatus,
    ValidationStatus,
    SteamPhase,
    ThermalStressLevel,
    ControlAction,
    AlarmSeverity,
    # Steam Properties
    SteamConditions,
    # Operating Point
    SuperheaterOperatingPoint,
    # Desuperheater Spray
    SprayWaterQuality,
    DesuperheaterSprayInput,
    DesuperheaterSprayOutput,
    # Temperature Control
    PIDParameters,
    TemperatureControlInput,
    TemperatureControlOutput,
    # Tube Metal Temperature
    TubeMetalTemperatureReading,
    TubeMetalTemperatureAnalysis,
    # Process Demand
    ProcessDemandReading,
    ProcessDemandSummary,
    # Safety Interlocks
    SafetyInterlockStatus,
    SafetySystemSummary,
    # Efficiency Metrics
    SuperheaterEfficiencyMetrics,
    # Alarms
    SuperheaterAlarm,
    # Comprehensive I/O
    SuperheaterControlInput,
    SuperheaterControlOutput,
    # Configuration
    SuperheaterControlConfig,
)

# =============================================================================
# SAFETY MODULE EXPORTS
# =============================================================================

from .safety import (
    # Enums
    SafetyStatus,
    ComplianceStatus,
    MaterialGrade,
    InterlockType,
    # Data Models
    SafetyCheckResult,
    ComplianceCheckResult,
    ThermalShockAssessment,
    InterlockStatusReport,
    CreepRuptureAnalysis,
    # Classes
    SafetyValidator,
    ASMEComplianceChecker,
    # Constants
    MATERIAL_MAX_ALLOWABLE_STRESS,
    MATERIAL_MAX_TEMP_F,
    CREEP_RUPTURE_PARAMS,
    REQUIRED_SUPERHEATER_INTERLOCKS,
    # Utility Functions
    celsius_to_fahrenheit,
    fahrenheit_to_celsius,
    bar_to_psi,
    psi_to_bar,
    mm_to_inches,
    inches_to_mm,
)

# =============================================================================
# PUBLIC API
# =============================================================================

__all__ = [
    # Version info
    "__version__",
    "__author__",
    "__agent_id__",
    "__agent_name__",

    # ==========================================================================
    # CONTROLLER MODULE EXPORTS
    # ==========================================================================

    # Main controller
    "SuperheaterController",

    # Configuration (Controller)
    "SuperheaterControllerConfig",
    "PIDConfig",
    "ControllerSafetyConfig",
    "ExplainabilityConfig",
    "create_default_controller_config",

    # Input/Output models (Controller)
    "SuperheaterControllerInput",
    "SuperheaterControllerOutput",

    # Condition models (Controller)
    "ControllerSteamConditions",
    "SprayWaterConditions",
    "ProcessDemand",

    # Output components
    "SpraySetpoint",
    "PIDOutput",
    "TemperaturePrediction",
    "SafetyAssessment",
    "ExplainabilityReport",
    "SHAPExplanation",
    "UncertaintyQuantification",

    # Controller Enums
    "ControllerControlAction",
    "ControllerSafetyStatus",
    "ControllerControlMode",
    "PredictionConfidence",

    # Calculators
    "SteamPropertyCalculator",
    "SprayWaterCalculator",
    "PIDController",
    "TemperaturePredictor",
    "ControllerSafetyValidator",
    "ExplainabilityEngine",

    # ==========================================================================
    # SCHEMAS MODULE EXPORTS
    # ==========================================================================

    # Schema Enums
    "ControlMode",
    "SprayValveStatus",
    "SchemasSafetyStatus",
    "ValidationStatus",
    "SteamPhase",
    "ThermalStressLevel",
    "ControlAction",
    "AlarmSeverity",

    # Steam Properties
    "SteamConditions",

    # Operating Point
    "SuperheaterOperatingPoint",

    # Desuperheater Spray
    "SprayWaterQuality",
    "DesuperheaterSprayInput",
    "DesuperheaterSprayOutput",

    # Temperature Control
    "PIDParameters",
    "TemperatureControlInput",
    "TemperatureControlOutput",

    # Tube Metal Temperature
    "TubeMetalTemperatureReading",
    "TubeMetalTemperatureAnalysis",

    # Process Demand
    "ProcessDemandReading",
    "ProcessDemandSummary",

    # Safety Interlocks (Schemas)
    "SafetyInterlockStatus",
    "SafetySystemSummary",

    # Efficiency Metrics
    "SuperheaterEfficiencyMetrics",

    # Alarms
    "SuperheaterAlarm",

    # Comprehensive I/O (Schemas)
    "SuperheaterControlInput",
    "SuperheaterControlOutput",

    # Configuration (Schemas)
    "SuperheaterControlConfig",

    # ==========================================================================
    # SAFETY MODULE EXPORTS
    # ==========================================================================

    # Safety Module Enums
    "SafetyStatus",
    "ComplianceStatus",
    "MaterialGrade",
    "InterlockType",

    # Safety Module Data Models
    "SafetyCheckResult",
    "ComplianceCheckResult",
    "ThermalShockAssessment",
    "InterlockStatusReport",
    "CreepRuptureAnalysis",

    # Safety Module Classes
    "SafetyValidator",
    "ASMEComplianceChecker",

    # Safety Module Constants
    "MATERIAL_MAX_ALLOWABLE_STRESS",
    "MATERIAL_MAX_TEMP_F",
    "CREEP_RUPTURE_PARAMS",
    "REQUIRED_SUPERHEATER_INTERLOCKS",

    # Utility Functions
    "celsius_to_fahrenheit",
    "fahrenheit_to_celsius",
    "bar_to_psi",
    "psi_to_bar",
    "mm_to_inches",
    "inches_to_mm",
]


# =============================================================================
# MODULE INITIALIZATION
# =============================================================================

def get_agent_info() -> dict:
    """
    Get agent information dictionary.

    Returns:
        Dictionary with agent metadata
    """
    return {
        "agent_id": __agent_id__,
        "agent_name": __agent_name__,
        "version": __version__,
        "author": __author__,
        "capabilities": [
            "spray_water_optimization",
            "pid_control",
            "temperature_prediction",
            "safety_validation",
            "asme_compliance",
            "shap_explainability",
            "uncertainty_quantification",
            "natural_language_summaries",
            "provenance_tracking",
        ],
        "standards": [
            "IAPWS-IF97",
            "ASME BPVC",
            "IEC 61511",
            "ISA-5.1",
            "NFPA 85",
            "API 530",
        ],
        "safety_levels": ["SIL-1", "SIL-2", "SIL-3"],
    }


def create_controller(
    equipment_id: str = "SH-001",
    target_temp_c: float = 540.0,
    **kwargs
) -> SuperheaterController:
    """
    Factory function to create a SuperheaterController with common defaults.

    This is a convenience function for quick controller creation.

    Args:
        equipment_id: Superheater equipment identifier
        target_temp_c: Default target outlet temperature (C)
        **kwargs: Additional configuration overrides

    Returns:
        Configured SuperheaterController instance

    Example:
        >>> controller = create_controller(
        ...     equipment_id="SH-002",
        ...     target_temp_c=535.0
        ... )
    """
    config = SuperheaterControllerConfig(
        equipment_id=equipment_id,
        default_target_temp_c=target_temp_c,
        **kwargs
    )
    return SuperheaterController(config)


# =============================================================================
# QUICK START HELPERS
# =============================================================================

def quick_spray_calculation(
    inlet_temp_c: float,
    target_temp_c: float,
    steam_flow_kg_s: float,
    spray_water_temp_c: float,
    pressure_bar: float,
) -> dict:
    """
    Quick spray water calculation without full controller initialization.

    This is a utility function for rapid calculations without the full
    controller overhead.

    Args:
        inlet_temp_c: Inlet steam temperature (C)
        target_temp_c: Target outlet temperature (C)
        steam_flow_kg_s: Steam mass flow rate (kg/s)
        spray_water_temp_c: Spray water temperature (C)
        pressure_bar: Steam pressure (bar)

    Returns:
        Dictionary with spray flow requirement and energy metrics

    Example:
        >>> result = quick_spray_calculation(
        ...     inlet_temp_c=550,
        ...     target_temp_c=540,
        ...     steam_flow_kg_s=50,
        ...     spray_water_temp_c=150,
        ...     pressure_bar=100
        ... )
        >>> print(f"Required spray: {result['spray_flow_kg_s']} kg/s")
    """
    steam_calc = SteamPropertyCalculator()
    spray_calc = SprayWaterCalculator(steam_calc)

    # Calculate saturation temperature
    t_sat = steam_calc.calculate_saturation_temperature(pressure_bar)

    # Calculate required spray flow
    spray_flow, energy = spray_calc.calculate_required_spray_flow(
        steam_flow_kg_s=steam_flow_kg_s,
        inlet_temp_c=inlet_temp_c,
        target_temp_c=target_temp_c,
        spray_water_temp_c=spray_water_temp_c,
        pressure_bar=pressure_bar
    )

    # Calculate enthalpies
    h_in = steam_calc.calculate_enthalpy(inlet_temp_c, pressure_bar)
    h_out = steam_calc.calculate_enthalpy(target_temp_c, pressure_bar)

    return {
        "spray_flow_kg_s": spray_flow,
        "energy_absorbed_kw": energy,
        "saturation_temp_c": t_sat,
        "inlet_superheat_c": inlet_temp_c - t_sat,
        "outlet_superheat_c": target_temp_c - t_sat,
        "inlet_enthalpy_kj_kg": h_in,
        "outlet_enthalpy_kj_kg": h_out,
        "enthalpy_reduction_kj_kg": h_in - h_out,
    }


def quick_pid_tuning(
    process_time_constant_s: float = 60.0,
    process_dead_time_s: float = 10.0,
    desired_response_time_s: float = 120.0,
) -> dict:
    """
    Quick PID tuning using Lambda method.

    Args:
        process_time_constant_s: Process time constant (seconds)
        process_dead_time_s: Process dead time (seconds)
        desired_response_time_s: Desired closed-loop response time (seconds)

    Returns:
        Dictionary with tuned PID parameters

    Example:
        >>> params = quick_pid_tuning(
        ...     process_time_constant_s=90,
        ...     process_dead_time_s=15,
        ...     desired_response_time_s=180
        ... )
        >>> print(f"Kp: {params['kp']}, Ki: {params['ki']}, Kd: {params['kd']}")
    """
    config = PIDController.tune_lambda(
        process_time_constant_s=process_time_constant_s,
        process_dead_time_s=process_dead_time_s,
        desired_response_time_s=desired_response_time_s
    )

    return {
        "kp": config.kp,
        "ki": config.ki,
        "kd": config.kd,
        "deadband_c": config.deadband_c,
        "max_rate_c_per_min": config.max_rate_c_per_min,
        "tuning_method": "lambda",
        "process_time_constant_s": process_time_constant_s,
        "process_dead_time_s": process_dead_time_s,
        "desired_response_time_s": desired_response_time_s,
    }


def quick_safety_check(
    outlet_temp_c: float,
    saturation_temp_c: float,
    tube_metal_temp_c: float = None,
    spray_valve_pct: float = 50.0,
    max_tube_temp_c: float = 600.0,
    min_superheat_c: float = 20.0,
) -> dict:
    """
    Quick safety constraint check.

    Args:
        outlet_temp_c: Outlet steam temperature (C)
        saturation_temp_c: Saturation temperature at pressure (C)
        tube_metal_temp_c: Tube metal temperature if measured (C)
        spray_valve_pct: Current spray valve position (%)
        max_tube_temp_c: Maximum allowable tube temperature (C)
        min_superheat_c: Minimum required superheat (C)

    Returns:
        Dictionary with safety assessment

    Example:
        >>> result = quick_safety_check(
        ...     outlet_temp_c=540,
        ...     saturation_temp_c=311,
        ...     tube_metal_temp_c=580,
        ...     spray_valve_pct=60
        ... )
        >>> print(f"Safety status: {result['status']}")
    """
    safety_config = ControllerSafetyConfig(
        max_tube_metal_temp_c=max_tube_temp_c,
        min_superheat_c=min_superheat_c,
    )

    validator = ControllerSafetyValidator(safety_config)
    assessment = validator.validate(
        current_temp_c=outlet_temp_c,
        saturation_temp_c=saturation_temp_c,
        tube_metal_temp_c=tube_metal_temp_c,
        spray_flow_pct=spray_valve_pct
    )

    return {
        "status": assessment.status.value,
        "tube_metal_margin_c": assessment.tube_metal_margin_c,
        "superheat_margin_c": assessment.superheat_margin_c,
        "spray_capacity_margin_pct": assessment.spray_capacity_margin_pct,
        "constraints_satisfied": assessment.constraints_satisfied,
        "constraints_violated": assessment.constraints_violated,
        "emergency_required": assessment.emergency_action_required,
    }
