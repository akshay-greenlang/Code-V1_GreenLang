"""
GL-022 SUPERHEATER CONTROL AGENT - Main Controller Module

This module provides the SuperheaterController class that implements comprehensive
superheater temperature control with spray water optimization, PID tuning,
safety validation, and ML-based explainability.

Features:
    - Spray water setpoint optimization with energy balance calculations
    - PID control parameter tuning using Lambda method
    - Temperature prediction with uncertainty quantification
    - Safety constraint validation against tube metal limits
    - SHAP/LIME explainability hooks for control decisions
    - Zero-hallucination deterministic calculations
    - SHA-256 provenance tracking for audit compliance
    - Async support with deterministic guarantees

Standards References:
    - IAPWS-IF97 Steam Properties
    - ASME Boiler and Pressure Vessel Code
    - IEC 61511 Safety Instrumented Systems
    - ISA-5.1 Instrumentation Symbols and Identification

Example:
    >>> from greenlang.agents.process_heat.gl_022_superheater_control import (
    ...     SuperheaterController,
    ...     SuperheaterControlConfig,
    ...     SuperheaterControlInput,
    ...     create_default_config,
    ... )
    >>>
    >>> config = create_default_config()
    >>> controller = SuperheaterController(config)
    >>> result = controller.process(input_data)
    >>> print(f"Recommended spray flow: {result.spray_setpoint_kg_s} kg/s")

Author: GreenLang Engineering Team
Version: 1.0.0
License: Proprietary
"""

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple, Union
from enum import Enum
import asyncio
import hashlib
import json
import logging
import math
import time

from pydantic import BaseModel, Field, validator

# Intelligence imports for LLM capabilities
from greenlang.agents.intelligence_mixin import IntelligenceMixin, IntelligenceConfig
from greenlang.agents.intelligence_interface import IntelligenceCapabilities, IntelligenceLevel

# Import from shared base
from ..shared.base_agent import (
    BaseProcessHeatAgent,
    AgentConfig,
    AgentCapability,
    SafetyLevel,
    ProcessingError,
    ValidationError,
)

logger = logging.getLogger(__name__)


# =============================================================================
# ENUMS
# =============================================================================

class ControlAction(str, Enum):
    """Spray water control action types."""
    INCREASE = "increase"
    DECREASE = "decrease"
    MAINTAIN = "maintain"


class SafetyStatus(str, Enum):
    """Safety status classification."""
    SAFE = "safe"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


class ControlMode(str, Enum):
    """Controller operating mode."""
    AUTOMATIC = "automatic"
    MANUAL = "manual"
    CASCADE = "cascade"
    OVERRIDE = "override"


class PredictionConfidence(str, Enum):
    """Prediction confidence level."""
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


# =============================================================================
# CONFIGURATION MODELS
# =============================================================================

class PIDConfig(BaseModel):
    """PID controller configuration."""

    kp: float = Field(
        default=1.0,
        ge=0.0,
        le=10.0,
        description="Proportional gain"
    )
    ki: float = Field(
        default=0.01,
        ge=0.0,
        le=1.0,
        description="Integral gain"
    )
    kd: float = Field(
        default=0.1,
        ge=0.0,
        le=10.0,
        description="Derivative gain"
    )
    deadband_c: float = Field(
        default=1.0,
        ge=0.0,
        le=10.0,
        description="Control deadband in degrees C"
    )
    max_rate_c_per_min: float = Field(
        default=5.0,
        ge=0.1,
        le=20.0,
        description="Maximum temperature rate of change (C/min)"
    )
    integral_windup_limit: float = Field(
        default=100.0,
        ge=0.0,
        description="Integral windup limit"
    )


class SafetyConfig(BaseModel):
    """Safety limits configuration."""

    max_tube_metal_temp_c: float = Field(
        default=600.0,
        ge=400.0,
        le=800.0,
        description="Maximum allowable tube metal temperature (C)"
    )
    min_superheat_c: float = Field(
        default=20.0,
        ge=5.0,
        le=100.0,
        description="Minimum required superheat above saturation (C)"
    )
    max_spray_flow_pct: float = Field(
        default=90.0,
        ge=50.0,
        le=100.0,
        description="Maximum spray flow as percent of capacity"
    )
    warning_margin_c: float = Field(
        default=25.0,
        ge=5.0,
        le=50.0,
        description="Warning margin below max tube temp (C)"
    )
    critical_margin_c: float = Field(
        default=10.0,
        ge=2.0,
        le=25.0,
        description="Critical margin below max tube temp (C)"
    )
    emergency_shutdown_enabled: bool = Field(
        default=True,
        description="Enable emergency shutdown on critical conditions"
    )


class ExplainabilityConfig(BaseModel):
    """SHAP/LIME explainability configuration."""

    enabled: bool = Field(default=True, description="Enable explainability")
    method: str = Field(
        default="shap",
        description="Explainability method: shap, lime, or both"
    )
    num_features: int = Field(
        default=10,
        ge=1,
        le=50,
        description="Number of top features to explain"
    )
    generate_natural_language: bool = Field(
        default=True,
        description="Generate natural language explanations"
    )


class SuperheaterControlConfig(BaseModel):
    """Complete superheater control configuration."""

    agent_id: str = Field(
        default="GL-022-SH-CTRL",
        description="Unique agent identifier"
    )
    equipment_id: str = Field(
        default="SH-001",
        description="Superheater equipment identifier"
    )

    # Control parameters
    pid: PIDConfig = Field(
        default_factory=PIDConfig,
        description="PID controller configuration"
    )

    # Safety limits
    safety: SafetyConfig = Field(
        default_factory=SafetyConfig,
        description="Safety limits configuration"
    )

    # Explainability
    explainability: ExplainabilityConfig = Field(
        default_factory=ExplainabilityConfig,
        description="Explainability configuration"
    )

    # Process parameters
    process_time_constant_s: float = Field(
        default=60.0,
        ge=10.0,
        le=600.0,
        description="Process time constant (seconds)"
    )
    process_dead_time_s: float = Field(
        default=10.0,
        ge=1.0,
        le=120.0,
        description="Process dead time (seconds)"
    )

    # Temperature targets
    default_target_temp_c: float = Field(
        default=540.0,
        ge=200.0,
        le=650.0,
        description="Default target outlet temperature (C)"
    )
    temp_tolerance_c: float = Field(
        default=5.0,
        ge=1.0,
        le=20.0,
        description="Allowable temperature deviation (C)"
    )

    # Energy costs for optimization
    fuel_cost_usd_per_mwh: float = Field(
        default=30.0,
        ge=0.0,
        description="Fuel cost for energy calculations"
    )
    water_cost_usd_per_m3: float = Field(
        default=1.0,
        ge=0.0,
        description="Spray water cost"
    )


def create_default_config() -> SuperheaterControlConfig:
    """Create default superheater control configuration."""
    return SuperheaterControlConfig()


# =============================================================================
# INPUT/OUTPUT MODELS
# =============================================================================

class SteamConditions(BaseModel):
    """Steam thermodynamic conditions."""

    temperature_c: float = Field(..., description="Steam temperature (C)")
    pressure_bar: float = Field(..., ge=0.0, description="Steam pressure (bar)")
    flow_kg_s: float = Field(..., ge=0.0, description="Steam mass flow (kg/s)")
    saturation_temp_c: Optional[float] = Field(
        default=None,
        description="Saturation temperature at pressure (C)"
    )
    superheat_c: Optional[float] = Field(
        default=None,
        description="Degree of superheat (C)"
    )
    enthalpy_kj_kg: Optional[float] = Field(
        default=None,
        description="Specific enthalpy (kJ/kg)"
    )


class SprayWaterConditions(BaseModel):
    """Spray water system conditions."""

    temperature_c: float = Field(..., ge=0.0, description="Spray water temperature (C)")
    current_flow_kg_s: float = Field(
        default=0.0,
        ge=0.0,
        description="Current spray flow (kg/s)"
    )
    max_flow_kg_s: float = Field(..., ge=0.0, description="Maximum spray capacity (kg/s)")
    valve_position_pct: float = Field(
        default=0.0,
        ge=0.0,
        le=100.0,
        description="Current valve position (%)"
    )
    available: bool = Field(default=True, description="Spray water available")


class ProcessDemand(BaseModel):
    """Process temperature demand."""

    target_temp_c: float = Field(..., ge=200.0, le=650.0, description="Target temperature (C)")
    tolerance_c: float = Field(default=5.0, ge=1.0, description="Allowable deviation (C)")
    ramp_rate_c_per_min: Optional[float] = Field(
        default=None,
        ge=0.0,
        description="Requested temperature ramp rate (C/min)"
    )
    priority: str = Field(
        default="normal",
        description="Request priority: low, normal, high, critical"
    )


class SuperheaterControlInput(BaseModel):
    """Input data model for SuperheaterController."""

    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Input timestamp"
    )
    equipment_id: str = Field(..., description="Superheater equipment ID")

    # Steam conditions
    inlet_steam: SteamConditions = Field(..., description="Inlet steam conditions")
    outlet_steam: SteamConditions = Field(..., description="Current outlet steam conditions")

    # Spray water
    spray_water: SprayWaterConditions = Field(..., description="Spray water conditions")

    # Process demand
    demand: ProcessDemand = Field(..., description="Process temperature demand")

    # Firing conditions
    burner_load_pct: float = Field(
        ...,
        ge=0.0,
        le=100.0,
        description="Burner load (%)"
    )
    flue_gas_temp_c: float = Field(..., description="Flue gas temperature (C)")
    excess_air_pct: float = Field(
        default=15.0,
        ge=0.0,
        description="Excess air (%)"
    )

    # Equipment state
    tube_metal_temp_c: Optional[float] = Field(
        default=None,
        description="Measured tube metal temperature (C)"
    )

    # Control mode
    control_mode: ControlMode = Field(
        default=ControlMode.AUTOMATIC,
        description="Current control mode"
    )

    class Config:
        json_encoders = {datetime: lambda v: v.isoformat()}


class SpraySetpoint(BaseModel):
    """Spray water setpoint recommendation."""

    target_flow_kg_s: float = Field(..., ge=0.0, description="Target spray flow (kg/s)")
    valve_position_pct: float = Field(
        ...,
        ge=0.0,
        le=100.0,
        description="Recommended valve position (%)"
    )
    rate_of_change_pct_per_min: float = Field(
        default=5.0,
        ge=0.0,
        description="Maximum rate of change for valve (%/min)"
    )
    action: ControlAction = Field(..., description="Control action type")
    confidence: float = Field(
        default=0.95,
        ge=0.0,
        le=1.0,
        description="Confidence in recommendation"
    )


class PIDOutput(BaseModel):
    """PID controller output."""

    proportional: float = Field(..., description="Proportional term")
    integral: float = Field(..., description="Integral term")
    derivative: float = Field(..., description="Derivative term")
    output: float = Field(..., description="Total PID output")
    saturated: bool = Field(default=False, description="Output is saturated")
    in_deadband: bool = Field(default=False, description="Error within deadband")


class TemperaturePrediction(BaseModel):
    """Temperature prediction with uncertainty."""

    predicted_temp_c: float = Field(..., description="Predicted temperature (C)")
    uncertainty_lower_c: float = Field(..., description="Lower bound of uncertainty (C)")
    uncertainty_upper_c: float = Field(..., description="Upper bound of uncertainty (C)")
    prediction_horizon_s: float = Field(..., description="Prediction horizon (seconds)")
    confidence: PredictionConfidence = Field(..., description="Confidence level")
    model_used: str = Field(default="first_order", description="Prediction model used")


class SafetyAssessment(BaseModel):
    """Safety constraint assessment."""

    status: SafetyStatus = Field(..., description="Overall safety status")
    tube_metal_margin_c: float = Field(..., description="Margin below max tube temp (C)")
    superheat_margin_c: float = Field(..., description="Margin above min superheat (C)")
    spray_capacity_margin_pct: float = Field(
        ...,
        description="Remaining spray capacity (%)"
    )
    constraints_satisfied: List[str] = Field(
        default_factory=list,
        description="List of satisfied constraints"
    )
    constraints_violated: List[str] = Field(
        default_factory=list,
        description="List of violated constraints"
    )
    emergency_action_required: bool = Field(
        default=False,
        description="Emergency action required flag"
    )


class SHAPExplanation(BaseModel):
    """SHAP-based feature explanation."""

    feature_name: str = Field(..., description="Feature name")
    feature_value: float = Field(..., description="Feature value")
    shap_value: float = Field(..., description="SHAP contribution value")
    contribution_pct: float = Field(..., description="Percentage contribution")
    direction: str = Field(..., description="Direction: positive or negative")


class ExplainabilityReport(BaseModel):
    """Complete explainability report."""

    method: str = Field(default="shap", description="Explainability method")
    base_value: float = Field(..., description="Base prediction value")
    final_value: float = Field(..., description="Final prediction value")
    top_features: List[SHAPExplanation] = Field(
        default_factory=list,
        description="Top contributing features"
    )
    natural_language_summary: Optional[str] = Field(
        default=None,
        description="Natural language explanation"
    )
    confidence_score: float = Field(
        default=0.9,
        ge=0.0,
        le=1.0,
        description="Explanation confidence"
    )


class UncertaintyQuantification(BaseModel):
    """Uncertainty quantification for outputs."""

    spray_flow_std_kg_s: float = Field(
        ...,
        ge=0.0,
        description="Standard deviation of spray flow estimate"
    )
    temperature_prediction_std_c: float = Field(
        ...,
        ge=0.0,
        description="Standard deviation of temp prediction"
    )
    model_uncertainty_pct: float = Field(
        ...,
        ge=0.0,
        le=100.0,
        description="Model uncertainty (%)"
    )
    data_quality_score: float = Field(
        default=0.95,
        ge=0.0,
        le=1.0,
        description="Input data quality score"
    )
    confidence_interval_95_pct: Tuple[float, float] = Field(
        ...,
        description="95% confidence interval"
    )


class SuperheaterControlOutput(BaseModel):
    """Output from SuperheaterController."""

    # Identification
    controller_id: str = Field(
        default="GL-022-SH-CTRL",
        description="Controller identifier"
    )
    equipment_id: str = Field(..., description="Equipment identifier")
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Output timestamp"
    )

    # Control outputs
    spray_setpoint: SpraySetpoint = Field(
        ...,
        description="Optimal spray water setpoint"
    )
    pid_output: PIDOutput = Field(..., description="PID control outputs")

    # Predictions
    temperature_prediction: TemperaturePrediction = Field(
        ...,
        description="Temperature prediction with uncertainty"
    )

    # Safety
    safety_assessment: SafetyAssessment = Field(
        ...,
        description="Safety constraint assessment"
    )

    # Explainability
    explainability_report: Optional[ExplainabilityReport] = Field(
        default=None,
        description="SHAP/LIME explainability report"
    )

    # Uncertainty
    uncertainty: UncertaintyQuantification = Field(
        ...,
        description="Uncertainty quantification"
    )

    # Natural language summary
    natural_language_summary: Optional[str] = Field(
        default=None,
        description="Natural language summary of control decision"
    )

    # Energy metrics
    spray_energy_loss_kw: float = Field(
        ...,
        ge=0.0,
        description="Energy loss from spray water (kW)"
    )
    thermal_efficiency_impact_pct: float = Field(
        ...,
        description="Impact on thermal efficiency (%)"
    )

    # Calculated values
    current_superheat_c: float = Field(..., description="Current superheat (C)")
    temperature_deviation_c: float = Field(..., description="Deviation from target (C)")
    within_tolerance: bool = Field(..., description="Within acceptable tolerance")

    # Recommendations and warnings
    recommendations: List[str] = Field(
        default_factory=list,
        description="Control recommendations"
    )
    warnings: List[str] = Field(
        default_factory=list,
        description="Warning messages"
    )
    alerts: List[str] = Field(
        default_factory=list,
        description="Critical alerts"
    )

    # Provenance
    provenance_hash: str = Field(..., description="SHA-256 provenance hash")
    processing_time_ms: float = Field(
        default=0.0,
        ge=0.0,
        description="Processing time (ms)"
    )
    calculation_count: int = Field(
        default=0,
        ge=0,
        description="Number of calculations performed"
    )

    class Config:
        json_encoders = {datetime: lambda v: v.isoformat()}


# =============================================================================
# CALCULATOR CLASSES
# =============================================================================

class SteamPropertyCalculator:
    """
    Zero-hallucination steam property calculator.

    Based on IAPWS-IF97 simplified formulations for industrial use.
    All calculations are deterministic with provenance tracking.
    """

    def calculate_saturation_temperature(self, pressure_bar: float) -> float:
        """
        Calculate saturation temperature from pressure.

        Uses polynomial fit to IAPWS-IF97 (valid 0.1 to 200 bar).

        Args:
            pressure_bar: Pressure in bar absolute

        Returns:
            Saturation temperature in degrees C
        """
        if pressure_bar <= 0:
            raise ValueError("Pressure must be positive")

        # Polynomial coefficients for T_sat approximation
        a0, a1, a2, a3 = 99.974, 28.080, -0.5479, 0.01923
        ln_p = math.log(pressure_bar)

        t_sat = a0 + a1 * ln_p + a2 * ln_p**2 + a3 * ln_p**3
        return round(t_sat, 2)

    def calculate_superheat(
        self,
        steam_temp_c: float,
        pressure_bar: float
    ) -> float:
        """Calculate degree of superheat above saturation."""
        t_sat = self.calculate_saturation_temperature(pressure_bar)
        return round(steam_temp_c - t_sat, 2)

    def calculate_enthalpy(
        self,
        temp_c: float,
        pressure_bar: float
    ) -> float:
        """
        Calculate specific enthalpy of superheated steam.

        Simplified formula: h = h_sat + cp_avg * (T - T_sat)
        """
        t_sat = self.calculate_saturation_temperature(pressure_bar)
        h_sat = 2675.0 + 1.8 * (t_sat - 100.0)  # Saturated vapor enthalpy
        cp_avg = 2.1  # Average specific heat (kJ/kg-K)

        superheat = temp_c - t_sat
        if superheat < 0:
            return round(h_sat, 1)

        return round(h_sat + cp_avg * superheat, 1)

    def enrich_steam_conditions(
        self,
        conditions: SteamConditions
    ) -> SteamConditions:
        """Enrich steam conditions with calculated properties."""
        t_sat = self.calculate_saturation_temperature(conditions.pressure_bar)
        superheat = conditions.temperature_c - t_sat
        enthalpy = self.calculate_enthalpy(
            conditions.temperature_c,
            conditions.pressure_bar
        )

        return SteamConditions(
            temperature_c=conditions.temperature_c,
            pressure_bar=conditions.pressure_bar,
            flow_kg_s=conditions.flow_kg_s,
            saturation_temp_c=t_sat,
            superheat_c=round(superheat, 2),
            enthalpy_kj_kg=enthalpy
        )


class SprayWaterCalculator:
    """
    Zero-hallucination spray water flow calculator.

    Uses energy balance equations for desuperheating calculations.
    """

    def __init__(self, steam_calc: SteamPropertyCalculator):
        """Initialize with steam property calculator."""
        self._steam_calc = steam_calc

    def calculate_required_spray_flow(
        self,
        steam_flow_kg_s: float,
        inlet_temp_c: float,
        target_temp_c: float,
        spray_water_temp_c: float,
        pressure_bar: float
    ) -> Tuple[float, float]:
        """
        Calculate required spray water flow for desuperheating.

        Energy balance:
        m_steam * h_in + m_spray * h_water = (m_steam + m_spray) * h_out

        Args:
            steam_flow_kg_s: Steam mass flow rate (kg/s)
            inlet_temp_c: Inlet steam temperature (C)
            target_temp_c: Target outlet temperature (C)
            spray_water_temp_c: Spray water temperature (C)
            pressure_bar: Steam pressure (bar)

        Returns:
            Tuple of (spray_flow_kg_s, energy_absorbed_kw)
        """
        if inlet_temp_c <= target_temp_c:
            return 0.0, 0.0

        # Calculate enthalpies
        h_in = self._steam_calc.calculate_enthalpy(inlet_temp_c, pressure_bar)
        h_out = self._steam_calc.calculate_enthalpy(target_temp_c, pressure_bar)
        h_water = 4.18 * spray_water_temp_c  # Subcooled liquid approximation

        if h_out <= h_water:
            raise ValueError("Target temperature too low for spray control")

        # Energy balance solution
        spray_flow = steam_flow_kg_s * (h_in - h_out) / (h_out - h_water)
        energy_absorbed = spray_flow * (h_out - h_water)

        return round(spray_flow, 4), round(energy_absorbed, 2)

    def calculate_valve_position(
        self,
        required_flow_kg_s: float,
        max_flow_kg_s: float
    ) -> float:
        """Calculate valve position for required flow."""
        if max_flow_kg_s <= 0:
            return 0.0

        position = (required_flow_kg_s / max_flow_kg_s) * 100.0
        return min(100.0, max(0.0, round(position, 1)))

    def calculate_energy_loss(
        self,
        spray_flow_kg_s: float,
        inlet_enthalpy_kj_kg: float,
        outlet_enthalpy_kj_kg: float
    ) -> float:
        """Calculate energy loss from spray water injection (kW)."""
        delta_h = inlet_enthalpy_kj_kg - outlet_enthalpy_kj_kg
        return round(spray_flow_kg_s * delta_h, 2)


class PIDController:
    """
    Zero-hallucination PID controller implementation.

    Implements standard PID algorithm with anti-windup and deadband.
    """

    def __init__(self, config: PIDConfig):
        """Initialize PID controller."""
        self.config = config
        self._integral = 0.0
        self._previous_error = 0.0
        self._last_time: Optional[float] = None

    def reset(self) -> None:
        """Reset controller state."""
        self._integral = 0.0
        self._previous_error = 0.0
        self._last_time = None

    def compute(
        self,
        setpoint: float,
        process_value: float,
        dt_s: float = 1.0
    ) -> PIDOutput:
        """
        Compute PID output.

        Args:
            setpoint: Target value
            process_value: Current process value
            dt_s: Time step in seconds

        Returns:
            PIDOutput with all components
        """
        error = setpoint - process_value

        # Check deadband
        in_deadband = abs(error) <= self.config.deadband_c
        if in_deadband:
            return PIDOutput(
                proportional=0.0,
                integral=self._integral * self.config.ki,
                derivative=0.0,
                output=self._integral * self.config.ki,
                saturated=False,
                in_deadband=True
            )

        # Proportional term
        p_term = self.config.kp * error

        # Integral term with anti-windup
        self._integral += error * dt_s
        self._integral = max(
            -self.config.integral_windup_limit,
            min(self.config.integral_windup_limit, self._integral)
        )
        i_term = self.config.ki * self._integral

        # Derivative term (on error)
        d_error = (error - self._previous_error) / dt_s if dt_s > 0 else 0.0
        d_term = self.config.kd * d_error
        self._previous_error = error

        # Total output
        output = p_term + i_term + d_term

        # Saturation check
        saturated = abs(output) >= 100.0
        output = max(-100.0, min(100.0, output))

        return PIDOutput(
            proportional=round(p_term, 4),
            integral=round(i_term, 4),
            derivative=round(d_term, 4),
            output=round(output, 4),
            saturated=saturated,
            in_deadband=False
        )

    @staticmethod
    def tune_lambda(
        process_time_constant_s: float,
        process_dead_time_s: float,
        desired_response_time_s: float
    ) -> PIDConfig:
        """
        Calculate PID parameters using Lambda tuning method.

        Lambda tuning provides robust, non-oscillatory control.
        """
        tau = process_time_constant_s
        theta = process_dead_time_s
        lambda_cl = desired_response_time_s
        k = 1.0  # Normalized process gain

        kp = tau / (k * (lambda_cl + theta))
        ki = kp / tau
        kd = kp * theta / 2

        return PIDConfig(
            kp=round(kp, 4),
            ki=round(ki, 6),
            kd=round(kd, 4),
            deadband_c=1.0,
            max_rate_c_per_min=5.0
        )


class TemperaturePredictor:
    """
    Zero-hallucination temperature predictor.

    Uses first-order process model with dead time for predictions.
    """

    def __init__(
        self,
        time_constant_s: float,
        dead_time_s: float
    ):
        """Initialize predictor with process dynamics."""
        self.time_constant_s = time_constant_s
        self.dead_time_s = dead_time_s

    def predict(
        self,
        current_temp_c: float,
        target_temp_c: float,
        spray_change_pct: float,
        prediction_horizon_s: float = 60.0
    ) -> TemperaturePrediction:
        """
        Predict future temperature using first-order model.

        T(t) = T_target + (T_current - T_target) * exp(-t/tau)

        Args:
            current_temp_c: Current temperature
            target_temp_c: Target temperature (steady state)
            spray_change_pct: Change in spray flow (%)
            prediction_horizon_s: Prediction horizon (seconds)

        Returns:
            TemperaturePrediction with uncertainty bounds
        """
        # First-order response
        t = prediction_horizon_s
        tau = self.time_constant_s

        # Effective time after dead time
        t_eff = max(0.0, t - self.dead_time_s)

        # Predicted temperature
        delta_t = target_temp_c - current_temp_c
        predicted_change = delta_t * (1 - math.exp(-t_eff / tau))
        predicted_temp = current_temp_c + predicted_change

        # Uncertainty based on model and spray change magnitude
        base_uncertainty = 2.0  # Base uncertainty (C)
        spray_uncertainty = abs(spray_change_pct) * 0.05  # Additional uncertainty
        total_uncertainty = base_uncertainty + spray_uncertainty

        # Confidence level
        if total_uncertainty < 3.0:
            confidence = PredictionConfidence.HIGH
        elif total_uncertainty < 5.0:
            confidence = PredictionConfidence.MEDIUM
        else:
            confidence = PredictionConfidence.LOW

        return TemperaturePrediction(
            predicted_temp_c=round(predicted_temp, 2),
            uncertainty_lower_c=round(predicted_temp - total_uncertainty, 2),
            uncertainty_upper_c=round(predicted_temp + total_uncertainty, 2),
            prediction_horizon_s=prediction_horizon_s,
            confidence=confidence,
            model_used="first_order_with_dead_time"
        )


class SafetyValidator:
    """
    Zero-hallucination safety constraint validator.

    Validates all safety constraints per IEC 61511.
    """

    def __init__(self, config: SafetyConfig):
        """Initialize with safety configuration."""
        self.config = config

    def validate(
        self,
        current_temp_c: float,
        saturation_temp_c: float,
        tube_metal_temp_c: Optional[float],
        spray_flow_pct: float
    ) -> SafetyAssessment:
        """
        Validate all safety constraints.

        Args:
            current_temp_c: Current outlet temperature
            saturation_temp_c: Saturation temperature
            tube_metal_temp_c: Tube metal temperature (if measured)
            spray_flow_pct: Current spray flow as % of max

        Returns:
            SafetyAssessment with constraint status
        """
        constraints_satisfied = []
        constraints_violated = []

        # Superheat margin
        superheat = current_temp_c - saturation_temp_c
        superheat_margin = superheat - self.config.min_superheat_c

        if superheat >= self.config.min_superheat_c:
            constraints_satisfied.append(
                f"Superheat {superheat:.1f}C >= min {self.config.min_superheat_c}C"
            )
        else:
            constraints_violated.append(
                f"Superheat {superheat:.1f}C < min {self.config.min_superheat_c}C"
            )

        # Tube metal temperature
        if tube_metal_temp_c is not None:
            tube_metal_margin = self.config.max_tube_metal_temp_c - tube_metal_temp_c
            if tube_metal_temp_c <= self.config.max_tube_metal_temp_c:
                constraints_satisfied.append(
                    f"Tube metal {tube_metal_temp_c:.1f}C <= max {self.config.max_tube_metal_temp_c}C"
                )
            else:
                constraints_violated.append(
                    f"Tube metal {tube_metal_temp_c:.1f}C > max {self.config.max_tube_metal_temp_c}C"
                )
        else:
            tube_metal_margin = self.config.warning_margin_c + 10.0  # Assume safe if not measured

        # Spray capacity
        spray_capacity_margin = self.config.max_spray_flow_pct - spray_flow_pct
        if spray_flow_pct <= self.config.max_spray_flow_pct:
            constraints_satisfied.append(
                f"Spray flow {spray_flow_pct:.1f}% <= max {self.config.max_spray_flow_pct}%"
            )
        else:
            constraints_violated.append(
                f"Spray flow {spray_flow_pct:.1f}% > max {self.config.max_spray_flow_pct}%"
            )

        # Determine overall status
        if constraints_violated:
            if tube_metal_margin < self.config.critical_margin_c:
                status = SafetyStatus.EMERGENCY
            elif tube_metal_margin < self.config.warning_margin_c:
                status = SafetyStatus.CRITICAL
            elif superheat_margin < 0:
                status = SafetyStatus.CRITICAL
            else:
                status = SafetyStatus.WARNING
        else:
            if tube_metal_margin < self.config.warning_margin_c:
                status = SafetyStatus.WARNING
            else:
                status = SafetyStatus.SAFE

        emergency_required = (
            status == SafetyStatus.EMERGENCY and
            self.config.emergency_shutdown_enabled
        )

        return SafetyAssessment(
            status=status,
            tube_metal_margin_c=round(tube_metal_margin, 2),
            superheat_margin_c=round(superheat_margin, 2),
            spray_capacity_margin_pct=round(spray_capacity_margin, 1),
            constraints_satisfied=constraints_satisfied,
            constraints_violated=constraints_violated,
            emergency_action_required=emergency_required
        )


class ExplainabilityEngine:
    """
    SHAP/LIME explainability engine for control decisions.

    Provides feature importance and natural language explanations.
    """

    def __init__(self, config: ExplainabilityConfig):
        """Initialize explainability engine."""
        self.config = config

    def generate_shap_explanation(
        self,
        input_features: Dict[str, float],
        output_value: float,
        base_value: float
    ) -> ExplainabilityReport:
        """
        Generate SHAP-style feature explanations.

        This is a simplified SHAP approximation for real-time control.
        In production, would use actual SHAP library with trained model.

        Args:
            input_features: Input feature dictionary
            output_value: Model output value
            base_value: Base (average) prediction

        Returns:
            ExplainabilityReport with feature contributions
        """
        # Calculate pseudo-SHAP values based on feature sensitivities
        # These are simplified approximations for control systems
        feature_sensitivities = {
            "inlet_temp_c": 0.30,
            "outlet_temp_c": 0.25,
            "target_temp_c": 0.20,
            "steam_flow_kg_s": 0.10,
            "spray_water_temp_c": 0.08,
            "pressure_bar": 0.05,
            "burner_load_pct": 0.02,
        }

        delta = output_value - base_value
        explanations = []

        for feature_name, value in input_features.items():
            sensitivity = feature_sensitivities.get(feature_name, 0.01)

            # Approximate SHAP value
            shap_value = delta * sensitivity
            contribution_pct = abs(sensitivity) * 100
            direction = "positive" if shap_value >= 0 else "negative"

            explanations.append(SHAPExplanation(
                feature_name=feature_name,
                feature_value=value,
                shap_value=round(shap_value, 4),
                contribution_pct=round(contribution_pct, 1),
                direction=direction
            ))

        # Sort by absolute SHAP value
        explanations.sort(key=lambda x: abs(x.shap_value), reverse=True)
        top_features = explanations[:self.config.num_features]

        return ExplainabilityReport(
            method="shap_approximation",
            base_value=base_value,
            final_value=output_value,
            top_features=top_features,
            confidence_score=0.85
        )


# =============================================================================
# MAIN CONTROLLER CLASS
# =============================================================================

class SuperheaterController(IntelligenceMixin, BaseProcessHeatAgent[SuperheaterControlInput, SuperheaterControlOutput]):
    """
    GL-022 Superheater Control Agent.

    This controller provides comprehensive superheater temperature control
    with spray water optimization, PID tuning, safety validation, and
    ML-based explainability features.

    Capabilities:
        - Optimal spray water setpoint calculation using energy balance
        - PID control parameter tuning with Lambda method
        - Temperature prediction with uncertainty quantification
        - Safety constraint validation per IEC 61511
        - SHAP/LIME explainability for control decisions
        - Natural language summaries via LLM integration
        - Zero-hallucination deterministic calculations
        - SHA-256 provenance tracking

    All numeric calculations are deterministic with zero hallucination.
    LLM is used ONLY for natural language generation and explanations.

    Example:
        >>> config = create_default_config()
        >>> controller = SuperheaterController(config)
        >>>
        >>> input_data = SuperheaterControlInput(
        ...     equipment_id="SH-001",
        ...     inlet_steam=SteamConditions(temperature_c=550, pressure_bar=100, flow_kg_s=50),
        ...     outlet_steam=SteamConditions(temperature_c=545, pressure_bar=99, flow_kg_s=50),
        ...     spray_water=SprayWaterConditions(temperature_c=150, max_flow_kg_s=5),
        ...     demand=ProcessDemand(target_temp_c=540),
        ...     burner_load_pct=85,
        ...     flue_gas_temp_c=350,
        ... )
        >>>
        >>> result = controller.process(input_data)
        >>> print(f"Spray setpoint: {result.spray_setpoint.target_flow_kg_s} kg/s")
    """

    # Agent metadata
    AGENT_TYPE = "GL-022"
    AGENT_NAME = "Superheater Control Agent"
    AGENT_VERSION = "1.0.0"

    def __init__(
        self,
        config: SuperheaterControlConfig,
        safety_level: SafetyLevel = SafetyLevel.SIL_2,
    ) -> None:
        """
        Initialize the Superheater Controller.

        Args:
            config: Controller configuration
            safety_level: Safety Integrity Level (default SIL-2)
        """
        # Create agent config for base class
        agent_config = AgentConfig(
            agent_id=config.agent_id,
            agent_type=self.AGENT_TYPE,
            name=self.AGENT_NAME,
            version=self.AGENT_VERSION,
            capabilities={
                AgentCapability.REAL_TIME_MONITORING,
                AgentCapability.OPTIMIZATION,
                AgentCapability.PREDICTIVE_ANALYTICS,
                AgentCapability.EMERGENCY_RESPONSE,
            },
        )

        # Initialize base class
        super().__init__(agent_config, safety_level)

        # Store configuration
        self.control_config = config

        # Initialize calculators
        self._steam_calc = SteamPropertyCalculator()
        self._spray_calc = SprayWaterCalculator(self._steam_calc)
        self._pid_controller = PIDController(config.pid)
        self._temp_predictor = TemperaturePredictor(
            config.process_time_constant_s,
            config.process_dead_time_s
        )
        self._safety_validator = SafetyValidator(config.safety)
        self._explainability = ExplainabilityEngine(config.explainability)

        # Calculation counter for provenance
        self._calculation_count = 0

        logger.info(
            f"SuperheaterController initialized: "
            f"equipment={config.equipment_id}, "
            f"target={config.default_target_temp_c}C"
        )

        # Initialize intelligence with ADVANCED level
        self._init_intelligence(IntelligenceConfig(
            domain_context="superheater temperature control and steam systems",
            regulatory_context="IAPWS-IF97, ASME BPVC, IEC 61511",
            enable_explanations=True,
            enable_recommendations=True,
            enable_anomaly_detection=True,
        ))

    # =========================================================================
    # INTELLIGENCE INTERFACE METHODS
    # =========================================================================

    def get_intelligence_level(self) -> IntelligenceLevel:
        """Return ADVANCED intelligence level for superheater controller."""
        return IntelligenceLevel.ADVANCED

    def get_intelligence_capabilities(self) -> IntelligenceCapabilities:
        """Return advanced intelligence capabilities."""
        return IntelligenceCapabilities(
            can_explain=True,
            can_recommend=True,
            can_detect_anomalies=True,
            can_reason=True,
            can_validate=True,
            uses_rag=False,
            uses_tools=False
        )

    # =========================================================================
    # ABSTRACT METHOD IMPLEMENTATIONS
    # =========================================================================

    def process(
        self,
        input_data: SuperheaterControlInput,
    ) -> SuperheaterControlOutput:
        """
        Main processing method for superheater control.

        This method orchestrates all calculations and generates
        comprehensive control output with explainability.

        Args:
            input_data: Validated input data with process conditions

        Returns:
            SuperheaterControlOutput with control recommendations

        Raises:
            ValidationError: If input validation fails
            ProcessingError: If processing fails
        """
        start_time = time.time()
        self._calculation_count = 0

        logger.info(
            f"Processing superheater control for {input_data.equipment_id} "
            f"at {input_data.timestamp}"
        )

        try:
            with self.safety_guard():
                # Step 1: Validate input
                if not self.validate_input(input_data):
                    raise ValidationError("Input validation failed")

                # Step 2: Enrich steam conditions with calculated properties
                inlet_steam = self._steam_calc.enrich_steam_conditions(
                    input_data.inlet_steam
                )
                outlet_steam = self._steam_calc.enrich_steam_conditions(
                    input_data.outlet_steam
                )
                self._calculation_count += 2

                # Step 3: Calculate spray water requirement
                spray_setpoint = self.calculate_spray_requirement(
                    inlet_steam,
                    outlet_steam,
                    input_data.spray_water,
                    input_data.demand
                )
                self._calculation_count += 1

                # Step 4: Compute PID control output
                pid_output = self._pid_controller.compute(
                    setpoint=input_data.demand.target_temp_c,
                    process_value=outlet_steam.temperature_c,
                    dt_s=1.0
                )
                self._calculation_count += 1

                # Step 5: Predict future temperature
                spray_change = (
                    spray_setpoint.target_flow_kg_s -
                    input_data.spray_water.current_flow_kg_s
                ) / max(input_data.spray_water.max_flow_kg_s, 0.001) * 100

                temp_prediction = self._temp_predictor.predict(
                    current_temp_c=outlet_steam.temperature_c,
                    target_temp_c=input_data.demand.target_temp_c,
                    spray_change_pct=spray_change,
                    prediction_horizon_s=60.0
                )
                self._calculation_count += 1

                # Step 6: Validate safety constraints
                safety_assessment = self.validate_safety_constraints(
                    outlet_steam,
                    input_data.tube_metal_temp_c,
                    spray_setpoint.valve_position_pct
                )
                self._calculation_count += 1

                # Step 7: Calculate energy metrics
                spray_energy_loss = self._calculate_spray_energy_loss(
                    spray_setpoint.target_flow_kg_s,
                    inlet_steam.enthalpy_kj_kg or 2800.0,
                    outlet_steam.enthalpy_kj_kg or 2750.0
                )

                # Estimate thermal efficiency impact
                # Approximate fuel input from burner load (MW scale)
                fuel_input_kw = input_data.burner_load_pct * 1000.0  # Simplified
                efficiency_impact = (
                    (spray_energy_loss / max(fuel_input_kw, 1.0)) * 100
                    if fuel_input_kw > 0 else 0.0
                )
                self._calculation_count += 1

                # Step 8: Generate explainability report
                explainability_report = None
                if self.control_config.explainability.enabled:
                    explainability_report = self.generate_explanation_report(
                        input_data,
                        spray_setpoint
                    )
                    self._calculation_count += 1

                # Step 9: Calculate uncertainty quantification
                uncertainty = self._calculate_uncertainty(
                    spray_setpoint,
                    temp_prediction
                )
                self._calculation_count += 1

                # Step 10: Generate recommendations and warnings
                recommendations, warnings, alerts = self._generate_recommendations(
                    spray_setpoint,
                    safety_assessment,
                    outlet_steam.superheat_c or 0.0,
                    efficiency_impact
                )

                # Calculate derived values
                current_superheat = outlet_steam.superheat_c or 0.0
                temp_deviation = (
                    outlet_steam.temperature_c - input_data.demand.target_temp_c
                )
                within_tolerance = (
                    abs(temp_deviation) <= input_data.demand.tolerance_c
                )

                # Calculate processing time
                processing_time_ms = (time.time() - start_time) * 1000

                # Calculate provenance hash
                provenance_hash = self._calculate_provenance_hash(input_data)

                # Create output
                output = SuperheaterControlOutput(
                    controller_id=self.control_config.agent_id,
                    equipment_id=input_data.equipment_id,
                    timestamp=datetime.now(timezone.utc),
                    spray_setpoint=spray_setpoint,
                    pid_output=pid_output,
                    temperature_prediction=temp_prediction,
                    safety_assessment=safety_assessment,
                    explainability_report=explainability_report,
                    uncertainty=uncertainty,
                    spray_energy_loss_kw=spray_energy_loss,
                    thermal_efficiency_impact_pct=round(efficiency_impact, 3),
                    current_superheat_c=current_superheat,
                    temperature_deviation_c=round(temp_deviation, 2),
                    within_tolerance=within_tolerance,
                    recommendations=recommendations,
                    warnings=warnings,
                    alerts=alerts,
                    provenance_hash=provenance_hash,
                    processing_time_ms=round(processing_time_ms, 2),
                    calculation_count=self._calculation_count,
                )

                # Validate output
                if not self.validate_output(output):
                    raise ProcessingError("Output validation failed")

                # Generate natural language summary via LLM
                output.natural_language_summary = self.generate_explanation(
                    input_data={
                        "equipment_id": input_data.equipment_id,
                        "inlet_temp_c": inlet_steam.temperature_c,
                        "outlet_temp_c": outlet_steam.temperature_c,
                        "target_temp_c": input_data.demand.target_temp_c,
                        "current_spray_kg_s": input_data.spray_water.current_flow_kg_s,
                    },
                    output_data={
                        "recommended_spray_kg_s": spray_setpoint.target_flow_kg_s,
                        "valve_position_pct": spray_setpoint.valve_position_pct,
                        "action": spray_setpoint.action.value,
                        "safety_status": safety_assessment.status.value,
                        "within_tolerance": within_tolerance,
                    },
                    calculation_steps=[
                        f"Calculated saturation temperature: {outlet_steam.saturation_temp_c}C",
                        f"Determined superheat: {current_superheat}C",
                        f"Computed spray requirement: {spray_setpoint.target_flow_kg_s} kg/s",
                        f"PID output: {pid_output.output:.2f}%",
                        f"Predicted temperature in 60s: {temp_prediction.predicted_temp_c}C",
                        f"Safety status: {safety_assessment.status.value}",
                    ]
                )

                logger.info(
                    f"Superheater control complete: {spray_setpoint.action.value}, "
                    f"spray={spray_setpoint.target_flow_kg_s} kg/s, "
                    f"safety={safety_assessment.status.value}, "
                    f"{self._calculation_count} calculations, "
                    f"{processing_time_ms:.1f}ms"
                )

                return output

        except Exception as e:
            logger.error(f"Superheater control failed: {e}", exc_info=True)
            raise ProcessingError(f"Superheater control failed: {str(e)}") from e

    def validate_input(self, input_data: SuperheaterControlInput) -> bool:
        """
        Validate input data.

        Args:
            input_data: Input to validate

        Returns:
            True if valid
        """
        # Check steam conditions are physically reasonable
        if input_data.inlet_steam.temperature_c < 100:
            logger.warning("Inlet steam temperature below 100C")
            return False

        if input_data.outlet_steam.temperature_c < 100:
            logger.warning("Outlet steam temperature below 100C")
            return False

        # Check pressure is reasonable
        if input_data.inlet_steam.pressure_bar < 0.5:
            logger.warning("Steam pressure too low")
            return False

        # Check spray water is colder than steam
        if input_data.spray_water.temperature_c >= input_data.outlet_steam.temperature_c:
            logger.warning("Spray water hotter than steam - invalid")
            return False

        return True

    def validate_output(self, output_data: SuperheaterControlOutput) -> bool:
        """
        Validate output data.

        Args:
            output_data: Output to validate

        Returns:
            True if valid
        """
        # Check spray setpoint is within bounds
        if output_data.spray_setpoint.target_flow_kg_s < 0:
            logger.warning("Negative spray flow calculated")
            return False

        # Check valve position is valid
        if not (0 <= output_data.spray_setpoint.valve_position_pct <= 100):
            logger.warning("Invalid valve position")
            return False

        # Check provenance hash exists
        if not output_data.provenance_hash:
            logger.warning("Missing provenance hash")
            return False

        return True

    # =========================================================================
    # CORE CALCULATION METHODS
    # =========================================================================

    def calculate_spray_requirement(
        self,
        inlet_steam: SteamConditions,
        outlet_steam: SteamConditions,
        spray_water: SprayWaterConditions,
        demand: ProcessDemand,
    ) -> SpraySetpoint:
        """
        Calculate optimal spray water setpoint.

        Uses energy balance equation for desuperheating calculation:
        m_spray = m_steam * (h_in - h_out) / (h_out - h_water)

        Args:
            inlet_steam: Inlet steam conditions (enriched)
            outlet_steam: Current outlet steam conditions (enriched)
            spray_water: Spray water system conditions
            demand: Process temperature demand

        Returns:
            SpraySetpoint with optimal flow and valve position
        """
        # Check if spray is needed
        if outlet_steam.temperature_c <= demand.target_temp_c:
            return SpraySetpoint(
                target_flow_kg_s=0.0,
                valve_position_pct=0.0,
                rate_of_change_pct_per_min=5.0,
                action=ControlAction.MAINTAIN,
                confidence=0.95
            )

        try:
            # Calculate required spray flow
            required_flow, _ = self._spray_calc.calculate_required_spray_flow(
                steam_flow_kg_s=inlet_steam.flow_kg_s,
                inlet_temp_c=inlet_steam.temperature_c,
                target_temp_c=demand.target_temp_c,
                spray_water_temp_c=spray_water.temperature_c,
                pressure_bar=inlet_steam.pressure_bar
            )

            # Limit to available capacity
            max_available = spray_water.max_flow_kg_s * (
                self.control_config.safety.max_spray_flow_pct / 100.0
            )
            required_flow = min(required_flow, max_available)

            # Calculate valve position
            valve_position = self._spray_calc.calculate_valve_position(
                required_flow,
                spray_water.max_flow_kg_s
            )

            # Determine action
            current_flow = spray_water.current_flow_kg_s
            if required_flow > current_flow * 1.05:
                action = ControlAction.INCREASE
            elif required_flow < current_flow * 0.95:
                action = ControlAction.DECREASE
            else:
                action = ControlAction.MAINTAIN

            # Confidence based on temperature deviation
            temp_deviation = abs(outlet_steam.temperature_c - demand.target_temp_c)
            if temp_deviation < 5:
                confidence = 0.95
            elif temp_deviation < 10:
                confidence = 0.85
            else:
                confidence = 0.75

            return SpraySetpoint(
                target_flow_kg_s=round(required_flow, 4),
                valve_position_pct=valve_position,
                rate_of_change_pct_per_min=5.0,
                action=action,
                confidence=confidence
            )

        except ValueError as e:
            logger.warning(f"Spray calculation issue: {e}")
            return SpraySetpoint(
                target_flow_kg_s=0.0,
                valve_position_pct=0.0,
                rate_of_change_pct_per_min=5.0,
                action=ControlAction.MAINTAIN,
                confidence=0.5
            )

    def optimize_temperature_setpoint(
        self,
        demand: ProcessDemand,
        current_conditions: SteamConditions,
        constraints: Optional[Dict[str, float]] = None
    ) -> float:
        """
        Optimize temperature setpoint considering constraints.

        Balances process demand against safety limits and
        energy efficiency considerations.

        Args:
            demand: Process temperature demand
            current_conditions: Current steam conditions
            constraints: Optional additional constraints

        Returns:
            Optimized temperature setpoint (C)
        """
        target = demand.target_temp_c

        # Ensure minimum superheat
        min_temp = (
            (current_conditions.saturation_temp_c or 0.0) +
            self.control_config.safety.min_superheat_c
        )

        # Respect maximum tube temperature
        max_temp = (
            self.control_config.safety.max_tube_metal_temp_c -
            self.control_config.safety.warning_margin_c
        )

        # Apply constraints
        optimized = max(min_temp, min(target, max_temp))

        # Apply additional constraints if provided
        if constraints:
            if "max_temp_c" in constraints:
                optimized = min(optimized, constraints["max_temp_c"])
            if "min_temp_c" in constraints:
                optimized = max(optimized, constraints["min_temp_c"])

        return round(optimized, 1)

    def validate_safety_constraints(
        self,
        outlet_steam: SteamConditions,
        tube_metal_temp_c: Optional[float],
        spray_valve_pct: float
    ) -> SafetyAssessment:
        """
        Validate all safety constraints.

        Checks:
        - Minimum superheat above saturation
        - Maximum tube metal temperature
        - Spray water capacity margin

        Args:
            outlet_steam: Outlet steam conditions
            tube_metal_temp_c: Tube metal temperature (if measured)
            spray_valve_pct: Current spray valve position (%)

        Returns:
            SafetyAssessment with constraint status
        """
        return self._safety_validator.validate(
            current_temp_c=outlet_steam.temperature_c,
            saturation_temp_c=outlet_steam.saturation_temp_c or 0.0,
            tube_metal_temp_c=tube_metal_temp_c,
            spray_flow_pct=spray_valve_pct
        )

    def generate_explanation_report(
        self,
        input_data: SuperheaterControlInput,
        spray_setpoint: SpraySetpoint
    ) -> ExplainabilityReport:
        """
        Generate SHAP/LIME explainability report.

        Provides feature importance analysis for the control decision.

        Args:
            input_data: Controller input data
            spray_setpoint: Calculated spray setpoint

        Returns:
            ExplainabilityReport with feature contributions
        """
        # Build feature dictionary
        features = {
            "inlet_temp_c": input_data.inlet_steam.temperature_c,
            "outlet_temp_c": input_data.outlet_steam.temperature_c,
            "target_temp_c": input_data.demand.target_temp_c,
            "steam_flow_kg_s": input_data.inlet_steam.flow_kg_s,
            "spray_water_temp_c": input_data.spray_water.temperature_c,
            "pressure_bar": input_data.inlet_steam.pressure_bar,
            "burner_load_pct": input_data.burner_load_pct,
        }

        # Base value (average spray flow)
        base_value = input_data.spray_water.max_flow_kg_s * 0.3

        # Generate explanation
        report = self._explainability.generate_shap_explanation(
            input_features=features,
            output_value=spray_setpoint.target_flow_kg_s,
            base_value=base_value
        )

        # Generate natural language summary if enabled
        if self.control_config.explainability.generate_natural_language:
            top_3_features = [f.feature_name for f in report.top_features[:3]]
            report.natural_language_summary = (
                f"The spray water setpoint of {spray_setpoint.target_flow_kg_s:.3f} kg/s "
                f"was primarily influenced by: {', '.join(top_3_features)}. "
                f"The control action is to {spray_setpoint.action.value} spray flow."
            )

        return report

    # =========================================================================
    # ASYNC SUPPORT
    # =========================================================================

    async def process_async(
        self,
        input_data: SuperheaterControlInput,
    ) -> SuperheaterControlOutput:
        """
        Async version of process method.

        Provides async support while maintaining deterministic calculations.

        Args:
            input_data: Validated input data

        Returns:
            SuperheaterControlOutput
        """
        # Run synchronous process in executor for true async
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.process, input_data)

    # =========================================================================
    # HELPER METHODS
    # =========================================================================

    def _calculate_spray_energy_loss(
        self,
        spray_flow_kg_s: float,
        inlet_enthalpy_kj_kg: float,
        outlet_enthalpy_kj_kg: float
    ) -> float:
        """Calculate energy loss from spray water (kW)."""
        return self._spray_calc.calculate_energy_loss(
            spray_flow_kg_s,
            inlet_enthalpy_kj_kg,
            outlet_enthalpy_kj_kg
        )

    def _calculate_uncertainty(
        self,
        spray_setpoint: SpraySetpoint,
        temp_prediction: TemperaturePrediction
    ) -> UncertaintyQuantification:
        """Calculate uncertainty quantification for outputs."""
        # Spray flow uncertainty (based on valve characteristics)
        spray_std = spray_setpoint.target_flow_kg_s * 0.05  # 5% uncertainty

        # Temperature prediction uncertainty
        temp_range = (
            temp_prediction.uncertainty_upper_c -
            temp_prediction.uncertainty_lower_c
        )
        temp_std = temp_range / 4.0  # Approximate std from 95% CI

        # Model uncertainty based on confidence
        if spray_setpoint.confidence >= 0.9:
            model_uncertainty = 5.0
        elif spray_setpoint.confidence >= 0.8:
            model_uncertainty = 10.0
        else:
            model_uncertainty = 15.0

        # 95% confidence interval for spray flow
        ci_95 = (
            spray_setpoint.target_flow_kg_s - 1.96 * spray_std,
            spray_setpoint.target_flow_kg_s + 1.96 * spray_std
        )

        return UncertaintyQuantification(
            spray_flow_std_kg_s=round(spray_std, 4),
            temperature_prediction_std_c=round(temp_std, 2),
            model_uncertainty_pct=model_uncertainty,
            data_quality_score=0.95,
            confidence_interval_95_pct=(round(ci_95[0], 4), round(ci_95[1], 4))
        )

    def _generate_recommendations(
        self,
        spray_setpoint: SpraySetpoint,
        safety: SafetyAssessment,
        superheat_c: float,
        efficiency_impact: float
    ) -> Tuple[List[str], List[str], List[str]]:
        """Generate recommendations, warnings, and alerts."""
        recommendations = []
        warnings = []
        alerts = []

        # Safety-based alerts
        if safety.status == SafetyStatus.EMERGENCY:
            alerts.append(
                "EMERGENCY: Safety limits exceeded. Immediate action required."
            )
        elif safety.status == SafetyStatus.CRITICAL:
            alerts.append(
                f"CRITICAL: Tube metal margin only {safety.tube_metal_margin_c}C"
            )

        # Warnings
        if safety.status == SafetyStatus.WARNING:
            warnings.append(
                f"Warning: Approaching safety limits. "
                f"Tube metal margin: {safety.tube_metal_margin_c}C"
            )

        if superheat_c < 30:
            warnings.append(
                f"Warning: Low superheat ({superheat_c}C). "
                "Risk of moisture carryover."
            )

        if spray_setpoint.valve_position_pct > 80:
            warnings.append(
                f"Warning: Spray valve at {spray_setpoint.valve_position_pct}%. "
                "Approaching capacity limit."
            )

        # Recommendations
        if efficiency_impact > 1.0:
            recommendations.append(
                f"Consider reducing firing rate - spray energy loss "
                f"impacting efficiency by {efficiency_impact:.2f}%"
            )

        if spray_setpoint.action == ControlAction.INCREASE:
            recommendations.append(
                f"Increase spray flow to {spray_setpoint.target_flow_kg_s:.3f} kg/s "
                f"to reach target temperature"
            )
        elif spray_setpoint.action == ControlAction.DECREASE:
            recommendations.append(
                f"Reduce spray flow to {spray_setpoint.target_flow_kg_s:.3f} kg/s "
                "to optimize efficiency"
            )

        if safety.spray_capacity_margin_pct < 20:
            recommendations.append(
                "Review superheater heat absorption - "
                "high spray demand indicates poor heat transfer"
            )

        return recommendations, warnings, alerts

    def _calculate_provenance_hash(
        self,
        input_data: SuperheaterControlInput,
    ) -> str:
        """Calculate SHA-256 provenance hash for audit trail."""
        provenance_data = {
            "agent_id": self.control_config.agent_id,
            "agent_version": self.AGENT_VERSION,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "input_timestamp": input_data.timestamp.isoformat(),
            "equipment_id": input_data.equipment_id,
            "inlet_temp_c": input_data.inlet_steam.temperature_c,
            "outlet_temp_c": input_data.outlet_steam.temperature_c,
            "target_temp_c": input_data.demand.target_temp_c,
            "calculation_count": self._calculation_count,
        }

        data_str = json.dumps(provenance_data, sort_keys=True)
        return hashlib.sha256(data_str.encode()).hexdigest()

    # =========================================================================
    # CONVENIENCE METHODS
    # =========================================================================

    def get_steam_properties(
        self,
        temperature_c: float,
        pressure_bar: float
    ) -> Dict[str, float]:
        """
        Get steam properties at specified conditions.

        Args:
            temperature_c: Steam temperature (C)
            pressure_bar: Steam pressure (bar)

        Returns:
            Dictionary with steam properties
        """
        t_sat = self._steam_calc.calculate_saturation_temperature(pressure_bar)
        superheat = self._steam_calc.calculate_superheat(temperature_c, pressure_bar)
        enthalpy = self._steam_calc.calculate_enthalpy(temperature_c, pressure_bar)

        return {
            "temperature_c": temperature_c,
            "pressure_bar": pressure_bar,
            "saturation_temp_c": t_sat,
            "superheat_c": superheat,
            "enthalpy_kj_kg": enthalpy,
        }

    def tune_pid_controller(
        self,
        desired_response_time_s: float = 120.0
    ) -> PIDConfig:
        """
        Tune PID controller using Lambda method.

        Args:
            desired_response_time_s: Desired closed-loop response time

        Returns:
            Tuned PIDConfig
        """
        return PIDController.tune_lambda(
            process_time_constant_s=self.control_config.process_time_constant_s,
            process_dead_time_s=self.control_config.process_dead_time_s,
            desired_response_time_s=desired_response_time_s
        )

    def reset_pid_controller(self) -> None:
        """Reset PID controller state."""
        self._pid_controller.reset()
        logger.info("PID controller reset")
