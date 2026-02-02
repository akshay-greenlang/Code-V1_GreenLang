"""
GL-014 EXCHANGERPRO - API Schemas

Pydantic models for REST API request/response validation.

Provides:
- ComputeKPIsRequest/Response for thermal calculations
- FoulingPredictionRequest/Response for fouling forecasts
- CleaningOptimizationRequest/Response for schedule optimization
- WhatIfRequest/Response for scenario analysis
- ExplanationResponse for explainability endpoints

All responses include:
- computation_hash for traceability
- timestamp (UTC)
- agent_version
- warnings array
"""

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, Field, field_validator, model_validator
import hashlib


# =============================================================================
# Constants
# =============================================================================

AGENT_VERSION = "1.0.0"
AGENT_ID = "GL-014"
AGENT_NAME = "EXCHANGERPRO"


# =============================================================================
# Enums
# =============================================================================

class ExchangerType(str, Enum):
    """Heat exchanger type classification per TEMA."""
    SHELL_AND_TUBE = "shell_and_tube"
    PLATE = "plate"
    PLATE_FIN = "plate_fin"
    AIR_COOLED = "air_cooled"
    DOUBLE_PIPE = "double_pipe"
    SPIRAL = "spiral"


class FlowArrangement(str, Enum):
    """Flow arrangement in heat exchanger."""
    COUNTER_FLOW = "counter_flow"
    PARALLEL_FLOW = "parallel_flow"
    CROSS_FLOW = "cross_flow"
    MULTI_PASS = "multi_pass"


class FluidPhase(str, Enum):
    """Fluid phase classification."""
    LIQUID = "liquid"
    GAS = "gas"
    TWO_PHASE = "two_phase"
    CONDENSING = "condensing"
    BOILING = "boiling"


class FoulingMechanism(str, Enum):
    """Fouling mechanism classification."""
    PARTICULATE = "particulate"
    BIOLOGICAL = "biological"
    CRYSTALLIZATION = "crystallization"
    CORROSION = "corrosion"
    CHEMICAL_REACTION = "chemical_reaction"
    COMBINED = "combined"


class OptimizationObjective(str, Enum):
    """Cleaning schedule optimization objective."""
    MINIMIZE_COST = "minimize_cost"
    MAXIMIZE_AVAILABILITY = "maximize_availability"
    MINIMIZE_ENERGY_LOSS = "minimize_energy_loss"
    BALANCE_COST_AVAILABILITY = "balance_cost_availability"


class ExplanationType(str, Enum):
    """Type of explanation requested."""
    LIME = "lime"
    SHAP = "shap"
    FEATURE_IMPORTANCE = "feature_importance"
    CAUSAL = "causal"
    NATURAL_LANGUAGE = "natural_language"


# =============================================================================
# Base Response Model
# =============================================================================

class BaseAPIResponse(BaseModel):
    """Base response model with standard traceability fields."""

    computation_hash: str = Field(
        ...,
        description="SHA-256 hash of computation inputs for traceability"
    )
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="UTC timestamp of computation"
    )
    agent_version: str = Field(
        default=AGENT_VERSION,
        description="Version of the GL-014 agent"
    )
    agent_id: str = Field(
        default=AGENT_ID,
        description="Agent identifier"
    )
    warnings: List[str] = Field(
        default_factory=list,
        description="Warning messages from computation"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "computation_hash": "a1b2c3d4e5f6...",
                "timestamp": "2025-12-27T10:30:00Z",
                "agent_version": "1.0.0",
                "agent_id": "GL-014",
                "warnings": []
            }
        }


# =============================================================================
# Fluid Stream Models
# =============================================================================

class FluidStreamInput(BaseModel):
    """Input model for a fluid stream (hot or cold side)."""

    stream_id: str = Field(
        ...,
        description="Unique identifier for the stream",
        min_length=1,
        max_length=64
    )
    fluid_name: str = Field(
        "water",
        description="Working fluid name (e.g., water, oil, air)"
    )
    phase: FluidPhase = Field(
        FluidPhase.LIQUID,
        description="Fluid phase"
    )
    inlet_temperature_C: float = Field(
        ...,
        description="Inlet temperature in Celsius",
        ge=-273.15,
        le=2000.0
    )
    outlet_temperature_C: float = Field(
        ...,
        description="Outlet temperature in Celsius",
        ge=-273.15,
        le=2000.0
    )
    mass_flow_rate_kg_s: float = Field(
        ...,
        description="Mass flow rate in kg/s",
        gt=0.0,
        le=10000.0
    )
    specific_heat_kJ_kgK: float = Field(
        4.186,
        description="Specific heat capacity in kJ/(kg*K)",
        gt=0.0,
        le=100.0
    )
    pressure_kPa: float = Field(
        101.325,
        description="Operating pressure in kPa",
        gt=0.0
    )
    density_kg_m3: Optional[float] = Field(
        None,
        description="Fluid density in kg/m3",
        gt=0.0
    )
    viscosity_Pa_s: Optional[float] = Field(
        None,
        description="Dynamic viscosity in Pa*s",
        gt=0.0
    )
    thermal_conductivity_W_mK: Optional[float] = Field(
        None,
        description="Thermal conductivity in W/(m*K)",
        gt=0.0
    )

    @field_validator("outlet_temperature_C")
    @classmethod
    def validate_temperature_difference(cls, v, info):
        """Ensure outlet temperature differs from inlet."""
        if "inlet_temperature_C" in info.data:
            inlet = info.data["inlet_temperature_C"]
            if abs(v - inlet) < 0.01:
                raise ValueError(
                    "Outlet temperature must differ from inlet temperature"
                )
        return v


# =============================================================================
# Compute KPIs Models
# =============================================================================

class ComputeKPIsRequest(BaseModel):
    """Request model for computing thermal KPIs."""

    exchanger_id: str = Field(
        ...,
        description="Unique exchanger identifier",
        min_length=1,
        max_length=64
    )
    exchanger_type: ExchangerType = Field(
        ExchangerType.SHELL_AND_TUBE,
        description="Heat exchanger type"
    )
    flow_arrangement: FlowArrangement = Field(
        FlowArrangement.COUNTER_FLOW,
        description="Flow arrangement"
    )
    hot_stream: FluidStreamInput = Field(
        ...,
        description="Hot side fluid stream data"
    )
    cold_stream: FluidStreamInput = Field(
        ...,
        description="Cold side fluid stream data"
    )
    design_area_m2: Optional[float] = Field(
        None,
        description="Design heat transfer area in m2",
        gt=0.0
    )
    design_U_W_m2K: Optional[float] = Field(
        None,
        description="Design overall heat transfer coefficient W/(m2*K)",
        gt=0.0
    )
    fouling_factor_hot_m2K_W: float = Field(
        0.0001,
        description="Hot side fouling factor in m2*K/W",
        ge=0.0
    )
    fouling_factor_cold_m2K_W: float = Field(
        0.0001,
        description="Cold side fouling factor in m2*K/W",
        ge=0.0
    )
    tube_count: Optional[int] = Field(
        None,
        description="Number of tubes (shell-and-tube)",
        gt=0
    )
    tube_length_m: Optional[float] = Field(
        None,
        description="Tube length in meters",
        gt=0.0
    )
    tube_od_mm: Optional[float] = Field(
        None,
        description="Tube outer diameter in mm",
        gt=0.0
    )
    tube_id_mm: Optional[float] = Field(
        None,
        description="Tube inner diameter in mm",
        gt=0.0
    )
    shell_id_mm: Optional[float] = Field(
        None,
        description="Shell inner diameter in mm",
        gt=0.0
    )
    baffle_spacing_mm: Optional[float] = Field(
        None,
        description="Baffle spacing in mm",
        gt=0.0
    )

    @model_validator(mode="after")
    def validate_temperature_driving_force(self):
        """Ensure valid temperature driving force exists."""
        hot_in = self.hot_stream.inlet_temperature_C
        hot_out = self.hot_stream.outlet_temperature_C
        cold_in = self.cold_stream.inlet_temperature_C
        cold_out = self.cold_stream.outlet_temperature_C

        # Hot stream should cool down
        if hot_out >= hot_in:
            raise ValueError(
                "Hot stream outlet must be cooler than inlet"
            )

        # Cold stream should heat up
        if cold_out <= cold_in:
            raise ValueError(
                "Cold stream outlet must be warmer than inlet"
            )

        # Check for temperature cross
        if hot_out < cold_in:
            # Temperature cross - may be valid for multi-pass
            pass

        return self

    class Config:
        json_schema_extra = {
            "example": {
                "exchanger_id": "HX-001",
                "exchanger_type": "shell_and_tube",
                "flow_arrangement": "counter_flow",
                "hot_stream": {
                    "stream_id": "HOT-001",
                    "fluid_name": "oil",
                    "phase": "liquid",
                    "inlet_temperature_C": 120.0,
                    "outlet_temperature_C": 60.0,
                    "mass_flow_rate_kg_s": 5.0,
                    "specific_heat_kJ_kgK": 2.1,
                    "pressure_kPa": 200.0
                },
                "cold_stream": {
                    "stream_id": "COLD-001",
                    "fluid_name": "water",
                    "phase": "liquid",
                    "inlet_temperature_C": 25.0,
                    "outlet_temperature_C": 55.0,
                    "mass_flow_rate_kg_s": 8.0,
                    "specific_heat_kJ_kgK": 4.186,
                    "pressure_kPa": 300.0
                },
                "design_area_m2": 50.0,
                "design_U_W_m2K": 500.0,
                "fouling_factor_hot_m2K_W": 0.0002,
                "fouling_factor_cold_m2K_W": 0.0001
            }
        }


class ThermalKPIsData(BaseModel):
    """Computed thermal KPIs."""

    Q_duty_kW: float = Field(
        ...,
        description="Heat duty in kW"
    )
    Q_hot_kW: float = Field(
        ...,
        description="Heat transferred from hot stream in kW"
    )
    Q_cold_kW: float = Field(
        ...,
        description="Heat transferred to cold stream in kW"
    )
    heat_balance_error_percent: float = Field(
        ...,
        description="Heat balance error percentage"
    )
    UA_W_K: float = Field(
        ...,
        description="Overall heat transfer coefficient-area product in W/K"
    )
    U_actual_W_m2K: Optional[float] = Field(
        None,
        description="Actual overall heat transfer coefficient W/(m2*K)"
    )
    LMTD_C: float = Field(
        ...,
        description="Log Mean Temperature Difference in Celsius"
    )
    LMTD_correction_factor: float = Field(
        1.0,
        description="LMTD correction factor F for multi-pass"
    )
    effectiveness_epsilon: float = Field(
        ...,
        description="Heat exchanger effectiveness (0-1)"
    )
    NTU: float = Field(
        ...,
        description="Number of Transfer Units"
    )
    capacity_ratio_Cr: float = Field(
        ...,
        description="Capacity ratio Cmin/Cmax"
    )
    C_hot_kW_K: float = Field(
        ...,
        description="Hot stream heat capacity rate in kW/K"
    )
    C_cold_kW_K: float = Field(
        ...,
        description="Cold stream heat capacity rate in kW/K"
    )
    delta_P_hot_kPa: Optional[float] = Field(
        None,
        description="Hot side pressure drop in kPa"
    )
    delta_P_cold_kPa: Optional[float] = Field(
        None,
        description="Cold side pressure drop in kPa"
    )
    fouling_resistance_m2K_W: float = Field(
        ...,
        description="Total fouling resistance in m2*K/W"
    )
    cleanliness_factor: float = Field(
        ...,
        description="Cleanliness factor (0-1, 1=clean)"
    )


class ComputeKPIsResponse(BaseAPIResponse):
    """Response model for thermal KPIs computation."""

    exchanger_id: str = Field(
        ...,
        description="Exchanger identifier"
    )
    kpis: ThermalKPIsData = Field(
        ...,
        description="Computed thermal KPIs"
    )
    design_comparison: Optional[Dict[str, float]] = Field(
        None,
        description="Comparison to design values if provided"
    )
    performance_status: str = Field(
        "nominal",
        description="Performance status: nominal, degraded, critical"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "computation_hash": "sha256:a1b2c3d4...",
                "timestamp": "2025-12-27T10:30:00Z",
                "agent_version": "1.0.0",
                "agent_id": "GL-014",
                "warnings": [],
                "exchanger_id": "HX-001",
                "kpis": {
                    "Q_duty_kW": 630.0,
                    "Q_hot_kW": 630.0,
                    "Q_cold_kW": 628.5,
                    "heat_balance_error_percent": 0.24,
                    "UA_W_K": 12600.0,
                    "U_actual_W_m2K": 450.0,
                    "LMTD_C": 50.0,
                    "LMTD_correction_factor": 1.0,
                    "effectiveness_epsilon": 0.75,
                    "NTU": 1.8,
                    "capacity_ratio_Cr": 0.65,
                    "C_hot_kW_K": 10.5,
                    "C_cold_kW_K": 33.49,
                    "delta_P_hot_kPa": 15.2,
                    "delta_P_cold_kPa": 22.8,
                    "fouling_resistance_m2K_W": 0.0003,
                    "cleanliness_factor": 0.85
                },
                "performance_status": "nominal"
            }
        }


# =============================================================================
# Historical KPIs Models
# =============================================================================

class TimeRange(BaseModel):
    """Time range for historical queries."""

    start: datetime = Field(
        ...,
        description="Start time (UTC)"
    )
    end: datetime = Field(
        ...,
        description="End time (UTC)"
    )

    @model_validator(mode="after")
    def validate_time_range(self):
        """Ensure start is before end."""
        if self.start >= self.end:
            raise ValueError("Start time must be before end time")
        return self


class KPIHistoryRequest(BaseModel):
    """Request for historical KPI data."""

    time_range: TimeRange = Field(
        ...,
        description="Time range for historical data"
    )
    resolution: str = Field(
        "1h",
        description="Data resolution: 1m, 5m, 15m, 1h, 1d"
    )
    kpis: List[str] = Field(
        default=["Q_duty_kW", "UA_W_K", "effectiveness_epsilon", "cleanliness_factor"],
        description="KPIs to retrieve"
    )

    @field_validator("resolution")
    @classmethod
    def validate_resolution(cls, v):
        """Validate resolution string."""
        valid_resolutions = ["1m", "5m", "15m", "30m", "1h", "4h", "1d"]
        if v not in valid_resolutions:
            raise ValueError(f"Resolution must be one of: {valid_resolutions}")
        return v


class KPIDataPoint(BaseModel):
    """Single KPI data point."""

    timestamp: datetime
    value: float
    quality: str = Field(
        "good",
        description="Data quality: good, uncertain, bad"
    )


class KPIHistoryResponse(BaseAPIResponse):
    """Response model for historical KPIs."""

    exchanger_id: str
    time_range: TimeRange
    resolution: str
    data_points_count: int
    kpi_series: Dict[str, List[KPIDataPoint]] = Field(
        ...,
        description="Time series data for each requested KPI"
    )
    statistics: Dict[str, Dict[str, float]] = Field(
        default_factory=dict,
        description="Statistical summary per KPI (min, max, mean, std)"
    )


# =============================================================================
# Fouling Prediction Models
# =============================================================================

class FoulingPredictionRequest(BaseModel):
    """Request model for fouling prediction."""

    exchanger_id: str = Field(
        ...,
        description="Exchanger identifier"
    )
    current_fouling_resistance_m2K_W: float = Field(
        ...,
        description="Current measured fouling resistance",
        ge=0.0
    )
    current_cleanliness_factor: float = Field(
        ...,
        description="Current cleanliness factor (0-1)",
        ge=0.0,
        le=1.0
    )
    operating_conditions: Dict[str, float] = Field(
        ...,
        description="Current operating conditions (temperatures, flows, etc.)"
    )
    fouling_mechanism: FoulingMechanism = Field(
        FoulingMechanism.COMBINED,
        description="Primary fouling mechanism"
    )
    prediction_horizon_days: int = Field(
        30,
        description="Prediction horizon in days",
        gt=0,
        le=365
    )
    confidence_level: float = Field(
        0.95,
        description="Confidence level for prediction intervals",
        gt=0.5,
        lt=1.0
    )
    include_uncertainty: bool = Field(
        True,
        description="Include uncertainty quantification"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "exchanger_id": "HX-001",
                "current_fouling_resistance_m2K_W": 0.0003,
                "current_cleanliness_factor": 0.85,
                "operating_conditions": {
                    "T_hot_in_C": 120.0,
                    "T_cold_in_C": 25.0,
                    "m_dot_hot_kg_s": 5.0,
                    "m_dot_cold_kg_s": 8.0,
                    "velocity_tube_m_s": 1.5
                },
                "fouling_mechanism": "combined",
                "prediction_horizon_days": 30,
                "confidence_level": 0.95,
                "include_uncertainty": True
            }
        }


class FoulingForecastPoint(BaseModel):
    """Single point in fouling forecast."""

    days_ahead: int
    date: datetime
    fouling_resistance_m2K_W: float
    cleanliness_factor: float
    effectiveness_degradation_percent: float
    confidence_lower: Optional[float] = None
    confidence_upper: Optional[float] = None


class FoulingPredictionResponse(BaseAPIResponse):
    """Response model for fouling prediction."""

    exchanger_id: str
    fouling_mechanism: FoulingMechanism
    prediction_horizon_days: int
    current_state: Dict[str, float] = Field(
        ...,
        description="Current fouling state"
    )
    forecast: List[FoulingForecastPoint] = Field(
        ...,
        description="Fouling forecast time series"
    )
    fouling_rate_m2K_W_per_day: float = Field(
        ...,
        description="Estimated fouling rate"
    )
    days_to_threshold: Optional[int] = Field(
        None,
        description="Days until cleaning threshold reached"
    )
    threshold_fouling_resistance_m2K_W: float = Field(
        ...,
        description="Fouling threshold for cleaning"
    )
    model_confidence: float = Field(
        ...,
        description="Model confidence score (0-1)"
    )
    feature_contributions: Dict[str, float] = Field(
        default_factory=dict,
        description="Feature contributions to fouling rate"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "computation_hash": "sha256:b2c3d4e5...",
                "timestamp": "2025-12-27T10:30:00Z",
                "agent_version": "1.0.0",
                "agent_id": "GL-014",
                "warnings": [],
                "exchanger_id": "HX-001",
                "fouling_mechanism": "combined",
                "prediction_horizon_days": 30,
                "current_state": {
                    "fouling_resistance_m2K_W": 0.0003,
                    "cleanliness_factor": 0.85
                },
                "forecast": [
                    {
                        "days_ahead": 7,
                        "date": "2026-01-03T00:00:00Z",
                        "fouling_resistance_m2K_W": 0.00035,
                        "cleanliness_factor": 0.82,
                        "effectiveness_degradation_percent": 3.5,
                        "confidence_lower": 0.00032,
                        "confidence_upper": 0.00038
                    }
                ],
                "fouling_rate_m2K_W_per_day": 7.14e-6,
                "days_to_threshold": 21,
                "threshold_fouling_resistance_m2K_W": 0.00045,
                "model_confidence": 0.87
            }
        }


class FoulingForecastResponse(BaseAPIResponse):
    """Response for fouling forecast retrieval."""

    exchanger_id: str
    forecast_generated_at: datetime
    forecast: List[FoulingForecastPoint]
    recommended_action: str
    next_cleaning_window: Optional[Dict[str, datetime]] = None


# =============================================================================
# Cleaning Optimization Models
# =============================================================================

class CleaningConstraints(BaseModel):
    """Constraints for cleaning optimization."""

    max_downtime_hours: float = Field(
        24.0,
        description="Maximum allowed downtime per cleaning",
        gt=0.0
    )
    min_days_between_cleanings: int = Field(
        7,
        description="Minimum days between cleaning events",
        gt=0
    )
    available_windows: Optional[List[TimeRange]] = Field(
        None,
        description="Available maintenance windows"
    )
    production_calendar: Optional[Dict[str, bool]] = Field(
        None,
        description="Production calendar (date -> is_production)"
    )
    cleaning_crew_availability: Optional[Dict[str, int]] = Field(
        None,
        description="Crew availability per day"
    )


class CleaningOptimizationRequest(BaseModel):
    """Request model for cleaning schedule optimization."""

    exchanger_id: str = Field(
        ...,
        description="Exchanger identifier"
    )
    optimization_horizon_days: int = Field(
        90,
        description="Optimization horizon in days",
        gt=0,
        le=365
    )
    objective: OptimizationObjective = Field(
        OptimizationObjective.MINIMIZE_COST,
        description="Optimization objective"
    )
    current_fouling_state: Dict[str, float] = Field(
        ...,
        description="Current fouling state"
    )
    fouling_forecast: Optional[List[FoulingForecastPoint]] = Field(
        None,
        description="Pre-computed fouling forecast (optional)"
    )
    cleaning_cost_usd: float = Field(
        5000.0,
        description="Cost per cleaning event in USD",
        gt=0.0
    )
    energy_cost_usd_per_kWh: float = Field(
        0.10,
        description="Energy cost in USD/kWh",
        gt=0.0
    )
    production_loss_usd_per_hour: float = Field(
        1000.0,
        description="Production loss during downtime in USD/hour",
        gt=0.0
    )
    cleaning_duration_hours: float = Field(
        8.0,
        description="Duration of cleaning operation in hours",
        gt=0.0
    )
    constraints: Optional[CleaningConstraints] = None

    class Config:
        json_schema_extra = {
            "example": {
                "exchanger_id": "HX-001",
                "optimization_horizon_days": 90,
                "objective": "minimize_cost",
                "current_fouling_state": {
                    "fouling_resistance_m2K_W": 0.0003,
                    "cleanliness_factor": 0.85
                },
                "cleaning_cost_usd": 5000.0,
                "energy_cost_usd_per_kWh": 0.10,
                "production_loss_usd_per_hour": 1000.0,
                "cleaning_duration_hours": 8.0
            }
        }


class CleaningEvent(BaseModel):
    """Recommended cleaning event."""

    event_id: str
    scheduled_date: datetime
    duration_hours: float
    estimated_cost_usd: float
    expected_improvement: Dict[str, float]
    rationale: str


class CleaningOptimizationResponse(BaseAPIResponse):
    """Response model for cleaning optimization."""

    exchanger_id: str
    optimization_horizon_days: int
    objective: OptimizationObjective
    recommended_schedule: List[CleaningEvent] = Field(
        ...,
        description="Recommended cleaning schedule"
    )
    total_cleaning_events: int
    total_estimated_cost_usd: float
    total_downtime_hours: float
    estimated_energy_savings_usd: float
    net_benefit_usd: float
    optimization_score: float = Field(
        ...,
        description="Optimization quality score (0-1)"
    )
    alternative_schedules: Optional[List[Dict[str, Any]]] = Field(
        None,
        description="Alternative schedule options"
    )
    sensitivity_analysis: Optional[Dict[str, float]] = Field(
        None,
        description="Cost sensitivity to parameters"
    )


# =============================================================================
# What-If Analysis Models
# =============================================================================

class WhatIfScenario(BaseModel):
    """Single what-if scenario definition."""

    scenario_id: str = Field(
        ...,
        description="Unique scenario identifier"
    )
    scenario_name: str = Field(
        ...,
        description="Human-readable scenario name"
    )
    parameter_changes: Dict[str, float] = Field(
        ...,
        description="Parameters to change and their new values"
    )


class WhatIfRequest(BaseModel):
    """Request model for what-if scenario analysis."""

    exchanger_id: str = Field(
        ...,
        description="Exchanger identifier"
    )
    base_conditions: Dict[str, float] = Field(
        ...,
        description="Current/baseline operating conditions"
    )
    scenarios: List[WhatIfScenario] = Field(
        ...,
        description="Scenarios to evaluate",
        min_length=1,
        max_length=20
    )
    kpis_to_evaluate: List[str] = Field(
        default=["Q_duty_kW", "effectiveness_epsilon", "delta_P_hot_kPa"],
        description="KPIs to calculate for each scenario"
    )
    include_comparison: bool = Field(
        True,
        description="Include comparison to baseline"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "exchanger_id": "HX-001",
                "base_conditions": {
                    "T_hot_in_C": 120.0,
                    "T_cold_in_C": 25.0,
                    "m_dot_hot_kg_s": 5.0,
                    "m_dot_cold_kg_s": 8.0
                },
                "scenarios": [
                    {
                        "scenario_id": "S1",
                        "scenario_name": "Increased hot flow",
                        "parameter_changes": {"m_dot_hot_kg_s": 6.0}
                    },
                    {
                        "scenario_id": "S2",
                        "scenario_name": "Higher cold inlet",
                        "parameter_changes": {"T_cold_in_C": 30.0}
                    }
                ],
                "kpis_to_evaluate": [
                    "Q_duty_kW",
                    "effectiveness_epsilon",
                    "delta_P_hot_kPa"
                ],
                "include_comparison": True
            }
        }


class ScenarioResult(BaseModel):
    """Results for a single what-if scenario."""

    scenario_id: str
    scenario_name: str
    applied_conditions: Dict[str, float]
    computed_kpis: Dict[str, float]
    comparison_to_base: Optional[Dict[str, float]] = Field(
        None,
        description="Percent change from baseline"
    )
    feasibility: str = Field(
        "feasible",
        description="Scenario feasibility: feasible, marginal, infeasible"
    )
    feasibility_issues: List[str] = Field(
        default_factory=list,
        description="Issues affecting feasibility"
    )


class WhatIfResponse(BaseAPIResponse):
    """Response model for what-if analysis."""

    exchanger_id: str
    base_kpis: Dict[str, float] = Field(
        ...,
        description="KPIs computed for base conditions"
    )
    scenario_results: List[ScenarioResult]
    best_scenario: Optional[str] = Field(
        None,
        description="Best scenario ID based on primary KPI"
    )
    ranking: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Scenarios ranked by performance"
    )


# =============================================================================
# Explanation Models
# =============================================================================

class ExplanationRequest(BaseModel):
    """Request for explanation (optional query parameters)."""

    explanation_type: ExplanationType = Field(
        ExplanationType.NATURAL_LANGUAGE,
        description="Type of explanation"
    )
    detail_level: str = Field(
        "standard",
        description="Detail level: brief, standard, detailed"
    )
    include_visualizations: bool = Field(
        False,
        description="Include visualization data"
    )


class FeatureContribution(BaseModel):
    """Feature contribution to a prediction/computation."""

    feature_name: str
    feature_value: float
    contribution: float
    contribution_percent: float
    direction: str = Field(
        ...,
        description="Impact direction: positive, negative, neutral"
    )


class ExplanationResponse(BaseAPIResponse):
    """Response model for explainability endpoint."""

    computation_id: str = Field(
        ...,
        description="ID of the computation being explained"
    )
    exchanger_id: str
    explanation_type: ExplanationType
    natural_language_summary: str = Field(
        ...,
        description="Human-readable explanation"
    )
    key_factors: List[str] = Field(
        ...,
        description="Key factors affecting the result"
    )
    feature_contributions: List[FeatureContribution] = Field(
        default_factory=list,
        description="Feature-level contributions"
    )
    confidence_score: float = Field(
        ...,
        description="Explanation confidence (0-1)"
    )
    methodology: str = Field(
        ...,
        description="Explanation methodology used"
    )
    visualizations: Optional[Dict[str, Any]] = Field(
        None,
        description="Visualization data if requested"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "computation_hash": "sha256:c3d4e5f6...",
                "timestamp": "2025-12-27T10:30:00Z",
                "agent_version": "1.0.0",
                "agent_id": "GL-014",
                "warnings": [],
                "computation_id": "comp_12345",
                "exchanger_id": "HX-001",
                "explanation_type": "natural_language",
                "natural_language_summary": "The heat duty decreased by 15% primarily due to increased fouling resistance on the hot side, which reduced the overall heat transfer coefficient by 12%.",
                "key_factors": [
                    "Hot side fouling (+35% contribution)",
                    "Reduced flow rate (+25% contribution)",
                    "Temperature approach (+20% contribution)"
                ],
                "feature_contributions": [
                    {
                        "feature_name": "fouling_resistance_hot",
                        "feature_value": 0.0004,
                        "contribution": 0.35,
                        "contribution_percent": 35.0,
                        "direction": "negative"
                    }
                ],
                "confidence_score": 0.92,
                "methodology": "LIME local explanation with 1000 perturbations"
            }
        }


# =============================================================================
# Audit Trail Models
# =============================================================================

class AuditEvent(BaseModel):
    """Single audit event."""

    event_id: str
    timestamp: datetime
    event_type: str
    user_id: Optional[str] = None
    api_key_id: Optional[str] = None
    endpoint: str
    method: str
    request_hash: str
    response_hash: Optional[str] = None
    status_code: int
    duration_ms: float
    exchanger_id: str
    computation_type: Optional[str] = None
    ip_address: Optional[str] = None


class AuditTrailRequest(BaseModel):
    """Request for audit trail data."""

    time_range: Optional[TimeRange] = None
    event_types: Optional[List[str]] = None
    limit: int = Field(100, ge=1, le=1000)
    offset: int = Field(0, ge=0)


class AuditTrailResponse(BaseAPIResponse):
    """Response model for audit trail."""

    exchanger_id: str
    total_events: int
    events: List[AuditEvent]
    pagination: Dict[str, int] = Field(
        ...,
        description="Pagination info: limit, offset, total"
    )


# =============================================================================
# Recommendation Models
# =============================================================================

class Recommendation(BaseModel):
    """Single recommendation."""

    recommendation_id: str
    priority: str = Field(
        ...,
        description="Priority: critical, high, medium, low"
    )
    category: str = Field(
        ...,
        description="Category: cleaning, maintenance, operational, monitoring"
    )
    title: str
    description: str
    rationale: str
    estimated_impact: Dict[str, float]
    estimated_cost_usd: Optional[float] = None
    recommended_timeframe: str
    confidence: float


class RecommendationsResponse(BaseAPIResponse):
    """Response model for recommendations endpoint."""

    exchanger_id: str
    recommendations: List[Recommendation]
    summary: str
    total_potential_savings_usd: Optional[float] = None


# =============================================================================
# Health & Metrics Models
# =============================================================================

class ComponentHealth(BaseModel):
    """Health status of a component."""

    name: str
    status: str = Field(
        ...,
        description="Status: healthy, degraded, unhealthy"
    )
    latency_ms: Optional[float] = None
    message: Optional[str] = None
    last_check: datetime


class HealthResponse(BaseModel):
    """Health check response."""

    status: str = Field(
        ...,
        description="Overall status: healthy, degraded, unhealthy"
    )
    version: str
    agent_id: str
    timestamp: datetime
    uptime_seconds: float
    components: List[ComponentHealth]

    class Config:
        json_schema_extra = {
            "example": {
                "status": "healthy",
                "version": "1.0.0",
                "agent_id": "GL-014",
                "timestamp": "2025-12-27T10:30:00Z",
                "uptime_seconds": 86400.0,
                "components": [
                    {
                        "name": "database",
                        "status": "healthy",
                        "latency_ms": 5.2,
                        "last_check": "2025-12-27T10:29:55Z"
                    },
                    {
                        "name": "ml_model",
                        "status": "healthy",
                        "latency_ms": 12.5,
                        "last_check": "2025-12-27T10:29:55Z"
                    }
                ]
            }
        }


class MetricValue(BaseModel):
    """Single metric value."""

    name: str
    value: float
    labels: Dict[str, str] = Field(default_factory=dict)
    timestamp: datetime


class MetricsResponse(BaseModel):
    """Prometheus-compatible metrics response."""

    metrics: List[MetricValue]

    def to_prometheus_format(self) -> str:
        """Convert to Prometheus exposition format."""
        lines = []
        for metric in self.metrics:
            labels_str = ""
            if metric.labels:
                label_pairs = [f'{k}="{v}"' for k, v in metric.labels.items()]
                labels_str = "{" + ",".join(label_pairs) + "}"
            lines.append(f"{metric.name}{labels_str} {metric.value}")
        return "\n".join(lines)


# =============================================================================
# Error Response Models
# =============================================================================

class ErrorDetail(BaseModel):
    """Detailed error information."""

    field: Optional[str] = None
    message: str
    code: str


class ErrorResponse(BaseModel):
    """Standard error response."""

    error: str = Field(
        ...,
        description="Error type"
    )
    message: str = Field(
        ...,
        description="Human-readable error message"
    )
    details: Optional[List[ErrorDetail]] = Field(
        None,
        description="Detailed error information"
    )
    request_id: Optional[str] = Field(
        None,
        description="Request ID for tracing"
    )
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )

    class Config:
        json_schema_extra = {
            "example": {
                "error": "validation_error",
                "message": "Invalid request parameters",
                "details": [
                    {
                        "field": "hot_stream.inlet_temperature_C",
                        "message": "Must be greater than outlet temperature for hot stream",
                        "code": "invalid_temperature_direction"
                    }
                ],
                "request_id": "req_abc123",
                "timestamp": "2025-12-27T10:30:00Z"
            }
        }


# =============================================================================
# Utility Functions
# =============================================================================

def compute_hash(data: Dict[str, Any]) -> str:
    """
    Compute SHA-256 hash of input data for traceability.

    Args:
        data: Dictionary of computation inputs

    Returns:
        SHA-256 hash string prefixed with 'sha256:'
    """
    import json

    # Sort keys for deterministic hashing
    serialized = json.dumps(data, sort_keys=True, default=str)
    hash_value = hashlib.sha256(serialized.encode()).hexdigest()
    return f"sha256:{hash_value}"


def create_response_with_hash(
    response_class: type,
    input_data: Dict[str, Any],
    **kwargs
) -> BaseAPIResponse:
    """
    Create a response with computed hash.

    Args:
        response_class: Response model class
        input_data: Input data to hash
        **kwargs: Response field values

    Returns:
        Response instance with computed hash
    """
    computation_hash = compute_hash(input_data)
    return response_class(
        computation_hash=computation_hash,
        **kwargs
    )
