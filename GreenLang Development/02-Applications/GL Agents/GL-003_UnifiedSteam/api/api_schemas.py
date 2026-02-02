"""
GL-003 UnifiedSteam API Schemas

Pydantic models for request/response validation and serialization.
Defines core data structures for steam system optimization API.
"""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import Any, Dict, List, Optional, Union
from uuid import UUID, uuid4

from pydantic import BaseModel, Field, field_validator, model_validator


# =============================================================================
# Enumerations
# =============================================================================

class SteamPhase(str, Enum):
    """Steam phase states."""
    SUBCOOLED_LIQUID = "subcooled_liquid"
    SATURATED_LIQUID = "saturated_liquid"
    TWO_PHASE = "two_phase"
    SATURATED_VAPOR = "saturated_vapor"
    SUPERHEATED_VAPOR = "superheated_vapor"
    SUPERCRITICAL = "supercritical"


class SteamRegion(str, Enum):
    """IAPWS IF97 regions."""
    REGION_1 = "region_1"  # Compressed liquid
    REGION_2 = "region_2"  # Superheated vapor
    REGION_3 = "region_3"  # Near-critical
    REGION_4 = "region_4"  # Two-phase
    REGION_5 = "region_5"  # High-temperature steam


class TrapType(str, Enum):
    """Steam trap types."""
    THERMOSTATIC = "thermostatic"
    THERMODYNAMIC = "thermodynamic"
    MECHANICAL = "mechanical"
    INVERTED_BUCKET = "inverted_bucket"
    FLOAT_THERMOSTATIC = "float_thermostatic"
    BIMETALLIC = "bimetallic"
    BELLOWS = "bellows"


class TrapCondition(str, Enum):
    """Steam trap condition states."""
    GOOD = "good"
    LEAKING = "leaking"
    BLOCKED = "blocked"
    BLOW_THROUGH = "blow_through"
    FAILED_OPEN = "failed_open"
    FAILED_CLOSED = "failed_closed"
    DEGRADED = "degraded"
    UNKNOWN = "unknown"


class OptimizationType(str, Enum):
    """Types of optimization."""
    DESUPERHEATER = "desuperheater"
    CONDENSATE_RECOVERY = "condensate_recovery"
    NETWORK = "network"
    PRESSURE_REDUCTION = "pressure_reduction"
    HEAT_RECOVERY = "heat_recovery"
    LOAD_BALANCING = "load_balancing"


class OptimizationObjective(str, Enum):
    """Optimization objectives."""
    MINIMIZE_COST = "minimize_cost"
    MINIMIZE_EMISSIONS = "minimize_emissions"
    MAXIMIZE_EFFICIENCY = "maximize_efficiency"
    MINIMIZE_ENERGY = "minimize_energy"
    BALANCE_COST_EMISSIONS = "balance_cost_emissions"


class RecommendationPriority(str, Enum):
    """Priority levels for recommendations."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFORMATIONAL = "informational"


class RecommendationStatus(str, Enum):
    """Status of recommendations."""
    PENDING = "pending"
    ACKNOWLEDGED = "acknowledged"
    IN_PROGRESS = "in_progress"
    IMPLEMENTED = "implemented"
    REJECTED = "rejected"
    EXPIRED = "expired"


class AlarmSeverity(str, Enum):
    """Alarm severity levels."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


class KPICategory(str, Enum):
    """KPI categories."""
    ENERGY = "energy"
    EFFICIENCY = "efficiency"
    COST = "cost"
    EMISSIONS = "emissions"
    RELIABILITY = "reliability"
    SAFETY = "safety"


# =============================================================================
# Base Models
# =============================================================================

class TimestampedModel(BaseModel):
    """Base model with timestamp fields."""
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: Optional[datetime] = None

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat(),
            UUID: lambda v: str(v),
            Decimal: lambda v: float(v),
        }


# =============================================================================
# Steam Properties Models
# =============================================================================

class SteamState(BaseModel):
    """Complete thermodynamic state of steam."""
    pressure_kpa: float = Field(..., ge=0, description="Pressure in kPa")
    temperature_c: float = Field(..., description="Temperature in Celsius")
    specific_enthalpy_kj_kg: float = Field(..., description="Specific enthalpy in kJ/kg")
    specific_entropy_kj_kg_k: float = Field(..., description="Specific entropy in kJ/(kg*K)")
    specific_volume_m3_kg: float = Field(..., ge=0, description="Specific volume in m3/kg")
    density_kg_m3: float = Field(..., ge=0, description="Density in kg/m3")
    quality: Optional[float] = Field(None, ge=0, le=1, description="Steam quality (0-1) for two-phase")
    phase: SteamPhase = Field(..., description="Phase state")
    region: SteamRegion = Field(..., description="IAPWS IF97 region")

    # Optional additional properties
    specific_internal_energy_kj_kg: Optional[float] = Field(None, description="Specific internal energy in kJ/kg")
    cp_kj_kg_k: Optional[float] = Field(None, ge=0, description="Isobaric heat capacity in kJ/(kg*K)")
    cv_kj_kg_k: Optional[float] = Field(None, ge=0, description="Isochoric heat capacity in kJ/(kg*K)")
    speed_of_sound_m_s: Optional[float] = Field(None, ge=0, description="Speed of sound in m/s")
    viscosity_pa_s: Optional[float] = Field(None, ge=0, description="Dynamic viscosity in Pa*s")
    thermal_conductivity_w_m_k: Optional[float] = Field(None, ge=0, description="Thermal conductivity in W/(m*K)")

    class Config:
        json_schema_extra = {
            "example": {
                "pressure_kpa": 1000.0,
                "temperature_c": 200.0,
                "specific_enthalpy_kj_kg": 2827.9,
                "specific_entropy_kj_kg_k": 6.694,
                "specific_volume_m3_kg": 0.2060,
                "density_kg_m3": 4.854,
                "phase": "superheated_vapor",
                "region": "region_2",
            }
        }


class SteamPropertiesRequest(BaseModel):
    """Request to compute steam properties."""
    request_id: UUID = Field(default_factory=uuid4)

    # Input specification (at least two must be provided)
    pressure_kpa: Optional[float] = Field(None, ge=0, description="Pressure in kPa")
    temperature_c: Optional[float] = Field(None, description="Temperature in Celsius")
    specific_enthalpy_kj_kg: Optional[float] = Field(None, description="Specific enthalpy in kJ/kg")
    specific_entropy_kj_kg_k: Optional[float] = Field(None, description="Specific entropy in kJ/(kg*K)")
    quality: Optional[float] = Field(None, ge=0, le=1, description="Steam quality for saturation")

    # Options
    include_transport_properties: bool = Field(default=False, description="Include viscosity, conductivity")
    include_derivatives: bool = Field(default=False, description="Include dh/dp, dh/dT, etc.")

    @model_validator(mode="after")
    def validate_inputs(self):
        """Ensure at least two independent properties are provided."""
        provided = sum([
            self.pressure_kpa is not None,
            self.temperature_c is not None,
            self.specific_enthalpy_kj_kg is not None,
            self.specific_entropy_kj_kg_k is not None,
            self.quality is not None,
        ])
        if provided < 2:
            raise ValueError("At least two independent properties must be provided")
        return self


class SteamPropertiesResponse(TimestampedModel):
    """Response with computed steam properties."""
    request_id: UUID
    success: bool
    steam_state: Optional[SteamState] = None
    computation_time_ms: float = Field(..., ge=0)
    error_message: Optional[str] = None

    # Derivatives if requested
    dh_dp_const_t: Optional[float] = None
    dh_dt_const_p: Optional[float] = None
    ds_dp_const_t: Optional[float] = None
    ds_dt_const_p: Optional[float] = None


# =============================================================================
# Balance Calculation Models
# =============================================================================

class StreamDefinition(BaseModel):
    """Definition of a steam/water stream for balance calculations."""
    stream_id: str = Field(..., min_length=1, max_length=100)
    stream_name: str = Field(..., min_length=1, max_length=255)
    mass_flow_kg_s: float = Field(..., ge=0, description="Mass flow rate in kg/s")
    pressure_kpa: float = Field(..., ge=0, description="Pressure in kPa")
    temperature_c: float = Field(..., description="Temperature in Celsius")
    is_inlet: bool = Field(..., description="True if inlet stream, False if outlet")

    # Optional measured values
    specific_enthalpy_kj_kg: Optional[float] = Field(None, description="Measured enthalpy if available")
    quality: Optional[float] = Field(None, ge=0, le=1, description="Steam quality if two-phase")


class EnthalpyBalanceRequest(BaseModel):
    """Request to compute enthalpy balance."""
    request_id: UUID = Field(default_factory=uuid4)
    equipment_id: str = Field(..., min_length=1, max_length=100)
    equipment_name: str = Field(..., min_length=1, max_length=255)

    streams: List[StreamDefinition] = Field(..., min_length=2)

    # Optional heat transfer
    heat_input_kw: float = Field(default=0.0, description="External heat input in kW")
    heat_loss_kw: float = Field(default=0.0, description="Heat loss to environment in kW")

    # Reference conditions
    reference_temperature_c: float = Field(default=25.0, description="Reference temperature for enthalpy")


class EnthalpyBalanceResponse(TimestampedModel):
    """Response with enthalpy balance results."""
    request_id: UUID
    equipment_id: str
    success: bool

    # Balance results
    total_inlet_enthalpy_kw: float = Field(..., description="Total inlet enthalpy flow in kW")
    total_outlet_enthalpy_kw: float = Field(..., description="Total outlet enthalpy flow in kW")
    enthalpy_imbalance_kw: float = Field(..., description="Enthalpy imbalance (should be ~0)")
    enthalpy_imbalance_percent: float = Field(..., description="Imbalance as percentage")

    # Stream details
    stream_enthalpies: Dict[str, float] = Field(default_factory=dict)

    # Quality indicators
    balance_closed: bool = Field(..., description="True if balance closes within tolerance")
    tolerance_percent: float = Field(default=2.0)
    data_quality_score: float = Field(..., ge=0, le=100)

    error_message: Optional[str] = None


class MassBalanceRequest(BaseModel):
    """Request to compute mass balance."""
    request_id: UUID = Field(default_factory=uuid4)
    equipment_id: str = Field(..., min_length=1, max_length=100)
    equipment_name: str = Field(..., min_length=1, max_length=255)

    streams: List[StreamDefinition] = Field(..., min_length=2)

    # Optional accumulation for transient analysis
    accumulation_rate_kg_s: float = Field(default=0.0, description="Accumulation rate for transient")


class MassBalanceResponse(TimestampedModel):
    """Response with mass balance results."""
    request_id: UUID
    equipment_id: str
    success: bool

    total_inlet_mass_flow_kg_s: float
    total_outlet_mass_flow_kg_s: float
    mass_imbalance_kg_s: float
    mass_imbalance_percent: float

    balance_closed: bool
    tolerance_percent: float = Field(default=1.0)
    data_quality_score: float = Field(..., ge=0, le=100)

    error_message: Optional[str] = None


# =============================================================================
# Optimization Models
# =============================================================================

class OptimizationRecommendation(BaseModel):
    """A single optimization recommendation."""
    recommendation_id: UUID = Field(default_factory=uuid4)
    recommendation_type: OptimizationType
    priority: RecommendationPriority
    status: RecommendationStatus = RecommendationStatus.PENDING

    title: str = Field(..., min_length=1, max_length=255)
    description: str = Field(..., min_length=1)
    rationale: str = Field(..., description="Why this recommendation is made")

    # Impact estimates
    estimated_energy_savings_kw: Optional[float] = Field(None, ge=0)
    estimated_cost_savings_usd_year: Optional[float] = Field(None)
    estimated_emissions_reduction_kg_co2_year: Optional[float] = Field(None, ge=0)
    estimated_payback_months: Optional[float] = Field(None, ge=0)

    # Implementation details
    affected_equipment: List[str] = Field(default_factory=list)
    required_actions: List[str] = Field(default_factory=list)
    implementation_complexity: str = Field(default="medium", pattern="^(low|medium|high)$")

    # Confidence and timing
    confidence_score: float = Field(..., ge=0, le=1)
    valid_until: Optional[datetime] = None

    created_at: datetime = Field(default_factory=datetime.utcnow)


class DesuperheaterOptimizationRequest(BaseModel):
    """Request for desuperheater optimization."""
    request_id: UUID = Field(default_factory=uuid4)
    desuperheater_id: str = Field(..., min_length=1, max_length=100)

    # Current operating conditions
    inlet_steam_pressure_kpa: float = Field(..., ge=0)
    inlet_steam_temperature_c: float = Field(...)
    inlet_steam_flow_kg_s: float = Field(..., ge=0)

    target_outlet_temperature_c: float = Field(...)
    target_temperature_tolerance_c: float = Field(default=2.0, ge=0)

    # Spray water conditions
    spray_water_pressure_kpa: float = Field(..., ge=0)
    spray_water_temperature_c: float = Field(...)

    # Constraints
    min_superheat_c: float = Field(default=10.0, ge=0, description="Minimum superheat above saturation")
    max_spray_water_flow_kg_s: Optional[float] = Field(None, ge=0)

    # Optimization options
    objective: OptimizationObjective = Field(default=OptimizationObjective.MINIMIZE_ENERGY)
    optimization_horizon_hours: int = Field(default=24, ge=1, le=168)


class DesuperheaterOptimizationResponse(TimestampedModel):
    """Response with desuperheater optimization results."""
    request_id: UUID
    desuperheater_id: str
    success: bool

    # Optimal setpoints
    optimal_spray_water_flow_kg_s: float = Field(..., ge=0)
    optimal_outlet_temperature_c: float

    # Predicted outlet state
    predicted_outlet_state: Optional[SteamState] = None

    # Energy and efficiency
    spray_water_energy_kw: float = Field(..., description="Energy added by spray water")
    desuperheating_efficiency: float = Field(..., ge=0, le=1)

    # Recommendations
    recommendations: List[OptimizationRecommendation] = Field(default_factory=list)

    computation_time_ms: float = Field(..., ge=0)
    error_message: Optional[str] = None


class CondensateOptimizationRequest(BaseModel):
    """Request for condensate recovery optimization."""
    request_id: UUID = Field(default_factory=uuid4)
    system_id: str = Field(..., min_length=1, max_length=100)

    # Current condensate conditions
    condensate_sources: List[Dict[str, Any]] = Field(..., min_length=1)
    current_recovery_rate_percent: float = Field(..., ge=0, le=100)

    # Flash steam recovery
    flash_tank_pressure_kpa: Optional[float] = Field(None, ge=0)
    flash_steam_utilization_percent: Optional[float] = Field(None, ge=0, le=100)

    # Heat recovery
    condensate_return_temperature_c: float = Field(...)
    makeup_water_temperature_c: float = Field(...)
    makeup_water_cost_usd_m3: float = Field(default=1.0, ge=0)

    # Treatment costs
    condensate_treatment_cost_usd_m3: float = Field(default=0.5, ge=0)
    makeup_treatment_cost_usd_m3: float = Field(default=2.0, ge=0)

    # Optimization settings
    objective: OptimizationObjective = Field(default=OptimizationObjective.MINIMIZE_COST)
    target_recovery_rate_percent: Optional[float] = Field(None, ge=0, le=100)


class CondensateOptimizationResponse(TimestampedModel):
    """Response with condensate recovery optimization results."""
    request_id: UUID
    system_id: str
    success: bool

    # Optimal recovery
    optimal_recovery_rate_percent: float = Field(..., ge=0, le=100)
    current_vs_optimal_delta_percent: float

    # Flash steam optimization
    optimal_flash_tank_pressure_kpa: Optional[float] = None
    flash_steam_recovery_kg_s: Optional[float] = None

    # Economic analysis
    annual_water_savings_m3: float = Field(..., ge=0)
    annual_energy_savings_mwh: float = Field(..., ge=0)
    annual_cost_savings_usd: float
    implementation_cost_usd: Optional[float] = None
    simple_payback_years: Optional[float] = None

    # Emissions
    annual_co2_reduction_tonnes: float = Field(..., ge=0)

    recommendations: List[OptimizationRecommendation] = Field(default_factory=list)

    computation_time_ms: float = Field(..., ge=0)
    error_message: Optional[str] = None


class NetworkOptimizationRequest(BaseModel):
    """Request for steam network optimization."""
    request_id: UUID = Field(default_factory=uuid4)
    network_id: str = Field(..., min_length=1, max_length=100)

    # Network topology (simplified)
    headers: List[Dict[str, Any]] = Field(..., min_length=1)
    consumers: List[Dict[str, Any]] = Field(..., min_length=1)
    generators: List[Dict[str, Any]] = Field(..., min_length=1)

    # Current demand
    total_demand_kg_s: float = Field(..., ge=0)
    demand_by_header: Dict[str, float] = Field(default_factory=dict)

    # Constraints
    min_header_pressure_kpa: Dict[str, float] = Field(default_factory=dict)
    max_header_pressure_kpa: Dict[str, float] = Field(default_factory=dict)

    # Optimization settings
    objective: OptimizationObjective = Field(default=OptimizationObjective.BALANCE_COST_EMISSIONS)
    cost_weight: float = Field(default=0.5, ge=0, le=1)
    emissions_weight: float = Field(default=0.5, ge=0, le=1)
    optimization_horizon_hours: int = Field(default=24, ge=1, le=168)


class NetworkOptimizationResponse(TimestampedModel):
    """Response with network optimization results."""
    request_id: UUID
    network_id: str
    success: bool

    # Optimal operating point
    optimal_header_pressures_kpa: Dict[str, float]
    optimal_generator_outputs_kg_s: Dict[str, float]
    optimal_letdown_flows_kg_s: Dict[str, float]

    # Performance metrics
    total_generation_kg_s: float
    total_consumption_kg_s: float
    network_efficiency_percent: float

    # Economic results
    total_operating_cost_usd_h: float
    marginal_cost_by_header_usd_kg: Dict[str, float]

    # Environmental
    total_emissions_kg_co2_h: float
    emissions_by_source: Dict[str, float]

    # Constraints
    all_constraints_satisfied: bool
    violated_constraints: List[str] = Field(default_factory=list)

    recommendations: List[OptimizationRecommendation] = Field(default_factory=list)

    computation_time_ms: float = Field(..., ge=0)
    solver_status: str = Field(default="optimal")
    error_message: Optional[str] = None


# =============================================================================
# Trap Diagnostics Models
# =============================================================================

class TrapStatus(BaseModel):
    """Current status of a steam trap."""
    trap_id: str = Field(..., min_length=1, max_length=100)
    trap_name: str = Field(..., min_length=1, max_length=255)
    trap_type: TrapType
    condition: TrapCondition
    condition_confidence: float = Field(..., ge=0, le=1)

    # Location
    location: str = Field(..., min_length=1, max_length=255)
    header_id: Optional[str] = None
    equipment_id: Optional[str] = None

    # Operating conditions
    inlet_pressure_kpa: Optional[float] = Field(None, ge=0)
    outlet_pressure_kpa: Optional[float] = Field(None, ge=0)
    inlet_temperature_c: Optional[float] = None
    outlet_temperature_c: Optional[float] = None
    differential_temperature_c: Optional[float] = None

    # Diagnostic metrics
    cycle_rate_per_min: Optional[float] = Field(None, ge=0)
    acoustic_signature_db: Optional[float] = None
    ultrasonic_reading: Optional[float] = None

    # Loss estimates
    estimated_steam_loss_kg_h: Optional[float] = Field(None, ge=0)
    estimated_energy_loss_kw: Optional[float] = Field(None, ge=0)
    estimated_annual_cost_loss_usd: Optional[float] = Field(None, ge=0)

    # Timestamps
    last_inspection_date: Optional[datetime] = None
    last_maintenance_date: Optional[datetime] = None
    installation_date: Optional[datetime] = None

    created_at: datetime = Field(default_factory=datetime.utcnow)


class TrapFailurePrediction(BaseModel):
    """Prediction of trap failure."""
    trap_id: str
    prediction_id: UUID = Field(default_factory=uuid4)

    # Prediction results
    failure_probability_30d: float = Field(..., ge=0, le=1)
    failure_probability_90d: float = Field(..., ge=0, le=1)
    predicted_failure_mode: TrapCondition
    predicted_remaining_life_days: Optional[float] = Field(None, ge=0)

    # Risk factors
    risk_factors: List[str] = Field(default_factory=list)
    risk_score: float = Field(..., ge=0, le=100)

    # Recommendations
    recommended_action: str
    recommended_action_date: Optional[datetime] = None
    priority: RecommendationPriority

    # Model confidence
    model_confidence: float = Field(..., ge=0, le=1)
    prediction_timestamp: datetime = Field(default_factory=datetime.utcnow)


class TrapDiagnosticsRequest(BaseModel):
    """Request for trap diagnostics."""
    request_id: UUID = Field(default_factory=uuid4)
    trap_id: str = Field(..., min_length=1, max_length=100)

    # Measurement data
    inlet_pressure_kpa: Optional[float] = Field(None, ge=0)
    outlet_pressure_kpa: Optional[float] = Field(None, ge=0)
    inlet_temperature_c: Optional[float] = None
    outlet_temperature_c: Optional[float] = None

    # Diagnostic data
    acoustic_data: Optional[List[float]] = None
    ultrasonic_data: Optional[List[float]] = None
    temperature_profile: Optional[List[float]] = None

    # Options
    include_failure_prediction: bool = Field(default=True)
    include_loss_estimation: bool = Field(default=True)


class TrapDiagnosticsResponse(TimestampedModel):
    """Response with trap diagnostics results."""
    request_id: UUID
    trap_id: str
    success: bool

    status: Optional[TrapStatus] = None
    failure_prediction: Optional[TrapFailurePrediction] = None

    # Diagnostic details
    diagnostic_method: str = Field(default="multi-sensor")
    diagnostic_confidence: float = Field(..., ge=0, le=1)

    # Anomalies detected
    anomalies_detected: List[str] = Field(default_factory=list)

    computation_time_ms: float = Field(..., ge=0)
    error_message: Optional[str] = None


class BatchTrapDiagnosticsRequest(BaseModel):
    """Request for batch trap diagnostics."""
    request_id: UUID = Field(default_factory=uuid4)
    traps: List[TrapDiagnosticsRequest] = Field(..., min_length=1, max_length=1000)

    # Batch options
    include_summary: bool = Field(default=True)
    include_prioritization: bool = Field(default=True)


class BatchTrapDiagnosticsResponse(TimestampedModel):
    """Response with batch trap diagnostics results."""
    request_id: UUID
    success: bool

    # Individual results
    results: List[TrapDiagnosticsResponse] = Field(default_factory=list)

    # Summary statistics
    total_traps: int
    traps_good: int
    traps_leaking: int
    traps_blocked: int
    traps_failed: int

    total_estimated_steam_loss_kg_h: float = Field(..., ge=0)
    total_estimated_energy_loss_kw: float = Field(..., ge=0)
    total_estimated_annual_cost_loss_usd: float = Field(..., ge=0)

    # Prioritized action list
    prioritized_actions: List[Dict[str, Any]] = Field(default_factory=list)

    computation_time_ms: float = Field(..., ge=0)
    error_message: Optional[str] = None


# =============================================================================
# Root Cause Analysis Models
# =============================================================================

class CausalFactor(BaseModel):
    """A causal factor identified in RCA."""
    factor_id: UUID = Field(default_factory=uuid4)
    factor_name: str = Field(..., min_length=1, max_length=255)
    factor_description: str

    # Causal strength
    causal_strength: float = Field(..., ge=0, le=1, description="Strength of causal relationship")
    confidence: float = Field(..., ge=0, le=1)

    # Direction
    is_root_cause: bool = Field(default=False)
    is_contributing_factor: bool = Field(default=True)

    # Evidence
    supporting_evidence: List[str] = Field(default_factory=list)
    data_sources: List[str] = Field(default_factory=list)

    # Related variables
    related_variables: List[str] = Field(default_factory=list)
    affected_outcomes: List[str] = Field(default_factory=list)


class CounterfactualScenario(BaseModel):
    """A counterfactual scenario for what-if analysis."""
    scenario_id: UUID = Field(default_factory=uuid4)
    scenario_name: str = Field(..., min_length=1, max_length=255)
    scenario_description: str

    # Intervention
    intervention_variable: str
    intervention_value: float
    baseline_value: float

    # Predicted outcome
    predicted_outcome: float
    baseline_outcome: float
    outcome_change: float
    outcome_change_percent: float

    # Confidence
    prediction_confidence: float = Field(..., ge=0, le=1)
    model_used: str


class RCARequest(BaseModel):
    """Request for root cause analysis."""
    request_id: UUID = Field(default_factory=uuid4)

    # Target event/anomaly
    target_event: str = Field(..., min_length=1, max_length=255, description="Event to analyze")
    event_timestamp: datetime
    event_severity: AlarmSeverity

    # Context
    affected_equipment: List[str] = Field(default_factory=list)
    affected_variables: List[str] = Field(default_factory=list)

    # Time window for analysis
    lookback_hours: int = Field(default=24, ge=1, le=168)
    lookahead_hours: int = Field(default=0, ge=0, le=24)

    # Analysis options
    include_counterfactuals: bool = Field(default=True)
    max_causal_factors: int = Field(default=10, ge=1, le=50)
    min_confidence_threshold: float = Field(default=0.5, ge=0, le=1)


class RCAResponse(TimestampedModel):
    """Response with root cause analysis results."""
    request_id: UUID
    success: bool

    # Target event
    target_event: str
    event_timestamp: datetime

    # Identified causes
    root_causes: List[CausalFactor] = Field(default_factory=list)
    contributing_factors: List[CausalFactor] = Field(default_factory=list)

    # Causal graph (simplified representation)
    causal_chain: List[str] = Field(default_factory=list, description="Ordered chain of causation")

    # Counterfactual analysis
    counterfactual_scenarios: List[CounterfactualScenario] = Field(default_factory=list)

    # Summary
    executive_summary: str
    recommended_actions: List[str] = Field(default_factory=list)

    # Model info
    analysis_method: str = Field(default="causal_discovery")
    model_confidence: float = Field(..., ge=0, le=1)

    computation_time_ms: float = Field(..., ge=0)
    error_message: Optional[str] = None


# =============================================================================
# Explainability Models
# =============================================================================

class FeatureContribution(BaseModel):
    """Contribution of a feature to a prediction/recommendation."""
    feature_name: str
    feature_value: Any
    contribution_score: float
    contribution_direction: str = Field(..., pattern="^(positive|negative|neutral)$")
    explanation: Optional[str] = None


class ExplainabilityRequest(BaseModel):
    """Request for explainability of a recommendation."""
    request_id: UUID = Field(default_factory=uuid4)
    recommendation_id: UUID
    explanation_type: str = Field(default="shap", pattern="^(shap|lime|both)$")
    max_features: int = Field(default=10, ge=1, le=50)
    include_counterfactuals: bool = Field(default=False)


class ExplainabilityResponse(TimestampedModel):
    """Response with explainability information."""
    request_id: UUID
    recommendation_id: UUID
    success: bool

    # SHAP values
    shap_feature_contributions: List[FeatureContribution] = Field(default_factory=list)
    shap_base_value: Optional[float] = None
    shap_output_value: Optional[float] = None

    # LIME explanation
    lime_feature_contributions: List[FeatureContribution] = Field(default_factory=list)
    lime_model_score: Optional[float] = None

    # Natural language explanation
    plain_english_explanation: str
    technical_explanation: Optional[str] = None

    # Key drivers
    key_drivers: List[str] = Field(default_factory=list)
    sensitivity_analysis: Optional[Dict[str, float]] = None

    # Counterfactual explanations
    counterfactual_changes: Optional[List[Dict[str, Any]]] = None

    computation_time_ms: float = Field(..., ge=0)
    error_message: Optional[str] = None


# =============================================================================
# KPI and Climate Impact Models
# =============================================================================

class KPIValue(BaseModel):
    """A single KPI measurement."""
    kpi_id: UUID = Field(default_factory=uuid4)
    kpi_name: str
    category: KPICategory
    current_value: float
    target_value: Optional[float] = None
    unit: str
    trend: Optional[str] = Field(None, pattern="^(up|down|stable)$")
    trend_percent: Optional[float] = None
    is_on_target: Optional[bool] = None
    measurement_timestamp: datetime = Field(default_factory=datetime.utcnow)


class EnergyMetrics(BaseModel):
    """Energy-related metrics."""
    total_steam_consumption_kg_h: float = Field(..., ge=0)
    total_steam_generation_kg_h: float = Field(..., ge=0)
    total_energy_consumption_mw: float = Field(..., ge=0)

    boiler_efficiency_percent: Optional[float] = Field(None, ge=0, le=100)
    system_efficiency_percent: Optional[float] = Field(None, ge=0, le=100)

    condensate_recovery_percent: float = Field(..., ge=0, le=100)
    flash_steam_recovery_percent: Optional[float] = Field(None, ge=0, le=100)

    energy_intensity_mj_per_unit: Optional[float] = Field(None, ge=0)


class EmissionsMetrics(BaseModel):
    """Emissions-related metrics."""
    total_co2_emissions_kg_h: float = Field(..., ge=0)
    co2_emissions_by_source: Dict[str, float] = Field(default_factory=dict)

    total_nox_emissions_kg_h: float = Field(default=0, ge=0)
    total_sox_emissions_kg_h: float = Field(default=0, ge=0)

    carbon_intensity_kg_co2_per_mwh: float = Field(..., ge=0)
    avoided_emissions_kg_co2_h: Optional[float] = Field(None, ge=0)


class KPIDashboardResponse(TimestampedModel):
    """Response with KPI dashboard data."""
    request_id: UUID = Field(default_factory=uuid4)
    success: bool

    # Time context
    period_start: datetime
    period_end: datetime
    aggregation_period: str = Field(default="hourly")

    # KPIs by category
    energy_kpis: List[KPIValue] = Field(default_factory=list)
    efficiency_kpis: List[KPIValue] = Field(default_factory=list)
    cost_kpis: List[KPIValue] = Field(default_factory=list)
    emissions_kpis: List[KPIValue] = Field(default_factory=list)
    reliability_kpis: List[KPIValue] = Field(default_factory=list)

    # Summary metrics
    overall_performance_score: float = Field(..., ge=0, le=100)
    kpis_on_target: int
    kpis_off_target: int
    kpis_improving: int
    kpis_declining: int

    # Alerts
    kpi_alerts: List[Dict[str, Any]] = Field(default_factory=list)

    error_message: Optional[str] = None


class ClimateImpactResponse(TimestampedModel):
    """Response with climate/energy impact data."""
    request_id: UUID = Field(default_factory=uuid4)
    success: bool

    # Time context
    period_start: datetime
    period_end: datetime

    # Energy metrics
    energy_metrics: EnergyMetrics

    # Emissions metrics
    emissions_metrics: EmissionsMetrics

    # Comparison to baseline
    baseline_period_start: Optional[datetime] = None
    baseline_period_end: Optional[datetime] = None
    emissions_vs_baseline_percent: Optional[float] = None
    energy_vs_baseline_percent: Optional[float] = None

    # Targets and compliance
    annual_emissions_target_tonnes_co2: Optional[float] = None
    ytd_emissions_tonnes_co2: Optional[float] = None
    on_track_for_target: Optional[bool] = None

    # Recommendations for improvement
    improvement_opportunities: List[OptimizationRecommendation] = Field(default_factory=list)

    # Certification/reporting
    reporting_standard: str = Field(default="GHG Protocol")
    verification_status: str = Field(default="unverified")

    error_message: Optional[str] = None


# =============================================================================
# Common Response Models
# =============================================================================

class ErrorResponse(BaseModel):
    """Standard error response."""
    error_code: str
    error_message: str
    details: Optional[Dict[str, Any]] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    request_id: Optional[UUID] = None


class PaginationParams(BaseModel):
    """Pagination parameters for list queries."""
    page: int = Field(default=1, ge=1)
    page_size: int = Field(default=20, ge=1, le=100)
    sort_by: Optional[str] = None
    sort_order: str = Field(default="desc", pattern="^(asc|desc)$")


class PaginatedResponse(BaseModel):
    """Generic paginated response wrapper."""
    items: List[Any]
    total_count: int
    page: int
    page_size: int
    total_pages: int
    has_next: bool
    has_previous: bool


class TimeRangeFilter(BaseModel):
    """Time range filter for queries."""
    start_time: datetime
    end_time: datetime

    @model_validator(mode="after")
    def validate_time_range(self):
        if self.end_time < self.start_time:
            raise ValueError("end_time must be after start_time")
        return self


# =============================================================================
# Health and Status Models
# =============================================================================

class ServiceHealth(BaseModel):
    """Health status of a service component."""
    service_name: str
    status: str = Field(..., pattern="^(healthy|degraded|unhealthy)$")
    latency_ms: float = Field(..., ge=0)
    last_check: datetime
    error_message: Optional[str] = None


class SystemStatus(BaseModel):
    """Overall system status."""
    status: str = Field(..., pattern="^(healthy|degraded|unhealthy)$")
    version: str
    uptime_seconds: float
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    services: List[ServiceHealth]
    active_connections: int
    requests_per_minute: float
