"""
GL-006 HEATRECLAIM - Schema Definitions

Pydantic models for all inputs, outputs, process data, optimization
results, and status reporting for the HEATRECLAIM agent.

All schemas support zero-hallucination principles with deterministic
calculations, SHA-256 provenance tracking, and regulatory compliance.
"""

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union
import uuid

from pydantic import BaseModel, Field, validator, root_validator

from .config import (
    StreamType,
    Phase,
    ExchangerType,
    FlowArrangement,
    OptimizationMode,
    OptimizationObjective,
)


# =============================================================================
# ENUMS
# =============================================================================

class OptimizationStatus(Enum):
    """Optimization execution status."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    TIMEOUT = "timeout"
    INFEASIBLE = "infeasible"


class CalculationType(Enum):
    """Types of calculations performed."""
    PINCH_ANALYSIS = "pinch_analysis"
    HEN_SYNTHESIS = "hen_synthesis"
    EXERGY = "exergy"
    ECONOMIC = "economic"
    LMTD = "lmtd"
    NTU = "ntu"
    PARETO = "pareto"
    UNCERTAINTY = "uncertainty"


class SeverityLevel(Enum):
    """Event severity levels."""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


class ConstraintStatus(Enum):
    """Constraint satisfaction status."""
    SATISFIED = "satisfied"
    ACTIVE = "active"  # Binding at limit
    VIOLATED = "violated"
    RELAXED = "relaxed"


# =============================================================================
# STREAM SCHEMAS
# =============================================================================

class HeatStream(BaseModel):
    """
    Process heat stream definition.

    Represents a hot or cold stream in the heat recovery network
    with full thermophysical properties and constraints.
    """

    stream_id: str = Field(..., description="Unique stream identifier")
    stream_name: str = Field(default="", description="Human-readable name")
    stream_type: StreamType = Field(..., description="Hot, cold, or utility")

    # Thermophysical properties
    fluid_name: str = Field(default="Water", description="Fluid identifier")
    phase: Phase = Field(default=Phase.LIQUID, description="Fluid phase")

    # Temperatures (°C)
    T_supply_C: float = Field(
        ...,
        ge=-200.0,
        le=1500.0,
        description="Supply (inlet) temperature"
    )
    T_target_C: float = Field(
        ...,
        ge=-200.0,
        le=1500.0,
        description="Target (outlet) temperature"
    )

    # Flow and properties
    m_dot_kg_s: float = Field(
        ...,
        ge=0.0,
        le=10000.0,
        description="Mass flow rate (kg/s)"
    )
    Cp_kJ_kgK: float = Field(
        ...,
        ge=0.1,
        le=10.0,
        description="Specific heat capacity (kJ/kg·K)"
    )

    # Heat capacity rate (derived)
    @property
    def FCp_kW_K(self) -> float:
        """Heat capacity rate F*Cp (kW/K)."""
        return self.m_dot_kg_s * self.Cp_kJ_kgK

    @property
    def duty_kW(self) -> float:
        """Heat duty (kW) - absolute value."""
        return abs(self.FCp_kW_K * (self.T_target_C - self.T_supply_C))

    # Additional properties
    pressure_kPa: float = Field(
        default=101.325,
        ge=1.0,
        le=50000.0,
        description="Operating pressure (kPa)"
    )
    density_kg_m3: Optional[float] = Field(
        default=None,
        ge=0.01,
        le=20000.0,
        description="Fluid density (kg/m³)"
    )
    viscosity_Pa_s: Optional[float] = Field(
        default=None,
        ge=1e-6,
        le=100.0,
        description="Dynamic viscosity (Pa·s)"
    )
    thermal_conductivity_W_mK: Optional[float] = Field(
        default=None,
        ge=0.01,
        le=500.0,
        description="Thermal conductivity (W/m·K)"
    )

    # Fouling
    fouling_factor_m2K_W: float = Field(
        default=0.0001,
        ge=0.0,
        le=0.01,
        description="Fouling resistance (m²·K/W)"
    )

    # Availability and constraints
    availability: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Stream availability factor"
    )
    min_approach_C: float = Field(
        default=10.0,
        ge=1.0,
        le=100.0,
        description="Minimum approach temperature (°C)"
    )

    # Operational constraints
    constraints: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional constraints"
    )

    # Metadata
    source_system: str = Field(default="manual", description="Data source")
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    data_quality: str = Field(default="good", description="Data quality flag")

    @validator("T_target_C")
    def validate_temperature_direction(cls, v, values):
        """Validate temperature change direction matches stream type."""
        if "stream_type" in values and "T_supply_C" in values:
            stream_type = values["stream_type"]
            T_supply = values["T_supply_C"]
            if stream_type == StreamType.HOT and v > T_supply:
                raise ValueError("Hot stream target must be lower than supply")
            if stream_type == StreamType.COLD and v < T_supply:
                raise ValueError("Cold stream target must be higher than supply")
        return v

    class Config:
        use_enum_values = True


class UtilityCost(BaseModel):
    """Utility cost specification."""

    utility_id: str = Field(..., description="Utility identifier")
    utility_name: str = Field(default="", description="Human-readable name")
    utility_type: str = Field(
        ...,
        description="hot_utility or cold_utility"
    )

    # Cost
    cost_usd_gj: float = Field(
        ...,
        ge=0.0,
        le=1000.0,
        description="Cost per GJ ($/GJ)"
    )
    cost_usd_kwh: Optional[float] = Field(
        default=None,
        description="Cost per kWh for electricity"
    )

    # Emissions
    emissions_factor_kgCO2e_GJ: float = Field(
        default=0.0,
        ge=0.0,
        le=500.0,
        description="CO2e emissions factor (kgCO2e/GJ)"
    )

    # Temperature levels
    T_supply_C: float = Field(..., description="Utility supply temperature")
    T_return_C: float = Field(..., description="Utility return temperature")

    # Constraints
    max_duty_kW: Optional[float] = Field(
        default=None,
        description="Maximum available duty"
    )
    min_duty_kW: float = Field(default=0.0, description="Minimum duty")


# =============================================================================
# EQUIPMENT SCHEMAS
# =============================================================================

class HeatExchanger(BaseModel):
    """
    Heat exchanger specification and performance.

    Represents either an existing exchanger (retrofit) or a
    proposed new exchanger in the HEN design.
    """

    exchanger_id: str = Field(..., description="Unique exchanger ID")
    exchanger_name: str = Field(default="", description="Human-readable name")
    exchanger_type: ExchangerType = Field(
        default=ExchangerType.SHELL_AND_TUBE
    )

    # Stream connections
    hot_stream_id: str = Field(..., description="Hot side stream ID")
    cold_stream_id: str = Field(..., description="Cold side stream ID")

    # Duty and temperatures
    duty_kW: float = Field(
        ...,
        ge=0.0,
        le=1e9,
        description="Heat duty (kW)"
    )
    hot_inlet_T_C: float = Field(..., description="Hot inlet temperature")
    hot_outlet_T_C: float = Field(..., description="Hot outlet temperature")
    cold_inlet_T_C: float = Field(..., description="Cold inlet temperature")
    cold_outlet_T_C: float = Field(..., description="Cold outlet temperature")

    # Approach temperatures
    delta_T_hot_end_C: float = Field(
        default=0.0,
        description="Hot end approach"
    )
    delta_T_cold_end_C: float = Field(
        default=0.0,
        description="Cold end approach"
    )
    LMTD_C: float = Field(default=0.0, description="Log mean temp difference")

    # Heat transfer
    UA_kW_K: float = Field(
        default=0.0,
        ge=0.0,
        description="Overall UA value (kW/K)"
    )
    U_W_m2K: float = Field(
        default=500.0,
        ge=10.0,
        le=10000.0,
        description="Overall heat transfer coefficient"
    )
    area_m2: float = Field(
        default=0.0,
        ge=0.0,
        le=100000.0,
        description="Heat transfer area (m²)"
    )

    # Flow arrangement
    flow_arrangement: FlowArrangement = Field(
        default=FlowArrangement.COUNTER_CURRENT
    )
    F_correction_factor: float = Field(
        default=1.0,
        ge=0.5,
        le=1.0,
        description="LMTD correction factor"
    )

    # Pressure drops
    hot_side_dp_kPa: float = Field(default=0.0, ge=0.0)
    cold_side_dp_kPa: float = Field(default=0.0, ge=0.0)

    # Fouling
    hot_fouling_m2K_W: float = Field(default=0.0001, ge=0.0)
    cold_fouling_m2K_W: float = Field(default=0.0001, ge=0.0)

    # Economics
    capital_cost_usd: float = Field(default=0.0, ge=0.0)
    installation_cost_usd: float = Field(default=0.0, ge=0.0)
    operating_cost_usd_yr: float = Field(default=0.0, ge=0.0)

    # Retrofit info
    is_existing: bool = Field(default=False, description="Existing equipment")
    is_reused: bool = Field(default=False, description="Reused in new design")

    # Exergy
    exergy_destruction_kW: float = Field(default=0.0, ge=0.0)
    exergy_efficiency: float = Field(default=0.0, ge=0.0, le=1.0)

    # Metadata
    provenance_hash: Optional[str] = Field(default=None)

    class Config:
        use_enum_values = True


class EquipmentConstraints(BaseModel):
    """Equipment and network constraints for optimization."""

    # Existing equipment
    existing_exchangers: List[HeatExchanger] = Field(
        default_factory=list,
        description="List of existing exchangers"
    )

    # New equipment limits
    max_new_exchangers: int = Field(
        default=20,
        ge=0,
        le=100,
        description="Maximum new exchangers allowed"
    )
    max_total_area_m2: Optional[float] = Field(
        default=None,
        description="Maximum total new area"
    )
    max_capex_usd: Optional[float] = Field(
        default=None,
        description="Capital budget constraint"
    )

    # Connectivity constraints
    allowed_matches: List[Tuple[str, str]] = Field(
        default_factory=list,
        description="Allowed stream pairs (retrofit)"
    )
    forbidden_matches: List[Tuple[str, str]] = Field(
        default_factory=list,
        description="Forbidden stream pairs"
    )

    # Maintenance windows
    maintenance_windows: Dict[str, List[Tuple[datetime, datetime]]] = Field(
        default_factory=dict,
        description="Equipment offline periods"
    )


# =============================================================================
# ANALYSIS RESULT SCHEMAS
# =============================================================================

class CompositePoint(BaseModel):
    """Point on composite curve."""

    temperature_C: float
    enthalpy_kW: float
    stream_ids: List[str] = Field(default_factory=list)


class PinchAnalysisResult(BaseModel):
    """
    Pinch analysis results.

    Contains composite curves, pinch temperature, and minimum
    utility requirements per pinch analysis methodology.
    """

    calculation_id: str = Field(
        default_factory=lambda: str(uuid.uuid4())
    )
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )

    # Pinch point
    pinch_temperature_C: float = Field(
        ...,
        description="Pinch temperature (°C)"
    )
    delta_t_min_C: float = Field(..., description="ΔTmin used")

    # Minimum utilities
    minimum_hot_utility_kW: float = Field(
        ...,
        ge=0.0,
        description="Minimum hot utility (kW)"
    )
    minimum_cold_utility_kW: float = Field(
        ...,
        ge=0.0,
        description="Minimum cold utility (kW)"
    )
    maximum_heat_recovery_kW: float = Field(
        ...,
        ge=0.0,
        description="Maximum recoverable heat (kW)"
    )

    # Composite curves
    hot_composite: List[CompositePoint] = Field(default_factory=list)
    cold_composite: List[CompositePoint] = Field(default_factory=list)
    grand_composite: List[CompositePoint] = Field(default_factory=list)

    # Stream data used
    hot_streams: List[str] = Field(default_factory=list)
    cold_streams: List[str] = Field(default_factory=list)

    # Heat cascade
    heat_cascade: List[Dict[str, float]] = Field(default_factory=list)

    # Utility breakdown
    utility_requirements: Dict[str, float] = Field(default_factory=dict)

    # Provenance
    input_hash: str = Field(default="", description="SHA-256 of inputs")
    output_hash: str = Field(default="", description="SHA-256 of outputs")
    formula_version: str = Field(default="PINCH_v1.0")

    # Validation
    is_valid: bool = Field(default=True)
    validation_messages: List[str] = Field(default_factory=list)

    class Config:
        use_enum_values = True


class ExergyAnalysisResult(BaseModel):
    """
    Exergy (second-law) analysis results.

    Quantifies thermodynamic irreversibility and identifies
    improvement opportunities.
    """

    calculation_id: str = Field(
        default_factory=lambda: str(uuid.uuid4())
    )
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )

    # Reference state
    reference_temperature_K: float = Field(default=298.15)
    reference_pressure_kPa: float = Field(default=101.325)

    # Total exergy metrics
    total_exergy_input_kW: float = Field(..., ge=0.0)
    total_exergy_output_kW: float = Field(..., ge=0.0)
    total_exergy_destruction_kW: float = Field(..., ge=0.0)
    exergy_efficiency: float = Field(..., ge=0.0, le=1.0)

    # Exergy by component
    exergy_by_exchanger: Dict[str, Dict[str, float]] = Field(
        default_factory=dict,
        description="Exergy destruction by exchanger ID"
    )
    exergy_by_utility: Dict[str, float] = Field(
        default_factory=dict,
        description="Exergy destruction by utility"
    )

    # Improvement potential
    improvement_potential_kW: float = Field(default=0.0, ge=0.0)
    improvement_potential_percent: float = Field(default=0.0, ge=0.0, le=100.0)

    # Entropy generation
    total_entropy_generation_kW_K: float = Field(default=0.0, ge=0.0)

    # Provenance
    input_hash: str = Field(default="")
    output_hash: str = Field(default="")
    formula_version: str = Field(default="EXERGY_v1.0")

    class Config:
        use_enum_values = True


class EconomicAnalysisResult(BaseModel):
    """
    Economic analysis results including capex, opex, and ROI metrics.
    """

    calculation_id: str = Field(
        default_factory=lambda: str(uuid.uuid4())
    )
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )

    # Capital costs
    total_capital_cost_usd: float = Field(..., ge=0.0)
    equipment_cost_usd: float = Field(default=0.0, ge=0.0)
    installation_cost_usd: float = Field(default=0.0, ge=0.0)
    piping_cost_usd: float = Field(default=0.0, ge=0.0)
    instrumentation_cost_usd: float = Field(default=0.0, ge=0.0)

    # Operating costs
    annual_operating_cost_usd: float = Field(..., ge=0.0)
    utility_cost_usd_yr: float = Field(default=0.0, ge=0.0)
    maintenance_cost_usd_yr: float = Field(default=0.0, ge=0.0)

    # Savings
    annual_utility_savings_usd: float = Field(default=0.0)
    annual_net_savings_usd: float = Field(default=0.0)

    # Investment metrics
    total_annual_cost_usd: float = Field(..., ge=0.0)
    payback_period_years: float = Field(default=0.0, ge=0.0)
    npv_usd: float = Field(default=0.0)
    irr_percent: Optional[float] = Field(default=None)
    roi_percent: float = Field(default=0.0)

    # Assumptions
    discount_rate: float = Field(default=0.10)
    project_lifetime_years: int = Field(default=20)
    operating_hours_per_year: int = Field(default=8000)

    # Emissions value
    co2_reduction_tonnes_yr: float = Field(default=0.0, ge=0.0)
    carbon_credit_value_usd_yr: float = Field(default=0.0, ge=0.0)

    # Provenance
    input_hash: str = Field(default="")
    output_hash: str = Field(default="")

    class Config:
        use_enum_values = True


# =============================================================================
# HEN DESIGN SCHEMAS
# =============================================================================

class HENDesign(BaseModel):
    """
    Complete heat exchanger network design.

    Represents the output of HEN synthesis including all exchangers,
    utility usage, and network topology.
    """

    design_id: str = Field(
        default_factory=lambda: str(uuid.uuid4())
    )
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )

    # Design metadata
    design_name: str = Field(default="", description="Design identifier")
    mode: OptimizationMode = Field(default=OptimizationMode.GRASSROOTS)

    # Network components
    exchangers: List[HeatExchanger] = Field(
        default_factory=list,
        description="All heat exchangers in network"
    )
    stream_splits: Dict[str, List[float]] = Field(
        default_factory=dict,
        description="Stream split fractions"
    )

    # Performance metrics
    total_heat_recovered_kW: float = Field(..., ge=0.0)
    hot_utility_required_kW: float = Field(..., ge=0.0)
    cold_utility_required_kW: float = Field(..., ge=0.0)

    # Equipment summary
    exchanger_count: int = Field(default=0, ge=0)
    new_exchanger_count: int = Field(default=0, ge=0)
    reused_exchanger_count: int = Field(default=0, ge=0)
    total_area_m2: float = Field(default=0.0, ge=0.0)

    # Economics
    economic_analysis: Optional[EconomicAnalysisResult] = Field(default=None)

    # Exergy
    exergy_analysis: Optional[ExergyAnalysisResult] = Field(default=None)

    # Constraint satisfaction
    pinch_violations: int = Field(default=0, ge=0)
    temperature_violations: int = Field(default=0, ge=0)
    all_constraints_satisfied: bool = Field(default=True)
    constraint_details: List[Dict[str, Any]] = Field(default_factory=list)

    # Provenance
    optimization_run_id: Optional[str] = Field(default=None)
    input_hash: str = Field(default="")
    output_hash: str = Field(default="")

    class Config:
        use_enum_values = True


# =============================================================================
# OPTIMIZATION SCHEMAS
# =============================================================================

class OptimizationRequest(BaseModel):
    """Request for heat recovery optimization."""

    request_id: str = Field(
        default_factory=lambda: str(uuid.uuid4())
    )
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )

    # Streams
    hot_streams: List[HeatStream] = Field(..., min_items=1)
    cold_streams: List[HeatStream] = Field(..., min_items=1)

    # Utilities
    hot_utilities: List[UtilityCost] = Field(default_factory=list)
    cold_utilities: List[UtilityCost] = Field(default_factory=list)

    # Equipment constraints
    equipment_constraints: Optional[EquipmentConstraints] = Field(default=None)

    # Optimization settings
    mode: OptimizationMode = Field(default=OptimizationMode.GRASSROOTS)
    objective: OptimizationObjective = Field(
        default=OptimizationObjective.MINIMIZE_COST
    )
    delta_t_min_C: float = Field(default=10.0, ge=1.0, le=100.0)

    # Options
    include_exergy_analysis: bool = Field(default=True)
    include_uncertainty: bool = Field(default=False)
    generate_pareto: bool = Field(default=False)
    n_pareto_points: int = Field(default=20, ge=5, le=100)

    # Solver settings
    max_time_seconds: float = Field(default=300.0, ge=10.0, le=3600.0)

    # Requester
    requested_by: str = Field(default="system")

    class Config:
        use_enum_values = True


class ParetoPoint(BaseModel):
    """Single point on Pareto frontier."""

    point_id: str = Field(default_factory=lambda: str(uuid.uuid4()))

    # Objective values
    objectives: Dict[str, float] = Field(..., description="Objective name -> value")

    # Design reference
    design_summary: Dict[str, Any] = Field(default_factory=dict)
    design: Optional[HENDesign] = Field(default=None)

    # Constraint satisfaction
    max_constraint_violation: float = Field(default=0.0, ge=0.0)
    is_feasible: bool = Field(default=True)

    # Ranking
    rank: int = Field(default=0, ge=0)
    crowding_distance: float = Field(default=0.0, ge=0.0)


class OptimizationResult(BaseModel):
    """Result from heat recovery optimization."""

    request_id: str = Field(..., description="Original request ID")
    optimization_id: str = Field(
        default_factory=lambda: str(uuid.uuid4())
    )
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    status: OptimizationStatus = Field(default=OptimizationStatus.COMPLETED)

    # Pinch analysis
    pinch_analysis: PinchAnalysisResult = Field(...)

    # Best design (or selected from Pareto)
    recommended_design: HENDesign = Field(...)

    # Pareto results (if multi-objective)
    pareto_points: List[ParetoPoint] = Field(default_factory=list)
    pareto_hypervolume: float = Field(default=0.0, ge=0.0)

    # Uncertainty results
    uncertainty_bounds: Dict[str, Tuple[float, float]] = Field(
        default_factory=dict,
        description="95% confidence intervals for key metrics"
    )
    robustness_score: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Design robustness under uncertainty"
    )

    # Execution details
    solver_used: str = Field(default="pulp_cbc")
    optimization_time_seconds: float = Field(default=0.0, ge=0.0)
    iterations: int = Field(default=0, ge=0)
    convergence_achieved: bool = Field(default=True)

    # Explanation
    explanation_summary: str = Field(default="")
    key_drivers: List[str] = Field(default_factory=list)
    binding_constraints: List[str] = Field(default_factory=list)

    # Implementation
    implementation_risk: str = Field(default="low")
    operator_approval_required: bool = Field(default=True)

    # Provenance
    input_hash: str = Field(default="")
    output_hash: str = Field(default="")
    configuration_version: str = Field(default="1.0.0")
    random_seed: Optional[int] = Field(default=None)

    class Config:
        use_enum_values = True


# =============================================================================
# EXPLAINABILITY SCHEMAS
# =============================================================================

class FeatureAttribution(BaseModel):
    """SHAP/LIME feature attribution."""

    feature_name: str
    feature_value: float
    attribution: float
    normalized_attribution: float = Field(default=0.0, ge=-1.0, le=1.0)


class ExplainabilityReport(BaseModel):
    """Comprehensive explainability report."""

    report_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    optimization_id: str = Field(...)

    # Engineering rationale
    constraint_rationale: List[str] = Field(default_factory=list)
    pinch_rule_explanations: List[str] = Field(default_factory=list)
    temperature_feasibility_notes: List[str] = Field(default_factory=list)

    # Statistical attribution
    shap_values: List[FeatureAttribution] = Field(default_factory=list)
    lime_values: List[FeatureAttribution] = Field(default_factory=list)

    # Sensitivity analysis
    sensitivity_analysis: Dict[str, Dict[str, float]] = Field(
        default_factory=dict
    )

    # Causal analysis
    causal_graph_available: bool = Field(default=False)
    causal_effects: Dict[str, float] = Field(default_factory=dict)

    # Counterfactuals
    counterfactual_scenarios: List[Dict[str, Any]] = Field(
        default_factory=list
    )

    # Summary
    executive_summary: str = Field(default="")

    # Audit fields
    provenance_hash: str = Field(default="")


# =============================================================================
# STATUS SCHEMAS
# =============================================================================

class AgentStatus(BaseModel):
    """GL-006 HEATRECLAIM agent status."""

    agent_id: str = Field(default="GL-006")
    agent_name: str = Field(default="HEATRECLAIM")
    agent_version: str = Field(default="1.0.0")

    # Health
    status: str = Field(default="running")
    health: str = Field(default="healthy")
    uptime_seconds: float = Field(default=0.0, ge=0.0)
    last_heartbeat: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )

    # Performance
    optimizations_performed: int = Field(default=0, ge=0)
    optimizations_successful: int = Field(default=0, ge=0)
    total_heat_recovered_GJ: float = Field(default=0.0, ge=0.0)
    total_cost_savings_usd: float = Field(default=0.0, ge=0.0)
    total_co2_avoided_tonnes: float = Field(default=0.0, ge=0.0)

    # Metrics
    avg_optimization_time_seconds: float = Field(default=0.0, ge=0.0)

    # Resources
    cpu_usage_percent: float = Field(default=0.0, ge=0.0, le=100.0)
    memory_usage_mb: float = Field(default=0.0, ge=0.0)

    # Integration
    kafka_connected: bool = Field(default=True)
    opcua_connected: bool = Field(default=True)
    graphql_ready: bool = Field(default=True)


class HealthCheckResponse(BaseModel):
    """Health check API response."""

    status: str = Field(default="healthy")
    version: str = Field(default="1.0.0")
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    uptime_seconds: float = Field(default=0.0)
    checks: Dict[str, str] = Field(default_factory=dict)


# =============================================================================
# EVENT SCHEMAS
# =============================================================================

class HeatReclaimEvent(BaseModel):
    """Event emitted by HEATRECLAIM agent."""

    event_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    event_type: str = Field(..., description="Event type")
    source: str = Field(default="GL-006")
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    severity: SeverityLevel = Field(default=SeverityLevel.INFO)
    payload: Dict[str, Any] = Field(default_factory=dict)
    correlation_id: Optional[str] = Field(default=None)

    class Config:
        use_enum_values = True


class CalculationEvent(BaseModel):
    """Calculation completion event for audit trail."""

    event_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    calculation_type: CalculationType = Field(...)

    # Inputs
    input_summary: Dict[str, Any] = Field(default_factory=dict)
    input_hash: str = Field(..., description="SHA-256 of inputs")

    # Outputs
    output_summary: Dict[str, Any] = Field(default_factory=dict)
    output_hash: str = Field(..., description="SHA-256 of outputs")

    # Provenance
    formula_id: str = Field(..., description="Formula/method identifier")
    formula_version: str = Field(default="1.0.0")
    deterministic: bool = Field(default=True)
    reproducible: bool = Field(default=True)

    # Performance
    calculation_time_ms: float = Field(default=0.0, ge=0.0)

    class Config:
        use_enum_values = True


# =============================================================================
# API RESPONSE SCHEMAS
# =============================================================================

class APIResponse(BaseModel):
    """Standard API response wrapper."""

    success: bool = Field(...)
    message: str = Field(default="")
    data: Optional[Any] = Field(default=None)
    errors: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    request_id: Optional[str] = Field(default=None)
    processing_time_ms: float = Field(default=0.0, ge=0.0)
