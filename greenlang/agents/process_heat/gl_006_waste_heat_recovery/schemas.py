"""
GL-006 WasteHeatRecovery Agent - Data Schemas Module

Centralized data schemas and models for type-safe data exchange
between modules and external systems.

This module provides:
    - Input/Output schemas for all analysis functions
    - API request/response models
    - Data validation schemas
    - Serialization helpers
    - Type definitions for external integrations
"""

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, Field, validator
import uuid


# =============================================================================
# ENUMS
# =============================================================================

class StreamTypeEnum(str, Enum):
    """Heat stream type classification."""
    HOT = "hot"
    COLD = "cold"


class FuelTypeEnum(str, Enum):
    """Fuel types for emissions calculation."""
    NATURAL_GAS = "natural_gas"
    DIESEL = "diesel"
    FUEL_OIL = "fuel_oil"
    COAL = "coal"
    PROPANE = "propane"
    ELECTRICITY = "electricity"
    STEAM = "steam"


class WasteHeatSourceTypeEnum(str, Enum):
    """Types of waste heat sources."""
    EXHAUST_GAS = "exhaust_gas"
    HOT_WATER = "hot_water"
    STEAM = "steam"
    PROCESS_AIR = "process_air"
    RADIATION = "radiation"
    CONDUCTION = "conduction"
    COOLING_WATER = "cooling_water"


class HeatSinkTypeEnum(str, Enum):
    """Types of heat sinks."""
    PROCESS_HEATING = "process_heating"
    PREHEATING = "preheating"
    SPACE_HEATING = "space_heating"
    DOMESTIC_HOT_WATER = "domestic_hot_water"
    ABSORPTION_COOLING = "absorption_cooling"
    POWER_GENERATION = "power_generation"


class AnalysisStatusEnum(str, Enum):
    """Analysis status codes."""
    SUCCESS = "success"
    WARNING = "warning"
    ERROR = "error"
    PARTIAL = "partial"


class FeasibilityLevelEnum(str, Enum):
    """Technical feasibility levels."""
    STRAIGHTFORWARD = "straightforward"
    MODERATE = "moderate"
    CHALLENGING = "challenging"
    NOT_FEASIBLE = "not_feasible"


class ComplexityLevelEnum(str, Enum):
    """Implementation complexity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    VERY_HIGH = "very_high"


# =============================================================================
# BASE SCHEMAS
# =============================================================================

class BaseSchema(BaseModel):
    """Base schema with common fields."""

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat(),
        }
        validate_assignment = True
        extra = "forbid"


class TimestampedSchema(BaseSchema):
    """Schema with timestamp."""
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="UTC timestamp"
    )


class IdentifiableSchema(TimestampedSchema):
    """Schema with ID and timestamp."""
    id: str = Field(
        default_factory=lambda: str(uuid.uuid4())[:8],
        description="Unique identifier"
    )


# =============================================================================
# INPUT SCHEMAS
# =============================================================================

class HeatStreamInput(BaseSchema):
    """Input schema for heat stream data."""

    name: str = Field(..., min_length=1, max_length=100, description="Stream name")
    stream_type: Optional[StreamTypeEnum] = Field(
        default=None,
        description="Hot or cold (auto-detected if not specified)"
    )
    supply_temp_f: float = Field(..., description="Supply temperature (F)")
    target_temp_f: float = Field(..., description="Target temperature (F)")
    mcp: float = Field(..., gt=0, description="Heat capacity flow rate (BTU/hr-F)")
    mass_flow_rate: Optional[float] = Field(
        default=None,
        gt=0,
        description="Mass flow rate (lb/hr)"
    )
    specific_heat: Optional[float] = Field(
        default=None,
        gt=0,
        description="Specific heat (BTU/lb-F)"
    )

    @validator("stream_type", always=True)
    def auto_detect_stream_type(cls, v, values):
        """Auto-detect stream type from temperatures."""
        if v is None:
            supply = values.get("supply_temp_f")
            target = values.get("target_temp_f")
            if supply is not None and target is not None:
                return StreamTypeEnum.HOT if supply > target else StreamTypeEnum.COLD
        return v


class WasteHeatSourceInput(BaseSchema):
    """Input schema for waste heat source."""

    source_id: str = Field(..., min_length=1, description="Source identifier")
    source_type: WasteHeatSourceTypeEnum = Field(..., description="Source type")
    temperature_f: float = Field(..., description="Source temperature (F)")
    flow_rate: float = Field(..., gt=0, description="Mass flow rate")
    flow_unit: str = Field(default="lb/hr", description="Flow rate unit")
    specific_heat: float = Field(
        default=0.24,
        gt=0,
        description="Specific heat (BTU/lb-F)"
    )
    availability_pct: float = Field(
        default=100.0,
        ge=0,
        le=100,
        description="Availability percentage"
    )
    operating_hours_yr: int = Field(
        default=8760,
        ge=0,
        le=8760,
        description="Annual operating hours"
    )
    min_discharge_temp_f: Optional[float] = Field(
        default=None,
        description="Minimum discharge temperature"
    )
    acid_dew_point_f: Optional[float] = Field(
        default=None,
        description="Acid dew point (for corrosion)"
    )


class HeatSinkInput(BaseSchema):
    """Input schema for heat sink."""

    sink_id: str = Field(..., min_length=1, description="Sink identifier")
    sink_type: HeatSinkTypeEnum = Field(..., description="Sink type")
    required_temperature_f: float = Field(..., description="Required temperature")
    inlet_temperature_f: float = Field(..., description="Inlet temperature")
    flow_rate: float = Field(..., gt=0, description="Mass flow rate")
    flow_unit: str = Field(default="lb/hr", description="Flow rate unit")
    specific_heat: float = Field(
        default=1.0,
        gt=0,
        description="Specific heat (BTU/lb-F)"
    )
    current_energy_source: FuelTypeEnum = Field(
        default=FuelTypeEnum.NATURAL_GAS,
        description="Current energy source"
    )
    current_cost_per_mmbtu: float = Field(
        default=5.0,
        ge=0,
        description="Current energy cost ($/MMBTU)"
    )


class ExergyStreamInput(BaseSchema):
    """Input schema for exergy analysis stream."""

    name: str = Field(..., min_length=1, description="Stream name")
    temp_f: float = Field(..., description="Temperature (F)")
    pressure_psia: float = Field(
        default=14.696,
        gt=0,
        description="Pressure (psia)"
    )
    mass_flow_lb_hr: float = Field(..., gt=0, description="Mass flow rate (lb/hr)")
    specific_heat_btu_lb_f: float = Field(
        default=0.24,
        gt=0,
        description="Specific heat (BTU/lb-F)"
    )
    is_inlet: bool = Field(default=True, description="Is inlet stream")
    chemical_exergy_btu_lb: float = Field(
        default=0.0,
        ge=0,
        description="Chemical exergy (BTU/lb)"
    )


class ProcessComponentInput(BaseSchema):
    """Input schema for process component."""

    name: str = Field(..., min_length=1, description="Component name")
    component_type: str = Field(..., description="Component type")
    inlet_streams: List[str] = Field(
        default_factory=list,
        description="Inlet stream names"
    )
    outlet_streams: List[str] = Field(
        default_factory=list,
        description="Outlet stream names"
    )
    heat_transfer_btu_hr: float = Field(
        default=0.0,
        description="Heat transfer rate"
    )
    heat_transfer_temp_f: Optional[float] = Field(
        default=None,
        description="Heat transfer temperature"
    )
    work_rate_btu_hr: float = Field(
        default=0.0,
        description="Work rate"
    )
    capital_cost_usd: float = Field(
        default=0.0,
        ge=0,
        description="Capital cost"
    )


class ProjectInput(BaseSchema):
    """Input schema for economic project."""

    name: str = Field(..., min_length=1, description="Project name")
    description: Optional[str] = Field(default=None, description="Description")
    capital_cost_usd: float = Field(..., ge=0, description="Capital cost")
    annual_operating_cost_usd: float = Field(
        default=0.0,
        ge=0,
        description="Annual O&M cost"
    )
    annual_energy_savings_usd: float = Field(
        ...,
        ge=0,
        description="Annual energy savings"
    )
    annual_energy_savings_mmbtu: Optional[float] = Field(
        default=None,
        ge=0,
        description="Annual energy savings (MMBTU)"
    )
    project_life_years: int = Field(
        default=20,
        ge=1,
        le=50,
        description="Project life"
    )
    co2_reduction_tons_yr: Optional[float] = Field(
        default=None,
        ge=0,
        description="Annual CO2 reduction"
    )


# =============================================================================
# OUTPUT SCHEMAS
# =============================================================================

class RecoveryOpportunityOutput(IdentifiableSchema):
    """Output schema for recovery opportunity."""

    source_id: str = Field(..., description="Source identifier")
    sink_id: str = Field(..., description="Sink identifier")
    recoverable_heat_btu_hr: float = Field(..., description="Recoverable heat rate")
    recoverable_heat_mmbtu_yr: float = Field(..., description="Annual recoverable heat")
    source_outlet_temp_f: float = Field(..., description="Source outlet temperature")
    sink_outlet_temp_f: float = Field(..., description="Sink outlet temperature")
    effectiveness: float = Field(..., ge=0, le=1, description="HX effectiveness")
    lmtd_f: float = Field(..., description="Log mean temperature difference")
    required_ua: float = Field(..., description="Required UA value")
    estimated_hx_area_ft2: float = Field(..., description="Estimated HX area")
    estimated_capital_cost: float = Field(..., description="Estimated capital cost")
    annual_savings: float = Field(..., description="Annual savings")
    simple_payback_years: float = Field(..., description="Simple payback period")
    npv_10yr: float = Field(..., description="10-year NPV")
    irr_pct: Optional[float] = Field(default=None, description="IRR percentage")
    technical_feasibility: FeasibilityLevelEnum = Field(..., description="Feasibility")
    implementation_complexity: ComplexityLevelEnum = Field(..., description="Complexity")
    notes: List[str] = Field(default_factory=list, description="Notes")


class PinchAnalysisOutput(IdentifiableSchema):
    """Output schema for pinch analysis."""

    pinch_temperature_f: float = Field(..., description="Pinch temperature")
    shifted_pinch_temp_f: float = Field(..., description="Shifted pinch temperature")
    pinch_above_temp_f: float = Field(..., description="Temperature above pinch")
    pinch_below_temp_f: float = Field(..., description="Temperature below pinch")
    minimum_hot_utility_btu_hr: float = Field(..., description="Min hot utility")
    minimum_cold_utility_btu_hr: float = Field(..., description="Min cold utility")
    maximum_heat_recovery_btu_hr: float = Field(..., description="Max heat recovery")
    total_hot_duty_btu_hr: float = Field(..., description="Total hot duty")
    total_cold_duty_btu_hr: float = Field(..., description="Total cold duty")
    delta_t_min_f: float = Field(..., description="Delta T minimum")
    stream_count: int = Field(..., description="Number of streams")
    provenance_hash: str = Field(..., description="Provenance hash")
    recommendations: List[str] = Field(default_factory=list, description="Recommendations")


class HENDesignOutput(IdentifiableSchema):
    """Output schema for HEN design."""

    total_units: int = Field(..., description="Total HX units")
    process_hx_units: int = Field(..., description="Process HX units")
    utility_hx_units: int = Field(..., description="Utility HX units")
    total_area_ft2: float = Field(..., description="Total HX area")
    total_capital_cost_usd: float = Field(..., description="Total capital cost")
    annual_utility_cost_usd: float = Field(..., description="Annual utility cost")
    total_heat_recovery_btu_hr: float = Field(..., description="Heat recovery rate")
    hot_utility_btu_hr: float = Field(..., description="Hot utility required")
    cold_utility_btu_hr: float = Field(..., description="Cold utility required")
    heat_recovery_fraction: float = Field(
        ...,
        ge=0,
        le=1,
        description="Recovery fraction"
    )
    is_pinch_compliant: bool = Field(..., description="Pinch compliance")
    provenance_hash: str = Field(..., description="Provenance hash")


class ExergyAnalysisOutput(IdentifiableSchema):
    """Output schema for exergy analysis."""

    dead_state_temp_f: float = Field(..., description="Dead state temperature")
    dead_state_pressure_psia: float = Field(..., description="Dead state pressure")
    total_exergy_input_btu_hr: float = Field(..., description="Total exergy input")
    total_exergy_output_btu_hr: float = Field(..., description="Total exergy output")
    total_exergy_destruction_btu_hr: float = Field(..., description="Total destruction")
    total_exergy_loss_btu_hr: float = Field(..., description="Total loss")
    second_law_efficiency_pct: float = Field(
        ...,
        ge=0,
        le=100,
        description="Second law efficiency"
    )
    total_improvement_potential_btu_hr: float = Field(
        ...,
        description="Improvement potential"
    )
    total_exergy_destruction_cost_usd_yr: float = Field(
        ...,
        description="Annual destruction cost"
    )
    provenance_hash: str = Field(..., description="Provenance hash")
    system_recommendations: List[str] = Field(
        default_factory=list,
        description="Recommendations"
    )


class EconomicAnalysisOutput(IdentifiableSchema):
    """Output schema for economic analysis."""

    project_name: str = Field(..., description="Project name")
    npv_usd: float = Field(..., description="Net Present Value")
    irr_pct: float = Field(..., description="Internal Rate of Return")
    mirr_pct: Optional[float] = Field(default=None, description="Modified IRR")
    simple_payback_years: float = Field(..., description="Simple payback")
    discounted_payback_years: Optional[float] = Field(
        default=None,
        description="Discounted payback"
    )
    roi_pct: float = Field(..., description="Return on Investment")
    savings_investment_ratio: float = Field(..., description="SIR")
    lcoe_usd_per_mmbtu: Optional[float] = Field(default=None, description="LCOE")
    total_lifetime_savings_usd: float = Field(..., description="Lifetime savings")
    total_lifetime_costs_usd: float = Field(..., description="Lifetime costs")
    net_benefit_usd: float = Field(..., description="Net benefit")
    economic_ranking: str = Field(..., description="Ranking")
    recommendation: str = Field(..., description="Recommendation")
    provenance_hash: str = Field(..., description="Provenance hash")


# =============================================================================
# API SCHEMAS
# =============================================================================

class AnalysisRequest(BaseSchema):
    """Generic analysis request schema."""

    request_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Request ID"
    )
    analysis_type: str = Field(..., description="Type of analysis")
    parameters: Dict[str, Any] = Field(
        default_factory=dict,
        description="Analysis parameters"
    )
    options: Dict[str, Any] = Field(
        default_factory=dict,
        description="Analysis options"
    )


class AnalysisResponse(IdentifiableSchema):
    """Generic analysis response schema."""

    request_id: str = Field(..., description="Original request ID")
    status: AnalysisStatusEnum = Field(..., description="Analysis status")
    result: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Analysis result"
    )
    errors: List[str] = Field(default_factory=list, description="Error messages")
    warnings: List[str] = Field(default_factory=list, description="Warning messages")
    execution_time_ms: float = Field(..., description="Execution time in ms")
    provenance_hash: str = Field(default="", description="Provenance hash")


class WasteHeatAnalysisRequest(BaseSchema):
    """Request schema for waste heat analysis."""

    sources: List[WasteHeatSourceInput] = Field(
        ...,
        min_items=1,
        description="Waste heat sources"
    )
    sinks: List[HeatSinkInput] = Field(
        ...,
        min_items=1,
        description="Heat sinks"
    )
    options: Dict[str, Any] = Field(
        default_factory=dict,
        description="Analysis options"
    )


class WasteHeatAnalysisResponse(IdentifiableSchema):
    """Response schema for waste heat analysis."""

    total_waste_heat_btu_hr: float = Field(..., description="Total waste heat")
    total_recoverable_btu_hr: float = Field(..., description="Total recoverable")
    recovery_potential_pct: float = Field(..., description="Recovery potential")
    opportunities: List[RecoveryOpportunityOutput] = Field(
        default_factory=list,
        description="Recovery opportunities"
    )
    pinch_temperature_f: Optional[float] = Field(
        default=None,
        description="Pinch temperature"
    )
    total_annual_savings: float = Field(..., description="Annual savings")
    total_capital_cost: float = Field(..., description="Total capital cost")
    portfolio_simple_payback: float = Field(..., description="Portfolio payback")
    recommendations: List[str] = Field(
        default_factory=list,
        description="Recommendations"
    )
    status: AnalysisStatusEnum = Field(..., description="Analysis status")
    provenance_hash: str = Field(default="", description="Provenance hash")


# =============================================================================
# SHAP EXPLAINABILITY SCHEMAS
# =============================================================================

class SHAPFeatureImportance(BaseSchema):
    """SHAP feature importance schema."""

    feature_name: str = Field(..., description="Feature name")
    shap_value: float = Field(..., description="SHAP value")
    feature_value: Any = Field(..., description="Feature value")
    contribution_direction: str = Field(
        ...,
        description="positive or negative"
    )
    importance_rank: int = Field(..., description="Importance rank")


class SHAPExplanation(IdentifiableSchema):
    """SHAP explanation output schema."""

    base_value: float = Field(..., description="Base (expected) value")
    predicted_value: float = Field(..., description="Predicted value")
    feature_importances: List[SHAPFeatureImportance] = Field(
        default_factory=list,
        description="Feature importances"
    )
    top_positive_features: List[str] = Field(
        default_factory=list,
        description="Top positive contributors"
    )
    top_negative_features: List[str] = Field(
        default_factory=list,
        description="Top negative contributors"
    )
    explanation_text: str = Field(
        default="",
        description="Human-readable explanation"
    )


# =============================================================================
# OPC-UA INTEGRATION SCHEMAS
# =============================================================================

class OPCUANodeValue(BaseSchema):
    """OPC-UA node value schema."""

    node_id: str = Field(..., description="Node identifier")
    value: Any = Field(..., description="Node value")
    data_type: str = Field(..., description="Data type")
    source_timestamp: datetime = Field(..., description="Source timestamp")
    server_timestamp: datetime = Field(..., description="Server timestamp")
    status_code: int = Field(..., description="OPC-UA status code")
    quality: str = Field(default="good", description="Data quality")


class OPCUASubscription(BaseSchema):
    """OPC-UA subscription schema."""

    subscription_id: str = Field(..., description="Subscription ID")
    node_ids: List[str] = Field(..., description="Subscribed node IDs")
    publishing_interval_ms: int = Field(
        default=1000,
        description="Publishing interval"
    )
    is_active: bool = Field(default=True, description="Subscription active")


class OPCUAConnectionStatus(BaseSchema):
    """OPC-UA connection status schema."""

    connected: bool = Field(..., description="Connection status")
    endpoint_url: str = Field(..., description="Server endpoint")
    server_state: str = Field(default="unknown", description="Server state")
    last_connected: Optional[datetime] = Field(
        default=None,
        description="Last connected time"
    )
    reconnect_attempts: int = Field(
        default=0,
        description="Reconnect attempts"
    )
    error_message: Optional[str] = Field(
        default=None,
        description="Error message"
    )


# =============================================================================
# VALIDATION HELPERS
# =============================================================================

def validate_streams_for_pinch(streams: List[HeatStreamInput]) -> List[str]:
    """Validate streams for pinch analysis."""
    errors = []

    if len(streams) < 2:
        errors.append("At least 2 streams required for pinch analysis")

    hot_streams = [s for s in streams if s.stream_type == StreamTypeEnum.HOT]
    cold_streams = [s for s in streams if s.stream_type == StreamTypeEnum.COLD]

    if not hot_streams:
        errors.append("At least one hot stream required")
    if not cold_streams:
        errors.append("At least one cold stream required")

    for stream in streams:
        if stream.stream_type == StreamTypeEnum.HOT:
            if stream.supply_temp_f <= stream.target_temp_f:
                errors.append(
                    f"Hot stream '{stream.name}' supply temp must be > target temp"
                )
        else:
            if stream.supply_temp_f >= stream.target_temp_f:
                errors.append(
                    f"Cold stream '{stream.name}' supply temp must be < target temp"
                )

    return errors


def validate_sources_and_sinks(
    sources: List[WasteHeatSourceInput],
    sinks: List[HeatSinkInput]
) -> List[str]:
    """Validate sources and sinks for waste heat analysis."""
    errors = []

    if not sources:
        errors.append("At least one waste heat source required")
    if not sinks:
        errors.append("At least one heat sink required")

    # Check for temperature feasibility
    if sources and sinks:
        max_source_temp = max(s.temperature_f for s in sources)
        min_sink_required = min(s.required_temperature_f for s in sinks)

        if max_source_temp < min_sink_required + 20:
            errors.append(
                f"No feasible heat recovery: max source temp ({max_source_temp}F) "
                f"is below min required sink temp ({min_sink_required}F) + approach"
            )

    return errors
