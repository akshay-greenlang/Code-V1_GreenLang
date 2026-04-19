"""
GL-011 FUELCRAFT - API Schemas

Pydantic models for API request/response validation with complete
provenance tracking via bundle_hash and input snapshot IDs.

All schemas include:
- schema_version for backward compatibility
- bundle_hash for complete audit trails
- input_snapshot_ids for data lineage tracking

Standards Compliance:
- ISO 14064 (GHG Quantification)
- GHG Protocol (Scope 1/2/3 boundaries)
- IEC 61511 (Functional Safety)
"""

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, Field, validator, root_validator
import hashlib
import uuid


# =============================================================================
# Enumerations
# =============================================================================

class RunStatus(str, Enum):
    """Optimization run status states."""
    PENDING = "pending"
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class OptimizationObjective(str, Enum):
    """Optimization objective types."""
    COST = "cost"
    CARBON = "carbon"
    BALANCED = "balanced"
    RELIABILITY = "reliability"


class FuelType(str, Enum):
    """Supported fuel types."""
    NATURAL_GAS = "natural_gas"
    FUEL_OIL_2 = "fuel_oil_2"
    FUEL_OIL_6 = "fuel_oil_6"
    COAL = "coal"
    BIOMASS = "biomass"
    HYDROGEN = "hydrogen"
    BIOGAS = "biogas"
    PROPANE = "propane"
    DIESEL = "diesel"


class EmissionScope(str, Enum):
    """GHG Protocol emission scopes."""
    SCOPE_1 = "scope_1"  # Direct emissions (combustion)
    SCOPE_2 = "scope_2"  # Indirect emissions (purchased energy)
    SCOPE_3 = "scope_3"  # Value chain emissions


class EmissionBoundary(str, Enum):
    """Emission calculation boundaries."""
    TTW = "tank_to_wheel"   # Direct combustion only
    WTT = "well_to_tank"    # Upstream only
    WTW = "well_to_wheel"   # Full lifecycle


# =============================================================================
# Base Models
# =============================================================================

class ProvenanceBase(BaseModel):
    """Base model with provenance tracking."""
    schema_version: str = Field("1.0.0", description="Schema version for compatibility")
    bundle_hash: Optional[str] = Field(None, description="SHA-256 hash of all inputs")

    def compute_bundle_hash(self) -> str:
        """Compute SHA-256 hash of model data for audit trail."""
        data_str = self.json(exclude={"bundle_hash"}, sort_keys=True)
        return hashlib.sha256(data_str.encode()).hexdigest()


# =============================================================================
# Time Window Models
# =============================================================================

class EffectiveTimeWindow(BaseModel):
    """Time window for optimization run."""
    start_time: datetime = Field(..., description="Start of optimization window")
    end_time: datetime = Field(..., description="End of optimization window")
    timezone: str = Field("UTC", description="Timezone for time window")
    granularity_minutes: int = Field(60, ge=15, le=1440, description="Time step granularity")

    @validator("end_time")
    def end_after_start(cls, v: datetime, values: Dict[str, Any]) -> datetime:
        """Validate end_time is after start_time."""
        if "start_time" in values and v <= values["start_time"]:
            raise ValueError("end_time must be after start_time")
        return v

    class Config:
        json_encoders = {datetime: lambda v: v.isoformat()}


# =============================================================================
# Constraint Models
# =============================================================================

class FuelConstraints(BaseModel):
    """Fuel-specific constraints for optimization."""
    fuel_type: FuelType = Field(..., description="Type of fuel")
    min_percentage: float = Field(0.0, ge=0.0, le=100.0, description="Minimum blend percentage")
    max_percentage: float = Field(100.0, ge=0.0, le=100.0, description="Maximum blend percentage")
    available_quantity_mmbtu: Optional[float] = Field(None, ge=0, description="Available quantity")
    lead_time_hours: Optional[float] = Field(None, ge=0, description="Procurement lead time")
    contract_minimum_mmbtu: Optional[float] = Field(None, ge=0, description="Contract minimum")
    contract_maximum_mmbtu: Optional[float] = Field(None, ge=0, description="Contract maximum")

    @validator("max_percentage")
    def max_ge_min(cls, v: float, values: Dict[str, Any]) -> float:
        """Validate max_percentage >= min_percentage."""
        if "min_percentage" in values and v < values["min_percentage"]:
            raise ValueError("max_percentage must be >= min_percentage")
        return v


class CarbonConstraints(BaseModel):
    """Carbon emission constraints."""
    max_emissions_mtco2e: Optional[float] = Field(
        None, ge=0, description="Maximum allowed emissions (metric tons CO2e)"
    )
    emission_boundary: EmissionBoundary = Field(
        EmissionBoundary.WTW, description="Emission calculation boundary"
    )
    carbon_price_usd_per_ton: Optional[float] = Field(
        None, ge=0, description="Internal carbon price for optimization"
    )
    renewable_minimum_percent: Optional[float] = Field(
        None, ge=0, le=100, description="Minimum renewable fuel percentage"
    )
    scope_1_limit_mtco2e: Optional[float] = Field(None, ge=0, description="Scope 1 limit")
    scope_2_limit_mtco2e: Optional[float] = Field(None, ge=0, description="Scope 2 limit")
    scope_3_limit_mtco2e: Optional[float] = Field(None, ge=0, description="Scope 3 limit")


class OperationalConstraints(BaseModel):
    """Operational constraints for the optimization."""
    min_fuel_pressure_psig: float = Field(15.0, ge=0, description="Minimum fuel pressure")
    max_fuel_pressure_psig: float = Field(100.0, ge=0, description="Maximum fuel pressure")
    min_heating_value_btu_scf: Optional[float] = Field(None, ge=0, description="Minimum LHV")
    max_heating_value_btu_scf: Optional[float] = Field(None, ge=0, description="Maximum LHV")
    max_sulfur_ppm: Optional[float] = Field(None, ge=0, description="Maximum sulfur content")
    storage_capacity_mmbtu: Optional[float] = Field(None, ge=0, description="Storage capacity")
    min_inventory_mmbtu: Optional[float] = Field(None, ge=0, description="Minimum inventory")


# =============================================================================
# Input Data Models
# =============================================================================

class FuelPriceInput(BaseModel):
    """Fuel price data input."""
    fuel_type: FuelType
    spot_price_usd_mmbtu: float = Field(..., ge=0, description="Spot price")
    forward_curve: Optional[Dict[str, float]] = Field(
        None, description="Forward price curve (date -> price)"
    )
    basis_differential_usd_mmbtu: float = Field(0.0, description="Location basis")
    transport_cost_usd_mmbtu: float = Field(0.0, ge=0, description="Transport cost")
    price_timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Price timestamp"
    )


class FuelPropertiesInput(BaseModel):
    """Fuel properties input."""
    fuel_type: FuelType
    lhv_btu_per_unit: float = Field(..., gt=0, description="Lower heating value")
    hhv_btu_per_unit: float = Field(..., gt=0, description="Higher heating value")
    density_lb_per_gallon: Optional[float] = Field(None, gt=0, description="Density")
    sulfur_ppm: float = Field(0.0, ge=0, description="Sulfur content")
    carbon_content_percent: float = Field(..., ge=0, le=100, description="Carbon content")
    moisture_percent: float = Field(0.0, ge=0, le=100, description="Moisture content")
    ash_percent: float = Field(0.0, ge=0, le=100, description="Ash content")
    unit: str = Field("scf", description="Unit for LHV/HHV (scf, gallon, ton, etc.)")


class EmissionFactorInput(BaseModel):
    """Emission factor input per ISO 14064."""
    fuel_type: FuelType
    co2_kg_per_mmbtu: float = Field(..., ge=0, description="CO2 emission factor")
    ch4_kg_per_mmbtu: float = Field(0.0, ge=0, description="CH4 emission factor")
    n2o_kg_per_mmbtu: float = Field(0.0, ge=0, description="N2O emission factor")
    boundary: EmissionBoundary = Field(..., description="Emission boundary")
    source: str = Field(..., description="Data source (EPA, IPCC, custom)")
    gwp_ar: str = Field("AR5", description="GWP Assessment Report version")

    def calculate_co2e_per_mmbtu(self) -> float:
        """Calculate CO2e using GWP factors (AR5)."""
        # AR5 GWP values: CH4=28, N2O=265
        gwp_ch4 = 28 if self.gwp_ar == "AR5" else 25  # AR4
        gwp_n2o = 265 if self.gwp_ar == "AR5" else 298  # AR4
        return self.co2_kg_per_mmbtu + (self.ch4_kg_per_mmbtu * gwp_ch4) + (self.n2o_kg_per_mmbtu * gwp_n2o)


class DemandForecastInput(BaseModel):
    """Demand forecast / burn plan input."""
    time_periods: List[datetime] = Field(..., description="Forecast time periods")
    demand_mmbtu: List[float] = Field(..., description="Demand per period (MMBtu)")
    uncertainty_percent: List[float] = Field(
        default=None, description="Demand uncertainty per period"
    )

    @validator("demand_mmbtu")
    def validate_demand_length(cls, v: List[float], values: Dict[str, Any]) -> List[float]:
        """Validate demand_mmbtu length matches time_periods."""
        if "time_periods" in values and len(v) != len(values["time_periods"]):
            raise ValueError("demand_mmbtu length must match time_periods length")
        return v


class InventoryInput(BaseModel):
    """Current inventory state input."""
    fuel_type: FuelType
    current_level_mmbtu: float = Field(..., ge=0, description="Current inventory level")
    tank_capacity_mmbtu: float = Field(..., gt=0, description="Tank capacity")
    min_operating_level_mmbtu: float = Field(0.0, ge=0, description="Minimum operating level")
    last_updated: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Last update timestamp"
    )


class ContractTermsInput(BaseModel):
    """Contract terms input."""
    contract_id: str = Field(..., description="Contract identifier")
    fuel_type: FuelType
    supplier: str = Field(..., description="Supplier name")
    take_or_pay_mmbtu: Optional[float] = Field(None, ge=0, description="Take-or-pay quantity")
    max_daily_quantity_mmbtu: Optional[float] = Field(None, ge=0, description="Max daily quantity")
    price_formula: Optional[str] = Field(None, description="Price formula reference")
    validity_start: datetime = Field(..., description="Contract start date")
    validity_end: datetime = Field(..., description="Contract end date")


# =============================================================================
# Request Models
# =============================================================================

class RunRequest(ProvenanceBase):
    """
    Optimization run request.

    This is the primary request model for creating a fuel mix optimization run.
    All inputs are captured for complete provenance tracking.
    """
    run_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique run identifier"
    )
    effective_time_window: EffectiveTimeWindow = Field(
        ..., description="Time window for optimization"
    )
    objective: OptimizationObjective = Field(
        OptimizationObjective.BALANCED,
        description="Optimization objective"
    )

    # Constraints
    fuel_constraints: List[FuelConstraints] = Field(
        default=[], description="Fuel-specific constraints"
    )
    carbon_constraints: Optional[CarbonConstraints] = Field(
        None, description="Carbon emission constraints"
    )
    operational_constraints: Optional[OperationalConstraints] = Field(
        None, description="Operational constraints"
    )

    # Input data
    fuel_prices: List[FuelPriceInput] = Field(..., description="Fuel price data")
    fuel_properties: List[FuelPropertiesInput] = Field(..., description="Fuel properties")
    emission_factors: List[EmissionFactorInput] = Field(..., description="Emission factors")
    demand_forecast: DemandForecastInput = Field(..., description="Demand forecast")
    inventory: List[InventoryInput] = Field(default=[], description="Current inventory")
    contract_terms: List[ContractTermsInput] = Field(default=[], description="Contract terms")

    # Metadata
    requester_id: str = Field(..., description="ID of requester (user/system)")
    request_timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Request timestamp"
    )
    priority: int = Field(1, ge=1, le=5, description="Request priority (1=highest)")
    callback_url: Optional[str] = Field(None, description="Webhook callback URL")

    # Input snapshot tracking
    input_snapshot_ids: Dict[str, str] = Field(
        default_factory=dict,
        description="SHA-256 hashes of individual input components"
    )

    @root_validator
    def compute_input_hashes(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        """Compute hashes for all input components."""
        import json

        def hash_component(data: Any) -> str:
            if data is None:
                return ""
            if isinstance(data, list):
                data_str = json.dumps([item.dict() if hasattr(item, 'dict') else item for item in data], sort_keys=True, default=str)
            elif hasattr(data, 'dict'):
                data_str = json.dumps(data.dict(), sort_keys=True, default=str)
            else:
                data_str = json.dumps(data, sort_keys=True, default=str)
            return hashlib.sha256(data_str.encode()).hexdigest()[:16]

        values["input_snapshot_ids"] = {
            "fuel_prices": hash_component(values.get("fuel_prices")),
            "fuel_properties": hash_component(values.get("fuel_properties")),
            "emission_factors": hash_component(values.get("emission_factors")),
            "demand_forecast": hash_component(values.get("demand_forecast")),
            "inventory": hash_component(values.get("inventory")),
            "contract_terms": hash_component(values.get("contract_terms")),
        }
        return values

    class Config:
        json_encoders = {datetime: lambda v: v.isoformat()}
        schema_extra = {
            "example": {
                "run_id": "run-2024-001",
                "schema_version": "1.0.0",
                "effective_time_window": {
                    "start_time": "2024-01-15T00:00:00Z",
                    "end_time": "2024-01-16T00:00:00Z",
                    "granularity_minutes": 60
                },
                "objective": "balanced",
                "fuel_prices": [
                    {"fuel_type": "natural_gas", "spot_price_usd_mmbtu": 3.50}
                ],
                "fuel_properties": [
                    {"fuel_type": "natural_gas", "lhv_btu_per_unit": 1020, "hhv_btu_per_unit": 1120, "carbon_content_percent": 75.0}
                ],
                "emission_factors": [
                    {"fuel_type": "natural_gas", "co2_kg_per_mmbtu": 53.06, "boundary": "well_to_wheel", "source": "EPA"}
                ],
                "demand_forecast": {
                    "time_periods": ["2024-01-15T00:00:00Z"],
                    "demand_mmbtu": [1000.0]
                },
                "requester_id": "operator-001"
            }
        }


# =============================================================================
# Output Models
# =============================================================================

class BlendRatioOutput(BaseModel):
    """Optimized blend ratio for a fuel type."""
    fuel_type: FuelType
    percentage: float = Field(..., ge=0, le=100, description="Blend percentage")
    quantity_mmbtu: float = Field(..., ge=0, description="Quantity in MMBtu")
    quantity_unit: str = Field("mmbtu", description="Quantity unit")
    cost_usd: float = Field(..., ge=0, description="Cost in USD")
    emissions_mtco2e: float = Field(..., ge=0, description="Emissions in metric tons CO2e")


class FuelMixOutput(BaseModel):
    """Complete fuel mix output for a time period."""
    period_start: datetime
    period_end: datetime
    blend_ratios: List[BlendRatioOutput]
    total_quantity_mmbtu: float = Field(..., ge=0)
    total_cost_usd: float = Field(..., ge=0)
    total_emissions_mtco2e: float = Field(..., ge=0)
    weighted_lhv_btu_per_unit: float = Field(..., gt=0)
    meets_constraints: bool = Field(True)
    constraint_violations: List[str] = Field(default=[])


class CostBreakdown(BaseModel):
    """Detailed cost breakdown."""
    fuel_costs_usd: float = Field(..., ge=0, description="Total fuel costs")
    transport_costs_usd: float = Field(0.0, ge=0, description="Transport costs")
    storage_costs_usd: float = Field(0.0, ge=0, description="Storage costs")
    carbon_costs_usd: float = Field(0.0, ge=0, description="Carbon costs (if priced)")
    contract_penalties_usd: float = Field(0.0, ge=0, description="Contract penalties")
    total_cost_usd: float = Field(..., ge=0, description="Total cost")
    cost_per_mmbtu: float = Field(..., ge=0, description="Weighted average cost per MMBtu")

    # Cost by fuel type
    cost_by_fuel: Dict[str, float] = Field(default_factory=dict)

    # Comparison metrics
    baseline_cost_usd: Optional[float] = Field(None, ge=0, description="Baseline cost")
    savings_usd: Optional[float] = Field(None, description="Cost savings vs baseline")
    savings_percent: Optional[float] = Field(None, description="Savings percentage")


class CarbonFootprint(BaseModel):
    """Carbon footprint breakdown per ISO 14064."""
    total_emissions_mtco2e: float = Field(..., ge=0, description="Total emissions")

    # By scope (GHG Protocol)
    scope_1_mtco2e: float = Field(..., ge=0, description="Scope 1 (direct)")
    scope_2_mtco2e: float = Field(0.0, ge=0, description="Scope 2 (indirect)")
    scope_3_mtco2e: float = Field(0.0, ge=0, description="Scope 3 (value chain)")

    # By boundary
    ttw_emissions_mtco2e: float = Field(..., ge=0, description="Tank-to-wheel emissions")
    wtt_emissions_mtco2e: float = Field(0.0, ge=0, description="Well-to-tank emissions")
    wtw_emissions_mtco2e: float = Field(..., ge=0, description="Well-to-wheel emissions")

    # By gas
    co2_mt: float = Field(..., ge=0)
    ch4_mt: float = Field(0.0, ge=0)
    n2o_mt: float = Field(0.0, ge=0)

    # By fuel
    emissions_by_fuel: Dict[str, float] = Field(default_factory=dict)

    # Intensity metrics
    carbon_intensity_kgco2e_per_mmbtu: float = Field(..., ge=0)

    # Comparison metrics
    baseline_emissions_mtco2e: Optional[float] = Field(None, ge=0)
    reduction_mtco2e: Optional[float] = Field(None)
    reduction_percent: Optional[float] = Field(None)


class ProcurementRecommendation(BaseModel):
    """Fuel procurement recommendation."""
    fuel_type: FuelType
    recommended_quantity_mmbtu: float = Field(..., ge=0)
    recommended_timing: datetime
    supplier: Optional[str] = Field(None)
    estimated_price_usd_mmbtu: float = Field(..., ge=0)
    contract_reference: Optional[str] = Field(None)
    urgency: str = Field("normal", description="low, normal, high, critical")
    notes: Optional[str] = Field(None)


class ExplainabilityOutput(BaseModel):
    """SHAP-based explainability output."""
    run_id: str
    feature_importance: Dict[str, float] = Field(
        ..., description="Feature importance scores"
    )
    top_drivers: List[Dict[str, Any]] = Field(
        ..., description="Top decision drivers with SHAP values"
    )
    sensitivity_analysis: Dict[str, Dict[str, float]] = Field(
        default_factory=dict, description="Sensitivity to key parameters"
    )
    decision_rationale: str = Field(
        ..., description="Human-readable decision rationale"
    )
    counterfactuals: List[Dict[str, Any]] = Field(
        default=[], description="Alternative scenarios"
    )
    calculation_hash: str = Field(
        ..., description="SHA-256 hash for reproducibility"
    )


# =============================================================================
# Response Models
# =============================================================================

class RunResponse(ProvenanceBase):
    """Response for run creation."""
    run_id: str = Field(..., description="Assigned run ID")
    status: RunStatus = Field(..., description="Current run status")
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    estimated_completion: Optional[datetime] = Field(None)
    queue_position: Optional[int] = Field(None)
    input_snapshot_ids: Dict[str, str] = Field(default_factory=dict)

    class Config:
        json_encoders = {datetime: lambda v: v.isoformat()}


class RunStatusResponse(ProvenanceBase):
    """Response for run status query."""
    run_id: str
    status: RunStatus
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    progress_percent: float = Field(0.0, ge=0, le=100)
    current_step: Optional[str] = None
    error_message: Optional[str] = None
    input_snapshot_ids: Dict[str, str] = Field(default_factory=dict)

    class Config:
        json_encoders = {datetime: lambda v: v.isoformat()}


class RecommendationResponse(ProvenanceBase):
    """Complete recommendation response."""
    run_id: str
    status: RunStatus
    effective_time_window: EffectiveTimeWindow

    # Optimized fuel mix
    fuel_mix: List[FuelMixOutput] = Field(..., description="Optimized fuel mix by period")

    # Aggregated results
    total_cost: CostBreakdown
    total_carbon: CarbonFootprint

    # Procurement recommendations
    procurement_recommendations: List[ProcurementRecommendation] = Field(default=[])

    # Metadata
    optimization_time_ms: float = Field(..., ge=0)
    solver_status: str
    objective_value: float

    # Provenance
    input_snapshot_ids: Dict[str, str] = Field(default_factory=dict)
    calculation_hash: str = Field(..., description="SHA-256 of calculation")
    model_version: str = Field("1.0.0")

    class Config:
        json_encoders = {datetime: lambda v: v.isoformat()}


class ExplainabilityResponse(ProvenanceBase):
    """Explainability endpoint response."""
    run_id: str
    explainability: ExplainabilityOutput

    class Config:
        json_encoders = {datetime: lambda v: v.isoformat()}


# =============================================================================
# Health Check Models
# =============================================================================

class ComponentHealth(BaseModel):
    """Individual component health status."""
    name: str
    status: str = Field(..., description="healthy, degraded, unhealthy")
    latency_ms: Optional[float] = None
    last_check: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    message: Optional[str] = None


class HealthResponse(BaseModel):
    """Liveness probe response."""
    status: str = Field("healthy", description="healthy, unhealthy")
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    version: str = Field("1.0.0")


class ReadinessResponse(BaseModel):
    """Readiness probe response with component status."""
    status: str = Field(..., description="ready, not_ready")
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    version: str = Field("1.0.0")
    components: List[ComponentHealth] = Field(default=[])
    uptime_seconds: float = Field(..., ge=0)


# =============================================================================
# Error Models
# =============================================================================

class ErrorResponse(BaseModel):
    """Standard error response."""
    error: str = Field(..., description="Error type")
    code: int = Field(..., description="HTTP status code")
    message: str = Field(..., description="Error message")
    details: Optional[Dict[str, Any]] = Field(None, description="Additional details")
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    request_id: Optional[str] = Field(None, description="Request ID for tracing")

    class Config:
        json_encoders = {datetime: lambda v: v.isoformat()}


class ValidationErrorDetail(BaseModel):
    """Validation error detail."""
    loc: List[Union[str, int]] = Field(..., description="Error location path")
    msg: str = Field(..., description="Error message")
    type: str = Field(..., description="Error type")


class ValidationErrorResponse(BaseModel):
    """Validation error response."""
    error: str = Field("validation_error")
    code: int = Field(422)
    message: str = Field("Request validation failed")
    details: List[ValidationErrorDetail] = Field(...)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    request_id: Optional[str] = Field(None)

    class Config:
        json_encoders = {datetime: lambda v: v.isoformat()}
