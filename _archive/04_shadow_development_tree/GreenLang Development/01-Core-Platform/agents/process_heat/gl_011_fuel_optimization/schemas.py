"""
GL-011 FUELCRAFT - Schema Definitions

Pydantic models for fuel optimization inputs, outputs, and results.
These schemas define the data contracts for the FuelOptimizationAgent.
"""

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple
import uuid

from pydantic import BaseModel, Field, validator


class FuelStatus(Enum):
    """Fuel system status."""
    ONLINE = "online"
    STANDBY = "standby"
    TRANSITIONING = "transitioning"
    OFFLINE = "offline"
    MAINTENANCE = "maintenance"
    EMERGENCY = "emergency"


class BlendStatus(Enum):
    """Blend optimization status."""
    OPTIMAL = "optimal"
    SUB_OPTIMAL = "sub_optimal"
    INFEASIBLE = "infeasible"
    MANUAL_OVERRIDE = "manual_override"


class InventoryAlertType(Enum):
    """Inventory alert types."""
    LOW_LEVEL = "low_level"
    CRITICAL_LEVEL = "critical_level"
    HIGH_LEVEL = "high_level"
    DELIVERY_REQUIRED = "delivery_required"
    DELIVERY_SCHEDULED = "delivery_scheduled"
    QUALITY_ISSUE = "quality_issue"


class FuelProperties(BaseModel):
    """Physical and chemical properties of a fuel."""

    fuel_type: str = Field(..., description="Fuel type identifier")
    fuel_name: str = Field(default="", description="Human-readable fuel name")

    # Heating values
    hhv_btu_lb: Optional[float] = Field(
        default=None,
        gt=0,
        description="Higher Heating Value (BTU/lb)"
    )
    hhv_btu_scf: Optional[float] = Field(
        default=None,
        gt=0,
        description="Higher Heating Value (BTU/SCF) for gases"
    )
    lhv_btu_lb: Optional[float] = Field(
        default=None,
        gt=0,
        description="Lower Heating Value (BTU/lb)"
    )
    lhv_btu_scf: Optional[float] = Field(
        default=None,
        gt=0,
        description="Lower Heating Value (BTU/SCF) for gases"
    )

    # Physical properties
    density_lb_scf: Optional[float] = Field(
        default=None,
        gt=0,
        description="Density (lb/SCF) for gases"
    )
    density_lb_gal: Optional[float] = Field(
        default=None,
        gt=0,
        description="Density (lb/gal) for liquids"
    )
    specific_gravity: Optional[float] = Field(
        default=None,
        gt=0,
        description="Specific gravity"
    )

    # Wobbe Index
    wobbe_index: Optional[float] = Field(
        default=None,
        description="Wobbe Index (BTU/SCF)"
    )
    wobbe_index_modified: Optional[float] = Field(
        default=None,
        description="Modified Wobbe Index"
    )

    # Composition (for gases)
    methane_pct: Optional[float] = Field(default=None, ge=0, le=100)
    ethane_pct: Optional[float] = Field(default=None, ge=0, le=100)
    propane_pct: Optional[float] = Field(default=None, ge=0, le=100)
    butane_pct: Optional[float] = Field(default=None, ge=0, le=100)
    nitrogen_pct: Optional[float] = Field(default=None, ge=0, le=100)
    co2_pct: Optional[float] = Field(default=None, ge=0, le=100)
    hydrogen_pct: Optional[float] = Field(default=None, ge=0, le=100)

    # Emissions
    co2_kg_mmbtu: Optional[float] = Field(
        default=None,
        ge=0,
        description="CO2 emission factor (kg/MMBTU)"
    )
    nox_lb_mmbtu: Optional[float] = Field(
        default=None,
        ge=0,
        description="NOx emission factor (lb/MMBTU)"
    )
    so2_lb_mmbtu: Optional[float] = Field(
        default=None,
        ge=0,
        description="SO2 emission factor (lb/MMBTU)"
    )

    # Quality
    sulfur_content_ppm: Optional[float] = Field(
        default=None,
        ge=0,
        description="Sulfur content (ppm)"
    )
    moisture_pct: Optional[float] = Field(
        default=None,
        ge=0,
        le=100,
        description="Moisture content (%)"
    )
    ash_pct: Optional[float] = Field(
        default=None,
        ge=0,
        le=100,
        description="Ash content (%)"
    )


class FuelPrice(BaseModel):
    """Current fuel price data."""

    fuel_type: str = Field(..., description="Fuel type identifier")
    price: float = Field(..., ge=0, description="Price value")
    unit: str = Field(default="USD/MMBTU", description="Price unit")
    source: str = Field(..., description="Price source")
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Price timestamp"
    )
    effective_until: Optional[datetime] = Field(
        default=None,
        description="Price valid until"
    )

    # Breakdown
    commodity_price: float = Field(..., ge=0, description="Base commodity price")
    transport_cost: float = Field(default=0.0, ge=0, description="Transport cost")
    basis_differential: float = Field(default=0.0, description="Regional basis")
    taxes: float = Field(default=0.0, ge=0, description="Applicable taxes")

    # Metadata
    contract_id: Optional[str] = Field(default=None, description="Contract reference")
    spot_or_contract: str = Field(default="spot", description="spot or contract")
    confidence: float = Field(default=1.0, ge=0, le=1, description="Price confidence")


class BlendRecommendation(BaseModel):
    """Fuel blend recommendation."""

    recommendation_id: str = Field(
        default_factory=lambda: str(uuid.uuid4())[:8],
        description="Recommendation identifier"
    )
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Recommendation timestamp"
    )
    status: BlendStatus = Field(
        default=BlendStatus.OPTIMAL,
        description="Optimization status"
    )

    # Blend composition
    blend_ratios: Dict[str, float] = Field(
        ...,
        description="Fuel type to percentage mapping"
    )
    primary_fuel: str = Field(..., description="Primary fuel in blend")

    # Calculated properties
    blended_hhv: float = Field(..., description="Blended HHV (BTU/SCF or BTU/lb)")
    blended_wobbe_index: Optional[float] = Field(
        default=None,
        description="Blended Wobbe Index"
    )
    blended_co2_factor: float = Field(
        ...,
        description="Blended CO2 emission factor (kg/MMBTU)"
    )

    # Cost analysis
    blended_cost_usd_mmbtu: float = Field(
        ...,
        ge=0,
        description="Blended fuel cost ($/MMBTU)"
    )
    cost_savings_usd_hr: Optional[float] = Field(
        default=None,
        description="Cost savings vs single fuel ($/hr)"
    )
    emissions_reduction_pct: Optional[float] = Field(
        default=None,
        description="Emissions reduction percentage"
    )

    # Constraints satisfaction
    wobbe_in_range: bool = Field(default=True, description="Wobbe Index in range")
    hhv_in_range: bool = Field(default=True, description="HHV in range")
    emissions_in_range: bool = Field(default=True, description="Emissions in range")

    # Implementation
    current_blend: Optional[Dict[str, float]] = Field(
        default=None,
        description="Current actual blend"
    )
    transition_time_minutes: Optional[int] = Field(
        default=None,
        description="Time to achieve recommended blend"
    )

    class Config:
        use_enum_values = True

    @validator("blend_ratios")
    def validate_blend_ratios(cls, v):
        """Ensure blend ratios sum to 100%."""
        total = sum(v.values())
        if abs(total - 100.0) > 0.1:
            raise ValueError(f"Blend ratios must sum to 100%, got {total}")
        return v


class SwitchingRecommendation(BaseModel):
    """Fuel switching recommendation."""

    recommendation_id: str = Field(
        default_factory=lambda: str(uuid.uuid4())[:8],
        description="Recommendation identifier"
    )
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Recommendation timestamp"
    )

    # Switch details
    recommended: bool = Field(..., description="Is switch recommended")
    current_fuel: str = Field(..., description="Current fuel type")
    recommended_fuel: str = Field(..., description="Recommended fuel type")
    trigger_reason: str = Field(..., description="Reason for recommendation")

    # Economic analysis
    current_cost_usd_hr: float = Field(..., ge=0, description="Current fuel cost")
    recommended_cost_usd_hr: float = Field(..., ge=0, description="Recommended fuel cost")
    savings_usd_hr: float = Field(..., description="Potential savings")
    payback_hours: Optional[float] = Field(
        default=None,
        description="Payback period"
    )

    # Transition details
    transition_time_minutes: int = Field(..., ge=0, description="Transition time")
    transition_cost_usd: float = Field(default=0.0, ge=0, description="Transition cost")
    efficiency_impact_pct: float = Field(
        default=0.0,
        description="Efficiency change (%)"
    )

    # Safety
    safety_checks_passed: bool = Field(default=True, description="Safety checks OK")
    safety_warnings: List[str] = Field(
        default_factory=list,
        description="Safety warnings"
    )
    requires_purge: bool = Field(default=False, description="Requires purge cycle")

    # Status
    operator_approval_required: bool = Field(
        default=True,
        description="Requires operator approval"
    )
    valid_until: datetime = Field(
        ...,
        description="Recommendation valid until"
    )


class InventoryStatus(BaseModel):
    """Fuel inventory status."""

    tank_id: str = Field(..., description="Tank identifier")
    fuel_type: str = Field(..., description="Fuel type in tank")
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Status timestamp"
    )

    # Levels
    current_level_gal: float = Field(..., ge=0, description="Current level (gallons)")
    current_level_pct: float = Field(..., ge=0, le=100, description="Current level (%)")
    capacity_gal: float = Field(..., gt=0, description="Tank capacity (gallons)")
    usable_capacity_gal: float = Field(
        ...,
        gt=0,
        description="Usable capacity (gallons)"
    )

    # Consumption
    consumption_rate_gal_hr: float = Field(
        default=0.0,
        ge=0,
        description="Current consumption rate"
    )
    avg_daily_consumption_gal: float = Field(
        default=0.0,
        ge=0,
        description="Average daily consumption"
    )
    days_of_supply: float = Field(
        default=0.0,
        ge=0,
        description="Days of supply remaining"
    )

    # Thresholds
    reorder_point_gal: float = Field(..., ge=0, description="Reorder point (gallons)")
    safety_stock_gal: float = Field(..., ge=0, description="Safety stock (gallons)")
    critical_level_gal: float = Field(..., ge=0, description="Critical level (gallons)")

    # Status
    level_status: str = Field(default="normal", description="Level status")
    alert_active: bool = Field(default=False, description="Alert active")
    alert_type: Optional[InventoryAlertType] = Field(
        default=None,
        description="Active alert type"
    )

    # Quality
    last_delivery_date: Optional[datetime] = Field(
        default=None,
        description="Last delivery date"
    )
    fuel_age_days: Optional[float] = Field(
        default=None,
        ge=0,
        description="Fuel age in days"
    )
    quality_status: str = Field(default="good", description="Fuel quality status")

    class Config:
        use_enum_values = True


class CostAnalysis(BaseModel):
    """Total cost of ownership analysis."""

    analysis_id: str = Field(
        default_factory=lambda: str(uuid.uuid4())[:8],
        description="Analysis identifier"
    )
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Analysis timestamp"
    )
    period_hours: float = Field(..., gt=0, description="Analysis period (hours)")

    # Cost breakdown
    fuel_cost_usd: float = Field(..., ge=0, description="Fuel purchase cost")
    transport_cost_usd: float = Field(default=0.0, ge=0, description="Transport cost")
    storage_cost_usd: float = Field(default=0.0, ge=0, description="Storage cost")
    carbon_cost_usd: float = Field(default=0.0, ge=0, description="Carbon cost")
    maintenance_cost_usd: float = Field(default=0.0, ge=0, description="Maintenance impact")
    total_cost_usd: float = Field(..., ge=0, description="Total cost")

    # Metrics
    cost_per_mmbtu: float = Field(..., ge=0, description="Cost per MMBTU")
    cost_per_unit_output: Optional[float] = Field(
        default=None,
        description="Cost per unit output"
    )

    # Emissions
    total_co2_kg: float = Field(default=0.0, ge=0, description="Total CO2 (kg)")
    total_co2_cost_usd: float = Field(default=0.0, ge=0, description="CO2 cost")
    co2_intensity_kg_mmbtu: float = Field(
        default=0.0,
        ge=0,
        description="CO2 intensity (kg/MMBTU)"
    )

    # Comparison
    baseline_cost_usd: Optional[float] = Field(
        default=None,
        description="Baseline cost for comparison"
    )
    savings_vs_baseline_usd: Optional[float] = Field(
        default=None,
        description="Savings vs baseline"
    )
    savings_pct: Optional[float] = Field(
        default=None,
        description="Savings percentage"
    )


class OptimizationResult(BaseModel):
    """Overall optimization result."""

    result_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Result identifier"
    )
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Result timestamp"
    )

    # Status
    optimization_status: str = Field(default="success", description="Optimization status")
    optimization_mode: str = Field(
        default="minimum_cost",
        description="Optimization mode used"
    )

    # Recommendations
    blend_recommendation: Optional[BlendRecommendation] = Field(
        default=None,
        description="Blend recommendation"
    )
    switching_recommendation: Optional[SwitchingRecommendation] = Field(
        default=None,
        description="Switching recommendation"
    )
    cost_analysis: Optional[CostAnalysis] = Field(
        default=None,
        description="Cost analysis"
    )

    # Summary metrics
    recommended_fuel_cost_usd_hr: float = Field(
        ...,
        ge=0,
        description="Recommended fuel cost ($/hr)"
    )
    current_fuel_cost_usd_hr: float = Field(
        ...,
        ge=0,
        description="Current fuel cost ($/hr)"
    )
    potential_savings_usd_hr: float = Field(
        default=0.0,
        description="Potential savings ($/hr)"
    )
    potential_savings_usd_year: float = Field(
        default=0.0,
        description="Potential annual savings"
    )

    # Emissions
    current_co2_kg_hr: float = Field(
        default=0.0,
        ge=0,
        description="Current CO2 emissions"
    )
    recommended_co2_kg_hr: float = Field(
        default=0.0,
        ge=0,
        description="Recommended CO2 emissions"
    )
    co2_reduction_kg_hr: float = Field(
        default=0.0,
        description="CO2 reduction"
    )

    # Confidence
    confidence_score: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Result confidence"
    )
    warnings: List[str] = Field(
        default_factory=list,
        description="Optimization warnings"
    )


class FuelOptimizationInput(BaseModel):
    """Input data for fuel optimization."""

    # Identity
    facility_id: str = Field(..., description="Facility identifier")
    request_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Request identifier"
    )
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Request timestamp"
    )

    # Current state
    current_fuel: str = Field(..., description="Current primary fuel")
    current_fuel_flow_rate: float = Field(
        ...,
        gt=0,
        description="Current fuel flow rate (lb/hr or SCF/hr)"
    )
    current_heat_input_mmbtu_hr: float = Field(
        ...,
        gt=0,
        description="Current heat input (MMBTU/hr)"
    )
    current_load_pct: float = Field(
        default=100.0,
        ge=0,
        le=120,
        description="Current load percentage"
    )

    # Blend state (if applicable)
    current_blend: Optional[Dict[str, float]] = Field(
        default=None,
        description="Current blend ratios if blending"
    )

    # Fuel prices (if provided, otherwise fetched)
    fuel_prices: Optional[Dict[str, FuelPrice]] = Field(
        default=None,
        description="Current fuel prices by type"
    )

    # Fuel properties (if provided, otherwise from database)
    fuel_properties: Optional[Dict[str, FuelProperties]] = Field(
        default=None,
        description="Fuel properties by type"
    )

    # Inventory status
    inventory_status: Optional[List[InventoryStatus]] = Field(
        default=None,
        description="Current inventory status"
    )

    # Demand forecast
    forecast_horizon_hours: int = Field(
        default=24,
        ge=1,
        le=168,
        description="Forecast horizon"
    )
    expected_load_profile: Optional[List[Tuple[int, float]]] = Field(
        default=None,
        description="Expected load profile (hour, load_pct)"
    )

    # Constraints
    max_emissions_kg_hr: Optional[float] = Field(
        default=None,
        description="Maximum emissions constraint"
    )
    min_efficiency_pct: Optional[float] = Field(
        default=None,
        ge=50,
        le=100,
        description="Minimum efficiency constraint"
    )
    excluded_fuels: Optional[List[str]] = Field(
        default=None,
        description="Fuels to exclude from optimization"
    )

    # Equipment status
    equipment_availability: Optional[Dict[str, bool]] = Field(
        default=None,
        description="Equipment availability by ID"
    )
    equipment_constraints: Optional[Dict[str, Dict]] = Field(
        default=None,
        description="Equipment-specific constraints"
    )


class FuelOptimizationOutput(BaseModel):
    """Complete output from fuel optimization."""

    # Identity
    facility_id: str = Field(..., description="Facility identifier")
    request_id: str = Field(..., description="Request identifier")
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Output timestamp"
    )

    # Status
    status: str = Field(default="success", description="Processing status")
    processing_time_ms: float = Field(
        default=0.0,
        ge=0,
        description="Processing time (ms)"
    )

    # Optimization result
    optimization_result: OptimizationResult = Field(
        ...,
        description="Optimization result"
    )

    # Current prices used
    fuel_prices_used: Dict[str, FuelPrice] = Field(
        default_factory=dict,
        description="Fuel prices used in optimization"
    )

    # Inventory alerts
    inventory_alerts: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Active inventory alerts"
    )

    # Delivery recommendations
    delivery_recommendations: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Delivery scheduling recommendations"
    )

    # KPIs
    kpis: Dict[str, float] = Field(
        default_factory=dict,
        description="Key performance indicators"
    )

    # Provenance
    provenance_hash: Optional[str] = Field(
        default=None,
        description="SHA-256 provenance hash"
    )
    input_hash: Optional[str] = Field(
        default=None,
        description="Input data hash"
    )

    # Metadata
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata"
    )
