"""
GL-001 ThermalCommand API Schemas

Pydantic models for request/response validation and serialization.
Defines core data structures for district heating optimization API.
"""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import Any, Dict, List, Optional, Union
from uuid import UUID, uuid4

from pydantic import BaseModel, Field, validator, root_validator


# =============================================================================
# Enumerations
# =============================================================================

class AssetType(str, Enum):
    """Types of thermal assets in the district heating network."""
    CHP = "chp"  # Combined Heat and Power
    BOILER = "boiler"
    HEAT_PUMP = "heat_pump"
    HEAT_STORAGE = "heat_storage"
    SOLAR_THERMAL = "solar_thermal"
    WASTE_HEAT = "waste_heat"
    ELECTRIC_HEATER = "electric_heater"


class AssetStatus(str, Enum):
    """Operational status of an asset."""
    ONLINE = "online"
    OFFLINE = "offline"
    STANDBY = "standby"
    MAINTENANCE = "maintenance"
    FAULT = "fault"
    RAMPING_UP = "ramping_up"
    RAMPING_DOWN = "ramping_down"


class ConstraintType(str, Enum):
    """Types of operational constraints."""
    CAPACITY_MIN = "capacity_min"
    CAPACITY_MAX = "capacity_max"
    RAMP_RATE = "ramp_rate"
    TEMPERATURE_MIN = "temperature_min"
    TEMPERATURE_MAX = "temperature_max"
    PRESSURE_MIN = "pressure_min"
    PRESSURE_MAX = "pressure_max"
    EMISSIONS_LIMIT = "emissions_limit"
    MUST_RUN = "must_run"
    MUST_OFF = "must_off"
    ENERGY_BALANCE = "energy_balance"
    STORAGE_LEVEL = "storage_level"


class ConstraintPriority(str, Enum):
    """Priority levels for constraints."""
    CRITICAL = "critical"  # Safety-critical, must never violate
    HIGH = "high"  # Regulatory or contractual
    MEDIUM = "medium"  # Operational efficiency
    LOW = "low"  # Preference-based


class AlarmSeverity(str, Enum):
    """Severity levels for alarms and safety events."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


class AlarmStatus(str, Enum):
    """Status of an alarm."""
    ACTIVE = "active"
    ACKNOWLEDGED = "acknowledged"
    RESOLVED = "resolved"
    SUPPRESSED = "suppressed"


class MaintenanceType(str, Enum):
    """Types of maintenance actions."""
    PREVENTIVE = "preventive"
    PREDICTIVE = "predictive"
    CORRECTIVE = "corrective"
    CONDITION_BASED = "condition_based"


class MaintenanceUrgency(str, Enum):
    """Urgency levels for maintenance triggers."""
    IMMEDIATE = "immediate"
    URGENT = "urgent"  # Within 24 hours
    SCHEDULED = "scheduled"  # Within normal schedule
    DEFERRABLE = "deferrable"  # Can be postponed


class ForecastType(str, Enum):
    """Types of forecasts."""
    DEMAND = "demand"
    TEMPERATURE = "temperature"
    ELECTRICITY_PRICE = "electricity_price"
    GAS_PRICE = "gas_price"
    SOLAR_IRRADIANCE = "solar_irradiance"


class OptimizationObjective(str, Enum):
    """Optimization objectives for dispatch planning."""
    MINIMIZE_COST = "minimize_cost"
    MINIMIZE_EMISSIONS = "minimize_emissions"
    MAXIMIZE_EFFICIENCY = "maximize_efficiency"
    BALANCE_COST_EMISSIONS = "balance_cost_emissions"


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


class GeoLocation(BaseModel):
    """Geographic location for assets."""
    latitude: float = Field(..., ge=-90, le=90)
    longitude: float = Field(..., ge=-180, le=180)
    altitude_m: Optional[float] = Field(None, description="Altitude in meters")
    address: Optional[str] = None


# =============================================================================
# Asset Models
# =============================================================================

class AssetCapacity(BaseModel):
    """Capacity specifications for an asset."""
    thermal_capacity_mw: float = Field(..., ge=0, description="Thermal capacity in MW")
    min_output_mw: float = Field(0, ge=0, description="Minimum output in MW")
    max_output_mw: float = Field(..., ge=0, description="Maximum output in MW")
    ramp_up_rate_mw_min: float = Field(..., ge=0, description="Ramp-up rate in MW/min")
    ramp_down_rate_mw_min: float = Field(..., ge=0, description="Ramp-down rate in MW/min")
    min_uptime_hours: float = Field(0, ge=0, description="Minimum uptime in hours")
    min_downtime_hours: float = Field(0, ge=0, description="Minimum downtime in hours")
    startup_time_minutes: float = Field(0, ge=0, description="Startup time in minutes")

    @validator("max_output_mw")
    def max_must_exceed_min(cls, v, values):
        if "min_output_mw" in values and v < values["min_output_mw"]:
            raise ValueError("max_output_mw must be >= min_output_mw")
        return v


class AssetEfficiency(BaseModel):
    """Efficiency parameters for an asset."""
    thermal_efficiency: float = Field(..., ge=0, le=1, description="Thermal efficiency (0-1)")
    electrical_efficiency: Optional[float] = Field(None, ge=0, le=1, description="Electrical efficiency for CHP")
    part_load_efficiency_curve: Optional[List[Dict[str, float]]] = Field(
        None, description="Efficiency at different load points [{load: 0.5, efficiency: 0.85}, ...]"
    )


class AssetEmissions(BaseModel):
    """Emissions characteristics for an asset."""
    co2_kg_per_mwh: float = Field(..., ge=0, description="CO2 emissions in kg/MWh")
    nox_kg_per_mwh: float = Field(0, ge=0, description="NOx emissions in kg/MWh")
    so2_kg_per_mwh: float = Field(0, ge=0, description="SO2 emissions in kg/MWh")
    particulate_kg_per_mwh: float = Field(0, ge=0, description="Particulate emissions in kg/MWh")


class AssetCost(BaseModel):
    """Cost structure for an asset."""
    fuel_cost_per_mwh: float = Field(..., ge=0, description="Fuel cost per MWh thermal")
    variable_om_per_mwh: float = Field(0, ge=0, description="Variable O&M cost per MWh")
    fixed_om_per_day: float = Field(0, ge=0, description="Fixed O&M cost per day")
    startup_cost: float = Field(0, ge=0, description="Cost per startup")
    shutdown_cost: float = Field(0, ge=0, description="Cost per shutdown")


class AssetHealth(BaseModel):
    """Health indicators for an asset."""
    health_score: float = Field(..., ge=0, le=100, description="Overall health score 0-100")
    remaining_useful_life_hours: Optional[float] = Field(None, ge=0)
    last_maintenance_date: Optional[datetime] = None
    next_scheduled_maintenance: Optional[datetime] = None
    operating_hours_since_maintenance: float = Field(0, ge=0)
    fault_indicators: List[str] = Field(default_factory=list)


class AssetState(TimestampedModel):
    """Complete state representation of a thermal asset."""
    asset_id: UUID = Field(default_factory=uuid4)
    asset_name: str = Field(..., min_length=1, max_length=255)
    asset_type: AssetType
    status: AssetStatus
    location: Optional[GeoLocation] = None

    # Current operating state
    current_output_mw: float = Field(0, ge=0, description="Current thermal output in MW")
    current_setpoint_mw: float = Field(0, ge=0, description="Current setpoint in MW")
    supply_temperature_c: float = Field(..., description="Supply temperature in Celsius")
    return_temperature_c: float = Field(..., description="Return temperature in Celsius")
    flow_rate_m3h: float = Field(0, ge=0, description="Flow rate in m3/h")

    # Specifications
    capacity: AssetCapacity
    efficiency: AssetEfficiency
    emissions: AssetEmissions
    cost: AssetCost
    health: AssetHealth

    # Storage-specific (for heat storage assets)
    storage_level_mwh: Optional[float] = Field(None, ge=0, description="Current storage level in MWh")
    storage_capacity_mwh: Optional[float] = Field(None, ge=0, description="Total storage capacity in MWh")

    # Metadata
    tags: Dict[str, str] = Field(default_factory=dict)

    class Config:
        schema_extra = {
            "example": {
                "asset_id": "550e8400-e29b-41d4-a716-446655440000",
                "asset_name": "CHP Unit 1",
                "asset_type": "chp",
                "status": "online",
                "current_output_mw": 45.5,
                "current_setpoint_mw": 50.0,
                "supply_temperature_c": 95.0,
                "return_temperature_c": 55.0,
                "flow_rate_m3h": 250.0,
            }
        }


# =============================================================================
# Constraint Models
# =============================================================================

class Constraint(TimestampedModel):
    """Operational constraint definition."""
    constraint_id: UUID = Field(default_factory=uuid4)
    name: str = Field(..., min_length=1, max_length=255)
    description: Optional[str] = None
    constraint_type: ConstraintType
    priority: ConstraintPriority

    # Constraint parameters
    asset_id: Optional[UUID] = Field(None, description="Specific asset (None for system-wide)")
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    target_value: Optional[float] = None
    tolerance: float = Field(0, ge=0, description="Allowed tolerance")

    # Temporal scope
    effective_from: datetime
    effective_until: Optional[datetime] = None
    time_of_day_start: Optional[str] = Field(None, description="HH:MM format")
    time_of_day_end: Optional[str] = Field(None, description="HH:MM format")

    # Status
    is_active: bool = True
    is_violated: bool = False
    violation_count: int = Field(0, ge=0)
    last_violation_at: Optional[datetime] = None

    class Config:
        schema_extra = {
            "example": {
                "constraint_id": "550e8400-e29b-41d4-a716-446655440001",
                "name": "System Max Temperature",
                "constraint_type": "temperature_max",
                "priority": "critical",
                "max_value": 120.0,
                "effective_from": "2025-01-01T00:00:00Z",
            }
        }


# =============================================================================
# KPI Models
# =============================================================================

class KPI(TimestampedModel):
    """Key Performance Indicator measurement."""
    kpi_id: UUID = Field(default_factory=uuid4)
    name: str = Field(..., min_length=1, max_length=255)
    description: Optional[str] = None
    category: str = Field(..., description="KPI category: cost, emissions, efficiency, reliability")

    # Values
    current_value: float
    target_value: Optional[float] = None
    unit: str = Field(..., description="Unit of measurement")

    # Time context
    measurement_timestamp: datetime
    aggregation_period: str = Field("hourly", description="hourly, daily, weekly, monthly")

    # Trend analysis
    previous_value: Optional[float] = None
    trend_direction: Optional[str] = Field(None, description="up, down, stable")
    percent_change: Optional[float] = None

    # Performance against target
    target_achievement_percent: Optional[float] = Field(None, ge=0)
    is_on_target: Optional[bool] = None

    class Config:
        schema_extra = {
            "example": {
                "kpi_id": "550e8400-e29b-41d4-a716-446655440002",
                "name": "System Efficiency",
                "category": "efficiency",
                "current_value": 92.5,
                "target_value": 95.0,
                "unit": "%",
            }
        }


# =============================================================================
# Dispatch Plan Models
# =============================================================================

class SetpointRecommendation(BaseModel):
    """Setpoint recommendation for an asset."""
    asset_id: UUID
    asset_name: str
    current_setpoint_mw: float
    recommended_setpoint_mw: float
    confidence: float = Field(..., ge=0, le=1, description="Confidence level 0-1")
    reason: str = Field(..., description="Explanation for recommendation")
    expected_cost_impact: Optional[float] = None
    expected_emissions_impact_kg: Optional[float] = None


class DispatchScheduleEntry(BaseModel):
    """Single entry in a dispatch schedule."""
    timestamp: datetime
    asset_id: UUID
    asset_name: str
    setpoint_mw: float = Field(..., ge=0)
    status: AssetStatus
    cost_per_mwh: float = Field(..., ge=0)
    emissions_kg_per_mwh: float = Field(..., ge=0)


class DispatchPlan(TimestampedModel):
    """Complete dispatch plan for the thermal network."""
    plan_id: UUID = Field(default_factory=uuid4)
    plan_version: int = Field(1, ge=1)
    plan_name: str = Field(..., min_length=1, max_length=255)
    description: Optional[str] = None

    # Optimization context
    objective: OptimizationObjective
    planning_horizon_hours: int = Field(24, ge=1, le=168)
    resolution_minutes: int = Field(15, ge=1, le=60)

    # Plan validity
    effective_from: datetime
    effective_until: datetime
    is_active: bool = True

    # Optimization results
    schedule: List[DispatchScheduleEntry] = Field(default_factory=list)
    setpoint_recommendations: List[SetpointRecommendation] = Field(default_factory=list)

    # Metrics
    total_thermal_output_mwh: float = Field(0, ge=0)
    total_cost: float = Field(0, ge=0)
    total_emissions_kg: float = Field(0, ge=0)
    average_efficiency: float = Field(0, ge=0, le=1)

    # Constraint status
    constraints_satisfied: int = Field(0, ge=0)
    constraints_violated: int = Field(0, ge=0)
    violated_constraint_ids: List[UUID] = Field(default_factory=list)

    # Model confidence
    optimization_score: float = Field(0, ge=0, le=100)
    solver_status: str = Field("optimal", description="optimal, feasible, infeasible, timeout")
    computation_time_seconds: float = Field(0, ge=0)

    class Config:
        schema_extra = {
            "example": {
                "plan_id": "550e8400-e29b-41d4-a716-446655440003",
                "plan_name": "Day-Ahead Dispatch",
                "objective": "balance_cost_emissions",
                "planning_horizon_hours": 24,
                "total_cost": 15000.0,
                "total_emissions_kg": 5000.0,
            }
        }


# =============================================================================
# Request/Response Models
# =============================================================================

class DemandUpdate(BaseModel):
    """Request to submit a demand update."""
    request_id: UUID = Field(default_factory=uuid4)
    timestamp: datetime = Field(default_factory=datetime.utcnow)

    # Demand forecast
    forecast_type: ForecastType = ForecastType.DEMAND
    forecast_horizon_hours: int = Field(24, ge=1, le=168)
    resolution_minutes: int = Field(15, ge=1, le=60)

    # Demand data
    demand_mw: List[float] = Field(..., min_items=1, description="Forecasted demand values")
    demand_timestamps: List[datetime] = Field(..., min_items=1)
    confidence_intervals: Optional[List[Dict[str, float]]] = Field(
        None, description="[{lower: float, upper: float}, ...]"
    )

    # Source metadata
    source_system: str = Field(..., description="System that generated the forecast")
    model_version: Optional[str] = None

    @root_validator(skip_on_failure=True)
    def validate_lengths(cls, values):
        demand = values.get("demand_mw", [])
        timestamps = values.get("demand_timestamps", [])
        if len(demand) != len(timestamps):
            raise ValueError("demand_mw and demand_timestamps must have same length")
        return values


class AllocationRequest(BaseModel):
    """Request for heat allocation optimization."""
    request_id: UUID = Field(default_factory=uuid4)
    timestamp: datetime = Field(default_factory=datetime.utcnow)

    # Allocation parameters
    target_output_mw: float = Field(..., ge=0, description="Target thermal output")
    time_window_minutes: int = Field(15, ge=1, le=60)

    # Optimization preferences
    objective: OptimizationObjective = OptimizationObjective.BALANCE_COST_EMISSIONS
    cost_weight: float = Field(0.5, ge=0, le=1, description="Weight for cost minimization")
    emissions_weight: float = Field(0.5, ge=0, le=1, description="Weight for emissions minimization")

    # Constraints
    asset_preferences: Optional[Dict[str, float]] = Field(
        None, description="Asset ID to preference weight mapping"
    )
    excluded_assets: List[UUID] = Field(default_factory=list)
    must_run_assets: List[UUID] = Field(default_factory=list)

    # Urgency
    is_emergency: bool = False
    response_timeout_seconds: int = Field(30, ge=1, le=300)

    @validator("emissions_weight")
    def weights_sum_to_one(cls, v, values):
        cost_weight = values.get("cost_weight", 0.5)
        if abs(cost_weight + v - 1.0) > 0.001:
            raise ValueError("cost_weight and emissions_weight must sum to 1.0")
        return v


class AllocationResponse(TimestampedModel):
    """Response to an allocation request."""
    request_id: UUID
    response_id: UUID = Field(default_factory=uuid4)

    # Status
    success: bool
    status_message: str

    # Allocation result
    allocated_output_mw: float = Field(0, ge=0)
    allocation_gap_mw: float = Field(0, description="Gap between target and allocated")

    # Per-asset allocations
    asset_allocations: List[SetpointRecommendation] = Field(default_factory=list)

    # Cost and emissions
    estimated_cost: float = Field(0, ge=0)
    estimated_emissions_kg: float = Field(0, ge=0)

    # Performance metrics
    optimization_time_ms: float = Field(0, ge=0)
    solver_iterations: int = Field(0, ge=0)


class ForecastData(TimestampedModel):
    """Forecast data response."""
    forecast_id: UUID = Field(default_factory=uuid4)
    forecast_type: ForecastType

    # Temporal scope
    forecast_horizon_hours: int
    resolution_minutes: int
    generated_at: datetime
    valid_from: datetime
    valid_until: datetime

    # Forecast values
    values: List[float]
    timestamps: List[datetime]
    unit: str

    # Uncertainty
    confidence_level: float = Field(0.95, ge=0, le=1)
    lower_bounds: Optional[List[float]] = None
    upper_bounds: Optional[List[float]] = None

    # Model metadata
    model_name: str
    model_version: str
    accuracy_metrics: Optional[Dict[str, float]] = None


# =============================================================================
# Alarm and Safety Models
# =============================================================================

class AlarmEvent(TimestampedModel):
    """Alarm or safety event."""
    alarm_id: UUID = Field(default_factory=uuid4)
    alarm_code: str = Field(..., min_length=1, max_length=50)
    name: str = Field(..., min_length=1, max_length=255)
    description: str

    # Severity and status
    severity: AlarmSeverity
    status: AlarmStatus

    # Source
    asset_id: Optional[UUID] = None
    asset_name: Optional[str] = None
    subsystem: str = Field(..., description="Affected subsystem")

    # Timing
    triggered_at: datetime
    acknowledged_at: Optional[datetime] = None
    acknowledged_by: Optional[str] = None
    resolved_at: Optional[datetime] = None
    resolved_by: Optional[str] = None

    # Context
    measured_value: Optional[float] = None
    threshold_value: Optional[float] = None
    unit: Optional[str] = None

    # Actions
    recommended_actions: List[str] = Field(default_factory=list)
    auto_response_triggered: bool = False
    auto_response_description: Optional[str] = None

    class Config:
        schema_extra = {
            "example": {
                "alarm_id": "550e8400-e29b-41d4-a716-446655440004",
                "alarm_code": "TEMP_HIGH_001",
                "name": "High Supply Temperature",
                "severity": "high",
                "status": "active",
                "measured_value": 125.0,
                "threshold_value": 120.0,
            }
        }


class AlarmAcknowledgement(BaseModel):
    """Request to acknowledge an alarm."""
    alarm_id: UUID
    acknowledged_by: str = Field(..., min_length=1, max_length=255)
    acknowledgement_note: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)


# =============================================================================
# Maintenance Models
# =============================================================================

class MaintenanceTrigger(TimestampedModel):
    """Maintenance trigger notification."""
    trigger_id: UUID = Field(default_factory=uuid4)
    asset_id: UUID
    asset_name: str

    # Maintenance details
    maintenance_type: MaintenanceType
    urgency: MaintenanceUrgency

    # Trigger information
    trigger_reason: str
    trigger_metric: str
    current_value: float
    threshold_value: float

    # Recommendations
    recommended_action: str
    estimated_duration_hours: float = Field(..., ge=0)
    estimated_cost: Optional[float] = Field(None, ge=0)

    # Scheduling
    recommended_start_date: datetime
    latest_start_date: Optional[datetime] = None

    # Impact assessment
    production_impact_mw: float = Field(0, ge=0)
    downtime_hours: float = Field(0, ge=0)

    class Config:
        schema_extra = {
            "example": {
                "trigger_id": "550e8400-e29b-41d4-a716-446655440005",
                "asset_name": "CHP Unit 1",
                "maintenance_type": "predictive",
                "urgency": "scheduled",
                "trigger_reason": "Operating hours exceeded threshold",
            }
        }


# =============================================================================
# Explainability Models
# =============================================================================

class FeatureImportance(BaseModel):
    """Feature importance from SHAP/LIME analysis."""
    feature_name: str
    importance_score: float
    direction: str = Field(..., description="positive, negative, neutral")
    description: Optional[str] = None


class DecisionExplanation(BaseModel):
    """Explanation for a specific optimization decision."""
    decision_id: UUID = Field(default_factory=uuid4)
    decision_type: str
    decision_description: str

    # Contributing factors
    primary_factors: List[FeatureImportance]
    secondary_factors: List[FeatureImportance] = Field(default_factory=list)

    # Counterfactuals
    alternative_decisions: Optional[List[str]] = None
    alternative_outcomes: Optional[List[str]] = None

    # Confidence
    explanation_confidence: float = Field(..., ge=0, le=1)


class ExplainabilitySummary(TimestampedModel):
    """Complete explainability summary for optimization decisions."""
    summary_id: UUID = Field(default_factory=uuid4)
    plan_id: UUID

    # Overall explanation
    executive_summary: str
    key_drivers: List[str]

    # SHAP analysis
    shap_values: Optional[Dict[str, float]] = None
    global_feature_importance: List[FeatureImportance]

    # LIME analysis (for specific predictions)
    lime_explanations: List[DecisionExplanation] = Field(default_factory=list)

    # What-if analysis
    sensitivity_analysis: Optional[Dict[str, List[Dict[str, float]]]] = None

    # Natural language explanations
    plain_english_summary: str
    technical_details: Optional[str] = None

    class Config:
        schema_extra = {
            "example": {
                "summary_id": "550e8400-e29b-41d4-a716-446655440006",
                "executive_summary": "Dispatch optimized for cost-emissions balance",
                "key_drivers": ["Low gas prices", "High demand forecast", "CHP efficiency"],
            }
        }


# =============================================================================
# Pagination and Query Models
# =============================================================================

class PaginationParams(BaseModel):
    """Pagination parameters for list queries."""
    page: int = Field(1, ge=1)
    page_size: int = Field(20, ge=1, le=100)
    sort_by: Optional[str] = None
    sort_order: str = Field("desc", pattern="^(asc|desc)$")


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

    @validator("end_time")
    def end_after_start(cls, v, values):
        if "start_time" in values and v < values["start_time"]:
            raise ValueError("end_time must be after start_time")
        return v


# =============================================================================
# Health and Status Models
# =============================================================================

class ServiceHealth(BaseModel):
    """Health status of a service component."""
    service_name: str
    status: str = Field(..., description="healthy, degraded, unhealthy")
    latency_ms: float = Field(..., ge=0)
    last_check: datetime
    error_message: Optional[str] = None


class SystemStatus(BaseModel):
    """Overall system status."""
    status: str = Field(..., description="healthy, degraded, unhealthy")
    version: str
    uptime_seconds: float
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    services: List[ServiceHealth]
    active_connections: int
    requests_per_minute: float
