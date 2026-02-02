"""
GL-001 ThermalCommand GraphQL API

Strawberry GraphQL implementation for district heating optimization.
Provides queries, mutations, and subscriptions for real-time operations.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime
from typing import Any, AsyncGenerator, Dict, List, Optional, Union
from uuid import UUID, uuid4

import strawberry
from strawberry.fastapi import GraphQLRouter
from strawberry.permission import BasePermission
from strawberry.types import Info
from strawberry.subscriptions import GRAPHQL_TRANSPORT_WS_PROTOCOL, GRAPHQL_WS_PROTOCOL

from .api_schemas import (
    AlarmEvent as AlarmEventSchema,
    AlarmSeverity,
    AlarmStatus,
    AssetCapacity as AssetCapacitySchema,
    AssetCost as AssetCostSchema,
    AssetEfficiency as AssetEfficiencySchema,
    AssetEmissions as AssetEmissionsSchema,
    AssetHealth as AssetHealthSchema,
    AssetState as AssetStateSchema,
    AssetStatus,
    AssetType,
    Constraint as ConstraintSchema,
    ConstraintPriority,
    ConstraintType,
    DecisionExplanation,
    DispatchPlan as DispatchPlanSchema,
    DispatchScheduleEntry,
    ExplainabilitySummary as ExplainabilitySummarySchema,
    FeatureImportance,
    ForecastData as ForecastDataSchema,
    ForecastType,
    KPI as KPISchema,
    MaintenanceTrigger as MaintenanceTriggerSchema,
    MaintenanceType,
    MaintenanceUrgency,
    OptimizationObjective,
    SetpointRecommendation as SetpointRecommendationSchema,
)
from .api_auth import (
    Permission,
    ThermalCommandUser,
    get_current_user,
)

logger = logging.getLogger(__name__)


# =============================================================================
# GraphQL Permission Classes
# =============================================================================

class IsAuthenticated(BasePermission):
    """Check if user is authenticated."""
    message = "User is not authenticated"

    async def has_permission(self, source: Any, info: Info, **kwargs) -> bool:
        request = info.context["request"]
        try:
            user = await get_current_user(request, None, None, None)
            info.context["user"] = user
            return True
        except Exception:
            return False


class HasPermission(BasePermission):
    """Check if user has required permission."""
    message = "User does not have required permission"

    def __init__(self, permission: Permission):
        self.permission = permission

    async def has_permission(self, source: Any, info: Info, **kwargs) -> bool:
        user = info.context.get("user")
        if not user:
            return False
        return user.has_permission(self.permission)


# =============================================================================
# GraphQL Types (Strawberry)
# =============================================================================

@strawberry.type
class GeoLocation:
    """Geographic location."""
    latitude: float
    longitude: float
    altitude_m: Optional[float] = None
    address: Optional[str] = None


@strawberry.type
class AssetCapacity:
    """Capacity specifications for an asset."""
    thermal_capacity_mw: float
    min_output_mw: float
    max_output_mw: float
    ramp_up_rate_mw_min: float
    ramp_down_rate_mw_min: float
    min_uptime_hours: float
    min_downtime_hours: float
    startup_time_minutes: float


@strawberry.type
class AssetEfficiency:
    """Efficiency parameters for an asset."""
    thermal_efficiency: float
    electrical_efficiency: Optional[float] = None


@strawberry.type
class AssetEmissions:
    """Emissions characteristics for an asset."""
    co2_kg_per_mwh: float
    nox_kg_per_mwh: float
    so2_kg_per_mwh: float
    particulate_kg_per_mwh: float


@strawberry.type
class AssetCost:
    """Cost structure for an asset."""
    fuel_cost_per_mwh: float
    variable_om_per_mwh: float
    fixed_om_per_day: float
    startup_cost: float
    shutdown_cost: float


@strawberry.type
class AssetHealth:
    """Health indicators for an asset."""
    health_score: float
    remaining_useful_life_hours: Optional[float] = None
    last_maintenance_date: Optional[datetime] = None
    next_scheduled_maintenance: Optional[datetime] = None
    operating_hours_since_maintenance: float
    fault_indicators: List[str]


@strawberry.enum
class AssetTypeEnum(strawberry.Enum):
    CHP = "chp"
    BOILER = "boiler"
    HEAT_PUMP = "heat_pump"
    HEAT_STORAGE = "heat_storage"
    SOLAR_THERMAL = "solar_thermal"
    WASTE_HEAT = "waste_heat"
    ELECTRIC_HEATER = "electric_heater"


@strawberry.enum
class AssetStatusEnum(strawberry.Enum):
    ONLINE = "online"
    OFFLINE = "offline"
    STANDBY = "standby"
    MAINTENANCE = "maintenance"
    FAULT = "fault"
    RAMPING_UP = "ramping_up"
    RAMPING_DOWN = "ramping_down"


@strawberry.enum
class ConstraintTypeEnum(strawberry.Enum):
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


@strawberry.enum
class ConstraintPriorityEnum(strawberry.Enum):
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


@strawberry.enum
class AlarmSeverityEnum(strawberry.Enum):
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


@strawberry.enum
class AlarmStatusEnum(strawberry.Enum):
    ACTIVE = "active"
    ACKNOWLEDGED = "acknowledged"
    RESOLVED = "resolved"
    SUPPRESSED = "suppressed"


@strawberry.enum
class MaintenanceTypeEnum(strawberry.Enum):
    PREVENTIVE = "preventive"
    PREDICTIVE = "predictive"
    CORRECTIVE = "corrective"
    CONDITION_BASED = "condition_based"


@strawberry.enum
class MaintenanceUrgencyEnum(strawberry.Enum):
    IMMEDIATE = "immediate"
    URGENT = "urgent"
    SCHEDULED = "scheduled"
    DEFERRABLE = "deferrable"


@strawberry.enum
class OptimizationObjectiveEnum(strawberry.Enum):
    MINIMIZE_COST = "minimize_cost"
    MINIMIZE_EMISSIONS = "minimize_emissions"
    MAXIMIZE_EFFICIENCY = "maximize_efficiency"
    BALANCE_COST_EMISSIONS = "balance_cost_emissions"


@strawberry.enum
class ForecastTypeEnum(strawberry.Enum):
    DEMAND = "demand"
    TEMPERATURE = "temperature"
    ELECTRICITY_PRICE = "electricity_price"
    GAS_PRICE = "gas_price"
    SOLAR_IRRADIANCE = "solar_irradiance"


@strawberry.type
class AssetState:
    """Complete state representation of a thermal asset."""
    asset_id: strawberry.ID
    asset_name: str
    asset_type: AssetTypeEnum
    status: AssetStatusEnum
    location: Optional[GeoLocation] = None

    # Current operating state
    current_output_mw: float
    current_setpoint_mw: float
    supply_temperature_c: float
    return_temperature_c: float
    flow_rate_m3h: float

    # Specifications
    capacity: AssetCapacity
    efficiency: AssetEfficiency
    emissions: AssetEmissions
    cost: AssetCost
    health: AssetHealth

    # Storage-specific
    storage_level_mwh: Optional[float] = None
    storage_capacity_mwh: Optional[float] = None

    # Timestamps
    created_at: datetime
    updated_at: Optional[datetime] = None


@strawberry.type
class Constraint:
    """Operational constraint definition."""
    constraint_id: strawberry.ID
    name: str
    description: Optional[str] = None
    constraint_type: ConstraintTypeEnum
    priority: ConstraintPriorityEnum

    # Constraint parameters
    asset_id: Optional[strawberry.ID] = None
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    target_value: Optional[float] = None
    tolerance: float

    # Temporal scope
    effective_from: datetime
    effective_until: Optional[datetime] = None
    time_of_day_start: Optional[str] = None
    time_of_day_end: Optional[str] = None

    # Status
    is_active: bool
    is_violated: bool
    violation_count: int
    last_violation_at: Optional[datetime] = None

    created_at: datetime
    updated_at: Optional[datetime] = None


@strawberry.type
class KPI:
    """Key Performance Indicator measurement."""
    kpi_id: strawberry.ID
    name: str
    description: Optional[str] = None
    category: str

    # Values
    current_value: float
    target_value: Optional[float] = None
    unit: str

    # Time context
    measurement_timestamp: datetime
    aggregation_period: str

    # Trend analysis
    previous_value: Optional[float] = None
    trend_direction: Optional[str] = None
    percent_change: Optional[float] = None

    # Performance
    target_achievement_percent: Optional[float] = None
    is_on_target: Optional[bool] = None

    created_at: datetime


@strawberry.type
class SetpointRecommendation:
    """Setpoint recommendation for an asset."""
    asset_id: strawberry.ID
    asset_name: str
    current_setpoint_mw: float
    recommended_setpoint_mw: float
    confidence: float
    reason: str
    expected_cost_impact: Optional[float] = None
    expected_emissions_impact_kg: Optional[float] = None


@strawberry.type
class ScheduleEntry:
    """Single entry in a dispatch schedule."""
    timestamp: datetime
    asset_id: strawberry.ID
    asset_name: str
    setpoint_mw: float
    status: AssetStatusEnum
    cost_per_mwh: float
    emissions_kg_per_mwh: float


@strawberry.type
class DispatchPlan:
    """Complete dispatch plan for the thermal network."""
    plan_id: strawberry.ID
    plan_version: int
    plan_name: str
    description: Optional[str] = None

    # Optimization context
    objective: OptimizationObjectiveEnum
    planning_horizon_hours: int
    resolution_minutes: int

    # Plan validity
    effective_from: datetime
    effective_until: datetime
    is_active: bool

    # Optimization results
    schedule: List[ScheduleEntry]
    setpoint_recommendations: List[SetpointRecommendation]

    # Metrics
    total_thermal_output_mwh: float
    total_cost: float
    total_emissions_kg: float
    average_efficiency: float

    # Constraint status
    constraints_satisfied: int
    constraints_violated: int
    violated_constraint_ids: List[strawberry.ID]

    # Model confidence
    optimization_score: float
    solver_status: str
    computation_time_seconds: float

    created_at: datetime
    updated_at: Optional[datetime] = None


@strawberry.type
class AlarmEvent:
    """Alarm or safety event."""
    alarm_id: strawberry.ID
    alarm_code: str
    name: str
    description: str

    severity: AlarmSeverityEnum
    status: AlarmStatusEnum

    asset_id: Optional[strawberry.ID] = None
    asset_name: Optional[str] = None
    subsystem: str

    triggered_at: datetime
    acknowledged_at: Optional[datetime] = None
    acknowledged_by: Optional[str] = None
    resolved_at: Optional[datetime] = None
    resolved_by: Optional[str] = None

    measured_value: Optional[float] = None
    threshold_value: Optional[float] = None
    unit: Optional[str] = None

    recommended_actions: List[str]
    auto_response_triggered: bool
    auto_response_description: Optional[str] = None

    created_at: datetime


@strawberry.type
class MaintenanceTrigger:
    """Maintenance trigger notification."""
    trigger_id: strawberry.ID
    asset_id: strawberry.ID
    asset_name: str

    maintenance_type: MaintenanceTypeEnum
    urgency: MaintenanceUrgencyEnum

    trigger_reason: str
    trigger_metric: str
    current_value: float
    threshold_value: float

    recommended_action: str
    estimated_duration_hours: float
    estimated_cost: Optional[float] = None

    recommended_start_date: datetime
    latest_start_date: Optional[datetime] = None

    production_impact_mw: float
    downtime_hours: float

    created_at: datetime


@strawberry.type
class FeatureImportanceType:
    """Feature importance from SHAP/LIME analysis."""
    feature_name: str
    importance_score: float
    direction: str
    description: Optional[str] = None


@strawberry.type
class DecisionExplanationType:
    """Explanation for a specific optimization decision."""
    decision_id: strawberry.ID
    decision_type: str
    decision_description: str
    primary_factors: List[FeatureImportanceType]
    secondary_factors: List[FeatureImportanceType]
    alternative_decisions: Optional[List[str]] = None
    alternative_outcomes: Optional[List[str]] = None
    explanation_confidence: float


@strawberry.type
class ExplainabilitySummary:
    """Complete explainability summary for optimization decisions."""
    summary_id: strawberry.ID
    plan_id: strawberry.ID

    executive_summary: str
    key_drivers: List[str]
    global_feature_importance: List[FeatureImportanceType]
    lime_explanations: List[DecisionExplanationType]

    plain_english_summary: str
    technical_details: Optional[str] = None

    created_at: datetime


@strawberry.type
class ForecastData:
    """Forecast data response."""
    forecast_id: strawberry.ID
    forecast_type: ForecastTypeEnum

    forecast_horizon_hours: int
    resolution_minutes: int
    generated_at: datetime
    valid_from: datetime
    valid_until: datetime

    values: List[float]
    timestamps: List[datetime]
    unit: str

    confidence_level: float
    lower_bounds: Optional[List[float]] = None
    upper_bounds: Optional[List[float]] = None

    model_name: str
    model_version: str


# =============================================================================
# Input Types
# =============================================================================

@strawberry.input
class DemandUpdateInput:
    """Input for submitting demand updates."""
    forecast_type: ForecastTypeEnum = ForecastTypeEnum.DEMAND
    forecast_horizon_hours: int = 24
    resolution_minutes: int = 15
    demand_mw: List[float]
    demand_timestamps: List[datetime]
    source_system: str
    model_version: Optional[str] = None


@strawberry.input
class AllocationRequestInput:
    """Input for requesting heat allocation."""
    target_output_mw: float
    time_window_minutes: int = 15
    objective: OptimizationObjectiveEnum = OptimizationObjectiveEnum.BALANCE_COST_EMISSIONS
    cost_weight: float = 0.5
    emissions_weight: float = 0.5
    excluded_assets: Optional[List[strawberry.ID]] = None
    must_run_assets: Optional[List[strawberry.ID]] = None
    is_emergency: bool = False
    response_timeout_seconds: int = 30


@strawberry.input
class AlarmAcknowledgementInput:
    """Input for acknowledging an alarm."""
    alarm_id: strawberry.ID
    acknowledged_by: str
    acknowledgement_note: Optional[str] = None


@strawberry.input
class TimeRangeInput:
    """Time range filter for queries."""
    start_time: datetime
    end_time: datetime


@strawberry.input
class PaginationInput:
    """Pagination parameters."""
    page: int = 1
    page_size: int = 20
    sort_by: Optional[str] = None
    sort_order: str = "desc"


# =============================================================================
# Response Types
# =============================================================================

@strawberry.type
class DemandUpdateResponse:
    """Response for demand update submission."""
    request_id: strawberry.ID
    success: bool
    message: str
    records_received: int
    records_validated: int
    data_quality_score: float


@strawberry.type
class AllocationResponse:
    """Response for allocation request."""
    request_id: strawberry.ID
    response_id: strawberry.ID
    success: bool
    status_message: str
    allocated_output_mw: float
    allocation_gap_mw: float
    asset_allocations: List[SetpointRecommendation]
    estimated_cost: float
    estimated_emissions_kg: float
    optimization_time_ms: float
    solver_iterations: int


@strawberry.type
class AlarmAcknowledgementResponse:
    """Response for alarm acknowledgement."""
    alarm_id: strawberry.ID
    success: bool
    message: str
    acknowledged_at: datetime
    acknowledged_by: str


@strawberry.type
class PaginatedAssets:
    """Paginated list of assets."""
    items: List[AssetState]
    total_count: int
    page: int
    page_size: int
    total_pages: int
    has_next: bool
    has_previous: bool


@strawberry.type
class PaginatedConstraints:
    """Paginated list of constraints."""
    items: List[Constraint]
    total_count: int
    page: int
    page_size: int
    total_pages: int
    has_next: bool
    has_previous: bool


@strawberry.type
class PaginatedAlarms:
    """Paginated list of alarms."""
    items: List[AlarmEvent]
    total_count: int
    page: int
    page_size: int
    total_pages: int
    has_next: bool
    has_previous: bool


# =============================================================================
# Data Layer (Mock Implementation)
# =============================================================================

class DataStore:
    """
    Mock data store for ThermalCommand.
    In production, this would interface with the actual database.
    """

    def __init__(self):
        self._dispatch_plans: Dict[str, DispatchPlanSchema] = {}
        self._asset_states: Dict[str, AssetStateSchema] = {}
        self._constraints: Dict[str, ConstraintSchema] = {}
        self._kpis: Dict[str, KPISchema] = {}
        self._alarms: Dict[str, AlarmEventSchema] = {}
        self._maintenance_triggers: Dict[str, MaintenanceTriggerSchema] = {}
        self._explainability: Dict[str, ExplainabilitySummarySchema] = {}
        self._forecasts: Dict[str, ForecastDataSchema] = {}

        # Event queues for subscriptions
        self._plan_update_queue: asyncio.Queue = asyncio.Queue()
        self._action_recommendation_queue: asyncio.Queue = asyncio.Queue()
        self._alarm_event_queue: asyncio.Queue = asyncio.Queue()
        self._maintenance_trigger_queue: asyncio.Queue = asyncio.Queue()

        # Initialize with sample data
        self._initialize_sample_data()

    def _initialize_sample_data(self):
        """Initialize with sample data for demonstration."""
        now = datetime.utcnow()

        # Sample asset
        asset_id = str(uuid4())
        self._asset_states[asset_id] = AssetStateSchema(
            asset_id=UUID(asset_id),
            asset_name="CHP Unit 1",
            asset_type=AssetType.CHP,
            status=AssetStatus.ONLINE,
            current_output_mw=45.5,
            current_setpoint_mw=50.0,
            supply_temperature_c=95.0,
            return_temperature_c=55.0,
            flow_rate_m3h=250.0,
            capacity=AssetCapacitySchema(
                thermal_capacity_mw=100.0,
                min_output_mw=20.0,
                max_output_mw=100.0,
                ramp_up_rate_mw_min=2.0,
                ramp_down_rate_mw_min=3.0,
                min_uptime_hours=4.0,
                min_downtime_hours=2.0,
                startup_time_minutes=30.0,
            ),
            efficiency=AssetEfficiencySchema(
                thermal_efficiency=0.88,
                electrical_efficiency=0.42,
            ),
            emissions=AssetEmissionsSchema(
                co2_kg_per_mwh=180.0,
                nox_kg_per_mwh=0.5,
                so2_kg_per_mwh=0.1,
                particulate_kg_per_mwh=0.02,
            ),
            cost=AssetCostSchema(
                fuel_cost_per_mwh=35.0,
                variable_om_per_mwh=2.5,
                fixed_om_per_day=500.0,
                startup_cost=1500.0,
                shutdown_cost=500.0,
            ),
            health=AssetHealthSchema(
                health_score=92.5,
                remaining_useful_life_hours=5000.0,
                last_maintenance_date=datetime(2025, 1, 1),
                operating_hours_since_maintenance=720.0,
                fault_indicators=[],
            ),
            created_at=now,
        )

        # Sample constraint
        constraint_id = str(uuid4())
        self._constraints[constraint_id] = ConstraintSchema(
            constraint_id=UUID(constraint_id),
            name="Max Supply Temperature",
            description="Maximum allowed supply temperature",
            constraint_type=ConstraintType.TEMPERATURE_MAX,
            priority=ConstraintPriority.CRITICAL,
            max_value=120.0,
            tolerance=2.0,
            effective_from=datetime(2025, 1, 1),
            is_active=True,
            is_violated=False,
            violation_count=0,
            created_at=now,
        )

        # Sample KPI
        kpi_id = str(uuid4())
        self._kpis[kpi_id] = KPISchema(
            kpi_id=UUID(kpi_id),
            name="System Efficiency",
            description="Overall thermal system efficiency",
            category="efficiency",
            current_value=92.5,
            target_value=95.0,
            unit="%",
            measurement_timestamp=now,
            aggregation_period="hourly",
            previous_value=91.8,
            trend_direction="up",
            percent_change=0.76,
            target_achievement_percent=97.4,
            is_on_target=False,
            created_at=now,
        )

        # Sample dispatch plan
        plan_id = str(uuid4())
        self._dispatch_plans[plan_id] = DispatchPlanSchema(
            plan_id=UUID(plan_id),
            plan_version=1,
            plan_name="Day-Ahead Dispatch Plan",
            description="Optimized dispatch for next 24 hours",
            objective=OptimizationObjective.BALANCE_COST_EMISSIONS,
            planning_horizon_hours=24,
            resolution_minutes=15,
            effective_from=now,
            effective_until=datetime(now.year, now.month, now.day + 1),
            is_active=True,
            schedule=[],
            setpoint_recommendations=[
                SetpointRecommendationSchema(
                    asset_id=UUID(asset_id),
                    asset_name="CHP Unit 1",
                    current_setpoint_mw=50.0,
                    recommended_setpoint_mw=55.0,
                    confidence=0.92,
                    reason="Increased demand forecast, higher efficiency at 55 MW",
                    expected_cost_impact=-150.0,
                    expected_emissions_impact_kg=-50.0,
                ),
            ],
            total_thermal_output_mwh=1200.0,
            total_cost=15000.0,
            total_emissions_kg=5400.0,
            average_efficiency=0.89,
            constraints_satisfied=12,
            constraints_violated=0,
            violated_constraint_ids=[],
            optimization_score=95.5,
            solver_status="optimal",
            computation_time_seconds=2.3,
            created_at=now,
        )

    async def get_current_plan(self) -> Optional[DispatchPlanSchema]:
        """Get the currently active dispatch plan."""
        now = datetime.utcnow()
        for plan in self._dispatch_plans.values():
            if plan.is_active and plan.effective_from <= now <= plan.effective_until:
                return plan
        # Return any active plan if no time-specific one found
        for plan in self._dispatch_plans.values():
            if plan.is_active:
                return plan
        return None

    async def get_asset_states(
        self,
        asset_ids: Optional[List[str]] = None,
        asset_types: Optional[List[AssetType]] = None,
        status: Optional[List[AssetStatus]] = None,
    ) -> List[AssetStateSchema]:
        """Get asset states with optional filtering."""
        assets = list(self._asset_states.values())

        if asset_ids:
            assets = [a for a in assets if str(a.asset_id) in asset_ids]
        if asset_types:
            assets = [a for a in assets if a.asset_type in asset_types]
        if status:
            assets = [a for a in assets if a.status in status]

        return assets

    async def get_constraints(
        self,
        is_active: Optional[bool] = None,
        constraint_types: Optional[List[ConstraintType]] = None,
        priorities: Optional[List[ConstraintPriority]] = None,
    ) -> List[ConstraintSchema]:
        """Get constraints with optional filtering."""
        constraints = list(self._constraints.values())

        if is_active is not None:
            constraints = [c for c in constraints if c.is_active == is_active]
        if constraint_types:
            constraints = [c for c in constraints if c.constraint_type in constraint_types]
        if priorities:
            constraints = [c for c in constraints if c.priority in priorities]

        return constraints

    async def get_kpis(
        self,
        category: Optional[str] = None,
        time_range: Optional[TimeRangeInput] = None,
    ) -> List[KPISchema]:
        """Get KPIs with optional filtering."""
        kpis = list(self._kpis.values())

        if category:
            kpis = [k for k in kpis if k.category == category]

        if time_range:
            kpis = [
                k for k in kpis
                if time_range.start_time <= k.measurement_timestamp <= time_range.end_time
            ]

        return kpis

    async def get_explainability_summary(self, plan_id: str) -> Optional[ExplainabilitySummarySchema]:
        """Get explainability summary for a plan."""
        # Generate sample explainability if not exists
        if plan_id not in self._explainability:
            plan = self._dispatch_plans.get(plan_id)
            if plan:
                summary = ExplainabilitySummarySchema(
                    summary_id=uuid4(),
                    plan_id=UUID(plan_id),
                    executive_summary="Dispatch optimized for cost-emissions balance with high confidence",
                    key_drivers=[
                        "Low natural gas prices",
                        "High demand forecast",
                        "CHP unit efficiency advantage",
                        "Storage level optimization",
                    ],
                    global_feature_importance=[
                        FeatureImportance(
                            feature_name="gas_price",
                            importance_score=0.35,
                            direction="negative",
                            description="Lower gas prices favor CHP operation",
                        ),
                        FeatureImportance(
                            feature_name="demand_forecast",
                            importance_score=0.28,
                            direction="positive",
                            description="Higher demand requires more capacity",
                        ),
                        FeatureImportance(
                            feature_name="electricity_price",
                            importance_score=0.22,
                            direction="positive",
                            description="Higher electricity prices favor CHP",
                        ),
                    ],
                    lime_explanations=[
                        DecisionExplanation(
                            decision_id=uuid4(),
                            decision_type="setpoint",
                            decision_description="CHP Unit 1 setpoint increased to 55 MW",
                            primary_factors=[
                                FeatureImportance(
                                    feature_name="efficiency_curve",
                                    importance_score=0.45,
                                    direction="positive",
                                    description="Higher efficiency at 55 MW vs 50 MW",
                                ),
                            ],
                            explanation_confidence=0.92,
                        ),
                    ],
                    plain_english_summary="The optimizer increased CHP Unit 1 output because gas prices are low and the unit operates more efficiently at higher loads. This reduces both cost and emissions.",
                    technical_details="MILP solver converged in 2.3s with 0.1% optimality gap",
                    created_at=datetime.utcnow(),
                )
                self._explainability[plan_id] = summary

        return self._explainability.get(plan_id)

    async def get_alarms(
        self,
        severity: Optional[List[AlarmSeverity]] = None,
        status: Optional[List[AlarmStatus]] = None,
        asset_id: Optional[str] = None,
    ) -> List[AlarmEventSchema]:
        """Get alarms with optional filtering."""
        alarms = list(self._alarms.values())

        if severity:
            alarms = [a for a in alarms if a.severity in severity]
        if status:
            alarms = [a for a in alarms if a.status in status]
        if asset_id:
            alarms = [a for a in alarms if str(a.asset_id) == asset_id]

        return alarms

    async def get_maintenance_triggers(
        self,
        urgency: Optional[List[MaintenanceUrgency]] = None,
        asset_id: Optional[str] = None,
    ) -> List[MaintenanceTriggerSchema]:
        """Get maintenance triggers with optional filtering."""
        triggers = list(self._maintenance_triggers.values())

        if urgency:
            triggers = [t for t in triggers if t.urgency in urgency]
        if asset_id:
            triggers = [t for t in triggers if str(t.asset_id) == asset_id]

        return triggers

    async def submit_demand_update(
        self,
        demand_update: DemandUpdateInput,
    ) -> DemandUpdateResponse:
        """Submit a demand update."""
        request_id = str(uuid4())

        # Validate data
        records_received = len(demand_update.demand_mw)
        records_validated = records_received  # All valid in mock
        quality_score = 98.5  # Mock quality score

        logger.info(f"Demand update received: {records_received} records")

        return DemandUpdateResponse(
            request_id=strawberry.ID(request_id),
            success=True,
            message="Demand update received and validated",
            records_received=records_received,
            records_validated=records_validated,
            data_quality_score=quality_score,
        )

    async def request_allocation(
        self,
        allocation_request: AllocationRequestInput,
    ) -> AllocationResponse:
        """Request heat allocation optimization."""
        request_id = str(uuid4())
        response_id = str(uuid4())

        # Mock allocation response
        assets = await self.get_asset_states()
        allocations = []

        remaining_output = allocation_request.target_output_mw
        for asset in assets:
            if remaining_output <= 0:
                break

            if allocation_request.excluded_assets and str(asset.asset_id) in allocation_request.excluded_assets:
                continue

            allocation = min(remaining_output, asset.capacity.max_output_mw)
            allocations.append(SetpointRecommendation(
                asset_id=strawberry.ID(str(asset.asset_id)),
                asset_name=asset.asset_name,
                current_setpoint_mw=asset.current_setpoint_mw,
                recommended_setpoint_mw=allocation,
                confidence=0.95,
                reason="Optimal allocation based on cost-emissions objective",
                expected_cost_impact=allocation * 35,  # Mock cost
                expected_emissions_impact_kg=allocation * 180,  # Mock emissions
            ))
            remaining_output -= allocation

        total_allocated = allocation_request.target_output_mw - remaining_output

        # Broadcast to subscription
        await self._action_recommendation_queue.put(allocations)

        return AllocationResponse(
            request_id=strawberry.ID(request_id),
            response_id=strawberry.ID(response_id),
            success=True,
            status_message="Allocation optimized successfully",
            allocated_output_mw=total_allocated,
            allocation_gap_mw=remaining_output,
            asset_allocations=allocations,
            estimated_cost=total_allocated * 38,
            estimated_emissions_kg=total_allocated * 175,
            optimization_time_ms=125.5,
            solver_iterations=42,
        )

    async def acknowledge_alarm(
        self,
        acknowledgement: AlarmAcknowledgementInput,
    ) -> AlarmAcknowledgementResponse:
        """Acknowledge an alarm."""
        alarm_id = str(acknowledgement.alarm_id)
        now = datetime.utcnow()

        if alarm_id in self._alarms:
            alarm = self._alarms[alarm_id]
            alarm.status = AlarmStatus.ACKNOWLEDGED
            alarm.acknowledged_at = now
            alarm.acknowledged_by = acknowledgement.acknowledged_by

        return AlarmAcknowledgementResponse(
            alarm_id=acknowledgement.alarm_id,
            success=True,
            message="Alarm acknowledged successfully",
            acknowledged_at=now,
            acknowledged_by=acknowledgement.acknowledged_by,
        )

    async def get_latest_forecast(
        self,
        forecast_type: ForecastType,
    ) -> Optional[ForecastDataSchema]:
        """Get the latest forecast of a specific type."""
        for forecast in self._forecasts.values():
            if forecast.forecast_type == forecast_type:
                return forecast

        # Generate mock forecast if not exists
        now = datetime.utcnow()
        forecast = ForecastDataSchema(
            forecast_id=uuid4(),
            forecast_type=forecast_type,
            forecast_horizon_hours=24,
            resolution_minutes=15,
            generated_at=now,
            valid_from=now,
            valid_until=datetime(now.year, now.month, now.day + 1),
            values=[50.0 + i * 0.5 for i in range(96)],  # 96 15-min intervals
            timestamps=[datetime(now.year, now.month, now.day, i // 4, (i % 4) * 15) for i in range(96)],
            unit="MW" if forecast_type == ForecastType.DEMAND else "C",
            confidence_level=0.95,
            model_name="ThermalForecast-v2",
            model_version="2.1.0",
            created_at=now,
        )
        self._forecasts[str(forecast.forecast_id)] = forecast
        return forecast


# Global data store instance
data_store = DataStore()


# =============================================================================
# Helper Functions for Type Conversion
# =============================================================================

def convert_asset_to_graphql(asset: AssetStateSchema) -> AssetState:
    """Convert Pydantic AssetState to GraphQL type."""
    return AssetState(
        asset_id=strawberry.ID(str(asset.asset_id)),
        asset_name=asset.asset_name,
        asset_type=AssetTypeEnum(asset.asset_type.value),
        status=AssetStatusEnum(asset.status.value),
        location=GeoLocation(
            latitude=asset.location.latitude,
            longitude=asset.location.longitude,
            altitude_m=asset.location.altitude_m,
            address=asset.location.address,
        ) if asset.location else None,
        current_output_mw=asset.current_output_mw,
        current_setpoint_mw=asset.current_setpoint_mw,
        supply_temperature_c=asset.supply_temperature_c,
        return_temperature_c=asset.return_temperature_c,
        flow_rate_m3h=asset.flow_rate_m3h,
        capacity=AssetCapacity(
            thermal_capacity_mw=asset.capacity.thermal_capacity_mw,
            min_output_mw=asset.capacity.min_output_mw,
            max_output_mw=asset.capacity.max_output_mw,
            ramp_up_rate_mw_min=asset.capacity.ramp_up_rate_mw_min,
            ramp_down_rate_mw_min=asset.capacity.ramp_down_rate_mw_min,
            min_uptime_hours=asset.capacity.min_uptime_hours,
            min_downtime_hours=asset.capacity.min_downtime_hours,
            startup_time_minutes=asset.capacity.startup_time_minutes,
        ),
        efficiency=AssetEfficiency(
            thermal_efficiency=asset.efficiency.thermal_efficiency,
            electrical_efficiency=asset.efficiency.electrical_efficiency,
        ),
        emissions=AssetEmissions(
            co2_kg_per_mwh=asset.emissions.co2_kg_per_mwh,
            nox_kg_per_mwh=asset.emissions.nox_kg_per_mwh,
            so2_kg_per_mwh=asset.emissions.so2_kg_per_mwh,
            particulate_kg_per_mwh=asset.emissions.particulate_kg_per_mwh,
        ),
        cost=AssetCost(
            fuel_cost_per_mwh=asset.cost.fuel_cost_per_mwh,
            variable_om_per_mwh=asset.cost.variable_om_per_mwh,
            fixed_om_per_day=asset.cost.fixed_om_per_day,
            startup_cost=asset.cost.startup_cost,
            shutdown_cost=asset.cost.shutdown_cost,
        ),
        health=AssetHealth(
            health_score=asset.health.health_score,
            remaining_useful_life_hours=asset.health.remaining_useful_life_hours,
            last_maintenance_date=asset.health.last_maintenance_date,
            next_scheduled_maintenance=asset.health.next_scheduled_maintenance,
            operating_hours_since_maintenance=asset.health.operating_hours_since_maintenance,
            fault_indicators=asset.health.fault_indicators,
        ),
        storage_level_mwh=asset.storage_level_mwh,
        storage_capacity_mwh=asset.storage_capacity_mwh,
        created_at=asset.created_at,
        updated_at=asset.updated_at,
    )


def convert_constraint_to_graphql(constraint: ConstraintSchema) -> Constraint:
    """Convert Pydantic Constraint to GraphQL type."""
    return Constraint(
        constraint_id=strawberry.ID(str(constraint.constraint_id)),
        name=constraint.name,
        description=constraint.description,
        constraint_type=ConstraintTypeEnum(constraint.constraint_type.value),
        priority=ConstraintPriorityEnum(constraint.priority.value),
        asset_id=strawberry.ID(str(constraint.asset_id)) if constraint.asset_id else None,
        min_value=constraint.min_value,
        max_value=constraint.max_value,
        target_value=constraint.target_value,
        tolerance=constraint.tolerance,
        effective_from=constraint.effective_from,
        effective_until=constraint.effective_until,
        time_of_day_start=constraint.time_of_day_start,
        time_of_day_end=constraint.time_of_day_end,
        is_active=constraint.is_active,
        is_violated=constraint.is_violated,
        violation_count=constraint.violation_count,
        last_violation_at=constraint.last_violation_at,
        created_at=constraint.created_at,
        updated_at=constraint.updated_at,
    )


def convert_kpi_to_graphql(kpi: KPISchema) -> KPI:
    """Convert Pydantic KPI to GraphQL type."""
    return KPI(
        kpi_id=strawberry.ID(str(kpi.kpi_id)),
        name=kpi.name,
        description=kpi.description,
        category=kpi.category,
        current_value=kpi.current_value,
        target_value=kpi.target_value,
        unit=kpi.unit,
        measurement_timestamp=kpi.measurement_timestamp,
        aggregation_period=kpi.aggregation_period,
        previous_value=kpi.previous_value,
        trend_direction=kpi.trend_direction,
        percent_change=kpi.percent_change,
        target_achievement_percent=kpi.target_achievement_percent,
        is_on_target=kpi.is_on_target,
        created_at=kpi.created_at,
    )


def convert_plan_to_graphql(plan: DispatchPlanSchema) -> DispatchPlan:
    """Convert Pydantic DispatchPlan to GraphQL type."""
    return DispatchPlan(
        plan_id=strawberry.ID(str(plan.plan_id)),
        plan_version=plan.plan_version,
        plan_name=plan.plan_name,
        description=plan.description,
        objective=OptimizationObjectiveEnum(plan.objective.value),
        planning_horizon_hours=plan.planning_horizon_hours,
        resolution_minutes=plan.resolution_minutes,
        effective_from=plan.effective_from,
        effective_until=plan.effective_until,
        is_active=plan.is_active,
        schedule=[
            ScheduleEntry(
                timestamp=entry.timestamp,
                asset_id=strawberry.ID(str(entry.asset_id)),
                asset_name=entry.asset_name,
                setpoint_mw=entry.setpoint_mw,
                status=AssetStatusEnum(entry.status.value),
                cost_per_mwh=entry.cost_per_mwh,
                emissions_kg_per_mwh=entry.emissions_kg_per_mwh,
            )
            for entry in plan.schedule
        ],
        setpoint_recommendations=[
            SetpointRecommendation(
                asset_id=strawberry.ID(str(rec.asset_id)),
                asset_name=rec.asset_name,
                current_setpoint_mw=rec.current_setpoint_mw,
                recommended_setpoint_mw=rec.recommended_setpoint_mw,
                confidence=rec.confidence,
                reason=rec.reason,
                expected_cost_impact=rec.expected_cost_impact,
                expected_emissions_impact_kg=rec.expected_emissions_impact_kg,
            )
            for rec in plan.setpoint_recommendations
        ],
        total_thermal_output_mwh=plan.total_thermal_output_mwh,
        total_cost=plan.total_cost,
        total_emissions_kg=plan.total_emissions_kg,
        average_efficiency=plan.average_efficiency,
        constraints_satisfied=plan.constraints_satisfied,
        constraints_violated=plan.constraints_violated,
        violated_constraint_ids=[strawberry.ID(str(vid)) for vid in plan.violated_constraint_ids],
        optimization_score=plan.optimization_score,
        solver_status=plan.solver_status,
        computation_time_seconds=plan.computation_time_seconds,
        created_at=plan.created_at,
        updated_at=plan.updated_at,
    )


def convert_explainability_to_graphql(summary: ExplainabilitySummarySchema) -> ExplainabilitySummary:
    """Convert Pydantic ExplainabilitySummary to GraphQL type."""
    return ExplainabilitySummary(
        summary_id=strawberry.ID(str(summary.summary_id)),
        plan_id=strawberry.ID(str(summary.plan_id)),
        executive_summary=summary.executive_summary,
        key_drivers=summary.key_drivers,
        global_feature_importance=[
            FeatureImportanceType(
                feature_name=fi.feature_name,
                importance_score=fi.importance_score,
                direction=fi.direction,
                description=fi.description,
            )
            for fi in summary.global_feature_importance
        ],
        lime_explanations=[
            DecisionExplanationType(
                decision_id=strawberry.ID(str(de.decision_id)),
                decision_type=de.decision_type,
                decision_description=de.decision_description,
                primary_factors=[
                    FeatureImportanceType(
                        feature_name=fi.feature_name,
                        importance_score=fi.importance_score,
                        direction=fi.direction,
                        description=fi.description,
                    )
                    for fi in de.primary_factors
                ],
                secondary_factors=[
                    FeatureImportanceType(
                        feature_name=fi.feature_name,
                        importance_score=fi.importance_score,
                        direction=fi.direction,
                        description=fi.description,
                    )
                    for fi in de.secondary_factors
                ],
                alternative_decisions=de.alternative_decisions,
                alternative_outcomes=de.alternative_outcomes,
                explanation_confidence=de.explanation_confidence,
            )
            for de in summary.lime_explanations
        ],
        plain_english_summary=summary.plain_english_summary,
        technical_details=summary.technical_details,
        created_at=summary.created_at,
    )


# =============================================================================
# GraphQL Query Type
# =============================================================================

@strawberry.type
class Query:
    """GraphQL Query type for ThermalCommand."""

    @strawberry.field(description="Get the current active dispatch plan")
    async def current_plan(self, info: Info) -> Optional[DispatchPlan]:
        """Get current dispatch plan."""
        plan = await data_store.get_current_plan()
        if plan:
            return convert_plan_to_graphql(plan)
        return None

    @strawberry.field(description="Get asset states with optional filtering")
    async def asset_states(
        self,
        info: Info,
        asset_ids: Optional[List[strawberry.ID]] = None,
        asset_types: Optional[List[AssetTypeEnum]] = None,
        status: Optional[List[AssetStatusEnum]] = None,
        pagination: Optional[PaginationInput] = None,
    ) -> PaginatedAssets:
        """Get asset states."""
        # Convert enum values
        types = [AssetType(t.value) for t in asset_types] if asset_types else None
        statuses = [AssetStatus(s.value) for s in status] if status else None
        ids = [str(aid) for aid in asset_ids] if asset_ids else None

        assets = await data_store.get_asset_states(ids, types, statuses)

        # Pagination
        page = pagination.page if pagination else 1
        page_size = pagination.page_size if pagination else 20
        total_count = len(assets)
        total_pages = (total_count + page_size - 1) // page_size

        start_idx = (page - 1) * page_size
        end_idx = start_idx + page_size
        paginated_assets = assets[start_idx:end_idx]

        return PaginatedAssets(
            items=[convert_asset_to_graphql(a) for a in paginated_assets],
            total_count=total_count,
            page=page,
            page_size=page_size,
            total_pages=total_pages,
            has_next=page < total_pages,
            has_previous=page > 1,
        )

    @strawberry.field(description="Get active constraints")
    async def constraints(
        self,
        info: Info,
        is_active: Optional[bool] = None,
        constraint_types: Optional[List[ConstraintTypeEnum]] = None,
        priorities: Optional[List[ConstraintPriorityEnum]] = None,
        pagination: Optional[PaginationInput] = None,
    ) -> PaginatedConstraints:
        """Get constraints."""
        types = [ConstraintType(t.value) for t in constraint_types] if constraint_types else None
        prios = [ConstraintPriority(p.value) for p in priorities] if priorities else None

        constraints = await data_store.get_constraints(is_active, types, prios)

        # Pagination
        page = pagination.page if pagination else 1
        page_size = pagination.page_size if pagination else 20
        total_count = len(constraints)
        total_pages = (total_count + page_size - 1) // page_size

        start_idx = (page - 1) * page_size
        end_idx = start_idx + page_size
        paginated = constraints[start_idx:end_idx]

        return PaginatedConstraints(
            items=[convert_constraint_to_graphql(c) for c in paginated],
            total_count=total_count,
            page=page,
            page_size=page_size,
            total_pages=total_pages,
            has_next=page < total_pages,
            has_previous=page > 1,
        )

    @strawberry.field(description="Get KPI metrics")
    async def kpis(
        self,
        info: Info,
        category: Optional[str] = None,
        time_range: Optional[TimeRangeInput] = None,
    ) -> List[KPI]:
        """Get KPIs."""
        kpis = await data_store.get_kpis(category, time_range)
        return [convert_kpi_to_graphql(k) for k in kpis]

    @strawberry.field(description="Get explainability summary for a dispatch plan")
    async def explainability_summary(
        self,
        info: Info,
        plan_id: strawberry.ID,
    ) -> Optional[ExplainabilitySummary]:
        """Get SHAP/LIME explainability summary."""
        summary = await data_store.get_explainability_summary(str(plan_id))
        if summary:
            return convert_explainability_to_graphql(summary)
        return None

    @strawberry.field(description="Get alarms and safety events")
    async def alarms(
        self,
        info: Info,
        severity: Optional[List[AlarmSeverityEnum]] = None,
        status: Optional[List[AlarmStatusEnum]] = None,
        asset_id: Optional[strawberry.ID] = None,
        pagination: Optional[PaginationInput] = None,
    ) -> PaginatedAlarms:
        """Get alarms."""
        sev = [AlarmSeverity(s.value) for s in severity] if severity else None
        stat = [AlarmStatus(s.value) for s in status] if status else None

        alarms = await data_store.get_alarms(sev, stat, str(asset_id) if asset_id else None)

        # Pagination
        page = pagination.page if pagination else 1
        page_size = pagination.page_size if pagination else 20
        total_count = len(alarms)
        total_pages = (total_count + page_size - 1) // page_size

        start_idx = (page - 1) * page_size
        end_idx = start_idx + page_size
        paginated = alarms[start_idx:end_idx]

        return PaginatedAlarms(
            items=[
                AlarmEvent(
                    alarm_id=strawberry.ID(str(a.alarm_id)),
                    alarm_code=a.alarm_code,
                    name=a.name,
                    description=a.description,
                    severity=AlarmSeverityEnum(a.severity.value),
                    status=AlarmStatusEnum(a.status.value),
                    asset_id=strawberry.ID(str(a.asset_id)) if a.asset_id else None,
                    asset_name=a.asset_name,
                    subsystem=a.subsystem,
                    triggered_at=a.triggered_at,
                    acknowledged_at=a.acknowledged_at,
                    acknowledged_by=a.acknowledged_by,
                    resolved_at=a.resolved_at,
                    resolved_by=a.resolved_by,
                    measured_value=a.measured_value,
                    threshold_value=a.threshold_value,
                    unit=a.unit,
                    recommended_actions=a.recommended_actions,
                    auto_response_triggered=a.auto_response_triggered,
                    auto_response_description=a.auto_response_description,
                    created_at=a.created_at,
                )
                for a in paginated
            ],
            total_count=total_count,
            page=page,
            page_size=page_size,
            total_pages=total_pages,
            has_next=page < total_pages,
            has_previous=page > 1,
        )

    @strawberry.field(description="Get maintenance triggers")
    async def maintenance_triggers(
        self,
        info: Info,
        urgency: Optional[List[MaintenanceUrgencyEnum]] = None,
        asset_id: Optional[strawberry.ID] = None,
    ) -> List[MaintenanceTrigger]:
        """Get maintenance triggers."""
        urg = [MaintenanceUrgency(u.value) for u in urgency] if urgency else None
        triggers = await data_store.get_maintenance_triggers(urg, str(asset_id) if asset_id else None)

        return [
            MaintenanceTrigger(
                trigger_id=strawberry.ID(str(t.trigger_id)),
                asset_id=strawberry.ID(str(t.asset_id)),
                asset_name=t.asset_name,
                maintenance_type=MaintenanceTypeEnum(t.maintenance_type.value),
                urgency=MaintenanceUrgencyEnum(t.urgency.value),
                trigger_reason=t.trigger_reason,
                trigger_metric=t.trigger_metric,
                current_value=t.current_value,
                threshold_value=t.threshold_value,
                recommended_action=t.recommended_action,
                estimated_duration_hours=t.estimated_duration_hours,
                estimated_cost=t.estimated_cost,
                recommended_start_date=t.recommended_start_date,
                latest_start_date=t.latest_start_date,
                production_impact_mw=t.production_impact_mw,
                downtime_hours=t.downtime_hours,
                created_at=t.created_at,
            )
            for t in triggers
        ]

    @strawberry.field(description="Get latest forecast by type")
    async def latest_forecast(
        self,
        info: Info,
        forecast_type: ForecastTypeEnum,
    ) -> Optional[ForecastData]:
        """Get latest forecast."""
        forecast = await data_store.get_latest_forecast(ForecastType(forecast_type.value))
        if forecast:
            return ForecastData(
                forecast_id=strawberry.ID(str(forecast.forecast_id)),
                forecast_type=ForecastTypeEnum(forecast.forecast_type.value),
                forecast_horizon_hours=forecast.forecast_horizon_hours,
                resolution_minutes=forecast.resolution_minutes,
                generated_at=forecast.generated_at,
                valid_from=forecast.valid_from,
                valid_until=forecast.valid_until,
                values=forecast.values,
                timestamps=forecast.timestamps,
                unit=forecast.unit,
                confidence_level=forecast.confidence_level,
                lower_bounds=forecast.lower_bounds,
                upper_bounds=forecast.upper_bounds,
                model_name=forecast.model_name,
                model_version=forecast.model_version,
            )
        return None


# =============================================================================
# GraphQL Mutation Type
# =============================================================================

@strawberry.type
class Mutation:
    """GraphQL Mutation type for ThermalCommand."""

    @strawberry.mutation(description="Submit demand forecast update")
    async def submit_demand_update(
        self,
        info: Info,
        input: DemandUpdateInput,
    ) -> DemandUpdateResponse:
        """Submit a demand update."""
        return await data_store.submit_demand_update(input)

    @strawberry.mutation(description="Request heat allocation optimization")
    async def request_allocation(
        self,
        info: Info,
        input: AllocationRequestInput,
    ) -> AllocationResponse:
        """Request heat allocation."""
        return await data_store.request_allocation(input)

    @strawberry.mutation(description="Acknowledge a safety alarm")
    async def acknowledge_alarm(
        self,
        info: Info,
        input: AlarmAcknowledgementInput,
    ) -> AlarmAcknowledgementResponse:
        """Acknowledge an alarm."""
        return await data_store.acknowledge_alarm(input)


# =============================================================================
# GraphQL Subscription Type
# =============================================================================

@strawberry.type
class Subscription:
    """GraphQL Subscription type for ThermalCommand."""

    @strawberry.subscription(description="Real-time dispatch plan updates")
    async def plan_updates(self, info: Info) -> AsyncGenerator[DispatchPlan, None]:
        """Subscribe to plan updates."""
        while True:
            try:
                # Wait for plan update with timeout
                plan = await asyncio.wait_for(
                    data_store._plan_update_queue.get(),
                    timeout=60.0,
                )
                yield convert_plan_to_graphql(plan)
            except asyncio.TimeoutError:
                # Send heartbeat by yielding current plan
                current_plan = await data_store.get_current_plan()
                if current_plan:
                    yield convert_plan_to_graphql(current_plan)

    @strawberry.subscription(description="Real-time setpoint recommendations stream")
    async def action_recommendations(self, info: Info) -> AsyncGenerator[List[SetpointRecommendation], None]:
        """Subscribe to action recommendations."""
        while True:
            try:
                recommendations = await asyncio.wait_for(
                    data_store._action_recommendation_queue.get(),
                    timeout=60.0,
                )
                yield recommendations
            except asyncio.TimeoutError:
                # Send empty list as heartbeat
                yield []

    @strawberry.subscription(description="Real-time alarm and safety event stream")
    async def alarm_safety_events(
        self,
        info: Info,
        severity_filter: Optional[List[AlarmSeverityEnum]] = None,
    ) -> AsyncGenerator[AlarmEvent, None]:
        """Subscribe to alarm events."""
        while True:
            try:
                alarm = await asyncio.wait_for(
                    data_store._alarm_event_queue.get(),
                    timeout=60.0,
                )

                # Apply severity filter
                if severity_filter:
                    filter_values = [s.value for s in severity_filter]
                    if alarm.severity.value not in filter_values:
                        continue

                yield AlarmEvent(
                    alarm_id=strawberry.ID(str(alarm.alarm_id)),
                    alarm_code=alarm.alarm_code,
                    name=alarm.name,
                    description=alarm.description,
                    severity=AlarmSeverityEnum(alarm.severity.value),
                    status=AlarmStatusEnum(alarm.status.value),
                    asset_id=strawberry.ID(str(alarm.asset_id)) if alarm.asset_id else None,
                    asset_name=alarm.asset_name,
                    subsystem=alarm.subsystem,
                    triggered_at=alarm.triggered_at,
                    acknowledged_at=alarm.acknowledged_at,
                    acknowledged_by=alarm.acknowledged_by,
                    resolved_at=alarm.resolved_at,
                    resolved_by=alarm.resolved_by,
                    measured_value=alarm.measured_value,
                    threshold_value=alarm.threshold_value,
                    unit=alarm.unit,
                    recommended_actions=alarm.recommended_actions,
                    auto_response_triggered=alarm.auto_response_triggered,
                    auto_response_description=alarm.auto_response_description,
                    created_at=alarm.created_at,
                )
            except asyncio.TimeoutError:
                continue

    @strawberry.subscription(description="Real-time maintenance trigger notifications")
    async def maintenance_triggers(
        self,
        info: Info,
        urgency_filter: Optional[List[MaintenanceUrgencyEnum]] = None,
    ) -> AsyncGenerator[MaintenanceTrigger, None]:
        """Subscribe to maintenance triggers."""
        while True:
            try:
                trigger = await asyncio.wait_for(
                    data_store._maintenance_trigger_queue.get(),
                    timeout=60.0,
                )

                # Apply urgency filter
                if urgency_filter:
                    filter_values = [u.value for u in urgency_filter]
                    if trigger.urgency.value not in filter_values:
                        continue

                yield MaintenanceTrigger(
                    trigger_id=strawberry.ID(str(trigger.trigger_id)),
                    asset_id=strawberry.ID(str(trigger.asset_id)),
                    asset_name=trigger.asset_name,
                    maintenance_type=MaintenanceTypeEnum(trigger.maintenance_type.value),
                    urgency=MaintenanceUrgencyEnum(trigger.urgency.value),
                    trigger_reason=trigger.trigger_reason,
                    trigger_metric=trigger.trigger_metric,
                    current_value=trigger.current_value,
                    threshold_value=trigger.threshold_value,
                    recommended_action=trigger.recommended_action,
                    estimated_duration_hours=trigger.estimated_duration_hours,
                    estimated_cost=trigger.estimated_cost,
                    recommended_start_date=trigger.recommended_start_date,
                    latest_start_date=trigger.latest_start_date,
                    production_impact_mw=trigger.production_impact_mw,
                    downtime_hours=trigger.downtime_hours,
                    created_at=trigger.created_at,
                )
            except asyncio.TimeoutError:
                continue


# =============================================================================
# Schema and Router
# =============================================================================

# Create Strawberry schema
schema = strawberry.Schema(
    query=Query,
    mutation=Mutation,
    subscription=Subscription,
)

# Create GraphQL router for FastAPI
graphql_app = GraphQLRouter(
    schema,
    subscription_protocols=[
        GRAPHQL_WS_PROTOCOL,
        GRAPHQL_TRANSPORT_WS_PROTOCOL,
    ],
    context_getter=lambda: {"request": None},  # Will be overridden in main app
)


# =============================================================================
# Export Functions
# =============================================================================

async def broadcast_plan_update(plan: DispatchPlanSchema) -> None:
    """Broadcast a plan update to all subscribers."""
    await data_store._plan_update_queue.put(plan)


async def broadcast_alarm_event(alarm: AlarmEventSchema) -> None:
    """Broadcast an alarm event to all subscribers."""
    await data_store._alarm_event_queue.put(alarm)


async def broadcast_maintenance_trigger(trigger: MaintenanceTriggerSchema) -> None:
    """Broadcast a maintenance trigger to all subscribers."""
    await data_store._maintenance_trigger_queue.put(trigger)
