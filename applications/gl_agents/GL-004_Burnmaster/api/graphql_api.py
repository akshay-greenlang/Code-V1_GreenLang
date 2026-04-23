"""
GL-004 BURNMASTER GraphQL API

Strawberry GraphQL schema for burner optimization operations.
Includes queries, mutations, and subscriptions for real-time data.
"""

import strawberry
from strawberry.types import Info
from strawberry.scalars import JSON
from typing import List, Optional, AsyncGenerator
from datetime import datetime, timedelta
from enum import Enum
import asyncio
import uuid
import logging

logger = logging.getLogger(__name__)


# ============================================================================
# Enums
# ============================================================================

@strawberry.enum
class OperatingModeGQL(Enum):
    NORMAL = "normal"
    ECO = "eco"
    HIGH_EFFICIENCY = "high_efficiency"
    MAINTENANCE = "maintenance"
    EMERGENCY = "emergency"
    STARTUP = "startup"
    SHUTDOWN = "shutdown"


@strawberry.enum
class BurnerStateGQL(Enum):
    RUNNING = "running"
    IDLE = "idle"
    STARTING = "starting"
    STOPPING = "stopping"
    ERROR = "error"
    MAINTENANCE = "maintenance"
    OFFLINE = "offline"


@strawberry.enum
class RecommendationPriorityGQL(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@strawberry.enum
class RecommendationStatusGQL(Enum):
    PENDING = "pending"
    ACCEPTED = "accepted"
    REJECTED = "rejected"
    IMPLEMENTED = "implemented"
    EXPIRED = "expired"


@strawberry.enum
class AlertSeverityGQL(Enum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@strawberry.enum
class AlertStatusGQL(Enum):
    ACTIVE = "active"
    ACKNOWLEDGED = "acknowledged"
    RESOLVED = "resolved"
    SUPPRESSED = "suppressed"


@strawberry.enum
class OptimizationStateGQL(Enum):
    IDLE = "idle"
    ANALYZING = "analyzing"
    OPTIMIZING = "optimizing"
    APPLYING = "applying"
    MONITORING = "monitoring"
    ERROR = "error"


# ============================================================================
# Types
# ============================================================================

@strawberry.type
class BurnerMetrics:
    """Real-time burner metrics."""
    firing_rate: float
    fuel_flow_rate: float
    air_flow_rate: float
    combustion_air_temp: float
    flue_gas_temp: float
    oxygen_level: float
    co_level: float
    nox_level: float
    efficiency: float
    heat_output: float


@strawberry.type
class Unit:
    """Burner unit type."""
    id: str
    name: str
    state: BurnerStateGQL
    mode: OperatingModeGQL
    metrics: BurnerMetrics
    uptime_hours: float
    last_maintenance: Optional[datetime]
    next_maintenance: Optional[datetime]
    active_alerts_count: int
    timestamp: datetime

    @strawberry.field
    def recommendations(self, limit: int = 10) -> List["Recommendation"]:
        """Get recommendations for this unit."""
        return get_recommendations_for_unit(self.id, limit)

    @strawberry.field
    def alerts(self, limit: int = 10) -> List["Alert"]:
        """Get alerts for this unit."""
        return get_alerts_for_unit(self.id, limit)

    @strawberry.field
    def kpis(self) -> "KPIDashboard":
        """Get KPIs for this unit."""
        return get_kpis_for_unit(self.id)


@strawberry.type
class KPIValue:
    """Single KPI value."""
    name: str
    value: float
    unit: str
    target: Optional[float]
    trend: Optional[str]
    change_percent: Optional[float]


@strawberry.type
class EmissionsKPIs:
    """Emissions KPIs."""
    co2_emissions: KPIValue
    nox_emissions: KPIValue
    co_emissions: KPIValue
    particulate_matter: KPIValue
    carbon_intensity: KPIValue


@strawberry.type
class EfficiencyKPIs:
    """Efficiency KPIs."""
    thermal_efficiency: KPIValue
    combustion_efficiency: KPIValue
    fuel_utilization: KPIValue
    heat_recovery: KPIValue
    overall_equipment_effectiveness: KPIValue


@strawberry.type
class OperationalKPIs:
    """Operational KPIs."""
    availability: KPIValue
    reliability: KPIValue
    mean_time_between_failures: KPIValue
    mean_time_to_repair: KPIValue
    capacity_utilization: KPIValue


@strawberry.type
class KPIDashboard:
    """KPI dashboard."""
    unit_id: str
    period_start: datetime
    period_end: datetime
    emissions: EmissionsKPIs
    efficiency: EfficiencyKPIs
    operational: OperationalKPIs
    overall_score: float
    timestamp: datetime


@strawberry.type
class RecommendationAction:
    """Recommendation action."""
    action_id: str
    description: str
    parameter: str
    current_value: float
    recommended_value: float
    unit: str
    auto_applicable: bool


@strawberry.type
class RecommendationImpact:
    """Recommendation impact."""
    efficiency_improvement: Optional[float]
    emissions_reduction: Optional[float]
    cost_savings: Optional[float]
    energy_savings: Optional[float]
    confidence_level: float


@strawberry.type
class Recommendation:
    """Optimization recommendation."""
    id: str
    unit_id: str
    title: str
    description: str
    priority: RecommendationPriorityGQL
    status: RecommendationStatusGQL
    category: str
    actions: List[RecommendationAction]
    impact: RecommendationImpact
    reasoning: str
    model_version: str
    valid_until: datetime
    created_at: datetime
    accepted_at: Optional[datetime]
    implemented_at: Optional[datetime]
    accepted_by: Optional[str]


@strawberry.type
class OptimizationMetrics:
    """Optimization engine metrics."""
    recommendations_generated: int
    recommendations_accepted: int
    recommendations_implemented: int
    average_confidence: float
    total_savings_achieved: float
    efficiency_improvement: float


@strawberry.type
class OptimizationStatus:
    """Optimization engine status."""
    unit_id: str
    state: OptimizationStateGQL
    is_active: bool
    last_analysis: Optional[datetime]
    next_analysis: Optional[datetime]
    analysis_interval_minutes: int
    metrics: OptimizationMetrics
    active_model: str
    model_accuracy: float
    data_quality_score: float
    constraints_active: List[str]
    timestamp: datetime


@strawberry.type
class Alert:
    """Alert type."""
    id: str
    unit_id: str
    severity: AlertSeverityGQL
    status: AlertStatusGQL
    title: str
    description: str
    source: str
    metric_name: Optional[str]
    metric_value: Optional[float]
    threshold: Optional[float]
    recommended_action: Optional[str]
    acknowledged_by: Optional[str]
    acknowledged_at: Optional[datetime]
    resolved_at: Optional[datetime]
    created_at: datetime


@strawberry.type
class ModeChangeResult:
    """Mode change result."""
    unit_id: str
    previous_mode: OperatingModeGQL
    new_mode: OperatingModeGQL
    status: str
    scheduled_time: Optional[datetime]
    estimated_completion: Optional[datetime]
    transition_steps: List[str]
    changed_by: str
    changed_at: datetime


@strawberry.type
class AcceptRecommendationResult:
    """Accept recommendation result."""
    recommendation_id: str
    status: RecommendationStatusGQL
    implementation_status: str
    scheduled_time: Optional[datetime]
    estimated_completion: Optional[datetime]
    accepted_by: str
    accepted_at: datetime


@strawberry.type
class AcknowledgeAlertResult:
    """Acknowledge alert result."""
    alert_id: str
    status: AlertStatusGQL
    acknowledged_by: str
    acknowledged_at: datetime


# ============================================================================
# Input Types
# ============================================================================

@strawberry.input
class AcceptRecommendationInput:
    """Input for accepting a recommendation."""
    auto_implement: bool = False
    scheduled_time: Optional[datetime] = None
    notes: Optional[str] = None
    override_safety_check: bool = False


@strawberry.input
class ModeChangeInput:
    """Input for changing operating mode."""
    new_mode: OperatingModeGQL
    reason: str
    scheduled_time: Optional[datetime] = None
    transition_duration_minutes: Optional[int] = None
    notify_operators: bool = True


@strawberry.input
class AcknowledgeAlertInput:
    """Input for acknowledging an alert."""
    notes: Optional[str] = None
    suppress_duration_minutes: Optional[int] = None


# ============================================================================
# Helper Functions
# ============================================================================

def get_mock_metrics() -> BurnerMetrics:
    """Generate mock metrics."""
    import random
    return BurnerMetrics(
        firing_rate=75.5 + random.uniform(-5, 5),
        fuel_flow_rate=120.0 + random.uniform(-10, 10),
        air_flow_rate=1500.0 + random.uniform(-50, 50),
        combustion_air_temp=35.0 + random.uniform(-2, 2),
        flue_gas_temp=180.0 + random.uniform(-10, 10),
        oxygen_level=3.5 + random.uniform(-0.5, 0.5),
        co_level=15.0 + random.uniform(-5, 5),
        nox_level=45.0 + random.uniform(-5, 5),
        efficiency=94.2 + random.uniform(-1, 1),
        heat_output=12.5 + random.uniform(-0.5, 0.5)
    )


def get_recommendations_for_unit(unit_id: str, limit: int) -> List[Recommendation]:
    """Get recommendations for a unit."""
    return [
        Recommendation(
            id=f"rec-{uuid.uuid4().hex[:8]}",
            unit_id=unit_id,
            title="Optimize Air-Fuel Ratio",
            description="Reduce excess air for improved efficiency",
            priority=RecommendationPriorityGQL.HIGH,
            status=RecommendationStatusGQL.PENDING,
            category="efficiency",
            actions=[
                RecommendationAction(
                    action_id="act-001",
                    description="Reduce excess air",
                    parameter="excess_air_percentage",
                    current_value=15.0,
                    recommended_value=12.0,
                    unit="%",
                    auto_applicable=True
                )
            ],
            impact=RecommendationImpact(
                efficiency_improvement=2.3,
                emissions_reduction=1.5,
                cost_savings=450.0,
                energy_savings=125.0,
                confidence_level=0.92
            ),
            reasoning="Historical analysis shows excess air above optimal range",
            model_version="v2.1.0",
            valid_until=datetime.utcnow() + timedelta(hours=24),
            created_at=datetime.utcnow(),
            accepted_at=None,
            implemented_at=None,
            accepted_by=None
        )
    ][:limit]


def get_alerts_for_unit(unit_id: str, limit: int) -> List[Alert]:
    """Get alerts for a unit."""
    return [
        Alert(
            id=f"alert-{uuid.uuid4().hex[:8]}",
            unit_id=unit_id,
            severity=AlertSeverityGQL.WARNING,
            status=AlertStatusGQL.ACTIVE,
            title="High Flue Gas Temperature",
            description="Flue gas temperature exceeds optimal range",
            source="temperature_monitor",
            metric_name="flue_gas_temp",
            metric_value=195.0,
            threshold=190.0,
            recommended_action="Check heat exchanger efficiency",
            acknowledged_by=None,
            acknowledged_at=None,
            resolved_at=None,
            created_at=datetime.utcnow() - timedelta(minutes=30)
        )
    ][:limit]


def get_kpis_for_unit(unit_id: str) -> KPIDashboard:
    """Get KPIs for a unit."""
    def create_kpi(name: str, value: float, unit: str, target: float) -> KPIValue:
        return KPIValue(
            name=name,
            value=value,
            unit=unit,
            target=target,
            trend="stable",
            change_percent=((value - target) / target) * 100
        )

    return KPIDashboard(
        unit_id=unit_id,
        period_start=datetime.utcnow() - timedelta(hours=24),
        period_end=datetime.utcnow(),
        emissions=EmissionsKPIs(
            co2_emissions=create_kpi("CO2 Emissions", 245.5, "kg/h", 250.0),
            nox_emissions=create_kpi("NOx Emissions", 42.3, "ppm", 50.0),
            co_emissions=create_kpi("CO Emissions", 14.5, "ppm", 20.0),
            particulate_matter=create_kpi("Particulate Matter", 8.2, "mg/m3", 10.0),
            carbon_intensity=create_kpi("Carbon Intensity", 0.45, "kg CO2/kWh", 0.50)
        ),
        efficiency=EfficiencyKPIs(
            thermal_efficiency=create_kpi("Thermal Efficiency", 92.5, "%", 95.0),
            combustion_efficiency=create_kpi("Combustion Efficiency", 94.2, "%", 95.0),
            fuel_utilization=create_kpi("Fuel Utilization", 89.5, "%", 90.0),
            heat_recovery=create_kpi("Heat Recovery", 78.3, "%", 80.0),
            overall_equipment_effectiveness=create_kpi("OEE", 87.5, "%", 90.0)
        ),
        operational=OperationalKPIs(
            availability=create_kpi("Availability", 98.5, "%", 99.0),
            reliability=create_kpi("Reliability", 97.2, "%", 98.0),
            mean_time_between_failures=create_kpi("MTBF", 720, "hours", 750),
            mean_time_to_repair=create_kpi("MTTR", 2.5, "hours", 2.0),
            capacity_utilization=create_kpi("Capacity Utilization", 78.5, "%", 80.0)
        ),
        overall_score=89.5,
        timestamp=datetime.utcnow()
    )


# ============================================================================
# Query
# ============================================================================

@strawberry.type
class Query:
    """GraphQL Query type."""

    @strawberry.field
    def unit(self, id: str) -> Optional[Unit]:
        """Get a single unit by ID."""
        units_data = {
            "burner-001": ("Main Boiler Burner 1", BurnerStateGQL.RUNNING, OperatingModeGQL.NORMAL),
            "burner-002": ("Main Boiler Burner 2", BurnerStateGQL.RUNNING, OperatingModeGQL.ECO)
        }

        if id not in units_data:
            return None

        name, state, mode = units_data[id]
        return Unit(
            id=id,
            name=name,
            state=state,
            mode=mode,
            metrics=get_mock_metrics(),
            uptime_hours=1250.5,
            last_maintenance=datetime.utcnow() - timedelta(days=30),
            next_maintenance=datetime.utcnow() + timedelta(days=60),
            active_alerts_count=2,
            timestamp=datetime.utcnow()
        )

    @strawberry.field
    def units(self, limit: int = 10, offset: int = 0) -> List[Unit]:
        """Get all units."""
        units_data = [
            ("burner-001", "Main Boiler Burner 1", BurnerStateGQL.RUNNING, OperatingModeGQL.NORMAL),
            ("burner-002", "Main Boiler Burner 2", BurnerStateGQL.RUNNING, OperatingModeGQL.ECO)
        ]

        units = []
        for unit_id, name, state, mode in units_data[offset:offset + limit]:
            units.append(Unit(
                id=unit_id,
                name=name,
                state=state,
                mode=mode,
                metrics=get_mock_metrics(),
                uptime_hours=1250.5,
                last_maintenance=datetime.utcnow() - timedelta(days=30),
                next_maintenance=datetime.utcnow() + timedelta(days=60),
                active_alerts_count=2,
                timestamp=datetime.utcnow()
            ))
        return units

    @strawberry.field
    def recommendations(
        self,
        unit_id: Optional[str] = None,
        status: Optional[RecommendationStatusGQL] = None,
        priority: Optional[RecommendationPriorityGQL] = None,
        limit: int = 20
    ) -> List[Recommendation]:
        """Get recommendations with optional filters."""
        recommendations = []
        unit_ids = [unit_id] if unit_id else ["burner-001", "burner-002"]

        for uid in unit_ids:
            recommendations.extend(get_recommendations_for_unit(uid, limit))

        if status:
            recommendations = [r for r in recommendations if r.status == status]
        if priority:
            recommendations = [r for r in recommendations if r.priority == priority]

        return recommendations[:limit]

    @strawberry.field
    def kpis(self, unit_id: str, period: str = "24h") -> KPIDashboard:
        """Get KPIs for a unit."""
        return get_kpis_for_unit(unit_id)

    @strawberry.field
    def alerts(
        self,
        unit_id: Optional[str] = None,
        severity: Optional[AlertSeverityGQL] = None,
        status: Optional[AlertStatusGQL] = None,
        limit: int = 50
    ) -> List[Alert]:
        """Get alerts with optional filters."""
        alerts = []
        unit_ids = [unit_id] if unit_id else ["burner-001", "burner-002"]

        for uid in unit_ids:
            alerts.extend(get_alerts_for_unit(uid, limit))

        if severity:
            alerts = [a for a in alerts if a.severity == severity]
        if status:
            alerts = [a for a in alerts if a.status == status]

        return alerts[:limit]

    @strawberry.field
    def optimization_status(self, unit_id: str) -> OptimizationStatus:
        """Get optimization engine status for a unit."""
        return OptimizationStatus(
            unit_id=unit_id,
            state=OptimizationStateGQL.MONITORING,
            is_active=True,
            last_analysis=datetime.utcnow() - timedelta(minutes=15),
            next_analysis=datetime.utcnow() + timedelta(minutes=15),
            analysis_interval_minutes=30,
            metrics=OptimizationMetrics(
                recommendations_generated=45,
                recommendations_accepted=38,
                recommendations_implemented=35,
                average_confidence=0.89,
                total_savings_achieved=12500.0,
                efficiency_improvement=3.2
            ),
            active_model="combustion_optimizer_v2.1",
            model_accuracy=0.94,
            data_quality_score=0.92,
            constraints_active=["emissions_limit", "safety_margin", "ramp_rate"],
            timestamp=datetime.utcnow()
        )


# ============================================================================
# Mutation
# ============================================================================

@strawberry.type
class Mutation:
    """GraphQL Mutation type."""

    @strawberry.mutation
    def accept_recommendation(
        self,
        recommendation_id: str,
        input: AcceptRecommendationInput
    ) -> AcceptRecommendationResult:
        """Accept a recommendation."""
        implementation_status = "scheduled" if input.scheduled_time else (
            "implementing" if input.auto_implement else "pending_manual"
        )

        return AcceptRecommendationResult(
            recommendation_id=recommendation_id,
            status=RecommendationStatusGQL.ACCEPTED,
            implementation_status=implementation_status,
            scheduled_time=input.scheduled_time,
            estimated_completion=input.scheduled_time or (
                datetime.utcnow() + timedelta(minutes=5) if input.auto_implement else None
            ),
            accepted_by="current_user@example.com",
            accepted_at=datetime.utcnow()
        )

    @strawberry.mutation
    def change_mode(
        self,
        unit_id: str,
        input: ModeChangeInput
    ) -> ModeChangeResult:
        """Change operating mode of a unit."""
        return ModeChangeResult(
            unit_id=unit_id,
            previous_mode=OperatingModeGQL.NORMAL,
            new_mode=input.new_mode,
            status="completed" if not input.scheduled_time else "scheduled",
            scheduled_time=input.scheduled_time,
            estimated_completion=input.scheduled_time or datetime.utcnow(),
            transition_steps=[
                "Validate safety conditions",
                "Adjust control parameters",
                "Verify stable operation"
            ],
            changed_by="current_user@example.com",
            changed_at=datetime.utcnow()
        )

    @strawberry.mutation
    def acknowledge_alert(
        self,
        alert_id: str,
        input: AcknowledgeAlertInput
    ) -> AcknowledgeAlertResult:
        """Acknowledge an alert."""
        return AcknowledgeAlertResult(
            alert_id=alert_id,
            status=AlertStatusGQL.ACKNOWLEDGED,
            acknowledged_by="current_user@example.com",
            acknowledged_at=datetime.utcnow()
        )


# ============================================================================
# Subscription
# ============================================================================

@strawberry.type
class Subscription:
    """GraphQL Subscription type for real-time updates."""

    @strawberry.subscription
    async def unit_status(self, unit_id: str) -> AsyncGenerator[Unit, None]:
        """Subscribe to real-time unit status updates."""
        while True:
            yield Unit(
                id=unit_id,
                name="Main Boiler Burner 1",
                state=BurnerStateGQL.RUNNING,
                mode=OperatingModeGQL.NORMAL,
                metrics=get_mock_metrics(),
                uptime_hours=1250.5,
                last_maintenance=datetime.utcnow() - timedelta(days=30),
                next_maintenance=datetime.utcnow() + timedelta(days=60),
                active_alerts_count=2,
                timestamp=datetime.utcnow()
            )
            await asyncio.sleep(5)  # Update every 5 seconds

    @strawberry.subscription
    async def new_recommendation(self, unit_id: str) -> AsyncGenerator[Recommendation, None]:
        """Subscribe to new recommendations for a unit."""
        while True:
            await asyncio.sleep(30)  # Check every 30 seconds
            recommendations = get_recommendations_for_unit(unit_id, 1)
            if recommendations:
                yield recommendations[0]

    @strawberry.subscription
    async def alert_triggered(
        self,
        unit_id: Optional[str] = None,
        min_severity: Optional[AlertSeverityGQL] = None
    ) -> AsyncGenerator[Alert, None]:
        """Subscribe to alerts."""
        while True:
            await asyncio.sleep(10)  # Check every 10 seconds
            alerts = get_alerts_for_unit(unit_id or "burner-001", 1)
            if alerts:
                alert = alerts[0]
                if min_severity is None or alert.severity.value >= min_severity.value:
                    yield alert


# ============================================================================
# Schema
# ============================================================================

schema = strawberry.Schema(
    query=Query,
    mutation=Mutation,
    subscription=Subscription
)
