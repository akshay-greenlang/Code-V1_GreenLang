"""
GL-004 BURNMASTER REST API

FastAPI REST endpoints for burner optimization operations.
Includes status monitoring, KPIs, recommendations, alerts, and mode management.
"""

from fastapi import APIRouter, Depends, HTTPException, status, Request, Query
from fastapi.responses import JSONResponse
from typing import List, Optional
from datetime import datetime, timedelta
import logging
import uuid

from .api_schemas import (
    BurnerStatusResponse, BurnerMetrics, BurnerState, OperatingMode,
    KPIDashboardResponse, KPIValue, EmissionsKPIs, EfficiencyKPIs, OperationalKPIs,
    RecommendationResponse, RecommendationAction, RecommendationImpact,
    RecommendationPriority, RecommendationStatus,
    AcceptRecommendationRequest, AcceptRecommendationResponse,
    OptimizationStatusResponse, OptimizationMetrics, OptimizationState,
    ModeChangeRequest, ModeChangeResponse,
    HistoryRequest, HistoryResponse, HistoryDataPoint,
    AlertResponse, AlertSeverity, AlertStatus, AlertAcknowledgeRequest,
    HealthResponse, ServiceHealth
)
from .api_auth import (
    User, get_current_user, require_roles, require_operator,
    require_engineer, require_admin, audit_logger, audit_action
)
from .api_schemas import UserRole
from .config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

# Create router
router = APIRouter(prefix="/api/v1", tags=["Burner Optimization"])


# ============================================================================
# Mock Data Store (Replace with actual database in production)
# ============================================================================

class MockDataStore:
    """Mock data store for demonstration."""

    def __init__(self):
        self.units = {
            "burner-001": {
                "name": "Main Boiler Burner 1",
                "state": BurnerState.RUNNING,
                "mode": OperatingMode.NORMAL,
                "uptime_hours": 1250.5
            },
            "burner-002": {
                "name": "Main Boiler Burner 2",
                "state": BurnerState.RUNNING,
                "mode": OperatingMode.ECO,
                "uptime_hours": 980.2
            }
        }
        self.recommendations = {}
        self.alerts = {}

    def get_unit(self, unit_id: str) -> Optional[dict]:
        return self.units.get(unit_id)

    def get_metrics(self, unit_id: str) -> BurnerMetrics:
        # Generate mock metrics
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


data_store = MockDataStore()


# ============================================================================
# Helper Functions
# ============================================================================

def validate_unit_access(unit_id: str, user: User):
    """Validate that user has access to the unit."""
    unit = data_store.get_unit(unit_id)
    if not unit:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Unit {unit_id} not found"
        )
    # Add tenant validation in production
    return unit


# ============================================================================
# Status Endpoints
# ============================================================================

@router.get(
    "/units/{unit_id}/status",
    response_model=BurnerStatusResponse,
    summary="Get burner status",
    description="Retrieve current operational status and real-time metrics for a burner unit.",
    responses={
        200: {"description": "Current burner status"},
        404: {"description": "Unit not found"},
        401: {"description": "Not authenticated"},
        403: {"description": "Insufficient permissions"}
    }
)
async def get_unit_status(
    request: Request,
    unit_id: str,
    current_user: User = Depends(require_operator())
) -> BurnerStatusResponse:
    """
    Get current burner status.

    Args:
        unit_id: Unique unit identifier
        current_user: Authenticated user

    Returns:
        Current burner status with metrics
    """
    unit = validate_unit_access(unit_id, current_user)

    # Log audit
    await audit_logger.log(
        user=current_user,
        request=request,
        action="read",
        resource_type="unit_status",
        resource_id=unit_id,
        status_code=200,
        response_time_ms=0
    )

    return BurnerStatusResponse(
        unit_id=unit_id,
        name=unit["name"],
        state=unit["state"],
        mode=unit["mode"],
        metrics=data_store.get_metrics(unit_id),
        uptime_hours=unit["uptime_hours"],
        last_maintenance=datetime.utcnow() - timedelta(days=30),
        next_maintenance=datetime.utcnow() + timedelta(days=60),
        active_alerts_count=2,
        timestamp=datetime.utcnow()
    )


# ============================================================================
# KPI Endpoints
# ============================================================================

@router.get(
    "/units/{unit_id}/kpis",
    response_model=KPIDashboardResponse,
    summary="Get KPI dashboard",
    description="Retrieve key performance indicators for a burner unit."
)
async def get_unit_kpis(
    request: Request,
    unit_id: str,
    period: str = Query("24h", description="Time period: 1h, 6h, 24h, 7d, 30d"),
    current_user: User = Depends(require_operator())
) -> KPIDashboardResponse:
    """Get KPI dashboard for a unit."""
    validate_unit_access(unit_id, current_user)

    # Parse period
    period_hours = {"1h": 1, "6h": 6, "24h": 24, "7d": 168, "30d": 720}.get(period, 24)
    period_end = datetime.utcnow()
    period_start = period_end - timedelta(hours=period_hours)

    # Generate mock KPIs
    def create_kpi(name: str, value: float, unit: str, target: float) -> KPIValue:
        return KPIValue(
            name=name,
            value=value,
            unit=unit,
            target=target,
            trend="stable" if abs(value - target) < target * 0.05 else ("up" if value > target else "down"),
            change_percent=((value - target) / target) * 100
        )

    emissions = EmissionsKPIs(
        co2_emissions=create_kpi("CO2 Emissions", 245.5, "kg/h", 250.0),
        nox_emissions=create_kpi("NOx Emissions", 42.3, "ppm", 50.0),
        co_emissions=create_kpi("CO Emissions", 14.5, "ppm", 20.0),
        particulate_matter=create_kpi("Particulate Matter", 8.2, "mg/m3", 10.0),
        carbon_intensity=create_kpi("Carbon Intensity", 0.45, "kg CO2/kWh", 0.50)
    )

    efficiency = EfficiencyKPIs(
        thermal_efficiency=create_kpi("Thermal Efficiency", 92.5, "%", 95.0),
        combustion_efficiency=create_kpi("Combustion Efficiency", 94.2, "%", 95.0),
        fuel_utilization=create_kpi("Fuel Utilization", 89.5, "%", 90.0),
        heat_recovery=create_kpi("Heat Recovery", 78.3, "%", 80.0),
        overall_equipment_effectiveness=create_kpi("OEE", 87.5, "%", 90.0)
    )

    operational = OperationalKPIs(
        availability=create_kpi("Availability", 98.5, "%", 99.0),
        reliability=create_kpi("Reliability", 97.2, "%", 98.0),
        mean_time_between_failures=create_kpi("MTBF", 720, "hours", 750),
        mean_time_to_repair=create_kpi("MTTR", 2.5, "hours", 2.0),
        capacity_utilization=create_kpi("Capacity Utilization", 78.5, "%", 80.0)
    )

    return KPIDashboardResponse(
        unit_id=unit_id,
        period_start=period_start,
        period_end=period_end,
        emissions=emissions,
        efficiency=efficiency,
        operational=operational,
        overall_score=89.5,
        comparison_baseline="last_week",
        timestamp=datetime.utcnow()
    )


# ============================================================================
# Recommendation Endpoints
# ============================================================================

@router.get(
    "/units/{unit_id}/recommendations",
    response_model=List[RecommendationResponse],
    summary="Get active recommendations",
    description="Retrieve active optimization recommendations for a burner unit."
)
async def get_recommendations(
    request: Request,
    unit_id: str,
    status_filter: Optional[RecommendationStatus] = Query(None, description="Filter by status"),
    priority_filter: Optional[RecommendationPriority] = Query(None, description="Filter by priority"),
    limit: int = Query(20, ge=1, le=100, description="Maximum results"),
    current_user: User = Depends(require_operator())
) -> List[RecommendationResponse]:
    """Get active recommendations for a unit."""
    validate_unit_access(unit_id, current_user)

    # Generate mock recommendations
    recommendations = [
        RecommendationResponse(
            recommendation_id=f"rec-{uuid.uuid4().hex[:8]}",
            unit_id=unit_id,
            title="Optimize Air-Fuel Ratio",
            description="Analysis indicates excess air levels are above optimal range. Reducing excess air can improve efficiency.",
            priority=RecommendationPriority.HIGH,
            status=RecommendationStatus.PENDING,
            category="efficiency",
            actions=[
                RecommendationAction(
                    action_id="act-001",
                    description="Reduce excess air percentage",
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
            reasoning="Historical data analysis shows optimal O2 levels between 2.5-3.5%. Current levels averaging 4.2%.",
            model_version="v2.1.0",
            valid_until=datetime.utcnow() + timedelta(hours=24),
            created_at=datetime.utcnow()
        ),
        RecommendationResponse(
            recommendation_id=f"rec-{uuid.uuid4().hex[:8]}",
            unit_id=unit_id,
            title="Adjust Burner Timing",
            description="Slight timing adjustment can reduce NOx emissions while maintaining efficiency.",
            priority=RecommendationPriority.MEDIUM,
            status=RecommendationStatus.PENDING,
            category="emissions",
            actions=[
                RecommendationAction(
                    action_id="act-002",
                    description="Advance ignition timing",
                    parameter="ignition_timing_offset",
                    current_value=0.0,
                    recommended_value=-2.0,
                    unit="degrees",
                    auto_applicable=False
                )
            ],
            impact=RecommendationImpact(
                efficiency_improvement=0.5,
                emissions_reduction=8.5,
                cost_savings=125.0,
                confidence_level=0.85
            ),
            reasoning="NOx formation analysis suggests timing adjustment can reduce peak flame temperatures.",
            model_version="v2.1.0",
            valid_until=datetime.utcnow() + timedelta(hours=48),
            created_at=datetime.utcnow() - timedelta(hours=2)
        )
    ]

    # Apply filters
    if status_filter:
        recommendations = [r for r in recommendations if r.status == status_filter]
    if priority_filter:
        recommendations = [r for r in recommendations if r.priority == priority_filter]

    return recommendations[:limit]


@router.post(
    "/units/{unit_id}/recommendations/{rec_id}/accept",
    response_model=AcceptRecommendationResponse,
    summary="Accept recommendation",
    description="Accept and optionally auto-implement a recommendation."
)
async def accept_recommendation(
    request: Request,
    unit_id: str,
    rec_id: str,
    accept_request: AcceptRecommendationRequest,
    current_user: User = Depends(require_engineer())
) -> AcceptRecommendationResponse:
    """Accept a recommendation."""
    validate_unit_access(unit_id, current_user)

    # Validate safety check override
    if accept_request.override_safety_check and UserRole.ADMIN not in current_user.roles:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Only admins can override safety checks"
        )

    # Log audit
    await audit_logger.log(
        user=current_user,
        request=request,
        action="accept_recommendation",
        resource_type="recommendation",
        resource_id=rec_id,
        status_code=200,
        response_time_ms=0,
        details={
            "auto_implement": accept_request.auto_implement,
            "scheduled_time": str(accept_request.scheduled_time) if accept_request.scheduled_time else None,
            "notes": accept_request.notes
        }
    )

    implementation_status = "scheduled" if accept_request.scheduled_time else (
        "implementing" if accept_request.auto_implement else "pending_manual"
    )

    return AcceptRecommendationResponse(
        recommendation_id=rec_id,
        status=RecommendationStatus.ACCEPTED,
        implementation_status=implementation_status,
        scheduled_time=accept_request.scheduled_time,
        estimated_completion=accept_request.scheduled_time or (
            datetime.utcnow() + timedelta(minutes=5) if accept_request.auto_implement else None
        ),
        accepted_by=current_user.email,
        accepted_at=datetime.utcnow()
    )


# ============================================================================
# Optimization Status Endpoints
# ============================================================================

@router.get(
    "/units/{unit_id}/optimization/status",
    response_model=OptimizationStatusResponse,
    summary="Get optimization status",
    description="Retrieve the current status of the optimization engine for a unit."
)
async def get_optimization_status(
    request: Request,
    unit_id: str,
    current_user: User = Depends(require_operator())
) -> OptimizationStatusResponse:
    """Get optimization engine status."""
    validate_unit_access(unit_id, current_user)

    return OptimizationStatusResponse(
        unit_id=unit_id,
        state=OptimizationState.MONITORING,
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
# Mode Change Endpoints
# ============================================================================

@router.post(
    "/units/{unit_id}/mode",
    response_model=ModeChangeResponse,
    summary="Change operating mode",
    description="Change the operating mode of a burner unit."
)
async def change_mode(
    request: Request,
    unit_id: str,
    mode_request: ModeChangeRequest,
    current_user: User = Depends(require_engineer())
) -> ModeChangeResponse:
    """Change unit operating mode."""
    unit = validate_unit_access(unit_id, current_user)

    # Validate mode transition
    current_mode = unit["mode"]
    if current_mode == OperatingMode.EMERGENCY and mode_request.new_mode != OperatingMode.MAINTENANCE:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Emergency mode can only transition to maintenance mode"
        )

    # Log audit
    await audit_logger.log(
        user=current_user,
        request=request,
        action="change_mode",
        resource_type="unit",
        resource_id=unit_id,
        status_code=200,
        response_time_ms=0,
        details={
            "previous_mode": current_mode.value,
            "new_mode": mode_request.new_mode.value,
            "reason": mode_request.reason
        }
    )

    # Update mode (in production, persist to database)
    data_store.units[unit_id]["mode"] = mode_request.new_mode

    return ModeChangeResponse(
        unit_id=unit_id,
        previous_mode=current_mode,
        new_mode=mode_request.new_mode,
        status="completed" if not mode_request.scheduled_time else "scheduled",
        scheduled_time=mode_request.scheduled_time,
        estimated_completion=mode_request.scheduled_time or datetime.utcnow(),
        transition_steps=[
            "Validate safety conditions",
            "Adjust control parameters",
            "Verify stable operation"
        ],
        changed_by=current_user.email,
        changed_at=datetime.utcnow()
    )


# ============================================================================
# History Endpoints
# ============================================================================

@router.get(
    "/units/{unit_id}/history",
    response_model=HistoryResponse,
    summary="Get historical data",
    description="Retrieve historical metrics data for a burner unit."
)
async def get_history(
    request: Request,
    unit_id: str,
    start_time: datetime = Query(..., description="Start of time range"),
    end_time: datetime = Query(..., description="End of time range"),
    metrics: List[str] = Query(["all"], description="Metrics to retrieve"),
    resolution: str = Query("1h", description="Data resolution"),
    current_user: User = Depends(require_operator())
) -> HistoryResponse:
    """Get historical data for a unit."""
    validate_unit_access(unit_id, current_user)

    # Validate time range
    if end_time <= start_time:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="end_time must be after start_time"
        )

    max_range = timedelta(days=30)
    if end_time - start_time > max_range:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Time range cannot exceed {max_range.days} days"
        )

    # Generate mock historical data
    import random
    data_points = []
    resolution_minutes = {"1m": 1, "5m": 5, "15m": 15, "1h": 60, "1d": 1440}.get(resolution, 60)
    current_time = start_time

    while current_time <= end_time:
        data_points.append(HistoryDataPoint(
            timestamp=current_time,
            values={
                "efficiency": 94.0 + random.uniform(-2, 2),
                "firing_rate": 75.0 + random.uniform(-10, 10),
                "oxygen_level": 3.5 + random.uniform(-0.5, 0.5),
                "co_level": 15.0 + random.uniform(-5, 5),
                "nox_level": 45.0 + random.uniform(-5, 5)
            }
        ))
        current_time += timedelta(minutes=resolution_minutes)

    return HistoryResponse(
        unit_id=unit_id,
        start_time=start_time,
        end_time=end_time,
        resolution=resolution,
        metrics=["efficiency", "firing_rate", "oxygen_level", "co_level", "nox_level"],
        data_points=data_points,
        total_points=len(data_points),
        statistics={
            "efficiency": {"min": 92.0, "max": 96.0, "avg": 94.0, "std": 0.8},
            "firing_rate": {"min": 65.0, "max": 85.0, "avg": 75.0, "std": 5.0}
        }
    )


# ============================================================================
# Alert Endpoints
# ============================================================================

@router.get(
    "/units/{unit_id}/alerts",
    response_model=List[AlertResponse],
    summary="Get active alerts",
    description="Retrieve active alerts for a burner unit."
)
async def get_alerts(
    request: Request,
    unit_id: str,
    severity_filter: Optional[AlertSeverity] = Query(None, description="Filter by severity"),
    status_filter: Optional[AlertStatus] = Query(None, description="Filter by status"),
    limit: int = Query(50, ge=1, le=100, description="Maximum results"),
    current_user: User = Depends(require_operator())
) -> List[AlertResponse]:
    """Get active alerts for a unit."""
    validate_unit_access(unit_id, current_user)

    # Generate mock alerts
    alerts = [
        AlertResponse(
            alert_id=f"alert-{uuid.uuid4().hex[:8]}",
            unit_id=unit_id,
            severity=AlertSeverity.WARNING,
            status=AlertStatus.ACTIVE,
            title="High Flue Gas Temperature",
            description="Flue gas temperature exceeds optimal range (190C threshold)",
            source="temperature_monitor",
            metric_name="flue_gas_temp",
            metric_value=195.0,
            threshold=190.0,
            recommended_action="Check heat exchanger efficiency and reduce firing rate if needed",
            created_at=datetime.utcnow() - timedelta(minutes=30)
        ),
        AlertResponse(
            alert_id=f"alert-{uuid.uuid4().hex[:8]}",
            unit_id=unit_id,
            severity=AlertSeverity.INFO,
            status=AlertStatus.ACTIVE,
            title="Maintenance Due Soon",
            description="Scheduled maintenance is due in 5 days",
            source="maintenance_scheduler",
            recommended_action="Schedule maintenance window",
            created_at=datetime.utcnow() - timedelta(hours=2)
        )
    ]

    # Apply filters
    if severity_filter:
        alerts = [a for a in alerts if a.severity == severity_filter]
    if status_filter:
        alerts = [a for a in alerts if a.status == status_filter]

    return alerts[:limit]


@router.post(
    "/units/{unit_id}/alerts/{alert_id}/acknowledge",
    response_model=AlertResponse,
    summary="Acknowledge alert",
    description="Acknowledge an active alert."
)
async def acknowledge_alert(
    request: Request,
    unit_id: str,
    alert_id: str,
    ack_request: AlertAcknowledgeRequest,
    current_user: User = Depends(require_operator())
) -> AlertResponse:
    """Acknowledge an alert."""
    validate_unit_access(unit_id, current_user)

    # Log audit
    await audit_logger.log(
        user=current_user,
        request=request,
        action="acknowledge_alert",
        resource_type="alert",
        resource_id=alert_id,
        status_code=200,
        response_time_ms=0,
        details={"notes": ack_request.notes}
    )

    return AlertResponse(
        alert_id=alert_id,
        unit_id=unit_id,
        severity=AlertSeverity.WARNING,
        status=AlertStatus.ACKNOWLEDGED,
        title="High Flue Gas Temperature",
        description="Flue gas temperature exceeds optimal range",
        source="temperature_monitor",
        metric_name="flue_gas_temp",
        metric_value=195.0,
        threshold=190.0,
        recommended_action="Check heat exchanger efficiency",
        acknowledged_by=current_user.email,
        acknowledged_at=datetime.utcnow(),
        created_at=datetime.utcnow() - timedelta(minutes=30)
    )


# ============================================================================
# Health Check Endpoint
# ============================================================================

@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Health check",
    description="Check API health status and dependencies.",
    tags=["System"]
)
async def health_check() -> HealthResponse:
    """
    Health check endpoint for load balancers and monitoring.

    Returns:
        Health status of API and dependencies
    """
    import time
    start_time = getattr(health_check, '_start_time', None)
    if start_time is None:
        health_check._start_time = time.time()
        start_time = health_check._start_time

    uptime = time.time() - start_time

    # Check service health (in production, actually ping services)
    services = [
        ServiceHealth(
            name="database",
            status="healthy",
            latency_ms=5.2,
            last_check=datetime.utcnow(),
            message="PostgreSQL connection pool healthy"
        ),
        ServiceHealth(
            name="redis",
            status="healthy",
            latency_ms=1.1,
            last_check=datetime.utcnow(),
            message="Redis cache operational"
        ),
        ServiceHealth(
            name="optimization_engine",
            status="healthy",
            latency_ms=12.5,
            last_check=datetime.utcnow(),
            message="Optimization service responding"
        )
    ]

    # Determine overall status
    statuses = [s.status for s in services]
    if all(s == "healthy" for s in statuses):
        overall_status = "healthy"
    elif any(s == "unhealthy" for s in statuses):
        overall_status = "unhealthy"
    else:
        overall_status = "degraded"

    return HealthResponse(
        status=overall_status,
        version=settings.app_version,
        environment=settings.environment,
        uptime_seconds=uptime,
        services=services,
        timestamp=datetime.utcnow()
    )
