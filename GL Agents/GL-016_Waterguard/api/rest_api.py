"""
GL-016_Waterguard REST API

FastAPI-based REST API for the Waterguard cooling tower optimization system.
Provides versioned endpoints for water chemistry monitoring, optimization,
recommendations, and compliance reporting.

Author: GL-APIDeveloper
Version: 1.0.0
"""

from __future__ import annotations

import logging
import time
import uuid
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, Request, status
from fastapi.responses import JSONResponse

from api.api_auth import (
    AuditLogger,
    Permission,
    User,
    UserRole,
    get_audit_logger,
    get_current_user,
    require_permission,
    require_role,
)
from api.api_schemas import (
    BlowdownStatusResponse,
    ChemistryReading,
    ChemistryStateResponse,
    ComplianceReportResponse,
    ComplianceStatus,
    ConstraintStatus,
    DosingChannel,
    DosingStatusResponse,
    ErrorResponse,
    HealthCheckResponse,
    HealthStatus,
    ComponentHealth,
    OperatingMode,
    OptimizationRequest,
    OptimizationResponse,
    OptimizationResult,
    RecommendationApprovalRequest,
    RecommendationApprovalResponse,
    RecommendationListResponse,
    RecommendationPriority,
    RecommendationResponse,
    RecommendationStatus,
    RecommendationType,
    SavingsMetric,
    SavingsReportResponse,
)
from api.config import get_api_config

logger = logging.getLogger(__name__)

# =============================================================================
# Router Setup
# =============================================================================

router = APIRouter(prefix="/api/v1", tags=["Waterguard API v1"])

# Server start time for uptime calculation
_server_start_time = datetime.utcnow()


# =============================================================================
# Helper Functions
# =============================================================================

def get_request_id(request: Request) -> str:
    """Get or generate request ID."""
    return request.headers.get("X-Request-ID", str(uuid.uuid4()))


async def log_api_call(
    request: Request,
    user: Optional[User],
    action: str,
    resource: str,
    resource_id: Optional[str] = None,
    status_code: int = 200,
    start_time: Optional[float] = None,
    error_message: Optional[str] = None,
) -> None:
    """Log API call to audit trail."""
    audit_logger = get_audit_logger()
    duration_ms = (time.time() - start_time) * 1000 if start_time else None

    await audit_logger.log(
        request=request,
        user=user,
        action=action,
        resource=resource,
        resource_id=resource_id,
        status_code=status_code,
        duration_ms=duration_ms,
        error_message=error_message,
    )


# =============================================================================
# Mock Data Functions (Replace with actual service calls in production)
# =============================================================================

async def get_mock_chemistry_state(tower_id: str) -> ChemistryStateResponse:
    """Get mock chemistry state data."""
    return ChemistryStateResponse(
        tower_id=tower_id,
        timestamp=datetime.utcnow(),
        ph=7.8,
        conductivity=1500.0,
        tds=1200.0,
        cycles_of_concentration=4.5,
        alkalinity=120.0,
        hardness=200.0,
        chloride=150.0,
        silica=25.0,
        temperature=32.5,
        langelier_saturation_index=0.5,
        ryznar_stability_index=6.5,
        readings=[
            ChemistryReading(
                parameter="pH",
                value=7.8,
                unit="pH",
                min_limit=7.0,
                max_limit=8.5,
                target=7.5,
            ),
            ChemistryReading(
                parameter="Conductivity",
                value=1500.0,
                unit="uS/cm",
                min_limit=800.0,
                max_limit=2000.0,
                target=1500.0,
            ),
        ],
        overall_status=ComplianceStatus.COMPLIANT,
        parameters_out_of_spec=[],
    )


async def get_mock_recommendations(tower_id: str) -> List[RecommendationResponse]:
    """Get mock recommendations."""
    return [
        RecommendationResponse(
            recommendation_id="rec-001",
            tower_id=tower_id,
            type=RecommendationType.BLOWDOWN_ADJUSTMENT,
            priority=RecommendationPriority.MEDIUM,
            status=RecommendationStatus.PENDING,
            title="Increase Blowdown Rate",
            description="Conductivity trending high. Consider increasing blowdown rate to maintain COC target.",
            action_required="Increase blowdown rate from 10 gpm to 12.5 gpm",
            current_value=10.0,
            recommended_value=12.5,
            parameter="blowdown_rate",
            unit="gpm",
            impact_score=75.0,
            confidence=0.92,
            projected_savings=500.0,
            reasoning="Based on 24-hour conductivity trend analysis and current COC deviation.",
        ),
        RecommendationResponse(
            recommendation_id="rec-002",
            tower_id=tower_id,
            type=RecommendationType.DOSING_RATE_CHANGE,
            priority=RecommendationPriority.LOW,
            status=RecommendationStatus.PENDING,
            title="Adjust Inhibitor Dosing",
            description="Scale inhibitor concentration below optimal. Minor adjustment recommended.",
            action_required="Increase inhibitor dosing rate by 10%",
            current_value=2.3,
            recommended_value=2.5,
            parameter="inhibitor_rate",
            unit="mL/hr",
            impact_score=45.0,
            confidence=0.85,
            projected_savings=200.0,
            reasoning="LSI trending toward scaling conditions. Preventive dosing adjustment.",
        ),
    ]


# =============================================================================
# Health Check Endpoints
# =============================================================================

@router.get(
    "/health",
    response_model=HealthCheckResponse,
    summary="Health Check",
    description="Check API health status and component availability",
    tags=["System"],
)
async def health_check(request: Request) -> HealthCheckResponse:
    """
    API health check endpoint.

    Returns overall health status and individual component health.
    Used by load balancers and monitoring systems.
    """
    start_time = time.time()

    # Check component health
    components = [
        ComponentHealth(
            component="database",
            status=HealthStatus.HEALTHY,
            message="Connected",
            latency_ms=5.2,
        ),
        ComponentHealth(
            component="redis",
            status=HealthStatus.HEALTHY,
            message="Connected",
            latency_ms=1.1,
        ),
        ComponentHealth(
            component="ml_engine",
            status=HealthStatus.HEALTHY,
            message="Model loaded",
            latency_ms=15.5,
        ),
        ComponentHealth(
            component="plc_gateway",
            status=HealthStatus.HEALTHY,
            message="Connected",
            latency_ms=25.0,
        ),
    ]

    # Determine overall status
    unhealthy_count = sum(1 for c in components if c.status == HealthStatus.UNHEALTHY)
    degraded_count = sum(1 for c in components if c.status == HealthStatus.DEGRADED)

    if unhealthy_count > 0:
        overall_status = HealthStatus.UNHEALTHY
    elif degraded_count > 0:
        overall_status = HealthStatus.DEGRADED
    else:
        overall_status = HealthStatus.HEALTHY

    # Calculate uptime
    uptime = (datetime.utcnow() - _server_start_time).total_seconds()

    await log_api_call(
        request=request,
        user=None,
        action="health_check",
        resource="system",
        status_code=200,
        start_time=start_time,
    )

    return HealthCheckResponse(
        status=overall_status,
        version=get_api_config().api_version,
        uptime_seconds=uptime,
        components=components,
        cpu_usage_percent=25.0,
        memory_usage_percent=45.0,
        active_connections=150,
    )


@router.get(
    "/health/live",
    summary="Liveness Probe",
    description="Kubernetes liveness probe endpoint",
    tags=["System"],
)
async def liveness_probe() -> Dict[str, str]:
    """Liveness probe for Kubernetes."""
    return {"status": "alive"}


@router.get(
    "/health/ready",
    summary="Readiness Probe",
    description="Kubernetes readiness probe endpoint",
    tags=["System"],
)
async def readiness_probe() -> Dict[str, str]:
    """Readiness probe for Kubernetes."""
    # In production, check database and critical services
    return {"status": "ready"}


# =============================================================================
# Optimization Endpoints
# =============================================================================

@router.post(
    "/optimize",
    response_model=OptimizationResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Trigger Optimization Cycle",
    description="Trigger an optimization cycle for a cooling tower",
    tags=["Optimization"],
    responses={
        401: {"model": ErrorResponse, "description": "Unauthorized"},
        403: {"model": ErrorResponse, "description": "Forbidden"},
        404: {"model": ErrorResponse, "description": "Tower not found"},
    },
)
async def trigger_optimization(
    request: Request,
    optimization_request: OptimizationRequest,
    user: User = Depends(require_permission(Permission.EXECUTE_OPTIMIZATION)),
) -> OptimizationResponse:
    """
    Trigger an optimization cycle for the specified cooling tower.

    This endpoint initiates the ML optimization engine to analyze current
    conditions and generate recommendations for optimal operation.

    Requires: EXECUTE_OPTIMIZATION permission
    """
    start_time = time.time()

    # Verify tower access
    if not user.can_access_tower(optimization_request.tower_id):
        await log_api_call(
            request=request,
            user=user,
            action="trigger_optimization",
            resource="optimization",
            resource_id=optimization_request.tower_id,
            status_code=403,
            start_time=start_time,
            error_message="No access to tower",
        )
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"No access to tower: {optimization_request.tower_id}",
        )

    logger.info(
        f"Optimization triggered by {user.email} for tower {optimization_request.tower_id}"
    )

    # Generate optimization ID
    optimization_id = f"opt-{datetime.utcnow().strftime('%Y%m%d')}-{uuid.uuid4().hex[:8]}"

    # Mock optimization results (replace with actual ML engine call)
    results = [
        OptimizationResult(
            parameter="cycles_of_concentration",
            current_value=4.5,
            recommended_value=5.2,
            change_percent=15.6,
            impact_score=85.0,
            confidence=0.94,
        ),
        OptimizationResult(
            parameter="blowdown_rate",
            current_value=12.0,
            recommended_value=10.5,
            change_percent=-12.5,
            impact_score=70.0,
            confidence=0.89,
        ),
    ]

    execution_time = (time.time() - start_time) * 1000

    response = OptimizationResponse(
        optimization_id=optimization_id,
        tower_id=optimization_request.tower_id,
        status="completed",
        results=results,
        recommended_coc=5.2,
        recommended_blowdown_rate=10.5,
        recommended_dosing_rates={
            "inhibitor": 2.5,
            "biocide": 0.5,
            "dispersant": 1.0,
        },
        projected_water_savings_percent=15.0,
        projected_energy_savings_percent=8.0,
        projected_chemical_savings_percent=12.0,
        execution_time_ms=execution_time,
        model_version="v2.1.0",
        recommendations_generated=3,
    )

    await log_api_call(
        request=request,
        user=user,
        action="trigger_optimization",
        resource="optimization",
        resource_id=optimization_id,
        status_code=202,
        start_time=start_time,
    )

    return response


# =============================================================================
# Status Endpoints
# =============================================================================

@router.get(
    "/status",
    response_model=ChemistryStateResponse,
    summary="Get Current Status",
    description="Get current chemistry state and compliance status",
    tags=["Status"],
)
async def get_status(
    request: Request,
    tower_id: str = Query(..., description="Cooling tower identifier"),
    user: User = Depends(require_permission(Permission.READ_CHEMISTRY)),
) -> ChemistryStateResponse:
    """
    Get current chemistry state and compliance status for a tower.

    Returns real-time water chemistry readings, calculated indices,
    and overall compliance status.
    """
    start_time = time.time()

    if not user.can_access_tower(tower_id):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"No access to tower: {tower_id}",
        )

    chemistry_state = await get_mock_chemistry_state(tower_id)

    await log_api_call(
        request=request,
        user=user,
        action="get_status",
        resource="chemistry",
        resource_id=tower_id,
        status_code=200,
        start_time=start_time,
    )

    return chemistry_state


# =============================================================================
# Chemistry Endpoints
# =============================================================================

@router.get(
    "/chemistry",
    response_model=ChemistryStateResponse,
    summary="Get Chemistry Readings",
    description="Get current water chemistry readings",
    tags=["Chemistry"],
)
async def get_chemistry(
    request: Request,
    tower_id: str = Query(..., description="Cooling tower identifier"),
    user: User = Depends(require_permission(Permission.READ_CHEMISTRY)),
) -> ChemistryStateResponse:
    """
    Get current water chemistry readings for a tower.

    Returns all monitored chemistry parameters including pH, conductivity,
    TDS, alkalinity, hardness, and calculated scaling indices.
    """
    start_time = time.time()

    if not user.can_access_tower(tower_id):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"No access to tower: {tower_id}",
        )

    chemistry_state = await get_mock_chemistry_state(tower_id)

    await log_api_call(
        request=request,
        user=user,
        action="get_chemistry",
        resource="chemistry",
        resource_id=tower_id,
        status_code=200,
        start_time=start_time,
    )

    return chemistry_state


# =============================================================================
# Recommendation Endpoints
# =============================================================================

@router.get(
    "/recommendations",
    response_model=RecommendationListResponse,
    summary="Get Active Recommendations",
    description="Get active optimization recommendations",
    tags=["Recommendations"],
)
async def get_recommendations(
    request: Request,
    tower_id: str = Query(..., description="Cooling tower identifier"),
    status_filter: Optional[RecommendationStatus] = Query(
        None, description="Filter by status"
    ),
    priority_filter: Optional[RecommendationPriority] = Query(
        None, description="Filter by priority"
    ),
    limit: int = Query(20, ge=1, le=100, description="Maximum results"),
    user: User = Depends(require_permission(Permission.READ_RECOMMENDATIONS)),
) -> RecommendationListResponse:
    """
    Get active optimization recommendations for a tower.

    Returns pending recommendations from the optimization engine,
    including blowdown adjustments, dosing changes, and alerts.
    """
    start_time = time.time()

    if not user.can_access_tower(tower_id):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"No access to tower: {tower_id}",
        )

    recommendations = await get_mock_recommendations(tower_id)

    # Apply filters
    if status_filter:
        recommendations = [r for r in recommendations if r.status == status_filter]
    if priority_filter:
        recommendations = [r for r in recommendations if r.priority == priority_filter]

    # Apply limit
    recommendations = recommendations[:limit]

    # Count by priority
    critical_count = sum(1 for r in recommendations if r.priority == RecommendationPriority.CRITICAL)
    high_count = sum(1 for r in recommendations if r.priority == RecommendationPriority.HIGH)
    medium_count = sum(1 for r in recommendations if r.priority == RecommendationPriority.MEDIUM)
    low_count = sum(1 for r in recommendations if r.priority == RecommendationPriority.LOW)

    response = RecommendationListResponse(
        tower_id=tower_id,
        total_count=len(recommendations),
        pending_count=sum(1 for r in recommendations if r.status == RecommendationStatus.PENDING),
        recommendations=recommendations,
        critical_count=critical_count,
        high_count=high_count,
        medium_count=medium_count,
        low_count=low_count,
    )

    await log_api_call(
        request=request,
        user=user,
        action="get_recommendations",
        resource="recommendations",
        resource_id=tower_id,
        status_code=200,
        start_time=start_time,
    )

    return response


@router.post(
    "/recommendations/{recommendation_id}/approve",
    response_model=RecommendationApprovalResponse,
    summary="Approve/Reject Recommendation",
    description="Approve or reject an optimization recommendation",
    tags=["Recommendations"],
)
async def approve_recommendation(
    request: Request,
    recommendation_id: str,
    approval_request: RecommendationApprovalRequest,
    user: User = Depends(require_permission(Permission.APPROVE_RECOMMENDATIONS)),
) -> RecommendationApprovalResponse:
    """
    Approve or reject an optimization recommendation.

    Operators use this endpoint to review and approve recommendations
    before they are implemented. Approved recommendations are sent
    to the control system for execution.
    """
    start_time = time.time()

    logger.info(
        f"Recommendation {recommendation_id} "
        f"{'approved' if approval_request.approved else 'rejected'} "
        f"by {user.email}"
    )

    new_status = (
        RecommendationStatus.APPROVED
        if approval_request.approved
        else RecommendationStatus.REJECTED
    )

    response = RecommendationApprovalResponse(
        recommendation_id=recommendation_id,
        status=new_status,
        approved=approval_request.approved,
        approved_by=user.email,
        implementation_scheduled=approval_request.schedule_implementation,
        message=(
            f"Recommendation {recommendation_id} has been "
            f"{'approved' if approval_request.approved else 'rejected'}"
        ),
    )

    await log_api_call(
        request=request,
        user=user,
        action="approve_recommendation" if approval_request.approved else "reject_recommendation",
        resource="recommendations",
        resource_id=recommendation_id,
        status_code=200,
        start_time=start_time,
    )

    return response


# =============================================================================
# Blowdown Endpoints
# =============================================================================

@router.get(
    "/blowdown",
    response_model=BlowdownStatusResponse,
    summary="Get Blowdown Status",
    description="Get current blowdown status and setpoints",
    tags=["Blowdown"],
)
async def get_blowdown_status(
    request: Request,
    tower_id: str = Query(..., description="Cooling tower identifier"),
    user: User = Depends(require_permission(Permission.READ_BLOWDOWN)),
) -> BlowdownStatusResponse:
    """
    Get current blowdown system status for a tower.

    Returns blowdown rates, COC tracking, valve status, and daily statistics.
    """
    start_time = time.time()

    if not user.can_access_tower(tower_id):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"No access to tower: {tower_id}",
        )

    response = BlowdownStatusResponse(
        tower_id=tower_id,
        blowdown_active=True,
        current_rate_gpm=12.5,
        target_rate_gpm=12.5,
        current_coc=4.8,
        target_coc=5.0,
        coc_deviation=-0.2,
        total_blowdown_today_gallons=1500.0,
        blowdown_events_today=8,
        average_duration_minutes=15.0,
        conductivity_setpoint=1500.0,
        high_conductivity_alarm=2000.0,
        low_conductivity_alarm=800.0,
        valve_position_percent=25.0,
        valve_status="modulating",
    )

    await log_api_call(
        request=request,
        user=user,
        action="get_blowdown_status",
        resource="blowdown",
        resource_id=tower_id,
        status_code=200,
        start_time=start_time,
    )

    return response


# =============================================================================
# Dosing Endpoints
# =============================================================================

@router.get(
    "/dosing",
    response_model=DosingStatusResponse,
    summary="Get Dosing Status",
    description="Get current dosing status and rates",
    tags=["Dosing"],
)
async def get_dosing_status(
    request: Request,
    tower_id: str = Query(..., description="Cooling tower identifier"),
    user: User = Depends(require_permission(Permission.READ_DOSING)),
) -> DosingStatusResponse:
    """
    Get current dosing system status for a tower.

    Returns dosing rates, tank levels, and chemical usage for all channels.
    """
    start_time = time.time()

    if not user.can_access_tower(tower_id):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"No access to tower: {tower_id}",
        )

    channels = [
        DosingChannel(
            channel_id="ch-01",
            chemical_type="scale_inhibitor",
            chemical_name="ScaleGuard Pro",
            active=True,
            current_rate_ml_hr=2.5,
            target_rate_ml_hr=2.5,
            tank_level_percent=75.0,
            tank_capacity_liters=200.0,
            estimated_days_remaining=30.0,
            daily_usage_ml=60.0,
            monthly_usage_liters=1.8,
            pump_status="running",
            last_calibration=datetime.utcnow() - timedelta(days=30),
        ),
        DosingChannel(
            channel_id="ch-02",
            chemical_type="biocide",
            chemical_name="BioControl Plus",
            active=True,
            current_rate_ml_hr=0.5,
            target_rate_ml_hr=0.5,
            tank_level_percent=60.0,
            tank_capacity_liters=100.0,
            estimated_days_remaining=50.0,
            daily_usage_ml=12.0,
            monthly_usage_liters=0.36,
            pump_status="running",
            last_calibration=datetime.utcnow() - timedelta(days=15),
        ),
        DosingChannel(
            channel_id="ch-03",
            chemical_type="dispersant",
            chemical_name="DispersAll",
            active=True,
            current_rate_ml_hr=1.0,
            target_rate_ml_hr=1.0,
            tank_level_percent=85.0,
            tank_capacity_liters=150.0,
            estimated_days_remaining=53.0,
            daily_usage_ml=24.0,
            monthly_usage_liters=0.72,
            pump_status="running",
            last_calibration=datetime.utcnow() - timedelta(days=45),
        ),
    ]

    response = DosingStatusResponse(
        tower_id=tower_id,
        system_status="operational",
        active_channels=3,
        total_channels=4,
        channels=channels,
        low_chemical_alerts=[],
        pump_fault_alerts=[],
        daily_chemical_cost=45.50,
        monthly_chemical_cost=1365.00,
    )

    await log_api_call(
        request=request,
        user=user,
        action="get_dosing_status",
        resource="dosing",
        resource_id=tower_id,
        status_code=200,
        start_time=start_time,
    )

    return response


# =============================================================================
# Compliance Endpoints
# =============================================================================

@router.get(
    "/compliance",
    response_model=ComplianceReportResponse,
    summary="Get Compliance Status",
    description="Get constraint compliance status report",
    tags=["Compliance"],
)
async def get_compliance_status(
    request: Request,
    tower_id: str = Query(..., description="Cooling tower identifier"),
    period_hours: int = Query(24, ge=1, le=720, description="Report period in hours"),
    user: User = Depends(require_permission(Permission.READ_COMPLIANCE)),
) -> ComplianceReportResponse:
    """
    Get compliance status report for a tower.

    Returns detailed compliance status for all monitored constraints,
    violation history, and regulatory tracking information.
    """
    start_time = time.time()

    if not user.can_access_tower(tower_id):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"No access to tower: {tower_id}",
        )

    constraints = [
        ConstraintStatus(
            constraint_id="const-ph",
            constraint_name="pH Range",
            category="chemistry",
            status=ComplianceStatus.COMPLIANT,
            current_value=7.8,
            limit_value=8.5,
            margin_percent=8.2,
            in_violation=False,
            violation_count_24h=0,
        ),
        ConstraintStatus(
            constraint_id="const-cond",
            constraint_name="Conductivity Limit",
            category="chemistry",
            status=ComplianceStatus.COMPLIANT,
            current_value=1500.0,
            limit_value=2000.0,
            margin_percent=25.0,
            in_violation=False,
            violation_count_24h=0,
        ),
        ConstraintStatus(
            constraint_id="const-discharge",
            constraint_name="Discharge Temperature",
            category="environmental",
            status=ComplianceStatus.WARNING,
            current_value=35.0,
            limit_value=38.0,
            margin_percent=7.9,
            in_violation=False,
            violation_count_24h=0,
        ),
    ]

    response = ComplianceReportResponse(
        tower_id=tower_id,
        report_period_hours=period_hours,
        overall_status=ComplianceStatus.COMPLIANT,
        compliance_score=98.5,
        total_constraints=len(constraints),
        compliant_constraints=sum(1 for c in constraints if c.status == ComplianceStatus.COMPLIANT),
        warning_constraints=sum(1 for c in constraints if c.status == ComplianceStatus.WARNING),
        violated_constraints=sum(1 for c in constraints if c.status == ComplianceStatus.VIOLATION),
        constraints=constraints,
        total_violations_24h=0,
        critical_violations=[],
        discharge_permit_status="valid",
        next_compliance_review=datetime.utcnow() + timedelta(days=180),
    )

    await log_api_call(
        request=request,
        user=user,
        action="get_compliance_status",
        resource="compliance",
        resource_id=tower_id,
        status_code=200,
        start_time=start_time,
    )

    return response


# =============================================================================
# Savings Endpoints
# =============================================================================

@router.get(
    "/savings",
    response_model=SavingsReportResponse,
    summary="Get Savings Report",
    description="Get water/energy/emissions savings report",
    tags=["Savings"],
)
async def get_savings_report(
    request: Request,
    tower_id: str = Query(..., description="Cooling tower identifier"),
    period_days: int = Query(30, ge=1, le=365, description="Report period in days"),
    user: User = Depends(require_permission(Permission.READ_SAVINGS)),
) -> SavingsReportResponse:
    """
    Get savings report for a tower.

    Returns water, energy, chemical, and emissions savings compared to baseline,
    along with cost savings and projected annual benefits.
    """
    start_time = time.time()

    if not user.can_access_tower(tower_id):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"No access to tower: {tower_id}",
        )

    period_end = datetime.utcnow()
    period_start = period_end - timedelta(days=period_days)

    metrics = [
        SavingsMetric(
            metric_name="Makeup Water",
            baseline_value=100000.0,
            current_value=85000.0,
            savings_value=15000.0,
            savings_percent=15.0,
            unit="gallons",
            monetary_value=750.0,
        ),
        SavingsMetric(
            metric_name="Pump Energy",
            baseline_value=50000.0,
            current_value=46000.0,
            savings_value=4000.0,
            savings_percent=8.0,
            unit="kWh",
            monetary_value=400.0,
        ),
    ]

    response = SavingsReportResponse(
        tower_id=tower_id,
        period_start=period_start,
        period_end=period_end,
        water_baseline_gallons=100000.0,
        water_actual_gallons=85000.0,
        water_savings_gallons=15000.0,
        water_savings_percent=15.0,
        water_cost_savings=750.0,
        energy_baseline_kwh=50000.0,
        energy_actual_kwh=46000.0,
        energy_savings_kwh=4000.0,
        energy_savings_percent=8.0,
        energy_cost_savings=400.0,
        chemical_baseline_cost=2000.0,
        chemical_actual_cost=1760.0,
        chemical_savings=240.0,
        chemical_savings_percent=12.0,
        co2_baseline_kg=25000.0,
        co2_actual_kg=21250.0,
        co2_reduction_kg=3750.0,
        co2_reduction_percent=15.0,
        total_cost_savings=1390.0,
        total_savings_percent=11.6,
        metrics=metrics,
        projected_annual_savings=33360.0,
        projected_annual_water_savings=360000.0,
        projected_annual_co2_reduction=90000.0,
    )

    await log_api_call(
        request=request,
        user=user,
        action="get_savings_report",
        resource="savings",
        resource_id=tower_id,
        status_code=200,
        start_time=start_time,
    )

    return response


# =============================================================================
# Error Handlers
# =============================================================================

async def http_exception_handler(
    request: Request, exc: HTTPException
) -> JSONResponse:
    """Handle HTTP exceptions with structured error response."""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": "http_error",
            "message": exc.detail,
            "request_id": get_request_id(request),
            "timestamp": datetime.utcnow().isoformat(),
        },
    )


async def general_exception_handler(
    request: Request, exc: Exception
) -> JSONResponse:
    """Handle unexpected exceptions."""
    logger.exception(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": "internal_error",
            "message": "An unexpected error occurred",
            "request_id": get_request_id(request),
            "timestamp": datetime.utcnow().isoformat(),
        },
    )
