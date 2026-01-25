"""
GL-020 ECONOPULSE API Routes

API endpoint definitions for Economizer Performance Monitoring.
Implements all routes for performance, fouling, alerts, efficiency,
soot blower integration, and reporting.

Agent ID: GL-020
Codename: ECONOPULSE
"""

import logging
from datetime import datetime, timedelta
from typing import List, Optional
from uuid import uuid4

from fastapi import APIRouter, Depends, HTTPException, Query, Request, status
from slowapi import Limiter
from slowapi.util import get_remote_address

from .schemas import (
    # Economizer schemas
    Economizer,
    EconomizerList,
    EconomizerCreate,
    EconomizerUpdate,
    EconomizerStatus,
    EconomizerType,
    # Performance schemas
    PerformanceMetrics,
    PerformanceHistory,
    PerformanceTrend,
    TrendDirection,
    # Fouling schemas
    FoulingStatus,
    FoulingHistory,
    FoulingPrediction,
    FoulingBaseline,
    FoulingBaselineResponse,
    FoulingSeverity,
    # Alert schemas
    Alert,
    AlertList,
    AlertAcknowledge,
    AlertThresholdConfig,
    AlertThresholdConfigResponse,
    AlertSeverity,
    AlertStatus,
    AlertType,
    # Efficiency schemas
    EfficiencyMetrics,
    EfficiencyLoss,
    EfficiencySavings,
    # Soot blower schemas
    SootBlowerStatusResponse,
    SootBlower,
    SootBlowerTrigger,
    SootBlowerTriggerResponse,
    CleaningHistory,
    CleaningOptimization,
    SootBlowerStatus,
    # Report schemas
    DailyReport,
    WeeklyReport,
    EfficiencyReport,
    ReportExport,
    ReportExportResponse,
    # Error schemas
    ErrorResponse,
    NotFoundResponse,
)

logger = logging.getLogger("gl-020-econopulse.routes")

# Create router
router = APIRouter()

# Rate limiter
limiter = Limiter(key_func=get_remote_address)


# =============================================================================
# Mock Data (Replace with database/service calls in production)
# =============================================================================

MOCK_ECONOMIZERS = {
    "econ-001": Economizer(
        id="econ-001",
        name="Economizer Unit A1",
        description="Primary economizer for Boiler 1",
        type=EconomizerType.FINNED_TUBE,
        location="Building A, Level 2",
        boiler_id="boiler-001",
        design_capacity_kw=500.0,
        design_pressure_drop_kpa=1.5,
        surface_area_m2=150.0,
        status=EconomizerStatus.ONLINE,
        created_at=datetime(2025, 1, 15, 8, 0, 0),
        updated_at=datetime(2025, 11, 9, 10, 30, 0),
        last_cleaned=datetime(2025, 10, 15, 14, 0, 0),
        tags={"plant": "main", "zone": "north"}
    ),
    "econ-002": Economizer(
        id="econ-002",
        name="Economizer Unit B1",
        description="Primary economizer for Boiler 2",
        type=EconomizerType.BARE_TUBE,
        location="Building B, Level 1",
        boiler_id="boiler-002",
        design_capacity_kw=450.0,
        design_pressure_drop_kpa=1.3,
        surface_area_m2=130.0,
        status=EconomizerStatus.ONLINE,
        created_at=datetime(2025, 2, 1, 8, 0, 0),
        updated_at=datetime(2025, 11, 9, 10, 30, 0),
        last_cleaned=datetime(2025, 10, 20, 10, 0, 0),
        tags={"plant": "main", "zone": "south"}
    ),
}

MOCK_ALERTS = {
    "alert-001": Alert(
        id="alert-001",
        economizer_id="econ-001",
        alert_type=AlertType.FOULING,
        severity=AlertSeverity.WARNING,
        status=AlertStatus.ACTIVE,
        title="Elevated Fouling Detected",
        message="Fouling score has exceeded warning threshold of 50",
        metric_name="fouling_score",
        metric_value=55.2,
        threshold_value=50.0,
        triggered_at=datetime(2025, 11, 9, 8, 15, 0),
        acknowledged_at=None,
        acknowledged_by=None,
        resolved_at=None,
        resolved_by=None,
        resolution_notes=None
    ),
}


# =============================================================================
# Helper Functions
# =============================================================================

def get_economizer_or_404(economizer_id: str) -> Economizer:
    """Get economizer by ID or raise 404."""
    if economizer_id not in MOCK_ECONOMIZERS:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Economizer {economizer_id} not found"
        )
    return MOCK_ECONOMIZERS[economizer_id]


def get_alert_or_404(alert_id: str) -> Alert:
    """Get alert by ID or raise 404."""
    if alert_id not in MOCK_ALERTS:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Alert {alert_id} not found"
        )
    return MOCK_ALERTS[alert_id]


# =============================================================================
# Economizer Endpoints
# =============================================================================

@router.get(
    "/economizers",
    response_model=EconomizerList,
    tags=["Economizers"],
    summary="List all monitored economizers",
    description="Retrieve a paginated list of all monitored economizers with optional filtering.",
    responses={
        200: {"description": "List of economizers"},
        401: {"model": ErrorResponse, "description": "Unauthorized"},
    }
)
@limiter.limit("1000/minute")
async def list_economizers(
    request: Request,
    page: int = Query(1, ge=1, description="Page number"),
    page_size: int = Query(20, ge=1, le=100, description="Items per page"),
    status: Optional[EconomizerStatus] = Query(None, description="Filter by status"),
    boiler_id: Optional[str] = Query(None, description="Filter by boiler ID"),
) -> EconomizerList:
    """
    List all monitored economizers.

    Supports filtering by status and boiler ID with pagination.
    """
    logger.info(f"Listing economizers: page={page}, page_size={page_size}")

    # Filter economizers
    items = list(MOCK_ECONOMIZERS.values())

    if status:
        items = [e for e in items if e.status == status]
    if boiler_id:
        items = [e for e in items if e.boiler_id == boiler_id]

    total = len(items)

    # Paginate
    start = (page - 1) * page_size
    end = start + page_size
    items = items[start:end]

    return EconomizerList(
        items=items,
        total=total,
        page=page,
        page_size=page_size,
        has_next=end < total,
        has_prev=page > 1
    )


@router.get(
    "/economizers/{economizer_id}",
    response_model=Economizer,
    tags=["Economizers"],
    summary="Get economizer details",
    description="Retrieve detailed information about a specific economizer.",
    responses={
        200: {"description": "Economizer details"},
        404: {"model": NotFoundResponse, "description": "Economizer not found"},
    }
)
@limiter.limit("1000/minute")
async def get_economizer(
    request: Request,
    economizer_id: str,
) -> Economizer:
    """Get detailed information about a specific economizer."""
    logger.info(f"Getting economizer: {economizer_id}")
    return get_economizer_or_404(economizer_id)


# =============================================================================
# Performance Monitoring Endpoints
# =============================================================================

@router.get(
    "/economizers/{economizer_id}/performance",
    response_model=PerformanceMetrics,
    tags=["Performance"],
    summary="Get current performance metrics",
    description="Retrieve real-time performance metrics for an economizer.",
    responses={
        200: {"description": "Current performance metrics"},
        404: {"model": NotFoundResponse, "description": "Economizer not found"},
    }
)
@limiter.limit("1000/minute")
async def get_performance_metrics(
    request: Request,
    economizer_id: str,
) -> PerformanceMetrics:
    """Get current real-time performance metrics."""
    logger.info(f"Getting performance metrics for: {economizer_id}")
    get_economizer_or_404(economizer_id)

    # Return mock performance data
    return PerformanceMetrics(
        economizer_id=economizer_id,
        timestamp=datetime.utcnow(),
        gas_inlet_temp_c=320.5,
        gas_outlet_temp_c=180.2,
        water_inlet_temp_c=105.0,
        water_outlet_temp_c=140.5,
        gas_flow_rate_kg_s=15.2,
        water_flow_rate_kg_s=8.5,
        gas_pressure_drop_kpa=1.8,
        water_pressure_drop_kpa=0.5,
        heat_transfer_kw=485.3,
        effectiveness_percent=78.5,
        overall_htc_w_m2k=45.2,
        approach_temp_c=39.7,
        data_quality="good"
    )


@router.get(
    "/economizers/{economizer_id}/performance/history",
    response_model=PerformanceHistory,
    tags=["Performance"],
    summary="Get historical performance data",
    description="Retrieve historical performance data for trend analysis.",
    responses={
        200: {"description": "Historical performance data"},
        404: {"model": NotFoundResponse, "description": "Economizer not found"},
    }
)
@limiter.limit("1000/minute")
async def get_performance_history(
    request: Request,
    economizer_id: str,
    start_time: Optional[datetime] = Query(None, description="Start time"),
    end_time: Optional[datetime] = Query(None, description="End time"),
    resolution: str = Query("1h", description="Data resolution: 1m, 5m, 15m, 1h, 1d"),
) -> PerformanceHistory:
    """Get historical performance data for the specified time range."""
    logger.info(f"Getting performance history for: {economizer_id}")
    get_economizer_or_404(economizer_id)

    # Default time range
    if not end_time:
        end_time = datetime.utcnow()
    if not start_time:
        start_time = end_time - timedelta(days=1)

    # Generate mock data points
    data_points = []
    current = start_time
    while current <= end_time:
        data_points.append({
            "timestamp": current,
            "heat_transfer_kw": 480.0 + (current.hour % 24) * 2,
            "effectiveness_percent": 77.0 + (current.hour % 12) * 0.5,
            "overall_htc_w_m2k": 44.0 + (current.hour % 6) * 0.3,
            "gas_pressure_drop_kpa": 1.7 + (current.hour % 8) * 0.05
        })
        current += timedelta(hours=1)

    return PerformanceHistory(
        economizer_id=economizer_id,
        start_time=start_time,
        end_time=end_time,
        resolution=resolution,
        data_points=data_points[:100],  # Limit response size
        statistics={
            "avg_heat_transfer_kw": 485.2,
            "min_effectiveness_percent": 75.0,
            "max_effectiveness_percent": 82.0,
            "avg_pressure_drop_kpa": 1.78
        }
    )


@router.get(
    "/economizers/{economizer_id}/trends",
    response_model=PerformanceTrend,
    tags=["Performance"],
    summary="Get performance trends",
    description="Analyze performance trends and predict future degradation.",
    responses={
        200: {"description": "Performance trend analysis"},
        404: {"model": NotFoundResponse, "description": "Economizer not found"},
    }
)
@limiter.limit("1000/minute")
async def get_performance_trends(
    request: Request,
    economizer_id: str,
    analysis_period_days: int = Query(30, ge=1, le=365, description="Analysis period in days"),
) -> PerformanceTrend:
    """Analyze performance trends over the specified period."""
    logger.info(f"Getting performance trends for: {economizer_id}")
    get_economizer_or_404(economizer_id)

    return PerformanceTrend(
        economizer_id=economizer_id,
        analysis_period_days=analysis_period_days,
        analyzed_at=datetime.utcnow(),
        effectiveness_trend=TrendDirection.DEGRADING,
        htc_trend=TrendDirection.DEGRADING,
        pressure_drop_trend=TrendDirection.DEGRADING,
        effectiveness_change_percent=-5.2,
        htc_change_percent=-8.1,
        pressure_drop_change_percent=12.5,
        days_until_intervention=14,
        confidence_percent=85.0
    )


# =============================================================================
# Fouling Analysis Endpoints
# =============================================================================

@router.get(
    "/economizers/{economizer_id}/fouling",
    response_model=FoulingStatus,
    tags=["Fouling"],
    summary="Get current fouling status",
    description="Retrieve current fouling status and severity assessment.",
    responses={
        200: {"description": "Current fouling status"},
        404: {"model": NotFoundResponse, "description": "Economizer not found"},
    }
)
@limiter.limit("1000/minute")
async def get_fouling_status(
    request: Request,
    economizer_id: str,
) -> FoulingStatus:
    """Get current fouling status and assessment."""
    logger.info(f"Getting fouling status for: {economizer_id}")
    economizer = get_economizer_or_404(economizer_id)

    days_since_cleaning = None
    if economizer.last_cleaned:
        days_since_cleaning = (datetime.utcnow() - economizer.last_cleaned).days

    return FoulingStatus(
        economizer_id=economizer_id,
        timestamp=datetime.utcnow(),
        severity=FoulingSeverity.MODERATE,
        fouling_factor=0.00025,
        fouling_score=45.0,
        effectiveness_loss_percent=8.5,
        pressure_drop_increase_percent=15.2,
        estimated_deposit_thickness_mm=1.2,
        cleaning_recommended=False,
        last_cleaned=economizer.last_cleaned,
        days_since_cleaning=days_since_cleaning
    )


@router.get(
    "/economizers/{economizer_id}/fouling/history",
    response_model=FoulingHistory,
    tags=["Fouling"],
    summary="Get fouling history",
    description="Retrieve historical fouling data including cleaning events.",
    responses={
        200: {"description": "Fouling history data"},
        404: {"model": NotFoundResponse, "description": "Economizer not found"},
    }
)
@limiter.limit("1000/minute")
async def get_fouling_history(
    request: Request,
    economizer_id: str,
    start_time: Optional[datetime] = Query(None, description="Start time"),
    end_time: Optional[datetime] = Query(None, description="End time"),
) -> FoulingHistory:
    """Get historical fouling data for trend analysis."""
    logger.info(f"Getting fouling history for: {economizer_id}")
    get_economizer_or_404(economizer_id)

    # Default time range
    if not end_time:
        end_time = datetime.utcnow()
    if not start_time:
        start_time = end_time - timedelta(days=30)

    return FoulingHistory(
        economizer_id=economizer_id,
        start_time=start_time,
        end_time=end_time,
        data_points=[
            {
                "timestamp": start_time + timedelta(days=i),
                "fouling_factor": 0.0001 + i * 0.00001,
                "fouling_score": 10.0 + i * 1.2,
                "severity": FoulingSeverity.LIGHT if i < 15 else FoulingSeverity.MODERATE
            }
            for i in range(30)
        ],
        cleaning_events=[datetime(2025, 10, 15, 14, 0, 0)],
        average_fouling_rate=0.000008
    )


@router.get(
    "/economizers/{economizer_id}/fouling/prediction",
    response_model=FoulingPrediction,
    tags=["Fouling"],
    summary="Get fouling prediction",
    description="Predict future fouling levels based on historical trends.",
    responses={
        200: {"description": "Fouling prediction"},
        404: {"model": NotFoundResponse, "description": "Economizer not found"},
    }
)
@limiter.limit("1000/minute")
async def get_fouling_prediction(
    request: Request,
    economizer_id: str,
) -> FoulingPrediction:
    """Predict future fouling levels."""
    logger.info(f"Getting fouling prediction for: {economizer_id}")
    get_economizer_or_404(economizer_id)

    return FoulingPrediction(
        economizer_id=economizer_id,
        predicted_at=datetime.utcnow(),
        current_fouling_score=45.0,
        predicted_fouling_score_7d=52.0,
        predicted_fouling_score_14d=60.0,
        predicted_fouling_score_30d=75.0,
        days_until_cleaning_threshold=18,
        recommended_cleaning_date=datetime.utcnow() + timedelta(days=18),
        confidence_percent=82.5,
        model_version="v2.1.0"
    )


@router.post(
    "/economizers/{economizer_id}/fouling/baseline",
    response_model=FoulingBaselineResponse,
    status_code=status.HTTP_201_CREATED,
    tags=["Fouling"],
    summary="Set clean baseline",
    description="Set or update the clean baseline for fouling calculations.",
    responses={
        201: {"description": "Baseline created successfully"},
        404: {"model": NotFoundResponse, "description": "Economizer not found"},
    }
)
@limiter.limit("100/minute")
async def set_fouling_baseline(
    request: Request,
    economizer_id: str,
    baseline: FoulingBaseline,
) -> FoulingBaselineResponse:
    """Set clean baseline for fouling calculations."""
    logger.info(f"Setting fouling baseline for: {economizer_id}")
    get_economizer_or_404(economizer_id)

    return FoulingBaselineResponse(
        id=f"baseline-{uuid4().hex[:8]}",
        economizer_id=economizer_id,
        baseline_type=baseline.baseline_type,
        reference_date=baseline.reference_date or datetime.utcnow(),
        effectiveness_percent=baseline.effectiveness_percent,
        pressure_drop_kpa=baseline.pressure_drop_kpa,
        overall_htc_w_m2k=baseline.overall_htc_w_m2k,
        notes=baseline.notes,
        created_at=datetime.utcnow(),
        created_by="user-001"
    )


# =============================================================================
# Alert Endpoints
# =============================================================================

@router.get(
    "/alerts",
    response_model=AlertList,
    tags=["Alerts"],
    summary="List all active alerts",
    description="Retrieve all active alerts with optional filtering.",
    responses={
        200: {"description": "List of alerts"},
    }
)
@limiter.limit("1000/minute")
async def list_alerts(
    request: Request,
    page: int = Query(1, ge=1, description="Page number"),
    page_size: int = Query(20, ge=1, le=100, description="Items per page"),
    status: Optional[AlertStatus] = Query(None, description="Filter by status"),
    severity: Optional[AlertSeverity] = Query(None, description="Filter by severity"),
    alert_type: Optional[AlertType] = Query(None, description="Filter by type"),
    economizer_id: Optional[str] = Query(None, description="Filter by economizer"),
) -> AlertList:
    """List all alerts with filtering."""
    logger.info("Listing alerts")

    items = list(MOCK_ALERTS.values())

    if status:
        items = [a for a in items if a.status == status]
    if severity:
        items = [a for a in items if a.severity == severity]
    if alert_type:
        items = [a for a in items if a.alert_type == alert_type]
    if economizer_id:
        items = [a for a in items if a.economizer_id == economizer_id]

    total = len(items)
    active_count = sum(1 for a in items if a.status == AlertStatus.ACTIVE)

    # Paginate
    start = (page - 1) * page_size
    end = start + page_size
    items = items[start:end]

    return AlertList(
        items=items,
        total=total,
        active_count=active_count,
        page=page,
        page_size=page_size
    )


@router.get(
    "/alerts/{alert_id}",
    response_model=Alert,
    tags=["Alerts"],
    summary="Get alert details",
    description="Retrieve detailed information about a specific alert.",
    responses={
        200: {"description": "Alert details"},
        404: {"model": NotFoundResponse, "description": "Alert not found"},
    }
)
@limiter.limit("1000/minute")
async def get_alert(
    request: Request,
    alert_id: str,
) -> Alert:
    """Get detailed information about an alert."""
    logger.info(f"Getting alert: {alert_id}")
    return get_alert_or_404(alert_id)


@router.put(
    "/alerts/{alert_id}/acknowledge",
    response_model=Alert,
    tags=["Alerts"],
    summary="Acknowledge alert",
    description="Acknowledge an active alert.",
    responses={
        200: {"description": "Alert acknowledged"},
        404: {"model": NotFoundResponse, "description": "Alert not found"},
    }
)
@limiter.limit("100/minute")
async def acknowledge_alert(
    request: Request,
    alert_id: str,
    acknowledge: AlertAcknowledge,
) -> Alert:
    """Acknowledge an active alert."""
    logger.info(f"Acknowledging alert: {alert_id}")
    alert = get_alert_or_404(alert_id)

    # Update alert (mock)
    alert.status = AlertStatus.ACKNOWLEDGED
    alert.acknowledged_at = datetime.utcnow()
    alert.acknowledged_by = "user-001"

    return alert


@router.get(
    "/economizers/{economizer_id}/alerts",
    response_model=AlertList,
    tags=["Alerts"],
    summary="Get alerts for economizer",
    description="Retrieve all alerts for a specific economizer.",
    responses={
        200: {"description": "List of alerts for economizer"},
        404: {"model": NotFoundResponse, "description": "Economizer not found"},
    }
)
@limiter.limit("1000/minute")
async def get_economizer_alerts(
    request: Request,
    economizer_id: str,
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
) -> AlertList:
    """Get alerts for a specific economizer."""
    logger.info(f"Getting alerts for economizer: {economizer_id}")
    get_economizer_or_404(economizer_id)

    items = [a for a in MOCK_ALERTS.values() if a.economizer_id == economizer_id]
    total = len(items)
    active_count = sum(1 for a in items if a.status == AlertStatus.ACTIVE)

    return AlertList(
        items=items,
        total=total,
        active_count=active_count,
        page=page,
        page_size=page_size
    )


@router.post(
    "/alerts/thresholds",
    response_model=AlertThresholdConfigResponse,
    status_code=status.HTTP_201_CREATED,
    tags=["Alerts"],
    summary="Configure alert thresholds",
    description="Configure alert thresholds for monitoring.",
    responses={
        201: {"description": "Thresholds configured"},
    }
)
@limiter.limit("100/minute")
async def configure_alert_thresholds(
    request: Request,
    config: AlertThresholdConfig,
) -> AlertThresholdConfigResponse:
    """Configure alert thresholds."""
    logger.info(f"Configuring alert thresholds for: {config.economizer_id or 'global'}")

    return AlertThresholdConfigResponse(
        id=f"threshold-{uuid4().hex[:8]}",
        economizer_id=config.economizer_id,
        thresholds=config.thresholds,
        created_at=datetime.utcnow(),
        updated_at=datetime.utcnow()
    )


# =============================================================================
# Efficiency Analysis Endpoints
# =============================================================================

@router.get(
    "/economizers/{economizer_id}/efficiency",
    response_model=EfficiencyMetrics,
    tags=["Efficiency"],
    summary="Get efficiency metrics",
    description="Retrieve current efficiency metrics and comparison to design values.",
    responses={
        200: {"description": "Efficiency metrics"},
        404: {"model": NotFoundResponse, "description": "Economizer not found"},
    }
)
@limiter.limit("1000/minute")
async def get_efficiency_metrics(
    request: Request,
    economizer_id: str,
) -> EfficiencyMetrics:
    """Get current efficiency metrics."""
    logger.info(f"Getting efficiency metrics for: {economizer_id}")
    get_economizer_or_404(economizer_id)

    return EfficiencyMetrics(
        economizer_id=economizer_id,
        timestamp=datetime.utcnow(),
        current_efficiency_percent=78.5,
        design_efficiency_percent=85.0,
        clean_efficiency_percent=83.0,
        thermal_efficiency_percent=80.2,
        heat_recovery_percent=76.8,
        efficiency_vs_design_percent=-7.6,
        efficiency_vs_clean_percent=-5.4,
        heat_recovered_kw=485.0,
        potential_heat_recovery_kw=520.0,
        heat_loss_kw=35.0
    )


@router.get(
    "/economizers/{economizer_id}/efficiency/loss",
    response_model=EfficiencyLoss,
    tags=["Efficiency"],
    summary="Get efficiency loss quantification",
    description="Quantify efficiency losses and their impacts.",
    responses={
        200: {"description": "Efficiency loss analysis"},
        404: {"model": NotFoundResponse, "description": "Economizer not found"},
    }
)
@limiter.limit("1000/minute")
async def get_efficiency_loss(
    request: Request,
    economizer_id: str,
    start_time: Optional[datetime] = Query(None, description="Analysis start time"),
    end_time: Optional[datetime] = Query(None, description="Analysis end time"),
) -> EfficiencyLoss:
    """Quantify efficiency losses over the analysis period."""
    logger.info(f"Getting efficiency loss for: {economizer_id}")
    get_economizer_or_404(economizer_id)

    if not end_time:
        end_time = datetime.utcnow()
    if not start_time:
        start_time = end_time - timedelta(days=7)

    return EfficiencyLoss(
        economizer_id=economizer_id,
        analysis_period_start=start_time,
        analysis_period_end=end_time,
        total_efficiency_loss_percent=7.6,
        fouling_loss_percent=5.2,
        operational_loss_percent=1.8,
        other_loss_percent=0.6,
        energy_loss_kwh=8520.0,
        energy_loss_gj=30.7,
        fuel_cost_usd=425.0,
        carbon_emissions_kg=1850.0,
        loss_trend=TrendDirection.DEGRADING,
        loss_change_percent=1.2
    )


@router.get(
    "/economizers/{economizer_id}/efficiency/savings",
    response_model=EfficiencySavings,
    tags=["Efficiency"],
    summary="Get potential savings",
    description="Calculate potential savings from efficiency improvements.",
    responses={
        200: {"description": "Potential savings analysis"},
        404: {"model": NotFoundResponse, "description": "Economizer not found"},
    }
)
@limiter.limit("1000/minute")
async def get_efficiency_savings(
    request: Request,
    economizer_id: str,
) -> EfficiencySavings:
    """Calculate potential savings from cleaning/maintenance."""
    logger.info(f"Getting efficiency savings for: {economizer_id}")
    get_economizer_or_404(economizer_id)

    return EfficiencySavings(
        economizer_id=economizer_id,
        calculated_at=datetime.utcnow(),
        current_efficiency_percent=78.5,
        target_efficiency_percent=83.0,
        energy_savings_kwh_year=45600.0,
        energy_savings_gj_year=164.2,
        fuel_savings_usd_year=2280.0,
        carbon_reduction_kg_year=9880.0,
        estimated_cleaning_cost_usd=500.0,
        payback_period_days=80.0,
        roi_percent=356.0,
        cleaning_recommended=True,
        optimal_cleaning_date=datetime.utcnow() + timedelta(days=11),
        recommendation_notes="Cleaning recommended based on fouling level and economic analysis."
    )


# =============================================================================
# Soot Blower Integration Endpoints
# =============================================================================

@router.get(
    "/economizers/{economizer_id}/soot-blowers",
    response_model=SootBlowerStatusResponse,
    tags=["Soot Blowers"],
    summary="Get soot blower status",
    description="Retrieve status of all soot blowers for an economizer.",
    responses={
        200: {"description": "Soot blower status"},
        404: {"model": NotFoundResponse, "description": "Economizer not found"},
    }
)
@limiter.limit("1000/minute")
async def get_soot_blower_status(
    request: Request,
    economizer_id: str,
) -> SootBlowerStatusResponse:
    """Get status of soot blowers."""
    logger.info(f"Getting soot blower status for: {economizer_id}")
    get_economizer_or_404(economizer_id)

    soot_blowers = [
        SootBlower(
            id=f"sb-{i:03d}",
            name=f"Soot Blower {i}",
            type="steam",
            status=SootBlowerStatus.IDLE,
            economizer_id=economizer_id,
            position=["inlet", "middle-1", "middle-2", "outlet"][i-1],
            last_operated=datetime(2025, 11, 8, 14, 0, 0),
            operation_count=1250 + i * 50,
            steam_consumption_kg_cycle=15.5
        )
        for i in range(1, 5)
    ]

    return SootBlowerStatusResponse(
        economizer_id=economizer_id,
        timestamp=datetime.utcnow(),
        soot_blowers=soot_blowers,
        total_count=4,
        available_count=4,
        operating_count=0
    )


@router.post(
    "/economizers/{economizer_id}/soot-blowers/trigger",
    response_model=SootBlowerTriggerResponse,
    status_code=status.HTTP_202_ACCEPTED,
    tags=["Soot Blowers"],
    summary="Trigger cleaning",
    description="Trigger soot blower cleaning cycle.",
    responses={
        202: {"description": "Cleaning triggered"},
        404: {"model": NotFoundResponse, "description": "Economizer not found"},
    }
)
@limiter.limit("10/minute")
async def trigger_cleaning(
    request: Request,
    economizer_id: str,
    trigger: SootBlowerTrigger,
) -> SootBlowerTriggerResponse:
    """Trigger soot blower cleaning cycle."""
    logger.info(f"Triggering cleaning for: {economizer_id}")
    get_economizer_or_404(economizer_id)

    triggered_ids = trigger.soot_blower_ids or ["sb-001", "sb-002", "sb-003", "sb-004"]

    return SootBlowerTriggerResponse(
        operation_id=f"op-{uuid4().hex[:8]}",
        economizer_id=economizer_id,
        triggered_at=datetime.utcnow(),
        scheduled_start=datetime.utcnow() + timedelta(seconds=trigger.delay_seconds),
        soot_blowers_triggered=triggered_ids,
        sequence=trigger.sequence,
        estimated_duration_seconds=len(triggered_ids) * 180,
        status="scheduled"
    )


@router.get(
    "/economizers/{economizer_id}/cleaning-history",
    response_model=CleaningHistory,
    tags=["Soot Blowers"],
    summary="Get cleaning history",
    description="Retrieve cleaning event history.",
    responses={
        200: {"description": "Cleaning history"},
        404: {"model": NotFoundResponse, "description": "Economizer not found"},
    }
)
@limiter.limit("1000/minute")
async def get_cleaning_history(
    request: Request,
    economizer_id: str,
    start_time: Optional[datetime] = Query(None, description="Start time"),
    end_time: Optional[datetime] = Query(None, description="End time"),
) -> CleaningHistory:
    """Get cleaning event history."""
    logger.info(f"Getting cleaning history for: {economizer_id}")
    get_economizer_or_404(economizer_id)

    if not end_time:
        end_time = datetime.utcnow()
    if not start_time:
        start_time = end_time - timedelta(days=30)

    return CleaningHistory(
        economizer_id=economizer_id,
        start_time=start_time,
        end_time=end_time,
        events=[
            {
                "id": "clean-001",
                "economizer_id": economizer_id,
                "cleaning_type": "soot_blower",
                "started_at": datetime(2025, 10, 15, 14, 0, 0),
                "completed_at": datetime(2025, 10, 15, 14, 15, 0),
                "duration_seconds": 900,
                "soot_blowers_used": ["sb-001", "sb-002", "sb-003", "sb-004"],
                "effectiveness_before": 72.5,
                "effectiveness_after": 82.0,
                "fouling_score_before": 55.0,
                "fouling_score_after": 15.0,
                "success": True,
                "notes": "Standard cleaning cycle completed"
            }
        ],
        total_events=5,
        successful_events=5,
        average_effectiveness_improvement=8.5
    )


@router.get(
    "/economizers/{economizer_id}/cleaning/optimization",
    response_model=CleaningOptimization,
    tags=["Soot Blowers"],
    summary="Get optimal cleaning schedule",
    description="Get recommendations for optimal cleaning schedule.",
    responses={
        200: {"description": "Cleaning optimization recommendations"},
        404: {"model": NotFoundResponse, "description": "Economizer not found"},
    }
)
@limiter.limit("1000/minute")
async def get_cleaning_optimization(
    request: Request,
    economizer_id: str,
) -> CleaningOptimization:
    """Get optimal cleaning schedule recommendations."""
    logger.info(f"Getting cleaning optimization for: {economizer_id}")
    get_economizer_or_404(economizer_id)

    return CleaningOptimization(
        economizer_id=economizer_id,
        calculated_at=datetime.utcnow(),
        current_interval_hours=168.0,
        optimal_interval_hours=144.0,
        cleaning_efficiency_percent=85.0,
        over_cleaning_risk=False,
        under_cleaning_risk=True,
        next_recommended_cleaning=datetime.utcnow() + timedelta(days=6),
        recommended_sequence="intensive",
        recommended_soot_blowers=["sb-001", "sb-002", "sb-003", "sb-004"],
        current_annual_cleaning_cost_usd=12000.0,
        optimized_annual_cleaning_cost_usd=11000.0,
        annual_savings_usd=1000.0,
        optimization_notes=[
            "Increase cleaning frequency by 14% based on fouling rate analysis",
            "Use intensive sequence due to elevated fouling in inlet section",
            "Consider chemical cleaning if soot blower effectiveness drops below 70%"
        ]
    )


# =============================================================================
# Report Endpoints
# =============================================================================

@router.get(
    "/reports/daily",
    response_model=DailyReport,
    tags=["Reports"],
    summary="Get daily performance report",
    description="Retrieve daily performance report for all economizers.",
    responses={
        200: {"description": "Daily report"},
    }
)
@limiter.limit("100/minute")
async def get_daily_report(
    request: Request,
    report_date: Optional[datetime] = Query(None, description="Report date (default: yesterday)"),
) -> DailyReport:
    """Get daily performance report."""
    logger.info("Getting daily report")

    if not report_date:
        report_date = datetime.utcnow() - timedelta(days=1)

    return DailyReport(
        report_date=report_date,
        generated_at=datetime.utcnow(),
        report_period_start=report_date.replace(hour=0, minute=0, second=0),
        report_period_end=report_date.replace(hour=23, minute=59, second=59),
        total_economizers=len(MOCK_ECONOMIZERS),
        online_economizers=len(MOCK_ECONOMIZERS),
        total_heat_recovered_kwh=58500.0,
        average_effectiveness_percent=79.2,
        average_fouling_score=38.5,
        total_alerts=3,
        critical_alerts=0,
        acknowledged_alerts=2,
        cleaning_events=1,
        successful_cleanings=1,
        economizer_summaries=[
            {
                "economizer_id": e.id,
                "economizer_name": e.name,
                "avg_effectiveness_percent": 78.5,
                "avg_fouling_score": 45.0,
                "heat_recovered_kwh": 11700.0,
                "cleaning_events": 0,
                "alerts_triggered": 1,
                "status": e.status.value
            }
            for e in MOCK_ECONOMIZERS.values()
        ]
    )


@router.get(
    "/reports/weekly",
    response_model=WeeklyReport,
    tags=["Reports"],
    summary="Get weekly summary",
    description="Retrieve weekly summary report with trends.",
    responses={
        200: {"description": "Weekly report"},
    }
)
@limiter.limit("100/minute")
async def get_weekly_report(
    request: Request,
    week: Optional[str] = Query(None, description="ISO week (e.g., 2025-W45)"),
) -> WeeklyReport:
    """Get weekly summary report."""
    logger.info("Getting weekly report")

    now = datetime.utcnow()
    if not week:
        week = now.strftime("%Y-W%W")

    return WeeklyReport(
        report_week=week,
        generated_at=now,
        report_period_start=now - timedelta(days=7),
        report_period_end=now,
        effectiveness_trend=TrendDirection.STABLE,
        fouling_trend=TrendDirection.DEGRADING,
        effectiveness_change_percent=-0.5,
        total_heat_recovered_kwh=405000.0,
        energy_savings_vs_previous_week_kwh=2500.0,
        estimated_fuel_savings_usd=125.0,
        carbon_reduction_kg=542.0,
        total_cleaning_events=3,
        maintenance_hours=2.5,
        total_alerts=12,
        resolved_alerts=10,
        recommendations=[
            "Schedule cleaning for Economizer A1 within next 7 days",
            "Review alert thresholds for Economizer B2 - false positive rate high",
            "Consider chemical cleaning for Economizer C1 - soot blower effectiveness declining"
        ]
    )


@router.get(
    "/reports/efficiency",
    response_model=EfficiencyReport,
    tags=["Reports"],
    summary="Get efficiency report",
    description="Retrieve comprehensive efficiency analysis report.",
    responses={
        200: {"description": "Efficiency report"},
    }
)
@limiter.limit("100/minute")
async def get_efficiency_report(
    request: Request,
    start_time: Optional[datetime] = Query(None, description="Report start time"),
    end_time: Optional[datetime] = Query(None, description="Report end time"),
) -> EfficiencyReport:
    """Get efficiency analysis report."""
    logger.info("Getting efficiency report")

    now = datetime.utcnow()
    if not end_time:
        end_time = now
    if not start_time:
        start_time = end_time - timedelta(days=30)

    return EfficiencyReport(
        report_id=f"eff-rpt-{uuid4().hex[:8]}",
        generated_at=now,
        report_period_start=start_time,
        report_period_end=end_time,
        total_economizers=len(MOCK_ECONOMIZERS),
        fleet_average_efficiency_percent=78.5,
        fleet_design_efficiency_percent=85.0,
        total_energy_loss_kwh=125000.0,
        total_fuel_cost_impact_usd=6250.0,
        total_carbon_impact_kg=27125.0,
        total_savings_potential_usd_year=75000.0,
        total_carbon_reduction_potential_kg_year=162750.0,
        economizer_efficiency=[
            {
                "economizer_id": e.id,
                "name": e.name,
                "current_efficiency": 78.5,
                "design_efficiency": 85.0,
                "loss_kwh": 25000.0,
                "savings_potential_usd_year": 15000.0
            }
            for e in MOCK_ECONOMIZERS.values()
        ],
        priority_actions=[
            "Clean Economizer A1 - highest efficiency loss",
            "Investigate sensor calibration on Economizer B2",
            "Schedule preventive maintenance for Economizer C1"
        ]
    )


@router.post(
    "/reports/export",
    response_model=ReportExportResponse,
    status_code=status.HTTP_202_ACCEPTED,
    tags=["Reports"],
    summary="Export report",
    description="Export report in specified format (PDF, Excel, CSV).",
    responses={
        202: {"description": "Export job started"},
    }
)
@limiter.limit("10/minute")
async def export_report(
    request: Request,
    export_request: ReportExport,
) -> ReportExportResponse:
    """Export report in specified format."""
    logger.info(f"Exporting report: {export_request.report_type} as {export_request.format.value}")

    export_id = f"export-{uuid4().hex[:8]}"

    return ReportExportResponse(
        export_id=export_id,
        status="processing",
        requested_at=datetime.utcnow(),
        completed_at=None,
        download_url=None,
        file_size_bytes=None,
        expires_at=None
    )
