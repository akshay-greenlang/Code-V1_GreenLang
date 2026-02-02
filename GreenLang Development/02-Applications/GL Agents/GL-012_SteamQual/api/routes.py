"""
GL-012 SteamQual REST API Routes

FastAPI REST endpoints for steam quality control.
Provides HTTP endpoints for quality estimation, carryover risk assessment,
quality state monitoring, events, recommendations, and metrics.

Latency Targets:
- Sensor-to-metric: < 5 seconds
- Event emission: < 10 seconds
"""

from __future__ import annotations

import hashlib
import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional
from uuid import UUID, uuid4

from fastapi import APIRouter, Depends, HTTPException, Query, Request, status
from fastapi.responses import JSONResponse

from .auth import (
    Permission,
    SteamQualUser,
    get_current_user,
    require_permissions,
    require_header_access,
    log_api_call,
    log_security_event,
)
from .schemas import (
    # Quality Estimation
    QualityEstimateRequest,
    QualityEstimateResponse,
    QualityEstimate,
    QualityLevel,
    MeasurementSource,
    # Carryover Risk
    CarryoverRiskRequest,
    CarryoverRiskResponse,
    CarryoverRiskAssessment,
    CarryoverRiskLevel,
    CarryoverRiskFactors,
    # Quality State
    QualityStateResponse,
    QualityKPI,
    # Events
    EventsRequest,
    EventsResponse,
    QualityEvent,
    EventType,
    EventSeverity,
    # Recommendations
    RecommendationsRequest,
    RecommendationsResponse,
    QualityRecommendation,
    RecommendationType,
    RecommendationPriority,
    ControlAction,
    # Metrics
    MetricsRequest,
    MetricsResponse,
    QualityMetrics,
    # Common
    ErrorResponse,
)


logger = logging.getLogger(__name__)

# Create router
router = APIRouter(prefix="/api/v1", tags=["Steam Quality Controller"])


# =============================================================================
# Quality Estimation Endpoints
# =============================================================================

@router.post(
    "/estimate-quality",
    response_model=QualityEstimateResponse,
    status_code=status.HTTP_200_OK,
    summary="Estimate steam quality",
    description="Estimate steam quality (dryness fraction) from sensor measurements",
    tags=["Quality Estimation"],
)
async def estimate_quality(
    request: Request,
    quality_request: QualityEstimateRequest,
    current_user: SteamQualUser = Depends(
        require_permissions(Permission.QUALITY_ESTIMATE)
    ),
) -> QualityEstimateResponse:
    """
    Estimate steam quality from measurements.

    This endpoint calculates steam quality (dryness fraction) using available
    sensor data. The estimation uses a zero-hallucination approach with
    deterministic thermodynamic calculations.

    Latency target: < 5 seconds sensor-to-metric
    """
    start_time = datetime.utcnow()
    measurements = quality_request.measurements

    logger.info(
        f"Quality estimation request for header {measurements.header_id} "
        f"from {current_user.username}"
    )

    try:
        # Compute quality estimate (deterministic calculation)
        quality_estimate = _calculate_quality_estimate(
            measurements=measurements,
            method=quality_request.method,
            include_uncertainty=quality_request.include_uncertainty,
            include_trend=quality_request.include_trend,
        )

        # Calculate deviation from target
        deviation = quality_request.target_quality_percent - quality_estimate.quality_percent
        is_below_alarm = quality_estimate.quality_percent < quality_request.alarm_threshold_percent

        # Calculate provenance hash for audit trail
        provenance_data = f"{quality_request.request_id}{measurements.model_dump_json()}{quality_estimate.model_dump_json()}"
        provenance_hash = hashlib.sha256(provenance_data.encode()).hexdigest()

        # Calculate processing time
        processing_time = (datetime.utcnow() - start_time).total_seconds() * 1000

        # Log API call for audit
        await log_api_call(
            request=request,
            user=current_user,
            action="estimate_quality",
            resource_type="quality",
            resource_id=measurements.header_id,
            success=True,
            details={"quality_percent": quality_estimate.quality_percent},
            latency_ms=processing_time,
        )

        return QualityEstimateResponse(
            request_id=quality_request.request_id,
            header_id=measurements.header_id,
            success=True,
            estimate=quality_estimate,
            target_quality_percent=quality_request.target_quality_percent,
            deviation_from_target=deviation,
            is_below_alarm_threshold=is_below_alarm,
            processing_time_ms=processing_time,
            sensor_timestamp=measurements.timestamp,
            provenance_hash=provenance_hash,
        )

    except ValueError as e:
        logger.warning(f"Invalid quality estimation input: {e}")
        processing_time = (datetime.utcnow() - start_time).total_seconds() * 1000
        return QualityEstimateResponse(
            request_id=quality_request.request_id,
            header_id=measurements.header_id,
            success=False,
            target_quality_percent=quality_request.target_quality_percent,
            processing_time_ms=processing_time,
            sensor_timestamp=measurements.timestamp,
            provenance_hash="",
            error_message=str(e),
        )

    except Exception as e:
        logger.error(f"Quality estimation failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Quality estimation failed",
        )


# =============================================================================
# Carryover Risk Assessment Endpoints
# =============================================================================

@router.post(
    "/assess-carryover-risk",
    response_model=CarryoverRiskResponse,
    status_code=status.HTTP_200_OK,
    summary="Assess carryover risk",
    description="Assess the risk of moisture carryover in steam headers",
    tags=["Carryover Risk"],
)
async def assess_carryover_risk(
    request: Request,
    risk_request: CarryoverRiskRequest,
    current_user: SteamQualUser = Depends(
        require_permissions(Permission.CARRYOVER_ASSESS)
    ),
) -> CarryoverRiskResponse:
    """
    Assess moisture carryover risk.

    Evaluates the probability and severity of moisture carryover based on
    current operating conditions, load changes, and separator efficiency.

    Returns risk factors, predicted impact, and time to threshold.
    """
    start_time = datetime.utcnow()

    logger.info(
        f"Carryover risk assessment for header {risk_request.header_id} "
        f"from {current_user.username}"
    )

    try:
        # Calculate carryover risk (deterministic assessment)
        assessment = _assess_carryover_risk(risk_request)

        # Check threshold
        exceeds_threshold = _risk_exceeds_threshold(
            assessment.risk_level,
            risk_request.risk_threshold
        )

        # Calculate provenance hash
        provenance_data = f"{risk_request.request_id}{risk_request.model_dump_json()}{assessment.model_dump_json()}"
        provenance_hash = hashlib.sha256(provenance_data.encode()).hexdigest()

        processing_time = (datetime.utcnow() - start_time).total_seconds() * 1000

        await log_api_call(
            request=request,
            user=current_user,
            action="assess_carryover_risk",
            resource_type="carryover",
            resource_id=risk_request.header_id,
            success=True,
            details={
                "risk_level": assessment.risk_level.value,
                "risk_probability": assessment.risk_probability,
            },
            latency_ms=processing_time,
        )

        return CarryoverRiskResponse(
            request_id=risk_request.request_id,
            header_id=risk_request.header_id,
            success=True,
            assessment=assessment,
            exceeds_threshold=exceeds_threshold,
            processing_time_ms=processing_time,
            provenance_hash=provenance_hash,
        )

    except Exception as e:
        logger.error(f"Carryover risk assessment failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Carryover risk assessment failed",
        )


# =============================================================================
# Quality State Endpoints
# =============================================================================

@router.get(
    "/quality-state/{header_id}",
    response_model=QualityStateResponse,
    status_code=status.HTTP_200_OK,
    summary="Get quality state",
    description="Get current quality state for a steam header",
    tags=["Quality State"],
)
async def get_quality_state(
    request: Request,
    header_id: str,
    current_user: SteamQualUser = Depends(
        require_permissions(Permission.STATE_READ)
    ),
) -> QualityStateResponse:
    """
    Get current quality state for a steam header.

    Returns current quality, operating conditions, statistics, KPIs,
    and active alarms for the specified header.
    """
    start_time = datetime.utcnow()

    logger.info(f"Quality state request for header {header_id} from {current_user.username}")

    try:
        # Check header access
        if not current_user.can_access_header(header_id):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Not authorized to access header {header_id}",
            )

        # Get current quality state (mock implementation)
        state = _get_quality_state(header_id)

        await log_api_call(
            request=request,
            user=current_user,
            action="get_quality_state",
            resource_type="state",
            resource_id=header_id,
            success=True,
            latency_ms=(datetime.utcnow() - start_time).total_seconds() * 1000,
        )

        return state

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get quality state: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get quality state",
        )


# =============================================================================
# Events Endpoints
# =============================================================================

@router.get(
    "/events",
    response_model=EventsResponse,
    status_code=status.HTTP_200_OK,
    summary="Get quality events",
    description="Get quality events with optional filtering",
    tags=["Events"],
)
async def get_events(
    request: Request,
    header_id: Optional[str] = Query(default=None, description="Filter by header"),
    start_time: Optional[datetime] = Query(default=None, description="Start time filter"),
    end_time: Optional[datetime] = Query(default=None, description="End time filter"),
    event_type: Optional[List[EventType]] = Query(default=None, description="Filter by event type"),
    severity: Optional[List[EventSeverity]] = Query(default=None, description="Filter by severity"),
    active_only: bool = Query(default=False, description="Only active events"),
    limit: int = Query(default=100, ge=1, le=1000, description="Maximum results"),
    offset: int = Query(default=0, ge=0, description="Pagination offset"),
    current_user: SteamQualUser = Depends(
        require_permissions(Permission.EVENTS_READ)
    ),
) -> EventsResponse:
    """
    Get quality events with filtering.

    Supports filtering by header, time range, event type, severity, and status.
    Results are paginated.

    Latency target: < 10 seconds for event emission
    """
    request_start = datetime.utcnow()

    logger.info(f"Events request from {current_user.username}")

    try:
        # Build events request
        events_request = EventsRequest(
            header_id=header_id,
            start_time=start_time,
            end_time=end_time,
            event_types=event_type,
            severity_filter=severity,
            active_only=active_only,
            limit=limit,
            offset=offset,
        )

        # Get events (mock implementation)
        events = _get_quality_events(events_request)

        processing_time = (datetime.utcnow() - request_start).total_seconds() * 1000

        return EventsResponse(
            success=True,
            events=events,
            total_count=len(events),
            active_count=sum(1 for e in events if e.is_active),
            limit=limit,
            offset=offset,
            has_more=len(events) == limit,
            processing_time_ms=processing_time,
        )

    except Exception as e:
        logger.error(f"Failed to get events: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get events",
        )


@router.post(
    "/events/{event_id}/acknowledge",
    response_model=Dict[str, Any],
    status_code=status.HTTP_200_OK,
    summary="Acknowledge event",
    description="Acknowledge a quality event",
    tags=["Events"],
)
async def acknowledge_event(
    request: Request,
    event_id: UUID,
    notes: Optional[str] = Query(default=None, description="Acknowledgment notes"),
    current_user: SteamQualUser = Depends(
        require_permissions(Permission.EVENTS_ACKNOWLEDGE)
    ),
) -> Dict[str, Any]:
    """
    Acknowledge a quality event.

    Marks the event as acknowledged by the current user with optional notes.
    """
    logger.info(f"Event acknowledgment: {event_id} by {current_user.username}")

    try:
        # Acknowledge event (mock implementation)
        await log_security_event(
            event_type="operation",
            action="acknowledge",
            resource_type="event",
            request=request,
            user=current_user,
            resource_id=str(event_id),
            success=True,
            details={"notes": notes},
        )

        return {
            "event_id": str(event_id),
            "acknowledged": True,
            "acknowledged_by": current_user.username,
            "acknowledged_at": datetime.utcnow().isoformat(),
            "notes": notes,
        }

    except Exception as e:
        logger.error(f"Failed to acknowledge event: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to acknowledge event",
        )


# =============================================================================
# Recommendations Endpoints
# =============================================================================

@router.post(
    "/recommendations",
    response_model=RecommendationsResponse,
    status_code=status.HTTP_200_OK,
    summary="Get control recommendations",
    description="Get control recommendations for quality improvement",
    tags=["Recommendations"],
)
async def get_recommendations(
    request: Request,
    rec_request: RecommendationsRequest,
    current_user: SteamQualUser = Depends(
        require_permissions(Permission.RECOMMENDATIONS_READ)
    ),
) -> RecommendationsResponse:
    """
    Get control recommendations for quality improvement.

    Analyzes current quality state and operating conditions to generate
    prioritized recommendations for improving steam quality.

    Returns actionable recommendations with expected impact and confidence.
    """
    start_time = datetime.utcnow()

    logger.info(
        f"Recommendations request for header {rec_request.header_id} "
        f"from {current_user.username}"
    )

    try:
        # Generate recommendations (mock implementation)
        recommendations = _generate_recommendations(rec_request)

        # Calculate aggregates
        total_improvement = sum(
            r.expected_quality_improvement_percent or 0
            for r in recommendations
        )
        total_energy_savings = sum(
            r.estimated_energy_savings_kw or 0
            for r in recommendations
        )

        # Calculate provenance hash
        provenance_data = f"{rec_request.request_id}{rec_request.model_dump_json()}"
        for r in recommendations:
            provenance_data += r.model_dump_json()
        provenance_hash = hashlib.sha256(provenance_data.encode()).hexdigest()

        processing_time = (datetime.utcnow() - start_time).total_seconds() * 1000

        await log_api_call(
            request=request,
            user=current_user,
            action="get_recommendations",
            resource_type="recommendations",
            resource_id=rec_request.header_id,
            success=True,
            details={"recommendation_count": len(recommendations)},
            latency_ms=processing_time,
        )

        return RecommendationsResponse(
            request_id=rec_request.request_id,
            header_id=rec_request.header_id,
            success=True,
            recommendations=recommendations,
            total_recommendations=len(recommendations),
            critical_count=sum(1 for r in recommendations if r.priority == RecommendationPriority.CRITICAL),
            high_priority_count=sum(1 for r in recommendations if r.priority == RecommendationPriority.HIGH),
            total_expected_improvement_percent=total_improvement if total_improvement > 0 else None,
            total_expected_energy_savings_kw=total_energy_savings if total_energy_savings > 0 else None,
            processing_time_ms=processing_time,
            provenance_hash=provenance_hash,
        )

    except Exception as e:
        logger.error(f"Failed to generate recommendations: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to generate recommendations",
        )


@router.post(
    "/recommendations/{recommendation_id}/implement",
    response_model=Dict[str, Any],
    status_code=status.HTTP_200_OK,
    summary="Implement recommendation",
    description="Mark a recommendation as implemented",
    tags=["Recommendations"],
)
async def implement_recommendation(
    request: Request,
    recommendation_id: UUID,
    notes: Optional[str] = Query(default=None, description="Implementation notes"),
    current_user: SteamQualUser = Depends(
        require_permissions(Permission.RECOMMENDATIONS_IMPLEMENT)
    ),
) -> Dict[str, Any]:
    """
    Mark a recommendation as implemented.

    Records that the recommendation has been implemented by the current user.
    """
    logger.info(f"Recommendation implementation: {recommendation_id} by {current_user.username}")

    try:
        await log_security_event(
            event_type="operation",
            action="implement",
            resource_type="recommendation",
            request=request,
            user=current_user,
            resource_id=str(recommendation_id),
            success=True,
            details={"notes": notes},
        )

        return {
            "recommendation_id": str(recommendation_id),
            "implemented": True,
            "implemented_by": current_user.username,
            "implemented_at": datetime.utcnow().isoformat(),
            "notes": notes,
        }

    except Exception as e:
        logger.error(f"Failed to implement recommendation: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to implement recommendation",
        )


# =============================================================================
# Metrics Endpoints
# =============================================================================

@router.get(
    "/metrics",
    response_model=MetricsResponse,
    status_code=status.HTTP_200_OK,
    summary="Get quality metrics",
    description="Get quality KPIs and metrics",
    tags=["Metrics"],
)
async def get_metrics(
    request: Request,
    header_id: Optional[str] = Query(default=None, description="Filter by header"),
    period_hours: int = Query(default=24, ge=1, le=720, description="Period in hours"),
    aggregation: str = Query(default="hourly", pattern="^(hourly|daily|weekly)$"),
    current_user: SteamQualUser = Depends(
        require_permissions(Permission.METRICS_READ)
    ),
) -> MetricsResponse:
    """
    Get quality metrics and KPIs.

    Returns quality statistics, target compliance, carryover metrics,
    and energy impact for the specified period.
    """
    start_time = datetime.utcnow()

    logger.info(f"Metrics request from {current_user.username}")

    try:
        metrics_request = MetricsRequest(
            header_id=header_id,
            period_hours=period_hours,
            aggregation=aggregation,
        )

        # Get metrics (mock implementation)
        metrics = _get_quality_metrics(metrics_request)

        # Calculate summary
        if metrics:
            overall_avg = sum(m.average_quality_percent for m in metrics) / len(metrics)
            overall_on_target = sum(m.time_on_target_percent for m in metrics) / len(metrics)
            total_events = sum(m.carryover_event_count for m in metrics)
        else:
            overall_avg = None
            overall_on_target = None
            total_events = 0

        processing_time = (datetime.utcnow() - start_time).total_seconds() * 1000

        return MetricsResponse(
            success=True,
            metrics=metrics,
            overall_average_quality=overall_avg,
            overall_time_on_target_percent=overall_on_target,
            total_carryover_events=total_events,
            processing_time_ms=processing_time,
        )

    except Exception as e:
        logger.error(f"Failed to get metrics: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get metrics",
        )


# =============================================================================
# Helper Functions - Zero Hallucination Calculations
# =============================================================================

def _calculate_quality_estimate(
    measurements,
    method: MeasurementSource,
    include_uncertainty: bool,
    include_trend: bool,
) -> QualityEstimate:
    """
    Calculate steam quality estimate from measurements.

    Uses deterministic thermodynamic calculations - NO LLM/ML for numeric values.
    """
    # Method 1: Conductivity-based estimation
    if measurements.steam_conductivity_us_cm and measurements.feedwater_conductivity_us_cm:
        # TDS carryover calculation: steam_TDS / boiler_TDS
        if measurements.blowdown_conductivity_us_cm:
            carryover_ratio = measurements.steam_conductivity_us_cm / measurements.blowdown_conductivity_us_cm
            moisture_fraction = carryover_ratio  # Simplified model
            quality_percent = max(0, min(100, (1 - moisture_fraction) * 100))
        else:
            quality_percent = 99.0  # Default high quality if no blowdown data
    # Method 2: Temperature-based estimation
    elif measurements.superheat_c is not None:
        if measurements.superheat_c > 5.0:
            # Superheated steam - quality is 100%
            quality_percent = 100.0
        elif measurements.superheat_c > 0:
            # Slight superheat
            quality_percent = 99.5 + (measurements.superheat_c / 10.0)
            quality_percent = min(100.0, quality_percent)
        else:
            # Saturated or wet - estimate from temperature depression
            quality_percent = 98.0 + measurements.superheat_c
            quality_percent = max(85.0, min(100.0, quality_percent))
    # Method 3: Separator-based estimation
    elif measurements.separator_dp_kpa is not None:
        # Higher DP suggests more moisture removal
        base_quality = 97.0
        dp_factor = min(measurements.separator_dp_kpa / 10.0, 2.0)
        quality_percent = base_quality + dp_factor
    else:
        # Default estimation based on operating conditions
        if measurements.boiler_load_percent:
            if measurements.boiler_load_percent > 90:
                quality_percent = 97.0  # Higher load can reduce quality
            elif measurements.boiler_load_percent < 50:
                quality_percent = 98.5  # Lower load generally better quality
            else:
                quality_percent = 98.0
        else:
            quality_percent = 98.0

    # Determine quality level
    if quality_percent >= 99.5:
        quality_level = QualityLevel.EXCELLENT
    elif quality_percent >= 98.0:
        quality_level = QualityLevel.GOOD
    elif quality_percent >= 95.0:
        quality_level = QualityLevel.ACCEPTABLE
    elif quality_percent >= 90.0:
        quality_level = QualityLevel.MARGINAL
    else:
        quality_level = QualityLevel.POOR

    # Calculate uncertainty bounds if requested
    lower_bound = None
    upper_bound = None
    if include_uncertainty:
        # Uncertainty depends on measurement availability
        uncertainty = 1.5  # Default +/- 1.5%
        if measurements.steam_conductivity_us_cm:
            uncertainty = 0.5  # Better with conductivity
        elif measurements.superheat_c is not None:
            uncertainty = 1.0  # Moderate with superheat
        lower_bound = max(0, quality_percent - uncertainty)
        upper_bound = min(100, quality_percent + uncertainty)

    # Calculate confidence score
    measurement_count = sum([
        measurements.pressure_kpa is not None,
        measurements.temperature_c is not None,
        measurements.steam_conductivity_us_cm is not None,
        measurements.superheat_c is not None,
        measurements.separator_dp_kpa is not None,
        measurements.drum_level_percent is not None,
    ])
    confidence_score = min(1.0, 0.5 + (measurement_count * 0.1))

    return QualityEstimate(
        quality_percent=quality_percent,
        quality_level=quality_level,
        quality_lower_bound=lower_bound,
        quality_upper_bound=upper_bound,
        confidence_score=confidence_score,
        moisture_content_percent=100 - quality_percent,
        specific_enthalpy_kj_kg=None,  # Would calculate from steam tables
        trend_direction="stable" if include_trend else None,
        trend_rate_percent_hour=0.0 if include_trend else None,
        estimation_method=method,
        measurement_count=measurement_count,
    )


def _assess_carryover_risk(risk_request: CarryoverRiskRequest) -> CarryoverRiskAssessment:
    """
    Assess carryover risk based on operating conditions.

    Uses deterministic risk model - NO LLM/ML for numeric values.
    """
    risk_factors = []
    total_risk_score = 0.0

    # Factor 1: Load level
    load_risk = 0.0
    if risk_request.boiler_load_percent > 95:
        load_risk = 0.4
        risk_factors.append(CarryoverRiskFactors(
            factor_name="High boiler load",
            factor_value=risk_request.boiler_load_percent,
            contribution_score=0.4,
            is_primary_driver=True,
            mitigation_action="Reduce boiler load or bring backup online",
        ))
    elif risk_request.boiler_load_percent > 85:
        load_risk = 0.2
        risk_factors.append(CarryoverRiskFactors(
            factor_name="Elevated boiler load",
            factor_value=risk_request.boiler_load_percent,
            contribution_score=0.2,
            mitigation_action="Monitor closely and prepare for load reduction",
        ))
    total_risk_score += load_risk * 25

    # Factor 2: Drum level
    drum_risk = 0.0
    if risk_request.drum_level_percent > 75:
        drum_risk = 0.35
        risk_factors.append(CarryoverRiskFactors(
            factor_name="High drum level",
            factor_value=risk_request.drum_level_percent,
            contribution_score=0.35,
            is_primary_driver=risk_request.drum_level_percent > 80,
            mitigation_action="Lower drum level setpoint",
        ))
    elif risk_request.drum_level_percent > 60:
        drum_risk = 0.15
        risk_factors.append(CarryoverRiskFactors(
            factor_name="Elevated drum level",
            factor_value=risk_request.drum_level_percent,
            contribution_score=0.15,
            mitigation_action="Adjust feedwater flow",
        ))
    total_risk_score += drum_risk * 25

    # Factor 3: Load changes
    if risk_request.load_change_occurring:
        total_risk_score += 15
        risk_factors.append(CarryoverRiskFactors(
            factor_name="Active load change",
            factor_value=1.0,
            contribution_score=0.15,
            mitigation_action="Wait for load stabilization",
        ))

    # Factor 4: Separator efficiency
    if risk_request.separator_efficiency_percent:
        if risk_request.separator_efficiency_percent < 90:
            sep_risk = (90 - risk_request.separator_efficiency_percent) / 100
            total_risk_score += sep_risk * 20
            risk_factors.append(CarryoverRiskFactors(
                factor_name="Reduced separator efficiency",
                factor_value=risk_request.separator_efficiency_percent,
                contribution_score=sep_risk,
                mitigation_action="Inspect and service separator",
            ))

    # Factor 5: TDS levels
    if risk_request.boiler_tds_ppm and risk_request.boiler_tds_ppm > 3000:
        tds_risk = min((risk_request.boiler_tds_ppm - 3000) / 2000, 0.3)
        total_risk_score += tds_risk * 15
        risk_factors.append(CarryoverRiskFactors(
            factor_name="High boiler TDS",
            factor_value=risk_request.boiler_tds_ppm,
            contribution_score=tds_risk,
            mitigation_action="Increase blowdown rate",
        ))

    # Determine risk level from total score
    if total_risk_score >= 60:
        risk_level = CarryoverRiskLevel.CRITICAL
        risk_probability = min(0.95, 0.6 + (total_risk_score - 60) / 100)
    elif total_risk_score >= 30:
        risk_level = CarryoverRiskLevel.HIGH
        risk_probability = 0.3 + (total_risk_score - 30) / 100
    elif total_risk_score >= 10:
        risk_level = CarryoverRiskLevel.MODERATE
        risk_probability = 0.1 + (total_risk_score - 10) / 100
    else:
        risk_level = CarryoverRiskLevel.LOW
        risk_probability = total_risk_score / 100

    # Find primary driver
    primary_driver = None
    if risk_factors:
        primary = max(risk_factors, key=lambda f: f.contribution_score)
        primary_driver = primary.factor_name

    return CarryoverRiskAssessment(
        risk_level=risk_level,
        risk_probability=risk_probability,
        risk_score=total_risk_score,
        risk_factors=risk_factors,
        primary_risk_driver=primary_driver,
        predicted_quality_impact_percent=risk_probability * 5,  # Up to 5% quality drop
        time_to_threshold_min=60 / max(risk_probability, 0.1) if risk_probability > 0.1 else None,
        model_confidence=0.85,
    )


def _risk_exceeds_threshold(
    current: CarryoverRiskLevel,
    threshold: CarryoverRiskLevel
) -> bool:
    """Check if current risk exceeds threshold."""
    order = [
        CarryoverRiskLevel.LOW,
        CarryoverRiskLevel.MODERATE,
        CarryoverRiskLevel.HIGH,
        CarryoverRiskLevel.CRITICAL,
    ]
    return order.index(current) >= order.index(threshold)


def _get_quality_state(header_id: str) -> QualityStateResponse:
    """Get current quality state for a header (mock implementation)."""
    now = datetime.utcnow()

    return QualityStateResponse(
        header_id=header_id,
        success=True,
        current_quality_percent=98.5,
        quality_level=QualityLevel.GOOD,
        steam_pressure_kpa=1000.0,
        steam_temperature_c=185.0,
        steam_flow_kg_s=5.5,
        boiler_load_percent=75.0,
        quality_mean_24h=98.2,
        quality_std_24h=0.8,
        quality_min_24h=96.5,
        quality_max_24h=99.2,
        carryover_risk_level=CarryoverRiskLevel.LOW,
        quality_kpis=[
            QualityKPI(
                kpi_name="Average Quality",
                current_value=98.2,
                target_value=99.0,
                unit="%",
                trend="stable",
                is_on_target=False,
            ),
            QualityKPI(
                kpi_name="Time on Target",
                current_value=85.0,
                target_value=95.0,
                unit="%",
                trend="up",
                is_on_target=False,
            ),
        ],
        active_alarms=[],
        last_measurement_time=now - timedelta(seconds=30),
        data_staleness_seconds=30.0,
    )


def _get_quality_events(events_request: EventsRequest) -> List[QualityEvent]:
    """Get quality events (mock implementation)."""
    now = datetime.utcnow()

    events = [
        QualityEvent(
            event_type=EventType.QUALITY_DEGRADATION,
            severity=EventSeverity.WARNING,
            header_id=events_request.header_id or "HDR-001",
            title="Quality dropped below target",
            description="Steam quality dropped to 97.5%, below target of 99%",
            quality_at_event=97.5,
            threshold_value=99.0,
            deviation=-1.5,
            start_time=now - timedelta(hours=2),
            duration_minutes=45.0,
            is_active=False,
        ),
        QualityEvent(
            event_type=EventType.CARRYOVER_DETECTED,
            severity=EventSeverity.ALERT,
            header_id=events_request.header_id or "HDR-001",
            title="Moisture carryover detected",
            description="Elevated conductivity indicates moisture carryover",
            quality_at_event=96.0,
            start_time=now - timedelta(days=1),
            end_time=now - timedelta(days=1, hours=-1),
            duration_minutes=60.0,
            is_active=False,
            estimated_steam_loss_kg=150.0,
        ),
    ]

    if events_request.active_only:
        events = [e for e in events if e.is_active]

    return events[:events_request.limit]


def _generate_recommendations(rec_request: RecommendationsRequest) -> List[QualityRecommendation]:
    """Generate control recommendations (mock implementation)."""
    recommendations = []

    quality_gap = rec_request.target_quality_percent - rec_request.current_quality_percent

    if quality_gap > 1.0:
        # Blowdown adjustment recommendation
        recommendations.append(QualityRecommendation(
            recommendation_type=RecommendationType.BLOWDOWN_ADJUSTMENT,
            priority=RecommendationPriority.HIGH,
            header_id=rec_request.header_id,
            title="Increase blowdown rate",
            description="Increase continuous blowdown to reduce TDS and improve quality",
            rationale="Elevated TDS levels are contributing to moisture carryover",
            actions=[
                ControlAction(
                    action_name="Increase blowdown",
                    action_type=RecommendationType.BLOWDOWN_ADJUSTMENT,
                    current_value=2.0,
                    recommended_value=3.5,
                    change_magnitude=1.5,
                    unit="% of steam flow",
                    is_automated=False,
                    requires_operator_approval=True,
                ),
            ],
            expected_quality_improvement_percent=1.2,
            expected_time_to_effect_min=30.0,
            confidence_score=0.85,
            estimated_water_savings_kg_h=-50.0,  # Negative = more water usage
        ))

    if rec_request.measurements and rec_request.measurements.drum_level_percent:
        if rec_request.measurements.drum_level_percent > 60:
            recommendations.append(QualityRecommendation(
                recommendation_type=RecommendationType.DRUM_LEVEL_SETPOINT,
                priority=RecommendationPriority.MEDIUM,
                header_id=rec_request.header_id,
                title="Lower drum level setpoint",
                description="Reduce drum level setpoint to improve steam separation",
                rationale="Lower drum level provides more disengagement space",
                actions=[
                    ControlAction(
                        action_name="Adjust drum level setpoint",
                        action_type=RecommendationType.DRUM_LEVEL_SETPOINT,
                        current_value=rec_request.measurements.drum_level_percent,
                        recommended_value=55.0,
                        unit="%",
                        is_automated=True,
                        requires_operator_approval=True,
                    ),
                ],
                expected_quality_improvement_percent=0.8,
                expected_time_to_effect_min=15.0,
                confidence_score=0.80,
            ))

    if rec_request.carryover_risk_level in [CarryoverRiskLevel.HIGH, CarryoverRiskLevel.CRITICAL]:
        recommendations.append(QualityRecommendation(
            recommendation_type=RecommendationType.LOAD_REDUCTION,
            priority=RecommendationPriority.CRITICAL,
            header_id=rec_request.header_id,
            title="Reduce boiler load",
            description="Reduce boiler load to mitigate carryover risk",
            rationale="High carryover risk requires immediate load reduction",
            actions=[
                ControlAction(
                    action_name="Reduce load",
                    action_type=RecommendationType.LOAD_REDUCTION,
                    current_value=95.0,
                    recommended_value=85.0,
                    change_magnitude=-10.0,
                    unit="% load",
                    is_automated=False,
                    requires_operator_approval=True,
                    estimated_implementation_time_min=5.0,
                ),
            ],
            expected_quality_improvement_percent=2.0,
            expected_time_to_effect_min=10.0,
            confidence_score=0.90,
        ))

    return recommendations[:rec_request.max_recommendations]


def _get_quality_metrics(metrics_request: MetricsRequest) -> List[QualityMetrics]:
    """Get quality metrics (mock implementation)."""
    now = datetime.utcnow()
    period_start = now - timedelta(hours=metrics_request.period_hours)

    return [
        QualityMetrics(
            header_id=metrics_request.header_id or "HDR-001",
            period_start=period_start,
            period_end=now,
            average_quality_percent=98.2,
            min_quality_percent=96.5,
            max_quality_percent=99.5,
            std_quality_percent=0.75,
            time_on_target_percent=85.0,
            time_below_alarm_percent=3.5,
            carryover_event_count=1,
            total_carryover_duration_min=45.0,
            estimated_energy_loss_kwh=125.0,
            estimated_steam_loss_kg=200.0,
            quality_trend="stable",
            improvement_vs_previous_period_percent=0.5,
        ),
    ]
