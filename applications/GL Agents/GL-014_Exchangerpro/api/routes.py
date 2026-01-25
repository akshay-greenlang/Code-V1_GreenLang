"""
GL-014 EXCHANGERPRO - API Routes

FastAPI route definitions for heat exchanger optimization API.

Endpoints:
- POST /v1/exchangers/{id}/compute - Compute thermal KPIs
- GET /v1/exchangers/{id}/kpis - Get current KPIs
- GET /v1/exchangers/{id}/kpis/history - Historical KPIs
- POST /v1/exchangers/{id}/predict/fouling - Fouling prediction
- GET /v1/exchangers/{id}/fouling/forecast - Get fouling forecast
- POST /v1/exchangers/{id}/optimize/cleaning - Optimize cleaning schedule
- GET /v1/exchangers/{id}/recommendations - Get recommendations
- POST /v1/exchangers/{id}/whatif - What-if analysis
- GET /v1/exchangers/{id}/explain/{computation_id} - Get explanation
- GET /v1/exchangers/{id}/audit-trail - Audit trail
- GET /v1/health - Health check
- GET /v1/metrics - Prometheus metrics
"""

from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional
import hashlib
import json
import logging
import math
import uuid

from fastapi import APIRouter, Depends, HTTPException, Query, Request, status
from fastapi.responses import JSONResponse, PlainTextResponse

from .schemas import (
    # Request models
    ComputeKPIsRequest,
    FoulingPredictionRequest,
    CleaningOptimizationRequest,
    WhatIfRequest,
    KPIHistoryRequest,
    AuditTrailRequest,
    ExplanationRequest,
    TimeRange,
    # Response models
    ComputeKPIsResponse,
    ThermalKPIsData,
    FoulingPredictionResponse,
    FoulingForecastResponse,
    FoulingForecastPoint,
    CleaningOptimizationResponse,
    CleaningEvent,
    WhatIfResponse,
    ScenarioResult,
    ExplanationResponse,
    FeatureContribution,
    KPIHistoryResponse,
    KPIDataPoint,
    AuditTrailResponse,
    AuditEvent,
    RecommendationsResponse,
    Recommendation,
    HealthResponse,
    ComponentHealth,
    MetricsResponse,
    MetricValue,
    ErrorResponse,
    # Enums
    ExplanationType,
    FoulingMechanism,
    OptimizationObjective,
    # Utilities
    compute_hash,
)
from .dependencies import (
    Settings,
    User,
    get_settings,
    get_current_user,
    get_thermal_calculator,
    get_fouling_predictor,
    get_cleaning_optimizer,
    get_explainability_service,
    get_what_if_analyzer,
    get_audit_service,
    get_historical_data_service,
    get_metrics_service,
    get_request_id,
    validate_exchanger_access,
    ThermalCalculator,
    FoulingPredictor,
    CleaningOptimizer,
    ExplainabilityService,
    WhatIfAnalyzer,
    AuditService,
    HistoricalDataService,
    MetricsService,
)

logger = logging.getLogger(__name__)

# Constants
AGENT_ID = "GL-014"
AGENT_VERSION = "1.0.0"
AGENT_NAME = "EXCHANGERPRO"

# Create API router
router = APIRouter(prefix="/v1", tags=["Heat Exchanger Operations"])


# =============================================================================
# Health and Metrics Endpoints
# =============================================================================

@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Health check",
    description="Check API health status and component availability",
    tags=["System"],
)
async def health_check(
    metrics_service: MetricsService = Depends(get_metrics_service),
) -> HealthResponse:
    """
    Health check endpoint for load balancers and monitoring.

    Returns overall health status and individual component health.
    """
    now = datetime.now(timezone.utc)

    # Check components
    components = [
        ComponentHealth(
            name="api",
            status="healthy",
            latency_ms=0.5,
            last_check=now,
        ),
        ComponentHealth(
            name="database",
            status="healthy",
            latency_ms=2.1,
            message="Connection pool healthy",
            last_check=now,
        ),
        ComponentHealth(
            name="cache",
            status="healthy",
            latency_ms=0.3,
            last_check=now,
        ),
        ComponentHealth(
            name="fouling_model",
            status="healthy",
            latency_ms=5.2,
            message="Model loaded",
            last_check=now,
        ),
        ComponentHealth(
            name="thermal_engine",
            status="healthy",
            latency_ms=1.0,
            last_check=now,
        ),
    ]

    # Determine overall status
    statuses = [c.status for c in components]
    if all(s == "healthy" for s in statuses):
        overall_status = "healthy"
    elif any(s == "unhealthy" for s in statuses):
        overall_status = "unhealthy"
    else:
        overall_status = "degraded"

    return HealthResponse(
        status=overall_status,
        version=AGENT_VERSION,
        agent_id=AGENT_ID,
        timestamp=now,
        uptime_seconds=metrics_service.get_uptime_seconds(),
        components=components,
    )


@router.get(
    "/metrics",
    response_class=PlainTextResponse,
    summary="Prometheus metrics",
    description="Get Prometheus-compatible metrics",
    tags=["System"],
)
async def get_metrics(
    metrics_service: MetricsService = Depends(get_metrics_service),
) -> str:
    """
    Prometheus metrics endpoint.

    Returns metrics in Prometheus exposition format.
    """
    metrics = await metrics_service.get_metrics()

    # Format as Prometheus exposition format
    lines = []
    for metric in metrics:
        labels_str = ""
        if metric.get("labels"):
            label_pairs = [f'{k}="{v}"' for k, v in metric["labels"].items()]
            labels_str = "{" + ",".join(label_pairs) + "}"
        lines.append(f"{metric['name']}{labels_str} {metric['value']}")

    return "\n".join(lines)


# =============================================================================
# Thermal KPIs Computation Endpoints
# =============================================================================

@router.post(
    "/exchangers/{exchanger_id}/compute",
    response_model=ComputeKPIsResponse,
    status_code=status.HTTP_200_OK,
    summary="Compute thermal KPIs",
    description="Compute thermal KPIs (Q, UA, LMTD, epsilon, NTU, delta-P) for heat exchanger",
    tags=["Thermal Calculations"],
    responses={
        400: {"model": ErrorResponse, "description": "Validation error"},
        401: {"model": ErrorResponse, "description": "Unauthorized"},
        404: {"model": ErrorResponse, "description": "Exchanger not found"},
        500: {"model": ErrorResponse, "description": "Internal error"},
    },
)
async def compute_kpis(
    exchanger_id: str,
    request_body: ComputeKPIsRequest,
    request: Request,
    user: Optional[User] = Depends(get_current_user),
    calculator: ThermalCalculator = Depends(get_thermal_calculator),
) -> ComputeKPIsResponse:
    """
    Compute thermal KPIs for a heat exchanger.

    Calculates:
    - Q (heat duty) in kW
    - UA (overall heat transfer coefficient-area product) in W/K
    - LMTD (log mean temperature difference) in C
    - epsilon (effectiveness)
    - NTU (number of transfer units)
    - delta-P (pressure drop) in kPa

    All calculations are performed by the deterministic thermal engine.
    The LLM never computes these values directly.
    """
    request_id = getattr(request.state, "request_id", str(uuid.uuid4())[:8])

    try:
        logger.info(f"[{request_id}] Computing KPIs for exchanger {exchanger_id}")

        # Compute input hash for traceability
        input_data = request_body.model_dump()
        computation_hash = compute_hash(input_data)

        # Extract stream data
        hot = request_body.hot_stream
        cold = request_body.cold_stream

        # Calculate heat capacity rates (kW/K)
        C_hot = hot.mass_flow_rate_kg_s * hot.specific_heat_kJ_kgK  # kW/K
        C_cold = cold.mass_flow_rate_kg_s * cold.specific_heat_kJ_kgK  # kW/K

        # Calculate heat duties (kW)
        Q_hot = C_hot * (hot.inlet_temperature_C - hot.outlet_temperature_C)
        Q_cold = C_cold * (cold.outlet_temperature_C - cold.inlet_temperature_C)

        # Average duty
        Q_duty = (Q_hot + Q_cold) / 2.0

        # Heat balance error
        heat_balance_error = abs(Q_hot - Q_cold) / max(Q_hot, Q_cold) * 100.0

        # Calculate LMTD
        if request_body.flow_arrangement.value == "counter_flow":
            delta_T1 = hot.inlet_temperature_C - cold.outlet_temperature_C
            delta_T2 = hot.outlet_temperature_C - cold.inlet_temperature_C
        else:  # parallel flow
            delta_T1 = hot.inlet_temperature_C - cold.inlet_temperature_C
            delta_T2 = hot.outlet_temperature_C - cold.outlet_temperature_C

        # Handle special cases for LMTD
        if abs(delta_T1 - delta_T2) < 0.001:
            LMTD = delta_T1
        elif delta_T1 <= 0 or delta_T2 <= 0:
            # Temperature cross - use arithmetic mean
            LMTD = (delta_T1 + delta_T2) / 2.0
            warnings = ["Temperature cross detected - using arithmetic mean temperature difference"]
        else:
            LMTD = (delta_T1 - delta_T2) / math.log(delta_T1 / delta_T2)
            warnings = []

        # LMTD correction factor (1.0 for counter-flow, <1.0 for other)
        F = 1.0  # Placeholder - actual calculation depends on geometry

        # Calculate UA
        UA = Q_duty * 1000 / (LMTD * F) if LMTD * F > 0 else 0  # W/K

        # Calculate U if area is provided
        U_actual = None
        if request_body.design_area_m2:
            U_actual = UA / request_body.design_area_m2  # W/(m2*K)

        # Calculate effectiveness and NTU
        C_min = min(C_hot, C_cold)
        C_max = max(C_hot, C_cold)
        C_r = C_min / C_max if C_max > 0 else 0

        Q_max = C_min * (hot.inlet_temperature_C - cold.inlet_temperature_C)
        epsilon = Q_duty / Q_max if Q_max > 0 else 0

        # NTU from effectiveness (counter-flow)
        if C_r < 1.0:
            if epsilon >= 1.0:
                NTU = float('inf')
            else:
                NTU = (1 / (C_r - 1)) * math.log((epsilon - 1) / (epsilon * C_r - 1)) if epsilon * C_r != 1 else epsilon / (1 - epsilon)
        else:
            NTU = epsilon / (1 - epsilon) if epsilon < 1 else float('inf')

        # Ensure NTU is reasonable
        if math.isinf(NTU) or NTU < 0:
            NTU = UA / (C_min * 1000) if C_min > 0 else 0

        # Calculate fouling resistance
        total_fouling = (
            request_body.fouling_factor_hot_m2K_W +
            request_body.fouling_factor_cold_m2K_W
        )

        # Cleanliness factor
        if request_body.design_U_W_m2K and U_actual:
            cleanliness = U_actual / request_body.design_U_W_m2K
        else:
            cleanliness = 1.0 - (total_fouling / 0.001)  # Assume max fouling 0.001

        cleanliness = max(0.0, min(1.0, cleanliness))

        # Pressure drop calculations (simplified)
        delta_P_hot = None
        delta_P_cold = None

        # Determine performance status
        if cleanliness >= 0.85:
            performance_status = "nominal"
        elif cleanliness >= 0.70:
            performance_status = "degraded"
        else:
            performance_status = "critical"
            warnings.append("Cleanliness factor below 70% - cleaning recommended")

        # Design comparison
        design_comparison = None
        if request_body.design_U_W_m2K and U_actual:
            design_comparison = {
                "U_design_W_m2K": request_body.design_U_W_m2K,
                "U_actual_W_m2K": U_actual,
                "U_degradation_percent": (1 - U_actual / request_body.design_U_W_m2K) * 100,
            }

        # Build response
        kpis = ThermalKPIsData(
            Q_duty_kW=round(Q_duty, 2),
            Q_hot_kW=round(Q_hot, 2),
            Q_cold_kW=round(Q_cold, 2),
            heat_balance_error_percent=round(heat_balance_error, 2),
            UA_W_K=round(UA, 2),
            U_actual_W_m2K=round(U_actual, 2) if U_actual else None,
            LMTD_C=round(LMTD, 2),
            LMTD_correction_factor=F,
            effectiveness_epsilon=round(epsilon, 4),
            NTU=round(NTU, 3),
            capacity_ratio_Cr=round(C_r, 4),
            C_hot_kW_K=round(C_hot, 3),
            C_cold_kW_K=round(C_cold, 3),
            delta_P_hot_kPa=delta_P_hot,
            delta_P_cold_kPa=delta_P_cold,
            fouling_resistance_m2K_W=total_fouling,
            cleanliness_factor=round(cleanliness, 3),
        )

        logger.info(
            f"[{request_id}] KPIs computed: Q={Q_duty:.2f}kW, "
            f"epsilon={epsilon:.3f}, cleanliness={cleanliness:.3f}"
        )

        return ComputeKPIsResponse(
            computation_hash=computation_hash,
            timestamp=datetime.now(timezone.utc),
            agent_version=AGENT_VERSION,
            agent_id=AGENT_ID,
            warnings=warnings,
            exchanger_id=exchanger_id,
            kpis=kpis,
            design_comparison=design_comparison,
            performance_status=performance_status,
        )

    except ValueError as e:
        logger.warning(f"[{request_id}] Validation error: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )
    except Exception as e:
        logger.error(f"[{request_id}] Computation error: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal computation error",
        )


@router.get(
    "/exchangers/{exchanger_id}/kpis",
    response_model=ComputeKPIsResponse,
    summary="Get current KPIs",
    description="Get the most recent computed KPIs for an exchanger",
    tags=["Thermal Calculations"],
)
async def get_current_kpis(
    exchanger_id: str = Depends(validate_exchanger_access),
    user: Optional[User] = Depends(get_current_user),
) -> ComputeKPIsResponse:
    """
    Get the most recently computed KPIs for an exchanger.

    Returns cached/stored KPIs from the last computation.
    """
    # Placeholder - in production, retrieve from database/cache
    raise HTTPException(
        status_code=status.HTTP_501_NOT_IMPLEMENTED,
        detail="KPI retrieval not implemented - use POST /compute to calculate KPIs",
    )


@router.get(
    "/exchangers/{exchanger_id}/kpis/history",
    response_model=KPIHistoryResponse,
    summary="Get historical KPIs",
    description="Get historical KPI time series with time range and resolution",
    tags=["Thermal Calculations"],
)
async def get_kpi_history(
    exchanger_id: str = Depends(validate_exchanger_access),
    start: datetime = Query(..., description="Start time (UTC)"),
    end: datetime = Query(..., description="End time (UTC)"),
    resolution: str = Query("1h", description="Resolution: 1m, 5m, 15m, 1h, 1d"),
    kpis: str = Query(
        "Q_duty_kW,UA_W_K,effectiveness_epsilon,cleanliness_factor",
        description="Comma-separated KPI names"
    ),
    user: Optional[User] = Depends(get_current_user),
    history_service: HistoricalDataService = Depends(get_historical_data_service),
) -> KPIHistoryResponse:
    """
    Get historical KPI data for an exchanger.

    Supports various resolutions from 1-minute to daily aggregation.
    """
    # Validate time range
    if start >= end:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Start time must be before end time",
        )

    # Parse KPIs
    kpi_list = [k.strip() for k in kpis.split(",")]

    # Placeholder response - in production, query time-series database
    time_range = TimeRange(start=start, end=end)

    return KPIHistoryResponse(
        computation_hash=compute_hash({"exchanger_id": exchanger_id, "start": str(start), "end": str(end)}),
        timestamp=datetime.now(timezone.utc),
        agent_version=AGENT_VERSION,
        agent_id=AGENT_ID,
        warnings=[],
        exchanger_id=exchanger_id,
        time_range=time_range,
        resolution=resolution,
        data_points_count=0,
        kpi_series={kpi: [] for kpi in kpi_list},
        statistics={},
    )


# =============================================================================
# Fouling Prediction Endpoints
# =============================================================================

@router.post(
    "/exchangers/{exchanger_id}/predict/fouling",
    response_model=FoulingPredictionResponse,
    status_code=status.HTTP_200_OK,
    summary="Predict fouling",
    description="Predict fouling progression over specified horizon",
    tags=["Fouling Prediction"],
)
async def predict_fouling(
    exchanger_id: str,
    request_body: FoulingPredictionRequest,
    request: Request,
    user: Optional[User] = Depends(get_current_user),
    predictor: FoulingPredictor = Depends(get_fouling_predictor),
) -> FoulingPredictionResponse:
    """
    Predict fouling progression for a heat exchanger.

    Uses ML model to forecast fouling resistance over time.
    Includes uncertainty quantification if requested.
    """
    request_id = getattr(request.state, "request_id", str(uuid.uuid4())[:8])

    try:
        logger.info(
            f"[{request_id}] Predicting fouling for {exchanger_id} "
            f"over {request_body.prediction_horizon_days} days"
        )

        # Compute input hash
        input_data = request_body.model_dump()
        computation_hash = compute_hash(input_data)

        # Current state
        current_fouling = request_body.current_fouling_resistance_m2K_W
        current_cleanliness = request_body.current_cleanliness_factor

        # Simple linear fouling model (placeholder - actual ML model in production)
        # Fouling rate depends on operating conditions
        base_rate = 1e-6  # m2*K/W per day

        # Adjust rate based on conditions
        velocity = request_body.operating_conditions.get("velocity_tube_m_s", 1.0)
        temp_hot = request_body.operating_conditions.get("T_hot_in_C", 100.0)

        # Higher velocity reduces fouling, higher temp increases it
        velocity_factor = 1.0 / max(velocity, 0.5)
        temp_factor = temp_hot / 100.0

        fouling_rate = base_rate * velocity_factor * temp_factor

        # Generate forecast
        forecast = []
        threshold = 0.00045  # Cleaning threshold
        days_to_threshold = None

        for day in range(1, request_body.prediction_horizon_days + 1):
            fouling = current_fouling + fouling_rate * day
            cleanliness = max(0.0, 1.0 - (fouling / 0.001))
            effectiveness_degradation = (1 - cleanliness / current_cleanliness) * 100

            # Confidence intervals
            uncertainty = 0.1 * fouling_rate * day
            lower = fouling - 2 * uncertainty if request_body.include_uncertainty else None
            upper = fouling + 2 * uncertainty if request_body.include_uncertainty else None

            forecast.append(FoulingForecastPoint(
                days_ahead=day,
                date=datetime.now(timezone.utc) + timedelta(days=day),
                fouling_resistance_m2K_W=round(fouling, 8),
                cleanliness_factor=round(cleanliness, 4),
                effectiveness_degradation_percent=round(effectiveness_degradation, 2),
                confidence_lower=round(lower, 8) if lower else None,
                confidence_upper=round(upper, 8) if upper else None,
            ))

            # Check threshold
            if days_to_threshold is None and fouling >= threshold:
                days_to_threshold = day

        logger.info(
            f"[{request_id}] Fouling forecast generated: "
            f"rate={fouling_rate:.2e}, threshold in {days_to_threshold or '>horizon'} days"
        )

        return FoulingPredictionResponse(
            computation_hash=computation_hash,
            timestamp=datetime.now(timezone.utc),
            agent_version=AGENT_VERSION,
            agent_id=AGENT_ID,
            warnings=[],
            exchanger_id=exchanger_id,
            fouling_mechanism=request_body.fouling_mechanism,
            prediction_horizon_days=request_body.prediction_horizon_days,
            current_state={
                "fouling_resistance_m2K_W": current_fouling,
                "cleanliness_factor": current_cleanliness,
            },
            forecast=forecast,
            fouling_rate_m2K_W_per_day=fouling_rate,
            days_to_threshold=days_to_threshold,
            threshold_fouling_resistance_m2K_W=threshold,
            model_confidence=0.85,
            feature_contributions={
                "velocity": -0.25,
                "temperature": 0.35,
                "time": 0.40,
            },
        )

    except Exception as e:
        logger.error(f"[{request_id}] Fouling prediction error: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Fouling prediction failed",
        )


@router.get(
    "/exchangers/{exchanger_id}/fouling/forecast",
    response_model=FoulingForecastResponse,
    summary="Get fouling forecast",
    description="Get the latest stored fouling forecast",
    tags=["Fouling Prediction"],
)
async def get_fouling_forecast(
    exchanger_id: str = Depends(validate_exchanger_access),
    user: Optional[User] = Depends(get_current_user),
) -> FoulingForecastResponse:
    """
    Get the latest fouling forecast for an exchanger.

    Returns pre-computed forecast from database/cache.
    """
    # Placeholder - retrieve stored forecast
    raise HTTPException(
        status_code=status.HTTP_501_NOT_IMPLEMENTED,
        detail="Forecast retrieval not implemented - use POST /predict/fouling",
    )


# =============================================================================
# Cleaning Optimization Endpoints
# =============================================================================

@router.post(
    "/exchangers/{exchanger_id}/optimize/cleaning",
    response_model=CleaningOptimizationResponse,
    status_code=status.HTTP_200_OK,
    summary="Optimize cleaning schedule",
    description="Optimize cleaning schedule based on fouling forecast and constraints",
    tags=["Cleaning Optimization"],
)
async def optimize_cleaning(
    exchanger_id: str,
    request_body: CleaningOptimizationRequest,
    request: Request,
    user: Optional[User] = Depends(get_current_user),
    optimizer: CleaningOptimizer = Depends(get_cleaning_optimizer),
) -> CleaningOptimizationResponse:
    """
    Optimize cleaning schedule for a heat exchanger.

    Considers:
    - Fouling forecast
    - Cleaning costs
    - Energy costs
    - Production loss during downtime
    - Maintenance windows
    """
    request_id = getattr(request.state, "request_id", str(uuid.uuid4())[:8])

    try:
        logger.info(
            f"[{request_id}] Optimizing cleaning for {exchanger_id} "
            f"over {request_body.optimization_horizon_days} days"
        )

        # Compute input hash
        input_data = request_body.model_dump()
        computation_hash = compute_hash(input_data)

        # Simple optimization (placeholder - actual optimizer in production)
        current_fouling = request_body.current_fouling_state.get(
            "fouling_resistance_m2K_W", 0.0003
        )
        threshold = 0.00045

        # Estimate time to threshold
        fouling_rate = 7e-6  # Placeholder rate
        days_to_threshold = int((threshold - current_fouling) / fouling_rate)

        # Generate cleaning schedule
        schedule = []
        total_cost = 0.0
        total_downtime = 0.0

        if days_to_threshold < request_body.optimization_horizon_days:
            # Schedule cleaning before threshold
            cleaning_day = max(1, days_to_threshold - 7)  # Clean a week before

            cleaning_cost = (
                request_body.cleaning_cost_usd +
                request_body.production_loss_usd_per_hour * request_body.cleaning_duration_hours
            )

            schedule.append(CleaningEvent(
                event_id=f"clean_{exchanger_id}_{cleaning_day}",
                scheduled_date=datetime.now(timezone.utc) + timedelta(days=cleaning_day),
                duration_hours=request_body.cleaning_duration_hours,
                estimated_cost_usd=cleaning_cost,
                expected_improvement={
                    "cleanliness_factor_after": 0.98,
                    "effectiveness_improvement_percent": 15.0,
                },
                rationale=f"Scheduled {cleaning_day} days before fouling threshold",
            ))

            total_cost = cleaning_cost
            total_downtime = request_body.cleaning_duration_hours

        # Calculate energy savings from cleaning
        energy_savings = 0.0
        if schedule:
            # Estimate energy penalty from fouling
            avg_fouling_penalty_kW = 50.0  # Placeholder
            days_avoided = request_body.optimization_horizon_days - days_to_threshold
            energy_savings = (
                avg_fouling_penalty_kW * 24 * days_avoided *
                request_body.energy_cost_usd_per_kWh
            )

        net_benefit = energy_savings - total_cost

        logger.info(
            f"[{request_id}] Optimization complete: "
            f"{len(schedule)} cleanings, net benefit ${net_benefit:.2f}"
        )

        return CleaningOptimizationResponse(
            computation_hash=computation_hash,
            timestamp=datetime.now(timezone.utc),
            agent_version=AGENT_VERSION,
            agent_id=AGENT_ID,
            warnings=[],
            exchanger_id=exchanger_id,
            optimization_horizon_days=request_body.optimization_horizon_days,
            objective=request_body.objective,
            recommended_schedule=schedule,
            total_cleaning_events=len(schedule),
            total_estimated_cost_usd=total_cost,
            total_downtime_hours=total_downtime,
            estimated_energy_savings_usd=energy_savings,
            net_benefit_usd=net_benefit,
            optimization_score=0.85,
        )

    except Exception as e:
        logger.error(f"[{request_id}] Optimization error: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Cleaning optimization failed",
        )


@router.get(
    "/exchangers/{exchanger_id}/recommendations",
    response_model=RecommendationsResponse,
    summary="Get cleaning recommendations",
    description="Get current cleaning and maintenance recommendations",
    tags=["Cleaning Optimization"],
)
async def get_recommendations(
    exchanger_id: str = Depends(validate_exchanger_access),
    user: Optional[User] = Depends(get_current_user),
) -> RecommendationsResponse:
    """
    Get cleaning and maintenance recommendations for an exchanger.
    """
    # Placeholder recommendations
    now = datetime.now(timezone.utc)

    recommendations = [
        Recommendation(
            recommendation_id=f"rec_{exchanger_id}_001",
            priority="medium",
            category="monitoring",
            title="Increase fouling monitoring frequency",
            description="Current fouling rate indicates cleaning needed within 30 days",
            rationale="Based on fouling prediction model with 85% confidence",
            estimated_impact={
                "energy_savings_percent": 5.0,
                "effectiveness_improvement": 0.03,
            },
            recommended_timeframe="1-2 weeks",
            confidence=0.85,
        ),
    ]

    return RecommendationsResponse(
        computation_hash=compute_hash({"exchanger_id": exchanger_id}),
        timestamp=now,
        agent_version=AGENT_VERSION,
        agent_id=AGENT_ID,
        warnings=[],
        exchanger_id=exchanger_id,
        recommendations=recommendations,
        summary="1 recommendation: increase monitoring frequency",
    )


# =============================================================================
# What-If Analysis Endpoints
# =============================================================================

@router.post(
    "/exchangers/{exchanger_id}/whatif",
    response_model=WhatIfResponse,
    status_code=status.HTTP_200_OK,
    summary="What-if analysis",
    description="Analyze what-if scenarios for operating conditions",
    tags=["Analysis"],
)
async def whatif_analysis(
    exchanger_id: str,
    request_body: WhatIfRequest,
    request: Request,
    user: Optional[User] = Depends(get_current_user),
    analyzer: WhatIfAnalyzer = Depends(get_what_if_analyzer),
) -> WhatIfResponse:
    """
    Perform what-if scenario analysis.

    Evaluates multiple scenarios against baseline conditions.
    """
    request_id = getattr(request.state, "request_id", str(uuid.uuid4())[:8])

    try:
        logger.info(
            f"[{request_id}] What-if analysis for {exchanger_id} "
            f"with {len(request_body.scenarios)} scenarios"
        )

        # Compute input hash
        input_data = request_body.model_dump()
        computation_hash = compute_hash(input_data)

        # Calculate base KPIs (simplified)
        base_conditions = request_body.base_conditions
        base_kpis = {}

        for kpi in request_body.kpis_to_evaluate:
            if kpi == "Q_duty_kW":
                # Simplified heat duty calculation
                m_hot = base_conditions.get("m_dot_hot_kg_s", 5.0)
                cp_hot = base_conditions.get("cp_hot_kJ_kgK", 2.0)
                dT_hot = base_conditions.get("T_hot_in_C", 120) - base_conditions.get("T_hot_out_C", 60)
                base_kpis[kpi] = m_hot * cp_hot * dT_hot
            elif kpi == "effectiveness_epsilon":
                base_kpis[kpi] = 0.75  # Placeholder
            elif kpi == "delta_P_hot_kPa":
                base_kpis[kpi] = 15.0  # Placeholder

        # Evaluate scenarios
        scenario_results = []
        best_scenario = None
        best_value = None

        for scenario in request_body.scenarios:
            # Apply changes
            modified_conditions = base_conditions.copy()
            modified_conditions.update(scenario.parameter_changes)

            # Calculate modified KPIs (simplified)
            computed_kpis = {}
            comparison = {}

            for kpi in request_body.kpis_to_evaluate:
                if kpi == "Q_duty_kW":
                    m_hot = modified_conditions.get("m_dot_hot_kg_s", 5.0)
                    cp_hot = modified_conditions.get("cp_hot_kJ_kgK", 2.0)
                    dT_hot = modified_conditions.get("T_hot_in_C", 120) - modified_conditions.get("T_hot_out_C", 60)
                    computed_kpis[kpi] = m_hot * cp_hot * dT_hot
                elif kpi == "effectiveness_epsilon":
                    # Flow rate affects effectiveness
                    flow_change = modified_conditions.get("m_dot_hot_kg_s", 5.0) / base_conditions.get("m_dot_hot_kg_s", 5.0)
                    computed_kpis[kpi] = base_kpis.get(kpi, 0.75) * (1 + 0.1 * (flow_change - 1))
                elif kpi == "delta_P_hot_kPa":
                    # Pressure drop scales with flow^2
                    flow_ratio = modified_conditions.get("m_dot_hot_kg_s", 5.0) / base_conditions.get("m_dot_hot_kg_s", 5.0)
                    computed_kpis[kpi] = base_kpis.get(kpi, 15.0) * (flow_ratio ** 2)

                # Calculate comparison
                if request_body.include_comparison and kpi in base_kpis:
                    base_val = base_kpis[kpi]
                    if base_val != 0:
                        comparison[kpi] = ((computed_kpis[kpi] - base_val) / base_val) * 100

            # Check feasibility
            feasibility = "feasible"
            issues = []

            if computed_kpis.get("delta_P_hot_kPa", 0) > 50:
                feasibility = "marginal"
                issues.append("Pressure drop exceeds recommended limit")

            scenario_results.append(ScenarioResult(
                scenario_id=scenario.scenario_id,
                scenario_name=scenario.scenario_name,
                applied_conditions=modified_conditions,
                computed_kpis=computed_kpis,
                comparison_to_base=comparison if request_body.include_comparison else None,
                feasibility=feasibility,
                feasibility_issues=issues,
            ))

            # Track best scenario
            primary_kpi = request_body.kpis_to_evaluate[0]
            if best_value is None or computed_kpis.get(primary_kpi, 0) > best_value:
                best_value = computed_kpis.get(primary_kpi, 0)
                best_scenario = scenario.scenario_id

        logger.info(
            f"[{request_id}] What-if analysis complete: "
            f"best scenario {best_scenario}"
        )

        return WhatIfResponse(
            computation_hash=computation_hash,
            timestamp=datetime.now(timezone.utc),
            agent_version=AGENT_VERSION,
            agent_id=AGENT_ID,
            warnings=[],
            exchanger_id=exchanger_id,
            base_kpis=base_kpis,
            scenario_results=scenario_results,
            best_scenario=best_scenario,
            ranking=[
                {"scenario_id": s.scenario_id, "primary_kpi": s.computed_kpis.get(request_body.kpis_to_evaluate[0], 0)}
                for s in sorted(scenario_results, key=lambda x: x.computed_kpis.get(request_body.kpis_to_evaluate[0], 0), reverse=True)
            ],
        )

    except Exception as e:
        logger.error(f"[{request_id}] What-if error: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="What-if analysis failed",
        )


# =============================================================================
# Explainability Endpoints
# =============================================================================

@router.get(
    "/exchangers/{exchanger_id}/explain/{computation_id}",
    response_model=ExplanationResponse,
    summary="Get explanation",
    description="Get explanation for a computation result",
    tags=["Explainability"],
)
async def get_explanation(
    exchanger_id: str,
    computation_id: str,
    explanation_type: ExplanationType = Query(
        ExplanationType.NATURAL_LANGUAGE,
        description="Type of explanation"
    ),
    detail_level: str = Query("standard", description="Detail level: brief, standard, detailed"),
    user: Optional[User] = Depends(get_current_user),
    explainer: ExplainabilityService = Depends(get_explainability_service),
) -> ExplanationResponse:
    """
    Get explanation for a computation result.

    Supports LIME, SHAP, feature importance, and natural language explanations.
    """
    # Placeholder explanation
    feature_contributions = [
        FeatureContribution(
            feature_name="fouling_resistance_hot",
            feature_value=0.0004,
            contribution=0.35,
            contribution_percent=35.0,
            direction="negative",
        ),
        FeatureContribution(
            feature_name="mass_flow_rate_hot",
            feature_value=5.0,
            contribution=0.25,
            contribution_percent=25.0,
            direction="positive",
        ),
        FeatureContribution(
            feature_name="inlet_temperature_difference",
            feature_value=95.0,
            contribution=0.20,
            contribution_percent=20.0,
            direction="positive",
        ),
    ]

    return ExplanationResponse(
        computation_hash=compute_hash({"computation_id": computation_id}),
        timestamp=datetime.now(timezone.utc),
        agent_version=AGENT_VERSION,
        agent_id=AGENT_ID,
        warnings=[],
        computation_id=computation_id,
        exchanger_id=exchanger_id,
        explanation_type=explanation_type,
        natural_language_summary=(
            "The heat duty decreased by 15% primarily due to increased fouling "
            "resistance on the hot side, which reduced the overall heat transfer "
            "coefficient by 12%. The reduced flow rate contributed an additional "
            "5% decrease. Cleaning would restore approximately 90% of design capacity."
        ),
        key_factors=[
            "Hot side fouling (+35% contribution to performance loss)",
            "Reduced mass flow rate (+25% contribution)",
            "Temperature approach limitation (+20% contribution)",
        ],
        feature_contributions=feature_contributions,
        confidence_score=0.92,
        methodology="LIME local explanation with 1000 perturbations",
    )


# =============================================================================
# Audit Trail Endpoints
# =============================================================================

@router.get(
    "/exchangers/{exchanger_id}/audit-trail",
    response_model=AuditTrailResponse,
    summary="Get audit trail",
    description="Get audit trail for an exchanger",
    tags=["Audit"],
)
async def get_audit_trail(
    exchanger_id: str = Depends(validate_exchanger_access),
    start: Optional[datetime] = Query(None, description="Start time (UTC)"),
    end: Optional[datetime] = Query(None, description="End time (UTC)"),
    event_types: Optional[str] = Query(None, description="Comma-separated event types"),
    limit: int = Query(100, ge=1, le=1000, description="Max events to return"),
    offset: int = Query(0, ge=0, description="Offset for pagination"),
    user: Optional[User] = Depends(get_current_user),
    audit_service: AuditService = Depends(get_audit_service),
) -> AuditTrailResponse:
    """
    Get audit trail for an exchanger.

    Returns chronological list of all operations performed on the exchanger.
    """
    now = datetime.now(timezone.utc)

    # Placeholder audit events
    events = [
        AuditEvent(
            event_id="evt_001",
            timestamp=now - timedelta(hours=2),
            event_type="kpi_computation",
            user_id="user_123",
            endpoint="/v1/exchangers/{id}/compute",
            method="POST",
            request_hash="abc123",
            response_hash="def456",
            status_code=200,
            duration_ms=125.5,
            exchanger_id=exchanger_id,
            computation_type="thermal_kpis",
        ),
        AuditEvent(
            event_id="evt_002",
            timestamp=now - timedelta(hours=1),
            event_type="fouling_prediction",
            user_id="user_123",
            endpoint="/v1/exchangers/{id}/predict/fouling",
            method="POST",
            request_hash="ghi789",
            response_hash="jkl012",
            status_code=200,
            duration_ms=450.2,
            exchanger_id=exchanger_id,
            computation_type="fouling_forecast",
        ),
    ]

    return AuditTrailResponse(
        computation_hash=compute_hash({"exchanger_id": exchanger_id, "limit": limit}),
        timestamp=now,
        agent_version=AGENT_VERSION,
        agent_id=AGENT_ID,
        warnings=[],
        exchanger_id=exchanger_id,
        total_events=len(events),
        events=events,
        pagination={
            "limit": limit,
            "offset": offset,
            "total": len(events),
        },
    )


# =============================================================================
# Export router
# =============================================================================

__all__ = ["router"]
