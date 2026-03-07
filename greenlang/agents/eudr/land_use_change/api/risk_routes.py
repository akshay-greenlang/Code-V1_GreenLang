# -*- coding: utf-8 -*-
"""
Risk Routes - AGENT-EUDR-005 Land Use Change Detector API

Endpoints for conversion risk assessment, urban encroachment analysis,
and asynchronous batch job management.

Endpoints:
    POST /risk/assess       - Assess conversion risk for a single plot
    POST /risk/batch        - Batch risk assessment
    GET  /risk/{plot_id}    - Get stored risk result
    POST /urban/analyze     - Analyze urban encroachment
    POST /urban/batch       - Batch urban encroachment analysis
    GET  /urban/{plot_id}   - Get stored urban result
    POST /batch             - Submit asynchronous batch job
    DELETE /batch/{batch_id} - Cancel batch job

Author: GreenLang Platform Team
Date: March 2026
PRD: AGENT-EUDR-005 Land Use Change Detector Agent (GL-EUDR-LUC-005)
"""

from __future__ import annotations

import logging
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List

from fastapi import APIRouter, Depends, HTTPException, Request, status

from greenlang.agents.eudr.land_use_change.api.dependencies import (
    AuthUser,
    ErrorResponse,
    get_land_use_service,
    get_request_id,
    rate_limit_heavy,
    rate_limit_standard,
    rate_limit_write,
    require_permission,
    validate_plot_id,
)
from greenlang.agents.eudr.land_use_change.api.schemas import (
    BatchJobResponse,
    BatchJobStatus,
    BatchJobSubmitRequest,
    InfrastructureFeature,
    PressureCorridor,
    RiskAssessRequest,
    RiskBatchRequest,
    RiskBatchResponse,
    RiskFactor,
    RiskResult,
    RiskTier,
    UrbanAnalyzeRequest,
    UrbanBatchRequest,
    UrbanBatchResponse,
    UrbanResult,
)

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Risk & Urban Analysis"])

# ---------------------------------------------------------------------------
# In-memory stores (replaced by database in production)
# ---------------------------------------------------------------------------

_risk_store: Dict[str, Dict[str, Any]] = {}
_urban_store: Dict[str, Dict[str, Any]] = {}
_batch_job_store: Dict[str, Dict[str, Any]] = {}


def _get_risk_store() -> Dict[str, Dict[str, Any]]:
    """Return the risk store. Replaceable for testing."""
    return _risk_store


def _get_urban_store() -> Dict[str, Dict[str, Any]]:
    """Return the urban store. Replaceable for testing."""
    return _urban_store


def _get_batch_job_store() -> Dict[str, Dict[str, Any]]:
    """Return the batch job store. Replaceable for testing."""
    return _batch_job_store


# ---------------------------------------------------------------------------
# Risk factor display names
# ---------------------------------------------------------------------------

_RISK_FACTOR_DISPLAY_NAMES: Dict[str, str] = {
    "transition_magnitude": "Transition Magnitude",
    "proximity_to_forest": "Proximity to Forest",
    "historical_deforestation_rate": "Historical Deforestation Rate",
    "commodity_pressure": "Commodity Pressure",
    "governance_score": "Governance Score",
    "protected_area_proximity": "Protected Area Proximity",
    "road_infrastructure_proximity": "Road Infrastructure Proximity",
    "population_density_change": "Population Density Change",
}


# ---------------------------------------------------------------------------
# POST /risk/assess
# ---------------------------------------------------------------------------


@router.post(
    "/risk/assess",
    response_model=RiskResult,
    status_code=status.HTTP_200_OK,
    summary="Assess conversion risk",
    description=(
        "Assess the risk of future land use conversion (deforestation "
        "or degradation) for a single plot. Evaluates 8 risk factors "
        "(transition magnitude, proximity to forest, historical "
        "deforestation rate, commodity pressure, governance score, "
        "protected area proximity, road infrastructure proximity, "
        "population density change) and produces a composite risk "
        "score with tier classification and conversion probability "
        "estimates at 6, 12, and 24 month horizons."
    ),
    responses={
        200: {"description": "Risk assessment result"},
        400: {"model": ErrorResponse, "description": "Invalid request"},
        401: {"model": ErrorResponse, "description": "Authentication required"},
        403: {"model": ErrorResponse, "description": "Insufficient permissions"},
        429: {"model": ErrorResponse, "description": "Rate limit exceeded"},
    },
)
async def assess_risk(
    body: RiskAssessRequest,
    request: Request,
    user: AuthUser = Depends(
        require_permission("eudr-luc:risk:write")
    ),
    _rate: None = Depends(rate_limit_write),
) -> RiskResult:
    """Assess conversion risk for a single plot.

    Args:
        body: Risk assessment request with coordinates and commodity.
        user: Authenticated user with risk:write permission.

    Returns:
        RiskResult with composite score, tier, and factor details.

    Raises:
        HTTPException: 400 if request invalid, 500 on processing error.
    """
    start = time.monotonic()
    plot_id = body.plot_id or f"luc-rsk-{uuid.uuid4().hex[:12]}"

    logger.info(
        "Risk assessment: user=%s plot=%s lat=%.6f lon=%.6f "
        "commodity=%s",
        user.user_id,
        plot_id,
        body.latitude,
        body.longitude,
        body.commodity.value,
    )

    try:
        service = get_land_use_service()

        result = service.assess_risk(
            latitude=body.latitude,
            longitude=body.longitude,
            commodity=body.commodity.value,
            polygon_wkt=body.polygon_wkt,
            include_factors=body.include_factors,
        )

        elapsed = time.monotonic() - start

        # Build risk factors
        risk_factors = []
        if body.include_factors:
            raw_factors = getattr(result, "risk_factors", [])
            for factor in raw_factors:
                factor_name = getattr(factor, "factor_name", "")
                risk_factors.append(
                    RiskFactor(
                        factor_name=factor_name,
                        display_name=_RISK_FACTOR_DISPLAY_NAMES.get(
                            factor_name, factor_name.replace("_", " ").title()
                        ),
                        score=getattr(factor, "score", 0.0),
                        weight=getattr(factor, "weight", 0.0),
                        weighted_score=getattr(
                            factor, "weighted_score", 0.0
                        ),
                        description=getattr(factor, "description", ""),
                        data_quality=getattr(
                            factor, "data_quality", "medium"
                        ),
                    )
                )

        composite_score = getattr(result, "composite_score", 0.0)
        risk_tier = getattr(result, "risk_tier", RiskTier.LOW)

        response = RiskResult(
            request_id=get_request_id(),
            plot_id=plot_id,
            composite_score=composite_score,
            risk_tier=risk_tier,
            risk_factors=risk_factors,
            probability_6m=getattr(result, "probability_6m", 0.0),
            probability_12m=getattr(result, "probability_12m", 0.0),
            probability_24m=getattr(result, "probability_24m", 0.0),
            commodity=body.commodity.value,
            recommendations=getattr(result, "recommendations", []),
            latitude=body.latitude,
            longitude=body.longitude,
            data_sources=getattr(result, "data_sources", []),
            processing_time_ms=elapsed * 1000,
            provenance_hash=getattr(result, "provenance_hash", ""),
        )

        # Store for retrieval
        store = _get_risk_store()
        store[plot_id] = {
            "plot_id": plot_id,
            "response_data": response.model_dump(mode="json"),
            "created_at": datetime.now(timezone.utc).isoformat(),
            "created_by": user.user_id,
        }

        logger.info(
            "Risk assessment completed: user=%s plot=%s "
            "score=%.3f tier=%s elapsed_ms=%.1f",
            user.user_id,
            plot_id,
            composite_score,
            getattr(risk_tier, "value", risk_tier),
            elapsed * 1000,
        )

        return response

    except ValueError as exc:
        logger.warning(
            "Risk assessment error: user=%s plot=%s error=%s",
            user.user_id,
            plot_id,
            exc,
        )
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        )
    except Exception as exc:
        logger.error(
            "Risk assessment failed: user=%s plot=%s error=%s",
            user.user_id,
            plot_id,
            exc,
            exc_info=True,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Risk assessment failed: {str(exc)}",
        )


# ---------------------------------------------------------------------------
# POST /risk/batch
# ---------------------------------------------------------------------------


@router.post(
    "/risk/batch",
    response_model=RiskBatchResponse,
    status_code=status.HTTP_200_OK,
    summary="Batch risk assessment",
    description=(
        "Assess conversion risk for multiple plots in a single request. "
        "Supports up to 5000 plots per batch."
    ),
    responses={
        200: {"description": "Batch risk results"},
        400: {"model": ErrorResponse, "description": "Invalid request"},
        401: {"model": ErrorResponse, "description": "Authentication required"},
        403: {"model": ErrorResponse, "description": "Insufficient permissions"},
        429: {"model": ErrorResponse, "description": "Rate limit exceeded"},
    },
)
async def assess_risk_batch(
    body: RiskBatchRequest,
    request: Request,
    user: AuthUser = Depends(
        require_permission("eudr-luc:risk:write")
    ),
    _rate: None = Depends(rate_limit_heavy),
) -> RiskBatchResponse:
    """Batch assess risk for multiple plots.

    Args:
        body: Batch request with list of plots.
        user: Authenticated user with risk:write permission.

    Returns:
        RiskBatchResponse with results and tier distribution.
    """
    start = time.monotonic()
    total = len(body.plots)

    logger.info(
        "Batch risk assessment: user=%s plots=%d",
        user.user_id,
        total,
    )

    results: List[RiskResult] = []
    successful = 0
    failed = 0
    tier_counts: Dict[str, int] = {}
    score_sum = 0.0

    try:
        service = get_land_use_service()
        store = _get_risk_store()

        for plot_req in body.plots:
            plot_id = (
                plot_req.plot_id or f"luc-rsk-{uuid.uuid4().hex[:12]}"
            )

            try:
                result = service.assess_risk(
                    latitude=plot_req.latitude,
                    longitude=plot_req.longitude,
                    commodity=plot_req.commodity.value,
                    polygon_wkt=plot_req.polygon_wkt,
                    include_factors=plot_req.include_factors,
                )

                composite_score = getattr(
                    result, "composite_score", 0.0
                )
                risk_tier = getattr(result, "risk_tier", RiskTier.LOW)
                tier_val = (
                    risk_tier.value
                    if hasattr(risk_tier, "value")
                    else str(risk_tier)
                )

                risk_result = RiskResult(
                    request_id=get_request_id(),
                    plot_id=plot_id,
                    composite_score=composite_score,
                    risk_tier=risk_tier,
                    risk_factors=[],
                    probability_6m=getattr(
                        result, "probability_6m", 0.0
                    ),
                    probability_12m=getattr(
                        result, "probability_12m", 0.0
                    ),
                    probability_24m=getattr(
                        result, "probability_24m", 0.0
                    ),
                    commodity=plot_req.commodity.value,
                    recommendations=getattr(
                        result, "recommendations", []
                    ),
                    latitude=plot_req.latitude,
                    longitude=plot_req.longitude,
                    data_sources=getattr(result, "data_sources", []),
                    provenance_hash=getattr(
                        result, "provenance_hash", ""
                    ),
                )

                results.append(risk_result)
                successful += 1
                score_sum += composite_score
                tier_counts[tier_val] = (
                    tier_counts.get(tier_val, 0) + 1
                )

                store[plot_id] = {
                    "plot_id": plot_id,
                    "response_data": risk_result.model_dump(
                        mode="json"
                    ),
                    "created_at": (
                        datetime.now(timezone.utc).isoformat()
                    ),
                    "created_by": user.user_id,
                }

            except Exception as exc:
                logger.warning(
                    "Batch risk failed for plot %s: %s",
                    plot_id,
                    exc,
                )
                failed += 1

        elapsed = time.monotonic() - start
        mean_score = score_sum / successful if successful > 0 else 0.0

        logger.info(
            "Batch risk completed: user=%s total=%d successful=%d "
            "failed=%d mean_score=%.3f elapsed_ms=%.1f",
            user.user_id,
            total,
            successful,
            failed,
            mean_score,
            elapsed * 1000,
        )

        return RiskBatchResponse(
            request_id=get_request_id(),
            results=results,
            total=total,
            successful=successful,
            failed=failed,
            by_tier=tier_counts,
            mean_composite_score=mean_score,
            processing_time_ms=elapsed * 1000,
        )

    except Exception as exc:
        logger.error(
            "Batch risk failed: user=%s error=%s",
            user.user_id,
            exc,
            exc_info=True,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Batch risk assessment failed: {str(exc)}",
        )


# ---------------------------------------------------------------------------
# GET /risk/{plot_id}
# ---------------------------------------------------------------------------


@router.get(
    "/risk/{plot_id}",
    response_model=RiskResult,
    status_code=status.HTTP_200_OK,
    summary="Get stored risk result",
    description="Retrieve a previously computed risk assessment by plot ID.",
    responses={
        200: {"description": "Risk result"},
        404: {"model": ErrorResponse, "description": "Plot not found"},
        401: {"model": ErrorResponse, "description": "Authentication required"},
        403: {"model": ErrorResponse, "description": "Insufficient permissions"},
    },
)
async def get_risk(
    plot_id: str,
    request: Request,
    user: AuthUser = Depends(
        require_permission("eudr-luc:risk:read")
    ),
    _rate: None = Depends(rate_limit_standard),
) -> RiskResult:
    """Retrieve a stored risk result by plot ID.

    Args:
        plot_id: Plot identifier to look up.
        user: Authenticated user with risk:read permission.

    Returns:
        RiskResult from the store.

    Raises:
        HTTPException: 404 if plot_id not found.
    """
    plot_id = validate_plot_id(plot_id)
    store = _get_risk_store()

    if plot_id not in store:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No risk result found for plot_id '{plot_id}'",
        )

    record = store[plot_id]
    return RiskResult(**record["response_data"])


# ---------------------------------------------------------------------------
# POST /urban/analyze
# ---------------------------------------------------------------------------


@router.post(
    "/urban/analyze",
    response_model=UrbanResult,
    status_code=status.HTTP_200_OK,
    summary="Analyze urban encroachment",
    description=(
        "Analyze urban encroachment around a plot within a configurable "
        "buffer zone. Detects infrastructure features, urban expansion "
        "pressure corridors, and estimates time to conversion."
    ),
    responses={
        200: {"description": "Urban encroachment result"},
        400: {"model": ErrorResponse, "description": "Invalid request"},
        401: {"model": ErrorResponse, "description": "Authentication required"},
        403: {"model": ErrorResponse, "description": "Insufficient permissions"},
        429: {"model": ErrorResponse, "description": "Rate limit exceeded"},
    },
)
async def analyze_urban(
    body: UrbanAnalyzeRequest,
    request: Request,
    user: AuthUser = Depends(
        require_permission("eudr-luc:risk:write")
    ),
    _rate: None = Depends(rate_limit_write),
) -> UrbanResult:
    """Analyze urban encroachment around a plot.

    Args:
        body: Urban analysis request with coordinates and buffer.
        user: Authenticated user with risk:write permission.

    Returns:
        UrbanResult with encroachment analysis.

    Raises:
        HTTPException: 400 if request invalid, 500 on processing error.
    """
    start = time.monotonic()
    plot_id = body.plot_id or f"luc-urb-{uuid.uuid4().hex[:12]}"

    logger.info(
        "Urban analysis: user=%s plot=%s lat=%.6f lon=%.6f "
        "buffer=%.1fkm from=%s to=%s",
        user.user_id,
        plot_id,
        body.latitude,
        body.longitude,
        body.buffer_km,
        body.date_from,
        body.date_to,
    )

    try:
        service = get_land_use_service()

        result = service.analyze_urban_encroachment(
            latitude=body.latitude,
            longitude=body.longitude,
            buffer_km=body.buffer_km,
            date_from=body.date_from,
            date_to=body.date_to,
            polygon_wkt=body.polygon_wkt,
        )

        elapsed = time.monotonic() - start

        # Build infrastructure features
        infrastructure = []
        raw_infra = getattr(result, "infrastructure_types", [])
        for feat in raw_infra:
            infrastructure.append(
                InfrastructureFeature(
                    feature_type=getattr(feat, "feature_type", ""),
                    distance_km=getattr(feat, "distance_km", 0.0),
                    bearing_degrees=getattr(
                        feat, "bearing_degrees", 0.0
                    ),
                    growth_rate_pct_year=getattr(
                        feat, "growth_rate_pct_year", None
                    ),
                )
            )

        # Build pressure corridors
        corridors = []
        raw_corridors = getattr(result, "pressure_corridors", [])
        for corr in raw_corridors:
            corridors.append(
                PressureCorridor(
                    corridor_id=getattr(
                        corr, "corridor_id",
                        f"corr-{uuid.uuid4().hex[:8]}",
                    ),
                    direction=getattr(corr, "direction", "N"),
                    width_km=getattr(corr, "width_km", 0.0),
                    expansion_rate_ha_year=getattr(
                        corr, "expansion_rate_ha_year", 0.0
                    ),
                    distance_to_plot_km=getattr(
                        corr, "distance_to_plot_km", 0.0
                    ),
                    estimated_arrival_months=getattr(
                        corr, "estimated_arrival_months", None
                    ),
                )
            )

        response = UrbanResult(
            request_id=get_request_id(),
            plot_id=plot_id,
            encroachment_detected=getattr(
                result, "encroachment_detected", False
            ),
            urban_expansion_rate_ha_year=getattr(
                result, "urban_expansion_rate_ha_year", 0.0
            ),
            urban_area_start_ha=getattr(
                result, "urban_area_start_ha", 0.0
            ),
            urban_area_end_ha=getattr(
                result, "urban_area_end_ha", 0.0
            ),
            infrastructure_types=infrastructure,
            pressure_corridors=corridors,
            time_to_conversion_months=getattr(
                result, "time_to_conversion_months", None
            ),
            buffer_km=body.buffer_km,
            date_from=body.date_from,
            date_to=body.date_to,
            latitude=body.latitude,
            longitude=body.longitude,
            data_sources=getattr(result, "data_sources", []),
            processing_time_ms=elapsed * 1000,
            provenance_hash=getattr(result, "provenance_hash", ""),
        )

        # Store for retrieval
        store = _get_urban_store()
        store[plot_id] = {
            "plot_id": plot_id,
            "response_data": response.model_dump(mode="json"),
            "created_at": datetime.now(timezone.utc).isoformat(),
            "created_by": user.user_id,
        }

        logger.info(
            "Urban analysis completed: user=%s plot=%s "
            "encroachment=%s rate=%.2f ha/yr corridors=%d "
            "elapsed_ms=%.1f",
            user.user_id,
            plot_id,
            response.encroachment_detected,
            response.urban_expansion_rate_ha_year,
            len(corridors),
            elapsed * 1000,
        )

        return response

    except ValueError as exc:
        logger.warning(
            "Urban analysis error: user=%s plot=%s error=%s",
            user.user_id,
            plot_id,
            exc,
        )
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        )
    except Exception as exc:
        logger.error(
            "Urban analysis failed: user=%s plot=%s error=%s",
            user.user_id,
            plot_id,
            exc,
            exc_info=True,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Urban encroachment analysis failed: {str(exc)}",
        )


# ---------------------------------------------------------------------------
# POST /urban/batch
# ---------------------------------------------------------------------------


@router.post(
    "/urban/batch",
    response_model=UrbanBatchResponse,
    status_code=status.HTTP_200_OK,
    summary="Batch urban encroachment analysis",
    description=(
        "Analyze urban encroachment for multiple plots in a single "
        "request. Supports up to 5000 plots per batch."
    ),
    responses={
        200: {"description": "Batch urban results"},
        400: {"model": ErrorResponse, "description": "Invalid request"},
        401: {"model": ErrorResponse, "description": "Authentication required"},
        403: {"model": ErrorResponse, "description": "Insufficient permissions"},
        429: {"model": ErrorResponse, "description": "Rate limit exceeded"},
    },
)
async def analyze_urban_batch(
    body: UrbanBatchRequest,
    request: Request,
    user: AuthUser = Depends(
        require_permission("eudr-luc:risk:write")
    ),
    _rate: None = Depends(rate_limit_heavy),
) -> UrbanBatchResponse:
    """Batch analyze urban encroachment for multiple plots.

    Args:
        body: Batch request with list of plots and optional buffer.
        user: Authenticated user with risk:write permission.

    Returns:
        UrbanBatchResponse with results and statistics.
    """
    start = time.monotonic()
    total = len(body.plots)

    logger.info(
        "Batch urban analysis: user=%s plots=%d",
        user.user_id,
        total,
    )

    results: List[UrbanResult] = []
    successful = 0
    failed = 0
    encroachment_count = 0
    rate_sum = 0.0

    try:
        service = get_land_use_service()
        store = _get_urban_store()

        for plot_req in body.plots:
            plot_id = (
                plot_req.plot_id
                or f"luc-urb-{uuid.uuid4().hex[:12]}"
            )
            buffer_km = body.buffer_km or plot_req.buffer_km

            try:
                result = service.analyze_urban_encroachment(
                    latitude=plot_req.latitude,
                    longitude=plot_req.longitude,
                    buffer_km=buffer_km,
                    date_from=plot_req.date_from,
                    date_to=plot_req.date_to,
                    polygon_wkt=plot_req.polygon_wkt,
                )

                encroachment = getattr(
                    result, "encroachment_detected", False
                )
                expansion_rate = getattr(
                    result, "urban_expansion_rate_ha_year", 0.0
                )

                urban_result = UrbanResult(
                    request_id=get_request_id(),
                    plot_id=plot_id,
                    encroachment_detected=encroachment,
                    urban_expansion_rate_ha_year=expansion_rate,
                    urban_area_start_ha=getattr(
                        result, "urban_area_start_ha", 0.0
                    ),
                    urban_area_end_ha=getattr(
                        result, "urban_area_end_ha", 0.0
                    ),
                    infrastructure_types=[],
                    pressure_corridors=[],
                    time_to_conversion_months=getattr(
                        result, "time_to_conversion_months", None
                    ),
                    buffer_km=buffer_km,
                    date_from=plot_req.date_from,
                    date_to=plot_req.date_to,
                    latitude=plot_req.latitude,
                    longitude=plot_req.longitude,
                    data_sources=getattr(result, "data_sources", []),
                    provenance_hash=getattr(
                        result, "provenance_hash", ""
                    ),
                )

                results.append(urban_result)
                successful += 1
                rate_sum += expansion_rate

                if encroachment:
                    encroachment_count += 1

                store[plot_id] = {
                    "plot_id": plot_id,
                    "response_data": urban_result.model_dump(
                        mode="json"
                    ),
                    "created_at": (
                        datetime.now(timezone.utc).isoformat()
                    ),
                    "created_by": user.user_id,
                }

            except Exception as exc:
                logger.warning(
                    "Batch urban failed for plot %s: %s",
                    plot_id,
                    exc,
                )
                failed += 1

        elapsed = time.monotonic() - start
        mean_rate = rate_sum / successful if successful > 0 else 0.0

        logger.info(
            "Batch urban completed: user=%s total=%d successful=%d "
            "failed=%d encroachment=%d mean_rate=%.2f elapsed_ms=%.1f",
            user.user_id,
            total,
            successful,
            failed,
            encroachment_count,
            mean_rate,
            elapsed * 1000,
        )

        return UrbanBatchResponse(
            request_id=get_request_id(),
            results=results,
            total=total,
            successful=successful,
            failed=failed,
            encroachment_count=encroachment_count,
            mean_expansion_rate=mean_rate,
            processing_time_ms=elapsed * 1000,
        )

    except Exception as exc:
        logger.error(
            "Batch urban failed: user=%s error=%s",
            user.user_id,
            exc,
            exc_info=True,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Batch urban analysis failed: {str(exc)}",
        )


# ---------------------------------------------------------------------------
# GET /urban/{plot_id}
# ---------------------------------------------------------------------------


@router.get(
    "/urban/{plot_id}",
    response_model=UrbanResult,
    status_code=status.HTTP_200_OK,
    summary="Get stored urban result",
    description=(
        "Retrieve a previously computed urban encroachment result "
        "by plot ID."
    ),
    responses={
        200: {"description": "Urban encroachment result"},
        404: {"model": ErrorResponse, "description": "Plot not found"},
        401: {"model": ErrorResponse, "description": "Authentication required"},
        403: {"model": ErrorResponse, "description": "Insufficient permissions"},
    },
)
async def get_urban(
    plot_id: str,
    request: Request,
    user: AuthUser = Depends(
        require_permission("eudr-luc:risk:read")
    ),
    _rate: None = Depends(rate_limit_standard),
) -> UrbanResult:
    """Retrieve a stored urban encroachment result by plot ID.

    Args:
        plot_id: Plot identifier to look up.
        user: Authenticated user with risk:read permission.

    Returns:
        UrbanResult from the store.

    Raises:
        HTTPException: 404 if plot_id not found.
    """
    plot_id = validate_plot_id(plot_id)
    store = _get_urban_store()

    if plot_id not in store:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=(
                f"No urban result found for plot_id '{plot_id}'"
            ),
        )

    record = store[plot_id]
    return UrbanResult(**record["response_data"])


# ---------------------------------------------------------------------------
# POST /batch
# ---------------------------------------------------------------------------


@router.post(
    "/batch",
    response_model=BatchJobResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Submit asynchronous batch job",
    description=(
        "Submit an asynchronous batch job for any analysis type "
        "(classification, transition, trajectory, verification, "
        "risk assessment, urban analysis, or report generation). "
        "Returns a job ID for polling status."
    ),
    responses={
        202: {"description": "Batch job accepted"},
        400: {"model": ErrorResponse, "description": "Invalid request"},
        401: {"model": ErrorResponse, "description": "Authentication required"},
        403: {"model": ErrorResponse, "description": "Insufficient permissions"},
        429: {"model": ErrorResponse, "description": "Rate limit exceeded"},
    },
)
async def submit_batch_job(
    body: BatchJobSubmitRequest,
    request: Request,
    user: AuthUser = Depends(
        require_permission("eudr-luc:batch:write")
    ),
    _rate: None = Depends(rate_limit_heavy),
) -> BatchJobResponse:
    """Submit an asynchronous batch job.

    Creates a batch job record and returns immediately with a job ID.
    The job is processed asynchronously by the background worker.

    Args:
        body: Batch job submission with type, parameters, and options.
        user: Authenticated user with batch:write permission.

    Returns:
        BatchJobResponse with job ID and initial status.
    """
    job_id = f"job-{uuid.uuid4().hex[:12]}"
    now = datetime.now(timezone.utc).replace(microsecond=0)

    logger.info(
        "Batch job submitted: user=%s job_id=%s type=%s priority=%d",
        user.user_id,
        job_id,
        body.job_type.value,
        body.priority,
    )

    # Count items from parameters
    total_items = 0
    plots = body.parameters.get("plots", [])
    if isinstance(plots, list):
        total_items = len(plots)

    response = BatchJobResponse(
        request_id=get_request_id(),
        job_id=job_id,
        job_type=body.job_type.value,
        status=BatchJobStatus.PENDING,
        progress_pct=0.0,
        total_items=total_items,
        completed_items=0,
        failed_items=0,
        submitted_at=now,
        tags=body.tags,
    )

    # Store job record
    store = _get_batch_job_store()
    store[job_id] = {
        "job_id": job_id,
        "job_type": body.job_type.value,
        "status": "pending",
        "parameters": body.parameters,
        "priority": body.priority,
        "callback_url": body.callback_url,
        "tags": body.tags,
        "total_items": total_items,
        "completed_items": 0,
        "failed_items": 0,
        "submitted_at": now.isoformat(),
        "submitted_by": user.user_id,
        "response_data": response.model_dump(mode="json"),
    }

    return response


# ---------------------------------------------------------------------------
# DELETE /batch/{batch_id}
# ---------------------------------------------------------------------------


@router.delete(
    "/batch/{batch_id}",
    response_model=BatchJobResponse,
    status_code=status.HTTP_200_OK,
    summary="Cancel batch job",
    description=(
        "Cancel a pending or processing batch job. Completed and "
        "already-cancelled jobs cannot be cancelled."
    ),
    responses={
        200: {"description": "Job cancelled"},
        404: {"model": ErrorResponse, "description": "Job not found"},
        409: {"model": ErrorResponse, "description": "Job cannot be cancelled"},
        401: {"model": ErrorResponse, "description": "Authentication required"},
        403: {"model": ErrorResponse, "description": "Insufficient permissions"},
    },
)
async def cancel_batch_job(
    batch_id: str,
    request: Request,
    user: AuthUser = Depends(
        require_permission("eudr-luc:batch:write")
    ),
    _rate: None = Depends(rate_limit_standard),
) -> BatchJobResponse:
    """Cancel a pending or processing batch job.

    Args:
        batch_id: Batch job ID to cancel.
        user: Authenticated user with batch:write permission.

    Returns:
        Updated BatchJobResponse with cancelled status.

    Raises:
        HTTPException: 404 if not found, 409 if not cancellable.
    """
    store = _get_batch_job_store()

    if batch_id not in store:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Batch job '{batch_id}' not found",
        )

    job = store[batch_id]
    current_status = job["status"]

    if current_status in ("completed", "cancelled", "failed"):
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=(
                f"Cannot cancel job with status '{current_status}'. "
                "Only pending or processing jobs can be cancelled."
            ),
        )

    # Update job status
    now = datetime.now(timezone.utc).replace(microsecond=0)
    job["status"] = "cancelled"

    logger.info(
        "Batch job cancelled: user=%s job_id=%s previous_status=%s",
        user.user_id,
        batch_id,
        current_status,
    )

    return BatchJobResponse(
        request_id=get_request_id(),
        job_id=batch_id,
        job_type=job["job_type"],
        status=BatchJobStatus.CANCELLED,
        progress_pct=0.0,
        total_items=job.get("total_items", 0),
        completed_items=job.get("completed_items", 0),
        failed_items=job.get("failed_items", 0),
        submitted_at=datetime.fromisoformat(job["submitted_at"]),
        completed_at=now,
        tags=job.get("tags", []),
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

__all__ = ["router"]
