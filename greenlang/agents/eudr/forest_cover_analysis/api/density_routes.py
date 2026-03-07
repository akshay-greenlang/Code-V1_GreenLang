# -*- coding: utf-8 -*-
"""
Density Routes - AGENT-EUDR-004 Forest Cover Analysis API

Endpoints for canopy density estimation including single-plot analysis,
batch processing, stored result retrieval, historical density time series,
and temporal density comparison.

Endpoints:
    POST /analyze         - Analyze canopy density for a single plot
    POST /batch           - Batch canopy density analysis
    GET  /{plot_id}       - Get stored density result
    GET  /{plot_id}/history - Get density history over time
    POST /compare         - Compare density between two dates

Author: GreenLang Platform Team
Date: March 2026
PRD: AGENT-EUDR-004 Forest Cover Analysis Agent (GL-EUDR-FCA-004)
"""

from __future__ import annotations

import logging
import time
import uuid
from datetime import date, datetime, timezone
from typing import Any, Dict, List

from fastapi import APIRouter, Depends, HTTPException, Query, Request, status

from greenlang.agents.eudr.forest_cover_analysis.api.dependencies import (
    AuthUser,
    ErrorResponse,
    PaginationParams,
    get_density_engine,
    get_forest_cover_service,
    get_pagination,
    get_request_id,
    rate_limit_standard,
    rate_limit_write,
    rate_limit_heavy,
    require_permission,
)
from greenlang.agents.eudr.forest_cover_analysis.api.schemas import (
    AnalyzeDensityRequest,
    BatchDensityRequest,
    CanopyDensityResponse,
    CompareDensityRequest,
    DensityComparisonResponse,
    DensityHistoryEntry,
    DensityHistoryResponse,
    PaginatedMeta,
)

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Canopy Density"])


# ---------------------------------------------------------------------------
# POST /analyze
# ---------------------------------------------------------------------------


@router.post(
    "/analyze",
    response_model=CanopyDensityResponse,
    status_code=status.HTTP_200_OK,
    summary="Analyze canopy density for a single plot",
    description=(
        "Estimate canopy density percentage for a production plot using "
        "satellite imagery. Supports multiple estimation methods: spectral, "
        "LiDAR, radar, fusion, and Hansen GFC. Returns density percentage, "
        "classification, FAO threshold compliance, and provenance hash."
    ),
    responses={
        200: {"description": "Canopy density analysis result"},
        400: {"model": ErrorResponse, "description": "Invalid request"},
        401: {"model": ErrorResponse, "description": "Authentication required"},
        403: {"model": ErrorResponse, "description": "Insufficient permissions"},
        429: {"model": ErrorResponse, "description": "Rate limit exceeded"},
    },
)
async def analyze_density(
    body: AnalyzeDensityRequest,
    request: Request,
    user: AuthUser = Depends(
        require_permission("eudr-fca:density:write")
    ),
    _rate: None = Depends(rate_limit_write),
) -> CanopyDensityResponse:
    """Analyze canopy density for a single plot.

    Estimates the canopy density percentage using the specified method
    and returns a density classification along with FAO forest threshold
    compliance status.

    Args:
        body: Density analysis request with plot polygon and method.
        user: Authenticated user with density:write permission.

    Returns:
        CanopyDensityResponse with density metrics and provenance.

    Raises:
        HTTPException: 400 if request invalid, 500 on processing error.
    """
    start = time.monotonic()
    request_id = get_request_id()

    logger.info(
        "Density analysis request: user=%s plot_id=%s method=%s biome=%s",
        user.user_id,
        body.plot_id,
        body.method.value,
        body.biome,
    )

    try:
        engine = get_density_engine()

        result = engine.analyze(
            plot_id=body.plot_id,
            polygon_wkt=body.polygon_wkt,
            imagery_date=body.imagery_date,
            method=body.method.value,
            biome=body.biome,
        )

        elapsed = time.monotonic() - start
        logger.info(
            "Density analysis completed: user=%s plot_id=%s "
            "density=%.1f%% class=%s elapsed_ms=%.1f",
            user.user_id,
            body.plot_id,
            getattr(result, "density_pct", 0.0),
            getattr(result, "density_class", "unknown"),
            elapsed * 1000,
        )

        return CanopyDensityResponse(
            request_id=request_id,
            plot_id=body.plot_id,
            density_pct=getattr(result, "density_pct", 0.0),
            density_class=getattr(result, "density_class", "unknown"),
            method=body.method.value,
            pixel_count=getattr(result, "pixel_count", 0),
            area_ha=getattr(result, "area_ha", 0.0),
            cloud_cover_pct=getattr(result, "cloud_cover_pct", 0.0),
            imagery_date=getattr(result, "imagery_date", body.imagery_date),
            biome=body.biome,
            fao_threshold_met=getattr(result, "fao_threshold_met", False),
            confidence=getattr(result, "confidence", 0.0),
            data_sources=getattr(result, "data_sources", []),
            processing_time_ms=elapsed * 1000,
            provenance_hash=getattr(result, "provenance_hash", ""),
        )

    except ValueError as exc:
        logger.warning(
            "Density analysis error: user=%s plot_id=%s error=%s",
            user.user_id,
            body.plot_id,
            exc,
        )
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        )
    except Exception as exc:
        logger.error(
            "Density analysis failed: user=%s plot_id=%s error=%s",
            user.user_id,
            body.plot_id,
            exc,
            exc_info=True,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Density analysis failed due to an internal error",
        )


# ---------------------------------------------------------------------------
# POST /batch
# ---------------------------------------------------------------------------


@router.post(
    "/batch",
    response_model=List[CanopyDensityResponse],
    status_code=status.HTTP_200_OK,
    summary="Batch canopy density analysis",
    description=(
        "Analyze canopy density for multiple plots in a single request. "
        "Supports up to 5,000 plots per batch. Returns results for each "
        "plot in the same order as the request."
    ),
    responses={
        200: {"description": "Batch density results"},
        400: {"model": ErrorResponse, "description": "Invalid request"},
        401: {"model": ErrorResponse, "description": "Authentication required"},
        403: {"model": ErrorResponse, "description": "Insufficient permissions"},
        429: {"model": ErrorResponse, "description": "Rate limit exceeded"},
    },
)
async def batch_density(
    body: BatchDensityRequest,
    request: Request,
    user: AuthUser = Depends(
        require_permission("eudr-fca:density:write")
    ),
    _rate: None = Depends(rate_limit_heavy),
) -> List[CanopyDensityResponse]:
    """Batch canopy density analysis for multiple plots.

    Processes density analysis for each plot in the request and
    returns results in the same order.

    Args:
        body: Batch request with list of plot analysis requests.
        user: Authenticated user with density:write permission.

    Returns:
        List of CanopyDensityResponse for each plot.

    Raises:
        HTTPException: 400 if request invalid, 500 on processing error.
    """
    start = time.monotonic()

    logger.info(
        "Batch density analysis: user=%s plots=%d",
        user.user_id,
        len(body.plots),
    )

    try:
        engine = get_density_engine()
        results = []

        for plot_req in body.plots:
            try:
                result = engine.analyze(
                    plot_id=plot_req.plot_id,
                    polygon_wkt=plot_req.polygon_wkt,
                    imagery_date=plot_req.imagery_date,
                    method=plot_req.method.value,
                    biome=plot_req.biome,
                )

                results.append(CanopyDensityResponse(
                    request_id=get_request_id(),
                    plot_id=plot_req.plot_id,
                    density_pct=getattr(result, "density_pct", 0.0),
                    density_class=getattr(result, "density_class", "unknown"),
                    method=plot_req.method.value,
                    pixel_count=getattr(result, "pixel_count", 0),
                    area_ha=getattr(result, "area_ha", 0.0),
                    cloud_cover_pct=getattr(result, "cloud_cover_pct", 0.0),
                    imagery_date=getattr(result, "imagery_date", None),
                    biome=plot_req.biome,
                    fao_threshold_met=getattr(result, "fao_threshold_met", False),
                    confidence=getattr(result, "confidence", 0.0),
                    data_sources=getattr(result, "data_sources", []),
                    provenance_hash=getattr(result, "provenance_hash", ""),
                ))
            except Exception as plot_exc:
                logger.warning(
                    "Batch density: plot %s failed: %s",
                    plot_req.plot_id,
                    plot_exc,
                )
                results.append(CanopyDensityResponse(
                    request_id=get_request_id(),
                    plot_id=plot_req.plot_id,
                    density_pct=0.0,
                    density_class="error",
                    method=plot_req.method.value,
                    confidence=0.0,
                ))

        elapsed = time.monotonic() - start
        logger.info(
            "Batch density completed: user=%s plots=%d elapsed_ms=%.1f",
            user.user_id,
            len(results),
            elapsed * 1000,
        )

        return results

    except Exception as exc:
        logger.error(
            "Batch density failed: user=%s error=%s",
            user.user_id,
            exc,
            exc_info=True,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Batch density analysis failed due to an internal error",
        )


# ---------------------------------------------------------------------------
# GET /{plot_id}
# ---------------------------------------------------------------------------


@router.get(
    "/{plot_id}",
    response_model=CanopyDensityResponse,
    summary="Get stored density result",
    description="Retrieve the most recent stored canopy density result for a plot.",
    responses={
        200: {"description": "Stored density result"},
        401: {"model": ErrorResponse, "description": "Authentication required"},
        403: {"model": ErrorResponse, "description": "Insufficient permissions"},
        404: {"model": ErrorResponse, "description": "Density result not found"},
    },
)
async def get_density(
    plot_id: str,
    request: Request,
    user: AuthUser = Depends(
        require_permission("eudr-fca:density:read")
    ),
    _rate: None = Depends(rate_limit_standard),
) -> CanopyDensityResponse:
    """Get the most recent stored density result for a plot.

    Args:
        plot_id: Plot identifier.
        user: Authenticated user with density:read permission.

    Returns:
        CanopyDensityResponse with stored density data.

    Raises:
        HTTPException: 404 if density result not found.
    """
    logger.info(
        "Density retrieval: user=%s plot_id=%s",
        user.user_id,
        plot_id,
    )

    try:
        engine = get_density_engine()
        result = engine.get_result(plot_id=plot_id)

        if result is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"No density result found for plot {plot_id}",
            )

        return CanopyDensityResponse(
            request_id=get_request_id(),
            plot_id=plot_id,
            density_pct=getattr(result, "density_pct", 0.0),
            density_class=getattr(result, "density_class", "unknown"),
            method=getattr(result, "method", "fusion"),
            pixel_count=getattr(result, "pixel_count", 0),
            area_ha=getattr(result, "area_ha", 0.0),
            cloud_cover_pct=getattr(result, "cloud_cover_pct", 0.0),
            imagery_date=getattr(result, "imagery_date", None),
            biome=getattr(result, "biome", None),
            fao_threshold_met=getattr(result, "fao_threshold_met", False),
            confidence=getattr(result, "confidence", 0.0),
            data_sources=getattr(result, "data_sources", []),
            timestamp=getattr(result, "timestamp", datetime.now(timezone.utc)),
            provenance_hash=getattr(result, "provenance_hash", ""),
        )

    except HTTPException:
        raise
    except Exception as exc:
        logger.error(
            "Density retrieval failed: user=%s plot_id=%s error=%s",
            user.user_id,
            plot_id,
            exc,
            exc_info=True,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Density retrieval failed due to an internal error",
        )


# ---------------------------------------------------------------------------
# GET /{plot_id}/history
# ---------------------------------------------------------------------------


@router.get(
    "/{plot_id}/history",
    response_model=DensityHistoryResponse,
    summary="Get density history over time",
    description=(
        "Retrieve the canopy density time series for a plot. Returns "
        "density observations sorted by date with trend analysis and "
        "summary statistics."
    ),
    responses={
        200: {"description": "Density history time series"},
        401: {"model": ErrorResponse, "description": "Authentication required"},
        403: {"model": ErrorResponse, "description": "Insufficient permissions"},
        404: {"model": ErrorResponse, "description": "No history found"},
    },
)
async def get_density_history(
    plot_id: str,
    request: Request,
    user: AuthUser = Depends(
        require_permission("eudr-fca:density:read")
    ),
    pagination: PaginationParams = Depends(get_pagination),
    start_date: date = Query(
        default=None,
        description="Filter history from this date",
    ),
    end_date: date = Query(
        default=None,
        description="Filter history until this date",
    ),
    _rate: None = Depends(rate_limit_standard),
) -> DensityHistoryResponse:
    """Get paginated density history time series for a plot.

    Args:
        plot_id: Plot identifier.
        pagination: Pagination parameters.
        start_date: Optional start date filter.
        end_date: Optional end date filter.
        user: Authenticated user with density:read permission.

    Returns:
        DensityHistoryResponse with time series entries and trend.

    Raises:
        HTTPException: 404 if no history found, 500 on error.
    """
    start = time.monotonic()

    logger.info(
        "Density history: user=%s plot_id=%s start=%s end=%s "
        "limit=%d offset=%d",
        user.user_id,
        plot_id,
        start_date,
        end_date,
        pagination.limit,
        pagination.offset,
    )

    try:
        engine = get_density_engine()

        history = engine.get_history(
            plot_id=plot_id,
            start_date=start_date,
            end_date=end_date,
            limit=pagination.limit,
            offset=pagination.offset,
        )

        if history is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"No density history found for plot {plot_id}",
            )

        items = getattr(history, "entries", [])
        total = getattr(history, "total", len(items))

        entries = [
            DensityHistoryEntry(
                date=getattr(entry, "date", date.today()),
                density_pct=getattr(entry, "density_pct", 0.0),
                method=getattr(entry, "method", "fusion"),
                confidence=getattr(entry, "confidence", 0.0),
                source=getattr(entry, "source", ""),
            )
            for entry in items
        ]

        elapsed = time.monotonic() - start

        return DensityHistoryResponse(
            request_id=get_request_id(),
            plot_id=plot_id,
            entries=entries,
            trend=getattr(history, "trend", "stable"),
            mean_density_pct=getattr(history, "mean_density_pct", 0.0),
            min_density_pct=getattr(history, "min_density_pct", 0.0),
            max_density_pct=getattr(history, "max_density_pct", 0.0),
            total_observations=total,
            meta=PaginatedMeta(
                total=total,
                limit=pagination.limit,
                offset=pagination.offset,
                has_more=(pagination.offset + pagination.limit) < total,
            ),
            processing_time_ms=elapsed * 1000,
            provenance_hash=getattr(history, "provenance_hash", ""),
        )

    except HTTPException:
        raise
    except Exception as exc:
        logger.error(
            "Density history failed: user=%s plot_id=%s error=%s",
            user.user_id,
            plot_id,
            exc,
            exc_info=True,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Density history retrieval failed due to an internal error",
        )


# ---------------------------------------------------------------------------
# POST /compare
# ---------------------------------------------------------------------------


@router.post(
    "/compare",
    response_model=DensityComparisonResponse,
    status_code=status.HTTP_200_OK,
    summary="Compare canopy density between two dates",
    description=(
        "Compare canopy density at two different dates for the same plot. "
        "Returns absolute and relative change with classification "
        "(gain, loss, degradation, or no change)."
    ),
    responses={
        200: {"description": "Density comparison result"},
        400: {"model": ErrorResponse, "description": "Invalid request"},
        401: {"model": ErrorResponse, "description": "Authentication required"},
        403: {"model": ErrorResponse, "description": "Insufficient permissions"},
        429: {"model": ErrorResponse, "description": "Rate limit exceeded"},
    },
)
async def compare_density(
    body: CompareDensityRequest,
    request: Request,
    user: AuthUser = Depends(
        require_permission("eudr-fca:density:write")
    ),
    _rate: None = Depends(rate_limit_write),
) -> DensityComparisonResponse:
    """Compare canopy density between two dates for a plot.

    Estimates density at both dates and computes absolute and relative
    change with a change classification.

    Args:
        body: Comparison request with plot polygon and two dates.
        user: Authenticated user with density:write permission.

    Returns:
        DensityComparisonResponse with change metrics.

    Raises:
        HTTPException: 400 if request invalid, 500 on processing error.
    """
    start = time.monotonic()

    logger.info(
        "Density comparison: user=%s plot_id=%s before=%s after=%s",
        user.user_id,
        body.plot_id,
        body.date_before,
        body.date_after,
    )

    try:
        engine = get_density_engine()

        result = engine.compare(
            plot_id=body.plot_id,
            polygon_wkt=body.polygon_wkt,
            date_before=body.date_before,
            date_after=body.date_after,
            method=body.method.value,
        )

        elapsed = time.monotonic() - start
        logger.info(
            "Density comparison completed: user=%s plot_id=%s "
            "change=%.1f%% classification=%s elapsed_ms=%.1f",
            user.user_id,
            body.plot_id,
            getattr(result, "density_change_pct", 0.0),
            getattr(result, "change_classification", "no_change"),
            elapsed * 1000,
        )

        return DensityComparisonResponse(
            request_id=get_request_id(),
            plot_id=body.plot_id,
            density_before_pct=getattr(result, "density_before_pct", 0.0),
            density_after_pct=getattr(result, "density_after_pct", 0.0),
            density_change_pct=getattr(result, "density_change_pct", 0.0),
            density_change_relative_pct=getattr(
                result, "density_change_relative_pct", 0.0
            ),
            date_before=body.date_before,
            date_after=body.date_after,
            change_classification=getattr(
                result, "change_classification", "no_change"
            ),
            method=body.method.value,
            confidence=getattr(result, "confidence", 0.0),
            processing_time_ms=elapsed * 1000,
            provenance_hash=getattr(result, "provenance_hash", ""),
        )

    except ValueError as exc:
        logger.warning(
            "Density comparison error: user=%s plot_id=%s error=%s",
            user.user_id,
            body.plot_id,
            exc,
        )
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        )
    except Exception as exc:
        logger.error(
            "Density comparison failed: user=%s plot_id=%s error=%s",
            user.user_id,
            body.plot_id,
            exc,
            exc_info=True,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Density comparison failed due to an internal error",
        )
