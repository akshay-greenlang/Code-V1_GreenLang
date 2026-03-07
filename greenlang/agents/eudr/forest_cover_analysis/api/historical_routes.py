# -*- coding: utf-8 -*-
"""
Historical Routes - AGENT-EUDR-004 Forest Cover Analysis API

Endpoints for historical forest cover reconstruction including single-plot
reconstruction at the EUDR cutoff date, batch processing, stored result
retrieval, cutoff vs current comparison, and data source provenance.

Endpoints:
    POST /reconstruct        - Reconstruct forest cover at cutoff date
    POST /batch              - Batch historical reconstruction
    GET  /{plot_id}          - Get stored reconstruction result
    POST /compare            - Compare cutoff vs current forest cover
    GET  /{plot_id}/sources  - Get data sources used for reconstruction

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

from fastapi import APIRouter, Depends, HTTPException, Request, status

from greenlang.agents.eudr.forest_cover_analysis.api.dependencies import (
    AuthUser,
    ErrorResponse,
    get_historical_engine,
    get_request_id,
    rate_limit_heavy,
    rate_limit_standard,
    rate_limit_write,
    require_permission,
)
from greenlang.agents.eudr.forest_cover_analysis.api.schemas import (
    BatchReconstructRequest,
    CompareHistoricalRequest,
    DataSourceInfo,
    DataSourcesResponse,
    HistoricalComparisonResponse,
    HistoricalCoverResponse,
    ReconstructHistoryRequest,
)

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Historical Reconstruction"])


# ---------------------------------------------------------------------------
# POST /reconstruct
# ---------------------------------------------------------------------------


@router.post(
    "/reconstruct",
    response_model=HistoricalCoverResponse,
    status_code=status.HTTP_200_OK,
    summary="Reconstruct forest cover at cutoff date",
    description=(
        "Reconstruct the forest cover state at the EUDR cutoff date "
        "(default: 2020-12-31) by compositing multi-temporal satellite "
        "imagery within a configurable window (default: 3 years). "
        "Returns forest cover percentage, canopy density, NDVI, and "
        "scene provenance."
    ),
    responses={
        200: {"description": "Historical reconstruction result"},
        400: {"model": ErrorResponse, "description": "Invalid request"},
        401: {"model": ErrorResponse, "description": "Authentication required"},
        403: {"model": ErrorResponse, "description": "Insufficient permissions"},
        429: {"model": ErrorResponse, "description": "Rate limit exceeded"},
    },
)
async def reconstruct_history(
    body: ReconstructHistoryRequest,
    request: Request,
    user: AuthUser = Depends(
        require_permission("eudr-fca:historical:write")
    ),
    _rate: None = Depends(rate_limit_write),
) -> HistoricalCoverResponse:
    """Reconstruct forest cover at the EUDR cutoff date.

    Composites satellite imagery within the specified window around
    the target date to estimate historical forest cover, canopy density,
    and vegetation indices.

    Args:
        body: Reconstruction request with plot polygon and target date.
        user: Authenticated user with historical:write permission.

    Returns:
        HistoricalCoverResponse with reconstructed cover metrics.

    Raises:
        HTTPException: 400 if request invalid, 500 on processing error.
    """
    start = time.monotonic()
    reconstruction_id = f"rec-{uuid.uuid4().hex[:12]}"

    # Calculate window dates
    window_start = date(
        body.target_date.year - body.window_years + 1, 1, 1
    )
    window_end = body.target_date

    logger.info(
        "Historical reconstruction: user=%s plot_id=%s target=%s "
        "window=%s..%s window_years=%d",
        user.user_id,
        body.plot_id,
        body.target_date,
        window_start,
        window_end,
        body.window_years,
    )

    try:
        engine = get_historical_engine()

        result = engine.reconstruct(
            plot_id=body.plot_id,
            polygon_wkt=body.polygon_wkt,
            target_date=body.target_date,
            window_years=body.window_years,
        )

        elapsed = time.monotonic() - start
        logger.info(
            "Historical reconstruction completed: user=%s plot_id=%s "
            "forest_cover=%.1f%% scenes=%d elapsed_ms=%.1f",
            user.user_id,
            body.plot_id,
            getattr(result, "forest_cover_pct", 0.0),
            getattr(result, "scenes_composited", 0),
            elapsed * 1000,
        )

        return HistoricalCoverResponse(
            request_id=get_request_id(),
            reconstruction_id=getattr(
                result, "reconstruction_id", reconstruction_id
            ),
            plot_id=body.plot_id,
            target_date=body.target_date,
            window_start=window_start,
            window_end=window_end,
            forest_cover_pct=getattr(result, "forest_cover_pct", 0.0),
            canopy_density_pct=getattr(result, "canopy_density_pct", 0.0),
            forest_area_ha=getattr(result, "forest_area_ha", 0.0),
            non_forest_area_ha=getattr(result, "non_forest_area_ha", 0.0),
            ndvi_mean=getattr(result, "ndvi_mean", 0.0),
            forest_type=getattr(result, "forest_type", None),
            scenes_composited=getattr(result, "scenes_composited", 0),
            cloud_free_coverage_pct=getattr(
                result, "cloud_free_coverage_pct", 0.0
            ),
            data_sources=getattr(result, "data_sources", []),
            confidence=getattr(result, "confidence", 0.0),
            processing_time_ms=elapsed * 1000,
            provenance_hash=getattr(result, "provenance_hash", ""),
        )

    except ValueError as exc:
        logger.warning(
            "Historical reconstruction error: user=%s plot_id=%s error=%s",
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
            "Historical reconstruction failed: user=%s plot_id=%s error=%s",
            user.user_id,
            body.plot_id,
            exc,
            exc_info=True,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Historical reconstruction failed due to an internal error",
        )


# ---------------------------------------------------------------------------
# POST /batch
# ---------------------------------------------------------------------------


@router.post(
    "/batch",
    response_model=List[HistoricalCoverResponse],
    status_code=status.HTTP_200_OK,
    summary="Batch historical reconstruction",
    description=(
        "Reconstruct forest cover at the cutoff date for multiple plots. "
        "Supports up to 5,000 plots per batch."
    ),
    responses={
        200: {"description": "Batch reconstruction results"},
        400: {"model": ErrorResponse, "description": "Invalid request"},
        401: {"model": ErrorResponse, "description": "Authentication required"},
        403: {"model": ErrorResponse, "description": "Insufficient permissions"},
        429: {"model": ErrorResponse, "description": "Rate limit exceeded"},
    },
)
async def batch_reconstruct(
    body: BatchReconstructRequest,
    request: Request,
    user: AuthUser = Depends(
        require_permission("eudr-fca:historical:write")
    ),
    _rate: None = Depends(rate_limit_heavy),
) -> List[HistoricalCoverResponse]:
    """Batch historical reconstruction for multiple plots.

    Args:
        body: Batch request with list of reconstruction requests.
        user: Authenticated user with historical:write permission.

    Returns:
        List of HistoricalCoverResponse for each plot.

    Raises:
        HTTPException: 400 if request invalid, 500 on processing error.
    """
    start = time.monotonic()

    logger.info(
        "Batch reconstruction: user=%s plots=%d",
        user.user_id,
        len(body.plots),
    )

    try:
        engine = get_historical_engine()
        results = []

        for plot_req in body.plots:
            reconstruction_id = f"rec-{uuid.uuid4().hex[:12]}"
            window_start = date(
                plot_req.target_date.year - plot_req.window_years + 1, 1, 1
            )

            try:
                result = engine.reconstruct(
                    plot_id=plot_req.plot_id,
                    polygon_wkt=plot_req.polygon_wkt,
                    target_date=plot_req.target_date,
                    window_years=plot_req.window_years,
                )

                results.append(HistoricalCoverResponse(
                    request_id=get_request_id(),
                    reconstruction_id=getattr(
                        result, "reconstruction_id", reconstruction_id
                    ),
                    plot_id=plot_req.plot_id,
                    target_date=plot_req.target_date,
                    window_start=window_start,
                    window_end=plot_req.target_date,
                    forest_cover_pct=getattr(result, "forest_cover_pct", 0.0),
                    canopy_density_pct=getattr(
                        result, "canopy_density_pct", 0.0
                    ),
                    confidence=getattr(result, "confidence", 0.0),
                    data_sources=getattr(result, "data_sources", []),
                    provenance_hash=getattr(result, "provenance_hash", ""),
                ))
            except Exception as plot_exc:
                logger.warning(
                    "Batch reconstruct: plot %s failed: %s",
                    plot_req.plot_id,
                    plot_exc,
                )
                results.append(HistoricalCoverResponse(
                    request_id=get_request_id(),
                    reconstruction_id=reconstruction_id,
                    plot_id=plot_req.plot_id,
                    target_date=plot_req.target_date,
                    window_start=window_start,
                    window_end=plot_req.target_date,
                    forest_cover_pct=0.0,
                    confidence=0.0,
                ))

        elapsed = time.monotonic() - start
        logger.info(
            "Batch reconstruction completed: user=%s plots=%d "
            "elapsed_ms=%.1f",
            user.user_id,
            len(results),
            elapsed * 1000,
        )

        return results

    except Exception as exc:
        logger.error(
            "Batch reconstruction failed: user=%s error=%s",
            user.user_id,
            exc,
            exc_info=True,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Batch reconstruction failed due to an internal error",
        )


# ---------------------------------------------------------------------------
# GET /{plot_id}
# ---------------------------------------------------------------------------


@router.get(
    "/{plot_id}",
    response_model=HistoricalCoverResponse,
    summary="Get stored reconstruction result",
    description=(
        "Retrieve the most recent stored historical reconstruction "
        "for a production plot."
    ),
    responses={
        200: {"description": "Stored reconstruction result"},
        401: {"model": ErrorResponse, "description": "Authentication required"},
        403: {"model": ErrorResponse, "description": "Insufficient permissions"},
        404: {"model": ErrorResponse, "description": "Reconstruction not found"},
    },
)
async def get_reconstruction(
    plot_id: str,
    request: Request,
    user: AuthUser = Depends(
        require_permission("eudr-fca:historical:read")
    ),
    _rate: None = Depends(rate_limit_standard),
) -> HistoricalCoverResponse:
    """Get the most recent stored reconstruction for a plot.

    Args:
        plot_id: Plot identifier.
        user: Authenticated user with historical:read permission.

    Returns:
        HistoricalCoverResponse with stored reconstruction data.

    Raises:
        HTTPException: 404 if reconstruction not found.
    """
    logger.info(
        "Reconstruction retrieval: user=%s plot_id=%s",
        user.user_id,
        plot_id,
    )

    try:
        engine = get_historical_engine()
        result = engine.get_result(plot_id=plot_id)

        if result is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"No reconstruction found for plot {plot_id}",
            )

        target = getattr(result, "target_date", date(2020, 12, 31))
        window_years = getattr(result, "window_years", 3)
        window_start = date(target.year - window_years + 1, 1, 1)

        return HistoricalCoverResponse(
            request_id=get_request_id(),
            reconstruction_id=getattr(result, "reconstruction_id", ""),
            plot_id=plot_id,
            target_date=target,
            window_start=window_start,
            window_end=target,
            forest_cover_pct=getattr(result, "forest_cover_pct", 0.0),
            canopy_density_pct=getattr(result, "canopy_density_pct", 0.0),
            forest_area_ha=getattr(result, "forest_area_ha", 0.0),
            non_forest_area_ha=getattr(result, "non_forest_area_ha", 0.0),
            ndvi_mean=getattr(result, "ndvi_mean", 0.0),
            forest_type=getattr(result, "forest_type", None),
            scenes_composited=getattr(result, "scenes_composited", 0),
            cloud_free_coverage_pct=getattr(
                result, "cloud_free_coverage_pct", 0.0
            ),
            data_sources=getattr(result, "data_sources", []),
            confidence=getattr(result, "confidence", 0.0),
            timestamp=getattr(result, "timestamp", datetime.now(timezone.utc)),
            provenance_hash=getattr(result, "provenance_hash", ""),
        )

    except HTTPException:
        raise
    except Exception as exc:
        logger.error(
            "Reconstruction retrieval failed: user=%s plot_id=%s error=%s",
            user.user_id,
            plot_id,
            exc,
            exc_info=True,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Reconstruction retrieval failed due to an internal error",
        )


# ---------------------------------------------------------------------------
# POST /compare
# ---------------------------------------------------------------------------


@router.post(
    "/compare",
    response_model=HistoricalComparisonResponse,
    status_code=status.HTTP_200_OK,
    summary="Compare cutoff vs current forest cover",
    description=(
        "Compare forest cover between the EUDR cutoff date and the "
        "current state. Returns absolute and relative change with "
        "deforestation and degradation detection."
    ),
    responses={
        200: {"description": "Historical comparison result"},
        400: {"model": ErrorResponse, "description": "Invalid request"},
        401: {"model": ErrorResponse, "description": "Authentication required"},
        403: {"model": ErrorResponse, "description": "Insufficient permissions"},
        429: {"model": ErrorResponse, "description": "Rate limit exceeded"},
    },
)
async def compare_historical(
    body: CompareHistoricalRequest,
    request: Request,
    user: AuthUser = Depends(
        require_permission("eudr-fca:historical:write")
    ),
    _rate: None = Depends(rate_limit_write),
) -> HistoricalComparisonResponse:
    """Compare forest cover between cutoff date and current state.

    Args:
        body: Comparison request with plot polygon and dates.
        user: Authenticated user with historical:write permission.

    Returns:
        HistoricalComparisonResponse with change metrics.

    Raises:
        HTTPException: 400 if request invalid, 500 on processing error.
    """
    start = time.monotonic()
    current = body.current_date or date.today()

    logger.info(
        "Historical comparison: user=%s plot_id=%s cutoff=%s current=%s",
        user.user_id,
        body.plot_id,
        body.cutoff_date,
        current,
    )

    try:
        engine = get_historical_engine()

        result = engine.compare(
            plot_id=body.plot_id,
            polygon_wkt=body.polygon_wkt,
            cutoff_date=body.cutoff_date,
            current_date=current,
        )

        elapsed = time.monotonic() - start
        logger.info(
            "Historical comparison completed: user=%s plot_id=%s "
            "change=%.1f%% deforestation=%s elapsed_ms=%.1f",
            user.user_id,
            body.plot_id,
            getattr(result, "forest_cover_change_pct", 0.0),
            getattr(result, "deforestation_detected", False),
            elapsed * 1000,
        )

        return HistoricalComparisonResponse(
            request_id=get_request_id(),
            plot_id=body.plot_id,
            cutoff_date=body.cutoff_date,
            current_date=current,
            cutoff_forest_cover_pct=getattr(
                result, "cutoff_forest_cover_pct", 0.0
            ),
            current_forest_cover_pct=getattr(
                result, "current_forest_cover_pct", 0.0
            ),
            forest_cover_change_pct=getattr(
                result, "forest_cover_change_pct", 0.0
            ),
            cutoff_density_pct=getattr(result, "cutoff_density_pct", 0.0),
            current_density_pct=getattr(result, "current_density_pct", 0.0),
            deforestation_detected=getattr(
                result, "deforestation_detected", False
            ),
            degradation_detected=getattr(
                result, "degradation_detected", False
            ),
            change_classification=getattr(
                result, "change_classification", "no_change"
            ),
            confidence=getattr(result, "confidence", 0.0),
            processing_time_ms=elapsed * 1000,
            provenance_hash=getattr(result, "provenance_hash", ""),
        )

    except ValueError as exc:
        logger.warning(
            "Historical comparison error: user=%s plot_id=%s error=%s",
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
            "Historical comparison failed: user=%s plot_id=%s error=%s",
            user.user_id,
            body.plot_id,
            exc,
            exc_info=True,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Historical comparison failed due to an internal error",
        )


# ---------------------------------------------------------------------------
# GET /{plot_id}/sources
# ---------------------------------------------------------------------------


@router.get(
    "/{plot_id}/sources",
    response_model=DataSourcesResponse,
    summary="Get data sources used for reconstruction",
    description=(
        "Retrieve the list of data sources (satellite scenes, LiDAR, "
        "radar) used in the historical reconstruction for a plot."
    ),
    responses={
        200: {"description": "Data sources used"},
        401: {"model": ErrorResponse, "description": "Authentication required"},
        403: {"model": ErrorResponse, "description": "Insufficient permissions"},
        404: {"model": ErrorResponse, "description": "No sources found"},
    },
)
async def get_sources(
    plot_id: str,
    request: Request,
    user: AuthUser = Depends(
        require_permission("eudr-fca:historical:read")
    ),
    _rate: None = Depends(rate_limit_standard),
) -> DataSourcesResponse:
    """Get data sources used for a plot's historical reconstruction.

    Args:
        plot_id: Plot identifier.
        user: Authenticated user with historical:read permission.

    Returns:
        DataSourcesResponse with data source provenance.

    Raises:
        HTTPException: 404 if no sources found.
    """
    start = time.monotonic()

    logger.info(
        "Data sources retrieval: user=%s plot_id=%s",
        user.user_id,
        plot_id,
    )

    try:
        engine = get_historical_engine()
        result = engine.get_sources(plot_id=plot_id)

        if result is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"No data sources found for plot {plot_id}",
            )

        raw_sources = getattr(result, "sources", [])
        sources = [
            DataSourceInfo(
                source_id=getattr(s, "source_id", ""),
                source_type=getattr(s, "source_type", "satellite"),
                provider=getattr(s, "provider", ""),
                acquisition_date=getattr(s, "acquisition_date", None),
                spatial_resolution_m=getattr(s, "spatial_resolution_m", 0.0),
                cloud_cover_pct=getattr(s, "cloud_cover_pct", 0.0),
                quality_score=getattr(s, "quality_score", 0.0),
                bands_used=getattr(s, "bands_used", []),
            )
            for s in raw_sources
        ]

        elapsed = time.monotonic() - start

        return DataSourcesResponse(
            request_id=get_request_id(),
            plot_id=plot_id,
            sources=sources,
            total_sources=len(sources),
            primary_source=getattr(result, "primary_source", None),
            processing_time_ms=elapsed * 1000,
            provenance_hash=getattr(result, "provenance_hash", ""),
        )

    except HTTPException:
        raise
    except Exception as exc:
        logger.error(
            "Data sources retrieval failed: user=%s plot_id=%s error=%s",
            user.user_id,
            plot_id,
            exc,
            exc_info=True,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Data sources retrieval failed due to an internal error",
        )
