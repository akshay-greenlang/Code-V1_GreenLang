# -*- coding: utf-8 -*-
"""
Trajectory Routes - AGENT-EUDR-005 Land Use Change Detector API

Endpoints for temporal land use change trajectory analysis including
single-plot analysis, batch processing, and stored result retrieval.

Endpoints:
    POST /analyze       - Analyze trajectory for a single plot
    POST /batch         - Batch trajectory analysis
    GET  /{plot_id}     - Get stored trajectory result

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
    get_trajectory_engine,
    rate_limit_heavy,
    rate_limit_standard,
    rate_limit_write,
    require_permission,
    validate_plot_id,
)
from greenlang.agents.eudr.land_use_change.api.schemas import (
    ChangeDate,
    LandUseCategory,
    NDVIDataPoint,
    TrajectoryAnalyzeRequest,
    TrajectoryBatchRequest,
    TrajectoryBatchResponse,
    TrajectoryResult,
    TrajectoryType,
    VisualizationData,
)

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Trajectory Analysis"])

# ---------------------------------------------------------------------------
# In-memory result store (replaced by database in production)
# ---------------------------------------------------------------------------

_trajectory_store: Dict[str, Dict[str, Any]] = {}


def _get_trajectory_store() -> Dict[str, Dict[str, Any]]:
    """Return the trajectory store. Replaceable for testing."""
    return _trajectory_store


# ---------------------------------------------------------------------------
# POST /analyze
# ---------------------------------------------------------------------------


@router.post(
    "/analyze",
    response_model=TrajectoryResult,
    status_code=status.HTTP_200_OK,
    summary="Analyze temporal trajectory",
    description=(
        "Analyze the temporal land use change trajectory for a single "
        "plot over a specified date range. Returns trajectory classification "
        "(stable, abrupt_change, gradual_change, oscillating, recovery), "
        "detected change points, NDVI time series, and visualization data. "
        "Requires a minimum temporal depth of 3 years."
    ),
    responses={
        200: {"description": "Trajectory analysis result"},
        400: {"model": ErrorResponse, "description": "Invalid request"},
        401: {"model": ErrorResponse, "description": "Authentication required"},
        403: {"model": ErrorResponse, "description": "Insufficient permissions"},
        429: {"model": ErrorResponse, "description": "Rate limit exceeded"},
    },
)
async def analyze_trajectory(
    body: TrajectoryAnalyzeRequest,
    request: Request,
    user: AuthUser = Depends(
        require_permission("eudr-luc:trajectories:write")
    ),
    _rate: None = Depends(rate_limit_write),
) -> TrajectoryResult:
    """Analyze temporal trajectory for a single plot.

    Invokes the TemporalTrajectoryAnalyzer engine to compute NDVI
    time series, detect change points, and classify the overall
    trajectory pattern.

    Args:
        body: Trajectory request with coordinates, date range, and time step.
        user: Authenticated user with trajectories:write permission.

    Returns:
        TrajectoryResult with classification, change dates, and NDVI series.

    Raises:
        HTTPException: 400 if request invalid, 500 on processing error.
    """
    start = time.monotonic()
    plot_id = body.plot_id or f"luc-tj-{uuid.uuid4().hex[:12]}"

    logger.info(
        "Trajectory analysis: user=%s plot=%s lat=%.6f lon=%.6f "
        "from=%s to=%s step=%d months",
        user.user_id,
        plot_id,
        body.latitude,
        body.longitude,
        body.date_from,
        body.date_to,
        body.time_step_months,
    )

    try:
        engine = get_trajectory_engine()

        result = engine.analyze(
            latitude=body.latitude,
            longitude=body.longitude,
            date_from=body.date_from,
            date_to=body.date_to,
            time_step_months=body.time_step_months,
            polygon_wkt=body.polygon_wkt,
        )

        elapsed = time.monotonic() - start

        # Build change dates
        change_dates = []
        raw_changes = getattr(result, "change_dates", [])
        for cd in raw_changes:
            change_dates.append(
                ChangeDate(
                    date=getattr(cd, "date", None),
                    from_category=getattr(
                        cd, "from_category", LandUseCategory.OTHER
                    ),
                    to_category=getattr(
                        cd, "to_category", LandUseCategory.OTHER
                    ),
                    magnitude=getattr(cd, "magnitude", 0.0),
                    confidence=getattr(cd, "confidence", 0.0),
                )
            )

        # Build NDVI series
        ndvi_series = []
        raw_ndvi = getattr(result, "ndvi_series", [])
        for point in raw_ndvi:
            ndvi_series.append(
                NDVIDataPoint(
                    date=getattr(point, "date", None),
                    ndvi=getattr(point, "ndvi", 0.0),
                    quality_flag=getattr(point, "quality_flag", "good"),
                    source=getattr(point, "source", ""),
                )
            )

        # Build visualization data
        viz_data = None
        raw_viz = getattr(result, "visualization_data", None)
        if raw_viz is not None:
            viz_data = VisualizationData(
                time_labels=getattr(raw_viz, "time_labels", []),
                ndvi_values=getattr(raw_viz, "ndvi_values", []),
                category_labels=getattr(raw_viz, "category_labels", []),
                change_markers=getattr(raw_viz, "change_markers", []),
            )

        trajectory_type = getattr(
            result, "trajectory_type", TrajectoryType.STABLE
        )

        response = TrajectoryResult(
            request_id=get_request_id(),
            plot_id=plot_id,
            trajectory_type=trajectory_type,
            change_dates=change_dates,
            confidence=getattr(result, "confidence", 0.0),
            ndvi_series=ndvi_series,
            mean_ndvi=getattr(result, "mean_ndvi", 0.0),
            ndvi_trend=getattr(result, "ndvi_trend", 0.0),
            ndvi_variance=getattr(result, "ndvi_variance", 0.0),
            visualization_data=viz_data,
            date_from=body.date_from,
            date_to=body.date_to,
            time_steps=getattr(result, "time_steps", len(ndvi_series)),
            latitude=body.latitude,
            longitude=body.longitude,
            data_sources=getattr(result, "data_sources", []),
            processing_time_ms=elapsed * 1000,
            provenance_hash=getattr(result, "provenance_hash", ""),
        )

        # Store for retrieval
        store = _get_trajectory_store()
        store[plot_id] = {
            "plot_id": plot_id,
            "response_data": response.model_dump(mode="json"),
            "created_at": datetime.now(timezone.utc).isoformat(),
            "created_by": user.user_id,
        }

        logger.info(
            "Trajectory analysis completed: user=%s plot=%s "
            "type=%s changes=%d time_steps=%d confidence=%.2f "
            "elapsed_ms=%.1f",
            user.user_id,
            plot_id,
            getattr(trajectory_type, "value", trajectory_type),
            len(change_dates),
            response.time_steps,
            response.confidence,
            elapsed * 1000,
        )

        return response

    except ValueError as exc:
        logger.warning(
            "Trajectory error: user=%s plot=%s error=%s",
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
            "Trajectory analysis failed: user=%s plot=%s error=%s",
            user.user_id,
            plot_id,
            exc,
            exc_info=True,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Trajectory analysis failed: {str(exc)}",
        )


# ---------------------------------------------------------------------------
# POST /batch
# ---------------------------------------------------------------------------


@router.post(
    "/batch",
    response_model=TrajectoryBatchResponse,
    status_code=status.HTTP_200_OK,
    summary="Batch trajectory analysis",
    description=(
        "Analyze temporal trajectories for multiple plots in a single "
        "request. Supports up to 5000 plots per batch. Returns per-plot "
        "trajectory classifications and aggregate statistics."
    ),
    responses={
        200: {"description": "Batch trajectory results"},
        400: {"model": ErrorResponse, "description": "Invalid request"},
        401: {"model": ErrorResponse, "description": "Authentication required"},
        403: {"model": ErrorResponse, "description": "Insufficient permissions"},
        429: {"model": ErrorResponse, "description": "Rate limit exceeded"},
    },
)
async def analyze_batch(
    body: TrajectoryBatchRequest,
    request: Request,
    user: AuthUser = Depends(
        require_permission("eudr-luc:trajectories:write")
    ),
    _rate: None = Depends(rate_limit_heavy),
) -> TrajectoryBatchResponse:
    """Batch analyze trajectories for multiple plots.

    Args:
        body: Batch request with list of plots and optional global dates.
        user: Authenticated user with trajectories:write permission.

    Returns:
        TrajectoryBatchResponse with results and aggregate statistics.

    Raises:
        HTTPException: 400 if request invalid, 500 on processing error.
    """
    start = time.monotonic()
    total = len(body.plots)

    logger.info(
        "Batch trajectory analysis: user=%s plots=%d",
        user.user_id,
        total,
    )

    results: List[TrajectoryResult] = []
    successful = 0
    failed = 0
    stable_count = 0
    changed_count = 0
    trajectory_counts: Dict[str, int] = {}

    try:
        engine = get_trajectory_engine()
        store = _get_trajectory_store()

        for plot_req in body.plots:
            plot_id = (
                plot_req.plot_id or f"luc-tj-{uuid.uuid4().hex[:12]}"
            )
            date_from = body.date_from or plot_req.date_from
            date_to = body.date_to or plot_req.date_to

            try:
                result = engine.analyze(
                    latitude=plot_req.latitude,
                    longitude=plot_req.longitude,
                    date_from=date_from,
                    date_to=date_to,
                    time_step_months=plot_req.time_step_months,
                    polygon_wkt=plot_req.polygon_wkt,
                )

                trajectory_type = getattr(
                    result, "trajectory_type", TrajectoryType.STABLE
                )
                tt_val = (
                    trajectory_type.value
                    if hasattr(trajectory_type, "value")
                    else str(trajectory_type)
                )

                trajectory = TrajectoryResult(
                    request_id=get_request_id(),
                    plot_id=plot_id,
                    trajectory_type=trajectory_type,
                    change_dates=[],
                    confidence=getattr(result, "confidence", 0.0),
                    ndvi_series=[],
                    mean_ndvi=getattr(result, "mean_ndvi", 0.0),
                    ndvi_trend=getattr(result, "ndvi_trend", 0.0),
                    ndvi_variance=getattr(result, "ndvi_variance", 0.0),
                    date_from=date_from,
                    date_to=date_to,
                    time_steps=getattr(result, "time_steps", 0),
                    latitude=plot_req.latitude,
                    longitude=plot_req.longitude,
                    data_sources=getattr(result, "data_sources", []),
                    provenance_hash=getattr(
                        result, "provenance_hash", ""
                    ),
                )

                results.append(trajectory)
                successful += 1

                trajectory_counts[tt_val] = (
                    trajectory_counts.get(tt_val, 0) + 1
                )

                if tt_val == "stable":
                    stable_count += 1
                else:
                    changed_count += 1

                store[plot_id] = {
                    "plot_id": plot_id,
                    "response_data": trajectory.model_dump(mode="json"),
                    "created_at": datetime.now(timezone.utc).isoformat(),
                    "created_by": user.user_id,
                }

            except Exception as exc:
                logger.warning(
                    "Batch trajectory failed for plot %s: %s",
                    plot_id,
                    exc,
                )
                failed += 1

        elapsed = time.monotonic() - start

        logger.info(
            "Batch trajectory completed: user=%s total=%d "
            "successful=%d failed=%d stable=%d changed=%d "
            "elapsed_ms=%.1f",
            user.user_id,
            total,
            successful,
            failed,
            stable_count,
            changed_count,
            elapsed * 1000,
        )

        return TrajectoryBatchResponse(
            request_id=get_request_id(),
            results=results,
            total=total,
            successful=successful,
            failed=failed,
            stable_count=stable_count,
            changed_count=changed_count,
            trajectory_distribution=trajectory_counts,
            processing_time_ms=elapsed * 1000,
        )

    except Exception as exc:
        logger.error(
            "Batch trajectory failed: user=%s error=%s",
            user.user_id,
            exc,
            exc_info=True,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Batch trajectory analysis failed: {str(exc)}",
        )


# ---------------------------------------------------------------------------
# GET /{plot_id}
# ---------------------------------------------------------------------------


@router.get(
    "/{plot_id}",
    response_model=TrajectoryResult,
    status_code=status.HTTP_200_OK,
    summary="Get stored trajectory result",
    description=(
        "Retrieve a previously computed trajectory analysis result "
        "by plot ID."
    ),
    responses={
        200: {"description": "Trajectory result"},
        404: {"model": ErrorResponse, "description": "Plot not found"},
        401: {"model": ErrorResponse, "description": "Authentication required"},
        403: {"model": ErrorResponse, "description": "Insufficient permissions"},
    },
)
async def get_trajectory(
    plot_id: str,
    request: Request,
    user: AuthUser = Depends(
        require_permission("eudr-luc:trajectories:read")
    ),
    _rate: None = Depends(rate_limit_standard),
) -> TrajectoryResult:
    """Retrieve a stored trajectory result by plot ID.

    Args:
        plot_id: Plot identifier to look up.
        user: Authenticated user with trajectories:read permission.

    Returns:
        TrajectoryResult from the store.

    Raises:
        HTTPException: 404 if plot_id not found.
    """
    plot_id = validate_plot_id(plot_id)
    store = _get_trajectory_store()

    if plot_id not in store:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=(
                f"No trajectory result found for plot_id '{plot_id}'"
            ),
        )

    record = store[plot_id]
    return TrajectoryResult(**record["response_data"])


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

__all__ = ["router"]
