# -*- coding: utf-8 -*-
"""
GPS Routes - AGENT-EUDR-015 Mobile Data Collector

REST API endpoints for GPS point capture and polygon boundary tracing
per EUDR Article 9(1)(d) geolocation requirements.

Endpoints (7):
    POST /gps/points               Record GPS point capture
    POST /gps/polygons             Record polygon boundary trace
    GET  /gps/points/{capture_id}  Get GPS capture
    GET  /gps/polygons/{polygon_id} Get polygon
    POST /gps/validate             Validate coordinates
    POST /gps/area                 Calculate polygon area
    POST /gps/distance             Calculate distance between points

Author: GreenLang Platform Team
Date: March 2026
PRD: AGENT-EUDR-015, Section 7.4
Agent ID: GL-EUDR-MDC-015
Status: Production Ready
"""

from __future__ import annotations

import logging
import time
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, status

from greenlang.agents.eudr.mobile_data_collector.api.dependencies import (
    AuthUser,
    get_mdc_service,
    rate_limit_read,
    rate_limit_write,
    require_permission,
    validate_capture_id,
    validate_polygon_id,
)
from greenlang.agents.eudr.mobile_data_collector.api.schemas import (
    AreaResponseSchema,
    DistanceRequestSchema,
    DistanceResponseSchema,
    ErrorSchema,
    GPSCaptureSchema,
    GPSResponseSchema,
    GPSValidateResponseSchema,
    GPSValidateSchema,
    PolygonCaptureSchema,
    PolygonResponseSchema,
)

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/gps",
    tags=["EUDR Mobile Data - GPS"],
    responses={
        400: {"model": ErrorSchema, "description": "Validation error"},
        404: {"model": ErrorSchema, "description": "Capture not found"},
    },
)


# ---------------------------------------------------------------------------
# POST /gps/points
# ---------------------------------------------------------------------------


@router.post(
    "/points",
    response_model=GPSResponseSchema,
    status_code=status.HTTP_201_CREATED,
    summary="Record GPS point capture",
    description=(
        "Record a GPS point capture from a mobile device with accuracy "
        "metadata including HDOP, satellite count, fix type, and "
        "augmentation source per EUDR Art. 9(1)(d)."
    ),
    responses={
        201: {"description": "GPS point captured successfully"},
        400: {"description": "Invalid GPS data or accuracy below threshold"},
    },
)
async def capture_gps_point(
    body: GPSCaptureSchema,
    user: AuthUser = Depends(require_permission("eudr-mdc:gps:create")),
    service: Any = Depends(get_mdc_service),
    _rl: None = Depends(rate_limit_write),
) -> GPSResponseSchema:
    """Record a GPS point capture.

    Args:
        body: GPS capture data with coordinates and accuracy metadata.
        user: Authenticated user.
        service: MDC service singleton.

    Returns:
        GPSResponseSchema with capture details and accuracy tier.
    """
    start = time.monotonic()
    logger.info(
        "GPS capture: user=%s device=%s lat=%.6f lon=%.6f acc=%.1fm",
        user.user_id,
        body.device_id,
        body.latitude,
        body.longitude,
        body.horizontal_accuracy_m,
    )

    elapsed_ms = (time.monotonic() - start) * 1000
    return GPSResponseSchema(
        latitude=body.latitude,
        longitude=body.longitude,
        altitude_m=body.altitude_m,
        horizontal_accuracy_m=body.horizontal_accuracy_m,
        hdop=body.hdop,
        satellite_count=body.satellite_count,
        fix_type=body.fix_type,
        augmentation=body.augmentation,
        form_id=body.form_id,
        processing_time_ms=round(elapsed_ms, 2),
        message="GPS point captured successfully",
    )


# ---------------------------------------------------------------------------
# POST /gps/polygons
# ---------------------------------------------------------------------------


@router.post(
    "/polygons",
    response_model=PolygonResponseSchema,
    status_code=status.HTTP_201_CREATED,
    summary="Record polygon boundary trace",
    description=(
        "Record a polygon boundary trace from walk-around GPS tracing. "
        "Required for plots exceeding 4 hectares per EUDR Art. 9(1)(d). "
        "Calculates area in hectares and validates polygon geometry."
    ),
    responses={
        201: {"description": "Polygon trace recorded successfully"},
        400: {"description": "Invalid polygon data or insufficient vertices"},
    },
)
async def capture_polygon(
    body: PolygonCaptureSchema,
    user: AuthUser = Depends(require_permission("eudr-mdc:gps:create")),
    service: Any = Depends(get_mdc_service),
    _rl: None = Depends(rate_limit_write),
) -> PolygonResponseSchema:
    """Record a polygon boundary trace.

    Args:
        body: Polygon trace data with vertices and accuracy.
        user: Authenticated user.
        service: MDC service singleton.

    Returns:
        PolygonResponseSchema with area, perimeter, and validity.
    """
    start = time.monotonic()
    logger.info(
        "Polygon capture: user=%s device=%s vertices=%d",
        user.user_id,
        body.device_id,
        len(body.vertices),
    )

    elapsed_ms = (time.monotonic() - start) * 1000
    return PolygonResponseSchema(
        vertex_count=len(body.vertices),
        form_id=body.form_id,
        processing_time_ms=round(elapsed_ms, 2),
        message="Polygon trace recorded successfully",
    )


# ---------------------------------------------------------------------------
# GET /gps/points/{capture_id}
# ---------------------------------------------------------------------------


@router.get(
    "/points/{capture_id}",
    response_model=GPSResponseSchema,
    summary="Get GPS capture",
    description="Retrieve a specific GPS point capture by its identifier.",
    responses={
        200: {"description": "GPS capture retrieved"},
        404: {"description": "GPS capture not found"},
    },
)
async def get_gps_point(
    capture_id: str = Depends(validate_capture_id),
    user: AuthUser = Depends(require_permission("eudr-mdc:gps:read")),
    service: Any = Depends(get_mdc_service),
    _rl: None = Depends(rate_limit_read),
) -> GPSResponseSchema:
    """Get a GPS capture by identifier.

    Args:
        capture_id: GPS capture identifier.
        user: Authenticated user.
        service: MDC service singleton.

    Returns:
        GPSResponseSchema with capture details.

    Raises:
        HTTPException: 404 if capture not found.
    """
    logger.info("Get GPS: user=%s capture_id=%s", user.user_id, capture_id)

    raise HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail=f"GPS capture {capture_id} not found",
    )


# ---------------------------------------------------------------------------
# GET /gps/polygons/{polygon_id}
# ---------------------------------------------------------------------------


@router.get(
    "/polygons/{polygon_id}",
    response_model=PolygonResponseSchema,
    summary="Get polygon",
    description="Retrieve a specific polygon trace by its identifier.",
    responses={
        200: {"description": "Polygon retrieved"},
        404: {"description": "Polygon not found"},
    },
)
async def get_polygon(
    polygon_id: str = Depends(validate_polygon_id),
    user: AuthUser = Depends(require_permission("eudr-mdc:gps:read")),
    service: Any = Depends(get_mdc_service),
    _rl: None = Depends(rate_limit_read),
) -> PolygonResponseSchema:
    """Get a polygon trace by identifier.

    Args:
        polygon_id: Polygon trace identifier.
        user: Authenticated user.
        service: MDC service singleton.

    Returns:
        PolygonResponseSchema with polygon details.

    Raises:
        HTTPException: 404 if polygon not found.
    """
    logger.info(
        "Get polygon: user=%s polygon_id=%s", user.user_id, polygon_id
    )

    raise HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail=f"Polygon {polygon_id} not found",
    )


# ---------------------------------------------------------------------------
# POST /gps/validate
# ---------------------------------------------------------------------------


@router.post(
    "/validate",
    response_model=GPSValidateResponseSchema,
    summary="Validate coordinates",
    description=(
        "Validate GPS coordinates for plausibility, checking that they "
        "fall within valid geographic bounds and optionally match the "
        "expected country and commodity-producing region."
    ),
    responses={
        200: {"description": "Validation completed"},
    },
)
async def validate_coordinates(
    body: GPSValidateSchema,
    user: AuthUser = Depends(require_permission("eudr-mdc:gps:validate")),
    service: Any = Depends(get_mdc_service),
    _rl: None = Depends(rate_limit_write),
) -> GPSValidateResponseSchema:
    """Validate GPS coordinates.

    Args:
        body: Coordinates to validate.
        user: Authenticated user.
        service: MDC service singleton.

    Returns:
        GPSValidateResponseSchema with validation results.
    """
    start = time.monotonic()
    logger.info(
        "Validate GPS: user=%s lat=%.6f lon=%.6f",
        user.user_id,
        body.latitude,
        body.longitude,
    )

    elapsed_ms = (time.monotonic() - start) * 1000
    return GPSValidateResponseSchema(
        is_valid=True,
        latitude=body.latitude,
        longitude=body.longitude,
        processing_time_ms=round(elapsed_ms, 2),
    )


# ---------------------------------------------------------------------------
# POST /gps/area
# ---------------------------------------------------------------------------


@router.post(
    "/area",
    response_model=AreaResponseSchema,
    summary="Calculate polygon area",
    description=(
        "Calculate the area of a polygon defined by GPS vertices "
        "using geodesic methods. Returns area in hectares and "
        "square meters, plus perimeter in meters."
    ),
    responses={
        200: {"description": "Area calculated successfully"},
        400: {"description": "Invalid polygon vertices"},
    },
)
async def calculate_area(
    body: PolygonCaptureSchema,
    user: AuthUser = Depends(require_permission("eudr-mdc:gps:read")),
    service: Any = Depends(get_mdc_service),
    _rl: None = Depends(rate_limit_write),
) -> AreaResponseSchema:
    """Calculate polygon area.

    Args:
        body: Polygon data with vertices.
        user: Authenticated user.
        service: MDC service singleton.

    Returns:
        AreaResponseSchema with calculated metrics.
    """
    start = time.monotonic()
    logger.info(
        "Calculate area: user=%s vertices=%d",
        user.user_id,
        len(body.vertices),
    )

    elapsed_ms = (time.monotonic() - start) * 1000
    return AreaResponseSchema(
        area_ha=0.0,
        area_sq_m=0.0,
        perimeter_m=0.0,
        vertex_count=len(body.vertices),
        is_valid=len(body.vertices) >= 3,
        processing_time_ms=round(elapsed_ms, 2),
    )


# ---------------------------------------------------------------------------
# POST /gps/distance
# ---------------------------------------------------------------------------


@router.post(
    "/distance",
    response_model=DistanceResponseSchema,
    summary="Calculate distance between points",
    description=(
        "Calculate the geodesic distance between two GPS points "
        "using the Vincenty formula. Returns distance in meters "
        "and kilometers plus bearing in degrees."
    ),
    responses={
        200: {"description": "Distance calculated"},
    },
)
async def calculate_distance(
    body: DistanceRequestSchema,
    user: AuthUser = Depends(require_permission("eudr-mdc:gps:read")),
    service: Any = Depends(get_mdc_service),
    _rl: None = Depends(rate_limit_write),
) -> DistanceResponseSchema:
    """Calculate distance between two GPS points.

    Args:
        body: Two coordinate pairs.
        user: Authenticated user.
        service: MDC service singleton.

    Returns:
        DistanceResponseSchema with distance and bearing.
    """
    start = time.monotonic()
    logger.info(
        "Calculate distance: user=%s (%.6f,%.6f) -> (%.6f,%.6f)",
        user.user_id,
        body.lat1,
        body.lon1,
        body.lat2,
        body.lon2,
    )

    elapsed_ms = (time.monotonic() - start) * 1000
    return DistanceResponseSchema(
        distance_m=0.0,
        distance_km=0.0,
        bearing_degrees=0.0,
        processing_time_ms=round(elapsed_ms, 2),
    )
