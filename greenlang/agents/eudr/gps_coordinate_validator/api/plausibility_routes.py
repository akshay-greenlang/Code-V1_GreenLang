# -*- coding: utf-8 -*-
"""
Plausibility Analysis Routes - AGENT-EUDR-007 GPS Coordinate Validator API

Endpoints for verifying the plausibility of GPS coordinates against
geographic reference data including land/ocean boundaries, country
boundaries, commodity growing regions, and elevation ranges.

Endpoints:
    POST /plausibility             - Full plausibility analysis
    POST /plausibility/land-ocean  - Land vs ocean detection
    POST /plausibility/country     - Country detection and matching
    POST /plausibility/commodity   - Commodity plausibility check
    POST /plausibility/elevation   - Elevation plausibility check

Author: GreenLang Platform Team
Date: March 2026
PRD: AGENT-EUDR-007 GPS Coordinate Validator (GL-EUDR-GPS-007)
"""

from __future__ import annotations

import hashlib
import logging
import time
from datetime import datetime, timezone
from typing import Any, Dict

from fastapi import APIRouter, Depends, HTTPException, Request, status

from greenlang.agents.eudr.gps_coordinate_validator.api.dependencies import (
    AuthUser,
    ErrorResponse,
    get_gps_validator_service,
    rate_limit_geocode,
    rate_limit_standard,
    require_permission,
)
from greenlang.agents.eudr.gps_coordinate_validator.api.schemas import (
    CommodityResponseSchema,
    CountryResponseSchema,
    ElevationResponseSchema,
    LandOceanResponseSchema,
    PlausibilityRequestSchema,
    PlausibilityResponseSchema,
)

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Plausibility Analysis"])


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _compute_provenance(data: str) -> str:
    """Compute SHA-256 provenance hash.

    Args:
        data: String to hash.

    Returns:
        Hex-encoded SHA-256 hash.
    """
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# POST /plausibility
# ---------------------------------------------------------------------------


@router.post(
    "/plausibility",
    response_model=PlausibilityResponseSchema,
    status_code=status.HTTP_200_OK,
    summary="Full plausibility analysis",
    description=(
        "Perform a comprehensive plausibility analysis for a GPS coordinate "
        "including land/ocean detection, country boundary matching, "
        "commodity growing region verification, elevation plausibility, "
        "urban area detection, protected area screening, and land use "
        "classification. Returns aggregated plausibility results."
    ),
    responses={
        200: {"description": "Plausibility analysis result"},
        400: {"model": ErrorResponse, "description": "Invalid request"},
        401: {"model": ErrorResponse, "description": "Authentication required"},
        403: {"model": ErrorResponse, "description": "Insufficient permissions"},
        429: {"model": ErrorResponse, "description": "Rate limit exceeded"},
    },
)
async def check_plausibility(
    body: PlausibilityRequestSchema,
    request: Request,
    user: AuthUser = Depends(
        require_permission("eudr-gcv:plausibility:read")
    ),
    _rate: None = Depends(rate_limit_standard),
) -> PlausibilityResponseSchema:
    """Perform full plausibility analysis on a coordinate.

    Combines land/ocean, country, commodity, elevation, urban area,
    protected area, and land use checks into a single assessment.

    Args:
        body: Plausibility request with coordinate and context.
        request: FastAPI request object.
        user: Authenticated user with plausibility:read permission.

    Returns:
        PlausibilityResponseSchema with comprehensive results.

    Raises:
        HTTPException: 400 if input invalid, 500 on processing error.
    """
    start = time.monotonic()
    logger.info(
        "Plausibility check: user=%s lat=%.6f lon=%.6f commodity=%s "
        "country=%s altitude=%s",
        user.user_id,
        body.latitude,
        body.longitude,
        body.commodity,
        body.country_iso,
        body.altitude,
    )

    try:
        service = get_gps_validator_service()

        result = service.check_plausibility(
            latitude=body.latitude,
            longitude=body.longitude,
            commodity=body.commodity,
            country_iso=body.country_iso,
            altitude=body.altitude,
        )

        elapsed = time.monotonic() - start
        provenance = _compute_provenance(
            f"plausibility|{body.latitude}|{body.longitude}|"
            f"{result.get('is_on_land', True)}"
        )

        logger.info(
            "Plausibility check completed: user=%s is_on_land=%s "
            "country_match=%s commodity_plausible=%s elapsed_ms=%.1f",
            user.user_id,
            result.get("is_on_land", True),
            result.get("country_match"),
            result.get("commodity_plausible"),
            elapsed * 1000,
        )

        return PlausibilityResponseSchema(
            is_on_land=result.get("is_on_land", True),
            detected_country=result.get("detected_country"),
            country_match=result.get("country_match"),
            commodity_plausible=result.get("commodity_plausible"),
            elevation_plausible=result.get("elevation_plausible"),
            is_urban=result.get("is_urban", False),
            is_protected=result.get("is_protected", False),
            land_use=result.get("land_use"),
            distance_to_coast_km=result.get("distance_to_coast_km"),
            details=result.get("details", {}),
            provenance_hash=provenance,
        )

    except ValueError as exc:
        logger.warning(
            "Plausibility check error: user=%s error=%s",
            user.user_id,
            exc,
        )
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        )
    except NotImplementedError as exc:
        raise HTTPException(
            status_code=status.HTTP_501_NOT_IMPLEMENTED,
            detail=str(exc),
        )
    except Exception as exc:
        logger.error(
            "Plausibility check failed: user=%s error=%s",
            user.user_id,
            exc,
            exc_info=True,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Plausibility check failed due to an internal error",
        )


# ---------------------------------------------------------------------------
# POST /plausibility/land-ocean
# ---------------------------------------------------------------------------


@router.post(
    "/plausibility/land-ocean",
    response_model=LandOceanResponseSchema,
    status_code=status.HTTP_200_OK,
    summary="Land vs ocean detection",
    description=(
        "Determine whether a coordinate falls on land or in the ocean. "
        "Returns the land/ocean classification with distance to the "
        "nearest coastline. Production plots must be on land for EUDR."
    ),
    responses={
        200: {"description": "Land/ocean detection result"},
        401: {"model": ErrorResponse, "description": "Authentication required"},
        403: {"model": ErrorResponse, "description": "Insufficient permissions"},
        429: {"model": ErrorResponse, "description": "Rate limit exceeded"},
    },
)
async def check_land_ocean(
    body: PlausibilityRequestSchema,
    request: Request,
    user: AuthUser = Depends(
        require_permission("eudr-gcv:plausibility:read")
    ),
    _rate: None = Depends(rate_limit_geocode),
) -> LandOceanResponseSchema:
    """Check if a coordinate is on land or in the ocean.

    Uses reference land/ocean mask data to classify the coordinate
    and calculates distance to the nearest coastline.

    Args:
        body: Request with latitude and longitude.
        request: FastAPI request object.
        user: Authenticated user with plausibility:read permission.

    Returns:
        LandOceanResponseSchema with classification and coast distance.

    Raises:
        HTTPException: 500 on processing error.
    """
    start = time.monotonic()
    logger.info(
        "Land/ocean check: user=%s lat=%.6f lon=%.6f",
        user.user_id,
        body.latitude,
        body.longitude,
    )

    try:
        service = get_gps_validator_service()

        result = service.check_land_ocean(
            latitude=body.latitude,
            longitude=body.longitude,
        )

        elapsed = time.monotonic() - start
        logger.info(
            "Land/ocean check completed: user=%s is_on_land=%s "
            "coast_km=%.1f elapsed_ms=%.1f",
            user.user_id,
            result.get("is_on_land", True),
            result.get("nearest_coast_km", 0.0),
            elapsed * 1000,
        )

        return LandOceanResponseSchema(
            is_on_land=result.get("is_on_land", True),
            nearest_coast_km=result.get("nearest_coast_km", 0.0),
            confidence=result.get("confidence", 1.0),
            data_source=result.get("data_source", "internal"),
        )

    except NotImplementedError as exc:
        raise HTTPException(
            status_code=status.HTTP_501_NOT_IMPLEMENTED,
            detail=str(exc),
        )
    except Exception as exc:
        logger.error(
            "Land/ocean check failed: user=%s error=%s",
            user.user_id,
            exc,
            exc_info=True,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Land/ocean check failed due to an internal error",
        )


# ---------------------------------------------------------------------------
# POST /plausibility/country
# ---------------------------------------------------------------------------


@router.post(
    "/plausibility/country",
    response_model=CountryResponseSchema,
    status_code=status.HTTP_200_OK,
    summary="Country detection and matching",
    description=(
        "Detect which country a coordinate falls within and optionally "
        "compare against a declared country code. Uses reference boundary "
        "data to resolve the country with distance to nearest border."
    ),
    responses={
        200: {"description": "Country detection result"},
        401: {"model": ErrorResponse, "description": "Authentication required"},
        403: {"model": ErrorResponse, "description": "Insufficient permissions"},
        429: {"model": ErrorResponse, "description": "Rate limit exceeded"},
    },
)
async def check_country(
    body: PlausibilityRequestSchema,
    request: Request,
    user: AuthUser = Depends(
        require_permission("eudr-gcv:plausibility:read")
    ),
    _rate: None = Depends(rate_limit_geocode),
) -> CountryResponseSchema:
    """Detect the country for a coordinate and compare with declared country.

    Uses reference boundary data to determine which country contains
    the coordinate. If a country_iso is provided, checks whether the
    detected country matches.

    Args:
        body: Request with coordinate and optional country_iso.
        request: FastAPI request object.
        user: Authenticated user with plausibility:read permission.

    Returns:
        CountryResponseSchema with detection and match results.

    Raises:
        HTTPException: 500 on processing error.
    """
    start = time.monotonic()
    logger.info(
        "Country check: user=%s lat=%.6f lon=%.6f declared=%s",
        user.user_id,
        body.latitude,
        body.longitude,
        body.country_iso,
    )

    try:
        service = get_gps_validator_service()

        result = service.check_country(
            latitude=body.latitude,
            longitude=body.longitude,
            declared_iso=body.country_iso,
        )

        elapsed = time.monotonic() - start
        logger.info(
            "Country check completed: user=%s detected=%s matches=%s "
            "elapsed_ms=%.1f",
            user.user_id,
            result.get("detected_iso"),
            result.get("matches_declared"),
            elapsed * 1000,
        )

        return CountryResponseSchema(
            detected_iso=result.get("detected_iso"),
            detected_name=result.get("detected_name"),
            matches_declared=result.get("matches_declared"),
            declared_iso=body.country_iso,
            distance_to_border_km=result.get("distance_to_border_km"),
        )

    except NotImplementedError as exc:
        raise HTTPException(
            status_code=status.HTTP_501_NOT_IMPLEMENTED,
            detail=str(exc),
        )
    except Exception as exc:
        logger.error(
            "Country check failed: user=%s error=%s",
            user.user_id,
            exc,
            exc_info=True,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Country check failed due to an internal error",
        )


# ---------------------------------------------------------------------------
# POST /plausibility/commodity
# ---------------------------------------------------------------------------


@router.post(
    "/plausibility/commodity",
    response_model=CommodityResponseSchema,
    status_code=status.HTTP_200_OK,
    summary="Commodity plausibility check",
    description=(
        "Verify whether a specific EUDR commodity is plausible at the "
        "given GPS coordinate. Checks latitude range (e.g., cocoa is "
        "tropical 20N-20S), elevation range (e.g., cocoa 0-800m), and "
        "known growing regions for the commodity."
    ),
    responses={
        200: {"description": "Commodity plausibility result"},
        400: {"model": ErrorResponse, "description": "Invalid commodity"},
        401: {"model": ErrorResponse, "description": "Authentication required"},
        403: {"model": ErrorResponse, "description": "Insufficient permissions"},
        429: {"model": ErrorResponse, "description": "Rate limit exceeded"},
    },
)
async def check_commodity(
    body: PlausibilityRequestSchema,
    request: Request,
    user: AuthUser = Depends(
        require_permission("eudr-gcv:plausibility:read")
    ),
    _rate: None = Depends(rate_limit_standard),
) -> CommodityResponseSchema:
    """Check if a commodity is plausible at the given coordinate.

    Evaluates latitude range, elevation range, and known growing
    regions for the specified EUDR commodity.

    Args:
        body: Request with coordinate and commodity.
        request: FastAPI request object.
        user: Authenticated user with plausibility:read permission.

    Returns:
        CommodityResponseSchema with plausibility assessment.

    Raises:
        HTTPException: 400 if commodity not provided, 500 on error.
    """
    if not body.commodity:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="commodity is required for commodity plausibility check",
        )

    start = time.monotonic()
    logger.info(
        "Commodity check: user=%s lat=%.6f lon=%.6f commodity=%s",
        user.user_id,
        body.latitude,
        body.longitude,
        body.commodity,
    )

    try:
        service = get_gps_validator_service()

        result = service.check_commodity_plausibility(
            latitude=body.latitude,
            longitude=body.longitude,
            commodity=body.commodity,
            altitude=body.altitude,
        )

        elapsed = time.monotonic() - start
        logger.info(
            "Commodity check completed: user=%s commodity=%s "
            "plausible=%s elapsed_ms=%.1f",
            user.user_id,
            body.commodity,
            result.get("is_plausible", False),
            elapsed * 1000,
        )

        return CommodityResponseSchema(
            is_plausible=result.get("is_plausible", False),
            reason=result.get("reason", ""),
            latitude_range=result.get("latitude_range"),
            elevation_range=result.get("elevation_range"),
            known_growing_regions=result.get("known_growing_regions", []),
        )

    except NotImplementedError as exc:
        raise HTTPException(
            status_code=status.HTTP_501_NOT_IMPLEMENTED,
            detail=str(exc),
        )
    except Exception as exc:
        logger.error(
            "Commodity check failed: user=%s error=%s",
            user.user_id,
            exc,
            exc_info=True,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Commodity check failed due to an internal error",
        )


# ---------------------------------------------------------------------------
# POST /plausibility/elevation
# ---------------------------------------------------------------------------


@router.post(
    "/plausibility/elevation",
    response_model=ElevationResponseSchema,
    status_code=status.HTTP_200_OK,
    summary="Elevation plausibility check",
    description=(
        "Check whether the elevation at a coordinate is plausible for "
        "the specified EUDR commodity. Uses SRTM or ASTER elevation data "
        "to estimate elevation if not provided, and compares against "
        "known commodity elevation ranges."
    ),
    responses={
        200: {"description": "Elevation plausibility result"},
        400: {"model": ErrorResponse, "description": "Invalid request"},
        401: {"model": ErrorResponse, "description": "Authentication required"},
        403: {"model": ErrorResponse, "description": "Insufficient permissions"},
        429: {"model": ErrorResponse, "description": "Rate limit exceeded"},
    },
)
async def check_elevation(
    body: PlausibilityRequestSchema,
    request: Request,
    user: AuthUser = Depends(
        require_permission("eudr-gcv:plausibility:read")
    ),
    _rate: None = Depends(rate_limit_geocode),
) -> ElevationResponseSchema:
    """Check elevation plausibility for a coordinate and commodity.

    Estimates or validates elevation against known commodity growing
    ranges. For example, cocoa typically grows at 0-800m ASL.

    Args:
        body: Request with coordinate, commodity, and optional altitude.
        request: FastAPI request object.
        user: Authenticated user with plausibility:read permission.

    Returns:
        ElevationResponseSchema with plausibility assessment.

    Raises:
        HTTPException: 500 on processing error.
    """
    start = time.monotonic()
    logger.info(
        "Elevation check: user=%s lat=%.6f lon=%.6f commodity=%s "
        "altitude=%s",
        user.user_id,
        body.latitude,
        body.longitude,
        body.commodity,
        body.altitude,
    )

    try:
        service = get_gps_validator_service()

        result = service.check_elevation(
            latitude=body.latitude,
            longitude=body.longitude,
            commodity=body.commodity,
            declared_altitude=body.altitude,
        )

        elapsed = time.monotonic() - start
        logger.info(
            "Elevation check completed: user=%s plausible=%s "
            "elevation_m=%.1f elapsed_ms=%.1f",
            user.user_id,
            result.get("is_plausible", True),
            result.get("elevation_m", 0.0),
            elapsed * 1000,
        )

        return ElevationResponseSchema(
            is_plausible=result.get("is_plausible", True),
            elevation_m=result.get("elevation_m", body.altitude or 0.0),
            commodity_range=result.get("commodity_range"),
            data_source=result.get("data_source", "srtm"),
            confidence=result.get("confidence", 1.0),
        )

    except NotImplementedError as exc:
        raise HTTPException(
            status_code=status.HTTP_501_NOT_IMPLEMENTED,
            detail=str(exc),
        )
    except Exception as exc:
        logger.error(
            "Elevation check failed: user=%s error=%s",
            user.user_id,
            exc,
            exc_info=True,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Elevation check failed due to an internal error",
        )
