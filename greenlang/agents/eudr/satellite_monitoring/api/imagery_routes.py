# -*- coding: utf-8 -*-
"""
Imagery Routes - AGENT-EUDR-003 Satellite Monitoring API

Endpoints for satellite imagery management including scene search,
band download, scene metadata retrieval, and data availability checks.

Endpoints:
    POST /search          - Search available satellite scenes for a polygon
    POST /download        - Download specific bands from a scene
    GET  /{scene_id}      - Get scene metadata
    GET  /availability    - Check data availability for a location

Author: GreenLang Platform Team
Date: March 2026
PRD: AGENT-EUDR-003 Satellite Monitoring Agent (GL-EUDR-SAT-003)
"""

from __future__ import annotations

import logging
import time
import uuid
from datetime import date, datetime, timezone
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query, Request, status

from greenlang.agents.eudr.satellite_monitoring.api.dependencies import (
    AuthUser,
    ErrorResponse,
    get_imagery_engine,
    rate_limit_heavy,
    rate_limit_standard,
    rate_limit_write,
    require_permission,
)
from greenlang.agents.eudr.satellite_monitoring.api.schemas import (
    AvailabilityResponse,
    DownloadBandsApiRequest,
    DownloadBandsApiResponse,
    SceneMetadataResponse,
    SearchScenesApiRequest,
    SearchScenesApiResponse,
)

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Satellite Imagery"])


# ---------------------------------------------------------------------------
# POST /search
# ---------------------------------------------------------------------------


@router.post(
    "/search",
    response_model=SearchScenesApiResponse,
    status_code=status.HTTP_200_OK,
    summary="Search available satellite scenes",
    description=(
        "Search for available satellite imagery scenes covering a polygon "
        "area within a specified date range. Supports filtering by source "
        "(Sentinel-2, Landsat 8/9, GFW, Planet) and maximum cloud cover. "
        "Returns scene metadata sorted by acquisition date."
    ),
    responses={
        200: {"description": "Scene search results"},
        400: {"model": ErrorResponse, "description": "Invalid request"},
        401: {"model": ErrorResponse, "description": "Authentication required"},
        403: {"model": ErrorResponse, "description": "Insufficient permissions"},
        429: {"model": ErrorResponse, "description": "Rate limit exceeded"},
    },
)
async def search_scenes(
    body: SearchScenesApiRequest,
    request: Request,
    user: AuthUser = Depends(
        require_permission("eudr-satellite:imagery:read")
    ),
    _rate: None = Depends(rate_limit_standard),
) -> SearchScenesApiResponse:
    """Search available satellite scenes for a polygon area.

    Queries configured satellite data providers for scenes covering
    the specified polygon within the date range. Filters by cloud
    cover and returns metadata sorted by acquisition date.

    Args:
        body: Search request with polygon, date range, source, cloud cover.
        user: Authenticated user with imagery:read permission.

    Returns:
        SearchScenesApiResponse with matching scenes.

    Raises:
        HTTPException: 400 if request invalid, 500 on processing error.
    """
    start = time.monotonic()
    logger.info(
        "Scene search request: user=%s source=%s start=%s end=%s "
        "cloud_max=%.1f vertices=%d",
        user.user_id,
        body.source,
        body.start_date,
        body.end_date,
        body.cloud_cover_max,
        len(body.polygon_vertices),
    )

    try:
        engine = get_imagery_engine()

        result = engine.search_scenes(
            polygon_vertices=body.polygon_vertices,
            start_date=body.start_date,
            end_date=body.end_date,
            source=body.source,
            cloud_cover_max=body.cloud_cover_max,
        )

        scenes = []
        for scene in getattr(result, "scenes", []):
            scenes.append(SceneMetadataResponse(
                scene_id=getattr(scene, "scene_id", ""),
                source=getattr(scene, "source", ""),
                acquisition_date=getattr(scene, "acquisition_date", datetime.now(timezone.utc)),
                cloud_cover_pct=getattr(scene, "cloud_cover_pct", 0.0),
                spatial_resolution_m=getattr(scene, "spatial_resolution_m", 10.0),
                tile_id=getattr(scene, "tile_id", ""),
                bounds=getattr(scene, "bounds", {}),
                available_bands=getattr(scene, "available_bands", []),
                quality_score=getattr(scene, "quality_score", 0.0),
                file_size_mb=getattr(scene, "file_size_mb", None),
                processing_level=getattr(scene, "processing_level", "L2A"),
                provenance_hash=getattr(scene, "provenance_hash", ""),
            ))

        elapsed = time.monotonic() - start
        logger.info(
            "Scene search completed: user=%s total_scenes=%d elapsed_ms=%.1f",
            user.user_id,
            len(scenes),
            elapsed * 1000,
        )

        return SearchScenesApiResponse(
            total_scenes=len(scenes),
            source_filter=body.source,
            cloud_cover_max=body.cloud_cover_max,
            scenes=scenes,
            search_area_ha=getattr(result, "search_area_ha", 0.0),
            processing_time_ms=elapsed * 1000,
        )

    except ValueError as exc:
        logger.warning(
            "Scene search error: user=%s error=%s",
            user.user_id,
            exc,
        )
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        )
    except Exception as exc:
        logger.error(
            "Scene search failed: user=%s error=%s",
            user.user_id,
            exc,
            exc_info=True,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Scene search failed due to an internal error",
        )


# ---------------------------------------------------------------------------
# POST /download
# ---------------------------------------------------------------------------


@router.post(
    "/download",
    response_model=DownloadBandsApiResponse,
    status_code=status.HTTP_200_OK,
    summary="Download specific bands from a satellite scene",
    description=(
        "Download specific spectral bands from a satellite scene for "
        "analysis. Supports clipping to a polygon area. Heavy operation "
        "with reduced rate limit (10/min)."
    ),
    responses={
        200: {"description": "Band download confirmation"},
        400: {"model": ErrorResponse, "description": "Invalid request"},
        401: {"model": ErrorResponse, "description": "Authentication required"},
        403: {"model": ErrorResponse, "description": "Insufficient permissions"},
        404: {"model": ErrorResponse, "description": "Scene not found"},
        429: {"model": ErrorResponse, "description": "Rate limit exceeded"},
    },
)
async def download_bands(
    body: DownloadBandsApiRequest,
    request: Request,
    user: AuthUser = Depends(
        require_permission("eudr-satellite:imagery:write")
    ),
    _rate: None = Depends(rate_limit_heavy),
) -> DownloadBandsApiResponse:
    """Download specific bands from a satellite scene.

    Retrieves spectral band data from the satellite data provider
    for the specified scene. Supports optional polygon clipping to
    reduce download size.

    Args:
        body: Download request with scene_id, bands, and optional clip.
        user: Authenticated user with imagery:write permission.

    Returns:
        DownloadBandsApiResponse with download confirmation and metadata.

    Raises:
        HTTPException: 400/404 if invalid, 500 on processing error.
    """
    start = time.monotonic()
    logger.info(
        "Band download request: user=%s scene_id=%s bands=%s",
        user.user_id,
        body.scene_id,
        body.bands,
    )

    try:
        engine = get_imagery_engine()

        result = engine.download_bands(
            scene_id=body.scene_id,
            bands=body.bands,
            polygon_clip=body.polygon_clip,
        )

        elapsed = time.monotonic() - start
        logger.info(
            "Band download completed: user=%s scene_id=%s bands=%d "
            "size_mb=%.1f elapsed_ms=%.1f",
            user.user_id,
            body.scene_id,
            len(getattr(result, "bands_downloaded", body.bands)),
            getattr(result, "total_size_mb", 0.0),
            elapsed * 1000,
        )

        return DownloadBandsApiResponse(
            scene_id=body.scene_id,
            bands_downloaded=getattr(result, "bands_downloaded", body.bands),
            band_metadata=getattr(result, "band_metadata", []),
            total_size_mb=getattr(result, "total_size_mb", 0.0),
            download_status=getattr(result, "download_status", "completed"),
            processing_time_ms=elapsed * 1000,
        )

    except ValueError as exc:
        logger.warning(
            "Band download error: user=%s scene_id=%s error=%s",
            user.user_id,
            body.scene_id,
            exc,
        )
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        )
    except FileNotFoundError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Scene {body.scene_id} not found",
        )
    except Exception as exc:
        logger.error(
            "Band download failed: user=%s scene_id=%s error=%s",
            user.user_id,
            body.scene_id,
            exc,
            exc_info=True,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Band download failed due to an internal error",
        )


# ---------------------------------------------------------------------------
# GET /{scene_id}
# ---------------------------------------------------------------------------


@router.get(
    "/{scene_id}",
    response_model=SceneMetadataResponse,
    summary="Get scene metadata",
    description="Retrieve full metadata for a specific satellite scene.",
    responses={
        200: {"description": "Scene metadata"},
        401: {"model": ErrorResponse, "description": "Authentication required"},
        403: {"model": ErrorResponse, "description": "Insufficient permissions"},
        404: {"model": ErrorResponse, "description": "Scene not found"},
    },
)
async def get_scene_metadata(
    scene_id: str,
    request: Request,
    user: AuthUser = Depends(
        require_permission("eudr-satellite:imagery:read")
    ),
    _rate: None = Depends(rate_limit_standard),
) -> SceneMetadataResponse:
    """Get metadata for a specific satellite scene.

    Args:
        scene_id: Scene identifier.
        user: Authenticated user with imagery:read permission.

    Returns:
        SceneMetadataResponse with full scene metadata.

    Raises:
        HTTPException: 404 if scene not found.
    """
    logger.info(
        "Scene metadata request: user=%s scene_id=%s",
        user.user_id,
        scene_id,
    )

    try:
        engine = get_imagery_engine()
        scene = engine.get_scene_metadata(scene_id=scene_id)

        if scene is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Scene {scene_id} not found",
            )

        return SceneMetadataResponse(
            scene_id=getattr(scene, "scene_id", scene_id),
            source=getattr(scene, "source", ""),
            acquisition_date=getattr(scene, "acquisition_date", datetime.now(timezone.utc)),
            cloud_cover_pct=getattr(scene, "cloud_cover_pct", 0.0),
            spatial_resolution_m=getattr(scene, "spatial_resolution_m", 10.0),
            tile_id=getattr(scene, "tile_id", ""),
            bounds=getattr(scene, "bounds", {}),
            available_bands=getattr(scene, "available_bands", []),
            quality_score=getattr(scene, "quality_score", 0.0),
            file_size_mb=getattr(scene, "file_size_mb", None),
            processing_level=getattr(scene, "processing_level", "L2A"),
            provenance_hash=getattr(scene, "provenance_hash", ""),
        )

    except HTTPException:
        raise
    except Exception as exc:
        logger.error(
            "Scene metadata retrieval failed: user=%s scene_id=%s error=%s",
            user.user_id,
            scene_id,
            exc,
            exc_info=True,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Scene metadata retrieval failed due to an internal error",
        )


# ---------------------------------------------------------------------------
# GET /availability
# ---------------------------------------------------------------------------


@router.get(
    "/availability",
    response_model=AvailabilityResponse,
    summary="Check data availability for a location",
    description=(
        "Check satellite imagery availability for a geographic location "
        "within a date range. Returns scene counts per source and best "
        "available cloud-free scene."
    ),
    responses={
        200: {"description": "Availability summary"},
        400: {"model": ErrorResponse, "description": "Invalid parameters"},
        401: {"model": ErrorResponse, "description": "Authentication required"},
        403: {"model": ErrorResponse, "description": "Insufficient permissions"},
    },
)
async def check_availability(
    request: Request,
    lat: float = Query(
        ..., ge=-90.0, le=90.0, description="Latitude (WGS84)"
    ),
    lon: float = Query(
        ..., ge=-180.0, le=180.0, description="Longitude (WGS84)"
    ),
    start_date: date = Query(
        ..., description="Start date (YYYY-MM-DD)"
    ),
    end_date: date = Query(
        ..., description="End date (YYYY-MM-DD)"
    ),
    user: AuthUser = Depends(
        require_permission("eudr-satellite:imagery:read")
    ),
    _rate: None = Depends(rate_limit_standard),
) -> AvailabilityResponse:
    """Check data availability for a location and date range.

    Args:
        lat: Latitude in decimal degrees (WGS84).
        lon: Longitude in decimal degrees (WGS84).
        start_date: Search window start date.
        end_date: Search window end date.
        user: Authenticated user with imagery:read permission.

    Returns:
        AvailabilityResponse with scene counts per source.

    Raises:
        HTTPException: 400 if dates invalid, 500 on error.
    """
    if end_date < start_date:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"end_date ({end_date}) must be on or after start_date ({start_date})",
        )

    logger.info(
        "Availability check: user=%s lat=%.6f lon=%.6f start=%s end=%s",
        user.user_id,
        lat,
        lon,
        start_date,
        end_date,
    )

    try:
        engine = get_imagery_engine()

        result = engine.check_availability(
            lat=lat,
            lon=lon,
            start_date=start_date,
            end_date=end_date,
        )

        return AvailabilityResponse(
            lat=lat,
            lon=lon,
            start_date=start_date,
            end_date=end_date,
            total_scenes=getattr(result, "total_scenes", 0),
            by_source=getattr(result, "by_source", {}),
            cloud_free_scenes=getattr(result, "cloud_free_scenes", 0),
            best_scene_id=getattr(result, "best_scene_id", None),
        )

    except ValueError as exc:
        logger.warning(
            "Availability check error: user=%s error=%s",
            user.user_id,
            exc,
        )
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        )
    except Exception as exc:
        logger.error(
            "Availability check failed: user=%s error=%s",
            user.user_id,
            exc,
            exc_info=True,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Availability check failed due to an internal error",
        )
