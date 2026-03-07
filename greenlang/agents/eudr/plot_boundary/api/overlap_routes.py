# -*- coding: utf-8 -*-
"""
Overlap Detection Routes - AGENT-EUDR-006 Plot Boundary Manager API

Endpoints for detecting spatial overlaps between plot boundaries using
R-tree spatial indexing, performing full registry scans, retrieving
overlap records, and suggesting overlap resolutions.

Endpoints:
    POST /overlaps/detect       - Detect overlaps for a specific plot
    POST /overlaps/scan         - Full registry overlap scan
    GET  /overlaps/{plot_id}    - Get overlap records for a plot
    POST /overlaps/resolve      - Suggest overlap resolution

Author: GreenLang Platform Team
Date: March 2026
PRD: AGENT-EUDR-006 Plot Boundary Manager Agent (GL-EUDR-PBM-006)
"""

from __future__ import annotations

import logging
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List

from fastapi import APIRouter, Depends, HTTPException, Path, Request, status

from greenlang.agents.eudr.plot_boundary.api.dependencies import (
    AuthUser,
    ErrorResponse,
    get_overlap_detector,
    rate_limit_scan,
    rate_limit_standard,
    rate_limit_write,
    require_permission,
    validate_plot_id,
)
from greenlang.agents.eudr.plot_boundary.api.schemas import (
    OverlapDetectRequestSchema,
    OverlapRecordSchema,
    OverlapResolutionRequestSchema,
    OverlapResolutionSchema,
    OverlapResponseSchema,
    OverlapScanRequestSchema,
    OverlapScanResponseSchema,
)

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Overlap Detection"])


# ---------------------------------------------------------------------------
# In-memory overlap store (replaced by database in production)
# ---------------------------------------------------------------------------

_overlap_store: Dict[str, List[Dict[str, Any]]] = {}


def _get_overlap_store() -> Dict[str, List[Dict[str, Any]]]:
    """Return the overlap store. Replaceable for testing."""
    return _overlap_store


# ---------------------------------------------------------------------------
# POST /overlaps/detect
# ---------------------------------------------------------------------------


@router.post(
    "/overlaps/detect",
    response_model=OverlapResponseSchema,
    status_code=status.HTTP_200_OK,
    summary="Detect overlaps for a plot",
    description=(
        "Detect spatial overlaps between a specified plot boundary and "
        "all other boundaries in the registry within the search radius. "
        "Uses R-tree spatial indexing for efficient candidate filtering "
        "followed by exact geometric intersection computation. Overlaps "
        "are classified by severity (minor, moderate, major, critical) "
        "based on configurable area fraction thresholds."
    ),
    responses={
        200: {"description": "Overlap detection results"},
        400: {"model": ErrorResponse, "description": "Invalid request"},
        401: {"model": ErrorResponse, "description": "Authentication required"},
        403: {"model": ErrorResponse, "description": "Insufficient permissions"},
        404: {"description": "Plot not found"},
        429: {"model": ErrorResponse, "description": "Rate limit exceeded"},
    },
)
async def detect_overlaps(
    body: OverlapDetectRequestSchema,
    request: Request,
    user: AuthUser = Depends(
        require_permission("eudr-boundary:overlaps:read")
    ),
    _rate: None = Depends(rate_limit_write),
) -> OverlapResponseSchema:
    """Detect spatial overlaps for a specific plot boundary.

    Searches for candidate boundaries within the specified radius using
    R-tree indexing, then computes exact geometric intersections to
    identify overlapping areas. Each overlap is classified by severity.

    Args:
        body: Overlap detection request with plot_id and search radius.
        user: Authenticated user with overlaps:read permission.

    Returns:
        OverlapResponseSchema with detected overlaps.

    Raises:
        HTTPException: 400 if invalid, 404 if plot not found.
    """
    start = time.monotonic()
    validated_id = validate_plot_id(body.plot_id)

    logger.info(
        "Overlap detection request: user=%s plot_id=%s radius_km=%.1f",
        user.user_id,
        validated_id,
        body.search_radius_km,
    )

    try:
        detector = get_overlap_detector()

        # Try to use real engine if available
        if hasattr(detector, "detect_overlaps"):
            result = detector.detect_overlaps(
                plot_id=validated_id,
                search_radius_km=body.search_radius_km,
                min_overlap_area_m2=body.min_overlap_area_m2,
            )
            elapsed = time.monotonic() - start
            logger.info(
                "Overlap detection completed: plot_id=%s overlaps=%d "
                "candidates=%d elapsed_ms=%.1f",
                validated_id,
                result.total_overlaps if hasattr(result, "total_overlaps") else 0,
                result.candidates_checked if hasattr(result, "candidates_checked") else 0,
                elapsed * 1000,
            )
            return result

        # Stub response for development
        elapsed = time.monotonic() - start
        logger.info(
            "Overlap detection completed (stub): plot_id=%s "
            "overlaps=0 elapsed_ms=%.1f",
            validated_id,
            elapsed * 1000,
        )

        return OverlapResponseSchema(
            plot_id=validated_id,
            total_overlaps=0,
            overlaps=[],
            search_radius_km=body.search_radius_km,
            candidates_checked=0,
            processing_time_ms=elapsed * 1000,
        )

    except ValueError as exc:
        logger.warning(
            "Overlap detection error: user=%s plot_id=%s error=%s",
            user.user_id,
            validated_id,
            exc,
        )
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        )
    except Exception as exc:
        logger.error(
            "Overlap detection failed: user=%s plot_id=%s error=%s",
            user.user_id,
            validated_id,
            exc,
            exc_info=True,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Overlap detection failed due to an internal error",
        )


# ---------------------------------------------------------------------------
# POST /overlaps/scan
# ---------------------------------------------------------------------------


@router.post(
    "/overlaps/scan",
    response_model=OverlapScanResponseSchema,
    status_code=status.HTTP_200_OK,
    summary="Full registry overlap scan",
    description=(
        "Perform a comprehensive overlap scan across all boundaries in "
        "the registry, optionally filtered by bounding box, commodity, "
        "or country. Uses R-tree indexing for efficient pairwise "
        "intersection testing. This is a computationally expensive "
        "operation with a strict rate limit."
    ),
    responses={
        200: {"description": "Scan results"},
        400: {"model": ErrorResponse, "description": "Invalid request"},
        401: {"model": ErrorResponse, "description": "Authentication required"},
        403: {"model": ErrorResponse, "description": "Insufficient permissions"},
        429: {"model": ErrorResponse, "description": "Rate limit exceeded"},
    },
)
async def scan_overlaps(
    body: OverlapScanRequestSchema,
    request: Request,
    user: AuthUser = Depends(
        require_permission("eudr-boundary:overlaps:write")
    ),
    _rate: None = Depends(rate_limit_scan),
) -> OverlapScanResponseSchema:
    """Perform a full registry overlap scan.

    Scans all boundaries (optionally filtered) for pairwise overlaps.
    This is an expensive operation using R-tree spatial indexing for
    efficient candidate pair generation.

    Args:
        body: Scan request with optional region and commodity filters.
        user: Authenticated user with overlaps:write permission.

    Returns:
        OverlapScanResponseSchema with all detected overlaps.
    """
    start = time.monotonic()

    logger.info(
        "Overlap scan request: user=%s commodity=%s country=%s "
        "max_results=%d",
        user.user_id,
        body.commodity,
        body.country_iso,
        body.max_results,
    )

    try:
        detector = get_overlap_detector()

        # Try to use real engine if available
        if hasattr(detector, "scan_registry"):
            result = detector.scan_registry(
                region_bbox=body.region_bbox,
                commodity=body.commodity,
                country_iso=body.country_iso,
                max_results=body.max_results,
            )
            elapsed = time.monotonic() - start
            logger.info(
                "Overlap scan completed: overlaps=%d scanned=%d "
                "elapsed_ms=%.1f",
                result.total_overlaps if hasattr(result, "total_overlaps") else 0,
                result.boundaries_scanned if hasattr(result, "boundaries_scanned") else 0,
                elapsed * 1000,
            )
            return result

        # Stub response for development
        elapsed = time.monotonic() - start
        logger.info(
            "Overlap scan completed (stub): overlaps=0 elapsed_ms=%.1f",
            elapsed * 1000,
        )

        return OverlapScanResponseSchema(
            total_overlaps=0,
            overlaps=[],
            boundaries_scanned=0,
            severity_summary={
                "minor": 0,
                "moderate": 0,
                "major": 0,
                "critical": 0,
            },
            processing_time_ms=elapsed * 1000,
        )

    except ValueError as exc:
        logger.warning(
            "Overlap scan error: user=%s error=%s",
            user.user_id,
            exc,
        )
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        )
    except Exception as exc:
        logger.error(
            "Overlap scan failed: user=%s error=%s",
            user.user_id,
            exc,
            exc_info=True,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Overlap scan failed due to an internal error",
        )


# ---------------------------------------------------------------------------
# GET /overlaps/{plot_id}
# ---------------------------------------------------------------------------


@router.get(
    "/overlaps/{plot_id}",
    response_model=OverlapResponseSchema,
    status_code=status.HTTP_200_OK,
    summary="Get overlap records for a plot",
    description=(
        "Retrieve all known overlap records for a specific plot boundary. "
        "Returns previously detected overlaps stored in the registry."
    ),
    responses={
        200: {"description": "Overlap records"},
        400: {"model": ErrorResponse, "description": "Invalid plot_id"},
        401: {"model": ErrorResponse, "description": "Authentication required"},
        403: {"model": ErrorResponse, "description": "Insufficient permissions"},
        404: {"description": "No overlap records found"},
    },
)
async def get_overlap_records(
    plot_id: str = Path(..., description="Plot identifier"),
    request: Request = None,
    user: AuthUser = Depends(
        require_permission("eudr-boundary:overlaps:read")
    ),
    _rate: None = Depends(rate_limit_standard),
) -> OverlapResponseSchema:
    """Retrieve overlap records for a specific plot.

    Returns all previously detected and stored overlap records
    involving the specified plot boundary.

    Args:
        plot_id: Plot identifier from URL path.
        user: Authenticated user with overlaps:read permission.

    Returns:
        OverlapResponseSchema with stored overlap records.

    Raises:
        HTTPException: 400 if plot_id invalid.
    """
    validated_id = validate_plot_id(plot_id)

    logger.info(
        "Get overlap records: user=%s plot_id=%s",
        user.user_id,
        validated_id,
    )

    store = _get_overlap_store()
    records = store.get(validated_id, [])

    overlaps = []
    for rec in records:
        try:
            overlaps.append(OverlapRecordSchema(**rec))
        except Exception as exc:
            logger.warning(
                "Skipping malformed overlap record for plot_id=%s: %s",
                validated_id,
                exc,
            )

    return OverlapResponseSchema(
        plot_id=validated_id,
        total_overlaps=len(overlaps),
        overlaps=overlaps,
        search_radius_km=0.0,
        candidates_checked=0,
        processing_time_ms=0.0,
    )


# ---------------------------------------------------------------------------
# POST /overlaps/resolve
# ---------------------------------------------------------------------------


@router.post(
    "/overlaps/resolve",
    response_model=OverlapResolutionSchema,
    status_code=status.HTTP_200_OK,
    summary="Suggest overlap resolution",
    description=(
        "Analyze an overlap between two plot boundaries and suggest a "
        "resolution strategy. Strategies include boundary adjustment, "
        "plot splitting, shared boundary negotiation, or regulatory "
        "escalation depending on overlap severity and context."
    ),
    responses={
        200: {"description": "Resolution suggestion"},
        400: {"model": ErrorResponse, "description": "Invalid request"},
        401: {"model": ErrorResponse, "description": "Authentication required"},
        403: {"model": ErrorResponse, "description": "Insufficient permissions"},
        404: {"description": "Overlap record not found"},
        429: {"model": ErrorResponse, "description": "Rate limit exceeded"},
    },
)
async def resolve_overlap(
    body: OverlapResolutionRequestSchema,
    request: Request,
    user: AuthUser = Depends(
        require_permission("eudr-boundary:overlaps:write")
    ),
    _rate: None = Depends(rate_limit_write),
) -> OverlapResolutionSchema:
    """Suggest a resolution strategy for an overlap.

    Analyzes the overlap geometry, affected boundaries, and severity
    to suggest the most appropriate resolution strategy.

    Args:
        body: Resolution request with overlap_id and plot identifiers.
        user: Authenticated user with overlaps:write permission.

    Returns:
        OverlapResolutionSchema with suggested resolution.

    Raises:
        HTTPException: 400 if request invalid, 500 on processing error.
    """
    start = time.monotonic()

    logger.info(
        "Overlap resolution request: user=%s overlap_id=%s "
        "plot_a=%s plot_b=%s",
        user.user_id,
        body.overlap_id,
        body.plot_id_a,
        body.plot_id_b,
    )

    try:
        detector = get_overlap_detector()

        # Try to use real engine if available
        if hasattr(detector, "suggest_resolution"):
            result = detector.suggest_resolution(
                overlap_id=body.overlap_id,
                plot_id_a=body.plot_id_a,
                plot_id_b=body.plot_id_b,
                preferred_resolution=body.preferred_resolution,
            )
            elapsed = time.monotonic() - start
            logger.info(
                "Overlap resolution completed: overlap_id=%s "
                "suggestion=%s elapsed_ms=%.1f",
                body.overlap_id,
                result.suggested_resolution if hasattr(result, "suggested_resolution") else "unknown",
                elapsed * 1000,
            )
            return result

        # Stub response for development
        elapsed = time.monotonic() - start
        suggested = body.preferred_resolution or "boundary_adjustment"

        logger.info(
            "Overlap resolution completed (stub): overlap_id=%s "
            "suggestion=%s elapsed_ms=%.1f",
            body.overlap_id,
            suggested,
            elapsed * 1000,
        )

        return OverlapResolutionSchema(
            overlap_id=body.overlap_id,
            suggested_resolution=suggested,
            details={
                "plot_id_a": body.plot_id_a,
                "plot_id_b": body.plot_id_b,
                "analysis": (
                    "Overlap analysis requires boundary geometry data. "
                    "The suggested resolution is based on the preferred "
                    "strategy provided or the default boundary adjustment."
                ),
            },
            confidence=0.7,
            alternative_resolutions=[
                "boundary_adjustment",
                "plot_splitting",
                "shared_boundary_negotiation",
                "regulatory_escalation",
            ],
        )

    except ValueError as exc:
        logger.warning(
            "Overlap resolution error: user=%s error=%s",
            user.user_id,
            exc,
        )
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        )
    except Exception as exc:
        logger.error(
            "Overlap resolution failed: user=%s error=%s",
            user.user_id,
            exc,
            exc_info=True,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Overlap resolution failed due to an internal error",
        )
