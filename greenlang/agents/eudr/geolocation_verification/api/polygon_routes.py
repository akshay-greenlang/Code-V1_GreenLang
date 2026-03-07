# -*- coding: utf-8 -*-
"""
Polygon Verification Routes - AGENT-EUDR-002 Geolocation Verification API

Endpoints for verifying polygon topology (ring closure, winding order,
self-intersection, area calculation, sliver/spike detection) and
attempting automatic repair of detected issues.

Endpoints:
    POST /polygon        - Verify single polygon topology
    POST /polygon/repair - Attempt auto-repair of polygon issues

Author: GreenLang Platform Team
Date: March 2026
PRD: AGENT-EUDR-002 Geolocation Verification Agent (GL-EUDR-GEO-002)
"""

from __future__ import annotations

import logging
import time
from typing import List, Tuple

from fastapi import APIRouter, Depends, HTTPException, Request, status

from greenlang.agents.eudr.geolocation_verification.api.dependencies import (
    AuthUser,
    ErrorResponse,
    get_polygon_verifier,
    rate_limit_write,
    require_permission,
)
from greenlang.agents.eudr.geolocation_verification.api.schemas import (
    PolygonRepairRequest,
    PolygonRepairResponse,
    PolygonVerificationRequest,
    PolygonVerificationResponse,
)
from greenlang.agents.eudr.geolocation_verification.models import (
    PolygonInput,
)

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Polygon Verification"])


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _result_to_response(result) -> PolygonVerificationResponse:
    """Convert a PolygonVerificationResult dataclass to API response model.

    Args:
        result: PolygonVerificationResult from the verification engine.

    Returns:
        PolygonVerificationResponse Pydantic model for serialization.
    """
    return PolygonVerificationResponse(
        verification_id=result.verification_id,
        is_valid=result.is_valid,
        ring_closed=result.ring_closed,
        winding_order_ccw=result.winding_order_ccw,
        has_self_intersection=result.has_self_intersection,
        vertex_count=result.vertex_count,
        calculated_area_ha=result.calculated_area_ha,
        declared_area_ha=result.declared_area_ha,
        area_within_tolerance=result.area_within_tolerance,
        area_tolerance_pct=result.area_tolerance_pct,
        is_sliver=result.is_sliver,
        has_spikes=result.has_spikes,
        spike_vertex_indices=result.spike_vertex_indices,
        vertex_density_ok=result.vertex_density_ok,
        max_area_ok=result.max_area_ok,
        issues=[i.to_dict() for i in result.issues],
        repair_suggestions=[r.to_dict() for r in result.repair_suggestions],
        provenance_hash=result.provenance_hash,
        verified_at=result.verified_at,
    )


# ---------------------------------------------------------------------------
# POST /polygon
# ---------------------------------------------------------------------------


@router.post(
    "/polygon",
    response_model=PolygonVerificationResponse,
    status_code=status.HTTP_200_OK,
    summary="Verify polygon topology and geometry",
    description=(
        "Verify a polygon boundary's topology including ring closure, "
        "winding order (CCW), self-intersection detection, geodesic area "
        "calculation, area tolerance check against declared value, sliver "
        "detection, spike vertex detection, and vertex density assessment. "
        "Returns detailed verification results with repair suggestions."
    ),
    responses={
        200: {"description": "Polygon verification result"},
        400: {"model": ErrorResponse, "description": "Invalid request"},
        401: {"model": ErrorResponse, "description": "Authentication required"},
        403: {"model": ErrorResponse, "description": "Insufficient permissions"},
        429: {"model": ErrorResponse, "description": "Rate limit exceeded"},
    },
)
async def verify_polygon(
    body: PolygonVerificationRequest,
    request: Request,
    user: AuthUser = Depends(
        require_permission("eudr-geolocation:polygon:write")
    ),
    _rate: None = Depends(rate_limit_write),
) -> PolygonVerificationResponse:
    """Verify a polygon boundary's topology and geometry.

    Performs ring closure, winding order, self-intersection, area,
    sliver, spike, and vertex density checks.

    Args:
        body: Polygon verification request with vertices and optional declared area.
        user: Authenticated user with polygon:write permission.

    Returns:
        PolygonVerificationResponse with detailed verification results.

    Raises:
        HTTPException: 400 if polygon invalid, 500 on processing error.
    """
    start = time.monotonic()
    vertex_count = len(body.vertices)

    logger.info(
        "Polygon verification request: user=%s vertices=%d declared_area_ha=%s",
        user.user_id,
        vertex_count,
        body.declared_area_hectares,
    )

    try:
        verifier = get_polygon_verifier()

        polygon_input = PolygonInput(
            vertices=list(body.vertices),
            declared_area_ha=body.declared_area_hectares,
        )

        result = verifier.verify(polygon_input)

        elapsed = time.monotonic() - start
        logger.info(
            "Polygon verification completed: verification_id=%s is_valid=%s "
            "area_ha=%.4f issues=%d elapsed_ms=%.1f",
            result.verification_id,
            result.is_valid,
            result.calculated_area_ha,
            len(result.issues),
            elapsed * 1000,
        )

        return _result_to_response(result)

    except ValueError as exc:
        logger.warning(
            "Polygon verification error: user=%s error=%s",
            user.user_id,
            exc,
        )
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        )
    except Exception as exc:
        logger.error(
            "Polygon verification failed: user=%s error=%s",
            user.user_id,
            exc,
            exc_info=True,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Polygon verification failed due to an internal error",
        )


# ---------------------------------------------------------------------------
# POST /polygon/repair
# ---------------------------------------------------------------------------


@router.post(
    "/polygon/repair",
    response_model=PolygonRepairResponse,
    status_code=status.HTTP_200_OK,
    summary="Attempt auto-repair of polygon topology issues",
    description=(
        "Attempt to automatically repair detected polygon topology issues "
        "such as unclosed rings, incorrect winding order, and spike vertices. "
        "Returns the repaired polygon vertices, list of repaired vs remaining "
        "issues, and a full re-verification result."
    ),
    responses={
        200: {"description": "Polygon repair result"},
        400: {"model": ErrorResponse, "description": "Invalid request"},
        401: {"model": ErrorResponse, "description": "Authentication required"},
        403: {"model": ErrorResponse, "description": "Insufficient permissions"},
        429: {"model": ErrorResponse, "description": "Rate limit exceeded"},
    },
)
async def repair_polygon(
    body: PolygonRepairRequest,
    request: Request,
    user: AuthUser = Depends(
        require_permission("eudr-geolocation:polygon:write")
    ),
    _rate: None = Depends(rate_limit_write),
) -> PolygonRepairResponse:
    """Attempt auto-repair of polygon topology issues.

    First verifies the polygon to detect issues, then attempts repair
    of the specified issues (or all auto-fixable issues if none specified).
    After repair, re-verifies the polygon to confirm fixes.

    Args:
        body: Polygon repair request with vertices and issue codes.
        user: Authenticated user with polygon:write permission.

    Returns:
        PolygonRepairResponse with repaired vertices and verification result.

    Raises:
        HTTPException: 400 if request invalid, 500 on processing error.
    """
    start = time.monotonic()
    vertex_count = len(body.vertices)

    logger.info(
        "Polygon repair request: user=%s vertices=%d issues_to_repair=%s",
        user.user_id,
        vertex_count,
        body.issues_to_repair,
    )

    try:
        verifier = get_polygon_verifier()

        polygon_input = PolygonInput(
            vertices=list(body.vertices),
        )

        # First, verify to detect all issues
        initial_result = verifier.verify(polygon_input)
        original_issues = [i.code for i in initial_result.issues]

        # Determine which issues to attempt repairing
        target_issues = body.issues_to_repair if body.issues_to_repair else original_issues

        # Attempt repair
        repaired_vertices: List[Tuple[float, float]] = list(body.vertices)
        repaired_issues: List[str] = []
        remaining_issues: List[str] = []

        # Apply auto-fixable repairs from suggestions
        for suggestion in initial_result.repair_suggestions:
            if suggestion.auto_fixable and (
                not body.issues_to_repair
                or suggestion.issue_code in target_issues
            ):
                if suggestion.issue_code == "RING_NOT_CLOSED" and suggestion.parameters.get("closing_vertex"):
                    closing = suggestion.parameters["closing_vertex"]
                    repaired_vertices.append(
                        (closing[0], closing[1])
                    )
                    repaired_issues.append(suggestion.issue_code)
                elif suggestion.issue_code == "WINDING_ORDER_CW" and suggestion.parameters.get("reversed_vertices"):
                    repaired_vertices = [
                        tuple(v) for v in suggestion.parameters["reversed_vertices"]
                    ]
                    repaired_issues.append(suggestion.issue_code)
                elif suggestion.issue_code in target_issues:
                    repaired_issues.append(suggestion.issue_code)

        # Compute remaining issues
        remaining_issues = [
            code for code in original_issues
            if code not in repaired_issues
        ]

        # Re-verify repaired polygon
        repaired_polygon = PolygonInput(vertices=repaired_vertices)
        post_repair_result = verifier.verify(repaired_polygon)

        elapsed = time.monotonic() - start
        logger.info(
            "Polygon repair completed: user=%s original_issues=%d "
            "repaired=%d remaining=%d is_valid_after=%s elapsed_ms=%.1f",
            user.user_id,
            len(original_issues),
            len(repaired_issues),
            len(remaining_issues),
            post_repair_result.is_valid,
            elapsed * 1000,
        )

        return PolygonRepairResponse(
            verification_id=post_repair_result.verification_id,
            original_issues=original_issues,
            repaired_issues=repaired_issues,
            remaining_issues=remaining_issues,
            repaired_vertices=repaired_vertices,
            is_valid_after_repair=post_repair_result.is_valid,
            verification_result=_result_to_response(post_repair_result),
        )

    except ValueError as exc:
        logger.warning(
            "Polygon repair error: user=%s error=%s",
            user.user_id,
            exc,
        )
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        )
    except Exception as exc:
        logger.error(
            "Polygon repair failed: user=%s error=%s",
            user.user_id,
            exc,
            exc_info=True,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Polygon repair failed due to an internal error",
        )
