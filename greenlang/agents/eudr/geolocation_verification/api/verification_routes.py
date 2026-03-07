# -*- coding: utf-8 -*-
"""
Verification Routes - AGENT-EUDR-002 Geolocation Verification API

Endpoints for protected area screening, deforestation verification,
and full plot verification (combining all checks into a single
comprehensive verification workflow).

Endpoints:
    POST /protected-areas                   - Screen plot against protected areas
    GET  /protected-areas/nearby            - List nearby protected areas
    POST /deforestation                     - Verify deforestation status
    GET  /deforestation/{plot_id}/evidence  - Get deforestation evidence package
    POST /plot                              - Full verification of single plot
    GET  /plot/{plot_id}                    - Get latest verification result
    GET  /plot/{plot_id}/history            - Get verification history

Author: GreenLang Platform Team
Date: March 2026
PRD: AGENT-EUDR-002 Geolocation Verification Agent (GL-EUDR-GEO-002)
"""

from __future__ import annotations

import logging
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, Request, status

from greenlang.agents.eudr.geolocation_verification.api.dependencies import (
    AuthUser,
    ErrorResponse,
    PaginationParams,
    get_coordinate_validator,
    get_deforestation_verifier,
    get_pagination,
    get_polygon_verifier,
    get_protected_area_checker,
    get_accuracy_scorer,
    get_verification_service,
    rate_limit_heavy,
    rate_limit_standard,
    rate_limit_write,
    require_permission,
)
from greenlang.agents.eudr.geolocation_verification.api.schemas import (
    DeforestationEvidenceResponse,
    DeforestationVerifyRequest,
    DeforestationVerifyResponse,
    NearbyProtectedAreasResponse,
    PlotVerificationHistoryResponse,
    PlotVerificationRequest,
    PlotVerificationResponse,
    ProtectedAreaScreenRequest,
    ProtectedAreaScreenResponse,
    CoordinateValidationResponse,
    PolygonVerificationResponse,
    PaginatedMeta,
)
from greenlang.agents.eudr.geolocation_verification.models import (
    CoordinateInput,
    PolygonInput,
)

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Verification"])


# ---------------------------------------------------------------------------
# In-memory stores (replaced by database in production)
# ---------------------------------------------------------------------------

_verification_store: Dict[str, Dict[str, Any]] = {}
_evidence_store: Dict[str, Dict[str, Any]] = {}


def _get_verification_store() -> Dict[str, Dict[str, Any]]:
    """Return the verification result store. Replaceable for testing."""
    return _verification_store


def _get_evidence_store() -> Dict[str, Dict[str, Any]]:
    """Return the evidence store. Replaceable for testing."""
    return _evidence_store


# ---------------------------------------------------------------------------
# POST /protected-areas
# ---------------------------------------------------------------------------


@router.post(
    "/protected-areas",
    response_model=ProtectedAreaScreenResponse,
    status_code=status.HTTP_200_OK,
    summary="Screen plot against protected areas",
    description=(
        "Screen a coordinate or polygon against the WDPA protected areas "
        "database. Checks for overlap with national parks, nature reserves, "
        "indigenous territories, and other protected areas within a "
        "configurable buffer zone. Returns overlap status and nearby areas."
    ),
    responses={
        200: {"description": "Protected area screening result"},
        400: {"model": ErrorResponse, "description": "Invalid request"},
        401: {"model": ErrorResponse, "description": "Authentication required"},
        403: {"model": ErrorResponse, "description": "Insufficient permissions"},
        429: {"model": ErrorResponse, "description": "Rate limit exceeded"},
    },
)
async def screen_protected_areas(
    body: ProtectedAreaScreenRequest,
    request: Request,
    user: AuthUser = Depends(
        require_permission("eudr-geolocation:protected-areas:write")
    ),
    _rate: None = Depends(rate_limit_write),
) -> ProtectedAreaScreenResponse:
    """Screen a plot location against protected area databases.

    Checks for overlap with WDPA-listed protected areas and returns
    detailed overlap information including area name, type, and
    percentage overlap.

    Args:
        body: Protected area screening request with coordinates and buffer.
        user: Authenticated user with protected-areas:write permission.

    Returns:
        ProtectedAreaScreenResponse with overlap status and nearby areas.

    Raises:
        HTTPException: 400 if request invalid, 500 on processing error.
    """
    start = time.monotonic()
    logger.info(
        "Protected area screening: user=%s lat=%.6f lon=%.6f buffer_km=%.1f",
        user.user_id,
        body.lat,
        body.lon,
        body.buffer_km,
    )

    try:
        checker = get_protected_area_checker()

        result = checker.check(
            lat=body.lat,
            lon=body.lon,
            polygon_vertices=body.polygon_vertices,
            buffer_km=body.buffer_km,
        )

        elapsed = time.monotonic() - start
        logger.info(
            "Protected area screening completed: user=%s overlaps=%s "
            "area_name=%s elapsed_ms=%.1f",
            user.user_id,
            result.overlaps_protected,
            result.protected_area_name,
            elapsed * 1000,
        )

        return ProtectedAreaScreenResponse(
            overlaps_protected=result.overlaps_protected,
            protected_area_name=result.protected_area_name,
            protected_area_type=result.protected_area_type,
            overlap_percentage=result.overlap_percentage,
            buffer_km_used=body.buffer_km,
        )

    except ValueError as exc:
        logger.warning(
            "Protected area screening error: user=%s error=%s",
            user.user_id,
            exc,
        )
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        )
    except Exception as exc:
        logger.error(
            "Protected area screening failed: user=%s error=%s",
            user.user_id,
            exc,
            exc_info=True,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Protected area screening failed due to an internal error",
        )


# ---------------------------------------------------------------------------
# GET /protected-areas/nearby
# ---------------------------------------------------------------------------


@router.get(
    "/protected-areas/nearby",
    response_model=NearbyProtectedAreasResponse,
    summary="List nearby protected areas",
    description=(
        "Query for protected areas within a given radius of a coordinate. "
        "Returns a list of protected areas with name, type, distance, "
        "area size, and IUCN category."
    ),
    responses={
        200: {"description": "List of nearby protected areas"},
        400: {"model": ErrorResponse, "description": "Invalid parameters"},
        401: {"model": ErrorResponse, "description": "Authentication required"},
        403: {"model": ErrorResponse, "description": "Insufficient permissions"},
        429: {"model": ErrorResponse, "description": "Rate limit exceeded"},
    },
)
async def list_nearby_protected_areas(
    request: Request,
    lat: float = Query(
        ..., ge=-90.0, le=90.0, description="Latitude in decimal degrees (WGS84)"
    ),
    lon: float = Query(
        ..., ge=-180.0, le=180.0, description="Longitude in decimal degrees (WGS84)"
    ),
    radius_km: float = Query(
        default=50.0, ge=0.1, le=500.0, description="Search radius in kilometres"
    ),
    user: AuthUser = Depends(
        require_permission("eudr-geolocation:protected-areas:read")
    ),
    _rate: None = Depends(rate_limit_standard),
) -> NearbyProtectedAreasResponse:
    """List protected areas within a search radius of a coordinate.

    Args:
        lat: Query latitude.
        lon: Query longitude.
        radius_km: Search radius in kilometres.
        user: Authenticated user with protected-areas:read permission.

    Returns:
        NearbyProtectedAreasResponse with list of nearby protected areas.

    Raises:
        HTTPException: 400 if parameters invalid, 500 on processing error.
    """
    start = time.monotonic()
    logger.info(
        "Nearby protected areas query: user=%s lat=%.6f lon=%.6f radius_km=%.1f",
        user.user_id,
        lat,
        lon,
        radius_km,
    )

    try:
        checker = get_protected_area_checker()

        areas = checker.find_nearby(
            lat=lat,
            lon=lon,
            radius_km=radius_km,
        )

        elapsed = time.monotonic() - start
        logger.info(
            "Nearby protected areas query completed: user=%s found=%d elapsed_ms=%.1f",
            user.user_id,
            len(areas),
            elapsed * 1000,
        )

        return NearbyProtectedAreasResponse(
            lat=lat,
            lon=lon,
            radius_km=radius_km,
            total_found=len(areas),
            areas=areas,
        )

    except Exception as exc:
        logger.error(
            "Nearby protected areas query failed: user=%s error=%s",
            user.user_id,
            exc,
            exc_info=True,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Nearby protected areas query failed due to an internal error",
        )


# ---------------------------------------------------------------------------
# POST /deforestation
# ---------------------------------------------------------------------------


@router.post(
    "/deforestation",
    response_model=DeforestationVerifyResponse,
    status_code=status.HTTP_200_OK,
    summary="Verify deforestation status for a plot",
    description=(
        "Verify whether a production plot has been subject to deforestation "
        "after the EUDR cutoff date (December 31, 2020 per Article 2(1)). "
        "Checks satellite-based deforestation alerts from GFW, JRC, and "
        "GLAD data sources. Returns alert count, forest loss area, and "
        "confidence level."
    ),
    responses={
        200: {"description": "Deforestation verification result"},
        400: {"model": ErrorResponse, "description": "Invalid request"},
        401: {"model": ErrorResponse, "description": "Authentication required"},
        403: {"model": ErrorResponse, "description": "Insufficient permissions"},
        429: {"model": ErrorResponse, "description": "Rate limit exceeded"},
    },
)
async def verify_deforestation(
    body: DeforestationVerifyRequest,
    request: Request,
    user: AuthUser = Depends(
        require_permission("eudr-geolocation:deforestation:write")
    ),
    _rate: None = Depends(rate_limit_heavy),
) -> DeforestationVerifyResponse:
    """Verify deforestation status for a production plot.

    Checks satellite deforestation alert databases for any post-cutoff
    deforestation events at the plot location.

    Args:
        body: Deforestation verification request with plot coordinates.
        user: Authenticated user with deforestation:write permission.

    Returns:
        DeforestationVerifyResponse with detection result and confidence.

    Raises:
        HTTPException: 400 if request invalid, 500 on processing error.
    """
    start = time.monotonic()
    logger.info(
        "Deforestation verification: user=%s plot_id=%s lat=%.6f lon=%.6f",
        user.user_id,
        body.plot_id,
        body.lat,
        body.lon,
    )

    try:
        verifier = get_deforestation_verifier()

        result = verifier.verify(
            plot_id=body.plot_id,
            lat=body.lat,
            lon=body.lon,
            polygon_vertices=body.polygon_vertices,
            commodity=body.commodity,
        )

        elapsed = time.monotonic() - start
        logger.info(
            "Deforestation verification completed: plot_id=%s detected=%s "
            "alerts=%d forest_loss_ha=%.4f elapsed_ms=%.1f",
            body.plot_id,
            result.deforestation_detected,
            result.alert_count,
            result.forest_loss_ha,
            elapsed * 1000,
        )

        return DeforestationVerifyResponse(
            plot_id=body.plot_id,
            deforestation_detected=result.deforestation_detected,
            alert_count=result.alert_count,
            forest_loss_ha=result.forest_loss_ha,
            cutoff_date=result.cutoff_date,
            confidence=result.confidence,
            data_sources=["GFW", "JRC", "GLAD"],
        )

    except ValueError as exc:
        logger.warning(
            "Deforestation verification error: user=%s plot_id=%s error=%s",
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
            "Deforestation verification failed: user=%s plot_id=%s error=%s",
            user.user_id,
            body.plot_id,
            exc,
            exc_info=True,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Deforestation verification failed due to an internal error",
        )


# ---------------------------------------------------------------------------
# GET /deforestation/{plot_id}/evidence
# ---------------------------------------------------------------------------


@router.get(
    "/deforestation/{plot_id}/evidence",
    response_model=DeforestationEvidenceResponse,
    summary="Get deforestation evidence package",
    description=(
        "Retrieve the full deforestation evidence package for a plot, "
        "including individual alerts, satellite imagery references, "
        "and historical forest cover timeline."
    ),
    responses={
        200: {"description": "Deforestation evidence package"},
        401: {"model": ErrorResponse, "description": "Authentication required"},
        403: {"model": ErrorResponse, "description": "Insufficient permissions"},
        404: {"model": ErrorResponse, "description": "Evidence not found"},
    },
)
async def get_deforestation_evidence(
    plot_id: str,
    request: Request,
    user: AuthUser = Depends(
        require_permission("eudr-geolocation:deforestation:read")
    ),
    _rate: None = Depends(rate_limit_standard),
) -> DeforestationEvidenceResponse:
    """Get the deforestation evidence package for a plot.

    Args:
        plot_id: Plot identifier.
        user: Authenticated user with deforestation:read permission.

    Returns:
        DeforestationEvidenceResponse with alerts, imagery, and timeline.

    Raises:
        HTTPException: 404 if evidence not found, 403 if unauthorized.
    """
    logger.info(
        "Deforestation evidence request: user=%s plot_id=%s",
        user.user_id,
        plot_id,
    )

    evidence_store = _get_evidence_store()
    evidence = evidence_store.get(plot_id)

    if evidence is None:
        # Attempt to generate evidence from deforestation verifier
        try:
            verifier = get_deforestation_verifier()
            evidence_data = verifier.get_evidence(plot_id=plot_id)

            if evidence_data is None:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"No deforestation evidence found for plot {plot_id}",
                )

            return DeforestationEvidenceResponse(
                plot_id=plot_id,
                evidence_id=f"ev-{uuid.uuid4().hex[:12]}",
                alert_details=getattr(evidence_data, "alert_details", []),
                satellite_imagery=getattr(evidence_data, "satellite_imagery", []),
                forest_cover_timeline=getattr(evidence_data, "forest_cover_timeline", []),
                provenance_hash=getattr(evidence_data, "provenance_hash", ""),
            )
        except HTTPException:
            raise
        except Exception as exc:
            logger.error(
                "Deforestation evidence retrieval failed: plot_id=%s error=%s",
                plot_id,
                exc,
                exc_info=True,
            )
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"No deforestation evidence found for plot {plot_id}",
            )

    return DeforestationEvidenceResponse(
        plot_id=plot_id,
        evidence_id=evidence.get("evidence_id", ""),
        alert_details=evidence.get("alert_details", []),
        satellite_imagery=evidence.get("satellite_imagery", []),
        forest_cover_timeline=evidence.get("forest_cover_timeline", []),
        provenance_hash=evidence.get("provenance_hash", ""),
    )


# ---------------------------------------------------------------------------
# POST /plot
# ---------------------------------------------------------------------------


@router.post(
    "/plot",
    response_model=PlotVerificationResponse,
    status_code=status.HTTP_200_OK,
    summary="Full verification of a single production plot",
    description=(
        "Execute a comprehensive verification of a single production plot "
        "combining coordinate validation, polygon topology verification "
        "(if vertices provided), protected area screening, deforestation "
        "cutoff verification, and composite accuracy scoring. The "
        "verification_level parameter controls depth: quick (coordinate "
        "only), standard (all checks), or deep (all checks + temporal "
        "analysis + enhanced satellite data)."
    ),
    responses={
        200: {"description": "Full plot verification result"},
        400: {"model": ErrorResponse, "description": "Invalid request"},
        401: {"model": ErrorResponse, "description": "Authentication required"},
        403: {"model": ErrorResponse, "description": "Insufficient permissions"},
        429: {"model": ErrorResponse, "description": "Rate limit exceeded"},
    },
)
async def verify_plot(
    body: PlotVerificationRequest,
    request: Request,
    user: AuthUser = Depends(
        require_permission("eudr-geolocation:verification:write")
    ),
    _rate: None = Depends(rate_limit_heavy),
) -> PlotVerificationResponse:
    """Execute full verification of a production plot.

    Runs coordinate validation, polygon verification, protected area
    screening, deforestation verification, and accuracy scoring
    based on the requested verification level.

    Args:
        body: Plot verification request with coordinates and metadata.
        user: Authenticated user with verification:write permission.

    Returns:
        PlotVerificationResponse with all check results and accuracy score.

    Raises:
        HTTPException: 400 if request invalid, 500 on processing error.
    """
    start = time.monotonic()
    verification_id = f"ver-{uuid.uuid4().hex[:12]}"

    logger.info(
        "Plot verification: user=%s plot_id=%s level=%s lat=%.6f lon=%.6f",
        user.user_id,
        body.plot_id,
        body.verification_level,
        body.lat,
        body.lon,
    )

    try:
        service = get_verification_service()

        result = service.verify_plot(
            plot_id=body.plot_id,
            lat=body.lat,
            lon=body.lon,
            polygon_vertices=body.polygon_vertices,
            declared_area_hectares=body.declared_area_hectares,
            declared_country_code=body.declared_country_code,
            commodity=body.commodity,
            verification_level=body.verification_level,
        )

        elapsed = time.monotonic() - start

        # Build response from service result
        coord_resp = None
        if hasattr(result, "coordinate_result") and result.coordinate_result:
            cr = result.coordinate_result
            coord_resp = CoordinateValidationResponse(
                validation_id=cr.validation_id,
                lat=cr.lat,
                lon=cr.lon,
                is_valid=cr.is_valid,
                wgs84_valid=cr.wgs84_valid,
                precision_decimal_places=cr.precision_decimal_places,
                precision_score=cr.precision_score,
                transposition_detected=cr.transposition_detected,
                country_match=cr.country_match,
                resolved_country=cr.resolved_country,
                is_on_land=cr.is_on_land,
                is_duplicate=cr.is_duplicate,
                elevation_m=cr.elevation_m,
                elevation_plausible=cr.elevation_plausible,
                cluster_anomaly=cr.cluster_anomaly,
                issues=[i.to_dict() for i in cr.issues],
                provenance_hash=cr.provenance_hash,
                validated_at=cr.validated_at,
            )

        poly_resp = None
        if hasattr(result, "polygon_result") and result.polygon_result:
            pr = result.polygon_result
            poly_resp = PolygonVerificationResponse(
                verification_id=pr.verification_id,
                is_valid=pr.is_valid,
                ring_closed=pr.ring_closed,
                winding_order_ccw=pr.winding_order_ccw,
                has_self_intersection=pr.has_self_intersection,
                vertex_count=pr.vertex_count,
                calculated_area_ha=pr.calculated_area_ha,
                declared_area_ha=pr.declared_area_ha,
                area_within_tolerance=pr.area_within_tolerance,
                area_tolerance_pct=pr.area_tolerance_pct,
                is_sliver=pr.is_sliver,
                has_spikes=pr.has_spikes,
                spike_vertex_indices=pr.spike_vertex_indices,
                vertex_density_ok=pr.vertex_density_ok,
                max_area_ok=pr.max_area_ok,
                issues=[i.to_dict() for i in pr.issues],
                repair_suggestions=[r.to_dict() for r in pr.repair_suggestions],
                provenance_hash=pr.provenance_hash,
                verified_at=pr.verified_at,
            )

        prot_resp = None
        if hasattr(result, "protected_area_result") and result.protected_area_result:
            par = result.protected_area_result
            prot_resp = ProtectedAreaScreenResponse(
                overlaps_protected=par.overlaps_protected,
                protected_area_name=par.protected_area_name,
                protected_area_type=par.protected_area_type,
                overlap_percentage=par.overlap_percentage,
            )

        defor_resp = None
        if hasattr(result, "deforestation_result") and result.deforestation_result:
            dr = result.deforestation_result
            defor_resp = DeforestationVerifyResponse(
                plot_id=body.plot_id,
                deforestation_detected=dr.deforestation_detected,
                alert_count=dr.alert_count,
                forest_loss_ha=dr.forest_loss_ha,
                cutoff_date=dr.cutoff_date,
                confidence=dr.confidence,
            )

        # Collect accuracy score
        accuracy_score = None
        quality_tier = None
        if hasattr(result, "accuracy_score") and result.accuracy_score:
            score = result.accuracy_score
            accuracy_score = score.to_dict()
            quality_tier = score.quality_tier.value

        # Collect all issues
        all_issues: List[Dict[str, Any]] = []
        if hasattr(result, "issues"):
            all_issues = [
                i.to_dict() if hasattr(i, "to_dict") else i
                for i in result.issues
            ]

        overall_pass = getattr(result, "overall_pass", True)
        provenance_hash = getattr(result, "provenance_hash", "")

        # Store verification result
        store = _get_verification_store()
        plot_results = store.setdefault(body.plot_id, {})
        plot_results[verification_id] = {
            "verification_id": verification_id,
            "plot_id": body.plot_id,
            "overall_pass": overall_pass,
            "quality_tier": quality_tier,
            "verification_level": body.verification_level,
            "verified_at": datetime.now(timezone.utc).isoformat(),
        }

        logger.info(
            "Plot verification completed: plot_id=%s verification_id=%s "
            "pass=%s tier=%s elapsed_ms=%.1f",
            body.plot_id,
            verification_id,
            overall_pass,
            quality_tier,
            elapsed * 1000,
        )

        return PlotVerificationResponse(
            verification_id=verification_id,
            plot_id=body.plot_id,
            verification_level=body.verification_level,
            overall_pass=overall_pass,
            coordinate_result=coord_resp,
            polygon_result=poly_resp,
            protected_area_result=prot_resp,
            deforestation_result=defor_resp,
            accuracy_score=accuracy_score,
            quality_tier=quality_tier,
            issues=all_issues,
            processing_time_ms=elapsed * 1000,
            provenance_hash=provenance_hash,
        )

    except ValueError as exc:
        logger.warning(
            "Plot verification error: user=%s plot_id=%s error=%s",
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
            "Plot verification failed: user=%s plot_id=%s error=%s",
            user.user_id,
            body.plot_id,
            exc,
            exc_info=True,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Plot verification failed due to an internal error",
        )


# ---------------------------------------------------------------------------
# GET /plot/{plot_id}
# ---------------------------------------------------------------------------


@router.get(
    "/plot/{plot_id}",
    response_model=PlotVerificationResponse,
    summary="Get latest verification result for a plot",
    description=(
        "Retrieve the most recent verification result for a production "
        "plot. Returns the full verification breakdown including coordinate, "
        "polygon, protected area, deforestation, and accuracy score results."
    ),
    responses={
        200: {"description": "Latest verification result"},
        401: {"model": ErrorResponse, "description": "Authentication required"},
        403: {"model": ErrorResponse, "description": "Insufficient permissions"},
        404: {"model": ErrorResponse, "description": "Plot not found"},
    },
)
async def get_plot_verification(
    plot_id: str,
    request: Request,
    user: AuthUser = Depends(
        require_permission("eudr-geolocation:verification:read")
    ),
    _rate: None = Depends(rate_limit_standard),
) -> PlotVerificationResponse:
    """Get the latest verification result for a plot.

    Args:
        plot_id: Plot identifier.
        user: Authenticated user with verification:read permission.

    Returns:
        PlotVerificationResponse with the latest verification result.

    Raises:
        HTTPException: 404 if plot not found.
    """
    logger.info(
        "Plot verification retrieval: user=%s plot_id=%s",
        user.user_id,
        plot_id,
    )

    try:
        service = get_verification_service()
        result = service.get_latest_verification(plot_id=plot_id)

        if result is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"No verification found for plot {plot_id}",
            )

        # Convert service result to API response
        accuracy_score = None
        quality_tier = None
        if hasattr(result, "accuracy_score") and result.accuracy_score:
            accuracy_score = result.accuracy_score.to_dict()
            quality_tier = result.accuracy_score.quality_tier.value

        return PlotVerificationResponse(
            verification_id=getattr(result, "verification_id", ""),
            plot_id=plot_id,
            verification_level=getattr(result, "verification_level", "standard"),
            overall_pass=getattr(result, "overall_pass", True),
            accuracy_score=accuracy_score,
            quality_tier=quality_tier,
            issues=[
                i.to_dict() if hasattr(i, "to_dict") else i
                for i in getattr(result, "issues", [])
            ],
            provenance_hash=getattr(result, "provenance_hash", ""),
        )

    except HTTPException:
        raise
    except Exception as exc:
        logger.error(
            "Plot verification retrieval failed: plot_id=%s error=%s",
            plot_id,
            exc,
            exc_info=True,
        )
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No verification found for plot {plot_id}",
        )


# ---------------------------------------------------------------------------
# GET /plot/{plot_id}/history
# ---------------------------------------------------------------------------


@router.get(
    "/plot/{plot_id}/history",
    response_model=PlotVerificationHistoryResponse,
    summary="Get verification history for a plot",
    description=(
        "Retrieve the full verification history for a production plot "
        "with pagination. Returns verification summaries ordered by "
        "date with the most recent first."
    ),
    responses={
        200: {"description": "Verification history"},
        401: {"model": ErrorResponse, "description": "Authentication required"},
        403: {"model": ErrorResponse, "description": "Insufficient permissions"},
        404: {"model": ErrorResponse, "description": "Plot not found"},
    },
)
async def get_plot_verification_history(
    plot_id: str,
    request: Request,
    pagination: PaginationParams = Depends(get_pagination),
    user: AuthUser = Depends(
        require_permission("eudr-geolocation:verification:read")
    ),
    _rate: None = Depends(rate_limit_standard),
) -> PlotVerificationHistoryResponse:
    """Get verification history for a plot with pagination.

    Args:
        plot_id: Plot identifier.
        pagination: Limit and offset for pagination.
        user: Authenticated user with verification:read permission.

    Returns:
        PlotVerificationHistoryResponse with paginated history.

    Raises:
        HTTPException: 404 if plot not found.
    """
    logger.info(
        "Plot verification history: user=%s plot_id=%s limit=%d offset=%d",
        user.user_id,
        plot_id,
        pagination.limit,
        pagination.offset,
    )

    try:
        service = get_verification_service()
        history = service.get_verification_history(
            plot_id=plot_id,
            limit=pagination.limit,
            offset=pagination.offset,
        )

        if history is None:
            # Fall back to in-memory store
            store = _get_verification_store()
            plot_data = store.get(plot_id, {})
            all_verifications = list(plot_data.values())
            total = len(all_verifications)
            page = all_verifications[
                pagination.offset : pagination.offset + pagination.limit
            ]

            return PlotVerificationHistoryResponse(
                plot_id=plot_id,
                total_verifications=total,
                verifications=page,
                meta=PaginatedMeta(
                    total=total,
                    limit=pagination.limit,
                    offset=pagination.offset,
                    has_more=(pagination.offset + pagination.limit) < total,
                ),
            )

        total = getattr(history, "total", 0)
        verifications = getattr(history, "verifications", [])

        return PlotVerificationHistoryResponse(
            plot_id=plot_id,
            total_verifications=total,
            verifications=verifications,
            meta=PaginatedMeta(
                total=total,
                limit=pagination.limit,
                offset=pagination.offset,
                has_more=(pagination.offset + pagination.limit) < total,
            ),
        )

    except Exception as exc:
        logger.error(
            "Plot verification history failed: plot_id=%s error=%s",
            plot_id,
            exc,
            exc_info=True,
        )
        # Return empty history rather than 500 for missing plots
        return PlotVerificationHistoryResponse(
            plot_id=plot_id,
            total_verifications=0,
            verifications=[],
            meta=PaginatedMeta(
                total=0,
                limit=pagination.limit,
                offset=pagination.offset,
                has_more=False,
            ),
        )
