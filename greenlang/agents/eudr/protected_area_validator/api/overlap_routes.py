# -*- coding: utf-8 -*-
"""
Overlap Detection Routes - AGENT-EUDR-022 Protected Area Validator API

Endpoints for detecting and analyzing spatial overlaps between supply chain
plots and protected areas. Supports single-plot detection, detailed analysis,
bulk processing, and lookups by plot or area.

Endpoints:
    POST /overlap/detect            - Detect plot-protected area overlaps
    POST /overlap/analyze           - Detailed overlap analysis
    POST /overlap/bulk              - Bulk overlap detection
    GET  /overlap/by-plot/{plot_id} - Get overlaps for a plot
    GET  /overlap/by-area/{area_id} - Get overlaps for a protected area

Auth: eudr-pav:overlap:{create|read}

Author: GreenLang Platform Team
Date: March 2026
PRD: AGENT-EUDR-022, OverlapDetector Engine
"""

from __future__ import annotations

import hashlib
import logging
import time
from decimal import Decimal
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Request, status

from greenlang.agents.eudr.protected_area_validator.api.dependencies import (
    AuthUser,
    get_overlap_detector,
    rate_limit_heavy,
    rate_limit_standard,
    rate_limit_write,
    require_permission,
)
from greenlang.agents.eudr.protected_area_validator.api.schemas import (
    DesignationStatusEnum,
    ErrorResponse,
    MetadataSchema,
    OverlapAnalyzeRequest,
    OverlapAnalyzeResponse,
    OverlapBulkRequest,
    OverlapBulkResponse,
    OverlapBulkResultEntry,
    OverlapByAreaResponse,
    OverlapByPlotResponse,
    OverlapDetectRequest,
    OverlapDetectResponse,
    OverlapEntry,
    OverlapTypeEnum,
    ProtectedAreaTypeEnum,
    ProvenanceInfo,
    RiskLevelEnum,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/overlap", tags=["Overlap Detection"])


def _compute_provenance(input_data: Any, output_data: Any) -> str:
    """Compute SHA-256 provenance hash for audit trail."""
    data_str = f"{input_data}{output_data}"
    return hashlib.sha256(data_str.encode("utf-8")).hexdigest()


def _parse_overlap_entry(o: Dict[str, Any]) -> OverlapEntry:
    """Parse a raw overlap result into an OverlapEntry schema."""
    return OverlapEntry(
        overlap_id=o.get("overlap_id", ""),
        area_id=o.get("area_id", ""),
        area_name=o.get("area_name", ""),
        area_type=ProtectedAreaTypeEnum(o.get("area_type", "other")),
        country_code=o.get("country_code", ""),
        overlap_type=OverlapTypeEnum(o.get("overlap_type", "none")),
        overlap_area_km2=Decimal(str(o.get("overlap_area_km2", 0)))
        if o.get("overlap_area_km2") is not None else None,
        overlap_percentage=Decimal(str(o.get("overlap_percentage", 0)))
        if o.get("overlap_percentage") is not None else None,
        distance_km=Decimal(str(o.get("distance_km", 0))),
        buffer_zone_km=Decimal(str(o.get("buffer_zone_km", 5))),
        is_within_buffer=o.get("is_within_buffer", False),
        risk_level=RiskLevelEnum(o.get("risk_level", "negligible")),
        designation_status=DesignationStatusEnum(o.get("designation_status", "unknown")),
        iucn_category=o.get("iucn_category"),
    )


# ---------------------------------------------------------------------------
# POST /overlap/detect
# ---------------------------------------------------------------------------


@router.post(
    "/detect",
    response_model=OverlapDetectResponse,
    status_code=status.HTTP_200_OK,
    summary="Detect plot-protected area overlaps",
    description=(
        "Detect spatial overlaps between a supply chain plot boundary and "
        "registered protected areas. Returns full, partial, buffer-only, "
        "and adjacent overlaps with risk levels."
    ),
    responses={
        200: {"description": "Overlap detection completed"},
        400: {"model": ErrorResponse, "description": "Invalid request"},
        401: {"model": ErrorResponse, "description": "Authentication required"},
        403: {"model": ErrorResponse, "description": "Insufficient permissions"},
        429: {"model": ErrorResponse, "description": "Rate limit exceeded"},
    },
)
async def detect_overlaps(
    request: Request,
    body: OverlapDetectRequest,
    user: AuthUser = Depends(
        require_permission("eudr-pav:overlap:create")
    ),
    _rate: None = Depends(rate_limit_write),
) -> OverlapDetectResponse:
    """Detect spatial overlaps between a plot and protected areas.

    Args:
        body: Detection request with plot boundary and search parameters.
        user: Authenticated user with overlap:create permission.

    Returns:
        OverlapDetectResponse with detected overlaps.
    """
    start = time.monotonic()

    try:
        engine = get_overlap_detector()
        result = engine.detect_overlaps(
            plot_id=body.plot_id,
            plot_boundary=[
                {"latitude": float(p.latitude), "longitude": float(p.longitude)}
                for p in body.plot_boundary.coordinates
            ],
            plot_center=(
                {"latitude": float(body.plot_center.latitude), "longitude": float(body.plot_center.longitude)}
                if body.plot_center else None
            ),
            include_buffer_zones=body.include_buffer_zones,
            max_distance_km=float(body.max_distance_km),
            area_types=[t.value for t in body.area_types] if body.area_types else None,
            commodities=[c.value for c in body.commodities] if body.commodities else None,
        )

        overlaps = [_parse_overlap_entry(o) for o in result.get("overlaps", [])]
        has_direct = any(
            o.overlap_type in (OverlapTypeEnum.FULL, OverlapTypeEnum.PARTIAL)
            for o in overlaps
        )
        has_buffer = any(o.is_within_buffer for o in overlaps)

        risk_priority = {
            RiskLevelEnum.CRITICAL: 5,
            RiskLevelEnum.HIGH: 4,
            RiskLevelEnum.MEDIUM: 3,
            RiskLevelEnum.LOW: 2,
            RiskLevelEnum.NEGLIGIBLE: 1,
        }
        highest_risk = max(
            (o.risk_level for o in overlaps),
            key=lambda r: risk_priority.get(r, 0),
            default=RiskLevelEnum.NEGLIGIBLE,
        )

        elapsed_ms = Decimal(str((time.monotonic() - start) * 1000))
        provenance_hash = _compute_provenance(
            f"detect_overlap:{body.plot_id}", str(len(overlaps)),
        )

        logger.info(
            "Overlap detection: plot_id=%s overlaps=%d direct=%s buffer=%s risk=%s operator=%s",
            body.plot_id,
            len(overlaps),
            has_direct,
            has_buffer,
            highest_risk.value,
            user.operator_id or user.user_id,
        )

        return OverlapDetectResponse(
            plot_id=body.plot_id,
            overlaps=overlaps,
            total_overlaps=len(overlaps),
            has_direct_overlap=has_direct,
            has_buffer_overlap=has_buffer,
            highest_risk_level=highest_risk,
            provenance=ProvenanceInfo(
                provenance_hash=provenance_hash,
                processing_time_ms=elapsed_ms.quantize(Decimal("0.01")),
            ),
            metadata=MetadataSchema(
                data_sources=["OverlapDetector", "WDPA", "OECM"],
            ),
        )

    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Overlap detection failed: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Overlap detection failed",
        )


# ---------------------------------------------------------------------------
# POST /overlap/analyze
# ---------------------------------------------------------------------------


@router.post(
    "/analyze",
    response_model=OverlapAnalyzeResponse,
    status_code=status.HTTP_200_OK,
    summary="Detailed overlap analysis between plot and protected area",
    description=(
        "Perform a detailed spatial analysis of the overlap between a specific "
        "plot and a specific protected area, including intersection geometry, "
        "risk assessment, and regulatory implications."
    ),
    responses={
        200: {"description": "Analysis completed"},
        400: {"model": ErrorResponse, "description": "Invalid request"},
        401: {"model": ErrorResponse, "description": "Authentication required"},
        403: {"model": ErrorResponse, "description": "Insufficient permissions"},
        404: {"model": ErrorResponse, "description": "Plot or area not found"},
    },
)
async def analyze_overlap(
    request: Request,
    body: OverlapAnalyzeRequest,
    user: AuthUser = Depends(
        require_permission("eudr-pav:overlap:create")
    ),
    _rate: None = Depends(rate_limit_write),
) -> OverlapAnalyzeResponse:
    """Perform detailed overlap analysis.

    Args:
        body: Analysis request with plot ID and area ID.
        user: Authenticated user with overlap:create permission.

    Returns:
        OverlapAnalyzeResponse with detailed analysis.
    """
    start = time.monotonic()

    try:
        engine = get_overlap_detector()
        result = engine.analyze_overlap(
            plot_id=body.plot_id,
            area_id=body.area_id,
            include_boundary_detail=body.include_boundary_detail,
            include_historical=body.include_historical,
            commodities=[c.value for c in body.commodities] if body.commodities else None,
        )

        if result is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Plot {body.plot_id} or area {body.area_id} not found",
            )

        elapsed_ms = Decimal(str((time.monotonic() - start) * 1000))
        provenance_hash = _compute_provenance(
            f"analyze_overlap:{body.plot_id}:{body.area_id}",
            result.get("overlap_type", "none"),
        )

        logger.info(
            "Overlap analyzed: plot=%s area=%s type=%s risk=%s operator=%s",
            body.plot_id,
            body.area_id,
            result.get("overlap_type", "none"),
            result.get("risk_level", "negligible"),
            user.operator_id or user.user_id,
        )

        return OverlapAnalyzeResponse(
            plot_id=body.plot_id,
            area_id=body.area_id,
            area_name=result.get("area_name", ""),
            overlap_type=OverlapTypeEnum(result.get("overlap_type", "none")),
            overlap_area_km2=Decimal(str(result.get("overlap_area_km2", 0))),
            overlap_percentage_plot=Decimal(str(result.get("overlap_percentage_plot", 0))),
            overlap_percentage_area=Decimal(str(result.get("overlap_percentage_area", 0))),
            distance_to_boundary_km=Decimal(str(result.get("distance_to_boundary_km", 0))),
            distance_to_core_km=Decimal(str(result.get("distance_to_core_km", 0)))
            if result.get("distance_to_core_km") is not None else None,
            risk_level=RiskLevelEnum(result.get("risk_level", "negligible")),
            risk_factors=result.get("risk_factors", []),
            regulatory_implications=result.get("regulatory_implications", []),
            recommended_actions=result.get("recommended_actions", []),
            historical_changes=result.get("historical_changes"),
            provenance=ProvenanceInfo(
                provenance_hash=provenance_hash,
                processing_time_ms=elapsed_ms.quantize(Decimal("0.01")),
            ),
            metadata=MetadataSchema(
                data_sources=["OverlapDetector", "WDPA"],
            ),
        )

    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Overlap analysis failed: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Overlap analysis failed",
        )


# ---------------------------------------------------------------------------
# POST /overlap/bulk
# ---------------------------------------------------------------------------


@router.post(
    "/bulk",
    response_model=OverlapBulkResponse,
    status_code=status.HTTP_200_OK,
    summary="Bulk overlap detection across multiple plots",
    description=(
        "Perform overlap detection across multiple supply chain plots "
        "in a single batch request. Maximum 500 plots per request."
    ),
    responses={
        200: {"description": "Bulk detection completed"},
        400: {"model": ErrorResponse, "description": "Invalid request"},
        401: {"model": ErrorResponse, "description": "Authentication required"},
        403: {"model": ErrorResponse, "description": "Insufficient permissions"},
        429: {"model": ErrorResponse, "description": "Rate limit exceeded"},
    },
)
async def bulk_detect_overlaps(
    request: Request,
    body: OverlapBulkRequest,
    user: AuthUser = Depends(
        require_permission("eudr-pav:overlap:create")
    ),
    _rate: None = Depends(rate_limit_heavy),
) -> OverlapBulkResponse:
    """Perform bulk overlap detection.

    Args:
        body: Bulk request with multiple plots.
        user: Authenticated user with overlap:create permission.

    Returns:
        OverlapBulkResponse with per-plot results.
    """
    start = time.monotonic()

    try:
        engine = get_overlap_detector()

        plots_data = []
        for p in body.plots:
            plots_data.append({
                "plot_id": p.plot_id,
                "plot_boundary": [
                    {"latitude": float(pt.latitude), "longitude": float(pt.longitude)}
                    for pt in p.plot_boundary.coordinates
                ],
                "plot_center": (
                    {"latitude": float(p.plot_center.latitude), "longitude": float(p.plot_center.longitude)}
                    if p.plot_center else None
                ),
                "include_buffer_zones": body.include_buffer_zones,
                "max_distance_km": float(p.max_distance_km),
            })

        result = engine.bulk_detect(plots=plots_data)

        results = []
        total_overlaps = 0
        plots_with_overlaps = 0
        plots_with_direct = 0
        failed = 0

        for r in result.get("results", []):
            overlaps = [_parse_overlap_entry(o) for o in r.get("overlaps", [])]
            has_direct = any(
                o.overlap_type in (OverlapTypeEnum.FULL, OverlapTypeEnum.PARTIAL)
                for o in overlaps
            )
            risk_priority = {
                RiskLevelEnum.CRITICAL: 5, RiskLevelEnum.HIGH: 4,
                RiskLevelEnum.MEDIUM: 3, RiskLevelEnum.LOW: 2,
                RiskLevelEnum.NEGLIGIBLE: 1,
            }
            highest = max(
                (o.risk_level for o in overlaps),
                key=lambda rl: risk_priority.get(rl, 0),
                default=RiskLevelEnum.NEGLIGIBLE,
            )

            results.append(OverlapBulkResultEntry(
                plot_id=r.get("plot_id", ""),
                total_overlaps=len(overlaps),
                has_direct_overlap=has_direct,
                highest_risk_level=highest,
                overlaps=overlaps,
                error=r.get("error"),
            ))

            if r.get("error"):
                failed += 1
            else:
                total_overlaps += len(overlaps)
                if overlaps:
                    plots_with_overlaps += 1
                if has_direct:
                    plots_with_direct += 1

        elapsed_ms = Decimal(str((time.monotonic() - start) * 1000))
        provenance_hash = _compute_provenance(
            f"bulk_detect:{len(body.plots)}", str(total_overlaps),
        )

        logger.info(
            "Bulk overlap detection: plots=%d overlaps=%d with_overlaps=%d failed=%d operator=%s",
            len(body.plots),
            total_overlaps,
            plots_with_overlaps,
            failed,
            user.operator_id or user.user_id,
        )

        return OverlapBulkResponse(
            results=results,
            total_plots_processed=len(body.plots),
            total_overlaps_found=total_overlaps,
            plots_with_overlaps=plots_with_overlaps,
            plots_with_direct_overlap=plots_with_direct,
            failed_count=failed,
            provenance=ProvenanceInfo(
                provenance_hash=provenance_hash,
                processing_time_ms=elapsed_ms.quantize(Decimal("0.01")),
            ),
            metadata=MetadataSchema(
                data_sources=["OverlapDetector", "WDPA", "OECM"],
            ),
        )

    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Bulk overlap detection failed: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Bulk overlap detection failed",
        )


# ---------------------------------------------------------------------------
# GET /overlap/by-plot/{plot_id}
# ---------------------------------------------------------------------------


@router.get(
    "/by-plot/{plot_id}",
    response_model=OverlapByPlotResponse,
    summary="Get overlaps for a specific plot",
    description=(
        "Retrieve all known overlaps between a supply chain plot and "
        "protected areas from previous detection runs."
    ),
    responses={
        200: {"description": "Overlaps retrieved"},
        401: {"model": ErrorResponse, "description": "Authentication required"},
        403: {"model": ErrorResponse, "description": "Insufficient permissions"},
        404: {"model": ErrorResponse, "description": "Plot not found"},
    },
)
async def get_overlaps_by_plot(
    plot_id: str,
    request: Request,
    user: AuthUser = Depends(
        require_permission("eudr-pav:overlap:read")
    ),
    _rate: None = Depends(rate_limit_standard),
) -> OverlapByPlotResponse:
    """Get overlaps for a specific plot.

    Args:
        plot_id: Supply chain plot identifier.
        user: Authenticated user with overlap:read permission.

    Returns:
        OverlapByPlotResponse with plot overlaps.
    """
    start = time.monotonic()

    try:
        engine = get_overlap_detector()
        result = engine.get_overlaps_by_plot(plot_id=plot_id)

        if result is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Plot not found: {plot_id}",
            )

        overlaps = [_parse_overlap_entry(o) for o in result.get("overlaps", [])]
        risk_priority = {
            RiskLevelEnum.CRITICAL: 5, RiskLevelEnum.HIGH: 4,
            RiskLevelEnum.MEDIUM: 3, RiskLevelEnum.LOW: 2,
            RiskLevelEnum.NEGLIGIBLE: 1,
        }
        highest = max(
            (o.risk_level for o in overlaps),
            key=lambda rl: risk_priority.get(rl, 0),
            default=RiskLevelEnum.NEGLIGIBLE,
        )

        elapsed_ms = Decimal(str((time.monotonic() - start) * 1000))
        provenance_hash = _compute_provenance(f"by_plot:{plot_id}", str(len(overlaps)))

        logger.info(
            "Overlaps by plot: plot_id=%s overlaps=%d operator=%s",
            plot_id,
            len(overlaps),
            user.operator_id or user.user_id,
        )

        return OverlapByPlotResponse(
            plot_id=plot_id,
            overlaps=overlaps,
            total_overlaps=len(overlaps),
            highest_risk_level=highest,
            provenance=ProvenanceInfo(
                provenance_hash=provenance_hash,
                processing_time_ms=elapsed_ms.quantize(Decimal("0.01")),
            ),
            metadata=MetadataSchema(
                data_sources=["OverlapDetector"],
            ),
        )

    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Overlaps by plot retrieval failed: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Overlaps by plot retrieval failed",
        )


# ---------------------------------------------------------------------------
# GET /overlap/by-area/{area_id}
# ---------------------------------------------------------------------------


@router.get(
    "/by-area/{area_id}",
    response_model=OverlapByAreaResponse,
    summary="Get overlaps for a specific protected area",
    description=(
        "Retrieve all known overlaps between a protected area and "
        "supply chain plots from previous detection runs."
    ),
    responses={
        200: {"description": "Overlaps retrieved"},
        401: {"model": ErrorResponse, "description": "Authentication required"},
        403: {"model": ErrorResponse, "description": "Insufficient permissions"},
        404: {"model": ErrorResponse, "description": "Area not found"},
    },
)
async def get_overlaps_by_area(
    area_id: str,
    request: Request,
    user: AuthUser = Depends(
        require_permission("eudr-pav:overlap:read")
    ),
    _rate: None = Depends(rate_limit_standard),
) -> OverlapByAreaResponse:
    """Get overlaps for a specific protected area.

    Args:
        area_id: Protected area identifier.
        user: Authenticated user with overlap:read permission.

    Returns:
        OverlapByAreaResponse with area overlaps.
    """
    start = time.monotonic()

    try:
        engine = get_overlap_detector()
        result = engine.get_overlaps_by_area(area_id=area_id)

        if result is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Protected area not found: {area_id}",
            )

        overlaps = [_parse_overlap_entry(o) for o in result.get("overlaps", [])]
        affected_plots = len({o.get("plot_id", "") for o in result.get("overlaps", [])})

        elapsed_ms = Decimal(str((time.monotonic() - start) * 1000))
        provenance_hash = _compute_provenance(f"by_area:{area_id}", str(len(overlaps)))

        logger.info(
            "Overlaps by area: area_id=%s overlaps=%d plots=%d operator=%s",
            area_id,
            len(overlaps),
            affected_plots,
            user.operator_id or user.user_id,
        )

        return OverlapByAreaResponse(
            area_id=area_id,
            area_name=result.get("area_name", ""),
            overlaps=overlaps,
            total_overlaps=len(overlaps),
            total_affected_plots=affected_plots,
            provenance=ProvenanceInfo(
                provenance_hash=provenance_hash,
                processing_time_ms=elapsed_ms.quantize(Decimal("0.01")),
            ),
            metadata=MetadataSchema(
                data_sources=["OverlapDetector"],
            ),
        )

    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Overlaps by area retrieval failed: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Overlaps by area retrieval failed",
        )
