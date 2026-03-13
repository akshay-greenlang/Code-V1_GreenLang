# -*- coding: utf-8 -*-
"""
Land Rights Overlap Routes - AGENT-EUDR-021 Indigenous Rights Checker API

Endpoints for spatial overlap analysis between supply chain plots and
indigenous territories. Detects full, partial, boundary, and buffer zone
overlaps to determine FPIC requirements and compliance risk.

Endpoints:
    POST /overlap/analyze                       - Analyze plot-territory overlaps
    GET  /overlap/by-plot/{plot_id}             - Get overlaps for a plot
    GET  /overlap/by-territory/{territory_id}   - Get overlaps for a territory
    POST /overlap/bulk                          - Bulk overlap analysis

Overlap types: full, partial, boundary, buffer_zone, none
Severity: critical (>80%), high (50-80%), medium (20-50%), low (<20%)

Author: GreenLang Platform Team
Date: March 2026
PRD: AGENT-EUDR-021, OverlapAnalyzer Engine
"""

from __future__ import annotations

import hashlib
import logging
import time
from decimal import Decimal
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query, Request, status

from greenlang.agents.eudr.indigenous_rights_checker.api.dependencies import (
    AuthUser,
    PaginationParams,
    get_overlap_analyzer,
    get_pagination,
    rate_limit_heavy,
    rate_limit_standard,
    rate_limit_write,
    require_permission,
)
from greenlang.agents.eudr.indigenous_rights_checker.api.schemas import (
    ErrorResponse,
    FPICStatusEnum,
    MetadataSchema,
    OverlapAnalyzeRequest,
    OverlapAnalyzeResponse,
    OverlapBulkRequest,
    OverlapBulkResponse,
    OverlapBulkResultEntry,
    OverlapEntry,
    OverlapListResponse,
    OverlapSeverityEnum,
    OverlapTypeEnum,
    PaginatedMeta,
    ProvenanceInfo,
    SortOrderEnum,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/overlap", tags=["Land Rights Overlap"])


def _compute_provenance(input_data: str, output_data: str) -> str:
    """Compute SHA-256 provenance hash for audit trail."""
    data_str = f"{input_data}{output_data}"
    return hashlib.sha256(data_str.encode("utf-8")).hexdigest()


def _build_overlap_entry(entry: dict) -> OverlapEntry:
    """Build an OverlapEntry from engine result dictionary."""
    return OverlapEntry(
        overlap_id=entry.get("overlap_id", ""),
        plot_id=entry.get("plot_id", ""),
        territory_id=entry.get("territory_id", ""),
        territory_name=entry.get("territory_name"),
        overlap_type=OverlapTypeEnum(entry.get("overlap_type", "no_overlap")),
        overlap_severity=OverlapSeverityEnum(
            entry.get("overlap_severity", "informational")
        ),
        overlap_area_ha=Decimal(str(entry.get("overlap_area_ha")))
        if entry.get("overlap_area_ha") is not None else None,
        overlap_percentage=Decimal(str(entry.get("overlap_percentage")))
        if entry.get("overlap_percentage") is not None else None,
        distance_km=Decimal(str(entry.get("distance_km")))
        if entry.get("distance_km") is not None else None,
        fpic_status=FPICStatusEnum(entry.get("fpic_status"))
        if entry.get("fpic_status") else None,
        community_name=entry.get("community_name"),
        requires_fpic=entry.get("requires_fpic", True),
        detected_at=entry.get("detected_at"),
    )


# ---------------------------------------------------------------------------
# POST /overlap/analyze
# ---------------------------------------------------------------------------


@router.post(
    "/analyze",
    response_model=OverlapAnalyzeResponse,
    status_code=status.HTTP_200_OK,
    summary="Analyze plot-territory overlaps",
    description=(
        "Analyze spatial overlaps between a supply chain plot and registered "
        "indigenous territories. Uses GIS intersection for full/partial overlap "
        "detection and optional buffer zone proximity analysis. Returns overlap "
        "details, severity, FPIC requirements, and aggregate risk score."
    ),
    responses={
        200: {"description": "Overlap analysis completed"},
        400: {"model": ErrorResponse, "description": "Invalid request"},
        401: {"model": ErrorResponse, "description": "Authentication required"},
        403: {"model": ErrorResponse, "description": "Insufficient permissions"},
        404: {"model": ErrorResponse, "description": "Plot not found"},
        429: {"model": ErrorResponse, "description": "Rate limit exceeded"},
    },
)
async def analyze_overlap(
    request: Request,
    body: OverlapAnalyzeRequest,
    user: AuthUser = Depends(
        require_permission("eudr-irc:overlap:analyze")
    ),
    _rate: None = Depends(rate_limit_write),
) -> OverlapAnalyzeResponse:
    """Analyze overlaps between a plot and indigenous territories.

    Args:
        body: Overlap analysis request with plot ID and optional boundary.
        user: Authenticated user with overlap:analyze permission.

    Returns:
        OverlapAnalyzeResponse with detected overlaps and risk score.
    """
    start = time.monotonic()

    try:
        engine = get_overlap_analyzer()

        plot_boundary_data = None
        if body.plot_boundary:
            plot_boundary_data = {
                "coordinates": [
                    {"latitude": float(p.latitude), "longitude": float(p.longitude)}
                    for p in body.plot_boundary.coordinates
                ],
                "srid": body.plot_boundary.srid,
            }

        result = engine.analyze_overlap(
            plot_id=body.plot_id,
            plot_boundary=plot_boundary_data,
            territory_ids=body.territory_ids,
            buffer_km=float(body.buffer_km) if body.buffer_km else None,
            include_buffer_zones=body.include_buffer_zones,
            commodity=body.commodity.value if body.commodity else None,
            analyzed_by=user.user_id,
        )

        overlaps = [
            _build_overlap_entry(entry) for entry in result.get("overlaps", [])
        ]
        has_critical = any(
            o.overlap_severity == OverlapSeverityEnum.CRITICAL for o in overlaps
        )
        requires_fpic = any(o.requires_fpic for o in overlaps)

        elapsed_ms = Decimal(str((time.monotonic() - start) * 1000))
        provenance_hash = _compute_provenance(
            f"analyze_overlap:{body.plot_id}:{body.buffer_km}",
            str(len(overlaps)),
        )

        logger.info(
            "Overlap analysis completed: plot_id=%s overlaps=%d critical=%s fpic_required=%s operator=%s",
            body.plot_id,
            len(overlaps),
            has_critical,
            requires_fpic,
            user.operator_id or user.user_id,
        )

        return OverlapAnalyzeResponse(
            plot_id=body.plot_id,
            overlaps=overlaps,
            total_overlaps=len(overlaps),
            has_critical_overlap=has_critical,
            requires_fpic=requires_fpic,
            risk_score=Decimal(str(result.get("risk_score", 0)))
            if result.get("risk_score") is not None else None,
            provenance=ProvenanceInfo(
                provenance_hash=provenance_hash,
                processing_time_ms=elapsed_ms.quantize(Decimal("0.01")),
            ),
            metadata=MetadataSchema(
                data_sources=["IndigenousRightsChecker", "OverlapAnalyzer"],
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
# GET /overlap/by-plot/{plot_id}
# ---------------------------------------------------------------------------


@router.get(
    "/by-plot/{plot_id}",
    response_model=OverlapListResponse,
    summary="Get overlaps for a plot",
    description=(
        "Retrieve all known territory overlaps for a specific supply chain "
        "plot. Returns previously analyzed overlap records with severity, "
        "type, and FPIC status."
    ),
    responses={
        200: {"description": "Plot overlaps retrieved"},
        401: {"model": ErrorResponse, "description": "Authentication required"},
        403: {"model": ErrorResponse, "description": "Insufficient permissions"},
        404: {"model": ErrorResponse, "description": "Plot not found"},
    },
)
async def get_overlaps_by_plot(
    plot_id: str,
    request: Request,
    overlap_type: Optional[OverlapTypeEnum] = Query(
        None, description="Filter by overlap type"
    ),
    severity: Optional[OverlapSeverityEnum] = Query(
        None, description="Filter by severity"
    ),
    sort_order: Optional[SortOrderEnum] = Query(
        SortOrderEnum.DESC, description="Sort order by detection date"
    ),
    pagination: PaginationParams = Depends(get_pagination),
    user: AuthUser = Depends(
        require_permission("eudr-irc:overlap:read")
    ),
    _rate: None = Depends(rate_limit_standard),
) -> OverlapListResponse:
    """Get all territory overlaps for a plot.

    Args:
        plot_id: Plot identifier.
        overlap_type: Optional overlap type filter.
        severity: Optional severity filter.
        sort_order: Sort direction.
        pagination: Pagination parameters.
        user: Authenticated user.

    Returns:
        OverlapListResponse with overlap records for the plot.
    """
    start = time.monotonic()

    try:
        engine = get_overlap_analyzer()
        result = engine.get_overlaps_by_plot(
            plot_id=plot_id,
            overlap_type=overlap_type.value if overlap_type else None,
            severity=severity.value if severity else None,
            sort_order=sort_order.value if sort_order else "desc",
            limit=pagination.limit,
            offset=pagination.offset,
        )

        if result is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Plot not found: {plot_id}",
            )

        overlaps = [
            _build_overlap_entry(entry) for entry in result.get("overlaps", [])
        ]
        total = result.get("total", len(overlaps))

        elapsed_ms = Decimal(str((time.monotonic() - start) * 1000))
        provenance_hash = _compute_provenance(
            f"overlaps_by_plot:{plot_id}:{overlap_type}:{severity}",
            str(total),
        )

        logger.info(
            "Overlaps by plot retrieved: plot_id=%s total=%d operator=%s",
            plot_id,
            total,
            user.operator_id or user.user_id,
        )

        return OverlapListResponse(
            overlaps=overlaps,
            total_overlaps=total,
            pagination=PaginatedMeta(
                total=total,
                limit=pagination.limit,
                offset=pagination.offset,
                has_more=(pagination.offset + pagination.limit) < total,
            ),
            provenance=ProvenanceInfo(
                provenance_hash=provenance_hash,
                processing_time_ms=elapsed_ms.quantize(Decimal("0.01")),
            ),
            metadata=MetadataSchema(
                data_sources=["IndigenousRightsChecker", "OverlapAnalyzer"],
            ),
        )

    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Overlap retrieval by plot failed: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Overlap retrieval by plot failed",
        )


# ---------------------------------------------------------------------------
# GET /overlap/by-territory/{territory_id}
# ---------------------------------------------------------------------------


@router.get(
    "/by-territory/{territory_id}",
    response_model=OverlapListResponse,
    summary="Get overlaps for a territory",
    description=(
        "Retrieve all known plot overlaps for a specific indigenous territory. "
        "Useful for territory owners and compliance teams to understand "
        "which supply chain plots encroach on indigenous lands."
    ),
    responses={
        200: {"description": "Territory overlaps retrieved"},
        401: {"model": ErrorResponse, "description": "Authentication required"},
        403: {"model": ErrorResponse, "description": "Insufficient permissions"},
        404: {"model": ErrorResponse, "description": "Territory not found"},
    },
)
async def get_overlaps_by_territory(
    territory_id: str,
    request: Request,
    overlap_type: Optional[OverlapTypeEnum] = Query(
        None, description="Filter by overlap type"
    ),
    severity: Optional[OverlapSeverityEnum] = Query(
        None, description="Filter by severity"
    ),
    sort_order: Optional[SortOrderEnum] = Query(
        SortOrderEnum.DESC, description="Sort order by detection date"
    ),
    pagination: PaginationParams = Depends(get_pagination),
    user: AuthUser = Depends(
        require_permission("eudr-irc:overlap:read")
    ),
    _rate: None = Depends(rate_limit_standard),
) -> OverlapListResponse:
    """Get all plot overlaps for an indigenous territory.

    Args:
        territory_id: Territory identifier.
        overlap_type: Optional overlap type filter.
        severity: Optional severity filter.
        sort_order: Sort direction.
        pagination: Pagination parameters.
        user: Authenticated user.

    Returns:
        OverlapListResponse with overlap records for the territory.
    """
    start = time.monotonic()

    try:
        engine = get_overlap_analyzer()
        result = engine.get_overlaps_by_territory(
            territory_id=territory_id,
            overlap_type=overlap_type.value if overlap_type else None,
            severity=severity.value if severity else None,
            sort_order=sort_order.value if sort_order else "desc",
            limit=pagination.limit,
            offset=pagination.offset,
        )

        if result is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Territory not found: {territory_id}",
            )

        overlaps = [
            _build_overlap_entry(entry) for entry in result.get("overlaps", [])
        ]
        total = result.get("total", len(overlaps))

        elapsed_ms = Decimal(str((time.monotonic() - start) * 1000))
        provenance_hash = _compute_provenance(
            f"overlaps_by_territory:{territory_id}:{overlap_type}:{severity}",
            str(total),
        )

        logger.info(
            "Overlaps by territory retrieved: territory_id=%s total=%d operator=%s",
            territory_id,
            total,
            user.operator_id or user.user_id,
        )

        return OverlapListResponse(
            overlaps=overlaps,
            total_overlaps=total,
            pagination=PaginatedMeta(
                total=total,
                limit=pagination.limit,
                offset=pagination.offset,
                has_more=(pagination.offset + pagination.limit) < total,
            ),
            provenance=ProvenanceInfo(
                provenance_hash=provenance_hash,
                processing_time_ms=elapsed_ms.quantize(Decimal("0.01")),
            ),
            metadata=MetadataSchema(
                data_sources=["IndigenousRightsChecker", "OverlapAnalyzer"],
            ),
        )

    except HTTPException:
        raise
    except Exception as exc:
        logger.error(
            "Overlap retrieval by territory failed: %s", exc, exc_info=True
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Overlap retrieval by territory failed",
        )


# ---------------------------------------------------------------------------
# POST /overlap/bulk
# ---------------------------------------------------------------------------


@router.post(
    "/bulk",
    response_model=OverlapBulkResponse,
    status_code=status.HTTP_200_OK,
    summary="Bulk overlap analysis",
    description=(
        "Perform overlap analysis for multiple plots in a single request. "
        "Efficient for batch supply chain screening. Supports up to 500 "
        "plots per request with optional buffer zone analysis."
    ),
    responses={
        200: {"description": "Bulk overlap analysis completed"},
        400: {"model": ErrorResponse, "description": "Invalid request (max 500 plots)"},
        401: {"model": ErrorResponse, "description": "Authentication required"},
        403: {"model": ErrorResponse, "description": "Insufficient permissions"},
        429: {"model": ErrorResponse, "description": "Rate limit exceeded"},
    },
)
async def bulk_overlap_analysis(
    request: Request,
    body: OverlapBulkRequest,
    user: AuthUser = Depends(
        require_permission("eudr-irc:overlap:analyze")
    ),
    _rate: None = Depends(rate_limit_heavy),
) -> OverlapBulkResponse:
    """Perform bulk overlap analysis for multiple plots.

    Args:
        body: Bulk analysis request with plot IDs and options.
        user: Authenticated user with overlap:analyze permission.

    Returns:
        OverlapBulkResponse with per-plot results.
    """
    start = time.monotonic()

    try:
        engine = get_overlap_analyzer()
        result = engine.bulk_analyze(
            plot_ids=body.plot_ids,
            buffer_km=float(body.buffer_km) if body.buffer_km else None,
            include_buffer_zones=body.include_buffer_zones,
            commodity=body.commodity.value if body.commodity else None,
            analyzed_by=user.user_id,
        )

        results = []
        plots_with_overlaps = 0
        plots_requiring_fpic = 0

        for plot_result in result.get("results", []):
            overlaps = [
                _build_overlap_entry(entry)
                for entry in plot_result.get("overlaps", [])
            ]
            has_critical = any(
                o.overlap_severity == OverlapSeverityEnum.CRITICAL for o in overlaps
            )
            requires_fpic = any(o.requires_fpic for o in overlaps)

            if overlaps:
                plots_with_overlaps += 1
            if requires_fpic:
                plots_requiring_fpic += 1

            results.append(
                OverlapBulkResultEntry(
                    plot_id=plot_result.get("plot_id", ""),
                    total_overlaps=len(overlaps),
                    has_critical_overlap=has_critical,
                    requires_fpic=requires_fpic,
                    risk_score=Decimal(str(plot_result.get("risk_score", 0)))
                    if plot_result.get("risk_score") is not None else None,
                    overlaps=overlaps,
                )
            )

        elapsed_ms = Decimal(str((time.monotonic() - start) * 1000))
        provenance_hash = _compute_provenance(
            f"bulk_overlap:{len(body.plot_ids)}:{body.buffer_km}",
            f"{plots_with_overlaps}/{plots_requiring_fpic}",
        )

        logger.info(
            "Bulk overlap analysis: plots=%d with_overlaps=%d requiring_fpic=%d operator=%s",
            len(body.plot_ids),
            plots_with_overlaps,
            plots_requiring_fpic,
            user.operator_id or user.user_id,
        )

        return OverlapBulkResponse(
            total_plots=len(body.plot_ids),
            plots_with_overlaps=plots_with_overlaps,
            plots_requiring_fpic=plots_requiring_fpic,
            results=results,
            provenance=ProvenanceInfo(
                provenance_hash=provenance_hash,
                processing_time_ms=elapsed_ms.quantize(Decimal("0.01")),
            ),
            metadata=MetadataSchema(
                data_sources=["IndigenousRightsChecker", "OverlapAnalyzer"],
            ),
        )

    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Bulk overlap analysis failed: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Bulk overlap analysis failed",
        )
