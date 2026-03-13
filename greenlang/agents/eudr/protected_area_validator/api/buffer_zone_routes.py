# -*- coding: utf-8 -*-
"""
Buffer Zone Routes - AGENT-EUDR-022 Protected Area Validator API

Endpoints for monitoring buffer zone compliance, detecting violations,
performing proximity analysis, and bulk monitoring of supply chain plots
against protected area buffer zones.

Endpoints:
    POST /buffer-zones/monitor  - Monitor buffer zone compliance for a plot
    GET  /buffer-zones/violations - List buffer zone violations
    POST /buffer-zones/analyze  - Proximity analysis for a point
    POST /buffer-zones/bulk     - Bulk buffer zone monitoring

Auth: eudr-pav:buffer-zone:{create|read}

Author: GreenLang Platform Team
Date: March 2026
PRD: AGENT-EUDR-022, BufferZoneMonitor Engine
"""

from __future__ import annotations

import hashlib
import logging
import time
from decimal import Decimal
from typing import Any, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, Request, status

from greenlang.agents.eudr.protected_area_validator.api.dependencies import (
    AuthUser,
    PaginationParams,
    get_buffer_zone_monitor,
    get_pagination,
    rate_limit_heavy,
    rate_limit_standard,
    rate_limit_write,
    require_permission,
)
from greenlang.agents.eudr.protected_area_validator.api.schemas import (
    BufferZoneAnalyzeEntry,
    BufferZoneAnalyzeRequest,
    BufferZoneAnalyzeResponse,
    BufferZoneBulkRequest,
    BufferZoneBulkResponse,
    BufferZoneBulkResultEntry,
    BufferZoneMonitorEntry,
    BufferZoneMonitorRequest,
    BufferZoneMonitorResponse,
    BufferZoneViolationEntry,
    BufferZoneViolationsResponse,
    ErrorResponse,
    GeoPointSchema,
    MetadataSchema,
    PaginatedMeta,
    ProtectedAreaTypeEnum,
    ProvenanceInfo,
    RiskLevelEnum,
    ViolationStatusEnum,
    ViolationTypeEnum,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/buffer-zones", tags=["Buffer Zones"])


def _compute_provenance(input_data: Any, output_data: Any) -> str:
    """Compute SHA-256 provenance hash for audit trail."""
    data_str = f"{input_data}{output_data}"
    return hashlib.sha256(data_str.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# POST /buffer-zones/monitor
# ---------------------------------------------------------------------------


@router.post(
    "/monitor",
    response_model=BufferZoneMonitorResponse,
    status_code=status.HTTP_200_OK,
    summary="Monitor buffer zone compliance for a plot",
    description=(
        "Check whether a supply chain plot violates any protected area "
        "buffer zones. Returns per-area compliance results with distance "
        "calculations and risk assessments."
    ),
    responses={
        200: {"description": "Buffer monitoring completed"},
        400: {"model": ErrorResponse, "description": "Invalid request"},
        401: {"model": ErrorResponse, "description": "Authentication required"},
        403: {"model": ErrorResponse, "description": "Insufficient permissions"},
        429: {"model": ErrorResponse, "description": "Rate limit exceeded"},
    },
)
async def monitor_buffer_zones(
    request: Request,
    body: BufferZoneMonitorRequest,
    user: AuthUser = Depends(
        require_permission("eudr-pav:buffer-zone:create")
    ),
    _rate: None = Depends(rate_limit_write),
) -> BufferZoneMonitorResponse:
    """Monitor buffer zone compliance for a plot.

    Args:
        body: Monitoring request with plot location and parameters.
        user: Authenticated user with buffer-zone:create permission.

    Returns:
        BufferZoneMonitorResponse with compliance results.
    """
    start = time.monotonic()

    try:
        engine = get_buffer_zone_monitor()
        result = engine.monitor(
            plot_id=body.plot_id,
            latitude=float(body.plot_center.latitude),
            longitude=float(body.plot_center.longitude),
            plot_boundary=(
                [{"latitude": float(p.latitude), "longitude": float(p.longitude)}
                 for p in body.plot_boundary.coordinates]
                if body.plot_boundary else None
            ),
            buffer_threshold_km=float(body.buffer_threshold_km),
            area_types=[t.value for t in body.area_types] if body.area_types else None,
        )

        results = []
        violations_count = 0
        for r in result.get("results", []):
            entry = BufferZoneMonitorEntry(
                area_id=r.get("area_id", ""),
                area_name=r.get("area_name", ""),
                area_type=ProtectedAreaTypeEnum(r.get("area_type", "other")),
                buffer_zone_km=Decimal(str(r.get("buffer_zone_km", 5))),
                distance_km=Decimal(str(r.get("distance_km", 0))),
                is_within_buffer=r.get("is_within_buffer", False),
                is_compliant=r.get("is_compliant", True),
                violation_type=(
                    ViolationTypeEnum(r["violation_type"])
                    if r.get("violation_type") else None
                ),
                risk_level=RiskLevelEnum(r.get("risk_level", "negligible")),
            )
            results.append(entry)
            if not entry.is_compliant:
                violations_count += 1

        is_overall_compliant = violations_count == 0

        elapsed_ms = Decimal(str((time.monotonic() - start) * 1000))
        provenance_hash = _compute_provenance(
            f"buffer_monitor:{body.plot_id}", str(violations_count),
        )

        logger.info(
            "Buffer monitoring: plot_id=%s areas=%d violations=%d compliant=%s operator=%s",
            body.plot_id,
            len(results),
            violations_count,
            is_overall_compliant,
            user.operator_id or user.user_id,
        )

        return BufferZoneMonitorResponse(
            plot_id=body.plot_id,
            results=results,
            total_areas_checked=len(results),
            violations_detected=violations_count,
            is_compliant=is_overall_compliant,
            provenance=ProvenanceInfo(
                provenance_hash=provenance_hash,
                processing_time_ms=elapsed_ms.quantize(Decimal("0.01")),
            ),
            metadata=MetadataSchema(
                data_sources=["BufferZoneMonitor", "WDPA"],
            ),
        )

    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Buffer zone monitoring failed: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Buffer zone monitoring failed",
        )


# ---------------------------------------------------------------------------
# GET /buffer-zones/violations
# ---------------------------------------------------------------------------


@router.get(
    "/violations",
    response_model=BufferZoneViolationsResponse,
    summary="List buffer zone violations",
    description=(
        "Retrieve a paginated list of buffer zone violations with optional "
        "filters for status, risk level, and plot/area identifiers."
    ),
    responses={
        200: {"description": "Violations listed"},
        401: {"model": ErrorResponse, "description": "Authentication required"},
        403: {"model": ErrorResponse, "description": "Insufficient permissions"},
    },
)
async def list_buffer_violations(
    request: Request,
    plot_id: Optional[str] = Query(None, description="Filter by plot ID"),
    area_id: Optional[str] = Query(None, description="Filter by area ID"),
    risk_level: Optional[RiskLevelEnum] = Query(None, description="Filter by risk level"),
    violation_status: Optional[ViolationStatusEnum] = Query(None, description="Filter by status"),
    pagination: PaginationParams = Depends(get_pagination),
    user: AuthUser = Depends(
        require_permission("eudr-pav:buffer-zone:read")
    ),
    _rate: None = Depends(rate_limit_standard),
) -> BufferZoneViolationsResponse:
    """List buffer zone violations with filters.

    Args:
        plot_id: Optional plot ID filter.
        area_id: Optional area ID filter.
        risk_level: Optional risk level filter.
        violation_status: Optional status filter.
        pagination: Pagination parameters.
        user: Authenticated user with buffer-zone:read permission.

    Returns:
        BufferZoneViolationsResponse with paginated violations.
    """
    start = time.monotonic()

    try:
        engine = get_buffer_zone_monitor()
        result = engine.list_violations(
            plot_id=plot_id,
            area_id=area_id,
            risk_level=risk_level.value if risk_level else None,
            status=violation_status.value if violation_status else None,
            limit=pagination.limit,
            offset=pagination.offset,
        )

        violations = []
        for v in result.get("violations", []):
            violations.append(
                BufferZoneViolationEntry(
                    violation_id=v.get("violation_id", ""),
                    plot_id=v.get("plot_id", ""),
                    area_id=v.get("area_id", ""),
                    area_name=v.get("area_name", ""),
                    distance_km=Decimal(str(v.get("distance_km", 0))),
                    buffer_zone_km=Decimal(str(v.get("buffer_zone_km", 5))),
                    penetration_km=Decimal(str(v.get("penetration_km", 0)))
                    if v.get("penetration_km") is not None else None,
                    violation_type=ViolationTypeEnum(v.get("violation_type", "buffer_breach")),
                    risk_level=RiskLevelEnum(v.get("risk_level", "medium")),
                    status=ViolationStatusEnum(v.get("status", "detected")),
                )
            )

        total = result.get("total", len(violations))

        elapsed_ms = Decimal(str((time.monotonic() - start) * 1000))
        provenance_hash = _compute_provenance("list_buffer_violations", str(total))

        logger.info(
            "Buffer violations listed: total=%d operator=%s",
            total,
            user.operator_id or user.user_id,
        )

        return BufferZoneViolationsResponse(
            violations=violations,
            total_violations=total,
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
                data_sources=["BufferZoneMonitor"],
            ),
        )

    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Buffer violations listing failed: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Buffer zone violations listing failed",
        )


# ---------------------------------------------------------------------------
# POST /buffer-zones/analyze
# ---------------------------------------------------------------------------


@router.post(
    "/analyze",
    response_model=BufferZoneAnalyzeResponse,
    status_code=status.HTTP_200_OK,
    summary="Proximity analysis for a geographic point",
    description=(
        "Perform proximity analysis between a geographic point and all "
        "protected areas within a specified radius. Returns distances, "
        "buffer zone status, and risk levels for each nearby area."
    ),
    responses={
        200: {"description": "Proximity analysis completed"},
        400: {"model": ErrorResponse, "description": "Invalid request"},
        401: {"model": ErrorResponse, "description": "Authentication required"},
        403: {"model": ErrorResponse, "description": "Insufficient permissions"},
        429: {"model": ErrorResponse, "description": "Rate limit exceeded"},
    },
)
async def analyze_buffer_zones(
    request: Request,
    body: BufferZoneAnalyzeRequest,
    user: AuthUser = Depends(
        require_permission("eudr-pav:buffer-zone:create")
    ),
    _rate: None = Depends(rate_limit_write),
) -> BufferZoneAnalyzeResponse:
    """Perform proximity analysis for a geographic point.

    Args:
        body: Analysis request with center point and radius.
        user: Authenticated user with buffer-zone:create permission.

    Returns:
        BufferZoneAnalyzeResponse with proximity results.
    """
    start = time.monotonic()

    try:
        engine = get_buffer_zone_monitor()
        result = engine.analyze_proximity(
            latitude=float(body.center.latitude),
            longitude=float(body.center.longitude),
            radius_km=float(body.radius_km),
            area_types=[t.value for t in body.area_types] if body.area_types else None,
            include_distances=body.include_distances,
        )

        nearby = []
        within_buffer_count = 0
        nearest_km = None

        for a in result.get("nearby_areas", []):
            distance = Decimal(str(a.get("distance_km", 0)))
            entry = BufferZoneAnalyzeEntry(
                area_id=a.get("area_id", ""),
                area_name=a.get("area_name", ""),
                area_type=ProtectedAreaTypeEnum(a.get("area_type", "other")),
                distance_km=distance,
                buffer_zone_km=Decimal(str(a.get("buffer_zone_km", 5))),
                is_within_buffer=a.get("is_within_buffer", False),
                bearing_degrees=Decimal(str(a.get("bearing_degrees", 0)))
                if a.get("bearing_degrees") is not None else None,
                risk_level=RiskLevelEnum(a.get("risk_level", "negligible")),
            )
            nearby.append(entry)
            if entry.is_within_buffer:
                within_buffer_count += 1
            if nearest_km is None or distance < nearest_km:
                nearest_km = distance

        elapsed_ms = Decimal(str((time.monotonic() - start) * 1000))
        provenance_hash = _compute_provenance(
            f"analyze_proximity:{body.center.latitude},{body.center.longitude}",
            str(len(nearby)),
        )

        logger.info(
            "Proximity analysis: lat=%s lon=%s areas=%d within_buffer=%d operator=%s",
            body.center.latitude,
            body.center.longitude,
            len(nearby),
            within_buffer_count,
            user.operator_id or user.user_id,
        )

        return BufferZoneAnalyzeResponse(
            center=body.center,
            nearby_areas=nearby,
            total_areas_found=len(nearby),
            areas_within_buffer=within_buffer_count,
            nearest_area_km=nearest_km,
            provenance=ProvenanceInfo(
                provenance_hash=provenance_hash,
                processing_time_ms=elapsed_ms.quantize(Decimal("0.01")),
            ),
            metadata=MetadataSchema(
                data_sources=["BufferZoneMonitor", "WDPA", "OECM"],
            ),
        )

    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Proximity analysis failed: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Proximity analysis failed",
        )


# ---------------------------------------------------------------------------
# POST /buffer-zones/bulk
# ---------------------------------------------------------------------------


@router.post(
    "/bulk",
    response_model=BufferZoneBulkResponse,
    status_code=status.HTTP_200_OK,
    summary="Bulk buffer zone monitoring",
    description=(
        "Perform buffer zone compliance monitoring across multiple "
        "supply chain plots in a single batch request (max 500)."
    ),
    responses={
        200: {"description": "Bulk monitoring completed"},
        400: {"model": ErrorResponse, "description": "Invalid request"},
        401: {"model": ErrorResponse, "description": "Authentication required"},
        403: {"model": ErrorResponse, "description": "Insufficient permissions"},
        429: {"model": ErrorResponse, "description": "Rate limit exceeded"},
    },
)
async def bulk_monitor_buffer_zones(
    request: Request,
    body: BufferZoneBulkRequest,
    user: AuthUser = Depends(
        require_permission("eudr-pav:buffer-zone:create")
    ),
    _rate: None = Depends(rate_limit_heavy),
) -> BufferZoneBulkResponse:
    """Perform bulk buffer zone monitoring.

    Args:
        body: Bulk monitoring request with multiple plots.
        user: Authenticated user with buffer-zone:create permission.

    Returns:
        BufferZoneBulkResponse with per-plot compliance results.
    """
    start = time.monotonic()

    try:
        engine = get_buffer_zone_monitor()

        plots_data = []
        for p in body.plots:
            plots_data.append({
                "plot_id": p.plot_id,
                "latitude": float(p.plot_center.latitude),
                "longitude": float(p.plot_center.longitude),
                "buffer_threshold_km": float(p.buffer_threshold_km),
            })

        result = engine.bulk_monitor(plots=plots_data)

        results = []
        compliant = 0
        non_compliant = 0
        total_violations = 0
        failed = 0

        for r in result.get("results", []):
            entry = BufferZoneBulkResultEntry(
                plot_id=r.get("plot_id", ""),
                is_compliant=r.get("is_compliant", True),
                violations_detected=r.get("violations_detected", 0),
                nearest_area_km=Decimal(str(r.get("nearest_area_km", 0)))
                if r.get("nearest_area_km") is not None else None,
                error=r.get("error"),
            )
            results.append(entry)

            if entry.error:
                failed += 1
            elif entry.is_compliant:
                compliant += 1
            else:
                non_compliant += 1
                total_violations += entry.violations_detected

        elapsed_ms = Decimal(str((time.monotonic() - start) * 1000))
        provenance_hash = _compute_provenance(
            f"bulk_buffer:{len(body.plots)}", str(total_violations),
        )

        logger.info(
            "Bulk buffer monitoring: plots=%d compliant=%d non_compliant=%d violations=%d operator=%s",
            len(body.plots),
            compliant,
            non_compliant,
            total_violations,
            user.operator_id or user.user_id,
        )

        return BufferZoneBulkResponse(
            results=results,
            total_plots_processed=len(body.plots),
            compliant_count=compliant,
            non_compliant_count=non_compliant,
            total_violations=total_violations,
            failed_count=failed,
            provenance=ProvenanceInfo(
                provenance_hash=provenance_hash,
                processing_time_ms=elapsed_ms.quantize(Decimal("0.01")),
            ),
            metadata=MetadataSchema(
                data_sources=["BufferZoneMonitor", "WDPA"],
            ),
        )

    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Bulk buffer monitoring failed: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Bulk buffer zone monitoring failed",
        )
