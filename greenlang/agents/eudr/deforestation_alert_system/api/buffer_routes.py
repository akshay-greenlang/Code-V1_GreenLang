# -*- coding: utf-8 -*-
"""
Spatial Buffer Monitoring Routes - AGENT-EUDR-020 Deforestation Alert System API

Endpoints for spatial buffer zone management supporting circular, polygon, and
adaptive buffer geometries with configurable 1-50 km radii at 64-point resolution
for proximity detection to EUDR supply chain plots.

Endpoints:
    POST /buffer/create           - Create buffer zone around a plot
    PUT  /buffer/{buffer_id}      - Update existing buffer zone
    POST /buffer/check            - Check point against active buffers
    GET  /buffer/violations       - List buffer zone violations
    GET  /buffer/zones            - List active buffer zones

Buffer radius: 1-50 km (default 10 km)
Resolution: 64 points per buffer geometry circle

Author: GreenLang Platform Team
Date: March 2026
PRD: AGENT-EUDR-020, SpatialBufferMonitor Engine
"""

from __future__ import annotations

import hashlib
import logging
import time
from decimal import Decimal
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, Request, status

from greenlang.agents.eudr.deforestation_alert_system.api.dependencies import (
    AuthUser,
    PaginationParams,
    get_buffer_monitor,
    get_pagination,
    rate_limit_standard,
    rate_limit_write,
    require_permission,
)
from greenlang.agents.eudr.deforestation_alert_system.api.schemas import (
    AlertSeverityEnum,
    BufferCheckRequest,
    BufferCheckResponse,
    BufferCheckResult,
    BufferCreateRequest,
    BufferCreateResponse,
    BufferTypeEnum,
    BufferUpdateRequest,
    BufferViolationEntry,
    BufferViolationsResponse,
    BufferZoneEntry,
    BufferZonesResponse,
    ErrorResponse,
    EUDRCommodityEnum,
    MetadataSchema,
    PaginatedMeta,
    ProvenanceInfo,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/buffer", tags=["Spatial Buffer Monitoring"])


def _compute_provenance(input_data: Any, output_data: Any) -> str:
    """Compute SHA-256 provenance hash for audit trail."""
    data_str = f"{input_data}{output_data}"
    return hashlib.sha256(data_str.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# POST /buffer/create
# ---------------------------------------------------------------------------


@router.post(
    "/create",
    response_model=BufferCreateResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create buffer zone around a supply chain plot",
    description=(
        "Create a spatial buffer zone around a supply chain plot for deforestation "
        "monitoring. Supports circular (default), polygon, and adaptive buffer "
        "geometries with configurable radius (1-50 km) and resolution (4-256 points)."
    ),
    responses={
        201: {"description": "Buffer zone created"},
        400: {"model": ErrorResponse, "description": "Invalid request"},
        401: {"model": ErrorResponse, "description": "Authentication required"},
        403: {"model": ErrorResponse, "description": "Insufficient permissions"},
        409: {"model": ErrorResponse, "description": "Buffer already exists for plot"},
    },
)
async def create_buffer(
    request: Request,
    body: BufferCreateRequest,
    user: AuthUser = Depends(
        require_permission("eudr-deforestation-alert:buffer:create")
    ),
    _rate: None = Depends(rate_limit_write),
) -> BufferCreateResponse:
    """Create a spatial buffer zone for plot monitoring.

    Args:
        body: Buffer creation request.
        user: Authenticated user with buffer:create permission.

    Returns:
        BufferCreateResponse with created buffer zone.
    """
    start = time.monotonic()

    try:
        engine = get_buffer_monitor()
        result = engine.create_buffer(
            plot_id=body.plot_id,
            latitude=float(body.center.latitude),
            longitude=float(body.center.longitude),
            radius_km=float(body.radius_km),
            buffer_type=body.buffer_type.value,
            polygon=[
                {"latitude": float(p.latitude), "longitude": float(p.longitude)}
                for p in body.polygon.coordinates
            ] if body.polygon else None,
            resolution_points=body.resolution_points,
            name=body.name,
            commodities=[c.value for c in body.commodities] if body.commodities else None,
            created_by=user.user_id,
        )

        buffer_data = result.get("buffer", {})
        buffer_entry = BufferZoneEntry(
            buffer_id=buffer_data.get("buffer_id", ""),
            plot_id=body.plot_id,
            center_latitude=Decimal(str(buffer_data.get("center_latitude", body.center.latitude))),
            center_longitude=Decimal(str(buffer_data.get("center_longitude", body.center.longitude))),
            radius_km=Decimal(str(buffer_data.get("radius_km", body.radius_km))),
            buffer_type=body.buffer_type,
            area_km2=Decimal(str(buffer_data.get("area_km2", 0)))
            if buffer_data.get("area_km2") else None,
            active=True,
            name=body.name,
            commodities=[EUDRCommodityEnum(c) for c in body.commodities] if body.commodities else [],
        )

        elapsed_ms = Decimal(str((time.monotonic() - start) * 1000))
        provenance_hash = _compute_provenance(
            f"create_buffer:{body.plot_id}:{body.radius_km}",
            buffer_entry.buffer_id,
        )

        logger.info(
            "Buffer created: buffer_id=%s plot_id=%s radius_km=%s operator=%s",
            buffer_entry.buffer_id,
            body.plot_id,
            body.radius_km,
            user.operator_id or user.user_id,
        )

        return BufferCreateResponse(
            buffer=buffer_entry,
            provenance=ProvenanceInfo(
                provenance_hash=provenance_hash,
                processing_time_ms=elapsed_ms.quantize(Decimal("0.01")),
            ),
            metadata=MetadataSchema(
                data_sources=["SpatialBufferMonitor"],
            ),
        )

    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Buffer creation failed: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Buffer zone creation failed",
        )


# ---------------------------------------------------------------------------
# PUT /buffer/{buffer_id}
# ---------------------------------------------------------------------------


@router.put(
    "/{buffer_id}",
    response_model=BufferCreateResponse,
    summary="Update existing buffer zone",
    description=(
        "Update properties of an existing buffer zone including radius, "
        "active status, name, commodities, and resolution."
    ),
    responses={
        200: {"description": "Buffer updated"},
        400: {"model": ErrorResponse, "description": "Invalid request"},
        401: {"model": ErrorResponse, "description": "Authentication required"},
        403: {"model": ErrorResponse, "description": "Insufficient permissions"},
        404: {"model": ErrorResponse, "description": "Buffer not found"},
    },
)
async def update_buffer(
    buffer_id: str,
    request: Request,
    body: BufferUpdateRequest,
    user: AuthUser = Depends(
        require_permission("eudr-deforestation-alert:buffer:update")
    ),
    _rate: None = Depends(rate_limit_write),
) -> BufferCreateResponse:
    """Update an existing buffer zone.

    Args:
        buffer_id: Buffer zone identifier to update.
        body: Update request with changed fields.
        user: Authenticated user with buffer:update permission.

    Returns:
        BufferCreateResponse with updated buffer zone.
    """
    start = time.monotonic()

    try:
        engine = get_buffer_monitor()
        result = engine.update_buffer(
            buffer_id=buffer_id,
            radius_km=float(body.radius_km) if body.radius_km else None,
            active=body.active,
            name=body.name,
            commodities=[c.value for c in body.commodities] if body.commodities else None,
            resolution_points=body.resolution_points,
            updated_by=user.user_id,
        )

        if result is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Buffer zone not found: {buffer_id}",
            )

        buffer_data = result.get("buffer", {})
        buffer_entry = BufferZoneEntry(
            buffer_id=buffer_id,
            plot_id=buffer_data.get("plot_id", ""),
            center_latitude=Decimal(str(buffer_data.get("center_latitude", 0))),
            center_longitude=Decimal(str(buffer_data.get("center_longitude", 0))),
            radius_km=Decimal(str(buffer_data.get("radius_km", 10))),
            buffer_type=BufferTypeEnum(buffer_data.get("buffer_type", "circular")),
            area_km2=Decimal(str(buffer_data.get("area_km2", 0)))
            if buffer_data.get("area_km2") else None,
            active=buffer_data.get("active", True),
            name=buffer_data.get("name"),
            commodities=[
                EUDRCommodityEnum(c) for c in buffer_data.get("commodities", [])
            ],
        )

        elapsed_ms = Decimal(str((time.monotonic() - start) * 1000))
        provenance_hash = _compute_provenance(
            f"update_buffer:{buffer_id}", str(buffer_entry.active)
        )

        logger.info(
            "Buffer updated: buffer_id=%s operator=%s",
            buffer_id,
            user.operator_id or user.user_id,
        )

        return BufferCreateResponse(
            buffer=buffer_entry,
            provenance=ProvenanceInfo(
                provenance_hash=provenance_hash,
                processing_time_ms=elapsed_ms.quantize(Decimal("0.01")),
            ),
            metadata=MetadataSchema(
                data_sources=["SpatialBufferMonitor"],
            ),
        )

    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Buffer update failed: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Buffer zone update failed",
        )


# ---------------------------------------------------------------------------
# POST /buffer/check
# ---------------------------------------------------------------------------


@router.post(
    "/check",
    response_model=BufferCheckResponse,
    status_code=status.HTTP_200_OK,
    summary="Check point against active buffer zones",
    description=(
        "Check whether a geographic point falls within any active buffer "
        "zones. Returns distance to buffer centers and affected commodities."
    ),
    responses={
        200: {"description": "Buffer check completed"},
        400: {"model": ErrorResponse, "description": "Invalid coordinates"},
        401: {"model": ErrorResponse, "description": "Authentication required"},
        403: {"model": ErrorResponse, "description": "Insufficient permissions"},
    },
)
async def check_buffer(
    request: Request,
    body: BufferCheckRequest,
    user: AuthUser = Depends(
        require_permission("eudr-deforestation-alert:buffer:read")
    ),
    _rate: None = Depends(rate_limit_standard),
) -> BufferCheckResponse:
    """Check if a point is within any active buffer zones.

    Args:
        body: Check request with coordinates.
        user: Authenticated user with buffer:read permission.

    Returns:
        BufferCheckResponse with affected buffers.
    """
    start = time.monotonic()

    try:
        engine = get_buffer_monitor()
        result = engine.check_point(
            latitude=float(body.latitude),
            longitude=float(body.longitude),
            detection_id=body.detection_id,
            include_distance=body.include_distance,
        )

        buffers_affected = []
        for buf in result.get("buffers_affected", []):
            buffers_affected.append(
                BufferCheckResult(
                    buffer_id=buf.get("buffer_id", ""),
                    plot_id=buf.get("plot_id", ""),
                    is_within_buffer=buf.get("is_within_buffer", False),
                    distance_km=Decimal(str(buf.get("distance_km", 0)))
                    if buf.get("distance_km") is not None else None,
                    buffer_radius_km=Decimal(str(buf.get("buffer_radius_km", 10))),
                    commodities=[
                        EUDRCommodityEnum(c) for c in buf.get("commodities", [])
                    ],
                )
            )

        is_within_any = any(b.is_within_buffer for b in buffers_affected)

        elapsed_ms = Decimal(str((time.monotonic() - start) * 1000))
        provenance_hash = _compute_provenance(
            f"check_buffer:{body.latitude},{body.longitude}",
            str(is_within_any),
        )

        logger.info(
            "Buffer check: lat=%s lon=%s within_any=%s affected=%d operator=%s",
            body.latitude,
            body.longitude,
            is_within_any,
            len(buffers_affected),
            user.operator_id or user.user_id,
        )

        return BufferCheckResponse(
            point_latitude=body.latitude,
            point_longitude=body.longitude,
            is_within_any_buffer=is_within_any,
            buffers_affected=buffers_affected,
            total_buffers_checked=result.get("total_buffers_checked", 0),
            provenance=ProvenanceInfo(
                provenance_hash=provenance_hash,
                processing_time_ms=elapsed_ms.quantize(Decimal("0.01")),
            ),
            metadata=MetadataSchema(
                data_sources=["SpatialBufferMonitor"],
            ),
        )

    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Buffer check failed: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Buffer zone check failed",
        )


# ---------------------------------------------------------------------------
# GET /buffer/violations
# ---------------------------------------------------------------------------


@router.get(
    "/violations",
    response_model=BufferViolationsResponse,
    summary="List buffer zone violations",
    description=(
        "List deforestation events that were detected within active buffer "
        "zones, representing proximity violations to supply chain plots."
    ),
    responses={
        200: {"description": "Violations listed"},
        401: {"model": ErrorResponse, "description": "Authentication required"},
        403: {"model": ErrorResponse, "description": "Insufficient permissions"},
    },
)
async def list_violations(
    request: Request,
    buffer_id: Optional[str] = Query(None, description="Filter by buffer ID"),
    severity: Optional[AlertSeverityEnum] = Query(None, description="Filter by severity"),
    pagination: PaginationParams = Depends(get_pagination),
    user: AuthUser = Depends(
        require_permission("eudr-deforestation-alert:buffer:read")
    ),
    _rate: None = Depends(rate_limit_standard),
) -> BufferViolationsResponse:
    """List buffer zone violations.

    Args:
        buffer_id: Optional buffer filter.
        severity: Optional severity filter.
        pagination: Pagination parameters.
        user: Authenticated user with buffer:read permission.

    Returns:
        BufferViolationsResponse with violation list.
    """
    start = time.monotonic()

    try:
        engine = get_buffer_monitor()
        result = engine.list_violations(
            buffer_id=buffer_id,
            severity=severity.value if severity else None,
            limit=pagination.limit,
            offset=pagination.offset,
        )

        violations = []
        for v in result.get("violations", []):
            violations.append(
                BufferViolationEntry(
                    violation_id=v.get("violation_id", ""),
                    buffer_id=v.get("buffer_id", ""),
                    plot_id=v.get("plot_id", ""),
                    alert_id=v.get("alert_id"),
                    detection_id=v.get("detection_id"),
                    latitude=Decimal(str(v.get("latitude", 0))),
                    longitude=Decimal(str(v.get("longitude", 0))),
                    distance_km=Decimal(str(v.get("distance_km", 0))),
                    area_ha=Decimal(str(v.get("area_ha", 0)))
                    if v.get("area_ha") is not None else None,
                    severity=AlertSeverityEnum(v.get("severity"))
                    if v.get("severity") else None,
                    detected_at=v.get("detected_at"),
                )
            )

        total = result.get("total_violations", len(violations))
        elapsed_ms = Decimal(str((time.monotonic() - start) * 1000))
        provenance_hash = _compute_provenance(
            f"violations:{buffer_id}:{severity}", str(total)
        )

        logger.info(
            "Buffer violations listed: total=%d operator=%s",
            total,
            user.operator_id or user.user_id,
        )

        return BufferViolationsResponse(
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
                data_sources=["SpatialBufferMonitor"],
            ),
        )

    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Buffer violations listing failed: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Buffer violations listing failed",
        )


# ---------------------------------------------------------------------------
# GET /buffer/zones
# ---------------------------------------------------------------------------


@router.get(
    "/zones",
    response_model=BufferZonesResponse,
    summary="List active buffer zones",
    description=(
        "List all active buffer zones with their configuration, coverage "
        "area, and violation counts."
    ),
    responses={
        200: {"description": "Buffer zones listed"},
        401: {"model": ErrorResponse, "description": "Authentication required"},
        403: {"model": ErrorResponse, "description": "Insufficient permissions"},
    },
)
async def list_zones(
    request: Request,
    active_only: bool = Query(True, description="Show only active zones"),
    pagination: PaginationParams = Depends(get_pagination),
    user: AuthUser = Depends(
        require_permission("eudr-deforestation-alert:buffer:read")
    ),
    _rate: None = Depends(rate_limit_standard),
) -> BufferZonesResponse:
    """List buffer zones.

    Args:
        active_only: Whether to show only active zones.
        pagination: Pagination parameters.
        user: Authenticated user with buffer:read permission.

    Returns:
        BufferZonesResponse with zone list.
    """
    start = time.monotonic()

    try:
        engine = get_buffer_monitor()
        result = engine.list_zones(
            active_only=active_only,
            limit=pagination.limit,
            offset=pagination.offset,
        )

        zones = []
        for z in result.get("zones", []):
            zones.append(
                BufferZoneEntry(
                    buffer_id=z.get("buffer_id", ""),
                    plot_id=z.get("plot_id", ""),
                    center_latitude=Decimal(str(z.get("center_latitude", 0))),
                    center_longitude=Decimal(str(z.get("center_longitude", 0))),
                    radius_km=Decimal(str(z.get("radius_km", 10))),
                    buffer_type=BufferTypeEnum(z.get("buffer_type", "circular")),
                    area_km2=Decimal(str(z.get("area_km2", 0)))
                    if z.get("area_km2") else None,
                    active=z.get("active", True),
                    name=z.get("name"),
                    commodities=[
                        EUDRCommodityEnum(c) for c in z.get("commodities", [])
                    ],
                    violation_count=z.get("violation_count", 0),
                    created_at=z.get("created_at"),
                )
            )

        total = result.get("total_zones", len(zones))
        active_count = result.get("active_zones", sum(1 for z in zones if z.active))
        total_area = Decimal(str(result.get("total_area_km2", 0))) if result.get("total_area_km2") else None

        elapsed_ms = Decimal(str((time.monotonic() - start) * 1000))
        provenance_hash = _compute_provenance(
            f"zones:{active_only}", str(total)
        )

        logger.info(
            "Buffer zones listed: total=%d active=%d operator=%s",
            total,
            active_count,
            user.operator_id or user.user_id,
        )

        return BufferZonesResponse(
            zones=zones,
            total_zones=total,
            active_zones=active_count,
            total_area_km2=total_area,
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
                data_sources=["SpatialBufferMonitor"],
            ),
        )

    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Buffer zones listing failed: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Buffer zones listing failed",
        )
