# -*- coding: utf-8 -*-
"""
Protected Area Routes - AGENT-EUDR-022 Protected Area Validator API

Endpoints for CRUD operations and spatial search on protected area records
sourced from WDPA, OECM, UNESCO, Ramsar, and national registries. Supports
IUCN management categories Ia-VI and special designations.

Endpoints:
    POST /protected-areas                - Register a new protected area
    GET  /protected-areas                - List protected areas with filters
    GET  /protected-areas/{area_id}      - Get protected area details
    PUT  /protected-areas/{area_id}      - Update protected area
    DELETE /protected-areas/{area_id}    - Archive (soft-delete) protected area
    POST /protected-areas/search         - Advanced spatial search

Auth: eudr-pav:protected-area:{create|read|update|delete}

Author: GreenLang Platform Team
Date: March 2026
PRD: AGENT-EUDR-022, ProtectedAreaEngine
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
    get_pagination,
    get_protected_area_engine,
    rate_limit_heavy,
    rate_limit_standard,
    rate_limit_write,
    require_permission,
)
from greenlang.agents.eudr.protected_area_validator.api.schemas import (
    DataSourceEnum,
    DesignationStatusEnum,
    ErrorResponse,
    MetadataSchema,
    PaginatedMeta,
    ProtectedAreaCreateRequest,
    ProtectedAreaEntry,
    ProtectedAreaListResponse,
    ProtectedAreaResponse,
    ProtectedAreaSearchRequest,
    ProtectedAreaSearchResponse,
    ProtectedAreaTypeEnum,
    ProtectedAreaUpdateRequest,
    ProvenanceInfo,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/protected-areas", tags=["Protected Areas"])


def _compute_provenance(input_data: Any, output_data: Any) -> str:
    """Compute SHA-256 provenance hash for audit trail."""
    data_str = f"{input_data}{output_data}"
    return hashlib.sha256(data_str.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# POST /protected-areas
# ---------------------------------------------------------------------------


@router.post(
    "",
    response_model=ProtectedAreaResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Register a new protected area",
    description=(
        "Register a new protected area in the system with boundary geometry, "
        "IUCN category, designation status, buffer zone configuration, and "
        "management metadata. Sources include WDPA, OECM, UNESCO, Ramsar."
    ),
    responses={
        201: {"description": "Protected area registered"},
        400: {"model": ErrorResponse, "description": "Invalid request"},
        401: {"model": ErrorResponse, "description": "Authentication required"},
        403: {"model": ErrorResponse, "description": "Insufficient permissions"},
        409: {"model": ErrorResponse, "description": "Area already exists"},
        429: {"model": ErrorResponse, "description": "Rate limit exceeded"},
    },
)
async def register_protected_area(
    request: Request,
    body: ProtectedAreaCreateRequest,
    user: AuthUser = Depends(
        require_permission("eudr-pav:protected-area:create")
    ),
    _rate: None = Depends(rate_limit_write),
) -> ProtectedAreaResponse:
    """Register a new protected area.

    Args:
        body: Protected area creation request with boundary and metadata.
        user: Authenticated user with protected-area:create permission.

    Returns:
        ProtectedAreaResponse with the created area record.
    """
    start = time.monotonic()

    try:
        engine = get_protected_area_engine()
        result = engine.register_area(
            name=body.name,
            area_type=body.area_type.value,
            country_code=body.country_code.upper(),
            wdpa_id=body.wdpa_id,
            designation_status=body.designation_status.value,
            designation_date=body.designation_date,
            total_area_km2=float(body.total_area_km2),
            marine_area_km2=float(body.marine_area_km2) if body.marine_area_km2 else None,
            boundary=[
                {"latitude": float(p.latitude), "longitude": float(p.longitude)}
                for p in body.boundary.coordinates
            ],
            centroid=(
                {"latitude": float(body.centroid.latitude), "longitude": float(body.centroid.longitude)}
                if body.centroid else None
            ),
            buffer_zone_km=float(body.buffer_zone_km),
            governance_type=body.governance_type,
            management_authority=body.management_authority,
            data_source=body.data_source.value,
            iucn_category=body.iucn_category,
            notes=body.notes,
            created_by=user.user_id,
        )

        if result is None:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to register protected area",
            )

        area_entry = ProtectedAreaEntry(
            area_id=result.get("area_id", ""),
            name=body.name,
            area_type=body.area_type,
            country_code=body.country_code.upper(),
            wdpa_id=body.wdpa_id,
            designation_status=body.designation_status,
            designation_date=body.designation_date,
            total_area_km2=body.total_area_km2,
            buffer_zone_km=body.buffer_zone_km,
            centroid_latitude=body.centroid.latitude if body.centroid else None,
            centroid_longitude=body.centroid.longitude if body.centroid else None,
            governance_type=body.governance_type,
            management_authority=body.management_authority,
            data_source=body.data_source,
            iucn_category=body.iucn_category,
        )

        elapsed_ms = Decimal(str((time.monotonic() - start) * 1000))
        provenance_hash = _compute_provenance(
            f"register_area:{body.name}:{body.country_code}",
            str(area_entry.area_id),
        )

        logger.info(
            "Protected area registered: area_id=%s name=%s country=%s operator=%s",
            area_entry.area_id,
            body.name,
            body.country_code,
            user.operator_id or user.user_id,
        )

        return ProtectedAreaResponse(
            area=area_entry,
            provenance=ProvenanceInfo(
                provenance_hash=provenance_hash,
                processing_time_ms=elapsed_ms.quantize(Decimal("0.01")),
            ),
            metadata=MetadataSchema(
                data_sources=[body.data_source.value, "ProtectedAreaEngine"],
            ),
        )

    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Protected area registration failed: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Protected area registration failed",
        )


# ---------------------------------------------------------------------------
# GET /protected-areas
# ---------------------------------------------------------------------------


@router.get(
    "",
    response_model=ProtectedAreaListResponse,
    summary="List protected areas with filters",
    description=(
        "Retrieve a paginated list of protected areas with optional filters "
        "for country code, area type, designation status, data source, and "
        "minimum/maximum area size."
    ),
    responses={
        200: {"description": "Protected areas listed"},
        401: {"model": ErrorResponse, "description": "Authentication required"},
        403: {"model": ErrorResponse, "description": "Insufficient permissions"},
    },
)
async def list_protected_areas(
    request: Request,
    country_code: Optional[str] = Query(None, description="Filter by country code"),
    area_type: Optional[ProtectedAreaTypeEnum] = Query(None, description="Filter by area type"),
    designation_status: Optional[DesignationStatusEnum] = Query(None, description="Filter by designation status"),
    data_source: Optional[DataSourceEnum] = Query(None, description="Filter by data source"),
    min_area_km2: Optional[float] = Query(None, ge=0, description="Minimum area in km2"),
    max_area_km2: Optional[float] = Query(None, ge=0, description="Maximum area in km2"),
    pagination: PaginationParams = Depends(get_pagination),
    user: AuthUser = Depends(
        require_permission("eudr-pav:protected-area:read")
    ),
    _rate: None = Depends(rate_limit_standard),
) -> ProtectedAreaListResponse:
    """List protected areas with optional filters.

    Args:
        country_code: Optional country code filter.
        area_type: Optional area type filter.
        designation_status: Optional designation status filter.
        data_source: Optional data source filter.
        min_area_km2: Optional minimum area filter.
        max_area_km2: Optional maximum area filter.
        pagination: Pagination parameters.
        user: Authenticated user with protected-area:read permission.

    Returns:
        ProtectedAreaListResponse with paginated results.
    """
    start = time.monotonic()

    try:
        engine = get_protected_area_engine()
        result = engine.list_areas(
            country_code=country_code.upper() if country_code else None,
            area_type=area_type.value if area_type else None,
            designation_status=designation_status.value if designation_status else None,
            data_source=data_source.value if data_source else None,
            min_area_km2=min_area_km2,
            max_area_km2=max_area_km2,
            limit=pagination.limit,
            offset=pagination.offset,
        )

        areas = []
        for a in result.get("areas", []):
            areas.append(
                ProtectedAreaEntry(
                    area_id=a.get("area_id", ""),
                    name=a.get("name", ""),
                    area_type=ProtectedAreaTypeEnum(a.get("area_type", "other")),
                    country_code=a.get("country_code", ""),
                    wdpa_id=a.get("wdpa_id"),
                    designation_status=DesignationStatusEnum(a.get("designation_status", "unknown")),
                    designation_date=a.get("designation_date"),
                    total_area_km2=Decimal(str(a.get("total_area_km2", 0))),
                    buffer_zone_km=Decimal(str(a.get("buffer_zone_km", 5))),
                    centroid_latitude=Decimal(str(a["centroid_latitude"])) if a.get("centroid_latitude") else None,
                    centroid_longitude=Decimal(str(a["centroid_longitude"])) if a.get("centroid_longitude") else None,
                    governance_type=a.get("governance_type"),
                    management_authority=a.get("management_authority"),
                    data_source=DataSourceEnum(a.get("data_source", "wdpa")),
                    iucn_category=a.get("iucn_category"),
                    is_active=a.get("is_active", True),
                )
            )

        total = result.get("total", len(areas))

        elapsed_ms = Decimal(str((time.monotonic() - start) * 1000))
        provenance_hash = _compute_provenance(
            f"list_areas:{country_code}:{area_type}",
            str(total),
        )

        logger.info(
            "Protected areas listed: total=%d limit=%d offset=%d operator=%s",
            total,
            pagination.limit,
            pagination.offset,
            user.operator_id or user.user_id,
        )

        return ProtectedAreaListResponse(
            areas=areas,
            total_areas=total,
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
                data_sources=["ProtectedAreaEngine", "WDPA"],
            ),
        )

    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Protected area listing failed: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Protected area listing failed",
        )


# ---------------------------------------------------------------------------
# GET /protected-areas/{area_id}
# ---------------------------------------------------------------------------


@router.get(
    "/{area_id}",
    response_model=ProtectedAreaResponse,
    summary="Get protected area details",
    description=(
        "Retrieve full details of a specific protected area by its identifier, "
        "including boundary geometry, designation info, and management metadata."
    ),
    responses={
        200: {"description": "Protected area details retrieved"},
        401: {"model": ErrorResponse, "description": "Authentication required"},
        403: {"model": ErrorResponse, "description": "Insufficient permissions"},
        404: {"model": ErrorResponse, "description": "Area not found"},
    },
)
async def get_protected_area(
    area_id: str,
    request: Request,
    user: AuthUser = Depends(
        require_permission("eudr-pav:protected-area:read")
    ),
    _rate: None = Depends(rate_limit_standard),
) -> ProtectedAreaResponse:
    """Get protected area details by ID.

    Args:
        area_id: Protected area identifier.
        user: Authenticated user with protected-area:read permission.

    Returns:
        ProtectedAreaResponse with area details.
    """
    start = time.monotonic()

    try:
        engine = get_protected_area_engine()
        result = engine.get_area(area_id=area_id)

        if result is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Protected area not found: {area_id}",
            )

        area_entry = ProtectedAreaEntry(
            area_id=result.get("area_id", area_id),
            name=result.get("name", ""),
            area_type=ProtectedAreaTypeEnum(result.get("area_type", "other")),
            country_code=result.get("country_code", ""),
            wdpa_id=result.get("wdpa_id"),
            designation_status=DesignationStatusEnum(result.get("designation_status", "unknown")),
            designation_date=result.get("designation_date"),
            total_area_km2=Decimal(str(result.get("total_area_km2", 0))),
            buffer_zone_km=Decimal(str(result.get("buffer_zone_km", 5))),
            centroid_latitude=Decimal(str(result["centroid_latitude"])) if result.get("centroid_latitude") else None,
            centroid_longitude=Decimal(str(result["centroid_longitude"])) if result.get("centroid_longitude") else None,
            governance_type=result.get("governance_type"),
            management_authority=result.get("management_authority"),
            data_source=DataSourceEnum(result.get("data_source", "wdpa")),
            iucn_category=result.get("iucn_category"),
            is_active=result.get("is_active", True),
        )

        elapsed_ms = Decimal(str((time.monotonic() - start) * 1000))
        provenance_hash = _compute_provenance(f"get_area:{area_id}", area_entry.name)

        logger.info(
            "Protected area retrieved: area_id=%s name=%s operator=%s",
            area_id,
            area_entry.name,
            user.operator_id or user.user_id,
        )

        return ProtectedAreaResponse(
            area=area_entry,
            provenance=ProvenanceInfo(
                provenance_hash=provenance_hash,
                processing_time_ms=elapsed_ms.quantize(Decimal("0.01")),
            ),
            metadata=MetadataSchema(
                data_sources=["ProtectedAreaEngine", area_entry.data_source.value],
            ),
        )

    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Protected area retrieval failed: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Protected area retrieval failed",
        )


# ---------------------------------------------------------------------------
# PUT /protected-areas/{area_id}
# ---------------------------------------------------------------------------


@router.put(
    "/{area_id}",
    response_model=ProtectedAreaResponse,
    summary="Update a protected area",
    description=(
        "Update an existing protected area's metadata, boundary, buffer zone "
        "configuration, or designation status. Partial updates are supported."
    ),
    responses={
        200: {"description": "Protected area updated"},
        400: {"model": ErrorResponse, "description": "Invalid request"},
        401: {"model": ErrorResponse, "description": "Authentication required"},
        403: {"model": ErrorResponse, "description": "Insufficient permissions"},
        404: {"model": ErrorResponse, "description": "Area not found"},
    },
)
async def update_protected_area(
    area_id: str,
    request: Request,
    body: ProtectedAreaUpdateRequest,
    user: AuthUser = Depends(
        require_permission("eudr-pav:protected-area:update")
    ),
    _rate: None = Depends(rate_limit_write),
) -> ProtectedAreaResponse:
    """Update a protected area.

    Args:
        area_id: Protected area identifier.
        body: Update request with partial field updates.
        user: Authenticated user with protected-area:update permission.

    Returns:
        ProtectedAreaResponse with updated area details.
    """
    start = time.monotonic()

    try:
        engine = get_protected_area_engine()

        update_data = {}
        if body.name is not None:
            update_data["name"] = body.name
        if body.designation_status is not None:
            update_data["designation_status"] = body.designation_status.value
        if body.total_area_km2 is not None:
            update_data["total_area_km2"] = float(body.total_area_km2)
        if body.buffer_zone_km is not None:
            update_data["buffer_zone_km"] = float(body.buffer_zone_km)
        if body.boundary is not None:
            update_data["boundary"] = [
                {"latitude": float(p.latitude), "longitude": float(p.longitude)}
                for p in body.boundary.coordinates
            ]
        if body.governance_type is not None:
            update_data["governance_type"] = body.governance_type
        if body.management_authority is not None:
            update_data["management_authority"] = body.management_authority
        if body.notes is not None:
            update_data["notes"] = body.notes

        result = engine.update_area(
            area_id=area_id,
            updated_by=user.user_id,
            **update_data,
        )

        if result is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Protected area not found: {area_id}",
            )

        area_entry = ProtectedAreaEntry(
            area_id=result.get("area_id", area_id),
            name=result.get("name", ""),
            area_type=ProtectedAreaTypeEnum(result.get("area_type", "other")),
            country_code=result.get("country_code", ""),
            wdpa_id=result.get("wdpa_id"),
            designation_status=DesignationStatusEnum(result.get("designation_status", "unknown")),
            designation_date=result.get("designation_date"),
            total_area_km2=Decimal(str(result.get("total_area_km2", 0))),
            buffer_zone_km=Decimal(str(result.get("buffer_zone_km", 5))),
            centroid_latitude=Decimal(str(result["centroid_latitude"])) if result.get("centroid_latitude") else None,
            centroid_longitude=Decimal(str(result["centroid_longitude"])) if result.get("centroid_longitude") else None,
            governance_type=result.get("governance_type"),
            management_authority=result.get("management_authority"),
            data_source=DataSourceEnum(result.get("data_source", "wdpa")),
            iucn_category=result.get("iucn_category"),
            is_active=result.get("is_active", True),
        )

        elapsed_ms = Decimal(str((time.monotonic() - start) * 1000))
        provenance_hash = _compute_provenance(
            f"update_area:{area_id}", str(update_data.keys()),
        )

        logger.info(
            "Protected area updated: area_id=%s fields=%s operator=%s",
            area_id,
            list(update_data.keys()),
            user.operator_id or user.user_id,
        )

        return ProtectedAreaResponse(
            area=area_entry,
            provenance=ProvenanceInfo(
                provenance_hash=provenance_hash,
                processing_time_ms=elapsed_ms.quantize(Decimal("0.01")),
            ),
            metadata=MetadataSchema(
                data_sources=["ProtectedAreaEngine"],
            ),
        )

    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Protected area update failed: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Protected area update failed",
        )


# ---------------------------------------------------------------------------
# DELETE /protected-areas/{area_id}
# ---------------------------------------------------------------------------


@router.delete(
    "/{area_id}",
    status_code=status.HTTP_200_OK,
    summary="Archive a protected area",
    description=(
        "Soft-delete (archive) a protected area. The record is retained "
        "for audit purposes but marked as inactive."
    ),
    responses={
        200: {"description": "Protected area archived"},
        401: {"model": ErrorResponse, "description": "Authentication required"},
        403: {"model": ErrorResponse, "description": "Insufficient permissions"},
        404: {"model": ErrorResponse, "description": "Area not found"},
    },
)
async def archive_protected_area(
    area_id: str,
    request: Request,
    user: AuthUser = Depends(
        require_permission("eudr-pav:protected-area:delete")
    ),
    _rate: None = Depends(rate_limit_write),
) -> dict:
    """Archive (soft-delete) a protected area.

    Args:
        area_id: Protected area identifier.
        user: Authenticated user with protected-area:delete permission.

    Returns:
        Confirmation of archival.
    """
    start = time.monotonic()

    try:
        engine = get_protected_area_engine()
        result = engine.archive_area(area_id=area_id, archived_by=user.user_id)

        if result is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Protected area not found: {area_id}",
            )

        elapsed_ms = round((time.monotonic() - start) * 1000, 2)

        logger.info(
            "Protected area archived: area_id=%s operator=%s",
            area_id,
            user.operator_id or user.user_id,
        )

        return {
            "status": "archived",
            "area_id": area_id,
            "archived_by": user.user_id,
            "processing_time_ms": elapsed_ms,
        }

    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Protected area archival failed: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Protected area archival failed",
        )


# ---------------------------------------------------------------------------
# POST /protected-areas/search
# ---------------------------------------------------------------------------


@router.post(
    "/search",
    response_model=ProtectedAreaSearchResponse,
    summary="Advanced spatial search of protected areas",
    description=(
        "Perform an advanced spatial search of protected areas using center+radius, "
        "bounding box, or polygon intersection queries. Supports filtering by "
        "country, area type, designation status, and area size range."
    ),
    responses={
        200: {"description": "Search results returned"},
        400: {"model": ErrorResponse, "description": "Invalid search parameters"},
        401: {"model": ErrorResponse, "description": "Authentication required"},
        403: {"model": ErrorResponse, "description": "Insufficient permissions"},
        429: {"model": ErrorResponse, "description": "Rate limit exceeded"},
    },
)
async def search_protected_areas(
    request: Request,
    body: ProtectedAreaSearchRequest,
    pagination: PaginationParams = Depends(get_pagination),
    user: AuthUser = Depends(
        require_permission("eudr-pav:protected-area:read")
    ),
    _rate: None = Depends(rate_limit_heavy),
) -> ProtectedAreaSearchResponse:
    """Advanced spatial search of protected areas.

    Args:
        body: Search request with spatial and attribute filters.
        pagination: Pagination parameters.
        user: Authenticated user with protected-area:read permission.

    Returns:
        ProtectedAreaSearchResponse with matching areas.
    """
    start = time.monotonic()

    try:
        engine = get_protected_area_engine()

        search_params: dict = {
            "include_buffer": body.include_buffer,
            "limit": pagination.limit,
            "offset": pagination.offset,
        }

        if body.center:
            search_params["latitude"] = float(body.center.latitude)
            search_params["longitude"] = float(body.center.longitude)
            search_params["radius_km"] = float(body.radius_km) if body.radius_km else 50.0
        if body.bounding_box:
            search_params["bounding_box"] = {
                "min_lat": float(body.bounding_box.min_latitude),
                "max_lat": float(body.bounding_box.max_latitude),
                "min_lon": float(body.bounding_box.min_longitude),
                "max_lon": float(body.bounding_box.max_longitude),
            }
        if body.polygon:
            search_params["polygon"] = [
                {"latitude": float(p.latitude), "longitude": float(p.longitude)}
                for p in body.polygon.coordinates
            ]
        if body.country_codes:
            search_params["country_codes"] = [c.upper() for c in body.country_codes]
        if body.area_types:
            search_params["area_types"] = [t.value for t in body.area_types]
        if body.designation_statuses:
            search_params["designation_statuses"] = [s.value for s in body.designation_statuses]
        if body.data_sources:
            search_params["data_sources"] = [s.value for s in body.data_sources]
        if body.min_area_km2 is not None:
            search_params["min_area_km2"] = float(body.min_area_km2)
        if body.max_area_km2 is not None:
            search_params["max_area_km2"] = float(body.max_area_km2)

        result = engine.search_areas(**search_params)

        areas = []
        for a in result.get("areas", []):
            areas.append(
                ProtectedAreaEntry(
                    area_id=a.get("area_id", ""),
                    name=a.get("name", ""),
                    area_type=ProtectedAreaTypeEnum(a.get("area_type", "other")),
                    country_code=a.get("country_code", ""),
                    wdpa_id=a.get("wdpa_id"),
                    designation_status=DesignationStatusEnum(a.get("designation_status", "unknown")),
                    designation_date=a.get("designation_date"),
                    total_area_km2=Decimal(str(a.get("total_area_km2", 0))),
                    buffer_zone_km=Decimal(str(a.get("buffer_zone_km", 5))),
                    centroid_latitude=Decimal(str(a["centroid_latitude"])) if a.get("centroid_latitude") else None,
                    centroid_longitude=Decimal(str(a["centroid_longitude"])) if a.get("centroid_longitude") else None,
                    governance_type=a.get("governance_type"),
                    management_authority=a.get("management_authority"),
                    data_source=DataSourceEnum(a.get("data_source", "wdpa")),
                    iucn_category=a.get("iucn_category"),
                    is_active=a.get("is_active", True),
                )
            )

        total = result.get("total", len(areas))

        elapsed_ms = Decimal(str((time.monotonic() - start) * 1000))
        provenance_hash = _compute_provenance("search_areas", str(total))

        logger.info(
            "Protected area search: results=%d total=%d operator=%s",
            len(areas),
            total,
            user.operator_id or user.user_id,
        )

        return ProtectedAreaSearchResponse(
            areas=areas,
            total_results=total,
            search_area_km2=Decimal(str(result.get("search_area_km2", 0)))
            if result.get("search_area_km2") else None,
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
                data_sources=["ProtectedAreaEngine", "WDPA", "OECM"],
            ),
        )

    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Protected area search failed: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Protected area search failed",
        )
