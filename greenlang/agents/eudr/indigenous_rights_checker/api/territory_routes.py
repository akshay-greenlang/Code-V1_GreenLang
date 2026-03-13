# -*- coding: utf-8 -*-
"""
Territory Management Routes - AGENT-EUDR-021 Indigenous Rights Checker API

Endpoints for indigenous territory registration, listing with filters,
detail retrieval, update, and archival. Territories define the spatial
boundaries of indigenous and traditional lands for EUDR overlap analysis,
FPIC verification, and compliance reporting.

Endpoints:
    POST   /territories                  - Register indigenous territory
    GET    /territories                  - List territories with filters
    GET    /territories/{territory_id}   - Get territory details
    PUT    /territories/{territory_id}   - Update territory
    DELETE /territories/{territory_id}   - Archive territory (soft delete)

All endpoints require JWT auth (SEC-001) and eudr-irc:territories:*
permissions (SEC-002). Rate limited to 100 req/min standard reads,
30 req/min writes.

Author: GreenLang Platform Team
Date: March 2026
PRD: AGENT-EUDR-021, TerritoryManager Engine
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
    get_pagination,
    get_territory_manager,
    rate_limit_standard,
    rate_limit_write,
    require_permission,
    validate_country_code,
)
from greenlang.agents.eudr.indigenous_rights_checker.api.schemas import (
    ErrorResponse,
    MetadataSchema,
    PaginatedMeta,
    ProvenanceInfo,
    RecognitionLevelEnum,
    SortOrderEnum,
    TerritoryCreateRequest,
    TerritoryEntry,
    TerritoryListResponse,
    TerritoryResponse,
    TerritoryStatusEnum,
    TerritoryTypeEnum,
    TerritoryUpdateRequest,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/territories", tags=["Territory Management"])


def _compute_provenance(input_data: str, output_data: str) -> str:
    """Compute SHA-256 provenance hash for audit trail."""
    data_str = f"{input_data}{output_data}"
    return hashlib.sha256(data_str.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# POST /territories
# ---------------------------------------------------------------------------


@router.post(
    "",
    response_model=TerritoryResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Register indigenous territory",
    description=(
        "Register a new indigenous or traditional territory with boundary "
        "polygon, legal recognition level, and community association. "
        "The territory will be used for EUDR overlap analysis and FPIC "
        "verification. Requires eudr-irc:territories:create permission."
    ),
    responses={
        201: {"description": "Territory registered successfully"},
        400: {"model": ErrorResponse, "description": "Invalid request data"},
        401: {"model": ErrorResponse, "description": "Authentication required"},
        403: {"model": ErrorResponse, "description": "Insufficient permissions"},
        409: {"model": ErrorResponse, "description": "Territory already exists"},
        429: {"model": ErrorResponse, "description": "Rate limit exceeded"},
    },
)
async def register_territory(
    request: Request,
    body: TerritoryCreateRequest,
    user: AuthUser = Depends(
        require_permission("eudr-irc:territories:create")
    ),
    _rate: None = Depends(rate_limit_write),
) -> TerritoryResponse:
    """Register a new indigenous territory.

    Args:
        body: Territory registration request with boundary and metadata.
        user: Authenticated user with territories:create permission.

    Returns:
        TerritoryResponse with the created territory record.
    """
    start = time.monotonic()

    try:
        engine = get_territory_manager()
        result = engine.register_territory(
            name=body.name,
            country_code=body.country_code,
            territory_type=body.territory_type.value,
            recognition_level=body.recognition_level.value,
            community_id=body.community_id,
            boundary={
                "coordinates": [
                    {"latitude": float(p.latitude), "longitude": float(p.longitude)}
                    for p in body.boundary.coordinates
                ],
                "srid": body.boundary.srid,
            },
            area_ha=float(body.area_ha) if body.area_ha else None,
            description=body.description,
            legal_reference=body.legal_reference,
            established_date=str(body.established_date) if body.established_date else None,
            tags=body.tags,
            created_by=user.user_id,
        )

        territory_data = result.get("territory", {})
        territory_entry = TerritoryEntry(
            territory_id=territory_data.get("territory_id", ""),
            name=territory_data.get("name", body.name),
            country_code=territory_data.get("country_code", body.country_code),
            territory_type=TerritoryTypeEnum(
                territory_data.get("territory_type", body.territory_type.value)
            ),
            recognition_level=RecognitionLevelEnum(
                territory_data.get("recognition_level", body.recognition_level.value)
            ),
            status=TerritoryStatusEnum(territory_data.get("status", "active")),
            community_id=territory_data.get("community_id", body.community_id),
            area_ha=Decimal(str(territory_data.get("area_ha", body.area_ha or 0)))
            if territory_data.get("area_ha") or body.area_ha else None,
            description=territory_data.get("description", body.description),
            legal_reference=territory_data.get("legal_reference", body.legal_reference),
            tags=territory_data.get("tags", body.tags),
            created_at=territory_data.get("created_at"),
        )

        elapsed_ms = Decimal(str((time.monotonic() - start) * 1000))
        provenance_hash = _compute_provenance(
            f"register_territory:{body.name}:{body.country_code}",
            territory_entry.territory_id,
        )

        logger.info(
            "Territory registered: territory_id=%s name=%s country=%s operator=%s",
            territory_entry.territory_id,
            territory_entry.name,
            territory_entry.country_code,
            user.operator_id or user.user_id,
        )

        return TerritoryResponse(
            territory=territory_entry,
            provenance=ProvenanceInfo(
                provenance_hash=provenance_hash,
                processing_time_ms=elapsed_ms.quantize(Decimal("0.01")),
            ),
            metadata=MetadataSchema(
                data_sources=["IndigenousRightsChecker", "TerritoryManager"],
            ),
        )

    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Territory registration failed: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Territory registration failed",
        )


# ---------------------------------------------------------------------------
# GET /territories
# ---------------------------------------------------------------------------


@router.get(
    "",
    response_model=TerritoryListResponse,
    summary="List indigenous territories with filters",
    description=(
        "Retrieve a paginated list of indigenous territories with optional "
        "filters for country, territory type, recognition level, status, "
        "and community. Results can be sorted by name or creation date."
    ),
    responses={
        200: {"description": "Territories listed successfully"},
        400: {"model": ErrorResponse, "description": "Invalid filter parameters"},
        401: {"model": ErrorResponse, "description": "Authentication required"},
        403: {"model": ErrorResponse, "description": "Insufficient permissions"},
    },
)
async def list_territories(
    request: Request,
    country_code: Optional[str] = Query(
        None, description="Filter by ISO 3166-1 alpha-2 country code"
    ),
    territory_type: Optional[TerritoryTypeEnum] = Query(
        None, description="Filter by territory type"
    ),
    recognition_level: Optional[RecognitionLevelEnum] = Query(
        None, description="Filter by recognition level"
    ),
    territory_status: Optional[TerritoryStatusEnum] = Query(
        None, alias="status", description="Filter by territory status"
    ),
    community_id: Optional[str] = Query(
        None, description="Filter by associated community ID"
    ),
    search: Optional[str] = Query(
        None, max_length=200, description="Search by territory name"
    ),
    sort_by: Optional[str] = Query(
        "created_at", description="Sort field (name, created_at, area_ha)"
    ),
    sort_order: Optional[SortOrderEnum] = Query(
        SortOrderEnum.DESC, description="Sort order"
    ),
    pagination: PaginationParams = Depends(get_pagination),
    user: AuthUser = Depends(
        require_permission("eudr-irc:territories:read")
    ),
    _rate: None = Depends(rate_limit_standard),
) -> TerritoryListResponse:
    """List territories with optional filters and pagination.

    Args:
        country_code: Optional country filter.
        territory_type: Optional territory type filter.
        recognition_level: Optional recognition level filter.
        territory_status: Optional status filter.
        community_id: Optional community filter.
        search: Optional name search.
        sort_by: Sort field.
        sort_order: Sort direction.
        pagination: Pagination parameters.
        user: Authenticated user.

    Returns:
        TerritoryListResponse with paginated territory list.
    """
    start = time.monotonic()

    try:
        engine = get_territory_manager()
        result = engine.list_territories(
            country_code=validate_country_code(country_code) if country_code else None,
            territory_type=territory_type.value if territory_type else None,
            recognition_level=recognition_level.value if recognition_level else None,
            status=territory_status.value if territory_status else None,
            community_id=community_id,
            search=search,
            sort_by=sort_by,
            sort_order=sort_order.value if sort_order else "desc",
            limit=pagination.limit,
            offset=pagination.offset,
        )

        territories = []
        for entry in result.get("territories", []):
            territories.append(
                TerritoryEntry(
                    territory_id=entry.get("territory_id", ""),
                    name=entry.get("name", ""),
                    country_code=entry.get("country_code", ""),
                    territory_type=TerritoryTypeEnum(
                        entry.get("territory_type", "indigenous_land")
                    ),
                    recognition_level=RecognitionLevelEnum(
                        entry.get("recognition_level", "customary_only")
                    ),
                    status=TerritoryStatusEnum(entry.get("status", "active")),
                    community_id=entry.get("community_id"),
                    area_ha=Decimal(str(entry.get("area_ha")))
                    if entry.get("area_ha") is not None else None,
                    description=entry.get("description"),
                    legal_reference=entry.get("legal_reference"),
                    tags=entry.get("tags"),
                    created_at=entry.get("created_at"),
                    updated_at=entry.get("updated_at"),
                )
            )

        total = result.get("total", len(territories))
        elapsed_ms = Decimal(str((time.monotonic() - start) * 1000))
        provenance_hash = _compute_provenance(
            f"list_territories:{country_code}:{territory_type}:{territory_status}",
            str(total),
        )

        logger.info(
            "Territories listed: total=%d returned=%d operator=%s",
            total,
            len(territories),
            user.operator_id or user.user_id,
        )

        return TerritoryListResponse(
            territories=territories,
            total_territories=total,
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
                data_sources=["IndigenousRightsChecker", "TerritoryManager"],
            ),
        )

    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Territory listing failed: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Territory listing failed",
        )


# ---------------------------------------------------------------------------
# GET /territories/{territory_id}
# ---------------------------------------------------------------------------


@router.get(
    "/{territory_id}",
    response_model=TerritoryResponse,
    summary="Get territory details",
    description=(
        "Retrieve detailed information for a specific indigenous territory "
        "including boundary polygon, legal recognition, community association, "
        "and classification tags."
    ),
    responses={
        200: {"description": "Territory details retrieved"},
        401: {"model": ErrorResponse, "description": "Authentication required"},
        403: {"model": ErrorResponse, "description": "Insufficient permissions"},
        404: {"model": ErrorResponse, "description": "Territory not found"},
    },
)
async def get_territory(
    territory_id: str,
    request: Request,
    user: AuthUser = Depends(
        require_permission("eudr-irc:territories:read")
    ),
    _rate: None = Depends(rate_limit_standard),
) -> TerritoryResponse:
    """Get detailed information for a specific territory.

    Args:
        territory_id: Unique territory identifier.
        user: Authenticated user.

    Returns:
        TerritoryResponse with full territory details.
    """
    start = time.monotonic()

    try:
        engine = get_territory_manager()
        result = engine.get_territory(territory_id=territory_id)

        if result is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Territory not found: {territory_id}",
            )

        territory_data = result.get("territory", {})
        boundary_data = territory_data.get("boundary")

        territory_entry = TerritoryEntry(
            territory_id=territory_data.get("territory_id", territory_id),
            name=territory_data.get("name", ""),
            country_code=territory_data.get("country_code", ""),
            territory_type=TerritoryTypeEnum(
                territory_data.get("territory_type", "indigenous_land")
            ),
            recognition_level=RecognitionLevelEnum(
                territory_data.get("recognition_level", "customary_only")
            ),
            status=TerritoryStatusEnum(territory_data.get("status", "active")),
            community_id=territory_data.get("community_id"),
            area_ha=Decimal(str(territory_data.get("area_ha")))
            if territory_data.get("area_ha") is not None else None,
            description=territory_data.get("description"),
            legal_reference=territory_data.get("legal_reference"),
            established_date=territory_data.get("established_date"),
            tags=territory_data.get("tags"),
            created_at=territory_data.get("created_at"),
            updated_at=territory_data.get("updated_at"),
        )

        elapsed_ms = Decimal(str((time.monotonic() - start) * 1000))
        provenance_hash = _compute_provenance(
            f"get_territory:{territory_id}",
            territory_entry.name,
        )

        logger.info(
            "Territory detail retrieved: territory_id=%s name=%s operator=%s",
            territory_id,
            territory_entry.name,
            user.operator_id or user.user_id,
        )

        return TerritoryResponse(
            territory=territory_entry,
            provenance=ProvenanceInfo(
                provenance_hash=provenance_hash,
                processing_time_ms=elapsed_ms.quantize(Decimal("0.01")),
            ),
            metadata=MetadataSchema(
                data_sources=["IndigenousRightsChecker", "TerritoryManager"],
            ),
        )

    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Territory detail retrieval failed: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Territory detail retrieval failed",
        )


# ---------------------------------------------------------------------------
# PUT /territories/{territory_id}
# ---------------------------------------------------------------------------


@router.put(
    "/{territory_id}",
    response_model=TerritoryResponse,
    summary="Update territory",
    description=(
        "Update an existing indigenous territory record. Supports partial "
        "updates - only provided fields are modified. Boundary polygon, "
        "recognition level, and status can all be updated."
    ),
    responses={
        200: {"description": "Territory updated successfully"},
        400: {"model": ErrorResponse, "description": "Invalid request data"},
        401: {"model": ErrorResponse, "description": "Authentication required"},
        403: {"model": ErrorResponse, "description": "Insufficient permissions"},
        404: {"model": ErrorResponse, "description": "Territory not found"},
    },
)
async def update_territory(
    territory_id: str,
    request: Request,
    body: TerritoryUpdateRequest,
    user: AuthUser = Depends(
        require_permission("eudr-irc:territories:update")
    ),
    _rate: None = Depends(rate_limit_write),
) -> TerritoryResponse:
    """Update an existing territory.

    Args:
        territory_id: Unique territory identifier.
        body: Partial update request.
        user: Authenticated user with territories:update permission.

    Returns:
        TerritoryResponse with updated territory record.
    """
    start = time.monotonic()

    try:
        engine = get_territory_manager()

        # Build update payload from non-None fields
        update_data = body.model_dump(exclude_none=True)
        if "territory_type" in update_data:
            update_data["territory_type"] = body.territory_type.value
        if "recognition_level" in update_data:
            update_data["recognition_level"] = body.recognition_level.value
        if "status" in update_data:
            update_data["status"] = body.status.value
        if "boundary" in update_data and body.boundary:
            update_data["boundary"] = {
                "coordinates": [
                    {"latitude": float(p.latitude), "longitude": float(p.longitude)}
                    for p in body.boundary.coordinates
                ],
                "srid": body.boundary.srid,
            }
        if "area_ha" in update_data and body.area_ha is not None:
            update_data["area_ha"] = float(body.area_ha)

        update_data["updated_by"] = user.user_id

        result = engine.update_territory(
            territory_id=territory_id,
            **update_data,
        )

        if result is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Territory not found: {territory_id}",
            )

        territory_data = result.get("territory", {})
        territory_entry = TerritoryEntry(
            territory_id=territory_data.get("territory_id", territory_id),
            name=territory_data.get("name", ""),
            country_code=territory_data.get("country_code", ""),
            territory_type=TerritoryTypeEnum(
                territory_data.get("territory_type", "indigenous_land")
            ),
            recognition_level=RecognitionLevelEnum(
                territory_data.get("recognition_level", "customary_only")
            ),
            status=TerritoryStatusEnum(territory_data.get("status", "active")),
            community_id=territory_data.get("community_id"),
            area_ha=Decimal(str(territory_data.get("area_ha")))
            if territory_data.get("area_ha") is not None else None,
            description=territory_data.get("description"),
            legal_reference=territory_data.get("legal_reference"),
            tags=territory_data.get("tags"),
            created_at=territory_data.get("created_at"),
            updated_at=territory_data.get("updated_at"),
        )

        elapsed_ms = Decimal(str((time.monotonic() - start) * 1000))
        provenance_hash = _compute_provenance(
            f"update_territory:{territory_id}:{list(update_data.keys())}",
            territory_entry.territory_id,
        )

        logger.info(
            "Territory updated: territory_id=%s fields=%s operator=%s",
            territory_id,
            list(update_data.keys()),
            user.operator_id or user.user_id,
        )

        return TerritoryResponse(
            territory=territory_entry,
            provenance=ProvenanceInfo(
                provenance_hash=provenance_hash,
                processing_time_ms=elapsed_ms.quantize(Decimal("0.01")),
            ),
            metadata=MetadataSchema(
                data_sources=["IndigenousRightsChecker", "TerritoryManager"],
            ),
        )

    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Territory update failed: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Territory update failed",
        )


# ---------------------------------------------------------------------------
# DELETE /territories/{territory_id}
# ---------------------------------------------------------------------------


@router.delete(
    "/{territory_id}",
    response_model=TerritoryResponse,
    summary="Archive territory",
    description=(
        "Soft-delete (archive) an indigenous territory. The territory record "
        "is retained for audit purposes with status set to 'archived'. "
        "Archived territories are excluded from active overlap analysis."
    ),
    responses={
        200: {"description": "Territory archived successfully"},
        401: {"model": ErrorResponse, "description": "Authentication required"},
        403: {"model": ErrorResponse, "description": "Insufficient permissions"},
        404: {"model": ErrorResponse, "description": "Territory not found"},
    },
)
async def archive_territory(
    territory_id: str,
    request: Request,
    user: AuthUser = Depends(
        require_permission("eudr-irc:territories:delete")
    ),
    _rate: None = Depends(rate_limit_write),
) -> TerritoryResponse:
    """Archive (soft delete) a territory.

    Args:
        territory_id: Unique territory identifier.
        user: Authenticated user with territories:delete permission.

    Returns:
        TerritoryResponse with archived territory record.
    """
    start = time.monotonic()

    try:
        engine = get_territory_manager()
        result = engine.archive_territory(
            territory_id=territory_id,
            archived_by=user.user_id,
        )

        if result is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Territory not found: {territory_id}",
            )

        territory_data = result.get("territory", {})
        territory_entry = TerritoryEntry(
            territory_id=territory_data.get("territory_id", territory_id),
            name=territory_data.get("name", ""),
            country_code=territory_data.get("country_code", ""),
            territory_type=TerritoryTypeEnum(
                territory_data.get("territory_type", "indigenous_land")
            ),
            recognition_level=RecognitionLevelEnum(
                territory_data.get("recognition_level", "customary_only")
            ),
            status=TerritoryStatusEnum.ARCHIVED,
            community_id=territory_data.get("community_id"),
            area_ha=Decimal(str(territory_data.get("area_ha")))
            if territory_data.get("area_ha") is not None else None,
            description=territory_data.get("description"),
            legal_reference=territory_data.get("legal_reference"),
            tags=territory_data.get("tags"),
            created_at=territory_data.get("created_at"),
            updated_at=territory_data.get("updated_at"),
        )

        elapsed_ms = Decimal(str((time.monotonic() - start) * 1000))
        provenance_hash = _compute_provenance(
            f"archive_territory:{territory_id}",
            "archived",
        )

        logger.info(
            "Territory archived: territory_id=%s name=%s operator=%s",
            territory_id,
            territory_entry.name,
            user.operator_id or user.user_id,
        )

        return TerritoryResponse(
            territory=territory_entry,
            provenance=ProvenanceInfo(
                provenance_hash=provenance_hash,
                processing_time_ms=elapsed_ms.quantize(Decimal("0.01")),
            ),
            metadata=MetadataSchema(
                data_sources=["IndigenousRightsChecker", "TerritoryManager"],
            ),
        )

    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Territory archival failed: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Territory archival failed",
        )
