# -*- coding: utf-8 -*-
"""
Indigenous Community Registry Routes - AGENT-EUDR-021 Indigenous Rights Checker API

Endpoints for registering and managing indigenous communities in the
GreenLang platform. Communities are linked to territories and used for
FPIC tracking, consultation management, and violation attribution.

Endpoints:
    POST /communities                     - Register indigenous community
    GET  /communities                     - List communities with filters
    GET  /communities/{community_id}      - Get community details

Communities are registered with name, country, population, language,
territory associations, and legal recognition status.

Author: GreenLang Platform Team
Date: March 2026
PRD: AGENT-EUDR-021, CommunityRegistry Engine
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
    get_community_registry,
    get_pagination,
    rate_limit_standard,
    rate_limit_write,
    require_permission,
    validate_country_code,
)
from greenlang.agents.eudr.indigenous_rights_checker.api.schemas import (
    CommunityEntry,
    CommunityListResponse,
    CommunityRegisterRequest,
    CommunityResponse,
    CommunityStatusEnum,
    ErrorResponse,
    MetadataSchema,
    PaginatedMeta,
    ProvenanceInfo,
    RecognitionLevelEnum,
    SortOrderEnum,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/communities", tags=["Indigenous Community Registry"])


def _compute_provenance(input_data: str, output_data: str) -> str:
    """Compute SHA-256 provenance hash for audit trail."""
    data_str = f"{input_data}{output_data}"
    return hashlib.sha256(data_str.encode("utf-8")).hexdigest()


def _build_community_entry(entry: dict) -> CommunityEntry:
    """Build a CommunityEntry from engine result dictionary."""
    return CommunityEntry(
        community_id=entry.get("community_id", ""),
        name=entry.get("name", ""),
        country_code=entry.get("country_code", ""),
        status=CommunityStatusEnum(entry.get("status", "active")),
        region=entry.get("region"),
        population=entry.get("population"),
        language=entry.get("language"),
        secondary_languages=entry.get("secondary_languages"),
        territory_ids=entry.get("territory_ids"),
        territory_count=entry.get("territory_count"),
        contact_info=entry.get("contact_info"),
        recognition_status=RecognitionLevelEnum(entry.get("recognition_status"))
        if entry.get("recognition_status") else None,
        description=entry.get("description"),
        tags=entry.get("tags"),
        created_at=entry.get("created_at"),
        updated_at=entry.get("updated_at"),
    )


# ---------------------------------------------------------------------------
# POST /communities
# ---------------------------------------------------------------------------


@router.post(
    "",
    response_model=CommunityResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Register indigenous community",
    description=(
        "Register a new indigenous community in the GreenLang platform. "
        "Communities are linked to territories for FPIC tracking and "
        "consultation management. Supports population estimates, language "
        "information, and legal recognition status."
    ),
    responses={
        201: {"description": "Community registered successfully"},
        400: {"model": ErrorResponse, "description": "Invalid request data"},
        401: {"model": ErrorResponse, "description": "Authentication required"},
        403: {"model": ErrorResponse, "description": "Insufficient permissions"},
        409: {"model": ErrorResponse, "description": "Community already exists"},
        429: {"model": ErrorResponse, "description": "Rate limit exceeded"},
    },
)
async def register_community(
    request: Request,
    body: CommunityRegisterRequest,
    user: AuthUser = Depends(
        require_permission("eudr-irc:consultations:create")
    ),
    _rate: None = Depends(rate_limit_write),
) -> CommunityResponse:
    """Register a new indigenous community.

    Args:
        body: Community registration request.
        user: Authenticated user with consultations:create permission.

    Returns:
        CommunityResponse with the created community record.
    """
    start = time.monotonic()

    try:
        engine = get_community_registry()
        result = engine.register_community(
            name=body.name,
            country_code=body.country_code,
            region=body.region,
            population=body.population,
            language=body.language,
            secondary_languages=body.secondary_languages,
            territory_ids=body.territory_ids,
            contact_info=body.contact_info,
            recognition_status=body.recognition_status.value if body.recognition_status else None,
            description=body.description,
            tags=body.tags,
            created_by=user.user_id,
        )

        community_data = result.get("community", {})
        community_entry = _build_community_entry(community_data)

        elapsed_ms = Decimal(str((time.monotonic() - start) * 1000))
        provenance_hash = _compute_provenance(
            f"register_community:{body.name}:{body.country_code}",
            community_entry.community_id,
        )

        logger.info(
            "Community registered: id=%s name=%s country=%s operator=%s",
            community_entry.community_id,
            body.name,
            body.country_code,
            user.operator_id or user.user_id,
        )

        return CommunityResponse(
            community=community_entry,
            provenance=ProvenanceInfo(
                provenance_hash=provenance_hash,
                processing_time_ms=elapsed_ms.quantize(Decimal("0.01")),
            ),
            metadata=MetadataSchema(
                data_sources=["IndigenousRightsChecker", "CommunityRegistry"],
            ),
        )

    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Community registration failed: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Community registration failed",
        )


# ---------------------------------------------------------------------------
# GET /communities
# ---------------------------------------------------------------------------


@router.get(
    "",
    response_model=CommunityListResponse,
    summary="List indigenous communities",
    description=(
        "Retrieve a paginated list of registered indigenous communities with "
        "optional filters for country, recognition status, territory "
        "association, and name search."
    ),
    responses={
        200: {"description": "Communities listed successfully"},
        400: {"model": ErrorResponse, "description": "Invalid filter parameters"},
        401: {"model": ErrorResponse, "description": "Authentication required"},
        403: {"model": ErrorResponse, "description": "Insufficient permissions"},
    },
)
async def list_communities(
    request: Request,
    country_code: Optional[str] = Query(
        None, description="Filter by ISO 3166-1 alpha-2 country code"
    ),
    recognition_status: Optional[RecognitionLevelEnum] = Query(
        None, description="Filter by recognition status"
    ),
    community_status: Optional[CommunityStatusEnum] = Query(
        None, alias="status", description="Filter by community status"
    ),
    territory_id: Optional[str] = Query(
        None, description="Filter by associated territory ID"
    ),
    search: Optional[str] = Query(
        None, max_length=200, description="Search by community name"
    ),
    sort_by: Optional[str] = Query(
        "name", description="Sort field (name, created_at, population)"
    ),
    sort_order: Optional[SortOrderEnum] = Query(
        SortOrderEnum.ASC, description="Sort order"
    ),
    pagination: PaginationParams = Depends(get_pagination),
    user: AuthUser = Depends(
        require_permission("eudr-irc:consultations:read")
    ),
    _rate: None = Depends(rate_limit_standard),
) -> CommunityListResponse:
    """List communities with optional filters and pagination.

    Args:
        country_code: Optional country filter.
        recognition_status: Optional recognition status filter.
        community_status: Optional status filter.
        territory_id: Optional territory association filter.
        search: Optional name search.
        sort_by: Sort field.
        sort_order: Sort direction.
        pagination: Pagination parameters.
        user: Authenticated user.

    Returns:
        CommunityListResponse with paginated community list.
    """
    start = time.monotonic()

    try:
        engine = get_community_registry()
        result = engine.list_communities(
            country_code=validate_country_code(country_code) if country_code else None,
            recognition_status=recognition_status.value if recognition_status else None,
            status=community_status.value if community_status else None,
            territory_id=territory_id,
            search=search,
            sort_by=sort_by,
            sort_order=sort_order.value if sort_order else "asc",
            limit=pagination.limit,
            offset=pagination.offset,
        )

        communities = [
            _build_community_entry(entry)
            for entry in result.get("communities", [])
        ]
        total = result.get("total", len(communities))

        elapsed_ms = Decimal(str((time.monotonic() - start) * 1000))
        provenance_hash = _compute_provenance(
            f"list_communities:{country_code}:{recognition_status}:{community_status}",
            str(total),
        )

        logger.info(
            "Communities listed: total=%d returned=%d operator=%s",
            total,
            len(communities),
            user.operator_id or user.user_id,
        )

        return CommunityListResponse(
            communities=communities,
            total_communities=total,
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
                data_sources=["IndigenousRightsChecker", "CommunityRegistry"],
            ),
        )

    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Community listing failed: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Community listing failed",
        )


# ---------------------------------------------------------------------------
# GET /communities/{community_id}
# ---------------------------------------------------------------------------


@router.get(
    "/{community_id}",
    response_model=CommunityResponse,
    summary="Get community details",
    description=(
        "Retrieve detailed information for a specific indigenous community "
        "including population, languages, territory associations, contact "
        "information, and legal recognition status."
    ),
    responses={
        200: {"description": "Community details retrieved"},
        401: {"model": ErrorResponse, "description": "Authentication required"},
        403: {"model": ErrorResponse, "description": "Insufficient permissions"},
        404: {"model": ErrorResponse, "description": "Community not found"},
    },
)
async def get_community(
    community_id: str,
    request: Request,
    user: AuthUser = Depends(
        require_permission("eudr-irc:consultations:read")
    ),
    _rate: None = Depends(rate_limit_standard),
) -> CommunityResponse:
    """Get detailed information for a specific community.

    Args:
        community_id: Unique community identifier.
        user: Authenticated user.

    Returns:
        CommunityResponse with full community details.
    """
    start = time.monotonic()

    try:
        engine = get_community_registry()
        result = engine.get_community(community_id=community_id)

        if result is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Community not found: {community_id}",
            )

        community_data = result.get("community", {})
        community_entry = _build_community_entry(community_data)

        elapsed_ms = Decimal(str((time.monotonic() - start) * 1000))
        provenance_hash = _compute_provenance(
            f"get_community:{community_id}",
            community_entry.name,
        )

        logger.info(
            "Community retrieved: id=%s name=%s country=%s operator=%s",
            community_id,
            community_entry.name,
            community_entry.country_code,
            user.operator_id or user.user_id,
        )

        return CommunityResponse(
            community=community_entry,
            provenance=ProvenanceInfo(
                provenance_hash=provenance_hash,
                processing_time_ms=elapsed_ms.quantize(Decimal("0.01")),
            ),
            metadata=MetadataSchema(
                data_sources=["IndigenousRightsChecker", "CommunityRegistry"],
            ),
        )

    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Community retrieval failed: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Community retrieval failed",
        )
