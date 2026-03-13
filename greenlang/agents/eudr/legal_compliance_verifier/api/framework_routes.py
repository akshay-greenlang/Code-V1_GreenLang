# -*- coding: utf-8 -*-
"""
Legal Framework Routes - AGENT-EUDR-023 Legal Compliance Verifier API

Endpoints for legal framework management including registration, listing,
details, updates, and advanced search across jurisdictions, commodities,
and regulatory scope per EUDR Articles 2, 3, 4, 8, 12.

Endpoints:
    POST /legal-frameworks           - Register a new legal framework
    GET  /legal-frameworks           - List legal frameworks (paginated)
    GET  /legal-frameworks/{id}      - Get framework details
    PUT  /legal-frameworks/{id}      - Update framework
    POST /legal-frameworks/search    - Advanced framework search

Author: GreenLang Platform Team
Date: March 2026
PRD: AGENT-EUDR-023, LegalFrameworkEngine
"""

from __future__ import annotations

import hashlib
import logging
import time
from decimal import Decimal
from typing import Any, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, Request, status

from greenlang.agents.eudr.legal_compliance_verifier.api.dependencies import (
    AuthUser,
    PaginationParams,
    get_framework_engine,
    get_pagination,
    rate_limit_standard,
    rate_limit_write,
    require_permission,
)
from greenlang.agents.eudr.legal_compliance_verifier.api.schemas import (
    EUDRCommodityEnum,
    ErrorResponse,
    FrameworkDetailResponse,
    FrameworkEntry,
    FrameworkListResponse,
    FrameworkRegisterRequest,
    FrameworkRegisterResponse,
    FrameworkSearchRequest,
    FrameworkSearchResponse,
    FrameworkStatusEnum,
    FrameworkUpdateRequest,
    JurisdictionTypeEnum,
    MetadataSchema,
    PaginatedMeta,
    ProvenanceInfo,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/legal-frameworks", tags=["Legal Frameworks"])


def _compute_provenance(input_data: Any, output_data: Any) -> str:
    """Compute SHA-256 provenance hash for audit trail."""
    data_str = f"{input_data}{output_data}"
    return hashlib.sha256(data_str.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# POST /legal-frameworks
# ---------------------------------------------------------------------------


@router.post(
    "",
    response_model=FrameworkRegisterResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Register a new legal framework",
    description=(
        "Register a legal framework (EU regulation, national law, bilateral "
        "agreement) for compliance tracking. Supports framework hierarchy "
        "via parent_framework_id for amendments."
    ),
    responses={
        201: {"description": "Framework registered successfully"},
        400: {"model": ErrorResponse, "description": "Invalid request"},
        401: {"model": ErrorResponse, "description": "Authentication required"},
        403: {"model": ErrorResponse, "description": "Insufficient permissions"},
        409: {"model": ErrorResponse, "description": "Framework already exists"},
        429: {"model": ErrorResponse, "description": "Rate limit exceeded"},
    },
)
async def register_framework(
    request: Request,
    body: FrameworkRegisterRequest,
    user: AuthUser = Depends(
        require_permission("eudr-lcv:framework:create")
    ),
    _rate: None = Depends(rate_limit_write),
) -> FrameworkRegisterResponse:
    """Register a new legal framework for compliance tracking.

    Args:
        body: Framework registration request with name, jurisdiction, etc.
        user: Authenticated user with framework:create permission.

    Returns:
        FrameworkRegisterResponse with registered framework details.
    """
    start = time.monotonic()

    try:
        engine = get_framework_engine()
        result = engine.register(
            name=body.name,
            jurisdiction=body.jurisdiction.value,
            country_codes=body.country_codes,
            effective_date=body.effective_date,
            expiry_date=body.expiry_date,
            description=body.description,
            reference_url=body.reference_url,
            articles=body.articles,
            commodities=[c.value for c in body.commodities] if body.commodities else None,
            parent_framework_id=body.parent_framework_id,
            created_by=user.user_id,
        )

        if result is None:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=f"Framework already exists: {body.name}",
            )

        framework = FrameworkEntry(
            framework_id=result.get("framework_id", ""),
            name=result.get("name", body.name),
            jurisdiction=JurisdictionTypeEnum(result.get("jurisdiction", body.jurisdiction.value)),
            country_codes=result.get("country_codes", body.country_codes),
            status=FrameworkStatusEnum(result.get("status", "active")),
            effective_date=body.effective_date,
            expiry_date=body.expiry_date,
            commodities=[EUDRCommodityEnum(c) for c in result.get("commodities", [])]
            if result.get("commodities") else [],
        )

        elapsed_ms = Decimal(str((time.monotonic() - start) * 1000))
        provenance_hash = _compute_provenance(
            f"framework_register:{body.name}",
            framework.framework_id,
        )

        logger.info(
            "Framework registered: id=%s name=%s jurisdiction=%s user=%s",
            framework.framework_id,
            body.name,
            body.jurisdiction.value,
            user.user_id,
        )

        return FrameworkRegisterResponse(
            framework=framework,
            provenance=ProvenanceInfo(
                provenance_hash=provenance_hash,
                processing_time_ms=elapsed_ms.quantize(Decimal("0.01")),
            ),
            metadata=MetadataSchema(
                data_sources=["LegalFrameworkEngine"],
            ),
        )

    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Framework registration failed: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Framework registration failed",
        )


# ---------------------------------------------------------------------------
# GET /legal-frameworks
# ---------------------------------------------------------------------------


@router.get(
    "",
    response_model=FrameworkListResponse,
    summary="List legal frameworks",
    description=(
        "Retrieve a paginated list of registered legal frameworks "
        "with optional filtering by jurisdiction, status, and commodity."
    ),
    responses={
        200: {"description": "Frameworks retrieved"},
        401: {"model": ErrorResponse, "description": "Authentication required"},
        403: {"model": ErrorResponse, "description": "Insufficient permissions"},
    },
)
async def list_frameworks(
    request: Request,
    jurisdiction: Optional[JurisdictionTypeEnum] = Query(
        None, description="Filter by jurisdiction type"
    ),
    framework_status: Optional[FrameworkStatusEnum] = Query(
        None, alias="status", description="Filter by status"
    ),
    commodity: Optional[EUDRCommodityEnum] = Query(
        None, description="Filter by commodity"
    ),
    country_code: Optional[str] = Query(
        None, description="Filter by country code"
    ),
    pagination: PaginationParams = Depends(get_pagination),
    user: AuthUser = Depends(
        require_permission("eudr-lcv:framework:read")
    ),
    _rate: None = Depends(rate_limit_standard),
) -> FrameworkListResponse:
    """List registered legal frameworks with pagination and filtering.

    Args:
        jurisdiction: Optional jurisdiction filter.
        framework_status: Optional status filter.
        commodity: Optional commodity filter.
        country_code: Optional country code filter.
        pagination: Pagination parameters.
        user: Authenticated user with framework:read permission.

    Returns:
        FrameworkListResponse with paginated frameworks.
    """
    start = time.monotonic()

    try:
        engine = get_framework_engine()
        result = engine.list_frameworks(
            jurisdiction=jurisdiction.value if jurisdiction else None,
            status=framework_status.value if framework_status else None,
            commodity=commodity.value if commodity else None,
            country_code=country_code,
            limit=pagination.limit,
            offset=pagination.offset,
        )

        frameworks = []
        for f in result.get("frameworks", []):
            frameworks.append(
                FrameworkEntry(
                    framework_id=f.get("framework_id", ""),
                    name=f.get("name", ""),
                    jurisdiction=JurisdictionTypeEnum(f.get("jurisdiction", "eu")),
                    country_codes=f.get("country_codes", []),
                    status=FrameworkStatusEnum(f.get("status", "active")),
                    effective_date=f.get("effective_date"),
                    expiry_date=f.get("expiry_date"),
                    commodities=[EUDRCommodityEnum(c) for c in f.get("commodities", [])],
                )
            )

        total = result.get("total", len(frameworks))

        elapsed_ms = Decimal(str((time.monotonic() - start) * 1000))
        provenance_hash = _compute_provenance(
            f"framework_list:{jurisdiction}:{framework_status}",
            str(total),
        )

        logger.info(
            "Frameworks listed: total=%d limit=%d offset=%d user=%s",
            total,
            pagination.limit,
            pagination.offset,
            user.user_id,
        )

        return FrameworkListResponse(
            frameworks=frameworks,
            total_frameworks=total,
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
                data_sources=["LegalFrameworkEngine"],
            ),
        )

    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Framework listing failed: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Framework listing failed",
        )


# ---------------------------------------------------------------------------
# GET /legal-frameworks/{framework_id}
# ---------------------------------------------------------------------------


@router.get(
    "/{framework_id}",
    response_model=FrameworkDetailResponse,
    summary="Get legal framework details",
    description=(
        "Retrieve full details of a legal framework including description, "
        "articles, amendment history, and parent framework references."
    ),
    responses={
        200: {"description": "Framework details retrieved"},
        401: {"model": ErrorResponse, "description": "Authentication required"},
        403: {"model": ErrorResponse, "description": "Insufficient permissions"},
        404: {"model": ErrorResponse, "description": "Framework not found"},
    },
)
async def get_framework_detail(
    framework_id: str,
    request: Request,
    user: AuthUser = Depends(
        require_permission("eudr-lcv:framework:read")
    ),
    _rate: None = Depends(rate_limit_standard),
) -> FrameworkDetailResponse:
    """Get detailed information about a legal framework.

    Args:
        framework_id: Unique framework identifier.
        user: Authenticated user with framework:read permission.

    Returns:
        FrameworkDetailResponse with full framework details.
    """
    start = time.monotonic()

    try:
        engine = get_framework_engine()
        result = engine.get_detail(framework_id=framework_id)

        if result is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Framework not found: {framework_id}",
            )

        framework = FrameworkEntry(
            framework_id=result.get("framework_id", framework_id),
            name=result.get("name", ""),
            jurisdiction=JurisdictionTypeEnum(result.get("jurisdiction", "eu")),
            country_codes=result.get("country_codes", []),
            status=FrameworkStatusEnum(result.get("status", "active")),
            effective_date=result.get("effective_date"),
            expiry_date=result.get("expiry_date"),
            commodities=[EUDRCommodityEnum(c) for c in result.get("commodities", [])],
        )

        elapsed_ms = Decimal(str((time.monotonic() - start) * 1000))
        provenance_hash = _compute_provenance(
            f"framework_detail:{framework_id}",
            framework.name,
        )

        logger.info(
            "Framework detail retrieved: id=%s name=%s user=%s",
            framework_id,
            framework.name,
            user.user_id,
        )

        return FrameworkDetailResponse(
            framework=framework,
            description=result.get("description"),
            reference_url=result.get("reference_url"),
            articles=result.get("articles", []),
            parent_framework_id=result.get("parent_framework_id"),
            amendment_history=result.get("amendment_history", []),
            provenance=ProvenanceInfo(
                provenance_hash=provenance_hash,
                processing_time_ms=elapsed_ms.quantize(Decimal("0.01")),
            ),
            metadata=MetadataSchema(
                data_sources=["LegalFrameworkEngine"],
            ),
        )

    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Framework detail retrieval failed: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Framework detail retrieval failed",
        )


# ---------------------------------------------------------------------------
# PUT /legal-frameworks/{framework_id}
# ---------------------------------------------------------------------------


@router.put(
    "/{framework_id}",
    response_model=FrameworkDetailResponse,
    summary="Update a legal framework",
    description=(
        "Update an existing legal framework's metadata, status, articles, "
        "or effective dates. Creates an amendment history entry."
    ),
    responses={
        200: {"description": "Framework updated"},
        400: {"model": ErrorResponse, "description": "Invalid request"},
        401: {"model": ErrorResponse, "description": "Authentication required"},
        403: {"model": ErrorResponse, "description": "Insufficient permissions"},
        404: {"model": ErrorResponse, "description": "Framework not found"},
    },
)
async def update_framework(
    framework_id: str,
    request: Request,
    body: FrameworkUpdateRequest,
    user: AuthUser = Depends(
        require_permission("eudr-lcv:framework:update")
    ),
    _rate: None = Depends(rate_limit_write),
) -> FrameworkDetailResponse:
    """Update an existing legal framework.

    Args:
        framework_id: Framework to update.
        body: Update request with changed fields.
        user: Authenticated user with framework:update permission.

    Returns:
        FrameworkDetailResponse with updated framework details.
    """
    start = time.monotonic()

    try:
        engine = get_framework_engine()
        result = engine.update(
            framework_id=framework_id,
            name=body.name,
            status=body.status.value if body.status else None,
            effective_date=body.effective_date,
            expiry_date=body.expiry_date,
            description=body.description,
            reference_url=body.reference_url,
            articles=body.articles,
            country_codes=body.country_codes,
            commodities=[c.value for c in body.commodities] if body.commodities else None,
            updated_by=user.user_id,
        )

        if result is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Framework not found: {framework_id}",
            )

        framework = FrameworkEntry(
            framework_id=result.get("framework_id", framework_id),
            name=result.get("name", ""),
            jurisdiction=JurisdictionTypeEnum(result.get("jurisdiction", "eu")),
            country_codes=result.get("country_codes", []),
            status=FrameworkStatusEnum(result.get("status", "active")),
            effective_date=result.get("effective_date"),
            expiry_date=result.get("expiry_date"),
            commodities=[EUDRCommodityEnum(c) for c in result.get("commodities", [])],
        )

        elapsed_ms = Decimal(str((time.monotonic() - start) * 1000))
        provenance_hash = _compute_provenance(
            f"framework_update:{framework_id}",
            str(result.get("status", "")),
        )

        logger.info(
            "Framework updated: id=%s user=%s",
            framework_id,
            user.user_id,
        )

        return FrameworkDetailResponse(
            framework=framework,
            description=result.get("description"),
            reference_url=result.get("reference_url"),
            articles=result.get("articles", []),
            parent_framework_id=result.get("parent_framework_id"),
            amendment_history=result.get("amendment_history", []),
            provenance=ProvenanceInfo(
                provenance_hash=provenance_hash,
                processing_time_ms=elapsed_ms.quantize(Decimal("0.01")),
            ),
            metadata=MetadataSchema(
                data_sources=["LegalFrameworkEngine"],
            ),
        )

    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Framework update failed: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Framework update failed",
        )


# ---------------------------------------------------------------------------
# POST /legal-frameworks/search
# ---------------------------------------------------------------------------


@router.post(
    "/search",
    response_model=FrameworkSearchResponse,
    summary="Advanced framework search",
    description=(
        "Search legal frameworks using multiple criteria including free-text "
        "query, jurisdiction, country, commodity, and date range."
    ),
    responses={
        200: {"description": "Search results returned"},
        400: {"model": ErrorResponse, "description": "Invalid request"},
        401: {"model": ErrorResponse, "description": "Authentication required"},
        403: {"model": ErrorResponse, "description": "Insufficient permissions"},
    },
)
async def search_frameworks(
    request: Request,
    body: FrameworkSearchRequest,
    pagination: PaginationParams = Depends(get_pagination),
    user: AuthUser = Depends(
        require_permission("eudr-lcv:framework:read")
    ),
    _rate: None = Depends(rate_limit_standard),
) -> FrameworkSearchResponse:
    """Search legal frameworks with advanced filtering.

    Args:
        body: Search criteria.
        pagination: Pagination parameters.
        user: Authenticated user with framework:read permission.

    Returns:
        FrameworkSearchResponse with matching frameworks.
    """
    start = time.monotonic()

    try:
        engine = get_framework_engine()
        result = engine.search(
            query=body.query,
            jurisdiction=body.jurisdiction.value if body.jurisdiction else None,
            country_code=body.country_code,
            status=body.status.value if body.status else None,
            commodity=body.commodity.value if body.commodity else None,
            effective_after=body.effective_after,
            effective_before=body.effective_before,
            limit=pagination.limit,
            offset=pagination.offset,
        )

        frameworks = []
        for f in result.get("frameworks", []):
            frameworks.append(
                FrameworkEntry(
                    framework_id=f.get("framework_id", ""),
                    name=f.get("name", ""),
                    jurisdiction=JurisdictionTypeEnum(f.get("jurisdiction", "eu")),
                    country_codes=f.get("country_codes", []),
                    status=FrameworkStatusEnum(f.get("status", "active")),
                    effective_date=f.get("effective_date"),
                    expiry_date=f.get("expiry_date"),
                    commodities=[EUDRCommodityEnum(c) for c in f.get("commodities", [])],
                )
            )

        total = result.get("total", len(frameworks))

        elapsed_ms = Decimal(str((time.monotonic() - start) * 1000))
        provenance_hash = _compute_provenance(
            f"framework_search:{body.query}:{body.jurisdiction}",
            str(total),
        )

        logger.info(
            "Framework search: query=%s results=%d user=%s",
            body.query,
            total,
            user.user_id,
        )

        return FrameworkSearchResponse(
            frameworks=frameworks,
            total_results=total,
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
                data_sources=["LegalFrameworkEngine"],
            ),
        )

    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Framework search failed: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Framework search failed",
        )
