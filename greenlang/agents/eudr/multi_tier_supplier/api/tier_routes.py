# -*- coding: utf-8 -*-
"""
Tier & Relationship Routes - AGENT-EUDR-008 Multi-Tier Supplier Tracker API

Endpoints for tier depth tracking, visibility scoring, gap detection,
and supplier relationship lifecycle management.

Tier Endpoints:
    GET  /tiers/{supplier_id}          - Get tier depth for supplier
    POST /tiers/assess                 - Assess tier depth for chain
    GET  /tiers/visibility             - Get visibility scores
    GET  /tiers/gaps                   - Get tier coverage gaps

Relationship Endpoints:
    POST /relationships                - Create relationship
    PUT  /relationships/{rel_id}       - Update relationship
    GET  /relationships/{supplier_id}  - Get supplier relationships
    POST /relationships/history        - Get relationship history

Author: GreenLang Platform Team
Date: March 2026
PRD: AGENT-EUDR-008 Multi-Tier Supplier Tracker (GL-EUDR-MST-008)
"""

from __future__ import annotations

import hashlib
import logging
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Path, Query, Request, status

from greenlang.agents.eudr.multi_tier_supplier.api.dependencies import (
    AuthUser,
    ErrorResponse,
    PaginationParams,
    get_pagination,
    get_supplier_service,
    rate_limit_standard,
    require_permission,
    validate_commodity,
)
from greenlang.agents.eudr.multi_tier_supplier.api.schemas import (
    CreateRelationshipSchema,
    RelationshipHistoryRequestSchema,
    RelationshipHistoryResponseSchema,
    RelationshipSchema,
    SupplierRelationshipsResponseSchema,
    TierDepthRequestSchema,
    TierDepthResponseSchema,
    TierGapsResponseSchema,
    UpdateRelationshipSchema,
    VisibilityScoreSchema,
)

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Tier Depth & Relationships"])


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _compute_provenance(data: str) -> str:
    """Compute SHA-256 provenance hash for audit trail.

    Args:
        data: String to hash.

    Returns:
        Hex-encoded SHA-256 hash.
    """
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


# ==========================================================================
# TIER DEPTH ENDPOINTS
# ==========================================================================


# ---------------------------------------------------------------------------
# GET /tiers/{supplier_id}
# ---------------------------------------------------------------------------


@router.get(
    "/tiers/{supplier_id}",
    response_model=TierDepthResponseSchema,
    status_code=status.HTTP_200_OK,
    summary="Get tier depth for a supplier",
    description=(
        "Retrieve the tier depth analysis for a specific supplier "
        "including per-tier statistics, visibility scores, and volume "
        "coverage metrics."
    ),
    responses={
        200: {"description": "Tier depth retrieved"},
        401: {"model": ErrorResponse, "description": "Authentication required"},
        403: {"model": ErrorResponse, "description": "Insufficient permissions"},
        404: {"model": ErrorResponse, "description": "Supplier not found"},
    },
)
async def get_tier_depth(
    supplier_id: str = Path(
        ..., min_length=1, max_length=100, description="Supplier identifier"
    ),
    commodity: Optional[str] = Depends(validate_commodity),
    request: Request = None,
    user: AuthUser = Depends(
        require_permission("eudr-mst:tiers:read")
    ),
    _rate: None = Depends(rate_limit_standard),
) -> TierDepthResponseSchema:
    """Get tier depth analysis for a supplier.

    Args:
        supplier_id: Supplier identifier.
        commodity: Optional EUDR commodity filter.
        request: FastAPI request object.
        user: Authenticated user with tiers:read permission.

    Returns:
        TierDepthResponseSchema with depth analysis.

    Raises:
        HTTPException: 404 if supplier not found.
    """
    start = time.monotonic()
    logger.info(
        "Get tier depth: user=%s supplier_id=%s commodity=%s",
        user.user_id,
        supplier_id,
        commodity,
    )

    try:
        service = get_supplier_service()
        result = service.get_tier_depth(
            supplier_id=supplier_id,
            commodity=commodity,
        )

        if result is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Supplier not found: {supplier_id}",
            )

        elapsed = time.monotonic() - start
        provenance = _compute_provenance(
            f"tier_depth|{supplier_id}|{commodity}|"
            f"{result.get('max_depth', 0)}|{elapsed}"
        )

        logger.info(
            "Tier depth retrieved: user=%s supplier_id=%s max_depth=%d "
            "elapsed_ms=%.1f",
            user.user_id,
            supplier_id,
            result.get("max_depth", 0),
            elapsed * 1000,
        )

        return TierDepthResponseSchema(
            root_supplier_id=supplier_id,
            commodity=commodity or "all",
            max_depth=result.get("max_depth", 0),
            total_suppliers=result.get("total_suppliers", 0),
            overall_visibility_score=result.get("overall_visibility_score", 0.0),
            volume_coverage_pct=result.get("volume_coverage_pct", 0.0),
            tier_summaries=result.get("tier_summaries", []),
            industry_avg_depth=result.get("industry_avg_depth"),
            elapsed_ms=elapsed * 1000,
            provenance_hash=provenance,
        )

    except HTTPException:
        raise
    except NotImplementedError as exc:
        raise HTTPException(
            status_code=status.HTTP_501_NOT_IMPLEMENTED,
            detail=str(exc),
        )
    except Exception as exc:
        logger.error(
            "Get tier depth failed: user=%s supplier_id=%s error=%s",
            user.user_id,
            supplier_id,
            exc,
            exc_info=True,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Tier depth retrieval failed due to an internal error",
        )


# ---------------------------------------------------------------------------
# POST /tiers/assess
# ---------------------------------------------------------------------------


@router.post(
    "/tiers/assess",
    response_model=TierDepthResponseSchema,
    status_code=status.HTTP_200_OK,
    summary="Assess tier depth for a supply chain",
    description=(
        "Perform a full tier depth assessment for a supply chain "
        "starting from a root supplier. Calculates visibility scores, "
        "coverage metrics, and per-tier statistics."
    ),
    responses={
        200: {"description": "Tier depth assessment completed"},
        400: {"model": ErrorResponse, "description": "Invalid input"},
        401: {"model": ErrorResponse, "description": "Authentication required"},
        403: {"model": ErrorResponse, "description": "Insufficient permissions"},
        429: {"model": ErrorResponse, "description": "Rate limit exceeded"},
    },
)
async def assess_tier_depth(
    body: TierDepthRequestSchema,
    request: Request,
    user: AuthUser = Depends(
        require_permission("eudr-mst:tiers:write")
    ),
    _rate: None = Depends(rate_limit_standard),
) -> TierDepthResponseSchema:
    """Assess tier depth for a supply chain.

    Args:
        body: Assessment request with root supplier and commodity.
        request: FastAPI request object.
        user: Authenticated user with tiers:write permission.

    Returns:
        TierDepthResponseSchema with assessment results.

    Raises:
        HTTPException: 400 on invalid input, 500 on internal error.
    """
    start = time.monotonic()
    logger.info(
        "Assess tier depth: user=%s root_supplier=%s commodity=%s",
        user.user_id,
        body.root_supplier_id,
        body.commodity,
    )

    try:
        service = get_supplier_service()

        result = service.assess_tier_depth(
            root_supplier_id=body.root_supplier_id,
            commodity=body.commodity,
            include_inactive=body.include_inactive,
        )

        elapsed = time.monotonic() - start
        provenance = _compute_provenance(
            f"assess_tier|{body.root_supplier_id}|{body.commodity}|"
            f"{result.get('max_depth', 0)}|{elapsed}"
        )

        logger.info(
            "Tier depth assessed: user=%s root=%s max_depth=%d "
            "total_suppliers=%d elapsed_ms=%.1f",
            user.user_id,
            body.root_supplier_id,
            result.get("max_depth", 0),
            result.get("total_suppliers", 0),
            elapsed * 1000,
        )

        return TierDepthResponseSchema(
            root_supplier_id=body.root_supplier_id,
            commodity=body.commodity,
            max_depth=result.get("max_depth", 0),
            total_suppliers=result.get("total_suppliers", 0),
            overall_visibility_score=result.get("overall_visibility_score", 0.0),
            volume_coverage_pct=result.get("volume_coverage_pct", 0.0),
            tier_summaries=result.get("tier_summaries", []),
            industry_avg_depth=result.get("industry_avg_depth"),
            elapsed_ms=elapsed * 1000,
            provenance_hash=provenance,
        )

    except ValueError as exc:
        logger.warning(
            "Assess tier depth validation error: user=%s error=%s",
            user.user_id,
            exc,
        )
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Tier depth assessment validation failed: {exc}",
        )
    except NotImplementedError as exc:
        raise HTTPException(
            status_code=status.HTTP_501_NOT_IMPLEMENTED,
            detail=str(exc),
        )
    except Exception as exc:
        logger.error(
            "Assess tier depth failed: user=%s error=%s",
            user.user_id,
            exc,
            exc_info=True,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Tier depth assessment failed due to an internal error",
        )


# ---------------------------------------------------------------------------
# GET /tiers/visibility
# ---------------------------------------------------------------------------


@router.get(
    "/tiers/visibility",
    response_model=VisibilityScoreSchema,
    status_code=status.HTTP_200_OK,
    summary="Get visibility scores",
    description=(
        "Retrieve supply chain visibility scores across all commodities "
        "and supplier chains. Provides an overall score and per-chain "
        "breakdowns."
    ),
    responses={
        200: {"description": "Visibility scores retrieved"},
        401: {"model": ErrorResponse, "description": "Authentication required"},
        403: {"model": ErrorResponse, "description": "Insufficient permissions"},
    },
)
async def get_visibility_scores(
    commodity: Optional[str] = Depends(validate_commodity),
    request: Request = None,
    user: AuthUser = Depends(
        require_permission("eudr-mst:tiers:read")
    ),
    _rate: None = Depends(rate_limit_standard),
) -> VisibilityScoreSchema:
    """Get supply chain visibility scores.

    Args:
        commodity: Optional EUDR commodity filter.
        request: FastAPI request object.
        user: Authenticated user with tiers:read permission.

    Returns:
        VisibilityScoreSchema with overall and per-chain scores.

    Raises:
        HTTPException: 500 on internal error.
    """
    start = time.monotonic()
    logger.info(
        "Get visibility scores: user=%s commodity=%s",
        user.user_id,
        commodity,
    )

    try:
        service = get_supplier_service()
        result = service.get_visibility_scores(
            commodity=commodity,
            tenant_id=user.tenant_id,
        )

        elapsed = time.monotonic() - start
        provenance = _compute_provenance(
            f"visibility|{commodity}|{result.get('overall_score', 0)}|{elapsed}"
        )

        logger.info(
            "Visibility scores retrieved: user=%s overall=%.1f elapsed_ms=%.1f",
            user.user_id,
            result.get("overall_score", 0),
            elapsed * 1000,
        )

        return VisibilityScoreSchema(
            scores=result.get("scores", []),
            overall_score=result.get("overall_score", 0.0),
            provenance_hash=provenance,
        )

    except NotImplementedError as exc:
        raise HTTPException(
            status_code=status.HTTP_501_NOT_IMPLEMENTED,
            detail=str(exc),
        )
    except Exception as exc:
        logger.error(
            "Get visibility scores failed: user=%s error=%s",
            user.user_id,
            exc,
            exc_info=True,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Visibility score retrieval failed due to an internal error",
        )


# ---------------------------------------------------------------------------
# GET /tiers/gaps
# ---------------------------------------------------------------------------


@router.get(
    "/tiers/gaps",
    response_model=TierGapsResponseSchema,
    status_code=status.HTTP_200_OK,
    summary="Get tier coverage gaps",
    description=(
        "Identify tier coverage gaps across the supply chain including "
        "missing tiers, low visibility levels, and incomplete profiles. "
        "Gaps are classified by severity: critical, major, minor."
    ),
    responses={
        200: {"description": "Tier gaps retrieved"},
        401: {"model": ErrorResponse, "description": "Authentication required"},
        403: {"model": ErrorResponse, "description": "Insufficient permissions"},
    },
)
async def get_tier_gaps(
    commodity: Optional[str] = Depends(validate_commodity),
    request: Request = None,
    user: AuthUser = Depends(
        require_permission("eudr-mst:tiers:read")
    ),
    _rate: None = Depends(rate_limit_standard),
) -> TierGapsResponseSchema:
    """Get tier coverage gap analysis.

    Args:
        commodity: Optional EUDR commodity filter.
        request: FastAPI request object.
        user: Authenticated user with tiers:read permission.

    Returns:
        TierGapsResponseSchema with gap details by severity.

    Raises:
        HTTPException: 500 on internal error.
    """
    start = time.monotonic()
    logger.info(
        "Get tier gaps: user=%s commodity=%s",
        user.user_id,
        commodity,
    )

    try:
        service = get_supplier_service()
        result = service.get_tier_gaps(
            commodity=commodity,
            tenant_id=user.tenant_id,
        )

        elapsed = time.monotonic() - start
        gaps = result.get("gaps", [])
        critical = sum(1 for g in gaps if g.get("severity") == "critical")
        major = sum(1 for g in gaps if g.get("severity") == "major")
        minor = sum(1 for g in gaps if g.get("severity") == "minor")

        provenance = _compute_provenance(
            f"gaps|{commodity}|{len(gaps)}|{critical}|{elapsed}"
        )

        logger.info(
            "Tier gaps retrieved: user=%s total=%d critical=%d "
            "major=%d minor=%d elapsed_ms=%.1f",
            user.user_id,
            len(gaps),
            critical,
            major,
            minor,
            elapsed * 1000,
        )

        return TierGapsResponseSchema(
            total_gaps=len(gaps),
            critical_gaps=critical,
            major_gaps=major,
            minor_gaps=minor,
            gaps=gaps,
            elapsed_ms=elapsed * 1000,
            provenance_hash=provenance,
        )

    except NotImplementedError as exc:
        raise HTTPException(
            status_code=status.HTTP_501_NOT_IMPLEMENTED,
            detail=str(exc),
        )
    except Exception as exc:
        logger.error(
            "Get tier gaps failed: user=%s error=%s",
            user.user_id,
            exc,
            exc_info=True,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Tier gap analysis failed due to an internal error",
        )


# ==========================================================================
# RELATIONSHIP ENDPOINTS
# ==========================================================================


# ---------------------------------------------------------------------------
# POST /relationships
# ---------------------------------------------------------------------------


@router.post(
    "/relationships",
    response_model=RelationshipSchema,
    status_code=status.HTTP_201_CREATED,
    summary="Create supplier relationship",
    description=(
        "Create a new supplier-to-supplier relationship in the supply "
        "chain hierarchy. Defines the parent-child link, commodity, "
        "volume, frequency, and confidence level."
    ),
    responses={
        201: {"description": "Relationship created"},
        400: {"model": ErrorResponse, "description": "Invalid input"},
        401: {"model": ErrorResponse, "description": "Authentication required"},
        403: {"model": ErrorResponse, "description": "Insufficient permissions"},
        409: {"model": ErrorResponse, "description": "Relationship already exists"},
        429: {"model": ErrorResponse, "description": "Rate limit exceeded"},
    },
)
async def create_relationship(
    body: CreateRelationshipSchema,
    request: Request,
    user: AuthUser = Depends(
        require_permission("eudr-mst:relationships:write")
    ),
    _rate: None = Depends(rate_limit_standard),
) -> RelationshipSchema:
    """Create a supplier relationship.

    Args:
        body: Relationship data with parent/child IDs and attributes.
        request: FastAPI request object.
        user: Authenticated user with relationships:write permission.

    Returns:
        Created RelationshipSchema.

    Raises:
        HTTPException: 400 on validation error, 409 on duplicate.
    """
    start = time.monotonic()
    logger.info(
        "Create relationship: user=%s parent=%s child=%s commodity=%s",
        user.user_id,
        body.parent_supplier_id,
        body.child_supplier_id,
        body.commodity,
    )

    try:
        service = get_supplier_service()

        result = service.create_relationship(
            data=body.model_dump(),
            created_by=user.user_id,
        )

        elapsed = time.monotonic() - start
        rel_id = result.get("relationship_id", str(uuid.uuid4()))
        provenance = _compute_provenance(
            f"rel_create|{rel_id}|{body.parent_supplier_id}|"
            f"{body.child_supplier_id}|{elapsed}"
        )

        logger.info(
            "Relationship created: user=%s rel_id=%s elapsed_ms=%.1f",
            user.user_id,
            rel_id,
            elapsed * 1000,
        )

        return RelationshipSchema(
            relationship_id=rel_id,
            parent_supplier_id=body.parent_supplier_id,
            child_supplier_id=body.child_supplier_id,
            commodity=body.commodity,
            relationship_state=body.relationship_state,
            volume_tonnes=body.volume_tonnes,
            frequency=body.frequency,
            is_exclusive=body.is_exclusive,
            start_date=body.start_date,
            end_date=body.end_date,
            confidence=body.confidence,
            strength_score=result.get("strength_score"),
            metadata=body.metadata,
            provenance_hash=provenance,
        )

    except ValueError as exc:
        logger.warning(
            "Create relationship validation error: user=%s error=%s",
            user.user_id,
            exc,
        )
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Relationship creation validation failed: {exc}",
        )
    except NotImplementedError as exc:
        raise HTTPException(
            status_code=status.HTTP_501_NOT_IMPLEMENTED,
            detail=str(exc),
        )
    except Exception as exc:
        logger.error(
            "Create relationship failed: user=%s error=%s",
            user.user_id,
            exc,
            exc_info=True,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Relationship creation failed due to an internal error",
        )


# ---------------------------------------------------------------------------
# PUT /relationships/{rel_id}
# ---------------------------------------------------------------------------


@router.put(
    "/relationships/{rel_id}",
    response_model=RelationshipSchema,
    status_code=status.HTTP_200_OK,
    summary="Update supplier relationship",
    description=(
        "Update an existing supplier relationship. Supports state "
        "transitions (active -> suspended -> terminated), volume "
        "changes, and confidence updates. All changes are audited."
    ),
    responses={
        200: {"description": "Relationship updated"},
        400: {"model": ErrorResponse, "description": "Invalid input"},
        401: {"model": ErrorResponse, "description": "Authentication required"},
        403: {"model": ErrorResponse, "description": "Insufficient permissions"},
        404: {"model": ErrorResponse, "description": "Relationship not found"},
        429: {"model": ErrorResponse, "description": "Rate limit exceeded"},
    },
)
async def update_relationship(
    body: UpdateRelationshipSchema,
    rel_id: str = Path(
        ..., min_length=1, max_length=100, description="Relationship identifier"
    ),
    request: Request = None,
    user: AuthUser = Depends(
        require_permission("eudr-mst:relationships:write")
    ),
    _rate: None = Depends(rate_limit_standard),
) -> RelationshipSchema:
    """Update an existing supplier relationship.

    Args:
        body: Partial update data with optional reason.
        rel_id: Relationship identifier to update.
        request: FastAPI request object.
        user: Authenticated user with relationships:write permission.

    Returns:
        Updated RelationshipSchema.

    Raises:
        HTTPException: 404 if relationship not found.
    """
    start = time.monotonic()
    update_fields = body.model_dump(exclude_none=True)
    logger.info(
        "Update relationship: user=%s rel_id=%s fields=%s",
        user.user_id,
        rel_id,
        list(update_fields.keys()),
    )

    try:
        service = get_supplier_service()

        result = service.update_relationship(
            relationship_id=rel_id,
            updates=update_fields,
            updated_by=user.user_id,
        )

        if result is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Relationship not found: {rel_id}",
            )

        elapsed = time.monotonic() - start
        provenance = _compute_provenance(
            f"rel_update|{rel_id}|{elapsed}"
        )

        logger.info(
            "Relationship updated: user=%s rel_id=%s elapsed_ms=%.1f",
            user.user_id,
            rel_id,
            elapsed * 1000,
        )

        result["provenance_hash"] = provenance
        return RelationshipSchema(**result)

    except HTTPException:
        raise
    except ValueError as exc:
        logger.warning(
            "Update relationship validation error: user=%s error=%s",
            user.user_id,
            exc,
        )
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Relationship update validation failed: {exc}",
        )
    except NotImplementedError as exc:
        raise HTTPException(
            status_code=status.HTTP_501_NOT_IMPLEMENTED,
            detail=str(exc),
        )
    except Exception as exc:
        logger.error(
            "Update relationship failed: user=%s rel_id=%s error=%s",
            user.user_id,
            rel_id,
            exc,
            exc_info=True,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Relationship update failed due to an internal error",
        )


# ---------------------------------------------------------------------------
# GET /relationships/{supplier_id}
# ---------------------------------------------------------------------------


@router.get(
    "/relationships/{supplier_id}",
    response_model=SupplierRelationshipsResponseSchema,
    status_code=status.HTTP_200_OK,
    summary="Get supplier relationships",
    description=(
        "Retrieve all upstream (parent) and downstream (child) "
        "relationships for a specific supplier. Includes relationship "
        "state, volume, and strength scores."
    ),
    responses={
        200: {"description": "Relationships retrieved"},
        401: {"model": ErrorResponse, "description": "Authentication required"},
        403: {"model": ErrorResponse, "description": "Insufficient permissions"},
        404: {"model": ErrorResponse, "description": "Supplier not found"},
    },
)
async def get_supplier_relationships(
    supplier_id: str = Path(
        ..., min_length=1, max_length=100, description="Supplier identifier"
    ),
    request: Request = None,
    user: AuthUser = Depends(
        require_permission("eudr-mst:relationships:read")
    ),
    _rate: None = Depends(rate_limit_standard),
) -> SupplierRelationshipsResponseSchema:
    """Get all relationships for a supplier.

    Args:
        supplier_id: Supplier identifier.
        request: FastAPI request object.
        user: Authenticated user with relationships:read permission.

    Returns:
        SupplierRelationshipsResponseSchema with upstream and downstream.

    Raises:
        HTTPException: 404 if supplier not found.
    """
    start = time.monotonic()
    logger.info(
        "Get relationships: user=%s supplier_id=%s",
        user.user_id,
        supplier_id,
    )

    try:
        service = get_supplier_service()
        result = service.get_supplier_relationships(
            supplier_id=supplier_id,
        )

        if result is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Supplier not found: {supplier_id}",
            )

        elapsed = time.monotonic() - start
        upstream = result.get("upstream", [])
        downstream = result.get("downstream", [])
        total = len(upstream) + len(downstream)

        provenance = _compute_provenance(
            f"relationships|{supplier_id}|{total}|{elapsed}"
        )

        logger.info(
            "Relationships retrieved: user=%s supplier_id=%s "
            "upstream=%d downstream=%d elapsed_ms=%.1f",
            user.user_id,
            supplier_id,
            len(upstream),
            len(downstream),
            elapsed * 1000,
        )

        return SupplierRelationshipsResponseSchema(
            supplier_id=supplier_id,
            upstream=upstream,
            downstream=downstream,
            total_relationships=total,
            elapsed_ms=elapsed * 1000,
            provenance_hash=provenance,
        )

    except HTTPException:
        raise
    except NotImplementedError as exc:
        raise HTTPException(
            status_code=status.HTTP_501_NOT_IMPLEMENTED,
            detail=str(exc),
        )
    except Exception as exc:
        logger.error(
            "Get relationships failed: user=%s supplier_id=%s error=%s",
            user.user_id,
            supplier_id,
            exc,
            exc_info=True,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Relationship retrieval failed due to an internal error",
        )


# ---------------------------------------------------------------------------
# POST /relationships/history
# ---------------------------------------------------------------------------


@router.post(
    "/relationships/history",
    response_model=RelationshipHistoryResponseSchema,
    status_code=status.HTTP_200_OK,
    summary="Get relationship change history",
    description=(
        "Retrieve the change history for supplier relationships. "
        "Includes all state transitions, volume changes, and metadata "
        "updates with actor and timestamp for audit compliance."
    ),
    responses={
        200: {"description": "Relationship history retrieved"},
        400: {"model": ErrorResponse, "description": "Invalid input"},
        401: {"model": ErrorResponse, "description": "Authentication required"},
        403: {"model": ErrorResponse, "description": "Insufficient permissions"},
        429: {"model": ErrorResponse, "description": "Rate limit exceeded"},
    },
)
async def get_relationship_history(
    body: RelationshipHistoryRequestSchema,
    request: Request,
    user: AuthUser = Depends(
        require_permission("eudr-mst:relationships:read")
    ),
    _rate: None = Depends(rate_limit_standard),
) -> RelationshipHistoryResponseSchema:
    """Get relationship change history.

    Args:
        body: History request with supplier ID and optional filters.
        request: FastAPI request object.
        user: Authenticated user with relationships:read permission.

    Returns:
        RelationshipHistoryResponseSchema with change entries.

    Raises:
        HTTPException: 400 on invalid input, 500 on internal error.
    """
    start = time.monotonic()
    logger.info(
        "Get relationship history: user=%s supplier_id=%s rel_id=%s",
        user.user_id,
        body.supplier_id,
        body.relationship_id,
    )

    try:
        service = get_supplier_service()

        result = service.get_relationship_history(
            supplier_id=body.supplier_id,
            relationship_id=body.relationship_id,
            start_date=body.start_date,
            end_date=body.end_date,
            limit=body.limit,
            offset=body.offset,
        )

        elapsed = time.monotonic() - start
        changes = result.get("changes", [])
        total = result.get("total_changes", len(changes))

        provenance = _compute_provenance(
            f"rel_history|{body.supplier_id}|{total}|{elapsed}"
        )

        logger.info(
            "Relationship history retrieved: user=%s supplier_id=%s "
            "total=%d returned=%d elapsed_ms=%.1f",
            user.user_id,
            body.supplier_id,
            total,
            len(changes),
            elapsed * 1000,
        )

        return RelationshipHistoryResponseSchema(
            supplier_id=body.supplier_id,
            total_changes=total,
            changes=changes,
            limit=body.limit,
            offset=body.offset,
            has_more=(body.offset + body.limit) < total,
            elapsed_ms=elapsed * 1000,
            provenance_hash=provenance,
        )

    except ValueError as exc:
        logger.warning(
            "Relationship history validation error: user=%s error=%s",
            user.user_id,
            exc,
        )
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Relationship history query validation failed: {exc}",
        )
    except NotImplementedError as exc:
        raise HTTPException(
            status_code=status.HTTP_501_NOT_IMPLEMENTED,
            detail=str(exc),
        )
    except Exception as exc:
        logger.error(
            "Get relationship history failed: user=%s error=%s",
            user.user_id,
            exc,
            exc_info=True,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Relationship history retrieval failed due to an internal error",
        )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

__all__ = ["router"]
