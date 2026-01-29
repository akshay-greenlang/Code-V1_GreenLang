"""
FastAPI route handlers for Review Console API.

This module provides all REST API endpoints for the review console,
including queue management, resolution processing, and statistics.

Endpoints:
    - GET /api/queue: List review queue items with filtering
    - GET /api/queue/{item_id}: Get single item details
    - POST /api/queue/{item_id}/resolve: Resolve an item
    - POST /api/queue/{item_id}/reject: Reject/escalate an item
    - GET /api/stats: Dashboard statistics
    - POST /api/vocabulary/suggest: Suggest new vocabulary entry
    - GET /api/health: Health check

Example:
    >>> from review_console.api.routes import router
    >>> app.include_router(router, prefix="/api")
"""

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
import math

from fastapi import APIRouter, Depends, HTTPException, Query, Request, status
from fastapi.responses import JSONResponse
from sqlalchemy.ext.asyncio import AsyncSession
import structlog

from review_console.config import get_settings
from review_console.db.session import get_db, check_database_health
from review_console.db.models import ReviewStatus, AuditAction
from review_console.services.queue import ReviewQueueService
from review_console.services.resolution import ResolutionService, ResolutionError
from review_console.api.models import (
    # Request models
    ResolveItemRequest,
    RejectItemRequest,
    VocabularySuggestionRequest,
    QueueFilterParams,
    # Response models
    ReviewQueueItemResponse,
    ReviewQueueItemDetailResponse,
    ResolutionResponse,
    QueueStatsResponse,
    VocabularySuggestionResponse,
    PaginatedResponse,
    HealthResponse,
    ErrorResponse,
    CandidateInfo,
    ContextInfo,
    ReviewStatusEnum,
    EntityTypeEnum,
)
from review_console.api.auth import get_current_user, User

logger = structlog.get_logger()
settings = get_settings()

router = APIRouter(tags=["Review Queue"])


# ============================================================================
# Helper Functions
# ============================================================================


def get_client_ip(request: Request) -> Optional[str]:
    """Extract client IP from request, handling proxies."""
    forwarded = request.headers.get("X-Forwarded-For")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else None


def get_user_agent(request: Request) -> Optional[str]:
    """Extract user agent from request."""
    return request.headers.get("User-Agent")


# ============================================================================
# Queue Endpoints
# ============================================================================


@router.get(
    "/queue",
    response_model=PaginatedResponse[ReviewQueueItemResponse],
    summary="List review queue items",
    description="""
    Retrieve a paginated list of review queue items with optional filtering.

    Supports filtering by:
    - entity_type: Filter by entity type (fuel, material, process)
    - org_id: Filter by organization ID
    - status: Filter by review status
    - date_from/date_to: Filter by creation date range
    - min_confidence/max_confidence: Filter by confidence score
    - assigned_to: Filter by assigned reviewer
    - search: Search in input_text
    """,
    responses={
        200: {"description": "Paginated list of review queue items"},
        401: {"model": ErrorResponse, "description": "Unauthorized"},
        500: {"model": ErrorResponse, "description": "Internal server error"},
    },
)
async def get_queue_items(
    request: Request,
    entity_type: Optional[EntityTypeEnum] = Query(None, description="Filter by entity type"),
    org_id: Optional[str] = Query(None, description="Filter by organization ID"),
    status: Optional[ReviewStatusEnum] = Query(None, description="Filter by status"),
    date_from: Optional[datetime] = Query(None, description="Filter by created date (from)"),
    date_to: Optional[datetime] = Query(None, description="Filter by created date (to)"),
    min_confidence: Optional[float] = Query(None, ge=0.0, le=1.0, description="Min confidence"),
    max_confidence: Optional[float] = Query(None, ge=0.0, le=1.0, description="Max confidence"),
    assigned_to: Optional[str] = Query(None, description="Filter by assigned reviewer"),
    search: Optional[str] = Query(None, max_length=200, description="Search in input_text"),
    page: int = Query(1, ge=1, description="Page number (1-indexed)"),
    page_size: int = Query(20, ge=1, le=100, description="Items per page"),
    order_by: str = Query("created_at", description="Field to order by"),
    order_desc: bool = Query(True, description="Order descending"),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> PaginatedResponse[ReviewQueueItemResponse]:
    """
    List review queue items with filtering and pagination.

    Args:
        entity_type: Filter by entity type (fuel, material, process)
        org_id: Filter by organization ID
        status: Filter by review status
        date_from: Filter by created date (from)
        date_to: Filter by created date (to)
        min_confidence: Filter by minimum confidence score
        max_confidence: Filter by maximum confidence score
        assigned_to: Filter by assigned reviewer
        search: Search in input_text field
        page: Page number (1-indexed)
        page_size: Number of items per page
        order_by: Field to order by
        order_desc: Whether to order descending
        db: Database session
        current_user: Authenticated user

    Returns:
        Paginated list of review queue items
    """
    # Build filter params
    filters = QueueFilterParams(
        entity_type=entity_type,
        org_id=org_id,
        status=status,
        date_from=date_from,
        date_to=date_to,
        min_confidence=min_confidence,
        max_confidence=max_confidence,
        assigned_to=assigned_to,
        search=search,
    )

    # Get items
    service = ReviewQueueService(db)
    items, total = await service.get_items(
        filters=filters,
        page=page,
        page_size=page_size,
        order_by=order_by,
        order_desc=order_desc,
    )

    # Convert to response models
    response_items = [
        ReviewQueueItemResponse(
            id=str(item.id),
            input_text=item.input_text,
            entity_type=item.entity_type,
            org_id=item.org_id,
            top_candidate_id=item.top_candidate_id,
            top_candidate_name=item.top_candidate_name,
            confidence=item.confidence,
            match_method=item.match_method,
            status=ReviewStatusEnum(item.status.value),
            priority=item.priority,
            assigned_to=item.assigned_to,
            created_at=item.created_at,
            updated_at=item.updated_at,
        )
        for item in items
    ]

    # Calculate pagination info
    total_pages = math.ceil(total / page_size) if total > 0 else 1

    return PaginatedResponse(
        items=response_items,
        total=total,
        page=page,
        page_size=page_size,
        total_pages=total_pages,
        has_next=page < total_pages,
        has_prev=page > 1,
    )


@router.get(
    "/queue/{item_id}",
    response_model=ReviewQueueItemDetailResponse,
    summary="Get review queue item details",
    description="""
    Retrieve detailed information about a single review queue item,
    including all candidates, context, and resolution (if resolved).
    """,
    responses={
        200: {"description": "Review queue item details"},
        401: {"model": ErrorResponse, "description": "Unauthorized"},
        404: {"model": ErrorResponse, "description": "Item not found"},
        500: {"model": ErrorResponse, "description": "Internal server error"},
    },
)
async def get_queue_item(
    request: Request,
    item_id: str,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> ReviewQueueItemDetailResponse:
    """
    Get detailed information about a single review queue item.

    Args:
        item_id: UUID of the review queue item
        db: Database session
        current_user: Authenticated user

    Returns:
        Detailed review queue item information

    Raises:
        HTTPException: 404 if item not found
    """
    service = ReviewQueueService(db)
    item = await service.get_item_by_id(item_id, include_resolution=True)

    if not item:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"error": "ITEM_NOT_FOUND", "message": f"Item {item_id} not found"},
        )

    # Create audit entry for view
    await service.create_audit_entry(
        item_id=item_id,
        action=AuditAction.ITEM_VIEWED,
        actor_id=current_user.id,
        actor_email=current_user.email,
        ip_address=get_client_ip(request),
        user_agent=get_user_agent(request),
    )

    # Parse candidates from JSON
    candidates_data = item.candidates if isinstance(item.candidates, list) else []
    candidates = [
        CandidateInfo(
            id=c.get("id", ""),
            name=c.get("name", ""),
            score=c.get("score", 0.0),
            source=c.get("source", "unknown"),
            match_method=c.get("match_method"),
            metadata=c.get("metadata", {}),
        )
        for c in candidates_data
    ]

    # Parse context from JSON
    context_data = item.context if isinstance(item.context, dict) else {}
    context = ContextInfo(
        industry_sector=context_data.get("industry_sector"),
        region=context_data.get("region"),
        temporal_scope=context_data.get("temporal_scope"),
        source_field=context_data.get("source_field"),
        additional_hints=context_data.get("additional_hints", {}),
    )

    # Build resolution response if exists
    resolution_response = None
    if item.resolution:
        resolution_response = ResolutionResponse(
            id=str(item.resolution.id),
            item_id=str(item.resolution.item_id),
            canonical_id=item.resolution.canonical_id,
            canonical_name=item.resolution.canonical_name,
            reviewer_id=item.resolution.reviewer_id,
            reviewer_email=item.resolution.reviewer_email,
            notes=item.resolution.notes,
            confidence_override=item.resolution.confidence_override,
            created_at=item.resolution.created_at,
        )

    return ReviewQueueItemDetailResponse(
        id=str(item.id),
        input_text=item.input_text,
        entity_type=item.entity_type,
        org_id=item.org_id,
        source_record_id=item.source_record_id,
        pipeline_id=item.pipeline_id,
        top_candidate_id=item.top_candidate_id,
        top_candidate_name=item.top_candidate_name,
        confidence=item.confidence,
        match_method=item.match_method,
        status=ReviewStatusEnum(item.status.value),
        priority=item.priority,
        assigned_to=item.assigned_to,
        candidates=candidates,
        context=context,
        vocabulary_version=item.vocabulary_version,
        created_at=item.created_at,
        updated_at=item.updated_at,
        resolved_at=item.resolved_at,
        resolved_by=item.resolved_by,
        resolution=resolution_response,
    )


@router.post(
    "/queue/{item_id}/resolve",
    response_model=ResolutionResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Resolve a review queue item",
    description="""
    Resolve a review queue item by selecting a canonical entity.

    Creates a resolution record and updates the item status to RESOLVED.
    An audit trail entry is automatically created.
    """,
    responses={
        201: {"description": "Item resolved successfully"},
        400: {"model": ErrorResponse, "description": "Invalid request"},
        401: {"model": ErrorResponse, "description": "Unauthorized"},
        404: {"model": ErrorResponse, "description": "Item not found"},
        409: {"model": ErrorResponse, "description": "Item already resolved"},
        500: {"model": ErrorResponse, "description": "Internal server error"},
    },
)
async def resolve_item(
    request: Request,
    item_id: str,
    resolve_request: ResolveItemRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> ResolutionResponse:
    """
    Resolve a review queue item with a selected canonical entity.

    Args:
        item_id: UUID of the review queue item
        resolve_request: Resolution details
        db: Database session
        current_user: Authenticated user

    Returns:
        Created resolution record

    Raises:
        HTTPException: 404 if item not found, 409 if already resolved
    """
    service = ResolutionService(db)

    try:
        resolution = await service.resolve_item(
            item_id=item_id,
            canonical_id=resolve_request.selected_canonical_id,
            canonical_name=resolve_request.selected_canonical_name or resolve_request.selected_canonical_id,
            reviewer_id=current_user.id,
            reviewer_email=current_user.email,
            notes=resolve_request.reviewer_notes,
            confidence_override=resolve_request.confidence_override,
            ip_address=get_client_ip(request),
            user_agent=get_user_agent(request),
        )

        return ResolutionResponse(
            id=str(resolution.id),
            item_id=str(resolution.item_id),
            canonical_id=resolution.canonical_id,
            canonical_name=resolution.canonical_name,
            reviewer_id=resolution.reviewer_id,
            reviewer_email=resolution.reviewer_email,
            notes=resolution.notes,
            confidence_override=resolution.confidence_override,
            created_at=resolution.created_at,
        )

    except ResolutionError as e:
        if e.code == "ITEM_NOT_FOUND":
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail={"error": e.code, "message": e.message},
            )
        elif e.code == "ITEM_ALREADY_RESOLVED":
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail={"error": e.code, "message": e.message},
            )
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={"error": e.code, "message": e.message},
            )


@router.post(
    "/queue/{item_id}/reject",
    response_model=ReviewQueueItemResponse,
    summary="Reject or escalate a review queue item",
    description="""
    Reject a review queue item or escalate it to a senior reviewer.

    If escalate_to is provided, the item is escalated to that user.
    Otherwise, the item is marked as rejected.
    An audit trail entry is automatically created.
    """,
    responses={
        200: {"description": "Item rejected/escalated successfully"},
        400: {"model": ErrorResponse, "description": "Invalid request"},
        401: {"model": ErrorResponse, "description": "Unauthorized"},
        404: {"model": ErrorResponse, "description": "Item not found"},
        409: {"model": ErrorResponse, "description": "Item already closed"},
        500: {"model": ErrorResponse, "description": "Internal server error"},
    },
)
async def reject_item(
    request: Request,
    item_id: str,
    reject_request: RejectItemRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> ReviewQueueItemResponse:
    """
    Reject or escalate a review queue item.

    Args:
        item_id: UUID of the review queue item
        reject_request: Rejection details
        db: Database session
        current_user: Authenticated user

    Returns:
        Updated review queue item

    Raises:
        HTTPException: 404 if item not found, 409 if already closed
    """
    service = ResolutionService(db)

    try:
        item = await service.reject_item(
            item_id=item_id,
            reason=reject_request.reason,
            reviewer_id=current_user.id,
            reviewer_email=current_user.email,
            escalate_to=reject_request.escalate_to,
            ip_address=get_client_ip(request),
            user_agent=get_user_agent(request),
        )

        return ReviewQueueItemResponse(
            id=str(item.id),
            input_text=item.input_text,
            entity_type=item.entity_type,
            org_id=item.org_id,
            top_candidate_id=item.top_candidate_id,
            top_candidate_name=item.top_candidate_name,
            confidence=item.confidence,
            match_method=item.match_method,
            status=ReviewStatusEnum(item.status.value),
            priority=item.priority,
            assigned_to=item.assigned_to,
            created_at=item.created_at,
            updated_at=item.updated_at,
        )

    except ResolutionError as e:
        if e.code == "ITEM_NOT_FOUND":
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail={"error": e.code, "message": e.message},
            )
        elif e.code == "ITEM_ALREADY_CLOSED":
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail={"error": e.code, "message": e.message},
            )
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={"error": e.code, "message": e.message},
            )


# ============================================================================
# Statistics Endpoints
# ============================================================================


@router.get(
    "/stats",
    response_model=QueueStatsResponse,
    summary="Get dashboard statistics",
    description="""
    Retrieve aggregate statistics for the review queue dashboard.

    Includes:
    - Pending/in-progress/resolved counts
    - Average resolution time
    - Breakdown by entity type and organization
    """,
    responses={
        200: {"description": "Dashboard statistics"},
        401: {"model": ErrorResponse, "description": "Unauthorized"},
        500: {"model": ErrorResponse, "description": "Internal server error"},
    },
)
async def get_stats(
    request: Request,
    org_id: Optional[str] = Query(None, description="Filter stats by organization"),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> QueueStatsResponse:
    """
    Get dashboard statistics for the review queue.

    Args:
        org_id: Optional organization ID to filter stats
        db: Database session
        current_user: Authenticated user

    Returns:
        Queue statistics
    """
    service = ReviewQueueService(db)
    stats = await service.get_stats(org_id=org_id)

    return QueueStatsResponse(**stats)


# ============================================================================
# Vocabulary Suggestion Endpoints
# ============================================================================


@router.post(
    "/vocabulary/suggest",
    response_model=VocabularySuggestionResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Suggest a new vocabulary entry",
    description="""
    Suggest a new vocabulary entry when no suitable canonical match exists.

    Creates a vocabulary suggestion for governance review.
    Optionally creates a GitHub PR if configured.
    """,
    responses={
        201: {"description": "Suggestion created successfully"},
        400: {"model": ErrorResponse, "description": "Invalid request"},
        401: {"model": ErrorResponse, "description": "Unauthorized"},
        500: {"model": ErrorResponse, "description": "Internal server error"},
    },
)
async def suggest_vocabulary(
    request: Request,
    suggestion_request: VocabularySuggestionRequest,
    org_id: str = Query(..., description="Organization ID"),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> VocabularySuggestionResponse:
    """
    Suggest a new vocabulary entry for governance review.

    Args:
        suggestion_request: Vocabulary suggestion details
        org_id: Organization ID
        db: Database session
        current_user: Authenticated user

    Returns:
        Created vocabulary suggestion
    """
    service = ResolutionService(db)

    suggestion = await service.create_vocabulary_suggestion(
        entity_type=suggestion_request.entity_type.value,
        canonical_name=suggestion_request.canonical_name,
        aliases=suggestion_request.aliases,
        source=suggestion_request.source,
        suggested_by=current_user.id,
        suggested_by_email=current_user.email,
        org_id=org_id,
        properties=suggestion_request.properties,
        ip_address=get_client_ip(request),
        user_agent=get_user_agent(request),
    )

    return VocabularySuggestionResponse(
        id=str(suggestion.id),
        entity_type=suggestion.entity_type,
        canonical_name=suggestion.canonical_name,
        aliases=suggestion.aliases,
        source=suggestion.source,
        status=suggestion.status,
        pr_url=suggestion.pr_url,
        pr_number=suggestion.pr_number,
        suggested_by=suggestion.suggested_by,
        created_at=suggestion.created_at,
    )


# ============================================================================
# Health Endpoints
# ============================================================================


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Health check",
    description="Check API health status and dependency connectivity.",
    responses={
        200: {"description": "Service is healthy"},
        503: {"model": ErrorResponse, "description": "Service is unhealthy"},
    },
)
async def health_check(request: Request) -> HealthResponse:
    """
    Check API health status.

    Verifies connectivity to all dependencies (database, redis).

    Returns:
        Health status with dependency checks
    """
    from review_console import __version__

    # Check database
    db_healthy = await check_database_health()

    # Check redis (placeholder - implement based on your redis client)
    redis_healthy = True  # TODO: Implement actual redis health check

    checks = {
        "database": db_healthy,
        "redis": redis_healthy,
    }

    all_healthy = all(checks.values())

    response = HealthResponse(
        status="healthy" if all_healthy else "unhealthy",
        timestamp=datetime.now(timezone.utc),
        version=__version__,
        checks=checks,
    )

    if not all_healthy:
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content=response.model_dump(mode="json"),
        )

    return response
