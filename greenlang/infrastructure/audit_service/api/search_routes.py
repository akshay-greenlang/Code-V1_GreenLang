# -*- coding: utf-8 -*-
"""
Audit Search REST API Routes - SEC-005

FastAPI router for advanced audit event search:

    POST /api/v1/audit/search - Advanced search with aggregations

Supports LogQL-like query syntax:
    - field:value (exact match)
    - field:"quoted value" (exact match with spaces)
    - field:value* (prefix match)
    - AND, OR operators

Author: GreenLang Framework Team
Date: February 2026
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional
from uuid import UUID

logger = logging.getLogger(__name__)

try:
    from fastapi import APIRouter, Depends, HTTPException, Query, Request, status
    from pydantic import BaseModel, ConfigDict, Field

    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False
    APIRouter = object  # type: ignore[misc, assignment]
    Depends = None  # type: ignore[assignment]
    HTTPException = Exception  # type: ignore[misc, assignment]
    Query = None  # type: ignore[assignment]
    Request = None  # type: ignore[assignment]
    status = None  # type: ignore[assignment]
    BaseModel = object  # type: ignore[misc, assignment]
    ConfigDict = None  # type: ignore[assignment]
    Field = None  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Request/Response Models
# ---------------------------------------------------------------------------

if FASTAPI_AVAILABLE:

    class TimeRangeRequest(BaseModel):
        """Time range for search queries."""

        start: datetime = Field(..., description="Start of time range (inclusive)")
        end: datetime = Field(..., description="End of time range (exclusive)")

    class SearchRequest(BaseModel):
        """Advanced search request."""

        model_config = ConfigDict(
            json_schema_extra={
                "examples": [
                    {
                        "query": 'event_type:login AND severity:error',
                        "time_range": {
                            "start": "2026-02-01T00:00:00Z",
                            "end": "2026-02-06T23:59:59Z",
                        },
                        "aggregations": ["category", "severity"],
                    }
                ]
            }
        )

        query: Optional[str] = Field(
            None,
            description="LogQL-like query: field:value AND/OR field:value. "
                        "Supported fields: event_type, category, severity, user_email, "
                        "user_id, resource_type, outcome, ip_address, action, operation",
        )
        time_range: Optional[TimeRangeRequest] = Field(
            None, description="Time range filter"
        )
        categories: Optional[List[str]] = Field(
            None, description="Filter by event categories"
        )
        severities: Optional[List[str]] = Field(
            None, description="Filter by severity levels"
        )
        event_types: Optional[List[str]] = Field(
            None, description="Filter by event types"
        )
        outcomes: Optional[List[str]] = Field(
            None, description="Filter by outcomes: success, failure, error"
        )
        user_ids: Optional[List[str]] = Field(
            None, description="Filter by user IDs"
        )
        organization_ids: Optional[List[str]] = Field(
            None, description="Filter by organization IDs"
        )
        resource_types: Optional[List[str]] = Field(
            None, description="Filter by resource types"
        )
        ip_addresses: Optional[List[str]] = Field(
            None, description="Filter by IP addresses"
        )
        aggregations: Optional[List[str]] = Field(
            None,
            description="Fields to aggregate: category, severity, event_type, outcome, "
                        "user_id, resource_type",
        )
        limit: int = Field(100, ge=1, le=1000, description="Maximum results")
        offset: int = Field(0, ge=0, description="Result offset for pagination")

    class AggregationBucketResponse(BaseModel):
        """Single aggregation bucket."""

        key: str = Field(..., description="Bucket key/value")
        count: int = Field(..., description="Number of events in bucket")
        percentage: float = Field(0.0, description="Percentage of total")

    class SearchAggregationResponse(BaseModel):
        """Aggregation results for a field."""

        field: str = Field(..., description="Aggregated field name")
        buckets: List[AggregationBucketResponse] = Field(
            ..., description="Aggregation buckets"
        )
        total: int = Field(..., description="Total count across all buckets")

    class SearchHitResponse(BaseModel):
        """Single search hit (audit event)."""

        model_config = ConfigDict(from_attributes=True)

        id: UUID = Field(..., description="Event UUID")
        performed_at: datetime = Field(..., description="Event timestamp")
        category: str = Field(..., description="Event category")
        severity: str = Field(..., description="Severity level")
        event_type: str = Field(..., description="Event type")
        operation: str = Field(..., description="Operation performed")
        user_email: Optional[str] = Field(None, description="Acting user email")
        resource_type: Optional[str] = Field(None, description="Resource type")
        resource_path: Optional[str] = Field(None, description="Resource path")
        outcome: str = Field(..., description="Event outcome")
        change_summary: Optional[str] = Field(None, description="Summary of changes")
        metadata: Dict[str, Any] = Field(default_factory=dict)

    class SearchResponse(BaseModel):
        """Search response with hits and aggregations."""

        hits: List[SearchHitResponse] = Field(..., description="Matching events")
        total: int = Field(..., description="Total matching events")
        aggregations: List[SearchAggregationResponse] = Field(
            default_factory=list, description="Aggregation results"
        )
        query_time_ms: float = Field(0.0, description="Query execution time in ms")
        limit: int = Field(..., description="Requested limit")
        offset: int = Field(..., description="Requested offset")


# ---------------------------------------------------------------------------
# Dependencies
# ---------------------------------------------------------------------------


def _get_repository() -> Any:
    """FastAPI dependency that provides the AuditEventRepository.

    Returns:
        The AuditEventRepository singleton.

    Raises:
        HTTPException 503: If repository is not available.
    """
    try:
        from greenlang.infrastructure.audit_service.repository import (
            get_audit_repository,
        )
        return get_audit_repository()
    except (ImportError, RuntimeError) as exc:
        logger.error("Audit repository not available: %s", exc)
        raise HTTPException(
            status_code=503,
            detail="Audit service is not available.",
        )


# ---------------------------------------------------------------------------
# Router Definition
# ---------------------------------------------------------------------------

if FASTAPI_AVAILABLE:
    import time

    from greenlang.infrastructure.audit_service.models import (
        EventCategory,
        EventOutcome,
        SearchQuery,
        SeverityLevel,
        TimeRange,
    )

    search_router = APIRouter(
        prefix="/api/v1/audit",
        tags=["Audit Search"],
        responses={
            400: {"description": "Bad Request"},
            422: {"description": "Validation Error"},
            500: {"description": "Internal Server Error"},
            503: {"description": "Service Unavailable"},
        },
    )

    @search_router.post(
        "/search",
        response_model=SearchResponse,
        summary="Advanced audit search",
        description="Search audit events with LogQL-like queries and aggregations.",
        operation_id="search_audit_events",
    )
    async def search_audit_events(
        request: SearchRequest,
        repository: Any = Depends(_get_repository),
    ) -> SearchResponse:
        """Advanced search for audit events.

        Supports:
        - LogQL-like query syntax (field:value AND/OR field:value)
        - Time range filtering
        - Multiple filter criteria
        - Field aggregations for analytics

        Query Syntax Examples:
        - event_type:login
        - severity:error AND category:authentication
        - user_email:admin@example.com OR user_email:root@example.com
        - event_type:permission* (prefix match)
        - resource_type:"user.profile" (quoted for special chars)

        Args:
            request: Search request with query and filters.
            repository: Injected repository.

        Returns:
            Search results with hits and aggregations.
        """
        start_time = time.time()

        try:
            # Convert request to internal SearchQuery
            time_range: Optional[TimeRange] = None
            if request.time_range:
                time_range = TimeRange(
                    start=request.time_range.start,
                    end=request.time_range.end,
                )

            # Parse categories
            categories: Optional[List[EventCategory]] = None
            if request.categories:
                try:
                    categories = [EventCategory(c.lower()) for c in request.categories]
                except ValueError as exc:
                    raise HTTPException(
                        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                        detail=f"Invalid category: {exc}",
                    )

            # Parse severities
            severities: Optional[List[SeverityLevel]] = None
            if request.severities:
                try:
                    severities = [SeverityLevel(s.lower()) for s in request.severities]
                except ValueError as exc:
                    raise HTTPException(
                        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                        detail=f"Invalid severity: {exc}",
                    )

            # Parse outcomes
            outcomes: Optional[List[EventOutcome]] = None
            if request.outcomes:
                try:
                    outcomes = [EventOutcome(o.lower()) for o in request.outcomes]
                except ValueError as exc:
                    raise HTTPException(
                        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                        detail=f"Invalid outcome: {exc}",
                    )

            # Parse user IDs
            user_ids: Optional[List[UUID]] = None
            if request.user_ids:
                try:
                    user_ids = [UUID(uid) for uid in request.user_ids]
                except ValueError as exc:
                    raise HTTPException(
                        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                        detail=f"Invalid user_id format: {exc}",
                    )

            # Parse organization IDs
            org_ids: Optional[List[UUID]] = None
            if request.organization_ids:
                try:
                    org_ids = [UUID(oid) for oid in request.organization_ids]
                except ValueError as exc:
                    raise HTTPException(
                        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                        detail=f"Invalid organization_id format: {exc}",
                    )

            search_query = SearchQuery(
                query=request.query,
                time_range=time_range,
                categories=categories,
                severities=severities,
                event_types=request.event_types,
                outcomes=outcomes,
                user_ids=user_ids,
                organization_ids=org_ids,
                resource_types=request.resource_types,
                ip_addresses=request.ip_addresses,
                aggregations=request.aggregations,
            )

            events, total_count, aggregations = await repository.search(
                query=search_query,
                limit=request.limit,
                offset=request.offset,
            )

            # Convert to response models
            hits = [
                SearchHitResponse(
                    id=e.id,
                    performed_at=e.performed_at,
                    category=e.category.value,
                    severity=e.severity.value,
                    event_type=e.event_type,
                    operation=e.operation,
                    user_email=e.user_email,
                    resource_type=e.resource_type,
                    resource_path=e.resource_path,
                    outcome=e.outcome.value,
                    change_summary=e.change_summary,
                    metadata=e.metadata,
                )
                for e in events
            ]

            agg_responses = [
                SearchAggregationResponse(
                    field=agg.field,
                    buckets=[
                        AggregationBucketResponse(
                            key=b.key,
                            count=b.count,
                            percentage=b.percentage,
                        )
                        for b in agg.buckets
                    ],
                    total=agg.total,
                )
                for agg in aggregations
            ]

            query_time_ms = (time.time() - start_time) * 1000

            return SearchResponse(
                hits=hits,
                total=total_count,
                aggregations=agg_responses,
                query_time_ms=round(query_time_ms, 2),
                limit=request.limit,
                offset=request.offset,
            )

        except HTTPException:
            raise
        except Exception as exc:
            logger.exception("Search failed")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Search failed: {exc}",
            )

    # SEC-001: Apply authentication and permission protection
    try:
        from greenlang.infrastructure.auth_service.route_protector import (
            protect_router,
        )
        protect_router(search_router)
    except ImportError:
        pass  # auth_service not available

else:
    search_router = None  # type: ignore[assignment]
    logger.warning("FastAPI not available - search_router is None")


__all__ = ["search_router"]
