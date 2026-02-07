# -*- coding: utf-8 -*-
"""
Audit Statistics REST API Routes - SEC-005

FastAPI router for audit statistics and analytics:

    GET /api/v1/audit/stats     - Event statistics (counts by type, severity)
    GET /api/v1/audit/timeline  - Activity timeline for user/resource
    GET /api/v1/audit/hotspots  - Top 10 users, resources, IPs

Uses TimescaleDB continuous aggregates for efficient analytics.

Author: GreenLang Framework Team
Date: February 2026
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
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
# Response Models
# ---------------------------------------------------------------------------

if FASTAPI_AVAILABLE:

    class TimeRangeResponse(BaseModel):
        """Time range in statistics response."""

        start: datetime = Field(..., description="Start of time range")
        end: datetime = Field(..., description="End of time range")

    class EventStatisticsResponse(BaseModel):
        """Audit event statistics response."""

        total_events: int = Field(..., description="Total number of events")
        events_by_category: Dict[str, int] = Field(
            ..., description="Event counts by category"
        )
        events_by_severity: Dict[str, int] = Field(
            ..., description="Event counts by severity level"
        )
        events_by_outcome: Dict[str, int] = Field(
            ..., description="Event counts by outcome"
        )
        unique_users: int = Field(..., description="Number of unique users")
        unique_resources: int = Field(..., description="Number of unique resource types")
        time_range: TimeRangeResponse = Field(..., description="Query time range")

    class TimelineEntryResponse(BaseModel):
        """Single entry in activity timeline."""

        timestamp: datetime = Field(..., description="Event timestamp")
        event_type: str = Field(..., description="Event type")
        category: str = Field(..., description="Event category")
        severity: str = Field(..., description="Severity level")
        description: str = Field(..., description="Event description")
        resource_path: Optional[str] = Field(None, description="Resource path")
        outcome: str = Field(..., description="Event outcome")

    class TimelineResponse(BaseModel):
        """Activity timeline response."""

        entity_type: str = Field(..., description="Entity type (user or resource)")
        entity_id: str = Field(..., description="Entity identifier")
        entries: List[TimelineEntryResponse] = Field(..., description="Timeline entries")
        total: int = Field(..., description="Total entries returned")

    class SeverityBreakdown(BaseModel):
        """Severity breakdown for hotspot entries."""

        error: int = Field(0, description="Error count")
        warning: int = Field(0, description="Warning count")

    class HotspotEntryResponse(BaseModel):
        """Single hotspot entry (top user/resource/IP)."""

        identifier: str = Field(..., description="Entity identifier")
        label: Optional[str] = Field(None, description="Human-readable label")
        event_count: int = Field(..., description="Number of events")
        last_seen: datetime = Field(..., description="Last activity timestamp")
        severity_breakdown: SeverityBreakdown = Field(
            ..., description="Event severity breakdown"
        )

    class HotspotsResponse(BaseModel):
        """Top N hotspots response."""

        hotspot_type: str = Field(
            ..., description="Type of hotspot: users, resources, ips"
        )
        entries: List[HotspotEntryResponse] = Field(..., description="Hotspot entries")
        since: datetime = Field(..., description="Start of analysis period")
        total: int = Field(..., description="Number of entries returned")


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


def _get_tenant_id(request: Request) -> Optional[str]:
    """Extract tenant ID from request headers."""
    return request.headers.get("x-tenant-id")


# ---------------------------------------------------------------------------
# Router Definition
# ---------------------------------------------------------------------------

if FASTAPI_AVAILABLE:

    stats_router = APIRouter(
        prefix="/api/v1/audit",
        tags=["Audit Statistics"],
        responses={
            400: {"description": "Bad Request"},
            422: {"description": "Validation Error"},
            500: {"description": "Internal Server Error"},
            503: {"description": "Service Unavailable"},
        },
    )

    @stats_router.get(
        "/stats",
        response_model=EventStatisticsResponse,
        summary="Get event statistics",
        description="Retrieve audit event statistics for a time range.",
        operation_id="get_audit_statistics",
    )
    async def get_audit_statistics(
        request: Request,
        since: Optional[datetime] = Query(
            None, description="Start of time range (default: 7 days ago)"
        ),
        until: Optional[datetime] = Query(
            None, description="End of time range (default: now)"
        ),
        organization_id: Optional[str] = Query(
            None, description="Filter by organization ID"
        ),
        repository: Any = Depends(_get_repository),
    ) -> EventStatisticsResponse:
        """Get audit event statistics.

        Returns counts of events by category, severity, and outcome,
        along with unique user and resource counts.

        Uses TimescaleDB continuous aggregates (audit.hourly_audit_summary)
        for efficient querying of large datasets.

        Args:
            request: HTTP request.
            since: Start of time range.
            until: End of time range.
            organization_id: Organization filter.
            repository: Injected repository.

        Returns:
            Event statistics.
        """
        # Default time range: last 7 days
        if not until:
            until = datetime.now(timezone.utc)
        if not since:
            since = until - timedelta(days=7)

        # Parse organization_id
        org_uuid: Optional[UUID] = None
        if organization_id:
            try:
                org_uuid = UUID(organization_id)
            except ValueError:
                raise HTTPException(
                    status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                    detail=f"Invalid organization_id format: {organization_id}",
                )

        # Override from header if not provided
        if not org_uuid:
            tenant_id = _get_tenant_id(request)
            if tenant_id:
                try:
                    org_uuid = UUID(tenant_id)
                except ValueError:
                    pass

        try:
            stats = await repository.get_statistics(
                since=since,
                until=until,
                organization_id=org_uuid,
            )

            return EventStatisticsResponse(
                total_events=stats.total_events,
                events_by_category=stats.events_by_category,
                events_by_severity=stats.events_by_severity,
                events_by_outcome=stats.events_by_outcome,
                unique_users=stats.unique_users,
                unique_resources=stats.unique_resources,
                time_range=TimeRangeResponse(
                    start=stats.time_range.start,
                    end=stats.time_range.end,
                ),
            )

        except Exception as exc:
            logger.exception("Failed to get audit statistics")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to get statistics: {exc}",
            )

    @stats_router.get(
        "/timeline",
        response_model=TimelineResponse,
        summary="Get activity timeline",
        description="Retrieve activity timeline for a user or resource.",
        operation_id="get_activity_timeline",
    )
    async def get_activity_timeline(
        entity_type: str = Query(
            ..., description="Entity type: user or resource"
        ),
        entity_id: str = Query(..., description="Entity identifier (user_id or resource_type)"),
        since: Optional[datetime] = Query(
            None, description="Start of time range (default: 7 days ago)"
        ),
        until: Optional[datetime] = Query(
            None, description="End of time range (default: now)"
        ),
        limit: int = Query(100, ge=1, le=500, description="Maximum entries to return"),
        repository: Any = Depends(_get_repository),
    ) -> TimelineResponse:
        """Get activity timeline for a user or resource.

        Provides a chronological view of all audit events related to
        a specific user or resource type.

        Args:
            entity_type: Type of entity (user or resource).
            entity_id: Entity identifier.
            since: Start of time range.
            until: End of time range.
            limit: Maximum entries.
            repository: Injected repository.

        Returns:
            Activity timeline.
        """
        if entity_type not in ("user", "resource"):
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=f"Invalid entity_type '{entity_type}'. Must be 'user' or 'resource'.",
            )

        # Validate user_id format if entity_type is user
        if entity_type == "user":
            try:
                UUID(entity_id)
            except ValueError:
                raise HTTPException(
                    status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                    detail=f"Invalid user_id format: {entity_id}",
                )

        # Default time range
        if not until:
            until = datetime.now(timezone.utc)
        if not since:
            since = until - timedelta(days=7)

        try:
            timeline_entries = await repository.get_timeline(
                entity_type=entity_type,
                entity_id=entity_id,
                since=since,
                until=until,
                limit=limit,
            )

            entries = [
                TimelineEntryResponse(
                    timestamp=datetime.fromisoformat(e["timestamp"]),
                    event_type=e["event_type"],
                    category=e["category"],
                    severity=e["severity"],
                    description=e["description"],
                    resource_path=e.get("resource_path"),
                    outcome=e["outcome"],
                )
                for e in timeline_entries
            ]

            return TimelineResponse(
                entity_type=entity_type,
                entity_id=entity_id,
                entries=entries,
                total=len(entries),
            )

        except Exception as exc:
            logger.exception("Failed to get activity timeline")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to get timeline: {exc}",
            )

    @stats_router.get(
        "/hotspots",
        response_model=HotspotsResponse,
        summary="Get activity hotspots",
        description="Retrieve top users, resources, or IPs by event count.",
        operation_id="get_activity_hotspots",
    )
    async def get_activity_hotspots(
        request: Request,
        hotspot_type: str = Query(
            ..., description="Hotspot type: users, resources, or ips"
        ),
        since: Optional[datetime] = Query(
            None, description="Start of time range (default: 24 hours ago)"
        ),
        limit: int = Query(10, ge=1, le=100, description="Maximum entries to return"),
        organization_id: Optional[str] = Query(
            None, description="Filter by organization ID"
        ),
        repository: Any = Depends(_get_repository),
    ) -> HotspotsResponse:
        """Get top activity hotspots.

        Identifies the most active users, resources, or IP addresses
        based on audit event counts. Useful for:
        - Security monitoring (unusual activity detection)
        - Capacity planning
        - User behavior analysis

        Args:
            request: HTTP request.
            hotspot_type: Type of hotspot (users, resources, ips).
            since: Start of time range.
            limit: Maximum entries.
            organization_id: Organization filter.
            repository: Injected repository.

        Returns:
            Top N hotspot entries.
        """
        if hotspot_type not in ("users", "resources", "ips"):
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=f"Invalid hotspot_type '{hotspot_type}'. "
                       "Must be 'users', 'resources', or 'ips'.",
            )

        # Default: last 24 hours
        if not since:
            since = datetime.now(timezone.utc) - timedelta(hours=24)

        # Parse organization_id
        org_uuid: Optional[UUID] = None
        if organization_id:
            try:
                org_uuid = UUID(organization_id)
            except ValueError:
                raise HTTPException(
                    status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                    detail=f"Invalid organization_id format: {organization_id}",
                )

        # Override from header if not provided
        if not org_uuid:
            tenant_id = _get_tenant_id(request)
            if tenant_id:
                try:
                    org_uuid = UUID(tenant_id)
                except ValueError:
                    pass

        try:
            hotspots = await repository.get_hotspots(
                hotspot_type=hotspot_type,
                since=since,
                limit=limit,
                organization_id=org_uuid,
            )

            entries = [
                HotspotEntryResponse(
                    identifier=h.identifier,
                    label=h.label,
                    event_count=h.event_count,
                    last_seen=h.last_seen,
                    severity_breakdown=SeverityBreakdown(
                        error=h.severity_breakdown.get("error", 0),
                        warning=h.severity_breakdown.get("warning", 0),
                    ),
                )
                for h in hotspots
            ]

            return HotspotsResponse(
                hotspot_type=hotspot_type,
                entries=entries,
                since=since,
                total=len(entries),
            )

        except Exception as exc:
            logger.exception("Failed to get hotspots")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to get hotspots: {exc}",
            )

    # SEC-001: Apply authentication and permission protection
    try:
        from greenlang.infrastructure.auth_service.route_protector import (
            protect_router,
        )
        protect_router(stats_router)
    except ImportError:
        pass  # auth_service not available

else:
    stats_router = None  # type: ignore[assignment]
    logger.warning("FastAPI not available - stats_router is None")


__all__ = ["stats_router"]
