# -*- coding: utf-8 -*-
"""
Audit Events REST API Routes - SEC-005

FastAPI router for audit event listing and retrieval endpoints:

    GET  /api/v1/audit/events            - List events with filters and pagination
    GET  /api/v1/audit/events/{event_id} - Get single event details

Author: GreenLang Framework Team
Date: February 2026
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, List, Optional
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

    class AuditEventResponse(BaseModel):
        """Full audit event response."""

        model_config = ConfigDict(from_attributes=True)

        id: UUID = Field(..., description="Event UUID")
        event_id: Optional[str] = Field(None, description="External correlation ID")
        trace_id: Optional[str] = Field(None, description="Distributed trace ID")
        performed_at: datetime = Field(..., description="Event timestamp")
        category: str = Field(..., description="Event category")
        severity: str = Field(..., description="Severity level")
        event_type: str = Field(..., description="Event type identifier")
        operation: str = Field(..., description="Operation type")
        user_id: Optional[UUID] = Field(None, description="Acting user ID")
        user_email: Optional[str] = Field(None, description="Acting user email")
        organization_id: Optional[UUID] = Field(None, description="Organization/tenant ID")
        resource_type: Optional[str] = Field(None, description="Resource type affected")
        resource_path: Optional[str] = Field(None, description="Resource path")
        action: Optional[str] = Field(None, description="Action performed")
        outcome: str = Field("success", description="Event outcome")
        ip_address: Optional[str] = Field(None, description="Client IP address")
        change_summary: Optional[str] = Field(None, description="Summary of changes")
        metadata: dict = Field(default_factory=dict, description="Additional metadata")

    class AuditEventDetailResponse(AuditEventResponse):
        """Detailed audit event response with full data."""

        old_data: Optional[dict] = Field(None, description="State before change")
        new_data: Optional[dict] = Field(None, description="State after change")
        changed_fields: Optional[List[str]] = Field(None, description="List of changed fields")
        user_agent: Optional[str] = Field(None, description="Client user agent")
        request_id: Optional[UUID] = Field(None, description="Request ID")
        session_id: Optional[UUID] = Field(None, description="Session ID")
        error_message: Optional[str] = Field(None, description="Error message if failed")
        error_code: Optional[str] = Field(None, description="Error code if failed")
        tags: Optional[List[str]] = Field(None, description="Event tags")
        data_classification: Optional[str] = Field(None, description="Data classification")
        gdpr_relevant: bool = Field(False, description="GDPR relevant flag")

    class AuditEventsListResponse(BaseModel):
        """Paginated list of audit events."""

        items: List[AuditEventResponse] = Field(..., description="Audit events")
        total: Optional[int] = Field(None, description="Total matching events (if known)")
        cursor: Optional[str] = Field(None, description="Current cursor position")
        next_cursor: Optional[str] = Field(None, description="Next page cursor")
        has_more: bool = Field(False, description="Whether more results exist")


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
    """Extract tenant ID from request headers.

    Args:
        request: The FastAPI Request object.

    Returns:
        Tenant ID string or None.
    """
    return request.headers.get("x-tenant-id")


def _get_user_id(request: Request) -> Optional[str]:
    """Extract user ID from request headers.

    Args:
        request: The FastAPI Request object.

    Returns:
        User ID string or None.
    """
    return request.headers.get("x-user-id")


# ---------------------------------------------------------------------------
# Router Definition
# ---------------------------------------------------------------------------

if FASTAPI_AVAILABLE:
    from greenlang.infrastructure.audit_service.models import (
        AuditEvent,
        EventOutcome,
        SeverityLevel,
    )

    events_router = APIRouter(
        prefix="/api/v1/audit/events",
        tags=["Audit Events"],
        responses={
            400: {"description": "Bad Request"},
            403: {"description": "Forbidden"},
            404: {"description": "Event Not Found"},
            500: {"description": "Internal Server Error"},
            503: {"description": "Service Unavailable"},
        },
    )

    @events_router.get(
        "",
        response_model=AuditEventsListResponse,
        summary="List audit events",
        description="Retrieve audit events with optional filters and cursor-based pagination.",
        operation_id="list_audit_events",
    )
    async def list_audit_events(
        request: Request,
        since: Optional[datetime] = Query(
            None, description="Start of time range (ISO 8601)"
        ),
        until: Optional[datetime] = Query(
            None, description="End of time range (ISO 8601)"
        ),
        event_types: Optional[str] = Query(
            None, description="Comma-separated event types to filter"
        ),
        severity: Optional[str] = Query(
            None, description="Filter by severity: debug, info, warning, error, critical"
        ),
        tenant_id: Optional[str] = Query(
            None, description="Filter by tenant/organization ID"
        ),
        user_id: Optional[str] = Query(
            None, description="Filter by acting user ID"
        ),
        resource_type: Optional[str] = Query(
            None, description="Filter by resource type"
        ),
        result: Optional[str] = Query(
            None, description="Filter by outcome: success, failure, error"
        ),
        limit: int = Query(50, ge=1, le=1000, description="Maximum events to return"),
        cursor: Optional[str] = Query(
            None, description="Pagination cursor from previous response"
        ),
        repository: Any = Depends(_get_repository),
    ) -> AuditEventsListResponse:
        """List audit events with filters and cursor-based pagination.

        Supports filtering by time range, event types, severity, tenant,
        user, resource type, and outcome. Uses cursor-based pagination
        for efficient traversal of large result sets.

        Args:
            request: HTTP request.
            since: Start of time range.
            until: End of time range.
            event_types: Comma-separated event types.
            severity: Severity level filter.
            tenant_id: Tenant/organization filter.
            user_id: User filter.
            resource_type: Resource type filter.
            result: Outcome filter.
            limit: Maximum results.
            cursor: Pagination cursor.
            repository: Injected repository.

        Returns:
            Paginated list of audit events.
        """
        # Parse event types
        event_types_list: Optional[List[str]] = None
        if event_types:
            event_types_list = [t.strip() for t in event_types.split(",")]

        # Parse severity
        severity_enum: Optional[SeverityLevel] = None
        if severity:
            try:
                severity_enum = SeverityLevel(severity.lower())
            except ValueError:
                raise HTTPException(
                    status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                    detail=f"Invalid severity '{severity}'. "
                           f"Allowed: {[s.value for s in SeverityLevel]}",
                )

        # Parse outcome
        outcome_enum: Optional[EventOutcome] = None
        if result:
            try:
                outcome_enum = EventOutcome(result.lower())
            except ValueError:
                raise HTTPException(
                    status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                    detail=f"Invalid result '{result}'. "
                           f"Allowed: {[o.value for o in EventOutcome]}",
                )

        # Parse user_id to UUID if provided
        user_uuid: Optional[UUID] = None
        if user_id:
            try:
                user_uuid = UUID(user_id)
            except ValueError:
                raise HTTPException(
                    status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                    detail=f"Invalid user_id format: {user_id}",
                )

        # Override tenant_id from header if not provided in query
        if not tenant_id:
            tenant_id = _get_tenant_id(request)

        try:
            events, next_cursor = await repository.list_events(
                since=since,
                until=until,
                event_types=event_types_list,
                severity=severity_enum,
                tenant_id=tenant_id,
                user_id=user_uuid,
                resource_type=resource_type,
                result=outcome_enum,
                limit=limit,
                cursor=cursor,
            )

            items = [
                AuditEventResponse(
                    id=e.id,
                    event_id=None,
                    trace_id=None,
                    performed_at=e.performed_at,
                    category=e.category.value,
                    severity=e.severity.value,
                    event_type=e.event_type,
                    operation=e.operation,
                    user_id=None,
                    user_email=e.user_email,
                    organization_id=None,
                    resource_type=e.resource_type,
                    resource_path=None,
                    action=None,
                    outcome=e.outcome.value,
                    ip_address=None,
                    change_summary=None,
                    metadata={},
                )
                for e in events
            ]

            return AuditEventsListResponse(
                items=items,
                total=None,  # Not available with cursor pagination
                cursor=cursor,
                next_cursor=next_cursor,
                has_more=next_cursor is not None,
            )

        except Exception as exc:
            logger.exception("Failed to list audit events")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to list audit events: {exc}",
            )

    @events_router.get(
        "/{event_id}",
        response_model=AuditEventDetailResponse,
        summary="Get audit event details",
        description="Retrieve a single audit event by its UUID.",
        operation_id="get_audit_event",
    )
    async def get_audit_event(
        event_id: UUID,
        repository: Any = Depends(_get_repository),
    ) -> AuditEventDetailResponse:
        """Get detailed audit event by ID.

        Args:
            event_id: Event UUID.
            repository: Injected repository.

        Returns:
            Full audit event details.

        Raises:
            HTTPException 404: If event not found.
        """
        try:
            event = await repository.get_event(event_id)
        except Exception as exc:
            logger.exception("Failed to get audit event %s", event_id)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to get audit event: {exc}",
            )

        if event is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Audit event '{event_id}' not found.",
            )

        return AuditEventDetailResponse(
            id=event.id,
            event_id=event.event_id,
            trace_id=event.trace_id,
            performed_at=event.performed_at,
            category=event.category.value,
            severity=event.severity.value,
            event_type=event.event_type,
            operation=event.operation,
            user_id=event.user_id,
            user_email=event.user_email,
            organization_id=event.organization_id,
            resource_type=event.resource_type,
            resource_path=event.resource_path,
            action=event.action,
            outcome=event.outcome.value,
            ip_address=event.ip_address,
            change_summary=event.change_summary,
            metadata=event.metadata,
            old_data=event.old_data,
            new_data=event.new_data,
            changed_fields=event.changed_fields,
            user_agent=event.user_agent,
            request_id=event.request_id,
            session_id=event.session_id,
            error_message=event.error_message,
            error_code=event.error_code,
            tags=event.tags,
            data_classification=event.data_classification,
            gdpr_relevant=event.gdpr_relevant,
        )

    # SEC-001: Apply authentication and permission protection
    try:
        from greenlang.infrastructure.auth_service.route_protector import (
            protect_router,
        )
        protect_router(events_router)
    except ImportError:
        pass  # auth_service not available

else:
    events_router = None  # type: ignore[assignment]
    logger.warning("FastAPI not available - events_router is None")


__all__ = ["events_router"]
