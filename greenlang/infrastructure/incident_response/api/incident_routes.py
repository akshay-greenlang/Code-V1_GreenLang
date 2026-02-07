# -*- coding: utf-8 -*-
"""
Incident Response REST API Routes - SEC-010

FastAPI router for incident management:

    GET  /api/v1/secops/incidents              - List incidents with pagination/filters
    GET  /api/v1/secops/incidents/{id}         - Get incident details
    POST /api/v1/secops/incidents/{id}/acknowledge - Acknowledge incident
    POST /api/v1/secops/incidents/{id}/assign  - Assign responder
    POST /api/v1/secops/incidents/{id}/execute-playbook - Execute playbook
    PUT  /api/v1/secops/incidents/{id}/resolve - Resolve incident
    PUT  /api/v1/secops/incidents/{id}/close   - Close incident
    GET  /api/v1/secops/incidents/{id}/timeline - Get timeline
    GET  /api/v1/secops/incidents/metrics      - MTTD/MTTR metrics

Author: GreenLang Security Team
Date: February 2026
Status: Production Ready
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
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

    class IncidentResponse(BaseModel):
        """Incident response model."""

        model_config = ConfigDict(from_attributes=True)

        id: UUID = Field(..., description="Incident UUID")
        incident_number: str = Field(..., description="Human-readable incident number")
        title: str = Field(..., description="Incident title")
        description: Optional[str] = Field(None, description="Detailed description")
        severity: str = Field(..., description="Severity level (P0-P3)")
        status: str = Field(..., description="Current status")
        incident_type: str = Field(..., description="Incident type")
        source: str = Field(..., description="Alert source")
        detected_at: datetime = Field(..., description="Detection timestamp")
        acknowledged_at: Optional[datetime] = Field(None, description="Acknowledgment time")
        resolved_at: Optional[datetime] = Field(None, description="Resolution time")
        closed_at: Optional[datetime] = Field(None, description="Closure time")
        assignee_id: Optional[UUID] = Field(None, description="Assignee UUID")
        assignee_name: Optional[str] = Field(None, description="Assignee name")
        playbook_id: Optional[str] = Field(None, description="Associated playbook")
        playbook_execution_id: Optional[UUID] = Field(None, description="Execution ID")
        affected_systems: List[str] = Field(default_factory=list, description="Affected systems")
        affected_users: Optional[int] = Field(None, description="Affected user count")
        tags: List[str] = Field(default_factory=list, description="Tags")
        mttd_seconds: Optional[float] = Field(None, description="Mean Time to Detect")
        mttr_seconds: Optional[float] = Field(None, description="Mean Time to Resolve")

    class IncidentListResponse(BaseModel):
        """Paginated list of incidents."""

        items: List[IncidentResponse] = Field(..., description="Incidents")
        total: int = Field(..., description="Total count")
        limit: int = Field(..., description="Page size")
        offset: int = Field(..., description="Offset")
        has_more: bool = Field(..., description="More results available")

    class TimelineEventResponse(BaseModel):
        """Timeline event response model."""

        model_config = ConfigDict(from_attributes=True)

        id: UUID = Field(..., description="Event UUID")
        incident_id: UUID = Field(..., description="Incident UUID")
        event_type: str = Field(..., description="Event type")
        timestamp: datetime = Field(..., description="Event timestamp")
        actor_id: Optional[UUID] = Field(None, description="Actor UUID")
        actor_name: Optional[str] = Field(None, description="Actor name")
        description: str = Field(..., description="Event description")
        old_value: Optional[str] = Field(None, description="Previous value")
        new_value: Optional[str] = Field(None, description="New value")

    class TimelineResponse(BaseModel):
        """Incident timeline response."""

        incident_id: UUID = Field(..., description="Incident UUID")
        incident_number: str = Field(..., description="Incident number")
        events: List[TimelineEventResponse] = Field(..., description="Timeline events")

    class AcknowledgeRequest(BaseModel):
        """Acknowledge incident request."""

        acknowledger_name: Optional[str] = Field(None, description="Acknowledger name")
        comment: Optional[str] = Field(None, description="Acknowledgment comment")

    class AssignRequest(BaseModel):
        """Assign responder request."""

        assignee_id: UUID = Field(..., description="Assignee UUID")
        assignee_name: str = Field(..., description="Assignee name")
        comment: Optional[str] = Field(None, description="Assignment comment")

    class ExecutePlaybookRequest(BaseModel):
        """Execute playbook request."""

        playbook_id: str = Field(..., description="Playbook to execute")
        dry_run: bool = Field(False, description="Run in dry-run mode")

    class PlaybookExecutionResponse(BaseModel):
        """Playbook execution response."""

        model_config = ConfigDict(from_attributes=True)

        id: UUID = Field(..., description="Execution UUID")
        incident_id: UUID = Field(..., description="Incident UUID")
        playbook_id: str = Field(..., description="Playbook ID")
        playbook_name: str = Field(..., description="Playbook name")
        status: str = Field(..., description="Execution status")
        started_at: Optional[datetime] = Field(None, description="Start time")
        completed_at: Optional[datetime] = Field(None, description="Completion time")
        steps_completed: int = Field(0, description="Steps completed")
        steps_total: int = Field(..., description="Total steps")
        dry_run: bool = Field(False, description="Dry run mode")

    class ResolveRequest(BaseModel):
        """Resolve incident request."""

        resolution_notes: Optional[str] = Field(None, description="Resolution notes")

    class CloseRequest(BaseModel):
        """Close incident request."""

        closure_notes: Optional[str] = Field(None, description="Closure notes")
        generate_post_mortem: bool = Field(True, description="Generate post-mortem")

    class MetricsSummaryResponse(BaseModel):
        """Incident metrics summary."""

        period_start: datetime = Field(..., description="Period start")
        period_end: datetime = Field(..., description="Period end")
        total_incidents: int = Field(0, description="Total incidents")
        incidents_by_severity: Dict[str, int] = Field(default_factory=dict)
        incidents_by_status: Dict[str, int] = Field(default_factory=dict)
        mttd_seconds_avg: Optional[float] = Field(None, description="Average MTTD")
        mttd_seconds_p50: Optional[float] = Field(None, description="P50 MTTD")
        mttd_seconds_p95: Optional[float] = Field(None, description="P95 MTTD")
        mttr_seconds_avg: Optional[float] = Field(None, description="Average MTTR")
        mttr_seconds_p50: Optional[float] = Field(None, description="P50 MTTR")
        mttr_seconds_p95: Optional[float] = Field(None, description="P95 MTTR")
        sla_compliance_rate: Optional[float] = Field(None, description="SLA compliance %")
        playbook_automation_rate: Optional[float] = Field(None, description="Automation %")

    class PlaybookListResponse(BaseModel):
        """List of available playbooks."""

        playbooks: List[Dict[str, Any]] = Field(..., description="Available playbooks")


# ---------------------------------------------------------------------------
# Dependencies
# ---------------------------------------------------------------------------


def _get_user_id(request: Request) -> Optional[UUID]:
    """Extract user ID from request headers."""
    user_str = request.headers.get("x-user-id")
    if user_str:
        try:
            return UUID(user_str)
        except ValueError:
            return None
    return None


def _get_user_name(request: Request) -> Optional[str]:
    """Extract user name from request headers."""
    return request.headers.get("x-user-name")


# ---------------------------------------------------------------------------
# Router Definition
# ---------------------------------------------------------------------------

if FASTAPI_AVAILABLE:

    incident_router = APIRouter(
        prefix="/incidents",
        tags=["Incident Response"],
        responses={
            400: {"description": "Bad Request"},
            403: {"description": "Forbidden"},
            404: {"description": "Incident Not Found"},
            500: {"description": "Internal Server Error"},
        },
    )

    @incident_router.get(
        "",
        response_model=IncidentListResponse,
        summary="List incidents",
        description="Query incidents with optional filters and pagination.",
        operation_id="list_incidents",
    )
    async def list_incidents(
        request: Request,
        status_filter: Optional[str] = Query(
            None,
            alias="status",
            description="Filter by status: detected, acknowledged, investigating, remediating, resolved, closed",
        ),
        severity: Optional[str] = Query(
            None,
            description="Filter by severity: P0, P1, P2, P3",
        ),
        incident_type: Optional[str] = Query(
            None,
            alias="type",
            description="Filter by incident type",
        ),
        since: Optional[datetime] = Query(
            None,
            description="Incidents detected after this time",
        ),
        until: Optional[datetime] = Query(
            None,
            description="Incidents detected before this time",
        ),
        limit: int = Query(50, ge=1, le=200, description="Page size"),
        offset: int = Query(0, ge=0, description="Offset"),
    ) -> IncidentListResponse:
        """List incidents with optional filters."""
        from greenlang.infrastructure.incident_response.tracker import get_tracker
        from greenlang.infrastructure.incident_response.models import (
            IncidentStatus,
            EscalationLevel,
            IncidentType as IT,
        )

        tracker = get_tracker()

        # Convert filter strings to enums
        status_enum = None
        if status_filter:
            try:
                status_enum = IncidentStatus(status_filter.lower())
            except ValueError:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Invalid status: {status_filter}",
                )

        severity_enum = None
        if severity:
            try:
                severity_enum = EscalationLevel(severity.upper())
            except ValueError:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Invalid severity: {severity}",
                )

        type_enum = None
        if incident_type:
            try:
                type_enum = IT(incident_type.lower())
            except ValueError:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Invalid incident type: {incident_type}",
                )

        incidents = await tracker.list_incidents(
            status=status_enum,
            severity=severity_enum,
            incident_type=type_enum,
            limit=limit + 1,  # Fetch one extra to check has_more
            offset=offset,
        )

        # Check if there are more results
        has_more = len(incidents) > limit
        if has_more:
            incidents = incidents[:limit]

        items = [
            IncidentResponse(
                id=inc.id,
                incident_number=inc.incident_number,
                title=inc.title,
                description=inc.description,
                severity=inc.severity.value,
                status=inc.status.value,
                incident_type=inc.incident_type.value,
                source=inc.source.value,
                detected_at=inc.detected_at,
                acknowledged_at=inc.acknowledged_at,
                resolved_at=inc.resolved_at,
                closed_at=inc.closed_at,
                assignee_id=inc.assignee_id,
                assignee_name=inc.assignee_name,
                playbook_id=inc.playbook_id,
                playbook_execution_id=inc.playbook_execution_id,
                affected_systems=inc.affected_systems,
                affected_users=inc.affected_users,
                tags=inc.tags,
                mttd_seconds=inc.get_mttd_seconds(),
                mttr_seconds=inc.get_mttr_seconds(),
            )
            for inc in incidents
        ]

        return IncidentListResponse(
            items=items,
            total=len(items),
            limit=limit,
            offset=offset,
            has_more=has_more,
        )

    @incident_router.get(
        "/metrics",
        response_model=MetricsSummaryResponse,
        summary="Get incident metrics",
        description="Get MTTD, MTTR, and other incident response metrics.",
        operation_id="get_incident_metrics",
    )
    async def get_incident_metrics(
        request: Request,
        days: int = Query(30, ge=1, le=365, description="Number of days to analyze"),
    ) -> MetricsSummaryResponse:
        """Get incident metrics for a time period."""
        from datetime import timedelta
        from greenlang.infrastructure.incident_response.tracker import get_tracker

        tracker = get_tracker()

        end_time = datetime.now(timezone.utc)
        start_time = end_time - timedelta(days=days)

        # Get all incidents for the period
        all_incidents = await tracker.list_incidents(limit=10000, offset=0)

        # Filter by time period
        incidents = [
            inc for inc in all_incidents
            if inc.detected_at >= start_time
        ]

        # Calculate metrics
        by_severity: Dict[str, int] = {}
        by_status: Dict[str, int] = {}
        mttd_values: List[float] = []
        mttr_values: List[float] = []
        playbook_count = 0
        sla_compliant = 0

        for inc in incidents:
            # Count by severity
            sev = inc.severity.value
            by_severity[sev] = by_severity.get(sev, 0) + 1

            # Count by status
            st = inc.status.value
            by_status[st] = by_status.get(st, 0) + 1

            # MTTD
            mttd = inc.get_mttd_seconds()
            if mttd is not None:
                mttd_values.append(mttd)
                # Check SLA (15 min for P0, 60 min for P1, etc.)
                sla_map = {"P0": 900, "P1": 3600, "P2": 14400, "P3": 86400}
                sla_threshold = sla_map.get(sev, 3600)
                if mttd <= sla_threshold:
                    sla_compliant += 1

            # MTTR
            mttr = inc.get_mttr_seconds()
            if mttr is not None:
                mttr_values.append(mttr)

            # Playbook usage
            if inc.playbook_execution_id:
                playbook_count += 1

        # Calculate aggregates
        def percentile(values: List[float], p: float) -> Optional[float]:
            if not values:
                return None
            sorted_vals = sorted(values)
            idx = int(len(sorted_vals) * p / 100)
            return sorted_vals[min(idx, len(sorted_vals) - 1)]

        return MetricsSummaryResponse(
            period_start=start_time,
            period_end=end_time,
            total_incidents=len(incidents),
            incidents_by_severity=by_severity,
            incidents_by_status=by_status,
            mttd_seconds_avg=sum(mttd_values) / len(mttd_values) if mttd_values else None,
            mttd_seconds_p50=percentile(mttd_values, 50),
            mttd_seconds_p95=percentile(mttd_values, 95),
            mttr_seconds_avg=sum(mttr_values) / len(mttr_values) if mttr_values else None,
            mttr_seconds_p50=percentile(mttr_values, 50),
            mttr_seconds_p95=percentile(mttr_values, 95),
            sla_compliance_rate=(sla_compliant / len(incidents) * 100) if incidents else None,
            playbook_automation_rate=(playbook_count / len(incidents) * 100) if incidents else None,
        )

    @incident_router.get(
        "/playbooks",
        response_model=PlaybookListResponse,
        summary="List available playbooks",
        description="Get list of available remediation playbooks.",
        operation_id="list_playbooks",
    )
    async def list_playbooks(request: Request) -> PlaybookListResponse:
        """List available playbooks."""
        from greenlang.infrastructure.incident_response.playbook_executor import (
            get_playbook_executor,
        )

        executor = get_playbook_executor()
        playbooks = executor.get_available_playbooks()

        return PlaybookListResponse(playbooks=playbooks)

    @incident_router.get(
        "/{incident_id}",
        response_model=IncidentResponse,
        summary="Get incident details",
        description="Get detailed information about a specific incident.",
        operation_id="get_incident",
    )
    async def get_incident(
        incident_id: UUID,
        request: Request,
    ) -> IncidentResponse:
        """Get incident by ID."""
        from greenlang.infrastructure.incident_response.tracker import get_tracker

        tracker = get_tracker()
        incident = await tracker.get_incident(incident_id)

        if not incident:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Incident {incident_id} not found",
            )

        return IncidentResponse(
            id=incident.id,
            incident_number=incident.incident_number,
            title=incident.title,
            description=incident.description,
            severity=incident.severity.value,
            status=incident.status.value,
            incident_type=incident.incident_type.value,
            source=incident.source.value,
            detected_at=incident.detected_at,
            acknowledged_at=incident.acknowledged_at,
            resolved_at=incident.resolved_at,
            closed_at=incident.closed_at,
            assignee_id=incident.assignee_id,
            assignee_name=incident.assignee_name,
            playbook_id=incident.playbook_id,
            playbook_execution_id=incident.playbook_execution_id,
            affected_systems=incident.affected_systems,
            affected_users=incident.affected_users,
            tags=incident.tags,
            mttd_seconds=incident.get_mttd_seconds(),
            mttr_seconds=incident.get_mttr_seconds(),
        )

    @incident_router.post(
        "/{incident_id}/acknowledge",
        response_model=IncidentResponse,
        summary="Acknowledge incident",
        description="Acknowledge an incident to indicate response has begun.",
        operation_id="acknowledge_incident",
    )
    async def acknowledge_incident(
        incident_id: UUID,
        request: Request,
        body: AcknowledgeRequest,
    ) -> IncidentResponse:
        """Acknowledge an incident."""
        from greenlang.infrastructure.incident_response.tracker import get_tracker
        from greenlang.infrastructure.incident_response.models import IncidentStatus

        tracker = get_tracker()
        user_id = _get_user_id(request)
        user_name = body.acknowledger_name or _get_user_name(request)

        incident = await tracker.update_status(
            incident_id=incident_id,
            new_status=IncidentStatus.ACKNOWLEDGED,
            actor_id=user_id,
            actor_name=user_name,
            comment=body.comment,
        )

        if not incident:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Incident {incident_id} not found",
            )

        return IncidentResponse(
            id=incident.id,
            incident_number=incident.incident_number,
            title=incident.title,
            description=incident.description,
            severity=incident.severity.value,
            status=incident.status.value,
            incident_type=incident.incident_type.value,
            source=incident.source.value,
            detected_at=incident.detected_at,
            acknowledged_at=incident.acknowledged_at,
            resolved_at=incident.resolved_at,
            closed_at=incident.closed_at,
            assignee_id=incident.assignee_id,
            assignee_name=incident.assignee_name,
            playbook_id=incident.playbook_id,
            playbook_execution_id=incident.playbook_execution_id,
            affected_systems=incident.affected_systems,
            affected_users=incident.affected_users,
            tags=incident.tags,
            mttd_seconds=incident.get_mttd_seconds(),
            mttr_seconds=incident.get_mttr_seconds(),
        )

    @incident_router.post(
        "/{incident_id}/assign",
        response_model=IncidentResponse,
        summary="Assign responder",
        description="Assign a responder to the incident.",
        operation_id="assign_incident",
    )
    async def assign_incident(
        incident_id: UUID,
        request: Request,
        body: AssignRequest,
    ) -> IncidentResponse:
        """Assign a responder to an incident."""
        from greenlang.infrastructure.incident_response.tracker import get_tracker

        tracker = get_tracker()
        user_id = _get_user_id(request)
        user_name = _get_user_name(request)

        incident = await tracker.assign_responder(
            incident_id=incident_id,
            assignee_id=body.assignee_id,
            assignee_name=body.assignee_name,
            assigned_by=user_id,
            assigned_by_name=user_name,
        )

        if not incident:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Incident {incident_id} not found",
            )

        return IncidentResponse(
            id=incident.id,
            incident_number=incident.incident_number,
            title=incident.title,
            description=incident.description,
            severity=incident.severity.value,
            status=incident.status.value,
            incident_type=incident.incident_type.value,
            source=incident.source.value,
            detected_at=incident.detected_at,
            acknowledged_at=incident.acknowledged_at,
            resolved_at=incident.resolved_at,
            closed_at=incident.closed_at,
            assignee_id=incident.assignee_id,
            assignee_name=incident.assignee_name,
            playbook_id=incident.playbook_id,
            playbook_execution_id=incident.playbook_execution_id,
            affected_systems=incident.affected_systems,
            affected_users=incident.affected_users,
            tags=incident.tags,
            mttd_seconds=incident.get_mttd_seconds(),
            mttr_seconds=incident.get_mttr_seconds(),
        )

    @incident_router.post(
        "/{incident_id}/execute-playbook",
        response_model=PlaybookExecutionResponse,
        summary="Execute playbook",
        description="Execute an automated remediation playbook for the incident.",
        operation_id="execute_playbook",
    )
    async def execute_playbook(
        incident_id: UUID,
        request: Request,
        body: ExecutePlaybookRequest,
    ) -> PlaybookExecutionResponse:
        """Execute a playbook for an incident."""
        from greenlang.infrastructure.incident_response.tracker import get_tracker
        from greenlang.infrastructure.incident_response.playbook_executor import (
            get_playbook_executor,
        )

        tracker = get_tracker()
        executor = get_playbook_executor()
        user_name = _get_user_name(request) or "api"

        incident = await tracker.get_incident(incident_id)
        if not incident:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Incident {incident_id} not found",
            )

        try:
            result = await executor.execute(
                incident=incident,
                playbook_id=body.playbook_id,
                dry_run=body.dry_run,
                triggered_by=user_name,
            )
        except ValueError as e:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=str(e),
            )
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Playbook execution failed: {e}",
            )

        execution = result.execution

        return PlaybookExecutionResponse(
            id=execution.id,
            incident_id=execution.incident_id,
            playbook_id=execution.playbook_id,
            playbook_name=execution.playbook_name,
            status=execution.status.value,
            started_at=execution.started_at,
            completed_at=execution.completed_at,
            steps_completed=execution.steps_completed,
            steps_total=execution.steps_total,
            dry_run=execution.dry_run,
        )

    @incident_router.put(
        "/{incident_id}/resolve",
        response_model=IncidentResponse,
        summary="Resolve incident",
        description="Mark an incident as resolved.",
        operation_id="resolve_incident",
    )
    async def resolve_incident(
        incident_id: UUID,
        request: Request,
        body: ResolveRequest,
    ) -> IncidentResponse:
        """Resolve an incident."""
        from greenlang.infrastructure.incident_response.tracker import get_tracker
        from greenlang.infrastructure.incident_response.models import IncidentStatus

        tracker = get_tracker()
        user_id = _get_user_id(request)
        user_name = _get_user_name(request)

        incident = await tracker.update_status(
            incident_id=incident_id,
            new_status=IncidentStatus.RESOLVED,
            actor_id=user_id,
            actor_name=user_name,
            comment=body.resolution_notes,
        )

        if not incident:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Incident {incident_id} not found",
            )

        return IncidentResponse(
            id=incident.id,
            incident_number=incident.incident_number,
            title=incident.title,
            description=incident.description,
            severity=incident.severity.value,
            status=incident.status.value,
            incident_type=incident.incident_type.value,
            source=incident.source.value,
            detected_at=incident.detected_at,
            acknowledged_at=incident.acknowledged_at,
            resolved_at=incident.resolved_at,
            closed_at=incident.closed_at,
            assignee_id=incident.assignee_id,
            assignee_name=incident.assignee_name,
            playbook_id=incident.playbook_id,
            playbook_execution_id=incident.playbook_execution_id,
            affected_systems=incident.affected_systems,
            affected_users=incident.affected_users,
            tags=incident.tags,
            mttd_seconds=incident.get_mttd_seconds(),
            mttr_seconds=incident.get_mttr_seconds(),
        )

    @incident_router.put(
        "/{incident_id}/close",
        response_model=IncidentResponse,
        summary="Close incident",
        description="Close an incident and optionally generate post-mortem.",
        operation_id="close_incident",
    )
    async def close_incident(
        incident_id: UUID,
        request: Request,
        body: CloseRequest,
    ) -> IncidentResponse:
        """Close an incident."""
        from greenlang.infrastructure.incident_response.tracker import get_tracker
        from greenlang.infrastructure.incident_response.models import IncidentStatus

        tracker = get_tracker()
        user_id = _get_user_id(request)
        user_name = _get_user_name(request)

        incident = await tracker.update_status(
            incident_id=incident_id,
            new_status=IncidentStatus.CLOSED,
            actor_id=user_id,
            actor_name=user_name,
            comment=body.closure_notes,
        )

        if not incident:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Incident {incident_id} not found",
            )

        # Generate post-mortem if requested
        if body.generate_post_mortem:
            await tracker.generate_post_mortem(incident_id, created_by=user_id)

        return IncidentResponse(
            id=incident.id,
            incident_number=incident.incident_number,
            title=incident.title,
            description=incident.description,
            severity=incident.severity.value,
            status=incident.status.value,
            incident_type=incident.incident_type.value,
            source=incident.source.value,
            detected_at=incident.detected_at,
            acknowledged_at=incident.acknowledged_at,
            resolved_at=incident.resolved_at,
            closed_at=incident.closed_at,
            assignee_id=incident.assignee_id,
            assignee_name=incident.assignee_name,
            playbook_id=incident.playbook_id,
            playbook_execution_id=incident.playbook_execution_id,
            affected_systems=incident.affected_systems,
            affected_users=incident.affected_users,
            tags=incident.tags,
            mttd_seconds=incident.get_mttd_seconds(),
            mttr_seconds=incident.get_mttr_seconds(),
        )

    @incident_router.get(
        "/{incident_id}/timeline",
        response_model=TimelineResponse,
        summary="Get incident timeline",
        description="Get the chronological timeline of events for an incident.",
        operation_id="get_incident_timeline",
    )
    async def get_incident_timeline(
        incident_id: UUID,
        request: Request,
    ) -> TimelineResponse:
        """Get incident timeline."""
        from greenlang.infrastructure.incident_response.tracker import get_tracker

        tracker = get_tracker()
        incident = await tracker.get_incident(incident_id)

        if not incident:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Incident {incident_id} not found",
            )

        timeline = await tracker.get_timeline(incident_id)

        events = [
            TimelineEventResponse(
                id=event.id,
                incident_id=event.incident_id,
                event_type=event.event_type.value,
                timestamp=event.timestamp,
                actor_id=event.actor_id,
                actor_name=event.actor_name,
                description=event.description,
                old_value=event.old_value,
                new_value=event.new_value,
            )
            for event in timeline
        ]

        return TimelineResponse(
            incident_id=incident_id,
            incident_number=incident.incident_number,
            events=events,
        )

    # Apply route protection
    try:
        from greenlang.infrastructure.auth_service.route_protector import (
            protect_router,
        )

        protect_router(incident_router)
    except ImportError:
        pass

else:
    incident_router = None  # type: ignore[assignment]
    logger.warning("FastAPI not available - incident_router is None")


__all__ = [
    "incident_router",
    "FASTAPI_AVAILABLE",
]
