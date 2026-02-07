# -*- coding: utf-8 -*-
"""
Auditor Request Handler - SEC-009 Phase 5

Handles information requests from auditors with priority-based SLA tracking.
Manages the request lifecycle from creation through resolution, including
assignment, escalation, and overdue tracking.

SLA Targets by Priority:
    - CRITICAL: 4 hours
    - HIGH: 24 hours
    - NORMAL: 48 hours
    - LOW: 72 hours

Example:
    >>> handler = AuditorRequestHandler()
    >>> request = await handler.create_request(
    ...     auditor_id=auditor_uuid,
    ...     request=RequestCreate(
    ...         title="Access control population",
    ...         description="Please provide the full population of access control exceptions",
    ...         priority=RequestPriority.HIGH,
    ...     ),
    ... )
    >>> await handler.assign_request(request.request_id, assignee_uuid)
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# SLA hours by priority
SLA_HOURS = {
    "critical": 4,
    "high": 24,
    "normal": 48,
    "low": 72,
}


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------


class RequestPriority(str, Enum):
    """Priority levels for auditor requests with associated SLA."""

    CRITICAL = "critical"
    """4-hour SLA - Audit blocking issues."""

    HIGH = "high"
    """24-hour SLA - Required for current testing phase."""

    NORMAL = "normal"
    """48-hour SLA - Standard information requests."""

    LOW = "low"
    """72-hour SLA - Nice-to-have or clarification requests."""


class RequestStatus(str, Enum):
    """Status of an auditor request."""

    OPEN = "open"
    """Request submitted, awaiting assignment."""

    ASSIGNED = "assigned"
    """Request assigned to a team member."""

    IN_PROGRESS = "in_progress"
    """Actively being worked on."""

    PENDING_INFO = "pending_info"
    """Waiting for additional information from auditor."""

    UNDER_REVIEW = "under_review"
    """Response being reviewed before sending."""

    RESOLVED = "resolved"
    """Request has been resolved with evidence/response."""

    CLOSED = "closed"
    """Request closed by auditor."""

    ESCALATED = "escalated"
    """Request escalated to management."""


class RequestCategory(str, Enum):
    """Categories for auditor requests."""

    EVIDENCE = "evidence"
    """Request for specific evidence or documentation."""

    CLARIFICATION = "clarification"
    """Clarification on existing evidence or responses."""

    WALKTHROUGH = "walkthrough"
    """Request for process walkthrough or demonstration."""

    POPULATION = "population"
    """Request for complete population data for sampling."""

    EXCEPTION = "exception"
    """Inquiry about exceptions or deviations."""

    CONTROL_TEST = "control_test"
    """Request related to control testing."""

    POLICY = "policy"
    """Request for policies or procedures."""

    OTHER = "other"
    """Other request types."""


# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------


class RequestCreate(BaseModel):
    """Request creation input.

    Attributes:
        title: Short title for the request.
        description: Detailed description of what is needed.
        priority: Request priority (determines SLA).
        category: Request category for routing.
        related_criterion: SOC 2 criterion this relates to.
        attachments: Optional attachment references.
    """

    model_config = ConfigDict(extra="forbid")

    title: str = Field(
        ...,
        min_length=5,
        max_length=256,
        description="Short request title.",
    )
    description: str = Field(
        ...,
        min_length=10,
        max_length=4096,
        description="Detailed description.",
    )
    priority: RequestPriority = Field(
        default=RequestPriority.NORMAL,
        description="Request priority.",
    )
    category: RequestCategory = Field(
        default=RequestCategory.EVIDENCE,
        description="Request category.",
    )
    related_criterion: Optional[str] = Field(
        default=None,
        max_length=32,
        description="Related SOC 2 criterion (e.g., CC6.1).",
    )
    attachments: List[str] = Field(
        default_factory=list,
        description="Attachment file IDs.",
    )


class AuditorRequest(BaseModel):
    """Full auditor request with tracking information.

    Attributes:
        request_id: Unique request identifier.
        auditor_id: ID of the requesting auditor.
        title: Request title.
        description: Detailed description.
        priority: Request priority.
        category: Request category.
        status: Current status.
        related_criterion: Related SOC 2 criterion.
        assignee_id: Assigned team member ID.
        sla_deadline: SLA deadline based on priority.
        created_at: Request creation time.
        updated_at: Last update time.
        resolved_at: Resolution time.
        response: Response provided to auditor.
        evidence_ids: Evidence items linked to response.
        notes: Internal notes.
        escalated: Whether request has been escalated.
        escalation_reason: Reason for escalation.
    """

    model_config = ConfigDict(extra="forbid")

    request_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique request identifier.",
    )
    auditor_id: str = Field(
        ...,
        description="ID of requesting auditor.",
    )
    title: str = Field(
        ...,
        description="Request title.",
    )
    description: str = Field(
        ...,
        description="Detailed description.",
    )
    priority: RequestPriority = Field(
        default=RequestPriority.NORMAL,
        description="Request priority.",
    )
    category: RequestCategory = Field(
        default=RequestCategory.EVIDENCE,
        description="Request category.",
    )
    status: RequestStatus = Field(
        default=RequestStatus.OPEN,
        description="Current status.",
    )
    related_criterion: Optional[str] = Field(
        default=None,
        description="Related SOC 2 criterion.",
    )
    assignee_id: Optional[str] = Field(
        default=None,
        description="Assigned team member ID.",
    )
    sla_deadline: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc) + timedelta(hours=48),
        description="SLA deadline.",
    )
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Creation timestamp.",
    )
    updated_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Last update timestamp.",
    )
    resolved_at: Optional[datetime] = Field(
        default=None,
        description="Resolution timestamp.",
    )
    response: str = Field(
        default="",
        max_length=8192,
        description="Response to auditor.",
    )
    evidence_ids: List[str] = Field(
        default_factory=list,
        description="Linked evidence IDs.",
    )
    notes: str = Field(
        default="",
        max_length=4096,
        description="Internal notes.",
    )
    escalated: bool = Field(
        default=False,
        description="Whether escalated.",
    )
    escalation_reason: str = Field(
        default="",
        max_length=1024,
        description="Escalation reason.",
    )

    @property
    def is_overdue(self) -> bool:
        """Check if request is past SLA deadline."""
        if self.status in (RequestStatus.RESOLVED, RequestStatus.CLOSED):
            return False
        return datetime.now(timezone.utc) > self.sla_deadline

    @property
    def time_to_sla(self) -> timedelta:
        """Time remaining until SLA deadline."""
        return self.sla_deadline - datetime.now(timezone.utc)


class RequestComment(BaseModel):
    """Comment on an auditor request.

    Attributes:
        comment_id: Unique comment identifier.
        request_id: Associated request ID.
        author_id: Comment author ID.
        author_type: Whether author is auditor or internal.
        content: Comment content.
        created_at: Comment timestamp.
        attachments: Attachment IDs.
    """

    model_config = ConfigDict(extra="forbid")

    comment_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique comment identifier.",
    )
    request_id: str = Field(
        ...,
        description="Associated request ID.",
    )
    author_id: str = Field(
        ...,
        description="Comment author ID.",
    )
    author_type: str = Field(
        default="internal",
        description="Author type: 'auditor' or 'internal'.",
    )
    content: str = Field(
        ...,
        min_length=1,
        max_length=4096,
        description="Comment content.",
    )
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Comment timestamp.",
    )
    attachments: List[str] = Field(
        default_factory=list,
        description="Attachment IDs.",
    )


# ---------------------------------------------------------------------------
# Auditor Request Handler
# ---------------------------------------------------------------------------


class AuditorRequestHandler:
    """Handle auditor information requests with SLA tracking.

    Manages the complete request lifecycle including creation, assignment,
    status updates, resolution, and escalation.

    Attributes:
        _requests: Stored requests by request_id.
        _comments: Comments by request_id.
        _escalation_handlers: Callback handlers for escalations.
    """

    def __init__(self) -> None:
        """Initialize the request handler."""
        self._requests: Dict[str, AuditorRequest] = {}
        self._comments: Dict[str, List[RequestComment]] = {}
        self._escalation_handlers: List[Any] = []
        logger.info("AuditorRequestHandler initialized")

    # ------------------------------------------------------------------
    # Request Creation
    # ------------------------------------------------------------------

    async def create_request(
        self,
        auditor_id: uuid.UUID,
        request: RequestCreate,
    ) -> AuditorRequest:
        """Create a new auditor request.

        Args:
            auditor_id: ID of the requesting auditor.
            request: Request details.

        Returns:
            Created AuditorRequest with SLA deadline.
        """
        auditor_id_str = str(auditor_id)

        # Calculate SLA deadline
        sla_hours = SLA_HOURS[request.priority.value]
        sla_deadline = datetime.now(timezone.utc) + timedelta(hours=sla_hours)

        auditor_request = AuditorRequest(
            auditor_id=auditor_id_str,
            title=request.title,
            description=request.description,
            priority=request.priority,
            category=request.category,
            related_criterion=request.related_criterion,
            sla_deadline=sla_deadline,
        )

        self._requests[auditor_request.request_id] = auditor_request
        self._comments[auditor_request.request_id] = []

        logger.info(
            "Created request %s (priority=%s, sla=%s)",
            auditor_request.request_id[:8],
            request.priority.value,
            sla_deadline.isoformat(),
        )

        return auditor_request

    # ------------------------------------------------------------------
    # Assignment
    # ------------------------------------------------------------------

    async def assign_request(
        self,
        request_id: uuid.UUID,
        assignee_id: uuid.UUID,
    ) -> AuditorRequest:
        """Assign a request to a team member.

        Args:
            request_id: Request identifier.
            assignee_id: Team member to assign.

        Returns:
            Updated AuditorRequest.

        Raises:
            ValueError: If request not found.
        """
        request_id_str = str(request_id)
        request = self._requests.get(request_id_str)

        if request is None:
            raise ValueError(f"Request {request_id_str} not found")

        request.assignee_id = str(assignee_id)
        request.status = RequestStatus.ASSIGNED
        request.updated_at = datetime.now(timezone.utc)

        logger.info(
            "Assigned request %s to %s",
            request_id_str[:8],
            str(assignee_id)[:8],
        )

        return request

    async def reassign_request(
        self,
        request_id: uuid.UUID,
        new_assignee_id: uuid.UUID,
        reason: str = "",
    ) -> AuditorRequest:
        """Reassign a request to a different team member.

        Args:
            request_id: Request identifier.
            new_assignee_id: New team member.
            reason: Reason for reassignment.

        Returns:
            Updated AuditorRequest.
        """
        request_id_str = str(request_id)
        request = self._requests.get(request_id_str)

        if request is None:
            raise ValueError(f"Request {request_id_str} not found")

        old_assignee = request.assignee_id
        request.assignee_id = str(new_assignee_id)
        request.updated_at = datetime.now(timezone.utc)

        if reason:
            request.notes += f"\nReassigned from {old_assignee} to {new_assignee_id}: {reason}"

        logger.info(
            "Reassigned request %s from %s to %s",
            request_id_str[:8],
            old_assignee[:8] if old_assignee else "unassigned",
            str(new_assignee_id)[:8],
        )

        return request

    # ------------------------------------------------------------------
    # Status Updates
    # ------------------------------------------------------------------

    async def update_status(
        self,
        request_id: uuid.UUID,
        status: str,
        notes: str = "",
    ) -> AuditorRequest:
        """Update request status.

        Args:
            request_id: Request identifier.
            status: New status value.
            notes: Optional notes for the update.

        Returns:
            Updated AuditorRequest.

        Raises:
            ValueError: If request not found or invalid status.
        """
        request_id_str = str(request_id)
        request = self._requests.get(request_id_str)

        if request is None:
            raise ValueError(f"Request {request_id_str} not found")

        try:
            new_status = RequestStatus(status)
        except ValueError:
            raise ValueError(f"Invalid status: {status}")

        old_status = request.status
        request.status = new_status
        request.updated_at = datetime.now(timezone.utc)

        if notes:
            request.notes += f"\n[{old_status.value} -> {new_status.value}] {notes}"

        logger.info(
            "Updated request %s status: %s -> %s",
            request_id_str[:8],
            old_status.value,
            new_status.value,
        )

        return request

    # ------------------------------------------------------------------
    # Resolution
    # ------------------------------------------------------------------

    async def resolve_request(
        self,
        request_id: uuid.UUID,
        evidence_ids: List[uuid.UUID],
        notes: str,
    ) -> AuditorRequest:
        """Resolve a request with evidence and notes.

        Args:
            request_id: Request identifier.
            evidence_ids: Evidence items linked to response.
            notes: Response notes for auditor.

        Returns:
            Resolved AuditorRequest.

        Raises:
            ValueError: If request not found.
        """
        request_id_str = str(request_id)
        request = self._requests.get(request_id_str)

        if request is None:
            raise ValueError(f"Request {request_id_str} not found")

        request.status = RequestStatus.RESOLVED
        request.evidence_ids = [str(eid) for eid in evidence_ids]
        request.response = notes
        request.resolved_at = datetime.now(timezone.utc)
        request.updated_at = datetime.now(timezone.utc)

        # Calculate response time
        response_time = request.resolved_at - request.created_at
        met_sla = request.resolved_at <= request.sla_deadline

        logger.info(
            "Resolved request %s (response_time=%s, met_sla=%s)",
            request_id_str[:8],
            str(response_time),
            met_sla,
        )

        return request

    async def close_request(
        self,
        request_id: uuid.UUID,
        auditor_id: uuid.UUID,
    ) -> AuditorRequest:
        """Close a resolved request (auditor action).

        Args:
            request_id: Request identifier.
            auditor_id: Auditor closing the request.

        Returns:
            Closed AuditorRequest.
        """
        request_id_str = str(request_id)
        request = self._requests.get(request_id_str)

        if request is None:
            raise ValueError(f"Request {request_id_str} not found")

        if request.auditor_id != str(auditor_id):
            raise ValueError("Only the requesting auditor can close this request")

        if request.status != RequestStatus.RESOLVED:
            raise ValueError("Only resolved requests can be closed")

        request.status = RequestStatus.CLOSED
        request.updated_at = datetime.now(timezone.utc)

        logger.info("Closed request %s", request_id_str[:8])
        return request

    async def reopen_request(
        self,
        request_id: uuid.UUID,
        reason: str,
    ) -> AuditorRequest:
        """Reopen a resolved/closed request.

        Args:
            request_id: Request identifier.
            reason: Reason for reopening.

        Returns:
            Reopened AuditorRequest.
        """
        request_id_str = str(request_id)
        request = self._requests.get(request_id_str)

        if request is None:
            raise ValueError(f"Request {request_id_str} not found")

        if request.status not in (RequestStatus.RESOLVED, RequestStatus.CLOSED):
            raise ValueError("Only resolved or closed requests can be reopened")

        request.status = RequestStatus.IN_PROGRESS
        request.updated_at = datetime.now(timezone.utc)
        request.resolved_at = None
        request.notes += f"\nReopened: {reason}"

        # Extend SLA by the original duration
        sla_hours = SLA_HOURS[request.priority.value]
        request.sla_deadline = datetime.now(timezone.utc) + timedelta(hours=sla_hours)

        logger.info("Reopened request %s: %s", request_id_str[:8], reason)
        return request

    # ------------------------------------------------------------------
    # Escalation
    # ------------------------------------------------------------------

    async def escalate_request(
        self,
        request_id: uuid.UUID,
        reason: str = "",
    ) -> AuditorRequest:
        """Escalate a request to management.

        Args:
            request_id: Request identifier.
            reason: Reason for escalation.

        Returns:
            Escalated AuditorRequest.
        """
        request_id_str = str(request_id)
        request = self._requests.get(request_id_str)

        if request is None:
            raise ValueError(f"Request {request_id_str} not found")

        request.escalated = True
        request.escalation_reason = reason or "Escalated per request"
        request.status = RequestStatus.ESCALATED
        request.updated_at = datetime.now(timezone.utc)

        # Notify escalation handlers
        for handler in self._escalation_handlers:
            try:
                await handler(request)
            except Exception as exc:
                logger.error("Escalation handler failed: %s", exc)

        logger.warning(
            "Escalated request %s: %s",
            request_id_str[:8],
            request.escalation_reason,
        )

        return request

    def register_escalation_handler(self, handler: Any) -> None:
        """Register a callback for escalation events.

        Args:
            handler: Async callable that receives the escalated request.
        """
        self._escalation_handlers.append(handler)

    # ------------------------------------------------------------------
    # SLA Management
    # ------------------------------------------------------------------

    def _calculate_sla(self, priority: RequestPriority) -> datetime:
        """Calculate SLA deadline based on priority.

        Args:
            priority: Request priority.

        Returns:
            SLA deadline datetime.
        """
        hours = SLA_HOURS[priority.value]
        return datetime.now(timezone.utc) + timedelta(hours=hours)

    async def get_overdue_requests(self) -> List[AuditorRequest]:
        """Get all requests that are past their SLA deadline.

        Returns:
            List of overdue requests.
        """
        now = datetime.now(timezone.utc)
        overdue = [
            r for r in self._requests.values()
            if r.status not in (RequestStatus.RESOLVED, RequestStatus.CLOSED)
            and r.sla_deadline < now
        ]

        return sorted(overdue, key=lambda r: r.sla_deadline)

    async def get_at_risk_requests(self, hours: int = 4) -> List[AuditorRequest]:
        """Get requests that will be overdue within specified hours.

        Args:
            hours: Hours until SLA deadline.

        Returns:
            List of at-risk requests.
        """
        threshold = datetime.now(timezone.utc) + timedelta(hours=hours)
        at_risk = [
            r for r in self._requests.values()
            if r.status not in (RequestStatus.RESOLVED, RequestStatus.CLOSED)
            and r.sla_deadline <= threshold
            and r.sla_deadline > datetime.now(timezone.utc)
        ]

        return sorted(at_risk, key=lambda r: r.sla_deadline)

    # ------------------------------------------------------------------
    # Comments
    # ------------------------------------------------------------------

    async def add_comment(
        self,
        request_id: uuid.UUID,
        author_id: uuid.UUID,
        content: str,
        author_type: str = "internal",
        attachments: Optional[List[str]] = None,
    ) -> RequestComment:
        """Add a comment to a request.

        Args:
            request_id: Request identifier.
            author_id: Comment author ID.
            content: Comment content.
            author_type: 'auditor' or 'internal'.
            attachments: Optional attachment IDs.

        Returns:
            Created RequestComment.
        """
        request_id_str = str(request_id)

        if request_id_str not in self._requests:
            raise ValueError(f"Request {request_id_str} not found")

        comment = RequestComment(
            request_id=request_id_str,
            author_id=str(author_id),
            author_type=author_type,
            content=content,
            attachments=attachments or [],
        )

        self._comments[request_id_str].append(comment)

        # Update request timestamp
        self._requests[request_id_str].updated_at = datetime.now(timezone.utc)

        logger.debug(
            "Added comment to request %s by %s",
            request_id_str[:8],
            str(author_id)[:8],
        )

        return comment

    async def get_comments(self, request_id: uuid.UUID) -> List[RequestComment]:
        """Get all comments for a request.

        Args:
            request_id: Request identifier.

        Returns:
            List of comments, oldest first.
        """
        request_id_str = str(request_id)
        return self._comments.get(request_id_str, [])

    # ------------------------------------------------------------------
    # Query Methods
    # ------------------------------------------------------------------

    async def get_request(self, request_id: uuid.UUID) -> Optional[AuditorRequest]:
        """Get a request by ID.

        Args:
            request_id: Request identifier.

        Returns:
            AuditorRequest if found.
        """
        return self._requests.get(str(request_id))

    async def list_requests(
        self,
        auditor_id: Optional[uuid.UUID] = None,
        assignee_id: Optional[uuid.UUID] = None,
        status: Optional[RequestStatus] = None,
        priority: Optional[RequestPriority] = None,
        category: Optional[RequestCategory] = None,
        limit: int = 50,
        offset: int = 0,
    ) -> List[AuditorRequest]:
        """List requests with optional filtering.

        Args:
            auditor_id: Filter by auditor.
            assignee_id: Filter by assignee.
            status: Filter by status.
            priority: Filter by priority.
            category: Filter by category.
            limit: Maximum results.
            offset: Results offset.

        Returns:
            List of matching requests.
        """
        requests = list(self._requests.values())

        if auditor_id:
            requests = [r for r in requests if r.auditor_id == str(auditor_id)]

        if assignee_id:
            requests = [r for r in requests if r.assignee_id == str(assignee_id)]

        if status:
            requests = [r for r in requests if r.status == status]

        if priority:
            requests = [r for r in requests if r.priority == priority]

        if category:
            requests = [r for r in requests if r.category == category]

        # Sort by priority and creation time
        priority_order = {
            RequestPriority.CRITICAL: 0,
            RequestPriority.HIGH: 1,
            RequestPriority.NORMAL: 2,
            RequestPriority.LOW: 3,
        }
        requests.sort(key=lambda r: (priority_order[r.priority], r.created_at))

        return requests[offset : offset + limit]

    async def get_statistics(self) -> Dict[str, Any]:
        """Get request statistics.

        Returns:
            Dictionary with request metrics.
        """
        requests = list(self._requests.values())

        by_status: Dict[str, int] = {}
        by_priority: Dict[str, int] = {}
        overdue_count = 0
        avg_response_time_hours = 0.0
        resolved_count = 0

        for request in requests:
            by_status[request.status.value] = by_status.get(request.status.value, 0) + 1
            by_priority[request.priority.value] = by_priority.get(request.priority.value, 0) + 1

            if request.is_overdue:
                overdue_count += 1

            if request.resolved_at:
                resolved_count += 1
                response_time = (request.resolved_at - request.created_at).total_seconds() / 3600
                avg_response_time_hours += response_time

        if resolved_count > 0:
            avg_response_time_hours /= resolved_count

        return {
            "total_requests": len(requests),
            "by_status": by_status,
            "by_priority": by_priority,
            "overdue_count": overdue_count,
            "avg_response_time_hours": round(avg_response_time_hours, 2),
            "sla_met_rate": self._calculate_sla_met_rate(requests),
        }

    def _calculate_sla_met_rate(self, requests: List[AuditorRequest]) -> float:
        """Calculate percentage of requests resolved within SLA.

        Args:
            requests: List of requests.

        Returns:
            SLA met rate as percentage.
        """
        resolved = [r for r in requests if r.resolved_at is not None]
        if not resolved:
            return 100.0

        met_sla = sum(1 for r in resolved if r.resolved_at <= r.sla_deadline)
        return round((met_sla / len(resolved)) * 100, 1)


__all__ = [
    "AuditorRequestHandler",
    "AuditorRequest",
    "RequestCreate",
    "RequestComment",
    "RequestPriority",
    "RequestStatus",
    "RequestCategory",
]
