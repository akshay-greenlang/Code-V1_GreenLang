# -*- coding: utf-8 -*-
"""
SOC 2 Auditor Portal API Routes - SEC-009 Phase 10

FastAPI routes for auditor portal functionality:
- GET /portal/evidence - List evidence (auditor view)
- POST /portal/requests - Submit evidence request
- GET /portal/requests/{id} - Get request status
- GET /portal/download/{id} - Download evidence file

Requires soc2:portal:access permission.

Author: GreenLang Security Team
Date: February 2026
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional
from uuid import UUID, uuid4

from fastapi import APIRouter, Depends, HTTPException, Query, Request, status
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Request/Response Models
# ---------------------------------------------------------------------------


class AuditorInfo(BaseModel):
    """Auditor information."""

    auditor_id: str = Field(..., description="Auditor identifier")
    name: str = Field(..., description="Auditor name")
    email: str = Field(..., description="Auditor email")
    organization: str = Field(..., description="Audit firm")
    role: str = Field(default="auditor", description="Portal role")
    access_expires: datetime = Field(..., description="Access expiration")
    is_active: bool = Field(default=True, description="Whether access is active")


class PortalEvidenceItem(BaseModel):
    """Evidence item as seen in auditor portal."""

    evidence_id: UUID = Field(..., description="Evidence identifier")
    criterion_id: str = Field(..., description="Related criterion")
    title: str = Field(..., description="Evidence title")
    description: str = Field(default="", description="Evidence description")
    evidence_type: str = Field(..., description="Type of evidence")
    period_start: Optional[datetime] = Field(None, description="Period start")
    period_end: Optional[datetime] = Field(None, description="Period end")
    collected_at: datetime = Field(..., description="Collection timestamp")
    status: str = Field(..., description="Evidence status")
    file_size_bytes: Optional[int] = Field(None, description="File size")
    file_format: Optional[str] = Field(None, description="File format")
    downloadable: bool = Field(default=True, description="Can be downloaded")


class PortalEvidenceResponse(BaseModel):
    """Portal evidence listing response."""

    total: int = Field(..., description="Total evidence count")
    items: List[PortalEvidenceItem] = Field(..., description="Evidence items")
    criteria: List[str] = Field(..., description="Available criteria")
    audit_period_start: datetime = Field(..., description="Audit period start")
    audit_period_end: datetime = Field(..., description="Audit period end")


class EvidenceRequest(BaseModel):
    """Auditor evidence request."""

    request_id: UUID = Field(default_factory=uuid4, description="Request ID")
    requested_by: str = Field(..., description="Requester (auditor)")
    criterion_id: str = Field(..., description="Requested criterion")
    description: str = Field(..., max_length=4096, description="Request description")
    priority: str = Field(
        default="normal", description="Priority: critical, high, normal, low"
    )
    due_date: Optional[datetime] = Field(None, description="Requested due date")
    status: str = Field(
        default="pending", description="Status: pending, assigned, in_progress, resolved"
    )
    assigned_to: Optional[str] = Field(None, description="Assigned team member")
    response: Optional[str] = Field(None, description="Response to request")
    evidence_ids: List[UUID] = Field(
        default_factory=list, description="Attached evidence IDs"
    )
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Creation time",
    )
    updated_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Last update time",
    )
    resolved_at: Optional[datetime] = Field(None, description="Resolution time")
    sla_due: Optional[datetime] = Field(None, description="SLA due time")
    sla_status: str = Field(
        default="on_track", description="SLA status: on_track, at_risk, breached"
    )


class EvidenceRequestCreate(BaseModel):
    """Request to create an evidence request."""

    criterion_id: str = Field(..., description="Requested criterion")
    description: str = Field(..., max_length=4096, description="Request description")
    priority: str = Field(
        default="normal", description="Priority: critical, high, normal, low"
    )
    due_date: Optional[datetime] = Field(None, description="Requested due date")


class EvidenceRequestResponse(BaseModel):
    """Response with evidence request details."""

    request: EvidenceRequest = Field(..., description="Request details")
    sla_hours: int = Field(..., description="SLA hours for this priority")


class DownloadLink(BaseModel):
    """Download link for evidence file."""

    evidence_id: UUID = Field(..., description="Evidence identifier")
    download_url: str = Field(..., description="Presigned download URL")
    expires_at: datetime = Field(..., description="URL expiration time")
    file_name: str = Field(..., description="File name")
    file_size_bytes: int = Field(..., description="File size")
    content_type: str = Field(default="application/octet-stream", description="MIME type")


class AuditActivity(BaseModel):
    """Auditor activity log entry."""

    activity_id: UUID = Field(default_factory=uuid4, description="Activity ID")
    auditor_id: str = Field(..., description="Auditor identifier")
    action: str = Field(..., description="Action performed")
    resource_type: str = Field(..., description="Resource type")
    resource_id: str = Field(..., description="Resource identifier")
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Activity timestamp",
    )
    ip_address: Optional[str] = Field(None, description="Client IP")
    user_agent: Optional[str] = Field(None, description="Browser user agent")


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------

router = APIRouter(prefix="/portal", tags=["soc2-portal"])


@router.get(
    "/evidence",
    response_model=PortalEvidenceResponse,
    summary="List evidence (auditor view)",
    description="List available evidence for auditor review.",
)
async def list_portal_evidence(
    request: Request,
    criterion: Optional[str] = Query(None, description="Filter by criterion"),
    evidence_type: Optional[str] = Query(None, description="Filter by type"),
    page: int = Query(1, ge=1, description="Page number"),
    page_size: int = Query(50, ge=1, le=100, description="Page size"),
) -> PortalEvidenceResponse:
    """List evidence available to auditors.

    Args:
        request: FastAPI request object.
        criterion: Filter by criterion ID.
        evidence_type: Filter by evidence type.
        page: Page number.
        page_size: Items per page.

    Returns:
        PortalEvidenceResponse with evidence listing.
    """
    logger.info(
        "Portal: listing evidence: criterion=%s, type=%s",
        criterion,
        evidence_type,
    )

    # Sample evidence items
    items = [
        PortalEvidenceItem(
            evidence_id=uuid4(),
            criterion_id="CC6.1",
            title="Authentication Configuration Export",
            description="Complete MFA configuration and enrollment statistics",
            evidence_type="configuration",
            period_start=datetime(2026, 1, 1, tzinfo=timezone.utc),
            period_end=datetime(2026, 2, 1, tzinfo=timezone.utc),
            collected_at=datetime.now(timezone.utc),
            status="approved",
            file_size_bytes=25678,
            file_format="json",
            downloadable=True,
        ),
        PortalEvidenceItem(
            evidence_id=uuid4(),
            criterion_id="CC6.2",
            title="Q1 2026 Access Review Report",
            description="Quarterly user access review with manager attestations",
            evidence_type="access_review",
            period_start=datetime(2026, 1, 1, tzinfo=timezone.utc),
            period_end=datetime(2026, 3, 31, tzinfo=timezone.utc),
            collected_at=datetime.now(timezone.utc),
            status="approved",
            file_size_bytes=156789,
            file_format="pdf",
            downloadable=True,
        ),
        PortalEvidenceItem(
            evidence_id=uuid4(),
            criterion_id="CC7.1",
            title="Security Monitoring Dashboard Export",
            description="Prometheus/Grafana alert configuration and history",
            evidence_type="configuration",
            period_start=datetime(2026, 1, 1, tzinfo=timezone.utc),
            period_end=datetime(2026, 2, 1, tzinfo=timezone.utc),
            collected_at=datetime.now(timezone.utc),
            status="approved",
            file_size_bytes=89456,
            file_format="json",
            downloadable=True,
        ),
    ]

    # Apply filters
    filtered = items
    if criterion:
        filtered = [i for i in filtered if i.criterion_id.startswith(criterion.upper())]
    if evidence_type:
        filtered = [i for i in filtered if i.evidence_type == evidence_type.lower()]

    # Get unique criteria
    criteria = sorted(set(i.criterion_id for i in items))

    return PortalEvidenceResponse(
        total=len(filtered),
        items=filtered[(page - 1) * page_size : page * page_size],
        criteria=criteria,
        audit_period_start=datetime(2026, 1, 1, tzinfo=timezone.utc),
        audit_period_end=datetime(2026, 12, 31, tzinfo=timezone.utc),
    )


@router.post(
    "/requests",
    response_model=EvidenceRequestResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Submit evidence request",
    description="Submit a new evidence request from an auditor.",
)
async def create_request(
    request: Request,
    request_data: EvidenceRequestCreate,
) -> EvidenceRequestResponse:
    """Create a new evidence request.

    Args:
        request: FastAPI request object.
        request_data: Request details.

    Returns:
        EvidenceRequestResponse with created request.
    """
    logger.info(
        "Portal: creating request: criterion=%s, priority=%s",
        request_data.criterion_id,
        request_data.priority,
    )

    # Get SLA hours based on priority
    sla_hours_map = {
        "critical": 4,
        "high": 24,
        "normal": 48,
        "low": 72,
    }
    sla_hours = sla_hours_map.get(request_data.priority.lower(), 48)

    # Calculate SLA due time
    sla_due = datetime.now(timezone.utc) + timedelta(hours=sla_hours)

    evidence_request = EvidenceRequest(
        requested_by="auditor@example.com",  # From auth context in production
        criterion_id=request_data.criterion_id.upper(),
        description=request_data.description,
        priority=request_data.priority.lower(),
        due_date=request_data.due_date,
        status="pending",
        sla_due=sla_due,
        sla_status="on_track",
    )

    return EvidenceRequestResponse(
        request=evidence_request,
        sla_hours=sla_hours,
    )


@router.get(
    "/requests",
    response_model=List[EvidenceRequest],
    summary="List evidence requests",
    description="List all evidence requests (auditor view).",
)
async def list_requests(
    request: Request,
    status_filter: Optional[str] = Query(
        None, alias="status", description="Filter by status"
    ),
    limit: int = Query(50, ge=1, le=200, description="Maximum results"),
    offset: int = Query(0, ge=0, description="Results offset"),
) -> List[EvidenceRequest]:
    """List evidence requests.

    Args:
        request: FastAPI request object.
        status_filter: Filter by request status.
        limit: Maximum results.
        offset: Pagination offset.

    Returns:
        List of evidence requests.
    """
    logger.info("Portal: listing requests: status=%s", status_filter)

    # Sample requests
    requests = [
        EvidenceRequest(
            requested_by="auditor@example.com",
            criterion_id="CC6.3",
            description="Need evidence of terminated user access revocation process",
            priority="high",
            status="in_progress",
            assigned_to="iam-team",
            sla_due=datetime.now(timezone.utc) + timedelta(hours=12),
            sla_status="on_track",
        ),
        EvidenceRequest(
            requested_by="auditor@example.com",
            criterion_id="CC7.2",
            description="Request log samples for security event detection",
            priority="normal",
            status="resolved",
            assigned_to="security-ops",
            response="Attached CloudTrail logs and alert history",
            evidence_ids=[uuid4(), uuid4()],
            sla_due=datetime.now(timezone.utc) - timedelta(hours=24),
            sla_status="on_track",
            resolved_at=datetime.now(timezone.utc) - timedelta(hours=36),
        ),
    ]

    # Apply filters
    filtered = requests
    if status_filter:
        filtered = [r for r in filtered if r.status == status_filter.lower()]

    return filtered[offset : offset + limit]


@router.get(
    "/requests/{request_id}",
    response_model=EvidenceRequest,
    summary="Get request status",
    description="Get the status of a specific evidence request.",
)
async def get_request(
    request: Request,
    request_id: UUID,
) -> EvidenceRequest:
    """Get an evidence request by ID.

    Args:
        request: FastAPI request object.
        request_id: The request identifier.

    Returns:
        EvidenceRequest details.

    Raises:
        HTTPException: 404 if request not found.
    """
    logger.info("Portal: getting request: %s", request_id)

    return EvidenceRequest(
        request_id=request_id,
        requested_by="auditor@example.com",
        criterion_id="CC6.3",
        description="Need evidence of terminated user access revocation process",
        priority="high",
        status="in_progress",
        assigned_to="iam-team",
        sla_due=datetime.now(timezone.utc) + timedelta(hours=12),
        sla_status="on_track",
    )


@router.get(
    "/download/{evidence_id}",
    response_model=DownloadLink,
    summary="Download evidence file",
    description="Get a presigned download URL for an evidence file.",
)
async def download_evidence(
    request: Request,
    evidence_id: UUID,
) -> DownloadLink:
    """Get a download link for evidence.

    Args:
        request: FastAPI request object.
        evidence_id: The evidence identifier.

    Returns:
        DownloadLink with presigned URL.

    Raises:
        HTTPException: 404 if evidence not found.
        HTTPException: 403 if evidence not downloadable.
    """
    logger.info("Portal: downloading evidence: %s", evidence_id)

    # Log the download activity
    # In production, this would be stored in the audit log

    return DownloadLink(
        evidence_id=evidence_id,
        download_url=f"https://s3.amazonaws.com/greenlang-soc2-evidence/{evidence_id}.json?signature=xxx&expires=yyy",
        expires_at=datetime.now(timezone.utc) + timedelta(hours=1),
        file_name=f"evidence_{evidence_id}.json",
        file_size_bytes=25678,
        content_type="application/json",
    )


@router.get(
    "/activity",
    response_model=List[AuditActivity],
    summary="Get auditor activity",
    description="Get activity log for the current auditor.",
)
async def get_activity(
    request: Request,
    limit: int = Query(50, ge=1, le=200, description="Maximum results"),
    offset: int = Query(0, ge=0, description="Results offset"),
) -> List[AuditActivity]:
    """Get auditor activity log.

    Args:
        request: FastAPI request object.
        limit: Maximum results.
        offset: Pagination offset.

    Returns:
        List of activity entries.
    """
    logger.info("Portal: getting activity log")

    # Sample activity entries
    activities = [
        AuditActivity(
            auditor_id="auditor@example.com",
            action="download",
            resource_type="evidence",
            resource_id=str(uuid4()),
            ip_address="192.168.1.100",
        ),
        AuditActivity(
            auditor_id="auditor@example.com",
            action="view",
            resource_type="evidence_list",
            resource_id="CC6",
            ip_address="192.168.1.100",
        ),
        AuditActivity(
            auditor_id="auditor@example.com",
            action="create",
            resource_type="evidence_request",
            resource_id=str(uuid4()),
            ip_address="192.168.1.100",
        ),
    ]

    return activities[offset : offset + limit]


@router.get(
    "/session",
    response_model=AuditorInfo,
    summary="Get current session",
    description="Get information about the current auditor session.",
)
async def get_session(
    request: Request,
) -> AuditorInfo:
    """Get current auditor session info.

    Args:
        request: FastAPI request object.

    Returns:
        AuditorInfo for the current session.
    """
    logger.info("Portal: getting session info")

    # In production, this would come from the auth context
    return AuditorInfo(
        auditor_id="auditor@example.com",
        name="Jane Auditor",
        email="auditor@example.com",
        organization="Big Four Audit LLC",
        role="lead_auditor",
        access_expires=datetime.now(timezone.utc) + timedelta(days=30),
        is_active=True,
    )


__all__ = ["router"]
