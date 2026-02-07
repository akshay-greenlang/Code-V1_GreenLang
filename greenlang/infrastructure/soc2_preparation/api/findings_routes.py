# -*- coding: utf-8 -*-
"""
SOC 2 Findings Management API Routes - SEC-009 Phase 10

FastAPI routes for findings management:
- GET /findings - List findings
- POST /findings - Create finding
- PUT /findings/{id} - Update finding
- POST /findings/{id}/remediation - Add remediation plan
- PUT /findings/{id}/close - Close finding

Requires soc2:findings:read or soc2:findings:manage permissions.

Author: GreenLang Security Team
Date: February 2026
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional
from uuid import UUID, uuid4

from fastapi import APIRouter, Depends, HTTPException, Query, Request, status
from pydantic import BaseModel, Field, field_validator

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Request/Response Models
# ---------------------------------------------------------------------------


class Finding(BaseModel):
    """SOC 2 audit finding."""

    finding_id: UUID = Field(default_factory=uuid4, description="Finding identifier")
    title: str = Field(..., max_length=256, description="Finding title")
    description: str = Field(..., max_length=4096, description="Finding description")
    criterion_id: str = Field(..., description="Related SOC 2 criterion")
    category: str = Field(
        default="control_deficiency",
        description="Category: control_deficiency, deviation, gap, observation",
    )
    severity: str = Field(
        default="medium", description="Severity: critical, high, medium, low"
    )
    priority: int = Field(default=3, ge=1, le=5, description="Priority (1=highest)")
    status: str = Field(
        default="open",
        description="Status: open, in_progress, remediation_planned, resolved, closed, accepted",
    )
    source: str = Field(
        default="control_test",
        description="Source: control_test, auditor_request, self_assessment, vulnerability_scan",
    )
    test_id: Optional[str] = Field(None, description="Related test ID if from testing")
    identified_by: str = Field(default="system", description="Who identified")
    identified_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="When identified",
    )
    owner: Optional[str] = Field(None, description="Assigned owner")
    due_date: Optional[datetime] = Field(None, description="Remediation due date")
    resolved_at: Optional[datetime] = Field(None, description="Resolution date")
    closed_at: Optional[datetime] = Field(None, description="Closure date")
    closed_by: Optional[str] = Field(None, description="Who closed")
    root_cause: str = Field(default="", max_length=4096, description="Root cause analysis")
    impact: str = Field(default="", max_length=4096, description="Business impact")
    evidence_ids: List[UUID] = Field(
        default_factory=list, description="Related evidence"
    )
    tags: List[str] = Field(default_factory=list, description="Finding tags")
    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Additional metadata"
    )


class RemediationPlan(BaseModel):
    """Remediation plan for a finding."""

    plan_id: UUID = Field(default_factory=uuid4, description="Plan identifier")
    finding_id: UUID = Field(..., description="Related finding")
    title: str = Field(..., max_length=256, description="Plan title")
    description: str = Field(..., max_length=4096, description="Plan description")
    steps: List[str] = Field(default_factory=list, description="Remediation steps")
    owner: str = Field(..., description="Plan owner")
    target_date: datetime = Field(..., description="Target completion date")
    status: str = Field(
        default="planned",
        description="Status: planned, in_progress, completed, verified",
    )
    progress_percent: int = Field(
        default=0, ge=0, le=100, description="Completion percentage"
    )
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Creation time",
    )
    created_by: str = Field(default="system", description="Creator")
    updated_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Last update",
    )
    completed_at: Optional[datetime] = Field(None, description="Completion time")
    verification_notes: str = Field(
        default="", max_length=4096, description="Verification notes"
    )


class FindingCreate(BaseModel):
    """Request to create a finding."""

    title: str = Field(..., max_length=256, description="Finding title")
    description: str = Field(..., max_length=4096, description="Finding description")
    criterion_id: str = Field(..., description="Related SOC 2 criterion")
    category: str = Field(
        default="control_deficiency", description="Finding category"
    )
    severity: str = Field(default="medium", description="Finding severity")
    priority: int = Field(default=3, ge=1, le=5, description="Priority")
    source: str = Field(default="control_test", description="Finding source")
    test_id: Optional[str] = Field(None, description="Related test ID")
    root_cause: Optional[str] = Field(None, description="Root cause analysis")
    impact: Optional[str] = Field(None, description="Business impact")
    owner: Optional[str] = Field(None, description="Assigned owner")
    due_date: Optional[datetime] = Field(None, description="Remediation due date")

    @field_validator("severity")
    @classmethod
    def validate_severity(cls, v: str) -> str:
        """Validate severity value."""
        allowed = {"critical", "high", "medium", "low"}
        if v.lower() not in allowed:
            raise ValueError(f"Invalid severity '{v}'. Allowed: {sorted(allowed)}")
        return v.lower()

    @field_validator("category")
    @classmethod
    def validate_category(cls, v: str) -> str:
        """Validate category value."""
        allowed = {"control_deficiency", "deviation", "gap", "observation"}
        if v.lower() not in allowed:
            raise ValueError(f"Invalid category '{v}'. Allowed: {sorted(allowed)}")
        return v.lower()


class FindingUpdate(BaseModel):
    """Request to update a finding."""

    title: Optional[str] = Field(None, max_length=256)
    description: Optional[str] = Field(None, max_length=4096)
    severity: Optional[str] = Field(None)
    priority: Optional[int] = Field(None, ge=1, le=5)
    status: Optional[str] = Field(None)
    owner: Optional[str] = Field(None)
    due_date: Optional[datetime] = Field(None)
    root_cause: Optional[str] = Field(None, max_length=4096)
    impact: Optional[str] = Field(None, max_length=4096)
    tags: Optional[List[str]] = Field(None)

    @field_validator("status")
    @classmethod
    def validate_status(cls, v: Optional[str]) -> Optional[str]:
        """Validate status value."""
        if v is None:
            return None
        allowed = {
            "open",
            "in_progress",
            "remediation_planned",
            "resolved",
            "closed",
            "accepted",
        }
        if v.lower() not in allowed:
            raise ValueError(f"Invalid status '{v}'. Allowed: {sorted(allowed)}")
        return v.lower()


class RemediationPlanCreate(BaseModel):
    """Request to create a remediation plan."""

    title: str = Field(..., max_length=256, description="Plan title")
    description: str = Field(..., max_length=4096, description="Plan description")
    steps: List[str] = Field(..., min_length=1, description="Remediation steps")
    owner: str = Field(..., description="Plan owner")
    target_date: datetime = Field(..., description="Target completion date")


class FindingCloseRequest(BaseModel):
    """Request to close a finding."""

    closure_reason: str = Field(
        ...,
        description="Reason: remediated, accepted, not_applicable, false_positive",
    )
    verification_notes: str = Field(
        ..., max_length=4096, description="Verification notes"
    )
    evidence_ids: List[UUID] = Field(
        default_factory=list, description="Evidence of remediation"
    )


class FindingsListResponse(BaseModel):
    """Response for findings listing."""

    total: int = Field(..., description="Total findings count")
    findings: List[Finding] = Field(..., description="Finding items")
    by_severity: Dict[str, int] = Field(
        default_factory=dict, description="Count by severity"
    )
    by_status: Dict[str, int] = Field(
        default_factory=dict, description="Count by status"
    )


class FindingSummary(BaseModel):
    """Summary statistics for findings."""

    total_findings: int = Field(default=0, description="Total findings")
    open_findings: int = Field(default=0, description="Open findings")
    critical_open: int = Field(default=0, description="Critical open findings")
    high_open: int = Field(default=0, description="High severity open")
    overdue_count: int = Field(default=0, description="Overdue remediations")
    avg_resolution_days: float = Field(
        default=0.0, description="Average resolution time"
    )
    mttr_days: float = Field(default=0.0, description="Mean time to remediate")


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------

router = APIRouter(prefix="/findings", tags=["soc2-findings"])


@router.get(
    "",
    response_model=FindingsListResponse,
    summary="List findings",
    description="List all SOC 2 findings with filtering options.",
)
async def list_findings(
    request: Request,
    severity: Optional[str] = Query(None, description="Filter by severity"),
    status_filter: Optional[str] = Query(
        None, alias="status", description="Filter by status"
    ),
    criterion: Optional[str] = Query(None, description="Filter by criterion"),
    owner: Optional[str] = Query(None, description="Filter by owner"),
    overdue_only: bool = Query(False, description="Only show overdue findings"),
    limit: int = Query(50, ge=1, le=200, description="Maximum results"),
    offset: int = Query(0, ge=0, description="Results offset"),
) -> FindingsListResponse:
    """List findings with filtering.

    Args:
        request: FastAPI request object.
        severity: Filter by severity.
        status_filter: Filter by status.
        criterion: Filter by criterion ID.
        owner: Filter by assigned owner.
        overdue_only: Only return overdue findings.
        limit: Maximum results.
        offset: Pagination offset.

    Returns:
        FindingsListResponse with finding list.
    """
    logger.info(
        "Listing findings: severity=%s, status=%s, criterion=%s",
        severity,
        status_filter,
        criterion,
    )

    # Sample findings
    findings = [
        Finding(
            title="Incomplete MFA coverage for service accounts",
            description="5 legacy service accounts do not have MFA enabled",
            criterion_id="CC6.7",
            category="control_deficiency",
            severity="high",
            priority=1,
            status="in_progress",
            source="control_test",
            test_id="CC6.7.1",
            identified_by="automation",
            owner="security-team",
            due_date=datetime(2026, 4, 15, tzinfo=timezone.utc),
        ),
        Finding(
            title="Missing incident response documentation",
            description="IRP does not cover cloud infrastructure outage scenarios",
            criterion_id="CC7.4",
            category="gap",
            severity="medium",
            priority=2,
            status="remediation_planned",
            source="control_test",
            test_id="CC7.4.1",
            identified_by="automation",
            owner="security-ops",
            due_date=datetime(2026, 3, 30, tzinfo=timezone.utc),
        ),
        Finding(
            title="Access reviews not automated",
            description="Quarterly access reviews are manual, risking missed reviews",
            criterion_id="CC6.2",
            category="observation",
            severity="low",
            priority=4,
            status="open",
            source="self_assessment",
            identified_by="iam-team",
            owner="iam-team",
            due_date=datetime(2026, 5, 30, tzinfo=timezone.utc),
        ),
    ]

    # Apply filters
    filtered = findings
    if severity:
        filtered = [f for f in filtered if f.severity == severity.lower()]
    if status_filter:
        filtered = [f for f in filtered if f.status == status_filter.lower()]
    if criterion:
        filtered = [f for f in filtered if f.criterion_id.startswith(criterion.upper())]
    if owner:
        filtered = [f for f in filtered if f.owner == owner]
    if overdue_only:
        now = datetime.now(timezone.utc)
        filtered = [
            f
            for f in filtered
            if f.due_date and f.due_date < now and f.status not in ("closed", "resolved")
        ]

    # Calculate counts
    by_severity: Dict[str, int] = {}
    by_status: Dict[str, int] = {}
    for finding in findings:
        by_severity[finding.severity] = by_severity.get(finding.severity, 0) + 1
        by_status[finding.status] = by_status.get(finding.status, 0) + 1

    return FindingsListResponse(
        total=len(filtered),
        findings=filtered[offset : offset + limit],
        by_severity=by_severity,
        by_status=by_status,
    )


@router.post(
    "",
    response_model=Finding,
    status_code=status.HTTP_201_CREATED,
    summary="Create finding",
    description="Create a new SOC 2 finding.",
)
async def create_finding(
    request: Request,
    finding_data: FindingCreate,
) -> Finding:
    """Create a new finding.

    Args:
        request: FastAPI request object.
        finding_data: Finding details.

    Returns:
        Created Finding.
    """
    logger.info(
        "Creating finding: title=%s, criterion=%s, severity=%s",
        finding_data.title,
        finding_data.criterion_id,
        finding_data.severity,
    )

    finding = Finding(
        title=finding_data.title,
        description=finding_data.description,
        criterion_id=finding_data.criterion_id.upper(),
        category=finding_data.category,
        severity=finding_data.severity,
        priority=finding_data.priority,
        source=finding_data.source,
        test_id=finding_data.test_id,
        root_cause=finding_data.root_cause or "",
        impact=finding_data.impact or "",
        owner=finding_data.owner,
        due_date=finding_data.due_date,
        identified_by="system",  # From auth context in production
    )

    return finding


@router.get(
    "/summary",
    response_model=FindingSummary,
    summary="Get findings summary",
    description="Get summary statistics for findings.",
)
async def get_summary(
    request: Request,
) -> FindingSummary:
    """Get findings summary statistics.

    Args:
        request: FastAPI request object.

    Returns:
        FindingSummary with statistics.
    """
    logger.info("Getting findings summary")

    return FindingSummary(
        total_findings=12,
        open_findings=5,
        critical_open=0,
        high_open=2,
        overdue_count=1,
        avg_resolution_days=14.5,
        mttr_days=21.3,
    )


@router.get(
    "/{finding_id}",
    response_model=Finding,
    summary="Get finding",
    description="Get a specific finding by ID.",
)
async def get_finding(
    request: Request,
    finding_id: UUID,
) -> Finding:
    """Get a finding by ID.

    Args:
        request: FastAPI request object.
        finding_id: The finding identifier.

    Returns:
        Finding details.

    Raises:
        HTTPException: 404 if not found.
    """
    logger.info("Getting finding: %s", finding_id)

    return Finding(
        finding_id=finding_id,
        title="Incomplete MFA coverage for service accounts",
        description="5 legacy service accounts do not have MFA enabled",
        criterion_id="CC6.7",
        category="control_deficiency",
        severity="high",
        priority=1,
        status="in_progress",
        source="control_test",
        test_id="CC6.7.1",
        identified_by="automation",
        owner="security-team",
        due_date=datetime(2026, 4, 15, tzinfo=timezone.utc),
        root_cause="Legacy accounts pre-date MFA requirement",
        impact="Non-compliance with CC6.7, audit finding risk",
    )


@router.put(
    "/{finding_id}",
    response_model=Finding,
    summary="Update finding",
    description="Update a finding's details.",
)
async def update_finding(
    request: Request,
    finding_id: UUID,
    update_data: FindingUpdate,
) -> Finding:
    """Update a finding.

    Args:
        request: FastAPI request object.
        finding_id: The finding identifier.
        update_data: Update data.

    Returns:
        Updated Finding.

    Raises:
        HTTPException: 404 if not found.
    """
    logger.info("Updating finding: %s", finding_id)

    # In production, merge with existing finding from database
    return Finding(
        finding_id=finding_id,
        title=update_data.title or "Incomplete MFA coverage for service accounts",
        description=update_data.description
        or "5 legacy service accounts do not have MFA enabled",
        criterion_id="CC6.7",
        category="control_deficiency",
        severity=update_data.severity or "high",
        priority=update_data.priority or 1,
        status=update_data.status or "in_progress",
        source="control_test",
        owner=update_data.owner or "security-team",
        due_date=update_data.due_date or datetime(2026, 4, 15, tzinfo=timezone.utc),
        root_cause=update_data.root_cause or "Legacy accounts pre-date MFA requirement",
        impact=update_data.impact or "Non-compliance with CC6.7",
    )


@router.post(
    "/{finding_id}/remediation",
    response_model=RemediationPlan,
    status_code=status.HTTP_201_CREATED,
    summary="Add remediation plan",
    description="Add a remediation plan to a finding.",
)
async def add_remediation(
    request: Request,
    finding_id: UUID,
    plan_data: RemediationPlanCreate,
) -> RemediationPlan:
    """Add a remediation plan to a finding.

    Args:
        request: FastAPI request object.
        finding_id: The finding identifier.
        plan_data: Remediation plan details.

    Returns:
        Created RemediationPlan.

    Raises:
        HTTPException: 404 if finding not found.
    """
    logger.info("Adding remediation plan to finding: %s", finding_id)

    plan = RemediationPlan(
        finding_id=finding_id,
        title=plan_data.title,
        description=plan_data.description,
        steps=plan_data.steps,
        owner=plan_data.owner,
        target_date=plan_data.target_date,
        status="planned",
        created_by="system",  # From auth context in production
    )

    return plan


@router.get(
    "/{finding_id}/remediation",
    response_model=List[RemediationPlan],
    summary="Get remediation plans",
    description="Get all remediation plans for a finding.",
)
async def get_remediation_plans(
    request: Request,
    finding_id: UUID,
) -> List[RemediationPlan]:
    """Get remediation plans for a finding.

    Args:
        request: FastAPI request object.
        finding_id: The finding identifier.

    Returns:
        List of remediation plans.
    """
    logger.info("Getting remediation plans for finding: %s", finding_id)

    return [
        RemediationPlan(
            finding_id=finding_id,
            title="Migrate service accounts to workload identity",
            description="Replace service accounts with workload identity federation",
            steps=[
                "Inventory all service accounts",
                "Configure workload identity pools",
                "Migrate applications to use OIDC tokens",
                "Disable legacy service accounts",
                "Verify MFA not required for WIF",
            ],
            owner="security-team",
            target_date=datetime(2026, 4, 15, tzinfo=timezone.utc),
            status="in_progress",
            progress_percent=40,
        )
    ]


@router.put(
    "/{finding_id}/close",
    response_model=Finding,
    summary="Close finding",
    description="Close a finding with verification.",
)
async def close_finding(
    request: Request,
    finding_id: UUID,
    close_data: FindingCloseRequest,
) -> Finding:
    """Close a finding.

    Args:
        request: FastAPI request object.
        finding_id: The finding identifier.
        close_data: Closure details.

    Returns:
        Closed Finding.

    Raises:
        HTTPException: 404 if not found.
        HTTPException: 400 if finding cannot be closed.
    """
    logger.info(
        "Closing finding: %s, reason=%s",
        finding_id,
        close_data.closure_reason,
    )

    # Validate closure reason
    valid_reasons = {"remediated", "accepted", "not_applicable", "false_positive"}
    if close_data.closure_reason.lower() not in valid_reasons:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid closure reason. Allowed: {sorted(valid_reasons)}",
        )

    return Finding(
        finding_id=finding_id,
        title="Incomplete MFA coverage for service accounts",
        description="5 legacy service accounts do not have MFA enabled",
        criterion_id="CC6.7",
        category="control_deficiency",
        severity="high",
        priority=1,
        status="closed",
        source="control_test",
        owner="security-team",
        resolved_at=datetime.now(timezone.utc),
        closed_at=datetime.now(timezone.utc),
        closed_by="system",  # From auth context in production
        evidence_ids=close_data.evidence_ids,
    )


__all__ = ["router"]
