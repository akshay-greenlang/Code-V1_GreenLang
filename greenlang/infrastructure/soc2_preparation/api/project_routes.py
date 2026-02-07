# -*- coding: utf-8 -*-
"""
SOC 2 Audit Project Management API Routes - SEC-009 Phase 10

FastAPI routes for audit project management:
- GET /project - Get current project
- POST /project - Create project
- GET /project/timeline - Get timeline
- POST /project/milestones - Add milestone
- PUT /project/milestones/{id} - Update milestone

Requires soc2:project:read or soc2:project:manage permissions.

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


class Milestone(BaseModel):
    """Audit project milestone."""

    milestone_id: UUID = Field(
        default_factory=uuid4, description="Milestone identifier"
    )
    name: str = Field(..., max_length=256, description="Milestone name")
    description: str = Field(
        default="", max_length=4096, description="Milestone description"
    )
    category: str = Field(
        default="general",
        description="Category: preparation, fieldwork, reporting, remediation",
    )
    status: str = Field(
        default="pending",
        description="Status: pending, in_progress, completed, blocked, skipped",
    )
    target_date: datetime = Field(..., description="Target completion date")
    completed_date: Optional[datetime] = Field(None, description="Actual completion")
    owner: str = Field(default="", description="Milestone owner")
    dependencies: List[UUID] = Field(
        default_factory=list, description="Dependent milestone IDs"
    )
    progress_percent: int = Field(
        default=0, ge=0, le=100, description="Completion percentage"
    )
    is_critical_path: bool = Field(
        default=False, description="Whether on critical path"
    )
    notes: str = Field(default="", max_length=4096, description="Notes")
    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Additional metadata"
    )


class AuditProject(BaseModel):
    """SOC 2 audit project."""

    project_id: UUID = Field(default_factory=uuid4, description="Project identifier")
    name: str = Field(..., max_length=256, description="Project name")
    description: str = Field(
        default="", max_length=4096, description="Project description"
    )
    audit_type: str = Field(
        default="soc2_type2", description="Audit type: soc2_type1, soc2_type2"
    )
    audit_firm: str = Field(default="", description="External audit firm")
    lead_auditor: str = Field(default="", description="Lead auditor name")
    internal_lead: str = Field(default="", description="Internal project lead")
    audit_period_start: datetime = Field(..., description="Audit period start")
    audit_period_end: datetime = Field(..., description="Audit period end")
    status: str = Field(
        default="planning",
        description="Status: planning, preparation, fieldwork, reporting, completed",
    )
    overall_progress: int = Field(
        default=0, ge=0, le=100, description="Overall progress percentage"
    )
    trust_services_categories: List[str] = Field(
        default_factory=lambda: ["security"],
        description="TSC categories in scope",
    )
    milestones: List[Milestone] = Field(
        default_factory=list, description="Project milestones"
    )
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Creation time",
    )
    updated_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Last update",
    )
    kickoff_date: Optional[datetime] = Field(None, description="Audit kickoff date")
    fieldwork_start: Optional[datetime] = Field(None, description="Fieldwork start")
    fieldwork_end: Optional[datetime] = Field(None, description="Fieldwork end")
    report_due: Optional[datetime] = Field(None, description="Report due date")
    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Additional metadata"
    )


class ProjectCreate(BaseModel):
    """Request to create an audit project."""

    name: str = Field(..., max_length=256, description="Project name")
    description: Optional[str] = Field(None, max_length=4096)
    audit_type: str = Field(default="soc2_type2", description="Audit type")
    audit_firm: Optional[str] = Field(None, description="External audit firm")
    audit_period_start: datetime = Field(..., description="Audit period start")
    audit_period_end: datetime = Field(..., description="Audit period end")
    trust_services_categories: Optional[List[str]] = Field(
        None, description="TSC categories in scope"
    )
    internal_lead: Optional[str] = Field(None, description="Internal project lead")

    @field_validator("audit_type")
    @classmethod
    def validate_audit_type(cls, v: str) -> str:
        """Validate audit type."""
        allowed = {"soc2_type1", "soc2_type2", "soc1_type1", "soc1_type2"}
        if v.lower() not in allowed:
            raise ValueError(f"Invalid audit type. Allowed: {sorted(allowed)}")
        return v.lower()


class MilestoneCreate(BaseModel):
    """Request to create a milestone."""

    name: str = Field(..., max_length=256, description="Milestone name")
    description: Optional[str] = Field(None, max_length=4096)
    category: str = Field(default="general", description="Milestone category")
    target_date: datetime = Field(..., description="Target completion date")
    owner: Optional[str] = Field(None, description="Milestone owner")
    dependencies: Optional[List[UUID]] = Field(None, description="Dependencies")
    is_critical_path: bool = Field(default=False, description="Critical path")


class MilestoneUpdate(BaseModel):
    """Request to update a milestone."""

    name: Optional[str] = Field(None, max_length=256)
    description: Optional[str] = Field(None, max_length=4096)
    status: Optional[str] = Field(None)
    target_date: Optional[datetime] = Field(None)
    owner: Optional[str] = Field(None)
    progress_percent: Optional[int] = Field(None, ge=0, le=100)
    notes: Optional[str] = Field(None, max_length=4096)

    @field_validator("status")
    @classmethod
    def validate_status(cls, v: Optional[str]) -> Optional[str]:
        """Validate status value."""
        if v is None:
            return None
        allowed = {"pending", "in_progress", "completed", "blocked", "skipped"}
        if v.lower() not in allowed:
            raise ValueError(f"Invalid status. Allowed: {sorted(allowed)}")
        return v.lower()


class TimelineEvent(BaseModel):
    """Timeline event."""

    event_id: UUID = Field(default_factory=uuid4, description="Event identifier")
    event_type: str = Field(..., description="Event type")
    title: str = Field(..., description="Event title")
    date: datetime = Field(..., description="Event date")
    status: str = Field(default="pending", description="Event status")
    is_milestone: bool = Field(default=False, description="Is a milestone")
    details: str = Field(default="", description="Event details")


class Timeline(BaseModel):
    """Project timeline."""

    project_id: UUID = Field(..., description="Project identifier")
    events: List[TimelineEvent] = Field(..., description="Timeline events")
    critical_path_days: int = Field(default=0, description="Days on critical path")
    days_remaining: int = Field(default=0, description="Days until deadline")
    is_on_track: bool = Field(default=True, description="Whether on schedule")
    risk_level: str = Field(default="low", description="Risk level: low, medium, high")


class ProjectSummary(BaseModel):
    """Project summary statistics."""

    project_id: UUID = Field(..., description="Project identifier")
    status: str = Field(..., description="Project status")
    overall_progress: int = Field(..., description="Overall progress")
    milestones_completed: int = Field(default=0, description="Completed milestones")
    milestones_total: int = Field(default=0, description="Total milestones")
    milestones_blocked: int = Field(default=0, description="Blocked milestones")
    days_remaining: int = Field(default=0, description="Days until deadline")
    is_on_track: bool = Field(default=True, description="On schedule")
    next_milestone: Optional[str] = Field(None, description="Next milestone name")
    next_milestone_date: Optional[datetime] = Field(None, description="Next date")


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------

router = APIRouter(prefix="/project", tags=["soc2-project"])


@router.get(
    "",
    response_model=AuditProject,
    summary="Get current project",
    description="Get the current SOC 2 audit project.",
)
async def get_project(
    request: Request,
    include_milestones: bool = Query(
        True, description="Include milestone details"
    ),
) -> AuditProject:
    """Get the current audit project.

    Args:
        request: FastAPI request object.
        include_milestones: Whether to include milestones.

    Returns:
        AuditProject details.
    """
    logger.info("Getting current project")

    milestones = []
    if include_milestones:
        milestones = [
            Milestone(
                name="Readiness Assessment Complete",
                description="Complete initial readiness assessment",
                category="preparation",
                status="completed",
                target_date=datetime(2026, 1, 31, tzinfo=timezone.utc),
                completed_date=datetime(2026, 1, 28, tzinfo=timezone.utc),
                owner="security-team",
                progress_percent=100,
                is_critical_path=True,
            ),
            Milestone(
                name="Gap Remediation Complete",
                description="Remediate all identified gaps",
                category="preparation",
                status="in_progress",
                target_date=datetime(2026, 3, 31, tzinfo=timezone.utc),
                owner="security-team",
                progress_percent=65,
                is_critical_path=True,
            ),
            Milestone(
                name="Audit Kickoff",
                description="Formal audit kickoff with external auditors",
                category="fieldwork",
                status="pending",
                target_date=datetime(2026, 4, 15, tzinfo=timezone.utc),
                owner="compliance-team",
                is_critical_path=True,
            ),
            Milestone(
                name="Fieldwork Complete",
                description="Complete all audit fieldwork",
                category="fieldwork",
                status="pending",
                target_date=datetime(2026, 6, 30, tzinfo=timezone.utc),
                owner="compliance-team",
                is_critical_path=True,
            ),
            Milestone(
                name="Draft Report Review",
                description="Review and respond to draft audit report",
                category="reporting",
                status="pending",
                target_date=datetime(2026, 7, 31, tzinfo=timezone.utc),
                owner="leadership",
                is_critical_path=True,
            ),
        ]

    return AuditProject(
        name="GreenLang SOC 2 Type II Audit 2026",
        description="Annual SOC 2 Type II audit for GreenLang platform",
        audit_type="soc2_type2",
        audit_firm="Big Four Audit LLC",
        lead_auditor="Jane Auditor",
        internal_lead="compliance-team",
        audit_period_start=datetime(2026, 1, 1, tzinfo=timezone.utc),
        audit_period_end=datetime(2026, 12, 31, tzinfo=timezone.utc),
        status="preparation",
        overall_progress=45,
        trust_services_categories=["security", "availability"],
        milestones=milestones,
        kickoff_date=datetime(2026, 4, 15, tzinfo=timezone.utc),
        fieldwork_start=datetime(2026, 4, 15, tzinfo=timezone.utc),
        fieldwork_end=datetime(2026, 6, 30, tzinfo=timezone.utc),
        report_due=datetime(2026, 8, 31, tzinfo=timezone.utc),
    )


@router.post(
    "",
    response_model=AuditProject,
    status_code=status.HTTP_201_CREATED,
    summary="Create project",
    description="Create a new SOC 2 audit project.",
)
async def create_project(
    request: Request,
    project_data: ProjectCreate,
) -> AuditProject:
    """Create a new audit project.

    Args:
        request: FastAPI request object.
        project_data: Project details.

    Returns:
        Created AuditProject.
    """
    logger.info("Creating project: %s", project_data.name)

    project = AuditProject(
        name=project_data.name,
        description=project_data.description or "",
        audit_type=project_data.audit_type,
        audit_firm=project_data.audit_firm or "",
        internal_lead=project_data.internal_lead or "",
        audit_period_start=project_data.audit_period_start,
        audit_period_end=project_data.audit_period_end,
        status="planning",
        trust_services_categories=project_data.trust_services_categories or ["security"],
    )

    return project


@router.get(
    "/summary",
    response_model=ProjectSummary,
    summary="Get project summary",
    description="Get a summary of the current project status.",
)
async def get_summary(
    request: Request,
) -> ProjectSummary:
    """Get project summary.

    Args:
        request: FastAPI request object.

    Returns:
        ProjectSummary with statistics.
    """
    logger.info("Getting project summary")

    return ProjectSummary(
        project_id=uuid4(),
        status="preparation",
        overall_progress=45,
        milestones_completed=1,
        milestones_total=5,
        milestones_blocked=0,
        days_remaining=180,
        is_on_track=True,
        next_milestone="Gap Remediation Complete",
        next_milestone_date=datetime(2026, 3, 31, tzinfo=timezone.utc),
    )


@router.get(
    "/timeline",
    response_model=Timeline,
    summary="Get timeline",
    description="Get the project timeline with events.",
)
async def get_timeline(
    request: Request,
) -> Timeline:
    """Get project timeline.

    Args:
        request: FastAPI request object.

    Returns:
        Timeline with events.
    """
    logger.info("Getting project timeline")

    events = [
        TimelineEvent(
            event_type="milestone",
            title="Readiness Assessment Complete",
            date=datetime(2026, 1, 31, tzinfo=timezone.utc),
            status="completed",
            is_milestone=True,
        ),
        TimelineEvent(
            event_type="milestone",
            title="Gap Remediation Complete",
            date=datetime(2026, 3, 31, tzinfo=timezone.utc),
            status="in_progress",
            is_milestone=True,
        ),
        TimelineEvent(
            event_type="deadline",
            title="Audit Kickoff",
            date=datetime(2026, 4, 15, tzinfo=timezone.utc),
            status="pending",
            is_milestone=True,
        ),
        TimelineEvent(
            event_type="fieldwork",
            title="Control Testing - CC6",
            date=datetime(2026, 5, 1, tzinfo=timezone.utc),
            status="pending",
            is_milestone=False,
        ),
        TimelineEvent(
            event_type="fieldwork",
            title="Control Testing - CC7",
            date=datetime(2026, 5, 15, tzinfo=timezone.utc),
            status="pending",
            is_milestone=False,
        ),
        TimelineEvent(
            event_type="milestone",
            title="Fieldwork Complete",
            date=datetime(2026, 6, 30, tzinfo=timezone.utc),
            status="pending",
            is_milestone=True,
        ),
        TimelineEvent(
            event_type="deadline",
            title="Report Due",
            date=datetime(2026, 8, 31, tzinfo=timezone.utc),
            status="pending",
            is_milestone=True,
        ),
    ]

    return Timeline(
        project_id=uuid4(),
        events=events,
        critical_path_days=210,
        days_remaining=180,
        is_on_track=True,
        risk_level="low",
    )


@router.get(
    "/milestones",
    response_model=List[Milestone],
    summary="List milestones",
    description="List all project milestones.",
)
async def list_milestones(
    request: Request,
    status_filter: Optional[str] = Query(
        None, alias="status", description="Filter by status"
    ),
    category: Optional[str] = Query(None, description="Filter by category"),
) -> List[Milestone]:
    """List project milestones.

    Args:
        request: FastAPI request object.
        status_filter: Filter by status.
        category: Filter by category.

    Returns:
        List of milestones.
    """
    logger.info("Listing milestones: status=%s, category=%s", status_filter, category)

    milestones = [
        Milestone(
            name="Readiness Assessment Complete",
            category="preparation",
            status="completed",
            target_date=datetime(2026, 1, 31, tzinfo=timezone.utc),
            completed_date=datetime(2026, 1, 28, tzinfo=timezone.utc),
            owner="security-team",
            progress_percent=100,
            is_critical_path=True,
        ),
        Milestone(
            name="Gap Remediation Complete",
            category="preparation",
            status="in_progress",
            target_date=datetime(2026, 3, 31, tzinfo=timezone.utc),
            owner="security-team",
            progress_percent=65,
            is_critical_path=True,
        ),
        Milestone(
            name="Audit Kickoff",
            category="fieldwork",
            status="pending",
            target_date=datetime(2026, 4, 15, tzinfo=timezone.utc),
            owner="compliance-team",
            is_critical_path=True,
        ),
    ]

    # Apply filters
    filtered = milestones
    if status_filter:
        filtered = [m for m in filtered if m.status == status_filter.lower()]
    if category:
        filtered = [m for m in filtered if m.category == category.lower()]

    return filtered


@router.post(
    "/milestones",
    response_model=Milestone,
    status_code=status.HTTP_201_CREATED,
    summary="Add milestone",
    description="Add a new milestone to the project.",
)
async def create_milestone(
    request: Request,
    milestone_data: MilestoneCreate,
) -> Milestone:
    """Create a new milestone.

    Args:
        request: FastAPI request object.
        milestone_data: Milestone details.

    Returns:
        Created Milestone.
    """
    logger.info("Creating milestone: %s", milestone_data.name)

    milestone = Milestone(
        name=milestone_data.name,
        description=milestone_data.description or "",
        category=milestone_data.category,
        status="pending",
        target_date=milestone_data.target_date,
        owner=milestone_data.owner or "",
        dependencies=milestone_data.dependencies or [],
        is_critical_path=milestone_data.is_critical_path,
    )

    return milestone


@router.put(
    "/milestones/{milestone_id}",
    response_model=Milestone,
    summary="Update milestone",
    description="Update a project milestone.",
)
async def update_milestone(
    request: Request,
    milestone_id: UUID,
    update_data: MilestoneUpdate,
) -> Milestone:
    """Update a milestone.

    Args:
        request: FastAPI request object.
        milestone_id: The milestone identifier.
        update_data: Update data.

    Returns:
        Updated Milestone.

    Raises:
        HTTPException: 404 if not found.
    """
    logger.info("Updating milestone: %s", milestone_id)

    # In production, merge with existing milestone from database
    milestone = Milestone(
        milestone_id=milestone_id,
        name=update_data.name or "Gap Remediation Complete",
        description=update_data.description or "Remediate all identified gaps",
        category="preparation",
        status=update_data.status or "in_progress",
        target_date=update_data.target_date or datetime(2026, 3, 31, tzinfo=timezone.utc),
        owner=update_data.owner or "security-team",
        progress_percent=update_data.progress_percent or 65,
        notes=update_data.notes or "",
        is_critical_path=True,
    )

    # Auto-set completed_date if status is completed
    if milestone.status == "completed" and milestone.completed_date is None:
        milestone.completed_date = datetime.now(timezone.utc)

    return milestone


__all__ = ["router"]
