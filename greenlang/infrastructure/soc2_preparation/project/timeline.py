# -*- coding: utf-8 -*-
"""
Audit Timeline Module - SEC-009 Phase 8

Manages audit project timelines, milestones, and critical path analysis for
SOC 2 Type II audit preparation. Supports project creation, milestone tracking,
delay detection, and timeline visualization.

Phases:
    - PLANNING: Initial scoping and audit firm selection
    - KICKOFF: Audit kickoff meetings and scope confirmation
    - FIELDWORK: Evidence collection and control testing
    - COMPLETION: Finding resolution and management responses
    - REPORT: Final report generation and distribution

Classes:
    - AuditPhase: Enumeration of audit project phases
    - AuditProject: Core project data model
    - AuditMilestone: Milestone tracking model
    - AuditTimeline: Timeline orchestration class

Example:
    >>> timeline = AuditTimeline(config)
    >>> project = await timeline.create_timeline(AuditProjectCreate(
    ...     project_name="SOC 2 Type II 2026",
    ...     start_date=date(2026, 1, 1),
    ...     target_end_date=date(2026, 12, 31),
    ... ))
    >>> await timeline.add_milestone(project.project_id, milestone)

Author: GreenLang Security Team
Date: February 2026
PRD: SEC-009 SOC 2 Type II Preparation
"""

from __future__ import annotations

import logging
import uuid
from datetime import date, datetime, timedelta, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------


class AuditPhase(str, Enum):
    """Phases of a SOC 2 Type II audit project.

    The audit progresses through these phases sequentially:
        PLANNING -> KICKOFF -> FIELDWORK -> COMPLETION -> REPORT
    """

    PLANNING = "planning"
    """Initial planning, scoping, and audit firm selection."""

    KICKOFF = "kickoff"
    """Audit kickoff meetings and scope confirmation."""

    FIELDWORK = "fieldwork"
    """Evidence collection, walkthroughs, and control testing."""

    COMPLETION = "completion"
    """Finding resolution and management response preparation."""

    REPORT = "report"
    """Final report generation, review, and distribution."""


class MilestoneStatus(str, Enum):
    """Status of an audit milestone."""

    NOT_STARTED = "not_started"
    """Milestone work has not begun."""

    IN_PROGRESS = "in_progress"
    """Milestone work is underway."""

    COMPLETED = "completed"
    """Milestone has been achieved."""

    DELAYED = "delayed"
    """Milestone is past its target date and not completed."""

    AT_RISK = "at_risk"
    """Milestone is approaching deadline with incomplete work."""

    CANCELLED = "cancelled"
    """Milestone was cancelled."""


# ---------------------------------------------------------------------------
# Phase Configuration
# ---------------------------------------------------------------------------


_PHASE_ORDER = [
    AuditPhase.PLANNING,
    AuditPhase.KICKOFF,
    AuditPhase.FIELDWORK,
    AuditPhase.COMPLETION,
    AuditPhase.REPORT,
]

_DEFAULT_PHASE_DURATION_WEEKS = {
    AuditPhase.PLANNING: 4,
    AuditPhase.KICKOFF: 1,
    AuditPhase.FIELDWORK: 8,
    AuditPhase.COMPLETION: 4,
    AuditPhase.REPORT: 2,
}


# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------


class MilestoneCreate(BaseModel):
    """Input model for creating a new milestone."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    name: str = Field(
        ...,
        min_length=1,
        max_length=256,
        description="Milestone name.",
    )
    description: str = Field(
        default="",
        max_length=2048,
        description="Detailed description of the milestone.",
    )
    phase: AuditPhase = Field(
        ...,
        description="Audit phase this milestone belongs to.",
    )
    target_date: date = Field(
        ...,
        description="Target completion date for the milestone.",
    )
    depends_on: List[str] = Field(
        default_factory=list,
        description="List of milestone IDs this milestone depends on.",
    )
    owner_id: Optional[str] = Field(
        default=None,
        max_length=256,
        description="User ID of the milestone owner.",
    )


class AuditMilestone(BaseModel):
    """Audit milestone tracking model.

    Attributes:
        milestone_id: Unique identifier for the milestone.
        project_id: Parent project ID.
        name: Milestone name.
        description: Detailed description.
        phase: Audit phase this milestone belongs to.
        status: Current milestone status.
        target_date: Target completion date.
        actual_date: Actual completion date (if completed).
        depends_on: List of milestone IDs this depends on.
        owner_id: User ID of the milestone owner.
        created_at: Creation timestamp.
        updated_at: Last modification timestamp.
    """

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    milestone_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique identifier for the milestone.",
    )
    project_id: str = Field(
        ...,
        description="Parent project ID.",
    )
    name: str = Field(
        ...,
        min_length=1,
        max_length=256,
        description="Milestone name.",
    )
    description: str = Field(
        default="",
        max_length=2048,
        description="Detailed description of the milestone.",
    )
    phase: AuditPhase = Field(
        ...,
        description="Audit phase this milestone belongs to.",
    )
    status: MilestoneStatus = Field(
        default=MilestoneStatus.NOT_STARTED,
        description="Current milestone status.",
    )
    target_date: date = Field(
        ...,
        description="Target completion date for the milestone.",
    )
    actual_date: Optional[date] = Field(
        default=None,
        description="Actual completion date (if completed).",
    )
    depends_on: List[str] = Field(
        default_factory=list,
        description="List of milestone IDs this milestone depends on.",
    )
    owner_id: Optional[str] = Field(
        default=None,
        max_length=256,
        description="User ID of the milestone owner.",
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional milestone metadata.",
    )
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Creation timestamp (UTC).",
    )
    updated_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Last modification timestamp (UTC).",
    )

    @property
    def is_overdue(self) -> bool:
        """Check if the milestone is past its target date and not completed."""
        if self.status == MilestoneStatus.COMPLETED:
            return False
        return date.today() > self.target_date

    @property
    def days_until_due(self) -> int:
        """Calculate days until the target date (negative if overdue)."""
        return (self.target_date - date.today()).days


class DelayedMilestone(BaseModel):
    """Information about a delayed milestone."""

    model_config = ConfigDict(extra="forbid")

    milestone: AuditMilestone = Field(
        ...,
        description="The delayed milestone.",
    )
    days_overdue: int = Field(
        ...,
        description="Number of days past the target date.",
    )
    impact_assessment: str = Field(
        default="",
        description="Assessment of the delay's impact on the project.",
    )
    blocking_milestones: List[str] = Field(
        default_factory=list,
        description="Milestone IDs that are blocked by this delay.",
    )


class AuditProjectCreate(BaseModel):
    """Input model for creating a new audit project."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    project_name: str = Field(
        ...,
        min_length=1,
        max_length=256,
        description="Name of the audit project.",
    )
    description: str = Field(
        default="",
        max_length=4096,
        description="Detailed project description.",
    )
    start_date: date = Field(
        ...,
        description="Project start date.",
    )
    target_end_date: date = Field(
        ...,
        description="Target project completion date.",
    )
    audit_period_start: Optional[date] = Field(
        default=None,
        description="Start of the audit observation period.",
    )
    audit_period_end: Optional[date] = Field(
        default=None,
        description="End of the audit observation period.",
    )
    auditor_firm: str = Field(
        default="",
        max_length=256,
        description="Name of the audit firm.",
    )
    project_lead_id: Optional[str] = Field(
        default=None,
        max_length=256,
        description="User ID of the project lead.",
    )
    tsc_categories: List[str] = Field(
        default_factory=lambda: ["security"],
        description="Trust service categories in scope.",
    )


class AuditProject(BaseModel):
    """Core audit project data model.

    Attributes:
        project_id: Unique identifier for the project.
        project_name: Name of the audit project.
        description: Detailed project description.
        current_phase: Current audit phase.
        start_date: Project start date.
        target_end_date: Target completion date.
        actual_end_date: Actual completion date (if completed).
        audit_period_start: Start of the audit observation period.
        audit_period_end: End of the audit observation period.
        auditor_firm: Name of the audit firm.
        project_lead_id: User ID of the project lead.
        tsc_categories: Trust service categories in scope.
        milestones: List of project milestones.
        metadata: Additional project metadata.
        created_at: Creation timestamp.
        updated_at: Last modification timestamp.
    """

    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
        json_schema_extra={
            "examples": [
                {
                    "project_id": "proj-123",
                    "project_name": "SOC 2 Type II 2026",
                    "current_phase": "planning",
                    "start_date": "2026-01-01",
                    "target_end_date": "2026-12-31",
                }
            ]
        },
    )

    project_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique identifier for the project.",
    )
    project_name: str = Field(
        ...,
        min_length=1,
        max_length=256,
        description="Name of the audit project.",
    )
    description: str = Field(
        default="",
        max_length=4096,
        description="Detailed project description.",
    )
    current_phase: AuditPhase = Field(
        default=AuditPhase.PLANNING,
        description="Current audit phase.",
    )
    start_date: date = Field(
        ...,
        description="Project start date.",
    )
    target_end_date: date = Field(
        ...,
        description="Target project completion date.",
    )
    actual_end_date: Optional[date] = Field(
        default=None,
        description="Actual completion date (if completed).",
    )
    audit_period_start: Optional[date] = Field(
        default=None,
        description="Start of the audit observation period.",
    )
    audit_period_end: Optional[date] = Field(
        default=None,
        description="End of the audit observation period.",
    )
    auditor_firm: str = Field(
        default="",
        max_length=256,
        description="Name of the audit firm.",
    )
    project_lead_id: Optional[str] = Field(
        default=None,
        max_length=256,
        description="User ID of the project lead.",
    )
    tsc_categories: List[str] = Field(
        default_factory=lambda: ["security"],
        description="Trust service categories in scope.",
    )
    milestones: List[AuditMilestone] = Field(
        default_factory=list,
        description="List of project milestones.",
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional project metadata.",
    )
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Creation timestamp (UTC).",
    )
    updated_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Last modification timestamp (UTC).",
    )

    @field_validator("tsc_categories")
    @classmethod
    def validate_tsc_categories(cls, v: List[str]) -> List[str]:
        """Validate TSC categories are valid."""
        allowed = {
            "security",
            "availability",
            "confidentiality",
            "processing_integrity",
            "privacy",
        }
        normalized = [c.strip().lower() for c in v]
        for cat in normalized:
            if cat not in allowed:
                raise ValueError(f"Invalid TSC category: {cat}. Allowed: {allowed}")
        return normalized

    @property
    def progress_percentage(self) -> float:
        """Calculate overall project progress based on milestone completion."""
        if not self.milestones:
            return 0.0
        completed = sum(
            1 for m in self.milestones if m.status == MilestoneStatus.COMPLETED
        )
        return (completed / len(self.milestones)) * 100

    @property
    def is_on_track(self) -> bool:
        """Check if the project is on track based on milestone status."""
        overdue = [m for m in self.milestones if m.is_overdue]
        return len(overdue) == 0


class TimelineView(BaseModel):
    """Timeline visualization data for a project."""

    model_config = ConfigDict(extra="forbid")

    project_id: str = Field(..., description="Project ID.")
    project_name: str = Field(..., description="Project name.")
    start_date: date = Field(..., description="Project start date.")
    end_date: date = Field(..., description="Project end date.")
    current_phase: AuditPhase = Field(..., description="Current phase.")
    phases: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Phase information with dates.",
    )
    milestones: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Milestone information for visualization.",
    )
    progress_percentage: float = Field(
        default=0.0,
        description="Overall progress percentage.",
    )
    critical_path: List[str] = Field(
        default_factory=list,
        description="Milestone IDs on the critical path.",
    )


# ---------------------------------------------------------------------------
# Audit Timeline
# ---------------------------------------------------------------------------


class AuditTimeline:
    """Audit timeline and milestone orchestration.

    Manages the creation and tracking of audit project timelines, including
    milestone management, critical path analysis, and delay detection.

    Attributes:
        config: Configuration instance.
        _projects: In-memory project storage.

    Example:
        >>> timeline = AuditTimeline(config)
        >>> project = await timeline.create_timeline(project_create)
        >>> await timeline.add_milestone(project.project_id, milestone_create)
    """

    def __init__(self, config: Any = None) -> None:
        """Initialize AuditTimeline.

        Args:
            config: Optional configuration instance.
        """
        self.config = config
        self._projects: Dict[str, AuditProject] = {}
        logger.info("AuditTimeline initialized")

    async def create_timeline(
        self,
        project: AuditProjectCreate,
    ) -> AuditProject:
        """Create a new audit project with timeline.

        Creates the project and generates default milestones for each phase
        based on the project duration.

        Args:
            project: Project creation parameters.

        Returns:
            The created AuditProject with generated milestones.
        """
        start_time = datetime.now(timezone.utc)

        # Create project
        audit_project = AuditProject(
            project_name=project.project_name,
            description=project.description,
            start_date=project.start_date,
            target_end_date=project.target_end_date,
            audit_period_start=project.audit_period_start,
            audit_period_end=project.audit_period_end,
            auditor_firm=project.auditor_firm,
            project_lead_id=project.project_lead_id,
            tsc_categories=project.tsc_categories,
        )

        # Generate default milestones
        milestones = self._generate_default_milestones(audit_project)
        audit_project.milestones = milestones

        # Store project
        self._projects[audit_project.project_id] = audit_project

        elapsed_ms = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
        logger.info(
            "Audit project created: id=%s, name='%s', "
            "milestones=%d, elapsed=%.2fms",
            audit_project.project_id,
            audit_project.project_name,
            len(milestones),
            elapsed_ms,
        )

        return audit_project

    async def add_milestone(
        self,
        project_id: str,
        milestone: MilestoneCreate,
    ) -> AuditMilestone:
        """Add a milestone to an existing project.

        Args:
            project_id: ID of the project to add the milestone to.
            milestone: Milestone creation parameters.

        Returns:
            The created AuditMilestone.

        Raises:
            ValueError: If project not found.
        """
        project = await self._get_project(project_id)

        audit_milestone = AuditMilestone(
            project_id=project_id,
            name=milestone.name,
            description=milestone.description,
            phase=milestone.phase,
            target_date=milestone.target_date,
            depends_on=milestone.depends_on,
            owner_id=milestone.owner_id,
        )

        project.milestones.append(audit_milestone)
        project.updated_at = datetime.now(timezone.utc)

        logger.info(
            "Milestone added: project=%s, milestone=%s, name='%s', phase=%s",
            project_id,
            audit_milestone.milestone_id,
            audit_milestone.name,
            audit_milestone.phase.value,
        )

        return audit_milestone

    async def update_milestone(
        self,
        milestone_id: str,
        status: Optional[str] = None,
        actual_date: Optional[date] = None,
    ) -> None:
        """Update a milestone's status and/or actual completion date.

        Args:
            milestone_id: ID of the milestone to update.
            status: New status (optional).
            actual_date: Actual completion date (optional).

        Raises:
            ValueError: If milestone not found.
        """
        milestone, project = await self._find_milestone(milestone_id)

        if status is not None:
            milestone.status = MilestoneStatus(status)

        if actual_date is not None:
            milestone.actual_date = actual_date
            if milestone.status != MilestoneStatus.COMPLETED:
                milestone.status = MilestoneStatus.COMPLETED

        milestone.updated_at = datetime.now(timezone.utc)
        project.updated_at = datetime.now(timezone.utc)

        logger.info(
            "Milestone updated: id=%s, status=%s, actual_date=%s",
            milestone_id,
            milestone.status.value,
            actual_date,
        )

    def get_critical_path(self, project_id: str) -> List[AuditMilestone]:
        """Calculate the critical path for a project.

        The critical path is the sequence of dependent milestones that
        determines the minimum project duration.

        Args:
            project_id: ID of the project.

        Returns:
            List of milestones on the critical path, in order.

        Raises:
            ValueError: If project not found.
        """
        project = self._projects.get(project_id)
        if project is None:
            raise ValueError(f"Project '{project_id}' not found.")

        if not project.milestones:
            return []

        # Build dependency graph
        milestone_map = {m.milestone_id: m for m in project.milestones}

        # Find milestones with no dependents (end nodes)
        has_dependents = set()
        for m in project.milestones:
            for dep_id in m.depends_on:
                has_dependents.add(dep_id)

        end_nodes = [
            m for m in project.milestones if m.milestone_id not in has_dependents
        ]

        # Calculate longest path to each end node
        critical_path: List[AuditMilestone] = []
        max_duration = 0

        for end_node in end_nodes:
            path = self._calculate_path_to_milestone(end_node, milestone_map)
            duration = sum(
                (m.target_date - project.start_date).days for m in path
            )
            if duration > max_duration:
                max_duration = duration
                critical_path = path

        return critical_path

    def check_delays(self, project_id: str) -> List[DelayedMilestone]:
        """Check for delayed milestones in a project.

        Args:
            project_id: ID of the project to check.

        Returns:
            List of DelayedMilestone objects for overdue milestones.

        Raises:
            ValueError: If project not found.
        """
        project = self._projects.get(project_id)
        if project is None:
            raise ValueError(f"Project '{project_id}' not found.")

        today = date.today()
        delayed = []

        for milestone in project.milestones:
            if milestone.status == MilestoneStatus.COMPLETED:
                continue
            if milestone.target_date < today:
                days_overdue = (today - milestone.target_date).days

                # Find milestones blocked by this delay
                blocking = [
                    m.milestone_id
                    for m in project.milestones
                    if milestone.milestone_id in m.depends_on
                ]

                delayed.append(
                    DelayedMilestone(
                        milestone=milestone,
                        days_overdue=days_overdue,
                        impact_assessment=self._assess_delay_impact(
                            milestone, days_overdue, project
                        ),
                        blocking_milestones=blocking,
                    )
                )

        # Sort by days overdue (most overdue first)
        delayed.sort(key=lambda d: d.days_overdue, reverse=True)

        logger.debug(
            "Delay check: project=%s, delayed_count=%d",
            project_id,
            len(delayed),
        )

        return delayed

    def get_timeline_view(self, project_id: str) -> TimelineView:
        """Get timeline visualization data for a project.

        Args:
            project_id: ID of the project.

        Returns:
            TimelineView with visualization data.

        Raises:
            ValueError: If project not found.
        """
        project = self._projects.get(project_id)
        if project is None:
            raise ValueError(f"Project '{project_id}' not found.")

        # Calculate phase dates
        phases = []
        current_date = project.start_date
        total_days = (project.target_end_date - project.start_date).days

        for phase in _PHASE_ORDER:
            weeks = _DEFAULT_PHASE_DURATION_WEEKS[phase]
            phase_days = weeks * 7
            phase_end = current_date + timedelta(days=phase_days)
            if phase_end > project.target_end_date:
                phase_end = project.target_end_date

            phases.append(
                {
                    "phase": phase.value,
                    "start_date": current_date.isoformat(),
                    "end_date": phase_end.isoformat(),
                    "duration_days": (phase_end - current_date).days,
                    "is_current": phase == project.current_phase,
                }
            )
            current_date = phase_end

        # Format milestones for visualization
        milestones = []
        for m in project.milestones:
            milestones.append(
                {
                    "milestone_id": m.milestone_id,
                    "name": m.name,
                    "phase": m.phase.value,
                    "status": m.status.value,
                    "target_date": m.target_date.isoformat(),
                    "actual_date": m.actual_date.isoformat() if m.actual_date else None,
                    "is_overdue": m.is_overdue,
                    "days_until_due": m.days_until_due,
                }
            )

        # Get critical path
        critical_path = self.get_critical_path(project_id)
        critical_path_ids = [m.milestone_id for m in critical_path]

        return TimelineView(
            project_id=project.project_id,
            project_name=project.project_name,
            start_date=project.start_date,
            end_date=project.target_end_date,
            current_phase=project.current_phase,
            phases=phases,
            milestones=milestones,
            progress_percentage=project.progress_percentage,
            critical_path=critical_path_ids,
        )

    async def advance_phase(self, project_id: str) -> AuditPhase:
        """Advance the project to the next phase.

        Args:
            project_id: ID of the project.

        Returns:
            The new current phase.

        Raises:
            ValueError: If project not found or already in final phase.
        """
        project = await self._get_project(project_id)

        current_idx = _PHASE_ORDER.index(project.current_phase)
        if current_idx >= len(_PHASE_ORDER) - 1:
            raise ValueError(
                f"Project is already in the final phase: {project.current_phase.value}"
            )

        new_phase = _PHASE_ORDER[current_idx + 1]
        project.current_phase = new_phase
        project.updated_at = datetime.now(timezone.utc)

        logger.info(
            "Project phase advanced: project=%s, new_phase=%s",
            project_id,
            new_phase.value,
        )

        return new_phase

    async def get_project(self, project_id: str) -> Optional[AuditProject]:
        """Get a project by ID.

        Args:
            project_id: ID of the project.

        Returns:
            AuditProject if found, None otherwise.
        """
        return self._projects.get(project_id)

    async def list_projects(self) -> List[AuditProject]:
        """List all audit projects.

        Returns:
            List of all AuditProject objects.
        """
        return list(self._projects.values())

    # -----------------------------------------------------------------------
    # Private Methods
    # -----------------------------------------------------------------------

    async def _get_project(self, project_id: str) -> AuditProject:
        """Get a project or raise ValueError if not found."""
        project = self._projects.get(project_id)
        if project is None:
            raise ValueError(f"Project '{project_id}' not found.")
        return project

    async def _find_milestone(
        self,
        milestone_id: str,
    ) -> tuple[AuditMilestone, AuditProject]:
        """Find a milestone across all projects."""
        for project in self._projects.values():
            for milestone in project.milestones:
                if milestone.milestone_id == milestone_id:
                    return milestone, project
        raise ValueError(f"Milestone '{milestone_id}' not found.")

    def _generate_default_milestones(
        self,
        project: AuditProject,
    ) -> List[AuditMilestone]:
        """Generate default milestones for a project based on phases."""
        milestones = []
        current_date = project.start_date
        total_days = (project.target_end_date - project.start_date).days

        # Default milestone templates for each phase
        templates = {
            AuditPhase.PLANNING: [
                "Audit firm selection",
                "Scope definition",
                "Control matrix review",
                "Risk assessment completion",
            ],
            AuditPhase.KICKOFF: [
                "Kickoff meeting",
                "Document request list received",
            ],
            AuditPhase.FIELDWORK: [
                "Evidence collection start",
                "Control walkthroughs",
                "Sample testing",
                "Evidence collection complete",
            ],
            AuditPhase.COMPLETION: [
                "Draft findings received",
                "Management responses submitted",
                "Finding remediation complete",
            ],
            AuditPhase.REPORT: [
                "Draft report review",
                "Final report issued",
            ],
        }

        for phase in _PHASE_ORDER:
            phase_weeks = _DEFAULT_PHASE_DURATION_WEEKS[phase]
            phase_duration = timedelta(weeks=phase_weeks)
            phase_end = min(current_date + phase_duration, project.target_end_date)
            phase_milestones = templates.get(phase, [])

            # Distribute milestones evenly within the phase
            if phase_milestones:
                days_per_milestone = (phase_end - current_date).days / len(
                    phase_milestones
                )
                for i, name in enumerate(phase_milestones):
                    milestone_date = current_date + timedelta(
                        days=int((i + 1) * days_per_milestone)
                    )
                    if milestone_date > project.target_end_date:
                        milestone_date = project.target_end_date

                    milestones.append(
                        AuditMilestone(
                            project_id=project.project_id,
                            name=name,
                            phase=phase,
                            target_date=milestone_date,
                        )
                    )

            current_date = phase_end

        return milestones

    def _calculate_path_to_milestone(
        self,
        milestone: AuditMilestone,
        milestone_map: Dict[str, AuditMilestone],
    ) -> List[AuditMilestone]:
        """Calculate the path from start to a milestone following dependencies."""
        path = [milestone]
        current = milestone

        while current.depends_on:
            # Get the dependency with the latest target date
            deps = [
                milestone_map[dep_id]
                for dep_id in current.depends_on
                if dep_id in milestone_map
            ]
            if not deps:
                break
            latest_dep = max(deps, key=lambda m: m.target_date)
            path.insert(0, latest_dep)
            current = latest_dep

        return path

    def _assess_delay_impact(
        self,
        milestone: AuditMilestone,
        days_overdue: int,
        project: AuditProject,
    ) -> str:
        """Assess the impact of a milestone delay."""
        if days_overdue <= 3:
            return "Minor delay - manageable with expedited effort"
        elif days_overdue <= 7:
            return "Moderate delay - may impact subsequent milestones"
        elif days_overdue <= 14:
            return "Significant delay - likely to impact project timeline"
        else:
            return "Critical delay - immediate escalation required"


__all__ = [
    "AuditPhase",
    "MilestoneStatus",
    "MilestoneCreate",
    "AuditMilestone",
    "DelayedMilestone",
    "AuditProjectCreate",
    "AuditProject",
    "TimelineView",
    "AuditTimeline",
]
