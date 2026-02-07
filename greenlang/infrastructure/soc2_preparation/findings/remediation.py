# -*- coding: utf-8 -*-
"""
Remediation Workflow - SEC-009 Phase 6

Manage finding remediation lifecycle with state machine, SLA tracking,
progress updates, and evidence collection. Ensures all findings are
properly remediated and documented.

Remediation States:
    identified -> acknowledged -> planned -> in_progress ->
    implemented -> tested -> closed

SLA by Severity:
    - MATERIAL_WEAKNESS: 30 days
    - SIGNIFICANT_DEFICIENCY: 60 days
    - CONTROL_DEFICIENCY: 90 days
    - EXCEPTION: 120 days

Example:
    >>> workflow = RemediationWorkflow()
    >>> plan = await workflow.create_remediation_plan(
    ...     finding_id=finding_uuid,
    ...     plan=RemediationPlan(
    ...         description="Enable MFA for all admin accounts",
    ...         steps=["Identify accounts", "Enable MFA", "Verify enrollment"],
    ...         target_date=datetime.now() + timedelta(days=14),
    ...         owner_id=owner_uuid,
    ...     ),
    ... )
    >>> await workflow.update_progress(finding_uuid, progress=50)
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator

from greenlang.infrastructure.soc2_preparation.findings.tracker import (
    Finding,
    FindingClassification,
    FindingStatus,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# SLA days by severity classification
SLA_BY_SEVERITY = {
    FindingClassification.MATERIAL_WEAKNESS: 30,
    FindingClassification.SIGNIFICANT_DEFICIENCY: 60,
    FindingClassification.CONTROL_DEFICIENCY: 90,
    FindingClassification.EXCEPTION: 120,
}


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------


class RemediationState(str, Enum):
    """States in the remediation workflow."""

    IDENTIFIED = "identified"
    """Finding identified, no plan yet."""

    ACKNOWLEDGED = "acknowledged"
    """Finding acknowledged by responsible party."""

    PLANNED = "planned"
    """Remediation plan created and approved."""

    IN_PROGRESS = "in_progress"
    """Remediation work is actively underway."""

    IMPLEMENTED = "implemented"
    """Remediation implemented, awaiting testing."""

    TESTED = "tested"
    """Remediation tested and verified effective."""

    CLOSED = "closed"
    """Finding closed after successful remediation."""

    BLOCKED = "blocked"
    """Remediation blocked by dependency or issue."""


class PlanStatus(str, Enum):
    """Status of a remediation plan."""

    DRAFT = "draft"
    """Plan is being drafted."""

    PENDING_APPROVAL = "pending_approval"
    """Plan submitted for approval."""

    APPROVED = "approved"
    """Plan has been approved."""

    IN_PROGRESS = "in_progress"
    """Plan execution in progress."""

    COMPLETED = "completed"
    """Plan has been completed."""

    REVISED = "revised"
    """Plan has been revised."""


# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------


class RemediationPlan(BaseModel):
    """Remediation plan for a finding.

    Attributes:
        plan_id: Unique plan identifier.
        finding_id: Associated finding ID.
        description: Overall remediation approach.
        steps: Ordered remediation steps.
        target_date: Target completion date.
        owner_id: Plan owner ID.
        owner_name: Plan owner name.
        approver_id: Plan approver ID.
        approval_date: When plan was approved.
        status: Plan status.
        resources_required: Resources needed.
        estimated_hours: Estimated effort in hours.
        actual_hours: Actual hours spent.
        dependencies: Other finding/plan dependencies.
        risks: Identified implementation risks.
        created_at: Plan creation time.
        updated_at: Last update time.
    """

    model_config = ConfigDict(extra="forbid")

    plan_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique plan identifier.",
    )
    finding_id: str = Field(
        default="",
        description="Associated finding ID.",
    )
    description: str = Field(
        ...,
        min_length=10,
        max_length=4096,
        description="Remediation approach description.",
    )
    steps: List[str] = Field(
        default_factory=list,
        description="Ordered remediation steps.",
    )
    target_date: datetime = Field(
        ...,
        description="Target completion date.",
    )
    owner_id: str = Field(
        default="",
        description="Plan owner ID.",
    )
    owner_name: str = Field(
        default="",
        description="Plan owner name.",
    )
    approver_id: Optional[str] = Field(
        default=None,
        description="Plan approver ID.",
    )
    approval_date: Optional[datetime] = Field(
        default=None,
        description="Approval timestamp.",
    )
    status: PlanStatus = Field(
        default=PlanStatus.DRAFT,
        description="Plan status.",
    )
    resources_required: List[str] = Field(
        default_factory=list,
        description="Required resources.",
    )
    estimated_hours: float = Field(
        default=0.0,
        ge=0,
        description="Estimated effort in hours.",
    )
    actual_hours: float = Field(
        default=0.0,
        ge=0,
        description="Actual hours spent.",
    )
    dependencies: List[str] = Field(
        default_factory=list,
        description="Dependency finding/plan IDs.",
    )
    risks: List[str] = Field(
        default_factory=list,
        description="Implementation risks.",
    )
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Creation timestamp.",
    )
    updated_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Last update timestamp.",
    )


class RemediationProgress(BaseModel):
    """Progress update for a remediation.

    Attributes:
        progress_id: Unique progress ID.
        finding_id: Associated finding ID.
        progress_percentage: Completion percentage (0-100).
        steps_completed: Number of steps completed.
        total_steps: Total number of steps.
        status_update: Status description.
        blockers: Current blockers if any.
        next_actions: Planned next actions.
        updated_by: Who made the update.
        updated_at: Update timestamp.
    """

    model_config = ConfigDict(extra="forbid")

    progress_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique progress ID.",
    )
    finding_id: str = Field(
        ...,
        description="Associated finding ID.",
    )
    progress_percentage: int = Field(
        default=0,
        ge=0,
        le=100,
        description="Completion percentage.",
    )
    steps_completed: int = Field(
        default=0,
        ge=0,
        description="Steps completed.",
    )
    total_steps: int = Field(
        default=0,
        ge=0,
        description="Total steps.",
    )
    status_update: str = Field(
        default="",
        max_length=2048,
        description="Status description.",
    )
    blockers: List[str] = Field(
        default_factory=list,
        description="Current blockers.",
    )
    next_actions: List[str] = Field(
        default_factory=list,
        description="Planned next actions.",
    )
    updated_by: str = Field(
        default="",
        description="Who made the update.",
    )
    updated_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Update timestamp.",
    )


class RemediationEvidence(BaseModel):
    """Evidence supporting remediation completion.

    Attributes:
        evidence_id: Evidence item ID.
        finding_id: Associated finding ID.
        evidence_type: Type of evidence.
        description: Evidence description.
        file_id: Uploaded file ID if applicable.
        url: External URL if applicable.
        uploaded_by: Who uploaded the evidence.
        uploaded_at: Upload timestamp.
    """

    model_config = ConfigDict(extra="forbid")

    evidence_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Evidence item ID.",
    )
    finding_id: str = Field(
        ...,
        description="Associated finding ID.",
    )
    evidence_type: str = Field(
        default="document",
        description="Evidence type: screenshot, document, log, config, etc.",
    )
    description: str = Field(
        default="",
        max_length=1024,
        description="Evidence description.",
    )
    file_id: Optional[str] = Field(
        default=None,
        description="Uploaded file ID.",
    )
    url: Optional[str] = Field(
        default=None,
        description="External URL.",
    )
    uploaded_by: str = Field(
        default="",
        description="Who uploaded.",
    )
    uploaded_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Upload timestamp.",
    )


# ---------------------------------------------------------------------------
# Remediation Workflow
# ---------------------------------------------------------------------------


class RemediationWorkflow:
    """Manage finding remediation lifecycle.

    Provides state machine for remediation progress, plan management,
    progress tracking, evidence collection, and SLA monitoring.

    Attributes:
        _plans: Remediation plans by plan_id.
        _progress: Progress updates by finding_id.
        _evidence: Evidence items by finding_id.
        _findings: Reference to findings (injected).
        _state_transitions: Valid state transitions.
    """

    # Valid state transitions
    STATE_TRANSITIONS = {
        RemediationState.IDENTIFIED: [RemediationState.ACKNOWLEDGED],
        RemediationState.ACKNOWLEDGED: [RemediationState.PLANNED],
        RemediationState.PLANNED: [RemediationState.IN_PROGRESS],
        RemediationState.IN_PROGRESS: [
            RemediationState.IMPLEMENTED,
            RemediationState.BLOCKED,
        ],
        RemediationState.BLOCKED: [RemediationState.IN_PROGRESS],
        RemediationState.IMPLEMENTED: [RemediationState.TESTED, RemediationState.IN_PROGRESS],
        RemediationState.TESTED: [RemediationState.CLOSED, RemediationState.IN_PROGRESS],
        RemediationState.CLOSED: [],
    }

    def __init__(self) -> None:
        """Initialize the remediation workflow."""
        self._plans: Dict[str, RemediationPlan] = {}
        self._plans_by_finding: Dict[str, str] = {}  # finding_id -> plan_id
        self._progress: Dict[str, List[RemediationProgress]] = {}
        self._evidence: Dict[str, List[RemediationEvidence]] = {}
        self._finding_states: Dict[str, RemediationState] = {}
        logger.info("RemediationWorkflow initialized")

    # ------------------------------------------------------------------
    # Plan Management
    # ------------------------------------------------------------------

    async def create_remediation_plan(
        self,
        finding_id: uuid.UUID,
        plan: RemediationPlan,
    ) -> RemediationPlan:
        """Create a remediation plan for a finding.

        Args:
            finding_id: Finding identifier.
            plan: Remediation plan details.

        Returns:
            Created RemediationPlan.
        """
        finding_id_str = str(finding_id)

        # Set finding_id on plan
        plan.finding_id = finding_id_str

        # Store plan
        self._plans[plan.plan_id] = plan
        self._plans_by_finding[finding_id_str] = plan.plan_id

        # Update state
        self._finding_states[finding_id_str] = RemediationState.PLANNED

        # Initialize progress tracking
        self._progress[finding_id_str] = []
        self._evidence[finding_id_str] = []

        logger.info(
            "Created remediation plan %s for finding %s (target=%s)",
            plan.plan_id[:8],
            finding_id_str[:8],
            plan.target_date.isoformat(),
        )

        return plan

    async def approve_plan(
        self,
        plan_id: str,
        approver_id: uuid.UUID,
    ) -> RemediationPlan:
        """Approve a remediation plan.

        Args:
            plan_id: Plan identifier.
            approver_id: Approver user ID.

        Returns:
            Updated plan.

        Raises:
            ValueError: If plan not found or not pending approval.
        """
        plan = self._plans.get(plan_id)
        if plan is None:
            raise ValueError(f"Plan {plan_id} not found")

        plan.approver_id = str(approver_id)
        plan.approval_date = datetime.now(timezone.utc)
        plan.status = PlanStatus.APPROVED
        plan.updated_at = datetime.now(timezone.utc)

        logger.info("Approved plan %s", plan_id[:8])
        return plan

    async def get_plan(self, finding_id: uuid.UUID) -> Optional[RemediationPlan]:
        """Get remediation plan for a finding.

        Args:
            finding_id: Finding identifier.

        Returns:
            RemediationPlan if exists.
        """
        finding_id_str = str(finding_id)
        plan_id = self._plans_by_finding.get(finding_id_str)
        if plan_id:
            return self._plans.get(plan_id)
        return None

    # ------------------------------------------------------------------
    # Progress Tracking
    # ------------------------------------------------------------------

    async def update_progress(
        self,
        finding_id: uuid.UUID,
        progress: int,
        status_update: str = "",
        updated_by: str = "",
        blockers: Optional[List[str]] = None,
        next_actions: Optional[List[str]] = None,
    ) -> RemediationProgress:
        """Update remediation progress.

        Args:
            finding_id: Finding identifier.
            progress: Completion percentage (0-100).
            status_update: Status description.
            updated_by: Who made the update.
            blockers: Current blockers.
            next_actions: Planned next actions.

        Returns:
            Created RemediationProgress entry.

        Raises:
            ValueError: If progress is out of range.
        """
        if not 0 <= progress <= 100:
            raise ValueError("Progress must be between 0 and 100")

        finding_id_str = str(finding_id)

        # Get plan to calculate steps
        plan = await self.get_plan(finding_id)
        steps_completed = 0
        total_steps = 0
        if plan:
            total_steps = len(plan.steps)
            steps_completed = int(total_steps * progress / 100)

        progress_entry = RemediationProgress(
            finding_id=finding_id_str,
            progress_percentage=progress,
            steps_completed=steps_completed,
            total_steps=total_steps,
            status_update=status_update,
            blockers=blockers or [],
            next_actions=next_actions or [],
            updated_by=updated_by,
        )

        if finding_id_str not in self._progress:
            self._progress[finding_id_str] = []
        self._progress[finding_id_str].append(progress_entry)

        # Update state based on progress
        if progress == 100:
            self._finding_states[finding_id_str] = RemediationState.IMPLEMENTED
        elif progress > 0:
            self._finding_states[finding_id_str] = RemediationState.IN_PROGRESS

        # Check for blockers
        if blockers:
            self._finding_states[finding_id_str] = RemediationState.BLOCKED

        logger.info(
            "Updated progress for finding %s: %d%%",
            finding_id_str[:8],
            progress,
        )

        return progress_entry

    async def get_progress_history(
        self,
        finding_id: uuid.UUID,
    ) -> List[RemediationProgress]:
        """Get progress history for a finding.

        Args:
            finding_id: Finding identifier.

        Returns:
            List of progress updates, oldest first.
        """
        return self._progress.get(str(finding_id), [])

    async def get_current_progress(
        self,
        finding_id: uuid.UUID,
    ) -> Optional[RemediationProgress]:
        """Get most recent progress update.

        Args:
            finding_id: Finding identifier.

        Returns:
            Most recent RemediationProgress.
        """
        history = self._progress.get(str(finding_id), [])
        return history[-1] if history else None

    # ------------------------------------------------------------------
    # Evidence Management
    # ------------------------------------------------------------------

    async def upload_evidence(
        self,
        finding_id: uuid.UUID,
        evidence: List[RemediationEvidence],
    ) -> List[RemediationEvidence]:
        """Upload evidence supporting remediation.

        Args:
            finding_id: Finding identifier.
            evidence: List of evidence items.

        Returns:
            Stored evidence items.
        """
        finding_id_str = str(finding_id)

        if finding_id_str not in self._evidence:
            self._evidence[finding_id_str] = []

        for item in evidence:
            item.finding_id = finding_id_str
            self._evidence[finding_id_str].append(item)

        logger.info(
            "Uploaded %d evidence items for finding %s",
            len(evidence),
            finding_id_str[:8],
        )

        return evidence

    async def get_evidence(
        self,
        finding_id: uuid.UUID,
    ) -> List[RemediationEvidence]:
        """Get all evidence for a finding.

        Args:
            finding_id: Finding identifier.

        Returns:
            List of evidence items.
        """
        return self._evidence.get(str(finding_id), [])

    # ------------------------------------------------------------------
    # SLA Management
    # ------------------------------------------------------------------

    def _calculate_sla(self, severity: FindingClassification) -> int:
        """Calculate SLA days based on severity.

        Args:
            severity: Finding classification.

        Returns:
            SLA duration in days.
        """
        return SLA_BY_SEVERITY.get(severity, 90)

    async def check_overdue(self) -> List[Dict[str, Any]]:
        """Check for findings past their SLA deadline.

        Returns:
            List of overdue finding info.
        """
        now = datetime.now(timezone.utc)
        overdue: List[Dict[str, Any]] = []

        for finding_id, plan in self._plans_by_finding.items():
            plan_obj = self._plans.get(plan)
            state = self._finding_states.get(finding_id, RemediationState.IDENTIFIED)

            if state == RemediationState.CLOSED:
                continue

            if plan_obj and plan_obj.target_date < now:
                days_overdue = (now - plan_obj.target_date).days
                overdue.append({
                    "finding_id": finding_id,
                    "plan_id": plan,
                    "target_date": plan_obj.target_date.isoformat(),
                    "days_overdue": days_overdue,
                    "state": state.value,
                    "owner_name": plan_obj.owner_name,
                })

        return sorted(overdue, key=lambda x: x["days_overdue"], reverse=True)

    async def get_at_risk(self, days: int = 7) -> List[Dict[str, Any]]:
        """Get findings at risk of missing SLA.

        Args:
            days: Days until SLA to consider at-risk.

        Returns:
            List of at-risk finding info.
        """
        now = datetime.now(timezone.utc)
        threshold = now + timedelta(days=days)
        at_risk: List[Dict[str, Any]] = []

        for finding_id, plan in self._plans_by_finding.items():
            plan_obj = self._plans.get(plan)
            state = self._finding_states.get(finding_id, RemediationState.IDENTIFIED)

            if state == RemediationState.CLOSED:
                continue

            if plan_obj and now < plan_obj.target_date <= threshold:
                days_remaining = (plan_obj.target_date - now).days
                at_risk.append({
                    "finding_id": finding_id,
                    "plan_id": plan,
                    "target_date": plan_obj.target_date.isoformat(),
                    "days_remaining": days_remaining,
                    "state": state.value,
                    "owner_name": plan_obj.owner_name,
                })

        return sorted(at_risk, key=lambda x: x["days_remaining"])

    # ------------------------------------------------------------------
    # State Management
    # ------------------------------------------------------------------

    async def transition_state(
        self,
        finding_id: uuid.UUID,
        new_state: RemediationState,
        notes: str = "",
    ) -> RemediationState:
        """Transition finding to a new remediation state.

        Args:
            finding_id: Finding identifier.
            new_state: Target state.
            notes: Transition notes.

        Returns:
            New state.

        Raises:
            ValueError: If transition is not valid.
        """
        finding_id_str = str(finding_id)
        current_state = self._finding_states.get(finding_id_str, RemediationState.IDENTIFIED)

        valid_transitions = self.STATE_TRANSITIONS.get(current_state, [])
        if new_state not in valid_transitions:
            raise ValueError(
                f"Cannot transition from {current_state.value} to {new_state.value}. "
                f"Valid transitions: {[s.value for s in valid_transitions]}"
            )

        self._finding_states[finding_id_str] = new_state

        logger.info(
            "Transitioned finding %s: %s -> %s",
            finding_id_str[:8],
            current_state.value,
            new_state.value,
        )

        return new_state

    async def get_state(self, finding_id: uuid.UUID) -> RemediationState:
        """Get current remediation state for a finding.

        Args:
            finding_id: Finding identifier.

        Returns:
            Current RemediationState.
        """
        return self._finding_states.get(str(finding_id), RemediationState.IDENTIFIED)

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------

    async def get_statistics(self) -> Dict[str, Any]:
        """Get remediation workflow statistics.

        Returns:
            Dictionary with remediation metrics.
        """
        by_state: Dict[str, int] = {}
        total_plans = len(self._plans)
        completed_plans = 0
        overdue_count = 0

        for finding_id, state in self._finding_states.items():
            by_state[state.value] = by_state.get(state.value, 0) + 1
            if state == RemediationState.CLOSED:
                completed_plans += 1

        overdue = await self.check_overdue()
        overdue_count = len(overdue)

        return {
            "total_plans": total_plans,
            "by_state": by_state,
            "completed": completed_plans,
            "in_progress": by_state.get(RemediationState.IN_PROGRESS.value, 0),
            "blocked": by_state.get(RemediationState.BLOCKED.value, 0),
            "overdue_count": overdue_count,
        }


__all__ = [
    "RemediationWorkflow",
    "RemediationPlan",
    "RemediationProgress",
    "RemediationEvidence",
    "RemediationState",
    "PlanStatus",
]
