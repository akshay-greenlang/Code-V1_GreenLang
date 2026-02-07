# -*- coding: utf-8 -*-
"""
Finding Tracker - SEC-009 Phase 6

Track and classify audit findings with severity assessment. Provides
finding creation, classification, owner assignment, and status tracking
with comprehensive reporting capabilities.

Classification Criteria:
    - EXCEPTION: Isolated deviation, low impact, easily remediated
    - CONTROL_DEFICIENCY: Control gap affecting 10-25% of transactions
    - SIGNIFICANT_DEFICIENCY: Control gap affecting 25-50% of transactions
    - MATERIAL_WEAKNESS: Systemic control failure affecting >50%

Example:
    >>> tracker = FindingTracker()
    >>> finding = await tracker.create_finding(
    ...     FindingCreate(
    ...         title="Access reviews not completed",
    ...         criterion_id="CC6.3",
    ...         description="Q3 access reviews for 3 systems not completed",
    ...     )
    ... )
    >>> classification = tracker.classify_finding(finding)
    >>> print(classification)  # FindingClassification.CONTROL_DEFICIENCY
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
# Enumerations
# ---------------------------------------------------------------------------


class FindingClassification(str, Enum):
    """Classification levels for audit findings."""

    EXCEPTION = "exception"
    """Minor deviation from control procedure. Low impact, easily remediated.
    SLA: 120 days."""

    CONTROL_DEFICIENCY = "control_deficiency"
    """Control not operating effectively for portion of population.
    SLA: 90 days."""

    SIGNIFICANT_DEFICIENCY = "significant_deficiency"
    """Material weakness requiring management attention.
    SLA: 60 days."""

    MATERIAL_WEAKNESS = "material_weakness"
    """Critical control failure with systemic impact.
    SLA: 30 days."""


class FindingStatus(str, Enum):
    """Status of an audit finding."""

    IDENTIFIED = "identified"
    """Finding has been identified and documented."""

    ACKNOWLEDGED = "acknowledged"
    """Finding has been acknowledged by responsible party."""

    PLANNED = "planned"
    """Remediation plan has been created."""

    IN_PROGRESS = "in_progress"
    """Remediation is actively being worked on."""

    IMPLEMENTED = "implemented"
    """Remediation has been implemented, awaiting testing."""

    TESTING = "testing"
    """Remediation is being tested for effectiveness."""

    PENDING_CLOSURE = "pending_closure"
    """Closure request submitted, awaiting verification."""

    CLOSED = "closed"
    """Finding has been remediated and verified."""

    REOPENED = "reopened"
    """Previously closed finding has been reopened."""

    ACCEPTED = "accepted"
    """Risk has been formally accepted (for low-impact findings)."""


class FindingSource(str, Enum):
    """Source of the audit finding."""

    CONTROL_TESTING = "control_testing"
    """From automated or manual control testing."""

    AUDITOR_INQUIRY = "auditor_inquiry"
    """From auditor information request."""

    SELF_ASSESSMENT = "self_assessment"
    """From internal self-assessment."""

    INCIDENT = "incident"
    """From security or operational incident."""

    CONTINUOUS_MONITORING = "continuous_monitoring"
    """From continuous monitoring systems."""


# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------


class FindingCreate(BaseModel):
    """Input for creating a new finding.

    Attributes:
        title: Short title for the finding.
        criterion_id: Related SOC 2 criterion.
        description: Detailed description of the finding.
        source: Source of the finding.
        test_id: Related control test ID if applicable.
        evidence_ids: Evidence supporting the finding.
        impact_description: Description of business impact.
        affected_population: Number or percentage of affected items.
        total_population: Total population size for context.
    """

    model_config = ConfigDict(extra="forbid")

    title: str = Field(
        ...,
        min_length=5,
        max_length=256,
        description="Short finding title.",
    )
    criterion_id: str = Field(
        ...,
        min_length=2,
        max_length=32,
        description="Related SOC 2 criterion (e.g., CC6.1).",
    )
    description: str = Field(
        ...,
        min_length=10,
        max_length=4096,
        description="Detailed description.",
    )
    source: FindingSource = Field(
        default=FindingSource.CONTROL_TESTING,
        description="Source of the finding.",
    )
    test_id: Optional[str] = Field(
        default=None,
        description="Related control test ID.",
    )
    evidence_ids: List[str] = Field(
        default_factory=list,
        description="Supporting evidence IDs.",
    )
    impact_description: str = Field(
        default="",
        max_length=2048,
        description="Business impact description.",
    )
    affected_population: int = Field(
        default=0,
        ge=0,
        description="Number of affected items.",
    )
    total_population: int = Field(
        default=0,
        ge=0,
        description="Total population size.",
    )


class Finding(BaseModel):
    """Full audit finding with tracking information.

    Attributes:
        finding_id: Unique finding identifier.
        title: Short finding title.
        criterion_id: Related SOC 2 criterion.
        description: Detailed description.
        source: Finding source.
        classification: Finding severity classification.
        status: Current status.
        owner_id: Assigned owner ID.
        owner_name: Assigned owner name.
        test_id: Related control test ID.
        evidence_ids: Supporting evidence IDs.
        impact_description: Business impact description.
        affected_population: Number of affected items.
        total_population: Total population size.
        affected_percentage: Calculated percentage affected.
        root_cause: Root cause analysis.
        remediation_plan_id: Associated remediation plan.
        sla_deadline: Remediation SLA deadline.
        created_at: Finding creation time.
        updated_at: Last update time.
        closed_at: Closure time.
        notes: Internal notes.
    """

    model_config = ConfigDict(extra="forbid")

    finding_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique finding identifier.",
    )
    title: str = Field(
        ...,
        description="Short finding title.",
    )
    criterion_id: str = Field(
        ...,
        description="Related SOC 2 criterion.",
    )
    description: str = Field(
        ...,
        description="Detailed description.",
    )
    source: FindingSource = Field(
        default=FindingSource.CONTROL_TESTING,
        description="Finding source.",
    )
    classification: Optional[FindingClassification] = Field(
        default=None,
        description="Severity classification.",
    )
    status: FindingStatus = Field(
        default=FindingStatus.IDENTIFIED,
        description="Current status.",
    )
    owner_id: Optional[str] = Field(
        default=None,
        description="Assigned owner ID.",
    )
    owner_name: str = Field(
        default="",
        description="Assigned owner name.",
    )
    test_id: Optional[str] = Field(
        default=None,
        description="Related control test ID.",
    )
    evidence_ids: List[str] = Field(
        default_factory=list,
        description="Supporting evidence IDs.",
    )
    impact_description: str = Field(
        default="",
        description="Business impact description.",
    )
    affected_population: int = Field(
        default=0,
        description="Number of affected items.",
    )
    total_population: int = Field(
        default=0,
        description="Total population size.",
    )
    root_cause: str = Field(
        default="",
        max_length=4096,
        description="Root cause analysis.",
    )
    remediation_plan_id: Optional[str] = Field(
        default=None,
        description="Associated remediation plan ID.",
    )
    sla_deadline: Optional[datetime] = Field(
        default=None,
        description="Remediation SLA deadline.",
    )
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Creation timestamp.",
    )
    updated_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Last update timestamp.",
    )
    closed_at: Optional[datetime] = Field(
        default=None,
        description="Closure timestamp.",
    )
    notes: str = Field(
        default="",
        max_length=4096,
        description="Internal notes.",
    )

    @property
    def affected_percentage(self) -> float:
        """Calculate percentage of population affected."""
        if self.total_population == 0:
            return 0.0
        return (self.affected_population / self.total_population) * 100

    @property
    def is_overdue(self) -> bool:
        """Check if finding is past SLA deadline."""
        if self.sla_deadline is None:
            return False
        if self.status == FindingStatus.CLOSED:
            return False
        return datetime.now(timezone.utc) > self.sla_deadline

    @property
    def days_open(self) -> int:
        """Number of days the finding has been open."""
        end_time = self.closed_at or datetime.now(timezone.utc)
        return (end_time - self.created_at).days


class FindingSummary(BaseModel):
    """Summary statistics for findings.

    Attributes:
        total: Total number of findings.
        by_classification: Count by classification level.
        by_status: Count by status.
        by_criterion: Count by SOC 2 criterion.
        overdue_count: Number of overdue findings.
        avg_days_to_close: Average days to close findings.
    """

    total: int = Field(default=0)
    by_classification: Dict[str, int] = Field(default_factory=dict)
    by_status: Dict[str, int] = Field(default_factory=dict)
    by_criterion: Dict[str, int] = Field(default_factory=dict)
    overdue_count: int = Field(default=0)
    avg_days_to_close: float = Field(default=0.0)


class FindingAge(BaseModel):
    """Finding age information for aging report.

    Attributes:
        finding_id: Finding identifier.
        title: Finding title.
        classification: Finding classification.
        days_open: Days the finding has been open.
        sla_deadline: SLA deadline.
        days_until_sla: Days until SLA (negative if overdue).
        owner_name: Assigned owner.
    """

    finding_id: str
    title: str
    classification: Optional[FindingClassification]
    days_open: int
    sla_deadline: Optional[datetime]
    days_until_sla: Optional[int]
    owner_name: str


# ---------------------------------------------------------------------------
# Finding Tracker
# ---------------------------------------------------------------------------


class FindingTracker:
    """Track and classify audit findings.

    Provides finding lifecycle management including creation, classification,
    owner assignment, and status tracking.

    Attributes:
        _findings: Stored findings by finding_id.
        _sla_days: SLA days by classification.
    """

    # SLA days by classification
    SLA_BY_CLASSIFICATION = {
        FindingClassification.MATERIAL_WEAKNESS: 30,
        FindingClassification.SIGNIFICANT_DEFICIENCY: 60,
        FindingClassification.CONTROL_DEFICIENCY: 90,
        FindingClassification.EXCEPTION: 120,
    }

    def __init__(self) -> None:
        """Initialize the finding tracker."""
        self._findings: Dict[str, Finding] = {}
        logger.info("FindingTracker initialized")

    # ------------------------------------------------------------------
    # Finding Creation
    # ------------------------------------------------------------------

    async def create_finding(self, finding: FindingCreate) -> Finding:
        """Create a new audit finding.

        Args:
            finding: Finding details.

        Returns:
            Created Finding with generated ID.
        """
        new_finding = Finding(
            title=finding.title,
            criterion_id=finding.criterion_id,
            description=finding.description,
            source=finding.source,
            test_id=finding.test_id,
            evidence_ids=finding.evidence_ids,
            impact_description=finding.impact_description,
            affected_population=finding.affected_population,
            total_population=finding.total_population,
        )

        # Auto-classify based on affected percentage
        new_finding.classification = self.classify_finding(new_finding)

        # Set SLA deadline
        if new_finding.classification:
            sla_days = self.SLA_BY_CLASSIFICATION[new_finding.classification]
            new_finding.sla_deadline = datetime.now(timezone.utc) + timedelta(days=sla_days)

        self._findings[new_finding.finding_id] = new_finding

        logger.info(
            "Created finding %s: %s (classification=%s)",
            new_finding.finding_id[:8],
            new_finding.title,
            new_finding.classification.value if new_finding.classification else "unclassified",
        )

        return new_finding

    # ------------------------------------------------------------------
    # Classification
    # ------------------------------------------------------------------

    def classify_finding(self, finding: Finding) -> FindingClassification:
        """Classify a finding based on its characteristics.

        Classification is based on:
        - Affected percentage of population
        - Presence of compensating controls
        - Impact severity

        Args:
            finding: Finding to classify.

        Returns:
            Appropriate FindingClassification.
        """
        affected_pct = finding.affected_percentage

        # Classification based on affected percentage
        if affected_pct >= 50:
            return FindingClassification.MATERIAL_WEAKNESS
        elif affected_pct >= 25:
            return FindingClassification.SIGNIFICANT_DEFICIENCY
        elif affected_pct >= 10:
            return FindingClassification.CONTROL_DEFICIENCY
        else:
            return FindingClassification.EXCEPTION

    def reclassify_finding(
        self,
        finding_id: str,
        classification: FindingClassification,
        reason: str,
    ) -> Finding:
        """Manually reclassify a finding.

        Args:
            finding_id: Finding identifier.
            classification: New classification.
            reason: Reason for reclassification.

        Returns:
            Updated Finding.

        Raises:
            ValueError: If finding not found.
        """
        finding = self._findings.get(finding_id)
        if finding is None:
            raise ValueError(f"Finding {finding_id} not found")

        old_classification = finding.classification
        finding.classification = classification
        finding.updated_at = datetime.now(timezone.utc)
        finding.notes += f"\nReclassified from {old_classification.value if old_classification else 'none'} to {classification.value}: {reason}"

        # Update SLA based on new classification
        sla_days = self.SLA_BY_CLASSIFICATION[classification]
        finding.sla_deadline = finding.created_at + timedelta(days=sla_days)

        logger.info(
            "Reclassified finding %s: %s -> %s",
            finding_id[:8],
            old_classification.value if old_classification else "none",
            classification.value,
        )

        return finding

    # ------------------------------------------------------------------
    # Owner Assignment
    # ------------------------------------------------------------------

    async def assign_owner(
        self,
        finding_id: uuid.UUID,
        owner_id: uuid.UUID,
        owner_name: str = "",
    ) -> Finding:
        """Assign an owner to a finding.

        Args:
            finding_id: Finding identifier.
            owner_id: Owner user ID.
            owner_name: Owner display name.

        Returns:
            Updated Finding.

        Raises:
            ValueError: If finding not found.
        """
        finding_id_str = str(finding_id)
        finding = self._findings.get(finding_id_str)

        if finding is None:
            raise ValueError(f"Finding {finding_id_str} not found")

        finding.owner_id = str(owner_id)
        finding.owner_name = owner_name
        finding.updated_at = datetime.now(timezone.utc)

        # Update status if still identified
        if finding.status == FindingStatus.IDENTIFIED:
            finding.status = FindingStatus.ACKNOWLEDGED

        logger.info(
            "Assigned finding %s to %s",
            finding_id_str[:8],
            owner_name or str(owner_id)[:8],
        )

        return finding

    # ------------------------------------------------------------------
    # Status Management
    # ------------------------------------------------------------------

    async def update_status(
        self,
        finding_id: uuid.UUID,
        status: str,
        notes: str = "",
    ) -> Finding:
        """Update finding status.

        Args:
            finding_id: Finding identifier.
            status: New status value.
            notes: Optional notes.

        Returns:
            Updated Finding.

        Raises:
            ValueError: If finding not found or invalid status.
        """
        finding_id_str = str(finding_id)
        finding = self._findings.get(finding_id_str)

        if finding is None:
            raise ValueError(f"Finding {finding_id_str} not found")

        try:
            new_status = FindingStatus(status)
        except ValueError:
            raise ValueError(f"Invalid status: {status}")

        old_status = finding.status
        finding.status = new_status
        finding.updated_at = datetime.now(timezone.utc)

        if notes:
            finding.notes += f"\n[{old_status.value} -> {new_status.value}] {notes}"

        if new_status == FindingStatus.CLOSED:
            finding.closed_at = datetime.now(timezone.utc)

        logger.info(
            "Updated finding %s status: %s -> %s",
            finding_id_str[:8],
            old_status.value,
            new_status.value,
        )

        return finding

    async def add_root_cause(
        self,
        finding_id: uuid.UUID,
        root_cause: str,
    ) -> Finding:
        """Add or update root cause analysis.

        Args:
            finding_id: Finding identifier.
            root_cause: Root cause description.

        Returns:
            Updated Finding.
        """
        finding_id_str = str(finding_id)
        finding = self._findings.get(finding_id_str)

        if finding is None:
            raise ValueError(f"Finding {finding_id_str} not found")

        finding.root_cause = root_cause
        finding.updated_at = datetime.now(timezone.utc)

        logger.debug("Added root cause to finding %s", finding_id_str[:8])
        return finding

    # ------------------------------------------------------------------
    # Query Methods
    # ------------------------------------------------------------------

    async def get_finding(self, finding_id: uuid.UUID) -> Optional[Finding]:
        """Get a finding by ID.

        Args:
            finding_id: Finding identifier.

        Returns:
            Finding if found.
        """
        return self._findings.get(str(finding_id))

    async def list_findings(
        self,
        status: Optional[FindingStatus] = None,
        classification: Optional[FindingClassification] = None,
        owner_id: Optional[uuid.UUID] = None,
        criterion_id: Optional[str] = None,
        overdue_only: bool = False,
        limit: int = 50,
        offset: int = 0,
    ) -> List[Finding]:
        """List findings with optional filtering.

        Args:
            status: Filter by status.
            classification: Filter by classification.
            owner_id: Filter by owner.
            criterion_id: Filter by criterion.
            overdue_only: Only return overdue findings.
            limit: Maximum results.
            offset: Results offset.

        Returns:
            List of matching findings.
        """
        findings = list(self._findings.values())

        if status:
            findings = [f for f in findings if f.status == status]

        if classification:
            findings = [f for f in findings if f.classification == classification]

        if owner_id:
            findings = [f for f in findings if f.owner_id == str(owner_id)]

        if criterion_id:
            findings = [f for f in findings if f.criterion_id == criterion_id]

        if overdue_only:
            findings = [f for f in findings if f.is_overdue]

        # Sort by classification severity, then by SLA deadline
        classification_order = {
            FindingClassification.MATERIAL_WEAKNESS: 0,
            FindingClassification.SIGNIFICANT_DEFICIENCY: 1,
            FindingClassification.CONTROL_DEFICIENCY: 2,
            FindingClassification.EXCEPTION: 3,
            None: 4,
        }
        findings.sort(
            key=lambda f: (
                classification_order.get(f.classification, 4),
                f.sla_deadline or datetime.max.replace(tzinfo=timezone.utc),
            )
        )

        return findings[offset : offset + limit]

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    async def get_finding_summary(self) -> FindingSummary:
        """Get summary statistics for all findings.

        Returns:
            FindingSummary with counts and metrics.
        """
        findings = list(self._findings.values())

        summary = FindingSummary(total=len(findings))

        closed_days: List[int] = []

        for finding in findings:
            # Count by classification
            if finding.classification:
                key = finding.classification.value
                summary.by_classification[key] = summary.by_classification.get(key, 0) + 1

            # Count by status
            key = finding.status.value
            summary.by_status[key] = summary.by_status.get(key, 0) + 1

            # Count by criterion
            criterion = finding.criterion_id.split(".")[0]
            summary.by_criterion[criterion] = summary.by_criterion.get(criterion, 0) + 1

            # Check overdue
            if finding.is_overdue:
                summary.overdue_count += 1

            # Track closure time
            if finding.closed_at:
                days = (finding.closed_at - finding.created_at).days
                closed_days.append(days)

        # Calculate average days to close
        if closed_days:
            summary.avg_days_to_close = sum(closed_days) / len(closed_days)

        return summary

    async def get_aging_report(self) -> List[FindingAge]:
        """Get aging report for all open findings.

        Returns:
            List of FindingAge entries sorted by days open.
        """
        findings = [
            f for f in self._findings.values()
            if f.status not in (FindingStatus.CLOSED, FindingStatus.ACCEPTED)
        ]

        aging_report: List[FindingAge] = []
        now = datetime.now(timezone.utc)

        for finding in findings:
            days_until_sla = None
            if finding.sla_deadline:
                days_until_sla = (finding.sla_deadline - now).days

            aging_report.append(
                FindingAge(
                    finding_id=finding.finding_id,
                    title=finding.title,
                    classification=finding.classification,
                    days_open=finding.days_open,
                    sla_deadline=finding.sla_deadline,
                    days_until_sla=days_until_sla,
                    owner_name=finding.owner_name,
                )
            )

        # Sort by days open (most aged first)
        aging_report.sort(key=lambda a: a.days_open, reverse=True)

        return aging_report


__all__ = [
    "FindingTracker",
    "Finding",
    "FindingCreate",
    "FindingClassification",
    "FindingStatus",
    "FindingSource",
    "FindingSummary",
    "FindingAge",
]
