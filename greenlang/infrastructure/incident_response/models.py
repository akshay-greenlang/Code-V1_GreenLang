# -*- coding: utf-8 -*-
"""
Incident Response Models - SEC-010

Pydantic models for security incidents, alerts, playbook executions, and
related data structures. All models support JSON serialization for API
responses and database storage.

Example:
    >>> from greenlang.infrastructure.incident_response.models import (
    ...     Alert,
    ...     Incident,
    ...     EscalationLevel,
    ... )
    >>> alert = Alert(
    ...     source="prometheus",
    ...     alert_type="high_cpu",
    ...     severity=EscalationLevel.P2,
    ...     message="CPU usage above 90%",
    ... )

Author: GreenLang Security Team
Date: February 2026
Status: Production Ready
"""

from __future__ import annotations

import hashlib
import logging
from datetime import datetime, timezone, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Set
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field, field_validator

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class EscalationLevel(str, Enum):
    """Incident severity/priority levels (P0-P3).

    P0 - Critical: Production down, data breach, active attack
    P1 - High: Major degradation, potential data exposure
    P2 - Medium: Limited impact, isolated issue
    P3 - Low: Minor issue, informational
    """

    P0 = "P0"  # Critical - 15 min response
    P1 = "P1"  # High - 1 hour response
    P2 = "P2"  # Medium - 4 hours response
    P3 = "P3"  # Low - 24 hours response

    @classmethod
    def from_severity_score(cls, score: float) -> EscalationLevel:
        """Convert numeric severity score to escalation level.

        Args:
            score: Severity score (0.0-10.0).

        Returns:
            Corresponding escalation level.
        """
        if score >= 9.0:
            return cls.P0
        elif score >= 7.0:
            return cls.P1
        elif score >= 4.0:
            return cls.P2
        else:
            return cls.P3


class IncidentStatus(str, Enum):
    """Incident lifecycle status.

    Status progression:
        detected -> acknowledged -> investigating -> remediating -> resolved -> closed
    """

    DETECTED = "detected"
    ACKNOWLEDGED = "acknowledged"
    INVESTIGATING = "investigating"
    REMEDIATING = "remediating"
    RESOLVED = "resolved"
    CLOSED = "closed"

    @classmethod
    def get_valid_transitions(cls, current: IncidentStatus) -> Set[IncidentStatus]:
        """Get valid status transitions from current status.

        Args:
            current: Current incident status.

        Returns:
            Set of valid next statuses.
        """
        transitions = {
            cls.DETECTED: {cls.ACKNOWLEDGED, cls.CLOSED},
            cls.ACKNOWLEDGED: {cls.INVESTIGATING, cls.CLOSED},
            cls.INVESTIGATING: {cls.REMEDIATING, cls.RESOLVED, cls.CLOSED},
            cls.REMEDIATING: {cls.RESOLVED, cls.INVESTIGATING, cls.CLOSED},
            cls.RESOLVED: {cls.CLOSED, cls.INVESTIGATING},
            cls.CLOSED: set(),  # Terminal state
        }
        return transitions.get(current, set())


class IncidentType(str, Enum):
    """Types of security incidents."""

    CREDENTIAL_COMPROMISE = "credential_compromise"
    DDOS_ATTACK = "ddos_attack"
    DATA_BREACH = "data_breach"
    MALWARE = "malware"
    UNAUTHORIZED_ACCESS = "unauthorized_access"
    SESSION_HIJACK = "session_hijack"
    BRUTE_FORCE = "brute_force"
    SQL_INJECTION = "sql_injection"
    XSS_ATTACK = "xss_attack"
    API_ABUSE = "api_abuse"
    INSIDER_THREAT = "insider_threat"
    PRIVILEGE_ESCALATION = "privilege_escalation"
    DATA_EXFILTRATION = "data_exfiltration"
    RANSOMWARE = "ransomware"
    PHISHING = "phishing"
    SUPPLY_CHAIN = "supply_chain"
    CONFIGURATION_DRIFT = "configuration_drift"
    COMPLIANCE_VIOLATION = "compliance_violation"
    AVAILABILITY = "availability"
    UNKNOWN = "unknown"


class AlertSource(str, Enum):
    """Sources of security alerts."""

    PROMETHEUS = "prometheus"
    LOKI = "loki"
    GUARDDUTY = "guardduty"
    CLOUDTRAIL = "cloudtrail"
    SECURITY_SCANNER = "security_scanner"
    WAF = "waf"
    IDS = "ids"
    SIEM = "siem"
    MANUAL = "manual"
    API = "api"
    WEBHOOK = "webhook"


class PlaybookStatus(str, Enum):
    """Playbook execution status."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    ROLLED_BACK = "rolled_back"
    REQUIRES_APPROVAL = "requires_approval"


class TimelineEventType(str, Enum):
    """Types of incident timeline events."""

    CREATED = "created"
    STATUS_CHANGE = "status_change"
    ASSIGNED = "assigned"
    ESCALATED = "escalated"
    COMMENT = "comment"
    PLAYBOOK_STARTED = "playbook_started"
    PLAYBOOK_COMPLETED = "playbook_completed"
    PLAYBOOK_FAILED = "playbook_failed"
    NOTIFICATION_SENT = "notification_sent"
    ALERT_CORRELATED = "alert_correlated"
    EVIDENCE_COLLECTED = "evidence_collected"
    RESOLVED = "resolved"
    CLOSED = "closed"
    REOPENED = "reopened"


# ---------------------------------------------------------------------------
# Alert Model
# ---------------------------------------------------------------------------


class Alert(BaseModel):
    """Security alert from monitoring systems.

    Attributes:
        id: Unique alert identifier.
        source: Alert source system.
        alert_type: Type/name of the alert.
        severity: Severity level (P0-P3).
        message: Human-readable alert message.
        description: Detailed description.
        raw_data: Original alert data from source.
        labels: Key-value labels/tags.
        annotations: Additional annotations.
        fingerprint: Deduplication fingerprint.
        received_at: When alert was received.
        starts_at: When alert condition started.
        ends_at: When alert condition ended (if resolved).
        incident_id: Associated incident ID (if correlated).
    """

    model_config = ConfigDict(from_attributes=True)

    id: UUID = Field(default_factory=uuid4, description="Unique alert ID")
    source: AlertSource = Field(..., description="Alert source system")
    alert_type: str = Field(..., description="Alert type/name")
    severity: EscalationLevel = Field(..., description="Alert severity")
    message: str = Field(..., description="Alert message")
    description: Optional[str] = Field(None, description="Detailed description")
    raw_data: Dict[str, Any] = Field(default_factory=dict, description="Raw alert data")
    labels: Dict[str, str] = Field(default_factory=dict, description="Labels/tags")
    annotations: Dict[str, str] = Field(default_factory=dict, description="Annotations")
    fingerprint: Optional[str] = Field(None, description="Dedup fingerprint")
    received_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="When received",
    )
    starts_at: Optional[datetime] = Field(None, description="Alert start time")
    ends_at: Optional[datetime] = Field(None, description="Alert end time")
    incident_id: Optional[UUID] = Field(None, description="Correlated incident")

    def calculate_fingerprint(self) -> str:
        """Calculate fingerprint for deduplication.

        Returns:
            SHA-256 hash fingerprint.
        """
        components = [
            self.source.value,
            self.alert_type,
            self.severity.value,
            str(sorted(self.labels.items())),
        ]
        fingerprint_str = "|".join(components)
        return hashlib.sha256(fingerprint_str.encode()).hexdigest()[:32]

    def model_post_init(self, __context: Any) -> None:
        """Generate fingerprint after initialization."""
        if self.fingerprint is None:
            self.fingerprint = self.calculate_fingerprint()


# ---------------------------------------------------------------------------
# Incident Model
# ---------------------------------------------------------------------------


class Incident(BaseModel):
    """Security incident record.

    Attributes:
        id: Unique incident identifier.
        incident_number: Human-readable incident number (e.g., INC-2026-0001).
        title: Short incident title.
        description: Detailed incident description.
        severity: Incident severity (P0-P3).
        status: Current status.
        incident_type: Type of security incident.
        source: Original alert source.
        detected_at: When incident was detected.
        acknowledged_at: When incident was acknowledged.
        resolved_at: When incident was resolved.
        closed_at: When incident was closed.
        assignee_id: Assigned responder UUID.
        assignee_name: Assigned responder name.
        playbook_id: Associated playbook ID.
        playbook_execution_id: Current playbook execution ID.
        related_alerts: List of correlated alert IDs.
        affected_systems: List of affected systems/services.
        affected_users: Estimated number of affected users.
        tags: Incident tags.
        metadata: Additional metadata.
        provenance_hash: SHA-256 hash for audit trail.
    """

    model_config = ConfigDict(from_attributes=True)

    id: UUID = Field(default_factory=uuid4, description="Unique incident ID")
    incident_number: str = Field(..., description="Human-readable number")
    title: str = Field(..., max_length=200, description="Incident title")
    description: Optional[str] = Field(None, description="Detailed description")
    severity: EscalationLevel = Field(..., description="Severity level")
    status: IncidentStatus = Field(
        default=IncidentStatus.DETECTED,
        description="Current status",
    )
    incident_type: IncidentType = Field(
        default=IncidentType.UNKNOWN,
        description="Incident type",
    )
    source: AlertSource = Field(..., description="Original alert source")

    # Timestamps
    detected_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Detection time",
    )
    acknowledged_at: Optional[datetime] = Field(None, description="Acknowledgment time")
    resolved_at: Optional[datetime] = Field(None, description="Resolution time")
    closed_at: Optional[datetime] = Field(None, description="Closure time")

    # Assignment
    assignee_id: Optional[UUID] = Field(None, description="Assignee UUID")
    assignee_name: Optional[str] = Field(None, description="Assignee name")

    # Playbook
    playbook_id: Optional[str] = Field(None, description="Associated playbook")
    playbook_execution_id: Optional[UUID] = Field(None, description="Execution ID")

    # Related data
    related_alerts: List[UUID] = Field(default_factory=list, description="Alert IDs")
    affected_systems: List[str] = Field(default_factory=list, description="Systems")
    affected_users: Optional[int] = Field(None, description="Affected users count")
    tags: List[str] = Field(default_factory=list, description="Tags")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Metadata")

    # Audit
    provenance_hash: Optional[str] = Field(None, description="Audit hash")

    @field_validator("title")
    @classmethod
    def validate_title(cls, v: str) -> str:
        """Ensure title is not empty."""
        if not v or not v.strip():
            raise ValueError("Title cannot be empty")
        return v.strip()

    def calculate_provenance_hash(self) -> str:
        """Calculate SHA-256 hash for audit trail.

        Returns:
            SHA-256 hash of incident data.
        """
        data = {
            "id": str(self.id),
            "incident_number": self.incident_number,
            "title": self.title,
            "severity": self.severity.value,
            "status": self.status.value,
            "incident_type": self.incident_type.value,
            "detected_at": self.detected_at.isoformat(),
        }
        data_str = str(sorted(data.items()))
        return hashlib.sha256(data_str.encode()).hexdigest()

    def can_transition_to(self, new_status: IncidentStatus) -> bool:
        """Check if transition to new status is valid.

        Args:
            new_status: Target status.

        Returns:
            True if transition is valid.
        """
        valid_transitions = IncidentStatus.get_valid_transitions(self.status)
        return new_status in valid_transitions

    def get_mttd_seconds(self) -> Optional[float]:
        """Calculate Mean Time to Detect in seconds.

        Returns:
            MTTD in seconds, or None if not yet acknowledged.
        """
        if self.acknowledged_at and self.detected_at:
            return (self.acknowledged_at - self.detected_at).total_seconds()
        return None

    def get_mttr_seconds(self) -> Optional[float]:
        """Calculate Mean Time to Respond in seconds.

        Returns:
            MTTR in seconds, or None if not yet resolved.
        """
        if self.resolved_at and self.detected_at:
            return (self.resolved_at - self.detected_at).total_seconds()
        return None

    def get_mtts_seconds(self) -> Optional[float]:
        """Calculate Mean Time to Start (remediation) in seconds.

        Returns:
            MTTS in seconds, or None if not acknowledged.
        """
        if self.acknowledged_at and self.detected_at:
            return (self.acknowledged_at - self.detected_at).total_seconds()
        return None

    def is_sla_breached(self, sla_minutes: int) -> bool:
        """Check if response SLA has been breached.

        Args:
            sla_minutes: SLA threshold in minutes.

        Returns:
            True if SLA is breached.
        """
        if self.status in (IncidentStatus.RESOLVED, IncidentStatus.CLOSED):
            return False

        if self.acknowledged_at:
            return False

        threshold = self.detected_at + timedelta(minutes=sla_minutes)
        return datetime.now(timezone.utc) > threshold


# ---------------------------------------------------------------------------
# Playbook Execution Model
# ---------------------------------------------------------------------------


class PlaybookStep(BaseModel):
    """A single step in a playbook execution.

    Attributes:
        step_number: Step order number.
        name: Step name.
        description: Step description.
        status: Step status.
        started_at: When step started.
        completed_at: When step completed.
        duration_seconds: Step duration.
        output: Step output/result.
        error: Error message if failed.
        rollback_data: Data needed for rollback.
    """

    model_config = ConfigDict(from_attributes=True)

    step_number: int = Field(..., description="Step order")
    name: str = Field(..., description="Step name")
    description: Optional[str] = Field(None, description="Step description")
    status: PlaybookStatus = Field(
        default=PlaybookStatus.PENDING,
        description="Step status",
    )
    started_at: Optional[datetime] = Field(None, description="Start time")
    completed_at: Optional[datetime] = Field(None, description="Completion time")
    duration_seconds: Optional[float] = Field(None, description="Duration")
    output: Optional[Dict[str, Any]] = Field(None, description="Step output")
    error: Optional[str] = Field(None, description="Error message")
    rollback_data: Optional[Dict[str, Any]] = Field(None, description="Rollback data")


class PlaybookExecution(BaseModel):
    """Playbook execution record.

    Attributes:
        id: Unique execution identifier.
        incident_id: Associated incident ID.
        playbook_id: Playbook identifier.
        playbook_name: Human-readable playbook name.
        status: Execution status.
        started_at: When execution started.
        completed_at: When execution completed.
        steps_completed: Number of completed steps.
        steps_total: Total number of steps.
        current_step: Currently executing step.
        steps: List of step details.
        execution_log: Detailed execution log.
        triggered_by: Who/what triggered execution.
        dry_run: Whether this was a dry run.
        rollback_available: Whether rollback is possible.
        metadata: Additional metadata.
    """

    model_config = ConfigDict(from_attributes=True)

    id: UUID = Field(default_factory=uuid4, description="Execution ID")
    incident_id: UUID = Field(..., description="Incident ID")
    playbook_id: str = Field(..., description="Playbook identifier")
    playbook_name: str = Field(..., description="Playbook name")
    status: PlaybookStatus = Field(
        default=PlaybookStatus.PENDING,
        description="Execution status",
    )
    started_at: Optional[datetime] = Field(None, description="Start time")
    completed_at: Optional[datetime] = Field(None, description="Completion time")
    steps_completed: int = Field(default=0, description="Completed steps")
    steps_total: int = Field(..., description="Total steps")
    current_step: Optional[int] = Field(None, description="Current step number")
    steps: List[PlaybookStep] = Field(default_factory=list, description="Step details")
    execution_log: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Execution log",
    )
    triggered_by: Optional[str] = Field(None, description="Trigger source")
    dry_run: bool = Field(default=False, description="Dry run mode")
    rollback_available: bool = Field(default=False, description="Rollback possible")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Metadata")

    def get_duration_seconds(self) -> Optional[float]:
        """Calculate execution duration in seconds.

        Returns:
            Duration in seconds, or None if not completed.
        """
        if self.started_at and self.completed_at:
            return (self.completed_at - self.started_at).total_seconds()
        return None

    def get_progress_percentage(self) -> float:
        """Calculate execution progress percentage.

        Returns:
            Progress percentage (0-100).
        """
        if self.steps_total == 0:
            return 0.0
        return (self.steps_completed / self.steps_total) * 100.0

    def add_log_entry(
        self,
        message: str,
        level: str = "info",
        step_number: Optional[int] = None,
    ) -> None:
        """Add an entry to the execution log.

        Args:
            message: Log message.
            level: Log level (info, warning, error).
            step_number: Associated step number.
        """
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": level,
            "message": message,
        }
        if step_number is not None:
            entry["step"] = step_number
        self.execution_log.append(entry)


# ---------------------------------------------------------------------------
# Timeline Event Model
# ---------------------------------------------------------------------------


class TimelineEvent(BaseModel):
    """Incident timeline event.

    Attributes:
        id: Unique event identifier.
        incident_id: Associated incident ID.
        event_type: Type of event.
        timestamp: When event occurred.
        actor_id: User/system that triggered event.
        actor_name: Actor display name.
        description: Event description.
        old_value: Previous value (for changes).
        new_value: New value (for changes).
        metadata: Additional event data.
    """

    model_config = ConfigDict(from_attributes=True)

    id: UUID = Field(default_factory=uuid4, description="Event ID")
    incident_id: UUID = Field(..., description="Incident ID")
    event_type: TimelineEventType = Field(..., description="Event type")
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Event time",
    )
    actor_id: Optional[UUID] = Field(None, description="Actor UUID")
    actor_name: Optional[str] = Field(None, description="Actor name")
    description: str = Field(..., description="Event description")
    old_value: Optional[str] = Field(None, description="Previous value")
    new_value: Optional[str] = Field(None, description="New value")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Metadata")


# ---------------------------------------------------------------------------
# Post-Mortem Model
# ---------------------------------------------------------------------------


class PostMortem(BaseModel):
    """Incident post-mortem document.

    Attributes:
        id: Unique post-mortem identifier.
        incident_id: Associated incident ID.
        incident_number: Incident number reference.
        title: Post-mortem title.
        summary: Executive summary.
        timeline: Incident timeline.
        root_cause: Root cause analysis.
        impact: Impact assessment.
        detection: How was it detected.
        response: Response summary.
        lessons_learned: Key lessons.
        action_items: Follow-up action items.
        created_at: Creation timestamp.
        created_by: Author UUID.
        status: Draft or published.
        reviewed_by: Reviewers.
    """

    model_config = ConfigDict(from_attributes=True)

    id: UUID = Field(default_factory=uuid4, description="Post-mortem ID")
    incident_id: UUID = Field(..., description="Incident ID")
    incident_number: str = Field(..., description="Incident number")
    title: str = Field(..., description="Title")
    summary: Optional[str] = Field(None, description="Executive summary")
    timeline: List[Dict[str, Any]] = Field(default_factory=list, description="Timeline")
    root_cause: Optional[str] = Field(None, description="Root cause")
    impact: Optional[str] = Field(None, description="Impact assessment")
    detection: Optional[str] = Field(None, description="Detection details")
    response: Optional[str] = Field(None, description="Response summary")
    lessons_learned: List[str] = Field(default_factory=list, description="Lessons")
    action_items: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Action items",
    )
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Creation time",
    )
    created_by: Optional[UUID] = Field(None, description="Author UUID")
    status: str = Field(default="draft", description="Status")
    reviewed_by: List[UUID] = Field(default_factory=list, description="Reviewers")


# ---------------------------------------------------------------------------
# Metrics Summary Model
# ---------------------------------------------------------------------------


class IncidentMetricsSummary(BaseModel):
    """Summary metrics for incident response.

    Attributes:
        period_start: Start of measurement period.
        period_end: End of measurement period.
        total_incidents: Total incidents in period.
        incidents_by_severity: Counts by severity.
        incidents_by_type: Counts by type.
        incidents_by_status: Counts by status.
        mttd_seconds_avg: Average MTTD.
        mttd_seconds_p50: Median MTTD.
        mttd_seconds_p95: 95th percentile MTTD.
        mttr_seconds_avg: Average MTTR.
        mttr_seconds_p50: Median MTTR.
        mttr_seconds_p95: 95th percentile MTTR.
        sla_compliance_rate: SLA compliance percentage.
        playbook_automation_rate: Auto-remediation percentage.
    """

    model_config = ConfigDict(from_attributes=True)

    period_start: datetime = Field(..., description="Period start")
    period_end: datetime = Field(..., description="Period end")
    total_incidents: int = Field(default=0, description="Total incidents")
    incidents_by_severity: Dict[str, int] = Field(
        default_factory=dict,
        description="By severity",
    )
    incidents_by_type: Dict[str, int] = Field(
        default_factory=dict,
        description="By type",
    )
    incidents_by_status: Dict[str, int] = Field(
        default_factory=dict,
        description="By status",
    )
    mttd_seconds_avg: Optional[float] = Field(None, description="Avg MTTD")
    mttd_seconds_p50: Optional[float] = Field(None, description="P50 MTTD")
    mttd_seconds_p95: Optional[float] = Field(None, description="P95 MTTD")
    mttr_seconds_avg: Optional[float] = Field(None, description="Avg MTTR")
    mttr_seconds_p50: Optional[float] = Field(None, description="P50 MTTR")
    mttr_seconds_p95: Optional[float] = Field(None, description="P95 MTTR")
    sla_compliance_rate: Optional[float] = Field(None, description="SLA compliance %")
    playbook_automation_rate: Optional[float] = Field(
        None,
        description="Automation %",
    )


__all__ = [
    # Enums
    "EscalationLevel",
    "IncidentStatus",
    "IncidentType",
    "AlertSource",
    "PlaybookStatus",
    "TimelineEventType",
    # Models
    "Alert",
    "Incident",
    "PlaybookStep",
    "PlaybookExecution",
    "TimelineEvent",
    "PostMortem",
    "IncidentMetricsSummary",
]
