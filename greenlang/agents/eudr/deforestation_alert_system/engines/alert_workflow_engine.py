# -*- coding: utf-8 -*-
"""
AlertWorkflowEngine - AGENT-EUDR-020 Engine 7: Alert Lifecycle Workflow Management

Manages the full alert lifecycle from initial detection through triage,
investigation, resolution, or escalation. Provides configurable SLA tracking,
assignment management, note-taking, escalation rules, and a complete audit
trail for every state transition.

Workflow States:
    PENDING -> TRIAGED -> INVESTIGATING -> RESOLVED / ESCALATED / FALSE_POSITIVE
    ESCALATED -> INVESTIGATING / RESOLVED
    RESOLVED -> CLOSED / REOPEN (-> INVESTIGATING)
    FALSE_POSITIVE -> REOPEN (-> INVESTIGATING)
    All terminal: CLOSED, EXPIRED

SLA Defaults (configurable):
    - Triage:        4 hours from PENDING
    - Investigation: 48 hours from TRIAGED
    - Resolution:    168 hours (7 days) from INVESTIGATING

Auto-Escalation:
    - When any SLA deadline is breached, the alert auto-escalates to the next
      level (up to max_escalation_levels = 3).
    - Each escalation level halves the remaining SLA window.

Zero-Hallucination Guarantees:
    - All SLA calculations use deterministic datetime arithmetic.
    - State transitions use explicit allowed-transitions validation.
    - Escalation level tracking uses simple integer increment.
    - All timestamps are UTC with microseconds zeroed.
    - SHA-256 provenance hashes on all output objects.
    - No ML/LLM in any workflow decision path.

Performance Targets:
    - State transition: <10ms
    - SLA status check: <5ms
    - Batch SLA audit: <500ms for 1000 alerts

Regulatory References:
    - EUDR Article 10: Risk assessment and timely response.
    - EUDR Article 11: Risk mitigation measures.
    - EUDR Article 31: Five-year record retention for audit trails.

Author: GreenLang Platform Team
Date: March 2026
PRD: PRD-AGENT-EUDR-020, Engine 7 (Alert Workflow Engine)
Agent ID: GL-EUDR-DAS-020
Status: Production Ready
"""

from __future__ import annotations

import hashlib
import json
import logging
import threading
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from decimal import Decimal, ROUND_HALF_UP, InvalidOperation
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module version for provenance tracking
# ---------------------------------------------------------------------------

_MODULE_VERSION: str = "1.0.0"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _utcnow() -> datetime:
    """Return current UTC datetime with microseconds zeroed."""
    return datetime.now(timezone.utc).replace(microsecond=0)


def _compute_hash(data: Any) -> str:
    """Compute a deterministic SHA-256 hash of arbitrary data.

    Args:
        data: Data to hash (dict, dataclass with to_dict, or other).

    Returns:
        SHA-256 hex digest string (64 characters).
    """
    if hasattr(data, "to_dict"):
        serializable = data.to_dict()
    elif isinstance(data, dict):
        serializable = data
    else:
        serializable = str(data)
    raw = json.dumps(serializable, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _generate_id(prefix: str = "wf") -> str:
    """Generate a unique identifier with a given prefix.

    Args:
        prefix: ID prefix string.

    Returns:
        ID in format ``{prefix}-{hex12}``.
    """
    return f"{prefix}-{uuid.uuid4().hex[:12]}"


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------


class WorkflowStatus(str, Enum):
    """Alert workflow lifecycle status.

    Values:
        PENDING: Newly created alert awaiting triage.
        TRIAGED: Alert has been triaged and prioritized.
        INVESTIGATING: Alert is under active investigation.
        RESOLVED: Alert investigation completed with resolution.
        ESCALATED: Alert escalated to higher authority.
        FALSE_POSITIVE: Alert determined to be a false positive.
        EXPIRED: Alert expired due to inactivity.
        CLOSED: Alert fully closed (terminal state).
    """

    PENDING = "PENDING"
    TRIAGED = "TRIAGED"
    INVESTIGATING = "INVESTIGATING"
    RESOLVED = "RESOLVED"
    ESCALATED = "ESCALATED"
    FALSE_POSITIVE = "FALSE_POSITIVE"
    EXPIRED = "EXPIRED"
    CLOSED = "CLOSED"


class WorkflowAction(str, Enum):
    """Actions that trigger workflow state transitions.

    Values:
        TRIAGE: Initial assessment and prioritization.
        ASSIGN: Assign alert to an investigator.
        INVESTIGATE: Begin active investigation.
        RESOLVE: Mark alert as resolved.
        ESCALATE: Escalate alert to higher level.
        CLOSE: Close a resolved or false-positive alert.
        REOPEN: Reopen a resolved or false-positive alert.
        ADD_NOTE: Add a note without state change.
        MARK_FALSE_POSITIVE: Mark alert as false positive.
        EXPIRE: Mark alert as expired (system action).
    """

    TRIAGE = "TRIAGE"
    ASSIGN = "ASSIGN"
    INVESTIGATE = "INVESTIGATE"
    RESOLVE = "RESOLVE"
    ESCALATE = "ESCALATE"
    CLOSE = "CLOSE"
    REOPEN = "REOPEN"
    ADD_NOTE = "ADD_NOTE"
    MARK_FALSE_POSITIVE = "MARK_FALSE_POSITIVE"
    EXPIRE = "EXPIRE"


class AlertPriority(str, Enum):
    """Alert priority levels.

    Values:
        CRITICAL: Immediate action required.
        HIGH: Action required within triage SLA.
        MEDIUM: Action required within standard SLA.
        LOW: Action during normal workflow.
    """

    CRITICAL = "CRITICAL"
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"


class SLAStatus(str, Enum):
    """SLA compliance status.

    Values:
        WITHIN_SLA: Deadline not yet reached.
        AT_RISK: Within 25% of deadline.
        BREACHED: Deadline has passed.
        NOT_APPLICABLE: SLA does not apply to current state.
    """

    WITHIN_SLA = "WITHIN_SLA"
    AT_RISK = "AT_RISK"
    BREACHED = "BREACHED"
    NOT_APPLICABLE = "NOT_APPLICABLE"


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Default SLA deadlines (hours).
DEFAULT_SLA_TRIAGE_HOURS: int = 4
DEFAULT_SLA_INVESTIGATION_HOURS: int = 48
DEFAULT_SLA_RESOLUTION_HOURS: int = 168  # 7 days

#: Maximum escalation levels.
DEFAULT_MAX_ESCALATION_LEVELS: int = 3

#: Priority-based SLA multipliers (lower priority = more time).
PRIORITY_SLA_MULTIPLIERS: Dict[str, Decimal] = {
    AlertPriority.CRITICAL.value: Decimal("0.5"),
    AlertPriority.HIGH.value: Decimal("0.75"),
    AlertPriority.MEDIUM.value: Decimal("1.0"),
    AlertPriority.LOW.value: Decimal("1.5"),
}

#: Escalation SLA reduction factor (each level halves remaining time).
ESCALATION_SLA_FACTOR: Decimal = Decimal("0.5")

#: SLA at-risk threshold (percentage of time remaining).
SLA_AT_RISK_THRESHOLD: Decimal = Decimal("0.25")

#: Valid state transitions map: {from_status: {action: to_status}}.
VALID_TRANSITIONS: Dict[str, Dict[str, str]] = {
    WorkflowStatus.PENDING.value: {
        WorkflowAction.TRIAGE.value: WorkflowStatus.TRIAGED.value,
        WorkflowAction.ESCALATE.value: WorkflowStatus.ESCALATED.value,
        WorkflowAction.EXPIRE.value: WorkflowStatus.EXPIRED.value,
    },
    WorkflowStatus.TRIAGED.value: {
        WorkflowAction.INVESTIGATE.value: WorkflowStatus.INVESTIGATING.value,
        WorkflowAction.ESCALATE.value: WorkflowStatus.ESCALATED.value,
        WorkflowAction.ASSIGN.value: WorkflowStatus.TRIAGED.value,
    },
    WorkflowStatus.INVESTIGATING.value: {
        WorkflowAction.RESOLVE.value: WorkflowStatus.RESOLVED.value,
        WorkflowAction.ESCALATE.value: WorkflowStatus.ESCALATED.value,
        WorkflowAction.MARK_FALSE_POSITIVE.value: WorkflowStatus.FALSE_POSITIVE.value,
    },
    WorkflowStatus.ESCALATED.value: {
        WorkflowAction.INVESTIGATE.value: WorkflowStatus.INVESTIGATING.value,
        WorkflowAction.RESOLVE.value: WorkflowStatus.RESOLVED.value,
    },
    WorkflowStatus.RESOLVED.value: {
        WorkflowAction.CLOSE.value: WorkflowStatus.CLOSED.value,
        WorkflowAction.REOPEN.value: WorkflowStatus.INVESTIGATING.value,
    },
    WorkflowStatus.FALSE_POSITIVE.value: {
        WorkflowAction.REOPEN.value: WorkflowStatus.INVESTIGATING.value,
        WorkflowAction.CLOSE.value: WorkflowStatus.CLOSED.value,
    },
    WorkflowStatus.EXPIRED.value: {
        WorkflowAction.REOPEN.value: WorkflowStatus.INVESTIGATING.value,
    },
    WorkflowStatus.CLOSED.value: {},
}

#: Terminal states (no further transitions except ADD_NOTE).
TERMINAL_STATES: frozenset = frozenset({
    WorkflowStatus.CLOSED.value,
})

#: SLA-applicable states with their corresponding SLA type.
SLA_STATE_MAP: Dict[str, str] = {
    WorkflowStatus.PENDING.value: "triage",
    WorkflowStatus.TRIAGED.value: "investigation",
    WorkflowStatus.INVESTIGATING.value: "resolution",
    WorkflowStatus.ESCALATED.value: "resolution",
}

#: Maximum notes per alert.
MAX_NOTES_PER_ALERT: int = 500

#: Maximum batch size for SLA audit.
MAX_BATCH_SIZE: int = 5000


# ---------------------------------------------------------------------------
# Data Classes
# ---------------------------------------------------------------------------


@dataclass
class WorkflowTransition:
    """Record of a single workflow state transition.

    Attributes:
        transition_id: Unique transition identifier.
        from_status: Previous workflow status.
        to_status: New workflow status.
        action: Action that triggered the transition.
        actor: User or system that performed the action.
        timestamp: When the transition occurred (UTC ISO).
        notes: Optional notes for the transition.
        metadata: Additional transition metadata.
    """

    transition_id: str = ""
    from_status: str = ""
    to_status: str = ""
    action: str = ""
    actor: str = ""
    timestamp: str = ""
    notes: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary.

        Returns:
            Dictionary representation.
        """
        return {
            "transition_id": self.transition_id,
            "from_status": self.from_status,
            "to_status": self.to_status,
            "action": self.action,
            "actor": self.actor,
            "timestamp": self.timestamp,
            "notes": self.notes,
            "metadata": self.metadata,
        }


@dataclass
class WorkflowNote:
    """A note attached to an alert workflow.

    Attributes:
        note_id: Unique note identifier.
        alert_id: Alert this note belongs to.
        author: Note author.
        content: Note content text.
        timestamp: When the note was added.
        category: Note category (investigation, resolution, general).
    """

    note_id: str = ""
    alert_id: str = ""
    author: str = ""
    content: str = ""
    timestamp: str = ""
    category: str = "general"

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "note_id": self.note_id,
            "alert_id": self.alert_id,
            "author": self.author,
            "content": self.content,
            "timestamp": self.timestamp,
            "category": self.category,
        }


@dataclass
class SLAConfig:
    """SLA configuration for workflow deadlines.

    Attributes:
        triage_hours: Hours allowed for triage.
        investigation_hours: Hours allowed for investigation.
        resolution_hours: Hours allowed for resolution.
    """

    triage_hours: int = DEFAULT_SLA_TRIAGE_HOURS
    investigation_hours: int = DEFAULT_SLA_INVESTIGATION_HOURS
    resolution_hours: int = DEFAULT_SLA_RESOLUTION_HOURS

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "triage_hours": self.triage_hours,
            "investigation_hours": self.investigation_hours,
            "resolution_hours": self.resolution_hours,
        }


@dataclass
class WorkflowState:
    """Complete workflow state for an alert.

    Attributes:
        state_id: Unique state identifier.
        alert_id: Alert this state belongs to.
        current_status: Current workflow status.
        previous_status: Previous workflow status.
        assigned_to: Currently assigned investigator.
        priority: Alert priority level.
        sla_deadline: Current SLA deadline (UTC ISO).
        sla_status: Current SLA compliance status.
        escalation_level: Current escalation level (0-3).
        transitions_history: Complete transition history.
        notes: Alert notes.
        resolution: Resolution description (if resolved).
        false_positive_reason: Reason (if false positive).
        created_at: When the workflow was created.
        updated_at: When the workflow was last updated.
        resolved_at: When the workflow was resolved.
        closed_at: When the workflow was closed.
        time_in_current_state_hours: Hours in current state.
        total_processing_hours: Total hours from creation.
        provenance_hash: SHA-256 hash.
    """

    state_id: str = ""
    alert_id: str = ""
    current_status: str = WorkflowStatus.PENDING.value
    previous_status: str = ""
    assigned_to: str = ""
    priority: str = AlertPriority.MEDIUM.value
    sla_deadline: str = ""
    sla_status: str = SLAStatus.WITHIN_SLA.value
    escalation_level: int = 0
    transitions_history: List[Dict[str, Any]] = field(default_factory=list)
    notes: List[Dict[str, Any]] = field(default_factory=list)
    resolution: str = ""
    false_positive_reason: str = ""
    created_at: str = ""
    updated_at: str = ""
    resolved_at: str = ""
    closed_at: str = ""
    time_in_current_state_hours: float = 0.0
    total_processing_hours: float = 0.0
    provenance_hash: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "state_id": self.state_id,
            "alert_id": self.alert_id,
            "current_status": self.current_status,
            "previous_status": self.previous_status,
            "assigned_to": self.assigned_to,
            "priority": self.priority,
            "sla_deadline": self.sla_deadline,
            "sla_status": self.sla_status,
            "escalation_level": self.escalation_level,
            "transitions_history": self.transitions_history,
            "notes": self.notes,
            "resolution": self.resolution,
            "false_positive_reason": self.false_positive_reason,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "resolved_at": self.resolved_at,
            "closed_at": self.closed_at,
            "time_in_current_state_hours": self.time_in_current_state_hours,
            "total_processing_hours": self.total_processing_hours,
            "provenance_hash": self.provenance_hash,
        }


@dataclass
class SLAReport:
    """SLA compliance report for one or more alerts.

    Attributes:
        report_id: Unique report identifier.
        total_alerts: Total alerts assessed.
        within_sla_count: Alerts within SLA.
        at_risk_count: Alerts at risk of SLA breach.
        breached_count: Alerts with breached SLA.
        not_applicable_count: Alerts where SLA does not apply.
        sla_compliance_pct: Overall SLA compliance percentage.
        mean_time_to_triage_hours: Average triage time.
        mean_time_to_resolution_hours: Average resolution time.
        alert_details: Per-alert SLA details.
        report_timestamp: When report was generated.
        provenance_hash: SHA-256 hash.
    """

    report_id: str = ""
    total_alerts: int = 0
    within_sla_count: int = 0
    at_risk_count: int = 0
    breached_count: int = 0
    not_applicable_count: int = 0
    sla_compliance_pct: Decimal = Decimal("0")
    mean_time_to_triage_hours: float = 0.0
    mean_time_to_resolution_hours: float = 0.0
    alert_details: List[Dict[str, Any]] = field(default_factory=list)
    report_timestamp: str = ""
    provenance_hash: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "report_id": self.report_id,
            "total_alerts": self.total_alerts,
            "within_sla_count": self.within_sla_count,
            "at_risk_count": self.at_risk_count,
            "breached_count": self.breached_count,
            "not_applicable_count": self.not_applicable_count,
            "sla_compliance_pct": str(self.sla_compliance_pct),
            "mean_time_to_triage_hours": self.mean_time_to_triage_hours,
            "mean_time_to_resolution_hours": self.mean_time_to_resolution_hours,
            "alert_details": self.alert_details,
            "report_timestamp": self.report_timestamp,
            "provenance_hash": self.provenance_hash,
        }


# ---------------------------------------------------------------------------
# AlertWorkflowEngine
# ---------------------------------------------------------------------------


class AlertWorkflowEngine:
    """Production-grade alert lifecycle workflow management engine.

    Manages deforestation alert workflows from PENDING through TRIAGED,
    INVESTIGATING, RESOLVED/ESCALATED/FALSE_POSITIVE, and CLOSED states.
    Provides configurable SLA tracking, auto-escalation, assignment,
    notes, and complete audit trail for EUDR compliance.

    Thread Safety:
        All mutable state is protected by a reentrant lock.

    Zero-Hallucination:
        All SLA calculations use deterministic datetime arithmetic.
        State transitions use explicit allowed-transitions map.
        No ML/LLM in any workflow decision. No probabilistic routing.

    Attributes:
        _sla_config: SLA deadline configuration.
        _max_escalation_levels: Maximum escalation levels.
        _auto_escalate: Whether to auto-escalate on SLA breach.
        _workflow_states: In-memory workflow state store.
        _lock: Reentrant lock for thread-safe access.

    Example:
        >>> engine = AlertWorkflowEngine()
        >>> result = engine.create_workflow("alert-001")
        >>> assert result["current_status"] == "PENDING"
        >>> result = engine.triage("alert-001", "HIGH", "analyst-1")
        >>> assert result["current_status"] == "TRIAGED"
    """

    def __init__(
        self,
        sla_config: Optional[SLAConfig] = None,
        max_escalation_levels: Optional[int] = None,
        auto_escalate: bool = True,
    ) -> None:
        """Initialize AlertWorkflowEngine.

        Args:
            sla_config: SLA deadline configuration.
            max_escalation_levels: Maximum escalation levels.
            auto_escalate: Enable auto-escalation on SLA breach.
        """
        self._sla_config: SLAConfig = sla_config or SLAConfig()
        self._max_escalation_levels: int = (
            max_escalation_levels if max_escalation_levels is not None
            else DEFAULT_MAX_ESCALATION_LEVELS
        )
        self._auto_escalate: bool = auto_escalate
        self._workflow_states: Dict[str, WorkflowState] = {}
        self._lock: threading.RLock = threading.RLock()

        logger.info(
            "AlertWorkflowEngine initialized (version=%s, triage_h=%d, "
            "investigation_h=%d, resolution_h=%d, max_escalation=%d, "
            "auto_escalate=%s)",
            _MODULE_VERSION,
            self._sla_config.triage_hours,
            self._sla_config.investigation_hours,
            self._sla_config.resolution_hours,
            self._max_escalation_levels,
            self._auto_escalate,
        )

    # ------------------------------------------------------------------
    # Workflow Creation
    # ------------------------------------------------------------------

    def create_workflow(
        self,
        alert_id: str,
        priority: str = AlertPriority.MEDIUM.value,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Create a new workflow for an alert in PENDING state.

        Args:
            alert_id: Alert identifier.
            priority: Initial priority level.
            metadata: Optional initial metadata.

        Returns:
            Dictionary with initial workflow state.

        Raises:
            ValueError: If alert_id is empty or workflow already exists.
        """
        if not alert_id:
            raise ValueError("alert_id must be non-empty")

        with self._lock:
            if alert_id in self._workflow_states:
                raise ValueError(f"Workflow already exists for alert: {alert_id}")

        now = _utcnow()
        sla_deadline = self._calculate_sla_deadline(
            WorkflowStatus.PENDING.value, priority
        )

        state = WorkflowState(
            state_id=_generate_id("ws"),
            alert_id=alert_id,
            current_status=WorkflowStatus.PENDING.value,
            priority=priority,
            sla_deadline=sla_deadline.isoformat(),
            sla_status=SLAStatus.WITHIN_SLA.value,
            created_at=now.isoformat(),
            updated_at=now.isoformat(),
        )
        state.provenance_hash = _compute_hash(state)

        with self._lock:
            self._workflow_states[alert_id] = state

        logger.info("Workflow created: alert=%s priority=%s", alert_id, priority)
        return state.to_dict()

    # ------------------------------------------------------------------
    # Workflow Transitions
    # ------------------------------------------------------------------

    def triage(
        self,
        alert_id: str,
        priority: str,
        actor: str,
        notes: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Move alert from PENDING to TRIAGED.

        Args:
            alert_id: Alert identifier.
            priority: Assigned priority level.
            actor: User performing the triage.
            notes: Optional triage notes.

        Returns:
            Updated workflow state dictionary.

        Raises:
            ValueError: If transition is invalid.
        """
        return self._execute_transition(
            alert_id=alert_id,
            action=WorkflowAction.TRIAGE.value,
            actor=actor,
            notes=notes,
            priority_override=priority,
        )

    def assign(
        self,
        alert_id: str,
        assignee: str,
        actor: str,
        notes: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Assign an alert to an investigator.

        Args:
            alert_id: Alert identifier.
            assignee: User to assign the alert to.
            actor: User performing the assignment.
            notes: Optional assignment notes.

        Returns:
            Updated workflow state dictionary.

        Raises:
            ValueError: If alert not found.
        """
        if not assignee:
            raise ValueError("assignee must be non-empty")

        return self._execute_transition(
            alert_id=alert_id,
            action=WorkflowAction.ASSIGN.value,
            actor=actor,
            notes=notes,
            assignee_override=assignee,
        )

    def investigate(
        self,
        alert_id: str,
        actor: str,
        findings: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Move alert to INVESTIGATING state.

        Args:
            alert_id: Alert identifier.
            actor: User starting the investigation.
            findings: Optional initial findings.

        Returns:
            Updated workflow state dictionary.
        """
        return self._execute_transition(
            alert_id=alert_id,
            action=WorkflowAction.INVESTIGATE.value,
            actor=actor,
            notes=findings,
        )

    def resolve(
        self,
        alert_id: str,
        resolution: str,
        actor: str,
        notes: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Resolve an alert.

        Args:
            alert_id: Alert identifier.
            resolution: Resolution description.
            actor: User resolving the alert.
            notes: Optional resolution notes.

        Returns:
            Updated workflow state dictionary.

        Raises:
            ValueError: If resolution is empty.
        """
        if not resolution:
            raise ValueError("resolution must be non-empty")

        return self._execute_transition(
            alert_id=alert_id,
            action=WorkflowAction.RESOLVE.value,
            actor=actor,
            notes=notes,
            resolution_text=resolution,
        )

    def escalate(
        self,
        alert_id: str,
        reason: str,
        actor: str,
        notes: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Escalate an alert to a higher level.

        Args:
            alert_id: Alert identifier.
            reason: Escalation reason.
            actor: User escalating the alert.
            notes: Optional escalation notes.

        Returns:
            Updated workflow state dictionary.
        """
        if not reason:
            raise ValueError("reason must be non-empty")

        return self._execute_transition(
            alert_id=alert_id,
            action=WorkflowAction.ESCALATE.value,
            actor=actor,
            notes=notes or reason,
            escalation_reason=reason,
        )

    def mark_false_positive(
        self,
        alert_id: str,
        reason: str,
        actor: str,
        notes: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Mark an alert as false positive.

        Args:
            alert_id: Alert identifier.
            reason: Reason for false positive determination.
            actor: User making the determination.
            notes: Optional notes.

        Returns:
            Updated workflow state dictionary.
        """
        if not reason:
            raise ValueError("reason must be non-empty")

        return self._execute_transition(
            alert_id=alert_id,
            action=WorkflowAction.MARK_FALSE_POSITIVE.value,
            actor=actor,
            notes=notes,
            false_positive_reason=reason,
        )

    def close(
        self,
        alert_id: str,
        actor: str,
        notes: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Close a resolved or false-positive alert.

        Args:
            alert_id: Alert identifier.
            actor: User closing the alert.
            notes: Optional closing notes.

        Returns:
            Updated workflow state dictionary.
        """
        return self._execute_transition(
            alert_id=alert_id,
            action=WorkflowAction.CLOSE.value,
            actor=actor,
            notes=notes,
        )

    def reopen(
        self,
        alert_id: str,
        reason: str,
        actor: str,
    ) -> Dict[str, Any]:
        """Reopen a resolved, false-positive, or expired alert.

        Args:
            alert_id: Alert identifier.
            reason: Reason for reopening.
            actor: User reopening the alert.

        Returns:
            Updated workflow state dictionary.
        """
        if not reason:
            raise ValueError("reason must be non-empty")

        return self._execute_transition(
            alert_id=alert_id,
            action=WorkflowAction.REOPEN.value,
            actor=actor,
            notes=reason,
        )

    def add_note(
        self,
        alert_id: str,
        content: str,
        author: str,
        category: str = "general",
    ) -> Dict[str, Any]:
        """Add a note to an alert without changing state.

        Args:
            alert_id: Alert identifier.
            content: Note content.
            author: Note author.
            category: Note category.

        Returns:
            Updated workflow state dictionary.

        Raises:
            ValueError: If alert not found or content empty.
        """
        if not content:
            raise ValueError("content must be non-empty")

        state = self._get_state(alert_id)

        note = WorkflowNote(
            note_id=_generate_id("nt"),
            alert_id=alert_id,
            author=author,
            content=content,
            timestamp=_utcnow().isoformat(),
            category=category,
        )

        with self._lock:
            if len(state.notes) >= MAX_NOTES_PER_ALERT:
                raise ValueError(
                    f"Maximum notes ({MAX_NOTES_PER_ALERT}) reached for alert {alert_id}"
                )
            state.notes.append(note.to_dict())
            state.updated_at = _utcnow().isoformat()
            state.provenance_hash = _compute_hash(state)

        logger.info("Note added: alert=%s author=%s", alert_id, author)
        return state.to_dict()

    # ------------------------------------------------------------------
    # SLA Management
    # ------------------------------------------------------------------

    def get_sla_status(
        self,
        alert_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Check SLA compliance status for one or all alerts.

        Args:
            alert_id: Specific alert to check, or None for all.

        Returns:
            SLA report dictionary.
        """
        now = _utcnow()

        if alert_id:
            state = self._get_state(alert_id)
            sla_info = self._compute_sla_status(state, now)
            return sla_info

        # All alerts
        with self._lock:
            all_states = list(self._workflow_states.values())

        within = 0
        at_risk = 0
        breached = 0
        not_applicable = 0
        details: List[Dict[str, Any]] = []
        triage_times: List[float] = []
        resolution_times: List[float] = []

        for state in all_states:
            sla_info = self._compute_sla_status(state, now)
            details.append(sla_info)

            status = sla_info.get("sla_status", "NOT_APPLICABLE")
            if status == SLAStatus.WITHIN_SLA.value:
                within += 1
            elif status == SLAStatus.AT_RISK.value:
                at_risk += 1
            elif status == SLAStatus.BREACHED.value:
                breached += 1
            else:
                not_applicable += 1

            # Track triage and resolution times
            if sla_info.get("time_to_triage_hours") is not None:
                triage_times.append(sla_info["time_to_triage_hours"])
            if sla_info.get("time_to_resolution_hours") is not None:
                resolution_times.append(sla_info["time_to_resolution_hours"])

        total = len(all_states)
        applicable = within + at_risk + breached
        compliance_pct = (
            (Decimal(str(within + at_risk)) / Decimal(str(applicable)) * Decimal("100"))
            .quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
            if applicable > 0
            else Decimal("100")
        )

        mean_triage = (
            sum(triage_times) / len(triage_times) if triage_times else 0.0
        )
        mean_resolution = (
            sum(resolution_times) / len(resolution_times)
            if resolution_times else 0.0
        )

        report = SLAReport(
            report_id=_generate_id("sla"),
            total_alerts=total,
            within_sla_count=within,
            at_risk_count=at_risk,
            breached_count=breached,
            not_applicable_count=not_applicable,
            sla_compliance_pct=compliance_pct,
            mean_time_to_triage_hours=round(mean_triage, 2),
            mean_time_to_resolution_hours=round(mean_resolution, 2),
            alert_details=details,
            report_timestamp=now.isoformat(),
        )
        report.provenance_hash = _compute_hash(report)

        return report.to_dict()

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def get_workflow_state(self, alert_id: str) -> Dict[str, Any]:
        """Get current workflow state for an alert.

        Args:
            alert_id: Alert identifier.

        Returns:
            Workflow state dictionary.

        Raises:
            ValueError: If alert not found.
        """
        state = self._get_state(alert_id)
        # Update time tracking
        self._update_time_tracking(state)
        return state.to_dict()

    def get_alerts_by_status(
        self,
        status: str,
    ) -> List[Dict[str, Any]]:
        """Get all alerts with a given workflow status.

        Args:
            status: Workflow status to filter by.

        Returns:
            List of workflow state dictionaries.
        """
        with self._lock:
            matching = [
                s.to_dict() for s in self._workflow_states.values()
                if s.current_status == status
            ]
        return matching

    def get_statistics(self) -> Dict[str, Any]:
        """Get engine statistics.

        Returns:
            Dictionary with engine state and configuration.
        """
        with self._lock:
            total = len(self._workflow_states)
            status_counts: Dict[str, int] = {}
            for state in self._workflow_states.values():
                sc = state.current_status
                status_counts[sc] = status_counts.get(sc, 0) + 1

        return {
            "engine": "AlertWorkflowEngine",
            "version": _MODULE_VERSION,
            "total_workflows": total,
            "status_counts": status_counts,
            "sla_config": self._sla_config.to_dict(),
            "max_escalation_levels": self._max_escalation_levels,
            "auto_escalate": self._auto_escalate,
        }

    # ------------------------------------------------------------------
    # Internal: Transition Execution
    # ------------------------------------------------------------------

    def _execute_transition(
        self,
        alert_id: str,
        action: str,
        actor: str,
        notes: Optional[str] = None,
        priority_override: Optional[str] = None,
        assignee_override: Optional[str] = None,
        resolution_text: Optional[str] = None,
        false_positive_reason: Optional[str] = None,
        escalation_reason: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Execute a workflow state transition.

        Validates the transition, records it, updates state, recalculates
        SLA deadlines, and optionally auto-escalates.

        Args:
            alert_id: Alert identifier.
            action: Workflow action to execute.
            actor: User or system performing the action.
            notes: Optional transition notes.
            priority_override: Override priority level.
            assignee_override: Override assignee.
            resolution_text: Resolution description.
            false_positive_reason: False positive reason.
            escalation_reason: Escalation reason.

        Returns:
            Updated workflow state dictionary.

        Raises:
            ValueError: If transition is invalid.
        """
        start_time = time.monotonic()

        state = self._get_state(alert_id)
        current = state.current_status

        # Validate transition
        if not self._validate_transition(current, action):
            raise ValueError(
                f"Invalid transition: {current} -> {action}. "
                f"Allowed actions from {current}: "
                f"{list(VALID_TRANSITIONS.get(current, {}).keys())}"
            )

        # Get target status
        target = VALID_TRANSITIONS[current][action]
        now = _utcnow()

        # Record transition
        transition = WorkflowTransition(
            transition_id=_generate_id("tr"),
            from_status=current,
            to_status=target,
            action=action,
            actor=actor,
            timestamp=now.isoformat(),
            notes=notes or "",
            metadata={
                "escalation_reason": escalation_reason or "",
                "resolution_text": resolution_text or "",
                "false_positive_reason": false_positive_reason or "",
            },
        )

        with self._lock:
            # Update state
            state.previous_status = current
            state.current_status = target
            state.updated_at = now.isoformat()
            state.transitions_history.append(transition.to_dict())

            if priority_override:
                state.priority = priority_override

            if assignee_override:
                state.assigned_to = assignee_override

            if resolution_text:
                state.resolution = resolution_text
                state.resolved_at = now.isoformat()

            if false_positive_reason:
                state.false_positive_reason = false_positive_reason

            if action == WorkflowAction.CLOSE.value:
                state.closed_at = now.isoformat()

            # Handle escalation level
            if action == WorkflowAction.ESCALATE.value:
                state.escalation_level = min(
                    state.escalation_level + 1,
                    self._max_escalation_levels,
                )

            # Recalculate SLA deadline for new state
            if target in SLA_STATE_MAP:
                sla_deadline = self._calculate_sla_deadline(
                    target,
                    state.priority,
                    state.escalation_level,
                )
                state.sla_deadline = sla_deadline.isoformat()
                state.sla_status = SLAStatus.WITHIN_SLA.value
            else:
                state.sla_status = SLAStatus.NOT_APPLICABLE.value

            # Add note if provided
            if notes:
                note = WorkflowNote(
                    note_id=_generate_id("nt"),
                    alert_id=alert_id,
                    author=actor,
                    content=notes,
                    timestamp=now.isoformat(),
                    category=action.lower(),
                )
                state.notes.append(note.to_dict())

            # Update time tracking
            self._update_time_tracking(state)

            state.provenance_hash = _compute_hash(state)

        processing_ms = (time.monotonic() - start_time) * 1000.0

        logger.info(
            "Workflow transition: alert=%s %s -> %s (action=%s, actor=%s, "
            "escalation=%d) time_ms=%.1f",
            alert_id, current, target, action, actor,
            state.escalation_level, processing_ms,
        )

        result = state.to_dict()
        result["processing_time_ms"] = round(processing_ms, 3)
        return result

    # ------------------------------------------------------------------
    # Internal: Validation
    # ------------------------------------------------------------------

    def _validate_transition(self, from_status: str, action: str) -> bool:
        """Validate that a transition is allowed.

        Args:
            from_status: Current workflow status.
            action: Proposed action.

        Returns:
            True if transition is valid.
        """
        allowed = VALID_TRANSITIONS.get(from_status, {})
        return action in allowed

    # ------------------------------------------------------------------
    # Internal: SLA Calculation
    # ------------------------------------------------------------------

    def _calculate_sla_deadline(
        self,
        status: str,
        priority: str,
        escalation_level: int = 0,
    ) -> datetime:
        """Calculate SLA deadline for a given state and priority.

        Args:
            status: Target workflow status.
            priority: Alert priority.
            escalation_level: Current escalation level.

        Returns:
            SLA deadline as UTC datetime.
        """
        now = _utcnow()
        sla_type = SLA_STATE_MAP.get(status, "resolution")

        # Base hours from SLA config
        if sla_type == "triage":
            base_hours = self._sla_config.triage_hours
        elif sla_type == "investigation":
            base_hours = self._sla_config.investigation_hours
        else:
            base_hours = self._sla_config.resolution_hours

        # Apply priority multiplier
        multiplier = PRIORITY_SLA_MULTIPLIERS.get(
            priority, Decimal("1.0")
        )
        adjusted_hours = Decimal(str(base_hours)) * multiplier

        # Apply escalation reduction
        for _ in range(escalation_level):
            adjusted_hours = adjusted_hours * ESCALATION_SLA_FACTOR

        # Minimum 1 hour
        final_hours = max(Decimal("1"), adjusted_hours)

        deadline = now + timedelta(hours=float(final_hours))
        return deadline.replace(microsecond=0)

    def _compute_sla_status(
        self,
        state: WorkflowState,
        now: datetime,
    ) -> Dict[str, Any]:
        """Compute SLA status for a workflow state.

        Args:
            state: Workflow state to assess.
            now: Current UTC time.

        Returns:
            Dictionary with SLA status details.
        """
        result: Dict[str, Any] = {
            "alert_id": state.alert_id,
            "current_status": state.current_status,
            "priority": state.priority,
            "sla_deadline": state.sla_deadline,
            "escalation_level": state.escalation_level,
        }

        if state.current_status not in SLA_STATE_MAP:
            result["sla_status"] = SLAStatus.NOT_APPLICABLE.value
            result["time_remaining_hours"] = None
            return result

        if not state.sla_deadline:
            result["sla_status"] = SLAStatus.NOT_APPLICABLE.value
            return result

        try:
            deadline = datetime.fromisoformat(state.sla_deadline)
            if deadline.tzinfo is None:
                deadline = deadline.replace(tzinfo=timezone.utc)
        except ValueError:
            result["sla_status"] = SLAStatus.NOT_APPLICABLE.value
            return result

        remaining = (deadline - now).total_seconds() / 3600.0

        if remaining <= 0:
            sla_status = SLAStatus.BREACHED.value
        else:
            # Calculate total SLA window
            try:
                created = datetime.fromisoformat(state.created_at)
                if created.tzinfo is None:
                    created = created.replace(tzinfo=timezone.utc)
                total_window = (deadline - created).total_seconds() / 3600.0
            except (ValueError, TypeError):
                total_window = remaining + 1.0

            if total_window > 0:
                remaining_fraction = Decimal(str(remaining)) / Decimal(str(total_window))
                if remaining_fraction <= SLA_AT_RISK_THRESHOLD:
                    sla_status = SLAStatus.AT_RISK.value
                else:
                    sla_status = SLAStatus.WITHIN_SLA.value
            else:
                sla_status = SLAStatus.BREACHED.value

        result["sla_status"] = sla_status
        result["time_remaining_hours"] = round(remaining, 2)

        # Calculate triage time if applicable
        if state.transitions_history:
            for tr in state.transitions_history:
                if tr.get("action") == WorkflowAction.TRIAGE.value:
                    try:
                        created = datetime.fromisoformat(state.created_at)
                        triaged = datetime.fromisoformat(tr["timestamp"])
                        if created.tzinfo is None:
                            created = created.replace(tzinfo=timezone.utc)
                        if triaged.tzinfo is None:
                            triaged = triaged.replace(tzinfo=timezone.utc)
                        triage_hours = (triaged - created).total_seconds() / 3600.0
                        result["time_to_triage_hours"] = round(triage_hours, 2)
                    except (ValueError, KeyError):
                        pass
                    break

        # Resolution time
        if state.resolved_at:
            try:
                created = datetime.fromisoformat(state.created_at)
                resolved = datetime.fromisoformat(state.resolved_at)
                if created.tzinfo is None:
                    created = created.replace(tzinfo=timezone.utc)
                if resolved.tzinfo is None:
                    resolved = resolved.replace(tzinfo=timezone.utc)
                resolution_hours = (resolved - created).total_seconds() / 3600.0
                result["time_to_resolution_hours"] = round(resolution_hours, 2)
            except (ValueError, TypeError):
                pass

        return result

    # ------------------------------------------------------------------
    # Internal: Auto-Escalation Check
    # ------------------------------------------------------------------

    def check_auto_escalation(self) -> List[str]:
        """Check all alerts for SLA breaches and auto-escalate.

        Returns:
            List of alert IDs that were auto-escalated.
        """
        if not self._auto_escalate:
            return []

        now = _utcnow()
        escalated: List[str] = []

        with self._lock:
            alerts_to_check = [
                (aid, state) for aid, state in self._workflow_states.items()
                if state.current_status in SLA_STATE_MAP
                and state.escalation_level < self._max_escalation_levels
            ]

        for alert_id, state in alerts_to_check:
            sla_info = self._compute_sla_status(state, now)
            if sla_info.get("sla_status") == SLAStatus.BREACHED.value:
                try:
                    # Auto-escalate if allowed from current state
                    if self._validate_transition(
                        state.current_status,
                        WorkflowAction.ESCALATE.value,
                    ):
                        self._execute_transition(
                            alert_id=alert_id,
                            action=WorkflowAction.ESCALATE.value,
                            actor="system:auto-escalation",
                            notes=f"Auto-escalated due to SLA breach "
                                  f"(level {state.escalation_level + 1})",
                            escalation_reason="SLA_BREACH",
                        )
                        escalated.append(alert_id)
                except (ValueError, Exception) as exc:
                    logger.warning(
                        "Auto-escalation failed for alert %s: %s",
                        alert_id, exc,
                    )

        if escalated:
            logger.info(
                "Auto-escalation: %d alerts escalated due to SLA breaches",
                len(escalated),
            )

        return escalated

    # ------------------------------------------------------------------
    # Internal: State Access
    # ------------------------------------------------------------------

    def _get_state(self, alert_id: str) -> WorkflowState:
        """Get workflow state by alert ID.

        Args:
            alert_id: Alert identifier.

        Returns:
            WorkflowState object.

        Raises:
            ValueError: If alert not found.
        """
        if not alert_id:
            raise ValueError("alert_id must be non-empty")

        with self._lock:
            state = self._workflow_states.get(alert_id)
            if state is None:
                raise ValueError(f"No workflow found for alert: {alert_id}")
            return state

    def _update_time_tracking(self, state: WorkflowState) -> None:
        """Update time-in-state and total processing time.

        Args:
            state: WorkflowState to update.
        """
        now = _utcnow()
        try:
            created = datetime.fromisoformat(state.created_at)
            if created.tzinfo is None:
                created = created.replace(tzinfo=timezone.utc)
            state.total_processing_hours = round(
                (now - created).total_seconds() / 3600.0, 2
            )
        except (ValueError, TypeError):
            pass

        try:
            updated = datetime.fromisoformat(state.updated_at)
            if updated.tzinfo is None:
                updated = updated.replace(tzinfo=timezone.utc)
            state.time_in_current_state_hours = round(
                (now - updated).total_seconds() / 3600.0, 2
            )
        except (ValueError, TypeError):
            pass

    def clear_all(self) -> int:
        """Clear all workflow states.

        Returns:
            Number of workflows cleared.
        """
        with self._lock:
            count = len(self._workflow_states)
            self._workflow_states.clear()
        logger.info("Cleared all workflows (%d entries)", count)
        return count
