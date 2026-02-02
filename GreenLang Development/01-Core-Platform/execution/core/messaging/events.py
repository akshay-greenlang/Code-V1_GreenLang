# -*- coding: utf-8 -*-
"""
StandardEvents - Event catalog for GreenLang agent communication.

This module defines the standard event types used across all GreenLang agents
for consistent event-driven communication and orchestration.

Example:
    >>> from greenlang.core.messaging import StandardEvents, Event
    >>> event = Event(
    ...     event_id="evt-123",
    ...     event_type=StandardEvents.AGENT_STARTED,
    ...     source_agent="GL-001",
    ...     payload={"status": "ready"}
    ... )

Author: GreenLang Framework Team
Date: December 2025
Status: Production Ready
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, Optional
import uuid


class EventPriority(str, Enum):
    """Event priority levels for message routing and processing."""

    CRITICAL = "critical"  # System-critical events requiring immediate action
    HIGH = "high"  # High-priority events (safety, compliance violations)
    MEDIUM = "medium"  # Standard operational events
    LOW = "low"  # Informational events, logging

    def __lt__(self, other: "EventPriority") -> bool:
        """Enable priority comparison for queue ordering."""
        order = {
            EventPriority.CRITICAL: 0,
            EventPriority.HIGH: 1,
            EventPriority.MEDIUM: 2,
            EventPriority.LOW: 3,
        }
        return order[self] < order[other]


class StandardEvents:
    """
    Standard event types across all GreenLang agents.

    These event types ensure consistent communication patterns across
    the entire GreenLang agent ecosystem.

    Categories:
        - Lifecycle: Agent lifecycle events
        - Calculation: Calculation and processing events
        - Orchestration: Multi-agent coordination events
        - Integration: External system integration events
        - Compliance: Regulatory compliance events
        - Safety: Safety interlock and alert events
        - Data: Data processing and validation events
    """

    # ==================== LIFECYCLE EVENTS ====================
    AGENT_STARTED = "agent.started"
    """Agent has started and is ready to receive requests."""

    AGENT_STOPPED = "agent.stopped"
    """Agent has stopped gracefully."""

    AGENT_ERROR = "agent.error"
    """Agent encountered an error during operation."""

    AGENT_HEARTBEAT = "agent.heartbeat"
    """Agent heartbeat for health monitoring."""

    AGENT_CONFIGURATION_CHANGED = "agent.configuration_changed"
    """Agent configuration has been updated."""

    # ==================== CALCULATION EVENTS ====================
    CALCULATION_STARTED = "calculation.started"
    """Calculation has begun processing."""

    CALCULATION_COMPLETED = "calculation.completed"
    """Calculation has completed successfully."""

    CALCULATION_FAILED = "calculation.failed"
    """Calculation failed with an error."""

    CALCULATION_VALIDATED = "calculation.validated"
    """Calculation results have been validated."""

    CALCULATION_INVALIDATED = "calculation.invalidated"
    """Calculation results failed validation."""

    # ==================== ORCHESTRATION EVENTS ====================
    TASK_ASSIGNED = "orchestration.task_assigned"
    """Task has been assigned to an agent."""

    TASK_STARTED = "orchestration.task_started"
    """Agent has started working on a task."""

    TASK_COMPLETED = "orchestration.task_completed"
    """Agent has completed a task."""

    TASK_FAILED = "orchestration.task_failed"
    """Task execution failed."""

    COORDINATION_REQUESTED = "orchestration.coordination_requested"
    """Coordination between agents has been requested."""

    COORDINATION_COMPLETED = "orchestration.coordination_completed"
    """Multi-agent coordination has completed."""

    WORKFLOW_STARTED = "orchestration.workflow_started"
    """Multi-step workflow has started."""

    WORKFLOW_COMPLETED = "orchestration.workflow_completed"
    """Multi-step workflow has completed."""

    WORKFLOW_FAILED = "orchestration.workflow_failed"
    """Workflow execution failed."""

    # ==================== INTEGRATION EVENTS ====================
    INTEGRATION_CALL_STARTED = "integration.call_started"
    """External system call has started."""

    INTEGRATION_CALL_COMPLETED = "integration.call_completed"
    """External system call completed successfully."""

    INTEGRATION_CALL_FAILED = "integration.call_failed"
    """External system call failed."""

    INTEGRATION_DATA_RECEIVED = "integration.data_received"
    """Data received from external system."""

    INTEGRATION_DATA_SENT = "integration.data_sent"
    """Data sent to external system."""

    INTEGRATION_CONNECTION_ESTABLISHED = "integration.connection_established"
    """Connection to external system established."""

    INTEGRATION_CONNECTION_LOST = "integration.connection_lost"
    """Connection to external system lost."""

    # ==================== COMPLIANCE EVENTS ====================
    COMPLIANCE_CHECK_STARTED = "compliance.check_started"
    """Compliance validation has started."""

    COMPLIANCE_CHECK_PASSED = "compliance.check_passed"
    """All compliance checks passed."""

    COMPLIANCE_VIOLATION_DETECTED = "compliance.violation_detected"
    """Compliance violation has been detected."""

    COMPLIANCE_THRESHOLD_EXCEEDED = "compliance.threshold_exceeded"
    """Compliance threshold has been exceeded."""

    COMPLIANCE_REPORT_GENERATED = "compliance.report_generated"
    """Compliance report has been generated."""

    # ==================== SAFETY EVENTS ====================
    SAFETY_ALERT = "safety.alert"
    """Safety alert has been triggered."""

    SAFETY_INTERLOCK_TRIGGERED = "safety.interlock_triggered"
    """Safety interlock has been activated."""

    SAFETY_INTERLOCK_RELEASED = "safety.interlock_released"
    """Safety interlock has been released."""

    SAFETY_LIMIT_EXCEEDED = "safety.limit_exceeded"
    """Safety limit has been exceeded."""

    SAFETY_EMERGENCY_SHUTDOWN = "safety.emergency_shutdown"
    """Emergency shutdown initiated."""

    # ==================== DATA EVENTS ====================
    DATA_RECEIVED = "data.received"
    """Data has been received for processing."""

    DATA_VALIDATED = "data.validated"
    """Data has passed validation."""

    DATA_VALIDATION_FAILED = "data.validation_failed"
    """Data validation failed."""

    DATA_TRANSFORMED = "data.transformed"
    """Data has been transformed."""

    DATA_STORED = "data.stored"
    """Data has been persisted to storage."""

    DATA_QUALITY_ISSUE = "data.quality_issue"
    """Data quality issue detected."""

    # ==================== AUDIT & PROVENANCE EVENTS ====================
    AUDIT_LOG_CREATED = "audit.log_created"
    """Audit log entry has been created."""

    PROVENANCE_RECORDED = "provenance.recorded"
    """Provenance information has been recorded."""

    PROVENANCE_HASH_COMPUTED = "provenance.hash_computed"
    """Provenance hash has been computed."""

    # ==================== METRICS & MONITORING EVENTS ====================
    METRICS_COLLECTED = "metrics.collected"
    """Metrics have been collected."""

    PERFORMANCE_DEGRADATION = "metrics.performance_degradation"
    """Performance degradation detected."""

    THRESHOLD_WARNING = "metrics.threshold_warning"
    """Metric threshold warning triggered."""

    @classmethod
    def all_events(cls) -> list[str]:
        """Return list of all standard event types."""
        return [
            value
            for name, value in vars(cls).items()
            if isinstance(value, str) and not name.startswith("_")
        ]

    @classmethod
    def is_lifecycle_event(cls, event_type: str) -> bool:
        """Check if event is a lifecycle event."""
        return event_type.startswith("agent.")

    @classmethod
    def is_calculation_event(cls, event_type: str) -> bool:
        """Check if event is a calculation event."""
        return event_type.startswith("calculation.")

    @classmethod
    def is_orchestration_event(cls, event_type: str) -> bool:
        """Check if event is an orchestration event."""
        return event_type.startswith("orchestration.")

    @classmethod
    def is_integration_event(cls, event_type: str) -> bool:
        """Check if event is an integration event."""
        return event_type.startswith("integration.")

    @classmethod
    def is_compliance_event(cls, event_type: str) -> bool:
        """Check if event is a compliance event."""
        return event_type.startswith("compliance.")

    @classmethod
    def is_safety_event(cls, event_type: str) -> bool:
        """Check if event is a safety event."""
        return event_type.startswith("safety.")

    @classmethod
    def is_data_event(cls, event_type: str) -> bool:
        """Check if event is a data event."""
        return event_type.startswith("data.")


@dataclass
class Event:
    """
    Event message for agent communication.

    This is a lightweight event structure that works with the MessageBus.
    It provides a simpler, event-focused interface compared to the full Message class.

    Attributes:
        event_id: Unique event identifier
        event_type: Type of event (use StandardEvents constants)
        source_agent: ID of the agent that emitted the event
        payload: Event data (any serializable content)
        priority: Event priority level
        timestamp: ISO 8601 timestamp of event creation
        correlation_id: Optional ID for tracking related events
        target_agent: Optional specific recipient agent
        metadata: Additional metadata for routing/filtering
    """

    event_id: str
    event_type: str
    source_agent: str
    payload: Dict[str, Any]
    priority: EventPriority = EventPriority.MEDIUM
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    correlation_id: Optional[str] = None
    target_agent: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate event after initialization."""
        if not self.event_id:
            self.event_id = str(uuid.uuid4())
        if not self.source_agent:
            raise ValueError("source_agent is required")
        if not self.event_type:
            raise ValueError("event_type is required")

    def to_dict(self) -> Dict[str, Any]:
        """Convert event to dictionary for serialization."""
        return {
            "event_id": self.event_id,
            "event_type": self.event_type,
            "source_agent": self.source_agent,
            "payload": self.payload,
            "priority": self.priority.value,
            "timestamp": self.timestamp,
            "correlation_id": self.correlation_id,
            "target_agent": self.target_agent,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Event":
        """Create Event from dictionary."""
        return cls(
            event_id=data.get("event_id", str(uuid.uuid4())),
            event_type=data["event_type"],
            source_agent=data["source_agent"],
            payload=data.get("payload", {}),
            priority=EventPriority(data.get("priority", "medium")),
            timestamp=data.get(
                "timestamp", datetime.now(timezone.utc).isoformat()
            ),
            correlation_id=data.get("correlation_id"),
            target_agent=data.get("target_agent"),
            metadata=data.get("metadata", {}),
        )

    def is_high_priority(self) -> bool:
        """Check if event is high or critical priority."""
        return self.priority in (EventPriority.CRITICAL, EventPriority.HIGH)

    def is_safety_related(self) -> bool:
        """Check if event is safety-related."""
        return StandardEvents.is_safety_event(self.event_type)

    def is_compliance_related(self) -> bool:
        """Check if event is compliance-related."""
        return StandardEvents.is_compliance_event(self.event_type)


def create_event(
    event_type: str,
    source_agent: str,
    payload: Dict[str, Any],
    priority: EventPriority = EventPriority.MEDIUM,
    correlation_id: Optional[str] = None,
    target_agent: Optional[str] = None,
) -> Event:
    """
    Factory function for creating events.

    Args:
        event_type: Type of event (use StandardEvents constants)
        source_agent: ID of the agent emitting the event
        payload: Event data
        priority: Event priority level
        correlation_id: Optional correlation ID for tracking
        target_agent: Optional specific recipient

    Returns:
        Configured Event instance

    Example:
        >>> event = create_event(
        ...     event_type=StandardEvents.CALCULATION_COMPLETED,
        ...     source_agent="GL-001",
        ...     payload={"result": 42.5, "units": "kW"}
        ... )
    """
    return Event(
        event_id=str(uuid.uuid4()),
        event_type=event_type,
        source_agent=source_agent,
        payload=payload,
        priority=priority,
        correlation_id=correlation_id,
        target_agent=target_agent,
    )


__all__ = [
    "Event",
    "EventPriority",
    "StandardEvents",
    "create_event",
]
