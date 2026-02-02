"""
Event GraphQL Types for Real-Time Subscriptions

Defines GraphQL types for real-time event streaming via subscriptions.
Supports agent events, calculation progress, and system notifications.

Features:
- Agent lifecycle events (created, started, completed, failed)
- Calculation progress streaming
- System health events
- Compliance alerts

Example:
    subscription {
        agentEvents(agentId: "GL-022") {
            eventType
            timestamp
            data
        }
    }
"""

import strawberry
from strawberry.scalars import JSON
from datetime import datetime
from typing import Any, Dict, List, Optional, Union
from enum import Enum


# =============================================================================
# Custom Scalars
# =============================================================================


@strawberry.scalar(
    description="Date and time in ISO 8601 format",
    serialize=lambda v: v.isoformat() if v else None,
    parse_value=lambda v: datetime.fromisoformat(v) if v else None,
)
class DateTime:
    """Custom DateTime scalar for ISO 8601 format."""
    pass


# =============================================================================
# Enums
# =============================================================================


@strawberry.enum
class EventTypeEnum(Enum):
    """Event types for subscriptions."""

    # Agent lifecycle events
    AGENT_CREATED = "agent.created"
    AGENT_UPDATED = "agent.updated"
    AGENT_DELETED = "agent.deleted"
    AGENT_CERTIFIED = "agent.certified"
    AGENT_DEPRECATED = "agent.deprecated"
    AGENT_STATUS_CHANGED = "agent.status_changed"

    # Execution events
    EXECUTION_STARTED = "execution.started"
    EXECUTION_PROGRESS = "execution.progress"
    EXECUTION_COMPLETED = "execution.completed"
    EXECUTION_FAILED = "execution.failed"
    EXECUTION_TIMEOUT = "execution.timeout"
    EXECUTION_CANCELLED = "execution.cancelled"

    # Calculation events
    CALCULATION_STARTED = "calculation.started"
    CALCULATION_PROGRESS = "calculation.progress"
    CALCULATION_COMPLETED = "calculation.completed"
    CALCULATION_FAILED = "calculation.failed"
    CALCULATION_VALIDATED = "calculation.validated"

    # Validation events
    VALIDATION_PASSED = "validation.passed"
    VALIDATION_FAILED = "validation.failed"
    GOLDEN_TEST_PASSED = "goldentest.passed"
    GOLDEN_TEST_FAILED = "goldentest.failed"

    # Compliance events
    COMPLIANCE_CHECK = "compliance.check"
    COMPLIANCE_ALERT = "compliance.alert"
    COMPLIANCE_DEADLINE = "compliance.deadline"
    REGULATORY_UPDATE = "regulatory.update"

    # System events
    SYSTEM_STARTUP = "system.startup"
    SYSTEM_SHUTDOWN = "system.shutdown"
    HEALTH_CHECK = "health.check"
    RATE_LIMIT_EXCEEDED = "ratelimit.exceeded"
    MAINTENANCE_SCHEDULED = "maintenance.scheduled"

    # Audit events
    AUDIT_LOG = "audit.log"
    PROVENANCE_VERIFIED = "provenance.verified"
    DATA_QUALITY_ALERT = "quality.alert"


@strawberry.enum
class EventSeverityEnum(Enum):
    """Event severity levels."""

    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@strawberry.enum
class EventSourceEnum(Enum):
    """Event source types."""

    AGENT = "agent"
    SYSTEM = "system"
    USER = "user"
    SCHEDULER = "scheduler"
    EXTERNAL = "external"
    COMPLIANCE = "compliance"


# =============================================================================
# Base Event Types
# =============================================================================


@strawberry.type
class ProgressType:
    """Progress information for long-running operations."""

    percent: int = strawberry.field(description="Progress percentage (0-100)")
    current_step: str = strawberry.field(description="Current step description")
    total_steps: int = strawberry.field(description="Total number of steps")
    current_step_number: int = strawberry.field(description="Current step number")
    estimated_remaining_seconds: Optional[int] = strawberry.field(
        default=None,
        description="Estimated remaining time in seconds"
    )
    message: Optional[str] = strawberry.field(
        default=None,
        description="Progress message"
    )
    details: JSON = strawberry.field(
        default_factory=dict,
        description="Additional progress details"
    )


@strawberry.type
class AgentEventType:
    """Event from an agent."""

    # Event identification
    event_id: str = strawberry.field(description="Unique event ID")
    event_type: EventTypeEnum = strawberry.field(description="Type of event")
    timestamp: DateTime = strawberry.field(description="Event timestamp")

    # Event source
    source: EventSourceEnum = strawberry.field(description="Event source")
    agent_id: str = strawberry.field(description="Agent ID that generated event")
    agent_name: Optional[str] = strawberry.field(
        default=None,
        description="Agent name"
    )

    # Event data
    severity: EventSeverityEnum = strawberry.field(
        default=EventSeverityEnum.INFO,
        description="Event severity"
    )
    message: str = strawberry.field(description="Event message")
    data: JSON = strawberry.field(
        default_factory=dict,
        description="Event payload data"
    )

    # Context
    execution_id: Optional[str] = strawberry.field(
        default=None,
        description="Related execution ID"
    )
    calculation_id: Optional[str] = strawberry.field(
        default=None,
        description="Related calculation ID"
    )
    correlation_id: Optional[str] = strawberry.field(
        default=None,
        description="Correlation ID for tracing"
    )

    # Multi-tenancy
    tenant_id: str = strawberry.field(description="Tenant ID")
    user_id: Optional[str] = strawberry.field(
        default=None,
        description="User ID if user-initiated"
    )


@strawberry.type
class ExecutionEventType:
    """Execution lifecycle event."""

    event_id: str = strawberry.field(description="Unique event ID")
    event_type: EventTypeEnum = strawberry.field(description="Type of event")
    timestamp: DateTime = strawberry.field(description="Event timestamp")

    # Execution details
    execution_id: str = strawberry.field(description="Execution ID")
    agent_id: str = strawberry.field(description="Agent being executed")
    status: str = strawberry.field(description="Execution status")

    # Progress
    progress: Optional[ProgressType] = strawberry.field(
        default=None,
        description="Execution progress"
    )

    # Results (for completion)
    result_summary: Optional[str] = strawberry.field(
        default=None,
        description="Brief result summary"
    )
    duration_ms: Optional[float] = strawberry.field(
        default=None,
        description="Execution duration in milliseconds"
    )

    # Errors (for failure)
    error_message: Optional[str] = strawberry.field(
        default=None,
        description="Error message if failed"
    )
    error_code: Optional[str] = strawberry.field(
        default=None,
        description="Error code if failed"
    )

    # Context
    tenant_id: str = strawberry.field(description="Tenant ID")
    user_id: Optional[str] = strawberry.field(
        default=None,
        description="User who initiated execution"
    )


@strawberry.type
class CalculationProgressType:
    """Real-time calculation progress update."""

    # Identification
    calculation_id: str = strawberry.field(description="Calculation ID")
    execution_id: str = strawberry.field(description="Execution ID")
    agent_id: str = strawberry.field(description="Agent performing calculation")

    # Status
    status: str = strawberry.field(description="Current status")
    timestamp: DateTime = strawberry.field(description="Update timestamp")

    # Progress
    progress: ProgressType = strawberry.field(description="Progress details")

    # Intermediate results
    intermediate_value: Optional[float] = strawberry.field(
        default=None,
        description="Intermediate result value"
    )
    intermediate_unit: Optional[str] = strawberry.field(
        default=None,
        description="Intermediate result unit"
    )

    # Messages
    message: Optional[str] = strawberry.field(
        default=None,
        description="Progress message"
    )
    warnings: List[str] = strawberry.field(
        default_factory=list,
        description="Warnings encountered"
    )


@strawberry.type
class SystemEventType:
    """System-level event."""

    event_id: str = strawberry.field(description="Unique event ID")
    event_type: EventTypeEnum = strawberry.field(description="Type of event")
    timestamp: DateTime = strawberry.field(description="Event timestamp")

    # System info
    component: str = strawberry.field(description="System component")
    instance_id: str = strawberry.field(description="Instance identifier")
    environment: str = strawberry.field(description="Environment (prod, staging, dev)")

    # Event details
    severity: EventSeverityEnum = strawberry.field(description="Event severity")
    message: str = strawberry.field(description="Event message")
    data: JSON = strawberry.field(
        default_factory=dict,
        description="Event data"
    )

    # Health info (for health events)
    health_score: Optional[float] = strawberry.field(
        default=None,
        description="System health score"
    )
    uptime_seconds: Optional[int] = strawberry.field(
        default=None,
        description="System uptime in seconds"
    )


@strawberry.type
class ComplianceEventType:
    """Compliance-related event."""

    event_id: str = strawberry.field(description="Unique event ID")
    event_type: EventTypeEnum = strawberry.field(description="Type of event")
    timestamp: DateTime = strawberry.field(description="Event timestamp")

    # Compliance details
    framework: str = strawberry.field(description="Regulatory framework (CBAM, CSRD, etc.)")
    requirement_id: Optional[str] = strawberry.field(
        default=None,
        description="Specific requirement ID"
    )

    # Alert info
    severity: EventSeverityEnum = strawberry.field(description="Alert severity")
    message: str = strawberry.field(description="Alert message")
    description: Optional[str] = strawberry.field(
        default=None,
        description="Detailed description"
    )

    # Deadline info
    deadline: Optional[DateTime] = strawberry.field(
        default=None,
        description="Related deadline if applicable"
    )
    days_until_deadline: Optional[int] = strawberry.field(
        default=None,
        description="Days until deadline"
    )

    # Recommendations
    recommended_actions: List[str] = strawberry.field(
        default_factory=list,
        description="Recommended actions"
    )

    # Related entities
    affected_agents: List[str] = strawberry.field(
        default_factory=list,
        description="Affected agent IDs"
    )

    # Context
    tenant_id: str = strawberry.field(description="Tenant ID")


# =============================================================================
# Union Event Type
# =============================================================================


# Note: Strawberry doesn't support Union types in subscriptions well,
# so we use a generic event wrapper instead


@strawberry.type
class GenericEventType:
    """Generic event wrapper for subscriptions."""

    event_id: str = strawberry.field(description="Unique event ID")
    event_type: str = strawberry.field(description="Event type string")
    timestamp: DateTime = strawberry.field(description="Event timestamp")
    source: str = strawberry.field(description="Event source")
    severity: str = strawberry.field(description="Event severity")
    message: str = strawberry.field(description="Event message")
    data: JSON = strawberry.field(description="Event payload")

    # Optional context
    agent_id: Optional[str] = strawberry.field(default=None, description="Agent ID")
    execution_id: Optional[str] = strawberry.field(default=None, description="Execution ID")
    calculation_id: Optional[str] = strawberry.field(default=None, description="Calculation ID")
    correlation_id: Optional[str] = strawberry.field(default=None, description="Correlation ID")
    tenant_id: Optional[str] = strawberry.field(default=None, description="Tenant ID")


# =============================================================================
# Subscription Filter Inputs
# =============================================================================


@strawberry.input
class EventFilterInput:
    """Filter for event subscriptions."""

    event_types: Optional[List[str]] = strawberry.field(
        default=None,
        description="Filter by event types"
    )
    agent_ids: Optional[List[str]] = strawberry.field(
        default=None,
        description="Filter by agent IDs"
    )
    severity: Optional[List[str]] = strawberry.field(
        default=None,
        description="Filter by severity levels"
    )
    source: Optional[str] = strawberry.field(
        default=None,
        description="Filter by source"
    )


# =============================================================================
# Event Stream Types
# =============================================================================


@strawberry.type
class EventStreamMetrics:
    """Metrics for event stream."""

    events_per_minute: float = strawberry.field(description="Events per minute")
    events_per_hour: int = strawberry.field(description="Events in last hour")
    active_subscriptions: int = strawberry.field(description="Active subscription count")
    event_types_active: List[str] = strawberry.field(description="Active event types")
    oldest_event_age_seconds: int = strawberry.field(description="Age of oldest buffered event")


@strawberry.type
class EventHistoryType:
    """Historical event record."""

    events: List[GenericEventType] = strawberry.field(description="Event list")
    total_count: int = strawberry.field(description="Total matching events")
    has_more: bool = strawberry.field(description="More events available")
    oldest_timestamp: Optional[DateTime] = strawberry.field(
        default=None,
        description="Oldest event timestamp"
    )
    newest_timestamp: Optional[DateTime] = strawberry.field(
        default=None,
        description="Newest event timestamp"
    )
