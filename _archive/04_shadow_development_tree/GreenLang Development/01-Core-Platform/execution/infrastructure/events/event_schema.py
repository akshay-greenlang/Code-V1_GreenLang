"""
Event Schema Definitions for GreenLang

This module provides Avro-compatible event schema definitions for
GreenLang's event-driven architecture.

Features:
- Base event schemas
- Domain and integration events
- Schema versioning
- Avro serialization support
- Schema registry integration

Example:
    >>> schema = EventSchema.create_schema("EmissionCalculated")
    >>> event = DomainEvent(type="EmissionCalculated", data={"co2": 150.5})
"""

import hashlib
import json
import logging
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Type, TypeVar
from uuid import uuid4

from pydantic import BaseModel, Field, validator

logger = logging.getLogger(__name__)

T = TypeVar("T", bound="BaseEvent")


class EventType(str, Enum):
    """Standard event types in GreenLang."""
    # Domain Events
    EMISSION_CALCULATED = "emission.calculated"
    EMISSION_VALIDATED = "emission.validated"
    EMISSION_REPORTED = "emission.reported"

    # Compliance Events
    COMPLIANCE_CHECK_STARTED = "compliance.check.started"
    COMPLIANCE_CHECK_COMPLETED = "compliance.check.completed"
    COMPLIANCE_VIOLATION_DETECTED = "compliance.violation.detected"

    # Data Events
    DATA_INGESTED = "data.ingested"
    DATA_TRANSFORMED = "data.transformed"
    DATA_VALIDATED = "data.validated"
    DATA_EXPORTED = "data.exported"

    # Agent Events
    AGENT_STARTED = "agent.started"
    AGENT_COMPLETED = "agent.completed"
    AGENT_FAILED = "agent.failed"
    AGENT_RETRY = "agent.retry"

    # Saga Events
    SAGA_STARTED = "saga.started"
    SAGA_STEP_COMPLETED = "saga.step.completed"
    SAGA_STEP_FAILED = "saga.step.failed"
    SAGA_COMPLETED = "saga.completed"
    SAGA_COMPENSATED = "saga.compensated"

    # Audit Events
    AUDIT_ENTRY_CREATED = "audit.entry.created"
    AUDIT_TRAIL_SEALED = "audit.trail.sealed"


class EventPriority(str, Enum):
    """Event priority levels."""
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    CRITICAL = "critical"


class EventMetadata(BaseModel):
    """Metadata for all events."""
    event_id: str = Field(default_factory=lambda: str(uuid4()))
    correlation_id: Optional[str] = Field(default=None, description="Correlation ID for tracing")
    causation_id: Optional[str] = Field(default=None, description="ID of causing event")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    version: int = Field(default=1, description="Event schema version")
    source: str = Field(default="greenlang", description="Event source system")
    priority: EventPriority = Field(default=EventPriority.NORMAL)
    trace_id: Optional[str] = Field(default=None, description="Distributed tracing ID")
    span_id: Optional[str] = Field(default=None, description="Span ID for tracing")
    user_id: Optional[str] = Field(default=None, description="User who triggered event")
    tenant_id: Optional[str] = Field(default=None, description="Multi-tenant ID")

    def with_correlation(self, correlation_id: str) -> "EventMetadata":
        """Create new metadata with correlation ID."""
        return self.copy(update={"correlation_id": correlation_id})


class BaseEvent(BaseModel):
    """
    Base event model for all GreenLang events.

    All events in the system inherit from this base class,
    which provides common fields and serialization.

    Attributes:
        event_type: Type of the event
        metadata: Event metadata
        data: Event payload
        provenance_hash: SHA-256 hash for audit trail
    """
    event_type: str = Field(..., description="Event type identifier")
    metadata: EventMetadata = Field(default_factory=EventMetadata)
    data: Dict[str, Any] = Field(default_factory=dict, description="Event payload")
    provenance_hash: str = Field(default="", description="SHA-256 provenance hash")

    def __init__(self, **kwargs):
        """Initialize event and calculate provenance hash."""
        super().__init__(**kwargs)
        if not self.provenance_hash:
            self.provenance_hash = self._calculate_provenance()

    def _calculate_provenance(self) -> str:
        """Calculate SHA-256 provenance hash."""
        hash_data = {
            "event_type": self.event_type,
            "event_id": self.metadata.event_id,
            "timestamp": self.metadata.timestamp.isoformat(),
            "data": self.data,
        }
        hash_str = json.dumps(hash_data, sort_keys=True)
        return hashlib.sha256(hash_str.encode()).hexdigest()

    def to_avro_dict(self) -> Dict[str, Any]:
        """Convert to Avro-compatible dictionary."""
        return {
            "event_type": self.event_type,
            "event_id": self.metadata.event_id,
            "correlation_id": self.metadata.correlation_id,
            "causation_id": self.metadata.causation_id,
            "timestamp": self.metadata.timestamp.isoformat(),
            "version": self.metadata.version,
            "source": self.metadata.source,
            "priority": self.metadata.priority.value,
            "data": self.data,
            "provenance_hash": self.provenance_hash,
        }

    @classmethod
    def from_avro_dict(cls: Type[T], avro_dict: Dict[str, Any]) -> T:
        """Create event from Avro dictionary."""
        metadata = EventMetadata(
            event_id=avro_dict.get("event_id", str(uuid4())),
            correlation_id=avro_dict.get("correlation_id"),
            causation_id=avro_dict.get("causation_id"),
            timestamp=datetime.fromisoformat(avro_dict["timestamp"]),
            version=avro_dict.get("version", 1),
            source=avro_dict.get("source", "greenlang"),
            priority=EventPriority(avro_dict.get("priority", "normal")),
        )
        return cls(
            event_type=avro_dict["event_type"],
            metadata=metadata,
            data=avro_dict.get("data", {}),
            provenance_hash=avro_dict.get("provenance_hash", ""),
        )

    def create_follow_up(
        self,
        event_type: str,
        data: Dict[str, Any]
    ) -> "BaseEvent":
        """Create a follow-up event with correlation."""
        return BaseEvent(
            event_type=event_type,
            metadata=EventMetadata(
                correlation_id=self.metadata.correlation_id or self.metadata.event_id,
                causation_id=self.metadata.event_id,
                source=self.metadata.source,
                tenant_id=self.metadata.tenant_id,
            ),
            data=data,
        )


class DomainEvent(BaseEvent):
    """
    Domain event for internal bounded context events.

    Domain events represent something that happened in the domain
    and are used for internal communication within a bounded context.

    Example:
        >>> event = DomainEvent(
        ...     event_type="emission.calculated",
        ...     aggregate_id="scope1-2024-001",
        ...     aggregate_type="EmissionReport",
        ...     data={"total_co2": 1500.5}
        ... )
    """
    aggregate_id: str = Field(..., description="Aggregate root identifier")
    aggregate_type: str = Field(..., description="Type of aggregate")
    aggregate_version: int = Field(default=1, description="Aggregate version")

    def to_avro_dict(self) -> Dict[str, Any]:
        """Convert to Avro-compatible dictionary."""
        base = super().to_avro_dict()
        base.update({
            "aggregate_id": self.aggregate_id,
            "aggregate_type": self.aggregate_type,
            "aggregate_version": self.aggregate_version,
        })
        return base


class IntegrationEvent(BaseEvent):
    """
    Integration event for cross-boundary communication.

    Integration events are used for communication between
    bounded contexts or external systems.

    Example:
        >>> event = IntegrationEvent(
        ...     event_type="report.published",
        ...     target_systems=["erp", "reporting"],
        ...     data={"report_id": "RPT-2024-001"}
        ... )
    """
    target_systems: List[str] = Field(
        default_factory=list,
        description="Target systems for this event"
    )
    requires_ack: bool = Field(
        default=False,
        description="Whether acknowledgment is required"
    )
    ttl_seconds: Optional[int] = Field(
        default=None,
        description="Time-to-live in seconds"
    )

    def to_avro_dict(self) -> Dict[str, Any]:
        """Convert to Avro-compatible dictionary."""
        base = super().to_avro_dict()
        base.update({
            "target_systems": self.target_systems,
            "requires_ack": self.requires_ack,
            "ttl_seconds": self.ttl_seconds,
        })
        return base


class EventSchema:
    """
    Avro schema manager for GreenLang events.

    Provides schema definitions and validation for event serialization.
    """

    # Base event schema in Avro format
    BASE_SCHEMA = {
        "type": "record",
        "name": "GreenLangEvent",
        "namespace": "com.greenlang.events",
        "fields": [
            {"name": "event_type", "type": "string"},
            {"name": "event_id", "type": "string"},
            {"name": "correlation_id", "type": ["null", "string"], "default": None},
            {"name": "causation_id", "type": ["null", "string"], "default": None},
            {"name": "timestamp", "type": "string"},
            {"name": "version", "type": "int", "default": 1},
            {"name": "source", "type": "string", "default": "greenlang"},
            {"name": "priority", "type": "string", "default": "normal"},
            {"name": "data", "type": {"type": "map", "values": "string"}},
            {"name": "provenance_hash", "type": "string"},
        ]
    }

    # Domain event schema
    DOMAIN_EVENT_SCHEMA = {
        "type": "record",
        "name": "DomainEvent",
        "namespace": "com.greenlang.events",
        "fields": [
            *BASE_SCHEMA["fields"],
            {"name": "aggregate_id", "type": "string"},
            {"name": "aggregate_type", "type": "string"},
            {"name": "aggregate_version", "type": "int", "default": 1},
        ]
    }

    # Integration event schema
    INTEGRATION_EVENT_SCHEMA = {
        "type": "record",
        "name": "IntegrationEvent",
        "namespace": "com.greenlang.events",
        "fields": [
            *BASE_SCHEMA["fields"],
            {"name": "target_systems", "type": {"type": "array", "items": "string"}},
            {"name": "requires_ack", "type": "boolean", "default": False},
            {"name": "ttl_seconds", "type": ["null", "int"], "default": None},
        ]
    }

    # Event type specific schemas
    _schemas: Dict[str, Dict] = {}

    @classmethod
    def register_schema(cls, event_type: str, schema: Dict) -> None:
        """
        Register a custom schema for an event type.

        Args:
            event_type: Event type identifier
            schema: Avro schema definition
        """
        cls._schemas[event_type] = schema
        logger.info(f"Registered schema for event type: {event_type}")

    @classmethod
    def get_schema(cls, event_type: str) -> Dict:
        """
        Get schema for an event type.

        Args:
            event_type: Event type identifier

        Returns:
            Avro schema definition
        """
        return cls._schemas.get(event_type, cls.BASE_SCHEMA)

    @classmethod
    def create_emission_event_schema(cls) -> Dict:
        """Create schema for emission events."""
        return {
            "type": "record",
            "name": "EmissionEvent",
            "namespace": "com.greenlang.events.emission",
            "fields": [
                *cls.BASE_SCHEMA["fields"],
                {"name": "scope", "type": "string"},
                {"name": "category", "type": "string"},
                {"name": "value", "type": "double"},
                {"name": "unit", "type": "string"},
                {"name": "emission_factor_id", "type": ["null", "string"]},
                {"name": "calculation_method", "type": "string"},
                {"name": "reporting_period", "type": "string"},
            ]
        }

    @classmethod
    def create_compliance_event_schema(cls) -> Dict:
        """Create schema for compliance events."""
        return {
            "type": "record",
            "name": "ComplianceEvent",
            "namespace": "com.greenlang.events.compliance",
            "fields": [
                *cls.BASE_SCHEMA["fields"],
                {"name": "framework", "type": "string"},
                {"name": "requirement_id", "type": "string"},
                {"name": "status", "type": "string"},
                {"name": "evidence_references", "type": {"type": "array", "items": "string"}},
                {"name": "findings", "type": ["null", {"type": "array", "items": "string"}]},
            ]
        }

    @classmethod
    def validate_event(cls, event: BaseEvent) -> bool:
        """
        Validate event against its schema.

        Args:
            event: Event to validate

        Returns:
            True if valid
        """
        # Basic validation - schema validation would use fastavro
        required_fields = ["event_type", "metadata", "data"]
        for field in required_fields:
            if not getattr(event, field, None):
                logger.error(f"Event missing required field: {field}")
                return False

        if not event.provenance_hash:
            logger.error("Event missing provenance hash")
            return False

        return True

    @classmethod
    def get_all_schemas(cls) -> Dict[str, Dict]:
        """Get all registered schemas."""
        return {
            "base": cls.BASE_SCHEMA,
            "domain": cls.DOMAIN_EVENT_SCHEMA,
            "integration": cls.INTEGRATION_EVENT_SCHEMA,
            **cls._schemas,
        }


# Pre-register standard schemas
EventSchema.register_schema(
    EventType.EMISSION_CALCULATED.value,
    EventSchema.create_emission_event_schema()
)
EventSchema.register_schema(
    EventType.COMPLIANCE_CHECK_COMPLETED.value,
    EventSchema.create_compliance_event_schema()
)
