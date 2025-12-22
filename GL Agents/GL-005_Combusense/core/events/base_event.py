# -*- coding: utf-8 -*-
"""
GL-005 CombustionControlAgent - Base Event Classes

This module defines the foundational event classes for event sourcing.
All domain events inherit from DomainEvent and carry immutable event data
with full provenance tracking.

Design Principles:
    - Events are immutable (frozen=True)
    - Events carry their own metadata (timestamp, version, causation)
    - Events are serializable to JSON for persistence
    - Events support schema versioning for forward compatibility

Example:
    >>> event = DomainEvent(
    ...     aggregate_id="combustion-001",
    ...     event_type="ControlSetpointChanged",
    ...     payload={"fuel_flow": 1000.0, "air_flow": 12500.0}
    ... )
    >>> print(event.event_id)
    >>> print(event.provenance_hash)
"""

from __future__ import annotations

import hashlib
import json
import logging
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Dict, Generic, Optional, TypeVar
from uuid import uuid4

from pydantic import BaseModel, Field, field_validator, ConfigDict

logger = logging.getLogger(__name__)

# Type variable for event payload
TPayload = TypeVar("TPayload", bound=BaseModel)


class EventMetadata(BaseModel):
    """
    Metadata attached to every domain event.

    Provides full traceability and provenance for audit trails.

    Attributes:
        event_id: Unique identifier for this event
        timestamp: When the event occurred (UTC)
        version: Schema version for forward compatibility
        correlation_id: Links related events together
        causation_id: ID of the event that caused this event
        actor_id: Who/what triggered this event
        source: System component that produced the event
    """

    model_config = ConfigDict(frozen=True)

    event_id: str = Field(
        default_factory=lambda: str(uuid4()),
        description="Unique event identifier (UUID)"
    )
    timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="Event timestamp in UTC"
    )
    version: int = Field(
        default=1,
        ge=1,
        description="Event schema version"
    )
    correlation_id: Optional[str] = Field(
        default=None,
        description="Correlation ID for tracing related events"
    )
    causation_id: Optional[str] = Field(
        default=None,
        description="ID of the event that caused this event"
    )
    actor_id: Optional[str] = Field(
        default="GL-005",
        description="Agent or user that triggered this event"
    )
    source: str = Field(
        default="CombustionControlAgent",
        description="Source system/component"
    )


class DomainEvent(BaseModel, ABC):
    """
    Base class for all domain events in the combustion control system.

    Domain events represent facts that have occurred in the system.
    They are immutable, timestamped, and carry full provenance information
    for audit trails and deterministic replay.

    Inheritance Pattern:
        class ControlSetpointChanged(DomainEvent):
            fuel_flow_setpoint: float
            air_flow_setpoint: float

    Attributes:
        aggregate_id: ID of the aggregate this event belongs to
        aggregate_type: Type name of the aggregate
        event_type: Name of this event type
        sequence_number: Order within the aggregate's event stream
        metadata: Event metadata (timestamp, version, etc.)
        payload: Event-specific data (in subclasses)
        provenance_hash: SHA-256 hash for verification

    Example:
        >>> event = ControlSetpointChanged(
        ...     aggregate_id="burner-001",
        ...     fuel_flow_setpoint=1000.0,
        ...     air_flow_setpoint=12500.0
        ... )
        >>> assert event.event_type == "ControlSetpointChanged"
    """

    model_config = ConfigDict(frozen=True)

    # Aggregate identification
    aggregate_id: str = Field(
        ...,
        min_length=1,
        description="Unique identifier of the aggregate this event belongs to"
    )
    aggregate_type: str = Field(
        default="CombustionControlAggregate",
        description="Type name of the aggregate"
    )

    # Event identification
    event_type: str = Field(
        default="",
        description="Type name of this event"
    )
    sequence_number: int = Field(
        default=0,
        ge=0,
        description="Sequence number within aggregate's event stream"
    )

    # Metadata
    metadata: EventMetadata = Field(
        default_factory=EventMetadata,
        description="Event metadata"
    )

    # Provenance
    provenance_hash: str = Field(
        default="",
        description="SHA-256 hash for audit trail"
    )

    def model_post_init(self, __context: Any) -> None:
        """Post-initialization to set computed fields."""
        # Set event_type from class name if not provided
        if not self.event_type:
            object.__setattr__(self, "event_type", self.__class__.__name__)

        # Calculate provenance hash if not set
        if not self.provenance_hash:
            hash_value = self._calculate_provenance_hash()
            object.__setattr__(self, "provenance_hash", hash_value)

    def _calculate_provenance_hash(self) -> str:
        """
        Calculate SHA-256 hash of event data for audit trail.

        The hash covers:
            - aggregate_id
            - event_type
            - sequence_number
            - metadata.event_id
            - metadata.timestamp
            - All payload fields

        Returns:
            Hexadecimal SHA-256 hash string
        """
        # Build hashable data excluding the hash field itself
        hash_data = {
            "aggregate_id": self.aggregate_id,
            "aggregate_type": self.aggregate_type,
            "event_type": self.event_type,
            "sequence_number": self.sequence_number,
            "event_id": self.metadata.event_id,
            "timestamp": self.metadata.timestamp.isoformat(),
        }

        # Add all other fields (payload) excluding computed fields
        for field_name, field_value in self.model_dump().items():
            if field_name not in ["provenance_hash", "metadata"]:
                hash_data[field_name] = field_value

        # Create deterministic JSON string
        hash_input = json.dumps(hash_data, sort_keys=True, default=str)
        return hashlib.sha256(hash_input.encode()).hexdigest()

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert event to dictionary for serialization.

        Returns:
            Dictionary representation of the event
        """
        return self.model_dump()

    def to_json(self) -> str:
        """
        Convert event to JSON string.

        Returns:
            JSON string representation
        """
        return self.model_dump_json()

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DomainEvent":
        """
        Create event from dictionary.

        Args:
            data: Dictionary with event data

        Returns:
            DomainEvent instance
        """
        return cls(**data)

    @classmethod
    def from_json(cls, json_str: str) -> "DomainEvent":
        """
        Create event from JSON string.

        Args:
            json_str: JSON string with event data

        Returns:
            DomainEvent instance
        """
        data = json.loads(json_str)
        return cls.from_dict(data)

    @property
    def event_id(self) -> str:
        """Get event ID from metadata."""
        return self.metadata.event_id

    @property
    def timestamp(self) -> datetime:
        """Get timestamp from metadata."""
        return self.metadata.timestamp

    def with_sequence_number(self, seq: int) -> "DomainEvent":
        """
        Create a copy with a new sequence number.

        Args:
            seq: New sequence number

        Returns:
            New event instance with updated sequence number
        """
        data = self.model_dump()
        data["sequence_number"] = seq
        data["provenance_hash"] = ""  # Will be recalculated
        return self.__class__(**data)

    def with_causation(self, causation_id: str) -> "DomainEvent":
        """
        Create a copy with causation ID set.

        Args:
            causation_id: ID of the causing event

        Returns:
            New event instance with causation ID
        """
        data = self.model_dump()
        metadata_data = data.get("metadata", {})
        metadata_data["causation_id"] = causation_id
        data["metadata"] = EventMetadata(**metadata_data)
        data["provenance_hash"] = ""  # Will be recalculated
        return self.__class__(**data)

    def with_correlation(self, correlation_id: str) -> "DomainEvent":
        """
        Create a copy with correlation ID set.

        Args:
            correlation_id: Correlation ID for tracing

        Returns:
            New event instance with correlation ID
        """
        data = self.model_dump()
        metadata_data = data.get("metadata", {})
        metadata_data["correlation_id"] = correlation_id
        data["metadata"] = EventMetadata(**metadata_data)
        data["provenance_hash"] = ""  # Will be recalculated
        return self.__class__(**data)

    def verify_hash(self) -> bool:
        """
        Verify the provenance hash is correct.

        Returns:
            True if hash matches, False if tampered
        """
        expected_hash = self._calculate_provenance_hash()
        return self.provenance_hash == expected_hash

    def __str__(self) -> str:
        """Human-readable string representation."""
        return (
            f"{self.event_type}("
            f"aggregate_id={self.aggregate_id}, "
            f"seq={self.sequence_number}, "
            f"timestamp={self.metadata.timestamp.isoformat()})"
        )

    def __repr__(self) -> str:
        """Detailed string representation."""
        return (
            f"<{self.__class__.__name__} "
            f"event_id={self.metadata.event_id} "
            f"aggregate_id={self.aggregate_id} "
            f"seq={self.sequence_number}>"
        )


class EventEnvelope(BaseModel):
    """
    Wrapper for events during storage and transmission.

    Provides additional metadata for event routing and processing.

    Attributes:
        event: The wrapped domain event
        stream_name: Name of the event stream
        position: Global position in the event log
        stored_at: When the event was persisted
    """

    model_config = ConfigDict(frozen=True)

    event: Dict[str, Any] = Field(
        ...,
        description="Serialized domain event"
    )
    event_type: str = Field(
        ...,
        description="Type name for deserialization"
    )
    stream_name: str = Field(
        ...,
        description="Event stream name (aggregate_type-aggregate_id)"
    )
    position: int = Field(
        default=0,
        ge=0,
        description="Global position in event log"
    )
    stored_at: datetime = Field(
        default_factory=datetime.utcnow,
        description="When event was stored"
    )

    @classmethod
    def wrap(
        cls,
        event: DomainEvent,
        stream_name: Optional[str] = None,
        position: int = 0
    ) -> "EventEnvelope":
        """
        Wrap a domain event in an envelope.

        Args:
            event: Domain event to wrap
            stream_name: Optional stream name override
            position: Global position in log

        Returns:
            EventEnvelope containing the event
        """
        if stream_name is None:
            stream_name = f"{event.aggregate_type}-{event.aggregate_id}"

        return cls(
            event=event.to_dict(),
            event_type=event.event_type,
            stream_name=stream_name,
            position=position,
            stored_at=datetime.utcnow()
        )
