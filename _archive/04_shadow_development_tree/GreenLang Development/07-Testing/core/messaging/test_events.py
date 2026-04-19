# -*- coding: utf-8 -*-
"""
Test suite for Event and StandardEvents.

Tests cover:
- Event creation and validation
- StandardEvents catalog
- Event serialization
- Event helpers

Author: GreenLang Framework Team
Date: December 2025
"""

import pytest
from datetime import datetime

from greenlang.core.messaging import (
    Event,
    EventPriority,
    StandardEvents,
    create_event,
)


class TestEvent:
    """Test Event class."""

    def test_event_creation(self):
        """Test creating an event with all fields."""
        event = Event(
            event_id="evt-123",
            event_type=StandardEvents.AGENT_STARTED,
            source_agent="GL-001",
            payload={"status": "ready"},
            priority=EventPriority.HIGH,
            correlation_id="corr-456",
            target_agent="orchestrator",
            metadata={"region": "us-west"},
        )

        assert event.event_id == "evt-123"
        assert event.event_type == StandardEvents.AGENT_STARTED
        assert event.source_agent == "GL-001"
        assert event.payload["status"] == "ready"
        assert event.priority == EventPriority.HIGH
        assert event.correlation_id == "corr-456"
        assert event.target_agent == "orchestrator"
        assert event.metadata["region"] == "us-west"

    def test_event_auto_id_generation(self):
        """Test that event_id is auto-generated if not provided."""
        event = Event(
            event_id="",  # Will be auto-generated
            event_type=StandardEvents.CALCULATION_COMPLETED,
            source_agent="GL-002",
            payload={"result": 42},
        )

        assert event.event_id != ""
        assert len(event.event_id) > 0

    def test_event_timestamp_generation(self):
        """Test that timestamp is auto-generated."""
        event = Event(
            event_id="evt-456",
            event_type=StandardEvents.DATA_RECEIVED,
            source_agent="GL-003",
            payload={},
        )

        assert event.timestamp is not None
        # Verify it's a valid ISO 8601 timestamp
        datetime.fromisoformat(event.timestamp.replace("Z", "+00:00"))

    def test_event_validation_missing_source(self):
        """Test that event creation fails without source_agent."""
        with pytest.raises(ValueError, match="source_agent is required"):
            Event(
                event_id="evt-789",
                event_type=StandardEvents.AGENT_ERROR,
                source_agent="",  # Empty source
                payload={},
            )

    def test_event_validation_missing_type(self):
        """Test that event creation fails without event_type."""
        with pytest.raises(ValueError, match="event_type is required"):
            Event(
                event_id="evt-789",
                event_type="",  # Empty type
                source_agent="GL-001",
                payload={},
            )

    def test_event_to_dict(self):
        """Test converting event to dictionary."""
        event = Event(
            event_id="evt-123",
            event_type=StandardEvents.COMPLIANCE_CHECK_PASSED,
            source_agent="GL-004",
            payload={"checks": 10, "passed": 10},
            priority=EventPriority.MEDIUM,
        )

        event_dict = event.to_dict()

        assert event_dict["event_id"] == "evt-123"
        assert event_dict["event_type"] == StandardEvents.COMPLIANCE_CHECK_PASSED
        assert event_dict["source_agent"] == "GL-004"
        assert event_dict["payload"]["checks"] == 10
        assert event_dict["priority"] == "medium"

    def test_event_from_dict(self):
        """Test creating event from dictionary."""
        event_data = {
            "event_id": "evt-456",
            "event_type": StandardEvents.SAFETY_ALERT,
            "source_agent": "GL-005",
            "payload": {"severity": "high"},
            "priority": "high",
            "correlation_id": "corr-789",
        }

        event = Event.from_dict(event_data)

        assert event.event_id == "evt-456"
        assert event.event_type == StandardEvents.SAFETY_ALERT
        assert event.source_agent == "GL-005"
        assert event.payload["severity"] == "high"
        assert event.priority == EventPriority.HIGH

    def test_event_is_high_priority(self):
        """Test checking if event is high priority."""
        high_event = Event(
            event_id="evt-1",
            event_type=StandardEvents.SAFETY_INTERLOCK_TRIGGERED,
            source_agent="GL-001",
            payload={},
            priority=EventPriority.HIGH,
        )

        critical_event = Event(
            event_id="evt-2",
            event_type=StandardEvents.SAFETY_EMERGENCY_SHUTDOWN,
            source_agent="GL-001",
            payload={},
            priority=EventPriority.CRITICAL,
        )

        medium_event = Event(
            event_id="evt-3",
            event_type=StandardEvents.AGENT_HEARTBEAT,
            source_agent="GL-001",
            payload={},
            priority=EventPriority.MEDIUM,
        )

        assert high_event.is_high_priority()
        assert critical_event.is_high_priority()
        assert not medium_event.is_high_priority()

    def test_event_is_safety_related(self):
        """Test checking if event is safety-related."""
        safety_event = Event(
            event_id="evt-1",
            event_type=StandardEvents.SAFETY_ALERT,
            source_agent="GL-001",
            payload={},
        )

        non_safety_event = Event(
            event_id="evt-2",
            event_type=StandardEvents.AGENT_STARTED,
            source_agent="GL-001",
            payload={},
        )

        assert safety_event.is_safety_related()
        assert not non_safety_event.is_safety_related()

    def test_event_is_compliance_related(self):
        """Test checking if event is compliance-related."""
        compliance_event = Event(
            event_id="evt-1",
            event_type=StandardEvents.COMPLIANCE_VIOLATION_DETECTED,
            source_agent="GL-001",
            payload={},
        )

        non_compliance_event = Event(
            event_id="evt-2",
            event_type=StandardEvents.DATA_RECEIVED,
            source_agent="GL-001",
            payload={},
        )

        assert compliance_event.is_compliance_related()
        assert not non_compliance_event.is_compliance_related()


class TestEventPriority:
    """Test EventPriority enum."""

    def test_priority_comparison(self):
        """Test that priorities can be compared."""
        assert EventPriority.CRITICAL < EventPriority.HIGH
        assert EventPriority.HIGH < EventPriority.MEDIUM
        assert EventPriority.MEDIUM < EventPriority.LOW

    def test_priority_values(self):
        """Test priority string values."""
        assert EventPriority.CRITICAL.value == "critical"
        assert EventPriority.HIGH.value == "high"
        assert EventPriority.MEDIUM.value == "medium"
        assert EventPriority.LOW.value == "low"


class TestStandardEvents:
    """Test StandardEvents catalog."""

    def test_lifecycle_events_exist(self):
        """Test that lifecycle events are defined."""
        assert hasattr(StandardEvents, "AGENT_STARTED")
        assert hasattr(StandardEvents, "AGENT_STOPPED")
        assert hasattr(StandardEvents, "AGENT_ERROR")
        assert hasattr(StandardEvents, "AGENT_HEARTBEAT")

    def test_calculation_events_exist(self):
        """Test that calculation events are defined."""
        assert hasattr(StandardEvents, "CALCULATION_STARTED")
        assert hasattr(StandardEvents, "CALCULATION_COMPLETED")
        assert hasattr(StandardEvents, "CALCULATION_FAILED")

    def test_orchestration_events_exist(self):
        """Test that orchestration events are defined."""
        assert hasattr(StandardEvents, "TASK_ASSIGNED")
        assert hasattr(StandardEvents, "TASK_COMPLETED")
        assert hasattr(StandardEvents, "WORKFLOW_STARTED")
        assert hasattr(StandardEvents, "COORDINATION_REQUESTED")

    def test_integration_events_exist(self):
        """Test that integration events are defined."""
        assert hasattr(StandardEvents, "INTEGRATION_CALL_STARTED")
        assert hasattr(StandardEvents, "INTEGRATION_CALL_COMPLETED")
        assert hasattr(StandardEvents, "INTEGRATION_DATA_RECEIVED")

    def test_compliance_events_exist(self):
        """Test that compliance events are defined."""
        assert hasattr(StandardEvents, "COMPLIANCE_CHECK_STARTED")
        assert hasattr(StandardEvents, "COMPLIANCE_CHECK_PASSED")
        assert hasattr(StandardEvents, "COMPLIANCE_VIOLATION_DETECTED")

    def test_safety_events_exist(self):
        """Test that safety events are defined."""
        assert hasattr(StandardEvents, "SAFETY_ALERT")
        assert hasattr(StandardEvents, "SAFETY_INTERLOCK_TRIGGERED")
        assert hasattr(StandardEvents, "SAFETY_EMERGENCY_SHUTDOWN")

    def test_data_events_exist(self):
        """Test that data events are defined."""
        assert hasattr(StandardEvents, "DATA_RECEIVED")
        assert hasattr(StandardEvents, "DATA_VALIDATED")
        assert hasattr(StandardEvents, "DATA_VALIDATION_FAILED")

    def test_all_events_method(self):
        """Test retrieving all event types."""
        all_events = StandardEvents.all_events()

        assert isinstance(all_events, list)
        assert len(all_events) > 0
        assert StandardEvents.AGENT_STARTED in all_events
        assert StandardEvents.CALCULATION_COMPLETED in all_events
        assert StandardEvents.SAFETY_ALERT in all_events

    def test_is_lifecycle_event(self):
        """Test identifying lifecycle events."""
        assert StandardEvents.is_lifecycle_event(StandardEvents.AGENT_STARTED)
        assert StandardEvents.is_lifecycle_event(StandardEvents.AGENT_ERROR)
        assert not StandardEvents.is_lifecycle_event(StandardEvents.CALCULATION_COMPLETED)

    def test_is_calculation_event(self):
        """Test identifying calculation events."""
        assert StandardEvents.is_calculation_event(StandardEvents.CALCULATION_STARTED)
        assert StandardEvents.is_calculation_event(StandardEvents.CALCULATION_COMPLETED)
        assert not StandardEvents.is_calculation_event(StandardEvents.AGENT_STARTED)

    def test_is_orchestration_event(self):
        """Test identifying orchestration events."""
        assert StandardEvents.is_orchestration_event(StandardEvents.TASK_ASSIGNED)
        assert StandardEvents.is_orchestration_event(StandardEvents.WORKFLOW_STARTED)
        assert not StandardEvents.is_orchestration_event(StandardEvents.AGENT_STARTED)

    def test_is_integration_event(self):
        """Test identifying integration events."""
        assert StandardEvents.is_integration_event(
            StandardEvents.INTEGRATION_CALL_STARTED
        )
        assert not StandardEvents.is_integration_event(StandardEvents.AGENT_STARTED)

    def test_is_compliance_event(self):
        """Test identifying compliance events."""
        assert StandardEvents.is_compliance_event(StandardEvents.COMPLIANCE_CHECK_PASSED)
        assert not StandardEvents.is_compliance_event(StandardEvents.AGENT_STARTED)

    def test_is_safety_event(self):
        """Test identifying safety events."""
        assert StandardEvents.is_safety_event(StandardEvents.SAFETY_ALERT)
        assert StandardEvents.is_safety_event(StandardEvents.SAFETY_INTERLOCK_TRIGGERED)
        assert not StandardEvents.is_safety_event(StandardEvents.AGENT_STARTED)

    def test_is_data_event(self):
        """Test identifying data events."""
        assert StandardEvents.is_data_event(StandardEvents.DATA_RECEIVED)
        assert StandardEvents.is_data_event(StandardEvents.DATA_VALIDATED)
        assert not StandardEvents.is_data_event(StandardEvents.AGENT_STARTED)


class TestCreateEvent:
    """Test create_event factory function."""

    def test_create_event_basic(self):
        """Test creating event with factory function."""
        event = create_event(
            event_type=StandardEvents.AGENT_STARTED,
            source_agent="GL-001",
            payload={"status": "ready"},
        )

        assert event.event_type == StandardEvents.AGENT_STARTED
        assert event.source_agent == "GL-001"
        assert event.payload["status"] == "ready"
        assert event.priority == EventPriority.MEDIUM  # Default
        assert event.event_id != ""  # Auto-generated

    def test_create_event_with_priority(self):
        """Test creating event with custom priority."""
        event = create_event(
            event_type=StandardEvents.SAFETY_ALERT,
            source_agent="GL-001",
            payload={"severity": "critical"},
            priority=EventPriority.CRITICAL,
        )

        assert event.priority == EventPriority.CRITICAL

    def test_create_event_with_correlation(self):
        """Test creating event with correlation ID."""
        event = create_event(
            event_type=StandardEvents.CALCULATION_COMPLETED,
            source_agent="GL-002",
            payload={"result": 42},
            correlation_id="req-123",
        )

        assert event.correlation_id == "req-123"

    def test_create_event_with_target(self):
        """Test creating event with target agent."""
        event = create_event(
            event_type=StandardEvents.TASK_ASSIGNED,
            source_agent="orchestrator",
            payload={"task_id": "task-456"},
            target_agent="GL-003",
        )

        assert event.target_agent == "GL-003"
