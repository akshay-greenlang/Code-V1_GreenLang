"""
Tests for Violation Handler

Tests violation handling including:
- Actuation blocking
- Alarm/event emission
- Immutable audit record creation
- Escalation procedures
"""

import pytest
from datetime import datetime, timedelta
import json

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from safety.violation_handler import (
    ViolationHandler,
    ViolationEvent,
    EscalationLevel,
    EscalationRule,
    NotificationType,
    NotificationTarget,
    ViolationHandlerFactory,
)
from safety.safety_schemas import (
    BoundaryViolation,
    ViolationType,
    ViolationSeverity,
    PolicyType,
)


class TestViolationHandlerInitialization:
    """Test violation handler initialization."""

    @pytest.fixture
    def handler(self):
        """Create fresh violation handler."""
        return ViolationHandler()

    def test_initialization(self, handler):
        """Test handler initializes correctly."""
        assert handler is not None
        stats = handler.get_statistics()
        assert stats["violations_handled"] == 0

    def test_default_escalation_rules(self, handler):
        """Test default escalation rules are loaded."""
        # Default rules should exist
        # Check by handling different severity violations


class TestViolationHandling:
    """Test basic violation handling."""

    @pytest.fixture
    def handler(self):
        """Create fresh violation handler."""
        return ViolationHandler()

    def test_handle_violation_returns_event(self, handler):
        """Test handling violation returns event."""
        violation = BoundaryViolation(
            policy_id="TEST_001",
            tag_id="TIC-101",
            requested_value=250.0,
            boundary_value=200.0,
            violation_type=ViolationType.OVER_MAX,
            severity=ViolationSeverity.CRITICAL,
            message="Temperature exceeds maximum",
        )
        event = handler.handle_violation(violation)
        assert event is not None
        assert isinstance(event, ViolationEvent)
        assert event.violation == violation

    def test_violation_event_has_timestamp(self, handler):
        """Test violation event has timestamp."""
        violation = BoundaryViolation(
            policy_id="TEST_001",
            tag_id="TIC-101",
            violation_type=ViolationType.OVER_MAX,
        )
        event = handler.handle_violation(violation)
        assert event.timestamp is not None

    def test_violation_event_has_action(self, handler):
        """Test violation event has action taken."""
        violation = BoundaryViolation(
            policy_id="TEST_001",
            tag_id="TIC-101",
            violation_type=ViolationType.OVER_MAX,
        )
        event = handler.handle_violation(violation)
        assert event.action_taken == "ACTUATION_BLOCKED"


class TestEscalation:
    """Test escalation level determination."""

    @pytest.fixture
    def handler(self):
        """Create fresh violation handler."""
        return ViolationHandler()

    def test_warning_escalation(self, handler):
        """Test warning severity gets low escalation."""
        violation = BoundaryViolation(
            policy_id="TEST_001",
            tag_id="TIC-101",
            violation_type=ViolationType.OVER_MAX,
            severity=ViolationSeverity.WARNING,
        )
        event = handler.handle_violation(violation)
        assert event.escalation_level <= EscalationLevel.LEVEL_1

    def test_critical_escalation(self, handler):
        """Test critical severity gets higher escalation."""
        violation = BoundaryViolation(
            policy_id="TEST_001",
            tag_id="TIC-101",
            violation_type=ViolationType.OVER_MAX,
            severity=ViolationSeverity.CRITICAL,
        )
        event = handler.handle_violation(violation)
        assert event.escalation_level >= EscalationLevel.LEVEL_2

    def test_emergency_escalation(self, handler):
        """Test emergency severity gets highest escalation."""
        violation = BoundaryViolation(
            policy_id="TEST_001",
            tag_id="TIC-101",
            violation_type=ViolationType.SIS_VIOLATION,
            severity=ViolationSeverity.EMERGENCY,
        )
        event = handler.handle_violation(violation)
        assert event.escalation_level >= EscalationLevel.LEVEL_3

    def test_sis_violation_highest_escalation(self, handler):
        """Test SIS violation gets highest escalation."""
        violation = BoundaryViolation(
            policy_id="SIS_INDEPENDENCE",
            tag_id="SIS-101",
            violation_type=ViolationType.SIS_VIOLATION,
            severity=ViolationSeverity.EMERGENCY,
        )
        event = handler.handle_violation(violation)
        assert event.escalation_level == EscalationLevel.LEVEL_4


class TestNotificationTargets:
    """Test notification target management."""

    @pytest.fixture
    def handler(self):
        """Create fresh violation handler."""
        return ViolationHandler()

    def test_add_notification_target(self, handler):
        """Test adding notification target."""
        target = NotificationTarget(
            target_id="test_log",
            notification_type=NotificationType.LOG,
            endpoint="test.log",
        )
        handler.add_notification_target(target)

    def test_remove_notification_target(self, handler):
        """Test removing notification target."""
        target = NotificationTarget(
            target_id="test_log",
            notification_type=NotificationType.LOG,
            endpoint="test.log",
        )
        handler.add_notification_target(target)
        result = handler.remove_notification_target("test_log")
        assert result == True

    def test_remove_nonexistent_target(self, handler):
        """Test removing nonexistent target."""
        result = handler.remove_notification_target("nonexistent")
        assert result == False

    def test_notification_sent_on_violation(self, handler):
        """Test notifications are sent on violation."""
        target = NotificationTarget(
            target_id="test_log",
            notification_type=NotificationType.LOG,
            endpoint="test.log",
            min_severity=ViolationSeverity.WARNING,
            escalation_level=EscalationLevel.LEVEL_0,
        )
        handler.add_notification_target(target)

        violation = BoundaryViolation(
            policy_id="TEST_001",
            tag_id="TIC-101",
            violation_type=ViolationType.OVER_MAX,
            severity=ViolationSeverity.CRITICAL,
        )
        event = handler.handle_violation(violation)
        assert "test_log" in event.notifications_sent


class TestEscalationRules:
    """Test custom escalation rules."""

    def test_add_escalation_rule(self):
        """Test adding custom escalation rule."""
        handler = ViolationHandler()
        rule = EscalationRule(
            rule_id="CUSTOM_001",
            violation_type=ViolationType.RATE_EXCEEDED,
            escalation_level=EscalationLevel.LEVEL_3,
        )
        handler.add_escalation_rule(rule)

    def test_custom_rule_applies(self):
        """Test custom rule is applied."""
        rule = EscalationRule(
            rule_id="CUSTOM_001",
            policy_id_pattern="CUSTOM_*",
            escalation_level=EscalationLevel.LEVEL_3,
        )
        handler = ViolationHandler(escalation_rules=[rule])

        violation = BoundaryViolation(
            policy_id="CUSTOM_POLICY",
            tag_id="TIC-101",
            violation_type=ViolationType.OVER_MAX,
            severity=ViolationSeverity.WARNING,  # Low severity but custom rule
        )
        event = handler.handle_violation(violation)
        # Custom rule should elevate escalation
        assert event.escalation_level >= EscalationLevel.LEVEL_3


class TestAuditTrail:
    """Test audit trail functionality."""

    @pytest.fixture
    def handler(self):
        """Create fresh violation handler."""
        return ViolationHandler()

    def test_violation_creates_audit_record(self, handler):
        """Test violation creates audit record."""
        violation = BoundaryViolation(
            policy_id="TEST_001",
            tag_id="TIC-101",
            violation_type=ViolationType.OVER_MAX,
        )
        handler.handle_violation(violation)

        records = handler.get_audit_records()
        assert len(records) > 0

    def test_audit_chain_integrity(self, handler):
        """Test audit chain maintains integrity."""
        for i in range(3):
            violation = BoundaryViolation(
                policy_id=f"TEST_{i:03d}",
                tag_id="TIC-101",
                violation_type=ViolationType.OVER_MAX,
            )
            handler.handle_violation(violation)

        assert handler.verify_audit_chain() == True

    def test_get_audit_records_since(self, handler):
        """Test getting audit records since timestamp."""
        violation = BoundaryViolation(
            policy_id="TEST_001",
            tag_id="TIC-101",
            violation_type=ViolationType.OVER_MAX,
        )
        handler.handle_violation(violation)

        # Get records from future
        future = datetime.utcnow() + timedelta(hours=1)
        records = handler.get_audit_records(since=future)
        assert len(records) == 0


class TestViolationEvents:
    """Test violation event retrieval."""

    @pytest.fixture
    def handler(self):
        """Create fresh violation handler."""
        return ViolationHandler()

    def test_get_violation_events(self, handler):
        """Test getting violation events."""
        violation = BoundaryViolation(
            policy_id="TEST_001",
            tag_id="TIC-101",
            violation_type=ViolationType.OVER_MAX,
        )
        handler.handle_violation(violation)

        events = handler.get_violation_events()
        assert len(events) > 0

    def test_get_events_since(self, handler):
        """Test getting events since timestamp."""
        violation = BoundaryViolation(
            policy_id="TEST_001",
            tag_id="TIC-101",
            violation_type=ViolationType.OVER_MAX,
        )
        handler.handle_violation(violation)

        future = datetime.utcnow() + timedelta(hours=1)
        events = handler.get_violation_events(since=future)
        assert len(events) == 0


class TestStatistics:
    """Test statistics collection."""

    @pytest.fixture
    def handler(self):
        """Create fresh violation handler."""
        return ViolationHandler()

    def test_violations_handled_count(self, handler):
        """Test violations handled count."""
        for i in range(5):
            violation = BoundaryViolation(
                policy_id=f"TEST_{i:03d}",
                tag_id="TIC-101",
                violation_type=ViolationType.OVER_MAX,
            )
            handler.handle_violation(violation)

        stats = handler.get_statistics()
        assert stats["violations_handled"] == 5

    def test_escalation_counts(self, handler):
        """Test escalation level counts."""
        # Warning violation
        handler.handle_violation(BoundaryViolation(
            policy_id="TEST_001",
            tag_id="TIC-101",
            violation_type=ViolationType.OVER_MAX,
            severity=ViolationSeverity.WARNING,
        ))

        # Critical violation
        handler.handle_violation(BoundaryViolation(
            policy_id="TEST_002",
            tag_id="TIC-101",
            violation_type=ViolationType.OVER_MAX,
            severity=ViolationSeverity.CRITICAL,
        ))

        stats = handler.get_statistics()
        # Should have some escalations at different levels
        assert stats["violations_handled"] == 2


class TestViolationRate:
    """Test violation rate tracking."""

    @pytest.fixture
    def handler(self):
        """Create fresh violation handler."""
        return ViolationHandler()

    def test_violation_rate_calculation(self, handler):
        """Test violation rate calculation."""
        for i in range(5):
            violation = BoundaryViolation(
                policy_id="TEST_001",
                tag_id="TIC-101",
                violation_type=ViolationType.OVER_MAX,
            )
            handler.handle_violation(violation)

        rate = handler.get_violation_rate(window_seconds=60)
        assert rate > 0


class TestAuditExport:
    """Test audit trail export."""

    @pytest.fixture
    def handler(self):
        """Create fresh violation handler."""
        return ViolationHandler()

    def test_export_json(self, handler):
        """Test exporting audit trail as JSON."""
        violation = BoundaryViolation(
            policy_id="TEST_001",
            tag_id="TIC-101",
            violation_type=ViolationType.OVER_MAX,
        )
        handler.handle_violation(violation)

        export = handler.export_audit_trail(format="json")
        data = json.loads(export)
        assert isinstance(data, list)
        assert len(data) > 0

    def test_export_invalid_format_raises(self, handler):
        """Test invalid export format raises error."""
        with pytest.raises(ValueError):
            handler.export_audit_trail(format="xml")


class TestAuditCallback:
    """Test audit callback functionality."""

    def test_audit_callback_invoked(self):
        """Test audit callback is invoked."""
        records_received = []

        def callback(record):
            records_received.append(record)

        handler = ViolationHandler(audit_callback=callback)

        violation = BoundaryViolation(
            policy_id="TEST_001",
            tag_id="TIC-101",
            violation_type=ViolationType.OVER_MAX,
        )
        handler.handle_violation(violation)

        assert len(records_received) > 0


class TestFactory:
    """Test factory methods."""

    def test_create_handler(self):
        """Test creating handler via factory."""
        handler = ViolationHandlerFactory.create_handler()
        assert handler is not None

    def test_create_production_handler(self):
        """Test creating production handler."""
        handler = ViolationHandlerFactory.create_production_handler()
        assert handler is not None


class TestViolationEventModel:
    """Test ViolationEvent model."""

    def test_event_has_id(self):
        """Test event has unique ID."""
        violation = BoundaryViolation(
            policy_id="TEST_001",
            tag_id="TIC-101",
            violation_type=ViolationType.OVER_MAX,
        )
        event = ViolationEvent(
            violation=violation,
            escalation_level=EscalationLevel.LEVEL_1,
            action_taken="BLOCKED",
        )
        assert event.event_id is not None

    def test_event_provenance_hash(self):
        """Test event has provenance hash (computed externally)."""
        violation = BoundaryViolation(
            policy_id="TEST_001",
            tag_id="TIC-101",
            violation_type=ViolationType.OVER_MAX,
        )
        handler = ViolationHandler()
        event = handler.handle_violation(violation)
        assert event.provenance_hash != ""


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
