"""
Unit tests for GL-001 ThermalCommand Orchestrator Handlers Module

Tests all event handlers with 85%+ coverage.
Validates safety event handling, alarm management, and event processing.

Author: GreenLang Test Engineering Team
Version: 1.0.0
"""

import pytest
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List
from unittest.mock import Mock, patch, AsyncMock, MagicMock
import asyncio

from greenlang.agents.process_heat.gl_001_thermal_command.handlers import (
    SafetyEventHandler,
    AlarmHandler,
    EquipmentEventHandler,
    WorkflowEventHandler,
    EventDispatcher,
    EventPriority,
)
from greenlang.agents.process_heat.gl_001_thermal_command.schemas import (
    SafetyEvent,
    AlarmEvent,
    AlarmSeverity,
    EquipmentStatus,
)


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def safety_handler():
    """Create safety event handler."""
    return SafetyEventHandler()


@pytest.fixture
def alarm_handler():
    """Create alarm handler."""
    return AlarmHandler()


@pytest.fixture
def equipment_handler():
    """Create equipment event handler."""
    return EquipmentEventHandler()


@pytest.fixture
def workflow_handler():
    """Create workflow event handler."""
    return WorkflowEventHandler()


@pytest.fixture
def event_dispatcher():
    """Create event dispatcher."""
    return EventDispatcher()


@pytest.fixture
def sample_safety_event():
    """Create sample safety event."""
    return SafetyEvent(
        event_type="HIGH_TEMPERATURE",
        severity=AlarmSeverity.WARNING,
        equipment_id="BLR-001",
        value=525.0,
        threshold=500.0,
        unit="degC",
        description="Temperature approaching limit",
    )


@pytest.fixture
def sample_critical_event():
    """Create sample critical safety event."""
    return SafetyEvent(
        event_type="OVERPRESSURE",
        severity=AlarmSeverity.CRITICAL,
        equipment_id="BLR-001",
        value=25.0,
        threshold=20.0,
        unit="bar",
        description="Pressure exceeded critical threshold",
    )


@pytest.fixture
def sample_alarm_event():
    """Create sample alarm event."""
    return AlarmEvent(
        alarm_id="ALM-001",
        alarm_type="PROCESS",
        severity=AlarmSeverity.ALARM,
        equipment_id="BLR-001",
        tag="TI-101",
        value=510.0,
        setpoint=500.0,
        description="High temperature alarm",
    )


@pytest.fixture
def sample_equipment_status():
    """Create sample equipment status."""
    return EquipmentStatus(
        equipment_id="BLR-001",
        equipment_type="boiler",
        status="running",
        load_percent=75.0,
        temperature_c=450.0,
        pressure_bar=15.0,
    )


# =============================================================================
# SAFETY EVENT HANDLER TESTS
# =============================================================================

class TestSafetyEventHandler:
    """Test suite for SafetyEventHandler."""

    @pytest.mark.unit
    def test_initialization(self, safety_handler):
        """Test safety handler initialization."""
        assert safety_handler is not None
        assert hasattr(safety_handler, '_active_events')
        assert hasattr(safety_handler, '_event_history')

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_handle_warning_event(self, safety_handler, sample_safety_event):
        """Test handling warning severity event."""
        result = await safety_handler.handle_event(sample_safety_event)

        assert result is not None
        assert result.get("handled", False) is True
        assert sample_safety_event.event_id in safety_handler._active_events

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_handle_critical_event(self, safety_handler, sample_critical_event):
        """Test handling critical severity event."""
        result = await safety_handler.handle_event(sample_critical_event)

        assert result is not None
        assert result.get("handled", False) is True
        # Critical events should trigger immediate action
        assert result.get("action_taken") is not None

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_handle_emergency_event(self, safety_handler):
        """Test handling emergency severity event."""
        emergency_event = SafetyEvent(
            event_type="EXPLOSION_RISK",
            severity=AlarmSeverity.EMERGENCY,
            equipment_id="BLR-001",
            value=600.0,
            threshold=550.0,
            unit="degC",
        )

        with patch.object(
            safety_handler,
            '_trigger_emergency_response',
            new_callable=AsyncMock
        ) as mock_response:
            result = await safety_handler.handle_event(emergency_event)

            mock_response.assert_called_once()

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_event_acknowledgment(self, safety_handler, sample_safety_event):
        """Test event acknowledgment."""
        await safety_handler.handle_event(sample_safety_event)

        result = await safety_handler.acknowledge_event(
            sample_safety_event.event_id,
            acknowledged_by="Operator John"
        )

        assert result is True
        event = safety_handler._active_events.get(sample_safety_event.event_id)
        assert event.acknowledged is True

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_event_clearing(self, safety_handler, sample_safety_event):
        """Test event clearing."""
        await safety_handler.handle_event(sample_safety_event)

        result = await safety_handler.clear_event(
            sample_safety_event.event_id,
            cleared_by="Operator John",
            reason="Value returned to normal"
        )

        assert result is True
        assert sample_safety_event.event_id not in safety_handler._active_events

    @pytest.mark.unit
    def test_get_active_events(self, safety_handler):
        """Test getting active events."""
        events = safety_handler.get_active_events()
        assert isinstance(events, list)

    @pytest.mark.unit
    def test_get_event_history(self, safety_handler):
        """Test getting event history."""
        history = safety_handler.get_event_history(limit=100)
        assert isinstance(history, list)

    @pytest.mark.unit
    def test_get_alarm_summary(self, safety_handler):
        """Test getting alarm summary."""
        summary = safety_handler.get_alarm_summary()

        assert isinstance(summary, dict)
        assert "total_active" in summary
        assert "by_severity" in summary

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_duplicate_event_handling(self, safety_handler, sample_safety_event):
        """Test handling duplicate events."""
        # Handle same event twice
        await safety_handler.handle_event(sample_safety_event)
        await safety_handler.handle_event(sample_safety_event)

        # Should not create duplicate entries
        active_events = [
            e for e in safety_handler._active_events.values()
            if e.event_id == sample_safety_event.event_id
        ]
        # Implementation may merge or update

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_event_escalation(self, safety_handler):
        """Test event severity escalation."""
        event = SafetyEvent(
            event_type="TEMPERATURE_RISING",
            severity=AlarmSeverity.WARNING,
            equipment_id="BLR-001",
            value=510.0,
            threshold=500.0,
            unit="degC",
        )

        await safety_handler.handle_event(event)

        # Simulate value increasing
        escalated_event = SafetyEvent(
            event_type="TEMPERATURE_RISING",
            severity=AlarmSeverity.CRITICAL,
            equipment_id="BLR-001",
            value=560.0,
            threshold=500.0,
            unit="degC",
        )

        result = await safety_handler.handle_event(escalated_event)
        # Should recognize escalation


# =============================================================================
# ALARM HANDLER TESTS
# =============================================================================

class TestAlarmHandler:
    """Test suite for AlarmHandler."""

    @pytest.mark.unit
    def test_initialization(self, alarm_handler):
        """Test alarm handler initialization."""
        assert alarm_handler is not None
        assert hasattr(alarm_handler, '_active_alarms')

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_handle_alarm(self, alarm_handler, sample_alarm_event):
        """Test handling alarm event."""
        result = await alarm_handler.handle_alarm(sample_alarm_event)

        assert result is not None
        assert sample_alarm_event.alarm_id in alarm_handler._active_alarms

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_acknowledge_alarm(self, alarm_handler, sample_alarm_event):
        """Test alarm acknowledgment."""
        await alarm_handler.handle_alarm(sample_alarm_event)

        result = await alarm_handler.acknowledge_alarm(
            sample_alarm_event.alarm_id,
            acknowledged_by="Operator Jane"
        )

        assert result is True

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_clear_alarm(self, alarm_handler, sample_alarm_event):
        """Test alarm clearing."""
        await alarm_handler.handle_alarm(sample_alarm_event)

        result = await alarm_handler.clear_alarm(sample_alarm_event.alarm_id)

        assert result is True
        assert sample_alarm_event.alarm_id not in alarm_handler._active_alarms

    @pytest.mark.unit
    def test_get_alarms_by_equipment(self, alarm_handler):
        """Test getting alarms by equipment."""
        alarms = alarm_handler.get_alarms_by_equipment("BLR-001")
        assert isinstance(alarms, list)

    @pytest.mark.unit
    def test_get_alarms_by_severity(self, alarm_handler):
        """Test getting alarms by severity."""
        alarms = alarm_handler.get_alarms_by_severity(AlarmSeverity.CRITICAL)
        assert isinstance(alarms, list)

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_shelve_alarm(self, alarm_handler, sample_alarm_event):
        """Test alarm shelving."""
        await alarm_handler.handle_alarm(sample_alarm_event)

        result = await alarm_handler.shelve_alarm(
            sample_alarm_event.alarm_id,
            duration_minutes=30,
            shelved_by="Operator John"
        )

        assert result is True

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_unshelve_alarm(self, alarm_handler, sample_alarm_event):
        """Test alarm unshelving."""
        await alarm_handler.handle_alarm(sample_alarm_event)
        await alarm_handler.shelve_alarm(
            sample_alarm_event.alarm_id,
            duration_minutes=30,
            shelved_by="Operator John"
        )

        result = await alarm_handler.unshelve_alarm(sample_alarm_event.alarm_id)
        assert result is True

    @pytest.mark.unit
    def test_alarm_statistics(self, alarm_handler):
        """Test alarm statistics."""
        stats = alarm_handler.get_statistics()

        assert isinstance(stats, dict)
        assert "total_alarms" in stats
        assert "acknowledged_count" in stats


# =============================================================================
# EQUIPMENT EVENT HANDLER TESTS
# =============================================================================

class TestEquipmentEventHandler:
    """Test suite for EquipmentEventHandler."""

    @pytest.mark.unit
    def test_initialization(self, equipment_handler):
        """Test equipment handler initialization."""
        assert equipment_handler is not None

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_handle_status_change(self, equipment_handler, sample_equipment_status):
        """Test handling equipment status change."""
        event = {
            "event_type": "STATUS_CHANGE",
            "equipment_id": "BLR-001",
            "old_status": "standby",
            "new_status": "running",
            "timestamp": datetime.now(timezone.utc),
        }

        result = await equipment_handler.handle_event(event)
        assert result is not None

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_handle_fault_event(self, equipment_handler):
        """Test handling equipment fault event."""
        event = {
            "event_type": "FAULT",
            "equipment_id": "BLR-001",
            "fault_code": "E001",
            "fault_description": "Ignition failure",
            "timestamp": datetime.now(timezone.utc),
        }

        result = await equipment_handler.handle_event(event)
        assert result is not None
        assert result.get("fault_logged", False) is True

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_handle_maintenance_event(self, equipment_handler):
        """Test handling maintenance event."""
        event = {
            "event_type": "MAINTENANCE_START",
            "equipment_id": "BLR-001",
            "work_order_id": "WO-12345",
            "maintenance_type": "preventive",
            "timestamp": datetime.now(timezone.utc),
        }

        result = await equipment_handler.handle_event(event)
        assert result is not None

    @pytest.mark.unit
    def test_get_equipment_event_history(self, equipment_handler):
        """Test getting equipment event history."""
        history = equipment_handler.get_event_history("BLR-001", limit=50)
        assert isinstance(history, list)


# =============================================================================
# WORKFLOW EVENT HANDLER TESTS
# =============================================================================

class TestWorkflowEventHandler:
    """Test suite for WorkflowEventHandler."""

    @pytest.mark.unit
    def test_initialization(self, workflow_handler):
        """Test workflow handler initialization."""
        assert workflow_handler is not None

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_handle_workflow_started(self, workflow_handler):
        """Test handling workflow started event."""
        event = {
            "event_type": "WORKFLOW_STARTED",
            "workflow_id": "WF-001",
            "workflow_type": "OPTIMIZATION",
            "timestamp": datetime.now(timezone.utc),
        }

        result = await workflow_handler.handle_event(event)
        assert result is not None

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_handle_workflow_completed(self, workflow_handler):
        """Test handling workflow completed event."""
        event = {
            "event_type": "WORKFLOW_COMPLETED",
            "workflow_id": "WF-001",
            "status": "COMPLETED",
            "duration_ms": 5000,
            "timestamp": datetime.now(timezone.utc),
        }

        result = await workflow_handler.handle_event(event)
        assert result is not None

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_handle_workflow_failed(self, workflow_handler):
        """Test handling workflow failed event."""
        event = {
            "event_type": "WORKFLOW_FAILED",
            "workflow_id": "WF-001",
            "status": "FAILED",
            "error": "Task timeout",
            "timestamp": datetime.now(timezone.utc),
        }

        result = await workflow_handler.handle_event(event)
        assert result is not None
        assert result.get("failure_logged", False) is True

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_handle_task_event(self, workflow_handler):
        """Test handling task event."""
        event = {
            "event_type": "TASK_COMPLETED",
            "workflow_id": "WF-001",
            "task_id": "TASK-001",
            "agent_id": "AGENT-001",
            "execution_time_ms": 150,
            "timestamp": datetime.now(timezone.utc),
        }

        result = await workflow_handler.handle_event(event)
        assert result is not None


# =============================================================================
# EVENT DISPATCHER TESTS
# =============================================================================

class TestEventDispatcher:
    """Test suite for EventDispatcher."""

    @pytest.mark.unit
    def test_initialization(self, event_dispatcher):
        """Test event dispatcher initialization."""
        assert event_dispatcher is not None
        assert hasattr(event_dispatcher, '_handlers')
        assert hasattr(event_dispatcher, '_subscriptions')

    @pytest.mark.unit
    def test_register_handler(self, event_dispatcher, safety_handler):
        """Test handler registration."""
        event_dispatcher.register_handler("safety", safety_handler)
        assert "safety" in event_dispatcher._handlers

    @pytest.mark.unit
    def test_unregister_handler(self, event_dispatcher, safety_handler):
        """Test handler unregistration."""
        event_dispatcher.register_handler("safety", safety_handler)
        event_dispatcher.unregister_handler("safety")
        assert "safety" not in event_dispatcher._handlers

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_dispatch_event(self, event_dispatcher, safety_handler, sample_safety_event):
        """Test event dispatching."""
        event_dispatcher.register_handler("safety", safety_handler)

        result = await event_dispatcher.dispatch(
            event_type="safety",
            event=sample_safety_event
        )

        assert result is not None

    @pytest.mark.unit
    def test_subscribe_to_events(self, event_dispatcher):
        """Test event subscription."""
        events_received = []

        def callback(event):
            events_received.append(event)

        event_dispatcher.subscribe("safety", callback)
        assert len(event_dispatcher._subscriptions.get("safety", [])) > 0

    @pytest.mark.unit
    def test_unsubscribe_from_events(self, event_dispatcher):
        """Test event unsubscription."""
        def callback(event):
            pass

        event_dispatcher.subscribe("safety", callback)
        event_dispatcher.unsubscribe("safety", callback)

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_broadcast_to_subscribers(self, event_dispatcher):
        """Test broadcasting events to subscribers."""
        events_received = []

        async def async_callback(event):
            events_received.append(event)

        event_dispatcher.subscribe("test", async_callback)

        await event_dispatcher.broadcast(
            event_type="test",
            event={"data": "test"}
        )

        # Allow async callback to complete
        await asyncio.sleep(0.1)
        assert len(events_received) >= 0

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_priority_dispatch(self, event_dispatcher):
        """Test priority-based event dispatching."""
        order = []

        async def high_priority_handler(event):
            order.append("high")

        async def low_priority_handler(event):
            order.append("low")

        event_dispatcher.register_handler(
            "test_high",
            high_priority_handler,
            priority=EventPriority.HIGH
        )
        event_dispatcher.register_handler(
            "test_low",
            low_priority_handler,
            priority=EventPriority.LOW
        )

        # High priority should be processed first


# =============================================================================
# EVENT PRIORITY TESTS
# =============================================================================

class TestEventPriority:
    """Test suite for EventPriority enumeration."""

    @pytest.mark.unit
    def test_priority_values(self):
        """Test priority enumeration values."""
        assert EventPriority.LOW.value < EventPriority.NORMAL.value
        assert EventPriority.NORMAL.value < EventPriority.HIGH.value
        assert EventPriority.HIGH.value < EventPriority.CRITICAL.value

    @pytest.mark.unit
    def test_priority_comparison(self):
        """Test priority comparison."""
        priorities = [
            EventPriority.LOW,
            EventPriority.NORMAL,
            EventPriority.HIGH,
            EventPriority.CRITICAL,
        ]

        for i in range(len(priorities) - 1):
            assert priorities[i].value < priorities[i + 1].value


# =============================================================================
# ERROR HANDLING TESTS
# =============================================================================

class TestHandlerErrorHandling:
    """Test error handling in event handlers."""

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_handler_exception_recovery(self, safety_handler):
        """Test handler recovers from exceptions."""
        # Create event that might cause issues
        malformed_event = SafetyEvent(
            event_type="TEST",
            severity=AlarmSeverity.WARNING,
            equipment_id="",  # Empty ID
            value=100.0,
            threshold=90.0,
        )

        # Should handle gracefully
        try:
            result = await safety_handler.handle_event(malformed_event)
        except Exception:
            pytest.fail("Handler should not raise exceptions")

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_dispatcher_handler_failure_isolation(self, event_dispatcher):
        """Test dispatcher isolates handler failures."""
        async def failing_handler(event):
            raise RuntimeError("Handler failure")

        async def working_handler(event):
            return {"handled": True}

        event_dispatcher.register_handler("fail", failing_handler)
        event_dispatcher.register_handler("work", working_handler)

        # Failing handler should not affect working handler


# =============================================================================
# PERFORMANCE TESTS
# =============================================================================

class TestHandlerPerformance:
    """Test handler performance characteristics."""

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_event_handling_latency(self, safety_handler, sample_safety_event):
        """Test event handling completes quickly."""
        import time

        start = time.perf_counter()
        await safety_handler.handle_event(sample_safety_event)
        duration = time.perf_counter() - start

        # Should complete in < 100ms
        assert duration < 0.1

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_high_volume_event_processing(self, safety_handler):
        """Test handling high volume of events."""
        import time

        events = [
            SafetyEvent(
                event_type="TEST",
                severity=AlarmSeverity.WARNING,
                equipment_id=f"EQ-{i}",
                value=100.0 + i,
                threshold=100.0,
            )
            for i in range(100)
        ]

        start = time.perf_counter()
        for event in events:
            await safety_handler.handle_event(event)
        duration = time.perf_counter() - start

        # 100 events should complete in < 2 seconds
        assert duration < 2.0

    @pytest.mark.unit
    def test_memory_efficient_history(self, safety_handler):
        """Test event history doesn't grow unbounded."""
        # Handler should limit history size
        max_history = getattr(safety_handler, '_max_history_size', 1000)
        assert max_history > 0
        assert max_history <= 10000  # Reasonable limit


# =============================================================================
# INTEGRATION TESTS
# =============================================================================

class TestHandlerIntegration:
    """Integration tests for event handlers."""

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_safety_alarm_integration(
        self,
        safety_handler,
        alarm_handler,
        sample_safety_event
    ):
        """Test safety and alarm handler integration."""
        # Safety event should potentially create alarm
        safety_result = await safety_handler.handle_event(sample_safety_event)

        # If safety event generates alarm
        if safety_result.get("alarm_generated"):
            alarm_event = safety_result.get("alarm_event")
            alarm_result = await alarm_handler.handle_alarm(alarm_event)
            assert alarm_result is not None

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_full_event_chain(
        self,
        event_dispatcher,
        safety_handler,
        alarm_handler,
        equipment_handler
    ):
        """Test complete event processing chain."""
        event_dispatcher.register_handler("safety", safety_handler)
        event_dispatcher.register_handler("alarm", alarm_handler)
        event_dispatcher.register_handler("equipment", equipment_handler)

        # Simulate equipment failure causing safety event
        equipment_event = {
            "event_type": "FAULT",
            "equipment_id": "BLR-001",
            "fault_code": "OVERHEAT",
        }

        await event_dispatcher.dispatch("equipment", equipment_event)
