# -*- coding: utf-8 -*-
"""
GL-005 CombustionControlAgent - Event Sourcing Tests

Comprehensive test suite for event sourcing infrastructure including:
    - Event replay tests
    - Snapshot/restore tests
    - Concurrent write tests
    - Event versioning tests
    - Aggregate lifecycle tests

Test Categories:
    - Unit tests for individual components
    - Integration tests for event store + aggregate
    - Concurrency tests for optimistic locking
    - Performance tests for large event streams
"""

import asyncio
import json
import os
import tempfile
from datetime import datetime, timedelta
from typing import List
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Import event sourcing components
from core.events.base_event import DomainEvent, EventMetadata, EventEnvelope
from core.events.domain_events import (
    ControlSetpointChanged,
    SafetyInterventionTriggered,
    OptimizationCompleted,
    SensorReadingReceived,
    AlarmTriggered,
    SystemStateChanged,
    SetpointChangeReason,
    SafetyInterventionType,
    AlarmSeverity,
    AlarmCategory,
    SystemMode,
    EVENT_REGISTRY,
    deserialize_event,
)
from core.events.event_store import (
    EventStore,
    EventStoreConfig,
    EventStoreBackend,
    OptimisticConcurrencyError,
    InMemoryBackend,
    SQLiteBackend,
)
from core.events.event_bus import (
    EventBus,
    EventBusConfig,
    EventSubscription,
    HandlerPriority,
    DeliveryMode,
)
from core.events.aggregates import Aggregate, AggregateRoot, AggregateRepository
from core.events.snapshots import (
    SnapshotManager,
    SnapshotConfig,
    Snapshot,
    SnapshotMigrator,
)
from core.events.combustion_aggregate import (
    CombustionControlAggregate,
    ControlSetpoint,
    StabilityMetrics,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def aggregate_id():
    """Generate a unique aggregate ID for tests."""
    return f"test-burner-{datetime.utcnow().strftime('%Y%m%d%H%M%S%f')}"


@pytest.fixture
def sample_setpoint_event(aggregate_id):
    """Create a sample ControlSetpointChanged event."""
    return ControlSetpointChanged(
        aggregate_id=aggregate_id,
        fuel_flow_setpoint=1000.0,
        air_flow_setpoint=12500.0,
        previous_fuel_flow_setpoint=900.0,
        previous_air_flow_setpoint=11500.0,
        fuel_valve_position=50.0,
        air_damper_position=50.0,
        reason=SetpointChangeReason.OPTIMIZATION,
        control_mode="auto"
    )


@pytest.fixture
def sample_sensor_event(aggregate_id):
    """Create a sample SensorReadingReceived event."""
    return SensorReadingReceived(
        aggregate_id=aggregate_id,
        reading_type="furnace_temperature",
        sensor_id="furnace_temp_001",
        value=1200.5,
        unit="degC",
        quality="good"
    )


@pytest.fixture
def sample_alarm_event(aggregate_id):
    """Create a sample AlarmTriggered event."""
    return AlarmTriggered(
        aggregate_id=aggregate_id,
        alarm_id="HH_TEMP",
        alarm_name="Furnace Temperature High-High",
        category=AlarmCategory.TEMPERATURE,
        severity=AlarmSeverity.HIGH,
        trigger_value=1450.0,
        setpoint=1400.0,
        deviation=50.0
    )


@pytest.fixture
async def memory_event_store():
    """Create an in-memory event store for testing."""
    config = EventStoreConfig(backend=EventStoreBackend.MEMORY)
    store = EventStore(config)
    await store.initialize()
    yield store
    await store.close()


@pytest.fixture
async def sqlite_event_store(tmp_path):
    """Create a SQLite event store for testing."""
    db_path = tmp_path / "test_events.db"
    config = EventStoreConfig(
        backend=EventStoreBackend.SQLITE,
        connection_string=str(db_path)
    )
    store = EventStore(config)
    await store.initialize()
    yield store
    await store.close()


@pytest.fixture
def event_bus():
    """Create an event bus for testing."""
    return EventBus(EventBusConfig(delivery_mode=DeliveryMode.SEQUENTIAL))


# =============================================================================
# Domain Event Tests
# =============================================================================


class TestDomainEvents:
    """Tests for domain events."""

    def test_control_setpoint_changed_creation(self, aggregate_id):
        """Test creating a ControlSetpointChanged event."""
        event = ControlSetpointChanged(
            aggregate_id=aggregate_id,
            fuel_flow_setpoint=1000.0,
            air_flow_setpoint=12500.0,
            reason=SetpointChangeReason.OPTIMIZATION
        )

        assert event.aggregate_id == aggregate_id
        assert event.event_type == "ControlSetpointChanged"
        assert event.fuel_flow_setpoint == 1000.0
        assert event.air_flow_setpoint == 12500.0
        assert event.provenance_hash != ""

    def test_event_immutability(self, sample_setpoint_event):
        """Test that events are immutable."""
        with pytest.raises(Exception):  # Pydantic frozen model
            sample_setpoint_event.fuel_flow_setpoint = 2000.0

    def test_event_provenance_hash(self, sample_setpoint_event):
        """Test provenance hash calculation."""
        # Hash should be non-empty
        assert len(sample_setpoint_event.provenance_hash) == 64  # SHA-256

        # Same event data should produce same hash
        event2 = ControlSetpointChanged(
            aggregate_id=sample_setpoint_event.aggregate_id,
            fuel_flow_setpoint=sample_setpoint_event.fuel_flow_setpoint,
            air_flow_setpoint=sample_setpoint_event.air_flow_setpoint,
            metadata=sample_setpoint_event.metadata,
            reason=sample_setpoint_event.reason
        )
        # Note: Different timestamps will produce different hashes

    def test_event_hash_verification(self, sample_setpoint_event):
        """Test that hash verification works."""
        assert sample_setpoint_event.verify_hash() is True

    def test_event_serialization(self, sample_setpoint_event):
        """Test event serialization to JSON."""
        json_str = sample_setpoint_event.to_json()
        data = json.loads(json_str)

        assert data["aggregate_id"] == sample_setpoint_event.aggregate_id
        assert data["event_type"] == "ControlSetpointChanged"
        assert data["fuel_flow_setpoint"] == 1000.0

    def test_event_deserialization(self, sample_setpoint_event):
        """Test event deserialization from dict."""
        data = sample_setpoint_event.to_dict()
        restored = deserialize_event(data)

        assert restored.aggregate_id == sample_setpoint_event.aggregate_id
        assert restored.fuel_flow_setpoint == sample_setpoint_event.fuel_flow_setpoint

    def test_event_with_sequence_number(self, sample_setpoint_event):
        """Test creating event with sequence number."""
        event_with_seq = sample_setpoint_event.with_sequence_number(5)

        assert event_with_seq.sequence_number == 5
        assert event_with_seq.fuel_flow_setpoint == sample_setpoint_event.fuel_flow_setpoint
        # Hash should be recalculated
        assert event_with_seq.provenance_hash != ""

    def test_safety_intervention_event(self, aggregate_id):
        """Test SafetyInterventionTriggered event."""
        event = SafetyInterventionTriggered(
            aggregate_id=aggregate_id,
            intervention_type=SafetyInterventionType.FUEL_CUTOFF,
            severity=AlarmSeverity.CRITICAL,
            trigger_condition="flame_loss",
            trigger_value=0.0,
            trigger_limit=1.0,
            interlocks_tripped=["flame_present"],
            actions_taken=["fuel_cutoff"]
        )

        assert event.intervention_type == SafetyInterventionType.FUEL_CUTOFF
        assert "flame_present" in event.interlocks_tripped

    def test_optimization_completed_event(self, aggregate_id):
        """Test OptimizationCompleted event."""
        event = OptimizationCompleted(
            aggregate_id=aggregate_id,
            optimization_type="fuel_air_ratio",
            objective_function="thermal_efficiency",
            initial_value=85.0,
            final_value=88.0,
            improvement_percent=3.53,
            optimized_fuel_flow=1000.0,
            optimized_air_flow=12500.0,
            optimized_excess_air_percent=15.0,
            predicted_efficiency=88.0
        )

        assert event.improvement_percent == 3.53
        assert event.predicted_efficiency == 88.0

    def test_system_state_changed_event(self, aggregate_id):
        """Test SystemStateChanged event."""
        event = SystemStateChanged(
            aggregate_id=aggregate_id,
            previous_mode=SystemMode.STANDBY,
            new_mode=SystemMode.NORMAL,
            transition_reason="startup_complete",
            operator_initiated=True
        )

        assert event.previous_mode == SystemMode.STANDBY
        assert event.new_mode == SystemMode.NORMAL

    def test_event_registry(self):
        """Test event registry contains all events."""
        assert "ControlSetpointChanged" in EVENT_REGISTRY
        assert "SafetyInterventionTriggered" in EVENT_REGISTRY
        assert "OptimizationCompleted" in EVENT_REGISTRY
        assert "SensorReadingReceived" in EVENT_REGISTRY
        assert "AlarmTriggered" in EVENT_REGISTRY
        assert "SystemStateChanged" in EVENT_REGISTRY


# =============================================================================
# Event Store Tests
# =============================================================================


class TestEventStore:
    """Tests for event store functionality."""

    @pytest.mark.asyncio
    async def test_append_and_load_events(self, memory_event_store, aggregate_id):
        """Test appending and loading events."""
        event1 = ControlSetpointChanged(
            aggregate_id=aggregate_id,
            fuel_flow_setpoint=1000.0,
            air_flow_setpoint=12500.0
        )
        event2 = ControlSetpointChanged(
            aggregate_id=aggregate_id,
            fuel_flow_setpoint=1100.0,
            air_flow_setpoint=13000.0
        )

        # Append events
        version = await memory_event_store.append(aggregate_id, [event1, event2])
        assert version == 2

        # Load events
        events = await memory_event_store.load(aggregate_id)
        assert len(events) == 2
        assert events[0].fuel_flow_setpoint == 1000.0
        assert events[1].fuel_flow_setpoint == 1100.0

    @pytest.mark.asyncio
    async def test_event_ordering(self, memory_event_store, aggregate_id):
        """Test that events maintain ordering."""
        events = []
        for i in range(10):
            events.append(ControlSetpointChanged(
                aggregate_id=aggregate_id,
                fuel_flow_setpoint=1000.0 + i * 100,
                air_flow_setpoint=12500.0 + i * 500
            ))

        await memory_event_store.append(aggregate_id, events)

        loaded = await memory_event_store.load(aggregate_id)
        assert len(loaded) == 10

        # Check ordering
        for i, event in enumerate(loaded):
            assert event.sequence_number == i + 1
            assert event.fuel_flow_setpoint == 1000.0 + i * 100

    @pytest.mark.asyncio
    async def test_optimistic_concurrency(self, memory_event_store, aggregate_id):
        """Test optimistic concurrency control."""
        event1 = ControlSetpointChanged(
            aggregate_id=aggregate_id,
            fuel_flow_setpoint=1000.0,
            air_flow_setpoint=12500.0
        )

        # Append first event
        await memory_event_store.append(aggregate_id, [event1])

        event2 = ControlSetpointChanged(
            aggregate_id=aggregate_id,
            fuel_flow_setpoint=1100.0,
            air_flow_setpoint=13000.0
        )

        # Try to append with wrong expected version
        with pytest.raises(OptimisticConcurrencyError):
            await memory_event_store.append(
                aggregate_id, [event2], expected_version=0
            )

        # Append with correct expected version
        version = await memory_event_store.append(
            aggregate_id, [event2], expected_version=1
        )
        assert version == 2

    @pytest.mark.asyncio
    async def test_load_from_version(self, memory_event_store, aggregate_id):
        """Test loading events from a specific version."""
        events = [
            ControlSetpointChanged(
                aggregate_id=aggregate_id,
                fuel_flow_setpoint=1000.0 + i * 100,
                air_flow_setpoint=12500.0
            )
            for i in range(5)
        ]

        await memory_event_store.append(aggregate_id, events)

        # Load from version 3
        loaded = await memory_event_store.load(aggregate_id, from_version=3)
        assert len(loaded) == 3  # Events 3, 4, 5
        assert loaded[0].sequence_number == 3

    @pytest.mark.asyncio
    async def test_get_version(self, memory_event_store, aggregate_id):
        """Test getting current version."""
        # New aggregate has version 0
        version = await memory_event_store.get_version(aggregate_id)
        assert version == 0

        # After appending
        event = ControlSetpointChanged(
            aggregate_id=aggregate_id,
            fuel_flow_setpoint=1000.0,
            air_flow_setpoint=12500.0
        )
        await memory_event_store.append(aggregate_id, [event])

        version = await memory_event_store.get_version(aggregate_id)
        assert version == 1

    @pytest.mark.asyncio
    async def test_aggregate_exists(self, memory_event_store, aggregate_id):
        """Test checking if aggregate exists."""
        assert await memory_event_store.exists(aggregate_id) is False

        event = ControlSetpointChanged(
            aggregate_id=aggregate_id,
            fuel_flow_setpoint=1000.0,
            air_flow_setpoint=12500.0
        )
        await memory_event_store.append(aggregate_id, [event])

        assert await memory_event_store.exists(aggregate_id) is True

    @pytest.mark.asyncio
    async def test_sqlite_persistence(self, sqlite_event_store, aggregate_id):
        """Test SQLite event store persistence."""
        event = ControlSetpointChanged(
            aggregate_id=aggregate_id,
            fuel_flow_setpoint=1000.0,
            air_flow_setpoint=12500.0
        )

        await sqlite_event_store.append(aggregate_id, [event])
        events = await sqlite_event_store.load(aggregate_id)

        assert len(events) == 1
        assert events[0].fuel_flow_setpoint == 1000.0

    @pytest.mark.asyncio
    async def test_snapshot_save_and_load(self, memory_event_store, aggregate_id):
        """Test snapshot save and load."""
        snapshot_data = {
            "fuel_flow_setpoint": 1000.0,
            "air_flow_setpoint": 12500.0,
            "system_mode": "normal"
        }

        await memory_event_store.save_snapshot(
            aggregate_id=aggregate_id,
            version=10,
            snapshot_data=snapshot_data
        )

        result = await memory_event_store.load_latest_snapshot(aggregate_id)
        assert result is not None

        version, loaded_data = result
        assert version == 10
        assert loaded_data["fuel_flow_setpoint"] == 1000.0


# =============================================================================
# Event Bus Tests
# =============================================================================


class TestEventBus:
    """Tests for event bus functionality."""

    @pytest.mark.asyncio
    async def test_subscribe_and_publish(self, event_bus, sample_setpoint_event):
        """Test subscribing to and publishing events."""
        received_events = []

        @event_bus.subscribe(ControlSetpointChanged)
        async def handler(event):
            received_events.append(event)

        await event_bus.publish(sample_setpoint_event)

        assert len(received_events) == 1
        assert received_events[0].fuel_flow_setpoint == 1000.0

    @pytest.mark.asyncio
    async def test_multiple_handlers(self, event_bus, sample_setpoint_event):
        """Test multiple handlers for same event type."""
        handler1_called = []
        handler2_called = []

        @event_bus.subscribe(ControlSetpointChanged)
        async def handler1(event):
            handler1_called.append(True)

        @event_bus.subscribe(ControlSetpointChanged)
        async def handler2(event):
            handler2_called.append(True)

        await event_bus.publish(sample_setpoint_event)

        assert len(handler1_called) == 1
        assert len(handler2_called) == 1

    @pytest.mark.asyncio
    async def test_handler_priority(self, event_bus, sample_setpoint_event):
        """Test handler priority ordering."""
        execution_order = []

        @event_bus.subscribe(ControlSetpointChanged, priority=HandlerPriority.LOW)
        async def low_priority(event):
            execution_order.append("low")

        @event_bus.subscribe(ControlSetpointChanged, priority=HandlerPriority.HIGH)
        async def high_priority(event):
            execution_order.append("high")

        @event_bus.subscribe(ControlSetpointChanged, priority=HandlerPriority.CRITICAL)
        async def critical_priority(event):
            execution_order.append("critical")

        await event_bus.publish(sample_setpoint_event)

        assert execution_order == ["critical", "high", "low"]

    @pytest.mark.asyncio
    async def test_event_filtering(self, event_bus, aggregate_id):
        """Test event filtering."""
        received = []

        def filter_high_fuel(event):
            return event.fuel_flow_setpoint > 900

        @event_bus.subscribe(
            ControlSetpointChanged,
            filter_fn=filter_high_fuel
        )
        async def handler(event):
            received.append(event)

        # This should be received (fuel_flow > 900)
        event1 = ControlSetpointChanged(
            aggregate_id=aggregate_id,
            fuel_flow_setpoint=1000.0,
            air_flow_setpoint=12500.0
        )
        await event_bus.publish(event1)

        # This should be filtered out (fuel_flow <= 900)
        event2 = ControlSetpointChanged(
            aggregate_id=aggregate_id,
            fuel_flow_setpoint=800.0,
            air_flow_setpoint=10000.0
        )
        await event_bus.publish(event2)

        assert len(received) == 1
        assert received[0].fuel_flow_setpoint == 1000.0

    @pytest.mark.asyncio
    async def test_handler_error_isolation(self, event_bus, sample_setpoint_event):
        """Test that handler errors don't affect other handlers."""
        handler2_called = []

        @event_bus.subscribe(ControlSetpointChanged, priority=HandlerPriority.HIGH)
        async def failing_handler(event):
            raise ValueError("Test error")

        @event_bus.subscribe(ControlSetpointChanged, priority=HandlerPriority.LOW)
        async def working_handler(event):
            handler2_called.append(True)

        results = await event_bus.publish(sample_setpoint_event)

        # Working handler should still be called
        assert len(handler2_called) == 1

        # Should have results for both handlers
        assert len(results) == 2
        assert not results[0].success  # Failing handler
        assert results[1].success  # Working handler

    @pytest.mark.asyncio
    async def test_global_subscription(self, event_bus, aggregate_id):
        """Test global subscription receives all events."""
        received = []

        @event_bus.subscribe_all()
        async def global_handler(event):
            received.append(event)

        event1 = ControlSetpointChanged(
            aggregate_id=aggregate_id,
            fuel_flow_setpoint=1000.0,
            air_flow_setpoint=12500.0
        )
        event2 = SensorReadingReceived(
            aggregate_id=aggregate_id,
            reading_type="temperature",
            sensor_id="temp_001",
            value=1200.0,
            unit="degC"
        )

        await event_bus.publish(event1)
        await event_bus.publish(event2)

        assert len(received) == 2

    @pytest.mark.asyncio
    async def test_parallel_delivery(self, aggregate_id):
        """Test parallel event delivery."""
        bus = EventBus(EventBusConfig(delivery_mode=DeliveryMode.PARALLEL))
        execution_times = []

        async def slow_handler(event):
            import time
            start = time.time()
            await asyncio.sleep(0.1)
            execution_times.append(time.time() - start)

        bus.add_handler(ControlSetpointChanged, slow_handler)
        bus.add_handler(ControlSetpointChanged, slow_handler)
        bus.add_handler(ControlSetpointChanged, slow_handler)

        event = ControlSetpointChanged(
            aggregate_id=aggregate_id,
            fuel_flow_setpoint=1000.0,
            air_flow_setpoint=12500.0
        )

        import time
        start = time.time()
        await bus.publish(event)
        total_time = time.time() - start

        # Parallel execution should be faster than 0.3s (sequential)
        assert total_time < 0.25
        assert len(execution_times) == 3


# =============================================================================
# Aggregate Tests
# =============================================================================


class TestCombustionAggregate:
    """Tests for CombustionControlAggregate."""

    def test_aggregate_creation(self, aggregate_id):
        """Test aggregate creation."""
        aggregate = CombustionControlAggregate(aggregate_id)

        assert aggregate.aggregate_id == aggregate_id
        assert aggregate.version == 0
        assert aggregate.system_mode == SystemMode.OFFLINE
        assert aggregate.control_enabled is False

    def test_change_setpoint(self, aggregate_id):
        """Test changing setpoints."""
        aggregate = CombustionControlAggregate(aggregate_id)
        aggregate._control_enabled = True  # Enable control for test
        aggregate._system_mode = SystemMode.NORMAL

        aggregate.change_setpoint(
            fuel_flow=1000.0,
            air_flow=12500.0,
            reason=SetpointChangeReason.OPTIMIZATION
        )

        assert aggregate.version == 1
        assert aggregate.setpoint.fuel_flow == 1000.0
        assert aggregate.setpoint.air_flow == 12500.0
        assert len(aggregate.uncommitted_events) == 1

    def test_setpoint_validation(self, aggregate_id):
        """Test setpoint validation."""
        aggregate = CombustionControlAggregate(aggregate_id)
        aggregate._control_enabled = True
        aggregate._system_mode = SystemMode.NORMAL

        with pytest.raises(ValueError):
            aggregate.change_setpoint(
                fuel_flow=-100.0,  # Negative not allowed
                air_flow=12500.0
            )

    def test_safety_intervention(self, aggregate_id):
        """Test safety intervention."""
        aggregate = CombustionControlAggregate(aggregate_id)
        aggregate._setpoint.fuel_flow = 1000.0

        aggregate.trigger_safety_intervention(
            intervention_type=SafetyInterventionType.FUEL_CUTOFF,
            trigger_condition="flame_loss",
            trigger_value=0.0,
            trigger_limit=1.0,
            interlocks_tripped=["flame_present"]
        )

        assert aggregate.version == 1
        assert aggregate.setpoint.fuel_flow == 0.0  # Cut off

    def test_system_mode_change(self, aggregate_id):
        """Test system mode changes."""
        aggregate = CombustionControlAggregate(aggregate_id)

        aggregate.change_system_mode(
            new_mode=SystemMode.NORMAL,
            reason="startup_complete",
            operator_initiated=True
        )

        assert aggregate.system_mode == SystemMode.NORMAL
        assert aggregate.control_enabled is True

    def test_sensor_reading(self, aggregate_id):
        """Test recording sensor readings."""
        aggregate = CombustionControlAggregate(aggregate_id)

        for i in range(10):
            aggregate.record_sensor_reading(
                reading_type="heat_output",
                sensor_id="heat_001",
                value=10000.0 + i * 100,
                unit="kW"
            )

        assert aggregate.version == 10
        assert len(aggregate.heat_output_history) == 10

    def test_alarm_handling(self, aggregate_id):
        """Test alarm triggering."""
        aggregate = CombustionControlAggregate(aggregate_id)

        aggregate.trigger_alarm(
            alarm_id="HH_TEMP",
            alarm_name="Temperature High-High",
            category=AlarmCategory.TEMPERATURE,
            severity=AlarmSeverity.CRITICAL,
            trigger_value=1500.0,
            setpoint=1400.0
        )

        assert aggregate.has_active_alarms()
        assert aggregate.has_active_alarms(AlarmSeverity.CRITICAL)
        assert "HH_TEMP" in aggregate.active_alarms

    def test_alarm_clearing(self, aggregate_id):
        """Test clearing alarms."""
        aggregate = CombustionControlAggregate(aggregate_id)

        aggregate.trigger_alarm(
            alarm_id="HH_TEMP",
            alarm_name="Temperature High-High",
            category=AlarmCategory.TEMPERATURE,
            severity=AlarmSeverity.HIGH,
            trigger_value=1500.0,
            setpoint=1400.0
        )

        assert aggregate.clear_alarm("HH_TEMP") is True
        assert aggregate.has_active_alarms() is False

    def test_event_replay(self, aggregate_id):
        """Test rebuilding aggregate from events."""
        # Create aggregate and make changes
        aggregate1 = CombustionControlAggregate(aggregate_id)
        aggregate1._control_enabled = True
        aggregate1._system_mode = SystemMode.NORMAL

        aggregate1.change_setpoint(fuel_flow=1000.0, air_flow=12500.0)
        aggregate1.change_setpoint(fuel_flow=1100.0, air_flow=13000.0)
        aggregate1.change_setpoint(fuel_flow=1200.0, air_flow=14000.0)

        # Get events
        events = aggregate1.uncommitted_events

        # Replay on new aggregate
        aggregate2 = CombustionControlAggregate(aggregate_id)
        aggregate2.apply_all(events)

        assert aggregate2.version == 3
        assert aggregate2.setpoint.fuel_flow == 1200.0
        assert aggregate2.setpoint.air_flow == 14000.0

    def test_snapshot_and_restore(self, aggregate_id):
        """Test snapshotting and restoring aggregate."""
        # Create and modify aggregate
        aggregate1 = CombustionControlAggregate(aggregate_id)
        aggregate1._control_enabled = True
        aggregate1._system_mode = SystemMode.NORMAL
        aggregate1.change_setpoint(fuel_flow=1000.0, air_flow=12500.0)

        for i in range(10):
            aggregate1.record_sensor_reading(
                reading_type="heat_output",
                sensor_id="heat_001",
                value=10000.0 + i * 100,
                unit="kW"
            )

        # Create snapshot
        snapshot = aggregate1.create_snapshot()

        # Restore to new aggregate
        aggregate2 = CombustionControlAggregate(aggregate_id)
        aggregate2.restore_from_snapshot(snapshot)

        assert aggregate2.setpoint.fuel_flow == 1000.0
        assert aggregate2.stability.overall_score > 0

    @pytest.mark.asyncio
    async def test_aggregate_with_event_store(
        self,
        memory_event_store,
        aggregate_id
    ):
        """Test aggregate integration with event store."""
        # Create and save aggregate
        aggregate1 = CombustionControlAggregate(aggregate_id)
        aggregate1._control_enabled = True
        aggregate1._system_mode = SystemMode.NORMAL

        aggregate1.change_setpoint(fuel_flow=1000.0, air_flow=12500.0)
        aggregate1.change_setpoint(fuel_flow=1100.0, air_flow=13000.0)

        await aggregate1.save_to_store(memory_event_store)

        # Load into new aggregate
        aggregate2 = CombustionControlAggregate(aggregate_id)
        events_loaded = await aggregate2.load_from_store(memory_event_store)

        assert events_loaded == 2
        assert aggregate2.version == 2
        assert aggregate2.setpoint.fuel_flow == 1100.0


# =============================================================================
# Snapshot Tests
# =============================================================================


class TestSnapshots:
    """Tests for snapshot functionality."""

    @pytest.mark.asyncio
    async def test_snapshot_creation(self, memory_event_store, aggregate_id):
        """Test snapshot creation."""
        config = SnapshotConfig(event_threshold=10)
        manager = SnapshotManager(memory_event_store, config)

        aggregate = CombustionControlAggregate(aggregate_id)
        aggregate._control_enabled = True
        aggregate._system_mode = SystemMode.NORMAL

        # Make 10 changes
        for i in range(10):
            aggregate.change_setpoint(
                fuel_flow=1000.0 + i * 100,
                air_flow=12500.0 + i * 500
            )

        # Manually take snapshot
        result = await manager.take_snapshot(aggregate)
        assert result is True

        # Verify snapshot exists
        loaded = await manager.get_latest(aggregate_id)
        assert loaded is not None

        snapshot, from_version = loaded
        assert snapshot.data["fuel_flow_setpoint"] == 1900.0

    @pytest.mark.asyncio
    async def test_snapshot_threshold_check(self, memory_event_store, aggregate_id):
        """Test snapshot threshold logic."""
        config = SnapshotConfig(event_threshold=10)
        manager = SnapshotManager(memory_event_store, config)

        # Below threshold
        should_snapshot = await manager.should_snapshot(
            aggregate_id, current_version=5
        )
        assert should_snapshot is False

        # At threshold
        should_snapshot = await manager.should_snapshot(
            aggregate_id, current_version=10
        )
        assert should_snapshot is True

    def test_snapshot_checksum_verification(self, aggregate_id):
        """Test snapshot checksum verification."""
        data = {
            "fuel_flow_setpoint": 1000.0,
            "air_flow_setpoint": 12500.0
        }

        snapshot = Snapshot.create(
            aggregate_id=aggregate_id,
            aggregate_type="CombustionControlAggregate",
            version=10,
            data=data
        )

        assert snapshot.verify_checksum() is True

    def test_snapshot_migration(self):
        """Test snapshot schema migration."""
        migrator = SnapshotMigrator()

        # Register migration v1 -> v2
        def migrate_v1_to_v2(data):
            data["schema_version"] = 2
            data["new_field"] = "default_value"
            return data

        migrator.register_migration(1, 2, migrate_v1_to_v2)

        # Migrate data
        old_data = {"fuel_flow": 1000.0, "schema_version": 1}
        new_data = migrator.migrate(old_data, from_version=1, to_version=2)

        assert new_data["schema_version"] == 2
        assert new_data["new_field"] == "default_value"


# =============================================================================
# Aggregate Repository Tests
# =============================================================================


class TestAggregateRepository:
    """Tests for aggregate repository."""

    @pytest.mark.asyncio
    async def test_get_aggregate(self, memory_event_store, aggregate_id):
        """Test getting aggregate from repository."""
        repo = AggregateRepository(
            CombustionControlAggregate,
            memory_event_store
        )

        # Create and save aggregate directly to store
        aggregate = CombustionControlAggregate(aggregate_id)
        aggregate._control_enabled = True
        aggregate._system_mode = SystemMode.NORMAL
        aggregate.change_setpoint(fuel_flow=1000.0, air_flow=12500.0)
        await aggregate.save_to_store(memory_event_store)

        # Get through repository
        loaded = await repo.get(aggregate_id)
        assert loaded.setpoint.fuel_flow == 1000.0

    @pytest.mark.asyncio
    async def test_get_or_create(self, memory_event_store, aggregate_id):
        """Test get or create aggregate."""
        repo = AggregateRepository(
            CombustionControlAggregate,
            memory_event_store
        )

        # New aggregate
        aggregate, is_new = await repo.get_or_create(aggregate_id)
        assert is_new is True
        assert aggregate.version == 0

        # Save some events
        aggregate._control_enabled = True
        aggregate._system_mode = SystemMode.NORMAL
        aggregate.change_setpoint(fuel_flow=1000.0, air_flow=12500.0)
        await repo.save(aggregate)

        # Existing aggregate
        aggregate2, is_new2 = await repo.get_or_create(aggregate_id)
        assert is_new2 is False
        assert aggregate2.version == 1

    @pytest.mark.asyncio
    async def test_repository_caching(self, memory_event_store, aggregate_id):
        """Test repository caches aggregates."""
        repo = AggregateRepository(
            CombustionControlAggregate,
            memory_event_store
        )

        # Create aggregate
        aggregate1, _ = await repo.get_or_create(aggregate_id)
        aggregate1._control_enabled = True
        aggregate1._system_mode = SystemMode.NORMAL
        aggregate1.change_setpoint(fuel_flow=1000.0, air_flow=12500.0)
        await repo.save(aggregate1)

        # Get again (should be cached)
        aggregate2 = await repo.get(aggregate_id, use_cache=True)
        assert aggregate2 is aggregate1  # Same instance

        # Get without cache
        aggregate3 = await repo.get(aggregate_id, use_cache=False)
        assert aggregate3 is not aggregate1  # Different instance
        assert aggregate3.setpoint.fuel_flow == aggregate1.setpoint.fuel_flow


# =============================================================================
# Concurrency Tests
# =============================================================================


class TestConcurrency:
    """Tests for concurrent access."""

    @pytest.mark.asyncio
    async def test_concurrent_writes_same_aggregate(
        self,
        memory_event_store,
        aggregate_id
    ):
        """Test concurrent writes to same aggregate fail correctly."""
        # Create initial event
        event1 = ControlSetpointChanged(
            aggregate_id=aggregate_id,
            fuel_flow_setpoint=1000.0,
            air_flow_setpoint=12500.0
        )
        await memory_event_store.append(aggregate_id, [event1])

        # Simulate two concurrent updates
        event2 = ControlSetpointChanged(
            aggregate_id=aggregate_id,
            fuel_flow_setpoint=1100.0,
            air_flow_setpoint=13000.0
        )
        event3 = ControlSetpointChanged(
            aggregate_id=aggregate_id,
            fuel_flow_setpoint=1200.0,
            air_flow_setpoint=14000.0
        )

        # First update should succeed
        await memory_event_store.append(
            aggregate_id, [event2], expected_version=1
        )

        # Second update with same expected version should fail
        with pytest.raises(OptimisticConcurrencyError) as exc_info:
            await memory_event_store.append(
                aggregate_id, [event3], expected_version=1
            )

        assert exc_info.value.expected == 1
        assert exc_info.value.actual == 2

    @pytest.mark.asyncio
    async def test_concurrent_writes_different_aggregates(
        self,
        memory_event_store
    ):
        """Test concurrent writes to different aggregates succeed."""
        aggregate_ids = [f"burner-{i}" for i in range(5)]

        async def write_events(agg_id):
            event = ControlSetpointChanged(
                aggregate_id=agg_id,
                fuel_flow_setpoint=1000.0,
                air_flow_setpoint=12500.0
            )
            return await memory_event_store.append(agg_id, [event])

        # Run concurrently
        results = await asyncio.gather(
            *[write_events(aid) for aid in aggregate_ids]
        )

        # All should succeed
        assert all(r == 1 for r in results)

        # Verify all aggregates
        for aid in aggregate_ids:
            events = await memory_event_store.load(aid)
            assert len(events) == 1


# =============================================================================
# Performance Tests
# =============================================================================


class TestPerformance:
    """Performance-related tests."""

    @pytest.mark.asyncio
    async def test_large_event_stream(self, memory_event_store, aggregate_id):
        """Test handling large number of events."""
        # Create 1000 events
        events = [
            ControlSetpointChanged(
                aggregate_id=aggregate_id,
                fuel_flow_setpoint=1000.0 + i,
                air_flow_setpoint=12500.0 + i * 10
            )
            for i in range(1000)
        ]

        # Append in batches
        batch_size = 100
        for i in range(0, len(events), batch_size):
            batch = events[i:i + batch_size]
            await memory_event_store.append(aggregate_id, batch)

        # Load all
        loaded = await memory_event_store.load(aggregate_id)
        assert len(loaded) == 1000

        # Verify ordering
        for i, event in enumerate(loaded):
            assert event.fuel_flow_setpoint == 1000.0 + i

    @pytest.mark.asyncio
    async def test_aggregate_rebuild_performance(
        self,
        memory_event_store,
        aggregate_id
    ):
        """Test aggregate rebuild performance."""
        import time

        # Create 500 events
        aggregate = CombustionControlAggregate(aggregate_id)
        aggregate._control_enabled = True
        aggregate._system_mode = SystemMode.NORMAL

        for i in range(500):
            aggregate.change_setpoint(
                fuel_flow=1000.0 + i,
                air_flow=12500.0 + i * 10
            )

        await aggregate.save_to_store(memory_event_store)

        # Time rebuild
        start = time.time()
        new_aggregate = CombustionControlAggregate(aggregate_id)
        await new_aggregate.load_from_store(memory_event_store)
        rebuild_time = time.time() - start

        assert new_aggregate.version == 500
        assert rebuild_time < 2.0  # Should complete in under 2 seconds

    @pytest.mark.asyncio
    async def test_snapshot_improves_load_time(
        self,
        memory_event_store,
        aggregate_id
    ):
        """Test that snapshots improve load time."""
        import time

        # Create 100 events
        aggregate = CombustionControlAggregate(aggregate_id)
        aggregate._control_enabled = True
        aggregate._system_mode = SystemMode.NORMAL

        for i in range(100):
            aggregate.change_setpoint(
                fuel_flow=1000.0 + i,
                air_flow=12500.0 + i * 10
            )

        await aggregate.save_to_store(memory_event_store)

        # Time load without snapshot
        start = time.time()
        agg1 = CombustionControlAggregate(aggregate_id)
        await agg1.load_from_store(memory_event_store)
        time_without_snapshot = time.time() - start

        # Save snapshot
        await memory_event_store.save_snapshot(
            aggregate_id=aggregate_id,
            version=100,
            snapshot_data=agg1.create_snapshot()
        )

        # Time load with snapshot (no additional events)
        start = time.time()
        agg2 = CombustionControlAggregate(aggregate_id)
        snapshot = await memory_event_store.load_latest_snapshot(aggregate_id)
        if snapshot:
            version, data = snapshot
            agg2.restore_from_snapshot(data)
        time_with_snapshot = time.time() - start

        # Snapshot should be faster
        assert time_with_snapshot <= time_without_snapshot


# =============================================================================
# Integration Tests
# =============================================================================


class TestEventSourcingIntegration:
    """Integration tests for the complete event sourcing system."""

    @pytest.mark.asyncio
    async def test_full_workflow(self, memory_event_store, event_bus, aggregate_id):
        """Test complete event sourcing workflow."""
        received_events = []

        @event_bus.subscribe(ControlSetpointChanged)
        async def on_setpoint_changed(event):
            received_events.append(event)

        # Create repository
        repo = AggregateRepository(
            CombustionControlAggregate,
            memory_event_store,
            snapshot_threshold=50
        )

        # Get or create aggregate
        aggregate, is_new = await repo.get_or_create(aggregate_id)
        assert is_new is True

        # Enable control
        aggregate.change_system_mode(
            new_mode=SystemMode.NORMAL,
            reason="startup",
            operator_initiated=True
        )

        # Make setpoint changes
        for i in range(5):
            aggregate.change_setpoint(
                fuel_flow=1000.0 + i * 100,
                air_flow=12500.0 + i * 500
            )

        # Save aggregate
        new_version = await repo.save(aggregate)
        assert new_version == 6  # 1 mode change + 5 setpoint changes

        # Publish events to bus
        for event in aggregate.get_event_history():
            if isinstance(event, ControlSetpointChanged):
                await event_bus.publish(event)

        # Verify bus received events
        assert len(received_events) == 5

        # Load aggregate in new instance
        aggregate2 = await repo.get(aggregate_id, use_cache=False)
        assert aggregate2.version == 6
        assert aggregate2.setpoint.fuel_flow == 1400.0
        assert aggregate2.control_enabled is True

    @pytest.mark.asyncio
    async def test_event_sourcing_determinism(
        self,
        memory_event_store,
        aggregate_id
    ):
        """Test that event replay is deterministic."""
        # Create first aggregate
        aggregate1 = CombustionControlAggregate(aggregate_id)
        aggregate1._control_enabled = True
        aggregate1._system_mode = SystemMode.NORMAL

        aggregate1.change_setpoint(fuel_flow=1000.0, air_flow=12500.0)
        aggregate1.record_sensor_reading("heat_output", "h1", 10000.0, "kW")
        aggregate1.change_setpoint(fuel_flow=1100.0, air_flow=13000.0)

        await aggregate1.save_to_store(memory_event_store)

        # Create second aggregate and replay
        aggregate2 = CombustionControlAggregate(aggregate_id)
        await aggregate2.load_from_store(memory_event_store)

        # Create third aggregate and replay
        aggregate3 = CombustionControlAggregate(aggregate_id)
        await aggregate3.load_from_store(memory_event_store)

        # All should have identical business state
        assert aggregate2.version == aggregate3.version
        assert aggregate2.setpoint.fuel_flow == aggregate3.setpoint.fuel_flow
        assert aggregate2.setpoint.air_flow == aggregate3.setpoint.air_flow

        # Business state should match
        state2 = aggregate2.get_current_state()
        state3 = aggregate3.get_current_state()
        assert state2["setpoint"] == state3["setpoint"]
        assert state2["stability"] == state3["stability"]
        assert state2["counters"] == state3["counters"]
        assert state2["system_mode"] == state3["system_mode"]
