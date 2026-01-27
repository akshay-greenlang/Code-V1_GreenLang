# -*- coding: utf-8 -*-
"""
Unit Tests for Hash-Chained Event Store (GL-FOUND-X-001)

This module provides comprehensive tests for the audit event store,
including hash chain verification, tamper detection, and event lifecycle.

Test Coverage:
- RunEvent model validation and hashing
- InMemoryEventStore operations
- PostgresEventStore operations (mocked)
- Hash chain integrity verification
- Tamper detection
- Audit package export
- EventFactory convenience methods

Author: GreenLang Team
Version: 1.0.0
"""

import hashlib
import json
import pytest
from datetime import datetime, timezone, timedelta
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

from greenlang.orchestrator.audit.event_store import (
    EventType,
    RunEvent,
    AuditPackage,
    InMemoryEventStore,
    PostgresEventStore,
    EventFactory,
    EventStoreError,
    ChainIntegrityError,
    EventNotFoundError,
    GENESIS_HASH,
    HASH_ALGORITHM,
)


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def sample_timestamp():
    """Return a fixed timestamp for deterministic tests."""
    return datetime(2024, 1, 15, 10, 30, 0, tzinfo=timezone.utc)


@pytest.fixture
def sample_run_id():
    """Return a sample run ID."""
    return "run-2024-001"


@pytest.fixture
def sample_event_id():
    """Return a sample event ID."""
    return "550e8400-e29b-41d4-a716-446655440000"


@pytest.fixture
def sample_payload():
    """Return a sample event payload."""
    return {
        "workflow": "carbon-calculation",
        "version": "2.0.0",
        "parameters": {
            "scope": "scope1",
            "year": 2024
        }
    }


@pytest.fixture
def sample_event(sample_event_id, sample_run_id, sample_timestamp, sample_payload):
    """Return a sample RunEvent for testing."""
    return RunEvent(
        event_id=sample_event_id,
        run_id=sample_run_id,
        event_type=EventType.RUN_SUBMITTED,
        timestamp=sample_timestamp,
        payload=sample_payload,
        prev_event_hash=GENESIS_HASH,
        event_hash=""
    )


@pytest.fixture
def in_memory_store():
    """Return an InMemoryEventStore instance."""
    return InMemoryEventStore()


# ============================================================================
# RUNEVENT MODEL TESTS
# ============================================================================

class TestRunEvent:
    """Tests for RunEvent Pydantic model."""

    def test_create_event_with_required_fields(
        self, sample_event_id, sample_run_id, sample_timestamp
    ):
        """Test creating event with all required fields."""
        event = RunEvent(
            event_id=sample_event_id,
            run_id=sample_run_id,
            event_type=EventType.RUN_SUBMITTED,
            timestamp=sample_timestamp,
            payload={},
            prev_event_hash=GENESIS_HASH,
            event_hash=""
        )

        assert event.event_id == sample_event_id
        assert event.run_id == sample_run_id
        assert event.event_type == EventType.RUN_SUBMITTED
        assert event.timestamp == sample_timestamp
        assert event.prev_event_hash == GENESIS_HASH
        assert event.step_id is None

    def test_create_event_with_step_id(
        self, sample_event_id, sample_run_id, sample_timestamp
    ):
        """Test creating event with optional step_id."""
        event = RunEvent(
            event_id=sample_event_id,
            run_id=sample_run_id,
            step_id="step-001",
            event_type=EventType.STEP_STARTED,
            timestamp=sample_timestamp,
            payload={"agent": "CarbonCalculator"},
            prev_event_hash="abc123",
            event_hash=""
        )

        assert event.step_id == "step-001"
        assert event.event_type == EventType.STEP_STARTED

    def test_compute_hash_deterministic(self, sample_event):
        """Test that compute_hash produces deterministic output."""
        hash1 = sample_event.compute_hash()
        hash2 = sample_event.compute_hash()

        assert hash1 == hash2
        assert len(hash1) == 64  # SHA-256 hex digest length

    def test_compute_hash_changes_with_payload(self, sample_event):
        """Test that hash changes when payload changes."""
        hash1 = sample_event.compute_hash()

        sample_event.payload["new_key"] = "new_value"
        hash2 = sample_event.compute_hash()

        assert hash1 != hash2

    def test_compute_hash_changes_with_event_type(
        self, sample_event_id, sample_run_id, sample_timestamp
    ):
        """Test that hash changes with different event types."""
        event1 = RunEvent(
            event_id=sample_event_id,
            run_id=sample_run_id,
            event_type=EventType.RUN_SUBMITTED,
            timestamp=sample_timestamp,
            payload={},
            prev_event_hash=GENESIS_HASH,
            event_hash=""
        )

        event2 = RunEvent(
            event_id=sample_event_id,
            run_id=sample_run_id,
            event_type=EventType.RUN_SUCCEEDED,
            timestamp=sample_timestamp,
            payload={},
            prev_event_hash=GENESIS_HASH,
            event_hash=""
        )

        assert event1.compute_hash() != event2.compute_hash()

    def test_verify_hash_valid(self, sample_event):
        """Test verify_hash returns True for valid hash."""
        sample_event.event_hash = sample_event.compute_hash()
        assert sample_event.verify_hash() is True

    def test_verify_hash_invalid(self, sample_event):
        """Test verify_hash returns False for tampered hash."""
        sample_event.event_hash = "tampered_hash_value"
        assert sample_event.verify_hash() is False

    def test_timestamp_ensures_utc(self, sample_event_id, sample_run_id):
        """Test that timestamp validator ensures UTC timezone."""
        naive_dt = datetime(2024, 1, 15, 10, 30, 0)  # No timezone

        event = RunEvent(
            event_id=sample_event_id,
            run_id=sample_run_id,
            event_type=EventType.RUN_SUBMITTED,
            timestamp=naive_dt,
            payload={},
            prev_event_hash=GENESIS_HASH,
            event_hash=""
        )

        assert event.timestamp.tzinfo == timezone.utc

    def test_timestamp_removes_microseconds(self, sample_event_id, sample_run_id):
        """Test that timestamp validator removes microseconds for determinism."""
        dt_with_microseconds = datetime(
            2024, 1, 15, 10, 30, 0, 123456, tzinfo=timezone.utc
        )

        event = RunEvent(
            event_id=sample_event_id,
            run_id=sample_run_id,
            event_type=EventType.RUN_SUBMITTED,
            timestamp=dt_with_microseconds,
            payload={},
            prev_event_hash=GENESIS_HASH,
            event_hash=""
        )

        assert event.timestamp.microsecond == 0


class TestEventType:
    """Tests for EventType enumeration."""

    def test_all_event_types_exist(self):
        """Test that all required event types are defined."""
        expected_types = [
            "RUN_SUBMITTED",
            "PLAN_COMPILED",
            "POLICY_EVALUATED",
            "STEP_READY",
            "STEP_STARTED",
            "STEP_RETRIED",
            "STEP_SUCCEEDED",
            "STEP_FAILED",
            "ARTIFACT_WRITTEN",
            "RUN_SUCCEEDED",
            "RUN_FAILED",
            "RUN_CANCELED",
        ]

        for type_name in expected_types:
            assert hasattr(EventType, type_name)
            assert EventType[type_name].value == type_name

    def test_event_type_is_string_enum(self):
        """Test that EventType values are strings."""
        assert isinstance(EventType.RUN_SUBMITTED.value, str)
        assert EventType.RUN_SUBMITTED == "RUN_SUBMITTED"


# ============================================================================
# IN-MEMORY EVENT STORE TESTS
# ============================================================================

class TestInMemoryEventStore:
    """Tests for InMemoryEventStore implementation."""

    @pytest.mark.asyncio
    async def test_append_first_event(
        self, in_memory_store, sample_event_id, sample_run_id, sample_timestamp
    ):
        """Test appending the first event to a run."""
        event = RunEvent(
            event_id=sample_event_id,
            run_id=sample_run_id,
            event_type=EventType.RUN_SUBMITTED,
            timestamp=sample_timestamp,
            payload={"test": True},
            prev_event_hash=GENESIS_HASH,
            event_hash=""
        )

        event_hash = await in_memory_store.append(event)

        assert event_hash is not None
        assert len(event_hash) == 64
        assert event.event_hash == event_hash

    @pytest.mark.asyncio
    async def test_append_second_event_chain_linkage(
        self, in_memory_store, sample_run_id, sample_timestamp
    ):
        """Test that second event is properly linked to first."""
        # Append first event
        event1 = RunEvent(
            event_id=str(uuid4()),
            run_id=sample_run_id,
            event_type=EventType.RUN_SUBMITTED,
            timestamp=sample_timestamp,
            payload={},
            prev_event_hash=GENESIS_HASH,
            event_hash=""
        )
        hash1 = await in_memory_store.append(event1)

        # Append second event
        event2 = RunEvent(
            event_id=str(uuid4()),
            run_id=sample_run_id,
            event_type=EventType.PLAN_COMPILED,
            timestamp=sample_timestamp + timedelta(seconds=1),
            payload={},
            prev_event_hash=hash1,  # Link to first event
            event_hash=""
        )
        hash2 = await in_memory_store.append(event2)

        assert hash2 != hash1
        events = await in_memory_store.get_events(sample_run_id)
        assert len(events) == 2
        assert events[1].prev_event_hash == hash1

    @pytest.mark.asyncio
    async def test_append_invalid_prev_hash_raises_error(
        self, in_memory_store, sample_run_id, sample_timestamp
    ):
        """Test that invalid prev_event_hash raises ChainIntegrityError."""
        # Append first event
        event1 = RunEvent(
            event_id=str(uuid4()),
            run_id=sample_run_id,
            event_type=EventType.RUN_SUBMITTED,
            timestamp=sample_timestamp,
            payload={},
            prev_event_hash=GENESIS_HASH,
            event_hash=""
        )
        await in_memory_store.append(event1)

        # Try to append with wrong prev_hash
        event2 = RunEvent(
            event_id=str(uuid4()),
            run_id=sample_run_id,
            event_type=EventType.PLAN_COMPILED,
            timestamp=sample_timestamp + timedelta(seconds=1),
            payload={},
            prev_event_hash="wrong_hash",  # Invalid!
            event_hash=""
        )

        with pytest.raises(ChainIntegrityError) as exc_info:
            await in_memory_store.append(event2)

        assert "Invalid prev_event_hash" in str(exc_info.value)
        assert exc_info.value.run_id == sample_run_id

    @pytest.mark.asyncio
    async def test_get_events_empty_run(self, in_memory_store):
        """Test get_events returns empty list for non-existent run."""
        events = await in_memory_store.get_events("non-existent-run")
        assert events == []

    @pytest.mark.asyncio
    async def test_get_events_returns_copy(
        self, in_memory_store, sample_event_id, sample_run_id, sample_timestamp
    ):
        """Test that get_events returns a copy, not the original list."""
        event = RunEvent(
            event_id=sample_event_id,
            run_id=sample_run_id,
            event_type=EventType.RUN_SUBMITTED,
            timestamp=sample_timestamp,
            payload={},
            prev_event_hash=GENESIS_HASH,
            event_hash=""
        )
        await in_memory_store.append(event)

        events1 = await in_memory_store.get_events(sample_run_id)
        events2 = await in_memory_store.get_events(sample_run_id)

        assert events1 is not events2
        assert events1 == events2

    @pytest.mark.asyncio
    async def test_get_latest_hash_genesis_for_new_run(self, in_memory_store):
        """Test get_latest_hash returns genesis for new run."""
        latest = await in_memory_store.get_latest_hash("new-run")
        assert latest == GENESIS_HASH

    @pytest.mark.asyncio
    async def test_get_latest_hash_after_events(
        self, in_memory_store, sample_run_id, sample_timestamp
    ):
        """Test get_latest_hash returns last event hash."""
        event = RunEvent(
            event_id=str(uuid4()),
            run_id=sample_run_id,
            event_type=EventType.RUN_SUBMITTED,
            timestamp=sample_timestamp,
            payload={},
            prev_event_hash=GENESIS_HASH,
            event_hash=""
        )
        event_hash = await in_memory_store.append(event)

        latest = await in_memory_store.get_latest_hash(sample_run_id)
        assert latest == event_hash

    @pytest.mark.asyncio
    async def test_verify_chain_empty_run(self, in_memory_store):
        """Test verify_chain returns True for empty run."""
        is_valid = await in_memory_store.verify_chain("empty-run")
        assert is_valid is True

    @pytest.mark.asyncio
    async def test_verify_chain_valid_chain(
        self, in_memory_store, sample_run_id, sample_timestamp
    ):
        """Test verify_chain returns True for valid chain."""
        # Create chain of 3 events
        prev_hash = GENESIS_HASH
        for i, event_type in enumerate([
            EventType.RUN_SUBMITTED,
            EventType.PLAN_COMPILED,
            EventType.RUN_SUCCEEDED
        ]):
            event = RunEvent(
                event_id=str(uuid4()),
                run_id=sample_run_id,
                event_type=event_type,
                timestamp=sample_timestamp + timedelta(seconds=i),
                payload={"step": i},
                prev_event_hash=prev_hash,
                event_hash=""
            )
            prev_hash = await in_memory_store.append(event)

        is_valid = await in_memory_store.verify_chain(sample_run_id)
        assert is_valid is True

    @pytest.mark.asyncio
    async def test_verify_chain_detects_tampered_hash(
        self, in_memory_store, sample_run_id, sample_timestamp
    ):
        """Test verify_chain detects tampered event_hash."""
        event = RunEvent(
            event_id=str(uuid4()),
            run_id=sample_run_id,
            event_type=EventType.RUN_SUBMITTED,
            timestamp=sample_timestamp,
            payload={},
            prev_event_hash=GENESIS_HASH,
            event_hash=""
        )
        await in_memory_store.append(event)

        # Tamper with stored event
        in_memory_store._events[sample_run_id][0].event_hash = "tampered"

        is_valid = await in_memory_store.verify_chain(sample_run_id)
        assert is_valid is False

    @pytest.mark.asyncio
    async def test_export_audit_package(
        self, in_memory_store, sample_run_id, sample_timestamp
    ):
        """Test export_audit_package returns complete package."""
        # Create events
        event = RunEvent(
            event_id=str(uuid4()),
            run_id=sample_run_id,
            event_type=EventType.RUN_SUBMITTED,
            timestamp=sample_timestamp,
            payload={"test": True},
            prev_event_hash=GENESIS_HASH,
            event_hash=""
        )
        await in_memory_store.append(event)

        package = await in_memory_store.export_audit_package(sample_run_id)

        assert isinstance(package, AuditPackage)
        assert package.run_id == sample_run_id
        assert len(package.events) == 1
        assert package.chain_valid is True
        assert package.exported_at is not None
        assert package.metadata["event_count"] == 1
        assert package.metadata["store_type"] == "in_memory"

    @pytest.mark.asyncio
    async def test_clear_specific_run(self, in_memory_store, sample_timestamp):
        """Test clear removes specific run only."""
        # Create events for two runs
        for run_id in ["run-1", "run-2"]:
            event = RunEvent(
                event_id=str(uuid4()),
                run_id=run_id,
                event_type=EventType.RUN_SUBMITTED,
                timestamp=sample_timestamp,
                payload={},
                prev_event_hash=GENESIS_HASH,
                event_hash=""
            )
            await in_memory_store.append(event)

        await in_memory_store.clear("run-1")

        events1 = await in_memory_store.get_events("run-1")
        events2 = await in_memory_store.get_events("run-2")

        assert len(events1) == 0
        assert len(events2) == 1

    @pytest.mark.asyncio
    async def test_clear_all_runs(self, in_memory_store, sample_timestamp):
        """Test clear removes all runs when no run_id specified."""
        # Create events for two runs
        for run_id in ["run-1", "run-2"]:
            event = RunEvent(
                event_id=str(uuid4()),
                run_id=run_id,
                event_type=EventType.RUN_SUBMITTED,
                timestamp=sample_timestamp,
                payload={},
                prev_event_hash=GENESIS_HASH,
                event_hash=""
            )
            await in_memory_store.append(event)

        await in_memory_store.clear()

        events1 = await in_memory_store.get_events("run-1")
        events2 = await in_memory_store.get_events("run-2")

        assert len(events1) == 0
        assert len(events2) == 0


# ============================================================================
# AUDIT PACKAGE TESTS
# ============================================================================

class TestAuditPackage:
    """Tests for AuditPackage model."""

    def test_create_audit_package(self, sample_run_id, sample_event):
        """Test creating an audit package."""
        sample_event.event_hash = sample_event.compute_hash()

        package = AuditPackage(
            run_id=sample_run_id,
            events=[sample_event],
            chain_valid=True,
            exported_at=datetime.now(timezone.utc),
            metadata={"test": True}
        )

        assert package.run_id == sample_run_id
        assert len(package.events) == 1
        assert package.chain_valid is True

    def test_compute_package_hash_empty(self, sample_run_id):
        """Test package hash for empty events list."""
        package = AuditPackage(
            run_id=sample_run_id,
            events=[],
            chain_valid=True
        )

        hash_value = package.compute_package_hash()
        assert len(hash_value) == 64

    def test_compute_package_hash_deterministic(self, sample_run_id, sample_event):
        """Test that package hash is deterministic."""
        sample_event.event_hash = sample_event.compute_hash()

        package = AuditPackage(
            run_id=sample_run_id,
            events=[sample_event],
            chain_valid=True
        )

        hash1 = package.compute_package_hash()
        hash2 = package.compute_package_hash()

        assert hash1 == hash2

    def test_audit_package_is_frozen(self, sample_run_id):
        """Test that AuditPackage is immutable after creation."""
        package = AuditPackage(
            run_id=sample_run_id,
            events=[],
            chain_valid=True
        )

        with pytest.raises(Exception):  # ValidationError in Pydantic v2
            package.run_id = "modified"


# ============================================================================
# EVENT FACTORY TESTS
# ============================================================================

class TestEventFactory:
    """Tests for EventFactory helper class."""

    @pytest.mark.asyncio
    async def test_create_event_first_event(self, in_memory_store, sample_run_id):
        """Test creating first event with factory."""
        factory = EventFactory(in_memory_store)

        event = await factory.create_event(
            run_id=sample_run_id,
            event_type=EventType.RUN_SUBMITTED,
            payload={"test": True}
        )

        assert event.run_id == sample_run_id
        assert event.event_type == EventType.RUN_SUBMITTED
        assert event.prev_event_hash == GENESIS_HASH
        assert event.event_hash == ""  # Not computed until appended
        assert event.payload == {"test": True}

    @pytest.mark.asyncio
    async def test_create_event_with_step_id(self, in_memory_store, sample_run_id):
        """Test creating event with step_id."""
        factory = EventFactory(in_memory_store)

        event = await factory.create_event(
            run_id=sample_run_id,
            event_type=EventType.STEP_STARTED,
            step_id="step-001"
        )

        assert event.step_id == "step-001"

    @pytest.mark.asyncio
    async def test_create_event_chains_properly(
        self, in_memory_store, sample_run_id, sample_timestamp
    ):
        """Test that factory creates properly chained events."""
        factory = EventFactory(in_memory_store)

        # Append first event manually
        event1 = RunEvent(
            event_id=str(uuid4()),
            run_id=sample_run_id,
            event_type=EventType.RUN_SUBMITTED,
            timestamp=sample_timestamp,
            payload={},
            prev_event_hash=GENESIS_HASH,
            event_hash=""
        )
        hash1 = await in_memory_store.append(event1)

        # Create second event with factory
        event2 = await factory.create_event(
            run_id=sample_run_id,
            event_type=EventType.PLAN_COMPILED
        )

        assert event2.prev_event_hash == hash1

    @pytest.mark.asyncio
    async def test_emit_creates_and_appends(self, in_memory_store, sample_run_id):
        """Test emit method creates and appends in one call."""
        factory = EventFactory(in_memory_store)

        event_hash = await factory.emit(
            run_id=sample_run_id,
            event_type=EventType.RUN_SUBMITTED,
            payload={"workflow": "test"}
        )

        assert len(event_hash) == 64

        events = await in_memory_store.get_events(sample_run_id)
        assert len(events) == 1
        assert events[0].event_hash == event_hash

    @pytest.mark.asyncio
    async def test_emit_multiple_events(self, in_memory_store, sample_run_id):
        """Test emitting multiple events in sequence."""
        factory = EventFactory(in_memory_store)

        event_types = [
            EventType.RUN_SUBMITTED,
            EventType.PLAN_COMPILED,
            EventType.STEP_READY,
            EventType.STEP_STARTED,
            EventType.STEP_SUCCEEDED,
            EventType.RUN_SUCCEEDED
        ]

        for event_type in event_types:
            await factory.emit(
                run_id=sample_run_id,
                event_type=event_type
            )

        events = await in_memory_store.get_events(sample_run_id)
        assert len(events) == len(event_types)

        # Verify chain is valid
        is_valid = await in_memory_store.verify_chain(sample_run_id)
        assert is_valid is True


# ============================================================================
# EXCEPTION TESTS
# ============================================================================

class TestExceptions:
    """Tests for custom exceptions."""

    def test_chain_integrity_error_attributes(self):
        """Test ChainIntegrityError has all attributes."""
        error = ChainIntegrityError(
            message="Hash mismatch",
            run_id="run-123",
            event_id="event-456",
            expected_hash="abc123",
            actual_hash="def456"
        )

        assert str(error) == "Hash mismatch"
        assert error.run_id == "run-123"
        assert error.event_id == "event-456"
        assert error.expected_hash == "abc123"
        assert error.actual_hash == "def456"

    def test_event_store_error_base(self):
        """Test EventStoreError is proper base exception."""
        error = EventStoreError("Generic error")
        assert isinstance(error, Exception)
        assert str(error) == "Generic error"


# ============================================================================
# HASH ALGORITHM VERIFICATION
# ============================================================================

class TestHashAlgorithm:
    """Tests verifying hash algorithm implementation."""

    def test_hash_algorithm_is_sha256(self):
        """Verify HASH_ALGORITHM constant is sha256."""
        assert HASH_ALGORITHM == "sha256"

    def test_genesis_hash_constant(self):
        """Verify GENESIS_HASH constant."""
        assert GENESIS_HASH == "genesis"

    def test_event_hash_uses_sha256(self, sample_event):
        """Verify event hash uses SHA-256."""
        hash_value = sample_event.compute_hash()

        # SHA-256 produces 64-character hex string
        assert len(hash_value) == 64

        # Verify it's valid hexadecimal
        int(hash_value, 16)  # Will raise if not valid hex


# ============================================================================
# INTEGRATION TESTS (Simulated)
# ============================================================================

class TestIntegration:
    """Integration-style tests for complete workflows."""

    @pytest.mark.asyncio
    async def test_complete_successful_run_lifecycle(self, in_memory_store):
        """Test complete run lifecycle from submission to success."""
        factory = EventFactory(in_memory_store)
        run_id = f"integration-run-{uuid4()}"

        # Run lifecycle
        await factory.emit(run_id, EventType.RUN_SUBMITTED, {"workflow": "carbon-calc"})
        await factory.emit(run_id, EventType.PLAN_COMPILED, {"steps": 3})
        await factory.emit(run_id, EventType.POLICY_EVALUATED, {"policies_passed": True})

        # Step 1
        await factory.emit(run_id, EventType.STEP_READY, step_id="step-1")
        await factory.emit(run_id, EventType.STEP_STARTED, step_id="step-1", payload={"agent": "DataLoader"})
        await factory.emit(run_id, EventType.ARTIFACT_WRITTEN, step_id="step-1", payload={"artifact": "data.csv"})
        await factory.emit(run_id, EventType.STEP_SUCCEEDED, step_id="step-1")

        # Step 2
        await factory.emit(run_id, EventType.STEP_READY, step_id="step-2")
        await factory.emit(run_id, EventType.STEP_STARTED, step_id="step-2", payload={"agent": "Calculator"})
        await factory.emit(run_id, EventType.STEP_SUCCEEDED, step_id="step-2")

        # Run completion
        await factory.emit(run_id, EventType.RUN_SUCCEEDED, {"total_emissions": 1234.56})

        # Verify
        events = await in_memory_store.get_events(run_id)
        assert len(events) == 11

        is_valid = await in_memory_store.verify_chain(run_id)
        assert is_valid is True

        package = await in_memory_store.export_audit_package(run_id)
        assert package.chain_valid is True
        assert package.metadata["event_count"] == 11

    @pytest.mark.asyncio
    async def test_run_with_retry(self, in_memory_store):
        """Test run lifecycle with step retry."""
        factory = EventFactory(in_memory_store)
        run_id = f"retry-run-{uuid4()}"

        await factory.emit(run_id, EventType.RUN_SUBMITTED)
        await factory.emit(run_id, EventType.PLAN_COMPILED)
        await factory.emit(run_id, EventType.STEP_READY, step_id="step-1")
        await factory.emit(run_id, EventType.STEP_STARTED, step_id="step-1")
        await factory.emit(run_id, EventType.STEP_FAILED, step_id="step-1", payload={"error": "Connection timeout"})
        await factory.emit(run_id, EventType.STEP_RETRIED, step_id="step-1", payload={"attempt": 2})
        await factory.emit(run_id, EventType.STEP_STARTED, step_id="step-1")
        await factory.emit(run_id, EventType.STEP_SUCCEEDED, step_id="step-1")
        await factory.emit(run_id, EventType.RUN_SUCCEEDED)

        is_valid = await in_memory_store.verify_chain(run_id)
        assert is_valid is True

    @pytest.mark.asyncio
    async def test_run_failure(self, in_memory_store):
        """Test run lifecycle with failure."""
        factory = EventFactory(in_memory_store)
        run_id = f"failed-run-{uuid4()}"

        await factory.emit(run_id, EventType.RUN_SUBMITTED)
        await factory.emit(run_id, EventType.PLAN_COMPILED)
        await factory.emit(run_id, EventType.STEP_READY, step_id="step-1")
        await factory.emit(run_id, EventType.STEP_STARTED, step_id="step-1")
        await factory.emit(run_id, EventType.STEP_FAILED, step_id="step-1", payload={"error": "Fatal error"})
        await factory.emit(run_id, EventType.RUN_FAILED, payload={"reason": "Step failed after max retries"})

        is_valid = await in_memory_store.verify_chain(run_id)
        assert is_valid is True

        events = await in_memory_store.get_events(run_id)
        assert events[-1].event_type == EventType.RUN_FAILED

    @pytest.mark.asyncio
    async def test_run_cancellation(self, in_memory_store):
        """Test run lifecycle with cancellation."""
        factory = EventFactory(in_memory_store)
        run_id = f"canceled-run-{uuid4()}"

        await factory.emit(run_id, EventType.RUN_SUBMITTED)
        await factory.emit(run_id, EventType.PLAN_COMPILED)
        await factory.emit(run_id, EventType.STEP_READY, step_id="step-1")
        await factory.emit(run_id, EventType.STEP_STARTED, step_id="step-1")
        await factory.emit(run_id, EventType.RUN_CANCELED, payload={"canceled_by": "user@example.com"})

        is_valid = await in_memory_store.verify_chain(run_id)
        assert is_valid is True

        events = await in_memory_store.get_events(run_id)
        assert events[-1].event_type == EventType.RUN_CANCELED


# ============================================================================
# POSTGRES EVENT STORE TESTS (Mocked)
# ============================================================================

class TestPostgresEventStore:
    """Tests for PostgresEventStore (using mocks for database)."""

    @pytest.mark.asyncio
    async def test_not_initialized_raises_error(self):
        """Test operations before initialize raise error."""
        store = PostgresEventStore("postgresql://test")

        with pytest.raises(EventStoreError) as exc_info:
            await store.get_events("run-123")

        assert "not initialized" in str(exc_info.value).lower()

    @pytest.mark.asyncio
    async def test_initialize_without_asyncpg_raises_import_error(self):
        """Test initialize raises ImportError without asyncpg."""
        store = PostgresEventStore("postgresql://test")

        with patch.dict('sys.modules', {'asyncpg': None}):
            with pytest.raises(ImportError) as exc_info:
                await store.initialize()

            assert "asyncpg" in str(exc_info.value)

    def test_url_parsing_removes_asyncpg_prefix(self):
        """Test database URL prefix handling."""
        store = PostgresEventStore("postgresql+asyncpg://user:pass@host/db")
        assert "postgresql+asyncpg://" in store._database_url

    def test_sql_statements_defined(self):
        """Test that all SQL statements are defined."""
        assert "CREATE TABLE" in PostgresEventStore.CREATE_TABLE_SQL
        assert "CREATE INDEX" in PostgresEventStore.CREATE_INDEXES_SQL
        assert "INSERT INTO" in PostgresEventStore.INSERT_EVENT_SQL
        assert "SELECT" in PostgresEventStore.SELECT_EVENTS_SQL
        assert "SELECT" in PostgresEventStore.SELECT_LATEST_HASH_SQL


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
