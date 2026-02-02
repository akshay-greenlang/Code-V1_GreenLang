# -*- coding: utf-8 -*-
"""
Hash Chain Verification Tests for GL-FOUND-X-001 Orchestrator

Tests for the audit event hash chain that ensures tamper-evidence:

1. Hash Chain Integrity
   - Each event links to previous via prev_event_hash
   - First event links to "genesis"
   - Chain verification detects tampering

2. Event Hash Computation
   - Hashes are deterministic
   - Hash includes: event_id, prev_hash, type, timestamp, payload
   - Same event produces same hash

3. Artifact Checksum Verification
   - Artifacts have SHA-256 checksums
   - Checksums match for identical content
   - Verification detects corruption

4. Cross-Environment Consistency
   - Hash chain works with mock S3 store
   - Hash chain works with mock K8s executor

Author: GreenLang Team
Version: 1.0.0
Coverage Target: 85%+
"""

import hashlib
import json
from datetime import datetime, timezone
from typing import Dict, Any, List
from uuid import uuid4

import pytest
import pytest_asyncio

from greenlang.utilities.determinism import (
    DeterministicClock,
    freeze_time,
    unfreeze_time,
)
from greenlang.orchestrator.audit.event_store import (
    EventType,
    RunEvent,
    AuditPackage,
    InMemoryEventStore,
    EventFactory,
    ChainIntegrityError,
    GENESIS_HASH,
    HASH_ALGORITHM,
)


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def frozen_time_value():
    """Fixed timestamp for deterministic testing."""
    return datetime(2026, 1, 15, 10, 30, 0, tzinfo=timezone.utc)


@pytest.fixture
def event_store_fresh():
    """Fresh in-memory event store for each test."""
    return InMemoryEventStore()


@pytest.fixture
def event_factory(event_store_fresh):
    """Event factory with fresh store."""
    return EventFactory(event_store_fresh)


@pytest_asyncio.fixture
async def populated_event_store(event_store_fresh, frozen_time_value):
    """
    Event store populated with a complete run lifecycle.

    Events:
    1. RUN_SUBMITTED
    2. PLAN_COMPILED
    3. STEP_STARTED (step-1)
    4. STEP_SUCCEEDED (step-1)
    5. RUN_SUCCEEDED
    """
    freeze_time(frozen_time_value)
    try:
        run_id = "run-populated-001"

        # Event 1: RUN_SUBMITTED
        event1 = RunEvent(
            event_id="evt-001",
            run_id=run_id,
            event_type=EventType.RUN_SUBMITTED,
            timestamp=DeterministicClock.now(timezone.utc),
            payload={"pipeline": "test-pipeline", "tenant": "tenant-001"},
            prev_event_hash=GENESIS_HASH,
        )
        await event_store_fresh.append(event1)

        # Event 2: PLAN_COMPILED
        event2 = RunEvent(
            event_id="evt-002",
            run_id=run_id,
            event_type=EventType.PLAN_COMPILED,
            timestamp=DeterministicClock.now(timezone.utc),
            payload={"plan_id": "plan-abc123", "step_count": 1},
            prev_event_hash=event1.event_hash,
        )
        await event_store_fresh.append(event2)

        # Event 3: STEP_STARTED
        event3 = RunEvent(
            event_id="evt-003",
            run_id=run_id,
            step_id="step-1",
            event_type=EventType.STEP_STARTED,
            timestamp=DeterministicClock.now(timezone.utc),
            payload={"agent": "GL-TEST-X-001"},
            prev_event_hash=event2.event_hash,
        )
        await event_store_fresh.append(event3)

        # Event 4: STEP_SUCCEEDED
        event4 = RunEvent(
            event_id="evt-004",
            run_id=run_id,
            step_id="step-1",
            event_type=EventType.STEP_SUCCEEDED,
            timestamp=DeterministicClock.now(timezone.utc),
            payload={"output_hash": "abc123"},
            prev_event_hash=event3.event_hash,
        )
        await event_store_fresh.append(event4)

        # Event 5: RUN_SUCCEEDED
        event5 = RunEvent(
            event_id="evt-005",
            run_id=run_id,
            event_type=EventType.RUN_SUCCEEDED,
            timestamp=DeterministicClock.now(timezone.utc),
            payload={"duration_ms": 1500},
            prev_event_hash=event4.event_hash,
        )
        await event_store_fresh.append(event5)

        return event_store_fresh, run_id
    finally:
        unfreeze_time()


# =============================================================================
# HASH CHAIN INTEGRITY TESTS
# =============================================================================

class TestHashChainIntegrity:
    """Tests for hash chain integrity verification."""

    @pytest.mark.asyncio
    async def test_first_event_links_to_genesis(
        self,
        event_store_fresh,
        frozen_time_value,
    ):
        """First event in a run has prev_event_hash = 'genesis'."""
        freeze_time(frozen_time_value)
        try:
            run_id = "run-genesis-test"

            event = RunEvent(
                event_id="evt-first",
                run_id=run_id,
                event_type=EventType.RUN_SUBMITTED,
                timestamp=DeterministicClock.now(timezone.utc),
                payload={},
                prev_event_hash=GENESIS_HASH,
            )

            event_hash = await event_store_fresh.append(event)

            events = await event_store_fresh.get_events(run_id)
            assert len(events) == 1
            assert events[0].prev_event_hash == GENESIS_HASH
            assert events[0].event_hash == event_hash
        finally:
            unfreeze_time()

    @pytest.mark.asyncio
    async def test_chain_links_are_correct(self, populated_event_store):
        """Each event links to the previous event's hash."""
        store, run_id = populated_event_store

        events = await store.get_events(run_id)
        assert len(events) == 5

        # First event links to genesis
        assert events[0].prev_event_hash == GENESIS_HASH

        # Each subsequent event links to previous
        for i in range(1, len(events)):
            assert events[i].prev_event_hash == events[i - 1].event_hash

    @pytest.mark.asyncio
    async def test_chain_verification_passes_for_valid_chain(
        self,
        populated_event_store,
    ):
        """Valid chain passes verification."""
        store, run_id = populated_event_store

        is_valid = await store.verify_chain(run_id)
        assert is_valid is True

    @pytest.mark.asyncio
    async def test_chain_verification_fails_for_broken_link(
        self,
        event_store_fresh,
        frozen_time_value,
    ):
        """Verification fails when chain links are broken."""
        freeze_time(frozen_time_value)
        try:
            run_id = "run-broken-chain"

            # First event
            event1 = RunEvent(
                event_id="evt-1",
                run_id=run_id,
                event_type=EventType.RUN_SUBMITTED,
                timestamp=DeterministicClock.now(timezone.utc),
                payload={},
                prev_event_hash=GENESIS_HASH,
            )
            await event_store_fresh.append(event1)

            # Second event with WRONG prev_event_hash
            event2 = RunEvent(
                event_id="evt-2",
                run_id=run_id,
                event_type=EventType.PLAN_COMPILED,
                timestamp=DeterministicClock.now(timezone.utc),
                payload={},
                prev_event_hash="wrong-hash-value",  # Intentionally wrong
            )

            # Should raise ChainIntegrityError
            with pytest.raises(ChainIntegrityError) as exc_info:
                await event_store_fresh.append(event2)

            assert "Invalid prev_event_hash" in str(exc_info.value)
        finally:
            unfreeze_time()

    @pytest.mark.asyncio
    async def test_empty_chain_is_valid(self, event_store_fresh):
        """Empty run (no events) has valid chain."""
        is_valid = await event_store_fresh.verify_chain("run-nonexistent")
        assert is_valid is True

    @pytest.mark.asyncio
    async def test_chain_verification_detects_tampered_hash(
        self,
        populated_event_store,
    ):
        """Tampering with stored hash is detected."""
        store, run_id = populated_event_store

        # Get events and tamper with one
        events = store._events[run_id]
        original_hash = events[2].event_hash
        events[2].event_hash = "tampered_hash_value"

        # Verification should fail
        is_valid = await store.verify_chain(run_id)
        assert is_valid is False

        # Restore for cleanup
        events[2].event_hash = original_hash


# =============================================================================
# EVENT HASH COMPUTATION TESTS
# =============================================================================

class TestEventHashComputation:
    """Tests for deterministic event hash computation."""

    def test_event_hash_is_deterministic(self, frozen_time_value):
        """Same event data produces same hash."""
        freeze_time(frozen_time_value)
        try:
            hashes = []
            for _ in range(100):
                event = RunEvent(
                    event_id="evt-deterministic",
                    run_id="run-hash-test",
                    event_type=EventType.RUN_SUBMITTED,
                    timestamp=DeterministicClock.now(timezone.utc),
                    payload={"key": "value"},
                    prev_event_hash=GENESIS_HASH,
                )
                hashes.append(event.compute_hash())

            assert len(set(hashes)) == 1
        finally:
            unfreeze_time()

    def test_hash_includes_all_required_fields(self, frozen_time_value):
        """Hash changes when any required field changes."""
        freeze_time(frozen_time_value)
        try:
            base_event = RunEvent(
                event_id="evt-base",
                run_id="run-base",
                event_type=EventType.RUN_SUBMITTED,
                timestamp=DeterministicClock.now(timezone.utc),
                payload={"key": "value"},
                prev_event_hash=GENESIS_HASH,
            )
            base_hash = base_event.compute_hash()

            # Change event_id
            event_id_changed = RunEvent(
                event_id="evt-different",  # Changed
                run_id="run-base",
                event_type=EventType.RUN_SUBMITTED,
                timestamp=DeterministicClock.now(timezone.utc),
                payload={"key": "value"},
                prev_event_hash=GENESIS_HASH,
            )
            assert event_id_changed.compute_hash() != base_hash

            # Change event_type
            type_changed = RunEvent(
                event_id="evt-base",
                run_id="run-base",
                event_type=EventType.PLAN_COMPILED,  # Changed
                timestamp=DeterministicClock.now(timezone.utc),
                payload={"key": "value"},
                prev_event_hash=GENESIS_HASH,
            )
            assert type_changed.compute_hash() != base_hash

            # Change payload
            payload_changed = RunEvent(
                event_id="evt-base",
                run_id="run-base",
                event_type=EventType.RUN_SUBMITTED,
                timestamp=DeterministicClock.now(timezone.utc),
                payload={"key": "different"},  # Changed
                prev_event_hash=GENESIS_HASH,
            )
            assert payload_changed.compute_hash() != base_hash

            # Change prev_event_hash
            prev_hash_changed = RunEvent(
                event_id="evt-base",
                run_id="run-base",
                event_type=EventType.RUN_SUBMITTED,
                timestamp=DeterministicClock.now(timezone.utc),
                payload={"key": "value"},
                prev_event_hash="different_prev_hash",  # Changed
            )
            assert prev_hash_changed.compute_hash() != base_hash
        finally:
            unfreeze_time()

    def test_hash_is_sha256(self, frozen_time_value):
        """Event hash is SHA-256 (64 hex chars)."""
        freeze_time(frozen_time_value)
        try:
            event = RunEvent(
                event_id="evt-sha256",
                run_id="run-sha256",
                event_type=EventType.RUN_SUBMITTED,
                timestamp=DeterministicClock.now(timezone.utc),
                payload={},
                prev_event_hash=GENESIS_HASH,
            )
            hash_value = event.compute_hash()

            assert len(hash_value) == 64
            assert all(c in "0123456789abcdef" for c in hash_value)
        finally:
            unfreeze_time()

    def test_verify_hash_returns_true_for_valid(self, frozen_time_value):
        """verify_hash returns True when hash is correct."""
        freeze_time(frozen_time_value)
        try:
            event = RunEvent(
                event_id="evt-verify",
                run_id="run-verify",
                event_type=EventType.RUN_SUBMITTED,
                timestamp=DeterministicClock.now(timezone.utc),
                payload={"test": True},
                prev_event_hash=GENESIS_HASH,
            )
            event.event_hash = event.compute_hash()

            assert event.verify_hash() is True
        finally:
            unfreeze_time()

    def test_verify_hash_returns_false_for_tampered(self, frozen_time_value):
        """verify_hash returns False when hash is tampered."""
        freeze_time(frozen_time_value)
        try:
            event = RunEvent(
                event_id="evt-tamper",
                run_id="run-tamper",
                event_type=EventType.RUN_SUBMITTED,
                timestamp=DeterministicClock.now(timezone.utc),
                payload={"test": True},
                prev_event_hash=GENESIS_HASH,
            )
            event.event_hash = "tampered_hash_value"

            assert event.verify_hash() is False
        finally:
            unfreeze_time()


# =============================================================================
# ARTIFACT CHECKSUM TESTS
# =============================================================================

class TestArtifactChecksums:
    """Tests for artifact checksum computation and verification."""

    @pytest.mark.asyncio
    async def test_artifact_checksum_is_deterministic(self, mock_artifact_store):
        """Same content produces same checksum."""
        content = b"test artifact content"

        # Write same content multiple times
        checksums = []
        for i in range(10):
            metadata = await mock_artifact_store.write_artifact(
                run_id=f"run-checksum-{i}",
                step_id="step-1",
                name="test.txt",
                data=content,
                media_type="text/plain",
                tenant_id="tenant-test",
            )
            checksums.append(metadata.checksum)

        assert len(set(checksums)) == 1

    @pytest.mark.asyncio
    async def test_different_content_different_checksum(self, mock_artifact_store):
        """Different content produces different checksums."""
        checksum1 = (await mock_artifact_store.write_artifact(
            run_id="run-diff-1",
            step_id="step-1",
            name="test.txt",
            data=b"content A",
            media_type="text/plain",
            tenant_id="tenant-test",
        )).checksum

        checksum2 = (await mock_artifact_store.write_artifact(
            run_id="run-diff-2",
            step_id="step-1",
            name="test.txt",
            data=b"content B",
            media_type="text/plain",
            tenant_id="tenant-test",
        )).checksum

        assert checksum1 != checksum2

    @pytest.mark.asyncio
    async def test_verify_checksum_passes_for_valid(self, mock_artifact_store):
        """Checksum verification passes for valid artifacts."""
        metadata = await mock_artifact_store.write_artifact(
            run_id="run-verify",
            step_id="step-1",
            name="valid.txt",
            data=b"valid content",
            media_type="text/plain",
            tenant_id="tenant-test",
        )

        is_valid = await mock_artifact_store.verify_checksum(
            uri=metadata.uri,
            expected_checksum=metadata.checksum,
        )

        assert is_valid is True

    @pytest.mark.asyncio
    async def test_verify_checksum_fails_for_wrong_checksum(
        self,
        mock_artifact_store,
    ):
        """Checksum verification fails for wrong checksum."""
        metadata = await mock_artifact_store.write_artifact(
            run_id="run-wrong-checksum",
            step_id="step-1",
            name="test.txt",
            data=b"some content",
            media_type="text/plain",
            tenant_id="tenant-test",
        )

        is_valid = await mock_artifact_store.verify_checksum(
            uri=metadata.uri,
            expected_checksum="wrong_checksum_value",
        )

        assert is_valid is False

    @pytest.mark.asyncio
    async def test_checksum_is_sha256(self, mock_artifact_store):
        """Artifact checksum is SHA-256 format."""
        metadata = await mock_artifact_store.write_artifact(
            run_id="run-sha256-artifact",
            step_id="step-1",
            name="test.txt",
            data=b"test content",
            media_type="text/plain",
            tenant_id="tenant-test",
        )

        assert len(metadata.checksum) == 64
        assert all(c in "0123456789abcdef" for c in metadata.checksum)


# =============================================================================
# AUDIT PACKAGE TESTS
# =============================================================================

class TestAuditPackage:
    """Tests for audit package export and verification."""

    @pytest.mark.asyncio
    async def test_export_audit_package_includes_all_events(
        self,
        populated_event_store,
    ):
        """Exported audit package contains all events."""
        store, run_id = populated_event_store

        package = await store.export_audit_package(run_id)

        assert len(package.events) == 5
        assert package.run_id == run_id

    @pytest.mark.asyncio
    async def test_audit_package_chain_valid_flag(
        self,
        populated_event_store,
    ):
        """Audit package chain_valid flag is correct."""
        store, run_id = populated_event_store

        package = await store.export_audit_package(run_id)

        assert package.chain_valid is True

    @pytest.mark.asyncio
    async def test_audit_package_contains_metadata(
        self,
        populated_event_store,
    ):
        """Audit package includes expected metadata."""
        store, run_id = populated_event_store

        package = await store.export_audit_package(run_id)

        assert "event_count" in package.metadata
        assert package.metadata["event_count"] == 5
        assert "store_type" in package.metadata
        assert "hash_algorithm" in package.metadata
        assert package.metadata["hash_algorithm"] == HASH_ALGORITHM

    @pytest.mark.asyncio
    async def test_audit_package_hash_is_deterministic(
        self,
        populated_event_store,
    ):
        """Audit package hash is deterministic."""
        store, run_id = populated_event_store

        packages = [
            await store.export_audit_package(run_id)
            for _ in range(10)
        ]

        hashes = [p.compute_package_hash() for p in packages]
        assert len(set(hashes)) == 1

    @pytest.mark.asyncio
    async def test_empty_package_has_valid_hash(self, event_store_fresh):
        """Empty audit package has a defined hash."""
        package = await event_store_fresh.export_audit_package("nonexistent-run")

        assert len(package.events) == 0
        assert package.chain_valid is True

        pkg_hash = package.compute_package_hash()
        assert len(pkg_hash) == 64


# =============================================================================
# EVENT FACTORY TESTS
# =============================================================================

class TestEventFactory:
    """Tests for the EventFactory helper."""

    @pytest.mark.asyncio
    async def test_factory_creates_with_correct_prev_hash(
        self,
        event_store_fresh,
        frozen_time_value,
    ):
        """Factory automatically links to previous event."""
        freeze_time(frozen_time_value)
        try:
            factory = EventFactory(event_store_fresh)
            run_id = "run-factory-test"

            # First event should link to genesis
            event1 = await factory.create_event(
                run_id=run_id,
                event_type=EventType.RUN_SUBMITTED,
                payload={"test": 1},
            )
            assert event1.prev_event_hash == GENESIS_HASH

            # Append it
            await event_store_fresh.append(event1)

            # Second event should link to first
            event2 = await factory.create_event(
                run_id=run_id,
                event_type=EventType.PLAN_COMPILED,
                payload={"test": 2},
            )
            assert event2.prev_event_hash == event1.event_hash
        finally:
            unfreeze_time()

    @pytest.mark.asyncio
    async def test_factory_emit_creates_and_appends(
        self,
        event_store_fresh,
        frozen_time_value,
    ):
        """Factory emit() creates and appends in one call."""
        freeze_time(frozen_time_value)
        try:
            factory = EventFactory(event_store_fresh)
            run_id = "run-emit-test"

            event_hash = await factory.emit(
                run_id=run_id,
                event_type=EventType.RUN_SUBMITTED,
                payload={"emitted": True},
            )

            events = await event_store_fresh.get_events(run_id)
            assert len(events) == 1
            assert events[0].event_hash == event_hash
            assert events[0].payload == {"emitted": True}
        finally:
            unfreeze_time()


# =============================================================================
# CROSS-ENVIRONMENT TESTS
# =============================================================================

class TestHashChainCrossEnvironment:
    """Tests for hash chain consistency across mock environments."""

    @pytest.mark.asyncio
    async def test_hash_chain_with_mock_artifact_store(
        self,
        event_store_fresh,
        mock_artifact_store,
        frozen_time_value,
    ):
        """Hash chain works alongside mock artifact store."""
        freeze_time(frozen_time_value)
        try:
            run_id = "run-cross-env"

            # Write artifact
            artifact = await mock_artifact_store.write_artifact(
                run_id=run_id,
                step_id="step-1",
                name="output.json",
                data=json.dumps({"result": 42}).encode(),
                media_type="application/json",
                tenant_id="tenant-test",
            )

            # Record in event store
            factory = EventFactory(event_store_fresh)

            await factory.emit(
                run_id=run_id,
                event_type=EventType.RUN_SUBMITTED,
                payload={"pipeline": "test"},
            )

            await factory.emit(
                run_id=run_id,
                event_type=EventType.ARTIFACT_WRITTEN,
                step_id="step-1",
                payload={
                    "artifact_uri": artifact.uri,
                    "checksum": artifact.checksum,
                },
            )

            # Verify chain
            is_valid = await event_store_fresh.verify_chain(run_id)
            assert is_valid is True

            # Verify artifact
            is_artifact_valid = await mock_artifact_store.verify_checksum(
                uri=artifact.uri,
                expected_checksum=artifact.checksum,
            )
            assert is_artifact_valid is True
        finally:
            unfreeze_time()

    @pytest.mark.asyncio
    async def test_hash_chain_with_mock_k8s_executor(
        self,
        event_store_fresh,
        mock_k8s_executor,
        mock_artifact_store,
        run_context_factory,
        frozen_time_value,
    ):
        """Hash chain works alongside mock K8s executor."""
        freeze_time(frozen_time_value)
        try:
            run_id = "run-k8s-chain"

            factory = EventFactory(event_store_fresh)

            # Emit run submitted
            await factory.emit(
                run_id=run_id,
                event_type=EventType.RUN_SUBMITTED,
                payload={"executor": "k8s"},
            )

            # Create context and "execute"
            context = run_context_factory(
                run_id=run_id,
                step_id="step-k8s",
            )

            # Pre-populate result for mock executor
            mock_artifact_store.store_mock_result(
                run_id=run_id,
                step_id="step-k8s",
                tenant_id=context.tenant_id,
                result={"outputs": {"value": 100}},
            )

            # Emit step started
            await factory.emit(
                run_id=run_id,
                event_type=EventType.STEP_STARTED,
                step_id="step-k8s",
                payload={"context_hash": context.compute_hash()},
            )

            # Execute via mock
            from greenlang.orchestrator.executors.base import ResourceProfile

            result = await mock_k8s_executor.execute(
                context=context,
                container_image="test-image:latest",
                resources=ResourceProfile(),
                namespace="test",
                input_uri="s3://bucket/input.json",
                output_uri="s3://bucket/output/",
            )

            # Emit step completed
            await factory.emit(
                run_id=run_id,
                event_type=EventType.STEP_SUCCEEDED,
                step_id="step-k8s",
                payload={"status": result.status.value},
            )

            # Verify chain
            is_valid = await event_store_fresh.verify_chain(run_id)
            assert is_valid is True

            events = await event_store_fresh.get_events(run_id)
            assert len(events) == 3
        finally:
            unfreeze_time()


# =============================================================================
# EDGE CASE TESTS
# =============================================================================

class TestHashChainEdgeCases:
    """Edge case tests for hash chain."""

    @pytest.mark.asyncio
    async def test_large_payload_hash(self, event_store_fresh, frozen_time_value):
        """Large payloads are hashed correctly."""
        freeze_time(frozen_time_value)
        try:
            run_id = "run-large-payload"
            large_payload = {
                "data": "x" * 10000,
                "nested": {"array": list(range(1000))},
            }

            event = RunEvent(
                event_id="evt-large",
                run_id=run_id,
                event_type=EventType.STEP_SUCCEEDED,
                timestamp=DeterministicClock.now(timezone.utc),
                payload=large_payload,
                prev_event_hash=GENESIS_HASH,
            )

            event_hash = await event_store_fresh.append(event)

            assert len(event_hash) == 64

            is_valid = await event_store_fresh.verify_chain(run_id)
            assert is_valid is True
        finally:
            unfreeze_time()

    @pytest.mark.asyncio
    async def test_special_characters_in_payload(
        self,
        event_store_fresh,
        frozen_time_value,
    ):
        """Special characters in payload don't break hashing."""
        freeze_time(frozen_time_value)
        try:
            run_id = "run-special-chars"
            special_payload = {
                "unicode": "Hello, world! Queueing algorithms.",
                "quotes": 'He said "Hello"',
                "newlines": "line1\nline2\r\nline3",
                "backslash": "path\\to\\file",
            }

            event = RunEvent(
                event_id="evt-special",
                run_id=run_id,
                event_type=EventType.RUN_SUBMITTED,
                timestamp=DeterministicClock.now(timezone.utc),
                payload=special_payload,
                prev_event_hash=GENESIS_HASH,
            )

            # Should not raise
            event_hash = await event_store_fresh.append(event)
            assert len(event_hash) == 64

            # Verify chain
            is_valid = await event_store_fresh.verify_chain(run_id)
            assert is_valid is True
        finally:
            unfreeze_time()

    @pytest.mark.asyncio
    async def test_empty_payload(self, event_store_fresh, frozen_time_value):
        """Empty payload is handled correctly."""
        freeze_time(frozen_time_value)
        try:
            event = RunEvent(
                event_id="evt-empty",
                run_id="run-empty-payload",
                event_type=EventType.RUN_SUBMITTED,
                timestamp=DeterministicClock.now(timezone.utc),
                payload={},
                prev_event_hash=GENESIS_HASH,
            )

            event_hash = await event_store_fresh.append(event)
            assert len(event_hash) == 64
        finally:
            unfreeze_time()

    @pytest.mark.asyncio
    async def test_concurrent_runs_independent_chains(
        self,
        event_store_fresh,
        frozen_time_value,
    ):
        """Multiple runs have independent hash chains."""
        freeze_time(frozen_time_value)
        try:
            factory = EventFactory(event_store_fresh)

            # Create events for two runs
            await factory.emit("run-A", EventType.RUN_SUBMITTED, {"run": "A"})
            await factory.emit("run-B", EventType.RUN_SUBMITTED, {"run": "B"})
            await factory.emit("run-A", EventType.PLAN_COMPILED, {"plan": "A"})
            await factory.emit("run-B", EventType.PLAN_COMPILED, {"plan": "B"})

            # Both chains should be valid
            assert await event_store_fresh.verify_chain("run-A") is True
            assert await event_store_fresh.verify_chain("run-B") is True

            # Events should be separate
            events_a = await event_store_fresh.get_events("run-A")
            events_b = await event_store_fresh.get_events("run-B")

            assert len(events_a) == 2
            assert len(events_b) == 2

            # Hashes should be different
            assert events_a[0].event_hash != events_b[0].event_hash
        finally:
            unfreeze_time()
