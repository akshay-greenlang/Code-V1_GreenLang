"""
Tests for Event Envelope Module - GL-001 ThermalCommand

Comprehensive test coverage for EventEnvelope, EnvelopeMetadata,
and SchemaVersion classes.

Author: GreenLang Process Heat Team
Version: 1.0.0
"""

import json
import pytest
from datetime import datetime, timezone, timedelta
from pydantic import ValidationError

from ..event_envelope import (
    EventEnvelope,
    EnvelopeMetadata,
    EnvelopeBatch,
    SchemaVersion,
    SchemaCompatibility,
)


class TestSchemaVersion:
    """Tests for SchemaVersion model."""

    def test_default_version(self) -> None:
        """Test default schema version values."""
        version = SchemaVersion()

        assert version.major == 1
        assert version.minor == 0
        assert version.patch == 0
        assert version.compatibility == SchemaCompatibility.BACKWARD
        assert version.fingerprint is not None
        assert len(version.fingerprint) == 64  # SHA-256 hex length

    def test_version_string(self) -> None:
        """Test version string generation."""
        version = SchemaVersion(major=2, minor=3, patch=4)

        assert version.version_string == "2.3.4"

    def test_backward_compatibility_check(self) -> None:
        """Test backward compatibility check."""
        old_version = SchemaVersion(major=1, minor=0, patch=0)
        new_version = SchemaVersion(
            major=1,
            minor=1,
            patch=0,
            compatibility=SchemaCompatibility.BACKWARD,
        )

        assert new_version.is_compatible_with(old_version)

        # Incompatible: different major version
        breaking_version = SchemaVersion(major=2, minor=0, patch=0)
        assert not breaking_version.is_compatible_with(old_version)

    def test_forward_compatibility_check(self) -> None:
        """Test forward compatibility check."""
        old_version = SchemaVersion(
            major=1,
            minor=1,
            patch=0,
            compatibility=SchemaCompatibility.FORWARD,
        )
        new_version = SchemaVersion(major=1, minor=0, patch=0)

        assert old_version.is_compatible_with(new_version)

    def test_full_compatibility_check(self) -> None:
        """Test full compatibility check."""
        version1 = SchemaVersion(
            major=1,
            minor=0,
            patch=0,
            compatibility=SchemaCompatibility.FULL,
        )
        version2 = SchemaVersion(major=1, minor=5, patch=3)

        assert version1.is_compatible_with(version2)

    def test_no_compatibility_check(self) -> None:
        """Test no compatibility mode (strict matching)."""
        version1 = SchemaVersion(
            major=1,
            minor=0,
            patch=0,
            compatibility=SchemaCompatibility.NONE,
        )
        version2 = SchemaVersion(major=1, minor=0, patch=5)
        version3 = SchemaVersion(major=1, minor=1, patch=0)

        assert version1.is_compatible_with(version2)
        assert not version1.is_compatible_with(version3)


class TestEnvelopeMetadata:
    """Tests for EnvelopeMetadata model."""

    def test_default_metadata(self) -> None:
        """Test default metadata generation."""
        metadata = EnvelopeMetadata(
            source="test-source",
            event_type="gl001.telemetry.normalized",
        )

        assert metadata.envelope_id.startswith("env-")
        assert metadata.correlation_id.startswith("corr-")
        assert metadata.source == "test-source"
        assert metadata.event_type == "gl001.telemetry.normalized"
        assert metadata.priority == 5
        assert metadata.retry_count == 0
        assert metadata.timestamp.tzinfo == timezone.utc

    def test_metadata_with_all_fields(self) -> None:
        """Test metadata with all fields specified."""
        now = datetime.now(timezone.utc)
        metadata = EnvelopeMetadata(
            envelope_id="env-test123",
            correlation_id="corr-test456",
            causation_id="cause-test789",
            timestamp=now,
            source="test-source",
            event_type="gl001.safety.events",
            trace_id="a" * 32,
            span_id="b" * 16,
            partition_key="equipment-001",
            idempotency_key="idem-123",
            priority=1,
            ttl_seconds=3600,
            tags={"env": "test", "version": "1.0"},
        )

        assert metadata.envelope_id == "env-test123"
        assert metadata.correlation_id == "corr-test456"
        assert metadata.causation_id == "cause-test789"
        assert metadata.timestamp == now
        assert metadata.trace_id == "a" * 32
        assert metadata.span_id == "b" * 16
        assert metadata.partition_key == "equipment-001"
        assert metadata.priority == 1
        assert metadata.ttl_seconds == 3600
        assert metadata.tags["env"] == "test"

    def test_event_type_validation(self) -> None:
        """Test event type pattern validation."""
        # Valid event types
        valid_types = [
            "gl001.telemetry.normalized",
            "gl001.plan.dispatch",
            "gl001.safety.events",
        ]

        for event_type in valid_types:
            metadata = EnvelopeMetadata(
                source="test",
                event_type=event_type,
            )
            assert metadata.event_type == event_type

        # Invalid event types
        invalid_types = [
            "gl002.telemetry.normalized",  # Wrong prefix
            "gl001.telemetry",  # Missing third part
            "gl001.Telemetry.Normalized",  # Uppercase
        ]

        for event_type in invalid_types:
            with pytest.raises(ValidationError):
                EnvelopeMetadata(
                    source="test",
                    event_type=event_type,
                )

    def test_timestamp_utc_conversion(self) -> None:
        """Test timestamp conversion to UTC."""
        # Naive datetime should be converted to UTC
        naive_dt = datetime(2024, 1, 15, 12, 0, 0)
        metadata = EnvelopeMetadata(
            source="test",
            event_type="gl001.telemetry.normalized",
            timestamp=naive_dt,
        )
        assert metadata.timestamp.tzinfo == timezone.utc

        # ISO string should be parsed
        metadata2 = EnvelopeMetadata(
            source="test",
            event_type="gl001.telemetry.normalized",
            timestamp="2024-01-15T12:00:00Z",
        )
        assert metadata2.timestamp.tzinfo == timezone.utc

    def test_with_retry(self) -> None:
        """Test retry count increment."""
        metadata = EnvelopeMetadata(
            source="test",
            event_type="gl001.telemetry.normalized",
        )
        assert metadata.retry_count == 0

        retried = metadata.with_retry()
        assert retried.retry_count == 1
        assert metadata.retry_count == 0  # Original unchanged

        retried2 = retried.with_retry()
        assert retried2.retry_count == 2

    def test_is_expired(self) -> None:
        """Test TTL expiration check."""
        # No TTL - never expires
        metadata = EnvelopeMetadata(
            source="test",
            event_type="gl001.telemetry.normalized",
        )
        assert not metadata.is_expired()

        # With TTL - not yet expired
        metadata_with_ttl = EnvelopeMetadata(
            source="test",
            event_type="gl001.telemetry.normalized",
            ttl_seconds=3600,
        )
        assert not metadata_with_ttl.is_expired()

        # Expired metadata
        old_timestamp = datetime.now(timezone.utc) - timedelta(hours=2)
        expired_metadata = EnvelopeMetadata(
            source="test",
            event_type="gl001.telemetry.normalized",
            timestamp=old_timestamp,
            ttl_seconds=3600,
        )
        assert expired_metadata.is_expired()

    def test_priority_bounds(self) -> None:
        """Test priority value bounds."""
        # Valid priorities
        for priority in [1, 5, 10]:
            metadata = EnvelopeMetadata(
                source="test",
                event_type="gl001.telemetry.normalized",
                priority=priority,
            )
            assert metadata.priority == priority

        # Invalid priorities
        with pytest.raises(ValidationError):
            EnvelopeMetadata(
                source="test",
                event_type="gl001.telemetry.normalized",
                priority=0,
            )

        with pytest.raises(ValidationError):
            EnvelopeMetadata(
                source="test",
                event_type="gl001.telemetry.normalized",
                priority=11,
            )


class TestEventEnvelope:
    """Tests for EventEnvelope model."""

    def test_create_envelope(self) -> None:
        """Test envelope creation using factory method."""
        payload = {"sensor_id": "T-101", "value": 450.5}

        envelope = EventEnvelope.create(
            event_type="gl001.telemetry.normalized",
            source="test-source",
            payload=payload,
        )

        assert envelope.metadata.event_type == "gl001.telemetry.normalized"
        assert envelope.metadata.source == "test-source"
        assert envelope.payload == payload
        assert envelope.provenance_hash is not None
        assert len(envelope.provenance_hash) == 64

    def test_create_envelope_with_options(self) -> None:
        """Test envelope creation with all options."""
        payload = {"data": "test"}

        envelope = EventEnvelope.create(
            event_type="gl001.safety.events",
            source="safety-engine",
            payload=payload,
            correlation_id="corr-custom",
            causation_id="cause-custom",
            partition_key="equipment-001",
            priority=1,
            ttl_seconds=1800,
            tags={"critical": "true"},
            trace_id="a" * 32,
            span_id="b" * 16,
        )

        assert envelope.metadata.correlation_id == "corr-custom"
        assert envelope.metadata.causation_id == "cause-custom"
        assert envelope.metadata.partition_key == "equipment-001"
        assert envelope.metadata.priority == 1
        assert envelope.metadata.ttl_seconds == 1800
        assert envelope.metadata.tags["critical"] == "true"
        assert envelope.metadata.trace_id == "a" * 32
        assert envelope.metadata.span_id == "b" * 16

    def test_from_causation(self) -> None:
        """Test envelope creation from causing envelope."""
        original = EventEnvelope.create(
            event_type="gl001.telemetry.normalized",
            source="collector",
            payload={"raw": "data"},
            correlation_id="corr-chain-test",
            trace_id="a" * 32,
        )

        caused = EventEnvelope.from_causation(
            causing_envelope=original,
            event_type="gl001.plan.dispatch",
            source="optimizer",
            payload={"plan": "data"},
        )

        # Should inherit correlation chain
        assert caused.metadata.correlation_id == original.metadata.correlation_id
        assert caused.metadata.causation_id == original.metadata.envelope_id
        assert caused.metadata.trace_id == original.metadata.trace_id
        assert caused.metadata.source == "optimizer"
        assert caused.metadata.event_type == "gl001.plan.dispatch"

    def test_provenance_hash_computation(self) -> None:
        """Test that provenance hash is computed correctly."""
        payload = {"sensor_id": "T-101", "value": 450.5}

        envelope1 = EventEnvelope.create(
            event_type="gl001.telemetry.normalized",
            source="test",
            payload=payload,
        )

        envelope2 = EventEnvelope.create(
            event_type="gl001.telemetry.normalized",
            source="test",
            payload=payload,
        )

        # Different envelopes should have different hashes
        # (different envelope_ids and timestamps)
        assert envelope1.provenance_hash != envelope2.provenance_hash

    def test_verify_provenance(self) -> None:
        """Test provenance verification."""
        envelope = EventEnvelope.create(
            event_type="gl001.telemetry.normalized",
            source="test",
            payload={"data": "test"},
        )

        assert envelope.verify_provenance()

    def test_to_kafka_message(self) -> None:
        """Test conversion to Kafka message format."""
        envelope = EventEnvelope.create(
            event_type="gl001.telemetry.normalized",
            source="test",
            payload={"data": "test"},
            partition_key="test-key",
            trace_id="a" * 32,
        )

        kafka_msg = envelope.to_kafka_message()

        assert "key" in kafka_msg
        assert "value" in kafka_msg
        assert "headers" in kafka_msg
        assert "timestamp_ms" in kafka_msg

        assert kafka_msg["key"] == b"test-key"
        assert isinstance(kafka_msg["value"], bytes)

        # Check headers
        headers_dict = {h[0]: h[1] for h in kafka_msg["headers"]}
        assert b"correlation_id" in [h[0] for h in kafka_msg["headers"]]
        assert b"event_type" in [h[0] for h in kafka_msg["headers"]]
        assert b"trace_id" in [h[0] for h in kafka_msg["headers"]]

    def test_from_kafka_message(self) -> None:
        """Test creation from Kafka message."""
        original = EventEnvelope.create(
            event_type="gl001.telemetry.normalized",
            source="test",
            payload={"data": "test"},
        )

        kafka_msg = original.to_kafka_message()
        restored = EventEnvelope.from_kafka_message(kafka_msg["value"])

        assert restored.metadata.envelope_id == original.metadata.envelope_id
        assert restored.metadata.correlation_id == original.metadata.correlation_id
        assert restored.metadata.event_type == original.metadata.event_type
        assert restored.provenance_hash == original.provenance_hash


class TestEnvelopeBatch:
    """Tests for EnvelopeBatch model."""

    def test_create_empty_batch(self) -> None:
        """Test empty batch creation."""
        batch = EnvelopeBatch()

        assert batch.batch_id.startswith("batch-")
        assert batch.correlation_id.startswith("corr-")
        assert batch.size == 0
        assert batch.envelopes == []

    def test_add_envelope_to_batch(self) -> None:
        """Test adding envelopes to batch."""
        batch = EnvelopeBatch()

        envelope1 = EventEnvelope.create(
            event_type="gl001.telemetry.normalized",
            source="test",
            payload={"data": "1"},
        )
        envelope2 = EventEnvelope.create(
            event_type="gl001.telemetry.normalized",
            source="test",
            payload={"data": "2"},
        )

        batch.add_envelope(envelope1)
        assert batch.size == 1

        batch.add_envelope(envelope2)
        assert batch.size == 2

        # Batch hash should be updated
        assert batch.batch_hash is not None
        assert len(batch.batch_hash) == 64

    def test_batch_hash_changes_with_envelopes(self) -> None:
        """Test that batch hash changes when envelopes are added."""
        batch = EnvelopeBatch()

        envelope1 = EventEnvelope.create(
            event_type="gl001.telemetry.normalized",
            source="test",
            payload={"data": "1"},
        )

        batch.add_envelope(envelope1)
        hash1 = batch.batch_hash

        envelope2 = EventEnvelope.create(
            event_type="gl001.telemetry.normalized",
            source="test",
            payload={"data": "2"},
        )

        batch.add_envelope(envelope2)
        hash2 = batch.batch_hash

        assert hash1 != hash2

    def test_verify_batch(self) -> None:
        """Test batch verification."""
        batch = EnvelopeBatch()

        for i in range(5):
            envelope = EventEnvelope.create(
                event_type="gl001.telemetry.normalized",
                source="test",
                payload={"data": str(i)},
            )
            batch.add_envelope(envelope)

        assert batch.verify_batch()

    def test_batch_with_initial_envelopes(self) -> None:
        """Test batch creation with initial envelopes."""
        envelopes = [
            EventEnvelope.create(
                event_type="gl001.telemetry.normalized",
                source="test",
                payload={"data": str(i)},
            )
            for i in range(3)
        ]

        batch = EnvelopeBatch(
            correlation_id="corr-batch-test",
            envelopes=envelopes,
        )

        assert batch.size == 3
        assert batch.correlation_id == "corr-batch-test"
        assert batch.batch_hash is not None
