# -*- coding: utf-8 -*-
"""
End-to-end integration tests for Encryption Service (SEC-003)

Tests full encryption workflows including key generation, encryption,
decryption, caching, rotation, and audit trail with mocked external
dependencies (KMS, database).

Coverage targets: Complete workflow coverage
"""

from __future__ import annotations

import asyncio
import json
import secrets
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Attempt to import encryption service modules
# ---------------------------------------------------------------------------
try:
    from greenlang.infrastructure.encryption_service.encryption_service import (
        EncryptionService,
        EncryptionContext,
    )
    from greenlang.infrastructure.encryption_service.key_management import (
        DEKCache,
        KeyManager,
    )
    from greenlang.infrastructure.encryption_service.envelope_encryption import (
        EnvelopeEncryption,
    )
    from greenlang.infrastructure.encryption_service.field_encryption import (
        FieldEncryption,
    )
    _HAS_ENCRYPTION_MODULES = True
except ImportError:
    _HAS_ENCRYPTION_MODULES = False

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(not _HAS_ENCRYPTION_MODULES, reason="Encryption modules not installed"),
]


# ============================================================================
# Helpers and Fixtures
# ============================================================================


@dataclass
class MockKMSClient:
    """Mock AWS KMS client for e2e tests."""
    key_store: Dict[str, bytes] = field(default_factory=dict)

    async def generate_data_key(
        self,
        KeyId: str,
        KeySpec: str = "AES_256",
        EncryptionContext: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        """Generate a data key pair."""
        plaintext = secrets.token_bytes(32)
        # Simulate KMS encryption (just XOR with a fixed key for testing)
        encrypted = self._mock_encrypt(plaintext, EncryptionContext or {})
        return {
            "Plaintext": plaintext,
            "CiphertextBlob": encrypted,
            "KeyId": f"arn:aws:kms:us-east-1:123456789012:key/{KeyId}",
        }

    async def decrypt(
        self,
        CiphertextBlob: bytes,
        EncryptionContext: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        """Decrypt a data key."""
        plaintext = self._mock_decrypt(CiphertextBlob, EncryptionContext or {})
        return {
            "Plaintext": plaintext,
            "KeyId": "arn:aws:kms:us-east-1:123456789012:key/test-key",
        }

    def _mock_encrypt(self, plaintext: bytes, context: Dict[str, str]) -> bytes:
        """Mock KMS encryption."""
        # Store mapping for decryption
        context_hash = self._hash_context(context)
        encrypted = secrets.token_bytes(len(plaintext) + 32)
        self.key_store[encrypted.hex()] = (plaintext, context_hash)
        return encrypted

    def _mock_decrypt(self, ciphertext: bytes, context: Dict[str, str]) -> bytes:
        """Mock KMS decryption."""
        context_hash = self._hash_context(context)
        stored = self.key_store.get(ciphertext.hex())
        if stored is None:
            raise Exception("InvalidCiphertextException")
        plaintext, stored_context = stored
        if stored_context != context_hash:
            raise Exception("InvalidCiphertextException: context mismatch")
        return plaintext

    @staticmethod
    def _hash_context(context: Dict[str, str]) -> str:
        import hashlib
        data = json.dumps(context, sort_keys=True)
        return hashlib.sha256(data.encode()).hexdigest()


@dataclass
class MockAuditStore:
    """Mock audit log store for e2e tests."""
    events: List[Dict[str, Any]] = field(default_factory=list)

    async def log_event(self, event: Dict[str, Any]) -> None:
        self.events.append({
            **event,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event_id": str(uuid.uuid4()),
        })

    def get_events(self, event_type: Optional[str] = None) -> List[Dict[str, Any]]:
        if event_type is None:
            return self.events
        return [e for e in self.events if e.get("event_type") == event_type]

    def clear(self) -> None:
        self.events.clear()


class MockMetricsCollector:
    """Mock metrics collector for e2e tests."""

    def __init__(self):
        self.operations = []
        self.latencies = []
        self.cache_hits = 0
        self.cache_misses = 0
        self.failures = []

    def record_operation(self, operation: str, **kwargs):
        self.operations.append({"operation": operation, **kwargs})

    def record_latency(self, operation: str, latency_ms: float):
        self.latencies.append({"operation": operation, "latency_ms": latency_ms})

    def record_cache_hit(self):
        self.cache_hits += 1

    def record_cache_miss(self):
        self.cache_misses += 1

    def record_failure(self, operation: str, error: str):
        self.failures.append({"operation": operation, "error": error})


@pytest.fixture
def mock_kms():
    """Create a mock KMS client."""
    return MockKMSClient()


@pytest.fixture
def mock_audit():
    """Create a mock audit store."""
    return MockAuditStore()


@pytest.fixture
def mock_metrics():
    """Create a mock metrics collector."""
    return MockMetricsCollector()


# ============================================================================
# Test: Full Encryption Flow
# ============================================================================


class TestFullEncryptionFlow:
    """End-to-end tests for complete encryption workflows."""

    @pytest.mark.asyncio
    async def test_full_encryption_flow(self, mock_kms, mock_audit, mock_metrics):
        """Test complete encryption flow from plaintext to ciphertext."""
        if not _HAS_ENCRYPTION_MODULES:
            pytest.skip("Encryption modules not available")

        # Setup components
        envelope = EnvelopeEncryption(kms_client=mock_kms, key_id="test-key")
        cache = DEKCache(max_size=100, ttl_seconds=3600, metrics=mock_metrics)
        key_manager = KeyManager(envelope=envelope, cache=cache, metrics=mock_metrics)

        class MockAuditLogger:
            async def log_encrypt(self, **kwargs):
                await mock_audit.log_event({"event_type": "encrypt", **kwargs})

            async def log_decrypt(self, **kwargs):
                await mock_audit.log_event({"event_type": "decrypt", **kwargs})

            async def log_failure(self, **kwargs):
                await mock_audit.log_event({"event_type": "failure", **kwargs})

        service = EncryptionService(
            key_manager=key_manager,
            audit_logger=MockAuditLogger(),
            metrics=mock_metrics,
        )

        # Execute encryption
        context = EncryptionContext(
            tenant_id="tenant-e2e",
            user_id="user-e2e",
            data_class="confidential",
        )
        plaintext = b"Sensitive e2e test data"

        encrypted = await service.encrypt(plaintext, context)

        # Verify encrypted data structure
        assert encrypted is not None
        assert encrypted.ciphertext != plaintext
        assert len(encrypted.nonce) == 12
        assert encrypted.key_version is not None

    @pytest.mark.asyncio
    async def test_full_decryption_flow(self, mock_kms, mock_audit, mock_metrics):
        """Test complete decryption flow from ciphertext to plaintext."""
        if not _HAS_ENCRYPTION_MODULES:
            pytest.skip("Encryption modules not available")

        envelope = EnvelopeEncryption(kms_client=mock_kms, key_id="test-key")
        cache = DEKCache(max_size=100, ttl_seconds=3600, metrics=mock_metrics)
        key_manager = KeyManager(envelope=envelope, cache=cache, metrics=mock_metrics)

        class MockAuditLogger:
            async def log_encrypt(self, **kwargs):
                await mock_audit.log_event({"event_type": "encrypt", **kwargs})

            async def log_decrypt(self, **kwargs):
                await mock_audit.log_event({"event_type": "decrypt", **kwargs})

            async def log_failure(self, **kwargs):
                await mock_audit.log_event({"event_type": "failure", **kwargs})

        service = EncryptionService(
            key_manager=key_manager,
            audit_logger=MockAuditLogger(),
            metrics=mock_metrics,
        )

        context = EncryptionContext(
            tenant_id="tenant-e2e",
            user_id="user-e2e",
            data_class="confidential",
        )
        plaintext = b"Sensitive data for decryption test"

        # Encrypt then decrypt
        encrypted = await service.encrypt(plaintext, context)
        decrypted = await service.decrypt(encrypted, context)

        assert decrypted == plaintext


# ============================================================================
# Test: Field Encryption Flow
# ============================================================================


class TestFieldEncryptionFlow:
    """End-to-end tests for field-level encryption."""

    @pytest.mark.asyncio
    async def test_field_encryption_flow(self, mock_kms, mock_audit, mock_metrics):
        """Test field encryption for database storage."""
        if not _HAS_ENCRYPTION_MODULES:
            pytest.skip("Encryption modules not available")

        envelope = EnvelopeEncryption(kms_client=mock_kms, key_id="test-key")
        cache = DEKCache(max_size=100, ttl_seconds=3600, metrics=mock_metrics)
        key_manager = KeyManager(envelope=envelope, cache=cache, metrics=mock_metrics)

        class MockAuditLogger:
            async def log_encrypt(self, **kwargs):
                await mock_audit.log_event({"event_type": "encrypt", **kwargs})

            async def log_decrypt(self, **kwargs):
                await mock_audit.log_event({"event_type": "decrypt", **kwargs})

            async def log_failure(self, **kwargs):
                pass

        encryption_service = EncryptionService(
            key_manager=key_manager,
            audit_logger=MockAuditLogger(),
            metrics=mock_metrics,
        )

        field_enc = FieldEncryption(
            encryption_service=encryption_service,
        )

        # Encrypt a field
        context = FieldContext(
            tenant_id="tenant-field",
            table_name="users",
            field_name="ssn",
            record_id="user-123",
        )

        value = "123-45-6789"
        encrypted = await field_enc.encrypt_field(value, context)

        assert encrypted is not None
        assert encrypted != value

        # Decrypt the field
        decrypted = await field_enc.decrypt_field(encrypted, context)
        assert decrypted == value


# ============================================================================
# Test: Cache Population Flow
# ============================================================================


class TestCachePopulationFlow:
    """End-to-end tests for cache behavior."""

    @pytest.mark.asyncio
    async def test_cache_population_flow(self, mock_kms, mock_audit, mock_metrics):
        """Test that cache is properly populated and used."""
        if not _HAS_ENCRYPTION_MODULES:
            pytest.skip("Encryption modules not available")

        envelope = EnvelopeEncryption(kms_client=mock_kms, key_id="test-key")
        cache = DEKCache(max_size=100, ttl_seconds=3600, metrics=mock_metrics)
        key_manager = KeyManager(envelope=envelope, cache=cache, metrics=mock_metrics)

        class MockAuditLogger:
            async def log_encrypt(self, **kwargs):
                pass

            async def log_decrypt(self, **kwargs):
                pass

            async def log_failure(self, **kwargs):
                pass

        service = EncryptionService(
            key_manager=key_manager,
            audit_logger=MockAuditLogger(),
            metrics=mock_metrics,
        )

        context = EncryptionContext(
            tenant_id="tenant-cache",
            user_id="user-cache",
            data_class="internal",
        )

        # First encryption - cache miss
        await service.encrypt(b"data1", context)
        assert mock_metrics.cache_misses >= 1
        initial_misses = mock_metrics.cache_misses

        # Second encryption - cache hit
        await service.encrypt(b"data2", context)
        assert mock_metrics.cache_hits >= 1


# ============================================================================
# Test: Cache Invalidation Flow
# ============================================================================


class TestCacheInvalidationFlow:
    """End-to-end tests for cache invalidation."""

    @pytest.mark.asyncio
    async def test_cache_invalidation_flow(self, mock_kms, mock_audit, mock_metrics):
        """Test that cache invalidation works correctly."""
        if not _HAS_ENCRYPTION_MODULES:
            pytest.skip("Encryption modules not available")

        envelope = EnvelopeEncryption(kms_client=mock_kms, key_id="test-key")
        cache = DEKCache(max_size=100, ttl_seconds=3600, metrics=mock_metrics)
        key_manager = KeyManager(envelope=envelope, cache=cache, metrics=mock_metrics)

        context = {"tenant_id": "tenant-invalidate", "data_class": "confidential"}

        # Populate cache
        await key_manager.get_or_create_dek(context)
        initial_misses = mock_metrics.cache_misses

        # Invalidate
        cache.invalidate(context)

        # Next access should miss
        await key_manager.get_or_create_dek(context)
        assert mock_metrics.cache_misses > initial_misses


# ============================================================================
# Test: Key Rotation Flow
# ============================================================================


class TestKeyRotationFlow:
    """End-to-end tests for key rotation."""

    @pytest.mark.asyncio
    async def test_key_rotation_flow(self, mock_kms, mock_audit, mock_metrics):
        """Test key rotation invalidates old key and generates new."""
        if not _HAS_ENCRYPTION_MODULES:
            pytest.skip("Encryption modules not available")

        envelope = EnvelopeEncryption(kms_client=mock_kms, key_id="test-key")
        cache = DEKCache(max_size=100, ttl_seconds=3600, metrics=mock_metrics)
        key_manager = KeyManager(envelope=envelope, cache=cache, metrics=mock_metrics)

        context = {"tenant_id": "tenant-rotate", "data_class": "confidential"}

        # Get initial key
        key1, version1, _ = await key_manager.get_or_create_dek(context)

        # Rotate
        await key_manager.rotate_key(context)

        # Get new key
        key2, version2, _ = await key_manager.get_or_create_dek(context)

        # Keys should be different
        assert key1 != key2
        assert version1 != version2


# ============================================================================
# Test: Audit Trail Complete
# ============================================================================


class TestAuditTrailComplete:
    """End-to-end tests for audit trail completeness."""

    @pytest.mark.asyncio
    async def test_audit_trail_complete(self, mock_kms, mock_audit, mock_metrics):
        """Test that all operations are properly audited."""
        if not _HAS_ENCRYPTION_MODULES:
            pytest.skip("Encryption modules not available")

        envelope = EnvelopeEncryption(kms_client=mock_kms, key_id="test-key")
        cache = DEKCache(max_size=100, ttl_seconds=3600, metrics=mock_metrics)
        key_manager = KeyManager(envelope=envelope, cache=cache, metrics=mock_metrics)

        class MockAuditLogger:
            async def log_encrypt(self, **kwargs):
                await mock_audit.log_event({"event_type": "encrypt", **kwargs})

            async def log_decrypt(self, **kwargs):
                await mock_audit.log_event({"event_type": "decrypt", **kwargs})

            async def log_failure(self, **kwargs):
                await mock_audit.log_event({"event_type": "failure", **kwargs})

        service = EncryptionService(
            key_manager=key_manager,
            audit_logger=MockAuditLogger(),
            metrics=mock_metrics,
        )

        context = EncryptionContext(
            tenant_id="tenant-audit",
            user_id="user-audit",
            data_class="confidential",
        )

        # Clear audit store
        mock_audit.clear()

        # Perform operations
        encrypted = await service.encrypt(b"audit test data", context)
        await service.decrypt(encrypted, context)

        # Verify audit events
        encrypt_events = mock_audit.get_events("encrypt")
        decrypt_events = mock_audit.get_events("decrypt")

        assert len(encrypt_events) >= 1
        assert len(decrypt_events) >= 1


# ============================================================================
# Test: Metrics Complete
# ============================================================================


class TestMetricsComplete:
    """End-to-end tests for metrics completeness."""

    @pytest.mark.asyncio
    async def test_metrics_complete(self, mock_kms, mock_audit, mock_metrics):
        """Test that all operations record metrics."""
        if not _HAS_ENCRYPTION_MODULES:
            pytest.skip("Encryption modules not available")

        envelope = EnvelopeEncryption(kms_client=mock_kms, key_id="test-key")
        cache = DEKCache(max_size=100, ttl_seconds=3600, metrics=mock_metrics)
        key_manager = KeyManager(envelope=envelope, cache=cache, metrics=mock_metrics)

        class MockAuditLogger:
            async def log_encrypt(self, **kwargs):
                pass

            async def log_decrypt(self, **kwargs):
                pass

            async def log_failure(self, **kwargs):
                pass

        service = EncryptionService(
            key_manager=key_manager,
            audit_logger=MockAuditLogger(),
            metrics=mock_metrics,
        )

        context = EncryptionContext(
            tenant_id="tenant-metrics",
            user_id="user-metrics",
            data_class="internal",
        )

        # Perform operations
        encrypted = await service.encrypt(b"metrics test", context)
        await service.decrypt(encrypted, context)

        # Verify metrics were recorded
        assert len(mock_metrics.operations) >= 2  # encrypt + decrypt
        assert mock_metrics.cache_misses >= 1 or mock_metrics.cache_hits >= 1


# ============================================================================
# Test: Error Handling Flow
# ============================================================================


class TestErrorHandlingFlow:
    """End-to-end tests for error handling."""

    @pytest.mark.asyncio
    async def test_error_handling_flow(self, mock_kms, mock_audit, mock_metrics):
        """Test that errors are properly handled and logged."""
        if not _HAS_ENCRYPTION_MODULES:
            pytest.skip("Encryption modules not available")

        envelope = EnvelopeEncryption(kms_client=mock_kms, key_id="test-key")
        cache = DEKCache(max_size=100, ttl_seconds=3600, metrics=mock_metrics)
        key_manager = KeyManager(envelope=envelope, cache=cache, metrics=mock_metrics)

        class MockAuditLogger:
            async def log_encrypt(self, **kwargs):
                await mock_audit.log_event({"event_type": "encrypt", **kwargs})

            async def log_decrypt(self, **kwargs):
                await mock_audit.log_event({"event_type": "decrypt", **kwargs})

            async def log_failure(self, **kwargs):
                await mock_audit.log_event({"event_type": "failure", **kwargs})

        service = EncryptionService(
            key_manager=key_manager,
            audit_logger=MockAuditLogger(),
            metrics=mock_metrics,
        )

        context1 = EncryptionContext(
            tenant_id="tenant-error",
            user_id="user-error",
            data_class="confidential",
        )
        context2 = EncryptionContext(
            tenant_id="different-tenant",
            user_id="user-error",
            data_class="confidential",
        )

        # Encrypt with one context
        encrypted = await service.encrypt(b"error test", context1)

        # Try to decrypt with different context (should fail)
        mock_audit.clear()
        with pytest.raises(Exception):
            await service.decrypt(encrypted, context2)

        # Verify failure was logged
        failure_events = mock_audit.get_events("failure")
        assert len(failure_events) >= 1


# ============================================================================
# Test: Concurrent Operations
# ============================================================================


class TestConcurrentOperations:
    """End-to-end tests for concurrent operations."""

    @pytest.mark.asyncio
    async def test_concurrent_operations(self, mock_kms, mock_audit, mock_metrics):
        """Test concurrent encryption/decryption operations."""
        if not _HAS_ENCRYPTION_MODULES:
            pytest.skip("Encryption modules not available")

        envelope = EnvelopeEncryption(kms_client=mock_kms, key_id="test-key")
        cache = DEKCache(max_size=100, ttl_seconds=3600, metrics=mock_metrics)
        key_manager = KeyManager(envelope=envelope, cache=cache, metrics=mock_metrics)

        class MockAuditLogger:
            async def log_encrypt(self, **kwargs):
                pass

            async def log_decrypt(self, **kwargs):
                pass

            async def log_failure(self, **kwargs):
                pass

        service = EncryptionService(
            key_manager=key_manager,
            audit_logger=MockAuditLogger(),
            metrics=mock_metrics,
        )

        async def encrypt_decrypt_roundtrip(i: int):
            context = EncryptionContext(
                tenant_id=f"tenant-{i % 3}",  # 3 different tenants
                user_id=f"user-{i}",
                data_class="internal",
            )
            plaintext = f"concurrent data {i}".encode()
            encrypted = await service.encrypt(plaintext, context)
            decrypted = await service.decrypt(encrypted, context)
            return decrypted == plaintext

        # Run 20 concurrent operations
        results = await asyncio.gather(*[
            encrypt_decrypt_roundtrip(i) for i in range(20)
        ])

        # All should succeed
        assert all(results)
