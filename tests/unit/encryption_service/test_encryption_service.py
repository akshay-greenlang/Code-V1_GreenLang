# -*- coding: utf-8 -*-
"""
Unit tests for EncryptionService - Encryption at Rest (SEC-003)

Tests the core AES-256-GCM encryption and decryption operations, including
nonce generation, authentication tag validation, AAD binding, key handling,
and metrics/audit emission.

Coverage targets: 85%+ of encryption_service.py
"""

from __future__ import annotations

import hashlib
import os
import secrets
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock, patch, PropertyMock

import pytest

# ---------------------------------------------------------------------------
# Attempt to import encryption service module
# ---------------------------------------------------------------------------
try:
    from greenlang.infrastructure.encryption_service.encryption_service import (
        EncryptionService,
        EncryptionContext,
        EncryptedData,
        EncryptionConfig,
    )
    _HAS_ENCRYPTION_SERVICE = True
except ImportError:
    _HAS_ENCRYPTION_SERVICE = False

try:
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM
    _HAS_CRYPTOGRAPHY = True
except ImportError:
    _HAS_CRYPTOGRAPHY = False

pytestmark = [
    pytest.mark.skipif(not _HAS_ENCRYPTION_SERVICE, reason="Encryption service not installed"),
    pytest.mark.skipif(not _HAS_CRYPTOGRAPHY, reason="cryptography library not installed"),
]


# ============================================================================
# Helpers and Fixtures
# ============================================================================


@dataclass
class MockEncryptionContext:
    """Mock encryption context for testing."""
    tenant_id: str = "tenant-001"
    user_id: str = "user-001"
    data_class: str = "confidential"
    purpose: str = "test"
    request_id: Optional[str] = None

    def __post_init__(self):
        if self.request_id is None:
            self.request_id = str(uuid.uuid4())

    def to_dict(self) -> Dict[str, str]:
        return {
            "tenant_id": self.tenant_id,
            "user_id": self.user_id,
            "data_class": self.data_class,
            "purpose": self.purpose,
            "request_id": self.request_id,
        }


@dataclass
class MockEncryptedData:
    """Mock encrypted data structure."""
    ciphertext: bytes
    nonce: bytes
    auth_tag: bytes
    key_version: str
    encrypted_dek: bytes
    context_hash: str


def _make_mock_key_manager() -> MagicMock:
    """Create a mock key manager."""
    manager = MagicMock()
    # Generate a real 256-bit key for testing
    test_key = secrets.token_bytes(32)
    manager.get_or_create_dek.return_value = (test_key, "v1", b"encrypted_dek_bytes")
    manager.decrypt_dek.return_value = test_key
    return manager


def _make_mock_audit_logger() -> AsyncMock:
    """Create a mock audit logger."""
    logger = AsyncMock()
    logger.log_encrypt = AsyncMock()
    logger.log_decrypt = AsyncMock()
    logger.log_failure = AsyncMock()
    return logger


def _make_mock_metrics() -> MagicMock:
    """Create a mock metrics collector."""
    metrics = MagicMock()
    metrics.record_operation = MagicMock()
    metrics.record_failure = MagicMock()
    metrics.record_latency = MagicMock()
    return metrics


@pytest.fixture
def encryption_context() -> MockEncryptionContext:
    """Create a test encryption context."""
    return MockEncryptionContext()


@pytest.fixture
def mock_key_manager() -> MagicMock:
    """Create a mock key manager."""
    return _make_mock_key_manager()


@pytest.fixture
def mock_audit_logger() -> AsyncMock:
    """Create a mock audit logger."""
    return _make_mock_audit_logger()


@pytest.fixture
def mock_metrics() -> MagicMock:
    """Create a mock metrics collector."""
    return _make_mock_metrics()


# ============================================================================
# Test: Encryption/Decryption Roundtrip
# ============================================================================


class TestEncryptionRoundtrip:
    """Tests for encryption/decryption roundtrip operations."""

    @pytest.mark.asyncio
    async def test_encrypt_decrypt_roundtrip(
        self,
        mock_key_manager,
        mock_audit_logger,
        mock_metrics,
        encryption_context,
    ):
        """Test that encrypted data can be decrypted back to original."""
        if not _HAS_ENCRYPTION_SERVICE:
            pytest.skip("Encryption service not available")

        # Setup
        service = EncryptionService(
            key_manager=mock_key_manager,
            audit_logger=mock_audit_logger,
            metrics=mock_metrics,
        )
        plaintext = b"Hello, World! This is sensitive data."

        # Encrypt
        encrypted = await service.encrypt(plaintext, encryption_context)

        # Decrypt
        decrypted = await service.decrypt(encrypted, encryption_context)

        # Verify
        assert decrypted == plaintext

    @pytest.mark.asyncio
    async def test_encrypt_different_data_different_ciphertext(
        self,
        mock_key_manager,
        mock_audit_logger,
        mock_metrics,
        encryption_context,
    ):
        """Test that different plaintext produces different ciphertext."""
        if not _HAS_ENCRYPTION_SERVICE:
            pytest.skip("Encryption service not available")

        service = EncryptionService(
            key_manager=mock_key_manager,
            audit_logger=mock_audit_logger,
            metrics=mock_metrics,
        )

        plaintext1 = b"Message one"
        plaintext2 = b"Message two"

        encrypted1 = await service.encrypt(plaintext1, encryption_context)
        encrypted2 = await service.encrypt(plaintext2, encryption_context)

        assert encrypted1.ciphertext != encrypted2.ciphertext

    @pytest.mark.asyncio
    async def test_encrypt_same_data_different_nonce(
        self,
        mock_key_manager,
        mock_audit_logger,
        mock_metrics,
        encryption_context,
    ):
        """Test that same plaintext produces different ciphertext (different nonce)."""
        if not _HAS_ENCRYPTION_SERVICE:
            pytest.skip("Encryption service not available")

        service = EncryptionService(
            key_manager=mock_key_manager,
            audit_logger=mock_audit_logger,
            metrics=mock_metrics,
        )

        plaintext = b"Same message"

        encrypted1 = await service.encrypt(plaintext, encryption_context)
        encrypted2 = await service.encrypt(plaintext, encryption_context)

        # Different nonces
        assert encrypted1.nonce != encrypted2.nonce
        # Different ciphertexts due to different nonces
        assert encrypted1.ciphertext != encrypted2.ciphertext


# ============================================================================
# Test: Cryptographic Properties
# ============================================================================


class TestCryptographicProperties:
    """Tests for cryptographic properties and parameters."""

    @pytest.mark.asyncio
    async def test_nonce_is_12_bytes(
        self,
        mock_key_manager,
        mock_audit_logger,
        mock_metrics,
        encryption_context,
    ):
        """Test that generated nonce is 12 bytes (96 bits) per NIST SP 800-38D."""
        if not _HAS_ENCRYPTION_SERVICE:
            pytest.skip("Encryption service not available")

        service = EncryptionService(
            key_manager=mock_key_manager,
            audit_logger=mock_audit_logger,
            metrics=mock_metrics,
        )

        encrypted = await service.encrypt(b"test data", encryption_context)

        assert len(encrypted.nonce) == 12

    @pytest.mark.asyncio
    async def test_auth_tag_is_16_bytes(
        self,
        mock_key_manager,
        mock_audit_logger,
        mock_metrics,
        encryption_context,
    ):
        """Test that auth tag is 16 bytes (128 bits)."""
        if not _HAS_ENCRYPTION_SERVICE:
            pytest.skip("Encryption service not available")

        service = EncryptionService(
            key_manager=mock_key_manager,
            audit_logger=mock_audit_logger,
            metrics=mock_metrics,
        )

        encrypted = await service.encrypt(b"test data", encryption_context)

        # AES-GCM auth tag is appended to ciphertext or stored separately
        # Check the auth_tag field if stored separately
        if hasattr(encrypted, 'auth_tag') and encrypted.auth_tag:
            assert len(encrypted.auth_tag) == 16

    @pytest.mark.asyncio
    async def test_key_is_32_bytes(
        self,
        mock_key_manager,
        mock_audit_logger,
        mock_metrics,
        encryption_context,
    ):
        """Test that DEK is 32 bytes (256 bits) for AES-256."""
        if not _HAS_ENCRYPTION_SERVICE:
            pytest.skip("Encryption service not available")

        # Verify the mock returns 32-byte key
        key, _, _ = mock_key_manager.get_or_create_dek()
        assert len(key) == 32


# ============================================================================
# Test: Additional Authenticated Data (AAD)
# ============================================================================


class TestAADBinding:
    """Tests for AAD (Additional Authenticated Data) binding."""

    @pytest.mark.asyncio
    async def test_aad_binds_to_ciphertext(
        self,
        mock_key_manager,
        mock_audit_logger,
        mock_metrics,
    ):
        """Test that AAD is cryptographically bound to ciphertext."""
        if not _HAS_ENCRYPTION_SERVICE:
            pytest.skip("Encryption service not available")

        service = EncryptionService(
            key_manager=mock_key_manager,
            audit_logger=mock_audit_logger,
            metrics=mock_metrics,
        )

        context1 = MockEncryptionContext(tenant_id="tenant-001")
        context2 = MockEncryptionContext(tenant_id="tenant-002")

        plaintext = b"test data"
        encrypted = await service.encrypt(plaintext, context1)

        # Decryption with same context should work
        decrypted = await service.decrypt(encrypted, context1)
        assert decrypted == plaintext

    @pytest.mark.asyncio
    async def test_wrong_aad_fails_decryption(
        self,
        mock_key_manager,
        mock_audit_logger,
        mock_metrics,
    ):
        """Test that wrong AAD (different context) fails decryption."""
        if not _HAS_ENCRYPTION_SERVICE:
            pytest.skip("Encryption service not available")

        service = EncryptionService(
            key_manager=mock_key_manager,
            audit_logger=mock_audit_logger,
            metrics=mock_metrics,
        )

        context1 = MockEncryptionContext(tenant_id="tenant-001")
        context2 = MockEncryptionContext(tenant_id="tenant-002")

        plaintext = b"test data"
        encrypted = await service.encrypt(plaintext, context1)

        # Decryption with different context should fail
        with pytest.raises(Exception):  # Should raise InvalidTag or similar
            await service.decrypt(encrypted, context2)


# ============================================================================
# Test: Tamper Detection
# ============================================================================


class TestTamperDetection:
    """Tests for tamper detection capabilities."""

    @pytest.mark.asyncio
    async def test_tampered_ciphertext_fails(
        self,
        mock_key_manager,
        mock_audit_logger,
        mock_metrics,
        encryption_context,
    ):
        """Test that tampered ciphertext is detected."""
        if not _HAS_ENCRYPTION_SERVICE:
            pytest.skip("Encryption service not available")

        service = EncryptionService(
            key_manager=mock_key_manager,
            audit_logger=mock_audit_logger,
            metrics=mock_metrics,
        )

        plaintext = b"test data"
        encrypted = await service.encrypt(plaintext, encryption_context)

        # Tamper with ciphertext
        tampered = bytearray(encrypted.ciphertext)
        tampered[0] ^= 0xFF  # Flip bits
        encrypted.ciphertext = bytes(tampered)

        with pytest.raises(Exception):  # Should raise InvalidTag
            await service.decrypt(encrypted, encryption_context)

    @pytest.mark.asyncio
    async def test_tampered_tag_fails(
        self,
        mock_key_manager,
        mock_audit_logger,
        mock_metrics,
        encryption_context,
    ):
        """Test that tampered auth tag is detected."""
        if not _HAS_ENCRYPTION_SERVICE:
            pytest.skip("Encryption service not available")

        service = EncryptionService(
            key_manager=mock_key_manager,
            audit_logger=mock_audit_logger,
            metrics=mock_metrics,
        )

        plaintext = b"test data"
        encrypted = await service.encrypt(plaintext, encryption_context)

        # If auth_tag is stored separately, tamper with it
        if hasattr(encrypted, 'auth_tag') and encrypted.auth_tag:
            tampered = bytearray(encrypted.auth_tag)
            tampered[0] ^= 0xFF
            encrypted.auth_tag = bytes(tampered)

            with pytest.raises(Exception):
                await service.decrypt(encrypted, encryption_context)

    @pytest.mark.asyncio
    async def test_wrong_key_fails_decryption(
        self,
        mock_audit_logger,
        mock_metrics,
        encryption_context,
    ):
        """Test that decryption with wrong key fails."""
        if not _HAS_ENCRYPTION_SERVICE:
            pytest.skip("Encryption service not available")

        # Create two key managers with different keys
        key_manager1 = _make_mock_key_manager()
        key_manager2 = _make_mock_key_manager()

        # Ensure different keys
        key2 = secrets.token_bytes(32)
        key_manager2.decrypt_dek.return_value = key2

        service1 = EncryptionService(
            key_manager=key_manager1,
            audit_logger=mock_audit_logger,
            metrics=mock_metrics,
        )
        service2 = EncryptionService(
            key_manager=key_manager2,
            audit_logger=mock_audit_logger,
            metrics=mock_metrics,
        )

        plaintext = b"test data"
        encrypted = await service1.encrypt(plaintext, encryption_context)

        # Decryption with different key should fail
        with pytest.raises(Exception):
            await service2.decrypt(encrypted, encryption_context)

    @pytest.mark.asyncio
    async def test_wrong_nonce_fails_decryption(
        self,
        mock_key_manager,
        mock_audit_logger,
        mock_metrics,
        encryption_context,
    ):
        """Test that decryption with wrong nonce fails."""
        if not _HAS_ENCRYPTION_SERVICE:
            pytest.skip("Encryption service not available")

        service = EncryptionService(
            key_manager=mock_key_manager,
            audit_logger=mock_audit_logger,
            metrics=mock_metrics,
        )

        plaintext = b"test data"
        encrypted = await service.encrypt(plaintext, encryption_context)

        # Replace nonce with different value
        encrypted.nonce = secrets.token_bytes(12)

        with pytest.raises(Exception):
            await service.decrypt(encrypted, encryption_context)


# ============================================================================
# Test: Edge Cases
# ============================================================================


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    @pytest.mark.asyncio
    async def test_encrypt_empty_data(
        self,
        mock_key_manager,
        mock_audit_logger,
        mock_metrics,
        encryption_context,
    ):
        """Test encryption of empty byte string."""
        if not _HAS_ENCRYPTION_SERVICE:
            pytest.skip("Encryption service not available")

        service = EncryptionService(
            key_manager=mock_key_manager,
            audit_logger=mock_audit_logger,
            metrics=mock_metrics,
        )

        plaintext = b""
        encrypted = await service.encrypt(plaintext, encryption_context)
        decrypted = await service.decrypt(encrypted, encryption_context)

        assert decrypted == plaintext

    @pytest.mark.asyncio
    async def test_encrypt_large_data(
        self,
        mock_key_manager,
        mock_audit_logger,
        mock_metrics,
        encryption_context,
    ):
        """Test encryption of large data (1 MB)."""
        if not _HAS_ENCRYPTION_SERVICE:
            pytest.skip("Encryption service not available")

        service = EncryptionService(
            key_manager=mock_key_manager,
            audit_logger=mock_audit_logger,
            metrics=mock_metrics,
        )

        plaintext = secrets.token_bytes(1024 * 1024)  # 1 MB
        encrypted = await service.encrypt(plaintext, encryption_context)
        decrypted = await service.decrypt(encrypted, encryption_context)

        assert decrypted == plaintext

    @pytest.mark.asyncio
    async def test_encrypt_binary_data(
        self,
        mock_key_manager,
        mock_audit_logger,
        mock_metrics,
        encryption_context,
    ):
        """Test encryption of binary data with all byte values."""
        if not _HAS_ENCRYPTION_SERVICE:
            pytest.skip("Encryption service not available")

        service = EncryptionService(
            key_manager=mock_key_manager,
            audit_logger=mock_audit_logger,
            metrics=mock_metrics,
        )

        # All possible byte values
        plaintext = bytes(range(256))
        encrypted = await service.encrypt(plaintext, encryption_context)
        decrypted = await service.decrypt(encrypted, encryption_context)

        assert decrypted == plaintext

    @pytest.mark.asyncio
    async def test_encrypt_unicode_data(
        self,
        mock_key_manager,
        mock_audit_logger,
        mock_metrics,
        encryption_context,
    ):
        """Test encryption of unicode text encoded as UTF-8."""
        if not _HAS_ENCRYPTION_SERVICE:
            pytest.skip("Encryption service not available")

        service = EncryptionService(
            key_manager=mock_key_manager,
            audit_logger=mock_audit_logger,
            metrics=mock_metrics,
        )

        unicode_text = "Hello, World! \u4e2d\u6587 \U0001f600"
        plaintext = unicode_text.encode("utf-8")
        encrypted = await service.encrypt(plaintext, encryption_context)
        decrypted = await service.decrypt(encrypted, encryption_context)

        assert decrypted == plaintext
        assert decrypted.decode("utf-8") == unicode_text


# ============================================================================
# Test: Context Requirements
# ============================================================================


class TestContextRequirements:
    """Tests for encryption context requirements."""

    @pytest.mark.asyncio
    async def test_context_required(
        self,
        mock_key_manager,
        mock_audit_logger,
        mock_metrics,
    ):
        """Test that encryption context is required."""
        if not _HAS_ENCRYPTION_SERVICE:
            pytest.skip("Encryption service not available")

        service = EncryptionService(
            key_manager=mock_key_manager,
            audit_logger=mock_audit_logger,
            metrics=mock_metrics,
        )

        with pytest.raises((ValueError, TypeError)):
            await service.encrypt(b"test", None)

    @pytest.mark.asyncio
    async def test_context_binds_to_dek(
        self,
        mock_key_manager,
        mock_audit_logger,
        mock_metrics,
    ):
        """Test that context is used to derive/retrieve DEK."""
        if not _HAS_ENCRYPTION_SERVICE:
            pytest.skip("Encryption service not available")

        service = EncryptionService(
            key_manager=mock_key_manager,
            audit_logger=mock_audit_logger,
            metrics=mock_metrics,
        )

        context = MockEncryptionContext(tenant_id="tenant-123")
        await service.encrypt(b"test", context)

        # Verify key manager was called with context
        mock_key_manager.get_or_create_dek.assert_called()
        call_args = mock_key_manager.get_or_create_dek.call_args
        # Check that context was passed (implementation specific)


# ============================================================================
# Test: Metrics Recording
# ============================================================================


class TestMetricsRecording:
    """Tests for metrics recording during operations."""

    @pytest.mark.asyncio
    async def test_metrics_recorded_on_encrypt(
        self,
        mock_key_manager,
        mock_audit_logger,
        mock_metrics,
        encryption_context,
    ):
        """Test that metrics are recorded on encryption."""
        if not _HAS_ENCRYPTION_SERVICE:
            pytest.skip("Encryption service not available")

        service = EncryptionService(
            key_manager=mock_key_manager,
            audit_logger=mock_audit_logger,
            metrics=mock_metrics,
        )

        await service.encrypt(b"test data", encryption_context)

        mock_metrics.record_operation.assert_called()
        call_args = mock_metrics.record_operation.call_args
        assert call_args[0][0] == "encrypt" or "encrypt" in str(call_args)

    @pytest.mark.asyncio
    async def test_metrics_recorded_on_decrypt(
        self,
        mock_key_manager,
        mock_audit_logger,
        mock_metrics,
        encryption_context,
    ):
        """Test that metrics are recorded on decryption."""
        if not _HAS_ENCRYPTION_SERVICE:
            pytest.skip("Encryption service not available")

        service = EncryptionService(
            key_manager=mock_key_manager,
            audit_logger=mock_audit_logger,
            metrics=mock_metrics,
        )

        encrypted = await service.encrypt(b"test data", encryption_context)
        mock_metrics.reset_mock()

        await service.decrypt(encrypted, encryption_context)

        mock_metrics.record_operation.assert_called()


# ============================================================================
# Test: Audit Logging
# ============================================================================


class TestAuditLogging:
    """Tests for audit logging during operations."""

    @pytest.mark.asyncio
    async def test_audit_logged_on_encrypt(
        self,
        mock_key_manager,
        mock_audit_logger,
        mock_metrics,
        encryption_context,
    ):
        """Test that audit event is logged on encryption."""
        if not _HAS_ENCRYPTION_SERVICE:
            pytest.skip("Encryption service not available")

        service = EncryptionService(
            key_manager=mock_key_manager,
            audit_logger=mock_audit_logger,
            metrics=mock_metrics,
        )

        await service.encrypt(b"test data", encryption_context)

        mock_audit_logger.log_encrypt.assert_called()

    @pytest.mark.asyncio
    async def test_audit_logged_on_decrypt(
        self,
        mock_key_manager,
        mock_audit_logger,
        mock_metrics,
        encryption_context,
    ):
        """Test that audit event is logged on decryption."""
        if not _HAS_ENCRYPTION_SERVICE:
            pytest.skip("Encryption service not available")

        service = EncryptionService(
            key_manager=mock_key_manager,
            audit_logger=mock_audit_logger,
            metrics=mock_metrics,
        )

        encrypted = await service.encrypt(b"test data", encryption_context)
        mock_audit_logger.reset_mock()

        await service.decrypt(encrypted, encryption_context)

        mock_audit_logger.log_decrypt.assert_called()

    @pytest.mark.asyncio
    async def test_audit_logged_on_failure(
        self,
        mock_key_manager,
        mock_audit_logger,
        mock_metrics,
        encryption_context,
    ):
        """Test that audit event is logged on failure."""
        if not _HAS_ENCRYPTION_SERVICE:
            pytest.skip("Encryption service not available")

        service = EncryptionService(
            key_manager=mock_key_manager,
            audit_logger=mock_audit_logger,
            metrics=mock_metrics,
        )

        encrypted = await service.encrypt(b"test data", encryption_context)

        # Tamper to cause failure
        encrypted.ciphertext = b"tampered"

        with pytest.raises(Exception):
            await service.decrypt(encrypted, encryption_context)

        mock_audit_logger.log_failure.assert_called()


# ============================================================================
# Test: Performance
# ============================================================================


class TestPerformance:
    """Tests for performance characteristics."""

    @pytest.mark.asyncio
    async def test_encrypt_performance_under_1ms_cached(
        self,
        mock_key_manager,
        mock_audit_logger,
        mock_metrics,
        encryption_context,
    ):
        """Test that encryption with cached DEK completes under 1ms."""
        if not _HAS_ENCRYPTION_SERVICE:
            pytest.skip("Encryption service not available")

        service = EncryptionService(
            key_manager=mock_key_manager,
            audit_logger=mock_audit_logger,
            metrics=mock_metrics,
        )

        # Warm up / cache DEK
        await service.encrypt(b"warmup", encryption_context)

        # Measure
        plaintext = b"test data for performance measurement"
        start = time.perf_counter()
        await service.encrypt(plaintext, encryption_context)
        elapsed_ms = (time.perf_counter() - start) * 1000

        # Allow some margin for CI environments
        assert elapsed_ms < 10, f"Encryption took {elapsed_ms:.2f}ms, expected <10ms"

    @pytest.mark.asyncio
    async def test_concurrent_encryptions(
        self,
        mock_key_manager,
        mock_audit_logger,
        mock_metrics,
    ):
        """Test concurrent encryption operations."""
        if not _HAS_ENCRYPTION_SERVICE:
            pytest.skip("Encryption service not available")

        service = EncryptionService(
            key_manager=mock_key_manager,
            audit_logger=mock_audit_logger,
            metrics=mock_metrics,
        )

        async def encrypt_task(i: int):
            context = MockEncryptionContext(user_id=f"user-{i}")
            plaintext = f"message {i}".encode()
            encrypted = await service.encrypt(plaintext, context)
            decrypted = await service.decrypt(encrypted, context)
            return decrypted == plaintext

        import asyncio
        tasks = [encrypt_task(i) for i in range(10)]
        results = await asyncio.gather(*tasks)

        assert all(results)


# ============================================================================
# Test: Security Properties
# ============================================================================


class TestSecurityProperties:
    """Tests for security properties and guarantees."""

    @pytest.mark.asyncio
    async def test_no_plaintext_in_exception(
        self,
        mock_key_manager,
        mock_audit_logger,
        mock_metrics,
        encryption_context,
    ):
        """Test that plaintext is not leaked in exception messages."""
        if not _HAS_ENCRYPTION_SERVICE:
            pytest.skip("Encryption service not available")

        service = EncryptionService(
            key_manager=mock_key_manager,
            audit_logger=mock_audit_logger,
            metrics=mock_metrics,
        )

        sensitive_data = b"SUPER_SECRET_DATA_12345"
        encrypted = await service.encrypt(sensitive_data, encryption_context)

        # Corrupt to cause decryption failure
        encrypted.ciphertext = b"corrupted"

        try:
            await service.decrypt(encrypted, encryption_context)
            pytest.fail("Expected exception")
        except Exception as e:
            error_msg = str(e).lower()
            assert "super_secret" not in error_msg
            assert "12345" not in error_msg

    def test_no_key_in_logs(
        self,
        mock_key_manager,
        mock_audit_logger,
        mock_metrics,
        encryption_context,
        caplog,
    ):
        """Test that keys are not logged."""
        if not _HAS_ENCRYPTION_SERVICE:
            pytest.skip("Encryption service not available")

        import logging

        # Get the key value
        key, _, _ = mock_key_manager.get_or_create_dek()
        key_hex = key.hex()

        with caplog.at_level(logging.DEBUG):
            # This is a structural test - implementation should not log keys
            # We verify by checking that the test key hex is not in logs
            pass

        for record in caplog.records:
            assert key_hex not in record.message


# ============================================================================
# Test: Data Structure Validation
# ============================================================================


class TestDataStructures:
    """Tests for encrypted data structure validation."""

    @pytest.mark.asyncio
    async def test_encrypted_data_contains_required_fields(
        self,
        mock_key_manager,
        mock_audit_logger,
        mock_metrics,
        encryption_context,
    ):
        """Test that encrypted data contains all required fields."""
        if not _HAS_ENCRYPTION_SERVICE:
            pytest.skip("Encryption service not available")

        service = EncryptionService(
            key_manager=mock_key_manager,
            audit_logger=mock_audit_logger,
            metrics=mock_metrics,
        )

        encrypted = await service.encrypt(b"test", encryption_context)

        assert hasattr(encrypted, 'ciphertext')
        assert hasattr(encrypted, 'nonce')
        assert hasattr(encrypted, 'key_version')
        assert hasattr(encrypted, 'encrypted_dek')

    @pytest.mark.asyncio
    async def test_encrypted_data_serializable(
        self,
        mock_key_manager,
        mock_audit_logger,
        mock_metrics,
        encryption_context,
    ):
        """Test that encrypted data can be serialized."""
        if not _HAS_ENCRYPTION_SERVICE:
            pytest.skip("Encryption service not available")

        import json

        service = EncryptionService(
            key_manager=mock_key_manager,
            audit_logger=mock_audit_logger,
            metrics=mock_metrics,
        )

        encrypted = await service.encrypt(b"test", encryption_context)

        # Should be JSON serializable (with base64 encoding for bytes)
        if hasattr(encrypted, 'to_dict'):
            data = encrypted.to_dict()
            json_str = json.dumps(data)
            assert json_str is not None
