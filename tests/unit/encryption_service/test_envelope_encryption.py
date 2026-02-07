# -*- coding: utf-8 -*-
"""
Unit tests for Envelope Encryption - Encryption at Rest (SEC-003)

Tests the KMS envelope encryption pattern, including data key generation,
DEK encryption/decryption via KMS, encryption context binding, and proper
error handling for KMS operations.

Coverage targets: 85%+ of envelope_encryption.py
"""

from __future__ import annotations

import secrets
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock, patch, PropertyMock

import pytest

# ---------------------------------------------------------------------------
# Attempt to import envelope encryption module
# ---------------------------------------------------------------------------
try:
    from greenlang.infrastructure.encryption_service.envelope_encryption import (
        EnvelopeEncryption,
        KMSClient,
        DataKeyPair,
        EnvelopeEncryptionConfig,
    )
    _HAS_ENVELOPE_ENCRYPTION = True
except ImportError:
    _HAS_ENVELOPE_ENCRYPTION = False

pytestmark = [
    pytest.mark.skipif(not _HAS_ENVELOPE_ENCRYPTION, reason="Envelope encryption not installed"),
]


# ============================================================================
# Helpers and Fixtures
# ============================================================================


@dataclass
class MockEncryptionContext:
    """Mock encryption context for KMS operations."""
    tenant_id: str = "tenant-001"
    data_class: str = "confidential"
    purpose: str = "data_encryption"

    def to_dict(self) -> Dict[str, str]:
        return {
            "tenant_id": self.tenant_id,
            "data_class": self.data_class,
            "purpose": self.purpose,
        }


def _make_mock_kms_client() -> MagicMock:
    """Create a mock KMS client."""
    client = MagicMock()

    # Generate test key pair
    plaintext_key = secrets.token_bytes(32)
    encrypted_key = secrets.token_bytes(64)  # Simulated encrypted DEK

    # Mock generate_data_key
    client.generate_data_key = AsyncMock(return_value={
        "Plaintext": plaintext_key,
        "CiphertextBlob": encrypted_key,
        "KeyId": "arn:aws:kms:us-east-1:123456789012:key/test-key-id",
    })

    # Mock decrypt (for decrypting DEK)
    client.decrypt = AsyncMock(return_value={
        "Plaintext": plaintext_key,
        "KeyId": "arn:aws:kms:us-east-1:123456789012:key/test-key-id",
    })

    # Store for verification
    client._test_plaintext_key = plaintext_key
    client._test_encrypted_key = encrypted_key

    return client


def _make_mock_metrics() -> MagicMock:
    """Create a mock metrics collector."""
    metrics = MagicMock()
    metrics.record_kms_call = MagicMock()
    metrics.record_kms_latency = MagicMock()
    metrics.record_kms_error = MagicMock()
    return metrics


@pytest.fixture
def mock_kms_client() -> MagicMock:
    """Create a mock KMS client."""
    return _make_mock_kms_client()


@pytest.fixture
def mock_metrics() -> MagicMock:
    """Create a mock metrics collector."""
    return _make_mock_metrics()


@pytest.fixture
def encryption_context() -> MockEncryptionContext:
    """Create a test encryption context."""
    return MockEncryptionContext()


# ============================================================================
# Test: Data Key Generation
# ============================================================================


class TestDataKeyGeneration:
    """Tests for data key generation via KMS."""

    @pytest.mark.asyncio
    async def test_generate_data_key_returns_both_keys(
        self,
        mock_kms_client,
        mock_metrics,
        encryption_context,
    ):
        """Test that generate_data_key returns both plaintext and encrypted keys."""
        if not _HAS_ENVELOPE_ENCRYPTION:
            pytest.skip("Envelope encryption not available")

        envelope = EnvelopeEncryption(
            kms_client=mock_kms_client,
            key_id="test-key-id",
            metrics=mock_metrics,
        )

        key_pair = await envelope.generate_data_key(encryption_context.to_dict())

        assert key_pair.plaintext_key is not None
        assert key_pair.encrypted_key is not None

    @pytest.mark.asyncio
    async def test_plaintext_key_is_32_bytes(
        self,
        mock_kms_client,
        mock_metrics,
        encryption_context,
    ):
        """Test that plaintext DEK is 32 bytes (256 bits)."""
        if not _HAS_ENVELOPE_ENCRYPTION:
            pytest.skip("Envelope encryption not available")

        envelope = EnvelopeEncryption(
            kms_client=mock_kms_client,
            key_id="test-key-id",
            metrics=mock_metrics,
        )

        key_pair = await envelope.generate_data_key(encryption_context.to_dict())

        assert len(key_pair.plaintext_key) == 32

    @pytest.mark.asyncio
    async def test_encrypted_key_is_different(
        self,
        mock_kms_client,
        mock_metrics,
        encryption_context,
    ):
        """Test that encrypted key is different from plaintext key."""
        if not _HAS_ENVELOPE_ENCRYPTION:
            pytest.skip("Envelope encryption not available")

        envelope = EnvelopeEncryption(
            kms_client=mock_kms_client,
            key_id="test-key-id",
            metrics=mock_metrics,
        )

        key_pair = await envelope.generate_data_key(encryption_context.to_dict())

        assert key_pair.plaintext_key != key_pair.encrypted_key


# ============================================================================
# Test: Data Key Decryption
# ============================================================================


class TestDataKeyDecryption:
    """Tests for data key decryption via KMS."""

    @pytest.mark.asyncio
    async def test_decrypt_data_key_returns_plaintext(
        self,
        mock_kms_client,
        mock_metrics,
        encryption_context,
    ):
        """Test that decrypt_data_key returns the plaintext key."""
        if not _HAS_ENVELOPE_ENCRYPTION:
            pytest.skip("Envelope encryption not available")

        envelope = EnvelopeEncryption(
            kms_client=mock_kms_client,
            key_id="test-key-id",
            metrics=mock_metrics,
        )

        # Generate key first
        key_pair = await envelope.generate_data_key(encryption_context.to_dict())

        # Decrypt the encrypted key
        decrypted = await envelope.decrypt_data_key(
            key_pair.encrypted_key,
            encryption_context.to_dict(),
        )

        # Should return plaintext key
        assert decrypted == key_pair.plaintext_key

    @pytest.mark.asyncio
    async def test_encryption_context_required(
        self,
        mock_kms_client,
        mock_metrics,
    ):
        """Test that encryption context is required for key generation."""
        if not _HAS_ENVELOPE_ENCRYPTION:
            pytest.skip("Envelope encryption not available")

        envelope = EnvelopeEncryption(
            kms_client=mock_kms_client,
            key_id="test-key-id",
            metrics=mock_metrics,
        )

        # Empty context should be rejected or result in default context
        with pytest.raises((ValueError, TypeError)):
            await envelope.generate_data_key(None)

    @pytest.mark.asyncio
    async def test_encryption_context_bound_to_key(
        self,
        mock_kms_client,
        mock_metrics,
    ):
        """Test that encryption context is cryptographically bound to key."""
        if not _HAS_ENVELOPE_ENCRYPTION:
            pytest.skip("Envelope encryption not available")

        envelope = EnvelopeEncryption(
            kms_client=mock_kms_client,
            key_id="test-key-id",
            metrics=mock_metrics,
        )

        context1 = {"tenant_id": "tenant-001"}
        await envelope.generate_data_key(context1)

        # Verify KMS was called with the context
        mock_kms_client.generate_data_key.assert_called()
        call_kwargs = mock_kms_client.generate_data_key.call_args.kwargs
        assert "EncryptionContext" in call_kwargs or call_kwargs.get("encryption_context")

    @pytest.mark.asyncio
    async def test_wrong_context_fails_decrypt(
        self,
        mock_kms_client,
        mock_metrics,
    ):
        """Test that wrong context fails decryption."""
        if not _HAS_ENVELOPE_ENCRYPTION:
            pytest.skip("Envelope encryption not available")

        # Configure mock to fail on wrong context
        mock_kms_client.decrypt = AsyncMock(
            side_effect=Exception("InvalidCiphertextException: context mismatch")
        )

        envelope = EnvelopeEncryption(
            kms_client=mock_kms_client,
            key_id="test-key-id",
            metrics=mock_metrics,
        )

        with pytest.raises(Exception):
            await envelope.decrypt_data_key(
                b"some_encrypted_key",
                {"tenant_id": "wrong-tenant"},
            )


# ============================================================================
# Test: Envelope Encrypt/Decrypt
# ============================================================================


class TestEnvelopeEncryption:
    """Tests for full envelope encryption operations."""

    @pytest.mark.asyncio
    async def test_envelope_encrypt_roundtrip(
        self,
        mock_kms_client,
        mock_metrics,
        encryption_context,
    ):
        """Test full envelope encryption roundtrip."""
        if not _HAS_ENVELOPE_ENCRYPTION:
            pytest.skip("Envelope encryption not available")

        envelope = EnvelopeEncryption(
            kms_client=mock_kms_client,
            key_id="test-key-id",
            metrics=mock_metrics,
        )

        plaintext = b"Hello, envelope encryption!"
        context_dict = encryption_context.to_dict()

        # Encrypt
        encrypted = await envelope.encrypt(plaintext, context_dict)

        # Decrypt
        decrypted = await envelope.decrypt(encrypted, context_dict)

        assert decrypted == plaintext

    @pytest.mark.asyncio
    async def test_envelope_stores_encrypted_dek(
        self,
        mock_kms_client,
        mock_metrics,
        encryption_context,
    ):
        """Test that envelope stores the encrypted DEK."""
        if not _HAS_ENVELOPE_ENCRYPTION:
            pytest.skip("Envelope encryption not available")

        envelope = EnvelopeEncryption(
            kms_client=mock_kms_client,
            key_id="test-key-id",
            metrics=mock_metrics,
        )

        plaintext = b"Test data"
        context_dict = encryption_context.to_dict()

        encrypted = await envelope.encrypt(plaintext, context_dict)

        # Encrypted envelope should contain the encrypted DEK
        assert hasattr(encrypted, 'encrypted_dek') or 'encrypted_dek' in getattr(encrypted, '__dict__', {})


# ============================================================================
# Test: KMS Calls
# ============================================================================


class TestKMSCalls:
    """Tests for KMS API call behavior."""

    @pytest.mark.asyncio
    async def test_kms_called_on_generate(
        self,
        mock_kms_client,
        mock_metrics,
        encryption_context,
    ):
        """Test that KMS is called when generating data key."""
        if not _HAS_ENVELOPE_ENCRYPTION:
            pytest.skip("Envelope encryption not available")

        envelope = EnvelopeEncryption(
            kms_client=mock_kms_client,
            key_id="test-key-id",
            metrics=mock_metrics,
        )

        await envelope.generate_data_key(encryption_context.to_dict())

        mock_kms_client.generate_data_key.assert_called_once()

    @pytest.mark.asyncio
    async def test_kms_called_on_decrypt(
        self,
        mock_kms_client,
        mock_metrics,
        encryption_context,
    ):
        """Test that KMS is called when decrypting data key."""
        if not _HAS_ENVELOPE_ENCRYPTION:
            pytest.skip("Envelope encryption not available")

        envelope = EnvelopeEncryption(
            kms_client=mock_kms_client,
            key_id="test-key-id",
            metrics=mock_metrics,
        )

        await envelope.decrypt_data_key(
            mock_kms_client._test_encrypted_key,
            encryption_context.to_dict(),
        )

        mock_kms_client.decrypt.assert_called_once()

    @pytest.mark.asyncio
    async def test_kms_error_propagated(
        self,
        mock_metrics,
        encryption_context,
    ):
        """Test that KMS errors are properly propagated."""
        if not _HAS_ENVELOPE_ENCRYPTION:
            pytest.skip("Envelope encryption not available")

        # Create client that always fails
        failing_client = MagicMock()
        failing_client.generate_data_key = AsyncMock(
            side_effect=Exception("KMSInternalException: service error")
        )

        envelope = EnvelopeEncryption(
            kms_client=failing_client,
            key_id="test-key-id",
            metrics=mock_metrics,
        )

        with pytest.raises(Exception) as exc_info:
            await envelope.generate_data_key(encryption_context.to_dict())

        assert "KMS" in str(exc_info.value) or "service" in str(exc_info.value).lower()


# ============================================================================
# Test: Metrics Recording
# ============================================================================


class TestEnvelopeMetrics:
    """Tests for metrics recording during KMS operations."""

    @pytest.mark.asyncio
    async def test_metrics_recorded_for_kms_calls(
        self,
        mock_kms_client,
        mock_metrics,
        encryption_context,
    ):
        """Test that metrics are recorded for KMS API calls."""
        if not _HAS_ENVELOPE_ENCRYPTION:
            pytest.skip("Envelope encryption not available")

        envelope = EnvelopeEncryption(
            kms_client=mock_kms_client,
            key_id="test-key-id",
            metrics=mock_metrics,
        )

        await envelope.generate_data_key(encryption_context.to_dict())

        # Should record KMS call metric
        assert mock_metrics.record_kms_call.called or mock_metrics.record_kms_latency.called

    @pytest.mark.asyncio
    async def test_metrics_recorded_for_kms_errors(
        self,
        mock_metrics,
        encryption_context,
    ):
        """Test that metrics are recorded for KMS errors."""
        if not _HAS_ENVELOPE_ENCRYPTION:
            pytest.skip("Envelope encryption not available")

        failing_client = MagicMock()
        failing_client.generate_data_key = AsyncMock(
            side_effect=Exception("KMS error")
        )

        envelope = EnvelopeEncryption(
            kms_client=failing_client,
            key_id="test-key-id",
            metrics=mock_metrics,
        )

        with pytest.raises(Exception):
            await envelope.generate_data_key(encryption_context.to_dict())

        # Should record error metric
        mock_metrics.record_kms_error.assert_called()


# ============================================================================
# Test: Key ID Configuration
# ============================================================================


class TestKeyIdConfiguration:
    """Tests for KMS key ID configuration."""

    @pytest.mark.asyncio
    async def test_key_id_used_in_generate(
        self,
        mock_kms_client,
        mock_metrics,
        encryption_context,
    ):
        """Test that configured key ID is used in generate_data_key."""
        if not _HAS_ENVELOPE_ENCRYPTION:
            pytest.skip("Envelope encryption not available")

        key_id = "alias/greenlang-encryption-key"
        envelope = EnvelopeEncryption(
            kms_client=mock_kms_client,
            key_id=key_id,
            metrics=mock_metrics,
        )

        await envelope.generate_data_key(encryption_context.to_dict())

        # Verify key_id was passed to KMS
        call_kwargs = mock_kms_client.generate_data_key.call_args.kwargs
        assert call_kwargs.get("KeyId") == key_id or call_kwargs.get("key_id") == key_id

    @pytest.mark.asyncio
    async def test_key_arn_accepted(
        self,
        mock_kms_client,
        mock_metrics,
        encryption_context,
    ):
        """Test that full key ARN is accepted."""
        if not _HAS_ENVELOPE_ENCRYPTION:
            pytest.skip("Envelope encryption not available")

        key_arn = "arn:aws:kms:us-east-1:123456789012:key/12345678-1234-1234-1234-123456789012"
        envelope = EnvelopeEncryption(
            kms_client=mock_kms_client,
            key_id=key_arn,
            metrics=mock_metrics,
        )

        # Should not raise
        key_pair = await envelope.generate_data_key(encryption_context.to_dict())
        assert key_pair is not None


# ============================================================================
# Test: Key Spec
# ============================================================================


class TestKeySpec:
    """Tests for data key specification."""

    @pytest.mark.asyncio
    async def test_aes_256_key_spec(
        self,
        mock_kms_client,
        mock_metrics,
        encryption_context,
    ):
        """Test that AES_256 key spec is used."""
        if not _HAS_ENVELOPE_ENCRYPTION:
            pytest.skip("Envelope encryption not available")

        envelope = EnvelopeEncryption(
            kms_client=mock_kms_client,
            key_id="test-key-id",
            metrics=mock_metrics,
        )

        await envelope.generate_data_key(encryption_context.to_dict())

        # Verify AES_256 key spec was requested
        call_kwargs = mock_kms_client.generate_data_key.call_args.kwargs
        key_spec = call_kwargs.get("KeySpec") or call_kwargs.get("key_spec")
        assert key_spec in ["AES_256", "aes_256", None]  # None means default


# ============================================================================
# Test: Error Handling
# ============================================================================


class TestEnvelopeErrorHandling:
    """Tests for error handling in envelope encryption."""

    @pytest.mark.asyncio
    async def test_invalid_encrypted_key_rejected(
        self,
        mock_kms_client,
        mock_metrics,
        encryption_context,
    ):
        """Test that invalid encrypted key is rejected."""
        if not _HAS_ENVELOPE_ENCRYPTION:
            pytest.skip("Envelope encryption not available")

        mock_kms_client.decrypt = AsyncMock(
            side_effect=Exception("InvalidCiphertextException")
        )

        envelope = EnvelopeEncryption(
            kms_client=mock_kms_client,
            key_id="test-key-id",
            metrics=mock_metrics,
        )

        with pytest.raises(Exception) as exc_info:
            await envelope.decrypt_data_key(
                b"invalid_encrypted_key",
                encryption_context.to_dict(),
            )

        assert "InvalidCiphertext" in str(exc_info.value) or "invalid" in str(exc_info.value).lower()

    @pytest.mark.asyncio
    async def test_access_denied_error(
        self,
        mock_kms_client,
        mock_metrics,
        encryption_context,
    ):
        """Test handling of access denied errors."""
        if not _HAS_ENVELOPE_ENCRYPTION:
            pytest.skip("Envelope encryption not available")

        mock_kms_client.generate_data_key = AsyncMock(
            side_effect=Exception("AccessDeniedException: not authorized")
        )

        envelope = EnvelopeEncryption(
            kms_client=mock_kms_client,
            key_id="test-key-id",
            metrics=mock_metrics,
        )

        with pytest.raises(Exception) as exc_info:
            await envelope.generate_data_key(encryption_context.to_dict())

        assert "AccessDenied" in str(exc_info.value) or "authorized" in str(exc_info.value).lower()

    @pytest.mark.asyncio
    async def test_key_disabled_error(
        self,
        mock_kms_client,
        mock_metrics,
        encryption_context,
    ):
        """Test handling of disabled key errors."""
        if not _HAS_ENVELOPE_ENCRYPTION:
            pytest.skip("Envelope encryption not available")

        mock_kms_client.generate_data_key = AsyncMock(
            side_effect=Exception("DisabledException: key is disabled")
        )

        envelope = EnvelopeEncryption(
            kms_client=mock_kms_client,
            key_id="test-key-id",
            metrics=mock_metrics,
        )

        with pytest.raises(Exception) as exc_info:
            await envelope.generate_data_key(encryption_context.to_dict())

        assert "Disabled" in str(exc_info.value) or "disabled" in str(exc_info.value).lower()


# ============================================================================
# Test: Grant Tokens
# ============================================================================


class TestGrantTokens:
    """Tests for KMS grant token support."""

    @pytest.mark.asyncio
    async def test_grant_tokens_passed_to_kms(
        self,
        mock_kms_client,
        mock_metrics,
        encryption_context,
    ):
        """Test that grant tokens are passed to KMS if provided."""
        if not _HAS_ENVELOPE_ENCRYPTION:
            pytest.skip("Envelope encryption not available")

        envelope = EnvelopeEncryption(
            kms_client=mock_kms_client,
            key_id="test-key-id",
            metrics=mock_metrics,
        )

        grant_tokens = ["grant-token-1", "grant-token-2"]

        # If the implementation supports grant tokens
        if hasattr(envelope, 'generate_data_key_with_grants'):
            await envelope.generate_data_key_with_grants(
                encryption_context.to_dict(),
                grant_tokens=grant_tokens,
            )

            call_kwargs = mock_kms_client.generate_data_key.call_args.kwargs
            assert call_kwargs.get("GrantTokens") == grant_tokens
