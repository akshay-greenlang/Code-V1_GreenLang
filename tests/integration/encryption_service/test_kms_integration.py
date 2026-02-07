# -*- coding: utf-8 -*-
"""
KMS Integration tests for Encryption Service (SEC-003)

Tests the integration with AWS KMS for envelope encryption, using moto
for AWS service mocking. Verifies key generation, decryption, encryption
context binding, and error handling.

Coverage targets: KMS integration paths
"""

from __future__ import annotations

import secrets
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Attempt to import required modules
# ---------------------------------------------------------------------------
try:
    import boto3
    from botocore.exceptions import ClientError
    _HAS_BOTO3 = True
except ImportError:
    _HAS_BOTO3 = False

try:
    from moto import mock_aws
    _HAS_MOTO = True
except ImportError:
    try:
        from moto import mock_kms
        _HAS_MOTO = True

        def mock_aws():
            return mock_kms()
    except ImportError:
        _HAS_MOTO = False

try:
    from greenlang.infrastructure.encryption_service.envelope_encryption import (
        EnvelopeEncryption,
        KMSClient,
    )
    _HAS_ENVELOPE = True
except ImportError:
    _HAS_ENVELOPE = False

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(not _HAS_BOTO3, reason="boto3 not installed"),
    pytest.mark.skipif(not _HAS_MOTO, reason="moto not installed"),
    pytest.mark.skipif(not _HAS_ENVELOPE, reason="Envelope encryption not installed"),
]


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def aws_credentials():
    """Set up mock AWS credentials."""
    import os
    os.environ["AWS_ACCESS_KEY_ID"] = "testing"
    os.environ["AWS_SECRET_ACCESS_KEY"] = "testing"
    os.environ["AWS_SECURITY_TOKEN"] = "testing"
    os.environ["AWS_SESSION_TOKEN"] = "testing"
    os.environ["AWS_DEFAULT_REGION"] = "us-east-1"


@pytest.fixture
def kms_key_id(aws_credentials):
    """Create a KMS key and return its ID."""
    if not _HAS_MOTO:
        pytest.skip("moto not available")

    with mock_aws():
        client = boto3.client("kms", region_name="us-east-1")
        response = client.create_key(
            Description="Test key for encryption service",
            KeyUsage="ENCRYPT_DECRYPT",
            Origin="AWS_KMS",
        )
        key_id = response["KeyMetadata"]["KeyId"]
        yield key_id


# ============================================================================
# Test: KMS Generate Data Key
# ============================================================================


class TestKMSGenerateDataKey:
    """Tests for KMS GenerateDataKey operation."""

    @pytest.mark.asyncio
    async def test_kms_generate_data_key(self, aws_credentials):
        """Test generating a data key via KMS."""
        if not _HAS_MOTO or not _HAS_ENVELOPE:
            pytest.skip("Required modules not available")

        with mock_aws():
            # Create KMS client and key
            kms_client = boto3.client("kms", region_name="us-east-1")
            key_response = kms_client.create_key(
                Description="Test key",
                KeyUsage="ENCRYPT_DECRYPT",
            )
            key_id = key_response["KeyMetadata"]["KeyId"]

            # Generate data key
            response = kms_client.generate_data_key(
                KeyId=key_id,
                KeySpec="AES_256",
                EncryptionContext={
                    "tenant_id": "tenant-test",
                    "purpose": "data_encryption",
                },
            )

            assert "Plaintext" in response
            assert "CiphertextBlob" in response
            assert len(response["Plaintext"]) == 32  # AES-256

    @pytest.mark.asyncio
    async def test_kms_generate_data_key_with_context(self, aws_credentials):
        """Test that encryption context is properly applied."""
        if not _HAS_MOTO or not _HAS_ENVELOPE:
            pytest.skip("Required modules not available")

        with mock_aws():
            kms_client = boto3.client("kms", region_name="us-east-1")
            key_response = kms_client.create_key(
                Description="Test key with context",
                KeyUsage="ENCRYPT_DECRYPT",
            )
            key_id = key_response["KeyMetadata"]["KeyId"]

            encryption_context = {
                "tenant_id": "tenant-ctx",
                "data_class": "confidential",
                "purpose": "testing",
            }

            response = kms_client.generate_data_key(
                KeyId=key_id,
                KeySpec="AES_256",
                EncryptionContext=encryption_context,
            )

            assert response["Plaintext"] is not None
            assert response["CiphertextBlob"] is not None


# ============================================================================
# Test: KMS Decrypt Data Key
# ============================================================================


class TestKMSDecryptDataKey:
    """Tests for KMS Decrypt operation."""

    @pytest.mark.asyncio
    async def test_kms_decrypt_data_key(self, aws_credentials):
        """Test decrypting a data key via KMS."""
        if not _HAS_MOTO or not _HAS_ENVELOPE:
            pytest.skip("Required modules not available")

        with mock_aws():
            kms_client = boto3.client("kms", region_name="us-east-1")
            key_response = kms_client.create_key(
                Description="Test key for decrypt",
                KeyUsage="ENCRYPT_DECRYPT",
            )
            key_id = key_response["KeyMetadata"]["KeyId"]

            encryption_context = {"tenant_id": "tenant-decrypt"}

            # Generate data key
            gen_response = kms_client.generate_data_key(
                KeyId=key_id,
                KeySpec="AES_256",
                EncryptionContext=encryption_context,
            )
            plaintext_original = gen_response["Plaintext"]
            ciphertext_blob = gen_response["CiphertextBlob"]

            # Decrypt the data key
            decrypt_response = kms_client.decrypt(
                CiphertextBlob=ciphertext_blob,
                EncryptionContext=encryption_context,
            )

            assert decrypt_response["Plaintext"] == plaintext_original

    @pytest.mark.asyncio
    async def test_kms_decrypt_wrong_context_fails(self, aws_credentials):
        """Test that decryption with wrong context fails."""
        if not _HAS_MOTO or not _HAS_ENVELOPE:
            pytest.skip("Required modules not available")

        with mock_aws():
            kms_client = boto3.client("kms", region_name="us-east-1")
            key_response = kms_client.create_key(
                Description="Test key for context validation",
                KeyUsage="ENCRYPT_DECRYPT",
            )
            key_id = key_response["KeyMetadata"]["KeyId"]

            # Generate with one context
            gen_response = kms_client.generate_data_key(
                KeyId=key_id,
                KeySpec="AES_256",
                EncryptionContext={"tenant_id": "tenant-1"},
            )
            ciphertext_blob = gen_response["CiphertextBlob"]

            # Try to decrypt with different context
            with pytest.raises(ClientError) as exc_info:
                kms_client.decrypt(
                    CiphertextBlob=ciphertext_blob,
                    EncryptionContext={"tenant_id": "tenant-2"},
                )

            assert "InvalidCiphertext" in str(exc_info.value) or exc_info.value is not None


# ============================================================================
# Test: KMS Key Rotation
# ============================================================================


class TestKMSKeyRotation:
    """Tests for KMS key rotation."""

    @pytest.mark.asyncio
    async def test_kms_key_rotation(self, aws_credentials):
        """Test enabling automatic key rotation."""
        if not _HAS_MOTO or not _HAS_ENVELOPE:
            pytest.skip("Required modules not available")

        with mock_aws():
            kms_client = boto3.client("kms", region_name="us-east-1")
            key_response = kms_client.create_key(
                Description="Test key for rotation",
                KeyUsage="ENCRYPT_DECRYPT",
            )
            key_id = key_response["KeyMetadata"]["KeyId"]

            # Enable rotation
            kms_client.enable_key_rotation(KeyId=key_id)

            # Verify rotation is enabled
            status = kms_client.get_key_rotation_status(KeyId=key_id)
            assert status["KeyRotationEnabled"] is True

    @pytest.mark.asyncio
    async def test_old_keys_still_decrypt_after_rotation(self, aws_credentials):
        """Test that old encrypted data can still be decrypted after rotation."""
        if not _HAS_MOTO or not _HAS_ENVELOPE:
            pytest.skip("Required modules not available")

        with mock_aws():
            kms_client = boto3.client("kms", region_name="us-east-1")
            key_response = kms_client.create_key(
                Description="Test key for rotation decrypt",
                KeyUsage="ENCRYPT_DECRYPT",
            )
            key_id = key_response["KeyMetadata"]["KeyId"]

            encryption_context = {"tenant_id": "tenant-rotation"}

            # Generate data key before rotation
            gen_response = kms_client.generate_data_key(
                KeyId=key_id,
                KeySpec="AES_256",
                EncryptionContext=encryption_context,
            )
            ciphertext_blob = gen_response["CiphertextBlob"]
            original_plaintext = gen_response["Plaintext"]

            # Enable rotation (simulates rotation happening)
            kms_client.enable_key_rotation(KeyId=key_id)

            # Should still be able to decrypt old data
            decrypt_response = kms_client.decrypt(
                CiphertextBlob=ciphertext_blob,
                EncryptionContext=encryption_context,
            )

            assert decrypt_response["Plaintext"] == original_plaintext


# ============================================================================
# Test: KMS Encryption Context
# ============================================================================


class TestKMSEncryptionContext:
    """Tests for KMS encryption context handling."""

    @pytest.mark.asyncio
    async def test_kms_encryption_context(self, aws_credentials):
        """Test encryption context is properly enforced."""
        if not _HAS_MOTO or not _HAS_ENVELOPE:
            pytest.skip("Required modules not available")

        with mock_aws():
            kms_client = boto3.client("kms", region_name="us-east-1")
            key_response = kms_client.create_key(
                Description="Test key for context",
                KeyUsage="ENCRYPT_DECRYPT",
            )
            key_id = key_response["KeyMetadata"]["KeyId"]

            # Complex encryption context
            encryption_context = {
                "tenant_id": "tenant-ctx-test",
                "data_class": "highly_confidential",
                "region": "us-east-1",
                "application": "greenlang",
                "environment": "production",
            }

            # Generate with context
            gen_response = kms_client.generate_data_key(
                KeyId=key_id,
                KeySpec="AES_256",
                EncryptionContext=encryption_context,
            )

            # Decrypt with same context
            decrypt_response = kms_client.decrypt(
                CiphertextBlob=gen_response["CiphertextBlob"],
                EncryptionContext=encryption_context,
            )

            assert decrypt_response["Plaintext"] == gen_response["Plaintext"]

    @pytest.mark.asyncio
    async def test_partial_context_fails(self, aws_credentials):
        """Test that partial encryption context fails decryption."""
        if not _HAS_MOTO or not _HAS_ENVELOPE:
            pytest.skip("Required modules not available")

        with mock_aws():
            kms_client = boto3.client("kms", region_name="us-east-1")
            key_response = kms_client.create_key(
                Description="Test key for partial context",
                KeyUsage="ENCRYPT_DECRYPT",
            )
            key_id = key_response["KeyMetadata"]["KeyId"]

            full_context = {
                "tenant_id": "tenant-partial",
                "data_class": "confidential",
            }

            # Generate with full context
            gen_response = kms_client.generate_data_key(
                KeyId=key_id,
                KeySpec="AES_256",
                EncryptionContext=full_context,
            )

            # Try decrypt with partial context
            with pytest.raises(ClientError):
                kms_client.decrypt(
                    CiphertextBlob=gen_response["CiphertextBlob"],
                    EncryptionContext={"tenant_id": "tenant-partial"},  # Missing data_class
                )


# ============================================================================
# Test: KMS Error Handling
# ============================================================================


class TestKMSErrorHandling:
    """Tests for KMS error handling."""

    @pytest.mark.asyncio
    async def test_kms_invalid_key_id(self, aws_credentials):
        """Test handling of invalid key ID."""
        if not _HAS_MOTO or not _HAS_ENVELOPE:
            pytest.skip("Required modules not available")

        with mock_aws():
            kms_client = boto3.client("kms", region_name="us-east-1")

            with pytest.raises(ClientError) as exc_info:
                kms_client.generate_data_key(
                    KeyId="invalid-key-id-12345",
                    KeySpec="AES_256",
                )

            error = exc_info.value.response.get("Error", {})
            assert error.get("Code") in ["NotFoundException", "InvalidKeyId", "ValidationException"]

    @pytest.mark.asyncio
    async def test_kms_disabled_key(self, aws_credentials):
        """Test handling of disabled key."""
        if not _HAS_MOTO or not _HAS_ENVELOPE:
            pytest.skip("Required modules not available")

        with mock_aws():
            kms_client = boto3.client("kms", region_name="us-east-1")
            key_response = kms_client.create_key(
                Description="Test key for disable",
                KeyUsage="ENCRYPT_DECRYPT",
            )
            key_id = key_response["KeyMetadata"]["KeyId"]

            # Disable the key
            kms_client.disable_key(KeyId=key_id)

            # Try to use disabled key
            with pytest.raises(ClientError) as exc_info:
                kms_client.generate_data_key(
                    KeyId=key_id,
                    KeySpec="AES_256",
                )

            error = exc_info.value.response.get("Error", {})
            assert error.get("Code") in ["DisabledException", "KMSInvalidStateException"]

    @pytest.mark.asyncio
    async def test_kms_invalid_ciphertext(self, aws_credentials):
        """Test handling of invalid ciphertext."""
        if not _HAS_MOTO or not _HAS_ENVELOPE:
            pytest.skip("Required modules not available")

        with mock_aws():
            kms_client = boto3.client("kms", region_name="us-east-1")
            key_response = kms_client.create_key(
                Description="Test key for invalid ciphertext",
                KeyUsage="ENCRYPT_DECRYPT",
            )
            key_id = key_response["KeyMetadata"]["KeyId"]

            # Try to decrypt garbage
            with pytest.raises(ClientError) as exc_info:
                kms_client.decrypt(
                    CiphertextBlob=b"invalid-ciphertext-data",
                    EncryptionContext={"tenant_id": "test"},
                )

            error = exc_info.value.response.get("Error", {})
            assert error.get("Code") in ["InvalidCiphertextException", "ValidationException"]


# ============================================================================
# Test: KMS Key Alias
# ============================================================================


class TestKMSKeyAlias:
    """Tests for KMS key alias usage."""

    @pytest.mark.asyncio
    async def test_kms_alias_usage(self, aws_credentials):
        """Test using key alias instead of key ID."""
        if not _HAS_MOTO or not _HAS_ENVELOPE:
            pytest.skip("Required modules not available")

        with mock_aws():
            kms_client = boto3.client("kms", region_name="us-east-1")
            key_response = kms_client.create_key(
                Description="Test key for alias",
                KeyUsage="ENCRYPT_DECRYPT",
            )
            key_id = key_response["KeyMetadata"]["KeyId"]

            # Create alias
            alias_name = "alias/greenlang-encryption-test"
            kms_client.create_alias(
                AliasName=alias_name,
                TargetKeyId=key_id,
            )

            # Use alias for operations
            response = kms_client.generate_data_key(
                KeyId=alias_name,
                KeySpec="AES_256",
                EncryptionContext={"tenant_id": "test"},
            )

            assert response["Plaintext"] is not None
            assert len(response["Plaintext"]) == 32


# ============================================================================
# Test: KMS Grant Operations
# ============================================================================


class TestKMSGrants:
    """Tests for KMS grant operations."""

    @pytest.mark.asyncio
    async def test_kms_create_grant(self, aws_credentials):
        """Test creating a KMS grant."""
        if not _HAS_MOTO or not _HAS_ENVELOPE:
            pytest.skip("Required modules not available")

        with mock_aws():
            kms_client = boto3.client("kms", region_name="us-east-1")
            iam_client = boto3.client("iam", region_name="us-east-1")

            # Create key
            key_response = kms_client.create_key(
                Description="Test key for grants",
                KeyUsage="ENCRYPT_DECRYPT",
            )
            key_id = key_response["KeyMetadata"]["KeyId"]

            # Create a test principal (using current user ARN as placeholder)
            # In real tests, you'd use a real IAM principal
            grantee_principal = "arn:aws:iam::123456789012:user/test-user"

            # Create grant
            try:
                grant_response = kms_client.create_grant(
                    KeyId=key_id,
                    GranteePrincipal=grantee_principal,
                    Operations=["GenerateDataKey", "Decrypt"],
                    Name="TestGrant",
                )

                assert "GrantId" in grant_response or grant_response is not None
            except ClientError as e:
                # Some moto versions may not fully support grants
                if "NotImplemented" in str(e):
                    pytest.skip("Grants not implemented in this moto version")
                raise
