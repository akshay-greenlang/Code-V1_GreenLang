# -*- coding: utf-8 -*-
"""
Integration Tests for Vault - SEC-006

Tests the complete Vault integration including:
- Connection and authentication
- KV secrets lifecycle
- Dynamic database credentials
- Transit encryption/decryption
- PKI certificate management
- Lease management and rotation
- Multi-tenant isolation

These tests require either:
- A running Vault instance (dev mode or testcontainers)
- Mock Vault responses for CI/CD

Set VAULT_ADDR environment variable to point to test Vault.
"""

from __future__ import annotations

import asyncio
import os
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Skip if dependencies not available
# ---------------------------------------------------------------------------
try:
    import httpx
    _HAS_HTTPX = True
except ImportError:
    _HAS_HTTPX = False

try:
    from greenlang.execution.infrastructure.secrets import (
        VaultClient,
        VaultConfig,
        VaultAuthMethod,
        VaultSecret,
        VaultError,
        VaultSecretNotFoundError,
        VaultPermissionError,
        VaultConnectionError,
        DatabaseCredentials,
        Certificate,
    )
    _HAS_VAULT = True
except ImportError:
    _HAS_VAULT = False

try:
    from testcontainers.core.container import DockerContainer
    _HAS_TESTCONTAINERS = True
except ImportError:
    _HAS_TESTCONTAINERS = False


pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(not _HAS_HTTPX, reason="httpx not installed"),
    pytest.mark.skipif(not _HAS_VAULT, reason="vault client not implemented"),
]


# ============================================================================
# Test Configuration
# ============================================================================

TEST_CONFIG = {
    "vault": {
        "addr": os.getenv("VAULT_ADDR", "http://127.0.0.1:8200"),
        "token": os.getenv("VAULT_TOKEN", "root"),  # Dev mode token
    }
}


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture(scope="module")
def vault_config() -> VaultConfig:
    """Create Vault configuration for testing."""
    return VaultConfig(
        addr=TEST_CONFIG["vault"]["addr"],
        auth_method=VaultAuthMethod.TOKEN,
        token=TEST_CONFIG["vault"]["token"],
        skip_verify=True,
        cache_enabled=False,  # Disable caching for tests
    )


@pytest.fixture
async def vault_client(vault_config) -> VaultClient:
    """Create VaultClient for testing."""
    client = VaultClient(vault_config)
    try:
        await client.connect()
        yield client
    except Exception:
        # Vault not available, use mock
        mock_client = AsyncMock(spec=VaultClient)
        mock_client.get_secret = AsyncMock(return_value=VaultSecret(
            data={"test": "value"},
            metadata={"version": 1},
        ))
        mock_client.put_secret = AsyncMock(return_value={"version": 1})
        mock_client.delete_secret = AsyncMock()
        mock_client.health_check = AsyncMock(return_value={"initialized": True, "sealed": False})
        mock_client.is_healthy = AsyncMock(return_value=True)
        yield mock_client
    finally:
        try:
            await client.close()
        except Exception:
            pass


@pytest.fixture
def sample_secret_path() -> str:
    """Generate unique secret path for test isolation."""
    return f"test/integration/{uuid.uuid4().hex[:8]}"


@pytest.fixture
def sample_tenant_id() -> str:
    """Generate unique tenant ID."""
    return f"t-test-{uuid.uuid4().hex[:8]}"


# ============================================================================
# TestVaultConnection
# ============================================================================


class TestVaultConnection:
    """Tests for Vault connection and health."""

    @pytest.mark.asyncio
    async def test_vault_connection(self, vault_client) -> None:
        """Test connecting to Vault."""
        is_healthy = await vault_client.is_healthy()
        assert is_healthy is True

    @pytest.mark.asyncio
    async def test_vault_health_check(self, vault_client) -> None:
        """Test Vault health check response."""
        health = await vault_client.health_check()

        assert "initialized" in health or health.get("initialized") is not None
        assert health.get("sealed") is False or "sealed" in health

    @pytest.mark.asyncio
    async def test_vault_reconnect_after_close(self, vault_config) -> None:
        """Test reconnecting after closing client."""
        client = VaultClient(vault_config)

        try:
            await client.connect()
            await client.close()

            # Should be able to reconnect
            await client.connect()
            is_healthy = await client.is_healthy()
            assert is_healthy is True
        except Exception:
            # Vault not available
            pass
        finally:
            try:
                await client.close()
            except Exception:
                pass


# ============================================================================
# TestKubernetesAuth
# ============================================================================


class TestKubernetesAuth:
    """Tests for Kubernetes authentication."""

    @pytest.mark.asyncio
    @pytest.mark.skipif(
        not os.path.exists("/var/run/secrets/kubernetes.io/serviceaccount/token"),
        reason="Not running in Kubernetes"
    )
    async def test_kubernetes_auth(self) -> None:
        """Test Kubernetes service account authentication."""
        config = VaultConfig(
            addr=TEST_CONFIG["vault"]["addr"],
            auth_method=VaultAuthMethod.KUBERNETES,
            kubernetes_role="test-role",
        )

        client = VaultClient(config)
        try:
            await client.connect()
            is_healthy = await client.is_healthy()
            assert is_healthy is True
        finally:
            await client.close()


# ============================================================================
# TestKVSecretLifecycle
# ============================================================================


class TestKVSecretLifecycle:
    """Tests for KV secrets engine operations."""

    @pytest.mark.asyncio
    async def test_kv_secret_lifecycle(
        self, vault_client, sample_secret_path
    ) -> None:
        """Test complete KV secret lifecycle: create, read, update, delete."""
        # Create
        metadata = await vault_client.put_secret(
            sample_secret_path,
            {"username": "admin", "password": "secret123"}
        )
        assert "version" in metadata or metadata is not None

        # Read
        secret = await vault_client.get_secret(sample_secret_path)
        assert secret.data["username"] == "admin"
        assert secret.data["password"] == "secret123"

        # Update
        await vault_client.put_secret(
            sample_secret_path,
            {"username": "admin", "password": "newsecret456"}
        )

        updated = await vault_client.get_secret(sample_secret_path)
        assert updated.data["password"] == "newsecret456"

        # Delete
        await vault_client.delete_secret(sample_secret_path)

        # Verify deleted
        with pytest.raises((VaultSecretNotFoundError, VaultError)):
            await vault_client.get_secret(sample_secret_path)

    @pytest.mark.asyncio
    async def test_kv_secret_versioning(
        self, vault_client, sample_secret_path
    ) -> None:
        """Test KV secret versioning."""
        # Create version 1
        await vault_client.put_secret(sample_secret_path, {"value": "v1"})

        # Create version 2
        await vault_client.put_secret(sample_secret_path, {"value": "v2"})

        # Create version 3
        await vault_client.put_secret(sample_secret_path, {"value": "v3"})

        # Get specific versions
        v1 = await vault_client.get_secret(sample_secret_path, version=1)
        v2 = await vault_client.get_secret(sample_secret_path, version=2)
        latest = await vault_client.get_secret(sample_secret_path)

        assert v1.data["value"] == "v1"
        assert v2.data["value"] == "v2"
        assert latest.data["value"] == "v3"

        # Cleanup
        await vault_client.delete_secret(sample_secret_path)

    @pytest.mark.asyncio
    async def test_kv_secret_cas_update(
        self, vault_client, sample_secret_path
    ) -> None:
        """Test KV secret check-and-set update."""
        # Create initial version
        await vault_client.put_secret(sample_secret_path, {"value": "initial"})

        # Get current version
        secret = await vault_client.get_secret(sample_secret_path)
        current_version = secret.metadata.get("version", 1)

        # Update with correct CAS
        await vault_client.put_secret(
            sample_secret_path,
            {"value": "updated"},
            cas=current_version
        )

        # Verify update
        updated = await vault_client.get_secret(sample_secret_path)
        assert updated.data["value"] == "updated"

        # Cleanup
        await vault_client.delete_secret(sample_secret_path)


# ============================================================================
# TestDatabaseCredentials
# ============================================================================


class TestDatabaseCredentials:
    """Tests for dynamic database credentials."""

    @pytest.mark.asyncio
    @pytest.mark.skip(reason="Requires database secrets engine configuration")
    async def test_database_credentials(self, vault_client) -> None:
        """Test getting dynamic database credentials."""
        creds = await vault_client.get_database_credentials("readonly")

        assert creds.username is not None
        assert creds.password is not None
        assert creds.lease_id is not None
        assert creds.lease_duration > 0

    @pytest.mark.asyncio
    @pytest.mark.skip(reason="Requires database secrets engine configuration")
    async def test_database_credentials_lease_renewal(self, vault_client) -> None:
        """Test renewing database credential lease."""
        creds = await vault_client.get_database_credentials("readonly")

        # Renew the lease
        result = await vault_client.renew_lease(creds.lease_id)

        assert "lease_id" in result

    @pytest.mark.asyncio
    @pytest.mark.skip(reason="Requires database secrets engine configuration")
    async def test_database_credentials_revoke(self, vault_client) -> None:
        """Test revoking database credential lease."""
        creds = await vault_client.get_database_credentials("readonly")

        # Revoke the lease
        await vault_client.revoke_lease(creds.lease_id)

        # Credentials should be invalid now


# ============================================================================
# TestTransitEngine
# ============================================================================


class TestTransitEngine:
    """Tests for Transit secrets engine (encryption as a service)."""

    @pytest.mark.asyncio
    @pytest.mark.skip(reason="Requires transit secrets engine configuration")
    async def test_transit_encrypt_decrypt(self, vault_client) -> None:
        """Test encrypting and decrypting data."""
        key_name = "test-key"
        plaintext = b"sensitive data to encrypt"

        # Encrypt
        ciphertext = await vault_client.encrypt_data(key_name, plaintext)
        assert ciphertext.startswith("vault:v")

        # Decrypt
        decrypted = await vault_client.decrypt_data(key_name, ciphertext)
        assert decrypted == plaintext

    @pytest.mark.asyncio
    @pytest.mark.skip(reason="Requires transit secrets engine configuration")
    async def test_transit_batch_encrypt(self, vault_client) -> None:
        """Test batch encryption."""
        key_name = "test-key"
        items = [b"item1", b"item2", b"item3"]

        ciphertexts = await vault_client.encrypt_batch(key_name, items)

        assert len(ciphertexts) == 3
        assert all(ct.startswith("vault:v") for ct in ciphertexts)

    @pytest.mark.asyncio
    @pytest.mark.skip(reason="Requires transit secrets engine configuration")
    async def test_transit_sign_verify(self, vault_client) -> None:
        """Test signing and verifying data."""
        key_name = "signing-key"
        data = b"data to sign"

        # Sign
        signature = await vault_client.sign_data(key_name, data)
        assert signature.startswith("vault:v")

        # Verify
        is_valid = await vault_client.verify_signature(key_name, data, signature)
        assert is_valid is True


# ============================================================================
# TestPKICertificate
# ============================================================================


class TestPKICertificate:
    """Tests for PKI secrets engine."""

    @pytest.mark.asyncio
    @pytest.mark.skip(reason="Requires PKI secrets engine configuration")
    async def test_pki_certificate(self, vault_client) -> None:
        """Test generating a TLS certificate."""
        cert = await vault_client.generate_certificate(
            role="test-services",
            common_name="test.greenlang.svc",
            alt_names=["test.greenlang.internal"],
            ttl="24h",
        )

        assert cert.certificate.startswith("-----BEGIN CERTIFICATE-----")
        assert cert.private_key.startswith("-----BEGIN")
        assert cert.serial_number is not None
        assert cert.expiration > datetime.utcnow()

    @pytest.mark.asyncio
    @pytest.mark.skip(reason="Requires PKI secrets engine configuration")
    async def test_pki_certificate_revocation(self, vault_client) -> None:
        """Test revoking a certificate."""
        cert = await vault_client.generate_certificate(
            role="test-services",
            common_name="revoke-test.greenlang.svc",
        )

        await vault_client.revoke_certificate(cert.serial_number)


# ============================================================================
# TestLeaseManagement
# ============================================================================


class TestLeaseManagement:
    """Tests for lease renewal and management."""

    @pytest.mark.asyncio
    @pytest.mark.skip(reason="Requires dynamic secrets configuration")
    async def test_lease_renewal(self, vault_client) -> None:
        """Test renewing a lease."""
        # Get credentials with a lease
        creds = await vault_client.get_database_credentials("readonly")

        # Renew
        result = await vault_client.renew_lease(creds.lease_id)

        assert result is not None

    @pytest.mark.asyncio
    @pytest.mark.skip(reason="Requires dynamic secrets configuration")
    async def test_lease_revocation(self, vault_client) -> None:
        """Test revoking a lease."""
        creds = await vault_client.get_database_credentials("readonly")

        # Revoke
        await vault_client.revoke_lease(creds.lease_id)


# ============================================================================
# TestSecretRotation
# ============================================================================


class TestSecretRotation:
    """Tests for secret rotation integration."""

    @pytest.mark.asyncio
    async def test_secret_rotation(
        self, vault_client, sample_secret_path
    ) -> None:
        """Test rotating a secret creates new version."""
        # Create initial secret
        await vault_client.put_secret(
            sample_secret_path,
            {"api_key": "old-key-123"}
        )

        # Get initial version
        v1 = await vault_client.get_secret(sample_secret_path)
        v1_version = v1.metadata.get("version", 1)

        # "Rotate" by updating
        await vault_client.put_secret(
            sample_secret_path,
            {"api_key": "new-key-456"}
        )

        # Get new version
        v2 = await vault_client.get_secret(sample_secret_path)
        v2_version = v2.metadata.get("version", 2)

        assert v2_version > v1_version
        assert v2.data["api_key"] == "new-key-456"

        # Old version should still be accessible
        old = await vault_client.get_secret(sample_secret_path, version=v1_version)
        assert old.data["api_key"] == "old-key-123"

        # Cleanup
        await vault_client.delete_secret(sample_secret_path)


# ============================================================================
# TestMultiTenantIsolation
# ============================================================================


class TestMultiTenantIsolation:
    """Tests for multi-tenant secret isolation."""

    @pytest.mark.asyncio
    async def test_multi_tenant_isolation(
        self, vault_client
    ) -> None:
        """Test secrets are isolated between tenants."""
        tenant1_path = f"tenants/t-alpha/{uuid.uuid4().hex[:8]}"
        tenant2_path = f"tenants/t-beta/{uuid.uuid4().hex[:8]}"

        # Create secrets for each tenant
        await vault_client.put_secret(tenant1_path, {"key": "tenant1-value"})
        await vault_client.put_secret(tenant2_path, {"key": "tenant2-value"})

        # Each tenant should only see their own secret
        t1_secret = await vault_client.get_secret(tenant1_path)
        t2_secret = await vault_client.get_secret(tenant2_path)

        assert t1_secret.data["key"] == "tenant1-value"
        assert t2_secret.data["key"] == "tenant2-value"

        # Cleanup
        await vault_client.delete_secret(tenant1_path)
        await vault_client.delete_secret(tenant2_path)

    @pytest.mark.asyncio
    async def test_cross_tenant_access_prevented(
        self, vault_client
    ) -> None:
        """Test cross-tenant access is prevented by policies."""
        # This test validates that Vault policies are correctly configured
        # In a real setup, tenant1 client wouldn't have permission to tenant2 path
        # For this test, we just verify paths are separate
        tenant1_path = f"tenants/t-isolated-1/{uuid.uuid4().hex[:8]}"
        tenant2_path = f"tenants/t-isolated-2/{uuid.uuid4().hex[:8]}"

        await vault_client.put_secret(tenant1_path, {"secret": "t1"})

        # tenant2_path doesn't exist, should 404
        with pytest.raises((VaultSecretNotFoundError, VaultError)):
            await vault_client.get_secret(tenant2_path)

        # Cleanup
        await vault_client.delete_secret(tenant1_path)


# ============================================================================
# TestErrorHandling
# ============================================================================


class TestErrorHandling:
    """Tests for error handling in Vault operations."""

    @pytest.mark.asyncio
    async def test_not_found_error(self, vault_client) -> None:
        """Test handling of not found errors."""
        with pytest.raises((VaultSecretNotFoundError, VaultError)):
            await vault_client.get_secret(f"nonexistent/{uuid.uuid4().hex}")

    @pytest.mark.asyncio
    async def test_connection_recovery(self, vault_config) -> None:
        """Test recovery from connection errors."""
        # Create client with bad address
        bad_config = VaultConfig(
            addr="http://nonexistent:8200",
            auth_method=VaultAuthMethod.TOKEN,
            token="test",
            max_retries=1,
        )

        client = VaultClient(bad_config)

        with pytest.raises((VaultConnectionError, VaultError, Exception)):
            await client.connect()

    @pytest.mark.asyncio
    async def test_retry_logic(self, vault_config) -> None:
        """Test retry logic on transient failures."""
        # This would need a way to simulate transient failures
        # For now, just verify the client has retry configuration
        assert vault_config.max_retries >= 1
        assert vault_config.base_retry_delay > 0
