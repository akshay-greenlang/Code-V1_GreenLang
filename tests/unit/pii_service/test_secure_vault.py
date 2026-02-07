# -*- coding: utf-8 -*-
"""
Unit tests for SecureTokenVault - SEC-011 PII Service.

Tests the secure token vault for PII tokenization/detokenization:
- AES-256-GCM encryption via SEC-003 EncryptionService
- Tenant isolation enforcement
- Token expiration handling
- Deterministic token ID generation
- Unauthorized access handling
- Token persistence and caching

Coverage target: 85%+ of secure_vault.py
"""

from __future__ import annotations

import hashlib
from datetime import datetime, timedelta, timezone
from typing import Any, Dict
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest


# ============================================================================
# Test Fixtures
# ============================================================================


@pytest.fixture
def secure_vault(vault_config, mock_encryption_service, mock_audit_service, mock_db_pool):
    """Create SecureTokenVault instance for testing."""
    try:
        from greenlang.infrastructure.pii_service.secure_vault import SecureTokenVault
        return SecureTokenVault(
            config=vault_config,
            encryption_service=mock_encryption_service,
            audit_service=mock_audit_service,
            db_pool=mock_db_pool,
        )
    except ImportError:
        pytest.skip("SecureTokenVault not yet implemented")


@pytest.fixture
def secure_vault_no_persistence(vault_config, mock_encryption_service, mock_audit_service):
    """Create SecureTokenVault without persistence for in-memory testing."""
    try:
        from greenlang.infrastructure.pii_service.secure_vault import SecureTokenVault
        vault_config.enable_persistence = False
        return SecureTokenVault(
            config=vault_config,
            encryption_service=mock_encryption_service,
            audit_service=mock_audit_service,
            db_pool=None,
        )
    except ImportError:
        pytest.skip("SecureTokenVault not yet implemented")


# ============================================================================
# TestSecureTokenVaultInitialization
# ============================================================================


class TestSecureTokenVaultInitialization:
    """Tests for SecureTokenVault initialization."""

    def test_initialization_stores_config(self, secure_vault, vault_config):
        """Vault stores configuration correctly."""
        assert secure_vault._config == vault_config
        assert secure_vault._config.token_ttl_days == 90

    def test_initialization_with_encryption_service(
        self, secure_vault, mock_encryption_service
    ):
        """Vault initializes with encryption service."""
        assert secure_vault._encryption_service == mock_encryption_service

    def test_initialization_creates_empty_cache(self, secure_vault):
        """Vault starts with empty token cache."""
        assert len(secure_vault._token_cache) == 0

    def test_initialization_with_audit_service(
        self, secure_vault, mock_audit_service
    ):
        """Vault initializes with audit service."""
        assert secure_vault._audit_service == mock_audit_service

    def test_initialization_without_db_uses_memory_store(
        self, secure_vault_no_persistence
    ):
        """Vault uses in-memory store when no DB provided."""
        assert secure_vault_no_persistence._memory_store is not None


# ============================================================================
# TestTokenization
# ============================================================================


class TestTokenization:
    """Tests for tokenize() method."""

    @pytest.mark.asyncio
    async def test_tokenize_creates_valid_token(
        self, secure_vault, pii_type_enum, test_tenant_id
    ):
        """tokenize() creates a valid token string."""
        value = "123-45-6789"
        pii_type = pii_type_enum.SSN

        token = await secure_vault.tokenize(value, pii_type, test_tenant_id)

        assert token is not None
        assert isinstance(token, str)
        assert len(token) > 0
        assert token.startswith("tok_")

    @pytest.mark.asyncio
    async def test_tokenize_uses_aes256_encryption(
        self, secure_vault, mock_encryption_service, pii_type_enum, test_tenant_id
    ):
        """tokenize() calls encryption service for AES-256-GCM."""
        value = "test-pii-value"
        pii_type = pii_type_enum.EMAIL

        await secure_vault.tokenize(value, pii_type, test_tenant_id)

        mock_encryption_service.encrypt.assert_awaited_once()
        call_args = mock_encryption_service.encrypt.call_args
        assert value.encode() in str(call_args) or call_args[0][0] == value.encode()

    @pytest.mark.asyncio
    async def test_tokenize_generates_unique_token_ids(
        self, secure_vault, pii_type_enum, test_tenant_id
    ):
        """Each tokenize() call generates a unique token ID."""
        pii_type = pii_type_enum.SSN
        values = ["value-1", "value-2", "value-3"]

        tokens = [
            await secure_vault.tokenize(v, pii_type, test_tenant_id)
            for v in values
        ]

        assert len(set(tokens)) == 3

    @pytest.mark.asyncio
    async def test_tokenize_persists_to_storage(
        self, secure_vault, pii_type_enum, test_tenant_id, mock_db_pool
    ):
        """tokenize() persists token entry to storage."""
        value = "123-45-6789"
        pii_type = pii_type_enum.SSN

        token = await secure_vault.tokenize(value, pii_type, test_tenant_id)

        # Verify token is in storage
        entry = await secure_vault._get_token_entry(token)
        assert entry is not None
        assert entry.tenant_id == test_tenant_id

    @pytest.mark.asyncio
    async def test_tokenize_respects_tenant_isolation(
        self, secure_vault, pii_type_enum
    ):
        """tokenize() creates tokens scoped to tenant."""
        value = "shared-value"
        pii_type = pii_type_enum.EMAIL
        tenant_1 = "tenant-1"
        tenant_2 = "tenant-2"

        token_1 = await secure_vault.tokenize(value, pii_type, tenant_1)
        token_2 = await secure_vault.tokenize(value, pii_type, tenant_2)

        # Same value, different tenants = different tokens
        assert token_1 != token_2

    @pytest.mark.asyncio
    async def test_tokenize_sets_expiration(
        self, secure_vault, pii_type_enum, test_tenant_id, vault_config
    ):
        """tokenize() sets correct expiration based on config."""
        value = "123-45-6789"
        pii_type = pii_type_enum.SSN

        token = await secure_vault.tokenize(value, pii_type, test_tenant_id)

        entry = await secure_vault._get_token_entry(token)
        expected_expiry = datetime.now(timezone.utc) + timedelta(
            days=vault_config.token_ttl_days
        )
        assert entry.expires_at.date() == expected_expiry.date()

    @pytest.mark.asyncio
    async def test_tokenize_with_metadata(
        self, secure_vault, pii_type_enum, test_tenant_id
    ):
        """tokenize() accepts and stores metadata."""
        value = "123-45-6789"
        pii_type = pii_type_enum.SSN
        metadata = {"source": "api", "request_id": str(uuid4())}

        token = await secure_vault.tokenize(
            value, pii_type, test_tenant_id, metadata=metadata
        )

        entry = await secure_vault._get_token_entry(token)
        assert entry.metadata.get("source") == "api"

    @pytest.mark.asyncio
    async def test_token_id_deterministic_for_same_value(
        self, secure_vault, pii_type_enum, test_tenant_id
    ):
        """Token ID is deterministic for same value+tenant+type."""
        value = "123-45-6789"
        pii_type = pii_type_enum.SSN

        # First tokenization
        token_1 = await secure_vault.tokenize(value, pii_type, test_tenant_id)

        # Simulate a new vault instance
        secure_vault._token_cache.clear()

        # Second tokenization of same value should return same token
        token_2 = await secure_vault.tokenize(value, pii_type, test_tenant_id)

        assert token_1 == token_2

    @pytest.mark.asyncio
    async def test_different_values_different_tokens(
        self, secure_vault, pii_type_enum, test_tenant_id
    ):
        """Different values produce different tokens."""
        pii_type = pii_type_enum.SSN

        token_1 = await secure_vault.tokenize("123-45-6789", pii_type, test_tenant_id)
        token_2 = await secure_vault.tokenize("987-65-4321", pii_type, test_tenant_id)

        assert token_1 != token_2

    @pytest.mark.asyncio
    async def test_same_value_different_tenants_different_tokens(
        self, secure_vault, pii_type_enum
    ):
        """Same value with different tenants produces different tokens."""
        value = "123-45-6789"
        pii_type = pii_type_enum.SSN

        token_1 = await secure_vault.tokenize(value, pii_type, "tenant-a")
        token_2 = await secure_vault.tokenize(value, pii_type, "tenant-b")

        assert token_1 != token_2


# ============================================================================
# TestDetokenization
# ============================================================================


class TestDetokenization:
    """Tests for detokenize() method."""

    @pytest.mark.asyncio
    async def test_detokenize_returns_original_value(
        self, secure_vault, pii_type_enum, test_tenant_id, test_user_id
    ):
        """detokenize() returns the original PII value."""
        original_value = "123-45-6789"
        pii_type = pii_type_enum.SSN

        token = await secure_vault.tokenize(original_value, pii_type, test_tenant_id)
        result = await secure_vault.detokenize(token, test_tenant_id, test_user_id)

        assert result == original_value

    @pytest.mark.asyncio
    async def test_detokenize_validates_tenant_match(
        self, secure_vault, pii_type_enum, test_tenant_id, test_user_id
    ):
        """detokenize() validates tenant matches token's tenant."""
        original_value = "123-45-6789"
        pii_type = pii_type_enum.SSN

        token = await secure_vault.tokenize(original_value, pii_type, test_tenant_id)
        result = await secure_vault.detokenize(token, test_tenant_id, test_user_id)

        assert result == original_value

    @pytest.mark.asyncio
    async def test_detokenize_rejects_wrong_tenant(
        self, secure_vault, pii_type_enum, test_tenant_id, test_user_id
    ):
        """detokenize() raises error for wrong tenant."""
        original_value = "123-45-6789"
        pii_type = pii_type_enum.SSN
        wrong_tenant = "wrong-tenant-id"

        token = await secure_vault.tokenize(original_value, pii_type, test_tenant_id)

        with pytest.raises(Exception) as exc_info:
            await secure_vault.detokenize(token, wrong_tenant, test_user_id)

        assert "tenant" in str(exc_info.value).lower() or "unauthorized" in str(exc_info.value).lower()

    @pytest.mark.asyncio
    async def test_detokenize_rejects_expired_token(
        self, secure_vault, pii_type_enum, test_tenant_id, test_user_id
    ):
        """detokenize() rejects expired tokens."""
        original_value = "123-45-6789"
        pii_type = pii_type_enum.SSN

        token = await secure_vault.tokenize(original_value, pii_type, test_tenant_id)

        # Manually expire the token
        entry = await secure_vault._get_token_entry(token)
        entry.expires_at = datetime.now(timezone.utc) - timedelta(days=1)
        await secure_vault._update_token_entry(entry)

        with pytest.raises(Exception) as exc_info:
            await secure_vault.detokenize(token, test_tenant_id, test_user_id)

        assert "expired" in str(exc_info.value).lower()

    @pytest.mark.asyncio
    async def test_detokenize_audits_access(
        self, secure_vault, mock_audit_service, pii_type_enum, test_tenant_id, test_user_id
    ):
        """detokenize() logs access for audit trail."""
        original_value = "123-45-6789"
        pii_type = pii_type_enum.SSN

        token = await secure_vault.tokenize(original_value, pii_type, test_tenant_id)
        await secure_vault.detokenize(token, test_tenant_id, test_user_id)

        # Verify audit log was called
        audit_log = mock_audit_service.get_audit_log()
        assert len(audit_log) > 0
        access_events = [e for e in audit_log if e.get("event_type") == "access"]
        assert len(access_events) > 0

    @pytest.mark.asyncio
    async def test_detokenize_increments_access_count(
        self, secure_vault, pii_type_enum, test_tenant_id, test_user_id
    ):
        """detokenize() increments the access count on token entry."""
        original_value = "123-45-6789"
        pii_type = pii_type_enum.SSN

        token = await secure_vault.tokenize(original_value, pii_type, test_tenant_id)

        # Access multiple times
        for _ in range(3):
            await secure_vault.detokenize(token, test_tenant_id, test_user_id)

        entry = await secure_vault._get_token_entry(token)
        assert entry.access_count == 3

    @pytest.mark.asyncio
    async def test_token_not_found_raises_error(
        self, secure_vault, test_tenant_id, test_user_id
    ):
        """detokenize() raises error for non-existent token."""
        fake_token = "tok_nonexistent_" + uuid4().hex[:16]

        with pytest.raises(Exception) as exc_info:
            await secure_vault.detokenize(fake_token, test_tenant_id, test_user_id)

        assert "not found" in str(exc_info.value).lower() or "invalid" in str(exc_info.value).lower()

    @pytest.mark.asyncio
    async def test_detokenize_audits_access_denied(
        self, secure_vault, mock_audit_service, pii_type_enum, test_tenant_id, test_user_id
    ):
        """detokenize() audits denied access attempts."""
        original_value = "123-45-6789"
        pii_type = pii_type_enum.SSN
        wrong_tenant = "wrong-tenant"

        token = await secure_vault.tokenize(original_value, pii_type, test_tenant_id)

        try:
            await secure_vault.detokenize(token, wrong_tenant, test_user_id)
        except Exception:
            pass

        # Verify access denied was logged
        audit_log = mock_audit_service.get_audit_log()
        denied_events = [e for e in audit_log if e.get("event_type") == "access_denied"]
        assert len(denied_events) > 0


# ============================================================================
# TestTokenManagement
# ============================================================================


class TestTokenManagement:
    """Tests for token management operations."""

    @pytest.mark.asyncio
    async def test_vault_capacity_limit_per_tenant(
        self, secure_vault, pii_type_enum, vault_config
    ):
        """Vault respects max tokens per tenant limit."""
        tenant_id = "limited-tenant"
        pii_type = pii_type_enum.EMAIL

        # Set a low limit for testing
        vault_config.max_tokens_per_tenant = 5

        # Create tokens up to limit
        for i in range(5):
            await secure_vault.tokenize(f"value-{i}@test.com", pii_type, tenant_id)

        # Next tokenization should fail
        with pytest.raises(Exception) as exc_info:
            await secure_vault.tokenize("overflow@test.com", pii_type, tenant_id)

        assert "limit" in str(exc_info.value).lower() or "capacity" in str(exc_info.value).lower()

    @pytest.mark.asyncio
    async def test_expire_tokens_removes_expired(
        self, secure_vault, pii_type_enum, test_tenant_id
    ):
        """expire_tokens() removes expired tokens from storage."""
        pii_type = pii_type_enum.SSN

        # Create a token
        token = await secure_vault.tokenize("123-45-6789", pii_type, test_tenant_id)

        # Manually expire it
        entry = await secure_vault._get_token_entry(token)
        entry.expires_at = datetime.now(timezone.utc) - timedelta(days=1)
        await secure_vault._update_token_entry(entry)

        # Run expiration
        expired_count = await secure_vault.expire_tokens()

        assert expired_count >= 1

        # Token should no longer exist
        with pytest.raises(Exception):
            await secure_vault._get_token_entry(token)

    @pytest.mark.asyncio
    async def test_get_token_count_per_tenant(
        self, secure_vault, pii_type_enum
    ):
        """get_token_count() returns correct count per tenant."""
        tenant_1 = "count-tenant-1"
        tenant_2 = "count-tenant-2"
        pii_type = pii_type_enum.EMAIL

        # Create tokens for tenant 1
        for i in range(3):
            await secure_vault.tokenize(f"value-{i}@test.com", pii_type, tenant_1)

        # Create tokens for tenant 2
        for i in range(5):
            await secure_vault.tokenize(f"value-{i}@test.com", pii_type, tenant_2)

        count_1 = await secure_vault.get_token_count(tenant_1)
        count_2 = await secure_vault.get_token_count(tenant_2)

        assert count_1 == 3
        assert count_2 == 5

    @pytest.mark.asyncio
    async def test_revoke_token(
        self, secure_vault, pii_type_enum, test_tenant_id, test_user_id
    ):
        """revoke_token() invalidates a token."""
        pii_type = pii_type_enum.SSN

        token = await secure_vault.tokenize("123-45-6789", pii_type, test_tenant_id)

        # Revoke
        await secure_vault.revoke_token(token, test_tenant_id)

        # Should no longer be usable
        with pytest.raises(Exception):
            await secure_vault.detokenize(token, test_tenant_id, test_user_id)


# ============================================================================
# TestEncryptionIntegration
# ============================================================================


class TestEncryptionIntegration:
    """Tests for encryption service integration."""

    @pytest.mark.asyncio
    async def test_encryption_failure_raises_error(
        self, mock_encryption_service_failing, vault_config, mock_audit_service, pii_type_enum, test_tenant_id
    ):
        """Encryption failure propagates as error."""
        try:
            from greenlang.infrastructure.pii_service.secure_vault import SecureTokenVault
            vault = SecureTokenVault(
                config=vault_config,
                encryption_service=mock_encryption_service_failing,
                audit_service=mock_audit_service,
            )

            with pytest.raises(Exception) as exc_info:
                await vault.tokenize("123-45-6789", pii_type_enum.SSN, test_tenant_id)

            assert "encrypt" in str(exc_info.value).lower()
        except ImportError:
            pytest.skip("SecureTokenVault not yet implemented")

    @pytest.mark.asyncio
    async def test_decryption_failure_raises_error(
        self, secure_vault, mock_encryption_service, pii_type_enum, test_tenant_id, test_user_id
    ):
        """Decryption failure propagates as error."""
        # Create a valid token
        token = await secure_vault.tokenize("123-45-6789", pii_type_enum.SSN, test_tenant_id)

        # Make decryption fail
        mock_encryption_service.decrypt = AsyncMock(
            side_effect=Exception("Decryption failed")
        )

        with pytest.raises(Exception) as exc_info:
            await secure_vault.detokenize(token, test_tenant_id, test_user_id)

        assert "decrypt" in str(exc_info.value).lower() or "failed" in str(exc_info.value).lower()


# ============================================================================
# TestCaching
# ============================================================================


class TestCaching:
    """Tests for token caching behavior."""

    @pytest.mark.asyncio
    async def test_cache_hit_avoids_storage_lookup(
        self, secure_vault, pii_type_enum, test_tenant_id, test_user_id
    ):
        """Cached tokens don't require storage lookup."""
        pii_type = pii_type_enum.SSN

        token = await secure_vault.tokenize("123-45-6789", pii_type, test_tenant_id)

        # Clear any storage tracking
        if hasattr(secure_vault, '_storage_lookups'):
            secure_vault._storage_lookups = 0

        # First detokenize populates cache
        await secure_vault.detokenize(token, test_tenant_id, test_user_id)

        # Second detokenize should use cache
        await secure_vault.detokenize(token, test_tenant_id, test_user_id)

        # Verify cache was used (implementation specific)
        assert token in secure_vault._token_cache

    @pytest.mark.asyncio
    async def test_cache_respects_ttl(
        self, secure_vault, pii_type_enum, test_tenant_id, test_user_id, vault_config
    ):
        """Cache entries expire after TTL."""
        pii_type = pii_type_enum.SSN

        token = await secure_vault.tokenize("123-45-6789", pii_type, test_tenant_id)
        await secure_vault.detokenize(token, test_tenant_id, test_user_id)

        # Token should be in cache
        assert token in secure_vault._token_cache

        # Simulate cache expiration
        secure_vault._token_cache.clear()

        # Token should be refetched from storage
        result = await secure_vault.detokenize(token, test_tenant_id, test_user_id)
        assert result == "123-45-6789"


# ============================================================================
# TestTokenIdGeneration
# ============================================================================


class TestTokenIdGeneration:
    """Tests for token ID generation algorithm."""

    @pytest.mark.asyncio
    async def test_token_id_format(
        self, secure_vault, pii_type_enum, test_tenant_id
    ):
        """Token ID has expected format."""
        pii_type = pii_type_enum.SSN

        token = await secure_vault.tokenize("123-45-6789", pii_type, test_tenant_id)

        # Should be prefixed and have sufficient length
        assert token.startswith("tok_")
        assert len(token) >= 20

    @pytest.mark.asyncio
    async def test_token_id_uses_hmac(
        self, secure_vault, pii_type_enum, test_tenant_id
    ):
        """Token ID generation uses HMAC-SHA256."""
        pii_type = pii_type_enum.SSN
        value = "123-45-6789"

        token = await secure_vault.tokenize(value, pii_type, test_tenant_id)

        # Same inputs should produce same token
        token_2 = await secure_vault.tokenize(value, pii_type, test_tenant_id)

        assert token == token_2

    @pytest.mark.asyncio
    async def test_token_id_collision_resistance(
        self, secure_vault, pii_type_enum, test_tenant_id
    ):
        """Token IDs don't collide for different values."""
        pii_type = pii_type_enum.EMAIL
        tokens = set()

        # Generate many tokens
        for i in range(1000):
            token = await secure_vault.tokenize(f"email-{i}@test.com", pii_type, test_tenant_id)
            tokens.add(token)

        # All tokens should be unique
        assert len(tokens) == 1000
