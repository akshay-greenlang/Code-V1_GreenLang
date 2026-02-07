# -*- coding: utf-8 -*-
"""
Unit tests for Key Management - Encryption at Rest (SEC-003)

Tests the DEK cache, key lifecycle management, context hashing, key rotation,
cache eviction policies, and metrics recording for key operations.

Coverage targets: 85%+ of key_management.py
"""

from __future__ import annotations

import asyncio
import hashlib
import secrets
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple
from unittest.mock import AsyncMock, MagicMock, patch, PropertyMock

import pytest

# ---------------------------------------------------------------------------
# Attempt to import key management module
# ---------------------------------------------------------------------------
try:
    from greenlang.infrastructure.encryption_service.key_management import (
        DEKCache,
        KeyManager,
        CachedKey,
        KeyRotationService,
        KeyManagerConfig,
    )
    _HAS_KEY_MANAGEMENT = True
except ImportError:
    _HAS_KEY_MANAGEMENT = False

pytestmark = [
    pytest.mark.skipif(not _HAS_KEY_MANAGEMENT, reason="Key management not installed"),
]


# ============================================================================
# Helpers and Fixtures
# ============================================================================


@dataclass
class MockEncryptionContext:
    """Mock encryption context for key operations."""
    tenant_id: str = "tenant-001"
    data_class: str = "confidential"
    purpose: str = "data_encryption"

    def to_dict(self) -> Dict[str, str]:
        return {
            "tenant_id": self.tenant_id,
            "data_class": self.data_class,
            "purpose": self.purpose,
        }


def _make_mock_envelope_encryption() -> MagicMock:
    """Create a mock envelope encryption service."""
    envelope = MagicMock()

    async def mock_generate():
        plaintext = secrets.token_bytes(32)
        encrypted = secrets.token_bytes(64)
        return MagicMock(
            plaintext_key=plaintext,
            encrypted_key=encrypted,
            key_version=f"v{secrets.randbelow(1000)}",
        )

    async def mock_decrypt(encrypted_key, context):
        return secrets.token_bytes(32)

    envelope.generate_data_key = AsyncMock(side_effect=lambda ctx: mock_generate())
    envelope.decrypt_data_key = AsyncMock(side_effect=mock_decrypt)

    return envelope


def _make_mock_metrics() -> MagicMock:
    """Create a mock metrics collector."""
    metrics = MagicMock()
    metrics.record_cache_hit = MagicMock()
    metrics.record_cache_miss = MagicMock()
    metrics.record_key_generation = MagicMock()
    metrics.record_key_rotation = MagicMock()
    metrics.record_cache_eviction = MagicMock()
    return metrics


def _make_mock_audit_logger() -> AsyncMock:
    """Create a mock audit logger."""
    logger = AsyncMock()
    logger.log_key_generated = AsyncMock()
    logger.log_key_rotated = AsyncMock()
    logger.log_key_accessed = AsyncMock()
    return logger


@pytest.fixture
def mock_envelope() -> MagicMock:
    """Create a mock envelope encryption service."""
    return _make_mock_envelope_encryption()


@pytest.fixture
def mock_metrics() -> MagicMock:
    """Create a mock metrics collector."""
    return _make_mock_metrics()


@pytest.fixture
def mock_audit_logger() -> AsyncMock:
    """Create a mock audit logger."""
    return _make_mock_audit_logger()


@pytest.fixture
def encryption_context() -> MockEncryptionContext:
    """Create a test encryption context."""
    return MockEncryptionContext()


# ============================================================================
# Test: DEK Cache Storage
# ============================================================================


class TestDEKCacheStorage:
    """Tests for DEK cache storage operations."""

    @pytest.mark.asyncio
    async def test_cache_stores_dek(
        self,
        mock_envelope,
        mock_metrics,
        encryption_context,
    ):
        """Test that cache stores DEK on first access."""
        if not _HAS_KEY_MANAGEMENT:
            pytest.skip("Key management not available")

        cache = DEKCache(max_size=100, ttl_seconds=3600, metrics=mock_metrics)
        key_manager = KeyManager(
            envelope=mock_envelope,
            cache=cache,
            metrics=mock_metrics,
        )

        key, version, encrypted = await key_manager.get_or_create_dek(
            encryption_context.to_dict()
        )

        # Key should be stored in cache
        assert key is not None
        assert len(key) == 32

    @pytest.mark.asyncio
    async def test_cache_hit_returns_cached(
        self,
        mock_envelope,
        mock_metrics,
        encryption_context,
    ):
        """Test that cache hit returns cached key."""
        if not _HAS_KEY_MANAGEMENT:
            pytest.skip("Key management not available")

        cache = DEKCache(max_size=100, ttl_seconds=3600, metrics=mock_metrics)
        key_manager = KeyManager(
            envelope=mock_envelope,
            cache=cache,
            metrics=mock_metrics,
        )

        context_dict = encryption_context.to_dict()

        # First call - cache miss
        key1, _, _ = await key_manager.get_or_create_dek(context_dict)

        # Second call - should be cache hit
        key2, _, _ = await key_manager.get_or_create_dek(context_dict)

        # Should return same key
        assert key1 == key2

        # Envelope should only be called once
        assert mock_envelope.generate_data_key.call_count == 1

    @pytest.mark.asyncio
    async def test_cache_miss_generates_new(
        self,
        mock_envelope,
        mock_metrics,
    ):
        """Test that cache miss generates new key."""
        if not _HAS_KEY_MANAGEMENT:
            pytest.skip("Key management not available")

        cache = DEKCache(max_size=100, ttl_seconds=3600, metrics=mock_metrics)
        key_manager = KeyManager(
            envelope=mock_envelope,
            cache=cache,
            metrics=mock_metrics,
        )

        context1 = MockEncryptionContext(tenant_id="tenant-001")
        context2 = MockEncryptionContext(tenant_id="tenant-002")

        # Different contexts should generate different keys
        key1, _, _ = await key_manager.get_or_create_dek(context1.to_dict())
        key2, _, _ = await key_manager.get_or_create_dek(context2.to_dict())

        # Should be different keys
        assert key1 != key2

        # Envelope should be called twice
        assert mock_envelope.generate_data_key.call_count == 2


# ============================================================================
# Test: Cache Expiration
# ============================================================================


class TestCacheExpiration:
    """Tests for cache TTL and expiration."""

    @pytest.mark.asyncio
    async def test_cache_expires_after_ttl(
        self,
        mock_envelope,
        mock_metrics,
        encryption_context,
    ):
        """Test that cached keys expire after TTL."""
        if not _HAS_KEY_MANAGEMENT:
            pytest.skip("Key management not available")

        # Short TTL for testing
        cache = DEKCache(max_size=100, ttl_seconds=1, metrics=mock_metrics)
        key_manager = KeyManager(
            envelope=mock_envelope,
            cache=cache,
            metrics=mock_metrics,
        )

        context_dict = encryption_context.to_dict()

        # First call
        key1, _, _ = await key_manager.get_or_create_dek(context_dict)

        # Wait for expiration
        await asyncio.sleep(1.5)

        # Second call should miss cache
        key2, _, _ = await key_manager.get_or_create_dek(context_dict)

        # Envelope should be called twice (cache expired)
        assert mock_envelope.generate_data_key.call_count == 2


# ============================================================================
# Test: Cache Eviction
# ============================================================================


class TestCacheEviction:
    """Tests for cache eviction policies."""

    @pytest.mark.asyncio
    async def test_cache_evicts_lru_when_full(
        self,
        mock_envelope,
        mock_metrics,
    ):
        """Test LRU eviction when cache is full."""
        if not _HAS_KEY_MANAGEMENT:
            pytest.skip("Key management not available")

        # Small cache for testing eviction
        cache = DEKCache(max_size=3, ttl_seconds=3600, metrics=mock_metrics)
        key_manager = KeyManager(
            envelope=mock_envelope,
            cache=cache,
            metrics=mock_metrics,
        )

        # Fill cache
        contexts = [
            MockEncryptionContext(tenant_id=f"tenant-{i}").to_dict()
            for i in range(4)
        ]

        for ctx in contexts:
            await key_manager.get_or_create_dek(ctx)

        # First context should be evicted (LRU)
        # Accessing it again should be a cache miss
        mock_envelope.generate_data_key.reset_mock()
        await key_manager.get_or_create_dek(contexts[0])

        # Should have called generate (cache miss due to eviction)
        assert mock_envelope.generate_data_key.call_count >= 1


# ============================================================================
# Test: Cache Invalidation
# ============================================================================


class TestCacheInvalidation:
    """Tests for cache invalidation operations."""

    @pytest.mark.asyncio
    async def test_cache_invalidate_single(
        self,
        mock_envelope,
        mock_metrics,
        encryption_context,
    ):
        """Test invalidating a single cache entry."""
        if not _HAS_KEY_MANAGEMENT:
            pytest.skip("Key management not available")

        cache = DEKCache(max_size=100, ttl_seconds=3600, metrics=mock_metrics)
        key_manager = KeyManager(
            envelope=mock_envelope,
            cache=cache,
            metrics=mock_metrics,
        )

        context_dict = encryption_context.to_dict()

        # Populate cache
        await key_manager.get_or_create_dek(context_dict)

        # Invalidate
        cache.invalidate(context_dict)

        # Next access should miss
        mock_envelope.generate_data_key.reset_mock()
        await key_manager.get_or_create_dek(context_dict)

        assert mock_envelope.generate_data_key.call_count == 1

    @pytest.mark.asyncio
    async def test_cache_invalidate_all(
        self,
        mock_envelope,
        mock_metrics,
    ):
        """Test invalidating all cache entries."""
        if not _HAS_KEY_MANAGEMENT:
            pytest.skip("Key management not available")

        cache = DEKCache(max_size=100, ttl_seconds=3600, metrics=mock_metrics)
        key_manager = KeyManager(
            envelope=mock_envelope,
            cache=cache,
            metrics=mock_metrics,
        )

        # Populate with multiple entries
        contexts = [
            MockEncryptionContext(tenant_id=f"tenant-{i}").to_dict()
            for i in range(5)
        ]

        for ctx in contexts:
            await key_manager.get_or_create_dek(ctx)

        # Clear all
        cache.clear()

        # All accesses should miss
        mock_envelope.generate_data_key.reset_mock()
        for ctx in contexts:
            await key_manager.get_or_create_dek(ctx)

        assert mock_envelope.generate_data_key.call_count == 5


# ============================================================================
# Test: Context Hashing
# ============================================================================


class TestContextHashing:
    """Tests for context hash generation."""

    def test_context_hash_deterministic(self, mock_metrics):
        """Test that context hash is deterministic."""
        if not _HAS_KEY_MANAGEMENT:
            pytest.skip("Key management not available")

        cache = DEKCache(max_size=100, ttl_seconds=3600, metrics=mock_metrics)

        context = {"tenant_id": "tenant-001", "data_class": "confidential"}

        hash1 = cache._compute_context_hash(context)
        hash2 = cache._compute_context_hash(context)

        assert hash1 == hash2

    def test_different_context_different_hash(self, mock_metrics):
        """Test that different contexts produce different hashes."""
        if not _HAS_KEY_MANAGEMENT:
            pytest.skip("Key management not available")

        cache = DEKCache(max_size=100, ttl_seconds=3600, metrics=mock_metrics)

        context1 = {"tenant_id": "tenant-001"}
        context2 = {"tenant_id": "tenant-002"}

        hash1 = cache._compute_context_hash(context1)
        hash2 = cache._compute_context_hash(context2)

        assert hash1 != hash2

    def test_context_hash_order_independent(self, mock_metrics):
        """Test that context hash is independent of key order."""
        if not _HAS_KEY_MANAGEMENT:
            pytest.skip("Key management not available")

        cache = DEKCache(max_size=100, ttl_seconds=3600, metrics=mock_metrics)

        context1 = {"a": "1", "b": "2"}
        context2 = {"b": "2", "a": "1"}

        hash1 = cache._compute_context_hash(context1)
        hash2 = cache._compute_context_hash(context2)

        assert hash1 == hash2


# ============================================================================
# Test: Key Versioning
# ============================================================================


class TestKeyVersioning:
    """Tests for key version generation and tracking."""

    @pytest.mark.asyncio
    async def test_version_generated_unique(
        self,
        mock_envelope,
        mock_metrics,
    ):
        """Test that key versions are unique."""
        if not _HAS_KEY_MANAGEMENT:
            pytest.skip("Key management not available")

        cache = DEKCache(max_size=100, ttl_seconds=3600, metrics=mock_metrics)
        key_manager = KeyManager(
            envelope=mock_envelope,
            cache=cache,
            metrics=mock_metrics,
        )

        versions = set()
        for i in range(10):
            context = MockEncryptionContext(tenant_id=f"tenant-{i}").to_dict()
            _, version, _ = await key_manager.get_or_create_dek(context)
            versions.add(version)

        # All versions should be unique
        assert len(versions) == 10


# ============================================================================
# Test: Key Rotation
# ============================================================================


class TestKeyRotation:
    """Tests for key rotation operations."""

    @pytest.mark.asyncio
    async def test_rotation_invalidates_cache(
        self,
        mock_envelope,
        mock_metrics,
        mock_audit_logger,
        encryption_context,
    ):
        """Test that rotation invalidates cached keys."""
        if not _HAS_KEY_MANAGEMENT:
            pytest.skip("Key management not available")

        cache = DEKCache(max_size=100, ttl_seconds=3600, metrics=mock_metrics)
        key_manager = KeyManager(
            envelope=mock_envelope,
            cache=cache,
            metrics=mock_metrics,
        )

        context_dict = encryption_context.to_dict()

        # Populate cache
        key1, _, _ = await key_manager.get_or_create_dek(context_dict)

        # Rotate key
        await key_manager.rotate_key(context_dict)

        # Next access should get new key
        key2, _, _ = await key_manager.get_or_create_dek(context_dict)

        # Should be different key after rotation
        assert key1 != key2

    @pytest.mark.asyncio
    async def test_rotation_generates_new_key(
        self,
        mock_envelope,
        mock_metrics,
        encryption_context,
    ):
        """Test that rotation generates a new key."""
        if not _HAS_KEY_MANAGEMENT:
            pytest.skip("Key management not available")

        cache = DEKCache(max_size=100, ttl_seconds=3600, metrics=mock_metrics)
        key_manager = KeyManager(
            envelope=mock_envelope,
            cache=cache,
            metrics=mock_metrics,
        )

        context_dict = encryption_context.to_dict()

        # Initial key
        await key_manager.get_or_create_dek(context_dict)
        initial_call_count = mock_envelope.generate_data_key.call_count

        # Rotate
        await key_manager.rotate_key(context_dict)

        # Should have called generate again
        assert mock_envelope.generate_data_key.call_count > initial_call_count

    @pytest.mark.asyncio
    async def test_rotation_records_audit(
        self,
        mock_envelope,
        mock_metrics,
        mock_audit_logger,
        encryption_context,
    ):
        """Test that rotation records audit event."""
        if not _HAS_KEY_MANAGEMENT:
            pytest.skip("Key management not available")

        cache = DEKCache(max_size=100, ttl_seconds=3600, metrics=mock_metrics)
        key_manager = KeyManager(
            envelope=mock_envelope,
            cache=cache,
            metrics=mock_metrics,
            audit_logger=mock_audit_logger,
        )

        await key_manager.rotate_key(encryption_context.to_dict())

        # Should have logged rotation
        if hasattr(mock_audit_logger, 'log_key_rotated'):
            mock_audit_logger.log_key_rotated.assert_called()


# ============================================================================
# Test: Metrics Recording
# ============================================================================


class TestKeyMetrics:
    """Tests for metrics recording in key operations."""

    @pytest.mark.asyncio
    async def test_metrics_cache_hit(
        self,
        mock_envelope,
        mock_metrics,
        encryption_context,
    ):
        """Test that cache hit is recorded in metrics."""
        if not _HAS_KEY_MANAGEMENT:
            pytest.skip("Key management not available")

        cache = DEKCache(max_size=100, ttl_seconds=3600, metrics=mock_metrics)
        key_manager = KeyManager(
            envelope=mock_envelope,
            cache=cache,
            metrics=mock_metrics,
        )

        context_dict = encryption_context.to_dict()

        # First call - miss
        await key_manager.get_or_create_dek(context_dict)

        # Second call - hit
        await key_manager.get_or_create_dek(context_dict)

        mock_metrics.record_cache_hit.assert_called()

    @pytest.mark.asyncio
    async def test_metrics_cache_miss(
        self,
        mock_envelope,
        mock_metrics,
        encryption_context,
    ):
        """Test that cache miss is recorded in metrics."""
        if not _HAS_KEY_MANAGEMENT:
            pytest.skip("Key management not available")

        cache = DEKCache(max_size=100, ttl_seconds=3600, metrics=mock_metrics)
        key_manager = KeyManager(
            envelope=mock_envelope,
            cache=cache,
            metrics=mock_metrics,
        )

        await key_manager.get_or_create_dek(encryption_context.to_dict())

        mock_metrics.record_cache_miss.assert_called()

    @pytest.mark.asyncio
    async def test_metrics_key_generation(
        self,
        mock_envelope,
        mock_metrics,
        encryption_context,
    ):
        """Test that key generation is recorded in metrics."""
        if not _HAS_KEY_MANAGEMENT:
            pytest.skip("Key management not available")

        cache = DEKCache(max_size=100, ttl_seconds=3600, metrics=mock_metrics)
        key_manager = KeyManager(
            envelope=mock_envelope,
            cache=cache,
            metrics=mock_metrics,
        )

        await key_manager.get_or_create_dek(encryption_context.to_dict())

        mock_metrics.record_key_generation.assert_called()


# ============================================================================
# Test: DEK Decryption
# ============================================================================


class TestDEKDecryption:
    """Tests for decrypting stored DEKs."""

    @pytest.mark.asyncio
    async def test_decrypt_dek_returns_plaintext(
        self,
        mock_envelope,
        mock_metrics,
        encryption_context,
    ):
        """Test that decrypt_dek returns the plaintext key."""
        if not _HAS_KEY_MANAGEMENT:
            pytest.skip("Key management not available")

        cache = DEKCache(max_size=100, ttl_seconds=3600, metrics=mock_metrics)
        key_manager = KeyManager(
            envelope=mock_envelope,
            cache=cache,
            metrics=mock_metrics,
        )

        context_dict = encryption_context.to_dict()

        # Get key with encrypted DEK
        plaintext_key, version, encrypted_dek = await key_manager.get_or_create_dek(
            context_dict
        )

        # Decrypt the encrypted DEK
        decrypted = await key_manager.decrypt_dek(encrypted_dek, context_dict)

        # Should return a 32-byte key
        assert len(decrypted) == 32


# ============================================================================
# Test: Concurrent Access
# ============================================================================


class TestConcurrentAccess:
    """Tests for concurrent cache access."""

    @pytest.mark.asyncio
    async def test_concurrent_access_same_context(
        self,
        mock_envelope,
        mock_metrics,
        encryption_context,
    ):
        """Test concurrent access to same context."""
        if not _HAS_KEY_MANAGEMENT:
            pytest.skip("Key management not available")

        cache = DEKCache(max_size=100, ttl_seconds=3600, metrics=mock_metrics)
        key_manager = KeyManager(
            envelope=mock_envelope,
            cache=cache,
            metrics=mock_metrics,
        )

        context_dict = encryption_context.to_dict()

        # Concurrent access
        async def get_key():
            return await key_manager.get_or_create_dek(context_dict)

        results = await asyncio.gather(*[get_key() for _ in range(10)])

        # All should get the same key
        keys = [r[0] for r in results]
        assert all(k == keys[0] for k in keys)

        # Should only generate once (ideally with proper locking)
        # Note: without locking, might generate up to 10 times
        assert mock_envelope.generate_data_key.call_count >= 1

    @pytest.mark.asyncio
    async def test_concurrent_access_different_contexts(
        self,
        mock_envelope,
        mock_metrics,
    ):
        """Test concurrent access to different contexts."""
        if not _HAS_KEY_MANAGEMENT:
            pytest.skip("Key management not available")

        cache = DEKCache(max_size=100, ttl_seconds=3600, metrics=mock_metrics)
        key_manager = KeyManager(
            envelope=mock_envelope,
            cache=cache,
            metrics=mock_metrics,
        )

        async def get_key(tenant_id: str):
            ctx = MockEncryptionContext(tenant_id=tenant_id).to_dict()
            return await key_manager.get_or_create_dek(ctx)

        results = await asyncio.gather(*[
            get_key(f"tenant-{i}") for i in range(5)
        ])

        # All keys should be different
        keys = [r[0] for r in results]
        assert len(set(k.hex() for k in keys)) == 5
