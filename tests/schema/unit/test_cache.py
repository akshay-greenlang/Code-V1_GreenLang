# -*- coding: utf-8 -*-
"""
Unit tests for the IR Cache Service (GL-FOUND-X-002 Task 5.4).

This module provides comprehensive tests for:
    - IRCacheService: LRU cache with TTL expiration
    - CacheEntry: Cache entry model with access tracking
    - CacheMetrics: Cache metrics model
    - CacheWarmupScheduler: Background cache warm-up

Test Categories:
    - Basic get/put operations
    - LRU eviction policy
    - TTL expiration
    - Memory-based eviction
    - Cache invalidation
    - Thread safety
    - Metrics tracking
    - Background warm-up scheduler

Author: GreenLang Framework Team
Version: 0.1.0
GL-FOUND-X-002: Schema Compiler & Validator - Task 5.4
"""

import threading
import time
from datetime import datetime, timedelta
from typing import Dict, Optional
from unittest.mock import MagicMock, patch

import pytest

from greenlang.schema.compiler.ir import COMPILER_VERSION, PropertyIR, SchemaIR
from greenlang.schema.models.schema_ref import SchemaRef
from greenlang.schema.registry.cache import (
    CacheEntry,
    CacheMetrics,
    CacheWarmupScheduler,
    IRCacheService,
)


# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def sample_schema_ref() -> SchemaRef:
    """Create a sample schema reference for testing."""
    return SchemaRef(schema_id="test/schema", version="1.0.0")


@pytest.fixture
def sample_schema_ref_with_variant() -> SchemaRef:
    """Create a sample schema reference with variant."""
    return SchemaRef(schema_id="test/schema", version="1.0.0", variant="strict")


@pytest.fixture
def another_schema_ref() -> SchemaRef:
    """Create another schema reference for testing."""
    return SchemaRef(schema_id="other/schema", version="2.0.0")


@pytest.fixture
def sample_ir() -> SchemaIR:
    """Create a sample SchemaIR for testing."""
    return SchemaIR(
        schema_id="test/schema",
        version="1.0.0",
        schema_hash="a" * 64,  # Valid 64-char SHA-256
        compiled_at=datetime.utcnow(),
        compiler_version=COMPILER_VERSION,
        properties={},
        required_paths=set(),
    )


@pytest.fixture
def another_ir() -> SchemaIR:
    """Create another SchemaIR for testing."""
    return SchemaIR(
        schema_id="other/schema",
        version="2.0.0",
        schema_hash="b" * 64,
        compiled_at=datetime.utcnow(),
        compiler_version=COMPILER_VERSION,
        properties={},
        required_paths=set(),
    )


@pytest.fixture
def cache_service() -> IRCacheService:
    """Create a fresh cache service for each test."""
    return IRCacheService(max_size=10, ttl_seconds=3600)


@pytest.fixture
def small_cache() -> IRCacheService:
    """Create a small cache for LRU eviction tests."""
    return IRCacheService(max_size=3, ttl_seconds=3600)


@pytest.fixture
def short_ttl_cache() -> IRCacheService:
    """Create a cache with short TTL for expiration tests."""
    return IRCacheService(max_size=10, ttl_seconds=1)  # 1 second TTL


@pytest.fixture
def memory_limited_cache() -> IRCacheService:
    """Create a cache with memory limit."""
    return IRCacheService(
        max_size=100,
        ttl_seconds=3600,
        max_memory_bytes=50_000,  # 50KB limit
    )


# =============================================================================
# CACHE ENTRY TESTS
# =============================================================================


class TestCacheEntry:
    """Tests for the CacheEntry model."""

    def test_create_entry(self, sample_ir):
        """Test creating a cache entry."""
        entry = CacheEntry(
            ir=sample_ir,
            created_at=datetime.utcnow(),
            last_accessed=datetime.utcnow(),
            access_count=0,
            size_bytes=1024,
        )

        assert entry.ir == sample_ir
        assert entry.access_count == 0
        assert entry.size_bytes == 1024

    def test_touch_updates_access(self, sample_ir):
        """Test that touch updates access time and count."""
        entry = CacheEntry(
            ir=sample_ir,
            created_at=datetime.utcnow(),
            last_accessed=datetime.utcnow(),
            access_count=0,
            size_bytes=1024,
        )

        original_access = entry.last_accessed
        original_count = entry.access_count

        time.sleep(0.01)  # Small delay to ensure time difference
        entry.touch()

        assert entry.access_count == original_count + 1
        assert entry.last_accessed > original_access

    def test_age_seconds(self, sample_ir):
        """Test age calculation."""
        past_time = datetime.utcnow() - timedelta(seconds=10)
        entry = CacheEntry(
            ir=sample_ir,
            created_at=past_time,
            last_accessed=past_time,
            access_count=0,
            size_bytes=1024,
        )

        age = entry.age_seconds()
        assert age >= 10.0
        assert age < 15.0  # Allow some tolerance

    def test_idle_seconds(self, sample_ir):
        """Test idle time calculation."""
        now = datetime.utcnow()
        past_time = now - timedelta(seconds=5)
        entry = CacheEntry(
            ir=sample_ir,
            created_at=now,
            last_accessed=past_time,
            access_count=1,
            size_bytes=1024,
        )

        idle = entry.idle_seconds()
        assert idle >= 5.0
        assert idle < 10.0  # Allow some tolerance


# =============================================================================
# CACHE METRICS TESTS
# =============================================================================


class TestCacheMetrics:
    """Tests for the CacheMetrics model."""

    def test_create_metrics(self):
        """Test creating cache metrics."""
        metrics = CacheMetrics(
            total_entries=10,
            total_size_bytes=50000,
            hit_count=80,
            miss_count=20,
            hit_rate=0.8,
            eviction_count=5,
            oldest_entry_age_seconds=3600.0,
            newest_entry_age_seconds=60.0,
        )

        assert metrics.total_entries == 10
        assert metrics.hit_rate == 0.8
        assert metrics.eviction_count == 5

    def test_metrics_is_frozen(self):
        """Test that metrics are immutable."""
        metrics = CacheMetrics(
            total_entries=10,
            total_size_bytes=50000,
            hit_count=80,
            miss_count=20,
            hit_rate=0.8,
            eviction_count=5,
            oldest_entry_age_seconds=3600.0,
            newest_entry_age_seconds=60.0,
        )

        # CacheMetrics is frozen, so this should raise an error
        with pytest.raises(Exception):  # ValidationError or AttributeError
            metrics.total_entries = 20


# =============================================================================
# IR CACHE SERVICE - BASIC OPERATIONS
# =============================================================================


class TestIRCacheServiceBasic:
    """Basic tests for IRCacheService get/put operations."""

    def test_init_with_defaults(self):
        """Test initialization with default values."""
        cache = IRCacheService()
        assert cache.max_size == 1000
        assert cache.ttl_seconds == 3600
        assert cache.max_memory_bytes is None

    def test_init_with_custom_values(self):
        """Test initialization with custom values."""
        cache = IRCacheService(
            max_size=500,
            ttl_seconds=1800,
            max_memory_bytes=100_000,
        )
        assert cache.max_size == 500
        assert cache.ttl_seconds == 1800
        assert cache.max_memory_bytes == 100_000

    def test_init_invalid_max_size(self):
        """Test initialization with invalid max_size."""
        with pytest.raises(ValueError, match="max_size must be >= 1"):
            IRCacheService(max_size=0)

    def test_init_invalid_ttl(self):
        """Test initialization with invalid ttl_seconds."""
        with pytest.raises(ValueError, match="ttl_seconds must be >= 1"):
            IRCacheService(ttl_seconds=0)

    def test_put_and_get(
        self, cache_service, sample_schema_ref, sample_ir
    ):
        """Test basic put and get operations."""
        cache_service.put(sample_schema_ref, sample_ir)
        retrieved = cache_service.get(sample_schema_ref)

        assert retrieved is not None
        assert retrieved.schema_id == sample_ir.schema_id
        assert retrieved.version == sample_ir.version

    def test_get_missing_returns_none(self, cache_service, sample_schema_ref):
        """Test that getting a missing entry returns None."""
        retrieved = cache_service.get(sample_schema_ref)
        assert retrieved is None

    def test_put_overwrites_existing(
        self, cache_service, sample_schema_ref, sample_ir, another_ir
    ):
        """Test that put overwrites existing entries."""
        cache_service.put(sample_schema_ref, sample_ir)

        # Create modified IR with same ref
        modified_ir = SchemaIR(
            schema_id=sample_ir.schema_id,
            version=sample_ir.version,
            schema_hash="c" * 64,  # Different hash
            compiled_at=datetime.utcnow(),
            compiler_version=COMPILER_VERSION,
            properties={
                "/test": PropertyIR(path="/test", type="string", required=True)
            },
            required_paths={"/test"},
        )

        cache_service.put(sample_schema_ref, modified_ir)
        retrieved = cache_service.get(sample_schema_ref)

        assert retrieved is not None
        assert retrieved.schema_hash == "c" * 64

    def test_contains_true_when_cached(
        self, cache_service, sample_schema_ref, sample_ir
    ):
        """Test contains returns True when entry is cached."""
        cache_service.put(sample_schema_ref, sample_ir)
        assert cache_service.contains(sample_schema_ref) is True

    def test_contains_false_when_not_cached(
        self, cache_service, sample_schema_ref
    ):
        """Test contains returns False when entry is not cached."""
        assert cache_service.contains(sample_schema_ref) is False

    def test_keys_returns_all_keys(
        self, cache_service, sample_schema_ref, another_schema_ref,
        sample_ir, another_ir
    ):
        """Test keys returns all cache keys."""
        cache_service.put(sample_schema_ref, sample_ir)
        cache_service.put(another_schema_ref, another_ir)

        keys = cache_service.keys()
        assert len(keys) == 2


# =============================================================================
# IR CACHE SERVICE - CACHE KEY GENERATION
# =============================================================================


class TestCacheKeyGeneration:
    """Tests for cache key generation."""

    def test_make_key_basic(self, cache_service, sample_schema_ref):
        """Test basic cache key generation."""
        key = cache_service._make_key(sample_schema_ref)
        expected = f"test/schema:1.0.0:{COMPILER_VERSION}"
        assert key == expected

    def test_make_key_with_variant(
        self, cache_service, sample_schema_ref_with_variant
    ):
        """Test cache key generation with variant."""
        key = cache_service._make_key(sample_schema_ref_with_variant)
        expected = f"test/schema:1.0.0:strict:{COMPILER_VERSION}"
        assert key == expected

    def test_make_key_different_compiler_version(
        self, cache_service, sample_schema_ref
    ):
        """Test cache key generation with different compiler version."""
        key = cache_service._make_key(sample_schema_ref, "0.2.0")
        expected = "test/schema:1.0.0:0.2.0"
        assert key == expected

    def test_different_schemas_have_different_keys(
        self, cache_service, sample_schema_ref, another_schema_ref
    ):
        """Test that different schemas produce different keys."""
        key1 = cache_service._make_key(sample_schema_ref)
        key2 = cache_service._make_key(another_schema_ref)
        assert key1 != key2


# =============================================================================
# IR CACHE SERVICE - LRU EVICTION
# =============================================================================


class TestLRUEviction:
    """Tests for LRU eviction policy."""

    def test_evicts_oldest_when_full(
        self, small_cache, sample_ir
    ):
        """Test that oldest entry is evicted when cache is full."""
        refs = [
            SchemaRef(schema_id=f"schema{i}", version="1.0.0")
            for i in range(5)
        ]
        irs = [
            SchemaIR(
                schema_id=f"schema{i}",
                version="1.0.0",
                schema_hash=f"{i}" * 64,
                compiled_at=datetime.utcnow(),
                compiler_version=COMPILER_VERSION,
            )
            for i in range(5)
        ]

        # Add 3 entries (cache max is 3)
        for i in range(3):
            small_cache.put(refs[i], irs[i])
            time.sleep(0.01)  # Ensure distinct access times

        # Verify all 3 are cached
        assert small_cache.contains(refs[0])
        assert small_cache.contains(refs[1])
        assert small_cache.contains(refs[2])

        # Add 4th entry, should evict first
        small_cache.put(refs[3], irs[3])

        # First entry should be evicted
        assert not small_cache.contains(refs[0])
        assert small_cache.contains(refs[1])
        assert small_cache.contains(refs[2])
        assert small_cache.contains(refs[3])

    def test_access_updates_lru_order(
        self, small_cache, sample_ir
    ):
        """Test that accessing an entry updates its LRU position."""
        refs = [
            SchemaRef(schema_id=f"schema{i}", version="1.0.0")
            for i in range(4)
        ]
        irs = [
            SchemaIR(
                schema_id=f"schema{i}",
                version="1.0.0",
                schema_hash=f"{i}" * 64,
                compiled_at=datetime.utcnow(),
                compiler_version=COMPILER_VERSION,
            )
            for i in range(4)
        ]

        # Add 3 entries
        for i in range(3):
            small_cache.put(refs[i], irs[i])
            time.sleep(0.01)

        # Access the first entry to move it to end
        small_cache.get(refs[0])

        # Add 4th entry, should evict second (not first)
        small_cache.put(refs[3], irs[3])

        # First should still be cached (was accessed)
        # Second should be evicted
        assert small_cache.contains(refs[0])
        assert not small_cache.contains(refs[1])
        assert small_cache.contains(refs[2])
        assert small_cache.contains(refs[3])


# =============================================================================
# IR CACHE SERVICE - TTL EXPIRATION
# =============================================================================


class TestTTLExpiration:
    """Tests for TTL-based expiration."""

    def test_expired_entry_returns_none(
        self, short_ttl_cache, sample_schema_ref, sample_ir
    ):
        """Test that expired entries return None."""
        short_ttl_cache.put(sample_schema_ref, sample_ir)

        # Entry should be available immediately
        assert short_ttl_cache.get(sample_schema_ref) is not None

        # Wait for TTL to expire
        time.sleep(1.5)

        # Entry should be expired
        assert short_ttl_cache.get(sample_schema_ref) is None

    def test_expired_entry_not_in_contains(
        self, short_ttl_cache, sample_schema_ref, sample_ir
    ):
        """Test that expired entries are not in contains."""
        short_ttl_cache.put(sample_schema_ref, sample_ir)

        assert short_ttl_cache.contains(sample_schema_ref) is True

        time.sleep(1.5)

        assert short_ttl_cache.contains(sample_schema_ref) is False

    def test_expired_entries_evicted_on_put(
        self, short_ttl_cache, sample_schema_ref, another_schema_ref,
        sample_ir, another_ir
    ):
        """Test that expired entries are evicted during put."""
        short_ttl_cache.put(sample_schema_ref, sample_ir)

        time.sleep(1.5)

        # Put another entry, should trigger cleanup
        short_ttl_cache.put(another_schema_ref, another_ir)

        # Original entry should be evicted
        assert not short_ttl_cache.contains(sample_schema_ref)
        # New entry should exist
        assert short_ttl_cache.contains(another_schema_ref)


# =============================================================================
# IR CACHE SERVICE - MEMORY LIMITS
# =============================================================================


class TestMemoryLimits:
    """Tests for memory-based eviction."""

    def test_memory_limit_triggers_eviction(self, memory_limited_cache):
        """Test that memory limit triggers eviction."""
        # Create IRs that will exceed memory limit
        refs = []
        irs = []

        for i in range(10):
            ref = SchemaRef(schema_id=f"schema{i}", version="1.0.0")
            # Create IR with lots of properties to increase size
            # Each PropertyIR requires a path field
            properties = {
                f"/prop{j}": PropertyIR(
                    path=f"/prop{j}",
                    type="string",
                    required=True
                )
                for j in range(100)
            }
            ir = SchemaIR(
                schema_id=f"schema{i}",
                version="1.0.0",
                schema_hash=f"{i}" * 64,
                compiled_at=datetime.utcnow(),
                compiler_version=COMPILER_VERSION,
                properties=properties,
            )
            refs.append(ref)
            irs.append(ir)
            memory_limited_cache.put(ref, ir)

        # Some entries should have been evicted due to memory limit
        cached_count = sum(
            1 for ref in refs if memory_limited_cache.contains(ref)
        )

        # Not all entries should be cached (memory limit exceeded)
        assert cached_count < len(refs)


# =============================================================================
# IR CACHE SERVICE - INVALIDATION
# =============================================================================


class TestCacheInvalidation:
    """Tests for cache invalidation."""

    def test_invalidate_specific_entry(
        self, cache_service, sample_schema_ref, another_schema_ref,
        sample_ir, another_ir
    ):
        """Test invalidating a specific entry."""
        cache_service.put(sample_schema_ref, sample_ir)
        cache_service.put(another_schema_ref, another_ir)

        count = cache_service.invalidate(sample_schema_ref, COMPILER_VERSION)

        assert count == 1
        assert not cache_service.contains(sample_schema_ref)
        assert cache_service.contains(another_schema_ref)

    def test_invalidate_all_versions(
        self, cache_service, sample_schema_ref, sample_ir
    ):
        """Test invalidating all compiler versions of a schema."""
        # Add same schema with different compiler versions
        cache_service.put(sample_schema_ref, sample_ir, "0.1.0")
        cache_service.put(sample_schema_ref, sample_ir, "0.2.0")

        # Invalidate all versions
        count = cache_service.invalidate(sample_schema_ref)

        assert count == 2
        assert not cache_service.contains(sample_schema_ref, "0.1.0")
        assert not cache_service.contains(sample_schema_ref, "0.2.0")

    def test_invalidate_nonexistent_returns_zero(
        self, cache_service, sample_schema_ref
    ):
        """Test invalidating nonexistent entry returns 0."""
        count = cache_service.invalidate(sample_schema_ref)
        assert count == 0

    def test_clear_removes_all_entries(
        self, cache_service, sample_schema_ref, another_schema_ref,
        sample_ir, another_ir
    ):
        """Test that clear removes all entries."""
        cache_service.put(sample_schema_ref, sample_ir)
        cache_service.put(another_schema_ref, another_ir)

        count = cache_service.clear()

        assert count == 2
        assert len(cache_service.keys()) == 0


# =============================================================================
# IR CACHE SERVICE - METRICS
# =============================================================================


class TestCacheMetricsTracking:
    """Tests for cache metrics tracking."""

    def test_hit_count_incremented(
        self, cache_service, sample_schema_ref, sample_ir
    ):
        """Test that hit count is incremented on cache hit."""
        cache_service.put(sample_schema_ref, sample_ir)

        cache_service.get(sample_schema_ref)
        cache_service.get(sample_schema_ref)

        metrics = cache_service.get_metrics()
        assert metrics.hit_count == 2

    def test_miss_count_incremented(
        self, cache_service, sample_schema_ref
    ):
        """Test that miss count is incremented on cache miss."""
        cache_service.get(sample_schema_ref)
        cache_service.get(sample_schema_ref)

        metrics = cache_service.get_metrics()
        assert metrics.miss_count == 2

    def test_hit_rate_calculated(
        self, cache_service, sample_schema_ref, another_schema_ref, sample_ir
    ):
        """Test that hit rate is calculated correctly."""
        cache_service.put(sample_schema_ref, sample_ir)

        # 2 hits
        cache_service.get(sample_schema_ref)
        cache_service.get(sample_schema_ref)

        # 2 misses
        cache_service.get(another_schema_ref)
        cache_service.get(another_schema_ref)

        metrics = cache_service.get_metrics()
        assert metrics.hit_rate == 0.5

    def test_eviction_count_tracked(self, small_cache, sample_ir):
        """Test that eviction count is tracked."""
        refs = [
            SchemaRef(schema_id=f"schema{i}", version="1.0.0")
            for i in range(5)
        ]
        irs = [
            SchemaIR(
                schema_id=f"schema{i}",
                version="1.0.0",
                schema_hash=f"{i}" * 64,
                compiled_at=datetime.utcnow(),
                compiler_version=COMPILER_VERSION,
            )
            for i in range(5)
        ]

        # Add 5 entries to a cache of size 3
        for i in range(5):
            small_cache.put(refs[i], irs[i])

        metrics = small_cache.get_metrics()
        assert metrics.eviction_count == 2  # 5 - 3 = 2 evictions

    def test_total_entries_count(
        self, cache_service, sample_schema_ref, another_schema_ref,
        sample_ir, another_ir
    ):
        """Test that total entries is correct."""
        cache_service.put(sample_schema_ref, sample_ir)
        cache_service.put(another_schema_ref, another_ir)

        metrics = cache_service.get_metrics()
        assert metrics.total_entries == 2

    def test_entry_age_tracking(
        self, cache_service, sample_schema_ref, sample_ir
    ):
        """Test that entry ages are tracked."""
        cache_service.put(sample_schema_ref, sample_ir)
        time.sleep(0.1)

        metrics = cache_service.get_metrics()

        assert metrics.oldest_entry_age_seconds >= 0.1
        assert metrics.newest_entry_age_seconds >= 0.1


# =============================================================================
# IR CACHE SERVICE - WARMUP
# =============================================================================


class TestCacheWarmup:
    """Tests for cache warm-up functionality."""

    def test_warmup_caches_schemas(
        self, cache_service, sample_schema_ref, another_schema_ref,
        sample_ir, another_ir
    ):
        """Test that warm-up caches provided schemas."""
        refs = [sample_schema_ref, another_schema_ref]

        def compile_func(ref: SchemaRef) -> Optional[SchemaIR]:
            if ref.schema_id == "test/schema":
                return sample_ir
            elif ref.schema_id == "other/schema":
                return another_ir
            return None

        results = cache_service.warm_up(refs, compile_func)

        assert results["test/schema"] is True
        assert results["other/schema"] is True
        assert cache_service.contains(sample_schema_ref)
        assert cache_service.contains(another_schema_ref)

    def test_warmup_skips_already_cached(
        self, cache_service, sample_schema_ref, sample_ir
    ):
        """Test that warm-up skips already cached schemas."""
        cache_service.put(sample_schema_ref, sample_ir)

        compile_called = []

        def compile_func(ref: SchemaRef) -> Optional[SchemaIR]:
            compile_called.append(ref.schema_id)
            return sample_ir

        results = cache_service.warm_up([sample_schema_ref], compile_func)

        assert results["test/schema"] is True
        assert len(compile_called) == 0  # Compile was not called

    def test_warmup_handles_compile_failure(
        self, cache_service, sample_schema_ref
    ):
        """Test that warm-up handles compilation failures."""
        def compile_func(ref: SchemaRef) -> Optional[SchemaIR]:
            return None  # Simulate failure

        results = cache_service.warm_up([sample_schema_ref], compile_func)

        assert results["test/schema"] is False

    def test_warmup_handles_compile_exception(
        self, cache_service, sample_schema_ref
    ):
        """Test that warm-up handles compilation exceptions."""
        def compile_func(ref: SchemaRef) -> Optional[SchemaIR]:
            raise RuntimeError("Compilation failed")

        results = cache_service.warm_up([sample_schema_ref], compile_func)

        assert results["test/schema"] is False


# =============================================================================
# IR CACHE SERVICE - THREAD SAFETY
# =============================================================================


class TestThreadSafety:
    """Tests for thread safety."""

    def test_concurrent_reads(self, cache_service, sample_schema_ref, sample_ir):
        """Test concurrent read operations."""
        cache_service.put(sample_schema_ref, sample_ir)

        results = []
        errors = []

        def reader():
            try:
                for _ in range(100):
                    ir = cache_service.get(sample_schema_ref)
                    results.append(ir is not None)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=reader) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        assert all(results)

    def test_concurrent_writes(self, cache_service):
        """Test concurrent write operations."""
        errors = []

        def writer(thread_id: int):
            try:
                for i in range(50):
                    ref = SchemaRef(
                        schema_id=f"schema{thread_id}_{i}",
                        version="1.0.0"
                    )
                    # Ensure hash is exactly 64 chars
                    hash_base = f"{thread_id:02d}{i:04d}"
                    schema_hash = (hash_base * 11)[:64]  # Repeat to get 64 chars
                    ir = SchemaIR(
                        schema_id=f"schema{thread_id}_{i}",
                        version="1.0.0",
                        schema_hash=schema_hash,
                        compiled_at=datetime.utcnow(),
                        compiler_version=COMPILER_VERSION,
                    )
                    cache_service.put(ref, ir)
            except Exception as e:
                errors.append(e)

        threads = [
            threading.Thread(target=writer, args=(i,))
            for i in range(5)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0

    def test_concurrent_read_write(
        self, cache_service, sample_schema_ref, sample_ir
    ):
        """Test concurrent read and write operations."""
        cache_service.put(sample_schema_ref, sample_ir)
        errors = []

        def reader():
            try:
                for _ in range(100):
                    cache_service.get(sample_schema_ref)
            except Exception as e:
                errors.append(e)

        def writer():
            try:
                for i in range(100):
                    ref = SchemaRef(
                        schema_id=f"schema_{i}",
                        version="1.0.0"
                    )
                    # Ensure hash is exactly 64 chars
                    hash_base = f"{i:06d}"
                    schema_hash = (hash_base * 11)[:64]
                    ir = SchemaIR(
                        schema_id=f"schema_{i}",
                        version="1.0.0",
                        schema_hash=schema_hash,
                        compiled_at=datetime.utcnow(),
                        compiler_version=COMPILER_VERSION,
                    )
                    cache_service.put(ref, ir)
            except Exception as e:
                errors.append(e)

        threads = [
            threading.Thread(target=reader) for _ in range(5)
        ] + [
            threading.Thread(target=writer) for _ in range(5)
        ]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0


# =============================================================================
# CACHE WARMUP SCHEDULER TESTS
# =============================================================================


class TestCacheWarmupScheduler:
    """Tests for CacheWarmupScheduler."""

    def test_init_with_defaults(self, cache_service):
        """Test scheduler initialization with defaults."""
        scheduler = CacheWarmupScheduler(cache_service)
        assert scheduler.interval_seconds == 300
        assert scheduler.cache == cache_service

    def test_init_with_custom_interval(self, cache_service):
        """Test scheduler initialization with custom interval."""
        scheduler = CacheWarmupScheduler(cache_service, interval_seconds=60)
        assert scheduler.interval_seconds == 60

    def test_init_invalid_interval(self, cache_service):
        """Test scheduler initialization with invalid interval."""
        with pytest.raises(ValueError, match="interval_seconds must be >= 1"):
            CacheWarmupScheduler(cache_service, interval_seconds=0)

    def test_add_popular_schema(self, cache_service, sample_schema_ref):
        """Test adding a popular schema."""
        scheduler = CacheWarmupScheduler(cache_service)
        scheduler.add_popular_schema(sample_schema_ref)

        schemas = scheduler.get_popular_schemas()
        assert sample_schema_ref.to_uri() in schemas

    def test_remove_popular_schema(self, cache_service, sample_schema_ref):
        """Test removing a popular schema."""
        scheduler = CacheWarmupScheduler(cache_service)
        scheduler.add_popular_schema(sample_schema_ref)

        result = scheduler.remove_popular_schema(sample_schema_ref)

        assert result is True
        assert sample_schema_ref.to_uri() not in scheduler.get_popular_schemas()

    def test_remove_nonexistent_schema(self, cache_service, sample_schema_ref):
        """Test removing a nonexistent schema."""
        scheduler = CacheWarmupScheduler(cache_service)
        result = scheduler.remove_popular_schema(sample_schema_ref)
        assert result is False

    def test_start_stop(self, cache_service):
        """Test starting and stopping the scheduler."""
        scheduler = CacheWarmupScheduler(cache_service, interval_seconds=1)

        scheduler.start()
        assert scheduler.is_running() is True

        scheduler.stop()
        assert scheduler.is_running() is False

    def test_start_twice_no_error(self, cache_service):
        """Test that starting twice doesn't cause errors."""
        scheduler = CacheWarmupScheduler(cache_service, interval_seconds=1)

        scheduler.start()
        scheduler.start()  # Should not raise

        assert scheduler.is_running() is True
        scheduler.stop()

    def test_trigger_warmup(
        self, cache_service, sample_schema_ref, sample_ir
    ):
        """Test manual warm-up trigger."""
        def compile_func(ref: SchemaRef) -> Optional[SchemaIR]:
            return sample_ir

        scheduler = CacheWarmupScheduler(
            cache_service,
            compile_func=compile_func
        )
        scheduler.add_popular_schema(sample_schema_ref)

        results = scheduler.trigger_warmup()

        assert results["test/schema"] is True
        assert cache_service.contains(sample_schema_ref)

    def test_set_compile_func(self, cache_service, sample_schema_ref, sample_ir):
        """Test setting compile function after initialization."""
        scheduler = CacheWarmupScheduler(cache_service)
        scheduler.add_popular_schema(sample_schema_ref)

        # Initially no compile function
        results = scheduler.trigger_warmup()
        assert results == {}

        # Set compile function
        def compile_func(ref: SchemaRef) -> Optional[SchemaIR]:
            return sample_ir

        scheduler.set_compile_func(compile_func)
        results = scheduler.trigger_warmup()

        assert results["test/schema"] is True

    def test_background_warmup_runs(
        self, cache_service, sample_schema_ref, sample_ir
    ):
        """Test that background warm-up runs."""
        compile_called = []

        def compile_func(ref: SchemaRef) -> Optional[SchemaIR]:
            compile_called.append(ref.schema_id)
            return sample_ir

        scheduler = CacheWarmupScheduler(
            cache_service,
            interval_seconds=1,
            compile_func=compile_func
        )
        scheduler.add_popular_schema(sample_schema_ref)

        scheduler.start()
        time.sleep(1.5)  # Wait for at least one cycle
        scheduler.stop()

        assert len(compile_called) >= 1


# =============================================================================
# SIZE ESTIMATION TESTS
# =============================================================================


class TestSizeEstimation:
    """Tests for IR size estimation."""

    def test_estimate_size_basic(self, cache_service, sample_ir):
        """Test basic size estimation."""
        size = cache_service._estimate_size(sample_ir)
        assert size > 0

    def test_estimate_size_larger_ir(self, cache_service):
        """Test size estimation for larger IR."""
        small_ir = SchemaIR(
            schema_id="small",
            version="1.0.0",
            schema_hash="a" * 64,
            compiled_at=datetime.utcnow(),
            compiler_version=COMPILER_VERSION,
        )

        # Create proper PropertyIR objects for the larger IR
        properties = {
            f"/prop{i}": PropertyIR(
                path=f"/prop{i}",
                type="string",
                required=True
            )
            for i in range(100)
        }
        large_ir = SchemaIR(
            schema_id="large",
            version="1.0.0",
            schema_hash="b" * 64,
            compiled_at=datetime.utcnow(),
            compiler_version=COMPILER_VERSION,
            properties=properties,
        )

        small_size = cache_service._estimate_size(small_ir)
        large_size = cache_service._estimate_size(large_ir)

        assert large_size > small_size


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
