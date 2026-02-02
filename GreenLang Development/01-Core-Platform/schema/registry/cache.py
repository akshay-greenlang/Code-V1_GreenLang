# -*- coding: utf-8 -*-
"""
IR Cache Service for GL-FOUND-X-002 (GreenLang Schema Compiler & Validator).

This module implements a thread-safe LRU cache for compiled schema Intermediate
Representations (IR). The cache provides fast access to pre-compiled schemas,
reducing compilation overhead and improving validation throughput.

Key Features:
    - Thread-safe LRU (Least Recently Used) eviction policy
    - TTL-based expiration for automatic cache invalidation
    - Configurable memory limits with size estimation
    - Background warm-up scheduler for popular schemas
    - Comprehensive metrics for monitoring and optimization
    - Cache key based on (schema_id, version, compiler_version)

Design Principles:
    - Thread safety via RLock for all cache operations
    - Lazy eviction (expired entries removed on access or during LRU eviction)
    - Memory-aware caching with configurable limits
    - Provenance tracking via compiler version in cache key

Performance Characteristics:
    - O(1) get/put operations (amortized)
    - O(n) full cache clear
    - O(k) warm-up where k is number of schemas to pre-compile

Example:
    >>> from greenlang.schema.registry.cache import IRCacheService
    >>> from greenlang.schema.models.schema_ref import SchemaRef
    >>>
    >>> cache = IRCacheService(max_size=1000, ttl_seconds=3600)
    >>> ref = SchemaRef(schema_id="emissions/activity", version="1.3.0")
    >>> ir = compiler.compile(schema_source).ir
    >>> cache.put(ref, ir)
    >>> cached_ir = cache.get(ref)

Author: GreenLang Framework Team
Version: 0.1.0
GL-FOUND-X-002: Schema Compiler & Validator - Task 5.4
"""

from __future__ import annotations

import hashlib
import logging
import sys
import threading
import time
from collections import OrderedDict
from datetime import datetime, timedelta
from typing import Any, Callable, Dict, List, Optional, Set, TYPE_CHECKING

from pydantic import BaseModel, ConfigDict, Field

from greenlang.schema.compiler.ir import COMPILER_VERSION, SchemaIR
from greenlang.schema.constants import SCHEMA_CACHE_MAX_SIZE, SCHEMA_CACHE_TTL_SECONDS

if TYPE_CHECKING:
    from greenlang.schema.models.schema_ref import SchemaRef

logger = logging.getLogger(__name__)


# =============================================================================
# CACHE ENTRY MODEL
# =============================================================================


class CacheEntry(BaseModel):
    """
    Entry in the IR cache containing compiled schema and metadata.

    Each cache entry stores the compiled SchemaIR along with timing and
    access information for LRU eviction and TTL expiration.

    Attributes:
        ir: The compiled SchemaIR object
        created_at: Timestamp when entry was added to cache
        last_accessed: Timestamp of most recent access
        access_count: Number of times this entry has been accessed
        size_bytes: Estimated memory size of the IR in bytes

    Example:
        >>> entry = CacheEntry(
        ...     ir=schema_ir,
        ...     created_at=datetime.utcnow(),
        ...     last_accessed=datetime.utcnow(),
        ...     access_count=0,
        ...     size_bytes=1024
        ... )
    """

    ir: SchemaIR = Field(..., description="The compiled SchemaIR object")
    created_at: datetime = Field(
        default_factory=datetime.utcnow,
        description="Timestamp when entry was added to cache"
    )
    last_accessed: datetime = Field(
        default_factory=datetime.utcnow,
        description="Timestamp of most recent access"
    )
    access_count: int = Field(
        default=0,
        ge=0,
        description="Number of times this entry has been accessed"
    )
    size_bytes: int = Field(
        default=0,
        ge=0,
        description="Estimated memory size of the IR in bytes"
    )

    model_config = ConfigDict(
        frozen=False,  # Allow modification for access tracking
        extra="forbid",
        arbitrary_types_allowed=True,
    )

    def touch(self) -> None:
        """Update access time and increment access count."""
        self.last_accessed = datetime.utcnow()
        self.access_count += 1

    def age_seconds(self) -> float:
        """Calculate entry age in seconds."""
        return (datetime.utcnow() - self.created_at).total_seconds()

    def idle_seconds(self) -> float:
        """Calculate time since last access in seconds."""
        return (datetime.utcnow() - self.last_accessed).total_seconds()


# =============================================================================
# CACHE METRICS MODEL
# =============================================================================


class CacheMetrics(BaseModel):
    """
    Comprehensive cache metrics for monitoring and optimization.

    Provides detailed statistics about cache performance including
    hit/miss rates, size, evictions, and entry age distribution.

    Attributes:
        total_entries: Current number of entries in cache
        total_size_bytes: Total memory used by cached entries
        hit_count: Number of successful cache lookups
        miss_count: Number of cache misses
        hit_rate: Ratio of hits to total lookups (0.0 to 1.0)
        eviction_count: Number of entries evicted (LRU or expired)
        oldest_entry_age_seconds: Age of oldest entry in seconds
        newest_entry_age_seconds: Age of newest entry in seconds

    Example:
        >>> metrics = cache.get_metrics()
        >>> print(f"Hit rate: {metrics.hit_rate:.2%}")
        Hit rate: 85.50%
    """

    total_entries: int = Field(
        default=0,
        ge=0,
        description="Current number of entries in cache"
    )
    total_size_bytes: int = Field(
        default=0,
        ge=0,
        description="Total memory used by cached entries"
    )
    hit_count: int = Field(
        default=0,
        ge=0,
        description="Number of successful cache lookups"
    )
    miss_count: int = Field(
        default=0,
        ge=0,
        description="Number of cache misses"
    )
    hit_rate: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Ratio of hits to total lookups"
    )
    eviction_count: int = Field(
        default=0,
        ge=0,
        description="Number of entries evicted"
    )
    oldest_entry_age_seconds: float = Field(
        default=0.0,
        ge=0.0,
        description="Age of oldest entry in seconds"
    )
    newest_entry_age_seconds: float = Field(
        default=0.0,
        ge=0.0,
        description="Age of newest entry in seconds"
    )

    model_config = ConfigDict(frozen=True, extra="forbid")


# =============================================================================
# IR CACHE SERVICE
# =============================================================================


class IRCacheService:
    """
    Thread-safe LRU cache for compiled schema Intermediate Representations.

    This service provides efficient caching of compiled SchemaIR objects
    with LRU eviction, TTL-based expiration, and optional memory limits.

    The cache key is composed of (schema_id, version, compiler_version) to
    ensure cache invalidation when the compiler changes.

    Thread Safety:
        All public methods are thread-safe via RLock. Multiple threads can
        safely read and write to the cache concurrently.

    Eviction Policy:
        - LRU (Least Recently Used): Oldest accessed entries evicted first
        - TTL: Entries expire after ttl_seconds
        - Memory: Optional max_memory_bytes limit

    Attributes:
        max_size: Maximum number of entries in cache
        ttl_seconds: Time-to-live for cache entries
        max_memory_bytes: Optional memory limit in bytes

    Example:
        >>> cache = IRCacheService(max_size=1000, ttl_seconds=3600)
        >>> ref = SchemaRef(schema_id="test", version="1.0.0")
        >>> cache.put(ref, compiled_ir)
        >>> ir = cache.get(ref)  # Returns cached IR or None
    """

    def __init__(
        self,
        max_size: int = SCHEMA_CACHE_MAX_SIZE,
        ttl_seconds: int = SCHEMA_CACHE_TTL_SECONDS,
        max_memory_bytes: Optional[int] = None,
    ) -> None:
        """
        Initialize the IR cache service.

        Args:
            max_size: Maximum number of entries to cache (default: 1000)
            ttl_seconds: Time-to-live for entries in seconds (default: 3600)
            max_memory_bytes: Optional maximum memory limit in bytes.
                If set, evicts entries when memory limit is exceeded.

        Raises:
            ValueError: If max_size < 1 or ttl_seconds < 1
        """
        if max_size < 1:
            raise ValueError(f"max_size must be >= 1, got {max_size}")
        if ttl_seconds < 1:
            raise ValueError(f"ttl_seconds must be >= 1, got {ttl_seconds}")

        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self.max_memory_bytes = max_memory_bytes

        # OrderedDict maintains insertion order for LRU
        self._cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self._lock = threading.RLock()

        # Metrics counters
        self._hit_count = 0
        self._miss_count = 0
        self._eviction_count = 0

        logger.debug(
            f"IRCacheService initialized: max_size={max_size}, "
            f"ttl_seconds={ttl_seconds}, max_memory_bytes={max_memory_bytes}"
        )

    def _make_key(
        self,
        schema_ref: "SchemaRef",
        compiler_version: str = COMPILER_VERSION,
    ) -> str:
        """
        Create cache key from schema reference and compiler version.

        The cache key incorporates the compiler version to ensure automatic
        cache invalidation when the compiler changes.

        Args:
            schema_ref: Schema reference (id, version, variant)
            compiler_version: Version of the compiler (default: current)

        Returns:
            Unique cache key string

        Example:
            >>> key = cache._make_key(ref, "0.1.0")
            >>> # Returns: "emissions/activity:1.3.0::0.1.0"
        """
        variant_part = f":{schema_ref.variant}" if schema_ref.variant else ""
        return f"{schema_ref.schema_id}:{schema_ref.version}{variant_part}:{compiler_version}"

    def get(
        self,
        schema_ref: "SchemaRef",
        compiler_version: str = COMPILER_VERSION,
    ) -> Optional[SchemaIR]:
        """
        Get IR from cache.

        Looks up the schema IR by reference and compiler version. Returns None
        if not found or if the entry has expired. Updates access time and
        moves entry to end of LRU queue on hit.

        Args:
            schema_ref: Schema reference to look up
            compiler_version: Compiler version for cache key (default: current)

        Returns:
            Cached SchemaIR if found and not expired, None otherwise

        Example:
            >>> ir = cache.get(schema_ref)
            >>> if ir is not None:
            ...     # Use cached IR
            ...     pass
        """
        key = self._make_key(schema_ref, compiler_version)

        with self._lock:
            entry = self._cache.get(key)

            if entry is None:
                self._miss_count += 1
                logger.debug(f"Cache miss: {key}")
                return None

            # Check TTL expiration
            if self._is_expired(entry):
                # Remove expired entry
                del self._cache[key]
                self._eviction_count += 1
                self._miss_count += 1
                logger.debug(f"Cache expired: {key}")
                return None

            # Cache hit: update access and move to end (most recently used)
            entry.touch()
            self._cache.move_to_end(key)
            self._hit_count += 1
            logger.debug(f"Cache hit: {key}")
            return entry.ir

    def put(
        self,
        schema_ref: "SchemaRef",
        ir: SchemaIR,
        compiler_version: str = COMPILER_VERSION,
    ) -> None:
        """
        Store IR in cache.

        Adds or updates a cache entry. If the cache is full, evicts the
        least recently used entry first. Also evicts expired entries
        opportunistically.

        Args:
            schema_ref: Schema reference for cache key
            ir: Compiled SchemaIR to cache
            compiler_version: Compiler version for cache key (default: current)

        Example:
            >>> cache.put(schema_ref, compiled_ir)
        """
        key = self._make_key(schema_ref, compiler_version)
        size_bytes = self._estimate_size(ir)

        with self._lock:
            # Remove existing entry if present (will be re-added at end)
            if key in self._cache:
                del self._cache[key]

            # Evict expired entries first
            self._evict_expired()

            # Evict LRU entries if at capacity
            while len(self._cache) >= self.max_size:
                evicted_key = self._evict_lru()
                if evicted_key is None:
                    break

            # Evict entries if memory limit exceeded
            if self.max_memory_bytes is not None:
                while self._get_total_size() + size_bytes > self.max_memory_bytes:
                    evicted_key = self._evict_lru()
                    if evicted_key is None:
                        break

            # Add new entry
            entry = CacheEntry(
                ir=ir,
                created_at=datetime.utcnow(),
                last_accessed=datetime.utcnow(),
                access_count=0,
                size_bytes=size_bytes,
            )
            self._cache[key] = entry
            logger.debug(f"Cache put: {key} (size={size_bytes} bytes)")

    def invalidate(
        self,
        schema_ref: "SchemaRef",
        compiler_version: Optional[str] = None,
    ) -> int:
        """
        Invalidate cache entries for a schema.

        If compiler_version is provided, only invalidates the specific entry.
        If compiler_version is None, invalidates all versions of the schema.

        Args:
            schema_ref: Schema reference to invalidate
            compiler_version: Optional compiler version. If None, invalidates
                all compiler versions for this schema.

        Returns:
            Number of entries removed

        Example:
            >>> # Invalidate specific version
            >>> count = cache.invalidate(ref, "0.1.0")
            >>> # Invalidate all compiler versions
            >>> count = cache.invalidate(ref)
        """
        removed_count = 0

        with self._lock:
            if compiler_version is not None:
                # Invalidate specific entry
                key = self._make_key(schema_ref, compiler_version)
                if key in self._cache:
                    del self._cache[key]
                    removed_count = 1
                    logger.debug(f"Cache invalidated: {key}")
            else:
                # Invalidate all compiler versions for this schema
                variant_part = f":{schema_ref.variant}" if schema_ref.variant else ""
                prefix = f"{schema_ref.schema_id}:{schema_ref.version}{variant_part}:"

                keys_to_remove = [
                    key for key in self._cache.keys()
                    if key.startswith(prefix)
                ]

                for key in keys_to_remove:
                    del self._cache[key]
                    removed_count += 1
                    logger.debug(f"Cache invalidated: {key}")

        return removed_count

    def clear(self) -> int:
        """
        Clear all cache entries.

        Returns:
            Number of entries removed

        Example:
            >>> count = cache.clear()
            >>> print(f"Cleared {count} entries")
        """
        with self._lock:
            count = len(self._cache)
            self._cache.clear()
            logger.info(f"Cache cleared: {count} entries removed")
            return count

    def warm_up(
        self,
        schema_refs: List["SchemaRef"],
        compile_func: Callable[["SchemaRef"], Optional[SchemaIR]],
    ) -> Dict[str, bool]:
        """
        Pre-compile and cache schemas (warm-up).

        Compiles the specified schemas and adds them to the cache. This is
        useful for pre-loading frequently used schemas at startup.

        Args:
            schema_refs: List of schema references to warm up
            compile_func: Function that compiles a schema reference and
                returns the SchemaIR (or None on failure)

        Returns:
            Dictionary mapping schema_id -> success status

        Example:
            >>> def compile_schema(ref):
            ...     return compiler.compile(registry.resolve(ref)).ir
            >>> results = cache.warm_up([ref1, ref2], compile_schema)
            >>> print(results)
            {'emissions/activity': True, 'energy/consumption': False}
        """
        results: Dict[str, bool] = {}

        for schema_ref in schema_refs:
            key = schema_ref.schema_id
            try:
                # Check if already cached
                cached_ir = self.get(schema_ref)
                if cached_ir is not None:
                    results[key] = True
                    logger.debug(f"Warm-up skipped (already cached): {key}")
                    continue

                # Compile and cache
                ir = compile_func(schema_ref)
                if ir is not None:
                    self.put(schema_ref, ir)
                    results[key] = True
                    logger.info(f"Warm-up success: {key}")
                else:
                    results[key] = False
                    logger.warning(f"Warm-up failed (compile returned None): {key}")

            except Exception as e:
                results[key] = False
                logger.error(f"Warm-up failed for {key}: {e}")

        return results

    def get_metrics(self) -> CacheMetrics:
        """
        Get cache metrics for monitoring.

        Returns:
            CacheMetrics object with current statistics

        Example:
            >>> metrics = cache.get_metrics()
            >>> print(f"Hit rate: {metrics.hit_rate:.2%}")
            >>> print(f"Entries: {metrics.total_entries}")
        """
        with self._lock:
            total_entries = len(self._cache)
            total_size = self._get_total_size()

            # Calculate hit rate
            total_requests = self._hit_count + self._miss_count
            hit_rate = self._hit_count / total_requests if total_requests > 0 else 0.0

            # Calculate entry ages
            oldest_age = 0.0
            newest_age = 0.0

            if total_entries > 0:
                ages = [entry.age_seconds() for entry in self._cache.values()]
                oldest_age = max(ages)
                newest_age = min(ages)

            return CacheMetrics(
                total_entries=total_entries,
                total_size_bytes=total_size,
                hit_count=self._hit_count,
                miss_count=self._miss_count,
                hit_rate=hit_rate,
                eviction_count=self._eviction_count,
                oldest_entry_age_seconds=oldest_age,
                newest_entry_age_seconds=newest_age,
            )

    def _evict_expired(self) -> int:
        """
        Evict all expired entries from cache.

        This method is called internally during put operations to
        clean up expired entries opportunistically.

        Returns:
            Number of entries evicted

        Note:
            Must be called with lock held.
        """
        expired_keys = [
            key for key, entry in self._cache.items()
            if self._is_expired(entry)
        ]

        for key in expired_keys:
            del self._cache[key]
            self._eviction_count += 1
            logger.debug(f"Evicted expired entry: {key}")

        return len(expired_keys)

    def _evict_lru(self) -> Optional[str]:
        """
        Evict the least recently used entry.

        Returns:
            The evicted key, or None if cache is empty

        Note:
            Must be called with lock held.
        """
        if not self._cache:
            return None

        # OrderedDict iteration order is insertion order
        # First item is least recently used
        key = next(iter(self._cache))
        del self._cache[key]
        self._eviction_count += 1
        logger.debug(f"LRU eviction: {key}")
        return key

    def _estimate_size(self, ir: SchemaIR) -> int:
        """
        Estimate memory size of an IR in bytes.

        Uses a heuristic based on JSON serialization size plus overhead
        for Python object structures.

        Args:
            ir: SchemaIR to estimate size for

        Returns:
            Estimated size in bytes
        """
        try:
            # Estimate based on JSON representation
            json_str = ir.model_dump_json()
            # Add overhead for Python object structures (roughly 2x JSON size)
            return len(json_str.encode("utf-8")) * 2
        except Exception:
            # Fallback: use sys.getsizeof with recursion estimate
            try:
                base_size = sys.getsizeof(ir)
                # Estimate for nested structures
                return base_size * 10
            except Exception:
                # Ultimate fallback: assume 10KB per IR
                return 10_240

    def _is_expired(self, entry: CacheEntry) -> bool:
        """
        Check if a cache entry has expired based on TTL.

        Args:
            entry: Cache entry to check

        Returns:
            True if entry is expired, False otherwise
        """
        return entry.age_seconds() > self.ttl_seconds

    def _get_total_size(self) -> int:
        """
        Get total size of all cached entries in bytes.

        Returns:
            Total size in bytes

        Note:
            Must be called with lock held.
        """
        return sum(entry.size_bytes for entry in self._cache.values())

    def contains(
        self,
        schema_ref: "SchemaRef",
        compiler_version: str = COMPILER_VERSION,
    ) -> bool:
        """
        Check if a schema is in the cache (and not expired).

        Args:
            schema_ref: Schema reference to check
            compiler_version: Compiler version for cache key

        Returns:
            True if cached and not expired, False otherwise
        """
        key = self._make_key(schema_ref, compiler_version)

        with self._lock:
            entry = self._cache.get(key)
            if entry is None:
                return False
            return not self._is_expired(entry)

    def keys(self) -> List[str]:
        """
        Get all cache keys (for debugging/monitoring).

        Returns:
            List of all cache keys
        """
        with self._lock:
            return list(self._cache.keys())


# =============================================================================
# CACHE WARMUP SCHEDULER
# =============================================================================


class CacheWarmupScheduler:
    """
    Background scheduler for cache warm-up operations.

    This scheduler runs in a background thread and periodically compiles
    and caches popular schemas to ensure they're ready for validation.

    Usage:
        1. Create scheduler with cache reference
        2. Add popular schemas via add_popular_schema()
        3. Start the scheduler with start()
        4. Stop with stop() when shutting down

    Thread Safety:
        The scheduler runs in its own thread and is safe to use
        concurrently with the cache.

    Attributes:
        cache: The IRCacheService to warm up
        interval_seconds: Time between warm-up cycles

    Example:
        >>> scheduler = CacheWarmupScheduler(cache, interval_seconds=300)
        >>> scheduler.add_popular_schema(ref1)
        >>> scheduler.add_popular_schema(ref2)
        >>> scheduler.start()
        >>> # ... application runs ...
        >>> scheduler.stop()
    """

    def __init__(
        self,
        cache: IRCacheService,
        interval_seconds: int = 300,
        compile_func: Optional[Callable[["SchemaRef"], Optional[SchemaIR]]] = None,
    ) -> None:
        """
        Initialize the warm-up scheduler.

        Args:
            cache: The IRCacheService to warm up
            interval_seconds: Time between warm-up cycles (default: 5 minutes)
            compile_func: Function to compile schemas. If None, warm_up_loop
                will skip compilation (useful for testing).

        Raises:
            ValueError: If interval_seconds < 1
        """
        if interval_seconds < 1:
            raise ValueError(f"interval_seconds must be >= 1, got {interval_seconds}")

        self.cache = cache
        self.interval_seconds = interval_seconds
        self._compile_func = compile_func

        self._popular_schemas: Set[str] = set()  # Store as URIs for easy serialization
        self._schema_refs: Dict[str, "SchemaRef"] = {}  # URI -> SchemaRef mapping
        self._lock = threading.Lock()
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

        logger.debug(
            f"CacheWarmupScheduler initialized: interval={interval_seconds}s"
        )

    def add_popular_schema(self, schema_ref: "SchemaRef") -> None:
        """
        Mark a schema as popular for warm-up.

        Popular schemas will be pre-compiled and cached during warm-up cycles.

        Args:
            schema_ref: Schema reference to add to warm-up list

        Example:
            >>> scheduler.add_popular_schema(ref)
        """
        uri = schema_ref.to_uri()
        with self._lock:
            self._popular_schemas.add(uri)
            self._schema_refs[uri] = schema_ref
        logger.debug(f"Added popular schema: {uri}")

    def remove_popular_schema(self, schema_ref: "SchemaRef") -> bool:
        """
        Remove a schema from the popular list.

        Args:
            schema_ref: Schema reference to remove

        Returns:
            True if removed, False if not found
        """
        uri = schema_ref.to_uri()
        with self._lock:
            if uri in self._popular_schemas:
                self._popular_schemas.discard(uri)
                self._schema_refs.pop(uri, None)
                logger.debug(f"Removed popular schema: {uri}")
                return True
            return False

    def get_popular_schemas(self) -> List[str]:
        """
        Get list of popular schema URIs.

        Returns:
            List of schema URIs marked as popular
        """
        with self._lock:
            return list(self._popular_schemas)

    def start(self) -> None:
        """
        Start the background warm-up thread.

        The thread will run until stop() is called, performing warm-up
        cycles at the configured interval.

        Example:
            >>> scheduler.start()
        """
        if self._thread is not None and self._thread.is_alive():
            logger.warning("Warm-up scheduler already running")
            return

        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._warm_up_loop,
            name="IRCacheWarmupScheduler",
            daemon=True,
        )
        self._thread.start()
        logger.info("Warm-up scheduler started")

    def stop(self, timeout: float = 5.0) -> None:
        """
        Stop the background warm-up thread.

        Args:
            timeout: Maximum time to wait for thread to stop (seconds)

        Example:
            >>> scheduler.stop()
        """
        self._stop_event.set()

        if self._thread is not None:
            self._thread.join(timeout=timeout)
            if self._thread.is_alive():
                logger.warning("Warm-up thread did not stop within timeout")
            else:
                logger.info("Warm-up scheduler stopped")

        self._thread = None

    def is_running(self) -> bool:
        """
        Check if the scheduler is running.

        Returns:
            True if the background thread is running
        """
        return self._thread is not None and self._thread.is_alive()

    def trigger_warmup(self) -> Dict[str, bool]:
        """
        Manually trigger a warm-up cycle.

        This runs synchronously in the calling thread, not the background thread.

        Returns:
            Dictionary mapping schema_id -> success status
        """
        return self._do_warmup()

    def _warm_up_loop(self) -> None:
        """
        Background warm-up loop.

        Runs until stop_event is set, performing warm-up cycles
        at the configured interval.
        """
        logger.debug("Warm-up loop started")

        while not self._stop_event.is_set():
            try:
                # Perform warm-up
                results = self._do_warmup()
                if results:
                    success_count = sum(1 for v in results.values() if v)
                    total_count = len(results)
                    logger.info(
                        f"Warm-up cycle complete: {success_count}/{total_count} schemas cached"
                    )

            except Exception as e:
                logger.error(f"Warm-up cycle failed: {e}", exc_info=True)

            # Wait for next cycle or stop signal
            self._stop_event.wait(timeout=self.interval_seconds)

        logger.debug("Warm-up loop stopped")

    def _do_warmup(self) -> Dict[str, bool]:
        """
        Perform a single warm-up cycle.

        Returns:
            Dictionary mapping schema_id -> success status
        """
        if self._compile_func is None:
            logger.debug("Warm-up skipped: no compile function configured")
            return {}

        # Get copy of popular schemas
        with self._lock:
            refs_to_warmup = list(self._schema_refs.values())

        if not refs_to_warmup:
            return {}

        return self.cache.warm_up(refs_to_warmup, self._compile_func)

    def set_compile_func(
        self,
        compile_func: Callable[["SchemaRef"], Optional[SchemaIR]],
    ) -> None:
        """
        Set the compile function for warm-up.

        Args:
            compile_func: Function that compiles a schema reference
        """
        self._compile_func = compile_func


# =============================================================================
# MODULE EXPORTS
# =============================================================================


__all__ = [
    # Models
    "CacheEntry",
    "CacheMetrics",
    # Services
    "IRCacheService",
    "CacheWarmupScheduler",
]
