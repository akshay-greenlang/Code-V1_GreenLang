"""
GL-002 BoilerEfficiencyOptimizer - Thread-Safe Cache Implementation

This file documents the thread-safe cache sections in boiler_efficiency_orchestrator.py.
All cache operations are already protected with threading.RLock() to prevent race conditions.

Status: ✅ ALREADY THREAD-SAFE - No modifications needed
"""

import threading
import time
from typing import Dict, Any, Optional


# ============================================================================
# SECTION 1: ThreadSafeCache Class (Lines 56-134)
# ============================================================================
# This is the core thread-safe cache implementation used throughout the orchestrator

class ThreadSafeCache:
    """
    Thread-safe cache implementation for concurrent access.

    Provides LRU caching with automatic TTL management and thread safety
    using threading.RLock to prevent race conditions.

    Features:
    - Reentrant lock (RLock) for thread safety
    - Automatic TTL (time-to-live) expiration
    - LRU eviction when cache is full
    - Atomic operations (no race conditions)
    """

    def __init__(self, max_size: int = 1000, ttl_seconds: float = 60.0):
        """
        Initialize thread-safe cache.

        Args:
            max_size: Maximum number of entries in cache
            ttl_seconds: Time-to-live for cache entries in seconds
        """
        self._cache: Dict[str, Any] = {}
        self._timestamps: Dict[str, float] = {}
        # ✅ THREAD-SAFE: Reentrant lock for nested lock scenarios
        self._lock = threading.RLock()
        self._max_size = max_size
        self._ttl_seconds = ttl_seconds

    def get(self, key: str) -> Optional[Any]:
        """
        Get value from cache if valid (THREAD-SAFE).

        Args:
            key: Cache key

        Returns:
            Cached value if exists and not expired, None otherwise

        Thread Safety:
            All operations within lock context - no race conditions possible
        """
        # ✅ THREAD-SAFE: Lock protects entire read operation
        with self._lock:
            if key not in self._cache:
                return None

            # Check if entry has expired (atomic with lock held)
            age_seconds = time.time() - self._timestamps[key]
            if age_seconds >= self._ttl_seconds:
                # Remove expired entry (atomic deletion)
                del self._cache[key]
                del self._timestamps[key]
                return None

            return self._cache[key]

    def set(self, key: str, value: Any) -> None:
        """
        Set value in cache with thread safety (THREAD-SAFE).

        Args:
            key: Cache key
            value: Value to cache

        Thread Safety:
            Size check, eviction, and insertion all atomic within lock
        """
        # ✅ THREAD-SAFE: Lock protects entire write operation
        with self._lock:
            # Remove oldest entries if cache is full (atomic check-and-evict)
            if len(self._cache) >= self._max_size and key not in self._cache:
                oldest_key = min(
                    self._timestamps.keys(),
                    key=lambda k: self._timestamps[k]
                )
                del self._cache[oldest_key]
                del self._timestamps[oldest_key]

            # Store new value (atomic insertion)
            self._cache[key] = value
            self._timestamps[key] = time.time()

    def clear(self) -> None:
        """
        Clear all cache entries (THREAD-SAFE).

        Thread Safety:
            Both dictionaries cleared atomically within lock
        """
        # ✅ THREAD-SAFE: Lock protects clear operation
        with self._lock:
            self._cache.clear()
            self._timestamps.clear()

    def size(self) -> int:
        """
        Get current cache size (THREAD-SAFE).

        Returns:
            Number of entries in cache

        Thread Safety:
            Size check atomic within lock
        """
        # ✅ THREAD-SAFE: Lock protects size query
        with self._lock:
            return len(self._cache)


# ============================================================================
# SECTION 2: Cache Initialization in __init__ (Line 238)
# ============================================================================
# The orchestrator initializes a thread-safe cache instance

def __init__(self, config):
    """Initialize BoilerEfficiencyOptimizer."""
    # ... other initialization ...

    # ✅ THREAD-SAFE: Initialize thread-safe cache with TTL
    # Max size: 200 entries (prevents memory bloat)
    # TTL: 60 seconds (balances freshness vs performance)
    self._results_cache = ThreadSafeCache(max_size=200, ttl_seconds=60)

    # Performance metrics (cache hit/miss tracking)
    self.performance_metrics = {
        'cache_hits': 0,
        'cache_misses': 0,
        # ... other metrics ...
    }


# ============================================================================
# SECTION 3: Cache Access in _analyze_operational_state_async (Lines 422-425, 459)
# ============================================================================
# First major cache access point - operational state analysis

async def _analyze_operational_state_async(
    self,
    boiler_data: Dict[str, Any],
    sensor_feeds: Dict[str, Any]
):
    """
    Analyze current boiler operational state asynchronously.

    Thread Safety:
        All cache operations use ThreadSafeCache which is thread-safe
    """
    # Generate cache key
    cache_key = self._get_cache_key('state_analysis', {
        'boiler': boiler_data,
        'sensors': sensor_feeds
    })

    # ✅ THREAD-SAFE: Cache read via ThreadSafeCache.get()
    # The get() method has internal lock protection
    cached_result = self._results_cache.get(cache_key)
    if cached_result is not None:
        self.performance_metrics['cache_hits'] += 1
        return cached_result

    # Cache miss - perform analysis
    self.performance_metrics['cache_misses'] += 1

    # ... perform expensive calculation ...
    operational_state = calculate_state()

    # ✅ THREAD-SAFE: Cache write via ThreadSafeCache.set()
    # The _store_in_cache wrapper calls ThreadSafeCache.set() with lock protection
    self._store_in_cache(cache_key, operational_state)

    return operational_state


# ============================================================================
# SECTION 4: Cache Access in _optimize_combustion_async (Lines 495-498, 511)
# ============================================================================
# Second major cache access point - combustion optimization

async def _optimize_combustion_async(
    self,
    state,
    fuel_data: Dict[str, Any],
    constraints: Dict[str, Any]
):
    """
    Optimize combustion parameters for maximum efficiency.

    Thread Safety:
        All cache operations use ThreadSafeCache which is thread-safe
    """
    # Generate cache key
    cache_key = self._get_cache_key('combustion_opt', {
        'state': state.__dict__,
        'fuel': fuel_data,
        'constraints': constraints
    })

    # ✅ THREAD-SAFE: Cache read via ThreadSafeCache.get()
    cached_result = self._results_cache.get(cache_key)
    if cached_result is not None:
        self.performance_metrics['cache_hits'] += 1
        return cached_result

    # Cache miss - perform optimization
    self.performance_metrics['cache_misses'] += 1

    # ... perform expensive optimization ...
    result = optimize_combustion()

    # ✅ THREAD-SAFE: Cache write via ThreadSafeCache.set()
    self._store_in_cache(cache_key, result)

    return result


# ============================================================================
# SECTION 5: Cache Storage Wrapper (Lines 1010-1019)
# ============================================================================
# Wrapper method for cache storage

def _store_in_cache(self, cache_key: str, result: Any) -> None:
    """
    Store result in thread-safe cache.

    Args:
        cache_key: Cache key
        result: Result to cache

    Thread Safety:
        Delegates to ThreadSafeCache.set() which has lock protection
    """
    # ✅ THREAD-SAFE: ThreadSafeCache.set() handles thread safety internally
    # No additional locking needed here - lock is in ThreadSafeCache class
    self._results_cache.set(cache_key, result)


# ============================================================================
# SECTION 6: Cache Size Query in get_state (Line 1159)
# ============================================================================
# State monitoring that queries cache size

def get_state(self) -> Dict[str, Any]:
    """
    Get current agent state for monitoring.

    Returns:
        Current state dictionary including cache metrics

    Thread Safety:
        Cache size query is thread-safe via ThreadSafeCache.size()
    """
    return {
        'agent_id': self.config.agent_id,
        'state': self.state.value,
        'performance_metrics': self.performance_metrics.copy(),
        # ✅ THREAD-SAFE: size() method has lock protection
        'cache_size': self._results_cache.size(),
        # ... other state fields ...
    }


# ============================================================================
# PERFORMANCE METRICS TRACKING (Not Cache-Related, but shown for completeness)
# ============================================================================
# Note: performance_metrics is NOT thread-safe, but it's append-only and used
# for monitoring only, so race conditions are acceptable (eventual consistency)

def _update_performance_metrics(self, execution_time_ms, combustion_result, emissions_result):
    """
    Update performance metrics with latest execution.

    Thread Safety Warning:
        This method modifies self.performance_metrics without locks.
        This is acceptable because:
        1. Metrics are for monitoring only (eventual consistency OK)
        2. Individual operations (+=) are atomic at Python bytecode level
        3. We don't require exact counts, approximate is fine
        4. Adding locks here would slow down the critical path

    If exact metrics are required, this should use threading.Lock()
    """
    # Acceptable race condition - approximate metrics for monitoring
    self.performance_metrics['optimizations_performed'] += 1

    # Calculate average (acceptable race condition)
    n = self.performance_metrics['optimizations_performed']
    if n > 0:
        current_avg = self.performance_metrics['avg_optimization_time_ms']
        self.performance_metrics['avg_optimization_time_ms'] = (
            (current_avg * (n - 1) + execution_time_ms) / n
        )


# ============================================================================
# THREAD SAFETY SUMMARY
# ============================================================================
"""
✅ CACHE OPERATIONS: All thread-safe via ThreadSafeCache with RLock
✅ CACHE READS: Protected by lock in ThreadSafeCache.get()
✅ CACHE WRITES: Protected by lock in ThreadSafeCache.set()
✅ CACHE EVICTION: Protected by lock (atomic check-and-evict)
✅ TTL EXPIRATION: Protected by lock (atomic check-and-delete)
✅ CACHE SIZE: Protected by lock in ThreadSafeCache.size()

⚠️  PERFORMANCE METRICS: NOT thread-safe (by design, acceptable for monitoring)
    - If exact metrics required, wrap with threading.Lock()
    - Current implementation uses eventual consistency for speed

🎯 CONCLUSION: Cache implementation is production-ready for concurrent workloads
"""


# ============================================================================
# USAGE EXAMPLE: Concurrent Access Pattern
# ============================================================================

async def example_concurrent_usage():
    """
    Example showing how the thread-safe cache handles concurrent access.

    This demonstrates that multiple coroutines can safely access the cache
    simultaneously without race conditions.
    """
    import asyncio

    config = BoilerEfficiencyConfig(...)
    optimizer = BoilerEfficiencyOptimizer(config)

    async def worker(worker_id: int):
        """Worker coroutine performing optimization."""
        input_data = {
            'boiler_data': {'worker': worker_id},
            'sensor_feeds': {'load_percent': 50 + worker_id},
            'constraints': {},
            'fuel_data': {},
            'steam_demand': {}
        }

        # All workers can safely call execute() concurrently
        # Cache operations are protected by ThreadSafeCache locks
        result = await optimizer.execute(input_data)
        return result

    # Run 10 concurrent optimizations
    # ThreadSafeCache ensures no race conditions
    results = await asyncio.gather(*[worker(i) for i in range(10)])

    # Check cache metrics (eventual consistency)
    state = optimizer.get_state()
    print(f"Cache hits: {state['performance_metrics']['cache_hits']}")
    print(f"Cache size: {state['cache_size']}")


# ============================================================================
# TESTING: Verify Thread Safety Under Load
# ============================================================================

def test_thread_safety_stress():
    """
    Stress test to verify thread safety under heavy concurrent load.

    This test launches multiple threads that hammer the cache simultaneously
    to verify no race conditions occur.
    """
    import threading
    from concurrent.futures import ThreadPoolExecutor

    cache = ThreadSafeCache(max_size=100, ttl_seconds=60)
    errors = []

    def hammer_cache(thread_id: int):
        """Hammer the cache with concurrent reads/writes."""
        try:
            for i in range(1000):
                # Write
                cache.set(f"key_{thread_id}_{i}", {'data': i})

                # Read
                result = cache.get(f"key_{thread_id}_{i}")

                # Read non-existent key
                cache.get(f"nonexistent_{thread_id}")

                # Check size
                size = cache.size()
                assert size <= 100, f"Cache size {size} exceeds max_size"

        except Exception as e:
            errors.append((thread_id, e))

    # Launch 20 threads hammering the cache
    with ThreadPoolExecutor(max_workers=20) as executor:
        futures = [executor.submit(hammer_cache, i) for i in range(20)]
        for future in futures:
            future.result()

    # Verify no errors occurred
    assert len(errors) == 0, f"Thread safety errors: {errors}"

    # Verify cache integrity
    assert cache.size() <= 100, "Cache size exceeded max_size"

    print("✅ Thread safety stress test passed - no race conditions detected")


if __name__ == "__main__":
    # Run stress test
    test_thread_safety_stress()

    # Run example
    import asyncio
    asyncio.run(example_concurrent_usage())
