"""
GreenLang L2 Redis Cache Implementation

Distributed Redis-based cache with pub/sub invalidation, compression, and high availability.
Designed for multi-instance deployments with cache coherence.

Features:
- Redis async client with connection pooling
- Pub/sub based cache invalidation
- Compression (gzip) for large values
- MessagePack serialization for performance
- Redis Sentinel support for HA
- Comprehensive metrics and monitoring
- Automatic reconnection and circuit breaker

Author: GreenLang Infrastructure Team (TEAM 2)
Date: 2025-11-08
Version: 5.0.0
"""

import asyncio
import gzip
import json
import logging
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Union

try:
    import redis.asyncio as aioredis
    from redis.asyncio import Redis, ConnectionPool
    from redis.asyncio.sentinel import Sentinel
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    aioredis = None
    Redis = None
    ConnectionPool = None
    Sentinel = None

try:
    import msgpack
    MSGPACK_AVAILABLE = True
except ImportError:
    MSGPACK_AVAILABLE = False
    msgpack = None

from .architecture import CacheLayer, CacheLayerConfig

logger = logging.getLogger(__name__)


class CircuitState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"  # Normal operation
    OPEN = "open"      # Failing, reject requests
    HALF_OPEN = "half_open"  # Testing recovery


@dataclass
class CircuitBreaker:
    """
    Circuit breaker for Redis operations.

    Prevents cascading failures by stopping requests to unhealthy Redis.
    """
    failure_threshold: int = 5
    recovery_timeout: float = 60.0  # seconds
    half_open_max_calls: int = 3

    def __post_init__(self):
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.last_failure_time = 0.0
        self.half_open_calls = 0

    def record_success(self) -> None:
        """Record a successful operation."""
        if self.state == CircuitState.HALF_OPEN:
            self.state = CircuitState.CLOSED
            self.failure_count = 0
            self.half_open_calls = 0
            logger.info("Circuit breaker closed (recovered)")

    def record_failure(self) -> None:
        """Record a failed operation."""
        self.failure_count += 1
        self.last_failure_time = time.time()

        if self.failure_count >= self.failure_threshold:
            if self.state != CircuitState.OPEN:
                self.state = CircuitState.OPEN
                logger.warning(
                    f"Circuit breaker opened after {self.failure_count} failures"
                )

    def can_attempt(self) -> bool:
        """Check if operation can be attempted."""
        if self.state == CircuitState.CLOSED:
            return True

        if self.state == CircuitState.OPEN:
            # Check if recovery timeout elapsed
            if time.time() - self.last_failure_time >= self.recovery_timeout:
                self.state = CircuitState.HALF_OPEN
                self.half_open_calls = 0
                logger.info("Circuit breaker half-open (testing recovery)")
                return True
            return False

        if self.state == CircuitState.HALF_OPEN:
            if self.half_open_calls < self.half_open_max_calls:
                self.half_open_calls += 1
                return True
            return False

        return False


@dataclass
class RedisMetrics:
    """
    Metrics for Redis cache operations.

    Tracks hits, misses, latency, errors, and pub/sub events.
    """
    hits: int = 0
    misses: int = 0
    sets: int = 0
    deletes: int = 0
    errors: int = 0
    pubsub_messages: int = 0
    compression_ratio: float = 1.0
    latencies_ms: List[float] = None

    def __post_init__(self):
        if self.latencies_ms is None:
            self.latencies_ms = []

    def record_hit(self, latency_ms: float) -> None:
        """Record cache hit."""
        self.hits += 1
        self._record_latency(latency_ms)

    def record_miss(self) -> None:
        """Record cache miss."""
        self.misses += 1

    def record_set(self, latency_ms: float) -> None:
        """Record cache set."""
        self.sets += 1
        self._record_latency(latency_ms)

    def record_delete(self) -> None:
        """Record cache delete."""
        self.deletes += 1

    def record_error(self) -> None:
        """Record error."""
        self.errors += 1

    def record_pubsub_message(self) -> None:
        """Record pub/sub message."""
        self.pubsub_messages += 1

    def _record_latency(self, latency_ms: float) -> None:
        """Record latency (keep last 10k)."""
        self.latencies_ms.append(latency_ms)
        if len(self.latencies_ms) > 10000:
            self.latencies_ms = self.latencies_ms[-10000:]

    def get_hit_rate(self) -> float:
        """Calculate hit rate."""
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0

    def get_percentile(self, p: int) -> float:
        """Get latency percentile."""
        if not self.latencies_ms:
            return 0.0
        sorted_lat = sorted(self.latencies_ms)
        idx = int(len(sorted_lat) * p / 100)
        return sorted_lat[min(idx, len(sorted_lat) - 1)]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "hits": self.hits,
            "misses": self.misses,
            "sets": self.sets,
            "deletes": self.deletes,
            "errors": self.errors,
            "pubsub_messages": self.pubsub_messages,
            "hit_rate": self.get_hit_rate(),
            "compression_ratio": self.compression_ratio,
            "p50_latency_ms": self.get_percentile(50),
            "p95_latency_ms": self.get_percentile(95),
            "p99_latency_ms": self.get_percentile(99),
        }


class L2RedisCache:
    """
    High-performance distributed Redis cache.

    Features:
    - Connection pooling for performance
    - Pub/sub for cache invalidation
    - Compression for network efficiency
    - MessagePack serialization
    - Circuit breaker for resilience
    - Redis Sentinel support for HA

    Example:
        >>> cache = L2RedisCache(
        ...     host="localhost",
        ...     port=6379,
        ...     pool_size=50
        ... )
        >>> await cache.start()
        >>> await cache.set("key", {"data": "value"}, ttl=3600)
        >>> value = await cache.get("key")
        >>> await cache.stop()
    """

    def __init__(
        self,
        host: str = "localhost",
        port: int = 6379,
        db: int = 0,
        password: Optional[str] = None,
        pool_size: int = 50,
        default_ttl_seconds: int = 3600,
        compression_threshold_bytes: int = 1024,
        use_msgpack: bool = True,
        pubsub_channel: str = "gl:cache:invalidations",
        sentinel_enabled: bool = False,
        sentinel_hosts: Optional[List[str]] = None,
        sentinel_service: str = "mymaster",
        enable_metrics: bool = True
    ):
        """
        Initialize L2 Redis cache.

        Args:
            host: Redis host
            port: Redis port
            db: Redis database number
            password: Optional Redis password
            pool_size: Connection pool size
            default_ttl_seconds: Default TTL for entries
            compression_threshold_bytes: Compress values larger than this
            use_msgpack: Use MessagePack serialization
            pubsub_channel: Pub/sub channel for invalidations
            sentinel_enabled: Use Redis Sentinel for HA
            sentinel_hosts: List of Sentinel hosts
            sentinel_service: Sentinel service name
            enable_metrics: Enable metrics collection
        """
        if not REDIS_AVAILABLE:
            raise ImportError(
                "redis package required for L2 cache. "
                "Install with: pip install redis[hiredis]"
            )

        self._host = host
        self._port = port
        self._db = db
        self._password = password
        self._pool_size = pool_size
        self._default_ttl = default_ttl_seconds
        self._compression_threshold = compression_threshold_bytes
        self._use_msgpack = use_msgpack and MSGPACK_AVAILABLE
        self._pubsub_channel = pubsub_channel
        self._sentinel_enabled = sentinel_enabled
        self._sentinel_hosts = sentinel_hosts or []
        self._sentinel_service = sentinel_service
        self._enable_metrics = enable_metrics

        # Redis clients
        self._redis: Optional[Redis] = None
        self._sentinel: Optional[Sentinel] = None
        self._pool: Optional[ConnectionPool] = None
        self._pubsub = None

        # Circuit breaker
        self._circuit_breaker = CircuitBreaker()

        # Metrics
        self._metrics = RedisMetrics() if enable_metrics else None

        # Pub/sub
        self._pubsub_task: Optional[asyncio.Task] = None
        self._invalidation_callbacks: List[callable] = []
        self._running = False

        logger.info(
            f"Initialized L2 Redis cache: "
            f"host={host}:{port}, pool_size={pool_size}, "
            f"msgpack={self._use_msgpack}, sentinel={sentinel_enabled}"
        )

    async def start(self) -> None:
        """Start Redis connection and pub/sub listener."""
        try:
            if self._sentinel_enabled:
                await self._start_sentinel()
            else:
                await self._start_direct()

            # Start pub/sub listener
            self._running = True
            self._pubsub_task = asyncio.create_task(self._pubsub_listener())

            logger.info("L2 Redis cache started")
        except Exception as e:
            logger.error(f"Failed to start L2 cache: {e}", exc_info=True)
            raise

    async def _start_direct(self) -> None:
        """Start direct Redis connection."""
        self._pool = ConnectionPool(
            host=self._host,
            port=self._port,
            db=self._db,
            password=self._password,
            max_connections=self._pool_size,
            decode_responses=False  # We handle encoding
        )
        self._redis = Redis(connection_pool=self._pool)

        # Test connection
        await self._redis.ping()

    async def _start_sentinel(self) -> None:
        """Start Redis Sentinel connection for HA."""
        sentinel_nodes = []
        for host_port in self._sentinel_hosts:
            if ':' in host_port:
                h, p = host_port.split(':')
                sentinel_nodes.append((h, int(p)))
            else:
                sentinel_nodes.append((host_port, 26379))

        self._sentinel = Sentinel(
            sentinel_nodes,
            password=self._password,
            db=self._db
        )
        self._redis = self._sentinel.master_for(
            self._sentinel_service,
            password=self._password,
            db=self._db
        )

        # Test connection
        await self._redis.ping()

    async def stop(self) -> None:
        """Stop Redis connection and pub/sub listener."""
        self._running = False

        if self._pubsub_task:
            self._pubsub_task.cancel()
            try:
                await self._pubsub_task
            except asyncio.CancelledError:
                pass

        if self._pubsub:
            await self._pubsub.unsubscribe(self._pubsub_channel)
            await self._pubsub.close()

        if self._redis:
            await self._redis.close()

        if self._pool:
            await self._pool.disconnect()

        logger.info("L2 Redis cache stopped")

    @asynccontextmanager
    async def _circuit_protected(self):
        """Context manager for circuit breaker protection."""
        if not self._circuit_breaker.can_attempt():
            raise Exception("Circuit breaker is open")

        try:
            yield
            self._circuit_breaker.record_success()
        except Exception as e:
            self._circuit_breaker.record_failure()
            if self._metrics:
                self._metrics.record_error()
            raise

    async def get(self, key: str) -> Optional[Any]:
        """
        Get value from Redis cache.

        Args:
            key: Cache key

        Returns:
            Cached value or None if not found

        Example:
            >>> value = await cache.get("my_key")
        """
        start_time = time.perf_counter()

        try:
            async with self._circuit_protected():
                raw_value = await self._redis.get(key)

                if raw_value is None:
                    if self._metrics:
                        self._metrics.record_miss()
                    return None

                # Deserialize
                value = self._deserialize(raw_value)

                # Record metrics
                if self._metrics:
                    latency_ms = (time.perf_counter() - start_time) * 1000
                    self._metrics.record_hit(latency_ms)

                return value

        except Exception as e:
            logger.error(f"Error getting key {key}: {e}")
            if self._metrics:
                self._metrics.record_miss()
            return None

    async def set(
        self,
        key: str,
        value: Any,
        ttl: Optional[int] = None
    ) -> bool:
        """
        Set value in Redis cache.

        Args:
            key: Cache key
            value: Value to cache
            ttl: Optional TTL in seconds

        Returns:
            True if successful

        Example:
            >>> await cache.set("my_key", {"data": "value"}, ttl=7200)
        """
        start_time = time.perf_counter()
        ttl_seconds = ttl if ttl is not None else self._default_ttl

        try:
            async with self._circuit_protected():
                # Serialize
                serialized = self._serialize(value)

                # Set with TTL
                await self._redis.setex(key, ttl_seconds, serialized)

                # Record metrics
                if self._metrics:
                    latency_ms = (time.perf_counter() - start_time) * 1000
                    self._metrics.record_set(latency_ms)

                return True

        except Exception as e:
            logger.error(f"Error setting key {key}: {e}", exc_info=True)
            return False

    async def delete(self, key: str) -> bool:
        """
        Delete key from Redis cache.

        Args:
            key: Cache key

        Returns:
            True if deleted

        Example:
            >>> await cache.delete("my_key")
        """
        try:
            async with self._circuit_protected():
                result = await self._redis.delete(key)

                if self._metrics:
                    self._metrics.record_delete()

                # Publish invalidation
                await self._publish_invalidation([key])

                return result > 0

        except Exception as e:
            logger.error(f"Error deleting key {key}: {e}")
            return False

    async def delete_pattern(self, pattern: str) -> int:
        """
        Delete all keys matching pattern.

        Args:
            pattern: Key pattern (e.g., "workflow:*")

        Returns:
            Number of keys deleted

        Example:
            >>> deleted = await cache.delete_pattern("workflow:123:*")
        """
        try:
            async with self._circuit_protected():
                # Find matching keys
                keys = []
                async for key in self._redis.scan_iter(match=pattern):
                    keys.append(key)

                if not keys:
                    return 0

                # Delete in batches
                count = 0
                batch_size = 1000
                for i in range(0, len(keys), batch_size):
                    batch = keys[i:i + batch_size]
                    count += await self._redis.delete(*batch)

                # Publish invalidation
                await self._publish_invalidation(
                    [k.decode() if isinstance(k, bytes) else k for k in keys]
                )

                logger.info(f"Deleted {count} keys matching pattern: {pattern}")
                return count

        except Exception as e:
            logger.error(f"Error deleting pattern {pattern}: {e}")
            return 0

    async def exists(self, key: str) -> bool:
        """Check if key exists."""
        try:
            async with self._circuit_protected():
                return await self._redis.exists(key) > 0
        except Exception:
            return False

    async def clear(self) -> None:
        """Clear all keys (use with caution!)."""
        try:
            async with self._circuit_protected():
                await self._redis.flushdb()
                logger.warning("L2 cache cleared (flushdb)")
        except Exception as e:
            logger.error(f"Error clearing cache: {e}")

    def _serialize(self, value: Any) -> bytes:
        """
        Serialize value with optional compression.

        Args:
            value: Value to serialize

        Returns:
            Serialized bytes
        """
        # Serialize
        if self._use_msgpack:
            data = msgpack.packb(value, use_bin_type=True)
        else:
            data = json.dumps(value, default=str).encode('utf-8')

        # Compress if large enough
        if len(data) >= self._compression_threshold:
            compressed = gzip.compress(data)
            if self._metrics:
                ratio = len(data) / len(compressed) if compressed else 1.0
                self._metrics.compression_ratio = ratio
            # Prefix with marker for compression
            return b'\x1f\x8b' + compressed[2:]  # gzip magic bytes
        else:
            return data

    def _deserialize(self, data: bytes) -> Any:
        """
        Deserialize value with optional decompression.

        Args:
            data: Serialized bytes

        Returns:
            Deserialized value
        """
        # Check for gzip magic bytes
        if data[:2] == b'\x1f\x8b':
            # Decompress
            data = gzip.decompress(data)

        # Deserialize
        if self._use_msgpack:
            return msgpack.unpackb(data, raw=False)
        else:
            return json.loads(data.decode('utf-8'))

    async def _publish_invalidation(self, keys: List[str]) -> None:
        """
        Publish cache invalidation event.

        Args:
            keys: List of keys to invalidate
        """
        try:
            message = json.dumps({
                "keys": keys,
                "timestamp": time.time()
            })
            await self._redis.publish(self._pubsub_channel, message)
        except Exception as e:
            logger.error(f"Error publishing invalidation: {e}")

    async def _pubsub_listener(self) -> None:
        """Background task to listen for invalidation events."""
        try:
            self._pubsub = self._redis.pubsub()
            await self._pubsub.subscribe(self._pubsub_channel)

            while self._running:
                try:
                    message = await self._pubsub.get_message(
                        ignore_subscribe_messages=True,
                        timeout=1.0
                    )

                    if message and message['type'] == 'message':
                        await self._handle_invalidation(message['data'])

                except asyncio.TimeoutError:
                    continue
                except Exception as e:
                    logger.error(f"Error in pub/sub listener: {e}")
                    await asyncio.sleep(1)

        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error(f"Pub/sub listener error: {e}", exc_info=True)

    async def _handle_invalidation(self, data: bytes) -> None:
        """
        Handle invalidation message.

        Args:
            data: Message data
        """
        try:
            message = json.loads(data.decode('utf-8'))
            keys = message.get('keys', [])

            if self._metrics:
                self._metrics.record_pubsub_message()

            # Call registered callbacks
            for callback in self._invalidation_callbacks:
                try:
                    await callback(keys)
                except Exception as e:
                    logger.error(f"Error in invalidation callback: {e}")

        except Exception as e:
            logger.error(f"Error handling invalidation: {e}")

    def register_invalidation_callback(self, callback: callable) -> None:
        """
        Register callback for invalidation events.

        Args:
            callback: Async function to call on invalidation

        Example:
            >>> async def on_invalidate(keys):
            ...     print(f"Invalidated: {keys}")
            >>> cache.register_invalidation_callback(on_invalidate)
        """
        self._invalidation_callbacks.append(callback)

    async def get_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.

        Returns:
            Dictionary with stats
        """
        stats = {
            "circuit_state": self._circuit_breaker.state.value,
            "circuit_failures": self._circuit_breaker.failure_count,
        }

        # Add Redis info
        try:
            info = await self._redis.info()
            stats.update({
                "redis_version": info.get("redis_version"),
                "connected_clients": info.get("connected_clients"),
                "used_memory_mb": info.get("used_memory", 0) / (1024 * 1024),
                "keyspace_hits": info.get("keyspace_hits", 0),
                "keyspace_misses": info.get("keyspace_misses", 0),
            })
        except Exception as e:
            logger.error(f"Error getting Redis info: {e}")

        # Add metrics
        if self._metrics:
            stats.update(self._metrics.to_dict())

        return stats
