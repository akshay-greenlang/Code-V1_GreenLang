"""
Redis-backed Distributed Rate Limiter for GreenLang

This module provides distributed rate limiting using Redis as the backend.
Supports multiple rate limiting strategies with atomic Lua scripts for
consistency across multiple application instances.

Features:
- Redis-backed token bucket algorithm
- Redis-backed sliding window log
- Redis-backed fixed window counter
- Redis-backed leaky bucket
- Lua scripts for atomic operations
- Automatic fallback to in-memory on Redis unavailable
- Connection pooling
- TTL-based key expiration

Example:
    >>> config = RedisRateLimiterConfig(redis_url="redis://localhost:6379")
    >>> limiter = RedisRateLimiter(config)
    >>> if await limiter.acquire("user-123"):
    ...     process_request()
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

from pydantic import BaseModel, Field

from greenlang.infrastructure.api.rate_limiter import (
    RateLimitInfo,
    RateLimitScope,
    RateLimitStrategy,
    RateLimiter,
    RateLimiterConfig,
)

logger = logging.getLogger(__name__)

# Try to import redis
try:
    import redis.asyncio as aioredis
    from redis.asyncio.connection import ConnectionPool
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    aioredis = None
    ConnectionPool = None


# ============================================================================
# Lua Scripts for Atomic Operations
# ============================================================================

# Token Bucket Lua Script
# KEYS[1] = bucket key for tokens
# KEYS[2] = bucket key for last_update timestamp
# ARGV[1] = bucket capacity
# ARGV[2] = fill rate (tokens per second)
# ARGV[3] = tokens to acquire
# ARGV[4] = current timestamp (seconds with decimals)
# ARGV[5] = TTL in seconds
# Returns: [success (0 or 1), current_tokens, time_until_available]
TOKEN_BUCKET_SCRIPT = """
local tokens_key = KEYS[1]
local timestamp_key = KEYS[2]
local capacity = tonumber(ARGV[1])
local fill_rate = tonumber(ARGV[2])
local requested = tonumber(ARGV[3])
local now = tonumber(ARGV[4])
local ttl = tonumber(ARGV[5])

-- Get current state
local current_tokens = tonumber(redis.call('GET', tokens_key)) or capacity
local last_update = tonumber(redis.call('GET', timestamp_key)) or now

-- Calculate tokens to add based on elapsed time
local elapsed = math.max(0, now - last_update)
local tokens_to_add = elapsed * fill_rate
local new_tokens = math.min(capacity, current_tokens + tokens_to_add)

-- Update timestamp
redis.call('SET', timestamp_key, tostring(now), 'EX', ttl)

-- Check if we can acquire
if new_tokens >= requested then
    new_tokens = new_tokens - requested
    redis.call('SET', tokens_key, tostring(new_tokens), 'EX', ttl)
    return {1, new_tokens, 0}
else
    -- Not enough tokens
    redis.call('SET', tokens_key, tostring(new_tokens), 'EX', ttl)
    local tokens_needed = requested - new_tokens
    local time_until_available = tokens_needed / fill_rate
    return {0, new_tokens, time_until_available}
end
"""

# Sliding Window Log Lua Script
# KEYS[1] = window key (sorted set)
# ARGV[1] = window size in seconds
# ARGV[2] = max requests
# ARGV[3] = current timestamp
# ARGV[4] = TTL in seconds
# Returns: [success (0 or 1), remaining, oldest_timestamp]
SLIDING_WINDOW_SCRIPT = """
local window_key = KEYS[1]
local window_size = tonumber(ARGV[1])
local max_requests = tonumber(ARGV[2])
local now = tonumber(ARGV[3])
local ttl = tonumber(ARGV[4])

-- Remove expired entries
local window_start = now - window_size
redis.call('ZREMRANGEBYSCORE', window_key, '-inf', window_start)

-- Count current requests
local current_count = redis.call('ZCARD', window_key)

-- Check if we can add
if current_count < max_requests then
    -- Add request with current timestamp as score
    redis.call('ZADD', window_key, now, tostring(now) .. ':' .. tostring(math.random(1000000)))
    redis.call('EXPIRE', window_key, ttl)
    local remaining = max_requests - current_count - 1
    local oldest = redis.call('ZRANGE', window_key, 0, 0, 'WITHSCORES')
    local oldest_ts = oldest[2] or now
    return {1, remaining, oldest_ts}
else
    -- Rate limited
    local oldest = redis.call('ZRANGE', window_key, 0, 0, 'WITHSCORES')
    local oldest_ts = oldest[2] or now
    return {0, 0, oldest_ts}
end
"""

# Fixed Window Counter Lua Script
# KEYS[1] = counter key
# ARGV[1] = window size in seconds
# ARGV[2] = max requests
# ARGV[3] = current timestamp
# ARGV[4] = count to add
# Returns: [success (0 or 1), remaining, window_reset_at]
FIXED_WINDOW_SCRIPT = """
local counter_key = KEYS[1]
local window_size = tonumber(ARGV[1])
local max_requests = tonumber(ARGV[2])
local now = tonumber(ARGV[3])
local count = tonumber(ARGV[4])

-- Calculate window boundaries
local window_start = math.floor(now / window_size) * window_size
local window_key = counter_key .. ':' .. tostring(window_start)
local window_reset = window_start + window_size

-- Get current count
local current = tonumber(redis.call('GET', window_key)) or 0

-- Check if we can increment
if current + count <= max_requests then
    redis.call('INCRBY', window_key, count)
    -- Set expiry slightly after window end to ensure cleanup
    redis.call('EXPIREAT', window_key, math.ceil(window_reset) + 1)
    local remaining = max_requests - current - count
    return {1, remaining, window_reset}
else
    -- Rate limited
    return {0, math.max(0, max_requests - current), window_reset}
end
"""

# Leaky Bucket Lua Script
# KEYS[1] = water level key
# KEYS[2] = last leak timestamp key
# ARGV[1] = bucket capacity
# ARGV[2] = leak rate (requests per second)
# ARGV[3] = amount to add
# ARGV[4] = current timestamp
# ARGV[5] = TTL in seconds
# Returns: [success (0 or 1), current_level, time_until_available]
LEAKY_BUCKET_SCRIPT = """
local level_key = KEYS[1]
local timestamp_key = KEYS[2]
local capacity = tonumber(ARGV[1])
local leak_rate = tonumber(ARGV[2])
local amount = tonumber(ARGV[3])
local now = tonumber(ARGV[4])
local ttl = tonumber(ARGV[5])

-- Get current state
local current_level = tonumber(redis.call('GET', level_key)) or 0
local last_leak = tonumber(redis.call('GET', timestamp_key)) or now

-- Calculate leaked amount
local elapsed = math.max(0, now - last_leak)
local leaked = elapsed * leak_rate
local new_level = math.max(0, current_level - leaked)

-- Update timestamp
redis.call('SET', timestamp_key, tostring(now), 'EX', ttl)

-- Check if we can add
if new_level + amount <= capacity then
    new_level = new_level + amount
    redis.call('SET', level_key, tostring(new_level), 'EX', ttl)
    return {1, new_level, 0}
else
    -- Bucket would overflow
    redis.call('SET', level_key, tostring(new_level), 'EX', ttl)
    local excess = (new_level + amount) - capacity
    local time_until_available = excess / leak_rate
    return {0, new_level, time_until_available}
end
"""


@dataclass
class RedisRateLimiterConfig:
    """Configuration for Redis-backed rate limiter."""

    # Rate limiting strategy
    strategy: RateLimitStrategy = RateLimitStrategy.TOKEN_BUCKET
    scope: RateLimitScope = RateLimitScope.PER_CLIENT

    # Token bucket / Leaky bucket settings
    tokens_per_second: float = 10.0
    bucket_size: int = 100

    # Window settings (sliding and fixed)
    window_size_seconds: int = 60
    max_requests: int = 100

    # Redis connection
    redis_url: str = "redis://localhost:6379"
    redis_prefix: str = "ratelimit:"
    redis_db: int = 0

    # Connection pool settings
    pool_min_connections: int = 1
    pool_max_connections: int = 10
    connection_timeout: float = 5.0
    socket_timeout: float = 5.0

    # TTL for Redis keys (should be > window_size)
    key_ttl_seconds: int = 3600

    # Fallback behavior
    fallback_to_memory: bool = True

    # Headers
    enable_headers: bool = True
    limit_header: str = "X-RateLimit-Limit"
    remaining_header: str = "X-RateLimit-Remaining"
    reset_header: str = "X-RateLimit-Reset"

    # Client identification
    client_id_header: str = "X-API-Key"
    use_ip_fallback: bool = True


class RedisRateLimiter:
    """
    Redis-backed distributed rate limiter.

    Provides distributed rate limiting across multiple application instances
    using Redis as the coordination backend. Uses Lua scripts for atomic
    operations to ensure consistency.

    Attributes:
        config: Redis rate limiter configuration
        redis: Redis client instance
        fallback_limiter: In-memory fallback limiter

    Example:
        >>> config = RedisRateLimiterConfig(
        ...     redis_url="redis://localhost:6379",
        ...     strategy=RateLimitStrategy.TOKEN_BUCKET,
        ...     tokens_per_second=10,
        ...     bucket_size=100
        ... )
        >>> limiter = RedisRateLimiter(config)
        >>> await limiter.connect()
        >>> if await limiter.acquire("client-123"):
        ...     process_request()
        >>> await limiter.close()
    """

    def __init__(self, config: Optional[RedisRateLimiterConfig] = None):
        """
        Initialize Redis rate limiter.

        Args:
            config: Redis rate limiter configuration
        """
        if not REDIS_AVAILABLE:
            raise ImportError(
                "redis package not installed. "
                "Install with: pip install redis[hiredis]"
            )

        self.config = config or RedisRateLimiterConfig()
        self._redis: Optional[aioredis.Redis] = None
        self._pool: Optional[ConnectionPool] = None
        self._connected: bool = False
        self._lock = asyncio.Lock()

        # Lua script SHAs (cached after first load)
        self._token_bucket_sha: Optional[str] = None
        self._sliding_window_sha: Optional[str] = None
        self._fixed_window_sha: Optional[str] = None
        self._leaky_bucket_sha: Optional[str] = None

        # Fallback limiter (in-memory)
        self._fallback_limiter: Optional[RateLimiter] = None
        if self.config.fallback_to_memory:
            fallback_config = RateLimiterConfig(
                strategy=self.config.strategy,
                scope=self.config.scope,
                tokens_per_second=self.config.tokens_per_second,
                bucket_size=self.config.bucket_size,
                window_size_seconds=self.config.window_size_seconds,
                max_requests=self.config.max_requests,
            )
            self._fallback_limiter = RateLimiter(fallback_config)

        logger.info(
            f"RedisRateLimiter initialized: {self.config.strategy.value}, "
            f"redis_url={self.config.redis_url}"
        )

    async def connect(self) -> bool:
        """
        Connect to Redis.

        Returns:
            True if connected successfully
        """
        if self._connected:
            return True

        async with self._lock:
            if self._connected:
                return True

            try:
                # Create connection pool
                self._pool = ConnectionPool.from_url(
                    self.config.redis_url,
                    db=self.config.redis_db,
                    max_connections=self.config.pool_max_connections,
                    socket_connect_timeout=self.config.connection_timeout,
                    socket_timeout=self.config.socket_timeout,
                )

                # Create Redis client
                self._redis = aioredis.Redis(connection_pool=self._pool)

                # Test connection
                await self._redis.ping()

                # Load Lua scripts
                await self._load_scripts()

                self._connected = True
                logger.info("Redis connection established")
                return True

            except Exception as e:
                logger.error(f"Failed to connect to Redis: {e}")
                self._connected = False
                return False

    async def _load_scripts(self) -> None:
        """Load Lua scripts into Redis."""
        if not self._redis:
            return

        try:
            self._token_bucket_sha = await self._redis.script_load(TOKEN_BUCKET_SCRIPT)
            self._sliding_window_sha = await self._redis.script_load(SLIDING_WINDOW_SCRIPT)
            self._fixed_window_sha = await self._redis.script_load(FIXED_WINDOW_SCRIPT)
            self._leaky_bucket_sha = await self._redis.script_load(LEAKY_BUCKET_SCRIPT)
            logger.debug("Lua scripts loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load Lua scripts: {e}")
            raise

    async def close(self) -> None:
        """Close Redis connection."""
        async with self._lock:
            if self._redis:
                await self._redis.close()
                self._redis = None
            if self._pool:
                await self._pool.disconnect()
                self._pool = None
            self._connected = False
            logger.info("Redis connection closed")

    def _get_key(
        self,
        client_id: Optional[str] = None,
        route: Optional[str] = None
    ) -> str:
        """
        Get rate limit key based on scope.

        Args:
            client_id: Client identifier
            route: Route path

        Returns:
            Redis key string
        """
        prefix = self.config.redis_prefix

        if self.config.scope == RateLimitScope.GLOBAL:
            return f"{prefix}global"
        elif self.config.scope == RateLimitScope.PER_CLIENT:
            return f"{prefix}client:{client_id or 'unknown'}"
        elif self.config.scope == RateLimitScope.PER_ROUTE:
            return f"{prefix}route:{route or 'default'}"
        else:  # PER_CLIENT_ROUTE
            return f"{prefix}client:{client_id or 'unknown'}:route:{route or 'default'}"

    async def _use_fallback(
        self,
        operation: str,
        client_id: Optional[str] = None,
        route: Optional[str] = None,
        tokens: int = 1
    ) -> Union[bool, RateLimitInfo]:
        """
        Use fallback in-memory limiter.

        Args:
            operation: 'acquire' or 'check'
            client_id: Client identifier
            route: Route path
            tokens: Tokens to acquire

        Returns:
            Result from fallback limiter
        """
        if not self._fallback_limiter:
            if operation == 'acquire':
                logger.warning("No fallback limiter, allowing request")
                return True
            else:
                return RateLimitInfo(
                    limit=self.config.max_requests,
                    remaining=self.config.max_requests,
                    reset_at=datetime.utcnow(),
                )

        if operation == 'acquire':
            return await self._fallback_limiter.acquire(client_id, route, tokens)
        else:
            return await self._fallback_limiter.check(client_id, route)

    async def acquire(
        self,
        client_id: Optional[str] = None,
        route: Optional[str] = None,
        tokens: int = 1
    ) -> bool:
        """
        Try to acquire tokens/requests.

        Args:
            client_id: Client identifier
            route: Route identifier
            tokens: Number of tokens to acquire

        Returns:
            True if acquired
        """
        # Ensure connected
        if not self._connected:
            if not await self.connect():
                return await self._use_fallback('acquire', client_id, route, tokens)

        key = self._get_key(client_id, route)
        now = time.time()

        try:
            if self.config.strategy == RateLimitStrategy.TOKEN_BUCKET:
                return await self._acquire_token_bucket(key, tokens, now)
            elif self.config.strategy == RateLimitStrategy.SLIDING_WINDOW:
                return await self._acquire_sliding_window(key, now)
            elif self.config.strategy == RateLimitStrategy.FIXED_WINDOW:
                return await self._acquire_fixed_window(key, tokens, now)
            elif self.config.strategy == RateLimitStrategy.LEAKY_BUCKET:
                return await self._acquire_leaky_bucket(key, tokens, now)
            else:
                logger.warning(f"Unknown strategy: {self.config.strategy}")
                return True

        except Exception as e:
            logger.error(f"Redis rate limit error: {e}")
            self._connected = False
            return await self._use_fallback('acquire', client_id, route, tokens)

    async def _acquire_token_bucket(
        self,
        key: str,
        tokens: int,
        now: float
    ) -> bool:
        """Execute token bucket acquisition via Lua script."""
        tokens_key = f"{key}:tokens"
        timestamp_key = f"{key}:ts"

        result = await self._redis.evalsha(
            self._token_bucket_sha,
            2,  # number of keys
            tokens_key,
            timestamp_key,
            str(self.config.bucket_size),
            str(self.config.tokens_per_second),
            str(tokens),
            str(now),
            str(self.config.key_ttl_seconds),
        )

        success = int(result[0]) == 1
        return success

    async def _acquire_sliding_window(
        self,
        key: str,
        now: float
    ) -> bool:
        """Execute sliding window acquisition via Lua script."""
        window_key = f"{key}:window"

        result = await self._redis.evalsha(
            self._sliding_window_sha,
            1,  # number of keys
            window_key,
            str(self.config.window_size_seconds),
            str(self.config.max_requests),
            str(now),
            str(self.config.key_ttl_seconds),
        )

        success = int(result[0]) == 1
        return success

    async def _acquire_fixed_window(
        self,
        key: str,
        count: int,
        now: float
    ) -> bool:
        """Execute fixed window acquisition via Lua script."""
        result = await self._redis.evalsha(
            self._fixed_window_sha,
            1,  # number of keys
            key,
            str(self.config.window_size_seconds),
            str(self.config.max_requests),
            str(now),
            str(count),
        )

        success = int(result[0]) == 1
        return success

    async def _acquire_leaky_bucket(
        self,
        key: str,
        amount: int,
        now: float
    ) -> bool:
        """Execute leaky bucket acquisition via Lua script."""
        level_key = f"{key}:level"
        timestamp_key = f"{key}:ts"

        result = await self._redis.evalsha(
            self._leaky_bucket_sha,
            2,  # number of keys
            level_key,
            timestamp_key,
            str(self.config.bucket_size),
            str(self.config.tokens_per_second),
            str(amount),
            str(now),
            str(self.config.key_ttl_seconds),
        )

        success = int(result[0]) == 1
        return success

    async def check(
        self,
        client_id: Optional[str] = None,
        route: Optional[str] = None
    ) -> RateLimitInfo:
        """
        Check rate limit status without consuming.

        Args:
            client_id: Client identifier
            route: Route identifier

        Returns:
            Rate limit information
        """
        # Ensure connected
        if not self._connected:
            if not await self.connect():
                return await self._use_fallback('check', client_id, route)

        key = self._get_key(client_id, route)
        now = time.time()

        try:
            if self.config.strategy == RateLimitStrategy.TOKEN_BUCKET:
                return await self._check_token_bucket(key, now)
            elif self.config.strategy == RateLimitStrategy.SLIDING_WINDOW:
                return await self._check_sliding_window(key, now)
            elif self.config.strategy == RateLimitStrategy.FIXED_WINDOW:
                return await self._check_fixed_window(key, now)
            elif self.config.strategy == RateLimitStrategy.LEAKY_BUCKET:
                return await self._check_leaky_bucket(key, now)
            else:
                return RateLimitInfo(
                    limit=self.config.max_requests,
                    remaining=self.config.max_requests,
                    reset_at=datetime.utcnow(),
                )

        except Exception as e:
            logger.error(f"Redis rate limit check error: {e}")
            self._connected = False
            return await self._use_fallback('check', client_id, route)

    async def _check_token_bucket(
        self,
        key: str,
        now: float
    ) -> RateLimitInfo:
        """Check token bucket status."""
        tokens_key = f"{key}:tokens"
        timestamp_key = f"{key}:ts"

        # Get current state without modifying
        pipe = self._redis.pipeline()
        pipe.get(tokens_key)
        pipe.get(timestamp_key)
        results = await pipe.execute()

        current_tokens = float(results[0]) if results[0] else float(self.config.bucket_size)
        last_update = float(results[1]) if results[1] else now

        # Calculate current tokens
        elapsed = max(0, now - last_update)
        tokens_added = elapsed * self.config.tokens_per_second
        available = min(self.config.bucket_size, current_tokens + tokens_added)

        remaining = int(available)
        time_until_available = 0.0 if available >= 1 else (1 - available) / self.config.tokens_per_second

        return RateLimitInfo(
            limit=self.config.bucket_size,
            remaining=remaining,
            reset_at=datetime.utcnow() + timedelta(seconds=time_until_available),
            retry_after=int(time_until_available) + 1 if remaining <= 0 else None,
        )

    async def _check_sliding_window(
        self,
        key: str,
        now: float
    ) -> RateLimitInfo:
        """Check sliding window status."""
        window_key = f"{key}:window"
        window_start = now - self.config.window_size_seconds

        # Count requests in window
        count = await self._redis.zcount(window_key, window_start, '+inf')
        remaining = max(0, self.config.max_requests - count)

        # Get oldest entry for reset time
        oldest = await self._redis.zrange(window_key, 0, 0, withscores=True)
        if oldest:
            oldest_ts = oldest[0][1]
            reset_at = datetime.utcfromtimestamp(oldest_ts + self.config.window_size_seconds)
        else:
            reset_at = datetime.utcnow()

        retry_after = None
        if remaining <= 0 and oldest:
            retry_after = int(oldest[0][1] + self.config.window_size_seconds - now) + 1

        return RateLimitInfo(
            limit=self.config.max_requests,
            remaining=remaining,
            reset_at=reset_at,
            retry_after=retry_after,
        )

    async def _check_fixed_window(
        self,
        key: str,
        now: float
    ) -> RateLimitInfo:
        """Check fixed window status."""
        window_start = int(now // self.config.window_size_seconds) * self.config.window_size_seconds
        window_key = f"{key}:{window_start}"
        window_reset = window_start + self.config.window_size_seconds

        # Get current count
        count = await self._redis.get(window_key)
        current = int(count) if count else 0
        remaining = max(0, self.config.max_requests - current)

        retry_after = None
        if remaining <= 0:
            retry_after = int(window_reset - now) + 1

        return RateLimitInfo(
            limit=self.config.max_requests,
            remaining=remaining,
            reset_at=datetime.utcfromtimestamp(window_reset),
            retry_after=retry_after,
        )

    async def _check_leaky_bucket(
        self,
        key: str,
        now: float
    ) -> RateLimitInfo:
        """Check leaky bucket status."""
        level_key = f"{key}:level"
        timestamp_key = f"{key}:ts"

        # Get current state
        pipe = self._redis.pipeline()
        pipe.get(level_key)
        pipe.get(timestamp_key)
        results = await pipe.execute()

        current_level = float(results[0]) if results[0] else 0.0
        last_leak = float(results[1]) if results[1] else now

        # Calculate current level
        elapsed = max(0, now - last_leak)
        leaked = elapsed * self.config.tokens_per_second
        actual_level = max(0.0, current_level - leaked)

        remaining = int(self.config.bucket_size - actual_level)
        time_until_available = 0.0
        if remaining <= 0:
            excess = actual_level - (self.config.bucket_size - 1)
            time_until_available = excess / self.config.tokens_per_second

        return RateLimitInfo(
            limit=self.config.bucket_size,
            remaining=max(0, remaining),
            reset_at=datetime.utcnow() + timedelta(seconds=time_until_available),
            retry_after=int(time_until_available) + 1 if remaining <= 0 else None,
        )

    async def reset(
        self,
        client_id: Optional[str] = None,
        route: Optional[str] = None
    ) -> None:
        """
        Reset rate limit for a key.

        Args:
            client_id: Client identifier
            route: Route identifier
        """
        if not self._connected:
            if not await self.connect():
                if self._fallback_limiter:
                    await self._fallback_limiter.reset(client_id, route)
                return

        key = self._get_key(client_id, route)

        try:
            # Delete all related keys
            keys_to_delete = [
                f"{key}:tokens",
                f"{key}:ts",
                f"{key}:window",
                f"{key}:level",
            ]

            # For fixed window, we need to find the current window key
            now = time.time()
            window_start = int(now // self.config.window_size_seconds) * self.config.window_size_seconds
            keys_to_delete.append(f"{key}:{window_start}")

            await self._redis.delete(*keys_to_delete)
            logger.info(f"Reset rate limit for key: {key}")

        except Exception as e:
            logger.error(f"Failed to reset rate limit: {e}")
            if self._fallback_limiter:
                await self._fallback_limiter.reset(client_id, route)

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get rate limiter statistics.

        Returns:
            Statistics dictionary
        """
        return {
            "strategy": self.config.strategy.value,
            "scope": self.config.scope.value,
            "redis_connected": self._connected,
            "redis_url": self.config.redis_url,
            "fallback_enabled": self.config.fallback_to_memory,
        }

    async def health_check(self) -> Dict[str, Any]:
        """
        Perform health check on Redis connection.

        Returns:
            Health check results
        """
        result = {
            "redis_available": False,
            "latency_ms": None,
            "fallback_active": False,
        }

        try:
            if self._redis and self._connected:
                start = time.monotonic()
                await self._redis.ping()
                latency = (time.monotonic() - start) * 1000
                result["redis_available"] = True
                result["latency_ms"] = round(latency, 2)
            else:
                result["fallback_active"] = self._fallback_limiter is not None

        except Exception as e:
            logger.warning(f"Health check failed: {e}")
            result["fallback_active"] = self._fallback_limiter is not None
            self._connected = False

        return result


# ============================================================================
# Factory Function
# ============================================================================

def create_rate_limiter(
    config: Optional[Union[RateLimiterConfig, RedisRateLimiterConfig]] = None,
    use_redis: bool = False,
    redis_url: Optional[str] = None,
) -> Union[RateLimiter, RedisRateLimiter]:
    """
    Factory function to create appropriate rate limiter.

    Args:
        config: Rate limiter configuration
        use_redis: Whether to use Redis-backed limiter
        redis_url: Redis URL (creates RedisRateLimiterConfig if provided)

    Returns:
        RateLimiter or RedisRateLimiter instance

    Example:
        >>> # In-memory limiter
        >>> limiter = create_rate_limiter()

        >>> # Redis-backed limiter
        >>> limiter = create_rate_limiter(use_redis=True, redis_url="redis://localhost:6379")
    """
    if redis_url or use_redis:
        if not REDIS_AVAILABLE:
            logger.warning("Redis not available, falling back to in-memory limiter")
            return RateLimiter(config if isinstance(config, RateLimiterConfig) else None)

        if isinstance(config, RedisRateLimiterConfig):
            return RedisRateLimiter(config)

        redis_config = RedisRateLimiterConfig(
            redis_url=redis_url or "redis://localhost:6379",
        )
        if isinstance(config, RateLimiterConfig):
            redis_config.strategy = config.strategy
            redis_config.scope = config.scope
            redis_config.tokens_per_second = config.tokens_per_second
            redis_config.bucket_size = config.bucket_size
            redis_config.window_size_seconds = config.window_size_seconds
            redis_config.max_requests = config.max_requests

        return RedisRateLimiter(redis_config)

    if isinstance(config, RateLimiterConfig):
        return RateLimiter(config)

    return RateLimiter()


# ============================================================================
# Async Context Manager
# ============================================================================

class ManagedRedisRateLimiter:
    """
    Async context manager for RedisRateLimiter.

    Handles connection lifecycle automatically.

    Example:
        >>> config = RedisRateLimiterConfig(redis_url="redis://localhost:6379")
        >>> async with ManagedRedisRateLimiter(config) as limiter:
        ...     if await limiter.acquire("client-123"):
        ...         process_request()
    """

    def __init__(self, config: Optional[RedisRateLimiterConfig] = None):
        """Initialize managed limiter."""
        self._limiter = RedisRateLimiter(config)

    async def __aenter__(self) -> RedisRateLimiter:
        """Connect to Redis on entry."""
        await self._limiter.connect()
        return self._limiter

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Close connection on exit."""
        await self._limiter.close()
