"""Rate Limiting and Throttling Handler.

Production-grade rate limiting using token bucket and leaky bucket algorithms:
- Handle 429 Too Many Requests responses
- Respect Retry-After headers
- Token bucket algorithm for burst handling
- Leaky bucket algorithm for smooth rate limiting
- Per-tenant and per-endpoint rate limiting

Inspired by AWS API Gateway, Stripe, and cloud provider rate limiters.

Author: GreenLang Resilience Team
Date: November 2025
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from threading import Lock
from typing import Any, Callable, Dict, Optional, TypeVar

logger = logging.getLogger(__name__)

F = TypeVar("F", bound=Callable[..., Any])


# ==============================================================================
# Exceptions
# ==============================================================================


class RateLimitExceeded(Exception):
    """Raised when rate limit is exceeded."""

    def __init__(
        self,
        limit: int,
        window: float,
        retry_after: Optional[float] = None
    ):
        self.limit = limit
        self.window = window
        self.retry_after = retry_after

        msg = f"Rate limit exceeded: {limit} requests per {window}s"
        if retry_after:
            msg += f". Retry after {retry_after}s"

        super().__init__(msg)


# ==============================================================================
# Configuration
# ==============================================================================


class RateLimitAlgorithm(str, Enum):
    """Rate limiting algorithms."""
    TOKEN_BUCKET = "token_bucket"
    LEAKY_BUCKET = "leaky_bucket"
    FIXED_WINDOW = "fixed_window"
    SLIDING_WINDOW = "sliding_window"


@dataclass
class RateLimitConfig:
    """Configuration for rate limiting.

    Attributes:
        requests_per_second: Maximum requests per second
        burst_size: Maximum burst size for token bucket (default: 2x rate)
        algorithm: Rate limiting algorithm (default: token_bucket)
        respect_retry_after: Respect Retry-After header (default: True)
        raise_on_limit: Raise exception when limit exceeded (default: True)
        wait_on_limit: Wait when limit exceeded (default: False)
        on_limit_exceeded: Callback when limit exceeded
    """
    requests_per_second: float
    burst_size: Optional[int] = None
    algorithm: RateLimitAlgorithm = RateLimitAlgorithm.TOKEN_BUCKET
    respect_retry_after: bool = True
    raise_on_limit: bool = True
    wait_on_limit: bool = False
    on_limit_exceeded: Optional[Callable[[float], None]] = None

    def __post_init__(self):
        """Initialize computed fields."""
        if self.burst_size is None:
            # Default burst size is 2x the rate
            self.burst_size = max(1, int(self.requests_per_second * 2))


# ==============================================================================
# Token Bucket Algorithm
# ==============================================================================


class TokenBucket:
    """Token bucket rate limiter.

    Allows bursts up to bucket capacity while maintaining average rate.

    Attributes:
        rate: Tokens added per second
        capacity: Maximum bucket capacity
        tokens: Current token count
        last_update: Last update timestamp
    """

    def __init__(self, rate: float, capacity: int):
        """Initialize token bucket.

        Args:
            rate: Tokens added per second
            capacity: Maximum bucket capacity
        """
        self.rate = rate
        self.capacity = capacity
        self.tokens = float(capacity)
        self.last_update = time.time()
        self._lock = Lock()

    def consume(self, tokens: int = 1) -> bool:
        """Attempt to consume tokens.

        Args:
            tokens: Number of tokens to consume

        Returns:
            True if tokens consumed, False if insufficient
        """
        with self._lock:
            now = time.time()
            elapsed = now - self.last_update

            # Add new tokens based on elapsed time
            self.tokens = min(
                self.capacity,
                self.tokens + elapsed * self.rate
            )
            self.last_update = now

            # Check if enough tokens available
            if self.tokens >= tokens:
                self.tokens -= tokens
                return True
            else:
                return False

    def wait_time(self, tokens: int = 1) -> float:
        """Calculate wait time until tokens available.

        Args:
            tokens: Number of tokens needed

        Returns:
            Wait time in seconds
        """
        with self._lock:
            now = time.time()
            elapsed = now - self.last_update

            # Calculate current tokens
            current_tokens = min(
                self.capacity,
                self.tokens + elapsed * self.rate
            )

            if current_tokens >= tokens:
                return 0.0

            # Calculate time needed to acquire tokens
            tokens_needed = tokens - current_tokens
            wait_time = tokens_needed / self.rate

            return wait_time

    def reset(self) -> None:
        """Reset bucket to full capacity."""
        with self._lock:
            self.tokens = float(self.capacity)
            self.last_update = time.time()


# ==============================================================================
# Leaky Bucket Algorithm
# ==============================================================================


class LeakyBucket:
    """Leaky bucket rate limiter.

    Enforces smooth output rate regardless of input bursts.

    Attributes:
        rate: Leak rate (requests per second)
        capacity: Maximum bucket capacity
        level: Current bucket level
        last_leak: Last leak timestamp
    """

    def __init__(self, rate: float, capacity: int):
        """Initialize leaky bucket.

        Args:
            rate: Leak rate (requests per second)
            capacity: Maximum bucket capacity
        """
        self.rate = rate
        self.capacity = capacity
        self.level = 0.0
        self.last_leak = time.time()
        self._lock = Lock()

    def consume(self, amount: int = 1) -> bool:
        """Attempt to add to bucket.

        Args:
            amount: Amount to add

        Returns:
            True if added, False if bucket full
        """
        with self._lock:
            self._leak()

            # Check if bucket has capacity
            if self.level + amount <= self.capacity:
                self.level += amount
                return True
            else:
                return False

    def _leak(self) -> None:
        """Leak water from bucket based on elapsed time."""
        now = time.time()
        elapsed = now - self.last_leak

        # Calculate amount leaked
        leaked = elapsed * self.rate
        self.level = max(0.0, self.level - leaked)
        self.last_leak = now

    def wait_time(self, amount: int = 1) -> float:
        """Calculate wait time until space available.

        Args:
            amount: Amount needed

        Returns:
            Wait time in seconds
        """
        with self._lock:
            self._leak()

            if self.level + amount <= self.capacity:
                return 0.0

            # Calculate time needed for bucket to drain
            overflow = (self.level + amount) - self.capacity
            wait_time = overflow / self.rate

            return wait_time

    def reset(self) -> None:
        """Reset bucket to empty."""
        with self._lock:
            self.level = 0.0
            self.last_leak = time.time()


# ==============================================================================
# Fixed Window Rate Limiter
# ==============================================================================


class FixedWindowLimiter:
    """Fixed window rate limiter.

    Resets counter at fixed intervals.

    Attributes:
        limit: Maximum requests per window
        window_size: Window size in seconds
        count: Current request count
        window_start: Current window start time
    """

    def __init__(self, limit: int, window_size: float):
        """Initialize fixed window limiter.

        Args:
            limit: Maximum requests per window
            window_size: Window size in seconds
        """
        self.limit = limit
        self.window_size = window_size
        self.count = 0
        self.window_start = time.time()
        self._lock = Lock()

    def consume(self, amount: int = 1) -> bool:
        """Attempt to consume from window.

        Args:
            amount: Number of requests

        Returns:
            True if allowed, False if limit exceeded
        """
        with self._lock:
            now = time.time()

            # Check if window expired
            if now - self.window_start >= self.window_size:
                self.count = 0
                self.window_start = now

            # Check if under limit
            if self.count + amount <= self.limit:
                self.count += amount
                return True
            else:
                return False

    def wait_time(self, amount: int = 1) -> float:
        """Calculate wait time until next window.

        Args:
            amount: Number of requests

        Returns:
            Wait time in seconds
        """
        with self._lock:
            now = time.time()

            # Check if window expired
            if now - self.window_start >= self.window_size:
                return 0.0

            # If current count allows, no wait
            if self.count + amount <= self.limit:
                return 0.0

            # Wait until next window
            return self.window_size - (now - self.window_start)

    def reset(self) -> None:
        """Reset window."""
        with self._lock:
            self.count = 0
            self.window_start = time.time()


# ==============================================================================
# Rate Limiter Manager
# ==============================================================================


class RateLimiter:
    """Multi-tenant rate limiter with configurable algorithms.

    Manages rate limits per tenant/endpoint combination.
    """

    def __init__(
        self,
        default_config: Optional[RateLimitConfig] = None
    ):
        """Initialize rate limiter manager.

        Args:
            default_config: Default rate limit configuration
        """
        self.default_config = default_config or RateLimitConfig(
            requests_per_second=10.0,
            algorithm=RateLimitAlgorithm.TOKEN_BUCKET,
        )

        # Limiters by key (tenant:endpoint)
        self._limiters: Dict[str, Any] = {}
        self._configs: Dict[str, RateLimitConfig] = {}
        self._lock = Lock()

    def configure(
        self,
        key: str,
        config: RateLimitConfig
    ) -> None:
        """Configure rate limit for specific key.

        Args:
            key: Rate limit key (e.g., "tenant_id:endpoint")
            config: Rate limit configuration
        """
        with self._lock:
            self._configs[key] = config

            # Create limiter based on algorithm
            if config.algorithm == RateLimitAlgorithm.TOKEN_BUCKET:
                self._limiters[key] = TokenBucket(
                    rate=config.requests_per_second,
                    capacity=config.burst_size or int(config.requests_per_second * 2),
                )
            elif config.algorithm == RateLimitAlgorithm.LEAKY_BUCKET:
                self._limiters[key] = LeakyBucket(
                    rate=config.requests_per_second,
                    capacity=config.burst_size or int(config.requests_per_second * 2),
                )
            elif config.algorithm == RateLimitAlgorithm.FIXED_WINDOW:
                self._limiters[key] = FixedWindowLimiter(
                    limit=config.burst_size or int(config.requests_per_second),
                    window_size=1.0,
                )

    def check_limit(
        self,
        key: str,
        amount: int = 1,
        wait: bool = False,
    ) -> bool:
        """Check if request allowed under rate limit.

        Args:
            key: Rate limit key
            amount: Number of requests
            wait: Wait if limit exceeded

        Returns:
            True if allowed, False if limit exceeded

        Raises:
            RateLimitExceeded: If limit exceeded and configured to raise
        """
        # Get or create limiter
        limiter = self._get_or_create_limiter(key)
        config = self._configs.get(key, self.default_config)

        # Check limit
        allowed = limiter.consume(amount)

        if not allowed:
            wait_time = limiter.wait_time(amount)

            logger.warning(
                f"Rate limit exceeded for {key}. "
                f"Wait time: {wait_time:.2f}s"
            )

            # Callback
            if config.on_limit_exceeded:
                config.on_limit_exceeded(wait_time)

            # Wait if configured
            if wait or config.wait_on_limit:
                logger.info(f"Waiting {wait_time:.2f}s for rate limit...")
                time.sleep(wait_time)
                return limiter.consume(amount)

            # Raise if configured
            if config.raise_on_limit:
                raise RateLimitExceeded(
                    limit=int(config.requests_per_second),
                    window=1.0,
                    retry_after=wait_time,
                )

        return allowed

    async def async_check_limit(
        self,
        key: str,
        amount: int = 1,
        wait: bool = False,
    ) -> bool:
        """Async version of check_limit.

        Args:
            key: Rate limit key
            amount: Number of requests
            wait: Wait if limit exceeded

        Returns:
            True if allowed, False if limit exceeded
        """
        # Get or create limiter
        limiter = self._get_or_create_limiter(key)
        config = self._configs.get(key, self.default_config)

        # Check limit
        allowed = limiter.consume(amount)

        if not allowed:
            wait_time = limiter.wait_time(amount)

            logger.warning(
                f"Rate limit exceeded for {key}. "
                f"Wait time: {wait_time:.2f}s"
            )

            # Callback
            if config.on_limit_exceeded:
                config.on_limit_exceeded(wait_time)

            # Wait if configured
            if wait or config.wait_on_limit:
                logger.info(f"Waiting {wait_time:.2f}s for rate limit...")
                await asyncio.sleep(wait_time)
                return limiter.consume(amount)

            # Raise if configured
            if config.raise_on_limit:
                raise RateLimitExceeded(
                    limit=int(config.requests_per_second),
                    window=1.0,
                    retry_after=wait_time,
                )

        return allowed

    def _get_or_create_limiter(self, key: str) -> Any:
        """Get or create limiter for key.

        Args:
            key: Rate limit key

        Returns:
            Rate limiter instance
        """
        if key not in self._limiters:
            # Use default config
            self.configure(key, self.default_config)

        return self._limiters[key]

    def reset(self, key: Optional[str] = None) -> None:
        """Reset rate limiters.

        Args:
            key: Specific key to reset (or all if None)
        """
        with self._lock:
            if key:
                if key in self._limiters:
                    self._limiters[key].reset()
            else:
                for limiter in self._limiters.values():
                    limiter.reset()

    def get_stats(self, key: str) -> Dict[str, Any]:
        """Get rate limiter statistics.

        Args:
            key: Rate limit key

        Returns:
            Statistics dictionary
        """
        if key not in self._limiters:
            return {}

        limiter = self._limiters[key]
        config = self._configs.get(key, self.default_config)

        stats = {
            "algorithm": config.algorithm.value,
            "requests_per_second": config.requests_per_second,
            "burst_size": config.burst_size,
        }

        if isinstance(limiter, TokenBucket):
            stats["current_tokens"] = limiter.tokens
            stats["capacity"] = limiter.capacity

        elif isinstance(limiter, LeakyBucket):
            stats["current_level"] = limiter.level
            stats["capacity"] = limiter.capacity

        elif isinstance(limiter, FixedWindowLimiter):
            stats["current_count"] = limiter.count
            stats["limit"] = limiter.limit
            stats["window_size"] = limiter.window_size

        return stats


# ==============================================================================
# Global Rate Limiter Instance
# ==============================================================================


_global_rate_limiter: Optional[RateLimiter] = None


def get_rate_limiter() -> RateLimiter:
    """Get global rate limiter instance.

    Returns:
        RateLimiter instance
    """
    global _global_rate_limiter
    if _global_rate_limiter is None:
        _global_rate_limiter = RateLimiter()
    return _global_rate_limiter


__all__ = [
    "RateLimiter",
    "TokenBucket",
    "LeakyBucket",
    "FixedWindowLimiter",
    "RateLimitConfig",
    "RateLimitAlgorithm",
    "RateLimitExceeded",
    "get_rate_limiter",
]
