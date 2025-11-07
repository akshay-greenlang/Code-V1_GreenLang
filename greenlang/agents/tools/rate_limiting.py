"""
GreenLang Tool Rate Limiting
=============================

Production-grade token bucket rate limiter for tool execution.

Features:
- Token bucket algorithm for smooth rate limiting
- Per-tool and per-user rate limits
- Configurable burst capacity
- Thread-safe operation
- Minimal performance overhead (<1ms per check)
- Automatic token refill

Author: GreenLang Framework Team
Date: November 2025
Status: Production Ready
"""

from __future__ import annotations

import time
import threading
import logging
from typing import Dict, Optional, Tuple
from dataclasses import dataclass
from collections import defaultdict

logger = logging.getLogger(__name__)


# ==============================================================================
# Rate Limit Exception
# ==============================================================================

class RateLimitExceeded(Exception):
    """
    Exception raised when rate limit is exceeded.

    Attributes:
        message: Error message
        retry_after: Seconds to wait before retry
        tool_name: Name of rate-limited tool
        user_id: User ID if applicable
    """

    def __init__(
        self,
        message: str,
        retry_after: float,
        tool_name: str,
        user_id: Optional[str] = None
    ):
        """
        Initialize rate limit exception.

        Args:
            message: Error message
            retry_after: Seconds to wait before retry
            tool_name: Tool name
            user_id: User ID (optional)
        """
        super().__init__(message)
        self.retry_after = retry_after
        self.tool_name = tool_name
        self.user_id = user_id


# ==============================================================================
# Token Bucket
# ==============================================================================

@dataclass
class TokenBucket:
    """
    Token bucket for rate limiting.

    Implements the token bucket algorithm:
    - Tokens are added at a constant rate
    - Each request consumes tokens
    - Bucket has a maximum capacity (burst size)
    - Requests are denied when bucket is empty
    """

    rate: float  # Tokens per second
    capacity: int  # Maximum tokens (burst size)
    tokens: float  # Current token count
    last_update: float  # Last refill timestamp

    def refill(self) -> None:
        """Refill tokens based on elapsed time."""
        now = time.time()
        elapsed = now - self.last_update

        # Add tokens based on rate and elapsed time
        self.tokens = min(
            self.capacity,
            self.tokens + (elapsed * self.rate)
        )
        self.last_update = now

    def consume(self, tokens: int = 1) -> bool:
        """
        Try to consume tokens from bucket.

        Args:
            tokens: Number of tokens to consume

        Returns:
            True if tokens consumed, False if insufficient tokens
        """
        self.refill()

        if self.tokens >= tokens:
            self.tokens -= tokens
            return True

        return False

    def get_wait_time(self, tokens: int = 1) -> float:
        """
        Get time to wait until enough tokens are available.

        Args:
            tokens: Number of tokens needed

        Returns:
            Seconds to wait (0 if tokens available now)
        """
        self.refill()

        if self.tokens >= tokens:
            return 0.0

        # Calculate time needed for tokens to refill
        tokens_needed = tokens - self.tokens
        wait_time = tokens_needed / self.rate

        return wait_time

    @classmethod
    def create(cls, rate: int, capacity: int) -> TokenBucket:
        """
        Create a new token bucket.

        Args:
            rate: Tokens per second
            capacity: Maximum burst capacity

        Returns:
            New TokenBucket instance
        """
        return cls(
            rate=float(rate),
            capacity=capacity,
            tokens=float(capacity),  # Start full
            last_update=time.time()
        )


# ==============================================================================
# Rate Limiter
# ==============================================================================

class RateLimiter:
    """
    Thread-safe rate limiter using token bucket algorithm.

    Supports:
    - Per-tool rate limits
    - Per-user rate limits
    - Combined per-tool-per-user limits
    - Configurable burst capacity
    - Automatic token refill

    Example:
        >>> limiter = RateLimiter(rate=10, burst=20, per_tool=True)
        >>> if limiter.check_limit("calculate_emissions"):
        ...     limiter.consume("calculate_emissions")
        ...     # Execute tool
        ... else:
        ...     wait_time = limiter.get_wait_time("calculate_emissions")
        ...     print(f"Rate limited. Retry after {wait_time:.2f}s")
    """

    def __init__(
        self,
        rate: int = 10,
        burst: int = 20,
        per_tool: bool = True,
        per_user: bool = False,
        per_tool_limits: Optional[Dict[str, Tuple[int, int]]] = None
    ):
        """
        Initialize rate limiter.

        Args:
            rate: Default rate limit (tokens per second)
            burst: Default burst capacity (maximum tokens)
            per_tool: Enable per-tool rate limiting
            per_user: Enable per-user rate limiting
            per_tool_limits: Override limits for specific tools
                Format: {"tool_name": (rate, burst)}
        """
        self.default_rate = rate
        self.default_burst = burst
        self.per_tool = per_tool
        self.per_user = per_user
        self.per_tool_limits = per_tool_limits or {}

        # Thread-safe bucket storage
        self._lock = threading.RLock()
        self._buckets: Dict[str, TokenBucket] = {}

        # Statistics
        self._total_requests = 0
        self._limited_requests = 0

        self.logger = logging.getLogger(__name__)

    def _get_bucket_key(self, tool_name: str, user_id: Optional[str] = None) -> str:
        """
        Generate bucket key based on configuration.

        Args:
            tool_name: Tool name
            user_id: User ID (optional)

        Returns:
            Bucket key string
        """
        if self.per_user and user_id:
            if self.per_tool:
                return f"{tool_name}:{user_id}"
            else:
                return user_id
        elif self.per_tool:
            return tool_name
        else:
            return "global"

    def _get_or_create_bucket(
        self,
        tool_name: str,
        user_id: Optional[str] = None
    ) -> TokenBucket:
        """
        Get or create token bucket for key.

        Args:
            tool_name: Tool name
            user_id: User ID (optional)

        Returns:
            TokenBucket instance
        """
        key = self._get_bucket_key(tool_name, user_id)

        with self._lock:
            if key not in self._buckets:
                # Check for tool-specific limits
                if tool_name in self.per_tool_limits:
                    rate, burst = self.per_tool_limits[tool_name]
                else:
                    rate, burst = self.default_rate, self.default_burst

                self._buckets[key] = TokenBucket.create(rate, burst)

            return self._buckets[key]

    def check_limit(
        self,
        tool_name: str,
        user_id: Optional[str] = None,
        tokens: int = 1
    ) -> bool:
        """
        Check if request is within rate limit.

        This does NOT consume tokens. Use consume() after checking.

        Args:
            tool_name: Tool name
            user_id: User ID (optional)
            tokens: Number of tokens required (default: 1)

        Returns:
            True if within limit, False if rate limited
        """
        bucket = self._get_or_create_bucket(tool_name, user_id)

        with self._lock:
            self._total_requests += 1
            bucket.refill()

            if bucket.tokens >= tokens:
                return True

            self._limited_requests += 1
            self.logger.warning(
                f"Rate limit check failed for tool={tool_name}, "
                f"user={user_id}, tokens_needed={tokens}, "
                f"tokens_available={bucket.tokens:.2f}"
            )
            return False

    def consume(
        self,
        tool_name: str,
        user_id: Optional[str] = None,
        tokens: int = 1
    ) -> bool:
        """
        Consume tokens from bucket.

        Args:
            tool_name: Tool name
            user_id: User ID (optional)
            tokens: Number of tokens to consume (default: 1)

        Returns:
            True if tokens consumed, False if insufficient tokens
        """
        bucket = self._get_or_create_bucket(tool_name, user_id)

        with self._lock:
            success = bucket.consume(tokens)

            if success:
                self.logger.debug(
                    f"Consumed {tokens} token(s) for tool={tool_name}, "
                    f"user={user_id}, remaining={bucket.tokens:.2f}"
                )
            else:
                self.logger.warning(
                    f"Failed to consume {tokens} token(s) for tool={tool_name}, "
                    f"user={user_id}, available={bucket.tokens:.2f}"
                )

            return success

    def get_wait_time(
        self,
        tool_name: str,
        user_id: Optional[str] = None,
        tokens: int = 1
    ) -> float:
        """
        Get time to wait before next request.

        Args:
            tool_name: Tool name
            user_id: User ID (optional)
            tokens: Number of tokens needed (default: 1)

        Returns:
            Seconds to wait (0 if tokens available now)
        """
        bucket = self._get_or_create_bucket(tool_name, user_id)

        with self._lock:
            wait_time = bucket.get_wait_time(tokens)

            if wait_time > 0:
                self.logger.debug(
                    f"Wait time for tool={tool_name}, user={user_id}: {wait_time:.2f}s"
                )

            return wait_time

    def check_and_consume(
        self,
        tool_name: str,
        user_id: Optional[str] = None,
        tokens: int = 1
    ) -> None:
        """
        Check rate limit and consume tokens atomically.

        Raises RateLimitExceeded if limit exceeded.

        Args:
            tool_name: Tool name
            user_id: User ID (optional)
            tokens: Number of tokens to consume (default: 1)

        Raises:
            RateLimitExceeded: If rate limit exceeded
        """
        with self._lock:
            if not self.check_limit(tool_name, user_id, tokens):
                wait_time = self.get_wait_time(tool_name, user_id, tokens)
                raise RateLimitExceeded(
                    message=f"Rate limit exceeded for tool '{tool_name}'. "
                            f"Retry after {wait_time:.2f} seconds.",
                    retry_after=wait_time,
                    tool_name=tool_name,
                    user_id=user_id
                )

            # Consume tokens
            if not self.consume(tool_name, user_id, tokens):
                # Should not happen since we just checked, but handle anyway
                wait_time = self.get_wait_time(tool_name, user_id, tokens)
                raise RateLimitExceeded(
                    message=f"Failed to consume tokens for tool '{tool_name}'. "
                            f"Retry after {wait_time:.2f} seconds.",
                    retry_after=wait_time,
                    tool_name=tool_name,
                    user_id=user_id
                )

    def reset(self, tool_name: Optional[str] = None, user_id: Optional[str] = None) -> None:
        """
        Reset rate limit buckets.

        Args:
            tool_name: Reset specific tool (None = reset all)
            user_id: Reset specific user (None = reset all)
        """
        with self._lock:
            if tool_name is None and user_id is None:
                # Reset all buckets
                self._buckets.clear()
                self.logger.info("Reset all rate limit buckets")
            else:
                # Reset specific bucket
                key = self._get_bucket_key(tool_name or "", user_id)
                if key in self._buckets:
                    del self._buckets[key]
                    self.logger.info(f"Reset rate limit bucket: {key}")

    def get_stats(self) -> Dict[str, any]:
        """
        Get rate limiting statistics.

        Returns:
            Dictionary with statistics
        """
        with self._lock:
            limited_rate = (
                (self._limited_requests / self._total_requests * 100)
                if self._total_requests > 0
                else 0.0
            )

            return {
                "total_requests": self._total_requests,
                "limited_requests": self._limited_requests,
                "limited_percentage": round(limited_rate, 2),
                "active_buckets": len(self._buckets),
                "config": {
                    "default_rate": self.default_rate,
                    "default_burst": self.default_burst,
                    "per_tool": self.per_tool,
                    "per_user": self.per_user,
                    "per_tool_limits": self.per_tool_limits,
                }
            }

    def get_bucket_status(
        self,
        tool_name: str,
        user_id: Optional[str] = None
    ) -> Dict[str, any]:
        """
        Get current status of a specific bucket.

        Args:
            tool_name: Tool name
            user_id: User ID (optional)

        Returns:
            Dictionary with bucket status
        """
        bucket = self._get_or_create_bucket(tool_name, user_id)

        with self._lock:
            bucket.refill()
            return {
                "key": self._get_bucket_key(tool_name, user_id),
                "rate": bucket.rate,
                "capacity": bucket.capacity,
                "available_tokens": round(bucket.tokens, 2),
                "utilization_percentage": round(
                    (1 - bucket.tokens / bucket.capacity) * 100, 2
                ),
            }

    def __repr__(self) -> str:
        return (
            f"RateLimiter(rate={self.default_rate}, "
            f"burst={self.default_burst}, "
            f"per_tool={self.per_tool}, "
            f"per_user={self.per_user})"
        )


# ==============================================================================
# Global Rate Limiter Instance
# ==============================================================================

_global_rate_limiter: Optional[RateLimiter] = None


def get_rate_limiter() -> RateLimiter:
    """
    Get global rate limiter instance.

    Returns:
        Global RateLimiter instance
    """
    global _global_rate_limiter

    if _global_rate_limiter is None:
        _global_rate_limiter = RateLimiter()

    return _global_rate_limiter


def configure_rate_limiter(
    rate: int = 10,
    burst: int = 20,
    per_tool: bool = True,
    per_user: bool = False,
    per_tool_limits: Optional[Dict[str, Tuple[int, int]]] = None
) -> RateLimiter:
    """
    Configure global rate limiter.

    Args:
        rate: Default rate limit (tokens per second)
        burst: Default burst capacity
        per_tool: Enable per-tool rate limiting
        per_user: Enable per-user rate limiting
        per_tool_limits: Override limits for specific tools

    Returns:
        Configured RateLimiter instance
    """
    global _global_rate_limiter

    _global_rate_limiter = RateLimiter(
        rate=rate,
        burst=burst,
        per_tool=per_tool,
        per_user=per_user,
        per_tool_limits=per_tool_limits
    )

    logger.info(f"Configured global rate limiter: {_global_rate_limiter}")
    return _global_rate_limiter
