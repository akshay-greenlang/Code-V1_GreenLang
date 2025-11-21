# -*- coding: utf-8 -*-
"""
RateLimiter - Production-ready rate limiting with token bucket algorithm.

Implements token bucket rate limiting to prevent exceeding API rate limits
for LLM providers (e.g., Anthropic: 1000 req/min, OpenAI: 10000 req/min).

Features:
- Token bucket algorithm for smooth rate limiting
- Per-provider configuration
- Request queuing with timeout
- Token refill based on time elapsed
- Thread-safe implementation
- Metrics tracking

Example:
    >>> limiter = RateLimiter(requests_per_minute=1000, tokens_per_minute=100000)
    >>> async with limiter.acquire(tokens=1500):
    ...     response = await api_call()
"""

import asyncio
import logging
import time
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class RateLimiterStats:
    """Statistics for rate limiter monitoring."""
    total_requests: int = 0
    total_tokens: int = 0
    rejected_requests: int = 0
    wait_time_total_ms: float = 0.0
    current_tokens: float = 0.0
    current_requests: float = 0.0


class RateLimitExceededError(Exception):
    """Exception raised when rate limit is exceeded and cannot wait."""
    def __init__(self, message: str, retry_after: float):
        super().__init__(message)
        self.retry_after = retry_after


class TokenBucket:
    """
    Token bucket implementation for rate limiting.

    The token bucket algorithm allows bursts of traffic while maintaining
    an average rate. Tokens are added at a constant rate, and each request
    consumes tokens.

    Attributes:
        capacity: Maximum number of tokens in bucket
        refill_rate: Tokens added per second
        tokens: Current number of tokens available
    """

    def __init__(self, capacity: float, refill_rate: float):
        """
        Initialize token bucket.

        Args:
            capacity: Maximum bucket capacity (tokens)
            refill_rate: Tokens added per second
        """
        self.capacity = capacity
        self.refill_rate = refill_rate
        self.tokens = capacity
        self.last_refill = time.time()
        self._lock = asyncio.Lock()

    async def consume(self, tokens: float, timeout: Optional[float] = None) -> float:
        """
        Consume tokens from bucket, waiting if necessary.

        Args:
            tokens: Number of tokens to consume
            timeout: Maximum time to wait (seconds), None for no limit

        Returns:
            Time waited in seconds

        Raises:
            RateLimitExceededError: If timeout exceeded or tokens > capacity
        """
        async with self._lock:
            # Refill tokens based on time elapsed
            await self._refill()

            # Check if request can ever be satisfied
            if tokens > self.capacity:
                raise RateLimitExceededError(
                    f"Request size ({tokens} tokens) exceeds bucket capacity ({self.capacity})",
                    retry_after=0.0,
                )

            # Calculate wait time needed
            if self.tokens >= tokens:
                # Sufficient tokens available
                self.tokens -= tokens
                return 0.0

            # Need to wait for more tokens
            tokens_needed = tokens - self.tokens
            wait_time = tokens_needed / self.refill_rate

            if timeout is not None and wait_time > timeout:
                raise RateLimitExceededError(
                    f"Rate limit exceeded - would need to wait {wait_time:.1f}s (timeout: {timeout}s)",
                    retry_after=wait_time,
                )

            # Wait for tokens to refill
            await asyncio.sleep(wait_time)
            await self._refill()
            self.tokens -= tokens

            return wait_time

    async def _refill(self) -> None:
        """Refill tokens based on time elapsed."""
        now = time.time()
        time_elapsed = now - self.last_refill

        # Add tokens based on time elapsed
        tokens_to_add = time_elapsed * self.refill_rate
        self.tokens = min(self.capacity, self.tokens + tokens_to_add)

        self.last_refill = now

    async def peek(self) -> float:
        """Get current number of tokens without consuming."""
        async with self._lock:
            await self._refill()
            return self.tokens


class RateLimiter:
    """
    Production-ready rate limiter with dual limits (requests + tokens).

    Implements rate limiting for both request count and token usage,
    ensuring API limits are respected while maximizing throughput.

    Typical limits:
    - Anthropic: 1000 req/min, 100K tokens/min
    - OpenAI: 10000 req/min, 2M tokens/min

    Attributes:
        request_bucket: Token bucket for request count
        token_bucket: Token bucket for token usage
    """

    def __init__(
        self,
        requests_per_minute: int = 1000,
        tokens_per_minute: int = 100000,
        enable_queuing: bool = True,
        max_wait_time: Optional[float] = 30.0,
        name: str = "default",
    ):
        """
        Initialize rate limiter.

        Args:
            requests_per_minute: Maximum requests per minute
            tokens_per_minute: Maximum tokens per minute
            enable_queuing: Whether to queue requests when limit reached
            max_wait_time: Maximum time to wait for rate limit (seconds)
            name: Name for logging and identification
        """
        self.requests_per_minute = requests_per_minute
        self.tokens_per_minute = tokens_per_minute
        self.enable_queuing = enable_queuing
        self.max_wait_time = max_wait_time
        self.name = name

        # Create token buckets
        # Convert per-minute to per-second
        self.request_bucket = TokenBucket(
            capacity=requests_per_minute / 60.0 * 10,  # 10 seconds of burst capacity
            refill_rate=requests_per_minute / 60.0,  # requests per second
        )

        self.token_bucket = TokenBucket(
            capacity=tokens_per_minute / 60.0 * 10,  # 10 seconds of burst capacity
            refill_rate=tokens_per_minute / 60.0,  # tokens per second
        )

        # Statistics
        self._stats = RateLimiterStats()

        self._logger = logging.getLogger(f"{__name__}.{name}")
        self._logger.info(
            f"Initialized RateLimiter: {requests_per_minute} req/min, "
            f"{tokens_per_minute} tokens/min"
        )

    async def acquire(self, tokens: int = 0) -> "RateLimiterContext":
        """
        Acquire rate limit for a request.

        Args:
            tokens: Number of tokens this request will use (for pre-allocation)

        Returns:
            Context manager for rate limit acquisition

        Example:
            >>> async with limiter.acquire(tokens=1500):
            ...     response = await api_call()
        """
        return RateLimiterContext(self, tokens)

    async def _acquire_internal(self, tokens: int) -> float:
        """
        Internal method to acquire rate limit.

        Args:
            tokens: Number of tokens to reserve

        Returns:
            Total wait time in seconds

        Raises:
            RateLimitExceededError: If rate limit exceeded
        """
        self._stats.total_requests += 1
        if tokens > 0:
            self._stats.total_tokens += tokens

        total_wait = 0.0

        try:
            # Acquire request rate limit
            wait_time = await self.request_bucket.consume(
                tokens=1.0,
                timeout=self.max_wait_time if self.enable_queuing else 0.0,
            )
            total_wait += wait_time

            # Acquire token rate limit (if tokens specified)
            if tokens > 0:
                wait_time = await self.token_bucket.consume(
                    tokens=float(tokens),
                    timeout=self.max_wait_time if self.enable_queuing else 0.0,
                )
                total_wait += wait_time

            if total_wait > 0:
                self._logger.debug(
                    f"Rate limiter '{self.name}' waited {total_wait:.2f}s "
                    f"(requests={self._stats.total_requests}, tokens={tokens})"
                )

            self._stats.wait_time_total_ms += total_wait * 1000

            return total_wait

        except RateLimitExceededError:
            self._stats.rejected_requests += 1
            raise

    @property
    def stats(self) -> RateLimiterStats:
        """Get rate limiter statistics."""
        return self._stats

    async def get_available_capacity(self) -> tuple[float, float]:
        """
        Get current available capacity.

        Returns:
            Tuple of (available_requests, available_tokens)
        """
        available_requests = await self.request_bucket.peek()
        available_tokens = await self.token_bucket.peek()
        return (available_requests, available_tokens)


class RateLimiterContext:
    """Context manager for rate limiter acquisition."""

    def __init__(self, limiter: RateLimiter, tokens: int):
        """
        Initialize context.

        Args:
            limiter: RateLimiter instance
            tokens: Number of tokens to reserve
        """
        self.limiter = limiter
        self.tokens = tokens
        self.wait_time = 0.0

    async def __aenter__(self):
        """Acquire rate limit."""
        self.wait_time = await self.limiter._acquire_internal(self.tokens)
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Release rate limit (no-op for token bucket)."""
        return False
