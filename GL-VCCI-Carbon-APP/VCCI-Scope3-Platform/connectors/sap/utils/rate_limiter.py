# SAP Rate Limiter
# Token bucket algorithm for distributed rate limiting

"""
Rate Limiter with Token Bucket Algorithm
=========================================

Provides distributed rate limiting for SAP API calls using Redis-backed
token bucket algorithm.

Features:
---------
- Token bucket algorithm
- Redis-based distributed rate limiting
- Per-endpoint rate limiting
- Configurable limits (default: 10 requests/minute)
- Automatic throttling
- Rate limit tracking and metrics

Usage:
------
```python
from connectors.sap.utils.rate_limiter import RateLimiter

# Create rate limiter
limiter = RateLimiter(rate=10, per=60)  # 10 requests per 60 seconds

# Check if request is allowed
if limiter.acquire("/api/purchaseorders"):
    # Make API call
    response = call_api()
else:
    # Rate limited - wait and retry
    print("Rate limited")

# Get current rate limit status
status = limiter.get_status("/api/purchaseorders")
print(f"Tokens remaining: {status['tokens_remaining']}")
```
"""

import logging
import time
from typing import Dict, Optional

import redis
from redis import Redis
from redis.exceptions import RedisError

# Configure logger
logger = logging.getLogger(__name__)


class RateLimiter:
    """
    Token bucket rate limiter using Redis for distributed rate limiting.

    The token bucket algorithm allows bursts of traffic while maintaining
    a long-term average rate limit.
    """

    def __init__(
        self,
        rate: int = 10,
        per: int = 60,
        redis_client: Optional[Redis] = None,
        redis_url: str = "redis://localhost:6379/0",
        key_prefix: str = "sap:ratelimit",
    ):
        """
        Initialize the rate limiter.

        Args:
            rate: Number of requests allowed per period
            per: Time period in seconds (default: 60)
            redis_client: Existing Redis client (optional)
            redis_url: Redis connection URL (default: localhost)
            key_prefix: Prefix for Redis keys (default: "sap:ratelimit")
        """
        self.rate = rate
        self.per = per
        self.key_prefix = key_prefix

        # Initialize Redis client
        if redis_client:
            self.redis = redis_client
        else:
            try:
                self.redis = redis.from_url(
                    redis_url,
                    decode_responses=True,
                    socket_connect_timeout=5,
                )
                # Test connection
                self.redis.ping()
                logger.info(f"Connected to Redis at {redis_url}")
            except RedisError as e:
                logger.error(f"Failed to connect to Redis: {e}")
                raise

    def acquire(self, endpoint: str, tokens: int = 1) -> bool:
        """
        Attempt to acquire tokens from the bucket.

        Args:
            endpoint: API endpoint identifier
            tokens: Number of tokens to acquire (default: 1)

        Returns:
            True if tokens acquired successfully, False if rate limited
        """
        key = f"{self.key_prefix}:{endpoint}"
        now = time.time()

        try:
            # Use Redis pipeline for atomic operations
            pipe = self.redis.pipeline()

            # Get current bucket state
            bucket_data = self.redis.get(key)

            if bucket_data:
                # Parse existing bucket state
                last_refill_str, tokens_str = bucket_data.split(":")
                last_refill = float(last_refill_str)
                available_tokens = float(tokens_str)
            else:
                # Initialize new bucket
                last_refill = now
                available_tokens = float(self.rate)

            # Calculate token refill
            time_passed = now - last_refill
            refill_tokens = (time_passed / self.per) * self.rate
            available_tokens = min(available_tokens + refill_tokens, float(self.rate))
            last_refill = now

            # Check if enough tokens available
            if available_tokens >= tokens:
                # Consume tokens
                available_tokens -= tokens
                success = True
            else:
                # Not enough tokens
                success = False

            # Update bucket state in Redis
            bucket_value = f"{last_refill}:{available_tokens}"
            pipe.set(key, bucket_value, ex=self.per * 2)  # TTL: 2x the period
            pipe.execute()

            if success:
                logger.debug(
                    f"Rate limit acquired for {endpoint}. "
                    f"Tokens remaining: {available_tokens:.2f}/{self.rate}"
                )
            else:
                logger.warning(
                    f"Rate limit exceeded for {endpoint}. "
                    f"Tokens available: {available_tokens:.2f}/{self.rate}"
                )

            return success

        except RedisError as e:
            logger.error(f"Redis error in rate limiter: {e}")
            # Fail open - allow request if Redis is down
            return True

    def get_status(self, endpoint: str) -> Dict[str, float]:
        """
        Get current rate limit status for an endpoint.

        Args:
            endpoint: API endpoint identifier

        Returns:
            Dictionary with rate limit status:
            - tokens_remaining: Tokens available
            - tokens_max: Maximum tokens
            - reset_time: Time until bucket is full (seconds)
        """
        key = f"{self.key_prefix}:{endpoint}"
        now = time.time()

        try:
            bucket_data = self.redis.get(key)

            if bucket_data:
                last_refill_str, tokens_str = bucket_data.split(":")
                last_refill = float(last_refill_str)
                available_tokens = float(tokens_str)

                # Calculate current tokens with refill
                time_passed = now - last_refill
                refill_tokens = (time_passed / self.per) * self.rate
                current_tokens = min(
                    available_tokens + refill_tokens, float(self.rate)
                )

                # Calculate time until full
                if current_tokens < self.rate:
                    tokens_needed = self.rate - current_tokens
                    reset_time = (tokens_needed / self.rate) * self.per
                else:
                    reset_time = 0.0

                return {
                    "tokens_remaining": current_tokens,
                    "tokens_max": float(self.rate),
                    "reset_time": reset_time,
                }
            else:
                # No bucket exists - full capacity
                return {
                    "tokens_remaining": float(self.rate),
                    "tokens_max": float(self.rate),
                    "reset_time": 0.0,
                }

        except RedisError as e:
            logger.error(f"Redis error getting rate limit status: {e}")
            return {
                "tokens_remaining": float(self.rate),
                "tokens_max": float(self.rate),
                "reset_time": 0.0,
            }

    def wait_if_needed(self, endpoint: str, timeout: float = 60.0) -> bool:
        """
        Wait until rate limit allows the request.

        Args:
            endpoint: API endpoint identifier
            timeout: Maximum time to wait in seconds (default: 60)

        Returns:
            True if acquired within timeout, False if timeout exceeded
        """
        start_time = time.time()

        while True:
            if self.acquire(endpoint):
                return True

            # Check timeout
            elapsed = time.time() - start_time
            if elapsed >= timeout:
                logger.error(
                    f"Rate limit wait timeout ({timeout}s) exceeded for {endpoint}"
                )
                return False

            # Get status to determine wait time
            status = self.get_status(endpoint)
            wait_time = min(status["reset_time"] / 2, 5.0)  # Wait up to 5s at a time

            logger.info(
                f"Rate limited for {endpoint}. "
                f"Waiting {wait_time:.2f}s. Tokens remaining: {status['tokens_remaining']:.2f}"
            )
            time.sleep(wait_time)

    def reset(self, endpoint: str) -> None:
        """
        Reset rate limit for an endpoint (for testing/admin).

        Args:
            endpoint: API endpoint identifier
        """
        key = f"{self.key_prefix}:{endpoint}"
        try:
            self.redis.delete(key)
            logger.info(f"Rate limit reset for {endpoint}")
        except RedisError as e:
            logger.error(f"Redis error resetting rate limit: {e}")
