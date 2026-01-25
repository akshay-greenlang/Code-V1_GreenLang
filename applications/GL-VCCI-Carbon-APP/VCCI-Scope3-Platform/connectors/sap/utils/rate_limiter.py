# -*- coding: utf-8 -*-
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
from greenlang.determinism import FinancialDecimal

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

from greenlang.cache import get_cache_manager, initialize_cache_manager, CacheManager

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
        redis_client: Optional[object] = None,  # Deprecated, kept for compatibility
        redis_url: str = "redis://localhost:6379/0",  # Deprecated
        key_prefix: str = "sap:ratelimit",
        cache_manager: Optional[CacheManager] = None,
    ):
        """
        Initialize the rate limiter.

        Args:
            rate: Number of requests allowed per period
            per: Time period in seconds (default: 60)
            redis_client: Deprecated - kept for compatibility
            redis_url: Deprecated - kept for compatibility
            key_prefix: Prefix for cache keys (default: "sap:ratelimit")
            cache_manager: Optional CacheManager instance (will use global if not provided)
        """
        self.rate = rate
        self.per = per
        self.key_prefix = key_prefix
        self._namespace = "rate_limiter"

        # Initialize CacheManager
        if cache_manager:
            self.cache_manager = cache_manager
        else:
            self.cache_manager = get_cache_manager()
            if self.cache_manager is None:
                logger.warning(
                    "CacheManager not initialized. Rate limiter functionality may be limited. "
                    "Consider initializing CacheManager at application startup."
                )

        logger.info(f"Rate limiter initialized: {rate} requests per {per}s")

    async def acquire(self, endpoint: str, tokens: int = 1) -> bool:
        """
        Attempt to acquire tokens from the bucket.

        Args:
            endpoint: API endpoint identifier
            tokens: Number of tokens to acquire (default: 1)

        Returns:
            True if tokens acquired successfully, False if rate limited
        """
        if not self.cache_manager:
            logger.warning("CacheManager not available, allowing request")
            return True

        key = f"{self.key_prefix}:{endpoint}"
        now = time.time()

        try:
            # Get current bucket state from cache
            bucket_data = await self.cache_manager.get(key, namespace=self._namespace)

            if bucket_data:
                # Parse existing bucket state
                last_refill = bucket_data.get("last_refill", now)
                available_tokens = bucket_data.get("tokens", FinancialDecimal.from_string(self.rate))
            else:
                # Initialize new bucket
                last_refill = now
                available_tokens = FinancialDecimal.from_string(self.rate)

            # Calculate token refill
            time_passed = now - last_refill
            refill_tokens = (time_passed / self.per) * self.rate
            available_tokens = min(available_tokens + refill_tokens, FinancialDecimal.from_string(self.rate))
            last_refill = now

            # Check if enough tokens available
            if available_tokens >= tokens:
                # Consume tokens
                available_tokens -= tokens
                success = True
            else:
                # Not enough tokens
                success = False

            # Update bucket state in cache
            bucket_value = {
                "last_refill": last_refill,
                "tokens": available_tokens
            }
            await self.cache_manager.set(
                key,
                bucket_value,
                ttl=self.per * 2,  # TTL: 2x the period
                namespace=self._namespace
            )

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

        except Exception as e:
            logger.error(f"Error in rate limiter: {e}")
            # Fail open - allow request if cache is down
            return True

    async def get_status(self, endpoint: str) -> Dict[str, float]:
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
        if not self.cache_manager:
            return {
                "tokens_remaining": FinancialDecimal.from_string(self.rate),
                "tokens_max": FinancialDecimal.from_string(self.rate),
                "reset_time": 0.0,
            }

        key = f"{self.key_prefix}:{endpoint}"
        now = time.time()

        try:
            bucket_data = await self.cache_manager.get(key, namespace=self._namespace)

            if bucket_data:
                last_refill = bucket_data.get("last_refill", now)
                available_tokens = bucket_data.get("tokens", FinancialDecimal.from_string(self.rate))

                # Calculate current tokens with refill
                time_passed = now - last_refill
                refill_tokens = (time_passed / self.per) * self.rate
                current_tokens = min(
                    available_tokens + refill_tokens, FinancialDecimal.from_string(self.rate)
                )

                # Calculate time until full
                if current_tokens < self.rate:
                    tokens_needed = self.rate - current_tokens
                    reset_time = (tokens_needed / self.rate) * self.per
                else:
                    reset_time = 0.0

                return {
                    "tokens_remaining": current_tokens,
                    "tokens_max": FinancialDecimal.from_string(self.rate),
                    "reset_time": reset_time,
                }
            else:
                # No bucket exists - full capacity
                return {
                    "tokens_remaining": FinancialDecimal.from_string(self.rate),
                    "tokens_max": FinancialDecimal.from_string(self.rate),
                    "reset_time": 0.0,
                }

        except Exception as e:
            logger.error(f"Error getting rate limit status: {e}")
            return {
                "tokens_remaining": FinancialDecimal.from_string(self.rate),
                "tokens_max": FinancialDecimal.from_string(self.rate),
                "reset_time": 0.0,
            }

    async def wait_if_needed(self, endpoint: str, timeout: float = 60.0) -> bool:
        """
        Wait until rate limit allows the request.

        Args:
            endpoint: API endpoint identifier
            timeout: Maximum time to wait in seconds (default: 60)

        Returns:
            True if acquired within timeout, False if timeout exceeded
        """
        import asyncio
        start_time = time.time()

        while True:
            if await self.acquire(endpoint):
                return True

            # Check timeout
            elapsed = time.time() - start_time
            if elapsed >= timeout:
                logger.error(
                    f"Rate limit wait timeout ({timeout}s) exceeded for {endpoint}"
                )
                return False

            # Get status to determine wait time
            status = await self.get_status(endpoint)
            wait_time = min(status["reset_time"] / 2, 5.0)  # Wait up to 5s at a time

            logger.info(
                f"Rate limited for {endpoint}. "
                f"Waiting {wait_time:.2f}s. Tokens remaining: {status['tokens_remaining']:.2f}"
            )
            await asyncio.sleep(wait_time)

    async def reset(self, endpoint: str) -> None:
        """
        Reset rate limit for an endpoint (for testing/admin).

        Args:
            endpoint: API endpoint identifier
        """
        if not self.cache_manager:
            return

        key = f"{self.key_prefix}:{endpoint}"
        try:
            await self.cache_manager.invalidate(key, namespace=self._namespace)
            logger.info(f"Rate limit reset for {endpoint}")
        except Exception as e:
            logger.error(f"Error resetting rate limit: {e}")
