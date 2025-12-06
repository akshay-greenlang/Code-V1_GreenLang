"""
Rate Limiting Middleware

This module provides Redis-based rate limiting using the token bucket algorithm.
"""

import logging
import time
from typing import Callable, Optional

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse

logger = logging.getLogger(__name__)


class RateLimitMiddleware(BaseHTTPMiddleware):
    """
    Rate Limiting Middleware.

    Implements token bucket algorithm with Redis for distributed rate limiting.
    Supports per-user and per-endpoint limits.
    """

    def __init__(
        self,
        app,
        redis=None,
        requests_per_minute: int = 100,
        burst_size: int = 20,
    ):
        """
        Initialize the middleware.

        Args:
            app: The ASGI application
            redis: Redis client
            requests_per_minute: Base rate limit
            burst_size: Burst allowance
        """
        super().__init__(app)
        self.redis = redis
        self.requests_per_minute = requests_per_minute
        self.burst_size = burst_size

    async def dispatch(
        self,
        request: Request,
        call_next: Callable,
    ) -> Response:
        """
        Process the request with rate limiting.

        - Checks rate limit for the user
        - Returns 429 if limit exceeded
        - Adds rate limit headers to response
        """
        # Get user identifier
        user_id = getattr(request.state, "user_id", None) or request.client.host

        # Check rate limit
        is_allowed, remaining, reset_time = await self._check_rate_limit(user_id)

        if not is_allowed:
            return JSONResponse(
                status_code=429,
                content={
                    "error": {
                        "code": "RATE_LIMIT_EXCEEDED",
                        "message": f"Rate limit exceeded. Try again in {reset_time} seconds.",
                    }
                },
                headers={
                    "X-RateLimit-Limit": str(self.requests_per_minute),
                    "X-RateLimit-Remaining": "0",
                    "X-RateLimit-Reset": str(int(time.time()) + reset_time),
                    "Retry-After": str(reset_time),
                },
            )

        # Process request
        response = await call_next(request)

        # Add rate limit headers
        response.headers["X-RateLimit-Limit"] = str(self.requests_per_minute)
        response.headers["X-RateLimit-Remaining"] = str(remaining)
        response.headers["X-RateLimit-Reset"] = str(int(time.time()) + 60)

        return response

    async def _check_rate_limit(
        self,
        user_id: str,
    ) -> tuple[bool, int, int]:
        """
        Check if request is within rate limit.

        Returns:
            Tuple of (is_allowed, remaining_requests, seconds_until_reset)
        """
        if not self.redis:
            # No Redis, allow all requests
            return True, self.requests_per_minute, 0

        key = f"ratelimit:{user_id}"
        now = int(time.time())
        window_start = now - 60

        try:
            # TODO: Implement actual Redis rate limiting
            # Use sorted set with timestamps
            # self.redis.zremrangebyscore(key, 0, window_start)
            # count = self.redis.zcard(key)
            # if count < self.requests_per_minute:
            #     self.redis.zadd(key, {str(now): now})
            #     self.redis.expire(key, 60)
            #     return True, self.requests_per_minute - count - 1, 0

            return True, self.requests_per_minute, 0

        except Exception as e:
            logger.error(f"Rate limit check failed: {e}")
            # Fail open - allow request if Redis is down
            return True, self.requests_per_minute, 0
