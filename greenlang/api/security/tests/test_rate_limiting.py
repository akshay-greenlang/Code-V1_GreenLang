"""
Tests for Rate Limiting

This module tests the rate limiting implementation.
"""

import time
import asyncio
import pytest
from unittest.mock import Mock, patch, AsyncMock

from fastapi import FastAPI, Request
from fastapi.testclient import TestClient

from greenlang.api.security.rate_limiting import (
    RateLimiter,
    RateLimitConfig,
    TokenBucket,
    RateLimitMiddleware,
    RateLimitPeriod
)


class TestTokenBucket:
    """Test TokenBucket algorithm."""

    def test_token_bucket_initialization(self):
        """Test token bucket initialization."""
        bucket = TokenBucket(capacity=10, refill_rate=2.0)
        assert bucket.capacity == 10
        assert bucket.refill_rate == 2.0
        assert bucket.tokens == 10.0

    def test_consume_tokens_success(self):
        """Test consuming tokens when available."""
        bucket = TokenBucket(capacity=10, refill_rate=2.0)
        allowed, wait_time = bucket.consume(5)

        assert allowed is True
        assert wait_time == 0
        assert bucket.tokens == 5.0

    def test_consume_tokens_failure(self):
        """Test consuming tokens when insufficient."""
        bucket = TokenBucket(capacity=10, refill_rate=2.0)
        bucket.tokens = 3.0

        allowed, wait_time = bucket.consume(5)

        assert allowed is False
        assert wait_time > 0
        assert bucket.tokens == 3.0

    def test_refill_tokens(self):
        """Test token refilling over time."""
        bucket = TokenBucket(capacity=10, refill_rate=2.0)
        bucket.tokens = 0

        # Simulate time passing
        bucket.last_refill = time.time() - 2  # 2 seconds ago
        bucket._refill()

        assert bucket.tokens == pytest.approx(4.0, rel=0.1)

    def test_refill_respects_capacity(self):
        """Test that refill doesn't exceed capacity."""
        bucket = TokenBucket(capacity=10, refill_rate=5.0)
        bucket.tokens = 8.0

        bucket.last_refill = time.time() - 1
        bucket._refill()

        assert bucket.tokens == 10.0  # Should not exceed capacity

    def test_reset_bucket(self):
        """Test resetting bucket to full capacity."""
        bucket = TokenBucket(capacity=10, refill_rate=2.0)
        bucket.tokens = 3.0

        bucket.reset()

        assert bucket.tokens == 10.0


class TestRateLimiter:
    """Test RateLimiter class."""

    @pytest.fixture
    def rate_limit_config(self):
        """Create test rate limit configuration."""
        return RateLimitConfig(
            redis_url=None,  # Use local rate limiting for tests
            default_limit=100,
            default_period=RateLimitPeriod.MINUTE,
            enable_distributed=False
        )

    @pytest.fixture
    def rate_limiter(self, rate_limit_config):
        """Create RateLimiter instance."""
        return RateLimiter(rate_limit_config)

    def test_parse_rate_limit_valid(self, rate_limiter):
        """Test parsing valid rate limit strings."""
        limit, period = rate_limiter.parse_rate_limit("100/minute")
        assert limit == 100
        assert period == 60.0

        limit, period = rate_limiter.parse_rate_limit("10/hour")
        assert limit == 10
        assert period == 3600.0

    def test_parse_rate_limit_invalid(self, rate_limiter):
        """Test parsing invalid rate limit strings."""
        with pytest.raises(ValueError):
            rate_limiter.parse_rate_limit("invalid")

        with pytest.raises(ValueError):
            rate_limiter.parse_rate_limit("100/invalid")

    def test_get_identifier_with_user(self, rate_limiter):
        """Test getting identifier for authenticated user."""
        request = Mock(spec=Request)
        request.state.user_id = "user123"

        identifier = rate_limiter._get_identifier(request)
        assert identifier == "user:user123"

    def test_get_identifier_with_ip(self, rate_limiter):
        """Test getting identifier for IP address."""
        request = Mock(spec=Request)
        request.state = Mock(spec=[])  # No user_id attribute
        request.client = Mock(host="192.168.1.1")

        identifier = rate_limiter._get_identifier(request)
        assert identifier == "ip:192.168.1.1"

    def test_get_cache_key(self, rate_limiter):
        """Test cache key generation."""
        key = rate_limiter._get_cache_key("user:123", "/api/data")
        assert key.startswith("greenlang:ratelimit:")
        assert len(key) > 20

    def test_check_local_limit_allowed(self, rate_limiter):
        """Test local rate limit check when allowed."""
        allowed, remaining, wait_time = rate_limiter._check_local_limit(
            "test_key", 10, 60.0
        )

        assert allowed is True
        assert remaining >= 0
        assert wait_time == 0

    def test_check_local_limit_blocked(self, rate_limiter):
        """Test local rate limit check when blocked."""
        # Exhaust the limit
        key = "test_key_blocked"
        for _ in range(10):
            rate_limiter._check_local_limit(key, 10, 60.0)

        # Next request should be blocked
        allowed, remaining, wait_time = rate_limiter._check_local_limit(
            key, 10, 60.0
        )

        assert allowed is False
        assert remaining == 0
        assert wait_time > 0

    def test_cleanup_local_buckets(self, rate_limiter):
        """Test cleanup of local buckets."""
        # Create many buckets
        for i in range(1100):
            rate_limiter.local_buckets[f"key_{i}"] = TokenBucket(10, 1.0)

        rate_limiter._cleanup_local_buckets()

        assert len(rate_limiter.local_buckets) <= 500

    @pytest.mark.asyncio
    async def test_check_rate_limit(self, rate_limiter):
        """Test check_rate_limit method."""
        request = Mock(spec=Request)
        request.url.path = "/api/data"
        request.state = Mock(spec=[])
        request.client = Mock(host="192.168.1.1")

        allowed, metadata = await rate_limiter.check_rate_limit(request)

        assert allowed is True
        assert "limit" in metadata
        assert "remaining" in metadata
        assert metadata["retry_after"] is None


class TestRateLimitMiddleware:
    """Test rate limit middleware integration."""

    @pytest.fixture
    def app(self):
        """Create test FastAPI app."""
        app = FastAPI()

        @app.get("/api/data")
        async def get_data():
            return {"data": "test"}

        @app.post("/api/limited")
        async def limited_endpoint():
            return {"status": "ok"}

        return app

    @pytest.fixture
    def client_with_rate_limit(self, app):
        """Create test client with rate limiting."""
        config = RateLimitConfig(
            default_limit=5,
            default_period=RateLimitPeriod.MINUTE,
            enable_distributed=False,
            endpoint_limits={
                "/api/limited": "2/minute"
            }
        )

        app.add_middleware(RateLimitMiddleware, config=config)
        return TestClient(app)

    def test_rate_limit_headers(self, client_with_rate_limit):
        """Test that rate limit headers are added."""
        response = client_with_rate_limit.get("/api/data")

        assert response.status_code == 200
        assert "X-RateLimit-Limit" in response.headers
        assert "X-RateLimit-Remaining" in response.headers
        assert "X-RateLimit-Reset" in response.headers

    def test_rate_limit_exceeded(self, client_with_rate_limit):
        """Test rate limit exceeded response."""
        # Make requests up to the limit
        for i in range(5):
            response = client_with_rate_limit.get("/api/data")
            assert response.status_code == 200

        # Next request should be rate limited
        response = client_with_rate_limit.get("/api/data")
        assert response.status_code == 429
        assert "Retry-After" in response.headers

    def test_endpoint_specific_limit(self, client_with_rate_limit):
        """Test endpoint-specific rate limits."""
        # Limited to 2 requests per minute
        response1 = client_with_rate_limit.post("/api/limited")
        assert response1.status_code == 200

        response2 = client_with_rate_limit.post("/api/limited")
        assert response2.status_code == 200

        response3 = client_with_rate_limit.post("/api/limited")
        assert response3.status_code == 429


class TestRateLimitDecorator:
    """Test rate limit decorator."""

    @pytest.fixture
    def app(self):
        """Create test FastAPI app with rate limit decorator."""
        app = FastAPI()

        config = RateLimitConfig(enable_distributed=False)
        limiter = RateLimiter(config)

        @app.get("/api/test")
        @limiter.limit("3/minute")
        async def test_endpoint(request: Request):
            return {"status": "ok"}

        return app

    @pytest.fixture
    def client(self, app):
        """Create test client."""
        return TestClient(app)

    def test_decorator_rate_limit(self, client):
        """Test rate limiting via decorator."""
        # First 3 requests should succeed
        for _ in range(3):
            response = client.get("/api/test")
            assert response.status_code == 200

        # Fourth request should be rate limited
        response = client.get("/api/test")
        assert response.status_code == 429
        assert "Rate limit exceeded" in response.json()["detail"]


@pytest.mark.asyncio
class TestRedisRateLimiting:
    """Test Redis-backed rate limiting."""

    @pytest.fixture
    async def mock_redis(self):
        """Create mock Redis client."""
        mock = AsyncMock()
        mock.pipeline.return_value.__aenter__.return_value = mock
        mock.execute.return_value = [None, 5, None, None]  # Mock pipeline results
        mock.zrange.return_value = [(b"timestamp", time.time())]
        return mock

    @pytest.fixture
    async def rate_limiter_with_redis(self, mock_redis):
        """Create rate limiter with mock Redis."""
        config = RateLimitConfig(
            redis_url="redis://localhost:6379",
            enable_distributed=True
        )
        limiter = RateLimiter(config)
        limiter.redis = mock_redis
        return limiter

    async def test_check_redis_limit_allowed(self, rate_limiter_with_redis):
        """Test Redis rate limit check when allowed."""
        rate_limiter_with_redis.redis.pipeline.return_value.__aenter__.return_value.execute.return_value = [
            None, 5, None, None
        ]

        allowed, remaining, reset_time = await rate_limiter_with_redis._check_redis_limit(
            "test_key", 10, 60.0
        )

        assert allowed is True
        assert remaining == 5
        assert reset_time > 0

    async def test_check_redis_limit_blocked(self, rate_limiter_with_redis):
        """Test Redis rate limit check when blocked."""
        rate_limiter_with_redis.redis.pipeline.return_value.__aenter__.return_value.execute.return_value = [
            None, 11, None, None
        ]

        allowed, remaining, reset_time = await rate_limiter_with_redis._check_redis_limit(
            "test_key", 10, 60.0
        )

        assert allowed is False
        assert remaining == 0
        assert reset_time > 0