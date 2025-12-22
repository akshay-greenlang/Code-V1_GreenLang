"""
GL-006 HEATRECLAIM - API Middleware

Custom middleware for authentication, rate limiting,
logging, and request/response processing.
"""

from datetime import datetime, timezone
from typing import Any, Callable, Dict, Optional
import hashlib
import logging
import time
import uuid

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response, JSONResponse

logger = logging.getLogger(__name__)


class RateLimitMiddleware(BaseHTTPMiddleware):
    """
    Rate limiting middleware.

    Implements token bucket algorithm for rate limiting.
    Limits requests per client IP address.
    """

    def __init__(
        self,
        app,
        requests_per_minute: int = 60,
        burst_size: int = 10,
    ) -> None:
        """
        Initialize rate limiter.

        Args:
            app: ASGI application
            requests_per_minute: Sustained request rate
            burst_size: Maximum burst above rate limit
        """
        super().__init__(app)
        self.requests_per_minute = requests_per_minute
        self.burst_size = burst_size
        self.tokens: Dict[str, float] = {}
        self.last_update: Dict[str, float] = {}

    async def dispatch(
        self,
        request: Request,
        call_next: Callable,
    ) -> Response:
        """Process request with rate limiting."""
        # Get client identifier
        client_ip = request.client.host if request.client else "unknown"
        client_id = f"{client_ip}"

        # Skip rate limiting for health checks
        if request.url.path in ["/health", "/api/v1/health"]:
            return await call_next(request)

        # Check rate limit
        if not self._check_rate_limit(client_id):
            return JSONResponse(
                status_code=429,
                content={
                    "error": "Rate limit exceeded",
                    "message": f"Maximum {self.requests_per_minute} requests per minute",
                    "retry_after_seconds": 60,
                },
                headers={"Retry-After": "60"},
            )

        response = await call_next(request)

        # Add rate limit headers
        remaining = self._get_remaining_tokens(client_id)
        response.headers["X-RateLimit-Limit"] = str(self.requests_per_minute)
        response.headers["X-RateLimit-Remaining"] = str(int(remaining))

        return response

    def _check_rate_limit(self, client_id: str) -> bool:
        """Check if request is within rate limit."""
        now = time.time()

        if client_id not in self.tokens:
            self.tokens[client_id] = float(self.burst_size)
            self.last_update[client_id] = now

        # Add tokens based on time elapsed
        elapsed = now - self.last_update[client_id]
        token_rate = self.requests_per_minute / 60.0
        self.tokens[client_id] = min(
            self.burst_size,
            self.tokens[client_id] + elapsed * token_rate
        )
        self.last_update[client_id] = now

        # Check if token available
        if self.tokens[client_id] >= 1.0:
            self.tokens[client_id] -= 1.0
            return True

        return False

    def _get_remaining_tokens(self, client_id: str) -> float:
        """Get remaining tokens for client."""
        return self.tokens.get(client_id, 0)


class AuthenticationMiddleware(BaseHTTPMiddleware):
    """
    Authentication middleware.

    Supports:
    - API key authentication
    - JWT token authentication
    - Service account authentication
    """

    def __init__(
        self,
        app,
        api_keys: Optional[Dict[str, str]] = None,
        require_auth: bool = False,
        exempt_paths: Optional[list] = None,
    ) -> None:
        """
        Initialize authentication middleware.

        Args:
            app: ASGI application
            api_keys: Valid API keys (key -> user_id mapping)
            require_auth: Require authentication for all requests
            exempt_paths: Paths that don't require authentication
        """
        super().__init__(app)
        self.api_keys = api_keys or {}
        self.require_auth = require_auth
        self.exempt_paths = exempt_paths or [
            "/health",
            "/api/v1/health",
            "/docs",
            "/redoc",
            "/openapi.json",
        ]

    async def dispatch(
        self,
        request: Request,
        call_next: Callable,
    ) -> Response:
        """Process request with authentication."""
        # Skip auth for exempt paths
        if any(request.url.path.startswith(path) for path in self.exempt_paths):
            return await call_next(request)

        # Check for API key
        api_key = request.headers.get("X-API-Key")
        auth_header = request.headers.get("Authorization")

        user_id = None

        if api_key:
            user_id = self._validate_api_key(api_key)
        elif auth_header and auth_header.startswith("Bearer "):
            token = auth_header[7:]
            user_id = self._validate_jwt_token(token)

        if self.require_auth and not user_id:
            return JSONResponse(
                status_code=401,
                content={
                    "error": "Unauthorized",
                    "message": "Valid API key or JWT token required",
                },
                headers={"WWW-Authenticate": "Bearer"},
            )

        # Add user info to request state
        request.state.user_id = user_id
        request.state.authenticated = user_id is not None

        response = await call_next(request)
        return response

    def _validate_api_key(self, api_key: str) -> Optional[str]:
        """Validate API key and return user ID."""
        return self.api_keys.get(api_key)

    def _validate_jwt_token(self, token: str) -> Optional[str]:
        """Validate JWT token and return user ID."""
        # Placeholder - implement JWT validation
        try:
            # In production, use proper JWT validation
            # import jwt
            # payload = jwt.decode(token, SECRET_KEY, algorithms=["HS256"])
            # return payload.get("sub")
            return None
        except Exception:
            return None


class LoggingMiddleware(BaseHTTPMiddleware):
    """
    Request/response logging middleware.

    Logs:
    - Request method, path, client IP
    - Response status code
    - Processing time
    - Request ID for tracing
    """

    def __init__(
        self,
        app,
        log_request_body: bool = False,
        log_response_body: bool = False,
        exclude_paths: Optional[list] = None,
    ) -> None:
        """
        Initialize logging middleware.

        Args:
            app: ASGI application
            log_request_body: Log request body content
            log_response_body: Log response body content
            exclude_paths: Paths to exclude from logging
        """
        super().__init__(app)
        self.log_request_body = log_request_body
        self.log_response_body = log_response_body
        self.exclude_paths = exclude_paths or ["/health"]

    async def dispatch(
        self,
        request: Request,
        call_next: Callable,
    ) -> Response:
        """Process request with logging."""
        # Skip logging for excluded paths
        if any(request.url.path.startswith(path) for path in self.exclude_paths):
            return await call_next(request)

        # Generate request ID
        request_id = str(uuid.uuid4())[:8]
        request.state.request_id = request_id

        # Log request
        client_ip = request.client.host if request.client else "unknown"
        start_time = time.time()

        logger.info(
            f"[{request_id}] {request.method} {request.url.path} "
            f"from {client_ip}"
        )

        if self.log_request_body and request.method in ["POST", "PUT", "PATCH"]:
            try:
                body = await request.body()
                logger.debug(f"[{request_id}] Request body: {body[:500]}")
            except Exception:
                pass

        # Process request
        try:
            response = await call_next(request)
        except Exception as e:
            logger.error(
                f"[{request_id}] Error processing request: {e}",
                exc_info=True,
            )
            raise

        # Calculate processing time
        process_time = time.time() - start_time

        # Log response
        logger.info(
            f"[{request_id}] {response.status_code} "
            f"({process_time*1000:.1f}ms)"
        )

        # Add headers
        response.headers["X-Request-ID"] = request_id
        response.headers["X-Process-Time"] = f"{process_time:.3f}"

        return response


class ProvenanceMiddleware(BaseHTTPMiddleware):
    """
    Provenance tracking middleware.

    Adds computation provenance headers to responses
    for audit trail compliance.
    """

    def __init__(self, app) -> None:
        """Initialize provenance middleware."""
        super().__init__(app)

    async def dispatch(
        self,
        request: Request,
        call_next: Callable,
    ) -> Response:
        """Process request with provenance tracking."""
        response = await call_next(request)

        # Add provenance headers
        response.headers["X-GL-Agent"] = "GL-006-HEATRECLAIM"
        response.headers["X-GL-Version"] = "1.0.0"
        response.headers["X-GL-Timestamp"] = datetime.now(timezone.utc).isoformat()

        return response


class CacheMiddleware(BaseHTTPMiddleware):
    """
    Response caching middleware.

    Caches GET responses for deterministic queries.
    """

    def __init__(
        self,
        app,
        cache_ttl_seconds: int = 300,
        max_cache_size: int = 1000,
    ) -> None:
        """
        Initialize cache middleware.

        Args:
            app: ASGI application
            cache_ttl_seconds: Cache time-to-live
            max_cache_size: Maximum cached responses
        """
        super().__init__(app)
        self.cache_ttl = cache_ttl_seconds
        self.max_cache_size = max_cache_size
        self._cache: Dict[str, tuple] = {}

    async def dispatch(
        self,
        request: Request,
        call_next: Callable,
    ) -> Response:
        """Process request with caching."""
        # Only cache GET requests
        if request.method != "GET":
            return await call_next(request)

        # Generate cache key
        cache_key = self._generate_cache_key(request)

        # Check cache
        cached = self._get_cached(cache_key)
        if cached:
            response = Response(
                content=cached["content"],
                status_code=cached["status_code"],
                media_type=cached["media_type"],
            )
            response.headers["X-Cache"] = "HIT"
            return response

        # Process request
        response = await call_next(request)

        # Cache successful responses
        if response.status_code == 200:
            body = b""
            async for chunk in response.body_iterator:
                body += chunk

            self._set_cached(cache_key, {
                "content": body,
                "status_code": response.status_code,
                "media_type": response.media_type,
            })

            # Return new response with body
            new_response = Response(
                content=body,
                status_code=response.status_code,
                media_type=response.media_type,
            )
            new_response.headers["X-Cache"] = "MISS"
            return new_response

        return response

    def _generate_cache_key(self, request: Request) -> str:
        """Generate cache key from request."""
        key_data = f"{request.method}:{request.url.path}:{request.url.query}"
        return hashlib.sha256(key_data.encode()).hexdigest()[:16]

    def _get_cached(self, key: str) -> Optional[Dict[str, Any]]:
        """Get cached response if valid."""
        if key not in self._cache:
            return None

        data, timestamp = self._cache[key]
        if time.time() - timestamp > self.cache_ttl:
            del self._cache[key]
            return None

        return data

    def _set_cached(self, key: str, data: Dict[str, Any]) -> None:
        """Cache response."""
        # Evict oldest if at capacity
        if len(self._cache) >= self.max_cache_size:
            oldest = min(self._cache.items(), key=lambda x: x[1][1])
            del self._cache[oldest[0]]

        self._cache[key] = (data, time.time())
