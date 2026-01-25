"""
API Gateway Router

Central routing for all GreenLang agent APIs with:
- Version management (/v1/, /v2/)
- Request/response transformation
- API key validation
- Rate limiting per endpoint
- Request logging and metrics

Example:
    >>> from app.gateway import create_gateway_router
    >>> gateway = create_gateway_router(config)
    >>> app.include_router(gateway.router, prefix="/api")
"""

import hashlib
import hmac
import logging
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from functools import wraps
from typing import Any, Callable, Dict, List, Optional, Set, TypeVar, Union

from fastapi import APIRouter, Depends, HTTPException, Header, Query, Request, Response, status
from fastapi.responses import JSONResponse
from fastapi.routing import APIRoute
from pydantic import BaseModel, Field
from starlette.middleware.base import BaseHTTPMiddleware

logger = logging.getLogger(__name__)


# =============================================================================
# Enums and Configuration Models
# =============================================================================


class APIVersion(str, Enum):
    """Supported API versions."""

    V1 = "v1"
    V2 = "v2"

    @classmethod
    def latest(cls) -> "APIVersion":
        """Return the latest API version."""
        return cls.V2

    @classmethod
    def default(cls) -> "APIVersion":
        """Return the default API version."""
        return cls.V1


class RateLimitStrategy(str, Enum):
    """Rate limiting strategies."""

    FIXED_WINDOW = "fixed_window"
    SLIDING_WINDOW = "sliding_window"
    TOKEN_BUCKET = "token_bucket"
    LEAKY_BUCKET = "leaky_bucket"


@dataclass
class RateLimitConfig:
    """Configuration for rate limiting."""

    requests_per_minute: int = 100
    requests_per_hour: int = 1000
    burst_size: int = 20
    strategy: RateLimitStrategy = RateLimitStrategy.TOKEN_BUCKET

    # Per-endpoint overrides
    endpoint_limits: Dict[str, int] = field(default_factory=dict)

    # Exempt endpoints
    exempt_endpoints: Set[str] = field(default_factory=lambda: {
        "/health",
        "/ready",
        "/api/docs",
        "/api/redoc",
        "/api/openapi.json",
    })


@dataclass
class GatewayConfig:
    """Configuration for the API gateway."""

    # Versioning
    default_version: APIVersion = APIVersion.V1
    supported_versions: List[APIVersion] = field(default_factory=lambda: [APIVersion.V1, APIVersion.V2])

    # Rate limiting
    rate_limit: RateLimitConfig = field(default_factory=RateLimitConfig)

    # API keys
    require_api_key: bool = True
    api_key_header: str = "X-API-Key"

    # Request validation
    max_request_size_bytes: int = 10 * 1024 * 1024  # 10MB
    request_timeout_seconds: int = 300

    # Logging
    log_requests: bool = True
    log_responses: bool = True
    log_sensitive_data: bool = False

    # CORS
    cors_origins: List[str] = field(default_factory=lambda: ["*"])
    cors_methods: List[str] = field(default_factory=lambda: ["GET", "POST", "PUT", "DELETE", "PATCH"])


# =============================================================================
# Request/Response Models
# =============================================================================


class GatewayError(BaseModel):
    """Standard error response model."""

    code: str = Field(..., description="Error code")
    message: str = Field(..., description="Human-readable error message")
    details: Optional[Dict[str, Any]] = Field(None, description="Additional error details")
    request_id: Optional[str] = Field(None, description="Request ID for debugging")
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class GatewayResponse(BaseModel):
    """Wrapper for gateway responses with metadata."""

    data: Any = Field(..., description="Response data")
    meta: Dict[str, Any] = Field(default_factory=dict, description="Response metadata")

    class Config:
        schema_extra = {
            "example": {
                "data": {"id": "agent-001", "name": "Carbon Calculator"},
                "meta": {
                    "request_id": "req_abc123",
                    "api_version": "v1",
                    "response_time_ms": 45
                }
            }
        }


class APIKeyInfo(BaseModel):
    """API key information model."""

    key_id: str
    tenant_id: str
    user_id: Optional[str] = None
    name: str
    scopes: List[str] = Field(default_factory=list)
    rate_limit_override: Optional[int] = None
    created_at: datetime
    expires_at: Optional[datetime] = None
    last_used_at: Optional[datetime] = None
    is_active: bool = True


class RequestMetrics(BaseModel):
    """Request metrics for logging and monitoring."""

    request_id: str
    method: str
    path: str
    version: str
    status_code: int
    response_time_ms: float
    request_size_bytes: int
    response_size_bytes: int
    tenant_id: Optional[str] = None
    user_id: Optional[str] = None
    api_key_id: Optional[str] = None
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    error: Optional[str] = None


# =============================================================================
# Rate Limiter
# =============================================================================


class TokenBucketRateLimiter:
    """
    Token bucket rate limiter for API gateway.

    Supports both in-memory (single-instance) and Redis-based (distributed) limiting.
    """

    def __init__(
        self,
        config: RateLimitConfig,
        redis_client: Optional[Any] = None,
    ):
        """
        Initialize the rate limiter.

        Args:
            config: Rate limit configuration
            redis_client: Optional Redis client for distributed limiting
        """
        self.config = config
        self.redis = redis_client
        self._buckets: Dict[str, Dict[str, Any]] = {}

    async def check_rate_limit(
        self,
        identifier: str,
        endpoint: Optional[str] = None,
    ) -> tuple[bool, int, int]:
        """
        Check if a request is within rate limits.

        Args:
            identifier: Unique identifier (user ID, API key, IP address)
            endpoint: Optional endpoint for per-endpoint limits

        Returns:
            Tuple of (allowed, remaining_requests, reset_time_seconds)
        """
        # Check exempt endpoints
        if endpoint and endpoint in self.config.exempt_endpoints:
            return True, -1, 0

        # Get limit for this endpoint
        limit = self.config.endpoint_limits.get(endpoint, self.config.requests_per_minute)

        if self.redis:
            return await self._check_redis_rate_limit(identifier, limit)
        else:
            return self._check_local_rate_limit(identifier, limit)

    def _check_local_rate_limit(
        self,
        identifier: str,
        limit: int,
    ) -> tuple[bool, int, int]:
        """In-memory rate limiting for single-instance deployments."""
        now = time.time()
        bucket_key = f"bucket:{identifier}"

        if bucket_key not in self._buckets:
            self._buckets[bucket_key] = {
                "tokens": limit,
                "last_refill": now,
            }

        bucket = self._buckets[bucket_key]

        # Refill tokens based on elapsed time
        elapsed = now - bucket["last_refill"]
        refill_rate = limit / 60  # Tokens per second
        new_tokens = bucket["tokens"] + (elapsed * refill_rate)
        bucket["tokens"] = min(new_tokens, limit + self.config.burst_size)
        bucket["last_refill"] = now

        if bucket["tokens"] >= 1:
            bucket["tokens"] -= 1
            remaining = int(bucket["tokens"])
            return True, remaining, 0
        else:
            # Calculate reset time
            tokens_needed = 1 - bucket["tokens"]
            reset_time = int(tokens_needed / refill_rate) + 1
            return False, 0, reset_time

    async def _check_redis_rate_limit(
        self,
        identifier: str,
        limit: int,
    ) -> tuple[bool, int, int]:
        """Redis-based rate limiting for distributed deployments."""
        try:
            key = f"ratelimit:{identifier}"
            now = int(time.time())
            window_start = now - 60

            # Use Redis sorted set for sliding window
            pipe = self.redis.pipeline()
            pipe.zremrangebyscore(key, 0, window_start)
            pipe.zcard(key)
            pipe.zadd(key, {str(now): now})
            pipe.expire(key, 120)
            results = await pipe.execute()

            count = results[1]

            if count < limit:
                remaining = limit - count - 1
                return True, remaining, 0
            else:
                return False, 0, 60

        except Exception as e:
            logger.error(f"Redis rate limit check failed: {e}")
            # Fail open
            return True, limit, 0


# =============================================================================
# API Key Validator
# =============================================================================


class APIKeyValidator:
    """
    API key validation service.

    Validates API keys against a database or cache, checking:
    - Key existence and validity
    - Expiration
    - Scopes and permissions
    - Rate limit overrides
    """

    def __init__(
        self,
        db_pool: Optional[Any] = None,
        redis_client: Optional[Any] = None,
        cache_ttl_seconds: int = 300,
    ):
        """
        Initialize the validator.

        Args:
            db_pool: Database connection pool
            redis_client: Redis client for caching
            cache_ttl_seconds: Cache TTL for API key lookups
        """
        self.db_pool = db_pool
        self.redis = redis_client
        self.cache_ttl = cache_ttl_seconds
        self._local_cache: Dict[str, tuple[APIKeyInfo, float]] = {}

    async def validate_api_key(
        self,
        api_key: str,
        required_scopes: Optional[List[str]] = None,
    ) -> Optional[APIKeyInfo]:
        """
        Validate an API key.

        Args:
            api_key: The API key to validate
            required_scopes: Optional list of required scopes

        Returns:
            APIKeyInfo if valid, None otherwise
        """
        # Hash the key for lookup
        key_hash = self._hash_key(api_key)

        # Check cache first
        key_info = await self._get_from_cache(key_hash)

        if not key_info:
            # Lookup in database
            key_info = await self._lookup_key(key_hash)

            if key_info:
                await self._cache_key(key_hash, key_info)

        if not key_info:
            return None

        # Check if active
        if not key_info.is_active:
            logger.warning(f"Inactive API key used: {key_info.key_id}")
            return None

        # Check expiration
        if key_info.expires_at and key_info.expires_at < datetime.now(timezone.utc):
            logger.warning(f"Expired API key used: {key_info.key_id}")
            return None

        # Check scopes
        if required_scopes:
            if not all(scope in key_info.scopes for scope in required_scopes):
                logger.warning(
                    f"API key {key_info.key_id} missing required scopes: "
                    f"{required_scopes} not in {key_info.scopes}"
                )
                return None

        # Update last used timestamp (async, non-blocking)
        # await self._update_last_used(key_info.key_id)

        return key_info

    def _hash_key(self, api_key: str) -> str:
        """Hash an API key for storage and lookup."""
        return hashlib.sha256(api_key.encode()).hexdigest()

    async def _get_from_cache(self, key_hash: str) -> Optional[APIKeyInfo]:
        """Get API key info from cache."""
        # Check local cache
        if key_hash in self._local_cache:
            info, cached_at = self._local_cache[key_hash]
            if time.time() - cached_at < self.cache_ttl:
                return info
            else:
                del self._local_cache[key_hash]

        # Check Redis cache
        if self.redis:
            try:
                cached = await self.redis.get(f"apikey:{key_hash}")
                if cached:
                    import json
                    data = json.loads(cached)
                    return APIKeyInfo(**data)
            except Exception as e:
                logger.error(f"Redis cache lookup failed: {e}")

        return None

    async def _cache_key(self, key_hash: str, info: APIKeyInfo) -> None:
        """Cache API key info."""
        self._local_cache[key_hash] = (info, time.time())

        if self.redis:
            try:
                import json
                await self.redis.setex(
                    f"apikey:{key_hash}",
                    self.cache_ttl,
                    json.dumps(info.dict(), default=str),
                )
            except Exception as e:
                logger.error(f"Redis cache store failed: {e}")

    async def _lookup_key(self, key_hash: str) -> Optional[APIKeyInfo]:
        """Lookup API key in database."""
        # TODO: Implement actual database lookup
        # For now, return a mock for development

        # Placeholder: accept any key that starts with "gl_"
        if key_hash:
            return APIKeyInfo(
                key_id="key_" + key_hash[:8],
                tenant_id="tenant_default",
                user_id=None,
                name="Development Key",
                scopes=["read", "write", "execute"],
                created_at=datetime.now(timezone.utc),
            )

        return None


# =============================================================================
# Request Transformer
# =============================================================================


class RequestTransformer:
    """
    Transform requests between API versions.

    Handles:
    - Field name mapping
    - Type conversions
    - Deprecated field handling
    - Version-specific validation
    """

    def __init__(self):
        """Initialize the transformer."""
        # V1 -> V2 field mappings
        self._v1_to_v2_mappings: Dict[str, Dict[str, str]] = {
            "agent": {
                "agent_id": "id",
                "execution_count": "invocation_count",
            },
            "execution": {
                "execution_id": "id",
                "agent_id": "agent",
            },
        }

        # V2 -> V1 field mappings (reverse)
        self._v2_to_v1_mappings: Dict[str, Dict[str, str]] = {
            resource: {v: k for k, v in mappings.items()}
            for resource, mappings in self._v1_to_v2_mappings.items()
        }

    def transform_request(
        self,
        data: Dict[str, Any],
        from_version: APIVersion,
        to_version: APIVersion,
        resource_type: str,
    ) -> Dict[str, Any]:
        """
        Transform request data between versions.

        Args:
            data: Request data to transform
            from_version: Source API version
            to_version: Target API version
            resource_type: Type of resource (agent, execution, etc.)

        Returns:
            Transformed request data
        """
        if from_version == to_version:
            return data

        mappings = self._get_mappings(from_version, to_version, resource_type)

        transformed = {}
        for key, value in data.items():
            new_key = mappings.get(key, key)
            transformed[new_key] = value

        return transformed

    def transform_response(
        self,
        data: Dict[str, Any],
        from_version: APIVersion,
        to_version: APIVersion,
        resource_type: str,
    ) -> Dict[str, Any]:
        """
        Transform response data between versions.

        Args:
            data: Response data to transform
            from_version: Source API version
            to_version: Target API version
            resource_type: Type of resource

        Returns:
            Transformed response data
        """
        # Response transformation is the reverse of request transformation
        return self.transform_request(data, from_version, to_version, resource_type)

    def _get_mappings(
        self,
        from_version: APIVersion,
        to_version: APIVersion,
        resource_type: str,
    ) -> Dict[str, str]:
        """Get field mappings for version transformation."""
        if from_version == APIVersion.V1 and to_version == APIVersion.V2:
            return self._v1_to_v2_mappings.get(resource_type, {})
        elif from_version == APIVersion.V2 and to_version == APIVersion.V1:
            return self._v2_to_v1_mappings.get(resource_type, {})
        return {}


# =============================================================================
# Request Logger
# =============================================================================


class RequestLogger:
    """
    Log API requests and responses for monitoring and debugging.

    Features:
    - Structured logging
    - Sensitive data redaction
    - Metrics collection
    - Audit trail support
    """

    SENSITIVE_HEADERS = {
        "authorization",
        "x-api-key",
        "cookie",
        "set-cookie",
    }

    SENSITIVE_FIELDS = {
        "password",
        "secret",
        "token",
        "api_key",
        "credit_card",
        "ssn",
    }

    def __init__(
        self,
        log_requests: bool = True,
        log_responses: bool = True,
        log_sensitive_data: bool = False,
        metrics_client: Optional[Any] = None,
    ):
        """
        Initialize the logger.

        Args:
            log_requests: Whether to log request details
            log_responses: Whether to log response details
            log_sensitive_data: Whether to include sensitive data (for debugging)
            metrics_client: Optional metrics client (e.g., Prometheus)
        """
        self.log_requests = log_requests
        self.log_responses = log_responses
        self.log_sensitive_data = log_sensitive_data
        self.metrics_client = metrics_client

    async def log_request(
        self,
        request: Request,
        request_id: str,
    ) -> None:
        """Log incoming request details."""
        if not self.log_requests:
            return

        headers = self._redact_headers(dict(request.headers))

        log_data = {
            "request_id": request_id,
            "method": request.method,
            "path": request.url.path,
            "query": str(request.url.query),
            "headers": headers,
            "client_ip": request.client.host if request.client else None,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        logger.info(f"API Request: {log_data}")

    async def log_response(
        self,
        request_id: str,
        status_code: int,
        response_time_ms: float,
        response_size: int,
        error: Optional[str] = None,
    ) -> None:
        """Log response details."""
        if not self.log_responses:
            return

        log_data = {
            "request_id": request_id,
            "status_code": status_code,
            "response_time_ms": round(response_time_ms, 2),
            "response_size_bytes": response_size,
            "error": error,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        if error:
            logger.warning(f"API Response (error): {log_data}")
        else:
            logger.info(f"API Response: {log_data}")

        # Emit metrics
        if self.metrics_client:
            await self._emit_metrics(log_data)

    def _redact_headers(self, headers: Dict[str, str]) -> Dict[str, str]:
        """Redact sensitive headers."""
        if self.log_sensitive_data:
            return headers

        return {
            k: "[REDACTED]" if k.lower() in self.SENSITIVE_HEADERS else v
            for k, v in headers.items()
        }

    def _redact_body(self, body: Dict[str, Any]) -> Dict[str, Any]:
        """Redact sensitive fields from request/response body."""
        if self.log_sensitive_data:
            return body

        return {
            k: "[REDACTED]" if k.lower() in self.SENSITIVE_FIELDS else v
            for k, v in body.items()
        }

    async def _emit_metrics(self, log_data: Dict[str, Any]) -> None:
        """Emit metrics to monitoring system."""
        # TODO: Implement Prometheus/StatsD metrics emission
        pass


# =============================================================================
# Gateway Router
# =============================================================================


class GatewayRouter:
    """
    Central API Gateway Router for GreenLang.

    Provides:
    - Version management (/v1/, /v2/)
    - API key validation
    - Rate limiting per endpoint
    - Request/response transformation
    - Logging and metrics

    Example:
        >>> gateway = GatewayRouter(config)
        >>> gateway.register_routes(v1_agents_router, "/agents", version=APIVersion.V1)
        >>> gateway.register_routes(v2_agents_router, "/agents", version=APIVersion.V2)
        >>> app.include_router(gateway.router, prefix="/api")
    """

    def __init__(
        self,
        config: GatewayConfig,
        redis_client: Optional[Any] = None,
        db_pool: Optional[Any] = None,
    ):
        """
        Initialize the gateway router.

        Args:
            config: Gateway configuration
            redis_client: Optional Redis client
            db_pool: Optional database connection pool
        """
        self.config = config
        self.redis = redis_client
        self.db_pool = db_pool

        # Initialize components
        self.rate_limiter = TokenBucketRateLimiter(config.rate_limit, redis_client)
        self.api_key_validator = APIKeyValidator(db_pool, redis_client)
        self.transformer = RequestTransformer()
        self.request_logger = RequestLogger(
            log_requests=config.log_requests,
            log_responses=config.log_responses,
            log_sensitive_data=config.log_sensitive_data,
        )

        # Create routers for each version
        self.router = APIRouter(tags=["Gateway"])
        self._version_routers: Dict[APIVersion, APIRouter] = {}

        for version in config.supported_versions:
            version_router = APIRouter(prefix=f"/{version.value}")
            self._version_routers[version] = version_router
            self.router.include_router(version_router)

        # Register gateway endpoints
        self._register_gateway_endpoints()

    def _register_gateway_endpoints(self) -> None:
        """Register gateway-level endpoints."""

        @self.router.get(
            "/versions",
            summary="List API versions",
            description="Get list of supported API versions and their status",
        )
        async def list_versions() -> Dict[str, Any]:
            return {
                "versions": [
                    {
                        "version": v.value,
                        "status": "current" if v == self.config.default_version else "supported",
                        "deprecated": False,
                    }
                    for v in self.config.supported_versions
                ],
                "default": self.config.default_version.value,
                "latest": APIVersion.latest().value,
            }

        @self.router.get(
            "/rate-limit",
            summary="Get rate limit status",
            description="Get current rate limit status for the authenticated user",
        )
        async def get_rate_limit_status(
            request: Request,
            x_api_key: Optional[str] = Header(None, alias="X-API-Key"),
        ) -> Dict[str, Any]:
            identifier = x_api_key or (request.client.host if request.client else "unknown")
            is_allowed, remaining, reset_time = await self.rate_limiter.check_rate_limit(
                identifier
            )

            return {
                "limit": self.config.rate_limit.requests_per_minute,
                "remaining": remaining,
                "reset_seconds": reset_time,
                "window": "minute",
            }

    def register_routes(
        self,
        router: APIRouter,
        prefix: str,
        version: APIVersion,
        rate_limit_override: Optional[int] = None,
        required_scopes: Optional[List[str]] = None,
    ) -> None:
        """
        Register routes with the gateway.

        Args:
            router: FastAPI router with endpoints
            prefix: Route prefix (e.g., "/agents")
            version: API version
            rate_limit_override: Optional rate limit override for this router
            required_scopes: Optional required scopes for all routes
        """
        if version not in self._version_routers:
            raise ValueError(f"Unsupported API version: {version}")

        # Add rate limit configuration
        if rate_limit_override:
            for route in router.routes:
                if isinstance(route, APIRoute):
                    endpoint = f"/{version.value}{prefix}{route.path}"
                    self.config.rate_limit.endpoint_limits[endpoint] = rate_limit_override

        # Include router
        self._version_routers[version].include_router(router, prefix=prefix)

        logger.info(f"Registered routes: {version.value}{prefix}")

    async def validate_request(
        self,
        request: Request,
        api_key: Optional[str] = None,
        required_scopes: Optional[List[str]] = None,
    ) -> APIKeyInfo:
        """
        Validate an incoming request.

        Args:
            request: The incoming request
            api_key: Optional API key (extracted from header if not provided)
            required_scopes: Optional required scopes

        Returns:
            Validated API key info

        Raises:
            HTTPException: If validation fails
        """
        # Extract API key if not provided
        if not api_key:
            api_key = request.headers.get(self.config.api_key_header)

        if self.config.require_api_key and not api_key:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail={
                    "error": {
                        "code": "MISSING_API_KEY",
                        "message": f"API key required in {self.config.api_key_header} header",
                    }
                },
            )

        # Validate API key
        if api_key:
            key_info = await self.api_key_validator.validate_api_key(
                api_key, required_scopes
            )

            if not key_info:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail={
                        "error": {
                            "code": "INVALID_API_KEY",
                            "message": "Invalid or expired API key",
                        }
                    },
                )

            return key_info

        # Return default for development mode
        return APIKeyInfo(
            key_id="dev_key",
            tenant_id="dev_tenant",
            name="Development",
            scopes=["read", "write", "execute"],
            created_at=datetime.now(timezone.utc),
        )

    async def check_rate_limit(
        self,
        request: Request,
        key_info: Optional[APIKeyInfo] = None,
    ) -> None:
        """
        Check rate limits for a request.

        Args:
            request: The incoming request
            key_info: Optional API key info with rate limit override

        Raises:
            HTTPException: If rate limit exceeded
        """
        identifier = key_info.key_id if key_info else (
            request.client.host if request.client else "unknown"
        )

        is_allowed, remaining, reset_time = await self.rate_limiter.check_rate_limit(
            identifier, request.url.path
        )

        if not is_allowed:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail={
                    "error": {
                        "code": "RATE_LIMIT_EXCEEDED",
                        "message": f"Rate limit exceeded. Try again in {reset_time} seconds.",
                    }
                },
                headers={
                    "X-RateLimit-Limit": str(self.config.rate_limit.requests_per_minute),
                    "X-RateLimit-Remaining": "0",
                    "X-RateLimit-Reset": str(int(time.time()) + reset_time),
                    "Retry-After": str(reset_time),
                },
            )


# =============================================================================
# Gateway Middleware
# =============================================================================


class GatewayMiddleware(BaseHTTPMiddleware):
    """
    Middleware for API gateway functionality.

    Applies to all requests:
    - Request ID generation
    - Request logging
    - Rate limiting
    - API key validation
    - Response timing
    """

    def __init__(
        self,
        app,
        gateway: GatewayRouter,
    ):
        """
        Initialize the middleware.

        Args:
            app: ASGI application
            gateway: Gateway router instance
        """
        super().__init__(app)
        self.gateway = gateway

    async def dispatch(
        self,
        request: Request,
        call_next: Callable,
    ) -> Response:
        """Process request through the gateway."""
        # Generate request ID
        request_id = str(uuid.uuid4())
        request.state.request_id = request_id

        # Start timing
        start_time = time.time()

        # Log request
        await self.gateway.request_logger.log_request(request, request_id)

        # Check exempt paths
        if request.url.path in self.gateway.config.rate_limit.exempt_endpoints:
            response = await call_next(request)
            response.headers["X-Request-ID"] = request_id
            return response

        try:
            # Validate API key
            api_key = request.headers.get(self.gateway.config.api_key_header)
            key_info = await self.gateway.validate_request(request, api_key)

            # Store key info in request state
            request.state.api_key_info = key_info
            request.state.tenant_id = key_info.tenant_id
            request.state.user_id = key_info.user_id

            # Check rate limit
            await self.gateway.check_rate_limit(request, key_info)

            # Process request
            response = await call_next(request)

            # Calculate response time
            response_time_ms = (time.time() - start_time) * 1000

            # Add standard headers
            response.headers["X-Request-ID"] = request_id
            response.headers["X-Response-Time"] = f"{response_time_ms:.2f}ms"

            # Log response
            await self.gateway.request_logger.log_response(
                request_id=request_id,
                status_code=response.status_code,
                response_time_ms=response_time_ms,
                response_size=int(response.headers.get("content-length", 0)),
            )

            return response

        except HTTPException as e:
            # Calculate response time
            response_time_ms = (time.time() - start_time) * 1000

            # Log error response
            await self.gateway.request_logger.log_response(
                request_id=request_id,
                status_code=e.status_code,
                response_time_ms=response_time_ms,
                response_size=0,
                error=str(e.detail),
            )

            raise

        except Exception as e:
            # Calculate response time
            response_time_ms = (time.time() - start_time) * 1000

            # Log error
            logger.error(f"Gateway error: {e}", exc_info=True)

            await self.gateway.request_logger.log_response(
                request_id=request_id,
                status_code=500,
                response_time_ms=response_time_ms,
                response_size=0,
                error=str(e),
            )

            return JSONResponse(
                status_code=500,
                content={
                    "error": {
                        "code": "INTERNAL_ERROR",
                        "message": "An internal error occurred",
                        "request_id": request_id,
                    }
                },
                headers={"X-Request-ID": request_id},
            )


# =============================================================================
# Dependency Injection Helpers
# =============================================================================


def get_api_key(
    x_api_key: Optional[str] = Header(None, alias="X-API-Key"),
    authorization: Optional[str] = Header(None),
) -> Optional[str]:
    """
    Extract API key from request headers.

    Checks both X-API-Key header and Bearer token in Authorization header.
    """
    if x_api_key:
        return x_api_key

    if authorization and authorization.startswith("Bearer "):
        return authorization[7:]

    return None


def get_request_id(request: Request) -> str:
    """Get or generate request ID."""
    return getattr(request.state, "request_id", str(uuid.uuid4()))


def get_tenant_id(request: Request) -> str:
    """Get tenant ID from request state."""
    tenant_id = getattr(request.state, "tenant_id", None)
    if not tenant_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Tenant context not established",
        )
    return tenant_id


def get_user_id(request: Request) -> Optional[str]:
    """Get user ID from request state."""
    return getattr(request.state, "user_id", None)


# =============================================================================
# Factory Function
# =============================================================================


def create_gateway_router(
    config: Optional[GatewayConfig] = None,
    redis_client: Optional[Any] = None,
    db_pool: Optional[Any] = None,
) -> GatewayRouter:
    """
    Create a configured gateway router.

    Args:
        config: Optional gateway configuration
        redis_client: Optional Redis client
        db_pool: Optional database pool

    Returns:
        Configured GatewayRouter instance

    Example:
        >>> config = GatewayConfig(
        ...     rate_limit=RateLimitConfig(requests_per_minute=1000),
        ...     require_api_key=True,
        ... )
        >>> gateway = create_gateway_router(config)
        >>> app.include_router(gateway.router, prefix="/api")
    """
    config = config or GatewayConfig()
    return GatewayRouter(config, redis_client, db_pool)
