"""
Tenant Context Middleware - Multi-Tenancy Request Isolation

This module provides middleware for extracting tenant context from requests,
validating tenant status, and enforcing tenant isolation across all API operations.

The middleware extracts tenant identity from:
1. JWT token claims (tenant_id)
2. Subdomain extraction (e.g., acme.greenlang.io)
3. X-Tenant-ID header (for service-to-service calls)

Features:
- Tenant context injection into request state
- Tenant status validation (active, suspended, etc.)
- Usage tracking per request
- Rate limiting per tenant
- Feature flag enforcement

Example:
    >>> app.add_middleware(
    ...     TenantContextMiddleware,
    ...     secret_key="your-jwt-secret",
    ...     redis_client=redis
    ... )
"""

import logging
import time
from contextvars import ContextVar
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional
from uuid import UUID

import jwt
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse

from models.tenant import Tenant, TenantStatus, SubscriptionTier

logger = logging.getLogger(__name__)

# Context variable for tenant context (accessible anywhere in the request lifecycle)
_tenant_context: ContextVar[Optional["TenantContext"]] = ContextVar(
    "tenant_context", default=None
)


@dataclass
class TenantContext:
    """
    Tenant context for the current request.

    This dataclass holds all tenant-related information needed
    during request processing.

    Attributes:
        tenant_id: External tenant identifier
        tenant_uuid: Internal tenant UUID
        name: Tenant organization name
        slug: URL-safe slug
        subscription_tier: Subscription level
        status: Tenant status
        feature_flags: Enabled features
        quotas: Resource quotas
        current_usage: Current resource usage
        user_id: Authenticated user ID
        user_roles: User's roles within the tenant
        request_id: Unique request identifier
        start_time: Request start timestamp
    """

    tenant_id: str
    tenant_uuid: UUID
    name: str
    slug: str
    subscription_tier: SubscriptionTier
    status: TenantStatus
    feature_flags: Dict[str, bool] = field(default_factory=dict)
    quotas: Dict[str, int] = field(default_factory=dict)
    current_usage: Dict[str, int] = field(default_factory=dict)
    user_id: Optional[str] = None
    user_roles: List[str] = field(default_factory=list)
    request_id: Optional[str] = None
    start_time: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def is_feature_enabled(self, feature_name: str) -> bool:
        """Check if a feature is enabled for this tenant."""
        return self.feature_flags.get(feature_name, False)

    def check_quota(self, quota_name: str, increment: int = 1) -> bool:
        """Check if an operation would exceed quota."""
        limit = self.quotas.get(quota_name, 0)
        if limit == -1:  # Unlimited
            return True
        current = self.current_usage.get(quota_name, 0)
        return (current + increment) <= limit

    def has_role(self, role: str) -> bool:
        """Check if user has a specific role."""
        return role in self.user_roles

    def is_admin(self) -> bool:
        """Check if user is an admin."""
        return "admin" in self.user_roles

    def get_elapsed_time_ms(self) -> float:
        """Get elapsed time since request start in milliseconds."""
        return (time.time() - self.start_time) * 1000

    @classmethod
    def from_tenant(
        cls,
        tenant: Tenant,
        user_id: Optional[str] = None,
        user_roles: Optional[List[str]] = None,
        request_id: Optional[str] = None,
    ) -> "TenantContext":
        """Create TenantContext from Tenant model."""
        return cls(
            tenant_id=tenant.tenant_id,
            tenant_uuid=tenant.id,
            name=tenant.name,
            slug=tenant.slug,
            subscription_tier=tenant.subscription_tier,
            status=tenant.status,
            feature_flags=tenant.get_effective_feature_flags(),
            quotas=tenant.get_effective_quotas(),
            current_usage=tenant.current_usage or {},
            user_id=user_id,
            user_roles=user_roles or [],
            request_id=request_id,
        )


def get_tenant_context() -> Optional[TenantContext]:
    """
    Get the current tenant context.

    This function retrieves the tenant context from the context variable,
    making it accessible from anywhere in the request lifecycle.

    Returns:
        Current TenantContext or None if not set

    Example:
        >>> context = get_tenant_context()
        >>> if context:
        ...     print(f"Processing request for tenant: {context.tenant_id}")
    """
    return _tenant_context.get()


def set_tenant_context(context: TenantContext) -> None:
    """
    Set the tenant context for the current request.

    Args:
        context: TenantContext to set
    """
    _tenant_context.set(context)


def clear_tenant_context() -> None:
    """Clear the tenant context."""
    _tenant_context.set(None)


class TenantContextMiddleware(BaseHTTPMiddleware):
    """
    Tenant Context Middleware.

    Extracts tenant information from requests and injects tenant context
    into the request state for downstream handlers.

    The middleware:
    1. Extracts tenant ID from JWT, subdomain, or header
    2. Loads tenant from cache or database
    3. Validates tenant is active
    4. Injects TenantContext into request.state
    5. Tracks request usage metrics

    Attributes:
        secret_key: JWT signing secret
        algorithm: JWT algorithm
        redis_client: Redis client for caching
        tenant_cache_ttl: Cache TTL in seconds
        public_paths: Paths that don't require tenant context

    Example:
        >>> app.add_middleware(
        ...     TenantContextMiddleware,
        ...     secret_key="your-secret",
        ...     redis_client=redis
        ... )
    """

    # Paths that don't require tenant context
    PUBLIC_PATHS = {
        "/health",
        "/ready",
        "/docs",
        "/redoc",
        "/openapi.json",
        "/v1/auth/login",
        "/v1/auth/register",
        "/v1/auth/refresh",
    }

    # Paths that require system-level access (no tenant context)
    SYSTEM_PATHS = {
        "/v1/admin/",
        "/v1/system/",
    }

    def __init__(
        self,
        app,
        secret_key: str,
        algorithm: str = "HS256",
        redis_client: Optional[Any] = None,
        tenant_cache_ttl: int = 300,
        db_session_factory: Optional[Callable] = None,
    ):
        """
        Initialize the middleware.

        Args:
            app: The ASGI application
            secret_key: JWT signing secret
            algorithm: JWT algorithm (default: HS256)
            redis_client: Redis client for tenant caching
            tenant_cache_ttl: Tenant cache TTL in seconds
            db_session_factory: Async database session factory
        """
        super().__init__(app)
        self.secret_key = secret_key
        self.algorithm = algorithm
        self.redis_client = redis_client
        self.tenant_cache_ttl = tenant_cache_ttl
        self.db_session_factory = db_session_factory

        # In-memory tenant cache (fallback if no Redis)
        self._tenant_cache: Dict[str, Dict[str, Any]] = {}

    async def dispatch(
        self,
        request: Request,
        call_next: Callable,
    ) -> Response:
        """
        Process the request.

        Steps:
        1. Skip public/system paths
        2. Extract tenant ID from JWT/subdomain/header
        3. Load and validate tenant
        4. Inject context into request.state
        5. Track usage after response
        """
        request_id = self._generate_request_id()
        start_time = time.time()

        # Add request ID to state
        request.state.request_id = request_id

        # Skip public paths
        if self._is_public_path(request.url.path):
            return await call_next(request)

        # Skip system paths (handled by separate admin auth)
        if self._is_system_path(request.url.path):
            return await call_next(request)

        try:
            # Extract tenant ID and user info from request
            tenant_id, user_id, user_roles = await self._extract_tenant_info(request)

            if not tenant_id:
                return self._create_error_response(
                    status_code=401,
                    error_code="TENANT_NOT_IDENTIFIED",
                    message="Could not identify tenant from request",
                )

            # Load tenant
            tenant = await self._load_tenant(tenant_id)

            if not tenant:
                return self._create_error_response(
                    status_code=404,
                    error_code="TENANT_NOT_FOUND",
                    message=f"Tenant '{tenant_id}' not found",
                )

            # Validate tenant status
            validation_error = self._validate_tenant_status(tenant)
            if validation_error:
                return validation_error

            # Create and set tenant context
            context = TenantContext.from_tenant(
                tenant=tenant,
                user_id=user_id,
                user_roles=user_roles,
                request_id=request_id,
            )
            set_tenant_context(context)

            # Inject into request state
            request.state.tenant_context = context
            request.state.tenant_id = tenant.tenant_id
            request.state.tenant_uuid = str(tenant.id)
            request.state.user_id = user_id
            request.state.user_roles = user_roles

            # Check rate limit
            rate_limit_error = await self._check_rate_limit(context)
            if rate_limit_error:
                return rate_limit_error

            # Process request
            response = await call_next(request)

            # Track usage after successful response
            await self._track_usage(context, request, response)

            # Add tenant headers to response
            response.headers["X-Tenant-ID"] = tenant.tenant_id
            response.headers["X-Request-ID"] = request_id

            return response

        except jwt.ExpiredSignatureError:
            logger.warning(f"Expired JWT token for request {request_id}")
            return self._create_error_response(
                status_code=401,
                error_code="TOKEN_EXPIRED",
                message="Authentication token has expired",
            )

        except jwt.InvalidTokenError as e:
            logger.warning(f"Invalid JWT token: {e}")
            return self._create_error_response(
                status_code=401,
                error_code="INVALID_TOKEN",
                message="Invalid authentication token",
            )

        except Exception as e:
            logger.error(f"Tenant context middleware error: {e}", exc_info=True)
            return self._create_error_response(
                status_code=500,
                error_code="INTERNAL_ERROR",
                message="An internal error occurred",
            )

        finally:
            # Clear tenant context
            clear_tenant_context()

            # Log request completion
            elapsed_ms = (time.time() - start_time) * 1000
            logger.info(
                f"Request {request_id} completed in {elapsed_ms:.2f}ms",
                extra={
                    "request_id": request_id,
                    "path": request.url.path,
                    "method": request.method,
                    "elapsed_ms": elapsed_ms,
                },
            )

    async def _extract_tenant_info(
        self, request: Request
    ) -> tuple[Optional[str], Optional[str], List[str]]:
        """
        Extract tenant ID and user info from request.

        Extraction order:
        1. JWT token claims
        2. X-Tenant-ID header
        3. Subdomain extraction

        Returns:
            Tuple of (tenant_id, user_id, user_roles)
        """
        tenant_id = None
        user_id = None
        user_roles = []

        # 1. Try JWT token
        auth_header = request.headers.get("Authorization")
        if auth_header and auth_header.startswith("Bearer "):
            token = auth_header.split(" ")[1]
            try:
                payload = jwt.decode(
                    token,
                    self.secret_key,
                    algorithms=[self.algorithm],
                )
                tenant_id = payload.get("tenant_id")
                user_id = payload.get("sub")
                user_roles = payload.get("roles", [])

                logger.debug(
                    f"Extracted from JWT: tenant={tenant_id}, user={user_id}"
                )
            except jwt.PyJWTError:
                # Will be caught by outer exception handler
                raise

        # 2. Try X-Tenant-ID header (service-to-service)
        if not tenant_id:
            tenant_id = request.headers.get("X-Tenant-ID")
            if tenant_id:
                # Verify service API key for header-based tenant
                api_key = request.headers.get("X-API-Key")
                if not api_key:
                    logger.warning("X-Tenant-ID without X-API-Key")
                    tenant_id = None
                else:
                    logger.debug(f"Tenant from header: {tenant_id}")

        # 3. Try subdomain extraction
        if not tenant_id:
            tenant_id = self._extract_tenant_from_subdomain(request)

        return tenant_id, user_id, user_roles

    def _extract_tenant_from_subdomain(self, request: Request) -> Optional[str]:
        """
        Extract tenant slug from subdomain.

        Example: acme.greenlang.io -> acme

        Args:
            request: The incoming request

        Returns:
            Tenant slug or None
        """
        host = request.headers.get("Host", "")

        # Skip localhost and IP addresses
        if host.startswith("localhost") or host.startswith("127."):
            return None

        # Extract subdomain
        parts = host.split(".")
        if len(parts) >= 3:
            # e.g., acme.greenlang.io -> acme
            subdomain = parts[0]

            # Skip common non-tenant subdomains
            if subdomain in ("www", "api", "app", "admin", "staging", "dev"):
                return None

            return f"t-{subdomain}"

        return None

    async def _load_tenant(self, tenant_id: str) -> Optional[Tenant]:
        """
        Load tenant from cache or database.

        Args:
            tenant_id: Tenant identifier

        Returns:
            Tenant object or None
        """
        # Try cache first
        cached = await self._get_cached_tenant(tenant_id)
        if cached:
            return cached

        # Load from database
        tenant = await self._load_tenant_from_db(tenant_id)

        if tenant:
            # Cache for future requests
            await self._cache_tenant(tenant)

        return tenant

    async def _get_cached_tenant(self, tenant_id: str) -> Optional[Tenant]:
        """Get tenant from cache."""
        cache_key = f"tenant:{tenant_id}"

        # Try Redis cache
        if self.redis_client:
            try:
                cached_data = await self.redis_client.get(cache_key)
                if cached_data:
                    # Reconstruct tenant from cached data
                    # In production, use proper serialization
                    return self._deserialize_tenant(cached_data)
            except Exception as e:
                logger.warning(f"Redis cache error: {e}")

        # Try in-memory cache
        if tenant_id in self._tenant_cache:
            entry = self._tenant_cache[tenant_id]
            if time.time() < entry["expires_at"]:
                return entry["tenant"]
            else:
                del self._tenant_cache[tenant_id]

        return None

    async def _cache_tenant(self, tenant: Tenant) -> None:
        """Cache tenant for future requests."""
        cache_key = f"tenant:{tenant.tenant_id}"

        # Cache in Redis
        if self.redis_client:
            try:
                await self.redis_client.setex(
                    cache_key,
                    self.tenant_cache_ttl,
                    self._serialize_tenant(tenant),
                )
            except Exception as e:
                logger.warning(f"Redis cache write error: {e}")

        # Cache in memory (fallback)
        self._tenant_cache[tenant.tenant_id] = {
            "tenant": tenant,
            "expires_at": time.time() + self.tenant_cache_ttl,
        }

    async def _load_tenant_from_db(self, tenant_id: str) -> Optional[Tenant]:
        """
        Load tenant from database.

        Args:
            tenant_id: Tenant identifier

        Returns:
            Tenant object or None
        """
        if not self.db_session_factory:
            logger.warning("No database session factory configured")
            # Return mock tenant for development
            return self._create_mock_tenant(tenant_id)

        try:
            async with self.db_session_factory() as session:
                from sqlalchemy import select
                from models.tenant import Tenant

                result = await session.execute(
                    select(Tenant).where(Tenant.tenant_id == tenant_id)
                )
                return result.scalar_one_or_none()

        except Exception as e:
            logger.error(f"Database error loading tenant: {e}", exc_info=True)
            return None

    def _create_mock_tenant(self, tenant_id: str) -> Tenant:
        """Create a mock tenant for development/testing."""
        import uuid

        tenant = Tenant()
        tenant.id = uuid.uuid4()
        tenant.tenant_id = tenant_id
        tenant.name = f"Mock Tenant ({tenant_id})"
        tenant.slug = tenant_id.replace("t-", "")
        tenant.status = TenantStatus.ACTIVE
        tenant.subscription_tier = SubscriptionTier.PRO
        tenant.is_active = True
        tenant.settings = {}
        tenant.quotas = {}
        tenant.current_usage = {}
        tenant.feature_flags = {}
        tenant.created_at = datetime.utcnow()
        tenant.updated_at = datetime.utcnow()

        return tenant

    def _validate_tenant_status(self, tenant: Tenant) -> Optional[JSONResponse]:
        """
        Validate tenant status.

        Returns error response if tenant is not operational.
        """
        if tenant.status == TenantStatus.PENDING:
            return self._create_error_response(
                status_code=403,
                error_code="TENANT_PENDING",
                message="Tenant account is pending activation",
            )

        if tenant.status == TenantStatus.SUSPENDED:
            return self._create_error_response(
                status_code=403,
                error_code="TENANT_SUSPENDED",
                message="Tenant account is suspended. Please contact support.",
            )

        if tenant.status == TenantStatus.DEACTIVATED:
            return self._create_error_response(
                status_code=403,
                error_code="TENANT_DEACTIVATED",
                message="Tenant account has been deactivated",
            )

        if not tenant.is_active:
            return self._create_error_response(
                status_code=403,
                error_code="TENANT_INACTIVE",
                message="Tenant account is not active",
            )

        return None

    async def _check_rate_limit(
        self, context: TenantContext
    ) -> Optional[JSONResponse]:
        """
        Check if tenant has exceeded rate limit.

        Returns error response if rate limit exceeded.
        """
        # Get rate limit from quotas
        rate_limit = context.quotas.get("api_calls_per_minute", 100)

        if rate_limit == -1:  # Unlimited
            return None

        if not self.redis_client:
            return None

        try:
            # Use sliding window rate limiting
            cache_key = f"rate_limit:{context.tenant_id}:{int(time.time() // 60)}"

            current_count = await self.redis_client.incr(cache_key)

            # Set expiry on first request in window
            if current_count == 1:
                await self.redis_client.expire(cache_key, 60)

            if current_count > rate_limit:
                return self._create_error_response(
                    status_code=429,
                    error_code="RATE_LIMIT_EXCEEDED",
                    message=f"Rate limit exceeded. Limit: {rate_limit}/minute",
                    headers={
                        "Retry-After": "60",
                        "X-RateLimit-Limit": str(rate_limit),
                        "X-RateLimit-Remaining": "0",
                    },
                )

            return None

        except Exception as e:
            logger.warning(f"Rate limit check error: {e}")
            return None

    async def _track_usage(
        self,
        context: TenantContext,
        request: Request,
        response: Response,
    ) -> None:
        """
        Track usage metrics for the request.

        Increments counters for billing and analytics.
        """
        if not self.redis_client:
            return

        try:
            # Track API calls
            today = datetime.utcnow().strftime("%Y-%m-%d")
            month = datetime.utcnow().strftime("%Y-%m")

            # Daily API calls
            await self.redis_client.hincrby(
                f"usage:{context.tenant_id}:daily:{today}",
                "api_calls",
                1,
            )

            # Monthly API calls (for quota tracking)
            await self.redis_client.hincrby(
                f"usage:{context.tenant_id}:monthly:{month}",
                "api_calls",
                1,
            )

            # Track by endpoint
            endpoint = f"{request.method}:{request.url.path}"
            await self.redis_client.hincrby(
                f"usage:{context.tenant_id}:endpoints:{today}",
                endpoint,
                1,
            )

        except Exception as e:
            logger.warning(f"Usage tracking error: {e}")

    def _is_public_path(self, path: str) -> bool:
        """Check if path is public (no auth required)."""
        return path in self.PUBLIC_PATHS

    def _is_system_path(self, path: str) -> bool:
        """Check if path is a system path."""
        return any(path.startswith(p) for p in self.SYSTEM_PATHS)

    def _generate_request_id(self) -> str:
        """Generate unique request ID."""
        import uuid

        return str(uuid.uuid4())

    def _create_error_response(
        self,
        status_code: int,
        error_code: str,
        message: str,
        headers: Optional[Dict[str, str]] = None,
    ) -> JSONResponse:
        """Create standardized error response."""
        response = JSONResponse(
            status_code=status_code,
            content={
                "error": {
                    "code": error_code,
                    "message": message,
                }
            },
        )

        if headers:
            for key, value in headers.items():
                response.headers[key] = value

        return response

    def _serialize_tenant(self, tenant: Tenant) -> str:
        """Serialize tenant for caching."""
        import json

        return json.dumps(tenant.to_dict())

    def _deserialize_tenant(self, data: str) -> Optional[Tenant]:
        """Deserialize tenant from cache."""
        import json
        from uuid import UUID

        try:
            tenant_dict = json.loads(data)

            tenant = Tenant()
            tenant.id = UUID(tenant_dict["id"])
            tenant.tenant_id = tenant_dict["tenant_id"]
            tenant.name = tenant_dict["name"]
            tenant.slug = tenant_dict["slug"]
            tenant.status = TenantStatus(tenant_dict["status"])
            tenant.subscription_tier = SubscriptionTier(
                tenant_dict["subscription_tier"]
            )
            tenant.is_active = tenant_dict["is_active"]
            tenant.settings = tenant_dict.get("settings", {})
            tenant.quotas = tenant_dict.get("quotas", {})
            tenant.current_usage = tenant_dict.get("current_usage", {})
            tenant.feature_flags = tenant_dict.get("feature_flags", {})
            tenant.created_at = datetime.fromisoformat(tenant_dict["created_at"])
            tenant.updated_at = datetime.fromisoformat(tenant_dict["updated_at"])

            return tenant

        except Exception as e:
            logger.warning(f"Tenant deserialization error: {e}")
            return None


def require_feature(feature_name: str) -> Callable:
    """
    Decorator to require a specific feature flag.

    Usage:
        @router.get("/advanced-report")
        @require_feature("advanced_analytics")
        async def get_advanced_report(request: Request):
            ...

    Args:
        feature_name: Name of the required feature

    Returns:
        Decorator function
    """
    from functools import wraps

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            context = get_tenant_context()

            if not context:
                return JSONResponse(
                    status_code=401,
                    content={
                        "error": {
                            "code": "NO_TENANT_CONTEXT",
                            "message": "Tenant context not available",
                        }
                    },
                )

            if not context.is_feature_enabled(feature_name):
                return JSONResponse(
                    status_code=403,
                    content={
                        "error": {
                            "code": "FEATURE_NOT_AVAILABLE",
                            "message": f"Feature '{feature_name}' is not available for your subscription tier",
                            "upgrade_url": "/settings/billing",
                        }
                    },
                )

            return await func(*args, **kwargs)

        return wrapper

    return decorator


def require_quota(quota_name: str, increment: int = 1) -> Callable:
    """
    Decorator to check quota before executing endpoint.

    Usage:
        @router.post("/executions")
        @require_quota("executions_per_month")
        async def create_execution(request: Request):
            ...

    Args:
        quota_name: Name of the quota to check
        increment: Amount to check against quota

    Returns:
        Decorator function
    """
    from functools import wraps

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            context = get_tenant_context()

            if not context:
                return JSONResponse(
                    status_code=401,
                    content={
                        "error": {
                            "code": "NO_TENANT_CONTEXT",
                            "message": "Tenant context not available",
                        }
                    },
                )

            if not context.check_quota(quota_name, increment):
                limit = context.quotas.get(quota_name, 0)
                current = context.current_usage.get(quota_name, 0)

                return JSONResponse(
                    status_code=429,
                    content={
                        "error": {
                            "code": "QUOTA_EXCEEDED",
                            "message": f"Quota '{quota_name}' exceeded. Limit: {limit}, Current: {current}",
                            "quota_name": quota_name,
                            "limit": limit,
                            "current": current,
                            "upgrade_url": "/settings/billing",
                        }
                    },
                )

            return await func(*args, **kwargs)

        return wrapper

    return decorator


def require_role(role: str) -> Callable:
    """
    Decorator to require a specific role.

    Usage:
        @router.delete("/agents/{agent_id}")
        @require_role("admin")
        async def delete_agent(request: Request, agent_id: str):
            ...

    Args:
        role: Required role name

    Returns:
        Decorator function
    """
    from functools import wraps

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            context = get_tenant_context()

            if not context:
                return JSONResponse(
                    status_code=401,
                    content={
                        "error": {
                            "code": "NO_TENANT_CONTEXT",
                            "message": "Tenant context not available",
                        }
                    },
                )

            if not context.has_role(role):
                return JSONResponse(
                    status_code=403,
                    content={
                        "error": {
                            "code": "INSUFFICIENT_PERMISSIONS",
                            "message": f"Role '{role}' required for this operation",
                        }
                    },
                )

            return await func(*args, **kwargs)

        return wrapper

    return decorator
