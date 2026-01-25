"""
GL-007 FurnacePulse - API Middleware

Custom middleware for RBAC enforcement, audit logging,
request tracking, rate limiting, and error handling.

Middleware Components:
- RBACMiddleware: Role-based access control enforcement
- AuditLoggingMiddleware: Comprehensive audit logging for compliance
- RequestIDMiddleware: Request ID tracking for distributed tracing
- RateLimitMiddleware: Token bucket rate limiting
- ErrorHandlingMiddleware: Consistent error response formatting
- ProvenanceMiddleware: Computation provenance headers

Author: GreenLang API Team
Version: 1.0.0
"""

from datetime import datetime, timezone
from enum import Enum
from functools import wraps
from typing import Any, Callable, Dict, List, Optional, Set
import hashlib
import json
import logging
import time
import uuid

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response, JSONResponse
from pydantic import BaseModel

logger = logging.getLogger(__name__)


# =============================================================================
# Enums and Models
# =============================================================================

class UserRole(str, Enum):
    """User roles for RBAC."""
    OPERATOR = "operator"       # Can view data, acknowledge alerts
    ENGINEER = "engineer"       # Can view data, acknowledge alerts, view predictions
    SAFETY = "safety"           # Can view compliance, generate evidence packages
    ADMIN = "admin"             # Full access
    SERVICE = "service"         # Machine-to-machine service accounts
    READONLY = "readonly"       # Read-only access to non-sensitive data


class User(BaseModel):
    """Authenticated user model."""
    user_id: str
    email: str
    roles: List[UserRole]
    tenant_id: str
    permissions: Set[str] = set()


class AuditLogEntry(BaseModel):
    """Audit log entry for compliance tracking."""
    log_id: str
    timestamp: datetime
    request_id: str
    user_id: Optional[str]
    tenant_id: Optional[str]
    action: str
    resource_type: str
    resource_id: Optional[str]
    method: str
    path: str
    query_params: Dict[str, Any]
    status_code: int
    response_time_ms: float
    ip_address: str
    user_agent: str
    success: bool
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = {}


# =============================================================================
# Role Permissions Configuration
# =============================================================================

# Define which roles can access which endpoints
ROLE_PERMISSIONS: Dict[UserRole, Set[str]] = {
    UserRole.OPERATOR: {
        "GET:/api/v1/health",
        "GET:/api/v1/status",
        "GET:/api/v1/furnaces/*/kpis",
        "GET:/api/v1/furnaces/*/hotspots",
        "GET:/api/v1/furnaces/*/tmt",
        "GET:/api/v1/furnaces/*/rul",
        "GET:/api/v1/alerts",
        "POST:/api/v1/alerts/*/acknowledge",
        "GET:/api/v1/metrics",
    },
    UserRole.ENGINEER: {
        "GET:/api/v1/health",
        "GET:/api/v1/status",
        "GET:/api/v1/furnaces/*/kpis",
        "GET:/api/v1/furnaces/*/hotspots",
        "GET:/api/v1/furnaces/*/tmt",
        "GET:/api/v1/furnaces/*/rul",
        "GET:/api/v1/alerts",
        "POST:/api/v1/alerts/*/acknowledge",
        "GET:/api/v1/explain/*",
        "GET:/api/v1/metrics",
    },
    UserRole.SAFETY: {
        "GET:/api/v1/health",
        "GET:/api/v1/status",
        "GET:/api/v1/furnaces/*/kpis",
        "GET:/api/v1/furnaces/*/hotspots",
        "GET:/api/v1/furnaces/*/tmt",
        "GET:/api/v1/furnaces/*/rul",
        "GET:/api/v1/furnaces/*/compliance",
        "POST:/api/v1/furnaces/*/evidence",
        "GET:/api/v1/alerts",
        "POST:/api/v1/alerts/*/acknowledge",
        "GET:/api/v1/explain/*",
        "GET:/api/v1/metrics",
    },
    UserRole.ADMIN: {
        "*",  # Full access
    },
    UserRole.SERVICE: {
        "GET:/api/v1/health",
        "GET:/api/v1/status",
        "GET:/api/v1/furnaces/*/kpis",
        "GET:/api/v1/furnaces/*/hotspots",
        "GET:/api/v1/furnaces/*/tmt",
        "GET:/api/v1/furnaces/*/rul",
        "GET:/api/v1/furnaces/*/compliance",
        "GET:/api/v1/alerts",
        "GET:/api/v1/explain/*",
        "GET:/api/v1/metrics",
    },
    UserRole.READONLY: {
        "GET:/api/v1/health",
        "GET:/api/v1/status",
        "GET:/api/v1/furnaces/*/kpis",
        "GET:/api/v1/furnaces/*/tmt",
        "GET:/api/v1/alerts",
        "GET:/api/v1/metrics",
    },
}

# Endpoints that require specific roles (more restrictive than general permissions)
PROTECTED_ENDPOINTS: Dict[str, Set[UserRole]] = {
    "GET:/api/v1/furnaces/*/compliance": {UserRole.SAFETY, UserRole.ADMIN},
    "POST:/api/v1/furnaces/*/evidence": {UserRole.SAFETY, UserRole.ADMIN},
}


# =============================================================================
# In-Memory Stores (Replace with Redis/DB in production)
# =============================================================================

# Token bucket state for rate limiting
_rate_limit_tokens: Dict[str, float] = {}
_rate_limit_last_update: Dict[str, float] = {}

# Audit log buffer (would be async-written to database in production)
_audit_log_buffer: List[AuditLogEntry] = []


# =============================================================================
# Helper Functions
# =============================================================================

def _normalize_path(path: str) -> str:
    """
    Normalize path for permission matching.
    Replaces path parameters with wildcards.

    Example: /api/v1/furnaces/FRN-001/kpis -> /api/v1/furnaces/*/kpis
    """
    parts = path.split("/")
    normalized = []

    for part in parts:
        # Detect path parameters (alphanumeric IDs, UUIDs, etc.)
        if part and (
            part.startswith("FRN-") or
            part.startswith("ALT-") or
            part.startswith("EVP-") or
            part.startswith("rul-") or
            part.startswith("hs-") or
            len(part) == 36 and "-" in part  # UUID
        ):
            normalized.append("*")
        else:
            normalized.append(part)

    return "/".join(normalized)


def _check_permission(
    method: str,
    path: str,
    roles: List[UserRole],
) -> bool:
    """
    Check if any of the user's roles grant permission for the endpoint.

    Args:
        method: HTTP method (GET, POST, etc.)
        path: Request path
        roles: User's roles

    Returns:
        True if access is permitted.
    """
    normalized_path = _normalize_path(path)
    permission_key = f"{method}:{normalized_path}"

    for role in roles:
        role_perms = ROLE_PERMISSIONS.get(role, set())

        # Admin has full access
        if "*" in role_perms:
            return True

        # Check exact match
        if permission_key in role_perms:
            return True

        # Check wildcard patterns in permissions
        for perm in role_perms:
            if _match_permission(perm, permission_key):
                return True

    return False


def _match_permission(pattern: str, target: str) -> bool:
    """
    Match a permission pattern against a target.
    Supports * wildcards.

    Args:
        pattern: Permission pattern (e.g., "GET:/api/v1/furnaces/*/kpis")
        target: Target to match (e.g., "GET:/api/v1/furnaces/*/kpis")

    Returns:
        True if pattern matches target.
    """
    pattern_parts = pattern.split("/")
    target_parts = target.split("/")

    if len(pattern_parts) != len(target_parts):
        return False

    for p, t in zip(pattern_parts, target_parts):
        if p != "*" and p != t:
            return False

    return True


def _validate_jwt_token(token: str) -> Optional[User]:
    """
    Validate JWT token and return user.
    In production, this would verify the token signature and expiration.

    Args:
        token: JWT bearer token

    Returns:
        User object if valid, None otherwise.
    """
    # Mock implementation - replace with actual JWT validation
    try:
        # In production:
        # import jwt
        # payload = jwt.decode(token, SECRET_KEY, algorithms=["HS256"])
        # return User(**payload)

        # For development, return mock user based on token
        if token.startswith("admin_"):
            return User(
                user_id="admin-001",
                email="admin@example.com",
                roles=[UserRole.ADMIN],
                tenant_id="tenant-001",
            )
        elif token.startswith("safety_"):
            return User(
                user_id="safety-001",
                email="safety@example.com",
                roles=[UserRole.SAFETY],
                tenant_id="tenant-001",
            )
        elif token.startswith("engineer_"):
            return User(
                user_id="eng-001",
                email="engineer@example.com",
                roles=[UserRole.ENGINEER],
                tenant_id="tenant-001",
            )
        elif token.startswith("operator_"):
            return User(
                user_id="op-001",
                email="operator@example.com",
                roles=[UserRole.OPERATOR],
                tenant_id="tenant-001",
            )
        elif token.startswith("service_"):
            return User(
                user_id="svc-001",
                email="service@example.com",
                roles=[UserRole.SERVICE],
                tenant_id="tenant-001",
            )
        else:
            return None

    except Exception as e:
        logger.warning(f"JWT validation failed: {e}")
        return None


def _validate_api_key(api_key: str) -> Optional[User]:
    """
    Validate API key and return associated user.

    Args:
        api_key: API key from X-API-Key header

    Returns:
        User object if valid, None otherwise.
    """
    # Mock implementation - replace with database lookup
    api_key_map = {
        "test-api-key-admin": User(
            user_id="api-admin",
            email="api@example.com",
            roles=[UserRole.ADMIN],
            tenant_id="tenant-001",
        ),
        "test-api-key-service": User(
            user_id="api-service",
            email="service@example.com",
            roles=[UserRole.SERVICE],
            tenant_id="tenant-001",
        ),
    }

    return api_key_map.get(api_key)


# =============================================================================
# Dependency Injection Functions
# =============================================================================

async def get_current_user(request: Request) -> Optional[User]:
    """
    FastAPI dependency to get the current authenticated user.

    Args:
        request: FastAPI request

    Returns:
        User object if authenticated, None otherwise.
    """
    return getattr(request.state, "user", None)


def require_roles(required_roles: List[UserRole]):
    """
    FastAPI dependency factory for role-based access control.

    Args:
        required_roles: List of roles that can access the endpoint

    Returns:
        Dependency function that checks user roles.
    """
    async def check_roles(request: Request) -> User:
        user = getattr(request.state, "user", None)

        if not user:
            from fastapi import HTTPException, status
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Authentication required",
                headers={"WWW-Authenticate": "Bearer"},
            )

        # Check if user has any of the required roles
        if not any(role in user.roles for role in required_roles):
            # Admin always has access
            if UserRole.ADMIN not in user.roles:
                from fastapi import HTTPException, status
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail=f"Requires one of roles: {[r.value for r in required_roles]}",
                )

        return user

    return check_roles


# =============================================================================
# RBAC Middleware
# =============================================================================

class RBACMiddleware(BaseHTTPMiddleware):
    """
    Role-Based Access Control middleware.

    Authenticates requests via JWT token or API key and enforces
    role-based permissions for API endpoints.
    """

    def __init__(
        self,
        app,
        require_auth: bool = True,
        exempt_paths: Optional[List[str]] = None,
    ) -> None:
        """
        Initialize RBAC middleware.

        Args:
            app: ASGI application
            require_auth: Whether to require authentication (default True)
            exempt_paths: Paths exempt from authentication
        """
        super().__init__(app)
        self.require_auth = require_auth
        self.exempt_paths = exempt_paths or [
            "/health",
            "/api/v1/health",
            "/docs",
            "/redoc",
            "/openapi.json",
            "/api/v1/openapi.json",
            "/api/v1/metrics",  # Prometheus scraping
        ]

    async def dispatch(
        self,
        request: Request,
        call_next: Callable,
    ) -> Response:
        """Process request with RBAC enforcement."""
        # Skip auth for exempt paths
        if any(request.url.path.startswith(path) for path in self.exempt_paths):
            return await call_next(request)

        # Extract authentication credentials
        user = None
        auth_header = request.headers.get("Authorization")
        api_key = request.headers.get("X-API-Key")

        if auth_header and auth_header.startswith("Bearer "):
            token = auth_header[7:]
            user = _validate_jwt_token(token)
        elif api_key:
            user = _validate_api_key(api_key)

        # Enforce authentication if required
        if self.require_auth and not user:
            return JSONResponse(
                status_code=401,
                content={
                    "error": "unauthorized",
                    "message": "Valid API key or JWT token required",
                },
                headers={"WWW-Authenticate": "Bearer"},
            )

        # Store user in request state
        request.state.user = user
        request.state.authenticated = user is not None

        # Check role-based permissions
        if user and self.require_auth:
            method = request.method
            path = request.url.path

            if not _check_permission(method, path, user.roles):
                logger.warning(
                    f"Access denied for user {user.user_id} "
                    f"({[r.value for r in user.roles]}) to {method} {path}"
                )
                return JSONResponse(
                    status_code=403,
                    content={
                        "error": "forbidden",
                        "message": "Insufficient permissions for this operation",
                        "required_roles": self._get_required_roles(method, path),
                    },
                )

        response = await call_next(request)
        return response

    def _get_required_roles(self, method: str, path: str) -> List[str]:
        """Get roles required for an endpoint."""
        normalized_path = _normalize_path(path)
        permission_key = f"{method}:{normalized_path}"

        required = []
        for role, perms in ROLE_PERMISSIONS.items():
            if "*" in perms or permission_key in perms:
                required.append(role.value)
            else:
                for perm in perms:
                    if _match_permission(perm, permission_key):
                        required.append(role.value)
                        break

        return required


# =============================================================================
# Audit Logging Middleware
# =============================================================================

class AuditLoggingMiddleware(BaseHTTPMiddleware):
    """
    Comprehensive audit logging middleware for compliance.

    Logs all API requests with user context, timing, and outcome
    for regulatory audit trails.
    """

    def __init__(
        self,
        app,
        log_request_body: bool = False,
        log_response_body: bool = False,
        exclude_paths: Optional[List[str]] = None,
        sensitive_params: Optional[List[str]] = None,
    ) -> None:
        """
        Initialize audit logging middleware.

        Args:
            app: ASGI application
            log_request_body: Whether to log request bodies
            log_response_body: Whether to log response bodies
            exclude_paths: Paths to exclude from logging
            sensitive_params: Parameters to redact in logs
        """
        super().__init__(app)
        self.log_request_body = log_request_body
        self.log_response_body = log_response_body
        self.exclude_paths = exclude_paths or ["/health", "/metrics"]
        self.sensitive_params = sensitive_params or [
            "password", "token", "api_key", "secret", "credential"
        ]

    async def dispatch(
        self,
        request: Request,
        call_next: Callable,
    ) -> Response:
        """Process request with audit logging."""
        # Skip logging for excluded paths
        if any(request.url.path.startswith(path) for path in self.exclude_paths):
            return await call_next(request)

        # Get or create request ID
        request_id = getattr(request.state, "request_id", str(uuid.uuid4())[:8])

        # Extract user info
        user = getattr(request.state, "user", None)
        user_id = user.user_id if user else None
        tenant_id = user.tenant_id if user else None

        # Get client info
        client_ip = request.client.host if request.client else "unknown"
        user_agent = request.headers.get("User-Agent", "unknown")

        # Sanitize query params
        query_params = dict(request.query_params)
        for param in self.sensitive_params:
            if param in query_params:
                query_params[param] = "[REDACTED]"

        # Determine resource type and ID from path
        resource_type, resource_id = self._extract_resource_info(request.url.path)

        # Start timing
        start_time = time.time()

        # Process request
        error_message = None
        try:
            response = await call_next(request)
            success = response.status_code < 400
        except Exception as e:
            error_message = str(e)
            success = False
            raise
        finally:
            # Calculate response time
            response_time_ms = (time.time() - start_time) * 1000

            # Create audit log entry
            log_entry = AuditLogEntry(
                log_id=str(uuid.uuid4()),
                timestamp=datetime.now(timezone.utc),
                request_id=request_id,
                user_id=user_id,
                tenant_id=tenant_id,
                action=self._determine_action(request.method, resource_type),
                resource_type=resource_type,
                resource_id=resource_id,
                method=request.method,
                path=request.url.path,
                query_params=query_params,
                status_code=response.status_code if success else 500,
                response_time_ms=response_time_ms,
                ip_address=client_ip,
                user_agent=user_agent,
                success=success,
                error_message=error_message,
            )

            # Log to structured logger
            self._write_audit_log(log_entry)

        return response

    def _extract_resource_info(self, path: str) -> tuple:
        """Extract resource type and ID from path."""
        parts = path.strip("/").split("/")

        resource_type = "unknown"
        resource_id = None

        # Parse path to extract resource info
        if "furnaces" in parts:
            resource_type = "furnace"
            idx = parts.index("furnaces")
            if idx + 1 < len(parts):
                resource_id = parts[idx + 1]
        elif "alerts" in parts:
            resource_type = "alert"
            idx = parts.index("alerts")
            if idx + 1 < len(parts) and not parts[idx + 1].startswith("acknowledge"):
                resource_id = parts[idx + 1]
        elif "explain" in parts:
            resource_type = "prediction"
            idx = parts.index("explain")
            if idx + 1 < len(parts):
                resource_id = parts[idx + 1]

        return resource_type, resource_id

    def _determine_action(self, method: str, resource_type: str) -> str:
        """Determine action name from method and resource."""
        action_map = {
            ("GET", "furnace"): "view_furnace",
            ("GET", "alert"): "view_alerts",
            ("POST", "alert"): "acknowledge_alert",
            ("GET", "prediction"): "view_explanation",
            ("POST", "furnace"): "generate_evidence",
        }

        return action_map.get((method, resource_type), f"{method.lower()}_{resource_type}")

    def _write_audit_log(self, entry: AuditLogEntry) -> None:
        """Write audit log entry."""
        # In production, write to database or audit log service
        log_data = entry.model_dump()
        log_data["timestamp"] = log_data["timestamp"].isoformat()

        logger.info(
            f"AUDIT [{entry.request_id}] "
            f"user={entry.user_id} "
            f"action={entry.action} "
            f"resource={entry.resource_type}/{entry.resource_id} "
            f"status={entry.status_code} "
            f"time={entry.response_time_ms:.1f}ms"
        )

        # Buffer for batch writing
        _audit_log_buffer.append(entry)

        # Flush buffer if too large
        if len(_audit_log_buffer) >= 100:
            self._flush_audit_buffer()

    def _flush_audit_buffer(self) -> None:
        """Flush audit log buffer to persistent storage."""
        # In production, batch write to database
        _audit_log_buffer.clear()


# =============================================================================
# Request ID Middleware
# =============================================================================

class RequestIDMiddleware(BaseHTTPMiddleware):
    """
    Request ID tracking middleware.

    Generates unique request IDs for distributed tracing and
    adds them to request state and response headers.
    """

    def __init__(
        self,
        app,
        header_name: str = "X-Request-ID",
        generate_if_missing: bool = True,
    ) -> None:
        """
        Initialize request ID middleware.

        Args:
            app: ASGI application
            header_name: Header name for request ID
            generate_if_missing: Generate ID if not provided
        """
        super().__init__(app)
        self.header_name = header_name
        self.generate_if_missing = generate_if_missing

    async def dispatch(
        self,
        request: Request,
        call_next: Callable,
    ) -> Response:
        """Process request with request ID tracking."""
        # Get or generate request ID
        request_id = request.headers.get(self.header_name)

        if not request_id and self.generate_if_missing:
            request_id = str(uuid.uuid4())[:12]

        # Store in request state
        request.state.request_id = request_id

        # Process request
        response = await call_next(request)

        # Add to response headers
        if request_id:
            response.headers[self.header_name] = request_id
            response.headers["X-Correlation-ID"] = request_id

        return response


# =============================================================================
# Rate Limit Middleware
# =============================================================================

class RateLimitMiddleware(BaseHTTPMiddleware):
    """
    Token bucket rate limiting middleware.

    Limits requests per client IP with configurable rates
    and burst allowances.
    """

    def __init__(
        self,
        app,
        requests_per_minute: int = 60,
        burst_size: int = 10,
        key_func: Optional[Callable[[Request], str]] = None,
    ) -> None:
        """
        Initialize rate limit middleware.

        Args:
            app: ASGI application
            requests_per_minute: Sustained request rate
            burst_size: Maximum burst above rate limit
            key_func: Custom function to extract rate limit key
        """
        super().__init__(app)
        self.requests_per_minute = requests_per_minute
        self.burst_size = burst_size
        self.key_func = key_func or self._default_key_func

    def _default_key_func(self, request: Request) -> str:
        """Default key function using client IP."""
        client_ip = request.client.host if request.client else "unknown"

        # Include tenant ID if available
        user = getattr(request.state, "user", None)
        if user:
            return f"{user.tenant_id}:{client_ip}"

        return client_ip

    async def dispatch(
        self,
        request: Request,
        call_next: Callable,
    ) -> Response:
        """Process request with rate limiting."""
        # Skip rate limiting for health checks
        if request.url.path in ["/health", "/api/v1/health"]:
            return await call_next(request)

        # Get client identifier
        client_id = self.key_func(request)

        # Check rate limit
        if not self._check_rate_limit(client_id):
            return JSONResponse(
                status_code=429,
                content={
                    "error": "rate_limit_exceeded",
                    "message": f"Rate limit of {self.requests_per_minute} requests/minute exceeded",
                    "retry_after_seconds": 60,
                },
                headers={
                    "Retry-After": "60",
                    "X-RateLimit-Limit": str(self.requests_per_minute),
                    "X-RateLimit-Remaining": "0",
                    "X-RateLimit-Reset": str(int(time.time()) + 60),
                },
            )

        response = await call_next(request)

        # Add rate limit headers
        remaining = self._get_remaining_tokens(client_id)
        response.headers["X-RateLimit-Limit"] = str(self.requests_per_minute)
        response.headers["X-RateLimit-Remaining"] = str(int(remaining))
        response.headers["X-RateLimit-Reset"] = str(int(time.time()) + 60)

        return response

    def _check_rate_limit(self, client_id: str) -> bool:
        """Check if request is within rate limit using token bucket."""
        now = time.time()

        if client_id not in _rate_limit_tokens:
            _rate_limit_tokens[client_id] = float(self.burst_size)
            _rate_limit_last_update[client_id] = now

        # Replenish tokens based on elapsed time
        elapsed = now - _rate_limit_last_update[client_id]
        token_rate = self.requests_per_minute / 60.0
        _rate_limit_tokens[client_id] = min(
            self.burst_size,
            _rate_limit_tokens[client_id] + elapsed * token_rate
        )
        _rate_limit_last_update[client_id] = now

        # Check if token available
        if _rate_limit_tokens[client_id] >= 1.0:
            _rate_limit_tokens[client_id] -= 1.0
            return True

        return False

    def _get_remaining_tokens(self, client_id: str) -> float:
        """Get remaining tokens for client."""
        return _rate_limit_tokens.get(client_id, 0)


# =============================================================================
# Error Handling Middleware
# =============================================================================

class ErrorHandlingMiddleware(BaseHTTPMiddleware):
    """
    Consistent error response formatting middleware.

    Catches exceptions and formats them as JSON responses
    with appropriate status codes.
    """

    def __init__(
        self,
        app,
        include_traceback: bool = False,
    ) -> None:
        """
        Initialize error handling middleware.

        Args:
            app: ASGI application
            include_traceback: Include traceback in error response (dev only)
        """
        super().__init__(app)
        self.include_traceback = include_traceback

    async def dispatch(
        self,
        request: Request,
        call_next: Callable,
    ) -> Response:
        """Process request with error handling."""
        try:
            response = await call_next(request)
            return response

        except ValueError as e:
            logger.warning(f"Validation error: {e}")
            return JSONResponse(
                status_code=400,
                content={
                    "error": "validation_error",
                    "message": str(e),
                    "request_id": getattr(request.state, "request_id", None),
                },
            )

        except PermissionError as e:
            logger.warning(f"Permission denied: {e}")
            return JSONResponse(
                status_code=403,
                content={
                    "error": "forbidden",
                    "message": str(e),
                    "request_id": getattr(request.state, "request_id", None),
                },
            )

        except FileNotFoundError as e:
            return JSONResponse(
                status_code=404,
                content={
                    "error": "not_found",
                    "message": str(e),
                    "request_id": getattr(request.state, "request_id", None),
                },
            )

        except Exception as e:
            logger.error(f"Internal error: {e}", exc_info=True)

            content = {
                "error": "internal_error",
                "message": "An internal error occurred",
                "request_id": getattr(request.state, "request_id", None),
            }

            if self.include_traceback:
                import traceback
                content["traceback"] = traceback.format_exc()

            return JSONResponse(
                status_code=500,
                content=content,
            )


# =============================================================================
# Provenance Middleware
# =============================================================================

class ProvenanceMiddleware(BaseHTTPMiddleware):
    """
    Computation provenance tracking middleware.

    Adds headers for audit trail compliance and
    deterministic computation verification.
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
        response.headers["X-GL-Agent"] = "GL-007-FurnacePulse"
        response.headers["X-GL-Version"] = "1.0.0"
        response.headers["X-GL-Timestamp"] = datetime.now(timezone.utc).isoformat()

        # Add request-specific hash if available
        request_id = getattr(request.state, "request_id", None)
        if request_id:
            hash_input = f"{request_id}:{request.method}:{request.url.path}"
            provenance_hash = hashlib.sha256(hash_input.encode()).hexdigest()[:16]
            response.headers["X-GL-Provenance"] = provenance_hash

        return response
