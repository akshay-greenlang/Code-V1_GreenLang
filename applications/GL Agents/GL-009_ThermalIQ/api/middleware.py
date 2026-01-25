"""
GL-009 ThermalIQ - API Middleware

Custom middleware for authentication, rate limiting,
audit logging, provenance tracking, and error handling.

Middleware:
- AuthenticationMiddleware: JWT/API key authentication
- RateLimitMiddleware: Token bucket rate limiting
- AuditMiddleware: Request/response audit logging
- ProvenanceMiddleware: Computation provenance headers
- ErrorHandlingMiddleware: Structured error responses
"""

from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional, Set
import hashlib
import json
import logging
import time
import uuid

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response, JSONResponse

logger = logging.getLogger(__name__)


# =============================================================================
# Authentication Middleware
# =============================================================================

class AuthenticationMiddleware(BaseHTTPMiddleware):
    """
    Authentication middleware.

    Supports:
    - API key authentication (X-API-Key header)
    - JWT token authentication (Authorization: Bearer)
    - Service account authentication
    - Tenant isolation for multi-tenant deployments
    """

    def __init__(
        self,
        app,
        api_keys: Optional[Dict[str, str]] = None,
        require_auth: bool = False,
        exempt_paths: Optional[List[str]] = None,
        jwt_secret: Optional[str] = None,
        jwt_algorithm: str = "HS256",
    ) -> None:
        """
        Initialize authentication middleware.

        Args:
            app: ASGI application
            api_keys: Valid API keys (key -> user_id mapping)
            require_auth: Require authentication for all requests
            exempt_paths: Paths that don't require authentication
            jwt_secret: Secret key for JWT validation
            jwt_algorithm: JWT signing algorithm
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
            "/api/openapi.json",
            "/metrics",
            "/api/v1/metrics",
        ]
        self.jwt_secret = jwt_secret
        self.jwt_algorithm = jwt_algorithm

    async def dispatch(
        self,
        request: Request,
        call_next: Callable,
    ) -> Response:
        """Process request with authentication."""
        # Skip auth for exempt paths
        if self._is_exempt_path(request.url.path):
            return await call_next(request)

        # Extract credentials
        api_key = request.headers.get("X-API-Key")
        auth_header = request.headers.get("Authorization")

        user_id = None
        tenant_id = None
        roles: List[str] = []

        if api_key:
            user_info = self._validate_api_key(api_key)
            if user_info:
                user_id = user_info.get("user_id")
                tenant_id = user_info.get("tenant_id")
                roles = user_info.get("roles", [])
        elif auth_header and auth_header.startswith("Bearer "):
            token = auth_header[7:]
            user_info = self._validate_jwt_token(token)
            if user_info:
                user_id = user_info.get("sub")
                tenant_id = user_info.get("tenant_id")
                roles = user_info.get("roles", [])

        # Check if auth is required
        if self.require_auth and not user_id:
            return JSONResponse(
                status_code=401,
                content={
                    "error": "unauthorized",
                    "message": "Valid API key or JWT token required",
                    "details": {
                        "supported_auth": ["X-API-Key header", "Bearer token"]
                    }
                },
                headers={"WWW-Authenticate": "Bearer"},
            )

        # Add user info to request state
        request.state.user_id = user_id
        request.state.tenant_id = tenant_id
        request.state.roles = roles
        request.state.authenticated = user_id is not None

        response = await call_next(request)

        # Add auth info headers (for debugging)
        if user_id:
            response.headers["X-Authenticated-User"] = user_id[:8] + "..."

        return response

    def _is_exempt_path(self, path: str) -> bool:
        """Check if path is exempt from authentication."""
        return any(path.startswith(exempt) for exempt in self.exempt_paths)

    def _validate_api_key(self, api_key: str) -> Optional[Dict[str, Any]]:
        """Validate API key and return user info."""
        user_id = self.api_keys.get(api_key)
        if user_id:
            return {
                "user_id": user_id,
                "tenant_id": "default",
                "roles": ["user"]
            }
        return None

    def _validate_jwt_token(self, token: str) -> Optional[Dict[str, Any]]:
        """Validate JWT token and return user info."""
        if not self.jwt_secret:
            return None

        try:
            # In production, use proper JWT validation
            # import jwt
            # payload = jwt.decode(token, self.jwt_secret, algorithms=[self.jwt_algorithm])
            # return payload
            return None
        except Exception as e:
            logger.warning(f"JWT validation failed: {e}")
            return None


# =============================================================================
# Rate Limit Middleware
# =============================================================================

class RateLimitMiddleware(BaseHTTPMiddleware):
    """
    Rate limiting middleware using token bucket algorithm.

    Features:
    - Per-client rate limiting
    - Configurable burst size
    - Separate limits for different endpoint categories
    - Graceful degradation with retry headers
    """

    def __init__(
        self,
        app,
        requests_per_minute: int = 60,
        burst_size: int = 10,
        exempt_paths: Optional[List[str]] = None,
        endpoint_limits: Optional[Dict[str, int]] = None,
    ) -> None:
        """
        Initialize rate limiter.

        Args:
            app: ASGI application
            requests_per_minute: Sustained request rate per client
            burst_size: Maximum burst above rate limit
            exempt_paths: Paths exempt from rate limiting
            endpoint_limits: Custom limits for specific endpoints
        """
        super().__init__(app)
        self.requests_per_minute = requests_per_minute
        self.burst_size = burst_size
        self.exempt_paths = exempt_paths or ["/health", "/api/v1/health"]
        self.endpoint_limits = endpoint_limits or {}

        # Token buckets per client
        self.tokens: Dict[str, float] = {}
        self.last_update: Dict[str, float] = {}

        # Cleanup old entries periodically
        self._max_clients = 10000

    async def dispatch(
        self,
        request: Request,
        call_next: Callable,
    ) -> Response:
        """Process request with rate limiting."""
        # Get client identifier
        client_ip = request.client.host if request.client else "unknown"
        user_id = getattr(request.state, "user_id", None)
        client_id = user_id or client_ip

        # Skip rate limiting for exempt paths
        if self._is_exempt_path(request.url.path):
            return await call_next(request)

        # Get limit for this endpoint
        limit = self._get_endpoint_limit(request.url.path)

        # Check rate limit
        if not self._check_rate_limit(client_id, limit):
            retry_after = self._get_retry_after(client_id)
            return JSONResponse(
                status_code=429,
                content={
                    "error": "rate_limit_exceeded",
                    "message": f"Maximum {limit} requests per minute",
                    "retry_after_seconds": retry_after,
                    "limit": limit,
                },
                headers={
                    "Retry-After": str(int(retry_after)),
                    "X-RateLimit-Limit": str(limit),
                    "X-RateLimit-Remaining": "0",
                    "X-RateLimit-Reset": str(int(time.time() + retry_after)),
                },
            )

        response = await call_next(request)

        # Add rate limit headers
        remaining = self._get_remaining_tokens(client_id)
        response.headers["X-RateLimit-Limit"] = str(limit)
        response.headers["X-RateLimit-Remaining"] = str(int(remaining))
        response.headers["X-RateLimit-Reset"] = str(int(time.time() + 60))

        return response

    def _is_exempt_path(self, path: str) -> bool:
        """Check if path is exempt from rate limiting."""
        return any(path.startswith(exempt) for exempt in self.exempt_paths)

    def _get_endpoint_limit(self, path: str) -> int:
        """Get rate limit for endpoint."""
        for pattern, limit in self.endpoint_limits.items():
            if pattern in path:
                return limit
        return self.requests_per_minute

    def _check_rate_limit(self, client_id: str, limit: int) -> bool:
        """Check if request is within rate limit."""
        now = time.time()

        # Initialize bucket if needed
        if client_id not in self.tokens:
            self._cleanup_if_needed()
            self.tokens[client_id] = float(self.burst_size)
            self.last_update[client_id] = now

        # Add tokens based on time elapsed
        elapsed = now - self.last_update[client_id]
        token_rate = limit / 60.0
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
        return max(0, self.tokens.get(client_id, 0))

    def _get_retry_after(self, client_id: str) -> float:
        """Calculate time until next token available."""
        if client_id not in self.tokens:
            return 0
        tokens_needed = 1.0 - self.tokens[client_id]
        token_rate = self.requests_per_minute / 60.0
        return max(1, tokens_needed / token_rate) if token_rate > 0 else 60

    def _cleanup_if_needed(self) -> None:
        """Remove old entries if at capacity."""
        if len(self.tokens) >= self._max_clients:
            # Remove oldest entries
            sorted_clients = sorted(
                self.last_update.items(),
                key=lambda x: x[1]
            )
            for client_id, _ in sorted_clients[:self._max_clients // 2]:
                del self.tokens[client_id]
                del self.last_update[client_id]


# =============================================================================
# Audit Middleware
# =============================================================================

class AuditMiddleware(BaseHTTPMiddleware):
    """
    Audit logging middleware.

    Logs all API requests for compliance and debugging:
    - Request method, path, client info
    - Response status and timing
    - User/tenant identification
    - Request/response bodies (configurable)
    """

    def __init__(
        self,
        app,
        log_request_body: bool = False,
        log_response_body: bool = False,
        exclude_paths: Optional[List[str]] = None,
        sensitive_headers: Optional[Set[str]] = None,
        audit_logger: Optional[logging.Logger] = None,
    ) -> None:
        """
        Initialize audit middleware.

        Args:
            app: ASGI application
            log_request_body: Log request body content
            log_response_body: Log response body content
            exclude_paths: Paths to exclude from audit logging
            sensitive_headers: Headers to redact in logs
            audit_logger: Custom logger for audit records
        """
        super().__init__(app)
        self.log_request_body = log_request_body
        self.log_response_body = log_response_body
        self.exclude_paths = exclude_paths or ["/health", "/metrics"]
        self.sensitive_headers = sensitive_headers or {
            "authorization",
            "x-api-key",
            "cookie",
            "set-cookie"
        }
        self.audit_logger = audit_logger or logging.getLogger("thermaliq.audit")

    async def dispatch(
        self,
        request: Request,
        call_next: Callable,
    ) -> Response:
        """Process request with audit logging."""
        # Skip logging for excluded paths
        if self._is_excluded_path(request.url.path):
            return await call_next(request)

        # Generate request ID
        request_id = str(uuid.uuid4())[:12]
        request.state.request_id = request_id

        # Extract request info
        client_ip = request.client.host if request.client else "unknown"
        user_id = getattr(request.state, "user_id", None)
        tenant_id = getattr(request.state, "tenant_id", None)

        # Build audit record
        audit_record = {
            "request_id": request_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "method": request.method,
            "path": request.url.path,
            "query": str(request.url.query) if request.url.query else None,
            "client_ip": client_ip,
            "user_id": user_id,
            "tenant_id": tenant_id,
            "user_agent": request.headers.get("user-agent", "unknown")[:100],
            "headers": self._sanitize_headers(dict(request.headers)),
        }

        # Log request body if enabled
        if self.log_request_body and request.method in ["POST", "PUT", "PATCH"]:
            try:
                body = await request.body()
                audit_record["request_body_size"] = len(body)
                if len(body) < 10000:  # Only log small bodies
                    audit_record["request_body"] = body.decode("utf-8", errors="replace")[:1000]
            except Exception:
                pass

        start_time = time.time()

        # Log incoming request
        self.audit_logger.info(
            f"[{request_id}] REQUEST {request.method} {request.url.path} "
            f"from {client_ip} user={user_id}"
        )

        # Process request
        try:
            response = await call_next(request)
        except Exception as e:
            processing_time = (time.time() - start_time) * 1000
            audit_record["status"] = 500
            audit_record["error"] = str(e)
            audit_record["processing_time_ms"] = processing_time

            self.audit_logger.error(
                f"[{request_id}] ERROR 500 in {processing_time:.1f}ms: {e}"
            )
            raise

        # Complete audit record
        processing_time = (time.time() - start_time) * 1000
        audit_record["status"] = response.status_code
        audit_record["processing_time_ms"] = processing_time

        # Log response
        log_level = logging.INFO if response.status_code < 400 else logging.WARNING
        self.audit_logger.log(
            log_level,
            f"[{request_id}] RESPONSE {response.status_code} in {processing_time:.1f}ms"
        )

        # Add request ID to response headers
        response.headers["X-Request-ID"] = request_id
        response.headers["X-Process-Time-Ms"] = f"{processing_time:.1f}"

        return response

    def _is_excluded_path(self, path: str) -> bool:
        """Check if path is excluded from audit logging."""
        return any(path.startswith(excluded) for excluded in self.exclude_paths)

    def _sanitize_headers(self, headers: Dict[str, str]) -> Dict[str, str]:
        """Redact sensitive headers."""
        sanitized = {}
        for key, value in headers.items():
            if key.lower() in self.sensitive_headers:
                sanitized[key] = "[REDACTED]"
            else:
                sanitized[key] = value[:100] if len(value) > 100 else value
        return sanitized


# =============================================================================
# Provenance Middleware
# =============================================================================

class ProvenanceMiddleware(BaseHTTPMiddleware):
    """
    Computation provenance tracking middleware.

    Adds provenance headers to responses for:
    - Audit trail compliance
    - Result reproducibility
    - Version tracking
    - Computation fingerprinting
    """

    def __init__(
        self,
        app,
        agent_name: str = "GL-009-ThermalIQ",
        version: str = "1.0.0",
        environment: str = "production",
    ) -> None:
        """
        Initialize provenance middleware.

        Args:
            app: ASGI application
            agent_name: GreenLang agent identifier
            version: API version
            environment: Deployment environment
        """
        super().__init__(app)
        self.agent_name = agent_name
        self.version = version
        self.environment = environment

    async def dispatch(
        self,
        request: Request,
        call_next: Callable,
    ) -> Response:
        """Process request with provenance tracking."""
        response = await call_next(request)

        # Add provenance headers
        response.headers["X-GL-Agent"] = self.agent_name
        response.headers["X-GL-Version"] = self.version
        response.headers["X-GL-Environment"] = self.environment
        response.headers["X-GL-Timestamp"] = datetime.now(timezone.utc).isoformat()

        # Add computation fingerprint for POST requests
        if request.method == "POST":
            request_id = getattr(request.state, "request_id", str(uuid.uuid4())[:8])
            fingerprint = hashlib.sha256(
                f"{request_id}:{self.version}:{time.time()}".encode()
            ).hexdigest()[:12]
            response.headers["X-GL-Computation-ID"] = fingerprint

        return response


# =============================================================================
# Error Handling Middleware
# =============================================================================

class ErrorHandlingMiddleware(BaseHTTPMiddleware):
    """
    Centralized error handling middleware.

    Provides:
    - Consistent error response format
    - Exception logging
    - Stack trace redaction in production
    - Custom error handlers
    """

    def __init__(
        self,
        app,
        debug: bool = False,
        include_stack_trace: bool = False,
        custom_handlers: Optional[Dict[type, Callable]] = None,
    ) -> None:
        """
        Initialize error handling middleware.

        Args:
            app: ASGI application
            debug: Enable debug mode with detailed errors
            include_stack_trace: Include stack traces in responses
            custom_handlers: Custom handlers for specific exceptions
        """
        super().__init__(app)
        self.debug = debug
        self.include_stack_trace = include_stack_trace
        self.custom_handlers = custom_handlers or {}

    async def dispatch(
        self,
        request: Request,
        call_next: Callable,
    ) -> Response:
        """Process request with error handling."""
        request_id = getattr(request.state, "request_id", str(uuid.uuid4())[:8])

        try:
            response = await call_next(request)
            return response

        except Exception as e:
            # Check for custom handler
            for exc_type, handler in self.custom_handlers.items():
                if isinstance(e, exc_type):
                    return await handler(request, e)

            # Log the error
            logger.error(
                f"[{request_id}] Unhandled error: {type(e).__name__}: {e}",
                exc_info=True
            )

            # Build error response
            error_response = {
                "error": self._get_error_code(e),
                "message": str(e) if self.debug else "An internal error occurred",
                "request_id": request_id,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }

            if self.include_stack_trace and self.debug:
                import traceback
                error_response["stack_trace"] = traceback.format_exc()

            if self.debug:
                error_response["exception_type"] = type(e).__name__

            status_code = self._get_status_code(e)

            return JSONResponse(
                status_code=status_code,
                content=error_response,
                headers={
                    "X-Request-ID": request_id,
                    "X-Error-Type": type(e).__name__,
                }
            )

    def _get_error_code(self, exc: Exception) -> str:
        """Get error code from exception type."""
        error_codes = {
            "ValueError": "validation_error",
            "KeyError": "not_found",
            "PermissionError": "forbidden",
            "TimeoutError": "timeout",
            "ConnectionError": "service_unavailable",
        }
        return error_codes.get(type(exc).__name__, "internal_error")

    def _get_status_code(self, exc: Exception) -> int:
        """Get HTTP status code from exception type."""
        status_codes = {
            "ValueError": 400,
            "KeyError": 404,
            "PermissionError": 403,
            "TimeoutError": 504,
            "ConnectionError": 503,
            "NotImplementedError": 501,
        }
        return status_codes.get(type(exc).__name__, 500)


# =============================================================================
# Request Timing Middleware
# =============================================================================

class TimingMiddleware(BaseHTTPMiddleware):
    """
    Request timing middleware.

    Tracks request processing time and adds timing headers.
    """

    def __init__(self, app) -> None:
        """Initialize timing middleware."""
        super().__init__(app)

    async def dispatch(
        self,
        request: Request,
        call_next: Callable,
    ) -> Response:
        """Process request with timing."""
        start_time = time.perf_counter()

        response = await call_next(request)

        process_time = (time.perf_counter() - start_time) * 1000  # ms

        response.headers["X-Process-Time"] = f"{process_time:.2f}ms"
        response.headers["Server-Timing"] = f"total;dur={process_time:.2f}"

        return response


# =============================================================================
# CORS Middleware Configuration
# =============================================================================

def get_cors_config() -> Dict[str, Any]:
    """Get CORS middleware configuration."""
    return {
        "allow_origins": [
            "https://*.greenlang.io",
            "http://localhost:3000",
            "http://localhost:8000",
        ],
        "allow_credentials": True,
        "allow_methods": ["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        "allow_headers": [
            "Authorization",
            "Content-Type",
            "X-API-Key",
            "X-Request-ID",
            "X-Correlation-ID",
        ],
        "expose_headers": [
            "X-Request-ID",
            "X-Process-Time",
            "X-RateLimit-Limit",
            "X-RateLimit-Remaining",
            "X-GL-Agent",
            "X-GL-Version",
        ],
        "max_age": 600,
    }
