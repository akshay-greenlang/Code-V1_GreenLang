"""
GL-014 EXCHANGERPRO - API Middleware

Custom middleware for authentication, rate limiting,
logging, and request/response processing.

Provides:
- RequestLoggingMiddleware: Correlation ID tracking, request/response logging
- RateLimitMiddleware: Token bucket rate limiting per client
- AuthenticationMiddleware: API key and JWT authentication
- ErrorHandlingMiddleware: Structured error responses
- ProvenanceMiddleware: Computation provenance headers
"""

from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional, Set
import hashlib
import json
import logging
import time
import uuid
import re

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response, JSONResponse
from starlette.types import ASGIApp

logger = logging.getLogger(__name__)

# Agent constants
AGENT_ID = "GL-014"
AGENT_NAME = "EXCHANGERPRO"
AGENT_VERSION = "1.0.0"


# =============================================================================
# Request Logging Middleware
# =============================================================================

class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """
    Request/response logging middleware with correlation IDs.

    Features:
    - Generates unique correlation ID for each request
    - Logs request method, path, client IP
    - Logs response status code and processing time
    - Optionally logs request/response bodies
    - Adds tracing headers to responses
    """

    def __init__(
        self,
        app: ASGIApp,
        log_request_body: bool = False,
        log_response_body: bool = False,
        exclude_paths: Optional[List[str]] = None,
        max_body_log_size: int = 1000,
    ) -> None:
        """
        Initialize logging middleware.

        Args:
            app: ASGI application
            log_request_body: Log request body content
            log_response_body: Log response body content
            exclude_paths: Paths to exclude from logging
            max_body_log_size: Max body size to log (bytes)
        """
        super().__init__(app)
        self.log_request_body = log_request_body
        self.log_response_body = log_response_body
        self.exclude_paths = exclude_paths or ["/health", "/metrics"]
        self.max_body_log_size = max_body_log_size

    async def dispatch(
        self,
        request: Request,
        call_next: Callable,
    ) -> Response:
        """Process request with logging and correlation ID."""
        # Skip logging for excluded paths
        if any(request.url.path.startswith(path) for path in self.exclude_paths):
            return await call_next(request)

        # Generate or extract correlation ID
        correlation_id = request.headers.get(
            "X-Correlation-ID",
            str(uuid.uuid4())
        )
        request_id = str(uuid.uuid4())[:8]

        # Store IDs in request state
        request.state.correlation_id = correlation_id
        request.state.request_id = request_id

        # Get client info
        client_ip = self._get_client_ip(request)
        user_agent = request.headers.get("User-Agent", "unknown")

        # Log request
        start_time = time.time()
        logger.info(
            f"[{request_id}] {request.method} {request.url.path} "
            f"| client={client_ip} | correlation_id={correlation_id}"
        )

        # Optionally log request body
        if self.log_request_body and request.method in ["POST", "PUT", "PATCH"]:
            await self._log_request_body(request, request_id)

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
        process_time_ms = (time.time() - start_time) * 1000

        # Log response
        logger.info(
            f"[{request_id}] {response.status_code} "
            f"| {process_time_ms:.1f}ms"
        )

        # Add tracing headers
        response.headers["X-Request-ID"] = request_id
        response.headers["X-Correlation-ID"] = correlation_id
        response.headers["X-Process-Time-Ms"] = f"{process_time_ms:.1f}"

        return response

    def _get_client_ip(self, request: Request) -> str:
        """Extract client IP from request."""
        # Check for forwarded headers
        forwarded = request.headers.get("X-Forwarded-For")
        if forwarded:
            return forwarded.split(",")[0].strip()

        real_ip = request.headers.get("X-Real-IP")
        if real_ip:
            return real_ip

        return request.client.host if request.client else "unknown"

    async def _log_request_body(self, request: Request, request_id: str) -> None:
        """Log request body with size limit."""
        try:
            body = await request.body()
            if len(body) <= self.max_body_log_size:
                logger.debug(f"[{request_id}] Request body: {body.decode()}")
            else:
                logger.debug(
                    f"[{request_id}] Request body (truncated): "
                    f"{body[:self.max_body_log_size].decode()}..."
                )
        except Exception as e:
            logger.debug(f"[{request_id}] Could not log request body: {e}")


# =============================================================================
# Rate Limiting Middleware
# =============================================================================

class RateLimitMiddleware(BaseHTTPMiddleware):
    """
    Rate limiting middleware using token bucket algorithm.

    Features:
    - Per-client rate limiting by IP or API key
    - Configurable rate and burst size
    - Rate limit headers in responses
    - Exempt paths (e.g., health checks)
    """

    def __init__(
        self,
        app: ASGIApp,
        requests_per_minute: int = 60,
        burst_size: int = 10,
        exempt_paths: Optional[List[str]] = None,
        rate_limit_by: str = "ip",  # "ip" or "api_key"
    ) -> None:
        """
        Initialize rate limiter.

        Args:
            app: ASGI application
            requests_per_minute: Sustained request rate
            burst_size: Maximum burst above rate limit
            exempt_paths: Paths exempt from rate limiting
            rate_limit_by: Rate limit key: "ip" or "api_key"
        """
        super().__init__(app)
        self.requests_per_minute = requests_per_minute
        self.burst_size = burst_size
        self.exempt_paths = exempt_paths or [
            "/health",
            "/metrics",
            "/docs",
            "/redoc",
            "/openapi.json",
        ]
        self.rate_limit_by = rate_limit_by

        # Token buckets per client
        self.tokens: Dict[str, float] = {}
        self.last_update: Dict[str, float] = {}

    async def dispatch(
        self,
        request: Request,
        call_next: Callable,
    ) -> Response:
        """Process request with rate limiting."""
        # Skip rate limiting for exempt paths
        if any(request.url.path.startswith(path) for path in self.exempt_paths):
            return await call_next(request)

        # Get client identifier
        client_id = self._get_client_id(request)

        # Check rate limit
        if not self._check_rate_limit(client_id):
            retry_after = self._get_retry_after(client_id)
            return JSONResponse(
                status_code=429,
                content={
                    "error": "rate_limit_exceeded",
                    "message": f"Rate limit of {self.requests_per_minute} requests/minute exceeded",
                    "retry_after_seconds": retry_after,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                },
                headers={
                    "Retry-After": str(int(retry_after)),
                    "X-RateLimit-Limit": str(self.requests_per_minute),
                    "X-RateLimit-Remaining": "0",
                    "X-RateLimit-Reset": str(int(time.time() + retry_after)),
                },
            )

        # Process request
        response = await call_next(request)

        # Add rate limit headers
        remaining = self._get_remaining_tokens(client_id)
        response.headers["X-RateLimit-Limit"] = str(self.requests_per_minute)
        response.headers["X-RateLimit-Remaining"] = str(int(remaining))
        response.headers["X-RateLimit-Reset"] = str(
            int(time.time() + 60)  # Reset in next minute
        )

        return response

    def _get_client_id(self, request: Request) -> str:
        """Get client identifier for rate limiting."""
        if self.rate_limit_by == "api_key":
            api_key = request.headers.get("X-API-Key")
            if api_key:
                return f"key:{hashlib.sha256(api_key.encode()).hexdigest()[:16]}"

        # Default to IP
        forwarded = request.headers.get("X-Forwarded-For")
        if forwarded:
            return f"ip:{forwarded.split(',')[0].strip()}"

        client_ip = request.client.host if request.client else "unknown"
        return f"ip:{client_ip}"

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
        return max(0, self.tokens.get(client_id, 0))

    def _get_retry_after(self, client_id: str) -> float:
        """Calculate seconds until next token available."""
        tokens_needed = 1.0 - self.tokens.get(client_id, 0)
        token_rate = self.requests_per_minute / 60.0
        return tokens_needed / token_rate if token_rate > 0 else 60.0


# =============================================================================
# Authentication Middleware
# =============================================================================

class AuthenticationMiddleware(BaseHTTPMiddleware):
    """
    Authentication middleware supporting API key and JWT.

    Features:
    - API key validation via X-API-Key header
    - JWT Bearer token validation
    - Role-based access control (RBAC)
    - Exempt paths for public endpoints
    """

    def __init__(
        self,
        app: ASGIApp,
        api_keys: Optional[Dict[str, Dict[str, Any]]] = None,
        jwt_secret: Optional[str] = None,
        jwt_algorithm: str = "HS256",
        require_auth: bool = False,
        exempt_paths: Optional[List[str]] = None,
    ) -> None:
        """
        Initialize authentication middleware.

        Args:
            app: ASGI application
            api_keys: Valid API keys mapping to user info
            jwt_secret: Secret for JWT validation
            jwt_algorithm: JWT algorithm
            require_auth: Require authentication for all requests
            exempt_paths: Paths that don't require authentication
        """
        super().__init__(app)
        self.api_keys = api_keys or {}
        self.jwt_secret = jwt_secret
        self.jwt_algorithm = jwt_algorithm
        self.require_auth = require_auth
        self.exempt_paths = exempt_paths or [
            "/health",
            "/metrics",
            "/docs",
            "/redoc",
            "/openapi.json",
            "/api/v1/health",
            "/api/v1/metrics",
        ]

    async def dispatch(
        self,
        request: Request,
        call_next: Callable,
    ) -> Response:
        """Process request with authentication."""
        # Skip auth for exempt paths
        if self._is_exempt_path(request.url.path):
            request.state.user = None
            request.state.authenticated = False
            return await call_next(request)

        # Try API key authentication
        api_key = request.headers.get("X-API-Key")
        if api_key:
            user_info = self._validate_api_key(api_key)
            if user_info:
                request.state.user = user_info
                request.state.authenticated = True
                request.state.auth_method = "api_key"
                return await call_next(request)

        # Try JWT authentication
        auth_header = request.headers.get("Authorization")
        if auth_header and auth_header.startswith("Bearer "):
            token = auth_header[7:]
            user_info = self._validate_jwt_token(token)
            if user_info:
                request.state.user = user_info
                request.state.authenticated = True
                request.state.auth_method = "jwt"
                return await call_next(request)

        # Check if authentication is required
        if self.require_auth:
            return JSONResponse(
                status_code=401,
                content={
                    "error": "unauthorized",
                    "message": "Valid API key or JWT token required",
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                },
                headers={"WWW-Authenticate": "Bearer"},
            )

        # Allow unauthenticated access
        request.state.user = None
        request.state.authenticated = False
        return await call_next(request)

    def _is_exempt_path(self, path: str) -> bool:
        """Check if path is exempt from authentication."""
        for exempt in self.exempt_paths:
            if path == exempt or path.startswith(exempt + "/"):
                return True
        return False

    def _validate_api_key(self, api_key: str) -> Optional[Dict[str, Any]]:
        """
        Validate API key and return user info.

        In production, this would query a database or cache.
        """
        # Hash the API key for lookup (keys stored hashed)
        key_hash = hashlib.sha256(api_key.encode()).hexdigest()

        user_info = self.api_keys.get(key_hash)
        if user_info:
            return {
                "user_id": user_info.get("user_id"),
                "tenant_id": user_info.get("tenant_id"),
                "roles": user_info.get("roles", []),
                "api_key_id": user_info.get("key_id"),
            }

        # Also check unhashed for development
        user_info = self.api_keys.get(api_key)
        if user_info:
            return {
                "user_id": user_info.get("user_id"),
                "tenant_id": user_info.get("tenant_id"),
                "roles": user_info.get("roles", []),
                "api_key_id": user_info.get("key_id"),
            }

        return None

    def _validate_jwt_token(self, token: str) -> Optional[Dict[str, Any]]:
        """
        Validate JWT token and return user info.

        In production, implement proper JWT validation.
        """
        if not self.jwt_secret:
            return None

        try:
            import jwt

            payload = jwt.decode(
                token,
                self.jwt_secret,
                algorithms=[self.jwt_algorithm]
            )

            return {
                "user_id": payload.get("sub"),
                "tenant_id": payload.get("tenant_id"),
                "roles": payload.get("roles", []),
                "email": payload.get("email"),
            }
        except ImportError:
            logger.warning("PyJWT not installed, JWT authentication disabled")
            return None
        except Exception as e:
            logger.debug(f"JWT validation failed: {e}")
            return None


# =============================================================================
# Error Handling Middleware
# =============================================================================

class ErrorHandlingMiddleware(BaseHTTPMiddleware):
    """
    Error handling middleware for structured error responses.

    Features:
    - Catches unhandled exceptions
    - Formats consistent error responses
    - Logs errors with request context
    - Hides internal details in production
    """

    def __init__(
        self,
        app: ASGIApp,
        debug: bool = False,
        include_stack_trace: bool = False,
    ) -> None:
        """
        Initialize error handling middleware.

        Args:
            app: ASGI application
            debug: Enable debug mode
            include_stack_trace: Include stack traces in responses
        """
        super().__init__(app)
        self.debug = debug
        self.include_stack_trace = include_stack_trace

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
            return self._create_error_response(
                status_code=400,
                error="validation_error",
                message=str(e),
                request=request,
            )

        except PermissionError as e:
            return self._create_error_response(
                status_code=403,
                error="forbidden",
                message=str(e) or "Access denied",
                request=request,
            )

        except FileNotFoundError as e:
            return self._create_error_response(
                status_code=404,
                error="not_found",
                message=str(e) or "Resource not found",
                request=request,
            )

        except TimeoutError as e:
            return self._create_error_response(
                status_code=504,
                error="timeout",
                message=str(e) or "Request timed out",
                request=request,
            )

        except Exception as e:
            # Log unexpected errors
            request_id = getattr(request.state, "request_id", "unknown")
            logger.error(
                f"[{request_id}] Unhandled exception: {e}",
                exc_info=True,
            )

            # Create error response
            if self.debug:
                import traceback
                details = traceback.format_exc() if self.include_stack_trace else None
            else:
                details = None

            return self._create_error_response(
                status_code=500,
                error="internal_error",
                message="An internal error occurred" if not self.debug else str(e),
                request=request,
                details=details,
            )

    def _create_error_response(
        self,
        status_code: int,
        error: str,
        message: str,
        request: Request,
        details: Optional[str] = None,
    ) -> JSONResponse:
        """Create structured error response."""
        request_id = getattr(request.state, "request_id", None)
        correlation_id = getattr(request.state, "correlation_id", None)

        content = {
            "error": error,
            "message": message,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "path": str(request.url.path),
        }

        if request_id:
            content["request_id"] = request_id
        if correlation_id:
            content["correlation_id"] = correlation_id
        if details:
            content["details"] = details

        return JSONResponse(
            status_code=status_code,
            content=content,
        )


# =============================================================================
# Provenance Middleware
# =============================================================================

class ProvenanceMiddleware(BaseHTTPMiddleware):
    """
    Provenance tracking middleware for audit compliance.

    Features:
    - Adds agent identification headers
    - Tracks computation timestamps
    - Enables response traceability
    """

    def __init__(
        self,
        app: ASGIApp,
        agent_id: str = AGENT_ID,
        agent_name: str = AGENT_NAME,
        agent_version: str = AGENT_VERSION,
    ) -> None:
        """
        Initialize provenance middleware.

        Args:
            app: ASGI application
            agent_id: Agent identifier
            agent_name: Agent name
            agent_version: Agent version
        """
        super().__init__(app)
        self.agent_id = agent_id
        self.agent_name = agent_name
        self.agent_version = agent_version

    async def dispatch(
        self,
        request: Request,
        call_next: Callable,
    ) -> Response:
        """Process request with provenance tracking."""
        response = await call_next(request)

        # Add provenance headers
        response.headers["X-GL-Agent-ID"] = self.agent_id
        response.headers["X-GL-Agent-Name"] = self.agent_name
        response.headers["X-GL-Agent-Version"] = self.agent_version
        response.headers["X-GL-Timestamp"] = datetime.now(timezone.utc).isoformat()

        return response


# =============================================================================
# Audit Logging Middleware
# =============================================================================

class AuditLoggingMiddleware(BaseHTTPMiddleware):
    """
    Audit logging middleware for compliance tracking.

    Features:
    - Logs all API operations with user context
    - Records request/response hashes
    - Stores audit events for compliance
    """

    def __init__(
        self,
        app: ASGIApp,
        audit_logger: Optional[logging.Logger] = None,
        log_to_file: bool = True,
        audit_file_path: Optional[str] = None,
        exclude_paths: Optional[List[str]] = None,
    ) -> None:
        """
        Initialize audit logging middleware.

        Args:
            app: ASGI application
            audit_logger: Logger for audit events
            log_to_file: Write audit log to file
            audit_file_path: Path to audit log file
            exclude_paths: Paths to exclude from audit
        """
        super().__init__(app)
        self.audit_logger = audit_logger or logging.getLogger("audit")
        self.log_to_file = log_to_file
        self.audit_file_path = audit_file_path
        self.exclude_paths = exclude_paths or ["/health", "/metrics"]

    async def dispatch(
        self,
        request: Request,
        call_next: Callable,
    ) -> Response:
        """Process request with audit logging."""
        # Skip audit for excluded paths
        if any(request.url.path.startswith(path) for path in self.exclude_paths):
            return await call_next(request)

        # Capture request info
        start_time = time.time()
        request_id = getattr(request.state, "request_id", str(uuid.uuid4())[:8])

        # Get request body hash if applicable
        request_hash = None
        if request.method in ["POST", "PUT", "PATCH"]:
            try:
                body = await request.body()
                request_hash = hashlib.sha256(body).hexdigest()[:16]
            except Exception:
                pass

        # Process request
        response = await call_next(request)

        # Calculate duration
        duration_ms = (time.time() - start_time) * 1000

        # Get user info
        user = getattr(request.state, "user", None)
        user_id = user.get("user_id") if user else None
        api_key_id = user.get("api_key_id") if user else None

        # Extract exchanger ID from path if present
        exchanger_id = self._extract_exchanger_id(request.url.path)

        # Create audit event
        audit_event = {
            "event_id": str(uuid.uuid4()),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event_type": "api_request",
            "request_id": request_id,
            "user_id": user_id,
            "api_key_id": api_key_id,
            "method": request.method,
            "path": request.url.path,
            "exchanger_id": exchanger_id,
            "status_code": response.status_code,
            "duration_ms": round(duration_ms, 2),
            "request_hash": request_hash,
            "client_ip": self._get_client_ip(request),
        }

        # Log audit event
        self.audit_logger.info(json.dumps(audit_event))

        return response

    def _extract_exchanger_id(self, path: str) -> Optional[str]:
        """Extract exchanger ID from path."""
        match = re.search(r"/exchangers/([^/]+)", path)
        return match.group(1) if match else None

    def _get_client_ip(self, request: Request) -> str:
        """Extract client IP from request."""
        forwarded = request.headers.get("X-Forwarded-For")
        if forwarded:
            return forwarded.split(",")[0].strip()
        return request.client.host if request.client else "unknown"


# =============================================================================
# Request Validation Middleware
# =============================================================================

class RequestValidationMiddleware(BaseHTTPMiddleware):
    """
    Request validation middleware.

    Features:
    - Validates Content-Type headers
    - Enforces request size limits
    - Validates required headers
    """

    def __init__(
        self,
        app: ASGIApp,
        max_request_size_bytes: int = 10 * 1024 * 1024,  # 10MB
        require_content_type: bool = True,
        allowed_content_types: Optional[List[str]] = None,
    ) -> None:
        """
        Initialize request validation middleware.

        Args:
            app: ASGI application
            max_request_size_bytes: Maximum request body size
            require_content_type: Require Content-Type header for POST/PUT
            allowed_content_types: Allowed Content-Type values
        """
        super().__init__(app)
        self.max_request_size = max_request_size_bytes
        self.require_content_type = require_content_type
        self.allowed_content_types = allowed_content_types or [
            "application/json",
            "application/x-www-form-urlencoded",
            "multipart/form-data",
        ]

    async def dispatch(
        self,
        request: Request,
        call_next: Callable,
    ) -> Response:
        """Process request with validation."""
        # Check Content-Length
        content_length = request.headers.get("Content-Length")
        if content_length:
            try:
                size = int(content_length)
                if size > self.max_request_size:
                    return JSONResponse(
                        status_code=413,
                        content={
                            "error": "request_too_large",
                            "message": f"Request body exceeds maximum size of {self.max_request_size} bytes",
                            "timestamp": datetime.now(timezone.utc).isoformat(),
                        },
                    )
            except ValueError:
                pass

        # Check Content-Type for POST/PUT/PATCH
        if self.require_content_type and request.method in ["POST", "PUT", "PATCH"]:
            content_type = request.headers.get("Content-Type", "")
            content_type_base = content_type.split(";")[0].strip()

            if content_type_base and content_type_base not in self.allowed_content_types:
                return JSONResponse(
                    status_code=415,
                    content={
                        "error": "unsupported_media_type",
                        "message": f"Content-Type must be one of: {self.allowed_content_types}",
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                    },
                )

        return await call_next(request)


# =============================================================================
# Export all middleware
# =============================================================================

__all__ = [
    "RequestLoggingMiddleware",
    "RateLimitMiddleware",
    "AuthenticationMiddleware",
    "ErrorHandlingMiddleware",
    "ProvenanceMiddleware",
    "AuditLoggingMiddleware",
    "RequestValidationMiddleware",
]
