"""
GL-015 INSULSCAN - API Middleware

Custom middleware for authentication, rate limiting,
logging, and request/response processing.

Provides:
- RequestIDMiddleware: Unique request ID injection
- RequestLoggingMiddleware: Request/response logging with correlation IDs
- RateLimitMiddleware: Token bucket rate limiting per client
- ErrorHandlingMiddleware: Structured error responses with proper HTTP codes
- CORSMiddleware configuration helper
- RequestValidationMiddleware: Content-Type and size validation
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
AGENT_ID = "GL-015"
AGENT_NAME = "INSULSCAN"
AGENT_VERSION = "1.0.0"


# =============================================================================
# Request ID Middleware
# =============================================================================

class RequestIDMiddleware(BaseHTTPMiddleware):
    """
    Request ID injection middleware.

    Features:
    - Generates unique request ID for each request
    - Accepts existing X-Request-ID header if provided
    - Adds request ID to response headers
    - Stores in request.state for access in route handlers
    """

    def __init__(
        self,
        app: ASGIApp,
        header_name: str = "X-Request-ID",
        generator: Optional[Callable[[], str]] = None,
    ) -> None:
        """
        Initialize request ID middleware.

        Args:
            app: ASGI application
            header_name: Header name for request ID
            generator: Custom ID generator function
        """
        super().__init__(app)
        self.header_name = header_name
        self.generator = generator or (lambda: str(uuid.uuid4()))

    async def dispatch(
        self,
        request: Request,
        call_next: Callable,
    ) -> Response:
        """Process request with ID injection."""
        # Get existing or generate new request ID
        request_id = request.headers.get(self.header_name)
        if not request_id:
            request_id = self.generator()

        # Store in request state
        request.state.request_id = request_id

        # Also generate/get correlation ID for distributed tracing
        correlation_id = request.headers.get(
            "X-Correlation-ID",
            str(uuid.uuid4())
        )
        request.state.correlation_id = correlation_id

        # Process request
        response = await call_next(request)

        # Add IDs to response headers
        response.headers[self.header_name] = request_id
        response.headers["X-Correlation-ID"] = correlation_id

        return response


# =============================================================================
# Request Logging Middleware
# =============================================================================

class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """
    Request/response logging middleware with correlation IDs.

    Features:
    - Logs request method, path, client IP
    - Logs response status code and processing time
    - Optionally logs request/response bodies
    - Adds timing headers to responses
    - Excludes health check endpoints from verbose logging
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
        self.exclude_paths = exclude_paths or [
            "/api/v1/health",
            "/api/v1/metrics",
            "/health",
            "/metrics",
        ]
        self.max_body_log_size = max_body_log_size

    async def dispatch(
        self,
        request: Request,
        call_next: Callable,
    ) -> Response:
        """Process request with logging."""
        # Check if path should be excluded
        should_log = not any(
            request.url.path.startswith(path) for path in self.exclude_paths
        )

        # Get request ID
        request_id = getattr(request.state, "request_id", str(uuid.uuid4())[:8])

        # Get client info
        client_ip = self._get_client_ip(request)

        # Log request (if not excluded)
        start_time = time.time()
        if should_log:
            logger.info(
                f"[{request_id}] --> {request.method} {request.url.path} "
                f"| client={client_ip}"
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

        # Log response (if not excluded)
        if should_log:
            logger.info(
                f"[{request_id}] <-- {response.status_code} "
                f"| {process_time_ms:.1f}ms"
            )

        # Add timing header
        response.headers["X-Process-Time-Ms"] = f"{process_time_ms:.1f}"

        return response

    def _get_client_ip(self, request: Request) -> str:
        """Extract client IP from request."""
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
    - Proper 429 response with Retry-After header
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
            "/api/v1/health",
            "/api/v1/metrics",
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
            request_id = getattr(request.state, "request_id", "unknown")

            logger.warning(
                f"[{request_id}] Rate limit exceeded for {client_id}"
            )

            return JSONResponse(
                status_code=429,
                content={
                    "error": "rate_limit_exceeded",
                    "message": f"Rate limit of {self.requests_per_minute} requests/minute exceeded",
                    "retry_after_seconds": int(retry_after),
                    "request_id": request_id,
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
        response.headers["X-RateLimit-Reset"] = str(int(time.time() + 60))

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
# Error Handling Middleware
# =============================================================================

class ErrorHandlingMiddleware(BaseHTTPMiddleware):
    """
    Error handling middleware for structured error responses.

    Features:
    - Catches unhandled exceptions
    - Maps Python exceptions to HTTP status codes
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
        request_id = getattr(request.state, "request_id", "unknown")

        try:
            response = await call_next(request)
            return response

        except ValueError as e:
            logger.warning(f"[{request_id}] Validation error: {e}")
            return self._create_error_response(
                status_code=400,
                error="validation_error",
                message=str(e),
                request_id=request_id,
            )

        except PermissionError as e:
            logger.warning(f"[{request_id}] Permission denied: {e}")
            return self._create_error_response(
                status_code=403,
                error="forbidden",
                message=str(e) or "Access denied",
                request_id=request_id,
            )

        except FileNotFoundError as e:
            logger.warning(f"[{request_id}] Not found: {e}")
            return self._create_error_response(
                status_code=404,
                error="not_found",
                message=str(e) or "Resource not found",
                request_id=request_id,
            )

        except TimeoutError as e:
            logger.error(f"[{request_id}] Timeout: {e}")
            return self._create_error_response(
                status_code=504,
                error="timeout",
                message=str(e) or "Request timed out",
                request_id=request_id,
            )

        except NotImplementedError as e:
            logger.warning(f"[{request_id}] Not implemented: {e}")
            return self._create_error_response(
                status_code=501,
                error="not_implemented",
                message=str(e) or "Feature not implemented",
                request_id=request_id,
            )

        except Exception as e:
            # Log unexpected errors with full traceback
            logger.error(
                f"[{request_id}] Unhandled exception: {e}",
                exc_info=True,
            )

            # Create error response
            if self.debug:
                import traceback
                details = traceback.format_exc() if self.include_stack_trace else None
                message = str(e)
            else:
                details = None
                message = "An internal error occurred"

            return self._create_error_response(
                status_code=500,
                error="internal_error",
                message=message,
                request_id=request_id,
                details=details,
            )

    def _create_error_response(
        self,
        status_code: int,
        error: str,
        message: str,
        request_id: str,
        details: Optional[str] = None,
    ) -> JSONResponse:
        """Create structured error response."""
        content = {
            "error": error,
            "message": message,
            "request_id": request_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "agent_id": AGENT_ID,
        }

        if details:
            content["details"] = details

        return JSONResponse(
            status_code=status_code,
            content=content,
        )


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
    - Returns proper 413/415 responses
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
        request_id = getattr(request.state, "request_id", "unknown")

        # Check Content-Length
        content_length = request.headers.get("Content-Length")
        if content_length:
            try:
                size = int(content_length)
                if size > self.max_request_size:
                    logger.warning(
                        f"[{request_id}] Request too large: {size} bytes"
                    )
                    return JSONResponse(
                        status_code=413,
                        content={
                            "error": "request_too_large",
                            "message": f"Request body exceeds maximum size of {self.max_request_size} bytes",
                            "request_id": request_id,
                            "timestamp": datetime.now(timezone.utc).isoformat(),
                        },
                    )
            except ValueError:
                pass

        # Check Content-Type for POST/PUT/PATCH
        if self.require_content_type and request.method in ["POST", "PUT", "PATCH"]:
            content_type = request.headers.get("Content-Type", "")
            content_type_base = content_type.split(";")[0].strip()

            # Allow empty content type for requests without body
            if content_type_base and content_type_base not in self.allowed_content_types:
                logger.warning(
                    f"[{request_id}] Unsupported content type: {content_type_base}"
                )
                return JSONResponse(
                    status_code=415,
                    content={
                        "error": "unsupported_media_type",
                        "message": f"Content-Type must be one of: {self.allowed_content_types}",
                        "request_id": request_id,
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                    },
                )

        return await call_next(request)


# =============================================================================
# Authentication Middleware
# =============================================================================

class AuthenticationMiddleware(BaseHTTPMiddleware):
    """
    Authentication middleware supporting API key and JWT.

    Features:
    - API key validation via X-API-Key header
    - JWT Bearer token validation
    - Exempt paths for public endpoints
    - Stores user info in request.state
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
            "/api/v1/health",
            "/api/v1/metrics",
            "/health",
            "/metrics",
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
        request_id = getattr(request.state, "request_id", "unknown")

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
            logger.warning(f"[{request_id}] Authentication required but not provided")
            return JSONResponse(
                status_code=401,
                content={
                    "error": "unauthorized",
                    "message": "Valid API key or JWT token required",
                    "request_id": request_id,
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
        """Validate API key and return user info."""
        # Check unhashed key (for development)
        user_info = self.api_keys.get(api_key)
        if user_info:
            return {
                "user_id": user_info.get("user_id"),
                "tenant_id": user_info.get("tenant_id"),
                "roles": user_info.get("roles", []),
                "api_key_id": user_info.get("key_id"),
            }

        # Check hashed key (for production)
        key_hash = hashlib.sha256(api_key.encode()).hexdigest()
        user_info = self.api_keys.get(key_hash)
        if user_info:
            return {
                "user_id": user_info.get("user_id"),
                "tenant_id": user_info.get("tenant_id"),
                "roles": user_info.get("roles", []),
                "api_key_id": user_info.get("key_id"),
            }

        return None

    def _validate_jwt_token(self, token: str) -> Optional[Dict[str, Any]]:
        """Validate JWT token and return user info."""
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
# CORS Configuration Helper
# =============================================================================

def get_cors_config(
    allow_origins: Optional[List[str]] = None,
    allow_credentials: bool = True,
    allow_methods: Optional[List[str]] = None,
    allow_headers: Optional[List[str]] = None,
    expose_headers: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Get CORS middleware configuration.

    Args:
        allow_origins: Allowed origins (default: ["*"])
        allow_credentials: Allow credentials
        allow_methods: Allowed HTTP methods
        allow_headers: Allowed headers
        expose_headers: Headers to expose to browser

    Returns:
        Dictionary of CORS configuration for CORSMiddleware
    """
    return {
        "allow_origins": allow_origins or ["*"],
        "allow_credentials": allow_credentials,
        "allow_methods": allow_methods or ["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        "allow_headers": allow_headers or ["*"],
        "expose_headers": expose_headers or [
            "X-Request-ID",
            "X-Correlation-ID",
            "X-Process-Time-Ms",
            "X-RateLimit-Limit",
            "X-RateLimit-Remaining",
            "X-RateLimit-Reset",
            "X-GL-Agent-ID",
            "X-GL-Agent-Name",
            "X-GL-Agent-Version",
            "X-GL-Timestamp",
        ],
    }


# =============================================================================
# Export all middleware
# =============================================================================

__all__ = [
    "RequestIDMiddleware",
    "RequestLoggingMiddleware",
    "RateLimitMiddleware",
    "ErrorHandlingMiddleware",
    "RequestValidationMiddleware",
    "AuthenticationMiddleware",
    "ProvenanceMiddleware",
    "get_cors_config",
    # Constants
    "AGENT_ID",
    "AGENT_NAME",
    "AGENT_VERSION",
]
