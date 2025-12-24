"""
GL-011 FUELCRAFT - API Middleware

Custom middleware for request processing including:
- Request ID generation for tracing
- Audit logging for compliance
- Rate limiting with token bucket algorithm
- Error handling with structured responses
- Provenance tracking headers

Standards Compliance:
- IEC 61511 (Functional Safety)
- ISO 14064 (GHG Quantification audit trails)
"""

from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional
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
# Request ID Middleware
# =============================================================================

class RequestIDMiddleware(BaseHTTPMiddleware):
    """
    Request ID middleware for distributed tracing.

    Generates a unique request ID for each incoming request
    and adds it to response headers for end-to-end tracing.
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
        """Process request with ID tracking."""
        # Get or generate request ID
        request_id = request.headers.get(self.header_name)

        if not request_id and self.generate_if_missing:
            request_id = str(uuid.uuid4())[:8]

        # Store in request state for access in handlers
        request.state.request_id = request_id

        # Process request
        response = await call_next(request)

        # Add request ID to response headers
        if request_id:
            response.headers[self.header_name] = request_id

        return response


# =============================================================================
# Audit Logging Middleware
# =============================================================================

class AuditLoggingMiddleware(BaseHTTPMiddleware):
    """
    Audit logging middleware for compliance.

    Logs all API requests and responses with:
    - Timestamp
    - Request details (method, path, client IP)
    - Response status and timing
    - User identification (if authenticated)

    Compliant with ISO 14064 audit trail requirements.
    """

    def __init__(
        self,
        app,
        log_request_body: bool = False,
        log_response_body: bool = False,
        exclude_paths: Optional[List[str]] = None,
        audit_logger: Optional[logging.Logger] = None,
    ) -> None:
        """
        Initialize audit logging middleware.

        Args:
            app: ASGI application
            log_request_body: Include request body in logs
            log_response_body: Include response body in logs
            exclude_paths: Paths to exclude from logging
            audit_logger: Custom logger for audit entries
        """
        super().__init__(app)
        self.log_request_body = log_request_body
        self.log_response_body = log_response_body
        self.exclude_paths = exclude_paths or [
            "/health/live",
            "/health/ready",
            "/health/startup",
            "/docs",
            "/redoc",
            "/openapi.json",
        ]
        self.audit_logger = audit_logger or logging.getLogger("fuelcraft.audit")

    async def dispatch(
        self,
        request: Request,
        call_next: Callable,
    ) -> Response:
        """Process request with audit logging."""
        # Skip excluded paths
        if any(request.url.path.startswith(path) for path in self.exclude_paths):
            return await call_next(request)

        # Capture request details
        request_id = getattr(request.state, "request_id", str(uuid.uuid4())[:8])
        client_ip = request.client.host if request.client else "unknown"
        start_time = time.time()

        # Log request
        audit_entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "request_id": request_id,
            "event_type": "api_request",
            "method": request.method,
            "path": request.url.path,
            "query_params": str(request.query_params),
            "client_ip": client_ip,
            "user_agent": request.headers.get("User-Agent", ""),
        }

        # Capture request body if enabled
        if self.log_request_body and request.method in ["POST", "PUT", "PATCH"]:
            try:
                body = await request.body()
                # Hash sensitive data
                body_hash = hashlib.sha256(body).hexdigest()[:16]
                audit_entry["request_body_hash"] = body_hash
                audit_entry["request_body_size"] = len(body)
            except Exception:
                pass

        self.audit_logger.info(f"[{request_id}] REQUEST: {json.dumps(audit_entry)}")

        # Process request
        try:
            response = await call_next(request)
            process_time = time.time() - start_time

            # Log response
            response_entry = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "request_id": request_id,
                "event_type": "api_response",
                "status_code": response.status_code,
                "process_time_ms": round(process_time * 1000, 2),
            }

            self.audit_logger.info(
                f"[{request_id}] RESPONSE: {response.status_code} "
                f"({process_time*1000:.1f}ms)"
            )

            # Add timing header
            response.headers["X-Process-Time"] = f"{process_time:.3f}"

            return response

        except Exception as e:
            process_time = time.time() - start_time

            # Log error
            error_entry = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "request_id": request_id,
                "event_type": "api_error",
                "error_type": type(e).__name__,
                "error_message": str(e),
                "process_time_ms": round(process_time * 1000, 2),
            }

            self.audit_logger.error(
                f"[{request_id}] ERROR: {json.dumps(error_entry)}"
            )

            raise


# =============================================================================
# Rate Limiting Middleware
# =============================================================================

class RateLimitMiddleware(BaseHTTPMiddleware):
    """
    Rate limiting middleware using token bucket algorithm.

    Limits requests per client IP address with:
    - Sustained rate limit (requests per minute)
    - Burst capacity for temporary spikes
    - Per-client tracking
    - Retry-After header on rate limit

    Exempt paths (like health checks) are not rate limited.
    """

    def __init__(
        self,
        app,
        requests_per_minute: int = 100,
        burst_size: int = 20,
        exempt_paths: Optional[List[str]] = None,
    ) -> None:
        """
        Initialize rate limiter.

        Args:
            app: ASGI application
            requests_per_minute: Sustained request rate
            burst_size: Maximum burst above rate limit
            exempt_paths: Paths exempt from rate limiting
        """
        super().__init__(app)
        self.requests_per_minute = requests_per_minute
        self.burst_size = burst_size
        self.exempt_paths = exempt_paths or [
            "/health/live",
            "/health/ready",
            "/health/startup",
        ]

        # Token bucket state per client
        self._tokens: Dict[str, float] = {}
        self._last_update: Dict[str, float] = {}

    async def dispatch(
        self,
        request: Request,
        call_next: Callable,
    ) -> Response:
        """Process request with rate limiting."""
        # Skip rate limiting for exempt paths
        if any(request.url.path.startswith(path) for path in self.exempt_paths):
            return await call_next(request)

        # Get client identifier (IP + optional API key)
        client_ip = request.client.host if request.client else "unknown"
        api_key = request.headers.get("X-API-Key", "")
        client_id = f"{client_ip}:{api_key[:8] if api_key else 'anon'}"

        # Check rate limit
        if not self._check_rate_limit(client_id):
            retry_after = 60  # Seconds until tokens refill

            return JSONResponse(
                status_code=429,
                content={
                    "error": "rate_limit_exceeded",
                    "code": 429,
                    "message": f"Rate limit exceeded. Maximum {self.requests_per_minute} requests per minute.",
                    "retry_after_seconds": retry_after,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                },
                headers={
                    "Retry-After": str(retry_after),
                    "X-RateLimit-Limit": str(self.requests_per_minute),
                    "X-RateLimit-Remaining": "0",
                },
            )

        # Process request
        response = await call_next(request)

        # Add rate limit headers
        remaining = int(self._get_remaining_tokens(client_id))
        response.headers["X-RateLimit-Limit"] = str(self.requests_per_minute)
        response.headers["X-RateLimit-Remaining"] = str(remaining)

        return response

    def _check_rate_limit(self, client_id: str) -> bool:
        """Check if request is within rate limit using token bucket."""
        now = time.time()

        # Initialize bucket if new client
        if client_id not in self._tokens:
            self._tokens[client_id] = float(self.burst_size)
            self._last_update[client_id] = now

        # Add tokens based on time elapsed
        elapsed = now - self._last_update[client_id]
        token_rate = self.requests_per_minute / 60.0
        self._tokens[client_id] = min(
            self.burst_size,
            self._tokens[client_id] + elapsed * token_rate
        )
        self._last_update[client_id] = now

        # Check if token available
        if self._tokens[client_id] >= 1.0:
            self._tokens[client_id] -= 1.0
            return True

        return False

    def _get_remaining_tokens(self, client_id: str) -> float:
        """Get remaining tokens for client."""
        return self._tokens.get(client_id, 0)


# =============================================================================
# Error Handling Middleware
# =============================================================================

class ErrorHandlingMiddleware(BaseHTTPMiddleware):
    """
    Error handling middleware with structured responses.

    Catches exceptions and returns consistent error responses:
    - Structured JSON error format
    - Request ID for tracing
    - Safe error messages (no internal details in production)
    - Logging of full exception details
    """

    def __init__(
        self,
        app,
        debug: bool = False,
        include_traceback: bool = False,
    ) -> None:
        """
        Initialize error handling middleware.

        Args:
            app: ASGI application
            debug: Enable debug mode with detailed errors
            include_traceback: Include traceback in responses (debug only)
        """
        super().__init__(app)
        self.debug = debug
        self.include_traceback = include_traceback and debug

    async def dispatch(
        self,
        request: Request,
        call_next: Callable,
    ) -> Response:
        """Process request with error handling."""
        try:
            return await call_next(request)

        except Exception as e:
            request_id = getattr(request.state, "request_id", None)

            # Log full exception
            logger.error(
                f"Unhandled exception in request {request_id}: {e}",
                exc_info=True,
            )

            # Build error response
            error_response = {
                "error": "internal_server_error",
                "code": 500,
                "message": "An unexpected error occurred",
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }

            if request_id:
                error_response["request_id"] = request_id

            # Add details in debug mode
            if self.debug:
                error_response["details"] = {
                    "type": type(e).__name__,
                    "message": str(e),
                }

                if self.include_traceback:
                    import traceback
                    error_response["traceback"] = traceback.format_exc()

            return JSONResponse(
                status_code=500,
                content=error_response,
            )


# =============================================================================
# Provenance Middleware
# =============================================================================

class ProvenanceMiddleware(BaseHTTPMiddleware):
    """
    Provenance tracking middleware for audit compliance.

    Adds headers to responses for data lineage:
    - X-GL-Agent: Agent identifier
    - X-GL-Version: API version
    - X-GL-Timestamp: Processing timestamp
    - X-GL-Bundle-Hash: Computation hash (if available)

    Compliant with ISO 14064 provenance requirements.
    """

    def __init__(
        self,
        app,
        agent_name: str = "GL-011-FUELCRAFT",
        version: str = "1.0.0",
    ) -> None:
        """
        Initialize provenance middleware.

        Args:
            app: ASGI application
            agent_name: Agent identifier
            version: API version
        """
        super().__init__(app)
        self.agent_name = agent_name
        self.version = version

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
        response.headers["X-GL-Timestamp"] = datetime.now(timezone.utc).isoformat()

        # Add bundle hash if present in response
        # (This would be set by the handler for calculation responses)
        if hasattr(request.state, "bundle_hash"):
            response.headers["X-GL-Bundle-Hash"] = request.state.bundle_hash

        return response


# =============================================================================
# Authentication Middleware
# =============================================================================

class AuthenticationMiddleware(BaseHTTPMiddleware):
    """
    Authentication middleware supporting:
    - API key authentication (X-API-Key header)
    - JWT Bearer token authentication
    - Service account authentication

    Exempt paths (like health checks) bypass authentication.
    """

    def __init__(
        self,
        app,
        api_keys: Optional[Dict[str, str]] = None,
        require_auth: bool = False,
        exempt_paths: Optional[List[str]] = None,
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
            "/health/live",
            "/health/ready",
            "/health/startup",
            "/docs",
            "/redoc",
            "/openapi.json",
            "/info",
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
        auth_method = None

        if api_key:
            user_id = self._validate_api_key(api_key)
            auth_method = "api_key"
        elif auth_header and auth_header.startswith("Bearer "):
            token = auth_header[7:]
            user_id = self._validate_jwt_token(token)
            auth_method = "jwt"

        # Require auth check
        if self.require_auth and not user_id:
            return JSONResponse(
                status_code=401,
                content={
                    "error": "unauthorized",
                    "code": 401,
                    "message": "Authentication required. Provide API key or Bearer token.",
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                },
                headers={"WWW-Authenticate": "Bearer"},
            )

        # Store user info in request state
        request.state.user_id = user_id
        request.state.authenticated = user_id is not None
        request.state.auth_method = auth_method

        response = await call_next(request)
        return response

    def _validate_api_key(self, api_key: str) -> Optional[str]:
        """Validate API key and return user ID."""
        return self.api_keys.get(api_key)

    def _validate_jwt_token(self, token: str) -> Optional[str]:
        """Validate JWT token and return user ID."""
        # Placeholder - implement JWT validation with PyJWT
        try:
            # import jwt
            # payload = jwt.decode(token, SECRET_KEY, algorithms=["HS256"])
            # return payload.get("sub")
            return None
        except Exception:
            return None


# =============================================================================
# CORS Middleware Configuration
# =============================================================================

def get_cors_config() -> Dict[str, Any]:
    """
    Get CORS configuration for the API.

    Returns production-safe CORS settings.
    """
    return {
        "allow_origins": [
            "https://app.greenlang.io",
            "https://admin.greenlang.io",
        ],
        "allow_credentials": True,
        "allow_methods": ["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        "allow_headers": [
            "Authorization",
            "X-API-Key",
            "X-Request-ID",
            "Content-Type",
            "Accept",
        ],
        "expose_headers": [
            "X-Request-ID",
            "X-Process-Time",
            "X-RateLimit-Limit",
            "X-RateLimit-Remaining",
            "X-GL-Agent",
            "X-GL-Version",
            "X-GL-Timestamp",
            "X-GL-Bundle-Hash",
        ],
        "max_age": 86400,  # 24 hours
    }
