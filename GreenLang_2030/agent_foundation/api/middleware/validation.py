"""
Request Validation Middleware for FastAPI.

This middleware validates all incoming requests to prevent:
- Oversized requests (DoS)
- Invalid content types
- Missing required headers
- Malicious headers
- Request flooding

Example:
    >>> from fastapi import FastAPI
    >>> from api.middleware.validation import create_validation_middleware
    >>>
    >>> app = FastAPI()
    >>> app.middleware("http")(create_validation_middleware())
"""

from typing import Callable, Optional, List, Set
from fastapi import Request, Response, HTTPException, status
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp
import time
import logging
from datetime import datetime, timedelta
from collections import defaultdict
import hashlib

from security.input_validation import InputValidator

logger = logging.getLogger(__name__)


class RequestValidationConfig:
    """Configuration for request validation middleware."""

    # Request size limits
    MAX_REQUEST_SIZE = 10_000_000  # 10MB
    MAX_HEADER_SIZE = 8192  # 8KB
    MAX_URL_LENGTH = 2048
    MAX_QUERY_PARAMS = 50

    # Rate limiting
    RATE_LIMIT_ENABLED = True
    RATE_LIMIT_REQUESTS = 100  # requests per window
    RATE_LIMIT_WINDOW = 60  # seconds

    # Content type validation
    ALLOWED_CONTENT_TYPES = {
        'application/json',
        'application/x-www-form-urlencoded',
        'multipart/form-data',
        'text/plain'
    }

    # Header validation
    MAX_USER_AGENT_LENGTH = 500
    REQUIRED_HEADERS = {'user-agent'}  # Headers that must be present

    # Security headers to block
    BLOCKED_HEADERS = {
        'x-forwarded-host',  # Potential host header injection
    }


class RateLimiter:
    """
    Simple in-memory rate limiter.

    For production, use Redis-based rate limiting.

    Example:
        >>> limiter = RateLimiter(requests=100, window=60)
        >>> if not limiter.allow_request("192.168.1.1"):
        ...     raise HTTPException(429, "Rate limit exceeded")
    """

    def __init__(self, requests: int = 100, window: int = 60):
        """
        Initialize rate limiter.

        Args:
            requests: Number of requests allowed per window
            window: Time window in seconds
        """
        self.requests = requests
        self.window = window
        self.clients: Dict[str, List[float]] = defaultdict(list)

    def allow_request(self, client_id: str) -> bool:
        """
        Check if request is allowed for client.

        Args:
            client_id: Client identifier (IP address, API key, etc.)

        Returns:
            True if request allowed, False otherwise
        """
        now = time.time()
        cutoff = now - self.window

        # Remove old requests
        self.clients[client_id] = [
            req_time for req_time in self.clients[client_id]
            if req_time > cutoff
        ]

        # Check limit
        if len(self.clients[client_id]) >= self.requests:
            logger.warning(
                f"Rate limit exceeded for client: {client_id}",
                extra={"client": client_id, "requests": len(self.clients[client_id])}
            )
            return False

        # Add current request
        self.clients[client_id].append(now)
        return True

    def cleanup(self):
        """Remove expired entries (call periodically)."""
        now = time.time()
        cutoff = now - self.window

        for client_id in list(self.clients.keys()):
            self.clients[client_id] = [
                req_time for req_time in self.clients[client_id]
                if req_time > cutoff
            ]

            if not self.clients[client_id]:
                del self.clients[client_id]


class RequestValidationMiddleware(BaseHTTPMiddleware):
    """
    Validates all incoming HTTP requests.

    This middleware provides defense-in-depth security by validating:
    - Request size
    - Content type
    - Headers
    - URL length
    - Rate limiting

    Example:
        >>> app.add_middleware(RequestValidationMiddleware, config=config)
    """

    def __init__(
        self,
        app: ASGIApp,
        config: Optional[RequestValidationConfig] = None,
        rate_limiter: Optional[RateLimiter] = None
    ):
        """
        Initialize validation middleware.

        Args:
            app: ASGI application
            config: Validation configuration
            rate_limiter: Rate limiter instance
        """
        super().__init__(app)
        self.config = config or RequestValidationConfig()
        self.rate_limiter = rate_limiter or RateLimiter(
            requests=self.config.RATE_LIMIT_REQUESTS,
            window=self.config.RATE_LIMIT_WINDOW
        )
        self.validator = InputValidator()

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """
        Validate request and call next middleware.

        Args:
            request: Incoming HTTP request
            call_next: Next middleware in chain

        Returns:
            HTTP response

        Raises:
            HTTPException: If validation fails
        """
        try:
            # Validate request
            self._validate_request(request)

            # Process request
            response = await call_next(request)

            # Add security headers to response
            self._add_security_headers(response)

            return response

        except HTTPException:
            raise

        except Exception as e:
            logger.error(
                f"Unexpected error in validation middleware: {e}",
                exc_info=True
            )
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Internal server error"
            )

    def _validate_request(self, request: Request) -> None:
        """
        Validate incoming request.

        Args:
            request: HTTP request to validate

        Raises:
            HTTPException: If validation fails
        """
        # 1. Rate limiting
        if self.config.RATE_LIMIT_ENABLED:
            client_id = self._get_client_id(request)
            if not self.rate_limiter.allow_request(client_id):
                raise HTTPException(
                    status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                    detail="Rate limit exceeded. Please try again later."
                )

        # 2. Validate URL length
        url_str = str(request.url)
        if len(url_str) > self.config.MAX_URL_LENGTH:
            logger.warning(
                f"URL too long: {len(url_str)} bytes",
                extra={"url_length": len(url_str), "max": self.config.MAX_URL_LENGTH}
            )
            raise HTTPException(
                status_code=status.HTTP_414_URI_TOO_LONG,
                detail=f"URL too long (max {self.config.MAX_URL_LENGTH} characters)"
            )

        # 3. Validate query parameters count
        if len(request.query_params) > self.config.MAX_QUERY_PARAMS:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Too many query parameters (max {self.config.MAX_QUERY_PARAMS})"
            )

        # 4. Validate headers
        self._validate_headers(request)

        # 5. Validate content type for POST/PUT/PATCH
        if request.method in ['POST', 'PUT', 'PATCH']:
            self._validate_content_type(request)

        # 6. Validate content length
        self._validate_content_length(request)

    def _validate_headers(self, request: Request) -> None:
        """
        Validate HTTP headers.

        Args:
            request: HTTP request

        Raises:
            HTTPException: If header validation fails
        """
        # Check required headers
        for required_header in self.config.REQUIRED_HEADERS:
            if required_header not in request.headers:
                logger.warning(
                    f"Missing required header: {required_header}",
                    extra={"header": required_header}
                )
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Missing required header: {required_header}"
                )

        # Check blocked headers
        for blocked_header in self.config.BLOCKED_HEADERS:
            if blocked_header in request.headers:
                logger.warning(
                    f"Blocked header detected: {blocked_header}",
                    extra={"header": blocked_header, "value": request.headers[blocked_header]}
                )
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Header not allowed: {blocked_header}"
                )

        # Validate user-agent length
        user_agent = request.headers.get('user-agent', '')
        if len(user_agent) > self.config.MAX_USER_AGENT_LENGTH:
            logger.warning(
                f"User-Agent header too long: {len(user_agent)} bytes",
                extra={"length": len(user_agent)}
            )
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"User-Agent header too long (max {self.config.MAX_USER_AGENT_LENGTH} characters)"
            )

        # Check for suspicious patterns in headers
        for header_name, header_value in request.headers.items():
            # Check for SQL injection in headers
            try:
                self.validator.validate_no_sql_injection(header_value, header_name)
            except ValueError as e:
                logger.warning(
                    f"Suspicious header value detected: {header_name}",
                    extra={"header": header_name, "error": str(e)}
                )
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Invalid header value: {header_name}"
                )

    def _validate_content_type(self, request: Request) -> None:
        """
        Validate content-type header.

        Args:
            request: HTTP request

        Raises:
            HTTPException: If content-type invalid
        """
        content_type = request.headers.get('content-type', '')

        if not content_type:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Content-Type header required for this request"
            )

        # Extract base content type (remove charset, boundary, etc.)
        base_content_type = content_type.split(';')[0].strip().lower()

        # Check if allowed
        if base_content_type not in self.config.ALLOWED_CONTENT_TYPES:
            logger.warning(
                f"Invalid content type: {base_content_type}",
                extra={"content_type": base_content_type}
            )
            raise HTTPException(
                status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
                detail=f"Content-Type must be one of: {', '.join(self.config.ALLOWED_CONTENT_TYPES)}"
            )

    def _validate_content_length(self, request: Request) -> None:
        """
        Validate content-length header.

        Args:
            request: HTTP request

        Raises:
            HTTPException: If content-length too large
        """
        content_length = request.headers.get('content-length')

        if content_length:
            try:
                length = int(content_length)

                if length > self.config.MAX_REQUEST_SIZE:
                    logger.warning(
                        f"Request body too large: {length} bytes",
                        extra={"content_length": length, "max": self.config.MAX_REQUEST_SIZE}
                    )
                    raise HTTPException(
                        status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                        detail=f"Request body too large (max {self.config.MAX_REQUEST_SIZE} bytes)"
                    )

            except ValueError:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Invalid Content-Length header"
                )

    def _get_client_id(self, request: Request) -> str:
        """
        Get client identifier for rate limiting.

        Args:
            request: HTTP request

        Returns:
            Client identifier (IP address or API key)
        """
        # Try to get API key from header
        api_key = request.headers.get('x-api-key')
        if api_key:
            return hashlib.sha256(api_key.encode()).hexdigest()[:16]

        # Fall back to IP address
        # Check for X-Forwarded-For (behind proxy)
        forwarded_for = request.headers.get('x-forwarded-for')
        if forwarded_for:
            # Take first IP (client IP)
            client_ip = forwarded_for.split(',')[0].strip()
        else:
            # Direct connection
            client_ip = request.client.host if request.client else "unknown"

        return client_ip

    def _add_security_headers(self, response: Response) -> None:
        """
        Add security headers to response.

        Args:
            response: HTTP response to enhance
        """
        # OWASP recommended security headers
        response.headers['X-Content-Type-Options'] = 'nosniff'
        response.headers['X-Frame-Options'] = 'DENY'
        response.headers['X-XSS-Protection'] = '1; mode=block'
        response.headers['Strict-Transport-Security'] = 'max-age=31536000; includeSubDomains'
        response.headers['Referrer-Policy'] = 'strict-origin-when-cross-origin'

        # Content Security Policy
        response.headers['Content-Security-Policy'] = (
            "default-src 'self'; "
            "script-src 'self' 'unsafe-inline' 'unsafe-eval'; "
            "style-src 'self' 'unsafe-inline'; "
            "img-src 'self' data: https:; "
            "font-src 'self'; "
            "connect-src 'self'; "
            "frame-ancestors 'none'"
        )


def create_validation_middleware(
    config: Optional[RequestValidationConfig] = None,
    rate_limiter: Optional[RateLimiter] = None
) -> Callable:
    """
    Create validation middleware factory.

    Args:
        config: Validation configuration
        rate_limiter: Rate limiter instance

    Returns:
        Middleware callable

    Example:
        >>> app.middleware("http")(create_validation_middleware())
    """
    async def validation_middleware(request: Request, call_next: Callable) -> Response:
        """Validation middleware function."""
        middleware = RequestValidationMiddleware(
            app=None,  # Not needed for function-based middleware
            config=config,
            rate_limiter=rate_limiter
        )
        return await middleware.dispatch(request, call_next)

    return validation_middleware


# Exception handlers for security errors


async def rate_limit_handler(request: Request, exc: HTTPException) -> JSONResponse:
    """Handle rate limit exceeded errors."""
    return JSONResponse(
        status_code=status.HTTP_429_TOO_MANY_REQUESTS,
        content={
            "error": "rate_limit_exceeded",
            "message": "Too many requests. Please try again later.",
            "retry_after": 60
        },
        headers={
            "Retry-After": "60"
        }
    )


async def validation_error_handler(request: Request, exc: HTTPException) -> JSONResponse:
    """Handle validation errors."""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": "validation_error",
            "message": str(exc.detail),
            "path": request.url.path
        }
    )
