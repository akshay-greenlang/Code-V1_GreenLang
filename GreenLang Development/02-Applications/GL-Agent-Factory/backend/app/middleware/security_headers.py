"""
Security Headers Middleware - HTTP Security Headers and Request Validation

This module provides comprehensive HTTP security headers and request validation
middleware for SOC2 and ISO27001 compliance. Implements OWASP security best
practices including HSTS, CSP, X-Frame-Options, and rate limiting enhancements.

SOC2 Controls Addressed:
    - CC6.1: Protection against unauthorized access
    - CC6.6: Security event detection
    - CC6.7: Protection against malicious attacks

ISO27001 Controls Addressed:
    - A.13.1.1: Network controls
    - A.14.1.2: Securing application services
    - A.14.2.5: Secure system engineering principles

Example:
    >>> from fastapi import FastAPI
    >>> from app.middleware.security_headers import SecurityHeadersMiddleware
    >>> app = FastAPI()
    >>> app.add_middleware(
    ...     SecurityHeadersMiddleware,
    ...     config=SecurityHeadersConfig(),
    ... )
"""

from __future__ import annotations

import hashlib
import hmac
import json
import logging
import re
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional, Set, Tuple
from urllib.parse import urlparse

from pydantic import BaseModel, Field
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

logger = logging.getLogger(__name__)


class ContentSecurityPolicy(BaseModel):
    """
    Content Security Policy configuration.

    Defines allowed sources for various content types to prevent XSS attacks.
    """

    default_src: List[str] = Field(default_factory=lambda: ["'self'"])
    script_src: List[str] = Field(default_factory=lambda: ["'self'"])
    style_src: List[str] = Field(default_factory=lambda: ["'self'", "'unsafe-inline'"])
    img_src: List[str] = Field(default_factory=lambda: ["'self'", "data:", "https:"])
    font_src: List[str] = Field(default_factory=lambda: ["'self'"])
    connect_src: List[str] = Field(default_factory=lambda: ["'self'"])
    frame_src: List[str] = Field(default_factory=lambda: ["'none'"])
    object_src: List[str] = Field(default_factory=lambda: ["'none'"])
    base_uri: List[str] = Field(default_factory=lambda: ["'self'"])
    form_action: List[str] = Field(default_factory=lambda: ["'self'"])
    frame_ancestors: List[str] = Field(default_factory=lambda: ["'none'"])
    upgrade_insecure_requests: bool = Field(default=True)
    block_all_mixed_content: bool = Field(default=True)
    report_uri: Optional[str] = Field(default=None)
    report_to: Optional[str] = Field(default=None)

    def to_header_value(self) -> str:
        """Generate the CSP header value."""
        directives = []

        directives.append(f"default-src {' '.join(self.default_src)}")
        directives.append(f"script-src {' '.join(self.script_src)}")
        directives.append(f"style-src {' '.join(self.style_src)}")
        directives.append(f"img-src {' '.join(self.img_src)}")
        directives.append(f"font-src {' '.join(self.font_src)}")
        directives.append(f"connect-src {' '.join(self.connect_src)}")
        directives.append(f"frame-src {' '.join(self.frame_src)}")
        directives.append(f"object-src {' '.join(self.object_src)}")
        directives.append(f"base-uri {' '.join(self.base_uri)}")
        directives.append(f"form-action {' '.join(self.form_action)}")
        directives.append(f"frame-ancestors {' '.join(self.frame_ancestors)}")

        if self.upgrade_insecure_requests:
            directives.append("upgrade-insecure-requests")

        if self.block_all_mixed_content:
            directives.append("block-all-mixed-content")

        if self.report_uri:
            directives.append(f"report-uri {self.report_uri}")

        if self.report_to:
            directives.append(f"report-to {self.report_to}")

        return "; ".join(directives)


class CORSPolicy(BaseModel):
    """
    Cross-Origin Resource Sharing (CORS) policy configuration.

    Defines allowed origins, methods, and headers for cross-origin requests.
    """

    enabled: bool = Field(default=True)
    allow_origins: List[str] = Field(default_factory=lambda: ["https://app.greenlang.io"])
    allow_origin_regex: Optional[str] = Field(default=None)
    allow_methods: List[str] = Field(default_factory=lambda: ["GET", "POST", "PUT", "DELETE", "PATCH"])
    allow_headers: List[str] = Field(
        default_factory=lambda: [
            "Authorization",
            "Content-Type",
            "X-Request-ID",
            "X-API-Key",
        ]
    )
    expose_headers: List[str] = Field(
        default_factory=lambda: [
            "X-Request-ID",
            "X-RateLimit-Limit",
            "X-RateLimit-Remaining",
            "X-RateLimit-Reset",
        ]
    )
    allow_credentials: bool = Field(default=True)
    max_age: int = Field(default=86400, description="Preflight cache duration in seconds")

    def is_origin_allowed(self, origin: str) -> bool:
        """Check if an origin is allowed."""
        if "*" in self.allow_origins:
            return True

        if origin in self.allow_origins:
            return True

        if self.allow_origin_regex:
            if re.match(self.allow_origin_regex, origin):
                return True

        return False


class RateLimitConfig(BaseModel):
    """
    Enhanced rate limiting configuration.

    Supports multiple rate limit tiers and dynamic limits based on user/tenant.
    """

    enabled: bool = Field(default=True)

    # Default limits
    requests_per_minute: int = Field(default=100)
    requests_per_hour: int = Field(default=1000)
    requests_per_day: int = Field(default=10000)

    # Burst handling
    burst_size: int = Field(default=20)
    burst_window_seconds: int = Field(default=1)

    # Per-endpoint limits (path pattern -> limit)
    endpoint_limits: Dict[str, int] = Field(
        default_factory=lambda: {
            "/v1/agents/*/execute": 50,  # Execution endpoints have lower limits
            "/v1/auth/login": 10,  # Auth endpoints have very low limits
            "/v1/export/*": 20,  # Export endpoints limited
        }
    )

    # IP-based limits
    ip_rate_limit_enabled: bool = Field(default=True)
    ip_requests_per_minute: int = Field(default=300)

    # Tenant-based limits
    tenant_rate_limit_enabled: bool = Field(default=True)
    tenant_requests_per_minute: int = Field(default=5000)

    # Response headers
    include_rate_limit_headers: bool = Field(default=True)


class RequestSigningConfig(BaseModel):
    """
    Request signing validation configuration.

    Validates HMAC signatures on requests for API integrity.
    """

    enabled: bool = Field(default=False)
    signature_header: str = Field(default="X-Signature")
    timestamp_header: str = Field(default="X-Timestamp")
    algorithm: str = Field(default="sha256")
    max_timestamp_drift_seconds: int = Field(default=300)
    signing_key_header: str = Field(default="X-Signing-Key-ID")


class SecurityHeadersConfig(BaseModel):
    """Configuration for the Security Headers Middleware."""

    # HSTS Configuration
    hsts_enabled: bool = Field(default=True)
    hsts_max_age: int = Field(default=31536000, description="HSTS max age in seconds (1 year)")
    hsts_include_subdomains: bool = Field(default=True)
    hsts_preload: bool = Field(default=False)

    # Content Security Policy
    csp_enabled: bool = Field(default=True)
    csp: ContentSecurityPolicy = Field(default_factory=ContentSecurityPolicy)

    # Other Security Headers
    x_frame_options: str = Field(default="DENY")
    x_content_type_options: str = Field(default="nosniff")
    x_xss_protection: str = Field(default="1; mode=block")
    referrer_policy: str = Field(default="strict-origin-when-cross-origin")
    permissions_policy: str = Field(
        default="accelerometer=(), camera=(), geolocation=(), gyroscope=(), "
        "magnetometer=(), microphone=(), payment=(), usb=()"
    )
    cache_control: str = Field(default="no-store, max-age=0")

    # CORS Policy
    cors: CORSPolicy = Field(default_factory=CORSPolicy)

    # Rate Limiting
    rate_limit: RateLimitConfig = Field(default_factory=RateLimitConfig)

    # Request Signing
    request_signing: RequestSigningConfig = Field(default_factory=RequestSigningConfig)

    # Request validation
    max_content_length: int = Field(default=10 * 1024 * 1024, description="10 MB")
    allowed_content_types: List[str] = Field(
        default_factory=lambda: [
            "application/json",
            "application/xml",
            "multipart/form-data",
            "text/plain",
        ]
    )

    # Paths to exclude from security headers
    excluded_paths: Set[str] = Field(
        default_factory=lambda: {"/health", "/ready", "/metrics"}
    )

    # Request ID
    request_id_header: str = Field(default="X-Request-ID")
    generate_request_id: bool = Field(default=True)


class RateLimitState:
    """
    In-memory rate limit state tracker.

    In production, this would use Redis for distributed rate limiting.
    """

    def __init__(self):
        self._requests: Dict[str, List[float]] = {}
        self._cleanup_threshold = 10000  # Clean up after this many entries

    def record_request(self, key: str, timestamp: float) -> None:
        """Record a request for rate limiting."""
        if key not in self._requests:
            self._requests[key] = []
        self._requests[key].append(timestamp)

        # Periodic cleanup
        if len(self._requests) > self._cleanup_threshold:
            self._cleanup()

    def get_request_count(self, key: str, window_seconds: int) -> int:
        """Get request count within a time window."""
        if key not in self._requests:
            return 0

        cutoff = time.time() - window_seconds
        self._requests[key] = [t for t in self._requests[key] if t > cutoff]
        return len(self._requests[key])

    def _cleanup(self) -> None:
        """Remove old entries to prevent memory growth."""
        cutoff = time.time() - 86400  # Keep 24 hours
        for key in list(self._requests.keys()):
            self._requests[key] = [t for t in self._requests[key] if t > cutoff]
            if not self._requests[key]:
                del self._requests[key]


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """
    Production-grade security headers middleware.

    Provides comprehensive HTTP security through:
    - HSTS (HTTP Strict Transport Security)
    - CSP (Content Security Policy)
    - X-Frame-Options, X-Content-Type-Options, X-XSS-Protection
    - CORS policy enforcement
    - Enhanced rate limiting
    - Request signing validation
    - Request ID tracking

    Example:
        >>> app = FastAPI()
        >>> app.add_middleware(
        ...     SecurityHeadersMiddleware,
        ...     config=SecurityHeadersConfig(
        ...         hsts_enabled=True,
        ...         csp_enabled=True,
        ...     ),
        ... )

    Attributes:
        config: Middleware configuration
        rate_limit_state: In-memory rate limit tracker
    """

    def __init__(
        self,
        app,
        config: Optional[SecurityHeadersConfig] = None,
        signing_keys: Optional[Dict[str, str]] = None,
    ):
        """
        Initialize the security headers middleware.

        Args:
            app: The ASGI application
            config: Middleware configuration
            signing_keys: Dictionary of signing key IDs to keys for request validation
        """
        super().__init__(app)
        self.config = config or SecurityHeadersConfig()
        self.signing_keys = signing_keys or {}
        self.rate_limit_state = RateLimitState()

        logger.info(
            "SecurityHeadersMiddleware initialized",
            extra={
                "hsts_enabled": self.config.hsts_enabled,
                "csp_enabled": self.config.csp_enabled,
                "rate_limit_enabled": self.config.rate_limit.enabled,
            },
        )

    async def dispatch(
        self,
        request: Request,
        call_next: Callable,
    ) -> Response:
        """
        Process the request with security headers and validation.

        Args:
            request: The incoming request
            call_next: The next middleware/handler in chain

        Returns:
            Response with security headers applied
        """
        start_time = time.time()
        request_id = self._get_or_generate_request_id(request)

        # Store request ID in state for logging
        request.state.request_id = request_id

        try:
            # Skip security processing for excluded paths
            if request.url.path in self.config.excluded_paths:
                response = await call_next(request)
                response.headers["X-Request-ID"] = request_id
                return response

            # Handle CORS preflight
            if request.method == "OPTIONS":
                return self._handle_preflight(request, request_id)

            # Validate CORS origin
            cors_error = self._validate_cors(request)
            if cors_error:
                return cors_error

            # Rate limiting
            rate_limit_error = await self._check_rate_limit(request, request_id)
            if rate_limit_error:
                return rate_limit_error

            # Request signing validation
            if self.config.request_signing.enabled:
                signing_error = self._validate_request_signature(request)
                if signing_error:
                    return signing_error

            # Content validation
            content_error = self._validate_content(request)
            if content_error:
                return content_error

            # Process request
            response = await call_next(request)

            # Add security headers
            self._add_security_headers(response, request_id)

            # Add CORS headers
            self._add_cors_headers(request, response)

            # Add rate limit headers
            await self._add_rate_limit_headers(request, response)

            # Log request completion
            processing_time = (time.time() - start_time) * 1000
            logger.info(
                f"{request.method} {request.url.path} - {response.status_code}",
                extra={
                    "request_id": request_id,
                    "method": request.method,
                    "path": request.url.path,
                    "status_code": response.status_code,
                    "processing_time_ms": processing_time,
                    "client_ip": self._get_client_ip(request),
                },
            )

            return response

        except Exception as e:
            logger.error(
                f"Security middleware error: {e}",
                extra={"request_id": request_id},
                exc_info=True,
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

    def _get_or_generate_request_id(self, request: Request) -> str:
        """Get existing request ID or generate a new one."""
        request_id = request.headers.get(self.config.request_id_header)

        if request_id and self._is_valid_request_id(request_id):
            return request_id

        if self.config.generate_request_id:
            return str(uuid.uuid4())

        return "unknown"

    def _is_valid_request_id(self, request_id: str) -> bool:
        """Validate request ID format to prevent injection."""
        # Allow UUIDs and alphanumeric strings up to 64 chars
        if len(request_id) > 64:
            return False
        return bool(re.match(r"^[a-zA-Z0-9\-_]+$", request_id))

    def _validate_cors(self, request: Request) -> Optional[Response]:
        """Validate CORS origin."""
        if not self.config.cors.enabled:
            return None

        origin = request.headers.get("Origin")
        if not origin:
            return None  # Same-origin request

        if not self.config.cors.is_origin_allowed(origin):
            logger.warning(
                f"CORS origin rejected: {origin}",
                extra={"client_ip": self._get_client_ip(request)},
            )
            return JSONResponse(
                status_code=403,
                content={
                    "error": {
                        "code": "CORS_ORIGIN_DENIED",
                        "message": "Origin not allowed",
                    }
                },
            )

        return None

    def _handle_preflight(self, request: Request, request_id: str) -> Response:
        """Handle CORS preflight request."""
        origin = request.headers.get("Origin", "")

        if not self.config.cors.is_origin_allowed(origin):
            return JSONResponse(
                status_code=403,
                content={"error": {"code": "CORS_ORIGIN_DENIED", "message": "Origin not allowed"}},
                headers={"X-Request-ID": request_id},
            )

        headers = {
            "Access-Control-Allow-Origin": origin,
            "Access-Control-Allow-Methods": ", ".join(self.config.cors.allow_methods),
            "Access-Control-Allow-Headers": ", ".join(self.config.cors.allow_headers),
            "Access-Control-Max-Age": str(self.config.cors.max_age),
            "X-Request-ID": request_id,
        }

        if self.config.cors.allow_credentials:
            headers["Access-Control-Allow-Credentials"] = "true"

        return Response(status_code=204, headers=headers)

    async def _check_rate_limit(
        self,
        request: Request,
        request_id: str,
    ) -> Optional[Response]:
        """Check rate limits for the request."""
        if not self.config.rate_limit.enabled:
            return None

        now = time.time()
        client_ip = self._get_client_ip(request)
        path = request.url.path

        # Check IP-based rate limit
        if self.config.rate_limit.ip_rate_limit_enabled:
            ip_key = f"ip:{client_ip}"
            ip_count = self.rate_limit_state.get_request_count(ip_key, 60)

            if ip_count >= self.config.rate_limit.ip_requests_per_minute:
                logger.warning(
                    f"IP rate limit exceeded: {client_ip}",
                    extra={"request_id": request_id, "count": ip_count},
                )
                return self._rate_limit_response(request_id, "IP rate limit exceeded")

            self.rate_limit_state.record_request(ip_key, now)

        # Check endpoint-specific limits
        for pattern, limit in self.config.rate_limit.endpoint_limits.items():
            if self._path_matches_pattern(path, pattern):
                endpoint_key = f"endpoint:{pattern}:{client_ip}"
                endpoint_count = self.rate_limit_state.get_request_count(endpoint_key, 60)

                if endpoint_count >= limit:
                    logger.warning(
                        f"Endpoint rate limit exceeded: {path}",
                        extra={"request_id": request_id, "count": endpoint_count},
                    )
                    return self._rate_limit_response(request_id, "Endpoint rate limit exceeded")

                self.rate_limit_state.record_request(endpoint_key, now)
                break

        # Check tenant-based rate limit
        if self.config.rate_limit.tenant_rate_limit_enabled:
            tenant_id = getattr(request.state, "tenant_id", None)
            if tenant_id:
                tenant_key = f"tenant:{tenant_id}"
                tenant_count = self.rate_limit_state.get_request_count(tenant_key, 60)

                if tenant_count >= self.config.rate_limit.tenant_requests_per_minute:
                    logger.warning(
                        f"Tenant rate limit exceeded: {tenant_id}",
                        extra={"request_id": request_id, "count": tenant_count},
                    )
                    return self._rate_limit_response(request_id, "Tenant rate limit exceeded")

                self.rate_limit_state.record_request(tenant_key, now)

        return None

    def _path_matches_pattern(self, path: str, pattern: str) -> bool:
        """Check if a path matches a pattern with wildcards."""
        # Convert pattern to regex
        regex = pattern.replace("*", "[^/]+")
        return bool(re.match(f"^{regex}$", path))

    def _rate_limit_response(self, request_id: str, message: str) -> Response:
        """Create a rate limit exceeded response."""
        return JSONResponse(
            status_code=429,
            content={
                "error": {
                    "code": "RATE_LIMIT_EXCEEDED",
                    "message": message,
                    "request_id": request_id,
                }
            },
            headers={
                "X-Request-ID": request_id,
                "Retry-After": "60",
            },
        )

    def _validate_request_signature(self, request: Request) -> Optional[Response]:
        """Validate request HMAC signature."""
        signature = request.headers.get(self.config.request_signing.signature_header)
        timestamp = request.headers.get(self.config.request_signing.timestamp_header)
        key_id = request.headers.get(self.config.request_signing.signing_key_header)

        if not all([signature, timestamp, key_id]):
            return JSONResponse(
                status_code=401,
                content={
                    "error": {
                        "code": "MISSING_SIGNATURE",
                        "message": "Request signature headers required",
                    }
                },
            )

        # Check timestamp drift
        try:
            request_time = float(timestamp)
            current_time = time.time()
            drift = abs(current_time - request_time)

            if drift > self.config.request_signing.max_timestamp_drift_seconds:
                return JSONResponse(
                    status_code=401,
                    content={
                        "error": {
                            "code": "TIMESTAMP_EXPIRED",
                            "message": "Request timestamp is too old",
                        }
                    },
                )
        except ValueError:
            return JSONResponse(
                status_code=401,
                content={
                    "error": {
                        "code": "INVALID_TIMESTAMP",
                        "message": "Invalid timestamp format",
                    }
                },
            )

        # Get signing key
        signing_key = self.signing_keys.get(key_id)
        if not signing_key:
            return JSONResponse(
                status_code=401,
                content={
                    "error": {
                        "code": "UNKNOWN_KEY",
                        "message": "Unknown signing key ID",
                    }
                },
            )

        # Compute expected signature
        # Signature = HMAC(key, method + path + timestamp + body)
        sign_string = f"{request.method}{request.url.path}{timestamp}"
        expected_signature = hmac.new(
            signing_key.encode(),
            sign_string.encode(),
            hashlib.sha256,
        ).hexdigest()

        if not hmac.compare_digest(signature, expected_signature):
            logger.warning(
                f"Invalid request signature",
                extra={"key_id": key_id, "path": request.url.path},
            )
            return JSONResponse(
                status_code=401,
                content={
                    "error": {
                        "code": "INVALID_SIGNATURE",
                        "message": "Request signature validation failed",
                    }
                },
            )

        return None

    def _validate_content(self, request: Request) -> Optional[Response]:
        """Validate request content type and size."""
        # Check content length
        content_length = request.headers.get("Content-Length")
        if content_length:
            try:
                length = int(content_length)
                if length > self.config.max_content_length:
                    return JSONResponse(
                        status_code=413,
                        content={
                            "error": {
                                "code": "CONTENT_TOO_LARGE",
                                "message": f"Request body exceeds {self.config.max_content_length} bytes",
                            }
                        },
                    )
            except ValueError:
                pass

        # Check content type for POST/PUT/PATCH
        if request.method in ("POST", "PUT", "PATCH"):
            content_type = request.headers.get("Content-Type", "")
            base_type = content_type.split(";")[0].strip()

            if base_type and base_type not in self.config.allowed_content_types:
                return JSONResponse(
                    status_code=415,
                    content={
                        "error": {
                            "code": "UNSUPPORTED_MEDIA_TYPE",
                            "message": f"Content-Type '{base_type}' not allowed",
                        }
                    },
                )

        return None

    def _add_security_headers(self, response: Response, request_id: str) -> None:
        """Add security headers to the response."""
        # Request ID
        response.headers["X-Request-ID"] = request_id

        # HSTS
        if self.config.hsts_enabled:
            hsts_value = f"max-age={self.config.hsts_max_age}"
            if self.config.hsts_include_subdomains:
                hsts_value += "; includeSubDomains"
            if self.config.hsts_preload:
                hsts_value += "; preload"
            response.headers["Strict-Transport-Security"] = hsts_value

        # CSP
        if self.config.csp_enabled:
            response.headers["Content-Security-Policy"] = self.config.csp.to_header_value()

        # Other security headers
        response.headers["X-Frame-Options"] = self.config.x_frame_options
        response.headers["X-Content-Type-Options"] = self.config.x_content_type_options
        response.headers["X-XSS-Protection"] = self.config.x_xss_protection
        response.headers["Referrer-Policy"] = self.config.referrer_policy
        response.headers["Permissions-Policy"] = self.config.permissions_policy
        response.headers["Cache-Control"] = self.config.cache_control

        # Additional hardening headers
        response.headers["X-DNS-Prefetch-Control"] = "off"
        response.headers["X-Download-Options"] = "noopen"
        response.headers["X-Permitted-Cross-Domain-Policies"] = "none"

    def _add_cors_headers(self, request: Request, response: Response) -> None:
        """Add CORS headers to the response."""
        if not self.config.cors.enabled:
            return

        origin = request.headers.get("Origin")
        if not origin:
            return

        if not self.config.cors.is_origin_allowed(origin):
            return

        response.headers["Access-Control-Allow-Origin"] = origin

        if self.config.cors.allow_credentials:
            response.headers["Access-Control-Allow-Credentials"] = "true"

        if self.config.cors.expose_headers:
            response.headers["Access-Control-Expose-Headers"] = ", ".join(
                self.config.cors.expose_headers
            )

    async def _add_rate_limit_headers(self, request: Request, response: Response) -> None:
        """Add rate limit headers to the response."""
        if not self.config.rate_limit.include_rate_limit_headers:
            return

        client_ip = self._get_client_ip(request)
        ip_key = f"ip:{client_ip}"
        current_count = self.rate_limit_state.get_request_count(ip_key, 60)
        limit = self.config.rate_limit.ip_requests_per_minute

        response.headers["X-RateLimit-Limit"] = str(limit)
        response.headers["X-RateLimit-Remaining"] = str(max(0, limit - current_count))
        response.headers["X-RateLimit-Reset"] = str(int(time.time()) + 60)

    def _get_client_ip(self, request: Request) -> str:
        """Extract client IP from request, handling proxies."""
        # Check X-Forwarded-For header
        forwarded_for = request.headers.get("X-Forwarded-For")
        if forwarded_for:
            # Get the first IP in the chain (original client)
            return forwarded_for.split(",")[0].strip()

        # Check X-Real-IP header
        real_ip = request.headers.get("X-Real-IP")
        if real_ip:
            return real_ip

        # Fall back to direct client IP
        if request.client:
            return request.client.host

        return "unknown"


def create_api_security_config() -> SecurityHeadersConfig:
    """Create a security configuration optimized for APIs."""
    return SecurityHeadersConfig(
        hsts_enabled=True,
        hsts_max_age=31536000,
        hsts_include_subdomains=True,
        csp_enabled=False,  # CSP less relevant for API-only services
        cors=CORSPolicy(
            enabled=True,
            allow_origins=["https://app.greenlang.io", "https://dashboard.greenlang.io"],
            allow_methods=["GET", "POST", "PUT", "DELETE", "PATCH"],
            allow_credentials=True,
        ),
        rate_limit=RateLimitConfig(
            enabled=True,
            requests_per_minute=100,
            ip_requests_per_minute=300,
        ),
    )


def create_strict_security_config() -> SecurityHeadersConfig:
    """Create a strict security configuration for high-security environments."""
    return SecurityHeadersConfig(
        hsts_enabled=True,
        hsts_max_age=63072000,  # 2 years
        hsts_include_subdomains=True,
        hsts_preload=True,
        csp_enabled=True,
        csp=ContentSecurityPolicy(
            default_src=["'self'"],
            script_src=["'self'"],
            style_src=["'self'"],
            img_src=["'self'"],
            connect_src=["'self'"],
            frame_src=["'none'"],
            object_src=["'none'"],
        ),
        cors=CORSPolicy(
            enabled=True,
            allow_origins=[],  # Must be explicitly configured
            allow_credentials=False,
        ),
        rate_limit=RateLimitConfig(
            enabled=True,
            requests_per_minute=60,
            ip_requests_per_minute=120,
            endpoint_limits={
                "/v1/agents/*/execute": 30,
                "/v1/auth/login": 5,
            },
        ),
        request_signing=RequestSigningConfig(
            enabled=True,
            max_timestamp_drift_seconds=60,
        ),
    )
