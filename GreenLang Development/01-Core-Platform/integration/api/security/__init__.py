"""
GreenLang API Security Module

This module provides comprehensive security features for the GreenLang API
including CSRF protection, rate limiting, and security headers.

Example:
    >>> from greenlang.api.security import setup_security
    >>> app = FastAPI()
    >>> setup_security(app)
"""

from typing import Optional

from fastapi import FastAPI

from .csrf import CSRFProtect, CSRFConfig, generate_csrf_token, validate_csrf_token
from .rate_limiting import RateLimiter, RateLimitConfig, RateLimitMiddleware
from .headers import SecurityHeadersMiddleware, SecurityHeadersConfig, SECURITY_PRESETS

__all__ = [
    # CSRF
    "CSRFProtect",
    "CSRFConfig",
    "generate_csrf_token",
    "validate_csrf_token",
    # Rate Limiting
    "RateLimiter",
    "RateLimitConfig",
    "RateLimitMiddleware",
    # Security Headers
    "SecurityHeadersMiddleware",
    "SecurityHeadersConfig",
    "SECURITY_PRESETS",
    # Setup functions
    "setup_security",
    "create_security_config"
]


def create_security_config(
    # CSRF settings
    csrf_secret_key: str = None,
    csrf_token_expiry: int = 3600,
    csrf_exempt_paths: list = None,
    # Rate limiting settings
    redis_url: str = None,
    default_rate_limit: str = "100/minute",
    enable_distributed_rate_limit: bool = True,
    # Security headers settings
    security_preset: str = "balanced",
    enable_hsts: bool = True,
    enable_csp: bool = True,
    csp_report_uri: str = None
) -> dict:
    """
    Create security configuration for GreenLang API.

    Args:
        csrf_secret_key: Secret key for CSRF token signing
        csrf_token_expiry: CSRF token expiry in seconds
        csrf_exempt_paths: Paths exempt from CSRF protection
        redis_url: Redis URL for distributed rate limiting
        default_rate_limit: Default rate limit
        enable_distributed_rate_limit: Enable Redis-backed rate limiting
        security_preset: Security headers preset ('strict', 'balanced', 'relaxed')
        enable_hsts: Enable HSTS header
        enable_csp: Enable CSP header
        csp_report_uri: URI for CSP violation reports

    Returns:
        Dictionary with security configurations

    Example:
        >>> config = create_security_config(
        >>>     csrf_secret_key="secret",
        >>>     redis_url="redis://localhost:6379",
        >>>     security_preset="strict"
        >>> )
    """
    import os

    # Default CSRF secret if not provided
    if not csrf_secret_key:
        csrf_secret_key = os.environ.get("CSRF_SECRET_KEY", "change-this-in-production")

    # CSRF configuration
    csrf_config = CSRFConfig(
        secret_key=csrf_secret_key,
        token_expiry_seconds=csrf_token_expiry,
        exempt_paths=set(csrf_exempt_paths or [
            "/health",
            "/metrics",
            "/docs",
            "/redoc",
            "/openapi.json"
        ])
    )

    # Rate limiting configuration
    rate_limit_config = RateLimitConfig(
        redis_url=redis_url or os.environ.get("REDIS_URL"),
        default_limit=100,
        default_period="minute",
        enable_distributed=enable_distributed_rate_limit and bool(redis_url),
        endpoint_limits={
            "/api/auth/login": "5/minute",
            "/api/auth/register": "3/minute",
            "/api/auth/reset-password": "3/hour",
            "/api/calculate": "20/minute",
            "/api/report": "10/minute"
        }
    )

    # Parse default rate limit
    if default_rate_limit:
        parts = default_rate_limit.split("/")
        if len(parts) == 2:
            rate_limit_config.default_limit = int(parts[0])
            rate_limit_config.default_period = parts[1]

    # Security headers configuration
    if security_preset in SECURITY_PRESETS:
        headers_config = SECURITY_PRESETS[security_preset]
    else:
        headers_config = SecurityHeadersConfig()

    # Override specific settings
    headers_config.enable_hsts = enable_hsts
    headers_config.enable_csp = enable_csp
    if csp_report_uri:
        headers_config.csp_report_uri = csp_report_uri

    return {
        "csrf": csrf_config,
        "rate_limit": rate_limit_config,
        "headers": headers_config
    }


def setup_security(
    app: FastAPI,
    csrf_config: Optional[CSRFConfig] = None,
    rate_limit_config: Optional[RateLimitConfig] = None,
    headers_config: Optional[SecurityHeadersConfig] = None,
    enable_csrf: bool = True,
    enable_rate_limit: bool = True,
    enable_security_headers: bool = True
):
    """
    Setup all security middleware for FastAPI application.

    Args:
        app: FastAPI application instance
        csrf_config: CSRF protection configuration
        rate_limit_config: Rate limiting configuration
        headers_config: Security headers configuration
        enable_csrf: Enable CSRF protection
        enable_rate_limit: Enable rate limiting
        enable_security_headers: Enable security headers

    Example:
        >>> from fastapi import FastAPI
        >>> from greenlang.api.security import setup_security
        >>>
        >>> app = FastAPI(title="GreenLang API")
        >>>
        >>> # Setup with defaults
        >>> setup_security(app)
        >>>
        >>> # Or with custom configuration
        >>> config = create_security_config(
        >>>     csrf_secret_key="your-secret",
        >>>     redis_url="redis://localhost:6379",
        >>>     security_preset="strict"
        >>> )
        >>> setup_security(
        >>>     app,
        >>>     csrf_config=config["csrf"],
        >>>     rate_limit_config=config["rate_limit"],
        >>>     headers_config=config["headers"]
        >>> )
    """
    import logging
    logger = logging.getLogger(__name__)

    # Order matters: Add middleware in reverse order of execution
    # (last added is executed first)

    # 1. Security Headers (should be last to add headers to all responses)
    if enable_security_headers:
        if not headers_config:
            headers_config = SECURITY_PRESETS["balanced"]

        app.add_middleware(
            SecurityHeadersMiddleware,
            config=headers_config
        )
        logger.info("Security headers middleware enabled")

    # 2. Rate Limiting
    if enable_rate_limit:
        if not rate_limit_config:
            rate_limit_config = RateLimitConfig()

        app.add_middleware(
            RateLimitMiddleware,
            config=rate_limit_config
        )
        logger.info("Rate limiting middleware enabled")

    # 3. CSRF Protection (for state-changing operations)
    if enable_csrf:
        if not csrf_config:
            import os
            csrf_config = CSRFConfig(
                secret_key=os.environ.get("CSRF_SECRET_KEY", "change-this-in-production")
            )

        csrf_protect = CSRFProtect(csrf_config)
        app.add_middleware(csrf_protect)
        logger.info("CSRF protection middleware enabled")

    logger.info("All security middleware configured successfully")