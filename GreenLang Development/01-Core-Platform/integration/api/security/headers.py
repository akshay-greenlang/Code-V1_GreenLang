"""
Security Headers for GreenLang API

This module implements security headers middleware to protect
against common web vulnerabilities.

Example:
    >>> from greenlang.api.security.headers import SecurityHeadersMiddleware
    >>> app.add_middleware(SecurityHeadersMiddleware)
"""

from typing import Optional, Dict, Any, List
from enum import Enum

from fastapi import Request, Response
from pydantic import BaseModel, Field, validator
import logging

logger = logging.getLogger(__name__)


class FrameOptions(str, Enum):
    """X-Frame-Options values."""
    DENY = "DENY"
    SAMEORIGIN = "SAMEORIGIN"
    ALLOW_FROM = "ALLOW-FROM"


class ReferrerPolicy(str, Enum):
    """Referrer-Policy values."""
    NO_REFERRER = "no-referrer"
    NO_REFERRER_WHEN_DOWNGRADE = "no-referrer-when-downgrade"
    ORIGIN = "origin"
    ORIGIN_WHEN_CROSS_ORIGIN = "origin-when-cross-origin"
    SAME_ORIGIN = "same-origin"
    STRICT_ORIGIN = "strict-origin"
    STRICT_ORIGIN_WHEN_CROSS_ORIGIN = "strict-origin-when-cross-origin"
    UNSAFE_URL = "unsafe-url"


class ContentSecurityPolicyDirective(BaseModel):
    """Content Security Policy directive configuration."""

    default_src: List[str] = Field(default=["'self'"], description="Default source")
    script_src: List[str] = Field(default=["'self'"], description="Script sources")
    style_src: List[str] = Field(default=["'self'"], description="Style sources")
    img_src: List[str] = Field(default=["'self'", "data:", "https:"], description="Image sources")
    font_src: List[str] = Field(default=["'self'"], description="Font sources")
    connect_src: List[str] = Field(default=["'self'"], description="Connection sources")
    media_src: List[str] = Field(default=["'self'"], description="Media sources")
    object_src: List[str] = Field(default=["'none'"], description="Object sources")
    frame_src: List[str] = Field(default=["'none'"], description="Frame sources")
    frame_ancestors: List[str] = Field(default=["'none'"], description="Frame ancestors")
    base_uri: List[str] = Field(default=["'self'"], description="Base URI")
    form_action: List[str] = Field(default=["'self'"], description="Form action")
    upgrade_insecure_requests: bool = Field(True, description="Upgrade insecure requests")
    block_all_mixed_content: bool = Field(True, description="Block mixed content")


class SecurityHeadersConfig(BaseModel):
    """Configuration for security headers."""

    # X-Content-Type-Options
    x_content_type_options: str = Field(
        "nosniff",
        description="Prevent MIME type sniffing"
    )

    # X-Frame-Options
    x_frame_options: FrameOptions = Field(
        FrameOptions.DENY,
        description="Prevent clickjacking attacks"
    )
    x_frame_options_allow_from: Optional[str] = Field(
        None,
        description="Domain to allow framing from (if using ALLOW-FROM)"
    )

    # X-XSS-Protection
    x_xss_protection: str = Field(
        "1; mode=block",
        description="Enable XSS protection"
    )

    # Strict-Transport-Security
    strict_transport_security: str = Field(
        "max-age=31536000; includeSubDomains; preload",
        description="Force HTTPS connections"
    )
    enable_hsts: bool = Field(
        True,
        description="Enable HSTS header"
    )

    # Content-Security-Policy
    content_security_policy: Optional[ContentSecurityPolicyDirective] = Field(
        default_factory=ContentSecurityPolicyDirective,
        description="Content Security Policy configuration"
    )
    enable_csp: bool = Field(
        True,
        description="Enable CSP header"
    )
    csp_report_only: bool = Field(
        False,
        description="Use CSP in report-only mode"
    )
    csp_report_uri: Optional[str] = Field(
        None,
        description="URI for CSP violation reports"
    )

    # Referrer-Policy
    referrer_policy: ReferrerPolicy = Field(
        ReferrerPolicy.STRICT_ORIGIN_WHEN_CROSS_ORIGIN,
        description="Control referrer information"
    )

    # Permissions-Policy (formerly Feature-Policy)
    permissions_policy: Dict[str, List[str]] = Field(
        default_factory=lambda: {
            "geolocation": ["'none'"],
            "camera": ["'none'"],
            "microphone": ["'none'"],
            "payment": ["'none'"],
            "usb": ["'none'"],
            "magnetometer": ["'none'"],
            "gyroscope": ["'none'"],
            "accelerometer": ["'none'"]
        },
        description="Control browser features"
    )
    enable_permissions_policy: bool = Field(
        True,
        description="Enable Permissions-Policy header"
    )

    # X-Permitted-Cross-Domain-Policies
    x_permitted_cross_domain_policies: str = Field(
        "none",
        description="Control cross-domain content handling"
    )

    # Additional security headers
    x_download_options: str = Field(
        "noopen",
        description="Prevent IE from opening downloads"
    )

    # Cache control for sensitive content
    cache_control: str = Field(
        "no-store, no-cache, must-revalidate, proxy-revalidate",
        description="Cache control for sensitive responses"
    )
    pragma: str = Field(
        "no-cache",
        description="HTTP/1.0 cache control"
    )
    expires: str = Field(
        "0",
        description="Expire immediately"
    )

    # Custom headers
    custom_headers: Dict[str, str] = Field(
        default_factory=dict,
        description="Additional custom headers"
    )

    # Exclude patterns
    exclude_paths: List[str] = Field(
        default_factory=lambda: ["/health", "/metrics", "/docs", "/redoc", "/openapi.json"],
        description="Paths to exclude from security headers"
    )


class SecurityHeadersMiddleware:
    """
    Security headers middleware for FastAPI.

    This middleware adds security headers to all responses to protect
    against common web vulnerabilities.

    Attributes:
        config: Security headers configuration

    Example:
        >>> from fastapi import FastAPI
        >>> app = FastAPI()
        >>> app.add_middleware(
        >>>     SecurityHeadersMiddleware,
        >>>     config=SecurityHeadersConfig()
        >>> )
    """

    def __init__(
        self,
        app,
        config: Optional[SecurityHeadersConfig] = None
    ):
        """
        Initialize security headers middleware.

        Args:
            app: FastAPI application
            config: Optional security headers configuration
        """
        self.app = app
        self.config = config or SecurityHeadersConfig()

    def _build_csp_header(self) -> str:
        """
        Build Content-Security-Policy header value.

        Returns:
            CSP header string
        """
        if not self.config.content_security_policy:
            return ""

        csp = self.config.content_security_policy
        directives = []

        # Add each directive
        if csp.default_src:
            directives.append(f"default-src {' '.join(csp.default_src)}")
        if csp.script_src:
            directives.append(f"script-src {' '.join(csp.script_src)}")
        if csp.style_src:
            directives.append(f"style-src {' '.join(csp.style_src)}")
        if csp.img_src:
            directives.append(f"img-src {' '.join(csp.img_src)}")
        if csp.font_src:
            directives.append(f"font-src {' '.join(csp.font_src)}")
        if csp.connect_src:
            directives.append(f"connect-src {' '.join(csp.connect_src)}")
        if csp.media_src:
            directives.append(f"media-src {' '.join(csp.media_src)}")
        if csp.object_src:
            directives.append(f"object-src {' '.join(csp.object_src)}")
        if csp.frame_src:
            directives.append(f"frame-src {' '.join(csp.frame_src)}")
        if csp.frame_ancestors:
            directives.append(f"frame-ancestors {' '.join(csp.frame_ancestors)}")
        if csp.base_uri:
            directives.append(f"base-uri {' '.join(csp.base_uri)}")
        if csp.form_action:
            directives.append(f"form-action {' '.join(csp.form_action)}")

        # Add boolean directives
        if csp.upgrade_insecure_requests:
            directives.append("upgrade-insecure-requests")
        if csp.block_all_mixed_content:
            directives.append("block-all-mixed-content")

        # Add report-uri if configured
        if self.config.csp_report_uri:
            directives.append(f"report-uri {self.config.csp_report_uri}")

        return "; ".join(directives)

    def _build_permissions_policy_header(self) -> str:
        """
        Build Permissions-Policy header value.

        Returns:
            Permissions-Policy header string
        """
        if not self.config.permissions_policy:
            return ""

        policies = []
        for feature, allowlist in self.config.permissions_policy.items():
            if allowlist:
                policies.append(f'{feature}=({" ".join(allowlist)})')
            else:
                policies.append(f'{feature}=()')

        return ", ".join(policies)

    def _should_apply_headers(self, request: Request) -> bool:
        """
        Check if security headers should be applied to request.

        Args:
            request: FastAPI request object

        Returns:
            True if headers should be applied
        """
        path = request.url.path

        # Check exclude patterns
        for exclude_path in self.config.exclude_paths:
            if path.startswith(exclude_path):
                return False

        return True

    async def __call__(self, request: Request, call_next):
        """
        Add security headers to response.

        Args:
            request: FastAPI request object
            call_next: Next middleware in chain

        Returns:
            Response with security headers
        """
        # Process request
        response = await call_next(request)

        # Check if headers should be applied
        if not self._should_apply_headers(request):
            return response

        # Add security headers

        # Content type options
        if self.config.x_content_type_options:
            response.headers["X-Content-Type-Options"] = self.config.x_content_type_options

        # Frame options
        if self.config.x_frame_options:
            if self.config.x_frame_options == FrameOptions.ALLOW_FROM:
                if self.config.x_frame_options_allow_from:
                    response.headers["X-Frame-Options"] = (
                        f"ALLOW-FROM {self.config.x_frame_options_allow_from}"
                    )
            else:
                response.headers["X-Frame-Options"] = self.config.x_frame_options.value

        # XSS protection
        if self.config.x_xss_protection:
            response.headers["X-XSS-Protection"] = self.config.x_xss_protection

        # HSTS (only on HTTPS)
        if self.config.enable_hsts and request.url.scheme == "https":
            response.headers["Strict-Transport-Security"] = (
                self.config.strict_transport_security
            )

        # Content Security Policy
        if self.config.enable_csp:
            csp_value = self._build_csp_header()
            if csp_value:
                header_name = (
                    "Content-Security-Policy-Report-Only"
                    if self.config.csp_report_only
                    else "Content-Security-Policy"
                )
                response.headers[header_name] = csp_value

        # Referrer Policy
        if self.config.referrer_policy:
            response.headers["Referrer-Policy"] = self.config.referrer_policy.value

        # Permissions Policy
        if self.config.enable_permissions_policy:
            permissions_value = self._build_permissions_policy_header()
            if permissions_value:
                response.headers["Permissions-Policy"] = permissions_value

        # Cross domain policies
        if self.config.x_permitted_cross_domain_policies:
            response.headers["X-Permitted-Cross-Domain-Policies"] = (
                self.config.x_permitted_cross_domain_policies
            )

        # Download options (IE specific)
        if self.config.x_download_options:
            response.headers["X-Download-Options"] = self.config.x_download_options

        # Cache control for sensitive content
        # Only apply to non-static content
        content_type = response.headers.get("content-type", "")
        if "text/html" in content_type or "application/json" in content_type:
            response.headers["Cache-Control"] = self.config.cache_control
            response.headers["Pragma"] = self.config.pragma
            response.headers["Expires"] = self.config.expires

        # Custom headers
        for header_name, header_value in self.config.custom_headers.items():
            response.headers[header_name] = header_value

        # Log security headers applied
        logger.debug(f"Security headers applied to {request.url.path}")

        return response


def create_secure_headers(
    enable_hsts: bool = True,
    enable_csp: bool = True,
    frame_options: FrameOptions = FrameOptions.DENY,
    custom_csp: Optional[Dict[str, List[str]]] = None
) -> Dict[str, str]:
    """
    Create a dictionary of security headers.

    Args:
        enable_hsts: Enable HSTS header
        enable_csp: Enable CSP header
        frame_options: X-Frame-Options value
        custom_csp: Custom CSP directives

    Returns:
        Dictionary of security headers

    Example:
        >>> headers = create_secure_headers()
        >>> response.headers.update(headers)
    """
    headers = {
        "X-Content-Type-Options": "nosniff",
        "X-Frame-Options": frame_options.value,
        "X-XSS-Protection": "1; mode=block",
        "Referrer-Policy": "strict-origin-when-cross-origin",
        "X-Permitted-Cross-Domain-Policies": "none"
    }

    if enable_hsts:
        headers["Strict-Transport-Security"] = (
            "max-age=31536000; includeSubDomains; preload"
        )

    if enable_csp:
        csp_config = ContentSecurityPolicyDirective()
        if custom_csp:
            for key, value in custom_csp.items():
                if hasattr(csp_config, key):
                    setattr(csp_config, key, value)

        # Build basic CSP
        csp_parts = [
            f"default-src {' '.join(csp_config.default_src)}",
            f"script-src {' '.join(csp_config.script_src)}",
            f"style-src {' '.join(csp_config.style_src)}",
            f"img-src {' '.join(csp_config.img_src)}",
            f"frame-ancestors {' '.join(csp_config.frame_ancestors)}",
            "upgrade-insecure-requests"
        ]
        headers["Content-Security-Policy"] = "; ".join(csp_parts)

    return headers


# Presets for common security configurations
SECURITY_PRESETS = {
    "strict": SecurityHeadersConfig(
        x_frame_options=FrameOptions.DENY,
        enable_hsts=True,
        enable_csp=True,
        content_security_policy=ContentSecurityPolicyDirective(
            default_src=["'self'"],
            script_src=["'self'"],
            style_src=["'self'"],
            img_src=["'self'"],
            frame_ancestors=["'none'"]
        ),
        referrer_policy=ReferrerPolicy.NO_REFERRER
    ),
    "balanced": SecurityHeadersConfig(
        x_frame_options=FrameOptions.SAMEORIGIN,
        enable_hsts=True,
        enable_csp=True,
        content_security_policy=ContentSecurityPolicyDirective(
            default_src=["'self'"],
            script_src=["'self'", "'unsafe-inline'"],
            style_src=["'self'", "'unsafe-inline'"],
            img_src=["'self'", "data:", "https:"],
            frame_ancestors=["'self'"]
        ),
        referrer_policy=ReferrerPolicy.STRICT_ORIGIN_WHEN_CROSS_ORIGIN
    ),
    "relaxed": SecurityHeadersConfig(
        x_frame_options=FrameOptions.SAMEORIGIN,
        enable_hsts=True,
        enable_csp=False,
        referrer_policy=ReferrerPolicy.ORIGIN_WHEN_CROSS_ORIGIN
    )
}