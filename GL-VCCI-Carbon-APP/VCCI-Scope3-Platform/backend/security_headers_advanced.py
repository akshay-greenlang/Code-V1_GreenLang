"""
Advanced Security Headers Middleware
GL-VCCI Scope 3 Platform

Implements comprehensive security headers including:
- HSTS (HTTP Strict Transport Security)
- CSP (Content Security Policy) with violation reporting
- X-Frame-Options, X-Content-Type-Options
- Expect-CT (Certificate Transparency)
- NEL (Network Error Logging)
- Feature-Policy / Permissions-Policy
- Referrer-Policy
- Report-URI for security violation reporting

Version: 1.0.0
Security Enhancement: 2025-11-09
"""

import os
import logging
from typing import Optional, Callable
from datetime import datetime

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp

logger = logging.getLogger(__name__)

# Configuration
ENVIRONMENT = os.getenv("ENVIRONMENT", "dev")
IS_PRODUCTION = ENVIRONMENT == "production"

# Security headers configuration
HSTS_MAX_AGE = int(os.getenv("HSTS_MAX_AGE", "31536000"))  # 1 year
HSTS_INCLUDE_SUBDOMAINS = os.getenv("HSTS_INCLUDE_SUBDOMAINS", "true").lower() == "true"
HSTS_PRELOAD = os.getenv("HSTS_PRELOAD", "true").lower() == "true"

# CSP configuration
CSP_ENABLED = os.getenv("CSP_ENABLED", "true").lower() == "true"
CSP_REPORT_URI = os.getenv("CSP_REPORT_URI", "/api/security/csp-report")
CSP_REPORT_ONLY = os.getenv("CSP_REPORT_ONLY", "false").lower() == "true"

# Expect-CT configuration
EXPECT_CT_ENABLED = os.getenv("EXPECT_CT_ENABLED", "true").lower() == "true"
EXPECT_CT_MAX_AGE = int(os.getenv("EXPECT_CT_MAX_AGE", "86400"))  # 24 hours
EXPECT_CT_ENFORCE = os.getenv("EXPECT_CT_ENFORCE", "false").lower() == "true"
EXPECT_CT_REPORT_URI = os.getenv("EXPECT_CT_REPORT_URI", "/api/security/ct-report")

# NEL configuration
NEL_ENABLED = os.getenv("NEL_ENABLED", "true").lower() == "true"
NEL_REPORT_TO = os.getenv("NEL_REPORT_TO", "default")
NEL_MAX_AGE = int(os.getenv("NEL_MAX_AGE", "2592000"))  # 30 days


class SecurityHeadersConfig:
    """Configuration for security headers."""

    def __init__(
        self,
        hsts_max_age: int = HSTS_MAX_AGE,
        hsts_include_subdomains: bool = HSTS_INCLUDE_SUBDOMAINS,
        hsts_preload: bool = HSTS_PRELOAD,
        csp_enabled: bool = CSP_ENABLED,
        csp_report_uri: Optional[str] = CSP_REPORT_URI,
        csp_report_only: bool = CSP_REPORT_ONLY,
        expect_ct_enabled: bool = EXPECT_CT_ENABLED,
        expect_ct_max_age: int = EXPECT_CT_MAX_AGE,
        expect_ct_enforce: bool = EXPECT_CT_ENFORCE,
        expect_ct_report_uri: Optional[str] = EXPECT_CT_REPORT_URI,
        nel_enabled: bool = NEL_ENABLED,
        nel_report_to: str = NEL_REPORT_TO,
        nel_max_age: int = NEL_MAX_AGE,
    ):
        self.hsts_max_age = hsts_max_age
        self.hsts_include_subdomains = hsts_include_subdomains
        self.hsts_preload = hsts_preload
        self.csp_enabled = csp_enabled
        self.csp_report_uri = csp_report_uri
        self.csp_report_only = csp_report_only
        self.expect_ct_enabled = expect_ct_enabled
        self.expect_ct_max_age = expect_ct_max_age
        self.expect_ct_enforce = expect_ct_enforce
        self.expect_ct_report_uri = expect_ct_report_uri
        self.nel_enabled = nel_enabled
        self.nel_report_to = nel_report_to
        self.nel_max_age = nel_max_age


def build_csp_header(
    report_uri: Optional[str] = None,
    report_only: bool = False,
) -> tuple[str, str]:
    """
    Build Content Security Policy header.

    Args:
        report_uri: URI for CSP violation reports
        report_only: If True, use Content-Security-Policy-Report-Only

    Returns:
        Tuple of (header_name, header_value)

    Example:
        >>> name, value = build_csp_header("/api/csp-report")
    """
    # CSP directives
    directives = [
        "default-src 'self'",  # Default: only same origin
        "script-src 'self' 'unsafe-inline' 'unsafe-eval' https://cdn.jsdelivr.net",  # Scripts
        "style-src 'self' 'unsafe-inline' https://cdn.jsdelivr.net https://fonts.googleapis.com",  # Styles
        "img-src 'self' data: https:",  # Images
        "font-src 'self' data: https://fonts.gstatic.com",  # Fonts
        "connect-src 'self' https://api.anthropic.com https://api.openai.com",  # API connections
        "frame-ancestors 'none'",  # Don't allow framing (alternative to X-Frame-Options)
        "base-uri 'self'",  # Restrict <base> tag
        "form-action 'self'",  # Restrict form submissions
        "upgrade-insecure-requests",  # Upgrade HTTP to HTTPS
    ]

    # Add report URI if provided
    if report_uri:
        directives.append(f"report-uri {report_uri}")

    csp_value = "; ".join(directives)

    # Choose header name based on report_only
    header_name = (
        "Content-Security-Policy-Report-Only" if report_only else "Content-Security-Policy"
    )

    return header_name, csp_value


def build_expect_ct_header(
    max_age: int = EXPECT_CT_MAX_AGE,
    enforce: bool = False,
    report_uri: Optional[str] = None,
) -> str:
    """
    Build Expect-CT header value.

    Args:
        max_age: Max age in seconds
        enforce: Whether to enforce CT requirements
        report_uri: URI for CT violation reports

    Returns:
        Header value

    Example:
        >>> value = build_expect_ct_header(86400, enforce=True, report_uri="/api/ct-report")
    """
    parts = [f"max-age={max_age}"]

    if enforce:
        parts.append("enforce")

    if report_uri:
        parts.append(f'report-uri="{report_uri}"')

    return ", ".join(parts)


def build_nel_header(
    report_to: str = "default",
    max_age: int = NEL_MAX_AGE,
    include_subdomains: bool = True,
) -> str:
    """
    Build Network Error Logging header value.

    Args:
        report_to: Report-To group name
        max_age: Max age in seconds
        include_subdomains: Whether to include subdomains

    Returns:
        Header value (JSON string)

    Example:
        >>> value = build_nel_header("default", 2592000)
    """
    nel_config = {
        "report_to": report_to,
        "max_age": max_age,
        "include_subdomains": include_subdomains,
    }

    import json

    return json.dumps(nel_config)


def build_report_to_header(
    group: str = "default",
    max_age: int = 86400,
    endpoints: Optional[list] = None,
) -> str:
    """
    Build Report-To header for NEL and CSP reporting.

    Args:
        group: Report group name
        max_age: Max age in seconds
        endpoints: List of reporting endpoints

    Returns:
        Header value (JSON string)

    Example:
        >>> value = build_report_to_header(
        ...     "default",
        ...     endpoints=[{"url": "https://example.com/reports"}]
        ... )
    """
    if endpoints is None:
        endpoints = [{"url": "/api/security/reports"}]

    report_config = {
        "group": group,
        "max_age": max_age,
        "endpoints": endpoints,
    }

    import json

    return json.dumps(report_config)


def build_permissions_policy_header() -> str:
    """
    Build Permissions-Policy header (formerly Feature-Policy).

    Restricts browser features that the application can use.

    Returns:
        Header value

    Example:
        >>> value = build_permissions_policy_header()
    """
    # Disable unnecessary features
    policies = [
        "geolocation=()",  # No geolocation
        "microphone=()",  # No microphone
        "camera=()",  # No camera
        "payment=()",  # No payment
        "usb=()",  # No USB
        "magnetometer=()",  # No magnetometer
        "gyroscope=()",  # No gyroscope
        "accelerometer=()",  # No accelerometer
        "ambient-light-sensor=()",  # No ambient light sensor
    ]

    return ", ".join(policies)


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """
    Middleware to add advanced security headers to all responses.

    Usage:
        ```python
        from fastapi import FastAPI
        app = FastAPI()

        # Add with default config
        app.add_middleware(SecurityHeadersMiddleware)

        # Or with custom config
        config = SecurityHeadersConfig(
            hsts_max_age=63072000,  # 2 years
            csp_enabled=True,
        )
        app.add_middleware(SecurityHeadersMiddleware, config=config)
        ```
    """

    def __init__(
        self,
        app: ASGIApp,
        config: Optional[SecurityHeadersConfig] = None,
    ):
        super().__init__(app)
        self.config = config or SecurityHeadersConfig()

        logger.info("Security headers middleware initialized")

    async def dispatch(
        self,
        request: Request,
        call_next: Callable,
    ) -> Response:
        """Add security headers to response."""
        response = await call_next(request)

        # HSTS (HTTP Strict Transport Security)
        # Only add on HTTPS or in production
        if IS_PRODUCTION or request.url.scheme == "https":
            hsts_value = f"max-age={self.config.hsts_max_age}"

            if self.config.hsts_include_subdomains:
                hsts_value += "; includeSubDomains"

            if self.config.hsts_preload:
                hsts_value += "; preload"

            response.headers["Strict-Transport-Security"] = hsts_value

        # Content Security Policy
        if self.config.csp_enabled:
            csp_name, csp_value = build_csp_header(
                self.config.csp_report_uri,
                self.config.csp_report_only,
            )
            response.headers[csp_name] = csp_value

        # X-Frame-Options (defense in depth with CSP frame-ancestors)
        response.headers["X-Frame-Options"] = "DENY"

        # X-Content-Type-Options (prevent MIME sniffing)
        response.headers["X-Content-Type-Options"] = "nosniff"

        # X-XSS-Protection (legacy, but still useful for older browsers)
        response.headers["X-XSS-Protection"] = "1; mode=block"

        # Referrer-Policy
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"

        # Permissions-Policy
        response.headers["Permissions-Policy"] = build_permissions_policy_header()

        # Expect-CT (Certificate Transparency)
        if self.config.expect_ct_enabled and IS_PRODUCTION:
            expect_ct_value = build_expect_ct_header(
                self.config.expect_ct_max_age,
                self.config.expect_ct_enforce,
                self.config.expect_ct_report_uri,
            )
            response.headers["Expect-CT"] = expect_ct_value

        # Network Error Logging (NEL)
        if self.config.nel_enabled and IS_PRODUCTION:
            nel_value = build_nel_header(
                self.config.nel_report_to,
                self.config.nel_max_age,
            )
            response.headers["NEL"] = nel_value

            # Report-To (required for NEL and CSP reporting)
            report_to_value = build_report_to_header(
                self.config.nel_report_to,
                self.config.nel_max_age,
            )
            response.headers["Report-To"] = report_to_value

        # Remove server header to not reveal server software
        if "Server" in response.headers:
            del response.headers["Server"]

        # Remove X-Powered-By if present
        if "X-Powered-By" in response.headers:
            del response.headers["X-Powered-By"]

        return response


# Subresource Integrity (SRI) helper
def generate_sri_hash(content: str, algorithm: str = "sha384") -> str:
    """
    Generate Subresource Integrity hash for a script or stylesheet.

    Args:
        content: Script or stylesheet content
        algorithm: Hash algorithm (sha256, sha384, or sha512)

    Returns:
        SRI hash string

    Example:
        >>> sri_hash = generate_sri_hash(script_content)
        >>> # Use in HTML: <script src="..." integrity="sha384-{hash}">
    """
    import hashlib
    import base64

    if algorithm == "sha256":
        hasher = hashlib.sha256()
    elif algorithm == "sha384":
        hasher = hashlib.sha384()
    elif algorithm == "sha512":
        hasher = hashlib.sha512()
    else:
        raise ValueError(f"Unsupported algorithm: {algorithm}")

    hasher.update(content.encode("utf-8"))
    hash_bytes = hasher.digest()
    hash_b64 = base64.b64encode(hash_bytes).decode("utf-8")

    return f"{algorithm}-{hash_b64}"


# CSP violation report handler
class CSPViolationReport:
    """CSP violation report data structure."""

    def __init__(self, data: dict):
        self.document_uri = data.get("document-uri")
        self.violated_directive = data.get("violated-directive")
        self.effective_directive = data.get("effective-directive")
        self.original_policy = data.get("original-policy")
        self.blocked_uri = data.get("blocked-uri")
        self.status_code = data.get("status-code")
        self.source_file = data.get("source-file")
        self.line_number = data.get("line-number")
        self.column_number = data.get("column-number")
        self.timestamp = datetime.utcnow()

    def to_dict(self):
        """Convert to dictionary for logging."""
        return {
            "document_uri": self.document_uri,
            "violated_directive": self.violated_directive,
            "effective_directive": self.effective_directive,
            "blocked_uri": self.blocked_uri,
            "status_code": self.status_code,
            "source_file": self.source_file,
            "line_number": self.line_number,
            "column_number": self.column_number,
            "timestamp": self.timestamp.isoformat(),
        }


async def log_csp_violation(report: CSPViolationReport):
    """
    Log CSP violation for monitoring.

    In production, you might want to send this to a SIEM or
    security monitoring service.

    Args:
        report: CSP violation report

    Example:
        >>> report = CSPViolationReport(violation_data)
        >>> await log_csp_violation(report)
    """
    logger.warning(
        f"CSP Violation: {report.violated_directive} "
        f"blocked {report.blocked_uri} on {report.document_uri}"
    )

    # In production, send to monitoring system
    # await send_to_siem(report.to_dict())


# Expect-CT violation report handler
class CTViolationReport:
    """Certificate Transparency violation report data structure."""

    def __init__(self, data: dict):
        self.hostname = data.get("hostname")
        self.port = data.get("port")
        self.effective_expiration_date = data.get("effective-expiration-date")
        self.served_certificate_chain = data.get("served-certificate-chain", [])
        self.validated_certificate_chain = data.get("validated-certificate-chain", [])
        self.scts = data.get("scts", [])
        self.timestamp = datetime.utcnow()

    def to_dict(self):
        """Convert to dictionary for logging."""
        return {
            "hostname": self.hostname,
            "port": self.port,
            "effective_expiration_date": self.effective_expiration_date,
            "timestamp": self.timestamp.isoformat(),
        }


async def log_ct_violation(report: CTViolationReport):
    """
    Log Certificate Transparency violation.

    Args:
        report: CT violation report

    Example:
        >>> report = CTViolationReport(violation_data)
        >>> await log_ct_violation(report)
    """
    logger.error(
        f"Certificate Transparency Violation: {report.hostname}:{report.port}"
    )

    # In production, send to security team immediately
    # await send_security_alert(report.to_dict())


# Example FastAPI endpoints for violation reporting
def create_security_report_endpoints():
    """
    Create FastAPI endpoints for security violation reporting.

    Returns:
        List of FastAPI route definitions

    Example:
        Add to your FastAPI app:
        ```python
        from fastapi import FastAPI, Request
        app = FastAPI()

        @app.post("/api/security/csp-report")
        async def csp_report_endpoint(request: Request):
            data = await request.json()
            report = CSPViolationReport(data.get("csp-report", {}))
            await log_csp_violation(report)
            return {"status": "received"}

        @app.post("/api/security/ct-report")
        async def ct_report_endpoint(request: Request):
            data = await request.json()
            report = CTViolationReport(data)
            await log_ct_violation(report)
            return {"status": "received"}
        ```
    """
    pass


__all__ = [
    "SecurityHeadersConfig",
    "SecurityHeadersMiddleware",
    "build_csp_header",
    "build_expect_ct_header",
    "build_nel_header",
    "build_report_to_header",
    "build_permissions_policy_header",
    "generate_sri_hash",
    "CSPViolationReport",
    "CTViolationReport",
    "log_csp_violation",
    "log_ct_violation",
]
