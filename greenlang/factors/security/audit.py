# -*- coding: utf-8 -*-
"""
Security audit helpers for the Factors API.

Provides functions to check security posture of the API deployment:
  - Required HTTP security headers
  - CORS configuration
  - Authentication/authorization configuration

Each check returns a list of SecurityFinding objects with severity,
category, message, and remediation guidance.

Example:
    >>> findings = check_headers(mock_request)
    >>> for f in findings:
    ...     print(f.severity, f.message)
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from enum import Enum
from typing import Any, List, Optional

logger = logging.getLogger(__name__)


class Severity(str, Enum):
    """Severity level for security findings."""

    CRITICAL = "critical"
    WARNING = "warning"
    INFO = "info"


@dataclass
class SecurityFinding:
    """A single security audit finding.

    Attributes:
        severity: critical, warning, or info.
        category: Finding category (e.g., "headers", "cors", "auth").
        message: Description of the finding.
        remediation: Recommended fix.
    """

    severity: Severity
    category: str
    message: str
    remediation: str

    def to_dict(self) -> dict:
        """Serialize to dictionary."""
        return {
            "severity": self.severity.value,
            "category": self.category,
            "message": self.message,
            "remediation": self.remediation,
        }


# ── Required security headers ─────────────────────────────────────

_REQUIRED_HEADERS = {
    "Strict-Transport-Security": {
        "expected_prefix": "max-age=",
        "severity": Severity.CRITICAL,
        "remediation": (
            "Add Strict-Transport-Security header with max-age >= 31536000. "
            "Example: Strict-Transport-Security: max-age=31536000; includeSubDomains"
        ),
    },
    "X-Content-Type-Options": {
        "expected_value": "nosniff",
        "severity": Severity.WARNING,
        "remediation": "Add X-Content-Type-Options: nosniff header.",
    },
    "X-Frame-Options": {
        "expected_value": "DENY",
        "severity": Severity.WARNING,
        "remediation": "Add X-Frame-Options: DENY header to prevent clickjacking.",
    },
    "Content-Security-Policy": {
        "expected_prefix": "default-src",
        "severity": Severity.WARNING,
        "remediation": (
            "Add Content-Security-Policy header. "
            "Example: default-src 'self'; script-src 'none'"
        ),
    },
    "X-Request-ID": {
        "severity": Severity.INFO,
        "remediation": (
            "Add X-Request-ID header for request tracing. "
            "Generate a UUID per request for correlation."
        ),
    },
    "Cache-Control": {
        "severity": Severity.INFO,
        "remediation": (
            "Add Cache-Control header to control response caching. "
            "Sensitive endpoints should use no-store."
        ),
    },
}


def check_headers(request: Any) -> List[SecurityFinding]:
    """Check for required security headers in an HTTP request/response.

    Examines the response headers (or request headers if checking inbound)
    for presence and correctness of security-critical headers.

    Args:
        request: An object with a 'headers' attribute (dict-like).
            Accepts FastAPI Request, httpx Response, or plain dict.

    Returns:
        List of SecurityFinding objects for missing or misconfigured headers.
    """
    findings: List[SecurityFinding] = []

    if isinstance(request, dict):
        headers = request
    elif hasattr(request, "headers"):
        headers = dict(request.headers)
    else:
        findings.append(SecurityFinding(
            severity=Severity.WARNING,
            category="headers",
            message="Cannot inspect headers: unsupported request type",
            remediation="Pass a dict or object with .headers attribute.",
        ))
        return findings

    # Normalize header names to title case for comparison
    normalized = {k.title(): v for k, v in headers.items()}

    for header_name, config in _REQUIRED_HEADERS.items():
        title_name = header_name.title()
        value = normalized.get(title_name, normalized.get(header_name))

        if value is None:
            findings.append(SecurityFinding(
                severity=config.get("severity", Severity.WARNING),
                category="headers",
                message="Missing security header: %s" % header_name,
                remediation=config.get("remediation", "Add %s header." % header_name),
            ))
            continue

        # Check expected value if specified
        expected_value = config.get("expected_value")
        if expected_value and value.lower() != expected_value.lower():
            findings.append(SecurityFinding(
                severity=Severity.WARNING,
                category="headers",
                message="%s header has unexpected value: %r (expected %r)"
                        % (header_name, value, expected_value),
                remediation=config.get("remediation", ""),
            ))

        # Check expected prefix if specified
        expected_prefix = config.get("expected_prefix")
        if expected_prefix and not value.lower().startswith(expected_prefix.lower()):
            findings.append(SecurityFinding(
                severity=Severity.WARNING,
                category="headers",
                message="%s header value does not start with %r: %r"
                        % (header_name, expected_prefix, value),
                remediation=config.get("remediation", ""),
            ))

    return findings


# ── CORS configuration ────────────────────────────────────────────

# Dangerous CORS origins that should never be allowed in production
_DANGEROUS_ORIGINS = {"*", "null"}

# Maximum safe max_age for CORS preflight cache (24 hours)
_MAX_CORS_MAX_AGE = 86400


def check_cors_config(app: Any) -> List[SecurityFinding]:
    """Validate CORS configuration of a FastAPI/Starlette application.

    Checks for:
      - Wildcard (*) origins in production
      - Credentials allowed with wildcard origins
      - Excessively long preflight cache
      - Missing CORS configuration

    Args:
        app: A FastAPI/Starlette application, or a dict with CORS config.

    Returns:
        List of SecurityFinding objects for CORS issues.
    """
    findings: List[SecurityFinding] = []

    cors_config = _extract_cors_config(app)

    if cors_config is None:
        findings.append(SecurityFinding(
            severity=Severity.INFO,
            category="cors",
            message="No CORS middleware detected",
            remediation=(
                "If this API is called from browsers, configure CORSMiddleware "
                "with specific allowed origins."
            ),
        ))
        return findings

    # Check origins
    origins = cors_config.get("allow_origins", [])
    if "*" in origins:
        env = os.getenv("GL_ENV", "production").lower()
        if env in ("production", "staging", "prod"):
            findings.append(SecurityFinding(
                severity=Severity.CRITICAL,
                category="cors",
                message="CORS allows all origins (*) in %s environment" % env,
                remediation=(
                    "Restrict allow_origins to specific domains. "
                    "Never use '*' in production."
                ),
            ))
        else:
            findings.append(SecurityFinding(
                severity=Severity.WARNING,
                category="cors",
                message="CORS allows all origins (*) in %s environment" % env,
                remediation=(
                    "Consider restricting allow_origins even in non-production."
                ),
            ))

    # Check credentials with wildcard
    allow_credentials = cors_config.get("allow_credentials", False)
    if allow_credentials and "*" in origins:
        findings.append(SecurityFinding(
            severity=Severity.CRITICAL,
            category="cors",
            message="CORS allows credentials with wildcard origin",
            remediation=(
                "Never combine allow_credentials=True with allow_origins=['*']. "
                "This is a browser security violation."
            ),
        ))

    # Check allowed methods
    methods = cors_config.get("allow_methods", [])
    if "*" in methods:
        findings.append(SecurityFinding(
            severity=Severity.WARNING,
            category="cors",
            message="CORS allows all HTTP methods (*)",
            remediation="Restrict allow_methods to only needed methods (GET, POST, OPTIONS).",
        ))

    # Check max_age
    max_age = cors_config.get("max_age", 600)
    if isinstance(max_age, (int, float)) and max_age > _MAX_CORS_MAX_AGE:
        findings.append(SecurityFinding(
            severity=Severity.WARNING,
            category="cors",
            message="CORS max_age (%d) exceeds recommended maximum (%d)"
                    % (max_age, _MAX_CORS_MAX_AGE),
            remediation="Set max_age to %d or lower." % _MAX_CORS_MAX_AGE,
        ))

    return findings


def _extract_cors_config(app: Any) -> Optional[dict]:
    """Extract CORS configuration from a FastAPI app or dict.

    Args:
        app: FastAPI/Starlette app or dict with CORS settings.

    Returns:
        Dict with CORS configuration, or None if not found.
    """
    if isinstance(app, dict):
        return app

    # Try to find CORSMiddleware in FastAPI middleware stack
    try:
        if hasattr(app, "middleware_stack"):
            middleware = app.middleware_stack
            while middleware is not None:
                cls_name = type(middleware).__name__
                if "CORS" in cls_name or "cors" in cls_name.lower():
                    config = {}
                    for attr in ("allow_origins", "allow_methods", "allow_headers",
                                 "allow_credentials", "max_age"):
                        val = getattr(middleware, attr, None)
                        if val is not None:
                            config[attr] = val
                    return config if config else None
                middleware = getattr(middleware, "app", None)
    except Exception:
        pass

    # Try user_middleware list (FastAPI stores middleware before build)
    try:
        if hasattr(app, "user_middleware"):
            for mw in app.user_middleware:
                cls = getattr(mw, "cls", None)
                if cls and "CORS" in getattr(cls, "__name__", ""):
                    return mw.kwargs if hasattr(mw, "kwargs") else None
    except Exception:
        pass

    return None


# ── Auth configuration ─────────────────────────────────────────────


def check_auth_config() -> List[SecurityFinding]:
    """Validate JWT and API key authentication configuration.

    Checks environment variables and configuration for:
      - JWT secret key presence and strength
      - API key enforcement
      - Token expiration settings
      - HTTPS enforcement

    Returns:
        List of SecurityFinding objects for auth configuration issues.
    """
    findings: List[SecurityFinding] = []

    # Check JWT secret
    jwt_secret = os.getenv("GL_JWT_SECRET", "")
    if not jwt_secret:
        findings.append(SecurityFinding(
            severity=Severity.CRITICAL,
            category="auth",
            message="GL_JWT_SECRET environment variable not set",
            remediation=(
                "Set GL_JWT_SECRET to a strong random value (at least 32 characters). "
                "Generate with: python -c \"import secrets; print(secrets.token_urlsafe(48))\""
            ),
        ))
    elif len(jwt_secret) < 32:
        findings.append(SecurityFinding(
            severity=Severity.CRITICAL,
            category="auth",
            message="GL_JWT_SECRET is too short (%d chars, minimum 32)" % len(jwt_secret),
            remediation="Use a JWT secret of at least 32 characters.",
        ))
    elif jwt_secret in ("secret", "changeme", "test", "development"):
        findings.append(SecurityFinding(
            severity=Severity.CRITICAL,
            category="auth",
            message="GL_JWT_SECRET uses a well-known insecure value",
            remediation="Replace with a cryptographically random secret.",
        ))

    # Check JWT algorithm
    jwt_alg = os.getenv("GL_JWT_ALGORITHM", "HS256")
    if jwt_alg == "none":
        findings.append(SecurityFinding(
            severity=Severity.CRITICAL,
            category="auth",
            message="JWT algorithm set to 'none' (no signature verification)",
            remediation="Use RS256 or HS256 algorithm for JWT tokens.",
        ))
    elif jwt_alg not in ("RS256", "HS256", "ES256"):
        findings.append(SecurityFinding(
            severity=Severity.WARNING,
            category="auth",
            message="JWT algorithm %r is not a recommended algorithm" % jwt_alg,
            remediation="Use RS256 (preferred), ES256, or HS256.",
        ))

    # Check token expiration
    token_exp = os.getenv("GL_JWT_EXPIRATION_MINUTES", "")
    if not token_exp:
        findings.append(SecurityFinding(
            severity=Severity.WARNING,
            category="auth",
            message="GL_JWT_EXPIRATION_MINUTES not set (tokens may not expire)",
            remediation="Set GL_JWT_EXPIRATION_MINUTES to a reasonable value (e.g., 60).",
        ))
    elif token_exp.isdigit() and int(token_exp) > 1440:
        findings.append(SecurityFinding(
            severity=Severity.WARNING,
            category="auth",
            message="JWT token expiration is %s minutes (> 24 hours)" % token_exp,
            remediation="Consider shorter token lifetimes (60-480 minutes).",
        ))

    # Check HTTPS enforcement
    gl_env = os.getenv("GL_ENV", "").lower()
    force_https = os.getenv("GL_FORCE_HTTPS", "").lower()
    if gl_env in ("production", "staging", "prod") and force_https not in ("1", "true", "yes"):
        findings.append(SecurityFinding(
            severity=Severity.CRITICAL,
            category="auth",
            message="HTTPS not enforced in %s environment" % gl_env,
            remediation="Set GL_FORCE_HTTPS=true in production environments.",
        ))

    # Check API key requirement
    require_api_key = os.getenv("GL_REQUIRE_API_KEY", "").lower()
    if gl_env in ("production", "prod") and require_api_key not in ("1", "true", "yes"):
        findings.append(SecurityFinding(
            severity=Severity.WARNING,
            category="auth",
            message="API key requirement not explicitly enabled in production",
            remediation="Set GL_REQUIRE_API_KEY=true to enforce API key authentication.",
        ))

    return findings
