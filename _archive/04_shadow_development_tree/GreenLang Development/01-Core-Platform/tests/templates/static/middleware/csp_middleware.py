# -*- coding: utf-8 -*-
"""
Content Security Policy (CSP) Middleware for FastAPI/Flask

Adds security headers to prevent XSS, clickjacking, and other attacks.
Compatible with both FastAPI and Flask frameworks.

Usage (FastAPI):
    from fastapi import FastAPI
    from csp_middleware import SecurityHeadersMiddleware

    app = FastAPI()
    app.add_middleware(SecurityHeadersMiddleware)

Usage (Flask):
    from flask import Flask
    from csp_middleware import add_security_headers

    app = Flask(__name__)

    @app.after_request
    def apply_security_headers(response):
        return add_security_headers(response)
"""

from typing import Callable, Dict, Optional
import re


class SecurityHeadersMiddleware:
    """
    ASGI Middleware for adding security headers (FastAPI compatible)
    """

    def __init__(
        self,
        app,
        csp_policy: Optional[str] = None,
        report_uri: Optional[str] = None,
        report_only: bool = False
    ):
        self.app = app
        self.csp_policy = csp_policy or self._default_csp_policy()
        self.report_uri = report_uri
        self.report_only = report_only

    @staticmethod
    def _default_csp_policy() -> str:
        """
        Default Content Security Policy for GreenLang applications.

        This policy:
        - Allows resources only from same origin by default
        - Allows scripts from self and trusted CDNs (jsDelivr for DOMPurify)
        - Allows inline styles (required for many frameworks, but disabled inline scripts)
        - Blocks all plugins, frames from other origins
        - Reports violations if report-uri is configured
        """
        return (
            "default-src 'self'; "
            "script-src 'self' https://cdn.jsdelivr.net; "
            "style-src 'self' 'unsafe-inline'; "
            "img-src 'self' data: https:; "
            "font-src 'self' data:; "
            "connect-src 'self'; "
            "frame-ancestors 'none'; "
            "base-uri 'self'; "
            "form-action 'self'; "
            "object-src 'none'; "
            "upgrade-insecure-requests"
        )

    @staticmethod
    def _strict_csp_policy() -> str:
        """
        Strict CSP policy (no inline styles, stricter rules).
        Use this for production environments with proper CSP nonce implementation.
        """
        return (
            "default-src 'self'; "
            "script-src 'self' https://cdn.jsdelivr.net; "
            "style-src 'self'; "
            "img-src 'self' data: https:; "
            "font-src 'self'; "
            "connect-src 'self'; "
            "frame-ancestors 'none'; "
            "base-uri 'self'; "
            "form-action 'self'; "
            "object-src 'none'; "
            "upgrade-insecure-requests"
        )

    def _get_security_headers(self) -> Dict[str, str]:
        """
        Returns all security headers to be added to responses.
        """
        csp_header = "Content-Security-Policy-Report-Only" if self.report_only else "Content-Security-Policy"
        csp_value = self.csp_policy

        if self.report_uri:
            csp_value += f"; report-uri {self.report_uri}"

        return {
            # Content Security Policy
            csp_header: csp_value,

            # Prevent clickjacking
            "X-Frame-Options": "DENY",

            # Prevent MIME type sniffing
            "X-Content-Type-Options": "nosniff",

            # XSS Protection (legacy, but still useful for older browsers)
            "X-XSS-Protection": "1; mode=block",

            # Referrer Policy (don't leak URLs to external sites)
            "Referrer-Policy": "strict-origin-when-cross-origin",

            # Permissions Policy (disable dangerous features)
            "Permissions-Policy": (
                "accelerometer=(), "
                "camera=(), "
                "geolocation=(), "
                "gyroscope=(), "
                "magnetometer=(), "
                "microphone=(), "
                "payment=(), "
                "usb=()"
            ),

            # Strict Transport Security (HTTPS only, if using HTTPS)
            # Uncomment in production with HTTPS
            # "Strict-Transport-Security": "max-age=31536000; includeSubDomains",
        }

    async def __call__(self, scope, receive, send):
        """
        ASGI middleware implementation
        """
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        async def send_with_headers(message):
            if message["type"] == "http.response.start":
                headers = dict(message.get("headers", []))
                security_headers = self._get_security_headers()

                for name, value in security_headers.items():
                    headers[name.lower().encode()] = value.encode()

                message["headers"] = list(headers.items())

            await send(message)

        await self.app(scope, receive, send_with_headers)


def add_security_headers(response, csp_policy: Optional[str] = None, report_only: bool = False):
    """
    Flask/Werkzeug compatible function to add security headers.

    Args:
        response: Flask response object
        csp_policy: Custom CSP policy (optional)
        report_only: If True, uses Content-Security-Policy-Report-Only header

    Returns:
        Modified response with security headers
    """
    middleware = SecurityHeadersMiddleware(None, csp_policy=csp_policy, report_only=report_only)
    headers = middleware._get_security_headers()

    for name, value in headers.items():
        response.headers[name] = value

    return response


def validate_csp_policy(policy: str) -> tuple[bool, list[str]]:
    """
    Validates a Content Security Policy string.

    Args:
        policy: CSP policy string

    Returns:
        Tuple of (is_valid, list_of_errors)
    """
    errors = []

    # Check for dangerous directives
    dangerous_patterns = [
        (r"script-src[^;]*'unsafe-eval'", "unsafe-eval in script-src is dangerous"),
        (r"default-src[^;]*\*", "wildcard in default-src is too permissive"),
        (r"script-src[^;]*\*", "wildcard in script-src is dangerous"),
    ]

    for pattern, error_msg in dangerous_patterns:
        if re.search(pattern, policy, re.IGNORECASE):
            errors.append(error_msg)

    # Check for recommended directives
    recommended_directives = [
        ("default-src", "default-src directive is recommended"),
        ("script-src", "script-src directive is recommended"),
        ("object-src", "object-src 'none' is recommended"),
    ]

    for directive, error_msg in recommended_directives:
        if directive not in policy.lower():
            errors.append(f"Missing: {error_msg}")

    return len(errors) == 0, errors


def generate_csp_nonce() -> str:
    """
    Generates a cryptographically secure nonce for CSP.

    Use this to allow specific inline scripts while blocking others.

    Example usage in template:
        <script nonce="{{ csp_nonce }}">
            // Inline script here
        </script>

    And in CSP header:
        script-src 'self' 'nonce-{{ csp_nonce }}'
    """
    import secrets
    import base64

    random_bytes = secrets.token_bytes(16)
    return base64.b64encode(random_bytes).decode('utf-8')


# Example configurations for different environments
CSP_CONFIGS = {
    "development": {
        "csp_policy": SecurityHeadersMiddleware._default_csp_policy(),
        "report_only": True,  # Don't block in dev, just report
    },
    "staging": {
        "csp_policy": SecurityHeadersMiddleware._default_csp_policy(),
        "report_only": False,
        "report_uri": "/api/csp-report",  # Log violations
    },
    "production": {
        "csp_policy": SecurityHeadersMiddleware._strict_csp_policy(),
        "report_only": False,
        "report_uri": "/api/csp-report",
    },
}


if __name__ == "__main__":
    # Test CSP validation
    print("Testing CSP validation...")

    test_policies = [
        SecurityHeadersMiddleware._default_csp_policy(),
        SecurityHeadersMiddleware._strict_csp_policy(),
        "default-src *; script-src * 'unsafe-eval'",  # Bad policy
    ]

    for i, policy in enumerate(test_policies):
        is_valid, errors = validate_csp_policy(policy)
        print(f"\nPolicy {i + 1}:")
        print(f"Valid: {is_valid}")
        if errors:
            print("Errors:")
            for error in errors:
                print(f"  - {error}")
