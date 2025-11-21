"""
Tests for Security Headers

This module tests the security headers implementation.
"""

import pytest
from unittest.mock import Mock, AsyncMock

from fastapi import FastAPI, Request, Response
from fastapi.testclient import TestClient

from greenlang.api.security.headers import (
    SecurityHeadersMiddleware,
    SecurityHeadersConfig,
    ContentSecurityPolicyDirective,
    FrameOptions,
    ReferrerPolicy,
    create_secure_headers,
    SECURITY_PRESETS
)


class TestSecurityHeadersConfig:
    """Test security headers configuration."""

    def test_default_config(self):
        """Test default security headers configuration."""
        config = SecurityHeadersConfig()

        assert config.x_content_type_options == "nosniff"
        assert config.x_frame_options == FrameOptions.DENY
        assert config.x_xss_protection == "1; mode=block"
        assert config.enable_hsts is True
        assert config.enable_csp is True

    def test_custom_config(self):
        """Test custom security headers configuration."""
        config = SecurityHeadersConfig(
            x_frame_options=FrameOptions.SAMEORIGIN,
            enable_hsts=False,
            referrer_policy=ReferrerPolicy.NO_REFERRER
        )

        assert config.x_frame_options == FrameOptions.SAMEORIGIN
        assert config.enable_hsts is False
        assert config.referrer_policy == ReferrerPolicy.NO_REFERRER

    def test_csp_directive_config(self):
        """Test CSP directive configuration."""
        csp = ContentSecurityPolicyDirective(
            default_src=["'self'", "example.com"],
            script_src=["'self'", "'unsafe-inline'"],
            style_src=["'self'"],
            img_src=["'self'", "data:", "https:"]
        )

        assert "'self'" in csp.default_src
        assert "example.com" in csp.default_src
        assert "'unsafe-inline'" in csp.script_src
        assert "https:" in csp.img_src


class TestSecurityHeadersMiddleware:
    """Test security headers middleware."""

    @pytest.fixture
    def app(self):
        """Create test FastAPI app."""
        app = FastAPI()

        @app.get("/")
        async def root():
            return {"message": "OK"}

        @app.get("/health")
        async def health():
            return {"status": "healthy"}

        @app.get("/api/data")
        async def get_data():
            return {"data": "test"}

        return app

    @pytest.fixture
    def client_with_headers(self, app):
        """Create test client with security headers."""
        config = SecurityHeadersConfig()
        app.add_middleware(SecurityHeadersMiddleware, config=config)
        return TestClient(app)

    def test_basic_security_headers(self, client_with_headers):
        """Test that basic security headers are added."""
        response = client_with_headers.get("/")

        assert response.headers.get("X-Content-Type-Options") == "nosniff"
        assert response.headers.get("X-Frame-Options") == "DENY"
        assert response.headers.get("X-XSS-Protection") == "1; mode=block"
        assert response.headers.get("Referrer-Policy") == "strict-origin-when-cross-origin"

    def test_csp_header(self, app):
        """Test Content-Security-Policy header."""
        config = SecurityHeadersConfig(
            enable_csp=True,
            content_security_policy=ContentSecurityPolicyDirective(
                default_src=["'self'"],
                script_src=["'self'", "'unsafe-inline'"]
            )
        )
        app.add_middleware(SecurityHeadersMiddleware, config=config)
        client = TestClient(app)

        response = client.get("/")
        csp = response.headers.get("Content-Security-Policy")

        assert csp is not None
        assert "default-src 'self'" in csp
        assert "script-src 'self' 'unsafe-inline'" in csp

    def test_csp_report_only(self, app):
        """Test CSP in report-only mode."""
        config = SecurityHeadersConfig(
            enable_csp=True,
            csp_report_only=True,
            csp_report_uri="/csp-report"
        )
        app.add_middleware(SecurityHeadersMiddleware, config=config)
        client = TestClient(app)

        response = client.get("/")

        assert "Content-Security-Policy-Report-Only" in response.headers
        assert "Content-Security-Policy" not in response.headers
        csp = response.headers.get("Content-Security-Policy-Report-Only")
        assert "report-uri /csp-report" in csp

    def test_hsts_header_https_only(self, app):
        """Test that HSTS is only added for HTTPS."""
        config = SecurityHeadersConfig(enable_hsts=True)
        app.add_middleware(SecurityHeadersMiddleware, config=config)

        # Create mock request with HTTPS
        @app.middleware("http")
        async def mock_https(request: Request, call_next):
            request.url._url = request.url._url.replace("http://", "https://")
            return await call_next(request)

        client = TestClient(app)
        response = client.get("/")

        # Note: TestClient uses http by default, so HSTS won't be added
        assert "Strict-Transport-Security" not in response.headers

    def test_permissions_policy(self, app):
        """Test Permissions-Policy header."""
        config = SecurityHeadersConfig(
            enable_permissions_policy=True,
            permissions_policy={
                "geolocation": ["'self'"],
                "camera": ["'none'"],
                "microphone": ["'none'"]
            }
        )
        app.add_middleware(SecurityHeadersMiddleware, config=config)
        client = TestClient(app)

        response = client.get("/")
        permissions = response.headers.get("Permissions-Policy")

        assert permissions is not None
        assert "geolocation=('self')" in permissions
        assert "camera=('none')" in permissions

    def test_cache_control_headers(self, app):
        """Test cache control headers for sensitive content."""
        config = SecurityHeadersConfig(
            cache_control="no-store, no-cache",
            pragma="no-cache",
            expires="0"
        )
        app.add_middleware(SecurityHeadersMiddleware, config=config)
        client = TestClient(app)

        response = client.get("/api/data")

        # Cache headers should be set for JSON responses
        assert "no-store" in response.headers.get("Cache-Control", "")
        assert response.headers.get("Pragma") == "no-cache"
        assert response.headers.get("Expires") == "0"

    def test_custom_headers(self, app):
        """Test custom security headers."""
        config = SecurityHeadersConfig(
            custom_headers={
                "X-Custom-Header": "custom-value",
                "X-Another-Header": "another-value"
            }
        )
        app.add_middleware(SecurityHeadersMiddleware, config=config)
        client = TestClient(app)

        response = client.get("/")

        assert response.headers.get("X-Custom-Header") == "custom-value"
        assert response.headers.get("X-Another-Header") == "another-value"

    def test_exclude_paths(self, app):
        """Test that excluded paths don't get security headers."""
        config = SecurityHeadersConfig(
            exclude_paths=["/health"]
        )

        middleware = SecurityHeadersMiddleware(app, config)

        # Test excluded path
        request = Mock(spec=Request)
        request.url.path = "/health"
        assert middleware._should_apply_headers(request) is False

        # Test included path
        request.url.path = "/api/data"
        assert middleware._should_apply_headers(request) is True

    def test_frame_options_allow_from(self, app):
        """Test X-Frame-Options with ALLOW-FROM."""
        config = SecurityHeadersConfig(
            x_frame_options=FrameOptions.ALLOW_FROM,
            x_frame_options_allow_from="https://example.com"
        )
        app.add_middleware(SecurityHeadersMiddleware, config=config)
        client = TestClient(app)

        response = client.get("/")

        assert response.headers.get("X-Frame-Options") == "ALLOW-FROM https://example.com"


class TestSecurityPresets:
    """Test security header presets."""

    def test_strict_preset(self):
        """Test strict security preset."""
        config = SECURITY_PRESETS["strict"]

        assert config.x_frame_options == FrameOptions.DENY
        assert config.enable_hsts is True
        assert config.enable_csp is True
        assert config.referrer_policy == ReferrerPolicy.NO_REFERRER

        csp = config.content_security_policy
        assert csp.default_src == ["'self'"]
        assert csp.script_src == ["'self'"]
        assert csp.frame_ancestors == ["'none'"]

    def test_balanced_preset(self):
        """Test balanced security preset."""
        config = SECURITY_PRESETS["balanced"]

        assert config.x_frame_options == FrameOptions.SAMEORIGIN
        assert config.enable_hsts is True
        assert config.enable_csp is True

        csp = config.content_security_policy
        assert "'unsafe-inline'" in csp.script_src
        assert "'unsafe-inline'" in csp.style_src

    def test_relaxed_preset(self):
        """Test relaxed security preset."""
        config = SECURITY_PRESETS["relaxed"]

        assert config.x_frame_options == FrameOptions.SAMEORIGIN
        assert config.enable_hsts is True
        assert config.enable_csp is False


class TestCreateSecureHeaders:
    """Test create_secure_headers utility function."""

    def test_create_basic_headers(self):
        """Test creating basic security headers."""
        headers = create_secure_headers()

        assert headers["X-Content-Type-Options"] == "nosniff"
        assert headers["X-Frame-Options"] == "DENY"
        assert headers["X-XSS-Protection"] == "1; mode=block"
        assert "Strict-Transport-Security" in headers

    def test_create_headers_without_hsts(self):
        """Test creating headers without HSTS."""
        headers = create_secure_headers(enable_hsts=False)

        assert "Strict-Transport-Security" not in headers

    def test_create_headers_without_csp(self):
        """Test creating headers without CSP."""
        headers = create_secure_headers(enable_csp=False)

        assert "Content-Security-Policy" not in headers

    def test_create_headers_with_custom_csp(self):
        """Test creating headers with custom CSP."""
        headers = create_secure_headers(
            enable_csp=True,
            custom_csp={
                "default_src": ["'self'", "example.com"],
                "script_src": ["'self'", "'unsafe-eval'"]
            }
        )

        csp = headers["Content-Security-Policy"]
        assert "default-src 'self' example.com" in csp
        assert "script-src 'self' 'unsafe-eval'" in csp


class TestCSPBuilder:
    """Test CSP header building."""

    def test_build_basic_csp(self):
        """Test building basic CSP header."""
        config = SecurityHeadersConfig()
        middleware = SecurityHeadersMiddleware(None, config)

        csp = middleware._build_csp_header()

        assert "default-src 'self'" in csp
        assert "script-src 'self'" in csp
        assert "style-src 'self'" in csp
        assert "upgrade-insecure-requests" in csp

    def test_build_csp_with_report_uri(self):
        """Test building CSP with report-uri."""
        config = SecurityHeadersConfig(
            csp_report_uri="/api/csp-report"
        )
        middleware = SecurityHeadersMiddleware(None, config)

        csp = middleware._build_csp_header()

        assert "report-uri /api/csp-report" in csp

    def test_build_empty_csp(self):
        """Test building CSP when disabled."""
        config = SecurityHeadersConfig(
            content_security_policy=None
        )
        middleware = SecurityHeadersMiddleware(None, config)

        csp = middleware._build_csp_header()

        assert csp == ""


class TestPermissionsPolicyBuilder:
    """Test Permissions-Policy header building."""

    def test_build_permissions_policy(self):
        """Test building Permissions-Policy header."""
        config = SecurityHeadersConfig(
            permissions_policy={
                "geolocation": ["'self'", "example.com"],
                "camera": ["'none'"],
                "microphone": []
            }
        )
        middleware = SecurityHeadersMiddleware(None, config)

        policy = middleware._build_permissions_policy_header()

        assert "geolocation=('self' example.com)" in policy
        assert "camera=('none')" in policy
        assert "microphone=()" in policy

    def test_build_empty_permissions_policy(self):
        """Test building empty Permissions-Policy."""
        config = SecurityHeadersConfig(
            permissions_policy={}
        )
        middleware = SecurityHeadersMiddleware(None, config)

        policy = middleware._build_permissions_policy_header()

        assert policy == ""