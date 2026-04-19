"""
Tests for CSRF Protection

This module tests the CSRF protection implementation.
"""

import time
import pytest
from unittest.mock import Mock, patch

from fastapi import FastAPI, Request, HTTPException
from fastapi.testclient import TestClient

from greenlang.api.security.csrf import (
    CSRFProtect,
    CSRFConfig,
    generate_csrf_token,
    validate_csrf_token
)


class TestCSRFToken:
    """Test CSRF token generation and validation."""

    def test_generate_csrf_token(self):
        """Test CSRF token generation."""
        token = generate_csrf_token()
        assert token is not None
        assert len(token) >= 32
        assert isinstance(token, str)

    def test_validate_csrf_token(self):
        """Test CSRF token validation."""
        token = generate_csrf_token()
        assert validate_csrf_token(token, token) is True
        assert validate_csrf_token(token, "different") is False

    def test_csrf_token_uniqueness(self):
        """Test that generated tokens are unique."""
        tokens = [generate_csrf_token() for _ in range(100)]
        assert len(set(tokens)) == 100


class TestCSRFProtect:
    """Test CSRFProtect class."""

    @pytest.fixture
    def csrf_config(self):
        """Create test CSRF configuration."""
        return CSRFConfig(
            secret_key="test-secret-key",
            token_length=32,
            token_expiry_seconds=3600,
            cookie_name="csrf_token",
            header_name="X-CSRF-Token"
        )

    @pytest.fixture
    def csrf_protect(self, csrf_config):
        """Create CSRFProtect instance."""
        return CSRFProtect(csrf_config)

    def test_generate_token_with_signature(self, csrf_protect):
        """Test token generation with HMAC signature."""
        token = csrf_protect.generate_csrf_token()

        assert token.token is not None
        assert token.signature is not None
        assert token.timestamp > 0
        assert len(token.token) >= 32

    def test_validate_token_success(self, csrf_protect):
        """Test successful token validation."""
        token = csrf_protect.generate_csrf_token()

        is_valid = csrf_protect.validate_token(token.token, token.signature)
        assert is_valid is True

    def test_validate_token_invalid_signature(self, csrf_protect):
        """Test token validation with invalid signature."""
        token = csrf_protect.generate_csrf_token()

        is_valid = csrf_protect.validate_token(token.token, "invalid-signature")
        assert is_valid is False

    def test_validate_token_not_in_cache(self, csrf_protect):
        """Test token validation when token not in cache."""
        is_valid = csrf_protect.validate_token("non-existent-token", "signature")
        assert is_valid is False

    def test_validate_token_expired(self, csrf_protect):
        """Test token validation when expired."""
        # Generate token and modify cache timestamp
        token = csrf_protect.generate_csrf_token()
        csrf_protect._token_cache[token.token] = time.time() - 7200  # 2 hours ago

        is_valid = csrf_protect.validate_token(token.token, token.signature)
        assert is_valid is False
        assert token.token not in csrf_protect._token_cache

    def test_cleanup_expired_tokens(self, csrf_protect):
        """Test cleanup of expired tokens."""
        # Generate tokens
        token1 = csrf_protect.generate_csrf_token()
        token2 = csrf_protect.generate_csrf_token()

        # Make one expired
        csrf_protect._token_cache[token1.token] = time.time() - 7200

        # Force cleanup
        csrf_protect._last_cleanup = 0
        csrf_protect._cleanup_expired_tokens()

        assert token1.token not in csrf_protect._token_cache
        assert token2.token in csrf_protect._token_cache

    def test_is_exempt_safe_method(self, csrf_protect):
        """Test exemption for safe HTTP methods."""
        request = Mock(spec=Request)

        request.method = "GET"
        assert csrf_protect._is_exempt(request) is True

        request.method = "POST"
        assert csrf_protect._is_exempt(request) is False

    def test_is_exempt_path(self, csrf_protect):
        """Test exemption for specific paths."""
        csrf_protect.config.exempt_paths = {"/health", "/metrics"}

        request = Mock(spec=Request)
        request.method = "POST"
        request.url.path = "/health"
        assert csrf_protect._is_exempt(request) is True

        request.url.path = "/api/data"
        assert csrf_protect._is_exempt(request) is False


class TestCSRFMiddleware:
    """Test CSRF middleware integration."""

    @pytest.fixture
    def app(self):
        """Create test FastAPI app."""
        app = FastAPI()

        @app.get("/")
        async def root():
            return {"message": "OK"}

        @app.post("/api/data")
        async def create_data(data: dict):
            return {"created": True}

        @app.get("/health")
        async def health():
            return {"status": "healthy"}

        return app

    @pytest.fixture
    def client_with_csrf(self, app):
        """Create test client with CSRF protection."""
        csrf_config = CSRFConfig(
            secret_key="test-secret",
            exempt_paths={"/health"}
        )
        csrf_protect = CSRFProtect(csrf_config)

        # Add middleware to app
        @app.middleware("http")
        async def csrf_middleware(request: Request, call_next):
            return await csrf_protect(request, call_next)

        return TestClient(app), csrf_protect

    def test_get_request_no_csrf_required(self, client_with_csrf):
        """Test that GET requests don't require CSRF token."""
        client, _ = client_with_csrf
        response = client.get("/")
        assert response.status_code == 200

    def test_post_request_without_token(self, client_with_csrf):
        """Test POST request without CSRF token."""
        client, _ = client_with_csrf
        response = client.post("/api/data", json={"test": "data"})
        assert response.status_code == 403
        assert "CSRF token missing" in response.json()["detail"]

    def test_post_request_with_valid_token(self, client_with_csrf):
        """Test POST request with valid CSRF token."""
        client, csrf_protect = client_with_csrf

        # Generate token
        token = csrf_protect.generate_csrf_token()
        token_string = f"{token.token}:{token.signature}"

        # Make request with token in header
        response = client.post(
            "/api/data",
            json={"test": "data"},
            headers={"X-CSRF-Token": token_string}
        )
        assert response.status_code == 200

    def test_post_request_with_invalid_token(self, client_with_csrf):
        """Test POST request with invalid CSRF token."""
        client, _ = client_with_csrf

        response = client.post(
            "/api/data",
            json={"test": "data"},
            headers={"X-CSRF-Token": "invalid:token"}
        )
        assert response.status_code == 403

    def test_exempt_path(self, client_with_csrf):
        """Test that exempt paths don't require CSRF."""
        client, _ = client_with_csrf
        response = client.get("/health")
        assert response.status_code == 200

    def test_token_in_cookie(self, client_with_csrf):
        """Test CSRF token from cookie."""
        client, csrf_protect = client_with_csrf

        # Generate token
        token = csrf_protect.generate_csrf_token()
        token_string = f"{token.token}:{token.signature}"

        # Make request with token in cookie
        response = client.post(
            "/api/data",
            json={"test": "data"},
            cookies={"csrf_token": token_string}
        )
        assert response.status_code == 200