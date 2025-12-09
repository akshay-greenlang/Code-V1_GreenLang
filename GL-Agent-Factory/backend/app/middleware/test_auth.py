"""
Unit Tests for JWT Authentication Middleware

This module provides comprehensive tests for the JWTAuthMiddleware,
covering JWT token validation, API key authentication, and error handling.

Test Coverage:
- JWT token validation (valid, expired, invalid signature)
- Token claims extraction
- API key validation
- Public path exclusion
- Error response formatting
- Token blacklist functionality
- Token refresh warnings

Example:
    >>> pytest backend/app/middleware/test_auth.py -v
"""

import time
from datetime import datetime, timedelta, timezone
from typing import Any, Dict
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import FastAPI, Request
from fastapi.testclient import TestClient
from jose import jwt
from starlette.responses import JSONResponse

from app.middleware.auth import (
    APIKeyInfo,
    APIKeyStore,
    AuthError,
    JWTAuthMiddleware,
    TokenBlacklist,
    TokenClaims,
    create_jwt_token,
    decode_jwt_token,
)


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def secret_key() -> str:
    """Test secret key for JWT signing."""
    return "test-secret-key-for-unit-testing-only"


@pytest.fixture
def algorithm() -> str:
    """Test algorithm for JWT."""
    return "HS256"


@pytest.fixture
def test_user_id() -> str:
    """Test user ID."""
    return "user-test-123"


@pytest.fixture
def test_tenant_id() -> str:
    """Test tenant ID."""
    return "tenant-test-abc"


@pytest.fixture
def valid_token(secret_key: str, algorithm: str, test_user_id: str, test_tenant_id: str) -> str:
    """Create a valid JWT token for testing."""
    return create_jwt_token(
        user_id=test_user_id,
        tenant_id=test_tenant_id,
        secret_key=secret_key,
        algorithm=algorithm,
        expires_in_hours=1,
        roles=["user", "developer"],
        permissions=["read", "write"],
    )


@pytest.fixture
def expired_token(secret_key: str, algorithm: str, test_user_id: str, test_tenant_id: str) -> str:
    """Create an expired JWT token for testing."""
    current_time = int(time.time())
    payload = {
        "sub": test_user_id,
        "tenant_id": test_tenant_id,
        "roles": ["user"],
        "permissions": [],
        "iat": current_time - 7200,  # 2 hours ago
        "exp": current_time - 3600,  # Expired 1 hour ago
        "jti": "expired-token-jti",
        "token_type": "access",
    }
    return jwt.encode(payload, secret_key, algorithm=algorithm)


@pytest.fixture
def token_expiring_soon(secret_key: str, algorithm: str, test_user_id: str, test_tenant_id: str) -> str:
    """Create a token expiring within 5 minutes."""
    current_time = int(time.time())
    payload = {
        "sub": test_user_id,
        "tenant_id": test_tenant_id,
        "roles": ["user"],
        "permissions": [],
        "iat": current_time - 3600,
        "exp": current_time + 120,  # Expires in 2 minutes
        "jti": "expiring-soon-jti",
        "token_type": "access",
    }
    return jwt.encode(payload, secret_key, algorithm=algorithm)


@pytest.fixture
def app_with_auth(secret_key: str, algorithm: str) -> FastAPI:
    """Create a FastAPI app with JWT auth middleware."""
    app = FastAPI()
    app.add_middleware(
        JWTAuthMiddleware,
        secret_key=secret_key,
        algorithm=algorithm,
    )

    @app.get("/protected")
    async def protected_endpoint(request: Request) -> Dict[str, Any]:
        return {
            "user_id": getattr(request.state, "user_id", None),
            "tenant_id": getattr(request.state, "tenant_id", None),
            "roles": getattr(request.state, "roles", []),
            "auth_method": getattr(request.state, "auth_method", None),
        }

    @app.get("/health")
    async def health_endpoint() -> Dict[str, str]:
        return {"status": "healthy"}

    return app


@pytest.fixture
def client(app_with_auth: FastAPI) -> TestClient:
    """Create a test client for the app."""
    return TestClient(app_with_auth)


# =============================================================================
# TokenClaims Model Tests
# =============================================================================


class TestTokenClaims:
    """Tests for TokenClaims Pydantic model."""

    def test_valid_claims(self) -> None:
        """Test creating TokenClaims with valid data."""
        claims = TokenClaims(
            sub="user-123",
            tenant_id="tenant-abc",
            roles=["admin"],
            permissions=["read", "write"],
            iat=1234567890,
            exp=1234571490,
        )
        assert claims.sub == "user-123"
        assert claims.tenant_id == "tenant-abc"
        assert claims.roles == ["admin"]
        assert claims.permissions == ["read", "write"]

    def test_roles_string_converted_to_list(self) -> None:
        """Test that string roles are converted to list."""
        claims = TokenClaims(
            sub="user-123",
            tenant_id="tenant-abc",
            roles="admin",  # type: ignore
        )
        assert claims.roles == ["admin"]

    def test_roles_none_converted_to_empty_list(self) -> None:
        """Test that None roles are converted to empty list."""
        claims = TokenClaims(
            sub="user-123",
            tenant_id="tenant-abc",
            roles=None,  # type: ignore
        )
        assert claims.roles == []

    def test_default_token_type(self) -> None:
        """Test default token type is 'access'."""
        claims = TokenClaims(
            sub="user-123",
            tenant_id="tenant-abc",
        )
        assert claims.token_type == "access"


# =============================================================================
# APIKeyInfo Model Tests
# =============================================================================


class TestAPIKeyInfo:
    """Tests for APIKeyInfo Pydantic model."""

    def test_valid_api_key_info(self) -> None:
        """Test creating APIKeyInfo with valid data."""
        key_info = APIKeyInfo(
            key_id="key-001",
            tenant_id="tenant-abc",
            user_id="service-account-1",
            roles=["api-user"],
            permissions=["read"],
        )
        assert key_info.key_id == "key-001"
        assert key_info.tenant_id == "tenant-abc"
        assert key_info.user_id == "service-account-1"

    def test_api_key_with_expiration(self) -> None:
        """Test APIKeyInfo with expiration date."""
        expires = datetime.now(timezone.utc) + timedelta(days=30)
        key_info = APIKeyInfo(
            key_id="key-001",
            tenant_id="tenant-abc",
            user_id="service-account-1",
            expires_at=expires,
        )
        assert key_info.expires_at == expires


# =============================================================================
# APIKeyStore Tests
# =============================================================================


class TestAPIKeyStore:
    """Tests for APIKeyStore functionality."""

    @pytest.fixture
    def api_key_store(self) -> APIKeyStore:
        """Create an API key store instance."""
        return APIKeyStore()

    def test_validate_key_format_valid(self, api_key_store: APIKeyStore) -> None:
        """Test valid API key format validation."""
        valid_keys = [
            "gl_dev_00000000000000000000000000000001",
            "gl_prod_abcdefghijklmnopqrstuvwxyz123456",
            "gl_staging_ABCDEFGHIJKLMNOPQRSTUVWXYZ1234",
            "gl_test_0123456789abcdefABCDEF0123456789",
        ]
        for key in valid_keys:
            assert api_key_store._validate_key_format(key) is True

    def test_validate_key_format_invalid(self, api_key_store: APIKeyStore) -> None:
        """Test invalid API key format rejection."""
        invalid_keys = [
            "invalid-key",
            "gl_invalid_00000000000000000000000000000001",  # Wrong env
            "gl_dev_short",  # Too short
            "gl_dev_00000000000000000000000000000001extra",  # Too long
            "sk_dev_00000000000000000000000000000001",  # Wrong prefix
            "",
        ]
        for key in invalid_keys:
            assert api_key_store._validate_key_format(key) is False

    @pytest.mark.asyncio
    async def test_validate_dev_api_key(self, api_key_store: APIKeyStore) -> None:
        """Test validating the development API key."""
        key_info = await api_key_store.validate_api_key(
            "gl_dev_00000000000000000000000000000001"
        )
        assert key_info is not None
        assert key_info.key_id == "dev-key-001"
        assert key_info.tenant_id == "default"
        assert "developer" in key_info.roles

    @pytest.mark.asyncio
    async def test_validate_invalid_api_key(self, api_key_store: APIKeyStore) -> None:
        """Test validating an invalid API key."""
        key_info = await api_key_store.validate_api_key(
            "gl_dev_99999999999999999999999999999999"
        )
        assert key_info is None

    @pytest.mark.asyncio
    async def test_validate_empty_api_key(self, api_key_store: APIKeyStore) -> None:
        """Test validating an empty API key."""
        key_info = await api_key_store.validate_api_key("")
        assert key_info is None

    def test_register_and_retrieve_api_key(self, api_key_store: APIKeyStore) -> None:
        """Test registering and retrieving a custom API key."""
        custom_key = "gl_test_customkey123456789012345678"
        key_info = APIKeyInfo(
            key_id="custom-key-001",
            tenant_id="custom-tenant",
            user_id="custom-user",
            roles=["custom-role"],
        )
        api_key_store.register_api_key(custom_key, key_info)

        # Retrieve from cache
        key_hash = api_key_store._hash_key(custom_key)
        cached = api_key_store._get_from_cache(key_hash)
        assert cached is not None
        assert cached.key_id == "custom-key-001"


# =============================================================================
# TokenBlacklist Tests
# =============================================================================


class TestTokenBlacklist:
    """Tests for TokenBlacklist functionality."""

    @pytest.fixture
    def blacklist(self) -> TokenBlacklist:
        """Create a token blacklist instance."""
        return TokenBlacklist()

    def test_blacklist_token(self, blacklist: TokenBlacklist) -> None:
        """Test adding a token to the blacklist."""
        jti = "test-jti-001"
        exp = int(time.time()) + 3600  # Expires in 1 hour
        blacklist.blacklist_token(jti, exp)
        assert blacklist.is_blacklisted(jti) is True

    def test_non_blacklisted_token(self, blacklist: TokenBlacklist) -> None:
        """Test checking a non-blacklisted token."""
        assert blacklist.is_blacklisted("non-existent-jti") is False

    def test_expired_blacklist_entry(self, blacklist: TokenBlacklist) -> None:
        """Test that expired blacklist entries are removed."""
        jti = "expired-jti"
        exp = int(time.time()) - 1  # Already expired
        blacklist.blacklist_token(jti, exp)
        assert blacklist.is_blacklisted(jti) is False

    def test_cleanup_expired(self, blacklist: TokenBlacklist) -> None:
        """Test cleaning up expired blacklist entries."""
        current_time = int(time.time())

        # Add expired entries
        blacklist.blacklist_token("expired-1", current_time - 100)
        blacklist.blacklist_token("expired-2", current_time - 50)

        # Add valid entry
        blacklist.blacklist_token("valid-1", current_time + 3600)

        removed = blacklist.cleanup_expired()
        assert removed == 2
        assert blacklist.is_blacklisted("valid-1") is True


# =============================================================================
# JWT Token Utility Tests
# =============================================================================


class TestJWTTokenUtilities:
    """Tests for JWT token utility functions."""

    def test_create_jwt_token(self, secret_key: str, algorithm: str) -> None:
        """Test creating a JWT token."""
        token = create_jwt_token(
            user_id="user-123",
            tenant_id="tenant-abc",
            secret_key=secret_key,
            algorithm=algorithm,
            roles=["admin"],
        )
        assert token is not None
        assert len(token) > 0

        # Verify token can be decoded
        payload = jwt.decode(token, secret_key, algorithms=[algorithm])
        assert payload["sub"] == "user-123"
        assert payload["tenant_id"] == "tenant-abc"
        assert payload["roles"] == ["admin"]

    def test_decode_jwt_token(self, valid_token: str, secret_key: str, algorithm: str) -> None:
        """Test decoding a JWT token."""
        payload = decode_jwt_token(
            token=valid_token,
            secret_key=secret_key,
            algorithm=algorithm,
        )
        assert "sub" in payload
        assert "tenant_id" in payload
        assert "exp" in payload

    def test_decode_expired_token_with_verify(
        self, expired_token: str, secret_key: str, algorithm: str
    ) -> None:
        """Test decoding an expired token with expiration verification."""
        with pytest.raises(Exception):  # ExpiredSignatureError
            decode_jwt_token(
                token=expired_token,
                secret_key=secret_key,
                algorithm=algorithm,
                verify_exp=True,
            )

    def test_decode_expired_token_without_verify(
        self, expired_token: str, secret_key: str, algorithm: str
    ) -> None:
        """Test decoding an expired token without expiration verification."""
        payload = decode_jwt_token(
            token=expired_token,
            secret_key=secret_key,
            algorithm=algorithm,
            verify_exp=False,
        )
        assert payload["sub"] is not None

    def test_token_includes_jti(self, secret_key: str, algorithm: str) -> None:
        """Test that created tokens include JTI claim."""
        token = create_jwt_token(
            user_id="user-123",
            tenant_id="tenant-abc",
            secret_key=secret_key,
        )
        payload = jwt.decode(token, secret_key, algorithms=[algorithm])
        assert "jti" in payload
        assert len(payload["jti"]) > 0


# =============================================================================
# Middleware Integration Tests
# =============================================================================


class TestJWTAuthMiddleware:
    """Integration tests for JWTAuthMiddleware."""

    def test_public_path_no_auth_required(self, client: TestClient) -> None:
        """Test that public paths don't require authentication."""
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "healthy"

    def test_protected_path_requires_auth(self, client: TestClient) -> None:
        """Test that protected paths require authentication."""
        response = client.get("/protected")
        assert response.status_code == 401
        assert "error" in response.json()
        assert response.json()["error"]["code"] == "UNAUTHORIZED"

    def test_valid_jwt_authentication(
        self, client: TestClient, valid_token: str, test_user_id: str, test_tenant_id: str
    ) -> None:
        """Test successful JWT authentication."""
        response = client.get(
            "/protected",
            headers={"Authorization": f"Bearer {valid_token}"},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["user_id"] == test_user_id
        assert data["tenant_id"] == test_tenant_id
        assert data["auth_method"] == "jwt"

    def test_expired_jwt_rejected(self, client: TestClient, expired_token: str) -> None:
        """Test that expired tokens are rejected."""
        response = client.get(
            "/protected",
            headers={"Authorization": f"Bearer {expired_token}"},
        )
        assert response.status_code == 401
        assert response.json()["error"]["code"] == "TOKEN_EXPIRED"
        assert "X-Token-Expired" in response.headers

    def test_invalid_jwt_signature(
        self, client: TestClient, test_user_id: str, test_tenant_id: str
    ) -> None:
        """Test that tokens with invalid signatures are rejected."""
        # Create token with different secret
        token = create_jwt_token(
            user_id=test_user_id,
            tenant_id=test_tenant_id,
            secret_key="wrong-secret-key",
        )
        response = client.get(
            "/protected",
            headers={"Authorization": f"Bearer {token}"},
        )
        assert response.status_code == 401
        assert response.json()["error"]["code"] == "INVALID_TOKEN"

    def test_malformed_authorization_header(self, client: TestClient) -> None:
        """Test handling of malformed Authorization header."""
        response = client.get(
            "/protected",
            headers={"Authorization": "InvalidFormat"},
        )
        assert response.status_code == 401

    def test_empty_bearer_token(self, client: TestClient) -> None:
        """Test handling of empty Bearer token."""
        response = client.get(
            "/protected",
            headers={"Authorization": "Bearer "},
        )
        assert response.status_code == 401

    def test_token_expiring_soon_header(
        self, client: TestClient, token_expiring_soon: str
    ) -> None:
        """Test that expiring-soon tokens trigger warning header."""
        response = client.get(
            "/protected",
            headers={"Authorization": f"Bearer {token_expiring_soon}"},
        )
        assert response.status_code == 200
        assert response.headers.get("X-Token-Expiring-Soon") == "true"

    def test_www_authenticate_header_on_401(self, client: TestClient) -> None:
        """Test that 401 responses include WWW-Authenticate header."""
        response = client.get("/protected")
        assert response.status_code == 401
        assert "WWW-Authenticate" in response.headers


class TestAPIKeyAuthentication:
    """Tests for API key authentication."""

    def test_valid_dev_api_key(self, client: TestClient) -> None:
        """Test authentication with valid development API key."""
        response = client.get(
            "/protected",
            headers={"X-API-Key": "gl_dev_00000000000000000000000000000001"},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["user_id"] == "dev-service-account"
        assert data["tenant_id"] == "default"
        assert data["auth_method"] == "api_key"

    def test_invalid_api_key(self, client: TestClient) -> None:
        """Test rejection of invalid API key."""
        response = client.get(
            "/protected",
            headers={"X-API-Key": "gl_dev_invalid00000000000000000000"},
        )
        assert response.status_code == 401
        assert response.json()["error"]["code"] == "INVALID_API_KEY"

    def test_malformed_api_key(self, client: TestClient) -> None:
        """Test rejection of malformed API key."""
        response = client.get(
            "/protected",
            headers={"X-API-Key": "invalid-format"},
        )
        assert response.status_code == 401

    def test_jwt_takes_precedence_over_api_key(
        self, client: TestClient, valid_token: str, test_user_id: str
    ) -> None:
        """Test that JWT authentication takes precedence over API key."""
        response = client.get(
            "/protected",
            headers={
                "Authorization": f"Bearer {valid_token}",
                "X-API-Key": "gl_dev_00000000000000000000000000000001",
            },
        )
        assert response.status_code == 200
        data = response.json()
        # JWT user should be used, not API key user
        assert data["user_id"] == test_user_id
        assert data["auth_method"] == "jwt"


# =============================================================================
# Error Response Tests
# =============================================================================


class TestErrorResponses:
    """Tests for error response formatting."""

    def test_error_response_structure(self, client: TestClient) -> None:
        """Test that error responses have correct structure."""
        response = client.get("/protected")
        assert response.status_code == 401
        error = response.json()["error"]
        assert "code" in error
        assert "message" in error

    def test_error_code_header(self, client: TestClient) -> None:
        """Test that X-Error-Code header is set."""
        response = client.get("/protected")
        assert "X-Error-Code" in response.headers
        assert response.headers["X-Error-Code"] == "UNAUTHORIZED"


# =============================================================================
# Middleware Configuration Tests
# =============================================================================


class TestMiddlewareConfiguration:
    """Tests for middleware configuration validation."""

    def test_missing_secret_key_raises_error(self) -> None:
        """Test that missing secret key raises ValueError."""
        app = FastAPI()
        with pytest.raises(ValueError, match="secret_key is required"):
            app.add_middleware(
                JWTAuthMiddleware,
                secret_key="",
            )

    def test_unsupported_algorithm_raises_error(self) -> None:
        """Test that unsupported algorithm raises ValueError."""
        app = FastAPI()
        with pytest.raises(ValueError, match="Unsupported algorithm"):
            app.add_middleware(
                JWTAuthMiddleware,
                secret_key="test-secret",
                algorithm="UNSUPPORTED",
            )

    def test_default_secret_warning(self, caplog: pytest.LogCaptureFixture) -> None:
        """Test that using default secret logs a warning."""
        import logging

        with caplog.at_level(logging.WARNING):
            app = FastAPI()
            app.add_middleware(
                JWTAuthMiddleware,
                secret_key="change-me-in-production",
            )
        assert any("CHANGE THIS IN PRODUCTION" in record.message for record in caplog.records)


# =============================================================================
# Token Claims Extraction Tests
# =============================================================================


class TestTokenClaimsExtraction:
    """Tests for token claims extraction and injection."""

    def test_roles_injected_into_request_state(
        self, client: TestClient, valid_token: str
    ) -> None:
        """Test that roles are properly injected into request state."""
        response = client.get(
            "/protected",
            headers={"Authorization": f"Bearer {valid_token}"},
        )
        assert response.status_code == 200
        data = response.json()
        assert "user" in data["roles"]
        assert "developer" in data["roles"]

    def test_token_without_tenant_uses_default(
        self, client: TestClient, secret_key: str, algorithm: str
    ) -> None:
        """Test that tokens without tenant_id use 'default'."""
        current_time = int(time.time())
        payload = {
            "sub": "user-no-tenant",
            "roles": [],
            "iat": current_time,
            "exp": current_time + 3600,
        }
        token = jwt.encode(payload, secret_key, algorithm=algorithm)

        response = client.get(
            "/protected",
            headers={"Authorization": f"Bearer {token}"},
        )
        assert response.status_code == 200
        assert response.json()["tenant_id"] == "default"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
