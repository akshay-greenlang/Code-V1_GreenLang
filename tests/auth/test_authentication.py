# -*- coding: utf-8 -*-
"""
Unit Tests for Authentication

Tests JWT authentication, API key authentication, and OAuth2.
"""

import pytest
from unittest.mock import Mock, patch
from datetime import datetime, timedelta
import jwt


class TestJWTAuthentication:
    """Test JWT token-based authentication"""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Setup test auth"""
        try:
            from greenlang.auth.jwt_auth import JWTAuth
            self.auth = JWTAuth(secret="test-secret-key")
        except ImportError:
            pytest.skip("Auth module not available")

    def test_create_access_token(self):
        """Test creating JWT access token"""
        user_data = {
            "user_id": "user-123",
            "tenant_id": "tenant-abc",
            "email": "test@example.com"
        }

        token = self.auth.create_access_token(user_data)

        assert token is not None
        assert isinstance(token, str)
        assert len(token) > 50  # JWT tokens are long

    def test_verify_valid_token(self):
        """Test verifying valid JWT token"""
        user_data = {"user_id": "user-123"}

        token = self.auth.create_access_token(user_data)
        decoded = self.auth.verify_token(token)

        assert decoded["user_id"] == "user-123"

    def test_verify_expired_token(self):
        """Test verifying expired token fails"""
        # Create token that expires immediately
        token = self.auth.create_access_token(
            {"user_id": "user-123"},
            expires_delta=timedelta(seconds=-1)
        )

        with pytest.raises(jwt.ExpiredSignatureError):
            self.auth.verify_token(token)

    def test_verify_invalid_token(self):
        """Test verifying invalid token fails"""
        invalid_token = "invalid.token.here"

        with pytest.raises(jwt.InvalidTokenError):
            self.auth.verify_token(invalid_token)

    def test_token_contains_claims(self):
        """Test token contains expected claims"""
        user_data = {
            "user_id": "user-123",
            "tenant_id": "tenant-abc",
            "roles": ["admin", "user"]
        }

        token = self.auth.create_access_token(user_data)
        decoded = self.auth.verify_token(token)

        assert decoded["user_id"] == "user-123"
        assert decoded["tenant_id"] == "tenant-abc"
        assert decoded["roles"] == ["admin", "user"]


class TestAPIKeyAuthentication:
    """Test API key authentication"""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Setup API key auth"""
        try:
            from greenlang.auth.api_key_auth import APIKeyAuth
            self.auth = APIKeyAuth()
        except ImportError:
            pytest.skip("API key auth not available")

    def test_generate_api_key(self):
        """Test generating API key"""
        api_key = self.auth.generate_api_key("user-123")

        assert api_key is not None
        assert len(api_key) >= 32  # Minimum key length

    def test_validate_valid_api_key(self):
        """Test validating valid API key"""
        api_key = self.auth.generate_api_key("user-123")

        is_valid = self.auth.validate_api_key(api_key)

        assert is_valid is True

    def test_validate_invalid_api_key(self):
        """Test validating invalid API key"""
        invalid_key = "invalid-key-12345"

        is_valid = self.auth.validate_api_key(invalid_key)

        assert is_valid is False

    def test_revoke_api_key(self):
        """Test revoking API key"""
        api_key = self.auth.generate_api_key("user-123")

        # Should be valid initially
        assert self.auth.validate_api_key(api_key) is True

        # Revoke key
        self.auth.revoke_api_key(api_key)

        # Should be invalid after revocation
        assert self.auth.validate_api_key(api_key) is False


class TestAuthorization:
    """Test role-based authorization"""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Setup authorization"""
        try:
            from greenlang.auth.authorization import Authorization
            self.authz = Authorization()
        except ImportError:
            pytest.skip("Authorization module not available")

    def test_check_permission_admin(self):
        """Test admin has all permissions"""
        user = {
            "user_id": "user-123",
            "roles": ["admin"]
        }

        assert self.authz.has_permission(user, "calculate") is True
        assert self.authz.has_permission(user, "view_data") is True
        assert self.authz.has_permission(user, "delete") is True

    def test_check_permission_read_only(self):
        """Test read-only user has limited permissions"""
        user = {
            "user_id": "user-456",
            "roles": ["viewer"]
        }

        assert self.authz.has_permission(user, "view_data") is True
        assert self.authz.has_permission(user, "calculate") is False
        assert self.authz.has_permission(user, "delete") is False

    def test_multi_tenant_isolation(self):
        """Test tenant isolation in authorization"""
        user1 = {"user_id": "user-1", "tenant_id": "tenant-a"}
        user2 = {"user_id": "user-2", "tenant_id": "tenant-b"}

        resource = {"tenant_id": "tenant-a"}

        # User 1 should have access
        assert self.authz.can_access_resource(user1, resource) is True

        # User 2 should NOT have access (different tenant)
        assert self.authz.can_access_resource(user2, resource) is False


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
