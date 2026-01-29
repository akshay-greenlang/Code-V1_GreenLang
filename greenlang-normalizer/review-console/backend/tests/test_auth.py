"""
Tests for Review Console authentication.

This module contains unit tests for JWT token creation, validation,
and role-based access control.
"""

import pytest
from datetime import datetime, timezone, timedelta

from review_console.api.auth import (
    create_access_token,
    decode_token,
    Role,
    User,
    TokenData,
)
from review_console.config import get_settings

settings = get_settings()


class TestTokenCreation:
    """Tests for JWT token creation."""

    def test_create_basic_token(self):
        """Test creating a basic JWT token."""
        token = create_access_token(
            user_id="user-123",
            email="user@test.com",
        )

        assert token is not None
        assert isinstance(token, str)
        assert len(token) > 0

    def test_create_token_with_roles(self):
        """Test creating token with roles."""
        token = create_access_token(
            user_id="user-123",
            email="user@test.com",
            roles=["reviewer", "admin"],
        )

        token_data = decode_token(token)
        assert "reviewer" in token_data.roles
        assert "admin" in token_data.roles

    def test_create_token_with_expiration(self):
        """Test creating token with custom expiration."""
        expires_delta = timedelta(hours=2)
        token = create_access_token(
            user_id="user-123",
            email="user@test.com",
            expires_delta=expires_delta,
        )

        token_data = decode_token(token)
        now = datetime.now(timezone.utc)

        # Token should expire in approximately 2 hours
        time_diff = (token_data.exp - now).total_seconds()
        assert 7100 < time_diff < 7300  # Allow some tolerance

    def test_create_token_with_org_id(self):
        """Test creating token with organization ID."""
        token = create_access_token(
            user_id="user-123",
            email="user@test.com",
            org_id="org-456",
        )

        token_data = decode_token(token)
        assert token_data.org_id == "org-456"


class TestTokenDecoding:
    """Tests for JWT token decoding."""

    def test_decode_valid_token(self):
        """Test decoding a valid token."""
        token = create_access_token(
            user_id="user-123",
            email="user@test.com",
            name="Test User",
            roles=["reviewer"],
        )

        token_data = decode_token(token)

        assert token_data.sub == "user-123"
        assert token_data.email == "user@test.com"
        assert token_data.name == "Test User"
        assert "reviewer" in token_data.roles

    def test_decode_invalid_token(self):
        """Test decoding an invalid token."""
        from fastapi import HTTPException

        with pytest.raises(HTTPException) as exc_info:
            decode_token("invalid-token")

        assert exc_info.value.status_code == 401

    def test_decode_expired_token(self):
        """Test decoding an expired token."""
        from fastapi import HTTPException

        # Create token that expires immediately
        token = create_access_token(
            user_id="user-123",
            email="user@test.com",
            expires_delta=timedelta(seconds=-1),
        )

        with pytest.raises(HTTPException) as exc_info:
            decode_token(token)

        assert exc_info.value.status_code == 401


class TestUserModel:
    """Tests for User model."""

    def test_user_has_role(self):
        """Test checking if user has a role."""
        user = User(
            id="user-123",
            email="user@test.com",
            roles=[Role.REVIEWER],
        )

        assert user.has_role(Role.REVIEWER) is True
        assert user.has_role(Role.ADMIN) is False
        assert user.has_role(Role.VIEWER) is False

    def test_admin_has_all_roles(self):
        """Test that admin has access to all roles."""
        user = User(
            id="admin-123",
            email="admin@test.com",
            roles=[Role.ADMIN],
        )

        assert user.has_role(Role.ADMIN) is True
        assert user.has_role(Role.REVIEWER) is True
        assert user.has_role(Role.VIEWER) is True

    def test_can_review(self):
        """Test can_review method."""
        reviewer = User(
            id="user-123",
            email="user@test.com",
            roles=[Role.REVIEWER],
        )
        viewer = User(
            id="user-456",
            email="viewer@test.com",
            roles=[Role.VIEWER],
        )

        assert reviewer.can_review() is True
        assert viewer.can_review() is False

    def test_can_admin(self):
        """Test can_admin method."""
        admin = User(
            id="admin-123",
            email="admin@test.com",
            roles=[Role.ADMIN],
        )
        reviewer = User(
            id="user-123",
            email="user@test.com",
            roles=[Role.REVIEWER],
        )

        assert admin.can_admin() is True
        assert reviewer.can_admin() is False


class TestRoles:
    """Tests for Role enum."""

    def test_role_values(self):
        """Test role enum values."""
        assert Role.ADMIN.value == "admin"
        assert Role.REVIEWER.value == "reviewer"
        assert Role.VIEWER.value == "viewer"
        assert Role.API.value == "api"

    def test_role_from_string(self):
        """Test creating role from string."""
        role = Role("reviewer")
        assert role == Role.REVIEWER

    def test_invalid_role_string(self):
        """Test creating role from invalid string."""
        with pytest.raises(ValueError):
            Role("invalid_role")
