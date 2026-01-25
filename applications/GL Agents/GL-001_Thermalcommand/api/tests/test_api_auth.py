"""
Tests for GL-001 ThermalCommand API Authentication

Unit tests for authentication, authorization, and RBAC.
"""

import pytest
from datetime import datetime, timedelta
from uuid import uuid4

from api.api_auth import (
    AuthConfig,
    Role,
    Permission,
    ROLE_PERMISSIONS,
    ThermalCommandUser,
    TokenData,
    get_password_hash,
    verify_password,
    create_access_token,
    create_refresh_token,
    decode_token,
    generate_api_key,
    verify_api_key,
    require_permissions,
    require_roles,
)


# =============================================================================
# Configuration Tests
# =============================================================================

class TestAuthConfig:
    """Tests for AuthConfig model."""

    def test_default_config(self):
        """Test creating default auth configuration."""
        config = AuthConfig(
            jwt_secret_key="test-secret-key-at-least-32-characters",
        )
        assert config.jwt_algorithm == "HS256"
        assert config.jwt_access_token_expire_minutes == 30
        assert config.jwt_refresh_token_expire_days == 7

    def test_custom_config(self):
        """Test creating custom auth configuration."""
        config = AuthConfig(
            jwt_secret_key="test-secret-key-at-least-32-characters",
            jwt_algorithm="HS512",
            jwt_access_token_expire_minutes=60,
            mtls_enabled=True,
        )
        assert config.jwt_algorithm == "HS512"
        assert config.jwt_access_token_expire_minutes == 60
        assert config.mtls_enabled is True

    def test_secret_key_minimum_length(self):
        """Test that secret key must be at least 32 characters."""
        with pytest.raises(ValueError):
            AuthConfig(jwt_secret_key="short")


# =============================================================================
# Role and Permission Tests
# =============================================================================

class TestRolePermissions:
    """Tests for role-based permissions."""

    def test_admin_has_all_permissions(self):
        """Test that admin role has all permissions."""
        admin_perms = ROLE_PERMISSIONS[Role.ADMIN]
        assert admin_perms == set(Permission)

    def test_operator_permissions(self):
        """Test operator role permissions."""
        operator_perms = ROLE_PERMISSIONS[Role.OPERATOR]
        assert Permission.DISPATCH_READ in operator_perms
        assert Permission.DISPATCH_WRITE in operator_perms
        assert Permission.DISPATCH_EXECUTE in operator_perms
        assert Permission.ASSET_READ in operator_perms
        assert Permission.ASSET_CONTROL in operator_perms
        assert Permission.ALARM_ACKNOWLEDGE in operator_perms
        # Operator should not have admin permissions
        assert Permission.USER_MANAGE not in operator_perms
        assert Permission.SYSTEM_ADMIN not in operator_perms

    def test_viewer_permissions(self):
        """Test viewer role has only read permissions."""
        viewer_perms = ROLE_PERMISSIONS[Role.VIEWER]
        # Should have read permissions
        assert Permission.DISPATCH_READ in viewer_perms
        assert Permission.ASSET_READ in viewer_perms
        # Should not have write permissions
        assert Permission.DISPATCH_WRITE not in viewer_perms
        assert Permission.ASSET_WRITE not in viewer_perms
        assert Permission.ASSET_CONTROL not in viewer_perms

    def test_analyst_permissions(self):
        """Test analyst role permissions."""
        analyst_perms = ROLE_PERMISSIONS[Role.ANALYST]
        assert Permission.AUDIT_READ in analyst_perms
        assert Permission.KPI_READ in analyst_perms
        # Should not have operational permissions
        assert Permission.DISPATCH_EXECUTE not in analyst_perms


# =============================================================================
# User Model Tests
# =============================================================================

class TestThermalCommandUser:
    """Tests for ThermalCommandUser model."""

    @pytest.fixture
    def operator_user(self):
        """Create operator user for testing."""
        return ThermalCommandUser(
            user_id=uuid4(),
            username="operator1",
            email="operator@example.com",
            tenant_id=uuid4(),
            roles=[Role.OPERATOR],
            created_at=datetime.utcnow(),
        )

    @pytest.fixture
    def admin_user(self):
        """Create admin user for testing."""
        return ThermalCommandUser(
            user_id=uuid4(),
            username="admin1",
            email="admin@example.com",
            tenant_id=uuid4(),
            roles=[Role.ADMIN],
            created_at=datetime.utcnow(),
        )

    @pytest.fixture
    def multi_role_user(self):
        """Create user with multiple roles."""
        return ThermalCommandUser(
            user_id=uuid4(),
            username="engineer1",
            email="engineer@example.com",
            tenant_id=uuid4(),
            roles=[Role.ENGINEER, Role.ANALYST],
            created_at=datetime.utcnow(),
        )

    def test_has_permission_from_role(self, operator_user):
        """Test permission check from role."""
        assert operator_user.has_permission(Permission.DISPATCH_READ)
        assert operator_user.has_permission(Permission.ALARM_ACKNOWLEDGE)
        assert not operator_user.has_permission(Permission.USER_MANAGE)

    def test_has_direct_permission(self):
        """Test direct permission assignment."""
        user = ThermalCommandUser(
            user_id=uuid4(),
            username="special_user",
            email="special@example.com",
            tenant_id=uuid4(),
            roles=[Role.VIEWER],
            permissions={Permission.DISPATCH_EXECUTE},  # Direct permission
            created_at=datetime.utcnow(),
        )
        # Should have direct permission
        assert user.has_permission(Permission.DISPATCH_EXECUTE)
        # Should have role-based permission
        assert user.has_permission(Permission.DISPATCH_READ)

    def test_has_any_permission(self, operator_user):
        """Test checking for any of multiple permissions."""
        assert operator_user.has_any_permission([
            Permission.DISPATCH_READ,
            Permission.USER_MANAGE,
        ])
        assert not operator_user.has_any_permission([
            Permission.USER_MANAGE,
            Permission.SYSTEM_ADMIN,
        ])

    def test_has_all_permissions(self, operator_user):
        """Test checking for all permissions."""
        assert operator_user.has_all_permissions([
            Permission.DISPATCH_READ,
            Permission.DISPATCH_WRITE,
        ])
        assert not operator_user.has_all_permissions([
            Permission.DISPATCH_READ,
            Permission.USER_MANAGE,
        ])

    def test_get_all_permissions(self, multi_role_user):
        """Test getting all permissions from multiple roles."""
        all_perms = multi_role_user.get_all_permissions()
        # Should have permissions from both roles
        assert Permission.CONFIG_WRITE in all_perms  # From ENGINEER
        assert Permission.AUDIT_READ in all_perms  # From ANALYST

    def test_can_access_asset_no_restrictions(self, operator_user):
        """Test asset access with no restrictions."""
        asset_id = uuid4()
        assert operator_user.can_access_asset(asset_id)

    def test_can_access_asset_with_restrictions(self):
        """Test asset access with allowed asset list."""
        allowed_asset = uuid4()
        other_asset = uuid4()

        user = ThermalCommandUser(
            user_id=uuid4(),
            username="scoped_user",
            email="scoped@example.com",
            tenant_id=uuid4(),
            roles=[Role.OPERATOR],
            allowed_asset_ids=[allowed_asset],
            created_at=datetime.utcnow(),
        )

        assert user.can_access_asset(allowed_asset)
        assert not user.can_access_asset(other_asset)


# =============================================================================
# Password Hashing Tests
# =============================================================================

class TestPasswordHashing:
    """Tests for password hashing utilities."""

    def test_hash_and_verify_password(self):
        """Test password hashing and verification."""
        password = "SecurePassword123!"
        hashed = get_password_hash(password)

        assert hashed != password
        assert verify_password(password, hashed)

    def test_wrong_password_fails(self):
        """Test that wrong password fails verification."""
        password = "SecurePassword123!"
        wrong_password = "WrongPassword456!"
        hashed = get_password_hash(password)

        assert not verify_password(wrong_password, hashed)

    def test_hash_is_unique(self):
        """Test that same password produces different hashes."""
        password = "SecurePassword123!"
        hash1 = get_password_hash(password)
        hash2 = get_password_hash(password)

        # bcrypt adds random salt, so hashes should be different
        assert hash1 != hash2
        # But both should verify correctly
        assert verify_password(password, hash1)
        assert verify_password(password, hash2)


# =============================================================================
# JWT Token Tests
# =============================================================================

class TestJWTTokens:
    """Tests for JWT token creation and validation."""

    @pytest.fixture
    def config(self):
        """Create test auth configuration."""
        return AuthConfig(
            jwt_secret_key="test-secret-key-for-jwt-tokens-32chars",
            jwt_access_token_expire_minutes=30,
            jwt_refresh_token_expire_days=7,
        )

    @pytest.fixture
    def test_user(self):
        """Create test user."""
        return ThermalCommandUser(
            user_id=uuid4(),
            username="test_user",
            email="test@example.com",
            tenant_id=uuid4(),
            roles=[Role.OPERATOR],
            created_at=datetime.utcnow(),
        )

    def test_create_access_token(self, config, test_user):
        """Test creating access token."""
        token = create_access_token(test_user, config)

        assert token is not None
        assert len(token) > 0
        assert "." in token  # JWT format

    def test_decode_access_token(self, config, test_user):
        """Test decoding access token."""
        token = create_access_token(test_user, config)
        token_data = decode_token(token, config)

        assert token_data.sub == str(test_user.user_id)
        assert token_data.username == test_user.username
        assert token_data.email == test_user.email
        assert token_data.tenant_id == str(test_user.tenant_id)
        assert Role.OPERATOR.value in token_data.roles

    def test_token_expiration(self, config, test_user):
        """Test token expiration time."""
        token = create_access_token(test_user, config)
        token_data = decode_token(token, config)

        expected_exp = datetime.utcnow() + timedelta(minutes=config.jwt_access_token_expire_minutes)

        # Allow 1 second tolerance
        assert abs((token_data.exp - expected_exp).total_seconds()) < 1

    def test_custom_expiration(self, config, test_user):
        """Test custom token expiration."""
        custom_delta = timedelta(hours=2)
        token = create_access_token(test_user, config, expires_delta=custom_delta)
        token_data = decode_token(token, config)

        expected_exp = datetime.utcnow() + custom_delta
        assert abs((token_data.exp - expected_exp).total_seconds()) < 1

    def test_create_refresh_token(self, config, test_user):
        """Test creating refresh token."""
        token = create_refresh_token(test_user, config)
        token_data = decode_token(token, config)

        assert token_data.scope == "refresh"
        # Refresh token should have longer expiration
        expected_exp = datetime.utcnow() + timedelta(days=config.jwt_refresh_token_expire_days)
        assert abs((token_data.exp - expected_exp).total_seconds()) < 1

    def test_token_with_permissions(self, config):
        """Test token includes direct permissions."""
        user = ThermalCommandUser(
            user_id=uuid4(),
            username="perm_user",
            email="perm@example.com",
            tenant_id=uuid4(),
            roles=[Role.VIEWER],
            permissions={Permission.DISPATCH_EXECUTE},
            created_at=datetime.utcnow(),
        )

        token = create_access_token(user, config)
        token_data = decode_token(token, config)

        assert Permission.DISPATCH_EXECUTE.value in token_data.permissions

    def test_invalid_token_fails(self, config):
        """Test that invalid token fails decoding."""
        from jose import JWTError

        with pytest.raises(JWTError):
            decode_token("invalid.token.here", config)

    def test_wrong_secret_fails(self, test_user):
        """Test that token with wrong secret fails."""
        from jose import JWTError

        config1 = AuthConfig(jwt_secret_key="secret-key-one-at-least-32-chars")
        config2 = AuthConfig(jwt_secret_key="secret-key-two-at-least-32-chars")

        token = create_access_token(test_user, config1)

        with pytest.raises(JWTError):
            decode_token(token, config2)


# =============================================================================
# API Key Tests
# =============================================================================

class TestAPIKeys:
    """Tests for API key generation and verification."""

    def test_generate_api_key(self):
        """Test API key generation."""
        full_key, key_hash = generate_api_key()

        assert full_key.startswith("tc_")
        assert len(full_key) > 40  # Prefix + random part
        assert len(key_hash) == 64  # SHA256 hex digest

    def test_custom_prefix(self):
        """Test API key with custom prefix."""
        full_key, _ = generate_api_key(prefix="custom_")
        assert full_key.startswith("custom_")

    def test_verify_api_key(self):
        """Test API key verification."""
        full_key, key_hash = generate_api_key()

        assert verify_api_key(full_key, key_hash)

    def test_wrong_key_fails(self):
        """Test that wrong key fails verification."""
        _, key_hash = generate_api_key()
        wrong_key, _ = generate_api_key()

        assert not verify_api_key(wrong_key, key_hash)

    def test_keys_are_unique(self):
        """Test that generated keys are unique."""
        key1, _ = generate_api_key()
        key2, _ = generate_api_key()

        assert key1 != key2


# =============================================================================
# Permission Dependency Tests
# =============================================================================

class TestPermissionDependencies:
    """Tests for permission checking dependencies."""

    @pytest.fixture
    def operator_user(self):
        """Create operator user."""
        return ThermalCommandUser(
            user_id=uuid4(),
            username="operator",
            email="operator@example.com",
            tenant_id=uuid4(),
            roles=[Role.OPERATOR],
            created_at=datetime.utcnow(),
        )

    @pytest.fixture
    def viewer_user(self):
        """Create viewer user."""
        return ThermalCommandUser(
            user_id=uuid4(),
            username="viewer",
            email="viewer@example.com",
            tenant_id=uuid4(),
            roles=[Role.VIEWER],
            created_at=datetime.utcnow(),
        )

    def test_user_permissions_correctly_set(self, operator_user, viewer_user):
        """Test that user permissions are correctly configured."""
        # Operator should have execute permissions
        assert operator_user.has_permission(Permission.DISPATCH_EXECUTE)
        # Viewer should not
        assert not viewer_user.has_permission(Permission.DISPATCH_EXECUTE)


# =============================================================================
# Token Data Tests
# =============================================================================

class TestTokenData:
    """Tests for TokenData model."""

    def test_token_data_creation(self):
        """Test creating token data."""
        now = datetime.utcnow()
        token_data = TokenData(
            sub="user-123",
            username="testuser",
            email="test@example.com",
            tenant_id="tenant-456",
            roles=["operator"],
            permissions=["dispatch:read"],
            exp=now + timedelta(hours=1),
            iat=now,
            scope="access",
        )

        assert token_data.sub == "user-123"
        assert "operator" in token_data.roles
        assert "dispatch:read" in token_data.permissions
        assert token_data.scope == "access"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
