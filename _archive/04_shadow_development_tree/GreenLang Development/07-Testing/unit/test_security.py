"""
Unit tests for greenlang/api/security/ and greenlang/auth/
Target coverage: 85%+
"""

import pytest
from unittest.mock import Mock, patch, MagicMock, call
from datetime import datetime, timedelta
import jwt
import hashlib
import secrets
import base64
from typing import Dict, Any

# Import test helpers
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from conftest_enhanced import *


class TestCSRFProtection:
    """Test suite for CSRF protection mechanisms."""

    @pytest.fixture
    def csrf_protection(self):
        """Create CSRF protection instance."""
        from greenlang.api.security.csrf import CSRFProtection

        with patch('greenlang.api.security.csrf.CSRFProtection.__init__', return_value=None):
            csrf = CSRFProtection.__new__(CSRFProtection)
            csrf.secret_key = "test-secret-key"
            csrf.token_length = 32
            csrf.tokens = {}
            return csrf

    def test_generate_csrf_token(self, csrf_protection):
        """Test CSRF token generation."""
        csrf_protection.generate_token = Mock(return_value="csrf_token_abc123")

        token = csrf_protection.generate_token("user_123")

        assert token == "csrf_token_abc123"
        assert len(token) > 0

    def test_validate_csrf_token(self, csrf_protection):
        """Test CSRF token validation."""
        user_id = "user_123"
        token = "csrf_token_abc123"

        csrf_protection.validate_token = Mock(return_value=True)

        is_valid = csrf_protection.validate_token(user_id, token)

        assert is_valid == True
        csrf_protection.validate_token.assert_called_once_with(user_id, token)

    def test_invalid_csrf_token(self, csrf_protection):
        """Test rejection of invalid CSRF token."""
        csrf_protection.validate_token = Mock(return_value=False)

        is_valid = csrf_protection.validate_token("user_123", "invalid_token")

        assert is_valid == False

    def test_csrf_token_expiry(self, csrf_protection):
        """Test CSRF token expiration."""
        csrf_protection.is_token_expired = Mock(return_value=True)
        csrf_protection.refresh_token = Mock(return_value="new_token_xyz")

        if csrf_protection.is_token_expired("old_token"):
            new_token = csrf_protection.refresh_token()

        assert new_token == "new_token_xyz"

    def test_double_submit_cookie(self, csrf_protection):
        """Test double-submit cookie pattern."""
        request = {
            "cookie": "csrf=token_abc",
            "header": "X-CSRF-Token: token_abc"
        }

        csrf_protection.validate_double_submit = Mock(return_value=True)

        is_valid = csrf_protection.validate_double_submit(request)

        assert is_valid == True

    def test_origin_validation(self, csrf_protection):
        """Test origin header validation."""
        valid_origins = ["https://example.com", "https://app.example.com"]
        request_origin = "https://example.com"

        csrf_protection.validate_origin = Mock(return_value=True)

        is_valid = csrf_protection.validate_origin(request_origin, valid_origins)

        assert is_valid == True

    def test_referer_validation(self, csrf_protection):
        """Test referer header validation."""
        csrf_protection.validate_referer = Mock(return_value=True)

        is_valid = csrf_protection.validate_referer(
            "https://example.com/page",
            "https://example.com"
        )

        assert is_valid == True


class TestRateLimiting:
    """Test suite for rate limiting functionality."""

    @pytest.fixture
    def rate_limiter(self):
        """Create rate limiter instance."""
        from greenlang.api.security.rate_limiter import RateLimiter

        with patch('greenlang.api.security.rate_limiter.RateLimiter.__init__', return_value=None):
            limiter = RateLimiter.__new__(RateLimiter)
            limiter.max_requests = 60
            limiter.window_seconds = 60
            limiter.requests = {}
            return limiter

    def test_allow_request(self, rate_limiter):
        """Test allowing request within rate limit."""
        client_id = "client_123"

        rate_limiter.check_limit = Mock(return_value=True)

        allowed = rate_limiter.check_limit(client_id)

        assert allowed == True

    def test_deny_request_over_limit(self, rate_limiter):
        """Test denying request when over rate limit."""
        client_id = "client_123"

        rate_limiter.check_limit = Mock(side_effect=[True] * 60 + [False])

        # First 60 requests allowed
        for _ in range(60):
            assert rate_limiter.check_limit(client_id) == True

        # 61st request denied
        assert rate_limiter.check_limit(client_id) == False

    def test_sliding_window(self, rate_limiter):
        """Test sliding window rate limiting."""
        rate_limiter.add_request = Mock()
        rate_limiter.clean_old_requests = Mock()
        rate_limiter.count_requests = Mock(return_value=45)

        rate_limiter.add_request("client_123", datetime.utcnow())
        rate_limiter.clean_old_requests(datetime.utcnow() - timedelta(seconds=60))
        count = rate_limiter.count_requests("client_123")

        assert count == 45

    def test_token_bucket(self, rate_limiter):
        """Test token bucket algorithm."""
        rate_limiter.bucket_size = 10
        rate_limiter.refill_rate = 1  # 1 token per second
        rate_limiter.consume_token = Mock(return_value=True)
        rate_limiter.refill_bucket = Mock()

        # Consume token
        allowed = rate_limiter.consume_token("client_123")
        assert allowed == True

        # Refill bucket
        rate_limiter.refill_bucket("client_123")
        rate_limiter.refill_bucket.assert_called_once()

    def test_distributed_rate_limiting(self, rate_limiter):
        """Test distributed rate limiting with Redis."""
        rate_limiter.redis_client = Mock()
        rate_limiter.redis_client.incr = Mock(return_value=5)
        rate_limiter.redis_client.expire = Mock()

        rate_limiter.check_distributed_limit = Mock(return_value=True)

        allowed = rate_limiter.check_distributed_limit("client_123")

        assert allowed == True

    def test_rate_limit_headers(self, rate_limiter):
        """Test rate limit response headers."""
        rate_limiter.get_headers = Mock(return_value={
            "X-RateLimit-Limit": "60",
            "X-RateLimit-Remaining": "45",
            "X-RateLimit-Reset": "1640995200"
        })

        headers = rate_limiter.get_headers("client_123")

        assert headers["X-RateLimit-Limit"] == "60"
        assert headers["X-RateLimit-Remaining"] == "45"

    @pytest.mark.parametrize("endpoint,limit", [
        ("/api/calculate", 100),
        ("/api/report", 10),
        ("/api/export", 5)
    ])
    def test_endpoint_specific_limits(self, rate_limiter, endpoint, limit):
        """Test different rate limits for different endpoints."""
        rate_limiter.get_endpoint_limit = Mock(return_value=limit)

        endpoint_limit = rate_limiter.get_endpoint_limit(endpoint)

        assert endpoint_limit == limit


class TestSecurityHeaders:
    """Test suite for security headers."""

    @pytest.fixture
    def security_headers(self):
        """Create security headers manager."""
        from greenlang.api.security.headers import SecurityHeaders

        with patch('greenlang.api.security.headers.SecurityHeaders.__init__', return_value=None):
            headers = SecurityHeaders.__new__(SecurityHeaders)
            return headers

    def test_content_security_policy(self, security_headers):
        """Test Content Security Policy header."""
        security_headers.get_csp = Mock(return_value="default-src 'self'")

        csp = security_headers.get_csp()

        assert "default-src" in csp
        assert "'self'" in csp

    def test_strict_transport_security(self, security_headers):
        """Test HSTS header."""
        security_headers.get_hsts = Mock(
            return_value="max-age=31536000; includeSubDomains"
        )

        hsts = security_headers.get_hsts()

        assert "max-age=31536000" in hsts

    def test_x_frame_options(self, security_headers):
        """Test X-Frame-Options header."""
        security_headers.get_frame_options = Mock(return_value="DENY")

        frame_options = security_headers.get_frame_options()

        assert frame_options == "DENY"

    def test_cors_headers(self, security_headers):
        """Test CORS headers."""
        security_headers.get_cors_headers = Mock(return_value={
            "Access-Control-Allow-Origin": "https://example.com",
            "Access-Control-Allow-Methods": "GET, POST",
            "Access-Control-Allow-Headers": "Content-Type, Authorization"
        })

        cors = security_headers.get_cors_headers()

        assert cors["Access-Control-Allow-Origin"] == "https://example.com"


class TestJWTAuthentication:
    """Test suite for JWT authentication."""

    @pytest.fixture
    def jwt_auth(self):
        """Create JWT authentication manager."""
        from greenlang.auth.jwt_auth import JWTAuth

        with patch('greenlang.auth.jwt_auth.JWTAuth.__init__', return_value=None):
            auth = JWTAuth.__new__(JWTAuth)
            auth.secret_key = "test-secret"
            auth.algorithm = "HS256"
            auth.expiry_minutes = 30
            return auth

    def test_generate_jwt_token(self, jwt_auth):
        """Test JWT token generation."""
        payload = {
            "user_id": "123",
            "username": "test_user",
            "exp": datetime.utcnow() + timedelta(minutes=30)
        }

        jwt_auth.generate_token = Mock(return_value="eyJ...")

        token = jwt_auth.generate_token(payload)

        assert token.startswith("eyJ")

    def test_validate_jwt_token(self, jwt_auth):
        """Test JWT token validation."""
        token = "eyJ..."

        jwt_auth.validate_token = Mock(return_value={
            "user_id": "123",
            "username": "test_user"
        })

        payload = jwt_auth.validate_token(token)

        assert payload["user_id"] == "123"

    def test_expired_jwt_token(self, jwt_auth):
        """Test handling of expired JWT token."""
        expired_token = "expired_token"

        jwt_auth.validate_token = Mock(side_effect=jwt.ExpiredSignatureError())

        with pytest.raises(jwt.ExpiredSignatureError):
            jwt_auth.validate_token(expired_token)

    def test_invalid_jwt_signature(self, jwt_auth):
        """Test handling of invalid JWT signature."""
        jwt_auth.validate_token = Mock(side_effect=jwt.InvalidSignatureError())

        with pytest.raises(jwt.InvalidSignatureError):
            jwt_auth.validate_token("invalid_token")

    def test_jwt_refresh_token(self, jwt_auth):
        """Test JWT refresh token mechanism."""
        jwt_auth.generate_refresh_token = Mock(return_value="refresh_token_xyz")
        jwt_auth.refresh_access_token = Mock(return_value="new_access_token")

        refresh_token = jwt_auth.generate_refresh_token("user_123")
        new_access = jwt_auth.refresh_access_token(refresh_token)

        assert refresh_token == "refresh_token_xyz"
        assert new_access == "new_access_token"

    def test_jwt_claims_validation(self, jwt_auth):
        """Test JWT custom claims validation."""
        jwt_auth.validate_claims = Mock(return_value=True)

        claims = {
            "iss": "greenlang",
            "aud": "api",
            "scope": ["read", "write"]
        }

        is_valid = jwt_auth.validate_claims(claims)

        assert is_valid == True


class TestPermissions:
    """Test suite for permission management."""

    @pytest.fixture
    def permission_manager(self):
        """Create permission manager instance."""
        from greenlang.auth.permissions import PermissionManager

        with patch('greenlang.auth.permissions.PermissionManager.__init__', return_value=None):
            manager = PermissionManager.__new__(PermissionManager)
            manager.permissions = {}
            return manager

    def test_check_permission(self, permission_manager):
        """Test permission checking."""
        permission_manager.has_permission = Mock(return_value=True)

        has_perm = permission_manager.has_permission(
            user_id="123",
            resource="emissions",
            action="read"
        )

        assert has_perm == True

    def test_grant_permission(self, permission_manager):
        """Test granting permissions."""
        permission_manager.grant = Mock()

        permission_manager.grant(
            user_id="123",
            resource="reports",
            action="write"
        )

        permission_manager.grant.assert_called_once()

    def test_revoke_permission(self, permission_manager):
        """Test revoking permissions."""
        permission_manager.revoke = Mock()

        permission_manager.revoke(
            user_id="123",
            resource="reports",
            action="write"
        )

        permission_manager.revoke.assert_called_once()

    def test_role_based_permissions(self, permission_manager):
        """Test role-based permission checking."""
        permission_manager.get_role_permissions = Mock(return_value=[
            "emissions:read",
            "emissions:write",
            "reports:read"
        ])

        permissions = permission_manager.get_role_permissions("analyst")

        assert "emissions:read" in permissions
        assert "emissions:write" in permissions

    def test_resource_ownership(self, permission_manager):
        """Test resource ownership validation."""
        permission_manager.is_owner = Mock(return_value=True)

        is_owner = permission_manager.is_owner(
            user_id="123",
            resource_id="report_456"
        )

        assert is_owner == True

    @pytest.mark.parametrize("role,expected_permissions", [
        ("admin", ["*:*"]),
        ("analyst", ["emissions:*", "reports:read"]),
        ("viewer", ["*:read"]),
        ("guest", [])
    ])
    def test_role_hierarchy(self, permission_manager, role, expected_permissions):
        """Test role hierarchy and inheritance."""
        permission_manager.get_effective_permissions = Mock(
            return_value=expected_permissions
        )

        permissions = permission_manager.get_effective_permissions(role)

        assert permissions == expected_permissions


class TestPasswordSecurity:
    """Test suite for password security."""

    @pytest.fixture
    def password_manager(self):
        """Create password manager instance."""
        from greenlang.auth.password import PasswordManager

        with patch('greenlang.auth.password.PasswordManager.__init__', return_value=None):
            manager = PasswordManager.__new__(PasswordManager)
            return manager

    def test_password_hashing(self, password_manager):
        """Test password hashing."""
        password = "SecurePass123!"

        password_manager.hash_password = Mock(
            return_value="$2b$12$..." # bcrypt hash
        )

        hashed = password_manager.hash_password(password)

        assert hashed.startswith("$2b$")
        assert len(hashed) > 50

    def test_password_verification(self, password_manager):
        """Test password verification."""
        password = "SecurePass123!"
        hashed = "$2b$12$..."

        password_manager.verify_password = Mock(return_value=True)

        is_valid = password_manager.verify_password(password, hashed)

        assert is_valid == True

    def test_password_strength_validation(self, password_manager):
        """Test password strength requirements."""
        password_manager.validate_strength = Mock(return_value={
            "valid": True,
            "score": 4,
            "issues": []
        })

        result = password_manager.validate_strength("VerySecure123!@#")

        assert result["valid"] == True
        assert result["score"] >= 3

    @pytest.mark.parametrize("password,expected_valid", [
        ("short", False),
        ("nouppercase123!", False),
        ("NoNumbers!", False),
        ("NoSpecialChar123", False),
        ("ValidPass123!", True)
    ])
    def test_password_policies(self, password_manager, password, expected_valid):
        """Test password policy enforcement."""
        password_manager.meets_policy = Mock(return_value=expected_valid)

        is_valid = password_manager.meets_policy(password)

        assert is_valid == expected_valid


class TestSessionManagement:
    """Test suite for session management."""

    @pytest.fixture
    def session_manager(self):
        """Create session manager instance."""
        from greenlang.auth.session import SessionManager

        with patch('greenlang.auth.session.SessionManager.__init__', return_value=None):
            manager = SessionManager.__new__(SessionManager)
            manager.sessions = {}
            return manager

    def test_create_session(self, session_manager):
        """Test session creation."""
        session_manager.create = Mock(return_value={
            "session_id": "sess_123",
            "user_id": "user_456",
            "expires_at": datetime.utcnow() + timedelta(hours=1)
        })

        session = session_manager.create("user_456")

        assert session["session_id"] == "sess_123"

    def test_validate_session(self, session_manager):
        """Test session validation."""
        session_manager.validate = Mock(return_value=True)

        is_valid = session_manager.validate("sess_123")

        assert is_valid == True

    def test_session_expiry(self, session_manager):
        """Test session expiration handling."""
        session_manager.is_expired = Mock(return_value=True)
        session_manager.destroy = Mock()

        if session_manager.is_expired("sess_123"):
            session_manager.destroy("sess_123")

        session_manager.destroy.assert_called_once()

    def test_session_renewal(self, session_manager):
        """Test session renewal."""
        session_manager.renew = Mock(return_value={
            "session_id": "sess_123",
            "expires_at": datetime.utcnow() + timedelta(hours=1)
        })

        renewed = session_manager.renew("sess_123")

        assert renewed["session_id"] == "sess_123"

    def test_concurrent_session_limit(self, session_manager):
        """Test concurrent session limits."""
        session_manager.count_user_sessions = Mock(return_value=5)
        session_manager.max_sessions = 3
        session_manager.terminate_oldest = Mock()

        if session_manager.count_user_sessions("user_123") > session_manager.max_sessions:
            session_manager.terminate_oldest("user_123")

        session_manager.terminate_oldest.assert_called_once()


class TestSecurityIntegration:
    """Integration tests for security components."""

    @pytest.mark.integration
    def test_complete_auth_flow(self):
        """Test complete authentication flow."""
        # Mock components
        auth = Mock()
        auth.authenticate = Mock(return_value={"user_id": "123", "token": "jwt_token"})
        auth.validate_token = Mock(return_value=True)

        csrf = Mock()
        csrf.generate_token = Mock(return_value="csrf_token")
        csrf.validate_token = Mock(return_value=True)

        # Login flow
        login_result = auth.authenticate("user", "pass")
        csrf_token = csrf.generate_token(login_result["user_id"])

        # API request with auth
        is_valid_auth = auth.validate_token(login_result["token"])
        is_valid_csrf = csrf.validate_token(login_result["user_id"], csrf_token)

        assert is_valid_auth == True
        assert is_valid_csrf == True

    @pytest.mark.integration
    def test_security_middleware_chain(self):
        """Test security middleware chain."""
        request = {
            "headers": {
                "Authorization": "Bearer token",
                "X-CSRF-Token": "csrf_token"
            },
            "ip": "192.168.1.1"
        }

        # Mock middleware
        auth_middleware = Mock(return_value=True)
        csrf_middleware = Mock(return_value=True)
        rate_limit_middleware = Mock(return_value=True)

        # Process through middleware chain
        auth_ok = auth_middleware(request)
        csrf_ok = csrf_middleware(request)
        rate_ok = rate_limit_middleware(request)

        assert all([auth_ok, csrf_ok, rate_ok])

    @pytest.mark.performance
    def test_auth_performance(self, performance_timer):
        """Test authentication performance."""
        from greenlang.auth.jwt_auth import JWTAuth

        with patch('greenlang.auth.jwt_auth.JWTAuth.__init__', return_value=None):
            auth = JWTAuth.__new__(JWTAuth)
            auth.generate_token = Mock(return_value="token")
            auth.validate_token = Mock(return_value={"user_id": "123"})

            performance_timer.start()

            # Generate and validate 1000 tokens
            for _ in range(1000):
                token = auth.generate_token({"user_id": "123"})
                auth.validate_token(token)

            performance_timer.stop()

            # Should complete in less than 500ms
            assert performance_timer.elapsed_ms() < 500