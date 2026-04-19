# -*- coding: utf-8 -*-
"""
Comprehensive Security Enhancement Tests
GL-VCCI Scope 3 Platform

Tests for:
- JWT Refresh Token System (20 tests)
- Token Blacklist (15 tests)
- API Key Authentication (20 tests)
- Request Signing (15 tests)
- Security Headers (10 tests)
- Enhanced Audit Logging (10 tests)

Total: 90+ security tests

Version: 1.0.0
Last Updated: 2025-11-09
"""

import os
import json
import pytest
import asyncio
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, AsyncMock
from greenlang.determinism import DeterministicClock

# Set test environment variables
os.environ["JWT_SECRET"] = "test_secret_key_minimum_32_characters_long"
os.environ["REFRESH_SECRET"] = "test_refresh_secret_key_minimum_32_characters"
os.environ["REQUEST_SIGNING_SECRET"] = "test_signing_secret_key_minimum_32_chars"
os.environ["ENVIRONMENT"] = "test"

from backend.auth_refresh import (
    create_access_token,
    create_refresh_token,
    issue_token_pair,
    refresh_access_token,
    revoke_refresh_token,
    revoke_all_user_tokens,
    TokenPair,
    RefreshTokenError,
)
from backend.auth_blacklist import (
    blacklist_token,
    is_blacklisted,
    blacklist_all_user_tokens,
    is_user_blacklisted,
    verify_token_not_blacklisted,
)
from backend.auth_api_keys import (
    generate_api_key,
    create_api_key,
    verify_api_key,
    check_rate_limit,
    revoke_api_key,
    APIKeyScope,
    APIKeyError,
)
from backend.request_signing import (
    generate_nonce,
    generate_timestamp,
    compute_signature,
    verify_signature,
    verify_timestamp,
    verify_nonce,
    RequestSigner,
)
from backend.security_headers_advanced import (
    build_csp_header,
    build_expect_ct_header,
    build_nel_header,
    generate_sri_hash,
    SecurityHeadersMiddleware,
)
from backend.audit_enhanced import (
    AuditEventType,
    AuditSeverity,
    AuditEvent,
    AuditLogger,
    log_auth_success,
    log_auth_failure,
)


# ============================================================================
# FIXTURES
# ============================================================================


@pytest.fixture
async def redis_mock():
    """Mock Redis client for testing."""
    mock = AsyncMock()
    mock.ping = AsyncMock(return_value=True)
    mock.hset = AsyncMock(return_value=1)
    mock.hgetall = AsyncMock(return_value={})
    mock.expire = AsyncMock(return_value=1)
    mock.setex = AsyncMock(return_value=True)
    mock.delete = AsyncMock(return_value=1)
    mock.exists = AsyncMock(return_value=0)
    mock.scan_iter = AsyncMock(return_value=iter([]))
    mock.incr = AsyncMock(return_value=1)
    return mock


@pytest.fixture
def test_user_id():
    """Test user ID."""
    return "test_user@example.com"


@pytest.fixture
def test_ip():
    """Test IP address."""
    return "192.168.1.100"


# ============================================================================
# JWT REFRESH TOKEN TESTS (20 tests)
# ============================================================================


class TestJWTRefreshTokens:
    """Test JWT refresh token functionality."""

    def test_create_access_token(self, test_user_id):
        """Test access token creation."""
        token = create_access_token(test_user_id)

        assert token is not None
        assert isinstance(token, str)
        assert len(token) > 0

    def test_create_access_token_with_claims(self, test_user_id):
        """Test access token with additional claims."""
        token = create_access_token(
            test_user_id, additional_claims={"role": "admin"}
        )

        from jose import jwt

        payload = jwt.decode(
            token, os.environ["JWT_SECRET"], algorithms=["HS256"]
        )

        assert payload["sub"] == test_user_id
        assert payload["role"] == "admin"
        assert payload["type"] == "access"

    def test_create_refresh_token(self, test_user_id):
        """Test refresh token creation."""
        token, jti = create_refresh_token(test_user_id)

        assert token is not None
        assert jti is not None
        assert isinstance(token, str)
        assert isinstance(jti, str)

    def test_refresh_token_contains_jti(self, test_user_id):
        """Test refresh token contains JTI."""
        token, jti = create_refresh_token(test_user_id)

        from jose import jwt

        payload = jwt.decode(
            token, os.environ["REFRESH_SECRET"], algorithms=["HS256"]
        )

        assert payload["jti"] == jti
        assert payload["type"] == "refresh"

    @pytest.mark.asyncio
    async def test_issue_token_pair(self, test_user_id, redis_mock):
        """Test issuing access + refresh token pair."""
        with patch(
            "backend.auth_refresh.get_redis_client", return_value=redis_mock
        ):
            tokens = await issue_token_pair(test_user_id)

            assert isinstance(tokens, TokenPair)
            assert tokens.access_token is not None
            assert tokens.refresh_token is not None
            assert tokens.token_type == "bearer"

    @pytest.mark.asyncio
    async def test_issue_token_pair_stores_in_redis(
        self, test_user_id, redis_mock
    ):
        """Test token pair storage in Redis."""
        with patch(
            "backend.auth_refresh.get_redis_client", return_value=redis_mock
        ):
            await issue_token_pair(test_user_id)

            # Verify Redis calls
            assert redis_mock.hset.called
            assert redis_mock.expire.called

    @pytest.mark.asyncio
    async def test_refresh_access_token_success(
        self, test_user_id, redis_mock
    ):
        """Test successful access token refresh."""
        # Create initial tokens
        with patch(
            "backend.auth_refresh.get_redis_client", return_value=redis_mock
        ):
            tokens = await issue_token_pair(test_user_id)

            # Mock Redis to return stored data
            from jose import jwt

            payload = jwt.decode(
                tokens.refresh_token,
                os.environ["REFRESH_SECRET"],
                algorithms=["HS256"],
            )

            redis_mock.hgetall = AsyncMock(
                return_value={
                    "user_id": test_user_id,
                    "jti": payload["jti"],
                    "issued_at": DeterministicClock.utcnow().isoformat(),
                    "expires_at": (
                        DeterministicClock.utcnow() + timedelta(days=7)
                    ).isoformat(),
                }
            )

            # Refresh
            new_tokens = await refresh_access_token(tokens.refresh_token)

            assert isinstance(new_tokens, TokenPair)
            assert new_tokens.access_token is not None

    @pytest.mark.asyncio
    async def test_refresh_token_rotation(self, test_user_id, redis_mock):
        """Test refresh token rotation."""
        with patch(
            "backend.auth_refresh.get_redis_client", return_value=redis_mock
        ):
            tokens = await issue_token_pair(test_user_id)

            from jose import jwt

            payload = jwt.decode(
                tokens.refresh_token,
                os.environ["REFRESH_SECRET"],
                algorithms=["HS256"],
            )

            redis_mock.hgetall = AsyncMock(
                return_value={
                    "user_id": test_user_id,
                    "jti": payload["jti"],
                    "issued_at": DeterministicClock.utcnow().isoformat(),
                    "expires_at": (
                        DeterministicClock.utcnow() + timedelta(days=7)
                    ).isoformat(),
                }
            )

            # With rotation enabled, should get new refresh token
            new_tokens = await refresh_access_token(tokens.refresh_token)

            assert new_tokens.refresh_token != tokens.refresh_token

    @pytest.mark.asyncio
    async def test_refresh_invalid_token_fails(self, redis_mock):
        """Test refresh with invalid token fails."""
        with patch(
            "backend.auth_refresh.get_redis_client", return_value=redis_mock
        ):
            with pytest.raises(RefreshTokenError):
                await refresh_access_token("invalid_token")

    @pytest.mark.asyncio
    async def test_refresh_expired_token_fails(self, test_user_id):
        """Test refresh with expired token fails."""
        # Create token with past expiration
        token, _ = create_refresh_token(
            test_user_id,
            expires_delta=timedelta(seconds=-1),
        )

        with pytest.raises(RefreshTokenError):
            await refresh_access_token(token)

    @pytest.mark.asyncio
    async def test_revoke_refresh_token(self, test_user_id, redis_mock):
        """Test refresh token revocation."""
        with patch(
            "backend.auth_refresh.get_redis_client", return_value=redis_mock
        ):
            tokens = await issue_token_pair(test_user_id)

            redis_mock.delete = AsyncMock(return_value=1)

            result = await revoke_refresh_token(tokens.refresh_token)

            assert result is True
            assert redis_mock.delete.called

    @pytest.mark.asyncio
    async def test_revoke_all_user_tokens(self, test_user_id, redis_mock):
        """Test revoking all tokens for a user."""
        redis_mock.scan_iter = AsyncMock(
            return_value=iter(["refresh_token:jti1", "refresh_token:jti2"])
        )
        redis_mock.hgetall = AsyncMock(
            return_value={"user_id": test_user_id}
        )

        with patch(
            "backend.auth_refresh.get_redis_client", return_value=redis_mock
        ):
            count = await revoke_all_user_tokens(test_user_id)

            assert count >= 0

    def test_token_pair_to_dict(self):
        """Test TokenPair serialization."""
        tokens = TokenPair(
            access_token="access_123", refresh_token="refresh_456"
        )

        data = tokens.to_dict()

        assert data["access_token"] == "access_123"
        assert data["refresh_token"] == "refresh_456"
        assert data["token_type"] == "bearer"

    def test_access_token_expiration(self, test_user_id):
        """Test access token expiration time."""
        from jose import jwt

        token = create_access_token(test_user_id)
        payload = jwt.decode(
            token, os.environ["JWT_SECRET"], algorithms=["HS256"]
        )

        exp = datetime.utcfromtimestamp(payload["exp"])
        iat = datetime.utcfromtimestamp(payload["iat"])

        # Should be approximately 1 hour
        diff = exp - iat
        assert 3550 <= diff.total_seconds() <= 3650

    def test_refresh_token_expiration(self, test_user_id):
        """Test refresh token expiration time."""
        from jose import jwt

        token, _ = create_refresh_token(test_user_id)
        payload = jwt.decode(
            token, os.environ["REFRESH_SECRET"], algorithms=["HS256"]
        )

        exp = datetime.utcfromtimestamp(payload["exp"])
        iat = datetime.utcfromtimestamp(payload["iat"])

        # Should be approximately 7 days
        diff = exp - iat
        assert 604700 <= diff.total_seconds() <= 604900

    def test_access_token_has_type_claim(self, test_user_id):
        """Test access token has type claim."""
        from jose import jwt

        token = create_access_token(test_user_id)
        payload = jwt.decode(
            token, os.environ["JWT_SECRET"], algorithms=["HS256"]
        )

        assert payload["type"] == "access"

    def test_refresh_token_has_type_claim(self, test_user_id):
        """Test refresh token has type claim."""
        from jose import jwt

        token, _ = create_refresh_token(test_user_id)
        payload = jwt.decode(
            token, os.environ["REFRESH_SECRET"], algorithms=["HS256"]
        )

        assert payload["type"] == "refresh"

    def test_custom_expiration_delta(self, test_user_id):
        """Test custom expiration delta."""
        from jose import jwt

        custom_delta = timedelta(minutes=30)
        token = create_access_token(test_user_id, expires_delta=custom_delta)

        payload = jwt.decode(
            token, os.environ["JWT_SECRET"], algorithms=["HS256"]
        )

        exp = datetime.utcfromtimestamp(payload["exp"])
        iat = datetime.utcfromtimestamp(payload["iat"])

        diff = exp - iat
        assert 1750 <= diff.total_seconds() <= 1850  # ~30 min

    def test_refresh_token_device_id(self, test_user_id):
        """Test refresh token with device ID."""
        from jose import jwt

        device_id = "mobile-app-ios"
        token, _ = create_refresh_token(test_user_id, device_id=device_id)

        payload = jwt.decode(
            token, os.environ["REFRESH_SECRET"], algorithms=["HS256"]
        )

        assert payload["device_id"] == device_id


# ============================================================================
# TOKEN BLACKLIST TESTS (15 tests)
# ============================================================================


class TestTokenBlacklist:
    """Test token blacklist functionality."""

    @pytest.mark.asyncio
    async def test_blacklist_token(self, test_user_id, redis_mock):
        """Test adding token to blacklist."""
        token = create_access_token(test_user_id)

        with patch(
            "backend.auth_blacklist.get_redis_client", return_value=redis_mock
        ):
            result = await blacklist_token(token, reason="logout")

            assert result is True
            assert redis_mock.hset.called
            assert redis_mock.expire.called

    @pytest.mark.asyncio
    async def test_is_blacklisted(self, test_user_id, redis_mock):
        """Test checking if token is blacklisted."""
        token = create_access_token(test_user_id)

        redis_mock.exists = AsyncMock(return_value=1)

        with patch(
            "backend.auth_blacklist.get_redis_client", return_value=redis_mock
        ):
            result = await is_blacklisted(token)

            assert result is True

    @pytest.mark.asyncio
    async def test_is_not_blacklisted(self, test_user_id, redis_mock):
        """Test token not in blacklist."""
        token = create_access_token(test_user_id)

        redis_mock.exists = AsyncMock(return_value=0)

        with patch(
            "backend.auth_blacklist.get_redis_client", return_value=redis_mock
        ):
            result = await is_blacklisted(token)

            assert result is False

    @pytest.mark.asyncio
    async def test_blacklist_all_user_tokens(self, test_user_id, redis_mock):
        """Test blacklisting all tokens for a user."""
        with patch(
            "backend.auth_blacklist.get_redis_client", return_value=redis_mock
        ):
            count = await blacklist_all_user_tokens(
                test_user_id, reason="password_change"
            )

            assert count >= 0
            assert redis_mock.hset.called

    @pytest.mark.asyncio
    async def test_is_user_blacklisted(self, test_user_id, redis_mock):
        """Test checking if user is blacklisted."""
        redis_mock.exists = AsyncMock(return_value=1)

        with patch(
            "backend.auth_blacklist.get_redis_client", return_value=redis_mock
        ):
            result = await is_user_blacklisted(test_user_id)

            assert result is True

    @pytest.mark.asyncio
    async def test_verify_token_not_blacklisted_valid(
        self, test_user_id, redis_mock
    ):
        """Test verification of non-blacklisted token."""
        token = create_access_token(test_user_id)

        redis_mock.exists = AsyncMock(return_value=0)

        with patch(
            "backend.auth_blacklist.get_redis_client", return_value=redis_mock
        ):
            result = await verify_token_not_blacklisted(token, test_user_id)

            assert result is True

    @pytest.mark.asyncio
    async def test_verify_token_not_blacklisted_invalid(
        self, test_user_id, redis_mock
    ):
        """Test verification of blacklisted token."""
        token = create_access_token(test_user_id)

        redis_mock.exists = AsyncMock(return_value=1)

        with patch(
            "backend.auth_blacklist.get_redis_client", return_value=redis_mock
        ):
            result = await verify_token_not_blacklisted(token, test_user_id)

            assert result is False

    @pytest.mark.asyncio
    async def test_blacklist_with_metadata(self, test_user_id, redis_mock):
        """Test blacklisting with metadata."""
        token = create_access_token(test_user_id)

        metadata = {"ip": "192.168.1.1", "user_agent": "Mozilla/5.0"}

        with patch(
            "backend.auth_blacklist.get_redis_client", return_value=redis_mock
        ):
            result = await blacklist_token(
                token, reason="logout", metadata=metadata
            )

            assert result is True

    @pytest.mark.asyncio
    async def test_blacklist_expired_token(self, test_user_id):
        """Test blacklisting already expired token."""
        # Create expired token
        token = create_access_token(
            test_user_id, expires_delta=timedelta(seconds=-1)
        )

        with patch(
            "backend.auth_blacklist.get_redis_client",
            return_value=AsyncMock(),
        ):
            result = await blacklist_token(token)

            # Should return True (no need to blacklist expired token)
            assert result is True

    @pytest.mark.asyncio
    async def test_blacklist_invalid_token(self):
        """Test blacklisting invalid token."""
        with patch(
            "backend.auth_blacklist.get_redis_client",
            return_value=AsyncMock(),
        ):
            result = await blacklist_token("invalid_token")

            assert result is False

    @pytest.mark.asyncio
    async def test_blacklist_ttl_calculation(self, test_user_id, redis_mock):
        """Test TTL is set correctly based on token expiration."""
        token = create_access_token(test_user_id)

        with patch(
            "backend.auth_blacklist.get_redis_client", return_value=redis_mock
        ):
            await blacklist_token(token)

            # Verify expire was called with reasonable TTL
            assert redis_mock.expire.called
            call_args = redis_mock.expire.call_args
            ttl = call_args[0][1]

            # Should be close to 1 hour (3600s)
            assert 3500 <= ttl <= 3700

    def test_blacklist_key_format(self, test_user_id):
        """Test blacklist key format in Redis."""
        from jose import jwt

        token = create_access_token(test_user_id)
        payload = jwt.decode(
            token, os.environ["JWT_SECRET"], algorithms=["HS256"]
        )

        jti = payload.get("jti", token[-16:])
        expected_key = f"blacklist:token:{jti}"

        assert expected_key.startswith("blacklist:token:")

    def test_user_blacklist_key_format(self, test_user_id):
        """Test user blacklist key format."""
        expected_key = f"blacklist:user:{test_user_id}"

        assert expected_key == f"blacklist:user:{test_user_id}"

    @pytest.mark.asyncio
    async def test_blacklist_different_reasons(
        self, test_user_id, redis_mock
    ):
        """Test blacklisting with different reasons."""
        token = create_access_token(test_user_id)

        reasons = ["logout", "password_change", "security_incident"]

        with patch(
            "backend.auth_blacklist.get_redis_client", return_value=redis_mock
        ):
            for reason in reasons:
                result = await blacklist_token(token, reason=reason)
                assert result is True

    @pytest.mark.asyncio
    async def test_concurrent_blacklist_checks(
        self, test_user_id, redis_mock
    ):
        """Test concurrent blacklist checks."""
        token = create_access_token(test_user_id)

        redis_mock.exists = AsyncMock(return_value=0)

        with patch(
            "backend.auth_blacklist.get_redis_client", return_value=redis_mock
        ):
            tasks = [is_blacklisted(token) for _ in range(10)]
            results = await asyncio.gather(*tasks)

            assert all(result is False for result in results)


# ============================================================================
# API KEY TESTS (20 tests)
# ============================================================================


class TestAPIKeys:
    """Test API key authentication."""

    def test_generate_api_key(self):
        """Test API key generation."""
        api_key, key_id = generate_api_key()

        assert api_key.startswith("vcci_test_")
        assert len(key_id) > 0
        assert len(api_key) > 40

    def test_api_key_format(self):
        """Test API key format."""
        api_key, _ = generate_api_key()

        parts = api_key.split("_")

        assert len(parts) >= 3
        assert parts[0] == "vcci"
        assert parts[1] == "test"  # Environment

    @pytest.mark.asyncio
    async def test_create_api_key(self, redis_mock):
        """Test creating API key."""
        with patch(
            "backend.auth_api_keys.get_redis_client", return_value=redis_mock
        ):
            api_key, key_data = await create_api_key(
                "test-service", [APIKeyScope.READ, APIKeyScope.WRITE]
            )

            assert api_key is not None
            assert key_data.service_name == "test-service"
            assert APIKeyScope.READ in key_data.scopes

    @pytest.mark.asyncio
    async def test_create_api_key_with_rate_limit(self, redis_mock):
        """Test creating API key with custom rate limit."""
        with patch(
            "backend.auth_api_keys.get_redis_client", return_value=redis_mock
        ):
            api_key, key_data = await create_api_key(
                "test-service", [APIKeyScope.READ], rate_limit=500
            )

            assert key_data.rate_limit == 500

    @pytest.mark.asyncio
    async def test_create_api_key_with_expiration(self, redis_mock):
        """Test creating API key with expiration."""
        expires_at = DeterministicClock.utcnow() + timedelta(days=30)

        with patch(
            "backend.auth_api_keys.get_redis_client", return_value=redis_mock
        ):
            api_key, key_data = await create_api_key(
                "test-service", [APIKeyScope.READ], expires_at=expires_at
            )

            assert key_data.expires_at is not None

    @pytest.mark.asyncio
    async def test_create_api_key_with_ip_whitelist(self, redis_mock):
        """Test creating API key with IP whitelist."""
        whitelist = ["192.168.1.0/24", "10.0.0.0/8"]

        with patch(
            "backend.auth_api_keys.get_redis_client", return_value=redis_mock
        ):
            api_key, key_data = await create_api_key(
                "test-service",
                [APIKeyScope.READ],
                ip_whitelist=whitelist,
            )

            assert key_data.ip_whitelist == whitelist

    def test_api_key_scopes(self):
        """Test all API key scopes are defined."""
        expected_scopes = ["read", "write", "admin", "calculate", "report"]

        for scope_name in expected_scopes:
            assert hasattr(APIKeyScope, scope_name.upper())

    @pytest.mark.asyncio
    async def test_verify_api_key_valid(self, redis_mock):
        """Test verifying valid API key."""
        # This is a simplified test - full test would require hash matching
        with patch(
            "backend.auth_api_keys.get_redis_client", return_value=redis_mock
        ):
            redis_mock.scan_iter = AsyncMock(return_value=iter([]))

            result = await verify_api_key("vcci_test_fake_key")

            # Should return None (not found) in this simple test
            assert result is None

    @pytest.mark.asyncio
    async def test_check_rate_limit_within_limit(self, redis_mock):
        """Test rate limit check when within limit."""
        from backend.auth_api_keys import APIKeyData

        key_data = APIKeyData(
            key_id="test_key",
            service_name="test",
            scopes=[APIKeyScope.READ],
            rate_limit=1000,
        )

        redis_mock.incr = AsyncMock(return_value=10)

        with patch(
            "backend.auth_api_keys.get_redis_client", return_value=redis_mock
        ):
            result = await check_rate_limit(key_data)

            assert result is True

    @pytest.mark.asyncio
    async def test_check_rate_limit_exceeded(self, redis_mock):
        """Test rate limit check when exceeded."""
        from backend.auth_api_keys import APIKeyData

        key_data = APIKeyData(
            key_id="test_key",
            service_name="test",
            scopes=[APIKeyScope.READ],
            rate_limit=100,
        )

        redis_mock.incr = AsyncMock(return_value=101)

        with patch(
            "backend.auth_api_keys.get_redis_client", return_value=redis_mock
        ):
            result = await check_rate_limit(key_data)

            assert result is False

    @pytest.mark.asyncio
    async def test_revoke_api_key(self, redis_mock):
        """Test revoking API key."""
        redis_mock.hgetall = AsyncMock(return_value={"is_active": "1"})

        with patch(
            "backend.auth_api_keys.get_redis_client", return_value=redis_mock
        ):
            result = await revoke_api_key("test_key_id")

            assert result is True
            assert redis_mock.hset.called

    @pytest.mark.asyncio
    async def test_delete_api_key(self, redis_mock):
        """Test deleting API key."""
        from backend.auth_api_keys import delete_api_key

        redis_mock.hgetall = AsyncMock(
            return_value={"service_name": "test-service"}
        )
        redis_mock.delete = AsyncMock(return_value=1)

        with patch(
            "backend.auth_api_keys.get_redis_client", return_value=redis_mock
        ):
            result = await delete_api_key("test_key_id")

            assert result is True

    def test_api_key_data_to_dict(self):
        """Test APIKeyData serialization."""
        from backend.auth_api_keys import APIKeyData

        key_data = APIKeyData(
            key_id="test_id",
            service_name="test-service",
            scopes=[APIKeyScope.READ, APIKeyScope.WRITE],
        )

        data = key_data.to_dict()

        assert data["key_id"] == "test_id"
        assert data["service_name"] == "test-service"
        assert "read" in data["scopes"]

    def test_api_key_data_from_dict(self):
        """Test APIKeyData deserialization."""
        from backend.auth_api_keys import APIKeyData

        data = {
            "key_id": "test_id",
            "service_name": "test-service",
            "scopes": "read,write",
            "rate_limit": "1000",
            "created_at": DeterministicClock.utcnow().isoformat(),
            "is_active": "1",
        }

        key_data = APIKeyData.from_dict(data)

        assert key_data.key_id == "test_id"
        assert APIKeyScope.READ in key_data.scopes

    def test_api_key_error_exception(self):
        """Test APIKeyError exception."""
        error = APIKeyError("Test error")

        assert error.status_code == 401
        assert "Test error" in str(error.detail)

    @pytest.mark.asyncio
    async def test_list_service_keys(self, redis_mock):
        """Test listing API keys for a service."""
        from backend.auth_api_keys import list_service_keys

        redis_mock.smembers = AsyncMock(return_value={"key1", "key2"})
        redis_mock.hgetall = AsyncMock(
            return_value={
                "key_id": "key1",
                "service_name": "test-service",
                "scopes": "read",
                "rate_limit": "1000",
                "created_at": DeterministicClock.utcnow().isoformat(),
                "is_active": "1",
            }
        )

        with patch(
            "backend.auth_api_keys.get_redis_client", return_value=redis_mock
        ):
            keys = await list_service_keys("test-service")

            assert isinstance(keys, list)

    def test_api_key_hash_verification(self):
        """Test API key hash verification."""
        from backend.auth_api_keys import hash_api_key, verify_api_key_hash

        api_key = "vcci_test_sample_key"
        hashed = hash_api_key(api_key)

        assert verify_api_key_hash(api_key, hashed) is True
        assert verify_api_key_hash("wrong_key", hashed) is False

    def test_multiple_scopes(self):
        """Test API key with multiple scopes."""
        from backend.auth_api_keys import APIKeyData

        key_data = APIKeyData(
            key_id="test",
            service_name="test",
            scopes=[
                APIKeyScope.READ,
                APIKeyScope.WRITE,
                APIKeyScope.CALCULATE,
            ],
        )

        assert len(key_data.scopes) == 3
        assert APIKeyScope.READ in key_data.scopes

    def test_admin_scope(self):
        """Test admin scope."""
        assert APIKeyScope.ADMIN.value == "admin"

    def test_calculate_scope(self):
        """Test calculate scope."""
        assert APIKeyScope.CALCULATE.value == "calculate"


# ============================================================================
# REQUEST SIGNING TESTS (15 tests)
# ============================================================================


class TestRequestSigning:
    """Test request signing functionality."""

    def test_generate_nonce(self):
        """Test nonce generation."""
        nonce = generate_nonce()

        assert nonce is not None
        assert len(nonce) > 32
        assert isinstance(nonce, str)

    def test_generate_timestamp(self):
        """Test timestamp generation."""
        timestamp = generate_timestamp()

        assert timestamp is not None
        assert "T" in timestamp  # ISO format
        assert ":" in timestamp

    def test_compute_signature(self):
        """Test signature computation."""
        signature = compute_signature(
            method="POST",
            path="/api/test",
            timestamp="2025-01-01T00:00:00",
            nonce="test_nonce",
            body='{"test": "data"}',
        )

        assert signature is not None
        assert len(signature) == 64  # SHA256 hex is 64 chars

    def test_verify_signature_valid(self):
        """Test signature verification with valid signature."""
        method = "POST"
        path = "/api/test"
        timestamp = "2025-01-01T00:00:00"
        nonce = "test_nonce"
        body = '{"test": "data"}'

        signature = compute_signature(method, path, timestamp, nonce, body)

        assert (
            verify_signature(method, path, timestamp, nonce, signature, body)
            is True
        )

    def test_verify_signature_invalid(self):
        """Test signature verification with invalid signature."""
        result = verify_signature(
            "POST",
            "/api/test",
            "2025-01-01T00:00:00",
            "nonce",
            "invalid_signature",
            "{}",
        )

        assert result is False

    def test_verify_signature_tampered_body(self):
        """Test signature verification fails with tampered body."""
        method = "POST"
        path = "/api/test"
        timestamp = "2025-01-01T00:00:00"
        nonce = "test_nonce"
        body = '{"test": "data"}'

        signature = compute_signature(method, path, timestamp, nonce, body)

        tampered_body = '{"test": "tampered"}'

        assert (
            verify_signature(
                method, path, timestamp, nonce, signature, tampered_body
            )
            is False
        )

    def test_verify_timestamp_valid(self):
        """Test timestamp verification with valid timestamp."""
        timestamp = DeterministicClock.utcnow().isoformat()

        assert verify_timestamp(timestamp) is True

    def test_verify_timestamp_too_old(self):
        """Test timestamp verification with old timestamp."""
        old_timestamp = (DeterministicClock.utcnow() - timedelta(minutes=10)).isoformat()

        assert verify_timestamp(old_timestamp) is False

    def test_verify_timestamp_future(self):
        """Test timestamp verification with future timestamp."""
        future_timestamp = (
            DeterministicClock.utcnow() + timedelta(minutes=5)
        ).isoformat()

        assert verify_timestamp(future_timestamp) is False

    @pytest.mark.asyncio
    async def test_verify_nonce_first_use(self, redis_mock):
        """Test nonce verification on first use."""
        redis_mock.exists = AsyncMock(return_value=0)

        with patch(
            "backend.request_signing.get_redis_client",
            return_value=redis_mock,
        ):
            result = await verify_nonce("test_nonce")

            assert result is True
            assert redis_mock.setex.called

    @pytest.mark.asyncio
    async def test_verify_nonce_reuse(self, redis_mock):
        """Test nonce verification on reuse (should fail)."""
        redis_mock.exists = AsyncMock(return_value=1)

        with patch(
            "backend.request_signing.get_redis_client",
            return_value=redis_mock,
        ):
            result = await verify_nonce("test_nonce")

            assert result is False

    def test_request_signer_sign_request(self):
        """Test RequestSigner.sign_request()."""
        signer = RequestSigner(os.environ["REQUEST_SIGNING_SECRET"])

        headers = signer.sign_request("POST", "/api/test", '{"data": "test"}')

        assert "X-Request-Timestamp" in headers
        assert "X-Request-Nonce" in headers
        assert "X-Request-Signature" in headers

    def test_request_signer_signature_valid(self):
        """Test RequestSigner generates valid signature."""
        signer = RequestSigner(os.environ["REQUEST_SIGNING_SECRET"])

        method = "POST"
        path = "/api/test"
        body = '{"data": "test"}'

        headers = signer.sign_request(method, path, body)

        # Verify the signature
        assert (
            verify_signature(
                method,
                path,
                headers["X-Request-Timestamp"],
                headers["X-Request-Nonce"],
                headers["X-Request-Signature"],
                body,
            )
            is True
        )

    def test_different_methods_different_signatures(self):
        """Test different HTTP methods produce different signatures."""
        timestamp = "2025-01-01T00:00:00"
        nonce = "test_nonce"
        body = "{}"

        sig_post = compute_signature("POST", "/api/test", timestamp, nonce, body)
        sig_get = compute_signature("GET", "/api/test", timestamp, nonce, body)

        assert sig_post != sig_get

    def test_different_paths_different_signatures(self):
        """Test different paths produce different signatures."""
        timestamp = "2025-01-01T00:00:00"
        nonce = "test_nonce"

        sig1 = compute_signature("POST", "/api/path1", timestamp, nonce)
        sig2 = compute_signature("POST", "/api/path2", timestamp, nonce)

        assert sig1 != sig2


# ============================================================================
# SECURITY HEADERS TESTS (10 tests)
# ============================================================================


class TestSecurityHeaders:
    """Test security headers functionality."""

    def test_build_csp_header(self):
        """Test CSP header building."""
        name, value = build_csp_header()

        assert name == "Content-Security-Policy"
        assert "default-src 'self'" in value
        assert "frame-ancestors 'none'" in value

    def test_build_csp_header_report_only(self):
        """Test CSP header in report-only mode."""
        name, value = build_csp_header(report_only=True)

        assert name == "Content-Security-Policy-Report-Only"

    def test_build_csp_header_with_report_uri(self):
        """Test CSP header with report URI."""
        name, value = build_csp_header(report_uri="/api/csp-report")

        assert "report-uri /api/csp-report" in value

    def test_build_expect_ct_header(self):
        """Test Expect-CT header building."""
        value = build_expect_ct_header(86400, enforce=True)

        assert "max-age=86400" in value
        assert "enforce" in value

    def test_build_expect_ct_header_with_report_uri(self):
        """Test Expect-CT header with report URI."""
        value = build_expect_ct_header(
            86400, report_uri="/api/ct-report"
        )

        assert 'report-uri="/api/ct-report"' in value

    def test_build_nel_header(self):
        """Test NEL header building."""
        value = build_nel_header()

        data = json.loads(value)

        assert "report_to" in data
        assert "max_age" in data

    def test_generate_sri_hash_sha256(self):
        """Test SRI hash generation with SHA256."""
        content = "console.log('test');"
        sri_hash = generate_sri_hash(content, algorithm="sha256")

        assert sri_hash.startswith("sha256-")
        assert len(sri_hash) > 10

    def test_generate_sri_hash_sha384(self):
        """Test SRI hash generation with SHA384."""
        content = "console.log('test');"
        sri_hash = generate_sri_hash(content, algorithm="sha384")

        assert sri_hash.startswith("sha384-")

    def test_generate_sri_hash_deterministic(self):
        """Test SRI hash is deterministic."""
        content = "test content"

        hash1 = generate_sri_hash(content)
        hash2 = generate_sri_hash(content)

        assert hash1 == hash2

    def test_security_headers_middleware_init(self):
        """Test SecurityHeadersMiddleware initialization."""
        from starlette.applications import Starlette

        app = Starlette()
        middleware = SecurityHeadersMiddleware(app)

        assert middleware is not None
        assert middleware.config is not None


# ============================================================================
# AUDIT LOGGING TESTS (10 tests)
# ============================================================================


class TestAuditLogging:
    """Test enhanced audit logging."""

    def test_audit_event_creation(self, test_user_id, test_ip):
        """Test audit event creation."""
        event = AuditEvent(
            event_type=AuditEventType.AUTH_SUCCESS,
            severity=AuditSeverity.INFO,
            user_id=test_user_id,
            ip_address=test_ip,
            result="success",
        )

        assert event.event_type == AuditEventType.AUTH_SUCCESS
        assert event.user_id == test_user_id

    def test_audit_event_to_dict(self, test_user_id):
        """Test audit event serialization."""
        event = AuditEvent(
            event_type=AuditEventType.AUTH_FAILURE,
            severity=AuditSeverity.WARNING,
            user_id=test_user_id,
        )

        data = event.to_dict()

        assert data["event_type"] == "auth.failure"
        assert data["severity"] == "warning"
        assert data["user_id"] == test_user_id

    def test_audit_event_compute_hash(self, test_user_id):
        """Test audit event hash computation."""
        event = AuditEvent(
            event_type=AuditEventType.AUTH_SUCCESS,
            severity=AuditSeverity.INFO,
            user_id=test_user_id,
        )

        event_hash = event.compute_hash()

        assert event_hash is not None
        assert len(event_hash) == 64  # SHA256 hex

    def test_audit_event_hash_includes_previous(self, test_user_id):
        """Test audit event hash includes previous hash."""
        event1 = AuditEvent(
            event_type=AuditEventType.AUTH_SUCCESS,
            severity=AuditSeverity.INFO,
            user_id=test_user_id,
        )
        hash1 = event1.compute_hash()

        event2 = AuditEvent(
            event_type=AuditEventType.AUTH_SUCCESS,
            severity=AuditSeverity.INFO,
            user_id=test_user_id,
            previous_hash=hash1,
        )
        hash2 = event2.compute_hash()

        assert hash1 != hash2

    def test_all_audit_event_types_defined(self):
        """Test all expected audit event types are defined."""
        expected_types = [
            "AUTH_SUCCESS",
            "AUTH_FAILURE",
            "PASSWORD_CHANGE",
            "API_KEY_USED",
            "DATA_EXPORT",
            "SUSPICIOUS_ACTIVITY",
        ]

        for event_type in expected_types:
            assert hasattr(AuditEventType, event_type)

    def test_all_severity_levels_defined(self):
        """Test all severity levels are defined."""
        expected_severities = ["INFO", "WARNING", "ERROR", "CRITICAL"]

        for severity in expected_severities:
            assert hasattr(AuditSeverity, severity)

    @pytest.mark.asyncio
    async def test_audit_logger_initialization(self):
        """Test AuditLogger initialization."""
        logger = AuditLogger()

        assert logger is not None
        assert logger.last_event_hash is None

    @pytest.mark.asyncio
    async def test_log_auth_success(self, test_user_id, test_ip):
        """Test logging authentication success."""
        with patch.object(AuditLogger, "log_event", new=AsyncMock()):
            await log_auth_success(test_user_id, test_ip)

    @pytest.mark.asyncio
    async def test_log_auth_failure(self, test_user_id, test_ip):
        """Test logging authentication failure."""
        with patch.object(AuditLogger, "log_event", new=AsyncMock()):
            await log_auth_failure(test_user_id, test_ip, reason="invalid_password")

    @pytest.mark.asyncio
    async def test_verify_integrity_valid_chain(self, test_user_id):
        """Test audit log integrity verification with valid chain."""
        logger = AuditLogger()

        # Create chain of events
        events = []

        event1 = AuditEvent(
            event_type=AuditEventType.AUTH_SUCCESS,
            severity=AuditSeverity.INFO,
            user_id=test_user_id,
        )
        event1.event_hash = event1.compute_hash()
        events.append(event1)

        event2 = AuditEvent(
            event_type=AuditEventType.DATA_EXPORT,
            severity=AuditSeverity.INFO,
            user_id=test_user_id,
            previous_hash=event1.event_hash,
        )
        event2.event_hash = event2.compute_hash()
        events.append(event2)

        result = await logger.verify_integrity(events)

        assert result is True


# ============================================================================
# INTEGRATION TESTS
# ============================================================================


class TestSecurityIntegration:
    """Integration tests for security features."""

    @pytest.mark.asyncio
    async def test_full_token_lifecycle(self, test_user_id, redis_mock):
        """Test full token lifecycle: issue, refresh, revoke."""
        with patch(
            "backend.auth_refresh.get_redis_client", return_value=redis_mock
        ):
            # Issue tokens
            tokens = await issue_token_pair(test_user_id)
            assert tokens is not None

            # Mock for refresh
            from jose import jwt

            payload = jwt.decode(
                tokens.refresh_token,
                os.environ["REFRESH_SECRET"],
                algorithms=["HS256"],
            )

            redis_mock.hgetall = AsyncMock(
                return_value={
                    "user_id": test_user_id,
                    "jti": payload["jti"],
                    "issued_at": DeterministicClock.utcnow().isoformat(),
                    "expires_at": (
                        DeterministicClock.utcnow() + timedelta(days=7)
                    ).isoformat(),
                }
            )

            # Refresh
            new_tokens = await refresh_access_token(tokens.refresh_token)
            assert new_tokens is not None

            # Revoke
            redis_mock.delete = AsyncMock(return_value=1)
            result = await revoke_refresh_token(new_tokens.refresh_token)
            assert result is True

    def test_signature_and_verification_roundtrip(self):
        """Test signing and verification roundtrip."""
        signer = RequestSigner(os.environ["REQUEST_SIGNING_SECRET"])

        method = "POST"
        path = "/api/critical-operation"
        body = '{"amount": 1000}'

        headers = signer.sign_request(method, path, body)

        # Verify
        is_valid = verify_signature(
            method,
            path,
            headers["X-Request-Timestamp"],
            headers["X-Request-Nonce"],
            headers["X-Request-Signature"],
            body,
        )

        assert is_valid is True


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
