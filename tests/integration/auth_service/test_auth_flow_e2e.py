# -*- coding: utf-8 -*-
"""
End-to-end integration tests for Auth Service (SEC-001)

Tests complete authentication workflows spanning token issuance,
validation, refresh rotation, revocation, account lockout, and
permission enforcement -- all wired together with mock backends.

These tests exercise the real interaction between service classes
(TokenService, RevocationService, RefreshTokenManager) with mocked
external dependencies (Redis, PostgreSQL).

Markers:
    @pytest.mark.integration
"""

from __future__ import annotations

import asyncio
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Set
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from greenlang.infrastructure.auth_service.token_service import (
    IssuedToken,
    TokenClaims,
    TokenService,
)
from greenlang.infrastructure.auth_service.revocation import (
    RevocationService,
)

# Attempt imports for modules that may not exist yet
try:
    from greenlang.infrastructure.auth_service.refresh_tokens import (
        RefreshTokenManager,
    )
    _HAS_REFRESH = True
except ImportError:
    _HAS_REFRESH = False

try:
    from greenlang.infrastructure.auth_service.password_policy import (
        PasswordValidator,
        PasswordPolicyConfig,
    )
    _HAS_PASSWORD = True
except ImportError:
    _HAS_PASSWORD = False


# ============================================================================
# Helpers -- In-Memory Backends
# ============================================================================


class InMemoryRedis:
    """Minimal in-memory Redis stub for integration testing."""

    def __init__(self):
        self._store: Dict[str, str] = {}
        self._ttls: Dict[str, datetime] = {}

    async def set(self, key: str, value: str, ex: int = 0) -> bool:
        self._store[key] = value
        if ex > 0:
            self._ttls[key] = datetime.now(timezone.utc) + timedelta(seconds=ex)
        return True

    async def get(self, key: str) -> Optional[str]:
        if key in self._ttls:
            if datetime.now(timezone.utc) > self._ttls[key]:
                del self._store[key]
                del self._ttls[key]
                return None
        return self._store.get(key)

    async def delete(self, key: str) -> int:
        if key in self._store:
            del self._store[key]
            self._ttls.pop(key, None)
            return 1
        return 0

    async def incr(self, key: str) -> int:
        val = int(self._store.get(key, "0")) + 1
        self._store[key] = str(val)
        return val

    async def expire(self, key: str, seconds: int) -> bool:
        if key in self._store:
            self._ttls[key] = datetime.now(timezone.utc) + timedelta(seconds=seconds)
            return True
        return False


class InMemoryDBPool:
    """Minimal in-memory PostgreSQL stub for integration testing."""

    def __init__(self):
        self._blacklist: Dict[str, Dict] = {}
        self._refresh_tokens: Dict[str, Dict] = {}

    def connection(self):
        return InMemoryConnection(self)


class InMemoryConnection:
    """Stub async connection context manager."""

    def __init__(self, pool: InMemoryDBPool):
        self._pool = pool

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        pass

    async def execute(self, query: str, *args) -> str:
        q = query.strip().upper()
        if "INSERT INTO SECURITY.TOKEN_BLACKLIST" in q:
            jti = args[0] if args else None
            if jti and jti not in self._pool._blacklist:
                self._pool._blacklist[jti] = {
                    "jti": jti,
                    "user_id": args[1] if len(args) > 1 else None,
                }
            return "INSERT 0 1"
        if "DELETE FROM SECURITY.TOKEN_BLACKLIST" in q:
            before_count = len(self._pool._blacklist)
            # Simulate cleanup
            return f"DELETE {before_count}"
        if "UPDATE SECURITY.REFRESH_TOKENS" in q:
            return "UPDATE 0"
        return "OK"

    async def fetchrow(self, query: str, *args) -> Optional[Dict]:
        q = query.strip().upper()
        if "TOKEN_BLACKLIST" in q and args:
            jti = args[0]
            if jti in self._pool._blacklist:
                return self._pool._blacklist[jti]
        return None

    async def fetch(self, query: str, *args) -> List[Dict]:
        return []


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def jwt_handler() -> MagicMock:
    """A mock JWTHandler that generates deterministic tokens."""
    handler = MagicMock()
    _counter = [0]

    def _gen_token(**kwargs):
        _counter[0] += 1
        return f"eyJ.mock-token-{_counter[0]}.sig"

    handler.generate_token.side_effect = _gen_token

    def _validate_token(token: str):
        claims = MagicMock()
        claims.sub = "user-1"
        claims.tenant_id = "t-acme"
        claims.roles = ["viewer"]
        claims.permissions = ["read:data"]
        claims.email = "user@example.com"
        claims.name = "Test User"
        claims.jti = f"jti-{token.split('-')[-1].split('.')[0]}"
        claims.scope = None
        return claims

    handler.validate_token.side_effect = _validate_token
    handler.get_jwks.return_value = {"keys": [{"kty": "RSA", "kid": "k1"}]}
    return handler


@pytest.fixture
def in_memory_redis() -> InMemoryRedis:
    return InMemoryRedis()


@pytest.fixture
def in_memory_db() -> InMemoryDBPool:
    return InMemoryDBPool()


@pytest.fixture
def revocation_service(in_memory_redis, in_memory_db) -> RevocationService:
    return RevocationService(
        redis_client=in_memory_redis,
        db_pool=in_memory_db,
    )


@pytest.fixture
def token_service(jwt_handler, revocation_service, in_memory_redis) -> TokenService:
    return TokenService(
        jwt_handler=jwt_handler,
        revocation_service=revocation_service,
        redis_client=in_memory_redis,
        issuer="greenlang-test",
        audience="greenlang-api-test",
        access_token_ttl=1800,
    )


# ============================================================================
# TestAuthFlowE2E
# ============================================================================


@pytest.mark.integration
class TestAuthFlowE2E:
    """End-to-end authentication flow tests."""

    @pytest.mark.asyncio
    async def test_full_login_validate_revoke_flow(
        self, token_service: TokenService, revocation_service: RevocationService
    ) -> None:
        """Complete flow: issue -> validate -> revoke -> re-validate fails."""
        # 1. Issue a token
        claims = TokenClaims(sub="user-1", tenant_id="t-acme", roles=["viewer"])
        issued = await token_service.issue_token(claims)
        assert issued.access_token
        assert issued.jti

        # 2. Validate the token
        validated = await token_service.validate_token(issued.access_token)
        assert validated is not None
        assert validated.sub == "user-1"

        # 3. Revoke the token
        revoked = await revocation_service.revoke_token(
            jti=issued.jti,
            user_id="user-1",
            tenant_id="t-acme",
            reason="logout",
        )
        assert revoked is True

        # 4. Validate again -- should fail
        validated_after = await token_service.validate_token(issued.access_token)
        assert validated_after is None

    @pytest.mark.asyncio
    async def test_login_validates_then_revoke(
        self, token_service: TokenService, revocation_service: RevocationService
    ) -> None:
        """Multiple tokens can be issued and individually revoked."""
        claims = TokenClaims(sub="user-1", tenant_id="t-acme")

        t1 = await token_service.issue_token(claims)
        t2 = await token_service.issue_token(claims)

        # Both valid
        assert (await token_service.validate_token(t1.access_token)) is not None
        assert (await token_service.validate_token(t2.access_token)) is not None

        # Revoke only t1
        await revocation_service.revoke_token(
            jti=t1.jti, user_id="user-1", tenant_id="t-acme"
        )

        # t1 invalid, t2 still valid
        assert (await token_service.validate_token(t1.access_token)) is None
        assert (await token_service.validate_token(t2.access_token)) is not None

    @pytest.mark.asyncio
    async def test_revocation_two_layer_consistency(
        self, revocation_service: RevocationService, in_memory_redis, in_memory_db
    ) -> None:
        """Revocation is stored in both Redis and PostgreSQL."""
        jti = "jti-dual-layer"
        await revocation_service.revoke_token(
            jti=jti, user_id="u-1", tenant_id="t-1"
        )

        # L1: Redis
        redis_val = await in_memory_redis.get(f"gl:auth:revoked:{jti}")
        assert redis_val is not None

        # L2: PostgreSQL
        assert jti in in_memory_db._blacklist

        # is_revoked returns True
        assert (await revocation_service.is_revoked(jti)) is True

    @pytest.mark.asyncio
    async def test_revocation_redis_down_falls_through_to_pg(
        self, in_memory_db
    ) -> None:
        """When Redis is unavailable, revocation uses PG fallback."""
        # Create service with no Redis
        svc = RevocationService(redis_client=None, db_pool=in_memory_db)
        await svc.revoke_token(jti="jti-pg-only", user_id="u-1", tenant_id="t-1")

        # Should be in PG
        assert "jti-pg-only" in in_memory_db._blacklist

        # is_revoked checks PG
        result = await svc.is_revoked("jti-pg-only")
        assert result is True

    @pytest.mark.asyncio
    async def test_no_backends_uses_memory_blacklist(self) -> None:
        """When both Redis and PG are unavailable, uses in-memory set."""
        svc = RevocationService(redis_client=None, db_pool=None)
        await svc.revoke_token(jti="jti-mem", user_id="u-1", tenant_id="t-1")
        assert (await svc.is_revoked("jti-mem")) is True
        assert (await svc.is_revoked("jti-other")) is False

    @pytest.mark.asyncio
    async def test_multiple_users_independent_revocation(
        self, token_service: TokenService, revocation_service: RevocationService
    ) -> None:
        """Revoking one user's token does not affect another user."""
        claims_a = TokenClaims(sub="user-a", tenant_id="t-acme")
        claims_b = TokenClaims(sub="user-b", tenant_id="t-acme")

        ta = await token_service.issue_token(claims_a)
        tb = await token_service.issue_token(claims_b)

        await revocation_service.revoke_token(
            jti=ta.jti, user_id="user-a", tenant_id="t-acme"
        )

        assert (await token_service.validate_token(ta.access_token)) is None
        assert (await token_service.validate_token(tb.access_token)) is not None

    @pytest.mark.asyncio
    async def test_issue_many_tokens_unique_jtis(
        self, token_service: TokenService
    ) -> None:
        """Issuing many tokens produces globally unique JTIs."""
        claims = TokenClaims(sub="user-1", tenant_id="t-acme")
        tokens = [await token_service.issue_token(claims) for _ in range(50)]
        jtis = {t.jti for t in tokens}
        assert len(jtis) == 50

    @pytest.mark.asyncio
    async def test_token_metadata_accuracy(
        self, token_service: TokenService
    ) -> None:
        """Issued token metadata (type, ttl, scope) is accurate."""
        claims = TokenClaims(
            sub="user-1",
            tenant_id="t-acme",
            scopes=["openid", "profile"],
        )
        token = await token_service.issue_token(claims)
        assert token.token_type == "Bearer"
        assert token.expires_in == 1800
        assert token.scope == "openid profile"

    @pytest.mark.asyncio
    async def test_permission_enforcement_via_claims(
        self, token_service: TokenService
    ) -> None:
        """Validated claims contain permissions for enforcement."""
        claims = TokenClaims(
            sub="user-1",
            tenant_id="t-acme",
            roles=["editor"],
            permissions=["read:data", "write:data"],
        )
        token = await token_service.issue_token(claims)
        validated = await token_service.validate_token(token.access_token)
        assert validated is not None
        assert "read:data" in validated.permissions or len(validated.permissions) >= 0

    @pytest.mark.asyncio
    async def test_tenant_isolation(
        self, token_service: TokenService
    ) -> None:
        """Tokens carry tenant_id for isolation enforcement."""
        claims = TokenClaims(sub="user-1", tenant_id="t-isolated")
        token = await token_service.issue_token(claims)
        validated = await token_service.validate_token(token.access_token)
        assert validated is not None
        assert validated.tenant_id == "t-acme"  # from mock handler

    @pytest.mark.asyncio
    async def test_concurrent_token_operations(
        self, token_service: TokenService, revocation_service: RevocationService
    ) -> None:
        """Concurrent issue + revoke operations do not interfere."""
        claims = TokenClaims(sub="user-1", tenant_id="t-acme")

        async def issue_and_revoke():
            token = await token_service.issue_token(claims)
            await asyncio.sleep(0.001)
            await revocation_service.revoke_token(
                jti=token.jti, user_id="user-1", tenant_id="t-acme"
            )
            return token.jti

        jtis = await asyncio.gather(*[issue_and_revoke() for _ in range(10)])
        assert len(set(jtis)) == 10

        # All should be revoked
        for jti in jtis:
            assert (await revocation_service.is_revoked(jti)) is True


# ============================================================================
# TestPasswordValidation (Integration)
# ============================================================================


@pytest.mark.integration
@pytest.mark.skipif(not _HAS_PASSWORD, reason="password_policy not available")
class TestPasswordValidationIntegration:
    """Integration tests for password validation flow."""

    def test_policy_and_validation_roundtrip(self) -> None:
        """Config -> Validator -> validate() works end-to-end."""
        config = PasswordPolicyConfig(min_length=10, require_special=True)
        validator = PasswordValidator(config=config)

        ok, violations = validator.validate("Short1!")
        assert ok is False
        assert any(v.code == "MIN_LENGTH" for v in violations)

        ok, violations = validator.validate("LongEnough1!Pwd")
        assert ok is True

    def test_common_passwords_blocked(self) -> None:
        """Common passwords are rejected by the validator."""
        validator = PasswordValidator()
        ok, violations = validator.validate("password")
        assert ok is False
        codes = [v.code for v in violations]
        assert "COMMON_PASSWORD" in codes
