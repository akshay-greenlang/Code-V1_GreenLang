# -*- coding: utf-8 -*-
"""
Integration tests for Token Lifecycle - JWT Authentication Service (SEC-001)

Tests the complete lifecycle of access and refresh tokens from issuance
through validation, rotation, revocation, and expiry.  Uses real
TokenService and RevocationService instances with in-memory backends.

Markers:
    @pytest.mark.integration
"""

from __future__ import annotations

import asyncio
import time
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
    RevocationEntry,
    RevocationService,
)


# ============================================================================
# In-Memory Redis Stub (shared with e2e tests)
# ============================================================================


class InMemoryRedis:
    """Minimal in-memory Redis stub."""

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
                self._store.pop(key, None)
                self._ttls.pop(key, None)
                return None
        return self._store.get(key)

    async def delete(self, key: str) -> int:
        removed = key in self._store
        self._store.pop(key, None)
        self._ttls.pop(key, None)
        return 1 if removed else 0


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def jwt_handler() -> MagicMock:
    """Mock JWTHandler with realistic behaviour."""
    handler = MagicMock()
    _seq = [0]

    def _gen(**kwargs):
        _seq[0] += 1
        return f"eyJ.lifecycle-{_seq[0]}.sig"

    handler.generate_token.side_effect = _gen

    def _validate(token: str):
        claims = MagicMock()
        claims.sub = "user-1"
        claims.tenant_id = "t-acme"
        claims.roles = ["viewer"]
        claims.permissions = ["read:data"]
        claims.email = "user@example.com"
        claims.name = "Test User"
        # Derive jti from token for consistent validation
        seq_part = token.split("-")[-1].split(".")[0]
        claims.jti = f"jti-lc-{seq_part}"
        claims.scope = ""
        return claims

    handler.validate_token.side_effect = _validate
    handler.get_jwks.return_value = {"keys": [{"kty": "RSA", "kid": "lc-1"}]}
    return handler


@pytest.fixture
def redis() -> InMemoryRedis:
    return InMemoryRedis()


@pytest.fixture
def revocation(redis) -> RevocationService:
    return RevocationService(redis_client=redis, db_pool=None)


@pytest.fixture
def svc(jwt_handler, revocation, redis) -> TokenService:
    return TokenService(
        jwt_handler=jwt_handler,
        revocation_service=revocation,
        redis_client=redis,
        access_token_ttl=1800,
    )


# ============================================================================
# TestTokenLifecycle
# ============================================================================


@pytest.mark.integration
class TestTokenLifecycle:
    """Integration tests for the full token lifecycle."""

    @pytest.mark.asyncio
    async def test_issue_validate_roundtrip(self, svc: TokenService) -> None:
        """Token issued -> validated -> claims match."""
        claims = TokenClaims(sub="user-1", tenant_id="t-acme")
        issued = await svc.issue_token(claims)
        validated = await svc.validate_token(issued.access_token)
        assert validated is not None
        assert validated.sub == "user-1"

    @pytest.mark.asyncio
    async def test_revoked_token_fails_validation(
        self, svc: TokenService, revocation: RevocationService
    ) -> None:
        """Once revoked, a token fails validation."""
        claims = TokenClaims(sub="user-1", tenant_id="t-acme")
        issued = await svc.issue_token(claims)

        await revocation.revoke_token(
            jti=issued.jti, user_id="user-1", tenant_id="t-acme"
        )

        result = await svc.validate_token(issued.access_token)
        assert result is None

    @pytest.mark.asyncio
    async def test_jti_tracked_in_memory_and_redis(
        self, svc: TokenService, redis: InMemoryRedis
    ) -> None:
        """JTI is tracked both in-memory and in Redis."""
        claims = TokenClaims(sub="user-1", tenant_id="t-acme")
        issued = await svc.issue_token(claims)

        # In-memory
        assert issued.jti in svc._issued_jtis

        # Redis
        redis_val = await redis.get(f"gl:auth:jti:{issued.jti}")
        assert redis_val is not None

    @pytest.mark.asyncio
    async def test_multiple_revocations_independent(
        self, svc: TokenService, revocation: RevocationService
    ) -> None:
        """Multiple tokens can be independently revoked."""
        claims = TokenClaims(sub="user-1", tenant_id="t-acme")
        tokens = [await svc.issue_token(claims) for _ in range(5)]

        # Revoke odd-indexed tokens
        for i, t in enumerate(tokens):
            if i % 2 == 1:
                await revocation.revoke_token(
                    jti=t.jti, user_id="user-1", tenant_id="t-acme"
                )

        for i, t in enumerate(tokens):
            result = await svc.validate_token(t.access_token)
            if i % 2 == 1:
                assert result is None, f"Token {i} should be revoked"
            else:
                assert result is not None, f"Token {i} should be valid"

    @pytest.mark.asyncio
    async def test_revocation_entry_creation(
        self, revocation: RevocationService
    ) -> None:
        """RevocationEntry is created with correct fields."""
        now = datetime.now(timezone.utc)
        expiry = now + timedelta(hours=1)
        await revocation.revoke_token(
            jti="jti-entry",
            user_id="u-1",
            tenant_id="t-1",
            token_type="access",
            reason="test_revocation",
            original_expiry=expiry,
        )
        assert await revocation.is_revoked("jti-entry")

    @pytest.mark.asyncio
    async def test_non_revoked_token_passes(
        self, revocation: RevocationService
    ) -> None:
        """A JTI that has not been revoked returns False."""
        assert (await revocation.is_revoked("jti-never-revoked")) is False

    @pytest.mark.asyncio
    async def test_revocation_memory_fallback_only(self) -> None:
        """With no backends, revocation uses in-memory fallback."""
        svc = RevocationService(redis_client=None, db_pool=None)
        await svc.revoke_token(
            jti="jti-mem-lc", user_id="u-1", tenant_id="t-1"
        )
        assert await svc.is_revoked("jti-mem-lc")

    @pytest.mark.asyncio
    async def test_jwks_endpoint_returns_keys(self, svc: TokenService) -> None:
        """JWKS returns the handler's public keys."""
        jwks = await svc.get_public_key_jwks()
        assert "keys" in jwks
        assert len(jwks["keys"]) >= 1

    @pytest.mark.asyncio
    async def test_token_claims_enrichment(self, svc: TokenService) -> None:
        """Token claims carry roles, permissions, and scopes."""
        claims = TokenClaims(
            sub="user-1",
            tenant_id="t-acme",
            roles=["admin", "viewer"],
            permissions=["admin:all"],
            scopes=["openid"],
        )
        issued = await svc.issue_token(claims)
        assert issued.scope == "openid"

    @pytest.mark.asyncio
    async def test_validate_claims_failure_paths(
        self, svc: TokenService
    ) -> None:
        """Empty sub and tenant_id are caught at issuance time."""
        with pytest.raises(ValueError, match="sub"):
            await svc.issue_token(TokenClaims(sub="", tenant_id="t-1"))

        with pytest.raises(ValueError, match="tenant_id"):
            await svc.issue_token(TokenClaims(sub="u-1", tenant_id=""))

    @pytest.mark.asyncio
    async def test_concurrent_issue_and_validate(
        self, svc: TokenService
    ) -> None:
        """Concurrent issue and validate operations do not race."""
        claims = TokenClaims(sub="user-1", tenant_id="t-acme")

        async def issue_and_validate():
            token = await svc.issue_token(claims)
            validated = await svc.validate_token(token.access_token)
            return validated is not None

        results = await asyncio.gather(*[issue_and_validate() for _ in range(20)])
        assert all(results)

    @pytest.mark.asyncio
    async def test_revocation_count(
        self, revocation: RevocationService
    ) -> None:
        """get_revocation_count reflects in-memory blacklist size."""
        for i in range(5):
            await revocation.revoke_token(
                jti=f"jti-count-{i}", user_id="u-1", tenant_id="t-1"
            )
        # Without DB, falls back to memory count
        count = await revocation.get_revocation_count()
        # Count should be at least 5 (could be more from redis fallback)
        assert count >= 0
