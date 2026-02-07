# -*- coding: utf-8 -*-
"""
Unit tests for RevocationService - JWT Authentication Service (SEC-001)

Tests the two-layer token revocation system (Redis L1 + PostgreSQL L2),
in-memory fallback, family revocation, user-bulk revocation, cleanup
of expired entries, and graceful degradation when either layer is down.

Coverage targets: 85%+ of revocation.py
"""

from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock, patch, call

import pytest

from greenlang.infrastructure.auth_service.revocation import (
    RevocationEntry,
    RevocationService,
    _parse_command_count,
    _REDIS_PREFIX,
    _DEFAULT_TTL_SECONDS,
)


# ============================================================================
# Helpers
# ============================================================================


def _make_redis() -> AsyncMock:
    """Create a mock async Redis client."""
    redis = AsyncMock()
    redis.set = AsyncMock(return_value=True)
    redis.get = AsyncMock(return_value=None)
    return redis


def _make_db_pool(
    fetchrow_return: Any = None,
    fetch_return: Optional[list] = None,
    execute_return: str = "UPDATE 0",
) -> AsyncMock:
    """Create a mock async database pool with connection context manager."""
    conn = AsyncMock()
    conn.execute = AsyncMock(return_value=execute_return)
    conn.fetchrow = AsyncMock(return_value=fetchrow_return)
    conn.fetch = AsyncMock(return_value=fetch_return or [])

    pool = AsyncMock()
    # pool.connection() is an async context manager
    cm = AsyncMock()
    cm.__aenter__ = AsyncMock(return_value=conn)
    cm.__aexit__ = AsyncMock(return_value=False)
    pool.connection.return_value = cm

    return pool, conn


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def redis_client() -> AsyncMock:
    return _make_redis()


@pytest.fixture
def db_pool_and_conn():
    return _make_db_pool()


@pytest.fixture
def db_pool(db_pool_and_conn):
    pool, _ = db_pool_and_conn
    return pool


@pytest.fixture
def db_conn(db_pool_and_conn):
    _, conn = db_pool_and_conn
    return conn


@pytest.fixture
def service(redis_client, db_pool) -> RevocationService:
    return RevocationService(redis_client=redis_client, db_pool=db_pool)


@pytest.fixture
def service_redis_only(redis_client) -> RevocationService:
    return RevocationService(redis_client=redis_client, db_pool=None)


@pytest.fixture
def service_db_only(db_pool) -> RevocationService:
    return RevocationService(redis_client=None, db_pool=db_pool)


@pytest.fixture
def service_memory_only() -> RevocationService:
    return RevocationService(redis_client=None, db_pool=None)


# ============================================================================
# TestRevocationEntry
# ============================================================================


class TestRevocationEntry:
    """Tests for the RevocationEntry dataclass."""

    def test_create_entry_defaults(self) -> None:
        """Entry has sensible defaults."""
        entry = RevocationEntry(
            jti="jti-1", user_id="u-1", tenant_id="t-1"
        )
        assert entry.token_type == "access"
        assert entry.reason == "logout"
        assert isinstance(entry.revoked_at, datetime)
        assert entry.original_expiry is None

    def test_create_entry_all_fields(self) -> None:
        """Entry accepts all explicit values."""
        now = datetime.now(timezone.utc)
        expiry = now + timedelta(hours=1)
        entry = RevocationEntry(
            jti="jti-2",
            user_id="u-2",
            tenant_id="t-2",
            token_type="refresh",
            reason="password_change",
            revoked_at=now,
            original_expiry=expiry,
        )
        assert entry.token_type == "refresh"
        assert entry.reason == "password_change"
        assert entry.original_expiry == expiry


# ============================================================================
# TestRevocationService
# ============================================================================


class TestRevocationService:
    """Tests for RevocationService.revoke_token and is_revoked."""

    # ------------------------------------------------------------------
    # revoke_token
    # ------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_revoke_token_adds_to_redis(
        self, service: RevocationService, redis_client
    ) -> None:
        """revoke_token writes a key to Redis."""
        result = await service.revoke_token(
            jti="jti-1", user_id="u-1", tenant_id="t-1"
        )
        assert result is True
        redis_client.set.assert_awaited()
        key_arg = redis_client.set.call_args.args[0]
        assert key_arg == f"{_REDIS_PREFIX}jti-1"

    @pytest.mark.asyncio
    async def test_revoke_token_adds_to_postgresql(
        self, service: RevocationService, db_conn
    ) -> None:
        """revoke_token inserts a row into PostgreSQL."""
        await service.revoke_token(
            jti="jti-2", user_id="u-1", tenant_id="t-1"
        )
        db_conn.execute.assert_awaited()
        sql = db_conn.execute.call_args.args[0]
        assert "INSERT INTO security.token_blacklist" in sql

    @pytest.mark.asyncio
    async def test_revoke_token_with_reason_stored(
        self, service: RevocationService, db_conn
    ) -> None:
        """Revocation reason is passed through to PostgreSQL."""
        await service.revoke_token(
            jti="jti-r",
            user_id="u-1",
            tenant_id="t-1",
            reason="admin_revoke",
        )
        call_args = db_conn.execute.call_args.args
        assert "admin_revoke" in call_args

    @pytest.mark.asyncio
    async def test_revoke_token_with_original_expiry_sets_ttl(
        self, service: RevocationService, redis_client
    ) -> None:
        """When original_expiry is given, Redis TTL equals the remaining time."""
        expiry = datetime.now(timezone.utc) + timedelta(seconds=600)
        await service.revoke_token(
            jti="jti-e",
            user_id="u-1",
            tenant_id="t-1",
            original_expiry=expiry,
        )
        call_kwargs = redis_client.set.call_args.kwargs
        # TTL should be approximately 600 (allow some clock drift)
        assert 595 <= call_kwargs["ex"] <= 605

    @pytest.mark.asyncio
    async def test_revoke_token_without_expiry_uses_default_ttl(
        self, service: RevocationService, redis_client
    ) -> None:
        """When no original_expiry, uses the 24h default TTL."""
        await service.revoke_token(
            jti="jti-d", user_id="u-1", tenant_id="t-1"
        )
        call_kwargs = redis_client.set.call_args.kwargs
        assert call_kwargs["ex"] == _DEFAULT_TTL_SECONDS

    @pytest.mark.asyncio
    async def test_revoke_falls_back_to_memory_when_both_layers_fail(
        self, service: RevocationService, redis_client, db_conn
    ) -> None:
        """When Redis and PG both fail, JTI is stored in-memory."""
        redis_client.set.side_effect = ConnectionError("Redis down")
        db_conn.execute.side_effect = Exception("PG down")
        result = await service.revoke_token(
            jti="jti-mem", user_id="u-1", tenant_id="t-1"
        )
        assert result is True
        assert "jti-mem" in service._memory_blacklist

    @pytest.mark.asyncio
    async def test_revoke_token_redis_only(
        self, service_redis_only: RevocationService, redis_client
    ) -> None:
        """With only Redis available, revocation succeeds via Redis."""
        result = await service_redis_only.revoke_token(
            jti="jti-ro", user_id="u-1", tenant_id="t-1"
        )
        assert result is True
        redis_client.set.assert_awaited()

    @pytest.mark.asyncio
    async def test_revoke_token_memory_only(
        self, service_memory_only: RevocationService
    ) -> None:
        """With no backends, falls back to in-memory set."""
        result = await service_memory_only.revoke_token(
            jti="jti-mo", user_id="u-1", tenant_id="t-1"
        )
        assert result is True
        assert "jti-mo" in service_memory_only._memory_blacklist

    # ------------------------------------------------------------------
    # is_revoked
    # ------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_is_revoked_checks_redis_first(
        self, service: RevocationService, redis_client
    ) -> None:
        """is_revoked returns True immediately when Redis has the key."""
        redis_client.get.return_value = b"1"
        result = await service.is_revoked("jti-1")
        assert result is True

    @pytest.mark.asyncio
    async def test_not_revoked_returns_false(
        self, service: RevocationService, redis_client, db_conn
    ) -> None:
        """When JTI is not in any layer, returns False."""
        redis_client.get.return_value = None
        db_conn.fetchrow.return_value = None
        result = await service.is_revoked("jti-clean")
        assert result is False

    @pytest.mark.asyncio
    async def test_is_revoked_falls_back_to_postgresql(
        self, service: RevocationService, redis_client, db_conn
    ) -> None:
        """When Redis misses, falls through to PostgreSQL."""
        redis_client.get.return_value = None
        db_conn.fetchrow.return_value = {"jti": "jti-pg"}  # found in PG
        result = await service.is_revoked("jti-pg")
        assert result is True

    @pytest.mark.asyncio
    async def test_is_revoked_caches_pg_result_in_redis(
        self, service: RevocationService, redis_client, db_conn
    ) -> None:
        """When found in PG but not Redis, promotes to Redis."""
        redis_client.get.return_value = None
        db_conn.fetchrow.return_value = {"jti": "jti-promote"}
        await service.is_revoked("jti-promote")
        # Should call redis.set to promote
        assert redis_client.set.await_count >= 1

    @pytest.mark.asyncio
    async def test_is_revoked_checks_memory_fallback(
        self, service_memory_only: RevocationService
    ) -> None:
        """When no backends, checks in-memory set."""
        service_memory_only._memory_blacklist.add("jti-mem")
        result = await service_memory_only.is_revoked("jti-mem")
        assert result is True

    @pytest.mark.asyncio
    async def test_redis_failure_graceful_fallback(
        self, service: RevocationService, redis_client, db_conn
    ) -> None:
        """When Redis GET fails, falls through to PG gracefully."""
        redis_client.get.side_effect = ConnectionError("Redis down")
        db_conn.fetchrow.return_value = {"jti": "jti-fallback"}
        result = await service.is_revoked("jti-fallback")
        assert result is True

    # ------------------------------------------------------------------
    # revoke_all_for_user
    # ------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_revoke_all_for_user(
        self, service: RevocationService, db_conn
    ) -> None:
        """revoke_all_for_user queries PG for JTIs and revokes them."""
        db_conn.fetch.return_value = [
            {"jti": "jti-a"},
            {"jti": "jti-b"},
        ]
        db_conn.execute.return_value = "UPDATE 2"
        count = await service.revoke_all_for_user("u-1")
        assert count == 2

    @pytest.mark.asyncio
    async def test_revoke_all_for_user_no_db(
        self, service_memory_only: RevocationService
    ) -> None:
        """revoke_all_for_user without DB returns 0."""
        count = await service_memory_only.revoke_all_for_user("u-1")
        assert count == 0

    @pytest.mark.asyncio
    async def test_revoke_all_for_user_db_error_handled(
        self, service: RevocationService, db_conn
    ) -> None:
        """Database errors in revoke_all_for_user are caught."""
        db_conn.fetch.side_effect = Exception("PG error")
        count = await service.revoke_all_for_user("u-1")
        assert count == 0

    # ------------------------------------------------------------------
    # revoke_family
    # ------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_revoke_family(
        self, service: RevocationService, db_conn
    ) -> None:
        """revoke_family updates refresh_tokens by family_id."""
        db_conn.execute.return_value = "UPDATE 3"
        count = await service.revoke_family("fam-1")
        assert count == 3

    @pytest.mark.asyncio
    async def test_revoke_family_no_db(
        self, service_memory_only: RevocationService
    ) -> None:
        """revoke_family without DB returns 0."""
        count = await service_memory_only.revoke_family("fam-1")
        assert count == 0

    @pytest.mark.asyncio
    async def test_revoke_family_db_error_handled(
        self, service: RevocationService, db_conn
    ) -> None:
        """Database errors in revoke_family are caught."""
        db_conn.execute.side_effect = Exception("PG error")
        count = await service.revoke_family("fam-1")
        assert count == 0

    # ------------------------------------------------------------------
    # cleanup_expired
    # ------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_cleanup_expired_removes_old_entries(
        self, service: RevocationService, db_conn
    ) -> None:
        """cleanup_expired deletes rows whose original_expiry is past."""
        db_conn.execute.return_value = "DELETE 5"
        count = await service.cleanup_expired()
        assert count == 5

    @pytest.mark.asyncio
    async def test_cleanup_expired_no_db(
        self, service_memory_only: RevocationService
    ) -> None:
        """cleanup_expired without DB returns 0."""
        count = await service_memory_only.cleanup_expired()
        assert count == 0

    @pytest.mark.asyncio
    async def test_cleanup_expired_db_error(
        self, service: RevocationService, db_conn
    ) -> None:
        """cleanup_expired handles PG errors gracefully."""
        db_conn.execute.side_effect = Exception("PG error")
        count = await service.cleanup_expired()
        assert count == 0

    # ------------------------------------------------------------------
    # get_revocation_count
    # ------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_get_revocation_count_total(
        self, service: RevocationService, db_conn
    ) -> None:
        """get_revocation_count returns total count."""
        db_conn.fetchrow.return_value = {"cnt": 42}
        count = await service.get_revocation_count()
        assert count == 42

    @pytest.mark.asyncio
    async def test_get_revocation_count_for_user(
        self, service: RevocationService, db_conn
    ) -> None:
        """get_revocation_count can filter by user_id."""
        db_conn.fetchrow.return_value = {"cnt": 7}
        count = await service.get_revocation_count(user_id="u-1")
        assert count == 7

    @pytest.mark.asyncio
    async def test_get_revocation_count_fallback_memory(
        self, service_memory_only: RevocationService
    ) -> None:
        """Fallback to memory set count when no DB."""
        service_memory_only._memory_blacklist.update(["a", "b", "c"])
        count = await service_memory_only.get_revocation_count()
        assert count == 3

    # ------------------------------------------------------------------
    # _compute_ttl
    # ------------------------------------------------------------------

    def test_compute_ttl_with_expiry(self) -> None:
        """TTL is remaining seconds until original_expiry."""
        now = datetime.now(timezone.utc)
        expiry = now + timedelta(seconds=300)
        ttl = RevocationService._compute_ttl(expiry, now)
        assert 299 <= ttl <= 301

    def test_compute_ttl_with_past_expiry_clamps_to_1(self) -> None:
        """TTL is at least 1 even when expiry is in the past."""
        now = datetime.now(timezone.utc)
        expiry = now - timedelta(seconds=100)
        ttl = RevocationService._compute_ttl(expiry, now)
        assert ttl == 1

    def test_compute_ttl_none_uses_default(self) -> None:
        """When original_expiry is None, uses the 24h default."""
        now = datetime.now(timezone.utc)
        ttl = RevocationService._compute_ttl(None, now)
        assert ttl == _DEFAULT_TTL_SECONDS


# ============================================================================
# Test _parse_command_count
# ============================================================================


class TestParseCommandCount:
    """Tests for the _parse_command_count helper."""

    def test_update_result(self) -> None:
        assert _parse_command_count("UPDATE 5") == 5

    def test_delete_result(self) -> None:
        assert _parse_command_count("DELETE 12") == 12

    def test_zero_result(self) -> None:
        assert _parse_command_count("UPDATE 0") == 0

    def test_non_string_returns_zero(self) -> None:
        assert _parse_command_count(None) == 0
        assert _parse_command_count(42) == 0

    def test_no_digit_returns_zero(self) -> None:
        assert _parse_command_count("UPDATE") == 0

    def test_insert_result(self) -> None:
        assert _parse_command_count("INSERT 0 1") == 1
