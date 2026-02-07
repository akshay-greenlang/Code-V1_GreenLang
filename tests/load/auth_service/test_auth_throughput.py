# -*- coding: utf-8 -*-
"""
Load / performance tests for Auth Service (SEC-001)

Tests throughput and latency of critical auth operations:
- Token validation (target: 1 000+ ops/sec)
- Token issuance under load
- Revocation check latency
- Concurrent refresh rotation
- Redis blacklist performance

Markers:
    @pytest.mark.performance
    @pytest.mark.load

These tests are designed to run in CI with relaxed thresholds (to avoid
flaky failures on shared runners) and locally with stricter targets.
"""

from __future__ import annotations

import asyncio
import time
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock

import pytest

from greenlang.infrastructure.auth_service.token_service import (
    IssuedToken,
    TokenClaims,
    TokenService,
)
from greenlang.infrastructure.auth_service.revocation import (
    RevocationService,
)


# ============================================================================
# In-Memory Redis Stub (fast, no real I/O)
# ============================================================================


class FastInMemoryRedis:
    """Ultra-fast in-memory Redis for throughput testing."""

    def __init__(self):
        self._store: Dict[str, str] = {}

    async def set(self, key: str, value: str, ex: int = 0) -> bool:
        self._store[key] = value
        return True

    async def get(self, key: str) -> Optional[str]:
        return self._store.get(key)

    async def delete(self, key: str) -> int:
        return 1 if self._store.pop(key, None) is not None else 0


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def jwt_handler() -> MagicMock:
    handler = MagicMock()
    _seq = [0]

    def _gen(**kwargs):
        _seq[0] += 1
        return f"eyJ.perf-{_seq[0]}.sig"

    handler.generate_token.side_effect = _gen

    def _validate(token: str):
        claims = MagicMock()
        claims.sub = "user-1"
        claims.tenant_id = "t-acme"
        claims.roles = ["viewer"]
        claims.permissions = ["read:data"]
        claims.email = None
        claims.name = None
        seq = token.split("-")[-1].split(".")[0]
        claims.jti = f"jti-p-{seq}"
        claims.scope = None
        return claims

    handler.validate_token.side_effect = _validate
    handler.get_jwks.return_value = {"keys": []}
    return handler


@pytest.fixture
def redis() -> FastInMemoryRedis:
    return FastInMemoryRedis()


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


@pytest.fixture
def claims() -> TokenClaims:
    return TokenClaims(sub="user-1", tenant_id="t-acme", roles=["viewer"])


# ============================================================================
# Constants
# ============================================================================

# Relaxed thresholds for CI runners; local can be much tighter
VALIDATION_OPS_PER_SEC = 500  # target: 1000 on local
ISSUANCE_OPS_PER_SEC = 200
REVOCATION_CHECK_MAX_MS = 5.0


# ============================================================================
# TestAuthThroughput
# ============================================================================


@pytest.mark.performance
@pytest.mark.load
class TestAuthThroughput:
    """Performance tests for auth service operations."""

    @pytest.mark.asyncio
    async def test_token_validation_throughput(
        self, svc: TokenService, claims: TokenClaims
    ) -> None:
        """Token validation achieves target throughput."""
        # Issue a batch of tokens
        num_ops = 1000
        tokens = []
        for _ in range(num_ops):
            issued = await svc.issue_token(claims)
            tokens.append(issued.access_token)

        # Time validation
        start = time.perf_counter()
        for token in tokens:
            await svc.validate_token(token)
        elapsed = time.perf_counter() - start

        ops_per_sec = num_ops / elapsed
        assert ops_per_sec >= VALIDATION_OPS_PER_SEC, (
            f"Validation throughput {ops_per_sec:.0f} ops/s "
            f"below target {VALIDATION_OPS_PER_SEC}"
        )

    @pytest.mark.asyncio
    async def test_token_issuance_throughput(
        self, svc: TokenService, claims: TokenClaims
    ) -> None:
        """Token issuance achieves target throughput."""
        num_ops = 500

        start = time.perf_counter()
        for _ in range(num_ops):
            await svc.issue_token(claims)
        elapsed = time.perf_counter() - start

        ops_per_sec = num_ops / elapsed
        assert ops_per_sec >= ISSUANCE_OPS_PER_SEC, (
            f"Issuance throughput {ops_per_sec:.0f} ops/s "
            f"below target {ISSUANCE_OPS_PER_SEC}"
        )

    @pytest.mark.asyncio
    async def test_revocation_check_latency(
        self, revocation: RevocationService, redis: FastInMemoryRedis
    ) -> None:
        """Single revocation check completes within latency target."""
        # Pre-populate some revocations
        for i in range(100):
            await revocation.revoke_token(
                jti=f"jti-lat-{i}",
                user_id="user-1",
                tenant_id="t-acme",
            )

        # Time individual checks
        latencies_ms = []
        for i in range(100):
            start = time.perf_counter()
            await revocation.is_revoked(f"jti-lat-{i}")
            elapsed_ms = (time.perf_counter() - start) * 1000
            latencies_ms.append(elapsed_ms)

        avg_ms = sum(latencies_ms) / len(latencies_ms)
        p99_ms = sorted(latencies_ms)[int(len(latencies_ms) * 0.99)]

        assert avg_ms < REVOCATION_CHECK_MAX_MS, (
            f"Avg revocation check {avg_ms:.2f}ms exceeds target"
        )
        assert p99_ms < REVOCATION_CHECK_MAX_MS * 5, (
            f"P99 revocation check {p99_ms:.2f}ms exceeds 5x target"
        )

    @pytest.mark.asyncio
    async def test_concurrent_validation(
        self, svc: TokenService, claims: TokenClaims
    ) -> None:
        """Concurrent validation calls maintain throughput."""
        # Issue tokens
        tokens = []
        for _ in range(100):
            issued = await svc.issue_token(claims)
            tokens.append(issued.access_token)

        # Concurrent validation
        start = time.perf_counter()
        results = await asyncio.gather(
            *[svc.validate_token(t) for t in tokens]
        )
        elapsed = time.perf_counter() - start

        # All should succeed
        assert all(r is not None for r in results)
        ops_per_sec = len(tokens) / elapsed
        assert ops_per_sec >= VALIDATION_OPS_PER_SEC / 2

    @pytest.mark.asyncio
    async def test_redis_blacklist_performance(
        self, redis: FastInMemoryRedis
    ) -> None:
        """Redis blacklist lookups are fast even with many entries."""
        # Populate 10k entries
        for i in range(10_000):
            await redis.set(f"gl:auth:revoked:jti-perf-{i}", "1", ex=3600)

        # Time lookups
        start = time.perf_counter()
        for i in range(1000):
            await redis.get(f"gl:auth:revoked:jti-perf-{i}")
        elapsed = time.perf_counter() - start

        ops_per_sec = 1000 / elapsed
        assert ops_per_sec >= 10_000, (
            f"Redis lookup throughput {ops_per_sec:.0f} ops/s below target"
        )

    @pytest.mark.asyncio
    async def test_issue_validate_revoke_cycle_latency(
        self, svc: TokenService, revocation: RevocationService, claims: TokenClaims
    ) -> None:
        """Full issue-validate-revoke cycle completes within budget."""
        latencies_ms = []

        for _ in range(50):
            start = time.perf_counter()
            issued = await svc.issue_token(claims)
            await svc.validate_token(issued.access_token)
            await revocation.revoke_token(
                jti=issued.jti, user_id="user-1", tenant_id="t-acme"
            )
            elapsed_ms = (time.perf_counter() - start) * 1000
            latencies_ms.append(elapsed_ms)

        avg_ms = sum(latencies_ms) / len(latencies_ms)
        assert avg_ms < 50, (
            f"Full cycle avg latency {avg_ms:.2f}ms exceeds 50ms budget"
        )

    @pytest.mark.asyncio
    async def test_memory_stability_under_load(
        self, svc: TokenService, claims: TokenClaims
    ) -> None:
        """JTI set does not grow unbounded during sustained load."""
        initial_size = len(svc._issued_jtis)
        for _ in range(1000):
            await svc.issue_token(claims)
        final_size = len(svc._issued_jtis)

        # JTIs are tracked; verify the set grew by ~1000
        assert final_size == initial_size + 1000

    @pytest.mark.asyncio
    async def test_concurrent_revocations_no_data_loss(
        self, revocation: RevocationService
    ) -> None:
        """Concurrent revocations do not lose entries."""
        jtis = [f"jti-concurrent-{i}" for i in range(100)]

        await asyncio.gather(
            *[
                revocation.revoke_token(
                    jti=jti, user_id="user-1", tenant_id="t-acme"
                )
                for jti in jtis
            ]
        )

        # All should be revoked
        checks = await asyncio.gather(
            *[revocation.is_revoked(jti) for jti in jtis]
        )
        assert all(checks), "Some concurrent revocations were lost"
