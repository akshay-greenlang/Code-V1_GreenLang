# -*- coding: utf-8 -*-
"""
Token Revocation Service - JWT Authentication Service (SEC-001)

Implements a two-layer token revocation system:

* **L1 -- Redis SET with per-JTI TTL** (fast, hot path, O(1) lookup).
* **L2 -- PostgreSQL ``security.token_blacklist`` table** (durable
  fallback, audit-grade persistence).

On ``revoke_token`` both layers are written.  On ``is_revoked`` Redis
is checked first; if the key is absent *and* Redis is available, the
service falls through to PostgreSQL.  Entries found only in PG are
promoted back to Redis so subsequent checks hit L1.

Expired blacklist entries are cleaned from PostgreSQL by
``cleanup_expired`` which should be invoked from a K8s CronJob on a
daily cadence.

All methods are ``async``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Set

logger = logging.getLogger(__name__)

# Redis key prefix for the revocation set
_REDIS_PREFIX = "gl:auth:revoked:"

# Default TTL ceiling when original_expiry is unknown (24 h)
_DEFAULT_TTL_SECONDS = 86_400


# ---------------------------------------------------------------------------
# Value objects
# ---------------------------------------------------------------------------


@dataclass
class RevocationEntry:
    """A single JTI revocation record.

    Attributes:
        jti: The revoked JWT identifier.
        user_id: Owner of the revoked token.
        tenant_id: Tenant scope.
        token_type: ``"access"`` or ``"refresh"``.
        reason: Human-readable revocation reason.
        revoked_at: UTC timestamp of the revocation event.
        original_expiry: The token's original expiry so we can prune
            the entry after it would have expired naturally.
    """

    jti: str
    user_id: str
    tenant_id: str
    token_type: str = "access"
    reason: str = "logout"
    revoked_at: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    original_expiry: Optional[datetime] = None


# ---------------------------------------------------------------------------
# Service
# ---------------------------------------------------------------------------


class RevocationService:
    """Manages the JTI blacklist for token revocation.

    Uses a two-layer cache-aside pattern:

    1. **Redis** -- ``SET`` with TTL equal to the token's remaining
       lifetime.  Provides sub-millisecond revocation checks.
    2. **PostgreSQL** -- ``security.token_blacklist`` table.  Provides
       durable storage and audit trail.  Used as a fallback when the
       Redis key has been evicted or Redis is down.

    Both layers are optional.  When neither is supplied the service
    falls back to an in-memory ``set`` (suitable for tests and single-
    process development).

    Args:
        redis_client: An async Redis client (e.g. ``redis.asyncio``).
        db_pool: An async PostgreSQL connection pool (e.g.
            ``asyncpg.Pool`` or ``psycopg_pool.AsyncConnectionPool``).
    """

    def __init__(
        self,
        redis_client: Any = None,
        db_pool: Any = None,
    ) -> None:
        self._redis = redis_client
        self._db = db_pool

        # Fallback in-memory set for environments without Redis/PG
        self._memory_blacklist: Set[str] = set()

        logger.info(
            "RevocationService initialised  redis=%s  db=%s",
            "yes" if redis_client else "no",
            "yes" if db_pool else "no",
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def revoke_token(
        self,
        jti: str,
        user_id: str,
        tenant_id: str,
        token_type: str = "access",
        reason: str = "logout",
        original_expiry: Optional[datetime] = None,
    ) -> bool:
        """Revoke a token by adding its JTI to the blacklist.

        Writes to *both* Redis (L1) and PostgreSQL (L2) when available.

        Args:
            jti: JWT identifier to revoke.
            user_id: Owner of the token.
            tenant_id: Tenant scope.
            token_type: ``"access"`` or ``"refresh"``.
            reason: Human-readable reason (e.g. ``"logout"``,
                ``"password_change"``, ``"admin_revoke"``).
            original_expiry: When the token would have expired
                naturally.  Used to set the Redis TTL so the key
                auto-evicts after the token is no longer valid.

        Returns:
            ``True`` if at least one layer accepted the revocation.
        """
        now = datetime.now(timezone.utc)
        ttl = self._compute_ttl(original_expiry, now)

        entry = RevocationEntry(
            jti=jti,
            user_id=user_id,
            tenant_id=tenant_id,
            token_type=token_type,
            reason=reason,
            revoked_at=now,
            original_expiry=original_expiry,
        )

        success = False

        # L1 -- Redis
        redis_ok = await self._redis_revoke(jti, ttl)
        success = success or redis_ok

        # L2 -- PostgreSQL
        pg_ok = await self._pg_revoke(entry)
        success = success or pg_ok

        # Fallback in-memory
        if not redis_ok and not pg_ok:
            self._memory_blacklist.add(jti)
            success = True

        logger.info(
            "Token revoked  jti=%s  user=%s  tenant=%s  reason=%s  "
            "redis=%s  pg=%s",
            jti,
            user_id,
            tenant_id,
            reason,
            redis_ok,
            pg_ok,
        )
        return success

    async def is_revoked(self, jti: str) -> bool:
        """Check whether a JTI has been revoked.

        Lookup order: Redis --> PostgreSQL --> in-memory set.

        When the JTI is found in PostgreSQL but not in Redis, it is
        promoted back into Redis so subsequent calls are served from L1.

        Args:
            jti: The JWT identifier to check.

        Returns:
            ``True`` if the JTI is present in any revocation layer.
        """
        # L1 -- Redis (fast path)
        redis_result = await self._redis_is_revoked(jti)
        if redis_result is True:
            return True

        # L2 -- PostgreSQL
        pg_result = await self._pg_is_revoked(jti)
        if pg_result is True:
            # Promote to Redis for subsequent fast lookups
            await self._redis_revoke(jti, _DEFAULT_TTL_SECONDS)
            return True

        # Fallback in-memory
        return jti in self._memory_blacklist

    async def revoke_all_for_user(
        self,
        user_id: str,
        reason: str = "bulk_revoke",
    ) -> int:
        """Revoke all tokens belonging to a specific user.

        This is typically invoked on password change or account
        compromise.  Fetches outstanding JTIs from PostgreSQL and
        revokes each one.

        Args:
            user_id: The user whose tokens should be revoked.
            reason: Revocation reason for audit trail.

        Returns:
            The number of tokens revoked.
        """
        count = 0

        if self._db is not None:
            try:
                jtis = await self._pg_fetch_jtis_for_user(user_id)
                for jti in jtis:
                    await self._redis_revoke(jti, _DEFAULT_TTL_SECONDS)
                    count += 1

                # Bulk-update PostgreSQL
                count = max(count, await self._pg_revoke_all_for_user(user_id, reason))
            except Exception as exc:
                logger.error(
                    "revoke_all_for_user failed  user=%s: %s", user_id, exc
                )
        else:
            # In-memory only -- no user-level index available
            logger.warning(
                "revoke_all_for_user called without DB pool; "
                "only future checks on known JTIs will succeed"
            )

        logger.info(
            "Bulk revocation complete  user=%s  reason=%s  count=%d",
            user_id,
            reason,
            count,
        )
        return count

    async def revoke_family(
        self,
        family_id: str,
        reason: str = "family_revoke",
    ) -> int:
        """Revoke all refresh tokens sharing a ``family_id``.

        Used by the ``RefreshTokenManager`` when token reuse is
        detected (potential theft).

        Args:
            family_id: The refresh-token family identifier.
            reason: Revocation reason.

        Returns:
            Number of tokens revoked.
        """
        count = 0

        if self._db is not None:
            try:
                count = await self._pg_revoke_family(family_id, reason)
            except Exception as exc:
                logger.error(
                    "revoke_family failed  family=%s: %s", family_id, exc
                )

        logger.info(
            "Family revocation complete  family=%s  reason=%s  count=%d",
            family_id,
            reason,
            count,
        )
        return count

    async def cleanup_expired(self) -> int:
        """Remove expired revocation entries from PostgreSQL.

        Should be called periodically (e.g. daily K8s CronJob).
        Entries whose ``original_expiry`` is in the past can be safely
        pruned because the corresponding token can no longer pass
        signature + expiry validation anyway.

        Returns:
            Number of rows deleted.
        """
        if self._db is None:
            return 0

        try:
            return await self._pg_cleanup_expired()
        except Exception as exc:
            logger.error("cleanup_expired failed: %s", exc)
            return 0

    async def get_revocation_count(
        self,
        user_id: Optional[str] = None,
    ) -> int:
        """Count active revocation entries.

        Args:
            user_id: When supplied, count only entries for this user.

        Returns:
            Number of active blacklist entries.
        """
        if self._db is not None:
            try:
                return await self._pg_count(user_id)
            except Exception as exc:
                logger.error("get_revocation_count failed: %s", exc)

        # Fallback: in-memory count (no user filtering)
        return len(self._memory_blacklist)

    # ------------------------------------------------------------------
    # Redis helpers (L1)
    # ------------------------------------------------------------------

    async def _redis_revoke(self, jti: str, ttl: int) -> bool:
        """Write a revocation entry to Redis with a bounded TTL."""
        if self._redis is None:
            return False
        try:
            key = f"{_REDIS_PREFIX}{jti}"
            await self._redis.set(key, "1", ex=max(ttl, 1))
            return True
        except Exception as exc:
            logger.warning("Redis revoke failed for jti=%s: %s", jti, exc)
            return False

    async def _redis_is_revoked(self, jti: str) -> Optional[bool]:
        """Check Redis for a revoked JTI.  Returns None on error."""
        if self._redis is None:
            return None
        try:
            key = f"{_REDIS_PREFIX}{jti}"
            result = await self._redis.get(key)
            return result is not None
        except Exception as exc:
            logger.warning("Redis is_revoked check failed for jti=%s: %s", jti, exc)
            return None

    # ------------------------------------------------------------------
    # PostgreSQL helpers (L2)
    # ------------------------------------------------------------------

    async def _pg_revoke(self, entry: RevocationEntry) -> bool:
        """Insert a revocation record into PostgreSQL."""
        if self._db is None:
            return False
        try:
            async with self._db.connection() as conn:
                await conn.execute(
                    """
                    INSERT INTO security.token_blacklist
                        (jti, user_id, tenant_id, token_type, reason,
                         revoked_at, original_expiry)
                    VALUES ($1, $2, $3, $4, $5, $6, $7)
                    ON CONFLICT (jti) DO NOTHING
                    """,
                    entry.jti,
                    entry.user_id,
                    entry.tenant_id,
                    entry.token_type,
                    entry.reason,
                    entry.revoked_at,
                    entry.original_expiry,
                )
            return True
        except Exception as exc:
            logger.error("PG revoke failed for jti=%s: %s", entry.jti, exc)
            return False

    async def _pg_is_revoked(self, jti: str) -> bool:
        """Check PostgreSQL for a revoked JTI."""
        if self._db is None:
            return False
        try:
            async with self._db.connection() as conn:
                row = await conn.fetchrow(
                    "SELECT 1 FROM security.token_blacklist WHERE jti = $1",
                    jti,
                )
            return row is not None
        except Exception as exc:
            logger.error("PG is_revoked check failed for jti=%s: %s", jti, exc)
            return False

    async def _pg_fetch_jtis_for_user(self, user_id: str) -> List[str]:
        """Fetch all non-expired JTIs for a user from PostgreSQL."""
        if self._db is None:
            return []
        try:
            async with self._db.connection() as conn:
                rows = await conn.fetch(
                    """
                    SELECT jti FROM security.token_blacklist
                    WHERE user_id = $1
                      AND (original_expiry IS NULL
                           OR original_expiry > NOW())
                    """,
                    user_id,
                )
            return [row["jti"] for row in rows]
        except Exception as exc:
            logger.error("PG fetch JTIs for user=%s failed: %s", user_id, exc)
            return []

    async def _pg_revoke_all_for_user(
        self, user_id: str, reason: str
    ) -> int:
        """Mark all refresh tokens for a user as revoked in PG."""
        if self._db is None:
            return 0
        try:
            async with self._db.connection() as conn:
                result = await conn.execute(
                    """
                    UPDATE security.refresh_tokens
                    SET status = 'revoked',
                        revoked_at = NOW(),
                        revoke_reason = $2
                    WHERE user_id = $1
                      AND status = 'active'
                    """,
                    user_id,
                    reason,
                )
                # asyncpg returns "UPDATE N"
                return _parse_command_count(result)
        except Exception as exc:
            logger.error(
                "PG revoke_all_for_user failed  user=%s: %s", user_id, exc
            )
            return 0

    async def _pg_revoke_family(self, family_id: str, reason: str) -> int:
        """Revoke all refresh tokens in a family."""
        if self._db is None:
            return 0
        try:
            async with self._db.connection() as conn:
                result = await conn.execute(
                    """
                    UPDATE security.refresh_tokens
                    SET status = 'revoked',
                        revoked_at = NOW(),
                        revoke_reason = $2
                    WHERE family_id = $1
                      AND status IN ('active', 'rotated')
                    """,
                    family_id,
                    reason,
                )
                return _parse_command_count(result)
        except Exception as exc:
            logger.error(
                "PG revoke_family failed  family=%s: %s", family_id, exc
            )
            return 0

    async def _pg_cleanup_expired(self) -> int:
        """Delete expired revocation entries from PostgreSQL."""
        if self._db is None:
            return 0
        try:
            async with self._db.connection() as conn:
                result = await conn.execute(
                    """
                    DELETE FROM security.token_blacklist
                    WHERE original_expiry IS NOT NULL
                      AND original_expiry < NOW()
                    """
                )
                return _parse_command_count(result)
        except Exception as exc:
            logger.error("PG cleanup_expired failed: %s", exc)
            return 0

    async def _pg_count(self, user_id: Optional[str] = None) -> int:
        """Count revocation entries, optionally filtered by user."""
        if self._db is None:
            return 0
        try:
            async with self._db.connection() as conn:
                if user_id:
                    row = await conn.fetchrow(
                        "SELECT COUNT(*) AS cnt FROM security.token_blacklist WHERE user_id = $1",
                        user_id,
                    )
                else:
                    row = await conn.fetchrow(
                        "SELECT COUNT(*) AS cnt FROM security.token_blacklist"
                    )
                return row["cnt"] if row else 0
        except Exception as exc:
            logger.error("PG get_revocation_count failed: %s", exc)
            return 0

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_ttl(
        original_expiry: Optional[datetime],
        now: datetime,
    ) -> int:
        """Compute a Redis TTL from the token's original expiry."""
        if original_expiry is not None:
            remaining = int((original_expiry - now).total_seconds())
            return max(remaining, 1)
        return _DEFAULT_TTL_SECONDS


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------


def _parse_command_count(result: Any) -> int:
    """Extract the affected-row count from an asyncpg command result.

    asyncpg returns strings like ``"UPDATE 5"`` or ``"DELETE 12"``.
    """
    if isinstance(result, str):
        parts = result.split()
        if len(parts) >= 2 and parts[-1].isdigit():
            return int(parts[-1])
    return 0
