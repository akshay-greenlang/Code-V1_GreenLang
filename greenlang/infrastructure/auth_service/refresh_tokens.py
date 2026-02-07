# -*- coding: utf-8 -*-
"""
Refresh Token Manager - JWT Authentication Service (SEC-001)

Implements **opaque** refresh tokens with rotation and family-based
reuse detection:

* Tokens are random 48-byte URL-safe strings generated via
  ``secrets.token_urlsafe(48)``.
* Only the **SHA-256 hash** is persisted in
  ``security.refresh_tokens``; the plain-text value is returned to the
  client exactly once.
* Each token belongs to a *family* (UUID).  When a token is rotated a
  new token is issued in the same family and the old token is marked
  ``rotated``.
* **Reuse detection**: if a rotated token is presented again, the
  *entire* family is revoked (all tokens descended from the same login
  session), because a replay implies the token was leaked.
* A short *grace period* (default 5 s) prevents race conditions when
  concurrent requests carry the same refresh token.

Storage layout:
    PostgreSQL ``security.refresh_tokens`` (durable, source of truth).
    Redis ``gl:auth:rt:{sha256}`` (optional, hot-path lookup cache).

All methods are ``async``.
"""

from __future__ import annotations

import hashlib
import logging
import secrets
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Redis key prefix for refresh-token lookup cache
_RT_REDIS_PREFIX = "gl:auth:rt:"


# ---------------------------------------------------------------------------
# Value objects
# ---------------------------------------------------------------------------


@dataclass
class RefreshTokenResult:
    """Returned to the caller after issuing or rotating a refresh token.

    Attributes:
        token: The plain opaque token string -- returned to the client
            and **never** stored server-side.
        family_id: The refresh-token family this token belongs to.
        expires_at: Absolute UTC expiry of the token.
        is_new_family: ``True`` when a brand-new family was created
            (first login), ``False`` on rotation.
    """

    token: str
    family_id: str
    expires_at: datetime
    is_new_family: bool


@dataclass
class RefreshTokenRecord:
    """Internal representation of a refresh-token row in PostgreSQL.

    Attributes:
        id: Primary key (UUID).
        token_hash: SHA-256 hex digest of the opaque token.
        user_id: Owner of the token.
        tenant_id: Tenant scope.
        family_id: Family identifier for rotation tracking.
        status: ``"active"``, ``"rotated"``, or ``"revoked"``.
        device_fingerprint: Optional device identifier.
        ip_address: IP address at issuance time.
        user_agent: User-Agent string at issuance time.
        created_at: When the token was issued.
        expires_at: When the token expires.
        rotated_at: When the token was replaced by a successor.
        revoked_at: When the token was revoked.
        revoke_reason: Why the token was revoked.
    """

    id: str
    token_hash: str
    user_id: str
    tenant_id: str
    family_id: str
    status: str = "active"
    device_fingerprint: Optional[str] = None
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    created_at: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    expires_at: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    rotated_at: Optional[datetime] = None
    revoked_at: Optional[datetime] = None
    revoke_reason: Optional[str] = None


# ---------------------------------------------------------------------------
# Service
# ---------------------------------------------------------------------------


class RefreshTokenManager:
    """Manages opaque refresh-token lifecycle with rotation and reuse
    detection.

    Args:
        db_pool: Async PostgreSQL connection pool.
        redis_client: Async Redis client (optional, for hot-path
            cache).
        revocation_service: ``RevocationService`` used to revoke an
            entire family when reuse is detected.
        token_lifetime_days: Refresh-token lifetime in days.
        max_family_size: Maximum number of tokens in a single family
            before forced revocation (prevents infinite rotation).
        reuse_grace_seconds: Grace window during which a just-rotated
            token is still accepted (handles concurrent requests).

    Example:
        >>> mgr = RefreshTokenManager(db_pool=pool)
        >>> result = await mgr.issue_refresh_token(
        ...     user_id="u-1", tenant_id="t-acme"
        ... )
        >>> rotated = await mgr.rotate_refresh_token(result.token)
    """

    def __init__(
        self,
        db_pool: Any = None,
        redis_client: Any = None,
        revocation_service: Any = None,
        token_lifetime_days: int = 7,
        max_family_size: int = 30,
        reuse_grace_seconds: int = 5,
    ) -> None:
        self._db = db_pool
        self._redis = redis_client
        self._revocation_service = revocation_service
        self._token_lifetime_days = token_lifetime_days
        self._max_family_size = max_family_size
        self._reuse_grace_seconds = reuse_grace_seconds

        # In-memory store for environments without PostgreSQL (tests)
        self._memory_store: Dict[str, RefreshTokenRecord] = {}

        logger.info(
            "RefreshTokenManager initialised  lifetime=%dd  "
            "max_family=%d  grace=%ds",
            token_lifetime_days,
            max_family_size,
            reuse_grace_seconds,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def issue_refresh_token(
        self,
        user_id: str,
        tenant_id: str,
        family_id: Optional[str] = None,
        device_fingerprint: Optional[str] = None,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None,
    ) -> RefreshTokenResult:
        """Issue a new refresh token.

        If ``family_id`` is *None* a brand-new family is created (first
        login).  Otherwise the token is added to an existing family
        (rotation).

        Args:
            user_id: Owner of the token.
            tenant_id: Tenant scope.
            family_id: Existing family ID, or *None* for a new family.
            device_fingerprint: Optional device identifier.
            ip_address: Client IP at issuance.
            user_agent: Client User-Agent at issuance.

        Returns:
            ``RefreshTokenResult`` with the plain token (returned
            exactly once), family ID, and expiry.
        """
        is_new_family = family_id is None
        if is_new_family:
            family_id = str(uuid.uuid4())

        # Check family size to prevent abuse
        if not is_new_family:
            family_size = await self._get_family_size(family_id)
            if family_size >= self._max_family_size:
                logger.warning(
                    "Family size limit reached  family=%s  size=%d",
                    family_id,
                    family_size,
                )
                await self.revoke_family(family_id)
                # Start a new family
                family_id = str(uuid.uuid4())
                is_new_family = True

        # Generate opaque token
        plain_token = secrets.token_urlsafe(48)
        token_hash = self._hash_token(plain_token)

        now = datetime.now(timezone.utc)
        expires_at = now + timedelta(days=self._token_lifetime_days)

        record = RefreshTokenRecord(
            id=str(uuid.uuid4()),
            token_hash=token_hash,
            user_id=user_id,
            tenant_id=tenant_id,
            family_id=family_id,
            status="active",
            device_fingerprint=device_fingerprint,
            ip_address=ip_address,
            user_agent=user_agent,
            created_at=now,
            expires_at=expires_at,
        )

        await self._store_record(record)
        await self._cache_record(record)

        logger.info(
            "Refresh token issued  user=%s  tenant=%s  family=%s  "
            "new_family=%s  expires=%s",
            user_id,
            tenant_id,
            family_id,
            is_new_family,
            expires_at.isoformat(),
        )

        return RefreshTokenResult(
            token=plain_token,
            family_id=family_id,
            expires_at=expires_at,
            is_new_family=is_new_family,
        )

    async def rotate_refresh_token(self, token: str) -> RefreshTokenResult:
        """Rotate a refresh token: invalidate the old one and issue a
        new token in the same family.

        The rotation pipeline:

        1. Hash the incoming token.
        2. Look up the record by hash.
        3. If the record is ``active`` and not expired -- mark it
           ``rotated`` and issue a new token in the same family.
        4. If the record is ``rotated`` *and* outside the grace period --
           **reuse detected**: revoke the entire family.
        5. If the record is ``revoked`` or expired -- reject.

        Args:
            token: The plain opaque refresh token.

        Returns:
            ``RefreshTokenResult`` for the newly issued token.

        Raises:
            ValueError: If the token is invalid, expired, or reuse is
                detected.
        """
        token_hash = self._hash_token(token)
        record = await self._lookup_record(token_hash)

        if record is None:
            raise ValueError("Refresh token not found")

        now = datetime.now(timezone.utc)

        # Check expiry
        if now >= record.expires_at:
            raise ValueError("Refresh token expired")

        # Check status
        if record.status == "revoked":
            raise ValueError("Refresh token has been revoked")

        if record.status == "rotated":
            # Grace period check
            if record.rotated_at is not None:
                elapsed = (now - record.rotated_at).total_seconds()
                if elapsed <= self._reuse_grace_seconds:
                    # Within grace period -- allow (concurrent request)
                    logger.debug(
                        "Refresh token reuse within grace period  "
                        "family=%s  elapsed=%.1fs",
                        record.family_id,
                        elapsed,
                    )
                    return await self.issue_refresh_token(
                        user_id=record.user_id,
                        tenant_id=record.tenant_id,
                        family_id=record.family_id,
                        device_fingerprint=record.device_fingerprint,
                        ip_address=record.ip_address,
                        user_agent=record.user_agent,
                    )

            # Reuse detected outside grace period -- COMPROMISE
            logger.warning(
                "REFRESH TOKEN REUSE DETECTED  family=%s  user=%s  "
                "tenant=%s -- revoking entire family",
                record.family_id,
                record.user_id,
                record.tenant_id,
            )
            await self.revoke_family(record.family_id)
            raise ValueError(
                "Refresh token reuse detected; family revoked"
            )

        # Status is "active" -- proceed with rotation
        await self._mark_rotated(record.id, now)

        new_result = await self.issue_refresh_token(
            user_id=record.user_id,
            tenant_id=record.tenant_id,
            family_id=record.family_id,
            device_fingerprint=record.device_fingerprint,
            ip_address=record.ip_address,
            user_agent=record.user_agent,
        )

        logger.info(
            "Refresh token rotated  family=%s  user=%s  old_id=%s",
            record.family_id,
            record.user_id,
            record.id,
        )
        return new_result

    async def revoke_token(
        self,
        token: str,
        reason: str = "logout",
    ) -> bool:
        """Revoke a specific refresh token.

        Args:
            token: The plain opaque refresh token.
            reason: Human-readable reason.

        Returns:
            ``True`` if the token was found and revoked.
        """
        token_hash = self._hash_token(token)
        record = await self._lookup_record(token_hash)

        if record is None:
            logger.warning("Revoke called for unknown refresh token")
            return False

        await self._mark_revoked(record.id, reason)
        await self._invalidate_cache(token_hash)

        logger.info(
            "Refresh token revoked  id=%s  family=%s  reason=%s",
            record.id,
            record.family_id,
            reason,
        )
        return True

    async def revoke_family(self, family_id: str) -> int:
        """Revoke all tokens in a family.

        Args:
            family_id: The family identifier.

        Returns:
            Number of tokens revoked.
        """
        count = await self._revoke_family_records(family_id)

        # Also notify the RevocationService if available
        if self._revocation_service is not None:
            try:
                await self._revocation_service.revoke_family(
                    family_id, reason="family_revoke"
                )
            except Exception as exc:
                logger.error(
                    "RevocationService.revoke_family failed: %s", exc
                )

        logger.info(
            "Refresh token family revoked  family=%s  count=%d",
            family_id,
            count,
        )
        return count

    async def revoke_all_for_user(self, user_id: str) -> int:
        """Revoke all refresh tokens for a user.

        Args:
            user_id: The user whose tokens should be revoked.

        Returns:
            Number of tokens revoked.
        """
        count = await self._revoke_user_records(user_id)

        logger.info(
            "All refresh tokens revoked for user  user=%s  count=%d",
            user_id,
            count,
        )
        return count

    async def cleanup_expired(self) -> int:
        """Remove expired and old rotated tokens from PostgreSQL.

        Rotated tokens older than 2x the token lifetime are deleted to
        keep the table manageable.

        Returns:
            Number of rows deleted.
        """
        cutoff = datetime.now(timezone.utc) - timedelta(
            days=self._token_lifetime_days * 2
        )
        return await self._delete_old_records(cutoff)

    # ------------------------------------------------------------------
    # Storage helpers -- PostgreSQL
    # ------------------------------------------------------------------

    async def _store_record(self, record: RefreshTokenRecord) -> None:
        """Insert a refresh-token record into PostgreSQL."""
        if self._db is not None:
            try:
                async with self._db.connection() as conn:
                    await conn.execute(
                        """
                        INSERT INTO security.refresh_tokens
                            (id, token_hash, user_id, tenant_id,
                             family_id, status, device_fingerprint,
                             ip_address, user_agent, created_at,
                             expires_at)
                        VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9,
                                $10, $11)
                        """,
                        record.id,
                        record.token_hash,
                        record.user_id,
                        record.tenant_id,
                        record.family_id,
                        record.status,
                        record.device_fingerprint,
                        record.ip_address,
                        record.user_agent,
                        record.created_at,
                        record.expires_at,
                    )
            except Exception as exc:
                logger.error("PG store refresh token failed: %s", exc)
                # Fall through to in-memory
                self._memory_store[record.token_hash] = record
        else:
            self._memory_store[record.token_hash] = record

    async def _lookup_record(
        self, token_hash: str
    ) -> Optional[RefreshTokenRecord]:
        """Look up a refresh-token record by its hash."""
        # Try Redis cache first
        cached = await self._cache_lookup(token_hash)
        if cached is not None:
            return cached

        # PostgreSQL
        if self._db is not None:
            try:
                async with self._db.connection() as conn:
                    row = await conn.fetchrow(
                        """
                        SELECT id, token_hash, user_id, tenant_id,
                               family_id, status, device_fingerprint,
                               ip_address, user_agent, created_at,
                               expires_at, rotated_at, revoked_at,
                               revoke_reason
                        FROM security.refresh_tokens
                        WHERE token_hash = $1
                        """,
                        token_hash,
                    )
                if row is not None:
                    return self._row_to_record(row)
            except Exception as exc:
                logger.error("PG lookup refresh token failed: %s", exc)

        # In-memory fallback
        return self._memory_store.get(token_hash)

    async def _mark_rotated(
        self, record_id: str, rotated_at: datetime
    ) -> None:
        """Mark a refresh-token record as rotated."""
        if self._db is not None:
            try:
                async with self._db.connection() as conn:
                    await conn.execute(
                        """
                        UPDATE security.refresh_tokens
                        SET status = 'rotated', rotated_at = $2
                        WHERE id = $1
                        """,
                        record_id,
                        rotated_at,
                    )
            except Exception as exc:
                logger.error("PG mark_rotated failed  id=%s: %s", record_id, exc)
        else:
            for rec in self._memory_store.values():
                if rec.id == record_id:
                    rec.status = "rotated"
                    rec.rotated_at = rotated_at
                    break

    async def _mark_revoked(self, record_id: str, reason: str) -> None:
        """Mark a single refresh-token record as revoked."""
        now = datetime.now(timezone.utc)
        if self._db is not None:
            try:
                async with self._db.connection() as conn:
                    await conn.execute(
                        """
                        UPDATE security.refresh_tokens
                        SET status = 'revoked',
                            revoked_at = $2,
                            revoke_reason = $3
                        WHERE id = $1
                        """,
                        record_id,
                        now,
                        reason,
                    )
            except Exception as exc:
                logger.error("PG mark_revoked failed  id=%s: %s", record_id, exc)
        else:
            for rec in self._memory_store.values():
                if rec.id == record_id:
                    rec.status = "revoked"
                    rec.revoked_at = now
                    rec.revoke_reason = reason
                    break

    async def _revoke_family_records(self, family_id: str) -> int:
        """Revoke all records in a family."""
        now = datetime.now(timezone.utc)
        if self._db is not None:
            try:
                async with self._db.connection() as conn:
                    result = await conn.execute(
                        """
                        UPDATE security.refresh_tokens
                        SET status = 'revoked',
                            revoked_at = $2,
                            revoke_reason = 'family_revoke'
                        WHERE family_id = $1
                          AND status IN ('active', 'rotated')
                        """,
                        family_id,
                        now,
                    )
                    return _parse_command_count(result)
            except Exception as exc:
                logger.error(
                    "PG revoke_family_records failed  family=%s: %s",
                    family_id,
                    exc,
                )
                return 0

        # In-memory fallback
        count = 0
        for rec in self._memory_store.values():
            if rec.family_id == family_id and rec.status in ("active", "rotated"):
                rec.status = "revoked"
                rec.revoked_at = now
                rec.revoke_reason = "family_revoke"
                count += 1
        return count

    async def _revoke_user_records(self, user_id: str) -> int:
        """Revoke all records for a user."""
        now = datetime.now(timezone.utc)
        if self._db is not None:
            try:
                async with self._db.connection() as conn:
                    result = await conn.execute(
                        """
                        UPDATE security.refresh_tokens
                        SET status = 'revoked',
                            revoked_at = $2,
                            revoke_reason = 'user_revoke'
                        WHERE user_id = $1
                          AND status IN ('active', 'rotated')
                        """,
                        user_id,
                        now,
                    )
                    return _parse_command_count(result)
            except Exception as exc:
                logger.error(
                    "PG revoke_user_records failed  user=%s: %s",
                    user_id,
                    exc,
                )
                return 0

        # In-memory fallback
        count = 0
        for rec in self._memory_store.values():
            if rec.user_id == user_id and rec.status in ("active", "rotated"):
                rec.status = "revoked"
                rec.revoked_at = now
                rec.revoke_reason = "user_revoke"
                count += 1
        return count

    async def _get_family_size(self, family_id: str) -> int:
        """Count the number of tokens in a family."""
        if self._db is not None:
            try:
                async with self._db.connection() as conn:
                    row = await conn.fetchrow(
                        """
                        SELECT COUNT(*) AS cnt
                        FROM security.refresh_tokens
                        WHERE family_id = $1
                        """,
                        family_id,
                    )
                    return row["cnt"] if row else 0
            except Exception as exc:
                logger.error(
                    "PG get_family_size failed  family=%s: %s",
                    family_id,
                    exc,
                )
                return 0

        return sum(
            1
            for rec in self._memory_store.values()
            if rec.family_id == family_id
        )

    async def _delete_old_records(self, cutoff: datetime) -> int:
        """Delete records older than *cutoff*."""
        if self._db is not None:
            try:
                async with self._db.connection() as conn:
                    result = await conn.execute(
                        """
                        DELETE FROM security.refresh_tokens
                        WHERE expires_at < $1
                          AND status IN ('revoked', 'rotated')
                        """,
                        cutoff,
                    )
                    deleted = _parse_command_count(result)
                    logger.info(
                        "Refresh token cleanup  deleted=%d  cutoff=%s",
                        deleted,
                        cutoff.isoformat(),
                    )
                    return deleted
            except Exception as exc:
                logger.error("PG cleanup_expired failed: %s", exc)
                return 0

        # In-memory fallback
        to_remove = [
            h
            for h, rec in self._memory_store.items()
            if rec.expires_at < cutoff
            and rec.status in ("revoked", "rotated")
        ]
        for h in to_remove:
            del self._memory_store[h]
        return len(to_remove)

    # ------------------------------------------------------------------
    # Redis cache helpers
    # ------------------------------------------------------------------

    async def _cache_record(self, record: RefreshTokenRecord) -> None:
        """Cache a record in Redis for fast lookup."""
        if self._redis is None:
            return
        try:
            import json

            key = f"{_RT_REDIS_PREFIX}{record.token_hash}"
            ttl = int(
                (record.expires_at - datetime.now(timezone.utc)).total_seconds()
            )
            payload = {
                "id": record.id,
                "token_hash": record.token_hash,
                "user_id": record.user_id,
                "tenant_id": record.tenant_id,
                "family_id": record.family_id,
                "status": record.status,
                "device_fingerprint": record.device_fingerprint,
                "ip_address": record.ip_address,
                "user_agent": record.user_agent,
                "created_at": record.created_at.isoformat(),
                "expires_at": record.expires_at.isoformat(),
            }
            await self._redis.set(key, json.dumps(payload), ex=max(ttl, 1))
        except Exception as exc:
            logger.warning("Redis cache_record failed: %s", exc)

    async def _cache_lookup(
        self, token_hash: str
    ) -> Optional[RefreshTokenRecord]:
        """Attempt to read a record from Redis cache."""
        if self._redis is None:
            return None
        try:
            import json

            key = f"{_RT_REDIS_PREFIX}{token_hash}"
            raw = await self._redis.get(key)
            if raw is None:
                return None

            data = json.loads(raw)
            return RefreshTokenRecord(
                id=data["id"],
                token_hash=data["token_hash"],
                user_id=data["user_id"],
                tenant_id=data["tenant_id"],
                family_id=data["family_id"],
                status=data["status"],
                device_fingerprint=data.get("device_fingerprint"),
                ip_address=data.get("ip_address"),
                user_agent=data.get("user_agent"),
                created_at=datetime.fromisoformat(data["created_at"]),
                expires_at=datetime.fromisoformat(data["expires_at"]),
            )
        except Exception as exc:
            logger.warning("Redis cache_lookup failed: %s", exc)
            return None

    async def _invalidate_cache(self, token_hash: str) -> None:
        """Remove a record from the Redis cache."""
        if self._redis is None:
            return
        try:
            key = f"{_RT_REDIS_PREFIX}{token_hash}"
            await self._redis.delete(key)
        except Exception as exc:
            logger.warning("Redis invalidate_cache failed: %s", exc)

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    @staticmethod
    def _hash_token(plain_token: str) -> str:
        """Compute the SHA-256 hex digest of a plain token."""
        return hashlib.sha256(plain_token.encode("utf-8")).hexdigest()

    @staticmethod
    def _row_to_record(row: Any) -> RefreshTokenRecord:
        """Convert a database row (asyncpg Record) to a dataclass."""
        return RefreshTokenRecord(
            id=str(row["id"]),
            token_hash=row["token_hash"],
            user_id=row["user_id"],
            tenant_id=row["tenant_id"],
            family_id=row["family_id"],
            status=row["status"],
            device_fingerprint=row.get("device_fingerprint"),
            ip_address=row.get("ip_address"),
            user_agent=row.get("user_agent"),
            created_at=row["created_at"],
            expires_at=row["expires_at"],
            rotated_at=row.get("rotated_at"),
            revoked_at=row.get("revoked_at"),
            revoke_reason=row.get("revoke_reason"),
        )


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------


def _parse_command_count(result: Any) -> int:
    """Extract affected-row count from asyncpg command result."""
    if isinstance(result, str):
        parts = result.split()
        if len(parts) >= 2 and parts[-1].isdigit():
            return int(parts[-1])
    return 0
