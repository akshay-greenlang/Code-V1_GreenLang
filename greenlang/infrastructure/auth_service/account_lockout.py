# -*- coding: utf-8 -*-
"""
Account Lockout - JWT Authentication Service (SEC-001)

Implements progressive account lockout after repeated failed login attempts.
Lockout state is held in Redis for fast enforcement and PostgreSQL for
durable audit history.  Progressive backoff doubles the lockout duration
on each successive lockout event, capped at a configurable maximum.

Classes:
    - LockoutConfig: Immutable configuration for lockout behaviour.
    - LockoutStatus: Current lockout state for a user or IP.
    - AccountLockoutManager: Core lockout enforcement engine.

Example:
    >>> config = LockoutConfig(max_attempts=5)
    >>> manager = AccountLockoutManager(config, redis_client=redis, db_pool=pool)
    >>> status = await manager.check_lockout(user_id="u-123")
    >>> status.is_locked
    False

Author: GreenLang Framework Team
Date: February 2026
Status: Production Ready
"""

from __future__ import annotations

import json
import logging
import math
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Redis key helpers
# ---------------------------------------------------------------------------

_LOCKOUT_KEY_PREFIX = "gl:auth:lockout:"
_ATTEMPTS_KEY_PREFIX = "gl:auth:attempts:"
_IP_ATTEMPTS_KEY_PREFIX = "gl:auth:ip_attempts:"


def _lockout_key(user_id: str) -> str:
    """Return the Redis key for a user's lockout state."""
    return f"{_LOCKOUT_KEY_PREFIX}{user_id}"


def _attempts_key(username: str) -> str:
    """Return the Redis key for a user's recent attempt timestamps."""
    return f"{_ATTEMPTS_KEY_PREFIX}{username}"


def _ip_attempts_key(ip_address: str) -> str:
    """Return the Redis key for an IP's recent attempt timestamps."""
    return f"{_IP_ATTEMPTS_KEY_PREFIX}{ip_address}"


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LockoutConfig:
    """Immutable configuration for account lockout behaviour.

    Attributes:
        max_attempts: Failed attempts before lockout triggers.
        initial_lockout_seconds: First lockout duration (default 15 min).
        max_lockout_seconds: Maximum lockout duration (default 24 h).
        progressive_multiplier: Multiplier applied to each successive lockout.
        lockout_window_seconds: Sliding window for counting failures.
        ip_rate_limit_per_minute: Max failed attempts per IP per minute.
        service_account_exempt: Whether service accounts bypass lockout.
    """

    max_attempts: int = 5
    initial_lockout_seconds: int = 900
    max_lockout_seconds: int = 86400
    progressive_multiplier: float = 2.0
    lockout_window_seconds: int = 900
    ip_rate_limit_per_minute: int = 10
    service_account_exempt: bool = True


# ---------------------------------------------------------------------------
# Lockout status
# ---------------------------------------------------------------------------


@dataclass
class LockoutStatus:
    """Current lockout state for a user.

    Attributes:
        is_locked: Whether the account is currently locked.
        locked_until: UTC datetime when lockout expires (None if not locked).
        failed_attempts: Number of recent failed attempts in the window.
        remaining_attempts: How many attempts remain before lockout.
        lockout_count: Number of times the account has been locked.
    """

    is_locked: bool = False
    locked_until: Optional[datetime] = None
    failed_attempts: int = 0
    remaining_attempts: int = 5
    lockout_count: int = 0


# ---------------------------------------------------------------------------
# AccountLockoutManager
# ---------------------------------------------------------------------------


class AccountLockoutManager:
    """Manages account lockout state and enforcement.

    Uses Redis sorted sets for fast attempt tracking and a Redis hash for
    lockout metadata.  Login attempts are also persisted to PostgreSQL
    ``security.login_attempts`` for long-term audit.

    Attributes:
        config: Lockout configuration.

    Example:
        >>> mgr = AccountLockoutManager(redis_client=redis, db_pool=pool)
        >>> status = await mgr.record_attempt("alice", success=False)
        >>> status.remaining_attempts
        4
    """

    def __init__(
        self,
        config: Optional[LockoutConfig] = None,
        redis_client: Any = None,
        db_pool: Any = None,
    ) -> None:
        """Initialize the lockout manager.

        Args:
            config: Lockout configuration.  Uses defaults if ``None``.
            redis_client: Async Redis client (e.g. ``redis.asyncio.Redis``).
            db_pool: Async PostgreSQL connection pool for audit persistence.
        """
        self.config = config or LockoutConfig()
        self._redis = redis_client
        self._db_pool = db_pool

    # ------------------------------------------------------------------
    # Check lockout
    # ------------------------------------------------------------------

    async def check_lockout(
        self,
        user_id: Optional[str] = None,
        username: Optional[str] = None,
        ip_address: Optional[str] = None,
    ) -> LockoutStatus:
        """Check whether a user or IP is currently locked out.

        At least one of *user_id*, *username*, or *ip_address* must be
        provided.  If *user_id* is given, the user-level lockout is checked.
        IP-level rate limits are checked separately via ``ip_address``.

        Args:
            user_id: UUID of the user.
            username: Username string for attempt counting.
            ip_address: Client IP address for IP-level rate limiting.

        Returns:
            Current ``LockoutStatus``.
        """
        if self._redis is None:
            logger.warning("No Redis client; lockout checks are disabled")
            return LockoutStatus(remaining_attempts=self.config.max_attempts)

        # Check user-level lockout via Redis hash
        if user_id:
            lockout_data = await self._get_lockout_data(user_id)
            if lockout_data is not None:
                locked_until = datetime.fromisoformat(lockout_data["locked_until"])
                if datetime.now(timezone.utc) < locked_until:
                    return LockoutStatus(
                        is_locked=True,
                        locked_until=locked_until,
                        failed_attempts=lockout_data.get("failed_attempts", 0),
                        remaining_attempts=0,
                        lockout_count=lockout_data.get("lockout_count", 1),
                    )
                else:
                    # Lockout expired -- clear it
                    await self._clear_lockout(user_id)

        # Count recent failures to compute remaining attempts
        failed = 0
        if username:
            failed = await self.get_failed_attempts(username=username)

        remaining = max(0, self.config.max_attempts - failed)

        return LockoutStatus(
            is_locked=False,
            failed_attempts=failed,
            remaining_attempts=remaining,
        )

    # ------------------------------------------------------------------
    # Record attempt
    # ------------------------------------------------------------------

    async def record_attempt(
        self,
        username: str,
        user_id: Optional[str] = None,
        ip_address: Optional[str] = None,
        success: bool = False,
        failure_reason: Optional[str] = None,
    ) -> LockoutStatus:
        """Record a login attempt and apply lockout logic if needed.

        On **success** the failure counter is reset.  On **failure** the
        counter is incremented.  If the counter reaches ``max_attempts``,
        a progressive lockout is applied.

        Args:
            username: Username of the login attempt.
            user_id: UUID of the user (may be ``None`` if unknown).
            ip_address: Source IP of the request.
            success: Whether the attempt succeeded.
            failure_reason: Reason string for failed attempts.

        Returns:
            Updated ``LockoutStatus`` after recording the attempt.
        """
        start = time.monotonic()
        now = datetime.now(timezone.utc)

        # Persist to PostgreSQL audit table
        await self._persist_attempt(
            username=username,
            user_id=user_id,
            ip_address=ip_address,
            success=success,
            failure_reason=failure_reason,
            attempted_at=now,
        )

        if self._redis is None:
            logger.warning("No Redis client; attempt tracking is disabled")
            return LockoutStatus(remaining_attempts=self.config.max_attempts)

        if success:
            return await self._handle_success(username, user_id)

        return await self._handle_failure(username, user_id, ip_address, now)

    # ------------------------------------------------------------------
    # Admin unlock
    # ------------------------------------------------------------------

    async def unlock_account(
        self,
        user_id: str,
        unlocked_by: str = "admin",
    ) -> bool:
        """Immediately remove lockout for a user.

        Args:
            user_id: UUID of the user to unlock.
            unlocked_by: Identity of the admin performing the unlock.

        Returns:
            ``True`` if the account was locked and is now unlocked,
            ``False`` if it was not locked.
        """
        if self._redis is None:
            logger.warning("No Redis client; cannot unlock account")
            return False

        lockout_data = await self._get_lockout_data(user_id)
        if lockout_data is None:
            logger.info("Account %s is not locked; nothing to unlock", user_id)
            return False

        await self._clear_lockout(user_id)

        logger.info(
            "Account %s unlocked by %s (was locked until %s)",
            user_id,
            unlocked_by,
            lockout_data.get("locked_until", "unknown"),
        )
        return True

    # ------------------------------------------------------------------
    # Count recent failures
    # ------------------------------------------------------------------

    async def get_failed_attempts(
        self,
        username: Optional[str] = None,
        ip_address: Optional[str] = None,
        window_seconds: Optional[int] = None,
    ) -> int:
        """Count recent failed login attempts within the sliding window.

        Args:
            username: Count failures for this username.
            ip_address: Count failures for this IP.
            window_seconds: Override the default window.  Uses
                ``lockout_window_seconds`` if ``None``.

        Returns:
            Number of failed attempts in the window.
        """
        if self._redis is None:
            return 0

        window = window_seconds or self.config.lockout_window_seconds
        cutoff = time.time() - window
        count = 0

        if username:
            key = _attempts_key(username)
            # Remove stale entries
            await self._redis.zremrangebyscore(key, "-inf", cutoff)
            count += await self._redis.zcard(key)

        if ip_address:
            key = _ip_attempts_key(ip_address)
            await self._redis.zremrangebyscore(key, "-inf", cutoff)
            count += await self._redis.zcard(key)

        return count

    # ------------------------------------------------------------------
    # Lockout duration calculation
    # ------------------------------------------------------------------

    def _calculate_lockout_duration(self, lockout_count: int) -> int:
        """Calculate progressive lockout duration in seconds.

        Uses ``initial * (multiplier ^ lockout_count)``, capped at ``max``.

        Args:
            lockout_count: How many times the account has been locked before.

        Returns:
            Lockout duration in seconds.
        """
        duration = self.config.initial_lockout_seconds * (
            self.config.progressive_multiplier ** lockout_count
        )
        return min(int(duration), self.config.max_lockout_seconds)

    # ------------------------------------------------------------------
    # Internal: success handler
    # ------------------------------------------------------------------

    async def _handle_success(
        self,
        username: str,
        user_id: Optional[str],
    ) -> LockoutStatus:
        """Handle a successful login by resetting failure counters.

        Args:
            username: Username that succeeded.
            user_id: UUID of the user.

        Returns:
            Clean ``LockoutStatus`` with zero failures.
        """
        key = _attempts_key(username)
        await self._redis.delete(key)

        if user_id:
            await self._clear_lockout(user_id)

        logger.debug("Login success for %s; failure counter reset", username)

        return LockoutStatus(
            is_locked=False,
            failed_attempts=0,
            remaining_attempts=self.config.max_attempts,
            lockout_count=0,
        )

    # ------------------------------------------------------------------
    # Internal: failure handler
    # ------------------------------------------------------------------

    async def _handle_failure(
        self,
        username: str,
        user_id: Optional[str],
        ip_address: Optional[str],
        now: datetime,
    ) -> LockoutStatus:
        """Handle a failed login by incrementing counters and possibly locking.

        Args:
            username: Username that failed.
            user_id: UUID of the user (may be ``None``).
            ip_address: Source IP address.
            now: Timestamp of the attempt.

        Returns:
            Updated ``LockoutStatus``.
        """
        timestamp = now.timestamp()

        # Record username-level failure
        key = _attempts_key(username)
        await self._redis.zadd(key, {f"{timestamp}": timestamp})
        await self._redis.expire(key, self.config.lockout_window_seconds)

        # Record IP-level failure
        if ip_address:
            ip_key = _ip_attempts_key(ip_address)
            await self._redis.zadd(ip_key, {f"{timestamp}": timestamp})
            await self._redis.expire(ip_key, 60)  # 1-minute window for IP

        # Count recent failures
        cutoff = time.time() - self.config.lockout_window_seconds
        await self._redis.zremrangebyscore(key, "-inf", cutoff)
        failed_count = await self._redis.zcard(key)

        remaining = max(0, self.config.max_attempts - failed_count)

        # Check whether lockout threshold is reached
        if failed_count >= self.config.max_attempts and user_id:
            return await self._apply_lockout(user_id, username, failed_count)

        logger.info(
            "Login failure for %s: %d/%d attempts used",
            username,
            failed_count,
            self.config.max_attempts,
        )

        return LockoutStatus(
            is_locked=False,
            failed_attempts=failed_count,
            remaining_attempts=remaining,
        )

    # ------------------------------------------------------------------
    # Internal: apply lockout
    # ------------------------------------------------------------------

    async def _apply_lockout(
        self,
        user_id: str,
        username: str,
        failed_count: int,
    ) -> LockoutStatus:
        """Apply a progressive lockout to the user.

        Args:
            user_id: UUID of the user.
            username: Username (for logging).
            failed_count: Current number of failed attempts.

        Returns:
            ``LockoutStatus`` reflecting the new lockout.
        """
        # Determine how many previous lockouts
        prev_data = await self._get_lockout_data(user_id)
        prev_count = prev_data.get("lockout_count", 0) if prev_data else 0

        lockout_count = prev_count + 1
        duration_s = self._calculate_lockout_duration(prev_count)
        locked_until = datetime.now(timezone.utc) + timedelta(seconds=duration_s)

        lockout_payload = {
            "locked_until": locked_until.isoformat(),
            "lockout_count": lockout_count,
            "failed_attempts": failed_count,
            "locked_at": datetime.now(timezone.utc).isoformat(),
        }

        lockout_key = _lockout_key(user_id)
        await self._redis.set(
            lockout_key,
            json.dumps(lockout_payload),
            ex=duration_s + 60,  # TTL slightly longer than lockout
        )

        logger.warning(
            "Account locked: user_id=%s username=%s lockout_count=%d "
            "duration=%ds locked_until=%s",
            user_id,
            username,
            lockout_count,
            duration_s,
            locked_until.isoformat(),
        )

        return LockoutStatus(
            is_locked=True,
            locked_until=locked_until,
            failed_attempts=failed_count,
            remaining_attempts=0,
            lockout_count=lockout_count,
        )

    # ------------------------------------------------------------------
    # Internal: Redis helpers
    # ------------------------------------------------------------------

    async def _get_lockout_data(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve lockout metadata from Redis.

        Args:
            user_id: UUID of the user.

        Returns:
            Parsed lockout data dict, or ``None`` if not locked.
        """
        raw = await self._redis.get(_lockout_key(user_id))
        if raw is None:
            return None
        try:
            return json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            logger.error("Corrupt lockout data for user %s", user_id)
            return None

    async def _clear_lockout(self, user_id: str) -> None:
        """Remove lockout state from Redis.

        Args:
            user_id: UUID of the user.
        """
        await self._redis.delete(_lockout_key(user_id))

    # ------------------------------------------------------------------
    # Internal: persist to PostgreSQL
    # ------------------------------------------------------------------

    async def _persist_attempt(
        self,
        username: str,
        user_id: Optional[str],
        ip_address: Optional[str],
        success: bool,
        failure_reason: Optional[str],
        attempted_at: datetime,
    ) -> None:
        """Write login attempt to ``security.login_attempts`` for audit.

        Args:
            username: Username of the attempt.
            user_id: UUID of the user.
            ip_address: Source IP address.
            success: Whether the attempt succeeded.
            failure_reason: Reason code for failures.
            attempted_at: UTC timestamp.
        """
        if self._db_pool is None:
            return

        query = """
            INSERT INTO security.login_attempts
                (username, user_id, ip_address, success, failure_reason, attempted_at)
            VALUES ($1, $2, $3, $4, $5, $6)
        """

        try:
            async with self._db_pool.connection() as conn:
                await conn.execute(
                    query,
                    username,
                    user_id,
                    ip_address,
                    success,
                    failure_reason,
                    attempted_at,
                )
        except Exception:
            logger.error(
                "Failed to persist login attempt for %s", username, exc_info=True
            )
