# -*- coding: utf-8 -*-
"""
RBAC Cache - RBAC Authorization Service (SEC-002)

Redis-backed permission cache with pub/sub invalidation for the RBAC
authorization layer.  Provides low-latency (<1ms) permission lookups
by caching aggregated permission sets per ``(tenant_id, user_id)``
pair in Redis.

Cache key pattern::

    gl:rbac:perms:{tenant_id}:{user_id}

Invalidation is broadcast via Redis PUBLISH on the
``gl:rbac:invalidate`` channel.  All service replicas subscribe to
this channel and evict relevant local keys when an invalidation
event arrives.

**Failure handling:** Every Redis operation is wrapped in try/except.
When Redis is unavailable the cache degrades gracefully -- methods
return ``None`` on reads and silently swallow errors on writes.

Example:
    >>> cache = RBACCache(redis_client, config)
    >>> await cache.set_permissions("t-1", "u-1", ["agents:read", "agents:list"])
    >>> perms = await cache.get_permissions("t-1", "u-1")
    >>> print(perms)
    ['agents:read', 'agents:list']

Author: GreenLang Framework Team
Date: February 2026
Status: Production Ready
"""

from __future__ import annotations

import json
import logging
from typing import Any, Callable, Coroutine, Dict, List, Optional

logger = logging.getLogger(__name__)

# Default TTL for cached permission sets (seconds)
_DEFAULT_TTL = 300


class RBACCache:
    """Redis permission cache with pub/sub invalidation.

    All methods are async and safe to call when Redis is unavailable.
    On Redis failure, get operations return ``None`` and write operations
    are silently skipped with a warning log.

    Args:
        redis_client: An async Redis client (e.g. ``redis.asyncio.Redis``).
            May be ``None`` for environments without Redis (cache
            becomes a no-op).
        config: RBAC service configuration providing ``cache_ttl``,
            ``redis_key_prefix``, and ``invalidation_channel``.

    Attributes:
        _redis: The async Redis client instance.
        _prefix: Key prefix for all cache entries.
        _channel: Redis pub/sub channel for invalidation events.
        _ttl: Default TTL in seconds for cached permission sets.
    """

    def __init__(self, redis_client: Any, config: Any) -> None:
        """Initialize the RBAC cache.

        Args:
            redis_client: Async Redis client or ``None``.
            config: RBACServiceConfig dataclass instance.
        """
        self._redis = redis_client
        self._prefix = getattr(config, "redis_key_prefix", "gl:rbac")
        self._channel = getattr(config, "invalidation_channel", "gl:rbac:invalidate")
        self._ttl = getattr(config, "cache_ttl", _DEFAULT_TTL)

        logger.info(
            "RBACCache initialised  redis=%s  prefix=%s  ttl=%ds  channel=%s",
            "yes" if redis_client else "no",
            self._prefix,
            self._ttl,
            self._channel,
        )

    # ------------------------------------------------------------------
    # Key helpers
    # ------------------------------------------------------------------

    def _perms_key(self, tenant_id: str, user_id: str) -> str:
        """Build the Redis key for a user's cached permission set.

        Args:
            tenant_id: UUID of the tenant.
            user_id: UUID of the user.

        Returns:
            Redis key string.
        """
        return f"{self._prefix}:perms:{tenant_id}:{user_id}"

    def _perms_pattern(self, tenant_id: Optional[str] = None) -> str:
        """Build a SCAN pattern for permission cache keys.

        Args:
            tenant_id: If provided, scope pattern to a single tenant.
                If ``None``, matches all tenants.

        Returns:
            Redis SCAN pattern string.
        """
        if tenant_id:
            return f"{self._prefix}:perms:{tenant_id}:*"
        return f"{self._prefix}:perms:*"

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    async def get_permissions(
        self,
        tenant_id: str,
        user_id: str,
    ) -> Optional[List[str]]:
        """Get cached permission strings for a user.

        Args:
            tenant_id: UUID of the tenant.
            user_id: UUID of the user.

        Returns:
            List of permission strings if cached, ``None`` on cache
            miss or Redis failure.
        """
        if self._redis is None:
            return None

        key = self._perms_key(tenant_id, user_id)
        try:
            raw = await self._redis.get(key)
            if raw is None:
                return None
            permissions: List[str] = json.loads(raw)
            logger.debug("Cache HIT  key=%s  count=%d", key, len(permissions))
            return permissions
        except Exception as exc:
            logger.warning("RBACCache.get_permissions failed: %s", exc)
            return None

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    async def set_permissions(
        self,
        tenant_id: str,
        user_id: str,
        permissions: List[str],
        ttl: Optional[int] = None,
    ) -> bool:
        """Cache a user's aggregated permission set.

        Args:
            tenant_id: UUID of the tenant.
            user_id: UUID of the user.
            permissions: List of permission strings to cache.
            ttl: Override TTL in seconds (uses default if ``None``).

        Returns:
            ``True`` if the value was stored, ``False`` on failure.
        """
        if self._redis is None:
            return False

        key = self._perms_key(tenant_id, user_id)
        effective_ttl = ttl if ttl is not None else self._ttl
        try:
            payload = json.dumps(permissions)
            await self._redis.set(key, payload, ex=effective_ttl)
            logger.debug(
                "Cache SET  key=%s  count=%d  ttl=%ds",
                key,
                len(permissions),
                effective_ttl,
            )
            return True
        except Exception as exc:
            logger.warning("RBACCache.set_permissions failed: %s", exc)
            return False

    # ------------------------------------------------------------------
    # Invalidation
    # ------------------------------------------------------------------

    async def invalidate_user(self, tenant_id: str, user_id: str) -> bool:
        """Invalidate the cached permissions for a single user.

        Args:
            tenant_id: UUID of the tenant.
            user_id: UUID of the user.

        Returns:
            ``True`` if the key was deleted, ``False`` on failure or miss.
        """
        if self._redis is None:
            return False

        key = self._perms_key(tenant_id, user_id)
        try:
            deleted = await self._redis.delete(key)
            logger.info("Cache invalidate user  key=%s  deleted=%d", key, deleted)
            return deleted > 0
        except Exception as exc:
            logger.warning("RBACCache.invalidate_user failed: %s", exc)
            return False

    async def invalidate_tenant(self, tenant_id: str) -> int:
        """Invalidate all cached permissions for a tenant.

        Uses SCAN to find matching keys to avoid blocking Redis with
        a KEYS command.

        Args:
            tenant_id: UUID of the tenant.

        Returns:
            Number of keys deleted.
        """
        if self._redis is None:
            return 0

        pattern = self._perms_pattern(tenant_id)
        return await self._scan_and_delete(pattern)

    async def invalidate_all(self) -> int:
        """Invalidate all RBAC permission cache entries.

        Uses SCAN to find and delete all ``gl:rbac:perms:*`` keys.

        Returns:
            Number of keys deleted.
        """
        if self._redis is None:
            return 0

        pattern = self._perms_pattern()
        return await self._scan_and_delete(pattern)

    async def _scan_and_delete(self, pattern: str) -> int:
        """SCAN for keys matching pattern and delete them in batches.

        Args:
            pattern: Redis SCAN pattern.

        Returns:
            Total number of keys deleted.
        """
        deleted_total = 0
        try:
            cursor: Any = "0"
            while True:
                cursor, keys = await self._redis.scan(
                    cursor=cursor, match=pattern, count=200
                )
                if keys:
                    deleted = await self._redis.delete(*keys)
                    deleted_total += deleted

                # cursor is 0 or b"0" when scan is complete
                if cursor == 0 or cursor == b"0":
                    break

            logger.info(
                "Cache bulk invalidate  pattern=%s  deleted=%d",
                pattern,
                deleted_total,
            )
        except Exception as exc:
            logger.warning("RBACCache._scan_and_delete failed: %s", exc)

        return deleted_total

    # ------------------------------------------------------------------
    # Pub/Sub: publish
    # ------------------------------------------------------------------

    async def publish_invalidation(
        self,
        event_type: str,
        tenant_id: str,
        user_id: Optional[str] = None,
    ) -> bool:
        """Publish an invalidation event to the Redis pub/sub channel.

        All service replicas subscribing to the channel will receive
        the event and can evict their local caches accordingly.

        The message payload is a JSON object::

            {
                "event_type": "user_role_changed",
                "tenant_id": "...",
                "user_id": "..."   // optional
            }

        Args:
            event_type: Invalidation event type (e.g.
                ``"user_role_changed"``, ``"role_updated"``,
                ``"tenant_purge"``).
            tenant_id: UUID of the affected tenant.
            user_id: UUID of the affected user (optional).

        Returns:
            ``True`` if the message was published, ``False`` on failure.
        """
        if self._redis is None:
            return False

        message: Dict[str, Any] = {
            "event_type": event_type,
            "tenant_id": tenant_id,
        }
        if user_id:
            message["user_id"] = user_id

        try:
            payload = json.dumps(message)
            await self._redis.publish(self._channel, payload)
            logger.debug(
                "Published invalidation  channel=%s  payload=%s",
                self._channel,
                payload,
            )
            return True
        except Exception as exc:
            logger.warning("RBACCache.publish_invalidation failed: %s", exc)
            return False

    # ------------------------------------------------------------------
    # Pub/Sub: subscribe
    # ------------------------------------------------------------------

    async def subscribe_invalidations(
        self,
        callback: Callable[[Dict[str, Any]], Coroutine[Any, Any, None]],
    ) -> None:
        """Subscribe to the invalidation channel and dispatch events.

        This is a long-running coroutine that listens indefinitely.
        It should be started as a background task.

        The callback receives the parsed JSON message as a dictionary
        with keys ``event_type``, ``tenant_id``, and optionally
        ``user_id``.

        Args:
            callback: Async function to call for each invalidation event.

        Raises:
            RuntimeError: If Redis client is not available.
        """
        if self._redis is None:
            raise RuntimeError("Cannot subscribe: Redis client is not available")

        try:
            pubsub = self._redis.pubsub()
            await pubsub.subscribe(self._channel)
            logger.info(
                "Subscribed to RBAC invalidation channel: %s", self._channel
            )

            async for raw_message in pubsub.listen():
                if raw_message["type"] != "message":
                    continue

                try:
                    data = json.loads(raw_message["data"])
                    await callback(data)
                except json.JSONDecodeError:
                    logger.warning(
                        "Invalid JSON on invalidation channel: %s",
                        raw_message["data"],
                    )
                except Exception as exc:
                    logger.error(
                        "Invalidation callback error: %s", exc, exc_info=True
                    )
        except Exception as exc:
            logger.error(
                "RBACCache.subscribe_invalidations failed: %s",
                exc,
                exc_info=True,
            )
