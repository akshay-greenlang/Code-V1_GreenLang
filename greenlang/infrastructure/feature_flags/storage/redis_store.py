# -*- coding: utf-8 -*-
"""
Redis Feature Flag Storage - INFRA-008

Async Redis-backed implementation of ``IFlagStorage`` that serves as the
L2 cache layer in the multi-layer storage architecture. Designed for
sub-millisecond reads in the flag evaluation hot path.

Features:
    - Async Redis via ``redis.asyncio`` with ConnectionPool
    - Structured key patterns: ``ff:{env}:flag:{key}``, ``ff:{env}:rules:{key}``, etc.
    - Configurable TTL on all keys (default 300s)
    - Circuit breaker: opens after 5 consecutive failures, recovers after 60s
    - Pub/sub support on channel ``ff:updates`` for cache invalidation
    - Graceful degradation: returns None on errors instead of raising
    - Health check endpoint

Follows the Redis patterns established in ``greenlang.utilities.cache.redis_client``:
    - ``socket_timeout=5.0``
    - ``socket_keepalive=True``
    - ``decode_responses=True``
    - ``retry_on_timeout=True``

Example:
    >>> from greenlang.infrastructure.feature_flags.config import FeatureFlagConfig
    >>> config = FeatureFlagConfig(redis_url="redis://localhost:6379/2")
    >>> store = RedisFlagStorage(config)
    >>> await store.initialize()
    >>> await store.save_flag(flag)
    >>> result = await store.get_flag("enable-scope3-calc")
    >>> await store.close()
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

from greenlang.infrastructure.feature_flags.config import FeatureFlagConfig
from greenlang.infrastructure.feature_flags.models import (
    AuditLogEntry,
    FeatureFlag,
    FlagOverride,
    FlagRule,
    FlagStatus,
    FlagVariant,
)
from greenlang.infrastructure.feature_flags.storage.base import IFlagStorage

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Circuit breaker (async version following greenlang.db.connection pattern)
# ---------------------------------------------------------------------------


class _CircuitState(str, Enum):
    """Circuit breaker states."""

    CLOSED = "closed"  # healthy -- requests flow normally
    OPEN = "open"  # unhealthy -- requests are rejected
    HALF_OPEN = "half_open"  # recovery -- limited probes allowed


class _AsyncCircuitBreaker:
    """Async-safe circuit breaker for Redis operations.

    Opens after ``failure_threshold`` consecutive failures, rejects
    calls for ``recovery_timeout`` seconds, then transitions to
    half-open to probe for recovery.

    Args:
        failure_threshold: Failures before the circuit opens.
        recovery_timeout: Seconds to wait before probing recovery.
        half_open_max_calls: Probes allowed in half-open state.
    """

    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: float = 60.0,
        half_open_max_calls: int = 3,
    ) -> None:
        self._failure_threshold = failure_threshold
        self._recovery_timeout = recovery_timeout
        self._half_open_max_calls = half_open_max_calls

        self._state = _CircuitState.CLOSED
        self._failure_count = 0
        self._last_failure_time = 0.0
        self._half_open_calls = 0
        self._lock = asyncio.Lock()

    @property
    def state(self) -> _CircuitState:
        """Current circuit state (non-locking read for metrics)."""
        return self._state

    async def record_success(self) -> None:
        """Record a successful operation."""
        async with self._lock:
            if self._state == _CircuitState.HALF_OPEN:
                self._state = _CircuitState.CLOSED
                self._failure_count = 0
                self._half_open_calls = 0
                logger.info("Redis circuit breaker: recovered (CLOSED)")
            elif self._state == _CircuitState.CLOSED:
                self._failure_count = 0

    async def record_failure(self) -> None:
        """Record a failed operation."""
        async with self._lock:
            self._failure_count += 1
            self._last_failure_time = time.monotonic()
            if self._failure_count >= self._failure_threshold:
                if self._state != _CircuitState.OPEN:
                    self._state = _CircuitState.OPEN
                    logger.error(
                        "Redis circuit breaker: OPEN after %d failures",
                        self._failure_count,
                    )

    async def can_attempt(self) -> bool:
        """Check whether an operation is permitted."""
        async with self._lock:
            if self._state == _CircuitState.CLOSED:
                return True

            if self._state == _CircuitState.OPEN:
                elapsed = time.monotonic() - self._last_failure_time
                if elapsed >= self._recovery_timeout:
                    self._state = _CircuitState.HALF_OPEN
                    self._half_open_calls = 0
                    logger.info("Redis circuit breaker: HALF_OPEN (probing)")
                    return True
                return False

            if self._state == _CircuitState.HALF_OPEN:
                if self._half_open_calls < self._half_open_max_calls:
                    self._half_open_calls += 1
                    return True
                return False

            return False


# ---------------------------------------------------------------------------
# RedisFlagStorage
# ---------------------------------------------------------------------------


class RedisFlagStorage(IFlagStorage):
    """Async Redis storage backend for feature flags (L2 cache layer).

    All keys use the pattern ``ff:{environment}:{entity}:{key}`` and
    expire after a configurable TTL.  A circuit breaker protects the
    application from Redis outages.

    Args:
        config: FeatureFlagConfig instance.
        on_invalidation: Optional callback invoked when a pub/sub
            invalidation message is received. Signature: ``(flag_key: str) -> None``.
    """

    def __init__(
        self,
        config: FeatureFlagConfig,
        on_invalidation: Optional[Callable[[str], Any]] = None,
    ) -> None:
        self._config = config
        self._env = config.environment
        self._ttl = config.redis_cache_ttl_seconds or 300
        self._channel = config.pubsub_channel
        self._on_invalidation = on_invalidation

        # Lazy imports so the module can be imported without redis installed
        self._redis_mod: Any = None
        self._pool: Any = None
        self._client: Any = None
        self._pubsub: Any = None
        self._listener_task: Optional[asyncio.Task] = None
        self._initialized = False

        self._circuit = _AsyncCircuitBreaker(
            failure_threshold=config.circuit_breaker_failure_threshold,
            recovery_timeout=config.circuit_breaker_recovery_timeout,
            half_open_max_calls=config.circuit_breaker_half_open_max_calls,
        )

        logger.info(
            "RedisFlagStorage created (env=%s, ttl=%ds, channel=%s)",
            self._env,
            self._ttl,
            self._channel,
        )

    # ------------------------------------------------------------------
    # Key helpers
    # ------------------------------------------------------------------

    def _key(self, entity: str, key: str) -> str:
        """Build a namespaced Redis key."""
        return f"ff:{self._env}:{entity}:{key}"

    def _flag_key(self, key: str) -> str:
        return self._key("flag", key)

    def _rules_key(self, flag_key: str) -> str:
        return self._key("rules", flag_key)

    def _overrides_key(self, flag_key: str) -> str:
        return self._key("overrides", flag_key)

    def _variants_key(self, flag_key: str) -> str:
        return self._key("variants", flag_key)

    def _audit_key(self, flag_key: str) -> str:
        return self._key("audit", flag_key)

    def _all_flags_key(self) -> str:
        return self._key("index", "all_flags")

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    @staticmethod
    def _serialize(model: Any) -> str:
        """Serialize a Pydantic model (or list of models) to JSON string."""
        if isinstance(model, list):
            return json.dumps([m.model_dump(mode="json") for m in model])
        return model.model_dump_json()

    @staticmethod
    def _deserialize_flag(data: str) -> FeatureFlag:
        return FeatureFlag.model_validate_json(data)

    @staticmethod
    def _deserialize_flags(data: str) -> List[FeatureFlag]:
        return [FeatureFlag.model_validate(d) for d in json.loads(data)]

    @staticmethod
    def _deserialize_rules(data: str) -> List[FlagRule]:
        return [FlagRule.model_validate(d) for d in json.loads(data)]

    @staticmethod
    def _deserialize_overrides(data: str) -> List[FlagOverride]:
        return [FlagOverride.model_validate(d) for d in json.loads(data)]

    @staticmethod
    def _deserialize_variants(data: str) -> List[FlagVariant]:
        return [FlagVariant.model_validate(d) for d in json.loads(data)]

    @staticmethod
    def _deserialize_audit(data: str) -> List[AuditLogEntry]:
        return [AuditLogEntry.model_validate(d) for d in json.loads(data)]

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def initialize(self) -> None:
        """Create Redis connection pool and start the pub/sub listener."""
        if self._initialized:
            return

        try:
            import redis.asyncio as aioredis
            self._redis_mod = aioredis
        except ImportError:
            logger.error(
                "redis.asyncio not available. "
                "Install with: pip install redis>=4.5.0"
            )
            return

        try:
            self._pool = aioredis.ConnectionPool.from_url(
                self._config.redis_url or "redis://localhost:6379/0",
                max_connections=self._config.redis_max_connections,
                socket_timeout=self._config.redis_socket_timeout,
                socket_keepalive=self._config.redis_socket_keepalive,
                decode_responses=self._config.redis_decode_responses,
                retry_on_timeout=self._config.redis_retry_on_timeout,
            )
            self._client = aioredis.Redis(connection_pool=self._pool)

            # Verify connectivity
            await self._client.ping()
            self._initialized = True

            # Start pub/sub listener
            self._start_pubsub_listener()

            logger.info(
                "RedisFlagStorage initialised (url=%s)",
                self._config.redis_url,
            )
        except Exception as exc:
            logger.error("Failed to initialise Redis connection: %s", exc)
            await self._circuit.record_failure()

    def _start_pubsub_listener(self) -> None:
        """Spawn a background task that listens for invalidation messages."""
        if self._listener_task is not None:
            return
        self._listener_task = asyncio.create_task(
            self._pubsub_listen(), name="redis-ff-pubsub"
        )

    async def _pubsub_listen(self) -> None:
        """Background coroutine: subscribe to the invalidation channel."""
        if self._client is None:
            return
        try:
            self._pubsub = self._client.pubsub()
            await self._pubsub.subscribe(self._channel)
            logger.info("Subscribed to Redis pub/sub channel '%s'", self._channel)

            async for message in self._pubsub.listen():
                if message["type"] != "message":
                    continue
                flag_key = message.get("data", "")
                if flag_key and self._on_invalidation:
                    try:
                        result = self._on_invalidation(flag_key)
                        if asyncio.iscoroutine(result):
                            await result
                    except Exception as cb_exc:
                        logger.warning(
                            "Invalidation callback error for '%s': %s",
                            flag_key, cb_exc,
                        )
        except asyncio.CancelledError:
            logger.info("Pub/sub listener cancelled")
        except Exception as exc:
            logger.error("Pub/sub listener error: %s", exc)

    async def _publish_invalidation(self, flag_key: str) -> None:
        """Publish a cache invalidation message for a flag key."""
        if self._client is None:
            return
        try:
            await self._client.publish(self._channel, flag_key)
            logger.debug("Published invalidation for '%s'", flag_key)
        except Exception as exc:
            logger.warning("Failed to publish invalidation for '%s': %s", flag_key, exc)

    async def close(self) -> None:
        """Close Redis connections and stop the pub/sub listener."""
        if self._listener_task is not None:
            self._listener_task.cancel()
            try:
                await self._listener_task
            except asyncio.CancelledError:
                pass
            self._listener_task = None

        if self._pubsub is not None:
            try:
                await self._pubsub.unsubscribe(self._channel)
                await self._pubsub.close()
            except Exception:
                pass
            self._pubsub = None

        if self._client is not None:
            try:
                await self._client.close()
            except Exception:
                pass
            self._client = None

        if self._pool is not None:
            try:
                await self._pool.disconnect()
            except Exception:
                pass
            self._pool = None

        self._initialized = False
        logger.info("RedisFlagStorage closed")

    # Keep backward-compatible alias
    async def shutdown(self) -> None:
        """Alias for ``close()``."""
        await self.close()

    # ------------------------------------------------------------------
    # Safe Redis helpers (circuit breaker + graceful degradation)
    # ------------------------------------------------------------------

    async def _safe_get(self, key: str) -> Optional[str]:
        """GET with circuit breaker. Returns None on any failure."""
        if not self._initialized or self._client is None:
            return None
        if not await self._circuit.can_attempt():
            return None
        try:
            value = await self._client.get(key)
            await self._circuit.record_success()
            return value
        except Exception as exc:
            await self._circuit.record_failure()
            logger.warning("Redis GET failed for '%s': %s", key, exc)
            return None

    async def _safe_set(
        self, key: str, value: str, ttl: Optional[int] = None
    ) -> bool:
        """SETEX with circuit breaker. Returns False on any failure."""
        if not self._initialized or self._client is None:
            return False
        if not await self._circuit.can_attempt():
            return False
        try:
            await self._client.setex(key, ttl or self._ttl, value)
            await self._circuit.record_success()
            return True
        except Exception as exc:
            await self._circuit.record_failure()
            logger.warning("Redis SET failed for '%s': %s", key, exc)
            return False

    async def _safe_delete(self, *keys: str) -> int:
        """DEL with circuit breaker. Returns 0 on any failure."""
        if not self._initialized or self._client is None:
            return 0
        if not await self._circuit.can_attempt():
            return 0
        try:
            result = await self._client.delete(*keys)
            await self._circuit.record_success()
            return result
        except Exception as exc:
            await self._circuit.record_failure()
            logger.warning("Redis DEL failed: %s", exc)
            return 0

    # ------------------------------------------------------------------
    # IFlagStorage -- Flags
    # ------------------------------------------------------------------

    async def get_flag(self, key: str) -> Optional[FeatureFlag]:
        """Retrieve a flag from Redis by key."""
        data = await self._safe_get(self._flag_key(key))
        if data is None:
            return None
        try:
            return self._deserialize_flag(data)
        except Exception as exc:
            logger.warning("Failed to deserialize flag '%s': %s", key, exc)
            return None

    async def get_all_flags(
        self,
        status_filter: Optional[FlagStatus] = None,
        tag_filter: Optional[str] = None,
    ) -> List[FeatureFlag]:
        """Retrieve all flags from the Redis index.

        If the index key is missing the result will be an empty list.
        Filtering is applied in-memory after deserialization.
        """
        data = await self._safe_get(self._all_flags_key())
        if data is None:
            return []
        try:
            flags = self._deserialize_flags(data)
        except Exception as exc:
            logger.warning("Failed to deserialize all-flags index: %s", exc)
            return []

        if status_filter is not None:
            flags = [f for f in flags if f.status == status_filter]
        if tag_filter is not None:
            tag_lower = tag_filter.lower()
            flags = [f for f in flags if tag_lower in f.tags]
        return flags

    async def save_flag(self, flag: FeatureFlag) -> FeatureFlag:
        """Persist a flag to Redis and update the all-flags index."""
        await self._safe_set(self._flag_key(flag.key), self._serialize(flag))
        # Update the all-flags index
        await self._rebuild_flag_index(flag)
        await self._publish_invalidation(flag.key)
        return flag

    async def _rebuild_flag_index(self, changed_flag: FeatureFlag) -> None:
        """Update the all-flags index with the changed flag.

        Performs a read-modify-write on the index. This is acceptable
        for L2 cache because exact consistency is not required (L3 is
        the source of truth).
        """
        existing = await self.get_all_flags()
        updated = [f for f in existing if f.key != changed_flag.key]
        updated.append(changed_flag)
        await self._safe_set(
            self._all_flags_key(),
            self._serialize(updated),
        )

    async def delete_flag(self, key: str) -> bool:
        """Delete a flag and its associated rules/overrides/variants from Redis."""
        keys_to_delete = [
            self._flag_key(key),
            self._rules_key(key),
            self._overrides_key(key),
            self._variants_key(key),
            self._audit_key(key),
        ]
        count = await self._safe_delete(*keys_to_delete)
        # Remove from index
        existing = await self.get_all_flags()
        remaining = [f for f in existing if f.key != key]
        if len(remaining) != len(existing):
            await self._safe_set(
                self._all_flags_key(),
                self._serialize(remaining),
            )
        await self._publish_invalidation(key)
        return count > 0

    # ------------------------------------------------------------------
    # IFlagStorage -- Rules
    # ------------------------------------------------------------------

    async def get_rules(self, flag_key: str) -> List[FlagRule]:
        """Retrieve rules from Redis, sorted by priority."""
        data = await self._safe_get(self._rules_key(flag_key))
        if data is None:
            return []
        try:
            rules = self._deserialize_rules(data)
            rules.sort(key=lambda r: r.priority)
            return rules
        except Exception as exc:
            logger.warning("Failed to deserialize rules for '%s': %s", flag_key, exc)
            return []

    async def save_rule(self, rule: FlagRule) -> FlagRule:
        """Upsert a rule into the rules list for a flag."""
        existing = await self.get_rules(rule.flag_key)
        updated = [r for r in existing if r.rule_id != rule.rule_id]
        updated.append(rule)
        await self._safe_set(
            self._rules_key(rule.flag_key), self._serialize(updated)
        )
        return rule

    # ------------------------------------------------------------------
    # IFlagStorage -- Overrides
    # ------------------------------------------------------------------

    async def get_overrides(self, flag_key: str) -> List[FlagOverride]:
        """Retrieve overrides from Redis."""
        data = await self._safe_get(self._overrides_key(flag_key))
        if data is None:
            return []
        try:
            return self._deserialize_overrides(data)
        except Exception as exc:
            logger.warning(
                "Failed to deserialize overrides for '%s': %s", flag_key, exc
            )
            return []

    async def save_override(self, override: FlagOverride) -> FlagOverride:
        """Upsert an override by (flag_key, scope_type, scope_value)."""
        existing = await self.get_overrides(override.flag_key)
        updated = [
            o for o in existing
            if not (
                o.scope_type == override.scope_type
                and o.scope_value == override.scope_value
            )
        ]
        updated.append(override)
        await self._safe_set(
            self._overrides_key(override.flag_key), self._serialize(updated)
        )
        return override

    # ------------------------------------------------------------------
    # IFlagStorage -- Variants
    # ------------------------------------------------------------------

    async def get_variants(self, flag_key: str) -> List[FlagVariant]:
        """Retrieve variants from Redis."""
        data = await self._safe_get(self._variants_key(flag_key))
        if data is None:
            return []
        try:
            return self._deserialize_variants(data)
        except Exception as exc:
            logger.warning(
                "Failed to deserialize variants for '%s': %s", flag_key, exc
            )
            return []

    async def save_variant(self, variant: FlagVariant) -> FlagVariant:
        """Upsert a variant by (flag_key, variant_key)."""
        existing = await self.get_variants(variant.flag_key)
        updated = [
            v for v in existing if v.variant_key != variant.variant_key
        ]
        updated.append(variant)
        await self._safe_set(
            self._variants_key(variant.flag_key), self._serialize(updated)
        )
        return variant

    # ------------------------------------------------------------------
    # IFlagStorage -- Audit Log
    # ------------------------------------------------------------------

    async def log_audit(self, entry: AuditLogEntry) -> None:
        """Append an audit entry to the Redis list for a flag.

        Audit entries in Redis are capped at the last 500 entries per
        flag. PostgreSQL (L3) is the authoritative audit store.
        """
        rkey = self._audit_key(entry.flag_key)
        if not self._initialized or self._client is None:
            return
        if not await self._circuit.can_attempt():
            return
        try:
            await self._client.rpush(rkey, self._serialize(entry))
            await self._client.ltrim(rkey, -500, -1)  # keep last 500
            await self._client.expire(rkey, self._ttl * 10)  # longer TTL for audit
            await self._circuit.record_success()
        except Exception as exc:
            await self._circuit.record_failure()
            logger.warning("Redis audit log append failed: %s", exc)

    # Keep backward-compatible alias
    async def append_audit_log(self, entry: AuditLogEntry) -> None:
        """Alias for ``log_audit``."""
        await self.log_audit(entry)

    async def get_audit_log(
        self,
        flag_key: str,
        limit: int = 50,
    ) -> List[AuditLogEntry]:
        """Retrieve audit log entries from Redis (most recent first)."""
        if not self._initialized or self._client is None:
            return []
        if not await self._circuit.can_attempt():
            return []
        try:
            rkey = self._audit_key(flag_key)
            # Get the last ``limit`` entries (stored chronologically)
            raw_entries = await self._client.lrange(rkey, -limit, -1)
            await self._circuit.record_success()
            entries: List[AuditLogEntry] = []
            for raw in reversed(raw_entries):  # reverse for newest-first
                try:
                    entries.append(AuditLogEntry.model_validate_json(raw))
                except Exception:
                    continue
            return entries
        except Exception as exc:
            await self._circuit.record_failure()
            logger.warning("Redis audit log read failed: %s", exc)
            return []

    # ------------------------------------------------------------------
    # Health
    # ------------------------------------------------------------------

    async def health_check(self) -> Dict[str, object]:
        """Check Redis connectivity and circuit breaker state."""
        result: Dict[str, object] = {
            "healthy": False,
            "backend": "RedisFlagStorage",
            "circuit_state": self._circuit.state.value,
            "initialized": self._initialized,
        }
        if not self._initialized or self._client is None:
            return result

        try:
            pong = await self._client.ping()
            result["healthy"] = bool(pong)
            result["ping"] = "PONG" if pong else "FAIL"
            info = await self._client.info("memory")
            result["used_memory_human"] = info.get("used_memory_human", "unknown")
        except Exception as exc:
            result["error"] = str(exc)

        return result

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    async def invalidate(self, key: str) -> bool:
        """Remove a single flag from L2 cache and publish invalidation."""
        count = await self._safe_delete(self._flag_key(key))
        await self._publish_invalidation(key)
        return count > 0
