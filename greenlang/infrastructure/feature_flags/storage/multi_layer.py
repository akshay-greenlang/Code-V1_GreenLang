# -*- coding: utf-8 -*-
"""
Multi-Layer Feature Flag Storage - INFRA-008

Cascading storage strategy that reads from the fastest available layer
and writes through all layers to maintain consistency.

Layer hierarchy:
    L1 (in-memory)  -- sub-microsecond reads, LRU + TTL eviction
    L2 (Redis)       -- sub-millisecond reads, pub/sub invalidation
    L3 (PostgreSQL)  -- persistent source-of-truth, audit log

Read path:
    L1 hit -> return
    L1 miss -> L2 hit -> populate L1 -> return
    L2 miss -> L3 hit -> populate L1 + L2 -> return
    all miss -> None

Write path:
    write to L3 (source of truth) -> invalidate L1 + L2 -> publish update

Features:
    - Async cache warming on startup (load active flags from L3 into L1/L2)
    - Cache invalidation via Redis pub/sub listener
    - Per-layer cache hit/miss metrics
    - Graceful degradation: if Redis is down, skip L2; if Postgres is down,
      serve from cache

Example:
    >>> from greenlang.infrastructure.feature_flags.config import FeatureFlagConfig
    >>> storage = await MultiLayerFlagStorage.create(FeatureFlagConfig())
    >>> flag = await storage.get_flag("enable-scope3-calc")
    >>> metrics = storage.get_metrics()
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

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
from greenlang.infrastructure.feature_flags.storage.memory import InMemoryFlagStorage

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Cache metrics
# ---------------------------------------------------------------------------


@dataclass
class _LayerMetrics:
    """Per-layer cache hit/miss counters."""

    hits: int = 0
    misses: int = 0

    @property
    def total(self) -> int:
        return self.hits + self.misses

    @property
    def hit_rate(self) -> float:
        if self.total == 0:
            return 0.0
        return self.hits / self.total

    def to_dict(self) -> Dict[str, Any]:
        return {
            "hits": self.hits,
            "misses": self.misses,
            "total": self.total,
            "hit_rate": round(self.hit_rate, 4),
        }


@dataclass
class CacheMetrics:
    """Aggregated cache metrics across all layers."""

    l1: _LayerMetrics = field(default_factory=_LayerMetrics)
    l2: _LayerMetrics = field(default_factory=_LayerMetrics)
    l3: _LayerMetrics = field(default_factory=_LayerMetrics)
    cache_warms: int = 0
    invalidations: int = 0
    start_time: float = field(default_factory=time.monotonic)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "l1": self.l1.to_dict(),
            "l2": self.l2.to_dict(),
            "l3": self.l3.to_dict(),
            "cache_warms": self.cache_warms,
            "invalidations": self.invalidations,
            "uptime_seconds": round(time.monotonic() - self.start_time, 2),
        }


# ---------------------------------------------------------------------------
# MultiLayerFlagStorage
# ---------------------------------------------------------------------------


class MultiLayerFlagStorage(IFlagStorage):
    """Cascading multi-layer storage with read-through and write-through semantics.

    L1 (in-memory) is always present. L2 (Redis) and L3 (PostgreSQL) are
    optional and injected at construction time. If L2 or L3 experience
    errors they are silently skipped (graceful degradation) with a warning
    logged.

    Attributes:
        _l1: In-memory cache (always present).
        _l2: Optional Redis cache.
        _l3: Optional PostgreSQL persistent store.
        _metrics: Per-layer hit/miss counters.
    """

    def __init__(
        self,
        l1: Optional[InMemoryFlagStorage] = None,
        l2: Optional[IFlagStorage] = None,
        l3: Optional[IFlagStorage] = None,
        config: Optional[FeatureFlagConfig] = None,
    ) -> None:
        """Initialize multi-layer storage.

        Args:
            l1: In-memory cache layer. Created automatically if None.
            l2: Optional Redis cache layer.
            l3: Optional PostgreSQL persistence layer.
            config: Optional configuration for tuning cache TTLs etc.
        """
        cfg = config or FeatureFlagConfig()
        self._l1: InMemoryFlagStorage = l1 or InMemoryFlagStorage(
            max_size=cfg.cache_ttl_seconds,
            default_ttl=float(cfg.cache_ttl_seconds),
        )
        self._l2: Optional[IFlagStorage] = l2
        self._l3: Optional[IFlagStorage] = l3
        self._config = cfg
        self._metrics = CacheMetrics()
        self._pubsub_task: Optional[asyncio.Task] = None

        logger.info(
            "MultiLayerFlagStorage initialised: L1=memory, L2=%s, L3=%s",
            type(self._l2).__name__ if self._l2 else "none",
            type(self._l3).__name__ if self._l3 else "none",
        )

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    async def create(
        cls,
        config: Optional[FeatureFlagConfig] = None,
    ) -> "MultiLayerFlagStorage":
        """Factory that builds the full L1/L2/L3 stack from config.

        Creates Redis and PostgreSQL backends if their URLs are configured,
        initialises them, warms the cache, and starts the pub/sub listener.

        Args:
            config: Feature flag configuration.

        Returns:
            A fully initialised MultiLayerFlagStorage.
        """
        cfg = config or FeatureFlagConfig()

        l1 = InMemoryFlagStorage(
            max_size=10_000,
            default_ttl=float(cfg.cache_ttl_seconds),
        )

        l2: Optional[IFlagStorage] = None
        l3: Optional[IFlagStorage] = None

        # L3 -- PostgreSQL
        if cfg.postgres_dsn:
            try:
                from greenlang.infrastructure.feature_flags.storage.postgres_store import (
                    PostgresFlagStorage,
                )
                pg = PostgresFlagStorage(cfg)
                await pg.initialize()
                l3 = pg
            except Exception as exc:
                logger.warning("L3 (PostgreSQL) initialisation failed: %s", exc)

        # L2 -- Redis (pass invalidation callback that clears L1)
        if cfg.redis_url:
            try:
                from greenlang.infrastructure.feature_flags.storage.redis_store import (
                    RedisFlagStorage,
                )
                store = cls(l1=l1, l2=None, l3=l3, config=cfg)
                redis = RedisFlagStorage(
                    cfg,
                    on_invalidation=store._handle_invalidation,
                )
                await redis.initialize()
                store._l2 = redis
                # Warm cache from L3
                await store.warm_cache()
                return store
            except Exception as exc:
                logger.warning("L2 (Redis) initialisation failed: %s", exc)

        store = cls(l1=l1, l2=l2, l3=l3, config=cfg)
        await store.warm_cache()
        return store

    # ------------------------------------------------------------------
    # Pub/sub invalidation callback
    # ------------------------------------------------------------------

    def _handle_invalidation(self, flag_key: str) -> Any:
        """Called by the Redis pub/sub listener when a flag is updated.

        Invalidates the L1 cache entry so the next read will fall through
        to L2/L3.
        """
        self._metrics.invalidations += 1
        logger.debug("Pub/sub invalidation received for '%s'", flag_key)
        # Return a coroutine so the pub/sub listener can await it
        return self._l1.invalidate(flag_key)

    # ------------------------------------------------------------------
    # IFlagStorage -- Flags
    # ------------------------------------------------------------------

    async def get_flag(self, key: str) -> Optional[FeatureFlag]:
        """Read-through flag lookup: L1 -> L2 -> L3.

        Populates lower-layer caches on miss for subsequent reads.
        """
        # L1
        flag = await self._l1.get_flag(key)
        if flag is not None:
            self._metrics.l1.hits += 1
            return flag
        self._metrics.l1.misses += 1

        # L2
        if self._l2 is not None:
            try:
                flag = await self._l2.get_flag(key)
            except Exception as exc:
                logger.warning("L2 get_flag('%s') failed: %s", key, exc)
                flag = None
            if flag is not None:
                self._metrics.l2.hits += 1
                await self._l1.save_flag(flag)
                return flag
            self._metrics.l2.misses += 1

        # L3
        if self._l3 is not None:
            try:
                flag = await self._l3.get_flag(key)
            except Exception as exc:
                logger.warning("L3 get_flag('%s') failed: %s", key, exc)
                flag = None
            if flag is not None:
                self._metrics.l3.hits += 1
                await self._l1.save_flag(flag)
                if self._l2 is not None:
                    try:
                        await self._l2.save_flag(flag)
                    except Exception as exc:
                        logger.warning("L2 backfill failed for '%s': %s", key, exc)
                return flag
            self._metrics.l3.misses += 1

        return None

    async def get_all_flags(
        self,
        status_filter: Optional[FlagStatus] = None,
        tag_filter: Optional[str] = None,
    ) -> List[FeatureFlag]:
        """Get all flags from the deepest available layer.

        Falls back to shallower layers on error.
        """
        # Prefer L3 for completeness
        if self._l3 is not None:
            try:
                return await self._l3.get_all_flags(
                    status_filter=status_filter, tag_filter=tag_filter
                )
            except Exception as exc:
                logger.warning("L3 get_all_flags failed: %s", exc)

        if self._l2 is not None:
            try:
                return await self._l2.get_all_flags(
                    status_filter=status_filter, tag_filter=tag_filter
                )
            except Exception as exc:
                logger.warning("L2 get_all_flags failed: %s", exc)

        return await self._l1.get_all_flags(
            status_filter=status_filter, tag_filter=tag_filter
        )

    async def save_flag(self, flag: FeatureFlag) -> FeatureFlag:
        """Write-through: persist to L3, then update L1 and invalidate L2.

        If L3 is unavailable the flag is still saved to L1 (and L2 if
        available) so the system remains functional.
        """
        # L3 first (source of truth)
        if self._l3 is not None:
            try:
                flag = await self._l3.save_flag(flag)
            except Exception as exc:
                logger.warning("L3 save_flag('%s') failed: %s", flag.key, exc)

        # L1 always updated
        await self._l1.save_flag(flag)

        # L2 -- save then publish invalidation
        if self._l2 is not None:
            try:
                await self._l2.save_flag(flag)
            except Exception as exc:
                logger.warning("L2 save_flag('%s') failed: %s", flag.key, exc)

        return flag

    async def delete_flag(self, key: str) -> bool:
        """Delete from all layers. Returns True if found in any layer."""
        found = False

        # L3 first
        if self._l3 is not None:
            try:
                found = await self._l3.delete_flag(key) or found
            except Exception as exc:
                logger.warning("L3 delete_flag('%s') failed: %s", key, exc)

        # L2
        if self._l2 is not None:
            try:
                found = await self._l2.delete_flag(key) or found
            except Exception as exc:
                logger.warning("L2 delete_flag('%s') failed: %s", key, exc)

        # L1
        found = await self._l1.delete_flag(key) or found
        return found

    # ------------------------------------------------------------------
    # Extended flag queries (delegated to L1)
    # ------------------------------------------------------------------

    async def list_flags(
        self,
        status: Optional[FlagStatus] = None,
        tag: Optional[str] = None,
        owner: Optional[str] = None,
        flag_type: Optional[str] = None,
        offset: int = 0,
        limit: int = 100,
    ) -> List[FeatureFlag]:
        """List flags with filtering and pagination.

        Delegates to L1 which holds the warmed cache. For paginated
        production queries consider calling L3 directly.
        """
        return await self._l1.list_flags(
            status=status, tag=tag, owner=owner,
            flag_type=flag_type, offset=offset, limit=limit,
        )

    async def count_flags(
        self,
        status: Optional[FlagStatus] = None,
        tag: Optional[str] = None,
        owner: Optional[str] = None,
        flag_type: Optional[str] = None,
    ) -> int:
        """Count flags matching filters (delegates to L1)."""
        return await self._l1.count_flags(
            status=status, tag=tag, owner=owner, flag_type=flag_type,
        )

    # ------------------------------------------------------------------
    # IFlagStorage -- Rules
    # ------------------------------------------------------------------

    async def get_rules(self, flag_key: str) -> List[FlagRule]:
        """Read-through rule lookup: L1 -> L2 -> L3."""
        rules = await self._l1.get_rules(flag_key)
        if rules:
            return rules

        if self._l2 is not None:
            try:
                rules = await self._l2.get_rules(flag_key)
            except Exception as exc:
                logger.warning("L2 get_rules('%s') failed: %s", flag_key, exc)
                rules = []
            if rules:
                for rule in rules:
                    await self._l1.save_rule(rule)
                return rules

        if self._l3 is not None:
            try:
                rules = await self._l3.get_rules(flag_key)
            except Exception as exc:
                logger.warning("L3 get_rules('%s') failed: %s", flag_key, exc)
                rules = []
            if rules:
                for rule in rules:
                    await self._l1.save_rule(rule)
                if self._l2 is not None:
                    try:
                        for rule in rules:
                            await self._l2.save_rule(rule)
                    except Exception as exc:
                        logger.warning("L2 backfill rules failed: %s", exc)
                return rules

        return []

    async def save_rule(self, rule: FlagRule) -> FlagRule:
        """Write-through rule save: L3 -> L1, invalidate L2."""
        if self._l3 is not None:
            try:
                rule = await self._l3.save_rule(rule)
            except Exception as exc:
                logger.warning("L3 save_rule failed: %s", exc)

        await self._l1.save_rule(rule)

        if self._l2 is not None:
            try:
                await self._l2.save_rule(rule)
            except Exception as exc:
                logger.warning("L2 save_rule failed: %s", exc)

        return rule

    async def delete_rule(self, flag_key: str, rule_id: str) -> bool:
        """Delete a rule from all layers."""
        found = False

        if self._l3 is not None:
            try:
                found = await self._l3.delete_rule(flag_key, rule_id) or found
            except Exception as exc:
                logger.warning("L3 delete_rule failed: %s", exc)

        if self._l2 is not None:
            try:
                found = await self._l2.delete_rule(flag_key, rule_id) or found
            except Exception as exc:
                logger.warning("L2 delete_rule failed: %s", exc)

        found = await self._l1.delete_rule(flag_key, rule_id) or found
        return found

    # ------------------------------------------------------------------
    # IFlagStorage -- Overrides
    # ------------------------------------------------------------------

    async def get_overrides(self, flag_key: str) -> List[FlagOverride]:
        """Read-through override lookup: L1 -> L2 -> L3."""
        overrides = await self._l1.get_overrides(flag_key)
        if overrides:
            return overrides

        if self._l2 is not None:
            try:
                overrides = await self._l2.get_overrides(flag_key)
            except Exception as exc:
                logger.warning("L2 get_overrides('%s') failed: %s", flag_key, exc)
                overrides = []
            if overrides:
                for o in overrides:
                    await self._l1.save_override(o)
                return overrides

        if self._l3 is not None:
            try:
                overrides = await self._l3.get_overrides(flag_key)
            except Exception as exc:
                logger.warning("L3 get_overrides('%s') failed: %s", flag_key, exc)
                overrides = []
            if overrides:
                for o in overrides:
                    await self._l1.save_override(o)
                if self._l2 is not None:
                    try:
                        for o in overrides:
                            await self._l2.save_override(o)
                    except Exception as exc:
                        logger.warning("L2 backfill overrides failed: %s", exc)
                return overrides

        return []

    async def save_override(self, override: FlagOverride) -> FlagOverride:
        """Write-through override save."""
        if self._l3 is not None:
            try:
                override = await self._l3.save_override(override)
            except Exception as exc:
                logger.warning("L3 save_override failed: %s", exc)

        await self._l1.save_override(override)

        if self._l2 is not None:
            try:
                await self._l2.save_override(override)
            except Exception as exc:
                logger.warning("L2 save_override failed: %s", exc)

        return override

    async def delete_override(
        self, flag_key: str, scope_type: str, scope_value: str
    ) -> bool:
        """Delete an override from all layers."""
        found = False

        if self._l3 is not None:
            try:
                found = await self._l3.delete_override(
                    flag_key, scope_type, scope_value
                ) or found
            except Exception as exc:
                logger.warning("L3 delete_override failed: %s", exc)

        if self._l2 is not None:
            try:
                found = await self._l2.delete_override(
                    flag_key, scope_type, scope_value
                ) or found
            except Exception as exc:
                logger.warning("L2 delete_override failed: %s", exc)

        found = await self._l1.delete_override(
            flag_key, scope_type, scope_value
        ) or found
        return found

    # ------------------------------------------------------------------
    # IFlagStorage -- Variants
    # ------------------------------------------------------------------

    async def get_variants(self, flag_key: str) -> List[FlagVariant]:
        """Read-through variant lookup: L1 -> L2 -> L3."""
        variants = await self._l1.get_variants(flag_key)
        if variants:
            return variants

        if self._l2 is not None:
            try:
                variants = await self._l2.get_variants(flag_key)
            except Exception as exc:
                logger.warning("L2 get_variants('%s') failed: %s", flag_key, exc)
                variants = []
            if variants:
                for v in variants:
                    await self._l1.save_variant(v)
                return variants

        if self._l3 is not None:
            try:
                variants = await self._l3.get_variants(flag_key)
            except Exception as exc:
                logger.warning("L3 get_variants('%s') failed: %s", flag_key, exc)
                variants = []
            if variants:
                for v in variants:
                    await self._l1.save_variant(v)
                if self._l2 is not None:
                    try:
                        for v in variants:
                            await self._l2.save_variant(v)
                    except Exception as exc:
                        logger.warning("L2 backfill variants failed: %s", exc)
                return variants

        return []

    async def save_variant(self, variant: FlagVariant) -> FlagVariant:
        """Write-through variant save."""
        if self._l3 is not None:
            try:
                variant = await self._l3.save_variant(variant)
            except Exception as exc:
                logger.warning("L3 save_variant failed: %s", exc)

        await self._l1.save_variant(variant)

        if self._l2 is not None:
            try:
                await self._l2.save_variant(variant)
            except Exception as exc:
                logger.warning("L2 save_variant failed: %s", exc)

        return variant

    async def delete_variant(self, flag_key: str, variant_key: str) -> bool:
        """Delete a variant from all layers."""
        found = False

        if self._l3 is not None:
            try:
                found = await self._l3.delete_variant(
                    flag_key, variant_key
                ) or found
            except Exception as exc:
                logger.warning("L3 delete_variant failed: %s", exc)

        if self._l2 is not None:
            try:
                found = await self._l2.delete_variant(
                    flag_key, variant_key
                ) or found
            except Exception as exc:
                logger.warning("L2 delete_variant failed: %s", exc)

        found = await self._l1.delete_variant(flag_key, variant_key) or found
        return found

    # ------------------------------------------------------------------
    # IFlagStorage -- Audit Log
    # ------------------------------------------------------------------

    async def log_audit(self, entry: AuditLogEntry) -> None:
        """Write audit log to all layers.

        L3 (PostgreSQL) is the authoritative audit store. L1/L2 keep a
        bounded recent history for fast reads.
        """
        # L3 first (authoritative)
        if self._l3 is not None:
            try:
                await self._l3.log_audit(entry)
            except Exception as exc:
                logger.warning("L3 log_audit failed: %s", exc)

        # L1 always
        await self._l1.log_audit(entry)

        # L2
        if self._l2 is not None:
            try:
                await self._l2.log_audit(entry)
            except Exception as exc:
                logger.warning("L2 log_audit failed: %s", exc)

    # Keep backward-compatible alias
    async def append_audit_log(self, entry: AuditLogEntry) -> None:
        """Alias for ``log_audit``."""
        await self.log_audit(entry)

    async def get_audit_log(
        self,
        flag_key: str,
        limit: int = 50,
    ) -> List[AuditLogEntry]:
        """Retrieve audit log from L3 first, falling back to L1.

        L3 is authoritative. L1 is the fallback for when Postgres is
        unavailable.
        """
        if self._l3 is not None:
            try:
                return await self._l3.get_audit_log(flag_key, limit=limit)
            except Exception as exc:
                logger.warning("L3 get_audit_log failed: %s", exc)

        return await self._l1.get_audit_log(flag_key, limit=limit)

    # ------------------------------------------------------------------
    # Cache warming
    # ------------------------------------------------------------------

    async def warm_cache(self) -> int:
        """Pre-populate L1 (and L2) from L3.

        Loads all active flags and their associated rules, overrides, and
        variants from the deepest available layer into the cache layers.

        Returns:
            Number of flags loaded into L1 cache.
        """
        start = time.monotonic()
        source_layer = "L3" if self._l3 else ("L2" if self._l2 else "L1")

        # Get all flags from the deepest layer
        all_flags = await self.get_all_flags()
        loaded = 0

        for flag in all_flags:
            await self._l1.save_flag(flag)
            loaded += 1

            # Warm rules, overrides, variants for active flags
            if flag.status in (FlagStatus.ACTIVE, FlagStatus.ROLLED_OUT):
                # Get from deepest and save to L1 (get_rules does read-through)
                await self.get_rules(flag.key)
                await self.get_overrides(flag.key)
                await self.get_variants(flag.key)

        self._metrics.cache_warms += 1
        elapsed_ms = (time.monotonic() - start) * 1000
        logger.info(
            "Cache warmed from %s: %d flags in %.1fms",
            source_layer, loaded, elapsed_ms,
        )
        return loaded

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------

    def get_metrics(self) -> Dict[str, Any]:
        """Return cache hit/miss metrics for all layers."""
        return self._metrics.to_dict()

    # ------------------------------------------------------------------
    # Health / Lifecycle
    # ------------------------------------------------------------------

    async def health_check(self) -> Dict[str, object]:
        """Return health status of all layers."""
        result: Dict[str, object] = {
            "healthy": False,
            "backend": "MultiLayerFlagStorage",
            "layers": {},
        }

        layers_info: Dict[str, Any] = {}

        # L1 (always healthy if present)
        try:
            l1_health = await self._l1.health_check()
            layers_info["l1"] = l1_health
        except Exception as exc:
            layers_info["l1"] = {"healthy": False, "error": str(exc)}

        # L2
        if self._l2 is not None:
            try:
                l2_health = await self._l2.health_check()
                layers_info["l2"] = l2_health
            except Exception as exc:
                layers_info["l2"] = {"healthy": False, "error": str(exc)}
                logger.warning("L2 health check failed: %s", exc)
        else:
            layers_info["l2"] = {"configured": False}

        # L3
        if self._l3 is not None:
            try:
                l3_health = await self._l3.health_check()
                layers_info["l3"] = l3_health
            except Exception as exc:
                layers_info["l3"] = {"healthy": False, "error": str(exc)}
                logger.warning("L3 health check failed: %s", exc)
        else:
            layers_info["l3"] = {"configured": False}

        result["layers"] = layers_info
        result["metrics"] = self._metrics.to_dict()

        # Overall: healthy if L1 is up (we can always serve from cache)
        l1_ok = layers_info.get("l1", {})
        result["healthy"] = l1_ok.get("healthy", False) if isinstance(l1_ok, dict) else False

        return result

    async def close(self) -> None:
        """Shut down all storage layers and background tasks."""
        if self._pubsub_task is not None:
            self._pubsub_task.cancel()
            try:
                await self._pubsub_task
            except asyncio.CancelledError:
                pass
            self._pubsub_task = None

        if self._l2 is not None:
            try:
                await self._l2.close()
            except Exception as exc:
                logger.warning("L2 close failed: %s", exc)

        if self._l3 is not None:
            try:
                await self._l3.close()
            except Exception as exc:
                logger.warning("L3 close failed: %s", exc)

        await self._l1.close()
        logger.info("MultiLayerFlagStorage closed all layers")

    # Keep backward-compatible aliases
    async def initialize(self) -> None:
        """Initialize all storage layers and warm cache."""
        await self._l1.initialize()
        if self._l2 is not None:
            try:
                await self._l2.initialize()
            except Exception as exc:
                logger.warning("L2 initialize failed: %s", exc)
        if self._l3 is not None:
            try:
                await self._l3.initialize()
            except Exception as exc:
                logger.warning("L3 initialize failed: %s", exc)
        await self.warm_cache()
        logger.info("MultiLayerFlagStorage initialized all layers")

    async def shutdown(self) -> None:
        """Alias for ``close()``."""
        await self.close()
