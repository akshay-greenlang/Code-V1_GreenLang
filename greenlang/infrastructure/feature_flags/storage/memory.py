# -*- coding: utf-8 -*-
"""
In-Memory Feature Flag Storage - INFRA-008

Thread-safe, asyncio-safe in-memory storage backend for feature flags.
Serves as the L1 cache layer in the multi-layer architecture and as the
default standalone backend for development and testing environments.

Features:
    - asyncio.Lock for concurrency safety across coroutines
    - LRU eviction with configurable ``max_size``
    - Per-entry TTL with lazy expiration on read
    - Audit log with configurable max entries
    - O(1) flag lookup by key via OrderedDict

Data is lost on process restart. Use MultiLayerFlagStorage with Redis
and/or PostgreSQL backends for persistence in staging and production.

Example:
    >>> storage = InMemoryFlagStorage(max_size=5000, default_ttl=30.0)
    >>> await storage.save_flag(flag)
    >>> result = await storage.get_flag("enable-scope3-calc")
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

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
# Internal cache entry wrapper
# ---------------------------------------------------------------------------


@dataclass
class _CacheEntry:
    """Wrapper that stores a value alongside monotonic-clock TTL metadata."""

    value: Any
    created_at: float = field(default_factory=time.monotonic)
    ttl: float = 0.0  # 0 means no expiration

    @property
    def is_expired(self) -> bool:
        """Return True if this entry has outlived its TTL."""
        if self.ttl <= 0:
            return False
        return (time.monotonic() - self.created_at) > self.ttl


# ---------------------------------------------------------------------------
# InMemoryFlagStorage
# ---------------------------------------------------------------------------


class InMemoryFlagStorage(IFlagStorage):
    """Fully async, LRU-evicting in-memory feature flag storage.

    All data structures are guarded by a single ``asyncio.Lock``.
    The flag cache is an ``OrderedDict`` providing O(1) LRU eviction
    when entries exceed ``max_size``.  Every cache entry carries an
    optional TTL so stale data is lazily discarded on the next read.

    Use-cases:

    1. **L1 cache** -- fastest layer in the multi-layer architecture.
    2. **Standalone backend** -- local development and unit tests.

    Args:
        max_size: Maximum number of flag entries before LRU eviction.
        default_ttl: Default TTL in seconds for flag entries (0 = no expiry).
        audit_max_entries: Maximum number of audit log entries retained.
    """

    def __init__(
        self,
        max_size: int = 10_000,
        default_ttl: float = 0.0,
        audit_max_entries: int = 10_000,
    ) -> None:
        self._max_size: int = max(1, max_size)
        self._default_ttl: float = max(0.0, default_ttl)
        self._audit_max_entries: int = max(100, audit_max_entries)

        # Primary data stores -- all guarded by ``_lock``
        self._flags: OrderedDict[str, _CacheEntry] = OrderedDict()
        self._rules: Dict[str, List[FlagRule]] = {}
        self._overrides: Dict[str, List[FlagOverride]] = {}
        self._variants: Dict[str, List[FlagVariant]] = {}
        self._audit_log: Dict[str, List[AuditLogEntry]] = {}

        self._lock = asyncio.Lock()

        logger.info(
            "InMemoryFlagStorage initialised "
            "(max_size=%d, default_ttl=%.1fs, audit_max=%d)",
            self._max_size,
            self._default_ttl,
            self._audit_max_entries,
        )

    # ------------------------------------------------------------------
    # Internal helpers (must be called while holding ``_lock``)
    # ------------------------------------------------------------------

    def _evict_expired(self) -> int:
        """Remove expired entries from the flag cache. Returns count."""
        expired_keys = [k for k, e in self._flags.items() if e.is_expired]
        for k in expired_keys:
            del self._flags[k]
        return len(expired_keys)

    def _evict_lru(self) -> None:
        """Evict least-recently-used entries until size <= max_size."""
        while len(self._flags) > self._max_size:
            evicted_key, _ = self._flags.popitem(last=False)
            logger.debug("LRU eviction: removed flag '%s'", evicted_key)

    def _touch(self, key: str) -> None:
        """Promote a key to most-recently-used position."""
        self._flags.move_to_end(key)

    # ------------------------------------------------------------------
    # IFlagStorage -- Flags
    # ------------------------------------------------------------------

    async def get_flag(self, key: str) -> Optional[FeatureFlag]:
        """Retrieve a flag by key with lazy TTL expiration."""
        async with self._lock:
            entry = self._flags.get(key)
            if entry is None:
                return None
            if entry.is_expired:
                del self._flags[key]
                logger.debug("Flag '%s' expired and removed from L1", key)
                return None
            self._touch(key)
            return entry.value

    async def get_all_flags(
        self,
        status_filter: Optional[FlagStatus] = None,
        tag_filter: Optional[str] = None,
    ) -> List[FeatureFlag]:
        """Return all non-expired flags, optionally filtered by status/tag."""
        async with self._lock:
            self._evict_expired()
            results: List[FeatureFlag] = []
            for entry in self._flags.values():
                flag: FeatureFlag = entry.value
                if status_filter is not None and flag.status != status_filter:
                    continue
                if tag_filter is not None and tag_filter.lower() not in flag.tags:
                    continue
                results.append(flag)
            return results

    async def save_flag(self, flag: FeatureFlag) -> FeatureFlag:
        """Upsert a flag into the L1 cache and return it."""
        async with self._lock:
            self._flags[flag.key] = _CacheEntry(
                value=flag,
                ttl=self._default_ttl,
            )
            self._touch(flag.key)
            self._evict_lru()
        logger.debug("Saved flag '%s' to L1 cache", flag.key)
        return flag

    async def delete_flag(self, key: str) -> bool:
        """Delete a flag and cascade to rules, overrides, and variants."""
        async with self._lock:
            existed = key in self._flags
            self._flags.pop(key, None)
            self._rules.pop(key, None)
            self._overrides.pop(key, None)
            self._variants.pop(key, None)
        if existed:
            logger.info("Deleted flag '%s' and cascaded sub-entities from L1", key)
        return existed

    # ------------------------------------------------------------------
    # Extended flag queries (used by API layer, not part of IFlagStorage)
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
        """List flags with optional filtering and pagination.

        Args:
            status: Filter by flag status.
            tag: Filter by tag (case-insensitive).
            owner: Filter by owner (case-insensitive).
            flag_type: Filter by flag type string.
            offset: Number of results to skip.
            limit: Maximum number of results to return.

        Returns:
            Paginated list of matching flags, newest first.
        """
        async with self._lock:
            self._evict_expired()
            flags = [e.value for e in self._flags.values() if not e.is_expired]

        if status is not None:
            flags = [f for f in flags if f.status == status]
        if tag is not None:
            tag_lower = tag.lower()
            flags = [f for f in flags if tag_lower in f.tags]
        if owner is not None:
            owner_lower = owner.lower()
            flags = [f for f in flags if f.owner.lower() == owner_lower]
        if flag_type is not None:
            flags = [f for f in flags if f.flag_type.value == flag_type]

        flags.sort(key=lambda f: f.created_at, reverse=True)
        return flags[offset: offset + limit]

    async def count_flags(
        self,
        status: Optional[FlagStatus] = None,
        tag: Optional[str] = None,
        owner: Optional[str] = None,
        flag_type: Optional[str] = None,
    ) -> int:
        """Count flags matching the given filters.

        Args:
            status: Filter by flag status.
            tag: Filter by tag.
            owner: Filter by owner.
            flag_type: Filter by flag type string.

        Returns:
            Count of matching flags.
        """
        matching = await self.list_flags(
            status=status, tag=tag, owner=owner, flag_type=flag_type,
            offset=0, limit=999_999,
        )
        return len(matching)

    # ------------------------------------------------------------------
    # IFlagStorage -- Rules
    # ------------------------------------------------------------------

    async def get_rules(self, flag_key: str) -> List[FlagRule]:
        """Return rules for a flag sorted by ascending priority."""
        async with self._lock:
            rules = list(self._rules.get(flag_key, []))
        rules.sort(key=lambda r: r.priority)
        return rules

    async def save_rule(self, rule: FlagRule) -> FlagRule:
        """Upsert a targeting rule, replacing any existing rule with the same rule_id."""
        async with self._lock:
            if rule.flag_key not in self._rules:
                self._rules[rule.flag_key] = []
            self._rules[rule.flag_key] = [
                r for r in self._rules[rule.flag_key] if r.rule_id != rule.rule_id
            ]
            self._rules[rule.flag_key].append(rule)
        logger.debug("Saved rule '%s' for flag '%s'", rule.rule_id, rule.flag_key)
        return rule

    async def delete_rule(self, flag_key: str, rule_id: str) -> bool:
        """Delete a rule by flag key and rule ID.

        Args:
            flag_key: Parent flag key.
            rule_id: Rule identifier to delete.

        Returns:
            True if the rule existed and was deleted.
        """
        async with self._lock:
            if flag_key not in self._rules:
                return False
            before = len(self._rules[flag_key])
            self._rules[flag_key] = [
                r for r in self._rules[flag_key] if r.rule_id != rule_id
            ]
            return len(self._rules[flag_key]) < before

    # ------------------------------------------------------------------
    # IFlagStorage -- Overrides
    # ------------------------------------------------------------------

    async def get_overrides(self, flag_key: str) -> List[FlagOverride]:
        """Return all overrides for a flag."""
        async with self._lock:
            return list(self._overrides.get(flag_key, []))

    async def save_override(self, override: FlagOverride) -> FlagOverride:
        """Upsert an override, keyed by (flag_key, scope_type, scope_value)."""
        async with self._lock:
            if override.flag_key not in self._overrides:
                self._overrides[override.flag_key] = []
            self._overrides[override.flag_key] = [
                o for o in self._overrides[override.flag_key]
                if not (
                    o.scope_type == override.scope_type
                    and o.scope_value == override.scope_value
                )
            ]
            self._overrides[override.flag_key].append(override)
        logger.debug(
            "Saved override for flag '%s' scope=%s:%s",
            override.flag_key, override.scope_type, override.scope_value,
        )
        return override

    async def delete_override(
        self, flag_key: str, scope_type: str, scope_value: str
    ) -> bool:
        """Delete an override by flag key, scope type, and scope value.

        Args:
            flag_key: Parent flag key.
            scope_type: Override scope type.
            scope_value: Override scope value.

        Returns:
            True if the override existed and was deleted.
        """
        async with self._lock:
            if flag_key not in self._overrides:
                return False
            before = len(self._overrides[flag_key])
            self._overrides[flag_key] = [
                o for o in self._overrides[flag_key]
                if not (
                    o.scope_type == scope_type and o.scope_value == scope_value
                )
            ]
            return len(self._overrides[flag_key]) < before

    # ------------------------------------------------------------------
    # IFlagStorage -- Variants
    # ------------------------------------------------------------------

    async def get_variants(self, flag_key: str) -> List[FlagVariant]:
        """Return all variants for a flag."""
        async with self._lock:
            return list(self._variants.get(flag_key, []))

    async def save_variant(self, variant: FlagVariant) -> FlagVariant:
        """Upsert a variant, keyed by (flag_key, variant_key)."""
        async with self._lock:
            if variant.flag_key not in self._variants:
                self._variants[variant.flag_key] = []
            self._variants[variant.flag_key] = [
                v for v in self._variants[variant.flag_key]
                if v.variant_key != variant.variant_key
            ]
            self._variants[variant.flag_key].append(variant)
        logger.debug(
            "Saved variant '%s' for flag '%s'",
            variant.variant_key, variant.flag_key,
        )
        return variant

    async def delete_variant(self, flag_key: str, variant_key: str) -> bool:
        """Delete a variant by flag key and variant key.

        Args:
            flag_key: Parent flag key.
            variant_key: Variant identifier to delete.

        Returns:
            True if the variant existed and was deleted.
        """
        async with self._lock:
            if flag_key not in self._variants:
                return False
            before = len(self._variants[flag_key])
            self._variants[flag_key] = [
                v for v in self._variants[flag_key]
                if v.variant_key != variant_key
            ]
            return len(self._variants[flag_key]) < before

    # ------------------------------------------------------------------
    # IFlagStorage -- Audit Log
    # ------------------------------------------------------------------

    async def log_audit(self, entry: AuditLogEntry) -> None:
        """Append an immutable audit entry, trimming oldest if over capacity."""
        async with self._lock:
            if entry.flag_key not in self._audit_log:
                self._audit_log[entry.flag_key] = []
            self._audit_log[entry.flag_key].append(entry)
            # Trim from the front (oldest) if over the global cap
            total = sum(len(v) for v in self._audit_log.values())
            if total > self._audit_max_entries:
                self._trim_audit_log()

    # Keep backward-compatible alias for callers using the old name
    async def append_audit_log(self, entry: AuditLogEntry) -> None:
        """Alias for ``log_audit`` for backward compatibility."""
        await self.log_audit(entry)

    def _trim_audit_log(self) -> None:
        """Remove oldest entries across all flag keys until within capacity.

        Must be called while holding ``_lock``.
        """
        # Collect all entries with their flag_key, sort by created_at
        all_entries: List[tuple] = []
        for fk, entries in self._audit_log.items():
            for e in entries:
                all_entries.append((e.created_at, fk, e))
        all_entries.sort(key=lambda t: t[0])

        # Keep only the newest audit_max_entries
        to_keep = all_entries[-self._audit_max_entries:]
        self._audit_log.clear()
        for _, fk, entry in to_keep:
            if fk not in self._audit_log:
                self._audit_log[fk] = []
            self._audit_log[fk].append(entry)

    async def get_audit_log(
        self,
        flag_key: str,
        limit: int = 50,
    ) -> List[AuditLogEntry]:
        """Return the most recent audit entries for a flag.

        Args:
            flag_key: The flag key to query.
            limit: Maximum number of entries to return.

        Returns:
            Entries in reverse chronological order.
        """
        async with self._lock:
            entries = list(self._audit_log.get(flag_key, []))
        entries.sort(key=lambda e: e.created_at, reverse=True)
        return entries[:limit]

    # ------------------------------------------------------------------
    # Health / Lifecycle
    # ------------------------------------------------------------------

    async def health_check(self) -> Dict[str, object]:
        """Return health status including current cache metrics."""
        async with self._lock:
            expired_count = self._evict_expired()
            return {
                "healthy": True,
                "backend": "InMemoryFlagStorage",
                "flags_count": len(self._flags),
                "rules_count": sum(len(r) for r in self._rules.values()),
                "overrides_count": sum(len(o) for o in self._overrides.values()),
                "variants_count": sum(len(v) for v in self._variants.values()),
                "audit_log_count": sum(len(a) for a in self._audit_log.values()),
                "expired_evicted": expired_count,
                "max_size": self._max_size,
                "default_ttl": self._default_ttl,
            }

    async def close(self) -> None:
        """Clear all data and release resources."""
        async with self._lock:
            self._flags.clear()
            self._rules.clear()
            self._overrides.clear()
            self._variants.clear()
            self._audit_log.clear()
        logger.info("InMemoryFlagStorage closed and all data cleared")

    # Keep backward-compatible aliases
    async def initialize(self) -> None:
        """No-op for in-memory storage. Provided for interface symmetry."""
        logger.info("InMemoryFlagStorage ready")

    async def shutdown(self) -> None:
        """Alias for ``close()``."""
        await self.close()

    # ------------------------------------------------------------------
    # Convenience (non-interface) methods
    # ------------------------------------------------------------------

    async def invalidate(self, key: str) -> bool:
        """Remove a single flag from cache without deleting sub-entities.

        Used by the multi-layer cache to invalidate L1 entries without
        destroying rule/override/variant data (which will be re-populated
        from lower layers on the next miss).

        Args:
            key: The flag key to invalidate.

        Returns:
            True if the key was present and removed.
        """
        async with self._lock:
            return self._flags.pop(key, None) is not None

    async def size(self) -> int:
        """Return the current number of non-expired flag entries."""
        async with self._lock:
            self._evict_expired()
            return len(self._flags)
