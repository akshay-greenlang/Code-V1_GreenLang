"""
GreenLang Cache Invalidation Strategies

Comprehensive invalidation strategies for maintaining cache coherence and freshness.

Strategies:
- TTL-based: Automatic expiration
- LRU eviction: Memory pressure based
- Event-based: Triggered by data updates
- Pattern-based: Bulk invalidation by key patterns
- Version-based: Version mismatch detection

Author: GreenLang Infrastructure Team (TEAM 2)
Date: 2025-11-08
Version: 5.0.0
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set
from collections import defaultdict

logger = logging.getLogger(__name__)


class InvalidationEvent(Enum):
    """Types of invalidation events."""
    WORKFLOW_UPDATED = "workflow_updated"
    WORKFLOW_DELETED = "workflow_deleted"
    AGENT_UPDATED = "agent_updated"
    AGENT_DELETED = "agent_deleted"
    CONFIG_CHANGED = "config_changed"
    DATA_UPDATED = "data_updated"
    USER_UPDATED = "user_updated"
    MANUAL = "manual"


@dataclass
class InvalidationRule:
    """
    Rule for automatic cache invalidation.

    Attributes:
        event: Event that triggers invalidation
        key_pattern: Pattern of keys to invalidate
        namespace: Optional namespace
        delay_seconds: Delay before invalidation (for batching)
        condition: Optional condition function
    """
    event: InvalidationEvent
    key_pattern: str
    namespace: Optional[str] = None
    delay_seconds: float = 0.0
    condition: Optional[Callable[[Dict[str, Any]], bool]] = None


@dataclass
class VersionedCacheEntry:
    """
    Cache entry with version tracking.

    Attributes:
        value: Cached value
        version: Version identifier
        created_at: Creation time
    """
    value: Any
    version: str
    created_at: float


class TTLInvalidationManager:
    """
    Manages TTL-based cache invalidation.

    Tracks expiration times and performs background cleanup.
    """

    def __init__(self, cleanup_interval_seconds: int = 60):
        """
        Initialize TTL invalidation manager.

        Args:
            cleanup_interval_seconds: Interval for cleanup task
        """
        self._cleanup_interval = cleanup_interval_seconds
        self._expiry_map: Dict[str, float] = {}  # key -> expiry_time
        self._cleanup_task: Optional[asyncio.Task] = None
        self._running = False
        self._invalidation_callback: Optional[Callable] = None

    async def start(self) -> None:
        """Start background cleanup task."""
        self._running = True
        self._cleanup_task = asyncio.create_task(self._cleanup_loop())
        logger.info("TTL invalidation manager started")

    async def stop(self) -> None:
        """Stop background cleanup task."""
        self._running = False
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass

    def register_entry(self, key: str, ttl_seconds: int) -> None:
        """
        Register cache entry with TTL.

        Args:
            key: Cache key
            ttl_seconds: Time to live in seconds
        """
        expiry_time = time.time() + ttl_seconds
        self._expiry_map[key] = expiry_time

    def unregister_entry(self, key: str) -> None:
        """Unregister cache entry."""
        self._expiry_map.pop(key, None)

    def set_invalidation_callback(
        self,
        callback: Callable[[List[str]], None]
    ) -> None:
        """
        Set callback for invalidation.

        Args:
            callback: Async function to call with expired keys
        """
        self._invalidation_callback = callback

    async def _cleanup_loop(self) -> None:
        """Background task to clean up expired entries."""
        while self._running:
            try:
                await asyncio.sleep(self._cleanup_interval)
                await self._cleanup_expired()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in TTL cleanup loop: {e}", exc_info=True)

    async def _cleanup_expired(self) -> None:
        """Remove expired entries."""
        now = time.time()
        expired_keys = [
            key for key, expiry in self._expiry_map.items()
            if expiry <= now
        ]

        if expired_keys:
            # Remove from tracking
            for key in expired_keys:
                del self._expiry_map[key]

            # Notify callback
            if self._invalidation_callback:
                try:
                    await self._invalidation_callback(expired_keys)
                except Exception as e:
                    logger.error(f"Error in invalidation callback: {e}")

            logger.debug(f"Cleaned up {len(expired_keys)} expired entries")


class EventBasedInvalidationManager:
    """
    Manages event-based cache invalidation.

    Listens for data update events and invalidates related cache entries.
    """

    def __init__(self):
        """Initialize event-based invalidation manager."""
        self._rules: List[InvalidationRule] = []
        self._pending_invalidations: Dict[str, Set[str]] = defaultdict(set)
        self._invalidation_callback: Optional[Callable] = None
        self._flush_task: Optional[asyncio.Task] = None
        self._running = False

    async def start(self) -> None:
        """Start invalidation manager."""
        self._running = True
        self._flush_task = asyncio.create_task(self._flush_loop())
        logger.info("Event-based invalidation manager started")

    async def stop(self) -> None:
        """Stop invalidation manager."""
        self._running = False
        if self._flush_task:
            self._flush_task.cancel()
            try:
                await self._flush_task
            except asyncio.CancelledError:
                pass

    def add_rule(self, rule: InvalidationRule) -> None:
        """
        Add invalidation rule.

        Args:
            rule: InvalidationRule to add
        """
        self._rules.append(rule)
        logger.info(f"Added invalidation rule: {rule.event.value} -> {rule.key_pattern}")

    def remove_rule(self, event: InvalidationEvent, key_pattern: str) -> None:
        """
        Remove invalidation rule.

        Args:
            event: Event type
            key_pattern: Key pattern
        """
        self._rules = [
            r for r in self._rules
            if not (r.event == event and r.key_pattern == key_pattern)
        ]

    def set_invalidation_callback(
        self,
        callback: Callable[[List[str]], None]
    ) -> None:
        """Set callback for invalidation."""
        self._invalidation_callback = callback

    async def trigger_event(
        self,
        event: InvalidationEvent,
        context: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Trigger invalidation event.

        Args:
            event: Event type
            context: Optional event context
        """
        context = context or {}

        # Find matching rules
        for rule in self._rules:
            if rule.event != event:
                continue

            # Check condition if present
            if rule.condition and not rule.condition(context):
                continue

            # Generate key pattern
            key_pattern = rule.key_pattern.format(**context)

            # Schedule invalidation
            if rule.delay_seconds > 0:
                # Batch for later
                delay_key = f"{rule.delay_seconds}"
                self._pending_invalidations[delay_key].add(key_pattern)
            else:
                # Immediate invalidation
                await self._invalidate_pattern(key_pattern)

    async def _invalidate_pattern(self, pattern: str) -> None:
        """Invalidate keys matching pattern."""
        if self._invalidation_callback:
            try:
                # For simplicity, pass pattern as-is
                # In production, you'd resolve pattern to actual keys
                await self._invalidation_callback([pattern])
            except Exception as e:
                logger.error(f"Error invalidating pattern {pattern}: {e}")

    async def _flush_loop(self) -> None:
        """Background task to flush pending invalidations."""
        while self._running:
            try:
                await asyncio.sleep(1.0)

                # Flush pending invalidations
                for delay_key, patterns in list(self._pending_invalidations.items()):
                    delay = float(delay_key)
                    # Simple delay check (in production, track scheduled time)
                    for pattern in patterns:
                        await self._invalidate_pattern(pattern)

                    # Clear after flush
                    del self._pending_invalidations[delay_key]

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in flush loop: {e}", exc_info=True)


class VersionBasedInvalidationManager:
    """
    Manages version-based cache invalidation.

    Tracks versions and invalidates on mismatch.
    """

    def __init__(self):
        """Initialize version-based invalidation manager."""
        self._versions: Dict[str, str] = {}  # resource_id -> version
        self._cache_versions: Dict[str, str] = {}  # cache_key -> version

    def set_resource_version(self, resource_id: str, version: str) -> None:
        """
        Set current version for a resource.

        Args:
            resource_id: Resource identifier
            version: Version string
        """
        old_version = self._versions.get(resource_id)
        self._versions[resource_id] = version

        if old_version and old_version != version:
            logger.info(
                f"Resource version updated: {resource_id} "
                f"{old_version} -> {version}"
            )

    def get_resource_version(self, resource_id: str) -> Optional[str]:
        """Get current version for resource."""
        return self._versions.get(resource_id)

    def set_cache_version(self, cache_key: str, version: str) -> None:
        """Set version for cached entry."""
        self._cache_versions[cache_key] = version

    def is_valid(self, cache_key: str, resource_id: str) -> bool:
        """
        Check if cached version matches current version.

        Args:
            cache_key: Cache key
            resource_id: Resource identifier

        Returns:
            True if versions match
        """
        cache_version = self._cache_versions.get(cache_key)
        current_version = self._versions.get(resource_id)

        if cache_version is None or current_version is None:
            return False

        return cache_version == current_version

    def invalidate_resource(self, resource_id: str) -> List[str]:
        """
        Invalidate all cache entries for a resource.

        Args:
            resource_id: Resource identifier

        Returns:
            List of invalidated cache keys
        """
        # Find all cache keys for this resource
        # (simplified - in production, maintain reverse mapping)
        invalidated = []
        resource_version = self._versions.get(resource_id)

        for cache_key, version in list(self._cache_versions.items()):
            # Simple heuristic: check if resource_id in key
            if resource_id in cache_key:
                if version != resource_version:
                    del self._cache_versions[cache_key]
                    invalidated.append(cache_key)

        return invalidated


class PatternBasedInvalidationManager:
    """
    Manages pattern-based bulk invalidation.

    Supports wildcard patterns for efficient bulk operations.
    """

    def __init__(self):
        """Initialize pattern-based invalidation manager."""
        self._key_registry: Set[str] = set()  # Track all keys
        self._namespace_keys: Dict[str, Set[str]] = defaultdict(set)

    def register_key(self, key: str, namespace: Optional[str] = None) -> None:
        """
        Register a cache key.

        Args:
            key: Cache key
            namespace: Optional namespace
        """
        self._key_registry.add(key)
        if namespace:
            self._namespace_keys[namespace].add(key)

    def unregister_key(self, key: str, namespace: Optional[str] = None) -> None:
        """Unregister a cache key."""
        self._key_registry.discard(key)
        if namespace:
            self._namespace_keys[namespace].discard(key)

    def find_matching_keys(self, pattern: str) -> List[str]:
        """
        Find all keys matching pattern.

        Args:
            pattern: Wildcard pattern (e.g., "workflow:*", "*:123:*")

        Returns:
            List of matching keys
        """
        import fnmatch

        matching = []
        for key in self._key_registry:
            if fnmatch.fnmatch(key, pattern):
                matching.append(key)

        return matching

    def find_namespace_keys(self, namespace: str) -> List[str]:
        """Get all keys in namespace."""
        return list(self._namespace_keys.get(namespace, set()))

    def clear_namespace(self, namespace: str) -> List[str]:
        """
        Clear all keys in namespace.

        Args:
            namespace: Namespace to clear

        Returns:
            List of cleared keys
        """
        keys = self.find_namespace_keys(namespace)
        self._namespace_keys.pop(namespace, None)

        for key in keys:
            self._key_registry.discard(key)

        return keys


class UnifiedInvalidationManager:
    """
    Unified manager combining all invalidation strategies.

    Provides a single interface for all invalidation operations.
    """

    def __init__(self):
        """Initialize unified invalidation manager."""
        self._ttl_manager = TTLInvalidationManager()
        self._event_manager = EventBasedInvalidationManager()
        self._version_manager = VersionBasedInvalidationManager()
        self._pattern_manager = PatternBasedInvalidationManager()

        self._invalidation_callback: Optional[Callable] = None

    async def start(self) -> None:
        """Start all invalidation managers."""
        await self._ttl_manager.start()
        await self._event_manager.start()

        # Set callbacks
        self._ttl_manager.set_invalidation_callback(self._handle_invalidation)
        self._event_manager.set_invalidation_callback(self._handle_invalidation)

        logger.info("Unified invalidation manager started")

    async def stop(self) -> None:
        """Stop all invalidation managers."""
        await self._ttl_manager.stop()
        await self._event_manager.stop()

    def set_invalidation_callback(
        self,
        callback: Callable[[List[str]], None]
    ) -> None:
        """Set main invalidation callback."""
        self._invalidation_callback = callback

    # TTL methods
    def register_ttl(self, key: str, ttl_seconds: int) -> None:
        """Register key with TTL."""
        self._ttl_manager.register_entry(key, ttl_seconds)

    def unregister_ttl(self, key: str) -> None:
        """Unregister key from TTL tracking."""
        self._ttl_manager.unregister_entry(key)

    # Event methods
    def add_event_rule(self, rule: InvalidationRule) -> None:
        """Add event-based invalidation rule."""
        self._event_manager.add_rule(rule)

    async def trigger_event(
        self,
        event: InvalidationEvent,
        context: Optional[Dict[str, Any]] = None
    ) -> None:
        """Trigger invalidation event."""
        await self._event_manager.trigger_event(event, context)

    # Version methods
    def set_version(self, resource_id: str, version: str) -> None:
        """Set resource version."""
        self._version_manager.set_resource_version(resource_id, version)

    def check_version(self, cache_key: str, resource_id: str) -> bool:
        """Check if cached version is valid."""
        return self._version_manager.is_valid(cache_key, resource_id)

    # Pattern methods
    def register_key(self, key: str, namespace: Optional[str] = None) -> None:
        """Register key for pattern matching."""
        self._pattern_manager.register_key(key, namespace)

    def find_pattern(self, pattern: str) -> List[str]:
        """Find keys matching pattern."""
        return self._pattern_manager.find_matching_keys(pattern)

    async def invalidate_pattern(self, pattern: str) -> int:
        """
        Invalidate all keys matching pattern.

        Args:
            pattern: Wildcard pattern

        Returns:
            Number of keys invalidated
        """
        keys = self.find_pattern(pattern)
        if keys:
            await self._handle_invalidation(keys)
        return len(keys)

    async def _handle_invalidation(self, keys: List[str]) -> None:
        """Handle invalidation from any source."""
        if self._invalidation_callback:
            try:
                await self._invalidation_callback(keys)
                logger.info(f"Invalidated {len(keys)} keys")
            except Exception as e:
                logger.error(f"Error in invalidation callback: {e}")

    def get_stats(self) -> Dict[str, Any]:
        """Get invalidation statistics."""
        return {
            "ttl_tracked_keys": len(self._ttl_manager._expiry_map),
            "event_rules": len(self._event_manager._rules),
            "registered_keys": len(self._pattern_manager._key_registry),
            "namespaces": len(self._pattern_manager._namespace_keys),
        }
