# -*- coding: utf-8 -*-
"""
PII Allowlist Manager - SEC-011 PII Detection/Redaction Enhancements

Manages PII detection allowlists with support for per-tenant configuration,
pattern caching, and persistence. Provides the core logic for determining
whether a detected PII value should be excluded from enforcement.

Features:
    - Multiple pattern types (regex, exact, prefix, suffix, contains)
    - Compiled regex caching for performance
    - Per-tenant and global allowlists
    - Expiration handling
    - PostgreSQL persistence (optional)
    - Prometheus metrics integration

Usage:
    >>> from greenlang.infrastructure.pii_service.allowlist import AllowlistManager
    >>> manager = AllowlistManager(config)
    >>> await manager.initialize()
    >>> is_allowed, entry = await manager.is_allowed("test@example.com", PIIType.EMAIL)
    >>> if is_allowed:
    ...     print(f"Allowed: {entry.reason}")

Author: GreenLang Framework Team
Date: February 2026
PRD: SEC-011 PII Detection/Redaction Enhancements
Status: Production Ready
"""

from __future__ import annotations

import hashlib
import logging
import re
from datetime import datetime
from typing import Any, Dict, List, Optional, Pattern, Tuple
from uuid import UUID

from pydantic import BaseModel, Field

# Import PIIType from existing PII scanner
from greenlang.infrastructure.security_scanning.pii_scanner import PIIType

from greenlang.infrastructure.pii_service.allowlist.patterns import (
    AllowlistEntry,
    DEFAULT_ALLOWLISTS,
    PatternType,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


class AllowlistConfig(BaseModel):
    """Configuration for the AllowlistManager.

    Attributes:
        enable_defaults: Whether to load default allowlists on init.
        enable_persistence: Whether to persist entries to database.
        cache_compiled_patterns: Whether to cache compiled regex patterns.
        max_entries_per_tenant: Maximum allowlist entries per tenant.
        max_global_entries: Maximum global allowlist entries.
        pattern_compile_timeout_ms: Timeout for regex compilation.
        enable_metrics: Whether to emit Prometheus metrics.
    """

    enable_defaults: bool = Field(
        default=True,
        description="Load default allowlists on initialization"
    )
    enable_persistence: bool = Field(
        default=True,
        description="Persist entries to database"
    )
    cache_compiled_patterns: bool = Field(
        default=True,
        description="Cache compiled regex patterns"
    )
    max_entries_per_tenant: int = Field(
        default=1000,
        ge=10,
        le=100000,
        description="Maximum entries per tenant"
    )
    max_global_entries: int = Field(
        default=500,
        ge=10,
        le=10000,
        description="Maximum global entries"
    )
    pattern_compile_timeout_ms: int = Field(
        default=100,
        ge=10,
        le=1000,
        description="Regex compilation timeout in milliseconds"
    )
    enable_metrics: bool = Field(
        default=True,
        description="Emit Prometheus metrics"
    )

    model_config = {"frozen": True}


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class AllowlistError(Exception):
    """Base exception for allowlist operations."""

    pass


class InvalidPatternError(AllowlistError):
    """Raised when a pattern is invalid."""

    def __init__(self, pattern: str, reason: str):
        self.pattern = pattern
        self.reason = reason
        super().__init__(f"Invalid pattern '{pattern}': {reason}")


class EntryNotFoundError(AllowlistError):
    """Raised when an allowlist entry is not found."""

    def __init__(self, entry_id: UUID):
        self.entry_id = entry_id
        super().__init__(f"Allowlist entry not found: {entry_id}")


class EntryLimitExceededError(AllowlistError):
    """Raised when entry limit is exceeded."""

    def __init__(self, limit: int, scope: str):
        self.limit = limit
        self.scope = scope
        super().__init__(f"Entry limit ({limit}) exceeded for {scope}")


# ---------------------------------------------------------------------------
# Metrics (lazy initialization)
# ---------------------------------------------------------------------------

_metrics_initialized = False
_pii_allowlist_matches_total = None
_pii_allowlist_checks_total = None
_pii_allowlist_entries_total = None


def _init_metrics() -> None:
    """Initialize Prometheus metrics lazily."""
    global _metrics_initialized, _pii_allowlist_matches_total
    global _pii_allowlist_checks_total, _pii_allowlist_entries_total

    if _metrics_initialized:
        return

    try:
        from prometheus_client import Counter, Gauge

        _pii_allowlist_matches_total = Counter(
            "gl_pii_allowlist_matches_total",
            "Total allowlist matches (false positives avoided)",
            ["pii_type", "pattern_type", "tenant_id"]
        )
        _pii_allowlist_checks_total = Counter(
            "gl_pii_allowlist_checks_total",
            "Total allowlist checks performed",
            ["pii_type", "tenant_id"]
        )
        _pii_allowlist_entries_total = Gauge(
            "gl_pii_allowlist_entries_total",
            "Current number of allowlist entries",
            ["pii_type", "scope"]
        )
        _metrics_initialized = True
    except ImportError:
        logger.debug("prometheus_client not available, metrics disabled")
        _metrics_initialized = True


# ---------------------------------------------------------------------------
# Allowlist Manager
# ---------------------------------------------------------------------------


class AllowlistManager:
    """Manage PII detection allowlists.

    Provides pattern-based allowlisting for PII detection with support for
    multiple pattern types, per-tenant configuration, and persistence.

    Attributes:
        config: AllowlistManager configuration.

    Example:
        >>> config = AllowlistConfig(enable_defaults=True)
        >>> manager = AllowlistManager(config)
        >>> await manager.initialize()
        >>> allowed, entry = await manager.is_allowed("test@example.com", PIIType.EMAIL)
    """

    def __init__(
        self,
        config: Optional[AllowlistConfig] = None,
        db_pool: Optional[Any] = None,
    ) -> None:
        """Initialize AllowlistManager.

        Args:
            config: Manager configuration.
            db_pool: Optional database pool for persistence.
        """
        self._config = config or AllowlistConfig()
        self._db_pool = db_pool
        self._entries: Dict[PIIType, List[AllowlistEntry]] = {}
        self._compiled_patterns: Dict[str, Pattern] = {}
        self._initialized = False

        if self._config.enable_metrics:
            _init_metrics()

    async def initialize(self) -> None:
        """Initialize the allowlist manager.

        Loads default allowlists and persisted entries.
        """
        if self._initialized:
            return

        logger.info("Initializing AllowlistManager")

        # Load defaults
        if self._config.enable_defaults:
            self._load_defaults()

        # Load persisted entries
        if self._config.enable_persistence and self._db_pool:
            await self._load_from_database()

        self._initialized = True
        logger.info(
            "AllowlistManager initialized with %d entries",
            self._get_total_entry_count()
        )

    async def close(self) -> None:
        """Close the allowlist manager and clean up resources."""
        self._compiled_patterns.clear()
        self._entries.clear()
        self._initialized = False
        logger.info("AllowlistManager closed")

    # -------------------------------------------------------------------------
    # Core Operations
    # -------------------------------------------------------------------------

    async def is_allowed(
        self,
        value: str,
        pii_type: PIIType,
        tenant_id: Optional[str] = None,
    ) -> Tuple[bool, Optional[AllowlistEntry]]:
        """Check if a value is in the allowlist.

        Checks both global and tenant-specific allowlists for a match.

        Args:
            value: The PII value to check.
            pii_type: Type of PII.
            tenant_id: Optional tenant for tenant-specific allowlists.

        Returns:
            Tuple of (is_allowed, matching_entry).
            If allowed, matching_entry contains the matched AllowlistEntry.
            If not allowed, matching_entry is None.
        """
        if not self._initialized:
            await self.initialize()

        # Record check metric
        if _pii_allowlist_checks_total:
            _pii_allowlist_checks_total.labels(
                pii_type=pii_type.value,
                tenant_id=tenant_id or "global"
            ).inc()

        # Get entries for this PII type
        entries = await self._get_entries(pii_type, tenant_id)

        for entry in entries:
            # Skip inactive entries
            if not entry.is_active():
                continue

            # Check tenant match
            if not entry.matches_tenant(tenant_id):
                continue

            # Check pattern match
            if self._matches(value, entry):
                # Record match metric
                if _pii_allowlist_matches_total:
                    _pii_allowlist_matches_total.labels(
                        pii_type=pii_type.value,
                        pattern_type=entry.pattern_type.value,
                        tenant_id=tenant_id or "global"
                    ).inc()

                logger.debug(
                    "Allowlist match: value_hash=%s, pii_type=%s, reason=%s",
                    hashlib.sha256(value.encode()).hexdigest()[:8],
                    pii_type.value,
                    entry.reason
                )
                return True, entry

        return False, None

    async def add_entry(self, entry: AllowlistEntry) -> AllowlistEntry:
        """Add a new allowlist entry.

        Args:
            entry: The entry to add.

        Returns:
            The added entry.

        Raises:
            InvalidPatternError: If the pattern is invalid.
            EntryLimitExceededError: If entry limit is exceeded.
        """
        if not self._initialized:
            await self.initialize()

        # Validate regex pattern
        if entry.pattern_type == PatternType.REGEX:
            try:
                re.compile(entry.pattern)
            except re.error as e:
                raise InvalidPatternError(entry.pattern, str(e))

        # Check entry limits
        await self._check_entry_limits(entry)

        # Add to in-memory store
        if entry.pii_type not in self._entries:
            self._entries[entry.pii_type] = []
        self._entries[entry.pii_type].append(entry)

        # Persist to database
        if self._config.enable_persistence and self._db_pool:
            await self._persist_entry(entry)

        # Update metrics
        self._update_entry_metrics(entry.pii_type)

        logger.info(
            "Added allowlist entry: id=%s, pii_type=%s, pattern_type=%s",
            entry.id,
            entry.pii_type.value,
            entry.pattern_type.value
        )

        return entry

    async def remove_entry(self, entry_id: UUID) -> bool:
        """Remove an allowlist entry.

        Args:
            entry_id: ID of the entry to remove.

        Returns:
            True if removed, False if not found.

        Raises:
            EntryNotFoundError: If entry not found.
        """
        if not self._initialized:
            await self.initialize()

        # Find and remove entry
        for pii_type, entries in self._entries.items():
            for i, entry in enumerate(entries):
                if entry.id == entry_id:
                    del entries[i]

                    # Remove from compiled cache
                    cache_key = str(entry_id)
                    if cache_key in self._compiled_patterns:
                        del self._compiled_patterns[cache_key]

                    # Remove from database
                    if self._config.enable_persistence and self._db_pool:
                        await self._delete_entry_from_db(entry_id)

                    # Update metrics
                    self._update_entry_metrics(pii_type)

                    logger.info("Removed allowlist entry: id=%s", entry_id)
                    return True

        raise EntryNotFoundError(entry_id)

    async def update_entry(
        self,
        entry_id: UUID,
        updates: Dict[str, Any]
    ) -> AllowlistEntry:
        """Update an existing allowlist entry.

        Args:
            entry_id: ID of the entry to update.
            updates: Dictionary of fields to update.

        Returns:
            The updated entry.

        Raises:
            EntryNotFoundError: If entry not found.
            InvalidPatternError: If updated pattern is invalid.
        """
        if not self._initialized:
            await self.initialize()

        # Find entry
        for entries in self._entries.values():
            for i, entry in enumerate(entries):
                if entry.id == entry_id:
                    # Validate pattern if being updated
                    if "pattern" in updates:
                        pattern_type = updates.get(
                            "pattern_type",
                            entry.pattern_type
                        )
                        if pattern_type == PatternType.REGEX:
                            try:
                                re.compile(updates["pattern"])
                            except re.error as e:
                                raise InvalidPatternError(updates["pattern"], str(e))

                    # Create updated entry
                    entry_dict = entry.model_dump()
                    entry_dict.update(updates)
                    updated_entry = AllowlistEntry(**entry_dict)
                    entries[i] = updated_entry

                    # Invalidate compiled pattern cache
                    cache_key = str(entry_id)
                    if cache_key in self._compiled_patterns:
                        del self._compiled_patterns[cache_key]

                    # Persist update
                    if self._config.enable_persistence and self._db_pool:
                        await self._update_entry_in_db(updated_entry)

                    logger.info("Updated allowlist entry: id=%s", entry_id)
                    return updated_entry

        raise EntryNotFoundError(entry_id)

    async def list_entries(
        self,
        pii_type: Optional[PIIType] = None,
        tenant_id: Optional[str] = None,
        include_expired: bool = False,
        include_disabled: bool = False,
    ) -> List[AllowlistEntry]:
        """List allowlist entries.

        Args:
            pii_type: Filter by PII type (None for all).
            tenant_id: Filter by tenant (None for global only).
            include_expired: Include expired entries.
            include_disabled: Include disabled entries.

        Returns:
            List of matching AllowlistEntry objects.
        """
        if not self._initialized:
            await self.initialize()

        results: List[AllowlistEntry] = []

        # Determine which PII types to check
        pii_types = [pii_type] if pii_type else list(self._entries.keys())

        for pt in pii_types:
            entries = self._entries.get(pt, [])
            for entry in entries:
                # Filter by tenant
                if tenant_id is not None and not entry.matches_tenant(tenant_id):
                    continue

                # Filter by expiration
                if not include_expired and entry.is_expired():
                    continue

                # Filter by enabled status
                if not include_disabled and not entry.enabled:
                    continue

                results.append(entry)

        return results

    async def get_entry(self, entry_id: UUID) -> AllowlistEntry:
        """Get a specific allowlist entry by ID.

        Args:
            entry_id: ID of the entry.

        Returns:
            The AllowlistEntry.

        Raises:
            EntryNotFoundError: If entry not found.
        """
        if not self._initialized:
            await self.initialize()

        for entries in self._entries.values():
            for entry in entries:
                if entry.id == entry_id:
                    return entry

        raise EntryNotFoundError(entry_id)

    # -------------------------------------------------------------------------
    # Pattern Matching
    # -------------------------------------------------------------------------

    def _matches(self, value: str, entry: AllowlistEntry) -> bool:
        """Check if a value matches an allowlist entry.

        Args:
            value: The value to check.
            entry: The allowlist entry to match against.

        Returns:
            True if the value matches the entry's pattern.
        """
        if entry.pattern_type == PatternType.EXACT:
            return value == entry.pattern

        elif entry.pattern_type == PatternType.REGEX:
            pattern = self._get_compiled_pattern(entry.id, entry.pattern)
            return bool(pattern.match(value))

        elif entry.pattern_type == PatternType.PREFIX:
            return value.startswith(entry.pattern)

        elif entry.pattern_type == PatternType.SUFFIX:
            return value.endswith(entry.pattern)

        elif entry.pattern_type == PatternType.CONTAINS:
            return entry.pattern in value

        return False

    def _get_compiled_pattern(
        self,
        entry_id: UUID,
        pattern: str
    ) -> Pattern:
        """Get or compile a regex pattern with caching.

        Args:
            entry_id: ID of the entry (for cache key).
            pattern: The regex pattern string.

        Returns:
            Compiled regex Pattern object.
        """
        if not self._config.cache_compiled_patterns:
            return re.compile(pattern, re.IGNORECASE)

        cache_key = str(entry_id)
        if cache_key not in self._compiled_patterns:
            self._compiled_patterns[cache_key] = re.compile(
                pattern,
                re.IGNORECASE
            )

        return self._compiled_patterns[cache_key]

    # -------------------------------------------------------------------------
    # Internal Methods
    # -------------------------------------------------------------------------

    def _load_defaults(self) -> None:
        """Load default allowlist entries."""
        for pii_type, entries in DEFAULT_ALLOWLISTS.items():
            if pii_type not in self._entries:
                self._entries[pii_type] = []

            # Deep copy to avoid modifying the defaults
            for entry in entries:
                self._entries[pii_type].append(
                    AllowlistEntry(**entry.model_dump())
                )

        logger.debug(
            "Loaded %d default allowlist entries",
            self._get_total_entry_count()
        )

    async def _load_from_database(self) -> None:
        """Load allowlist entries from database."""
        if not self._db_pool:
            return

        try:
            async with self._db_pool.acquire() as conn:
                rows = await conn.fetch("""
                    SELECT id, pii_type, pattern, pattern_type, reason,
                           created_by, created_at, expires_at, tenant_id,
                           enabled, metadata
                    FROM pii_service.allowlist
                    WHERE enabled = true
                    ORDER BY created_at ASC
                """)

                for row in rows:
                    entry = AllowlistEntry(
                        id=row["id"],
                        pii_type=PIIType(row["pii_type"]),
                        pattern=row["pattern"],
                        pattern_type=PatternType(row["pattern_type"]),
                        reason=row["reason"],
                        created_by=row["created_by"],
                        created_at=row["created_at"],
                        expires_at=row["expires_at"],
                        tenant_id=row["tenant_id"],
                        enabled=row["enabled"],
                        metadata=row["metadata"] or {},
                    )

                    if entry.pii_type not in self._entries:
                        self._entries[entry.pii_type] = []
                    self._entries[entry.pii_type].append(entry)

                logger.info("Loaded %d entries from database", len(rows))

        except Exception as e:
            logger.error("Failed to load allowlist from database: %s", e)

    async def _persist_entry(self, entry: AllowlistEntry) -> None:
        """Persist an entry to the database."""
        if not self._db_pool:
            return

        try:
            async with self._db_pool.acquire() as conn:
                await conn.execute("""
                    INSERT INTO pii_service.allowlist
                    (id, pii_type, pattern, pattern_type, reason, created_by,
                     created_at, expires_at, tenant_id, enabled, metadata)
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)
                    ON CONFLICT (id) DO UPDATE SET
                        pattern = EXCLUDED.pattern,
                        pattern_type = EXCLUDED.pattern_type,
                        reason = EXCLUDED.reason,
                        expires_at = EXCLUDED.expires_at,
                        enabled = EXCLUDED.enabled,
                        metadata = EXCLUDED.metadata
                """,
                    entry.id,
                    entry.pii_type.value,
                    entry.pattern,
                    entry.pattern_type.value,
                    entry.reason,
                    entry.created_by,
                    entry.created_at,
                    entry.expires_at,
                    entry.tenant_id,
                    entry.enabled,
                    entry.metadata,
                )
        except Exception as e:
            logger.error("Failed to persist allowlist entry: %s", e)

    async def _update_entry_in_db(self, entry: AllowlistEntry) -> None:
        """Update an entry in the database."""
        await self._persist_entry(entry)

    async def _delete_entry_from_db(self, entry_id: UUID) -> None:
        """Delete an entry from the database."""
        if not self._db_pool:
            return

        try:
            async with self._db_pool.acquire() as conn:
                await conn.execute(
                    "DELETE FROM pii_service.allowlist WHERE id = $1",
                    entry_id
                )
        except Exception as e:
            logger.error("Failed to delete allowlist entry: %s", e)

    async def _get_entries(
        self,
        pii_type: PIIType,
        tenant_id: Optional[str]
    ) -> List[AllowlistEntry]:
        """Get all entries for a PII type and tenant.

        Returns global entries plus tenant-specific entries.

        Args:
            pii_type: The PII type.
            tenant_id: The tenant ID (can be None).

        Returns:
            List of applicable AllowlistEntry objects.
        """
        entries = self._entries.get(pii_type, [])

        # Filter to global + tenant-specific
        return [
            entry for entry in entries
            if entry.matches_tenant(tenant_id)
        ]

    async def _check_entry_limits(self, entry: AllowlistEntry) -> None:
        """Check if adding an entry would exceed limits.

        Args:
            entry: The entry to be added.

        Raises:
            EntryLimitExceededError: If limit would be exceeded.
        """
        if entry.tenant_id is None:
            # Global entry
            global_count = sum(
                1 for entries in self._entries.values()
                for e in entries if e.tenant_id is None
            )
            if global_count >= self._config.max_global_entries:
                raise EntryLimitExceededError(
                    self._config.max_global_entries,
                    "global"
                )
        else:
            # Tenant-specific entry
            tenant_count = sum(
                1 for entries in self._entries.values()
                for e in entries if e.tenant_id == entry.tenant_id
            )
            if tenant_count >= self._config.max_entries_per_tenant:
                raise EntryLimitExceededError(
                    self._config.max_entries_per_tenant,
                    f"tenant:{entry.tenant_id}"
                )

    def _get_total_entry_count(self) -> int:
        """Get total number of entries."""
        return sum(len(entries) for entries in self._entries.values())

    def _update_entry_metrics(self, pii_type: PIIType) -> None:
        """Update entry count metrics for a PII type."""
        if not _pii_allowlist_entries_total:
            return

        entries = self._entries.get(pii_type, [])
        global_count = sum(1 for e in entries if e.tenant_id is None)
        tenant_count = sum(1 for e in entries if e.tenant_id is not None)

        _pii_allowlist_entries_total.labels(
            pii_type=pii_type.value,
            scope="global"
        ).set(global_count)

        _pii_allowlist_entries_total.labels(
            pii_type=pii_type.value,
            scope="tenant"
        ).set(tenant_count)


# ---------------------------------------------------------------------------
# Factory Function
# ---------------------------------------------------------------------------

_global_manager: Optional[AllowlistManager] = None


def get_allowlist_manager(
    config: Optional[AllowlistConfig] = None,
    db_pool: Optional[Any] = None,
) -> AllowlistManager:
    """Get or create the global AllowlistManager instance.

    Args:
        config: Optional configuration.
        db_pool: Optional database pool.

    Returns:
        The AllowlistManager instance.
    """
    global _global_manager

    if _global_manager is None:
        _global_manager = AllowlistManager(config, db_pool)

    return _global_manager


def reset_allowlist_manager() -> None:
    """Reset the global AllowlistManager instance."""
    global _global_manager
    _global_manager = None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

__all__ = [
    "AllowlistConfig",
    "AllowlistManager",
    "AllowlistError",
    "InvalidPatternError",
    "EntryNotFoundError",
    "EntryLimitExceededError",
    "get_allowlist_manager",
    "reset_allowlist_manager",
]
