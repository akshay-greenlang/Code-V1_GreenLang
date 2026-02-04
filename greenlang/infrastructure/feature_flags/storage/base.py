# -*- coding: utf-8 -*-
"""
Feature Flag Storage Interface - INFRA-008

Defines the abstract storage interface (IFlagStorage) that all storage backends
must implement. This provides a clean abstraction boundary between the evaluation
engine and the persistence layer, enabling swappable backends (memory, Redis,
PostgreSQL, multi-layer) without changing evaluation logic.

All methods are asynchronous to support non-blocking I/O in the evaluation
hot path.

Example:
    >>> class MyStorage(IFlagStorage):
    ...     async def get_flag(self, key: str) -> Optional[FeatureFlag]:
    ...         return self._flags.get(key)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, List, Optional

from greenlang.infrastructure.feature_flags.models import (
    AuditLogEntry,
    FeatureFlag,
    FlagOverride,
    FlagRule,
    FlagStatus,
    FlagVariant,
)


class IFlagStorage(ABC):
    """Abstract interface for feature flag storage backends.

    Every storage backend (in-memory, Redis, PostgreSQL, multi-layer)
    implements this interface. The evaluation engine depends only on
    IFlagStorage, never on a concrete backend.

    All methods are async to allow non-blocking database and cache calls.
    Implementations must be safe for concurrent access from multiple
    asyncio tasks.
    """

    # ------------------------------------------------------------------
    # Flag CRUD
    # ------------------------------------------------------------------

    @abstractmethod
    async def get_flag(self, key: str) -> Optional[FeatureFlag]:
        """Retrieve a single feature flag by key.

        Args:
            key: Unique flag identifier.

        Returns:
            The FeatureFlag if found, or None.
        """

    @abstractmethod
    async def get_all_flags(
        self,
        status_filter: Optional[FlagStatus] = None,
        tag_filter: Optional[str] = None,
    ) -> List[FeatureFlag]:
        """Retrieve all feature flags, optionally filtered.

        Args:
            status_filter: If provided, only return flags with this status.
            tag_filter: If provided, only return flags containing this tag.

        Returns:
            List of matching FeatureFlag instances (may be empty).
        """

    @abstractmethod
    async def save_flag(self, flag: FeatureFlag) -> FeatureFlag:
        """Create or update a feature flag.

        Implementations should use ``flag.key`` as the unique identifier.
        If a flag with the same key exists it is overwritten (upsert).

        Args:
            flag: The FeatureFlag to persist.

        Returns:
            The persisted FeatureFlag (may have updated server-side fields).
        """

    @abstractmethod
    async def delete_flag(self, key: str) -> bool:
        """Delete a feature flag and its associated rules, overrides, and variants.

        Args:
            key: Unique flag identifier.

        Returns:
            True if the flag was found and deleted, False otherwise.
        """

    # ------------------------------------------------------------------
    # Rules
    # ------------------------------------------------------------------

    @abstractmethod
    async def get_rules(self, flag_key: str) -> List[FlagRule]:
        """Retrieve all targeting rules for a specific flag.

        Rules are returned sorted by priority (ascending -- lower number
        means higher priority).

        Args:
            flag_key: The flag key to get rules for.

        Returns:
            List of FlagRule instances, sorted by priority.
        """

    @abstractmethod
    async def save_rule(self, rule: FlagRule) -> FlagRule:
        """Create or update a targeting rule.

        Uses ``rule.rule_id`` as the unique identifier within the flag.

        Args:
            rule: The FlagRule to persist.

        Returns:
            The persisted FlagRule.
        """

    # ------------------------------------------------------------------
    # Overrides
    # ------------------------------------------------------------------

    @abstractmethod
    async def get_overrides(self, flag_key: str) -> List[FlagOverride]:
        """Retrieve all overrides for a specific flag.

        Args:
            flag_key: The flag key to get overrides for.

        Returns:
            List of FlagOverride instances for the given flag.
        """

    @abstractmethod
    async def save_override(self, override: FlagOverride) -> FlagOverride:
        """Create or update a flag override.

        Uses the combination of ``(flag_key, scope_type, scope_value)`` as
        the natural key for upsert semantics.

        Args:
            override: The FlagOverride to persist.

        Returns:
            The persisted FlagOverride.
        """

    # ------------------------------------------------------------------
    # Variants
    # ------------------------------------------------------------------

    @abstractmethod
    async def get_variants(self, flag_key: str) -> List[FlagVariant]:
        """Retrieve all variants for a specific flag.

        Args:
            flag_key: The flag key to get variants for.

        Returns:
            List of FlagVariant instances for the given flag.
        """

    @abstractmethod
    async def save_variant(self, variant: FlagVariant) -> FlagVariant:
        """Create or update a flag variant.

        Uses ``(flag_key, variant_key)`` as the natural key.

        Args:
            variant: The FlagVariant to persist.

        Returns:
            The persisted FlagVariant.
        """

    # ------------------------------------------------------------------
    # Audit Log
    # ------------------------------------------------------------------

    @abstractmethod
    async def log_audit(self, entry: AuditLogEntry) -> None:
        """Append an immutable audit log entry.

        Implementations must never modify or delete existing entries.

        Args:
            entry: The AuditLogEntry to record.
        """

    @abstractmethod
    async def get_audit_log(
        self,
        flag_key: str,
        limit: int = 50,
    ) -> List[AuditLogEntry]:
        """Retrieve audit log entries for a flag, most recent first.

        Args:
            flag_key: The flag key to query audit history for.
            limit: Maximum number of entries to return (default 50).

        Returns:
            List of AuditLogEntry instances in reverse chronological order.
        """

    # ------------------------------------------------------------------
    # Health / Lifecycle
    # ------------------------------------------------------------------

    async def health_check(self) -> Dict[str, object]:
        """Check the health of this storage backend.

        Returns:
            Dictionary with at least a ``"healthy"`` boolean key.
        """
        return {"healthy": True, "backend": self.__class__.__name__}

    async def close(self) -> None:
        """Release any resources held by this storage backend.

        Subclasses that manage connection pools, background tasks, or
        file handles must override this method.
        """
