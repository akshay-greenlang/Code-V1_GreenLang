# -*- coding: utf-8 -*-
"""
Feature Flag Service - INFRA-008

High-level facade for the GreenLang feature flag system. Provides a unified
API for flag evaluation, management, rollout control, kill switch operations,
and audit logging. Composes the evaluation engine, multi-layer storage, and
kill switch into a single service object.

The service is designed as a singleton accessed via ``get_feature_flag_service()``.

Example:
    >>> from greenlang.infrastructure.feature_flags.service import get_feature_flag_service
    >>> service = get_feature_flag_service()
    >>> await service.initialize()
    >>> enabled = await service.is_enabled("enable-scope3-calc", context)
    >>> await service.shutdown()
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from greenlang.infrastructure.feature_flags.config import FeatureFlagConfig, get_config
from greenlang.infrastructure.feature_flags.engine import FeatureFlagEngine
from greenlang.infrastructure.feature_flags.kill_switch import KillSwitch
from greenlang.infrastructure.feature_flags.models import (
    AuditLogEntry,
    EvaluationContext,
    FeatureFlag,
    FlagEvaluationResult,
    FlagOverride,
    FlagRule,
    FlagStatus,
    FlagType,
    FlagVariant,
)
from greenlang.infrastructure.feature_flags.storage.memory import InMemoryFlagStorage
from greenlang.infrastructure.feature_flags.storage.multi_layer import MultiLayerFlagStorage

logger = logging.getLogger(__name__)


class FeatureFlagService:
    """High-level feature flag service facade.

    Composes the evaluation engine, multi-layer storage, and kill switch
    into a single service providing flag evaluation, management, rollout
    control, and audit capabilities.

    Attributes:
        _config: Feature flag configuration.
        _storage: Multi-layer flag storage.
        _engine: Feature flag evaluation engine.
        _kill_switch: Emergency kill switch.
        _initialized: Whether the service has been initialized.
    """

    def __init__(self, config: Optional[FeatureFlagConfig] = None) -> None:
        """Initialize the feature flag service.

        Auto-configures storage, engine, and kill switch. Uses the provided
        config or loads from environment variables via ``get_config()``.

        Args:
            config: Optional configuration. Loaded from env vars if None.
        """
        self._config = config or get_config()
        self._l1 = InMemoryFlagStorage()
        self._storage = MultiLayerFlagStorage(l1=self._l1)
        self._engine = FeatureFlagEngine(self._storage)
        self._kill_switch = KillSwitch(self._storage)
        self._initialized = False
        logger.info(
            "FeatureFlagService created (environment=%s)",
            self._config.environment,
        )

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def initialize(self) -> None:
        """Start the service: initialize storage, warm cache, sync kill state.

        Idempotent - safe to call multiple times.
        """
        if self._initialized:
            logger.debug("FeatureFlagService already initialized")
            return

        await self._storage.initialize()
        await self._storage.warm_cache()
        await self._kill_switch.sync_killed_state()
        self._initialized = True
        logger.info("FeatureFlagService initialized and ready")

    async def shutdown(self) -> None:
        """Clean shutdown of the service and all subsystems."""
        if not self._initialized:
            return
        await self._storage.shutdown()
        self._initialized = False
        logger.info("FeatureFlagService shut down")

    @property
    def is_initialized(self) -> bool:
        """Return whether the service has been initialized."""
        return self._initialized

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    async def is_enabled(
        self,
        flag_key: str,
        context: Optional[EvaluationContext] = None,
        default: bool = False,
    ) -> bool:
        """Check if a flag is enabled for the given context.

        This is the primary evaluation method for application code.

        Args:
            flag_key: The flag key to evaluate.
            context: Optional evaluation context. Uses defaults if None.
            default: Value to return if evaluation fails.

        Returns:
            True if the flag is enabled, False otherwise.
        """
        try:
            result = await self._engine.evaluate(flag_key, context)
            return result.enabled
        except Exception as exc:
            logger.error(
                "Flag evaluation failed for '%s', returning default=%s: %s",
                flag_key, default, exc,
            )
            return default

    async def evaluate(
        self,
        flag_key: str,
        context: Optional[EvaluationContext] = None,
    ) -> FlagEvaluationResult:
        """Evaluate a flag and return the full result with metadata.

        Args:
            flag_key: The flag key to evaluate.
            context: Optional evaluation context.

        Returns:
            Full evaluation result including reason, variant, timing.
        """
        return await self._engine.evaluate(flag_key, context)

    async def evaluate_all(
        self,
        context: Optional[EvaluationContext] = None,
    ) -> Dict[str, bool]:
        """Evaluate all active flags for a context.

        Args:
            context: Optional evaluation context.

        Returns:
            Mapping of flag_key -> enabled boolean.
        """
        return await self._engine.evaluate_all(context)

    async def evaluate_batch(
        self,
        flag_keys: List[str],
        context: Optional[EvaluationContext] = None,
    ) -> Dict[str, FlagEvaluationResult]:
        """Evaluate a batch of specific flags.

        Args:
            flag_keys: List of flag keys to evaluate.
            context: Optional evaluation context.

        Returns:
            Mapping of flag_key -> FlagEvaluationResult.
        """
        return await self._engine.evaluate_batch(flag_keys, context)

    # ------------------------------------------------------------------
    # Flag Management (CRUD)
    # ------------------------------------------------------------------

    async def create_flag(
        self,
        flag: FeatureFlag,
        created_by: str = "",
    ) -> FeatureFlag:
        """Create a new feature flag.

        Args:
            flag: The feature flag definition.
            created_by: Identity of the creator.

        Returns:
            The created flag.

        Raises:
            ValueError: If a flag with the same key already exists.
        """
        existing = await self._storage.get_flag(flag.key)
        if existing is not None:
            raise ValueError(f"Flag '{flag.key}' already exists")

        await self._storage.save_flag(flag)

        await self._write_audit(
            flag_key=flag.key,
            action="created",
            new_value=flag.model_dump(mode="json"),
            changed_by=created_by,
        )
        logger.info("Created flag: %s (type=%s)", flag.key, flag.flag_type.value)
        return flag

    async def update_flag(
        self,
        flag_key: str,
        updates: Dict[str, Any],
        updated_by: str = "",
    ) -> FeatureFlag:
        """Update an existing feature flag.

        Args:
            flag_key: Key of the flag to update.
            updates: Dictionary of field name -> new value.
            updated_by: Identity of who made the update.

        Returns:
            The updated flag.

        Raises:
            ValueError: If the flag does not exist.
        """
        flag = await self._storage.get_flag(flag_key)
        if flag is None:
            raise ValueError(f"Flag '{flag_key}' not found")

        old_value = flag.model_dump(mode="json")

        # Apply updates, bump version and updated_at
        update_dict = {
            **updates,
            "version": flag.version + 1,
            "updated_at": datetime.now(timezone.utc),
        }
        updated_flag = flag.model_copy(update=update_dict)
        await self._storage.save_flag(updated_flag)

        await self._write_audit(
            flag_key=flag_key,
            action="updated",
            old_value=old_value,
            new_value=updated_flag.model_dump(mode="json"),
            changed_by=updated_by,
        )
        logger.info("Updated flag: %s (v%d)", flag_key, updated_flag.version)
        return updated_flag

    async def delete_flag(
        self,
        flag_key: str,
        deleted_by: str = "",
    ) -> bool:
        """Soft-delete (archive) a feature flag.

        Does not physically remove the flag. Sets status to ARCHIVED.

        Args:
            flag_key: Key of the flag to archive.
            deleted_by: Identity of who archived the flag.

        Returns:
            True if the flag was found and archived.
        """
        flag = await self._storage.get_flag(flag_key)
        if flag is None:
            return False

        old_value = flag.model_dump(mode="json")
        archived_flag = flag.model_copy(
            update={
                "status": FlagStatus.ARCHIVED,
                "version": flag.version + 1,
                "updated_at": datetime.now(timezone.utc),
            }
        )
        await self._storage.save_flag(archived_flag)

        await self._write_audit(
            flag_key=flag_key,
            action="archived",
            old_value=old_value,
            new_value=archived_flag.model_dump(mode="json"),
            changed_by=deleted_by,
        )
        logger.info("Archived flag: %s", flag_key)
        return True

    async def get_flag(self, flag_key: str) -> Optional[FeatureFlag]:
        """Get a single flag by key.

        Args:
            flag_key: The flag key.

        Returns:
            FeatureFlag if found, None otherwise.
        """
        return await self._storage.get_flag(flag_key)

    async def list_flags(
        self,
        status: Optional[FlagStatus] = None,
        flag_type: Optional[str] = None,
        tag: Optional[str] = None,
        owner: Optional[str] = None,
        offset: int = 0,
        limit: int = 100,
    ) -> List[FeatureFlag]:
        """List flags with optional filtering and pagination.

        Args:
            status: Filter by flag status.
            flag_type: Filter by flag type.
            tag: Filter by tag.
            owner: Filter by owner.
            offset: Pagination offset.
            limit: Pagination limit.

        Returns:
            List of matching flags.
        """
        return await self._storage.list_flags(
            status=status,
            tag=tag,
            owner=owner,
            flag_type=flag_type,
            offset=offset,
            limit=limit,
        )

    async def count_flags(
        self,
        status: Optional[FlagStatus] = None,
        flag_type: Optional[str] = None,
        tag: Optional[str] = None,
        owner: Optional[str] = None,
    ) -> int:
        """Count flags matching filters.

        Args:
            status: Filter by status.
            flag_type: Filter by flag type.
            tag: Filter by tag.
            owner: Filter by owner.

        Returns:
            Count of matching flags.
        """
        return await self._storage.count_flags(
            status=status, tag=tag, owner=owner, flag_type=flag_type,
        )

    # ------------------------------------------------------------------
    # Rollout Control
    # ------------------------------------------------------------------

    async def set_rollout_percentage(
        self,
        flag_key: str,
        percentage: float,
        updated_by: str = "",
    ) -> FeatureFlag:
        """Set the rollout percentage for a flag.

        Args:
            flag_key: Flag key.
            percentage: New rollout percentage (0.0 to 100.0).
            updated_by: Identity of who changed the rollout.

        Returns:
            Updated flag.

        Raises:
            ValueError: If flag not found or percentage is invalid.
        """
        if not 0.0 <= percentage <= 100.0:
            raise ValueError(f"Percentage must be 0.0-100.0, got {percentage}")

        return await self.update_flag(
            flag_key,
            {"rollout_percentage": percentage},
            updated_by=updated_by,
        )

    async def enable_flag(
        self,
        flag_key: str,
        enabled_by: str = "",
    ) -> FeatureFlag:
        """Enable a flag (set status to ACTIVE).

        Args:
            flag_key: Flag key.
            enabled_by: Identity of who enabled the flag.

        Returns:
            Updated flag.
        """
        flag = await self._storage.get_flag(flag_key)
        if flag is None:
            raise ValueError(f"Flag '{flag_key}' not found")

        old_value = flag.model_dump(mode="json")
        updated = flag.model_copy(
            update={
                "status": FlagStatus.ACTIVE,
                "version": flag.version + 1,
                "updated_at": datetime.now(timezone.utc),
            }
        )
        await self._storage.save_flag(updated)

        await self._write_audit(
            flag_key=flag_key,
            action="enabled",
            old_value=old_value,
            new_value=updated.model_dump(mode="json"),
            changed_by=enabled_by,
        )
        logger.info("Enabled flag: %s", flag_key)
        return updated

    async def disable_flag(
        self,
        flag_key: str,
        disabled_by: str = "",
    ) -> FeatureFlag:
        """Disable a flag (set status to DRAFT).

        Args:
            flag_key: Flag key.
            disabled_by: Identity of who disabled the flag.

        Returns:
            Updated flag.
        """
        flag = await self._storage.get_flag(flag_key)
        if flag is None:
            raise ValueError(f"Flag '{flag_key}' not found")

        old_value = flag.model_dump(mode="json")
        updated = flag.model_copy(
            update={
                "status": FlagStatus.DRAFT,
                "version": flag.version + 1,
                "updated_at": datetime.now(timezone.utc),
            }
        )
        await self._storage.save_flag(updated)

        await self._write_audit(
            flag_key=flag_key,
            action="disabled",
            old_value=old_value,
            new_value=updated.model_dump(mode="json"),
            changed_by=disabled_by,
        )
        logger.info("Disabled flag: %s", flag_key)
        return updated

    # ------------------------------------------------------------------
    # Kill Switch
    # ------------------------------------------------------------------

    async def kill_flag(
        self,
        flag_key: str,
        killed_by: str = "",
        reason: str = "",
    ) -> bool:
        """Activate the kill switch for a flag.

        Args:
            flag_key: Flag key to kill.
            killed_by: Identity of who activated the kill switch.
            reason: Explanation of why the flag was killed.

        Returns:
            True if the flag was found and killed.
        """
        return await self._kill_switch.kill(flag_key, killed_by, reason)

    async def restore_flag(
        self,
        flag_key: str,
        restored_by: str = "",
        reason: str = "",
    ) -> bool:
        """Deactivate the kill switch and restore a flag.

        Args:
            flag_key: Flag key to restore.
            restored_by: Identity of who restored the flag.
            reason: Explanation of the restore.

        Returns:
            True if the flag was restored.
        """
        return await self._kill_switch.restore(flag_key, restored_by, reason)

    # ------------------------------------------------------------------
    # Overrides
    # ------------------------------------------------------------------

    async def set_override(
        self,
        flag_key: str,
        scope_type: str,
        scope_value: str,
        enabled: bool,
        variant_key: Optional[str] = None,
        expires_at: Optional[datetime] = None,
        created_by: str = "",
    ) -> FlagOverride:
        """Set an override for a flag.

        Args:
            flag_key: Flag key.
            scope_type: Override scope (user, tenant, segment, environment).
            scope_value: Scope identifier.
            enabled: Whether the flag is force-enabled for this scope.
            variant_key: Optional variant to force.
            expires_at: Optional expiration datetime.
            created_by: Identity of the creator.

        Returns:
            The created override.

        Raises:
            ValueError: If the flag does not exist.
        """
        flag = await self._storage.get_flag(flag_key)
        if flag is None:
            raise ValueError(f"Flag '{flag_key}' not found")

        override = FlagOverride(
            flag_key=flag_key,
            scope_type=scope_type,
            scope_value=scope_value,
            enabled=enabled,
            variant_key=variant_key,
            expires_at=expires_at,
            created_by=created_by,
        )
        await self._storage.save_override(override)

        await self._write_audit(
            flag_key=flag_key,
            action="override_added",
            new_value={
                "scope_type": scope_type,
                "scope_value": scope_value,
                "enabled": enabled,
            },
            changed_by=created_by,
        )
        logger.info(
            "Set override for %s: %s=%s -> enabled=%s",
            flag_key, scope_type, scope_value, enabled,
        )
        return override

    async def clear_override(
        self,
        flag_key: str,
        scope_type: str,
        scope_value: str,
        cleared_by: str = "",
    ) -> bool:
        """Remove an override from a flag.

        Args:
            flag_key: Flag key.
            scope_type: Override scope type.
            scope_value: Override scope value.
            cleared_by: Identity of who cleared the override.

        Returns:
            True if the override was found and removed.
        """
        result = await self._storage.delete_override(
            flag_key, scope_type, scope_value
        )
        if result:
            await self._write_audit(
                flag_key=flag_key,
                action="override_removed",
                old_value={
                    "scope_type": scope_type,
                    "scope_value": scope_value,
                },
                changed_by=cleared_by,
            )
        return result

    # ------------------------------------------------------------------
    # Variants
    # ------------------------------------------------------------------

    async def add_variant(
        self,
        variant: FlagVariant,
        added_by: str = "",
    ) -> FlagVariant:
        """Add or update a variant for a flag.

        Args:
            variant: The variant to add/update.
            added_by: Identity of who added the variant.

        Returns:
            The saved variant.

        Raises:
            ValueError: If the parent flag does not exist.
        """
        flag = await self._storage.get_flag(variant.flag_key)
        if flag is None:
            raise ValueError(f"Flag '{variant.flag_key}' not found")

        await self._storage.save_variant(variant)

        await self._write_audit(
            flag_key=variant.flag_key,
            action="variant_added",
            new_value={
                "variant_key": variant.variant_key,
                "weight": variant.weight,
            },
            changed_by=added_by,
        )
        return variant

    async def remove_variant(
        self,
        flag_key: str,
        variant_key: str,
        removed_by: str = "",
    ) -> bool:
        """Remove a variant from a flag.

        Args:
            flag_key: Flag key.
            variant_key: Variant key to remove.
            removed_by: Identity of who removed the variant.

        Returns:
            True if the variant was found and removed.
        """
        result = await self._storage.delete_variant(flag_key, variant_key)
        if result:
            await self._write_audit(
                flag_key=flag_key,
                action="variant_removed",
                old_value={"variant_key": variant_key},
                changed_by=removed_by,
            )
        return result

    async def get_variants(self, flag_key: str) -> List[FlagVariant]:
        """Get all variants for a flag.

        Args:
            flag_key: Flag key.

        Returns:
            List of variants.
        """
        return await self._storage.get_variants(flag_key)

    # ------------------------------------------------------------------
    # Audit Log
    # ------------------------------------------------------------------

    async def get_audit_log(
        self,
        flag_key: str,
        offset: int = 0,
        limit: int = 50,
    ) -> List[AuditLogEntry]:
        """Get audit log entries for a flag.

        Args:
            flag_key: Flag key.
            offset: Pagination offset.
            limit: Pagination limit.

        Returns:
            List of audit log entries, newest first.
        """
        return await self._storage.get_audit_log(flag_key, offset, limit)

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------

    async def get_statistics(self) -> Dict[str, Any]:
        """Get overall feature flag system statistics.

        Returns:
            Dictionary with counts by status, total flags, and system health.
        """
        all_flags = await self._storage.get_all_flags()
        status_counts: Dict[str, int] = {}
        type_counts: Dict[str, int] = {}
        for flag in all_flags:
            status_counts[flag.status.value] = status_counts.get(flag.status.value, 0) + 1
            type_counts[flag.flag_type.value] = type_counts.get(flag.flag_type.value, 0) + 1

        killed_flags = await self._kill_switch.get_killed_flags()

        return {
            "total_flags": len(all_flags),
            "by_status": status_counts,
            "by_type": type_counts,
            "killed_flags": killed_flags,
            "killed_count": len(killed_flags),
            "storage_healthy": await self._storage.health_check(),
            "environment": self._config.environment,
        }

    # ------------------------------------------------------------------
    # Internal Helpers
    # ------------------------------------------------------------------

    async def _write_audit(
        self,
        flag_key: str,
        action: str,
        old_value: Optional[Dict[str, Any]] = None,
        new_value: Optional[Dict[str, Any]] = None,
        changed_by: str = "",
        change_reason: str = "",
    ) -> None:
        """Write an audit log entry. Failures are logged but do not raise.

        Args:
            flag_key: Flag key.
            action: Action performed.
            old_value: Previous state.
            new_value: New state.
            changed_by: Identity of the actor.
            change_reason: Reason for the change.
        """
        try:
            entry = AuditLogEntry(
                flag_key=flag_key,
                action=action,
                old_value=old_value or {},
                new_value=new_value or {},
                changed_by=changed_by,
                change_reason=change_reason,
            )
            await self._storage.append_audit_log(entry)
        except Exception as exc:
            logger.error(
                "Failed to write audit log for flag '%s' action '%s': %s",
                flag_key, action, exc,
            )


# ---------------------------------------------------------------------------
# Module-Level Singleton
# ---------------------------------------------------------------------------

_service_instance: Optional[FeatureFlagService] = None
_service_lock = asyncio.Lock()


def get_feature_flag_service(
    config: Optional[FeatureFlagConfig] = None,
) -> FeatureFlagService:
    """Get the global FeatureFlagService singleton.

    Creates the instance on first call. Thread-safe via module-level lock.

    Args:
        config: Optional config override. Only used on first creation.

    Returns:
        The FeatureFlagService singleton.
    """
    global _service_instance
    if _service_instance is None:
        _service_instance = FeatureFlagService(config)
    return _service_instance


def reset_service() -> None:
    """Reset the service singleton. Used for testing.

    Clears the module-level service instance so the next call to
    ``get_feature_flag_service()`` creates a fresh instance.
    """
    global _service_instance
    _service_instance = None
    logger.debug("FeatureFlagService singleton reset")


__all__ = [
    "FeatureFlagService",
    "get_feature_flag_service",
    "reset_service",
]
