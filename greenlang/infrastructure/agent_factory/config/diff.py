"""
Config Diff - Agent Factory Config (INFRA-010)

Computes structured diffs between two agent configurations. Identifies
added, removed, and modified fields. Detects breaking changes that could
impact agent behaviour (e.g., resource limits decreased, timeout shortened).
Supports generating reverse diffs for rollback.

Classes:
    - ChangeType: Enumeration of change types.
    - ConfigChange: Single field-level change.
    - ConfigDiff: Structured diff between two configurations.

Example:
    >>> diff = ConfigDiff.compare(old_config, new_config)
    >>> for change in diff.changes:
    ...     print(f"{change.field_path}: {change.old_value} -> {change.new_value}")
    >>> if diff.has_breaking_changes:
    ...     print("WARNING: breaking changes detected!")
    >>> rollback = diff.reverse()
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Change Type
# ---------------------------------------------------------------------------


class ChangeType(str, Enum):
    """Types of configuration changes."""

    ADDED = "added"
    """A new field was added."""

    REMOVED = "removed"
    """An existing field was removed."""

    MODIFIED = "modified"
    """An existing field's value changed."""


# ---------------------------------------------------------------------------
# Config Change
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ConfigChange:
    """Single field-level change between two configurations.

    Attributes:
        field_path: Dot-notation path to the changed field (e.g., 'resource_limits.memory_limit_mb').
        old_value: Previous value (None for ADDED).
        new_value: New value (None for REMOVED).
        change_type: Type of change.
        is_breaking: Whether this change could break agent behaviour.
        breaking_reason: Explanation of why the change is breaking.
    """

    field_path: str
    old_value: Any = None
    new_value: Any = None
    change_type: ChangeType = ChangeType.MODIFIED
    is_breaking: bool = False
    breaking_reason: str = ""


# ---------------------------------------------------------------------------
# Breaking Change Rules
# ---------------------------------------------------------------------------

# Fields where a decrease is considered a breaking change
_DECREASE_IS_BREAKING: Set[str] = {
    "timeout_seconds",
    "resource_limits.cpu_limit_cores",
    "resource_limits.memory_limit_mb",
    "resource_limits.max_execution_seconds",
    "resource_limits.max_concurrent",
    "retry_config.max_attempts",
    "circuit_breaker_config.sliding_window_size_s",
}

# Fields where a change is always considered breaking
_ALWAYS_BREAKING: Set[str] = {
    "enabled",
    "agent_key",
    "schema_version",
}

# Fields where an increase in threshold is breaking (tighter constraints)
_INCREASE_IS_BREAKING: Set[str] = {
    "circuit_breaker_config.failure_rate_threshold",
}


# ---------------------------------------------------------------------------
# Config Diff
# ---------------------------------------------------------------------------


@dataclass
class ConfigDiff:
    """Structured diff between two agent configurations.

    Attributes:
        agent_key: The agent this diff applies to.
        old_version: Version of the old configuration.
        new_version: Version of the new configuration.
        changes: List of individual field changes.
    """

    agent_key: str = ""
    old_version: int = 0
    new_version: int = 0
    changes: List[ConfigChange] = field(default_factory=list)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def has_changes(self) -> bool:
        """Whether any changes were detected."""
        return len(self.changes) > 0

    @property
    def has_breaking_changes(self) -> bool:
        """Whether any breaking changes were detected."""
        return any(c.is_breaking for c in self.changes)

    @property
    def breaking_changes(self) -> List[ConfigChange]:
        """Return only breaking changes."""
        return [c for c in self.changes if c.is_breaking]

    @property
    def added_fields(self) -> List[ConfigChange]:
        """Return only added fields."""
        return [c for c in self.changes if c.change_type == ChangeType.ADDED]

    @property
    def removed_fields(self) -> List[ConfigChange]:
        """Return only removed fields."""
        return [c for c in self.changes if c.change_type == ChangeType.REMOVED]

    @property
    def modified_fields(self) -> List[ConfigChange]:
        """Return only modified fields."""
        return [c for c in self.changes if c.change_type == ChangeType.MODIFIED]

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def compare(
        cls,
        old_config: Dict[str, Any],
        new_config: Dict[str, Any],
        agent_key: str = "",
    ) -> ConfigDiff:
        """Compare two configuration dictionaries and produce a diff.

        Args:
            old_config: Previous configuration (dict or model_dump).
            new_config: New configuration (dict or model_dump).
            agent_key: Agent identifier for context.

        Returns:
            ConfigDiff with all detected changes.
        """
        changes: List[ConfigChange] = []
        old_version = old_config.get("version", 0)
        new_version = new_config.get("version", 0)

        # Recursively compare
        _compare_dicts(old_config, new_config, "", changes)

        diff = cls(
            agent_key=agent_key or new_config.get("agent_key", ""),
            old_version=old_version,
            new_version=new_version,
            changes=changes,
        )

        if diff.has_breaking_changes:
            logger.warning(
                "ConfigDiff: %d breaking changes detected for '%s' (v%d -> v%d)",
                len(diff.breaking_changes), diff.agent_key,
                old_version, new_version,
            )

        return diff

    # ------------------------------------------------------------------
    # Reverse (Rollback)
    # ------------------------------------------------------------------

    def reverse(self) -> ConfigDiff:
        """Generate a reverse diff for rollback.

        Returns:
            ConfigDiff that would undo the changes in this diff.
        """
        reversed_changes: List[ConfigChange] = []
        for change in self.changes:
            if change.change_type == ChangeType.ADDED:
                reversed_changes.append(ConfigChange(
                    field_path=change.field_path,
                    old_value=change.new_value,
                    new_value=None,
                    change_type=ChangeType.REMOVED,
                ))
            elif change.change_type == ChangeType.REMOVED:
                reversed_changes.append(ConfigChange(
                    field_path=change.field_path,
                    old_value=None,
                    new_value=change.old_value,
                    change_type=ChangeType.ADDED,
                ))
            elif change.change_type == ChangeType.MODIFIED:
                reversed_changes.append(ConfigChange(
                    field_path=change.field_path,
                    old_value=change.new_value,
                    new_value=change.old_value,
                    change_type=ChangeType.MODIFIED,
                ))

        return ConfigDiff(
            agent_key=self.agent_key,
            old_version=self.new_version,
            new_version=self.old_version,
            changes=reversed_changes,
        )

    # ------------------------------------------------------------------
    # Apply
    # ------------------------------------------------------------------

    def apply(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Apply this diff to a configuration dictionary.

        Args:
            config: The base configuration to apply changes to.

        Returns:
            New dictionary with all changes applied.
        """
        result = _deep_copy_dict(config)

        for change in self.changes:
            parts = change.field_path.split(".")
            if change.change_type == ChangeType.REMOVED:
                _delete_nested(result, parts)
            else:
                _set_nested(result, parts, change.new_value)

        return result

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    def summary(self) -> str:
        """Generate a human-readable summary of the diff.

        Returns:
            Multi-line string summary.
        """
        if not self.changes:
            return f"No changes for '{self.agent_key}'."

        lines = [
            f"Config diff for '{self.agent_key}' "
            f"(v{self.old_version} -> v{self.new_version}): "
            f"{len(self.changes)} change(s)",
        ]

        for change in self.changes:
            prefix = "[BREAKING] " if change.is_breaking else ""
            if change.change_type == ChangeType.ADDED:
                lines.append(f"  + {prefix}{change.field_path} = {change.new_value!r}")
            elif change.change_type == ChangeType.REMOVED:
                lines.append(f"  - {prefix}{change.field_path} (was {change.old_value!r})")
            else:
                lines.append(
                    f"  ~ {prefix}{change.field_path}: "
                    f"{change.old_value!r} -> {change.new_value!r}"
                )
                if change.breaking_reason:
                    lines.append(f"    Reason: {change.breaking_reason}")

        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Internal Helpers
# ---------------------------------------------------------------------------


def _compare_dicts(
    old: Dict[str, Any],
    new: Dict[str, Any],
    prefix: str,
    changes: List[ConfigChange],
) -> None:
    """Recursively compare two dictionaries and accumulate changes."""
    all_keys = set(old.keys()) | set(new.keys())

    # Skip internal/metadata fields from diff
    skip_keys = {"updated_at"}

    for key in sorted(all_keys):
        if key in skip_keys:
            continue

        path = f"{prefix}.{key}" if prefix else key
        old_val = old.get(key)
        new_val = new.get(key)

        if key not in old:
            # Added
            is_breaking, reason = _check_breaking(path, None, new_val, ChangeType.ADDED)
            changes.append(ConfigChange(
                field_path=path,
                old_value=None,
                new_value=new_val,
                change_type=ChangeType.ADDED,
                is_breaking=is_breaking,
                breaking_reason=reason,
            ))
        elif key not in new:
            # Removed
            is_breaking, reason = _check_breaking(path, old_val, None, ChangeType.REMOVED)
            changes.append(ConfigChange(
                field_path=path,
                old_value=old_val,
                new_value=None,
                change_type=ChangeType.REMOVED,
                is_breaking=is_breaking,
                breaking_reason=reason,
            ))
        elif isinstance(old_val, dict) and isinstance(new_val, dict):
            # Recurse into nested dicts
            _compare_dicts(old_val, new_val, path, changes)
        elif old_val != new_val:
            # Modified
            is_breaking, reason = _check_breaking(path, old_val, new_val, ChangeType.MODIFIED)
            changes.append(ConfigChange(
                field_path=path,
                old_value=old_val,
                new_value=new_val,
                change_type=ChangeType.MODIFIED,
                is_breaking=is_breaking,
                breaking_reason=reason,
            ))


def _check_breaking(
    path: str,
    old_val: Any,
    new_val: Any,
    change_type: ChangeType,
) -> tuple[bool, str]:
    """Determine if a change is breaking and return the reason."""
    # Always-breaking fields
    if path in _ALWAYS_BREAKING:
        if change_type == ChangeType.MODIFIED:
            return True, f"'{path}' is a critical field"
        if change_type == ChangeType.REMOVED:
            return True, f"'{path}' was removed"

    # Decrease-is-breaking fields
    if path in _DECREASE_IS_BREAKING and change_type == ChangeType.MODIFIED:
        try:
            if float(new_val) < float(old_val):
                return True, (
                    f"'{path}' decreased from {old_val} to {new_val} "
                    f"(may cause failures)"
                )
        except (TypeError, ValueError):
            pass

    # Increase-is-breaking (tighter constraints)
    if path in _INCREASE_IS_BREAKING and change_type == ChangeType.MODIFIED:
        try:
            if float(new_val) > float(old_val):
                return True, (
                    f"'{path}' increased from {old_val} to {new_val} "
                    f"(tighter constraint)"
                )
        except (TypeError, ValueError):
            pass

    # Removal of any field is potentially breaking
    if change_type == ChangeType.REMOVED:
        return False, ""

    return False, ""


def _deep_copy_dict(d: Dict[str, Any]) -> Dict[str, Any]:
    """Deep copy a dictionary without importing copy module."""
    result: Dict[str, Any] = {}
    for k, v in d.items():
        if isinstance(v, dict):
            result[k] = _deep_copy_dict(v)
        elif isinstance(v, list):
            result[k] = list(v)
        else:
            result[k] = v
    return result


def _set_nested(d: Dict[str, Any], parts: List[str], value: Any) -> None:
    """Set a value at a nested path in a dictionary."""
    for part in parts[:-1]:
        if part not in d or not isinstance(d[part], dict):
            d[part] = {}
        d = d[part]
    d[parts[-1]] = value


def _delete_nested(d: Dict[str, Any], parts: List[str]) -> None:
    """Delete a key at a nested path in a dictionary."""
    for part in parts[:-1]:
        if part not in d or not isinstance(d[part], dict):
            return
        d = d[part]
    d.pop(parts[-1], None)


__all__ = [
    "ChangeType",
    "ConfigChange",
    "ConfigDiff",
]
