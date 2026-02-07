# -*- coding: utf-8 -*-
"""
Version Compatibility Matrix - Track agent-to-agent and agent-to-platform compatibility.

Maintains a matrix of tested compatibility relationships between agent versions.
Supports auto-detection of breaking changes via input/output schema comparison
and persists the matrix to PostgreSQL.

Example:
    >>> matrix = VersionCompatibilityMatrix()
    >>> matrix.record("agent-a", "1.0.0", "agent-b", "2.1.0", CompatibilityStatus.COMPATIBLE)
    >>> status = matrix.check("agent-a", "1.0.0", "agent-b", "2.1.0")
    >>> assert status == CompatibilityStatus.COMPATIBLE

Author: GreenLang Platform Team
Date: February 2026
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------


class CompatibilityStatus(str, Enum):
    """Status of a compatibility check between two agent versions."""

    COMPATIBLE = "compatible"
    """The two versions have been tested and work together."""

    INCOMPATIBLE = "incompatible"
    """The two versions have been tested and do NOT work together."""

    UNTESTED = "untested"
    """The combination has not been tested yet."""


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CompatibilityEntry:
    """A single compatibility record between two agent versions.

    Attributes:
        agent_a: First agent key.
        version_a: First agent version.
        agent_b: Second agent key.
        version_b: Second agent version.
        status: Compatibility status.
        tested_at: Timestamp when the compatibility was tested.
        notes: Optional notes about the compatibility result.
    """

    agent_a: str
    version_a: str
    agent_b: str
    version_b: str
    status: CompatibilityStatus
    tested_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    notes: str = ""

    @property
    def key(self) -> Tuple[str, str, str, str]:
        """Return the normalized lookup key (sorted agent pair)."""
        pair_a = (self.agent_a, self.version_a)
        pair_b = (self.agent_b, self.version_b)
        if pair_a <= pair_b:
            return (self.agent_a, self.version_a, self.agent_b, self.version_b)
        return (self.agent_b, self.version_b, self.agent_a, self.version_a)


# ---------------------------------------------------------------------------
# Schema diff helper
# ---------------------------------------------------------------------------


def _compute_schema_diff(
    old_schema: Dict[str, Any],
    new_schema: Dict[str, Any],
) -> Dict[str, Any]:
    """Compare two JSON schemas and identify changes.

    Detects:
      - Added required fields (potentially breaking)
      - Removed fields (breaking)
      - Changed field types (breaking)
      - Added optional fields (non-breaking)

    Args:
        old_schema: Previous JSON schema.
        new_schema: New JSON schema.

    Returns:
        Dict with keys: added, removed, type_changed, compatible.
    """
    old_fields = old_schema.get("fields", old_schema.get("properties", {}))
    new_fields = new_schema.get("fields", new_schema.get("properties", {}))
    old_required = set(old_schema.get("required", []))
    new_required = set(new_schema.get("required", []))

    old_keys = set(old_fields.keys())
    new_keys = set(new_fields.keys())

    added = new_keys - old_keys
    removed = old_keys - new_keys

    type_changed: List[str] = []
    for key in old_keys & new_keys:
        old_type = old_fields[key].get("type") if isinstance(old_fields[key], dict) else str(old_fields[key])
        new_type = new_fields[key].get("type") if isinstance(new_fields[key], dict) else str(new_fields[key])
        if old_type != new_type:
            type_changed.append(key)

    # Breaking: removed fields, type changes, or new required fields
    new_required_fields = (added & new_required)
    is_compatible = (
        len(removed) == 0
        and len(type_changed) == 0
        and len(new_required_fields) == 0
    )

    return {
        "added": sorted(added),
        "removed": sorted(removed),
        "type_changed": sorted(type_changed),
        "new_required_fields": sorted(new_required_fields),
        "compatible": is_compatible,
    }


# ---------------------------------------------------------------------------
# Compatibility Matrix
# ---------------------------------------------------------------------------


class VersionCompatibilityMatrix:
    """Track and query agent version compatibility.

    Stores compatibility entries in memory with an option to persist
    to PostgreSQL via async methods.

    Attributes:
        entries: In-memory compatibility entries indexed by tuple key.
        platform_version: Current platform version for platform-level checks.
    """

    def __init__(self, platform_version: str = "1.0.0") -> None:
        """Initialize the compatibility matrix.

        Args:
            platform_version: Current GreenLang platform version.
        """
        self.platform_version = platform_version
        self._entries: Dict[Tuple[str, str, str, str], CompatibilityEntry] = {}

    def record(
        self,
        agent_a: str,
        version_a: str,
        agent_b: str,
        version_b: str,
        status: CompatibilityStatus,
        notes: str = "",
    ) -> CompatibilityEntry:
        """Record a compatibility test result.

        Args:
            agent_a: First agent key.
            version_a: First agent version.
            agent_b: Second agent key.
            version_b: Second agent version.
            status: Compatibility result.
            notes: Optional notes.

        Returns:
            The created CompatibilityEntry.
        """
        entry = CompatibilityEntry(
            agent_a=agent_a,
            version_a=version_a,
            agent_b=agent_b,
            version_b=version_b,
            status=status,
            notes=notes,
        )
        self._entries[entry.key] = entry
        logger.info(
            "Recorded compatibility: %s@%s <-> %s@%s = %s",
            agent_a, version_a, agent_b, version_b, status.value,
        )
        return entry

    def check(
        self,
        agent_a: str,
        version_a: str,
        agent_b: str,
        version_b: str,
    ) -> CompatibilityStatus:
        """Check compatibility between two agent versions.

        Args:
            agent_a: First agent key.
            version_a: First agent version.
            agent_b: Second agent key.
            version_b: Second agent version.

        Returns:
            Compatibility status (COMPATIBLE, INCOMPATIBLE, or UNTESTED).
        """
        # Normalize key order
        pair_a = (agent_a, version_a)
        pair_b = (agent_b, version_b)
        if pair_a <= pair_b:
            key = (agent_a, version_a, agent_b, version_b)
        else:
            key = (agent_b, version_b, agent_a, version_a)

        entry = self._entries.get(key)
        if entry is None:
            return CompatibilityStatus.UNTESTED
        return entry.status

    def check_platform(
        self,
        agent_key: str,
        agent_version: str,
    ) -> CompatibilityStatus:
        """Check agent compatibility with the current platform version.

        Args:
            agent_key: Agent key.
            agent_version: Agent version.

        Returns:
            Compatibility status.
        """
        return self.check(
            agent_key, agent_version, "__platform__", self.platform_version
        )

    def detect_breaking_changes(
        self,
        old_input_schema: Dict[str, Any],
        new_input_schema: Dict[str, Any],
        old_output_schema: Dict[str, Any],
        new_output_schema: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Auto-detect breaking changes via schema comparison.

        Args:
            old_input_schema: Previous input schema.
            new_input_schema: New input schema.
            old_output_schema: Previous output schema.
            new_output_schema: New output schema.

        Returns:
            Dict with input_diff, output_diff, and is_breaking flag.
        """
        input_diff = _compute_schema_diff(old_input_schema, new_input_schema)
        output_diff = _compute_schema_diff(old_output_schema, new_output_schema)
        is_breaking = not input_diff["compatible"] or not output_diff["compatible"]

        if is_breaking:
            logger.warning(
                "Breaking changes detected: input=%s, output=%s",
                input_diff,
                output_diff,
            )

        return {
            "input_diff": input_diff,
            "output_diff": output_diff,
            "is_breaking": is_breaking,
        }

    def get_all_entries(self) -> List[CompatibilityEntry]:
        """Return all recorded compatibility entries."""
        return list(self._entries.values())

    def get_entries_for_agent(
        self, agent_key: str
    ) -> List[CompatibilityEntry]:
        """Return all compatibility entries involving a specific agent.

        Args:
            agent_key: Agent key to filter by.

        Returns:
            List of matching entries.
        """
        return [
            e for e in self._entries.values()
            if e.agent_a == agent_key or e.agent_b == agent_key
        ]

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the matrix to a dictionary for persistence."""
        return {
            "platform_version": self.platform_version,
            "entries": [
                {
                    "agent_a": e.agent_a,
                    "version_a": e.version_a,
                    "agent_b": e.agent_b,
                    "version_b": e.version_b,
                    "status": e.status.value,
                    "tested_at": e.tested_at,
                    "notes": e.notes,
                }
                for e in self._entries.values()
            ],
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> VersionCompatibilityMatrix:
        """Deserialize the matrix from a dictionary.

        Args:
            data: Serialized matrix data.

        Returns:
            Reconstructed VersionCompatibilityMatrix.
        """
        matrix = cls(platform_version=data.get("platform_version", "1.0.0"))
        for entry_data in data.get("entries", []):
            entry = CompatibilityEntry(
                agent_a=entry_data["agent_a"],
                version_a=entry_data["version_a"],
                agent_b=entry_data["agent_b"],
                version_b=entry_data["version_b"],
                status=CompatibilityStatus(entry_data["status"]),
                tested_at=entry_data.get("tested_at", ""),
                notes=entry_data.get("notes", ""),
            )
            matrix._entries[entry.key] = entry
        return matrix
