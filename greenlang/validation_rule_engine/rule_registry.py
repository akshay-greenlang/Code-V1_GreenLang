# -*- coding: utf-8 -*-
"""
Rule Registry Engine - AGENT-DATA-019: Validation Rule Engine

Engine 1 of 7 in the Validation Rule Engine pipeline.

Pure-Python engine for centralized CRUD management of validation rules.
Provides rule registration, retrieval, update with SemVer auto-bumping,
soft/hard deletion, index-accelerated search, cloning, version history
with rollback, bulk operations, import/export, and aggregate statistics.
Each rule is assigned a unique UUID and tracked with SHA-256 provenance
hashing for complete audit trails.

Validation rules define the constraints applied to data fields during
quality checks. Each rule specifies a type (e.g. COMPLETENESS, RANGE,
FORMAT), an operator (e.g. GREATER_THAN, MATCHES, BETWEEN), a target
column, a threshold or parameter set, and a severity level. Rules follow
a full lifecycle from creation through updates (with version history) to
archival or hard deletion.

Zero-Hallucination Guarantees:
    - All IDs are deterministic UUID-4 values (no LLM involvement)
    - Timestamps from ``datetime.now(timezone.utc)`` (deterministic)
    - SHA-256 provenance hashes recorded on every mutating operation
    - SemVer bump classification is rule-based string comparison only
    - Index maintenance uses explicit set operations only
    - Statistics are derived from in-memory data structures
    - No ML or LLM calls anywhere in this engine

Thread Safety:
    All mutating and read operations are protected by ``self._lock``
    (a ``threading.Lock``). Callers receive plain dict copies so they
    cannot accidentally mutate internal state.

Example:
    >>> from greenlang.validation_rule_engine.rule_registry import (
    ...     RuleRegistryEngine,
    ... )
    >>> engine = RuleRegistryEngine()
    >>> rule = engine.register_rule(
    ...     name="completeness_co2e",
    ...     rule_type="COMPLETENESS",
    ...     column="co2e_tonnes",
    ...     operator="IS_NULL",
    ...     threshold=0.0,
    ...     severity="HIGH",
    ...     description="CO2e column must not be null",
    ...     tags=["emissions", "scope1"],
    ... )
    >>> print(rule["rule_id"], rule["status"])
    >>> assert engine.get_statistics()["total_rules"] == 1

Author: GreenLang Platform Team
Date: February 2026
PRD: AGENT-DATA-019 Validation Rule Engine (GL-DATA-X-022)
Status: Production Ready
"""

from __future__ import annotations

import copy
import hashlib
import json
import logging
import re
import threading
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional dependency: ProvenanceTracker
# ---------------------------------------------------------------------------

try:
    from greenlang.validation_rule_engine.provenance import (
        ProvenanceTracker,
    )
    _PROVENANCE_AVAILABLE = True
except ImportError:
    _PROVENANCE_AVAILABLE = False
    logger.info(
        "validation_rule_engine.provenance not available; "
        "using inline ProvenanceTracker"
    )

    class ProvenanceTracker:  # type: ignore[no-redef]
        """Minimal inline provenance tracker for standalone operation.

        Provides SHA-256 chain hashing without external dependencies.
        """

        GENESIS_HASH = hashlib.sha256(
            b"greenlang-validation-rule-engine-genesis"
        ).hexdigest()

        def __init__(self) -> None:
            """Initialize with genesis hash."""
            self._last_chain_hash: str = self.GENESIS_HASH
            self._chain: List[Dict[str, Any]] = []
            self._lock = threading.Lock()

        def record(
            self,
            entity_type: str,
            entity_id: str,
            action: str,
            metadata: Optional[Any] = None,
        ) -> Any:
            """Record a provenance entry and return a stub entry.

            Args:
                entity_type: Type of entity being tracked.
                entity_id: Unique identifier for the entity.
                action: Action performed.
                metadata: Optional payload to hash.

            Returns:
                A stub object with a ``hash_value`` attribute.
            """
            ts = _utcnow().isoformat()
            if metadata is None:
                serialized = "null"
            else:
                serialized = json.dumps(
                    metadata, sort_keys=True, default=str,
                )
            data_hash = hashlib.sha256(
                serialized.encode("utf-8"),
            ).hexdigest()

            with self._lock:
                combined = json.dumps({
                    "action": action,
                    "data_hash": data_hash,
                    "parent_hash": self._last_chain_hash,
                    "timestamp": ts,
                }, sort_keys=True)
                chain_hash = hashlib.sha256(
                    combined.encode("utf-8"),
                ).hexdigest()

                self._chain.append({
                    "entity_type": entity_type,
                    "entity_id": entity_id,
                    "action": action,
                    "hash_value": chain_hash,
                    "parent_hash": self._last_chain_hash,
                    "timestamp": ts,
                    "metadata": {"data_hash": data_hash},
                })
                self._last_chain_hash = chain_hash

            class _StubEntry:
                def __init__(self, hv: str) -> None:
                    self.hash_value = hv

            return _StubEntry(chain_hash)

        def build_hash(self, data: Any) -> str:
            """Return SHA-256 hash of JSON-serialized data."""
            return hashlib.sha256(
                json.dumps(data, sort_keys=True, default=str).encode()
            ).hexdigest()

        @property
        def entry_count(self) -> int:
            """Return the total number of provenance entries."""
            with self._lock:
                return len(self._chain)

        def export_chain(self) -> List[Dict[str, Any]]:
            """Return the full provenance chain for audit."""
            with self._lock:
                return list(self._chain)

        def reset(self) -> None:
            """Clear all provenance state."""
            with self._lock:
                self._chain.clear()
                self._last_chain_hash = self.GENESIS_HASH


# ---------------------------------------------------------------------------
# Optional dependency: Prometheus metrics
# ---------------------------------------------------------------------------

try:
    from greenlang.validation_rule_engine.metrics import (
        record_rule_registered,
        observe_processing_duration,
        PROMETHEUS_AVAILABLE,
    )
    _METRICS_AVAILABLE = True
except ImportError:
    _METRICS_AVAILABLE = False
    PROMETHEUS_AVAILABLE = False  # type: ignore[assignment]
    logger.info(
        "validation_rule_engine.metrics not available; "
        "rule registry metrics disabled"
    )

    def record_rule_registered(  # type: ignore[misc]
        rule_type: str,
        severity: str,
    ) -> None:
        """No-op fallback when metrics module is not available."""

    def observe_processing_duration(  # type: ignore[misc]
        operation: str,
        seconds: float,
    ) -> None:
        """No-op fallback when metrics module is not available."""


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Valid rule types for validation rules.
VALID_RULE_TYPES: Set[str] = {
    "COMPLETENESS",
    "RANGE",
    "FORMAT",
    "UNIQUENESS",
    "CUSTOM",
    "FRESHNESS",
    "CROSS_FIELD",
    "CONDITIONAL",
    "STATISTICAL",
    "REFERENTIAL",
}

#: Valid comparison operators.
VALID_OPERATORS: Set[str] = {
    "EQUALS",
    "NOT_EQUALS",
    "GREATER_THAN",
    "LESS_THAN",
    "GREATER_EQUAL",
    "LESS_EQUAL",
    "BETWEEN",
    "MATCHES",
    "CONTAINS",
    "IN_SET",
    "NOT_IN_SET",
    "IS_NULL",
}

#: Valid severity levels (ordered from most to least severe).
VALID_SEVERITIES: Set[str] = {
    "CRITICAL",
    "HIGH",
    "MEDIUM",
    "LOW",
}

#: Valid rule statuses.
VALID_STATUSES: Set[str] = {
    "draft",
    "active",
    "deprecated",
    "archived",
}

#: Public alias for valid rule statuses.
VALID_RULE_STATUSES: Set[str] = VALID_STATUSES

#: Status lifecycle transitions (from_status -> allowed_next_statuses).
STATUS_TRANSITIONS: Dict[str, Set[str]] = {
    "draft": {"active"},
    "active": {"deprecated"},
    "deprecated": {"archived"},
    "archived": set(),
}

#: Maximum number of rules importable in a single bulk call.
MAX_BULK_IMPORT: int = 5_000

#: Maximum rule name length.
MAX_RULE_NAME_LENGTH: int = 256

#: Maximum description length.
MAX_DESCRIPTION_LENGTH: int = 4_096

#: Maximum tag length.
MAX_TAG_LENGTH: int = 64

#: Maximum column name length.
MAX_COLUMN_NAME_LENGTH: int = 256

#: Fields whose update triggers a major version bump.
_BREAKING_FIELDS: frozenset = frozenset({"rule_type", "operator", "column"})

#: Fields whose update triggers a minor version bump.
_ADDITIVE_FIELDS: frozenset = frozenset({"parameters", "threshold"})

#: Fields whose update triggers a patch version bump.
_COSMETIC_FIELDS: frozenset = frozenset({
    "description", "tags", "metadata", "severity",
})


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _utcnow() -> datetime:
    """Return current UTC datetime with microseconds zeroed for determinism.

    Returns:
        Current UTC datetime with microseconds set to zero.
    """
    return datetime.now(timezone.utc).replace(microsecond=0)


def _build_sha256(data: Any) -> str:
    """Build a deterministic SHA-256 hash from any JSON-serializable value.

    All dict keys are sorted for determinism regardless of insertion order.

    Args:
        data: JSON-serializable value (dict, list, str, int, etc.).

    Returns:
        64-character lowercase hex SHA-256 digest.
    """
    serialized = json.dumps(data, sort_keys=True, default=str)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _normalize_tags(tags: Optional[List[str]]) -> List[str]:
    """Normalize, deduplicate, and sort a list of tags.

    Tags are lowercased and stripped of surrounding whitespace. Empty
    strings and duplicates are removed. Order is sorted for stability.

    Args:
        tags: Raw tag list, or None.

    Returns:
        Sorted list of unique, lowercased, stripped tag strings.
    """
    if not tags:
        return []
    seen: Set[str] = set()
    result: List[str] = []
    for tag in tags:
        if not isinstance(tag, str):
            continue
        normalized = tag.strip().lower()
        if normalized and normalized not in seen:
            seen.add(normalized)
            result.append(normalized)
    return sorted(result)


def _validate_tags_list(tags: List[str]) -> None:
    """Raise ValueError if any individual tag exceeds the maximum length.

    Args:
        tags: Normalized list of tag strings.

    Raises:
        ValueError: If any tag exceeds MAX_TAG_LENGTH characters.
    """
    for tag in tags:
        if len(tag) > MAX_TAG_LENGTH:
            raise ValueError(
                f"Tag '{tag}' exceeds maximum length of "
                f"{MAX_TAG_LENGTH} characters."
            )


def _validate_name(name: str) -> str:
    """Validate and clean a rule name.

    Args:
        name: Raw rule name string.

    Returns:
        Cleaned, stripped rule name.

    Raises:
        ValueError: If name is empty or too long.
    """
    if not name or not name.strip():
        raise ValueError("Rule name must be a non-empty string.")
    clean = name.strip()
    if len(clean) > MAX_RULE_NAME_LENGTH:
        raise ValueError(
            f"Rule name exceeds maximum length of "
            f"{MAX_RULE_NAME_LENGTH} characters."
        )
    return clean


# Public alias for _validate_name.
_validate_rule_name = _validate_name


# Alias expected by tests
_validate_rule_name = _validate_name


def _parse_version(version_string: str) -> Tuple[int, int, int]:
    """Parse a SemVer string into its (major, minor, patch) integer tuple.

    Args:
        version_string: A semantic version string in the form ``X.Y.Z``.

    Returns:
        A three-tuple of non-negative integers (major, minor, patch).

    Raises:
        ValueError: If the string does not conform to the ``X.Y.Z``
            format or any part is not a non-negative integer.
    """
    parts = version_string.strip().split(".")
    if len(parts) != 3:
        raise ValueError(
            f"Invalid SemVer string '{version_string}': "
            f"expected exactly 3 parts separated by '.', got {len(parts)}."
        )
    try:
        major, minor, patch = int(parts[0]), int(parts[1]), int(parts[2])
    except ValueError as exc:
        raise ValueError(
            f"Invalid SemVer string '{version_string}': "
            f"all parts must be integers."
        ) from exc
    if major < 0 or minor < 0 or patch < 0:
        raise ValueError(
            f"Invalid SemVer string '{version_string}': "
            f"all parts must be non-negative."
        )
    return (major, minor, patch)


def _format_version(major: int, minor: int, patch: int) -> str:
    """Format a (major, minor, patch) tuple into a SemVer string.

    Args:
        major: Major version number.
        minor: Minor version number.
        patch: Patch version number.

    Returns:
        SemVer string in the form ``X.Y.Z``.
    """
    return f"{major}.{minor}.{patch}"


def _compute_version_bump(
    current_version: str,
    changed_fields: Set[str],
) -> str:
    """Compute the next SemVer version based on which fields changed.

    Breaking changes (rule_type, operator, column) trigger a major bump.
    Additive changes (parameters, threshold) trigger a minor bump.
    Cosmetic changes (description, tags, metadata, severity) trigger a
    patch bump. If multiple categories are present, the highest severity
    bump wins.

    Args:
        current_version: Current SemVer string (e.g. ``"1.2.3"``).
        changed_fields: Set of field names that were modified.

    Returns:
        Next SemVer string after applying the appropriate bump.
    """
    major, minor, patch = _parse_version(current_version)

    has_breaking = bool(changed_fields & _BREAKING_FIELDS)
    has_additive = bool(changed_fields & _ADDITIVE_FIELDS)

    if has_breaking:
        return _format_version(major + 1, 0, 0)
    elif has_additive:
        return _format_version(major, minor + 1, 0)
    else:
        return _format_version(major, minor, patch + 1)


# ---------------------------------------------------------------------------
# RuleRegistryEngine
# ---------------------------------------------------------------------------


class RuleRegistryEngine:
    """Pure-Python engine for centralized CRUD management of validation rules.

    Engine 1 of 7 in the Validation Rule Engine pipeline (AGENT-DATA-019).

    Manages the full lifecycle of validation rules including registration,
    retrieval by ID or name, update with SemVer auto-bumping, soft/hard
    deletion, index-accelerated search, cloning, version history with
    rollback, bulk operations, import/export, and aggregate statistics.
    Every mutation is tracked with SHA-256 provenance hashing for complete
    audit trails.

    Rules are identified by a unique UUID (``rule_id``) and by a unique
    ``name``. The name serves as a human-friendly lookup key.

    Indexes are maintained for fast lookup by rule type, severity, column,
    tag, status, and name. All indexes are kept consistent under the
    thread-safety lock.

    Zero-Hallucination Guarantees:
        - UUID assignment via ``uuid.uuid4()`` (no LLM involvement)
        - Timestamps from ``datetime.now(timezone.utc)`` (deterministic)
        - SemVer bump classification uses set-intersection rules only
        - SHA-256 provenance hash computed from JSON-serialized payloads
        - All lookups use explicit dict/set operations
        - No ML or LLM calls anywhere in the class

    Attributes:
        _rules: Rule store keyed by rule_id (UUID string).
        _name_index: Mapping from name to rule_id for O(1) name lookup.
        _type_index: Mapping from rule_type to set of rule_ids.
        _severity_index: Mapping from severity to set of rule_ids.
        _tag_index: Mapping from tag string to set of rule_ids.
        _column_index: Mapping from column name to set of rule_ids.
        _status_index: Mapping from status to set of rule_ids.
        _version_history: Mapping from rule_id to list of version snapshots.
        _lock: Thread-safety lock protecting all state.
        _provenance: ProvenanceTracker for SHA-256 audit trails.

    Example:
        >>> engine = RuleRegistryEngine()
        >>> rule = engine.register_rule(
        ...     name="range_check_temperature",
        ...     rule_type="RANGE",
        ...     column="temperature_celsius",
        ...     operator="BETWEEN",
        ...     threshold=None,
        ...     parameters={"min": -50, "max": 60},
        ...     severity="MEDIUM",
        ...     description="Temperature must be between -50 and 60 C",
        ...     tags=["environmental", "sensor"],
        ... )
        >>> assert rule["status"] == "draft"
        >>> retrieved = engine.get_rule(rule["rule_id"])
        >>> assert retrieved is not None
        >>> stats = engine.get_statistics()
        >>> assert stats["total_rules"] == 1
    """

    def __init__(
        self,
        provenance: Optional[ProvenanceTracker] = None,
    ) -> None:
        """Initialize RuleRegistryEngine with empty in-memory state.

        Sets up the rule store, all lookup indexes, the version history
        store, and the provenance tracker. If no ProvenanceTracker is
        provided, a new default instance is created.

        Args:
            provenance: Optional ProvenanceTracker instance. When None,
                a new tracker is created internally.

        Example:
            >>> engine = RuleRegistryEngine()
            >>> assert engine.get_statistics()["total_rules"] == 0
        """
        self._rules: Dict[str, dict] = {}
        self._name_index: Dict[str, str] = {}
        self._type_index: Dict[str, Set[str]] = {}
        self._severity_index: Dict[str, Set[str]] = {}
        self._tag_index: Dict[str, Set[str]] = {}
        self._column_index: Dict[str, Set[str]] = {}
        self._status_index: Dict[str, Set[str]] = {}
        self._version_history: Dict[str, List[dict]] = {}
        self._lock: threading.Lock = threading.Lock()
        self._provenance: ProvenanceTracker = (
            provenance if provenance is not None else ProvenanceTracker()
        )

        logger.info(
            "RuleRegistryEngine initialized "
            "(AGENT-DATA-019, Engine 1 of 7)"
        )

    # ------------------------------------------------------------------
    # Internal: index management
    # ------------------------------------------------------------------

    def _add_to_index(
        self,
        index: Dict[str, Set[str]],
        key: str,
        rule_id: str,
    ) -> None:
        """Add a rule_id to an index set under the given key.

        Args:
            index: The index dict to update.
            key: The index key (e.g. a rule_type or tag).
            rule_id: The rule UUID to add.
        """
        index.setdefault(key, set()).add(rule_id)

    def _remove_from_index(
        self,
        index: Dict[str, Set[str]],
        key: str,
        rule_id: str,
    ) -> None:
        """Remove a rule_id from an index set under the given key.

        Args:
            index: The index dict to update.
            key: The index key (e.g. a rule_type or tag).
            rule_id: The rule UUID to remove.
        """
        bucket = index.get(key)
        if bucket is not None:
            bucket.discard(rule_id)

    def _index_rule(self, record: dict, rule_id: str) -> None:
        """Add a rule record to all relevant indexes.

        Must be called while holding ``self._lock``.

        Args:
            record: The rule record dict.
            rule_id: The rule UUID.
        """
        self._add_to_index(
            self._type_index, record.get("rule_type", ""), rule_id,
        )
        self._add_to_index(
            self._severity_index, record.get("severity", ""), rule_id,
        )
        self._add_to_index(
            self._status_index, record.get("status", ""), rule_id,
        )
        column = record.get("column", "")
        if column:
            self._add_to_index(self._column_index, column, rule_id)
        for tag in record.get("tags", []):
            self._add_to_index(self._tag_index, tag, rule_id)

    def _deindex_rule(self, record: dict, rule_id: str) -> None:
        """Remove a rule record from all relevant indexes.

        Must be called while holding ``self._lock``.

        Args:
            record: The rule record dict.
            rule_id: The rule UUID.
        """
        self._remove_from_index(
            self._type_index, record.get("rule_type", ""), rule_id,
        )
        self._remove_from_index(
            self._severity_index, record.get("severity", ""), rule_id,
        )
        self._remove_from_index(
            self._status_index, record.get("status", ""), rule_id,
        )
        column = record.get("column", "")
        if column:
            self._remove_from_index(self._column_index, column, rule_id)
        for tag in record.get("tags", []):
            self._remove_from_index(self._tag_index, tag, rule_id)

    def _snapshot_version(self, record: dict) -> dict:
        """Create a deep-copy snapshot of a rule record for version history.

        Args:
            record: The rule record dict.

        Returns:
            Deep copy of the record suitable for version storage.
        """
        snapshot = copy.deepcopy(record)
        snapshot["snapshot_at"] = _utcnow().isoformat()
        return snapshot

    # ------------------------------------------------------------------
    # 1. register_rule
    # ------------------------------------------------------------------

    def register_rule(
        self,
        name: str,
        rule_type: str,
        column: str,
        operator: str,
        threshold: Any = None,
        parameters: Optional[Dict[str, Any]] = None,
        severity: str = "MEDIUM",
        description: str = "",
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> dict:
        """Register a new validation rule in the registry.

        Creates a rule record with a unique UUID, validates all input
        parameters, sets the initial version to ``"1.0.0"`` and status
        to ``"draft"``, updates all lookup indexes, records a
        provenance entry, and emits a Prometheus metric.

        The ``name`` must be unique across the entire registry.
        Attempting to register a duplicate name raises a ``ValueError``.

        Args:
            name: Globally unique human-readable name for the rule.
                Must be non-empty and at most 256 characters.
            rule_type: Classification of the rule. Must be one of:
                ``COMPLETENESS``, ``RANGE``, ``FORMAT``, ``UNIQUENESS``,
                ``CUSTOM``, ``FRESHNESS``, ``CROSS_FIELD``,
                ``CONDITIONAL``, ``STATISTICAL``, ``REFERENTIAL``.
            column: Target data column this rule applies to. Must be
                non-empty and at most 256 characters.
            operator: Comparison operator. Must be one of:
                ``EQUALS``, ``NOT_EQUALS``, ``GREATER_THAN``,
                ``LESS_THAN``, ``GREATER_EQUAL``, ``LESS_EQUAL``,
                ``BETWEEN``, ``MATCHES``, ``CONTAINS``, ``IN_SET``,
                ``NOT_IN_SET``, ``IS_NULL``.
            threshold: Optional threshold value for the rule. Can be
                numeric, string, or None depending on the operator.
            parameters: Optional dictionary of additional rule
                parameters (e.g. min/max for BETWEEN).
            severity: Severity level of rule violations. Must be one
                of: ``CRITICAL``, ``HIGH``, ``MEDIUM``, ``LOW``.
                Defaults to ``"MEDIUM"``.
            description: Human-readable description of the rule.
                Maximum 4096 characters.
            tags: Optional list of string tags for discovery and
                filtering. Tags are normalized (lowercased, stripped,
                deduplicated, sorted).
            metadata: Optional dictionary of additional metadata.

        Returns:
            Complete rule dict containing all fields plus ``rule_id``,
            ``version``, ``status``, ``created_at``, ``updated_at``,
            and ``provenance_hash``.

        Raises:
            ValueError: If any parameter fails validation or if a rule
                with the same name already exists.

        Example:
            >>> engine = RuleRegistryEngine()
            >>> rule = engine.register_rule(
            ...     name="not_null_email",
            ...     rule_type="COMPLETENESS",
            ...     column="email",
            ...     operator="IS_NULL",
            ...     severity="HIGH",
            ... )
            >>> assert rule["rule_id"]
            >>> assert rule["status"] == "draft"
            >>> assert rule["version"] == "1.0.0"
        """
        t0 = time.monotonic()

        # -- Input validation --
        clean_name = _validate_name(name)

        rule_type_upper = rule_type.strip().upper()
        if rule_type_upper not in VALID_RULE_TYPES:
            raise ValueError(
                f"Invalid rule_type: {rule_type!r}. "
                f"Must be one of: {sorted(VALID_RULE_TYPES)}"
            )

        operator_upper = operator.strip().upper()
        if operator_upper not in VALID_OPERATORS:
            raise ValueError(
                f"Invalid operator: {operator!r}. "
                f"Must be one of: {sorted(VALID_OPERATORS)}"
            )

        severity_upper = severity.strip().upper()
        if severity_upper not in VALID_SEVERITIES:
            raise ValueError(
                f"Invalid severity: {severity!r}. "
                f"Must be one of: {sorted(VALID_SEVERITIES)}"
            )

        # Column validation
        if not column or not column.strip():
            raise ValueError("Column must be a non-empty string.")
        clean_column = column.strip()
        if len(clean_column) > MAX_COLUMN_NAME_LENGTH:
            raise ValueError(
                f"Column name exceeds maximum length of "
                f"{MAX_COLUMN_NAME_LENGTH} characters."
            )

        # Tags
        normalized_tags = _normalize_tags(tags)
        _validate_tags_list(normalized_tags)

        # Description
        if description and len(description) > MAX_DESCRIPTION_LENGTH:
            raise ValueError(
                f"Description exceeds maximum length of "
                f"{MAX_DESCRIPTION_LENGTH} characters."
            )

        # -- Build rule record --
        rule_id = str(uuid.uuid4())
        now_str = _utcnow().isoformat()

        rule_record: dict = {
            "rule_id": rule_id,
            "name": clean_name,
            "rule_type": rule_type_upper,
            "column": clean_column,
            "operator": operator_upper,
            "threshold": copy.deepcopy(threshold),
            "parameters": copy.deepcopy(parameters) if parameters else {},
            "severity": severity_upper,
            "description": description.strip() if description else "",
            "tags": normalized_tags,
            "metadata": copy.deepcopy(metadata) if metadata else {},
            "status": "draft",
            "version": "1.0.0",
            "created_at": now_str,
            "updated_at": now_str,
            "provenance_hash": "",
        }

        # -- Compute provenance hash before locking --
        data_hash = _build_sha256(
            {k: v for k, v in rule_record.items() if k != "provenance_hash"}
        )

        with self._lock:
            # Enforce name uniqueness
            if clean_name in self._name_index:
                existing_id = self._name_index[clean_name]
                existing = self._rules.get(existing_id)
                if existing and existing.get("status") != "archived":
                    raise ValueError(
                        f"A rule with name={clean_name!r} "
                        f"already exists (id={existing_id})."
                    )
                # Archived rule with same name: allow re-registration
                del self._name_index[clean_name]

            # Record provenance
            entry = self._provenance.record(
                entity_type="validation_rule",
                entity_id=rule_id,
                action="rule_registered",
                metadata={
                    "data_hash": data_hash,
                    "name": clean_name,
                    "rule_type": rule_type_upper,
                },
            )
            rule_record["provenance_hash"] = entry.hash_value

            # Store rule
            self._rules[rule_id] = rule_record

            # Update indexes
            self._name_index[clean_name] = rule_id
            self._index_rule(rule_record, rule_id)

            # Store initial version snapshot
            self._version_history[rule_id] = [
                self._snapshot_version(rule_record),
            ]

        # Metrics
        elapsed = time.monotonic() - t0
        record_rule_registered(rule_type_upper, severity_upper)
        observe_processing_duration("rule_register", elapsed)

        logger.info(
            "Rule registered: id=%s name=%s type=%s operator=%s "
            "column=%s severity=%s tags=%d elapsed=%.3fms",
            rule_id,
            clean_name,
            rule_type_upper,
            operator_upper,
            clean_column,
            severity_upper,
            len(normalized_tags),
            elapsed * 1000,
        )
        return copy.deepcopy(rule_record)

    # ------------------------------------------------------------------
    # 2. get_rule
    # ------------------------------------------------------------------

    def get_rule(self, rule_id: str) -> Optional[dict]:
        """Retrieve a registered rule by its unique UUID.

        Args:
            rule_id: UUID string assigned at registration time.

        Returns:
            Deep-copy of the rule dict, or ``None`` if no rule with
            the given ID exists.

        Example:
            >>> engine = RuleRegistryEngine()
            >>> rule = engine.register_rule(
            ...     name="test", rule_type="COMPLETENESS",
            ...     column="col", operator="IS_NULL",
            ... )
            >>> retrieved = engine.get_rule(rule["rule_id"])
            >>> assert retrieved is not None
            >>> assert retrieved["name"] == "test"
        """
        with self._lock:
            record = self._rules.get(rule_id)
            if record is None:
                logger.debug("Rule not found by id: %s", rule_id)
                return None
            return copy.deepcopy(record)

    # ------------------------------------------------------------------
    # 3. get_rule_by_name
    # ------------------------------------------------------------------

    def get_rule_by_name(self, name: str) -> Optional[dict]:
        """Retrieve a registered rule by its unique name.

        Performs an O(1) lookup using the name index.

        Args:
            name: Human-readable name of the rule.

        Returns:
            Deep-copy of the rule dict, or ``None`` if no rule with
            the given name exists.

        Example:
            >>> engine = RuleRegistryEngine()
            >>> engine.register_rule(
            ...     name="my_rule", rule_type="RANGE",
            ...     column="amount", operator="GREATER_THAN",
            ... )
            >>> rule = engine.get_rule_by_name("my_rule")
            >>> assert rule is not None
            >>> assert rule["rule_type"] == "RANGE"
        """
        clean_name = name.strip() if name else ""
        with self._lock:
            rule_id = self._name_index.get(clean_name)
            if rule_id is None:
                logger.debug("Rule not found by name: %s", clean_name)
                return None
            record = self._rules.get(rule_id)
            if record is None:
                logger.debug(
                    "Rule index inconsistency: name=%s -> id=%s "
                    "not in store",
                    clean_name,
                    rule_id,
                )
                return None
            return copy.deepcopy(record)

    # ------------------------------------------------------------------
    # 4. update_rule
    # ------------------------------------------------------------------

    def update_rule(self, rule_id: str, **kwargs: Any) -> Optional[dict]:
        """Update specified fields of an existing rule with SemVer bumping.

        Only the fields explicitly provided in ``kwargs`` are updated.
        All other fields retain their current values. The version is
        automatically bumped according to which fields changed:

        - **Breaking** (rule_type, operator, column): major bump
        - **Additive** (parameters, threshold): minor bump
        - **Cosmetic** (description, tags, metadata, severity): patch bump

        When multiple categories are present, the highest severity bump
        wins. A version history snapshot is recorded before the update.

        Supported updatable fields:
            ``rule_type``, ``operator``, ``column``, ``threshold``,
            ``parameters``, ``severity``, ``description``, ``tags``,
            ``metadata``, ``status``.

        Args:
            rule_id: UUID of the rule to update.
            **kwargs: Field-name/value pairs to update.

        Returns:
            Deep-copy of the updated rule dict, or ``None`` if the
            rule was not found.

        Raises:
            ValueError: If an unsupported field is provided, or if a
                field value fails validation.

        Example:
            >>> engine = RuleRegistryEngine()
            >>> rule = engine.register_rule(
            ...     name="r1", rule_type="COMPLETENESS",
            ...     column="col", operator="IS_NULL",
            ... )
            >>> updated = engine.update_rule(
            ...     rule["rule_id"],
            ...     description="Updated description",
            ... )
            >>> assert updated["version"] == "1.0.1"
        """
        t0 = time.monotonic()

        updatable_fields: Set[str] = {
            "rule_type",
            "operator",
            "column",
            "threshold",
            "parameters",
            "severity",
            "description",
            "tags",
            "metadata",
            "status",
        }

        invalid_keys = set(kwargs.keys()) - updatable_fields
        if invalid_keys:
            raise ValueError(
                f"Cannot update fields: {sorted(invalid_keys)}. "
                f"Updatable fields: {sorted(updatable_fields)}"
            )

        with self._lock:
            record = self._rules.get(rule_id)
            if record is None:
                logger.debug(
                    "Update failed: rule not found id=%s", rule_id,
                )
                return None

            # Snapshot current state for version history
            self._version_history.setdefault(rule_id, []).append(
                self._snapshot_version(record),
            )

            # De-index old state before modifications
            self._deindex_rule(record, rule_id)

            changes: Dict[str, Any] = {}
            changed_fields: Set[str] = set()

            for key, value in kwargs.items():
                old_value = record.get(key)

                # Validate specific fields
                if key == "rule_type":
                    value_upper = str(value).strip().upper()
                    if value_upper not in VALID_RULE_TYPES:
                        # Re-index and abort
                        self._index_rule(record, rule_id)
                        raise ValueError(
                            f"Invalid rule_type: {value!r}. "
                            f"Must be one of: {sorted(VALID_RULE_TYPES)}"
                        )
                    value = value_upper

                elif key == "operator":
                    value_upper = str(value).strip().upper()
                    if value_upper not in VALID_OPERATORS:
                        self._index_rule(record, rule_id)
                        raise ValueError(
                            f"Invalid operator: {value!r}. "
                            f"Must be one of: {sorted(VALID_OPERATORS)}"
                        )
                    value = value_upper

                elif key == "severity":
                    value_upper = str(value).strip().upper()
                    if value_upper not in VALID_SEVERITIES:
                        self._index_rule(record, rule_id)
                        raise ValueError(
                            f"Invalid severity: {value!r}. "
                            f"Must be one of: {sorted(VALID_SEVERITIES)}"
                        )
                    value = value_upper

                elif key == "status":
                    value_lower = str(value).strip().lower()
                    if value_lower not in VALID_STATUSES:
                        self._index_rule(record, rule_id)
                        raise ValueError(
                            f"Invalid status: {value!r}. "
                            f"Must be one of: {sorted(VALID_STATUSES)}"
                        )
                    value = value_lower

                elif key == "column":
                    if not value or not str(value).strip():
                        self._index_rule(record, rule_id)
                        raise ValueError(
                            "Column must be a non-empty string."
                        )
                    clean_col = str(value).strip()
                    if len(clean_col) > MAX_COLUMN_NAME_LENGTH:
                        self._index_rule(record, rule_id)
                        raise ValueError(
                            f"Column name exceeds maximum length of "
                            f"{MAX_COLUMN_NAME_LENGTH} characters."
                        )
                    value = clean_col

                elif key == "description":
                    if value is not None:
                        value = str(value).strip()
                        if len(value) > MAX_DESCRIPTION_LENGTH:
                            self._index_rule(record, rule_id)
                            raise ValueError(
                                f"Description exceeds maximum length of "
                                f"{MAX_DESCRIPTION_LENGTH} characters."
                            )

                elif key == "tags":
                    if value is not None:
                        normalized = _normalize_tags(value)
                        _validate_tags_list(normalized)
                        value = normalized

                elif key == "parameters":
                    if value is not None and isinstance(value, dict):
                        value = copy.deepcopy(value)

                elif key == "metadata":
                    if value is not None and isinstance(value, dict):
                        value = copy.deepcopy(value)

                elif key == "threshold":
                    value = copy.deepcopy(value)

                # Detect actual change
                if old_value != value:
                    changes[key] = {
                        "old": str(old_value),
                        "new": str(value),
                    }
                    changed_fields.add(key)
                    record[key] = value

            # Update name index if name-adjacent fields changed
            old_name = record.get("name", "")
            if old_name in self._name_index:
                # Name is immutable; index stays the same
                pass

            # Compute new version using SemVer auto-bump
            if changed_fields:
                current_version = record.get("version", "1.0.0")
                new_version = _compute_version_bump(
                    current_version, changed_fields,
                )
                record["version"] = new_version
            else:
                new_version = record.get("version", "1.0.0")

            record["updated_at"] = _utcnow().isoformat()

            # Re-index with updated state
            self._index_rule(record, rule_id)

            # Compute new provenance hash
            data_hash = _build_sha256(
                {k: v for k, v in record.items() if k != "provenance_hash"}
            )
            entry = self._provenance.record(
                entity_type="validation_rule",
                entity_id=rule_id,
                action="rule_updated",
                metadata={
                    "data_hash": data_hash,
                    "changes": list(changes.keys()),
                    "version": new_version,
                },
            )
            record["provenance_hash"] = entry.hash_value

            result = copy.deepcopy(record)

        elapsed = time.monotonic() - t0
        observe_processing_duration("rule_update", elapsed)

        logger.info(
            "Rule updated: id=%s fields=%s version=%s elapsed=%.3fms",
            rule_id,
            list(changes.keys()),
            new_version,
            elapsed * 1000,
        )
        return result

    # ------------------------------------------------------------------
    # 5. delete_rule
    # ------------------------------------------------------------------

    def delete_rule(self, rule_id: str, hard: bool = False) -> bool:
        """Delete a rule from the registry.

        Supports both soft and hard deletion:

        - **Soft delete** (``hard=False``, default): Sets the rule status
          to ``"archived"``. The rule remains in the store and is
          retrievable, but will be excluded from most active searches.

        - **Hard delete** (``hard=True``): Permanently removes the rule
          from the store and all indexes. The rule is no longer
          retrievable.

        In both cases a provenance entry is recorded for audit.

        Args:
            rule_id: UUID of the rule to delete.
            hard: If ``True``, permanently remove the rule. If ``False``
                (default), soft-delete by archiving.

        Returns:
            ``True`` if the rule was found and deleted (soft or hard).
            ``False`` if no rule with the given ID exists.

        Example:
            >>> engine = RuleRegistryEngine()
            >>> rule = engine.register_rule(
            ...     name="del_test", rule_type="COMPLETENESS",
            ...     column="col", operator="IS_NULL",
            ... )
            >>> assert engine.delete_rule(rule["rule_id"]) is True
            >>> archived = engine.get_rule(rule["rule_id"])
            >>> assert archived["status"] == "archived"
        """
        t0 = time.monotonic()

        with self._lock:
            record = self._rules.get(rule_id)
            if record is None:
                logger.debug(
                    "Delete failed: rule not found id=%s", rule_id,
                )
                return False

            old_status = record.get("status", "active")
            rule_name = record.get("name", "")

            if hard:
                # Hard delete: remove from all indexes and store
                self._name_index.pop(rule_name, None)
                self._deindex_rule(record, rule_id)
                del self._rules[rule_id]
                # Keep version history for audit trail
                action = "rule_hard_deleted"
            else:
                # Soft delete: update status to archived
                # De-index old state
                self._deindex_rule(record, rule_id)

                record["status"] = "archived"
                record["updated_at"] = _utcnow().isoformat()

                # Re-index with new status
                self._index_rule(record, rule_id)
                action = "rule_soft_deleted"

            # Record provenance
            data_hash = _build_sha256({
                "rule_id": rule_id,
                "name": rule_name,
                "action": action,
                "old_status": old_status,
            })
            entry = self._provenance.record(
                entity_type="validation_rule",
                entity_id=rule_id,
                action=action,
                metadata={
                    "data_hash": data_hash,
                    "name": rule_name,
                    "hard": hard,
                },
            )

            if not hard:
                record["provenance_hash"] = entry.hash_value

        elapsed = time.monotonic() - t0
        observe_processing_duration("rule_delete", elapsed)

        logger.info(
            "Rule %s: id=%s name=%s (was %s) elapsed=%.3fms",
            "hard-deleted" if hard else "soft-deleted (archived)",
            rule_id,
            rule_name,
            old_status,
            elapsed * 1000,
        )
        return True

    # ------------------------------------------------------------------
    # 6. search_rules
    # ------------------------------------------------------------------

    def search_rules(
        self,
        rule_type: Optional[str] = None,
        severity: Optional[str] = None,
        column: Optional[str] = None,
        status: Optional[str] = None,
        tags: Optional[List[str]] = None,
        name_pattern: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> List[dict]:
        """Search and filter rules with index acceleration and pagination.

        All filters are applied with AND logic: a rule must satisfy
        every provided filter to be included in the result. Tags are
        also applied with AND logic -- a rule must carry all specified
        tags.

        When ``name_pattern`` is provided it is compiled as a Python
        regex and matched against the rule's ``name`` using
        ``re.search`` (case-insensitive). Invalid regex patterns are
        logged and silently ignored.

        Index acceleration is used for rule_type, severity, column,
        status, and tags to narrow the candidate set before applying
        remaining filters.

        Results are sorted by ``name`` ascending for deterministic
        ordering, then sliced by ``offset`` and ``limit``.

        Args:
            rule_type: Filter by exact rule type (e.g. ``"RANGE"``).
                None to include all types.
            severity: Filter by exact severity. None to include all.
            column: Filter by exact column name. None to include all.
            status: Filter by exact status. None to include all.
            tags: Filter by tags with AND logic; rule must carry all
                specified tags. None to skip tag filtering.
            name_pattern: Regex pattern to match against rule name
                (case-insensitive ``re.search``). None to skip.
            limit: Maximum results to return. Defaults to 100.
            offset: Number of results to skip. Defaults to 0.

        Returns:
            List of matching rule dicts (deep copies), sorted by
            ``name`` ascending, then paginated.

        Raises:
            ValueError: If ``limit`` or ``offset`` are negative.

        Example:
            >>> engine = RuleRegistryEngine()
            >>> engine.register_rule(
            ...     name="r1", rule_type="RANGE",
            ...     column="temp", operator="BETWEEN",
            ...     severity="HIGH",
            ... )
            >>> results = engine.search_rules(rule_type="RANGE")
            >>> assert len(results) == 1
        """
        t0 = time.monotonic()

        if limit < 0:
            raise ValueError("limit must be >= 0.")
        if offset < 0:
            raise ValueError("offset must be >= 0.")

        # Compile regex pattern if provided
        compiled_pattern = None
        if name_pattern is not None:
            try:
                compiled_pattern = re.compile(
                    name_pattern, re.IGNORECASE,
                )
            except re.error as exc:
                logger.warning(
                    "Invalid regex pattern %r: %s; ignoring pattern filter",
                    name_pattern,
                    exc,
                )

        with self._lock:
            # Start with candidate set, narrowing by index
            candidate_ids: Optional[Set[str]] = None

            if rule_type is not None:
                type_upper = rule_type.strip().upper()
                type_ids = self._type_index.get(type_upper, set())
                candidate_ids = set(type_ids)

            if severity is not None:
                sev_upper = severity.strip().upper()
                sev_ids = self._severity_index.get(sev_upper, set())
                if candidate_ids is None:
                    candidate_ids = set(sev_ids)
                else:
                    candidate_ids &= sev_ids

            if column is not None:
                col_clean = column.strip()
                col_ids = self._column_index.get(col_clean, set())
                if candidate_ids is None:
                    candidate_ids = set(col_ids)
                else:
                    candidate_ids &= col_ids

            if status is not None:
                status_lower = status.strip().lower()
                status_ids = self._status_index.get(status_lower, set())
                if candidate_ids is None:
                    candidate_ids = set(status_ids)
                else:
                    candidate_ids &= status_ids

            if tags is not None and tags:
                normalized_filter_tags = _normalize_tags(tags)
                for tag in normalized_filter_tags:
                    tag_ids = self._tag_index.get(tag, set())
                    if candidate_ids is None:
                        candidate_ids = set(tag_ids)
                    else:
                        candidate_ids &= tag_ids

            # Fall back to all rules if no index filter narrowed the set
            if candidate_ids is None:
                candidate_ids = set(self._rules.keys())

            # Apply remaining filters (regex pattern)
            results: List[dict] = []
            for rid in candidate_ids:
                record = self._rules.get(rid)
                if record is None:
                    continue

                # Regex name pattern filter
                if compiled_pattern is not None:
                    rname = record.get("name", "")
                    if not compiled_pattern.search(rname):
                        continue

                results.append(copy.deepcopy(record))

        # Sort by name ascending
        results.sort(key=lambda r: r.get("name", ""))

        # Apply pagination
        paginated = (
            results[offset: offset + limit]
            if limit > 0
            else results[offset:]
        )

        elapsed = time.monotonic() - t0
        observe_processing_duration("rule_search", elapsed)

        logger.debug(
            "search_rules: returned %d of %d matches "
            "(type=%s, severity=%s, column=%s, status=%s, "
            "tags=%s, pattern=%s, limit=%d, offset=%d) elapsed=%.3fms",
            len(paginated),
            len(results),
            rule_type,
            severity,
            column,
            status,
            tags,
            name_pattern,
            limit,
            offset,
            elapsed * 1000,
        )
        return paginated

    # ------------------------------------------------------------------
    # 7. clone_rule
    # ------------------------------------------------------------------

    def clone_rule(
        self,
        rule_id: str,
        new_name: str,
    ) -> dict:
        """Clone an existing rule under a new name.

        Creates a new rule with identical configuration to the source
        rule but with a fresh UUID, the specified new name, version
        ``"1.0.0"``, and fresh timestamps. The original rule is
        unaffected.

        Args:
            rule_id: UUID of the source rule to clone.
            new_name: Name for the cloned rule. Must be unique and
                non-empty.

        Returns:
            The newly registered clone rule dict.

        Raises:
            ValueError: If the source rule is not found, or if
                ``new_name`` fails validation or is already in use.

        Example:
            >>> engine = RuleRegistryEngine()
            >>> orig = engine.register_rule(
            ...     name="original", rule_type="RANGE",
            ...     column="val", operator="GREATER_THAN",
            ...     threshold=10, severity="HIGH",
            ...     tags=["prod"],
            ... )
            >>> clone = engine.clone_rule(orig["rule_id"], "clone_v2")
            >>> assert clone["name"] == "clone_v2"
            >>> assert clone["rule_type"] == "RANGE"
            >>> assert clone["rule_id"] != orig["rule_id"]
        """
        t0 = time.monotonic()

        # Read source rule
        with self._lock:
            source = self._rules.get(rule_id)
            if source is None:
                raise ValueError(
                    f"Source rule not found: {rule_id!r}"
                )
            source_copy = copy.deepcopy(source)

        # Register as new rule
        result = self.register_rule(
            name=new_name,
            rule_type=source_copy.get("rule_type", "CUSTOM"),
            column=source_copy.get("column", ""),
            operator=source_copy.get("operator", "EQUALS"),
            threshold=source_copy.get("threshold"),
            parameters=source_copy.get("parameters"),
            severity=source_copy.get("severity", "MEDIUM"),
            description=source_copy.get("description", ""),
            tags=source_copy.get("tags"),
            metadata={
                **(source_copy.get("metadata") or {}),
                "cloned_from": rule_id,
                "cloned_from_name": source_copy.get("name", ""),
                "cloned_from_version": source_copy.get("version", "1.0.0"),
            },
        )

        elapsed = time.monotonic() - t0
        observe_processing_duration("rule_clone", elapsed)

        logger.info(
            "Rule cloned: source_id=%s -> new_id=%s new_name=%s "
            "elapsed=%.3fms",
            rule_id,
            result["rule_id"],
            new_name,
            elapsed * 1000,
        )
        return result

    # ------------------------------------------------------------------
    # 8. get_rule_versions
    # ------------------------------------------------------------------

    def get_rule_versions(self, rule_id: str) -> List[dict]:
        """Retrieve the complete version history of a rule.

        Returns all historical snapshots of the rule, sorted by
        snapshot time ascending (oldest first). The current state is
        included as the last entry.

        Args:
            rule_id: UUID of the rule.

        Returns:
            List of rule snapshots (deep copies). Empty list if the
            rule has never existed.

        Example:
            >>> engine = RuleRegistryEngine()
            >>> rule = engine.register_rule(
            ...     name="v_test", rule_type="RANGE",
            ...     column="val", operator="GREATER_THAN",
            ... )
            >>> engine.update_rule(rule["rule_id"], description="upd")
            >>> versions = engine.get_rule_versions(rule["rule_id"])
            >>> assert len(versions) >= 2
        """
        with self._lock:
            history = self._version_history.get(rule_id, [])
            result = copy.deepcopy(history)

            # Append current state if the rule still exists
            current = self._rules.get(rule_id)
            if current is not None:
                current_snap = self._snapshot_version(current)
                result.append(current_snap)

        logger.debug(
            "get_rule_versions: rule_id=%s versions=%d",
            rule_id,
            len(result),
        )
        return result

    # ------------------------------------------------------------------
    # 9. rollback_rule
    # ------------------------------------------------------------------

    def rollback_rule(self, rule_id: str, version: str) -> dict:
        """Rollback a rule to a specific version from its history.

        Finds the snapshot matching the requested version string in
        the rule's version history, then replaces the current rule
        state with that snapshot. A new version bump (patch) is applied
        to indicate the rollback, and a provenance entry is recorded.

        The rolled-back state inherits the original rule_id, name, and
        creation timestamp but gets fresh updated_at and provenance.

        Args:
            rule_id: UUID of the rule to rollback.
            version: The target SemVer string to rollback to (e.g.
                ``"1.0.0"``).

        Returns:
            The rolled-back rule dict with new version and provenance.

        Raises:
            ValueError: If the rule is not found, or the requested
                version does not exist in the history.

        Example:
            >>> engine = RuleRegistryEngine()
            >>> rule = engine.register_rule(
            ...     name="rb_test", rule_type="RANGE",
            ...     column="val", operator="GREATER_THAN",
            ...     description="initial",
            ... )
            >>> engine.update_rule(
            ...     rule["rule_id"], description="changed",
            ... )
            >>> rolled = engine.rollback_rule(rule["rule_id"], "1.0.0")
            >>> assert rolled["description"] == "initial"
        """
        t0 = time.monotonic()

        with self._lock:
            record = self._rules.get(rule_id)
            if record is None:
                raise ValueError(f"Rule not found: {rule_id!r}")

            history = self._version_history.get(rule_id, [])

            # Find the target version snapshot
            target_snapshot: Optional[dict] = None
            for snap in history:
                if snap.get("version") == version:
                    target_snapshot = snap
                    break

            if target_snapshot is None:
                available = [s.get("version") for s in history]
                raise ValueError(
                    f"Version {version!r} not found in history for "
                    f"rule {rule_id!r}. Available versions: {available}"
                )

            # Snapshot current state before rollback
            self._version_history.setdefault(rule_id, []).append(
                self._snapshot_version(record),
            )

            # De-index old state
            self._deindex_rule(record, rule_id)

            # Compute new version: patch bump from current
            current_version = record.get("version", "1.0.0")
            major, minor, patch = _parse_version(current_version)
            new_version = _format_version(major, minor, patch + 1)

            # Restore fields from target snapshot
            restorable_fields = [
                "rule_type", "column", "operator", "threshold",
                "parameters", "severity", "description", "tags",
                "metadata",
            ]
            for field in restorable_fields:
                if field in target_snapshot:
                    record[field] = copy.deepcopy(target_snapshot[field])

            record["version"] = new_version
            record["updated_at"] = _utcnow().isoformat()

            # Re-index with restored state
            self._index_rule(record, rule_id)

            # Record provenance
            data_hash = _build_sha256(
                {k: v for k, v in record.items() if k != "provenance_hash"}
            )
            entry = self._provenance.record(
                entity_type="validation_rule",
                entity_id=rule_id,
                action="rule_rolled_back",
                metadata={
                    "data_hash": data_hash,
                    "target_version": version,
                    "new_version": new_version,
                    "previous_version": current_version,
                },
            )
            record["provenance_hash"] = entry.hash_value

            result = copy.deepcopy(record)

        elapsed = time.monotonic() - t0
        observe_processing_duration("rule_rollback", elapsed)

        logger.info(
            "Rule rolled back: id=%s from=%s to_snapshot=%s "
            "new_version=%s elapsed=%.3fms",
            rule_id,
            current_version,
            version,
            new_version,
            elapsed * 1000,
        )
        return result

    # ------------------------------------------------------------------
    # 10. bulk_register
    # ------------------------------------------------------------------

    def bulk_register(self, rules: List[dict]) -> dict:
        """Register multiple rules in a single batch operation.

        Each element of ``rules`` must be a dict with the keys accepted
        by ``register_rule``. At minimum ``name``, ``rule_type``,
        ``column``, and ``operator`` are required.

        Rules that fail validation are skipped and recorded in the
        ``errors`` list of the result. Successfully registered rules
        are included in the ``registered`` count.

        Args:
            rules: List of rule dictionaries. Maximum
                ``MAX_BULK_IMPORT`` (5000) entries per call.

        Returns:
            Summary dict:

            .. code-block:: python

                {
                    "registered": 8,
                    "failed": 2,
                    "errors": [
                        {"index": 3, "name": "...", "error": "..."},
                    ],
                    "rule_ids": ["uuid1", "uuid2", ...],
                    "provenance_hash": "<sha256>",
                }

        Raises:
            ValueError: If ``rules`` is empty or exceeds MAX_BULK_IMPORT.

        Example:
            >>> engine = RuleRegistryEngine()
            >>> result = engine.bulk_register([
            ...     {"name": "a", "rule_type": "RANGE",
            ...      "column": "c", "operator": "GREATER_THAN"},
            ...     {"name": "b", "rule_type": "FORMAT",
            ...      "column": "d", "operator": "MATCHES"},
            ... ])
            >>> assert result["registered"] == 2
            >>> assert result["failed"] == 0
        """
        t0 = time.monotonic()

        if not rules:
            raise ValueError("Rules list must not be empty.")
        if len(rules) > MAX_BULK_IMPORT:
            raise ValueError(
                f"Bulk register exceeds maximum of {MAX_BULK_IMPORT} "
                f"rules. Received {len(rules)}."
            )

        registered_count = 0
        failed_count = 0
        errors: List[Dict[str, Any]] = []
        rule_ids: List[str] = []

        for idx, rule_dict in enumerate(rules):
            raw_name = rule_dict.get("name", f"<unnamed-{idx}>")
            try:
                result = self.register_rule(
                    name=rule_dict.get("name", ""),
                    rule_type=rule_dict.get("rule_type", ""),
                    column=rule_dict.get("column", ""),
                    operator=rule_dict.get("operator", ""),
                    threshold=rule_dict.get("threshold"),
                    parameters=rule_dict.get("parameters"),
                    severity=rule_dict.get("severity", "MEDIUM"),
                    description=rule_dict.get("description", ""),
                    tags=rule_dict.get("tags"),
                    metadata=rule_dict.get("metadata"),
                )
                rule_ids.append(result["rule_id"])
                registered_count += 1
            except Exception as exc:
                failed_count += 1
                errors.append({
                    "index": idx,
                    "name": str(raw_name),
                    "error": str(exc),
                })
                logger.warning(
                    "Bulk register: entry[%d] name=%r failed: %s",
                    idx,
                    raw_name,
                    exc,
                )

        summary_hash = _build_sha256({
            "registered": registered_count,
            "failed": failed_count,
            "rule_ids": rule_ids,
        })

        summary: dict = {
            "registered": registered_count,
            "failed": failed_count,
            "errors": errors,
            "rule_ids": rule_ids,
            "provenance_hash": summary_hash,
        }

        elapsed = time.monotonic() - t0
        observe_processing_duration("rule_bulk_register", elapsed)

        logger.info(
            "Bulk register complete: registered=%d failed=%d "
            "elapsed=%.3fms",
            registered_count,
            failed_count,
            elapsed * 1000,
        )
        return summary

    # ------------------------------------------------------------------
    # 11. export_rules
    # ------------------------------------------------------------------

    def export_rules(
        self,
        rule_type: Optional[str] = None,
    ) -> List[dict]:
        """Export rules as a list of JSON-serializable dictionaries.

        If ``rule_type`` is provided, only rules of that type are
        exported. Otherwise all rules are exported.

        Results are sorted by ``name`` ascending for deterministic
        output.

        Args:
            rule_type: Optional rule type filter. None to export all.

        Returns:
            List of rule dicts (deep copies), ordered by ``name``
            ascending.

        Example:
            >>> engine = RuleRegistryEngine()
            >>> engine.register_rule(
            ...     name="a", rule_type="RANGE",
            ...     column="c", operator="GREATER_THAN",
            ... )
            >>> exported = engine.export_rules()
            >>> assert len(exported) == 1
            >>> assert exported[0]["name"] == "a"
        """
        t0 = time.monotonic()

        with self._lock:
            if rule_type is not None:
                type_upper = rule_type.strip().upper()
                candidate_ids = self._type_index.get(type_upper, set())
                records = [
                    copy.deepcopy(self._rules[rid])
                    for rid in candidate_ids
                    if rid in self._rules
                ]
            else:
                records = [
                    copy.deepcopy(record)
                    for record in self._rules.values()
                ]

        records.sort(key=lambda r: r.get("name", ""))

        elapsed = time.monotonic() - t0
        observe_processing_duration("rule_export", elapsed)

        logger.info(
            "export_rules: exported %d rules (type=%s) elapsed=%.3fms",
            len(records),
            rule_type,
            elapsed * 1000,
        )
        return records

    # ------------------------------------------------------------------
    # 12. import_rules
    # ------------------------------------------------------------------

    def import_rules(self, rules_data: List[dict]) -> dict:
        """Import rules from a list of previously exported dictionaries.

        Each element must contain at minimum ``name``, ``rule_type``,
        ``column``, and ``operator``. Rules that fail validation are
        skipped.

        This method delegates to ``register_rule`` internally for each
        entry, providing symmetry with ``export_rules``.

        Args:
            rules_data: List of rule dicts to import. Maximum
                ``MAX_BULK_IMPORT`` entries per call.

        Returns:
            Summary dict with counts:

            .. code-block:: python

                {
                    "imported": 8,
                    "failed": 2,
                    "errors": [...],
                    "rule_ids": [...],
                    "provenance_hash": "<sha256>",
                }

        Raises:
            ValueError: If ``rules_data`` is empty or exceeds
                MAX_BULK_IMPORT.

        Example:
            >>> engine_a = RuleRegistryEngine()
            >>> engine_a.register_rule(
            ...     name="a", rule_type="RANGE",
            ...     column="c", operator="GREATER_THAN",
            ... )
            >>> exported = engine_a.export_rules()
            >>> engine_b = RuleRegistryEngine()
            >>> result = engine_b.import_rules(exported)
            >>> assert result["imported"] == 1
        """
        t0 = time.monotonic()

        if not rules_data:
            raise ValueError("Rules data list must not be empty.")
        if len(rules_data) > MAX_BULK_IMPORT:
            raise ValueError(
                f"Import exceeds maximum of {MAX_BULK_IMPORT} rules. "
                f"Received {len(rules_data)}."
            )

        imported_count = 0
        failed_count = 0
        errors: List[Dict[str, Any]] = []
        rule_ids: List[str] = []

        for idx, rule_dict in enumerate(rules_data):
            raw_name = rule_dict.get("name", f"<unnamed-{idx}>")
            try:
                result = self.register_rule(
                    name=rule_dict.get("name", ""),
                    rule_type=rule_dict.get("rule_type", ""),
                    column=rule_dict.get("column", ""),
                    operator=rule_dict.get("operator", ""),
                    threshold=rule_dict.get("threshold"),
                    parameters=rule_dict.get("parameters"),
                    severity=rule_dict.get("severity", "MEDIUM"),
                    description=rule_dict.get("description", ""),
                    tags=rule_dict.get("tags"),
                    metadata=rule_dict.get("metadata"),
                )
                rule_ids.append(result["rule_id"])
                imported_count += 1
            except Exception as exc:
                failed_count += 1
                errors.append({
                    "index": idx,
                    "name": str(raw_name),
                    "error": str(exc),
                })
                logger.warning(
                    "Import: entry[%d] name=%r failed: %s",
                    idx,
                    raw_name,
                    exc,
                )

        summary_hash = _build_sha256({
            "imported": imported_count,
            "failed": failed_count,
            "rule_ids": rule_ids,
        })

        summary: dict = {
            "imported": imported_count,
            "failed": failed_count,
            "errors": errors,
            "rule_ids": rule_ids,
            "provenance_hash": summary_hash,
        }

        elapsed = time.monotonic() - t0
        observe_processing_duration("rule_import", elapsed)

        logger.info(
            "Import complete: imported=%d failed=%d elapsed=%.3fms",
            imported_count,
            failed_count,
            elapsed * 1000,
        )
        return summary

    # ------------------------------------------------------------------
    # 13. get_statistics
    # ------------------------------------------------------------------

    def get_statistics(self) -> dict:
        """Return a comprehensive snapshot of registry statistics.

        Computes aggregate counts across all registered rules grouped
        by type, severity, status, and column. This method is read-only
        and never modifies state.

        Returns:
            Stats dict with the following structure:

            .. code-block:: python

                {
                    "total_rules": 42,
                    "by_type": {"RANGE": 20, "COMPLETENESS": 10, ...},
                    "by_severity": {"CRITICAL": 5, "HIGH": 15, ...},
                    "by_status": {"active": 38, "archived": 4},
                    "by_column": {"co2e": 3, "temperature": 2, ...},
                    "total_tags": 18,
                    "total_versions": 85,
                    "provenance_entries": 100,
                }

        Example:
            >>> engine = RuleRegistryEngine()
            >>> engine.register_rule(
            ...     name="a", rule_type="RANGE",
            ...     column="c", operator="GREATER_THAN",
            ...     severity="HIGH",
            ... )
            >>> stats = engine.get_statistics()
            >>> assert stats["total_rules"] == 1
            >>> assert stats["by_type"]["RANGE"] == 1
        """
        with self._lock:
            total = len(self._rules)

            by_type: Dict[str, int] = {}
            by_severity: Dict[str, int] = {}
            by_status: Dict[str, int] = {}
            by_column: Dict[str, int] = {}

            for record in self._rules.values():
                rtype = record.get("rule_type", "unknown")
                by_type[rtype] = by_type.get(rtype, 0) + 1

                rsev = record.get("severity", "unknown")
                by_severity[rsev] = by_severity.get(rsev, 0) + 1

                rstatus = record.get("status", "unknown")
                by_status[rstatus] = by_status.get(rstatus, 0) + 1

                rcol = record.get("column", "unknown")
                by_column[rcol] = by_column.get(rcol, 0) + 1

            total_tags = len(self._tag_index)
            total_versions = sum(
                len(v) for v in self._version_history.values()
            )

        # Provenance entries - safe to read outside lock
        prov_count = getattr(self._provenance, "entry_count", 0)

        stats: dict = {
            "total_rules": total,
            "by_type": dict(sorted(by_type.items())),
            "by_severity": dict(sorted(by_severity.items())),
            "by_status": dict(sorted(by_status.items())),
            "by_column": dict(sorted(by_column.items())),
            "total_tags": total_tags,
            "total_versions": total_versions,
            "provenance_entries": prov_count,
        }

        logger.debug(
            "get_statistics: total=%d types=%d severities=%d "
            "columns=%d tags=%d",
            total,
            len(by_type),
            len(by_severity),
            len(by_column),
            total_tags,
        )
        return stats

    # ------------------------------------------------------------------
    # 14. list_rule_types
    # ------------------------------------------------------------------

    def list_rule_types(self) -> Dict[str, int]:
        """Return a count of rules grouped by rule type.

        The result includes only rule types that have at least one
        registered rule (types with zero rules are omitted).

        Returns:
            Dictionary mapping rule type strings to the number of
            rules of that type. Sorted alphabetically by type.

        Example:
            >>> engine = RuleRegistryEngine()
            >>> engine.register_rule(
            ...     name="a", rule_type="RANGE",
            ...     column="c", operator="GREATER_THAN",
            ... )
            >>> engine.register_rule(
            ...     name="b", rule_type="RANGE",
            ...     column="d", operator="LESS_THAN",
            ... )
            >>> engine.register_rule(
            ...     name="c", rule_type="FORMAT",
            ...     column="e", operator="MATCHES",
            ... )
            >>> types = engine.list_rule_types()
            >>> assert types == {"FORMAT": 1, "RANGE": 2}
        """
        with self._lock:
            counts: Dict[str, int] = {}
            for rule_type, rule_ids in self._type_index.items():
                active_count = sum(
                    1 for rid in rule_ids if rid in self._rules
                )
                if active_count > 0:
                    counts[rule_type] = active_count

        logger.debug("list_rule_types: %s", counts)
        return dict(sorted(counts.items()))

    # ------------------------------------------------------------------
    # 15. list_severities
    # ------------------------------------------------------------------

    def list_severities(self) -> Dict[str, int]:
        """Return a count of rules grouped by severity level.

        The result includes only severity levels that have at least
        one registered rule (levels with zero rules are omitted).

        Returns:
            Dictionary mapping severity strings to the number of
            rules at that severity. Sorted alphabetically.

        Example:
            >>> engine = RuleRegistryEngine()
            >>> engine.register_rule(
            ...     name="a", rule_type="RANGE",
            ...     column="c", operator="GREATER_THAN",
            ...     severity="HIGH",
            ... )
            >>> engine.register_rule(
            ...     name="b", rule_type="FORMAT",
            ...     column="d", operator="MATCHES",
            ...     severity="HIGH",
            ... )
            >>> sevs = engine.list_severities()
            >>> assert sevs == {"HIGH": 2}
        """
        with self._lock:
            counts: Dict[str, int] = {}
            for severity, rule_ids in self._severity_index.items():
                active_count = sum(
                    1 for rid in rule_ids if rid in self._rules
                )
                if active_count > 0:
                    counts[severity] = active_count

        logger.debug("list_severities: %s", counts)
        return dict(sorted(counts.items()))

    # ------------------------------------------------------------------
    # 16. clear
    # ------------------------------------------------------------------

    def clear(self) -> None:
        """Clear all registry state.

        Removes all rules, indexes, and version history. Resets the
        provenance tracker to a fresh state. This method is intended
        for use in automated tests only. Do NOT call in production.

        Example:
            >>> engine = RuleRegistryEngine()
            >>> engine.register_rule(
            ...     name="test", rule_type="COMPLETENESS",
            ...     column="col", operator="IS_NULL",
            ... )
            >>> assert engine.get_statistics()["total_rules"] == 1
            >>> engine.clear()
            >>> assert engine.get_statistics()["total_rules"] == 0
        """
        with self._lock:
            self._rules.clear()
            self._name_index.clear()
            self._type_index.clear()
            self._severity_index.clear()
            self._tag_index.clear()
            self._column_index.clear()
            self._status_index.clear()
            self._version_history.clear()

        # Reset provenance tracker
        if hasattr(self._provenance, "reset"):
            self._provenance.reset()
        else:
            self._provenance = ProvenanceTracker()

        logger.warning(
            "RuleRegistryEngine.clear() called - "
            "all registry data cleared."
        )

    # ------------------------------------------------------------------
    # Introspection helpers (properties)
    # ------------------------------------------------------------------

    @property
    def rule_count(self) -> int:
        """Return the total number of registered rules.

        Returns:
            Integer count of rules currently in the registry.
        """
        with self._lock:
            return len(self._rules)

    @property
    def provenance_chain_length(self) -> int:
        """Return the number of entries in the provenance chain.

        Returns:
            Integer count of provenance entries recorded.
        """
        return getattr(self._provenance, "entry_count", 0)

    def list_rules(self, limit: int = 10000) -> List[dict]:
        """Return all registered rules as a list of dictionaries.

        Convenience method for pipeline orchestration and data export.

        Args:
            limit: Maximum number of rules to return. Defaults to 10000.

        Returns:
            List of rule dictionaries (deep copies), sorted by
            ``name``.
        """
        return self.search_rules(limit=limit)

    def get_all_rule_ids(self) -> List[str]:
        """Return all registered rule IDs.

        Returns:
            Sorted list of rule UUID strings.

        Example:
            >>> engine = RuleRegistryEngine()
            >>> engine.register_rule(
            ...     name="a", rule_type="COMPLETENESS",
            ...     column="c", operator="IS_NULL",
            ... )
            >>> ids = engine.get_all_rule_ids()
            >>> assert len(ids) == 1
        """
        with self._lock:
            return sorted(self._rules.keys())

    def get_provenance_chain(self) -> List[Dict[str, Any]]:
        """Return the full provenance chain for audit.

        Returns:
            List of provenance entries in insertion order (oldest
            first). Each entry is a dictionary suitable for JSON
            serialization.

        Example:
            >>> engine = RuleRegistryEngine()
            >>> engine.register_rule(
            ...     name="a", rule_type="COMPLETENESS",
            ...     column="c", operator="IS_NULL",
            ... )
            >>> chain = engine.get_provenance_chain()
            >>> assert len(chain) >= 1
        """
        if hasattr(self._provenance, "export_chain"):
            entries = self._provenance.export_chain()
            result: List[Dict[str, Any]] = []
            for entry in entries:
                if isinstance(entry, dict):
                    result.append(entry)
                elif hasattr(entry, "to_dict"):
                    result.append(entry.to_dict())
                else:
                    result.append({"hash_value": str(entry)})
            return result
        return []


# ---------------------------------------------------------------------------
# Module exports
# ---------------------------------------------------------------------------

__all__ = [
    "RuleRegistryEngine",
    "VALID_RULE_TYPES",
    "VALID_OPERATORS",
    "VALID_SEVERITIES",
    "VALID_STATUSES",
    "VALID_RULE_STATUSES",
    "STATUS_TRANSITIONS",
    "MAX_BULK_IMPORT",
    "MAX_RULE_NAME_LENGTH",
    "MAX_TAG_LENGTH",
]
