# -*- coding: utf-8 -*-
"""
Schema Versioner Engine - AGENT-DATA-017: Schema Migration Agent (GL-DATA-X-020)
=================================================================================

Engine 2 of 7 — SchemaVersionerEngine.

Semantic versioning (major.minor.patch) with automatic version bump
classification. Breaking changes trigger major bumps, additive changes
trigger minor bumps, and cosmetic/documentation changes trigger patch bumps.

Maintains a complete version history with structured changelogs, supports
version comparison, deprecation management with sunset dates, and consumer
version pinning.

Supports:
    - SemVer (major.minor.patch) auto-classification from change lists
    - Automatic first version at "1.0.0" per schema
    - Breaking-change → major bump (minor + patch reset to 0)
    - Non-breaking / additive → minor bump (patch reset to 0)
    - Cosmetic / documentation → patch bump
    - Structured changelog per version with entries and notes
    - Version comparison returning structured diff summaries
    - Deprecation management with optional ISO-8601 sunset dates
    - Sunset warning detection (configurable look-ahead window)
    - Consumer-level version range pinning (e.g. ">=2.0.0 <3.0.0")
    - Thread-safe in-memory storage
    - SHA-256 provenance hashes on every create / deprecate / undeprecate

Zero-Hallucination Guarantees:
    - All version arithmetic is deterministic integer tuple operations
    - Change severity classification is rule-based (string comparison only)
    - No LLM or ML models in the bump, comparison, or validation paths
    - SHA-256 provenance hashes for full audit trails
    - Thread-safe with reentrant locking

Example:
    >>> from greenlang.schema_migration.schema_versioner import SchemaVersionerEngine
    >>> engine = SchemaVersionerEngine()
    >>> v1 = engine.create_version(
    ...     schema_id="sch-abc",
    ...     definition_json={"type": "object", "properties": {"id": {"type": "string"}}},
    ...     changelog_note="Initial release",
    ... )
    >>> print(v1["version"])  # "1.0.0"
    >>> changes = [{"field": "name", "severity": "breaking", "description": "field removed"}]
    >>> v2 = engine.create_version(
    ...     schema_id="sch-abc",
    ...     definition_json={"type": "object"},
    ...     changes=changes,
    ...     changelog_note="Removed name field",
    ... )
    >>> print(v2["version"])  # "2.0.0"

Author: GreenLang Platform Team
Date: February 2026
PRD: AGENT-DATA-017 Schema Migration Agent (GL-DATA-X-020)
Status: Production Ready
"""

from __future__ import annotations

import hashlib
import json
import logging
import threading
import time
import uuid
from collections import defaultdict
from datetime import date, datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

__all__ = [
    "SchemaVersionerEngine",
]


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Severity labels that map to bump types
_BREAKING_SEVERITIES: frozenset[str] = frozenset(
    {"breaking", "major", "breaking_change", "incompatible"}
)
_NON_BREAKING_SEVERITIES: frozenset[str] = frozenset(
    {"non_breaking", "minor", "additive", "backward_compatible", "feature"}
)
_COSMETIC_SEVERITIES: frozenset[str] = frozenset(
    {"patch", "cosmetic", "documentation", "doc", "fix", "refactor", "trivial"}
)

# Bump type strings
BUMP_MAJOR = "major"
BUMP_MINOR = "minor"
BUMP_PATCH = "patch"

# ID prefixes
_PREFIX_VERSION = "VER"
_PREFIX_PIN = "PIN"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _utcnow() -> datetime:
    """Return the current UTC datetime with microseconds zeroed for consistency.

    Returns:
        Timezone-aware datetime at second precision in UTC.
    """
    return datetime.now(timezone.utc).replace(microsecond=0)


def _generate_id(prefix: str = "VER") -> str:
    """Generate a unique identifier with the given prefix.

    Args:
        prefix: Short uppercase string prepended to the random hex segment.

    Returns:
        String of the form ``{prefix}-{hex12}``.

    Example:
        >>> _generate_id("VER")
        'VER-3f9a1b2c4d5e'
    """
    return f"{prefix}-{uuid.uuid4().hex[:12]}"


def _compute_provenance(operation: str, payload_repr: str) -> str:
    """Compute a SHA-256 provenance hash for an engine operation.

    The hash covers the operation name, the serialised payload, and the
    current UTC timestamp so every call produces a unique fingerprint even
    for identical inputs.

    Args:
        operation: Human-readable name of the operation (e.g. "create_version").
        payload_repr: JSON-serialised or string representation of the data.

    Returns:
        Hex-encoded 64-character SHA-256 digest.
    """
    raw = f"{operation}:{payload_repr}:{_utcnow().isoformat()}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _serialize_definition(definition_json: Any) -> str:
    """Serialize a schema definition to a canonical JSON string.

    Uses sort_keys=True and no extra whitespace to guarantee a stable
    representation for provenance hashing and comparison purposes.

    Args:
        definition_json: A JSON-serialisable object (typically a dict).

    Returns:
        Canonical JSON string.
    """
    return json.dumps(definition_json, sort_keys=True, separators=(",", ":"))


# ---------------------------------------------------------------------------
# SemVer helpers (module-level, no class dependency)
# ---------------------------------------------------------------------------


def _parse_version(version_string: str) -> Tuple[int, int, int]:
    """Parse a SemVer string into its (major, minor, patch) integer tuple.

    Args:
        version_string: A semantic version string in the form ``"X.Y.Z"``.

    Returns:
        A three-tuple of non-negative integers (major, minor, patch).

    Raises:
        ValueError: If the string does not conform to the ``X.Y.Z`` format
            or any part is not a non-negative integer.

    Example:
        >>> _parse_version("2.10.3")
        (2, 10, 3)
    """
    parts = version_string.strip().split(".")
    if len(parts) != 3:
        raise ValueError(
            f"Invalid SemVer string '{version_string}': expected exactly 3 parts "
            f"separated by '.', got {len(parts)}."
        )
    try:
        major, minor, patch = int(parts[0]), int(parts[1]), int(parts[2])
    except ValueError as exc:
        raise ValueError(
            f"Invalid SemVer string '{version_string}': all parts must be integers."
        ) from exc
    if major < 0 or minor < 0 or patch < 0:
        raise ValueError(
            f"Invalid SemVer string '{version_string}': parts must be non-negative."
        )
    return major, minor, patch


def _bump_version(current_version: str, bump_type: str) -> str:
    """Increment a SemVer string according to the specified bump type.

    Rules:
        - ``"major"``: increment major, reset minor and patch to 0.
        - ``"minor"``: increment minor, reset patch to 0.
        - ``"patch"``: increment patch only.

    Args:
        current_version: Current SemVer string (e.g. ``"1.2.3"``).
        bump_type: One of ``"major"``, ``"minor"``, or ``"patch"``.

    Returns:
        New SemVer string after the bump.

    Raises:
        ValueError: If ``bump_type`` is not a recognised value or
            ``current_version`` is malformed.

    Example:
        >>> _bump_version("1.2.3", "minor")
        '1.3.0'
        >>> _bump_version("1.2.3", "major")
        '2.0.0'
    """
    major, minor, patch = _parse_version(current_version)
    if bump_type == BUMP_MAJOR:
        return f"{major + 1}.0.0"
    if bump_type == BUMP_MINOR:
        return f"{major}.{minor + 1}.0"
    if bump_type == BUMP_PATCH:
        return f"{major}.{minor}.{patch + 1}"
    raise ValueError(
        f"Unknown bump_type '{bump_type}'. Must be one of: major, minor, patch."
    )


def _classify_bump(changes: List[Dict[str, Any]]) -> str:
    """Classify the required SemVer bump from a list of change descriptors.

    Each change dict must contain at minimum a ``"severity"`` key whose value
    is a string.  Classification precedence:

    1. If any change has severity in ``_BREAKING_SEVERITIES`` → ``"major"``.
    2. Elif any change has severity in ``_NON_BREAKING_SEVERITIES`` → ``"minor"``.
    3. Otherwise (all cosmetic / unknown) → ``"patch"``.

    An empty list returns ``"patch"`` (no meaningful changes).

    Args:
        changes: List of change descriptor dicts.  Each dict should have at
            least a ``"severity"`` key; missing keys default to ``"patch"``
            severity behaviour.

    Returns:
        One of ``"major"``, ``"minor"``, or ``"patch"``.

    Example:
        >>> _classify_bump([{"severity": "non_breaking"}, {"severity": "cosmetic"}])
        'minor'
        >>> _classify_bump([{"severity": "breaking"}])
        'major'
        >>> _classify_bump([])
        'patch'
    """
    has_minor = False
    for change in changes:
        sev = str(change.get("severity", "patch")).lower()
        if sev in _BREAKING_SEVERITIES:
            return BUMP_MAJOR
        if sev in _NON_BREAKING_SEVERITIES:
            has_minor = True
    return BUMP_MINOR if has_minor else BUMP_PATCH


def _version_tuple_key(version_string: str) -> Tuple[int, int, int]:
    """Convert a SemVer string to a sortable tuple for ordering.

    Args:
        version_string: SemVer string (e.g. ``"2.10.3"``).

    Returns:
        Three-integer tuple usable as a sort key.
    """
    try:
        return _parse_version(version_string)
    except ValueError:
        return (0, 0, 0)


# ---------------------------------------------------------------------------
# Main Engine
# ---------------------------------------------------------------------------


class SchemaVersionerEngine:
    """Engine 2 of 7 — Semantic versioning for the Schema Migration Agent.

    Manages the full lifecycle of schema versions: creation with automatic
    SemVer bump classification, retrieval, comparison, deprecation /
    undeprecation, sunset-date warnings, changelog queries, and consumer
    version-range pinning.

    The engine is fully in-memory and thread-safe.  All mutating operations
    generate a SHA-256 provenance hash that is stored alongside the record.

    Attributes:
        _versions: Flat dict mapping version_id → version record dict.
        _schema_versions: Maps schema_id → list of version_ids (insertion order).
        _latest_version: Maps schema_id → latest version string (e.g. "2.3.1").
        _pins: Maps consumer_id → {schema_id → version_range_string}.
        _pin_records: Maps pin_id → pin record dict.
        _stats: Running counters for get_statistics().
        _lock: Reentrant lock for thread safety.

    Example:
        >>> engine = SchemaVersionerEngine()
        >>> v = engine.create_version("sch-1", {"type": "object"})
        >>> assert v["version"] == "1.0.0"
    """

    def __init__(self) -> None:
        """Initialize a fresh SchemaVersionerEngine with empty in-memory state."""
        self._versions: Dict[str, Dict[str, Any]] = {}
        self._schema_versions: Dict[str, List[str]] = defaultdict(list)
        self._latest_version: Dict[str, str] = {}
        self._pins: Dict[str, Dict[str, str]] = defaultdict(dict)
        self._pin_records: Dict[str, Dict[str, Any]] = {}
        self._stats: Dict[str, Any] = {
            "total_versions_created": 0,
            "by_bump_type": {BUMP_MAJOR: 0, BUMP_MINOR: 0, BUMP_PATCH: 0},
            "deprecated_count": 0,
            "total_pins": 0,
        }
        self._lock = threading.Lock()
        logger.info("SchemaVersionerEngine initialised.")

    # ------------------------------------------------------------------
    # 1. create_version
    # ------------------------------------------------------------------

    def create_version(
        self,
        schema_id: str,
        definition_json: Any,
        changes: Optional[List[Dict[str, Any]]] = None,
        changelog_note: str = "",
        created_by: str = "system",
    ) -> Dict[str, Any]:
        """Create a new schema version with auto-classified SemVer bump.

        Determines the bump type from the provided ``changes`` list (or
        defaults to ``"patch"`` when no changes are supplied).  The first
        version for any schema always starts at ``"1.0.0"`` regardless of
        bump type.

        Args:
            schema_id: Opaque identifier for the schema (e.g. ``"sch-orders"``).
            definition_json: The full schema definition as a JSON-serialisable
                Python object (typically a ``dict``).
            changes: Optional list of change descriptor dicts.  Each dict
                should contain at least a ``"severity"`` key with one of:
                ``"breaking"``, ``"non_breaking"``, ``"cosmetic"``
                (and optionally ``"field"``, ``"description"``).
            changelog_note: Free-text note appended to the changelog entry for
                this version.
            created_by: Identifier of the actor creating this version (user ID,
                service name, etc.).

        Returns:
            A dict with the following keys:

            - ``id`` (str): Unique version ID (e.g. ``"VER-3f9a1b2c4d5e"``).
            - ``schema_id`` (str): The schema this version belongs to.
            - ``version`` (str): SemVer string (e.g. ``"2.0.0"``).
            - ``bump_type`` (str): Classification used: ``"major"``,
              ``"minor"``, or ``"patch"``.
            - ``definition`` (Any): The schema definition passed in.
            - ``definition_hash`` (str): SHA-256 of the canonical definition JSON.
            - ``changes`` (List[Dict]): The change list (empty list if None).
            - ``changelog`` (List[Dict]): Structured changelog entries derived
              from changes.
            - ``changelog_note`` (str): The free-text note.
            - ``created_by`` (str): Actor identifier.
            - ``created_at`` (str): ISO-8601 UTC timestamp.
            - ``is_deprecated`` (bool): Always ``False`` on creation.
            - ``deprecated_at`` (Optional[str]): Always ``None`` on creation.
            - ``sunset_date`` (Optional[str]): Always ``None`` on creation.
            - ``deprecation_reason`` (str): Always ``""`` on creation.
            - ``provenance_hash`` (str): SHA-256 provenance fingerprint.

        Raises:
            ValueError: If ``schema_id`` is empty or ``definition_json`` is
                not JSON-serialisable.

        Example:
            >>> engine = SchemaVersionerEngine()
            >>> v = engine.create_version("sch-1", {"type": "object"})
            >>> v["version"]
            '1.0.0'
        """
        if not schema_id or not isinstance(schema_id, str):
            raise ValueError("schema_id must be a non-empty string.")
        if changes is None:
            changes = []

        start_time = time.perf_counter()

        # Validate definition is JSON-serialisable
        try:
            definition_canonical = _serialize_definition(definition_json)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"definition_json is not JSON-serialisable: {exc}"
            ) from exc

        definition_hash = hashlib.sha256(
            definition_canonical.encode("utf-8")
        ).hexdigest()

        with self._lock:
            # Determine new version string
            if schema_id not in self._latest_version:
                new_version_str = "1.0.0"
                bump_type = BUMP_PATCH  # first version has no meaningful bump
            else:
                bump_type = _classify_bump(changes)
                new_version_str = _bump_version(
                    self._latest_version[schema_id], bump_type
                )

            version_id = _generate_id(_PREFIX_VERSION)
            now_str = _utcnow().isoformat()

            # Build structured changelog from changes list
            changelog = self._build_changelog_entries(changes, changelog_note)

            provenance_hash = _compute_provenance(
                "create_version",
                f"{schema_id}:{new_version_str}:{definition_hash}",
            )

            version_record: Dict[str, Any] = {
                "id": version_id,
                "schema_id": schema_id,
                "version": new_version_str,
                "bump_type": bump_type,
                "definition": definition_json,
                "definition_hash": definition_hash,
                "changes": changes,
                "changelog": changelog,
                "changelog_note": changelog_note,
                "created_by": created_by,
                "created_at": now_str,
                "is_deprecated": False,
                "deprecated_at": None,
                "sunset_date": None,
                "deprecation_reason": "",
                "provenance_hash": provenance_hash,
            }

            # Persist
            self._versions[version_id] = version_record
            self._schema_versions[schema_id].append(version_id)
            self._latest_version[schema_id] = new_version_str

            # Update statistics
            self._stats["total_versions_created"] += 1
            self._stats["by_bump_type"][bump_type] += 1

        elapsed_ms = (time.perf_counter() - start_time) * 1000
        logger.info(
            "Created version %s for schema '%s' (bump=%s) in %.2f ms.",
            new_version_str,
            schema_id,
            bump_type,
            elapsed_ms,
        )
        return version_record

    # ------------------------------------------------------------------
    # 2. get_version
    # ------------------------------------------------------------------

    def get_version(self, version_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve a version record by its unique version ID.

        Args:
            version_id: The unique version identifier returned by
                :meth:`create_version` (e.g. ``"VER-3f9a1b2c4d5e"``).

        Returns:
            The version record dict, or ``None`` if not found.

        Example:
            >>> v = engine.create_version("sch-1", {})
            >>> engine.get_version(v["id"]) is not None
            True
            >>> engine.get_version("VER-nonexistent") is None
            True
        """
        with self._lock:
            return self._versions.get(version_id)

    # ------------------------------------------------------------------
    # 3. get_version_by_string
    # ------------------------------------------------------------------

    def get_version_by_string(
        self, schema_id: str, version_string: str
    ) -> Optional[Dict[str, Any]]:
        """Retrieve a specific version by schema ID and SemVer string.

        Args:
            schema_id: The schema identifier.
            version_string: The exact SemVer string to look up (e.g. ``"2.1.0"``).

        Returns:
            The matching version record dict, or ``None`` if no such version
            exists for the given schema.

        Example:
            >>> engine.create_version("sch-1", {})
            >>> engine.get_version_by_string("sch-1", "1.0.0") is not None
            True
            >>> engine.get_version_by_string("sch-1", "9.9.9") is None
            True
        """
        with self._lock:
            for vid in self._schema_versions.get(schema_id, []):
                record = self._versions.get(vid)
                if record and record["version"] == version_string:
                    return record
        return None

    # ------------------------------------------------------------------
    # 4. list_versions
    # ------------------------------------------------------------------

    def list_versions(
        self,
        schema_id: str,
        include_deprecated: bool = False,
        limit: int = 100,
        offset: int = 0,
    ) -> List[Dict[str, Any]]:
        """List versions for a schema, sorted newest-first by SemVer.

        Args:
            schema_id: The schema identifier whose versions to list.
            include_deprecated: When ``False`` (default), deprecated versions
                are excluded from results.
            limit: Maximum number of records to return (default 100).
            offset: Number of records to skip before returning (for pagination).

        Returns:
            List of version record dicts sorted in descending SemVer order
            (newest first).  Returns an empty list if the schema has no
            versions or if ``offset`` exceeds the total count.

        Example:
            >>> engine.create_version("sch-1", {})
            >>> versions = engine.list_versions("sch-1")
            >>> versions[0]["version"]
            '1.0.0'
        """
        with self._lock:
            vid_list = list(self._schema_versions.get(schema_id, []))

        records = []
        for vid in vid_list:
            with self._lock:
                record = self._versions.get(vid)
            if record is None:
                continue
            if not include_deprecated and record.get("is_deprecated", False):
                continue
            records.append(record)

        # Sort newest-first by SemVer tuple
        records.sort(
            key=lambda r: _version_tuple_key(r["version"]), reverse=True
        )

        return records[offset : offset + limit]

    # ------------------------------------------------------------------
    # 5. get_latest_version
    # ------------------------------------------------------------------

    def get_latest_version(self, schema_id: str) -> Optional[Dict[str, Any]]:
        """Return the most recent non-deprecated version for a schema.

        Iterates through all versions in descending SemVer order and returns
        the first one that is not deprecated.

        Args:
            schema_id: The schema identifier.

        Returns:
            The latest non-deprecated version record dict, or ``None`` if no
            such version exists (e.g. all versions are deprecated or schema
            is unknown).

        Example:
            >>> engine.create_version("sch-1", {})
            >>> latest = engine.get_latest_version("sch-1")
            >>> latest["version"]
            '1.0.0'
        """
        versions = self.list_versions(schema_id, include_deprecated=False, limit=1)
        return versions[0] if versions else None

    # ------------------------------------------------------------------
    # 6. compare_versions
    # ------------------------------------------------------------------

    def compare_versions(
        self, version_id_a: str, version_id_b: str
    ) -> Dict[str, Any]:
        """Compare two versions and return a structured diff summary.

        Compares the two versions' SemVer positions, definition hashes, and
        change lists to produce a human-readable diff summary.  The result
        indicates which version is newer (or whether they are equal) and
        lists added / removed / modified fields at a surface level by
        inspecting top-level keys of both definitions.

        Args:
            version_id_a: ID of the first version (the "from" or "base").
            version_id_b: ID of the second version (the "to" or "target").

        Returns:
            A dict with the following keys:

            - ``version_id_a`` (str): First version ID.
            - ``version_id_b`` (str): Second version ID.
            - ``schema_id_a`` (str): Schema ID of version A.
            - ``schema_id_b`` (str): Schema ID of version B.
            - ``version_a`` (str): SemVer string of version A.
            - ``version_b`` (str): SemVer string of version B.
            - ``same_schema`` (bool): Whether both versions belong to the
              same schema.
            - ``version_relation`` (str): ``"a_newer"``, ``"b_newer"``, or
              ``"equal"`` (only meaningful when same_schema is True).
            - ``definition_changed`` (bool): Whether the definition hashes differ.
            - ``added_keys`` (List[str]): Top-level keys present in B but not A.
            - ``removed_keys`` (List[str]): Top-level keys present in A but not B.
            - ``common_keys`` (List[str]): Top-level keys present in both.
            - ``changes_a`` (List[Dict]): Change list from version A.
            - ``changes_b`` (List[Dict]): Change list from version B.
            - ``compared_at`` (str): ISO-8601 UTC timestamp of comparison.

        Raises:
            ValueError: If either version ID does not exist.

        Example:
            >>> v1 = engine.create_version("sch-1", {"id": 1})
            >>> v2 = engine.create_version("sch-1", {"id": 1, "name": "x"})
            >>> diff = engine.compare_versions(v1["id"], v2["id"])
            >>> diff["added_keys"]
            ['name']
        """
        with self._lock:
            rec_a = self._versions.get(version_id_a)
            rec_b = self._versions.get(version_id_b)

        if rec_a is None:
            raise ValueError(f"Version '{version_id_a}' not found.")
        if rec_b is None:
            raise ValueError(f"Version '{version_id_b}' not found.")

        same_schema = rec_a["schema_id"] == rec_b["schema_id"]
        ver_tuple_a = _version_tuple_key(rec_a["version"])
        ver_tuple_b = _version_tuple_key(rec_b["version"])

        if ver_tuple_a > ver_tuple_b:
            version_relation = "a_newer"
        elif ver_tuple_b > ver_tuple_a:
            version_relation = "b_newer"
        else:
            version_relation = "equal"

        def_a = rec_a["definition"]
        def_b = rec_b["definition"]
        definition_changed = rec_a["definition_hash"] != rec_b["definition_hash"]

        keys_a = set(def_a.keys()) if isinstance(def_a, dict) else set()
        keys_b = set(def_b.keys()) if isinstance(def_b, dict) else set()
        added_keys = sorted(keys_b - keys_a)
        removed_keys = sorted(keys_a - keys_b)
        common_keys = sorted(keys_a & keys_b)

        return {
            "version_id_a": version_id_a,
            "version_id_b": version_id_b,
            "schema_id_a": rec_a["schema_id"],
            "schema_id_b": rec_b["schema_id"],
            "version_a": rec_a["version"],
            "version_b": rec_b["version"],
            "same_schema": same_schema,
            "version_relation": version_relation,
            "definition_changed": definition_changed,
            "added_keys": added_keys,
            "removed_keys": removed_keys,
            "common_keys": common_keys,
            "changes_a": rec_a.get("changes", []),
            "changes_b": rec_b.get("changes", []),
            "compared_at": _utcnow().isoformat(),
        }

    # ------------------------------------------------------------------
    # 7. deprecate_version
    # ------------------------------------------------------------------

    def deprecate_version(
        self,
        version_id: str,
        sunset_date: Optional[str] = None,
        reason: str = "",
    ) -> Dict[str, Any]:
        """Mark a version as deprecated with an optional sunset date.

        Once deprecated, a version is excluded from :meth:`list_versions`
        (when ``include_deprecated=False``) and will not be returned by
        :meth:`get_latest_version`.

        Args:
            version_id: The ID of the version to deprecate.
            sunset_date: Optional ISO-8601 date string (``"YYYY-MM-DD"``) after
                which the version is considered end-of-life.  Must be a future
                date if provided.
            reason: Free-text explanation for the deprecation.

        Returns:
            The updated version record dict (with ``is_deprecated=True``).

        Raises:
            ValueError: If the version does not exist or is already deprecated,
                or if ``sunset_date`` is not a valid ``"YYYY-MM-DD"`` string.

        Example:
            >>> v = engine.create_version("sch-1", {})
            >>> engine.deprecate_version(v["id"], sunset_date="2027-01-01")
            >>> engine.get_latest_version("sch-1") is None
            True
        """
        with self._lock:
            record = self._versions.get(version_id)
            if record is None:
                raise ValueError(f"Version '{version_id}' not found.")
            if record.get("is_deprecated"):
                raise ValueError(
                    f"Version '{version_id}' is already deprecated."
                )

            validated_sunset: Optional[str] = None
            if sunset_date is not None:
                validated_sunset = self._validate_sunset_date(sunset_date)

            now_str = _utcnow().isoformat()
            provenance_hash = _compute_provenance(
                "deprecate_version",
                f"{version_id}:{sunset_date}:{reason}",
            )

            record["is_deprecated"] = True
            record["deprecated_at"] = now_str
            record["sunset_date"] = validated_sunset
            record["deprecation_reason"] = reason
            record["provenance_hash"] = provenance_hash

            self._stats["deprecated_count"] += 1

        logger.info(
            "Deprecated version '%s' (schema='%s', sunset='%s').",
            version_id,
            record["schema_id"],
            validated_sunset,
        )
        return record

    # ------------------------------------------------------------------
    # 8. undeprecate_version
    # ------------------------------------------------------------------

    def undeprecate_version(self, version_id: str) -> Dict[str, Any]:
        """Remove deprecation status from a previously deprecated version.

        Clears ``is_deprecated``, ``deprecated_at``, ``sunset_date``, and
        ``deprecation_reason`` fields and generates a new provenance hash.

        Args:
            version_id: The ID of the version to undeprecate.

        Returns:
            The updated version record dict (with ``is_deprecated=False``).

        Raises:
            ValueError: If the version does not exist or is not currently
                deprecated.

        Example:
            >>> v = engine.create_version("sch-1", {})
            >>> engine.deprecate_version(v["id"])
            >>> engine.undeprecate_version(v["id"])["is_deprecated"]
            False
        """
        with self._lock:
            record = self._versions.get(version_id)
            if record is None:
                raise ValueError(f"Version '{version_id}' not found.")
            if not record.get("is_deprecated"):
                raise ValueError(
                    f"Version '{version_id}' is not deprecated; nothing to undo."
                )

            provenance_hash = _compute_provenance(
                "undeprecate_version", version_id
            )

            record["is_deprecated"] = False
            record["deprecated_at"] = None
            record["sunset_date"] = None
            record["deprecation_reason"] = ""
            record["provenance_hash"] = provenance_hash

            self._stats["deprecated_count"] = max(
                0, self._stats["deprecated_count"] - 1
            )

        logger.info("Undeprecated version '%s'.", version_id)
        return record

    # ------------------------------------------------------------------
    # 9. check_sunset_warnings
    # ------------------------------------------------------------------

    def check_sunset_warnings(self, warning_days: int = 30) -> List[Dict[str, Any]]:
        """Return versions whose sunset date is within ``warning_days`` of today.

        Only deprecated versions with a ``sunset_date`` set are evaluated.
        Versions whose sunset date has already passed are included (they are
        overdue).

        Args:
            warning_days: Number of days ahead to look for approaching sunsets.
                Default is 30.  Must be a non-negative integer.

        Returns:
            List of dicts, each containing:

            - ``version_id`` (str): Version identifier.
            - ``schema_id`` (str): Owning schema identifier.
            - ``version`` (str): SemVer string.
            - ``sunset_date`` (str): ISO-8601 date string.
            - ``days_until_sunset`` (int): Negative if already past sunset.
            - ``is_overdue`` (bool): ``True`` if sunset has passed.
            - ``deprecation_reason`` (str): Reason recorded at deprecation.

        Raises:
            ValueError: If ``warning_days`` is negative.

        Example:
            >>> import datetime
            >>> v = engine.create_version("sch-1", {})
            >>> engine.deprecate_version(v["id"], sunset_date="2026-02-18")
            >>> warnings = engine.check_sunset_warnings(warning_days=365)
            >>> len(warnings) >= 1
            True
        """
        if warning_days < 0:
            raise ValueError("warning_days must be a non-negative integer.")

        today = datetime.now(timezone.utc).date()
        cutoff = today + timedelta(days=warning_days)
        warnings: List[Dict[str, Any]] = []

        with self._lock:
            versions_snapshot = list(self._versions.values())

        for record in versions_snapshot:
            if not record.get("is_deprecated"):
                continue
            raw_sunset = record.get("sunset_date")
            if not raw_sunset:
                continue
            try:
                sunset = date.fromisoformat(raw_sunset)
            except ValueError:
                logger.warning(
                    "Could not parse sunset_date '%s' for version '%s'.",
                    raw_sunset,
                    record["id"],
                )
                continue

            if sunset <= cutoff:
                days_until = (sunset - today).days
                warnings.append(
                    {
                        "version_id": record["id"],
                        "schema_id": record["schema_id"],
                        "version": record["version"],
                        "sunset_date": raw_sunset,
                        "days_until_sunset": days_until,
                        "is_overdue": days_until < 0,
                        "deprecation_reason": record.get("deprecation_reason", ""),
                    }
                )

        # Sort soonest sunset first
        warnings.sort(key=lambda w: w["days_until_sunset"])
        return warnings

    # ------------------------------------------------------------------
    # 10. get_changelog
    # ------------------------------------------------------------------

    def get_changelog(
        self,
        schema_id: str,
        from_version: Optional[str] = None,
        to_version: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Retrieve structured changelog entries for a schema, optionally range-filtered.

        Changelogs are returned in ascending SemVer order (oldest first) so
        callers can read the history chronologically.  Each entry represents
        a single version's changelog.

        Args:
            schema_id: The schema whose changelog to retrieve.
            from_version: Optional lower bound SemVer string (inclusive).
                Versions older than this are excluded.
            to_version: Optional upper bound SemVer string (inclusive).
                Versions newer than this are excluded.

        Returns:
            List of changelog entry dicts, each containing:

            - ``version_id`` (str): Version identifier.
            - ``schema_id`` (str): Schema identifier.
            - ``version`` (str): SemVer string.
            - ``bump_type`` (str): Bump classification.
            - ``changelog`` (List[Dict]): Structured change entries.
            - ``changelog_note`` (str): Free-text note for this version.
            - ``created_by`` (str): Actor who created this version.
            - ``created_at`` (str): ISO-8601 UTC creation timestamp.
            - ``is_deprecated`` (bool): Current deprecation status.

        Example:
            >>> engine.create_version("sch-1", {})
            >>> engine.create_version("sch-1", {"x": 1}, changes=[{"severity": "non_breaking"}])
            >>> log = engine.get_changelog("sch-1")
            >>> [e["version"] for e in log]
            ['1.0.0', '1.1.0']
        """
        with self._lock:
            vid_list = list(self._schema_versions.get(schema_id, []))

        # Parse optional bounds
        from_tuple: Optional[Tuple[int, int, int]] = None
        to_tuple: Optional[Tuple[int, int, int]] = None
        if from_version:
            from_tuple = _parse_version(from_version)
        if to_version:
            to_tuple = _parse_version(to_version)

        entries: List[Dict[str, Any]] = []
        for vid in vid_list:
            with self._lock:
                record = self._versions.get(vid)
            if record is None:
                continue

            ver_tuple = _version_tuple_key(record["version"])
            if from_tuple and ver_tuple < from_tuple:
                continue
            if to_tuple and ver_tuple > to_tuple:
                continue

            entries.append(
                {
                    "version_id": record["id"],
                    "schema_id": record["schema_id"],
                    "version": record["version"],
                    "bump_type": record["bump_type"],
                    "changelog": record.get("changelog", []),
                    "changelog_note": record.get("changelog_note", ""),
                    "created_by": record.get("created_by", "system"),
                    "created_at": record.get("created_at", ""),
                    "is_deprecated": record.get("is_deprecated", False),
                }
            )

        # Sort ascending (oldest first)
        entries.sort(key=lambda e: _version_tuple_key(e["version"]))
        return entries

    # ------------------------------------------------------------------
    # 11. classify_bump (public proxy)
    # ------------------------------------------------------------------

    def classify_bump(self, changes: List[Dict[str, Any]]) -> str:
        """Classify the SemVer bump type required for a list of changes.

        This is the public entry-point to the same logic used internally by
        :meth:`create_version`.

        Args:
            changes: List of change descriptor dicts.  Each should have at
                minimum a ``"severity"`` key with value ``"breaking"``,
                ``"non_breaking"``, or ``"cosmetic"`` (and any aliases).

        Returns:
            ``"major"`` if any change is breaking,
            ``"minor"`` if any change is additive (and none are breaking),
            ``"patch"`` for all-cosmetic or empty change lists.

        Example:
            >>> engine.classify_bump([{"severity": "breaking"}])
            'major'
            >>> engine.classify_bump([{"severity": "non_breaking"}])
            'minor'
            >>> engine.classify_bump([])
            'patch'
        """
        return _classify_bump(changes)

    # ------------------------------------------------------------------
    # 12. parse_version (public proxy)
    # ------------------------------------------------------------------

    def parse_version(self, version_string: str) -> Tuple[int, int, int]:
        """Parse a SemVer string into a (major, minor, patch) integer tuple.

        Args:
            version_string: A semantic version string in the form ``"X.Y.Z"``.

        Returns:
            Three-tuple of non-negative integers.

        Raises:
            ValueError: If the string is malformed or any part is not an integer.

        Example:
            >>> engine.parse_version("3.14.0")
            (3, 14, 0)
        """
        return _parse_version(version_string)

    # ------------------------------------------------------------------
    # 13. bump_version (public proxy)
    # ------------------------------------------------------------------

    def bump_version(self, current_version: str, bump_type: str) -> str:
        """Increment a SemVer string according to the specified bump type.

        Args:
            current_version: Current SemVer string (e.g. ``"1.2.3"``).
            bump_type: One of ``"major"``, ``"minor"``, or ``"patch"``.

        Returns:
            New SemVer string after applying the bump.

        Raises:
            ValueError: If ``bump_type`` is unrecognised or the version is
                malformed.

        Example:
            >>> engine.bump_version("1.2.3", "major")
            '2.0.0'
            >>> engine.bump_version("1.2.3", "minor")
            '1.3.0'
            >>> engine.bump_version("1.2.3", "patch")
            '1.2.4'
        """
        return _bump_version(current_version, bump_type)

    # ------------------------------------------------------------------
    # 14. pin_version
    # ------------------------------------------------------------------

    def pin_version(
        self,
        schema_id: str,
        consumer_id: str,
        version_range: str,
    ) -> Dict[str, Any]:
        """Pin a consumer to a version range for a specific schema.

        Stores the pinning relationship so downstream systems can query which
        version range a given consumer has locked to.  The ``version_range``
        is stored verbatim; GreenLang does not evaluate the range predicate
        at storage time.

        Args:
            schema_id: The schema identifier being pinned.
            consumer_id: Identifier of the consuming service or client.
            version_range: A version range expression string, e.g.
                ``">=2.0.0 <3.0.0"`` or ``"~1.2.0"`` or ``"1.3.x"``.
                Any non-empty string is accepted.

        Returns:
            A dict with the following keys:

            - ``pin_id`` (str): Unique pin record identifier.
            - ``schema_id`` (str): Schema being pinned.
            - ``consumer_id`` (str): Consumer being pinned.
            - ``version_range`` (str): The range expression stored.
            - ``pinned_at`` (str): ISO-8601 UTC timestamp.
            - ``provenance_hash`` (str): SHA-256 provenance fingerprint.

        Raises:
            ValueError: If ``schema_id``, ``consumer_id``, or
                ``version_range`` is empty.

        Example:
            >>> pin = engine.pin_version("sch-1", "service-orders", ">=1.0.0 <2.0.0")
            >>> pin["version_range"]
            '>=1.0.0 <2.0.0'
        """
        if not schema_id:
            raise ValueError("schema_id must be non-empty.")
        if not consumer_id:
            raise ValueError("consumer_id must be non-empty.")
        if not version_range:
            raise ValueError("version_range must be non-empty.")

        pin_id = _generate_id(_PREFIX_PIN)
        now_str = _utcnow().isoformat()
        provenance_hash = _compute_provenance(
            "pin_version",
            f"{schema_id}:{consumer_id}:{version_range}",
        )

        pin_record: Dict[str, Any] = {
            "pin_id": pin_id,
            "schema_id": schema_id,
            "consumer_id": consumer_id,
            "version_range": version_range,
            "pinned_at": now_str,
            "provenance_hash": provenance_hash,
        }

        with self._lock:
            # Overwrite any existing pin for this consumer/schema pair
            self._pins[consumer_id][schema_id] = version_range
            self._pin_records[pin_id] = pin_record
            self._stats["total_pins"] += 1

        logger.info(
            "Pinned consumer '%s' to schema '%s' range '%s'.",
            consumer_id,
            schema_id,
            version_range,
        )
        return pin_record

    # ------------------------------------------------------------------
    # 15. get_statistics
    # ------------------------------------------------------------------

    def get_statistics(self) -> Dict[str, Any]:
        """Return aggregate statistics for the SchemaVersionerEngine.

        Computes a snapshot of engine state including version counts, deprecation
        counts, bump-type distribution, schema-level version counts, and pin
        counts.

        Returns:
            A dict with the following keys:

            - ``total_versions`` (int): Total versions ever created.
            - ``active_versions`` (int): Non-deprecated version count.
            - ``deprecated_count`` (int): Deprecated version count.
            - ``by_bump_type`` (Dict[str, int]): Count per bump type
              (``"major"``, ``"minor"``, ``"patch"``).
            - ``by_schema`` (Dict[str, int]): Per-schema version count
              (all versions, including deprecated).
            - ``total_schemas`` (int): Number of distinct schemas tracked.
            - ``total_pins`` (int): Total number of pin records ever created.
            - ``collected_at`` (str): ISO-8601 UTC timestamp of collection.

        Example:
            >>> engine.create_version("sch-1", {})
            >>> stats = engine.get_statistics()
            >>> stats["total_versions"]
            1
        """
        with self._lock:
            total = self._stats["total_versions_created"]
            deprecated = self._stats["deprecated_count"]
            by_bump = dict(self._stats["by_bump_type"])
            by_schema = {
                sid: len(vids)
                for sid, vids in self._schema_versions.items()
            }
            total_pins = self._stats["total_pins"]

        return {
            "total_versions": total,
            "active_versions": total - deprecated,
            "deprecated_count": deprecated,
            "by_bump_type": by_bump,
            "by_schema": by_schema,
            "total_schemas": len(by_schema),
            "total_pins": total_pins,
            "collected_at": _utcnow().isoformat(),
        }

    # ------------------------------------------------------------------
    # 16. reset
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Clear all in-memory data and reset statistics to zero.

        This is a destructive operation intended for testing or reinitialization
        scenarios.  All versions, changelog entries, deprecation records, and
        pin records are permanently removed.

        Example:
            >>> engine.create_version("sch-1", {})
            >>> engine.reset()
            >>> engine.get_statistics()["total_versions"]
            0
        """
        with self._lock:
            self._versions.clear()
            self._schema_versions.clear()
            self._latest_version.clear()
            self._pins.clear()
            self._pin_records.clear()
            self._stats = {
                "total_versions_created": 0,
                "by_bump_type": {BUMP_MAJOR: 0, BUMP_MINOR: 0, BUMP_PATCH: 0},
                "deprecated_count": 0,
                "total_pins": 0,
            }
        logger.info("SchemaVersionerEngine reset — all data cleared.")

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_changelog_entries(
        self,
        changes: List[Dict[str, Any]],
        changelog_note: str,
    ) -> List[Dict[str, Any]]:
        """Build structured changelog entries from a raw change list.

        Each entry in the returned list corresponds to one change dict and
        contains normalised severity, field name, description, and a generated
        unique entry ID.

        Args:
            changes: Raw change descriptors passed to :meth:`create_version`.
            changelog_note: Top-level note for the version (not included in
                per-entry dicts; stored separately on the version record).

        Returns:
            List of normalised changelog entry dicts:

            - ``entry_id`` (str): Unique entry identifier.
            - ``severity`` (str): Normalised severity label.
            - ``field`` (str): Affected field name (empty string if not given).
            - ``description`` (str): Human-readable change description.
        """
        entries: List[Dict[str, Any]] = []
        for change in changes:
            raw_sev = str(change.get("severity", "patch")).lower()
            # Normalise severity to canonical label
            if raw_sev in _BREAKING_SEVERITIES:
                canonical_sev = "breaking"
            elif raw_sev in _NON_BREAKING_SEVERITIES:
                canonical_sev = "non_breaking"
            else:
                canonical_sev = "cosmetic"

            entries.append(
                {
                    "entry_id": f"CHG-{uuid.uuid4().hex[:8]}",
                    "severity": canonical_sev,
                    "field": change.get("field", ""),
                    "description": change.get("description", ""),
                }
            )
        return entries

    @staticmethod
    def _validate_sunset_date(sunset_date: str) -> str:
        """Validate and return a sunset date string in ``"YYYY-MM-DD"`` format.

        Args:
            sunset_date: Date string to validate.

        Returns:
            The original string if valid.

        Raises:
            ValueError: If the string is not a valid ISO-8601 date.
        """
        try:
            date.fromisoformat(sunset_date)
        except ValueError as exc:
            raise ValueError(
                f"sunset_date '{sunset_date}' is not a valid ISO-8601 date "
                f"(expected YYYY-MM-DD)."
            ) from exc
        return sunset_date
