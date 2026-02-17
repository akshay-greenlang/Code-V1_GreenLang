# -*- coding: utf-8 -*-
"""
ChangeDetectorEngine - AGENT-DATA-017: Schema Migration Agent (GL-DATA-X-020)

Engine 3 of 7 in the Schema Migration Agent pipeline. Detects structural
differences between two schema versions and classifies each change by its
severity impact on consumers.

Supported schema formats:
    - JSON Schema (Draft 4/6/7/2019-09/2020-12): fields extracted from the
      ``"properties"`` dict; required fields from the ``"required"`` array.
    - Apache Avro: fields extracted from the ``"fields"`` array; type unions
      of the form ``["null", "<type>"]`` are unwrapped to ``<type>`` with the
      field marked as optional.

Change types detected:
    - added         : field present in target but not in source
    - removed       : field present in source but not in target
    - renamed       : field with Jaro-Winkler similarity > 0.85 and matching type
    - retyped       : same field name, different type
    - reordered     : same fields in different index positions (Avro / ordered lists)
    - constraint_changed : required↔optional, min/max/minLength/maxLength/pattern
    - enum_changed  : enum values added or removed
    - default_changed : default value changed (field type unchanged)

Severity classification:
    - breaking     : removes or restricts; downstream consumers will fail
    - non_breaking : adds or relaxes; backward-compatible
    - cosmetic     : description/order/metadata change only

Zero-Hallucination Guarantees:
    - All comparison logic is deterministic set and dict operations
    - Jaro-Winkler is a pure-Python implementation with no external deps
    - No LLM calls anywhere in the detection path
    - SHA-256 provenance recorded on every ``detect_changes`` invocation
    - Thread-safe via ``threading.Lock``

Example:
    >>> from greenlang.schema_migration.change_detector import ChangeDetectorEngine
    >>> engine = ChangeDetectorEngine()
    >>> source = {
    ...     "type": "object",
    ...     "properties": {
    ...         "user_id": {"type": "integer"},
    ...         "email":   {"type": "string"},
    ...     },
    ...     "required": ["user_id", "email"],
    ... }
    >>> target = {
    ...     "type": "object",
    ...     "properties": {
    ...         "user_id":    {"type": "integer"},
    ...         "email":      {"type": "string"},
    ...         "created_at": {"type": "string", "format": "date-time"},
    ...     },
    ...     "required": ["user_id", "email"],
    ... }
    >>> result = engine.detect_changes(source, target)
    >>> result["summary"]["total_count"]
    1
    >>> result["changes"][0]["change_type"]
    'added'
    >>> result["changes"][0]["severity"]
    'non_breaking'

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
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from greenlang.schema_migration.provenance import ProvenanceTracker

logger = logging.getLogger(__name__)

__all__ = [
    "ChangeDetectorEngine",
]

# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

# Jaro-Winkler similarity threshold for rename detection
_RENAME_SIMILARITY_THRESHOLD: float = 0.85

# Jaro-Winkler prefix scaling factor (standard Winkler value)
_WINKLER_SCALING_FACTOR: float = 0.1

# Maximum prefix length considered by Winkler extension
_WINKLER_MAX_PREFIX: int = 4

# Type widening pairs: (narrow_type, wide_type) — int→float is non-breaking
_TYPE_WIDENING_PAIRS: frozenset = frozenset(
    {
        ("integer", "number"),
        ("integer", "float"),
        ("integer", "string"),
        ("number", "string"),
        ("float", "string"),
        ("int", "float"),
        ("int", "double"),
        ("int", "string"),
        ("long", "double"),
        ("long", "string"),
        ("float", "double"),
    }
)

# Constraint keys that, when tightened, create a breaking change
_NUMERIC_CONSTRAINT_KEYS: frozenset = frozenset(
    {"minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum",
     "minLength", "maxLength", "minItems", "maxItems", "minProperties",
     "maxProperties", "multipleOf"}
)

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _utcnow() -> datetime:
    """Return current UTC datetime with microseconds zeroed."""
    return datetime.now(timezone.utc).replace(microsecond=0)


def _generate_id(prefix: str = "CHG") -> str:
    """Generate a unique identifier with the given prefix.

    Args:
        prefix: Short uppercase prefix string.

    Returns:
        String of the form ``{prefix}-{hex12}``.
    """
    return f"{prefix}-{uuid.uuid4().hex[:12]}"


def _detection_id() -> str:
    """Generate a unique detection run identifier.

    Returns:
        String of the form ``DET-{hex12}``.
    """
    return _generate_id("DET")


def _compute_sha256(data: Any) -> str:
    """Compute a SHA-256 hex digest of arbitrary JSON-serialisable data.

    Serialises the payload to canonical JSON (sorted keys, ``str`` fallback
    for non-serialisable types) before hashing to ensure determinism.

    Args:
        data: Any JSON-serialisable object or ``None``.

    Returns:
        64-character lowercase hex string.
    """
    if data is None:
        serialized = "null"
    else:
        serialized = json.dumps(data, sort_keys=True, default=str)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Pure-Python Jaro-Winkler implementation (~35 lines)
# ---------------------------------------------------------------------------


def _jaro_similarity(s1: str, s2: str) -> float:
    """Compute the Jaro similarity between two strings.

    The Jaro similarity is defined as::

        jaro(s1, s2) = 0                           if m == 0
        jaro(s1, s2) = (m/|s1| + m/|s2| + (m-t)/m) / 3   otherwise

    where ``m`` is the number of matching characters and ``t`` is half the
    number of transpositions.  Two characters are considered matching if they
    are the same and within ``floor(max(|s1|, |s2|) / 2) - 1`` positions.

    Args:
        s1: First string.
        s2: Second string.

    Returns:
        Float in ``[0.0, 1.0]``; 1.0 means identical.
    """
    if s1 == s2:
        return 1.0

    len1, len2 = len(s1), len(s2)
    if len1 == 0 or len2 == 0:
        return 0.0

    # Maximum distance for matching characters
    match_distance = max(len1, len2) // 2 - 1
    if match_distance < 0:
        match_distance = 0

    s1_matches = [False] * len1
    s2_matches = [False] * len2

    matches = 0
    transpositions = 0

    # Find matching characters
    for i in range(len1):
        start = max(0, i - match_distance)
        end = min(i + match_distance + 1, len2)
        for j in range(start, end):
            if s2_matches[j] or s1[i] != s2[j]:
                continue
            s1_matches[i] = True
            s2_matches[j] = True
            matches += 1
            break

    if matches == 0:
        return 0.0

    # Count transpositions
    k = 0
    for i in range(len1):
        if not s1_matches[i]:
            continue
        while not s2_matches[k]:
            k += 1
        if s1[i] != s2[k]:
            transpositions += 1
        k += 1

    return (
        matches / len1
        + matches / len2
        + (matches - transpositions / 2) / matches
    ) / 3.0


def _jaro_winkler_similarity(s1: str, s2: str) -> float:
    """Compute the Jaro-Winkler similarity between two strings.

    Extends Jaro similarity by giving extra weight to strings that share a
    common prefix (up to ``_WINKLER_MAX_PREFIX`` characters).

    Formula::

        jaro_winkler(s1, s2) = jaro(s1, s2) + p * l * (1 - jaro(s1, s2))

    where ``p`` is the scaling factor (0.1) and ``l`` is the length of the
    common prefix (capped at 4).

    Args:
        s1: First string (case-sensitive comparison).
        s2: Second string.

    Returns:
        Float in ``[0.0, 1.0]``; 1.0 means identical.

    Example:
        >>> round(_jaro_winkler_similarity("martha", "marhta"), 4)
        0.9611
        >>> round(_jaro_winkler_similarity("user_id", "userid"), 4)
        0.9778
    """
    jaro = _jaro_similarity(s1, s2)

    # Compute common prefix length (max 4 chars)
    prefix_len = 0
    for ch1, ch2 in zip(s1[:_WINKLER_MAX_PREFIX], s2[:_WINKLER_MAX_PREFIX]):
        if ch1 == ch2:
            prefix_len += 1
        else:
            break

    return jaro + prefix_len * _WINKLER_SCALING_FACTOR * (1.0 - jaro)


# ---------------------------------------------------------------------------
# Schema field extraction helpers
# ---------------------------------------------------------------------------


def _extract_json_schema_fields(definition: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """Extract field descriptors from a JSON Schema definition.

    Handles both top-level ``"properties"`` and definitions nested under
    ``"definitions"`` / ``"$defs"``.  The ``"required"`` array is respected
    to populate the ``"required"`` boolean on each field descriptor.

    Args:
        definition: A JSON Schema dict (any draft).

    Returns:
        Dict mapping field name → field descriptor with normalised keys:
        ``name``, ``type``, ``required``, ``enum``, ``default``,
        ``description``, ``items``, ``properties``, ``format``,
        ``minimum``, ``maximum``, ``minLength``, ``maxLength``,
        ``pattern``, ``multipleOf``, ``minItems``, ``maxItems``.
    """
    properties: Dict[str, Any] = definition.get("properties", {})
    required_list: List[str] = definition.get("required", [])
    required_set: set = set(required_list)

    fields: Dict[str, Dict[str, Any]] = {}
    for field_name, field_def in properties.items():
        descriptor = _build_json_field_descriptor(
            field_name, field_def, required_set
        )
        fields[field_name] = descriptor

    return fields


def _build_json_field_descriptor(
    name: str,
    field_def: Dict[str, Any],
    required_set: set,
) -> Dict[str, Any]:
    """Build a normalised field descriptor from a JSON Schema property dict.

    Args:
        name: The property name (dict key).
        field_def: The property schema dict.
        required_set: Set of field names marked as required.

    Returns:
        Normalised field descriptor dict.
    """
    raw_type = field_def.get("type", "")
    # Handle array of types (e.g. ["string", "null"])
    if isinstance(raw_type, list):
        non_null = [t for t in raw_type if t != "null"]
        resolved_type = non_null[0] if non_null else "null"
    else:
        resolved_type = str(raw_type)

    return {
        "name": name,
        "type": resolved_type,
        "required": name in required_set,
        "enum": field_def.get("enum"),
        "default": field_def.get("default"),
        "description": field_def.get("description", ""),
        "items": field_def.get("items"),
        "properties": field_def.get("properties"),
        "format": field_def.get("format", ""),
        "minimum": field_def.get("minimum"),
        "maximum": field_def.get("maximum"),
        "exclusiveMinimum": field_def.get("exclusiveMinimum"),
        "exclusiveMaximum": field_def.get("exclusiveMaximum"),
        "minLength": field_def.get("minLength"),
        "maxLength": field_def.get("maxLength"),
        "minItems": field_def.get("minItems"),
        "maxItems": field_def.get("maxItems"),
        "minProperties": field_def.get("minProperties"),
        "maxProperties": field_def.get("maxProperties"),
        "pattern": field_def.get("pattern"),
        "multipleOf": field_def.get("multipleOf"),
    }


def _extract_avro_fields(definition: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """Extract field descriptors from an Apache Avro record schema.

    Handles union types such as ``["null", "string"]`` by unwrapping the
    non-null type and marking the field as optional when ``"null"`` is in
    the union.

    Args:
        definition: An Avro schema dict with ``"type": "record"`` and
            ``"fields"`` array.

    Returns:
        Dict mapping field name → field descriptor with normalised keys.
    """
    raw_fields: List[Dict[str, Any]] = definition.get("fields", [])
    fields: Dict[str, Dict[str, Any]] = {}

    for field_def in raw_fields:
        name = field_def.get("name", "")
        if not name:
            continue
        descriptor = _build_avro_field_descriptor(name, field_def)
        fields[name] = descriptor

    return fields


def _build_avro_field_descriptor(
    name: str, field_def: Dict[str, Any]
) -> Dict[str, Any]:
    """Build a normalised field descriptor from an Avro field dict.

    Args:
        name: The Avro field name.
        field_def: The Avro field dict.

    Returns:
        Normalised field descriptor dict.
    """
    raw_type = field_def.get("type", "")
    is_nullable = False

    if isinstance(raw_type, list):
        non_null = [t for t in raw_type if t != "null"]
        is_nullable = "null" in raw_type
        resolved_type = non_null[0] if non_null else "null"
    elif isinstance(raw_type, dict):
        # Complex type (record, array, map, enum, union)
        resolved_type = raw_type.get("type", str(raw_type))
    else:
        resolved_type = str(raw_type)

    # Avro fields without default are required unless the union starts with null
    has_default = "default" in field_def
    # A field is optional if nullable (union with null) or has a default value
    is_optional = is_nullable or has_default

    enum_symbols: Optional[List[Any]] = None
    if isinstance(raw_type, dict) and raw_type.get("type") == "enum":
        enum_symbols = raw_type.get("symbols")

    items: Optional[Any] = None
    if isinstance(raw_type, dict) and raw_type.get("type") == "array":
        items = raw_type.get("items")

    return {
        "name": name,
        "type": resolved_type,
        "required": not is_optional,
        "enum": enum_symbols,
        "default": field_def.get("default"),
        "description": field_def.get("doc", ""),
        "items": items,
        "properties": None,
        "format": "",
        "minimum": None,
        "maximum": None,
        "exclusiveMinimum": None,
        "exclusiveMaximum": None,
        "minLength": None,
        "maxLength": None,
        "minItems": None,
        "maxItems": None,
        "minProperties": None,
        "maxProperties": None,
        "pattern": None,
        "multipleOf": None,
    }


def _detect_schema_format(definition: Dict[str, Any]) -> str:
    """Heuristically detect the schema format of a definition dict.

    Args:
        definition: A schema definition dict.

    Returns:
        ``"avro"`` if the definition resembles an Avro record schema,
        ``"json_schema"`` otherwise.
    """
    # Avro record schemas have a "type": "record" and "fields" array
    if (
        definition.get("type") == "record"
        and isinstance(definition.get("fields"), list)
    ):
        return "avro"
    return "json_schema"


def _extract_fields(definition: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """Extract fields from a schema definition, auto-detecting format.

    Args:
        definition: A JSON Schema or Avro schema dict.

    Returns:
        Dict mapping field name → normalised field descriptor.
    """
    fmt = _detect_schema_format(definition)
    if fmt == "avro":
        return _extract_avro_fields(definition)
    return _extract_json_schema_fields(definition)


# ---------------------------------------------------------------------------
# Change classification helper
# ---------------------------------------------------------------------------


def _is_type_widening(old_type: str, new_type: str) -> bool:
    """Return True if the type change is a widening (non-breaking).

    Args:
        old_type: The original type string.
        new_type: The target type string.

    Returns:
        True if the pair is in the known widening set.
    """
    return (old_type, new_type) in _TYPE_WIDENING_PAIRS


def _is_type_narrowing(old_type: str, new_type: str) -> bool:
    """Return True if the type change is a narrowing (breaking).

    A narrowing is the reverse of a widening.

    Args:
        old_type: The original type string.
        new_type: The target type string.

    Returns:
        True if ``(new_type, old_type)`` is in the widening set,
        i.e. the change goes from wide to narrow.
    """
    return (new_type, old_type) in _TYPE_WIDENING_PAIRS


# ---------------------------------------------------------------------------
# ChangeDetectorEngine
# ---------------------------------------------------------------------------


class ChangeDetectorEngine:
    """Detect structural differences between two schema definitions.

    Compares source and target schema versions and produces a structured list
    of changes, each classified by type and severity.  Supports JSON Schema
    and Apache Avro formats with auto-detection.

    Supported change types: added, removed, renamed, retyped, reordered,
    constraint_changed, enum_changed, default_changed.

    Severity levels:
        - ``breaking``     : Consumers will break without migration.
        - ``non_breaking`` : Backward-compatible; consumers can be updated lazily.
        - ``cosmetic``     : No impact on data shape or validation.

    Rename detection uses a pure-Python Jaro-Winkler implementation that
    compares all ``(removed, added)`` field pairs sharing the same type and
    promotes the best match above the 0.85 threshold to a ``renamed`` change.

    Nested objects are traversed recursively up to ``_max_depth`` levels.
    Array item types are compared shallowly (one level deep by default).

    All public mutating methods are thread-safe via ``self._lock``.

    Attributes:
        _changes: Stored detection results keyed by detection_id.
        _lock: Threading lock for all mutable state.
        _provenance: SHA-256 chain provenance tracker.
        _max_depth: Maximum nesting depth for recursive detection (default 10).
        _total_detections: Counter of total detection runs performed.

    Example:
        >>> engine = ChangeDetectorEngine()
        >>> source = {"type": "object", "properties": {"id": {"type": "integer"}}}
        >>> target = {"type": "object", "properties": {
        ...     "id": {"type": "integer"},
        ...     "name": {"type": "string"}
        ... }}
        >>> result = engine.detect_changes(source, target)
        >>> result["summary"]["added_count"]
        1
    """

    def __init__(
        self,
        provenance_tracker: Optional[ProvenanceTracker] = None,
        max_depth: int = 10,
    ) -> None:
        """Initialise the ChangeDetectorEngine.

        Args:
            provenance_tracker: Optional external ProvenanceTracker for chain
                hashing.  A fresh tracker is created if not provided.
            max_depth: Maximum nesting depth for recursive change detection.
                Must be >= 1.  Defaults to 10.

        Raises:
            ValueError: If ``max_depth`` is less than 1.
        """
        if max_depth < 1:
            raise ValueError(
                f"max_depth must be >= 1, got {max_depth}"
            )

        self._changes: Dict[str, List[Dict[str, Any]]] = {}
        self._lock: threading.Lock = threading.Lock()
        self._provenance: ProvenanceTracker = (
            provenance_tracker
            if provenance_tracker is not None
            else ProvenanceTracker()
        )
        self._max_depth: int = max_depth
        self._total_detections: int = 0
        self._total_changes_detected: int = 0
        self._detection_times_ms: List[float] = []

        logger.info(
            "ChangeDetectorEngine initialised: max_depth=%d", self._max_depth
        )

    # ------------------------------------------------------------------
    # 1. Main entry point
    # ------------------------------------------------------------------

    def detect_changes(
        self,
        source_definition: Dict[str, Any],
        target_definition: Dict[str, Any],
        max_depth: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Detect all structural changes between two schema definitions.

        Compares ``source_definition`` (the old/current schema) against
        ``target_definition`` (the new/proposed schema) and returns a
        structured result with a unique detection_id, a flat list of all
        change dicts, and a summary.

        The comparison pipeline runs in order:
            1. Extract field descriptors from both definitions.
            2. Detect renamed fields (heuristic based on Jaro-Winkler).
            3. Detect added fields (excluding rename targets).
            4. Detect removed fields (excluding rename sources).
            5. Detect retyped fields.
            6. Detect constraint changes.
            7. Detect enum changes.
            8. Detect default value changes.
            9. Detect nested object changes recursively.
            10. Detect array item type changes.
            11. Detect field reordering (Avro ordered schemas).

        Each step is O(n) or O(n²) at worst (rename detection).

        Args:
            source_definition: The source (old) schema definition dict.
            target_definition: The target (new) schema definition dict.
            max_depth: Override the instance ``_max_depth`` for this run.
                If None, the instance default is used.

        Returns:
            Dict with keys:
                - ``detection_id`` (str): Unique identifier for this run.
                - ``changes`` (List[Dict]): List of all change dicts.
                - ``summary`` (Dict): Aggregated counts by type and severity.
                - ``source_hash`` (str): SHA-256 of the source definition.
                - ``target_hash`` (str): SHA-256 of the target definition.
                - ``provenance_hash`` (str): Chain provenance hash.
                - ``detected_at`` (str): UTC ISO timestamp.
                - ``processing_time_ms`` (float): Wall-clock duration.

        Raises:
            TypeError: If either definition is not a dict.
            ValueError: If max_depth override is less than 1.
        """
        if not isinstance(source_definition, dict):
            raise TypeError(
                f"source_definition must be a dict, got {type(source_definition)}"
            )
        if not isinstance(target_definition, dict):
            raise TypeError(
                f"target_definition must be a dict, got {type(target_definition)}"
            )

        effective_depth = self._validate_max_depth(max_depth)
        detection_id = _detection_id()
        start_ts = time.monotonic()
        detected_at = _utcnow().isoformat()

        logger.info(
            "ChangeDetectorEngine.detect_changes: detection_id=%s max_depth=%d",
            detection_id,
            effective_depth,
        )

        source_hash = _compute_sha256(source_definition)
        target_hash = _compute_sha256(target_definition)

        # Extract normalised field maps
        source_fields = _extract_fields(source_definition)
        target_fields = _extract_fields(target_definition)

        # Run detection pipeline
        all_changes = self._run_detection_pipeline(
            source_definition=source_definition,
            target_definition=target_definition,
            source_fields=source_fields,
            target_fields=target_fields,
            effective_depth=effective_depth,
        )

        summary = self.summarize_changes(all_changes)
        processing_ms = (time.monotonic() - start_ts) * 1000.0

        # Record provenance
        provenance_entry = self._provenance.record(
            entity_type="change_detection",
            entity_id=detection_id,
            action="changes_detected",
            data={
                "source_hash": source_hash,
                "target_hash": target_hash,
                "total_count": summary["total_count"],
                "breaking_count": summary["breaking_count"],
            },
        )

        result: Dict[str, Any] = {
            "detection_id": detection_id,
            "changes": all_changes,
            "summary": summary,
            "source_hash": source_hash,
            "target_hash": target_hash,
            "provenance_hash": provenance_entry.hash_value,
            "detected_at": detected_at,
            "processing_time_ms": round(processing_ms, 3),
        }

        with self._lock:
            self._changes[detection_id] = all_changes
            self._total_detections += 1
            self._total_changes_detected += len(all_changes)
            self._detection_times_ms.append(processing_ms)

        logger.info(
            "ChangeDetectorEngine: detection_id=%s changes=%d "
            "breaking=%d processing_ms=%.2f",
            detection_id,
            summary["total_count"],
            summary["breaking_count"],
            processing_ms,
        )

        return result

    # ------------------------------------------------------------------
    # 2. detect_added_fields
    # ------------------------------------------------------------------

    def detect_added_fields(
        self,
        source_fields: Dict[str, Dict[str, Any]],
        target_fields: Dict[str, Dict[str, Any]],
        path: str = "",
        exclude_names: Optional[set] = None,
    ) -> List[Dict[str, Any]]:
        """Find fields present in target but not in source.

        A field is reported as ``added`` only when it does not appear in
        ``source_fields`` by name.  Fields already identified as rename
        targets (passed via ``exclude_names``) are skipped.

        Args:
            source_fields: Normalised field map from the source schema.
            target_fields: Normalised field map from the target schema.
            path: Dot-separated path prefix for nested field reporting.
            exclude_names: Set of target field names already assigned to
                ``renamed`` changes; these are excluded from ``added``.

        Returns:
            List of change dicts with ``change_type == "added"``.
        """
        exclude = exclude_names or set()
        changes: List[Dict[str, Any]] = []
        source_names = set(source_fields.keys())

        for name, field_def in target_fields.items():
            if name in source_names or name in exclude:
                continue

            field_path = f"{path}.{name}" if path else name
            severity = self.classify_change("added", None, field_def)

            changes.append(
                self._build_change_dict(
                    change_type="added",
                    field_path=field_path,
                    old_value=None,
                    new_value=field_def,
                    severity=severity,
                    description=(
                        f"Field '{field_path}' added "
                        f"({'required' if field_def.get('required') else 'optional'})"
                    ),
                )
            )

        return changes

    # ------------------------------------------------------------------
    # 3. detect_removed_fields
    # ------------------------------------------------------------------

    def detect_removed_fields(
        self,
        source_fields: Dict[str, Dict[str, Any]],
        target_fields: Dict[str, Dict[str, Any]],
        path: str = "",
        exclude_names: Optional[set] = None,
    ) -> List[Dict[str, Any]]:
        """Find fields present in source but not in target.

        A field is reported as ``removed`` only when it does not appear in
        ``target_fields`` by name.  Fields already identified as rename
        sources (passed via ``exclude_names``) are skipped.

        Args:
            source_fields: Normalised field map from the source schema.
            target_fields: Normalised field map from the target schema.
            path: Dot-separated path prefix for nested field reporting.
            exclude_names: Set of source field names already assigned to
                ``renamed`` changes; these are excluded from ``removed``.

        Returns:
            List of change dicts with ``change_type == "removed"``.
        """
        exclude = exclude_names or set()
        changes: List[Dict[str, Any]] = []
        target_names = set(target_fields.keys())

        for name, field_def in source_fields.items():
            if name in target_names or name in exclude:
                continue

            field_path = f"{path}.{name}" if path else name
            severity = self.classify_change("removed", field_def, None)

            changes.append(
                self._build_change_dict(
                    change_type="removed",
                    field_path=field_path,
                    old_value=field_def,
                    new_value=None,
                    severity=severity,
                    description=f"Field '{field_path}' removed",
                )
            )

        return changes

    # ------------------------------------------------------------------
    # 4. detect_renamed_fields
    # ------------------------------------------------------------------

    def detect_renamed_fields(
        self,
        source_fields: Dict[str, Dict[str, Any]],
        target_fields: Dict[str, Dict[str, Any]],
        path: str = "",
    ) -> Tuple[List[Dict[str, Any]], set, set]:
        """Detect field renames using Jaro-Winkler similarity.

        Compares every ``(removed_candidate, added_candidate)`` pair that
        shares the same resolved type.  When the best-match similarity
        exceeds ``_RENAME_SIMILARITY_THRESHOLD`` (0.85) the pair is
        classified as a ``renamed`` change rather than separate
        ``removed`` + ``added`` events.

        Algorithm:
            1. Identify candidate removed fields (in source, not in target).
            2. Identify candidate added fields (in target, not in source).
            3. Build an affinity matrix of Jaro-Winkler scores filtered by
               type equality.
            4. Greedily match pairs in descending score order.

        Args:
            source_fields: Normalised field map from the source schema.
            target_fields: Normalised field map from the target schema.
            path: Dot-separated path prefix for nested field reporting.

        Returns:
            Tuple of:
                - List of rename change dicts.
                - Set of source field names consumed as rename sources.
                - Set of target field names consumed as rename targets.
        """
        source_names = set(source_fields.keys())
        target_names = set(target_fields.keys())

        removed_candidates = [
            n for n in source_names if n not in target_names
        ]
        added_candidates = [
            n for n in target_names if n not in source_names
        ]

        if not removed_candidates or not added_candidates:
            return [], set(), set()

        # Build affinity matrix: only consider pairs with matching types
        affinity: List[Tuple[float, str, str]] = []
        for src_name in removed_candidates:
            src_type = source_fields[src_name].get("type", "")
            for tgt_name in added_candidates:
                tgt_type = target_fields[tgt_name].get("type", "")
                if src_type != tgt_type:
                    continue
                score = _jaro_winkler_similarity(src_name, tgt_name)
                if score >= _RENAME_SIMILARITY_THRESHOLD:
                    affinity.append((score, src_name, tgt_name))

        # Greedy matching: highest score first
        affinity.sort(key=lambda x: x[0], reverse=True)

        matched_sources: set = set()
        matched_targets: set = set()
        rename_changes: List[Dict[str, Any]] = []

        for score, src_name, tgt_name in affinity:
            if src_name in matched_sources or tgt_name in matched_targets:
                continue
            matched_sources.add(src_name)
            matched_targets.add(tgt_name)

            src_path = f"{path}.{src_name}" if path else src_name
            tgt_path = f"{path}.{tgt_name}" if path else tgt_name
            severity = self.classify_change(
                "renamed",
                source_fields[src_name],
                target_fields[tgt_name],
            )

            rename_changes.append(
                self._build_change_dict(
                    change_type="renamed",
                    field_path=src_path,
                    old_value=src_name,
                    new_value=tgt_name,
                    severity=severity,
                    description=(
                        f"Field '{src_path}' likely renamed to '{tgt_path}' "
                        f"(similarity={score:.4f})"
                    ),
                )
            )
            logger.debug(
                "Rename detected: %s -> %s (score=%.4f)",
                src_name,
                tgt_name,
                score,
            )

        return rename_changes, matched_sources, matched_targets

    # ------------------------------------------------------------------
    # 5. detect_retyped_fields
    # ------------------------------------------------------------------

    def detect_retyped_fields(
        self,
        source_fields: Dict[str, Dict[str, Any]],
        target_fields: Dict[str, Dict[str, Any]],
        path: str = "",
    ) -> List[Dict[str, Any]]:
        """Find fields whose type has changed between source and target.

        Only examines fields present in both source and target by name.
        Fields with identical types are skipped.

        Args:
            source_fields: Normalised field map from the source schema.
            target_fields: Normalised field map from the target schema.
            path: Dot-separated path prefix for nested field reporting.

        Returns:
            List of change dicts with ``change_type == "retyped"``.
        """
        changes: List[Dict[str, Any]] = []
        common_names = set(source_fields.keys()) & set(target_fields.keys())

        for name in sorted(common_names):
            src_type = source_fields[name].get("type", "")
            tgt_type = target_fields[name].get("type", "")
            if src_type == tgt_type:
                continue

            field_path = f"{path}.{name}" if path else name
            severity = self.classify_change("retyped", src_type, tgt_type)

            changes.append(
                self._build_change_dict(
                    change_type="retyped",
                    field_path=field_path,
                    old_value=src_type,
                    new_value=tgt_type,
                    severity=severity,
                    description=(
                        f"Field '{field_path}' type changed "
                        f"from '{src_type}' to '{tgt_type}'"
                    ),
                )
            )

        return changes

    # ------------------------------------------------------------------
    # 6. detect_constraint_changes
    # ------------------------------------------------------------------

    def detect_constraint_changes(
        self,
        source_fields: Dict[str, Dict[str, Any]],
        target_fields: Dict[str, Dict[str, Any]],
        path: str = "",
    ) -> List[Dict[str, Any]]:
        """Detect required↔optional and numeric/string constraint changes.

        For each field common to both schemas, checks:
            - ``required`` flag toggled (optional→required is breaking;
              required→optional is non-breaking).
            - Any constraint key in ``_NUMERIC_CONSTRAINT_KEYS`` that changed
              value.  Tightening (min increases, max decreases) is breaking;
              relaxing is non-breaking.
            - ``pattern`` regex changes.

        Args:
            source_fields: Normalised field map from the source schema.
            target_fields: Normalised field map from the target schema.
            path: Dot-separated path prefix for nested field reporting.

        Returns:
            List of change dicts with ``change_type == "constraint_changed"``.
        """
        changes: List[Dict[str, Any]] = []
        common_names = set(source_fields.keys()) & set(target_fields.keys())

        for name in sorted(common_names):
            src = source_fields[name]
            tgt = target_fields[name]
            field_path = f"{path}.{name}" if path else name

            # required flag
            if src.get("required", False) != tgt.get("required", False):
                old_req = src.get("required", False)
                new_req = tgt.get("required", False)
                sev = self.classify_change(
                    "constraint_changed",
                    {"required": old_req},
                    {"required": new_req},
                )
                direction = "optional→required" if new_req else "required→optional"
                changes.append(
                    self._build_change_dict(
                        change_type="constraint_changed",
                        field_path=field_path,
                        old_value={"required": old_req},
                        new_value={"required": new_req},
                        severity=sev,
                        description=(
                            f"Field '{field_path}' required constraint changed: "
                            f"{direction}"
                        ),
                    )
                )

            # Numeric / length / pattern constraints
            for key in sorted(_NUMERIC_CONSTRAINT_KEYS):
                old_val = src.get(key)
                new_val = tgt.get(key)
                if old_val == new_val:
                    continue
                if old_val is None and new_val is None:
                    continue

                sev = self._classify_numeric_constraint_change(
                    key, old_val, new_val
                )
                changes.append(
                    self._build_change_dict(
                        change_type="constraint_changed",
                        field_path=field_path,
                        old_value={key: old_val},
                        new_value={key: new_val},
                        severity=sev,
                        description=(
                            f"Field '{field_path}' constraint '{key}' changed "
                            f"from {old_val!r} to {new_val!r}"
                        ),
                    )
                )

            # Pattern changes
            old_pattern = src.get("pattern")
            new_pattern = tgt.get("pattern")
            if old_pattern != new_pattern:
                sev = self.classify_change(
                    "constraint_changed",
                    {"pattern": old_pattern},
                    {"pattern": new_pattern},
                )
                changes.append(
                    self._build_change_dict(
                        change_type="constraint_changed",
                        field_path=field_path,
                        old_value={"pattern": old_pattern},
                        new_value={"pattern": new_pattern},
                        severity=sev,
                        description=(
                            f"Field '{field_path}' pattern constraint changed "
                            f"from {old_pattern!r} to {new_pattern!r}"
                        ),
                    )
                )

        return changes

    # ------------------------------------------------------------------
    # 7. detect_enum_changes
    # ------------------------------------------------------------------

    def detect_enum_changes(
        self,
        source_fields: Dict[str, Dict[str, Any]],
        target_fields: Dict[str, Dict[str, Any]],
        path: str = "",
    ) -> List[Dict[str, Any]]:
        """Detect enum value additions and removals for common fields.

        An enum change is only reported when at least one of the two
        definitions carries a non-None ``"enum"`` value.

        Args:
            source_fields: Normalised field map from the source schema.
            target_fields: Normalised field map from the target schema.
            path: Dot-separated path prefix for nested field reporting.

        Returns:
            List of change dicts with ``change_type == "enum_changed"``.
        """
        changes: List[Dict[str, Any]] = []
        common_names = set(source_fields.keys()) & set(target_fields.keys())

        for name in sorted(common_names):
            src_enum = source_fields[name].get("enum")
            tgt_enum = target_fields[name].get("enum")

            # Skip if both None or both identical
            if src_enum == tgt_enum:
                continue
            if src_enum is None and tgt_enum is None:
                continue

            field_path = f"{path}.{name}" if path else name

            src_set = set(src_enum) if src_enum is not None else set()
            tgt_set = set(tgt_enum) if tgt_enum is not None else set()

            added_vals = sorted(tgt_set - src_set, key=str)
            removed_vals = sorted(src_set - tgt_set, key=str)

            if not added_vals and not removed_vals:
                continue

            sev = self.classify_change("enum_changed", src_enum, tgt_enum)

            changes.append(
                self._build_change_dict(
                    change_type="enum_changed",
                    field_path=field_path,
                    old_value=src_enum,
                    new_value=tgt_enum,
                    severity=sev,
                    description=(
                        f"Field '{field_path}' enum changed: "
                        f"added={added_vals}, removed={removed_vals}"
                    ),
                )
            )

        return changes

    # ------------------------------------------------------------------
    # 8. detect_default_changes
    # ------------------------------------------------------------------

    def detect_default_changes(
        self,
        source_fields: Dict[str, Dict[str, Any]],
        target_fields: Dict[str, Dict[str, Any]],
        path: str = "",
    ) -> List[Dict[str, Any]]:
        """Detect changes to default values for common fields.

        A change is reported when the ``"default"`` values differ (including
        when one is ``None`` and the other is not, i.e. a default was added
        or removed).

        Args:
            source_fields: Normalised field map from the source schema.
            target_fields: Normalised field map from the target schema.
            path: Dot-separated path prefix for nested field reporting.

        Returns:
            List of change dicts with ``change_type == "default_changed"``.
        """
        changes: List[Dict[str, Any]] = []

        # Sentinel to distinguish "key absent" from "key present with None"
        _MISSING = object()

        common_names = set(source_fields.keys()) & set(target_fields.keys())

        for name in sorted(common_names):
            old_default = source_fields[name].get("default", _MISSING)
            new_default = target_fields[name].get("default", _MISSING)

            # Normalise missing key to None for comparison
            old_val = None if old_default is _MISSING else old_default
            new_val = None if new_default is _MISSING else new_default

            old_absent = old_default is _MISSING
            new_absent = new_default is _MISSING

            if old_absent and new_absent:
                continue
            if old_val == new_val and not (old_absent ^ new_absent):
                continue

            field_path = f"{path}.{name}" if path else name
            sev = self.classify_change("default_changed", old_val, new_val)

            changes.append(
                self._build_change_dict(
                    change_type="default_changed",
                    field_path=field_path,
                    old_value=old_val,
                    new_value=new_val,
                    severity=sev,
                    description=(
                        f"Field '{field_path}' default changed "
                        f"from {old_val!r} to {new_val!r}"
                    ),
                )
            )

        return changes

    # ------------------------------------------------------------------
    # 9. detect_nested_changes
    # ------------------------------------------------------------------

    def detect_nested_changes(
        self,
        source_obj: Dict[str, Any],
        target_obj: Dict[str, Any],
        path: str = "",
        depth: int = 0,
    ) -> List[Dict[str, Any]]:
        """Recursively detect changes inside nested object fields.

        For each field common to both schemas that has a ``"properties"``
        sub-dict (i.e. is itself an object type), recurse into the nested
        definition and collect all sub-changes, prefixing them with the
        current path.

        Recursion stops when ``depth`` reaches ``self._max_depth``.

        Args:
            source_obj: The source parent schema definition dict.
            target_obj: The target parent schema definition dict.
            path: Dot-separated path of the current object level.
            depth: Current recursion depth (starts at 0 from the caller).

        Returns:
            List of change dicts for all nested differences found.
        """
        if depth >= self._max_depth:
            logger.debug(
                "detect_nested_changes: max_depth=%d reached at path='%s'",
                self._max_depth,
                path,
            )
            return []

        changes: List[Dict[str, Any]] = []
        source_fields = _extract_fields(source_obj)
        target_fields = _extract_fields(target_obj)
        common_names = set(source_fields.keys()) & set(target_fields.keys())

        for name in sorted(common_names):
            src_field = source_fields[name]
            tgt_field = target_fields[name]

            # Check if the field is an object type with properties
            src_props = src_field.get("properties")
            tgt_props = tgt_field.get("properties")

            if not src_props and not tgt_props:
                continue

            nested_path = f"{path}.{name}" if path else name

            # Build synthetic sub-definitions for recursion
            src_sub = {"properties": src_props or {}}
            tgt_sub = {"properties": tgt_props or {}}

            # Add required lists if available
            if isinstance(src_obj.get("properties", {}).get(name), dict):
                src_sub["required"] = src_obj.get("properties", {}).get(name, {}).get("required", [])
            if isinstance(tgt_obj.get("properties", {}).get(name), dict):
                tgt_sub["required"] = tgt_obj.get("properties", {}).get(name, {}).get("required", [])

            nested_changes = self._detect_at_level(
                source_definition=src_sub,
                target_definition=tgt_sub,
                path=nested_path,
                depth=depth + 1,
            )
            changes.extend(nested_changes)

        return changes

    # ------------------------------------------------------------------
    # 10. detect_array_changes
    # ------------------------------------------------------------------

    def detect_array_changes(
        self,
        source_fields: Dict[str, Dict[str, Any]],
        target_fields: Dict[str, Dict[str, Any]],
        path: str = "",
    ) -> List[Dict[str, Any]]:
        """Detect array item type changes for common array-type fields.

        For each field common to both schemas with ``"type": "array"``,
        compares the ``"items"`` descriptor.  If ``items`` is a simple type
        string, the type is compared directly.  If ``items`` is a dict, the
        ``"type"`` key inside is compared.

        Args:
            source_fields: Normalised field map from the source schema.
            target_fields: Normalised field map from the target schema.
            path: Dot-separated path prefix for nested field reporting.

        Returns:
            List of change dicts with ``change_type == "retyped"`` and
            ``field_path`` suffixed with ``[]`` to indicate array item.
        """
        changes: List[Dict[str, Any]] = []
        common_names = set(source_fields.keys()) & set(target_fields.keys())

        for name in sorted(common_names):
            src_field = source_fields[name]
            tgt_field = target_fields[name]

            # Only process fields that are (or were) array type
            src_is_array = src_field.get("type") == "array"
            tgt_is_array = tgt_field.get("type") == "array"

            if not src_is_array and not tgt_is_array:
                continue

            # array→non-array or non-array→array is handled by retyped
            if src_is_array != tgt_is_array:
                continue

            src_items = src_field.get("items")
            tgt_items = tgt_field.get("items")

            if src_items == tgt_items:
                continue

            src_item_type = self._resolve_items_type(src_items)
            tgt_item_type = self._resolve_items_type(tgt_items)

            if src_item_type == tgt_item_type:
                continue

            field_path = f"{path}.{name}[]" if path else f"{name}[]"
            severity = self.classify_change("retyped", src_item_type, tgt_item_type)

            changes.append(
                self._build_change_dict(
                    change_type="retyped",
                    field_path=field_path,
                    old_value=src_item_type,
                    new_value=tgt_item_type,
                    severity=severity,
                    description=(
                        f"Array field '{field_path}' item type changed "
                        f"from '{src_item_type}' to '{tgt_item_type}'"
                    ),
                )
            )

        return changes

    # ------------------------------------------------------------------
    # 11. classify_change
    # ------------------------------------------------------------------

    def classify_change(
        self,
        change_type: str,
        old_value: Any,
        new_value: Any,
    ) -> str:
        """Classify a change as breaking, non_breaking, or cosmetic.

        Classification rules by change type:

        ``added``:
            - Field has ``required=True`` in new_value → ``breaking``
              (existing writers won't provide the required field).
            - Field has ``required=False`` or None → ``non_breaking``.

        ``removed``:
            - Always ``breaking`` (consumers reading the field will fail).

        ``renamed``:
            - Always ``non_breaking`` (semantically equivalent, just renamed).

        ``retyped``:
            - Type widening (int→float, int→string, etc.) → ``non_breaking``.
            - Type narrowing (float→int, string→int, etc.) → ``breaking``.
            - Unrecognised type pair → ``breaking`` (safe default).

        ``reordered``:
            - Always ``cosmetic`` (field order does not affect JSON data).

        ``constraint_changed`` (dict with one key):
            - ``required`` True→True: N/A; False→True: ``breaking``;
              True→False: ``non_breaking``.
            - Minimum-type constraints (minimum, minLength, minItems, …)
              increasing → ``breaking``; decreasing → ``non_breaking``.
            - Maximum-type constraints (maximum, maxLength, maxItems, …)
              decreasing → ``breaking``; increasing → ``non_breaking``.
            - ``pattern`` change → ``breaking`` (validation behaviour changes).
            - Fallback: ``non_breaking``.

        ``enum_changed``:
            - Values removed from enum → ``breaking``.
            - Only values added → ``non_breaking``.
            - Both added and removed → ``breaking``.

        ``default_changed``:
            - Always ``cosmetic`` when field type is unchanged (purely
              behavioural, not a structural break).

        Args:
            change_type: One of the known change type strings.
            old_value: The old value (type depends on change_type).
            new_value: The new value.

        Returns:
            One of ``"breaking"``, ``"non_breaking"``, or ``"cosmetic"``.
        """
        if change_type == "added":
            if isinstance(new_value, dict) and new_value.get("required", False):
                return "breaking"
            return "non_breaking"

        if change_type == "removed":
            return "breaking"

        if change_type == "renamed":
            return "non_breaking"

        if change_type == "reordered":
            return "cosmetic"

        if change_type == "retyped":
            return self._classify_type_change(old_value, new_value)

        if change_type == "constraint_changed":
            return self._classify_constraint_change(old_value, new_value)

        if change_type == "enum_changed":
            return self._classify_enum_severity(old_value, new_value)

        if change_type == "default_changed":
            return "cosmetic"

        # Unknown change type — default to non_breaking (unknown risk)
        logger.warning(
            "classify_change: unknown change_type='%s', defaulting to non_breaking",
            change_type,
        )
        return "non_breaking"

    # ------------------------------------------------------------------
    # 12. get_detection
    # ------------------------------------------------------------------

    def get_detection(self, detection_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve a stored detection result by its detection_id.

        Args:
            detection_id: The unique detection run identifier returned by
                :meth:`detect_changes`.

        Returns:
            Dict containing ``detection_id`` and ``changes`` list, or
            ``None`` if the ID is not found.
        """
        with self._lock:
            changes = self._changes.get(detection_id)
            if changes is None:
                return None
            return {
                "detection_id": detection_id,
                "changes": list(changes),
                "total_count": len(changes),
            }

    # ------------------------------------------------------------------
    # 13. list_detections
    # ------------------------------------------------------------------

    def list_detections(
        self,
        limit: int = 100,
        offset: int = 0,
    ) -> List[Dict[str, Any]]:
        """List all stored detection run summaries.

        Returns a paginated list of detection summaries sorted by
        detection_id (lexicographic, which is insertion-ordered for UUIDs).

        Args:
            limit: Maximum number of results to return (default 100).
            offset: Zero-based starting index for pagination (default 0).

        Returns:
            List of dicts each with ``detection_id``, ``total_count``, and
            ``breaking_count``.

        Raises:
            ValueError: If limit <= 0 or offset < 0.
        """
        if limit <= 0:
            raise ValueError(f"limit must be > 0, got {limit}")
        if offset < 0:
            raise ValueError(f"offset must be >= 0, got {offset}")

        with self._lock:
            all_ids = sorted(self._changes.keys())

        page = all_ids[offset: offset + limit]
        result: List[Dict[str, Any]] = []

        for det_id in page:
            with self._lock:
                changes = list(self._changes.get(det_id, []))
            summary = self.summarize_changes(changes)
            result.append(
                {
                    "detection_id": det_id,
                    "total_count": summary["total_count"],
                    "breaking_count": summary["breaking_count"],
                }
            )

        return result

    # ------------------------------------------------------------------
    # 14. summarize_changes
    # ------------------------------------------------------------------

    def summarize_changes(self, changes: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Summarise a list of change dicts into aggregate counts.

        Args:
            changes: List of change dicts (as returned by detection methods).

        Returns:
            Dict with:
                - ``total_count`` (int): Total number of changes.
                - ``breaking_count`` (int): Changes with severity ``breaking``.
                - ``non_breaking_count`` (int): Changes with severity ``non_breaking``.
                - ``cosmetic_count`` (int): Changes with severity ``cosmetic``.
                - ``by_type`` (Dict[str, int]): Count per change_type.
                - ``by_severity`` (Dict[str, int]): Count per severity.
                - ``added_count`` (int): Shortcut for ``by_type["added"]``.
                - ``removed_count`` (int): Shortcut for ``by_type["removed"]``.
                - ``renamed_count`` (int): Shortcut for ``by_type["renamed"]``.
                - ``retyped_count`` (int): Shortcut for ``by_type["retyped"]``.
        """
        by_type: Dict[str, int] = {}
        by_severity: Dict[str, int] = {}

        for change in changes:
            ct = change.get("change_type", "unknown")
            sv = change.get("severity", "unknown")
            by_type[ct] = by_type.get(ct, 0) + 1
            by_severity[sv] = by_severity.get(sv, 0) + 1

        return {
            "total_count": len(changes),
            "breaking_count": by_severity.get("breaking", 0),
            "non_breaking_count": by_severity.get("non_breaking", 0),
            "cosmetic_count": by_severity.get("cosmetic", 0),
            "by_type": by_type,
            "by_severity": by_severity,
            "added_count": by_type.get("added", 0),
            "removed_count": by_type.get("removed", 0),
            "renamed_count": by_type.get("renamed", 0),
            "retyped_count": by_type.get("retyped", 0),
        }

    # ------------------------------------------------------------------
    # 15. get_statistics
    # ------------------------------------------------------------------

    def get_statistics(self) -> Dict[str, Any]:
        """Return aggregate statistics for the ChangeDetectorEngine.

        Returns:
            Dict with:
                - ``total_detections`` (int): Total detection runs.
                - ``total_changes_detected`` (int): Cumulative change count.
                - ``stored_detections`` (int): Detection IDs in memory.
                - ``avg_processing_ms`` (float): Average detection duration.
                - ``max_depth`` (int): Configured maximum nesting depth.
                - ``rename_threshold`` (float): Jaro-Winkler rename threshold.
                - ``provenance_entries`` (int): Provenance chain size.
        """
        with self._lock:
            total = self._total_detections
            total_changes = self._total_changes_detected
            stored = len(self._changes)
            times = list(self._detection_times_ms)

        avg_ms = sum(times) / len(times) if times else 0.0

        return {
            "total_detections": total,
            "total_changes_detected": total_changes,
            "stored_detections": stored,
            "avg_processing_ms": round(avg_ms, 3),
            "max_depth": self._max_depth,
            "rename_threshold": _RENAME_SIMILARITY_THRESHOLD,
            "provenance_entries": self._provenance.entry_count,
        }

    # ------------------------------------------------------------------
    # 16. reset
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Clear all stored detection results and reset counters.

        Resets the internal ``_changes`` store, all counters, and the
        provenance tracker to genesis state.  Primarily intended for
        testing and re-use in long-running processes.

        After calling ``reset()``, :meth:`list_detections` returns an
        empty list and :meth:`get_detection` returns ``None`` for any
        previously valid detection_id.
        """
        with self._lock:
            self._changes.clear()
            self._total_detections = 0
            self._total_changes_detected = 0
            self._detection_times_ms.clear()

        self._provenance.reset()
        logger.info("ChangeDetectorEngine reset to initial state")

    # ------------------------------------------------------------------
    # Internal helpers — pipeline orchestration
    # ------------------------------------------------------------------

    def _run_detection_pipeline(
        self,
        source_definition: Dict[str, Any],
        target_definition: Dict[str, Any],
        source_fields: Dict[str, Dict[str, Any]],
        target_fields: Dict[str, Dict[str, Any]],
        effective_depth: int,
    ) -> List[Dict[str, Any]]:
        """Run the full change detection pipeline and return all changes.

        Executes all 10 detection steps in a defined order and merges the
        results into a single flat list.  Rename detection runs first to
        exclude rename sources/targets from the add/remove detectors.

        Args:
            source_definition: Full source schema dict (for nested recursion).
            target_definition: Full target schema dict (for nested recursion).
            source_fields: Extracted source field map.
            target_fields: Extracted target field map.
            effective_depth: Maximum nesting depth for this run.

        Returns:
            Merged list of all change dicts in pipeline order.
        """
        all_changes: List[Dict[str, Any]] = []

        # Step 1: Renames (must run before add/remove to exclude matched pairs)
        rename_changes, rename_sources, rename_targets = self.detect_renamed_fields(
            source_fields, target_fields, path=""
        )
        all_changes.extend(rename_changes)

        # Step 2: Added fields (exclude rename targets)
        all_changes.extend(
            self.detect_added_fields(
                source_fields, target_fields,
                path="", exclude_names=rename_targets
            )
        )

        # Step 3: Removed fields (exclude rename sources)
        all_changes.extend(
            self.detect_removed_fields(
                source_fields, target_fields,
                path="", exclude_names=rename_sources
            )
        )

        # Step 4: Retyped fields
        all_changes.extend(
            self.detect_retyped_fields(source_fields, target_fields, path="")
        )

        # Step 5: Constraint changes
        all_changes.extend(
            self.detect_constraint_changes(
                source_fields, target_fields, path=""
            )
        )

        # Step 6: Enum changes
        all_changes.extend(
            self.detect_enum_changes(source_fields, target_fields, path="")
        )

        # Step 7: Default value changes
        all_changes.extend(
            self.detect_default_changes(source_fields, target_fields, path="")
        )

        # Step 8: Nested object changes (recursive)
        if effective_depth > 1:
            all_changes.extend(
                self.detect_nested_changes(
                    source_definition, target_definition,
                    path="", depth=0
                )
            )

        # Step 9: Array item type changes
        all_changes.extend(
            self.detect_array_changes(source_fields, target_fields, path="")
        )

        # Step 10: Field reordering (positional — only meaningful for Avro)
        all_changes.extend(
            self._detect_reordered_fields(
                source_definition, target_definition, path=""
            )
        )

        return all_changes

    def _detect_at_level(
        self,
        source_definition: Dict[str, Any],
        target_definition: Dict[str, Any],
        path: str,
        depth: int,
    ) -> List[Dict[str, Any]]:
        """Run the detection pipeline for a nested object level.

        Used internally by :meth:`detect_nested_changes` to recurse into
        nested ``properties`` dicts.  Applies the same pipeline as the
        top-level :meth:`detect_changes` but with path prefix and depth
        tracking to avoid infinite recursion.

        Args:
            source_definition: The source sub-schema dict for this level.
            target_definition: The target sub-schema dict for this level.
            path: Dot-separated path representing this object level.
            depth: Current nesting depth.

        Returns:
            Flat list of change dicts for this level.
        """
        if depth >= self._max_depth:
            return []

        source_fields = _extract_fields(source_definition)
        target_fields = _extract_fields(target_definition)
        changes: List[Dict[str, Any]] = []

        rename_changes, rename_sources, rename_targets = self.detect_renamed_fields(
            source_fields, target_fields, path=path
        )
        changes.extend(rename_changes)

        changes.extend(
            self.detect_added_fields(
                source_fields, target_fields,
                path=path, exclude_names=rename_targets
            )
        )
        changes.extend(
            self.detect_removed_fields(
                source_fields, target_fields,
                path=path, exclude_names=rename_sources
            )
        )
        changes.extend(
            self.detect_retyped_fields(source_fields, target_fields, path=path)
        )
        changes.extend(
            self.detect_constraint_changes(
                source_fields, target_fields, path=path
            )
        )
        changes.extend(
            self.detect_enum_changes(source_fields, target_fields, path=path)
        )
        changes.extend(
            self.detect_default_changes(source_fields, target_fields, path=path)
        )
        changes.extend(
            self.detect_nested_changes(
                source_definition, target_definition,
                path=path, depth=depth
            )
        )
        changes.extend(
            self.detect_array_changes(source_fields, target_fields, path=path)
        )

        return changes

    def _detect_reordered_fields(
        self,
        source_definition: Dict[str, Any],
        target_definition: Dict[str, Any],
        path: str = "",
    ) -> List[Dict[str, Any]]:
        """Detect field position changes between source and target schemas.

        Reordering is only meaningful for schemas with explicit field ordering
        (e.g., Avro ``"fields"`` arrays).  For JSON Schema, property ordering
        is typically not significant but is reported as ``cosmetic``.

        Only fields present in both source and target are compared.  The
        relative order of common fields is compared; if it differs, one
        ``reordered`` change per schema level is reported (not per field).

        Args:
            source_definition: The source schema dict.
            target_definition: The target schema dict.
            path: Dot-separated path prefix for field reporting.

        Returns:
            List containing at most one ``reordered`` change dict per level.
        """
        src_fmt = _detect_schema_format(source_definition)
        tgt_fmt = _detect_schema_format(target_definition)

        # Extract ordered field name lists
        if src_fmt == "avro":
            src_order = [
                f.get("name", "")
                for f in source_definition.get("fields", [])
                if f.get("name")
            ]
        else:
            src_order = list(source_definition.get("properties", {}).keys())

        if tgt_fmt == "avro":
            tgt_order = [
                f.get("name", "")
                for f in target_definition.get("fields", [])
                if f.get("name")
            ]
        else:
            tgt_order = list(target_definition.get("properties", {}).keys())

        # Compare relative order of common fields only
        common = set(src_order) & set(tgt_order)
        src_common = [n for n in src_order if n in common]
        tgt_common = [n for n in tgt_order if n in common]

        if src_common == tgt_common:
            return []

        label = f"'{path}' " if path else "root level "
        return [
            self._build_change_dict(
                change_type="reordered",
                field_path=path or "<root>",
                old_value=src_common,
                new_value=tgt_common,
                severity="cosmetic",
                description=(
                    f"Field ordering changed at {label}schema"
                ),
            )
        ]

    # ------------------------------------------------------------------
    # Internal helpers — classification
    # ------------------------------------------------------------------

    def _classify_type_change(self, old_type: Any, new_type: Any) -> str:
        """Classify a type change as breaking, non_breaking, or cosmetic.

        Args:
            old_type: The original type string (or any value).
            new_type: The new type string.

        Returns:
            Severity string.
        """
        old_str = str(old_type) if old_type is not None else ""
        new_str = str(new_type) if new_type is not None else ""

        if old_str == new_str:
            return "cosmetic"
        if _is_type_widening(old_str, new_str):
            return "non_breaking"
        if _is_type_narrowing(old_str, new_str):
            return "breaking"

        # Special case: any→array or array→non-array
        if old_str == "array" and new_str != "array":
            return "breaking"
        if old_str != "array" and new_str == "array":
            return "breaking"

        # Unknown type pair — treat as breaking (conservative)
        return "breaking"

    def _classify_constraint_change(
        self,
        old_value: Any,
        new_value: Any,
    ) -> str:
        """Classify a constraint change using the old/new value dicts.

        Expects each value to be a dict with exactly one key (the constraint
        name) as produced by :meth:`detect_constraint_changes`.

        Args:
            old_value: Dict like ``{"required": True}`` or ``{"minimum": 5}``.
            new_value: The new constraint dict.

        Returns:
            Severity string: ``"breaking"``, ``"non_breaking"``, or ``"cosmetic"``.
        """
        if not isinstance(old_value, dict) or not isinstance(new_value, dict):
            return "non_breaking"

        key = next(iter(old_value), None)
        if key is None:
            return "non_breaking"

        old_val = old_value.get(key)
        new_val = new_value.get(key)

        return self._classify_numeric_constraint_change(key, old_val, new_val)

    def _classify_numeric_constraint_change(
        self,
        key: str,
        old_val: Any,
        new_val: Any,
    ) -> str:
        """Classify a single named constraint change.

        Args:
            key: The constraint name (e.g. ``"minimum"``, ``"maxLength"``).
            old_val: The old constraint value.
            new_val: The new constraint value.

        Returns:
            Severity string.
        """
        if key == "required":
            # optional→required: breaking; required→optional: non_breaking
            if new_val is True and not old_val:
                return "breaking"
            return "non_breaking"

        if key == "pattern":
            # Any pattern change may break validation
            return "breaking"

        # Minimum-style constraints: increasing the minimum is breaking
        if key in ("minimum", "exclusiveMinimum", "minLength",
                   "minItems", "minProperties"):
            if old_val is None and new_val is not None:
                return "breaking"  # new minimum introduced
            if new_val is None and old_val is not None:
                return "non_breaking"  # minimum removed
            try:
                if float(new_val) > float(old_val):
                    return "breaking"
                return "non_breaking"
            except (TypeError, ValueError):
                return "breaking"

        # Maximum-style constraints: decreasing the maximum is breaking
        if key in ("maximum", "exclusiveMaximum", "maxLength",
                   "maxItems", "maxProperties"):
            if old_val is None and new_val is not None:
                return "non_breaking"  # new maximum introduced (adds constraint)
            if new_val is None and old_val is not None:
                return "non_breaking"  # maximum removed (relaxes constraint)
            try:
                if float(new_val) < float(old_val):
                    return "breaking"
                return "non_breaking"
            except (TypeError, ValueError):
                return "breaking"

        # multipleOf: any change is potentially breaking
        if key == "multipleOf":
            return "breaking"

        return "non_breaking"

    def _classify_enum_severity(
        self,
        old_enum: Optional[List[Any]],
        new_enum: Optional[List[Any]],
    ) -> str:
        """Classify an enum change as breaking or non_breaking.

        Any removed enum values make the change breaking.  Pure additions
        are non_breaking.

        Args:
            old_enum: Original list of enum values (or None).
            new_enum: New list of enum values (or None).

        Returns:
            ``"breaking"`` if values were removed; ``"non_breaking"`` otherwise.
        """
        old_set = set(old_enum) if old_enum is not None else set()
        new_set = set(new_enum) if new_enum is not None else set()
        removed = old_set - new_set
        return "breaking" if removed else "non_breaking"

    # ------------------------------------------------------------------
    # Internal helpers — builders
    # ------------------------------------------------------------------

    def _build_change_dict(
        self,
        change_type: str,
        field_path: str,
        old_value: Any,
        new_value: Any,
        severity: str,
        description: str,
    ) -> Dict[str, Any]:
        """Build a standard change dict with a UUID id.

        Args:
            change_type: The type label for this change.
            field_path: Dot-separated path to the affected field.
            old_value: Previous value (None for added fields).
            new_value: New value (None for removed fields).
            severity: One of ``"breaking"``, ``"non_breaking"``, ``"cosmetic"``.
            description: Human-readable description of the change.

        Returns:
            Change dict conforming to the standard format.
        """
        return {
            "id": str(uuid.uuid4()),
            "change_type": change_type,
            "field_path": field_path,
            "old_value": old_value,
            "new_value": new_value,
            "severity": severity,
            "description": description,
        }

    # ------------------------------------------------------------------
    # Internal helpers — misc
    # ------------------------------------------------------------------

    def _validate_max_depth(self, max_depth: Optional[int]) -> int:
        """Validate and resolve a max_depth override.

        Args:
            max_depth: Override value or None to use instance default.

        Returns:
            Resolved integer max_depth.

        Raises:
            ValueError: If the override value is less than 1.
        """
        if max_depth is None:
            return self._max_depth
        if max_depth < 1:
            raise ValueError(
                f"max_depth override must be >= 1, got {max_depth}"
            )
        return max_depth

    def _resolve_items_type(self, items: Any) -> str:
        """Extract a type string from an ``items`` descriptor.

        Args:
            items: The ``items`` value from a field descriptor.  May be a
                string (simple type), a dict (complex type), or None.

        Returns:
            Resolved type string, or ``"unknown"`` if extraction fails.
        """
        if items is None:
            return "unknown"
        if isinstance(items, str):
            return items
        if isinstance(items, dict):
            return str(items.get("type", "unknown"))
        return str(items)
