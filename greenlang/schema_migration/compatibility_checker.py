# -*- coding: utf-8 -*-
"""
CompatibilityCheckerEngine - Schema Compatibility Analysis Engine

This module implements the CompatibilityCheckerEngine for the Schema Migration
Agent (AGENT-DATA-017, GL-DATA-X-020). It is Engine 4 of 7 in the schema
migration pipeline.

The engine follows Confluent Schema Registry compatibility semantics and checks
whether schema versions are backward compatible, forward compatible, fully
compatible, or breaking. Every check result is stored with a SHA-256 provenance
hash for complete audit trail coverage.

Zero-Hallucination Guarantees:
    - All compatibility determinations are rule-table lookups, not LLM inference.
    - Type widening/narrowing decisions follow a static, deterministic matrix.
    - No numeric values are estimated or approximated.
    - SHA-256 provenance chains every check for tamper-evident audit.

Compatibility Level Definitions (Confluent Schema Registry semantics):
    - ``full``      : New schema can read old data AND old schema can read new data.
    - ``backward``  : New schema can read data written with the old schema.
    - ``forward``   : Old schema can read data written with the new schema.
    - ``breaking``  : Neither direction is compatible; migration is unsafe.

Example:
    >>> from greenlang.schema_migration.compatibility_checker import (
    ...     CompatibilityCheckerEngine,
    ... )
    >>> engine = CompatibilityCheckerEngine()
    >>> source = {"fields": {"id": {"type": "integer", "required": True}}}
    >>> target = {
    ...     "fields": {
    ...         "id": {"type": "integer", "required": True},
    ...         "name": {"type": "string", "required": False},
    ...     }
    ... }
    >>> result = engine.check_compatibility(source, target)
    >>> assert result["compatibility_level"] == "full"
    >>> assert result["check_id"] != ""

Author: GreenLang Platform Team
Date: February 2026
PRD: AGENT-DATA-017 Schema Migration Agent (GL-DATA-X-020)
Engine: 4 of 7 — CompatibilityCheckerEngine
Status: Production Ready
"""

from __future__ import annotations

import hashlib
import json
import logging
import threading
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Set, Tuple

from greenlang.schema_migration.provenance import ProvenanceTracker

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Ordered severity ranking — used by compare_compatibility_levels
_LEVEL_RANK: Dict[str, int] = {
    "full": 0,       # most permissive (best)
    "backward": 1,
    "forward": 2,
    "breaking": 3,   # most restrictive (worst)
}

# Inverse map: rank → level name
_RANK_LEVEL: Dict[int, str] = {v: k for k, v in _LEVEL_RANK.items()}

# JSON Schema primitive types understood by the engine
_PRIMITIVE_TYPES: Set[str] = {
    "string",
    "integer",
    "number",
    "boolean",
    "null",
    "array",
    "object",
}

# ---------------------------------------------------------------------------
# Type widening / narrowing matrices
# ---------------------------------------------------------------------------

# Widening: reading old (key[0]) data with new (key[1]) type is safe
# e.g. int stored on disk → new schema reads it as float — OK
TYPE_WIDENING: Dict[Tuple[str, str], bool] = {
    ("integer", "number"): True,   # int  → float  (value range preserved)
    ("integer", "string"): True,   # int  → string (string absorbs all)
    ("number", "string"): True,    # float → string
    ("boolean", "string"): True,   # bool → string
    ("boolean", "integer"): True,  # bool → int (0/1 encoding)
}

# Narrowing: the inverse direction — reading new data with old schema loses info
TYPE_NARROWING: Dict[Tuple[str, str], bool] = {
    ("number", "integer"): True,   # float → int   (fractional part lost)
    ("string", "integer"): True,   # string → int  (arbitrary strings break)
    ("string", "number"): True,    # string → float
    ("string", "boolean"): True,   # string → bool
    ("integer", "boolean"): True,  # int → bool    (values > 1 break)
}

# ---------------------------------------------------------------------------
# Compatibility rule matrix
# ---------------------------------------------------------------------------

# Maps change_type → {"backward": bool, "forward": bool, "notes": str}
# Source of truth: Confluent Schema Registry documentation + Avro spec
_COMPATIBILITY_RULES: Dict[str, Dict[str, Any]] = {
    "add_optional_field": {
        "backward": True,
        "forward": True,
        "level": "full",
        "notes": (
            "Adding an optional field is fully compatible: new schema ignores "
            "missing field when reading old data; old schema ignores unknown "
            "field when reading new data."
        ),
    },
    "add_required_field_with_default": {
        "backward": True,
        "forward": False,
        "level": "backward",
        "notes": (
            "Adding a required field with a default value is backward compatible "
            "only: new schema fills in the default when reading old data that "
            "lacks the field, but old schema cannot read new data that now "
            "requires the field."
        ),
    },
    "add_required_field_without_default": {
        "backward": False,
        "forward": False,
        "level": "breaking",
        "notes": (
            "Adding a required field without a default is breaking: new schema "
            "cannot read old data (missing required field) and old schema "
            "cannot read new data."
        ),
    },
    "remove_optional_field": {
        "backward": True,
        "forward": True,
        "level": "full",
        "notes": (
            "Removing an optional field is fully compatible: new schema simply "
            "stops reading the field; old schema treats the absent field as "
            "optional."
        ),
    },
    "remove_required_field": {
        "backward": False,
        "forward": False,
        "level": "breaking",
        "notes": (
            "Removing a required field is breaking: new schema cannot produce "
            "data with the field, but old schema expects it."
        ),
    },
    "rename_field": {
        "backward": False,
        "forward": False,
        "level": "breaking",
        "notes": (
            "Renaming a field is breaking: old data uses the old name; new "
            "data uses the new name; neither schema can transparently read "
            "the other's data without a field alias."
        ),
    },
    "type_widening": {
        "backward": True,
        "forward": False,
        "level": "backward",
        "notes": (
            "Type widening (e.g. int → float) is backward compatible only: "
            "new schema can decode old narrower values, but old schema "
            "cannot decode new wider values."
        ),
    },
    "type_narrowing": {
        "backward": False,
        "forward": False,
        "level": "breaking",
        "notes": (
            "Type narrowing (e.g. float → int) is breaking: old float values "
            "may not fit in the new int type; data loss is possible."
        ),
    },
    "add_enum_value": {
        "backward": True,
        "forward": False,
        "level": "backward",
        "notes": (
            "Adding an enum value is backward compatible only: new schema "
            "can read old data which never used the new value, but old "
            "schema will fail on new data that carries the new enum value."
        ),
    },
    "remove_enum_value": {
        "backward": False,
        "forward": True,
        "level": "forward",
        "notes": (
            "Removing an enum value is forward compatible only: old schema "
            "can read new data (the removed value is absent), but new schema "
            "may encounter the removed value in old data."
        ),
    },
    "change_default_value": {
        "backward": True,
        "forward": True,
        "level": "full",
        "notes": (
            "Changing a default value is fully compatible: defaults are "
            "applied only when the field is absent; wire data is unchanged."
        ),
    },
    "make_required_optional": {
        "backward": True,
        "forward": False,
        "level": "backward",
        "notes": (
            "Making a required field optional is backward compatible: new "
            "schema tolerates missing field in old data, but old schema "
            "still demands the field in new data."
        ),
    },
    "make_optional_required": {
        "backward": False,
        "forward": False,
        "level": "breaking",
        "notes": (
            "Making an optional field required is breaking (unless a default "
            "is simultaneously added): old data may not contain the field; "
            "the new schema would reject it."
        ),
    },
    "make_optional_required_with_default": {
        "backward": True,
        "forward": False,
        "level": "backward",
        "notes": (
            "Making an optional field required while adding a default is "
            "backward compatible only: new schema fills in the default when "
            "the field is absent in old data."
        ),
    },
    "reorder_fields": {
        "backward": True,
        "forward": True,
        "level": "full",
        "notes": (
            "Reordering fields is fully compatible for named-field schemas "
            "(JSON Schema, Avro): field lookup is by name, not position."
        ),
    },
    "change_description": {
        "backward": True,
        "forward": True,
        "level": "full",
        "notes": (
            "Description/annotation changes are documentation-only and have "
            "no impact on data compatibility."
        ),
    },
    "change_type_incompatible": {
        "backward": False,
        "forward": False,
        "level": "breaking",
        "notes": (
            "Changing a field to an incompatible type (e.g. string → object) "
            "is breaking: neither direction can safely decode the other's data."
        ),
    },
}

# ---------------------------------------------------------------------------
# Remediation template library
# ---------------------------------------------------------------------------

_REMEDIATION_TEMPLATES: Dict[str, str] = {
    "add_required_field_without_default": (
        "Add a default value to the new field so that old data lacking the "
        "field can still be decoded. This promotes the change from 'breaking' "
        "to 'backward' compatible."
    ),
    "remove_required_field": (
        "Deprecate the field first (keep it as optional with a note) across "
        "at least one schema version before removing it. This gives consumers "
        "time to adapt."
    ),
    "rename_field": (
        "Add the new field name as an alias alongside the original field in "
        "an intermediate schema version. Once all producers and consumers "
        "have migrated, remove the old field name."
    ),
    "type_narrowing": (
        "Avoid narrowing type changes. If you must narrow, introduce a new "
        "field with the narrower type and keep the old field optional with a "
        "deprecation marker."
    ),
    "remove_enum_value": (
        "Keep the removed enum value in an intermediate schema version marked "
        "as deprecated. Remove it only after confirming no stored data "
        "contains that value."
    ),
    "make_optional_required": (
        "Simultaneously add a default value to the field to make the change "
        "backward compatible, or keep the field optional and enforce "
        "presence at the application layer."
    ),
    "change_type_incompatible": (
        "Introduce a new field with the target type. Migrate data to populate "
        "the new field. Only then remove the old field through the normal "
        "deprecation path."
    ),
    "generic_breaking": (
        "Review the change against the compatibility rule matrix. Introduce "
        "an intermediate schema version that bridges the old and new "
        "structures to avoid a hard cutover."
    ),
}


# ---------------------------------------------------------------------------
# CompatibilityCheckerEngine
# ---------------------------------------------------------------------------


class CompatibilityCheckerEngine:
    """Engine 4 of 7: Schema Compatibility Checker for AGENT-DATA-017.

    Checks compatibility between two schema definitions following Confluent
    Schema Registry conventions. Implements the full rule matrix for all
    change types and produces structured check results with per-field issue
    details and remediation suggestions.

    Supported compatibility levels (Confluent semantics):
        - ``full``      — New schema can read old data AND old schema can
                          read new data.
        - ``backward``  — New schema can read data produced with the old schema.
        - ``forward``   — Old schema can read data produced with the new schema.
        - ``breaking``  — Neither direction is compatible; migration is unsafe.

    Thread Safety:
        All public methods acquire ``self._lock`` before mutating shared state.
        Reads of stored check results are also lock-protected. The engine is
        safe for concurrent use across multiple threads.

    Zero-Hallucination:
        All compatibility decisions are pure rule-table lookups backed by the
        ``_COMPATIBILITY_RULES`` constant dict and the ``TYPE_WIDENING`` /
        ``TYPE_NARROWING`` matrices.  No LLM calls are made for any numeric
        or compatibility determination.

    Attributes:
        _checks: Dict mapping check_id → stored check result dict.
        _lock: Thread lock protecting ``_checks`` mutations.
        _provenance: ProvenanceTracker instance for SHA-256 audit chains.
        _stats: Running statistics dict updated after every check.

    Example:
        >>> engine = CompatibilityCheckerEngine()
        >>> source = {"fields": {"amount": {"type": "integer", "required": True}}}
        >>> target = {"fields": {"amount": {"type": "number", "required": True}}}
        >>> result = engine.check_compatibility(source, target)
        >>> result["compatibility_level"]
        'backward'
        >>> result["issues"]
        []
    """

    def __init__(self, provenance: Optional[ProvenanceTracker] = None) -> None:
        """Initialize CompatibilityCheckerEngine.

        Args:
            provenance: Optional :class:`ProvenanceTracker` instance to use for
                SHA-256 audit chains.  If ``None``, a new tracker is created.

        Example:
            >>> engine = CompatibilityCheckerEngine()
            >>> engine.get_statistics()["total_checks"]
            0
        """
        self._checks: Dict[str, Dict[str, Any]] = {}
        self._lock: threading.Lock = threading.Lock()
        self._provenance: ProvenanceTracker = (
            provenance if provenance is not None else ProvenanceTracker()
        )
        self._stats: Dict[str, Any] = {
            "total_checks": 0,
            "full_compatible": 0,
            "backward_compatible": 0,
            "forward_compatible": 0,
            "breaking": 0,
            "total_issues_found": 0,
        }
        logger.info(
            "CompatibilityCheckerEngine initialized; provenance_id=%s",
            id(self._provenance),
        )

    # ------------------------------------------------------------------
    # 1. check_compatibility — main entry point
    # ------------------------------------------------------------------

    def check_compatibility(
        self,
        source_definition: Dict[str, Any],
        target_definition: Dict[str, Any],
        changes: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """Check compatibility between source and target schema definitions.

        This is the main entry point.  It orchestrates the backward and
        forward checks, determines the overall compatibility level, generates
        per-field issues, produces remediation suggestions, computes a
        provenance hash, and persists the result for later retrieval.

        Args:
            source_definition: The source (old) schema definition dict.  Must
                contain at least a ``"fields"`` key mapping field names to
                field descriptor dicts.
            target_definition: The target (new) schema definition dict.  Same
                structure as ``source_definition``.
            changes: Optional pre-computed list of change dicts
                (``{"change_type": ..., "field_path": ..., ...}``). When
                provided, the engine skips auto-diff and uses these changes
                directly, which is useful when the caller has already
                performed change detection via Engine 3.

        Returns:
            Dict with the following keys:

            - ``check_id`` (str): UUID4 identifier for this check result.
            - ``compatibility_level`` (str): ``"full"`` | ``"backward"`` |
              ``"forward"`` | ``"breaking"``.
            - ``backward_compatible`` (bool): Whether new schema reads old data.
            - ``forward_compatible`` (bool): Whether old schema reads new data.
            - ``issues`` (List[Dict]): Per-field issue descriptors.
            - ``recommendations`` (List[Dict]): Remediation suggestions.
            - ``provenance_hash`` (str): SHA-256 hash for audit trail.
            - ``checked_at`` (str): ISO-8601 UTC timestamp.
            - ``source_field_count`` (int): Number of fields in source.
            - ``target_field_count`` (int): Number of fields in target.
            - ``change_count`` (int): Number of detected changes.

        Raises:
            TypeError: If ``source_definition`` or ``target_definition`` are
                not dicts.
            ValueError: If either definition is missing the ``"fields"`` key.

        Example:
            >>> engine = CompatibilityCheckerEngine()
            >>> src = {"fields": {"id": {"type": "integer", "required": True}}}
            >>> tgt = {"fields": {
            ...     "id": {"type": "integer", "required": True},
            ...     "tag": {"type": "string", "required": False},
            ... }}
            >>> r = engine.check_compatibility(src, tgt)
            >>> r["compatibility_level"]
            'full'
        """
        self._validate_definition(source_definition, "source_definition")
        self._validate_definition(target_definition, "target_definition")

        if changes is None:
            changes = self._auto_diff(source_definition, target_definition)

        backward_result = self.check_backward_compatibility(
            source_definition, target_definition, changes
        )
        forward_result = self.check_forward_compatibility(
            source_definition, target_definition, changes
        )

        level = self.determine_compatibility_level(backward_result, forward_result)
        all_issues = backward_result["issues"] + forward_result["issues"]
        # Deduplicate issues with same field_path + issue_type
        all_issues = self._deduplicate_issues(all_issues)
        recommendations = self.generate_remediation(all_issues)

        check_id = str(uuid.uuid4())
        checked_at = datetime.now(timezone.utc).isoformat()
        provenance_hash = self._compute_check_hash(
            check_id, source_definition, target_definition, level, all_issues
        )

        result: Dict[str, Any] = {
            "check_id": check_id,
            "compatibility_level": level,
            "backward_compatible": backward_result["compatible"],
            "forward_compatible": forward_result["compatible"],
            "issues": all_issues,
            "recommendations": recommendations,
            "provenance_hash": provenance_hash,
            "checked_at": checked_at,
            "source_field_count": len(
                source_definition.get("fields", {})
            ),
            "target_field_count": len(
                target_definition.get("fields", {})
            ),
            "change_count": len(changes),
        }

        with self._lock:
            self._checks[check_id] = result
            self._update_stats(level, all_issues)

        self._provenance.record(
            entity_type="compatibility_check",
            entity_id=check_id,
            action="check_completed",
            data={
                "level": level,
                "issue_count": len(all_issues),
                "provenance_hash": provenance_hash,
            },
        )

        logger.info(
            "Compatibility check completed: check_id=%s level=%s "
            "issues=%d backward=%s forward=%s",
            check_id,
            level,
            len(all_issues),
            backward_result["compatible"],
            forward_result["compatible"],
        )
        return result

    # ------------------------------------------------------------------
    # 2. check_backward_compatibility
    # ------------------------------------------------------------------

    def check_backward_compatibility(
        self,
        source_definition: Dict[str, Any],
        target_definition: Dict[str, Any],
        changes: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """Check whether the new schema can read data written with the old schema.

        Backward compatibility means that a consumer using the *new* (target)
        schema can successfully decode data produced by a *old* (source)
        schema.  This fails when the new schema introduces required fields
        without defaults, removes fields that are required by old data, or
        narrows types.

        Args:
            source_definition: The old schema definition dict.
            target_definition: The new schema definition dict.
            changes: Optional pre-computed change list.  If ``None``, an
                automatic diff is performed.

        Returns:
            Dict with:

            - ``compatible`` (bool): ``True`` if backward compatible.
            - ``issues`` (List[Dict]): Issues that break backward compatibility.
            - ``checked_changes`` (int): Number of changes evaluated.

        Example:
            >>> engine = CompatibilityCheckerEngine()
            >>> src = {"fields": {"x": {"type": "integer", "required": True}}}
            >>> tgt = {"fields": {"x": {"type": "number", "required": True}}}
            >>> r = engine.check_backward_compatibility(src, tgt)
            >>> r["compatible"]
            True
        """
        if changes is None:
            changes = self._auto_diff(source_definition, target_definition)

        issues: List[Dict[str, Any]] = []
        for change in changes:
            assessment = self.assess_field_change(change)
            if not assessment["backward_compatible"]:
                issues.append(
                    self._build_issue(
                        field_path=change.get("field_path", "unknown"),
                        issue_type="backward_incompatible",
                        change_type=change.get("change_type", "unknown"),
                        severity="breaking",
                        description=assessment["description"],
                        remediation=assessment["remediation"],
                    )
                )

        compatible = len(issues) == 0
        logger.debug(
            "Backward compatibility: compatible=%s issues=%d changes_evaluated=%d",
            compatible,
            len(issues),
            len(changes),
        )
        return {
            "compatible": compatible,
            "issues": issues,
            "checked_changes": len(changes),
        }

    # ------------------------------------------------------------------
    # 3. check_forward_compatibility
    # ------------------------------------------------------------------

    def check_forward_compatibility(
        self,
        source_definition: Dict[str, Any],
        target_definition: Dict[str, Any],
        changes: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """Check whether the old schema can read data written with the new schema.

        Forward compatibility means that a consumer using the *old* (source)
        schema can successfully decode data produced by a *new* (target) schema.
        This fails when the new schema adds enum values the old schema does not
        know about, widens types beyond what the old schema can parse, or
        adds required fields the old schema would reject as unknown.

        Args:
            source_definition: The old schema definition dict.
            target_definition: The new schema definition dict.
            changes: Optional pre-computed change list.  If ``None``, an
                automatic diff is performed.

        Returns:
            Dict with:

            - ``compatible`` (bool): ``True`` if forward compatible.
            - ``issues`` (List[Dict]): Issues that break forward compatibility.
            - ``checked_changes`` (int): Number of changes evaluated.

        Example:
            >>> engine = CompatibilityCheckerEngine()
            >>> src = {"fields": {"status": {
            ...     "type": "string", "required": True,
            ...     "enum": ["active", "inactive"],
            ... }}}
            >>> tgt = {"fields": {"status": {
            ...     "type": "string", "required": True,
            ...     "enum": ["active", "inactive", "pending"],
            ... }}}
            >>> r = engine.check_forward_compatibility(src, tgt)
            >>> r["compatible"]
            False
        """
        if changes is None:
            changes = self._auto_diff(source_definition, target_definition)

        issues: List[Dict[str, Any]] = []
        for change in changes:
            assessment = self.assess_field_change(change)
            if not assessment["forward_compatible"]:
                issues.append(
                    self._build_issue(
                        field_path=change.get("field_path", "unknown"),
                        issue_type="forward_incompatible",
                        change_type=change.get("change_type", "unknown"),
                        severity="breaking",
                        description=assessment["description"],
                        remediation=assessment["remediation"],
                    )
                )

        compatible = len(issues) == 0
        logger.debug(
            "Forward compatibility: compatible=%s issues=%d changes_evaluated=%d",
            compatible,
            len(issues),
            len(changes),
        )
        return {
            "compatible": compatible,
            "issues": issues,
            "checked_changes": len(changes),
        }

    # ------------------------------------------------------------------
    # 4. check_full_compatibility
    # ------------------------------------------------------------------

    def check_full_compatibility(
        self,
        source_definition: Dict[str, Any],
        target_definition: Dict[str, Any],
        changes: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """Check whether the schema transition is fully compatible in both directions.

        Full compatibility requires that:
        - The new schema can read old data (backward compatible), AND
        - The old schema can read new data (forward compatible).

        This is the strictest level that still allows the schema to evolve.

        Args:
            source_definition: The old schema definition dict.
            target_definition: The new schema definition dict.
            changes: Optional pre-computed change list.  If ``None``, an
                automatic diff is performed.

        Returns:
            Dict with:

            - ``compatible`` (bool): ``True`` only if both directions pass.
            - ``backward_result`` (Dict): Full backward check result.
            - ``forward_result`` (Dict): Full forward check result.
            - ``issues`` (List[Dict]): Combined, deduplicated issue list.
            - ``level`` (str): Resulting compatibility level string.

        Example:
            >>> engine = CompatibilityCheckerEngine()
            >>> src = {"fields": {"x": {"type": "integer", "required": False}}}
            >>> tgt = {"fields": {"x": {"type": "integer", "required": False}}}
            >>> r = engine.check_full_compatibility(src, tgt)
            >>> r["compatible"]
            True
        """
        if changes is None:
            changes = self._auto_diff(source_definition, target_definition)

        backward_result = self.check_backward_compatibility(
            source_definition, target_definition, changes
        )
        forward_result = self.check_forward_compatibility(
            source_definition, target_definition, changes
        )

        level = self.determine_compatibility_level(backward_result, forward_result)
        all_issues = self._deduplicate_issues(
            backward_result["issues"] + forward_result["issues"]
        )
        compatible = level == "full"

        logger.debug(
            "Full compatibility check: compatible=%s level=%s",
            compatible,
            level,
        )
        return {
            "compatible": compatible,
            "backward_result": backward_result,
            "forward_result": forward_result,
            "issues": all_issues,
            "level": level,
        }

    # ------------------------------------------------------------------
    # 5. determine_compatibility_level
    # ------------------------------------------------------------------

    def determine_compatibility_level(
        self,
        backward_result: Dict[str, Any],
        forward_result: Dict[str, Any],
    ) -> str:
        """Determine the overall compatibility level from backward and forward results.

        The level is derived from the truth table:

        +----------+---------+------------+
        | backward | forward | level      |
        +==========+=========+============+
        | True     | True    | full       |
        +----------+---------+------------+
        | True     | False   | backward   |
        +----------+---------+------------+
        | False    | True    | forward    |
        +----------+---------+------------+
        | False    | False   | breaking   |
        +----------+---------+------------+

        Args:
            backward_result: Dict returned by :meth:`check_backward_compatibility`
                containing at minimum a ``"compatible"`` boolean key.
            forward_result: Dict returned by :meth:`check_forward_compatibility`
                containing at minimum a ``"compatible"`` boolean key.

        Returns:
            One of ``"full"``, ``"backward"``, ``"forward"``, or ``"breaking"``.

        Example:
            >>> engine = CompatibilityCheckerEngine()
            >>> level = engine.determine_compatibility_level(
            ...     {"compatible": True}, {"compatible": False}
            ... )
            >>> level
            'backward'
        """
        backward_ok: bool = bool(backward_result.get("compatible", False))
        forward_ok: bool = bool(forward_result.get("compatible", False))

        if backward_ok and forward_ok:
            return "full"
        if backward_ok and not forward_ok:
            return "backward"
        if not backward_ok and forward_ok:
            return "forward"
        return "breaking"

    # ------------------------------------------------------------------
    # 6. get_compatibility_rules
    # ------------------------------------------------------------------

    def get_compatibility_rules(self) -> Dict[str, Any]:
        """Return the full compatibility rule matrix used by this engine.

        The returned dict maps change_type strings to rule descriptors
        containing ``backward`` (bool), ``forward`` (bool), ``level`` (str),
        and ``notes`` (str) keys.  The matrix is a deep copy of the module-level
        constant to prevent external mutation.

        Returns:
            Dict mapping change_type → rule descriptor dict.

        Example:
            >>> engine = CompatibilityCheckerEngine()
            >>> rules = engine.get_compatibility_rules()
            >>> rules["add_optional_field"]["level"]
            'full'
            >>> rules["rename_field"]["backward"]
            False
        """
        import copy

        return copy.deepcopy(_COMPATIBILITY_RULES)

    # ------------------------------------------------------------------
    # 7. assess_field_change
    # ------------------------------------------------------------------

    def assess_field_change(self, change: Dict[str, Any]) -> Dict[str, Any]:
        """Assess a single schema change for backward and forward impact.

        Looks up the change_type in the compatibility rule matrix.  For
        ``"retyped"`` changes, delegates to :meth:`is_type_widening` and
        :meth:`is_type_narrowing` to pick the appropriate sub-rule.

        Args:
            change: Dict describing a single schema change.  Expected keys:

                - ``"change_type"`` (str): One of the keys in
                  ``_COMPATIBILITY_RULES``, or ``"retyped"`` for type changes.
                - ``"field_path"`` (str): Dot-notation path to the changed field.
                - ``"old_type"`` (str, optional): The old field type (for retyped).
                - ``"new_type"`` (str, optional): The new field type (for retyped).
                - ``"required"`` (bool, optional): Whether the field is required.
                - ``"has_default"`` (bool, optional): Whether a default is present.
                - Additional keys are passed through to the assessment result.

        Returns:
            Dict with:

            - ``"backward_compatible"`` (bool): Whether the change is backward OK.
            - ``"forward_compatible"`` (bool): Whether the change is forward OK.
            - ``"level"`` (str): Resulting compatibility level for this change.
            - ``"description"`` (str): Human-readable explanation.
            - ``"remediation"`` (str): Suggested fix if breaking.
            - ``"change_type"`` (str): Resolved change type used for lookup.

        Example:
            >>> engine = CompatibilityCheckerEngine()
            >>> change = {
            ...     "change_type": "add_optional_field",
            ...     "field_path": "metadata.tag",
            ... }
            >>> r = engine.assess_field_change(change)
            >>> r["backward_compatible"]
            True
            >>> r["level"]
            'full'
        """
        change_type: str = change.get("change_type", "")
        field_path: str = change.get("field_path", "unknown")

        # Handle retyped changes by resolving to widening / narrowing / incompatible
        if change_type == "retyped":
            change_type = self._resolve_retype_change(change)

        # Handle make_optional_required with default present
        if change_type == "make_optional_required":
            if change.get("has_default", False):
                change_type = "make_optional_required_with_default"

        rule = _COMPATIBILITY_RULES.get(change_type)
        if rule is None:
            # Unknown change type — treat as breaking and log a warning
            logger.warning(
                "Unknown change_type=%r for field_path=%s; treating as breaking",
                change_type,
                field_path,
            )
            return {
                "backward_compatible": False,
                "forward_compatible": False,
                "level": "breaking",
                "description": (
                    f"Unknown change type '{change_type}' on field '{field_path}'. "
                    "Treated as breaking by default."
                ),
                "remediation": _REMEDIATION_TEMPLATES["generic_breaking"],
                "change_type": change_type,
            }

        level = self.determine_compatibility_level(
            {"compatible": rule["backward"]},
            {"compatible": rule["forward"]},
        )
        remediation = _REMEDIATION_TEMPLATES.get(
            change_type, _REMEDIATION_TEMPLATES["generic_breaking"]
        )

        return {
            "backward_compatible": rule["backward"],
            "forward_compatible": rule["forward"],
            "level": level,
            "description": rule["notes"],
            "remediation": remediation if not (rule["backward"] and rule["forward"]) else "",
            "change_type": change_type,
        }

    # ------------------------------------------------------------------
    # 8. is_type_widening
    # ------------------------------------------------------------------

    def is_type_widening(self, old_type: str, new_type: str) -> bool:
        """Check whether a type change is a safe widening.

        A widening is safe when the new type can represent all values that
        the old type could represent, so that no data is lost when the new
        schema reads old data.

        Args:
            old_type: The original field type string (e.g. ``"integer"``).
            new_type: The new field type string (e.g. ``"number"``).

        Returns:
            ``True`` if the transition ``(old_type, new_type)`` is recorded
            in the ``TYPE_WIDENING`` matrix; ``False`` otherwise.

        Example:
            >>> engine = CompatibilityCheckerEngine()
            >>> engine.is_type_widening("integer", "number")
            True
            >>> engine.is_type_widening("number", "integer")
            False
        """
        return TYPE_WIDENING.get((old_type, new_type), False)

    # ------------------------------------------------------------------
    # 9. is_type_narrowing
    # ------------------------------------------------------------------

    def is_type_narrowing(self, old_type: str, new_type: str) -> bool:
        """Check whether a type change is a lossy narrowing.

        A narrowing is lossy when the new type cannot represent all values
        that the old type could represent, so that old data decoded with the
        new schema may lose precision or fail entirely.

        Args:
            old_type: The original field type string (e.g. ``"number"``).
            new_type: The new field type string (e.g. ``"integer"``).

        Returns:
            ``True`` if the transition ``(old_type, new_type)`` is recorded
            in the ``TYPE_NARROWING`` matrix; ``False`` otherwise.

        Example:
            >>> engine = CompatibilityCheckerEngine()
            >>> engine.is_type_narrowing("number", "integer")
            True
            >>> engine.is_type_narrowing("integer", "number")
            False
        """
        return TYPE_NARROWING.get((old_type, new_type), False)

    # ------------------------------------------------------------------
    # 10. generate_remediation
    # ------------------------------------------------------------------

    def generate_remediation(
        self, issues: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Generate structured remediation suggestions for a list of issues.

        For each issue that has a non-empty ``"change_type"``, this method
        looks up the corresponding remediation template and returns a list
        of actionable suggestion dicts.  Compatible changes (empty
        ``"remediation"`` field) are skipped.

        Args:
            issues: List of issue dicts as produced by
                :meth:`check_backward_compatibility` or
                :meth:`check_forward_compatibility`.

        Returns:
            List of remediation suggestion dicts, each containing:

            - ``"field_path"`` (str): Field this suggestion applies to.
            - ``"change_type"`` (str): The change that triggered the issue.
            - ``"issue_type"`` (str): ``"backward_incompatible"`` or
              ``"forward_incompatible"``.
            - ``"suggestion"`` (str): Human-readable remediation text.
            - ``"priority"`` (str): ``"high"`` for breaking changes, else
              ``"medium"``.

        Example:
            >>> engine = CompatibilityCheckerEngine()
            >>> issues = [{
            ...     "field_path": "user.age",
            ...     "change_type": "rename_field",
            ...     "issue_type": "backward_incompatible",
            ...     "severity": "breaking",
            ...     "description": "...",
            ...     "remediation": "...",
            ... }]
            >>> rems = engine.generate_remediation(issues)
            >>> rems[0]["change_type"]
            'rename_field'
        """
        remediation_list: List[Dict[str, Any]] = []
        seen: Set[str] = set()  # deduplicate by (field_path, change_type)

        for issue in issues:
            change_type: str = issue.get("change_type", "")
            field_path: str = issue.get("field_path", "unknown")
            dedup_key = f"{field_path}::{change_type}"

            if dedup_key in seen:
                continue
            seen.add(dedup_key)

            # Skip non-breaking issues (no remediation needed)
            if issue.get("severity", "") != "breaking":
                continue

            suggestion = _REMEDIATION_TEMPLATES.get(
                change_type, _REMEDIATION_TEMPLATES["generic_breaking"]
            )

            remediation_list.append(
                {
                    "field_path": field_path,
                    "change_type": change_type,
                    "issue_type": issue.get("issue_type", ""),
                    "suggestion": suggestion,
                    "priority": "high" if issue.get("severity") == "breaking" else "medium",
                }
            )

        logger.debug(
            "Generated %d remediation suggestions from %d issues",
            len(remediation_list),
            len(issues),
        )
        return remediation_list

    # ------------------------------------------------------------------
    # 11. get_check
    # ------------------------------------------------------------------

    def get_check(self, check_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve a stored check result by its check_id.

        Args:
            check_id: UUID string returned by :meth:`check_compatibility`.

        Returns:
            The stored check result dict, or ``None`` if not found.

        Example:
            >>> engine = CompatibilityCheckerEngine()
            >>> src = {"fields": {}}
            >>> tgt = {"fields": {}}
            >>> r = engine.check_compatibility(src, tgt)
            >>> stored = engine.get_check(r["check_id"])
            >>> stored is not None
            True
        """
        with self._lock:
            result = self._checks.get(check_id)

        if result is None:
            logger.debug("get_check: check_id=%s not found", check_id)
        return result

    # ------------------------------------------------------------------
    # 12. list_checks
    # ------------------------------------------------------------------

    def list_checks(
        self, limit: int = 100, offset: int = 0
    ) -> List[Dict[str, Any]]:
        """List stored check results with pagination.

        Results are returned in insertion order (oldest first).

        Args:
            limit: Maximum number of results to return.  Must be >= 1.
            offset: Number of results to skip from the beginning.  Must be >= 0.

        Returns:
            List of check result dicts, sorted by ``checked_at`` ascending.
            Returns an empty list if ``offset`` exceeds the total count.

        Raises:
            ValueError: If ``limit`` is less than 1 or ``offset`` is negative.

        Example:
            >>> engine = CompatibilityCheckerEngine()
            >>> src = {"fields": {}}
            >>> tgt = {"fields": {}}
            >>> engine.check_compatibility(src, tgt)  # doctest: +ELLIPSIS
            {...}
            >>> checks = engine.list_checks(limit=10, offset=0)
            >>> len(checks) >= 1
            True
        """
        if limit < 1:
            raise ValueError(f"limit must be >= 1, got {limit}")
        if offset < 0:
            raise ValueError(f"offset must be >= 0, got {offset}")

        with self._lock:
            all_checks = sorted(
                self._checks.values(),
                key=lambda c: c.get("checked_at", ""),
            )

        page = all_checks[offset: offset + limit]
        logger.debug(
            "list_checks: total=%d offset=%d limit=%d returned=%d",
            len(all_checks),
            offset,
            limit,
            len(page),
        )
        return page

    # ------------------------------------------------------------------
    # 13. compare_compatibility_levels
    # ------------------------------------------------------------------

    def compare_compatibility_levels(self, level_a: str, level_b: str) -> str:
        """Return the stricter of two compatibility levels.

        Levels are ranked as follows (strictest first):
        ``breaking`` > ``forward`` > ``backward`` > ``full``.

        Args:
            level_a: First compatibility level string.
            level_b: Second compatibility level string.

        Returns:
            The stricter level string.  If both are equal, returns that level.

        Raises:
            ValueError: If either level string is not recognised.

        Example:
            >>> engine = CompatibilityCheckerEngine()
            >>> engine.compare_compatibility_levels("full", "backward")
            'backward'
            >>> engine.compare_compatibility_levels("breaking", "forward")
            'breaking'
        """
        self._validate_level(level_a, "level_a")
        self._validate_level(level_b, "level_b")

        rank_a = _LEVEL_RANK[level_a]
        rank_b = _LEVEL_RANK[level_b]

        # Higher rank = stricter
        stricter = level_a if rank_a >= rank_b else level_b
        logger.debug(
            "compare_compatibility_levels: %s vs %s → %s",
            level_a,
            level_b,
            stricter,
        )
        return stricter

    # ------------------------------------------------------------------
    # 14. validate_transition
    # ------------------------------------------------------------------

    def validate_transition(
        self, current_level: str, required_level: str
    ) -> Dict[str, Any]:
        """Check whether a schema can transition at the required compatibility level.

        A transition is allowed when the schema's actual compatibility level
        is at least as permissive as the required level.  For example, a
        ``"full"`` compatible schema can satisfy a ``"backward"`` requirement,
        but a ``"backward"`` compatible schema cannot satisfy a ``"full"``
        requirement.

        Args:
            current_level: The actual compatibility level of the proposed
                schema transition.
            required_level: The minimum required level enforced by policy.

        Returns:
            Dict with:

            - ``"allowed"`` (bool): ``True`` if the transition satisfies the policy.
            - ``"current_level"`` (str): Passed-through current level.
            - ``"required_level"`` (str): Passed-through required level.
            - ``"reason"`` (str): Human-readable explanation.

        Raises:
            ValueError: If either level string is not recognised.

        Example:
            >>> engine = CompatibilityCheckerEngine()
            >>> r = engine.validate_transition("full", "backward")
            >>> r["allowed"]
            True
            >>> r = engine.validate_transition("backward", "full")
            >>> r["allowed"]
            False
        """
        self._validate_level(current_level, "current_level")
        self._validate_level(required_level, "required_level")

        current_rank = _LEVEL_RANK[current_level]
        required_rank = _LEVEL_RANK[required_level]

        # Lower rank = more permissive = better; current must be <= required rank
        allowed = current_rank <= required_rank

        if allowed:
            reason = (
                f"Schema transition at level '{current_level}' satisfies "
                f"the required '{required_level}' policy."
            )
        else:
            reason = (
                f"Schema transition at level '{current_level}' does NOT "
                f"satisfy the required '{required_level}' policy. "
                f"The transition is too restrictive."
            )

        logger.debug(
            "validate_transition: current=%s required=%s allowed=%s",
            current_level,
            required_level,
            allowed,
        )
        return {
            "allowed": allowed,
            "current_level": current_level,
            "required_level": required_level,
            "reason": reason,
        }

    # ------------------------------------------------------------------
    # 15. get_statistics
    # ------------------------------------------------------------------

    def get_statistics(self) -> Dict[str, Any]:
        """Return running statistics for all checks performed by this engine.

        Returns:
            Dict with:

            - ``"total_checks"`` (int): Total number of checks run.
            - ``"full_compatible"`` (int): Count of full-compatible results.
            - ``"backward_compatible"`` (int): Count of backward-only results.
            - ``"forward_compatible"`` (int): Count of forward-only results.
            - ``"breaking"`` (int): Count of breaking results.
            - ``"total_issues_found"`` (int): Cumulative count of issues found.
            - ``"stored_checks"`` (int): Number of checks in the in-memory store.
            - ``"provenance_entries"`` (int): Number of provenance records.

        Example:
            >>> engine = CompatibilityCheckerEngine()
            >>> stats = engine.get_statistics()
            >>> stats["total_checks"]
            0
        """
        with self._lock:
            stats = dict(self._stats)
            stats["stored_checks"] = len(self._checks)

        stats["provenance_entries"] = self._provenance.entry_count
        logger.debug("get_statistics: %s", stats)
        return stats

    # ------------------------------------------------------------------
    # 16. reset
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Clear all stored check results and reset statistics.

        Provenance entries are NOT cleared (the audit trail is immutable).
        Intended for test teardown to prevent state leakage between test cases.

        Example:
            >>> engine = CompatibilityCheckerEngine()
            >>> src = {"fields": {}}
            >>> engine.check_compatibility(src, src)  # doctest: +ELLIPSIS
            {...}
            >>> engine.reset()
            >>> engine.get_statistics()["total_checks"]
            0
        """
        with self._lock:
            self._checks.clear()
            self._stats = {
                "total_checks": 0,
                "full_compatible": 0,
                "backward_compatible": 0,
                "forward_compatible": 0,
                "breaking": 0,
                "total_issues_found": 0,
            }
        logger.info("CompatibilityCheckerEngine state reset (provenance preserved)")

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _validate_definition(
        self, definition: Any, name: str
    ) -> None:
        """Validate that a schema definition is a dict with a 'fields' key.

        Args:
            definition: The value to validate.
            name: Parameter name used in error messages.

        Raises:
            TypeError: If ``definition`` is not a dict.
            ValueError: If ``definition`` is missing the ``"fields"`` key.
        """
        if not isinstance(definition, dict):
            raise TypeError(
                f"{name} must be a dict, got {type(definition).__name__}"
            )
        if "fields" not in definition:
            raise ValueError(
                f"{name} must contain a 'fields' key; "
                f"got keys: {list(definition.keys())}"
            )

    def _validate_level(self, level: str, param_name: str) -> None:
        """Validate that a level string is one of the known compatibility levels.

        Args:
            level: Level string to validate.
            param_name: Parameter name for error messages.

        Raises:
            ValueError: If ``level`` is not in ``_LEVEL_RANK``.
        """
        if level not in _LEVEL_RANK:
            raise ValueError(
                f"{param_name} must be one of {sorted(_LEVEL_RANK.keys())}, "
                f"got '{level}'"
            )

    def _auto_diff(
        self,
        source: Dict[str, Any],
        target: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """Automatically diff two schema definitions and produce a change list.

        Compares the ``"fields"`` maps of source and target, generating one
        change dict per differing field.  This is a best-effort structural diff;
        callers that have already run Engine 3 (Change Detector) should pass
        pre-computed changes instead.

        Detected change types:
            - ``add_optional_field`` / ``add_required_field_with_default``
              / ``add_required_field_without_default`` — field in target only.
            - ``remove_optional_field`` / ``remove_required_field`` — field
              in source only.
            - ``retyped`` — field present in both but with different ``type``.
            - ``make_required_optional`` / ``make_optional_required`` — field
              present in both but with changed ``required`` flag.
            - ``change_default_value`` — field present in both, type matches,
              but ``default`` differs.
            - ``add_enum_value`` / ``remove_enum_value`` — enum sets changed.

        Args:
            source: Old schema definition with ``"fields"`` dict.
            target: New schema definition with ``"fields"`` dict.

        Returns:
            List of change dicts, each with at minimum ``"change_type"`` and
            ``"field_path"`` keys.
        """
        source_fields: Dict[str, Any] = source.get("fields", {})
        target_fields: Dict[str, Any] = target.get("fields", {})

        changes: List[Dict[str, Any]] = []

        source_keys = set(source_fields.keys())
        target_keys = set(target_fields.keys())

        # Fields added in target
        for field_name in target_keys - source_keys:
            changes.append(
                self._classify_added_field(field_name, target_fields[field_name])
            )

        # Fields removed in target
        for field_name in source_keys - target_keys:
            changes.append(
                self._classify_removed_field(field_name, source_fields[field_name])
            )

        # Fields present in both — check for mutations
        for field_name in source_keys & target_keys:
            field_changes = self._diff_field(
                field_name,
                source_fields[field_name],
                target_fields[field_name],
            )
            changes.extend(field_changes)

        logger.debug(
            "_auto_diff: source_fields=%d target_fields=%d changes=%d",
            len(source_fields),
            len(target_fields),
            len(changes),
        )
        return changes

    def _classify_added_field(
        self, field_name: str, field_def: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Classify an added field as optional, required-with-default, or required-without-default.

        Args:
            field_name: The name of the newly added field.
            field_def: The field descriptor dict from the target schema.

        Returns:
            Change dict with ``"change_type"``, ``"field_path"``, ``"required"``,
            and ``"has_default"`` keys.
        """
        required: bool = bool(field_def.get("required", False))
        has_default: bool = "default" in field_def

        if not required:
            change_type = "add_optional_field"
        elif has_default:
            change_type = "add_required_field_with_default"
        else:
            change_type = "add_required_field_without_default"

        return {
            "change_type": change_type,
            "field_path": field_name,
            "required": required,
            "has_default": has_default,
        }

    def _classify_removed_field(
        self, field_name: str, field_def: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Classify a removed field as optional or required removal.

        Args:
            field_name: The name of the removed field.
            field_def: The field descriptor dict from the source schema.

        Returns:
            Change dict with ``"change_type"`` and ``"field_path"`` keys.
        """
        required: bool = bool(field_def.get("required", False))
        change_type = "remove_required_field" if required else "remove_optional_field"
        return {
            "change_type": change_type,
            "field_path": field_name,
            "required": required,
        }

    def _diff_field(
        self,
        field_name: str,
        source_def: Dict[str, Any],
        target_def: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """Diff a field that exists in both source and target schemas.

        Checks for type changes, required flag changes, default value changes,
        and enum membership changes.

        Args:
            field_name: Field name / path identifier.
            source_def: Source field descriptor dict.
            target_def: Target field descriptor dict.

        Returns:
            List of change dicts (may be empty if the field is unchanged, or
            may contain multiple if several attributes changed).
        """
        changes: List[Dict[str, Any]] = []
        old_type: str = source_def.get("type", "")
        new_type: str = target_def.get("type", "")
        old_required: bool = bool(source_def.get("required", False))
        new_required: bool = bool(target_def.get("required", False))

        # Type change
        if old_type and new_type and old_type != new_type:
            changes.append(
                {
                    "change_type": "retyped",
                    "field_path": field_name,
                    "old_type": old_type,
                    "new_type": new_type,
                    "required": new_required,
                }
            )

        # Required flag change (only if type did not change — avoid double-counting)
        elif old_required != new_required:
            if old_required and not new_required:
                change_type = "make_required_optional"
            else:
                has_default = "default" in target_def
                change_type = (
                    "make_optional_required_with_default"
                    if has_default
                    else "make_optional_required"
                )
            changes.append(
                {
                    "change_type": change_type,
                    "field_path": field_name,
                    "required": new_required,
                    "has_default": "default" in target_def,
                }
            )

        # Default value change
        source_default = source_def.get("default", _SENTINEL)
        target_default = target_def.get("default", _SENTINEL)
        if source_default is not _SENTINEL or target_default is not _SENTINEL:
            if source_default != target_default:
                changes.append(
                    {
                        "change_type": "change_default_value",
                        "field_path": field_name,
                        "old_default": source_default,
                        "new_default": target_default,
                    }
                )

        # Enum changes
        old_enum: Optional[List[Any]] = source_def.get("enum")
        new_enum: Optional[List[Any]] = target_def.get("enum")
        if old_enum is not None or new_enum is not None:
            enum_changes = self._diff_enum(field_name, old_enum, new_enum)
            changes.extend(enum_changes)

        return changes

    def _diff_enum(
        self,
        field_name: str,
        old_enum: Optional[List[Any]],
        new_enum: Optional[List[Any]],
    ) -> List[Dict[str, Any]]:
        """Detect added and removed enum values.

        Args:
            field_name: Field name for change dict population.
            old_enum: Enum value list from source schema (may be None).
            new_enum: Enum value list from target schema (may be None).

        Returns:
            List of change dicts with ``"add_enum_value"`` or
            ``"remove_enum_value"`` change types.
        """
        changes: List[Dict[str, Any]] = []
        old_set: Set[Any] = set(old_enum) if old_enum else set()
        new_set: Set[Any] = set(new_enum) if new_enum else set()

        added_values = new_set - old_set
        removed_values = old_set - new_set

        if added_values:
            changes.append(
                {
                    "change_type": "add_enum_value",
                    "field_path": field_name,
                    "added_values": sorted(str(v) for v in added_values),
                }
            )
        if removed_values:
            changes.append(
                {
                    "change_type": "remove_enum_value",
                    "field_path": field_name,
                    "removed_values": sorted(str(v) for v in removed_values),
                }
            )
        return changes

    def _resolve_retype_change(self, change: Dict[str, Any]) -> str:
        """Resolve a 'retyped' change to a specific change_type string.

        Uses the TYPE_WIDENING and TYPE_NARROWING matrices.  Falls back to
        ``"change_type_incompatible"`` when neither direction applies.

        Args:
            change: Change dict with ``"old_type"`` and ``"new_type"`` keys.

        Returns:
            One of ``"type_widening"``, ``"type_narrowing"``, or
            ``"change_type_incompatible"``.
        """
        old_type: str = change.get("old_type", "")
        new_type: str = change.get("new_type", "")

        if self.is_type_widening(old_type, new_type):
            return "type_widening"
        if self.is_type_narrowing(old_type, new_type):
            return "type_narrowing"
        return "change_type_incompatible"

    def _build_issue(
        self,
        field_path: str,
        issue_type: str,
        change_type: str,
        severity: str,
        description: str,
        remediation: str,
    ) -> Dict[str, Any]:
        """Construct a structured issue dict.

        Args:
            field_path: Dot-notation field path.
            issue_type: ``"backward_incompatible"`` or ``"forward_incompatible"``.
            change_type: The change type that triggered the issue.
            severity: ``"breaking"`` or ``"non_breaking"``.
            description: Human-readable explanation of the issue.
            remediation: Suggested fix for the issue.

        Returns:
            Fully populated issue dict matching the Issue Dict Format spec.
        """
        return {
            "field_path": field_path,
            "issue_type": issue_type,
            "change_type": change_type,
            "severity": severity,
            "description": description,
            "remediation": remediation,
        }

    def _deduplicate_issues(
        self, issues: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Remove duplicate issues with the same field_path and issue_type.

        When backward and forward checks both produce an issue for the same
        field (e.g. a breaking rename), the combined list would contain two
        entries.  This method collapses them into one.

        Args:
            issues: Combined list of issues from backward and forward checks.

        Returns:
            Deduplicated list preserving the first occurrence of each unique
            ``(field_path, issue_type)`` combination.
        """
        seen: Set[Tuple[str, str]] = set()
        deduped: List[Dict[str, Any]] = []
        for issue in issues:
            key: Tuple[str, str] = (
                issue.get("field_path", ""),
                issue.get("issue_type", ""),
            )
            if key not in seen:
                seen.add(key)
                deduped.append(issue)
        return deduped

    def _compute_check_hash(
        self,
        check_id: str,
        source_definition: Dict[str, Any],
        target_definition: Dict[str, Any],
        level: str,
        issues: List[Dict[str, Any]],
    ) -> str:
        """Compute a SHA-256 provenance hash for a completed check.

        The hash covers the check_id, both schema definitions (JSON-serialized
        with sorted keys), the resulting level, and the full issues list.
        This ensures any tampering with the stored result is detectable.

        Args:
            check_id: UUID for this check.
            source_definition: The source schema dict.
            target_definition: The target schema dict.
            level: Resulting compatibility level string.
            issues: List of issue dicts for this check.

        Returns:
            Hex-encoded SHA-256 hash string.
        """
        payload = {
            "check_id": check_id,
            "source_definition": source_definition,
            "target_definition": target_definition,
            "level": level,
            "issues": issues,
        }
        serialized = json.dumps(payload, sort_keys=True, default=str)
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()

    def _update_stats(
        self, level: str, issues: List[Dict[str, Any]]
    ) -> None:
        """Update running statistics after a check completes.

        Must be called while ``self._lock`` is held.

        Args:
            level: Compatibility level determined for the check.
            issues: List of issues found in the check.
        """
        self._stats["total_checks"] += 1
        self._stats["total_issues_found"] += len(issues)

        level_key_map: Dict[str, str] = {
            "full": "full_compatible",
            "backward": "backward_compatible",
            "forward": "forward_compatible",
            "breaking": "breaking",
        }
        stat_key = level_key_map.get(level)
        if stat_key:
            self._stats[stat_key] += 1


# ---------------------------------------------------------------------------
# Module-level sentinel for missing default detection
# ---------------------------------------------------------------------------

class _SentinelType:
    """Singleton sentinel used to distinguish 'field absent' from 'field = None'."""

    _instance: Optional["_SentinelType"] = None

    def __new__(cls) -> "_SentinelType":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __repr__(self) -> str:
        return "<SENTINEL>"


_SENTINEL = _SentinelType()


# ---------------------------------------------------------------------------
# Public surface
# ---------------------------------------------------------------------------

__all__ = [
    # Main engine class
    "CompatibilityCheckerEngine",
    # Type matrices (exposed for external use / testing)
    "TYPE_WIDENING",
    "TYPE_NARROWING",
    # Rule constant (read-only reference)
    "_COMPATIBILITY_RULES",
]
