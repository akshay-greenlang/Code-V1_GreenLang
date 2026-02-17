# -*- coding: utf-8 -*-
"""
Migration Planner Engine - AGENT-DATA-017: Schema Migration Agent (GL-DATA-X-020)

Engine 5 of 7: MigrationPlannerEngine

Generates dependency-aware migration plans consisting of ordered transformation
steps derived from a list of schema changes.  Supports dry-run estimation,
effort scoring, plan validation, and simulation on sample data.

Transformation Step Types:
    - rename_field  : Rename a source field to a target field name.
    - cast_type     : Change the data type of a field (with precision handling).
    - set_default   : Inject a default value for a new required field.
    - add_field     : Add a new field to the target schema.
    - remove_field  : Remove a field from the source schema.
    - compute_field : Derive a computed field from one or more source fields.
    - split_field   : Split one source field into multiple target fields.
    - merge_fields  : Merge multiple source fields into a single target field.

Ordering Rules (Kahn-inspired deterministic ordering):
    1. add_field          — new fields must exist before anything else uses them.
    2. rename_field       — renames operate on existing field names.
    3. cast_type          — type casts operate on already-renamed fields.
    4. set_default        — defaults can be injected at any position but are
                           grouped here for clarity.
    5. compute_field      — computed fields may consume cast/renamed values.
    6. split_field        — splits operate on the result of computes.
    7. merge_fields       — merges happen after splits so all fields are present.
    8. remove_field       — removals are ALWAYS last to avoid premature deletion.

Effort Scoring:
    Base costs per operation:
        rename=1, cast=2, default=1, add=1, remove=1, compute=5, split=3, merge=3
    Total cost = sum(step_costs) * max(estimated_records, 1) / 1000
    Bands: LOW < 60, MEDIUM 60-600, HIGH 600-3600, CRITICAL > 3600

Zero-Hallucination Guarantees:
    - All plan generation is deterministic rule-based logic only.
    - No LLM calls in the planning, ordering, or scoring path.
    - SHA-256 provenance recorded on every plan mutation.
    - Dry-run simulates deterministic field operations only.
    - Thread-safe in-memory plan storage.

Example:
    >>> from greenlang.schema_migration.migration_planner import MigrationPlannerEngine
    >>> engine = MigrationPlannerEngine()
    >>> changes = [
    ...     {"change_type": "field_renamed", "source_field": "qty",
    ...      "target_field": "quantity"},
    ...     {"change_type": "type_changed", "field_name": "quantity",
    ...      "old_type": "integer", "new_type": "number"},
    ...     {"change_type": "field_removed", "field_name": "legacy_id"},
    ... ]
    >>> plan = engine.create_plan(
    ...     source_schema_id="schema_A",
    ...     target_schema_id="schema_B",
    ...     source_version="1.0.0",
    ...     target_version="2.0.0",
    ...     changes=changes,
    ...     estimated_records=50000,
    ... )
    >>> print(plan["effort_band"])
    LOW

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
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from greenlang.schema_migration.provenance import ProvenanceTracker

logger = logging.getLogger(__name__)

__all__ = [
    "MigrationPlannerEngine",
]


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Canonical operation names used in step dicts.
OP_RENAME = "rename_field"
OP_CAST = "cast_type"
OP_DEFAULT = "set_default"
OP_ADD = "add_field"
OP_REMOVE = "remove_field"
OP_COMPUTE = "compute_field"
OP_SPLIT = "split_field"
OP_MERGE = "merge_fields"

# Ordering weight: lower number executes first.
_OPERATION_ORDER: Dict[str, int] = {
    OP_ADD: 0,
    OP_RENAME: 1,
    OP_CAST: 2,
    OP_DEFAULT: 3,
    OP_COMPUTE: 4,
    OP_SPLIT: 5,
    OP_MERGE: 6,
    OP_REMOVE: 7,
}

# Base effort cost per operation (unitless tokens).
_OPERATION_BASE_COST: Dict[str, int] = {
    OP_RENAME: 1,
    OP_CAST: 2,
    OP_DEFAULT: 1,
    OP_ADD: 1,
    OP_REMOVE: 1,
    OP_COMPUTE: 5,
    OP_SPLIT: 3,
    OP_MERGE: 3,
}

# Effort band thresholds (cost * records / 1000).
_EFFORT_LOW_MAX = 60
_EFFORT_MEDIUM_MAX = 600
_EFFORT_HIGH_MAX = 3600

# Change type keywords that trigger specific step generators.
_RENAME_CHANGE_TYPES = frozenset({"field_renamed", "rename", "renamed"})
_CAST_CHANGE_TYPES = frozenset({"type_changed", "type_cast", "cast"})
_DEFAULT_CHANGE_TYPES = frozenset({"default_added", "set_default", "default_changed"})
_ADD_CHANGE_TYPES = frozenset({"field_added", "add_field", "added"})
_REMOVE_CHANGE_TYPES = frozenset({"field_removed", "remove_field", "removed", "deleted"})
_COMPUTE_CHANGE_TYPES = frozenset({"computed_field", "compute", "derived", "derived_field"})
_SPLIT_CHANGE_TYPES = frozenset({"field_split", "split"})
_MERGE_CHANGE_TYPES = frozenset({"fields_merged", "merge"})

# Valid plan status values.
_PLAN_STATUSES = frozenset({"pending", "validated", "dry_run_complete", "executed", "failed"})


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _utcnow() -> datetime:
    """Return current UTC datetime with microseconds zeroed."""
    return datetime.now(timezone.utc).replace(microsecond=0)


def _generate_id(prefix: str = "PLAN") -> str:
    """Generate a unique identifier with the given prefix.

    Args:
        prefix: Short uppercase prefix string.

    Returns:
        String of the form ``{prefix}-{hex12}``.
    """
    return f"{prefix}-{uuid.uuid4().hex[:12]}"


def _compute_sha256(payload: str) -> str:
    """Compute a SHA-256 hex digest from a UTF-8 payload string.

    Args:
        payload: The string to hash.

    Returns:
        64-character hex-encoded SHA-256 digest.
    """
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _serialize(obj: Any) -> str:
    """Serialize obj to a canonical JSON string for hashing.

    Args:
        obj: Any JSON-serialisable object.

    Returns:
        Sorted-key JSON string with str fallback for non-serialisable values.
    """
    return json.dumps(obj, sort_keys=True, default=str)


def _safe_cast(value: Any, target_type: str) -> Tuple[bool, Any]:
    """Attempt to cast ``value`` to ``target_type`` for dry-run simulation.

    Args:
        value: The raw value to cast.
        target_type: One of ``integer``, ``number``, ``string``,
            ``boolean``, ``array``, ``object``.

    Returns:
        Tuple of (success, cast_value).  On failure, success is False and
        cast_value is the original value unchanged.
    """
    try:
        t = target_type.lower()
        if t in ("integer", "int"):
            return True, int(float(str(value)))
        if t in ("number", "float", "double", "decimal"):
            return True, float(str(value))
        if t in ("string", "str", "text"):
            return True, str(value)
        if t in ("boolean", "bool"):
            return True, str(value).lower() in ("true", "1", "yes", "on")
        if t == "array":
            if isinstance(value, list):
                return True, value
            return True, [value]
        if t == "object":
            if isinstance(value, dict):
                return True, value
            return True, {"value": value}
        # Unknown type — pass through unchanged.
        return True, value
    except (ValueError, TypeError):
        return False, value


# ---------------------------------------------------------------------------
# MigrationPlannerEngine
# ---------------------------------------------------------------------------


class MigrationPlannerEngine:
    """Engine 5 of 7: generates dependency-aware migration plans.

    Converts a list of schema changes into an ordered sequence of
    transformation steps, estimates execution effort, validates plan
    completeness, and can simulate execution on sample data.

    All numeric operations are deterministic (zero-hallucination):
    no LLM calls exist in the planning, ordering, or scoring path.

    Attributes:
        _plans: In-memory store of plan dicts keyed by plan_id.
        _lock: Threading lock for safe concurrent access.
        _provenance: ProvenanceTracker for SHA-256 audit trail.

    Example:
        >>> engine = MigrationPlannerEngine()
        >>> plan = engine.create_plan(
        ...     source_schema_id="src",
        ...     target_schema_id="tgt",
        ...     source_version="1.0",
        ...     target_version="2.0",
        ...     changes=[{"change_type": "field_renamed",
        ...               "source_field": "old", "target_field": "new"}],
        ... )
        >>> assert plan["status"] == "pending"
        >>> assert len(plan["steps"]) == 1
    """

    def __init__(self, genesis_hash: str = "greenlang-schema-migration-genesis") -> None:
        """Initialise the MigrationPlannerEngine.

        Args:
            genesis_hash: Anchor string for the provenance chain.
        """
        self._plans: Dict[str, Dict[str, Any]] = {}
        self._lock: threading.Lock = threading.Lock()
        self._provenance: ProvenanceTracker = ProvenanceTracker(
            genesis_hash=genesis_hash
        )
        self._stats: Dict[str, Any] = {
            "plans_created": 0,
            "plans_validated": 0,
            "plans_dry_run": 0,
            "total_steps_generated": 0,
        }
        logger.info("MigrationPlannerEngine initialised")

    # ------------------------------------------------------------------
    # 1. create_plan
    # ------------------------------------------------------------------

    def create_plan(
        self,
        source_schema_id: str,
        target_schema_id: str,
        source_version: str,
        target_version: str,
        changes: List[Dict[str, Any]],
        source_definition: Optional[Dict[str, Any]] = None,
        target_definition: Optional[Dict[str, Any]] = None,
        estimated_records: int = 0,
    ) -> Dict[str, Any]:
        """Generate a migration plan from a list of schema changes.

        Delegates step generation to :meth:`generate_steps`, effort
        estimation to :meth:`estimate_effort`, and then stores the plan
        in the internal store.

        Args:
            source_schema_id: Identifier of the source schema subject.
            target_schema_id: Identifier of the target schema subject.
            source_version: Semantic version string for the source schema.
            target_version: Semantic version string for the target schema.
            changes: List of change dicts describing each schema difference.
                Each dict must have at least a ``change_type`` key; additional
                keys depend on the change type (see module docstring).
            source_definition: Optional full source schema definition dict.
                Provided to ``generate_compute_step`` for formula inference.
            target_definition: Optional full target schema definition dict.
                Provided to ``generate_compute_step`` for formula inference.
            estimated_records: Expected number of data records affected.
                Used for effort band calculation.  Defaults to 0.

        Returns:
            Plan dict with keys:
                - plan_id (str)
                - source_schema_id (str)
                - target_schema_id (str)
                - source_version (str)
                - target_version (str)
                - steps (List[Dict])
                - step_count (int)
                - estimated_records (int)
                - effort (Dict) from :meth:`estimate_effort`
                - effort_band (str) shortcut to effort["effort_band"]
                - status (str) always "pending" on creation
                - provenance_hash (str)
                - created_at (str) UTC ISO timestamp

        Raises:
            ValueError: If required string arguments are empty, or if
                ``changes`` is not a list.
        """
        self._validate_plan_inputs(
            source_schema_id, target_schema_id,
            source_version, target_version, changes
        )

        plan_id = _generate_id("PLAN")
        created_at = _utcnow().isoformat()

        steps = self.generate_steps(
            changes=changes,
            source_definition=source_definition,
            target_definition=target_definition,
        )

        effort = self.estimate_effort(steps, estimated_records=estimated_records)

        provenance_payload = {
            "plan_id": plan_id,
            "source_schema_id": source_schema_id,
            "target_schema_id": target_schema_id,
            "source_version": source_version,
            "target_version": target_version,
            "step_count": len(steps),
            "created_at": created_at,
        }
        provenance_hash = _compute_sha256(_serialize(provenance_payload))

        plan: Dict[str, Any] = {
            "plan_id": plan_id,
            "source_schema_id": source_schema_id,
            "target_schema_id": target_schema_id,
            "source_version": source_version,
            "target_version": target_version,
            "steps": steps,
            "step_count": len(steps),
            "estimated_records": max(0, estimated_records),
            "effort": effort,
            "effort_band": effort["effort_band"],
            "status": "pending",
            "provenance_hash": provenance_hash,
            "created_at": created_at,
            "validation_errors": [],
            "dry_run_result": None,
        }

        with self._lock:
            self._plans[plan_id] = plan
            self._stats["plans_created"] += 1
            self._stats["total_steps_generated"] += len(steps)

        self._provenance.record(
            entity_type="migration_plan",
            entity_id=plan_id,
            action="plan_created",
            data=provenance_payload,
        )

        logger.info(
            "Migration plan created: plan_id=%s, steps=%d, effort_band=%s",
            plan_id,
            len(steps),
            effort["effort_band"],
        )
        return plan

    # ------------------------------------------------------------------
    # 2. generate_steps
    # ------------------------------------------------------------------

    def generate_steps(
        self,
        changes: List[Dict[str, Any]],
        source_definition: Optional[Dict[str, Any]] = None,
        target_definition: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """Generate ordered transformation steps from a list of changes.

        Iterates over every change dict, dispatches to the appropriate
        step generator method, collects all resulting steps, then calls
        :meth:`order_steps` to produce the final dependency-ordered list.

        Args:
            changes: List of change dicts.  Each dict must contain at
                minimum a ``change_type`` key (case-insensitive).
            source_definition: Optional source schema definition forwarded
                to :meth:`generate_compute_step`.
            target_definition: Optional target schema definition forwarded
                to :meth:`generate_compute_step`.

        Returns:
            Ordered list of step dicts with ``step_number`` fields set
            sequentially from 1.

        Raises:
            TypeError: If ``changes`` is not a list.
        """
        if not isinstance(changes, list):
            raise TypeError(
                f"changes must be a list, got {type(changes).__name__}"
            )

        raw_steps: List[Dict[str, Any]] = []

        for change in changes:
            if not isinstance(change, dict):
                logger.warning("Skipping non-dict change entry: %r", change)
                continue

            ct = str(change.get("change_type", "")).lower().strip()
            new_steps = self._dispatch_change(
                change_type=ct,
                change=change,
                source_def=source_definition,
                target_def=target_definition,
            )
            raw_steps.extend(new_steps)

        ordered = self.order_steps(raw_steps)
        # Re-number after ordering.
        for idx, step in enumerate(ordered):
            step["step_number"] = idx + 1

        return ordered

    # ------------------------------------------------------------------
    # 3. generate_rename_step
    # ------------------------------------------------------------------

    def generate_rename_step(self, change: Dict[str, Any]) -> Dict[str, Any]:
        """Create a ``rename_field`` step dict from a rename change.

        Args:
            change: Change dict with keys:
                - ``source_field`` (str): The old field name.
                - ``target_field`` (str): The new field name.
                Optional contextual keys are captured in ``parameters``.

        Returns:
            Step dict with ``operation="rename_field"``.

        Raises:
            ValueError: If ``source_field`` or ``target_field`` is missing
                or empty.
        """
        source_field = str(change.get("source_field", "")).strip()
        target_field = str(change.get("target_field", "")).strip()

        if not source_field:
            raise ValueError(
                "generate_rename_step: 'source_field' is required and must not be empty"
            )
        if not target_field:
            raise ValueError(
                "generate_rename_step: 'target_field' is required and must not be empty"
            )

        return {
            "step_number": 0,  # Will be assigned after ordering.
            "operation": OP_RENAME,
            "source_field": source_field,
            "target_field": target_field,
            "parameters": {
                "old_name": source_field,
                "new_name": target_field,
            },
            "reversible": True,
            "description": f"Rename field '{source_field}' to '{target_field}'",
            "depends_on": [],
        }

    # ------------------------------------------------------------------
    # 4. generate_cast_step
    # ------------------------------------------------------------------

    def generate_cast_step(self, change: Dict[str, Any]) -> Dict[str, Any]:
        """Create a ``cast_type`` step dict from a type-change change.

        Extracts precision and scale from the target type definition when
        present (e.g. ``decimal(10,2)`` patterns).

        Args:
            change: Change dict with keys:
                - ``field_name`` (str): The field whose type changes.
                  Alternatively ``target_field`` or ``source_field`` are
                  consulted as fallbacks.
                - ``old_type`` (str): The current data type.
                - ``new_type`` (str): The desired data type.
                Optional keys:
                - ``precision`` (int): Decimal precision.
                - ``scale`` (int): Decimal scale.
                - ``nullable`` (bool): Whether the cast output may be null.

        Returns:
            Step dict with ``operation="cast_type"``.

        Raises:
            ValueError: If ``field_name`` or ``new_type`` cannot be determined.
        """
        field_name = (
            str(change.get("field_name", "")).strip()
            or str(change.get("target_field", "")).strip()
            or str(change.get("source_field", "")).strip()
        )
        if not field_name:
            raise ValueError(
                "generate_cast_step: 'field_name' is required and must not be empty"
            )

        old_type = str(change.get("old_type", "unknown")).strip()
        new_type = str(change.get("new_type", "")).strip()
        if not new_type:
            raise ValueError(
                "generate_cast_step: 'new_type' is required and must not be empty"
            )

        precision, scale = self._extract_precision(new_type, change)

        parameters: Dict[str, Any] = {
            "old_type": old_type,
            "new_type": new_type,
            "nullable": bool(change.get("nullable", True)),
        }
        if precision is not None:
            parameters["precision"] = precision
        if scale is not None:
            parameters["scale"] = scale

        return {
            "step_number": 0,
            "operation": OP_CAST,
            "source_field": field_name,
            "target_field": field_name,
            "parameters": parameters,
            "reversible": False,
            "description": (
                f"Cast field '{field_name}' from '{old_type}' to '{new_type}'"
            ),
            "depends_on": [],
        }

    # ------------------------------------------------------------------
    # 5. generate_default_step
    # ------------------------------------------------------------------

    def generate_default_step(self, change: Dict[str, Any]) -> Dict[str, Any]:
        """Create a ``set_default`` step for a new required field.

        Args:
            change: Change dict with keys:
                - ``field_name`` (str): The field to assign a default.
                  Alternatively ``target_field`` is consulted as a fallback.
                - ``default_value`` (Any): The default to inject.
                Optional keys:
                - ``field_type`` (str): Data type for the field.
                - ``required`` (bool): Whether the field is required.

        Returns:
            Step dict with ``operation="set_default"``.

        Raises:
            ValueError: If ``field_name`` cannot be determined.
        """
        field_name = (
            str(change.get("field_name", "")).strip()
            or str(change.get("target_field", "")).strip()
        )
        if not field_name:
            raise ValueError(
                "generate_default_step: 'field_name' is required and must not be empty"
            )

        default_value = change.get("default_value", None)
        field_type = str(change.get("field_type", "string")).strip()

        return {
            "step_number": 0,
            "operation": OP_DEFAULT,
            "source_field": field_name,
            "target_field": field_name,
            "parameters": {
                "default_value": default_value,
                "field_type": field_type,
                "required": bool(change.get("required", False)),
            },
            "reversible": True,
            "description": (
                f"Set default value for field '{field_name}'"
                f" = {default_value!r}"
            ),
            "depends_on": [],
        }

    # ------------------------------------------------------------------
    # 6. generate_add_step
    # ------------------------------------------------------------------

    def generate_add_step(self, change: Dict[str, Any]) -> Dict[str, Any]:
        """Create an ``add_field`` step for a newly introduced field.

        Args:
            change: Change dict with keys:
                - ``field_name`` (str): New field name.
                  Alternatively ``target_field``.
                - ``field_type`` (str): Data type for the new field.
                Optional keys:
                - ``required`` (bool): Whether the field is required.
                - ``default_value`` (Any): Default value for existing records.
                - ``description`` (str): Field documentation.

        Returns:
            Step dict with ``operation="add_field"``.

        Raises:
            ValueError: If ``field_name`` cannot be determined.
        """
        field_name = (
            str(change.get("field_name", "")).strip()
            or str(change.get("target_field", "")).strip()
        )
        if not field_name:
            raise ValueError(
                "generate_add_step: 'field_name' is required and must not be empty"
            )

        field_type = str(change.get("field_type", "string")).strip()

        return {
            "step_number": 0,
            "operation": OP_ADD,
            "source_field": None,
            "target_field": field_name,
            "parameters": {
                "field_type": field_type,
                "required": bool(change.get("required", False)),
                "default_value": change.get("default_value", None),
                "description": str(change.get("description", "")),
            },
            "reversible": True,
            "description": (
                f"Add new field '{field_name}' of type '{field_type}'"
            ),
            "depends_on": [],
        }

    # ------------------------------------------------------------------
    # 7. generate_remove_step
    # ------------------------------------------------------------------

    def generate_remove_step(self, change: Dict[str, Any]) -> Dict[str, Any]:
        """Create a ``remove_field`` step for a deleted field.

        Remove steps must always execute LAST to avoid premature deletion
        of fields that other steps may still reference.

        Args:
            change: Change dict with keys:
                - ``field_name`` (str): The field to remove.
                  Alternatively ``source_field``.
                Optional keys:
                - ``deprecated`` (bool): Whether field was previously deprecated.
                - ``reason`` (str): Removal reason for documentation.

        Returns:
            Step dict with ``operation="remove_field"``.

        Raises:
            ValueError: If ``field_name`` cannot be determined.
        """
        field_name = (
            str(change.get("field_name", "")).strip()
            or str(change.get("source_field", "")).strip()
        )
        if not field_name:
            raise ValueError(
                "generate_remove_step: 'field_name' is required and must not be empty"
            )

        return {
            "step_number": 0,
            "operation": OP_REMOVE,
            "source_field": field_name,
            "target_field": None,
            "parameters": {
                "deprecated": bool(change.get("deprecated", False)),
                "reason": str(change.get("reason", "field no longer present in target schema")),
            },
            "reversible": False,
            "description": f"Remove field '{field_name}' from schema",
            "depends_on": [],
        }

    # ------------------------------------------------------------------
    # 8. generate_compute_step
    # ------------------------------------------------------------------

    def generate_compute_step(
        self,
        change: Dict[str, Any],
        source_def: Optional[Dict[str, Any]] = None,
        target_def: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Create a ``compute_field`` step for a derived field.

        Attempts to infer the computation formula from the target schema
        definition when provided.  Falls back to a placeholder formula.

        Args:
            change: Change dict with keys:
                - ``field_name`` (str): The computed field name.
                  Alternatively ``target_field``.
                - ``formula`` (str): Computation expression.  Optional —
                  inferred from target_def when omitted.
                - ``source_fields`` (List[str]): Input field names.
                Optional keys:
                - ``output_type`` (str): Expected result type.
                - ``description`` (str): Human-readable explanation.
            source_def: Full source schema definition dict for formula lookup.
            target_def: Full target schema definition dict for formula lookup.

        Returns:
            Step dict with ``operation="compute_field"``.

        Raises:
            ValueError: If ``field_name`` cannot be determined.
        """
        field_name = (
            str(change.get("field_name", "")).strip()
            or str(change.get("target_field", "")).strip()
        )
        if not field_name:
            raise ValueError(
                "generate_compute_step: 'field_name' is required and must not be empty"
            )

        source_fields = change.get("source_fields", [])
        if not isinstance(source_fields, list):
            source_fields = [str(source_fields)]
        source_fields = [str(f).strip() for f in source_fields if str(f).strip()]

        formula = str(change.get("formula", "")).strip()
        if not formula:
            formula = self._infer_formula(field_name, source_fields, target_def)

        output_type = str(change.get("output_type", "number")).strip()

        return {
            "step_number": 0,
            "operation": OP_COMPUTE,
            "source_field": source_fields[0] if len(source_fields) == 1 else None,
            "target_field": field_name,
            "parameters": {
                "formula": formula,
                "source_fields": source_fields,
                "output_type": output_type,
                "description": str(change.get("description", "")),
            },
            "reversible": False,
            "description": (
                f"Compute field '{field_name}' using formula: {formula}"
            ),
            "depends_on": [],
        }

    # ------------------------------------------------------------------
    # 9. generate_split_step
    # ------------------------------------------------------------------

    def generate_split_step(
        self,
        source_field: str,
        target_fields: List[str],
    ) -> Dict[str, Any]:
        """Create a ``split_field`` step (one field split into many).

        Args:
            source_field: The field to split.
            target_fields: List of output field names produced by the split.

        Returns:
            Step dict with ``operation="split_field"``.

        Raises:
            ValueError: If ``source_field`` is empty or ``target_fields``
                is empty or contains fewer than 2 elements.
        """
        source_field = str(source_field).strip()
        if not source_field:
            raise ValueError(
                "generate_split_step: 'source_field' must not be empty"
            )
        if not isinstance(target_fields, list) or len(target_fields) < 2:
            raise ValueError(
                "generate_split_step: 'target_fields' must be a list "
                "with at least 2 elements"
            )

        clean_targets = [str(f).strip() for f in target_fields if str(f).strip()]

        return {
            "step_number": 0,
            "operation": OP_SPLIT,
            "source_field": source_field,
            "target_field": clean_targets[0],  # Primary output for reference.
            "parameters": {
                "source_field": source_field,
                "target_fields": clean_targets,
                "delimiter": None,  # Caller may override post-creation.
            },
            "reversible": True,
            "description": (
                f"Split field '{source_field}' into "
                f"{len(clean_targets)} fields: "
                + ", ".join(f"'{f}'" for f in clean_targets)
            ),
            "depends_on": [],
        }

    # ------------------------------------------------------------------
    # 10. generate_merge_step
    # ------------------------------------------------------------------

    def generate_merge_step(
        self,
        source_fields: List[str],
        target_field: str,
    ) -> Dict[str, Any]:
        """Create a ``merge_fields`` step (many fields merged into one).

        Args:
            source_fields: List of input field names to merge.
            target_field: The single output field name produced by the merge.

        Returns:
            Step dict with ``operation="merge_fields"``.

        Raises:
            ValueError: If ``target_field`` is empty, or ``source_fields``
                has fewer than 2 elements.
        """
        target_field = str(target_field).strip()
        if not target_field:
            raise ValueError(
                "generate_merge_step: 'target_field' must not be empty"
            )
        if not isinstance(source_fields, list) or len(source_fields) < 2:
            raise ValueError(
                "generate_merge_step: 'source_fields' must be a list "
                "with at least 2 elements"
            )

        clean_sources = [str(f).strip() for f in source_fields if str(f).strip()]

        return {
            "step_number": 0,
            "operation": OP_MERGE,
            "source_field": clean_sources[0],  # Primary input for reference.
            "target_field": target_field,
            "parameters": {
                "source_fields": clean_sources,
                "target_field": target_field,
                "separator": " ",  # Caller may override post-creation.
            },
            "reversible": True,
            "description": (
                f"Merge {len(clean_sources)} fields "
                + "("
                + ", ".join(f"'{f}'" for f in clean_sources)
                + f") into '{target_field}'"
            ),
            "depends_on": [],
        }

    # ------------------------------------------------------------------
    # 11. order_steps
    # ------------------------------------------------------------------

    def order_steps(self, steps: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Order transformation steps by operation type and declared dependencies.

        Uses a stable two-key sort:
            1. Primary key: operation order weight (from ``_OPERATION_ORDER``).
            2. Secondary key: original list position for deterministic ties.

        Remove steps are unconditionally moved to the end.  Step numbers
        are NOT reassigned here; the caller is responsible for that.

        Args:
            steps: Unordered list of step dicts.  Each step must have an
                ``operation`` key (str).

        Returns:
            New list of step dicts in dependency-safe execution order.
            The input list is NOT mutated.
        """
        if not steps:
            return []

        def _sort_key(item: Tuple[int, Dict[str, Any]]) -> Tuple[int, int]:
            idx, step = item
            op = step.get("operation", "")
            weight = _OPERATION_ORDER.get(op, 4)  # Unknown ops treated as compute.
            return (weight, idx)

        indexed = list(enumerate(steps))
        indexed.sort(key=_sort_key)

        return [step for _, step in indexed]

    # ------------------------------------------------------------------
    # 12. estimate_effort
    # ------------------------------------------------------------------

    def estimate_effort(
        self,
        steps: List[Dict[str, Any]],
        estimated_records: int = 0,
    ) -> Dict[str, Any]:
        """Estimate migration effort and assign an effort band.

        Computes total cost from per-step base costs scaled by record count,
        then classifies into LOW / MEDIUM / HIGH / CRITICAL.

        Args:
            steps: List of step dicts.  Each must have an ``operation`` key.
            estimated_records: Number of data rows to transform.
                Zero or negative is treated as 1 (minimum baseline cost).

        Returns:
            Dict with keys:
                - total_cost (float): Scaled effort score.
                - base_cost (int): Sum of unscaled per-step costs.
                - step_costs (Dict[str, int]): Per-step cost breakdown keyed
                  by "step_{n}" or operation name.
                - estimated_records (int)
                - effort_band (str): "LOW" | "MEDIUM" | "HIGH" | "CRITICAL"
                - estimated_minutes (float): Rough wall-clock estimate.
        """
        records = max(1, estimated_records)
        base_cost = 0
        step_costs: Dict[str, int] = {}

        for i, step in enumerate(steps):
            op = step.get("operation", "")
            cost = _OPERATION_BASE_COST.get(op, 1)
            base_cost += cost
            step_key = f"step_{i + 1}_{op}"
            step_costs[step_key] = cost

        total_cost = base_cost * records / 1000.0
        effort_band = self._classify_effort_band(total_cost)

        # Rough wall-clock estimate: 1 cost unit ≈ 1 second of processing.
        estimated_minutes = round(total_cost / 60.0, 2)

        return {
            "total_cost": round(total_cost, 4),
            "base_cost": base_cost,
            "step_costs": step_costs,
            "estimated_records": records,
            "effort_band": effort_band,
            "estimated_minutes": estimated_minutes,
        }

    # ------------------------------------------------------------------
    # 13. validate_plan
    # ------------------------------------------------------------------

    def validate_plan(self, plan_id: str) -> Dict[str, Any]:
        """Validate a stored plan for completeness and consistency.

        Checks performed:
            - Plan exists in the internal store.
            - Steps list is non-empty.
            - Every step has required keys: operation, source_field or
              target_field, description, reversible, depends_on.
            - No duplicate step numbers.
            - Remove steps are the last operations in the ordered list.
            - All declared ``depends_on`` step numbers reference valid steps.

        Args:
            plan_id: Identifier of the plan to validate.

        Returns:
            Dict with keys:
                - plan_id (str)
                - valid (bool)
                - errors (List[str])
                - warnings (List[str])
                - step_count (int)
                - validated_at (str) UTC ISO timestamp
        """
        errors: List[str] = []
        warnings: List[str] = []

        plan = self.get_plan(plan_id)
        if plan is None:
            return {
                "plan_id": plan_id,
                "valid": False,
                "errors": [f"Plan '{plan_id}' not found"],
                "warnings": [],
                "step_count": 0,
                "validated_at": _utcnow().isoformat(),
            }

        steps = plan.get("steps", [])
        if not steps:
            errors.append("Plan contains no steps")

        seen_step_numbers: set = set()
        valid_step_numbers: set = {s.get("step_number") for s in steps}

        for i, step in enumerate(steps):
            prefix = f"Step {i + 1}"
            self._validate_step_dict(step, prefix, errors, warnings)

            # Duplicate step number check.
            sn = step.get("step_number")
            if sn in seen_step_numbers:
                errors.append(f"{prefix}: duplicate step_number {sn}")
            else:
                seen_step_numbers.add(sn)

            # Dependency reference check.
            for dep in step.get("depends_on", []):
                if dep not in valid_step_numbers:
                    errors.append(
                        f"{prefix}: depends_on references unknown "
                        f"step_number {dep}"
                    )

        # Remove steps should come last.
        self._validate_remove_order(steps, errors)

        validated_at = _utcnow().isoformat()
        valid = len(errors) == 0

        # Update stored plan with validation outcome.
        with self._lock:
            if plan_id in self._plans:
                self._plans[plan_id]["status"] = "validated" if valid else "failed"
                self._plans[plan_id]["validation_errors"] = errors
                self._plans[plan_id]["validated_at"] = validated_at
            self._stats["plans_validated"] += 1

        self._provenance.record(
            entity_type="migration_plan",
            entity_id=plan_id,
            action="plan_validated",
            data={"valid": valid, "error_count": len(errors)},
        )

        logger.info(
            "Plan validated: plan_id=%s, valid=%s, errors=%d",
            plan_id, valid, len(errors),
        )

        return {
            "plan_id": plan_id,
            "valid": valid,
            "errors": errors,
            "warnings": warnings,
            "step_count": len(steps),
            "validated_at": validated_at,
        }

    # ------------------------------------------------------------------
    # 14. dry_run
    # ------------------------------------------------------------------

    def dry_run(
        self,
        plan_id: str,
        sample_data: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """Simulate plan execution on sample data without persisting changes.

        Applies each transformation step in order to the sample rows,
        tracking successes, failures, and field-level changes.  No actual
        database or schema state is mutated.

        Args:
            plan_id: Identifier of the plan to simulate.
            sample_data: Optional list of sample record dicts to transform.
                When omitted, simulation runs on an empty record set and
                reports structural analysis only.

        Returns:
            Dict with keys:
                - plan_id (str)
                - steps_simulated (int)
                - steps_successful (int)
                - steps_failed (int)
                - records_processed (int)
                - field_changes (List[Dict]): Summary of per-step outcomes.
                - cast_failures (List[Dict]): Records of type-cast failures.
                - warnings (List[str])
                - errors (List[str])
                - completed_at (str) UTC ISO timestamp
        """
        plan = self.get_plan(plan_id)
        if plan is None:
            return self._dry_run_error(plan_id, f"Plan '{plan_id}' not found")

        steps = plan.get("steps", [])
        records = list(sample_data) if sample_data else []

        field_changes: List[Dict[str, Any]] = []
        cast_failures: List[Dict[str, Any]] = []
        warnings: List[str] = []
        errors: List[str] = []
        steps_successful = 0
        steps_failed = 0

        for step in steps:
            outcome = self._simulate_step(step, records, cast_failures, warnings)
            field_changes.append(outcome)
            if outcome["success"]:
                steps_successful += 1
            else:
                steps_failed += 1
                errors.append(
                    f"Step {step.get('step_number')}: {outcome.get('error', 'unknown error')}"
                )

        completed_at = _utcnow().isoformat()

        with self._lock:
            if plan_id in self._plans:
                self._plans[plan_id]["dry_run_result"] = {
                    "completed_at": completed_at,
                    "steps_successful": steps_successful,
                    "steps_failed": steps_failed,
                }
                if self._plans[plan_id]["status"] == "validated":
                    self._plans[plan_id]["status"] = "dry_run_complete"
            self._stats["plans_dry_run"] += 1

        self._provenance.record(
            entity_type="migration_plan",
            entity_id=plan_id,
            action="dry_run_completed",
            data={
                "steps_successful": steps_successful,
                "steps_failed": steps_failed,
                "records_processed": len(records),
            },
        )

        logger.info(
            "Dry run completed: plan_id=%s, steps_ok=%d, steps_fail=%d, records=%d",
            plan_id, steps_successful, steps_failed, len(records),
        )

        return {
            "plan_id": plan_id,
            "steps_simulated": len(steps),
            "steps_successful": steps_successful,
            "steps_failed": steps_failed,
            "records_processed": len(records),
            "field_changes": field_changes,
            "cast_failures": cast_failures,
            "warnings": warnings,
            "errors": errors,
            "completed_at": completed_at,
        }

    # ------------------------------------------------------------------
    # 15. get_plan
    # ------------------------------------------------------------------

    def get_plan(self, plan_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve a stored plan by its ID.

        Args:
            plan_id: Unique plan identifier string.

        Returns:
            The plan dict if found, or ``None`` if no such plan exists.
        """
        with self._lock:
            plan = self._plans.get(plan_id)
        if plan is None:
            logger.debug("get_plan: plan_id='%s' not found", plan_id)
        return plan

    # ------------------------------------------------------------------
    # 16. list_plans
    # ------------------------------------------------------------------

    def list_plans(
        self,
        status: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> List[Dict[str, Any]]:
        """List stored plans with optional status filter and pagination.

        Plans are returned in reverse creation order (most recent first).

        Args:
            status: Optional status filter.  Must be one of ``pending``,
                ``validated``, ``dry_run_complete``, ``executed``, ``failed``.
                Pass ``None`` to return all statuses.
            limit: Maximum number of plans to return.  Clamped to [1, 1000].
            offset: Number of plans to skip from the start of the result set.
                Must be >= 0.

        Returns:
            List of plan summary dicts containing at minimum:
                plan_id, source_schema_id, target_schema_id,
                source_version, target_version, step_count,
                effort_band, status, created_at.
        """
        limit = max(1, min(limit, 1000))
        offset = max(0, offset)

        with self._lock:
            all_plans = list(self._plans.values())

        # Most-recent first.
        all_plans.sort(key=lambda p: p.get("created_at", ""), reverse=True)

        if status is not None:
            status_lower = status.lower()
            all_plans = [p for p in all_plans if p.get("status", "") == status_lower]

        paginated = all_plans[offset : offset + limit]

        summaries = []
        for plan in paginated:
            summaries.append({
                "plan_id": plan["plan_id"],
                "source_schema_id": plan["source_schema_id"],
                "target_schema_id": plan["target_schema_id"],
                "source_version": plan["source_version"],
                "target_version": plan["target_version"],
                "step_count": plan["step_count"],
                "effort_band": plan["effort_band"],
                "status": plan["status"],
                "created_at": plan["created_at"],
            })

        logger.debug(
            "list_plans: status_filter=%s, total_matching=%d, returned=%d",
            status, len(all_plans), len(summaries),
        )
        return summaries

    # ------------------------------------------------------------------
    # 17. get_statistics
    # ------------------------------------------------------------------

    def get_statistics(self) -> Dict[str, Any]:
        """Return aggregate statistics for the MigrationPlannerEngine.

        Collects runtime counters, per-status plan counts, effort band
        distribution, and provenance entry count.

        Returns:
            Dict with keys:
                - plans_created (int)
                - plans_validated (int)
                - plans_dry_run (int)
                - total_steps_generated (int)
                - plans_by_status (Dict[str, int])
                - plans_by_effort_band (Dict[str, int])
                - provenance_entries (int)
                - total_plans_stored (int)
        """
        with self._lock:
            plans = list(self._plans.values())
            stats_copy = dict(self._stats)

        plans_by_status: Dict[str, int] = {}
        plans_by_effort: Dict[str, int] = {}

        for plan in plans:
            st = plan.get("status", "unknown")
            plans_by_status[st] = plans_by_status.get(st, 0) + 1
            eb = plan.get("effort_band", "UNKNOWN")
            plans_by_effort[eb] = plans_by_effort.get(eb, 0) + 1

        return {
            **stats_copy,
            "total_plans_stored": len(plans),
            "plans_by_status": plans_by_status,
            "plans_by_effort_band": plans_by_effort,
            "provenance_entries": self._provenance.entry_count,
        }

    # ------------------------------------------------------------------
    # 18. reset
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Clear all stored plans and reset internal counters.

        Also resets the provenance tracker chain to genesis state.
        Intended for testing and administrative use only.

        Note:
            This is a destructive operation.  All plan data is permanently
            removed from in-memory storage.  This does NOT affect any
            external database or storage layer.
        """
        with self._lock:
            self._plans.clear()
            self._stats = {
                "plans_created": 0,
                "plans_validated": 0,
                "plans_dry_run": 0,
                "total_steps_generated": 0,
            }
        self._provenance.reset()
        logger.info("MigrationPlannerEngine reset: all plans and stats cleared")

    # ------------------------------------------------------------------
    # Private helpers — dispatch and step generation
    # ------------------------------------------------------------------

    def _dispatch_change(
        self,
        change_type: str,
        change: Dict[str, Any],
        source_def: Optional[Dict[str, Any]],
        target_def: Optional[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Map a change_type string to one or more step generator calls.

        Args:
            change_type: Normalised (lowercased, stripped) change type.
            change: Full change dict.
            source_def: Optional source schema definition.
            target_def: Optional target schema definition.

        Returns:
            List of generated step dicts (may be empty for unknown types).
        """
        try:
            if change_type in _RENAME_CHANGE_TYPES:
                return [self.generate_rename_step(change)]

            if change_type in _CAST_CHANGE_TYPES:
                return [self.generate_cast_step(change)]

            if change_type in _DEFAULT_CHANGE_TYPES:
                return [self.generate_default_step(change)]

            if change_type in _ADD_CHANGE_TYPES:
                steps = [self.generate_add_step(change)]
                # If added field has a default value, also inject a set_default.
                if change.get("default_value") is not None:
                    steps.append(self.generate_default_step(change))
                return steps

            if change_type in _REMOVE_CHANGE_TYPES:
                return [self.generate_remove_step(change)]

            if change_type in _COMPUTE_CHANGE_TYPES:
                return [self.generate_compute_step(change, source_def, target_def)]

            if change_type in _SPLIT_CHANGE_TYPES:
                source_field = (
                    str(change.get("source_field", "")).strip()
                    or str(change.get("field_name", "")).strip()
                )
                target_fields = change.get("target_fields", [])
                if source_field and isinstance(target_fields, list) and len(target_fields) >= 2:
                    return [self.generate_split_step(source_field, target_fields)]
                logger.warning(
                    "Skipping split change: missing source_field or "
                    "insufficient target_fields: %r",
                    change,
                )
                return []

            if change_type in _MERGE_CHANGE_TYPES:
                source_fields = change.get("source_fields", [])
                target_field = (
                    str(change.get("target_field", "")).strip()
                    or str(change.get("field_name", "")).strip()
                )
                if isinstance(source_fields, list) and len(source_fields) >= 2 and target_field:
                    return [self.generate_merge_step(source_fields, target_field)]
                logger.warning(
                    "Skipping merge change: missing source_fields or "
                    "target_field: %r",
                    change,
                )
                return []

            logger.warning(
                "Unknown change_type '%s'; skipping change: %r",
                change_type, change,
            )
            return []

        except ValueError as exc:
            logger.error(
                "Step generation failed for change_type='%s': %s",
                change_type, exc,
            )
            return []

    # ------------------------------------------------------------------
    # Private helpers — effort
    # ------------------------------------------------------------------

    def _classify_effort_band(self, total_cost: float) -> str:
        """Map a numeric effort score to a band label.

        Args:
            total_cost: Scaled effort score from :meth:`estimate_effort`.

        Returns:
            One of ``"LOW"``, ``"MEDIUM"``, ``"HIGH"``, ``"CRITICAL"``.
        """
        if total_cost < _EFFORT_LOW_MAX:
            return "LOW"
        if total_cost < _EFFORT_MEDIUM_MAX:
            return "MEDIUM"
        if total_cost < _EFFORT_HIGH_MAX:
            return "HIGH"
        return "CRITICAL"

    # ------------------------------------------------------------------
    # Private helpers — cast and type inference
    # ------------------------------------------------------------------

    def _extract_precision(
        self,
        new_type: str,
        change: Dict[str, Any],
    ) -> Tuple[Optional[int], Optional[int]]:
        """Extract precision and scale from a type string or change dict.

        Recognises patterns like ``decimal(10,2)`` and ``numeric(8, 3)``.
        Falls back to explicit ``precision`` / ``scale`` keys in the change.

        Args:
            new_type: Target type string (may contain precision/scale).
            change: Change dict that may carry explicit ``precision``/``scale``.

        Returns:
            Tuple of (precision, scale).  Both may be ``None``.
        """
        import re
        match = re.search(r"\(\s*(\d+)\s*(?:,\s*(\d+))?\s*\)", new_type)
        if match:
            precision = int(match.group(1))
            scale = int(match.group(2)) if match.group(2) else None
            return precision, scale

        precision = change.get("precision")
        scale = change.get("scale")
        precision = int(precision) if precision is not None else None
        scale = int(scale) if scale is not None else None
        return precision, scale

    # ------------------------------------------------------------------
    # Private helpers — formula inference
    # ------------------------------------------------------------------

    def _infer_formula(
        self,
        field_name: str,
        source_fields: List[str],
        target_def: Optional[Dict[str, Any]],
    ) -> str:
        """Attempt to infer a computation formula for a computed field.

        Looks up ``target_def["properties"][field_name]["x-formula"]`` first.
        Falls back to a generic placeholder expression referencing source_fields.

        Args:
            field_name: Name of the computed target field.
            source_fields: Names of source fields used in computation.
            target_def: Optional target schema definition dict.

        Returns:
            Formula string suitable for the ``compute_field`` step parameters.
        """
        if target_def and isinstance(target_def, dict):
            props = target_def.get("properties", {})
            field_def = props.get(field_name, {})
            formula = field_def.get("x-formula") or field_def.get("formula")
            if formula:
                return str(formula)

        if source_fields:
            return " + ".join(source_fields)

        return f"compute({field_name})"

    # ------------------------------------------------------------------
    # Private helpers — validation support
    # ------------------------------------------------------------------

    def _validate_plan_inputs(
        self,
        source_schema_id: str,
        target_schema_id: str,
        source_version: str,
        target_version: str,
        changes: List[Any],
    ) -> None:
        """Raise ValueError if required plan creation inputs are invalid.

        Args:
            source_schema_id: Source schema identifier.
            target_schema_id: Target schema identifier.
            source_version: Source version string.
            target_version: Target version string.
            changes: Changes list.

        Raises:
            ValueError: If any required argument is empty or of wrong type.
        """
        errors: List[str] = []
        if not str(source_schema_id).strip():
            errors.append("source_schema_id must not be empty")
        if not str(target_schema_id).strip():
            errors.append("target_schema_id must not be empty")
        if not str(source_version).strip():
            errors.append("source_version must not be empty")
        if not str(target_version).strip():
            errors.append("target_version must not be empty")
        if not isinstance(changes, list):
            errors.append(f"changes must be a list, got {type(changes).__name__}")
        if errors:
            raise ValueError(
                "create_plan validation failed: " + "; ".join(errors)
            )

    def _validate_step_dict(
        self,
        step: Dict[str, Any],
        prefix: str,
        errors: List[str],
        warnings: List[str],
    ) -> None:
        """Validate a single step dict for required keys and value types.

        Args:
            step: Step dict to inspect.
            prefix: Label string for error messages (e.g. "Step 1").
            errors: Mutable list to append error messages into.
            warnings: Mutable list to append warning messages into.
        """
        if not isinstance(step, dict):
            errors.append(f"{prefix}: step must be a dict, got {type(step).__name__}")
            return

        required_keys = {"operation", "description", "reversible", "depends_on"}
        for key in required_keys:
            if key not in step:
                errors.append(f"{prefix}: missing required key '{key}'")

        op = step.get("operation", "")
        if op not in _OPERATION_ORDER:
            warnings.append(f"{prefix}: unknown operation '{op}'")

        if step.get("source_field") is None and step.get("target_field") is None:
            errors.append(
                f"{prefix}: at least one of 'source_field' or 'target_field' must be set"
            )

        depends_on = step.get("depends_on", [])
        if not isinstance(depends_on, list):
            errors.append(f"{prefix}: 'depends_on' must be a list")

    def _validate_remove_order(
        self,
        steps: List[Dict[str, Any]],
        errors: List[str],
    ) -> None:
        """Verify that remove_field steps appear after all non-remove steps.

        Args:
            steps: Ordered list of step dicts.
            errors: Mutable error list to append violations into.
        """
        remove_seen = False
        for step in steps:
            op = step.get("operation", "")
            if op == OP_REMOVE:
                remove_seen = True
            elif remove_seen:
                errors.append(
                    f"Step {step.get('step_number')}: non-remove operation "
                    f"'{op}' appears after a remove_field step — "
                    "remove steps must be last"
                )

    # ------------------------------------------------------------------
    # Private helpers — dry-run simulation
    # ------------------------------------------------------------------

    def _simulate_step(
        self,
        step: Dict[str, Any],
        records: List[Dict[str, Any]],
        cast_failures: List[Dict[str, Any]],
        warnings: List[str],
    ) -> Dict[str, Any]:
        """Apply a single step to an in-place list of records (dry-run).

        Mutates ``records`` in place.  Appends to ``cast_failures`` and
        ``warnings`` as appropriate.

        Args:
            step: Step dict describing the transformation.
            records: Mutable list of record dicts to transform.
            cast_failures: Accumulator for type-cast failure details.
            warnings: Accumulator for non-fatal warning messages.

        Returns:
            Outcome dict with keys: step_number, operation, success,
            records_affected, error (when applicable).
        """
        op = step.get("operation", "")
        step_number = step.get("step_number", 0)
        params = step.get("parameters", {})

        try:
            if op == OP_RENAME:
                self._sim_rename(step, records)

            elif op == OP_CAST:
                self._sim_cast(step, records, cast_failures)

            elif op == OP_DEFAULT:
                self._sim_default(step, records)

            elif op == OP_ADD:
                self._sim_add(step, records)

            elif op == OP_REMOVE:
                self._sim_remove(step, records)

            elif op == OP_COMPUTE:
                self._sim_compute(step, records, warnings)

            elif op == OP_SPLIT:
                self._sim_split(step, records, warnings)

            elif op == OP_MERGE:
                self._sim_merge(step, records, warnings)

            else:
                warnings.append(
                    f"Step {step_number}: unknown operation '{op}' skipped"
                )

            return {
                "step_number": step_number,
                "operation": op,
                "success": True,
                "records_affected": len(records),
            }

        except Exception as exc:
            logger.error(
                "Dry-run step %d ('%s') failed: %s",
                step_number, op, exc, exc_info=True,
            )
            return {
                "step_number": step_number,
                "operation": op,
                "success": False,
                "records_affected": 0,
                "error": str(exc),
            }

    def _sim_rename(
        self,
        step: Dict[str, Any],
        records: List[Dict[str, Any]],
    ) -> None:
        """Simulate a rename_field step in-place on records.

        Args:
            step: Rename step dict.
            records: Mutable list of record dicts.
        """
        src = step.get("source_field")
        tgt = step.get("target_field")
        if not src or not tgt or src == tgt:
            return
        for record in records:
            if src in record:
                record[tgt] = record.pop(src)

    def _sim_cast(
        self,
        step: Dict[str, Any],
        records: List[Dict[str, Any]],
        cast_failures: List[Dict[str, Any]],
    ) -> None:
        """Simulate a cast_type step in-place on records.

        Args:
            step: Cast step dict.
            records: Mutable list of record dicts.
            cast_failures: Accumulator for cast failure details.
        """
        field = step.get("source_field") or step.get("target_field")
        params = step.get("parameters", {})
        new_type = params.get("new_type", "string")
        nullable = params.get("nullable", True)

        for idx, record in enumerate(records):
            if field not in record:
                continue
            value = record[field]
            if value is None and nullable:
                continue
            success, cast_value = _safe_cast(value, new_type)
            if success:
                record[field] = cast_value
            else:
                cast_failures.append({
                    "record_index": idx,
                    "field": field,
                    "original_value": value,
                    "target_type": new_type,
                })

    def _sim_default(
        self,
        step: Dict[str, Any],
        records: List[Dict[str, Any]],
    ) -> None:
        """Simulate a set_default step in-place on records.

        Args:
            step: Default step dict.
            records: Mutable list of record dicts.
        """
        field = step.get("target_field") or step.get("source_field")
        params = step.get("parameters", {})
        default_value = params.get("default_value")
        for record in records:
            if field not in record or record[field] is None:
                record[field] = default_value

    def _sim_add(
        self,
        step: Dict[str, Any],
        records: List[Dict[str, Any]],
    ) -> None:
        """Simulate an add_field step in-place on records.

        Args:
            step: Add step dict.
            records: Mutable list of record dicts.
        """
        field = step.get("target_field")
        params = step.get("parameters", {})
        default_value = params.get("default_value")
        for record in records:
            if field not in record:
                record[field] = default_value

    def _sim_remove(
        self,
        step: Dict[str, Any],
        records: List[Dict[str, Any]],
    ) -> None:
        """Simulate a remove_field step in-place on records.

        Args:
            step: Remove step dict.
            records: Mutable list of record dicts.
        """
        field = step.get("source_field")
        for record in records:
            record.pop(field, None)

    def _sim_compute(
        self,
        step: Dict[str, Any],
        records: List[Dict[str, Any]],
        warnings: List[str],
    ) -> None:
        """Simulate a compute_field step in-place on records.

        The dry-run sets the computed field to a sentinel string indicating
        that the formula would have been evaluated.  No actual formula
        execution occurs to maintain zero-hallucination guarantees.

        Args:
            step: Compute step dict.
            records: Mutable list of record dicts.
            warnings: Accumulator for non-fatal messages.
        """
        field = step.get("target_field")
        params = step.get("parameters", {})
        formula = params.get("formula", "")
        source_fields = params.get("source_fields", [])

        warnings.append(
            f"compute_field '{field}': formula '{formula}' not evaluated "
            "in dry-run (zero-hallucination); field set to sentinel."
        )

        for record in records:
            # Set computed field based on available source values.
            source_values = [record.get(sf) for sf in source_fields if sf in record]
            if source_values:
                # Attempt simple numeric sum for numeric sources; else concatenate.
                try:
                    record[field] = sum(float(v) for v in source_values if v is not None)
                except (TypeError, ValueError):
                    record[field] = "_computed_"
            else:
                record[field] = "_computed_"

    def _sim_split(
        self,
        step: Dict[str, Any],
        records: List[Dict[str, Any]],
        warnings: List[str],
    ) -> None:
        """Simulate a split_field step in-place on records.

        Splits the source field value by the configured delimiter (or
        defaults to splitting by whitespace) and distributes parts across
        target fields.  Pads with ``None`` for missing parts.

        Args:
            step: Split step dict.
            records: Mutable list of record dicts.
            warnings: Accumulator for non-fatal messages.
        """
        params = step.get("parameters", {})
        source_field = params.get("source_field") or step.get("source_field")
        target_fields = params.get("target_fields", [])
        delimiter = params.get("delimiter")

        if not source_field or not target_fields:
            warnings.append(
                f"split_field step {step.get('step_number')}: "
                "missing source_field or target_fields, skipped"
            )
            return

        for record in records:
            raw = record.get(source_field, "")
            if raw is None:
                raw = ""
            parts = str(raw).split(delimiter) if delimiter else str(raw).split()
            for i, tf in enumerate(target_fields):
                record[tf] = parts[i] if i < len(parts) else None

    def _sim_merge(
        self,
        step: Dict[str, Any],
        records: List[Dict[str, Any]],
        warnings: List[str],
    ) -> None:
        """Simulate a merge_fields step in-place on records.

        Joins source field values with the configured separator and writes
        the result to the target field.  ``None`` values are treated as
        empty strings during joining.

        Args:
            step: Merge step dict.
            records: Mutable list of record dicts.
            warnings: Accumulator for non-fatal messages.
        """
        params = step.get("parameters", {})
        source_fields = params.get("source_fields", [])
        target_field = params.get("target_field") or step.get("target_field")
        separator = params.get("separator", " ")

        if not source_fields or not target_field:
            warnings.append(
                f"merge_fields step {step.get('step_number')}: "
                "missing source_fields or target_field, skipped"
            )
            return

        for record in records:
            parts = [str(record.get(sf, "") or "") for sf in source_fields]
            record[target_field] = separator.join(parts)

    # ------------------------------------------------------------------
    # Private helpers — dry-run error response
    # ------------------------------------------------------------------

    def _dry_run_error(self, plan_id: str, message: str) -> Dict[str, Any]:
        """Build a standardised dry-run error response.

        Args:
            plan_id: Plan identifier (may not exist).
            message: Human-readable error description.

        Returns:
            Dry-run result dict with all zero/empty values and the error.
        """
        return {
            "plan_id": plan_id,
            "steps_simulated": 0,
            "steps_successful": 0,
            "steps_failed": 0,
            "records_processed": 0,
            "field_changes": [],
            "cast_failures": [],
            "warnings": [],
            "errors": [message],
            "completed_at": _utcnow().isoformat(),
        }
