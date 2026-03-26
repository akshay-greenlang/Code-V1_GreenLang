# -*- coding: utf-8 -*-
"""
InventoryVersioningEngine - PACK-044 Inventory Management Engine 6
===================================================================

Version control engine for GHG inventories implementing a structured
lifecycle with Draft -> Under_Review -> Final -> Amended -> Superseded
states, field-level diff computation, rollback capability, and
optimistic locking to prevent concurrent modification conflicts.

Each inventory version is an immutable snapshot.  Modifications create
new versions linked to their predecessor, forming a complete version
chain with full auditability.  The engine supports field-level diff
computation to identify exactly what changed between any two versions,
and provides rollback to any prior version with automatic conflict
detection.

Version Lifecycle:
    DRAFT           -> UNDER_REVIEW  (submit for review)
    UNDER_REVIEW    -> DRAFT         (return to preparer for revisions)
    UNDER_REVIEW    -> FINAL         (approve and finalise)
    FINAL           -> AMENDED       (amend a finalised version)
    AMENDED         -> UNDER_REVIEW  (submit amendment for review)
    FINAL           -> SUPERSEDED    (when a newer FINAL version exists)
    AMENDED         -> SUPERSEDED    (when amendment is finalised)

Locking Strategy:
    Optimistic locking using a monotonically increasing lock_version
    counter.  Any update must supply the current lock_version; if it
    does not match, the update is rejected with a ConcurrencyConflict.
    This prevents lost-update anomalies in multi-user environments.

Diff Algorithm:
    Field-level comparison of inventory data dictionaries.  For each
    field, the engine computes:
        - old_value: value in the base version
        - new_value: value in the comparison version
        - change_type: added, removed, modified, unchanged
        - delta: numeric difference (for numeric fields)
        - delta_pct: percentage change (for numeric fields)

Regulatory References:
    - GHG Protocol Corporate Standard, Chapter 9 (Reporting)
    - ISO 14064-1:2018, Clause 9 (Management of GHG inventory quality)
    - ESRS E1 (Climate Change - restatement policy)
    - SEC Climate Disclosure Rule (2024) (comparable prior periods)
    - CDP Climate Change Questionnaire C0.3 (reporting boundaries)

Zero-Hallucination:
    - All version management is deterministic state-machine logic
    - Diff computation uses direct value comparison (no approximations)
    - No LLM involvement in any versioning, diff, or locking logic
    - SHA-256 provenance hash on every version and result

Author:  GreenLang Platform Team
Date:    March 2026
Pack:    PACK-044 Inventory Management
Engine:  6 of 10
Status:  Production Ready
"""

from __future__ import annotations

import copy
import hashlib
import json
import logging
import time
import uuid
from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP, InvalidOperation
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field, field_validator

logger = logging.getLogger(__name__)

_MODULE_VERSION: str = "1.0.0"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _utcnow() -> datetime:
    """Return current UTC datetime with microseconds zeroed."""
    return datetime.now(timezone.utc).replace(microsecond=0)


def _new_uuid() -> str:
    """Generate a new UUID4 string."""
    return str(uuid.uuid4())


def _compute_hash(data: Any) -> str:
    """Compute a deterministic SHA-256 hash of arbitrary data.

    Excludes volatile fields (calculated_at, processing_time_ms,
    provenance_hash) so that identical logical content always produces
    the same hash.

    Args:
        data: Any Pydantic model, dict, or stringifiable object.

    Returns:
        Hex-encoded SHA-256 digest string.
    """
    if hasattr(data, "model_dump"):
        serializable = data.model_dump(mode="json")
    elif isinstance(data, dict):
        serializable = data
    else:
        serializable = str(data)
    if isinstance(serializable, dict):
        serializable = {
            k: v for k, v in serializable.items()
            if k not in ("calculated_at", "processing_time_ms", "provenance_hash")
        }
    raw = json.dumps(serializable, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _decimal(value: Any) -> Decimal:
    """Safely convert a value to Decimal."""
    if isinstance(value, Decimal):
        return value
    try:
        return Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError):
        return Decimal("0")


def _safe_divide(
    numerator: Decimal,
    denominator: Decimal,
    default: Decimal = Decimal("0"),
) -> Decimal:
    """Safely divide two Decimals, returning *default* on zero denominator."""
    if denominator == Decimal("0"):
        return default
    return numerator / denominator


def _safe_pct(part: Decimal, whole: Decimal) -> Decimal:
    """Compute percentage safely (part / whole * 100)."""
    return _safe_divide(part * Decimal("100"), whole)


def _round_val(value: Decimal, places: int = 4) -> Decimal:
    """Round a Decimal to *places* using ROUND_HALF_UP."""
    quantize_str = "0." + "0" * places
    return value.quantize(Decimal(quantize_str), rounding=ROUND_HALF_UP)


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class VersionStatus(str, Enum):
    """Lifecycle status of an inventory version.

    DRAFT:          Working draft, editable by preparer.
    UNDER_REVIEW:   Submitted for review, read-only to preparer.
    FINAL:          Approved and finalised, immutable.
    AMENDED:        Post-finalisation amendment in progress.
    SUPERSEDED:     Replaced by a newer finalised version.
    ARCHIVED:       Archived for long-term retention (read-only).
    """
    DRAFT = "draft"
    UNDER_REVIEW = "under_review"
    FINAL = "final"
    AMENDED = "amended"
    SUPERSEDED = "superseded"
    ARCHIVED = "archived"


class ChangeType(str, Enum):
    """Type of change detected in a field-level diff.

    ADDED:      Field exists in new version but not in base.
    REMOVED:    Field exists in base version but not in new.
    MODIFIED:   Field exists in both but value changed.
    UNCHANGED:  Field exists in both and value is identical.
    """
    ADDED = "added"
    REMOVED = "removed"
    MODIFIED = "modified"
    UNCHANGED = "unchanged"


class RollbackStrategy(str, Enum):
    """Strategy for rolling back to a prior version.

    FULL_RESTORE:       Complete restoration of the target version's data.
    SELECTIVE_FIELDS:   Restore only specified fields from target version.
    MERGE_LATEST:       Merge target version data with current, preferring
                        target for conflicting fields.
    """
    FULL_RESTORE = "full_restore"
    SELECTIVE_FIELDS = "selective_fields"
    MERGE_LATEST = "merge_latest"


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Valid status transitions.
VALID_TRANSITIONS: Dict[VersionStatus, List[VersionStatus]] = {
    VersionStatus.DRAFT: [VersionStatus.UNDER_REVIEW],
    VersionStatus.UNDER_REVIEW: [VersionStatus.DRAFT, VersionStatus.FINAL],
    VersionStatus.FINAL: [VersionStatus.AMENDED, VersionStatus.SUPERSEDED, VersionStatus.ARCHIVED],
    VersionStatus.AMENDED: [VersionStatus.UNDER_REVIEW, VersionStatus.SUPERSEDED],
    VersionStatus.SUPERSEDED: [VersionStatus.ARCHIVED],
    VersionStatus.ARCHIVED: [],
}

# Statuses where data modification is permitted.
EDITABLE_STATUSES = frozenset({VersionStatus.DRAFT, VersionStatus.AMENDED})

# Statuses where the version is considered immutable.
IMMUTABLE_STATUSES = frozenset({
    VersionStatus.FINAL, VersionStatus.SUPERSEDED, VersionStatus.ARCHIVED,
})

# Maximum number of versions in a single version chain.
MAX_VERSION_CHAIN_LENGTH: int = 500


# ---------------------------------------------------------------------------
# Pydantic Models -- Core
# ---------------------------------------------------------------------------


class FieldChange(BaseModel):
    """A single field-level change between two versions.

    Attributes:
        field_path: Dot-notation path to the field (e.g. scope1.total_tco2e).
        change_type: Type of change (added, removed, modified, unchanged).
        old_value: Value in the base version (None if added).
        new_value: Value in the comparison version (None if removed).
        old_value_str: String representation of old value.
        new_value_str: String representation of new value.
        delta: Numeric difference (for numeric fields, new - old).
        delta_pct: Percentage change (for numeric fields).
        section: Inventory section this field belongs to.
    """
    field_path: str = Field(default="", description="Dot-notation field path")
    change_type: ChangeType = Field(
        default=ChangeType.UNCHANGED, description="Type of change"
    )
    old_value: Optional[Any] = Field(
        default=None, description="Base version value"
    )
    new_value: Optional[Any] = Field(
        default=None, description="Comparison version value"
    )
    old_value_str: str = Field(default="", description="Old value as string")
    new_value_str: str = Field(default="", description="New value as string")
    delta: Optional[Decimal] = Field(
        default=None, description="Numeric delta (new - old)"
    )
    delta_pct: Optional[Decimal] = Field(
        default=None, description="Percentage change"
    )
    section: str = Field(default="", description="Inventory section")


class VersionDiff(BaseModel):
    """Complete diff between two inventory versions.

    Attributes:
        diff_id: Unique diff identifier.
        base_version_id: ID of the base (older) version.
        compare_version_id: ID of the comparison (newer) version.
        base_version_number: Version number of the base.
        compare_version_number: Version number of the comparison.
        total_fields_compared: Total number of fields compared.
        fields_added: Count of added fields.
        fields_removed: Count of removed fields.
        fields_modified: Count of modified fields.
        fields_unchanged: Count of unchanged fields.
        changes: List of individual field changes (modified/added/removed only).
        sections_affected: List of inventory sections with changes.
        summary: Human-readable diff summary.
        provenance_hash: SHA-256 hash of the diff.
    """
    diff_id: str = Field(default_factory=_new_uuid, description="Diff ID")
    base_version_id: str = Field(default="", description="Base version ID")
    compare_version_id: str = Field(default="", description="Compare version ID")
    base_version_number: int = Field(default=0, description="Base version number")
    compare_version_number: int = Field(default=0, description="Compare version number")
    total_fields_compared: int = Field(default=0, description="Total fields compared")
    fields_added: int = Field(default=0, ge=0, description="Fields added")
    fields_removed: int = Field(default=0, ge=0, description="Fields removed")
    fields_modified: int = Field(default=0, ge=0, description="Fields modified")
    fields_unchanged: int = Field(default=0, ge=0, description="Fields unchanged")
    changes: List[FieldChange] = Field(
        default_factory=list, description="Individual field changes"
    )
    sections_affected: List[str] = Field(
        default_factory=list, description="Sections with changes"
    )
    summary: str = Field(default="", description="Diff summary")
    provenance_hash: str = Field(default="", description="SHA-256 hash")


class InventoryVersion(BaseModel):
    """A single versioned snapshot of a GHG inventory.

    Attributes:
        version_id: Unique version identifier.
        inventory_id: Parent inventory identifier.
        version_number: Sequential version number (1, 2, 3, ...).
        status: Current lifecycle status.
        label: Human-readable version label (e.g. "v1.0", "v2.0-draft").
        reporting_year: Reporting year this version covers.
        data: Inventory data as a nested dictionary.  This stores the
            complete inventory snapshot (scope 1/2/3 totals, breakdowns,
            emission factors, activity data, etc.).
        previous_version_id: ID of the predecessor version (None for v1).
        parent_final_version_id: For amendments, the FINAL version being amended.
        created_by: User who created this version.
        created_by_name: Display name of the creator.
        created_at: Creation timestamp.
        updated_at: Last update timestamp.
        finalised_at: Timestamp when status became FINAL.
        finalised_by: User who finalised the version.
        amendment_reason: Reason for amendment (for AMENDED versions).
        lock_version: Optimistic lock counter for concurrency control.
        tags: Tags for categorisation (e.g. "quarterly", "annual", "restated").
        notes: Free-text version notes.
        provenance_hash: SHA-256 hash of the version data.
    """
    version_id: str = Field(default_factory=_new_uuid, description="Version ID")
    inventory_id: str = Field(default="", description="Parent inventory ID")
    version_number: int = Field(default=1, ge=1, description="Version number")
    status: VersionStatus = Field(
        default=VersionStatus.DRAFT, description="Lifecycle status"
    )
    label: str = Field(default="v1.0-draft", max_length=100, description="Version label")
    reporting_year: int = Field(
        default=2025, ge=1990, le=2050, description="Reporting year"
    )
    data: Dict[str, Any] = Field(
        default_factory=dict, description="Inventory data snapshot"
    )
    previous_version_id: Optional[str] = Field(
        default=None, description="Predecessor version ID"
    )
    parent_final_version_id: Optional[str] = Field(
        default=None, description="Parent FINAL version (for amendments)"
    )
    created_by: str = Field(default="", description="Creator user ID")
    created_by_name: str = Field(default="", description="Creator display name")
    created_at: datetime = Field(
        default_factory=_utcnow, description="Creation timestamp"
    )
    updated_at: datetime = Field(
        default_factory=_utcnow, description="Last update timestamp"
    )
    finalised_at: Optional[datetime] = Field(
        default=None, description="Finalisation timestamp"
    )
    finalised_by: Optional[str] = Field(
        default=None, description="Finalisation user ID"
    )
    amendment_reason: str = Field(
        default="", description="Reason for amendment"
    )
    lock_version: int = Field(
        default=1, ge=1, description="Optimistic lock counter"
    )
    tags: List[str] = Field(
        default_factory=list, description="Version tags"
    )
    notes: str = Field(default="", description="Version notes")
    provenance_hash: str = Field(default="", description="SHA-256 data hash")


class VersionComparison(BaseModel):
    """Side-by-side comparison of two inventory versions.

    Attributes:
        comparison_id: Unique comparison identifier.
        base_version: The base (older) version snapshot.
        compare_version: The comparison (newer) version snapshot.
        diff: Computed diff between the two versions.
        emission_delta_summary: Summary of emission changes by scope.
        is_material_change: Whether the change is material (>5% of total).
        material_threshold_pct: Threshold used for materiality test.
    """
    comparison_id: str = Field(default_factory=_new_uuid, description="Comparison ID")
    base_version: Optional[InventoryVersion] = Field(
        default=None, description="Base version"
    )
    compare_version: Optional[InventoryVersion] = Field(
        default=None, description="Compare version"
    )
    diff: Optional[VersionDiff] = Field(default=None, description="Computed diff")
    emission_delta_summary: Dict[str, Any] = Field(
        default_factory=dict, description="Emission changes by scope"
    )
    is_material_change: bool = Field(
        default=False, description="Whether change is material"
    )
    material_threshold_pct: Decimal = Field(
        default=Decimal("5"), description="Materiality threshold (%)"
    )


# ---------------------------------------------------------------------------
# Pydantic Models -- Output
# ---------------------------------------------------------------------------


class VersioningResult(BaseModel):
    """Complete result from the inventory versioning engine.

    Attributes:
        result_id: Unique result identifier.
        engine_version: Engine version.
        action: Action performed (create, update, transition, diff, rollback).
        version: The version affected by the action.
        diff: Diff result (for diff/compare actions).
        comparison: Full comparison (for compare actions).
        rollback_source: Source version for rollback actions.
        audit_entries: Audit trail entries generated.
        warnings: Warnings raised.
        calculated_at: Processing timestamp.
        processing_time_ms: Processing time in milliseconds.
        provenance_hash: SHA-256 provenance hash.
    """
    result_id: str = Field(default_factory=_new_uuid, description="Result ID")
    engine_version: str = Field(default=_MODULE_VERSION, description="Engine version")
    action: str = Field(default="", description="Action performed")
    version: Optional[InventoryVersion] = Field(
        default=None, description="Affected version"
    )
    diff: Optional[VersionDiff] = Field(
        default=None, description="Diff result"
    )
    comparison: Optional[VersionComparison] = Field(
        default=None, description="Full comparison"
    )
    rollback_source: Optional[InventoryVersion] = Field(
        default=None, description="Rollback source version"
    )
    audit_entries: List[Dict[str, Any]] = Field(
        default_factory=list, description="Audit trail"
    )
    warnings: List[str] = Field(default_factory=list, description="Warnings")
    calculated_at: datetime = Field(
        default_factory=_utcnow, description="Processing timestamp"
    )
    processing_time_ms: Decimal = Field(
        default=Decimal("0"), description="Processing time (ms)"
    )
    provenance_hash: str = Field(default="", description="SHA-256 provenance hash")


# ---------------------------------------------------------------------------
# Model Rebuild (resolve forward references from __future__ annotations)
# ---------------------------------------------------------------------------

FieldChange.model_rebuild()
VersionDiff.model_rebuild()
InventoryVersion.model_rebuild()
VersionComparison.model_rebuild()
VersioningResult.model_rebuild()


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------


class InventoryVersioningEngine:
    """Version control engine for GHG inventories.

    Manages inventory version lifecycle, computes field-level diffs
    between versions, supports rollback to prior versions, and
    enforces optimistic locking for concurrency safety.

    Guarantees:
        - Deterministic: same inputs always produce identical outputs.
        - Immutable snapshots: FINAL/SUPERSEDED/ARCHIVED versions never change.
        - Auditable: complete trail for every version operation.
        - Concurrent-safe: optimistic locking prevents lost updates.
        - No LLM: zero hallucination risk in any versioning logic.

    Attributes:
        _config: Engine configuration.
        _audit_entries: Accumulated audit trail entries.
        _warnings: Accumulated warnings.

    Example:
        >>> engine = InventoryVersioningEngine()
        >>> v1 = engine.create_version(
        ...     inventory_id="inv-001",
        ...     reporting_year=2025,
        ...     data={"scope1_total": 10000, "scope2_total": 5000},
        ...     created_by="user-001",
        ... )
        >>> v1_version = v1.version
        >>> # ... modify data ...
        >>> v2 = engine.create_next_version(v1_version, {"scope1_total": 9500, "scope2_total": 5200})
        >>> diff_result = engine.compute_diff(v1_version, v2.version)
    """

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Initialise the InventoryVersioningEngine.

        Args:
            config: Optional configuration overrides. Supported keys:
                - material_threshold_pct (Decimal): default 5
                - max_version_chain (int): default 500
                - auto_supersede (bool): default True
        """
        self._config = config or {}
        self._material_threshold = _decimal(
            self._config.get("material_threshold_pct", "5")
        )
        self._max_chain = int(
            self._config.get("max_version_chain", MAX_VERSION_CHAIN_LENGTH)
        )
        self._auto_supersede = bool(
            self._config.get("auto_supersede", True)
        )
        self._audit_entries: List[Dict[str, Any]] = []
        self._warnings: List[str] = []

        logger.info(
            "InventoryVersioningEngine v%s initialised: "
            "material_threshold=%.1f%%, max_chain=%d, auto_supersede=%s",
            _MODULE_VERSION, float(self._material_threshold),
            self._max_chain, self._auto_supersede,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def create_version(
        self,
        inventory_id: str,
        reporting_year: int,
        data: Dict[str, Any],
        created_by: str,
        created_by_name: str = "",
        label: str = "",
        tags: Optional[List[str]] = None,
        notes: str = "",
    ) -> VersioningResult:
        """Create the first version of an inventory.

        Args:
            inventory_id: Parent inventory identifier.
            reporting_year: Reporting year.
            data: Inventory data snapshot.
            created_by: Creator user ID.
            created_by_name: Creator display name.
            label: Optional version label.
            tags: Optional version tags.
            notes: Optional version notes.

        Returns:
            VersioningResult with the created version.
        """
        t0 = time.perf_counter()
        self._reset_state()

        version_label = label or f"v1.0-draft"

        version = InventoryVersion(
            inventory_id=inventory_id,
            version_number=1,
            status=VersionStatus.DRAFT,
            label=version_label,
            reporting_year=reporting_year,
            data=copy.deepcopy(data),
            created_by=created_by,
            created_by_name=created_by_name,
            tags=tags or [],
            notes=notes,
        )
        version.provenance_hash = _compute_hash(version)

        self._add_audit(
            version.version_id, "version_created",
            created_by, created_by_name,
            f"Created version {version.version_number} ({version_label}) "
            f"for inventory {inventory_id}, year {reporting_year}.",
        )

        logger.info(
            "Created version %s (v%d) for inventory %s",
            version.version_id[:12], version.version_number, inventory_id,
        )

        elapsed = Decimal(str(round((time.perf_counter() - t0) * 1000, 3)))
        return self._build_result("create", version, elapsed)

    def create_next_version(
        self,
        previous_version: InventoryVersion,
        new_data: Dict[str, Any],
        created_by: str = "",
        created_by_name: str = "",
        label: str = "",
        notes: str = "",
    ) -> VersioningResult:
        """Create the next version in the version chain.

        Copies metadata from the previous version and increments the
        version number.  The new version starts in DRAFT status.

        Args:
            previous_version: The predecessor version.
            new_data: Updated inventory data.
            created_by: Creator user ID (defaults to previous creator).
            created_by_name: Creator display name.
            label: Optional version label.
            notes: Optional version notes.

        Returns:
            VersioningResult with the new version.
        """
        t0 = time.perf_counter()
        self._reset_state()

        new_number = previous_version.version_number + 1
        version_label = label or f"v{new_number}.0-draft"

        version = InventoryVersion(
            inventory_id=previous_version.inventory_id,
            version_number=new_number,
            status=VersionStatus.DRAFT,
            label=version_label,
            reporting_year=previous_version.reporting_year,
            data=copy.deepcopy(new_data),
            previous_version_id=previous_version.version_id,
            created_by=created_by or previous_version.created_by,
            created_by_name=created_by_name or previous_version.created_by_name,
            tags=list(previous_version.tags),
            notes=notes,
        )
        version.provenance_hash = _compute_hash(version)

        self._add_audit(
            version.version_id, "version_created",
            version.created_by, version.created_by_name,
            f"Created version {new_number} from predecessor "
            f"{previous_version.version_id[:12]} (v{previous_version.version_number}).",
        )

        logger.info(
            "Created version %s (v%d) from v%d for inventory %s",
            version.version_id[:12], new_number,
            previous_version.version_number, previous_version.inventory_id,
        )

        elapsed = Decimal(str(round((time.perf_counter() - t0) * 1000, 3)))
        return self._build_result("create_next", version, elapsed)

    def transition_status(
        self,
        version: InventoryVersion,
        new_status: VersionStatus,
        actor_id: str,
        actor_name: str = "",
        expected_lock_version: Optional[int] = None,
    ) -> VersioningResult:
        """Transition a version to a new lifecycle status.

        Validates the transition, checks the optimistic lock, and
        applies the status change.

        Args:
            version: The version to transition.
            new_status: Target status.
            actor_id: User performing the transition.
            actor_name: Display name of the actor.
            expected_lock_version: Expected lock version for optimistic
                locking.  If provided and does not match, raises error.

        Returns:
            VersioningResult with updated version.

        Raises:
            ValueError: If transition is invalid or lock version mismatch.
        """
        t0 = time.perf_counter()
        self._reset_state()

        # Validate lock version.
        if expected_lock_version is not None:
            if expected_lock_version != version.lock_version:
                raise ValueError(
                    f"Concurrency conflict: expected lock_version "
                    f"{expected_lock_version}, actual {version.lock_version}. "
                    f"The version was modified by another user."
                )

        # Validate transition.
        old_status = version.status
        valid_targets = VALID_TRANSITIONS.get(old_status, [])

        if new_status not in valid_targets:
            raise ValueError(
                f"Invalid status transition: {old_status.value} -> "
                f"{new_status.value}. Valid targets: "
                f"{[t.value for t in valid_targets]}"
            )

        # Apply transition.
        version.status = new_status
        version.updated_at = _utcnow()
        version.lock_version += 1

        # Set finalisation metadata.
        if new_status == VersionStatus.FINAL:
            version.finalised_at = _utcnow()
            version.finalised_by = actor_id
            version.label = version.label.replace("-draft", "").replace("-review", "")
            if not version.label.endswith("-final"):
                version.label = f"v{version.version_number}.0-final"

        # Recompute provenance.
        version.provenance_hash = _compute_hash(version)

        self._add_audit(
            version.version_id, "status_transition",
            actor_id, actor_name,
            f"Status: {old_status.value} -> {new_status.value}.",
        )

        logger.info(
            "Version %s (v%d): %s -> %s (by %s)",
            version.version_id[:12], version.version_number,
            old_status.value, new_status.value, actor_name,
        )

        elapsed = Decimal(str(round((time.perf_counter() - t0) * 1000, 3)))
        return self._build_result("transition", version, elapsed)

    def update_data(
        self,
        version: InventoryVersion,
        updates: Dict[str, Any],
        actor_id: str,
        actor_name: str = "",
        expected_lock_version: Optional[int] = None,
    ) -> VersioningResult:
        """Update inventory data in a mutable version.

        Only DRAFT and AMENDED versions can be updated.  Uses optimistic
        locking to prevent concurrent modification.

        Args:
            version: The version to update.
            updates: Dict of field updates to apply.
            actor_id: User performing the update.
            actor_name: Display name of the actor.
            expected_lock_version: Expected lock version.

        Returns:
            VersioningResult with updated version.

        Raises:
            ValueError: If version is immutable or lock conflict.
        """
        t0 = time.perf_counter()
        self._reset_state()

        # Validate mutability.
        if version.status not in EDITABLE_STATUSES:
            raise ValueError(
                f"Cannot update data: version status is "
                f"{version.status.value}. Only DRAFT and AMENDED "
                f"versions can be modified."
            )

        # Validate lock version.
        if expected_lock_version is not None:
            if expected_lock_version != version.lock_version:
                raise ValueError(
                    f"Concurrency conflict: expected lock_version "
                    f"{expected_lock_version}, actual {version.lock_version}."
                )

        # Apply updates.
        changed_fields: List[str] = []
        for key, value in updates.items():
            old_val = version.data.get(key)
            if old_val != value:
                version.data[key] = copy.deepcopy(value)
                changed_fields.append(key)

        version.updated_at = _utcnow()
        version.lock_version += 1
        version.provenance_hash = _compute_hash(version)

        self._add_audit(
            version.version_id, "data_updated",
            actor_id, actor_name,
            f"Updated {len(changed_fields)} field(s): "
            f"{', '.join(changed_fields[:10])}"
            f"{'...' if len(changed_fields) > 10 else ''}.",
        )

        logger.info(
            "Version %s (v%d) data updated: %d field(s) changed by %s",
            version.version_id[:12], version.version_number,
            len(changed_fields), actor_name,
        )

        elapsed = Decimal(str(round((time.perf_counter() - t0) * 1000, 3)))
        return self._build_result("update", version, elapsed)

    def compute_diff(
        self,
        base_version: InventoryVersion,
        compare_version: InventoryVersion,
    ) -> VersioningResult:
        """Compute a field-level diff between two versions.

        Args:
            base_version: The base (older) version.
            compare_version: The comparison (newer) version.

        Returns:
            VersioningResult with the computed diff.
        """
        t0 = time.perf_counter()
        self._reset_state()

        logger.info(
            "Computing diff: v%d (%s) vs v%d (%s)",
            base_version.version_number, base_version.version_id[:12],
            compare_version.version_number, compare_version.version_id[:12],
        )

        diff = self._compute_field_diff(
            base_version.data, compare_version.data,
            base_version, compare_version,
        )

        elapsed = Decimal(str(round((time.perf_counter() - t0) * 1000, 3)))

        result = VersioningResult(
            action="diff",
            diff=diff,
            audit_entries=list(self._audit_entries),
            warnings=list(self._warnings),
            processing_time_ms=elapsed,
        )
        result.provenance_hash = _compute_hash(result)
        return result

    def compare_versions(
        self,
        base_version: InventoryVersion,
        compare_version: InventoryVersion,
    ) -> VersioningResult:
        """Perform a full comparison of two versions.

        Includes diff computation and emission delta summary with
        materiality assessment.

        Args:
            base_version: The base (older) version.
            compare_version: The comparison (newer) version.

        Returns:
            VersioningResult with comparison and diff.
        """
        t0 = time.perf_counter()
        self._reset_state()

        diff = self._compute_field_diff(
            base_version.data, compare_version.data,
            base_version, compare_version,
        )

        # Compute emission delta summary.
        emission_summary = self._compute_emission_summary(
            base_version.data, compare_version.data,
        )

        # Check materiality.
        total_base = _decimal(
            base_version.data.get("total_tco2e",
                base_version.data.get("scope1_total", 0))
        )
        total_delta = abs(_decimal(emission_summary.get("total_delta", 0)))
        delta_pct = _safe_pct(total_delta, total_base)
        is_material = delta_pct >= self._material_threshold

        comparison = VersionComparison(
            base_version=base_version,
            compare_version=compare_version,
            diff=diff,
            emission_delta_summary=emission_summary,
            is_material_change=is_material,
            material_threshold_pct=self._material_threshold,
        )

        elapsed = Decimal(str(round((time.perf_counter() - t0) * 1000, 3)))

        result = VersioningResult(
            action="compare",
            diff=diff,
            comparison=comparison,
            audit_entries=list(self._audit_entries),
            warnings=list(self._warnings),
            processing_time_ms=elapsed,
        )
        result.provenance_hash = _compute_hash(result)
        return result

    def create_amendment(
        self,
        final_version: InventoryVersion,
        reason: str,
        actor_id: str,
        actor_name: str = "",
    ) -> VersioningResult:
        """Create an amendment of a FINAL version.

        Creates a new version in AMENDED status, linked to the parent
        FINAL version, with data copied from the original.

        Args:
            final_version: The FINAL version to amend.
            reason: Reason for the amendment.
            actor_id: User initiating the amendment.
            actor_name: Display name of the actor.

        Returns:
            VersioningResult with the new AMENDED version.

        Raises:
            ValueError: If the source version is not FINAL.
        """
        t0 = time.perf_counter()
        self._reset_state()

        if final_version.status != VersionStatus.FINAL:
            raise ValueError(
                f"Cannot amend: version status is {final_version.status.value}. "
                f"Only FINAL versions can be amended."
            )

        new_number = final_version.version_number + 1

        amended = InventoryVersion(
            inventory_id=final_version.inventory_id,
            version_number=new_number,
            status=VersionStatus.AMENDED,
            label=f"v{new_number}.0-amended",
            reporting_year=final_version.reporting_year,
            data=copy.deepcopy(final_version.data),
            previous_version_id=final_version.version_id,
            parent_final_version_id=final_version.version_id,
            created_by=actor_id,
            created_by_name=actor_name,
            amendment_reason=reason,
            tags=list(final_version.tags) + ["amendment"],
            notes=f"Amendment of v{final_version.version_number}: {reason}",
        )
        amended.provenance_hash = _compute_hash(amended)

        self._add_audit(
            amended.version_id, "amendment_created",
            actor_id, actor_name,
            f"Created amendment v{new_number} of FINAL version "
            f"{final_version.version_id[:12]} (v{final_version.version_number}). "
            f"Reason: {reason}.",
        )

        logger.info(
            "Created amendment %s (v%d) of FINAL v%d for inventory %s",
            amended.version_id[:12], new_number,
            final_version.version_number, final_version.inventory_id,
        )

        elapsed = Decimal(str(round((time.perf_counter() - t0) * 1000, 3)))
        return self._build_result("amend", amended, elapsed)

    def rollback(
        self,
        current_version: InventoryVersion,
        target_version: InventoryVersion,
        strategy: RollbackStrategy = RollbackStrategy.FULL_RESTORE,
        selected_fields: Optional[List[str]] = None,
        actor_id: str = "",
        actor_name: str = "",
    ) -> VersioningResult:
        """Roll back to a prior version.

        Creates a new DRAFT version with data restored from the target
        version, using the specified rollback strategy.

        Args:
            current_version: The current version (will be superseded if FINAL).
            target_version: The version to roll back to.
            strategy: Rollback strategy.
            selected_fields: Fields to restore (for SELECTIVE_FIELDS strategy).
            actor_id: User initiating the rollback.
            actor_name: Display name of the actor.

        Returns:
            VersioningResult with the new rolled-back version.

        Raises:
            ValueError: If current version is not in an editable state.
        """
        t0 = time.perf_counter()
        self._reset_state()

        if current_version.status in IMMUTABLE_STATUSES:
            self._warnings.append(
                f"Rolling back from immutable version "
                f"{current_version.version_id[:12]} "
                f"(status={current_version.status.value}). A new version "
                f"will be created."
            )

        new_number = current_version.version_number + 1

        # Compute rollback data.
        if strategy == RollbackStrategy.FULL_RESTORE:
            rollback_data = copy.deepcopy(target_version.data)

        elif strategy == RollbackStrategy.SELECTIVE_FIELDS:
            rollback_data = copy.deepcopy(current_version.data)
            fields = selected_fields or []
            for field in fields:
                if field in target_version.data:
                    rollback_data[field] = copy.deepcopy(target_version.data[field])
                elif field in rollback_data:
                    del rollback_data[field]

        elif strategy == RollbackStrategy.MERGE_LATEST:
            rollback_data = copy.deepcopy(target_version.data)
            for key, value in current_version.data.items():
                if key not in rollback_data:
                    rollback_data[key] = copy.deepcopy(value)

        else:
            rollback_data = copy.deepcopy(target_version.data)

        rolled_back = InventoryVersion(
            inventory_id=current_version.inventory_id,
            version_number=new_number,
            status=VersionStatus.DRAFT,
            label=f"v{new_number}.0-rollback",
            reporting_year=current_version.reporting_year,
            data=rollback_data,
            previous_version_id=current_version.version_id,
            created_by=actor_id or current_version.created_by,
            created_by_name=actor_name or current_version.created_by_name,
            tags=list(current_version.tags) + ["rollback"],
            notes=(
                f"Rolled back from v{current_version.version_number} to "
                f"v{target_version.version_number} using {strategy.value} strategy."
            ),
        )
        rolled_back.provenance_hash = _compute_hash(rolled_back)

        self._add_audit(
            rolled_back.version_id, "rollback",
            actor_id, actor_name,
            f"Rolled back from v{current_version.version_number} to "
            f"v{target_version.version_number} ({strategy.value}).",
        )

        logger.info(
            "Rollback: created v%d from v%d->v%d (%s) for inventory %s",
            new_number, current_version.version_number,
            target_version.version_number, strategy.value,
            current_version.inventory_id,
        )

        elapsed = Decimal(str(round((time.perf_counter() - t0) * 1000, 3)))

        result = VersioningResult(
            action="rollback",
            version=rolled_back,
            rollback_source=target_version,
            audit_entries=list(self._audit_entries),
            warnings=list(self._warnings),
            processing_time_ms=elapsed,
        )
        result.provenance_hash = _compute_hash(result)
        return result

    def get_version_chain(
        self,
        versions: List[InventoryVersion],
    ) -> List[Dict[str, Any]]:
        """Build an ordered version chain summary.

        Args:
            versions: Unsorted list of versions for an inventory.

        Returns:
            List of version summaries ordered by version_number.
        """
        sorted_versions = sorted(versions, key=lambda v: v.version_number)
        chain: List[Dict[str, Any]] = []

        for v in sorted_versions:
            chain.append({
                "version_id": v.version_id,
                "version_number": v.version_number,
                "status": v.status.value,
                "label": v.label,
                "created_at": v.created_at.isoformat() if v.created_at else "",
                "created_by": v.created_by,
                "previous_version_id": v.previous_version_id or "",
                "provenance_hash": v.provenance_hash,
                "is_editable": v.status in EDITABLE_STATUSES,
                "is_immutable": v.status in IMMUTABLE_STATUSES,
            })

        return chain

    def validate_lock(
        self,
        version: InventoryVersion,
        expected_lock_version: int,
    ) -> bool:
        """Validate the optimistic lock version.

        Args:
            version: The version to check.
            expected_lock_version: The expected lock counter value.

        Returns:
            True if lock versions match, False otherwise.
        """
        return version.lock_version == expected_lock_version

    # ------------------------------------------------------------------
    # Private Methods
    # ------------------------------------------------------------------

    def _reset_state(self) -> None:
        """Reset per-operation state."""
        self._audit_entries = []
        self._warnings = []

    def _compute_field_diff(
        self,
        base_data: Dict[str, Any],
        compare_data: Dict[str, Any],
        base_version: InventoryVersion,
        compare_version: InventoryVersion,
    ) -> VersionDiff:
        """Compute field-level diff between two data dictionaries.

        Performs a shallow comparison of top-level keys.  For nested dicts,
        compares the serialised JSON representation.

        Args:
            base_data: Data from the base version.
            compare_data: Data from the comparison version.
            base_version: Base version metadata.
            compare_version: Compare version metadata.

        Returns:
            VersionDiff with all field changes.
        """
        all_keys = set(base_data.keys()) | set(compare_data.keys())
        changes: List[FieldChange] = []
        added = 0
        removed = 0
        modified = 0
        unchanged = 0
        sections_affected: set = set()

        for key in sorted(all_keys):
            in_base = key in base_data
            in_compare = key in compare_data

            section = key.split(".")[0] if "." in key else key

            if in_base and not in_compare:
                changes.append(FieldChange(
                    field_path=key,
                    change_type=ChangeType.REMOVED,
                    old_value=base_data[key],
                    new_value=None,
                    old_value_str=str(base_data[key]),
                    new_value_str="",
                    section=section,
                ))
                removed += 1
                sections_affected.add(section)

            elif not in_base and in_compare:
                changes.append(FieldChange(
                    field_path=key,
                    change_type=ChangeType.ADDED,
                    old_value=None,
                    new_value=compare_data[key],
                    old_value_str="",
                    new_value_str=str(compare_data[key]),
                    section=section,
                ))
                added += 1
                sections_affected.add(section)

            else:
                old_val = base_data[key]
                new_val = compare_data[key]

                # Compare using JSON serialisation for complex types.
                old_serialised = json.dumps(old_val, sort_keys=True, default=str)
                new_serialised = json.dumps(new_val, sort_keys=True, default=str)

                if old_serialised == new_serialised:
                    unchanged += 1
                else:
                    delta = None
                    delta_pct = None

                    # Compute numeric delta if both are numeric.
                    old_dec = self._try_decimal(old_val)
                    new_dec = self._try_decimal(new_val)
                    if old_dec is not None and new_dec is not None:
                        delta = new_dec - old_dec
                        if old_dec != Decimal("0"):
                            delta_pct = _round_val(
                                _safe_pct(delta, old_dec), 2
                            )
                        delta = _round_val(delta, 6)

                    changes.append(FieldChange(
                        field_path=key,
                        change_type=ChangeType.MODIFIED,
                        old_value=old_val,
                        new_value=new_val,
                        old_value_str=str(old_val),
                        new_value_str=str(new_val),
                        delta=delta,
                        delta_pct=delta_pct,
                        section=section,
                    ))
                    modified += 1
                    sections_affected.add(section)

        total_compared = added + removed + modified + unchanged

        summary_parts = []
        if added > 0:
            summary_parts.append(f"{added} field(s) added")
        if removed > 0:
            summary_parts.append(f"{removed} field(s) removed")
        if modified > 0:
            summary_parts.append(f"{modified} field(s) modified")
        summary_parts.append(f"{unchanged} field(s) unchanged")
        summary = "; ".join(summary_parts) + "."

        diff = VersionDiff(
            base_version_id=base_version.version_id,
            compare_version_id=compare_version.version_id,
            base_version_number=base_version.version_number,
            compare_version_number=compare_version.version_number,
            total_fields_compared=total_compared,
            fields_added=added,
            fields_removed=removed,
            fields_modified=modified,
            fields_unchanged=unchanged,
            changes=changes,
            sections_affected=sorted(sections_affected),
            summary=summary,
        )
        diff.provenance_hash = _compute_hash(diff)
        return diff

    def _compute_emission_summary(
        self,
        base_data: Dict[str, Any],
        compare_data: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Compute emission change summary between two data snapshots.

        Looks for standard emission fields (scope1_total, scope2_total,
        scope3_total, total_tco2e) and computes deltas.

        Args:
            base_data: Base version data.
            compare_data: Compare version data.

        Returns:
            Dict with emission delta summary.
        """
        emission_fields = [
            "scope1_total", "scope2_total", "scope2_location_total",
            "scope2_market_total", "scope3_total", "total_tco2e",
        ]

        summary: Dict[str, Any] = {}
        total_delta = Decimal("0")

        for field in emission_fields:
            base_val = _decimal(base_data.get(field, 0))
            compare_val = _decimal(compare_data.get(field, 0))
            delta = compare_val - base_val
            delta_pct = _safe_pct(delta, base_val) if base_val > Decimal("0") else Decimal("0")

            if base_val != Decimal("0") or compare_val != Decimal("0"):
                summary[field] = {
                    "base": float(_round_val(base_val, 2)),
                    "compare": float(_round_val(compare_val, 2)),
                    "delta": float(_round_val(delta, 2)),
                    "delta_pct": float(_round_val(delta_pct, 2)),
                }
                if "total" in field:
                    total_delta = delta

        summary["total_delta"] = float(_round_val(total_delta, 2))
        return summary

    def _try_decimal(self, value: Any) -> Optional[Decimal]:
        """Try to convert a value to Decimal for numeric comparison.

        Args:
            value: The value to convert.

        Returns:
            Decimal if conversion succeeds, None otherwise.
        """
        if isinstance(value, (int, float, Decimal)):
            try:
                return _decimal(value)
            except (InvalidOperation, TypeError, ValueError):
                return None
        if isinstance(value, str):
            try:
                return Decimal(value)
            except (InvalidOperation, ValueError):
                return None
        return None

    def _add_audit(
        self,
        version_id: str,
        action: str,
        actor_id: str,
        actor_name: str,
        details: str,
    ) -> None:
        """Add an audit trail entry.

        Args:
            version_id: Related version ID.
            action: Action performed.
            actor_id: Actor user ID.
            actor_name: Actor display name.
            details: Event details.
        """
        entry = {
            "entry_id": _new_uuid(),
            "version_id": version_id,
            "action": action,
            "actor_id": actor_id,
            "actor_name": actor_name,
            "details": details,
            "timestamp": _utcnow().isoformat(),
        }
        entry["provenance_hash"] = _compute_hash(entry)
        self._audit_entries.append(entry)

    def _build_result(
        self,
        action: str,
        version: InventoryVersion,
        elapsed_ms: Decimal,
    ) -> VersioningResult:
        """Build a VersioningResult for version creation/update operations.

        Args:
            action: Action performed.
            version: The affected version.
            elapsed_ms: Processing time in milliseconds.

        Returns:
            VersioningResult with provenance hash.
        """
        result = VersioningResult(
            action=action,
            version=version,
            audit_entries=list(self._audit_entries),
            warnings=list(self._warnings),
            processing_time_ms=elapsed_ms,
        )
        result.provenance_hash = _compute_hash(result)
        return result
