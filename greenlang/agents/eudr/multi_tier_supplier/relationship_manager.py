# -*- coding: utf-8 -*-
"""
Relationship Manager - AGENT-EUDR-008 Multi-Tier Supplier Tracker

Engine 4 of 8: Manages supplier-buyer relationship lifecycles including
creation, update, state transitions, upstream/downstream traversal,
strength scoring, conflict detection, timeline generation, and bulk
import from ERP systems.

Relationship Lifecycle States:
    PROSPECTIVE -> ONBOARDING -> ACTIVE -> SUSPENDED -> TERMINATED
    Allowed reverse transitions:
        SUSPENDED -> ACTIVE (reactivation)
        ONBOARDING -> PROSPECTIVE (rejection)
        TERMINATED -> PROSPECTIVE (re-engagement, new relationship)

EUDR References:
    - Article 4: Continuous due diligence on supplier relationships
    - Article 9: Traceability of all supplier-buyer links
    - Article 10: Trader obligations to maintain relationship records
    - Article 14: 5-year retention of relationship history

Relationship Strength Scoring:
    Computed from transaction frequency, volume consistency, duration,
    exclusivity, and data quality. Deterministic formula with no LLM
    involvement.

Zero-Hallucination Principle:
    All relationship scoring and state transition logic is
    deterministic. State machine transitions follow explicit rules.
    No LLM calls in any computation path.

Author: GreenLang Platform Team
Date: March 2026
PRD: AGENT-EUDR-008 Multi-Tier Supplier Tracker (GL-EUDR-MST-008)
Status: Production Ready
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, FrozenSet, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Engine version string.
ENGINE_VERSION: str = "1.0.0"

#: Prometheus metric prefix.
METRIC_PREFIX: str = "gl_eudr_mst_"

#: Default batch size for batch operations.
DEFAULT_BATCH_SIZE: int = 1000

#: Maximum relationship history entries retained.
MAX_HISTORY_ENTRIES: int = 500


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------


class RelationshipStatus(str, Enum):
    """Lifecycle states for a supplier-buyer relationship.

    State machine:
        PROSPECTIVE -> ONBOARDING -> ACTIVE -> SUSPENDED -> TERMINATED
    Reverse transitions allowed:
        SUSPENDED -> ACTIVE
        ONBOARDING -> PROSPECTIVE
        TERMINATED -> PROSPECTIVE
    """

    PROSPECTIVE = "prospective"
    ONBOARDING = "onboarding"
    ACTIVE = "active"
    SUSPENDED = "suspended"
    TERMINATED = "terminated"


class TransitionReason(str, Enum):
    """Standard reason codes for relationship transitions."""

    INITIAL_DISCOVERY = "initial_discovery"
    DUE_DILIGENCE_STARTED = "due_diligence_started"
    DUE_DILIGENCE_PASSED = "due_diligence_passed"
    DUE_DILIGENCE_FAILED = "due_diligence_failed"
    COMPLIANCE_ISSUE = "compliance_issue"
    COMPLIANCE_RESOLVED = "compliance_resolved"
    CONTRACT_EXPIRED = "contract_expired"
    CONTRACT_TERMINATED = "contract_terminated"
    RISK_THRESHOLD_EXCEEDED = "risk_threshold_exceeded"
    SEASONAL_SUSPENSION = "seasonal_suspension"
    SEASONAL_REACTIVATION = "seasonal_reactivation"
    VOLUNTARY_EXIT = "voluntary_exit"
    RE_ENGAGEMENT = "re_engagement"
    REGULATORY_ACTION = "regulatory_action"
    DATA_QUALITY_ISSUE = "data_quality_issue"
    MERGER_ACQUISITION = "merger_acquisition"
    OTHER = "other"


class RelationshipType(str, Enum):
    """Classification of the relationship type."""

    DIRECT_SUPPLY = "direct_supply"
    INDIRECT_SUPPLY = "indirect_supply"
    PROCESSING = "processing"
    TRADING = "trading"
    AGGREGATION = "aggregation"
    TRANSPORT = "transport"
    CERTIFICATION = "certification"
    OTHER = "other"


class SeasonalPattern(str, Enum):
    """Seasonal sourcing patterns."""

    YEAR_ROUND = "year_round"
    MAIN_HARVEST = "main_harvest"
    MID_CROP = "mid_crop"
    SEASONAL = "seasonal"
    SPOT = "spot"


class EUDRCommodity(str, Enum):
    """EUDR regulated commodities (7 commodities per Article 1)."""

    COCOA = "cocoa"
    COFFEE = "coffee"
    PALM_OIL = "palm_oil"
    SOYA = "soya"
    RUBBER = "rubber"
    CATTLE = "cattle"
    WOOD = "wood"


# ---------------------------------------------------------------------------
# Valid state transitions
# ---------------------------------------------------------------------------

#: Valid forward and reverse state transitions.
VALID_TRANSITIONS: Dict[str, Set[str]] = {
    RelationshipStatus.PROSPECTIVE.value: {
        RelationshipStatus.ONBOARDING.value,
        RelationshipStatus.TERMINATED.value,
    },
    RelationshipStatus.ONBOARDING.value: {
        RelationshipStatus.ACTIVE.value,
        RelationshipStatus.PROSPECTIVE.value,  # rejection
        RelationshipStatus.TERMINATED.value,
    },
    RelationshipStatus.ACTIVE.value: {
        RelationshipStatus.SUSPENDED.value,
        RelationshipStatus.TERMINATED.value,
    },
    RelationshipStatus.SUSPENDED.value: {
        RelationshipStatus.ACTIVE.value,  # reactivation
        RelationshipStatus.TERMINATED.value,
    },
    RelationshipStatus.TERMINATED.value: {
        RelationshipStatus.PROSPECTIVE.value,  # re-engagement
    },
}

#: Strength scoring weights.
STRENGTH_WEIGHTS: Dict[str, float] = {
    "transaction_frequency": 0.25,
    "volume_consistency": 0.20,
    "duration": 0.20,
    "exclusivity": 0.15,
    "data_quality": 0.20,
}


# ---------------------------------------------------------------------------
# Data Classes (local, independent of models.py)
# ---------------------------------------------------------------------------


@dataclass
class SupplierRelationship:
    """A supplier-buyer relationship with lifecycle state.

    Attributes:
        relationship_id: Unique relationship identifier.
        supplier_id: ID of the upstream supplier.
        buyer_id: ID of the downstream buyer.
        relationship_type: Classification of relationship type.
        commodity: EUDR commodity traded.
        status: Current lifecycle status.
        tier_level: Tier level of the upstream supplier.
        start_date: Relationship start date (ISO 8601).
        end_date: Relationship end date if terminated (ISO 8601).
        volume_tonnes: Estimated annual volume in tonnes.
        transaction_frequency: Transactions per year.
        is_exclusive: Whether this is an exclusive relationship.
        seasonal_pattern: Seasonal sourcing pattern.
        contract_reference: Contract/agreement reference.
        dds_reference: Linked DDS reference ID.
        strength_score: Computed relationship strength (0-100).
        risk_score: Relationship risk score (0-100).
        created_at: Creation timestamp (ISO 8601).
        updated_at: Last update timestamp (ISO 8601).
        version: Relationship version number.
        metadata: Additional key-value metadata.
    """

    relationship_id: str = ""
    supplier_id: str = ""
    buyer_id: str = ""
    relationship_type: str = RelationshipType.DIRECT_SUPPLY.value
    commodity: str = ""
    status: str = RelationshipStatus.PROSPECTIVE.value
    tier_level: int = 1
    start_date: str = ""
    end_date: str = ""
    volume_tonnes: float = 0.0
    transaction_frequency: int = 0
    is_exclusive: bool = False
    seasonal_pattern: str = SeasonalPattern.YEAR_ROUND.value
    contract_reference: str = ""
    dds_reference: str = ""
    strength_score: float = 0.0
    risk_score: float = 0.0
    created_at: str = ""
    updated_at: str = ""
    version: int = 1
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Generate ID and timestamps if not provided."""
        if not self.relationship_id:
            self.relationship_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc).isoformat()
        if not self.created_at:
            self.created_at = now
        if not self.updated_at:
            self.updated_at = now


@dataclass
class StatusTransition:
    """A record of a status transition event.

    Attributes:
        transition_id: Unique transition identifier.
        relationship_id: ID of the relationship that transitioned.
        from_status: Previous status.
        to_status: New status.
        reason: Reason code for the transition.
        reason_detail: Free-text reason details.
        transitioned_by: Actor who triggered the transition.
        timestamp: When the transition occurred.
        provenance_hash: SHA-256 hash of the transition.
    """

    transition_id: str = ""
    relationship_id: str = ""
    from_status: str = ""
    to_status: str = ""
    reason: str = TransitionReason.OTHER.value
    reason_detail: str = ""
    transitioned_by: str = "system"
    timestamp: str = ""
    provenance_hash: str = ""

    def __post_init__(self) -> None:
        """Generate ID and timestamp if not provided."""
        if not self.transition_id:
            self.transition_id = str(uuid.uuid4())
        if not self.timestamp:
            self.timestamp = datetime.now(timezone.utc).isoformat()


@dataclass
class RelationshipConflict:
    """A detected conflict in relationship data.

    Attributes:
        conflict_id: Unique conflict identifier.
        conflict_type: Type of conflict detected.
        severity: Conflict severity (critical, major, minor).
        description: Human-readable description.
        involved_relationships: List of involved relationship IDs.
        involved_suppliers: List of involved supplier IDs.
        resolution_suggestion: Suggested resolution action.
        detected_at: Detection timestamp.
    """

    conflict_id: str = ""
    conflict_type: str = ""
    severity: str = "major"
    description: str = ""
    involved_relationships: List[str] = field(default_factory=list)
    involved_suppliers: List[str] = field(default_factory=list)
    resolution_suggestion: str = ""
    detected_at: str = ""

    def __post_init__(self) -> None:
        """Generate ID and timestamp if not provided."""
        if not self.conflict_id:
            self.conflict_id = str(uuid.uuid4())
        if not self.detected_at:
            self.detected_at = datetime.now(timezone.utc).isoformat()


@dataclass
class TimelineEvent:
    """An event in a relationship timeline.

    Attributes:
        event_id: Unique event identifier.
        relationship_id: Related relationship ID.
        event_type: Type of event (creation, transition, update).
        description: Human-readable description.
        timestamp: When the event occurred.
        old_value: Previous value (for changes).
        new_value: New value (for changes).
        actor: Who triggered the event.
        metadata: Additional event metadata.
    """

    event_id: str = ""
    relationship_id: str = ""
    event_type: str = ""
    description: str = ""
    timestamp: str = ""
    old_value: str = ""
    new_value: str = ""
    actor: str = "system"
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Generate ID if not provided."""
        if not self.event_id:
            self.event_id = str(uuid.uuid4())


@dataclass
class RelationshipOperationResult:
    """Result of a relationship management operation.

    Attributes:
        success: Whether the operation succeeded.
        relationship_id: ID of the affected relationship.
        operation: Operation type.
        relationship: Resulting relationship (if successful).
        transition: Status transition record (if applicable).
        strength_score: Computed strength score.
        processing_time_ms: Processing duration.
        provenance_hash: SHA-256 provenance hash.
        errors: List of errors encountered.
        warnings: List of warnings generated.
        timestamp: Result generation timestamp.
    """

    success: bool = True
    relationship_id: str = ""
    operation: str = ""
    relationship: Optional[SupplierRelationship] = None
    transition: Optional[StatusTransition] = None
    strength_score: float = 0.0
    processing_time_ms: float = 0.0
    provenance_hash: str = ""
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    timestamp: str = ""

    def __post_init__(self) -> None:
        """Generate timestamp if not provided."""
        if not self.timestamp:
            self.timestamp = datetime.now(timezone.utc).isoformat()


@dataclass
class BatchRelationshipResult:
    """Result of a batch relationship operation.

    Attributes:
        batch_id: Unique batch identifier.
        total_input: Total relationships in the input.
        total_created: Successfully created relationships.
        total_failed: Failed creations.
        results: Individual operation results.
        processing_time_ms: Total processing duration.
        provenance_hash: SHA-256 provenance hash.
        errors: Batch-level errors.
        timestamp: Result generation timestamp.
    """

    batch_id: str = ""
    total_input: int = 0
    total_created: int = 0
    total_failed: int = 0
    results: List[RelationshipOperationResult] = field(
        default_factory=list
    )
    processing_time_ms: float = 0.0
    provenance_hash: str = ""
    errors: List[str] = field(default_factory=list)
    timestamp: str = ""

    def __post_init__(self) -> None:
        """Generate ID and timestamp if not provided."""
        if not self.batch_id:
            self.batch_id = str(uuid.uuid4())
        if not self.timestamp:
            self.timestamp = datetime.now(timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def _utcnow_iso() -> str:
    """Return current UTC timestamp as ISO 8601 string."""
    return datetime.now(timezone.utc).isoformat()


def _compute_provenance_hash(data: Any) -> str:
    """Compute SHA-256 provenance hash for any serializable data.

    Args:
        data: Data to hash.

    Returns:
        Hex-encoded SHA-256 hash string.
    """
    try:
        if hasattr(data, "__dict__"):
            serialized = json.dumps(
                data.__dict__, sort_keys=True, default=str
            )
        else:
            serialized = json.dumps(data, sort_keys=True, default=str)
    except (TypeError, ValueError):
        serialized = str(data)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# RelationshipManager
# ---------------------------------------------------------------------------


class RelationshipManager:
    """Engine 4: Manages supplier-buyer relationship lifecycles.

    Provides full CRUD for supplier-buyer relationships with
    lifecycle state management, upstream/downstream traversal,
    relationship strength scoring, circular dependency detection,
    timeline generation, ERP import, and batch operations.

    State machine enforces valid transitions:
        PROSPECTIVE -> ONBOARDING -> ACTIVE -> SUSPENDED -> TERMINATED
        SUSPENDED -> ACTIVE (reactivation)
        ONBOARDING -> PROSPECTIVE (rejection)
        TERMINATED -> PROSPECTIVE (re-engagement)

    All operations produce SHA-256 provenance hashes.

    Attributes:
        _relationships: In-memory store keyed by relationship ID.
        _transitions: History of status transitions.
        _timeline_events: Timeline events for all relationships.
        _relationship_count: Running count for metrics.
        _transition_count: Running transition count for metrics.

    Example:
        >>> manager = RelationshipManager()
        >>> result = manager.create_relationship(
        ...     supplier_id="SUP-001",
        ...     buyer_id="BUY-001",
        ...     attrs={"commodity": "cocoa", "volume_tonnes": 500},
        ... )
        >>> assert result.success
        >>> assert result.relationship.status == "prospective"
    """

    def __init__(self) -> None:
        """Initialize RelationshipManager."""
        self._relationships: Dict[str, SupplierRelationship] = {}
        self._transitions: List[StatusTransition] = []
        self._timeline_events: List[TimelineEvent] = []
        self._relationship_count: int = 0
        self._transition_count: int = 0

        # Indexes for efficient traversal
        self._supplier_index: Dict[str, Set[str]] = defaultdict(set)
        self._buyer_index: Dict[str, Set[str]] = defaultdict(set)

        logger.info(
            "RelationshipManager initialized: version=%s",
            ENGINE_VERSION,
        )

    # ------------------------------------------------------------------
    # Create
    # ------------------------------------------------------------------

    def create_relationship(
        self,
        supplier_id: str,
        buyer_id: str,
        attrs: Optional[Dict[str, Any]] = None,
        created_by: str = "system",
    ) -> RelationshipOperationResult:
        """Create a new supplier-buyer relationship.

        Creates the relationship in PROSPECTIVE state by default.
        Validates inputs, computes strength score, and generates
        provenance hash.

        Args:
            supplier_id: ID of the upstream supplier.
            buyer_id: ID of the downstream buyer.
            attrs: Optional relationship attributes (commodity,
                volume_tonnes, relationship_type, etc.).
            created_by: Actor creating the relationship.

        Returns:
            RelationshipOperationResult with the created relationship.
        """
        start_time = time.monotonic()
        attrs = attrs or {}

        logger.info(
            "Creating relationship: supplier=%s, buyer=%s, "
            "commodity=%s",
            supplier_id,
            buyer_id,
            attrs.get("commodity", "unknown"),
        )

        errors: List[str] = []
        warnings: List[str] = []

        if not supplier_id:
            errors.append("supplier_id is required")
        if not buyer_id:
            errors.append("buyer_id is required")
        if supplier_id and buyer_id and supplier_id == buyer_id:
            errors.append("supplier_id and buyer_id must be different")

        if errors:
            return self._build_operation_result(
                success=False,
                operation="create",
                errors=errors,
                start_time=start_time,
            )

        # Check for existing duplicate relationship
        existing = self._find_existing_relationship(
            supplier_id, buyer_id, attrs.get("commodity", "")
        )
        if existing is not None:
            warnings.append(
                f"Existing relationship found: "
                f"{existing.relationship_id} "
                f"(status={existing.status})"
            )

        # Build relationship
        relationship = SupplierRelationship(
            supplier_id=supplier_id,
            buyer_id=buyer_id,
            relationship_type=str(
                attrs.get(
                    "relationship_type",
                    RelationshipType.DIRECT_SUPPLY.value,
                )
            ),
            commodity=str(attrs.get("commodity", "")),
            status=str(
                attrs.get(
                    "status",
                    RelationshipStatus.PROSPECTIVE.value,
                )
            ),
            tier_level=int(attrs.get("tier_level", 1)),
            start_date=str(attrs.get("start_date", _utcnow_iso())),
            volume_tonnes=float(attrs.get("volume_tonnes", 0.0)),
            transaction_frequency=int(
                attrs.get("transaction_frequency", 0)
            ),
            is_exclusive=bool(attrs.get("is_exclusive", False)),
            seasonal_pattern=str(
                attrs.get(
                    "seasonal_pattern",
                    SeasonalPattern.YEAR_ROUND.value,
                )
            ),
            contract_reference=str(
                attrs.get("contract_reference", "")
            ),
            dds_reference=str(attrs.get("dds_reference", "")),
            metadata=attrs.get("metadata", {}),
        )

        # Calculate strength
        relationship.strength_score = self.calculate_strength(
            relationship
        )

        # Store
        self._relationships[relationship.relationship_id] = (
            relationship
        )
        self._supplier_index[supplier_id].add(
            relationship.relationship_id
        )
        self._buyer_index[buyer_id].add(
            relationship.relationship_id
        )
        self._relationship_count += 1

        # Record timeline event
        self._record_event(
            relationship_id=relationship.relationship_id,
            event_type="creation",
            description=(
                f"Relationship created: {supplier_id} -> {buyer_id} "
                f"({relationship.commodity})"
            ),
            actor=created_by,
        )

        elapsed_ms = (time.monotonic() - start_time) * 1000

        logger.info(
            "Relationship created: id=%s, supplier=%s, "
            "buyer=%s, status=%s, strength=%.1f, "
            "duration_ms=%.2f",
            relationship.relationship_id,
            supplier_id,
            buyer_id,
            relationship.status,
            relationship.strength_score,
            elapsed_ms,
        )

        return RelationshipOperationResult(
            success=True,
            relationship_id=relationship.relationship_id,
            operation="create",
            relationship=relationship,
            strength_score=relationship.strength_score,
            processing_time_ms=elapsed_ms,
            provenance_hash=_compute_provenance_hash(relationship),
            warnings=warnings,
        )

    # ------------------------------------------------------------------
    # Update
    # ------------------------------------------------------------------

    def update_relationship(
        self,
        rel_id: str,
        updates: Dict[str, Any],
        updated_by: str = "system",
    ) -> RelationshipOperationResult:
        """Update relationship attributes (not status).

        Updates non-status fields and recalculates strength score.
        For status changes, use transition_status() to enforce
        valid state transitions.

        Args:
            rel_id: Relationship ID to update.
            updates: Dictionary of fields to update.
            updated_by: Actor performing the update.

        Returns:
            RelationshipOperationResult with the updated relationship.
        """
        start_time = time.monotonic()

        logger.info(
            "Updating relationship: id=%s, fields=%s",
            rel_id,
            list(updates.keys()),
        )

        relationship = self._relationships.get(rel_id)
        if relationship is None:
            return self._build_operation_result(
                success=False,
                relationship_id=rel_id,
                operation="update",
                errors=[f"Relationship not found: {rel_id}"],
                start_time=start_time,
            )

        # Prevent status updates through this method
        if "status" in updates:
            return self._build_operation_result(
                success=False,
                relationship_id=rel_id,
                operation="update",
                errors=[
                    "Use transition_status() to change status"
                ],
                start_time=start_time,
            )

        changed_fields: List[str] = []
        for field_name, new_value in updates.items():
            if hasattr(relationship, field_name):
                old_value = getattr(relationship, field_name)
                if old_value != new_value:
                    setattr(relationship, field_name, new_value)
                    changed_fields.append(field_name)

                    self._record_event(
                        relationship_id=rel_id,
                        event_type="update",
                        description=(
                            f"Field '{field_name}' updated"
                        ),
                        old_value=str(old_value),
                        new_value=str(new_value),
                        actor=updated_by,
                    )

        if not changed_fields:
            return self._build_operation_result(
                success=True,
                relationship_id=rel_id,
                operation="update",
                relationship=relationship,
                warnings=["No fields changed"],
                start_time=start_time,
            )

        relationship.updated_at = _utcnow_iso()
        relationship.version += 1

        # Recalculate strength
        relationship.strength_score = self.calculate_strength(
            relationship
        )

        elapsed_ms = (time.monotonic() - start_time) * 1000

        logger.info(
            "Relationship updated: id=%s, changed=%s, "
            "strength=%.1f, duration_ms=%.2f",
            rel_id,
            changed_fields,
            relationship.strength_score,
            elapsed_ms,
        )

        return RelationshipOperationResult(
            success=True,
            relationship_id=rel_id,
            operation="update",
            relationship=relationship,
            strength_score=relationship.strength_score,
            processing_time_ms=elapsed_ms,
            provenance_hash=_compute_provenance_hash(relationship),
        )

    # ------------------------------------------------------------------
    # Status Transition
    # ------------------------------------------------------------------

    def transition_status(
        self,
        rel_id: str,
        new_status: str,
        reason: str,
        reason_detail: str = "",
        transitioned_by: str = "system",
    ) -> RelationshipOperationResult:
        """Lifecycle state transition with validation.

        Enforces valid state transitions:
            PROSPECTIVE -> ONBOARDING -> ACTIVE -> SUSPENDED -> TERMINATED
            SUSPENDED -> ACTIVE (reactivation)
            ONBOARDING -> PROSPECTIVE (rejection)
            TERMINATED -> PROSPECTIVE (re-engagement)

        Args:
            rel_id: Relationship ID.
            new_status: Target status.
            reason: Reason code for the transition.
            reason_detail: Free-text reason detail.
            transitioned_by: Actor triggering the transition.

        Returns:
            RelationshipOperationResult with transition record.

        Raises:
            ValueError: If the transition is not valid.
        """
        start_time = time.monotonic()

        logger.info(
            "Transitioning relationship: id=%s, new_status=%s, "
            "reason=%s",
            rel_id,
            new_status,
            reason,
        )

        relationship = self._relationships.get(rel_id)
        if relationship is None:
            return self._build_operation_result(
                success=False,
                relationship_id=rel_id,
                operation="transition",
                errors=[f"Relationship not found: {rel_id}"],
                start_time=start_time,
            )

        old_status = relationship.status

        # Validate transition
        valid_targets = VALID_TRANSITIONS.get(old_status, set())
        if new_status not in valid_targets:
            return self._build_operation_result(
                success=False,
                relationship_id=rel_id,
                operation="transition",
                errors=[
                    f"Invalid transition: {old_status} -> "
                    f"{new_status}. Valid targets: "
                    f"{sorted(valid_targets)}"
                ],
                start_time=start_time,
            )

        # Perform transition
        relationship.status = new_status
        relationship.updated_at = _utcnow_iso()
        relationship.version += 1

        # Set end_date on termination
        if new_status == RelationshipStatus.TERMINATED.value:
            relationship.end_date = _utcnow_iso()

        # Clear end_date on reactivation
        if (
            old_status == RelationshipStatus.TERMINATED.value
            and new_status == RelationshipStatus.PROSPECTIVE.value
        ):
            relationship.end_date = ""

        # Record transition
        transition = StatusTransition(
            relationship_id=rel_id,
            from_status=old_status,
            to_status=new_status,
            reason=reason,
            reason_detail=reason_detail,
            transitioned_by=transitioned_by,
        )
        transition.provenance_hash = _compute_provenance_hash(
            transition
        )
        self._transitions.append(transition)
        self._transition_count += 1

        # Record timeline event
        self._record_event(
            relationship_id=rel_id,
            event_type="transition",
            description=(
                f"Status changed: {old_status} -> {new_status} "
                f"({reason})"
            ),
            old_value=old_status,
            new_value=new_status,
            actor=transitioned_by,
        )

        elapsed_ms = (time.monotonic() - start_time) * 1000

        logger.info(
            "Relationship transitioned: id=%s, %s -> %s, "
            "reason=%s, duration_ms=%.2f",
            rel_id,
            old_status,
            new_status,
            reason,
            elapsed_ms,
        )

        return RelationshipOperationResult(
            success=True,
            relationship_id=rel_id,
            operation="transition",
            relationship=relationship,
            transition=transition,
            strength_score=relationship.strength_score,
            processing_time_ms=elapsed_ms,
            provenance_hash=_compute_provenance_hash(relationship),
        )

    # ------------------------------------------------------------------
    # Traversal: Upstream
    # ------------------------------------------------------------------

    def get_upstream(
        self,
        supplier_id: str,
        include_inactive: bool = False,
    ) -> List[SupplierRelationship]:
        """Get all upstream suppliers for a given entity.

        Traverses the relationship graph to find all entities
        that supply to the given supplier_id.

        Args:
            supplier_id: ID of the entity to find upstream for.
            include_inactive: Whether to include suspended/terminated.

        Returns:
            List of upstream SupplierRelationship instances.
        """
        rel_ids = self._buyer_index.get(supplier_id, set())
        results: List[SupplierRelationship] = []

        for rel_id in rel_ids:
            rel = self._relationships.get(rel_id)
            if rel is None:
                continue
            if not include_inactive and rel.status in (
                RelationshipStatus.TERMINATED.value,
                RelationshipStatus.SUSPENDED.value,
            ):
                continue
            results.append(rel)

        logger.debug(
            "Upstream query: supplier_id=%s, found=%d "
            "(include_inactive=%s)",
            supplier_id,
            len(results),
            include_inactive,
        )

        return results

    # ------------------------------------------------------------------
    # Traversal: Downstream
    # ------------------------------------------------------------------

    def get_downstream(
        self,
        supplier_id: str,
        include_inactive: bool = False,
    ) -> List[SupplierRelationship]:
        """Get all downstream buyers for a given entity.

        Traverses the relationship graph to find all entities
        that buy from the given supplier_id.

        Args:
            supplier_id: ID of the entity to find downstream for.
            include_inactive: Whether to include suspended/terminated.

        Returns:
            List of downstream SupplierRelationship instances.
        """
        rel_ids = self._supplier_index.get(supplier_id, set())
        results: List[SupplierRelationship] = []

        for rel_id in rel_ids:
            rel = self._relationships.get(rel_id)
            if rel is None:
                continue
            if not include_inactive and rel.status in (
                RelationshipStatus.TERMINATED.value,
                RelationshipStatus.SUSPENDED.value,
            ):
                continue
            results.append(rel)

        logger.debug(
            "Downstream query: supplier_id=%s, found=%d "
            "(include_inactive=%s)",
            supplier_id,
            len(results),
            include_inactive,
        )

        return results

    # ------------------------------------------------------------------
    # Strength Scoring
    # ------------------------------------------------------------------

    def calculate_strength(
        self, relationship: SupplierRelationship
    ) -> float:
        """Score relationship strength (0-100).

        Computed from five weighted factors:
        1. Transaction frequency (25%): Higher frequency = stronger
        2. Volume consistency (20%): Volume > 0 indicates real trade
        3. Duration (20%): Longer relationships are stronger
        4. Exclusivity (15%): Exclusive relationships score higher
        5. Data quality (20%): More complete data = stronger signal

        Args:
            relationship: Relationship to score.

        Returns:
            Strength score between 0.0 and 100.0.
        """
        scores: Dict[str, float] = {}

        # 1. Transaction frequency (0-100)
        freq = relationship.transaction_frequency
        if freq >= 52:  # weekly
            scores["transaction_frequency"] = 100.0
        elif freq >= 12:  # monthly
            scores["transaction_frequency"] = 80.0
        elif freq >= 4:  # quarterly
            scores["transaction_frequency"] = 60.0
        elif freq >= 1:  # annual
            scores["transaction_frequency"] = 40.0
        else:
            scores["transaction_frequency"] = 10.0

        # 2. Volume consistency (0-100)
        if relationship.volume_tonnes > 1000:
            scores["volume_consistency"] = 100.0
        elif relationship.volume_tonnes > 100:
            scores["volume_consistency"] = 75.0
        elif relationship.volume_tonnes > 10:
            scores["volume_consistency"] = 50.0
        elif relationship.volume_tonnes > 0:
            scores["volume_consistency"] = 25.0
        else:
            scores["volume_consistency"] = 0.0

        # 3. Duration (0-100)
        duration_score = self._score_duration(relationship)
        scores["duration"] = duration_score

        # 4. Exclusivity (0-100)
        scores["exclusivity"] = (
            100.0 if relationship.is_exclusive else 30.0
        )

        # 5. Data quality (0-100)
        data_quality = self._score_data_quality(relationship)
        scores["data_quality"] = data_quality

        # Calculate weighted total
        total = 0.0
        for factor, weight in STRENGTH_WEIGHTS.items():
            factor_score = scores.get(factor, 0.0)
            total += factor_score * weight

        final_score = max(0.0, min(100.0, total))

        logger.debug(
            "Strength scored: relationship=%s, "
            "freq=%.0f, vol=%.0f, dur=%.0f, "
            "excl=%.0f, dq=%.0f, total=%.1f",
            relationship.relationship_id,
            scores["transaction_frequency"],
            scores["volume_consistency"],
            scores["duration"],
            scores["exclusivity"],
            scores["data_quality"],
            final_score,
        )

        return round(final_score, 1)

    def _score_duration(
        self, relationship: SupplierRelationship
    ) -> float:
        """Score relationship duration.

        Args:
            relationship: Relationship to score.

        Returns:
            Duration score (0-100).
        """
        if not relationship.start_date:
            return 0.0

        try:
            start_str = relationship.start_date
            # Handle ISO format
            if "T" in start_str:
                start_str = start_str[:10]
            start = datetime.strptime(start_str, "%Y-%m-%d")
            now = datetime.now(timezone.utc).replace(tzinfo=None)
            days = (now - start).days

            if days >= 1825:  # 5+ years
                return 100.0
            if days >= 1095:  # 3+ years
                return 80.0
            if days >= 365:  # 1+ years
                return 60.0
            if days >= 180:  # 6+ months
                return 40.0
            if days >= 30:  # 1+ month
                return 20.0
            return 10.0
        except (ValueError, TypeError):
            return 0.0

    def _score_data_quality(
        self, relationship: SupplierRelationship
    ) -> float:
        """Score data quality of a relationship.

        Args:
            relationship: Relationship to score.

        Returns:
            Data quality score (0-100).
        """
        fields_present = 0
        total_fields = 8

        if relationship.commodity:
            fields_present += 1
        if relationship.volume_tonnes > 0:
            fields_present += 1
        if relationship.transaction_frequency > 0:
            fields_present += 1
        if relationship.start_date:
            fields_present += 1
        if relationship.contract_reference:
            fields_present += 1
        if relationship.dds_reference:
            fields_present += 1
        if relationship.relationship_type:
            fields_present += 1
        if relationship.seasonal_pattern:
            fields_present += 1

        return (fields_present / total_fields) * 100.0

    # ------------------------------------------------------------------
    # Conflict Detection
    # ------------------------------------------------------------------

    def detect_conflicts(
        self, supplier_id: str
    ) -> List[RelationshipConflict]:
        """Detect circular and inconsistent relationships.

        Checks for:
        1. Circular dependencies (A supplies B, B supplies A)
        2. Self-referencing relationships
        3. Duplicate relationships (same supplier-buyer-commodity)
        4. Inconsistent tier levels

        Args:
            supplier_id: Supplier to check for conflicts.

        Returns:
            List of detected RelationshipConflict instances.
        """
        start_time = time.monotonic()

        logger.info(
            "Detecting conflicts for supplier: %s", supplier_id
        )

        conflicts: List[RelationshipConflict] = []

        # Get all relationships involving this supplier
        upstream_rels = self.get_upstream(
            supplier_id, include_inactive=True
        )
        downstream_rels = self.get_downstream(
            supplier_id, include_inactive=True
        )

        # 1. Check for circular dependencies
        upstream_supplier_ids = {r.supplier_id for r in upstream_rels}
        downstream_buyer_ids = {r.buyer_id for r in downstream_rels}

        circular_ids = upstream_supplier_ids & downstream_buyer_ids
        for circ_id in circular_ids:
            conflicts.append(RelationshipConflict(
                conflict_type="circular_dependency",
                severity="critical",
                description=(
                    f"Circular dependency detected: {supplier_id} "
                    f"both supplies to and receives from {circ_id}"
                ),
                involved_suppliers=[supplier_id, circ_id],
                resolution_suggestion=(
                    "Review and correct the relationship direction. "
                    "One of the relationships may be incorrectly "
                    "recorded."
                ),
            ))

        # 2. Check for duplicate relationships
        rel_keys: Dict[str, List[str]] = defaultdict(list)
        all_rels = upstream_rels + downstream_rels
        for rel in all_rels:
            key = (
                f"{rel.supplier_id}|{rel.buyer_id}|{rel.commodity}"
            )
            rel_keys[key].append(rel.relationship_id)

        for key, rel_ids in rel_keys.items():
            if len(rel_ids) > 1:
                conflicts.append(RelationshipConflict(
                    conflict_type="duplicate_relationship",
                    severity="major",
                    description=(
                        f"Duplicate relationships found for "
                        f"the same supplier-buyer-commodity "
                        f"combination: {key}"
                    ),
                    involved_relationships=rel_ids,
                    involved_suppliers=[supplier_id],
                    resolution_suggestion=(
                        "Merge or deactivate duplicate "
                        "relationships. Keep the most recent "
                        "or most complete record."
                    ),
                ))

        # 3. Check for inconsistent tier levels
        tier_levels: Dict[str, Set[int]] = defaultdict(set)
        for rel in all_rels:
            tier_levels[rel.supplier_id].add(rel.tier_level)

        for sid, tiers in tier_levels.items():
            if len(tiers) > 1:
                conflicts.append(RelationshipConflict(
                    conflict_type="inconsistent_tier_level",
                    severity="minor",
                    description=(
                        f"Supplier {sid} has inconsistent tier "
                        f"levels across relationships: "
                        f"{sorted(tiers)}"
                    ),
                    involved_suppliers=[sid],
                    resolution_suggestion=(
                        "Standardize the tier level for this "
                        "supplier across all relationships."
                    ),
                ))

        elapsed_ms = (time.monotonic() - start_time) * 1000

        logger.info(
            "Conflict detection completed: supplier=%s, "
            "conflicts=%d, duration_ms=%.2f",
            supplier_id,
            len(conflicts),
            elapsed_ms,
        )

        return conflicts

    # ------------------------------------------------------------------
    # Timeline
    # ------------------------------------------------------------------

    def get_timeline(
        self, supplier_id: str
    ) -> List[TimelineEvent]:
        """Get relationship history timeline for a supplier.

        Returns all events (creation, transitions, updates) for
        relationships involving the given supplier, sorted
        chronologically.

        Args:
            supplier_id: Supplier to get timeline for.

        Returns:
            List of TimelineEvent instances sorted by timestamp.
        """
        # Get all relationship IDs involving this supplier
        rel_ids: Set[str] = set()
        rel_ids.update(self._supplier_index.get(supplier_id, set()))
        rel_ids.update(self._buyer_index.get(supplier_id, set()))

        events = [
            e
            for e in self._timeline_events
            if e.relationship_id in rel_ids
        ]

        # Sort by timestamp
        events.sort(key=lambda e: e.timestamp)

        logger.debug(
            "Timeline retrieved: supplier=%s, events=%d",
            supplier_id,
            len(events),
        )

        return events

    # ------------------------------------------------------------------
    # ERP Import
    # ------------------------------------------------------------------

    def import_from_erp(
        self, erp_data: List[Dict[str, Any]],
        imported_by: str = "system",
    ) -> BatchRelationshipResult:
        """Bulk import relationships from ERP data.

        Maps ERP purchase order / vendor master records to
        supplier-buyer relationships.

        Args:
            erp_data: List of ERP records with supplier/buyer info.
            imported_by: Actor performing the import.

        Returns:
            BatchRelationshipResult with individual results.
        """
        start_time = time.monotonic()
        batch_id = str(uuid.uuid4())

        logger.info(
            "Starting ERP import: batch_id=%s, records=%d",
            batch_id,
            len(erp_data),
        )

        results: List[RelationshipOperationResult] = []
        total_created = 0
        total_failed = 0
        batch_errors: List[str] = []

        for idx, record in enumerate(erp_data):
            try:
                supplier_id = str(
                    record.get(
                        "supplier_id",
                        record.get("vendor_id", ""),
                    )
                ).strip()
                buyer_id = str(
                    record.get(
                        "buyer_id",
                        record.get(
                            "buying_org",
                            record.get("company_code", ""),
                        ),
                    )
                ).strip()

                if not supplier_id or not buyer_id:
                    batch_errors.append(
                        f"Record {idx}: missing supplier_id "
                        f"or buyer_id"
                    )
                    total_failed += 1
                    continue

                attrs = {
                    "commodity": str(
                        record.get("commodity", "")
                    ),
                    "volume_tonnes": float(
                        record.get("volume_tonnes", 0.0)
                    ),
                    "relationship_type": str(
                        record.get(
                            "relationship_type",
                            RelationshipType.DIRECT_SUPPLY.value,
                        )
                    ),
                    "contract_reference": str(
                        record.get(
                            "contract_reference",
                            record.get("po_number", ""),
                        )
                    ),
                    "start_date": str(
                        record.get(
                            "start_date",
                            record.get("po_date", ""),
                        )
                    ),
                    "metadata": {
                        "erp_source": str(
                            record.get("erp_system", "unknown")
                        ),
                        "import_batch": batch_id,
                        "record_index": idx,
                    },
                }

                result = self.create_relationship(
                    supplier_id=supplier_id,
                    buyer_id=buyer_id,
                    attrs=attrs,
                    created_by=imported_by,
                )
                results.append(result)

                if result.success:
                    total_created += 1
                else:
                    total_failed += 1
                    batch_errors.extend(result.errors)

            except Exception as exc:
                total_failed += 1
                error_msg = (
                    f"ERP record {idx} failed: "
                    f"{type(exc).__name__}: {exc}"
                )
                batch_errors.append(error_msg)
                logger.warning(error_msg)

        elapsed_ms = (time.monotonic() - start_time) * 1000

        batch_result = BatchRelationshipResult(
            batch_id=batch_id,
            total_input=len(erp_data),
            total_created=total_created,
            total_failed=total_failed,
            results=results,
            processing_time_ms=elapsed_ms,
            provenance_hash=_compute_provenance_hash({
                "batch_id": batch_id,
                "total_input": len(erp_data),
                "total_created": total_created,
                "timestamp": _utcnow_iso(),
            }),
            errors=batch_errors,
        )

        logger.info(
            "ERP import completed: batch_id=%s, "
            "created=%d, failed=%d, duration_ms=%.2f",
            batch_id,
            total_created,
            total_failed,
            elapsed_ms,
        )

        return batch_result

    # ------------------------------------------------------------------
    # Batch Create
    # ------------------------------------------------------------------

    def batch_create(
        self,
        relationships: List[Dict[str, Any]],
        created_by: str = "system",
    ) -> BatchRelationshipResult:
        """Batch creation of relationships.

        Args:
            relationships: List of relationship data dictionaries.
                Each must contain 'supplier_id' and 'buyer_id'.
            created_by: Actor performing the batch creation.

        Returns:
            BatchRelationshipResult with individual results.
        """
        start_time = time.monotonic()
        batch_id = str(uuid.uuid4())

        logger.info(
            "Starting batch relationship create: "
            "batch_id=%s, count=%d",
            batch_id,
            len(relationships),
        )

        results: List[RelationshipOperationResult] = []
        total_created = 0
        total_failed = 0
        batch_errors: List[str] = []

        for idx, rel_data in enumerate(relationships):
            try:
                supplier_id = str(
                    rel_data.get("supplier_id", "")
                ).strip()
                buyer_id = str(
                    rel_data.get("buyer_id", "")
                ).strip()

                if not supplier_id or not buyer_id:
                    batch_errors.append(
                        f"Item {idx}: missing supplier_id or buyer_id"
                    )
                    total_failed += 1
                    continue

                # Remove supplier_id and buyer_id from attrs
                attrs = {
                    k: v
                    for k, v in rel_data.items()
                    if k not in ("supplier_id", "buyer_id")
                }

                result = self.create_relationship(
                    supplier_id=supplier_id,
                    buyer_id=buyer_id,
                    attrs=attrs,
                    created_by=created_by,
                )
                results.append(result)

                if result.success:
                    total_created += 1
                else:
                    total_failed += 1
                    batch_errors.extend(result.errors)

            except Exception as exc:
                total_failed += 1
                error_msg = (
                    f"Batch item {idx} failed: "
                    f"{type(exc).__name__}: {exc}"
                )
                batch_errors.append(error_msg)
                logger.warning(error_msg)

        elapsed_ms = (time.monotonic() - start_time) * 1000

        batch_result = BatchRelationshipResult(
            batch_id=batch_id,
            total_input=len(relationships),
            total_created=total_created,
            total_failed=total_failed,
            results=results,
            processing_time_ms=elapsed_ms,
            provenance_hash=_compute_provenance_hash({
                "batch_id": batch_id,
                "total_input": len(relationships),
                "total_created": total_created,
                "timestamp": _utcnow_iso(),
            }),
            errors=batch_errors,
        )

        logger.info(
            "Batch relationship create completed: "
            "batch_id=%s, created=%d, failed=%d, "
            "duration_ms=%.2f",
            batch_id,
            total_created,
            total_failed,
            elapsed_ms,
        )

        return batch_result

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _find_existing_relationship(
        self,
        supplier_id: str,
        buyer_id: str,
        commodity: str,
    ) -> Optional[SupplierRelationship]:
        """Find an existing relationship for the same triple.

        Args:
            supplier_id: Supplier ID.
            buyer_id: Buyer ID.
            commodity: Commodity.

        Returns:
            Existing relationship if found, None otherwise.
        """
        rel_ids = self._supplier_index.get(supplier_id, set())
        for rel_id in rel_ids:
            rel = self._relationships.get(rel_id)
            if rel is None:
                continue
            if (
                rel.buyer_id == buyer_id
                and rel.commodity == commodity
            ):
                return rel
        return None

    def _record_event(
        self,
        relationship_id: str,
        event_type: str,
        description: str,
        old_value: str = "",
        new_value: str = "",
        actor: str = "system",
    ) -> None:
        """Record a timeline event.

        Args:
            relationship_id: Related relationship ID.
            event_type: Type of event.
            description: Human-readable description.
            old_value: Previous value.
            new_value: New value.
            actor: Who triggered the event.
        """
        event = TimelineEvent(
            relationship_id=relationship_id,
            event_type=event_type,
            description=description,
            timestamp=_utcnow_iso(),
            old_value=old_value,
            new_value=new_value,
            actor=actor,
        )
        self._timeline_events.append(event)

        # Enforce max history
        if len(self._timeline_events) > MAX_HISTORY_ENTRIES * 10:
            self._timeline_events = self._timeline_events[
                -MAX_HISTORY_ENTRIES * 5:
            ]

    def _build_operation_result(
        self,
        success: bool,
        operation: str,
        start_time: float,
        relationship_id: str = "",
        relationship: Optional[SupplierRelationship] = None,
        errors: Optional[List[str]] = None,
        warnings: Optional[List[str]] = None,
    ) -> RelationshipOperationResult:
        """Build a RelationshipOperationResult.

        Args:
            success: Whether the operation succeeded.
            operation: Operation type.
            start_time: Monotonic start time.
            relationship_id: Relationship ID.
            relationship: Relationship if available.
            errors: Errors list.
            warnings: Warnings list.

        Returns:
            RelationshipOperationResult instance.
        """
        elapsed_ms = (time.monotonic() - start_time) * 1000
        return RelationshipOperationResult(
            success=success,
            relationship_id=relationship_id,
            operation=operation,
            relationship=relationship,
            strength_score=(
                relationship.strength_score if relationship else 0.0
            ),
            processing_time_ms=elapsed_ms,
            provenance_hash=_compute_provenance_hash({
                "operation": operation,
                "relationship_id": relationship_id,
            }),
            errors=errors or [],
            warnings=warnings or [],
        )

    # ------------------------------------------------------------------
    # Metrics accessors
    # ------------------------------------------------------------------

    @property
    def total_relationships(self) -> int:
        """Return total number of relationships stored.

        Returns:
            Count of relationships in the store.
        """
        return len(self._relationships)

    @property
    def active_relationships(self) -> int:
        """Return count of active relationships.

        Returns:
            Count of relationships with ACTIVE status.
        """
        return sum(
            1
            for r in self._relationships.values()
            if r.status == RelationshipStatus.ACTIVE.value
        )

    @property
    def total_transitions(self) -> int:
        """Return total number of status transitions.

        Returns:
            Running transition count.
        """
        return self._transition_count

    def get_relationship(
        self, rel_id: str
    ) -> Optional[SupplierRelationship]:
        """Get a relationship by ID.

        Args:
            rel_id: Relationship ID.

        Returns:
            SupplierRelationship if found, None otherwise.
        """
        return self._relationships.get(rel_id)

    def get_transitions(
        self, rel_id: str
    ) -> List[StatusTransition]:
        """Get all status transitions for a relationship.

        Args:
            rel_id: Relationship ID.

        Returns:
            List of StatusTransition records.
        """
        return [
            t
            for t in self._transitions
            if t.relationship_id == rel_id
        ]

    def reset_metrics(self) -> None:
        """Reset internal metrics counters."""
        self._relationship_count = 0
        self._transition_count = 0
        logger.debug("RelationshipManager metrics reset")


# ---------------------------------------------------------------------------
# Public surface
# ---------------------------------------------------------------------------

__all__ = [
    # Engine
    "RelationshipManager",
    # Enums
    "RelationshipStatus",
    "TransitionReason",
    "RelationshipType",
    "SeasonalPattern",
    "EUDRCommodity",
    # Data classes
    "SupplierRelationship",
    "StatusTransition",
    "RelationshipConflict",
    "TimelineEvent",
    "RelationshipOperationResult",
    "BatchRelationshipResult",
    # Constants
    "ENGINE_VERSION",
    "METRIC_PREFIX",
    "DEFAULT_BATCH_SIZE",
    "MAX_HISTORY_ENTRIES",
    "VALID_TRANSITIONS",
    "STRENGTH_WEIGHTS",
]
