"""
PACK-050 GHG Consolidation Pack - Consolidation Adjustment Engine
====================================================================

Records, manages and tracks manual adjustments, reclassifications,
and corrections applied during the GHG consolidation process.
Provides a complete audit trail with approval workflow, impact
calculation, and reversal capability.

Regulatory Basis:
    - GHG Protocol Corporate Standard (Chapter 7): Managing inventory
      quality, including corrections and adjustments.
    - ISO 14064-1:2018 (Clause 5.4): Recalculation and adjustments
      to GHG inventories.
    - ESRS E1-6: Requires disclosure of material restatements and
      corrections.
    - ISAE 3410: Assurance engagement requirements for verifying
      adjustments and corrections to GHG statements.

Adjustment Categories:
    - METHODOLOGY_CHANGE: Change in calculation methodology.
    - ERROR_CORRECTION: Correction of a data or calculation error.
    - SCOPE_RECLASSIFICATION: Reclassification between scopes.
    - TIMING_ADJUSTMENT: Timing cutoff adjustment for period end.
    - LATE_SUBMISSION: Late data from an entity.
    - DATA_QUALITY: Improvement to data quality / source change.
    - BOUNDARY_CHANGE: Inclusion/exclusion of entities.
    - OTHER: Catch-all for other adjustment types.

Capabilities:
    - Record adjustments with full categorisation and justification
    - Approval workflow (DRAFT -> SUBMITTED -> REVIEWED -> APPROVED/REJECTED)
    - Impact calculation (before vs after for each adjustment)
    - Adjustment reversal with audit trail
    - Batch adjustments for multiple entities
    - Adjustment summary and reporting

Zero-Hallucination:
    - All calculations use deterministic Decimal arithmetic
    - No LLM involvement in any calculation path
    - SHA-256 provenance hash on every result object

Author:  GreenLang Platform Team
Date:    March 2026
Pack:    PACK-050 GHG Consolidation
Engine:  8 of 10
Status:  Production Ready
"""

from __future__ import annotations

import hashlib
import json
import logging
import uuid
from datetime import datetime, date, timezone
from decimal import Decimal, ROUND_HALF_UP, InvalidOperation
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from greenlang.schemas import utcnow

logger = logging.getLogger(__name__)
_MODULE_VERSION = "1.0.0"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _new_uuid() -> str:
    """Generate a new UUID4 string."""
    return str(uuid.uuid4())

def _compute_hash(data: Any) -> str:
    """Compute SHA-256 provenance hash, excluding volatile fields."""
    if hasattr(data, "model_dump"):
        serializable = data.model_dump(mode="json")
    elif isinstance(data, dict):
        serializable = data
    else:
        serializable = str(data)
    if isinstance(serializable, dict):
        serializable = {
            k: v for k, v in serializable.items()
            if k not in ("created_at", "updated_at", "provenance_hash")
        }
    raw = json.dumps(serializable, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()

def _decimal(value: Any) -> Decimal:
    """Safely convert any value to Decimal."""
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
    """Divide safely, returning *default* when denominator is zero."""
    if denominator == Decimal("0"):
        return default
    return numerator / denominator

def _round2(value: Any) -> Decimal:
    """Round a value to two decimal places using ROUND_HALF_UP."""
    return Decimal(str(value)).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)

# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class AdjustmentCategory(str, Enum):
    """Categories of consolidation adjustments."""
    METHODOLOGY_CHANGE = "METHODOLOGY_CHANGE"
    ERROR_CORRECTION = "ERROR_CORRECTION"
    SCOPE_RECLASSIFICATION = "SCOPE_RECLASSIFICATION"
    TIMING_ADJUSTMENT = "TIMING_ADJUSTMENT"
    LATE_SUBMISSION = "LATE_SUBMISSION"
    DATA_QUALITY = "DATA_QUALITY"
    BOUNDARY_CHANGE = "BOUNDARY_CHANGE"
    OTHER = "OTHER"

class AdjustmentStatus(str, Enum):
    """Workflow status for adjustment approval."""
    DRAFT = "DRAFT"
    SUBMITTED = "SUBMITTED"
    REVIEWED = "REVIEWED"
    APPROVED = "APPROVED"
    REJECTED = "REJECTED"
    REVERSED = "REVERSED"

class ScopeTarget(str, Enum):
    """Scope the adjustment applies to."""
    SCOPE_1 = "SCOPE_1"
    SCOPE_2_LOCATION = "SCOPE_2_LOCATION"
    SCOPE_2_MARKET = "SCOPE_2_MARKET"
    SCOPE_3 = "SCOPE_3"
    TOTAL = "TOTAL"

# ---------------------------------------------------------------------------
# Pydantic Models
# ---------------------------------------------------------------------------

class AdjustmentRecord(BaseModel):
    """A single consolidation adjustment entry.

    Records the adjustment amount, categorisation, justification,
    and current approval status.
    """
    model_config = ConfigDict(
        json_encoders={Decimal: str},
        validate_default=True,
    )

    adjustment_id: str = Field(
        default_factory=_new_uuid,
        description="Unique adjustment identifier.",
    )
    reporting_year: int = Field(
        ...,
        ge=2000,
        le=2100,
        description="Reporting year this adjustment applies to.",
    )
    entity_id: str = Field(
        ...,
        description="Entity this adjustment applies to.",
    )
    entity_name: Optional[str] = Field(
        None,
        description="Human-readable entity name.",
    )
    category: str = Field(
        ...,
        description="Adjustment category.",
    )
    scope_target: str = Field(
        default=ScopeTarget.TOTAL.value,
        description="Scope the adjustment applies to.",
    )
    adjustment_amount_tco2e: Decimal = Field(
        ...,
        description="Adjustment amount (positive=increase, negative=decrease).",
    )
    before_value_tco2e: Decimal = Field(
        default=Decimal("0"),
        description="Emission value before adjustment.",
    )
    after_value_tco2e: Decimal = Field(
        default=Decimal("0"),
        description="Emission value after adjustment.",
    )
    justification: str = Field(
        ...,
        min_length=1,
        description="Explanation and justification for the adjustment.",
    )
    evidence_reference: Optional[str] = Field(
        None,
        description="Reference to supporting documentation.",
    )
    status: str = Field(
        default=AdjustmentStatus.DRAFT.value,
        description="Current approval status.",
    )
    created_by: Optional[str] = Field(
        None,
        description="User who created the adjustment.",
    )
    is_reversal: bool = Field(
        default=False,
        description="Whether this adjustment reverses a prior one.",
    )
    reversal_of_id: Optional[str] = Field(
        None,
        description="Adjustment ID this reverses (if applicable).",
    )
    batch_id: Optional[str] = Field(
        None,
        description="Batch ID if part of a batch adjustment.",
    )
    created_at: datetime = Field(
        default_factory=utcnow,
        description="When the adjustment was created.",
    )
    updated_at: datetime = Field(
        default_factory=utcnow,
        description="When the adjustment was last updated.",
    )
    provenance_hash: str = Field(
        default="",
        description="SHA-256 provenance hash.",
    )

    @field_validator(
        "adjustment_amount_tco2e", "before_value_tco2e",
        "after_value_tco2e", mode="before",
    )
    @classmethod
    def _coerce_decimal(cls, v: Any) -> Any:
        return Decimal(str(v))

    @field_validator("category")
    @classmethod
    def _validate_category(cls, v: str) -> str:
        valid = {c.value for c in AdjustmentCategory}
        if v.upper() not in valid:
            logger.warning("Adjustment category '%s' not standard; accepted.", v)
        return v.upper()

    @field_validator("status")
    @classmethod
    def _validate_status(cls, v: str) -> str:
        valid = {s.value for s in AdjustmentStatus}
        if v.upper() not in valid:
            raise ValueError(
                f"Invalid status '{v}'. Must be one of {sorted(valid)}."
            )
        return v.upper()

class AdjustmentApproval(BaseModel):
    """Approval or rejection record for an adjustment."""
    model_config = ConfigDict(validate_default=True)

    approval_id: str = Field(
        default_factory=_new_uuid,
        description="Unique approval record identifier.",
    )
    adjustment_id: str = Field(
        ...,
        description="The adjustment being approved/rejected.",
    )
    action: str = Field(
        ...,
        description="Action taken: SUBMITTED, REVIEWED, APPROVED, REJECTED.",
    )
    reviewer: str = Field(
        ...,
        description="Person performing this action.",
    )
    comments: Optional[str] = Field(
        None,
        description="Review comments.",
    )
    previous_status: str = Field(
        ...,
        description="Status before this action.",
    )
    new_status: str = Field(
        ...,
        description="Status after this action.",
    )
    timestamp: datetime = Field(
        default_factory=utcnow,
        description="When this action occurred.",
    )
    provenance_hash: str = Field(
        default="",
        description="SHA-256 provenance hash.",
    )

class AdjustmentImpact(BaseModel):
    """Impact analysis of one or more adjustments.

    Calculates the before and after totals by scope and overall.
    """
    model_config = ConfigDict(
        json_encoders={Decimal: str},
        validate_default=True,
    )

    impact_id: str = Field(
        default_factory=_new_uuid,
        description="Unique impact analysis identifier.",
    )
    reporting_year: int = Field(
        ...,
        description="Reporting year.",
    )
    adjustments_included: int = Field(
        default=0,
        description="Number of adjustments in this analysis.",
    )
    total_adjustment_tco2e: Decimal = Field(
        default=Decimal("0"),
        description="Sum of all adjustment amounts.",
    )
    scope1_adjustment: Decimal = Field(default=Decimal("0"))
    scope2_location_adjustment: Decimal = Field(default=Decimal("0"))
    scope2_market_adjustment: Decimal = Field(default=Decimal("0"))
    scope3_adjustment: Decimal = Field(default=Decimal("0"))
    by_category: Dict[str, Decimal] = Field(
        default_factory=dict,
        description="Adjustment totals by category.",
    )
    by_entity: Dict[str, Decimal] = Field(
        default_factory=dict,
        description="Adjustment totals by entity.",
    )
    net_impact_pct: Decimal = Field(
        default=Decimal("0"),
        description="Net impact as % of pre-adjustment total.",
    )
    pre_adjustment_total: Decimal = Field(
        default=Decimal("0"),
        description="Total emissions before adjustments.",
    )
    post_adjustment_total: Decimal = Field(
        default=Decimal("0"),
        description="Total emissions after adjustments.",
    )
    created_at: datetime = Field(
        default_factory=utcnow,
    )
    provenance_hash: str = Field(default="")

    @field_validator(
        "total_adjustment_tco2e", "scope1_adjustment",
        "scope2_location_adjustment", "scope2_market_adjustment",
        "scope3_adjustment", "net_impact_pct",
        "pre_adjustment_total", "post_adjustment_total",
        mode="before",
    )
    @classmethod
    def _coerce_decimal(cls, v: Any) -> Any:
        return Decimal(str(v))

class AdjustmentBatch(BaseModel):
    """A batch of adjustments applied together.

    Batches allow multiple adjustments to be submitted, reviewed,
    and approved as a group.
    """
    model_config = ConfigDict(
        json_encoders={Decimal: str},
        validate_default=True,
    )

    batch_id: str = Field(
        default_factory=_new_uuid,
        description="Unique batch identifier.",
    )
    reporting_year: int = Field(
        ...,
        description="Reporting year.",
    )
    description: str = Field(
        ...,
        description="Description of the batch.",
    )
    adjustment_ids: List[str] = Field(
        default_factory=list,
        description="Adjustment IDs in this batch.",
    )
    total_adjustment_tco2e: Decimal = Field(
        default=Decimal("0"),
        description="Sum of all adjustments in the batch.",
    )
    status: str = Field(
        default=AdjustmentStatus.DRAFT.value,
        description="Batch status.",
    )
    created_by: Optional[str] = Field(
        None,
        description="User who created the batch.",
    )
    created_at: datetime = Field(
        default_factory=utcnow,
    )
    provenance_hash: str = Field(default="")

    @field_validator("total_adjustment_tco2e", mode="before")
    @classmethod
    def _coerce_decimal(cls, v: Any) -> Any:
        return Decimal(str(v))

# ---------------------------------------------------------------------------
# Allowed status transitions
# ---------------------------------------------------------------------------

_ALLOWED_TRANSITIONS: Dict[str, List[str]] = {
    AdjustmentStatus.DRAFT.value: [
        AdjustmentStatus.SUBMITTED.value,
    ],
    AdjustmentStatus.SUBMITTED.value: [
        AdjustmentStatus.REVIEWED.value,
        AdjustmentStatus.REJECTED.value,
    ],
    AdjustmentStatus.REVIEWED.value: [
        AdjustmentStatus.APPROVED.value,
        AdjustmentStatus.REJECTED.value,
    ],
    AdjustmentStatus.APPROVED.value: [
        AdjustmentStatus.REVERSED.value,
    ],
    AdjustmentStatus.REJECTED.value: [],
    AdjustmentStatus.REVERSED.value: [],
}

# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

class ConsolidationAdjustmentEngine:
    """Manages consolidation adjustments with approval workflow.

    Provides full lifecycle management of manual adjustments
    including creation, approval workflow, impact analysis,
    reversal, and audit trail.

    Attributes:
        _adjustments: Dict mapping adjustment_id to AdjustmentRecord.
        _approvals: List of all AdjustmentApproval actions.
        _batches: Dict mapping batch_id to AdjustmentBatch.

    Example:
        >>> engine = ConsolidationAdjustmentEngine()
        >>> adj = engine.create_adjustment({
        ...     "reporting_year": 2025,
        ...     "entity_id": "ENT-A",
        ...     "category": "ERROR_CORRECTION",
        ...     "adjustment_amount_tco2e": "-500",
        ...     "before_value_tco2e": "10000",
        ...     "justification": "Fix calculation error in natural gas",
        ... })
        >>> engine.approve_adjustment(adj.adjustment_id, "reviewer1")
        >>> impact = engine.calculate_impact(2025)
    """

    def __init__(self) -> None:
        """Initialise the ConsolidationAdjustmentEngine."""
        self._adjustments: Dict[str, AdjustmentRecord] = {}
        self._approvals: List[AdjustmentApproval] = []
        self._batches: Dict[str, AdjustmentBatch] = {}
        logger.info(
            "ConsolidationAdjustmentEngine v%s initialised.",
            _MODULE_VERSION,
        )

    # ------------------------------------------------------------------
    # Adjustment CRUD
    # ------------------------------------------------------------------

    def create_adjustment(
        self,
        adjustment_data: Dict[str, Any],
    ) -> AdjustmentRecord:
        """Create a new adjustment in DRAFT status.

        Automatically calculates the after_value if before_value
        is provided.

        Args:
            adjustment_data: Dictionary of adjustment attributes.

        Returns:
            The created AdjustmentRecord.

        Raises:
            ValueError: If required fields are missing.
        """
        logger.info(
            "Creating adjustment for entity '%s', category '%s'.",
            adjustment_data.get("entity_id", "?"),
            adjustment_data.get("category", "?"),
        )

        if "adjustment_id" not in adjustment_data:
            adjustment_data["adjustment_id"] = _new_uuid()

        # Force DRAFT status on creation
        adjustment_data["status"] = AdjustmentStatus.DRAFT.value

        # Auto-calculate after_value
        before = _decimal(adjustment_data.get("before_value_tco2e", "0"))
        amount = _decimal(adjustment_data.get("adjustment_amount_tco2e", "0"))
        if "after_value_tco2e" not in adjustment_data:
            adjustment_data["after_value_tco2e"] = str(_round2(before + amount))

        adj = AdjustmentRecord(**adjustment_data)
        adj.provenance_hash = _compute_hash(adj)
        self._adjustments[adj.adjustment_id] = adj

        logger.info(
            "Adjustment '%s' created: entity='%s', category=%s, "
            "amount=%s tCO2e, status=DRAFT.",
            adj.adjustment_id,
            adj.entity_id,
            adj.category,
            adj.adjustment_amount_tco2e,
        )
        return adj

    # ------------------------------------------------------------------
    # Approval Workflow
    # ------------------------------------------------------------------

    def submit_adjustment(
        self,
        adjustment_id: str,
        submitter: str,
        comments: Optional[str] = None,
    ) -> AdjustmentRecord:
        """Submit an adjustment for review.

        Args:
            adjustment_id: The adjustment to submit.
            submitter: Person submitting.
            comments: Optional comments.

        Returns:
            Updated AdjustmentRecord.

        Raises:
            KeyError: If adjustment not found.
            ValueError: If status transition is not allowed.
        """
        return self._transition_status(
            adjustment_id,
            AdjustmentStatus.SUBMITTED.value,
            submitter,
            comments,
        )

    def review_adjustment(
        self,
        adjustment_id: str,
        reviewer: str,
        comments: Optional[str] = None,
    ) -> AdjustmentRecord:
        """Mark an adjustment as reviewed.

        Args:
            adjustment_id: The adjustment to review.
            reviewer: Person reviewing.
            comments: Review comments.

        Returns:
            Updated AdjustmentRecord.
        """
        return self._transition_status(
            adjustment_id,
            AdjustmentStatus.REVIEWED.value,
            reviewer,
            comments,
        )

    def approve_adjustment(
        self,
        adjustment_id: str,
        approver: str,
        comments: Optional[str] = None,
    ) -> AdjustmentRecord:
        """Approve an adjustment.

        Args:
            adjustment_id: The adjustment to approve.
            approver: Person approving.
            comments: Approval comments.

        Returns:
            Updated AdjustmentRecord.
        """
        return self._transition_status(
            adjustment_id,
            AdjustmentStatus.APPROVED.value,
            approver,
            comments,
        )

    def reject_adjustment(
        self,
        adjustment_id: str,
        reviewer: str,
        reason: str,
    ) -> AdjustmentRecord:
        """Reject an adjustment.

        Args:
            adjustment_id: The adjustment to reject.
            reviewer: Person rejecting.
            reason: Reason for rejection.

        Returns:
            Updated AdjustmentRecord.
        """
        return self._transition_status(
            adjustment_id,
            AdjustmentStatus.REJECTED.value,
            reviewer,
            reason,
        )

    def _transition_status(
        self,
        adjustment_id: str,
        new_status: str,
        actor: str,
        comments: Optional[str] = None,
    ) -> AdjustmentRecord:
        """Perform a status transition with validation.

        Args:
            adjustment_id: The adjustment to transition.
            new_status: Target status.
            actor: Person performing the action.
            comments: Optional comments.

        Returns:
            Updated AdjustmentRecord.

        Raises:
            KeyError: If adjustment not found.
            ValueError: If transition is not allowed.
        """
        if adjustment_id not in self._adjustments:
            raise KeyError(f"Adjustment '{adjustment_id}' not found.")

        adj = self._adjustments[adjustment_id]
        old_status = adj.status

        allowed = _ALLOWED_TRANSITIONS.get(old_status, [])
        if new_status not in allowed:
            raise ValueError(
                f"Cannot transition from '{old_status}' to '{new_status}'. "
                f"Allowed transitions: {allowed}."
            )

        # Record approval action
        approval = AdjustmentApproval(
            adjustment_id=adjustment_id,
            action=new_status,
            reviewer=actor,
            comments=comments,
            previous_status=old_status,
            new_status=new_status,
        )
        approval.provenance_hash = _compute_hash(approval)
        self._approvals.append(approval)

        # Update the adjustment
        updated_data = adj.model_dump()
        updated_data["status"] = new_status
        updated_data["updated_at"] = utcnow()
        updated = AdjustmentRecord(**updated_data)
        updated.provenance_hash = _compute_hash(updated)
        self._adjustments[adjustment_id] = updated

        logger.info(
            "Adjustment '%s' transitioned: %s -> %s by '%s'.",
            adjustment_id, old_status, new_status, actor,
        )
        return updated

    # ------------------------------------------------------------------
    # Reversal
    # ------------------------------------------------------------------

    def reverse_adjustment(
        self,
        adjustment_id: str,
        reverser: str,
        reason: str,
    ) -> AdjustmentRecord:
        """Reverse a previously approved adjustment.

        Creates a new reversal adjustment with the opposite amount
        and marks the original as REVERSED.

        Args:
            adjustment_id: The adjustment to reverse.
            reverser: Person performing the reversal.
            reason: Reason for reversal.

        Returns:
            The new reversal AdjustmentRecord.

        Raises:
            KeyError: If adjustment not found.
            ValueError: If adjustment is not in APPROVED status.
        """
        if adjustment_id not in self._adjustments:
            raise KeyError(f"Adjustment '{adjustment_id}' not found.")

        original = self._adjustments[adjustment_id]
        if original.status != AdjustmentStatus.APPROVED.value:
            raise ValueError(
                f"Can only reverse APPROVED adjustments. "
                f"Current status: '{original.status}'."
            )

        # Mark original as reversed
        self._transition_status(
            adjustment_id,
            AdjustmentStatus.REVERSED.value,
            reverser,
            f"Reversed: {reason}",
        )

        # Create reversal entry
        reversal_data = {
            "reporting_year": original.reporting_year,
            "entity_id": original.entity_id,
            "entity_name": original.entity_name,
            "category": original.category,
            "scope_target": original.scope_target,
            "adjustment_amount_tco2e": str(
                -original.adjustment_amount_tco2e
            ),
            "before_value_tco2e": str(original.after_value_tco2e),
            "justification": f"Reversal of adjustment '{adjustment_id}': {reason}",
            "evidence_reference": original.evidence_reference,
            "status": AdjustmentStatus.APPROVED.value,
            "created_by": reverser,
            "is_reversal": True,
            "reversal_of_id": adjustment_id,
        }
        reversal = self.create_adjustment(reversal_data)

        # Auto-approve the reversal
        reversal_data_upd = reversal.model_dump()
        reversal_data_upd["status"] = AdjustmentStatus.APPROVED.value
        reversal_data_upd["updated_at"] = utcnow()
        approved_reversal = AdjustmentRecord(**reversal_data_upd)
        approved_reversal.provenance_hash = _compute_hash(approved_reversal)
        self._adjustments[approved_reversal.adjustment_id] = approved_reversal

        logger.info(
            "Adjustment '%s' reversed by '%s'. Reversal ID: '%s'.",
            adjustment_id, reverser, approved_reversal.adjustment_id,
        )
        return approved_reversal

    # ------------------------------------------------------------------
    # Batch Adjustments
    # ------------------------------------------------------------------

    def create_batch(
        self,
        reporting_year: int,
        description: str,
        adjustments: List[Dict[str, Any]],
        created_by: Optional[str] = None,
    ) -> AdjustmentBatch:
        """Create a batch of adjustments.

        All adjustments in the batch share a common batch_id.

        Args:
            reporting_year: Reporting year.
            description: Batch description.
            adjustments: List of adjustment data dictionaries.
            created_by: User creating the batch.

        Returns:
            AdjustmentBatch with all adjustment IDs.
        """
        logger.info(
            "Creating adjustment batch: %d adjustment(s), year %d.",
            len(adjustments), reporting_year,
        )

        batch_id = _new_uuid()
        adj_ids: List[str] = []
        total = Decimal("0")

        for adj_data in adjustments:
            adj_data["batch_id"] = batch_id
            adj_data["reporting_year"] = reporting_year
            if created_by:
                adj_data["created_by"] = created_by
            adj = self.create_adjustment(adj_data)
            adj_ids.append(adj.adjustment_id)
            total += adj.adjustment_amount_tco2e

        batch = AdjustmentBatch(
            batch_id=batch_id,
            reporting_year=reporting_year,
            description=description,
            adjustment_ids=adj_ids,
            total_adjustment_tco2e=_round2(total),
            status=AdjustmentStatus.DRAFT.value,
            created_by=created_by,
        )
        batch.provenance_hash = _compute_hash(batch)
        self._batches[batch_id] = batch

        logger.info(
            "Batch '%s' created: %d adjustment(s), total=%s tCO2e.",
            batch_id, len(adj_ids), batch.total_adjustment_tco2e,
        )
        return batch

    # ------------------------------------------------------------------
    # Impact Calculation
    # ------------------------------------------------------------------

    def calculate_impact(
        self,
        reporting_year: int,
        pre_adjustment_total: Optional[
            Union[Decimal, str, int, float]
        ] = None,
        approved_only: bool = True,
    ) -> AdjustmentImpact:
        """Calculate the cumulative impact of adjustments.

        Aggregates all adjustments (optionally only approved ones)
        for a reporting year and computes scope-level and entity-level
        breakdowns.

        Args:
            reporting_year: The year to analyse.
            pre_adjustment_total: Pre-adjustment total for % calculation.
            approved_only: If True, only include APPROVED adjustments.

        Returns:
            AdjustmentImpact with full breakdown.
        """
        logger.info(
            "Calculating adjustment impact for year %d "
            "(approved_only=%s).",
            reporting_year, approved_only,
        )

        adjs = [
            a for a in self._adjustments.values()
            if a.reporting_year == reporting_year
        ]
        if approved_only:
            adjs = [
                a for a in adjs
                if a.status == AdjustmentStatus.APPROVED.value
            ]

        # Aggregate
        total_adj = Decimal("0")
        s1_adj = Decimal("0")
        s2_loc_adj = Decimal("0")
        s2_mkt_adj = Decimal("0")
        s3_adj = Decimal("0")
        by_category: Dict[str, Decimal] = {}
        by_entity: Dict[str, Decimal] = {}

        for a in adjs:
            amt = a.adjustment_amount_tco2e
            total_adj += amt

            # Scope allocation
            if a.scope_target == ScopeTarget.SCOPE_1.value:
                s1_adj += amt
            elif a.scope_target == ScopeTarget.SCOPE_2_LOCATION.value:
                s2_loc_adj += amt
            elif a.scope_target == ScopeTarget.SCOPE_2_MARKET.value:
                s2_mkt_adj += amt
            elif a.scope_target == ScopeTarget.SCOPE_3.value:
                s3_adj += amt
            # TOTAL scope target contributes to overall but not specific

            by_category[a.category] = (
                by_category.get(a.category, Decimal("0")) + amt
            )
            by_entity[a.entity_id] = (
                by_entity.get(a.entity_id, Decimal("0")) + amt
            )

        # Round breakdowns
        for k in by_category:
            by_category[k] = _round2(by_category[k])
        for k in by_entity:
            by_entity[k] = _round2(by_entity[k])

        pre_total = _decimal(pre_adjustment_total) if pre_adjustment_total else Decimal("0")
        post_total = _round2(pre_total + total_adj)
        impact_pct = _round2(
            _safe_divide(total_adj, pre_total) * Decimal("100")
        ) if pre_total != Decimal("0") else Decimal("0")

        result = AdjustmentImpact(
            reporting_year=reporting_year,
            adjustments_included=len(adjs),
            total_adjustment_tco2e=_round2(total_adj),
            scope1_adjustment=_round2(s1_adj),
            scope2_location_adjustment=_round2(s2_loc_adj),
            scope2_market_adjustment=_round2(s2_mkt_adj),
            scope3_adjustment=_round2(s3_adj),
            by_category=by_category,
            by_entity=by_entity,
            net_impact_pct=impact_pct,
            pre_adjustment_total=_round2(pre_total),
            post_adjustment_total=post_total,
        )
        result.provenance_hash = _compute_hash(result)

        logger.info(
            "Adjustment impact: %d adjustment(s), total=%s tCO2e, "
            "impact=%s%%.",
            len(adjs), result.total_adjustment_tco2e, impact_pct,
        )
        return result

    # ------------------------------------------------------------------
    # History and Accessors
    # ------------------------------------------------------------------

    def get_adjustment_history(
        self,
        adjustment_id: Optional[str] = None,
        entity_id: Optional[str] = None,
        reporting_year: Optional[int] = None,
        category: Optional[str] = None,
        status: Optional[str] = None,
    ) -> List[AdjustmentRecord]:
        """Get adjustment history with optional filters.

        Args:
            adjustment_id: Filter by specific adjustment.
            entity_id: Filter by entity.
            reporting_year: Filter by year.
            category: Filter by category.
            status: Filter by status.

        Returns:
            List of matching AdjustmentRecords.
        """
        results = list(self._adjustments.values())

        if adjustment_id is not None:
            results = [a for a in results if a.adjustment_id == adjustment_id]
        if entity_id is not None:
            results = [a for a in results if a.entity_id == entity_id]
        if reporting_year is not None:
            results = [a for a in results if a.reporting_year == reporting_year]
        if category is not None:
            results = [a for a in results if a.category == category.upper()]
        if status is not None:
            results = [a for a in results if a.status == status.upper()]

        # Sort by created_at
        results.sort(key=lambda a: a.created_at)

        logger.info("Adjustment history: %d record(s) returned.", len(results))
        return results

    def get_approval_history(
        self,
        adjustment_id: str,
    ) -> List[AdjustmentApproval]:
        """Get the approval history for a specific adjustment.

        Args:
            adjustment_id: The adjustment to query.

        Returns:
            List of AdjustmentApproval actions in chronological order.
        """
        return [
            a for a in self._approvals
            if a.adjustment_id == adjustment_id
        ]

    def get_adjustment(self, adjustment_id: str) -> AdjustmentRecord:
        """Retrieve an adjustment by ID.

        Args:
            adjustment_id: The adjustment ID.

        Returns:
            The AdjustmentRecord.

        Raises:
            KeyError: If not found.
        """
        if adjustment_id not in self._adjustments:
            raise KeyError(f"Adjustment '{adjustment_id}' not found.")
        return self._adjustments[adjustment_id]

    def get_batch(self, batch_id: str) -> AdjustmentBatch:
        """Retrieve a batch by ID.

        Args:
            batch_id: The batch ID.

        Returns:
            The AdjustmentBatch.

        Raises:
            KeyError: If not found.
        """
        if batch_id not in self._batches:
            raise KeyError(f"Batch '{batch_id}' not found.")
        return self._batches[batch_id]

    def get_adjustments_summary(
        self,
        reporting_year: int,
    ) -> Dict[str, Any]:
        """Get a summary of all adjustments for a year.

        Args:
            reporting_year: The year to summarise.

        Returns:
            Dictionary with counts and totals by status and category.
        """
        adjs = [
            a for a in self._adjustments.values()
            if a.reporting_year == reporting_year
        ]

        by_status: Dict[str, int] = {}
        by_category: Dict[str, int] = {}
        total_approved = Decimal("0")
        total_pending = Decimal("0")

        for a in adjs:
            by_status[a.status] = by_status.get(a.status, 0) + 1
            by_category[a.category] = by_category.get(a.category, 0) + 1
            if a.status == AdjustmentStatus.APPROVED.value:
                total_approved += a.adjustment_amount_tco2e
            elif a.status in (
                AdjustmentStatus.DRAFT.value,
                AdjustmentStatus.SUBMITTED.value,
                AdjustmentStatus.REVIEWED.value,
            ):
                total_pending += a.adjustment_amount_tco2e

        summary = {
            "reporting_year": reporting_year,
            "total_adjustments": len(adjs),
            "by_status": by_status,
            "by_category": by_category,
            "total_approved_tco2e": str(_round2(total_approved)),
            "total_pending_tco2e": str(_round2(total_pending)),
        }
        summary["provenance_hash"] = _compute_hash(summary)
        return summary
