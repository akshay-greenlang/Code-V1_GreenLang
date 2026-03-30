"""
PACK-050 GHG Consolidation Pack - Boundary Consolidation Engine
====================================================================

Implements GHG Protocol Chapter 3 organizational boundary
consolidation across three approaches: equity share, operational
control, and financial control. Manages boundary definitions,
materiality thresholds, boundary locking, and cross-approach
comparison to help organisations choose the most appropriate
consolidation approach.

Regulatory Basis:
    - GHG Protocol Corporate Standard (Chapter 3): Setting
      Organizational Boundaries - three approaches.
    - GHG Protocol Corporate Standard (Chapter 5): Tracking
      emissions over time with boundary consistency.
    - ISO 14064-1:2018 (Clause 5.1): Organisational boundaries.
    - ESRS E1: Scope of consolidation for GHG disclosures must
      be consistent with financial consolidation scope.

Capabilities:
    - Define boundary using equity share, operational control,
      or financial control approach
    - Apply materiality thresholds (exclude immaterial entities)
    - Compare boundary outcomes across all three approaches
    - Lock boundaries for a reporting period
    - Track boundary changes with justification
    - Entity inclusion/exclusion with audit trail

Calculation Methodology:
    Equity Share:
        inclusion_pct = equity_ownership_pct
    Operational Control:
        inclusion_pct = 100 if has_operational_control else 0
    Financial Control:
        inclusion_pct = 100 if has_financial_control else 0
    Materiality:
        is_material = entity_emissions / total_emissions >= threshold

Zero-Hallucination:
    - All calculations use deterministic Decimal arithmetic
    - No LLM involvement in any calculation path
    - SHA-256 provenance hash on every result object

Author:  GreenLang Platform Team
Date:    March 2026
Pack:    PACK-050 GHG Consolidation
Engine:  3 of 5
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
            if k not in ("created_at", "updated_at", "provenance_hash",
                         "locked_at")
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

def _round4(value: Any) -> Decimal:
    """Round a value to four decimal places using ROUND_HALF_UP."""
    return Decimal(str(value)).quantize(Decimal("0.0001"), rounding=ROUND_HALF_UP)

# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class ConsolidationApproach(str, Enum):
    """GHG Protocol consolidation approaches per Chapter 3."""
    EQUITY_SHARE = "EQUITY_SHARE"
    OPERATIONAL_CONTROL = "OPERATIONAL_CONTROL"
    FINANCIAL_CONTROL = "FINANCIAL_CONTROL"

class BoundaryStatus(str, Enum):
    """Lifecycle status of a boundary definition."""
    DRAFT = "DRAFT"
    UNDER_REVIEW = "UNDER_REVIEW"
    APPROVED = "APPROVED"
    LOCKED = "LOCKED"
    SUPERSEDED = "SUPERSEDED"

class ChangeJustification(str, Enum):
    """Standard justification types for boundary changes."""
    ACQUISITION = "ACQUISITION"
    DIVESTITURE = "DIVESTITURE"
    MERGER = "MERGER"
    RESTRUCTURING = "RESTRUCTURING"
    MATERIALITY = "MATERIALITY"
    REGULATORY_REQUIREMENT = "REGULATORY_REQUIREMENT"
    ERROR_CORRECTION = "ERROR_CORRECTION"
    OTHER = "OTHER"

# ---------------------------------------------------------------------------
# Default Configuration
# ---------------------------------------------------------------------------

DEFAULT_MATERIALITY_THRESHOLD_PCT = Decimal("5")

# ---------------------------------------------------------------------------
# Pydantic Models
# ---------------------------------------------------------------------------

class EntityInclusion(BaseModel):
    """Defines how a specific entity is included in the boundary.

    Links an entity to its inclusion percentage and consolidation
    approach, with exclusion capability and justification tracking.
    """
    model_config = ConfigDict(
        json_encoders={Decimal: str},
        validate_default=True,
    )

    entity_id: str = Field(
        ...,
        description="The entity being included/excluded.",
    )
    entity_name: Optional[str] = Field(
        None,
        description="Human-readable entity name.",
    )
    inclusion_pct: Decimal = Field(
        ...,
        ge=Decimal("0"),
        le=Decimal("100"),
        description="Percentage of emissions to include (0-100).",
    )
    equity_pct: Optional[Decimal] = Field(
        None,
        ge=Decimal("0"),
        le=Decimal("100"),
        description="Equity ownership percentage.",
    )
    has_operational_control: bool = Field(
        default=False,
        description="Whether reporting org has operational control.",
    )
    has_financial_control: bool = Field(
        default=False,
        description="Whether reporting org has financial control.",
    )
    is_excluded: bool = Field(
        default=False,
        description="Whether the entity is explicitly excluded.",
    )
    exclusion_reason: Optional[str] = Field(
        None,
        description="Short reason for exclusion.",
    )
    exclusion_justification: Optional[str] = Field(
        None,
        description="Detailed justification per GHG Protocol guidance.",
    )
    materiality_pct: Optional[Decimal] = Field(
        None,
        description="Entity emissions as percentage of corporate total.",
    )

    @field_validator("inclusion_pct", "equity_pct", "materiality_pct",
                     mode="before")
    @classmethod
    def _coerce_pct(cls, v: Any) -> Any:
        if v is not None:
            return Decimal(str(v))
        return v

class BoundaryDefinition(BaseModel):
    """Complete organizational boundary definition for a reporting period.

    Contains all entity inclusions/exclusions and boundary metadata.
    Can be locked to prevent modification after approval.
    """
    model_config = ConfigDict(
        json_encoders={Decimal: str},
        validate_default=True,
    )

    boundary_id: str = Field(
        default_factory=_new_uuid,
        description="Unique boundary identifier.",
    )
    reporting_year: int = Field(
        ...,
        ge=2000,
        le=2100,
        description="The reporting year this boundary covers.",
    )
    consolidation_approach: str = Field(
        ...,
        description="Primary consolidation approach.",
    )
    inclusions: List[EntityInclusion] = Field(
        default_factory=list,
        description="All entity inclusions (and exclusions).",
    )
    total_entities_included: int = Field(
        default=0,
        description="Count of included (non-excluded) entities.",
    )
    total_entities_excluded: int = Field(
        default=0,
        description="Count of excluded entities.",
    )
    avg_inclusion_pct: Decimal = Field(
        default=Decimal("0"),
        description="Average inclusion percentage across included entities.",
    )
    materiality_threshold_pct: Decimal = Field(
        default=DEFAULT_MATERIALITY_THRESHOLD_PCT,
        description="Materiality threshold applied.",
    )
    status: str = Field(
        default="DRAFT",
        description="Current status of this boundary definition.",
    )
    is_locked: bool = Field(
        default=False,
        description="Whether the boundary is locked.",
    )
    locked_at: Optional[datetime] = Field(
        None,
        description="When the boundary was locked.",
    )
    locked_by: Optional[str] = Field(
        None,
        description="Who locked the boundary.",
    )
    change_justifications: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="History of boundary change justifications.",
    )
    notes: Optional[str] = Field(
        None,
        description="Additional notes.",
    )
    created_at: datetime = Field(
        default_factory=utcnow,
        description="When the boundary was created.",
    )
    updated_at: datetime = Field(
        default_factory=utcnow,
        description="When the boundary was last updated.",
    )
    provenance_hash: str = Field(
        default="",
        description="SHA-256 hash.",
    )

    @field_validator("consolidation_approach")
    @classmethod
    def _validate_approach(cls, v: str) -> str:
        valid = {ca.value for ca in ConsolidationApproach}
        if v.upper() not in valid:
            raise ValueError(
                f"Invalid consolidation_approach '{v}'. "
                f"Must be one of {sorted(valid)}."
            )
        return v.upper()

    @field_validator("status")
    @classmethod
    def _validate_status(cls, v: str) -> str:
        valid = {bs.value for bs in BoundaryStatus}
        if v.upper() not in valid:
            raise ValueError(
                f"Invalid boundary status '{v}'. "
                f"Must be one of {sorted(valid)}."
            )
        return v.upper()

class BoundaryComparison(BaseModel):
    """Comparison of boundary outcomes across consolidation approaches.

    Shows how entity inclusion differs between equity share,
    operational control, and financial control approaches.
    """
    model_config = ConfigDict(
        json_encoders={Decimal: str},
        validate_default=True,
    )

    comparison_id: str = Field(
        default_factory=_new_uuid,
        description="Unique comparison identifier.",
    )
    reporting_year: int = Field(
        ...,
        description="The reporting year.",
    )
    entity_comparisons: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Per-entity inclusion under each approach.",
    )
    total_entities: int = Field(
        default=0,
        description="Total entities evaluated.",
    )
    equity_share_included: int = Field(
        default=0,
        description="Entities included under equity share.",
    )
    operational_control_included: int = Field(
        default=0,
        description="Entities included under operational control.",
    )
    financial_control_included: int = Field(
        default=0,
        description="Entities included under financial control.",
    )
    recommendation: Optional[str] = Field(
        None,
        description="Recommendation based on comparison.",
    )
    created_at: datetime = Field(
        default_factory=utcnow,
        description="When this comparison was generated.",
    )
    provenance_hash: str = Field(
        default="",
        description="SHA-256 hash.",
    )

class BoundaryLock(BaseModel):
    """Record of a boundary lock event."""
    model_config = ConfigDict(validate_default=True)

    lock_id: str = Field(
        default_factory=_new_uuid,
        description="Unique lock identifier.",
    )
    boundary_id: str = Field(
        ...,
        description="The locked boundary.",
    )
    locked_by: str = Field(
        ...,
        description="User who locked the boundary.",
    )
    locked_at: datetime = Field(
        default_factory=utcnow,
        description="When the lock was applied.",
    )
    reason: Optional[str] = Field(
        None,
        description="Reason for locking.",
    )
    provenance_hash: str = Field(
        default="",
        description="SHA-256 hash.",
    )

# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

class BoundaryConsolidationEngine:
    """Manages organizational boundary consolidation per GHG Protocol Ch. 3.

    Implements the three consolidation approaches, applies
    materiality thresholds, compares approaches, and manages
    boundary locking for reporting period finalization.

    Attributes:
        _boundaries: Dict mapping boundary_id to BoundaryDefinition.
        _locks: List of BoundaryLock records.
        _change_log: Append-only audit log.

    Example:
        >>> engine = BoundaryConsolidationEngine()
        >>> boundary = engine.define_boundary(
        ...     reporting_year=2025,
        ...     approach="EQUITY_SHARE",
        ...     entities=[
        ...         {"entity_id": "E1", "equity_pct": "100",
        ...          "has_operational_control": True},
        ...         {"entity_id": "E2", "equity_pct": "60"},
        ...     ],
        ... )
        >>> assert boundary.total_entities_included == 2
    """

    def __init__(self) -> None:
        """Initialise the BoundaryConsolidationEngine."""
        self._boundaries: Dict[str, BoundaryDefinition] = {}
        self._locks: List[BoundaryLock] = []
        self._change_log: List[Dict[str, Any]] = []
        logger.info(
            "BoundaryConsolidationEngine v%s initialised.", _MODULE_VERSION,
        )

    # ------------------------------------------------------------------
    # Boundary Definition
    # ------------------------------------------------------------------

    def define_boundary(
        self,
        reporting_year: int,
        approach: str,
        entities: List[Dict[str, Any]],
        materiality_threshold: Optional[Union[Decimal, str, float]] = None,
        notes: Optional[str] = None,
    ) -> BoundaryDefinition:
        """Define an organizational boundary for a reporting period.

        Automatically computes inclusion percentages based on the
        chosen approach and entity ownership/control attributes.

        Args:
            reporting_year: The reporting year.
            approach: Consolidation approach.
            entities: List of entity dicts, each must have entity_id
                and appropriate ownership/control fields.
            materiality_threshold: Optional threshold percentage.
            notes: Optional notes.

        Returns:
            The created BoundaryDefinition.

        Raises:
            ValueError: If approach is invalid.
        """
        approach_upper = approach.upper()
        logger.info(
            "Defining %s boundary for year %d with %d entity(ies).",
            approach_upper, reporting_year, len(entities),
        )

        threshold = _decimal(
            materiality_threshold
            if materiality_threshold is not None
            else DEFAULT_MATERIALITY_THRESHOLD_PCT
        )

        inclusions: List[EntityInclusion] = []
        for entity_data in entities:
            inclusion = self._build_entity_inclusion(
                entity_data, approach_upper
            )
            inclusions.append(inclusion)

        boundary = self._build_boundary(
            year=reporting_year,
            approach=approach_upper,
            inclusions=inclusions,
            threshold=threshold,
            notes=notes,
        )
        self._boundaries[boundary.boundary_id] = boundary

        self._change_log.append({
            "event": "BOUNDARY_DEFINED",
            "boundary_id": boundary.boundary_id,
            "year": reporting_year,
            "approach": approach_upper,
            "entities_included": boundary.total_entities_included,
            "timestamp": utcnow().isoformat(),
        })

        logger.info(
            "Boundary '%s' defined: %d included, %d excluded.",
            boundary.boundary_id,
            boundary.total_entities_included,
            boundary.total_entities_excluded,
        )
        return boundary

    def _build_entity_inclusion(
        self,
        entity_data: Dict[str, Any],
        approach: str,
    ) -> EntityInclusion:
        """Build an EntityInclusion from entity data and approach.

        Args:
            entity_data: Entity attributes dict.
            approach: Consolidation approach.

        Returns:
            Computed EntityInclusion.
        """
        entity_id = entity_data.get("entity_id", _new_uuid())
        entity_name = entity_data.get("entity_name")
        equity_pct = _decimal(entity_data.get("equity_pct", "0"))
        has_oper = entity_data.get("has_operational_control", False)
        has_fin = entity_data.get("has_financial_control", False)

        # Compute inclusion percentage based on approach
        if approach == ConsolidationApproach.EQUITY_SHARE.value:
            inclusion_pct = equity_pct
        elif approach == ConsolidationApproach.OPERATIONAL_CONTROL.value:
            inclusion_pct = Decimal("100") if has_oper else Decimal("0")
        elif approach == ConsolidationApproach.FINANCIAL_CONTROL.value:
            inclusion_pct = Decimal("100") if has_fin else Decimal("0")
        else:
            inclusion_pct = Decimal("0")

        return EntityInclusion(
            entity_id=entity_id,
            entity_name=entity_name,
            inclusion_pct=inclusion_pct,
            equity_pct=equity_pct,
            has_operational_control=has_oper,
            has_financial_control=has_fin,
        )

    def _build_boundary(
        self,
        year: int,
        approach: str,
        inclusions: List[EntityInclusion],
        threshold: Decimal = DEFAULT_MATERIALITY_THRESHOLD_PCT,
        notes: Optional[str] = None,
        changes: Optional[List[Dict[str, Any]]] = None,
    ) -> BoundaryDefinition:
        """Build a BoundaryDefinition from inclusions.

        Args:
            year: Reporting year.
            approach: Consolidation approach.
            inclusions: List of EntityInclusion records.
            threshold: Materiality threshold.
            notes: Optional notes.
            changes: Optional change history.

        Returns:
            Constructed BoundaryDefinition.
        """
        included = [i for i in inclusions if not i.is_excluded and i.inclusion_pct > Decimal("0")]
        excluded = [i for i in inclusions if i.is_excluded or i.inclusion_pct == Decimal("0")]

        if included:
            total_pct = sum(i.inclusion_pct for i in included)
            avg_pct = _round2(
                _safe_divide(total_pct, _decimal(len(included)))
            )
        else:
            avg_pct = Decimal("0")

        now = utcnow()
        boundary = BoundaryDefinition(
            reporting_year=year,
            consolidation_approach=approach,
            inclusions=inclusions,
            total_entities_included=len(included),
            total_entities_excluded=len(excluded),
            avg_inclusion_pct=avg_pct,
            materiality_threshold_pct=threshold,
            change_justifications=changes or [],
            notes=notes,
            created_at=now,
            updated_at=now,
        )
        boundary.provenance_hash = _compute_hash(boundary)
        return boundary

    # ------------------------------------------------------------------
    # Apply Approach
    # ------------------------------------------------------------------

    def apply_approach(
        self,
        boundary: BoundaryDefinition,
        new_approach: str,
    ) -> BoundaryDefinition:
        """Re-apply a different consolidation approach to a boundary.

        Recalculates all inclusion percentages using the new approach
        while preserving entity attributes.

        Args:
            boundary: Existing boundary definition.
            new_approach: New consolidation approach.

        Returns:
            Updated BoundaryDefinition with new approach applied.

        Raises:
            ValueError: If boundary is locked or approach is invalid.
        """
        if boundary.is_locked:
            raise ValueError(
                f"Boundary '{boundary.boundary_id}' is locked."
            )

        new_approach_upper = new_approach.upper()
        logger.info(
            "Applying approach %s to boundary '%s'.",
            new_approach_upper, boundary.boundary_id,
        )

        updated_inclusions: List[EntityInclusion] = []
        for incl in boundary.inclusions:
            entity_data = {
                "entity_id": incl.entity_id,
                "entity_name": incl.entity_name,
                "equity_pct": str(incl.equity_pct) if incl.equity_pct else "0",
                "has_operational_control": incl.has_operational_control,
                "has_financial_control": incl.has_financial_control,
            }
            new_incl = self._build_entity_inclusion(
                entity_data, new_approach_upper
            )
            # Preserve exclusion state
            if incl.is_excluded:
                new_incl = new_incl.model_copy(update={
                    "is_excluded": True,
                    "exclusion_reason": incl.exclusion_reason,
                    "exclusion_justification": incl.exclusion_justification,
                })
            updated_inclusions.append(new_incl)

        updated = self._build_boundary(
            year=boundary.reporting_year,
            approach=new_approach_upper,
            inclusions=updated_inclusions,
            threshold=boundary.materiality_threshold_pct,
            notes=boundary.notes,
            changes=boundary.change_justifications,
        )
        updated = updated.model_copy(update={
            "boundary_id": boundary.boundary_id,
        })
        updated.provenance_hash = _compute_hash(updated)
        self._boundaries[updated.boundary_id] = updated

        return updated

    # ------------------------------------------------------------------
    # Materiality
    # ------------------------------------------------------------------

    def apply_materiality_threshold(
        self,
        boundary: BoundaryDefinition,
        entity_emissions: Dict[str, Union[Decimal, str, float]],
        corporate_total: Union[Decimal, str, float],
        threshold: Optional[Union[Decimal, str, float]] = None,
    ) -> BoundaryDefinition:
        """Apply a materiality threshold to exclude immaterial entities.

        Entities whose emissions fall below the threshold percentage
        of the corporate total are marked as excluded with a
        MATERIALITY justification.

        Args:
            boundary: The boundary to apply threshold to.
            entity_emissions: Dict mapping entity_id to emissions.
            corporate_total: Total corporate emissions.
            threshold: Materiality threshold percentage.

        Returns:
            Updated BoundaryDefinition with immaterial entities excluded.

        Raises:
            ValueError: If boundary is locked.
        """
        if boundary.is_locked:
            raise ValueError(
                f"Boundary '{boundary.boundary_id}' is locked."
            )

        thresh = _decimal(
            threshold if threshold is not None
            else boundary.materiality_threshold_pct
        )
        corp_total = _decimal(corporate_total)

        logger.info(
            "Applying materiality threshold of %s%% to boundary '%s'.",
            thresh, boundary.boundary_id,
        )

        updated_inclusions: List[EntityInclusion] = []
        excluded_count = 0

        for incl in boundary.inclusions:
            emissions = _decimal(entity_emissions.get(incl.entity_id, "0"))
            materiality_pct = _round4(
                _safe_divide(emissions, corp_total) * Decimal("100")
            )

            if materiality_pct < thresh and not incl.is_excluded:
                updated = incl.model_copy(update={
                    "is_excluded": True,
                    "exclusion_reason": "Below materiality threshold",
                    "exclusion_justification": (
                        f"Entity emissions ({emissions} tCO2e) represent "
                        f"{materiality_pct}% of corporate total, below "
                        f"the {thresh}% materiality threshold."
                    ),
                    "materiality_pct": materiality_pct,
                })
                updated_inclusions.append(updated)
                excluded_count += 1
            else:
                updated = incl.model_copy(update={
                    "materiality_pct": materiality_pct,
                })
                updated_inclusions.append(updated)

        result = self._build_boundary(
            year=boundary.reporting_year,
            approach=boundary.consolidation_approach,
            inclusions=updated_inclusions,
            threshold=thresh,
            notes=boundary.notes,
            changes=boundary.change_justifications,
        )
        result = result.model_copy(update={
            "boundary_id": boundary.boundary_id,
        })
        result.provenance_hash = _compute_hash(result)
        self._boundaries[result.boundary_id] = result

        logger.info(
            "Materiality applied: %d entity(ies) excluded below %s%%.",
            excluded_count, thresh,
        )
        return result

    # ------------------------------------------------------------------
    # Cross-Approach Comparison
    # ------------------------------------------------------------------

    def compare_approaches(
        self,
        reporting_year: int,
        entities: List[Dict[str, Any]],
    ) -> BoundaryComparison:
        """Compare boundary outcomes across all three approaches.

        Evaluates every entity under equity share, operational
        control, and financial control to show the impact of
        approach choice on the consolidation boundary.

        Args:
            reporting_year: The reporting year.
            entities: List of entity dicts with ownership and
                control attributes.

        Returns:
            BoundaryComparison with per-entity and aggregate results.
        """
        logger.info(
            "Comparing consolidation approaches for year %d "
            "across %d entity(ies).",
            reporting_year, len(entities),
        )

        comparisons: List[Dict[str, Any]] = []
        equity_count = 0
        oper_count = 0
        fin_count = 0

        for entity_data in entities:
            entity_id = entity_data.get("entity_id", "")
            entity_name = entity_data.get("entity_name", entity_id)
            equity_pct = _decimal(entity_data.get("equity_pct", "0"))
            has_oper = entity_data.get("has_operational_control", False)
            has_fin = entity_data.get("has_financial_control", False)

            # Equity share inclusion
            eq_incl = equity_pct

            # Operational control inclusion
            oc_incl = Decimal("100") if has_oper else Decimal("0")

            # Financial control inclusion
            fc_incl = Decimal("100") if has_fin else Decimal("0")

            comparisons.append({
                "entity_id": entity_id,
                "entity_name": entity_name,
                "equity_pct": str(equity_pct),
                "equity_share_inclusion_pct": str(eq_incl),
                "operational_control_inclusion_pct": str(oc_incl),
                "financial_control_inclusion_pct": str(fc_incl),
                "has_operational_control": has_oper,
                "has_financial_control": has_fin,
            })

            if eq_incl > Decimal("0"):
                equity_count += 1
            if oc_incl > Decimal("0"):
                oper_count += 1
            if fc_incl > Decimal("0"):
                fin_count += 1

        # Generate recommendation
        if equity_count >= oper_count and equity_count >= fin_count:
            recommendation = (
                f"Equity share includes the most entities ({equity_count}). "
                f"Recommended for organisations with many JVs and associates."
            )
        elif oper_count >= fin_count:
            recommendation = (
                f"Operational control includes {oper_count} entities. "
                f"Recommended for organisations that manage operations directly."
            )
        else:
            recommendation = (
                f"Financial control includes {fin_count} entities. "
                f"Recommended for alignment with financial consolidation."
            )

        result = BoundaryComparison(
            reporting_year=reporting_year,
            entity_comparisons=comparisons,
            total_entities=len(entities),
            equity_share_included=equity_count,
            operational_control_included=oper_count,
            financial_control_included=fin_count,
            recommendation=recommendation,
        )
        result.provenance_hash = _compute_hash(result)

        logger.info(
            "Approach comparison: equity=%d, operational=%d, financial=%d.",
            equity_count, oper_count, fin_count,
        )
        return result

    # ------------------------------------------------------------------
    # Boundary Locking
    # ------------------------------------------------------------------

    def lock_boundary(
        self,
        boundary: BoundaryDefinition,
        locked_by: str,
        reason: Optional[str] = None,
    ) -> BoundaryDefinition:
        """Lock a boundary to prevent further modifications.

        Locking is typically done after all reviews are complete
        and the boundary is finalised for the reporting period.

        Args:
            boundary: The boundary to lock.
            locked_by: User who is locking.
            reason: Reason for locking.

        Returns:
            The locked BoundaryDefinition.

        Raises:
            ValueError: If already locked.
        """
        if boundary.is_locked:
            raise ValueError(
                f"Boundary '{boundary.boundary_id}' is already locked."
            )

        logger.info(
            "Locking boundary '%s' for year %d by '%s'.",
            boundary.boundary_id, boundary.reporting_year, locked_by,
        )

        now = utcnow()
        updated = boundary.model_copy(update={
            "is_locked": True,
            "locked_at": now,
            "locked_by": locked_by,
            "status": BoundaryStatus.LOCKED.value,
            "updated_at": now,
        })
        updated.provenance_hash = _compute_hash(updated)
        self._boundaries[updated.boundary_id] = updated

        lock_record = BoundaryLock(
            boundary_id=boundary.boundary_id,
            locked_by=locked_by,
            locked_at=now,
            reason=reason,
        )
        lock_record.provenance_hash = _compute_hash(lock_record)
        self._locks.append(lock_record)

        self._change_log.append({
            "event": "BOUNDARY_LOCKED",
            "boundary_id": boundary.boundary_id,
            "locked_by": locked_by,
            "reason": reason,
            "timestamp": now.isoformat(),
        })

        return updated

    # ------------------------------------------------------------------
    # Boundary Change Management
    # ------------------------------------------------------------------

    def record_boundary_change(
        self,
        boundary: BoundaryDefinition,
        justification_type: str,
        description: str,
        affected_entity_ids: List[str],
    ) -> BoundaryDefinition:
        """Record a boundary change with justification.

        Adds a justification record to the boundary's change
        history for audit trail purposes.

        Args:
            boundary: The boundary being modified.
            justification_type: Type of justification.
            description: Detailed description of the change.
            affected_entity_ids: Entities affected by the change.

        Returns:
            Updated BoundaryDefinition with new justification.

        Raises:
            ValueError: If boundary is locked.
        """
        if boundary.is_locked:
            raise ValueError(
                f"Boundary '{boundary.boundary_id}' is locked."
            )

        change_record = {
            "change_id": _new_uuid(),
            "justification_type": justification_type.upper(),
            "description": description,
            "affected_entity_ids": affected_entity_ids,
            "timestamp": utcnow().isoformat(),
        }

        updated_changes = list(boundary.change_justifications) + [change_record]
        updated = boundary.model_copy(update={
            "change_justifications": updated_changes,
            "updated_at": utcnow(),
        })
        updated.provenance_hash = _compute_hash(updated)
        self._boundaries[updated.boundary_id] = updated

        self._change_log.append({
            "event": "BOUNDARY_CHANGE_RECORDED",
            "boundary_id": boundary.boundary_id,
            "justification_type": justification_type.upper(),
            "affected_entities": len(affected_entity_ids),
            "timestamp": utcnow().isoformat(),
        })

        logger.info(
            "Boundary change recorded: %s affecting %d entity(ies).",
            justification_type, len(affected_entity_ids),
        )
        return updated

    # ------------------------------------------------------------------
    # Entity Management
    # ------------------------------------------------------------------

    def get_included_entities(
        self,
        boundary: BoundaryDefinition,
    ) -> List[EntityInclusion]:
        """Get all included (non-excluded) entities for a boundary.

        Args:
            boundary: The boundary definition.

        Returns:
            List of included EntityInclusion records.
        """
        return [
            i for i in boundary.inclusions
            if not i.is_excluded and i.inclusion_pct > Decimal("0")
        ]

    def get_excluded_entities(
        self,
        boundary: BoundaryDefinition,
    ) -> List[EntityInclusion]:
        """Get all excluded entities for a boundary.

        Args:
            boundary: The boundary definition.

        Returns:
            List of excluded EntityInclusion records.
        """
        return [
            i for i in boundary.inclusions
            if i.is_excluded or i.inclusion_pct == Decimal("0")
        ]

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    def get_boundary(self, boundary_id: str) -> BoundaryDefinition:
        """Retrieve a boundary by ID.

        Args:
            boundary_id: The boundary ID.

        Returns:
            The BoundaryDefinition.

        Raises:
            KeyError: If not found.
        """
        if boundary_id not in self._boundaries:
            raise KeyError(f"Boundary '{boundary_id}' not found.")
        return self._boundaries[boundary_id]

    def get_boundaries_for_year(
        self, year: int,
    ) -> List[BoundaryDefinition]:
        """Get all boundaries for a specific year.

        Args:
            year: The reporting year.

        Returns:
            List of BoundaryDefinitions.
        """
        return [
            b for b in self._boundaries.values()
            if b.reporting_year == year
        ]

    def get_all_boundaries(self) -> List[BoundaryDefinition]:
        """Return all boundary definitions.

        Returns:
            List of all BoundaryDefinitions.
        """
        return list(self._boundaries.values())

    def get_locks(self) -> List[BoundaryLock]:
        """Return all boundary lock records.

        Returns:
            List of BoundaryLock records.
        """
        return list(self._locks)

    def get_change_log(self) -> List[Dict[str, Any]]:
        """Return the complete change log.

        Returns:
            List of change log entries.
        """
        return list(self._change_log)
