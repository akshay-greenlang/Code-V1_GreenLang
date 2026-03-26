"""
PACK-049 GHG Multi-Site Management Pack - Site Boundary Engine
====================================================================

Manages organisational boundary definitions for multi-site GHG
reporting. Implements the three GHG Protocol consolidation approaches
(equity share, operational control, financial control), handles
boundary changes from M&A activity, and provides time-weighted
consolidation and materiality assessment.

Regulatory Basis:
    - GHG Protocol Corporate Standard (Chapter 3): Setting
      Organizational Boundaries - equity share, operational control,
      and financial control approaches.
    - GHG Protocol Corporate Standard (Chapter 5): Tracking emissions
      over time - structural changes and base year recalculation.
    - ISO 14064-1:2018 (Clause 5.1): Organisational boundaries -
      consolidation approaches and equity share calculation.
    - ESRS E1: Scope of consolidation for GHG disclosures.
    - PCAF Global GHG Accounting Standard: Attribution rules for
      financed emissions.

Capabilities:
    - Define boundaries using equity share, operational control,
      or financial control approaches
    - Add and exclude sites with full audit trail
    - Apply structural boundary changes (acquisitions, divestitures,
      mergers, insourcing, outsourcing)
    - Time-weighted consolidation for mid-year boundary changes
    - Materiality assessment for site exclusions
    - Boundary locking for reporting period finalisation
    - Boundary comparison across periods

Calculation Methodology:
    Time-Weighted Consolidation:
        Adjusted_pct = inclusion_pct * (days_included / total_days)
        Where:
            days_included = year_end - max(change_date, year_start) + 1
            total_days = year_end - year_start + 1

    Materiality Assessment:
        site_materiality_pct = site_emissions / corporate_total * 100
        is_material = site_materiality_pct >= threshold

Zero-Hallucination:
    - All calculations use deterministic Decimal arithmetic
    - No LLM involvement in any calculation path
    - SHA-256 provenance hash on every result object

Author:  GreenLang Platform Team
Date:    March 2026
Pack:    PACK-049 GHG Multi-Site Management
Engine:  3 of 10
Status:  Production Ready
"""

from __future__ import annotations

import hashlib
import json
import logging
import uuid
from datetime import datetime, date, timezone, timedelta
from decimal import Decimal, ROUND_HALF_UP, InvalidOperation
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

logger = logging.getLogger(__name__)
_MODULE_VERSION = "1.0.0"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _utcnow() -> datetime:
    """Return current UTC timestamp with second precision."""
    return datetime.now(timezone.utc).replace(microsecond=0)


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


def _days_in_year(year: int) -> int:
    """Return the number of days in the given year."""
    start = date(year, 1, 1)
    end = date(year, 12, 31)
    return (end - start).days + 1


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class ConsolidationApproach(str, Enum):
    """GHG Protocol consolidation approaches.

    EQUITY_SHARE: Organisation accounts for its share of emissions
        proportional to its equity interest.
    OPERATIONAL_CONTROL: 100% of emissions from operations where the
        organisation has operational control.
    FINANCIAL_CONTROL: 100% of emissions from operations where the
        organisation has financial control.
    """
    EQUITY_SHARE = "EQUITY_SHARE"
    OPERATIONAL_CONTROL = "OPERATIONAL_CONTROL"
    FINANCIAL_CONTROL = "FINANCIAL_CONTROL"


class OwnershipType(str, Enum):
    """Types of ownership relationships."""
    SUBSIDIARY = "SUBSIDIARY"
    JOINT_VENTURE = "JOINT_VENTURE"
    ASSOCIATE = "ASSOCIATE"
    FRANCHISE = "FRANCHISE"
    LEASED = "LEASED"
    JOINT_OPERATION = "JOINT_OPERATION"
    WHOLLY_OWNED = "WHOLLY_OWNED"


class ChangeType(str, Enum):
    """Types of boundary changes."""
    ACQUISITION = "ACQUISITION"
    DIVESTITURE = "DIVESTITURE"
    MERGER = "MERGER"
    INSOURCING = "INSOURCING"
    OUTSOURCING = "OUTSOURCING"
    ORGANIC_GROWTH = "ORGANIC_GROWTH"
    CLOSURE = "CLOSURE"
    EQUITY_CHANGE = "EQUITY_CHANGE"


class ScopeInclusion(str, Enum):
    """GHG scopes that can be included in the boundary."""
    SCOPE_1 = "SCOPE_1"
    SCOPE_2_LOCATION = "SCOPE_2_LOCATION"
    SCOPE_2_MARKET = "SCOPE_2_MARKET"
    SCOPE_3 = "SCOPE_3"


# ---------------------------------------------------------------------------
# Default Materiality Threshold
# ---------------------------------------------------------------------------

DEFAULT_MATERIALITY_THRESHOLD_PCT = Decimal("5")


# ---------------------------------------------------------------------------
# Pydantic Models
# ---------------------------------------------------------------------------


class EntityOwnership(BaseModel):
    """Describes the ownership structure of a legal entity.

    Used to determine inclusion percentages under the equity share
    and control-based consolidation approaches.
    """
    model_config = ConfigDict(
        json_encoders={Decimal: str},
        validate_default=True,
    )

    entity_id: str = Field(
        ...,
        description="Unique entity identifier.",
    )
    entity_name: str = Field(
        ...,
        description="Legal name of the entity.",
    )
    parent_entity_id: Optional[str] = Field(
        None,
        description="Parent entity ID (None for top-level).",
    )
    ownership_pct: Decimal = Field(
        ...,
        ge=Decimal("0"),
        le=Decimal("100"),
        description="Equity ownership percentage (0-100).",
    )
    has_operational_control: bool = Field(
        default=False,
        description="Whether the reporting entity has operational control.",
    )
    has_financial_control: bool = Field(
        default=False,
        description="Whether the reporting entity has financial control.",
    )
    ownership_type: str = Field(
        ...,
        description="Type of ownership relationship.",
    )

    @field_validator("ownership_pct", mode="before")
    @classmethod
    def _coerce_pct(cls, v: Any) -> Any:
        return Decimal(str(v))

    @field_validator("ownership_type")
    @classmethod
    def _validate_ownership_type(cls, v: str) -> str:
        valid = {ot.value for ot in OwnershipType}
        if v.upper() not in valid:
            logger.warning("Ownership type '%s' not standard; accepted.", v)
        return v.upper()


class BoundaryInclusion(BaseModel):
    """Defines how a specific site is included in the boundary.

    Links a site to its legal entity with inclusion percentage,
    consolidation approach, and scope coverage.
    """
    model_config = ConfigDict(
        json_encoders={Decimal: str},
        validate_default=True,
    )

    site_id: str = Field(
        ...,
        description="The site being included.",
    )
    entity_id: str = Field(
        ...,
        description="The legal entity that owns/operates the site.",
    )
    inclusion_pct: Decimal = Field(
        ...,
        ge=Decimal("0"),
        le=Decimal("100"),
        description="Percentage of emissions to include (0-100).",
    )
    consolidation_approach: str = Field(
        ...,
        description="The consolidation approach applied.",
    )
    included_scopes: List[str] = Field(
        default_factory=lambda: [
            ScopeInclusion.SCOPE_1.value,
            ScopeInclusion.SCOPE_2_LOCATION.value,
            ScopeInclusion.SCOPE_2_MARKET.value,
        ],
        description="GHG scopes included in the boundary.",
    )
    is_excluded: bool = Field(
        default=False,
        description="Whether the site is explicitly excluded.",
    )
    exclusion_reason: Optional[str] = Field(
        None,
        description="Reason for exclusion (if is_excluded=True).",
    )
    exclusion_justification: Optional[str] = Field(
        None,
        description="Detailed justification for exclusion.",
    )
    time_weighted_pct: Optional[Decimal] = Field(
        None,
        description="Time-weighted inclusion pct for mid-year changes.",
    )
    effective_from: Optional[date] = Field(
        None,
        description="Date from which this inclusion is effective.",
    )
    effective_to: Optional[date] = Field(
        None,
        description="Date until which this inclusion is effective.",
    )

    @field_validator("inclusion_pct", mode="before")
    @classmethod
    def _coerce_pct(cls, v: Any) -> Any:
        return Decimal(str(v))

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


class BoundaryChange(BaseModel):
    """Records a structural change to the organisational boundary.

    Tracks acquisitions, divestitures, mergers, and other events
    that alter the reporting boundary and may trigger base year
    recalculation.
    """
    model_config = ConfigDict(
        json_encoders={Decimal: str},
        validate_default=True,
    )

    change_id: str = Field(
        default_factory=_new_uuid,
        description="Unique change identifier.",
    )
    change_type: str = Field(
        ...,
        description="Type of boundary change.",
    )
    affected_site_ids: List[str] = Field(
        default_factory=list,
        description="Sites affected by this change.",
    )
    effective_date: date = Field(
        ...,
        description="Date the change takes effect.",
    )
    equity_before: Optional[Decimal] = Field(
        None,
        description="Equity percentage before the change.",
    )
    equity_after: Optional[Decimal] = Field(
        None,
        description="Equity percentage after the change.",
    )
    description: str = Field(
        ...,
        description="Description of the boundary change.",
    )
    requires_base_year_recalculation: bool = Field(
        default=False,
        description="Whether this change triggers base year recalculation.",
    )
    significance_pct: Optional[Decimal] = Field(
        None,
        description="Significance of the change as % of corporate emissions.",
    )
    created_at: datetime = Field(
        default_factory=_utcnow,
        description="When this change was recorded.",
    )

    @field_validator("change_type")
    @classmethod
    def _validate_change_type(cls, v: str) -> str:
        valid = {ct.value for ct in ChangeType}
        if v.upper() not in valid:
            logger.warning("Change type '%s' not standard; accepted.", v)
        return v.upper()

    @field_validator("equity_before", "equity_after", "significance_pct", mode="before")
    @classmethod
    def _coerce_decimal(cls, v: Any) -> Any:
        if v is not None:
            return Decimal(str(v))
        return v


class BoundaryDefinition(BaseModel):
    """Complete organisational boundary definition for a reporting year.

    Contains all site inclusions, exclusions, and aggregated
    boundary metadata. Can be locked for reporting finalisation.
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
    inclusions: List[BoundaryInclusion] = Field(
        default_factory=list,
        description="All site inclusions (and exclusions).",
    )
    changes: List[BoundaryChange] = Field(
        default_factory=list,
        description="Structural changes applied to this boundary.",
    )
    total_sites_included: int = Field(
        default=0,
        description="Count of included (non-excluded) sites.",
    )
    total_sites_excluded: int = Field(
        default=0,
        description="Count of excluded sites.",
    )
    avg_inclusion_pct: Decimal = Field(
        default=Decimal("0"),
        description="Average inclusion percentage across included sites.",
    )
    materiality_results: Dict[str, Any] = Field(
        default_factory=dict,
        description="Results of materiality assessments.",
    )
    is_locked: bool = Field(
        default=False,
        description="Whether the boundary is locked for reporting.",
    )
    locked_at: Optional[datetime] = Field(
        None,
        description="When the boundary was locked.",
    )
    locked_by: Optional[str] = Field(
        None,
        description="Who locked the boundary.",
    )
    notes: Optional[str] = Field(
        None,
        description="Additional notes.",
    )
    created_at: datetime = Field(
        default_factory=_utcnow,
        description="When the boundary was created.",
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


class MaterialityResult(BaseModel):
    """Result of a materiality assessment for a site."""
    model_config = ConfigDict(
        json_encoders={Decimal: str},
        validate_default=True,
    )

    site_id: str = Field(..., description="The assessed site.")
    site_emissions: Decimal = Field(
        ..., description="Total emissions for the site (tCO2e)."
    )
    corporate_total: Decimal = Field(
        ..., description="Total corporate emissions (tCO2e)."
    )
    materiality_pct: Decimal = Field(
        ..., description="Site emissions as percentage of corporate total."
    )
    threshold_pct: Decimal = Field(
        ..., description="Materiality threshold applied."
    )
    is_material: bool = Field(
        ..., description="Whether the site is considered material."
    )
    recommendation: str = Field(
        ..., description="Recommendation based on materiality."
    )
    provenance_hash: str = Field(
        default="", description="SHA-256 hash."
    )


class BoundaryComparison(BaseModel):
    """Comparison between two boundary definitions."""
    model_config = ConfigDict(
        json_encoders={Decimal: str},
        validate_default=True,
    )

    boundary_1_id: str = Field(
        ..., description="First boundary ID."
    )
    boundary_2_id: str = Field(
        ..., description="Second boundary ID."
    )
    year_1: int = Field(
        ..., description="First boundary year."
    )
    year_2: int = Field(
        ..., description="Second boundary year."
    )
    sites_added: List[str] = Field(
        default_factory=list,
        description="Sites in boundary 2 but not in boundary 1.",
    )
    sites_removed: List[str] = Field(
        default_factory=list,
        description="Sites in boundary 1 but not in boundary 2.",
    )
    sites_unchanged: List[str] = Field(
        default_factory=list,
        description="Sites present in both boundaries with same pct.",
    )
    sites_modified: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Sites present in both but with changed inclusion pct.",
    )
    inclusion_count_1: int = Field(
        default=0, description="Included sites in boundary 1."
    )
    inclusion_count_2: int = Field(
        default=0, description="Included sites in boundary 2."
    )
    net_change: int = Field(
        default=0, description="Net change in included sites."
    )
    provenance_hash: str = Field(
        default="", description="SHA-256 hash."
    )


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------


class SiteBoundaryEngine:
    """Manages organisational boundaries for multi-site GHG reporting.

    Implements the three GHG Protocol consolidation approaches,
    handles structural boundary changes, provides time-weighted
    consolidation for mid-year events, and materiality assessment.

    Attributes:
        _boundaries: Internal dict mapping boundary_id to BoundaryDefinition.
        _entities: Internal dict mapping entity_id to EntityOwnership.
        _change_log: Append-only audit log.

    Example:
        >>> engine = SiteBoundaryEngine()
        >>> boundary = engine.define_boundary(
        ...     year=2025,
        ...     approach="EQUITY_SHARE",
        ...     entities=[entity1, entity2],
        ...     sites={"site-001": "entity-001", "site-002": "entity-002"},
        ... )
        >>> assert boundary.total_sites_included == 2
    """

    def __init__(self) -> None:
        """Initialise the SiteBoundaryEngine with empty state."""
        self._boundaries: Dict[str, BoundaryDefinition] = {}
        self._entities: Dict[str, EntityOwnership] = {}
        self._change_log: List[Dict[str, Any]] = []
        logger.info("SiteBoundaryEngine v%s initialised.", _MODULE_VERSION)

    # ------------------------------------------------------------------
    # Entity Management
    # ------------------------------------------------------------------

    def register_entity(self, entity: EntityOwnership) -> EntityOwnership:
        """Register a legal entity for boundary calculations.

        Args:
            entity: The entity ownership record.

        Returns:
            The registered entity.
        """
        self._entities[entity.entity_id] = entity
        logger.info(
            "Entity '%s' (%s) registered with %.2f%% equity.",
            entity.entity_name,
            entity.entity_id,
            entity.ownership_pct,
        )
        return entity

    # ------------------------------------------------------------------
    # Boundary Definition
    # ------------------------------------------------------------------

    def define_boundary(
        self,
        year: int,
        approach: str,
        entities: List[EntityOwnership],
        sites: Dict[str, str],
        notes: Optional[str] = None,
    ) -> BoundaryDefinition:
        """Define an organisational boundary for a reporting year.

        Automatically determines inclusion percentages based on the
        chosen consolidation approach and entity ownership structures.

        Args:
            year: The reporting year.
            approach: Consolidation approach (EQUITY_SHARE,
                OPERATIONAL_CONTROL, FINANCIAL_CONTROL).
            entities: List of EntityOwnership records.
            sites: Dictionary mapping site_id to entity_id.
            notes: Optional notes.

        Returns:
            The created BoundaryDefinition.

        Raises:
            ValueError: If approach is invalid or entity not found.
        """
        approach_upper = approach.upper()
        valid_approaches = {ca.value for ca in ConsolidationApproach}
        if approach_upper not in valid_approaches:
            raise ValueError(
                f"Invalid approach '{approach}'. "
                f"Must be one of {sorted(valid_approaches)}."
            )

        logger.info(
            "Defining %s boundary for year %d with %d entities and %d sites.",
            approach_upper, year, len(entities), len(sites),
        )

        # Register entities
        entity_map: Dict[str, EntityOwnership] = {}
        for entity in entities:
            self.register_entity(entity)
            entity_map[entity.entity_id] = entity

        # Build inclusions
        inclusions: List[BoundaryInclusion] = []
        for site_id, entity_id in sites.items():
            if entity_id not in entity_map:
                raise ValueError(
                    f"Entity '{entity_id}' not found for site '{site_id}'."
                )
            entity = entity_map[entity_id]
            inclusion_pct = self._calculate_inclusion_pct(
                entity, approach_upper
            )

            inclusions.append(BoundaryInclusion(
                site_id=site_id,
                entity_id=entity_id,
                inclusion_pct=inclusion_pct,
                consolidation_approach=approach_upper,
            ))

        boundary = self._build_boundary(
            year=year,
            approach=approach_upper,
            inclusions=inclusions,
            notes=notes,
        )
        self._boundaries[boundary.boundary_id] = boundary

        self._change_log.append({
            "event": "BOUNDARY_DEFINED",
            "boundary_id": boundary.boundary_id,
            "year": year,
            "approach": approach_upper,
            "sites_included": boundary.total_sites_included,
            "timestamp": _utcnow().isoformat(),
        })

        logger.info(
            "Boundary '%s' defined: %d included, %d excluded.",
            boundary.boundary_id,
            boundary.total_sites_included,
            boundary.total_sites_excluded,
        )
        return boundary

    def _calculate_inclusion_pct(
        self,
        entity: EntityOwnership,
        approach: str,
    ) -> Decimal:
        """Calculate the inclusion percentage for an entity.

        Args:
            entity: The entity ownership record.
            approach: The consolidation approach.

        Returns:
            Inclusion percentage (0-100).
        """
        if approach == ConsolidationApproach.EQUITY_SHARE.value:
            return entity.ownership_pct

        elif approach == ConsolidationApproach.OPERATIONAL_CONTROL.value:
            if entity.has_operational_control:
                return Decimal("100")
            return Decimal("0")

        elif approach == ConsolidationApproach.FINANCIAL_CONTROL.value:
            if entity.has_financial_control:
                return Decimal("100")
            return Decimal("0")

        return Decimal("0")

    def _build_boundary(
        self,
        year: int,
        approach: str,
        inclusions: List[BoundaryInclusion],
        changes: Optional[List[BoundaryChange]] = None,
        notes: Optional[str] = None,
    ) -> BoundaryDefinition:
        """Build a BoundaryDefinition from inclusions.

        Computes aggregate metrics (counts, averages).

        Args:
            year: Reporting year.
            approach: Consolidation approach.
            inclusions: List of BoundaryInclusion records.
            changes: Optional boundary changes.
            notes: Optional notes.

        Returns:
            The constructed BoundaryDefinition.
        """
        included = [i for i in inclusions if not i.is_excluded]
        excluded = [i for i in inclusions if i.is_excluded]

        # Average inclusion pct among included sites
        if included:
            total_pct = sum(i.inclusion_pct for i in included)
            avg_pct = _round2(
                _safe_divide(total_pct, _decimal(len(included)))
            )
        else:
            avg_pct = Decimal("0")

        boundary = BoundaryDefinition(
            reporting_year=year,
            consolidation_approach=approach,
            inclusions=inclusions,
            changes=changes or [],
            total_sites_included=len(included),
            total_sites_excluded=len(excluded),
            avg_inclusion_pct=avg_pct,
            notes=notes,
        )
        boundary.provenance_hash = _compute_hash(boundary)
        return boundary

    # ------------------------------------------------------------------
    # Boundary Modification
    # ------------------------------------------------------------------

    def add_site_to_boundary(
        self,
        boundary: BoundaryDefinition,
        site_id: str,
        entity_id: str,
        inclusion_pct: Union[Decimal, str, int, float],
    ) -> BoundaryDefinition:
        """Add a site to an existing boundary.

        Args:
            boundary: The boundary to modify.
            site_id: The site to add.
            entity_id: The legal entity for the site.
            inclusion_pct: Inclusion percentage (0-100).

        Returns:
            The updated BoundaryDefinition.

        Raises:
            ValueError: If boundary is locked or site already exists.
        """
        if boundary.is_locked:
            raise ValueError(
                f"Boundary '{boundary.boundary_id}' is locked and "
                f"cannot be modified."
            )

        # Check for duplicates
        existing_ids = {i.site_id for i in boundary.inclusions}
        if site_id in existing_ids:
            raise ValueError(
                f"Site '{site_id}' already exists in boundary "
                f"'{boundary.boundary_id}'."
            )

        logger.info(
            "Adding site '%s' to boundary '%s' at %s%%.",
            site_id, boundary.boundary_id, inclusion_pct,
        )

        new_inclusion = BoundaryInclusion(
            site_id=site_id,
            entity_id=entity_id,
            inclusion_pct=_decimal(inclusion_pct),
            consolidation_approach=boundary.consolidation_approach,
        )

        updated_inclusions = list(boundary.inclusions) + [new_inclusion]
        updated = self._build_boundary(
            year=boundary.reporting_year,
            approach=boundary.consolidation_approach,
            inclusions=updated_inclusions,
            changes=boundary.changes,
            notes=boundary.notes,
        )
        # Preserve the original boundary_id
        updated = updated.model_copy(update={
            "boundary_id": boundary.boundary_id,
        })
        updated.provenance_hash = _compute_hash(updated)
        self._boundaries[updated.boundary_id] = updated

        self._change_log.append({
            "event": "SITE_ADDED_TO_BOUNDARY",
            "boundary_id": boundary.boundary_id,
            "site_id": site_id,
            "inclusion_pct": str(inclusion_pct),
            "timestamp": _utcnow().isoformat(),
        })

        return updated

    def exclude_site(
        self,
        boundary: BoundaryDefinition,
        site_id: str,
        reason: str,
        justification: Optional[str] = None,
    ) -> BoundaryDefinition:
        """Exclude a site from the boundary.

        Marks the site as excluded rather than removing it, preserving
        the audit trail.

        Args:
            boundary: The boundary to modify.
            site_id: The site to exclude.
            reason: Short reason for exclusion.
            justification: Detailed justification.

        Returns:
            The updated BoundaryDefinition.

        Raises:
            ValueError: If boundary is locked or site not found.
        """
        if boundary.is_locked:
            raise ValueError(
                f"Boundary '{boundary.boundary_id}' is locked."
            )

        logger.info(
            "Excluding site '%s' from boundary '%s': %s.",
            site_id, boundary.boundary_id, reason,
        )

        found = False
        updated_inclusions: List[BoundaryInclusion] = []
        for inclusion in boundary.inclusions:
            if inclusion.site_id == site_id:
                found = True
                excluded = inclusion.model_copy(update={
                    "is_excluded": True,
                    "exclusion_reason": reason,
                    "exclusion_justification": justification,
                })
                updated_inclusions.append(excluded)
            else:
                updated_inclusions.append(inclusion)

        if not found:
            raise ValueError(
                f"Site '{site_id}' not found in boundary "
                f"'{boundary.boundary_id}'."
            )

        updated = self._build_boundary(
            year=boundary.reporting_year,
            approach=boundary.consolidation_approach,
            inclusions=updated_inclusions,
            changes=boundary.changes,
            notes=boundary.notes,
        )
        updated = updated.model_copy(update={
            "boundary_id": boundary.boundary_id,
        })
        updated.provenance_hash = _compute_hash(updated)
        self._boundaries[updated.boundary_id] = updated

        self._change_log.append({
            "event": "SITE_EXCLUDED",
            "boundary_id": boundary.boundary_id,
            "site_id": site_id,
            "reason": reason,
            "timestamp": _utcnow().isoformat(),
        })

        return updated

    def apply_boundary_change(
        self,
        boundary: BoundaryDefinition,
        change: BoundaryChange,
    ) -> BoundaryDefinition:
        """Apply a structural boundary change.

        Handles acquisitions (adds sites), divestitures (excludes sites),
        equity changes (updates inclusion percentages), and other
        structural events.

        Args:
            boundary: The boundary to modify.
            change: The boundary change to apply.

        Returns:
            The updated BoundaryDefinition.

        Raises:
            ValueError: If boundary is locked.
        """
        if boundary.is_locked:
            raise ValueError(
                f"Boundary '{boundary.boundary_id}' is locked."
            )

        logger.info(
            "Applying %s change to boundary '%s' affecting %d site(s).",
            change.change_type,
            boundary.boundary_id,
            len(change.affected_site_ids),
        )

        updated_inclusions = list(boundary.inclusions)
        change_type = change.change_type.upper()

        if change_type == ChangeType.ACQUISITION.value:
            # Add new sites from acquisition
            existing_ids = {i.site_id for i in updated_inclusions}
            for site_id in change.affected_site_ids:
                if site_id not in existing_ids:
                    equity_pct = change.equity_after or Decimal("100")
                    new_inclusion = BoundaryInclusion(
                        site_id=site_id,
                        entity_id="acquired",
                        inclusion_pct=equity_pct,
                        consolidation_approach=boundary.consolidation_approach,
                        effective_from=change.effective_date,
                    )
                    updated_inclusions.append(new_inclusion)

        elif change_type == ChangeType.DIVESTITURE.value:
            # Exclude divested sites
            for i, incl in enumerate(updated_inclusions):
                if incl.site_id in change.affected_site_ids:
                    updated_inclusions[i] = incl.model_copy(update={
                        "is_excluded": True,
                        "exclusion_reason": f"Divested on {change.effective_date}",
                        "exclusion_justification": change.description,
                        "effective_to": change.effective_date,
                    })

        elif change_type == ChangeType.EQUITY_CHANGE.value:
            # Update equity percentages
            new_pct = change.equity_after or Decimal("0")
            for i, incl in enumerate(updated_inclusions):
                if incl.site_id in change.affected_site_ids:
                    updated_inclusions[i] = incl.model_copy(update={
                        "inclusion_pct": new_pct,
                    })

        elif change_type in (
            ChangeType.MERGER.value,
            ChangeType.INSOURCING.value,
        ):
            # Add sites at 100% inclusion
            existing_ids = {i.site_id for i in updated_inclusions}
            for site_id in change.affected_site_ids:
                if site_id not in existing_ids:
                    new_inclusion = BoundaryInclusion(
                        site_id=site_id,
                        entity_id="merged",
                        inclusion_pct=Decimal("100"),
                        consolidation_approach=boundary.consolidation_approach,
                        effective_from=change.effective_date,
                    )
                    updated_inclusions.append(new_inclusion)

        elif change_type in (
            ChangeType.OUTSOURCING.value,
            ChangeType.CLOSURE.value,
        ):
            # Exclude outsourced/closed sites
            for i, incl in enumerate(updated_inclusions):
                if incl.site_id in change.affected_site_ids:
                    updated_inclusions[i] = incl.model_copy(update={
                        "is_excluded": True,
                        "exclusion_reason": (
                            f"{change_type.title()} on {change.effective_date}"
                        ),
                        "exclusion_justification": change.description,
                        "effective_to": change.effective_date,
                    })

        # Record change
        updated_changes = list(boundary.changes) + [change]

        updated = self._build_boundary(
            year=boundary.reporting_year,
            approach=boundary.consolidation_approach,
            inclusions=updated_inclusions,
            changes=updated_changes,
            notes=boundary.notes,
        )
        updated = updated.model_copy(update={
            "boundary_id": boundary.boundary_id,
        })
        updated.provenance_hash = _compute_hash(updated)
        self._boundaries[updated.boundary_id] = updated

        self._change_log.append({
            "event": "BOUNDARY_CHANGE_APPLIED",
            "boundary_id": boundary.boundary_id,
            "change_id": change.change_id,
            "change_type": change_type,
            "affected_sites": change.affected_site_ids,
            "timestamp": _utcnow().isoformat(),
        })

        return updated

    # ------------------------------------------------------------------
    # Time-Weighted Consolidation
    # ------------------------------------------------------------------

    def time_weighted_consolidation(
        self,
        inclusion_pct: Union[Decimal, str, int, float],
        change_date: date,
        year_start: date,
        year_end: date,
    ) -> Decimal:
        """Calculate time-weighted inclusion percentage.

        When a boundary change occurs mid-year, the inclusion percentage
        is prorated based on the number of days the site was included.

        Formula:
            days_included = min(year_end, effective_to) - max(year_start, change_date) + 1
            total_days = year_end - year_start + 1
            time_weighted_pct = inclusion_pct * (days_included / total_days)

        Args:
            inclusion_pct: Base inclusion percentage.
            change_date: Date the change takes effect.
            year_start: Start of reporting year.
            year_end: End of reporting year.

        Returns:
            Time-weighted inclusion percentage rounded to 4 decimals.

        Raises:
            ValueError: If change_date is outside the reporting year.
        """
        pct = _decimal(inclusion_pct)

        if change_date > year_end:
            # Change is after reporting year: no inclusion
            return Decimal("0")

        if change_date <= year_start:
            # Change is before reporting year: full inclusion
            return pct

        effective_start = max(change_date, year_start)
        days_included = (year_end - effective_start).days + 1
        total_days = (year_end - year_start).days + 1

        if total_days <= 0:
            return Decimal("0")

        time_weight = _safe_divide(
            _decimal(days_included), _decimal(total_days)
        )
        weighted_pct = _round4(pct * time_weight)

        logger.debug(
            "Time-weighted: %s%% * (%d/%d days) = %s%%.",
            pct, days_included, total_days, weighted_pct,
        )
        return weighted_pct

    def apply_time_weighting_to_boundary(
        self,
        boundary: BoundaryDefinition,
    ) -> BoundaryDefinition:
        """Apply time weighting to all inclusions with effective dates.

        For each inclusion that has an effective_from date within the
        reporting year, calculates a time-weighted inclusion pct.

        Args:
            boundary: The boundary to update.

        Returns:
            Updated BoundaryDefinition with time_weighted_pct set.
        """
        year_start = date(boundary.reporting_year, 1, 1)
        year_end = date(boundary.reporting_year, 12, 31)

        updated_inclusions: List[BoundaryInclusion] = []
        for incl in boundary.inclusions:
            if incl.effective_from and not incl.is_excluded:
                tw_pct = self.time_weighted_consolidation(
                    incl.inclusion_pct,
                    incl.effective_from,
                    year_start,
                    year_end,
                )
                updated_inclusions.append(incl.model_copy(update={
                    "time_weighted_pct": tw_pct,
                }))
            elif incl.effective_to and not incl.is_excluded:
                # Site was included until effective_to
                days_included = (
                    min(incl.effective_to, year_end) - year_start
                ).days + 1
                total_days = (year_end - year_start).days + 1
                if days_included > 0 and total_days > 0:
                    tw = _safe_divide(
                        _decimal(days_included), _decimal(total_days)
                    )
                    tw_pct = _round4(incl.inclusion_pct * tw)
                else:
                    tw_pct = Decimal("0")
                updated_inclusions.append(incl.model_copy(update={
                    "time_weighted_pct": tw_pct,
                }))
            else:
                updated_inclusions.append(incl.model_copy(update={
                    "time_weighted_pct": incl.inclusion_pct,
                }))

        updated = self._build_boundary(
            year=boundary.reporting_year,
            approach=boundary.consolidation_approach,
            inclusions=updated_inclusions,
            changes=boundary.changes,
            notes=boundary.notes,
        )
        updated = updated.model_copy(update={
            "boundary_id": boundary.boundary_id,
        })
        updated.provenance_hash = _compute_hash(updated)
        self._boundaries[updated.boundary_id] = updated

        logger.info(
            "Time weighting applied to boundary '%s'.",
            boundary.boundary_id,
        )
        return updated

    # ------------------------------------------------------------------
    # Materiality Assessment
    # ------------------------------------------------------------------

    def assess_materiality(
        self,
        site_emissions: Union[Decimal, str, int, float],
        corporate_total: Union[Decimal, str, int, float],
        threshold: Union[Decimal, str, int, float, None] = None,
        site_id: str = "unknown",
    ) -> MaterialityResult:
        """Assess materiality of a site relative to corporate total.

        Per GHG Protocol guidance, sites can be excluded if their
        emissions are below a materiality threshold (typically 1-5%
        of total corporate emissions).

        Args:
            site_emissions: Total emissions for the site (tCO2e).
            corporate_total: Total corporate emissions (tCO2e).
            threshold: Materiality threshold percentage. Defaults to 5%.
            site_id: Optional site ID for the result.

        Returns:
            MaterialityResult with assessment.
        """
        site_em = _decimal(site_emissions)
        corp_total = _decimal(corporate_total)
        thresh = _decimal(
            threshold if threshold is not None
            else DEFAULT_MATERIALITY_THRESHOLD_PCT
        )

        materiality_pct = _round4(
            _safe_divide(site_em, corp_total) * Decimal("100")
        )
        is_material = materiality_pct >= thresh

        if is_material:
            recommendation = (
                f"Site is MATERIAL ({materiality_pct}% >= {thresh}% threshold). "
                f"Must be included in the reporting boundary."
            )
        else:
            recommendation = (
                f"Site is IMMATERIAL ({materiality_pct}% < {thresh}% threshold). "
                f"May be excluded with documented justification per GHG Protocol."
            )

        result = MaterialityResult(
            site_id=site_id,
            site_emissions=site_em,
            corporate_total=corp_total,
            materiality_pct=materiality_pct,
            threshold_pct=thresh,
            is_material=is_material,
            recommendation=recommendation,
        )
        result.provenance_hash = _compute_hash(result)

        logger.info(
            "Materiality for site '%s': %s%% (threshold=%s%%, material=%s).",
            site_id, materiality_pct, thresh, is_material,
        )
        return result

    def assess_all_sites_materiality(
        self,
        site_emissions_map: Dict[str, Union[Decimal, str, int, float]],
        corporate_total: Union[Decimal, str, int, float],
        threshold: Union[Decimal, str, int, float, None] = None,
    ) -> Dict[str, MaterialityResult]:
        """Assess materiality for all sites.

        Args:
            site_emissions_map: Dict mapping site_id to emissions.
            corporate_total: Total corporate emissions.
            threshold: Materiality threshold percentage.

        Returns:
            Dict mapping site_id to MaterialityResult.
        """
        results: Dict[str, MaterialityResult] = {}
        for site_id, emissions in site_emissions_map.items():
            results[site_id] = self.assess_materiality(
                site_emissions=emissions,
                corporate_total=corporate_total,
                threshold=threshold,
                site_id=site_id,
            )
        return results

    # ------------------------------------------------------------------
    # Boundary Locking
    # ------------------------------------------------------------------

    def lock_boundary(
        self,
        boundary: BoundaryDefinition,
        locked_by: Optional[str] = None,
    ) -> BoundaryDefinition:
        """Lock a boundary to prevent further modifications.

        Locking is typically done after all reviews are complete and
        the boundary is finalised for reporting.

        Args:
            boundary: The boundary to lock.
            locked_by: User who is locking the boundary.

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
            "Locking boundary '%s' for year %d.",
            boundary.boundary_id,
            boundary.reporting_year,
        )

        now = _utcnow()
        updated = boundary.model_copy(update={
            "is_locked": True,
            "locked_at": now,
            "locked_by": locked_by,
        })
        updated.provenance_hash = _compute_hash(updated)
        self._boundaries[updated.boundary_id] = updated

        self._change_log.append({
            "event": "BOUNDARY_LOCKED",
            "boundary_id": boundary.boundary_id,
            "locked_by": locked_by,
            "timestamp": now.isoformat(),
        })

        return updated

    # ------------------------------------------------------------------
    # Boundary Comparison
    # ------------------------------------------------------------------

    def compare_boundaries(
        self,
        boundary1: BoundaryDefinition,
        boundary2: BoundaryDefinition,
    ) -> BoundaryComparison:
        """Compare two boundary definitions.

        Identifies sites added, removed, unchanged, and modified
        between two boundaries (typically year-over-year).

        Args:
            boundary1: First boundary (typically earlier year).
            boundary2: Second boundary (typically later year).

        Returns:
            BoundaryComparison with detailed diff.
        """
        logger.info(
            "Comparing boundary '%s' (year %d) with '%s' (year %d).",
            boundary1.boundary_id,
            boundary1.reporting_year,
            boundary2.boundary_id,
            boundary2.reporting_year,
        )

        # Build maps of included sites
        map1: Dict[str, Decimal] = {}
        for incl in boundary1.inclusions:
            if not incl.is_excluded:
                map1[incl.site_id] = incl.inclusion_pct

        map2: Dict[str, Decimal] = {}
        for incl in boundary2.inclusions:
            if not incl.is_excluded:
                map2[incl.site_id] = incl.inclusion_pct

        set1 = set(map1.keys())
        set2 = set(map2.keys())

        sites_added = sorted(set2 - set1)
        sites_removed = sorted(set1 - set2)
        common = set1 & set2

        sites_unchanged: List[str] = []
        sites_modified: List[Dict[str, Any]] = []

        for site_id in sorted(common):
            pct1 = map1[site_id]
            pct2 = map2[site_id]
            if pct1 == pct2:
                sites_unchanged.append(site_id)
            else:
                sites_modified.append({
                    "site_id": site_id,
                    "pct_before": str(pct1),
                    "pct_after": str(pct2),
                    "change": str(pct2 - pct1),
                })

        result = BoundaryComparison(
            boundary_1_id=boundary1.boundary_id,
            boundary_2_id=boundary2.boundary_id,
            year_1=boundary1.reporting_year,
            year_2=boundary2.reporting_year,
            sites_added=sites_added,
            sites_removed=sites_removed,
            sites_unchanged=sites_unchanged,
            sites_modified=sites_modified,
            inclusion_count_1=len(map1),
            inclusion_count_2=len(map2),
            net_change=len(map2) - len(map1),
        )
        result.provenance_hash = _compute_hash(result)

        logger.info(
            "Comparison: +%d added, -%d removed, %d unchanged, %d modified.",
            len(sites_added),
            len(sites_removed),
            len(sites_unchanged),
            len(sites_modified),
        )
        return result

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

    def get_boundaries_for_year(self, year: int) -> List[BoundaryDefinition]:
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

    def get_included_sites(
        self, boundary: BoundaryDefinition,
    ) -> List[BoundaryInclusion]:
        """Get all included (non-excluded) sites for a boundary.

        Args:
            boundary: The boundary definition.

        Returns:
            List of non-excluded BoundaryInclusion records.
        """
        return [i for i in boundary.inclusions if not i.is_excluded]

    def get_excluded_sites(
        self, boundary: BoundaryDefinition,
    ) -> List[BoundaryInclusion]:
        """Get all excluded sites for a boundary.

        Args:
            boundary: The boundary definition.

        Returns:
            List of excluded BoundaryInclusion records.
        """
        return [i for i in boundary.inclusions if i.is_excluded]

    def get_change_log(self) -> List[Dict[str, Any]]:
        """Return the complete change log.

        Returns:
            List of change log entries.
        """
        return list(self._change_log)
