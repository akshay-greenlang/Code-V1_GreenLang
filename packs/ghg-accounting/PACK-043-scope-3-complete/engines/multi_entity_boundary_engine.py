# -*- coding: utf-8 -*-
"""
MultiEntityBoundaryEngine - PACK-043 Scope 3 Complete Pack Engine 3
=====================================================================

Manages Scope 3 organisational and operational boundaries for corporate
groups, joint ventures, associates, and franchises.  Implements three
GHG Protocol consolidation approaches (equity share, operational control,
financial control), handles inter-company elimination to prevent
double-counting, and supports mid-year boundary changes from acquisitions
or divestitures through time-weighted consolidation.

Consolidation Approaches:
    Equity Share:        Entity emissions * ownership percentage
    Operational Control: 100% if reporting org has operational control
    Financial Control:   100% if reporting org has financial control

Calculation Methodology:
    Proportional Consolidation:
        E_group = sum(E_entity_i * share_i)
        where share_i depends on consolidation approach

    Inter-company Elimination:
        Identify parent-subsidiary or sibling Scope 3 overlaps.
        Eliminate supplier-customer emissions between group entities.
        E_adjusted = E_gross - E_intercompany_overlap

    Time-Weighted Consolidation (mid-year changes):
        E_entity = E_annual * (days_in_boundary / days_in_year)

    Influence Assessment (Control Test):
        Operational: manages day-to-day operations
        Financial: directs financial and operating policies
        Equity: proportional to ownership interest

Regulatory References:
    - GHG Protocol Corporate Standard, Chapter 3 (Setting Organisational
      Boundaries)
    - GHG Protocol Scope 3 Standard, Chapter 5 (Identifying Scope 3
      Emissions)
    - ESRS 1.62-1.69 (Consolidation perimeter for sustainability
      reporting)
    - IFRS S2 Climate-Related Disclosures (consolidation consistent with
      financial statements)
    - ISO 14064-1:2018 Section 5.1 (Organisational boundaries)

Zero-Hallucination:
    - Ownership shares and control flags from user-provided entity data
    - All consolidation uses deterministic arithmetic
    - Time-weighting uses calendar day counts
    - No LLM involvement in any calculation path
    - SHA-256 provenance hash on every result

Author:  GreenLang Platform Team
Date:    March 2026
Pack:    PACK-043 Scope 3 Complete
Engine:  3 of 10
Status:  Production Ready
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
import uuid
from datetime import date, datetime, timezone
from decimal import Decimal, ROUND_HALF_UP, InvalidOperation
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field, field_validator

logger = logging.getLogger(__name__)

_MODULE_VERSION: str = "43.0.0"


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
    """Compute a deterministic SHA-256 hash of arbitrary data."""
    if hasattr(data, "model_dump"):
        serialisable = data.model_dump(mode="json")
    elif isinstance(data, dict):
        serialisable = data
    else:
        serialisable = str(data)
    if isinstance(serialisable, dict):
        serialisable = {
            k: v for k, v in serialisable.items()
            if k not in ("calculated_at", "processing_time_ms", "provenance_hash")
        }
    raw = json.dumps(serialisable, sort_keys=True, default=str)
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


def _round_val(value: Decimal, places: int = 6) -> Decimal:
    """Round a Decimal to *places* using ROUND_HALF_UP."""
    quantize_str = "0." + "0" * places
    return value.quantize(Decimal(quantize_str), rounding=ROUND_HALF_UP)


def _days_in_year(year: int) -> int:
    """Return number of days in a year (handles leap years)."""
    if year % 4 == 0 and (year % 100 != 0 or year % 400 == 0):
        return 366
    return 365


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class ConsolidationApproach(str, Enum):
    """GHG Protocol consolidation approach.

    EQUITY_SHARE:       Account for emissions proportional to equity share.
    OPERATIONAL_CONTROL: 100% of emissions for operationally controlled entities.
    FINANCIAL_CONTROL:   100% of emissions for financially controlled entities.
    """
    EQUITY_SHARE = "equity_share"
    OPERATIONAL_CONTROL = "operational_control"
    FINANCIAL_CONTROL = "financial_control"


class EntityType(str, Enum):
    """Type of entity in the corporate group.

    PARENT:      Ultimate parent / reporting entity.
    SUBSIDIARY:  Majority-owned subsidiary.
    JV:          Joint venture.
    ASSOCIATE:   Minority-owned associate.
    FRANCHISE:   Franchisee or licensed operation.
    INVESTMENT:  Portfolio investment (financial only).
    """
    PARENT = "parent"
    SUBSIDIARY = "subsidiary"
    JV = "joint_venture"
    ASSOCIATE = "associate"
    FRANCHISE = "franchise"
    INVESTMENT = "investment"


class ControlType(str, Enum):
    """Type of control over an entity.

    OPERATIONAL:   Day-to-day operational control.
    FINANCIAL:     Financial and operating policy control.
    SHARED:        Shared/joint control.
    SIGNIFICANT:   Significant influence but not control.
    NONE:          No control or influence.
    """
    OPERATIONAL = "operational"
    FINANCIAL = "financial"
    SHARED = "shared"
    SIGNIFICANT = "significant_influence"
    NONE = "none"


class BoundaryChangeType(str, Enum):
    """Type of boundary change event.

    ACQUISITION:  Acquiring a new entity.
    DIVESTITURE:  Divesting/selling an entity.
    MERGER:       Merging two entities.
    RESTRUCTURE:  Internal restructuring.
    JV_FORMED:    New joint venture formed.
    JV_DISSOLVED: Joint venture dissolved.
    """
    ACQUISITION = "acquisition"
    DIVESTITURE = "divestiture"
    MERGER = "merger"
    RESTRUCTURE = "restructure"
    JV_FORMED = "jv_formed"
    JV_DISSOLVED = "jv_dissolved"


class ConsolidationStatus(str, Enum):
    """Status of consolidation.

    COMPLETE: All entities consolidated.
    PARTIAL:  Some entities could not be consolidated.
    ERROR:    Consolidation failed.
    """
    COMPLETE = "complete"
    PARTIAL = "partial"
    ERROR = "error"


# ---------------------------------------------------------------------------
# Inter-company Elimination Rules
# ---------------------------------------------------------------------------
# Defines which Scope 3 category pairs can create double-counting
# when both entities are in the consolidation boundary.

INTERCOMPANY_OVERLAP_RULES: List[Dict[str, Any]] = [
    {
        "rule_id": "ICE-001",
        "name": "Upstream transport parent-subsidiary",
        "description": "Cat 4 of parent overlaps with Cat 9 of subsidiary (or vice versa)",
        "seller_category": 4,
        "buyer_category": 9,
        "elimination_method": "remove_buyer_side",
    },
    {
        "rule_id": "ICE-002",
        "name": "Purchased goods intra-group",
        "description": "Cat 1 of buyer overlaps with Cat 10/11/12 of seller within group",
        "seller_category": 10,
        "buyer_category": 1,
        "elimination_method": "remove_buyer_side",
    },
    {
        "rule_id": "ICE-003",
        "name": "Purchased goods intra-group (use phase)",
        "description": "Cat 1 of buyer overlaps with Cat 11 of seller within group",
        "seller_category": 11,
        "buyer_category": 1,
        "elimination_method": "remove_buyer_side",
    },
    {
        "rule_id": "ICE-004",
        "name": "Capital goods intra-group",
        "description": "Cat 2 of buyer overlaps with seller's downstream emissions",
        "seller_category": 10,
        "buyer_category": 2,
        "elimination_method": "remove_buyer_side",
    },
    {
        "rule_id": "ICE-005",
        "name": "Waste processing intra-group",
        "description": "Cat 5 of generator overlaps with Cat 1 of waste processor",
        "seller_category": 1,
        "buyer_category": 5,
        "elimination_method": "remove_seller_side",
    },
    {
        "rule_id": "ICE-006",
        "name": "Franchise emissions",
        "description": "Cat 14 of franchisor overlaps with Cat 1 of franchisee",
        "seller_category": 1,
        "buyer_category": 14,
        "elimination_method": "remove_buyer_side",
    },
    {
        "rule_id": "ICE-007",
        "name": "Investment emissions",
        "description": "Cat 15 of investor overlaps with investee Scope 1+2",
        "seller_category": 0,  # Scope 1+2
        "buyer_category": 15,
        "elimination_method": "remove_buyer_side",
    },
    {
        "rule_id": "ICE-008",
        "name": "Leased assets",
        "description": "Cat 8 of lessee overlaps with Cat 13 of lessor",
        "seller_category": 13,
        "buyer_category": 8,
        "elimination_method": "remove_lower_confidence",
    },
]


# ---------------------------------------------------------------------------
# Pydantic Models -- Input
# ---------------------------------------------------------------------------


class EntityEmissions(BaseModel):
    """Emissions data for a single entity.

    Attributes:
        scope1_tco2e: Scope 1 emissions.
        scope2_location_tco2e: Scope 2 location-based.
        scope2_market_tco2e: Scope 2 market-based.
        scope3_by_category: Scope 3 emissions by category (1-15).
        total_scope3_tco2e: Total Scope 3 (auto-summed if zero).
    """
    scope1_tco2e: Decimal = Field(default=Decimal("0"), ge=0, description="Scope 1")
    scope2_location_tco2e: Decimal = Field(
        default=Decimal("0"), ge=0, description="Scope 2 location"
    )
    scope2_market_tco2e: Decimal = Field(
        default=Decimal("0"), ge=0, description="Scope 2 market"
    )
    scope3_by_category: Dict[int, Decimal] = Field(
        default_factory=dict, description="Scope 3 by category"
    )
    total_scope3_tco2e: Decimal = Field(
        default=Decimal("0"), ge=0, description="Total Scope 3"
    )


class Entity(BaseModel):
    """An entity in the corporate hierarchy.

    Attributes:
        entity_id: Unique entity identifier.
        entity_name: Entity name.
        entity_type: Entity archetype.
        parent_entity_id: Parent entity ID (empty for top-level).
        country: Country code.
        equity_share_pct: Ownership share (0-100).
        has_operational_control: Whether reporting org has operational control.
        has_financial_control: Whether reporting org has financial control.
        control_type: Type of control.
        emissions: Entity emissions data.
        boundary_start_date: Date entity entered boundary.
        boundary_end_date: Date entity left boundary (None if still in).
        is_active: Whether entity is currently in boundary.
        intercompany_relationships: IDs of entities with commercial relationships.
    """
    entity_id: str = Field(default_factory=_new_uuid, description="Entity ID")
    entity_name: str = Field(default="", description="Entity name")
    entity_type: EntityType = Field(
        default=EntityType.SUBSIDIARY, description="Type"
    )
    parent_entity_id: str = Field(default="", description="Parent ID")
    country: str = Field(default="", max_length=2, description="Country")
    equity_share_pct: Decimal = Field(
        default=Decimal("100"), ge=0, le=100, description="Equity share %"
    )
    has_operational_control: bool = Field(
        default=True, description="Operational control"
    )
    has_financial_control: bool = Field(
        default=True, description="Financial control"
    )
    control_type: ControlType = Field(
        default=ControlType.OPERATIONAL, description="Control type"
    )
    emissions: EntityEmissions = Field(
        default_factory=EntityEmissions, description="Emissions"
    )
    boundary_start_date: Optional[date] = Field(
        default=None, description="Boundary start date"
    )
    boundary_end_date: Optional[date] = Field(
        default=None, description="Boundary end date"
    )
    is_active: bool = Field(default=True, description="Active in boundary")
    intercompany_relationships: List[str] = Field(
        default_factory=list, description="Related entity IDs"
    )


class EntityHierarchy(BaseModel):
    """Complete entity hierarchy for a corporate group.

    Attributes:
        group_id: Group identifier.
        group_name: Group name.
        reporting_year: Reporting year.
        reporting_entity_id: ID of the reporting entity (parent).
        entities: All entities in the group.
    """
    group_id: str = Field(default_factory=_new_uuid, description="Group ID")
    group_name: str = Field(default="", description="Group name")
    reporting_year: int = Field(default=2025, description="Year")
    reporting_entity_id: str = Field(default="", description="Reporting entity ID")
    entities: List[Entity] = Field(
        default_factory=list, description="Entities"
    )


class IntercompanyRelationship(BaseModel):
    """Commercial relationship between two group entities.

    Attributes:
        seller_entity_id: Selling entity.
        buyer_entity_id: Buying entity.
        transaction_type: Type of transaction.
        annual_value_eur: Annual transaction value.
        estimated_emissions_tco2e: Estimated emissions from transaction.
        affected_categories_seller: Seller's Scope 3 categories affected.
        affected_categories_buyer: Buyer's Scope 3 categories affected.
    """
    seller_entity_id: str = Field(default="", description="Seller ID")
    buyer_entity_id: str = Field(default="", description="Buyer ID")
    transaction_type: str = Field(
        default="goods_services", description="Transaction type"
    )
    annual_value_eur: Decimal = Field(
        default=Decimal("0"), ge=0, description="Annual value EUR"
    )
    estimated_emissions_tco2e: Decimal = Field(
        default=Decimal("0"), ge=0, description="Estimated tCO2e"
    )
    affected_categories_seller: List[int] = Field(
        default_factory=list, description="Seller categories"
    )
    affected_categories_buyer: List[int] = Field(
        default_factory=list, description="Buyer categories"
    )


class BoundaryChangeEvent(BaseModel):
    """A boundary change event (acquisition, divestiture, etc.).

    Attributes:
        event_id: Event identifier.
        change_type: Type of boundary change.
        entity_id: Entity affected.
        effective_date: Effective date of the change.
        equity_share_before: Equity share before event.
        equity_share_after: Equity share after event.
        description: Event description.
    """
    event_id: str = Field(default_factory=_new_uuid, description="Event ID")
    change_type: BoundaryChangeType = Field(..., description="Change type")
    entity_id: str = Field(default="", description="Entity ID")
    effective_date: date = Field(..., description="Effective date")
    equity_share_before: Decimal = Field(
        default=Decimal("0"), ge=0, le=100, description="Share before"
    )
    equity_share_after: Decimal = Field(
        default=Decimal("0"), ge=0, le=100, description="Share after"
    )
    description: str = Field(default="", description="Description")


# ---------------------------------------------------------------------------
# Pydantic Models -- Output
# ---------------------------------------------------------------------------


class EntityConsolidated(BaseModel):
    """Consolidated emissions for a single entity.

    Attributes:
        entity_id: Entity identifier.
        entity_name: Entity name.
        entity_type: Entity type.
        consolidation_share: Share applied (0-1).
        time_weight: Time weighting factor (0-1).
        gross_scope3_tco2e: Gross Scope 3 before share application.
        consolidated_scope3_tco2e: Scope 3 after share and time weighting.
        consolidated_scope3_by_category: Per-category consolidated Scope 3.
        consolidated_scope1_tco2e: Consolidated Scope 1.
        consolidated_scope2_tco2e: Consolidated Scope 2.
        included_in_boundary: Whether included in boundary.
        notes: Consolidation notes.
    """
    entity_id: str = Field(default="", description="Entity ID")
    entity_name: str = Field(default="", description="Name")
    entity_type: EntityType = Field(
        default=EntityType.SUBSIDIARY, description="Type"
    )
    consolidation_share: Decimal = Field(
        default=Decimal("1"), description="Share"
    )
    time_weight: Decimal = Field(default=Decimal("1"), description="Time weight")
    gross_scope3_tco2e: Decimal = Field(
        default=Decimal("0"), description="Gross Scope 3"
    )
    consolidated_scope3_tco2e: Decimal = Field(
        default=Decimal("0"), description="Consolidated Scope 3"
    )
    consolidated_scope3_by_category: Dict[int, Decimal] = Field(
        default_factory=dict, description="Scope 3 by category"
    )
    consolidated_scope1_tco2e: Decimal = Field(
        default=Decimal("0"), description="Consolidated Scope 1"
    )
    consolidated_scope2_tco2e: Decimal = Field(
        default=Decimal("0"), description="Consolidated Scope 2"
    )
    included_in_boundary: bool = Field(default=True, description="In boundary")
    notes: List[str] = Field(default_factory=list, description="Notes")


class EliminationEntry(BaseModel):
    """A single inter-company elimination entry.

    Attributes:
        rule_id: Elimination rule applied.
        seller_entity_id: Selling entity.
        buyer_entity_id: Buying entity.
        category_eliminated: Category where elimination applied.
        eliminated_tco2e: Emissions eliminated.
        method: Elimination method used.
        description: Description of elimination.
    """
    rule_id: str = Field(default="", description="Rule ID")
    seller_entity_id: str = Field(default="", description="Seller")
    buyer_entity_id: str = Field(default="", description="Buyer")
    category_eliminated: int = Field(default=0, description="Category")
    eliminated_tco2e: Decimal = Field(
        default=Decimal("0"), description="Eliminated tCO2e"
    )
    method: str = Field(default="", description="Method")
    description: str = Field(default="", description="Description")


class BoundaryDefinition(BaseModel):
    """Organisational boundary definition result.

    Attributes:
        boundary_id: Unique boundary identifier.
        group_id: Group identifier.
        group_name: Group name.
        approach: Consolidation approach.
        reporting_year: Reporting year.
        entities_in_boundary: Number of entities included.
        entities_excluded: Number of entities excluded.
        entity_details: Per-entity consolidation details.
        total_equity_weighted_coverage: Coverage by equity share.
        notes: Boundary definition notes.
    """
    boundary_id: str = Field(default_factory=_new_uuid, description="Boundary ID")
    group_id: str = Field(default="", description="Group ID")
    group_name: str = Field(default="", description="Group name")
    approach: ConsolidationApproach = Field(
        default=ConsolidationApproach.OPERATIONAL_CONTROL, description="Approach"
    )
    reporting_year: int = Field(default=2025, description="Year")
    entities_in_boundary: int = Field(default=0, description="In boundary")
    entities_excluded: int = Field(default=0, description="Excluded")
    entity_details: List[Dict[str, Any]] = Field(
        default_factory=list, description="Entity details"
    )
    total_equity_weighted_coverage: Decimal = Field(
        default=Decimal("0"), description="Equity coverage %"
    )
    notes: List[str] = Field(default_factory=list, description="Notes")


class ConsolidationResult(BaseModel):
    """Complete consolidation result.

    Attributes:
        consolidation_id: Unique identifier.
        group_id: Group identifier.
        group_name: Group name.
        approach: Consolidation approach used.
        reporting_year: Reporting year.
        entity_results: Per-entity consolidated results.
        total_scope1_tco2e: Group Scope 1.
        total_scope2_tco2e: Group Scope 2.
        total_scope3_gross_tco2e: Group Scope 3 before elimination.
        intercompany_eliminations: Elimination entries.
        total_eliminated_tco2e: Total eliminated.
        total_scope3_net_tco2e: Group Scope 3 after elimination.
        scope3_by_category: Group Scope 3 by category (net).
        boundary_definition: Boundary definition.
        entities_consolidated: Number of entities.
        boundary_changes: Boundary change events applied.
        warnings: Any warnings.
        status: Consolidation status.
        calculated_at: Timestamp.
        processing_time_ms: Processing duration.
        provenance_hash: SHA-256 hash.
    """
    consolidation_id: str = Field(
        default_factory=_new_uuid, description="Consolidation ID"
    )
    group_id: str = Field(default="", description="Group ID")
    group_name: str = Field(default="", description="Group name")
    approach: ConsolidationApproach = Field(
        default=ConsolidationApproach.OPERATIONAL_CONTROL, description="Approach"
    )
    reporting_year: int = Field(default=2025, description="Year")
    entity_results: List[EntityConsolidated] = Field(
        default_factory=list, description="Entity results"
    )
    total_scope1_tco2e: Decimal = Field(
        default=Decimal("0"), description="Group Scope 1"
    )
    total_scope2_tco2e: Decimal = Field(
        default=Decimal("0"), description="Group Scope 2"
    )
    total_scope3_gross_tco2e: Decimal = Field(
        default=Decimal("0"), description="Gross Scope 3"
    )
    intercompany_eliminations: List[EliminationEntry] = Field(
        default_factory=list, description="Eliminations"
    )
    total_eliminated_tco2e: Decimal = Field(
        default=Decimal("0"), description="Total eliminated"
    )
    total_scope3_net_tco2e: Decimal = Field(
        default=Decimal("0"), description="Net Scope 3"
    )
    scope3_by_category: Dict[int, Decimal] = Field(
        default_factory=dict, description="Scope 3 by category"
    )
    boundary_definition: Optional[BoundaryDefinition] = Field(
        default=None, description="Boundary"
    )
    entities_consolidated: int = Field(default=0, description="Entities count")
    boundary_changes: List[Dict[str, Any]] = Field(
        default_factory=list, description="Boundary changes"
    )
    warnings: List[str] = Field(default_factory=list, description="Warnings")
    status: ConsolidationStatus = Field(
        default=ConsolidationStatus.COMPLETE, description="Status"
    )
    calculated_at: datetime = Field(default_factory=_utcnow, description="Timestamp")
    processing_time_ms: Decimal = Field(
        default=Decimal("0"), description="Processing ms"
    )
    provenance_hash: str = Field(default="", description="SHA-256 hash")


class InfluenceAssessment(BaseModel):
    """Result of a control/influence assessment for an entity.

    Attributes:
        entity_id: Entity identifier.
        entity_name: Entity name.
        equity_share_pct: Ownership share.
        has_operational_control: Operational control flag.
        has_financial_control: Financial control flag.
        determined_control_type: Determined control type.
        consolidation_share_equity: Share under equity approach.
        consolidation_share_operational: Share under operational approach.
        consolidation_share_financial: Share under financial approach.
        recommendation: Recommended approach for this entity.
    """
    entity_id: str = Field(default="", description="Entity ID")
    entity_name: str = Field(default="", description="Name")
    equity_share_pct: Decimal = Field(default=Decimal("0"), description="Equity %")
    has_operational_control: bool = Field(default=False, description="Op control")
    has_financial_control: bool = Field(default=False, description="Fin control")
    determined_control_type: ControlType = Field(
        default=ControlType.NONE, description="Control type"
    )
    consolidation_share_equity: Decimal = Field(
        default=Decimal("0"), description="Equity share"
    )
    consolidation_share_operational: Decimal = Field(
        default=Decimal("0"), description="Operational share"
    )
    consolidation_share_financial: Decimal = Field(
        default=Decimal("0"), description="Financial share"
    )
    recommendation: str = Field(default="", description="Recommendation")


# ---------------------------------------------------------------------------
# Model Rebuild
# ---------------------------------------------------------------------------

EntityEmissions.model_rebuild()
Entity.model_rebuild()
EntityHierarchy.model_rebuild()
IntercompanyRelationship.model_rebuild()
BoundaryChangeEvent.model_rebuild()
EntityConsolidated.model_rebuild()
EliminationEntry.model_rebuild()
BoundaryDefinition.model_rebuild()
ConsolidationResult.model_rebuild()
InfluenceAssessment.model_rebuild()


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------


class MultiEntityBoundaryEngine:
    """Manage Scope 3 boundaries for corporate groups.

    Implements GHG Protocol consolidation approaches, inter-company
    elimination, and time-weighted boundary changes.  Supports equity
    share, operational control, and financial control approaches.

    Follows the zero-hallucination principle: all consolidation uses
    deterministic arithmetic based on ownership shares and control flags.

    Attributes:
        _warnings: Warnings generated during processing.

    Example:
        >>> engine = MultiEntityBoundaryEngine()
        >>> hierarchy = EntityHierarchy(
        ...     group_name="Acme Group",
        ...     entities=[
        ...         Entity(entity_name="Parent", entity_type=EntityType.PARENT,
        ...                equity_share_pct=Decimal("100")),
        ...         Entity(entity_name="Sub A", entity_type=EntityType.SUBSIDIARY,
        ...                equity_share_pct=Decimal("80")),
        ...     ],
        ... )
        >>> result = engine.consolidate_entities(
        ...     hierarchy.entities, ConsolidationApproach.EQUITY_SHARE
        ... )
        >>> print(result.total_scope3_net_tco2e)
    """

    def __init__(self) -> None:
        """Initialise MultiEntityBoundaryEngine."""
        self._warnings: List[str] = []
        logger.info(
            "MultiEntityBoundaryEngine v%s initialised", _MODULE_VERSION
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def define_boundary(
        self,
        entity_hierarchy: EntityHierarchy,
        approach: ConsolidationApproach = ConsolidationApproach.OPERATIONAL_CONTROL,
    ) -> BoundaryDefinition:
        """Define the organisational boundary for a group.

        Determines which entities are included and their consolidation
        shares under the chosen approach.

        Args:
            entity_hierarchy: Complete entity hierarchy.
            approach: Consolidation approach.

        Returns:
            BoundaryDefinition.
        """
        self._warnings = []
        entity_details: List[Dict[str, Any]] = []
        in_boundary = 0
        excluded = 0
        total_equity = Decimal("0")

        for entity in entity_hierarchy.entities:
            share = self._get_consolidation_share(entity, approach)
            included = share > Decimal("0") and entity.is_active

            if included:
                in_boundary += 1
                total_equity += entity.equity_share_pct / Decimal("100")
            else:
                excluded += 1

            entity_details.append({
                "entity_id": entity.entity_id,
                "entity_name": entity.entity_name,
                "entity_type": entity.entity_type.value,
                "equity_share_pct": str(entity.equity_share_pct),
                "consolidation_share": str(_round_val(share, 4)),
                "included_in_boundary": included,
                "control_type": entity.control_type.value,
            })

        return BoundaryDefinition(
            group_id=entity_hierarchy.group_id,
            group_name=entity_hierarchy.group_name,
            approach=approach,
            reporting_year=entity_hierarchy.reporting_year,
            entities_in_boundary=in_boundary,
            entities_excluded=excluded,
            entity_details=entity_details,
            total_equity_weighted_coverage=_round_val(
                total_equity * Decimal("100"), 2
            ),
            notes=list(self._warnings),
        )

    def consolidate_entities(
        self,
        entities: List[Entity],
        approach: ConsolidationApproach = ConsolidationApproach.OPERATIONAL_CONTROL,
        reporting_year: int = 2025,
        relationships: Optional[List[IntercompanyRelationship]] = None,
        boundary_changes: Optional[List[BoundaryChangeEvent]] = None,
        group_id: str = "",
        group_name: str = "",
    ) -> ConsolidationResult:
        """Consolidate all entities under the chosen approach.

        Main entry point for group consolidation.

        Args:
            entities: List of entities to consolidate.
            approach: Consolidation approach.
            reporting_year: Reporting year.
            relationships: Inter-company relationships.
            boundary_changes: Boundary change events.
            group_id: Group identifier.
            group_name: Group name.

        Returns:
            ConsolidationResult with full breakdown.

        Raises:
            ValueError: If no entities provided.
        """
        t0 = time.perf_counter()
        self._warnings = []

        if not entities:
            raise ValueError("At least one entity is required")

        logger.info(
            "Consolidating %d entities using %s approach",
            len(entities), approach.value,
        )

        # Step 1: Apply boundary changes (time weighting)
        change_details: List[Dict[str, Any]] = []
        if boundary_changes:
            change_details = self._apply_boundary_changes(
                entities, boundary_changes, reporting_year
            )

        # Step 2: Consolidate each entity
        entity_results = self._consolidate_all(
            entities, approach, reporting_year
        )

        # Step 3: Calculate gross totals
        gross_s3 = sum(
            (er.consolidated_scope3_tco2e for er in entity_results),
            Decimal("0"),
        )
        total_s1 = sum(
            (er.consolidated_scope1_tco2e for er in entity_results),
            Decimal("0"),
        )
        total_s2 = sum(
            (er.consolidated_scope2_tco2e for er in entity_results),
            Decimal("0"),
        )

        # Step 4: Aggregate by category
        cat_totals: Dict[int, Decimal] = {}
        for er in entity_results:
            for cat_num, val in er.consolidated_scope3_by_category.items():
                cat_totals[cat_num] = cat_totals.get(
                    cat_num, Decimal("0")
                ) + val

        # Step 5: Eliminate inter-company
        eliminations: List[EliminationEntry] = []
        if relationships:
            eliminations = self._eliminate_intercompany(
                entity_results, relationships, entities
            )

        total_eliminated = sum(
            (e.eliminated_tco2e for e in eliminations), Decimal("0")
        )
        net_s3 = gross_s3 - total_eliminated

        # Apply eliminations to category totals
        for elim in eliminations:
            cat = elim.category_eliminated
            if cat in cat_totals:
                cat_totals[cat] = max(
                    cat_totals[cat] - elim.eliminated_tco2e, Decimal("0")
                )

        # Round category totals
        cat_totals_rounded = {
            k: _round_val(v, 2) for k, v in cat_totals.items()
        }

        # Step 6: Create boundary definition
        hierarchy = EntityHierarchy(
            group_id=group_id, group_name=group_name,
            reporting_year=reporting_year, entities=entities,
        )
        boundary = self.define_boundary(hierarchy, approach)

        elapsed_ms = _decimal((time.perf_counter() - t0) * 1000)

        result = ConsolidationResult(
            group_id=group_id,
            group_name=group_name,
            approach=approach,
            reporting_year=reporting_year,
            entity_results=entity_results,
            total_scope1_tco2e=_round_val(total_s1, 2),
            total_scope2_tco2e=_round_val(total_s2, 2),
            total_scope3_gross_tco2e=_round_val(gross_s3, 2),
            intercompany_eliminations=eliminations,
            total_eliminated_tco2e=_round_val(total_eliminated, 2),
            total_scope3_net_tco2e=_round_val(net_s3, 2),
            scope3_by_category=cat_totals_rounded,
            boundary_definition=boundary,
            entities_consolidated=len(entity_results),
            boundary_changes=change_details,
            warnings=list(self._warnings),
            status=ConsolidationStatus.COMPLETE,
            processing_time_ms=_round_val(elapsed_ms, 2),
        )
        result.provenance_hash = self._compute_provenance(result)

        logger.info(
            "Consolidation complete: %d entities, gross=%.2f, "
            "eliminated=%.2f, net=%.2f tCO2e",
            len(entity_results), gross_s3, total_eliminated, net_s3,
        )
        return result

    def eliminate_intercompany(
        self,
        consolidated: ConsolidationResult,
        relationships: List[IntercompanyRelationship],
    ) -> List[EliminationEntry]:
        """Eliminate inter-company Scope 3 double-counting.

        Args:
            consolidated: Consolidation result.
            relationships: Inter-company relationships.

        Returns:
            List of EliminationEntry items.
        """
        return self._eliminate_intercompany(
            consolidated.entity_results,
            relationships,
            [],
        )

    def handle_boundary_change(
        self,
        change_type: BoundaryChangeType,
        entities: List[Entity],
        effective_date: date,
        reporting_year: int = 2025,
    ) -> List[Dict[str, Any]]:
        """Handle a boundary change event (acquisition, divestiture, etc.).

        Calculates time-weighted consolidation shares.

        Args:
            change_type: Type of boundary change.
            entities: Affected entities.
            effective_date: Effective date.
            reporting_year: Reporting year.

        Returns:
            List of change detail dicts.
        """
        changes = [
            BoundaryChangeEvent(
                change_type=change_type,
                entity_id=e.entity_id,
                effective_date=effective_date,
                equity_share_before=(
                    Decimal("0") if change_type == BoundaryChangeType.ACQUISITION
                    else e.equity_share_pct
                ),
                equity_share_after=(
                    e.equity_share_pct if change_type == BoundaryChangeType.ACQUISITION
                    else Decimal("0")
                ),
            )
            for e in entities
        ]
        return self._apply_boundary_changes(entities, changes, reporting_year)

    def assess_influence(
        self,
        entity: Entity,
    ) -> InfluenceAssessment:
        """Assess the control/influence level for an entity.

        Performs the GHG Protocol control test: operational control,
        financial control, and equity share assessment.

        Args:
            entity: Entity to assess.

        Returns:
            InfluenceAssessment with shares under each approach.
        """
        # Determine control type
        control = ControlType.NONE
        if entity.has_operational_control:
            control = ControlType.OPERATIONAL
        elif entity.has_financial_control:
            control = ControlType.FINANCIAL
        elif entity.equity_share_pct >= Decimal("50"):
            control = ControlType.FINANCIAL
        elif entity.equity_share_pct >= Decimal("20"):
            control = ControlType.SIGNIFICANT
        elif entity.entity_type == EntityType.JV:
            control = ControlType.SHARED

        # Shares under each approach
        equity_share = entity.equity_share_pct / Decimal("100")
        op_share = (
            Decimal("1") if entity.has_operational_control else Decimal("0")
        )
        fin_share = (
            Decimal("1") if entity.has_financial_control else Decimal("0")
        )

        # Recommendation
        if entity.entity_type == EntityType.JV:
            rec = (
                "Joint ventures: equity share approach recommended unless "
                "operational control is clearly established"
            )
        elif entity.entity_type == EntityType.ASSOCIATE:
            rec = (
                "Associates: equity share approach only (no control); "
                "include proportional share of Scope 3"
            )
        elif entity.entity_type == EntityType.FRANCHISE:
            rec = (
                "Franchises: consider operational control approach; "
                "franchisor Cat 14 captures franchise emissions"
            )
        elif entity.entity_type == EntityType.INVESTMENT:
            rec = (
                "Investments: equity share approach; Cat 15 captures "
                "financed emissions proportional to ownership"
            )
        else:
            rec = (
                "Subsidiary: use consistent approach with financial "
                "reporting; operational control is most common"
            )

        return InfluenceAssessment(
            entity_id=entity.entity_id,
            entity_name=entity.entity_name,
            equity_share_pct=entity.equity_share_pct,
            has_operational_control=entity.has_operational_control,
            has_financial_control=entity.has_financial_control,
            determined_control_type=control,
            consolidation_share_equity=_round_val(equity_share, 4),
            consolidation_share_operational=_round_val(op_share, 4),
            consolidation_share_financial=_round_val(fin_share, 4),
            recommendation=rec,
        )

    def aggregate_to_group(
        self,
        entity_results: List[EntityConsolidated],
        boundary: BoundaryDefinition,
    ) -> Dict[str, Any]:
        """Aggregate entity results to group level.

        Args:
            entity_results: Per-entity consolidated results.
            boundary: Boundary definition.

        Returns:
            Group-level summary dict.
        """
        total_s1 = sum(
            (er.consolidated_scope1_tco2e for er in entity_results),
            Decimal("0"),
        )
        total_s2 = sum(
            (er.consolidated_scope2_tco2e for er in entity_results),
            Decimal("0"),
        )
        total_s3 = sum(
            (er.consolidated_scope3_tco2e for er in entity_results),
            Decimal("0"),
        )

        cat_totals: Dict[int, Decimal] = {}
        for er in entity_results:
            for cat_num, val in er.consolidated_scope3_by_category.items():
                cat_totals[cat_num] = cat_totals.get(
                    cat_num, Decimal("0")
                ) + val

        return {
            "group_name": boundary.group_name,
            "approach": boundary.approach.value,
            "reporting_year": boundary.reporting_year,
            "entities_in_boundary": boundary.entities_in_boundary,
            "total_scope1_tco2e": str(_round_val(total_s1, 2)),
            "total_scope2_tco2e": str(_round_val(total_s2, 2)),
            "total_scope3_tco2e": str(_round_val(total_s3, 2)),
            "total_all_scopes_tco2e": str(
                _round_val(total_s1 + total_s2 + total_s3, 2)
            ),
            "scope3_by_category": {
                k: str(_round_val(v, 2)) for k, v in cat_totals.items()
            },
        }

    # ------------------------------------------------------------------
    # Internal Methods
    # ------------------------------------------------------------------

    def _get_consolidation_share(
        self,
        entity: Entity,
        approach: ConsolidationApproach,
    ) -> Decimal:
        """Determine consolidation share for an entity.

        Args:
            entity: Entity to evaluate.
            approach: Consolidation approach.

        Returns:
            Consolidation share as Decimal (0-1).
        """
        if approach == ConsolidationApproach.EQUITY_SHARE:
            return entity.equity_share_pct / Decimal("100")
        elif approach == ConsolidationApproach.OPERATIONAL_CONTROL:
            return Decimal("1") if entity.has_operational_control else Decimal("0")
        elif approach == ConsolidationApproach.FINANCIAL_CONTROL:
            return Decimal("1") if entity.has_financial_control else Decimal("0")
        return Decimal("0")

    def _calculate_time_weight(
        self,
        entity: Entity,
        reporting_year: int,
    ) -> Decimal:
        """Calculate time weighting for an entity based on boundary dates.

        Args:
            entity: Entity with boundary dates.
            reporting_year: Reporting year.

        Returns:
            Time weight (0-1).
        """
        year_start = date(reporting_year, 1, 1)
        year_end = date(reporting_year, 12, 31)
        days_in_yr = _days_in_year(reporting_year)

        start = entity.boundary_start_date or year_start
        end = entity.boundary_end_date or year_end

        # Clamp to reporting year
        effective_start = max(start, year_start)
        effective_end = min(end, year_end)

        if effective_end < effective_start:
            return Decimal("0")

        days_in_boundary = (effective_end - effective_start).days + 1
        return _safe_divide(
            _decimal(days_in_boundary), _decimal(days_in_yr)
        )

    def _consolidate_all(
        self,
        entities: List[Entity],
        approach: ConsolidationApproach,
        reporting_year: int,
    ) -> List[EntityConsolidated]:
        """Consolidate all entities.

        Args:
            entities: List of entities.
            approach: Consolidation approach.
            reporting_year: Reporting year.

        Returns:
            List of EntityConsolidated results.
        """
        results: List[EntityConsolidated] = []

        for entity in entities:
            if not entity.is_active:
                results.append(EntityConsolidated(
                    entity_id=entity.entity_id,
                    entity_name=entity.entity_name,
                    entity_type=entity.entity_type,
                    included_in_boundary=False,
                    notes=["Entity inactive / excluded from boundary"],
                ))
                continue

            share = self._get_consolidation_share(entity, approach)
            time_weight = self._calculate_time_weight(entity, reporting_year)

            if share <= Decimal("0"):
                results.append(EntityConsolidated(
                    entity_id=entity.entity_id,
                    entity_name=entity.entity_name,
                    entity_type=entity.entity_type,
                    consolidation_share=Decimal("0"),
                    included_in_boundary=False,
                    notes=[f"No control under {approach.value} approach"],
                ))
                continue

            # Calculate gross total Scope 3
            gross_s3 = entity.emissions.total_scope3_tco2e
            if gross_s3 <= Decimal("0"):
                gross_s3 = sum(
                    entity.emissions.scope3_by_category.values(), Decimal("0")
                )

            # Apply share and time weight
            effective_factor = share * time_weight
            consolidated_s3 = gross_s3 * effective_factor
            consolidated_s1 = entity.emissions.scope1_tco2e * effective_factor
            consolidated_s2 = (
                entity.emissions.scope2_market_tco2e * effective_factor
            )

            # Per-category consolidation
            cat_consolidated: Dict[int, Decimal] = {}
            for cat_num, val in entity.emissions.scope3_by_category.items():
                cat_consolidated[cat_num] = _round_val(
                    val * effective_factor, 2
                )

            notes: List[str] = []
            if time_weight < Decimal("1"):
                notes.append(
                    f"Time-weighted: {_round_val(time_weight * Decimal('100'), 1)}% "
                    f"of year"
                )

            results.append(EntityConsolidated(
                entity_id=entity.entity_id,
                entity_name=entity.entity_name,
                entity_type=entity.entity_type,
                consolidation_share=_round_val(share, 4),
                time_weight=_round_val(time_weight, 4),
                gross_scope3_tco2e=_round_val(gross_s3, 2),
                consolidated_scope3_tco2e=_round_val(consolidated_s3, 2),
                consolidated_scope3_by_category=cat_consolidated,
                consolidated_scope1_tco2e=_round_val(consolidated_s1, 2),
                consolidated_scope2_tco2e=_round_val(consolidated_s2, 2),
                included_in_boundary=True,
                notes=notes,
            ))

        return results

    def _eliminate_intercompany(
        self,
        entity_results: List[EntityConsolidated],
        relationships: List[IntercompanyRelationship],
        entities: List[Entity],
    ) -> List[EliminationEntry]:
        """Eliminate inter-company Scope 3 overlaps.

        Args:
            entity_results: Consolidated entity results.
            relationships: Inter-company relationships.
            entities: Original entity list (for lookup).

        Returns:
            List of EliminationEntry items.
        """
        eliminations: List[EliminationEntry] = []

        # Build entity lookup
        entity_map: Dict[str, EntityConsolidated] = {
            er.entity_id: er for er in entity_results
        }

        for rel in relationships:
            seller = entity_map.get(rel.seller_entity_id)
            buyer = entity_map.get(rel.buyer_entity_id)

            if not seller or not buyer:
                continue
            if not seller.included_in_boundary or not buyer.included_in_boundary:
                continue

            # Check each elimination rule
            for rule in INTERCOMPANY_OVERLAP_RULES:
                seller_cat = rule["seller_category"]
                buyer_cat = rule["buyer_category"]

                # Check if this relationship triggers the rule
                seller_has = (
                    seller_cat in rel.affected_categories_seller
                    or (seller_cat == 0 and True)
                )
                buyer_has = buyer_cat in rel.affected_categories_buyer

                if not (seller_has and buyer_has):
                    continue

                # Calculate elimination amount
                elimination_amount = self._calculate_elimination(
                    seller, buyer, rel, rule
                )

                if elimination_amount > Decimal("0"):
                    method = rule["elimination_method"]
                    elim_cat = (
                        buyer_cat
                        if method == "remove_buyer_side"
                        else seller_cat
                    )

                    eliminations.append(EliminationEntry(
                        rule_id=rule["rule_id"],
                        seller_entity_id=rel.seller_entity_id,
                        buyer_entity_id=rel.buyer_entity_id,
                        category_eliminated=elim_cat,
                        eliminated_tco2e=_round_val(elimination_amount, 2),
                        method=method,
                        description=(
                            f"{rule['name']}: eliminated {elimination_amount:.2f} "
                            f"tCO2e from Cat {elim_cat}"
                        ),
                    ))

        return eliminations

    def _calculate_elimination(
        self,
        seller: EntityConsolidated,
        buyer: EntityConsolidated,
        relationship: IntercompanyRelationship,
        rule: Dict[str, Any],
    ) -> Decimal:
        """Calculate the elimination amount for a specific rule.

        Args:
            seller: Selling entity consolidated result.
            buyer: Buying entity consolidated result.
            relationship: The inter-company relationship.
            rule: Elimination rule definition.

        Returns:
            Amount to eliminate in tCO2e.
        """
        if relationship.estimated_emissions_tco2e > Decimal("0"):
            return relationship.estimated_emissions_tco2e

        # Estimate from category data
        method = rule["elimination_method"]
        if method == "remove_buyer_side":
            cat = rule["buyer_category"]
            return buyer.consolidated_scope3_by_category.get(
                cat, Decimal("0")
            ) * Decimal("0.5")  # Conservative 50% overlap estimate
        elif method == "remove_seller_side":
            cat = rule["seller_category"]
            return seller.consolidated_scope3_by_category.get(
                cat, Decimal("0")
            ) * Decimal("0.5")
        else:
            # remove_lower_confidence: take smaller of the two
            seller_val = seller.consolidated_scope3_by_category.get(
                rule["seller_category"], Decimal("0")
            )
            buyer_val = buyer.consolidated_scope3_by_category.get(
                rule["buyer_category"], Decimal("0")
            )
            return min(seller_val, buyer_val) * Decimal("0.5")

    def _apply_boundary_changes(
        self,
        entities: List[Entity],
        changes: List[BoundaryChangeEvent],
        reporting_year: int,
    ) -> List[Dict[str, Any]]:
        """Apply boundary change events to entities.

        Sets boundary_start_date or boundary_end_date and updates
        equity shares based on the events.

        Args:
            entities: List of entities.
            changes: Boundary change events.
            reporting_year: Reporting year.

        Returns:
            List of change detail dicts.
        """
        entity_map = {e.entity_id: e for e in entities}
        details: List[Dict[str, Any]] = []

        for change in changes:
            entity = entity_map.get(change.entity_id)
            if not entity:
                self._warnings.append(
                    f"Boundary change references unknown entity: "
                    f"{change.entity_id}"
                )
                continue

            time_weight_before = self._calculate_time_weight(
                entity, reporting_year
            )

            if change.change_type == BoundaryChangeType.ACQUISITION:
                entity.boundary_start_date = change.effective_date
                entity.equity_share_pct = change.equity_share_after
                entity.is_active = True
            elif change.change_type == BoundaryChangeType.DIVESTITURE:
                entity.boundary_end_date = change.effective_date
                entity.equity_share_pct = change.equity_share_before
            elif change.change_type == BoundaryChangeType.JV_FORMED:
                entity.boundary_start_date = change.effective_date
                entity.equity_share_pct = change.equity_share_after
                entity.entity_type = EntityType.JV
            elif change.change_type == BoundaryChangeType.JV_DISSOLVED:
                entity.boundary_end_date = change.effective_date

            time_weight_after = self._calculate_time_weight(
                entity, reporting_year
            )

            details.append({
                "event_id": change.event_id,
                "change_type": change.change_type.value,
                "entity_id": change.entity_id,
                "entity_name": entity.entity_name,
                "effective_date": str(change.effective_date),
                "equity_share_before": str(change.equity_share_before),
                "equity_share_after": str(change.equity_share_after),
                "time_weight_before": str(_round_val(time_weight_before, 4)),
                "time_weight_after": str(_round_val(time_weight_after, 4)),
            })

        return details

    def _compute_provenance(self, result: ConsolidationResult) -> str:
        """Compute SHA-256 provenance hash.

        Args:
            result: Complete consolidation result.

        Returns:
            SHA-256 hex digest.
        """
        return _compute_hash(result)
