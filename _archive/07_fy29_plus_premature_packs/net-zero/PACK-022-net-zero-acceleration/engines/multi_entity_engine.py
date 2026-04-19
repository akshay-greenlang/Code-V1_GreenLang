# -*- coding: utf-8 -*-
"""
MultiEntityEngine - PACK-022 Net Zero Acceleration Engine 8
=============================================================

Multi-entity GHG consolidation engine supporting up to 50 subsidiaries
with three consolidation approaches (equity share, financial control,
operational control), intercompany emission elimination, base year
recalculation, and hierarchical target allocation.

GHG Protocol Corporate Standard Consolidation (Chapter 3):
    Organizations shall choose and consistently apply one of three
    consolidation approaches:

    1. Equity Share:  An organization accounts for GHG emissions from
       operations according to its share of equity in the operation.
       The equity share reflects economic interest, which is the extent
       of rights an organization has to the risks and rewards flowing
       from an operation.

    2. Financial Control:  An organization accounts for 100% of the GHG
       emissions from operations over which it has financial control,
       defined as the ability to direct the financial and operating
       policies of the operation with a view to gaining economic benefits.

    3. Operational Control:  An organization accounts for 100% of the GHG
       emissions from operations over which it has operational control,
       defined as the full authority to introduce and implement its
       operating policies at the operation.

Base Year Recalculation (GHG Protocol Ch. 5):
    The base year emissions shall be recalculated when structural changes
    (mergers, acquisitions, divestitures) result in a change of more than
    the significance threshold (typically 5%) of total base year emissions.

Intercompany Elimination:
    Emissions between group entities that would otherwise be double-counted
    (e.g., Scope 1 at one entity = Scope 3 Cat 1 at another entity within
    the same consolidation boundary) must be eliminated to avoid
    double-counting in the consolidated group inventory.

Features:
    - Three consolidation methods: equity share, financial control, operational control
    - Entity hierarchy up to 3 levels (parent -> subsidiary -> sub-subsidiary)
    - Equity share weighting with ownership percentage
    - Intercompany emission elimination
    - Base year recalculation with 5% significance threshold
    - Completeness tracking (actual vs estimated)
    - Currency normalization for intensity metrics
    - Target allocation (top-down and bottom-up)
    - Progress aggregation across all entities

Regulatory References:
    - GHG Protocol Corporate Standard Chapter 3 (Setting boundaries)
    - GHG Protocol Corporate Standard Chapter 5 (Tracking over time)
    - CSRD ESRS E1 (Scope 1, 2, 3 disclosure at group level)
    - SBTi Corporate Net-Zero Standard Section 5 (Boundary)
    - ISO 14064-1:2018 Section 5.2 (Organizational boundary)

Zero-Hallucination:
    - All consolidation uses Decimal arithmetic with ROUND_HALF_UP
    - Equity share weighting is deterministic multiplication
    - Intercompany elimination uses explicit pair-based identification
    - Base year recalculation uses threshold comparison only
    - SHA-256 provenance hash on every result

Author:  GreenLang Platform Team
Date:    March 2026
Pack:    PACK-022 Net Zero Acceleration
Status:  Production Ready
"""

from __future__ import annotations

import hashlib
import json
import logging
import uuid
from collections import defaultdict
from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP, InvalidOperation
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field, field_validator

from greenlang.schemas import utcnow

logger = logging.getLogger(__name__)

_MODULE_VERSION: str = "1.0.0"

MAX_ENTITIES: int = 50
MAX_HIERARCHY_DEPTH: int = 3
DEFAULT_SIGNIFICANCE_THRESHOLD: Decimal = Decimal("5")  # 5% for base year recalculation

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _new_uuid() -> str:
    """Generate a new UUID4 string."""
    return str(uuid.uuid4())

def _compute_hash(data: Any) -> str:
    """Compute a deterministic SHA-256 hash of arbitrary data."""
    if hasattr(data, "model_dump"):
        serializable = data.model_dump(mode="json")
    elif isinstance(data, dict):
        serializable = data
    else:
        serializable = str(data)
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
    numerator: Decimal, denominator: Decimal, default: Decimal = Decimal("0")
) -> Decimal:
    """Safely divide two Decimals, returning default on zero denominator."""
    if denominator == Decimal("0"):
        return default
    return numerator / denominator

def _safe_pct(
    part: Decimal, whole: Decimal, places: int = 2
) -> Decimal:
    """Calculate percentage safely."""
    if whole == Decimal("0"):
        return Decimal("0")
    return (part / whole * Decimal("100")).quantize(
        Decimal("0." + "0" * places), rounding=ROUND_HALF_UP
    )

def _round_val(value: Decimal, places: int = 4) -> Decimal:
    """Round a Decimal value to the specified number of decimal places."""
    quantize_str = "0." + "0" * places
    return value.quantize(Decimal(quantize_str), rounding=ROUND_HALF_UP)

# ---------------------------------------------------------------------------
# Reference Data: Common Reporting Currencies
# ---------------------------------------------------------------------------

CURRENCY_RATES_TO_USD: Dict[str, Decimal] = {
    "USD": Decimal("1.000"),
    "EUR": Decimal("1.085"),
    "GBP": Decimal("1.265"),
    "JPY": Decimal("0.0067"),
    "CHF": Decimal("1.115"),
    "CAD": Decimal("0.745"),
    "AUD": Decimal("0.660"),
    "CNY": Decimal("0.138"),
    "INR": Decimal("0.012"),
    "BRL": Decimal("0.200"),
    "KRW": Decimal("0.00075"),
    "SEK": Decimal("0.095"),
    "NOK": Decimal("0.092"),
    "DKK": Decimal("0.146"),
    "SGD": Decimal("0.745"),
    "HKD": Decimal("0.128"),
    "MXN": Decimal("0.058"),
    "ZAR": Decimal("0.055"),
}

# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class ConsolidationMethod(str, Enum):
    """GHG consolidation approach per GHG Protocol Chapter 3."""
    EQUITY_SHARE = "equity_share"
    FINANCIAL_CONTROL = "financial_control"
    OPERATIONAL_CONTROL = "operational_control"

class EntityType(str, Enum):
    """Type of entity in the corporate hierarchy."""
    PARENT = "parent"
    SUBSIDIARY = "subsidiary"
    SUB_SUBSIDIARY = "sub_subsidiary"
    JOINT_VENTURE = "joint_venture"
    ASSOCIATE = "associate"

class ReportingStatus(str, Enum):
    """Entity reporting completeness status."""
    ACTUAL = "actual"
    ESTIMATED = "estimated"
    PARTIAL = "partial"
    NOT_REPORTED = "not_reported"

class StructuralChangeType(str, Enum):
    """Type of structural change triggering base year recalculation."""
    ACQUISITION = "acquisition"
    DIVESTITURE = "divestiture"
    MERGER = "merger"
    OUTSOURCING = "outsourcing"
    INSOURCING = "insourcing"
    ORGANIC_GROWTH = "organic_growth"

class TargetAllocationType(str, Enum):
    """Target allocation method for distributing group target."""
    TOP_DOWN_EQUAL = "top_down_equal"
    TOP_DOWN_PROPORTIONAL = "top_down_proportional"
    BOTTOM_UP = "bottom_up"

class EliminationType(str, Enum):
    """Type of intercompany elimination."""
    SCOPE1_TO_SCOPE3 = "scope1_to_scope3"
    SCOPE2_TO_SCOPE3 = "scope2_to_scope3"
    INTERNAL_TRANSFER = "internal_transfer"

# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------

class EntityEmissions(BaseModel):
    """GHG emissions data for a single entity in one reporting year."""
    entity_id: str = Field(default_factory=_new_uuid, description="Entity identifier")
    entity_name: str = Field(description="Entity name")
    entity_type: EntityType = Field(default=EntityType.SUBSIDIARY, description="Entity type")
    parent_entity_id: Optional[str] = Field(default=None, description="Parent entity ID")
    hierarchy_level: int = Field(default=1, description="Hierarchy level (0=parent, 1=sub, 2=sub-sub)")
    country: str = Field(default="", description="Country code (ISO 3166-1 alpha-2)")
    currency: str = Field(default="USD", description="Reporting currency")
    ownership_pct: Decimal = Field(default=Decimal("100"), description="Equity ownership %")
    has_financial_control: bool = Field(default=False, description="Financial control flag")
    has_operational_control: bool = Field(default=False, description="Operational control flag")
    reporting_year: int = Field(description="Reporting year")
    scope1_emissions: Decimal = Field(default=Decimal("0"), description="Scope 1 tCO2e")
    scope2_location: Decimal = Field(default=Decimal("0"), description="Scope 2 location-based tCO2e")
    scope2_market: Decimal = Field(default=Decimal("0"), description="Scope 2 market-based tCO2e")
    scope3_emissions: Decimal = Field(default=Decimal("0"), description="Scope 3 tCO2e")
    total_emissions: Decimal = Field(default=Decimal("0"), description="Total tCO2e")
    revenue: Decimal = Field(default=Decimal("0"), description="Revenue in local currency")
    reporting_status: ReportingStatus = Field(
        default=ReportingStatus.ACTUAL, description="Reporting completeness"
    )
    created_at: datetime = Field(default_factory=utcnow, description="Creation timestamp")
    provenance_hash: str = Field(default="", description="SHA-256 provenance hash")

    @field_validator("ownership_pct", "scope1_emissions", "scope2_location",
                     "scope2_market", "scope3_emissions", "total_emissions",
                     "revenue", mode="before")
    @classmethod
    def _coerce_decimal(cls, v: Any) -> Decimal:
        return _decimal(v)

class IntercompanyElimination(BaseModel):
    """Intercompany emission elimination to prevent double counting."""
    elimination_id: str = Field(default_factory=_new_uuid, description="Elimination ID")
    seller_entity_id: str = Field(description="Seller/provider entity ID")
    buyer_entity_id: str = Field(description="Buyer/receiver entity ID")
    elimination_type: EliminationType = Field(description="Type of elimination")
    scope_at_seller: str = Field(description="Scope classification at seller (scope1/scope2)")
    scope_at_buyer: str = Field(description="Scope classification at buyer (scope3)")
    emissions_eliminated: Decimal = Field(description="Emissions eliminated (tCO2e)")
    description: str = Field(default="", description="Description of elimination")
    provenance_hash: str = Field(default="", description="SHA-256 provenance hash")

    @field_validator("emissions_eliminated", mode="before")
    @classmethod
    def _coerce_decimal(cls, v: Any) -> Decimal:
        return _decimal(v)

class StructuralChange(BaseModel):
    """Record of a structural change affecting the corporate boundary."""
    change_id: str = Field(default_factory=_new_uuid, description="Change identifier")
    change_type: StructuralChangeType = Field(description="Type of change")
    entity_id: str = Field(description="Affected entity ID")
    entity_name: str = Field(default="", description="Affected entity name")
    effective_year: int = Field(description="Year the change took effect")
    emissions_impact: Decimal = Field(description="Impact on base year emissions (tCO2e)")
    impact_pct: Decimal = Field(default=Decimal("0"), description="Impact as % of base year total")
    triggers_recalculation: bool = Field(default=False, description="Whether threshold exceeded")
    description: str = Field(default="", description="Change description")
    provenance_hash: str = Field(default="", description="SHA-256 provenance hash")

    @field_validator("emissions_impact", "impact_pct", mode="before")
    @classmethod
    def _coerce_decimal(cls, v: Any) -> Decimal:
        return _decimal(v)

class EntityTargetAllocation(BaseModel):
    """Target allocation for a single entity."""
    entity_id: str = Field(description="Entity identifier")
    entity_name: str = Field(default="", description="Entity name")
    allocated_target_pct: Decimal = Field(description="Allocated reduction target (%)")
    allocated_target_absolute: Decimal = Field(description="Allocated absolute target (tCO2e)")
    current_emissions: Decimal = Field(description="Current emissions (tCO2e)")
    required_reduction: Decimal = Field(description="Required reduction (tCO2e)")
    progress_pct: Decimal = Field(default=Decimal("0"), description="Current progress %")
    on_track: bool = Field(default=False, description="Whether on track")
    provenance_hash: str = Field(default="", description="SHA-256 provenance hash")

    @field_validator("allocated_target_pct", "allocated_target_absolute",
                     "current_emissions", "required_reduction",
                     "progress_pct", mode="before")
    @classmethod
    def _coerce_decimal(cls, v: Any) -> Decimal:
        return _decimal(v)

class GroupEmissions(BaseModel):
    """Consolidated group-level emissions."""
    group_id: str = Field(default_factory=_new_uuid, description="Group identifier")
    group_name: str = Field(default="", description="Group name")
    reporting_year: int = Field(description="Reporting year")
    consolidation_method: ConsolidationMethod = Field(description="Consolidation approach")
    scope1_total: Decimal = Field(description="Consolidated Scope 1 tCO2e")
    scope2_location_total: Decimal = Field(description="Consolidated Scope 2 location tCO2e")
    scope2_market_total: Decimal = Field(description="Consolidated Scope 2 market tCO2e")
    scope3_total: Decimal = Field(description="Consolidated Scope 3 tCO2e")
    total_emissions: Decimal = Field(description="Consolidated total tCO2e")
    eliminations_total: Decimal = Field(default=Decimal("0"), description="Eliminations tCO2e")
    entities_consolidated: int = Field(default=0, description="Number of entities")
    entities_actual: int = Field(default=0, description="Entities with actual data")
    entities_estimated: int = Field(default=0, description="Entities with estimated data")
    completeness_pct: Decimal = Field(default=Decimal("0"), description="Data completeness %")
    revenue_total: Decimal = Field(default=Decimal("0"), description="Group revenue (USD)")
    intensity_per_revenue: Decimal = Field(default=Decimal("0"), description="tCO2e per $M revenue")
    calculated_at: datetime = Field(default_factory=utcnow, description="Calculation timestamp")
    provenance_hash: str = Field(default="", description="SHA-256 provenance hash")

    @field_validator("scope1_total", "scope2_location_total", "scope2_market_total",
                     "scope3_total", "total_emissions", "eliminations_total",
                     "completeness_pct", "revenue_total",
                     "intensity_per_revenue", mode="before")
    @classmethod
    def _coerce_decimal(cls, v: Any) -> Decimal:
        return _decimal(v)

class BaseYearRecalculation(BaseModel):
    """Result of base year recalculation assessment."""
    recalculation_id: str = Field(default_factory=_new_uuid, description="Recalculation ID")
    base_year: int = Field(description="Original base year")
    original_base_year_emissions: Decimal = Field(description="Original base year total (tCO2e)")
    structural_changes: List[StructuralChange] = Field(
        default_factory=list, description="List of structural changes"
    )
    total_impact: Decimal = Field(description="Total impact of all changes (tCO2e)")
    total_impact_pct: Decimal = Field(description="Total impact as % of base year")
    significance_threshold_pct: Decimal = Field(description="Threshold for recalculation (%)")
    recalculation_required: bool = Field(description="Whether recalculation is triggered")
    recalculated_base_year_emissions: Optional[Decimal] = Field(
        default=None, description="Recalculated base year emissions (tCO2e)"
    )
    calculated_at: datetime = Field(default_factory=utcnow, description="Calculation timestamp")
    provenance_hash: str = Field(default="", description="SHA-256 provenance hash")

    @field_validator("original_base_year_emissions", "total_impact",
                     "total_impact_pct", "significance_threshold_pct",
                     mode="before")
    @classmethod
    def _coerce_decimal(cls, v: Any) -> Decimal:
        return _decimal(v)

    @field_validator("recalculated_base_year_emissions", mode="before")
    @classmethod
    def _coerce_decimal_opt(cls, v: Any) -> Optional[Decimal]:
        if v is None:
            return None
        return _decimal(v)

class ConsolidationResult(BaseModel):
    """Complete multi-entity consolidation result."""
    result_id: str = Field(default_factory=_new_uuid, description="Result identifier")
    group_emissions: GroupEmissions = Field(description="Consolidated group emissions")
    entity_breakdown: List[Dict[str, Any]] = Field(
        default_factory=list, description="Per-entity breakdown"
    )
    eliminations: List[IntercompanyElimination] = Field(
        default_factory=list, description="Intercompany eliminations applied"
    )
    completeness_pct: Decimal = Field(default=Decimal("0"), description="Overall completeness %")
    target_allocation: List[EntityTargetAllocation] = Field(
        default_factory=list, description="Target allocation per entity"
    )
    structural_changes: List[StructuralChange] = Field(
        default_factory=list, description="Structural changes assessed"
    )
    base_year_recalculation: Optional[BaseYearRecalculation] = Field(
        default=None, description="Base year recalculation if triggered"
    )
    calculated_at: datetime = Field(default_factory=utcnow, description="Calculation timestamp")
    provenance_hash: str = Field(default="", description="SHA-256 provenance hash")

    @field_validator("completeness_pct", mode="before")
    @classmethod
    def _coerce_decimal(cls, v: Any) -> Decimal:
        return _decimal(v)

# ---------------------------------------------------------------------------
# Engine Configuration
# ---------------------------------------------------------------------------

class MultiEntityConfig(BaseModel):
    """Configuration for the MultiEntityEngine."""
    default_consolidation: ConsolidationMethod = Field(
        default=ConsolidationMethod.OPERATIONAL_CONTROL,
        description="Default consolidation approach",
    )
    significance_threshold_pct: Decimal = Field(
        default=DEFAULT_SIGNIFICANCE_THRESHOLD,
        description="Base year recalculation threshold (%)",
    )
    max_entities: int = Field(
        default=MAX_ENTITIES, description="Maximum entities in group"
    )
    reporting_currency: str = Field(
        default="USD", description="Group reporting currency"
    )
    decimal_precision: int = Field(
        default=4, description="Decimal places for results"
    )

# ---------------------------------------------------------------------------
# Pydantic model_rebuild
# ---------------------------------------------------------------------------

EntityEmissions.model_rebuild()
IntercompanyElimination.model_rebuild()
StructuralChange.model_rebuild()
EntityTargetAllocation.model_rebuild()
GroupEmissions.model_rebuild()
BaseYearRecalculation.model_rebuild()
ConsolidationResult.model_rebuild()
MultiEntityConfig.model_rebuild()

# ---------------------------------------------------------------------------
# MultiEntityEngine
# ---------------------------------------------------------------------------

class MultiEntityEngine:
    """
    Multi-entity GHG consolidation engine.

    Consolidates emissions across up to 50 entities using equity share,
    financial control, or operational control approaches.  Supports
    intercompany elimination, base year recalculation, and hierarchical
    target allocation.

    Attributes:
        config: Engine configuration.
        _entities: In-memory entity emission store by (entity_id, year).
        _eliminations: Intercompany eliminations.
        _structural_changes: Registered structural changes.

    Example:
        >>> engine = MultiEntityEngine()
        >>> engine.add_entity(entity_data)
        >>> consolidated = engine.consolidate(2025, ConsolidationMethod.EQUITY_SHARE)
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize MultiEntityEngine.

        Args:
            config: Optional configuration dictionary.
        """
        if config and isinstance(config, dict):
            self.config = MultiEntityConfig(**config)
        elif config and isinstance(config, MultiEntityConfig):
            self.config = config
        else:
            self.config = MultiEntityConfig()

        self._entities: Dict[Tuple[str, int], EntityEmissions] = {}
        self._eliminations: List[IntercompanyElimination] = []
        self._structural_changes: List[StructuralChange] = []
        self._group_name: str = ""
        logger.info("MultiEntityEngine initialized (v%s)", _MODULE_VERSION)

    # -------------------------------------------------------------------
    # Entity Management
    # -------------------------------------------------------------------

    def set_group_name(self, name: str) -> None:
        """Set the corporate group name.

        Args:
            name: Group name.
        """
        self._group_name = name.strip()

    def add_entity(self, entity: EntityEmissions) -> EntityEmissions:
        """Add entity emissions data.

        Args:
            entity: EntityEmissions data.

        Returns:
            Entity with computed provenance hash and auto-calculated total.

        Raises:
            ValueError: If maximum entities exceeded or hierarchy too deep.
        """
        unique_ids = set(eid for (eid, _) in self._entities)
        if entity.entity_id not in unique_ids and len(unique_ids) >= self.config.max_entities:
            raise ValueError(
                f"Maximum {self.config.max_entities} entities allowed"
            )

        if entity.hierarchy_level > MAX_HIERARCHY_DEPTH:
            raise ValueError(
                f"Hierarchy level {entity.hierarchy_level} exceeds max depth {MAX_HIERARCHY_DEPTH}"
            )

        # Auto-calculate total if not set
        if entity.total_emissions == Decimal("0"):
            entity.total_emissions = (
                entity.scope1_emissions
                + entity.scope2_market
                + entity.scope3_emissions
            )

        entity.provenance_hash = _compute_hash(entity)
        key = (entity.entity_id, entity.reporting_year)
        self._entities[key] = entity

        logger.info(
            "Added entity %s (%s) year %d: S1=%.1f, S2=%.1f, S3=%.1f",
            entity.entity_id, entity.entity_name, entity.reporting_year,
            float(entity.scope1_emissions), float(entity.scope2_market),
            float(entity.scope3_emissions),
        )
        return entity

    def add_entities(self, entities: List[EntityEmissions]) -> int:
        """Add multiple entity emissions records.

        Args:
            entities: List of EntityEmissions objects.

        Returns:
            Number of entities added.
        """
        count = 0
        for entity in entities:
            self.add_entity(entity)
            count += 1
        return count

    # -------------------------------------------------------------------
    # Intercompany Eliminations
    # -------------------------------------------------------------------

    def add_elimination(self, elimination: IntercompanyElimination) -> IntercompanyElimination:
        """Register an intercompany emission elimination.

        Args:
            elimination: IntercompanyElimination data.

        Returns:
            Elimination with computed provenance hash.
        """
        elimination.provenance_hash = _compute_hash(elimination)
        self._eliminations.append(elimination)
        logger.info(
            "Added elimination: %s -> %s, %.1f tCO2e (%s)",
            elimination.seller_entity_id, elimination.buyer_entity_id,
            float(elimination.emissions_eliminated), elimination.elimination_type.value,
        )
        return elimination

    # -------------------------------------------------------------------
    # Currency Normalization
    # -------------------------------------------------------------------

    def _normalize_currency(
        self, amount: Decimal, from_currency: str
    ) -> Decimal:
        """Convert amount to reporting currency (USD by default).

        Args:
            amount: Amount in source currency.
            from_currency: Source currency code.

        Returns:
            Amount in reporting currency.
        """
        target = self.config.reporting_currency.upper()
        source = from_currency.upper()

        if source == target:
            return amount

        source_rate = CURRENCY_RATES_TO_USD.get(source, Decimal("1"))
        target_rate = CURRENCY_RATES_TO_USD.get(target, Decimal("1"))

        # Convert to USD then to target
        usd_amount = amount * source_rate
        return _safe_divide(usd_amount, target_rate, amount)

    # -------------------------------------------------------------------
    # Consolidation
    # -------------------------------------------------------------------

    def _get_consolidation_factor(
        self, entity: EntityEmissions, method: ConsolidationMethod
    ) -> Decimal:
        """Get the consolidation factor for an entity.

        Args:
            entity: Entity data.
            method: Consolidation method.

        Returns:
            Factor (0 to 1) to apply to entity emissions.
        """
        if method == ConsolidationMethod.EQUITY_SHARE:
            return entity.ownership_pct / Decimal("100")
        elif method == ConsolidationMethod.FINANCIAL_CONTROL:
            return Decimal("1") if entity.has_financial_control else Decimal("0")
        elif method == ConsolidationMethod.OPERATIONAL_CONTROL:
            return Decimal("1") if entity.has_operational_control else Decimal("0")
        return Decimal("0")

    def consolidate(
        self,
        reporting_year: int,
        method: Optional[ConsolidationMethod] = None,
    ) -> GroupEmissions:
        """Consolidate emissions across all entities for a reporting year.

        Args:
            reporting_year: Year to consolidate.
            method: Consolidation method (defaults to config).

        Returns:
            GroupEmissions with consolidated totals.

        Raises:
            ValueError: If no data for the reporting year.
        """
        if method is None:
            method = self.config.default_consolidation

        year_entities: List[EntityEmissions] = []
        for (eid, yr), entity in self._entities.items():
            if yr == reporting_year:
                year_entities.append(entity)

        if not year_entities:
            raise ValueError(f"No entity data for year {reporting_year}")

        prec = self.config.decimal_precision
        s1_total = Decimal("0")
        s2_loc_total = Decimal("0")
        s2_mkt_total = Decimal("0")
        s3_total = Decimal("0")
        revenue_total_usd = Decimal("0")
        actual_count = 0
        estimated_count = 0

        entity_breakdown: List[Dict[str, Any]] = []

        for entity in year_entities:
            factor = self._get_consolidation_factor(entity, method)

            adj_s1 = _round_val(entity.scope1_emissions * factor, prec)
            adj_s2_loc = _round_val(entity.scope2_location * factor, prec)
            adj_s2_mkt = _round_val(entity.scope2_market * factor, prec)
            adj_s3 = _round_val(entity.scope3_emissions * factor, prec)
            adj_total = adj_s1 + adj_s2_mkt + adj_s3

            s1_total += adj_s1
            s2_loc_total += adj_s2_loc
            s2_mkt_total += adj_s2_mkt
            s3_total += adj_s3

            # Normalize revenue
            rev_usd = self._normalize_currency(entity.revenue, entity.currency)
            revenue_total_usd += rev_usd

            if entity.reporting_status == ReportingStatus.ACTUAL:
                actual_count += 1
            else:
                estimated_count += 1

            entity_breakdown.append({
                "entity_id": entity.entity_id,
                "entity_name": entity.entity_name,
                "entity_type": entity.entity_type.value,
                "country": entity.country,
                "ownership_pct": str(entity.ownership_pct),
                "consolidation_factor": str(factor),
                "scope1_adjusted": str(adj_s1),
                "scope2_location_adjusted": str(adj_s2_loc),
                "scope2_market_adjusted": str(adj_s2_mkt),
                "scope3_adjusted": str(adj_s3),
                "total_adjusted": str(adj_total),
                "revenue_usd": str(_round_val(rev_usd, 2)),
                "reporting_status": entity.reporting_status.value,
            })

        # Apply intercompany eliminations
        elim_total = Decimal("0")
        for elim in self._eliminations:
            elim_total += elim.emissions_eliminated

        total_before_elim = s1_total + s2_mkt_total + s3_total
        total_after_elim = total_before_elim - elim_total

        completeness = _safe_pct(_decimal(actual_count), _decimal(len(year_entities)))

        # Intensity (tCO2e per $1M revenue)
        intensity = Decimal("0")
        if revenue_total_usd > Decimal("0"):
            intensity = _round_val(
                total_after_elim / (revenue_total_usd / Decimal("1000000")), prec
            )

        group_emissions = GroupEmissions(
            group_name=self._group_name,
            reporting_year=reporting_year,
            consolidation_method=method,
            scope1_total=_round_val(s1_total, prec),
            scope2_location_total=_round_val(s2_loc_total, prec),
            scope2_market_total=_round_val(s2_mkt_total, prec),
            scope3_total=_round_val(s3_total, prec),
            total_emissions=_round_val(total_after_elim, prec),
            eliminations_total=_round_val(elim_total, prec),
            entities_consolidated=len(year_entities),
            entities_actual=actual_count,
            entities_estimated=estimated_count,
            completeness_pct=completeness,
            revenue_total=_round_val(revenue_total_usd, 2),
            intensity_per_revenue=intensity,
        )
        group_emissions.provenance_hash = _compute_hash(group_emissions)

        logger.info(
            "Consolidated %d entities for %d (%s): total=%.1f tCO2e (elim=%.1f), completeness=%.0f%%",
            len(year_entities), reporting_year, method.value,
            float(total_after_elim), float(elim_total), float(completeness),
        )
        return group_emissions

    # -------------------------------------------------------------------
    # Base Year Recalculation
    # -------------------------------------------------------------------

    def add_structural_change(self, change: StructuralChange) -> StructuralChange:
        """Register a structural change for base year assessment.

        Args:
            change: StructuralChange data.

        Returns:
            Change with computed provenance hash.
        """
        change.provenance_hash = _compute_hash(change)
        self._structural_changes.append(change)
        logger.info(
            "Structural change registered: %s - %s (%s), impact=%.1f tCO2e",
            change.change_id, change.entity_name,
            change.change_type.value, float(change.emissions_impact),
        )
        return change

    def assess_base_year_recalculation(
        self,
        base_year: int,
        method: Optional[ConsolidationMethod] = None,
    ) -> BaseYearRecalculation:
        """Assess whether base year recalculation is required.

        Per GHG Protocol Chapter 5, recalculation is required when
        structural changes exceed the significance threshold.

        Args:
            base_year: The original base year.
            method: Consolidation method (defaults to config).

        Returns:
            BaseYearRecalculation assessment.

        Raises:
            ValueError: If no base year data available.
        """
        try:
            base_group = self.consolidate(base_year, method)
        except ValueError:
            raise ValueError(f"No entity data for base year {base_year}")

        original_emissions = base_group.total_emissions
        threshold = self.config.significance_threshold_pct
        prec = self.config.decimal_precision

        # Calculate cumulative impact of structural changes
        total_impact = Decimal("0")
        assessed_changes: List[StructuralChange] = []

        for change in self._structural_changes:
            impact_pct = _safe_pct(abs(change.emissions_impact), original_emissions)
            change.impact_pct = impact_pct
            change.triggers_recalculation = impact_pct >= threshold
            total_impact += change.emissions_impact
            assessed_changes.append(change)

        total_impact_pct = _safe_pct(abs(total_impact), original_emissions)
        recalc_required = total_impact_pct >= threshold

        recalculated = None
        if recalc_required:
            recalculated = _round_val(original_emissions + total_impact, prec)

        result = BaseYearRecalculation(
            base_year=base_year,
            original_base_year_emissions=original_emissions,
            structural_changes=assessed_changes,
            total_impact=_round_val(total_impact, prec),
            total_impact_pct=total_impact_pct,
            significance_threshold_pct=threshold,
            recalculation_required=recalc_required,
            recalculated_base_year_emissions=recalculated,
        )
        result.provenance_hash = _compute_hash(result)

        logger.info(
            "Base year %d assessment: impact=%.1f tCO2e (%.1f%%), threshold=%.0f%%, recalc=%s",
            base_year, float(total_impact), float(total_impact_pct),
            float(threshold), recalc_required,
        )
        return result

    # -------------------------------------------------------------------
    # Target Allocation
    # -------------------------------------------------------------------

    def allocate_targets(
        self,
        group_target_pct: Decimal,
        base_year: int,
        reporting_year: int,
        allocation_type: TargetAllocationType = TargetAllocationType.TOP_DOWN_PROPORTIONAL,
        method: Optional[ConsolidationMethod] = None,
    ) -> List[EntityTargetAllocation]:
        """Allocate group reduction target to individual entities.

        Args:
            group_target_pct: Group-level reduction target (%).
            base_year: Base year for the target.
            reporting_year: Current reporting year.
            allocation_type: Allocation method.
            method: Consolidation method.

        Returns:
            List of EntityTargetAllocation.

        Raises:
            ValueError: If insufficient data.
        """
        if method is None:
            method = self.config.default_consolidation

        prec = self.config.decimal_precision
        group_target_pct = _decimal(group_target_pct)

        # Get base year and current year entities
        base_entities: Dict[str, EntityEmissions] = {}
        for (eid, yr), entity in self._entities.items():
            if yr == base_year:
                base_entities[eid] = entity

        current_entities: Dict[str, EntityEmissions] = {}
        for (eid, yr), entity in self._entities.items():
            if yr == reporting_year:
                current_entities[eid] = entity

        if not base_entities:
            raise ValueError(f"No entity data for base year {base_year}")

        # Calculate total base year emissions (consolidated)
        total_base = Decimal("0")
        for entity in base_entities.values():
            factor = self._get_consolidation_factor(entity, method)
            total_base += entity.total_emissions * factor

        group_target_absolute = _round_val(
            total_base * group_target_pct / Decimal("100"), prec
        )

        allocations: List[EntityTargetAllocation] = []

        for eid, base_entity in base_entities.items():
            factor = self._get_consolidation_factor(base_entity, method)
            entity_base = base_entity.total_emissions * factor

            # Allocation
            if allocation_type == TargetAllocationType.TOP_DOWN_EQUAL:
                n = len(base_entities)
                alloc_target_abs = _safe_divide(
                    group_target_absolute, _decimal(n)
                )
                alloc_target_pct = group_target_pct
            elif allocation_type == TargetAllocationType.TOP_DOWN_PROPORTIONAL:
                proportion = _safe_divide(entity_base, total_base)
                alloc_target_abs = _round_val(group_target_absolute * proportion, prec)
                alloc_target_pct = group_target_pct  # Same % target for all
            else:
                # Bottom-up: entity keeps its own proportion
                proportion = _safe_divide(entity_base, total_base)
                alloc_target_abs = _round_val(group_target_absolute * proportion, prec)
                alloc_target_pct = group_target_pct

            # Calculate progress
            current_entity = current_entities.get(eid)
            current_emissions = Decimal("0")
            progress = Decimal("0")
            on_track = False

            if current_entity:
                current_emissions = current_entity.total_emissions * factor
                if entity_base > 0:
                    reduction_achieved = entity_base - current_emissions
                    required_total = alloc_target_abs
                    if required_total > 0:
                        progress = _safe_pct(reduction_achieved, required_total)
                    on_track = progress >= Decimal("100")

            alloc = EntityTargetAllocation(
                entity_id=eid,
                entity_name=base_entity.entity_name,
                allocated_target_pct=group_target_pct,
                allocated_target_absolute=_round_val(alloc_target_abs, prec),
                current_emissions=_round_val(current_emissions, prec),
                required_reduction=_round_val(alloc_target_abs, prec),
                progress_pct=progress,
                on_track=on_track,
            )
            alloc.provenance_hash = _compute_hash(alloc)
            allocations.append(alloc)

        logger.info(
            "Allocated %.0f%% target across %d entities (%s)",
            float(group_target_pct), len(allocations), allocation_type.value,
        )
        return allocations

    # -------------------------------------------------------------------
    # Completeness Tracking
    # -------------------------------------------------------------------

    def get_completeness_report(
        self, reporting_year: int
    ) -> Dict[str, Any]:
        """Generate a completeness report for a reporting year.

        Args:
            reporting_year: Year to assess.

        Returns:
            Dictionary with completeness metrics.
        """
        year_entities: List[EntityEmissions] = []
        for (eid, yr), entity in self._entities.items():
            if yr == reporting_year:
                year_entities.append(entity)

        status_counts: Dict[str, int] = {s.value: 0 for s in ReportingStatus}
        for entity in year_entities:
            status_counts[entity.reporting_status.value] += 1

        total = len(year_entities)
        actual = status_counts.get("actual", 0)
        completeness = _safe_pct(_decimal(actual), _decimal(total))

        report = {
            "reporting_year": reporting_year,
            "total_entities": total,
            "status_breakdown": status_counts,
            "completeness_pct": str(completeness),
            "entities_missing": [
                {
                    "entity_id": e.entity_id,
                    "entity_name": e.entity_name,
                    "status": e.reporting_status.value,
                }
                for e in year_entities
                if e.reporting_status != ReportingStatus.ACTUAL
            ],
            "provenance_hash": _compute_hash(status_counts),
        }
        return report

    # -------------------------------------------------------------------
    # Progress Aggregation
    # -------------------------------------------------------------------

    def aggregate_progress(
        self,
        base_year: int,
        reporting_year: int,
        group_target_pct: Decimal,
        method: Optional[ConsolidationMethod] = None,
    ) -> Dict[str, Any]:
        """Aggregate weighted progress across all entities.

        Args:
            base_year: Base year.
            reporting_year: Current year.
            group_target_pct: Group reduction target (%).
            method: Consolidation method.

        Returns:
            Dictionary with aggregated progress metrics.
        """
        if method is None:
            method = self.config.default_consolidation

        try:
            base_group = self.consolidate(base_year, method)
            current_group = self.consolidate(reporting_year, method)
        except ValueError as e:
            return {"error": str(e)}

        base_total = base_group.total_emissions
        current_total = current_group.total_emissions

        reduction_target = _round_val(
            base_total * _decimal(group_target_pct) / Decimal("100"),
            self.config.decimal_precision,
        )
        actual_reduction = base_total - current_total
        progress_pct = _safe_pct(actual_reduction, reduction_target)
        on_track = progress_pct >= Decimal("100")

        # Emissions change
        change_pct = Decimal("0")
        if base_total > 0:
            change_pct = _safe_pct(actual_reduction, base_total)

        result = {
            "base_year": base_year,
            "reporting_year": reporting_year,
            "base_year_emissions": str(base_total),
            "current_emissions": str(current_total),
            "reduction_target_pct": str(group_target_pct),
            "reduction_target_absolute": str(reduction_target),
            "actual_reduction": str(actual_reduction),
            "actual_reduction_pct": str(change_pct),
            "progress_pct": str(progress_pct),
            "on_track": on_track,
            "gap_to_target": str(reduction_target - actual_reduction),
            "provenance_hash": _compute_hash({
                "base": str(base_total),
                "current": str(current_total),
                "target": str(group_target_pct),
            }),
        }
        return result

    # -------------------------------------------------------------------
    # Full Pipeline
    # -------------------------------------------------------------------

    def run_full_consolidation(
        self,
        reporting_year: int,
        base_year: Optional[int] = None,
        group_target_pct: Optional[Decimal] = None,
        method: Optional[ConsolidationMethod] = None,
    ) -> ConsolidationResult:
        """Run a complete multi-entity consolidation pipeline.

        Args:
            reporting_year: Year to consolidate.
            base_year: Optional base year for target tracking.
            group_target_pct: Optional group reduction target.
            method: Consolidation method.

        Returns:
            Complete ConsolidationResult.
        """
        if method is None:
            method = self.config.default_consolidation

        logger.info(
            "Running full consolidation for year %d (%s)",
            reporting_year, method.value,
        )

        # Step 1: Consolidate
        group_emissions = self.consolidate(reporting_year, method)

        # Step 2: Get entity breakdown
        entity_breakdown: List[Dict[str, Any]] = []
        for (eid, yr), entity in self._entities.items():
            if yr == reporting_year:
                factor = self._get_consolidation_factor(entity, method)
                entity_breakdown.append({
                    "entity_id": entity.entity_id,
                    "entity_name": entity.entity_name,
                    "entity_type": entity.entity_type.value,
                    "consolidation_factor": str(factor),
                    "scope1": str(entity.scope1_emissions),
                    "scope2_market": str(entity.scope2_market),
                    "scope3": str(entity.scope3_emissions),
                    "total": str(entity.total_emissions),
                    "adjusted_total": str(
                        _round_val(entity.total_emissions * factor, self.config.decimal_precision)
                    ),
                    "reporting_status": entity.reporting_status.value,
                })

        # Step 3: Target allocation (if applicable)
        allocations: List[EntityTargetAllocation] = []
        if base_year and group_target_pct:
            try:
                allocations = self.allocate_targets(
                    group_target_pct, base_year, reporting_year,
                    TargetAllocationType.TOP_DOWN_PROPORTIONAL, method,
                )
            except ValueError as e:
                logger.warning("Target allocation skipped: %s", str(e))

        # Step 4: Base year recalculation assessment
        recalc = None
        if base_year and self._structural_changes:
            try:
                recalc = self.assess_base_year_recalculation(base_year, method)
            except ValueError as e:
                logger.warning("Base year recalc skipped: %s", str(e))

        completeness = group_emissions.completeness_pct

        result = ConsolidationResult(
            group_emissions=group_emissions,
            entity_breakdown=entity_breakdown,
            eliminations=self._eliminations,
            completeness_pct=completeness,
            target_allocation=allocations,
            structural_changes=self._structural_changes,
            base_year_recalculation=recalc,
        )
        result.provenance_hash = _compute_hash(result)

        logger.info(
            "Full consolidation complete: %d entities, total=%.1f tCO2e",
            len(entity_breakdown), float(group_emissions.total_emissions),
        )
        return result

    def clear(self) -> None:
        """Clear all stored data."""
        self._entities.clear()
        self._eliminations.clear()
        self._structural_changes.clear()
        self._group_name = ""
        logger.info("MultiEntityEngine cleared")
