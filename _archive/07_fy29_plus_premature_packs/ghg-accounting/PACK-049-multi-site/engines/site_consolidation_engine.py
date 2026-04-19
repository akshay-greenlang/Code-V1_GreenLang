"""
PACK-049 GHG Multi-Site Management Pack - Site Consolidation Engine
====================================================================

Consolidates GHG emissions from individual sites into a corporate-
level inventory. Applies equity adjustments, eliminates inter-site
transfers, reconciles bottom-up vs top-down totals, restates base
year inventories for structural changes, and provides contribution
and scope-level breakdowns.

Regulatory Basis:
    - GHG Protocol Corporate Standard (Chapter 3): Consolidation
      of emissions from all facilities using the chosen approach.
    - GHG Protocol Corporate Standard (Chapter 5): Recalculating
      base year emissions for structural changes.
    - ISO 14064-1:2018 (Clause 5.3): Quantification of GHG
      emissions at the organisational level.
    - GHG Protocol Corporate Standard (Chapter 7): Managing
      inventory quality - reconciliation and verification.
    - ESRS E1-6: Requires disclosure of consolidation approach
      and per-scope totals.

Calculation Methodology:
    Equity Adjustment:
        adjusted = original * (inclusion_pct / 100)

    Inter-Site Elimination (Double-Count Prevention):
        If site A sells electricity to site B, both in boundary:
            Site A Scope 1 includes generation emissions.
            Site B Scope 2 includes consumption emissions.
            Elimination removes the Scope 2 portion from Site B
            to avoid double-counting.

    Reconciliation:
        variance = bottom_up_total - top_down_total
        variance_pct = (variance / top_down_total) * 100
        within_tolerance = abs(variance_pct) <= tolerance_pct

    Base Year Restatement:
        For structural changes, the base year inventory is
        adjusted to include/exclude the affected sites' emissions
        as if the current boundary had existed in the base year.

Capabilities:
    - Full consolidation pipeline (equity adjustments + eliminations)
    - Equity share, operational control, financial control support
    - Inter-site transfer elimination (energy, waste, product)
    - Bottom-up vs top-down reconciliation
    - Base year restatement for structural changes
    - Per-site contribution analysis
    - Scope-level and category-level breakdowns

Zero-Hallucination:
    - All calculations use deterministic Decimal arithmetic
    - No LLM involvement in any calculation path
    - SHA-256 provenance hash on every result object

Author:  GreenLang Platform Team
Date:    March 2026
Pack:    PACK-049 GHG Multi-Site Management
Engine:  5 of 10
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

def _round4(value: Any) -> Decimal:
    """Round a value to four decimal places using ROUND_HALF_UP."""
    return Decimal(str(value)).quantize(Decimal("0.0001"), rounding=ROUND_HALF_UP)

# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class ConsolidationApproach(str, Enum):
    """GHG Protocol consolidation approaches."""
    EQUITY_SHARE = "EQUITY_SHARE"
    OPERATIONAL_CONTROL = "OPERATIONAL_CONTROL"
    FINANCIAL_CONTROL = "FINANCIAL_CONTROL"

class EliminationType(str, Enum):
    """Types of inter-site transfer eliminations."""
    INTERNAL_ELECTRICITY = "INTERNAL_ELECTRICITY"
    INTERNAL_STEAM = "INTERNAL_STEAM"
    INTERNAL_HEAT = "INTERNAL_HEAT"
    INTERNAL_COOLING = "INTERNAL_COOLING"
    INTERNAL_PRODUCT_TRANSFER = "INTERNAL_PRODUCT_TRANSFER"
    INTERNAL_WASTE_TRANSFER = "INTERNAL_WASTE_TRANSFER"
    INTERNAL_TRANSPORT = "INTERNAL_TRANSPORT"
    OTHER = "OTHER"

class ScopeType(str, Enum):
    """GHG emission scope categories."""
    SCOPE_1 = "SCOPE_1"
    SCOPE_2_LOCATION = "SCOPE_2_LOCATION"
    SCOPE_2_MARKET = "SCOPE_2_MARKET"
    SCOPE_3 = "SCOPE_3"

# ---------------------------------------------------------------------------
# Default Configuration
# ---------------------------------------------------------------------------

DEFAULT_RECONCILIATION_TOLERANCE_PCT = Decimal("5")
DEFAULT_SIGNIFICANCE_THRESHOLD_PCT = Decimal("1")

# ---------------------------------------------------------------------------
# Pydantic Models
# ---------------------------------------------------------------------------

class SiteTotal(BaseModel):
    """GHG emission totals for a single site.

    Contains scope-level totals in tCO2e (tonnes CO2 equivalent).
    Scope 3 is broken down by GHG Protocol category.
    """
    model_config = ConfigDict(
        json_encoders={Decimal: str},
        validate_default=True,
    )

    site_id: str = Field(
        ...,
        description="Site identifier.",
    )
    site_name: Optional[str] = Field(
        None,
        description="Human-readable site name.",
    )
    scope1: Decimal = Field(
        default=Decimal("0"),
        ge=Decimal("0"),
        description="Scope 1 direct emissions (tCO2e).",
    )
    scope2_location: Decimal = Field(
        default=Decimal("0"),
        ge=Decimal("0"),
        description="Scope 2 location-based emissions (tCO2e).",
    )
    scope2_market: Decimal = Field(
        default=Decimal("0"),
        ge=Decimal("0"),
        description="Scope 2 market-based emissions (tCO2e).",
    )
    scope3_categories: Dict[str, Decimal] = Field(
        default_factory=dict,
        description="Scope 3 emissions by category (tCO2e).",
    )
    total: Decimal = Field(
        default=Decimal("0"),
        ge=Decimal("0"),
        description="Total emissions (Scope 1 + 2 + 3) in tCO2e.",
    )
    quality_score: Decimal = Field(
        default=Decimal("3"),
        ge=Decimal("1"),
        le=Decimal("5"),
        description="Data quality score (1-5).",
    )
    is_estimated: bool = Field(
        default=False,
        description="Whether any data was estimated.",
    )
    inclusion_pct: Optional[Decimal] = Field(
        None,
        description="Boundary inclusion percentage (set after equity adj).",
    )
    entity_id: Optional[str] = Field(
        None,
        description="Legal entity owning this site.",
    )

    @field_validator(
        "scope1", "scope2_location", "scope2_market", "total",
        "quality_score", mode="before",
    )
    @classmethod
    def _coerce_decimal(cls, v: Any) -> Any:
        return Decimal(str(v))

    def recalculate_total(self) -> None:
        """Recalculate the total from scope components.

        Uses scope2_location (not market) for the total by default,
        plus all scope 3 categories.
        """
        scope3_total = sum(self.scope3_categories.values(), Decimal("0"))
        self.total = _round2(
            self.scope1 + self.scope2_location + scope3_total
        )

class EliminationEntry(BaseModel):
    """An inter-site transfer elimination entry.

    When two sites within the boundary transfer energy, products,
    or waste between each other, the receiving site's Scope 2 or 3
    emissions may double-count what the sending site already reports
    in Scope 1. Eliminations remove this double-count.
    """
    model_config = ConfigDict(
        json_encoders={Decimal: str},
        validate_default=True,
    )

    entry_id: str = Field(
        default_factory=_new_uuid,
        description="Unique elimination entry ID.",
    )
    from_site_id: str = Field(
        ...,
        description="Site providing the energy/product/service.",
    )
    to_site_id: str = Field(
        ...,
        description="Site receiving the energy/product/service.",
    )
    elimination_type: str = Field(
        ...,
        description="Type of transfer being eliminated.",
    )
    amount: Decimal = Field(
        ...,
        ge=Decimal("0"),
        description="Amount to eliminate (tCO2e).",
    )
    scope: str = Field(
        ...,
        description="Scope to eliminate from (usually SCOPE_2 or SCOPE_3).",
    )
    description: Optional[str] = Field(
        None,
        description="Description of the elimination.",
    )
    evidence_reference: Optional[str] = Field(
        None,
        description="Reference to supporting evidence.",
    )

    @field_validator("amount", mode="before")
    @classmethod
    def _coerce_amount(cls, v: Any) -> Any:
        return Decimal(str(v))

    @field_validator("elimination_type")
    @classmethod
    def _validate_elim_type(cls, v: str) -> str:
        valid = {et.value for et in EliminationType}
        if v.upper() not in valid:
            logger.warning("Elimination type '%s' not standard; accepted.", v)
        return v.upper()

class EquityAdjustment(BaseModel):
    """Record of an equity-share adjustment to site emissions.

    Documents the adjustment from 100% site emissions to the
    organisation's equity share.
    """
    model_config = ConfigDict(
        json_encoders={Decimal: str},
        validate_default=True,
    )

    site_id: str = Field(
        ...,
        description="The adjusted site.",
    )
    original_total: Decimal = Field(
        ...,
        description="Original total emissions before adjustment.",
    )
    inclusion_pct: Decimal = Field(
        ...,
        ge=Decimal("0"),
        le=Decimal("100"),
        description="Inclusion percentage applied.",
    )
    adjusted_total: Decimal = Field(
        ...,
        description="Emissions after equity adjustment.",
    )
    scope1_original: Decimal = Field(
        default=Decimal("0"), description="Original Scope 1."
    )
    scope1_adjusted: Decimal = Field(
        default=Decimal("0"), description="Adjusted Scope 1."
    )
    scope2_location_original: Decimal = Field(
        default=Decimal("0"), description="Original Scope 2 (location)."
    )
    scope2_location_adjusted: Decimal = Field(
        default=Decimal("0"), description="Adjusted Scope 2 (location)."
    )
    scope2_market_original: Decimal = Field(
        default=Decimal("0"), description="Original Scope 2 (market)."
    )
    scope2_market_adjusted: Decimal = Field(
        default=Decimal("0"), description="Adjusted Scope 2 (market)."
    )

    @field_validator(
        "original_total", "inclusion_pct", "adjusted_total",
        "scope1_original", "scope1_adjusted",
        "scope2_location_original", "scope2_location_adjusted",
        "scope2_market_original", "scope2_market_adjusted",
        mode="before",
    )
    @classmethod
    def _coerce_decimal(cls, v: Any) -> Any:
        return Decimal(str(v))

class ReconciliationResult(BaseModel):
    """Result of reconciling bottom-up vs top-down totals.

    Compares the sum of individual site totals (bottom-up) against
    an independent corporate-level total (top-down), such as from
    financial statements or utility billing data.
    """
    model_config = ConfigDict(
        json_encoders={Decimal: str},
        validate_default=True,
    )

    bottom_up_total: Decimal = Field(
        ...,
        description="Sum of consolidated site totals (tCO2e).",
    )
    top_down_total: Optional[Decimal] = Field(
        None,
        description="Independent corporate-level total (tCO2e).",
    )
    variance: Decimal = Field(
        default=Decimal("0"),
        description="Absolute variance (bottom_up - top_down).",
    )
    variance_pct: Decimal = Field(
        default=Decimal("0"),
        description="Variance as percentage of top-down total.",
    )
    within_tolerance: bool = Field(
        default=True,
        description="Whether variance is within acceptable tolerance.",
    )
    tolerance_pct: Decimal = Field(
        default=DEFAULT_RECONCILIATION_TOLERANCE_PCT,
        description="Tolerance threshold used.",
    )
    unreconciled_items: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Items that could not be reconciled.",
    )
    reconciliation_notes: Optional[str] = Field(
        None,
        description="Notes from the reconciliation process.",
    )
    provenance_hash: str = Field(
        default="",
        description="SHA-256 hash.",
    )

    @field_validator(
        "bottom_up_total", "variance", "variance_pct",
        "tolerance_pct", mode="before",
    )
    @classmethod
    def _coerce_decimal(cls, v: Any) -> Any:
        return Decimal(str(v))

class ConsolidationRun(BaseModel):
    """A complete consolidation run for a reporting period.

    Captures all inputs, adjustments, eliminations, reconciliation,
    and final consolidated totals.
    """
    model_config = ConfigDict(
        json_encoders={Decimal: str},
        validate_default=True,
    )

    run_id: str = Field(
        default_factory=_new_uuid,
        description="Unique consolidation run identifier.",
    )
    reporting_year: int = Field(
        ...,
        ge=2000,
        le=2100,
        description="Reporting year.",
    )
    boundary_id: str = Field(
        ...,
        description="Boundary definition used.",
    )
    consolidation_approach: str = Field(
        ...,
        description="Consolidation approach applied.",
    )
    site_totals: List[SiteTotal] = Field(
        default_factory=list,
        description="Per-site totals (after adjustments).",
    )
    original_site_totals: List[SiteTotal] = Field(
        default_factory=list,
        description="Per-site totals (before adjustments).",
    )
    eliminations: List[EliminationEntry] = Field(
        default_factory=list,
        description="Inter-site eliminations applied.",
    )
    equity_adjustments: List[EquityAdjustment] = Field(
        default_factory=list,
        description="Equity adjustments applied.",
    )
    reconciliation: Optional[ReconciliationResult] = Field(
        None,
        description="Reconciliation result.",
    )
    consolidated_scope1: Decimal = Field(
        default=Decimal("0"),
        description="Total consolidated Scope 1 (tCO2e).",
    )
    consolidated_scope2_location: Decimal = Field(
        default=Decimal("0"),
        description="Total consolidated Scope 2 location-based (tCO2e).",
    )
    consolidated_scope2_market: Decimal = Field(
        default=Decimal("0"),
        description="Total consolidated Scope 2 market-based (tCO2e).",
    )
    consolidated_scope3: Decimal = Field(
        default=Decimal("0"),
        description="Total consolidated Scope 3 (tCO2e).",
    )
    consolidated_total: Decimal = Field(
        default=Decimal("0"),
        description="Grand total (S1 + S2_location + S3) in tCO2e.",
    )
    total_eliminations: Decimal = Field(
        default=Decimal("0"),
        description="Total eliminated emissions (tCO2e).",
    )
    completeness_pct: Decimal = Field(
        default=Decimal("0"),
        description="Data completeness percentage.",
    )
    site_count: int = Field(
        default=0,
        description="Number of sites included.",
    )
    is_restated: bool = Field(
        default=False,
        description="Whether this is a restated base year run.",
    )
    restatement_reason: Optional[str] = Field(
        None,
        description="Reason for restatement.",
    )
    created_at: datetime = Field(
        default_factory=utcnow,
        description="When the run was created.",
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

class ContributionAnalysis(BaseModel):
    """Contribution analysis showing each site's share of total."""
    model_config = ConfigDict(
        json_encoders={Decimal: str},
        validate_default=True,
    )

    run_id: str = Field(
        ..., description="The consolidation run analysed."
    )
    total_emissions: Decimal = Field(
        ..., description="Total consolidated emissions."
    )
    site_contributions: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Per-site contributions sorted by share descending.",
    )
    top_10_pct: Decimal = Field(
        default=Decimal("0"),
        description="Cumulative share of top 10 sites.",
    )
    bottom_10_pct: Decimal = Field(
        default=Decimal("0"),
        description="Cumulative share of bottom 10 sites.",
    )
    provenance_hash: str = Field(
        default="", description="SHA-256 hash."
    )

class ScopeBreakdown(BaseModel):
    """Scope-level breakdown of consolidated emissions."""
    model_config = ConfigDict(
        json_encoders={Decimal: str},
        validate_default=True,
    )

    run_id: str = Field(
        ..., description="The consolidation run."
    )
    scope1_total: Decimal = Field(
        default=Decimal("0"), description="Total Scope 1."
    )
    scope1_pct: Decimal = Field(
        default=Decimal("0"), description="Scope 1 as % of total."
    )
    scope2_location_total: Decimal = Field(
        default=Decimal("0"), description="Total Scope 2 (location)."
    )
    scope2_location_pct: Decimal = Field(
        default=Decimal("0"), description="Scope 2 location as % of total."
    )
    scope2_market_total: Decimal = Field(
        default=Decimal("0"), description="Total Scope 2 (market)."
    )
    scope2_market_pct: Decimal = Field(
        default=Decimal("0"), description="Scope 2 market as % of total."
    )
    scope3_total: Decimal = Field(
        default=Decimal("0"), description="Total Scope 3."
    )
    scope3_pct: Decimal = Field(
        default=Decimal("0"), description="Scope 3 as % of total."
    )
    scope3_by_category: Dict[str, Decimal] = Field(
        default_factory=dict, description="Scope 3 by category."
    )
    grand_total: Decimal = Field(
        default=Decimal("0"), description="Grand total (S1+S2loc+S3)."
    )
    provenance_hash: str = Field(
        default="", description="SHA-256 hash."
    )

# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

class SiteConsolidationEngine:
    """Consolidates site-level GHG data into corporate inventory.

    Implements the full consolidation pipeline: equity adjustments,
    inter-site elimination, aggregation, reconciliation, and
    base year restatement.

    Attributes:
        _runs: Dict mapping run_id to ConsolidationRun.

    Example:
        >>> engine = SiteConsolidationEngine()
        >>> run = engine.consolidate(
        ...     boundary=boundary_def,
        ...     site_totals=[site1_totals, site2_totals],
        ... )
        >>> assert run.consolidated_total > Decimal("0")
    """

    def __init__(self) -> None:
        """Initialise the SiteConsolidationEngine."""
        self._runs: Dict[str, ConsolidationRun] = {}
        logger.info("SiteConsolidationEngine v%s initialised.", _MODULE_VERSION)

    # ------------------------------------------------------------------
    # Full Consolidation Pipeline
    # ------------------------------------------------------------------

    def consolidate(
        self,
        boundary_id: str,
        reporting_year: int,
        consolidation_approach: str,
        site_totals: List[SiteTotal],
        boundary_inclusions: Optional[Dict[str, Decimal]] = None,
        eliminations_data: Optional[List[Dict[str, Any]]] = None,
        top_down_total: Optional[Union[Decimal, str, int, float]] = None,
        tolerance_pct: Optional[Union[Decimal, str, int, float]] = None,
    ) -> ConsolidationRun:
        """Execute full consolidation pipeline.

        Steps:
            1. Store original site totals for audit trail
            2. Apply equity adjustments (if equity share approach)
            3. Identify and apply inter-site eliminations
            4. Aggregate to consolidated totals
            5. Reconcile against top-down total (if provided)
            6. Compute completeness and quality metrics
            7. Generate provenance hash

        Args:
            boundary_id: The boundary definition ID.
            reporting_year: The reporting year.
            consolidation_approach: Consolidation approach used.
            site_totals: Per-site emission totals.
            boundary_inclusions: Dict mapping site_id to inclusion_pct.
                Required for EQUITY_SHARE approach.
            eliminations_data: Optional list of elimination dicts.
            top_down_total: Optional corporate-level total for recon.
            tolerance_pct: Reconciliation tolerance percentage.

        Returns:
            Complete ConsolidationRun with all results.
        """
        logger.info(
            "Starting consolidation for year %d, boundary '%s', "
            "%d site(s), approach=%s.",
            reporting_year, boundary_id, len(site_totals),
            consolidation_approach,
        )

        # Store originals
        original_totals = [
            st.model_copy(deep=True) for st in site_totals
        ]

        # Step 1: Apply equity adjustments
        adjusted_totals = list(site_totals)
        equity_adjustments: List[EquityAdjustment] = []

        if boundary_inclusions:
            adjusted_totals, equity_adjustments = self.apply_equity_adjustments(
                site_totals, boundary_inclusions
            )

        # Step 2: Parse and apply eliminations
        eliminations: List[EliminationEntry] = []
        if eliminations_data:
            eliminations = self.identify_eliminations(
                adjusted_totals, eliminations_data
            )
            adjusted_totals = self.apply_eliminations(
                adjusted_totals, eliminations
            )

        # Step 3: Aggregate consolidated totals
        cons_s1 = Decimal("0")
        cons_s2_loc = Decimal("0")
        cons_s2_mkt = Decimal("0")
        cons_s3 = Decimal("0")

        for st in adjusted_totals:
            cons_s1 += st.scope1
            cons_s2_loc += st.scope2_location
            cons_s2_mkt += st.scope2_market
            cons_s3 += sum(st.scope3_categories.values(), Decimal("0"))

        cons_total = _round2(cons_s1 + cons_s2_loc + cons_s3)
        total_eliminated = sum(
            (e.amount for e in eliminations), Decimal("0")
        )

        # Step 4: Reconciliation
        reconciliation = None
        if top_down_total is not None:
            tol = (
                _decimal(tolerance_pct)
                if tolerance_pct is not None
                else DEFAULT_RECONCILIATION_TOLERANCE_PCT
            )
            reconciliation = self.reconcile(
                bottom_up=cons_total,
                top_down=_decimal(top_down_total),
                tolerance=tol,
            )

        # Step 5: Completeness
        non_estimated_count = sum(
            1 for st in adjusted_totals if not st.is_estimated
        )
        total_sites = len(adjusted_totals)
        completeness_pct = _round2(
            _safe_divide(
                _decimal(non_estimated_count),
                _decimal(total_sites),
            ) * Decimal("100")
        ) if total_sites > 0 else Decimal("0")

        # Build run
        run = ConsolidationRun(
            reporting_year=reporting_year,
            boundary_id=boundary_id,
            consolidation_approach=consolidation_approach,
            site_totals=adjusted_totals,
            original_site_totals=original_totals,
            eliminations=eliminations,
            equity_adjustments=equity_adjustments,
            reconciliation=reconciliation,
            consolidated_scope1=_round2(cons_s1),
            consolidated_scope2_location=_round2(cons_s2_loc),
            consolidated_scope2_market=_round2(cons_s2_mkt),
            consolidated_scope3=_round2(cons_s3),
            consolidated_total=cons_total,
            total_eliminations=_round2(total_eliminated),
            completeness_pct=completeness_pct,
            site_count=total_sites,
        )
        run.provenance_hash = _compute_hash(run)
        self._runs[run.run_id] = run

        logger.info(
            "Consolidation complete: S1=%s, S2loc=%s, S2mkt=%s, "
            "S3=%s, Total=%s tCO2e, %d site(s), %d elimination(s).",
            run.consolidated_scope1,
            run.consolidated_scope2_location,
            run.consolidated_scope2_market,
            run.consolidated_scope3,
            run.consolidated_total,
            total_sites,
            len(eliminations),
        )
        return run

    # ------------------------------------------------------------------
    # Equity Adjustments
    # ------------------------------------------------------------------

    def apply_equity_adjustments(
        self,
        site_totals: List[SiteTotal],
        boundary_inclusions: Dict[str, Decimal],
    ) -> Tuple[List[SiteTotal], List[EquityAdjustment]]:
        """Apply equity-share adjustments to site totals.

        For each site, multiplies all emission values by the
        inclusion percentage (equity share / 100).

        Args:
            site_totals: Original site emission totals.
            boundary_inclusions: Dict mapping site_id to inclusion_pct
                (0-100).

        Returns:
            Tuple of (adjusted SiteTotals, EquityAdjustment records).
        """
        logger.info(
            "Applying equity adjustments to %d site(s).",
            len(site_totals),
        )

        adjusted: List[SiteTotal] = []
        adjustments: List[EquityAdjustment] = []

        for st in site_totals:
            inclusion_pct = boundary_inclusions.get(
                st.site_id, Decimal("100")
            )
            multiplier = _safe_divide(inclusion_pct, Decimal("100"))

            if inclusion_pct == Decimal("100"):
                # No adjustment needed
                adjusted.append(st.model_copy(update={
                    "inclusion_pct": Decimal("100"),
                }))
                continue

            # Apply to all scopes
            adj_s1 = _round2(st.scope1 * multiplier)
            adj_s2_loc = _round2(st.scope2_location * multiplier)
            adj_s2_mkt = _round2(st.scope2_market * multiplier)

            adj_s3: Dict[str, Decimal] = {}
            for cat, val in st.scope3_categories.items():
                adj_s3[cat] = _round2(val * multiplier)

            original_total = st.total
            if original_total == Decimal("0"):
                s3_sum = sum(st.scope3_categories.values(), Decimal("0"))
                original_total = st.scope1 + st.scope2_location + s3_sum

            adj_total = _round2(
                adj_s1 + adj_s2_loc + sum(adj_s3.values(), Decimal("0"))
            )

            adjusted_st = st.model_copy(update={
                "scope1": adj_s1,
                "scope2_location": adj_s2_loc,
                "scope2_market": adj_s2_mkt,
                "scope3_categories": adj_s3,
                "total": adj_total,
                "inclusion_pct": inclusion_pct,
            })
            adjusted.append(adjusted_st)

            adjustments.append(EquityAdjustment(
                site_id=st.site_id,
                original_total=_round2(original_total),
                inclusion_pct=inclusion_pct,
                adjusted_total=adj_total,
                scope1_original=st.scope1,
                scope1_adjusted=adj_s1,
                scope2_location_original=st.scope2_location,
                scope2_location_adjusted=adj_s2_loc,
                scope2_market_original=st.scope2_market,
                scope2_market_adjusted=adj_s2_mkt,
            ))

            logger.debug(
                "Site '%s': %s * %s%% = %s tCO2e.",
                st.site_id,
                original_total,
                inclusion_pct,
                adj_total,
            )

        logger.info(
            "Equity adjustments applied: %d adjustment(s).",
            len(adjustments),
        )
        return adjusted, adjustments

    # ------------------------------------------------------------------
    # Eliminations
    # ------------------------------------------------------------------

    def identify_eliminations(
        self,
        site_totals: List[SiteTotal],
        elimination_rules: List[Dict[str, Any]],
    ) -> List[EliminationEntry]:
        """Identify inter-site transfers that need elimination.

        Parses elimination rule definitions and creates
        EliminationEntry records.

        Args:
            site_totals: Current site totals.
            elimination_rules: List of rule dicts with keys:
                from_site_id, to_site_id, elimination_type, amount,
                scope, description.

        Returns:
            List of EliminationEntry records.
        """
        logger.info(
            "Identifying eliminations from %d rule(s).",
            len(elimination_rules),
        )

        site_ids = {st.site_id for st in site_totals}
        eliminations: List[EliminationEntry] = []

        for rule in elimination_rules:
            from_id = rule.get("from_site_id", "")
            to_id = rule.get("to_site_id", "")

            # Both sites must be in the boundary
            if from_id not in site_ids:
                logger.warning(
                    "Elimination from_site '%s' not in boundary; skipped.",
                    from_id,
                )
                continue
            if to_id not in site_ids:
                logger.warning(
                    "Elimination to_site '%s' not in boundary; skipped.",
                    to_id,
                )
                continue

            amount = _decimal(rule.get("amount", "0"))
            if amount <= Decimal("0"):
                logger.warning(
                    "Elimination amount <= 0 between '%s' and '%s'; skipped.",
                    from_id, to_id,
                )
                continue

            elimination = EliminationEntry(
                from_site_id=from_id,
                to_site_id=to_id,
                elimination_type=rule.get(
                    "elimination_type", EliminationType.OTHER.value
                ),
                amount=amount,
                scope=rule.get("scope", ScopeType.SCOPE_2_LOCATION.value),
                description=rule.get("description"),
                evidence_reference=rule.get("evidence_reference"),
            )
            eliminations.append(elimination)

        logger.info("Identified %d valid elimination(s).", len(eliminations))
        return eliminations

    def apply_eliminations(
        self,
        site_totals: List[SiteTotal],
        eliminations: List[EliminationEntry],
    ) -> List[SiteTotal]:
        """Apply elimination entries to site totals.

        Subtracts the elimination amount from the receiving site's
        appropriate scope to prevent double-counting.

        Args:
            site_totals: Current site totals.
            eliminations: Elimination entries to apply.

        Returns:
            Adjusted site totals after eliminations.
        """
        if not eliminations:
            return site_totals

        logger.info(
            "Applying %d elimination(s) to site totals.",
            len(eliminations),
        )

        # Index by site_id for fast lookup
        totals_map: Dict[str, SiteTotal] = {
            st.site_id: st.model_copy(deep=True) for st in site_totals
        }

        for elim in eliminations:
            target = totals_map.get(elim.to_site_id)
            if target is None:
                continue

            scope = elim.scope.upper()
            amount = elim.amount

            if scope == ScopeType.SCOPE_1.value:
                target.scope1 = max(
                    Decimal("0"), _round2(target.scope1 - amount)
                )
            elif scope == ScopeType.SCOPE_2_LOCATION.value:
                target.scope2_location = max(
                    Decimal("0"), _round2(target.scope2_location - amount)
                )
            elif scope == ScopeType.SCOPE_2_MARKET.value:
                target.scope2_market = max(
                    Decimal("0"), _round2(target.scope2_market - amount)
                )
            elif scope == ScopeType.SCOPE_3.value:
                # Deduct proportionally from scope 3 categories
                s3_total = sum(
                    target.scope3_categories.values(), Decimal("0")
                )
                if s3_total > Decimal("0"):
                    for cat in target.scope3_categories:
                        share = _safe_divide(
                            target.scope3_categories[cat], s3_total
                        )
                        deduction = _round2(amount * share)
                        target.scope3_categories[cat] = max(
                            Decimal("0"),
                            target.scope3_categories[cat] - deduction,
                        )

            # Recalculate total
            target.recalculate_total()
            totals_map[elim.to_site_id] = target

            logger.debug(
                "Elimination: %s tCO2e from '%s' (%s) on site '%s'.",
                amount, elim.from_site_id, scope, elim.to_site_id,
            )

        # Preserve original ordering
        result: List[SiteTotal] = []
        for st in site_totals:
            result.append(totals_map[st.site_id])

        return result

    # ------------------------------------------------------------------
    # Reconciliation
    # ------------------------------------------------------------------

    def reconcile(
        self,
        bottom_up: Union[Decimal, str, int, float],
        top_down: Union[Decimal, str, int, float],
        tolerance: Union[Decimal, str, int, float, None] = None,
        unreconciled_items: Optional[List[Dict[str, Any]]] = None,
    ) -> ReconciliationResult:
        """Reconcile bottom-up site totals against a top-down total.

        Args:
            bottom_up: Sum of consolidated site emissions.
            top_down: Independent corporate-level total.
            tolerance: Acceptable variance percentage.
            unreconciled_items: List of known unreconciled items.

        Returns:
            ReconciliationResult with variance analysis.
        """
        bu = _decimal(bottom_up)
        td = _decimal(top_down)
        tol = _decimal(
            tolerance if tolerance is not None
            else DEFAULT_RECONCILIATION_TOLERANCE_PCT
        )

        variance = _round2(bu - td)
        variance_pct = _round2(
            _safe_divide(variance, td) * Decimal("100")
        ) if td != Decimal("0") else Decimal("0")

        within = abs(variance_pct) <= tol

        result = ReconciliationResult(
            bottom_up_total=bu,
            top_down_total=td,
            variance=variance,
            variance_pct=variance_pct,
            within_tolerance=within,
            tolerance_pct=tol,
            unreconciled_items=unreconciled_items or [],
        )
        result.provenance_hash = _compute_hash(result)

        status_str = "WITHIN" if within else "OUTSIDE"
        logger.info(
            "Reconciliation: BU=%s, TD=%s, variance=%s (%s%%), %s tolerance.",
            bu, td, variance, variance_pct, status_str,
        )
        return result

    # ------------------------------------------------------------------
    # Base Year Restatement
    # ------------------------------------------------------------------

    def restate_base_year(
        self,
        original_run: ConsolidationRun,
        boundary_changes: List[Dict[str, Any]],
    ) -> ConsolidationRun:
        """Restate base year emissions for structural changes.

        Per GHG Protocol Chapter 5, the base year inventory must be
        recalculated when structural changes (acquisitions, divestitures,
        mergers, etc.) meet the significance threshold.

        For additions (acquisitions, mergers): adds the new site's
        emissions to the base year as if they were always included.
        For removals (divestitures): removes the site's emissions.

        Args:
            original_run: The original base year consolidation run.
            boundary_changes: List of change dicts with keys:
                change_type ('ADD' or 'REMOVE'), site_id,
                emissions (Dict with scope1, scope2_location, etc.),
                reason.

        Returns:
            A new restated ConsolidationRun.
        """
        logger.info(
            "Restating base year run '%s' (year %d) for %d change(s).",
            original_run.run_id,
            original_run.reporting_year,
            len(boundary_changes),
        )

        restated_totals = [
            st.model_copy(deep=True) for st in original_run.site_totals
        ]
        existing_ids = {st.site_id for st in restated_totals}

        for change in boundary_changes:
            change_type = change.get("change_type", "").upper()
            site_id = change.get("site_id", "")
            emissions = change.get("emissions", {})
            reason = change.get("reason", "Structural change")

            if change_type == "ADD":
                if site_id in existing_ids:
                    logger.warning(
                        "Site '%s' already in base year; skipping ADD.",
                        site_id,
                    )
                    continue

                new_total = SiteTotal(
                    site_id=site_id,
                    site_name=change.get("site_name"),
                    scope1=_decimal(emissions.get("scope1", "0")),
                    scope2_location=_decimal(
                        emissions.get("scope2_location", "0")
                    ),
                    scope2_market=_decimal(
                        emissions.get("scope2_market", "0")
                    ),
                    scope3_categories={
                        k: _decimal(v)
                        for k, v in emissions.get(
                            "scope3_categories", {}
                        ).items()
                    },
                    is_estimated=change.get("is_estimated", True),
                    quality_score=_decimal(
                        change.get("quality_score", "2")
                    ),
                )
                new_total.recalculate_total()
                restated_totals.append(new_total)
                existing_ids.add(site_id)
                logger.info(
                    "Added site '%s' to base year (%s tCO2e).",
                    site_id, new_total.total,
                )

            elif change_type == "REMOVE":
                restated_totals = [
                    st for st in restated_totals
                    if st.site_id != site_id
                ]
                existing_ids.discard(site_id)
                logger.info("Removed site '%s' from base year.", site_id)

            else:
                logger.warning(
                    "Unknown change_type '%s'; skipping.", change_type
                )

        # Re-aggregate
        cons_s1 = sum((st.scope1 for st in restated_totals), Decimal("0"))
        cons_s2_loc = sum(
            (st.scope2_location for st in restated_totals), Decimal("0")
        )
        cons_s2_mkt = sum(
            (st.scope2_market for st in restated_totals), Decimal("0")
        )
        cons_s3 = Decimal("0")
        for st in restated_totals:
            cons_s3 += sum(st.scope3_categories.values(), Decimal("0"))

        cons_total = _round2(cons_s1 + cons_s2_loc + cons_s3)

        restated_run = ConsolidationRun(
            reporting_year=original_run.reporting_year,
            boundary_id=original_run.boundary_id,
            consolidation_approach=original_run.consolidation_approach,
            site_totals=restated_totals,
            original_site_totals=original_run.original_site_totals,
            eliminations=original_run.eliminations,
            equity_adjustments=original_run.equity_adjustments,
            consolidated_scope1=_round2(cons_s1),
            consolidated_scope2_location=_round2(cons_s2_loc),
            consolidated_scope2_market=_round2(cons_s2_mkt),
            consolidated_scope3=_round2(cons_s3),
            consolidated_total=cons_total,
            completeness_pct=original_run.completeness_pct,
            site_count=len(restated_totals),
            is_restated=True,
            restatement_reason=(
                f"Base year restated for {len(boundary_changes)} "
                f"structural change(s)."
            ),
        )
        restated_run.provenance_hash = _compute_hash(restated_run)
        self._runs[restated_run.run_id] = restated_run

        logger.info(
            "Base year restated: original=%s, restated=%s tCO2e (%+s).",
            original_run.consolidated_total,
            restated_run.consolidated_total,
            restated_run.consolidated_total - original_run.consolidated_total,
        )
        return restated_run

    # ------------------------------------------------------------------
    # Contribution Analysis
    # ------------------------------------------------------------------

    def get_contribution_analysis(
        self,
        run: ConsolidationRun,
    ) -> ContributionAnalysis:
        """Analyse each site's contribution to the consolidated total.

        Calculates the percentage share of each site and identifies
        the top and bottom contributors.

        Args:
            run: The consolidation run to analyse.

        Returns:
            ContributionAnalysis with per-site contributions.
        """
        logger.info(
            "Generating contribution analysis for run '%s'.",
            run.run_id,
        )

        total = run.consolidated_total
        contributions: List[Dict[str, Any]] = []

        for st in run.site_totals:
            site_total = st.total
            if site_total == Decimal("0") and total == Decimal("0"):
                share = Decimal("0")
            else:
                share = _round4(
                    _safe_divide(site_total, total) * Decimal("100")
                )

            contributions.append({
                "site_id": st.site_id,
                "site_name": st.site_name or st.site_id,
                "total_emissions": str(_round2(site_total)),
                "contribution_pct": str(share),
                "scope1": str(_round2(st.scope1)),
                "scope2_location": str(_round2(st.scope2_location)),
                "scope3": str(_round2(
                    sum(st.scope3_categories.values(), Decimal("0"))
                )),
                "quality_score": str(st.quality_score),
                "is_estimated": st.is_estimated,
            })

        # Sort by contribution descending
        contributions.sort(
            key=lambda x: _decimal(x["contribution_pct"]),
            reverse=True,
        )

        # Top and bottom 10
        top_10_pct = Decimal("0")
        for c in contributions[:10]:
            top_10_pct += _decimal(c["contribution_pct"])

        bottom_10_pct = Decimal("0")
        for c in contributions[-10:]:
            bottom_10_pct += _decimal(c["contribution_pct"])

        result = ContributionAnalysis(
            run_id=run.run_id,
            total_emissions=total,
            site_contributions=contributions,
            top_10_pct=_round2(top_10_pct),
            bottom_10_pct=_round2(bottom_10_pct),
        )
        result.provenance_hash = _compute_hash(result)

        logger.info(
            "Contribution analysis: %d site(s), top-10 = %s%% of total.",
            len(contributions),
            result.top_10_pct,
        )
        return result

    # ------------------------------------------------------------------
    # Scope Breakdown
    # ------------------------------------------------------------------

    def get_scope_breakdown(
        self,
        run: ConsolidationRun,
    ) -> ScopeBreakdown:
        """Get scope-level breakdown of consolidated emissions.

        Args:
            run: The consolidation run.

        Returns:
            ScopeBreakdown with per-scope totals and percentages.
        """
        total = run.consolidated_total

        # Scope percentages
        s1_pct = _round2(
            _safe_divide(run.consolidated_scope1, total) * Decimal("100")
        ) if total > Decimal("0") else Decimal("0")

        s2_loc_pct = _round2(
            _safe_divide(
                run.consolidated_scope2_location, total
            ) * Decimal("100")
        ) if total > Decimal("0") else Decimal("0")

        s2_mkt_pct = _round2(
            _safe_divide(
                run.consolidated_scope2_market, total
            ) * Decimal("100")
        ) if total > Decimal("0") else Decimal("0")

        s3_pct = _round2(
            _safe_divide(
                run.consolidated_scope3, total
            ) * Decimal("100")
        ) if total > Decimal("0") else Decimal("0")

        # Scope 3 by category
        s3_by_cat: Dict[str, Decimal] = {}
        for st in run.site_totals:
            for cat, val in st.scope3_categories.items():
                s3_by_cat[cat] = s3_by_cat.get(cat, Decimal("0")) + val

        for cat in s3_by_cat:
            s3_by_cat[cat] = _round2(s3_by_cat[cat])

        result = ScopeBreakdown(
            run_id=run.run_id,
            scope1_total=run.consolidated_scope1,
            scope1_pct=s1_pct,
            scope2_location_total=run.consolidated_scope2_location,
            scope2_location_pct=s2_loc_pct,
            scope2_market_total=run.consolidated_scope2_market,
            scope2_market_pct=s2_mkt_pct,
            scope3_total=run.consolidated_scope3,
            scope3_pct=s3_pct,
            scope3_by_category=s3_by_cat,
            grand_total=total,
        )
        result.provenance_hash = _compute_hash(result)

        logger.info(
            "Scope breakdown: S1=%s%%, S2loc=%s%%, S3=%s%%.",
            s1_pct, s2_loc_pct, s3_pct,
        )
        return result

    # ------------------------------------------------------------------
    # Variance Analysis
    # ------------------------------------------------------------------

    def compare_runs(
        self,
        run_a: ConsolidationRun,
        run_b: ConsolidationRun,
    ) -> Dict[str, Any]:
        """Compare two consolidation runs (e.g., year-over-year).

        Args:
            run_a: First run (typically earlier period).
            run_b: Second run (typically later period).

        Returns:
            Dictionary with scope-level and total variances.
        """
        def _variance(a: Decimal, b: Decimal) -> Dict[str, str]:
            diff = b - a
            pct = _round2(
                _safe_divide(diff, a) * Decimal("100")
            ) if a != Decimal("0") else Decimal("0")
            return {
                "period_a": str(_round2(a)),
                "period_b": str(_round2(b)),
                "absolute_change": str(_round2(diff)),
                "pct_change": str(pct),
            }

        comparison = {
            "run_a_id": run_a.run_id,
            "run_b_id": run_b.run_id,
            "year_a": run_a.reporting_year,
            "year_b": run_b.reporting_year,
            "scope1": _variance(
                run_a.consolidated_scope1, run_b.consolidated_scope1
            ),
            "scope2_location": _variance(
                run_a.consolidated_scope2_location,
                run_b.consolidated_scope2_location,
            ),
            "scope2_market": _variance(
                run_a.consolidated_scope2_market,
                run_b.consolidated_scope2_market,
            ),
            "scope3": _variance(
                run_a.consolidated_scope3, run_b.consolidated_scope3
            ),
            "total": _variance(
                run_a.consolidated_total, run_b.consolidated_total
            ),
            "site_count_a": run_a.site_count,
            "site_count_b": run_b.site_count,
            "site_count_change": run_b.site_count - run_a.site_count,
        }
        comparison["provenance_hash"] = _compute_hash(comparison)

        logger.info(
            "Run comparison: %d vs %d, total change=%s%%.",
            run_a.reporting_year,
            run_b.reporting_year,
            comparison["total"]["pct_change"],
        )
        return comparison

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    def get_run(self, run_id: str) -> ConsolidationRun:
        """Retrieve a consolidation run by ID.

        Args:
            run_id: The run ID.

        Returns:
            The ConsolidationRun.

        Raises:
            KeyError: If not found.
        """
        if run_id not in self._runs:
            raise KeyError(f"Consolidation run '{run_id}' not found.")
        return self._runs[run_id]

    def get_runs_for_year(self, year: int) -> List[ConsolidationRun]:
        """Get all consolidation runs for a year.

        Args:
            year: The reporting year.

        Returns:
            List of ConsolidationRuns.
        """
        return [
            r for r in self._runs.values()
            if r.reporting_year == year
        ]

    def get_all_runs(self) -> List[ConsolidationRun]:
        """Return all consolidation runs.

        Returns:
            List of all ConsolidationRuns.
        """
        return list(self._runs.values())
