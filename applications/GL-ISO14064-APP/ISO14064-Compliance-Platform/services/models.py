"""
ISO 14064-1:2018 Compliance Platform Domain Models

This module defines all Pydantic v2 domain models for the GL-ISO14064-APP v1.0
platform.  Models cover the full ISO 14064-1:2018 lifecycle: organizations,
boundaries, inventories, category emissions, GHG removals, significance
assessments, uncertainty analysis, data quality management, verification
workflows, cross-walk mappings, and report generation.

ISO 14064-1:2018 departs from the GHG Protocol scope model in favor of six
emission/removal categories.  These models faithfully implement that structure
while maintaining compatibility with the GreenLang MRV agent layer.

All monetary values are in USD.  All emissions are in metric tonnes CO2e
unless otherwise noted.  Timestamps are UTC.

Example:
    >>> org = Organization(name="Acme Corp", industry="manufacturing", country="US")
    >>> print(org.id)
"""

from __future__ import annotations

import hashlib
import uuid
from datetime import date, datetime
from decimal import Decimal
from typing import Any, Dict, List, Optional

from pydantic import Field, field_validator, model_validator
from greenlang.schemas import GreenLangBase, utcnow, new_uuid

from .config import (
    ActionCategory,
    ActionStatus,
    ConsolidationApproach,
    DataQualityTier,
    FindingSeverity,
    FindingStatus,
    GHGGas,
    GWPSource,
    InventoryStatus,
    ISOCategory,
    PermanenceLevel,
    QuantificationMethod,
    RemovalType,
    ReportFormat,
    ReportStatus,
    ReportingPeriod,
    SignificanceLevel,
    VerificationLevel,
    VerificationStage,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _new_id() -> str:
    """Generate a deterministic-safe UUID4 string."""
    return str(uuid.uuid4())


def _now() -> datetime:
    """UTC now truncated to seconds."""
    return datetime.utcnow().replace(microsecond=0)


def _sha256(payload: str) -> str:
    """SHA-256 hex digest for provenance tracking."""
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Organization & Entity Models
# ---------------------------------------------------------------------------

class Entity(GreenLangBase):
    """
    An organizational entity (subsidiary, facility, or operation).

    Entities form a hierarchy under an Organization and are subject to
    the chosen consolidation approach per ISO 14064-1 Clause 5.1.
    """

    id: str = Field(default_factory=_new_id, description="Unique entity ID")
    name: str = Field(..., min_length=1, max_length=255, description="Entity name")
    entity_type: str = Field(
        default="facility",
        description="subsidiary, facility, or operation",
    )
    parent_id: Optional[str] = Field(None, description="Parent entity ID for hierarchy")
    ownership_pct: Decimal = Field(
        default=Decimal("100.0"),
        ge=Decimal("0"),
        le=Decimal("100"),
        description="Ownership percentage (0-100) for equity share approach",
    )
    country: str = Field(..., min_length=2, max_length=3, description="ISO 3166-1 alpha-2/3")
    region: Optional[str] = Field(None, max_length=100, description="Sub-national region")
    employees: Optional[int] = Field(None, ge=0, description="Full-time equivalents")
    revenue: Optional[Decimal] = Field(None, ge=Decimal("0"), description="Annual revenue (USD)")
    floor_area_m2: Optional[Decimal] = Field(None, ge=Decimal("0"), description="Floor area in m2")
    production_units: Optional[Decimal] = Field(
        None, ge=Decimal("0"), description="Annual production units",
    )
    production_unit_name: Optional[str] = Field(None, description="Name of production unit")
    active: bool = Field(default=True, description="Whether entity is currently active")
    created_at: datetime = Field(default_factory=_now)
    updated_at: datetime = Field(default_factory=_now)


class Organization(GreenLangBase):
    """
    Top-level organization performing ISO 14064-1 GHG accounting.

    Holds the entity hierarchy, sector metadata, and references needed
    for boundary setting per ISO 14064-1 Clause 5.1.
    """

    id: str = Field(default_factory=_new_id, description="Unique organization ID")
    name: str = Field(..., min_length=1, max_length=500, description="Legal entity name")
    industry: str = Field(..., description="Industry sector (e.g. manufacturing, energy)")
    country: str = Field(..., min_length=2, max_length=3, description="HQ country code")
    description: Optional[str] = Field(None, max_length=2000, description="Company description")
    contact_person: Optional[str] = Field(
        None, max_length=255, description="Responsible person name",
    )
    contact_email: Optional[str] = Field(
        None, max_length=255, description="Responsible person email",
    )
    entities: List[Entity] = Field(default_factory=list, description="Owned entities")
    created_at: datetime = Field(default_factory=_now)
    updated_at: datetime = Field(default_factory=_now)


# ---------------------------------------------------------------------------
# Inventory Boundary Models
# ---------------------------------------------------------------------------

class InventoryBoundary(GreenLangBase):
    """
    Organizational and operational boundary for an ISO 14064-1 inventory.

    Implements ISO 14064-1 Clause 5.1 (organizational boundary) and
    Clause 5.2 (operational boundary with six categories).
    """

    id: str = Field(default_factory=_new_id)
    org_id: str = Field(..., description="Organization ID")
    consolidation_approach: ConsolidationApproach = Field(
        ...,
        description="Equity share, financial control, or operational control",
    )
    categories_included: List[ISOCategory] = Field(
        default_factory=lambda: list(ISOCategory),
        description="ISO 14064-1 categories included in operational boundary",
    )
    base_year: Optional[int] = Field(None, ge=1990, le=2100, description="Base year")
    reporting_year: int = Field(..., ge=1990, le=2100, description="Reporting year")
    reporting_period: ReportingPeriod = Field(
        default=ReportingPeriod.CALENDAR_YEAR,
        description="Reporting period type",
    )
    period_start: Optional[date] = Field(None, description="Custom period start date")
    period_end: Optional[date] = Field(None, description="Custom period end date")
    entity_ids: List[str] = Field(default_factory=list, description="Included entity IDs")
    exclusions: List[str] = Field(
        default_factory=list,
        description="Excluded categories with justifications",
    )
    created_at: datetime = Field(default_factory=_now)
    updated_at: datetime = Field(default_factory=_now)


# ---------------------------------------------------------------------------
# Base Year Models
# ---------------------------------------------------------------------------

class Recalculation(GreenLangBase):
    """A base year recalculation event per ISO 14064-1 Clause 5.3."""

    id: str = Field(default_factory=_new_id)
    trigger: str = Field(..., description="Structural or methodological change trigger")
    original_total: Decimal = Field(..., description="Original base year total (tCO2e)")
    new_total: Decimal = Field(..., description="Recalculated total (tCO2e)")
    original_by_category: Dict[str, Decimal] = Field(
        default_factory=dict,
        description="Original emissions by ISO category",
    )
    new_by_category: Dict[str, Decimal] = Field(
        default_factory=dict,
        description="Recalculated emissions by ISO category",
    )
    reason: str = Field(..., min_length=10, description="Detailed justification")
    affected_categories: List[ISOCategory] = Field(default_factory=list)
    recalculated_at: datetime = Field(default_factory=_now)
    approved_by: Optional[str] = Field(None)

    @property
    def change_pct(self) -> Decimal:
        """Percentage change from original to new total."""
        if self.original_total == 0:
            return Decimal("0")
        return ((self.new_total - self.original_total) / self.original_total) * 100


class BaseYear(GreenLangBase):
    """
    Base year definition with emissions snapshot per ISO 14064-1 Clause 5.3.

    The base year serves as the reference point against which performance
    is tracked.  Recalculation is required when structural, methodological,
    or data changes exceed the configured threshold.
    """

    id: str = Field(default_factory=_new_id)
    org_id: str = Field(..., description="Organization ID")
    year: int = Field(..., ge=1990, le=2100, description="Base year")
    category_1_emissions: Decimal = Field(default=Decimal("0"), ge=Decimal("0"))
    category_2_emissions: Decimal = Field(default=Decimal("0"), ge=Decimal("0"))
    category_3_emissions: Decimal = Field(default=Decimal("0"), ge=Decimal("0"))
    category_4_emissions: Decimal = Field(default=Decimal("0"), ge=Decimal("0"))
    category_5_emissions: Decimal = Field(default=Decimal("0"), ge=Decimal("0"))
    category_6_emissions: Decimal = Field(default=Decimal("0"), ge=Decimal("0"))
    total_emissions: Decimal = Field(default=Decimal("0"), ge=Decimal("0"))
    total_removals: Decimal = Field(default=Decimal("0"), ge=Decimal("0"))
    net_emissions: Decimal = Field(default=Decimal("0"), description="Gross minus removals")
    justification: str = Field(
        ...,
        min_length=10,
        description="Reason for selecting this base year",
    )
    locked: bool = Field(default=False, description="Whether the base year is locked")
    recalculations: List[Recalculation] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=_now)
    updated_at: datetime = Field(default_factory=_now)

    @property
    def current_total(self) -> Decimal:
        """Return latest total after any recalculations."""
        if self.recalculations:
            return self.recalculations[-1].new_total
        return self.total_emissions


# ---------------------------------------------------------------------------
# GHG Gas Breakdown
# ---------------------------------------------------------------------------

class GHGGasBreakdown(GreenLangBase):
    """
    Breakdown of emissions by individual GHG gas per ISO 14064-1 Clause 5.2.4.

    Reports each of the seven Kyoto gases plus a total in CO2e.
    """

    co2: Decimal = Field(default=Decimal("0"), ge=Decimal("0"), description="CO2 in tCO2e")
    ch4: Decimal = Field(default=Decimal("0"), ge=Decimal("0"), description="CH4 in tCO2e")
    n2o: Decimal = Field(default=Decimal("0"), ge=Decimal("0"), description="N2O in tCO2e")
    hfcs: Decimal = Field(default=Decimal("0"), ge=Decimal("0"), description="HFCs in tCO2e")
    pfcs: Decimal = Field(default=Decimal("0"), ge=Decimal("0"), description="PFCs in tCO2e")
    sf6: Decimal = Field(default=Decimal("0"), ge=Decimal("0"), description="SF6 in tCO2e")
    nf3: Decimal = Field(default=Decimal("0"), ge=Decimal("0"), description="NF3 in tCO2e")
    total_co2e: Decimal = Field(
        default=Decimal("0"), ge=Decimal("0"), description="Total in tCO2e",
    )

    def model_post_init(self, __context: Any) -> None:
        """Recompute total if not explicitly set."""
        if self.total_co2e == 0:
            computed = (
                self.co2 + self.ch4 + self.n2o + self.hfcs
                + self.pfcs + self.sf6 + self.nf3
            )
            if computed > 0:
                object.__setattr__(self, "total_co2e", computed)


# ---------------------------------------------------------------------------
# Emission Source & Category Models
# ---------------------------------------------------------------------------

class EmissionSource(GreenLangBase):
    """
    An individual emission source within an ISO 14064-1 category.

    Captures activity data, emission factor, quantification method,
    and data quality metadata required for audit trails.
    """

    id: str = Field(default_factory=_new_id)
    name: str = Field(..., min_length=1, max_length=255, description="Source description")
    iso_category: ISOCategory = Field(..., description="ISO 14064-1 category")
    entity_id: Optional[str] = Field(None, description="Reporting entity ID")
    facility_id: Optional[str] = Field(None, description="Facility ID if applicable")
    quantification_method: QuantificationMethod = Field(
        default=QuantificationMethod.CALCULATION_BASED,
        description="Quantification approach",
    )
    activity_data: Decimal = Field(
        default=Decimal("0"), ge=Decimal("0"), description="Activity data value",
    )
    activity_data_unit: str = Field(default="", description="Unit of activity data")
    emission_factor: Decimal = Field(
        default=Decimal("0"), ge=Decimal("0"), description="Emission factor value",
    )
    emission_factor_unit: str = Field(default="", description="Unit of emission factor")
    emission_factor_source: str = Field(
        default="", description="Reference source for emission factor",
    )
    gwp_source: GWPSource = Field(default=GWPSource.AR5, description="GWP values used")
    gas_breakdown: GHGGasBreakdown = Field(
        default_factory=GHGGasBreakdown,
        description="Emissions broken down by GHG gas",
    )
    total_tco2e: Decimal = Field(
        default=Decimal("0"), ge=Decimal("0"), description="Total emissions in tCO2e",
    )
    data_quality_tier: DataQualityTier = Field(
        default=DataQualityTier.TIER_1,
        description="Data quality tier for this source",
    )
    uncertainty_pct: Optional[Decimal] = Field(
        None, ge=Decimal("0"), description="Source-level uncertainty as +/- percentage",
    )
    notes: Optional[str] = Field(None, max_length=2000, description="Methodology notes")
    mrv_agent: Optional[str] = Field(None, description="MRV agent that computed this source")
    provenance_hash: str = Field(default="", description="SHA-256 hash for audit trail")
    created_at: datetime = Field(default_factory=_now)
    updated_at: datetime = Field(default_factory=_now)

    def model_post_init(self, __context: Any) -> None:
        """Compute provenance hash after initialization."""
        if not self.provenance_hash:
            payload = (
                f"{self.iso_category}:{self.name}:{self.activity_data}:"
                f"{self.emission_factor}:{self.total_tco2e}"
            )
            self.provenance_hash = _sha256(payload)


class FacilityEmissions(GreenLangBase):
    """Aggregated emissions for a single facility."""

    facility_id: str = Field(..., description="Facility entity ID")
    facility_name: str = Field(default="", description="Facility display name")
    by_category: Dict[str, Decimal] = Field(
        default_factory=dict, description="tCO2e by ISO category",
    )
    by_gas: GHGGasBreakdown = Field(default_factory=GHGGasBreakdown)
    total_tco2e: Decimal = Field(default=Decimal("0"), ge=Decimal("0"))
    source_count: int = Field(default=0, ge=0, description="Number of emission sources")


class EntityEmissions(GreenLangBase):
    """Aggregated emissions for an organizational entity."""

    entity_id: str = Field(..., description="Entity ID")
    entity_name: str = Field(default="", description="Entity display name")
    ownership_pct: Decimal = Field(default=Decimal("100.0"))
    by_category: Dict[str, Decimal] = Field(
        default_factory=dict, description="tCO2e by ISO category",
    )
    by_gas: GHGGasBreakdown = Field(default_factory=GHGGasBreakdown)
    facilities: List[FacilityEmissions] = Field(default_factory=list)
    total_tco2e: Decimal = Field(default=Decimal("0"), ge=Decimal("0"))
    attributed_tco2e: Decimal = Field(
        default=Decimal("0"),
        ge=Decimal("0"),
        description="After applying ownership percentage (equity share)",
    )


class CategoryResult(GreenLangBase):
    """
    Aggregated result for a single ISO 14064-1 category.

    Combines all emission sources within the category and provides
    gas-level breakdowns, data quality, and provenance.
    """

    iso_category: ISOCategory = Field(..., description="ISO 14064-1 category")
    category_name: str = Field(default="", description="Human-readable category name")
    sources: List[EmissionSource] = Field(default_factory=list)
    gas_breakdown: GHGGasBreakdown = Field(default_factory=GHGGasBreakdown)
    total_tco2e: Decimal = Field(default=Decimal("0"), ge=Decimal("0"))
    biogenic_co2: Decimal = Field(
        default=Decimal("0"),
        ge=Decimal("0"),
        description="Biogenic CO2 reported separately per ISO 14064-1",
    )
    data_quality_tier: DataQualityTier = Field(
        default=DataQualityTier.TIER_1,
        description="Predominant data quality tier for this category",
    )
    quantification_methods_used: List[QuantificationMethod] = Field(
        default_factory=list,
        description="Quantification methods applied in this category",
    )
    significance: SignificanceLevel = Field(
        default=SignificanceLevel.UNDER_REVIEW,
        description="Significance assessment result",
    )
    by_entity: Dict[str, Decimal] = Field(
        default_factory=dict, description="Emissions by entity ID",
    )
    provenance_hash: str = Field(default="", description="SHA-256 hash")

    def model_post_init(self, __context: Any) -> None:
        """Compute provenance hash."""
        if not self.provenance_hash:
            payload = (
                f"{self.iso_category}:{self.total_tco2e}:"
                f"{self.gas_breakdown.total_co2e}"
            )
            self.provenance_hash = _sha256(payload)


# ---------------------------------------------------------------------------
# Removals Models
# ---------------------------------------------------------------------------

class RemovalSource(GreenLangBase):
    """
    A GHG removal source per ISO 14064-1:2018 Clause 5.2.3.

    Captures the removal type, quantity, permanence level, and
    verification status for audit trail purposes.
    """

    id: str = Field(default_factory=_new_id)
    name: str = Field(..., min_length=1, max_length=255, description="Removal project name")
    removal_type: RemovalType = Field(..., description="Type of removal activity")
    quantification_method: QuantificationMethod = Field(
        default=QuantificationMethod.CALCULATION_BASED,
    )
    quantity_tco2e: Decimal = Field(
        default=Decimal("0"),
        ge=Decimal("0"),
        description="Quantity of CO2e removed in metric tonnes",
    )
    permanence: PermanenceLevel = Field(
        default=PermanenceLevel.LONG_TERM,
        description="Expected permanence of the removal",
    )
    permanence_years: Optional[int] = Field(
        None, ge=0, description="Estimated permanence duration in years",
    )
    reversal_risk_pct: Decimal = Field(
        default=Decimal("0"),
        ge=Decimal("0"),
        le=Decimal("100"),
        description="Estimated reversal risk percentage",
    )
    verification_status: VerificationStage = Field(
        default=VerificationStage.DRAFT,
        description="Current verification stage",
    )
    verification_body: Optional[str] = Field(None, description="Name of verification body")
    project_id: Optional[str] = Field(None, description="External project/registry ID")
    registry: Optional[str] = Field(
        None, description="Carbon registry (e.g. Verra, Gold Standard)",
    )
    vintage_year: Optional[int] = Field(
        None, ge=1990, le=2100, description="Credit vintage year",
    )
    entity_id: Optional[str] = Field(None, description="Reporting entity ID")
    notes: Optional[str] = Field(None, max_length=2000)
    provenance_hash: str = Field(default="")
    created_at: datetime = Field(default_factory=_now)
    updated_at: datetime = Field(default_factory=_now)

    def model_post_init(self, __context: Any) -> None:
        """Compute provenance hash."""
        if not self.provenance_hash:
            payload = (
                f"{self.removal_type}:{self.quantity_tco2e}:"
                f"{self.permanence}:{self.verification_status}"
            )
            self.provenance_hash = _sha256(payload)


class BiogenicEmissions(GreenLangBase):
    """
    Separate tracking of biogenic CO2 per ISO 14064-1 reporting requirements.

    Biogenic CO2 emissions must be quantified and reported separately from
    fossil-origin emissions.
    """

    total_biogenic_co2: Decimal = Field(
        default=Decimal("0"),
        ge=Decimal("0"),
        description="Total biogenic CO2 in tCO2",
    )
    by_category: Dict[str, Decimal] = Field(
        default_factory=dict, description="Biogenic CO2 breakdown by ISO category",
    )
    by_source: Dict[str, Decimal] = Field(
        default_factory=dict, description="Biogenic CO2 breakdown by source ID",
    )
    notes: Optional[str] = Field(
        None, description="Methodology notes for biogenic accounting",
    )


class NetEmissionsResult(GreenLangBase):
    """
    Net emissions calculation per ISO 14064-1:2018.

    Computes net = gross_emissions - total_removals and provides a
    complete breakdown for reporting.
    """

    gross_emissions: Decimal = Field(
        default=Decimal("0"),
        ge=Decimal("0"),
        description="Total gross GHG emissions (tCO2e)",
    )
    total_removals: Decimal = Field(
        default=Decimal("0"),
        ge=Decimal("0"),
        description="Total GHG removals (tCO2e)",
    )
    net_emissions: Decimal = Field(
        default=Decimal("0"),
        description="Net = gross - removals (can be negative)",
    )
    gross_by_category: Dict[str, Decimal] = Field(
        default_factory=dict, description="Gross emissions by ISO category",
    )
    removals_by_type: Dict[str, Decimal] = Field(
        default_factory=dict, description="Removals by removal type",
    )
    biogenic: BiogenicEmissions = Field(
        default_factory=BiogenicEmissions,
        description="Biogenic CO2 reported separately",
    )
    provenance_hash: str = Field(default="")

    def model_post_init(self, __context: Any) -> None:
        """Compute net and provenance hash."""
        if self.net_emissions == 0 and (
            self.gross_emissions > 0 or self.total_removals > 0
        ):
            object.__setattr__(
                self,
                "net_emissions",
                self.gross_emissions - self.total_removals,
            )
        if not self.provenance_hash:
            payload = (
                f"{self.gross_emissions}:{self.total_removals}:{self.net_emissions}"
            )
            self.provenance_hash = _sha256(payload)


# ---------------------------------------------------------------------------
# Significance Models
# ---------------------------------------------------------------------------

class SignificanceCriteria(GreenLangBase):
    """
    Individual criteria scores for significance assessment.

    ISO 14064-1:2018 Clause 5.2.2 requires organizations to assess the
    significance of indirect emission categories using defined criteria.
    """

    magnitude: Decimal = Field(
        default=Decimal("0"),
        ge=Decimal("0"),
        le=Decimal("100"),
        description="Estimated magnitude as % of total emissions",
    )
    influence: Decimal = Field(
        default=Decimal("0"),
        ge=Decimal("0"),
        le=Decimal("100"),
        description="Level of influence or control the organization has",
    )
    risk: Decimal = Field(
        default=Decimal("0"),
        ge=Decimal("0"),
        le=Decimal("100"),
        description="Risk exposure related to this category",
    )
    stakeholder: Decimal = Field(
        default=Decimal("0"),
        ge=Decimal("0"),
        le=Decimal("100"),
        description="Stakeholder concern level",
    )
    data_availability: Decimal = Field(
        default=Decimal("0"),
        ge=Decimal("0"),
        le=Decimal("100"),
        description="Availability and quality of data",
    )

    @property
    def composite_score(self) -> Decimal:
        """Weighted composite score (equal weights)."""
        total = (
            self.magnitude + self.influence + self.risk
            + self.stakeholder + self.data_availability
        )
        return total / 5


class SignificanceAssessment(GreenLangBase):
    """
    Significance assessment for an indirect ISO 14064-1 category (3-6).

    Per Clause 5.2.2, organizations must assess each indirect category and
    justify inclusion or exclusion based on defined criteria.
    """

    id: str = Field(default_factory=_new_id)
    iso_category: ISOCategory = Field(..., description="Category being assessed")
    criteria: SignificanceCriteria = Field(
        default_factory=SignificanceCriteria,
        description="Individual criteria scores",
    )
    threshold_pct: Decimal = Field(
        default=Decimal("1.0"),
        ge=Decimal("0"),
        description="Significance threshold used (%)",
    )
    result: SignificanceLevel = Field(
        default=SignificanceLevel.UNDER_REVIEW,
        description="Assessment outcome",
    )
    justification: str = Field(
        default="",
        description="Written justification for the assessment result",
    )
    estimated_emissions_tco2e: Optional[Decimal] = Field(
        None, ge=Decimal("0"), description="Estimated emissions for the category",
    )
    assessed_by: Optional[str] = Field(
        None, description="Person who performed assessment",
    )
    assessed_at: datetime = Field(default_factory=_now)
    review_notes: Optional[str] = Field(None, max_length=2000)


# ---------------------------------------------------------------------------
# Uncertainty Models
# ---------------------------------------------------------------------------

class UncertaintyResult(GreenLangBase):
    """
    Uncertainty analysis results per ISO 14064-1:2018 Clause 6.3.

    Supports Monte Carlo simulation output with confidence intervals
    at multiple levels.
    """

    id: str = Field(default_factory=_new_id)
    inventory_id: str = Field(default="")
    scope: str = Field(
        default="total",
        description="Scope of analysis: total, category_1, etc.",
    )
    lower_bound: Decimal = Field(
        default=Decimal("0"), description="Lower confidence bound (tCO2e)",
    )
    central_estimate: Decimal = Field(
        default=Decimal("0"),
        ge=Decimal("0"),
        description="Central estimate / mean (tCO2e)",
    )
    upper_bound: Decimal = Field(
        default=Decimal("0"), description="Upper confidence bound (tCO2e)",
    )
    confidence_level: int = Field(
        default=95, ge=1, le=99, description="Confidence level percentage",
    )
    std_dev: Decimal = Field(default=Decimal("0"), ge=Decimal("0"))
    cv_pct: Decimal = Field(
        default=Decimal("0"),
        ge=Decimal("0"),
        description="Coefficient of variation (%)",
    )
    methodology: str = Field(
        default="monte_carlo",
        description="monte_carlo, error_propagation, or expert_judgment",
    )
    iterations: int = Field(default=10_000, ge=0)
    by_category: Dict[str, Dict[str, Decimal]] = Field(
        default_factory=dict, description="Uncertainty bounds per ISO category",
    )
    sensitivity_ranking: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Sources ranked by contribution to overall uncertainty",
    )
    computed_at: datetime = Field(default_factory=_now)


# ---------------------------------------------------------------------------
# Quality Models
# ---------------------------------------------------------------------------

class DataQualityScore(GreenLangBase):
    """
    Composite data quality score per ISO 14064-1:2018 Clause 6.3.

    Evaluates quality across activity data, emission factors, and
    overall composite dimensions.
    """

    activity_data_score: Decimal = Field(
        default=Decimal("0"),
        ge=Decimal("0"),
        le=Decimal("100"),
        description="Activity data quality score (0-100)",
    )
    emission_factor_score: Decimal = Field(
        default=Decimal("0"),
        ge=Decimal("0"),
        le=Decimal("100"),
        description="Emission factor quality score (0-100)",
    )
    composite_score: Decimal = Field(
        default=Decimal("0"),
        ge=Decimal("0"),
        le=Decimal("100"),
        description="Weighted composite quality score (0-100)",
    )
    tier: DataQualityTier = Field(
        default=DataQualityTier.TIER_1, description="Overall data quality tier",
    )
    completeness_pct: Decimal = Field(
        default=Decimal("0"),
        ge=Decimal("0"),
        le=Decimal("100"),
        description="Data completeness percentage",
    )
    by_category: Dict[str, Decimal] = Field(
        default_factory=dict, description="Quality score per ISO category",
    )
    assessed_at: datetime = Field(default_factory=_now)


class CorrectiveAction(GreenLangBase):
    """
    A corrective action arising from a verification finding or quality issue.

    Per ISO 14064-1:2018 Clause 9 and ISO 14064-3:2019 guidance.
    """

    id: str = Field(default_factory=_new_id)
    finding_id: Optional[str] = Field(None, description="Related finding ID")
    description: str = Field(..., min_length=5, description="Action description")
    action_type: ActionCategory = Field(
        default=ActionCategory.DATA_IMPROVEMENT, description="Category of action",
    )
    assigned_to: Optional[str] = Field(None, description="Person responsible")
    deadline: Optional[date] = Field(None, description="Target completion date")
    status: ActionStatus = Field(default=ActionStatus.PLANNED)
    resolution_notes: Optional[str] = Field(None, max_length=2000)
    created_at: datetime = Field(default_factory=_now)
    completed_at: Optional[datetime] = Field(None)


class QualityManagementPlan(GreenLangBase):
    """
    Quality management plan per ISO 14064-1:2018 Clause 6.

    Documents procedures, audit schedules, and responsibilities for
    maintaining GHG inventory quality.
    """

    id: str = Field(default_factory=_new_id)
    org_id: str = Field(..., description="Organization ID")
    procedures: List[str] = Field(
        default_factory=list, description="Documented quality procedures",
    )
    audit_schedule: List[Dict[str, Any]] = Field(
        default_factory=list, description="Internal audit schedule entries",
    )
    responsibilities: Dict[str, str] = Field(
        default_factory=dict, description="Role -> responsibility mapping",
    )
    data_quality_objectives: Dict[str, str] = Field(
        default_factory=dict, description="Quality objective per ISO category",
    )
    review_frequency: str = Field(
        default="annual", description="Frequency of management review",
    )
    corrective_actions: List[CorrectiveAction] = Field(default_factory=list)
    last_review_date: Optional[date] = Field(None)
    next_review_date: Optional[date] = Field(None)
    created_at: datetime = Field(default_factory=_now)
    updated_at: datetime = Field(default_factory=_now)


# ---------------------------------------------------------------------------
# Management Plan Models
# ---------------------------------------------------------------------------

class ImprovementAction(GreenLangBase):
    """
    A planned improvement action per ISO 14064-1:2018 Clause 9.

    Tracks emission reduction projects, removal enhancements,
    and data quality improvements.
    """

    id: str = Field(default_factory=_new_id)
    name: str = Field(..., min_length=1, max_length=255, description="Action name")
    category: ActionCategory = Field(..., description="Action category")
    iso_category: Optional[ISOCategory] = Field(
        None, description="Target ISO 14064-1 category (if applicable)",
    )
    description: str = Field(default="", max_length=2000)
    target_reduction_tco2e: Optional[Decimal] = Field(
        None, ge=Decimal("0"), description="Target emission reduction (tCO2e)",
    )
    target_year: Optional[int] = Field(None, ge=1990, le=2100)
    timeline_start: Optional[date] = Field(None, description="Planned start date")
    timeline_end: Optional[date] = Field(None, description="Planned end date")
    estimated_cost_usd: Optional[Decimal] = Field(
        None, ge=Decimal("0"), description="Estimated cost in USD",
    )
    progress_pct: Decimal = Field(
        default=Decimal("0"),
        ge=Decimal("0"),
        le=Decimal("100"),
        description="Progress percentage (0-100)",
    )
    status: ActionStatus = Field(default=ActionStatus.PLANNED)
    assigned_to: Optional[str] = Field(None, description="Person responsible")
    notes: Optional[str] = Field(None, max_length=2000)
    created_at: datetime = Field(default_factory=_now)
    updated_at: datetime = Field(default_factory=_now)


class ManagementPlan(GreenLangBase):
    """
    GHG management plan per ISO 14064-1:2018 Clause 9.

    Documents the organization's planned actions to manage, reduce,
    and remove GHG emissions.
    """

    id: str = Field(default_factory=_new_id)
    org_id: str = Field(..., description="Organization ID")
    reporting_year: int = Field(..., ge=1990, le=2100)
    actions: List[ImprovementAction] = Field(default_factory=list)
    total_planned_reduction_tco2e: Decimal = Field(
        default=Decimal("0"), ge=Decimal("0"),
    )
    total_estimated_cost_usd: Decimal = Field(
        default=Decimal("0"), ge=Decimal("0"),
    )
    created_at: datetime = Field(default_factory=_now)
    updated_at: datetime = Field(default_factory=_now)

    def recalculate_totals(self) -> None:
        """Recalculate totals from individual actions."""
        self.total_planned_reduction_tco2e = sum(
            (a.target_reduction_tco2e or Decimal("0")) for a in self.actions
        )
        self.total_estimated_cost_usd = sum(
            (a.estimated_cost_usd or Decimal("0")) for a in self.actions
        )


# ---------------------------------------------------------------------------
# Verification Models
# ---------------------------------------------------------------------------

class Finding(GreenLangBase):
    """
    A verification finding per ISO 14064-3:2019.

    Captures observations, non-conformities, and improvement opportunities
    identified during internal review or external verification.
    """

    id: str = Field(default_factory=_new_id)
    description: str = Field(..., min_length=10, description="Finding details")
    iso_category: Optional[ISOCategory] = Field(None, description="Affected category")
    clause_reference: Optional[str] = Field(
        None, description="ISO 14064-1 clause reference (e.g. 5.2.4)",
    )
    severity: FindingSeverity = Field(default=FindingSeverity.LOW)
    status: FindingStatus = Field(default=FindingStatus.OPEN)
    resolution: Optional[str] = Field(None, description="Resolution description")
    corrective_action_id: Optional[str] = Field(
        None, description="Linked corrective action",
    )
    created_at: datetime = Field(default_factory=_now)
    resolved_at: Optional[datetime] = Field(None)


class FindingsSummary(GreenLangBase):
    """Summary of all findings from a verification engagement."""

    total_findings: int = Field(default=0, ge=0)
    open_count: int = Field(default=0, ge=0)
    in_progress_count: int = Field(default=0, ge=0)
    resolved_count: int = Field(default=0, ge=0)
    by_severity: Dict[str, int] = Field(
        default_factory=dict, description="Count by severity level",
    )
    by_category: Dict[str, int] = Field(
        default_factory=dict, description="Count by ISO category",
    )
    critical_open: bool = Field(
        default=False, description="Whether any critical findings remain open",
    )


class VerificationRecord(GreenLangBase):
    """
    A verification / assurance record per ISO 14064-3:2019.

    Tracks the full lifecycle from draft through external verification,
    including all findings and the final verification statement.
    """

    id: str = Field(default_factory=_new_id)
    inventory_id: str = Field(..., description="Inventory being verified")
    level: VerificationLevel = Field(
        default=VerificationLevel.NOT_VERIFIED,
        description="Assurance level (limited or reasonable)",
    )
    stage: VerificationStage = Field(
        default=VerificationStage.DRAFT,
        description="Current verification workflow stage",
    )
    verifier_id: Optional[str] = Field(None, description="Assigned verifier")
    verifier_name: Optional[str] = Field(None)
    verifier_organization: Optional[str] = Field(None)
    verifier_accreditation: Optional[str] = Field(
        None, description="Accreditation body / number",
    )
    findings: List[Finding] = Field(default_factory=list)
    findings_summary: Optional[FindingsSummary] = Field(None)
    statement: Optional[str] = Field(
        None, description="Verification statement text",
    )
    opinion: Optional[str] = Field(
        None, description="Qualified / unqualified / adverse / disclaimer",
    )
    scope_of_verification: Optional[str] = Field(
        None, description="What was verified (categories, period, etc.)",
    )
    materiality_threshold_pct: Decimal = Field(
        default=Decimal("5.0"), ge=Decimal("0"), le=Decimal("100"),
    )
    started_at: datetime = Field(default_factory=_now)
    submitted_at: Optional[datetime] = Field(None)
    completed_at: Optional[datetime] = Field(None)

    @property
    def open_findings_count(self) -> int:
        """Count of unresolved findings."""
        return sum(
            1 for f in self.findings
            if f.status in (FindingStatus.OPEN, FindingStatus.IN_PROGRESS)
        )

    @property
    def has_critical_findings(self) -> bool:
        """Whether any critical findings remain open."""
        return any(
            f.severity == FindingSeverity.CRITICAL
            and f.status in (FindingStatus.OPEN, FindingStatus.IN_PROGRESS)
            for f in self.findings
        )


# ---------------------------------------------------------------------------
# Cross-Walk Models
# ---------------------------------------------------------------------------

class CrossWalkMapping(GreenLangBase):
    """
    Maps an ISO 14064-1 category to a GHG Protocol scope.

    Enables side-by-side comparison between ISO 14064-1 and GHG Protocol
    reporting frameworks.
    """

    iso_category: ISOCategory = Field(..., description="ISO 14064-1 category")
    ghg_protocol_scope: str = Field(
        ...,
        description="GHG Protocol scope (scope_1, scope_2_location, scope_2_market, scope_3)",
    )
    ghg_protocol_categories: List[str] = Field(
        default_factory=list,
        description="Specific GHG Protocol sub-categories mapped",
    )
    detailed_mapping: Dict[str, str] = Field(
        default_factory=dict,
        description="Fine-grained source-level mapping details",
    )
    notes: Optional[str] = Field(None, description="Mapping notes or caveats")


class CrossWalkResult(GreenLangBase):
    """
    Side-by-side comparison of ISO 14064-1 vs GHG Protocol reporting.

    Useful for organizations that must report under both frameworks.
    """

    id: str = Field(default_factory=_new_id)
    org_id: str = Field(..., description="Organization ID")
    reporting_year: int = Field(..., ge=1990, le=2100)
    mappings: List[CrossWalkMapping] = Field(default_factory=list)
    iso_totals: Dict[str, Decimal] = Field(
        default_factory=dict, description="Totals by ISO category",
    )
    ghg_protocol_totals: Dict[str, Decimal] = Field(
        default_factory=dict, description="Totals by GHG Protocol scope",
    )
    reconciliation_difference: Decimal = Field(
        default=Decimal("0"),
        description="Difference between ISO and GHG Protocol totals",
    )
    reconciliation_notes: Optional[str] = Field(
        None, description="Explanation of any differences",
    )
    generated_at: datetime = Field(default_factory=_now)


# ---------------------------------------------------------------------------
# Report Models
# ---------------------------------------------------------------------------

class ReportSection(GreenLangBase):
    """A single section of a generated ISO 14064-1 report."""

    key: str = Field(..., description="Section key (e.g. organization_description)")
    title: str = Field(..., description="Display title")
    content: Dict[str, Any] = Field(default_factory=dict)
    order: int = Field(default=0)


class MandatoryElement(GreenLangBase):
    """
    One of the 14 mandatory reporting elements per ISO 14064-1:2018 Clause 9.

    Tracks whether each required element is satisfied in the report.
    """

    element_id: str = Field(
        ..., description="Mandatory element ID (MRE-01 through MRE-14)",
    )
    name: str = Field(..., description="Element description")
    clause_reference: str = Field(
        default="", description="ISO 14064-1 clause reference",
    )
    required: bool = Field(default=True)
    present: bool = Field(default=False, description="Whether satisfied in report")
    evidence: Optional[str] = Field(None, description="Evidence or section reference")
    notes: Optional[str] = Field(None, description="Additional notes")


class Disclosure(GreenLangBase):
    """A disclosure element within the ISO 14064-1 report."""

    id: str = Field(default_factory=_new_id)
    name: str = Field(..., description="Disclosure name")
    category: str = Field(..., description="Disclosure category")
    required: bool = Field(default=True, description="Whether mandatory")
    present: bool = Field(default=False, description="Whether satisfied")
    evidence: Optional[str] = Field(None, description="Evidence reference")


class Report(GreenLangBase):
    """A generated ISO 14064-1 compliance report."""

    id: str = Field(default_factory=_new_id)
    inventory_id: str = Field(..., description="Source inventory ID")
    org_id: str = Field(..., description="Organization ID")
    reporting_year: int = Field(..., ge=1990, le=2100)
    format: ReportFormat = Field(default=ReportFormat.JSON)
    status: ReportStatus = Field(default=ReportStatus.DRAFT)
    sections: List[ReportSection] = Field(default_factory=list)
    mandatory_elements: List[MandatoryElement] = Field(default_factory=list)
    disclosures: List[Disclosure] = Field(default_factory=list)
    mandatory_compliance_pct: Decimal = Field(
        default=Decimal("0"),
        ge=Decimal("0"),
        le=Decimal("100"),
        description="Percentage of mandatory elements satisfied",
    )
    generated_at: datetime = Field(default_factory=_now)
    file_path: Optional[str] = Field(None, description="Path to generated file")
    file_size_bytes: Optional[int] = Field(None, ge=0)
    provenance_hash: str = Field(default="")

    def model_post_init(self, __context: Any) -> None:
        """Compute provenance hash and mandatory compliance percentage."""
        if self.mandatory_elements:
            present = sum(1 for e in self.mandatory_elements if e.present)
            total = len(self.mandatory_elements)
            if total > 0:
                object.__setattr__(
                    self,
                    "mandatory_compliance_pct",
                    Decimal(str(round(present / total * 100, 1))),
                )
        if not self.provenance_hash:
            payload = f"{self.inventory_id}:{self.format}:{self.generated_at}"
            self.provenance_hash = _sha256(payload)


# ---------------------------------------------------------------------------
# ISO Inventory -- The Central Object
# ---------------------------------------------------------------------------

class ISOInventory(GreenLangBase):
    """
    Complete ISO 14064-1:2018 GHG inventory for an organization-year.

    This is the central data structure aggregating emissions across all six
    ISO categories, removals, net emissions, and all quality and verification
    metadata.
    """

    id: str = Field(default_factory=_new_id)
    org_id: str = Field(..., description="Organization ID")
    year: int = Field(..., ge=1990, le=2100, description="Reporting year")
    boundary: Optional[InventoryBoundary] = Field(None)
    gwp_source: GWPSource = Field(default=GWPSource.AR5, description="GWP values used")

    # -- Category Results ---------------------------------------------------
    category_1: Optional[CategoryResult] = Field(
        None, description="Category 1 - Direct GHG emissions and removals",
    )
    category_2: Optional[CategoryResult] = Field(
        None, description="Category 2 - Indirect from imported energy",
    )
    category_3: Optional[CategoryResult] = Field(
        None, description="Category 3 - Indirect from transportation",
    )
    category_4: Optional[CategoryResult] = Field(
        None, description="Category 4 - Indirect from products used",
    )
    category_5: Optional[CategoryResult] = Field(
        None, description="Category 5 - Indirect from products of the organization",
    )
    category_6: Optional[CategoryResult] = Field(
        None, description="Category 6 - Indirect from other sources",
    )

    # -- Aggregates ---------------------------------------------------------
    gross_emissions_tco2e: Decimal = Field(
        default=Decimal("0"),
        ge=Decimal("0"),
        description="Sum of all six categories",
    )

    # -- Removals -----------------------------------------------------------
    removal_sources: List[RemovalSource] = Field(default_factory=list)
    total_removals_tco2e: Decimal = Field(
        default=Decimal("0"), ge=Decimal("0"),
    )
    net_emissions_tco2e: Decimal = Field(
        default=Decimal("0"),
        description="Gross minus removals (may be negative)",
    )

    # -- Biogenic -----------------------------------------------------------
    biogenic: BiogenicEmissions = Field(default_factory=BiogenicEmissions)

    # -- Significance -------------------------------------------------------
    significance_assessments: List[SignificanceAssessment] = Field(
        default_factory=list,
    )

    # -- Quality ------------------------------------------------------------
    data_quality: Optional[DataQualityScore] = Field(None)
    uncertainty: Optional[UncertaintyResult] = Field(None)

    # -- Verification -------------------------------------------------------
    verification: Optional[VerificationRecord] = Field(None)

    # -- Status -------------------------------------------------------------
    status: InventoryStatus = Field(default=InventoryStatus.DRAFT)
    created_at: datetime = Field(default_factory=_now)
    updated_at: datetime = Field(default_factory=_now)
    provenance_hash: str = Field(default="")

    def recalculate_totals(self) -> None:
        """Recalculate gross, removals, and net totals from category data."""
        categories = [
            self.category_1, self.category_2, self.category_3,
            self.category_4, self.category_5, self.category_6,
        ]
        self.gross_emissions_tco2e = sum(
            (c.total_tco2e if c else Decimal("0")) for c in categories
        )
        self.total_removals_tco2e = sum(
            r.quantity_tco2e for r in self.removal_sources
        )
        self.net_emissions_tco2e = (
            self.gross_emissions_tco2e - self.total_removals_tco2e
        )
        self.updated_at = _now()

    def model_post_init(self, __context: Any) -> None:
        """Compute totals and provenance hash."""
        self.recalculate_totals()
        if not self.provenance_hash:
            payload = (
                f"{self.org_id}:{self.year}:{self.gross_emissions_tco2e}:"
                f"{self.net_emissions_tco2e}"
            )
            self.provenance_hash = _sha256(payload)


# ---------------------------------------------------------------------------
# Dashboard Models
# ---------------------------------------------------------------------------

class TrendDataPoint(GreenLangBase):
    """A single data point in a time-series trend."""

    year: int = Field(..., ge=1990, le=2100)
    value: Decimal = Field(default=Decimal("0"))
    label: Optional[str] = Field(
        None, description="Optional label (e.g. category name)",
    )


class CategoryBreakdown(GreenLangBase):
    """Breakdown of emissions for a single ISO category (dashboard)."""

    iso_category: ISOCategory = Field(...)
    category_name: str = Field(default="")
    total_tco2e: Decimal = Field(default=Decimal("0"), ge=Decimal("0"))
    pct_of_total: Decimal = Field(
        default=Decimal("0"), ge=Decimal("0"), le=Decimal("100"),
    )
    significance: SignificanceLevel = Field(
        default=SignificanceLevel.UNDER_REVIEW,
    )
    yoy_change_pct: Optional[Decimal] = Field(None)


class DashboardAlert(GreenLangBase):
    """An alert surfaced on the dashboard."""

    id: str = Field(default_factory=_new_id)
    severity: FindingSeverity = Field(default=FindingSeverity.LOW)
    title: str = Field(..., min_length=1, description="Alert title")
    message: str = Field(default="", description="Alert details")
    iso_category: Optional[ISOCategory] = Field(None)
    created_at: datetime = Field(default_factory=_now)
    dismissed: bool = Field(default=False)


class DashboardMetrics(GreenLangBase):
    """
    Aggregated dashboard metrics for a single ISO 14064-1 inventory year.

    Provides high-level KPIs, category breakdowns, gas breakdowns,
    year-over-year trends, and quality indicators.
    """

    org_id: str = Field(...)
    year: int = Field(...)
    total_gross_emissions: Decimal = Field(
        default=Decimal("0"), ge=Decimal("0"),
    )
    total_removals: Decimal = Field(default=Decimal("0"), ge=Decimal("0"))
    net_emissions: Decimal = Field(default=Decimal("0"))
    by_category: List[CategoryBreakdown] = Field(default_factory=list)
    by_gas: GHGGasBreakdown = Field(default_factory=GHGGasBreakdown)
    yoy_change_pct: Optional[Decimal] = Field(
        None, description="Year-over-year change in net emissions (%)",
    )
    quality_score: Decimal = Field(
        default=Decimal("0"), ge=Decimal("0"), le=Decimal("100"),
    )
    completeness_pct: Decimal = Field(
        default=Decimal("0"), ge=Decimal("0"), le=Decimal("100"),
    )
    verification_stage: VerificationStage = Field(
        default=VerificationStage.DRAFT,
    )
    trend: List[TrendDataPoint] = Field(default_factory=list)
    alerts: List[DashboardAlert] = Field(default_factory=list)
    biogenic_co2: Decimal = Field(default=Decimal("0"), ge=Decimal("0"))
    computed_at: datetime = Field(default_factory=_now)


# ---------------------------------------------------------------------------
# Request / Response Models
# ---------------------------------------------------------------------------

class CreateOrganizationRequest(GreenLangBase):
    """Request to create a new organization."""

    name: str = Field(..., min_length=1, max_length=500)
    industry: str = Field(..., min_length=1)
    country: str = Field(..., min_length=2, max_length=3)
    description: Optional[str] = Field(None, max_length=2000)
    contact_person: Optional[str] = Field(None, max_length=255)
    contact_email: Optional[str] = Field(None, max_length=255)


class AddEntityRequest(GreenLangBase):
    """Request to add an entity to an organization."""

    name: str = Field(..., min_length=1, max_length=255)
    entity_type: str = Field(default="facility")
    parent_id: Optional[str] = Field(None)
    ownership_pct: Decimal = Field(
        default=Decimal("100.0"), ge=Decimal("0"), le=Decimal("100"),
    )
    country: str = Field(..., min_length=2, max_length=3)
    region: Optional[str] = Field(None, max_length=100)
    employees: Optional[int] = Field(None, ge=0)
    revenue: Optional[Decimal] = Field(None, ge=Decimal("0"))
    floor_area_m2: Optional[Decimal] = Field(None, ge=Decimal("0"))
    production_units: Optional[Decimal] = Field(None, ge=Decimal("0"))
    production_unit_name: Optional[str] = Field(None)


class SetBoundaryRequest(GreenLangBase):
    """Request to set organizational/operational boundary."""

    consolidation_approach: ConsolidationApproach = Field(...)
    categories_included: List[ISOCategory] = Field(
        default_factory=lambda: list(ISOCategory),
    )
    reporting_year: int = Field(..., ge=1990, le=2100)
    reporting_period: ReportingPeriod = Field(
        default=ReportingPeriod.CALENDAR_YEAR,
    )
    period_start: Optional[date] = Field(None)
    period_end: Optional[date] = Field(None)
    entity_ids: Optional[List[str]] = Field(None)


class CreateInventoryRequest(GreenLangBase):
    """Request to create a new ISO 14064-1 GHG inventory."""

    year: int = Field(..., ge=1990, le=2100)
    consolidation_approach: Optional[ConsolidationApproach] = Field(None)
    categories: Optional[List[ISOCategory]] = Field(None)
    gwp_source: GWPSource = Field(default=GWPSource.AR5)


class AddEmissionSourceRequest(GreenLangBase):
    """Request to add an emission source to an inventory category."""

    name: str = Field(..., min_length=1, max_length=255)
    iso_category: ISOCategory = Field(...)
    entity_id: Optional[str] = Field(None)
    facility_id: Optional[str] = Field(None)
    quantification_method: QuantificationMethod = Field(
        default=QuantificationMethod.CALCULATION_BASED,
    )
    activity_data: Decimal = Field(default=Decimal("0"), ge=Decimal("0"))
    activity_data_unit: str = Field(default="")
    emission_factor: Decimal = Field(default=Decimal("0"), ge=Decimal("0"))
    emission_factor_unit: str = Field(default="")
    emission_factor_source: str = Field(default="")
    gwp_source: GWPSource = Field(default=GWPSource.AR5)
    total_tco2e: Decimal = Field(default=Decimal("0"), ge=Decimal("0"))
    data_quality_tier: DataQualityTier = Field(default=DataQualityTier.TIER_1)
    notes: Optional[str] = Field(None, max_length=2000)


class AddRemovalRequest(GreenLangBase):
    """Request to add a GHG removal source."""

    name: str = Field(..., min_length=1, max_length=255)
    removal_type: RemovalType = Field(...)
    quantification_method: QuantificationMethod = Field(
        default=QuantificationMethod.CALCULATION_BASED,
    )
    quantity_tco2e: Decimal = Field(default=Decimal("0"), ge=Decimal("0"))
    permanence: PermanenceLevel = Field(default=PermanenceLevel.LONG_TERM)
    permanence_years: Optional[int] = Field(None, ge=0)
    reversal_risk_pct: Decimal = Field(
        default=Decimal("0"), ge=Decimal("0"), le=Decimal("100"),
    )
    entity_id: Optional[str] = Field(None)
    project_id: Optional[str] = Field(None)
    registry: Optional[str] = Field(None)
    vintage_year: Optional[int] = Field(None, ge=1990, le=2100)
    notes: Optional[str] = Field(None, max_length=2000)


class RunSignificanceRequest(GreenLangBase):
    """Request to run significance assessment for indirect categories."""

    categories: List[ISOCategory] = Field(
        default_factory=lambda: [
            ISOCategory.CATEGORY_3_TRANSPORT,
            ISOCategory.CATEGORY_4_PRODUCTS_USED,
            ISOCategory.CATEGORY_5_PRODUCTS_FROM_ORG,
            ISOCategory.CATEGORY_6_OTHER,
        ],
        description="Categories to assess (typically 3-6)",
    )
    threshold_pct: Decimal = Field(
        default=Decimal("1.0"), ge=Decimal("0"), le=Decimal("100"),
    )
    criteria_overrides: Optional[Dict[str, SignificanceCriteria]] = Field(
        None, description="Per-category criteria overrides",
    )


class RunUncertaintyRequest(GreenLangBase):
    """Request to run uncertainty analysis on the inventory."""

    methodology: str = Field(
        default="monte_carlo",
        description="monte_carlo, error_propagation, or expert_judgment",
    )
    iterations: int = Field(default=10_000, ge=1_000, le=1_000_000)
    confidence_levels: List[int] = Field(default=[90, 95, 99])
    categories: Optional[List[ISOCategory]] = Field(
        None, description="Specific categories to analyze; all if not provided",
    )


class GenerateReportRequest(GreenLangBase):
    """Request to generate an ISO 14064-1 compliance report."""

    format: ReportFormat = Field(default=ReportFormat.JSON)
    sections: Optional[List[str]] = Field(
        None, description="Specific sections to include",
    )
    include_cross_walk: bool = Field(
        default=False,
        description="Include GHG Protocol cross-walk comparison",
    )
    include_management_plan: bool = Field(
        default=False,
        description="Include management plan section",
    )


class ExportDataRequest(GreenLangBase):
    """Request to export raw inventory data."""

    format: ReportFormat = Field(default=ReportFormat.CSV)
    categories: Optional[List[ISOCategory]] = Field(None)
    include_sources: bool = Field(
        default=True, description="Include source-level detail",
    )
    include_removals: bool = Field(default=True)
    include_biogenic: bool = Field(default=True)


class SetTargetRequest(GreenLangBase):
    """Request to set an emission reduction target."""

    name: str = Field(default="", max_length=255)
    target_type: str = Field(
        default="absolute", description="absolute or intensity",
    )
    iso_category: Optional[ISOCategory] = Field(
        None, description="Category-specific target",
    )
    base_year: int = Field(..., ge=1990, le=2100)
    base_year_emissions: Decimal = Field(..., ge=Decimal("0"))
    target_year: int = Field(..., ge=1990, le=2100)
    reduction_pct: Decimal = Field(
        ..., ge=Decimal("0"), le=Decimal("100"),
    )

    @field_validator("target_year")
    @classmethod
    def target_after_base(cls, v: int, info) -> int:
        """Target year must be after base year."""
        base = info.data.get("base_year")
        if base is not None and v <= base:
            raise ValueError("target_year must be after base_year")
        return v


class AddImprovementActionRequest(GreenLangBase):
    """Request to add an improvement action to the management plan."""

    name: str = Field(..., min_length=1, max_length=255)
    category: ActionCategory = Field(...)
    iso_category: Optional[ISOCategory] = Field(None)
    description: str = Field(default="", max_length=2000)
    target_reduction_tco2e: Optional[Decimal] = Field(
        None, ge=Decimal("0"),
    )
    target_year: Optional[int] = Field(None, ge=1990, le=2100)
    timeline_start: Optional[date] = Field(None)
    timeline_end: Optional[date] = Field(None)
    estimated_cost_usd: Optional[Decimal] = Field(None, ge=Decimal("0"))
    assigned_to: Optional[str] = Field(None)


class StartVerificationRequest(GreenLangBase):
    """Request to start a verification engagement."""

    level: VerificationLevel = Field(default=VerificationLevel.LIMITED)
    verifier_name: Optional[str] = Field(None)
    verifier_organization: Optional[str] = Field(None)
    verifier_accreditation: Optional[str] = Field(None)
    scope_of_verification: Optional[str] = Field(None)
    materiality_threshold_pct: Decimal = Field(
        default=Decimal("5.0"), ge=Decimal("0"), le=Decimal("100"),
    )


class AddFindingRequest(GreenLangBase):
    """Request to add a verification finding."""

    description: str = Field(..., min_length=10)
    iso_category: Optional[ISOCategory] = Field(None)
    clause_reference: Optional[str] = Field(None)
    severity: FindingSeverity = Field(default=FindingSeverity.LOW)


class UpdateSettingsRequest(GreenLangBase):
    """Request to update platform configuration settings."""

    default_consolidation_approach: Optional[ConsolidationApproach] = Field(None)
    default_gwp_source: Optional[GWPSource] = Field(None)
    significance_threshold_percent: Optional[Decimal] = Field(
        None, ge=Decimal("0"), le=Decimal("100"),
    )
    recalculation_threshold_percent: Optional[Decimal] = Field(
        None, ge=Decimal("0"), le=Decimal("100"),
    )
    monte_carlo_iterations: Optional[int] = Field(
        None, ge=1_000, le=1_000_000,
    )
    confidence_levels: Optional[List[int]] = Field(None)
    reporting_year: Optional[int] = Field(None, ge=1990, le=2100)
    default_report_format: Optional[ReportFormat] = Field(None)
    default_verification_level: Optional[VerificationLevel] = Field(None)
    log_level: Optional[str] = Field(None)


# ---------------------------------------------------------------------------
# Generic API Response Models
# ---------------------------------------------------------------------------

class ApiError(GreenLangBase):
    """Standard API error response."""

    code: str = Field(..., description="Error code (e.g. VALIDATION_ERROR)")
    message: str = Field(..., description="Human-readable error message")
    details: Optional[Dict[str, Any]] = Field(
        None, description="Additional error context",
    )
    timestamp: datetime = Field(default_factory=_now)


class ApiResponse(GreenLangBase):
    """Standard API success response wrapper."""

    success: bool = Field(default=True)
    data: Optional[Any] = Field(None, description="Response payload")
    message: str = Field(default="OK")
    errors: List[ApiError] = Field(default_factory=list)
    timestamp: datetime = Field(default_factory=_now)
    provenance_hash: Optional[str] = Field(
        None, description="Response-level provenance hash",
    )


class PaginatedResponse(GreenLangBase):
    """Paginated list response for collection endpoints."""

    items: List[Any] = Field(default_factory=list)
    total: int = Field(default=0, ge=0, description="Total item count")
    page: int = Field(default=1, ge=1, description="Current page number")
    page_size: int = Field(
        default=50, ge=1, le=500, description="Items per page",
    )
    total_pages: int = Field(
        default=0, ge=0, description="Total number of pages",
    )
    has_next: bool = Field(default=False)
    has_previous: bool = Field(default=False)

    def model_post_init(self, __context: Any) -> None:
        """Compute pagination metadata."""
        if self.page_size > 0 and self.total > 0:
            computed_pages = (self.total + self.page_size - 1) // self.page_size
            object.__setattr__(self, "total_pages", computed_pages)
            object.__setattr__(self, "has_next", self.page < computed_pages)
            object.__setattr__(self, "has_previous", self.page > 1)
