"""
GHG Corporate Platform Domain Models

This module defines all Pydantic domain models for the GL-GHG-APP v1.0
platform.  Models cover the full GHG Protocol Corporate Standard lifecycle:
organizations, boundaries, inventories, emissions, reporting, verification,
and target tracking.

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

from pydantic import BaseModel, Field, field_validator, model_validator

from .config import (
    ConsolidationApproach,
    DataQualityTier,
    EntityType,
    FindingSeverity,
    FindingType,
    GHGGas,
    IntensityDenominator,
    ReportFormat,
    Scope,
    Scope1Category,
    Scope3Category,
    TargetType,
    VerificationLevel,
    VerificationStatus,
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

class Entity(BaseModel):
    """
    An organizational entity (subsidiary, facility, or operation).

    Entities form a hierarchy under an Organization and are subject to
    the chosen consolidation approach.
    """

    id: str = Field(default_factory=_new_id, description="Unique entity ID")
    name: str = Field(..., min_length=1, max_length=255, description="Entity name")
    entity_type: EntityType = Field(..., description="SUBSIDIARY, FACILITY, or OPERATION")
    parent_id: Optional[str] = Field(None, description="Parent entity ID for hierarchy")
    ownership_pct: Decimal = Field(
        default=Decimal("100.0"),
        ge=Decimal("0"),
        le=Decimal("100"),
        description="Ownership percentage (0-100) for equity share approach",
    )
    country: str = Field(..., min_length=2, max_length=3, description="ISO 3166-1 alpha-2/3")
    employees: Optional[int] = Field(None, ge=0, description="Full-time equivalents")
    revenue: Optional[Decimal] = Field(None, ge=Decimal("0"), description="Annual revenue (USD)")
    floor_area_m2: Optional[Decimal] = Field(None, ge=Decimal("0"), description="Floor area in m2")
    production_units: Optional[Decimal] = Field(None, ge=Decimal("0"), description="Annual production units")
    production_unit_name: Optional[str] = Field(None, description="Name of production unit")
    active: bool = Field(default=True, description="Whether entity is currently active")
    created_at: datetime = Field(default_factory=_now)
    updated_at: datetime = Field(default_factory=_now)


class Organization(BaseModel):
    """
    Top-level organization performing GHG accounting.

    Holds the entity hierarchy and metadata needed for boundary setting.
    """

    id: str = Field(default_factory=_new_id, description="Unique organization ID")
    name: str = Field(..., min_length=1, max_length=500, description="Legal entity name")
    industry: str = Field(..., description="Industry sector (e.g. manufacturing, energy)")
    country: str = Field(..., min_length=2, max_length=3, description="HQ country code")
    description: Optional[str] = Field(None, max_length=2000, description="Company description")
    entities: List[Entity] = Field(default_factory=list, description="Owned entities")
    created_at: datetime = Field(default_factory=_now)
    updated_at: datetime = Field(default_factory=_now)


# ---------------------------------------------------------------------------
# Inventory Boundary Models
# ---------------------------------------------------------------------------

class ExclusionRecord(BaseModel):
    """Record of a scope/category exclusion with justification."""

    id: str = Field(default_factory=_new_id)
    scope: Scope = Field(..., description="Scope of the exclusion")
    category: Optional[str] = Field(None, description="Specific category excluded")
    reason: str = Field(..., min_length=10, description="Justification for exclusion")
    magnitude_pct: Decimal = Field(
        ...,
        ge=Decimal("0"),
        le=Decimal("100"),
        description="Estimated magnitude as % of total emissions",
    )
    approved_by: Optional[str] = Field(None, description="Approver ID")
    created_at: datetime = Field(default_factory=_now)


class InventoryBoundary(BaseModel):
    """
    Organizational and operational boundary for a GHG inventory.

    Implements GHG Protocol Ch 3 (organizational) and Ch 4 (operational).
    """

    id: str = Field(default_factory=_new_id)
    org_id: str = Field(..., description="Organization ID")
    consolidation_approach: ConsolidationApproach = Field(
        ...,
        description="Equity share, financial control, or operational control",
    )
    scopes: List[Scope] = Field(
        default_factory=lambda: [Scope.SCOPE_1, Scope.SCOPE_2_LOCATION, Scope.SCOPE_2_MARKET],
        description="Scopes included in the operational boundary",
    )
    base_year: Optional[int] = Field(None, ge=1990, le=2100, description="Base year")
    reporting_year: int = Field(..., ge=1990, le=2100, description="Reporting year")
    entity_ids: List[str] = Field(default_factory=list, description="Included entity IDs")
    exclusions: List[ExclusionRecord] = Field(
        default_factory=list,
        description="Scope/category exclusions",
    )
    created_at: datetime = Field(default_factory=_now)
    updated_at: datetime = Field(default_factory=_now)


# ---------------------------------------------------------------------------
# Base Year Models
# ---------------------------------------------------------------------------

class Recalculation(BaseModel):
    """A base year recalculation event per GHG Protocol Ch 6."""

    id: str = Field(default_factory=_new_id)
    trigger: str = Field(..., description="Structural change trigger type")
    original_value: Decimal = Field(..., description="Original base year emissions (tCO2e)")
    new_value: Decimal = Field(..., description="Recalculated emissions (tCO2e)")
    reason: str = Field(..., min_length=10, description="Detailed justification")
    affected_scopes: List[Scope] = Field(default_factory=list)
    recalculated_at: datetime = Field(default_factory=_now)
    approved_by: Optional[str] = Field(None)

    @property
    def change_pct(self) -> Decimal:
        """Percentage change from original to new value."""
        if self.original_value == 0:
            return Decimal("0")
        return ((self.new_value - self.original_value) / self.original_value) * 100


class BaseYear(BaseModel):
    """
    Base year definition with emissions snapshot per GHG Protocol Ch 6.

    The base year serves as the reference point for tracking emissions
    over time and must be recalculated under specific triggers.
    """

    id: str = Field(default_factory=_new_id)
    org_id: str = Field(..., description="Organization ID")
    year: int = Field(..., ge=1990, le=2100, description="Base year")
    scope1_emissions: Decimal = Field(default=Decimal("0"), ge=Decimal("0"))
    scope2_location_emissions: Decimal = Field(default=Decimal("0"), ge=Decimal("0"))
    scope2_market_emissions: Decimal = Field(default=Decimal("0"), ge=Decimal("0"))
    scope3_emissions: Decimal = Field(default=Decimal("0"), ge=Decimal("0"))
    total_emissions: Decimal = Field(default=Decimal("0"), ge=Decimal("0"))
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
            return self.recalculations[-1].new_value
        return self.total_emissions


# ---------------------------------------------------------------------------
# Scope Emissions Models
# ---------------------------------------------------------------------------

class ScopeEmissions(BaseModel):
    """
    Aggregated emissions for a single scope.

    Provides breakdowns by gas, by category, and by entity.
    """

    scope: Scope = Field(..., description="Emission scope")
    total_tco2e: Decimal = Field(default=Decimal("0"), ge=Decimal("0"))
    by_gas: Dict[str, Decimal] = Field(
        default_factory=dict,
        description="Emissions breakdown by GHG gas (key = GHGGas value)",
    )
    by_category: Dict[str, Decimal] = Field(
        default_factory=dict,
        description="Emissions breakdown by sub-category",
    )
    by_entity: Dict[str, Decimal] = Field(
        default_factory=dict,
        description="Emissions breakdown by entity ID",
    )
    biogenic_co2: Decimal = Field(
        default=Decimal("0"),
        ge=Decimal("0"),
        description="Biogenic CO2 reported separately per GHG Protocol",
    )
    data_quality_tier: DataQualityTier = Field(
        default=DataQualityTier.TIER_1,
        description="Predominant data quality tier for this scope",
    )
    methodology_notes: Optional[str] = Field(
        None,
        description="Notes on calculation methodology",
    )
    provenance_hash: str = Field(
        default="",
        description="SHA-256 hash for audit trail",
    )

    def model_post_init(self, __context: Any) -> None:
        """Compute provenance hash after initialization."""
        if not self.provenance_hash:
            payload = f"{self.scope}:{self.total_tco2e}:{self.by_gas}"
            self.provenance_hash = _sha256(payload)


# ---------------------------------------------------------------------------
# Intensity Metrics
# ---------------------------------------------------------------------------

class IntensityMetric(BaseModel):
    """A single GHG intensity metric (GHG Protocol Ch 12)."""

    id: str = Field(default_factory=_new_id)
    denominator: IntensityDenominator = Field(..., description="Denominator type")
    denominator_value: Decimal = Field(..., gt=Decimal("0"), description="Denominator value")
    denominator_unit: str = Field(default="", description="Unit of the denominator")
    intensity_value: Decimal = Field(..., ge=Decimal("0"), description="tCO2e per denominator unit")
    scope: Optional[Scope] = Field(None, description="If scope-specific (None = all scopes)")
    total_tco2e: Decimal = Field(default=Decimal("0"), description="Numerator emissions")
    unit: str = Field(default="tCO2e/unit", description="Full intensity unit string")


# ---------------------------------------------------------------------------
# Uncertainty Models
# ---------------------------------------------------------------------------

class ScopeUncertainty(BaseModel):
    """Uncertainty results for a single scope."""

    scope: Scope
    mean: Decimal = Field(default=Decimal("0"))
    p5: Decimal = Field(default=Decimal("0"))
    p50: Decimal = Field(default=Decimal("0"))
    p95: Decimal = Field(default=Decimal("0"))
    std_dev: Decimal = Field(default=Decimal("0"))
    cv: Decimal = Field(default=Decimal("0"), description="Coefficient of variation (%)")


class UncertaintyResult(BaseModel):
    """
    Combined uncertainty analysis results from Monte Carlo simulation.

    Follows GHG Protocol Ch 11 guidance on quantifying uncertainty.
    """

    id: str = Field(default_factory=_new_id)
    inventory_id: str = Field(default="")
    iterations: int = Field(default=10_000)
    mean: Decimal = Field(default=Decimal("0"), description="Mean total emissions")
    p5: Decimal = Field(default=Decimal("0"), description="5th percentile")
    p50: Decimal = Field(default=Decimal("0"), description="50th percentile (median)")
    p95: Decimal = Field(default=Decimal("0"), description="95th percentile")
    std_dev: Decimal = Field(default=Decimal("0"), description="Standard deviation")
    cv: Decimal = Field(default=Decimal("0"), description="Coefficient of variation (%)")
    confidence_level: Decimal = Field(default=Decimal("95.0"))
    by_scope: Dict[str, ScopeUncertainty] = Field(
        default_factory=dict,
        description="Per-scope uncertainty breakdowns",
    )
    sensitivity_ranking: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Sources ranked by contribution to uncertainty",
    )
    computed_at: datetime = Field(default_factory=_now)


# ---------------------------------------------------------------------------
# Completeness Models
# ---------------------------------------------------------------------------

class Disclosure(BaseModel):
    """A single mandatory or optional disclosure requirement."""

    id: str = Field(..., description="Disclosure identifier (e.g. MD-01)")
    name: str = Field(..., description="Disclosure name")
    category: str = Field(..., description="Disclosure category")
    required: bool = Field(default=True, description="Whether mandatory")
    present: bool = Field(default=False, description="Whether satisfied in inventory")
    evidence: Optional[str] = Field(None, description="Evidence reference")


class DataGap(BaseModel):
    """An identified gap in the GHG inventory."""

    id: str = Field(default_factory=_new_id)
    scope: Scope = Field(..., description="Scope where gap exists")
    category: Optional[str] = Field(None, description="Sub-category")
    description: str = Field(..., description="Gap description")
    severity: str = Field(default="medium", description="low / medium / high / critical")
    recommendation: str = Field(default="", description="Recommended action to fill gap")
    estimated_magnitude_pct: Optional[Decimal] = Field(
        None,
        description="Estimated impact as % of total scope emissions",
    )


class CompletenessResult(BaseModel):
    """Result of completeness analysis for an inventory."""

    id: str = Field(default_factory=_new_id)
    inventory_id: str = Field(default="")
    overall_pct: Decimal = Field(default=Decimal("0"), ge=Decimal("0"), le=Decimal("100"))
    mandatory_disclosures: List[Disclosure] = Field(default_factory=list)
    optional_disclosures: List[Disclosure] = Field(default_factory=list)
    scope3_materiality: Dict[str, bool] = Field(
        default_factory=dict,
        description="Per-category materiality assessment",
    )
    gaps: List[DataGap] = Field(default_factory=list)
    data_quality_score: Decimal = Field(default=Decimal("0"), ge=Decimal("0"), le=Decimal("100"))
    exclusion_assessment: Dict[str, Any] = Field(default_factory=dict)
    checked_at: datetime = Field(default_factory=_now)


# ---------------------------------------------------------------------------
# Report Models
# ---------------------------------------------------------------------------

class ReportSection(BaseModel):
    """A single section of a generated report."""

    key: str = Field(..., description="Section key (e.g. executive_summary)")
    title: str = Field(..., description="Display title")
    content: Dict[str, Any] = Field(default_factory=dict)
    order: int = Field(default=0)


class Report(BaseModel):
    """A generated GHG inventory report."""

    id: str = Field(default_factory=_new_id)
    inventory_id: str = Field(..., description="Source inventory ID")
    format: ReportFormat = Field(default=ReportFormat.JSON)
    sections: List[ReportSection] = Field(default_factory=list)
    generated_at: datetime = Field(default_factory=_now)
    file_path: Optional[str] = Field(None, description="Path to generated file")
    file_size_bytes: Optional[int] = Field(None, ge=0)
    provenance_hash: str = Field(default="")

    def model_post_init(self, __context: Any) -> None:
        """Compute provenance hash."""
        if not self.provenance_hash:
            payload = f"{self.inventory_id}:{self.format}:{self.generated_at}"
            self.provenance_hash = _sha256(payload)


# ---------------------------------------------------------------------------
# Verification Models
# ---------------------------------------------------------------------------

class VerificationFinding(BaseModel):
    """A finding from internal review or external verification."""

    id: str = Field(default_factory=_new_id)
    finding_type: FindingType = Field(..., description="Finding classification")
    description: str = Field(..., min_length=10, description="Finding details")
    scope: Optional[Scope] = Field(None, description="Affected scope")
    materiality: FindingSeverity = Field(
        default=FindingSeverity.LOW,
        description="Materiality/severity level",
    )
    resolved: bool = Field(default=False)
    resolution: Optional[str] = Field(None, description="Resolution description")
    created_at: datetime = Field(default_factory=_now)
    resolved_at: Optional[datetime] = Field(None)


class VerificationRecord(BaseModel):
    """
    A verification / assurance record for a GHG inventory.

    Tracks the full lifecycle from internal review through external assurance.
    """

    id: str = Field(default_factory=_new_id)
    inventory_id: str = Field(..., description="Inventory being verified")
    level: VerificationLevel = Field(
        default=VerificationLevel.INTERNAL_REVIEW,
    )
    verifier_id: Optional[str] = Field(None, description="Assigned verifier")
    verifier_name: Optional[str] = Field(None)
    verifier_organization: Optional[str] = Field(None)
    status: VerificationStatus = Field(default=VerificationStatus.DRAFT)
    findings: List[VerificationFinding] = Field(default_factory=list)
    statement: Optional[str] = Field(None, description="Verification statement text")
    opinion: Optional[str] = Field(
        None,
        description="Qualified / unqualified / adverse / disclaimer",
    )
    started_at: datetime = Field(default_factory=_now)
    submitted_at: Optional[datetime] = Field(None)
    completed_at: Optional[datetime] = Field(None)

    @property
    def open_findings_count(self) -> int:
        """Count of unresolved findings."""
        return sum(1 for f in self.findings if not f.resolved)

    @property
    def has_major_findings(self) -> bool:
        """Whether any major/critical findings remain unresolved."""
        return any(
            not f.resolved
            and f.materiality in (FindingSeverity.HIGH, FindingSeverity.CRITICAL)
            for f in self.findings
        )


# ---------------------------------------------------------------------------
# Target Tracking Models
# ---------------------------------------------------------------------------

class Target(BaseModel):
    """
    An emission reduction target (absolute or intensity-based).

    Supports SBTi alignment validation.
    """

    id: str = Field(default_factory=_new_id)
    org_id: str = Field(..., description="Organization ID")
    name: str = Field(default="", description="Target name / label")
    target_type: TargetType = Field(..., description="Absolute or intensity")
    scope: Scope = Field(..., description="Target scope")
    base_year: int = Field(..., ge=1990, le=2100)
    base_year_emissions: Decimal = Field(..., ge=Decimal("0"))
    target_year: int = Field(..., ge=1990, le=2100)
    reduction_pct: Decimal = Field(
        ...,
        ge=Decimal("0"),
        le=Decimal("100"),
        description="Target reduction percentage from base year",
    )
    sbti_aligned: bool = Field(default=False, description="Whether SBTi-validated")
    sbti_pathway: Optional[str] = Field(None, description="1.5C or well-below-2C")
    current_emissions: Optional[Decimal] = Field(None, ge=Decimal("0"))
    current_year: Optional[int] = Field(None)
    created_at: datetime = Field(default_factory=_now)
    updated_at: datetime = Field(default_factory=_now)

    @field_validator("target_year")
    @classmethod
    def target_after_base(cls, v: int, info) -> int:
        """Target year must be after base year."""
        base = info.data.get("base_year")
        if base is not None and v <= base:
            raise ValueError("target_year must be after base_year")
        return v

    @property
    def current_progress_pct(self) -> Decimal:
        """Calculate current progress toward target."""
        if self.current_emissions is None or self.base_year_emissions == 0:
            return Decimal("0")
        actual_reduction = self.base_year_emissions - self.current_emissions
        target_reduction = self.base_year_emissions * (self.reduction_pct / Decimal("100"))
        if target_reduction == 0:
            return Decimal("0")
        return min((actual_reduction / target_reduction) * 100, Decimal("100"))


# ---------------------------------------------------------------------------
# GHG Inventory -- The Central Object
# ---------------------------------------------------------------------------

class GHGInventory(BaseModel):
    """
    Complete GHG inventory for an organization-year.

    This is the central data structure aggregating all scope emissions,
    intensity metrics, uncertainty, and completeness assessments.
    """

    id: str = Field(default_factory=_new_id)
    org_id: str = Field(..., description="Organization ID")
    year: int = Field(..., ge=1990, le=2100, description="Reporting year")
    boundary: Optional[InventoryBoundary] = Field(None)
    scope1: Optional[ScopeEmissions] = Field(None, description="Scope 1 emissions")
    scope2_location: Optional[ScopeEmissions] = Field(
        None,
        description="Scope 2 location-based emissions",
    )
    scope2_market: Optional[ScopeEmissions] = Field(
        None,
        description="Scope 2 market-based emissions",
    )
    scope3: Optional[ScopeEmissions] = Field(None, description="Scope 3 emissions")
    grand_total_tco2e: Decimal = Field(
        default=Decimal("0"),
        ge=Decimal("0"),
        description="Grand total using Scope 2 market-based",
    )
    grand_total_location_tco2e: Decimal = Field(
        default=Decimal("0"),
        ge=Decimal("0"),
        description="Grand total using Scope 2 location-based",
    )
    intensity_metrics: List[IntensityMetric] = Field(default_factory=list)
    uncertainty: Optional[UncertaintyResult] = Field(None)
    data_quality_score: Decimal = Field(
        default=Decimal("0"),
        ge=Decimal("0"),
        le=Decimal("100"),
    )
    completeness: Optional[CompletenessResult] = Field(None)
    verification: Optional[VerificationRecord] = Field(None)
    status: str = Field(default="draft", description="draft / final / verified")
    created_at: datetime = Field(default_factory=_now)
    updated_at: datetime = Field(default_factory=_now)
    provenance_hash: str = Field(default="")

    def recalculate_totals(self) -> None:
        """Recalculate grand totals from scope emissions."""
        s1 = self.scope1.total_tco2e if self.scope1 else Decimal("0")
        s2m = self.scope2_market.total_tco2e if self.scope2_market else Decimal("0")
        s2l = self.scope2_location.total_tco2e if self.scope2_location else Decimal("0")
        s3 = self.scope3.total_tco2e if self.scope3 else Decimal("0")
        self.grand_total_tco2e = s1 + s2m + s3
        self.grand_total_location_tco2e = s1 + s2l + s3
        self.updated_at = _now()

    def model_post_init(self, __context: Any) -> None:
        """Compute provenance hash and totals."""
        self.recalculate_totals()
        if not self.provenance_hash:
            payload = f"{self.org_id}:{self.year}:{self.grand_total_tco2e}"
            self.provenance_hash = _sha256(payload)


# ---------------------------------------------------------------------------
# Dashboard Metrics
# ---------------------------------------------------------------------------

class DashboardMetrics(BaseModel):
    """Aggregated dashboard metrics for a single inventory year."""

    org_id: str = Field(...)
    year: int = Field(...)
    total_emissions: Decimal = Field(default=Decimal("0"))
    scope1_total: Decimal = Field(default=Decimal("0"))
    scope2_location_total: Decimal = Field(default=Decimal("0"))
    scope2_market_total: Decimal = Field(default=Decimal("0"))
    scope3_total: Decimal = Field(default=Decimal("0"))
    yoy_change_pct: Optional[Decimal] = Field(None)
    intensity_metrics: List[IntensityMetric] = Field(default_factory=list)
    data_quality_score: Decimal = Field(default=Decimal("0"))
    completeness_pct: Decimal = Field(default=Decimal("0"))
    target_progress_pct: Optional[Decimal] = Field(None)
    top_categories: List[Dict[str, Any]] = Field(default_factory=list)
    scope3_breakdown: Dict[str, Decimal] = Field(default_factory=dict)
    biogenic_co2: Decimal = Field(default=Decimal("0"))
    computed_at: datetime = Field(default_factory=_now)


# ---------------------------------------------------------------------------
# Request / Response Models
# ---------------------------------------------------------------------------

class CreateOrganizationRequest(BaseModel):
    """Request to create a new organization."""

    name: str = Field(..., min_length=1, max_length=500)
    industry: str = Field(..., min_length=1)
    country: str = Field(..., min_length=2, max_length=3)
    description: Optional[str] = Field(None, max_length=2000)


class AddEntityRequest(BaseModel):
    """Request to add an entity to an organization."""

    name: str = Field(..., min_length=1, max_length=255)
    entity_type: EntityType = Field(...)
    parent_id: Optional[str] = Field(None)
    ownership_pct: Decimal = Field(default=Decimal("100.0"), ge=Decimal("0"), le=Decimal("100"))
    country: str = Field(..., min_length=2, max_length=3)
    employees: Optional[int] = Field(None, ge=0)
    revenue: Optional[Decimal] = Field(None, ge=Decimal("0"))
    floor_area_m2: Optional[Decimal] = Field(None, ge=Decimal("0"))
    production_units: Optional[Decimal] = Field(None, ge=Decimal("0"))
    production_unit_name: Optional[str] = Field(None)


class UpdateEntityRequest(BaseModel):
    """Request to update an existing entity."""

    name: Optional[str] = Field(None, min_length=1, max_length=255)
    entity_type: Optional[EntityType] = Field(None)
    parent_id: Optional[str] = Field(None)
    ownership_pct: Optional[Decimal] = Field(None, ge=Decimal("0"), le=Decimal("100"))
    country: Optional[str] = Field(None, min_length=2, max_length=3)
    employees: Optional[int] = Field(None, ge=0)
    revenue: Optional[Decimal] = Field(None, ge=Decimal("0"))
    floor_area_m2: Optional[Decimal] = Field(None, ge=Decimal("0"))
    production_units: Optional[Decimal] = Field(None, ge=Decimal("0"))
    production_unit_name: Optional[str] = Field(None)
    active: Optional[bool] = Field(None)


class SetBoundaryRequest(BaseModel):
    """Request to set organizational/operational boundary."""

    consolidation_approach: ConsolidationApproach = Field(...)
    scopes: List[Scope] = Field(
        default_factory=lambda: [Scope.SCOPE_1, Scope.SCOPE_2_LOCATION, Scope.SCOPE_2_MARKET],
    )
    reporting_year: int = Field(..., ge=1990, le=2100)
    entity_ids: Optional[List[str]] = Field(None)


class CreateInventoryRequest(BaseModel):
    """Request to create a new GHG inventory."""

    year: int = Field(..., ge=1990, le=2100)
    consolidation_approach: Optional[ConsolidationApproach] = Field(None)
    scopes: Optional[List[Scope]] = Field(None)


class SetBaseYearRequest(BaseModel):
    """Request to set or update the base year."""

    year: int = Field(..., ge=1990, le=2100)
    scope1_emissions: Decimal = Field(default=Decimal("0"), ge=Decimal("0"))
    scope2_location_emissions: Decimal = Field(default=Decimal("0"), ge=Decimal("0"))
    scope2_market_emissions: Decimal = Field(default=Decimal("0"), ge=Decimal("0"))
    scope3_emissions: Decimal = Field(default=Decimal("0"), ge=Decimal("0"))
    justification: str = Field(..., min_length=10)


class RecalculateBaseYearRequest(BaseModel):
    """Request to recalculate the base year."""

    trigger: str = Field(..., description="Structural change trigger")
    new_scope1: Decimal = Field(default=Decimal("0"), ge=Decimal("0"))
    new_scope2_location: Decimal = Field(default=Decimal("0"), ge=Decimal("0"))
    new_scope2_market: Decimal = Field(default=Decimal("0"), ge=Decimal("0"))
    new_scope3: Decimal = Field(default=Decimal("0"), ge=Decimal("0"))
    reason: str = Field(..., min_length=10)


class SetTargetRequest(BaseModel):
    """Request to set an emission reduction target."""

    name: str = Field(default="")
    target_type: TargetType = Field(...)
    scope: Scope = Field(...)
    base_year: int = Field(..., ge=1990, le=2100)
    base_year_emissions: Decimal = Field(..., ge=Decimal("0"))
    target_year: int = Field(..., ge=1990, le=2100)
    reduction_pct: Decimal = Field(..., ge=Decimal("0"), le=Decimal("100"))
    sbti_aligned: bool = Field(default=False)
    sbti_pathway: Optional[str] = Field(None)


class GenerateReportRequest(BaseModel):
    """Request to generate a report."""

    format: ReportFormat = Field(default=ReportFormat.JSON)
    sections: Optional[List[str]] = Field(None, description="Specific sections to include")


class StartVerificationRequest(BaseModel):
    """Request to start verification."""

    level: VerificationLevel = Field(default=VerificationLevel.INTERNAL_REVIEW)
    reviewer_id: Optional[str] = Field(None)
    verifier_name: Optional[str] = Field(None)
    verifier_organization: Optional[str] = Field(None)


class AddFindingRequest(BaseModel):
    """Request to add a verification finding."""

    finding_type: FindingType = Field(...)
    description: str = Field(..., min_length=10)
    scope: Optional[Scope] = Field(None)
    materiality: FindingSeverity = Field(default=FindingSeverity.LOW)


class AggregateEmissionsRequest(BaseModel):
    """Request to aggregate emissions for an inventory."""

    inventory_id: str = Field(...)
    entity_data: Optional[Dict[str, Any]] = Field(
        None,
        description="Optional pre-loaded entity data; fetched if not provided",
    )
    scopes: Optional[List[Scope]] = Field(
        None,
        description="Specific scopes to aggregate; all if not provided",
    )


class ExclusionRequest(BaseModel):
    """Request to add a scope/category exclusion."""

    scope: Scope = Field(...)
    category: Optional[str] = Field(None)
    reason: str = Field(..., min_length=10)
    magnitude_pct: Decimal = Field(..., ge=Decimal("0"), le=Decimal("100"))
