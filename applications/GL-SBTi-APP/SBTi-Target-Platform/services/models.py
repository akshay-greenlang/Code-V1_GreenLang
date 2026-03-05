"""
GL-SBTi-APP v1.0 -- SBTi Target Validation Platform Domain Models

This module defines all Pydantic v2 domain models for the GL-SBTi-APP v1.0
platform.  Models cover the full SBTi target lifecycle: organizations,
emissions inventories, target definition, pathway calculation, near-term and
net-zero validation, Scope 3 screening, FLAG assessment, progress tracking,
temperature scoring, recalculation management, five-year review cycles,
financial institution portfolios, multi-framework alignment mapping,
gap analysis, report generation, and submission forms.

All monetary values are in USD unless otherwise noted.  All emissions are in
metric tonnes CO2e.  Timestamps are UTC.

Reference:
    - SBTi Corporate Net-Zero Standard v1.2 (April 2023)
    - SBTi Criteria and Recommendations v5.1 (April 2023)
    - SBTi Financial Institutions Net-Zero Standard v1.1 (April 2024)
    - GHG Protocol Corporate Standard (2004, revised 2015)
    - GHG Protocol Scope 3 Standard (2011)
    - PCAF Global GHG Accounting & Reporting Standard (2022)

Example:
    >>> from .config import TargetType, AmbitionLevel
    >>> target = Target(
    ...     tenant_id="tenant-1", org_id="org-1",
    ...     target_type=TargetType.NEAR_TERM,
    ...     ambition_level=AmbitionLevel.ONE_POINT_FIVE_C,
    ...     base_year=2020, target_year=2030, reduction_pct=Decimal("42"))
    >>> print(target.id)
"""

from __future__ import annotations

import hashlib
import uuid
from datetime import date, datetime
from decimal import Decimal
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator, model_validator

from .config import (
    AmbitionLevel,
    FIAssetClass,
    FITargetType,
    GapCategory,
    GapSeverity,
    NotificationType,
    PCAFDataQuality,
    RecalculationTrigger,
    ReportFormat,
    ReviewOutcome,
    SBTiSector,
    TargetMethod,
    TargetScope,
    TargetType,
    ValidationStatus,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _new_id() -> str:
    """Generate a UUID4 string."""
    return str(uuid.uuid4())


def _now() -> datetime:
    """UTC now truncated to seconds."""
    return datetime.utcnow().replace(microsecond=0)


def _sha256(payload: str) -> str:
    """SHA-256 hex digest for provenance tracking."""
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Organization Models
# ---------------------------------------------------------------------------

class Organization(BaseModel):
    """
    Organization registered on the SBTi Target Validation Platform.

    Holds the organizational profile, sector classification (ISIC, NACE,
    NAICS), OECD development status, and reporting boundary metadata.
    Multi-tenant via tenant_id.
    """

    id: str = Field(default_factory=_new_id, description="Unique organization ID")
    tenant_id: str = Field(..., description="Tenant ID for multi-tenancy isolation")
    name: str = Field(..., min_length=1, max_length=500, description="Legal entity name")
    sector: SBTiSector = Field(
        default=SBTiSector.GENERAL, description="SBTi sector classification",
    )
    isic_code: Optional[str] = Field(
        None, max_length=10, description="ISIC Rev. 4 code (e.g. 3510 for power generation)",
    )
    nace_code: Optional[str] = Field(
        None, max_length=10, description="NACE Rev. 2 code",
    )
    naics_code: Optional[str] = Field(
        None, max_length=10, description="NAICS code",
    )
    oecd_status: Optional[str] = Field(
        None, max_length=20,
        description="OECD development status (developed, emerging, developing)",
    )
    country: str = Field(..., min_length=2, max_length=3, description="HQ country ISO 3166")
    region: Optional[str] = Field(None, max_length=100, description="Geographic region")
    description: Optional[str] = Field(None, max_length=2000)
    employee_count: Optional[int] = Field(None, ge=0, description="Full-time equivalents")
    annual_revenue_usd: Optional[Decimal] = Field(None, ge=Decimal("0"))
    total_assets_usd: Optional[Decimal] = Field(None, ge=Decimal("0"))
    is_financial_institution: bool = Field(
        default=False, description="Whether FI-specific guidance applies",
    )
    is_flag_relevant: bool = Field(
        default=False, description="Whether FLAG guidance applies",
    )
    validation_status: ValidationStatus = Field(
        default=ValidationStatus.COMMITMENT_LETTER,
        description="Current SBTi validation status",
    )
    commitment_date: Optional[date] = Field(
        None, description="Date of SBTi commitment letter",
    )
    contact_person: Optional[str] = Field(None, max_length=255)
    contact_email: Optional[str] = Field(None, max_length=255)
    created_at: datetime = Field(default_factory=_now)
    updated_at: datetime = Field(default_factory=_now)


# ---------------------------------------------------------------------------
# Emissions Inventory Models
# ---------------------------------------------------------------------------

class Scope3CategoryEmissions(BaseModel):
    """Emissions data for a single Scope 3 category."""

    category_number: int = Field(..., ge=1, le=15, description="Category number (1-15)")
    category_name: str = Field(default="", max_length=255)
    emissions_tco2e: Decimal = Field(default=Decimal("0"), ge=Decimal("0"))
    percentage_of_scope3: Decimal = Field(
        default=Decimal("0"), ge=Decimal("0"), le=Decimal("100"),
    )
    data_quality: str = Field(
        default="estimated", description="measured, calculated, estimated, proxy, default",
    )
    methodology: Optional[str] = Field(None, max_length=500)
    mrv_agent_id: Optional[str] = Field(None, description="MRV agent source (e.g. MRV-014)")
    included_in_target: bool = Field(default=False)
    is_relevant: bool = Field(default=True, description="Whether category is relevant")
    exclusion_reason: Optional[str] = Field(None, max_length=500)


class EmissionsInventory(BaseModel):
    """
    Complete GHG emissions inventory for a given year.

    Tracks Scope 1, Scope 2 (location and market-based), Scope 3 total and
    per-category breakdown, FLAG emissions, and bioenergy CO2.  This model
    serves as the base year inventory and annual tracking record.
    """

    id: str = Field(default_factory=_new_id, description="Inventory ID")
    tenant_id: str = Field(..., description="Tenant ID")
    org_id: str = Field(..., description="Organization ID")
    year: int = Field(..., ge=2010, le=2100, description="Inventory year")
    is_base_year: bool = Field(default=False, description="Whether this is the base year inventory")
    scope1_tco2e: Decimal = Field(
        default=Decimal("0"), ge=Decimal("0"), description="Scope 1 total (tCO2e)",
    )
    scope2_location_tco2e: Decimal = Field(
        default=Decimal("0"), ge=Decimal("0"), description="Scope 2 location-based (tCO2e)",
    )
    scope2_market_tco2e: Decimal = Field(
        default=Decimal("0"), ge=Decimal("0"), description="Scope 2 market-based (tCO2e)",
    )
    scope3_total_tco2e: Decimal = Field(
        default=Decimal("0"), ge=Decimal("0"), description="Scope 3 total (tCO2e)",
    )
    scope3_categories: List[Scope3CategoryEmissions] = Field(
        default_factory=list, description="Per-category Scope 3 breakdown",
    )
    flag_tco2e: Decimal = Field(
        default=Decimal("0"), ge=Decimal("0"),
        description="FLAG (Forest, Land, Agriculture) emissions (tCO2e)",
    )
    bioenergy_co2_tco2e: Decimal = Field(
        default=Decimal("0"), ge=Decimal("0"),
        description="Biogenic CO2 emissions reported separately (tCO2e)",
    )
    total_s1_s2_tco2e: Decimal = Field(
        default=Decimal("0"), ge=Decimal("0"),
        description="Scope 1 + Scope 2 total (using market-based default)",
    )
    total_s1_s2_s3_tco2e: Decimal = Field(
        default=Decimal("0"), ge=Decimal("0"),
        description="Scope 1 + 2 + 3 total",
    )
    scope3_as_pct_of_total: Decimal = Field(
        default=Decimal("0"), ge=Decimal("0"), le=Decimal("100"),
        description="Scope 3 as percentage of total S1+S2+S3",
    )
    flag_as_pct_of_total: Decimal = Field(
        default=Decimal("0"), ge=Decimal("0"), le=Decimal("100"),
        description="FLAG as percentage of total S1+S2+S3",
    )
    scope1_coverage_pct: Decimal = Field(
        default=Decimal("100"), ge=Decimal("0"), le=Decimal("100"),
        description="Percentage of S1 emissions covered",
    )
    scope2_coverage_pct: Decimal = Field(
        default=Decimal("100"), ge=Decimal("0"), le=Decimal("100"),
        description="Percentage of S2 emissions covered",
    )
    verification_status: str = Field(
        default="not_verified",
        description="not_verified, limited, reasonable",
    )
    verifier_name: Optional[str] = Field(None, max_length=255)
    verification_standard: Optional[str] = Field(
        None, description="e.g. ISO 14064-3, ISAE 3000",
    )
    data_quality_overall: str = Field(
        default="estimated",
        description="measured, calculated, estimated, proxy, default",
    )
    notes: Optional[str] = Field(None, max_length=5000)
    provenance_hash: str = Field(default="")
    created_at: datetime = Field(default_factory=_now)
    updated_at: datetime = Field(default_factory=_now)

    def model_post_init(self, __context: Any) -> None:
        """Compute derived totals and provenance hash."""
        s1_s2 = self.scope1_tco2e + self.scope2_market_tco2e
        if self.total_s1_s2_tco2e == Decimal("0") and s1_s2 > 0:
            object.__setattr__(self, "total_s1_s2_tco2e", s1_s2)

        total = s1_s2 + self.scope3_total_tco2e
        if self.total_s1_s2_s3_tco2e == Decimal("0") and total > 0:
            object.__setattr__(self, "total_s1_s2_s3_tco2e", total)

        if total > 0 and self.scope3_as_pct_of_total == Decimal("0"):
            pct = (self.scope3_total_tco2e / total) * 100
            object.__setattr__(self, "scope3_as_pct_of_total", pct.quantize(Decimal("0.01")))

        if total > 0 and self.flag_as_pct_of_total == Decimal("0") and self.flag_tco2e > 0:
            fpct = (self.flag_tco2e / total) * 100
            object.__setattr__(self, "flag_as_pct_of_total", fpct.quantize(Decimal("0.01")))

        if not self.provenance_hash:
            payload = (
                f"{self.org_id}:{self.year}:"
                f"{self.scope1_tco2e}:{self.scope2_market_tco2e}:"
                f"{self.scope3_total_tco2e}"
            )
            self.provenance_hash = _sha256(payload)


# ---------------------------------------------------------------------------
# Target Models
# ---------------------------------------------------------------------------

class TargetScopeDetail(BaseModel):
    """
    Per-scope details within a target.

    Tracks scope-specific coverage, base-year emissions, target-year
    emissions, and reduction calculations.
    """

    scope: TargetScope = Field(..., description="Scope covered")
    base_year_emissions_tco2e: Decimal = Field(
        default=Decimal("0"), ge=Decimal("0"),
    )
    target_year_emissions_tco2e: Decimal = Field(
        default=Decimal("0"), ge=Decimal("0"),
    )
    reduction_pct: Decimal = Field(
        default=Decimal("0"), ge=Decimal("0"), le=Decimal("100"),
    )
    coverage_pct: Decimal = Field(
        default=Decimal("0"), ge=Decimal("0"), le=Decimal("100"),
        description="Coverage as percentage of total scope emissions",
    )
    included_categories: List[int] = Field(
        default_factory=list, description="Scope 3 category numbers included (1-15)",
    )
    intensity_metric: Optional[str] = Field(None, max_length=100)
    base_intensity_value: Optional[Decimal] = Field(None)
    target_intensity_value: Optional[Decimal] = Field(None)
    notes: Optional[str] = Field(None, max_length=2000)


class Target(BaseModel):
    """
    An SBTi science-based target definition.

    Covers near-term, long-term, and net-zero targets with all fields
    required for SBTi validation: type, scope, method, ambition level,
    base year, target year, reduction percentage, annual rate, coverage,
    intensity details, FLAG details, and lifecycle status.
    """

    id: str = Field(default_factory=_new_id, description="Target ID")
    tenant_id: str = Field(..., description="Tenant ID")
    org_id: str = Field(..., description="Organization ID")
    name: str = Field(
        default="", max_length=500,
        description="Target display name (e.g. 'Near-term Scope 1+2 ACA 1.5C')",
    )
    target_type: TargetType = Field(..., description="Near-term, long-term, or net-zero")
    scope: TargetScope = Field(..., description="Scope coverage of this target")
    method: TargetMethod = Field(
        default=TargetMethod.ABSOLUTE_CONTRACTION,
        description="Target-setting method",
    )
    ambition_level: AmbitionLevel = Field(
        default=AmbitionLevel.ONE_POINT_FIVE_C,
        description="Temperature ambition alignment",
    )
    base_year: int = Field(..., ge=2015, le=2030, description="Base year for the target")
    target_year: int = Field(..., ge=2025, le=2060, description="Target year")
    base_year_emissions_tco2e: Decimal = Field(
        default=Decimal("0"), ge=Decimal("0"),
        description="Total emissions in base year (within target boundary)",
    )
    target_year_emissions_tco2e: Decimal = Field(
        default=Decimal("0"), ge=Decimal("0"),
        description="Expected emissions in target year",
    )
    reduction_pct: Decimal = Field(
        default=Decimal("0"), ge=Decimal("0"), le=Decimal("100"),
        description="Percentage reduction from base year",
    )
    annual_linear_reduction_rate: Decimal = Field(
        default=Decimal("0"), ge=Decimal("0"), le=Decimal("100"),
        description="Annual linear reduction rate as percentage",
    )
    coverage_pct: Decimal = Field(
        default=Decimal("0"), ge=Decimal("0"), le=Decimal("100"),
        description="Coverage as fraction of total scope emissions",
    )
    scope_details: List[TargetScopeDetail] = Field(
        default_factory=list, description="Per-scope breakdown within this target",
    )

    # Intensity fields (for SDA / physical intensity targets)
    is_intensity_target: bool = Field(default=False)
    intensity_metric: Optional[str] = Field(None, max_length=100)
    intensity_metric_unit: Optional[str] = Field(None, max_length=100)
    base_intensity_value: Optional[Decimal] = Field(None)
    target_intensity_value: Optional[Decimal] = Field(None)
    production_base_year: Optional[Decimal] = Field(None, ge=Decimal("0"))
    production_target_year: Optional[Decimal] = Field(None, ge=Decimal("0"))

    # FLAG fields
    is_flag_target: bool = Field(default=False)
    flag_commodity: Optional[str] = Field(None, max_length=100)
    flag_base_intensity: Optional[Decimal] = Field(None)
    flag_target_intensity: Optional[Decimal] = Field(None)
    deforestation_commitment: bool = Field(
        default=False, description="Commitment to zero deforestation by 2025",
    )
    deforestation_commitment_date: Optional[date] = Field(None)

    # Status and lifecycle
    validation_status: ValidationStatus = Field(
        default=ValidationStatus.TARGET_SUBMITTED,
        description="Target validation lifecycle status",
    )
    submission_date: Optional[date] = Field(None, description="Date submitted to SBTi")
    validation_date: Optional[date] = Field(None, description="Date validated by SBTi")
    publication_date: Optional[date] = Field(None, description="Date published on SBTi website")
    expiry_date: Optional[date] = Field(None, description="Target expiry date")
    next_review_date: Optional[date] = Field(None, description="Next 5-year review date")

    notes: Optional[str] = Field(None, max_length=5000)
    provenance_hash: str = Field(default="")
    created_at: datetime = Field(default_factory=_now)
    updated_at: datetime = Field(default_factory=_now)

    @field_validator("target_year")
    @classmethod
    def target_year_after_base(cls, v: int, info) -> int:
        """Target year must be after base year."""
        base = info.data.get("base_year")
        if base is not None and v <= base:
            raise ValueError("target_year must be after base_year")
        return v

    def model_post_init(self, __context: Any) -> None:
        """Compute derived fields and provenance hash."""
        if (
            self.annual_linear_reduction_rate == Decimal("0")
            and self.reduction_pct > Decimal("0")
            and self.target_year > self.base_year
        ):
            years = self.target_year - self.base_year
            rate = self.reduction_pct / Decimal(str(years))
            object.__setattr__(self, "annual_linear_reduction_rate", rate.quantize(Decimal("0.01")))

        if (
            self.target_year_emissions_tco2e == Decimal("0")
            and self.base_year_emissions_tco2e > Decimal("0")
            and self.reduction_pct > Decimal("0")
        ):
            target_emissions = self.base_year_emissions_tco2e * (
                Decimal("1") - self.reduction_pct / Decimal("100")
            )
            object.__setattr__(self, "target_year_emissions_tco2e", target_emissions.quantize(Decimal("0.01")))

        if not self.provenance_hash:
            payload = (
                f"{self.org_id}:{self.target_type}:{self.scope}:"
                f"{self.base_year}:{self.target_year}:{self.reduction_pct}"
            )
            self.provenance_hash = _sha256(payload)


# ---------------------------------------------------------------------------
# Pathway Models
# ---------------------------------------------------------------------------

class PathwayMilestone(BaseModel):
    """A single year-by-year milestone in a calculated pathway."""

    year: int = Field(..., ge=2015, le=2100, description="Milestone year")
    expected_emissions_tco2e: Decimal = Field(
        default=Decimal("0"), ge=Decimal("0"),
        description="Expected absolute emissions at this year",
    )
    expected_intensity_value: Optional[Decimal] = Field(
        None, description="Expected intensity value for SDA targets",
    )
    cumulative_reduction_pct: Decimal = Field(
        default=Decimal("0"), ge=Decimal("0"), le=Decimal("100"),
        description="Cumulative reduction from base year",
    )
    annual_budget_tco2e: Optional[Decimal] = Field(
        None, ge=Decimal("0"), description="Annual carbon budget for this year",
    )


class Pathway(BaseModel):
    """
    Calculated emissions reduction pathway for a target.

    Contains year-by-year milestones showing the expected trajectory
    from base year to target year based on the chosen method.
    """

    id: str = Field(default_factory=_new_id, description="Pathway ID")
    tenant_id: str = Field(..., description="Tenant ID")
    org_id: str = Field(..., description="Organization ID")
    target_id: str = Field(..., description="Target ID")
    method: TargetMethod = Field(..., description="Calculation method")
    ambition_level: AmbitionLevel = Field(
        default=AmbitionLevel.ONE_POINT_FIVE_C,
    )
    base_year: int = Field(..., ge=2015, le=2030)
    target_year: int = Field(..., ge=2025, le=2060)
    base_emissions_tco2e: Decimal = Field(
        default=Decimal("0"), ge=Decimal("0"),
    )
    target_emissions_tco2e: Decimal = Field(
        default=Decimal("0"), ge=Decimal("0"),
    )
    annual_reduction_rate: Decimal = Field(
        default=Decimal("0"), ge=Decimal("0"), le=Decimal("100"),
    )
    milestones: List[PathwayMilestone] = Field(
        default_factory=list, description="Year-by-year pathway milestones",
    )
    cumulative_budget_tco2e: Optional[Decimal] = Field(
        None, ge=Decimal("0"), description="Total cumulative carbon budget",
    )
    sector: Optional[SBTiSector] = Field(None, description="Sector for SDA pathways")
    intensity_metric: Optional[str] = Field(None, max_length=100)
    provenance_hash: str = Field(default="")
    calculated_at: datetime = Field(default_factory=_now)

    def model_post_init(self, __context: Any) -> None:
        """Compute provenance hash."""
        if not self.provenance_hash:
            payload = (
                f"{self.target_id}:{self.method}:{self.base_year}:"
                f"{self.target_year}:{self.base_emissions_tco2e}"
            )
            self.provenance_hash = _sha256(payload)


# ---------------------------------------------------------------------------
# Validation Models
# ---------------------------------------------------------------------------

class CriterionCheck(BaseModel):
    """Result of evaluating a single SBTi validation criterion."""

    criterion_id: str = Field(..., description="Criterion ID (e.g. C1, NZ-C3)")
    criterion_name: str = Field(default="", max_length=255)
    result: str = Field(
        ..., description="pass, fail, not_applicable, insufficient_data",
    )
    message: str = Field(default="", max_length=2000, description="Explanation of result")
    details: Optional[Dict[str, Any]] = Field(
        None, description="Additional structured details (values, thresholds)",
    )
    evidence_ref: Optional[str] = Field(
        None, max_length=500, description="Reference to supporting evidence",
    )
    remediation: Optional[str] = Field(
        None, max_length=2000, description="Suggested remediation if failed",
    )


class ValidationSummary(BaseModel):
    """Summary statistics of a validation run."""

    total_criteria: int = Field(default=0, ge=0)
    passed: int = Field(default=0, ge=0)
    failed: int = Field(default=0, ge=0)
    not_applicable: int = Field(default=0, ge=0)
    insufficient_data: int = Field(default=0, ge=0)
    readiness_pct: Decimal = Field(
        default=Decimal("0"), ge=Decimal("0"), le=Decimal("100"),
        description="Percentage of applicable criteria passed",
    )


class ValidationResult(BaseModel):
    """
    Complete validation result for one or more targets.

    Contains per-criterion checks, summary statistics, and an overall
    readiness assessment for SBTi submission.
    """

    id: str = Field(default_factory=_new_id, description="Validation ID")
    tenant_id: str = Field(..., description="Tenant ID")
    org_id: str = Field(..., description="Organization ID")
    target_ids: List[str] = Field(default_factory=list, description="Target IDs validated")
    validation_type: str = Field(
        default="near_term", description="near_term, net_zero, or combined",
    )
    criterion_checks: List[CriterionCheck] = Field(
        default_factory=list, description="Per-criterion check results",
    )
    summary: ValidationSummary = Field(
        default_factory=ValidationSummary, description="Summary statistics",
    )
    overall_result: str = Field(
        default="insufficient_data",
        description="pass, fail, not_applicable, insufficient_data",
    )
    is_submission_ready: bool = Field(
        default=False, description="Whether targets are ready for SBTi submission",
    )
    blocking_criteria: List[str] = Field(
        default_factory=list, description="Criterion IDs that are blocking",
    )
    recommendations: List[str] = Field(
        default_factory=list, description="Recommendations for achieving readiness",
    )
    validated_at: datetime = Field(default_factory=_now)
    provenance_hash: str = Field(default="")

    def model_post_init(self, __context: Any) -> None:
        """Compute summary, overall result, and provenance hash."""
        if self.criterion_checks and self.summary.total_criteria == 0:
            total = len(self.criterion_checks)
            passed = sum(1 for c in self.criterion_checks if c.result == "pass")
            failed = sum(1 for c in self.criterion_checks if c.result == "fail")
            na = sum(1 for c in self.criterion_checks if c.result == "not_applicable")
            insuf = sum(1 for c in self.criterion_checks if c.result == "insufficient_data")
            applicable = total - na
            readiness = Decimal(str((passed / applicable * 100) if applicable > 0 else 0))
            object.__setattr__(self, "summary", ValidationSummary(
                total_criteria=total,
                passed=passed,
                failed=failed,
                not_applicable=na,
                insufficient_data=insuf,
                readiness_pct=readiness.quantize(Decimal("0.01")),
            ))
            if failed == 0 and insuf == 0 and passed > 0:
                object.__setattr__(self, "overall_result", "pass")
                object.__setattr__(self, "is_submission_ready", True)
            elif failed > 0:
                object.__setattr__(self, "overall_result", "fail")
                blocking = [c.criterion_id for c in self.criterion_checks if c.result == "fail"]
                object.__setattr__(self, "blocking_criteria", blocking)

        if not self.provenance_hash:
            payload = (
                f"{self.org_id}:{self.validation_type}:"
                f"{self.overall_result}:{self.validated_at}"
            )
            self.provenance_hash = _sha256(payload)


# ---------------------------------------------------------------------------
# Scope 3 Screening Models
# ---------------------------------------------------------------------------

class CategoryBreakdown(BaseModel):
    """Emissions breakdown for a single Scope 3 category in screening."""

    category_number: int = Field(..., ge=1, le=15)
    category_name: str = Field(default="")
    emissions_tco2e: Decimal = Field(default=Decimal("0"), ge=Decimal("0"))
    percentage_of_scope3: Decimal = Field(
        default=Decimal("0"), ge=Decimal("0"), le=Decimal("100"),
    )
    percentage_of_total: Decimal = Field(
        default=Decimal("0"), ge=Decimal("0"), le=Decimal("100"),
    )
    data_quality: str = Field(default="estimated")
    included_in_target: bool = Field(default=False)
    is_material: bool = Field(default=False, description="Whether category is material (>1%)")


class Scope3Screening(BaseModel):
    """
    Scope 3 screening result per SBTi criteria.

    Assesses whether Scope 3 exceeds 40% of total, breaks down categories,
    and determines required coverage for target setting.
    """

    id: str = Field(default_factory=_new_id, description="Screening ID")
    tenant_id: str = Field(..., description="Tenant ID")
    org_id: str = Field(..., description="Organization ID")
    inventory_id: str = Field(..., description="Emissions inventory ID")
    scope3_total_tco2e: Decimal = Field(default=Decimal("0"), ge=Decimal("0"))
    total_s1_s2_s3_tco2e: Decimal = Field(default=Decimal("0"), ge=Decimal("0"))
    scope3_pct_of_total: Decimal = Field(
        default=Decimal("0"), ge=Decimal("0"), le=Decimal("100"),
    )
    scope3_target_required: bool = Field(
        default=False, description="Whether Scope 3 target is required (>40%)",
    )
    category_breakdown: List[CategoryBreakdown] = Field(default_factory=list)
    material_categories: List[int] = Field(
        default_factory=list, description="Category numbers individually exceeding 1%",
    )
    required_coverage_tco2e: Decimal = Field(
        default=Decimal("0"), ge=Decimal("0"),
        description="Emissions that must be covered by target (67% near-term)",
    )
    current_coverage_tco2e: Decimal = Field(
        default=Decimal("0"), ge=Decimal("0"),
    )
    current_coverage_pct: Decimal = Field(
        default=Decimal("0"), ge=Decimal("0"), le=Decimal("100"),
    )
    coverage_sufficient: bool = Field(default=False)
    provenance_hash: str = Field(default="")
    screened_at: datetime = Field(default_factory=_now)

    def model_post_init(self, __context: Any) -> None:
        """Compute provenance hash."""
        if not self.provenance_hash:
            payload = (
                f"{self.org_id}:{self.scope3_total_tco2e}:"
                f"{self.scope3_pct_of_total}:{self.scope3_target_required}"
            )
            self.provenance_hash = _sha256(payload)


# ---------------------------------------------------------------------------
# FLAG Models
# ---------------------------------------------------------------------------

class CommodityData(BaseModel):
    """Commodity-level data for FLAG assessment."""

    commodity: str = Field(..., max_length=100, description="FLAG commodity name")
    base_intensity: Decimal = Field(
        default=Decimal("0"), ge=Decimal("0"),
        description="Base year intensity (tCO2e per tonne commodity)",
    )
    target_intensity: Decimal = Field(
        default=Decimal("0"), ge=Decimal("0"),
        description="Target year intensity (tCO2e per tonne commodity)",
    )
    pathway_type: str = Field(
        default="commodity", description="commodity or sector",
    )
    production_volume: Optional[Decimal] = Field(
        None, ge=Decimal("0"), description="Annual production volume (tonnes)",
    )
    total_emissions_tco2e: Optional[Decimal] = Field(None, ge=Decimal("0"))
    data_quality: str = Field(default="estimated")


class FLAGAssessment(BaseModel):
    """
    FLAG assessment per SBTi FLAG Guidance v1.0.

    Determines whether FLAG emissions trigger a separate FLAG target,
    evaluates commodity-level pathways, and checks deforestation commitments.
    """

    id: str = Field(default_factory=_new_id, description="FLAG assessment ID")
    tenant_id: str = Field(..., description="Tenant ID")
    org_id: str = Field(..., description="Organization ID")
    inventory_id: str = Field(..., description="Emissions inventory ID")
    flag_emissions_tco2e: Decimal = Field(
        default=Decimal("0"), ge=Decimal("0"),
    )
    total_emissions_tco2e: Decimal = Field(
        default=Decimal("0"), ge=Decimal("0"),
    )
    flag_pct_of_total: Decimal = Field(
        default=Decimal("0"), ge=Decimal("0"), le=Decimal("100"),
    )
    flag_target_required: bool = Field(
        default=False, description="Whether FLAG target is required (>=20%)",
    )
    commodity_data: List[CommodityData] = Field(default_factory=list)
    sector_pathway_rate: Decimal = Field(
        default=Decimal("3.03"), description="Sector-level annual reduction rate (%)",
    )
    deforestation_commitment: bool = Field(
        default=False, description="Commitment to zero deforestation by 2025",
    )
    deforestation_commitment_date: Optional[date] = Field(None)
    land_use_change_included: bool = Field(
        default=False, description="Whether LUC emissions are included",
    )
    notes: Optional[str] = Field(None, max_length=5000)
    provenance_hash: str = Field(default="")
    assessed_at: datetime = Field(default_factory=_now)

    def model_post_init(self, __context: Any) -> None:
        """Compute provenance hash."""
        if not self.provenance_hash:
            payload = (
                f"{self.org_id}:{self.flag_emissions_tco2e}:"
                f"{self.flag_pct_of_total}:{self.flag_target_required}"
            )
            self.provenance_hash = _sha256(payload)


# ---------------------------------------------------------------------------
# Sector Pathway Models
# ---------------------------------------------------------------------------

class SectorPathway(BaseModel):
    """
    Sector-specific intensity pathway for SDA target alignment.

    Contains the sector, intensity metric, base and target values,
    and annual data points for comparing company trajectory.
    """

    id: str = Field(default_factory=_new_id, description="Sector pathway ID")
    sector: SBTiSector = Field(..., description="SBTi sector")
    intensity_metric: str = Field(..., max_length=100, description="Intensity metric")
    intensity_unit: str = Field(default="", max_length=100)
    base_year: int = Field(..., ge=2015, le=2030)
    target_year: int = Field(..., ge=2030, le=2060)
    base_value: Decimal = Field(default=Decimal("0"), ge=Decimal("0"))
    target_value: Decimal = Field(default=Decimal("0"), ge=Decimal("0"))
    annual_points: Dict[int, Decimal] = Field(
        default_factory=dict,
        description="Year-by-year intensity values from the sector pathway",
    )
    source: str = Field(
        default="SBTi SDA", max_length=255, description="Pathway data source",
    )
    created_at: datetime = Field(default_factory=_now)


# ---------------------------------------------------------------------------
# Progress Tracking Models
# ---------------------------------------------------------------------------

class ProgressRecord(BaseModel):
    """
    Annual progress record against a target.

    Tracks actual emissions, pathway expected value, variance, and
    on-track assessment for each reporting year.
    """

    id: str = Field(default_factory=_new_id, description="Progress record ID")
    tenant_id: str = Field(..., description="Tenant ID")
    org_id: str = Field(..., description="Organization ID")
    target_id: str = Field(..., description="Target ID")
    year: int = Field(..., ge=2015, le=2100, description="Reporting year")
    actual_scope1_tco2e: Decimal = Field(default=Decimal("0"), ge=Decimal("0"))
    actual_scope2_tco2e: Decimal = Field(default=Decimal("0"), ge=Decimal("0"))
    actual_scope3_tco2e: Decimal = Field(default=Decimal("0"), ge=Decimal("0"))
    actual_total_tco2e: Decimal = Field(default=Decimal("0"), ge=Decimal("0"))
    actual_intensity_value: Optional[Decimal] = Field(None)
    pathway_expected_tco2e: Decimal = Field(
        default=Decimal("0"), ge=Decimal("0"),
        description="Expected emissions from calculated pathway",
    )
    pathway_expected_intensity: Optional[Decimal] = Field(None)
    variance_tco2e: Decimal = Field(
        default=Decimal("0"), description="Actual minus expected (negative = ahead)",
    )
    variance_pct: Decimal = Field(
        default=Decimal("0"), description="Variance as percentage of expected",
    )
    cumulative_reduction_pct: Decimal = Field(
        default=Decimal("0"), ge=Decimal("0"), le=Decimal("100"),
        description="Cumulative reduction from base year",
    )
    on_track: bool = Field(
        default=False, description="Whether emissions are at or below pathway",
    )
    data_quality: str = Field(default="estimated")
    provenance_hash: str = Field(default="")
    created_at: datetime = Field(default_factory=_now)

    def model_post_init(self, __context: Any) -> None:
        """Compute variance and provenance hash."""
        if self.variance_tco2e == Decimal("0") and self.pathway_expected_tco2e > 0:
            var = self.actual_total_tco2e - self.pathway_expected_tco2e
            object.__setattr__(self, "variance_tco2e", var)
            var_pct = (var / self.pathway_expected_tco2e) * 100
            object.__setattr__(self, "variance_pct", var_pct.quantize(Decimal("0.01")))
            object.__setattr__(self, "on_track", var <= Decimal("0"))

        if not self.provenance_hash:
            payload = (
                f"{self.target_id}:{self.year}:"
                f"{self.actual_total_tco2e}:{self.pathway_expected_tco2e}"
            )
            self.provenance_hash = _sha256(payload)


class ProgressSummary(BaseModel):
    """Summary of progress against a target across all tracked years."""

    target_id: str = Field(..., description="Target ID")
    base_year: int = Field(..., ge=2015, le=2030)
    target_year: int = Field(..., ge=2025, le=2060)
    base_emissions_tco2e: Decimal = Field(default=Decimal("0"), ge=Decimal("0"))
    latest_year: Optional[int] = Field(None)
    latest_emissions_tco2e: Optional[Decimal] = Field(None, ge=Decimal("0"))
    cumulative_reduction_pct: Decimal = Field(
        default=Decimal("0"), ge=Decimal("0"), le=Decimal("100"),
    )
    required_reduction_pct: Decimal = Field(
        default=Decimal("0"), ge=Decimal("0"), le=Decimal("100"),
    )
    current_annual_rate: Decimal = Field(
        default=Decimal("0"), description="Actual annualized rate achieved",
    )
    required_annual_rate: Decimal = Field(
        default=Decimal("0"), description="Required annual rate to meet target",
    )
    projected_target_year_emissions: Optional[Decimal] = Field(
        None, ge=Decimal("0"),
    )
    projected_achievement_pct: Decimal = Field(
        default=Decimal("0"), ge=Decimal("0"),
    )
    on_track: bool = Field(default=False)
    years_tracked: int = Field(default=0, ge=0)
    records: List[ProgressRecord] = Field(default_factory=list)
    computed_at: datetime = Field(default_factory=_now)


# ---------------------------------------------------------------------------
# Temperature Scoring Models
# ---------------------------------------------------------------------------

class ScopeTemperatureScore(BaseModel):
    """Temperature score for a specific scope."""

    scope: TargetScope = Field(..., description="Scope")
    score: Decimal = Field(
        default=Decimal("3.2"), ge=Decimal("1.0"), le=Decimal("6.0"),
        description="Temperature score in degrees C",
    )
    has_target: bool = Field(default=False)
    target_ambition: Optional[AmbitionLevel] = Field(None)
    confidence: Decimal = Field(
        default=Decimal("0.5"), ge=Decimal("0"), le=Decimal("1"),
    )


class TemperatureScore(BaseModel):
    """
    Company-level temperature score based on SBTi Temperature Rating methodology.

    Provides overall score, short-term and long-term breakdowns, and
    per-scope details.
    """

    id: str = Field(default_factory=_new_id, description="Temperature score ID")
    tenant_id: str = Field(..., description="Tenant ID")
    org_id: str = Field(..., description="Organization ID")
    company_score: Decimal = Field(
        default=Decimal("3.2"), ge=Decimal("1.0"), le=Decimal("6.0"),
        description="Overall company temperature score",
    )
    short_term_score: Decimal = Field(
        default=Decimal("3.2"), ge=Decimal("1.0"), le=Decimal("6.0"),
    )
    mid_term_score: Decimal = Field(
        default=Decimal("3.2"), ge=Decimal("1.0"), le=Decimal("6.0"),
    )
    long_term_score: Decimal = Field(
        default=Decimal("3.2"), ge=Decimal("1.0"), le=Decimal("6.0"),
    )
    scope_breakdown: List[ScopeTemperatureScore] = Field(
        default_factory=list, description="Per-scope temperature scores",
    )
    validation_status: ValidationStatus = Field(
        default=ValidationStatus.COMMITMENT_LETTER,
    )
    methodology_version: str = Field(default="v2.0", max_length=20)
    provenance_hash: str = Field(default="")
    scored_at: datetime = Field(default_factory=_now)

    def model_post_init(self, __context: Any) -> None:
        """Compute provenance hash."""
        if not self.provenance_hash:
            payload = (
                f"{self.org_id}:{self.company_score}:"
                f"{self.short_term_score}:{self.long_term_score}"
            )
            self.provenance_hash = _sha256(payload)


class PortfolioTemperature(BaseModel):
    """
    Portfolio-level temperature score (for financial institutions).

    Aggregates company-level scores weighted by investment exposure.
    """

    id: str = Field(default_factory=_new_id, description="Portfolio temperature ID")
    tenant_id: str = Field(..., description="Tenant ID")
    org_id: str = Field(..., description="Organization ID (FI)")
    weighted_score: Decimal = Field(
        default=Decimal("3.2"), ge=Decimal("1.0"), le=Decimal("6.0"),
    )
    holdings_count: int = Field(default=0, ge=0)
    holdings_with_targets: int = Field(default=0, ge=0)
    coverage_pct: Decimal = Field(
        default=Decimal("0"), ge=Decimal("0"), le=Decimal("100"),
    )
    short_term_score: Decimal = Field(
        default=Decimal("3.2"), ge=Decimal("1.0"), le=Decimal("6.0"),
    )
    long_term_score: Decimal = Field(
        default=Decimal("3.2"), ge=Decimal("1.0"), le=Decimal("6.0"),
    )
    alignment_1_5c_pct: Decimal = Field(
        default=Decimal("0"), ge=Decimal("0"), le=Decimal("100"),
    )
    alignment_wb2c_pct: Decimal = Field(
        default=Decimal("0"), ge=Decimal("0"), le=Decimal("100"),
    )
    provenance_hash: str = Field(default="")
    scored_at: datetime = Field(default_factory=_now)

    def model_post_init(self, __context: Any) -> None:
        """Compute provenance hash."""
        if not self.provenance_hash:
            payload = (
                f"{self.org_id}:{self.weighted_score}:"
                f"{self.holdings_count}:{self.coverage_pct}"
            )
            self.provenance_hash = _sha256(payload)


# ---------------------------------------------------------------------------
# Recalculation Models
# ---------------------------------------------------------------------------

class Recalculation(BaseModel):
    """
    Base-year recalculation record per SBTi criteria.

    Tracks the trigger event, original and recalculated emissions,
    percentage change, and whether revalidation is required.
    """

    id: str = Field(default_factory=_new_id, description="Recalculation ID")
    tenant_id: str = Field(..., description="Tenant ID")
    org_id: str = Field(..., description="Organization ID")
    target_id: str = Field(..., description="Affected target ID")
    trigger: RecalculationTrigger = Field(..., description="Recalculation trigger event")
    trigger_description: str = Field(
        default="", max_length=2000, description="Description of the trigger event",
    )
    trigger_date: date = Field(
        default_factory=lambda: date.today(), description="Date trigger occurred",
    )
    original_base_year_emissions_tco2e: Decimal = Field(
        default=Decimal("0"), ge=Decimal("0"),
    )
    recalculated_base_year_emissions_tco2e: Decimal = Field(
        default=Decimal("0"), ge=Decimal("0"),
    )
    pct_change: Decimal = Field(
        default=Decimal("0"), description="Percentage change from original",
    )
    exceeds_threshold: bool = Field(
        default=False, description="Whether change exceeds 5% threshold",
    )
    revalidation_required: bool = Field(
        default=False, description="Whether target revalidation is needed",
    )
    revalidation_status: Optional[str] = Field(
        None, description="pending, in_progress, completed",
    )
    methodology_used: Optional[str] = Field(None, max_length=500)
    approved_by: Optional[str] = Field(None, max_length=255)
    approved_at: Optional[datetime] = Field(None)
    notes: Optional[str] = Field(None, max_length=5000)
    provenance_hash: str = Field(default="")
    created_at: datetime = Field(default_factory=_now)

    def model_post_init(self, __context: Any) -> None:
        """Compute percentage change and provenance hash."""
        if (
            self.pct_change == Decimal("0")
            and self.original_base_year_emissions_tco2e > Decimal("0")
        ):
            change = abs(
                self.recalculated_base_year_emissions_tco2e
                - self.original_base_year_emissions_tco2e
            )
            pct = (change / self.original_base_year_emissions_tco2e) * 100
            object.__setattr__(self, "pct_change", pct.quantize(Decimal("0.01")))
            object.__setattr__(self, "exceeds_threshold", pct >= Decimal("5"))
            object.__setattr__(self, "revalidation_required", pct >= Decimal("5"))

        if not self.provenance_hash:
            payload = (
                f"{self.target_id}:{self.trigger}:"
                f"{self.original_base_year_emissions_tco2e}:"
                f"{self.recalculated_base_year_emissions_tco2e}"
            )
            self.provenance_hash = _sha256(payload)


# ---------------------------------------------------------------------------
# Five-Year Review Models
# ---------------------------------------------------------------------------

class FiveYearReview(BaseModel):
    """
    Five-year review cycle record per SBTi requirements.

    Tracks review trigger dates, deadlines, status, and readiness assessment.
    """

    id: str = Field(default_factory=_new_id, description="Review ID")
    tenant_id: str = Field(..., description="Tenant ID")
    org_id: str = Field(..., description="Organization ID")
    target_id: str = Field(..., description="Target ID under review")
    cycle_number: int = Field(default=1, ge=1, description="Review cycle number")
    trigger_date: date = Field(
        ..., description="Date the review was triggered (5 years from validation)",
    )
    deadline: date = Field(..., description="Deadline for completing the review")
    outcome: ReviewOutcome = Field(default=ReviewOutcome.PENDING)
    current_pathway_aligned: bool = Field(
        default=False, description="Whether current target is still aligned",
    )
    updated_ambition_level: Optional[AmbitionLevel] = Field(None)
    updated_reduction_pct: Optional[Decimal] = Field(None, ge=Decimal("0"), le=Decimal("100"))
    readiness_assessment: Optional[str] = Field(
        None, max_length=5000, description="Narrative assessment of readiness",
    )
    reviewer: Optional[str] = Field(None, max_length=255)
    completed_at: Optional[datetime] = Field(None)
    notes: Optional[str] = Field(None, max_length=5000)
    created_at: datetime = Field(default_factory=_now)
    updated_at: datetime = Field(default_factory=_now)


# ---------------------------------------------------------------------------
# Financial Institution Models
# ---------------------------------------------------------------------------

class PortfolioHolding(BaseModel):
    """A single holding within an FI portfolio for financed emissions tracking."""

    id: str = Field(default_factory=_new_id, description="Holding ID")
    company_name: str = Field(..., min_length=1, max_length=500)
    company_id: Optional[str] = Field(None, description="External company ID (LEI, ISIN)")
    asset_class: FIAssetClass = Field(..., description="PCAF asset class")
    exposure_usd: Decimal = Field(default=Decimal("0"), ge=Decimal("0"))
    financed_emissions_tco2e: Decimal = Field(
        default=Decimal("0"), ge=Decimal("0"),
    )
    attribution_factor: Decimal = Field(
        default=Decimal("0"), ge=Decimal("0"), le=Decimal("1"),
        description="Attribution factor (e.g. outstanding / EVIC)",
    )
    evic_usd: Optional[Decimal] = Field(None, ge=Decimal("0"))
    pcaf_dq: PCAFDataQuality = Field(default=PCAFDataQuality.DQ_5)
    has_sbti_target: bool = Field(default=False)
    validation_status: Optional[ValidationStatus] = Field(None)
    temperature_score: Optional[Decimal] = Field(
        None, ge=Decimal("1.0"), le=Decimal("6.0"),
    )
    sector: Optional[str] = Field(None, max_length=100)
    country: Optional[str] = Field(None, max_length=3)


class EngagementRecord(BaseModel):
    """Record of engagement with an investee company on target setting."""

    id: str = Field(default_factory=_new_id, description="Engagement record ID")
    holding_id: str = Field(..., description="Associated holding ID")
    investee_name: str = Field(..., min_length=1, max_length=500)
    engagement_type: str = Field(
        default="direct",
        description="direct, collaborative, escalation, proxy_voting",
    )
    status: str = Field(
        default="in_progress",
        description="planned, in_progress, successful, unsuccessful",
    )
    target_date: Optional[date] = Field(
        None, description="Target date for investee to set SBTi target",
    )
    milestones: List[str] = Field(
        default_factory=list, description="Engagement milestones achieved",
    )
    notes: Optional[str] = Field(None, max_length=5000)
    created_at: datetime = Field(default_factory=_now)
    updated_at: datetime = Field(default_factory=_now)


class FIPortfolio(BaseModel):
    """
    Financial institution portfolio for financed emissions and target tracking.

    Covers total financed emissions, portfolio coverage, temperature scoring,
    and target coverage by year (for SBTi-FI portfolio coverage approach).
    """

    id: str = Field(default_factory=_new_id, description="Portfolio ID")
    tenant_id: str = Field(..., description="Tenant ID")
    org_id: str = Field(..., description="Organization ID (FI)")
    name: str = Field(default="Main Portfolio", max_length=255)
    year: int = Field(..., ge=2020, le=2050, description="Reporting year")
    fi_target_type: Optional[FITargetType] = Field(
        None, description="SBTi FI target approach",
    )
    total_exposure_usd: Decimal = Field(default=Decimal("0"), ge=Decimal("0"))
    total_financed_emissions_tco2e: Decimal = Field(
        default=Decimal("0"), ge=Decimal("0"),
    )
    coverage_pct: Decimal = Field(
        default=Decimal("0"), ge=Decimal("0"), le=Decimal("100"),
    )
    holdings: List[PortfolioHolding] = Field(default_factory=list)
    holdings_count: int = Field(default=0, ge=0)
    holdings_with_sbti: int = Field(default=0, ge=0)
    sbti_coverage_pct: Decimal = Field(
        default=Decimal("0"), ge=Decimal("0"), le=Decimal("100"),
    )
    target_coverage_by_year: Dict[int, Decimal] = Field(
        default_factory=dict,
        description="SBTi target coverage (%) by future year",
    )
    temperature_score: Optional[Decimal] = Field(
        None, ge=Decimal("1.0"), le=Decimal("6.0"),
    )
    waci: Optional[Decimal] = Field(
        None, ge=Decimal("0"),
        description="Weighted Average Carbon Intensity (tCO2e/$M revenue)",
    )
    engagement_records: List[EngagementRecord] = Field(default_factory=list)
    asset_class_breakdown: Dict[str, Decimal] = Field(
        default_factory=dict,
    )
    avg_pcaf_dq: Optional[Decimal] = Field(
        None, ge=Decimal("1"), le=Decimal("5"),
    )
    provenance_hash: str = Field(default="")
    created_at: datetime = Field(default_factory=_now)
    updated_at: datetime = Field(default_factory=_now)

    def model_post_init(self, __context: Any) -> None:
        """Compute derived fields and provenance hash."""
        if self.holdings and self.holdings_count == 0:
            object.__setattr__(self, "holdings_count", len(self.holdings))
            object.__setattr__(
                self, "holdings_with_sbti",
                sum(1 for h in self.holdings if h.has_sbti_target),
            )
        if not self.provenance_hash:
            payload = (
                f"{self.org_id}:{self.year}:"
                f"{self.total_financed_emissions_tco2e}:{self.coverage_pct}"
            )
            self.provenance_hash = _sha256(payload)


# ---------------------------------------------------------------------------
# Framework Alignment Models
# ---------------------------------------------------------------------------

class AlignmentItem(BaseModel):
    """Alignment mapping between an SBTi requirement and a framework requirement."""

    sbti_reference: str = Field(..., description="SBTi criterion or section")
    framework_reference: str = Field(..., description="External framework reference")
    description: str = Field(default="", max_length=500)
    status: str = Field(
        default="aligned",
        description="aligned, partially_aligned, gap, not_applicable",
    )
    notes: Optional[str] = Field(None, max_length=1000)


class FrameworkMapping(BaseModel):
    """
    Cross-framework alignment mapping for a specific external framework.

    Shows how SBTi criteria and data align with CDP, TCFD, CSRD, etc.
    """

    id: str = Field(default_factory=_new_id, description="Mapping ID")
    tenant_id: str = Field(..., description="Tenant ID")
    org_id: str = Field(..., description="Organization ID")
    framework: str = Field(..., max_length=50, description="External framework key")
    framework_name: str = Field(default="", max_length=255)
    alignment_items: List[AlignmentItem] = Field(default_factory=list)
    total_items: int = Field(default=0, ge=0)
    aligned_count: int = Field(default=0, ge=0)
    coverage_pct: Decimal = Field(
        default=Decimal("0"), ge=Decimal("0"), le=Decimal("100"),
    )
    mapped_at: datetime = Field(default_factory=_now)

    def model_post_init(self, __context: Any) -> None:
        """Compute alignment coverage."""
        if self.alignment_items and self.total_items == 0:
            total = len(self.alignment_items)
            aligned = sum(1 for a in self.alignment_items if a.status == "aligned")
            object.__setattr__(self, "total_items", total)
            object.__setattr__(self, "aligned_count", aligned)
            if total > 0:
                pct = Decimal(str(aligned / total * 100))
                object.__setattr__(self, "coverage_pct", pct.quantize(Decimal("0.01")))


# ---------------------------------------------------------------------------
# Report Models
# ---------------------------------------------------------------------------

class Report(BaseModel):
    """
    Generated report from the SBTi platform.

    Covers validation reports, progress reports, submission packages,
    and multi-framework alignment reports.
    """

    id: str = Field(default_factory=_new_id, description="Report ID")
    tenant_id: str = Field(..., description="Tenant ID")
    org_id: str = Field(..., description="Organization ID")
    report_type: str = Field(
        ..., description="validation, progress, submission, framework_alignment, gap_analysis",
    )
    format: ReportFormat = Field(default=ReportFormat.PDF)
    title: str = Field(default="", max_length=500)
    description: Optional[str] = Field(None, max_length=2000)
    content: Optional[Dict[str, Any]] = Field(None)
    file_path: Optional[str] = Field(None, description="Storage path")
    file_size_bytes: Optional[int] = Field(None, ge=0)
    target_ids: List[str] = Field(default_factory=list)
    generated_by: Optional[str] = Field(None, max_length=255)
    generated_at: datetime = Field(default_factory=_now)
    provenance_hash: str = Field(default="")

    def model_post_init(self, __context: Any) -> None:
        """Compute provenance hash."""
        if not self.provenance_hash:
            payload = (
                f"{self.org_id}:{self.report_type}:"
                f"{self.format}:{self.generated_at}"
            )
            self.provenance_hash = _sha256(payload)


class SubmissionForm(BaseModel):
    """
    SBTi target submission form containing all required fields.

    Captures organizational data, target details, supporting evidence,
    and declaration for SBTi validation review.
    """

    id: str = Field(default_factory=_new_id, description="Submission ID")
    tenant_id: str = Field(..., description="Tenant ID")
    org_id: str = Field(..., description="Organization ID")
    target_ids: List[str] = Field(..., description="Target IDs being submitted")
    company_name: str = Field(..., min_length=1, max_length=500)
    sector: SBTiSector = Field(...)
    country: str = Field(..., min_length=2, max_length=3)
    contact_name: str = Field(..., min_length=1, max_length=255)
    contact_email: str = Field(..., max_length=255)
    contact_phone: Optional[str] = Field(None, max_length=50)
    base_year: int = Field(..., ge=2015, le=2030)
    base_year_scope1_tco2e: Decimal = Field(default=Decimal("0"), ge=Decimal("0"))
    base_year_scope2_tco2e: Decimal = Field(default=Decimal("0"), ge=Decimal("0"))
    base_year_scope3_tco2e: Decimal = Field(default=Decimal("0"), ge=Decimal("0"))
    scope2_method: str = Field(default="market", description="location or market")
    scope3_screening_complete: bool = Field(default=False)
    scope3_target_required: bool = Field(default=False)
    flag_target_required: bool = Field(default=False)
    near_term_target_description: Optional[str] = Field(None, max_length=5000)
    long_term_target_description: Optional[str] = Field(None, max_length=5000)
    net_zero_commitment: bool = Field(default=False)
    transition_plan_available: bool = Field(default=False)
    verification_status: str = Field(
        default="not_verified", description="not_verified, limited, reasonable",
    )
    recalculation_policy_defined: bool = Field(default=False)
    annual_disclosure_committed: bool = Field(default=False)
    declaration_signed: bool = Field(default=False)
    declaration_signer: Optional[str] = Field(None, max_length=255)
    declaration_date: Optional[date] = Field(None)
    supporting_documents: List[str] = Field(default_factory=list)
    status: str = Field(
        default="draft", description="draft, validated, submitted, accepted, returned",
    )
    submitted_at: Optional[datetime] = Field(None)
    sbti_reference_number: Optional[str] = Field(None, max_length=50)
    provenance_hash: str = Field(default="")
    created_at: datetime = Field(default_factory=_now)
    updated_at: datetime = Field(default_factory=_now)

    def model_post_init(self, __context: Any) -> None:
        """Compute provenance hash."""
        if not self.provenance_hash:
            payload = (
                f"{self.org_id}:{self.base_year}:{self.company_name}:"
                f"{self.status}:{self.created_at}"
            )
            self.provenance_hash = _sha256(payload)


# ---------------------------------------------------------------------------
# Gap Analysis Models
# ---------------------------------------------------------------------------

class GapItem(BaseModel):
    """A single identified gap in SBTi target readiness."""

    id: str = Field(default_factory=_new_id, description="Gap item ID")
    criterion_id: str = Field(
        default="", max_length=50,
        description="SBTi criterion ID if applicable (e.g. C5, NZ-C3)",
    )
    gap_category: GapCategory = Field(default=GapCategory.DATA)
    severity: GapSeverity = Field(default=GapSeverity.MEDIUM)
    description: str = Field(default="", max_length=2000)
    current_status: Optional[str] = Field(None, max_length=500)
    required_status: Optional[str] = Field(None, max_length=500)
    recommendation: str = Field(default="", max_length=2000)
    estimated_effort: str = Field(
        default="medium", description="low, medium, high",
    )
    estimated_days: Optional[int] = Field(None, ge=0)
    responsible_team: Optional[str] = Field(None, max_length=255)
    resolved: bool = Field(default=False)
    resolved_at: Optional[datetime] = Field(None)
    created_at: datetime = Field(default_factory=_now)


class ActionPlanItem(BaseModel):
    """A specific action in the gap closure action plan."""

    id: str = Field(default_factory=_new_id, description="Action ID")
    gap_id: str = Field(..., description="Associated gap item ID")
    title: str = Field(..., min_length=1, max_length=255)
    description: str = Field(default="", max_length=2000)
    priority: int = Field(default=1, ge=1, le=5, description="Priority (1=highest)")
    owner: Optional[str] = Field(None, max_length=255)
    due_date: Optional[date] = Field(None)
    status: str = Field(default="pending", description="pending, in_progress, completed")
    completed_at: Optional[datetime] = Field(None)


class GapAssessment(BaseModel):
    """
    Complete gap analysis for SBTi target readiness.

    Identifies all gaps, computes overall readiness, and generates
    a prioritized action plan for achieving submission readiness.
    """

    id: str = Field(default_factory=_new_id, description="Gap assessment ID")
    tenant_id: str = Field(..., description="Tenant ID")
    org_id: str = Field(..., description="Organization ID")
    assessment_type: str = Field(
        default="near_term", description="near_term, net_zero, combined",
    )
    gaps: List[GapItem] = Field(default_factory=list)
    total_gaps: int = Field(default=0, ge=0)
    critical_gaps: int = Field(default=0, ge=0)
    high_gaps: int = Field(default=0, ge=0)
    medium_gaps: int = Field(default=0, ge=0)
    low_gaps: int = Field(default=0, ge=0)
    overall_readiness_pct: Decimal = Field(
        default=Decimal("0"), ge=Decimal("0"), le=Decimal("100"),
    )
    action_plan: List[ActionPlanItem] = Field(default_factory=list)
    estimated_total_days: Optional[int] = Field(None, ge=0)
    assessed_at: datetime = Field(default_factory=_now)
    provenance_hash: str = Field(default="")

    def model_post_init(self, __context: Any) -> None:
        """Compute gap counts and provenance hash."""
        if self.gaps and self.total_gaps == 0:
            object.__setattr__(self, "total_gaps", len(self.gaps))
            object.__setattr__(
                self, "critical_gaps",
                sum(1 for g in self.gaps if g.severity == GapSeverity.CRITICAL),
            )
            object.__setattr__(
                self, "high_gaps",
                sum(1 for g in self.gaps if g.severity == GapSeverity.HIGH),
            )
            object.__setattr__(
                self, "medium_gaps",
                sum(1 for g in self.gaps if g.severity == GapSeverity.MEDIUM),
            )
            object.__setattr__(
                self, "low_gaps",
                sum(1 for g in self.gaps if g.severity == GapSeverity.LOW),
            )
        if not self.provenance_hash:
            payload = (
                f"{self.org_id}:{self.assessment_type}:"
                f"{self.total_gaps}:{self.overall_readiness_pct}"
            )
            self.provenance_hash = _sha256(payload)


# ---------------------------------------------------------------------------
# MRV Data Connector Models
# ---------------------------------------------------------------------------

class MRVDataPoint(BaseModel):
    """A data point retrieved from an MRV agent for SBTi target tracking."""

    agent_id: str = Field(..., description="MRV agent ID (e.g. MRV-001)")
    agent_name: str = Field(default="")
    scope: str = Field(default="", description="scope_1, scope_2, scope_3")
    scope3_category: Optional[int] = Field(None, ge=1, le=15)
    emissions_tco2e: Decimal = Field(default=Decimal("0"), ge=Decimal("0"))
    unit: str = Field(default="tCO2e")
    methodology: Optional[str] = Field(None)
    data_quality_score: float = Field(default=0.0, ge=0.0, le=100.0)
    reporting_year: int = Field(default=2026, ge=2020, le=2050)
    data_timestamp: datetime = Field(default_factory=_now)
    is_fresh: bool = Field(default=True)
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    provenance_hash: str = Field(default="")

    def model_post_init(self, __context: Any) -> None:
        """Compute provenance hash."""
        if not self.provenance_hash:
            payload = (
                f"{self.agent_id}:{self.emissions_tco2e}:"
                f"{self.reporting_year}:{self.data_timestamp}"
            )
            self.provenance_hash = _sha256(payload)


class MRVPopulationResult(BaseModel):
    """Result of auto-populating emissions data from MRV agents."""

    inventory_id: str = Field(...)
    data_points: List[MRVDataPoint] = Field(default_factory=list)
    agents_queried: int = Field(default=0, ge=0)
    agents_responded: int = Field(default=0, ge=0)
    scope1_total_tco2e: Decimal = Field(default=Decimal("0"), ge=Decimal("0"))
    scope2_location_tco2e: Decimal = Field(default=Decimal("0"), ge=Decimal("0"))
    scope2_market_tco2e: Decimal = Field(default=Decimal("0"), ge=Decimal("0"))
    scope3_total_tco2e: Decimal = Field(default=Decimal("0"), ge=Decimal("0"))
    scope3_by_category: Dict[int, Decimal] = Field(default_factory=dict)
    flag_tco2e: Decimal = Field(default=Decimal("0"), ge=Decimal("0"))
    data_freshness_valid: bool = Field(default=True)
    populated_at: datetime = Field(default_factory=_now)
    provenance_hash: str = Field(default="")

    def model_post_init(self, __context: Any) -> None:
        """Compute provenance hash."""
        if not self.provenance_hash:
            payload = (
                f"{self.inventory_id}:{self.scope1_total_tco2e}:"
                f"{self.scope2_market_tco2e}:{self.scope3_total_tco2e}"
            )
            self.provenance_hash = _sha256(payload)


# ---------------------------------------------------------------------------
# Dashboard Models
# ---------------------------------------------------------------------------

class DashboardAlert(BaseModel):
    """An alert surfaced on the SBTi dashboard."""

    id: str = Field(default_factory=_new_id)
    severity: GapSeverity = Field(default=GapSeverity.MEDIUM)
    title: str = Field(..., min_length=1)
    message: str = Field(default="")
    notification_type: Optional[NotificationType] = Field(None)
    created_at: datetime = Field(default_factory=_now)
    dismissed: bool = Field(default=False)


class DashboardMetrics(BaseModel):
    """
    Aggregated dashboard metrics for the SBTi platform.

    Provides target status, validation readiness, progress summary,
    upcoming review deadlines, and key alerts.
    """

    org_id: str = Field(...)
    validation_status: ValidationStatus = Field(
        default=ValidationStatus.COMMITMENT_LETTER,
    )
    total_targets: int = Field(default=0, ge=0)
    validated_targets: int = Field(default=0, ge=0)
    near_term_readiness_pct: Decimal = Field(
        default=Decimal("0"), ge=Decimal("0"), le=Decimal("100"),
    )
    net_zero_readiness_pct: Decimal = Field(
        default=Decimal("0"), ge=Decimal("0"), le=Decimal("100"),
    )
    scope3_target_required: bool = Field(default=False)
    flag_target_required: bool = Field(default=False)
    latest_scope1_tco2e: Optional[Decimal] = Field(None, ge=Decimal("0"))
    latest_scope2_tco2e: Optional[Decimal] = Field(None, ge=Decimal("0"))
    latest_scope3_tco2e: Optional[Decimal] = Field(None, ge=Decimal("0"))
    base_year_total_tco2e: Optional[Decimal] = Field(None, ge=Decimal("0"))
    cumulative_reduction_pct: Decimal = Field(
        default=Decimal("0"), ge=Decimal("0"), le=Decimal("100"),
    )
    on_track: bool = Field(default=False)
    temperature_score: Optional[Decimal] = Field(
        None, ge=Decimal("1.0"), le=Decimal("6.0"),
    )
    next_review_date: Optional[date] = Field(None)
    days_until_review: Optional[int] = Field(None)
    submission_deadline: Optional[date] = Field(None)
    days_until_submission: Optional[int] = Field(None)
    gap_summary: Dict[str, int] = Field(default_factory=dict)
    alerts: List[DashboardAlert] = Field(default_factory=list)
    computed_at: datetime = Field(default_factory=_now)


# ---------------------------------------------------------------------------
# Request / Response API Models
# ---------------------------------------------------------------------------

class CreateOrganizationRequest(BaseModel):
    """Request to register a new organization on the SBTi platform."""

    name: str = Field(..., min_length=1, max_length=500)
    sector: SBTiSector = Field(default=SBTiSector.GENERAL)
    country: str = Field(..., min_length=2, max_length=3)
    isic_code: Optional[str] = Field(None, max_length=10)
    nace_code: Optional[str] = Field(None, max_length=10)
    naics_code: Optional[str] = Field(None, max_length=10)
    oecd_status: Optional[str] = Field(None, max_length=20)
    description: Optional[str] = Field(None, max_length=2000)
    contact_person: Optional[str] = Field(None, max_length=255)
    contact_email: Optional[str] = Field(None, max_length=255)
    employee_count: Optional[int] = Field(None, ge=0)
    annual_revenue_usd: Optional[Decimal] = Field(None, ge=Decimal("0"))
    is_financial_institution: bool = Field(default=False)
    commitment_date: Optional[date] = Field(None)


class UpdateOrganizationRequest(BaseModel):
    """Request to update an existing organization."""

    name: Optional[str] = Field(None, min_length=1, max_length=500)
    sector: Optional[SBTiSector] = Field(None)
    country: Optional[str] = Field(None, min_length=2, max_length=3)
    isic_code: Optional[str] = Field(None, max_length=10)
    nace_code: Optional[str] = Field(None, max_length=10)
    naics_code: Optional[str] = Field(None, max_length=10)
    oecd_status: Optional[str] = Field(None, max_length=20)
    contact_person: Optional[str] = Field(None, max_length=255)
    contact_email: Optional[str] = Field(None, max_length=255)


class CreateInventoryRequest(BaseModel):
    """Request to create or update an emissions inventory."""

    year: int = Field(..., ge=2010, le=2100)
    is_base_year: bool = Field(default=False)
    scope1_tco2e: Decimal = Field(default=Decimal("0"), ge=Decimal("0"))
    scope2_location_tco2e: Decimal = Field(default=Decimal("0"), ge=Decimal("0"))
    scope2_market_tco2e: Decimal = Field(default=Decimal("0"), ge=Decimal("0"))
    scope3_total_tco2e: Decimal = Field(default=Decimal("0"), ge=Decimal("0"))
    scope3_categories: Optional[List[Scope3CategoryEmissions]] = Field(None)
    flag_tco2e: Decimal = Field(default=Decimal("0"), ge=Decimal("0"))
    bioenergy_co2_tco2e: Decimal = Field(default=Decimal("0"), ge=Decimal("0"))
    data_quality_overall: str = Field(default="estimated")


class CreateTargetRequest(BaseModel):
    """Request to create a new SBTi target."""

    name: Optional[str] = Field(None, max_length=500)
    target_type: TargetType = Field(...)
    scope: TargetScope = Field(...)
    method: TargetMethod = Field(default=TargetMethod.ABSOLUTE_CONTRACTION)
    ambition_level: AmbitionLevel = Field(default=AmbitionLevel.ONE_POINT_FIVE_C)
    base_year: int = Field(..., ge=2015, le=2030)
    target_year: int = Field(..., ge=2025, le=2060)
    reduction_pct: Decimal = Field(..., ge=Decimal("0"), le=Decimal("100"))
    coverage_pct: Decimal = Field(default=Decimal("95"), ge=Decimal("0"), le=Decimal("100"))
    is_intensity_target: bool = Field(default=False)
    intensity_metric: Optional[str] = Field(None, max_length=100)
    base_intensity_value: Optional[Decimal] = Field(None)
    target_intensity_value: Optional[Decimal] = Field(None)
    is_flag_target: bool = Field(default=False)
    flag_commodity: Optional[str] = Field(None, max_length=100)
    deforestation_commitment: bool = Field(default=False)


class UpdateTargetRequest(BaseModel):
    """Request to update an existing target."""

    name: Optional[str] = Field(None, max_length=500)
    reduction_pct: Optional[Decimal] = Field(None, ge=Decimal("0"), le=Decimal("100"))
    coverage_pct: Optional[Decimal] = Field(None, ge=Decimal("0"), le=Decimal("100"))
    validation_status: Optional[ValidationStatus] = Field(None)
    notes: Optional[str] = Field(None, max_length=5000)


class CalculatePathwayRequest(BaseModel):
    """Request to calculate an emissions pathway for a target."""

    target_id: str = Field(...)
    method: Optional[TargetMethod] = Field(None)
    include_milestones: bool = Field(default=True)
    granularity: str = Field(default="annual", description="annual or five_year")


class RunValidationRequest(BaseModel):
    """Request to run SBTi criteria validation."""

    target_ids: List[str] = Field(...)
    validation_type: str = Field(
        default="near_term", description="near_term, net_zero, combined",
    )
    include_recommendations: bool = Field(default=True)


class Scope3ScreeningRequest(BaseModel):
    """Request to run Scope 3 screening assessment."""

    inventory_id: str = Field(...)
    include_materiality: bool = Field(default=True)


class FLAGAssessmentRequest(BaseModel):
    """Request to run FLAG assessment."""

    inventory_id: str = Field(...)
    commodity_data: Optional[List[CommodityData]] = Field(None)
    include_deforestation_check: bool = Field(default=True)


class ProgressRecordRequest(BaseModel):
    """Request to record annual progress against a target."""

    target_id: str = Field(...)
    year: int = Field(..., ge=2015, le=2100)
    actual_scope1_tco2e: Decimal = Field(default=Decimal("0"), ge=Decimal("0"))
    actual_scope2_tco2e: Decimal = Field(default=Decimal("0"), ge=Decimal("0"))
    actual_scope3_tco2e: Decimal = Field(default=Decimal("0"), ge=Decimal("0"))
    actual_intensity_value: Optional[Decimal] = Field(None)
    data_quality: str = Field(default="estimated")


class TemperatureScoreRequest(BaseModel):
    """Request to calculate temperature score for an organization."""

    include_portfolio: bool = Field(default=False)
    methodology_version: str = Field(default="v2.0")


class RecalculationRequest(BaseModel):
    """Request to initiate a base-year recalculation."""

    target_id: str = Field(...)
    trigger: RecalculationTrigger = Field(...)
    trigger_description: str = Field(..., min_length=1, max_length=2000)
    trigger_date: Optional[date] = Field(None)
    recalculated_emissions_tco2e: Decimal = Field(..., ge=Decimal("0"))
    methodology_used: Optional[str] = Field(None, max_length=500)


class FIPortfolioRequest(BaseModel):
    """Request to create or update an FI portfolio."""

    name: str = Field(default="Main Portfolio", max_length=255)
    year: int = Field(..., ge=2020, le=2050)
    fi_target_type: Optional[FITargetType] = Field(None)
    holdings: Optional[List[PortfolioHolding]] = Field(None)
    target_coverage_by_year: Optional[Dict[int, Decimal]] = Field(None)


class GenerateReportRequest(BaseModel):
    """Request to generate a report."""

    report_type: str = Field(
        default="validation",
        description="validation, progress, submission, framework_alignment, gap_analysis",
    )
    format: ReportFormat = Field(default=ReportFormat.PDF)
    target_ids: Optional[List[str]] = Field(None)
    include_pathway: bool = Field(default=True)
    include_progress: bool = Field(default=True)
    include_gap_analysis: bool = Field(default=False)
    include_framework_mapping: bool = Field(default=False)


class RunGapAnalysisRequest(BaseModel):
    """Request to run gap analysis."""

    assessment_type: str = Field(
        default="near_term", description="near_term, net_zero, combined",
    )
    target_ids: Optional[List[str]] = Field(None)
    include_action_plan: bool = Field(default=True)


class FrameworkMappingRequest(BaseModel):
    """Request to generate framework alignment mapping."""

    framework: str = Field(..., max_length=50, description="Framework key (cdp, tcfd, csrd, etc.)")
    include_gap_analysis: bool = Field(default=False)


class FiveYearReviewRequest(BaseModel):
    """Request to initiate or complete a five-year review."""

    target_id: str = Field(...)
    updated_ambition_level: Optional[AmbitionLevel] = Field(None)
    updated_reduction_pct: Optional[Decimal] = Field(None, ge=Decimal("0"), le=Decimal("100"))
    readiness_assessment: Optional[str] = Field(None, max_length=5000)


class PopulateFromMRVRequest(BaseModel):
    """Request to populate emissions data from MRV agents."""

    inventory_id: str = Field(...)
    year: int = Field(..., ge=2020, le=2050)
    agents: Optional[List[str]] = Field(None)
    overwrite_existing: bool = Field(default=False)


class UpdateSettingsRequest(BaseModel):
    """Request to update platform configuration."""

    default_ambition: Optional[AmbitionLevel] = Field(None)
    default_base_year: Optional[int] = Field(None, ge=2015, le=2025)
    reporting_year: Optional[int] = Field(None, ge=1990, le=2100)
    default_report_format: Optional[ReportFormat] = Field(None)
    log_level: Optional[str] = Field(None)


# ---------------------------------------------------------------------------
# Generic API Response Models
# ---------------------------------------------------------------------------

class ApiError(BaseModel):
    """Standard API error response."""

    code: str = Field(..., description="Error code (e.g. VALIDATION_ERROR)")
    message: str = Field(..., description="Human-readable error message")
    details: Optional[Dict[str, Any]] = Field(None)
    timestamp: datetime = Field(default_factory=_now)


class ApiResponse(BaseModel):
    """Standard API success response wrapper."""

    success: bool = Field(default=True)
    data: Optional[Any] = Field(None, description="Response payload")
    message: str = Field(default="OK")
    errors: List[ApiError] = Field(default_factory=list)
    timestamp: datetime = Field(default_factory=_now)
    provenance_hash: Optional[str] = Field(None)


class PaginatedResponse(BaseModel):
    """Paginated list response for collection endpoints."""

    items: List[Any] = Field(default_factory=list)
    total: int = Field(default=0, ge=0)
    page: int = Field(default=1, ge=1)
    page_size: int = Field(default=50, ge=1, le=500)
    total_pages: int = Field(default=0, ge=0)
    has_next: bool = Field(default=False)
    has_previous: bool = Field(default=False)

    def model_post_init(self, __context: Any) -> None:
        """Compute pagination metadata."""
        if self.page_size > 0 and self.total > 0:
            computed_pages = (self.total + self.page_size - 1) // self.page_size
            object.__setattr__(self, "total_pages", computed_pages)
            object.__setattr__(self, "has_next", self.page < computed_pages)
            object.__setattr__(self, "has_previous", self.page > 1)
