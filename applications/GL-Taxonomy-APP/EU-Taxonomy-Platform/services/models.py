"""
GL-Taxonomy-APP v1.0 -- EU Taxonomy Platform Domain Models

This module defines all Pydantic v2 domain models for the GL-Taxonomy-APP v1.0
platform.  Models cover the full EU Taxonomy lifecycle: organizations, economic
activities, NACE mapping, eligibility screening, substantial contribution
assessment, DNSH checks, minimum safeguards, alignment results, KPI calculation
(turnover/CapEx/OpEx), GAR/BTAR computation, Article 8 reporting, EBA Pillar 3
disclosures, data quality assessment, gap analysis, and regulatory version
tracking.

All monetary values are in EUR unless otherwise noted.  Timestamps are UTC.

Reference:
    - Regulation (EU) 2020/852 (Taxonomy Regulation)
    - Delegated Regulation (EU) 2021/2139 (Climate Delegated Act)
    - Delegated Regulation (EU) 2023/2486 (Environmental Delegated Act)
    - Delegated Regulation (EU) 2021/2178 (Article 8 Disclosures)
    - EBA Pillar 3 ESG ITS (EBA/ITS/2022/01)

Example:
    >>> from .config import EnvironmentalObjective, AlignmentStatus
    >>> activity = EconomicActivity(
    ...     tenant_id="t-1", org_id="org-1",
    ...     activity_code="4.1", nace_codes=["D35.11"],
    ...     name="Solar PV electricity generation")
    >>> print(activity.id)
"""

from __future__ import annotations

import hashlib
import uuid
from datetime import date, datetime
from decimal import Decimal
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator, model_validator

from .config import (
    ActivityType,
    AlignmentStatus,
    AssessmentStatus,
    AssetClass,
    DataQualityDimension,
    DataQualityGrade,
    DelegatedAct,
    DNSHStatus,
    EntityType,
    EnvironmentalObjective,
    EPCRating,
    ExposureType,
    GapCategory,
    GapPriority,
    GARType,
    KPIType,
    ReportFormat,
    ReportTemplate,
    SafeguardTopic,
    SafeguardTestType,
    Sector,
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
# Core Models
# ---------------------------------------------------------------------------

class Organization(BaseModel):
    """
    Organization registered for EU Taxonomy alignment reporting.

    Captures legal entity identity, entity type (financial vs non-financial),
    sector classification, NFRD/CSRD reporting scope, and LEI.
    Multi-tenant via tenant_id.
    """

    id: str = Field(default_factory=_new_id, description="Unique organization ID")
    tenant_id: str = Field(..., description="Tenant ID for multi-tenancy isolation")
    name: str = Field(..., min_length=1, max_length=500, description="Legal entity name")
    entity_type: EntityType = Field(
        default=EntityType.NON_FINANCIAL, description="Financial or non-financial entity",
    )
    sector: Optional[Sector] = Field(None, description="Primary Taxonomy sector")
    country: str = Field(..., min_length=2, max_length=3, description="HQ country ISO 3166")
    lei: Optional[str] = Field(None, max_length=20, description="Legal Entity Identifier")
    nfrd_reporting: bool = Field(
        default=True, description="Subject to NFRD reporting obligations",
    )
    csrd_reporting: bool = Field(
        default=True, description="Subject to CSRD reporting obligations",
    )
    employee_count: Optional[int] = Field(None, ge=0, description="Full-time equivalents")
    annual_revenue_eur: Optional[Decimal] = Field(None, ge=Decimal("0"))
    total_assets_eur: Optional[Decimal] = Field(None, ge=Decimal("0"))
    contact_person: Optional[str] = Field(None, max_length=255)
    contact_email: Optional[str] = Field(None, max_length=255)
    created_at: datetime = Field(default_factory=_now)
    updated_at: datetime = Field(default_factory=_now)


class EconomicActivity(BaseModel):
    """
    An economic activity assessed under the EU Taxonomy.

    Contains the Taxonomy activity code, NACE codes, sector classification,
    environmental objectives, activity type, and Delegated Act reference.
    """

    id: str = Field(default_factory=_new_id, description="Activity ID")
    tenant_id: str = Field(default="default", description="Tenant ID")
    org_id: str = Field(..., description="Organization ID")
    activity_code: str = Field(
        ..., max_length=20, description="Taxonomy activity code (e.g. 4.1, 7.1)",
    )
    nace_codes: List[str] = Field(
        default_factory=list, description="NACE Rev. 2 codes (e.g. D35.11)",
    )
    sector: Optional[Sector] = Field(None, description="Taxonomy sector")
    name: str = Field(..., min_length=1, max_length=500, description="Activity name")
    description: Optional[str] = Field(None, max_length=2000)
    objectives: List[EnvironmentalObjective] = Field(
        default_factory=list, description="Eligible environmental objectives",
    )
    activity_type: Optional[ActivityType] = Field(
        None, description="own_performance, enabling, or transitional",
    )
    delegated_act: Optional[DelegatedAct] = Field(None, description="Applicable DA")
    turnover_eur: Decimal = Field(default=Decimal("0"), ge=Decimal("0"))
    capex_eur: Decimal = Field(default=Decimal("0"), ge=Decimal("0"))
    opex_eur: Decimal = Field(default=Decimal("0"), ge=Decimal("0"))
    period: str = Field(default="2025", description="Reporting period")
    alignment_status: AlignmentStatus = Field(default=AlignmentStatus.NOT_SCREENED)
    created_at: datetime = Field(default_factory=_now)
    updated_at: datetime = Field(default_factory=_now)


class NACEMapping(BaseModel):
    """NACE code mapping to EU Taxonomy activities."""

    id: str = Field(default_factory=_new_id)
    nace_code: str = Field(..., max_length=20, description="NACE Rev. 2 code")
    description: str = Field(default="", max_length=500)
    level: int = Field(default=4, ge=1, le=4, description="NACE hierarchy level")
    parent_code: Optional[str] = Field(None, max_length=20)
    taxonomy_activities: List[str] = Field(
        default_factory=list, description="Taxonomy activity codes mapped to this NACE",
    )
    created_at: datetime = Field(default_factory=_now)


# ---------------------------------------------------------------------------
# Screening Models
# ---------------------------------------------------------------------------

class ActivityEligibility(BaseModel):
    """Eligibility assessment result for a single activity."""

    id: str = Field(default_factory=_new_id)
    activity_code: str = Field(..., description="Taxonomy activity code")
    eligible: bool = Field(default=False)
    objectives: List[EnvironmentalObjective] = Field(default_factory=list)
    delegated_act: Optional[DelegatedAct] = Field(None)
    confidence: Decimal = Field(
        default=Decimal("0.0"), ge=Decimal("0.0"), le=Decimal("1.0"),
        description="Confidence score for eligibility determination",
    )
    de_minimis_applicable: bool = Field(
        default=False, description="Activity below de minimis threshold",
    )
    reason: str = Field(default="", max_length=2000)
    created_at: datetime = Field(default_factory=_now)


class EligibilityScreening(BaseModel):
    """
    Batch eligibility screening results for an organization.

    Summarizes how many activities were assessed, eligible, and excluded.
    """

    id: str = Field(default_factory=_new_id, description="Screening ID")
    tenant_id: str = Field(..., description="Tenant ID")
    org_id: str = Field(..., description="Organization ID")
    period: str = Field(..., description="Reporting period")
    activities_assessed: int = Field(default=0, ge=0)
    eligible_count: int = Field(default=0, ge=0)
    not_eligible_count: int = Field(default=0, ge=0)
    de_minimis_excluded: int = Field(default=0, ge=0)
    results: List[ActivityEligibility] = Field(default_factory=list)
    screening_date: date = Field(default_factory=lambda: date.today())
    provenance_hash: str = Field(default="")
    created_at: datetime = Field(default_factory=_now)

    def model_post_init(self, __context: Any) -> None:
        """Compute provenance hash."""
        if not self.provenance_hash:
            payload = (
                f"{self.org_id}:{self.period}:"
                f"{self.eligible_count}:{self.not_eligible_count}"
            )
            self.provenance_hash = _sha256(payload)


# ---------------------------------------------------------------------------
# Substantial Contribution Models
# ---------------------------------------------------------------------------

class ThresholdCheck(BaseModel):
    """A single quantitative threshold check for substantial contribution."""

    check_id: str = Field(default_factory=_new_id)
    metric: str = Field(..., max_length=255, description="Metric name")
    required: str = Field(..., description="Required threshold value")
    actual: str = Field(..., description="Actual measured/reported value")
    unit: str = Field(default="", max_length=100)
    pass_result: bool = Field(default=False)


class EvidenceItem(BaseModel):
    """Evidence document supporting an assessment."""

    evidence_id: str = Field(default_factory=_new_id)
    type: str = Field(
        default="document", description="document, measurement, certificate, third_party",
    )
    description: str = Field(default="", max_length=2000)
    document_ref: Optional[str] = Field(None, max_length=500)
    uploaded_at: datetime = Field(default_factory=_now)
    verified: bool = Field(default=False)


class TSCEvaluation(BaseModel):
    """Individual Technical Screening Criteria evaluation item."""

    criterion_id: str = Field(default_factory=_new_id)
    description: str = Field(default="", max_length=2000)
    threshold_value: Optional[str] = Field(None, max_length=255)
    actual_value: Optional[str] = Field(None, max_length=255)
    unit: str = Field(default="", max_length=100)
    pass_result: bool = Field(default=False)
    evidence_ref: Optional[str] = Field(None, max_length=500)


class SCAssessment(BaseModel):
    """
    Substantial Contribution assessment for an activity-objective pair.

    Evaluates whether an activity meets the Technical Screening Criteria
    for substantial contribution to a specific environmental objective.
    """

    id: str = Field(default_factory=_new_id, description="Assessment ID")
    tenant_id: str = Field(..., description="Tenant ID")
    org_id: str = Field(..., description="Organization ID")
    activity_code: str = Field(..., description="Taxonomy activity code")
    objective: EnvironmentalObjective = Field(..., description="Target objective")
    status: AssessmentStatus = Field(default=AssessmentStatus.DRAFT)
    sc_type: Optional[ActivityType] = Field(
        None, description="own_performance, enabling, or transitional",
    )
    threshold_checks: List[ThresholdCheck] = Field(default_factory=list)
    tsc_evaluations: List[TSCEvaluation] = Field(default_factory=list)
    evidence_items: List[EvidenceItem] = Field(default_factory=list)
    overall_pass: bool = Field(default=False)
    assessor_notes: Optional[str] = Field(None, max_length=5000)
    assessment_date: date = Field(default_factory=lambda: date.today())
    provenance_hash: str = Field(default="")
    created_at: datetime = Field(default_factory=_now)
    updated_at: datetime = Field(default_factory=_now)

    def model_post_init(self, __context: Any) -> None:
        """Compute provenance hash."""
        if not self.provenance_hash:
            payload = (
                f"{self.org_id}:{self.activity_code}:{self.objective}:"
                f"{self.overall_pass}"
            )
            self.provenance_hash = _sha256(payload)


# ---------------------------------------------------------------------------
# DNSH Models
# ---------------------------------------------------------------------------

class ObjectiveDNSH(BaseModel):
    """DNSH assessment result for a single environmental objective."""

    objective: EnvironmentalObjective = Field(...)
    status: DNSHStatus = Field(default=DNSHStatus.NOT_ASSESSED)
    criteria_checks: List[str] = Field(
        default_factory=list, description="Criteria checked",
    )
    evidence_items: List[EvidenceItem] = Field(default_factory=list)
    notes: Optional[str] = Field(None, max_length=2000)


class ClimateRiskAssessment(BaseModel):
    """Climate risk assessment for DNSH to climate adaptation (Appendix A)."""

    assessment_id: str = Field(default_factory=_new_id)
    physical_risks: List[str] = Field(
        default_factory=list, description="Identified physical climate risks",
    )
    adaptation_solutions: List[str] = Field(
        default_factory=list, description="Proposed adaptation solutions",
    )
    residual_risks: List[str] = Field(
        default_factory=list, description="Residual risks after adaptation",
    )
    assessment_horizon_years: int = Field(default=30, ge=10, le=100)
    rcp_scenario: str = Field(default="ssp2_45", description="RCP/SSP scenario used")
    created_at: datetime = Field(default_factory=_now)


class DNSHAssessment(BaseModel):
    """
    Do No Significant Harm assessment across all five non-SC objectives.

    For each objective not targeted for substantial contribution, verifies
    that the activity meets the DNSH criteria from the Delegated Act.
    """

    id: str = Field(default_factory=_new_id, description="Assessment ID")
    tenant_id: str = Field(..., description="Tenant ID")
    org_id: str = Field(..., description="Organization ID")
    activity_code: str = Field(..., description="Taxonomy activity code")
    sc_objective: EnvironmentalObjective = Field(
        ..., description="Objective for which SC is claimed",
    )
    objective_results: List[ObjectiveDNSH] = Field(default_factory=list)
    climate_risk_assessment: Optional[ClimateRiskAssessment] = Field(None)
    overall_pass: bool = Field(default=False)
    assessment_date: date = Field(default_factory=lambda: date.today())
    provenance_hash: str = Field(default="")
    created_at: datetime = Field(default_factory=_now)
    updated_at: datetime = Field(default_factory=_now)

    def model_post_init(self, __context: Any) -> None:
        """Compute provenance hash."""
        if not self.provenance_hash:
            payload = (
                f"{self.org_id}:{self.activity_code}:{self.sc_objective}:"
                f"{self.overall_pass}"
            )
            self.provenance_hash = _sha256(payload)


# ---------------------------------------------------------------------------
# Minimum Safeguards Models
# ---------------------------------------------------------------------------

class ProceduralCheck(BaseModel):
    """A single procedural safeguard check."""

    check_id: str = Field(default_factory=_new_id)
    description: str = Field(..., max_length=500)
    status: str = Field(default="not_assessed", description="pass, fail, not_assessed")
    evidence_ref: Optional[str] = Field(None, max_length=500)


class OutcomeCheck(BaseModel):
    """A single outcome-based safeguard check."""

    check_id: str = Field(default_factory=_new_id)
    description: str = Field(..., max_length=500)
    adverse_finding: bool = Field(default=False)
    details: Optional[str] = Field(None, max_length=2000)


class TopicAssessment(BaseModel):
    """Assessment result for a single safeguard topic."""

    topic: SafeguardTopic = Field(...)
    procedural_checks: List[ProceduralCheck] = Field(default_factory=list)
    outcome_checks: List[OutcomeCheck] = Field(default_factory=list)
    procedural_pass: bool = Field(default=False)
    outcome_pass: bool = Field(default=False)
    overall_pass: bool = Field(default=False)
    evidence_items: List[EvidenceItem] = Field(default_factory=list)
    notes: Optional[str] = Field(None, max_length=2000)


class SafeguardAssessment(BaseModel):
    """
    Minimum safeguards assessment across all four topics.

    Company-level assessment covering human rights, anti-corruption,
    taxation, and fair competition per Article 18.
    """

    id: str = Field(default_factory=_new_id, description="Assessment ID")
    tenant_id: str = Field(..., description="Tenant ID")
    org_id: str = Field(..., description="Organization ID")
    assessment_date: date = Field(default_factory=lambda: date.today())
    topics: List[TopicAssessment] = Field(default_factory=list)
    overall_pass: bool = Field(default=False)
    provenance_hash: str = Field(default="")
    created_at: datetime = Field(default_factory=_now)
    updated_at: datetime = Field(default_factory=_now)

    def model_post_init(self, __context: Any) -> None:
        """Compute provenance hash."""
        if not self.provenance_hash:
            payload = f"{self.org_id}:{self.assessment_date}:{self.overall_pass}"
            self.provenance_hash = _sha256(payload)


# ---------------------------------------------------------------------------
# KPI Models
# ---------------------------------------------------------------------------

class ActivityFinancials(BaseModel):
    """Financial data for a single economic activity within a KPI calculation."""

    activity_code: str = Field(..., description="Taxonomy activity code")
    turnover_eur: Decimal = Field(default=Decimal("0"), ge=Decimal("0"))
    capex_eur: Decimal = Field(default=Decimal("0"), ge=Decimal("0"))
    opex_eur: Decimal = Field(default=Decimal("0"), ge=Decimal("0"))
    eligible: bool = Field(default=False)
    aligned: bool = Field(default=False)
    objective: Optional[EnvironmentalObjective] = Field(None)
    activity_type: Optional[ActivityType] = Field(None)


class CapExPlan(BaseModel):
    """CapEx plan for an activity that is eligible but not yet aligned."""

    plan_id: str = Field(default_factory=_new_id)
    tenant_id: str = Field(..., description="Tenant ID")
    org_id: str = Field(..., description="Organization ID")
    activity_code: str = Field(..., description="Taxonomy activity code")
    start_year: int = Field(..., ge=2020, le=2100)
    end_year: int = Field(..., ge=2020, le=2100)
    planned_amounts_eur: Dict[int, Decimal] = Field(
        default_factory=dict, description="Planned CapEx by year (EUR)",
    )
    target_objective: Optional[EnvironmentalObjective] = Field(None)
    management_approved: bool = Field(default=False)
    approval_date: Optional[date] = Field(None)
    created_at: datetime = Field(default_factory=_now)
    updated_at: datetime = Field(default_factory=_now)

    @field_validator("end_year")
    @classmethod
    def end_after_start(cls, v: int, info) -> int:
        """End year must be on or after start year."""
        start = info.data.get("start_year")
        if start is not None and v < start:
            raise ValueError("end_year must be >= start_year")
        return v


class KPICalculation(BaseModel):
    """
    KPI calculation result for a single KPI type (turnover/CapEx/OpEx).

    Computes the proportion of eligible and aligned financial amounts
    relative to the total for the organization.
    """

    id: str = Field(default_factory=_new_id, description="Calculation ID")
    tenant_id: str = Field(..., description="Tenant ID")
    org_id: str = Field(..., description="Organization ID")
    period: str = Field(..., description="Reporting period")
    kpi_type: KPIType = Field(..., description="turnover, capex, or opex")
    total_amount_eur: Decimal = Field(default=Decimal("0"), ge=Decimal("0"))
    eligible_amount_eur: Decimal = Field(default=Decimal("0"), ge=Decimal("0"))
    aligned_amount_eur: Decimal = Field(default=Decimal("0"), ge=Decimal("0"))
    enabling_amount_eur: Decimal = Field(default=Decimal("0"), ge=Decimal("0"))
    transitional_amount_eur: Decimal = Field(default=Decimal("0"), ge=Decimal("0"))
    eligible_pct: Decimal = Field(
        default=Decimal("0"), ge=Decimal("0"), le=Decimal("100"),
    )
    aligned_pct: Decimal = Field(
        default=Decimal("0"), ge=Decimal("0"), le=Decimal("100"),
    )
    activity_breakdown: List[ActivityFinancials] = Field(default_factory=list)
    provenance_hash: str = Field(default="")
    created_at: datetime = Field(default_factory=_now)

    def model_post_init(self, __context: Any) -> None:
        """Compute percentages and provenance hash."""
        if self.total_amount_eur > 0:
            if self.eligible_pct == Decimal("0"):
                object.__setattr__(
                    self, "eligible_pct",
                    (self.eligible_amount_eur / self.total_amount_eur * 100).quantize(
                        Decimal("0.01")
                    ),
                )
            if self.aligned_pct == Decimal("0"):
                object.__setattr__(
                    self, "aligned_pct",
                    (self.aligned_amount_eur / self.total_amount_eur * 100).quantize(
                        Decimal("0.01")
                    ),
                )
        if not self.provenance_hash:
            payload = (
                f"{self.org_id}:{self.period}:{self.kpi_type}:"
                f"{self.aligned_amount_eur}:{self.total_amount_eur}"
            )
            self.provenance_hash = _sha256(payload)


class KPISummary(BaseModel):
    """Summary of all three KPIs for an organization in a period."""

    id: str = Field(default_factory=_new_id)
    tenant_id: str = Field(..., description="Tenant ID")
    org_id: str = Field(..., description="Organization ID")
    period: str = Field(..., description="Reporting period")
    turnover_kpi: Optional[KPICalculation] = Field(None)
    capex_kpi: Optional[KPICalculation] = Field(None)
    opex_kpi: Optional[KPICalculation] = Field(None)
    eligible_turnover_pct: Decimal = Field(default=Decimal("0"))
    eligible_capex_pct: Decimal = Field(default=Decimal("0"))
    eligible_opex_pct: Decimal = Field(default=Decimal("0"))
    aligned_turnover_pct: Decimal = Field(default=Decimal("0"))
    aligned_capex_pct: Decimal = Field(default=Decimal("0"))
    aligned_opex_pct: Decimal = Field(default=Decimal("0"))
    created_at: datetime = Field(default_factory=_now)


# ---------------------------------------------------------------------------
# GAR Models
# ---------------------------------------------------------------------------

class CoveredAssets(BaseModel):
    """Covered assets denominator computation for GAR."""

    total_assets_eur: Decimal = Field(default=Decimal("0"), ge=Decimal("0"))
    sovereign_excluded_eur: Decimal = Field(default=Decimal("0"), ge=Decimal("0"))
    central_bank_excluded_eur: Decimal = Field(default=Decimal("0"), ge=Decimal("0"))
    trading_book_excluded_eur: Decimal = Field(default=Decimal("0"), ge=Decimal("0"))
    derivatives_excluded_eur: Decimal = Field(default=Decimal("0"), ge=Decimal("0"))
    covered_total_eur: Decimal = Field(default=Decimal("0"), ge=Decimal("0"))

    def model_post_init(self, __context: Any) -> None:
        """Compute covered total from exclusions."""
        if self.covered_total_eur == Decimal("0") and self.total_assets_eur > 0:
            computed = (
                self.total_assets_eur
                - self.sovereign_excluded_eur
                - self.central_bank_excluded_eur
                - self.trading_book_excluded_eur
                - self.derivatives_excluded_eur
            )
            object.__setattr__(self, "covered_total_eur", max(computed, Decimal("0")))


class ExposureBreakdown(BaseModel):
    """Breakdown of GAR by exposure type."""

    exposure_type: ExposureType = Field(...)
    total_amount_eur: Decimal = Field(default=Decimal("0"), ge=Decimal("0"))
    eligible_amount_eur: Decimal = Field(default=Decimal("0"), ge=Decimal("0"))
    aligned_amount_eur: Decimal = Field(default=Decimal("0"), ge=Decimal("0"))
    counterparty_count: int = Field(default=0, ge=0)


class GARCalculation(BaseModel):
    """
    Green Asset Ratio calculation for a financial institution.

    GAR = aligned assets / covered assets, computed as stock or flow,
    using turnover-, CapEx-, or OpEx-based counterparty alignment ratios.
    """

    id: str = Field(default_factory=_new_id, description="Calculation ID")
    tenant_id: str = Field(..., description="Tenant ID")
    institution_id: str = Field(..., description="Financial institution org ID")
    period: str = Field(..., description="Reporting period")
    gar_type: GARType = Field(default=GARType.STOCK, description="stock or flow")
    covered_assets: Optional[CoveredAssets] = Field(None)
    aligned_assets_eur: Decimal = Field(default=Decimal("0"), ge=Decimal("0"))
    covered_assets_eur: Decimal = Field(default=Decimal("0"), ge=Decimal("0"))
    gar_percentage: Decimal = Field(
        default=Decimal("0"), ge=Decimal("0"), le=Decimal("100"),
    )
    sector_breakdown: List[Dict[str, Any]] = Field(
        default_factory=list, description="GAR by counterparty sector",
    )
    exposure_breakdown: List[ExposureBreakdown] = Field(default_factory=list)
    provenance_hash: str = Field(default="")
    created_at: datetime = Field(default_factory=_now)

    def model_post_init(self, __context: Any) -> None:
        """Compute GAR percentage and provenance hash."""
        if self.gar_percentage == Decimal("0") and self.covered_assets_eur > 0:
            object.__setattr__(
                self, "gar_percentage",
                (self.aligned_assets_eur / self.covered_assets_eur * 100).quantize(
                    Decimal("0.01")
                ),
            )
        if not self.provenance_hash:
            payload = (
                f"{self.institution_id}:{self.period}:{self.gar_type}:"
                f"{self.aligned_assets_eur}:{self.covered_assets_eur}"
            )
            self.provenance_hash = _sha256(payload)


class BTARCalculation(BaseModel):
    """
    Banking-Book Taxonomy Alignment Ratio (BTAR) calculation.

    Extends GAR to include non-CSRD obligated counterparties using
    estimated alignment ratios or sector-level proxies.
    """

    id: str = Field(default_factory=_new_id, description="Calculation ID")
    tenant_id: str = Field(..., description="Tenant ID")
    institution_id: str = Field(..., description="Financial institution org ID")
    period: str = Field(..., description="Reporting period")
    aligned_assets_eur: Decimal = Field(default=Decimal("0"), ge=Decimal("0"))
    extended_covered_assets_eur: Decimal = Field(
        default=Decimal("0"), ge=Decimal("0"),
        description="Covered assets including non-CSRD counterparties",
    )
    btar_percentage: Decimal = Field(
        default=Decimal("0"), ge=Decimal("0"), le=Decimal("100"),
    )
    provenance_hash: str = Field(default="")
    created_at: datetime = Field(default_factory=_now)

    def model_post_init(self, __context: Any) -> None:
        """Compute BTAR percentage and provenance hash."""
        if self.btar_percentage == Decimal("0") and self.extended_covered_assets_eur > 0:
            object.__setattr__(
                self, "btar_percentage",
                (self.aligned_assets_eur / self.extended_covered_assets_eur * 100).quantize(
                    Decimal("0.01")
                ),
            )
        if not self.provenance_hash:
            payload = (
                f"{self.institution_id}:{self.period}:"
                f"{self.aligned_assets_eur}:{self.extended_covered_assets_eur}"
            )
            self.provenance_hash = _sha256(payload)


# ---------------------------------------------------------------------------
# Alignment Models
# ---------------------------------------------------------------------------

class AlignmentResult(BaseModel):
    """
    Full alignment result for a single activity in a period.

    Records the outcome of all four alignment steps: eligibility,
    substantial contribution, DNSH, and minimum safeguards.
    """

    id: str = Field(default_factory=_new_id, description="Result ID")
    tenant_id: str = Field(..., description="Tenant ID")
    org_id: str = Field(..., description="Organization ID")
    activity_code: str = Field(..., description="Taxonomy activity code")
    period: str = Field(..., description="Reporting period")
    eligible: bool = Field(default=False)
    sc_pass: bool = Field(default=False)
    sc_objective: Optional[EnvironmentalObjective] = Field(None)
    dnsh_pass: bool = Field(default=False)
    ms_pass: bool = Field(default=False)
    aligned: bool = Field(default=False)
    alignment_status: AlignmentStatus = Field(default=AlignmentStatus.NOT_SCREENED)
    activity_type: Optional[ActivityType] = Field(None)
    alignment_details: Dict[str, Any] = Field(
        default_factory=dict, description="Detailed per-step results",
    )
    provenance_hash: str = Field(default="")
    created_at: datetime = Field(default_factory=_now)

    def model_post_init(self, __context: Any) -> None:
        """Derive aligned flag and provenance hash."""
        if self.eligible and self.sc_pass and self.dnsh_pass and self.ms_pass:
            object.__setattr__(self, "aligned", True)
            object.__setattr__(self, "alignment_status", AlignmentStatus.ALIGNED)
        elif self.eligible and not self.aligned:
            object.__setattr__(
                self, "alignment_status", AlignmentStatus.ELIGIBLE_NOT_ALIGNED,
            )
        elif not self.eligible:
            object.__setattr__(self, "alignment_status", AlignmentStatus.NOT_ELIGIBLE)
        if not self.provenance_hash:
            payload = (
                f"{self.org_id}:{self.activity_code}:{self.period}:"
                f"{self.aligned}"
            )
            self.provenance_hash = _sha256(payload)


class PortfolioAlignment(BaseModel):
    """Portfolio-level alignment summary for an organization."""

    id: str = Field(default_factory=_new_id, description="Portfolio alignment ID")
    tenant_id: str = Field(..., description="Tenant ID")
    org_id: str = Field(..., description="Organization ID")
    period: str = Field(..., description="Reporting period")
    total_activities: int = Field(default=0, ge=0)
    eligible_count: int = Field(default=0, ge=0)
    aligned_count: int = Field(default=0, ge=0)
    not_eligible_count: int = Field(default=0, ge=0)
    alignment_percentage: Decimal = Field(
        default=Decimal("0"), ge=Decimal("0"), le=Decimal("100"),
    )
    kpi_summary: Optional[KPISummary] = Field(None)
    results: List[AlignmentResult] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=_now)


# ---------------------------------------------------------------------------
# Reporting Models
# ---------------------------------------------------------------------------

class DisclosureReport(BaseModel):
    """Generated taxonomy disclosure report (Article 8 or EBA)."""

    id: str = Field(default_factory=_new_id, description="Report ID")
    tenant_id: str = Field(..., description="Tenant ID")
    org_id: str = Field(..., description="Organization ID")
    period: str = Field(..., description="Reporting period")
    template: ReportTemplate = Field(..., description="Report template used")
    format: ReportFormat = Field(default=ReportFormat.EXCEL)
    status: AssessmentStatus = Field(default=AssessmentStatus.DRAFT)
    generated_at: datetime = Field(default_factory=_now)
    download_url: Optional[str] = Field(None, max_length=1000)
    provenance_hash: str = Field(default="")
    created_at: datetime = Field(default_factory=_now)


class Article8Data(BaseModel):
    """Article 8 disclosure data for non-financial undertakings."""

    id: str = Field(default_factory=_new_id)
    tenant_id: str = Field(..., description="Tenant ID")
    org_id: str = Field(..., description="Organization ID")
    period: str = Field(..., description="Reporting period")
    turnover_template: Dict[str, Any] = Field(
        default_factory=dict, description="Turnover KPI template data",
    )
    capex_template: Dict[str, Any] = Field(
        default_factory=dict, description="CapEx KPI template data",
    )
    opex_template: Dict[str, Any] = Field(
        default_factory=dict, description="OpEx KPI template data",
    )
    qualitative_disclosures: Dict[str, str] = Field(
        default_factory=dict,
        description="Qualitative explanations (accounting policies, CapEx plans, etc.)",
    )
    created_at: datetime = Field(default_factory=_now)


class EBATemplateData(BaseModel):
    """EBA Pillar 3 template data for credit institutions."""

    id: str = Field(default_factory=_new_id)
    tenant_id: str = Field(..., description="Tenant ID")
    institution_id: str = Field(..., description="Financial institution org ID")
    period: str = Field(..., description="Reporting period")
    template_number: int = Field(..., ge=6, le=10, description="EBA template number")
    gar_data: Dict[str, Any] = Field(default_factory=dict)
    sector_breakdown: List[Dict[str, Any]] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=_now)


# ---------------------------------------------------------------------------
# Data Quality Models
# ---------------------------------------------------------------------------

class DimensionScore(BaseModel):
    """Score for a single data quality dimension."""

    dimension: DataQualityDimension = Field(...)
    score: Decimal = Field(
        default=Decimal("0"), ge=Decimal("0"), le=Decimal("1.0"),
    )
    issues: List[str] = Field(default_factory=list)
    evidence_count: int = Field(default=0, ge=0)
    target_score: Decimal = Field(default=Decimal("0.80"))


class DataQualityScore(BaseModel):
    """
    Overall data quality assessment for an organization's taxonomy data.

    Scores across five dimensions: completeness, accuracy, coverage,
    consistency, and timeliness.
    """

    id: str = Field(default_factory=_new_id, description="Score ID")
    tenant_id: str = Field(..., description="Tenant ID")
    org_id: str = Field(..., description="Organization ID")
    period: str = Field(..., description="Reporting period")
    overall_score: Decimal = Field(
        default=Decimal("0"), ge=Decimal("0"), le=Decimal("1.0"),
    )
    grade: DataQualityGrade = Field(default=DataQualityGrade.INSUFFICIENT)
    dimensions: List[DimensionScore] = Field(default_factory=list)
    improvement_actions: List[str] = Field(default_factory=list)
    provenance_hash: str = Field(default="")
    created_at: datetime = Field(default_factory=_now)

    def model_post_init(self, __context: Any) -> None:
        """Compute provenance hash."""
        if not self.provenance_hash:
            payload = f"{self.org_id}:{self.period}:{self.overall_score}"
            self.provenance_hash = _sha256(payload)


# ---------------------------------------------------------------------------
# Gap Analysis Models
# ---------------------------------------------------------------------------

class GapItem(BaseModel):
    """A single taxonomy alignment gap identified during assessment."""

    gap_id: str = Field(default_factory=_new_id)
    category: GapCategory = Field(..., description="Gap category")
    description: str = Field(..., max_length=2000)
    priority: GapPriority = Field(default=GapPriority.MEDIUM)
    current_status: str = Field(default="", max_length=500)
    target_status: str = Field(default="", max_length=500)
    action_required: str = Field(default="", max_length=2000)
    estimated_effort_days: Optional[int] = Field(None, ge=0)
    responsible_party: Optional[str] = Field(None, max_length=255)
    due_date: Optional[date] = Field(None)


class GapAssessment(BaseModel):
    """Gap analysis result for an organization's taxonomy readiness."""

    id: str = Field(default_factory=_new_id, description="Assessment ID")
    tenant_id: str = Field(..., description="Tenant ID")
    org_id: str = Field(..., description="Organization ID")
    period: str = Field(..., description="Reporting period")
    total_gaps: int = Field(default=0, ge=0)
    critical_gaps: int = Field(default=0, ge=0)
    high_priority: int = Field(default=0, ge=0)
    gaps: List[GapItem] = Field(default_factory=list)
    action_items: List[str] = Field(default_factory=list)
    assessment_date: date = Field(default_factory=lambda: date.today())
    created_at: datetime = Field(default_factory=_now)


# ---------------------------------------------------------------------------
# Request Models
# ---------------------------------------------------------------------------

class ScreenActivityRequest(BaseModel):
    """Request to screen activities for taxonomy eligibility."""

    org_id: str = Field(..., description="Organization ID")
    period: str = Field(..., description="Reporting period")
    activity_codes: List[str] = Field(
        default_factory=list, description="Activity codes to screen (empty = all)",
    )
    delegated_act: Optional[DelegatedAct] = Field(None)
    include_de_minimis: bool = Field(default=True)


class SCAssessmentRequest(BaseModel):
    """Request to assess substantial contribution for an activity."""

    org_id: str = Field(..., description="Organization ID")
    activity_code: str = Field(..., description="Taxonomy activity code")
    objective: EnvironmentalObjective = Field(..., description="Target objective")
    threshold_data: Dict[str, Any] = Field(
        default_factory=dict, description="Actual metric values for threshold checks",
    )
    evidence_refs: List[str] = Field(
        default_factory=list, description="Evidence document references",
    )


class DNSHAssessmentRequest(BaseModel):
    """Request to assess DNSH for an activity."""

    org_id: str = Field(..., description="Organization ID")
    activity_code: str = Field(..., description="Taxonomy activity code")
    sc_objective: EnvironmentalObjective = Field(
        ..., description="Objective for which SC is claimed",
    )
    objective_data: Dict[str, Dict[str, Any]] = Field(
        default_factory=dict,
        description="Per-objective DNSH evidence and check data",
    )
    include_climate_risk: bool = Field(
        default=True, description="Include Appendix A climate risk assessment",
    )


class SafeguardAssessmentRequest(BaseModel):
    """Request to assess minimum safeguards for an organization."""

    org_id: str = Field(..., description="Organization ID")
    topic_data: Dict[str, Dict[str, Any]] = Field(
        default_factory=dict,
        description="Per-topic procedural and outcome check data",
    )


class CalculateKPIRequest(BaseModel):
    """Request to calculate Article 8 KPIs."""

    org_id: str = Field(..., description="Organization ID")
    period: str = Field(..., description="Reporting period")
    kpi_types: List[KPIType] = Field(
        default_factory=lambda: [KPIType.TURNOVER, KPIType.CAPEX, KPIType.OPEX],
    )
    include_capex_plans: bool = Field(default=True)


class CalculateGARRequest(BaseModel):
    """Request to calculate GAR/BTAR for a financial institution."""

    institution_id: str = Field(..., description="Financial institution org ID")
    period: str = Field(..., description="Reporting period")
    gar_type: GARType = Field(default=GARType.STOCK)
    include_btar: bool = Field(default=False)
    kpi_basis: KPIType = Field(
        default=KPIType.TURNOVER,
        description="Counterparty KPI basis for alignment allocation",
    )


class GenerateReportRequest(BaseModel):
    """Request to generate a taxonomy disclosure report."""

    org_id: str = Field(..., description="Organization ID")
    period: str = Field(..., description="Reporting period")
    template: ReportTemplate = Field(..., description="Report template")
    format: ReportFormat = Field(default=ReportFormat.EXCEL)
    include_qualitative: bool = Field(default=True)


class AlignmentWorkflowRequest(BaseModel):
    """Request to run the full alignment workflow for an organization."""

    org_id: str = Field(..., description="Organization ID")
    period: str = Field(..., description="Reporting period")
    activity_codes: List[str] = Field(
        default_factory=list, description="Activity codes (empty = all eligible)",
    )
    delegated_act: Optional[DelegatedAct] = Field(None)
    run_gap_analysis: bool = Field(default=True)


class GapAnalysisRequest(BaseModel):
    """Request to run gap analysis for taxonomy readiness."""

    org_id: str = Field(..., description="Organization ID")
    period: str = Field(..., description="Reporting period")
    categories: List[GapCategory] = Field(
        default_factory=list, description="Gap categories to assess (empty = all)",
    )
    include_action_plan: bool = Field(default=True)
