"""
CDP Climate Change Disclosure Platform Domain Models

This module defines all Pydantic v2 domain models for the GL-CDP-APP v1.0
platform.  Models cover the full CDP disclosure lifecycle: organizations,
questionnaires, modules, questions, responses, versioning, evidence attachments,
review workflows, scoring results, gap analysis, benchmarking, supply chain,
transition plans, verification, and submissions.

CDP Climate Change uses 13 modules (M0-M13), 200+ questions, 17 scoring
categories, 8 scoring levels (D- through A), and 5 mandatory A-level
requirements.  These models faithfully implement that structure while
integrating with 30 MRV agents for auto-population of emissions data.

All monetary values are in USD.  All emissions are in metric tonnes CO2e
unless otherwise noted.  Timestamps are UTC.

Example:
    >>> org = CDPOrganization(name="Acme Corp", gics_sector="20", country="US")
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
    CDPModule,
    EffortLevel,
    GapLevel,
    GapSeverity,
    GICSector,
    QuestionType,
    ReportFormat,
    ResponseStatus,
    ScoringBand,
    ScoringCategory,
    ScoringLevel,
    SupplierStatus,
    TransitionTimeframe,
    VerificationAssurance,
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

class CDPOrganization(GreenLangBase):
    """
    Organization registered for CDP Climate Change disclosure.

    Holds the organizational profile, GICS sector classification,
    and reporting boundary metadata.
    """

    id: str = Field(default_factory=_new_id, description="Unique organization ID")
    name: str = Field(..., min_length=1, max_length=500, description="Legal entity name")
    gics_sector: str = Field(..., description="GICS sector code (e.g. 20 for Industrials)")
    gics_industry_group: Optional[str] = Field(None, description="GICS industry group code")
    country: str = Field(..., min_length=2, max_length=3, description="HQ country code ISO 3166")
    region: Optional[str] = Field(None, max_length=100, description="Geographic region")
    description: Optional[str] = Field(None, max_length=2000)
    contact_person: Optional[str] = Field(None, max_length=255)
    contact_email: Optional[str] = Field(None, max_length=255)
    employee_count: Optional[int] = Field(None, ge=0, description="Full-time equivalents")
    annual_revenue_usd: Optional[Decimal] = Field(None, ge=Decimal("0"))
    is_financial_services: bool = Field(default=False, description="FS sector triggers M12")
    cdp_account_number: Optional[str] = Field(None, max_length=50)
    created_at: datetime = Field(default_factory=_now)
    updated_at: datetime = Field(default_factory=_now)


# ---------------------------------------------------------------------------
# Questionnaire Models
# ---------------------------------------------------------------------------

class Questionnaire(GreenLangBase):
    """
    A CDP Climate Change questionnaire instance for an organization-year.

    Holds 13 modules with status tracking and overall completion.
    """

    id: str = Field(default_factory=_new_id)
    org_id: str = Field(..., description="Organization ID")
    year: int = Field(..., ge=2020, le=2030, description="Reporting year")
    version: str = Field(default="2026", description="Questionnaire version (2024/2025/2026)")
    status: ResponseStatus = Field(default=ResponseStatus.NOT_STARTED)
    total_questions: int = Field(default=0, ge=0)
    answered_questions: int = Field(default=0, ge=0)
    approved_questions: int = Field(default=0, ge=0)
    completion_pct: Decimal = Field(
        default=Decimal("0.0"), ge=Decimal("0"), le=Decimal("100"),
    )
    submission_deadline: Optional[date] = Field(None)
    submitted_at: Optional[datetime] = Field(None)
    created_at: datetime = Field(default_factory=_now)
    updated_at: datetime = Field(default_factory=_now)

    @property
    def is_complete(self) -> bool:
        """Whether all questions are answered."""
        return self.answered_questions >= self.total_questions > 0


class Module(GreenLangBase):
    """
    A module within a CDP questionnaire (M0 through M13).

    Tracks question completion and review status per module.
    """

    id: str = Field(default_factory=_new_id)
    questionnaire_id: str = Field(..., description="Parent questionnaire ID")
    module_code: CDPModule = Field(..., description="Module code (M0-M13)")
    name: str = Field(default="", description="Module display name")
    description: Optional[str] = Field(None)
    order: int = Field(default=0, ge=0)
    required: bool = Field(default=True)
    sector_specific: bool = Field(default=False)
    applicable_sectors: List[str] = Field(default_factory=list)
    total_questions: int = Field(default=0, ge=0)
    answered_questions: int = Field(default=0, ge=0)
    approved_questions: int = Field(default=0, ge=0)
    completion_pct: Decimal = Field(
        default=Decimal("0.0"), ge=Decimal("0"), le=Decimal("100"),
    )
    status: ResponseStatus = Field(default=ResponseStatus.NOT_STARTED)
    created_at: datetime = Field(default_factory=_now)
    updated_at: datetime = Field(default_factory=_now)


class QuestionOption(GreenLangBase):
    """A selectable option for single-select or multi-select questions."""

    value: str = Field(..., description="Option value/key")
    label: str = Field(..., description="Display text")
    score_points: float = Field(default=0.0, description="Points awarded for this selection")


class QuestionDependency(GreenLangBase):
    """Conditional dependency defining skip patterns between questions."""

    parent_question_id: str = Field(..., description="Question that triggers the condition")
    condition_type: str = Field(
        default="equals",
        description="Condition: equals, not_equals, contains, greater_than",
    )
    condition_value: str = Field(..., description="Value that triggers the dependency")
    action: str = Field(
        default="show",
        description="Action when condition met: show, hide, require",
    )


class Question(GreenLangBase):
    """
    A single CDP questionnaire question with metadata.

    Includes question text, guidance, type, scoring category mapping,
    conditional logic, and auto-population source reference.
    """

    id: str = Field(default_factory=_new_id)
    question_number: str = Field(..., description="CDP question number (e.g. C1.1a)")
    module_code: CDPModule = Field(..., description="Parent module")
    sub_section: Optional[str] = Field(None, description="Sub-section within module")
    question_text: str = Field(..., min_length=1, max_length=5000)
    guidance_text: Optional[str] = Field(None, max_length=10000)
    question_type: QuestionType = Field(default=QuestionType.TEXT)
    options: List[QuestionOption] = Field(default_factory=list)
    required: bool = Field(default=True)
    scoring_categories: List[str] = Field(
        default_factory=list,
        description="Scoring category IDs (SC01-SC17) that this question affects",
    )
    scoring_weight: float = Field(
        default=1.0, ge=0.0, le=10.0,
        description="Relative weight within its scoring category",
    )
    disclosure_points: float = Field(default=0.0, description="Points for any disclosure")
    awareness_points: float = Field(default=0.0, description="Points for awareness-level")
    management_points: float = Field(default=0.0, description="Points for management-level")
    leadership_points: float = Field(default=0.0, description="Points for leadership-level")
    max_points: float = Field(default=4.0, description="Maximum achievable points")
    dependencies: List[QuestionDependency] = Field(default_factory=list)
    table_columns: List[str] = Field(
        default_factory=list, description="Column headers for table-type questions",
    )
    auto_populate_source: Optional[str] = Field(
        None, description="MRV agent ID for auto-population",
    )
    example_response: Optional[str] = Field(None, max_length=5000)
    year_introduced: int = Field(default=2024, description="Year this question was introduced")
    year_retired: Optional[int] = Field(None, description="Year this question was retired")
    order: int = Field(default=0, ge=0)
    created_at: datetime = Field(default_factory=_now)


# ---------------------------------------------------------------------------
# Response Models
# ---------------------------------------------------------------------------

class ResponseVersion(GreenLangBase):
    """A historical version of a response for change tracking."""

    id: str = Field(default_factory=_new_id)
    response_id: str = Field(..., description="Parent response ID")
    version_number: int = Field(..., ge=1)
    content: str = Field(default="", max_length=50000, description="Response content text")
    table_data: Optional[List[Dict[str, Any]]] = Field(
        None, description="Tabular data for table-type questions",
    )
    numeric_value: Optional[Decimal] = Field(None, description="Numeric response value")
    selected_options: List[str] = Field(
        default_factory=list, description="Selected option values",
    )
    changed_by: Optional[str] = Field(None, description="User who made the change")
    change_reason: Optional[str] = Field(None, max_length=1000)
    created_at: datetime = Field(default_factory=_now)
    provenance_hash: str = Field(default="")

    def model_post_init(self, __context: Any) -> None:
        """Compute provenance hash."""
        if not self.provenance_hash:
            payload = f"{self.response_id}:{self.version_number}:{self.content}"
            self.provenance_hash = _sha256(payload)


class EvidenceAttachment(GreenLangBase):
    """Evidence document attached to a CDP response."""

    id: str = Field(default_factory=_new_id)
    response_id: str = Field(..., description="Associated response ID")
    file_name: str = Field(..., min_length=1, max_length=255)
    file_type: str = Field(default="pdf", description="File extension type")
    file_size_bytes: int = Field(default=0, ge=0)
    file_path: Optional[str] = Field(None, description="Storage path")
    description: Optional[str] = Field(None, max_length=500)
    uploaded_by: Optional[str] = Field(None, description="User who uploaded")
    uploaded_at: datetime = Field(default_factory=_now)
    provenance_hash: str = Field(default="")

    def model_post_init(self, __context: Any) -> None:
        """Compute provenance hash from file metadata."""
        if not self.provenance_hash:
            payload = f"{self.file_name}:{self.file_size_bytes}:{self.uploaded_at}"
            self.provenance_hash = _sha256(payload)


class ReviewComment(GreenLangBase):
    """A review comment on a response."""

    id: str = Field(default_factory=_new_id)
    response_id: str = Field(..., description="Associated response ID")
    reviewer: str = Field(..., description="Reviewer name or ID")
    comment: str = Field(..., min_length=1, max_length=2000)
    action: str = Field(
        default="comment",
        description="comment, approve, reject, request_change",
    )
    created_at: datetime = Field(default_factory=_now)


class ReviewWorkflow(GreenLangBase):
    """Review workflow tracking for a response."""

    id: str = Field(default_factory=_new_id)
    response_id: str = Field(..., description="Associated response ID")
    assigned_to: Optional[str] = Field(None, description="Assigned reviewer")
    assigned_by: Optional[str] = Field(None, description="Person who assigned")
    status: ResponseStatus = Field(default=ResponseStatus.DRAFT)
    comments: List[ReviewComment] = Field(default_factory=list)
    review_started_at: Optional[datetime] = Field(None)
    review_completed_at: Optional[datetime] = Field(None)
    approved_by: Optional[str] = Field(None)
    approved_at: Optional[datetime] = Field(None)
    created_at: datetime = Field(default_factory=_now)
    updated_at: datetime = Field(default_factory=_now)


class Response(GreenLangBase):
    """
    A response to a single CDP questionnaire question.

    Tracks content, status, scoring, versioning, evidence, and review workflow.
    """

    id: str = Field(default_factory=_new_id)
    questionnaire_id: str = Field(..., description="Parent questionnaire ID")
    question_id: str = Field(..., description="Question being answered")
    question_number: str = Field(default="", description="Question number for display")
    module_code: CDPModule = Field(default=CDPModule.M0_INTRODUCTION)
    status: ResponseStatus = Field(default=ResponseStatus.NOT_STARTED)
    content: str = Field(default="", max_length=50000, description="Response text content")
    table_data: Optional[List[Dict[str, Any]]] = Field(None)
    numeric_value: Optional[Decimal] = Field(None)
    selected_options: List[str] = Field(default_factory=list)
    is_auto_populated: bool = Field(default=False, description="Whether from MRV agent data")
    auto_populate_source: Optional[str] = Field(None, description="MRV agent source ID")
    manual_override: bool = Field(default=False, description="Whether manually overridden")
    override_justification: Optional[str] = Field(None, max_length=1000)
    current_version: int = Field(default=1, ge=1)
    versions: List[ResponseVersion] = Field(default_factory=list)
    evidence: List[EvidenceAttachment] = Field(default_factory=list)
    workflow: Optional[ReviewWorkflow] = Field(None)
    assigned_to: Optional[str] = Field(None, description="Team member assigned")
    score_disclosure: float = Field(default=0.0, ge=0.0)
    score_awareness: float = Field(default=0.0, ge=0.0)
    score_management: float = Field(default=0.0, ge=0.0)
    score_leadership: float = Field(default=0.0, ge=0.0)
    total_score: float = Field(default=0.0, ge=0.0)
    max_score: float = Field(default=4.0, ge=0.0)
    confidence: float = Field(
        default=0.0, ge=0.0, le=1.0,
        description="Confidence in auto-populated data",
    )
    last_saved_at: Optional[datetime] = Field(None)
    created_at: datetime = Field(default_factory=_now)
    updated_at: datetime = Field(default_factory=_now)
    provenance_hash: str = Field(default="")

    def model_post_init(self, __context: Any) -> None:
        """Compute provenance hash."""
        if not self.provenance_hash:
            payload = (
                f"{self.questionnaire_id}:{self.question_id}:"
                f"{self.content}:{self.status}"
            )
            self.provenance_hash = _sha256(payload)


# ---------------------------------------------------------------------------
# Scoring Models
# ---------------------------------------------------------------------------

class CategoryScore(GreenLangBase):
    """Score for a single CDP scoring category (1 of 17)."""

    category_id: str = Field(..., description="Scoring category ID (SC01-SC17)")
    category_name: str = Field(default="")
    raw_score: float = Field(default=0.0, ge=0.0, description="Raw score before weighting")
    max_possible: float = Field(default=0.0, ge=0.0)
    score_pct: float = Field(
        default=0.0, ge=0.0, le=100.0,
        description="Percentage score for this category",
    )
    weight_management: float = Field(default=0.0, ge=0.0)
    weight_leadership: float = Field(default=0.0, ge=0.0)
    weighted_score_mgmt: float = Field(default=0.0, ge=0.0)
    weighted_score_lead: float = Field(default=0.0, ge=0.0)
    level: ScoringLevel = Field(default=ScoringLevel.D_MINUS)
    band: ScoringBand = Field(default=ScoringBand.DISCLOSURE)
    question_count: int = Field(default=0, ge=0)
    answered_count: int = Field(default=0, ge=0)
    applicable: bool = Field(default=True, description="Whether category applies to this org")


class ARequirementStatus(GreenLangBase):
    """Status of one of the 5 A-level requirements."""

    requirement_id: str = Field(..., description="Requirement ID (AREQ01-AREQ05)")
    name: str = Field(default="")
    description: str = Field(default="")
    met: bool = Field(default=False)
    evidence: Optional[str] = Field(None, description="Evidence or reference")
    details: Optional[str] = Field(None, max_length=1000)


class ScoringResult(GreenLangBase):
    """
    Complete CDP scoring result for a questionnaire.

    Contains overall score, 17 category scores, A-level eligibility,
    and score comparison metadata.
    """

    id: str = Field(default_factory=_new_id)
    questionnaire_id: str = Field(..., description="Questionnaire ID")
    org_id: str = Field(..., description="Organization ID")
    year: int = Field(..., ge=2020, le=2030)
    overall_score_pct: float = Field(
        default=0.0, ge=0.0, le=100.0,
        description="Overall score percentage",
    )
    overall_level: ScoringLevel = Field(default=ScoringLevel.D_MINUS)
    overall_band: ScoringBand = Field(default=ScoringBand.DISCLOSURE)
    category_scores: List[CategoryScore] = Field(default_factory=list)
    a_requirements: List[ARequirementStatus] = Field(default_factory=list)
    a_eligible: bool = Field(default=False, description="Whether all A-requirements are met")
    completion_pct: float = Field(default=0.0, ge=0.0, le=100.0)
    total_questions: int = Field(default=0, ge=0)
    answered_questions: int = Field(default=0, ge=0)
    score_confidence: float = Field(
        default=0.0, ge=0.0, le=1.0,
        description="Confidence in the predicted score",
    )
    score_lower_bound: float = Field(default=0.0, ge=0.0, le=100.0)
    score_upper_bound: float = Field(default=0.0, ge=0.0, le=100.0)
    previous_year_score: Optional[float] = Field(None, description="Last year's score")
    score_delta: Optional[float] = Field(None, description="Change from previous year")
    simulated: bool = Field(default=False, description="Whether this is a what-if simulation")
    calculated_at: datetime = Field(default_factory=_now)
    provenance_hash: str = Field(default="")

    def model_post_init(self, __context: Any) -> None:
        """Compute provenance hash."""
        if not self.provenance_hash:
            payload = (
                f"{self.questionnaire_id}:{self.overall_score_pct}:"
                f"{self.overall_level}:{self.calculated_at}"
            )
            self.provenance_hash = _sha256(payload)


# ---------------------------------------------------------------------------
# Gap Analysis Models
# ---------------------------------------------------------------------------

class GapRecommendation(GreenLangBase):
    """Actionable recommendation for closing a gap."""

    title: str = Field(..., description="Recommendation title")
    description: str = Field(default="", max_length=2000)
    example_response: Optional[str] = Field(None, max_length=5000)
    reference: Optional[str] = Field(None, description="CDP guidance reference")


class GapItem(GreenLangBase):
    """A single identified gap in CDP questionnaire responses."""

    id: str = Field(default_factory=_new_id)
    question_id: str = Field(..., description="Question with the gap")
    question_number: str = Field(default="")
    module_code: CDPModule = Field(default=CDPModule.M0_INTRODUCTION)
    scoring_category: str = Field(default="", description="Affected scoring category ID")
    gap_level: GapLevel = Field(default=GapLevel.DISCLOSURE)
    severity: GapSeverity = Field(default=GapSeverity.MEDIUM)
    current_score: float = Field(default=0.0, ge=0.0)
    target_score: float = Field(default=0.0, ge=0.0)
    score_uplift: float = Field(
        default=0.0, ge=0.0,
        description="Expected overall score improvement if gap is closed",
    )
    effort: EffortLevel = Field(default=EffortLevel.MEDIUM)
    description: str = Field(default="", max_length=2000)
    recommendations: List[GapRecommendation] = Field(default_factory=list)
    resolved: bool = Field(default=False)
    resolved_at: Optional[datetime] = Field(None)
    created_at: datetime = Field(default_factory=_now)


class GapAnalysis(GreenLangBase):
    """
    Complete gap analysis result for a questionnaire.

    Identifies all gaps, prioritizes by score impact, and provides
    actionable recommendations with effort estimation.
    """

    id: str = Field(default_factory=_new_id)
    questionnaire_id: str = Field(..., description="Questionnaire ID")
    org_id: str = Field(..., description="Organization ID")
    year: int = Field(..., ge=2020, le=2030)
    total_gaps: int = Field(default=0, ge=0)
    critical_gaps: int = Field(default=0, ge=0)
    high_gaps: int = Field(default=0, ge=0)
    medium_gaps: int = Field(default=0, ge=0)
    low_gaps: int = Field(default=0, ge=0)
    gaps: List[GapItem] = Field(default_factory=list)
    total_potential_uplift: float = Field(
        default=0.0, ge=0.0,
        description="Total potential score improvement if all gaps closed",
    )
    current_score: float = Field(default=0.0, ge=0.0, le=100.0)
    projected_score: float = Field(
        default=0.0, ge=0.0, le=100.0,
        description="Projected score if all gaps closed",
    )
    gaps_by_module: Dict[str, int] = Field(default_factory=dict)
    gaps_by_category: Dict[str, int] = Field(default_factory=dict)
    gaps_by_severity: Dict[str, int] = Field(default_factory=dict)
    analyzed_at: datetime = Field(default_factory=_now)
    provenance_hash: str = Field(default="")

    def model_post_init(self, __context: Any) -> None:
        """Compute provenance hash and counts."""
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
                f"{self.questionnaire_id}:{self.total_gaps}:"
                f"{self.total_potential_uplift}:{self.analyzed_at}"
            )
            self.provenance_hash = _sha256(payload)


# ---------------------------------------------------------------------------
# Benchmarking Models
# ---------------------------------------------------------------------------

class PeerComparison(GreenLangBase):
    """Comparison data point for a single peer organization."""

    peer_id: str = Field(default_factory=_new_id, description="Anonymous peer ID")
    sector: str = Field(default="", description="GICS sector code")
    region: Optional[str] = Field(None, description="Geographic region")
    overall_score: float = Field(default=0.0, ge=0.0, le=100.0)
    level: ScoringLevel = Field(default=ScoringLevel.D_MINUS)
    category_scores: Dict[str, float] = Field(
        default_factory=dict, description="Score per category ID",
    )
    year: int = Field(default=2026, ge=2020, le=2030)


class SectorDistribution(GreenLangBase):
    """Score distribution data for a sector."""

    sector_code: str = Field(..., description="GICS sector code")
    sector_name: str = Field(default="")
    total_respondents: int = Field(default=0, ge=0)
    mean_score: float = Field(default=0.0, ge=0.0, le=100.0)
    median_score: float = Field(default=0.0, ge=0.0, le=100.0)
    p25_score: float = Field(default=0.0, ge=0.0, le=100.0)
    p75_score: float = Field(default=0.0, ge=0.0, le=100.0)
    min_score: float = Field(default=0.0, ge=0.0, le=100.0)
    max_score: float = Field(default=0.0, ge=0.0, le=100.0)
    a_list_count: int = Field(default=0, ge=0)
    a_list_pct: float = Field(default=0.0, ge=0.0, le=100.0)
    level_distribution: Dict[str, int] = Field(
        default_factory=dict,
        description="Count of orgs per scoring level",
    )


class Benchmark(GreenLangBase):
    """
    Benchmarking result comparing an organization against its peers.

    Provides sector, regional, and category-level comparisons.
    """

    id: str = Field(default_factory=_new_id)
    questionnaire_id: str = Field(..., description="Questionnaire ID")
    org_id: str = Field(..., description="Organization ID")
    year: int = Field(..., ge=2020, le=2030)
    org_score: float = Field(default=0.0, ge=0.0, le=100.0)
    org_level: ScoringLevel = Field(default=ScoringLevel.D_MINUS)
    sector_distribution: Optional[SectorDistribution] = Field(None)
    sector_rank: Optional[int] = Field(None, ge=1, description="Rank within sector")
    sector_percentile: Optional[float] = Field(None, ge=0.0, le=100.0)
    regional_rank: Optional[int] = Field(None, ge=1)
    regional_percentile: Optional[float] = Field(None, ge=0.0, le=100.0)
    category_comparison: Dict[str, Dict[str, float]] = Field(
        default_factory=dict,
        description="Per-category comparison: {cat_id: {org: x, sector_avg: y}}",
    )
    peers: List[PeerComparison] = Field(default_factory=list)
    custom_peer_group_id: Optional[str] = Field(None)
    historical_trend: List[Dict[str, Any]] = Field(
        default_factory=list, description="Sector average scores over years",
    )
    benchmarked_at: datetime = Field(default_factory=_now)


# ---------------------------------------------------------------------------
# Supply Chain Models
# ---------------------------------------------------------------------------

class SupplyChainRequest(GreenLangBase):
    """A CDP Supply Chain disclosure request to a supplier."""

    id: str = Field(default_factory=_new_id)
    org_id: str = Field(..., description="Requesting organization ID")
    supplier_name: str = Field(..., min_length=1, max_length=500)
    supplier_email: str = Field(..., max_length=255)
    supplier_id: Optional[str] = Field(None, description="Internal supplier ID")
    status: SupplierStatus = Field(default=SupplierStatus.NOT_INVITED)
    invited_at: Optional[datetime] = Field(None)
    invitation_expiry: Optional[date] = Field(None)
    reminder_count: int = Field(default=0, ge=0)
    sector: Optional[str] = Field(None, description="Supplier GICS sector")
    country: Optional[str] = Field(None, max_length=3)
    spend_usd: Optional[Decimal] = Field(None, ge=Decimal("0"))
    scope3_category: Optional[int] = Field(None, ge=1, le=15)
    created_at: datetime = Field(default_factory=_now)
    updated_at: datetime = Field(default_factory=_now)


class SupplierResponse(GreenLangBase):
    """A supplier's response to a CDP Supply Chain questionnaire."""

    id: str = Field(default_factory=_new_id)
    request_id: str = Field(..., description="Supply chain request ID")
    supplier_name: str = Field(default="")
    status: SupplierStatus = Field(default=SupplierStatus.IN_PROGRESS)
    cdp_score: Optional[ScoringLevel] = Field(None, description="Supplier's CDP score")
    scope1_emissions: Optional[Decimal] = Field(None, ge=Decimal("0"))
    scope2_emissions: Optional[Decimal] = Field(None, ge=Decimal("0"))
    scope3_emissions: Optional[Decimal] = Field(None, ge=Decimal("0"))
    total_emissions: Optional[Decimal] = Field(None, ge=Decimal("0"))
    has_science_based_target: bool = Field(default=False)
    has_transition_plan: bool = Field(default=False)
    engagement_score: float = Field(
        default=0.0, ge=0.0, le=100.0,
        description="Supplier engagement quality score",
    )
    data_quality_score: float = Field(
        default=0.0, ge=0.0, le=100.0,
    )
    submitted_at: Optional[datetime] = Field(None)
    reviewed_at: Optional[datetime] = Field(None)
    created_at: datetime = Field(default_factory=_now)
    updated_at: datetime = Field(default_factory=_now)


# ---------------------------------------------------------------------------
# Transition Plan Models
# ---------------------------------------------------------------------------

class TransitionMilestone(GreenLangBase):
    """A milestone in the 1.5C transition plan."""

    id: str = Field(default_factory=_new_id)
    plan_id: str = Field(..., description="Parent transition plan ID")
    name: str = Field(..., min_length=1, max_length=255)
    description: Optional[str] = Field(None, max_length=2000)
    timeframe: TransitionTimeframe = Field(default=TransitionTimeframe.MEDIUM_TERM)
    target_year: int = Field(..., ge=2024, le=2060)
    target_reduction_pct: Decimal = Field(
        default=Decimal("0"), ge=Decimal("0"), le=Decimal("100"),
    )
    target_absolute_tco2e: Optional[Decimal] = Field(None, ge=Decimal("0"))
    scope: str = Field(default="all", description="scope_1, scope_2, scope_3, all")
    technology_lever: Optional[str] = Field(None, max_length=255)
    capex_usd: Optional[Decimal] = Field(None, ge=Decimal("0"))
    opex_annual_usd: Optional[Decimal] = Field(None, ge=Decimal("0"))
    progress_pct: Decimal = Field(
        default=Decimal("0"), ge=Decimal("0"), le=Decimal("100"),
    )
    status: str = Field(default="planned", description="planned, in_progress, completed, delayed")
    created_at: datetime = Field(default_factory=_now)
    updated_at: datetime = Field(default_factory=_now)


class TransitionPlan(GreenLangBase):
    """
    1.5C-aligned transition plan for CDP A-level scoring.

    Covers pathway definition, milestones, technology levers,
    investment planning, and SBTi alignment.
    """

    id: str = Field(default_factory=_new_id)
    org_id: str = Field(..., description="Organization ID")
    questionnaire_id: Optional[str] = Field(None)
    name: str = Field(default="1.5C Transition Plan", max_length=255)
    description: Optional[str] = Field(None, max_length=5000)
    base_year: int = Field(default=2020, ge=1990, le=2030)
    base_year_emissions_tco2e: Decimal = Field(default=Decimal("0"), ge=Decimal("0"))
    target_year: int = Field(default=2050, ge=2030, le=2070)
    target_net_zero: bool = Field(default=True)
    interim_target_year: Optional[int] = Field(None, ge=2024, le=2060)
    interim_reduction_pct: Optional[Decimal] = Field(None, ge=Decimal("0"), le=Decimal("100"))
    annual_reduction_rate_pct: Decimal = Field(
        default=Decimal("4.2"),
        ge=Decimal("0"),
        le=Decimal("50"),
        description="Annual absolute reduction rate (SBTi requires >= 4.2%)",
    )
    sbti_status: str = Field(
        default="not_committed",
        description="not_committed, committed, targets_set, validated",
    )
    sbti_target_type: Optional[str] = Field(
        None, description="near_term, long_term, net_zero",
    )
    pathway_aligned: str = Field(
        default="well_below_2c",
        description="1.5c, well_below_2c, 2c, not_aligned",
    )
    milestones: List[TransitionMilestone] = Field(default_factory=list)
    total_capex_usd: Decimal = Field(default=Decimal("0"), ge=Decimal("0"))
    total_opex_annual_usd: Decimal = Field(default=Decimal("0"), ge=Decimal("0"))
    revenue_low_carbon_pct: Decimal = Field(
        default=Decimal("0"), ge=Decimal("0"), le=Decimal("100"),
        description="Percentage of revenue from low-carbon products/services",
    )
    is_public: bool = Field(default=False, description="Whether plan is publicly available")
    board_approved: bool = Field(default=False, description="Whether approved by board")
    board_approval_date: Optional[date] = Field(None)
    overall_progress_pct: Decimal = Field(
        default=Decimal("0"), ge=Decimal("0"), le=Decimal("100"),
    )
    created_at: datetime = Field(default_factory=_now)
    updated_at: datetime = Field(default_factory=_now)
    provenance_hash: str = Field(default="")

    def model_post_init(self, __context: Any) -> None:
        """Compute totals and provenance hash."""
        if self.milestones:
            capex = sum(
                (m.capex_usd or Decimal("0")) for m in self.milestones
            )
            opex = sum(
                (m.opex_annual_usd or Decimal("0")) for m in self.milestones
            )
            if capex > 0:
                object.__setattr__(self, "total_capex_usd", capex)
            if opex > 0:
                object.__setattr__(self, "total_opex_annual_usd", opex)
        if not self.provenance_hash:
            payload = (
                f"{self.org_id}:{self.base_year}:{self.target_year}:"
                f"{self.annual_reduction_rate_pct}"
            )
            self.provenance_hash = _sha256(payload)


# ---------------------------------------------------------------------------
# Verification Models
# ---------------------------------------------------------------------------

class VerificationRecord(GreenLangBase):
    """
    Third-party verification record for CDP emissions data.

    Tracks per-scope verification status, coverage, and assurance level.
    """

    id: str = Field(default_factory=_new_id)
    questionnaire_id: str = Field(..., description="Associated questionnaire ID")
    org_id: str = Field(..., description="Organization ID")
    scope: str = Field(..., description="scope_1, scope_2, scope_3, or scope_3_cat_N")
    assurance_level: VerificationAssurance = Field(default=VerificationAssurance.NOT_VERIFIED)
    coverage_pct: Decimal = Field(
        default=Decimal("0"), ge=Decimal("0"), le=Decimal("100"),
        description="Percentage of emissions verified",
    )
    verifier_name: Optional[str] = Field(None, max_length=255)
    verifier_organization: Optional[str] = Field(None, max_length=255)
    verifier_accreditation: Optional[str] = Field(None, max_length=255)
    verification_standard: Optional[str] = Field(
        None, description="e.g. ISO 14064-3, ISAE 3000, AA1000AS",
    )
    statement_date: Optional[date] = Field(None)
    statement_file_id: Optional[str] = Field(None, description="Evidence attachment ID")
    emissions_verified_tco2e: Optional[Decimal] = Field(None, ge=Decimal("0"))
    year: int = Field(default=2026, ge=2020, le=2030)
    created_at: datetime = Field(default_factory=_now)
    updated_at: datetime = Field(default_factory=_now)


# ---------------------------------------------------------------------------
# Submission Models
# ---------------------------------------------------------------------------

class Submission(GreenLangBase):
    """
    Final CDP questionnaire submission record.

    Tracks the submission event, completeness, and generated reports.
    """

    id: str = Field(default_factory=_new_id)
    questionnaire_id: str = Field(..., description="Questionnaire ID")
    org_id: str = Field(..., description="Organization ID")
    year: int = Field(..., ge=2020, le=2030)
    status: str = Field(
        default="draft",
        description="draft, validated, submitted, accepted",
    )
    completeness_pct: Decimal = Field(
        default=Decimal("0"), ge=Decimal("0"), le=Decimal("100"),
    )
    validation_errors: List[str] = Field(default_factory=list)
    validation_warnings: List[str] = Field(default_factory=list)
    submitted_by: Optional[str] = Field(None)
    submitted_at: Optional[datetime] = Field(None)
    confirmation_number: Optional[str] = Field(None, description="CDP confirmation number")
    report_pdf_path: Optional[str] = Field(None)
    report_excel_path: Optional[str] = Field(None)
    report_xml_path: Optional[str] = Field(None)
    created_at: datetime = Field(default_factory=_now)
    updated_at: datetime = Field(default_factory=_now)
    provenance_hash: str = Field(default="")

    def model_post_init(self, __context: Any) -> None:
        """Compute provenance hash."""
        if not self.provenance_hash:
            payload = (
                f"{self.questionnaire_id}:{self.year}:{self.status}:{self.created_at}"
            )
            self.provenance_hash = _sha256(payload)


# ---------------------------------------------------------------------------
# Historical Tracking Models
# ---------------------------------------------------------------------------

class YearlyScoreRecord(GreenLangBase):
    """Historical score record for one year."""

    year: int = Field(..., ge=2015, le=2030)
    overall_score: float = Field(default=0.0, ge=0.0, le=100.0)
    level: ScoringLevel = Field(default=ScoringLevel.D_MINUS)
    band: ScoringBand = Field(default=ScoringBand.DISCLOSURE)
    category_scores: Dict[str, float] = Field(default_factory=dict)
    completion_pct: float = Field(default=0.0, ge=0.0, le=100.0)
    submitted: bool = Field(default=False)


class HistoricalTrackingResult(GreenLangBase):
    """Multi-year historical tracking result."""

    org_id: str = Field(..., description="Organization ID")
    years: List[YearlyScoreRecord] = Field(default_factory=list)
    score_trend: str = Field(
        default="stable",
        description="improving, declining, stable",
    )
    best_year: Optional[int] = Field(None)
    best_score: Optional[float] = Field(None)
    improvement_rate_pct: float = Field(
        default=0.0, description="Average annual improvement rate",
    )
    category_trends: Dict[str, List[Dict[str, Any]]] = Field(default_factory=dict)
    generated_at: datetime = Field(default_factory=_now)


# ---------------------------------------------------------------------------
# Data Connector Models
# ---------------------------------------------------------------------------

class MRVDataPoint(GreenLangBase):
    """A data point retrieved from an MRV agent."""

    agent_id: str = Field(..., description="MRV agent ID (e.g. MRV-001)")
    agent_name: str = Field(default="")
    scope: str = Field(default="", description="scope_1, scope_2, scope_3")
    scope3_category: Optional[int] = Field(None, ge=1, le=15)
    emissions_tco2e: Decimal = Field(default=Decimal("0"), ge=Decimal("0"))
    unit: str = Field(default="tCO2e")
    methodology: Optional[str] = Field(None)
    data_quality_score: float = Field(default=0.0, ge=0.0, le=100.0)
    reporting_year: int = Field(default=2026, ge=2020, le=2030)
    data_timestamp: datetime = Field(default_factory=_now)
    is_fresh: bool = Field(default=True, description="Data within reporting period")
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


class AutoPopulationResult(GreenLangBase):
    """Result of auto-populating CDP responses from MRV agents."""

    questionnaire_id: str = Field(...)
    data_points: List[MRVDataPoint] = Field(default_factory=list)
    questions_populated: int = Field(default=0, ge=0)
    questions_skipped: int = Field(default=0, ge=0)
    scope1_total_tco2e: Decimal = Field(default=Decimal("0"), ge=Decimal("0"))
    scope2_location_tco2e: Decimal = Field(default=Decimal("0"), ge=Decimal("0"))
    scope2_market_tco2e: Decimal = Field(default=Decimal("0"), ge=Decimal("0"))
    scope3_total_tco2e: Decimal = Field(default=Decimal("0"), ge=Decimal("0"))
    scope3_by_category: Dict[int, Decimal] = Field(default_factory=dict)
    data_freshness_valid: bool = Field(default=True)
    reconciliation_status: str = Field(
        default="clean", description="clean, minor_differences, major_discrepancy",
    )
    reconciliation_delta_tco2e: Decimal = Field(default=Decimal("0"))
    populated_at: datetime = Field(default_factory=_now)
    provenance_hash: str = Field(default="")

    def model_post_init(self, __context: Any) -> None:
        """Compute provenance hash."""
        if not self.provenance_hash:
            payload = (
                f"{self.questionnaire_id}:{self.questions_populated}:"
                f"{self.scope1_total_tco2e}:{self.scope2_location_tco2e}:"
                f"{self.scope3_total_tco2e}"
            )
            self.provenance_hash = _sha256(payload)


# ---------------------------------------------------------------------------
# Dashboard Models
# ---------------------------------------------------------------------------

class DashboardAlert(GreenLangBase):
    """An alert surfaced on the CDP dashboard."""

    id: str = Field(default_factory=_new_id)
    severity: GapSeverity = Field(default=GapSeverity.MEDIUM)
    title: str = Field(..., min_length=1)
    message: str = Field(default="")
    module_code: Optional[CDPModule] = Field(None)
    created_at: datetime = Field(default_factory=_now)
    dismissed: bool = Field(default=False)


class DashboardMetrics(GreenLangBase):
    """
    Aggregated dashboard metrics for the CDP disclosure.

    Provides score gauge, module progress, gap summary, timeline,
    readiness, and A-level eligibility status.
    """

    org_id: str = Field(...)
    year: int = Field(...)
    predicted_score: float = Field(default=0.0, ge=0.0, le=100.0)
    predicted_level: ScoringLevel = Field(default=ScoringLevel.D_MINUS)
    predicted_band: ScoringBand = Field(default=ScoringBand.DISCLOSURE)
    completion_pct: float = Field(default=0.0, ge=0.0, le=100.0)
    answered_questions: int = Field(default=0, ge=0)
    total_questions: int = Field(default=0, ge=0)
    approved_questions: int = Field(default=0, ge=0)
    module_progress: Dict[str, float] = Field(
        default_factory=dict, description="Module code -> completion pct",
    )
    gap_summary: Dict[str, int] = Field(
        default_factory=dict,
        description="Gap counts by severity: critical/high/medium/low",
    )
    days_until_deadline: Optional[int] = Field(None)
    a_requirements_met: int = Field(default=0, ge=0, le=5)
    a_requirements_total: int = Field(default=5, ge=5, le=5)
    previous_year_score: Optional[float] = Field(None)
    score_delta: Optional[float] = Field(None)
    category_scores: Dict[str, float] = Field(
        default_factory=dict, description="Per-category score percentages",
    )
    alerts: List[DashboardAlert] = Field(default_factory=list)
    computed_at: datetime = Field(default_factory=_now)


# ---------------------------------------------------------------------------
# Request / Response API Models
# ---------------------------------------------------------------------------

class CreateOrganizationRequest(GreenLangBase):
    """Request to create a new CDP organization."""

    name: str = Field(..., min_length=1, max_length=500)
    gics_sector: str = Field(..., min_length=2, max_length=4)
    country: str = Field(..., min_length=2, max_length=3)
    description: Optional[str] = Field(None, max_length=2000)
    contact_person: Optional[str] = Field(None, max_length=255)
    contact_email: Optional[str] = Field(None, max_length=255)
    employee_count: Optional[int] = Field(None, ge=0)
    annual_revenue_usd: Optional[Decimal] = Field(None, ge=Decimal("0"))


class CreateQuestionnaireRequest(GreenLangBase):
    """Request to create a new CDP questionnaire."""

    year: int = Field(..., ge=2020, le=2030)
    version: str = Field(default="2026")


class SaveResponseRequest(GreenLangBase):
    """Request to save a response to a question."""

    question_id: str = Field(...)
    content: Optional[str] = Field(None, max_length=50000)
    table_data: Optional[List[Dict[str, Any]]] = Field(None)
    numeric_value: Optional[Decimal] = Field(None)
    selected_options: Optional[List[str]] = Field(None)
    manual_override: bool = Field(default=False)
    override_justification: Optional[str] = Field(None, max_length=1000)


class BulkSaveResponseRequest(GreenLangBase):
    """Request to save multiple responses at once."""

    responses: List[SaveResponseRequest] = Field(...)


class SubmitForReviewRequest(GreenLangBase):
    """Request to submit a response for review."""

    reviewer: str = Field(..., description="Reviewer user ID")
    comment: Optional[str] = Field(None, max_length=2000)


class ApproveResponseRequest(GreenLangBase):
    """Request to approve a response."""

    approved_by: str = Field(...)
    comment: Optional[str] = Field(None, max_length=2000)


class RejectResponseRequest(GreenLangBase):
    """Request to reject (return) a response."""

    rejected_by: str = Field(...)
    reason: str = Field(..., min_length=1, max_length=2000)


class SimulateScoreRequest(GreenLangBase):
    """Request to run a what-if scoring simulation."""

    changes: Dict[str, Dict[str, Any]] = Field(
        ...,
        description="Question ID -> response changes to simulate",
    )


class RunGapAnalysisRequest(GreenLangBase):
    """Request to run gap analysis."""

    target_level: ScoringLevel = Field(
        default=ScoringLevel.A,
        description="Target scoring level for gap identification",
    )
    modules: Optional[List[str]] = Field(
        None, description="Specific modules to analyze; all if None",
    )


class CreateBenchmarkRequest(GreenLangBase):
    """Request to create a benchmark comparison."""

    sector_code: Optional[str] = Field(None, description="GICS sector for comparison")
    region: Optional[str] = Field(None, description="Geographic region")
    custom_peer_ids: Optional[List[str]] = Field(None, description="Custom peer org IDs")


class InviteSupplierRequest(GreenLangBase):
    """Request to invite a supplier to CDP Supply Chain."""

    supplier_name: str = Field(..., min_length=1, max_length=500)
    supplier_email: str = Field(..., max_length=255)
    supplier_id: Optional[str] = Field(None)
    sector: Optional[str] = Field(None)
    country: Optional[str] = Field(None, max_length=3)
    spend_usd: Optional[Decimal] = Field(None, ge=Decimal("0"))
    scope3_category: Optional[int] = Field(None, ge=1, le=15)


class CreateTransitionPlanRequest(GreenLangBase):
    """Request to create a transition plan."""

    name: str = Field(default="1.5C Transition Plan", max_length=255)
    description: Optional[str] = Field(None, max_length=5000)
    base_year: int = Field(default=2020, ge=1990, le=2030)
    base_year_emissions_tco2e: Decimal = Field(default=Decimal("0"), ge=Decimal("0"))
    target_year: int = Field(default=2050, ge=2030, le=2070)
    annual_reduction_rate_pct: Decimal = Field(default=Decimal("4.2"), ge=Decimal("0"))
    pathway_aligned: str = Field(default="1.5c")


class AddMilestoneRequest(GreenLangBase):
    """Request to add a milestone to a transition plan."""

    name: str = Field(..., min_length=1, max_length=255)
    description: Optional[str] = Field(None, max_length=2000)
    timeframe: TransitionTimeframe = Field(default=TransitionTimeframe.MEDIUM_TERM)
    target_year: int = Field(..., ge=2024, le=2060)
    target_reduction_pct: Decimal = Field(default=Decimal("0"), ge=Decimal("0"))
    scope: str = Field(default="all")
    technology_lever: Optional[str] = Field(None, max_length=255)
    capex_usd: Optional[Decimal] = Field(None, ge=Decimal("0"))
    opex_annual_usd: Optional[Decimal] = Field(None, ge=Decimal("0"))


class AddVerificationRequest(GreenLangBase):
    """Request to add a verification record."""

    scope: str = Field(..., description="scope_1, scope_2, scope_3, scope_3_cat_N")
    assurance_level: VerificationAssurance = Field(default=VerificationAssurance.LIMITED)
    coverage_pct: Decimal = Field(default=Decimal("100"), ge=Decimal("0"), le=Decimal("100"))
    verifier_name: Optional[str] = Field(None, max_length=255)
    verifier_organization: Optional[str] = Field(None, max_length=255)
    verification_standard: Optional[str] = Field(None)
    emissions_verified_tco2e: Optional[Decimal] = Field(None, ge=Decimal("0"))


class GenerateReportRequest(GreenLangBase):
    """Request to generate a CDP report."""

    format: ReportFormat = Field(default=ReportFormat.PDF)
    include_scoring: bool = Field(default=True)
    include_gap_analysis: bool = Field(default=True)
    include_benchmarking: bool = Field(default=False)
    include_executive_summary: bool = Field(default=True)


class ExportXMLRequest(GreenLangBase):
    """Request to export CDP ORS-compatible XML."""

    validate_before_export: bool = Field(default=True)
    include_attachments: bool = Field(default=False)


class UpdateSettingsRequest(GreenLangBase):
    """Request to update platform configuration."""

    default_questionnaire_year: Optional[int] = Field(None, ge=2024, le=2030)
    scoring_weight_mode: Optional[str] = Field(None)
    submission_deadline_month: Optional[int] = Field(None, ge=1, le=12)
    submission_deadline_day: Optional[int] = Field(None, ge=1, le=31)
    response_max_length: Optional[int] = Field(None, ge=100, le=50000)
    auto_save_interval_seconds: Optional[int] = Field(None, ge=5, le=300)
    default_report_format: Optional[ReportFormat] = Field(None)
    log_level: Optional[str] = Field(None)


# ---------------------------------------------------------------------------
# Generic API Response Models
# ---------------------------------------------------------------------------

class ApiError(GreenLangBase):
    """Standard API error response."""

    code: str = Field(..., description="Error code (e.g. VALIDATION_ERROR)")
    message: str = Field(..., description="Human-readable error message")
    details: Optional[Dict[str, Any]] = Field(None)
    timestamp: datetime = Field(default_factory=_now)


class ApiResponse(GreenLangBase):
    """Standard API success response wrapper."""

    success: bool = Field(default=True)
    data: Optional[Any] = Field(None, description="Response payload")
    message: str = Field(default="OK")
    errors: List[ApiError] = Field(default_factory=list)
    timestamp: datetime = Field(default_factory=_now)
    provenance_hash: Optional[str] = Field(None)


class PaginatedResponse(GreenLangBase):
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
