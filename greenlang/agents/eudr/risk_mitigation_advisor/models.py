# -*- coding: utf-8 -*-
"""
Risk Mitigation Advisor Data Models - AGENT-EUDR-025

Pydantic v2 data models for the Risk Mitigation Advisor Agent covering
ML-powered strategy recommendation, remediation plan design, supplier
capacity building management, mitigation measure library, effectiveness
tracking with ROI analysis, continuous monitoring and adaptive management,
cost-benefit optimization with linear programming, stakeholder collaboration,
and audit-ready mitigation reporting.

Every model is designed for deterministic serialization and SHA-256
provenance hashing to ensure zero-hallucination, bit-perfect
reproducibility across all risk mitigation operations per
EU 2023/1115 Articles 8, 10, 11, 29, 31 and ISO 31000:2018.

Enumerations (14):
    - RiskCategory, ISO31000TreatmentType, ImplementationComplexity,
      PlanStatus, PlanPhaseType, MilestoneStatus, CapacityTier,
      EnrollmentStatus, TriggerEventType, AdjustmentType,
      StakeholderRole, ReportType, EUDRCommodity, EvidenceQuality

Core Models (16):
    - RiskInput, MitigationStrategy, CostEstimate, CostRange,
      RemediationPlan, PlanPhase, Milestone, EvidenceDocument,
      ResponsibleParty, EscalationTrigger, KPI, MitigationMeasure,
      MeasureApplicability, EffectivenessRecord, CapacityBuildingEnrollment,
      TriggerEvent

Request Models (9):
    - RecommendStrategiesRequest, CreatePlanRequest, EnrollSupplierRequest,
      SearchMeasuresRequest, MeasureEffectivenessRequest,
      OptimizeBudgetRequest, CollaborateRequest, GenerateReportRequest,
      AdaptiveScanRequest

Response Models (9):
    - RecommendStrategiesResponse, CreatePlanResponse,
      EnrollSupplierResponse, SearchMeasuresResponse,
      MeasureEffectivenessResponse, OptimizeBudgetResponse,
      CollaborateResponse, GenerateReportResponse, AdaptiveScanResponse

Author: GreenLang Platform Team
Date: March 2026
PRD: AGENT-EUDR-025 Risk Mitigation Advisor (GL-EUDR-RMA-025)
Status: Production Ready
"""

from __future__ import annotations

import uuid
from datetime import date, datetime, timezone
from decimal import Decimal
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _utcnow() -> datetime:
    """Return current UTC datetime with microseconds zeroed."""
    return datetime.now(timezone.utc).replace(microsecond=0)


def _new_uuid() -> str:
    """Return a new UUID4 string."""
    return str(uuid.uuid4())


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Service version string.
VERSION: str = "1.0.0"

#: EUDR deforestation cutoff date (31 December 2020), per Article 2(1).
EUDR_CUTOFF_DATE: str = "2020-12-31"

#: Maximum batch size for recommendation processing.
MAX_BATCH_SIZE: int = 1000

#: EUDR Article 31 data retention in years.
EUDR_RETENTION_YEARS: int = 5

#: Number of upstream risk agents consumed.
UPSTREAM_AGENT_COUNT: int = 9

#: Risk categories count per PRD.
RISK_CATEGORY_COUNT: int = 8

#: Minimum mitigation measures in library per PRD.
MIN_LIBRARY_MEASURES: int = 500

#: EUDR-regulated commodities per Article 1.
SUPPORTED_COMMODITIES: List[str] = [
    "cattle", "cocoa", "coffee", "palm_oil",
    "rubber", "soya", "wood",
]

#: ISO 31000 treatment types.
ISO_31000_TYPES: List[str] = ["avoid", "reduce", "share", "retain"]

#: Capacity building tier count.
CAPACITY_TIER_COUNT: int = 4

#: Training modules per commodity.
MODULES_PER_COMMODITY: int = 22

#: Default top-K strategy recommendations.
DEFAULT_TOP_K: int = 5


# ---------------------------------------------------------------------------
# Enumerations (14)
# ---------------------------------------------------------------------------


class RiskCategory(str, Enum):
    """Risk categories for EUDR mitigation strategy selection.

    Each category maps to a specific upstream risk assessment agent
    (EUDR-016 through EUDR-024) providing dimensional risk scores.
    """

    COUNTRY = "country"
    """Country-level risk from EUDR-016 Country Risk Evaluator."""

    SUPPLIER = "supplier"
    """Supplier-level risk from EUDR-017 Supplier Risk Scorer."""

    COMMODITY = "commodity"
    """Commodity-level risk from EUDR-018 Commodity Risk Analyzer."""

    CORRUPTION = "corruption"
    """Corruption risk from EUDR-019 Corruption Index Monitor."""

    DEFORESTATION = "deforestation"
    """Deforestation risk from EUDR-020 Deforestation Alert System."""

    INDIGENOUS_RIGHTS = "indigenous_rights"
    """Indigenous rights risk from EUDR-021 Indigenous Rights Checker."""

    PROTECTED_AREAS = "protected_areas"
    """Protected area risk from EUDR-022 Protected Area Validator."""

    LEGAL_COMPLIANCE = "legal_compliance"
    """Legal compliance risk from EUDR-023 Legal Compliance Verifier."""


class ISO31000TreatmentType(str, Enum):
    """ISO 31000:2018 risk treatment option types.

    Classifies mitigation strategies according to the international
    standard for risk management treatment options.
    """

    AVOID = "avoid"
    """Decide not to start or continue the activity creating the risk."""

    REDUCE = "reduce"
    """Remove the source or change the likelihood or consequences."""

    SHARE = "share"
    """Transfer risk through insurance, outsourcing, or partnerships."""

    RETAIN = "retain"
    """Accept the risk by informed decision with monitoring."""


class ImplementationComplexity(str, Enum):
    """Implementation complexity levels for mitigation measures.

    Categorizes the difficulty and resource requirements for
    deploying a mitigation measure.
    """

    LOW = "low"
    """Minimal resources, standard procedures, < 2 weeks setup."""

    MEDIUM = "medium"
    """Moderate resources, some customization, 2-8 weeks setup."""

    HIGH = "high"
    """Significant resources, expert involvement, 8-16 weeks setup."""

    VERY_HIGH = "very_high"
    """Major investment, multi-team coordination, > 16 weeks setup."""


class PlanStatus(str, Enum):
    """Remediation plan lifecycle status.

    Tracks the current state of a remediation plan through its
    complete lifecycle from creation to completion or abandonment.
    """

    DRAFT = "draft"
    """Plan created but not yet activated or approved."""

    ACTIVE = "active"
    """Plan approved and implementation has started."""

    ON_TRACK = "on_track"
    """Plan implementation proceeding on schedule."""

    AT_RISK = "at_risk"
    """Plan has issues that may cause delays if not addressed."""

    DELAYED = "delayed"
    """Plan has missed one or more milestone deadlines."""

    COMPLETED = "completed"
    """All milestones achieved and plan objectives met."""

    SUSPENDED = "suspended"
    """Plan temporarily paused due to external factors."""

    ABANDONED = "abandoned"
    """Plan permanently terminated before completion."""


class PlanPhaseType(str, Enum):
    """Remediation plan phase types.

    Defines the standard 4-phase structure for remediation plans
    aligned with ISO 31000 risk treatment process.
    """

    PREPARATION = "preparation"
    """Weeks 1-2: Baseline assessment, resource mobilization, planning."""

    IMPLEMENTATION = "implementation"
    """Weeks 3-8: Active deployment of mitigation measures."""

    VERIFICATION = "verification"
    """Weeks 9-10: Effectiveness assessment and validation."""

    MONITORING = "monitoring"
    """Ongoing: Continuous monitoring and adaptive management."""


class MilestoneStatus(str, Enum):
    """Milestone completion status within a remediation plan.

    Tracks individual milestone progress from pending through
    completion or overdue state.
    """

    PENDING = "pending"
    """Milestone not yet started."""

    IN_PROGRESS = "in_progress"
    """Milestone work has begun."""

    COMPLETED = "completed"
    """Milestone successfully achieved with evidence."""

    OVERDUE = "overdue"
    """Milestone has passed its due date without completion."""

    SKIPPED = "skipped"
    """Milestone deliberately skipped (with justification)."""


class CapacityTier(str, Enum):
    """Supplier capacity building tier levels.

    4-tier progressive capacity building framework aligned with
    supplier development best practices for EUDR compliance.
    """

    TIER_1_AWARENESS = "tier_1_awareness"
    """Basic EUDR requirements education and awareness (4 modules)."""

    TIER_2_BASIC = "tier_2_basic"
    """Data collection, GPS capture, documentation skills (8 modules)."""

    TIER_3_ADVANCED = "tier_3_advanced"
    """Sustainable agriculture, deforestation-free production (6 modules)."""

    TIER_4_LEADERSHIP = "tier_4_leadership"
    """Peer mentoring, community engagement, certification (4 modules)."""


class EnrollmentStatus(str, Enum):
    """Capacity building enrollment status.

    Tracks supplier progress through a capacity building program.
    """

    ACTIVE = "active"
    """Supplier actively progressing through modules."""

    PAUSED = "paused"
    """Enrollment temporarily paused."""

    COMPLETED = "completed"
    """All modules completed and competency verified."""

    WITHDRAWN = "withdrawn"
    """Supplier withdrawn from program."""


class TriggerEventType(str, Enum):
    """Types of monitoring trigger events requiring plan adjustment.

    Six trigger event types that activate adaptive management
    responses per continuous monitoring engine.
    """

    COUNTRY_RECLASSIFICATION = "country_reclassification"
    """Country risk level changed (e.g., standard to high)."""

    SUPPLIER_RISK_SPIKE = "supplier_risk_spike"
    """Supplier risk score increased by > 20%."""

    DEFORESTATION_ALERT = "deforestation_alert"
    """New deforestation alert on monitored plot."""

    INDIGENOUS_VIOLATION = "indigenous_violation"
    """Indigenous rights violation report in sourcing region."""

    PROTECTED_ENCROACHMENT = "protected_encroachment"
    """Protected area encroachment detection."""

    AUDIT_NONCONFORMANCE = "audit_nonconformance"
    """Audit non-conformance finding from EUDR-024."""


class AdjustmentType(str, Enum):
    """Types of adaptive plan adjustments.

    Five adjustment types that can be recommended when monitoring
    detects that existing mitigation is inadequate.
    """

    PLAN_ACCELERATION = "plan_acceleration"
    """Shorten timelines for existing plan milestones."""

    SCOPE_EXPANSION = "scope_expansion"
    """Add additional measures or extend plan coverage."""

    STRATEGY_REPLACEMENT = "strategy_replacement"
    """Swap underperforming measures for alternatives."""

    EMERGENCY_RESPONSE = "emergency_response"
    """Activate immediate action protocol (sourcing suspension)."""

    PLAN_DEESCALATION = "plan_deescalation"
    """Reduce mitigation intensity when risk decreases."""


class StakeholderRole(str, Enum):
    """Stakeholder roles in the collaboration hub.

    Six stakeholder types with differentiated access levels
    as defined in the PRD stakeholder access matrix.
    """

    INTERNAL_COMPLIANCE = "internal_compliance"
    """Internal compliance team - full access to all plans and data."""

    PROCUREMENT = "procurement"
    """Procurement team - full access with limited plan editing."""

    SUPPLIER = "supplier"
    """Supplier - access to own plan, progress reporting, evidence upload."""

    NGO_PARTNER = "ngo_partner"
    """NGO partner - landscape-level aggregates, joint plans."""

    CERTIFICATION_BODY = "certification_body"
    """Certification body - scheme-related plans, audit results."""

    COMPETENT_AUTHORITY = "competent_authority"
    """EU competent authority - read-only compliance documentation."""


class ReportType(str, Enum):
    """Types of mitigation reports generated.

    Seven report types covering regulatory submission, audit evidence,
    stakeholder communication, and executive dashboards.
    """

    DDS_MITIGATION = "dds_mitigation"
    """DDS Mitigation Section per Article 12(2)(d)."""

    AUTHORITY_PACKAGE = "authority_package"
    """Competent Authority Response Package per Articles 14-16."""

    ANNUAL_REVIEW = "annual_review"
    """Annual DDS Review Report per Article 8(3)."""

    SUPPLIER_SCORECARD = "supplier_scorecard"
    """Per-supplier Mitigation Scorecard."""

    PORTFOLIO_SUMMARY = "portfolio_summary"
    """Portfolio Mitigation Summary for board/investors."""

    RISK_MAPPING = "risk_mapping"
    """Risk-to-Mitigation Mapping Report for auditors."""

    EFFECTIVENESS_ANALYSIS = "effectiveness_analysis"
    """Effectiveness Analysis Report with ROI metrics."""


class EUDRCommodity(str, Enum):
    """EUDR-regulated commodities per Article 1.

    Seven commodity categories regulated under the EU Deforestation
    Regulation and their derived products.
    """

    CATTLE = "cattle"
    COCOA = "cocoa"
    COFFEE = "coffee"
    PALM_OIL = "palm_oil"
    RUBBER = "rubber"
    SOYA = "soya"
    WOOD = "wood"


class EvidenceQuality(str, Enum):
    """Quality classification for mitigation evidence.

    Categorizes the strength and reliability of evidence supporting
    mitigation measure effectiveness claims.
    """

    HIGH = "high"
    """Peer-reviewed research, randomized controlled trials."""

    MEDIUM = "medium"
    """Industry case studies, certification body data."""

    LOW = "low"
    """Expert opinion, anecdotal evidence, limited data."""

    UNVERIFIED = "unverified"
    """Community-contributed, not yet reviewed."""


# ---------------------------------------------------------------------------
# Core Models (16)
# ---------------------------------------------------------------------------


class CostRange(BaseModel):
    """Cost range with minimum and maximum EUR values.

    Used for mitigation measure cost estimates and expected risk
    reduction ranges with Decimal arithmetic for financial precision.

    Attributes:
        min_value: Minimum value (EUR or percentage).
        max_value: Maximum value (EUR or percentage).
        currency: Currency code (default EUR).
    """

    model_config = ConfigDict(strict=True, frozen=True)

    min_value: Decimal = Field(
        ..., ge=Decimal("0"), description="Minimum value"
    )
    max_value: Decimal = Field(
        ..., ge=Decimal("0"), description="Maximum value"
    )
    currency: str = Field(
        default="EUR", description="Currency code"
    )

    @model_validator(mode="after")
    def validate_range(self) -> "CostRange":
        """Validate that min_value <= max_value."""
        if self.min_value > self.max_value:
            raise ValueError(
                f"min_value ({self.min_value}) must be <= "
                f"max_value ({self.max_value})"
            )
        return self


class CostEstimate(BaseModel):
    """Cost estimate for a mitigation strategy or measure.

    Structured cost estimate with range, complexity level,
    and optional breakdown by cost category.

    Attributes:
        level: Cost level classification (low/medium/high).
        range_eur: EUR cost range with min/max.
        annual_recurring: Whether cost is annually recurring.
        breakdown: Optional cost breakdown by category.
    """

    model_config = ConfigDict(strict=True, frozen=True)

    level: str = Field(
        ..., description="Cost level: low, medium, high"
    )
    range_eur: CostRange = Field(
        ..., description="EUR cost range"
    )
    annual_recurring: bool = Field(
        default=False, description="Whether cost recurs annually"
    )
    breakdown: Optional[Dict[str, Decimal]] = Field(
        default=None, description="Cost breakdown by category"
    )


class MeasureApplicability(BaseModel):
    """Applicability criteria for a mitigation measure.

    Defines the contexts in which a mitigation measure is applicable:
    which commodities, countries, supply chain tiers, and company sizes.

    Attributes:
        commodities: Applicable EUDR commodities.
        countries: Applicable country ISO codes.
        supply_chain_tiers: Applicable supply chain tiers (1-5).
        company_sizes: Applicable company sizes.
        min_risk_score: Minimum risk score for applicability.
        max_risk_score: Maximum risk score for applicability.
    """

    model_config = ConfigDict(strict=True, frozen=True)

    commodities: List[str] = Field(
        default_factory=list,
        description="Applicable EUDR commodities",
    )
    countries: List[str] = Field(
        default_factory=list,
        description="Applicable country ISO 3166-1 alpha-2 codes",
    )
    supply_chain_tiers: List[int] = Field(
        default_factory=lambda: [1, 2, 3],
        description="Applicable supply chain tiers",
    )
    company_sizes: List[str] = Field(
        default_factory=lambda: ["sme", "large", "enterprise"],
        description="Applicable company sizes",
    )
    min_risk_score: Decimal = Field(
        default=Decimal("0"),
        description="Minimum risk score for applicability",
    )
    max_risk_score: Decimal = Field(
        default=Decimal("100"),
        description="Maximum risk score for applicability",
    )


class RiskInput(BaseModel):
    """Aggregated risk input from 9 upstream EUDR risk assessment agents.

    Captures the multi-dimensional risk profile for a specific supplier,
    country, and commodity combination, consumed by the Strategy Selector
    for mitigation recommendation generation.

    Attributes:
        operator_id: Operator identifier.
        supplier_id: Supplier identifier.
        country_code: ISO 3166-1 alpha-2 country code.
        commodity: EUDR commodity type.
        country_risk_score: Score from EUDR-016 (0-100).
        supplier_risk_score: Score from EUDR-017 (0-100).
        commodity_risk_score: Score from EUDR-018 (0-100).
        corruption_risk_score: Score from EUDR-019 (0-100).
        deforestation_risk_score: Score from EUDR-020 (0-100).
        indigenous_rights_score: Score from EUDR-021 (0-100).
        protected_areas_score: Score from EUDR-022 (0-100).
        legal_compliance_score: Score from EUDR-023 (0-100).
        audit_risk_score: Score from EUDR-024 (0-100).
        due_diligence_level: Required due diligence level.
        risk_factors: Detailed risk factor breakdown.
        assessment_date: Date of risk assessment.
    """

    model_config = ConfigDict(strict=True, frozen=True)

    operator_id: str = Field(
        ..., min_length=1, description="Operator identifier"
    )
    supplier_id: str = Field(
        ..., min_length=1, description="Supplier identifier"
    )
    country_code: str = Field(
        ..., min_length=2, max_length=2,
        description="ISO 3166-1 alpha-2 country code",
    )
    commodity: str = Field(
        ..., description="EUDR commodity type"
    )
    country_risk_score: Decimal = Field(
        default=Decimal("0"), ge=Decimal("0"), le=Decimal("100"),
        description="Country risk score from EUDR-016",
    )
    supplier_risk_score: Decimal = Field(
        default=Decimal("0"), ge=Decimal("0"), le=Decimal("100"),
        description="Supplier risk score from EUDR-017",
    )
    commodity_risk_score: Decimal = Field(
        default=Decimal("0"), ge=Decimal("0"), le=Decimal("100"),
        description="Commodity risk score from EUDR-018",
    )
    corruption_risk_score: Decimal = Field(
        default=Decimal("0"), ge=Decimal("0"), le=Decimal("100"),
        description="Corruption risk score from EUDR-019",
    )
    deforestation_risk_score: Decimal = Field(
        default=Decimal("0"), ge=Decimal("0"), le=Decimal("100"),
        description="Deforestation risk score from EUDR-020",
    )
    indigenous_rights_score: Decimal = Field(
        default=Decimal("0"), ge=Decimal("0"), le=Decimal("100"),
        description="Indigenous rights score from EUDR-021",
    )
    protected_areas_score: Decimal = Field(
        default=Decimal("0"), ge=Decimal("0"), le=Decimal("100"),
        description="Protected areas score from EUDR-022",
    )
    legal_compliance_score: Decimal = Field(
        default=Decimal("0"), ge=Decimal("0"), le=Decimal("100"),
        description="Legal compliance score from EUDR-023",
    )
    audit_risk_score: Decimal = Field(
        default=Decimal("0"), ge=Decimal("0"), le=Decimal("100"),
        description="Audit risk score from EUDR-024",
    )
    due_diligence_level: str = Field(
        default="standard",
        description="Required due diligence level: simplified, standard, enhanced",
    )
    risk_factors: Dict[str, Any] = Field(
        default_factory=dict,
        description="Detailed risk factor breakdown from upstream agents",
    )
    assessment_date: date = Field(
        default_factory=lambda: date.today(),
        description="Date of risk assessment",
    )

    @field_validator("commodity")
    @classmethod
    def validate_commodity(cls, v: str) -> str:
        """Validate commodity is an EUDR-regulated commodity."""
        valid = {
            "cattle", "cocoa", "coffee", "palm_oil",
            "rubber", "soya", "wood",
        }
        if v not in valid:
            raise ValueError(
                f"Invalid commodity: {v}. Must be one of {sorted(valid)}"
            )
        return v


class MitigationStrategy(BaseModel):
    """A recommended mitigation strategy for a specific risk profile.

    Generated by the Strategy Selector engine using ML-powered
    recommendation or deterministic rule-based fallback. Each strategy
    includes predicted effectiveness, cost estimates, SHAP explainability
    values, and provenance hash for audit trail.

    Attributes:
        strategy_id: Unique identifier for this strategy.
        name: Human-readable strategy name.
        description: Detailed description of the strategy.
        risk_categories: Risk categories this strategy addresses.
        iso_31000_type: ISO 31000 treatment type classification.
        target_risk_factors: Specific risk factors addressed.
        predicted_effectiveness: Predicted risk reduction 0-100.
        confidence_score: ML model confidence 0-1.
        cost_estimate: Structured cost estimate.
        implementation_complexity: Complexity level.
        time_to_effect_weeks: Expected weeks to measurable impact.
        prerequisite_conditions: Conditions that must be met first.
        eudr_articles: Linked EUDR articles for regulatory traceability.
        shap_explanation: SHAP values showing feature importance.
        measure_ids: Linked mitigation measures from library.
        model_version: ML model version used for recommendation.
        provenance_hash: SHA-256 hash for audit trail.
        created_at: Timestamp of strategy creation.
    """

    model_config = ConfigDict(strict=True, frozen=True)

    strategy_id: str = Field(
        default_factory=_new_uuid,
        description="Unique strategy identifier",
    )
    name: str = Field(
        ..., min_length=1, max_length=500,
        description="Human-readable strategy name",
    )
    description: str = Field(
        ..., min_length=1,
        description="Detailed strategy description",
    )
    risk_categories: List[RiskCategory] = Field(
        ..., min_length=1,
        description="Risk categories addressed",
    )
    iso_31000_type: ISO31000TreatmentType = Field(
        ..., description="ISO 31000 treatment type"
    )
    target_risk_factors: List[str] = Field(
        default_factory=list,
        description="Specific risk factors addressed",
    )
    predicted_effectiveness: Decimal = Field(
        ..., ge=Decimal("0"), le=Decimal("100"),
        description="Predicted risk reduction percentage",
    )
    confidence_score: Decimal = Field(
        ..., ge=Decimal("0"), le=Decimal("1"),
        description="ML model confidence score",
    )
    cost_estimate: CostEstimate = Field(
        ..., description="Structured cost estimate"
    )
    implementation_complexity: ImplementationComplexity = Field(
        ..., description="Implementation complexity level"
    )
    time_to_effect_weeks: int = Field(
        ..., ge=1, le=260,
        description="Expected weeks to measurable impact",
    )
    prerequisite_conditions: List[str] = Field(
        default_factory=list,
        description="Conditions that must be met first",
    )
    eudr_articles: List[str] = Field(
        default_factory=list,
        description="Linked EUDR articles",
    )
    shap_explanation: Dict[str, float] = Field(
        default_factory=dict,
        description="SHAP values for explainability",
    )
    measure_ids: List[str] = Field(
        default_factory=list,
        description="Linked mitigation measure IDs",
    )
    model_version: str = Field(
        default="1.0.0",
        description="ML model version used",
    )
    provenance_hash: str = Field(
        default="",
        description="SHA-256 provenance hash",
    )
    created_at: datetime = Field(
        default_factory=_utcnow,
        description="Strategy creation timestamp",
    )


class EvidenceDocument(BaseModel):
    """An evidence document uploaded for milestone verification.

    Attributes:
        document_id: Unique document identifier.
        name: Document filename.
        document_type: Type of evidence (report, photo, certificate).
        upload_date: Date of upload.
        uploaded_by: User who uploaded the document.
        file_size_bytes: File size in bytes.
        content_hash: SHA-256 hash of file content.
        storage_url: S3 storage URL.
    """

    model_config = ConfigDict(strict=True, frozen=True)

    document_id: str = Field(
        default_factory=_new_uuid, description="Document identifier"
    )
    name: str = Field(..., description="Document filename")
    document_type: str = Field(
        ..., description="Evidence type: report, photo, certificate, audit"
    )
    upload_date: datetime = Field(
        default_factory=_utcnow, description="Upload timestamp"
    )
    uploaded_by: str = Field(
        default="system", description="Uploader identifier"
    )
    file_size_bytes: int = Field(
        default=0, ge=0, description="File size in bytes"
    )
    content_hash: str = Field(
        default="", description="SHA-256 hash of file content"
    )
    storage_url: str = Field(
        default="", description="S3 storage URL"
    )


class ResponsibleParty(BaseModel):
    """A party responsible for plan milestone execution.

    Attributes:
        party_id: Unique party identifier.
        name: Party name.
        role: Stakeholder role.
        email: Contact email.
        organization: Organization name.
    """

    model_config = ConfigDict(strict=True, frozen=True)

    party_id: str = Field(
        default_factory=_new_uuid, description="Party identifier"
    )
    name: str = Field(..., description="Party name")
    role: StakeholderRole = Field(
        ..., description="Stakeholder role"
    )
    email: str = Field(default="", description="Contact email")
    organization: str = Field(default="", description="Organization name")


class EscalationTrigger(BaseModel):
    """An escalation trigger for plan monitoring.

    Attributes:
        trigger_id: Unique trigger identifier.
        condition: Condition that triggers escalation.
        threshold: Numeric threshold value.
        escalation_target: Role to escalate to.
        response_sla_hours: SLA for response in hours.
    """

    model_config = ConfigDict(strict=True, frozen=True)

    trigger_id: str = Field(
        default_factory=_new_uuid, description="Trigger identifier"
    )
    condition: str = Field(
        ..., description="Escalation condition description"
    )
    threshold: Decimal = Field(
        default=Decimal("0"), description="Numeric threshold"
    )
    escalation_target: str = Field(
        ..., description="Role to escalate to"
    )
    response_sla_hours: int = Field(
        default=24, ge=1, description="Response SLA in hours"
    )


class KPI(BaseModel):
    """Key Performance Indicator for a remediation plan.

    Attributes:
        kpi_id: Unique KPI identifier.
        name: KPI name.
        description: KPI description.
        target_value: Target numeric value.
        current_value: Current measured value.
        unit: Measurement unit.
        measurement_frequency: Frequency of measurement.
    """

    model_config = ConfigDict(strict=True, frozen=True)

    kpi_id: str = Field(
        default_factory=_new_uuid, description="KPI identifier"
    )
    name: str = Field(..., description="KPI name")
    description: str = Field(default="", description="KPI description")
    target_value: Decimal = Field(
        ..., description="Target numeric value"
    )
    current_value: Decimal = Field(
        default=Decimal("0"), description="Current measured value"
    )
    unit: str = Field(
        default="%", description="Measurement unit"
    )
    measurement_frequency: str = Field(
        default="monthly", description="Measurement frequency"
    )


class Milestone(BaseModel):
    """A SMART milestone within a remediation plan.

    Specific, Measurable, Achievable, Relevant, Time-bound milestones
    linked to EUDR article requirements and evidence upload requirements.

    Attributes:
        milestone_id: Unique milestone identifier.
        plan_id: Parent plan identifier.
        name: Milestone name.
        description: Detailed milestone description.
        phase: Plan phase this milestone belongs to.
        due_date: Target completion date.
        completed_date: Actual completion date.
        status: Current milestone status.
        kpi_target: Target KPI value for completion.
        evidence_required: Required evidence types for completion.
        evidence_uploaded: Uploaded evidence documents.
        eudr_article: Linked EUDR article.
        dependencies: IDs of milestones that must complete first.
    """

    model_config = ConfigDict(strict=True, frozen=True)

    milestone_id: str = Field(
        default_factory=_new_uuid, description="Milestone identifier"
    )
    plan_id: str = Field(
        ..., description="Parent plan identifier"
    )
    name: str = Field(
        ..., min_length=1, max_length=500,
        description="Milestone name",
    )
    description: str = Field(
        default="", description="Detailed milestone description"
    )
    phase: PlanPhaseType = Field(
        ..., description="Plan phase"
    )
    due_date: date = Field(
        ..., description="Target completion date"
    )
    completed_date: Optional[date] = Field(
        default=None, description="Actual completion date"
    )
    status: MilestoneStatus = Field(
        default=MilestoneStatus.PENDING, description="Milestone status"
    )
    kpi_target: Optional[str] = Field(
        default=None, description="Target KPI value"
    )
    evidence_required: List[str] = Field(
        default_factory=list, description="Required evidence types"
    )
    evidence_uploaded: List[EvidenceDocument] = Field(
        default_factory=list, description="Uploaded evidence documents"
    )
    eudr_article: Optional[str] = Field(
        default=None, description="Linked EUDR article"
    )
    dependencies: List[str] = Field(
        default_factory=list,
        description="IDs of prerequisite milestones",
    )


class PlanPhase(BaseModel):
    """A phase within a remediation plan.

    Attributes:
        phase_type: Phase type identifier.
        name: Phase name.
        description: Phase description.
        start_week: Starting week number.
        end_week: Ending week number.
        milestones: Milestones in this phase.
        budget_allocation_pct: Budget allocation percentage.
    """

    model_config = ConfigDict(strict=True, frozen=True)

    phase_type: PlanPhaseType = Field(
        ..., description="Phase type"
    )
    name: str = Field(..., description="Phase name")
    description: str = Field(
        default="", description="Phase description"
    )
    start_week: int = Field(
        ..., ge=1, description="Starting week number"
    )
    end_week: int = Field(
        ..., ge=1, description="Ending week number"
    )
    milestones: List[Milestone] = Field(
        default_factory=list, description="Phase milestones"
    )
    budget_allocation_pct: Decimal = Field(
        default=Decimal("25"), ge=Decimal("0"), le=Decimal("100"),
        description="Budget allocation percentage for this phase",
    )


class RemediationPlan(BaseModel):
    """A structured remediation plan for risk mitigation.

    Multi-phase plan with SMART milestones, responsible parties,
    KPIs, and escalation triggers linked to specific risk findings
    from upstream EUDR agents.

    Attributes:
        plan_id: Unique plan identifier.
        operator_id: Owner operator identifier.
        supplier_id: Target supplier identifier (if supplier-specific).
        plan_name: Human-readable plan name.
        risk_finding_ids: Linked risk finding identifiers.
        strategy_ids: Selected strategy identifiers.
        status: Current plan lifecycle status.
        phases: Multi-phase plan structure.
        milestones: All plan milestones (across phases).
        kpis: Key performance indicators.
        budget_allocated: Total budget allocated (EUR).
        budget_spent: Budget spent to date (EUR).
        start_date: Plan start date.
        target_end_date: Planned end date.
        actual_end_date: Actual end date (if completed).
        responsible_parties: Assigned responsible parties.
        escalation_triggers: Configured escalation triggers.
        plan_template: Template used for plan generation.
        version: Plan version number.
        provenance_hash: SHA-256 provenance hash.
        created_at: Plan creation timestamp.
        updated_at: Last update timestamp.
    """

    model_config = ConfigDict(strict=True, frozen=True)

    plan_id: str = Field(
        default_factory=_new_uuid, description="Plan identifier"
    )
    operator_id: str = Field(
        ..., min_length=1, description="Operator identifier"
    )
    supplier_id: Optional[str] = Field(
        default=None, description="Supplier identifier"
    )
    plan_name: str = Field(
        ..., min_length=1, max_length=500,
        description="Human-readable plan name",
    )
    risk_finding_ids: List[str] = Field(
        default_factory=list,
        description="Linked risk finding identifiers",
    )
    strategy_ids: List[str] = Field(
        default_factory=list,
        description="Selected strategy identifiers",
    )
    status: PlanStatus = Field(
        default=PlanStatus.DRAFT, description="Plan status"
    )
    phases: List[PlanPhase] = Field(
        default_factory=list, description="Plan phases"
    )
    milestones: List[Milestone] = Field(
        default_factory=list, description="All plan milestones"
    )
    kpis: List[KPI] = Field(
        default_factory=list, description="Key performance indicators"
    )
    budget_allocated: Decimal = Field(
        default=Decimal("0"), ge=Decimal("0"),
        description="Total budget allocated (EUR)",
    )
    budget_spent: Decimal = Field(
        default=Decimal("0"), ge=Decimal("0"),
        description="Budget spent to date (EUR)",
    )
    start_date: Optional[date] = Field(
        default=None, description="Plan start date"
    )
    target_end_date: Optional[date] = Field(
        default=None, description="Planned end date"
    )
    actual_end_date: Optional[date] = Field(
        default=None, description="Actual end date"
    )
    responsible_parties: List[ResponsibleParty] = Field(
        default_factory=list, description="Responsible parties"
    )
    escalation_triggers: List[EscalationTrigger] = Field(
        default_factory=list, description="Escalation triggers"
    )
    plan_template: Optional[str] = Field(
        default=None, description="Template used for generation"
    )
    version: int = Field(default=1, ge=1, description="Plan version")
    provenance_hash: str = Field(
        default="", description="SHA-256 provenance hash"
    )
    created_at: datetime = Field(
        default_factory=_utcnow, description="Creation timestamp"
    )
    updated_at: datetime = Field(
        default_factory=_utcnow, description="Last update timestamp"
    )


class MitigationMeasure(BaseModel):
    """A single mitigation measure from the 500+ measure library.

    Each measure includes effectiveness evidence, cost estimates,
    applicability criteria, and regulatory alignment for EUDR
    compliance and ISO 31000 risk treatment categorization.

    Attributes:
        measure_id: Unique measure identifier.
        name: Human-readable measure name.
        description: Detailed measure description.
        risk_category: Primary risk category addressed.
        sub_category: Risk sub-category.
        target_risk_factors: Specific risk factors addressed.
        applicability: Applicability criteria (commodities, countries).
        effectiveness_evidence: Supporting evidence sources.
        effectiveness_rating: Overall effectiveness rating 0-100.
        cost_estimate_eur: Cost range in EUR.
        implementation_complexity: Complexity level.
        time_to_effect_weeks: Expected weeks to impact.
        prerequisite_conditions: Required prerequisites.
        expected_risk_reduction_pct: Expected risk reduction range.
        iso_31000_type: ISO 31000 treatment type.
        eudr_articles: Linked EUDR articles.
        certification_schemes: Applicable certification schemes.
        tags: Searchable tags.
        version: Measure version.
        last_updated: Last update date.
    """

    model_config = ConfigDict(strict=True, frozen=True)

    measure_id: str = Field(
        default_factory=_new_uuid, description="Measure identifier"
    )
    name: str = Field(
        ..., min_length=1, max_length=500,
        description="Measure name",
    )
    description: str = Field(
        ..., min_length=1, description="Detailed description"
    )
    risk_category: RiskCategory = Field(
        ..., description="Primary risk category"
    )
    sub_category: str = Field(
        default="", description="Risk sub-category"
    )
    target_risk_factors: List[str] = Field(
        default_factory=list,
        description="Specific risk factors addressed",
    )
    applicability: MeasureApplicability = Field(
        default_factory=MeasureApplicability,
        description="Applicability criteria",
    )
    effectiveness_evidence: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Supporting evidence sources",
    )
    effectiveness_rating: Decimal = Field(
        default=Decimal("50"), ge=Decimal("0"), le=Decimal("100"),
        description="Overall effectiveness rating",
    )
    cost_estimate_eur: CostRange = Field(
        ..., description="Cost range in EUR"
    )
    implementation_complexity: ImplementationComplexity = Field(
        default=ImplementationComplexity.MEDIUM,
        description="Implementation complexity",
    )
    time_to_effect_weeks: int = Field(
        default=8, ge=1, le=260,
        description="Expected weeks to impact",
    )
    prerequisite_conditions: List[str] = Field(
        default_factory=list,
        description="Required prerequisites",
    )
    expected_risk_reduction_pct: CostRange = Field(
        ..., description="Expected risk reduction range (%)"
    )
    iso_31000_type: ISO31000TreatmentType = Field(
        default=ISO31000TreatmentType.REDUCE,
        description="ISO 31000 treatment type",
    )
    eudr_articles: List[str] = Field(
        default_factory=list,
        description="Linked EUDR articles",
    )
    certification_schemes: List[str] = Field(
        default_factory=list,
        description="Applicable certification schemes",
    )
    tags: List[str] = Field(
        default_factory=list,
        description="Searchable tags",
    )
    version: str = Field(
        default="1.0.0", description="Measure version"
    )
    last_updated: date = Field(
        default_factory=lambda: date.today(),
        description="Last update date",
    )


class EffectivenessRecord(BaseModel):
    """Before-after effectiveness measurement for a mitigation plan.

    Captures baseline and current risk scores, calculates reduction
    percentages, performs ROI analysis, and statistical significance
    testing using Decimal arithmetic for zero-hallucination precision.

    Attributes:
        record_id: Unique record identifier.
        plan_id: Parent plan identifier.
        supplier_id: Target supplier identifier.
        measurement_date: Date of measurement.
        baseline_risk_scores: Risk category to baseline score mapping.
        current_risk_scores: Risk category to current score mapping.
        risk_reduction_pct: Risk category to reduction percentage.
        composite_reduction_pct: Weighted composite reduction.
        predicted_reduction_pct: Strategy Selector prediction.
        deviation_pct: Actual vs predicted deviation.
        roi: Return on investment.
        cost_to_date: EUR spent to date.
        statistical_significance: Whether reduction is significant.
        p_value: Statistical p-value.
        provenance_hash: SHA-256 provenance hash.
    """

    model_config = ConfigDict(strict=True, frozen=True)

    record_id: str = Field(
        default_factory=_new_uuid, description="Record identifier"
    )
    plan_id: str = Field(
        ..., description="Parent plan identifier"
    )
    supplier_id: str = Field(
        ..., description="Supplier identifier"
    )
    measurement_date: datetime = Field(
        default_factory=_utcnow, description="Measurement timestamp"
    )
    baseline_risk_scores: Dict[str, Decimal] = Field(
        default_factory=dict,
        description="Risk category to baseline score",
    )
    current_risk_scores: Dict[str, Decimal] = Field(
        default_factory=dict,
        description="Risk category to current score",
    )
    risk_reduction_pct: Dict[str, Decimal] = Field(
        default_factory=dict,
        description="Risk category to reduction percentage",
    )
    composite_reduction_pct: Decimal = Field(
        default=Decimal("0"),
        description="Weighted composite reduction",
    )
    predicted_reduction_pct: Decimal = Field(
        default=Decimal("0"),
        description="Strategy Selector prediction",
    )
    deviation_pct: Decimal = Field(
        default=Decimal("0"),
        description="Actual vs predicted deviation",
    )
    roi: Optional[Decimal] = Field(
        default=None, description="Return on investment"
    )
    cost_to_date: Decimal = Field(
        default=Decimal("0"), ge=Decimal("0"),
        description="EUR spent to date",
    )
    statistical_significance: bool = Field(
        default=False,
        description="Whether reduction is statistically significant",
    )
    p_value: Optional[Decimal] = Field(
        default=None, description="Statistical p-value"
    )
    provenance_hash: str = Field(
        default="", description="SHA-256 provenance hash"
    )


class CapacityBuildingEnrollment(BaseModel):
    """A supplier enrollment in a capacity building program.

    Tracks supplier progress through 4-tier capacity building
    with commodity-specific training modules, competency assessments,
    and risk score correlation.

    Attributes:
        enrollment_id: Unique enrollment identifier.
        supplier_id: Enrolled supplier identifier.
        program_id: Capacity building program identifier.
        commodity: Target commodity for training.
        current_tier: Current capacity building tier (1-4).
        modules_completed: Number of modules completed.
        modules_total: Total number of modules.
        competency_scores: Module to competency score mapping.
        enrolled_date: Enrollment date.
        target_completion_date: Target completion date.
        status: Enrollment status.
        risk_score_at_enrollment: Risk score at enrollment time.
        current_risk_score: Current risk score.
        provenance_hash: SHA-256 provenance hash.
    """

    model_config = ConfigDict(strict=True, frozen=True)

    enrollment_id: str = Field(
        default_factory=_new_uuid, description="Enrollment identifier"
    )
    supplier_id: str = Field(
        ..., description="Supplier identifier"
    )
    program_id: str = Field(
        ..., description="Program identifier"
    )
    commodity: str = Field(
        ..., description="Target commodity"
    )
    current_tier: int = Field(
        default=1, ge=1, le=4, description="Current tier (1-4)"
    )
    modules_completed: int = Field(
        default=0, ge=0, description="Modules completed"
    )
    modules_total: int = Field(
        default=22, ge=1, description="Total modules"
    )
    competency_scores: Dict[str, Decimal] = Field(
        default_factory=dict,
        description="Module to competency score",
    )
    enrolled_date: date = Field(
        default_factory=lambda: date.today(),
        description="Enrollment date",
    )
    target_completion_date: date = Field(
        ..., description="Target completion date"
    )
    status: EnrollmentStatus = Field(
        default=EnrollmentStatus.ACTIVE, description="Enrollment status"
    )
    risk_score_at_enrollment: Decimal = Field(
        default=Decimal("0"), ge=Decimal("0"), le=Decimal("100"),
        description="Risk score at enrollment",
    )
    current_risk_score: Decimal = Field(
        default=Decimal("0"), ge=Decimal("0"), le=Decimal("100"),
        description="Current risk score",
    )
    provenance_hash: str = Field(
        default="", description="SHA-256 provenance hash"
    )


class TriggerEvent(BaseModel):
    """A monitoring trigger event requiring adaptive management response.

    Generated by the Continuous Monitoring Engine when upstream risk
    signals indicate that existing mitigation plans may be inadequate.

    Attributes:
        event_id: Unique event identifier.
        event_type: Type of trigger event.
        severity: Event severity (critical, high, medium, low).
        source_agent: Upstream agent that generated the signal.
        plan_ids: Affected remediation plan IDs.
        supplier_id: Affected supplier identifier.
        description: Human-readable event description.
        risk_score_before: Risk score before the event.
        risk_score_after: Risk score after the event.
        recommended_adjustment: Recommended adjustment type.
        response_sla_hours: SLA for response in hours.
        detected_at: Event detection timestamp.
        acknowledged_at: Event acknowledgment timestamp.
        resolved_at: Event resolution timestamp.
        provenance_hash: SHA-256 provenance hash.
    """

    model_config = ConfigDict(strict=True, frozen=True)

    event_id: str = Field(
        default_factory=_new_uuid, description="Event identifier"
    )
    event_type: TriggerEventType = Field(
        ..., description="Type of trigger event"
    )
    severity: str = Field(
        default="medium",
        description="Severity: critical, high, medium, low",
    )
    source_agent: str = Field(
        ..., description="Upstream agent identifier"
    )
    plan_ids: List[str] = Field(
        default_factory=list,
        description="Affected plan IDs",
    )
    supplier_id: Optional[str] = Field(
        default=None, description="Affected supplier"
    )
    description: str = Field(
        ..., description="Human-readable event description"
    )
    risk_score_before: Decimal = Field(
        default=Decimal("0"), ge=Decimal("0"), le=Decimal("100"),
        description="Risk score before event",
    )
    risk_score_after: Decimal = Field(
        default=Decimal("0"), ge=Decimal("0"), le=Decimal("100"),
        description="Risk score after event",
    )
    recommended_adjustment: AdjustmentType = Field(
        default=AdjustmentType.SCOPE_EXPANSION,
        description="Recommended adjustment type",
    )
    response_sla_hours: int = Field(
        default=48, ge=1, description="Response SLA hours"
    )
    detected_at: datetime = Field(
        default_factory=_utcnow, description="Detection timestamp"
    )
    acknowledged_at: Optional[datetime] = Field(
        default=None, description="Acknowledgment timestamp"
    )
    resolved_at: Optional[datetime] = Field(
        default=None, description="Resolution timestamp"
    )
    provenance_hash: str = Field(
        default="", description="SHA-256 provenance hash"
    )


# ---------------------------------------------------------------------------
# Request Models (9)
# ---------------------------------------------------------------------------


class RecommendStrategiesRequest(BaseModel):
    """Request to recommend mitigation strategies for a risk profile.

    Attributes:
        risk_input: Aggregated risk input from upstream agents.
        top_k: Number of top strategies to return.
        deterministic_mode: Force rule-based mode.
        include_shap: Include SHAP explanation values.
        budget_constraint_eur: Optional budget constraint.
    """

    model_config = ConfigDict(strict=True)

    risk_input: RiskInput = Field(
        ..., description="Aggregated risk input"
    )
    top_k: int = Field(
        default=5, ge=1, le=20,
        description="Number of strategies to recommend",
    )
    deterministic_mode: bool = Field(
        default=False, description="Force deterministic mode"
    )
    include_shap: bool = Field(
        default=True, description="Include SHAP values"
    )
    budget_constraint_eur: Optional[Decimal] = Field(
        default=None, ge=Decimal("0"),
        description="Optional budget constraint",
    )


class CreatePlanRequest(BaseModel):
    """Request to create a remediation plan.

    Attributes:
        operator_id: Owner operator identifier.
        supplier_id: Target supplier identifier.
        strategy_ids: Selected strategy identifiers.
        risk_finding_ids: Linked risk findings.
        template_name: Plan template to use.
        budget_eur: Allocated budget (EUR).
        target_duration_weeks: Target plan duration.
    """

    model_config = ConfigDict(strict=True)

    operator_id: str = Field(
        ..., min_length=1, description="Operator identifier"
    )
    supplier_id: Optional[str] = Field(
        default=None, description="Supplier identifier"
    )
    strategy_ids: List[str] = Field(
        default_factory=list, description="Strategy identifiers"
    )
    risk_finding_ids: List[str] = Field(
        default_factory=list, description="Risk finding identifiers"
    )
    template_name: Optional[str] = Field(
        default=None, description="Plan template name"
    )
    budget_eur: Decimal = Field(
        default=Decimal("10000"), ge=Decimal("0"),
        description="Budget allocation",
    )
    target_duration_weeks: int = Field(
        default=12, ge=1, le=260,
        description="Target duration in weeks",
    )


class EnrollSupplierRequest(BaseModel):
    """Request to enroll a supplier in capacity building.

    Attributes:
        supplier_id: Supplier to enroll.
        commodity: Target commodity.
        initial_tier: Starting tier (1-4).
        target_completion_weeks: Target completion in weeks.
    """

    model_config = ConfigDict(strict=True)

    supplier_id: str = Field(
        ..., min_length=1, description="Supplier identifier"
    )
    commodity: str = Field(
        ..., description="Target commodity"
    )
    initial_tier: int = Field(
        default=1, ge=1, le=4, description="Starting tier"
    )
    target_completion_weeks: int = Field(
        default=24, ge=4, le=104,
        description="Target completion in weeks",
    )


class SearchMeasuresRequest(BaseModel):
    """Request to search the mitigation measure library.

    Attributes:
        query: Full-text search query.
        risk_category: Filter by risk category.
        commodity: Filter by commodity.
        complexity: Filter by complexity level.
        max_cost_eur: Maximum cost filter.
        iso_31000_type: Filter by treatment type.
        limit: Maximum results to return.
        offset: Result offset for pagination.
    """

    model_config = ConfigDict(strict=True)

    query: Optional[str] = Field(
        default=None, description="Full-text search query"
    )
    risk_category: Optional[RiskCategory] = Field(
        default=None, description="Risk category filter"
    )
    commodity: Optional[str] = Field(
        default=None, description="Commodity filter"
    )
    complexity: Optional[ImplementationComplexity] = Field(
        default=None, description="Complexity filter"
    )
    max_cost_eur: Optional[Decimal] = Field(
        default=None, ge=Decimal("0"),
        description="Maximum cost filter",
    )
    iso_31000_type: Optional[ISO31000TreatmentType] = Field(
        default=None, description="Treatment type filter"
    )
    limit: int = Field(
        default=20, ge=1, le=100, description="Max results"
    )
    offset: int = Field(
        default=0, ge=0, description="Result offset"
    )


class MeasureEffectivenessRequest(BaseModel):
    """Request to measure mitigation effectiveness.

    Attributes:
        plan_id: Plan to measure.
        supplier_id: Supplier to measure.
        include_roi: Calculate ROI.
        include_statistics: Include statistical tests.
    """

    model_config = ConfigDict(strict=True)

    plan_id: str = Field(
        ..., min_length=1, description="Plan identifier"
    )
    supplier_id: str = Field(
        ..., min_length=1, description="Supplier identifier"
    )
    include_roi: bool = Field(
        default=True, description="Calculate ROI"
    )
    include_statistics: bool = Field(
        default=True, description="Include statistical tests"
    )


class OptimizeBudgetRequest(BaseModel):
    """Request to optimize mitigation budget allocation.

    Attributes:
        operator_id: Operator identifier.
        total_budget_eur: Total available budget.
        per_supplier_cap_eur: Per-supplier budget cap.
        category_budgets: Category-specific budget limits.
        supplier_ids: Suppliers to include in optimization.
        candidate_measure_ids: Candidate measures to consider.
        supplier_risk_scores: Risk scores per supplier for optimization.
    """

    model_config = ConfigDict(strict=True)

    operator_id: str = Field(
        ..., min_length=1, description="Operator identifier"
    )
    total_budget_eur: Decimal = Field(
        ..., gt=Decimal("0"), description="Total budget"
    )
    per_supplier_cap_eur: Optional[Decimal] = Field(
        default=None, gt=Decimal("0"),
        description="Per-supplier cap",
    )
    category_budgets: Dict[str, Decimal] = Field(
        default_factory=dict,
        description="Category budget limits",
    )
    supplier_ids: List[str] = Field(
        default_factory=list,
        description="Suppliers in scope",
    )
    candidate_measure_ids: List[str] = Field(
        default_factory=list,
        description="Candidate measures",
    )
    supplier_risk_scores: Dict[str, Decimal] = Field(
        default_factory=dict,
        description="Risk scores per supplier (0-100)",
    )


class CollaborateRequest(BaseModel):
    """Request for stakeholder collaboration action.

    Attributes:
        action: Collaboration action type.
        plan_id: Related plan identifier.
        stakeholder_role: Role of the acting stakeholder.
        message: Communication message.
        document_ids: Related document identifiers.
        task_assignments: Task assignment details.
    """

    model_config = ConfigDict(strict=True)

    action: str = Field(
        ..., description="Collaboration action: message, task, document, progress"
    )
    plan_id: str = Field(
        ..., min_length=1, description="Related plan identifier"
    )
    stakeholder_role: StakeholderRole = Field(
        ..., description="Acting stakeholder role"
    )
    message: Optional[str] = Field(
        default=None, description="Communication message"
    )
    document_ids: List[str] = Field(
        default_factory=list,
        description="Related document IDs",
    )
    task_assignments: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Task assignment details",
    )


class GenerateReportRequest(BaseModel):
    """Request to generate a mitigation report.

    Attributes:
        report_type: Type of report to generate.
        operator_id: Operator identifier.
        supplier_id: Optional supplier scope.
        format: Output format.
        language: Report language.
        date_range_start: Report period start.
        date_range_end: Report period end.
    """

    model_config = ConfigDict(strict=True)

    report_type: ReportType = Field(
        ..., description="Type of report"
    )
    operator_id: str = Field(
        ..., min_length=1, description="Operator identifier"
    )
    supplier_id: Optional[str] = Field(
        default=None, description="Supplier scope"
    )
    format: str = Field(
        default="pdf", description="Output format"
    )
    language: str = Field(
        default="en", description="Report language"
    )
    date_range_start: Optional[date] = Field(
        default=None, description="Period start"
    )
    date_range_end: Optional[date] = Field(
        default=None, description="Period end"
    )


class AdaptiveScanRequest(BaseModel):
    """Request for adaptive management monitoring scan.

    Attributes:
        operator_id: Operator identifier.
        plan_ids: Plans to scan (empty = all active).
        include_recommendations: Generate adjustment recommendations.
    """

    model_config = ConfigDict(strict=True)

    operator_id: str = Field(
        ..., min_length=1, description="Operator identifier"
    )
    plan_ids: List[str] = Field(
        default_factory=list,
        description="Plans to scan (empty = all active)",
    )
    include_recommendations: bool = Field(
        default=True,
        description="Generate adjustment recommendations",
    )


# ---------------------------------------------------------------------------
# Response Models (9)
# ---------------------------------------------------------------------------


class RecommendStrategiesResponse(BaseModel):
    """Response containing recommended mitigation strategies.

    Attributes:
        strategies: Ranked list of recommended strategies.
        composite_risk_score: Input composite risk score.
        risk_level: Overall risk level (low/medium/high/critical).
        model_type: ML model type used.
        model_version: ML model version.
        deterministic_mode: Whether deterministic mode was used.
        processing_time_ms: Processing duration in milliseconds.
        provenance_hash: SHA-256 provenance hash.
    """

    model_config = ConfigDict(strict=True, frozen=True)

    strategies: List[MitigationStrategy] = Field(
        default_factory=list, description="Ranked strategies"
    )
    composite_risk_score: Decimal = Field(
        default=Decimal("0"),
        description="Input composite risk score",
    )
    risk_level: str = Field(
        default="medium", description="Overall risk level"
    )
    model_type: str = Field(
        default="xgboost", description="ML model type"
    )
    model_version: str = Field(
        default="1.0.0", description="ML model version"
    )
    deterministic_mode: bool = Field(
        default=False, description="Deterministic mode used"
    )
    processing_time_ms: Decimal = Field(
        default=Decimal("0"),
        description="Processing duration ms",
    )
    provenance_hash: str = Field(
        default="", description="SHA-256 provenance hash"
    )


class CreatePlanResponse(BaseModel):
    """Response containing a created remediation plan.

    Attributes:
        plan: The created remediation plan.
        milestone_count: Number of generated milestones.
        kpi_count: Number of generated KPIs.
        processing_time_ms: Processing duration in milliseconds.
        provenance_hash: SHA-256 provenance hash.
    """

    model_config = ConfigDict(strict=True, frozen=True)

    plan: RemediationPlan = Field(
        ..., description="Created plan"
    )
    milestone_count: int = Field(
        default=0, description="Generated milestones"
    )
    kpi_count: int = Field(
        default=0, description="Generated KPIs"
    )
    processing_time_ms: Decimal = Field(
        default=Decimal("0"),
        description="Processing duration ms",
    )
    provenance_hash: str = Field(
        default="", description="SHA-256 provenance hash"
    )


class EnrollSupplierResponse(BaseModel):
    """Response containing a capacity building enrollment.

    Attributes:
        enrollment: The created enrollment.
        modules_assigned: Training modules assigned.
        processing_time_ms: Processing duration ms.
        provenance_hash: SHA-256 provenance hash.
    """

    model_config = ConfigDict(strict=True, frozen=True)

    enrollment: CapacityBuildingEnrollment = Field(
        ..., description="Created enrollment"
    )
    modules_assigned: int = Field(
        default=0, description="Modules assigned"
    )
    processing_time_ms: Decimal = Field(
        default=Decimal("0"), description="Processing duration ms"
    )
    provenance_hash: str = Field(
        default="", description="SHA-256 provenance hash"
    )


class SearchMeasuresResponse(BaseModel):
    """Response containing measure library search results.

    Attributes:
        measures: Matching mitigation measures.
        total_count: Total matching measures.
        page_size: Page size.
        offset: Result offset.
        search_time_ms: Search duration ms.
    """

    model_config = ConfigDict(strict=True, frozen=True)

    measures: List[MitigationMeasure] = Field(
        default_factory=list, description="Matching measures"
    )
    total_count: int = Field(
        default=0, description="Total matches"
    )
    page_size: int = Field(
        default=20, description="Page size"
    )
    offset: int = Field(
        default=0, description="Result offset"
    )
    search_time_ms: Decimal = Field(
        default=Decimal("0"), description="Search duration ms"
    )


class MeasureEffectivenessResponse(BaseModel):
    """Response containing effectiveness measurement results.

    Attributes:
        record: Effectiveness measurement record.
        is_underperforming: Whether measure is underperforming.
        recommended_action: Recommended follow-up action.
        processing_time_ms: Processing duration ms.
        provenance_hash: SHA-256 provenance hash.
    """

    model_config = ConfigDict(strict=True, frozen=True)

    record: EffectivenessRecord = Field(
        ..., description="Effectiveness record"
    )
    is_underperforming: bool = Field(
        default=False, description="Is underperforming"
    )
    recommended_action: Optional[str] = Field(
        default=None, description="Recommended action"
    )
    processing_time_ms: Decimal = Field(
        default=Decimal("0"), description="Processing duration ms"
    )
    provenance_hash: str = Field(
        default="", description="SHA-256 provenance hash"
    )


class OptimizeBudgetResponse(BaseModel):
    """Response containing budget optimization results.

    Attributes:
        allocations: Supplier to allocated measures mapping.
        total_budget_used: Total budget allocated.
        expected_risk_reduction: Expected aggregate reduction.
        pareto_frontier: Pareto-optimal budget/reduction points.
        sensitivity: Sensitivity analysis results.
        solver_status: LP solver status.
        processing_time_ms: Processing duration ms.
        provenance_hash: SHA-256 provenance hash.
    """

    model_config = ConfigDict(strict=True, frozen=True)

    allocations: Dict[str, List[Dict[str, Any]]] = Field(
        default_factory=dict,
        description="Supplier to measure allocations",
    )
    total_budget_used: Decimal = Field(
        default=Decimal("0"), description="Total budget used"
    )
    expected_risk_reduction: Decimal = Field(
        default=Decimal("0"),
        description="Expected aggregate reduction",
    )
    pareto_frontier: List[Dict[str, Decimal]] = Field(
        default_factory=list,
        description="Pareto frontier points",
    )
    sensitivity: Dict[str, Any] = Field(
        default_factory=dict,
        description="Sensitivity analysis",
    )
    solver_status: str = Field(
        default="optimal", description="Solver status"
    )
    processing_time_ms: Decimal = Field(
        default=Decimal("0"), description="Processing duration ms"
    )
    provenance_hash: str = Field(
        default="", description="SHA-256 provenance hash"
    )


class CollaborateResponse(BaseModel):
    """Response for a stakeholder collaboration action.

    Attributes:
        action_id: Unique action identifier.
        status: Action status (success/pending/denied).
        message: Response message.
        processing_time_ms: Processing duration ms.
    """

    model_config = ConfigDict(strict=True, frozen=True)

    action_id: str = Field(
        default_factory=_new_uuid, description="Action identifier"
    )
    status: str = Field(
        default="success", description="Action status"
    )
    message: str = Field(
        default="", description="Response message"
    )
    processing_time_ms: Decimal = Field(
        default=Decimal("0"), description="Processing duration ms"
    )


class GenerateReportResponse(BaseModel):
    """Response containing a generated mitigation report.

    Attributes:
        report_id: Unique report identifier.
        report_type: Type of report generated.
        format: Output format.
        language: Report language.
        storage_url: S3 storage URL for the report.
        file_size_bytes: Report file size.
        content_hash: SHA-256 hash of report content.
        generation_time_ms: Generation duration ms.
        provenance_hash: SHA-256 provenance hash.
    """

    model_config = ConfigDict(strict=True, frozen=True)

    report_id: str = Field(
        default_factory=_new_uuid, description="Report identifier"
    )
    report_type: ReportType = Field(
        ..., description="Report type"
    )
    format: str = Field(
        default="pdf", description="Output format"
    )
    language: str = Field(
        default="en", description="Report language"
    )
    storage_url: str = Field(
        default="", description="S3 storage URL"
    )
    file_size_bytes: int = Field(
        default=0, ge=0, description="File size"
    )
    content_hash: str = Field(
        default="", description="SHA-256 content hash"
    )
    generation_time_ms: Decimal = Field(
        default=Decimal("0"), description="Generation duration ms"
    )
    provenance_hash: str = Field(
        default="", description="SHA-256 provenance hash"
    )


class AdaptiveScanResponse(BaseModel):
    """Response containing adaptive management scan results.

    Attributes:
        trigger_events: Detected trigger events.
        adjustments_recommended: Recommended adjustments count.
        plans_affected: Number of affected plans.
        scan_time_ms: Scan duration ms.
        provenance_hash: SHA-256 provenance hash.
    """

    model_config = ConfigDict(strict=True, frozen=True)

    trigger_events: List[TriggerEvent] = Field(
        default_factory=list, description="Detected events"
    )
    adjustments_recommended: int = Field(
        default=0, description="Adjustments recommended"
    )
    plans_affected: int = Field(
        default=0, description="Plans affected"
    )
    scan_time_ms: Decimal = Field(
        default=Decimal("0"), description="Scan duration ms"
    )
    provenance_hash: str = Field(
        default="", description="SHA-256 provenance hash"
    )
