# -*- coding: utf-8 -*-
"""
Indigenous Rights Checker Data Models - AGENT-EUDR-021

Pydantic v2 data models for indigenous rights checking including:
- 15 enumerations (territory status, FPIC status, overlap type, violation
  type, workflow stage, consultation stage, grievance status, agreement
  status, confidence level, risk level, alert severity, data source,
  report type, report format, community recognition status)
- 12 core models (IndigenousTerritory, FPICAssessment, TerritoryOverlap,
  IndigenousCommunity, ConsultationRecord, GrievanceRecord,
  BenefitSharingAgreement, FPICWorkflow, WorkflowTransition,
  ViolationAlert, ComplianceReport, CountryIndigenousRightsScore)
- Request/response models for API endpoints
- Constants for EUDR dates, score bounds, batch limits

Zero-Hallucination: All models use Decimal for scores and percentages
to ensure bit-perfect reproducibility across calculations.

Example:
    >>> from greenlang.agents.eudr.indigenous_rights_checker.models import (
    ...     TerritoryLegalStatus, FPICStatus, OverlapType,
    ...     IndigenousTerritory, FPICAssessment,
    ... )
    >>> from decimal import Decimal
    >>> territory = IndigenousTerritory(
    ...     territory_id="t-001",
    ...     territory_name="Terra Indigena Yanomami",
    ...     people_name="Yanomami",
    ...     country_code="BR",
    ...     legal_status=TerritoryLegalStatus.TITLED,
    ...     data_source="funai",
    ...     provenance_hash="abc123",
    ... )

Author: GreenLang Platform Team
Date: March 2026
PRD: AGENT-EUDR-021 Indigenous Rights Checker (GL-EUDR-IRC-021)
Status: Production Ready
"""

from __future__ import annotations

from datetime import date, datetime
from decimal import Decimal
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

VERSION: str = "1.0.0"
EUDR_CUTOFF_DATE: str = "2020-12-31"
MAX_FPIC_SCORE: Decimal = Decimal("100")
MIN_FPIC_SCORE: Decimal = Decimal("0")
MAX_RISK_SCORE: Decimal = Decimal("100")
MIN_RISK_SCORE: Decimal = Decimal("0")
MAX_BATCH_SIZE: int = 10000
EUDR_RETENTION_YEARS: int = 5
SRID_WGS84: int = 4326

SUPPORTED_REGIONS: List[str] = [
    "amazon_basin",
    "congo_basin",
    "southeast_asia",
    "central_america",
    "west_africa",
    "east_africa",
    "south_asia",
    "oceania",
]

EUDR_COMMODITIES: List[str] = [
    "cattle",
    "cocoa",
    "coffee",
    "palm_oil",
    "rubber",
    "soya",
    "wood",
]


# ---------------------------------------------------------------------------
# Enumerations (15)
# ---------------------------------------------------------------------------


class TerritoryLegalStatus(str, Enum):
    """Legal recognition status of an indigenous territory.

    Per PRD F1.7: titled, declared, claimed, customary, pending, disputed.
    """

    TITLED = "titled"
    DECLARED = "declared"
    CLAIMED = "claimed"
    CUSTOMARY = "customary"
    PENDING = "pending"
    DISPUTED = "disputed"


class FPICStatus(str, Enum):
    """FPIC consent status classification.

    Per PRD F2.6: score >= 80 = obtained, 50-79 = partial, < 50 = missing.
    """

    CONSENT_OBTAINED = "consent_obtained"
    CONSENT_PARTIAL = "consent_partial"
    CONSENT_MISSING = "consent_missing"
    CONSENT_WITHDRAWN = "consent_withdrawn"
    CONSENT_DISPUTED = "consent_disputed"
    NOT_APPLICABLE = "not_applicable"


class OverlapType(str, Enum):
    """Territory overlap classification type.

    Per PRD F3.2: direct, partial, adjacent, proximate, none.
    """

    DIRECT = "direct"
    PARTIAL = "partial"
    ADJACENT = "adjacent"
    PROXIMATE = "proximate"
    NONE = "none"


class ViolationType(str, Enum):
    """Indigenous rights violation type controlled vocabulary.

    Per PRD F5.2: 10 violation categories.
    """

    LAND_SEIZURE = "land_seizure"
    FORCED_DISPLACEMENT = "forced_displacement"
    FPIC_VIOLATION = "fpic_violation"
    ENVIRONMENTAL_DAMAGE = "environmental_damage"
    PHYSICAL_VIOLENCE = "physical_violence"
    CULTURAL_DESTRUCTION = "cultural_destruction"
    RESTRICTION_OF_ACCESS = "restriction_of_access"
    BENEFIT_SHARING_BREACH = "benefit_sharing_breach"
    CONSULTATION_DENIAL = "consultation_denial"
    DISCRIMINATORY_POLICY = "discriminatory_policy"


class FPICWorkflowStage(str, Enum):
    """FPIC workflow stage-gate stages.

    Per PRD F7.1: 7-stage workflow plus terminal states.
    """

    IDENTIFICATION = "identification"
    INFORMATION_DISCLOSURE = "information_disclosure"
    CONSULTATION = "consultation"
    CONSENT_DECISION = "consent_decision"
    AGREEMENT = "agreement"
    IMPLEMENTATION = "implementation"
    MONITORING = "monitoring"
    CONSENT_WITHDRAWN = "consent_withdrawn"
    CONSENT_DENIED = "consent_denied"
    CONSENT_DEFERRED = "consent_deferred"


class ConsultationStage(str, Enum):
    """Community consultation lifecycle stages.

    Per PRD F4.1: 7 consultation stages.
    """

    IDENTIFIED = "identified"
    NOTIFIED = "notified"
    INFORMATION_SHARED = "information_shared"
    CONSULTATION_HELD = "consultation_held"
    RESPONSE_RECORDED = "response_recorded"
    AGREEMENT_REACHED = "agreement_reached"
    MONITORING_ACTIVE = "monitoring_active"


class GrievanceStatus(str, Enum):
    """Grievance lifecycle status.

    Per PRD F4.3: 7 grievance states.
    """

    SUBMITTED = "submitted"
    ACKNOWLEDGED = "acknowledged"
    INVESTIGATING = "investigating"
    RESPONDED = "responded"
    RESOLVED = "resolved"
    APPEALED = "appealed"
    CLOSED = "closed"


class AgreementStatus(str, Enum):
    """Benefit-sharing agreement status."""

    DRAFT = "draft"
    ACTIVE = "active"
    EXPIRED = "expired"
    TERMINATED = "terminated"
    RENEWED = "renewed"


class ConfidenceLevel(str, Enum):
    """Territory boundary accuracy confidence level."""

    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class RiskLevel(str, Enum):
    """Overlap and violation risk severity level."""

    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    NONE = "none"


class AlertSeverity(str, Enum):
    """Violation alert severity classification."""

    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class DataSource(str, Enum):
    """Territory data source identifiers."""

    LANDMARK = "landmark"
    RAISG = "raisg"
    FUNAI = "funai"
    BPN_AMAN = "bpn_aman"
    ACHPR = "achpr"
    NATIONAL_REGISTRY = "national_registry"


class ReportType(str, Enum):
    """Compliance report type identifiers."""

    INDIGENOUS_RIGHTS_COMPLIANCE = "indigenous_rights_compliance"
    DDS_SECTION = "dds_section"
    FSC_FPIC = "fsc_fpic"
    RSPO_FPIC = "rspo_fpic"
    SUPPLIER_SCORECARD = "supplier_scorecard"
    TREND_REPORT = "trend_report"
    EXECUTIVE_SUMMARY = "executive_summary"
    BI_EXPORT = "bi_export"


class ReportFormat(str, Enum):
    """Report output format."""

    PDF = "pdf"
    JSON = "json"
    HTML = "html"
    CSV = "csv"
    XLSX = "xlsx"


class CommunityRecognitionStatus(str, Enum):
    """Indigenous community legal recognition status."""

    CONSTITUTIONALLY_RECOGNIZED = "constitutionally_recognized"
    STATUTORY_RECOGNITION = "statutory_recognition"
    CUSTOMARY_ONLY = "customary_only"
    PENDING = "pending"
    DENIED_DISPUTED = "denied_disputed"


class CountryRiskLevel(str, Enum):
    """Country-level indigenous rights risk classification."""

    LOW = "low"
    STANDARD = "standard"
    HIGH = "high"


class ViolationAlertStatus(str, Enum):
    """Violation alert processing status."""

    ACTIVE = "active"
    INVESTIGATING = "investigating"
    RESOLVED = "resolved"
    FALSE_POSITIVE = "false_positive"
    ARCHIVED = "archived"


# ---------------------------------------------------------------------------
# Core Data Models (12)
# ---------------------------------------------------------------------------


class IndigenousTerritory(BaseModel):
    """Indigenous territory record with spatial data.

    Per PRD F1.7: territory with structured metadata including boundary,
    legal status, data source, and provenance hash.

    Example:
        >>> t = IndigenousTerritory(
        ...     territory_id="t-001",
        ...     territory_name="Terra Indigena Yanomami",
        ...     people_name="Yanomami",
        ...     country_code="BR",
        ...     legal_status=TerritoryLegalStatus.TITLED,
        ...     data_source="funai",
        ...     provenance_hash="a" * 64,
        ... )
    """

    model_config = ConfigDict(str_strip_whitespace=True)

    territory_id: str = Field(..., description="UUID territory identifier")
    territory_name: str = Field(..., min_length=1, description="Official name")
    indigenous_name: Optional[str] = Field(
        None, description="Name in indigenous language"
    )
    people_name: str = Field(
        ..., min_length=1, description="Indigenous people/ethnic group"
    )
    country_code: str = Field(
        ..., min_length=2, max_length=2, description="ISO 3166-1 alpha-2"
    )
    region: Optional[str] = Field(None, description="Administrative region")
    area_hectares: Optional[Decimal] = Field(
        None, ge=0, description="Total area in hectares"
    )
    legal_status: TerritoryLegalStatus = Field(
        ..., description="Legal recognition status"
    )
    recognition_date: Optional[date] = Field(
        None, description="Date of formal recognition"
    )
    governing_authority: Optional[str] = Field(
        None, description="National authority responsible"
    )
    boundary_geojson: Optional[Dict[str, Any]] = Field(
        None, description="GeoJSON polygon/multipolygon"
    )
    data_source: str = Field(
        ..., description="Data source identifier"
    )
    source_url: Optional[str] = Field(None, description="Source data URL")
    confidence: ConfidenceLevel = Field(
        default=ConfidenceLevel.MEDIUM,
        description="Boundary accuracy confidence",
    )
    version: int = Field(default=1, ge=1, description="Data version")
    provenance_hash: str = Field(
        ..., description="SHA-256 provenance hash"
    )
    last_verified: Optional[datetime] = Field(
        None, description="Last verification timestamp"
    )
    created_at: Optional[datetime] = Field(
        None, description="Record creation timestamp"
    )


class FPICAssessment(BaseModel):
    """FPIC documentation verification assessment result.

    Per PRD F2.1: 10-element deterministic scoring with Decimal precision.

    Example:
        >>> a = FPICAssessment(
        ...     assessment_id="a-001",
        ...     plot_id="p-001",
        ...     territory_id="t-001",
        ...     fpic_score=Decimal("85.50"),
        ...     fpic_status=FPICStatus.CONSENT_OBTAINED,
        ...     provenance_hash="b" * 64,
        ... )
    """

    model_config = ConfigDict(str_strip_whitespace=True)

    assessment_id: str = Field(..., description="UUID assessment identifier")
    plot_id: str = Field(..., description="Production plot ID")
    territory_id: str = Field(..., description="Overlapping territory ID")
    community_id: Optional[str] = Field(
        None, description="Affected community ID"
    )
    fpic_score: Decimal = Field(
        ..., ge=0, le=100, description="Composite FPIC score (0-100)"
    )
    fpic_status: FPICStatus = Field(..., description="FPIC classification")
    # 10-element breakdown scores
    community_identification_score: Decimal = Field(
        default=Decimal("0"), ge=0, le=100
    )
    information_disclosure_score: Decimal = Field(
        default=Decimal("0"), ge=0, le=100
    )
    prior_timing_score: Decimal = Field(
        default=Decimal("0"), ge=0, le=100
    )
    consultation_process_score: Decimal = Field(
        default=Decimal("0"), ge=0, le=100
    )
    community_representation_score: Decimal = Field(
        default=Decimal("0"), ge=0, le=100
    )
    consent_record_score: Decimal = Field(
        default=Decimal("0"), ge=0, le=100
    )
    absence_of_coercion_score: Decimal = Field(
        default=Decimal("0"), ge=0, le=100
    )
    agreement_documentation_score: Decimal = Field(
        default=Decimal("0"), ge=0, le=100
    )
    benefit_sharing_score: Decimal = Field(
        default=Decimal("0"), ge=0, le=100
    )
    monitoring_provisions_score: Decimal = Field(
        default=Decimal("0"), ge=0, le=100
    )
    country_specific_rules: Optional[str] = Field(
        None, description="Jurisdiction applied"
    )
    temporal_compliance: bool = Field(
        default=False, description="Consent prior to production"
    )
    coercion_flags: List[str] = Field(
        default_factory=list, description="Detected coercion indicators"
    )
    validity_start: Optional[date] = Field(None)
    validity_end: Optional[date] = Field(None)
    decision_rationale: Optional[str] = Field(None)
    provenance_hash: str = Field(..., description="SHA-256 provenance hash")
    version: int = Field(default=1, ge=1)
    assessed_at: Optional[datetime] = Field(None)


class TerritoryOverlap(BaseModel):
    """Territory overlap detection result.

    Per PRD F3.2: overlap classification with risk scoring.

    Example:
        >>> o = TerritoryOverlap(
        ...     overlap_id="o-001",
        ...     plot_id="p-001",
        ...     territory_id="t-001",
        ...     overlap_type=OverlapType.DIRECT,
        ...     distance_meters=Decimal("0"),
        ...     risk_score=Decimal("92.5"),
        ...     risk_level=RiskLevel.CRITICAL,
        ...     provenance_hash="c" * 64,
        ... )
    """

    model_config = ConfigDict(str_strip_whitespace=True)

    overlap_id: str = Field(..., description="UUID overlap identifier")
    plot_id: str = Field(..., description="Production plot ID")
    territory_id: str = Field(..., description="Overlapping territory ID")
    overlap_type: OverlapType = Field(..., description="Overlap classification")
    overlap_area_hectares: Optional[Decimal] = Field(
        None, ge=0, description="Overlap area for DIRECT/PARTIAL"
    )
    overlap_pct_of_plot: Optional[Decimal] = Field(
        None, ge=0, le=100, description="% of plot area"
    )
    overlap_pct_of_territory: Optional[Decimal] = Field(
        None, ge=0, le=100, description="% of territory area"
    )
    distance_meters: Decimal = Field(
        ..., ge=0, description="Min distance to territory boundary"
    )
    bearing_degrees: Optional[Decimal] = Field(
        None, ge=0, le=360, description="Direction to nearest boundary"
    )
    affected_communities: List[str] = Field(
        default_factory=list, description="Affected community IDs"
    )
    risk_score: Decimal = Field(
        ..., ge=0, le=100, description="Overlap risk score (0-100)"
    )
    risk_level: RiskLevel = Field(..., description="Risk severity level")
    deforestation_correlation: bool = Field(
        default=False, description="Cross-ref with EUDR-020"
    )
    provenance_hash: str = Field(..., description="SHA-256 provenance hash")
    detected_at: Optional[datetime] = Field(None)


class IndigenousCommunity(BaseModel):
    """Indigenous community registry record.

    Per PRD F6.1: structured community profile with legal protections.
    """

    model_config = ConfigDict(str_strip_whitespace=True)

    community_id: str = Field(..., description="UUID community identifier")
    community_name: str = Field(
        ..., min_length=1, description="Official community name"
    )
    indigenous_name: Optional[str] = Field(
        None, description="Name in indigenous language"
    )
    people_name: str = Field(..., min_length=1, description="People/ethnic group")
    language: Optional[str] = Field(None, description="Primary language")
    estimated_population: Optional[int] = Field(None, ge=0)
    country_code: str = Field(..., min_length=2, max_length=2)
    region: Optional[str] = Field(None)
    territory_ids: List[str] = Field(
        default_factory=list, description="Linked territory IDs"
    )
    legal_recognition_status: Optional[CommunityRecognitionStatus] = Field(None)
    applicable_legal_protections: List[str] = Field(default_factory=list)
    ilo_169_coverage: bool = Field(default=False)
    fpic_legal_requirement: bool = Field(default=False)
    representative_organizations: List[Dict[str, Any]] = Field(
        default_factory=list
    )
    contact_channels: List[Dict[str, Any]] = Field(default_factory=list)
    commodity_relevance: List[str] = Field(default_factory=list)
    engagement_history_summary: Dict[str, Any] = Field(default_factory=dict)
    data_source: Optional[str] = Field(None)
    provenance_hash: str = Field(..., description="SHA-256 provenance hash")
    created_at: Optional[datetime] = Field(None)
    updated_at: Optional[datetime] = Field(None)


class ConsultationRecord(BaseModel):
    """Community consultation activity record.

    Per PRD F4.1-F4.2: meeting details with attendees and outcomes.
    """

    model_config = ConfigDict(str_strip_whitespace=True)

    consultation_id: str = Field(...)
    community_id: str = Field(...)
    plot_id: Optional[str] = Field(None)
    territory_id: Optional[str] = Field(None)
    consultation_stage: ConsultationStage = Field(...)
    meeting_date: Optional[date] = Field(None)
    meeting_location: Optional[str] = Field(None)
    attendees: List[Dict[str, Any]] = Field(default_factory=list)
    agenda: Optional[str] = Field(None)
    minutes: Optional[str] = Field(None)
    outcomes: Optional[str] = Field(None)
    follow_up_actions: List[Dict[str, Any]] = Field(default_factory=list)
    documents_shared: List[Dict[str, Any]] = Field(default_factory=list)
    community_response: Optional[str] = Field(None)
    grievance_id: Optional[str] = Field(None)
    provenance_hash: str = Field(...)
    created_at: Optional[datetime] = Field(None)


class GrievanceRecord(BaseModel):
    """Community grievance record.

    Per PRD F4.3: grievance lifecycle with SLA tracking.
    """

    model_config = ConfigDict(str_strip_whitespace=True)

    grievance_id: str = Field(...)
    community_id: str = Field(...)
    territory_id: Optional[str] = Field(None)
    grievance_type: str = Field(..., min_length=1)
    description: str = Field(..., min_length=1)
    severity: AlertSeverity = Field(...)
    status: GrievanceStatus = Field(default=GrievanceStatus.SUBMITTED)
    submitted_at: Optional[datetime] = Field(None)
    acknowledged_at: Optional[datetime] = Field(None)
    investigation_deadline: Optional[datetime] = Field(None)
    resolution_deadline: Optional[datetime] = Field(None)
    response: Optional[str] = Field(None)
    resolution: Optional[str] = Field(None)
    resolved_at: Optional[datetime] = Field(None)
    sla_compliant: Optional[bool] = Field(None)
    provenance_hash: str = Field(...)


class BenefitSharingAgreement(BaseModel):
    """Benefit-sharing agreement record.

    Per PRD F4.4: agreement terms and compliance tracking.
    """

    model_config = ConfigDict(str_strip_whitespace=True)

    agreement_id: str = Field(...)
    community_id: str = Field(...)
    territory_id: Optional[str] = Field(None)
    operator_id: str = Field(...)
    agreement_type: str = Field(..., min_length=1)
    terms_summary: str = Field(..., min_length=1)
    monetary_benefits: Dict[str, Any] = Field(default_factory=dict)
    non_monetary_benefits: List[Dict[str, Any]] = Field(default_factory=list)
    effective_date: date = Field(...)
    expiry_date: Optional[date] = Field(None)
    renewal_required: bool = Field(default=True)
    status: AgreementStatus = Field(default=AgreementStatus.ACTIVE)
    compliance_status: str = Field(default="compliant")
    provenance_hash: str = Field(...)
    created_at: Optional[datetime] = Field(None)
    updated_at: Optional[datetime] = Field(None)


class FPICWorkflow(BaseModel):
    """FPIC workflow instance.

    Per PRD F7.1: 7-stage workflow with SLA tracking.
    """

    model_config = ConfigDict(str_strip_whitespace=True)

    workflow_id: str = Field(...)
    plot_id: str = Field(...)
    territory_id: str = Field(...)
    community_id: str = Field(...)
    current_stage: FPICWorkflowStage = Field(...)
    stage_history: List[Dict[str, Any]] = Field(default_factory=list)
    sla_status: str = Field(default="on_track")
    next_deadline: Optional[datetime] = Field(None)
    escalation_level: int = Field(default=0, ge=0)
    consent_decision: Optional[str] = Field(None)
    agreement_id: Optional[str] = Field(None)
    validity_end: Optional[date] = Field(None)
    provenance_hash: str = Field(...)
    created_at: Optional[datetime] = Field(None)
    updated_at: Optional[datetime] = Field(None)


class WorkflowTransition(BaseModel):
    """FPIC workflow state transition record.

    Per PRD F7.7: immutable transition log with provenance.
    """

    model_config = ConfigDict(str_strip_whitespace=True)

    transition_id: str = Field(...)
    workflow_id: str = Field(...)
    from_stage: str = Field(...)
    to_stage: str = Field(...)
    actor: str = Field(...)
    reason: Optional[str] = Field(None)
    supporting_evidence: List[Dict[str, Any]] = Field(default_factory=list)
    provenance_hash: str = Field(...)
    transitioned_at: Optional[datetime] = Field(None)


class ViolationAlert(BaseModel):
    """Indigenous rights violation alert.

    Per PRD F5.2: structured violation report with severity scoring.
    """

    model_config = ConfigDict(str_strip_whitespace=True)

    alert_id: str = Field(...)
    source: str = Field(..., min_length=1)
    source_url: Optional[str] = Field(None)
    source_document_hash: Optional[str] = Field(None)
    publication_date: date = Field(...)
    violation_type: ViolationType = Field(...)
    country_code: str = Field(..., min_length=2, max_length=2)
    region: Optional[str] = Field(None)
    location_lat: Optional[float] = Field(None, ge=-90, le=90)
    location_lon: Optional[float] = Field(None, ge=-180, le=180)
    affected_communities: List[str] = Field(default_factory=list)
    severity_score: Decimal = Field(..., ge=0, le=100)
    severity_level: AlertSeverity = Field(...)
    supply_chain_correlation: bool = Field(default=False)
    affected_plots: List[str] = Field(default_factory=list)
    affected_suppliers: List[str] = Field(default_factory=list)
    impact_assessment: Dict[str, Any] = Field(default_factory=dict)
    deduplication_group: Optional[str] = Field(None)
    status: ViolationAlertStatus = Field(default=ViolationAlertStatus.ACTIVE)
    provenance_hash: str = Field(...)
    detected_at: Optional[datetime] = Field(None)


class ComplianceReport(BaseModel):
    """Indigenous rights compliance report metadata.

    Per PRD F8.1: report with format, language, and provenance.
    """

    model_config = ConfigDict(str_strip_whitespace=True)

    report_id: str = Field(...)
    report_type: ReportType = Field(...)
    title: str = Field(..., min_length=1)
    format: ReportFormat = Field(...)
    language: str = Field(default="en")
    scope_type: str = Field(...)
    scope_ids: List[str] = Field(default_factory=list)
    parameters: Dict[str, Any] = Field(default_factory=dict)
    file_path: Optional[str] = Field(None)
    file_size_bytes: Optional[int] = Field(None, ge=0)
    provenance_hash: Optional[str] = Field(None)
    generated_by: Optional[str] = Field(None)
    generated_at: Optional[datetime] = Field(None)


class CountryIndigenousRightsScore(BaseModel):
    """Country-level indigenous rights protection score.

    Per PRD Section 7.4 table 12: composite scoring from multiple dimensions.
    """

    model_config = ConfigDict(str_strip_whitespace=True)

    score_id: str = Field(...)
    country_code: str = Field(..., min_length=2, max_length=2)
    ilo_169_ratified: bool = Field(default=False)
    ilo_169_ratification_date: Optional[date] = Field(None)
    fpic_legal_requirement: bool = Field(default=False)
    land_tenure_security_score: Decimal = Field(..., ge=0, le=100)
    indigenous_rights_recognition_score: Decimal = Field(..., ge=0, le=100)
    judicial_protection_score: Decimal = Field(..., ge=0, le=100)
    territory_demarcation_pct: Optional[Decimal] = Field(None, ge=0, le=100)
    active_land_conflicts: int = Field(default=0, ge=0)
    composite_indigenous_rights_score: Decimal = Field(..., ge=0, le=100)
    risk_level: CountryRiskLevel = Field(...)
    data_sources: List[str] = Field(default_factory=list)
    provenance_hash: str = Field(...)
    assessed_at: Optional[datetime] = Field(None)


class AuditLogEntry(BaseModel):
    """Immutable audit log entry.

    Per PRD Section 7.4 table 14: audit trail with before/after states.
    """

    model_config = ConfigDict(str_strip_whitespace=True)

    log_id: str = Field(...)
    action: str = Field(...)
    entity_type: str = Field(...)
    entity_id: str = Field(...)
    actor: str = Field(...)
    details: Dict[str, Any] = Field(default_factory=dict)
    previous_state: Optional[Dict[str, Any]] = Field(None)
    new_state: Optional[Dict[str, Any]] = Field(None)
    ip_address: Optional[str] = Field(None)
    provenance_hash: str = Field(...)
    created_at: Optional[datetime] = Field(None)


# ---------------------------------------------------------------------------
# Request Models
# ---------------------------------------------------------------------------


class DetectOverlapRequest(BaseModel):
    """Request to detect territory overlap for a single plot."""

    model_config = ConfigDict(str_strip_whitespace=True)

    plot_id: str = Field(...)
    latitude: Optional[float] = Field(None, ge=-90, le=90)
    longitude: Optional[float] = Field(None, ge=-180, le=180)
    plot_geojson: Optional[Dict[str, Any]] = Field(None)
    inner_buffer_km: Optional[float] = Field(None, gt=0)
    outer_buffer_km: Optional[float] = Field(None, gt=0)


class BatchOverlapRequest(BaseModel):
    """Request for batch overlap screening of multiple plots."""

    model_config = ConfigDict(str_strip_whitespace=True)

    plots: List[DetectOverlapRequest] = Field(
        ..., min_length=1, max_length=MAX_BATCH_SIZE
    )


class VerifyFPICRequest(BaseModel):
    """Request to verify FPIC documentation for a plot-territory pair."""

    model_config = ConfigDict(str_strip_whitespace=True)

    plot_id: str = Field(...)
    territory_id: str = Field(...)
    community_id: Optional[str] = Field(None)
    documentation: Dict[str, Any] = Field(default_factory=dict)
    production_start_date: Optional[date] = Field(None)
    country_code: Optional[str] = Field(None)


class CreateWorkflowRequest(BaseModel):
    """Request to create a new FPIC workflow."""

    model_config = ConfigDict(str_strip_whitespace=True)

    plot_id: str = Field(...)
    territory_id: str = Field(...)
    community_id: str = Field(...)
    initiator: str = Field(default="system")


class AdvanceWorkflowRequest(BaseModel):
    """Request to advance a FPIC workflow to the next stage."""

    model_config = ConfigDict(str_strip_whitespace=True)

    actor: str = Field(...)
    reason: Optional[str] = Field(None)
    supporting_evidence: List[Dict[str, Any]] = Field(default_factory=list)


class GenerateReportRequest(BaseModel):
    """Request to generate a compliance report."""

    model_config = ConfigDict(str_strip_whitespace=True)

    report_type: ReportType = Field(...)
    format: ReportFormat = Field(default=ReportFormat.PDF)
    language: str = Field(default="en")
    scope_type: str = Field(...)
    scope_ids: List[str] = Field(default_factory=list)
    parameters: Dict[str, Any] = Field(default_factory=dict)


class CorrelateViolationsRequest(BaseModel):
    """Request to correlate violations with supply chain."""

    model_config = ConfigDict(str_strip_whitespace=True)

    operator_id: Optional[str] = Field(None)
    plot_ids: List[str] = Field(default_factory=list)
    country_codes: List[str] = Field(default_factory=list)
    max_distance_km: float = Field(default=25.0, gt=0)


class RecordConsultationRequest(BaseModel):
    """Request to record a consultation activity."""

    model_config = ConfigDict(str_strip_whitespace=True)

    community_id: str = Field(...)
    plot_id: Optional[str] = Field(None)
    territory_id: Optional[str] = Field(None)
    consultation_stage: ConsultationStage = Field(...)
    meeting_date: Optional[date] = Field(None)
    meeting_location: Optional[str] = Field(None)
    attendees: List[Dict[str, Any]] = Field(default_factory=list)
    agenda: Optional[str] = Field(None)
    minutes: Optional[str] = Field(None)
    outcomes: Optional[str] = Field(None)


class SubmitGrievanceRequest(BaseModel):
    """Request to submit a community grievance."""

    model_config = ConfigDict(str_strip_whitespace=True)

    community_id: str = Field(...)
    territory_id: Optional[str] = Field(None)
    grievance_type: str = Field(..., min_length=1)
    description: str = Field(..., min_length=1)
    severity: AlertSeverity = Field(default=AlertSeverity.MEDIUM)


class HealthCheckRequest(BaseModel):
    """Health check request."""

    include_details: bool = Field(default=False)


# ---------------------------------------------------------------------------
# Response Models
# ---------------------------------------------------------------------------


class OverlapDetectionResponse(BaseModel):
    """Response from overlap detection."""

    model_config = ConfigDict(str_strip_whitespace=True)

    plot_id: str = Field(...)
    overlaps: List[TerritoryOverlap] = Field(default_factory=list)
    total_overlaps: int = Field(default=0)
    highest_risk_level: RiskLevel = Field(default=RiskLevel.NONE)
    processing_time_ms: float = Field(default=0.0)
    provenance_hash: str = Field(...)


class BatchOverlapResponse(BaseModel):
    """Response from batch overlap screening."""

    model_config = ConfigDict(str_strip_whitespace=True)

    total_plots: int = Field(...)
    plots_with_overlaps: int = Field(default=0)
    critical_count: int = Field(default=0)
    high_count: int = Field(default=0)
    medium_count: int = Field(default=0)
    low_count: int = Field(default=0)
    results: List[OverlapDetectionResponse] = Field(default_factory=list)
    processing_time_ms: float = Field(default=0.0)
    provenance_hash: str = Field(...)


class FPICVerificationResponse(BaseModel):
    """Response from FPIC verification."""

    model_config = ConfigDict(str_strip_whitespace=True)

    assessment: FPICAssessment = Field(...)
    processing_time_ms: float = Field(default=0.0)


class WorkflowStatusResponse(BaseModel):
    """Response with FPIC workflow status."""

    model_config = ConfigDict(str_strip_whitespace=True)

    workflow: FPICWorkflow = Field(...)
    transitions: List[WorkflowTransition] = Field(default_factory=list)


class ViolationCorrelationResponse(BaseModel):
    """Response from violation-supply chain correlation."""

    model_config = ConfigDict(str_strip_whitespace=True)

    total_violations: int = Field(default=0)
    correlated_violations: int = Field(default=0)
    alerts: List[ViolationAlert] = Field(default_factory=list)
    processing_time_ms: float = Field(default=0.0)
    provenance_hash: str = Field(...)


class HealthCheckResponse(BaseModel):
    """Health check response."""

    model_config = ConfigDict(str_strip_whitespace=True)

    status: str = Field(...)
    agent_id: str = Field(default="GL-EUDR-IRC-021")
    version: str = Field(default=VERSION)
    territory_count: Optional[int] = Field(None)
    community_count: Optional[int] = Field(None)
    active_workflows: Optional[int] = Field(None)
    active_violations: Optional[int] = Field(None)
    database_connected: bool = Field(default=False)
    redis_connected: bool = Field(default=False)
