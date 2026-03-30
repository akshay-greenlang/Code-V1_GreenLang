# -*- coding: utf-8 -*-
"""
API Schemas - AGENT-EUDR-024 Third-Party Audit Manager

Pydantic v2 request/response models for the REST API layer covering all
10 route domains: audit planning, auditor registry, audit execution,
non-conformance management, CAR lifecycle, certification schemes,
report generation, authority liaison, analytics, and admin.

All numeric fields use ``Decimal`` for precision (zero-hallucination).
All date/time fields use UTC-aware ``datetime``.

Schema Groups (10 domains + common):
    Common: ProvenanceInfo, MetadataSchema, PaginatedMeta, ErrorResponse,
            HealthResponse
    1. Planning: ScheduleGenerateRequest/Response, ScheduleListResponse,
       ScheduleUpdateRequest, TriggerAuditRequest/Response,
       AuditCreateRequest/Response, AuditListResponse, AuditDetailResponse,
       AuditUpdateRequest
    2. Auditor: AuditorRegisterRequest/Response, AuditorListResponse,
       AuditorDetailResponse, AuditorUpdateRequest, AuditorMatchRequest/Response
    3. Execution: ChecklistResponse, CriterionUpdateRequest/Response,
       EvidenceUploadRequest/Response, EvidenceListResponse, ProgressResponse
    4. NC: NCCreateRequest/Response, NCListResponse, NCDetailResponse,
       NCUpdateRequest, RCASubmitRequest/Response
    5. CAR: CARIssueRequest/Response, CARListResponse, CARDetailResponse,
       CARUpdateRequest, CARVerifyRequest/Response
    6. Scheme: CertificateListResponse, CertSyncRequest/Response,
       CoverageMatrixResponse
    7. Report: ReportGenerateRequest/Response, ReportDownloadResponse
    8. Authority: InteractionCreateRequest/Response, InteractionListResponse,
       InteractionUpdateRequest
    9. Analytics: FindingTrendsResponse, AuditorPerformanceResponse,
       ComplianceRatesResponse, CARPerformanceResponse, DashboardResponse
   10. Admin: HealthResponse, StatsResponse

Author: GreenLang Platform Team
Date: March 2026
PRD: AGENT-EUDR-024 Third-Party Audit Manager (GL-EUDR-TAM-024)
"""

from __future__ import annotations

import uuid
from datetime import date, datetime, timezone
from decimal import Decimal
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import ConfigDict, Field, field_validator

from greenlang.schemas import GreenLangBase, utcnow

def _new_id() -> str:
    """Generate a new UUID4 string identifier."""
    return str(uuid.uuid4())

# =============================================================================
# Enumerations (API-level mirrors for OpenAPI documentation)
# =============================================================================

class AuditTypeEnum(str, Enum):
    """Audit type classification."""
    FULL = "full"
    TARGETED = "targeted"
    SURVEILLANCE = "surveillance"
    UNSCHEDULED = "unscheduled"

class AuditModalityEnum(str, Enum):
    """Audit modality."""
    ON_SITE = "on_site"
    REMOTE = "remote"
    HYBRID = "hybrid"
    UNANNOUNCED = "unannounced"

class AuditStatusEnum(str, Enum):
    """Audit lifecycle status."""
    PLANNED = "planned"
    AUDITOR_ASSIGNED = "auditor_assigned"
    IN_PREPARATION = "in_preparation"
    IN_PROGRESS = "in_progress"
    FIELDWORK_COMPLETE = "fieldwork_complete"
    REPORT_DRAFTING = "report_drafting"
    REPORT_ISSUED = "report_issued"
    CAR_FOLLOW_UP = "car_follow_up"
    CLOSED = "closed"
    CANCELLED = "cancelled"

class NCSeverityEnum(str, Enum):
    """Non-conformance severity levels."""
    CRITICAL = "critical"
    MAJOR = "major"
    MINOR = "minor"
    OBSERVATION = "observation"

class NCStatusEnum(str, Enum):
    """NC lifecycle status."""
    OPEN = "open"
    ACKNOWLEDGED = "acknowledged"
    CAR_ISSUED = "car_issued"
    CAP_SUBMITTED = "cap_submitted"
    IN_PROGRESS = "in_progress"
    VERIFICATION_PENDING = "verification_pending"
    CLOSED = "closed"
    ESCALATED = "escalated"
    DISPUTED = "disputed"

class CARStatusEnum(str, Enum):
    """CAR lifecycle status."""
    ISSUED = "issued"
    ACKNOWLEDGED = "acknowledged"
    RCA_SUBMITTED = "rca_submitted"
    CAP_SUBMITTED = "cap_submitted"
    CAP_APPROVED = "cap_approved"
    IN_PROGRESS = "in_progress"
    EVIDENCE_SUBMITTED = "evidence_submitted"
    VERIFICATION_PENDING = "verification_pending"
    CLOSED = "closed"
    REJECTED = "rejected"
    OVERDUE = "overdue"
    ESCALATED = "escalated"

class CARSLAStatusEnum(str, Enum):
    """CAR SLA tracking status."""
    ON_TRACK = "on_track"
    WARNING = "warning"
    CRITICAL = "critical"
    OVERDUE = "overdue"

class CertSchemeEnum(str, Enum):
    """Supported certification schemes."""
    FSC = "fsc"
    PEFC = "pefc"
    RSPO = "rspo"
    RAINFOREST_ALLIANCE = "rainforest_alliance"
    ISCC = "iscc"

class CertStatusEnum(str, Enum):
    """Certificate status."""
    ACTIVE = "active"
    SUSPENDED = "suspended"
    TERMINATED = "terminated"
    EXPIRED = "expired"

class AccreditationStatusEnum(str, Enum):
    """Auditor accreditation status."""
    ACTIVE = "active"
    SUSPENDED = "suspended"
    WITHDRAWN = "withdrawn"
    EXPIRED = "expired"

class AuditorRoleEnum(str, Enum):
    """Audit team member roles."""
    LEAD_AUDITOR = "lead_auditor"
    CO_AUDITOR = "co_auditor"
    TECHNICAL_EXPERT = "technical_expert"
    OBSERVER = "observer"
    TRAINEE = "trainee"

class EvidenceTypeEnum(str, Enum):
    """Audit evidence types."""
    PERMIT = "permit"
    CERTIFICATE = "certificate"
    PHOTO = "photo"
    GPS_RECORD = "gps_record"
    INTERVIEW_TRANSCRIPT = "interview_transcript"
    LAB_RESULT = "lab_result"
    DOCUMENT_SCAN = "document_scan"
    OTHER = "other"

class CriterionResultEnum(str, Enum):
    """Checklist criterion assessment result."""
    PASS = "pass"
    FAIL = "fail"
    NA = "na"

class RootCauseMethodEnum(str, Enum):
    """Root cause analysis methods."""
    FIVE_WHYS = "five_whys"
    ISHIKAWA = "ishikawa"
    DIRECT = "direct"

class VerificationOutcomeEnum(str, Enum):
    """CAR verification outcome."""
    EFFECTIVE = "effective"
    NOT_EFFECTIVE = "not_effective"

class InteractionTypeEnum(str, Enum):
    """Competent authority interaction types."""
    DOCUMENT_REQUEST = "document_request"
    INSPECTION_NOTIFICATION = "inspection_notification"
    UNANNOUNCED_INSPECTION = "unannounced_inspection"
    CORRECTIVE_ACTION_ORDER = "corrective_action_order"
    INTERIM_MEASURE = "interim_measure"
    DEFINITIVE_MEASURE = "definitive_measure"
    INFORMATION_REQUEST = "information_request"

class InteractionStatusEnum(str, Enum):
    """Authority interaction status."""
    OPEN = "open"
    IN_PROGRESS = "in_progress"
    RESPONDED = "responded"
    CLOSED = "closed"

class ResponseSLAStatusEnum(str, Enum):
    """Authority response SLA status."""
    ON_TRACK = "on_track"
    WARNING = "warning"
    CRITICAL = "critical"
    OVERDUE = "overdue"

class ReportFormatEnum(str, Enum):
    """Report output formats."""
    PDF = "pdf"
    JSON = "json"
    HTML = "html"
    XLSX = "xlsx"
    XML = "xml"

class ReportLanguageEnum(str, Enum):
    """Report language options."""
    EN = "en"
    FR = "fr"
    DE = "de"
    ES = "es"
    PT = "pt"

class ScheduleStatusEnum(str, Enum):
    """Audit schedule entry status."""
    PLANNED = "planned"
    CONFIRMED = "confirmed"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    RESCHEDULED = "rescheduled"

class EUDRCommodityEnum(str, Enum):
    """EUDR regulated commodities."""
    CATTLE = "cattle"
    COCOA = "cocoa"
    COFFEE = "coffee"
    PALM_OIL = "palm_oil"
    RUBBER = "rubber"
    SOYA = "soya"
    WOOD = "wood"

class InterviewTypeEnum(str, Enum):
    """Stakeholder interview types."""
    COMMUNITY = "community"
    WORKER = "worker"
    MANAGEMENT = "management"
    GOVERNMENT = "government"
    OTHER = "other"

class FrequencyTierEnum(str, Enum):
    """Audit frequency tiers."""
    HIGH = "high"
    STANDARD = "standard"
    LOW = "low"

class AuditScopeEnum(str, Enum):
    """Audit scope types."""
    FULL = "full"
    TARGETED = "targeted"
    SURVEILLANCE = "surveillance"
    UNSCHEDULED = "unscheduled"

# =============================================================================
# Common Schemas
# =============================================================================

class ProvenanceInfo(GreenLangBase):
    """SHA-256 provenance tracking for zero-hallucination guarantee."""
    model_config = ConfigDict(frozen=True)

    provenance_hash: str = Field(..., description="SHA-256 hash of input+output")
    processing_time_ms: Decimal = Field(
        default=Decimal("0.00"),
        description="Processing time in milliseconds",
    )
    chain_hash: Optional[str] = Field(
        default=None, description="Previous hash in provenance chain"
    )

class MetadataSchema(GreenLangBase):
    """Standard response metadata."""
    model_config = ConfigDict(frozen=True)

    data_sources: List[str] = Field(
        default_factory=list, description="Data sources used"
    )
    agent_id: str = Field(
        default="GL-EUDR-TAM-024",
        description="Agent identifier",
    )
    agent_version: str = Field(
        default="1.0.0", description="Agent version"
    )
    timestamp: datetime = Field(
        default_factory=utcnow, description="Response timestamp (UTC)"
    )

class PaginatedMeta(GreenLangBase):
    """Pagination metadata for list responses."""
    model_config = ConfigDict(frozen=True)

    total: int = Field(..., description="Total matching records")
    limit: int = Field(..., description="Page size")
    offset: int = Field(..., description="Current offset")
    has_more: bool = Field(..., description="Whether more pages exist")

class ErrorResponse(GreenLangBase):
    """Standard error response."""
    model_config = ConfigDict(frozen=True)

    detail: str = Field(..., description="Error description")
    error_code: Optional[str] = Field(
        default=None, description="Machine-readable error code"
    )
    request_id: Optional[str] = Field(
        default=None, description="Request trace ID"
    )

class HealthResponse(GreenLangBase):
    """Health check response."""
    model_config = ConfigDict(frozen=True)

    status: str = Field(default="healthy", description="Service health status")
    agent_id: str = Field(
        default="GL-EUDR-TAM-024", description="Agent identifier"
    )
    agent: str = Field(default="EUDR-024", description="Agent short name")
    component: str = Field(
        default="third-party-audit-manager", description="Component name"
    )
    version: str = Field(default="1.0.0", description="Agent version")

# =============================================================================
# 1. Planning Schemas
# =============================================================================

class ScheduleGenerateRequest(GreenLangBase):
    """Request to generate risk-based audit schedule."""
    operator_id: str = Field(..., description="Operator UUID")
    planning_quarter: Optional[str] = Field(
        default=None, description="Target quarter (e.g. 2026-Q2)"
    )
    force_recalculate: bool = Field(
        default=False, description="Force recalculation of priority scores"
    )
    supplier_ids: Optional[List[str]] = Field(
        default=None, description="Specific suppliers to schedule (optional)"
    )

class ScheduleEntry(GreenLangBase):
    """Single audit schedule entry."""
    schedule_id: str = Field(default_factory=_new_id)
    operator_id: str = Field(...)
    supplier_id: str = Field(...)
    planned_quarter: str = Field(...)
    audit_type: AuditTypeEnum = Field(...)
    modality: AuditModalityEnum = Field(default=AuditModalityEnum.ON_SITE)
    priority_score: Decimal = Field(default=Decimal("0.00"))
    frequency_tier: FrequencyTierEnum = Field(default=FrequencyTierEnum.STANDARD)
    certification_scheme: Optional[str] = Field(default=None)
    status: ScheduleStatusEnum = Field(default=ScheduleStatusEnum.PLANNED)
    risk_factors: Dict[str, Any] = Field(default_factory=dict)
    scheduled_at: datetime = Field(default_factory=utcnow)

class ScheduleGenerateResponse(GreenLangBase):
    """Response from schedule generation."""
    schedule_entries: List[ScheduleEntry] = Field(default_factory=list)
    total_scheduled: int = Field(default=0)
    high_priority_count: int = Field(default=0)
    standard_priority_count: int = Field(default=0)
    low_priority_count: int = Field(default=0)
    provenance: ProvenanceInfo
    metadata: MetadataSchema = Field(default_factory=MetadataSchema)

class ScheduleListResponse(GreenLangBase):
    """Paginated schedule list."""
    schedules: List[ScheduleEntry] = Field(default_factory=list)
    pagination: PaginatedMeta
    provenance: ProvenanceInfo
    metadata: MetadataSchema = Field(default_factory=MetadataSchema)

class ScheduleUpdateRequest(GreenLangBase):
    """Request to update a schedule entry."""
    planned_quarter: Optional[str] = Field(default=None)
    audit_type: Optional[AuditTypeEnum] = Field(default=None)
    modality: Optional[AuditModalityEnum] = Field(default=None)
    assigned_auditor_id: Optional[str] = Field(default=None)
    status: Optional[ScheduleStatusEnum] = Field(default=None)

class TriggerAuditRequest(GreenLangBase):
    """Request to trigger an unscheduled audit."""
    operator_id: str = Field(..., description="Operator UUID")
    supplier_id: str = Field(..., description="Supplier UUID")
    trigger_reason: str = Field(..., description="Reason for unscheduled audit")
    audit_type: AuditTypeEnum = Field(default=AuditTypeEnum.UNSCHEDULED)
    modality: AuditModalityEnum = Field(default=AuditModalityEnum.ON_SITE)
    scope: AuditScopeEnum = Field(default=AuditScopeEnum.FULL)
    country_code: str = Field(..., min_length=2, max_length=2)
    commodity: EUDRCommodityEnum = Field(...)

class TriggerAuditResponse(GreenLangBase):
    """Response from audit trigger."""
    audit_id: str = Field(default_factory=_new_id)
    schedule_id: Optional[str] = Field(default=None)
    status: str = Field(default="scheduled")
    priority_score: Decimal = Field(default=Decimal("0.00"))
    provenance: ProvenanceInfo
    metadata: MetadataSchema = Field(default_factory=MetadataSchema)

class AuditCreateRequest(GreenLangBase):
    """Request to create a new audit."""
    operator_id: str = Field(..., description="Operator UUID")
    supplier_id: str = Field(..., description="Supplier UUID")
    audit_type: AuditTypeEnum = Field(default=AuditTypeEnum.FULL)
    modality: AuditModalityEnum = Field(default=AuditModalityEnum.ON_SITE)
    certification_scheme: Optional[CertSchemeEnum] = Field(default=None)
    planned_date: date = Field(..., description="Planned audit date")
    country_code: str = Field(..., min_length=2, max_length=2)
    commodity: EUDRCommodityEnum = Field(...)
    eudr_articles: List[str] = Field(default_factory=list)
    site_ids: List[str] = Field(default_factory=list)
    trigger_reason: Optional[str] = Field(default=None)

class AuditEntry(GreenLangBase):
    """Single audit record."""
    audit_id: str = Field(default_factory=_new_id)
    operator_id: str = Field(...)
    supplier_id: str = Field(...)
    audit_type: AuditTypeEnum = Field(...)
    modality: AuditModalityEnum = Field(default=AuditModalityEnum.ON_SITE)
    certification_scheme: Optional[str] = Field(default=None)
    planned_date: date = Field(...)
    actual_start_date: Optional[date] = Field(default=None)
    actual_end_date: Optional[date] = Field(default=None)
    lead_auditor_id: Optional[str] = Field(default=None)
    status: AuditStatusEnum = Field(default=AuditStatusEnum.PLANNED)
    priority_score: Decimal = Field(default=Decimal("0.00"))
    country_code: str = Field(...)
    commodity: str = Field(...)
    checklist_completion: Decimal = Field(default=Decimal("0.00"))
    findings_count: Dict[str, int] = Field(
        default_factory=lambda: {
            "critical": 0, "major": 0, "minor": 0, "observation": 0,
        }
    )
    evidence_count: int = Field(default=0)
    provenance_hash: str = Field(default="")
    created_at: datetime = Field(default_factory=utcnow)
    updated_at: datetime = Field(default_factory=utcnow)

class AuditCreateResponse(GreenLangBase):
    """Response from audit creation."""
    audit: AuditEntry
    provenance: ProvenanceInfo
    metadata: MetadataSchema = Field(default_factory=MetadataSchema)

class AuditListResponse(GreenLangBase):
    """Paginated audit list."""
    audits: List[AuditEntry] = Field(default_factory=list)
    pagination: PaginatedMeta
    provenance: ProvenanceInfo
    metadata: MetadataSchema = Field(default_factory=MetadataSchema)

class AuditDetailResponse(GreenLangBase):
    """Detailed audit response."""
    audit: AuditEntry
    team_assignments: List[Dict[str, Any]] = Field(default_factory=list)
    checklists: List[Dict[str, Any]] = Field(default_factory=list)
    recent_ncs: List[Dict[str, Any]] = Field(default_factory=list)
    provenance: ProvenanceInfo
    metadata: MetadataSchema = Field(default_factory=MetadataSchema)

class AuditUpdateRequest(GreenLangBase):
    """Request to update an audit."""
    status: Optional[AuditStatusEnum] = Field(default=None)
    actual_start_date: Optional[date] = Field(default=None)
    actual_end_date: Optional[date] = Field(default=None)
    lead_auditor_id: Optional[str] = Field(default=None)
    audit_team: Optional[List[Dict[str, str]]] = Field(default=None)
    modality: Optional[AuditModalityEnum] = Field(default=None)

# =============================================================================
# 2. Auditor Schemas
# =============================================================================

class AuditorRegisterRequest(GreenLangBase):
    """Request to register a new auditor."""
    full_name: str = Field(..., min_length=1, max_length=500)
    organization: str = Field(..., min_length=1, max_length=500)
    accreditation_body: Optional[str] = Field(default=None)
    accreditation_status: AccreditationStatusEnum = Field(
        default=AccreditationStatusEnum.ACTIVE
    )
    accreditation_expiry: Optional[date] = Field(default=None)
    accreditation_scope: List[str] = Field(default_factory=list)
    commodity_competencies: List[EUDRCommodityEnum] = Field(default_factory=list)
    scheme_qualifications: List[CertSchemeEnum] = Field(default_factory=list)
    country_expertise: List[str] = Field(default_factory=list)
    languages: List[str] = Field(default_factory=list)
    contact_email: Optional[str] = Field(default=None)

class AuditorEntry(GreenLangBase):
    """Single auditor record."""
    auditor_id: str = Field(default_factory=_new_id)
    full_name: str = Field(...)
    organization: str = Field(...)
    accreditation_body: Optional[str] = Field(default=None)
    accreditation_status: AccreditationStatusEnum = Field(
        default=AccreditationStatusEnum.ACTIVE
    )
    accreditation_expiry: Optional[date] = Field(default=None)
    commodity_competencies: List[str] = Field(default_factory=list)
    scheme_qualifications: List[str] = Field(default_factory=list)
    country_expertise: List[str] = Field(default_factory=list)
    languages: List[str] = Field(default_factory=list)
    audit_count: int = Field(default=0)
    performance_rating: Decimal = Field(default=Decimal("0.00"))
    cpd_hours: int = Field(default=0)
    cpd_compliant: bool = Field(default=True)
    created_at: datetime = Field(default_factory=utcnow)

class AuditorRegisterResponse(GreenLangBase):
    """Response from auditor registration."""
    auditor: AuditorEntry
    provenance: ProvenanceInfo
    metadata: MetadataSchema = Field(default_factory=MetadataSchema)

class AuditorListResponse(GreenLangBase):
    """Paginated auditor list."""
    auditors: List[AuditorEntry] = Field(default_factory=list)
    pagination: PaginatedMeta
    provenance: ProvenanceInfo
    metadata: MetadataSchema = Field(default_factory=MetadataSchema)

class AuditorDetailResponse(GreenLangBase):
    """Detailed auditor profile."""
    auditor: AuditorEntry
    recent_audits: List[Dict[str, Any]] = Field(default_factory=list)
    performance_history: List[Dict[str, Any]] = Field(default_factory=list)
    provenance: ProvenanceInfo
    metadata: MetadataSchema = Field(default_factory=MetadataSchema)

class AuditorUpdateRequest(GreenLangBase):
    """Request to update auditor profile."""
    full_name: Optional[str] = Field(default=None)
    organization: Optional[str] = Field(default=None)
    accreditation_status: Optional[AccreditationStatusEnum] = Field(default=None)
    accreditation_expiry: Optional[date] = Field(default=None)
    commodity_competencies: Optional[List[EUDRCommodityEnum]] = Field(default=None)
    scheme_qualifications: Optional[List[CertSchemeEnum]] = Field(default=None)
    country_expertise: Optional[List[str]] = Field(default=None)
    languages: Optional[List[str]] = Field(default=None)
    cpd_hours: Optional[int] = Field(default=None)
    contact_email: Optional[str] = Field(default=None)

class AuditorMatchRequest(GreenLangBase):
    """Request to match auditors to audit requirements."""
    audit_id: Optional[str] = Field(default=None)
    commodity: EUDRCommodityEnum = Field(...)
    scheme: Optional[CertSchemeEnum] = Field(default=None)
    country_code: str = Field(..., min_length=2, max_length=2)
    required_language: Optional[str] = Field(default=None)
    audit_date: date = Field(...)
    max_results: int = Field(default=10, ge=1, le=50)

class AuditorMatchEntry(GreenLangBase):
    """Single auditor match result."""
    auditor: AuditorEntry
    match_score: Decimal = Field(...)
    match_breakdown: Dict[str, Decimal] = Field(default_factory=dict)
    availability: bool = Field(default=True)
    conflict_of_interest: bool = Field(default=False)

class AuditorMatchResponse(GreenLangBase):
    """Response from auditor matching."""
    matches: List[AuditorMatchEntry] = Field(default_factory=list)
    total_candidates: int = Field(default=0)
    provenance: ProvenanceInfo
    metadata: MetadataSchema = Field(default_factory=MetadataSchema)

# =============================================================================
# 3. Execution Schemas
# =============================================================================

class CriterionEntry(GreenLangBase):
    """Single audit checklist criterion."""
    criterion_id: str = Field(...)
    category: str = Field(...)
    reference: str = Field(...)
    description: str = Field(...)
    eudr_article_mapping: Optional[str] = Field(default=None)
    result: Optional[CriterionResultEnum] = Field(default=None)
    evidence_ids: List[str] = Field(default_factory=list)
    auditor_notes: Optional[str] = Field(default=None)
    assessed_at: Optional[datetime] = Field(default=None)
    assessed_by: Optional[str] = Field(default=None)

class ChecklistEntry(GreenLangBase):
    """Audit checklist record."""
    checklist_id: str = Field(default_factory=_new_id)
    audit_id: str = Field(...)
    checklist_type: str = Field(...)
    checklist_version: str = Field(...)
    criteria: List[CriterionEntry] = Field(default_factory=list)
    completion_percentage: Decimal = Field(default=Decimal("0.00"))
    total_criteria: int = Field(default=0)
    passed_criteria: int = Field(default=0)
    failed_criteria: int = Field(default=0)
    na_criteria: int = Field(default=0)

class ChecklistResponse(GreenLangBase):
    """Audit checklist response."""
    checklists: List[ChecklistEntry] = Field(default_factory=list)
    overall_completion: Decimal = Field(default=Decimal("0.00"))
    provenance: ProvenanceInfo
    metadata: MetadataSchema = Field(default_factory=MetadataSchema)

class CriterionUpdateRequest(GreenLangBase):
    """Request to update a checklist criterion."""
    result: CriterionResultEnum = Field(...)
    evidence_ids: Optional[List[str]] = Field(default=None)
    auditor_notes: Optional[str] = Field(default=None)

class CriterionUpdateResponse(GreenLangBase):
    """Response from criterion update."""
    criterion: CriterionEntry
    checklist_completion: Decimal = Field(default=Decimal("0.00"))
    provenance: ProvenanceInfo
    metadata: MetadataSchema = Field(default_factory=MetadataSchema)

class EvidenceUploadRequest(GreenLangBase):
    """Request to register audit evidence."""
    evidence_type: EvidenceTypeEnum = Field(...)
    file_name: str = Field(...)
    file_path: Optional[str] = Field(default=None)
    file_size_bytes: Optional[int] = Field(default=None)
    mime_type: Optional[str] = Field(default=None)
    description: Optional[str] = Field(default=None)
    tags: List[str] = Field(default_factory=list)
    location_latitude: Optional[float] = Field(default=None)
    location_longitude: Optional[float] = Field(default=None)
    captured_date: Optional[datetime] = Field(default=None)
    sha256_hash: str = Field(..., min_length=64, max_length=64)

class EvidenceEntry(GreenLangBase):
    """Single evidence item."""
    evidence_id: str = Field(default_factory=_new_id)
    audit_id: str = Field(...)
    evidence_type: EvidenceTypeEnum = Field(...)
    file_name: str = Field(...)
    file_path: Optional[str] = Field(default=None)
    file_size_bytes: Optional[int] = Field(default=None)
    mime_type: Optional[str] = Field(default=None)
    description: Optional[str] = Field(default=None)
    tags: List[str] = Field(default_factory=list)
    sha256_hash: str = Field(...)
    uploaded_by: Optional[str] = Field(default=None)
    created_at: datetime = Field(default_factory=utcnow)

class EvidenceUploadResponse(GreenLangBase):
    """Response from evidence upload."""
    evidence: EvidenceEntry
    provenance: ProvenanceInfo
    metadata: MetadataSchema = Field(default_factory=MetadataSchema)

class EvidenceListResponse(GreenLangBase):
    """Paginated evidence list."""
    evidence_items: List[EvidenceEntry] = Field(default_factory=list)
    pagination: PaginatedMeta
    provenance: ProvenanceInfo
    metadata: MetadataSchema = Field(default_factory=MetadataSchema)

class ProgressResponse(GreenLangBase):
    """Real-time audit progress."""
    audit_id: str = Field(...)
    status: AuditStatusEnum = Field(...)
    checklist_completion: Decimal = Field(default=Decimal("0.00"))
    evidence_count: int = Field(default=0)
    findings_count: Dict[str, int] = Field(default_factory=dict)
    team_members: int = Field(default=0)
    days_elapsed: int = Field(default=0)
    estimated_completion: Optional[date] = Field(default=None)
    provenance: ProvenanceInfo
    metadata: MetadataSchema = Field(default_factory=MetadataSchema)

# =============================================================================
# 4. Non-Conformance Schemas
# =============================================================================

class NCCreateRequest(GreenLangBase):
    """Request to create a non-conformance finding."""
    finding_statement: str = Field(..., min_length=10)
    objective_evidence: str = Field(..., min_length=10)
    severity: NCSeverityEnum = Field(...)
    eudr_article: Optional[str] = Field(default=None)
    scheme_clause: Optional[str] = Field(default=None)
    article_2_40_category: Optional[str] = Field(default=None)
    evidence_ids: List[str] = Field(default_factory=list)

class NCEntry(GreenLangBase):
    """Single non-conformance record."""
    nc_id: str = Field(default_factory=_new_id)
    audit_id: str = Field(...)
    finding_statement: str = Field(...)
    objective_evidence: str = Field(...)
    severity: NCSeverityEnum = Field(...)
    eudr_article: Optional[str] = Field(default=None)
    scheme_clause: Optional[str] = Field(default=None)
    root_cause_analysis: Optional[Dict[str, Any]] = Field(default=None)
    root_cause_method: Optional[RootCauseMethodEnum] = Field(default=None)
    risk_impact_score: Decimal = Field(default=Decimal("0.00"))
    status: NCStatusEnum = Field(default=NCStatusEnum.OPEN)
    car_id: Optional[str] = Field(default=None)
    evidence_ids: List[str] = Field(default_factory=list)
    classification_rules_applied: List[str] = Field(default_factory=list)
    disputed: bool = Field(default=False)
    provenance_hash: str = Field(default="")
    detected_at: datetime = Field(default_factory=utcnow)
    resolved_at: Optional[datetime] = Field(default=None)

class NCCreateResponse(GreenLangBase):
    """Response from NC creation."""
    nc: NCEntry
    auto_classified: bool = Field(default=True)
    sla_deadline: Optional[datetime] = Field(default=None)
    provenance: ProvenanceInfo
    metadata: MetadataSchema = Field(default_factory=MetadataSchema)

class NCListResponse(GreenLangBase):
    """Paginated NC list."""
    non_conformances: List[NCEntry] = Field(default_factory=list)
    pagination: PaginatedMeta
    severity_summary: Dict[str, int] = Field(default_factory=dict)
    provenance: ProvenanceInfo
    metadata: MetadataSchema = Field(default_factory=MetadataSchema)

class NCDetailResponse(GreenLangBase):
    """Detailed NC response."""
    nc: NCEntry
    related_evidence: List[EvidenceEntry] = Field(default_factory=list)
    car: Optional[Dict[str, Any]] = Field(default=None)
    provenance: ProvenanceInfo
    metadata: MetadataSchema = Field(default_factory=MetadataSchema)

class NCUpdateRequest(GreenLangBase):
    """Request to update an NC."""
    status: Optional[NCStatusEnum] = Field(default=None)
    severity: Optional[NCSeverityEnum] = Field(default=None)
    disputed: Optional[bool] = Field(default=None)
    dispute_rationale: Optional[str] = Field(default=None)

class RCASubmitRequest(GreenLangBase):
    """Request to submit root cause analysis."""
    method: RootCauseMethodEnum = Field(...)
    analysis: Dict[str, Any] = Field(...)
    contributing_factors: List[str] = Field(default_factory=list)
    recommended_actions: List[str] = Field(default_factory=list)

class RCASubmitResponse(GreenLangBase):
    """Response from RCA submission."""
    nc_id: str = Field(...)
    root_cause_analysis: Dict[str, Any] = Field(...)
    root_cause_method: RootCauseMethodEnum = Field(...)
    provenance: ProvenanceInfo
    metadata: MetadataSchema = Field(default_factory=MetadataSchema)

# =============================================================================
# 5. CAR Schemas
# =============================================================================

class CARIssueRequest(GreenLangBase):
    """Request to issue a corrective action request."""
    nc_ids: List[str] = Field(..., min_length=1)
    audit_id: str = Field(...)
    supplier_id: str = Field(...)
    severity: NCSeverityEnum = Field(...)
    corrective_action_plan: Optional[Dict[str, Any]] = Field(default=None)

class CAREntry(GreenLangBase):
    """Single CAR record."""
    car_id: str = Field(default_factory=_new_id)
    nc_ids: List[str] = Field(default_factory=list)
    audit_id: str = Field(...)
    supplier_id: str = Field(...)
    severity: NCSeverityEnum = Field(...)
    sla_deadline: datetime = Field(...)
    sla_status: CARSLAStatusEnum = Field(default=CARSLAStatusEnum.ON_TRACK)
    status: CARStatusEnum = Field(default=CARStatusEnum.ISSUED)
    issued_by: str = Field(...)
    issued_at: datetime = Field(default_factory=utcnow)
    acknowledged_at: Optional[datetime] = Field(default=None)
    cap_submitted_at: Optional[datetime] = Field(default=None)
    verified_at: Optional[datetime] = Field(default=None)
    closed_at: Optional[datetime] = Field(default=None)
    corrective_action_plan: Optional[Dict[str, Any]] = Field(default=None)
    verification_outcome: Optional[VerificationOutcomeEnum] = Field(default=None)
    escalation_level: int = Field(default=0, ge=0, le=4)
    escalation_history: List[Dict[str, Any]] = Field(default_factory=list)
    provenance_hash: str = Field(default="")
    created_at: datetime = Field(default_factory=utcnow)
    updated_at: datetime = Field(default_factory=utcnow)

class CARIssueResponse(GreenLangBase):
    """Response from CAR issuance."""
    car: CAREntry
    sla_days: int = Field(...)
    provenance: ProvenanceInfo
    metadata: MetadataSchema = Field(default_factory=MetadataSchema)

class CARListResponse(GreenLangBase):
    """Paginated CAR list."""
    cars: List[CAREntry] = Field(default_factory=list)
    pagination: PaginatedMeta
    status_summary: Dict[str, int] = Field(default_factory=dict)
    provenance: ProvenanceInfo
    metadata: MetadataSchema = Field(default_factory=MetadataSchema)

class CARDetailResponse(GreenLangBase):
    """Detailed CAR response."""
    car: CAREntry
    related_ncs: List[NCEntry] = Field(default_factory=list)
    sla_timeline: List[Dict[str, Any]] = Field(default_factory=list)
    provenance: ProvenanceInfo
    metadata: MetadataSchema = Field(default_factory=MetadataSchema)

class CARUpdateRequest(GreenLangBase):
    """Request to update CAR status."""
    status: Optional[CARStatusEnum] = Field(default=None)
    corrective_action_plan: Optional[Dict[str, Any]] = Field(default=None)
    evidence_ids: Optional[List[str]] = Field(default=None)

class CARVerifyRequest(GreenLangBase):
    """Request to submit CAR verification."""
    verification_outcome: VerificationOutcomeEnum = Field(...)
    verification_evidence_ids: List[str] = Field(default_factory=list)
    verification_notes: Optional[str] = Field(default=None)

class CARVerifyResponse(GreenLangBase):
    """Response from CAR verification."""
    car: CAREntry
    provenance: ProvenanceInfo
    metadata: MetadataSchema = Field(default_factory=MetadataSchema)

# =============================================================================
# 6. Certification Scheme Schemas
# =============================================================================

class CertificateEntry(GreenLangBase):
    """Single certificate record."""
    certificate_id: str = Field(default_factory=_new_id)
    supplier_id: str = Field(...)
    scheme: CertSchemeEnum = Field(...)
    certificate_number: str = Field(...)
    status: CertStatusEnum = Field(default=CertStatusEnum.ACTIVE)
    scope: Optional[str] = Field(default=None)
    supply_chain_model: Optional[str] = Field(default=None)
    issue_date: Optional[date] = Field(default=None)
    expiry_date: Optional[date] = Field(default=None)
    certification_body: Optional[str] = Field(default=None)
    last_audit_date: Optional[date] = Field(default=None)
    next_audit_date: Optional[date] = Field(default=None)
    eudr_coverage_matrix: Dict[str, Any] = Field(default_factory=dict)
    synced_at: Optional[datetime] = Field(default=None)

class CertificateListResponse(GreenLangBase):
    """Paginated certificate list."""
    certificates: List[CertificateEntry] = Field(default_factory=list)
    pagination: PaginatedMeta
    provenance: ProvenanceInfo
    metadata: MetadataSchema = Field(default_factory=MetadataSchema)

class CertSyncRequest(GreenLangBase):
    """Request to trigger certification sync."""
    scheme: CertSchemeEnum = Field(...)
    supplier_ids: Optional[List[str]] = Field(default=None)
    force: bool = Field(default=False)

class CertSyncResponse(GreenLangBase):
    """Response from certification sync."""
    synced_count: int = Field(default=0)
    updated_count: int = Field(default=0)
    new_count: int = Field(default=0)
    errors: List[str] = Field(default_factory=list)
    provenance: ProvenanceInfo
    metadata: MetadataSchema = Field(default_factory=MetadataSchema)

class CoverageMatrixEntry(GreenLangBase):
    """EUDR coverage analysis for a certification scheme."""
    scheme: CertSchemeEnum = Field(...)
    certificate_number: Optional[str] = Field(default=None)
    covered_articles: List[str] = Field(default_factory=list)
    uncovered_articles: List[str] = Field(default_factory=list)
    coverage_percentage: Decimal = Field(default=Decimal("0.00"))
    gaps: List[str] = Field(default_factory=list)

class CoverageMatrixResponse(GreenLangBase):
    """EUDR coverage matrix for a supplier."""
    supplier_id: str = Field(...)
    schemes: List[CoverageMatrixEntry] = Field(default_factory=list)
    overall_coverage: Decimal = Field(default=Decimal("0.00"))
    remaining_gaps: List[str] = Field(default_factory=list)
    provenance: ProvenanceInfo
    metadata: MetadataSchema = Field(default_factory=MetadataSchema)

# =============================================================================
# 7. Report Schemas
# =============================================================================

class ReportGenerateRequest(GreenLangBase):
    """Request to generate an audit report."""
    report_type: str = Field(default="iso_19011")
    format: ReportFormatEnum = Field(default=ReportFormatEnum.PDF)
    language: ReportLanguageEnum = Field(default=ReportLanguageEnum.EN)
    include_evidence_package: bool = Field(default=False)
    sections: Optional[List[str]] = Field(default=None)

class ReportEntry(GreenLangBase):
    """Single report record."""
    report_id: str = Field(default_factory=_new_id)
    audit_id: str = Field(...)
    report_type: str = Field(default="iso_19011")
    report_version: int = Field(default=1)
    language: ReportLanguageEnum = Field(default=ReportLanguageEnum.EN)
    format: ReportFormatEnum = Field(default=ReportFormatEnum.PDF)
    file_path: Optional[str] = Field(default=None)
    file_size_bytes: Optional[int] = Field(default=None)
    sha256_hash: str = Field(default="")
    finding_count: Dict[str, int] = Field(default_factory=dict)
    evidence_package_path: Optional[str] = Field(default=None)
    generated_by: Optional[str] = Field(default=None)
    generated_at: datetime = Field(default_factory=utcnow)

class ReportGenerateResponse(GreenLangBase):
    """Response from report generation."""
    report: ReportEntry
    provenance: ProvenanceInfo
    metadata: MetadataSchema = Field(default_factory=MetadataSchema)

class ReportDownloadResponse(GreenLangBase):
    """Report download information."""
    report: ReportEntry
    download_url: Optional[str] = Field(default=None)
    expires_at: Optional[datetime] = Field(default=None)
    provenance: ProvenanceInfo
    metadata: MetadataSchema = Field(default_factory=MetadataSchema)

# =============================================================================
# 8. Authority Schemas
# =============================================================================

class InteractionCreateRequest(GreenLangBase):
    """Request to log a new authority interaction."""
    operator_id: str = Field(..., description="Operator UUID")
    authority_name: str = Field(..., min_length=1)
    member_state: str = Field(..., min_length=2, max_length=2)
    interaction_type: InteractionTypeEnum = Field(...)
    received_date: datetime = Field(...)
    response_deadline: datetime = Field(...)
    internal_tasks: List[Dict[str, Any]] = Field(default_factory=list)

class InteractionEntry(GreenLangBase):
    """Single authority interaction record."""
    interaction_id: str = Field(default_factory=_new_id)
    operator_id: str = Field(...)
    authority_name: str = Field(...)
    member_state: str = Field(...)
    interaction_type: InteractionTypeEnum = Field(...)
    received_date: datetime = Field(...)
    response_deadline: datetime = Field(...)
    response_sla_status: ResponseSLAStatusEnum = Field(
        default=ResponseSLAStatusEnum.ON_TRACK
    )
    internal_tasks: List[Dict[str, Any]] = Field(default_factory=list)
    response_submitted_at: Optional[datetime] = Field(default=None)
    authority_decision: Optional[str] = Field(default=None)
    enforcement_measures: List[Dict[str, Any]] = Field(default_factory=list)
    status: InteractionStatusEnum = Field(default=InteractionStatusEnum.OPEN)
    provenance_hash: str = Field(default="")
    created_at: datetime = Field(default_factory=utcnow)
    updated_at: datetime = Field(default_factory=utcnow)

class InteractionCreateResponse(GreenLangBase):
    """Response from interaction creation."""
    interaction: InteractionEntry
    provenance: ProvenanceInfo
    metadata: MetadataSchema = Field(default_factory=MetadataSchema)

class InteractionListResponse(GreenLangBase):
    """Paginated interaction list."""
    interactions: List[InteractionEntry] = Field(default_factory=list)
    pagination: PaginatedMeta
    provenance: ProvenanceInfo
    metadata: MetadataSchema = Field(default_factory=MetadataSchema)

class InteractionUpdateRequest(GreenLangBase):
    """Request to update an authority interaction."""
    status: Optional[InteractionStatusEnum] = Field(default=None)
    response_submitted_at: Optional[datetime] = Field(default=None)
    authority_decision: Optional[str] = Field(default=None)
    enforcement_measures: Optional[List[Dict[str, Any]]] = Field(default=None)
    evidence_package_id: Optional[str] = Field(default=None)

# =============================================================================
# 9. Analytics Schemas
# =============================================================================

class FindingTrendEntry(GreenLangBase):
    """Finding trend data point."""
    period: str = Field(...)
    severity: NCSeverityEnum = Field(...)
    count: int = Field(default=0)
    country_code: Optional[str] = Field(default=None)
    commodity: Optional[str] = Field(default=None)

class FindingTrendsResponse(GreenLangBase):
    """Finding trend analytics."""
    trends: List[FindingTrendEntry] = Field(default_factory=list)
    total_findings: int = Field(default=0)
    period_range: str = Field(default="")
    provenance: ProvenanceInfo
    metadata: MetadataSchema = Field(default_factory=MetadataSchema)

class AuditorPerformanceEntry(GreenLangBase):
    """Auditor performance data."""
    auditor_id: str = Field(...)
    auditor_name: str = Field(...)
    audit_count: int = Field(default=0)
    average_findings_per_audit: Decimal = Field(default=Decimal("0.00"))
    car_closure_rate: Decimal = Field(default=Decimal("0.00"))
    average_audit_duration_days: Decimal = Field(default=Decimal("0.00"))
    performance_rating: Decimal = Field(default=Decimal("0.00"))

class AuditorPerformanceResponse(GreenLangBase):
    """Auditor performance analytics."""
    auditors: List[AuditorPerformanceEntry] = Field(default_factory=list)
    total_auditors: int = Field(default=0)
    provenance: ProvenanceInfo
    metadata: MetadataSchema = Field(default_factory=MetadataSchema)

class ComplianceRateEntry(GreenLangBase):
    """Compliance rate data point."""
    period: str = Field(...)
    total_audits: int = Field(default=0)
    compliant_audits: int = Field(default=0)
    compliance_rate: Decimal = Field(default=Decimal("0.00"))
    critical_nc_rate: Decimal = Field(default=Decimal("0.00"))

class ComplianceRatesResponse(GreenLangBase):
    """Compliance rate trend analytics."""
    rates: List[ComplianceRateEntry] = Field(default_factory=list)
    overall_compliance_rate: Decimal = Field(default=Decimal("0.00"))
    provenance: ProvenanceInfo
    metadata: MetadataSchema = Field(default_factory=MetadataSchema)

class CARPerformanceEntry(GreenLangBase):
    """CAR performance data point."""
    period: str = Field(...)
    issued: int = Field(default=0)
    closed: int = Field(default=0)
    overdue: int = Field(default=0)
    average_closure_days: Decimal = Field(default=Decimal("0.00"))
    sla_compliance_rate: Decimal = Field(default=Decimal("0.00"))

class CARPerformanceResponse(GreenLangBase):
    """CAR lifecycle performance analytics."""
    performance: List[CARPerformanceEntry] = Field(default_factory=list)
    overall_sla_compliance: Decimal = Field(default=Decimal("0.00"))
    provenance: ProvenanceInfo
    metadata: MetadataSchema = Field(default_factory=MetadataSchema)

class DashboardResponse(GreenLangBase):
    """Executive dashboard aggregate data."""
    active_audits: int = Field(default=0)
    open_cars: int = Field(default=0)
    overdue_cars: int = Field(default=0)
    car_sla_compliance_rate: Decimal = Field(default=Decimal("0.00"))
    total_ncs_this_quarter: int = Field(default=0)
    critical_ncs_this_quarter: int = Field(default=0)
    audits_completed_this_quarter: int = Field(default=0)
    compliance_rate: Decimal = Field(default=Decimal("0.00"))
    pending_authority_responses: int = Field(default=0)
    active_certificates: int = Field(default=0)
    provenance: ProvenanceInfo
    metadata: MetadataSchema = Field(default_factory=MetadataSchema)

# =============================================================================
# 10. Admin Schemas
# =============================================================================

class StatsResponse(GreenLangBase):
    """Service statistics response."""
    total_audits: int = Field(default=0)
    total_auditors: int = Field(default=0)
    total_ncs: int = Field(default=0)
    total_cars: int = Field(default=0)
    total_certificates: int = Field(default=0)
    total_reports: int = Field(default=0)
    total_authority_interactions: int = Field(default=0)
    database_tables: int = Field(default=17)
    api_endpoints: int = Field(default=39)
    provenance: ProvenanceInfo
    metadata: MetadataSchema = Field(default_factory=MetadataSchema)

# =============================================================================
# Public API
# =============================================================================

__all__ = [
    # Common
    "ProvenanceInfo",
    "MetadataSchema",
    "PaginatedMeta",
    "ErrorResponse",
    "HealthResponse",
    # Enums
    "AuditTypeEnum",
    "AuditModalityEnum",
    "AuditStatusEnum",
    "NCSeverityEnum",
    "NCStatusEnum",
    "CARStatusEnum",
    "CARSLAStatusEnum",
    "CertSchemeEnum",
    "CertStatusEnum",
    "AccreditationStatusEnum",
    "AuditorRoleEnum",
    "EvidenceTypeEnum",
    "CriterionResultEnum",
    "RootCauseMethodEnum",
    "VerificationOutcomeEnum",
    "InteractionTypeEnum",
    "InteractionStatusEnum",
    "ResponseSLAStatusEnum",
    "ReportFormatEnum",
    "ReportLanguageEnum",
    "ScheduleStatusEnum",
    "EUDRCommodityEnum",
    "InterviewTypeEnum",
    "FrequencyTierEnum",
    "AuditScopeEnum",
    # Planning
    "ScheduleGenerateRequest",
    "ScheduleEntry",
    "ScheduleGenerateResponse",
    "ScheduleListResponse",
    "ScheduleUpdateRequest",
    "TriggerAuditRequest",
    "TriggerAuditResponse",
    "AuditCreateRequest",
    "AuditEntry",
    "AuditCreateResponse",
    "AuditListResponse",
    "AuditDetailResponse",
    "AuditUpdateRequest",
    # Auditor
    "AuditorRegisterRequest",
    "AuditorEntry",
    "AuditorRegisterResponse",
    "AuditorListResponse",
    "AuditorDetailResponse",
    "AuditorUpdateRequest",
    "AuditorMatchRequest",
    "AuditorMatchEntry",
    "AuditorMatchResponse",
    # Execution
    "CriterionEntry",
    "ChecklistEntry",
    "ChecklistResponse",
    "CriterionUpdateRequest",
    "CriterionUpdateResponse",
    "EvidenceUploadRequest",
    "EvidenceEntry",
    "EvidenceUploadResponse",
    "EvidenceListResponse",
    "ProgressResponse",
    # NC
    "NCCreateRequest",
    "NCEntry",
    "NCCreateResponse",
    "NCListResponse",
    "NCDetailResponse",
    "NCUpdateRequest",
    "RCASubmitRequest",
    "RCASubmitResponse",
    # CAR
    "CARIssueRequest",
    "CAREntry",
    "CARIssueResponse",
    "CARListResponse",
    "CARDetailResponse",
    "CARUpdateRequest",
    "CARVerifyRequest",
    "CARVerifyResponse",
    # Scheme
    "CertificateEntry",
    "CertificateListResponse",
    "CertSyncRequest",
    "CertSyncResponse",
    "CoverageMatrixEntry",
    "CoverageMatrixResponse",
    # Report
    "ReportGenerateRequest",
    "ReportEntry",
    "ReportGenerateResponse",
    "ReportDownloadResponse",
    # Authority
    "InteractionCreateRequest",
    "InteractionEntry",
    "InteractionCreateResponse",
    "InteractionListResponse",
    "InteractionUpdateRequest",
    # Analytics
    "FindingTrendEntry",
    "FindingTrendsResponse",
    "AuditorPerformanceEntry",
    "AuditorPerformanceResponse",
    "ComplianceRateEntry",
    "ComplianceRatesResponse",
    "CARPerformanceEntry",
    "CARPerformanceResponse",
    "DashboardResponse",
    # Admin
    "StatsResponse",
]
