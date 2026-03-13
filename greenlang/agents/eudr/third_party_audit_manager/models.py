# -*- coding: utf-8 -*-
"""
Third-Party Audit Manager Data Models - AGENT-EUDR-024

Pydantic v2 data models for the Third-Party Audit Manager Agent covering
audit lifecycle management, auditor qualification tracking, audit execution
monitoring, non-conformance detection and classification, corrective action
request (CAR) lifecycle management, certification scheme integration,
ISO 19011:2018 compliant audit report generation, competent authority
liaison workflows, and audit analytics with trend detection.

Every model is designed for deterministic serialization and SHA-256
provenance hashing to ensure zero-hallucination, bit-perfect
reproducibility across all third-party audit management operations per
EU 2023/1115 Articles 4, 9, 10, 11, 14-16, 18-23, 29, 31 and
ISO 19011:2018, ISO/IEC 17065:2012, ISO/IEC 17021-1:2015.

Enumerations (7):
    - AuditStatus, AuditScope, AuditModality, NCSeverity, CARStatus,
      CertificationScheme, AuthorityInteractionType

Core Models (10):
    - Audit, Auditor, AuditChecklist, AuditEvidence, NonConformance,
      RootCauseAnalysis, CorrectiveActionRequest, CertificateRecord,
      CompetentAuthorityInteraction, AuditReport

Request Models (7):
    - ScheduleAuditRequest, MatchAuditorRequest, ClassifyNCRequest,
      IssueCARRequest, GenerateReportRequest, LogAuthorityInteractionRequest,
      CalculateAnalyticsRequest

Response Models (7):
    - ScheduleAuditResponse, MatchAuditorResponse, ClassifyNCResponse,
      IssueCARResponse, GenerateReportResponse, LogAuthorityInteractionResponse,
      CalculateAnalyticsResponse

Author: GreenLang Platform Team
Date: March 2026
PRD: AGENT-EUDR-024 Third-Party Audit Manager (GL-EUDR-TAM-024)
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

#: Maximum number of records in a single batch processing job.
MAX_BATCH_SIZE: int = 1000

#: EUDR Article 31 data retention in years.
EUDR_RETENTION_YEARS: int = 5

#: Supported certification schemes.
SUPPORTED_SCHEMES: List[str] = [
    "fsc", "pefc", "rspo", "rainforest_alliance", "iscc",
]

#: EUDR-regulated commodities per Article 1.
SUPPORTED_COMMODITIES: List[str] = [
    "cattle", "cocoa", "coffee", "palm_oil", "rubber", "soya", "wood",
]

#: Supported report output formats.
SUPPORTED_REPORT_FORMATS: List[str] = [
    "pdf", "json", "html", "xlsx", "xml",
]

#: Supported report languages.
SUPPORTED_REPORT_LANGUAGES: List[str] = [
    "en", "fr", "de", "es", "pt",
]

#: NC severity to SLA days mapping.
NC_SEVERITY_SLA_DAYS: Dict[str, int] = {
    "critical": 30,
    "major": 90,
    "minor": 365,
}

#: EU Member State country codes.
EU_MEMBER_STATES: List[str] = [
    "AT", "BE", "BG", "HR", "CY", "CZ", "DK", "EE", "FI", "FR",
    "DE", "GR", "HU", "IE", "IT", "LV", "LT", "LU", "MT", "NL",
    "PL", "PT", "RO", "SK", "SI", "ES", "SE",
]


# ---------------------------------------------------------------------------
# Enumerations (7)
# ---------------------------------------------------------------------------


class AuditStatus(str, Enum):
    """Audit lifecycle status.

    Tracks the audit through its lifecycle from planning through
    execution, reporting, corrective action follow-up, and closure.
    Aligned with ISO 19011:2018 Clause 5 audit programme management.
    """

    PLANNED = "planned"
    """Scheduled in audit calendar, not yet started."""

    AUDITOR_ASSIGNED = "auditor_assigned"
    """Lead auditor and team confirmed for the audit."""

    IN_PREPARATION = "in_preparation"
    """Pre-audit documentation review and logistics underway."""

    IN_PROGRESS = "in_progress"
    """Audit execution is active (fieldwork or desk review)."""

    FIELDWORK_COMPLETE = "fieldwork_complete"
    """On-site work completed; findings being compiled."""

    REPORT_DRAFTING = "report_drafting"
    """Audit report being generated per ISO 19011 Clause 6.6."""

    REPORT_ISSUED = "report_issued"
    """Final audit report delivered to auditee and operator."""

    CAR_FOLLOW_UP = "car_follow_up"
    """Corrective action requests issued and being tracked."""

    CLOSED = "closed"
    """All CARs resolved, audit lifecycle complete."""

    CANCELLED = "cancelled"
    """Audit cancelled (rescheduled or no longer required)."""


class AuditScope(str, Enum):
    """Audit scope classification.

    Determines the breadth and depth of audit criteria to be assessed.
    Full scope covers all EUDR and certification scheme criteria;
    targeted scope focuses on specific risk areas identified through
    risk-based scheduling.
    """

    FULL = "full"
    """All EUDR criteria + certification scheme criteria assessed."""

    TARGETED = "targeted"
    """Specific risk areas only, based on risk assessment triggers."""

    SURVEILLANCE = "surveillance"
    """Maintenance/follow-up audit within certification cycle."""

    UNSCHEDULED = "unscheduled"
    """Event-triggered audit (deforestation alert, cert suspension)."""


class AuditModality(str, Enum):
    """Audit execution modality.

    Defines how the audit is conducted, ranging from full on-site
    physical inspection to remote desktop review, supporting both
    planned and unannounced audit approaches.
    """

    ON_SITE = "on_site"
    """Physical on-site audit with field verification."""

    REMOTE = "remote"
    """Desktop/remote audit via document review and video calls."""

    HYBRID = "hybrid"
    """Remote preparation phase + on-site verification phase."""

    UNANNOUNCED = "unannounced"
    """Unannounced on-site audit without advance notice."""


class NCSeverity(str, Enum):
    """Non-conformance severity classification.

    Three severity levels per ISO 19011:2018 and certification scheme
    conventions, plus observations which are improvement opportunities
    not requiring corrective action.
    """

    CRITICAL = "critical"
    """Immediate threat; systemic failure; evidence of fraud.
    Requires corrective action within 30 days.
    May trigger certification suspension."""

    MAJOR = "major"
    """Significant failure undermining system effectiveness.
    Requires corrective action within 90 days."""

    MINOR = "minor"
    """Isolated or minor deviation not undermining system.
    Requires corrective action within 365 days."""

    OBSERVATION = "observation"
    """Not an NC; opportunity for improvement (OFI).
    No corrective action required; tracked for trend analysis."""


class CARStatus(str, Enum):
    """Corrective Action Request lifecycle status.

    Nine-phase lifecycle from issuance through verified closure,
    with rejection and escalation states for non-compliant auditees.
    """

    ISSUED = "issued"
    """CAR issued to auditee with SLA deadline."""

    ACKNOWLEDGED = "acknowledged"
    """Auditee has confirmed receipt of the CAR."""

    RCA_SUBMITTED = "rca_submitted"
    """Root cause analysis submitted by auditee."""

    CAP_SUBMITTED = "cap_submitted"
    """Corrective action plan submitted for review."""

    CAP_APPROVED = "cap_approved"
    """Corrective action plan approved by auditor."""

    IN_PROGRESS = "in_progress"
    """Corrective action implementation underway."""

    EVIDENCE_SUBMITTED = "evidence_submitted"
    """Auditee has submitted closure evidence."""

    VERIFICATION_PENDING = "verification_pending"
    """Auditor reviewing closure evidence."""

    CLOSED = "closed"
    """CAR verified closed; corrective action effective."""

    REJECTED = "rejected"
    """Evidence rejected; returned to IN_PROGRESS."""

    OVERDUE = "overdue"
    """SLA deadline exceeded without closure."""

    ESCALATED = "escalated"
    """Escalated to higher management or authority."""


class CertificationScheme(str, Enum):
    """Major certification schemes for EUDR compliance verification.

    Five certification schemes that conduct audits relevant to EUDR
    due diligence. Each scheme has specific audit standards, NC
    classification, and recertification cycles.
    """

    FSC = "fsc"
    """Forest Stewardship Council: timber/wood; 5-year cycle."""

    PEFC = "pefc"
    """Programme for the Endorsement of Forest Certification; 5-year."""

    RSPO = "rspo"
    """Roundtable on Sustainable Palm Oil; 5-year cycle."""

    RAINFOREST_ALLIANCE = "rainforest_alliance"
    """Rainforest Alliance: cocoa/coffee/tea; 3-year cycle."""

    ISCC = "iscc"
    """International Sustainability and Carbon Certification; annual."""


class AuthorityInteractionType(str, Enum):
    """EU Member State competent authority interaction types.

    Covers all interaction types defined in EUDR Articles 14-20
    for operator engagement with enforcement authorities.
    """

    DOCUMENT_REQUEST = "document_request"
    """Article 15: Written request for DDS and supply chain evidence."""

    INSPECTION_NOTIFICATION = "inspection_notification"
    """Article 15: Advance notice of scheduled on-site inspection."""

    UNANNOUNCED_INSPECTION = "unannounced_inspection"
    """Article 15(2): Inspection without advance notice."""

    CORRECTIVE_ACTION_ORDER = "corrective_action_order"
    """Article 18: Authority-issued corrective action requirement."""

    INTERIM_MEASURE = "interim_measure"
    """Article 19: Interim suspension or restriction measure."""

    DEFINITIVE_MEASURE = "definitive_measure"
    """Article 20: Definitive enforcement measure (fine, ban)."""

    INFORMATION_REQUEST = "information_request"
    """Article 21: Request for self-disclosure information."""


# ---------------------------------------------------------------------------
# Core Models (10)
# ---------------------------------------------------------------------------


class Audit(BaseModel):
    """Core audit record for third-party EUDR compliance verification.

    Represents a single audit lifecycle from planning through execution,
    reporting, and corrective action follow-up. Links to supplier, auditor,
    certification scheme, and EUDR article scope.

    Attributes:
        audit_id: Unique audit identifier (UUID).
        operator_id: Operator owning this audit.
        supplier_id: Supplier being audited.
        audit_type: Audit scope (full/targeted/surveillance/unscheduled).
        modality: Audit modality (on_site/remote/hybrid/unannounced).
        certification_scheme: Certification scheme if scheme audit.
        eudr_articles: EUDR articles in audit scope.
        planned_date: Scheduled audit date.
        actual_start_date: Actual audit start date.
        actual_end_date: Actual audit completion date.
        lead_auditor_id: Assigned lead auditor identifier.
        audit_team: List of audit team member identifiers.
        status: Current audit lifecycle status.
        priority_score: Risk-based priority score (0-100).
        country_code: ISO 3166-1 alpha-2 country of audited site.
        commodity: Primary EUDR commodity being audited.
        site_ids: List of audited site identifiers.
        checklist_completion: Checklist completion percentage (0-100).
        findings_count: NC count by severity.
        evidence_count: Number of evidence items collected.
        report_id: Generated audit report identifier.
        trigger_reason: Reason for unscheduled/triggered audit.
        estimated_duration_days: Estimated audit duration in days.
        actual_cost_eur: Actual audit cost in EUR.
        provenance_hash: SHA-256 hash for audit trail.
        created_at: Record creation timestamp (UTC).
        updated_at: Record last update timestamp (UTC).
    """

    model_config = ConfigDict(
        str_strip_whitespace=True,
        validate_default=True,
    )

    audit_id: str = Field(
        default_factory=_new_uuid,
        description="Unique audit identifier (UUID)",
    )
    operator_id: str = Field(
        ...,
        min_length=1,
        description="Operator owning this audit",
    )
    supplier_id: str = Field(
        ...,
        min_length=1,
        description="Supplier being audited",
    )
    audit_type: AuditScope = Field(
        AuditScope.FULL,
        description="Audit scope classification",
    )
    modality: AuditModality = Field(
        AuditModality.ON_SITE,
        description="Audit execution modality",
    )
    certification_scheme: Optional[CertificationScheme] = Field(
        None,
        description="Certification scheme if scheme audit",
    )
    eudr_articles: List[str] = Field(
        default_factory=list,
        description="EUDR articles in audit scope",
    )
    planned_date: date = Field(
        ...,
        description="Scheduled audit date",
    )
    actual_start_date: Optional[date] = Field(
        None,
        description="Actual audit start date",
    )
    actual_end_date: Optional[date] = Field(
        None,
        description="Actual audit completion date",
    )
    lead_auditor_id: Optional[str] = Field(
        None,
        description="Assigned lead auditor identifier",
    )
    audit_team: List[str] = Field(
        default_factory=list,
        description="Audit team member identifiers",
    )
    status: AuditStatus = Field(
        AuditStatus.PLANNED,
        description="Current audit lifecycle status",
    )
    priority_score: Decimal = Field(
        Decimal("0"),
        ge=Decimal("0"),
        le=Decimal("100"),
        description="Risk-based priority score (0-100)",
    )
    country_code: str = Field(
        ...,
        min_length=2,
        max_length=3,
        description="ISO 3166-1 alpha-2 country code",
    )
    commodity: str = Field(
        ...,
        min_length=1,
        description="Primary EUDR commodity",
    )
    site_ids: List[str] = Field(
        default_factory=list,
        description="Audited site identifiers",
    )
    checklist_completion: Decimal = Field(
        Decimal("0"),
        ge=Decimal("0"),
        le=Decimal("100"),
        description="Checklist completion percentage (0-100)",
    )
    findings_count: Dict[str, int] = Field(
        default_factory=lambda: {
            "critical": 0, "major": 0, "minor": 0, "observation": 0,
        },
        description="NC count by severity",
    )
    evidence_count: int = Field(
        0,
        ge=0,
        description="Number of evidence items collected",
    )
    report_id: Optional[str] = Field(
        None,
        description="Generated audit report identifier",
    )
    trigger_reason: Optional[str] = Field(
        None,
        max_length=500,
        description="Reason for unscheduled/triggered audit",
    )
    estimated_duration_days: Optional[int] = Field(
        None,
        ge=1,
        description="Estimated audit duration in days",
    )
    actual_cost_eur: Optional[Decimal] = Field(
        None,
        ge=Decimal("0"),
        description="Actual audit cost in EUR",
    )
    provenance_hash: Optional[str] = Field(
        None,
        description="SHA-256 hash for audit trail",
    )
    created_at: datetime = Field(
        default_factory=_utcnow,
        description="Record creation timestamp (UTC)",
    )
    updated_at: datetime = Field(
        default_factory=_utcnow,
        description="Record last update timestamp (UTC)",
    )

    @field_validator("country_code")
    @classmethod
    def validate_country_code(cls, v: str) -> str:
        """Ensure country_code is uppercase."""
        return v.upper()


class Auditor(BaseModel):
    """Auditor profile with ISO/IEC 17065 and 17021-1 competence tracking.

    Maintains comprehensive auditor qualification data including
    accreditation status, commodity expertise, regional competence,
    conflict-of-interest declarations, performance history, and
    continuing professional development records.

    Attributes:
        auditor_id: Unique auditor identifier (UUID).
        full_name: Auditor full legal name.
        organization: Employing certification body or audit firm.
        accreditation_body: IAF MLA signatory accreditation body.
        accreditation_status: Current accreditation status.
        accreditation_expiry: Accreditation expiry date.
        accreditation_scope: Accreditation scope details.
        commodity_competencies: EUDR commodities qualified for.
        scheme_qualifications: Scheme-specific qualifications.
        country_expertise: Countries/regions of expertise.
        languages: Language capabilities (ISO 639-1 codes).
        conflict_of_interest: CoI declarations.
        audit_count: Total audits conducted.
        performance_rating: Performance rating (0-100).
        findings_per_audit: Average findings per audit.
        car_closure_rate: CAR closure rate percentage.
        cpd_hours: CPD hours completed.
        cpd_compliant: Whether CPD requirements are met.
        contact_email: Professional contact email.
        available_from: Next available date.
        provenance_hash: SHA-256 hash for audit trail.
        created_at: Record creation timestamp (UTC).
        updated_at: Record last update timestamp (UTC).
    """

    model_config = ConfigDict(
        str_strip_whitespace=True,
        validate_default=True,
    )

    auditor_id: str = Field(
        default_factory=_new_uuid,
        description="Unique auditor identifier (UUID)",
    )
    full_name: str = Field(
        ...,
        min_length=1,
        max_length=500,
        description="Auditor full legal name",
    )
    organization: str = Field(
        ...,
        min_length=1,
        max_length=500,
        description="Employing certification body or audit firm",
    )
    accreditation_body: Optional[str] = Field(
        None,
        max_length=200,
        description="IAF MLA signatory accreditation body",
    )
    accreditation_status: str = Field(
        "active",
        description="Accreditation status (active/suspended/withdrawn)",
    )
    accreditation_expiry: Optional[date] = Field(
        None,
        description="Accreditation expiry date",
    )
    accreditation_scope: List[str] = Field(
        default_factory=list,
        description="Accreditation scope details",
    )
    commodity_competencies: List[str] = Field(
        default_factory=list,
        description="EUDR commodities qualified for",
    )
    scheme_qualifications: List[str] = Field(
        default_factory=list,
        description="Scheme-specific qualifications (e.g. FSC Lead Auditor)",
    )
    country_expertise: List[str] = Field(
        default_factory=list,
        description="Countries of expertise (ISO 3166-1 alpha-2)",
    )
    languages: List[str] = Field(
        default_factory=list,
        description="Language capabilities (ISO 639-1 codes)",
    )
    conflict_of_interest: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Conflict-of-interest declarations",
    )
    audit_count: int = Field(
        0,
        ge=0,
        description="Total audits conducted",
    )
    performance_rating: Decimal = Field(
        Decimal("0"),
        ge=Decimal("0"),
        le=Decimal("100"),
        description="Performance rating (0-100)",
    )
    findings_per_audit: Decimal = Field(
        Decimal("0"),
        ge=Decimal("0"),
        description="Average findings per audit",
    )
    car_closure_rate: Decimal = Field(
        Decimal("0"),
        ge=Decimal("0"),
        le=Decimal("100"),
        description="CAR closure rate percentage",
    )
    cpd_hours: int = Field(
        0,
        ge=0,
        description="CPD hours completed",
    )
    cpd_compliant: bool = Field(
        True,
        description="Whether CPD requirements are met",
    )
    contact_email: Optional[str] = Field(
        None,
        max_length=500,
        description="Professional contact email",
    )
    available_from: Optional[date] = Field(
        None,
        description="Next available date",
    )
    provenance_hash: Optional[str] = Field(
        None,
        description="SHA-256 hash for audit trail",
    )
    created_at: datetime = Field(
        default_factory=_utcnow,
        description="Record creation timestamp (UTC)",
    )
    updated_at: datetime = Field(
        default_factory=_utcnow,
        description="Record last update timestamp (UTC)",
    )


class AuditChecklist(BaseModel):
    """Audit checklist with EUDR and scheme-specific criteria.

    Manages structured audit checklists for EUDR compliance assessment
    and certification scheme verification, tracking completion status,
    pass/fail per criterion, and evidence attachment.

    Attributes:
        checklist_id: Unique checklist identifier (UUID).
        audit_id: Parent audit identifier.
        checklist_type: Type (eudr, fsc, pefc, rspo, ra, iscc).
        checklist_version: Version string.
        criteria: List of audit criteria with status.
        completion_percentage: Percentage complete (0-100).
        total_criteria: Total number of criteria.
        passed_criteria: Number of criteria passed.
        failed_criteria: Number of criteria failed.
        na_criteria: Number of not-applicable criteria.
        provenance_hash: SHA-256 hash for audit trail.
        created_at: Record creation timestamp (UTC).
        updated_at: Record last update timestamp (UTC).
    """

    model_config = ConfigDict(
        str_strip_whitespace=True,
        validate_default=True,
    )

    checklist_id: str = Field(
        default_factory=_new_uuid,
        description="Unique checklist identifier (UUID)",
    )
    audit_id: str = Field(
        ...,
        description="Parent audit identifier",
    )
    checklist_type: str = Field(
        "eudr",
        description="Checklist type (eudr, fsc, pefc, rspo, ra, iscc)",
    )
    checklist_version: str = Field(
        "1.0.0",
        description="Checklist version string",
    )
    criteria: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="List of audit criteria with status",
    )
    completion_percentage: Decimal = Field(
        Decimal("0"),
        ge=Decimal("0"),
        le=Decimal("100"),
        description="Checklist completion percentage (0-100)",
    )
    total_criteria: int = Field(
        0,
        ge=0,
        description="Total number of criteria",
    )
    passed_criteria: int = Field(
        0,
        ge=0,
        description="Number of criteria passed",
    )
    failed_criteria: int = Field(
        0,
        ge=0,
        description="Number of criteria failed",
    )
    na_criteria: int = Field(
        0,
        ge=0,
        description="Number of not-applicable criteria",
    )
    provenance_hash: Optional[str] = Field(
        None,
        description="SHA-256 hash for audit trail",
    )
    created_at: datetime = Field(
        default_factory=_utcnow,
        description="Record creation timestamp (UTC)",
    )
    updated_at: datetime = Field(
        default_factory=_utcnow,
        description="Record last update timestamp (UTC)",
    )


class AuditEvidence(BaseModel):
    """Audit evidence item with integrity verification.

    Represents a single piece of evidence collected during audit
    execution, including document classification, metadata tagging,
    and SHA-256 hash for integrity verification.

    Attributes:
        evidence_id: Unique evidence identifier (UUID).
        audit_id: Parent audit identifier.
        evidence_type: Type classification.
        file_name: Original file name.
        file_size_bytes: File size in bytes.
        mime_type: MIME type of the file.
        sha256_hash: SHA-256 hash of file contents.
        description: Evidence description.
        tags: Metadata tags.
        collection_date: Date evidence was collected.
        collector_id: Person who collected the evidence.
        location: Location where evidence was collected.
        linked_criteria_ids: Linked checklist criteria.
        linked_nc_ids: Linked non-conformance findings.
        storage_path: S3 or file system path.
        provenance_hash: SHA-256 hash for audit trail.
        created_at: Record creation timestamp (UTC).
    """

    model_config = ConfigDict(
        str_strip_whitespace=True,
        validate_default=True,
    )

    evidence_id: str = Field(
        default_factory=_new_uuid,
        description="Unique evidence identifier (UUID)",
    )
    audit_id: str = Field(
        ...,
        description="Parent audit identifier",
    )
    evidence_type: str = Field(
        ...,
        description="Type (permit, certificate, photo, gps_record, "
        "interview_transcript, lab_result, document, other)",
    )
    file_name: str = Field(
        ...,
        min_length=1,
        max_length=500,
        description="Original file name",
    )
    file_size_bytes: int = Field(
        0,
        ge=0,
        description="File size in bytes",
    )
    mime_type: Optional[str] = Field(
        None,
        max_length=100,
        description="MIME type of the file",
    )
    sha256_hash: Optional[str] = Field(
        None,
        description="SHA-256 hash of file contents",
    )
    description: Optional[str] = Field(
        None,
        max_length=2000,
        description="Evidence description",
    )
    tags: Dict[str, str] = Field(
        default_factory=dict,
        description="Metadata tags (date, location, source, etc.)",
    )
    collection_date: Optional[date] = Field(
        None,
        description="Date evidence was collected",
    )
    collector_id: Optional[str] = Field(
        None,
        description="Person who collected the evidence",
    )
    location: Optional[str] = Field(
        None,
        max_length=500,
        description="Location where evidence was collected",
    )
    linked_criteria_ids: List[str] = Field(
        default_factory=list,
        description="Linked checklist criteria identifiers",
    )
    linked_nc_ids: List[str] = Field(
        default_factory=list,
        description="Linked non-conformance identifiers",
    )
    storage_path: Optional[str] = Field(
        None,
        description="S3 or file system storage path",
    )
    provenance_hash: Optional[str] = Field(
        None,
        description="SHA-256 hash for audit trail",
    )
    created_at: datetime = Field(
        default_factory=_utcnow,
        description="Record creation timestamp (UTC)",
    )


class RootCauseAnalysis(BaseModel):
    """Root cause analysis output using 5-Whys or Ishikawa framework.

    Structured RCA supporting both the 5-Whys sequential questioning
    framework and the Ishikawa (fishbone) diagram with 6 categories.

    Attributes:
        rca_id: Unique RCA identifier (UUID).
        nc_id: Parent non-conformance identifier.
        framework: RCA framework used (five_whys or ishikawa).
        five_whys: 5-Whys questioning sequence.
        ishikawa_categories: Ishikawa fishbone diagram categories.
        direct_cause: Identified direct cause.
        contributing_causes: Contributing causes.
        root_cause: Identified root cause.
        recommended_actions: Recommended corrective actions.
        analyst_id: Person who conducted the RCA.
        provenance_hash: SHA-256 hash for audit trail.
        created_at: Record creation timestamp (UTC).
    """

    model_config = ConfigDict(
        str_strip_whitespace=True,
        validate_default=True,
    )

    rca_id: str = Field(
        default_factory=_new_uuid,
        description="Unique RCA identifier (UUID)",
    )
    nc_id: str = Field(
        ...,
        description="Parent non-conformance identifier",
    )
    framework: str = Field(
        "five_whys",
        description="RCA framework (five_whys or ishikawa)",
    )
    five_whys: List[Dict[str, str]] = Field(
        default_factory=list,
        description="5-Whys questioning sequence "
        "[{'why': 'question', 'because': 'answer'}, ...]",
    )
    ishikawa_categories: Dict[str, List[str]] = Field(
        default_factory=lambda: {
            "people": [],
            "process": [],
            "equipment": [],
            "materials": [],
            "environment": [],
            "management": [],
        },
        description="Ishikawa fishbone diagram categories",
    )
    direct_cause: Optional[str] = Field(
        None,
        max_length=2000,
        description="Identified direct cause",
    )
    contributing_causes: List[str] = Field(
        default_factory=list,
        description="Contributing causes",
    )
    root_cause: Optional[str] = Field(
        None,
        max_length=2000,
        description="Identified root cause",
    )
    recommended_actions: List[str] = Field(
        default_factory=list,
        description="Recommended corrective actions",
    )
    analyst_id: Optional[str] = Field(
        None,
        description="Person who conducted the RCA",
    )
    provenance_hash: Optional[str] = Field(
        None,
        description="SHA-256 hash for audit trail",
    )
    created_at: datetime = Field(
        default_factory=_utcnow,
        description="Record creation timestamp (UTC)",
    )


class NonConformance(BaseModel):
    """Non-conformance finding from audit execution.

    Represents a structured audit finding classified by severity
    (critical/major/minor/observation) with links to EUDR articles,
    certification scheme clauses, evidence, and root cause analysis.

    Attributes:
        nc_id: Unique NC identifier (UUID).
        audit_id: Parent audit identifier.
        finding_statement: Description of what was found.
        objective_evidence: Evidence supporting the finding.
        severity: NC severity classification.
        eudr_article: Mapped EUDR article.
        scheme_clause: Mapped certification scheme clause.
        article_2_40_category: Mapped Article 2(40) legislation category.
        root_cause_analysis: RCA output.
        risk_impact_score: Risk impact score (0-100).
        status: NC lifecycle status.
        car_id: Linked CAR identifier.
        evidence_ids: Linked evidence item identifiers.
        disputed: Whether the NC is disputed by auditee.
        dispute_rationale: Auditee dispute rationale.
        classification_rule: Rule used for severity classification.
        provenance_hash: SHA-256 hash for audit trail.
        detected_at: NC detection timestamp (UTC).
        resolved_at: NC resolution timestamp (UTC).
    """

    model_config = ConfigDict(
        str_strip_whitespace=True,
        validate_default=True,
    )

    nc_id: str = Field(
        default_factory=_new_uuid,
        description="Unique NC identifier (UUID)",
    )
    audit_id: str = Field(
        ...,
        description="Parent audit identifier",
    )
    finding_statement: str = Field(
        ...,
        min_length=1,
        max_length=5000,
        description="Description of what was found",
    )
    objective_evidence: str = Field(
        ...,
        min_length=1,
        max_length=5000,
        description="Evidence supporting the finding",
    )
    severity: NCSeverity = Field(
        ...,
        description="NC severity classification",
    )
    eudr_article: Optional[str] = Field(
        None,
        max_length=50,
        description="Mapped EUDR article (e.g. 'Art. 9(1)')",
    )
    scheme_clause: Optional[str] = Field(
        None,
        max_length=100,
        description="Mapped certification scheme clause",
    )
    article_2_40_category: Optional[str] = Field(
        None,
        max_length=100,
        description="Mapped Article 2(40) legislation category",
    )
    root_cause_analysis: Optional[RootCauseAnalysis] = Field(
        None,
        description="Root cause analysis output",
    )
    risk_impact_score: Decimal = Field(
        Decimal("0"),
        ge=Decimal("0"),
        le=Decimal("100"),
        description="Risk impact score (0-100)",
    )
    status: str = Field(
        "open",
        description="NC lifecycle status (open, acknowledged, car_issued, "
        "cap_submitted, in_progress, verification_pending, closed, escalated)",
    )
    car_id: Optional[str] = Field(
        None,
        description="Linked corrective action request identifier",
    )
    evidence_ids: List[str] = Field(
        default_factory=list,
        description="Linked evidence item identifiers",
    )
    disputed: bool = Field(
        False,
        description="Whether the NC is disputed by auditee",
    )
    dispute_rationale: Optional[str] = Field(
        None,
        max_length=2000,
        description="Auditee dispute rationale",
    )
    classification_rule: Optional[str] = Field(
        None,
        max_length=500,
        description="Rule identifier used for severity classification",
    )
    provenance_hash: Optional[str] = Field(
        None,
        description="SHA-256 hash for audit trail",
    )
    detected_at: datetime = Field(
        default_factory=_utcnow,
        description="NC detection timestamp (UTC)",
    )
    resolved_at: Optional[datetime] = Field(
        None,
        description="NC resolution timestamp (UTC)",
    )


class CorrectiveActionRequest(BaseModel):
    """Corrective Action Request (CAR) with full lifecycle management.

    Manages the complete CAR lifecycle from issuance through verified
    closure with SLA enforcement, escalation rules, and evidence tracking.

    Attributes:
        car_id: Unique CAR identifier (UUID).
        nc_ids: Linked non-conformance identifiers.
        audit_id: Parent audit identifier.
        supplier_id: Auditee supplier identifier.
        severity: Highest severity of linked NCs.
        sla_deadline: Calculated SLA deadline from severity.
        sla_status: SLA compliance status.
        status: Current CAR lifecycle status.
        issued_by: Auditor or authority who issued the CAR.
        issued_at: CAR issuance timestamp.
        acknowledged_at: Auditee acknowledgment timestamp.
        rca_submitted_at: Root cause analysis submission timestamp.
        cap_submitted_at: Corrective action plan submission timestamp.
        cap_approved_at: CAP approval timestamp.
        evidence_submitted_at: Closure evidence submission timestamp.
        verified_at: Verification completion timestamp.
        closed_at: CAR closure timestamp.
        corrective_action_plan: Structured CAP content.
        verification_outcome: Verification result.
        escalation_level: Current escalation level (0-4).
        escalation_history: History of escalation events.
        authority_issued: Whether CAR was issued by competent authority.
        provenance_hash: SHA-256 hash for audit trail.
    """

    model_config = ConfigDict(
        str_strip_whitespace=True,
        validate_default=True,
    )

    car_id: str = Field(
        default_factory=_new_uuid,
        description="Unique CAR identifier (UUID)",
    )
    nc_ids: List[str] = Field(
        ...,
        min_length=1,
        description="Linked non-conformance identifiers",
    )
    audit_id: str = Field(
        ...,
        description="Parent audit identifier",
    )
    supplier_id: str = Field(
        ...,
        description="Auditee supplier identifier",
    )
    severity: NCSeverity = Field(
        ...,
        description="Highest severity of linked NCs",
    )
    sla_deadline: datetime = Field(
        ...,
        description="Calculated SLA deadline from severity",
    )
    sla_status: str = Field(
        "on_track",
        description="SLA status (on_track, warning, critical, overdue)",
    )
    status: CARStatus = Field(
        CARStatus.ISSUED,
        description="Current CAR lifecycle status",
    )
    issued_by: str = Field(
        ...,
        description="Auditor or authority who issued the CAR",
    )
    issued_at: datetime = Field(
        default_factory=_utcnow,
        description="CAR issuance timestamp",
    )
    acknowledged_at: Optional[datetime] = Field(
        None,
        description="Auditee acknowledgment timestamp",
    )
    rca_submitted_at: Optional[datetime] = Field(
        None,
        description="Root cause analysis submission timestamp",
    )
    cap_submitted_at: Optional[datetime] = Field(
        None,
        description="Corrective action plan submission timestamp",
    )
    cap_approved_at: Optional[datetime] = Field(
        None,
        description="CAP approval timestamp",
    )
    evidence_submitted_at: Optional[datetime] = Field(
        None,
        description="Closure evidence submission timestamp",
    )
    verified_at: Optional[datetime] = Field(
        None,
        description="Verification completion timestamp",
    )
    closed_at: Optional[datetime] = Field(
        None,
        description="CAR closure timestamp",
    )
    corrective_action_plan: Optional[Dict[str, Any]] = Field(
        None,
        description="Structured corrective action plan content",
    )
    verification_outcome: Optional[str] = Field(
        None,
        description="Verification result (effective, not_effective)",
    )
    escalation_level: int = Field(
        0,
        ge=0,
        le=4,
        description="Current escalation level (0-4)",
    )
    escalation_history: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="History of escalation events",
    )
    authority_issued: bool = Field(
        False,
        description="Whether CAR was issued by competent authority",
    )
    provenance_hash: Optional[str] = Field(
        None,
        description="SHA-256 hash for audit trail",
    )


class CertificateRecord(BaseModel):
    """Certification scheme certificate status record.

    Tracks certificate status for FSC, PEFC, RSPO, Rainforest Alliance,
    and ISCC certifications, including scope, expiry, and audit history.

    Attributes:
        certificate_id: Unique certificate record identifier (UUID).
        scheme: Certification scheme.
        certificate_number: Official certificate number from scheme.
        holder_name: Certificate holder organization name.
        holder_id: Certificate holder supplier identifier.
        status: Certificate status.
        scope: Certificate scope description.
        certified_products: List of certified products.
        certified_sites: List of certified site identifiers.
        issue_date: Certificate issue date.
        expiry_date: Certificate expiry date.
        last_audit_date: Date of last certification audit.
        next_audit_date: Date of next scheduled audit.
        recertification_cycle_years: Recertification cycle length.
        supply_chain_model: Supply chain model (IP/SG/MB).
        provenance_hash: SHA-256 hash for audit trail.
        created_at: Record creation timestamp (UTC).
        updated_at: Record last update timestamp (UTC).
    """

    model_config = ConfigDict(
        str_strip_whitespace=True,
        validate_default=True,
    )

    certificate_id: str = Field(
        default_factory=_new_uuid,
        description="Unique certificate record identifier (UUID)",
    )
    scheme: CertificationScheme = Field(
        ...,
        description="Certification scheme",
    )
    certificate_number: str = Field(
        ...,
        min_length=1,
        max_length=200,
        description="Official certificate number from scheme",
    )
    holder_name: str = Field(
        ...,
        min_length=1,
        max_length=500,
        description="Certificate holder organization name",
    )
    holder_id: str = Field(
        ...,
        description="Certificate holder supplier identifier",
    )
    status: str = Field(
        "active",
        description="Certificate status (active, suspended, terminated, expired)",
    )
    scope: Optional[str] = Field(
        None,
        max_length=1000,
        description="Certificate scope description",
    )
    certified_products: List[str] = Field(
        default_factory=list,
        description="List of certified products",
    )
    certified_sites: List[str] = Field(
        default_factory=list,
        description="List of certified site identifiers",
    )
    issue_date: Optional[date] = Field(
        None,
        description="Certificate issue date",
    )
    expiry_date: Optional[date] = Field(
        None,
        description="Certificate expiry date",
    )
    last_audit_date: Optional[date] = Field(
        None,
        description="Date of last certification audit",
    )
    next_audit_date: Optional[date] = Field(
        None,
        description="Date of next scheduled audit",
    )
    recertification_cycle_years: int = Field(
        5,
        ge=1,
        le=10,
        description="Recertification cycle length in years",
    )
    supply_chain_model: Optional[str] = Field(
        None,
        description="Supply chain model (IP=Identity Preserved, "
        "SG=Segregated, MB=Mass Balance)",
    )
    provenance_hash: Optional[str] = Field(
        None,
        description="SHA-256 hash for audit trail",
    )
    created_at: datetime = Field(
        default_factory=_utcnow,
        description="Record creation timestamp (UTC)",
    )
    updated_at: datetime = Field(
        default_factory=_utcnow,
        description="Record last update timestamp (UTC)",
    )


class CompetentAuthorityInteraction(BaseModel):
    """EU competent authority interaction record.

    Tracks all interactions with EU Member State competent authorities
    under EUDR Articles 14-16, including document requests, inspections,
    corrective action orders, and enforcement measures.

    Attributes:
        interaction_id: Unique interaction identifier (UUID).
        operator_id: Operator subject to the interaction.
        authority_name: Name of the competent authority.
        member_state: EU Member State (ISO 3166-1 alpha-2).
        interaction_type: Type of authority interaction.
        received_date: Date interaction was received.
        response_deadline: Response deadline timestamp.
        response_sla_status: SLA compliance status.
        internal_tasks: Internal response preparation tasks.
        evidence_package_id: Compiled evidence package identifier.
        response_submitted_at: Response submission timestamp.
        authority_decision: Authority decision outcome.
        enforcement_measures: Enforcement measures issued.
        status: Interaction lifecycle status.
        notes: Interaction notes.
        provenance_hash: SHA-256 hash for audit trail.
        created_at: Record creation timestamp (UTC).
        updated_at: Record last update timestamp (UTC).
    """

    model_config = ConfigDict(
        str_strip_whitespace=True,
        validate_default=True,
    )

    interaction_id: str = Field(
        default_factory=_new_uuid,
        description="Unique interaction identifier (UUID)",
    )
    operator_id: str = Field(
        ...,
        description="Operator subject to the interaction",
    )
    authority_name: str = Field(
        ...,
        min_length=1,
        max_length=500,
        description="Name of the competent authority",
    )
    member_state: str = Field(
        ...,
        min_length=2,
        max_length=3,
        description="EU Member State (ISO 3166-1 alpha-2)",
    )
    interaction_type: AuthorityInteractionType = Field(
        ...,
        description="Type of authority interaction",
    )
    received_date: datetime = Field(
        ...,
        description="Date interaction was received",
    )
    response_deadline: datetime = Field(
        ...,
        description="Response deadline timestamp",
    )
    response_sla_status: str = Field(
        "on_track",
        description="SLA status (on_track, warning, overdue)",
    )
    internal_tasks: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Internal response preparation tasks",
    )
    evidence_package_id: Optional[str] = Field(
        None,
        description="Compiled evidence package identifier",
    )
    response_submitted_at: Optional[datetime] = Field(
        None,
        description="Response submission timestamp",
    )
    authority_decision: Optional[str] = Field(
        None,
        max_length=1000,
        description="Authority decision outcome",
    )
    enforcement_measures: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Enforcement measures issued",
    )
    status: str = Field(
        "open",
        description="Interaction status (open, in_progress, responded, closed)",
    )
    notes: Optional[str] = Field(
        None,
        max_length=5000,
        description="Interaction notes",
    )
    provenance_hash: Optional[str] = Field(
        None,
        description="SHA-256 hash for audit trail",
    )
    created_at: datetime = Field(
        default_factory=_utcnow,
        description="Record creation timestamp (UTC)",
    )
    updated_at: datetime = Field(
        default_factory=_utcnow,
        description="Record last update timestamp (UTC)",
    )

    @field_validator("member_state")
    @classmethod
    def validate_member_state(cls, v: str) -> str:
        """Ensure member_state is uppercase."""
        return v.upper()


class AuditReport(BaseModel):
    """ISO 19011:2018 compliant audit report.

    Structured audit report containing all required sections per
    ISO 19011 Clause 6.6, including objectives, scope, criteria,
    findings, conclusions, and evidence package.

    Attributes:
        report_id: Unique report identifier (UUID).
        audit_id: Parent audit identifier.
        report_version: Report version number.
        report_format: Output format (pdf/json/html/xlsx/xml).
        language: Report language (ISO 639-1).
        title: Report title.
        audit_objectives: Audit objectives statement.
        audit_scope: Audit scope description.
        audit_criteria: Audit criteria used.
        audit_client: Audit client identification.
        audit_team_summary: Audit team credentials summary.
        dates_and_locations: Audit dates and locations.
        findings_summary: Summary of findings by severity.
        findings_detail: Detailed finding sections.
        audit_conclusions: Audit conclusions.
        audit_recommendations: Recommendations for improvement.
        sampling_rationale: ISO 19011 Annex A sampling rationale.
        evidence_package_id: Evidence package archive identifier.
        distribution_list: Report distribution recipients.
        report_hash: SHA-256 hash of complete report.
        amendment_history: Report amendment history.
        generated_at: Report generation timestamp.
        provenance_hash: SHA-256 hash for audit trail.
    """

    model_config = ConfigDict(
        str_strip_whitespace=True,
        validate_default=True,
    )

    report_id: str = Field(
        default_factory=_new_uuid,
        description="Unique report identifier (UUID)",
    )
    audit_id: str = Field(
        ...,
        description="Parent audit identifier",
    )
    report_version: int = Field(
        1,
        ge=1,
        description="Report version number",
    )
    report_format: str = Field(
        "pdf",
        description="Output format (pdf, json, html, xlsx, xml)",
    )
    language: str = Field(
        "en",
        description="Report language (ISO 639-1)",
    )
    title: str = Field(
        ...,
        min_length=1,
        max_length=500,
        description="Report title",
    )
    audit_objectives: str = Field(
        "",
        max_length=5000,
        description="Audit objectives statement",
    )
    audit_scope: str = Field(
        "",
        max_length=5000,
        description="Audit scope description",
    )
    audit_criteria: List[str] = Field(
        default_factory=list,
        description="Audit criteria used",
    )
    audit_client: Optional[str] = Field(
        None,
        max_length=500,
        description="Audit client identification",
    )
    audit_team_summary: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Audit team credentials summary",
    )
    dates_and_locations: List[Dict[str, str]] = Field(
        default_factory=list,
        description="Audit dates and locations",
    )
    findings_summary: Dict[str, int] = Field(
        default_factory=lambda: {
            "critical": 0, "major": 0, "minor": 0, "observation": 0,
        },
        description="Findings count by severity",
    )
    findings_detail: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Detailed finding sections per NC",
    )
    audit_conclusions: str = Field(
        "",
        max_length=5000,
        description="Audit conclusions",
    )
    audit_recommendations: List[str] = Field(
        default_factory=list,
        description="Recommendations for improvement",
    )
    sampling_rationale: Optional[Dict[str, Any]] = Field(
        None,
        description="ISO 19011 Annex A sampling rationale",
    )
    evidence_package_id: Optional[str] = Field(
        None,
        description="Evidence package archive identifier",
    )
    distribution_list: List[str] = Field(
        default_factory=list,
        description="Report distribution recipients",
    )
    report_hash: Optional[str] = Field(
        None,
        description="SHA-256 hash of complete report content",
    )
    amendment_history: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Report amendment history",
    )
    generated_at: datetime = Field(
        default_factory=_utcnow,
        description="Report generation timestamp",
    )
    provenance_hash: Optional[str] = Field(
        None,
        description="SHA-256 hash for audit trail",
    )


# ---------------------------------------------------------------------------
# Request Models (7)
# ---------------------------------------------------------------------------


class ScheduleAuditRequest(BaseModel):
    """Request to generate risk-based audit schedule for suppliers.

    Attributes:
        operator_id: Operator identifier.
        supplier_ids: Supplier identifiers to schedule.
        planning_year: Audit planning year.
        quarter: Optional quarter (1-4) for quarterly review.
        risk_weight_overrides: Optional override for risk weights.
        request_id: Client-provided request identifier.
    """

    model_config = ConfigDict(str_strip_whitespace=True)

    operator_id: str = Field(
        ...,
        description="Operator identifier",
    )
    supplier_ids: List[str] = Field(
        ...,
        min_length=1,
        description="Supplier identifiers to schedule",
    )
    planning_year: int = Field(
        ...,
        ge=2020,
        le=2100,
        description="Audit planning year",
    )
    quarter: Optional[int] = Field(
        None,
        ge=1,
        le=4,
        description="Quarter for quarterly review",
    )
    risk_weight_overrides: Optional[Dict[str, Decimal]] = Field(
        None,
        description="Optional override for risk weights",
    )
    request_id: Optional[str] = Field(
        None,
        description="Client-provided request identifier",
    )


class MatchAuditorRequest(BaseModel):
    """Request to match qualified auditors to an audit assignment.

    Attributes:
        audit_id: Audit to match auditors for.
        commodity: EUDR commodity being audited.
        country_code: Country of the audit site.
        scheme: Certification scheme if applicable.
        scope: Required audit scope.
        required_languages: Required language capabilities.
        exclude_auditor_ids: Auditors to exclude (CoI, rotation).
        max_results: Maximum number of matched auditors.
        request_id: Client-provided request identifier.
    """

    model_config = ConfigDict(str_strip_whitespace=True)

    audit_id: str = Field(
        ...,
        description="Audit to match auditors for",
    )
    commodity: str = Field(
        ...,
        description="EUDR commodity being audited",
    )
    country_code: str = Field(
        ...,
        min_length=2,
        max_length=3,
        description="Country of the audit site",
    )
    scheme: Optional[CertificationScheme] = Field(
        None,
        description="Certification scheme if applicable",
    )
    scope: AuditScope = Field(
        AuditScope.FULL,
        description="Required audit scope",
    )
    required_languages: List[str] = Field(
        default_factory=list,
        description="Required language capabilities (ISO 639-1)",
    )
    exclude_auditor_ids: List[str] = Field(
        default_factory=list,
        description="Auditors to exclude (CoI, rotation)",
    )
    max_results: int = Field(
        10,
        ge=1,
        le=100,
        description="Maximum number of matched auditors",
    )
    request_id: Optional[str] = Field(
        None,
        description="Client-provided request identifier",
    )


class ClassifyNCRequest(BaseModel):
    """Request to classify a non-conformance finding.

    Attributes:
        audit_id: Parent audit identifier.
        finding_statement: What was found.
        objective_evidence: Supporting evidence.
        eudr_article: Mapped EUDR article.
        scheme_clause: Mapped scheme clause.
        indicators: Structured indicators for rule-based classification.
        request_id: Client-provided request identifier.
    """

    model_config = ConfigDict(str_strip_whitespace=True)

    audit_id: str = Field(
        ...,
        description="Parent audit identifier",
    )
    finding_statement: str = Field(
        ...,
        min_length=1,
        max_length=5000,
        description="Description of what was found",
    )
    objective_evidence: str = Field(
        ...,
        min_length=1,
        max_length=5000,
        description="Evidence supporting the finding",
    )
    eudr_article: Optional[str] = Field(
        None,
        description="Mapped EUDR article",
    )
    scheme_clause: Optional[str] = Field(
        None,
        description="Mapped certification scheme clause",
    )
    indicators: Dict[str, Any] = Field(
        default_factory=dict,
        description="Structured indicators for rule-based classification",
    )
    request_id: Optional[str] = Field(
        None,
        description="Client-provided request identifier",
    )


class IssueCARRequest(BaseModel):
    """Request to issue a corrective action request.

    Attributes:
        nc_ids: Non-conformance identifiers to link.
        audit_id: Parent audit identifier.
        supplier_id: Auditee supplier identifier.
        issued_by: Auditor or authority issuing the CAR.
        authority_issued: Whether CAR is authority-issued.
        custom_sla_days: Optional custom SLA override.
        request_id: Client-provided request identifier.
    """

    model_config = ConfigDict(str_strip_whitespace=True)

    nc_ids: List[str] = Field(
        ...,
        min_length=1,
        description="Non-conformance identifiers to link",
    )
    audit_id: str = Field(
        ...,
        description="Parent audit identifier",
    )
    supplier_id: str = Field(
        ...,
        description="Auditee supplier identifier",
    )
    issued_by: str = Field(
        ...,
        description="Auditor or authority issuing the CAR",
    )
    authority_issued: bool = Field(
        False,
        description="Whether CAR is authority-issued (Art. 18)",
    )
    custom_sla_days: Optional[int] = Field(
        None,
        ge=1,
        description="Custom SLA deadline override (days)",
    )
    request_id: Optional[str] = Field(
        None,
        description="Client-provided request identifier",
    )


class GenerateReportRequest(BaseModel):
    """Request to generate an ISO 19011 compliant audit report.

    Attributes:
        audit_id: Audit to generate report for.
        report_format: Desired output format.
        language: Desired report language.
        include_evidence_package: Include evidence package archive.
        distribution_list: Report distribution recipients.
        request_id: Client-provided request identifier.
    """

    model_config = ConfigDict(str_strip_whitespace=True)

    audit_id: str = Field(
        ...,
        description="Audit to generate report for",
    )
    report_format: str = Field(
        "pdf",
        description="Desired output format (pdf, json, html, xlsx, xml)",
    )
    language: str = Field(
        "en",
        description="Desired report language (en, fr, de, es, pt)",
    )
    include_evidence_package: bool = Field(
        True,
        description="Include evidence package archive",
    )
    distribution_list: List[str] = Field(
        default_factory=list,
        description="Report distribution recipients",
    )
    request_id: Optional[str] = Field(
        None,
        description="Client-provided request identifier",
    )


class LogAuthorityInteractionRequest(BaseModel):
    """Request to log a competent authority interaction.

    Attributes:
        operator_id: Operator subject to the interaction.
        authority_name: Name of the competent authority.
        member_state: EU Member State code.
        interaction_type: Type of authority interaction.
        received_date: Date interaction was received.
        response_deadline_days: Response deadline in days.
        notes: Interaction notes.
        request_id: Client-provided request identifier.
    """

    model_config = ConfigDict(str_strip_whitespace=True)

    operator_id: str = Field(
        ...,
        description="Operator subject to the interaction",
    )
    authority_name: str = Field(
        ...,
        min_length=1,
        description="Name of the competent authority",
    )
    member_state: str = Field(
        ...,
        min_length=2,
        max_length=3,
        description="EU Member State code (ISO 3166-1 alpha-2)",
    )
    interaction_type: AuthorityInteractionType = Field(
        ...,
        description="Type of authority interaction",
    )
    received_date: Optional[datetime] = Field(
        None,
        description="Date interaction was received",
    )
    response_deadline_days: Optional[int] = Field(
        None,
        ge=1,
        description="Response deadline in days from received_date",
    )
    notes: Optional[str] = Field(
        None,
        max_length=5000,
        description="Interaction notes",
    )
    request_id: Optional[str] = Field(
        None,
        description="Client-provided request identifier",
    )


class CalculateAnalyticsRequest(BaseModel):
    """Request to calculate audit analytics and trends.

    Attributes:
        operator_id: Operator to analyze.
        time_period_start: Analysis period start date.
        time_period_end: Analysis period end date.
        group_by: Grouping dimensions.
        metrics: Metrics to calculate.
        request_id: Client-provided request identifier.
    """

    model_config = ConfigDict(str_strip_whitespace=True)

    operator_id: str = Field(
        ...,
        description="Operator to analyze",
    )
    time_period_start: Optional[date] = Field(
        None,
        description="Analysis period start date",
    )
    time_period_end: Optional[date] = Field(
        None,
        description="Analysis period end date",
    )
    group_by: List[str] = Field(
        default_factory=lambda: ["severity", "region", "commodity"],
        description="Grouping dimensions (severity, region, commodity, "
        "scheme, supplier)",
    )
    metrics: List[str] = Field(
        default_factory=lambda: [
            "finding_trends", "car_performance", "compliance_rate",
            "auditor_performance", "cost_analysis",
        ],
        description="Metrics to calculate",
    )
    request_id: Optional[str] = Field(
        None,
        description="Client-provided request identifier",
    )


# ---------------------------------------------------------------------------
# Response Models (7)
# ---------------------------------------------------------------------------


class ScheduleAuditResponse(BaseModel):
    """Response from audit scheduling engine.

    Attributes:
        scheduled_audits: List of scheduled audit records.
        total_scheduled: Total number of audits scheduled.
        risk_distribution: Distribution by risk tier.
        resource_summary: Resource allocation summary.
        conflicts_detected: Scheduling conflicts found.
        processing_time_ms: Processing duration in milliseconds.
        provenance_hash: SHA-256 hash for audit trail.
        request_id: Client-provided request identifier.
    """

    model_config = ConfigDict(str_strip_whitespace=True)

    scheduled_audits: List[Audit] = Field(
        default_factory=list,
        description="List of scheduled audit records",
    )
    total_scheduled: int = Field(
        0,
        ge=0,
        description="Total number of audits scheduled",
    )
    risk_distribution: Dict[str, int] = Field(
        default_factory=lambda: {"HIGH": 0, "STANDARD": 0, "LOW": 0},
        description="Distribution by risk tier",
    )
    resource_summary: Dict[str, Any] = Field(
        default_factory=dict,
        description="Resource allocation summary (auditor-days)",
    )
    conflicts_detected: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Scheduling conflicts found",
    )
    processing_time_ms: Decimal = Field(
        Decimal("0"),
        ge=Decimal("0"),
        description="Processing duration in milliseconds",
    )
    provenance_hash: Optional[str] = Field(
        None,
        description="SHA-256 hash for audit trail",
    )
    request_id: Optional[str] = Field(
        None,
        description="Client-provided request identifier",
    )


class MatchAuditorResponse(BaseModel):
    """Response from auditor matching engine.

    Attributes:
        matched_auditors: Ranked list of qualified auditors.
        total_matches: Total number of matching auditors.
        match_criteria: Criteria used for matching.
        processing_time_ms: Processing duration in milliseconds.
        provenance_hash: SHA-256 hash for audit trail.
        request_id: Client-provided request identifier.
    """

    model_config = ConfigDict(str_strip_whitespace=True)

    matched_auditors: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Ranked list of qualified auditors with scores",
    )
    total_matches: int = Field(
        0,
        ge=0,
        description="Total number of matching auditors",
    )
    match_criteria: Dict[str, Any] = Field(
        default_factory=dict,
        description="Criteria used for matching",
    )
    processing_time_ms: Decimal = Field(
        Decimal("0"),
        ge=Decimal("0"),
        description="Processing duration in milliseconds",
    )
    provenance_hash: Optional[str] = Field(
        None,
        description="SHA-256 hash for audit trail",
    )
    request_id: Optional[str] = Field(
        None,
        description="Client-provided request identifier",
    )


class ClassifyNCResponse(BaseModel):
    """Response from non-conformance classification engine.

    Attributes:
        non_conformance: Classified non-conformance record.
        classification_rationale: Rule-based classification rationale.
        matched_rules: Rules that triggered the classification.
        processing_time_ms: Processing duration in milliseconds.
        provenance_hash: SHA-256 hash for audit trail.
        request_id: Client-provided request identifier.
    """

    model_config = ConfigDict(str_strip_whitespace=True)

    non_conformance: NonConformance = Field(
        ...,
        description="Classified non-conformance record",
    )
    classification_rationale: str = Field(
        "",
        description="Rule-based classification rationale",
    )
    matched_rules: List[str] = Field(
        default_factory=list,
        description="Rule identifiers that triggered the classification",
    )
    processing_time_ms: Decimal = Field(
        Decimal("0"),
        ge=Decimal("0"),
        description="Processing duration in milliseconds",
    )
    provenance_hash: Optional[str] = Field(
        None,
        description="SHA-256 hash for audit trail",
    )
    request_id: Optional[str] = Field(
        None,
        description="Client-provided request identifier",
    )


class IssueCARResponse(BaseModel):
    """Response from CAR issuance.

    Attributes:
        car: Issued corrective action request.
        sla_details: SLA deadline details.
        processing_time_ms: Processing duration in milliseconds.
        provenance_hash: SHA-256 hash for audit trail.
        request_id: Client-provided request identifier.
    """

    model_config = ConfigDict(str_strip_whitespace=True)

    car: CorrectiveActionRequest = Field(
        ...,
        description="Issued corrective action request",
    )
    sla_details: Dict[str, Any] = Field(
        default_factory=dict,
        description="SLA deadline details (acknowledge, rca, cap deadlines)",
    )
    processing_time_ms: Decimal = Field(
        Decimal("0"),
        ge=Decimal("0"),
        description="Processing duration in milliseconds",
    )
    provenance_hash: Optional[str] = Field(
        None,
        description="SHA-256 hash for audit trail",
    )
    request_id: Optional[str] = Field(
        None,
        description="Client-provided request identifier",
    )


class GenerateReportResponse(BaseModel):
    """Response from audit report generation.

    Attributes:
        report: Generated audit report.
        report_size_bytes: Report file size in bytes.
        generation_time_ms: Report generation duration.
        processing_time_ms: Total processing duration in milliseconds.
        provenance_hash: SHA-256 hash for audit trail.
        request_id: Client-provided request identifier.
    """

    model_config = ConfigDict(str_strip_whitespace=True)

    report: AuditReport = Field(
        ...,
        description="Generated audit report",
    )
    report_size_bytes: int = Field(
        0,
        ge=0,
        description="Report file size in bytes",
    )
    generation_time_ms: Decimal = Field(
        Decimal("0"),
        ge=Decimal("0"),
        description="Report generation duration in milliseconds",
    )
    processing_time_ms: Decimal = Field(
        Decimal("0"),
        ge=Decimal("0"),
        description="Total processing duration in milliseconds",
    )
    provenance_hash: Optional[str] = Field(
        None,
        description="SHA-256 hash for audit trail",
    )
    request_id: Optional[str] = Field(
        None,
        description="Client-provided request identifier",
    )


class LogAuthorityInteractionResponse(BaseModel):
    """Response from competent authority interaction logging.

    Attributes:
        interaction: Logged authority interaction record.
        sla_details: Response SLA details.
        processing_time_ms: Processing duration in milliseconds.
        provenance_hash: SHA-256 hash for audit trail.
        request_id: Client-provided request identifier.
    """

    model_config = ConfigDict(str_strip_whitespace=True)

    interaction: CompetentAuthorityInteraction = Field(
        ...,
        description="Logged authority interaction record",
    )
    sla_details: Dict[str, Any] = Field(
        default_factory=dict,
        description="Response SLA details",
    )
    processing_time_ms: Decimal = Field(
        Decimal("0"),
        ge=Decimal("0"),
        description="Processing duration in milliseconds",
    )
    provenance_hash: Optional[str] = Field(
        None,
        description="SHA-256 hash for audit trail",
    )
    request_id: Optional[str] = Field(
        None,
        description="Client-provided request identifier",
    )


class CalculateAnalyticsResponse(BaseModel):
    """Response from audit analytics calculation.

    Attributes:
        finding_trends: NC finding trends over time.
        car_performance: CAR closure performance metrics.
        compliance_rate: Compliance rate tracking data.
        auditor_performance: Auditor performance benchmarks.
        cost_analysis: Audit cost analysis data.
        scheme_analysis: Certification scheme analysis.
        authority_analytics: Competent authority interaction analytics.
        summary_kpis: Summary KPI values.
        processing_time_ms: Processing duration in milliseconds.
        provenance_hash: SHA-256 hash for audit trail.
        request_id: Client-provided request identifier.
    """

    model_config = ConfigDict(str_strip_whitespace=True)

    finding_trends: Dict[str, Any] = Field(
        default_factory=dict,
        description="NC finding trends over time",
    )
    car_performance: Dict[str, Any] = Field(
        default_factory=dict,
        description="CAR closure performance metrics",
    )
    compliance_rate: Dict[str, Any] = Field(
        default_factory=dict,
        description="Compliance rate tracking data",
    )
    auditor_performance: Dict[str, Any] = Field(
        default_factory=dict,
        description="Auditor performance benchmarks",
    )
    cost_analysis: Dict[str, Any] = Field(
        default_factory=dict,
        description="Audit cost analysis data",
    )
    scheme_analysis: Dict[str, Any] = Field(
        default_factory=dict,
        description="Certification scheme analysis",
    )
    authority_analytics: Dict[str, Any] = Field(
        default_factory=dict,
        description="Competent authority interaction analytics",
    )
    summary_kpis: Dict[str, Any] = Field(
        default_factory=dict,
        description="Summary KPI values",
    )
    processing_time_ms: Decimal = Field(
        Decimal("0"),
        ge=Decimal("0"),
        description="Processing duration in milliseconds",
    )
    provenance_hash: Optional[str] = Field(
        None,
        description="SHA-256 hash for audit trail",
    )
    request_id: Optional[str] = Field(
        None,
        description="Client-provided request identifier",
    )
