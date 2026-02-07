# -*- coding: utf-8 -*-
"""
Compliance Automation Data Models - SEC-010 Phase 5

Pydantic v2 models for the GreenLang multi-compliance automation system.
Provides strongly-typed data structures for compliance frameworks, control
mappings, DSAR requests, consent records, data records, and compliance reports.

All datetime fields use UTC. All models enforce strict validation via Pydantic v2
field validators and model configuration.

Models:
    - ComplianceFramework: Enum of supported compliance frameworks
    - ComplianceStatus: Current compliance status for a framework
    - ControlMapping: Mapping between framework controls and technical controls
    - ControlStatus: Status of a specific control
    - DSARRequest: Data Subject Access Request model
    - DSARType: Enum of DSAR request types (Art. 15-22)
    - DSARStatus: Enum of DSAR processing statuses
    - ConsentRecord: User consent tracking record
    - ConsentPurpose: Enum of consent purposes
    - DataRecord: Discovered user data record
    - DataCategory: Enum of data categories
    - RetentionPolicy: Data retention policy definition
    - EvidenceSource: Compliance evidence source
    - ComplianceReport: Compliance assessment report
    - ComplianceGap: Identified compliance gap

Author: GreenLang Security Team
Date: February 2026
PRD: SEC-010 Security Operations Automation Platform
"""

from __future__ import annotations

import re
import uuid
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------


class ComplianceFramework(str, Enum):
    """Supported compliance frameworks.

    GreenLang supports continuous compliance monitoring for multiple
    regulatory and security frameworks.
    """

    ISO27001 = "iso27001"
    """ISO 27001:2022 Information Security Management System."""

    GDPR = "gdpr"
    """EU General Data Protection Regulation."""

    PCI_DSS = "pci_dss"
    """Payment Card Industry Data Security Standard v4.0."""

    CCPA = "ccpa"
    """California Consumer Privacy Act."""

    LGPD = "lgpd"
    """Brazil's Lei Geral de Protecao de Dados."""

    SOC2 = "soc2"
    """SOC 2 Type II (Trust Services Criteria)."""

    HIPAA = "hipaa"
    """Health Insurance Portability and Accountability Act."""


class ControlStatus(str, Enum):
    """Status of a compliance control."""

    NOT_APPLICABLE = "not_applicable"
    """Control is not applicable to the organization."""

    NOT_IMPLEMENTED = "not_implemented"
    """Control has not been implemented."""

    PARTIALLY_IMPLEMENTED = "partially_implemented"
    """Control is partially implemented."""

    IMPLEMENTED = "implemented"
    """Control is fully implemented but not verified."""

    VERIFIED = "verified"
    """Control is implemented and verified through testing."""

    NON_COMPLIANT = "non_compliant"
    """Control failed compliance check."""


class DSARType(str, Enum):
    """GDPR Data Subject Access Request types (Articles 15-22).

    Each type corresponds to a specific right under GDPR that data
    subjects can exercise.
    """

    ACCESS = "access"
    """Article 15 - Right of Access by the data subject."""

    RECTIFICATION = "rectification"
    """Article 16 - Right to rectification."""

    ERASURE = "erasure"
    """Article 17 - Right to erasure (right to be forgotten)."""

    RESTRICTION = "restriction"
    """Article 18 - Right to restriction of processing."""

    PORTABILITY = "portability"
    """Article 20 - Right to data portability."""

    OBJECTION = "objection"
    """Article 21 - Right to object."""


class DSARStatus(str, Enum):
    """Status of a DSAR request throughout its lifecycle."""

    SUBMITTED = "submitted"
    """Request has been received but not yet processed."""

    VERIFYING = "verifying"
    """Identity verification is in progress."""

    IDENTITY_VERIFIED = "identity_verified"
    """Identity has been verified, ready for processing."""

    PROCESSING = "processing"
    """Request is being processed (data discovery/compilation)."""

    PENDING_REVIEW = "pending_review"
    """Processing complete, awaiting review before delivery."""

    COMPLETED = "completed"
    """Request has been fulfilled and delivered to the subject."""

    REJECTED = "rejected"
    """Request was rejected (invalid, duplicate, or unverified)."""

    EXTENDED = "extended"
    """Request has been extended due to complexity."""

    CANCELLED = "cancelled"
    """Request was cancelled by the data subject."""


class ConsentPurpose(str, Enum):
    """Standard purposes for which consent may be collected."""

    MARKETING = "marketing"
    """Marketing communications and promotions."""

    ANALYTICS = "analytics"
    """Analytics and performance measurement."""

    PERSONALIZATION = "personalization"
    """Personalized content and recommendations."""

    THIRD_PARTY_SHARING = "third_party_sharing"
    """Sharing data with third parties."""

    RESEARCH = "research"
    """Research and product improvement."""

    ESSENTIAL = "essential"
    """Essential service functionality (often implied consent)."""

    CARBON_REPORTING = "carbon_reporting"
    """Carbon footprint reporting and disclosure."""

    SUPPLY_CHAIN_VISIBILITY = "supply_chain_visibility"
    """Supply chain transparency and tracking."""


class DataCategory(str, Enum):
    """Categories of data for retention and compliance purposes."""

    PII = "pii"
    """Personally Identifiable Information."""

    SENSITIVE_PII = "sensitive_pii"
    """Sensitive PII (race, religion, health, etc.)."""

    FINANCIAL = "financial"
    """Financial data (payment info, transactions)."""

    OPERATIONAL = "operational"
    """Operational/business data."""

    AUDIT = "audit"
    """Audit logs and compliance records."""

    SECURITY = "security"
    """Security logs and alerts."""

    CONSENT = "consent"
    """Consent records."""

    BACKUP = "backup"
    """Backup data."""

    EMISSIONS = "emissions"
    """Carbon emissions data."""

    SUPPLY_CHAIN = "supply_chain"
    """Supply chain and vendor data."""


# ---------------------------------------------------------------------------
# Core Models
# ---------------------------------------------------------------------------


class ComplianceStatus(BaseModel):
    """Current compliance status for a framework.

    Represents the overall compliance posture for a specific regulatory
    or security framework, including scores, gaps, and control counts.

    Attributes:
        framework: The compliance framework.
        score: Overall compliance score (0-100).
        status: Overall status (compliant, partial, non_compliant).
        gaps: List of identified compliance gaps.
        controls_total: Total number of controls in the framework.
        controls_compliant: Number of compliant controls.
        controls_non_compliant: Number of non-compliant controls.
        controls_not_applicable: Number of not-applicable controls.
        last_assessed: Timestamp of last assessment.
        next_assessment: Scheduled next assessment.
        assessor_id: ID of the user/system that performed the assessment.
    """

    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
        validate_default=True,
    )

    framework: ComplianceFramework = Field(
        ...,
        description="The compliance framework being assessed.",
    )
    score: float = Field(
        default=0.0,
        ge=0.0,
        le=100.0,
        description="Overall compliance score (0-100).",
    )
    status: str = Field(
        default="not_assessed",
        description="Overall status: compliant, partial, non_compliant, not_assessed.",
    )
    gaps: List[str] = Field(
        default_factory=list,
        description="List of identified compliance gap IDs.",
    )
    controls_total: int = Field(
        default=0,
        ge=0,
        description="Total number of controls in the framework.",
    )
    controls_compliant: int = Field(
        default=0,
        ge=0,
        description="Number of compliant controls.",
    )
    controls_non_compliant: int = Field(
        default=0,
        ge=0,
        description="Number of non-compliant controls.",
    )
    controls_not_applicable: int = Field(
        default=0,
        ge=0,
        description="Number of not-applicable controls.",
    )
    last_assessed: Optional[datetime] = Field(
        default=None,
        description="Timestamp of last assessment (UTC).",
    )
    next_assessment: Optional[datetime] = Field(
        default=None,
        description="Scheduled next assessment (UTC).",
    )
    assessor_id: Optional[str] = Field(
        default=None,
        max_length=256,
        description="ID of the user/system that performed the assessment.",
    )

    @field_validator("status")
    @classmethod
    def validate_status(cls, v: str) -> str:
        """Validate status is one of the allowed values."""
        allowed = {"compliant", "partial", "non_compliant", "not_assessed"}
        v_lower = v.strip().lower()
        if v_lower not in allowed:
            raise ValueError(
                f"Invalid status '{v}'. Allowed values: {sorted(allowed)}"
            )
        return v_lower

    @field_validator("last_assessed", "next_assessment")
    @classmethod
    def ensure_utc_datetime(cls, v: Optional[datetime]) -> Optional[datetime]:
        """Ensure datetime values are timezone-aware UTC."""
        if v is None:
            return v
        if v.tzinfo is None:
            return v.replace(tzinfo=timezone.utc)
        return v.astimezone(timezone.utc)


class ControlMapping(BaseModel):
    """Mapping between a framework control and technical controls.

    Maps abstract compliance framework controls (e.g., ISO 27001 A.8.5)
    to concrete technical controls implemented in GreenLang (e.g., SEC-001
    authentication controls).

    Attributes:
        id: Unique mapping identifier.
        framework: The compliance framework.
        framework_control: The control ID within the framework.
        framework_control_name: Human-readable control name.
        technical_controls: List of technical control IDs that implement this.
        evidence_sources: Sources of compliance evidence.
        status: Current implementation status.
        notes: Implementation notes.
        last_verified: Last verification timestamp.
    """

    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
    )

    id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique mapping identifier.",
    )
    framework: ComplianceFramework = Field(
        ...,
        description="The compliance framework.",
    )
    framework_control: str = Field(
        ...,
        min_length=1,
        max_length=50,
        description="The control ID within the framework (e.g., A.8.5).",
    )
    framework_control_name: str = Field(
        default="",
        max_length=256,
        description="Human-readable control name.",
    )
    technical_controls: List[str] = Field(
        default_factory=list,
        description="List of technical control IDs (e.g., ['SEC-001', 'SEC-002']).",
    )
    evidence_sources: List[str] = Field(
        default_factory=list,
        description="Sources of compliance evidence.",
    )
    status: ControlStatus = Field(
        default=ControlStatus.NOT_IMPLEMENTED,
        description="Current implementation status.",
    )
    notes: str = Field(
        default="",
        max_length=4096,
        description="Implementation notes.",
    )
    last_verified: Optional[datetime] = Field(
        default=None,
        description="Last verification timestamp (UTC).",
    )


class DSARRequest(BaseModel):
    """Data Subject Access Request (GDPR).

    Represents a request from a data subject to exercise their rights
    under GDPR Articles 15-22. Includes identity verification, processing
    status, and discovered data.

    Attributes:
        id: Unique request identifier (UUID).
        request_number: Human-readable request number (e.g., DSAR-2026-0001).
        request_type: Type of DSAR (access, erasure, etc.).
        subject_email: Email address of the data subject.
        subject_name: Name of the data subject.
        subject_id: Internal user ID if known.
        status: Current processing status.
        submitted_at: Request submission timestamp.
        due_date: SLA due date (30 days from submission).
        identity_verified_at: When identity was verified.
        verification_method: Method used to verify identity.
        processing_started_at: When processing began.
        completed_at: When request was fulfilled.
        data_discovered: Summary of discovered data.
        data_sources_scanned: List of data sources that were scanned.
        actions_taken: List of actions taken to fulfill the request.
        deletion_certificate_id: ID of deletion certificate (for erasure).
        export_file_url: URL to download exported data.
        assignee_id: ID of the staff member handling the request.
        notes: Internal processing notes.
        extension_reason: Reason for SLA extension (if extended).
    """

    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
        validate_default=True,
    )

    id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique request identifier.",
    )
    request_number: str = Field(
        default="",
        max_length=30,
        description="Human-readable request number (e.g., DSAR-2026-0001).",
    )
    request_type: DSARType = Field(
        ...,
        description="Type of DSAR request (Art. 15-22).",
    )
    subject_email: str = Field(
        ...,
        min_length=5,
        max_length=255,
        description="Email address of the data subject.",
    )
    subject_name: str = Field(
        default="",
        max_length=256,
        description="Name of the data subject.",
    )
    subject_id: Optional[str] = Field(
        default=None,
        max_length=256,
        description="Internal user ID if known.",
    )
    status: DSARStatus = Field(
        default=DSARStatus.SUBMITTED,
        description="Current processing status.",
    )
    submitted_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Request submission timestamp (UTC).",
    )
    due_date: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc) + timedelta(days=30),
        description="SLA due date (30 days from submission).",
    )
    identity_verified_at: Optional[datetime] = Field(
        default=None,
        description="When identity was verified (UTC).",
    )
    verification_method: Optional[str] = Field(
        default=None,
        max_length=100,
        description="Method used to verify identity (email, id_document, etc.).",
    )
    processing_started_at: Optional[datetime] = Field(
        default=None,
        description="When processing began (UTC).",
    )
    completed_at: Optional[datetime] = Field(
        default=None,
        description="When request was fulfilled (UTC).",
    )
    data_discovered: Dict[str, Any] = Field(
        default_factory=dict,
        description="Summary of discovered data by category.",
    )
    data_sources_scanned: List[str] = Field(
        default_factory=list,
        description="List of data sources that were scanned.",
    )
    actions_taken: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="List of actions taken to fulfill the request.",
    )
    deletion_certificate_id: Optional[str] = Field(
        default=None,
        description="ID of deletion certificate (for erasure requests).",
    )
    export_file_url: Optional[str] = Field(
        default=None,
        max_length=2048,
        description="URL to download exported data.",
    )
    assignee_id: Optional[str] = Field(
        default=None,
        max_length=256,
        description="ID of the staff member handling the request.",
    )
    notes: str = Field(
        default="",
        max_length=4096,
        description="Internal processing notes.",
    )
    extension_reason: Optional[str] = Field(
        default=None,
        max_length=1024,
        description="Reason for SLA extension (if extended).",
    )

    @field_validator("subject_email")
    @classmethod
    def validate_email(cls, v: str) -> str:
        """Basic email format validation."""
        email_pattern = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
        if not email_pattern.match(v.strip()):
            raise ValueError(f"Invalid email format: {v}")
        return v.strip().lower()

    @field_validator("submitted_at", "due_date", "identity_verified_at",
                     "processing_started_at", "completed_at")
    @classmethod
    def ensure_utc_datetime(cls, v: Optional[datetime]) -> Optional[datetime]:
        """Ensure datetime values are timezone-aware UTC."""
        if v is None:
            return v
        if v.tzinfo is None:
            return v.replace(tzinfo=timezone.utc)
        return v.astimezone(timezone.utc)

    @model_validator(mode="after")
    def generate_request_number(self) -> "DSARRequest":
        """Generate request number if not provided."""
        if not self.request_number:
            year = self.submitted_at.year
            short_id = self.id[:8].upper()
            object.__setattr__(
                self, "request_number", f"DSAR-{year}-{short_id}"
            )
        return self

    @property
    def is_overdue(self) -> bool:
        """Check if the request is past its due date."""
        return datetime.now(timezone.utc) > self.due_date

    @property
    def days_remaining(self) -> int:
        """Days remaining until due date (negative if overdue)."""
        delta = self.due_date - datetime.now(timezone.utc)
        return delta.days


class ConsentRecord(BaseModel):
    """Record of user consent for a specific purpose.

    Tracks when consent was granted, by whom, through which mechanism,
    and when/if it was revoked.

    Attributes:
        id: Unique consent record identifier.
        user_id: The user who granted consent.
        user_email: User's email address.
        purpose: The purpose for which consent was granted.
        granted_at: When consent was granted.
        revoked_at: When consent was revoked (None if active).
        source: How consent was obtained (web_form, api, import).
        source_details: Additional details about consent source.
        ip_address: IP address at time of consent.
        user_agent: Browser/client user agent.
        consent_version: Version of the consent text presented.
        metadata: Additional metadata about the consent.
    """

    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
    )

    id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique consent record identifier.",
    )
    user_id: str = Field(
        ...,
        min_length=1,
        max_length=256,
        description="The user who granted consent.",
    )
    user_email: Optional[str] = Field(
        default=None,
        max_length=255,
        description="User's email address.",
    )
    purpose: ConsentPurpose = Field(
        ...,
        description="The purpose for which consent was granted.",
    )
    granted_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="When consent was granted (UTC).",
    )
    revoked_at: Optional[datetime] = Field(
        default=None,
        description="When consent was revoked (UTC). None if still active.",
    )
    source: str = Field(
        default="web_form",
        max_length=100,
        description="How consent was obtained (web_form, api, import).",
    )
    source_details: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional details about consent source.",
    )
    ip_address: Optional[str] = Field(
        default=None,
        max_length=45,
        description="IP address at time of consent.",
    )
    user_agent: Optional[str] = Field(
        default=None,
        max_length=512,
        description="Browser/client user agent.",
    )
    consent_version: str = Field(
        default="1.0",
        max_length=20,
        description="Version of the consent text presented.",
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata about the consent.",
    )

    @property
    def is_active(self) -> bool:
        """Check if consent is currently active (not revoked)."""
        return self.revoked_at is None


class DataRecord(BaseModel):
    """Discovered user data record.

    Represents a piece of data discovered during DSAR processing or
    PII scanning. Used to compile data for access requests or identify
    data for erasure.

    Attributes:
        id: Unique record identifier.
        user_id: The user this data belongs to.
        source_system: System where data was found (postgres, s3, logs).
        source_location: Specific location (table name, bucket, etc.).
        data_type: Type of data (profile, transaction, log, etc.).
        data_category: Category of data (pii, financial, etc.).
        record_data: The actual data (may be redacted for display).
        discovered_at: When this data was discovered.
        last_accessed_at: When the data was last accessed.
        retention_expires_at: When data should be deleted.
        pii_fields: List of fields containing PII.
        is_sensitive: Whether this contains sensitive data.
    """

    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
    )

    id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique record identifier.",
    )
    user_id: str = Field(
        ...,
        min_length=1,
        max_length=256,
        description="The user this data belongs to.",
    )
    source_system: str = Field(
        ...,
        min_length=1,
        max_length=100,
        description="System where data was found (postgres, s3, logs).",
    )
    source_location: str = Field(
        ...,
        min_length=1,
        max_length=512,
        description="Specific location (table name, bucket path, etc.).",
    )
    data_type: str = Field(
        default="unknown",
        max_length=100,
        description="Type of data (profile, transaction, log, etc.).",
    )
    data_category: DataCategory = Field(
        default=DataCategory.OPERATIONAL,
        description="Category of data for retention purposes.",
    )
    record_data: Dict[str, Any] = Field(
        default_factory=dict,
        description="The actual data (may be redacted for display).",
    )
    discovered_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="When this data was discovered (UTC).",
    )
    last_accessed_at: Optional[datetime] = Field(
        default=None,
        description="When the data was last accessed (UTC).",
    )
    retention_expires_at: Optional[datetime] = Field(
        default=None,
        description="When data should be deleted (UTC).",
    )
    pii_fields: List[str] = Field(
        default_factory=list,
        description="List of fields containing PII.",
    )
    is_sensitive: bool = Field(
        default=False,
        description="Whether this contains sensitive data.",
    )


class RetentionPolicy(BaseModel):
    """Data retention policy definition.

    Defines how long data of a specific category should be retained
    and what action to take when the retention period expires.

    Attributes:
        id: Unique policy identifier.
        name: Human-readable policy name.
        data_category: Category of data this policy applies to.
        retention_days: Number of days to retain data.
        action: Action to take on expiry (delete, archive, anonymize).
        enabled: Whether this policy is active.
        last_enforced: When this policy was last enforced.
        exceptions: List of exception rules.
    """

    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
    )

    id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique policy identifier.",
    )
    name: str = Field(
        ...,
        min_length=1,
        max_length=256,
        description="Human-readable policy name.",
    )
    data_category: DataCategory = Field(
        ...,
        description="Category of data this policy applies to.",
    )
    retention_days: int = Field(
        ...,
        ge=1,
        le=3650,  # Max 10 years
        description="Number of days to retain data.",
    )
    action: str = Field(
        default="delete",
        description="Action to take on expiry: delete, archive, anonymize.",
    )
    enabled: bool = Field(
        default=True,
        description="Whether this policy is active.",
    )
    last_enforced: Optional[datetime] = Field(
        default=None,
        description="When this policy was last enforced (UTC).",
    )
    exceptions: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="List of exception rules.",
    )

    @field_validator("action")
    @classmethod
    def validate_action(cls, v: str) -> str:
        """Validate action is one of the allowed values."""
        allowed = {"delete", "archive", "anonymize"}
        v_lower = v.strip().lower()
        if v_lower not in allowed:
            raise ValueError(
                f"Invalid action '{v}'. Allowed values: {sorted(allowed)}"
            )
        return v_lower


class EvidenceSource(BaseModel):
    """Compliance evidence source definition.

    Defines where to collect evidence for a specific compliance control.

    Attributes:
        id: Unique evidence source identifier.
        name: Human-readable source name.
        source_type: Type of source (config, log, api, document).
        location: Where to find the evidence.
        query: Query or path to extract evidence.
        format: Expected format of evidence.
        collection_method: How to collect (automated, manual).
        last_collected: When evidence was last collected.
    """

    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
    )

    id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique evidence source identifier.",
    )
    name: str = Field(
        ...,
        min_length=1,
        max_length=256,
        description="Human-readable source name.",
    )
    source_type: str = Field(
        default="config",
        max_length=50,
        description="Type of source: config, log, api, document, screenshot.",
    )
    location: str = Field(
        default="",
        max_length=1024,
        description="Where to find the evidence.",
    )
    query: str = Field(
        default="",
        max_length=2048,
        description="Query or path to extract evidence.",
    )
    format: str = Field(
        default="json",
        max_length=50,
        description="Expected format of evidence.",
    )
    collection_method: str = Field(
        default="automated",
        description="How to collect: automated, manual.",
    )
    last_collected: Optional[datetime] = Field(
        default=None,
        description="When evidence was last collected (UTC).",
    )


class ComplianceReport(BaseModel):
    """Compliance assessment report.

    A comprehensive report of compliance status for one or more frameworks.

    Attributes:
        id: Unique report identifier.
        report_type: Type of report (assessment, audit, gap_analysis).
        frameworks: Frameworks covered in this report.
        generated_at: When the report was generated.
        generated_by: Who/what generated the report.
        period_start: Start of the assessment period.
        period_end: End of the assessment period.
        overall_score: Overall compliance score across frameworks.
        framework_scores: Scores per framework.
        executive_summary: High-level summary for executives.
        findings: Detailed findings.
        recommendations: Recommendations for improvement.
        attachments: List of attached evidence files.
    """

    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
    )

    id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique report identifier.",
    )
    report_type: str = Field(
        default="assessment",
        description="Type of report: assessment, audit, gap_analysis.",
    )
    frameworks: List[ComplianceFramework] = Field(
        default_factory=list,
        description="Frameworks covered in this report.",
    )
    generated_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="When the report was generated (UTC).",
    )
    generated_by: str = Field(
        default="system",
        max_length=256,
        description="Who/what generated the report.",
    )
    period_start: Optional[datetime] = Field(
        default=None,
        description="Start of the assessment period (UTC).",
    )
    period_end: Optional[datetime] = Field(
        default=None,
        description="End of the assessment period (UTC).",
    )
    overall_score: float = Field(
        default=0.0,
        ge=0.0,
        le=100.0,
        description="Overall compliance score across frameworks.",
    )
    framework_scores: Dict[str, float] = Field(
        default_factory=dict,
        description="Scores per framework.",
    )
    executive_summary: str = Field(
        default="",
        max_length=8192,
        description="High-level summary for executives.",
    )
    findings: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Detailed findings.",
    )
    recommendations: List[str] = Field(
        default_factory=list,
        description="Recommendations for improvement.",
    )
    attachments: List[str] = Field(
        default_factory=list,
        description="List of attached evidence files.",
    )


class ComplianceGap(BaseModel):
    """Identified compliance gap.

    Represents a gap between current state and compliance requirements.

    Attributes:
        id: Unique gap identifier.
        framework: The compliance framework.
        control_id: The control with the gap.
        title: Short title of the gap.
        description: Detailed description of the gap.
        severity: Severity of the gap (critical, high, medium, low).
        status: Current status (open, in_progress, remediated, accepted).
        identified_at: When the gap was identified.
        remediation_plan: Planned remediation steps.
        remediation_owner: Who is responsible for remediation.
        target_date: Target date for remediation.
        remediated_at: When the gap was remediated.
    """

    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
    )

    id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique gap identifier.",
    )
    framework: ComplianceFramework = Field(
        ...,
        description="The compliance framework.",
    )
    control_id: str = Field(
        ...,
        min_length=1,
        max_length=50,
        description="The control with the gap.",
    )
    title: str = Field(
        ...,
        min_length=1,
        max_length=256,
        description="Short title of the gap.",
    )
    description: str = Field(
        default="",
        max_length=4096,
        description="Detailed description of the gap.",
    )
    severity: str = Field(
        default="medium",
        description="Severity: critical, high, medium, low.",
    )
    status: str = Field(
        default="open",
        description="Status: open, in_progress, remediated, accepted, rejected.",
    )
    identified_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="When the gap was identified (UTC).",
    )
    remediation_plan: str = Field(
        default="",
        max_length=4096,
        description="Planned remediation steps.",
    )
    remediation_owner: Optional[str] = Field(
        default=None,
        max_length=256,
        description="Who is responsible for remediation.",
    )
    target_date: Optional[datetime] = Field(
        default=None,
        description="Target date for remediation (UTC).",
    )
    remediated_at: Optional[datetime] = Field(
        default=None,
        description="When the gap was remediated (UTC).",
    )

    @field_validator("severity")
    @classmethod
    def validate_severity(cls, v: str) -> str:
        """Validate severity is one of the allowed values."""
        allowed = {"critical", "high", "medium", "low"}
        v_lower = v.strip().lower()
        if v_lower not in allowed:
            raise ValueError(
                f"Invalid severity '{v}'. Allowed values: {sorted(allowed)}"
            )
        return v_lower

    @field_validator("status")
    @classmethod
    def validate_status(cls, v: str) -> str:
        """Validate status is one of the allowed values."""
        allowed = {"open", "in_progress", "remediated", "accepted", "rejected"}
        v_lower = v.strip().lower()
        if v_lower not in allowed:
            raise ValueError(
                f"Invalid status '{v}'. Allowed values: {sorted(allowed)}"
            )
        return v_lower


__all__ = [
    # Enums
    "ComplianceFramework",
    "ControlStatus",
    "DSARType",
    "DSARStatus",
    "ConsentPurpose",
    "DataCategory",
    # Models
    "ComplianceStatus",
    "ControlMapping",
    "DSARRequest",
    "ConsentRecord",
    "DataRecord",
    "RetentionPolicy",
    "EvidenceSource",
    "ComplianceReport",
    "ComplianceGap",
]
