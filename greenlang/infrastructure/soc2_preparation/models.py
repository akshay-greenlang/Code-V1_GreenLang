# -*- coding: utf-8 -*-
"""
SOC 2 Type II Preparation Data Models - SEC-009

Pydantic v2 models and SQLAlchemy-style schemas for the GreenLang SOC 2 Type II
audit preparation platform. Provides strongly-typed data structures for
assessments, evidence, control tests, auditor requests, findings, remediations,
attestations, and audit project management.

All datetime fields use UTC. All models enforce strict validation via Pydantic v2
field validators and model configuration.

Enums:
    - ScoreLevel: Assessment maturity scores (0-4)
    - FindingClassification: SOC 2 finding severity levels
    - RequestPriority: Auditor request priority levels
    - TestType: Control test types (design/operating effectiveness)
    - EvidenceType: Types of evidence artifacts
    - AssessmentStatus: Assessment lifecycle status
    - ControlStatus: Control implementation status
    - RemediationStatus: Remediation progress status
    - MilestoneStatus: Audit milestone status

Models:
    - Assessment: Self-assessment run
    - AssessmentCriteria: Individual criterion within an assessment
    - Evidence: Evidence artifact metadata
    - EvidencePackage: Collection of evidence for an auditor request
    - ControlTest: Test of control design or operating effectiveness
    - TestResult: Result of a control test
    - AuditorRequest: Request from external auditor (PBC list item)
    - Finding: Audit finding or observation
    - Remediation: Remediation plan for a finding
    - Attestation: Management attestation document
    - AttestationSignature: Electronic signature on attestation
    - AuditProject: SOC 2 audit engagement
    - AuditMilestone: Key audit milestone

Author: GreenLang Security Team
Date: February 2026
PRD: SEC-009 SOC 2 Type II Preparation
"""

from __future__ import annotations

import hashlib
import re
import uuid
from datetime import date, datetime, timezone
from decimal import Decimal
from enum import Enum, IntEnum
from typing import Any, Dict, List, Optional

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
    computed_field,
)


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------


class ScoreLevel(IntEnum):
    """Assessment maturity score levels (0-4 scale).

    Based on CMMI-like maturity model for control implementation.
    Used to measure readiness for SOC 2 Type II audit.
    """

    NOT_IMPLEMENTED = 0
    """Control does not exist or is completely absent."""

    PARTIAL = 1
    """Control exists but is incomplete, informal, or inconsistent."""

    IMPLEMENTED = 2
    """Control is implemented and documented but not formally tested."""

    TESTED = 3
    """Control is implemented, documented, and tested but not yet audit-ready."""

    COMPLIANT = 4
    """Control is fully implemented, tested, documented, and audit-ready."""


class FindingClassification(str, Enum):
    """SOC 2 audit finding classification levels.

    Severity levels per AICPA SOC 2 guidance for reporting findings.
    """

    OBSERVATION = "observation"
    """Minor issue that does not affect control effectiveness."""

    EXCEPTION = "exception"
    """Isolated deviation from control design or operating procedure."""

    DEFICIENCY = "deficiency"
    """Control weakness that could result in material misstatement."""

    SIGNIFICANT_DEFICIENCY = "significant_deficiency"
    """Deficiency important enough to merit attention by governance."""

    MATERIAL_WEAKNESS = "material_weakness"
    """Deficiency that could result in material deviation from criteria."""


class RequestPriority(str, Enum):
    """Priority levels for auditor requests (PBC list items)."""

    CRITICAL = "critical"
    """Must be fulfilled within 4 hours - audit blocker."""

    HIGH = "high"
    """Must be fulfilled within 24 hours - impacts audit timeline."""

    NORMAL = "normal"
    """Standard priority - 48 hour SLA."""

    LOW = "low"
    """Best effort - 72 hour SLA."""


class TestType(str, Enum):
    """Types of control tests for SOC 2 audits."""

    DESIGN = "design"
    """Test of control design - does the control address the risk?"""

    OPERATING = "operating"
    """Test of operating effectiveness - is the control working as designed?"""

    BOTH = "both"
    """Combined test of both design and operating effectiveness."""


class EvidenceType(str, Enum):
    """Types of evidence artifacts for SOC 2 audits."""

    DOCUMENT = "document"
    """Policy, procedure, or other documentation."""

    SCREENSHOT = "screenshot"
    """Screenshot of system configuration or output."""

    LOG = "log"
    """System log or audit trail extract."""

    REPORT = "report"
    """Generated report or analysis output."""

    CONFIGURATION = "configuration"
    """System configuration file or settings export."""

    INQUIRY = "inquiry"
    """Interview notes or inquiry response."""

    OBSERVATION = "observation"
    """Direct observation of process or control."""

    REPERFORMANCE = "reperformance"
    """Results of auditor reperformance of control."""

    SAMPLE = "sample"
    """Sample selection for testing."""

    TICKET = "ticket"
    """Support ticket, change request, or incident record."""

    METRIC = "metric"
    """Quantitative metric or KPI data."""


class AssessmentStatus(str, Enum):
    """Lifecycle status of a self-assessment run."""

    DRAFT = "draft"
    """Assessment is in progress, not yet complete."""

    IN_PROGRESS = "in_progress"
    """Assessment is actively being conducted."""

    COMPLETED = "completed"
    """Assessment is complete and ready for review."""

    REVIEWED = "reviewed"
    """Assessment has been reviewed by compliance officer."""

    APPROVED = "approved"
    """Assessment has been approved for use."""

    ARCHIVED = "archived"
    """Assessment is archived (superseded by newer assessment)."""


class ControlStatus(str, Enum):
    """Implementation status of a control."""

    NOT_STARTED = "not_started"
    """Control implementation has not started."""

    IN_DESIGN = "in_design"
    """Control is being designed."""

    IN_IMPLEMENTATION = "in_implementation"
    """Control is being implemented."""

    IMPLEMENTED = "implemented"
    """Control is implemented but not tested."""

    IN_TESTING = "in_testing"
    """Control is being tested."""

    OPERATING = "operating"
    """Control is operating and monitored."""

    NOT_APPLICABLE = "not_applicable"
    """Control is not applicable to this environment."""


class RemediationStatus(str, Enum):
    """Progress status of a remediation plan."""

    OPEN = "open"
    """Remediation not yet started."""

    IN_PROGRESS = "in_progress"
    """Remediation work is underway."""

    PENDING_VALIDATION = "pending_validation"
    """Remediation complete, awaiting validation."""

    VALIDATED = "validated"
    """Remediation validated as effective."""

    CLOSED = "closed"
    """Remediation complete and closed."""

    DEFERRED = "deferred"
    """Remediation deferred to future period."""

    RISK_ACCEPTED = "risk_accepted"
    """Risk accepted without remediation."""


class MilestoneStatus(str, Enum):
    """Status of an audit milestone."""

    PENDING = "pending"
    """Milestone not yet started."""

    IN_PROGRESS = "in_progress"
    """Milestone work is underway."""

    COMPLETED = "completed"
    """Milestone completed successfully."""

    DELAYED = "delayed"
    """Milestone is delayed."""

    AT_RISK = "at_risk"
    """Milestone is at risk of delay."""


class TrustServiceCategory(str, Enum):
    """SOC 2 Trust Service Categories."""

    SECURITY = "security"
    """Security (Common Criteria) - required for all SOC 2 reports."""

    AVAILABILITY = "availability"
    """Availability - optional category."""

    CONFIDENTIALITY = "confidentiality"
    """Confidentiality - optional category."""

    PROCESSING_INTEGRITY = "processing_integrity"
    """Processing Integrity - optional category."""

    PRIVACY = "privacy"
    """Privacy - optional category."""


# ---------------------------------------------------------------------------
# Base Model Configuration
# ---------------------------------------------------------------------------


class SOC2BaseModel(BaseModel):
    """Base model for all SOC 2 data models.

    Provides common configuration and validation patterns.
    """

    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
        validate_default=True,
        use_enum_values=True,
        json_schema_extra={"examples": []},
    )


# ---------------------------------------------------------------------------
# Assessment Models
# ---------------------------------------------------------------------------


class AssessmentCriteria(SOC2BaseModel):
    """Individual criterion assessment within a self-assessment.

    Represents the evaluation of a single SOC 2 Trust Service Criterion
    (e.g., CC1.1, CC6.5, A1.2) within a larger assessment.

    Attributes:
        criterion_id: SOC 2 criterion identifier (e.g., "CC6.1").
        assessment_id: Parent assessment UUID.
        score: Maturity score (0-4).
        control_status: Implementation status of the control.
        evidence_count: Number of evidence items collected.
        evidence_ids: List of evidence UUIDs linked to this criterion.
        gaps_identified: Description of identified gaps.
        recommendations: Recommendations for improvement.
        notes: Assessor notes.
        assessed_by: UUID of user who assessed this criterion.
        assessed_at: Timestamp of assessment.
    """

    criterion_id: str = Field(
        ...,
        min_length=2,
        max_length=20,
        description="SOC 2 criterion identifier (e.g., CC6.1, A1.2).",
        examples=["CC6.1", "CC1.1", "A1.2"],
    )
    assessment_id: uuid.UUID = Field(
        ...,
        description="Parent assessment UUID.",
    )
    score: ScoreLevel = Field(
        default=ScoreLevel.NOT_IMPLEMENTED,
        description="Maturity score (0-4).",
    )
    control_status: ControlStatus = Field(
        default=ControlStatus.NOT_STARTED,
        description="Implementation status of the control.",
    )
    evidence_count: int = Field(
        default=0,
        ge=0,
        description="Number of evidence items collected.",
    )
    evidence_ids: List[uuid.UUID] = Field(
        default_factory=list,
        description="List of evidence UUIDs linked to this criterion.",
    )
    gaps_identified: str = Field(
        default="",
        max_length=4000,
        description="Description of identified gaps.",
    )
    recommendations: str = Field(
        default="",
        max_length=4000,
        description="Recommendations for improvement.",
    )
    notes: str = Field(
        default="",
        max_length=4000,
        description="Assessor notes.",
    )
    assessed_by: Optional[uuid.UUID] = Field(
        default=None,
        description="UUID of user who assessed this criterion.",
    )
    assessed_at: Optional[datetime] = Field(
        default=None,
        description="Timestamp of assessment (UTC).",
    )

    @field_validator("criterion_id")
    @classmethod
    def validate_criterion_id(cls, v: str) -> str:
        """Validate criterion ID format."""
        pattern = r"^(CC[1-9](\.[1-9][0-9]?)?|A1\.[1-3]|C1\.[1-2]|PI1\.[1-2]|P[1-8]\.[0-9])$"
        v_upper = v.strip().upper()
        if not re.match(pattern, v_upper):
            raise ValueError(
                f"Invalid criterion ID '{v}'. Expected format like CC6.1, A1.2, P1.0"
            )
        return v_upper

    @field_validator("assessed_at")
    @classmethod
    def ensure_utc(cls, v: Optional[datetime]) -> Optional[datetime]:
        """Ensure datetime is timezone-aware UTC."""
        if v is None:
            return v
        if v.tzinfo is None:
            return v.replace(tzinfo=timezone.utc)
        return v.astimezone(timezone.utc)


class Assessment(SOC2BaseModel):
    """Self-assessment run for SOC 2 readiness.

    Represents a point-in-time assessment of the organization's readiness
    for a SOC 2 Type II audit across all applicable Trust Service Criteria.

    Attributes:
        id: Unique assessment identifier.
        name: Human-readable assessment name.
        description: Detailed description of assessment scope.
        status: Lifecycle status.
        tsc_categories: Trust Service Categories in scope.
        overall_score: Weighted overall maturity score (0-100).
        criteria_assessed: Number of criteria assessed.
        criteria_compliant: Number of criteria at COMPLIANT level.
        gaps_count: Number of gaps identified.
        evidence_count: Total evidence items collected.
        assessed_by: UUID of user who initiated assessment.
        reviewed_by: UUID of user who reviewed assessment.
        approved_by: UUID of user who approved assessment.
        created_at: Creation timestamp.
        updated_at: Last update timestamp.
        completed_at: Completion timestamp.
    """

    id: uuid.UUID = Field(
        default_factory=uuid.uuid4,
        description="Unique assessment identifier.",
    )
    name: str = Field(
        ...,
        min_length=1,
        max_length=256,
        description="Human-readable assessment name.",
        examples=["Q1 2026 SOC 2 Readiness Assessment"],
    )
    description: str = Field(
        default="",
        max_length=2000,
        description="Detailed description of assessment scope.",
    )
    status: AssessmentStatus = Field(
        default=AssessmentStatus.DRAFT,
        description="Lifecycle status.",
    )
    tsc_categories: List[TrustServiceCategory] = Field(
        default_factory=lambda: [TrustServiceCategory.SECURITY],
        description="Trust Service Categories in scope.",
    )
    overall_score: Decimal = Field(
        default=Decimal("0.0"),
        ge=Decimal("0"),
        le=Decimal("100"),
        description="Weighted overall maturity score (0-100).",
    )
    criteria_assessed: int = Field(
        default=0,
        ge=0,
        description="Number of criteria assessed.",
    )
    criteria_compliant: int = Field(
        default=0,
        ge=0,
        description="Number of criteria at COMPLIANT level.",
    )
    gaps_count: int = Field(
        default=0,
        ge=0,
        description="Number of gaps identified.",
    )
    evidence_count: int = Field(
        default=0,
        ge=0,
        description="Total evidence items collected.",
    )
    assessed_by: Optional[uuid.UUID] = Field(
        default=None,
        description="UUID of user who initiated assessment.",
    )
    reviewed_by: Optional[uuid.UUID] = Field(
        default=None,
        description="UUID of user who reviewed assessment.",
    )
    approved_by: Optional[uuid.UUID] = Field(
        default=None,
        description="UUID of user who approved assessment.",
    )
    criteria: List[AssessmentCriteria] = Field(
        default_factory=list,
        description="List of individual criterion assessments.",
    )
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Creation timestamp (UTC).",
    )
    updated_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Last update timestamp (UTC).",
    )
    completed_at: Optional[datetime] = Field(
        default=None,
        description="Completion timestamp (UTC).",
    )

    @computed_field
    @property
    def readiness_percentage(self) -> float:
        """Calculate readiness percentage based on compliant criteria."""
        if self.criteria_assessed == 0:
            return 0.0
        return round((self.criteria_compliant / self.criteria_assessed) * 100, 2)

    @computed_field
    @property
    def provenance_hash(self) -> str:
        """Calculate SHA-256 hash for audit trail."""
        data = f"{self.id}{self.name}{self.overall_score}{self.updated_at.isoformat()}"
        return hashlib.sha256(data.encode()).hexdigest()


# ---------------------------------------------------------------------------
# Evidence Models
# ---------------------------------------------------------------------------


class Evidence(SOC2BaseModel):
    """Evidence artifact metadata for SOC 2 audit.

    Represents a single piece of evidence that supports control effectiveness.
    The actual file is stored in S3; this model holds metadata.

    Attributes:
        id: Unique evidence identifier.
        title: Human-readable title.
        description: Detailed description of evidence.
        evidence_type: Type of evidence artifact.
        criterion_ids: SOC 2 criteria this evidence supports.
        file_path: S3 path to evidence file.
        file_name: Original filename.
        file_size_bytes: File size in bytes.
        file_hash: SHA-256 hash of file contents.
        mime_type: MIME type of file.
        collected_at: When evidence was collected.
        collection_method: How evidence was collected (manual/automated).
        retention_until: Retention end date.
        uploaded_by: UUID of user who uploaded evidence.
        verified_by: UUID of user who verified evidence.
        is_automated: Whether evidence was collected automatically.
        metadata: Additional structured metadata.
    """

    id: uuid.UUID = Field(
        default_factory=uuid.uuid4,
        description="Unique evidence identifier.",
    )
    title: str = Field(
        ...,
        min_length=1,
        max_length=256,
        description="Human-readable title.",
        examples=["Access Control Policy v2.1"],
    )
    description: str = Field(
        default="",
        max_length=2000,
        description="Detailed description of evidence.",
    )
    evidence_type: EvidenceType = Field(
        ...,
        description="Type of evidence artifact.",
    )
    criterion_ids: List[str] = Field(
        default_factory=list,
        description="SOC 2 criteria this evidence supports.",
    )
    file_path: str = Field(
        ...,
        min_length=1,
        max_length=1024,
        description="S3 path to evidence file.",
    )
    file_name: str = Field(
        ...,
        min_length=1,
        max_length=256,
        description="Original filename.",
    )
    file_size_bytes: int = Field(
        ...,
        ge=0,
        description="File size in bytes.",
    )
    file_hash: str = Field(
        ...,
        min_length=64,
        max_length=64,
        description="SHA-256 hash of file contents.",
    )
    mime_type: str = Field(
        default="application/octet-stream",
        max_length=128,
        description="MIME type of file.",
    )
    collected_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="When evidence was collected (UTC).",
    )
    collection_method: str = Field(
        default="manual",
        description="How evidence was collected (manual/automated).",
    )
    retention_until: Optional[date] = Field(
        default=None,
        description="Retention end date.",
    )
    uploaded_by: Optional[uuid.UUID] = Field(
        default=None,
        description="UUID of user who uploaded evidence.",
    )
    verified_by: Optional[uuid.UUID] = Field(
        default=None,
        description="UUID of user who verified evidence.",
    )
    is_automated: bool = Field(
        default=False,
        description="Whether evidence was collected automatically.",
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional structured metadata.",
    )
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Creation timestamp (UTC).",
    )

    @field_validator("file_hash")
    @classmethod
    def validate_hash(cls, v: str) -> str:
        """Validate SHA-256 hash format."""
        v_lower = v.strip().lower()
        if not re.match(r"^[a-f0-9]{64}$", v_lower):
            raise ValueError(f"Invalid SHA-256 hash format: {v}")
        return v_lower


class EvidencePackage(SOC2BaseModel):
    """Collection of evidence for an auditor request.

    Groups multiple evidence items together for delivery to auditors.

    Attributes:
        id: Unique package identifier.
        name: Package name.
        description: Package description.
        auditor_request_id: Linked auditor request UUID.
        evidence_ids: List of evidence UUIDs in this package.
        evidence_items: Nested evidence items (optional).
        package_hash: SHA-256 hash of entire package.
        status: Package preparation status.
        prepared_by: UUID of user who prepared package.
        delivered_at: When package was delivered to auditor.
    """

    id: uuid.UUID = Field(
        default_factory=uuid.uuid4,
        description="Unique package identifier.",
    )
    name: str = Field(
        ...,
        min_length=1,
        max_length=256,
        description="Package name.",
    )
    description: str = Field(
        default="",
        max_length=2000,
        description="Package description.",
    )
    auditor_request_id: Optional[uuid.UUID] = Field(
        default=None,
        description="Linked auditor request UUID.",
    )
    evidence_ids: List[uuid.UUID] = Field(
        default_factory=list,
        description="List of evidence UUIDs in this package.",
    )
    evidence_items: List[Evidence] = Field(
        default_factory=list,
        description="Nested evidence items (optional).",
    )
    package_hash: Optional[str] = Field(
        default=None,
        min_length=64,
        max_length=64,
        description="SHA-256 hash of entire package.",
    )
    status: str = Field(
        default="draft",
        description="Package preparation status.",
    )
    prepared_by: Optional[uuid.UUID] = Field(
        default=None,
        description="UUID of user who prepared package.",
    )
    delivered_at: Optional[datetime] = Field(
        default=None,
        description="When package was delivered to auditor (UTC).",
    )
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Creation timestamp (UTC).",
    )


# ---------------------------------------------------------------------------
# Control Test Models
# ---------------------------------------------------------------------------


class TestResult(SOC2BaseModel):
    """Result of a control test.

    Detailed test result including sample selection, procedures performed,
    and conclusions.

    Attributes:
        id: Unique result identifier.
        control_test_id: Parent control test UUID.
        sample_size: Number of items sampled.
        samples_tested: Number of samples actually tested.
        exceptions_found: Number of exceptions found.
        exception_rate: Exception rate as percentage.
        exception_details: Description of exceptions found.
        conclusion: Test conclusion.
        tested_by: UUID of tester.
        tested_at: Test timestamp.
        evidence_ids: Evidence supporting test result.
    """

    id: uuid.UUID = Field(
        default_factory=uuid.uuid4,
        description="Unique result identifier.",
    )
    control_test_id: uuid.UUID = Field(
        ...,
        description="Parent control test UUID.",
    )
    sample_size: int = Field(
        default=0,
        ge=0,
        description="Number of items sampled.",
    )
    samples_tested: int = Field(
        default=0,
        ge=0,
        description="Number of samples actually tested.",
    )
    exceptions_found: int = Field(
        default=0,
        ge=0,
        description="Number of exceptions found.",
    )
    exception_rate: Decimal = Field(
        default=Decimal("0.0"),
        ge=Decimal("0"),
        le=Decimal("100"),
        description="Exception rate as percentage.",
    )
    exception_details: str = Field(
        default="",
        max_length=4000,
        description="Description of exceptions found.",
    )
    conclusion: str = Field(
        default="",
        max_length=2000,
        description="Test conclusion.",
    )
    tested_by: Optional[uuid.UUID] = Field(
        default=None,
        description="UUID of tester.",
    )
    tested_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Test timestamp (UTC).",
    )
    evidence_ids: List[uuid.UUID] = Field(
        default_factory=list,
        description="Evidence supporting test result.",
    )

    @model_validator(mode="after")
    def calculate_exception_rate(self) -> "TestResult":
        """Auto-calculate exception rate if not set."""
        if self.samples_tested > 0 and self.exception_rate == Decimal("0.0"):
            rate = (self.exceptions_found / self.samples_tested) * 100
            object.__setattr__(self, "exception_rate", Decimal(str(round(rate, 2))))
        return self


class ControlTest(SOC2BaseModel):
    """Test of control design or operating effectiveness.

    Represents a test procedure applied to evaluate a control for
    SOC 2 compliance.

    Attributes:
        id: Unique test identifier.
        criterion_id: SOC 2 criterion being tested.
        control_id: Control being tested.
        test_type: Type of test (design/operating).
        test_name: Human-readable test name.
        test_procedure: Detailed test procedure.
        expected_result: Expected outcome if control is effective.
        actual_result: Actual test result.
        status: Test status.
        is_effective: Whether control was found effective.
        period_start: Start of test period.
        period_end: End of test period.
        results: Test results.
    """

    id: uuid.UUID = Field(
        default_factory=uuid.uuid4,
        description="Unique test identifier.",
    )
    criterion_id: str = Field(
        ...,
        min_length=2,
        max_length=20,
        description="SOC 2 criterion being tested.",
    )
    control_id: str = Field(
        ...,
        min_length=1,
        max_length=50,
        description="Control identifier being tested.",
    )
    test_type: TestType = Field(
        ...,
        description="Type of test (design/operating).",
    )
    test_name: str = Field(
        ...,
        min_length=1,
        max_length=256,
        description="Human-readable test name.",
    )
    test_procedure: str = Field(
        ...,
        min_length=1,
        max_length=4000,
        description="Detailed test procedure.",
    )
    expected_result: str = Field(
        default="",
        max_length=2000,
        description="Expected outcome if control is effective.",
    )
    actual_result: str = Field(
        default="",
        max_length=2000,
        description="Actual test result.",
    )
    status: str = Field(
        default="pending",
        description="Test status (pending/in_progress/completed).",
    )
    is_effective: Optional[bool] = Field(
        default=None,
        description="Whether control was found effective.",
    )
    period_start: Optional[date] = Field(
        default=None,
        description="Start of test period.",
    )
    period_end: Optional[date] = Field(
        default=None,
        description="End of test period.",
    )
    results: List[TestResult] = Field(
        default_factory=list,
        description="Test results.",
    )
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Creation timestamp (UTC).",
    )
    updated_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Last update timestamp (UTC).",
    )


# ---------------------------------------------------------------------------
# Auditor Request Models
# ---------------------------------------------------------------------------


class AuditorRequest(SOC2BaseModel):
    """Request from external auditor (PBC list item).

    Represents an item on the Prepared by Client (PBC) list - a request
    from the external auditor for specific evidence or information.

    Attributes:
        id: Unique request identifier.
        request_number: Auditor's reference number.
        title: Request title.
        description: Detailed request description.
        priority: Request priority level.
        criterion_ids: Related SOC 2 criteria.
        requested_by: Auditor name or identifier.
        requested_at: When request was made.
        due_at: SLA deadline.
        assigned_to: UUID of assigned staff member.
        status: Request status.
        response: Response to the request.
        responded_at: When response was provided.
        evidence_package_id: Linked evidence package.
        audit_project_id: Parent audit project.
        is_overdue: Whether request is past SLA.
        notes: Internal notes.
    """

    id: uuid.UUID = Field(
        default_factory=uuid.uuid4,
        description="Unique request identifier.",
    )
    request_number: str = Field(
        ...,
        min_length=1,
        max_length=50,
        description="Auditor's reference number.",
        examples=["PBC-001", "RFI-2026-0042"],
    )
    title: str = Field(
        ...,
        min_length=1,
        max_length=256,
        description="Request title.",
    )
    description: str = Field(
        default="",
        max_length=4000,
        description="Detailed request description.",
    )
    priority: RequestPriority = Field(
        default=RequestPriority.NORMAL,
        description="Request priority level.",
    )
    criterion_ids: List[str] = Field(
        default_factory=list,
        description="Related SOC 2 criteria.",
    )
    requested_by: str = Field(
        default="",
        max_length=256,
        description="Auditor name or identifier.",
    )
    requested_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="When request was made (UTC).",
    )
    due_at: datetime = Field(
        ...,
        description="SLA deadline (UTC).",
    )
    assigned_to: Optional[uuid.UUID] = Field(
        default=None,
        description="UUID of assigned staff member.",
    )
    status: str = Field(
        default="open",
        description="Request status (open/in_progress/fulfilled/on_hold).",
    )
    response: str = Field(
        default="",
        max_length=4000,
        description="Response to the request.",
    )
    responded_at: Optional[datetime] = Field(
        default=None,
        description="When response was provided (UTC).",
    )
    evidence_package_id: Optional[uuid.UUID] = Field(
        default=None,
        description="Linked evidence package.",
    )
    audit_project_id: Optional[uuid.UUID] = Field(
        default=None,
        description="Parent audit project.",
    )
    notes: str = Field(
        default="",
        max_length=2000,
        description="Internal notes.",
    )
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Creation timestamp (UTC).",
    )
    updated_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Last update timestamp (UTC).",
    )

    @computed_field
    @property
    def is_overdue(self) -> bool:
        """Check if request is past SLA deadline."""
        if self.status in ("fulfilled", "cancelled"):
            return False
        return datetime.now(timezone.utc) > self.due_at

    @computed_field
    @property
    def hours_until_due(self) -> float:
        """Hours until SLA deadline (negative if overdue)."""
        delta = self.due_at - datetime.now(timezone.utc)
        return round(delta.total_seconds() / 3600, 2)


# ---------------------------------------------------------------------------
# Finding and Remediation Models
# ---------------------------------------------------------------------------


class Remediation(SOC2BaseModel):
    """Remediation plan for a finding.

    Tracks the remediation effort for an identified finding or gap.

    Attributes:
        id: Unique remediation identifier.
        finding_id: Parent finding UUID.
        plan: Remediation plan description.
        owner: UUID of remediation owner.
        status: Remediation status.
        target_date: Target completion date.
        actual_completion_date: Actual completion date.
        effort_hours: Estimated effort in hours.
        actual_hours: Actual hours spent.
        validation_method: How remediation will be validated.
        validated_by: UUID of validator.
        validated_at: Validation timestamp.
        notes: Progress notes.
    """

    id: uuid.UUID = Field(
        default_factory=uuid.uuid4,
        description="Unique remediation identifier.",
    )
    finding_id: uuid.UUID = Field(
        ...,
        description="Parent finding UUID.",
    )
    plan: str = Field(
        ...,
        min_length=1,
        max_length=4000,
        description="Remediation plan description.",
    )
    owner: Optional[uuid.UUID] = Field(
        default=None,
        description="UUID of remediation owner.",
    )
    status: RemediationStatus = Field(
        default=RemediationStatus.OPEN,
        description="Remediation status.",
    )
    target_date: Optional[date] = Field(
        default=None,
        description="Target completion date.",
    )
    actual_completion_date: Optional[date] = Field(
        default=None,
        description="Actual completion date.",
    )
    effort_hours: int = Field(
        default=0,
        ge=0,
        description="Estimated effort in hours.",
    )
    actual_hours: int = Field(
        default=0,
        ge=0,
        description="Actual hours spent.",
    )
    validation_method: str = Field(
        default="",
        max_length=1000,
        description="How remediation will be validated.",
    )
    validated_by: Optional[uuid.UUID] = Field(
        default=None,
        description="UUID of validator.",
    )
    validated_at: Optional[datetime] = Field(
        default=None,
        description="Validation timestamp (UTC).",
    )
    notes: str = Field(
        default="",
        max_length=4000,
        description="Progress notes.",
    )
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Creation timestamp (UTC).",
    )
    updated_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Last update timestamp (UTC).",
    )


class Finding(SOC2BaseModel):
    """Audit finding or observation.

    Represents an issue identified during the SOC 2 audit or self-assessment.

    Attributes:
        id: Unique finding identifier.
        finding_number: Human-readable finding reference.
        title: Finding title.
        description: Detailed finding description.
        classification: Finding severity classification.
        criterion_ids: Related SOC 2 criteria.
        control_id: Related control identifier.
        root_cause: Root cause analysis.
        management_response: Management's response.
        status: Finding status.
        identified_by: UUID of person who identified finding.
        identified_at: When finding was identified.
        audit_project_id: Parent audit project.
        remediation: Linked remediation plan.
        is_repeat: Whether this is a repeat finding.
        prior_finding_id: Reference to prior finding if repeat.
    """

    id: uuid.UUID = Field(
        default_factory=uuid.uuid4,
        description="Unique finding identifier.",
    )
    finding_number: str = Field(
        ...,
        min_length=1,
        max_length=50,
        description="Human-readable finding reference.",
        examples=["F-2026-001", "OBS-003"],
    )
    title: str = Field(
        ...,
        min_length=1,
        max_length=256,
        description="Finding title.",
    )
    description: str = Field(
        ...,
        min_length=1,
        max_length=4000,
        description="Detailed finding description.",
    )
    classification: FindingClassification = Field(
        ...,
        description="Finding severity classification.",
    )
    criterion_ids: List[str] = Field(
        default_factory=list,
        description="Related SOC 2 criteria.",
    )
    control_id: str = Field(
        default="",
        max_length=50,
        description="Related control identifier.",
    )
    root_cause: str = Field(
        default="",
        max_length=2000,
        description="Root cause analysis.",
    )
    management_response: str = Field(
        default="",
        max_length=2000,
        description="Management's response.",
    )
    status: str = Field(
        default="open",
        description="Finding status (open/remediated/closed/risk_accepted).",
    )
    identified_by: Optional[uuid.UUID] = Field(
        default=None,
        description="UUID of person who identified finding.",
    )
    identified_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="When finding was identified (UTC).",
    )
    audit_project_id: Optional[uuid.UUID] = Field(
        default=None,
        description="Parent audit project.",
    )
    remediation: Optional[Remediation] = Field(
        default=None,
        description="Linked remediation plan.",
    )
    is_repeat: bool = Field(
        default=False,
        description="Whether this is a repeat finding.",
    )
    prior_finding_id: Optional[uuid.UUID] = Field(
        default=None,
        description="Reference to prior finding if repeat.",
    )
    evidence_ids: List[uuid.UUID] = Field(
        default_factory=list,
        description="Evidence supporting the finding.",
    )
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Creation timestamp (UTC).",
    )
    updated_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Last update timestamp (UTC).",
    )


# ---------------------------------------------------------------------------
# Attestation Models
# ---------------------------------------------------------------------------


class AttestationSignature(SOC2BaseModel):
    """Electronic signature on an attestation.

    Represents a digital signature by management on an attestation document.

    Attributes:
        id: Unique signature identifier.
        attestation_id: Parent attestation UUID.
        signer_id: UUID of signer.
        signer_name: Signer's full name.
        signer_title: Signer's job title.
        signed_at: Signature timestamp.
        signature_hash: Hash of signed content.
        ip_address: IP address of signer.
        user_agent: Browser user agent of signer.
    """

    id: uuid.UUID = Field(
        default_factory=uuid.uuid4,
        description="Unique signature identifier.",
    )
    attestation_id: uuid.UUID = Field(
        ...,
        description="Parent attestation UUID.",
    )
    signer_id: uuid.UUID = Field(
        ...,
        description="UUID of signer.",
    )
    signer_name: str = Field(
        ...,
        min_length=1,
        max_length=256,
        description="Signer's full name.",
    )
    signer_title: str = Field(
        ...,
        min_length=1,
        max_length=256,
        description="Signer's job title.",
    )
    signed_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Signature timestamp (UTC).",
    )
    signature_hash: str = Field(
        ...,
        min_length=64,
        max_length=64,
        description="SHA-256 hash of signed content.",
    )
    ip_address: Optional[str] = Field(
        default=None,
        max_length=45,
        description="IP address of signer.",
    )
    user_agent: Optional[str] = Field(
        default=None,
        max_length=512,
        description="Browser user agent of signer.",
    )


class Attestation(SOC2BaseModel):
    """Management attestation document.

    Represents a formal attestation by management regarding controls
    and representations made to auditors.

    Attributes:
        id: Unique attestation identifier.
        title: Attestation title.
        content: Attestation content/text.
        attestation_type: Type of attestation.
        audit_project_id: Parent audit project.
        required_signers: UUIDs of required signers.
        signatures: Collected signatures.
        status: Attestation status.
        effective_date: Date attestation becomes effective.
        expiration_date: Date attestation expires.
        supersedes_id: UUID of attestation this supersedes.
    """

    id: uuid.UUID = Field(
        default_factory=uuid.uuid4,
        description="Unique attestation identifier.",
    )
    title: str = Field(
        ...,
        min_length=1,
        max_length=256,
        description="Attestation title.",
    )
    content: str = Field(
        ...,
        min_length=1,
        max_length=50000,
        description="Attestation content/text.",
    )
    attestation_type: str = Field(
        ...,
        description="Type of attestation (management_rep/sod/subservice/etc).",
    )
    audit_project_id: Optional[uuid.UUID] = Field(
        default=None,
        description="Parent audit project.",
    )
    required_signers: List[uuid.UUID] = Field(
        default_factory=list,
        description="UUIDs of required signers.",
    )
    signatures: List[AttestationSignature] = Field(
        default_factory=list,
        description="Collected signatures.",
    )
    status: str = Field(
        default="draft",
        description="Attestation status (draft/pending_signatures/signed/expired).",
    )
    effective_date: Optional[date] = Field(
        default=None,
        description="Date attestation becomes effective.",
    )
    expiration_date: Optional[date] = Field(
        default=None,
        description="Date attestation expires.",
    )
    supersedes_id: Optional[uuid.UUID] = Field(
        default=None,
        description="UUID of attestation this supersedes.",
    )
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Creation timestamp (UTC).",
    )
    updated_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Last update timestamp (UTC).",
    )

    @computed_field
    @property
    def is_fully_signed(self) -> bool:
        """Check if all required signers have signed."""
        if not self.required_signers:
            return False
        signed_ids = {sig.signer_id for sig in self.signatures}
        return all(req_id in signed_ids for req_id in self.required_signers)

    @computed_field
    @property
    def content_hash(self) -> str:
        """SHA-256 hash of attestation content."""
        return hashlib.sha256(self.content.encode()).hexdigest()


# ---------------------------------------------------------------------------
# Audit Project Models
# ---------------------------------------------------------------------------


class AuditMilestone(SOC2BaseModel):
    """Key audit milestone.

    Represents a significant milestone in the SOC 2 audit engagement.

    Attributes:
        id: Unique milestone identifier.
        audit_project_id: Parent audit project.
        name: Milestone name.
        description: Milestone description.
        status: Milestone status.
        target_date: Target completion date.
        actual_date: Actual completion date.
        owner: UUID of milestone owner.
        dependencies: UUIDs of dependent milestones.
        notes: Progress notes.
    """

    id: uuid.UUID = Field(
        default_factory=uuid.uuid4,
        description="Unique milestone identifier.",
    )
    audit_project_id: uuid.UUID = Field(
        ...,
        description="Parent audit project.",
    )
    name: str = Field(
        ...,
        min_length=1,
        max_length=256,
        description="Milestone name.",
    )
    description: str = Field(
        default="",
        max_length=1000,
        description="Milestone description.",
    )
    status: MilestoneStatus = Field(
        default=MilestoneStatus.PENDING,
        description="Milestone status.",
    )
    target_date: date = Field(
        ...,
        description="Target completion date.",
    )
    actual_date: Optional[date] = Field(
        default=None,
        description="Actual completion date.",
    )
    owner: Optional[uuid.UUID] = Field(
        default=None,
        description="UUID of milestone owner.",
    )
    dependencies: List[uuid.UUID] = Field(
        default_factory=list,
        description="UUIDs of dependent milestones.",
    )
    notes: str = Field(
        default="",
        max_length=2000,
        description="Progress notes.",
    )
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Creation timestamp (UTC).",
    )
    updated_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Last update timestamp (UTC).",
    )

    @computed_field
    @property
    def is_delayed(self) -> bool:
        """Check if milestone is past target date without completion."""
        if self.status == MilestoneStatus.COMPLETED:
            return False
        return date.today() > self.target_date


class AuditProject(SOC2BaseModel):
    """SOC 2 audit engagement.

    Represents an entire SOC 2 Type II audit engagement from planning
    through report issuance.

    Attributes:
        id: Unique project identifier.
        name: Project name.
        description: Project description.
        audit_firm: Name of external audit firm.
        lead_auditor: Lead auditor name.
        tsc_categories: Trust Service Categories in scope.
        period_start: Audit period start date.
        period_end: Audit period end date.
        status: Project status.
        milestones: Project milestones.
        findings: Audit findings.
        attestations: Management attestations.
        compliance_officer_id: UUID of internal compliance officer.
        report_type: SOC 2 report type (Type I or Type II).
    """

    id: uuid.UUID = Field(
        default_factory=uuid.uuid4,
        description="Unique project identifier.",
    )
    name: str = Field(
        ...,
        min_length=1,
        max_length=256,
        description="Project name.",
        examples=["SOC 2 Type II 2026"],
    )
    description: str = Field(
        default="",
        max_length=2000,
        description="Project description.",
    )
    audit_firm: str = Field(
        default="",
        max_length=256,
        description="Name of external audit firm.",
    )
    lead_auditor: str = Field(
        default="",
        max_length=256,
        description="Lead auditor name.",
    )
    tsc_categories: List[TrustServiceCategory] = Field(
        default_factory=lambda: [TrustServiceCategory.SECURITY],
        description="Trust Service Categories in scope.",
    )
    period_start: date = Field(
        ...,
        description="Audit period start date.",
    )
    period_end: date = Field(
        ...,
        description="Audit period end date.",
    )
    status: str = Field(
        default="planning",
        description="Project status (planning/fieldwork/reporting/complete).",
    )
    milestones: List[AuditMilestone] = Field(
        default_factory=list,
        description="Project milestones.",
    )
    findings: List[Finding] = Field(
        default_factory=list,
        description="Audit findings.",
    )
    attestations: List[Attestation] = Field(
        default_factory=list,
        description="Management attestations.",
    )
    compliance_officer_id: Optional[uuid.UUID] = Field(
        default=None,
        description="UUID of internal compliance officer.",
    )
    report_type: str = Field(
        default="type_ii",
        description="SOC 2 report type (type_i or type_ii).",
    )
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Creation timestamp (UTC).",
    )
    updated_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Last update timestamp (UTC).",
    )

    @model_validator(mode="after")
    def validate_period(self) -> "AuditProject":
        """Validate audit period dates."""
        if self.period_start >= self.period_end:
            raise ValueError(
                f"period_start ({self.period_start}) must be "
                f"before period_end ({self.period_end})."
            )
        return self

    @computed_field
    @property
    def period_days(self) -> int:
        """Number of days in audit period."""
        return (self.period_end - self.period_start).days

    @computed_field
    @property
    def completion_percentage(self) -> float:
        """Calculate project completion based on milestones."""
        if not self.milestones:
            return 0.0
        completed = sum(
            1 for m in self.milestones if m.status == MilestoneStatus.COMPLETED
        )
        return round((completed / len(self.milestones)) * 100, 2)


# ---------------------------------------------------------------------------
# Export All
# ---------------------------------------------------------------------------


__all__ = [
    # Enums
    "ScoreLevel",
    "FindingClassification",
    "RequestPriority",
    "TestType",
    "EvidenceType",
    "AssessmentStatus",
    "ControlStatus",
    "RemediationStatus",
    "MilestoneStatus",
    "TrustServiceCategory",
    # Base
    "SOC2BaseModel",
    # Assessment
    "Assessment",
    "AssessmentCriteria",
    # Evidence
    "Evidence",
    "EvidencePackage",
    # Control Test
    "ControlTest",
    "TestResult",
    # Auditor Request
    "AuditorRequest",
    # Finding
    "Finding",
    "Remediation",
    # Attestation
    "Attestation",
    "AttestationSignature",
    # Audit Project
    "AuditProject",
    "AuditMilestone",
]
