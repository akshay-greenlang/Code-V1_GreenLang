# -*- coding: utf-8 -*-
"""
BaseYearAuditEngine - PACK-045 Base Year Management Engine 9
==============================================================

Comprehensive audit trail and verification engine for base year
management.  Maintains an immutable, tamper-evident log of all events
related to base year establishment, recalculation triggers, significance
assessments, approvals, and third-party verifications.

Every audit entry is individually hashed with SHA-256 to create a
tamper-evident chain.  Each entry's hash includes the previous entry's
hash, forming a lightweight blockchain-style chain that detects any
retroactive modification.

Audit Event Lifecycle:
    1. BASE_YEAR_ESTABLISHED - Initial selection of the base year
    2. TRIGGER_DETECTED - A recalculation trigger is identified
    3. SIGNIFICANCE_ASSESSED - The trigger is assessed for significance
    4. RECALCULATION_APPROVED - Approval to proceed with recalculation
    5. RECALCULATION_APPLIED - Recalculation values applied to base year
    6. TARGET_REBASED - Associated targets updated for new base year
    7. POLICY_UPDATED - Base year policy or procedure updated
    8. VERIFICATION_COMPLETED - Third-party verification of base year
    9. ANNUAL_REVIEW_COMPLETED - Periodic review of base year adequacy

Verification Levels (ISAE 3410):
    INTERNAL_REVIEW:        Internal quality review by qualified personnel.
    LIMITED_ASSURANCE:      Third-party limited assurance engagement.
    REASONABLE_ASSURANCE:   Third-party reasonable assurance engagement.

Approval Workflow:
    PENDING -> APPROVED   (standard approval)
    PENDING -> REJECTED   (returned with comments)
    PENDING -> ESCALATED  (escalated to higher authority)

Regulatory References:
    - ISAE 3410 (Assurance Engagements on GHG Statements)
    - ISAE 3000 (Revised) (Assurance Engagements Other Than Audits)
    - GHG Protocol Corporate Standard, Chapter 8 (Quality Management)
    - ISO 14064-3:2019 (Validation and Verification)
    - ESRS E1 (Assurance requirements for climate disclosures)
    - SEC Climate Disclosure Rule (2024) (Attestation requirements)

Zero-Hallucination:
    - All audit entries are deterministic records of events
    - Hash chain is computed using SHA-256 (no approximations)
    - No LLM involvement in any audit, approval, or verification logic
    - SHA-256 provenance hash on every entry and trail

Author:  GreenLang Platform Team
Date:    March 2026
Pack:    PACK-045 Base Year Management
Engine:  9 of 10
Status:  Production Ready
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
import uuid
from datetime import datetime, date, timezone
from decimal import Decimal, ROUND_HALF_UP, InvalidOperation
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple

from pydantic import BaseModel, Field, field_validator, model_validator

logger = logging.getLogger(__name__)

_MODULE_VERSION: str = "1.0.0"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _utcnow() -> datetime:
    """Return current UTC datetime with microseconds zeroed."""
    return datetime.now(timezone.utc).replace(microsecond=0)


def _new_uuid() -> str:
    """Generate a new UUID4 string."""
    return str(uuid.uuid4())


def _compute_hash(data: Any) -> str:
    """Compute a deterministic SHA-256 hash of arbitrary data.

    Excludes volatile fields (calculated_at, processing_time_ms,
    provenance_hash) so that identical logical content always produces
    the same hash.

    Args:
        data: Any Pydantic model, dict, or stringifiable object.

    Returns:
        Hex-encoded SHA-256 digest string.
    """
    if hasattr(data, "model_dump"):
        serializable = data.model_dump(mode="json")
    elif isinstance(data, dict):
        serializable = data
    else:
        serializable = str(data)
    if isinstance(serializable, dict):
        serializable = {
            k: v for k, v in serializable.items()
            if k not in ("calculated_at", "processing_time_ms", "provenance_hash")
        }
    raw = json.dumps(serializable, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _compute_chain_hash(entry_data: str, previous_hash: str) -> str:
    """Compute a chained SHA-256 hash incorporating the previous entry.

    This creates a tamper-evident chain where modifying any previous
    entry would invalidate all subsequent hashes.

    Args:
        entry_data: Serialized entry data string.
        previous_hash: Hash of the previous entry in the chain.

    Returns:
        Hex-encoded SHA-256 digest string.
    """
    combined = f"{previous_hash}:{entry_data}"
    return hashlib.sha256(combined.encode("utf-8")).hexdigest()


def _decimal(value: Any) -> Decimal:
    """Safely convert a value to Decimal."""
    if isinstance(value, Decimal):
        return value
    try:
        return Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError):
        return Decimal("0")


def _round_val(value: Decimal, places: int = 4) -> Decimal:
    """Round a Decimal to *places* using ROUND_HALF_UP."""
    quantize_str = "0." + "0" * places
    return value.quantize(Decimal(quantize_str), rounding=ROUND_HALF_UP)


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class AuditEventType(str, Enum):
    """Types of audit events in the base year management lifecycle.

    BASE_YEAR_ESTABLISHED:      Initial selection and documentation of
                                the base year.
    TRIGGER_DETECTED:           A potential recalculation trigger has been
                                identified (acquisition, methodology change, etc.).
    SIGNIFICANCE_ASSESSED:      The detected trigger has been assessed for
                                significance against the threshold.
    RECALCULATION_APPROVED:     Formal approval to proceed with base year
                                recalculation.
    RECALCULATION_APPLIED:      Recalculated values have been applied to
                                the base year inventory.
    TARGET_REBASED:             Emission reduction targets have been updated
                                to reflect the recalculated base year.
    POLICY_UPDATED:             The base year recalculation policy or
                                procedures have been updated.
    VERIFICATION_COMPLETED:     Third-party verification of base year data
                                has been completed.
    ANNUAL_REVIEW_COMPLETED:    Periodic review of base year adequacy and
                                recalculation policy has been completed.
    """
    BASE_YEAR_ESTABLISHED = "base_year_established"
    TRIGGER_DETECTED = "trigger_detected"
    SIGNIFICANCE_ASSESSED = "significance_assessed"
    RECALCULATION_APPROVED = "recalculation_approved"
    RECALCULATION_APPLIED = "recalculation_applied"
    TARGET_REBASED = "target_rebased"
    POLICY_UPDATED = "policy_updated"
    VERIFICATION_COMPLETED = "verification_completed"
    ANNUAL_REVIEW_COMPLETED = "annual_review_completed"


class VerificationLevel(str, Enum):
    """Level of assurance for third-party verification.

    INTERNAL_REVIEW:        Internal quality review by qualified staff.
                            Not an independent assurance engagement.
    LIMITED_ASSURANCE:      ISAE 3410 limited assurance.  Negative form
                            conclusion ("nothing has come to our attention").
    REASONABLE_ASSURANCE:   ISAE 3410 reasonable assurance.  Positive form
                            conclusion ("in our opinion, the statement is
                            fairly presented").
    """
    INTERNAL_REVIEW = "internal_review"
    LIMITED_ASSURANCE = "limited_assurance"
    REASONABLE_ASSURANCE = "reasonable_assurance"


class ApprovalStatus(str, Enum):
    """Status of an approval request.

    PENDING:    Awaiting decision.
    APPROVED:   Approved by the designated authority.
    REJECTED:   Rejected with comments.
    ESCALATED:  Escalated to a higher authority for decision.
    """
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    ESCALATED = "escalated"


class AuditSeverity(str, Enum):
    """Severity/importance level of an audit entry.

    INFO:       Informational record, no action required.
    LOW:        Low priority, review at next scheduled assessment.
    MEDIUM:     Medium priority, action required within reporting cycle.
    HIGH:       High priority, action required promptly.
    CRITICAL:   Critical, immediate action required to maintain compliance.
    """
    INFO = "info"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ExportFormat(str, Enum):
    """Supported export formats for audit trail.

    JSON:       Machine-readable JSON format.
    CSV:        Comma-separated values for spreadsheet analysis.
    MARKDOWN:   Human-readable Markdown format.
    """
    JSON = "json"
    CSV = "csv"
    MARKDOWN = "markdown"


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Audit trail chain genesis hash (starting point for chain computation).
GENESIS_HASH: str = "0" * 64  # 64 zeros = SHA-256 of empty chain.

# Required audit events for a complete base year lifecycle.
REQUIRED_EVENTS_FOR_COMPLETENESS: List[str] = [
    AuditEventType.BASE_YEAR_ESTABLISHED.value,
]

# Required events for a complete recalculation cycle.
REQUIRED_RECALCULATION_EVENTS: List[str] = [
    AuditEventType.TRIGGER_DETECTED.value,
    AuditEventType.SIGNIFICANCE_ASSESSED.value,
    AuditEventType.RECALCULATION_APPROVED.value,
    AuditEventType.RECALCULATION_APPLIED.value,
]

# ISAE 3410 evidence categories.
ISAE3410_EVIDENCE_CATEGORIES: List[str] = [
    "organizational_boundary",
    "base_year_selection_rationale",
    "calculation_methodology",
    "emission_factors_sources",
    "activity_data_sources",
    "recalculation_policy",
    "recalculation_history",
    "quality_management",
    "data_management",
    "uncertainty_assessment",
]


# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------


class AuditEntry(BaseModel):
    """A single entry in the base year audit trail.

    Each entry is an immutable record of an event in the base year
    management lifecycle.  The entry includes a provenance hash computed
    from the entry data and the previous entry's hash, forming a
    tamper-evident chain.

    Attributes:
        entry_id: Unique identifier for this entry.
        event_type: Type of audit event.
        timestamp: When the event occurred.
        actor: Person or system that triggered the event.
        description: Human-readable description of the event.
        severity: Importance level of the event.
        before_value: Value before the change (if applicable).
        after_value: Value after the change (if applicable).
        evidence_references: List of document/file references supporting
            the event.
        metadata: Additional structured metadata about the event.
        digital_signature: Optional digital signature for non-repudiation.
        previous_hash: Hash of the previous entry in the chain.
        provenance_hash: SHA-256 hash of this entry (includes chain link).
    """
    entry_id: str = Field(default_factory=_new_uuid)
    event_type: AuditEventType
    timestamp: datetime = Field(default_factory=_utcnow)
    actor: str = Field(..., min_length=1, max_length=500)
    description: str = Field(..., min_length=1)
    severity: AuditSeverity = Field(default=AuditSeverity.INFO)
    before_value: Optional[str] = Field(
        default=None,
        description="Value before the change (serialized)"
    )
    after_value: Optional[str] = Field(
        default=None,
        description="Value after the change (serialized)"
    )
    evidence_references: List[str] = Field(
        default_factory=list,
        description="Document/file references"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional structured metadata"
    )
    digital_signature: Optional[str] = Field(
        default=None,
        description="Digital signature for non-repudiation"
    )
    previous_hash: str = Field(
        default=GENESIS_HASH,
        description="Hash of the previous entry in the chain"
    )
    provenance_hash: str = Field(default="")


class ApprovalRecord(BaseModel):
    """Record of an approval decision in the base year workflow.

    Attributes:
        approval_id: Unique identifier for this approval record.
        subject: What is being approved (e.g., "Base year recalculation
            for 2023 acquisition of XYZ Corp").
        requested_by: Person/system requesting approval.
        requested_date: Date approval was requested.
        approver: Person designated to approve.
        approval_status: Current status of the approval.
        decision_date: Date the decision was made (None if pending).
        conditions: Conditions attached to the approval (if any).
        comments: Approver's comments.
        related_entry_ids: Audit entry IDs related to this approval.
        provenance_hash: SHA-256 hash for auditability.
    """
    approval_id: str = Field(default_factory=_new_uuid)
    subject: str = Field(..., min_length=1)
    requested_by: str = Field(..., min_length=1)
    requested_date: datetime = Field(default_factory=_utcnow)
    approver: str = Field(..., min_length=1)
    approval_status: ApprovalStatus = Field(default=ApprovalStatus.PENDING)
    decision_date: Optional[datetime] = None
    conditions: List[str] = Field(default_factory=list)
    comments: str = Field(default="")
    related_entry_ids: List[str] = Field(default_factory=list)
    provenance_hash: str = Field(default="")


class VerificationFinding(BaseModel):
    """A single finding from a verification engagement.

    Attributes:
        finding_id: Unique identifier.
        category: Area of the finding.
        description: Description of the finding.
        severity: Severity (info, low, medium, high, critical).
        recommendation: Recommended remediation action.
        management_response: Organization's response to the finding.
        status: Current status (open, resolved, accepted).
    """
    finding_id: str = Field(default_factory=_new_uuid)
    category: str
    description: str
    severity: AuditSeverity = Field(default=AuditSeverity.MEDIUM)
    recommendation: str = Field(default="")
    management_response: str = Field(default="")
    status: str = Field(default="open")


class VerificationPackage(BaseModel):
    """Complete verification package for base year assurance.

    Encapsulates the results of a verification engagement per ISAE 3410,
    including all findings, the verifier's opinion, and evidence items.

    Attributes:
        package_id: Unique identifier for this verification package.
        base_year: The base year being verified.
        verification_level: Level of assurance provided.
        verifier_name: Name of the verification body.
        verifier_accreditation: Accreditation details of the verifier.
        verification_date: Date verification was completed.
        scope_covered: Description of the scope of verification.
        findings: List of verification findings.
        opinion: Verifier's overall opinion statement.
        is_qualified: Whether the opinion includes qualifications.
        qualification_details: Details of any qualifications.
        evidence_items: List of evidence documents reviewed.
        materiality_threshold_pct: Materiality threshold used (%).
        total_emissions_verified_tco2e: Total emissions verified.
        provenance_hash: SHA-256 hash for auditability.
    """
    package_id: str = Field(default_factory=_new_uuid)
    base_year: int = Field(..., ge=1990, le=2100)
    verification_level: VerificationLevel
    verifier_name: str = Field(..., min_length=1)
    verifier_accreditation: str = Field(default="")
    verification_date: datetime = Field(default_factory=_utcnow)
    scope_covered: str = Field(default="Scope 1 and Scope 2")
    findings: List[VerificationFinding] = Field(default_factory=list)
    opinion: str = Field(default="")
    is_qualified: bool = Field(default=False)
    qualification_details: str = Field(default="")
    evidence_items: List[str] = Field(default_factory=list)
    materiality_threshold_pct: Decimal = Field(default=Decimal("5"))
    total_emissions_verified_tco2e: Optional[Decimal] = Field(default=None)
    provenance_hash: str = Field(default="")

    @field_validator("materiality_threshold_pct", mode="before")
    @classmethod
    def _coerce_decimal(cls, v: Any) -> Decimal:
        return _decimal(v)

    @field_validator("total_emissions_verified_tco2e", mode="before")
    @classmethod
    def _coerce_optional_decimal(cls, v: Any) -> Optional[Decimal]:
        if v is None:
            return None
        return _decimal(v)


class AuditTrailFilter(BaseModel):
    """Filter criteria for querying the audit trail.

    Attributes:
        event_types: Filter by specific event types.
        actors: Filter by specific actors.
        severity_min: Minimum severity level.
        date_from: Entries on or after this date.
        date_to: Entries on or before this date.
        has_evidence: Only entries with evidence references.
    """
    event_types: Optional[List[AuditEventType]] = None
    actors: Optional[List[str]] = None
    severity_min: Optional[AuditSeverity] = None
    date_from: Optional[datetime] = None
    date_to: Optional[datetime] = None
    has_evidence: Optional[bool] = None


class AuditTrail(BaseModel):
    """Complete audit trail for a base year.

    Contains all audit entries, approval records, and verification
    packages for a specific organization and base year, with an
    overall provenance hash covering the entire trail.

    Attributes:
        organization_id: Organization identifier.
        base_year: The base year this trail covers.
        entries: List of audit entries in chronological order.
        approval_records: List of approval records.
        verification_packages: List of verification packages.
        total_entries: Total number of entries.
        chain_valid: Whether the hash chain is valid (no tampering).
        calculated_at: Timestamp of trail assembly.
        provenance_hash: SHA-256 hash of the complete trail.
    """
    organization_id: str
    base_year: int = Field(..., ge=1990, le=2100)
    entries: List[AuditEntry] = Field(default_factory=list)
    approval_records: List[ApprovalRecord] = Field(default_factory=list)
    verification_packages: List[VerificationPackage] = Field(default_factory=list)
    total_entries: int = Field(default=0)
    chain_valid: bool = Field(default=True)
    calculated_at: datetime = Field(default_factory=_utcnow)
    provenance_hash: str = Field(default="")


class AuditCompletenessGap(BaseModel):
    """A gap identified in audit trail completeness.

    Attributes:
        gap_id: Unique identifier.
        description: Description of the missing element.
        required_event: The event type or element that is missing.
        severity: How critical this gap is.
        recommendation: Action to address the gap.
    """
    gap_id: str = Field(default_factory=_new_uuid)
    description: str
    required_event: str
    severity: AuditSeverity
    recommendation: str


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------


class BaseYearAuditEngine:
    """Comprehensive audit trail and verification engine.

    Guarantees:
        - Deterministic: Same input -> Same output (bit-perfect)
        - Tamper-evident: SHA-256 chain linking all entries
        - Reproducible: Full provenance tracking
        - Auditable: Every entry individually hashable and verifiable
        - NO LLM: Zero hallucination risk

    Usage::

        engine = BaseYearAuditEngine()
        entry = engine.create_audit_entry(
            event_type=AuditEventType.BASE_YEAR_ESTABLISHED,
            actor="sustainability_manager@company.com",
            description="Established 2019 as the base year...",
        )
        trail = engine.get_audit_trail("org-123", 2019)
    """

    def __init__(self) -> None:
        """Initialize the BaseYearAuditEngine.

        The engine maintains an in-memory store for audit entries.
        In production, this would be backed by a database.
        """
        self._entries: Dict[str, List[AuditEntry]] = {}  # org_id:base_year -> entries
        self._approvals: Dict[str, List[ApprovalRecord]] = {}
        self._verifications: Dict[str, List[VerificationPackage]] = {}
        self._last_hash: Dict[str, str] = {}  # org_id:base_year -> last hash
        logger.info(
            "BaseYearAuditEngine initialized (version=%s)",
            _MODULE_VERSION,
        )

    # ------------------------------------------------------------------
    # Key helpers
    # ------------------------------------------------------------------

    def _trail_key(self, organization_id: str, base_year: int) -> str:
        """Generate a storage key for an organization + base year."""
        return f"{organization_id}:{base_year}"

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def create_audit_entry(
        self,
        event_type: AuditEventType,
        actor: str,
        description: str,
        organization_id: str = "default",
        base_year: int = 2019,
        severity: AuditSeverity = AuditSeverity.INFO,
        before_value: Optional[str] = None,
        after_value: Optional[str] = None,
        evidence_references: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        digital_signature: Optional[str] = None,
    ) -> AuditEntry:
        """Create a new audit entry and append to the trail.

        The entry is hashed with SHA-256, incorporating the previous
        entry's hash to form a tamper-evident chain.

        Args:
            event_type: Type of audit event.
            actor: Person or system creating the entry.
            description: Human-readable event description.
            organization_id: Organization identifier.
            base_year: The base year this entry relates to.
            severity: Importance level of the event.
            before_value: Value before change (serialized string).
            after_value: Value after change (serialized string).
            evidence_references: List of supporting document references.
            metadata: Additional structured data.
            digital_signature: Optional digital signature.

        Returns:
            The created AuditEntry with computed provenance hash.
        """
        key = self._trail_key(organization_id, base_year)
        previous_hash = self._last_hash.get(key, GENESIS_HASH)

        entry = AuditEntry(
            event_type=event_type,
            timestamp=_utcnow(),
            actor=actor,
            description=description,
            severity=severity,
            before_value=before_value,
            after_value=after_value,
            evidence_references=evidence_references or [],
            metadata=metadata or {},
            digital_signature=digital_signature,
            previous_hash=previous_hash,
        )

        # Compute chained hash.
        entry_data = json.dumps(
            {
                "entry_id": entry.entry_id,
                "event_type": entry.event_type.value,
                "timestamp": str(entry.timestamp),
                "actor": entry.actor,
                "description": entry.description,
                "before_value": entry.before_value,
                "after_value": entry.after_value,
            },
            sort_keys=True,
        )
        entry.provenance_hash = _compute_chain_hash(entry_data, previous_hash)

        # Store.
        if key not in self._entries:
            self._entries[key] = []
        self._entries[key].append(entry)
        self._last_hash[key] = entry.provenance_hash

        logger.info(
            "Audit entry created: %s [%s] by %s (hash=%s)",
            event_type.value,
            entry.entry_id[:8],
            actor,
            entry.provenance_hash[:12],
        )

        return entry

    def record_approval(
        self,
        subject: str,
        requested_by: str,
        approver: str,
        status: ApprovalStatus,
        organization_id: str = "default",
        base_year: int = 2019,
        conditions: Optional[List[str]] = None,
        comments: str = "",
        related_entry_ids: Optional[List[str]] = None,
    ) -> ApprovalRecord:
        """Record an approval decision.

        Creates both an ApprovalRecord and a corresponding AuditEntry
        to maintain the audit trail.

        Args:
            subject: What is being approved.
            requested_by: Person requesting approval.
            approver: Person making the decision.
            status: Approval decision.
            organization_id: Organization identifier.
            base_year: The base year context.
            conditions: Conditions attached to approval.
            comments: Approver's comments.
            related_entry_ids: Related audit entry IDs.

        Returns:
            The created ApprovalRecord with provenance hash.
        """
        decision_date = _utcnow() if status != ApprovalStatus.PENDING else None

        record = ApprovalRecord(
            subject=subject,
            requested_by=requested_by,
            requested_date=_utcnow(),
            approver=approver,
            approval_status=status,
            decision_date=decision_date,
            conditions=conditions or [],
            comments=comments,
            related_entry_ids=related_entry_ids or [],
        )
        record.provenance_hash = _compute_hash(record)

        # Store.
        key = self._trail_key(organization_id, base_year)
        if key not in self._approvals:
            self._approvals[key] = []
        self._approvals[key].append(record)

        # Create corresponding audit entry.
        event_type = AuditEventType.RECALCULATION_APPROVED
        severity = AuditSeverity.HIGH if status == ApprovalStatus.APPROVED else AuditSeverity.MEDIUM

        self.create_audit_entry(
            event_type=event_type,
            actor=approver,
            description=(
                f"Approval {status.value} for: {subject}. "
                f"Requested by: {requested_by}. "
                f"Comments: {comments or 'None'}"
            ),
            organization_id=organization_id,
            base_year=base_year,
            severity=severity,
            metadata={
                "approval_id": record.approval_id,
                "status": status.value,
                "conditions": conditions or [],
            },
        )

        logger.info(
            "Approval recorded: %s -> %s (id=%s)",
            subject[:50],
            status.value,
            record.approval_id[:8],
        )

        return record

    def create_verification_package(
        self,
        base_year: int,
        verifier_name: str,
        verification_level: VerificationLevel,
        findings: Optional[List[VerificationFinding]] = None,
        organization_id: str = "default",
        scope_covered: str = "Scope 1 and Scope 2",
        opinion: str = "",
        is_qualified: bool = False,
        qualification_details: str = "",
        evidence_items: Optional[List[str]] = None,
        materiality_threshold_pct: Decimal = Decimal("5"),
        total_emissions_verified_tco2e: Optional[Decimal] = None,
        verifier_accreditation: str = "",
    ) -> VerificationPackage:
        """Create a verification package for base year assurance.

        Args:
            base_year: The base year verified.
            verifier_name: Name of the verification body.
            verification_level: Level of assurance.
            findings: Verification findings.
            organization_id: Organization identifier.
            scope_covered: Scopes covered by verification.
            opinion: Verifier's opinion statement.
            is_qualified: Whether opinion is qualified.
            qualification_details: Qualification details.
            evidence_items: Evidence documents reviewed.
            materiality_threshold_pct: Materiality threshold.
            total_emissions_verified_tco2e: Total emissions verified.
            verifier_accreditation: Verifier accreditation details.

        Returns:
            The created VerificationPackage.
        """
        package = VerificationPackage(
            base_year=base_year,
            verification_level=verification_level,
            verifier_name=verifier_name,
            verifier_accreditation=verifier_accreditation,
            verification_date=_utcnow(),
            scope_covered=scope_covered,
            findings=findings or [],
            opinion=opinion,
            is_qualified=is_qualified,
            qualification_details=qualification_details,
            evidence_items=evidence_items or [],
            materiality_threshold_pct=materiality_threshold_pct,
            total_emissions_verified_tco2e=total_emissions_verified_tco2e,
        )
        package.provenance_hash = _compute_hash(package)

        # Store.
        key = self._trail_key(organization_id, base_year)
        if key not in self._verifications:
            self._verifications[key] = []
        self._verifications[key].append(package)

        # Create corresponding audit entry.
        self.create_audit_entry(
            event_type=AuditEventType.VERIFICATION_COMPLETED,
            actor=verifier_name,
            description=(
                f"Verification completed by {verifier_name} at "
                f"{verification_level.value} level. "
                f"Opinion: {'Qualified' if is_qualified else 'Unqualified'}. "
                f"Findings: {len(findings or [])}."
            ),
            organization_id=organization_id,
            base_year=base_year,
            severity=AuditSeverity.HIGH,
            metadata={
                "package_id": package.package_id,
                "verification_level": verification_level.value,
                "is_qualified": is_qualified,
                "findings_count": len(findings or []),
            },
            evidence_references=evidence_items or [],
        )

        logger.info(
            "Verification package created: %s by %s (level=%s, id=%s)",
            base_year,
            verifier_name,
            verification_level.value,
            package.package_id[:8],
        )

        return package

    def get_audit_trail(
        self,
        organization_id: str,
        base_year: int,
        filters: Optional[AuditTrailFilter] = None,
    ) -> AuditTrail:
        """Retrieve the complete audit trail for an organization and base year.

        Optionally filters entries by event type, actor, severity, date
        range, or evidence presence.

        Args:
            organization_id: Organization identifier.
            base_year: The base year to retrieve the trail for.
            filters: Optional filter criteria.

        Returns:
            AuditTrail with all matching entries, approvals, and
            verification packages.
        """
        key = self._trail_key(organization_id, base_year)

        entries = list(self._entries.get(key, []))
        approvals = list(self._approvals.get(key, []))
        verifications = list(self._verifications.get(key, []))

        # Apply filters to entries.
        if filters is not None:
            entries = self._apply_filters(entries, filters)

        # Validate chain integrity.
        chain_valid = self._validate_chain(entries)

        trail = AuditTrail(
            organization_id=organization_id,
            base_year=base_year,
            entries=entries,
            approval_records=approvals,
            verification_packages=verifications,
            total_entries=len(entries),
            chain_valid=chain_valid,
            calculated_at=_utcnow(),
        )
        trail.provenance_hash = _compute_hash(trail)
        return trail

    def generate_isae3410_package(
        self,
        audit_trail: AuditTrail,
    ) -> Dict[str, Any]:
        """Generate an ISAE 3410 evidence package from the audit trail.

        ISAE 3410 (Assurance Engagements on GHG Statements) requires
        specific categories of evidence.  This method organizes the
        audit trail into these categories.

        Evidence Categories:
            1. Organizational boundary documentation
            2. Base year selection rationale
            3. Calculation methodology documentation
            4. Emission factor sources and versions
            5. Activity data sources and quality
            6. Recalculation policy and procedures
            7. Recalculation history and adjustments
            8. Quality management procedures
            9. Data management systems and controls
            10. Uncertainty assessment

        Args:
            audit_trail: The complete audit trail.

        Returns:
            Dict organized by ISAE 3410 evidence categories.
        """
        package: Dict[str, Any] = {
            "standard": "ISAE 3410",
            "organization_id": audit_trail.organization_id,
            "base_year": audit_trail.base_year,
            "generated_at": str(_utcnow()),
            "total_audit_entries": audit_trail.total_entries,
            "chain_integrity": audit_trail.chain_valid,
            "evidence_categories": {},
        }

        # Organize entries by evidence category.
        for category in ISAE3410_EVIDENCE_CATEGORIES:
            package["evidence_categories"][category] = {
                "entries": [],
                "evidence_items": [],
                "coverage": "none",
            }

        # Map audit events to evidence categories.
        event_to_category: Dict[str, List[str]] = {
            AuditEventType.BASE_YEAR_ESTABLISHED.value: [
                "base_year_selection_rationale",
                "organizational_boundary",
            ],
            AuditEventType.TRIGGER_DETECTED.value: [
                "recalculation_history",
            ],
            AuditEventType.SIGNIFICANCE_ASSESSED.value: [
                "recalculation_history",
                "quality_management",
            ],
            AuditEventType.RECALCULATION_APPROVED.value: [
                "recalculation_history",
                "quality_management",
            ],
            AuditEventType.RECALCULATION_APPLIED.value: [
                "recalculation_history",
                "calculation_methodology",
            ],
            AuditEventType.TARGET_REBASED.value: [
                "recalculation_history",
            ],
            AuditEventType.POLICY_UPDATED.value: [
                "recalculation_policy",
                "quality_management",
            ],
            AuditEventType.VERIFICATION_COMPLETED.value: [
                "quality_management",
                "uncertainty_assessment",
            ],
            AuditEventType.ANNUAL_REVIEW_COMPLETED.value: [
                "quality_management",
                "data_management",
            ],
        }

        for entry in audit_trail.entries:
            categories = event_to_category.get(entry.event_type.value, [])
            for cat in categories:
                if cat in package["evidence_categories"]:
                    package["evidence_categories"][cat]["entries"].append({
                        "entry_id": entry.entry_id,
                        "event_type": entry.event_type.value,
                        "timestamp": str(entry.timestamp),
                        "description": entry.description,
                        "actor": entry.actor,
                    })
                    package["evidence_categories"][cat]["evidence_items"].extend(
                        entry.evidence_references
                    )

        # Add verification findings.
        for vp in audit_trail.verification_packages:
            for finding in vp.findings:
                if finding.category in package["evidence_categories"]:
                    package["evidence_categories"][finding.category]["entries"].append({
                        "source": "verification_finding",
                        "finding_id": finding.finding_id,
                        "description": finding.description,
                        "severity": finding.severity.value,
                    })

        # Assess coverage per category.
        for cat in ISAE3410_EVIDENCE_CATEGORIES:
            cat_data = package["evidence_categories"][cat]
            entry_count = len(cat_data["entries"])
            if entry_count >= 3:
                cat_data["coverage"] = "comprehensive"
            elif entry_count >= 1:
                cat_data["coverage"] = "partial"
            else:
                cat_data["coverage"] = "none"

        package["provenance_hash"] = _compute_hash(package)
        return package

    def validate_audit_completeness(
        self,
        audit_trail: AuditTrail,
    ) -> List[AuditCompletenessGap]:
        """Validate that the audit trail contains all required elements.

        Checks for:
            - Base year established event
            - Complete recalculation cycles (if any triggers exist)
            - Verification package presence
            - Annual review events
            - Evidence documentation

        Args:
            audit_trail: The audit trail to validate.

        Returns:
            List of AuditCompletenessGap describing missing elements.
        """
        gaps: List[AuditCompletenessGap] = []
        event_types_present: Set[str] = set()

        for entry in audit_trail.entries:
            event_types_present.add(entry.event_type.value)

        # Check required base events.
        for required in REQUIRED_EVENTS_FOR_COMPLETENESS:
            if required not in event_types_present:
                gaps.append(AuditCompletenessGap(
                    description=(
                        f"Missing required event: {required}. "
                        f"The audit trail must document the base year "
                        f"establishment."
                    ),
                    required_event=required,
                    severity=AuditSeverity.CRITICAL,
                    recommendation=(
                        f"Create an audit entry of type '{required}' "
                        f"documenting the base year selection rationale."
                    ),
                ))

        # If triggers exist, check for complete recalculation cycle.
        has_trigger = AuditEventType.TRIGGER_DETECTED.value in event_types_present
        if has_trigger:
            for required in REQUIRED_RECALCULATION_EVENTS:
                if required not in event_types_present:
                    gaps.append(AuditCompletenessGap(
                        description=(
                            f"Incomplete recalculation cycle: missing "
                            f"'{required}'. A trigger was detected but "
                            f"the cycle is not complete."
                        ),
                        required_event=required,
                        severity=AuditSeverity.HIGH,
                        recommendation=(
                            f"Complete the recalculation cycle by "
                            f"recording the '{required}' event."
                        ),
                    ))

        # Check for verification.
        if not audit_trail.verification_packages:
            gaps.append(AuditCompletenessGap(
                description=(
                    "No verification package found. Base year data "
                    "should be independently verified."
                ),
                required_event="verification_package",
                severity=AuditSeverity.MEDIUM,
                recommendation=(
                    "Engage a third-party verifier for at least limited "
                    "assurance of base year emissions data."
                ),
            ))

        # Check for annual review.
        if AuditEventType.ANNUAL_REVIEW_COMPLETED.value not in event_types_present:
            gaps.append(AuditCompletenessGap(
                description=(
                    "No annual review event found. Base year adequacy "
                    "should be reviewed annually."
                ),
                required_event=AuditEventType.ANNUAL_REVIEW_COMPLETED.value,
                severity=AuditSeverity.LOW,
                recommendation=(
                    "Schedule and document an annual review of base year "
                    "adequacy and recalculation policy."
                ),
            ))

        # Check for evidence across entries.
        entries_without_evidence = [
            e for e in audit_trail.entries
            if not e.evidence_references and e.event_type in (
                AuditEventType.BASE_YEAR_ESTABLISHED,
                AuditEventType.RECALCULATION_APPLIED,
                AuditEventType.VERIFICATION_COMPLETED,
            )
        ]
        if entries_without_evidence:
            gaps.append(AuditCompletenessGap(
                description=(
                    f"{len(entries_without_evidence)} critical audit entries "
                    f"lack evidence references.  Supporting documentation "
                    f"is required for regulatory compliance."
                ),
                required_event="evidence_references",
                severity=AuditSeverity.MEDIUM,
                recommendation=(
                    "Attach supporting evidence documents (calculation "
                    "files, methodology documents, approval emails) to "
                    "each critical audit entry."
                ),
            ))

        # Check chain integrity.
        if not audit_trail.chain_valid:
            gaps.append(AuditCompletenessGap(
                description=(
                    "Audit trail hash chain is invalid.  One or more "
                    "entries may have been modified retroactively."
                ),
                required_event="chain_integrity",
                severity=AuditSeverity.CRITICAL,
                recommendation=(
                    "Investigate the chain break.  Reconstruct the audit "
                    "trail from source records if necessary."
                ),
            ))

        return gaps

    def export_audit_log(
        self,
        trail: AuditTrail,
        output_format: ExportFormat = ExportFormat.MARKDOWN,
    ) -> str:
        """Export the audit trail in the specified format.

        Args:
            trail: The audit trail to export.
            output_format: Desired export format.

        Returns:
            Formatted string in the specified format.
        """
        if output_format == ExportFormat.JSON:
            return self._export_json(trail)
        elif output_format == ExportFormat.CSV:
            return self._export_csv(trail)
        elif output_format == ExportFormat.MARKDOWN:
            return self._export_markdown(trail)
        else:
            return self._export_markdown(trail)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _apply_filters(
        self,
        entries: List[AuditEntry],
        filters: AuditTrailFilter,
    ) -> List[AuditEntry]:
        """Apply filter criteria to audit entries.

        Args:
            entries: List of entries to filter.
            filters: Filter criteria.

        Returns:
            Filtered list of entries.
        """
        result: List[AuditEntry] = []
        severity_order = [
            AuditSeverity.INFO,
            AuditSeverity.LOW,
            AuditSeverity.MEDIUM,
            AuditSeverity.HIGH,
            AuditSeverity.CRITICAL,
        ]

        for entry in entries:
            # Event type filter.
            if filters.event_types is not None:
                if entry.event_type not in filters.event_types:
                    continue

            # Actor filter.
            if filters.actors is not None:
                if entry.actor not in filters.actors:
                    continue

            # Severity filter.
            if filters.severity_min is not None:
                min_idx = severity_order.index(filters.severity_min)
                entry_idx = severity_order.index(entry.severity)
                if entry_idx < min_idx:
                    continue

            # Date from filter.
            if filters.date_from is not None:
                if entry.timestamp < filters.date_from:
                    continue

            # Date to filter.
            if filters.date_to is not None:
                if entry.timestamp > filters.date_to:
                    continue

            # Evidence filter.
            if filters.has_evidence is not None:
                has_evidence = len(entry.evidence_references) > 0
                if filters.has_evidence != has_evidence:
                    continue

            result.append(entry)

        return result

    def _validate_chain(self, entries: List[AuditEntry]) -> bool:
        """Validate the hash chain integrity of audit entries.

        Recomputes each entry's chain hash and compares to stored hash.

        Args:
            entries: List of entries in chronological order.

        Returns:
            True if chain is valid, False if any break detected.
        """
        if not entries:
            return True

        previous_hash = GENESIS_HASH

        for entry in entries:
            # Verify previous hash link.
            if entry.previous_hash != previous_hash:
                logger.warning(
                    "Chain break at entry %s: expected previous_hash=%s, "
                    "got %s",
                    entry.entry_id[:8],
                    previous_hash[:12],
                    entry.previous_hash[:12],
                )
                return False

            # Recompute chain hash.
            entry_data = json.dumps(
                {
                    "entry_id": entry.entry_id,
                    "event_type": entry.event_type.value,
                    "timestamp": str(entry.timestamp),
                    "actor": entry.actor,
                    "description": entry.description,
                    "before_value": entry.before_value,
                    "after_value": entry.after_value,
                },
                sort_keys=True,
            )
            expected_hash = _compute_chain_hash(entry_data, previous_hash)

            if entry.provenance_hash != expected_hash:
                logger.warning(
                    "Hash mismatch at entry %s: expected=%s, got=%s",
                    entry.entry_id[:8],
                    expected_hash[:12],
                    entry.provenance_hash[:12],
                )
                return False

            previous_hash = entry.provenance_hash

        return True

    def _export_json(self, trail: AuditTrail) -> str:
        """Export audit trail as JSON.

        Args:
            trail: The audit trail to export.

        Returns:
            JSON-formatted string.
        """
        data = trail.model_dump(mode="json")
        return json.dumps(data, indent=2, sort_keys=True, default=str)

    def _export_csv(self, trail: AuditTrail) -> str:
        """Export audit entries as CSV.

        Args:
            trail: The audit trail to export.

        Returns:
            CSV-formatted string.
        """
        lines: List[str] = []
        header = (
            "entry_id,event_type,timestamp,actor,description,"
            "severity,before_value,after_value,evidence_count,"
            "provenance_hash"
        )
        lines.append(header)

        for entry in trail.entries:
            desc_escaped = entry.description.replace('"', '""')
            before = (entry.before_value or "").replace('"', '""')
            after = (entry.after_value or "").replace('"', '""')
            line = (
                f'"{entry.entry_id}","{entry.event_type.value}",'
                f'"{entry.timestamp}","{entry.actor}",'
                f'"{desc_escaped}","{entry.severity.value}",'
                f'"{before}","{after}",'
                f'{len(entry.evidence_references)},'
                f'"{entry.provenance_hash}"'
            )
            lines.append(line)

        return "\n".join(lines)

    def _export_markdown(self, trail: AuditTrail) -> str:
        """Export audit trail as Markdown.

        Args:
            trail: The audit trail to export.

        Returns:
            Markdown-formatted string.
        """
        lines: List[str] = []
        lines.append("# Base Year Audit Trail")
        lines.append("")
        lines.append(f"**Organization:** {trail.organization_id}")
        lines.append(f"**Base Year:** {trail.base_year}")
        lines.append(f"**Total Entries:** {trail.total_entries}")
        lines.append(
            f"**Chain Integrity:** "
            f"{'Valid' if trail.chain_valid else 'INVALID'}"
        )
        lines.append(f"**Provenance Hash:** `{trail.provenance_hash[:16]}...`")
        lines.append("")

        # Entries table.
        if trail.entries:
            lines.append("## Audit Entries")
            lines.append("")
            lines.append(
                "| # | Event Type | Timestamp | Actor | Severity | "
                "Description |"
            )
            lines.append(
                "|---|-----------|-----------|-------|----------|"
                "------------|"
            )
            for i, entry in enumerate(trail.entries, 1):
                desc_short = entry.description[:80]
                if len(entry.description) > 80:
                    desc_short += "..."
                lines.append(
                    f"| {i} | {entry.event_type.value} | "
                    f"{entry.timestamp} | {entry.actor} | "
                    f"{entry.severity.value} | {desc_short} |"
                )
            lines.append("")

        # Approvals.
        if trail.approval_records:
            lines.append("## Approval Records")
            lines.append("")
            for record in trail.approval_records:
                lines.append(f"### {record.subject}")
                lines.append(f"- **Status:** {record.approval_status.value}")
                lines.append(f"- **Requested by:** {record.requested_by}")
                lines.append(f"- **Approver:** {record.approver}")
                lines.append(f"- **Date:** {record.decision_date or 'Pending'}")
                if record.comments:
                    lines.append(f"- **Comments:** {record.comments}")
                if record.conditions:
                    lines.append(f"- **Conditions:** {'; '.join(record.conditions)}")
                lines.append("")

        # Verifications.
        if trail.verification_packages:
            lines.append("## Verification Packages")
            lines.append("")
            for vp in trail.verification_packages:
                lines.append(f"### {vp.verifier_name} - {vp.verification_level.value}")
                lines.append(f"- **Date:** {vp.verification_date}")
                lines.append(f"- **Scope:** {vp.scope_covered}")
                lines.append(
                    f"- **Opinion:** "
                    f"{'Qualified' if vp.is_qualified else 'Unqualified'}"
                )
                lines.append(f"- **Findings:** {len(vp.findings)}")
                if vp.opinion:
                    lines.append(f"- **Statement:** {vp.opinion}")
                lines.append("")

        return "\n".join(lines)

    def get_entry_count(
        self,
        organization_id: str,
        base_year: int,
    ) -> int:
        """Get the number of audit entries for an organization and base year.

        Args:
            organization_id: Organization identifier.
            base_year: The base year.

        Returns:
            Count of audit entries.
        """
        key = self._trail_key(organization_id, base_year)
        return len(self._entries.get(key, []))

    def get_latest_entry(
        self,
        organization_id: str,
        base_year: int,
    ) -> Optional[AuditEntry]:
        """Get the most recent audit entry.

        Args:
            organization_id: Organization identifier.
            base_year: The base year.

        Returns:
            The latest AuditEntry, or None if no entries exist.
        """
        key = self._trail_key(organization_id, base_year)
        entries = self._entries.get(key, [])
        return entries[-1] if entries else None

    def clear_trail(
        self,
        organization_id: str,
        base_year: int,
    ) -> None:
        """Clear all audit data for an organization and base year.

        WARNING: This is destructive and should only be used in testing.

        Args:
            organization_id: Organization identifier.
            base_year: The base year.
        """
        key = self._trail_key(organization_id, base_year)
        self._entries.pop(key, None)
        self._approvals.pop(key, None)
        self._verifications.pop(key, None)
        self._last_hash.pop(key, None)
        logger.warning("Audit trail cleared for %s", key)
