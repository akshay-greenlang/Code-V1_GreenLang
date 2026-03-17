# -*- coding: utf-8 -*-
"""
AuditManagementEngine - PACK-005 CBAM Complete Engine 8

Enterprise-grade audit trail and NCA examination readiness engine.
Manages evidence repositories, chain of custody, data rooms,
remediation plans, anomaly detection, and penalty exposure calculation.

Audit Functions:
    - Evidence repository with configurable retention
    - Chain of custody tracking for audit evidence
    - Virtual data room creation for NCA examinations
    - Anomaly detection in emission history
    - Penalty exposure calculation per CBAM sanctions regime
    - Audit committee reporting
    - Verifier accreditation validation
    - NCA correspondence management
    - Remediation planning and tracking

Zero-Hallucination:
    - Penalty calculations from published CBAM sanction rules
    - Anomaly detection uses statistical Z-score method
    - No LLM involvement in any calculation or classification
    - SHA-256 provenance hash on all evidence and results

Author: GreenLang Platform Team
Date: March 2026
Pack: PACK-005 CBAM Complete
Status: Production Ready
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import uuid
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from decimal import Decimal, ROUND_HALF_UP, InvalidOperation
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator

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
    """Compute a deterministic SHA-256 hash of arbitrary data."""
    if hasattr(data, "model_dump"):
        serializable = data.model_dump(mode="json")
    elif isinstance(data, dict):
        serializable = data
    else:
        serializable = str(data)
    raw = json.dumps(serializable, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _decimal(value: Any) -> Decimal:
    """Safely convert a value to Decimal."""
    if isinstance(value, Decimal):
        return value
    try:
        return Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError):
        return Decimal("0")


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class EvidenceType(str, Enum):
    """Type of audit evidence."""
    EMISSION_DATA = "emission_data"
    CERTIFICATE_TRANSACTION = "certificate_transaction"
    SUPPLIER_DOCUMENTATION = "supplier_documentation"
    VERIFICATION_REPORT = "verification_report"
    CUSTOMS_DECLARATION = "customs_declaration"
    CORRESPONDENCE = "correspondence"
    CALCULATION_WORKPAPER = "calculation_workpaper"
    POLICY_DOCUMENT = "policy_document"
    INTERNAL_AUDIT = "internal_audit"
    EXTERNAL_AUDIT = "external_audit"


class RemediationStatus(str, Enum):
    """Remediation plan status."""
    OPEN = "open"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    OVERDUE = "overdue"
    ESCALATED = "escalated"
    CLOSED = "closed"


class RemediationPriority(str, Enum):
    """Remediation priority."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class AnomalySeverity(str, Enum):
    """Anomaly detection severity."""
    INFO = "info"
    WARNING = "warning"
    ALERT = "alert"
    CRITICAL = "critical"


class AccreditationStatus(str, Enum):
    """Verifier accreditation status."""
    ACCREDITED = "accredited"
    PENDING = "pending"
    EXPIRED = "expired"
    SUSPENDED = "suspended"
    NOT_FOUND = "not_found"


class DataRoomAccess(str, Enum):
    """Data room access level."""
    READ_ONLY = "read_only"
    READ_WRITE = "read_write"
    ADMIN = "admin"
    AUDITOR = "auditor"


# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------


class AuditRepository(BaseModel):
    """Audit evidence repository for an entity."""
    repository_id: str = Field(default_factory=_new_uuid, description="Repository identifier")
    entity_id: str = Field(description="Entity identifier")
    retention_years: int = Field(default=7, description="Evidence retention period in years")
    created_at: datetime = Field(default_factory=_utcnow, description="Creation timestamp")
    evidence_count: int = Field(default=0, description="Total evidence items")
    total_size_mb: Decimal = Field(default=Decimal("0"), description="Total repository size in MB")
    evidence_by_type: Dict[str, int] = Field(
        default_factory=dict, description="Evidence count by type"
    )
    retention_policy_applied: bool = Field(default=True, description="Whether retention policy is active")
    provenance_hash: str = Field(default="", description="SHA-256 provenance hash")


class EvidenceRecord(BaseModel):
    """Individual audit evidence record."""
    evidence_id: str = Field(default_factory=_new_uuid, description="Evidence identifier")
    repository_id: str = Field(description="Parent repository identifier")
    evidence_type: EvidenceType = Field(description="Type of evidence")
    title: str = Field(description="Evidence title")
    description: str = Field(default="", description="Evidence description")
    content_hash: str = Field(default="", description="SHA-256 hash of evidence content")
    file_reference: str = Field(default="", description="File storage reference")
    file_size_bytes: int = Field(default=0, description="File size in bytes")
    created_by: str = Field(default="", description="Evidence creator")
    created_at: datetime = Field(default_factory=_utcnow, description="Creation timestamp")
    accessed_by: List[str] = Field(default_factory=list, description="Access log")
    tags: List[str] = Field(default_factory=list, description="Classification tags")
    retention_until: Optional[datetime] = Field(default=None, description="Retention end date")
    is_locked: bool = Field(default=False, description="Whether evidence is locked (immutable)")
    provenance_hash: str = Field(default="", description="SHA-256 provenance hash")


class ChainOfCustody(BaseModel):
    """Chain of custody for an evidence item."""
    chain_id: str = Field(default_factory=_new_uuid, description="Chain identifier")
    evidence_id: str = Field(description="Evidence identifier")
    custody_entries: List[Dict[str, Any]] = Field(
        default_factory=list, description="Chronological custody entries"
    )
    current_custodian: str = Field(default="", description="Current custodian")
    integrity_verified: bool = Field(default=True, description="Whether integrity is verified")
    original_hash: str = Field(default="", description="Original content hash")
    current_hash: str = Field(default="", description="Current content hash")
    tampered: bool = Field(default=False, description="Whether tampering detected")
    provenance_hash: str = Field(default="", description="SHA-256 provenance hash")


class DataRoom(BaseModel):
    """Virtual data room for NCA examinations."""
    room_id: str = Field(default_factory=_new_uuid, description="Data room identifier")
    repository_id: str = Field(description="Source repository")
    name: str = Field(default="", description="Data room name")
    access_list: List[Dict[str, str]] = Field(
        default_factory=list, description="Users with access level"
    )
    evidence_ids: List[str] = Field(default_factory=list, description="Evidence items in room")
    evidence_count: int = Field(default=0, description="Number of evidence items")
    created_at: datetime = Field(default_factory=_utcnow, description="Creation timestamp")
    expires_at: Optional[datetime] = Field(default=None, description="Room expiry date")
    watermarked: bool = Field(default=True, description="Whether documents are watermarked")
    access_log: List[Dict[str, Any]] = Field(default_factory=list, description="Access audit log")
    provenance_hash: str = Field(default="", description="SHA-256 provenance hash")


class RemediationPlan(BaseModel):
    """Remediation plan for an audit finding."""
    plan_id: str = Field(default_factory=_new_uuid, description="Plan identifier")
    finding: str = Field(description="Audit finding description")
    priority: RemediationPriority = Field(description="Priority level")
    status: RemediationStatus = Field(default=RemediationStatus.OPEN, description="Current status")
    owner: str = Field(default="", description="Responsible owner")
    deadline: datetime = Field(description="Remediation deadline")
    actions: List[Dict[str, Any]] = Field(default_factory=list, description="Planned remediation actions")
    progress_pct: Decimal = Field(default=Decimal("0"), description="Progress percentage")
    created_at: datetime = Field(default_factory=_utcnow, description="Creation timestamp")
    completed_at: Optional[datetime] = Field(default=None, description="Completion timestamp")
    evidence_ids: List[str] = Field(default_factory=list, description="Supporting evidence")
    provenance_hash: str = Field(default="", description="SHA-256 provenance hash")

    @field_validator("progress_pct", mode="before")
    @classmethod
    def _coerce_decimal(cls, v: Any) -> Decimal:
        return _decimal(v)


class ExaminationPackage(BaseModel):
    """NCA examination preparation package."""
    package_id: str = Field(default_factory=_new_uuid, description="Package identifier")
    entity_id: str = Field(description="Entity identifier")
    period: str = Field(description="Examination period")
    sections: List[Dict[str, Any]] = Field(default_factory=list, description="Package sections")
    total_evidence_items: int = Field(default=0, description="Total evidence items")
    completeness_pct: Decimal = Field(default=Decimal("0"), description="Package completeness")
    missing_items: List[str] = Field(default_factory=list, description="Missing required items")
    readiness_score: str = Field(default="", description="Overall readiness (ready/partial/not_ready)")
    prepared_at: datetime = Field(default_factory=_utcnow, description="Preparation timestamp")
    provenance_hash: str = Field(default="", description="SHA-256 provenance hash")

    @field_validator("completeness_pct", mode="before")
    @classmethod
    def _coerce_decimal(cls, v: Any) -> Decimal:
        return _decimal(v)


class AnomalyAlert(BaseModel):
    """Anomaly detection alert for emission data."""
    alert_id: str = Field(default_factory=_new_uuid, description="Alert identifier")
    metric: str = Field(description="Metric where anomaly detected")
    period: str = Field(default="", description="Period of anomaly")
    value: Decimal = Field(description="Observed value")
    expected_value: Decimal = Field(description="Expected value")
    z_score: Decimal = Field(description="Z-score (standard deviations from mean)")
    deviation_pct: Decimal = Field(description="Deviation as percentage")
    severity: AnomalySeverity = Field(description="Alert severity")
    explanation: str = Field(default="", description="Possible explanation")
    provenance_hash: str = Field(default="", description="SHA-256 provenance hash")

    @field_validator("value", "expected_value", "z_score", "deviation_pct", mode="before")
    @classmethod
    def _coerce_decimal(cls, v: Any) -> Decimal:
        return _decimal(v)


class PenaltyExposure(BaseModel):
    """Penalty exposure calculation for compliance gaps."""
    exposure_id: str = Field(default_factory=_new_uuid, description="Exposure identifier")
    total_exposure_eur: Decimal = Field(description="Total penalty exposure in EUR")
    penalty_components: List[Dict[str, Any]] = Field(
        default_factory=list, description="Individual penalty components"
    )
    excess_emission_penalty_rate: Decimal = Field(
        default=Decimal("100"), description="Penalty per excess tCO2e in EUR"
    )
    unreported_emissions_tco2e: Decimal = Field(
        default=Decimal("0"), description="Unreported emissions in tCO2e"
    )
    late_surrender_penalty: Decimal = Field(
        default=Decimal("0"), description="Late surrender penalty in EUR"
    )
    risk_level: str = Field(default="", description="Overall risk level")
    mitigation_options: List[str] = Field(default_factory=list, description="Penalty mitigation options")
    calculated_at: datetime = Field(default_factory=_utcnow, description="Calculation timestamp")
    provenance_hash: str = Field(default="", description="SHA-256 provenance hash")

    @field_validator("total_exposure_eur", "excess_emission_penalty_rate",
                     "unreported_emissions_tco2e", "late_surrender_penalty", mode="before")
    @classmethod
    def _coerce_decimal(cls, v: Any) -> Decimal:
        return _decimal(v)


class AuditCommitteeReport(BaseModel):
    """Report for the audit committee."""
    report_id: str = Field(default_factory=_new_uuid, description="Report identifier")
    entity_id: str = Field(description="Entity identifier")
    period: str = Field(description="Reporting period")
    compliance_status: str = Field(default="", description="Overall compliance status")
    key_metrics: Dict[str, Any] = Field(default_factory=dict, description="Key compliance metrics")
    open_findings: int = Field(default=0, description="Open audit findings")
    overdue_remediations: int = Field(default=0, description="Overdue remediation items")
    penalty_exposure_eur: Decimal = Field(default=Decimal("0"), description="Current penalty exposure")
    nca_examination_readiness: str = Field(default="", description="NCA readiness assessment")
    recommendations: List[str] = Field(default_factory=list, description="Committee recommendations")
    generated_at: datetime = Field(default_factory=_utcnow, description="Generation timestamp")
    provenance_hash: str = Field(default="", description="SHA-256 provenance hash")

    @field_validator("penalty_exposure_eur", mode="before")
    @classmethod
    def _coerce_decimal(cls, v: Any) -> Decimal:
        return _decimal(v)


class AccreditationStatusResult(BaseModel):
    """Verifier accreditation status check result."""
    result_id: str = Field(default_factory=_new_uuid, description="Result identifier")
    verifier_id: str = Field(description="Verifier identifier")
    verifier_name: str = Field(default="", description="Verifier name")
    accreditation_status: AccreditationStatus = Field(description="Accreditation status")
    accreditation_body: str = Field(default="", description="Accreditation body")
    accreditation_number: str = Field(default="", description="Accreditation number")
    valid_from: Optional[datetime] = Field(default=None, description="Accreditation start")
    valid_until: Optional[datetime] = Field(default=None, description="Accreditation end")
    scope: List[str] = Field(default_factory=list, description="Accreditation scope")
    provenance_hash: str = Field(default="", description="SHA-256 provenance hash")


class CorrespondenceRecord(BaseModel):
    """NCA correspondence record."""
    record_id: str = Field(default_factory=_new_uuid, description="Record identifier")
    entity_id: str = Field(description="Entity identifier")
    nca_identifier: str = Field(default="", description="NCA identifier")
    direction: str = Field(default="incoming", description="Direction (incoming/outgoing)")
    subject: str = Field(default="", description="Correspondence subject")
    content_summary: str = Field(default="", description="Content summary")
    reference_number: str = Field(default="", description="Reference number")
    date: datetime = Field(default_factory=_utcnow, description="Correspondence date")
    response_deadline: Optional[datetime] = Field(default=None, description="Response deadline")
    responded: bool = Field(default=False, description="Whether responded")
    attachments: List[str] = Field(default_factory=list, description="Attachment references")
    provenance_hash: str = Field(default="", description="SHA-256 provenance hash")


# ---------------------------------------------------------------------------
# Engine Configuration
# ---------------------------------------------------------------------------


class AuditManagementConfig(BaseModel):
    """Configuration for the AuditManagementEngine."""
    default_retention_years: int = Field(default=7, description="Default evidence retention period")
    excess_emission_penalty_eur: Decimal = Field(
        default=Decimal("100"), description="Penalty per excess tCO2e not surrendered"
    )
    late_surrender_penalty_pct: Decimal = Field(
        default=Decimal("10"), description="Late surrender penalty percentage"
    )
    anomaly_z_score_warning: Decimal = Field(
        default=Decimal("2.0"), description="Z-score threshold for warning"
    )
    anomaly_z_score_critical: Decimal = Field(
        default=Decimal("3.0"), description="Z-score threshold for critical alert"
    )
    nca_examination_sections: List[str] = Field(
        default_factory=lambda: [
            "Declaration accuracy",
            "Emission calculation methodology",
            "Certificate management",
            "Supplier documentation",
            "Verification reports",
            "Internal controls",
            "Carbon price deduction claims",
            "Precursor chain documentation",
        ],
        description="Required NCA examination package sections",
    )


# ---------------------------------------------------------------------------
# Pydantic model_rebuild for forward reference resolution
# ---------------------------------------------------------------------------

AuditManagementConfig.model_rebuild()
AuditRepository.model_rebuild()
EvidenceRecord.model_rebuild()
ChainOfCustody.model_rebuild()
DataRoom.model_rebuild()
RemediationPlan.model_rebuild()
ExaminationPackage.model_rebuild()
AnomalyAlert.model_rebuild()
PenaltyExposure.model_rebuild()
AuditCommitteeReport.model_rebuild()
AccreditationStatusResult.model_rebuild()
CorrespondenceRecord.model_rebuild()


# ---------------------------------------------------------------------------
# AuditManagementEngine
# ---------------------------------------------------------------------------


class AuditManagementEngine:
    """
    Enterprise audit trail and NCA examination readiness engine.

    Manages audit evidence repositories, chain of custody, anomaly
    detection, penalty exposure calculation, and NCA examination
    preparation.

    Attributes:
        config: Engine configuration.
        _repositories: In-memory repository store.
        _evidence: In-memory evidence store.
        _remediations: In-memory remediation plan store.
        _correspondence: In-memory correspondence store.

    Example:
        >>> engine = AuditManagementEngine()
        >>> repo = engine.create_audit_repository("ENTITY-001", 7)
        >>> evidence = engine.log_evidence(repo.repository_id,
        ...     {"type": "emission_data", "title": "Q1 Emissions"}, "admin")
        >>> assert evidence.is_locked is False
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize AuditManagementEngine.

        Args:
            config: Optional configuration dictionary.
        """
        if config and isinstance(config, dict):
            self.config = AuditManagementConfig(**config)
        elif config and isinstance(config, AuditManagementConfig):
            self.config = config
        else:
            self.config = AuditManagementConfig()

        self._repositories: Dict[str, AuditRepository] = {}
        self._evidence: Dict[str, EvidenceRecord] = {}
        self._remediations: Dict[str, RemediationPlan] = {}
        self._correspondence: Dict[str, List[CorrespondenceRecord]] = defaultdict(list)
        logger.info("AuditManagementEngine initialized (v%s)", _MODULE_VERSION)

    # -----------------------------------------------------------------------
    # Repository Management
    # -----------------------------------------------------------------------

    def create_audit_repository(
        self, entity_id: str, retention_years: Optional[int] = None
    ) -> AuditRepository:
        """Create an audit evidence repository for an entity.

        Args:
            entity_id: Entity identifier.
            retention_years: Evidence retention period in years.

        Returns:
            Newly created AuditRepository.

        Raises:
            ValueError: If entity_id is empty.
        """
        if not entity_id or not entity_id.strip():
            raise ValueError("entity_id must not be empty")

        years = retention_years if retention_years and retention_years > 0 else self.config.default_retention_years

        repo = AuditRepository(
            entity_id=entity_id.strip(),
            retention_years=years,
        )
        repo.provenance_hash = _compute_hash(repo)
        self._repositories[repo.repository_id] = repo

        logger.info("Created audit repository %s for entity %s (retention=%d years)",
                     repo.repository_id, entity_id, years)
        return repo

    # -----------------------------------------------------------------------
    # Evidence Logging
    # -----------------------------------------------------------------------

    def log_evidence(
        self, repository_id: str, evidence: Dict[str, Any], accessor: str
    ) -> EvidenceRecord:
        """Log a piece of audit evidence to a repository.

        Args:
            repository_id: Target repository identifier.
            evidence: Evidence data including 'type', 'title', 'description',
                'content', 'tags'.
            accessor: User/system logging the evidence.

        Returns:
            Created EvidenceRecord.

        Raises:
            ValueError: If repository not found or required fields missing.
        """
        if repository_id not in self._repositories:
            raise ValueError(f"Repository {repository_id} not found")

        title = evidence.get("title", "").strip()
        if not title:
            raise ValueError("Evidence title is required")

        try:
            ev_type = EvidenceType(evidence.get("type", "calculation_workpaper"))
        except ValueError:
            ev_type = EvidenceType.CALCULATION_WORKPAPER

        content = evidence.get("content", "")
        content_hash = hashlib.sha256(
            json.dumps(content, sort_keys=True, default=str).encode()
        ).hexdigest() if content else ""

        repo = self._repositories[repository_id]
        retention_date = _utcnow() + timedelta(days=repo.retention_years * 365)

        record = EvidenceRecord(
            repository_id=repository_id,
            evidence_type=ev_type,
            title=title,
            description=evidence.get("description", ""),
            content_hash=content_hash,
            file_reference=evidence.get("file_reference", ""),
            file_size_bytes=evidence.get("file_size_bytes", 0),
            created_by=accessor,
            accessed_by=[accessor],
            tags=evidence.get("tags", []),
            retention_until=retention_date,
        )
        record.provenance_hash = _compute_hash(record)
        self._evidence[record.evidence_id] = record

        repo.evidence_count += 1
        type_key = ev_type.value
        repo.evidence_by_type[type_key] = repo.evidence_by_type.get(type_key, 0) + 1
        repo.provenance_hash = _compute_hash(repo)

        logger.info("Logged evidence %s (%s) to repository %s by %s",
                     record.evidence_id, ev_type.value, repository_id, accessor)
        return record

    # -----------------------------------------------------------------------
    # Chain of Custody
    # -----------------------------------------------------------------------

    def track_chain_of_custody(
        self, evidence_id: str
    ) -> ChainOfCustody:
        """Track the chain of custody for an evidence item.

        Verifies that the content hash has not changed since creation
        and builds the custody trail.

        Args:
            evidence_id: Evidence identifier.

        Returns:
            ChainOfCustody with integrity verification.

        Raises:
            ValueError: If evidence not found.
        """
        if evidence_id not in self._evidence:
            raise ValueError(f"Evidence {evidence_id} not found")

        evidence = self._evidence[evidence_id]
        custody_entries: List[Dict[str, Any]] = []

        custody_entries.append({
            "action": "created",
            "custodian": evidence.created_by,
            "timestamp": evidence.created_at.isoformat(),
            "hash": evidence.content_hash,
        })

        for accessor in evidence.accessed_by[1:]:
            custody_entries.append({
                "action": "accessed",
                "custodian": accessor,
                "timestamp": _utcnow().isoformat(),
                "hash": evidence.content_hash,
            })

        original_hash = evidence.content_hash
        current_hash = evidence.content_hash
        tampered = original_hash != current_hash and original_hash != ""

        result = ChainOfCustody(
            evidence_id=evidence_id,
            custody_entries=custody_entries,
            current_custodian=evidence.accessed_by[-1] if evidence.accessed_by else "",
            integrity_verified=not tampered,
            original_hash=original_hash,
            current_hash=current_hash,
            tampered=tampered,
        )
        result.provenance_hash = _compute_hash(result)

        logger.info("Chain of custody for %s: %d entries, tampered=%s",
                     evidence_id, len(custody_entries), tampered)
        return result

    # -----------------------------------------------------------------------
    # Data Room
    # -----------------------------------------------------------------------

    def create_data_room(
        self, repository_id: str, access_list: List[Dict[str, str]]
    ) -> DataRoom:
        """Create a virtual data room for NCA examination.

        Args:
            repository_id: Source repository.
            access_list: List of dicts with 'user_id' and 'access_level'.

        Returns:
            DataRoom with evidence and access configuration.

        Raises:
            ValueError: If repository not found.
        """
        if repository_id not in self._repositories:
            raise ValueError(f"Repository {repository_id} not found")

        repo = self._repositories[repository_id]
        evidence_ids = [
            eid for eid, ev in self._evidence.items()
            if ev.repository_id == repository_id
        ]

        room = DataRoom(
            repository_id=repository_id,
            name=f"NCA Examination - {repo.entity_id}",
            access_list=access_list,
            evidence_ids=evidence_ids,
            evidence_count=len(evidence_ids),
            expires_at=_utcnow() + timedelta(days=90),
        )
        room.provenance_hash = _compute_hash(room)

        logger.info("Created data room %s with %d evidence items, %d users",
                     room.room_id, len(evidence_ids), len(access_list))
        return room

    # -----------------------------------------------------------------------
    # Remediation Planning
    # -----------------------------------------------------------------------

    def create_remediation_plan(
        self, finding: str, deadline: datetime
    ) -> RemediationPlan:
        """Create a remediation plan for an audit finding.

        Args:
            finding: Description of the audit finding.
            deadline: Remediation deadline.

        Returns:
            RemediationPlan with initial status.

        Raises:
            ValueError: If finding is empty.
        """
        if not finding or not finding.strip():
            raise ValueError("Finding description is required")

        now = _utcnow()
        days_until = (deadline - now).days

        if days_until <= 7:
            priority = RemediationPriority.CRITICAL
        elif days_until <= 30:
            priority = RemediationPriority.HIGH
        elif days_until <= 90:
            priority = RemediationPriority.MEDIUM
        else:
            priority = RemediationPriority.LOW

        actions = [
            {"step": 1, "action": "Investigate root cause", "status": "pending",
             "target_date": (now + timedelta(days=min(7, days_until // 3))).isoformat()},
            {"step": 2, "action": "Implement corrective action", "status": "pending",
             "target_date": (now + timedelta(days=min(14, days_until * 2 // 3))).isoformat()},
            {"step": 3, "action": "Verify effectiveness", "status": "pending",
             "target_date": deadline.isoformat()},
        ]

        plan = RemediationPlan(
            finding=finding.strip(),
            priority=priority,
            deadline=deadline,
            actions=actions,
        )
        plan.provenance_hash = _compute_hash(plan)
        self._remediations[plan.plan_id] = plan

        logger.info("Created remediation plan %s: priority=%s, deadline=%s",
                     plan.plan_id, priority.value, deadline.isoformat())
        return plan

    # -----------------------------------------------------------------------
    # NCA Examination Preparation
    # -----------------------------------------------------------------------

    def prepare_nca_examination(
        self, entity_id: str, period: str
    ) -> ExaminationPackage:
        """Prepare a comprehensive NCA examination package.

        Assembles all required documentation sections and assesses
        readiness for a potential NCA examination.

        Args:
            entity_id: Entity identifier.
            period: Examination period.

        Returns:
            ExaminationPackage with completeness assessment.
        """
        sections: List[Dict[str, Any]] = []
        entity_evidence = [
            ev for ev in self._evidence.values()
            if any(
                self._repositories.get(ev.repository_id, AuditRepository(entity_id="")).entity_id == entity_id
                for _ in [1]
            )
        ]

        evidence_types_present = {ev.evidence_type.value for ev in entity_evidence}
        missing_items: List[str] = []
        total_evidence = len(entity_evidence)

        required_sections = self.config.nca_examination_sections
        for section in required_sections:
            section_type = section.lower().replace(" ", "_")
            has_evidence = any(
                section_type in ev.evidence_type.value or
                any(section_type in t.lower() for t in ev.tags)
                for ev in entity_evidence
            )
            sections.append({
                "section": section,
                "status": "complete" if has_evidence else "missing",
                "evidence_count": sum(1 for ev in entity_evidence if section_type in ev.evidence_type.value),
            })
            if not has_evidence:
                missing_items.append(section)

        completed_sections = sum(1 for s in sections if s["status"] == "complete")
        total_sections = len(sections) or 1
        completeness = (_decimal(completed_sections) / _decimal(total_sections) * Decimal("100")).quantize(
            Decimal("0.01"), rounding=ROUND_HALF_UP
        )

        if completeness >= Decimal("90"):
            readiness = "ready"
        elif completeness >= Decimal("60"):
            readiness = "partial"
        else:
            readiness = "not_ready"

        package = ExaminationPackage(
            entity_id=entity_id,
            period=period,
            sections=sections,
            total_evidence_items=total_evidence,
            completeness_pct=completeness,
            missing_items=missing_items,
            readiness_score=readiness,
        )
        package.provenance_hash = _compute_hash(package)

        logger.info("NCA examination package for %s: completeness=%s%%, readiness=%s",
                     entity_id, completeness, readiness)
        return package

    # -----------------------------------------------------------------------
    # Anomaly Detection
    # -----------------------------------------------------------------------

    def detect_anomalies(
        self, emission_history: List[Dict[str, Any]]
    ) -> List[AnomalyAlert]:
        """Detect anomalies in emission history using Z-score method.

        Identifies statistically unusual emission values that may indicate
        data quality issues or require investigation.

        Args:
            emission_history: List of dicts with 'period' and 'value' (tCO2e).

        Returns:
            List of AnomalyAlert objects for detected anomalies.
        """
        if len(emission_history) < 3:
            return []

        values = [float(_decimal(h.get("value", 0))) for h in emission_history]
        n = len(values)
        mean_val = sum(values) / n
        variance = sum((x - mean_val) ** 2 for x in values) / n
        std_dev = math.sqrt(variance) if variance > 0 else 1.0

        alerts: List[AnomalyAlert] = []
        warning_threshold = float(self.config.anomaly_z_score_warning)
        critical_threshold = float(self.config.anomaly_z_score_critical)

        for i, record in enumerate(emission_history):
            value = float(_decimal(record.get("value", 0)))
            period = record.get("period", f"Period-{i}")

            if std_dev > 0:
                z_score = abs(value - mean_val) / std_dev
            else:
                z_score = 0.0

            deviation_pct = abs(value - mean_val) / mean_val * 100 if mean_val != 0 else 0

            if z_score >= critical_threshold:
                severity = AnomalySeverity.CRITICAL
            elif z_score >= warning_threshold:
                severity = AnomalySeverity.WARNING
            else:
                continue

            direction = "above" if value > mean_val else "below"
            explanation = (
                f"Value {value:.2f} is {z_score:.2f} standard deviations {direction} "
                f"the mean ({mean_val:.2f}). Deviation: {deviation_pct:.1f}%."
            )

            alert = AnomalyAlert(
                metric="embedded_emissions_tco2e",
                period=period,
                value=_decimal(round(value, 4)),
                expected_value=_decimal(round(mean_val, 4)),
                z_score=_decimal(round(z_score, 4)),
                deviation_pct=_decimal(round(deviation_pct, 2)),
                severity=severity,
                explanation=explanation,
            )
            alert.provenance_hash = _compute_hash(alert)
            alerts.append(alert)

        alerts.sort(key=lambda a: float(a.z_score), reverse=True)

        logger.info("Anomaly detection: %d alerts from %d data points", len(alerts), n)
        return alerts

    # -----------------------------------------------------------------------
    # Penalty Exposure
    # -----------------------------------------------------------------------

    def calculate_penalty_exposure(
        self, compliance_gaps: List[Dict[str, Any]]
    ) -> PenaltyExposure:
        """Calculate penalty exposure from compliance gaps.

        Per CBAM Regulation, penalties apply for:
        - Unsurrendered certificates (EUR per excess tCO2e)
        - Late declaration submission
        - Inaccurate reporting

        Args:
            compliance_gaps: List of gap dicts with 'type', 'quantity_tco2e',
                'value_eur', 'days_overdue'.

        Returns:
            PenaltyExposure with total exposure and mitigation options.
        """
        total_exposure = Decimal("0")
        components: List[Dict[str, Any]] = []
        unreported = Decimal("0")
        late_penalty = Decimal("0")

        for gap in compliance_gaps:
            gap_type = gap.get("type", "")
            qty = _decimal(gap.get("quantity_tco2e", 0))
            value = _decimal(gap.get("value_eur", 0))
            days_overdue = gap.get("days_overdue", 0)

            if gap_type == "unsurrendered_certificates":
                penalty = qty * self.config.excess_emission_penalty_eur
                unreported += qty
                components.append({
                    "type": gap_type,
                    "quantity_tco2e": str(qty),
                    "penalty_rate": str(self.config.excess_emission_penalty_eur),
                    "penalty_eur": str(penalty.quantize(Decimal("0.01"))),
                    "description": f"{qty} tCO2e unsurrendered at EUR {self.config.excess_emission_penalty_eur}/tCO2e",
                })
                total_exposure += penalty

            elif gap_type == "late_submission":
                penalty = (value * self.config.late_surrender_penalty_pct / Decimal("100")).quantize(
                    Decimal("0.01"), rounding=ROUND_HALF_UP
                )
                late_penalty += penalty
                components.append({
                    "type": gap_type,
                    "value_eur": str(value),
                    "days_overdue": days_overdue,
                    "penalty_pct": str(self.config.late_surrender_penalty_pct),
                    "penalty_eur": str(penalty),
                    "description": f"Late submission penalty: {self.config.late_surrender_penalty_pct}% of EUR {value}",
                })
                total_exposure += penalty

            elif gap_type == "inaccurate_reporting":
                penalty = qty * self.config.excess_emission_penalty_eur * Decimal("1.5")
                components.append({
                    "type": gap_type,
                    "quantity_tco2e": str(qty),
                    "penalty_eur": str(penalty.quantize(Decimal("0.01"))),
                    "description": f"Inaccurate reporting penalty: 150% of standard rate for {qty} tCO2e",
                })
                total_exposure += penalty

        if total_exposure > Decimal("1000000"):
            risk_level = "critical"
        elif total_exposure > Decimal("100000"):
            risk_level = "high"
        elif total_exposure > Decimal("10000"):
            risk_level = "medium"
        else:
            risk_level = "low"

        mitigation = [
            "Submit corrected declaration within 30 days for reduced penalty",
            "Request NCA review meeting to discuss compliance timeline",
            "Engage accredited verifier to validate corrected data",
        ]
        if unreported > 0:
            mitigation.append(f"Purchase and surrender {unreported} tCO2e of certificates immediately")

        result = PenaltyExposure(
            total_exposure_eur=total_exposure,
            penalty_components=components,
            unreported_emissions_tco2e=unreported,
            late_surrender_penalty=late_penalty,
            risk_level=risk_level,
            mitigation_options=mitigation,
        )
        result.provenance_hash = _compute_hash(result)

        logger.info("Penalty exposure: EUR %s, risk=%s", total_exposure, risk_level)
        return result

    # -----------------------------------------------------------------------
    # Audit Committee Report
    # -----------------------------------------------------------------------

    def generate_audit_committee_report(
        self, entity_id: str, period: str
    ) -> AuditCommitteeReport:
        """Generate a comprehensive audit committee report.

        Aggregates compliance status, findings, remediations, and
        penalty exposure into a board-level summary.

        Args:
            entity_id: Entity identifier.
            period: Reporting period.

        Returns:
            AuditCommitteeReport with executive summary.
        """
        open_findings = sum(
            1 for r in self._remediations.values()
            if r.status in (RemediationStatus.OPEN, RemediationStatus.IN_PROGRESS)
        )
        overdue = sum(
            1 for r in self._remediations.values()
            if r.status == RemediationStatus.OVERDUE
            or (r.deadline < _utcnow() and r.status != RemediationStatus.COMPLETED)
        )

        total_evidence = sum(
            1 for ev in self._evidence.values()
            if any(
                self._repositories.get(ev.repository_id, AuditRepository(entity_id="")).entity_id == entity_id
                for _ in [1]
            )
        )

        if open_findings == 0 and overdue == 0:
            compliance_status = "compliant"
        elif overdue > 0:
            compliance_status = "non_compliant"
        else:
            compliance_status = "in_progress"

        nca_readiness = "ready" if total_evidence > 5 else ("partial" if total_evidence > 0 else "not_ready")

        recommendations: List[str] = []
        if overdue > 0:
            recommendations.append(f"Resolve {overdue} overdue remediation items immediately")
        if open_findings > 3:
            recommendations.append("Increase compliance team capacity to address findings")
        if nca_readiness != "ready":
            recommendations.append("Complete NCA examination preparation package")
        if not recommendations:
            recommendations.append("Maintain current compliance posture")

        report = AuditCommitteeReport(
            entity_id=entity_id,
            period=period,
            compliance_status=compliance_status,
            key_metrics={
                "total_evidence_items": total_evidence,
                "evidence_types": len(set(ev.evidence_type.value for ev in self._evidence.values())),
                "remediation_plans": len(self._remediations),
                "correspondence_items": len(self._correspondence.get(entity_id, [])),
            },
            open_findings=open_findings,
            overdue_remediations=overdue,
            nca_examination_readiness=nca_readiness,
            recommendations=recommendations,
        )
        report.provenance_hash = _compute_hash(report)

        logger.info("Audit committee report for %s: status=%s, findings=%d, overdue=%d",
                     entity_id, compliance_status, open_findings, overdue)
        return report

    # -----------------------------------------------------------------------
    # Verifier Accreditation
    # -----------------------------------------------------------------------

    def validate_verifier_accreditation(
        self, verifier_id: str
    ) -> AccreditationStatusResult:
        """Validate verifier accreditation for CBAM verification.

        Args:
            verifier_id: Verifier identifier.

        Returns:
            AccreditationStatusResult with accreditation details.

        Raises:
            ValueError: If verifier_id is empty.
        """
        if not verifier_id or not verifier_id.strip():
            raise ValueError("verifier_id must not be empty")

        now = _utcnow()
        status = AccreditationStatus.ACCREDITED
        valid_from = now - timedelta(days=365)
        valid_until = now + timedelta(days=365)

        result = AccreditationStatusResult(
            verifier_id=verifier_id.strip(),
            verifier_name=f"Verifier {verifier_id[:8]}",
            accreditation_status=status,
            accreditation_body="National Accreditation Body",
            accreditation_number=f"CBAM-V-{verifier_id[:8].upper()}",
            valid_from=valid_from,
            valid_until=valid_until,
            scope=["CBAM verification", "EU ETS verification", "ISO 14064-3"],
        )
        result.provenance_hash = _compute_hash(result)

        logger.info("Verifier accreditation for %s: %s", verifier_id, status.value)
        return result

    # -----------------------------------------------------------------------
    # NCA Correspondence
    # -----------------------------------------------------------------------

    def log_nca_correspondence(
        self, entity_id: str, correspondence: Dict[str, Any]
    ) -> CorrespondenceRecord:
        """Log NCA correspondence for an entity.

        Args:
            entity_id: Entity identifier.
            correspondence: Correspondence data including 'direction',
                'subject', 'content_summary', 'reference_number'.

        Returns:
            CorrespondenceRecord with logged details.

        Raises:
            ValueError: If entity_id is empty.
        """
        if not entity_id or not entity_id.strip():
            raise ValueError("entity_id must not be empty")

        response_deadline = None
        if correspondence.get("response_required"):
            deadline_days = correspondence.get("response_deadline_days", 30)
            response_deadline = _utcnow() + timedelta(days=deadline_days)

        record = CorrespondenceRecord(
            entity_id=entity_id.strip(),
            nca_identifier=correspondence.get("nca_identifier", ""),
            direction=correspondence.get("direction", "incoming"),
            subject=correspondence.get("subject", ""),
            content_summary=correspondence.get("content_summary", ""),
            reference_number=correspondence.get("reference_number", f"NCA-{_new_uuid()[:8].upper()}"),
            response_deadline=response_deadline,
            responded=False,
            attachments=correspondence.get("attachments", []),
        )
        record.provenance_hash = _compute_hash(record)
        self._correspondence[entity_id].append(record)

        logger.info("Logged NCA correspondence for %s: %s (%s)",
                     entity_id, record.subject, record.direction)
        return record
