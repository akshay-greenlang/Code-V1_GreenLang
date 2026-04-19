# -*- coding: utf-8 -*-
"""
ISO50001Bridge - Energy Management System Integration for PACK-031
====================================================================

This module provides integration with ISO 50001 Energy Management Systems (EnMS).
It manages EnMS documentation, internal audit scheduling, nonconformity tracking,
corrective actions, management review data aggregation, certification body
interfaces, and continual improvement evidence.

ISO 50001:2018 Clause Mapping:
    Clause 4: Context           -- Organisation context and scope
    Clause 5: Leadership        -- Energy policy and management commitment
    Clause 6: Planning          -- EnPI/EnB, objectives, targets, action plans
    Clause 7: Support           -- Resources, competence, awareness, communication
    Clause 8: Operation         -- Operational planning, design, procurement
    Clause 9: Performance       -- Monitoring, measurement, internal audit, review
    Clause 10: Improvement      -- Nonconformity, corrective action, continual improvement

Author: GreenLang Platform Team
Date: March 2026
Pack: PACK-031 Industrial Energy Audit
Status: Production Ready
"""

import hashlib
import json
import logging
import time
import uuid
from datetime import datetime, date, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from greenlang.schemas import utcnow

logger = logging.getLogger(__name__)

_MODULE_VERSION: str = "1.0.0"

def _new_uuid() -> str:
    """Generate a new UUID4 string."""
    return str(uuid.uuid4())

def _compute_hash(data: Any) -> str:
    """Compute SHA-256 hash for provenance tracking."""
    if hasattr(data, "model_dump"):
        serializable = data.model_dump(mode="json")
    elif isinstance(data, dict):
        serializable = data
    else:
        serializable = str(data)
    raw = json.dumps(serializable, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()

# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class ISO50001Clause(str, Enum):
    """ISO 50001:2018 main clause numbers."""

    CONTEXT = "clause_4"
    LEADERSHIP = "clause_5"
    PLANNING = "clause_6"
    SUPPORT = "clause_7"
    OPERATION = "clause_8"
    PERFORMANCE = "clause_9"
    IMPROVEMENT = "clause_10"

class DocumentStatus(str, Enum):
    """EnMS document lifecycle status."""

    DRAFT = "draft"
    UNDER_REVIEW = "under_review"
    APPROVED = "approved"
    SUPERSEDED = "superseded"
    OBSOLETE = "obsolete"

class NonconformitySeverity(str, Enum):
    """Nonconformity severity levels."""

    MAJOR = "major"
    MINOR = "minor"
    OBSERVATION = "observation"
    OPPORTUNITY = "opportunity_for_improvement"

class CorrectiveActionStatus(str, Enum):
    """Status of a corrective action."""

    OPEN = "open"
    IN_PROGRESS = "in_progress"
    IMPLEMENTED = "implemented"
    VERIFIED = "verified"
    CLOSED = "closed"

class CertificationStatus(str, Enum):
    """ISO 50001 certification status."""

    NOT_STARTED = "not_started"
    STAGE_1_SCHEDULED = "stage_1_scheduled"
    STAGE_1_COMPLETE = "stage_1_complete"
    STAGE_2_SCHEDULED = "stage_2_scheduled"
    CERTIFIED = "certified"
    SURVEILLANCE_DUE = "surveillance_due"
    RECERTIFICATION_DUE = "recertification_due"
    SUSPENDED = "suspended"
    WITHDRAWN = "withdrawn"

# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------

class EnMSDocument(BaseModel):
    """Energy Management System document record."""

    document_id: str = Field(default_factory=_new_uuid)
    title: str = Field(default="")
    clause: ISO50001Clause = Field(default=ISO50001Clause.CONTEXT)
    document_type: str = Field(default="procedure", description="policy|procedure|record|form")
    version: str = Field(default="1.0")
    status: DocumentStatus = Field(default=DocumentStatus.DRAFT)
    owner: str = Field(default="")
    last_review_date: Optional[date] = Field(None)
    next_review_date: Optional[date] = Field(None)
    description: str = Field(default="")

class InternalAuditRecord(BaseModel):
    """Internal EnMS audit record."""

    audit_id: str = Field(default_factory=_new_uuid)
    audit_date: Optional[date] = Field(None)
    scope: str = Field(default="")
    clauses_audited: List[str] = Field(default_factory=list)
    lead_auditor: str = Field(default="")
    audit_team: List[str] = Field(default_factory=list)
    findings_count: int = Field(default=0)
    major_nonconformities: int = Field(default=0)
    minor_nonconformities: int = Field(default=0)
    observations: int = Field(default=0)
    status: str = Field(default="planned")
    provenance_hash: str = Field(default="")

class Nonconformity(BaseModel):
    """Nonconformity record from internal or external audit."""

    nc_id: str = Field(default_factory=_new_uuid)
    audit_id: str = Field(default="")
    clause_reference: str = Field(default="")
    severity: NonconformitySeverity = Field(default=NonconformitySeverity.MINOR)
    description: str = Field(default="")
    root_cause: str = Field(default="")
    corrective_action_id: Optional[str] = Field(None)
    raised_date: Optional[date] = Field(None)
    closed_date: Optional[date] = Field(None)
    status: CorrectiveActionStatus = Field(default=CorrectiveActionStatus.OPEN)

class CorrectiveAction(BaseModel):
    """Corrective action for a nonconformity."""

    action_id: str = Field(default_factory=_new_uuid)
    nc_id: str = Field(default="")
    description: str = Field(default="")
    responsible_person: str = Field(default="")
    target_date: Optional[date] = Field(None)
    completion_date: Optional[date] = Field(None)
    verification_date: Optional[date] = Field(None)
    status: CorrectiveActionStatus = Field(default=CorrectiveActionStatus.OPEN)
    effectiveness_verified: bool = Field(default=False)

class ManagementReviewData(BaseModel):
    """Aggregated data for ISO 50001 management review (Clause 9.3)."""

    review_id: str = Field(default_factory=_new_uuid)
    review_date: Optional[date] = Field(None)
    period_start: Optional[date] = Field(None)
    period_end: Optional[date] = Field(None)
    enpi_performance: Dict[str, float] = Field(default_factory=dict)
    energy_targets_status: List[Dict[str, Any]] = Field(default_factory=list)
    internal_audit_summary: Dict[str, int] = Field(default_factory=dict)
    open_nonconformities: int = Field(default=0)
    corrective_actions_overdue: int = Field(default=0)
    energy_consumption_trend: str = Field(default="")
    improvement_opportunities: List[str] = Field(default_factory=list)
    resource_needs: List[str] = Field(default_factory=list)
    provenance_hash: str = Field(default="")

class CertificationRecord(BaseModel):
    """Certification body interaction record."""

    record_id: str = Field(default_factory=_new_uuid)
    certification_body: str = Field(default="")
    certificate_number: str = Field(default="")
    status: CertificationStatus = Field(default=CertificationStatus.NOT_STARTED)
    initial_certification_date: Optional[date] = Field(None)
    expiry_date: Optional[date] = Field(None)
    last_surveillance_date: Optional[date] = Field(None)
    next_surveillance_date: Optional[date] = Field(None)
    scope_description: str = Field(default="")

class ISO50001BridgeConfig(BaseModel):
    """Configuration for the ISO 50001 Bridge."""

    pack_id: str = Field(default="PACK-031")
    enable_provenance: bool = Field(default=True)
    internal_audit_frequency_months: int = Field(default=12, ge=1, le=24)
    management_review_frequency_months: int = Field(default=12, ge=1, le=24)
    surveillance_cycle_months: int = Field(default=12, ge=6, le=18)

# ---------------------------------------------------------------------------
# ISO50001Bridge
# ---------------------------------------------------------------------------

class ISO50001Bridge:
    """ISO 50001 Energy Management System integration.

    Manages EnMS documentation, internal audits, nonconformities,
    corrective actions, management reviews, and certification tracking.

    Attributes:
        config: Bridge configuration.
        _documents: EnMS document registry.
        _audits: Internal audit records.
        _nonconformities: Nonconformity records.
        _corrective_actions: Corrective action records.
        _certification: Certification status record.

    Example:
        >>> bridge = ISO50001Bridge()
        >>> doc = bridge.register_document("Energy Policy", ISO50001Clause.LEADERSHIP)
        >>> audit = bridge.schedule_internal_audit(["clause_6", "clause_8"])
    """

    def __init__(self, config: Optional[ISO50001BridgeConfig] = None) -> None:
        """Initialize the ISO 50001 Bridge."""
        self.config = config or ISO50001BridgeConfig()
        self.logger = logging.getLogger(self.__class__.__name__)
        self._documents: List[EnMSDocument] = []
        self._audits: List[InternalAuditRecord] = []
        self._nonconformities: List[Nonconformity] = []
        self._corrective_actions: List[CorrectiveAction] = []
        self._certification: Optional[CertificationRecord] = None
        self.logger.info("ISO50001Bridge initialized")

    # -------------------------------------------------------------------------
    # Document Management
    # -------------------------------------------------------------------------

    def register_document(
        self,
        title: str,
        clause: ISO50001Clause,
        document_type: str = "procedure",
        owner: str = "",
        description: str = "",
    ) -> EnMSDocument:
        """Register an EnMS document.

        Args:
            title: Document title.
            clause: ISO 50001 clause this document supports.
            document_type: Type of document (policy, procedure, record, form).
            owner: Document owner name.
            description: Brief description.

        Returns:
            Registered EnMSDocument.
        """
        doc = EnMSDocument(
            title=title,
            clause=clause,
            document_type=document_type,
            owner=owner,
            description=description,
            status=DocumentStatus.DRAFT,
        )
        self._documents.append(doc)
        self.logger.info("Document registered: '%s' (clause %s)", title, clause.value)
        return doc

    def get_document_matrix(self) -> Dict[str, List[Dict[str, Any]]]:
        """Get document matrix organized by ISO 50001 clause.

        Returns:
            Dict mapping clause to list of document summaries.
        """
        matrix: Dict[str, List[Dict[str, Any]]] = {}
        for clause in ISO50001Clause:
            clause_docs = [d for d in self._documents if d.clause == clause]
            matrix[clause.value] = [
                {
                    "document_id": d.document_id,
                    "title": d.title,
                    "type": d.document_type,
                    "status": d.status.value,
                    "version": d.version,
                    "owner": d.owner,
                }
                for d in clause_docs
            ]
        return matrix

    # -------------------------------------------------------------------------
    # Internal Audit Management
    # -------------------------------------------------------------------------

    def schedule_internal_audit(
        self,
        clauses: List[str],
        audit_date: Optional[date] = None,
        lead_auditor: str = "",
        scope: str = "EnMS full scope",
    ) -> InternalAuditRecord:
        """Schedule an internal EnMS audit.

        Args:
            clauses: List of ISO 50001 clause references to audit.
            audit_date: Planned audit date.
            lead_auditor: Lead auditor name.
            scope: Audit scope description.

        Returns:
            InternalAuditRecord.
        """
        record = InternalAuditRecord(
            audit_date=audit_date,
            scope=scope,
            clauses_audited=clauses,
            lead_auditor=lead_auditor,
            status="planned",
        )
        if self.config.enable_provenance:
            record.provenance_hash = _compute_hash(record)

        self._audits.append(record)
        self.logger.info(
            "Internal audit scheduled: %s, clauses=%s",
            record.audit_id, clauses,
        )
        return record

    def record_audit_findings(
        self,
        audit_id: str,
        major: int = 0,
        minor: int = 0,
        observations: int = 0,
    ) -> Optional[InternalAuditRecord]:
        """Record findings from an internal audit.

        Args:
            audit_id: Audit record identifier.
            major: Number of major nonconformities.
            minor: Number of minor nonconformities.
            observations: Number of observations.

        Returns:
            Updated InternalAuditRecord, or None if not found.
        """
        for audit in self._audits:
            if audit.audit_id == audit_id:
                audit.major_nonconformities = major
                audit.minor_nonconformities = minor
                audit.observations = observations
                audit.findings_count = major + minor + observations
                audit.status = "completed"
                if self.config.enable_provenance:
                    audit.provenance_hash = _compute_hash(audit)
                return audit
        return None

    # -------------------------------------------------------------------------
    # Nonconformity & Corrective Action
    # -------------------------------------------------------------------------

    def raise_nonconformity(
        self,
        audit_id: str,
        clause_reference: str,
        severity: NonconformitySeverity,
        description: str,
    ) -> Nonconformity:
        """Raise a nonconformity from an audit finding.

        Args:
            audit_id: Source audit identifier.
            clause_reference: ISO 50001 clause reference.
            severity: Nonconformity severity.
            description: Description of the nonconformity.

        Returns:
            Nonconformity record.
        """
        nc = Nonconformity(
            audit_id=audit_id,
            clause_reference=clause_reference,
            severity=severity,
            description=description,
            raised_date=date.today(),
        )
        self._nonconformities.append(nc)
        self.logger.info("Nonconformity raised: %s (%s)", nc.nc_id, severity.value)
        return nc

    def create_corrective_action(
        self,
        nc_id: str,
        description: str,
        responsible_person: str,
        target_date: Optional[date] = None,
    ) -> CorrectiveAction:
        """Create a corrective action for a nonconformity.

        Args:
            nc_id: Nonconformity identifier.
            description: Description of corrective action.
            responsible_person: Person responsible.
            target_date: Target completion date.

        Returns:
            CorrectiveAction record.
        """
        action = CorrectiveAction(
            nc_id=nc_id,
            description=description,
            responsible_person=responsible_person,
            target_date=target_date,
        )
        self._corrective_actions.append(action)

        # Link to nonconformity
        for nc in self._nonconformities:
            if nc.nc_id == nc_id:
                nc.corrective_action_id = action.action_id
                nc.status = CorrectiveActionStatus.IN_PROGRESS
                break

        self.logger.info("Corrective action created: %s for NC %s", action.action_id, nc_id)
        return action

    # -------------------------------------------------------------------------
    # Management Review
    # -------------------------------------------------------------------------

    def prepare_management_review(
        self,
        period_start: date,
        period_end: date,
        enpi_performance: Optional[Dict[str, float]] = None,
    ) -> ManagementReviewData:
        """Prepare aggregated data for management review.

        Args:
            period_start: Review period start.
            period_end: Review period end.
            enpi_performance: Dict of EnPI name to performance value.

        Returns:
            ManagementReviewData with aggregated inputs.
        """
        open_ncs = sum(
            1 for nc in self._nonconformities
            if nc.status in (CorrectiveActionStatus.OPEN, CorrectiveActionStatus.IN_PROGRESS)
        )
        overdue_cas = sum(
            1 for ca in self._corrective_actions
            if ca.target_date and ca.target_date < date.today()
            and ca.status not in (CorrectiveActionStatus.VERIFIED, CorrectiveActionStatus.CLOSED)
        )
        audit_summary = {
            "total_audits": len(self._audits),
            "total_findings": sum(a.findings_count for a in self._audits),
            "major_ncs": sum(a.major_nonconformities for a in self._audits),
            "minor_ncs": sum(a.minor_nonconformities for a in self._audits),
        }

        review = ManagementReviewData(
            review_date=date.today(),
            period_start=period_start,
            period_end=period_end,
            enpi_performance=enpi_performance or {},
            internal_audit_summary=audit_summary,
            open_nonconformities=open_ncs,
            corrective_actions_overdue=overdue_cas,
        )
        if self.config.enable_provenance:
            review.provenance_hash = _compute_hash(review)

        self.logger.info(
            "Management review prepared: period %s to %s, open NCs=%d",
            period_start, period_end, open_ncs,
        )
        return review

    # -------------------------------------------------------------------------
    # Certification Tracking
    # -------------------------------------------------------------------------

    def set_certification_status(
        self,
        certification_body: str,
        status: CertificationStatus,
        certificate_number: str = "",
        scope_description: str = "",
    ) -> CertificationRecord:
        """Set or update certification status.

        Args:
            certification_body: Name of the certification body.
            status: Current certification status.
            certificate_number: Certificate number (if certified).
            scope_description: Scope of certification.

        Returns:
            CertificationRecord.
        """
        if self._certification is None:
            self._certification = CertificationRecord()

        self._certification.certification_body = certification_body
        self._certification.status = status
        self._certification.certificate_number = certificate_number
        self._certification.scope_description = scope_description

        if status == CertificationStatus.CERTIFIED and not self._certification.initial_certification_date:
            self._certification.initial_certification_date = date.today()

        self.logger.info(
            "Certification status updated: body=%s, status=%s",
            certification_body, status.value,
        )
        return self._certification

    def get_certification_summary(self) -> Dict[str, Any]:
        """Get certification status summary.

        Returns:
            Dict with certification details.
        """
        if self._certification is None:
            return {"status": CertificationStatus.NOT_STARTED.value, "certified": False}

        return {
            "certification_body": self._certification.certification_body,
            "certificate_number": self._certification.certificate_number,
            "status": self._certification.status.value,
            "certified": self._certification.status == CertificationStatus.CERTIFIED,
            "initial_date": (
                self._certification.initial_certification_date.isoformat()
                if self._certification.initial_certification_date else None
            ),
            "scope": self._certification.scope_description,
        }

    # -------------------------------------------------------------------------
    # Continual Improvement Evidence
    # -------------------------------------------------------------------------

    def get_improvement_evidence(self) -> Dict[str, Any]:
        """Get evidence of continual improvement for certification audits.

        Returns:
            Dict summarizing improvement evidence.
        """
        closed_ncs = sum(
            1 for nc in self._nonconformities
            if nc.status == CorrectiveActionStatus.CLOSED
        )
        verified_cas = sum(
            1 for ca in self._corrective_actions
            if ca.effectiveness_verified
        )
        total_ncs = len(self._nonconformities)

        return {
            "total_nonconformities": total_ncs,
            "closed_nonconformities": closed_ncs,
            "closure_rate_pct": round((closed_ncs / total_ncs * 100) if total_ncs > 0 else 0, 1),
            "corrective_actions_verified": verified_cas,
            "total_corrective_actions": len(self._corrective_actions),
            "audits_conducted": len(self._audits),
            "documents_registered": len(self._documents),
            "clauses_covered": list(set(d.clause.value for d in self._documents)),
        }

    def check_health(self) -> Dict[str, Any]:
        """Check overall ISO 50001 system health.

        Returns:
            Dict with health metrics.
        """
        return {
            "documents": len(self._documents),
            "audits": len(self._audits),
            "open_nonconformities": sum(
                1 for nc in self._nonconformities
                if nc.status in (CorrectiveActionStatus.OPEN, CorrectiveActionStatus.IN_PROGRESS)
            ),
            "overdue_actions": sum(
                1 for ca in self._corrective_actions
                if ca.target_date and ca.target_date < date.today()
                and ca.status not in (CorrectiveActionStatus.VERIFIED, CorrectiveActionStatus.CLOSED)
            ),
            "certification_status": (
                self._certification.status.value if self._certification
                else CertificationStatus.NOT_STARTED.value
            ),
            "status": "healthy",
        }
