# -*- coding: utf-8 -*-
"""
CertificationBodyBridge - Audit Body Interface for ISO 50001 EnMS
===================================================================

This module provides integration with certification bodies for ISO 50001
EnMS certification. It manages certification body registration, audit
scheduling (Stage 1, Stage 2, surveillance, recertification), documentation
submission, audit report processing, and certificate lifecycle tracking.

Audit Types:
    - Stage 1: Documentation review and readiness assessment
    - Stage 2: On-site implementation audit
    - Surveillance: Annual surveillance audit (years 1 and 2)
    - Recertification: Full recertification audit (year 3)

Certificate Lifecycle:
    Active --> Surveillance Due --> Suspended --> Withdrawn
    Active --> Recertification Due --> Active (renewed)
    Active --> Expired

Author: GreenLang Platform Team
Date: March 2026
Pack: PACK-034 ISO 50001 Energy Management System
Status: Production Ready
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
import uuid
from datetime import datetime, timezone
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

class AuditType(str, Enum):
    """ISO 50001 certification audit types."""

    STAGE1 = "stage1"
    STAGE2 = "stage2"
    SURVEILLANCE = "surveillance"
    RECERTIFICATION = "recertification"

class CertificateStatus(str, Enum):
    """ISO 50001 certificate lifecycle status."""

    ACTIVE = "active"
    SUSPENDED = "suspended"
    WITHDRAWN = "withdrawn"
    EXPIRED = "expired"
    PENDING = "pending"

class FindingType(str, Enum):
    """Audit finding classification."""

    MAJOR_NONCONFORMITY = "major_nonconformity"
    MINOR_NONCONFORMITY = "minor_nonconformity"
    OBSERVATION = "observation"
    OPPORTUNITY_FOR_IMPROVEMENT = "opportunity_for_improvement"
    POSITIVE_PRACTICE = "positive_practice"

# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------

class CertificationBody(BaseModel):
    """Certification body profile."""

    body_id: str = Field(default_factory=_new_uuid)
    name: str = Field(default="", max_length=255)
    accreditation_body: str = Field(default="", description="Accreditation body (e.g., DAkkS, UKAS)")
    accreditation_number: str = Field(default="")
    country: str = Field(default="")
    contact_email: str = Field(default="")
    contact_phone: str = Field(default="")
    auditors: List[Dict[str, Any]] = Field(default_factory=list)
    iso50001_accredited: bool = Field(default=True)
    registered_at: datetime = Field(default_factory=utcnow)

class AuditSchedule(BaseModel):
    """Scheduled audit details."""

    schedule_id: str = Field(default_factory=_new_uuid)
    audit_type: AuditType = Field(...)
    certification_body_id: str = Field(default="")
    lead_auditor: str = Field(default="")
    planned_date_start: str = Field(default="")
    planned_date_end: str = Field(default="")
    actual_date_start: Optional[str] = Field(None)
    actual_date_end: Optional[str] = Field(None)
    duration_days: int = Field(default=2, ge=1)
    status: str = Field(default="planned", description="planned|confirmed|in_progress|completed|cancelled")
    scope_description: str = Field(default="")
    notes: str = Field(default="")
    provenance_hash: str = Field(default="")

class AuditReport(BaseModel):
    """Certification audit report."""

    report_id: str = Field(default_factory=_new_uuid)
    audit_type: AuditType = Field(...)
    schedule_id: str = Field(default="")
    certification_body_id: str = Field(default="")
    lead_auditor: str = Field(default="")
    audit_date: str = Field(default="")
    clauses_audited: List[str] = Field(default_factory=list)
    major_nonconformities: int = Field(default=0, ge=0)
    minor_nonconformities: int = Field(default=0, ge=0)
    observations: int = Field(default=0, ge=0)
    positive_practices: int = Field(default=0, ge=0)
    findings: List[Dict[str, Any]] = Field(default_factory=list)
    recommendation: str = Field(default="", description="certify|conditional|not_recommended")
    corrective_action_deadline: Optional[str] = Field(None)
    overall_assessment: str = Field(default="")
    energy_performance_improvement_noted: bool = Field(default=False)
    provenance_hash: str = Field(default="")

class CertificateInfo(BaseModel):
    """ISO 50001 certificate details."""

    certificate_id: str = Field(default_factory=_new_uuid)
    certificate_number: str = Field(default="")
    organization_name: str = Field(default="")
    scope: str = Field(default="")
    status: CertificateStatus = Field(default=CertificateStatus.PENDING)
    issue_date: Optional[str] = Field(None)
    expiry_date: Optional[str] = Field(None)
    certification_body_id: str = Field(default="")
    certification_body_name: str = Field(default="")
    iso_version: str = Field(default="2018")
    last_surveillance_date: Optional[str] = Field(None)
    next_surveillance_date: Optional[str] = Field(None)
    next_recertification_date: Optional[str] = Field(None)
    provenance_hash: str = Field(default="")

# ---------------------------------------------------------------------------
# CertificationBodyBridge
# ---------------------------------------------------------------------------

class CertificationBodyBridge:
    """Certification body interface for ISO 50001 EnMS.

    Manages certification body registration, audit scheduling,
    documentation submission, audit report processing, and certificate
    lifecycle tracking.

    Attributes:
        _bodies: Registered certification bodies.
        _schedules: Audit schedules.
        _reports: Audit reports.
        _certificates: Certificate records.

    Example:
        >>> bridge = CertificationBodyBridge()
        >>> body = bridge.register_certification_body({"name": "TUV SUD", ...})
        >>> schedule = bridge.schedule_audit(AuditType.STAGE1, {...})
        >>> status = bridge.track_certificate_status()
    """

    def __init__(self) -> None:
        """Initialize the Certification Body Bridge."""
        self.logger = logging.getLogger(self.__class__.__name__)
        self._bodies: Dict[str, CertificationBody] = {}
        self._schedules: Dict[str, AuditSchedule] = {}
        self._reports: Dict[str, AuditReport] = {}
        self._certificates: Dict[str, CertificateInfo] = {}
        self.logger.info("CertificationBodyBridge initialized")

    def register_certification_body(
        self, body_data: Dict[str, Any],
    ) -> CertificationBody:
        """Register a certification body.

        Args:
            body_data: Dict with certification body details.

        Returns:
            Registered CertificationBody.
        """
        body = CertificationBody(
            name=body_data.get("name", ""),
            accreditation_body=body_data.get("accreditation_body", ""),
            accreditation_number=body_data.get("accreditation_number", ""),
            country=body_data.get("country", ""),
            contact_email=body_data.get("contact_email", ""),
            contact_phone=body_data.get("contact_phone", ""),
            auditors=body_data.get("auditors", []),
            iso50001_accredited=body_data.get("iso50001_accredited", True),
        )
        self._bodies[body.body_id] = body
        self.logger.info("Certification body registered: %s (%s)", body.name, body.body_id)
        return body

    def schedule_audit(
        self,
        audit_type: AuditType,
        schedule_data: Dict[str, Any],
    ) -> AuditSchedule:
        """Schedule a certification audit.

        Args:
            audit_type: Type of audit (stage1, stage2, surveillance, recertification).
            schedule_data: Dict with scheduling details.

        Returns:
            AuditSchedule with scheduling details.
        """
        schedule = AuditSchedule(
            audit_type=audit_type,
            certification_body_id=schedule_data.get("certification_body_id", ""),
            lead_auditor=schedule_data.get("lead_auditor", ""),
            planned_date_start=schedule_data.get("date_start", ""),
            planned_date_end=schedule_data.get("date_end", ""),
            duration_days=schedule_data.get("duration_days", 2),
            scope_description=schedule_data.get("scope", "Full EnMS scope"),
            notes=schedule_data.get("notes", ""),
        )
        schedule.provenance_hash = _compute_hash(schedule)
        self._schedules[schedule.schedule_id] = schedule

        self.logger.info(
            "Audit scheduled: type=%s, date=%s, body=%s",
            audit_type.value, schedule.planned_date_start,
            schedule.certification_body_id,
        )
        return schedule

    def submit_documentation(
        self, document_package: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Submit documentation package to certification body.

        Args:
            document_package: Dict with document references and metadata.

        Returns:
            Dict with submission confirmation.
        """
        start = time.monotonic()

        required_documents = [
            "energy_policy",
            "scope_and_boundaries",
            "energy_review",
            "enpi_report",
            "action_plans",
            "internal_audit_report",
            "management_review_minutes",
        ]

        submitted = document_package.get("documents", [])
        submitted_names = [d.get("name", "") for d in submitted]
        missing = [doc for doc in required_documents if doc not in submitted_names]

        result = {
            "submission_id": _new_uuid(),
            "submitted_at": utcnow().isoformat(),
            "documents_submitted": len(submitted),
            "documents_required": len(required_documents),
            "missing_documents": missing,
            "submission_complete": len(missing) == 0,
            "message": (
                "All required documents submitted"
                if not missing
                else f"Missing {len(missing)} required documents"
            ),
            "duration_ms": round((time.monotonic() - start) * 1000, 1),
            "provenance_hash": _compute_hash(document_package),
        }

        self.logger.info(
            "Documentation submitted: %d/%d documents, complete=%s",
            len(submitted), len(required_documents), not missing,
        )
        return result

    def receive_audit_report(
        self, report_data: Dict[str, Any],
    ) -> AuditReport:
        """Process a received audit report.

        Args:
            report_data: Dict with audit report details.

        Returns:
            Processed AuditReport.
        """
        try:
            audit_type = AuditType(report_data.get("audit_type", "stage1"))
        except ValueError:
            audit_type = AuditType.STAGE1

        report = AuditReport(
            audit_type=audit_type,
            schedule_id=report_data.get("schedule_id", ""),
            certification_body_id=report_data.get("certification_body_id", ""),
            lead_auditor=report_data.get("lead_auditor", ""),
            audit_date=report_data.get("audit_date", ""),
            clauses_audited=report_data.get("clauses_audited", []),
            major_nonconformities=report_data.get("major_nonconformities", 0),
            minor_nonconformities=report_data.get("minor_nonconformities", 0),
            observations=report_data.get("observations", 0),
            positive_practices=report_data.get("positive_practices", 0),
            findings=report_data.get("findings", []),
            recommendation=report_data.get("recommendation", ""),
            corrective_action_deadline=report_data.get("corrective_action_deadline"),
            overall_assessment=report_data.get("overall_assessment", ""),
            energy_performance_improvement_noted=report_data.get(
                "energy_performance_improvement_noted", False
            ),
        )
        report.provenance_hash = _compute_hash(report)
        self._reports[report.report_id] = report

        self.logger.info(
            "Audit report received: type=%s, recommendation=%s, majors=%d, minors=%d",
            audit_type.value, report.recommendation,
            report.major_nonconformities, report.minor_nonconformities,
        )
        return report

    def track_certificate_status(self) -> List[Dict[str, Any]]:
        """Get status of all certificates.

        Returns:
            List of certificate status summaries.
        """
        return [
            {
                "certificate_id": cert.certificate_id,
                "certificate_number": cert.certificate_number,
                "organization": cert.organization_name,
                "status": cert.status.value,
                "issue_date": cert.issue_date,
                "expiry_date": cert.expiry_date,
                "next_surveillance": cert.next_surveillance_date,
                "next_recertification": cert.next_recertification_date,
            }
            for cert in self._certificates.values()
        ]

    def check_surveillance_due(self) -> Dict[str, Any]:
        """Check if any surveillance audits are due.

        Returns:
            Dict with surveillance audit status.
        """
        due_certificates: List[Dict[str, Any]] = []
        for cert in self._certificates.values():
            if cert.status == CertificateStatus.ACTIVE and cert.next_surveillance_date:
                due_certificates.append({
                    "certificate_id": cert.certificate_id,
                    "organization": cert.organization_name,
                    "next_surveillance_date": cert.next_surveillance_date,
                })

        return {
            "total_active_certificates": sum(
                1 for c in self._certificates.values()
                if c.status == CertificateStatus.ACTIVE
            ),
            "surveillance_due": len(due_certificates),
            "certificates": due_certificates,
        }

    def prepare_audit_package(
        self, audit_type: AuditType,
    ) -> Dict[str, Any]:
        """Prepare the document and evidence package for an audit.

        Args:
            audit_type: Type of upcoming audit.

        Returns:
            Dict with audit preparation checklist and status.
        """
        # Define required items per audit type
        audit_requirements: Dict[AuditType, List[str]] = {
            AuditType.STAGE1: [
                "energy_policy",
                "scope_and_boundaries",
                "energy_review_summary",
                "documented_information_index",
                "organization_chart",
                "enms_manual_or_procedures",
            ],
            AuditType.STAGE2: [
                "energy_policy",
                "scope_and_boundaries",
                "energy_review",
                "enpi_report",
                "baseline_documentation",
                "action_plans",
                "operational_control_procedures",
                "monitoring_measurement_plan",
                "internal_audit_report",
                "management_review_minutes",
                "corrective_action_records",
                "training_records",
            ],
            AuditType.SURVEILLANCE: [
                "enpi_performance_report",
                "internal_audit_report",
                "management_review_minutes",
                "corrective_action_status",
                "energy_performance_improvement_evidence",
                "continual_improvement_records",
            ],
            AuditType.RECERTIFICATION: [
                "energy_policy",
                "scope_and_boundaries",
                "energy_review",
                "enpi_report",
                "baseline_documentation",
                "action_plans",
                "operational_control_procedures",
                "monitoring_measurement_plan",
                "internal_audit_report",
                "management_review_minutes",
                "3_year_performance_trend",
                "continual_improvement_evidence",
            ],
        }

        required = audit_requirements.get(audit_type, [])

        return {
            "package_id": _new_uuid(),
            "audit_type": audit_type.value,
            "required_documents": required,
            "total_required": len(required),
            "prepared_at": utcnow().isoformat(),
            "status": "checklist_generated",
            "provenance_hash": _compute_hash({"type": audit_type.value, "docs": required}),
        }
