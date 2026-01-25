# -*- coding: utf-8 -*-
"""
GL-010 EmissionsGuardian - Offset Certificate Tracker

Carbon offset certificate lifecycle management.

Author: GreenLang GL-010 EmissionsGuardian
"""

from datetime import datetime, date
from decimal import Decimal
from typing import Any, Dict, List, Optional
import hashlib
import logging

from .schemas import (
    OffsetCertificate, OffsetStandard, OffsetProjectType,
    RetirementStatus, RetirementResult, VerificationResult
)

logger = logging.getLogger(__name__)


class RetirementWorkflow:
    """Offset retirement workflow management."""

    def __init__(self):
        self._pending: Dict[str, Dict[str, Any]] = {}

    def initiate(
        self,
        certificate: OffsetCertificate,
        beneficiary: str,
        reason: str
    ) -> str:
        """Initiate retirement workflow."""
        workflow_id = f"RET-{datetime.utcnow().strftime('%Y%m%d%H%M%S')}"

        self._pending[workflow_id] = {
            "certificate_id": certificate.certificate_id,
            "beneficiary": beneficiary,
            "reason": reason,
            "status": "pending_approval",
            "initiated_at": datetime.utcnow(),
        }

        return workflow_id

    def approve(self, workflow_id: str, approver: str) -> bool:
        """Approve retirement."""
        if workflow_id not in self._pending:
            return False

        self._pending[workflow_id]["status"] = "approved"
        self._pending[workflow_id]["approved_by"] = approver
        self._pending[workflow_id]["approved_at"] = datetime.utcnow()
        return True


class OffsetTracker:
    """
    Offset Certificate Tracker.

    Manages offset certificate lifecycle including:
    - Acquisition and registration
    - Verification and validation
    - Retirement with audit trail
    """

    def __init__(self):
        self._certificates: Dict[str, OffsetCertificate] = {}
        self._workflow = RetirementWorkflow()
        self._counter = 0
        logger.info("OffsetTracker initialized")

    def register(self, certificate: OffsetCertificate) -> None:
        """Register an offset certificate."""
        self._certificates[certificate.certificate_id] = certificate
        logger.info(f"Registered certificate: {certificate.certificate_id}")

    def get_certificate(
        self,
        certificate_id: str
    ) -> Optional[OffsetCertificate]:
        """Get certificate by ID."""
        return self._certificates.get(certificate_id)

    def get_certificates(
        self,
        facility_id: Optional[str] = None,
        status: Optional[RetirementStatus] = None,
        standard: Optional[OffsetStandard] = None
    ) -> List[OffsetCertificate]:
        """Get certificates with filters."""
        certs = list(self._certificates.values())

        if facility_id:
            certs = [c for c in certs if c.facility_id == facility_id]

        if status:
            certs = [c for c in certs if c.status == status]

        if standard:
            certs = [c for c in certs if c.standard == standard]

        return certs

    def verify(
        self,
        certificate_id: str
    ) -> VerificationResult:
        """Verify certificate authenticity."""
        cert = self._certificates.get(certificate_id)

        if not cert:
            return VerificationResult(
                certificate_id=certificate_id,
                is_valid=False,
                verification_date=datetime.utcnow(),
                registry_status="not_found",
                issues=["Certificate not found in registry"]
            )

        # Simulated verification
        issues = []
        is_valid = True

        if cert.status == RetirementStatus.RETIRED:
            issues.append("Certificate already retired")
            is_valid = False

        if cert.status == RetirementStatus.CANCELLED:
            issues.append("Certificate cancelled")
            is_valid = False

        return VerificationResult(
            certificate_id=certificate_id,
            is_valid=is_valid,
            verification_date=datetime.utcnow(),
            registry_status=cert.status.value,
            issues=issues
        )

    def retire(
        self,
        certificate_id: str,
        beneficiary: str,
        reason: str = "Compliance"
    ) -> Optional[RetirementResult]:
        """Retire an offset certificate."""
        cert = self._certificates.get(certificate_id)

        if not cert:
            return None

        if cert.status != RetirementStatus.ACTIVE:
            logger.warning(f"Cannot retire certificate in status: {cert.status}")
            return None

        # Update certificate
        cert.status = RetirementStatus.RETIRED
        cert.retirement_date = date.today()
        cert.retirement_reason = reason
        cert.beneficiary = beneficiary

        # Calculate provenance
        content = f"{certificate_id}|{cert.quantity}|{date.today()}"
        provenance_hash = hashlib.sha256(content.encode()).hexdigest()

        return RetirementResult(
            certificate_id=certificate_id,
            serial_number=cert.serial_number,
            quantity_retired=cert.quantity,
            retirement_date=date.today(),
            beneficiary=beneficiary,
            registry_confirmation=f"CONF-{datetime.utcnow().strftime('%Y%m%d%H%M%S')}",
            provenance_hash=provenance_hash
        )

    def get_available_quantity(
        self,
        facility_id: str,
        standard: Optional[OffsetStandard] = None
    ) -> Decimal:
        """Get total available (unretired) offset quantity."""
        certs = self.get_certificates(
            facility_id=facility_id,
            status=RetirementStatus.ACTIVE,
            standard=standard
        )
        return sum(c.quantity for c in certs)


__all__ = [
    "OffsetTracker",
    "RetirementWorkflow",
]
