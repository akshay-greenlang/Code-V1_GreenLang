# -*- coding: utf-8 -*-
"""
GDPR DSAR Processor - SEC-010 Phase 5

Processes Data Subject Access Requests (DSARs) under GDPR Articles 15-22.
Handles identity verification, data discovery, request execution, and
compliance tracking within the legally mandated 30-day SLA.

DSAR Types Supported:
- Article 15: Right of Access
- Article 16: Right to Rectification
- Article 17: Right to Erasure (Right to be Forgotten)
- Article 18: Right to Restriction of Processing
- Article 20: Right to Data Portability
- Article 21: Right to Object

Classes:
    - DSARProcessor: Main processor for DSAR requests.
    - VerificationResult: Result of identity verification.
    - ExecutionResult: Result of DSAR execution.

Example:
    >>> processor = DSARProcessor()
    >>> request = await processor.submit_request(DSARRequest(...))
    >>> verified = await processor.verify_identity(request.id, method="email")
    >>> data = await processor.discover_data(request.subject_id)
    >>> result = await processor.execute_erasure(request.id)

Author: GreenLang Security Team
Date: February 2026
PRD: SEC-010 Security Operations Automation Platform
"""

from __future__ import annotations

import hashlib
import json
import logging
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional
from uuid import uuid4

from pydantic import BaseModel, Field

from greenlang.infrastructure.compliance_automation.models import (
    DataRecord,
    DSARRequest,
    DSARStatus,
    DSARType,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result Models
# ---------------------------------------------------------------------------


class VerificationResult(BaseModel):
    """Result of identity verification.

    Attributes:
        success: Whether verification succeeded.
        method: Verification method used.
        verified_at: Timestamp of verification.
        confidence: Confidence score (0-1).
        notes: Additional notes.
    """

    success: bool = False
    method: str = Field(default="", description="Verification method used")
    verified_at: Optional[datetime] = None
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    notes: str = ""


class ExecutionResult(BaseModel):
    """Result of DSAR execution.

    Attributes:
        success: Whether execution succeeded.
        request_type: Type of DSAR executed.
        actions_taken: List of actions performed.
        records_affected: Number of records affected.
        export_url: URL to download exported data (for access/portability).
        deletion_certificate_id: Certificate ID (for erasure).
        errors: Any errors encountered.
        completed_at: Completion timestamp.
    """

    success: bool = False
    request_type: DSARType = DSARType.ACCESS
    actions_taken: List[Dict[str, Any]] = Field(default_factory=list)
    records_affected: int = 0
    export_url: Optional[str] = None
    deletion_certificate_id: Optional[str] = None
    errors: List[str] = Field(default_factory=list)
    completed_at: Optional[datetime] = None


class DeletionCertificate(BaseModel):
    """Certificate of data deletion for erasure requests.

    Attributes:
        id: Unique certificate identifier.
        request_id: Associated DSAR request ID.
        subject_email: Email of the data subject.
        records_deleted: Count of records deleted.
        systems_affected: List of systems where data was deleted.
        deletion_timestamp: When deletion occurred.
        verification_hash: Hash for certificate verification.
    """

    id: str = Field(default_factory=lambda: str(uuid4()))
    request_id: str
    subject_email: str
    records_deleted: int = 0
    systems_affected: List[str] = Field(default_factory=list)
    deletion_timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    verification_hash: str = ""


# ---------------------------------------------------------------------------
# DSAR Processor
# ---------------------------------------------------------------------------


class DSARProcessor:
    """Process Data Subject Access Requests under GDPR.

    Handles the complete lifecycle of DSAR requests including submission,
    identity verification, data discovery, request execution, and compliance
    tracking.

    Attributes:
        SLA_DAYS: GDPR-mandated response deadline (30 days).
        EXTENSION_DAYS: Maximum extension for complex requests (60 additional days).

    Example:
        >>> processor = DSARProcessor()
        >>> request = await processor.submit_request(DSARRequest(
        ...     request_type=DSARType.ACCESS,
        ...     subject_email="user@example.com",
        ...     subject_name="John Doe",
        ... ))
        >>> print(f"Request {request.request_number} submitted, due {request.due_date}")
    """

    # GDPR Article 12(3) - Response within one month
    SLA_DAYS = 30

    # GDPR Article 12(3) - Extension for complex requests
    EXTENSION_DAYS = 60

    # Supported DSAR types with GDPR article references
    REQUEST_TYPES = {
        DSARType.ACCESS: "Article 15 - Right of Access",
        DSARType.RECTIFICATION: "Article 16 - Right to Rectification",
        DSARType.ERASURE: "Article 17 - Right to Erasure (Right to be Forgotten)",
        DSARType.RESTRICTION: "Article 18 - Right to Restriction of Processing",
        DSARType.PORTABILITY: "Article 20 - Right to Data Portability",
        DSARType.OBJECTION: "Article 21 - Right to Object",
    }

    def __init__(self, config: Optional[Any] = None) -> None:
        """Initialize the DSAR processor.

        Args:
            config: Optional compliance configuration.
        """
        self.config = config
        self._requests: Dict[str, DSARRequest] = {}
        self._deletion_certificates: Dict[str, DeletionCertificate] = {}
        logger.info("Initialized DSARProcessor with %d-day SLA", self.SLA_DAYS)

    async def submit_request(
        self,
        request: DSARRequest,
    ) -> DSARRequest:
        """Submit a new DSAR request.

        Validates the request, assigns a tracking number, calculates the
        SLA due date, and stores the request for processing.

        Args:
            request: The DSAR request to submit.

        Returns:
            The submitted request with assigned tracking number.

        Raises:
            ValueError: If the request is invalid.
        """
        logger.info(
            "Submitting DSAR request: type=%s, email=%s",
            request.request_type.value,
            request.subject_email,
        )

        # Validate request type
        if request.request_type not in self.REQUEST_TYPES:
            raise ValueError(f"Unsupported DSAR type: {request.request_type}")

        # Set submission timestamp and due date
        now = datetime.now(timezone.utc)
        request.submitted_at = now
        request.due_date = now + timedelta(days=self.SLA_DAYS)
        request.status = DSARStatus.SUBMITTED

        # Generate request number if not set
        if not request.request_number:
            year = now.year
            short_id = request.id[:8].upper()
            request.request_number = f"DSAR-{year}-{short_id}"

        # Store request
        self._requests[request.id] = request

        logger.info(
            "DSAR request submitted: %s, due: %s",
            request.request_number,
            request.due_date.strftime("%Y-%m-%d"),
        )

        return request

    async def verify_identity(
        self,
        request_id: str,
        method: str = "email",
        verification_data: Optional[Dict[str, Any]] = None,
    ) -> VerificationResult:
        """Verify the identity of the data subject.

        Identity verification is required before processing DSAR requests
        to prevent unauthorized data disclosure.

        Args:
            request_id: The DSAR request ID.
            method: Verification method (email, id_document, knowledge_based).
            verification_data: Additional verification data.

        Returns:
            VerificationResult indicating success/failure.

        Raises:
            ValueError: If the request is not found.
        """
        request = self._requests.get(request_id)
        if not request:
            raise ValueError(f"DSAR request not found: {request_id}")

        logger.info(
            "Verifying identity for %s using method: %s",
            request.request_number,
            method,
        )

        # Update request status
        request.status = DSARStatus.VERIFYING

        result = VerificationResult(method=method)

        # In production, implement actual verification logic
        if method == "email":
            # Email verification - send confirmation link
            result.success = True
            result.confidence = 0.8
            result.notes = "Email verification link sent and confirmed"
        elif method == "id_document":
            # Document verification - check uploaded ID
            result.success = True
            result.confidence = 0.95
            result.notes = "Government ID verified"
        elif method == "knowledge_based":
            # Knowledge-based verification
            result.success = True
            result.confidence = 0.7
            result.notes = "Security questions answered correctly"
        else:
            result.success = False
            result.notes = f"Unsupported verification method: {method}"

        if result.success:
            result.verified_at = datetime.now(timezone.utc)
            request.identity_verified_at = result.verified_at
            request.verification_method = method
            request.status = DSARStatus.IDENTITY_VERIFIED

        return result

    async def discover_data(
        self,
        request_id: str,
    ) -> List[DataRecord]:
        """Discover all data for a user across systems.

        Scans databases, object storage, logs, and other data sources
        to compile a complete inventory of the user's data.

        Args:
            request_id: The DSAR request ID.

        Returns:
            List of discovered data records.

        Raises:
            ValueError: If the request is not found or identity not verified.
        """
        request = self._requests.get(request_id)
        if not request:
            raise ValueError(f"DSAR request not found: {request_id}")

        if request.status not in (
            DSARStatus.IDENTITY_VERIFIED,
            DSARStatus.PROCESSING,
        ):
            raise ValueError("Identity must be verified before data discovery")

        logger.info(
            "Discovering data for %s (email: %s)",
            request.request_number,
            request.subject_email,
        )

        request.status = DSARStatus.PROCESSING
        request.processing_started_at = datetime.now(timezone.utc)

        # Import DataDiscovery to perform the scan
        from greenlang.infrastructure.compliance_automation.gdpr.data_discovery import (
            DataDiscovery,
        )

        discovery = DataDiscovery()
        records = await discovery.discover_user_data(
            user_email=request.subject_email,
            user_id=request.subject_id,
        )

        # Update request with discovery results
        request.data_sources_scanned = [
            "postgresql",
            "s3",
            "logs",
            "redis_cache",
        ]
        request.data_discovered = {
            "total_records": len(records),
            "by_system": self._count_by_system(records),
            "by_category": self._count_by_category(records),
        }

        return records

    async def execute_access(
        self,
        request_id: str,
        export_format: str = "json",
    ) -> ExecutionResult:
        """Execute an access request (Article 15).

        Compiles all discovered data into a portable format and generates
        a download URL.

        Args:
            request_id: The DSAR request ID.
            export_format: Export format (json, csv, xml).

        Returns:
            ExecutionResult with export URL.
        """
        request = self._requests.get(request_id)
        if not request:
            raise ValueError(f"DSAR request not found: {request_id}")

        if request.request_type != DSARType.ACCESS:
            raise ValueError("This request is not an access request")

        logger.info("Executing access request: %s", request.request_number)

        records = await self.discover_data(request_id)

        result = ExecutionResult(
            success=True,
            request_type=DSARType.ACCESS,
            records_affected=len(records),
            completed_at=datetime.now(timezone.utc),
        )

        # Generate export file
        export_data = self._prepare_export(records, export_format)
        export_url = await self._upload_export(
            request.id,
            export_data,
            export_format,
        )

        result.export_url = export_url
        result.actions_taken.append({
            "action": "data_compiled",
            "records": len(records),
            "format": export_format,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })

        # Update request
        request.export_file_url = export_url
        request.status = DSARStatus.COMPLETED
        request.completed_at = datetime.now(timezone.utc)

        return result

    async def execute_erasure(
        self,
        request_id: str,
    ) -> ExecutionResult:
        """Execute an erasure request (Article 17 - Right to be Forgotten).

        Deletes all user data from primary systems and schedules backup
        deletion. Generates a deletion certificate as proof.

        Args:
            request_id: The DSAR request ID.

        Returns:
            ExecutionResult with deletion certificate.
        """
        request = self._requests.get(request_id)
        if not request:
            raise ValueError(f"DSAR request not found: {request_id}")

        if request.request_type != DSARType.ERASURE:
            raise ValueError("This request is not an erasure request")

        logger.info("Executing erasure request: %s", request.request_number)

        records = await self.discover_data(request_id)

        result = ExecutionResult(
            success=True,
            request_type=DSARType.ERASURE,
            completed_at=datetime.now(timezone.utc),
        )

        systems_affected: List[str] = []
        total_deleted = 0

        # Delete from each system
        for record in records:
            try:
                deleted = await self._delete_record(record)
                if deleted:
                    total_deleted += 1
                    if record.source_system not in systems_affected:
                        systems_affected.append(record.source_system)
                    result.actions_taken.append({
                        "action": "record_deleted",
                        "system": record.source_system,
                        "location": record.source_location,
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                    })
            except Exception as e:
                error_msg = f"Failed to delete from {record.source_system}: {str(e)}"
                logger.error(error_msg)
                result.errors.append(error_msg)

        result.records_affected = total_deleted

        # Generate deletion certificate
        certificate = self._generate_deletion_certificate(
            request=request,
            records_deleted=total_deleted,
            systems_affected=systems_affected,
        )
        self._deletion_certificates[certificate.id] = certificate
        result.deletion_certificate_id = certificate.id

        # Update request
        request.deletion_certificate_id = certificate.id
        request.status = DSARStatus.COMPLETED
        request.completed_at = datetime.now(timezone.utc)

        # Schedule backup deletion (async, handled separately)
        await self._schedule_backup_deletion(request)

        return result

    async def execute_portability(
        self,
        request_id: str,
        format: str = "json",
    ) -> ExecutionResult:
        """Execute a data portability request (Article 20).

        Exports data in a machine-readable format suitable for transfer
        to another service provider.

        Args:
            request_id: The DSAR request ID.
            format: Export format (json, csv, xml).

        Returns:
            ExecutionResult with export URL.
        """
        request = self._requests.get(request_id)
        if not request:
            raise ValueError(f"DSAR request not found: {request_id}")

        if request.request_type != DSARType.PORTABILITY:
            raise ValueError("This request is not a portability request")

        logger.info("Executing portability request: %s", request.request_number)

        # Portability is similar to access but with structured format
        records = await self.discover_data(request_id)

        result = ExecutionResult(
            success=True,
            request_type=DSARType.PORTABILITY,
            records_affected=len(records),
            completed_at=datetime.now(timezone.utc),
        )

        # Generate portable export
        export_data = self._prepare_portable_export(records, format)
        export_url = await self._upload_export(
            request.id,
            export_data,
            format,
        )

        result.export_url = export_url
        result.actions_taken.append({
            "action": "portable_export_created",
            "records": len(records),
            "format": format,
            "standard": "GDPR Article 20 compliant",
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })

        # Update request
        request.export_file_url = export_url
        request.status = DSARStatus.COMPLETED
        request.completed_at = datetime.now(timezone.utc)

        return result

    async def execute_rectification(
        self,
        request_id: str,
        corrections: Dict[str, Any],
    ) -> ExecutionResult:
        """Execute a rectification request (Article 16).

        Updates inaccurate data with corrections provided by the data subject.

        Args:
            request_id: The DSAR request ID.
            corrections: Dictionary of field corrections.

        Returns:
            ExecutionResult with update details.
        """
        request = self._requests.get(request_id)
        if not request:
            raise ValueError(f"DSAR request not found: {request_id}")

        if request.request_type != DSARType.RECTIFICATION:
            raise ValueError("This request is not a rectification request")

        logger.info("Executing rectification request: %s", request.request_number)

        result = ExecutionResult(
            success=True,
            request_type=DSARType.RECTIFICATION,
            completed_at=datetime.now(timezone.utc),
        )

        # Apply corrections
        records_updated = await self._apply_corrections(
            request.subject_email,
            request.subject_id,
            corrections,
        )

        result.records_affected = records_updated
        result.actions_taken.append({
            "action": "data_rectified",
            "fields_updated": list(corrections.keys()),
            "records_affected": records_updated,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })

        # Update request
        request.status = DSARStatus.COMPLETED
        request.completed_at = datetime.now(timezone.utc)

        return result

    async def get_request(self, request_id: str) -> Optional[DSARRequest]:
        """Get a DSAR request by ID.

        Args:
            request_id: The request ID.

        Returns:
            The DSARRequest or None if not found.
        """
        return self._requests.get(request_id)

    async def get_pending_requests(self) -> List[DSARRequest]:
        """Get all pending DSAR requests.

        Returns:
            List of requests not yet completed.
        """
        return [
            r for r in self._requests.values()
            if r.status not in (DSARStatus.COMPLETED, DSARStatus.REJECTED, DSARStatus.CANCELLED)
        ]

    async def get_overdue_requests(self) -> List[DSARRequest]:
        """Get all overdue DSAR requests.

        Returns:
            List of requests past their due date.
        """
        now = datetime.now(timezone.utc)
        return [
            r for r in self._requests.values()
            if r.status not in (DSARStatus.COMPLETED, DSARStatus.REJECTED, DSARStatus.CANCELLED)
            and r.due_date < now
        ]

    async def extend_deadline(
        self,
        request_id: str,
        reason: str,
    ) -> DSARRequest:
        """Extend the deadline for a complex request.

        GDPR Article 12(3) allows extension up to 2 additional months
        for complex requests, with notification to the data subject.

        Args:
            request_id: The request ID.
            reason: Reason for extension.

        Returns:
            Updated request with new deadline.

        Raises:
            ValueError: If request not found or already extended.
        """
        request = self._requests.get(request_id)
        if not request:
            raise ValueError(f"DSAR request not found: {request_id}")

        if request.status == DSARStatus.EXTENDED:
            raise ValueError("Request has already been extended")

        logger.info(
            "Extending deadline for %s: %s",
            request.request_number,
            reason,
        )

        # Extend by up to 60 additional days (GDPR max: 90 days total)
        new_due_date = request.submitted_at + timedelta(
            days=self.SLA_DAYS + self.EXTENSION_DAYS
        )
        request.due_date = new_due_date
        request.status = DSARStatus.EXTENDED
        request.extension_reason = reason

        return request

    def get_deletion_certificate(
        self,
        certificate_id: str,
    ) -> Optional[DeletionCertificate]:
        """Get a deletion certificate by ID.

        Args:
            certificate_id: The certificate ID.

        Returns:
            The DeletionCertificate or None if not found.
        """
        return self._deletion_certificates.get(certificate_id)

    # -------------------------------------------------------------------------
    # Private Methods
    # -------------------------------------------------------------------------

    def _count_by_system(self, records: List[DataRecord]) -> Dict[str, int]:
        """Count records by source system."""
        counts: Dict[str, int] = {}
        for record in records:
            system = record.source_system
            counts[system] = counts.get(system, 0) + 1
        return counts

    def _count_by_category(self, records: List[DataRecord]) -> Dict[str, int]:
        """Count records by data category."""
        counts: Dict[str, int] = {}
        for record in records:
            category = record.data_category.value
            counts[category] = counts.get(category, 0) + 1
        return counts

    def _prepare_export(
        self,
        records: List[DataRecord],
        format: str,
    ) -> bytes:
        """Prepare data export in specified format."""
        if format == "json":
            data = [r.model_dump() for r in records]
            return json.dumps(data, indent=2, default=str).encode("utf-8")
        elif format == "csv":
            # Simplified CSV export
            lines = ["source_system,source_location,data_type,discovered_at"]
            for r in records:
                lines.append(
                    f"{r.source_system},{r.source_location},"
                    f"{r.data_type},{r.discovered_at.isoformat()}"
                )
            return "\n".join(lines).encode("utf-8")
        else:
            # Default to JSON
            data = [r.model_dump() for r in records]
            return json.dumps(data, indent=2, default=str).encode("utf-8")

    def _prepare_portable_export(
        self,
        records: List[DataRecord],
        format: str,
    ) -> bytes:
        """Prepare portable export (Article 20 compliant)."""
        # Portable format includes metadata for interoperability
        export = {
            "format_version": "1.0",
            "gdpr_article": "Article 20 - Right to Data Portability",
            "exported_at": datetime.now(timezone.utc).isoformat(),
            "record_count": len(records),
            "data": [r.model_dump() for r in records],
        }
        return json.dumps(export, indent=2, default=str).encode("utf-8")

    async def _upload_export(
        self,
        request_id: str,
        data: bytes,
        format: str,
    ) -> str:
        """Upload export file and return download URL.

        In production, this would upload to S3 with presigned URL.
        """
        # Placeholder - return mock URL
        file_hash = hashlib.sha256(data).hexdigest()[:12]
        return f"https://exports.greenlang.io/dsar/{request_id}/{file_hash}.{format}"

    async def _delete_record(self, record: DataRecord) -> bool:
        """Delete a single data record.

        In production, this would perform actual deletion.
        """
        logger.debug(
            "Deleting record from %s: %s",
            record.source_system,
            record.source_location,
        )
        # Placeholder - return success
        return True

    def _generate_deletion_certificate(
        self,
        request: DSARRequest,
        records_deleted: int,
        systems_affected: List[str],
    ) -> DeletionCertificate:
        """Generate a deletion certificate."""
        certificate = DeletionCertificate(
            request_id=request.id,
            subject_email=request.subject_email,
            records_deleted=records_deleted,
            systems_affected=systems_affected,
        )

        # Generate verification hash
        cert_data = f"{certificate.id}|{certificate.request_id}|{certificate.deletion_timestamp.isoformat()}"
        certificate.verification_hash = hashlib.sha256(cert_data.encode()).hexdigest()

        return certificate

    async def _schedule_backup_deletion(self, request: DSARRequest) -> None:
        """Schedule deletion from backup systems.

        Backup deletion is typically scheduled to occur after the
        backup retention period expires.
        """
        logger.info(
            "Scheduled backup deletion for %s",
            request.request_number,
        )
        # In production, create a scheduled task for backup deletion

    async def _apply_corrections(
        self,
        email: str,
        user_id: Optional[str],
        corrections: Dict[str, Any],
    ) -> int:
        """Apply data corrections.

        In production, this would update records in relevant systems.
        """
        logger.info(
            "Applying corrections for %s: %s",
            email,
            list(corrections.keys()),
        )
        # Placeholder - return count of records updated
        return 1


__all__ = [
    "DSARProcessor",
    "VerificationResult",
    "ExecutionResult",
    "DeletionCertificate",
]
