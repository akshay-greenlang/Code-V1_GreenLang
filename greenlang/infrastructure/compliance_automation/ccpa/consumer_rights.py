# -*- coding: utf-8 -*-
"""
CCPA/LGPD Consumer Rights Processor - SEC-010 Phase 5

Processes consumer privacy rights requests under the California Consumer
Privacy Act (CCPA) and Brazil's Lei Geral de Protecao de Dados (LGPD).

CCPA Requirements:
- Respond within 45 days (can extend to 90)
- Verify consumer identity
- Free of charge (2 requests per 12-month period)
- Cannot discriminate against consumers exercising rights

LGPD Requirements:
- Respond within 15 days (simple requests)
- Free of charge
- Appoint Data Protection Officer (DPO)

Classes:
    - ConsumerRightsProcessor: Main processor for consumer rights requests.
    - ConsumerRequest: Request from a consumer.
    - ConsumerRequestResult: Result of processing a request.
    - PrivacyRegulation: Enum of supported privacy regulations.

Example:
    >>> processor = ConsumerRightsProcessor()
    >>> result = await processor.process_access_request(
    ...     email="consumer@example.com",
    ...     request_type="access",
    ... )

Author: GreenLang Security Team
Date: February 2026
PRD: SEC-010 Security Operations Automation Platform
"""

from __future__ import annotations

import hashlib
import json
import logging
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any, Dict, List, Optional
from uuid import uuid4

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------


class PrivacyRegulation(str, Enum):
    """Supported privacy regulations."""

    CCPA = "ccpa"
    """California Consumer Privacy Act."""

    LGPD = "lgpd"
    """Brazil's Lei Geral de Protecao de Dados."""


class CCPARequestType(str, Enum):
    """Types of CCPA consumer requests."""

    ACCESS = "access"
    """Right to Know - What data is collected."""

    CATEGORIES = "categories"
    """Right to Know - Categories of data."""

    DELETE = "delete"
    """Right to Delete - Request deletion."""

    OPT_OUT_SALE = "opt_out_sale"
    """Right to Opt-Out of Sale."""

    OPT_IN_SALE = "opt_in_sale"
    """Minor's Right to Opt-In to Sale."""


class LGPDRequestType(str, Enum):
    """Types of LGPD consumer requests."""

    CONFIRMATION = "confirmation"
    """Confirmation of data processing."""

    ACCESS = "access"
    """Access to personal data."""

    CORRECTION = "correction"
    """Correction of incomplete/inaccurate data."""

    ANONYMIZATION = "anonymization"
    """Anonymization, blocking, or elimination."""

    PORTABILITY = "portability"
    """Data portability."""

    ELIMINATION = "elimination"
    """Deletion of personal data."""

    SHARING_INFO = "sharing_info"
    """Information about third-party sharing."""


class RequestStatus(str, Enum):
    """Status of a consumer rights request."""

    SUBMITTED = "submitted"
    VERIFYING = "verifying"
    VERIFIED = "verified"
    PROCESSING = "processing"
    COMPLETED = "completed"
    DENIED = "denied"
    EXTENDED = "extended"


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


class ConsumerRequest(BaseModel):
    """A consumer privacy rights request.

    Attributes:
        id: Unique request identifier.
        request_number: Human-readable request number.
        regulation: The privacy regulation (CCPA or LGPD).
        request_type: Type of request.
        consumer_email: Consumer's email address.
        consumer_name: Consumer's name.
        consumer_id: Internal consumer/user ID if known.
        status: Current request status.
        submitted_at: When request was submitted.
        due_date: Response deadline.
        verified_at: When identity was verified.
        completed_at: When request was completed.
        is_california_resident: Whether consumer is CA resident (CCPA).
        is_brazil_resident: Whether consumer is Brazil resident (LGPD).
        data_discovered: Data found for the consumer.
        actions_taken: Actions performed.
        export_url: URL to download exported data.
        notes: Internal notes.
    """

    id: str = Field(default_factory=lambda: str(uuid4()))
    request_number: str = ""
    regulation: PrivacyRegulation = PrivacyRegulation.CCPA
    request_type: str
    consumer_email: str
    consumer_name: str = ""
    consumer_id: Optional[str] = None
    status: RequestStatus = RequestStatus.SUBMITTED
    submitted_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    due_date: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc) + timedelta(days=45)
    )
    verified_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    is_california_resident: bool = True
    is_brazil_resident: bool = False
    data_discovered: Dict[str, Any] = Field(default_factory=dict)
    actions_taken: List[Dict[str, Any]] = Field(default_factory=list)
    export_url: Optional[str] = None
    notes: str = ""


class ConsumerRequestResult(BaseModel):
    """Result of processing a consumer rights request.

    Attributes:
        request_id: The request ID.
        success: Whether processing succeeded.
        regulation: The privacy regulation.
        request_type: Type of request processed.
        data_provided: Data provided to consumer (for access requests).
        records_deleted: Number of records deleted (for delete requests).
        opt_out_applied: Whether opt-out was applied.
        export_url: URL to download data.
        errors: Any errors encountered.
        completed_at: Completion timestamp.
    """

    request_id: str
    success: bool = False
    regulation: PrivacyRegulation
    request_type: str
    data_provided: Optional[Dict[str, Any]] = None
    records_deleted: int = 0
    opt_out_applied: bool = False
    export_url: Optional[str] = None
    errors: List[str] = Field(default_factory=list)
    completed_at: Optional[datetime] = None


# ---------------------------------------------------------------------------
# Consumer Rights Processor
# ---------------------------------------------------------------------------


class ConsumerRightsProcessor:
    """Processes consumer privacy rights requests under CCPA and LGPD.

    Handles the complete lifecycle of consumer privacy requests including
    submission, verification, processing, and response delivery.

    Attributes:
        CCPA_SLA_DAYS: CCPA response deadline (45 days).
        CCPA_EXTENSION_DAYS: CCPA extension (additional 45 days).
        LGPD_SLA_DAYS: LGPD response deadline (15 days).

    Example:
        >>> processor = ConsumerRightsProcessor()
        >>> request = await processor.submit_request(
        ...     email="consumer@example.com",
        ...     request_type=CCPARequestType.ACCESS,
        ...     regulation=PrivacyRegulation.CCPA,
        ... )
    """

    # CCPA: 45 days, can extend to 90 total
    CCPA_SLA_DAYS = 45
    CCPA_EXTENSION_DAYS = 45

    # LGPD: 15 days for simple requests
    LGPD_SLA_DAYS = 15

    # CCPA data categories required in disclosure
    CCPA_DATA_CATEGORIES = [
        "identifiers",
        "personal_information",
        "protected_classifications",
        "commercial_information",
        "biometric_information",
        "internet_activity",
        "geolocation_data",
        "sensory_data",
        "professional_information",
        "education_information",
        "inferences",
    ]

    def __init__(self, config: Optional[Any] = None) -> None:
        """Initialize the consumer rights processor.

        Args:
            config: Optional compliance configuration.
        """
        self.config = config
        self._requests: Dict[str, ConsumerRequest] = {}
        self._opt_out_registry: Dict[str, bool] = {}

        logger.info("Initialized ConsumerRightsProcessor")

    async def submit_request(
        self,
        email: str,
        request_type: str,
        regulation: PrivacyRegulation = PrivacyRegulation.CCPA,
        name: str = "",
        is_resident: bool = True,
    ) -> ConsumerRequest:
        """Submit a new consumer rights request.

        Args:
            email: Consumer's email address.
            request_type: Type of request (access, delete, opt_out_sale, etc.).
            regulation: The privacy regulation (CCPA or LGPD).
            name: Consumer's name.
            is_resident: Whether consumer is a resident (CA for CCPA, Brazil for LGPD).

        Returns:
            The submitted ConsumerRequest.
        """
        logger.info(
            "Submitting %s request: email=%s, type=%s",
            regulation.value,
            email,
            request_type,
        )

        # Determine SLA based on regulation
        if regulation == PrivacyRegulation.CCPA:
            sla_days = self.CCPA_SLA_DAYS
        else:
            sla_days = self.LGPD_SLA_DAYS

        now = datetime.now(timezone.utc)

        request = ConsumerRequest(
            regulation=regulation,
            request_type=request_type,
            consumer_email=email.lower().strip(),
            consumer_name=name,
            submitted_at=now,
            due_date=now + timedelta(days=sla_days),
            is_california_resident=is_resident if regulation == PrivacyRegulation.CCPA else False,
            is_brazil_resident=is_resident if regulation == PrivacyRegulation.LGPD else False,
        )

        # Generate request number
        year = now.year
        prefix = "CCPA" if regulation == PrivacyRegulation.CCPA else "LGPD"
        short_id = request.id[:8].upper()
        request.request_number = f"{prefix}-{year}-{short_id}"

        self._requests[request.id] = request

        logger.info(
            "Request submitted: %s, due: %s",
            request.request_number,
            request.due_date.strftime("%Y-%m-%d"),
        )

        return request

    async def verify_identity(
        self,
        request_id: str,
        verification_data: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Verify consumer identity.

        CCPA requires businesses to verify the identity of consumers
        making requests before fulfilling them.

        Args:
            request_id: The request ID.
            verification_data: Data for verification.

        Returns:
            True if verification succeeded.
        """
        request = self._requests.get(request_id)
        if not request:
            raise ValueError(f"Request not found: {request_id}")

        request.status = RequestStatus.VERIFYING

        # In production, implement actual verification
        # (email verification, knowledge-based questions, ID document)

        request.status = RequestStatus.VERIFIED
        request.verified_at = datetime.now(timezone.utc)

        logger.info("Identity verified for request %s", request.request_number)

        return True

    async def verify_california_residence(
        self,
        request_id: str,
    ) -> bool:
        """Verify California residence for CCPA requests.

        Args:
            request_id: The request ID.

        Returns:
            True if California residence is verified.
        """
        request = self._requests.get(request_id)
        if not request:
            raise ValueError(f"Request not found: {request_id}")

        if request.regulation != PrivacyRegulation.CCPA:
            return True  # Not applicable

        # In production, verify residence through:
        # - IP geolocation
        # - Billing address
        # - Self-attestation with verification

        request.is_california_resident = True
        return True

    async def process_access_request(
        self,
        request_id: str,
    ) -> ConsumerRequestResult:
        """Process a Right to Know / Access request.

        Compiles and returns all personal information collected about
        the consumer.

        Args:
            request_id: The request ID.

        Returns:
            ConsumerRequestResult with discovered data.
        """
        request = self._requests.get(request_id)
        if not request:
            raise ValueError(f"Request not found: {request_id}")

        logger.info("Processing access request: %s", request.request_number)

        request.status = RequestStatus.PROCESSING

        result = ConsumerRequestResult(
            request_id=request_id,
            regulation=request.regulation,
            request_type=request.request_type,
        )

        # Discover consumer data
        from greenlang.infrastructure.compliance_automation.gdpr.data_discovery import (
            DataDiscovery,
        )

        discovery = DataDiscovery()
        records = await discovery.discover_user_data(
            user_email=request.consumer_email,
            user_id=request.consumer_id,
        )

        # Format data according to CCPA categories
        if request.regulation == PrivacyRegulation.CCPA:
            data_by_category = self._categorize_data_ccpa(records)
        else:
            data_by_category = self._format_data_lgpd(records)

        request.data_discovered = data_by_category

        # Generate export
        export_data = self._prepare_access_response(request, data_by_category)
        export_url = await self._upload_export(request.id, export_data)

        request.export_url = export_url
        request.status = RequestStatus.COMPLETED
        request.completed_at = datetime.now(timezone.utc)
        request.actions_taken.append({
            "action": "data_compiled",
            "categories": list(data_by_category.keys()),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })

        result.success = True
        result.data_provided = data_by_category
        result.export_url = export_url
        result.completed_at = datetime.now(timezone.utc)

        return result

    async def process_deletion_request(
        self,
        request_id: str,
    ) -> ConsumerRequestResult:
        """Process a Right to Delete request.

        Deletes personal information with certain exceptions
        (legal obligations, contracts, etc.).

        Args:
            request_id: The request ID.

        Returns:
            ConsumerRequestResult with deletion details.
        """
        request = self._requests.get(request_id)
        if not request:
            raise ValueError(f"Request not found: {request_id}")

        logger.info("Processing deletion request: %s", request.request_number)

        request.status = RequestStatus.PROCESSING

        result = ConsumerRequestResult(
            request_id=request_id,
            regulation=request.regulation,
            request_type=request.request_type,
        )

        # CCPA exceptions to deletion
        exceptions = self._check_deletion_exceptions(request)
        if exceptions:
            result.errors = exceptions
            logger.info(
                "Deletion exceptions apply for %s: %s",
                request.request_number,
                exceptions,
            )

        # Perform deletion (excluding exceptions)
        deleted_count = await self._delete_consumer_data(
            request.consumer_email,
            request.consumer_id,
            exceptions,
        )

        request.status = RequestStatus.COMPLETED
        request.completed_at = datetime.now(timezone.utc)
        request.actions_taken.append({
            "action": "data_deleted",
            "records_deleted": deleted_count,
            "exceptions_applied": exceptions,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })

        result.success = True
        result.records_deleted = deleted_count
        result.completed_at = datetime.now(timezone.utc)

        return result

    async def process_opt_out(
        self,
        request_id: str,
    ) -> ConsumerRequestResult:
        """Process an Opt-Out of Sale request.

        Records the consumer's preference not to have their personal
        information sold.

        Args:
            request_id: The request ID.

        Returns:
            ConsumerRequestResult confirming opt-out.
        """
        request = self._requests.get(request_id)
        if not request:
            raise ValueError(f"Request not found: {request_id}")

        logger.info("Processing opt-out request: %s", request.request_number)

        request.status = RequestStatus.PROCESSING

        result = ConsumerRequestResult(
            request_id=request_id,
            regulation=request.regulation,
            request_type=request.request_type,
        )

        # Record opt-out preference
        self._opt_out_registry[request.consumer_email] = True

        # Propagate to data partners
        await self._notify_data_partners_opt_out(request.consumer_email)

        request.status = RequestStatus.COMPLETED
        request.completed_at = datetime.now(timezone.utc)
        request.actions_taken.append({
            "action": "opt_out_recorded",
            "effective_date": datetime.now(timezone.utc).isoformat(),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })

        result.success = True
        result.opt_out_applied = True
        result.completed_at = datetime.now(timezone.utc)

        return result

    async def check_opt_out_status(self, email: str) -> bool:
        """Check if a consumer has opted out of sale.

        Args:
            email: Consumer's email address.

        Returns:
            True if consumer has opted out.
        """
        return self._opt_out_registry.get(email.lower().strip(), False)

    async def get_request(self, request_id: str) -> Optional[ConsumerRequest]:
        """Get a request by ID.

        Args:
            request_id: The request ID.

        Returns:
            The ConsumerRequest or None.
        """
        return self._requests.get(request_id)

    async def get_pending_requests(
        self,
        regulation: Optional[PrivacyRegulation] = None,
    ) -> List[ConsumerRequest]:
        """Get all pending requests.

        Args:
            regulation: Optional filter by regulation.

        Returns:
            List of pending requests.
        """
        pending = [
            r for r in self._requests.values()
            if r.status not in (RequestStatus.COMPLETED, RequestStatus.DENIED)
        ]

        if regulation:
            pending = [r for r in pending if r.regulation == regulation]

        return pending

    async def extend_deadline(
        self,
        request_id: str,
        reason: str,
    ) -> ConsumerRequest:
        """Extend the response deadline (CCPA allows one 45-day extension).

        Args:
            request_id: The request ID.
            reason: Reason for extension.

        Returns:
            Updated request with new deadline.
        """
        request = self._requests.get(request_id)
        if not request:
            raise ValueError(f"Request not found: {request_id}")

        if request.regulation != PrivacyRegulation.CCPA:
            raise ValueError("Extensions only apply to CCPA requests")

        if request.status == RequestStatus.EXTENDED:
            raise ValueError("Request has already been extended")

        # Extend by 45 days (CCPA max: 90 days total)
        new_due_date = request.submitted_at + timedelta(
            days=self.CCPA_SLA_DAYS + self.CCPA_EXTENSION_DAYS
        )
        request.due_date = new_due_date
        request.status = RequestStatus.EXTENDED
        request.notes = f"Extended: {reason}"

        logger.info(
            "Extended request %s to %s: %s",
            request.request_number,
            new_due_date.strftime("%Y-%m-%d"),
            reason,
        )

        return request

    # -------------------------------------------------------------------------
    # Private Methods
    # -------------------------------------------------------------------------

    def _categorize_data_ccpa(
        self,
        records: List[Any],
    ) -> Dict[str, Any]:
        """Categorize discovered data according to CCPA categories."""
        categorized: Dict[str, List] = {cat: [] for cat in self.CCPA_DATA_CATEGORIES}

        for record in records:
            # Map data to CCPA categories
            data_type = getattr(record, "data_type", "unknown")
            record_data = getattr(record, "record_data", {})

            if "email" in str(record_data) or "name" in str(record_data):
                categorized["identifiers"].append(record_data)

            if "address" in str(record_data) or "phone" in str(record_data):
                categorized["personal_information"].append(record_data)

            if "purchase" in data_type or "transaction" in data_type:
                categorized["commercial_information"].append(record_data)

            if "ip_address" in str(record_data) or "session" in data_type:
                categorized["internet_activity"].append(record_data)

            if "location" in str(record_data):
                categorized["geolocation_data"].append(record_data)

        # Remove empty categories
        return {k: v for k, v in categorized.items() if v}

    def _format_data_lgpd(
        self,
        records: List[Any],
    ) -> Dict[str, Any]:
        """Format discovered data for LGPD response."""
        formatted = {
            "dados_pessoais": [],  # Personal data
            "dados_sensiveis": [],  # Sensitive data
            "finalidade": [],  # Processing purposes
            "compartilhamento": [],  # Third-party sharing
        }

        for record in records:
            record_data = getattr(record, "record_data", {})
            category = getattr(record, "data_category", None)

            if category and category.value == "sensitive_pii":
                formatted["dados_sensiveis"].append(record_data)
            else:
                formatted["dados_pessoais"].append(record_data)

        # Remove empty sections
        return {k: v for k, v in formatted.items() if v}

    def _prepare_access_response(
        self,
        request: ConsumerRequest,
        data_by_category: Dict[str, Any],
    ) -> bytes:
        """Prepare access response for download."""
        response = {
            "request_number": request.request_number,
            "regulation": request.regulation.value,
            "consumer_email": request.consumer_email,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "data_categories": data_by_category,
        }

        if request.regulation == PrivacyRegulation.CCPA:
            response["disclosure"] = {
                "categories_collected": list(data_by_category.keys()),
                "purposes": ["Service provision", "Analytics", "Communication"],
                "sources": ["Direct collection", "Automatic collection"],
                "third_parties": ["Service providers", "Analytics partners"],
            }

        return json.dumps(response, indent=2, default=str).encode("utf-8")

    async def _upload_export(
        self,
        request_id: str,
        data: bytes,
    ) -> str:
        """Upload export file and return URL."""
        file_hash = hashlib.sha256(data).hexdigest()[:12]
        return f"https://privacy.greenlang.io/exports/{request_id}/{file_hash}.json"

    def _check_deletion_exceptions(
        self,
        request: ConsumerRequest,
    ) -> List[str]:
        """Check for CCPA deletion exceptions."""
        exceptions: List[str] = []

        # CCPA 1798.105(d) exceptions
        # In production, check if exceptions apply

        # Example: Active contract
        # if has_active_subscription(request.consumer_email):
        #     exceptions.append("Complete the transaction (active subscription)")

        # Example: Legal obligation
        # if has_regulatory_retention_requirement(request.consumer_email):
        #     exceptions.append("Comply with legal obligation")

        return exceptions

    async def _delete_consumer_data(
        self,
        email: str,
        user_id: Optional[str],
        exceptions: List[str],
    ) -> int:
        """Delete consumer data, respecting exceptions.

        In production, this would perform actual deletions.
        """
        logger.info(
            "Deleting consumer data: email=%s, exceptions=%s",
            email,
            exceptions,
        )
        # Placeholder - return simulated count
        return 50

    async def _notify_data_partners_opt_out(self, email: str) -> None:
        """Notify data partners of opt-out preference.

        In production, this would call partner APIs.
        """
        logger.info("Notifying data partners of opt-out: %s", email)


__all__ = [
    "ConsumerRightsProcessor",
    "ConsumerRequest",
    "ConsumerRequestResult",
    "PrivacyRegulation",
    "CCPARequestType",
    "LGPDRequestType",
    "RequestStatus",
]
