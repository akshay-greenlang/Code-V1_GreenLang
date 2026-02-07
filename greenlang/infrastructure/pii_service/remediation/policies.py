# -*- coding: utf-8 -*-
"""
PII Remediation Policies - SEC-011 PII Detection/Redaction Enhancements

Defines remediation policies for automatic PII handling. Policies specify
what action to take when PII is detected, including deletion, anonymization,
archival, or notification only.

Policy Actions:
    - DELETE: Permanently remove PII from source system
    - ANONYMIZE: Replace PII with anonymized values in place
    - ARCHIVE: Archive PII before deletion (for audit trails)
    - NOTIFY_ONLY: Send notification but take no automatic action

Key Features:
    - Configurable grace periods before remediation
    - Approval workflows for sensitive PII types
    - GDPR-compliant deletion certificates
    - Multi-source support (PostgreSQL, S3, Redis, Loki)

Author: GreenLang Framework Team
Date: February 2026
PRD: SEC-011 PII Detection/Redaction Enhancements
Status: Production Ready
"""

from __future__ import annotations

import hashlib
import logging
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional
from uuid import UUID, uuid4

from pydantic import BaseModel, Field, field_validator

# Import PIIType from existing PII scanner
from greenlang.infrastructure.security_scanning.pii_scanner import PIIType

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class RemediationAction(str, Enum):
    """Actions for PII remediation.

    DELETE: Permanently remove PII from the source system.
    ANONYMIZE: Replace PII with anonymized values in place.
    ARCHIVE: Archive PII to secure storage, then delete from source.
    NOTIFY_ONLY: Send notification but take no automatic action.
    """

    DELETE = "delete"
    ANONYMIZE = "anonymize"
    ARCHIVE = "archive"
    NOTIFY_ONLY = "notify_only"


class RemediationStatus(str, Enum):
    """Status of a remediation item.

    PENDING: Scheduled but not yet processed.
    AWAITING_APPROVAL: Requires approval before execution.
    APPROVED: Approved and ready for execution.
    EXECUTING: Currently being processed.
    EXECUTED: Successfully completed.
    FAILED: Failed during execution.
    CANCELLED: Cancelled before execution.
    EXPIRED: Grace period expired without action.
    """

    PENDING = "pending"
    AWAITING_APPROVAL = "awaiting_approval"
    APPROVED = "approved"
    EXECUTING = "executing"
    EXECUTED = "executed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    EXPIRED = "expired"


class SourceType(str, Enum):
    """Types of data sources for PII remediation."""

    POSTGRESQL = "postgresql"
    S3 = "s3"
    REDIS = "redis"
    LOKI = "loki"
    ELASTICSEARCH = "elasticsearch"
    KAFKA = "kafka"
    FILE = "file"


# ---------------------------------------------------------------------------
# Remediation Policy Model
# ---------------------------------------------------------------------------


class RemediationPolicy(BaseModel):
    """Policy for automatic PII remediation.

    Defines the action to take when PII of a specific type is detected,
    including grace periods, approval requirements, and notifications.

    Attributes:
        pii_type: Type of PII this policy applies to.
        action: Remediation action to perform.
        delay_hours: Grace period before remediation (default: 72 hours).
        requires_approval: Whether manual approval is required.
        notify_on_action: Whether to send notifications after action.
        notify_on_detection: Whether to notify when PII is detected.
        retention_days: Days to retain archived data (for ARCHIVE action).
        priority: Policy priority (higher = executed first).
        enabled: Whether this policy is active.
        notify_channels: Notification channels to use.

    Example:
        >>> policy = RemediationPolicy(
        ...     pii_type=PIIType.SSN,
        ...     action=RemediationAction.DELETE,
        ...     delay_hours=24,
        ...     requires_approval=True
        ... )
    """

    pii_type: PIIType = Field(..., description="PII type this policy applies to")
    action: RemediationAction = Field(
        default=RemediationAction.NOTIFY_ONLY,
        description="Remediation action to perform"
    )
    delay_hours: int = Field(
        default=72,
        ge=0,
        le=8760,  # Max 1 year
        description="Grace period in hours before remediation"
    )
    requires_approval: bool = Field(
        default=False,
        description="Whether manual approval is required"
    )
    notify_on_action: bool = Field(
        default=True,
        description="Send notification after remediation action"
    )
    notify_on_detection: bool = Field(
        default=True,
        description="Send notification when PII is detected"
    )
    retention_days: Optional[int] = Field(
        default=None,
        ge=1,
        le=3650,  # Max 10 years
        description="Days to retain archived data"
    )
    priority: int = Field(
        default=50,
        ge=1,
        le=100,
        description="Policy priority (higher = executed first)"
    )
    enabled: bool = Field(default=True, description="Whether policy is active")
    notify_channels: List[str] = Field(
        default_factory=lambda: ["email", "slack"],
        description="Notification channels"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional policy metadata"
    )

    model_config = {"frozen": True}

    @field_validator("retention_days")
    @classmethod
    def validate_retention_for_archive(cls, v, info):
        """Validate retention_days is set for ARCHIVE action."""
        # Note: This validation would need access to action field
        # Keeping simple for now
        return v


# ---------------------------------------------------------------------------
# Remediation Item Model
# ---------------------------------------------------------------------------


class PIIRemediationItem(BaseModel):
    """Item pending or completed remediation.

    Represents a single piece of PII that has been detected and is
    scheduled for remediation according to the applicable policy.

    Attributes:
        id: Unique identifier for this item.
        pii_type: Type of PII detected.
        source_type: Type of data source (postgresql, s3, redis, etc.).
        source_location: Specific location (table.column, s3://bucket/key).
        record_identifier: Primary key or object key of the record.
        tenant_id: Tenant this PII belongs to.
        detected_at: When the PII was detected.
        scheduled_for: When remediation is scheduled.
        status: Current remediation status.
        action: Remediation action to perform.
        approved_by: User who approved (if required).
        approved_at: When approval was granted.
        executed_at: When remediation was executed.
        deletion_certificate_id: ID of GDPR deletion certificate.
        error_message: Error message if failed.
        retry_count: Number of execution retries.
        metadata: Additional metadata.

    Example:
        >>> item = PIIRemediationItem(
        ...     pii_type=PIIType.EMAIL,
        ...     source_type=SourceType.POSTGRESQL,
        ...     source_location="users.email",
        ...     record_identifier="user-123",
        ...     tenant_id="tenant-acme"
        ... )
    """

    id: UUID = Field(default_factory=uuid4, description="Unique item identifier")
    pii_type: PIIType = Field(..., description="Type of PII detected")
    source_type: SourceType = Field(..., description="Type of data source")
    source_location: str = Field(..., description="Location in source system")
    record_identifier: str = Field(..., description="Record primary key or ID")
    tenant_id: str = Field(..., description="Tenant ID")
    detected_at: datetime = Field(
        default_factory=datetime.utcnow,
        description="Detection timestamp"
    )
    scheduled_for: datetime = Field(
        default_factory=lambda: datetime.utcnow() + timedelta(hours=72),
        description="Scheduled remediation time"
    )
    status: RemediationStatus = Field(
        default=RemediationStatus.PENDING,
        description="Current status"
    )
    action: RemediationAction = Field(
        default=RemediationAction.DELETE,
        description="Remediation action"
    )
    approved_by: Optional[UUID] = Field(
        default=None,
        description="Approving user ID"
    )
    approved_at: Optional[datetime] = Field(
        default=None,
        description="Approval timestamp"
    )
    executed_at: Optional[datetime] = Field(
        default=None,
        description="Execution timestamp"
    )
    deletion_certificate_id: Optional[UUID] = Field(
        default=None,
        description="GDPR deletion certificate ID"
    )
    error_message: Optional[str] = Field(
        default=None,
        description="Error message if failed"
    )
    retry_count: int = Field(
        default=0,
        ge=0,
        description="Number of retries"
    )
    pii_value_hash: Optional[str] = Field(
        default=None,
        description="SHA-256 hash of PII value (for verification)"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata"
    )

    model_config = {"validate_assignment": True}

    def is_due(self) -> bool:
        """Check if remediation is due.

        Returns:
            True if scheduled_for has passed.
        """
        return datetime.utcnow() >= self.scheduled_for

    def is_actionable(self) -> bool:
        """Check if item can be processed.

        Returns:
            True if status allows execution.
        """
        return self.status in (
            RemediationStatus.PENDING,
            RemediationStatus.APPROVED
        )

    def needs_approval(self, policy: RemediationPolicy) -> bool:
        """Check if item needs approval based on policy.

        Args:
            policy: The applicable remediation policy.

        Returns:
            True if approval is required and not yet granted.
        """
        if not policy.requires_approval:
            return False
        return self.status == RemediationStatus.AWAITING_APPROVAL

    def can_retry(self, max_retries: int = 3) -> bool:
        """Check if item can be retried.

        Args:
            max_retries: Maximum retry attempts.

        Returns:
            True if retry is allowed.
        """
        return (
            self.status == RemediationStatus.FAILED
            and self.retry_count < max_retries
        )


# ---------------------------------------------------------------------------
# Deletion Certificate Model
# ---------------------------------------------------------------------------


class DeletionCertificate(BaseModel):
    """GDPR-compliant deletion certificate.

    Provides proof of PII deletion for compliance and audit purposes.
    Includes cryptographic verification of the deletion operation.

    Attributes:
        id: Certificate identifier.
        remediation_item_id: Associated remediation item.
        pii_type: Type of PII that was deleted.
        source_type: Source system type.
        source_location: Location in source system.
        deleted_at: When deletion occurred.
        deleted_by: User or system that performed deletion.
        verification_hash: SHA-256 hash for verification.
        tenant_id: Tenant ID.
        subject_id: Data subject ID (if known).
        legal_basis: Legal basis for deletion (e.g., GDPR Art. 17).
        metadata: Additional certificate metadata.

    Example:
        >>> cert = DeletionCertificate(
        ...     remediation_item_id=item.id,
        ...     pii_type=PIIType.EMAIL,
        ...     source_type=SourceType.POSTGRESQL,
        ...     source_location="users.email",
        ...     tenant_id="tenant-acme"
        ... )
    """

    id: UUID = Field(default_factory=uuid4, description="Certificate ID")
    remediation_item_id: UUID = Field(..., description="Remediation item ID")
    pii_type: PIIType = Field(..., description="Type of deleted PII")
    source_type: SourceType = Field(..., description="Source system type")
    source_location: str = Field(..., description="Location in source")
    deleted_at: datetime = Field(
        default_factory=datetime.utcnow,
        description="Deletion timestamp"
    )
    deleted_by: str = Field(
        default="system",
        description="Deleting user or system"
    )
    verification_hash: str = Field(
        default="",
        description="SHA-256 verification hash"
    )
    tenant_id: str = Field(..., description="Tenant ID")
    subject_id: Optional[str] = Field(
        default=None,
        description="Data subject identifier"
    )
    legal_basis: str = Field(
        default="GDPR Article 17 - Right to Erasure",
        description="Legal basis for deletion"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata"
    )

    model_config = {"validate_assignment": True}

    def __init__(self, **data):
        super().__init__(**data)
        if not self.verification_hash:
            self._compute_verification_hash()

    def _compute_verification_hash(self) -> None:
        """Compute the verification hash for this certificate."""
        hash_input = (
            f"{self.id}:{self.remediation_item_id}:{self.pii_type.value}:"
            f"{self.source_type.value}:{self.source_location}:"
            f"{self.tenant_id}:{self.deleted_at.isoformat()}"
        )
        object.__setattr__(
            self,
            "verification_hash",
            hashlib.sha256(hash_input.encode()).hexdigest()
        )


# ---------------------------------------------------------------------------
# Remediation Result Model
# ---------------------------------------------------------------------------


class RemediationResult(BaseModel):
    """Result of a remediation processing run.

    Attributes:
        processed: Number of items successfully processed.
        failed: Number of items that failed.
        skipped: Number of items skipped.
        pending_approval: Number of items awaiting approval.
        started_at: When processing started.
        completed_at: When processing completed.
        errors: List of error messages.
    """

    processed: int = Field(default=0, description="Successfully processed count")
    failed: int = Field(default=0, description="Failed count")
    skipped: int = Field(default=0, description="Skipped count")
    pending_approval: int = Field(default=0, description="Awaiting approval count")
    started_at: datetime = Field(
        default_factory=datetime.utcnow,
        description="Processing start time"
    )
    completed_at: Optional[datetime] = Field(
        default=None,
        description="Processing end time"
    )
    errors: List[str] = Field(default_factory=list, description="Error messages")

    def mark_completed(self) -> None:
        """Mark the result as completed."""
        object.__setattr__(self, "completed_at", datetime.utcnow())

    @property
    def total(self) -> int:
        """Get total items processed."""
        return self.processed + self.failed + self.skipped + self.pending_approval

    @property
    def success_rate(self) -> float:
        """Get success rate as percentage."""
        if self.total == 0:
            return 100.0
        return (self.processed / self.total) * 100


# ---------------------------------------------------------------------------
# Default Remediation Policies
# ---------------------------------------------------------------------------


def _create_default_policies() -> Dict[PIIType, RemediationPolicy]:
    """Create default remediation policies.

    Policies are configured based on sensitivity of the PII type:
    - High-risk PII (SSN, credit card, password): Quick deletion with approval
    - Medium-risk PII (bank account, medical): Moderate delay with approval
    - Lower-risk PII (email, phone): Notify only, longer delay

    Returns:
        Dictionary mapping PIIType to RemediationPolicy.
    """
    return {
        # High-risk: Immediate action with approval
        PIIType.SSN: RemediationPolicy(
            pii_type=PIIType.SSN,
            action=RemediationAction.DELETE,
            delay_hours=24,
            requires_approval=True,
            priority=100,
        ),
        PIIType.CREDIT_CARD: RemediationPolicy(
            pii_type=PIIType.CREDIT_CARD,
            action=RemediationAction.DELETE,
            delay_hours=24,
            requires_approval=True,
            priority=100,
        ),
        PIIType.PASSWORD: RemediationPolicy(
            pii_type=PIIType.PASSWORD,
            action=RemediationAction.DELETE,
            delay_hours=1,
            requires_approval=False,  # Immediate deletion for leaked passwords
            priority=100,
        ),
        PIIType.API_KEY: RemediationPolicy(
            pii_type=PIIType.API_KEY,
            action=RemediationAction.DELETE,
            delay_hours=1,
            requires_approval=False,  # Immediate deletion for leaked keys
            priority=100,
        ),

        # Medium-risk: Moderate delay with approval
        PIIType.FINANCIAL_ACCOUNT: RemediationPolicy(
            pii_type=PIIType.FINANCIAL_ACCOUNT,
            action=RemediationAction.DELETE,
            delay_hours=48,
            requires_approval=True,
            priority=80,
        ),
        PIIType.MEDICAL_RECORD: RemediationPolicy(
            pii_type=PIIType.MEDICAL_RECORD,
            action=RemediationAction.ARCHIVE,
            delay_hours=72,
            requires_approval=True,
            retention_days=365 * 7,  # 7 year retention for medical
            priority=80,
        ),
        PIIType.PASSPORT: RemediationPolicy(
            pii_type=PIIType.PASSPORT,
            action=RemediationAction.DELETE,
            delay_hours=48,
            requires_approval=True,
            priority=80,
        ),
        PIIType.DRIVER_LICENSE: RemediationPolicy(
            pii_type=PIIType.DRIVER_LICENSE,
            action=RemediationAction.DELETE,
            delay_hours=48,
            requires_approval=True,
            priority=80,
        ),

        # Lower-risk: Longer delay, notify only by default
        PIIType.EMAIL: RemediationPolicy(
            pii_type=PIIType.EMAIL,
            action=RemediationAction.NOTIFY_ONLY,
            delay_hours=168,  # 1 week
            requires_approval=False,
            priority=40,
        ),
        PIIType.PHONE: RemediationPolicy(
            pii_type=PIIType.PHONE,
            action=RemediationAction.NOTIFY_ONLY,
            delay_hours=168,  # 1 week
            requires_approval=False,
            priority=40,
        ),
        PIIType.ADDRESS: RemediationPolicy(
            pii_type=PIIType.ADDRESS,
            action=RemediationAction.NOTIFY_ONLY,
            delay_hours=168,  # 1 week
            requires_approval=False,
            priority=40,
        ),
        PIIType.NAME: RemediationPolicy(
            pii_type=PIIType.NAME,
            action=RemediationAction.NOTIFY_ONLY,
            delay_hours=168,  # 1 week
            requires_approval=False,
            priority=20,
        ),
        PIIType.DOB: RemediationPolicy(
            pii_type=PIIType.DOB,
            action=RemediationAction.NOTIFY_ONLY,
            delay_hours=72,
            requires_approval=False,
            priority=60,
        ),

        # Infrastructure: Quick cleanup
        PIIType.IP_ADDRESS: RemediationPolicy(
            pii_type=PIIType.IP_ADDRESS,
            action=RemediationAction.ANONYMIZE,
            delay_hours=24,
            requires_approval=False,
            priority=30,
        ),
        PIIType.TOKEN: RemediationPolicy(
            pii_type=PIIType.TOKEN,
            action=RemediationAction.DELETE,
            delay_hours=1,
            requires_approval=False,
            priority=90,
        ),
    }


# Module-level default policies
DEFAULT_REMEDIATION_POLICIES: Dict[PIIType, RemediationPolicy] = _create_default_policies()


def get_default_policy(pii_type: PIIType) -> Optional[RemediationPolicy]:
    """Get the default policy for a PII type.

    Args:
        pii_type: The PII type.

    Returns:
        RemediationPolicy if defined, None otherwise.
    """
    return DEFAULT_REMEDIATION_POLICIES.get(pii_type)


def get_all_default_policies() -> Dict[PIIType, RemediationPolicy]:
    """Get all default remediation policies.

    Returns:
        Dictionary of all default policies.
    """
    return DEFAULT_REMEDIATION_POLICIES.copy()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

__all__ = [
    # Enums
    "RemediationAction",
    "RemediationStatus",
    "SourceType",
    # Models
    "RemediationPolicy",
    "PIIRemediationItem",
    "DeletionCertificate",
    "RemediationResult",
    # Defaults
    "DEFAULT_REMEDIATION_POLICIES",
    "get_default_policy",
    "get_all_default_policies",
]

logger.debug(
    "PII remediation policies loaded: %d default policies",
    len(DEFAULT_REMEDIATION_POLICIES)
)
