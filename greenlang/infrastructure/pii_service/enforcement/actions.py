# -*- coding: utf-8 -*-
"""
PII Enforcement Actions - SEC-011

Data models and handlers for enforcement actions taken on PII detections.
Captures the full audit trail of what action was taken, why, and when.

Classes:
    - PIIDetection: Represents a detected PII instance
    - ActionTaken: Record of an action taken on a detection
    - QuarantineItem: Item stored in quarantine for review
    - EnforcementResult: Complete result of an enforcement operation

Author: GreenLang Security Team
Date: February 2026
PRD: SEC-011 PII Detection/Redaction Enhancements
Status: Production Ready
"""

from __future__ import annotations

import hashlib
import logging
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field, field_validator, computed_field

from greenlang.infrastructure.pii_service.enforcement.policies import (
    EnforcementAction,
    EnforcementContext,
    EnforcementPolicy,
    PIIType,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Detection Models
# ---------------------------------------------------------------------------


class PIIDetection(BaseModel):
    """Represents a single PII detection in content.

    This model captures the location, type, and confidence of a PII match.
    The actual PII value is never stored; only a hash is retained for
    audit trail purposes.

    Attributes:
        id: Unique identifier for this detection.
        pii_type: Type of PII detected.
        confidence: Detection confidence score (0-1).
        start: Start position in the content.
        end: End position in the content.
        value_hash: SHA-256 hash of the detected value (never raw value).
        pattern_name: Name of the detection pattern that matched.
        context_snippet: Redacted context around the detection.
        detected_at: Timestamp of detection.
        metadata: Additional detection metadata.

    Example:
        >>> detection = PIIDetection(
        ...     pii_type=PIIType.SSN,
        ...     confidence=0.95,
        ...     start=45,
        ...     end=56,
        ...     value_hash="abc123...",
        ... )
    """

    model_config = ConfigDict(
        frozen=False,
        validate_assignment=True,
        extra="forbid",
    )

    id: UUID = Field(
        default_factory=uuid4,
        description="Unique detection identifier",
    )
    pii_type: PIIType = Field(
        ...,
        description="Type of PII detected",
    )
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Detection confidence score",
    )
    start: int = Field(
        ...,
        ge=0,
        description="Start position in content",
    )
    end: int = Field(
        ...,
        ge=0,
        description="End position in content",
    )
    value_hash: str = Field(
        default="",
        max_length=64,
        description="SHA-256 hash of detected value",
    )
    pattern_name: Optional[str] = Field(
        default=None,
        max_length=100,
        description="Detection pattern name",
    )
    context_snippet: Optional[str] = Field(
        default=None,
        max_length=200,
        description="Redacted context around detection",
    )
    detected_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Detection timestamp",
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata",
    )

    @field_validator("end")
    @classmethod
    def validate_end_after_start(cls, v: int, info) -> int:
        """Ensure end position is after start."""
        start = info.data.get("start", 0)
        if v < start:
            raise ValueError("End position must be >= start position")
        return v

    @computed_field
    @property
    def length(self) -> int:
        """Length of the detected PII value."""
        return self.end - self.start

    @classmethod
    def from_match(
        cls,
        pii_type: PIIType,
        value: str,
        start: int,
        end: int,
        confidence: float,
        pattern_name: Optional[str] = None,
        context: Optional[str] = None,
    ) -> "PIIDetection":
        """Create a detection from a matched value.

        The value is hashed immediately; raw PII is never stored.

        Args:
            pii_type: Type of PII.
            value: The matched PII value (will be hashed).
            start: Start position.
            end: End position.
            confidence: Detection confidence.
            pattern_name: Name of matching pattern.
            context: Surrounding context (will be truncated).

        Returns:
            PIIDetection instance.
        """
        value_hash = hashlib.sha256(value.encode()).hexdigest()

        # Truncate context if too long
        if context and len(context) > 200:
            context = context[:97] + "..." + context[-100:]

        return cls(
            pii_type=pii_type,
            confidence=confidence,
            start=start,
            end=end,
            value_hash=value_hash,
            pattern_name=pattern_name,
            context_snippet=context,
        )


# ---------------------------------------------------------------------------
# Action Models
# ---------------------------------------------------------------------------


class ActionTaken(BaseModel):
    """Record of an enforcement action taken on a detection.

    This provides a complete audit trail of what action was taken and why,
    supporting compliance requirements (GDPR, CCPA, SOC2).

    Attributes:
        id: Unique identifier for this action record.
        detection: The PII detection this action was for.
        action: The enforcement action that was taken.
        policy: The policy that triggered this action.
        reason: Human-readable explanation of why action was taken.
        timestamp: When the action was taken.
        success: Whether the action was successful.
        error_message: Error details if action failed.
        metadata: Additional action metadata.

    Example:
        >>> action = ActionTaken(
        ...     detection=detection,
        ...     action=EnforcementAction.BLOCK,
        ...     reason="SSN detected with 95% confidence",
        ... )
    """

    model_config = ConfigDict(
        frozen=False,
        validate_assignment=True,
        extra="forbid",
    )

    id: UUID = Field(
        default_factory=uuid4,
        description="Unique action identifier",
    )
    detection: PIIDetection = Field(
        ...,
        description="The detection this action is for",
    )
    action: EnforcementAction = Field(
        ...,
        description="Action that was taken",
    )
    policy: Optional[EnforcementPolicy] = Field(
        default=None,
        description="Policy that triggered the action",
    )
    reason: str = Field(
        ...,
        max_length=500,
        description="Why this action was taken",
    )
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="When action was taken",
    )
    success: bool = Field(
        default=True,
        description="Whether action succeeded",
    )
    error_message: Optional[str] = Field(
        default=None,
        max_length=1000,
        description="Error message if failed",
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata",
    )

    @classmethod
    def create(
        cls,
        detection: PIIDetection,
        action: EnforcementAction,
        reason: str,
        policy: Optional[EnforcementPolicy] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> "ActionTaken":
        """Create an action record.

        Args:
            detection: The PII detection.
            action: The action taken.
            reason: Why the action was taken.
            policy: The triggering policy.
            metadata: Additional metadata.

        Returns:
            ActionTaken instance.
        """
        return cls(
            detection=detection,
            action=action,
            reason=reason,
            policy=policy,
            metadata=metadata or {},
        )


# ---------------------------------------------------------------------------
# Quarantine Models
# ---------------------------------------------------------------------------


class QuarantineStatus(str):
    """Status values for quarantine items."""

    PENDING = "pending"
    RELEASED = "released"
    DELETED = "deleted"
    EXPIRED = "expired"


class QuarantineItem(BaseModel):
    """Item stored in quarantine for manual review.

    When PII is quarantined, the content is stored securely for review by
    authorized personnel. The item includes full context for review.

    Attributes:
        id: Unique quarantine item identifier.
        content_hash: SHA-256 hash of quarantined content.
        pii_type: Type of PII that triggered quarantine.
        detection_confidence: Confidence score of the detection.
        source_type: Type of source (api_request, storage, etc.).
        source_location: Path/identifier of the source.
        tenant_id: Tenant that owns this item.
        detected_at: When PII was detected.
        expires_at: When this item expires from quarantine.
        status: Current status (pending, released, deleted).
        reviewed_by: User who reviewed this item.
        reviewed_at: When item was reviewed.
        review_notes: Notes from the reviewer.
        metadata: Additional metadata.

    Example:
        >>> item = QuarantineItem(
        ...     content_hash="abc123...",
        ...     pii_type=PIIType.SSN,
        ...     detection_confidence=0.95,
        ...     source_type="api_request",
        ...     tenant_id="tenant-acme",
        ...     expires_at=datetime.now() + timedelta(hours=72),
        ... )
    """

    model_config = ConfigDict(
        frozen=False,
        validate_assignment=True,
        extra="forbid",
    )

    id: UUID = Field(
        default_factory=uuid4,
        description="Unique quarantine item ID",
    )
    content_hash: str = Field(
        ...,
        min_length=64,
        max_length=64,
        description="SHA-256 hash of quarantined content",
    )
    pii_type: PIIType = Field(
        ...,
        description="PII type that triggered quarantine",
    )
    detection_confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Detection confidence score",
    )
    source_type: str = Field(
        ...,
        max_length=50,
        description="Source type (api_request, storage, etc.)",
    )
    source_location: Optional[str] = Field(
        default=None,
        max_length=2048,
        description="Path or identifier of the source",
    )
    tenant_id: str = Field(
        ...,
        min_length=1,
        max_length=100,
        description="Tenant identifier",
    )
    detected_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="When PII was detected",
    )
    expires_at: datetime = Field(
        ...,
        description="When item expires from quarantine",
    )
    status: str = Field(
        default=QuarantineStatus.PENDING,
        description="Current quarantine status",
    )
    reviewed_by: Optional[str] = Field(
        default=None,
        max_length=100,
        description="User who reviewed item",
    )
    reviewed_at: Optional[datetime] = Field(
        default=None,
        description="When item was reviewed",
    )
    review_notes: Optional[str] = Field(
        default=None,
        max_length=2000,
        description="Notes from reviewer",
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata",
    )

    @classmethod
    def create(
        cls,
        content: str,
        detection: PIIDetection,
        context: EnforcementContext,
        ttl_hours: int = 72,
    ) -> "QuarantineItem":
        """Create a quarantine item from a detection.

        Args:
            content: The content being quarantined (will be hashed).
            detection: The PII detection.
            context: The enforcement context.
            ttl_hours: Hours until item expires.

        Returns:
            QuarantineItem instance.
        """
        content_hash = hashlib.sha256(content.encode()).hexdigest()
        expires_at = datetime.now(timezone.utc) + timedelta(hours=ttl_hours)

        return cls(
            content_hash=content_hash,
            pii_type=detection.pii_type,
            detection_confidence=detection.confidence,
            source_type=context.context_type,
            source_location=context.path,
            tenant_id=context.tenant_id,
            expires_at=expires_at,
        )

    @computed_field
    @property
    def is_expired(self) -> bool:
        """Check if this quarantine item has expired."""
        return datetime.now(timezone.utc) > self.expires_at

    @computed_field
    @property
    def is_pending(self) -> bool:
        """Check if item is still pending review."""
        return self.status == QuarantineStatus.PENDING and not self.is_expired


# ---------------------------------------------------------------------------
# Result Models
# ---------------------------------------------------------------------------


class EnforcementResult(BaseModel):
    """Complete result of an enforcement operation.

    Contains all information about what was detected, what actions were
    taken, and the resulting content (if not blocked).

    Attributes:
        id: Unique result identifier.
        blocked: Whether the content was blocked.
        original_content: Original content (for audit, never exposed).
        modified_content: Modified content after enforcement.
        detections: All PII detections found.
        actions_taken: All enforcement actions taken.
        context: The enforcement context.
        processing_time_ms: Time spent processing.
        timestamp: When enforcement was processed.
        error: Error message if processing failed.
        metadata: Additional result metadata.

    Example:
        >>> result = EnforcementResult(
        ...     blocked=True,
        ...     detections=[detection],
        ...     actions_taken=[action],
        ...     context=context,
        ... )
    """

    model_config = ConfigDict(
        frozen=False,
        validate_assignment=True,
        extra="forbid",
    )

    id: UUID = Field(
        default_factory=uuid4,
        description="Unique result identifier",
    )
    blocked: bool = Field(
        default=False,
        description="Whether content was blocked",
    )
    original_content: Optional[str] = Field(
        default=None,
        description="Original content (for internal audit only)",
        repr=False,  # Don't expose in repr
    )
    modified_content: Optional[str] = Field(
        default=None,
        description="Modified content after enforcement",
    )
    detections: List[PIIDetection] = Field(
        default_factory=list,
        description="All PII detections found",
    )
    actions_taken: List[ActionTaken] = Field(
        default_factory=list,
        description="All actions taken",
    )
    context: Optional[EnforcementContext] = Field(
        default=None,
        description="Enforcement context",
    )
    processing_time_ms: float = Field(
        default=0.0,
        ge=0.0,
        description="Processing time in milliseconds",
    )
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="When enforcement was processed",
    )
    error: Optional[str] = Field(
        default=None,
        max_length=2000,
        description="Error message if failed",
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata",
    )

    @computed_field
    @property
    def detection_count(self) -> int:
        """Number of PII detections."""
        return len(self.detections)

    @computed_field
    @property
    def blocked_types(self) -> List[str]:
        """List of PII types that caused blocking."""
        return [
            action.detection.pii_type.value
            for action in self.actions_taken
            if action.action == EnforcementAction.BLOCK
        ]

    @computed_field
    @property
    def redacted_count(self) -> int:
        """Number of redactions applied."""
        return sum(
            1 for action in self.actions_taken
            if action.action == EnforcementAction.REDACT
        )

    def to_response_dict(self) -> Dict[str, Any]:
        """Convert to a safe dictionary for API responses.

        Excludes sensitive fields like original_content.

        Returns:
            Dictionary safe for API response.
        """
        return {
            "id": str(self.id),
            "blocked": self.blocked,
            "detection_count": self.detection_count,
            "blocked_types": self.blocked_types,
            "redacted_count": self.redacted_count,
            "processing_time_ms": self.processing_time_ms,
            "timestamp": self.timestamp.isoformat(),
        }


# ---------------------------------------------------------------------------
# Exports
# ---------------------------------------------------------------------------

__all__ = [
    "PIIDetection",
    "ActionTaken",
    "QuarantineStatus",
    "QuarantineItem",
    "EnforcementResult",
]
