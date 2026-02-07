# -*- coding: utf-8 -*-
"""
PII Service Shared Models - SEC-011: PII Detection/Redaction Enhancements

Pydantic models shared across PII Service components:
- Detection results from scanners
- Tokenization entries for the vault
- Enforcement results and actions
- API request/response models

All models follow GreenLang patterns with full type hints,
validation, and serialization support.

Author: GreenLang Framework Team
Date: February 2026
PRD: SEC-011 PII Detection/Redaction Enhancements
Status: Production Ready
"""

from __future__ import annotations

import hashlib
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Union
from uuid import UUID, uuid4

from pydantic import BaseModel, Field, field_validator, model_validator


# ---------------------------------------------------------------------------
# Re-export PIIType from existing module
# ---------------------------------------------------------------------------

try:
    from greenlang.agents.foundation.pii_redaction import PIIType, RedactionStrategy
except ImportError:
    # Fallback definition if pii_redaction not available
    class PIIType(str, Enum):
        """Types of PII that can be detected."""

        PERSON_NAME = "person_name"
        EMAIL = "email"
        PHONE = "phone"
        SSN = "ssn"
        NATIONAL_ID = "national_id"
        PASSPORT = "passport"
        DRIVERS_LICENSE = "drivers_license"
        CREDIT_CARD = "credit_card"
        BANK_ACCOUNT = "bank_account"
        IBAN = "iban"
        MEDICAL_RECORD = "medical_record"
        HEALTH_INSURANCE_ID = "health_insurance_id"
        ADDRESS = "address"
        IP_ADDRESS = "ip_address"
        GPS_COORDINATES = "gps_coordinates"
        USERNAME = "username"
        PASSWORD = "password"
        API_KEY = "api_key"
        ORGANIZATION_NAME = "organization_name"
        DATE_OF_BIRTH = "date_of_birth"
        CUSTOM = "custom"

    class RedactionStrategy(str, Enum):
        """Strategies for redacting detected PII."""

        MASK = "mask"
        HASH = "hash"
        REPLACE = "replace"
        REMOVE = "remove"
        TOKENIZE = "tokenize"
        PARTIAL_MASK = "partial_mask"


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class EnforcementAction(str, Enum):
    """Actions when PII is detected by enforcement engine."""

    ALLOW = "allow"  # Log only, allow through
    REDACT = "redact"  # Redact and allow
    BLOCK = "block"  # Block the request
    QUARANTINE = "quarantine"  # Store for review, block
    TRANSFORM = "transform"  # Apply transformation (tokenize, hash)


class DetectionMethod(str, Enum):
    """Detection method used to find PII."""

    REGEX = "regex"
    ML = "ml"
    PRESIDIO = "presidio"
    HYBRID = "hybrid"


class ConfidenceLevel(str, Enum):
    """Confidence level categorization."""

    HIGH = "high"  # > 0.9
    MEDIUM = "medium"  # 0.7 - 0.9
    LOW = "low"  # 0.5 - 0.7
    UNCERTAIN = "uncertain"  # < 0.5


# ---------------------------------------------------------------------------
# Detection Models
# ---------------------------------------------------------------------------


class PIIDetection(BaseModel):
    """A single PII detection result.

    Represents one instance of detected PII in content with position,
    confidence, and context information.

    Attributes:
        id: Unique detection identifier.
        pii_type: Type of PII detected.
        value_hash: SHA-256 hash of detected value (never store raw).
        confidence: Detection confidence score (0-1).
        start: Start position in content.
        end: End position in content.
        context: Surrounding text (redacted).
        detection_method: Method used for detection.
        pattern_name: Name of pattern that matched (for regex).
        detected_at: Timestamp of detection.
    """

    id: UUID = Field(default_factory=uuid4, description="Detection ID")
    pii_type: PIIType = Field(..., description="Type of PII detected")
    value_hash: str = Field(..., description="SHA-256 hash of value")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score")
    start: int = Field(..., ge=0, description="Start position")
    end: int = Field(..., ge=0, description="End position")
    context: Optional[str] = Field(None, description="Surrounding context (redacted)")
    detection_method: DetectionMethod = Field(
        default=DetectionMethod.REGEX,
        description="Detection method",
    )
    pattern_name: Optional[str] = Field(None, description="Pattern name")
    detected_at: datetime = Field(
        default_factory=datetime.utcnow,
        description="Detection timestamp",
    )

    @property
    def confidence_level(self) -> ConfidenceLevel:
        """Categorize confidence into levels."""
        if self.confidence >= 0.9:
            return ConfidenceLevel.HIGH
        elif self.confidence >= 0.7:
            return ConfidenceLevel.MEDIUM
        elif self.confidence >= 0.5:
            return ConfidenceLevel.LOW
        else:
            return ConfidenceLevel.UNCERTAIN

    @property
    def length(self) -> int:
        """Length of detected PII value."""
        return self.end - self.start

    @classmethod
    def from_value(
        cls,
        value: str,
        pii_type: PIIType,
        start: int,
        end: int,
        confidence: float,
        detection_method: DetectionMethod = DetectionMethod.REGEX,
        context: Optional[str] = None,
        pattern_name: Optional[str] = None,
    ) -> PIIDetection:
        """Create detection from raw value (hashes automatically).

        Args:
            value: Raw PII value to hash.
            pii_type: Type of PII.
            start: Start position.
            end: End position.
            confidence: Confidence score.
            detection_method: Detection method used.
            context: Surrounding context.
            pattern_name: Pattern name.

        Returns:
            PIIDetection with hashed value.
        """
        value_hash = hashlib.sha256(value.encode()).hexdigest()
        return cls(
            pii_type=pii_type,
            value_hash=value_hash,
            confidence=confidence,
            start=start,
            end=end,
            context=context,
            detection_method=detection_method,
            pattern_name=pattern_name,
        )


class DetectionOptions(BaseModel):
    """Options for PII detection operations.

    Attributes:
        use_ml: Use ML-based detection (Presidio).
        apply_allowlist: Filter results through allowlist.
        min_confidence: Minimum confidence threshold.
        pii_types: Specific PII types to detect (None = all).
        tenant_id: Tenant for context-aware detection.
        source: Source identifier for metrics.
        max_detections: Maximum detections to return.
    """

    use_ml: bool = Field(default=True, description="Use ML-based detection")
    apply_allowlist: bool = Field(default=True, description="Apply allowlist")
    min_confidence: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Minimum confidence",
    )
    pii_types: Optional[List[PIIType]] = Field(
        None,
        description="PII types to detect (None = all)",
    )
    tenant_id: Optional[str] = Field(None, description="Tenant ID")
    source: str = Field(default="api", description="Detection source")
    max_detections: int = Field(
        default=1000,
        ge=1,
        le=10000,
        description="Maximum detections",
    )


# ---------------------------------------------------------------------------
# Token Vault Models
# ---------------------------------------------------------------------------


class EncryptedTokenEntry(BaseModel):
    """Entry in the secure token vault.

    Stores encrypted PII with tenant isolation and expiration.

    Attributes:
        token_id: Unique token identifier.
        pii_type: Type of PII stored.
        original_hash: SHA-256 hash of original value.
        encrypted_value: AES-256-GCM encrypted value.
        tenant_id: Owning tenant ID.
        created_at: Token creation timestamp.
        expires_at: Token expiration timestamp.
        access_count: Number of detokenization requests.
        last_accessed_at: Last access timestamp.
        metadata: Additional metadata.
    """

    token_id: str = Field(..., description="Token identifier")
    pii_type: PIIType = Field(..., description="Type of PII")
    original_hash: str = Field(..., description="SHA-256 of original value")
    encrypted_value: bytes = Field(..., description="AES-256-GCM encrypted value")
    tenant_id: str = Field(..., description="Owning tenant")
    created_at: datetime = Field(
        default_factory=datetime.utcnow,
        description="Creation time",
    )
    expires_at: datetime = Field(..., description="Expiration time")
    access_count: int = Field(default=0, ge=0, description="Access count")
    last_accessed_at: Optional[datetime] = Field(None, description="Last access")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Metadata")

    @property
    def is_expired(self) -> bool:
        """Check if token has expired."""
        return datetime.utcnow() > self.expires_at

    @property
    def days_until_expiry(self) -> int:
        """Days until expiration (negative if expired)."""
        delta = self.expires_at - datetime.utcnow()
        return delta.days

    model_config = {
        "arbitrary_types_allowed": True,
    }


# ---------------------------------------------------------------------------
# Redaction Models
# ---------------------------------------------------------------------------


class RedactionResult(BaseModel):
    """Result of a PII redaction operation.

    Attributes:
        original_length: Length of original content.
        redacted_content: Content with PII redacted.
        detections: List of PII detections.
        tokens_created: Tokens created for tokenize strategy.
        redaction_count: Number of redactions applied.
        processing_time_ms: Processing duration.
    """

    original_length: int = Field(..., ge=0, description="Original content length")
    redacted_content: str = Field(..., description="Redacted content")
    detections: List[PIIDetection] = Field(
        default_factory=list,
        description="PII detections",
    )
    tokens_created: List[str] = Field(
        default_factory=list,
        description="Tokens created",
    )
    redaction_count: int = Field(default=0, ge=0, description="Redactions applied")
    processing_time_ms: float = Field(default=0.0, ge=0, description="Processing time")

    @property
    def pii_found(self) -> bool:
        """Whether any PII was found."""
        return len(self.detections) > 0


class RedactionOptions(BaseModel):
    """Options for PII redaction operations.

    Attributes:
        detection_options: Options for detection phase.
        strategy: Default redaction strategy.
        strategy_overrides: Per-PII-type strategy overrides.
        create_tokens: Create tokens for tokenize strategy.
        tenant_id: Tenant for token creation.
    """

    detection_options: DetectionOptions = Field(
        default_factory=DetectionOptions,
        description="Detection options",
    )
    strategy: RedactionStrategy = Field(
        default=RedactionStrategy.REPLACE,
        description="Default redaction strategy",
    )
    strategy_overrides: Dict[PIIType, RedactionStrategy] = Field(
        default_factory=dict,
        description="Per-type strategy overrides",
    )
    create_tokens: bool = Field(
        default=True,
        description="Create tokens for tokenize strategy",
    )
    tenant_id: Optional[str] = Field(None, description="Tenant for tokens")


# ---------------------------------------------------------------------------
# Enforcement Models
# ---------------------------------------------------------------------------


class EnforcementContext(BaseModel):
    """Context for enforcement decision.

    Attributes:
        context_type: Type of context (api_request, log, stream).
        path: API path or source identifier.
        method: HTTP method (for API).
        tenant_id: Tenant identifier.
        user_id: User identifier.
        request_id: Request correlation ID.
        timestamp: Context timestamp.
        metadata: Additional metadata.
    """

    context_type: str = Field(..., description="Context type")
    path: Optional[str] = Field(None, description="API path or source")
    method: Optional[str] = Field(None, description="HTTP method")
    tenant_id: Optional[str] = Field(None, description="Tenant ID")
    user_id: Optional[str] = Field(None, description="User ID")
    request_id: Optional[str] = Field(None, description="Request ID")
    timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="Context timestamp",
    )
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Metadata")


class ActionTaken(BaseModel):
    """Record of an enforcement action.

    Attributes:
        detection: The PII detection that triggered action.
        action: Action taken.
        reason: Reason for the action.
        policy_id: ID of policy that matched.
        timestamp: When action was taken.
    """

    detection: PIIDetection = Field(..., description="Triggering detection")
    action: EnforcementAction = Field(..., description="Action taken")
    reason: str = Field(..., description="Action reason")
    policy_id: Optional[str] = Field(None, description="Matching policy ID")
    timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="Action timestamp",
    )


class EnforcementResult(BaseModel):
    """Result of enforcement engine processing.

    Attributes:
        blocked: Whether content was blocked.
        original_content: Original content (for audit).
        modified_content: Modified content (if redacted).
        detections: All PII detections.
        actions_taken: List of actions taken.
        context: Enforcement context.
        processing_time_ms: Processing duration.
    """

    blocked: bool = Field(default=False, description="Was content blocked")
    original_content: Optional[str] = Field(None, description="Original content")
    modified_content: Optional[str] = Field(None, description="Modified content")
    detections: List[PIIDetection] = Field(
        default_factory=list,
        description="PII detections",
    )
    actions_taken: List[ActionTaken] = Field(
        default_factory=list,
        description="Actions taken",
    )
    context: Optional[EnforcementContext] = Field(None, description="Context")
    processing_time_ms: float = Field(default=0.0, ge=0, description="Processing time")

    @property
    def action_summary(self) -> Dict[EnforcementAction, int]:
        """Count of actions by type."""
        summary: Dict[EnforcementAction, int] = {}
        for action in self.actions_taken:
            summary[action.action] = summary.get(action.action, 0) + 1
        return summary


# ---------------------------------------------------------------------------
# Allowlist Models
# ---------------------------------------------------------------------------


class AllowlistEntry(BaseModel):
    """Entry in the PII allowlist.

    Attributes:
        id: Entry identifier.
        pii_type: PII type this entry applies to.
        pattern: Pattern to match (regex or exact).
        pattern_type: Type of pattern matching.
        reason: Reason for allowlisting.
        created_by: User who created entry.
        created_at: Creation timestamp.
        expires_at: Expiration timestamp (None = never).
        tenant_id: Tenant-specific or global (None).
        enabled: Whether entry is active.
    """

    id: UUID = Field(default_factory=uuid4, description="Entry ID")
    pii_type: PIIType = Field(..., description="PII type")
    pattern: str = Field(..., description="Pattern to match")
    pattern_type: str = Field(
        default="regex",
        description="Pattern type (regex, exact, prefix, suffix, contains)",
    )
    reason: str = Field(..., description="Reason for allowlisting")
    created_by: UUID = Field(..., description="Creator user ID")
    created_at: datetime = Field(
        default_factory=datetime.utcnow,
        description="Creation time",
    )
    expires_at: Optional[datetime] = Field(None, description="Expiration time")
    tenant_id: Optional[str] = Field(None, description="Tenant (None = global)")
    enabled: bool = Field(default=True, description="Is entry active")

    @property
    def is_expired(self) -> bool:
        """Check if entry has expired."""
        if self.expires_at is None:
            return False
        return datetime.utcnow() > self.expires_at

    @property
    def is_global(self) -> bool:
        """Check if entry is global (not tenant-specific)."""
        return self.tenant_id is None


# ---------------------------------------------------------------------------
# API Models
# ---------------------------------------------------------------------------


class DetectRequest(BaseModel):
    """API request for PII detection."""

    content: str = Field(..., min_length=1, max_length=10_000_000, description="Content to scan")
    options: Optional[DetectionOptions] = Field(None, description="Detection options")


class DetectResponse(BaseModel):
    """API response for PII detection."""

    detections: List[PIIDetection] = Field(default_factory=list)
    detection_count: int = Field(default=0)
    pii_types_found: List[str] = Field(default_factory=list)
    processing_time_ms: float = Field(default=0.0)


class RedactRequest(BaseModel):
    """API request for PII redaction."""

    content: str = Field(..., min_length=1, max_length=10_000_000, description="Content to redact")
    options: Optional[RedactionOptions] = Field(None, description="Redaction options")


class RedactResponse(BaseModel):
    """API response for PII redaction."""

    redacted_content: str = Field(..., description="Redacted content")
    detections: List[PIIDetection] = Field(default_factory=list)
    tokens_created: List[str] = Field(default_factory=list)
    processing_time_ms: float = Field(default=0.0)


class TokenizeRequest(BaseModel):
    """API request for PII tokenization."""

    value: str = Field(..., min_length=1, max_length=10000, description="Value to tokenize")
    pii_type: PIIType = Field(..., description="Type of PII")
    tenant_id: str = Field(..., description="Tenant ID")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")


class TokenizeResponse(BaseModel):
    """API response for PII tokenization."""

    token: str = Field(..., description="Generated token")
    pii_type: str = Field(..., description="PII type")
    expires_at: datetime = Field(..., description="Expiration time")


class DetokenizeRequest(BaseModel):
    """API request for PII detokenization."""

    token: str = Field(..., description="Token to resolve")
    tenant_id: str = Field(..., description="Tenant ID for authorization")
    user_id: str = Field(..., description="User ID for audit")


class DetokenizeResponse(BaseModel):
    """API response for PII detokenization."""

    value: str = Field(..., description="Original value")
    pii_type: str = Field(..., description="PII type")


__all__ = [
    # Re-exported
    "PIIType",
    "RedactionStrategy",
    # Enums
    "EnforcementAction",
    "DetectionMethod",
    "ConfidenceLevel",
    # Detection
    "PIIDetection",
    "DetectionOptions",
    # Token vault
    "EncryptedTokenEntry",
    # Redaction
    "RedactionResult",
    "RedactionOptions",
    # Enforcement
    "EnforcementContext",
    "ActionTaken",
    "EnforcementResult",
    # Allowlist
    "AllowlistEntry",
    # API
    "DetectRequest",
    "DetectResponse",
    "RedactRequest",
    "RedactResponse",
    "TokenizeRequest",
    "TokenizeResponse",
    "DetokenizeRequest",
    "DetokenizeResponse",
]
