# -*- coding: utf-8 -*-
"""
PII Enforcement Policies - SEC-011

Defines the enforcement policies for handling detected PII. Each policy
specifies what action to take when a particular type of PII is detected,
based on context, confidence level, and tenant configuration.

Policies support:
    - Per-PII-type enforcement actions (allow, redact, block, quarantine, transform)
    - Minimum confidence thresholds for triggering actions
    - Context-aware enforcement (API, storage, logging, streaming)
    - Per-tenant policy overrides
    - Custom redaction placeholders

Author: GreenLang Security Team
Date: February 2026
PRD: SEC-011 PII Detection/Redaction Enhancements
Status: Production Ready
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class PIIType(str, Enum):
    """Types of PII that can be detected and enforced.

    This enum covers the 19 core PII types identified in SEC-011, plus
    extensibility for custom types.
    """

    # Personal Identifiers
    PERSON_NAME = "person_name"
    EMAIL = "email"
    PHONE = "phone"
    SSN = "ssn"  # Social Security Number (US)
    NATIONAL_ID = "national_id"  # Generic national ID
    PASSPORT = "passport"
    DRIVERS_LICENSE = "drivers_license"

    # Financial
    CREDIT_CARD = "credit_card"
    BANK_ACCOUNT = "bank_account"
    IBAN = "iban"

    # Health (HIPAA)
    MEDICAL_RECORD = "medical_record"
    HEALTH_INSURANCE_ID = "health_insurance_id"

    # Location
    ADDRESS = "address"
    IP_ADDRESS = "ip_address"
    GPS_COORDINATES = "gps_coordinates"

    # Online/Credentials
    USERNAME = "username"
    PASSWORD = "password"
    API_KEY = "api_key"

    # Organization
    ORGANIZATION_NAME = "organization_name"

    # Dates
    DATE_OF_BIRTH = "date_of_birth"

    # Custom/Other
    CUSTOM = "custom"


class EnforcementAction(str, Enum):
    """Actions to take when PII is detected.

    Each action has different implications for data flow:
        - ALLOW: Log the detection but allow data through unchanged
        - REDACT: Replace PII with redacted placeholder and allow
        - BLOCK: Reject the request/message entirely
        - QUARANTINE: Store for manual review and block
        - TRANSFORM: Apply transformation (tokenize, hash) and allow
    """

    ALLOW = "allow"
    REDACT = "redact"
    BLOCK = "block"
    QUARANTINE = "quarantine"
    TRANSFORM = "transform"


class TransformationType(str, Enum):
    """Types of transformation for TRANSFORM action."""

    TOKENIZE = "tokenize"  # Create reversible token via vault
    HASH = "hash"  # One-way SHA-256 hash
    MASK = "mask"  # Partial masking (e.g., ****1234)
    ENCRYPT = "encrypt"  # Encrypt with tenant key


class ContextType(str, Enum):
    """Enforcement context types for policy matching."""

    API_REQUEST = "api_request"
    API_RESPONSE = "api_response"
    STORAGE = "storage"
    LOGGING = "logging"
    STREAMING = "streaming"
    KAFKA_STREAM = "kafka_stream"
    KINESIS_STREAM = "kinesis_stream"
    BATCH_PROCESSING = "batch_processing"
    ALL = "*"


# ---------------------------------------------------------------------------
# Pydantic Models
# ---------------------------------------------------------------------------


class EnforcementPolicy(BaseModel):
    """Policy defining how to handle a specific PII type.

    Each policy specifies the enforcement action, minimum confidence threshold,
    applicable contexts, and notification preferences.

    Attributes:
        pii_type: The type of PII this policy applies to.
        action: What action to take when this PII is detected.
        min_confidence: Minimum confidence score (0-1) to trigger action.
        contexts: List of context types where this policy applies ("*" for all).
        notify: Whether to send notifications for this PII type.
        quarantine_ttl_hours: How long to retain quarantined items.
        custom_placeholder: Custom redaction placeholder text.
        transformation_type: Type of transformation for TRANSFORM action.
        enabled: Whether this policy is active.
        description: Human-readable description of the policy.

    Example:
        >>> policy = EnforcementPolicy(
        ...     pii_type=PIIType.SSN,
        ...     action=EnforcementAction.BLOCK,
        ...     min_confidence=0.8,
        ...     notify=True,
        ... )
    """

    model_config = ConfigDict(
        frozen=False,
        validate_assignment=True,
        extra="forbid",
    )

    pii_type: PIIType = Field(..., description="PII type this policy applies to")
    action: EnforcementAction = Field(
        default=EnforcementAction.ALLOW,
        description="Action to take when PII is detected",
    )
    min_confidence: float = Field(
        default=0.8,
        ge=0.0,
        le=1.0,
        description="Minimum confidence score to trigger action",
    )
    contexts: List[str] = Field(
        default_factory=lambda: ["*"],
        description="Context types where policy applies",
    )
    notify: bool = Field(
        default=True,
        description="Send notifications for detections",
    )
    quarantine_ttl_hours: int = Field(
        default=72,
        ge=1,
        le=8760,  # Max 1 year
        description="Hours to retain quarantined items",
    )
    custom_placeholder: Optional[str] = Field(
        default=None,
        max_length=100,
        description="Custom redaction placeholder",
    )
    transformation_type: TransformationType = Field(
        default=TransformationType.TOKENIZE,
        description="Transformation type for TRANSFORM action",
    )
    enabled: bool = Field(
        default=True,
        description="Whether policy is active",
    )
    description: Optional[str] = Field(
        default=None,
        max_length=500,
        description="Human-readable policy description",
    )

    @field_validator("contexts")
    @classmethod
    def validate_contexts(cls, v: List[str]) -> List[str]:
        """Validate context list is not empty."""
        if not v:
            return ["*"]
        return v

    def matches_context(self, context_type: str) -> bool:
        """Check if this policy applies to the given context.

        Args:
            context_type: The context type to check.

        Returns:
            True if policy applies to this context.
        """
        if "*" in self.contexts:
            return True
        return context_type in self.contexts

    def get_placeholder(self) -> str:
        """Get the redaction placeholder for this PII type.

        Returns:
            Placeholder string for redaction.
        """
        if self.custom_placeholder:
            return self.custom_placeholder
        return f"[{self.pii_type.value.upper()}]"


class EnforcementContext(BaseModel):
    """Context for making enforcement decisions.

    Captures all relevant information about where and how PII was detected,
    enabling context-aware policy evaluation.

    Attributes:
        context_type: Type of context (api_request, storage, etc.).
        path: URL path or file path if applicable.
        method: HTTP method if API context.
        tenant_id: Tenant identifier for multi-tenant isolation.
        user_id: User identifier if available.
        source: Source system identifier.
        request_id: Unique request/trace identifier.
        timestamp: When enforcement is being evaluated.
        metadata: Additional context metadata.

    Example:
        >>> context = EnforcementContext(
        ...     context_type="api_request",
        ...     path="/api/v1/reports",
        ...     method="POST",
        ...     tenant_id="tenant-acme",
        ...     user_id="user-123",
        ... )
    """

    model_config = ConfigDict(
        frozen=False,
        validate_assignment=True,
        extra="allow",
    )

    context_type: str = Field(
        ...,
        description="Type of enforcement context",
    )
    path: Optional[str] = Field(
        default=None,
        max_length=2048,
        description="URL path or file path",
    )
    method: Optional[str] = Field(
        default=None,
        pattern=r"^(GET|POST|PUT|PATCH|DELETE|HEAD|OPTIONS)$",
        description="HTTP method for API contexts",
    )
    tenant_id: str = Field(
        default="default",
        min_length=1,
        max_length=100,
        description="Tenant identifier",
    )
    user_id: Optional[str] = Field(
        default=None,
        max_length=100,
        description="User identifier",
    )
    source: Optional[str] = Field(
        default=None,
        max_length=200,
        description="Source system identifier",
    )
    request_id: Optional[str] = Field(
        default=None,
        max_length=100,
        description="Unique request/trace identifier",
    )
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Enforcement evaluation timestamp",
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional context metadata",
    )

    @field_validator("method", mode="before")
    @classmethod
    def uppercase_method(cls, v: Optional[str]) -> Optional[str]:
        """Ensure HTTP method is uppercase."""
        if v is not None:
            return v.upper()
        return v


# ---------------------------------------------------------------------------
# Default Policies
# ---------------------------------------------------------------------------


def _create_default_policies() -> Dict[PIIType, EnforcementPolicy]:
    """Create the default enforcement policies.

    Returns:
        Dictionary mapping PIIType to default EnforcementPolicy.

    Policy rationale:
        - SSN, Credit Card, Password, Bank Account: BLOCK (high-risk PII)
        - API Key: REDACT (secrets should be masked but not blocked)
        - Email, Phone, Person Name, Address: ALLOW with logging
        - IP Address, GPS, Organization: ALLOW (lower sensitivity)
    """
    return {
        # High-sensitivity PII - BLOCK by default
        PIIType.SSN: EnforcementPolicy(
            pii_type=PIIType.SSN,
            action=EnforcementAction.BLOCK,
            min_confidence=0.8,
            notify=True,
            description="Block requests containing Social Security Numbers",
        ),
        PIIType.CREDIT_CARD: EnforcementPolicy(
            pii_type=PIIType.CREDIT_CARD,
            action=EnforcementAction.BLOCK,
            min_confidence=0.8,
            notify=True,
            description="Block requests containing credit card numbers (PCI-DSS)",
        ),
        PIIType.PASSWORD: EnforcementPolicy(
            pii_type=PIIType.PASSWORD,
            action=EnforcementAction.BLOCK,
            min_confidence=0.7,
            notify=True,
            description="Block requests containing passwords in plaintext",
        ),
        PIIType.BANK_ACCOUNT: EnforcementPolicy(
            pii_type=PIIType.BANK_ACCOUNT,
            action=EnforcementAction.BLOCK,
            min_confidence=0.85,
            notify=True,
            description="Block requests containing bank account numbers",
        ),
        # Secrets - REDACT to prevent leakage
        PIIType.API_KEY: EnforcementPolicy(
            pii_type=PIIType.API_KEY,
            action=EnforcementAction.REDACT,
            min_confidence=0.75,
            notify=True,
            description="Redact API keys and tokens",
        ),
        # Medium-sensitivity PII - ALLOW with notification
        PIIType.EMAIL: EnforcementPolicy(
            pii_type=PIIType.EMAIL,
            action=EnforcementAction.ALLOW,
            min_confidence=0.9,
            notify=False,
            description="Allow emails through (common in business context)",
        ),
        PIIType.PHONE: EnforcementPolicy(
            pii_type=PIIType.PHONE,
            action=EnforcementAction.ALLOW,
            min_confidence=0.85,
            notify=False,
            description="Allow phone numbers through",
        ),
        PIIType.PERSON_NAME: EnforcementPolicy(
            pii_type=PIIType.PERSON_NAME,
            action=EnforcementAction.ALLOW,
            min_confidence=0.7,
            notify=False,
            description="Allow person names through",
        ),
        PIIType.ADDRESS: EnforcementPolicy(
            pii_type=PIIType.ADDRESS,
            action=EnforcementAction.ALLOW,
            min_confidence=0.7,
            notify=False,
            description="Allow addresses through",
        ),
        # Lower-sensitivity PII - ALLOW
        PIIType.IP_ADDRESS: EnforcementPolicy(
            pii_type=PIIType.IP_ADDRESS,
            action=EnforcementAction.ALLOW,
            min_confidence=0.8,
            notify=False,
            description="Allow IP addresses (operational data)",
        ),
        PIIType.GPS_COORDINATES: EnforcementPolicy(
            pii_type=PIIType.GPS_COORDINATES,
            action=EnforcementAction.ALLOW,
            min_confidence=0.8,
            notify=False,
            description="Allow GPS coordinates through",
        ),
        PIIType.ORGANIZATION_NAME: EnforcementPolicy(
            pii_type=PIIType.ORGANIZATION_NAME,
            action=EnforcementAction.ALLOW,
            min_confidence=0.7,
            notify=False,
            description="Allow organization names through",
        ),
        # Health PII - BLOCK for HIPAA compliance
        PIIType.MEDICAL_RECORD: EnforcementPolicy(
            pii_type=PIIType.MEDICAL_RECORD,
            action=EnforcementAction.BLOCK,
            min_confidence=0.8,
            notify=True,
            description="Block medical record numbers (HIPAA)",
        ),
        PIIType.HEALTH_INSURANCE_ID: EnforcementPolicy(
            pii_type=PIIType.HEALTH_INSURANCE_ID,
            action=EnforcementAction.BLOCK,
            min_confidence=0.8,
            notify=True,
            description="Block health insurance IDs (HIPAA)",
        ),
        # Identity documents - BLOCK
        PIIType.PASSPORT: EnforcementPolicy(
            pii_type=PIIType.PASSPORT,
            action=EnforcementAction.BLOCK,
            min_confidence=0.8,
            notify=True,
            description="Block passport numbers",
        ),
        PIIType.DRIVERS_LICENSE: EnforcementPolicy(
            pii_type=PIIType.DRIVERS_LICENSE,
            action=EnforcementAction.BLOCK,
            min_confidence=0.8,
            notify=True,
            description="Block driver's license numbers",
        ),
        PIIType.NATIONAL_ID: EnforcementPolicy(
            pii_type=PIIType.NATIONAL_ID,
            action=EnforcementAction.BLOCK,
            min_confidence=0.8,
            notify=True,
            description="Block national ID numbers",
        ),
        # Financial - REDACT
        PIIType.IBAN: EnforcementPolicy(
            pii_type=PIIType.IBAN,
            action=EnforcementAction.REDACT,
            min_confidence=0.85,
            notify=True,
            description="Redact IBAN numbers",
        ),
        # Dates
        PIIType.DATE_OF_BIRTH: EnforcementPolicy(
            pii_type=PIIType.DATE_OF_BIRTH,
            action=EnforcementAction.ALLOW,
            min_confidence=0.7,
            notify=False,
            description="Allow dates of birth through",
        ),
        # Username
        PIIType.USERNAME: EnforcementPolicy(
            pii_type=PIIType.USERNAME,
            action=EnforcementAction.ALLOW,
            min_confidence=0.7,
            notify=False,
            description="Allow usernames through",
        ),
        # Custom
        PIIType.CUSTOM: EnforcementPolicy(
            pii_type=PIIType.CUSTOM,
            action=EnforcementAction.ALLOW,
            min_confidence=0.8,
            notify=True,
            description="Custom PII type - review required",
        ),
    }


# Create module-level default policies
DEFAULT_POLICIES: Dict[PIIType, EnforcementPolicy] = _create_default_policies()


def get_default_policy(pii_type: PIIType) -> EnforcementPolicy:
    """Get the default policy for a PII type.

    Args:
        pii_type: The PII type to get policy for.

    Returns:
        The default EnforcementPolicy for this type.
    """
    if pii_type in DEFAULT_POLICIES:
        return DEFAULT_POLICIES[pii_type]

    # Fallback policy for unknown types
    logger.warning("No default policy for PII type: %s", pii_type)
    return EnforcementPolicy(
        pii_type=pii_type,
        action=EnforcementAction.ALLOW,
        min_confidence=0.8,
        notify=True,
        description=f"Fallback policy for {pii_type.value}",
    )


# ---------------------------------------------------------------------------
# Exports
# ---------------------------------------------------------------------------

__all__ = [
    "PIIType",
    "EnforcementAction",
    "TransformationType",
    "ContextType",
    "EnforcementPolicy",
    "EnforcementContext",
    "DEFAULT_POLICIES",
    "get_default_policy",
]
