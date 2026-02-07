# -*- coding: utf-8 -*-
"""
PII Allowlist Patterns - SEC-011 PII Detection/Redaction Enhancements

Pattern definitions for PII allowlists. Provides AllowlistEntry model and
DEFAULT_ALLOWLISTS dictionary containing pre-configured safe patterns for
known test data, reserved domains, and placeholder values.

Supported Pattern Types:
    - regex: Regular expression matching
    - exact: Exact string matching
    - prefix: String prefix matching
    - suffix: String suffix matching
    - contains: Substring matching

The default allowlists cover:
    - RFC 2606 reserved domains (example.com, test.com)
    - US fictional phone numbers (555-xxxx)
    - Stripe/payment test cards
    - Invalid SSN placeholders
    - Private IP address ranges

Author: GreenLang Framework Team
Date: February 2026
PRD: SEC-011 PII Detection/Redaction Enhancements
Status: Production Ready
"""

from __future__ import annotations

import logging
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional
from uuid import UUID, uuid4

from pydantic import BaseModel, Field, field_validator

# Import PIIType from existing PII scanner
from greenlang.infrastructure.security_scanning.pii_scanner import PIIType

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Pattern Type Enum
# ---------------------------------------------------------------------------


class PatternType(str, Enum):
    """Allowlist pattern matching types."""

    REGEX = "regex"
    EXACT = "exact"
    PREFIX = "prefix"
    SUFFIX = "suffix"
    CONTAINS = "contains"


# ---------------------------------------------------------------------------
# Allowlist Entry Model
# ---------------------------------------------------------------------------


class AllowlistEntry(BaseModel):
    """Single allowlist entry for PII detection exclusion.

    Represents a pattern that should be excluded from PII detection.
    Entries can be global (tenant_id=None) or tenant-specific.

    Attributes:
        id: Unique identifier for this entry.
        pii_type: Type of PII this entry applies to.
        pattern: The pattern to match (regex string or literal).
        pattern_type: How to interpret the pattern (regex, exact, etc.).
        reason: Human-readable explanation for this allowlist entry.
        created_by: UUID of user who created this entry (None for system defaults).
        created_at: When this entry was created.
        expires_at: Optional expiration timestamp.
        tenant_id: Tenant this applies to (None = global).
        enabled: Whether this entry is active.
        metadata: Additional metadata for this entry.

    Example:
        >>> entry = AllowlistEntry(
        ...     pii_type=PIIType.EMAIL,
        ...     pattern=r".*@example\\.com$",
        ...     pattern_type="regex",
        ...     reason="RFC 2606 reserved domain"
        ... )
        >>> entry.is_valid()
        True
    """

    id: UUID = Field(default_factory=uuid4, description="Unique entry identifier")
    pii_type: PIIType = Field(..., description="PII type this entry applies to")
    pattern: str = Field(..., min_length=1, description="Pattern to match")
    pattern_type: PatternType = Field(
        default=PatternType.REGEX,
        description="Pattern matching type"
    )
    reason: str = Field(..., min_length=1, description="Reason for allowlisting")
    created_by: Optional[UUID] = Field(
        default=None,
        description="User who created this entry (None for system defaults)"
    )
    created_at: datetime = Field(
        default_factory=datetime.utcnow,
        description="Creation timestamp"
    )
    expires_at: Optional[datetime] = Field(
        default=None,
        description="Optional expiration timestamp"
    )
    tenant_id: Optional[str] = Field(
        default=None,
        description="Tenant ID (None = global allowlist)"
    )
    enabled: bool = Field(default=True, description="Whether entry is active")
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata"
    )

    model_config = {"frozen": False, "validate_assignment": True}

    @field_validator("pattern")
    @classmethod
    def validate_pattern_not_empty(cls, v: str) -> str:
        """Validate pattern is not empty or whitespace only."""
        if not v or not v.strip():
            raise ValueError("Pattern cannot be empty or whitespace only")
        return v

    @field_validator("reason")
    @classmethod
    def validate_reason_not_empty(cls, v: str) -> str:
        """Validate reason is not empty or whitespace only."""
        if not v or not v.strip():
            raise ValueError("Reason cannot be empty or whitespace only")
        return v

    def is_expired(self) -> bool:
        """Check if this entry has expired.

        Returns:
            True if expires_at is set and has passed.
        """
        if self.expires_at is None:
            return False
        return datetime.utcnow() > self.expires_at

    def is_active(self) -> bool:
        """Check if this entry is currently active.

        Returns:
            True if enabled and not expired.
        """
        return self.enabled and not self.is_expired()

    def matches_tenant(self, tenant_id: Optional[str]) -> bool:
        """Check if this entry applies to a given tenant.

        Global entries (tenant_id=None) match all tenants.

        Args:
            tenant_id: Tenant to check against.

        Returns:
            True if this entry applies to the tenant.
        """
        # Global entries match all tenants
        if self.tenant_id is None:
            return True
        # Tenant-specific entries only match their tenant
        return self.tenant_id == tenant_id


# ---------------------------------------------------------------------------
# Default Allowlists
# ---------------------------------------------------------------------------


def _create_default_allowlists() -> Dict[PIIType, List[AllowlistEntry]]:
    """Create the default allowlist entries.

    These entries are loaded at startup and cover common test data patterns,
    reserved domains, and known safe values.

    Returns:
        Dictionary mapping PIIType to list of AllowlistEntry objects.
    """
    defaults: Dict[PIIType, List[AllowlistEntry]] = {}

    # -------------------------------------------------------------------------
    # Email Allowlists
    # -------------------------------------------------------------------------
    defaults[PIIType.EMAIL] = [
        AllowlistEntry(
            pii_type=PIIType.EMAIL,
            pattern=r".*@example\.(com|org|net)$",
            pattern_type=PatternType.REGEX,
            reason="RFC 2606 reserved domain"
        ),
        AllowlistEntry(
            pii_type=PIIType.EMAIL,
            pattern=r".*@test\.(com|org|net)$",
            pattern_type=PatternType.REGEX,
            reason="Test domain"
        ),
        AllowlistEntry(
            pii_type=PIIType.EMAIL,
            pattern=r".*@localhost$",
            pattern_type=PatternType.REGEX,
            reason="Localhost domain"
        ),
        AllowlistEntry(
            pii_type=PIIType.EMAIL,
            pattern=r"^noreply@.*$",
            pattern_type=PatternType.REGEX,
            reason="No-reply addresses are not personal"
        ),
        AllowlistEntry(
            pii_type=PIIType.EMAIL,
            pattern=r"^no-reply@.*$",
            pattern_type=PatternType.REGEX,
            reason="No-reply addresses are not personal"
        ),
        AllowlistEntry(
            pii_type=PIIType.EMAIL,
            pattern=r".*@greenlang\.io$",
            pattern_type=PatternType.REGEX,
            reason="Internal GreenLang domain"
        ),
        AllowlistEntry(
            pii_type=PIIType.EMAIL,
            pattern=r".*@invalid$",
            pattern_type=PatternType.REGEX,
            reason="RFC 2606 reserved TLD"
        ),
        AllowlistEntry(
            pii_type=PIIType.EMAIL,
            pattern=r".*@mailinator\.com$",
            pattern_type=PatternType.REGEX,
            reason="Disposable email service commonly used for testing"
        ),
    ]

    # -------------------------------------------------------------------------
    # Phone Number Allowlists
    # -------------------------------------------------------------------------
    defaults[PIIType.PHONE] = [
        AllowlistEntry(
            pii_type=PIIType.PHONE,
            pattern=r"555-\d{4}$",
            pattern_type=PatternType.REGEX,
            reason="US fictional phone numbers (555 exchange)"
        ),
        AllowlistEntry(
            pii_type=PIIType.PHONE,
            pattern=r"\+1-555-\d{3}-\d{4}$",
            pattern_type=PatternType.REGEX,
            reason="US fictional phone full format"
        ),
        AllowlistEntry(
            pii_type=PIIType.PHONE,
            pattern=r"\(555\)\s?\d{3}-\d{4}$",
            pattern_type=PatternType.REGEX,
            reason="US fictional phone with parentheses"
        ),
        AllowlistEntry(
            pii_type=PIIType.PHONE,
            pattern="000-000-0000",
            pattern_type=PatternType.EXACT,
            reason="Placeholder phone number"
        ),
        AllowlistEntry(
            pii_type=PIIType.PHONE,
            pattern="123-456-7890",
            pattern_type=PatternType.EXACT,
            reason="Common test phone number"
        ),
        AllowlistEntry(
            pii_type=PIIType.PHONE,
            pattern="+1 (555) 555-5555",
            pattern_type=PatternType.EXACT,
            reason="Common fictional phone"
        ),
    ]

    # -------------------------------------------------------------------------
    # Credit Card Allowlists (Test Cards)
    # -------------------------------------------------------------------------
    defaults[PIIType.CREDIT_CARD] = [
        # Stripe test cards
        AllowlistEntry(
            pii_type=PIIType.CREDIT_CARD,
            pattern="4111111111111111",
            pattern_type=PatternType.EXACT,
            reason="Stripe test Visa card"
        ),
        AllowlistEntry(
            pii_type=PIIType.CREDIT_CARD,
            pattern="4242424242424242",
            pattern_type=PatternType.EXACT,
            reason="Stripe test Visa card"
        ),
        AllowlistEntry(
            pii_type=PIIType.CREDIT_CARD,
            pattern="5555555555554444",
            pattern_type=PatternType.EXACT,
            reason="Stripe test Mastercard"
        ),
        AllowlistEntry(
            pii_type=PIIType.CREDIT_CARD,
            pattern="378282246310005",
            pattern_type=PatternType.EXACT,
            reason="Stripe test American Express"
        ),
        AllowlistEntry(
            pii_type=PIIType.CREDIT_CARD,
            pattern="6011111111111117",
            pattern_type=PatternType.EXACT,
            reason="Stripe test Discover"
        ),
        # PayPal sandbox test cards
        AllowlistEntry(
            pii_type=PIIType.CREDIT_CARD,
            pattern="4012888888881881",
            pattern_type=PatternType.EXACT,
            reason="PayPal sandbox test Visa"
        ),
        # Braintree test cards
        AllowlistEntry(
            pii_type=PIIType.CREDIT_CARD,
            pattern="4111111111111115",
            pattern_type=PatternType.EXACT,
            reason="Braintree test decline card"
        ),
        AllowlistEntry(
            pii_type=PIIType.CREDIT_CARD,
            pattern="5105105105105100",
            pattern_type=PatternType.EXACT,
            reason="Mastercard test card"
        ),
        # Placeholder cards
        AllowlistEntry(
            pii_type=PIIType.CREDIT_CARD,
            pattern="0000000000000000",
            pattern_type=PatternType.EXACT,
            reason="Placeholder card number"
        ),
    ]

    # -------------------------------------------------------------------------
    # SSN Allowlists
    # -------------------------------------------------------------------------
    defaults[PIIType.SSN] = [
        AllowlistEntry(
            pii_type=PIIType.SSN,
            pattern="000-00-0000",
            pattern_type=PatternType.EXACT,
            reason="Invalid SSN (area number 000)"
        ),
        AllowlistEntry(
            pii_type=PIIType.SSN,
            pattern="123-45-6789",
            pattern_type=PatternType.EXACT,
            reason="Common test/example SSN"
        ),
        AllowlistEntry(
            pii_type=PIIType.SSN,
            pattern="111-11-1111",
            pattern_type=PatternType.EXACT,
            reason="Common placeholder SSN"
        ),
        AllowlistEntry(
            pii_type=PIIType.SSN,
            pattern="999-99-9999",
            pattern_type=PatternType.EXACT,
            reason="Invalid SSN (area number 999)"
        ),
        AllowlistEntry(
            pii_type=PIIType.SSN,
            pattern=r"^9\d{2}-\d{2}-\d{4}$",
            pattern_type=PatternType.REGEX,
            reason="ITIN range (not SSN) - 9xx-xx-xxxx"
        ),
        AllowlistEntry(
            pii_type=PIIType.SSN,
            pattern=r"^666-\d{2}-\d{4}$",
            pattern_type=PatternType.REGEX,
            reason="Invalid SSN (area number 666 never assigned)"
        ),
    ]

    # -------------------------------------------------------------------------
    # IP Address Allowlists
    # -------------------------------------------------------------------------
    defaults[PIIType.IP_ADDRESS] = [
        AllowlistEntry(
            pii_type=PIIType.IP_ADDRESS,
            pattern=r"^127\.\d{1,3}\.\d{1,3}\.\d{1,3}$",
            pattern_type=PatternType.REGEX,
            reason="Localhost/loopback addresses"
        ),
        AllowlistEntry(
            pii_type=PIIType.IP_ADDRESS,
            pattern=r"^10\.\d{1,3}\.\d{1,3}\.\d{1,3}$",
            pattern_type=PatternType.REGEX,
            reason="Private Class A addresses (RFC 1918)"
        ),
        AllowlistEntry(
            pii_type=PIIType.IP_ADDRESS,
            pattern=r"^192\.168\.\d{1,3}\.\d{1,3}$",
            pattern_type=PatternType.REGEX,
            reason="Private Class C addresses (RFC 1918)"
        ),
        AllowlistEntry(
            pii_type=PIIType.IP_ADDRESS,
            pattern=r"^172\.(1[6-9]|2\d|3[01])\.\d{1,3}\.\d{1,3}$",
            pattern_type=PatternType.REGEX,
            reason="Private Class B addresses (RFC 1918)"
        ),
        AllowlistEntry(
            pii_type=PIIType.IP_ADDRESS,
            pattern="0.0.0.0",
            pattern_type=PatternType.EXACT,
            reason="Unspecified address"
        ),
        AllowlistEntry(
            pii_type=PIIType.IP_ADDRESS,
            pattern="255.255.255.255",
            pattern_type=PatternType.EXACT,
            reason="Broadcast address"
        ),
        AllowlistEntry(
            pii_type=PIIType.IP_ADDRESS,
            pattern=r"^169\.254\.\d{1,3}\.\d{1,3}$",
            pattern_type=PatternType.REGEX,
            reason="Link-local addresses (APIPA)"
        ),
    ]

    # -------------------------------------------------------------------------
    # API Key Allowlists
    # -------------------------------------------------------------------------
    defaults[PIIType.API_KEY] = [
        AllowlistEntry(
            pii_type=PIIType.API_KEY,
            pattern=r"^test[_-]?key[_-]?\d*$",
            pattern_type=PatternType.REGEX,
            reason="Test API key placeholder"
        ),
        AllowlistEntry(
            pii_type=PIIType.API_KEY,
            pattern=r"^sk_test_[a-zA-Z0-9]+$",
            pattern_type=PatternType.REGEX,
            reason="Stripe test secret key"
        ),
        AllowlistEntry(
            pii_type=PIIType.API_KEY,
            pattern=r"^pk_test_[a-zA-Z0-9]+$",
            pattern_type=PatternType.REGEX,
            reason="Stripe test publishable key"
        ),
        AllowlistEntry(
            pii_type=PIIType.API_KEY,
            pattern="your-api-key-here",
            pattern_type=PatternType.EXACT,
            reason="Documentation placeholder"
        ),
        AllowlistEntry(
            pii_type=PIIType.API_KEY,
            pattern="REPLACE_WITH_YOUR_KEY",
            pattern_type=PatternType.EXACT,
            reason="Documentation placeholder"
        ),
    ]

    # -------------------------------------------------------------------------
    # Password Allowlists (for testing only)
    # -------------------------------------------------------------------------
    defaults[PIIType.PASSWORD] = [
        AllowlistEntry(
            pii_type=PIIType.PASSWORD,
            pattern="password123",
            pattern_type=PatternType.EXACT,
            reason="Common test password"
        ),
        AllowlistEntry(
            pii_type=PIIType.PASSWORD,
            pattern="changeme",
            pattern_type=PatternType.EXACT,
            reason="Placeholder password"
        ),
        AllowlistEntry(
            pii_type=PIIType.PASSWORD,
            pattern="test1234",
            pattern_type=PatternType.EXACT,
            reason="Test password"
        ),
        AllowlistEntry(
            pii_type=PIIType.PASSWORD,
            pattern=r"^\*{4,}$",
            pattern_type=PatternType.REGEX,
            reason="Masked password (asterisks)"
        ),
        AllowlistEntry(
            pii_type=PIIType.PASSWORD,
            pattern=r"^x{4,}$",
            pattern_type=PatternType.REGEX,
            reason="Masked password (x characters)"
        ),
    ]

    return defaults


# Module-level default allowlists
DEFAULT_ALLOWLISTS: Dict[PIIType, List[AllowlistEntry]] = _create_default_allowlists()


def get_default_allowlist_count() -> int:
    """Get total count of default allowlist entries.

    Returns:
        Total number of default entries across all PII types.
    """
    return sum(len(entries) for entries in DEFAULT_ALLOWLISTS.values())


def get_allowlist_for_type(pii_type: PIIType) -> List[AllowlistEntry]:
    """Get default allowlist entries for a specific PII type.

    Args:
        pii_type: The PII type to get entries for.

    Returns:
        List of AllowlistEntry objects (empty list if none defined).
    """
    return DEFAULT_ALLOWLISTS.get(pii_type, [])


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

__all__ = [
    "PatternType",
    "AllowlistEntry",
    "DEFAULT_ALLOWLISTS",
    "get_default_allowlist_count",
    "get_allowlist_for_type",
]

logger.debug(
    "PII allowlist patterns loaded: %d entries across %d PII types",
    get_default_allowlist_count(),
    len(DEFAULT_ALLOWLISTS),
)
