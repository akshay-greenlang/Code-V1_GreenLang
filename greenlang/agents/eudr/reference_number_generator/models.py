# -*- coding: utf-8 -*-
"""
Reference Number Generator Models - AGENT-EUDR-038

Pydantic v2 models for EUDR reference number generation, validation,
batch processing, lifecycle management, collision detection, format
compliance, sequence tracking, and verification.

All models use string types for reference numbers (immutable identifiers)
and integer types for sequence counters. Deterministic checksum
computation ensures zero-hallucination compliance.

Author: GreenLang Platform Team
Date: March 2026
PRD: PRD-AGENT-EUDR-038 Reference Number Generator (GL-EUDR-RNG-038)
Regulation: EU 2023/1115 (EUDR) Articles 4, 9, 33
Status: Production Ready
"""
from __future__ import annotations

import enum
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from pydantic import Field
from greenlang.schemas import GreenLangBase


# ---------------------------------------------------------------------------
# Enums (12)
# ---------------------------------------------------------------------------


class ReferenceNumberStatus(str, enum.Enum):
    """Lifecycle status of a reference number."""

    RESERVED = "reserved"
    ACTIVE = "active"
    USED = "used"
    EXPIRED = "expired"
    REVOKED = "revoked"
    TRANSFERRED = "transferred"
    CANCELLED = "cancelled"


class MemberStateCode(str, enum.Enum):
    """EU member state ISO 3166-1 alpha-2 codes (27 states)."""

    AT = "AT"  # Austria
    BE = "BE"  # Belgium
    BG = "BG"  # Bulgaria
    HR = "HR"  # Croatia
    CY = "CY"  # Cyprus
    CZ = "CZ"  # Czechia
    DK = "DK"  # Denmark
    EE = "EE"  # Estonia
    FI = "FI"  # Finland
    FR = "FR"  # France
    DE = "DE"  # Germany
    GR = "GR"  # Greece
    HU = "HU"  # Hungary
    IE = "IE"  # Ireland
    IT = "IT"  # Italy
    LV = "LV"  # Latvia
    LT = "LT"  # Lithuania
    LU = "LU"  # Luxembourg
    MT = "MT"  # Malta
    NL = "NL"  # Netherlands
    PL = "PL"  # Poland
    PT = "PT"  # Portugal
    RO = "RO"  # Romania
    SK = "SK"  # Slovakia
    SI = "SI"  # Slovenia
    ES = "ES"  # Spain
    SE = "SE"  # Sweden


class ChecksumAlgorithm(str, enum.Enum):
    """Supported checksum algorithms for reference number validation."""

    LUHN = "luhn"
    ISO7064 = "iso7064"
    CRC16 = "crc16"
    MODULO97 = "modulo97"


class ValidationResult(str, enum.Enum):
    """Result of reference number format validation."""

    VALID = "valid"
    INVALID_FORMAT = "invalid_format"
    INVALID_CHECKSUM = "invalid_checksum"
    INVALID_MEMBER_STATE = "invalid_member_state"
    INVALID_SEQUENCE = "invalid_sequence"
    EXPIRED = "expired"
    REVOKED = "revoked"
    NOT_FOUND = "not_found"
    UNKNOWN = "unknown"


class BatchStatus(str, enum.Enum):
    """Status of a batch generation request."""

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    PARTIAL = "partial"
    FAILED = "failed"
    CANCELLED = "cancelled"
    TIMEOUT = "timeout"


class TransferReason(str, enum.Enum):
    """Reasons for transferring a reference number between operators."""

    OWNERSHIP_CHANGE = "ownership_change"
    MERGER_ACQUISITION = "merger_acquisition"
    OPERATIONAL_TRANSFER = "operational_transfer"
    REGULATORY_REASSIGNMENT = "regulatory_reassignment"
    ERROR_CORRECTION = "error_correction"


class RevocationReason(str, enum.Enum):
    """Reasons for revoking a reference number."""

    FRAUD = "fraud"
    NON_COMPLIANCE = "non_compliance"
    DUPLICATE = "duplicate"
    DATA_ERROR = "data_error"
    OPERATOR_REQUEST = "operator_request"
    REGULATORY_ORDER = "regulatory_order"
    SYSTEM_ERROR = "system_error"


class SequenceOverflowStrategy(str, enum.Enum):
    """Strategy when sequence counter reaches maximum value."""

    EXTEND = "extend"
    REJECT = "reject"
    ROLLOVER = "rollover"


class FormatVersion(str, enum.Enum):
    """Reference number format version identifiers."""

    V1_0 = "1.0"
    V1_1 = "1.1"
    V2_0 = "2.0"


class ValidatorType(str, enum.Enum):
    """Types of validation checks performed on reference numbers."""

    FORMAT = "format"
    CHECKSUM = "checksum"
    MEMBER_STATE = "member_state"
    SEQUENCE = "sequence"
    EXPIRATION = "expiration"
    EXISTENCE = "existence"
    LIFECYCLE = "lifecycle"


class AuditAction(str, enum.Enum):
    """Audit trail action types for reference number lifecycle events."""

    GENERATE = "generate"
    VALIDATE = "validate"
    ACTIVATE = "activate"
    USE = "use"
    EXPIRE = "expire"
    REVOKE = "revoke"
    TRANSFER = "transfer"
    CANCEL = "cancel"
    BATCH_GENERATE = "batch_generate"
    VERIFY = "verify"


class GenerationMode(str, enum.Enum):
    """Mode of reference number generation."""

    SINGLE = "single"
    BATCH = "batch"
    RESERVED = "reserved"
    SEQUENTIAL = "sequential"
    IDEMPOTENT = "idempotent"


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

AGENT_ID = "GL-EUDR-RNG-038"
AGENT_VERSION = "1.0.0"

VALID_MEMBER_STATES: List[str] = [ms.value for ms in MemberStateCode]

EUDR_COMMODITIES: List[str] = [
    "cattle", "cocoa", "coffee", "palm_oil",
    "rubber", "soya", "wood",
]


# ---------------------------------------------------------------------------
# Pydantic Models (15+)
# ---------------------------------------------------------------------------


class ReferenceNumberComponents(GreenLangBase):
    """Decomposed components of a reference number.

    Breaks down a reference number string into its constituent parts
    for validation, formatting, and reconstruction purposes.
    """

    prefix: str = Field(
        ..., description="Reference number prefix (e.g., 'EUDR')"
    )
    member_state: str = Field(
        ..., min_length=2, max_length=2,
        description="EU member state ISO 3166-1 alpha-2 code",
    )
    year: int = Field(
        ..., ge=2024, le=2099,
        description="Year of generation (4-digit)",
    )
    operator_code: str = Field(
        ..., description="Operator identification code"
    )
    sequence: int = Field(
        ..., ge=0, description="Sequential number within operator/year"
    )
    checksum: str = Field(
        ..., description="Checksum digit(s) for integrity validation"
    )

    model_config = {"frozen": False, "extra": "ignore"}


class ReferenceNumber(GreenLangBase):
    """A generated EUDR reference number with metadata.

    Represents a single reference number including its full string
    representation, decomposed components, lifecycle status, and
    provenance information.
    """

    reference_id: str = Field(
        ..., description="Internal unique identifier (UUID)"
    )
    reference_number: str = Field(
        ..., description="Full reference number string"
    )
    components: ReferenceNumberComponents = Field(
        ..., description="Decomposed reference number components"
    )
    operator_id: str = Field(
        ..., description="Operator who owns this reference number"
    )
    commodity: Optional[str] = Field(
        default=None, description="EUDR commodity associated with this reference"
    )
    status: ReferenceNumberStatus = ReferenceNumberStatus.ACTIVE
    format_version: str = Field(
        default="1.0", description="Format version used for generation"
    )
    checksum_algorithm: str = Field(
        default="luhn", description="Algorithm used for checksum"
    )
    generated_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    expires_at: Optional[datetime] = Field(
        default=None, description="Expiration timestamp"
    )
    used_at: Optional[datetime] = Field(
        default=None, description="Timestamp when marked as used"
    )
    revoked_at: Optional[datetime] = Field(
        default=None, description="Timestamp when revoked"
    )
    provenance_hash: str = Field(
        default="", description="SHA-256 provenance hash"
    )

    model_config = {"frozen": False, "extra": "ignore"}


class SequenceCounter(GreenLangBase):
    """Sequence counter state for an operator within a year.

    Tracks the current position of the sequential numbering for
    a specific operator and year combination, supporting atomic
    increment operations.
    """

    counter_id: str = Field(
        ..., description="Unique counter identifier"
    )
    operator_id: str = Field(
        ..., description="Operator identifier"
    )
    member_state: str = Field(
        ..., min_length=2, max_length=2,
        description="EU member state code",
    )
    year: int = Field(
        ..., ge=2024, le=2099,
        description="Sequence year",
    )
    current_value: int = Field(
        ..., ge=0, description="Current sequence value"
    )
    max_value: int = Field(
        ..., ge=1, description="Maximum allowed sequence value"
    )
    reserved_count: int = Field(
        default=0, ge=0, description="Number of reserved (pre-allocated) slots"
    )
    overflow_strategy: SequenceOverflowStrategy = (
        SequenceOverflowStrategy.EXTEND
    )
    last_incremented_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )

    model_config = {"frozen": False, "extra": "ignore"}


class BatchRequest(GreenLangBase):
    """Request for batch reference number generation.

    Specifies the parameters for generating multiple reference
    numbers in a single batch operation.
    """

    batch_id: str = Field(
        ..., description="Unique batch identifier"
    )
    operator_id: str = Field(
        ..., description="Operator requesting the batch"
    )
    member_state: str = Field(
        ..., min_length=2, max_length=2,
        description="EU member state code",
    )
    commodity: Optional[str] = Field(
        default=None, description="EUDR commodity"
    )
    count: int = Field(
        ..., ge=1, le=10000,
        description="Number of reference numbers to generate",
    )
    status: BatchStatus = BatchStatus.PENDING
    generated_count: int = Field(
        default=0, ge=0, description="Number successfully generated"
    )
    failed_count: int = Field(
        default=0, ge=0, description="Number that failed generation"
    )
    reference_numbers: List[str] = Field(
        default_factory=list,
        description="Generated reference number strings",
    )
    requested_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    completed_at: Optional[datetime] = None
    error_message: Optional[str] = None

    model_config = {"frozen": False, "extra": "ignore"}


class ValidationLog(GreenLangBase):
    """Record of a reference number validation check.

    Captures the outcome of validating a reference number including
    all checks performed and their individual results.
    """

    validation_id: str = Field(
        ..., description="Unique validation identifier"
    )
    reference_number: str = Field(
        ..., description="Reference number that was validated"
    )
    result: ValidationResult = Field(
        ..., description="Overall validation result"
    )
    checks_performed: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Individual check results",
    )
    is_valid: bool = Field(
        ..., description="True if all checks passed"
    )
    validated_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    validated_by: str = Field(
        default=AGENT_ID, description="Validator identity"
    )

    model_config = {"frozen": False, "extra": "ignore"}


class FormatRule(GreenLangBase):
    """Format rule for a specific member state.

    Defines the reference number format requirements for a
    particular EU member state, including custom prefixes,
    separator characters, and sequence constraints.
    """

    member_state: str = Field(
        ..., min_length=2, max_length=2,
        description="EU member state code",
    )
    country_name: str = Field(
        ..., description="Full country name"
    )
    prefix: str = Field(
        default="EUDR", description="Reference number prefix"
    )
    separator: str = Field(
        default="-", description="Component separator character"
    )
    sequence_digits: int = Field(
        default=6, ge=4, le=10,
        description="Number of digits in sequence component",
    )
    checksum_algorithm: str = Field(
        default="luhn", description="Checksum algorithm"
    )
    format_version: str = Field(
        default="1.0", description="Format version"
    )
    example: str = Field(
        default="", description="Example reference number"
    )

    model_config = {"frozen": False, "extra": "ignore"}


class CollisionRecord(GreenLangBase):
    """Record of a detected reference number collision.

    Captures information about an attempted duplicate reference
    number generation, including resolution details.
    """

    collision_id: str = Field(
        ..., description="Unique collision record identifier"
    )
    reference_number: str = Field(
        ..., description="Reference number that caused the collision"
    )
    operator_id: str = Field(
        ..., description="Operator involved"
    )
    attempt_number: int = Field(
        ..., ge=1, description="Retry attempt number"
    )
    resolved: bool = Field(
        default=False, description="Whether collision was resolved"
    )
    resolution_method: str = Field(
        default="", description="How the collision was resolved"
    )
    detected_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )

    model_config = {"frozen": False, "extra": "ignore"}


class TransferRecord(GreenLangBase):
    """Record of a reference number transfer between operators.

    Captures the transfer of ownership of a reference number from
    one operator to another, including reason and authorization.
    """

    transfer_id: str = Field(
        ..., description="Unique transfer record identifier"
    )
    reference_number: str = Field(
        ..., description="Reference number being transferred"
    )
    from_operator_id: str = Field(
        ..., description="Original operator (sender)"
    )
    to_operator_id: str = Field(
        ..., description="New operator (receiver)"
    )
    reason: TransferReason = Field(
        ..., description="Reason for transfer"
    )
    authorized_by: str = Field(
        ..., description="Identity authorizing the transfer"
    )
    transferred_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    provenance_hash: str = Field(
        default="", description="SHA-256 hash of transfer record"
    )

    model_config = {"frozen": False, "extra": "ignore"}


class GenerationRequest(GreenLangBase):
    """API request model for single reference number generation."""

    operator_id: str = Field(
        ..., description="Operator identifier"
    )
    member_state: str = Field(
        ..., min_length=2, max_length=2,
        description="EU member state code (ISO 3166-1 alpha-2)",
    )
    commodity: Optional[str] = Field(
        default=None, description="EUDR commodity"
    )
    idempotency_key: Optional[str] = Field(
        default=None,
        description="Idempotency key for retry safety",
    )

    model_config = {"frozen": False, "extra": "ignore"}


class GenerationResponse(GreenLangBase):
    """API response model for single reference number generation."""

    reference_id: str = Field(
        ..., description="Internal unique identifier"
    )
    reference_number: str = Field(
        ..., description="Generated reference number string"
    )
    operator_id: str = Field(
        ..., description="Owner operator"
    )
    member_state: str = Field(
        ..., description="Member state code"
    )
    status: ReferenceNumberStatus = ReferenceNumberStatus.ACTIVE
    format_version: str = Field(
        default="1.0", description="Format version"
    )
    checksum_algorithm: str = Field(
        default="luhn", description="Checksum algorithm used"
    )
    generated_at: str = Field(
        ..., description="Generation timestamp (ISO 8601)"
    )
    expires_at: Optional[str] = Field(
        default=None, description="Expiration timestamp (ISO 8601)"
    )
    provenance_hash: str = Field(
        default="", description="SHA-256 provenance hash"
    )

    model_config = {"frozen": False, "extra": "ignore"}


class BatchGenerationRequest(GreenLangBase):
    """API request model for batch reference number generation."""

    operator_id: str = Field(
        ..., description="Operator identifier"
    )
    member_state: str = Field(
        ..., min_length=2, max_length=2,
        description="EU member state code",
    )
    count: int = Field(
        ..., ge=1, le=10000,
        description="Number of reference numbers to generate",
    )
    commodity: Optional[str] = Field(
        default=None, description="EUDR commodity"
    )

    model_config = {"frozen": False, "extra": "ignore"}


class ValidationRequest(GreenLangBase):
    """API request model for reference number validation."""

    reference_number: str = Field(
        ..., description="Reference number to validate"
    )
    check_existence: bool = Field(
        default=True,
        description="Whether to check database for existence",
    )
    check_lifecycle: bool = Field(
        default=True,
        description="Whether to check lifecycle status",
    )

    model_config = {"frozen": False, "extra": "ignore"}


class ValidationResponse(GreenLangBase):
    """API response model for reference number validation."""

    reference_number: str = Field(
        ..., description="Reference number validated"
    )
    is_valid: bool = Field(
        ..., description="Overall validity"
    )
    result: ValidationResult = Field(
        ..., description="Detailed validation result"
    )
    checks: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Individual check results",
    )
    status: Optional[ReferenceNumberStatus] = Field(
        default=None, description="Current lifecycle status (if found)"
    )
    validated_at: str = Field(
        ..., description="Validation timestamp (ISO 8601)"
    )

    model_config = {"frozen": False, "extra": "ignore"}


class SequenceStatus(GreenLangBase):
    """Status summary for an operator's sequence counter."""

    operator_id: str = Field(
        ..., description="Operator identifier"
    )
    member_state: str = Field(
        ..., description="Member state code"
    )
    year: int = Field(
        ..., description="Sequence year"
    )
    current_value: int = Field(
        ..., description="Current sequence value"
    )
    max_value: int = Field(
        ..., description="Maximum sequence value"
    )
    available: int = Field(
        ..., description="Remaining available sequences"
    )
    utilization_percent: float = Field(
        ..., ge=0.0, le=100.0,
        description="Sequence utilization percentage",
    )
    overflow_strategy: str = Field(
        ..., description="Strategy when max is reached"
    )

    model_config = {"frozen": False, "extra": "ignore"}


class FormatTemplate(GreenLangBase):
    """Template defining the reference number format structure."""

    template_id: str = Field(
        ..., description="Unique template identifier"
    )
    version: str = Field(
        ..., description="Format version"
    )
    pattern: str = Field(
        ..., description="Format pattern with placeholders"
    )
    components: List[str] = Field(
        default_factory=list,
        description="Ordered list of component names",
    )
    regex_pattern: str = Field(
        default="", description="Regex pattern for validation"
    )
    example: str = Field(
        default="", description="Example reference number"
    )

    model_config = {"frozen": False, "extra": "ignore"}


class HealthStatus(GreenLangBase):
    """Health check response for the Reference Number Generator."""

    agent_id: str = AGENT_ID
    status: str = "healthy"
    version: str = AGENT_VERSION
    engines: Dict[str, str] = Field(default_factory=dict)
    database: bool = False
    redis: bool = False
    uptime_seconds: float = 0.0
    active_references: int = 0
    total_generated: int = 0

    model_config = {"frozen": False, "extra": "ignore"}
