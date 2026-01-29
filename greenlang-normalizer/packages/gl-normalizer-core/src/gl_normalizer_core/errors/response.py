"""
GLNORM Error Response Models for GL-FOUND-X-003.

This module provides Pydantic models for structured error responses in the
GreenLang Normalizer component. These models ensure consistent error formatting
across all API endpoints and internal error handling.

The response structure follows RFC 7807 (Problem Details for HTTP APIs) with
extensions for GreenLang-specific audit and remediation requirements.

Example:
    >>> from gl_normalizer_core.errors.response import GLNORMErrorResponse
    >>> error = GLNORMErrorResponse(
    ...     code="GLNORM-E100",
    ...     message="Failed to parse unit string",
    ...     details={"input": "kgCO2/kwh", "position": 5}
    ... )
    >>> print(error.model_dump_json(indent=2))
"""

from datetime import datetime
from decimal import Decimal
from typing import Any, Dict, List, Optional, Union
from uuid import UUID, uuid4

from pydantic import BaseModel, Field, field_validator, model_validator


class ErrorContext(BaseModel):
    """
    Contextual information about where the error occurred.

    This model captures the execution context when an error is raised,
    enabling debugging and audit trail reconstruction.

    Attributes:
        request_id: Unique identifier for the request
        trace_id: Distributed tracing identifier
        span_id: Span identifier within the trace
        operation: Name of the operation being performed
        input_hash: SHA-256 hash of the input data
        timestamp: When the error occurred
        component: Component name (always 'gl-normalizer')
        version: Component version
    """

    request_id: UUID = Field(
        default_factory=uuid4,
        description="Unique identifier for this request"
    )
    trace_id: Optional[str] = Field(
        default=None,
        description="Distributed tracing identifier (OpenTelemetry)"
    )
    span_id: Optional[str] = Field(
        default=None,
        description="Span identifier within the trace"
    )
    operation: str = Field(
        ...,
        description="Name of the operation that failed",
        examples=["normalize_unit", "convert_unit", "resolve_entity"]
    )
    input_hash: Optional[str] = Field(
        default=None,
        description="SHA-256 hash of input data for provenance",
        pattern=r"^[a-f0-9]{64}$"
    )
    timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="UTC timestamp when error occurred"
    )
    component: str = Field(
        default="gl-normalizer",
        description="Component name"
    )
    version: Optional[str] = Field(
        default=None,
        description="Component version (semver)"
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "request_id": "550e8400-e29b-41d4-a716-446655440000",
                "trace_id": "4bf92f3577b34da6a3ce929d0e0e4736",
                "span_id": "00f067aa0ba902b7",
                "operation": "normalize_unit",
                "input_hash": "a948904f2f0f479b8f8564cbf12dac6b18b1e8d4e8e7b0e6a5d9c2b1a0f3e4d5",
                "timestamp": "2024-01-15T10:30:00Z",
                "component": "gl-normalizer",
                "version": "1.2.3"
            }
        }
    }


class ErrorCandidate(BaseModel):
    """
    A candidate suggestion for resolving an ambiguous error.

    Used when multiple valid interpretations exist (e.g., E104_AMBIGUOUS_UNIT,
    E401_REFERENCE_AMBIGUOUS) to present options to the user.

    Attributes:
        value: The candidate value (unit, entity ID, etc.)
        label: Human-readable label for the candidate
        confidence: Confidence score (0.0-1.0)
        source: Source vocabulary or system
        metadata: Additional candidate-specific metadata
    """

    value: str = Field(
        ...,
        description="The candidate value"
    )
    label: str = Field(
        ...,
        description="Human-readable label"
    )
    confidence: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Confidence score (0.0-1.0)"
    )
    source: Optional[str] = Field(
        default=None,
        description="Source vocabulary or system"
    )
    metadata: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Additional candidate metadata"
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "value": "kg-co2e",
                "label": "Kilograms of CO2 equivalent",
                "confidence": 0.92,
                "source": "gl-vocab-units-v2",
                "metadata": {"dimension": "mass", "gwp_version": "AR6"}
            }
        }
    }


class ErrorSuggestion(BaseModel):
    """
    A suggestion for resolving or working around an error.

    Provides actionable remediation steps that can be automated or
    presented to users for manual resolution.

    Attributes:
        action: Action type (retry, provide_value, select_candidate, etc.)
        description: Human-readable description of the suggestion
        field: Field name that needs to be provided/corrected
        example: Example value for the field
        candidates: List of candidate values to choose from
        documentation_url: Link to relevant documentation
    """

    action: str = Field(
        ...,
        description="Action type for remediation",
        examples=["retry", "provide_value", "select_candidate", "review"]
    )
    description: str = Field(
        ...,
        description="Human-readable description of the suggestion"
    )
    field: Optional[str] = Field(
        default=None,
        description="Field name that needs to be provided or corrected"
    )
    example: Optional[str] = Field(
        default=None,
        description="Example value for the field"
    )
    candidates: Optional[List[ErrorCandidate]] = Field(
        default=None,
        description="List of candidate values to choose from"
    )
    documentation_url: Optional[str] = Field(
        default=None,
        description="URL to relevant documentation"
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "action": "provide_value",
                "description": "Provide the GWP version for CO2e conversion",
                "field": "gwp_version",
                "example": "AR6",
                "candidates": [
                    {"value": "AR6", "label": "IPCC AR6 (2021)", "confidence": None},
                    {"value": "AR5", "label": "IPCC AR5 (2014)", "confidence": None}
                ],
                "documentation_url": "https://docs.greenlang.io/normalizer/gwp"
            }
        }
    }


class ErrorDetail(BaseModel):
    """
    Detailed information about an error occurrence.

    Extends basic error information with full details needed for
    debugging, audit, and remediation.

    Attributes:
        code: GLNORM error code
        message: Human-readable error message
        category: Error category name
        severity: Error severity level
        details: Error-specific details
        context: Execution context
        suggestions: Remediation suggestions
        candidates: Candidate values for ambiguous errors
        related_errors: Related error codes
        is_recoverable: Whether error is potentially recoverable
        requires_human_review: Whether human review is recommended
    """

    code: str = Field(
        ...,
        description="GLNORM error code (e.g., GLNORM-E100)",
        pattern=r"^GLNORM-E\d{3}$"
    )
    message: str = Field(
        ...,
        description="Human-readable error message"
    )
    category: str = Field(
        ...,
        description="Error category name"
    )
    severity: str = Field(
        default="ERROR",
        description="Error severity level",
        pattern=r"^(CRITICAL|ERROR|WARNING|INFO)$"
    )
    details: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Error-specific details"
    )
    context: Optional[ErrorContext] = Field(
        default=None,
        description="Execution context"
    )
    suggestions: Optional[List[ErrorSuggestion]] = Field(
        default=None,
        description="Remediation suggestions"
    )
    candidates: Optional[List[ErrorCandidate]] = Field(
        default=None,
        description="Candidate values for ambiguous errors"
    )
    related_errors: Optional[List[str]] = Field(
        default=None,
        description="Related error codes"
    )
    is_recoverable: bool = Field(
        default=False,
        description="Whether error is potentially recoverable"
    )
    requires_human_review: bool = Field(
        default=False,
        description="Whether human review is recommended"
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "code": "GLNORM-E305",
                "message": "GWP version required for CO2e conversion",
                "category": "Conversion",
                "severity": "ERROR",
                "details": {
                    "source_unit": "kg-ch4",
                    "target_unit": "kg-co2e",
                    "missing_field": "gwp_version"
                },
                "is_recoverable": True,
                "requires_human_review": False
            }
        }
    }


class GLNORMErrorResponse(BaseModel):
    """
    Standard error response for GLNORM API endpoints.

    This is the top-level error response model returned by all GLNORM
    API endpoints. It follows RFC 7807 structure with GreenLang extensions.

    Attributes:
        success: Always False for error responses
        error: Detailed error information
        request_id: Request identifier for tracking
        timestamp: Response timestamp
        http_status: HTTP status code

    Example:
        >>> error = GLNORMErrorResponse(
        ...     error=ErrorDetail(
        ...         code="GLNORM-E100",
        ...         message="Failed to parse unit",
        ...         category="Unit Parsing"
        ...     )
        ... )
    """

    success: bool = Field(
        default=False,
        description="Always False for error responses"
    )
    error: ErrorDetail = Field(
        ...,
        description="Detailed error information"
    )
    request_id: UUID = Field(
        default_factory=uuid4,
        description="Unique request identifier"
    )
    timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="Response timestamp (UTC)"
    )
    http_status: int = Field(
        default=500,
        ge=400,
        le=599,
        description="HTTP status code"
    )

    @field_validator("success")
    @classmethod
    def success_must_be_false(cls, v: bool) -> bool:
        """Ensure success is always False for error responses."""
        if v is True:
            raise ValueError("success must be False for error responses")
        return v

    model_config = {
        "json_schema_extra": {
            "example": {
                "success": False,
                "error": {
                    "code": "GLNORM-E100",
                    "message": "Failed to parse unit string 'kgCO2/kwh'",
                    "category": "Unit Parsing",
                    "severity": "ERROR",
                    "details": {
                        "input": "kgCO2/kwh",
                        "position": 5,
                        "expected": "valid unit symbol"
                    },
                    "is_recoverable": False,
                    "requires_human_review": False
                },
                "request_id": "550e8400-e29b-41d4-a716-446655440000",
                "timestamp": "2024-01-15T10:30:00Z",
                "http_status": 400
            }
        }
    }


class GLNORMBatchErrorResponse(BaseModel):
    """
    Error response for batch operations.

    When processing multiple items, this response aggregates errors
    while still providing individual item results.

    Attributes:
        success: True if at least one item succeeded
        total_items: Total number of items in batch
        succeeded: Number of successfully processed items
        failed: Number of failed items
        errors: List of individual errors
        request_id: Batch request identifier
        timestamp: Response timestamp
    """

    success: bool = Field(
        ...,
        description="True if at least one item succeeded"
    )
    total_items: int = Field(
        ...,
        ge=0,
        description="Total number of items in batch"
    )
    succeeded: int = Field(
        ...,
        ge=0,
        description="Number of successfully processed items"
    )
    failed: int = Field(
        ...,
        ge=0,
        description="Number of failed items"
    )
    errors: List[ErrorDetail] = Field(
        default_factory=list,
        description="List of individual errors"
    )
    request_id: UUID = Field(
        default_factory=uuid4,
        description="Batch request identifier"
    )
    timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="Response timestamp (UTC)"
    )

    @model_validator(mode="after")
    def validate_counts(self) -> "GLNORMBatchErrorResponse":
        """Validate that counts are consistent."""
        if self.succeeded + self.failed != self.total_items:
            raise ValueError(
                f"succeeded ({self.succeeded}) + failed ({self.failed}) "
                f"must equal total_items ({self.total_items})"
            )
        return self

    model_config = {
        "json_schema_extra": {
            "example": {
                "success": True,
                "total_items": 100,
                "succeeded": 95,
                "failed": 5,
                "errors": [
                    {
                        "code": "GLNORM-E101",
                        "message": "Unknown unit 'xyz'",
                        "category": "Unit Parsing",
                        "severity": "ERROR",
                        "details": {"item_index": 12, "input": "xyz"}
                    }
                ],
                "request_id": "550e8400-e29b-41d4-a716-446655440000",
                "timestamp": "2024-01-15T10:30:00Z"
            }
        }
    }


class ValidationErrorItem(BaseModel):
    """
    Individual validation error item.

    Used for detailed field-level validation errors.

    Attributes:
        field: Field path (dot notation)
        message: Validation error message
        value: The invalid value
        constraint: The constraint that was violated
    """

    field: str = Field(
        ...,
        description="Field path using dot notation"
    )
    message: str = Field(
        ...,
        description="Validation error message"
    )
    value: Optional[Any] = Field(
        default=None,
        description="The invalid value"
    )
    constraint: Optional[str] = Field(
        default=None,
        description="The constraint that was violated"
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "field": "source_unit",
                "message": "Field is required",
                "value": None,
                "constraint": "required"
            }
        }
    }


class GLNORMValidationErrorResponse(BaseModel):
    """
    Validation error response for request validation failures.

    Provides detailed information about which fields failed validation
    and why, enabling clients to correct their requests.

    Attributes:
        success: Always False
        code: Always GLNORM-E100 for validation errors
        message: Summary message
        validation_errors: List of individual validation errors
        request_id: Request identifier
        timestamp: Response timestamp
        http_status: Always 400
    """

    success: bool = Field(
        default=False,
        description="Always False for validation errors"
    )
    code: str = Field(
        default="GLNORM-E100",
        description="Error code (always GLNORM-E100 for validation)"
    )
    message: str = Field(
        default="Request validation failed",
        description="Summary error message"
    )
    validation_errors: List[ValidationErrorItem] = Field(
        ...,
        description="List of individual validation errors"
    )
    request_id: UUID = Field(
        default_factory=uuid4,
        description="Request identifier"
    )
    timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="Response timestamp (UTC)"
    )
    http_status: int = Field(
        default=400,
        description="HTTP status code (always 400)"
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "success": False,
                "code": "GLNORM-E100",
                "message": "Request validation failed",
                "validation_errors": [
                    {
                        "field": "source_unit",
                        "message": "Field is required",
                        "value": None,
                        "constraint": "required"
                    },
                    {
                        "field": "value",
                        "message": "Must be greater than 0",
                        "value": -5,
                        "constraint": "gt=0"
                    }
                ],
                "request_id": "550e8400-e29b-41d4-a716-446655440000",
                "timestamp": "2024-01-15T10:30:00Z",
                "http_status": 400
            }
        }
    }


class AuditableError(BaseModel):
    """
    Error model with full audit trail information.

    Extends ErrorDetail with comprehensive audit fields required
    for regulatory compliance and forensic analysis.

    Attributes:
        error: The base error detail
        audit_id: Unique audit record identifier
        correlation_id: Correlation ID linking related operations
        provenance_hash: SHA-256 hash for data provenance
        input_snapshot: Sanitized snapshot of input data
        stack_trace: Stack trace (non-production only)
        environment: Environment identifier
        audit_timestamp: Audit record creation time
    """

    error: ErrorDetail = Field(
        ...,
        description="Base error detail"
    )
    audit_id: UUID = Field(
        default_factory=uuid4,
        description="Unique audit record identifier"
    )
    correlation_id: Optional[str] = Field(
        default=None,
        description="Correlation ID linking related operations"
    )
    provenance_hash: Optional[str] = Field(
        default=None,
        description="SHA-256 hash for data provenance",
        pattern=r"^[a-f0-9]{64}$"
    )
    input_snapshot: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Sanitized snapshot of input data (PII removed)"
    )
    stack_trace: Optional[str] = Field(
        default=None,
        description="Stack trace (non-production environments only)"
    )
    environment: str = Field(
        default="production",
        description="Environment identifier"
    )
    audit_timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="Audit record creation timestamp"
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "error": {
                    "code": "GLNORM-E602",
                    "message": "Audit integrity violation detected",
                    "category": "Audit",
                    "severity": "CRITICAL"
                },
                "audit_id": "550e8400-e29b-41d4-a716-446655440000",
                "correlation_id": "txn-2024-001",
                "provenance_hash": "a948904f2f0f479b8f8564cbf12dac6b18b1e8d4e8e7b0e6a5d9c2b1a0f3e4d5",
                "environment": "production",
                "audit_timestamp": "2024-01-15T10:30:00Z"
            }
        }
    }
