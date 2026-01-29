"""
Pydantic models for GL Normalizer Service API.

This module defines all request and response models for the Normalizer API,
implementing the GL-FOUND-X-003 specification for climate data normalization.

Models follow these conventions:
    - Request models end with 'Request'
    - Response models end with 'Response'
    - All models include comprehensive field descriptions
    - Examples are provided for OpenAPI documentation

Error Codes (GLNORM-*):
    GLNORM-001: Invalid input value
    GLNORM-002: Unknown unit
    GLNORM-003: Incompatible unit conversion
    GLNORM-004: Batch size exceeded
    GLNORM-005: Job not found
    GLNORM-006: Vocabulary not found
    GLNORM-007: Authentication failed
    GLNORM-008: Rate limit exceeded
    GLNORM-009: Internal processing error
    GLNORM-010: Validation failed
"""

from datetime import datetime
from enum import Enum
from typing import Any, Optional
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator


# ==============================================================================
# Enums
# ==============================================================================


class BatchMode(str, Enum):
    """
    Batch processing mode for handling failures.

    Attributes:
        PARTIAL: Continue processing on failures, return partial results
        FAIL_FAST: Stop processing on first failure
        THRESHOLD: Stop if failure rate exceeds configured threshold
    """

    PARTIAL = "PARTIAL"
    FAIL_FAST = "FAIL_FAST"
    THRESHOLD = "THRESHOLD"


class JobStatus(str, Enum):
    """
    Async job processing status.

    Attributes:
        PENDING: Job queued, not yet started
        PROCESSING: Job currently being processed
        COMPLETED: Job completed successfully
        FAILED: Job failed with errors
        CANCELLED: Job was cancelled
    """

    PENDING = "PENDING"
    PROCESSING = "PROCESSING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"


class ReviewReason(str, Enum):
    """
    Reasons why a normalization result needs human review.

    Attributes:
        LOW_CONFIDENCE: Confidence score below threshold
        AMBIGUOUS_UNIT: Multiple possible unit interpretations
        OUTLIER_VALUE: Value outside expected range
        UNKNOWN_ENTITY: Entity not found in vocabulary
        CUSTOM_RULE: Custom business rule triggered review
    """

    LOW_CONFIDENCE = "LOW_CONFIDENCE"
    AMBIGUOUS_UNIT = "AMBIGUOUS_UNIT"
    OUTLIER_VALUE = "OUTLIER_VALUE"
    UNKNOWN_ENTITY = "UNKNOWN_ENTITY"
    CUSTOM_RULE = "CUSTOM_RULE"


class ErrorCode(str, Enum):
    """
    Standardized error codes for the Normalizer API.

    All error codes follow the GLNORM-XXX pattern for easy identification
    and documentation reference.
    """

    INVALID_INPUT = "GLNORM-001"
    UNKNOWN_UNIT = "GLNORM-002"
    INCOMPATIBLE_CONVERSION = "GLNORM-003"
    BATCH_SIZE_EXCEEDED = "GLNORM-004"
    JOB_NOT_FOUND = "GLNORM-005"
    VOCABULARY_NOT_FOUND = "GLNORM-006"
    AUTHENTICATION_FAILED = "GLNORM-007"
    RATE_LIMIT_EXCEEDED = "GLNORM-008"
    INTERNAL_ERROR = "GLNORM-009"
    VALIDATION_FAILED = "GLNORM-010"


# ==============================================================================
# Base Models
# ==============================================================================


class APIResponse(BaseModel):
    """
    Base response model with API revision tracking.

    All API responses include the api_revision field for version tracking
    and backward compatibility management.
    """

    model_config = ConfigDict(
        populate_by_name=True,
        json_schema_extra={
            "example": {
                "api_revision": "2026-01-30"
            }
        }
    )

    api_revision: str = Field(
        default="2026-01-30",
        description="API revision date for version tracking",
        examples=["2026-01-30"]
    )


class ErrorDetail(BaseModel):
    """
    Detailed error information for failed operations.

    Attributes:
        code: GLNORM error code
        message: Human-readable error message
        field: Field that caused the error (if applicable)
        details: Additional error context
    """

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "code": "GLNORM-002",
                "message": "Unknown unit: 'tonnes'",
                "field": "unit",
                "details": {"suggestion": "Did you mean 'metric_ton'?"}
            }
        }
    )

    code: ErrorCode = Field(
        ...,
        description="GLNORM error code"
    )
    message: str = Field(
        ...,
        description="Human-readable error message",
        max_length=500
    )
    field: Optional[str] = Field(
        default=None,
        description="Field that caused the error"
    )
    details: Optional[dict[str, Any]] = Field(
        default=None,
        description="Additional error context"
    )


class ErrorResponse(APIResponse):
    """
    Standard error response format.

    All API errors return this structure for consistent client handling.
    """

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "api_revision": "2026-01-30",
                "error": {
                    "code": "GLNORM-002",
                    "message": "Unknown unit: 'tonnes'",
                    "field": "unit",
                    "details": {"suggestion": "Did you mean 'metric_ton'?"}
                },
                "request_id": "req_abc123xyz"
            }
        }
    )

    error: ErrorDetail = Field(
        ...,
        description="Error details"
    )
    request_id: Optional[str] = Field(
        default=None,
        description="Request ID for support reference"
    )


# ==============================================================================
# Normalize Single Value
# ==============================================================================


class NormalizeRequest(BaseModel):
    """
    Request model for single value normalization.

    This endpoint normalizes a single value with its unit to canonical form.
    Optional context fields improve normalization accuracy.

    Attributes:
        value: The numeric or string value to normalize
        unit: The unit of the input value
        target_unit: Optional target unit for conversion
        entity: Optional entity context (e.g., company name, facility ID)
        context: Optional additional context for disambiguation

    Example:
        >>> request = NormalizeRequest(
        ...     value="1500",
        ...     unit="kg CO2",
        ...     target_unit="metric_ton_co2e",
        ...     entity="facility_001",
        ...     context={"reporting_year": 2025}
        ... )
    """

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "value": "1500",
                "unit": "kg CO2",
                "target_unit": "metric_ton_co2e",
                "entity": "facility_001",
                "context": {"reporting_year": 2025, "scope": 1}
            }
        }
    )

    value: str | float | int = Field(
        ...,
        description="Value to normalize (numeric or string representation)",
        examples=["1500", 1500, 1500.5]
    )
    unit: str = Field(
        ...,
        description="Unit of the input value",
        min_length=1,
        max_length=100,
        examples=["kg CO2", "MWh", "gallons"]
    )
    target_unit: Optional[str] = Field(
        default=None,
        description="Target unit for conversion (uses canonical unit if not specified)",
        max_length=100,
        examples=["metric_ton_co2e", "kWh"]
    )
    entity: Optional[str] = Field(
        default=None,
        description="Entity context for disambiguation (company, facility, etc.)",
        max_length=255,
        examples=["facility_001", "ACME Corp"]
    )
    context: Optional[dict[str, Any]] = Field(
        default=None,
        description="Additional context for normalization",
        examples=[{"reporting_year": 2025, "scope": 1}]
    )

    @field_validator("value", mode="before")
    @classmethod
    def coerce_value(cls, v: Any) -> str | float | int:
        """Coerce value to acceptable types."""
        if isinstance(v, (str, int, float)):
            return v
        return str(v)


class NormalizeResponse(APIResponse):
    """
    Response model for single value normalization.

    Contains the normalized canonical value along with confidence metrics
    and audit information for compliance tracking.

    Attributes:
        canonical_value: Normalized value in canonical form
        canonical_unit: Canonical unit identifier
        confidence: Confidence score (0.0-1.0)
        needs_review: Whether human review is recommended
        review_reasons: Reasons for requiring review
        audit_id: Unique audit trail identifier
        source_value: Original input value (for reference)
        source_unit: Original input unit (for reference)
        conversion_factor: Applied conversion factor (if any)
        metadata: Additional normalization metadata
    """

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "api_revision": "2026-01-30",
                "canonical_value": 1.5,
                "canonical_unit": "metric_ton_co2e",
                "confidence": 0.95,
                "needs_review": False,
                "review_reasons": [],
                "audit_id": "aud_a1b2c3d4e5f6",
                "source_value": "1500",
                "source_unit": "kg CO2",
                "conversion_factor": 0.001,
                "metadata": {
                    "vocabulary_version": "2026.1",
                    "normalization_rule": "unit_conversion"
                }
            }
        }
    )

    canonical_value: float = Field(
        ...,
        description="Normalized value in canonical form"
    )
    canonical_unit: str = Field(
        ...,
        description="Canonical unit identifier",
        examples=["metric_ton_co2e", "kWh", "cubic_meter"]
    )
    confidence: float = Field(
        ...,
        description="Confidence score (0.0-1.0)",
        ge=0.0,
        le=1.0
    )
    needs_review: bool = Field(
        ...,
        description="Whether human review is recommended"
    )
    review_reasons: list[ReviewReason] = Field(
        default_factory=list,
        description="Reasons for requiring review"
    )
    audit_id: str = Field(
        ...,
        description="Unique audit trail identifier",
        examples=["aud_a1b2c3d4e5f6"]
    )
    source_value: str | float | int = Field(
        ...,
        description="Original input value"
    )
    source_unit: str = Field(
        ...,
        description="Original input unit"
    )
    conversion_factor: Optional[float] = Field(
        default=None,
        description="Applied conversion factor"
    )
    metadata: Optional[dict[str, Any]] = Field(
        default=None,
        description="Additional normalization metadata"
    )


# ==============================================================================
# Batch Normalization
# ==============================================================================


class BatchItem(BaseModel):
    """
    Single item in a batch normalization request.

    Attributes:
        id: Client-provided identifier for result correlation
        value: Value to normalize
        unit: Unit of the input value
        target_unit: Optional target unit
        entity: Optional entity context
        context: Optional additional context
    """

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "id": "item_001",
                "value": "1500",
                "unit": "kg CO2",
                "target_unit": "metric_ton_co2e"
            }
        }
    )

    id: str = Field(
        ...,
        description="Client-provided identifier for result correlation",
        min_length=1,
        max_length=255
    )
    value: str | float | int = Field(
        ...,
        description="Value to normalize"
    )
    unit: str = Field(
        ...,
        description="Unit of the input value",
        min_length=1,
        max_length=100
    )
    target_unit: Optional[str] = Field(
        default=None,
        description="Target unit for conversion",
        max_length=100
    )
    entity: Optional[str] = Field(
        default=None,
        description="Entity context",
        max_length=255
    )
    context: Optional[dict[str, Any]] = Field(
        default=None,
        description="Additional context"
    )


class BatchNormalizeRequest(BaseModel):
    """
    Request model for batch normalization.

    Supports up to 10,000 items in synchronous mode. For larger batches,
    use the async job API (POST /v1/jobs).

    Attributes:
        items: List of items to normalize
        batch_mode: How to handle failures
        threshold: Failure threshold for THRESHOLD mode (0.0-1.0)

    Example:
        >>> request = BatchNormalizeRequest(
        ...     items=[
        ...         BatchItem(id="1", value="100", unit="kg CO2"),
        ...         BatchItem(id="2", value="50", unit="MWh"),
        ...     ],
        ...     batch_mode=BatchMode.PARTIAL
        ... )
    """

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "items": [
                    {"id": "item_001", "value": "1500", "unit": "kg CO2"},
                    {"id": "item_002", "value": "250", "unit": "MWh"},
                    {"id": "item_003", "value": "1000", "unit": "gallons"}
                ],
                "batch_mode": "PARTIAL",
                "threshold": 0.1
            }
        }
    )

    items: list[BatchItem] = Field(
        ...,
        description="List of items to normalize",
        min_length=1,
        max_length=10000
    )
    batch_mode: BatchMode = Field(
        default=BatchMode.PARTIAL,
        description="How to handle failures during batch processing"
    )
    threshold: Optional[float] = Field(
        default=0.1,
        description="Failure threshold for THRESHOLD mode (0.0-1.0)",
        ge=0.0,
        le=1.0
    )


class BatchResultItem(BaseModel):
    """
    Single result item in a batch normalization response.

    Attributes:
        id: Client-provided identifier (from request)
        success: Whether normalization succeeded
        result: Normalization result (if successful)
        error: Error details (if failed)
    """

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "id": "item_001",
                "success": True,
                "result": {
                    "canonical_value": 1.5,
                    "canonical_unit": "metric_ton_co2e",
                    "confidence": 0.95,
                    "needs_review": False,
                    "audit_id": "aud_a1b2c3d4e5f6"
                }
            }
        }
    )

    id: str = Field(
        ...,
        description="Client-provided identifier"
    )
    success: bool = Field(
        ...,
        description="Whether normalization succeeded"
    )
    result: Optional[dict[str, Any]] = Field(
        default=None,
        description="Normalization result (if successful)"
    )
    error: Optional[ErrorDetail] = Field(
        default=None,
        description="Error details (if failed)"
    )


class BatchSummary(BaseModel):
    """
    Summary statistics for batch normalization.

    Attributes:
        total: Total items processed
        success: Successfully normalized items
        failed: Failed items
        needs_review: Items requiring human review
        processing_time_ms: Total processing time in milliseconds
    """

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "total": 100,
                "success": 95,
                "failed": 3,
                "needs_review": 2,
                "processing_time_ms": 1250
            }
        }
    )

    total: int = Field(
        ...,
        description="Total items processed",
        ge=0
    )
    success: int = Field(
        ...,
        description="Successfully normalized items",
        ge=0
    )
    failed: int = Field(
        ...,
        description="Failed items",
        ge=0
    )
    needs_review: int = Field(
        ...,
        description="Items requiring human review",
        ge=0
    )
    processing_time_ms: int = Field(
        ...,
        description="Total processing time in milliseconds",
        ge=0
    )


class BatchNormalizeResponse(APIResponse):
    """
    Response model for batch normalization.

    Contains results for all processed items along with summary statistics.
    """

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "api_revision": "2026-01-30",
                "results": [
                    {
                        "id": "item_001",
                        "success": True,
                        "result": {
                            "canonical_value": 1.5,
                            "canonical_unit": "metric_ton_co2e",
                            "confidence": 0.95,
                            "needs_review": False,
                            "audit_id": "aud_a1b2c3d4e5f6"
                        }
                    }
                ],
                "summary": {
                    "total": 100,
                    "success": 95,
                    "failed": 3,
                    "needs_review": 2,
                    "processing_time_ms": 1250
                }
            }
        }
    )

    results: list[BatchResultItem] = Field(
        ...,
        description="Results for each item"
    )
    summary: BatchSummary = Field(
        ...,
        description="Summary statistics"
    )


# ==============================================================================
# Async Jobs
# ==============================================================================


class CreateJobRequest(BaseModel):
    """
    Request model for creating an async normalization job.

    Use this endpoint for batch sizes exceeding 10,000 items.
    The job is processed asynchronously and results can be retrieved
    via the job status endpoint.

    Attributes:
        items: List of items to normalize (up to 1M items)
        batch_mode: How to handle failures
        callback_url: Optional webhook URL for completion notification
        priority: Job priority (1-10, higher = more urgent)
    """

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "items": [
                    {"id": "item_001", "value": "1500", "unit": "kg CO2"},
                    {"id": "item_002", "value": "250", "unit": "MWh"}
                ],
                "batch_mode": "PARTIAL",
                "callback_url": "https://example.com/webhooks/normalizer",
                "priority": 5
            }
        }
    )

    items: list[BatchItem] = Field(
        ...,
        description="List of items to normalize",
        min_length=1,
        max_length=1000000
    )
    batch_mode: BatchMode = Field(
        default=BatchMode.PARTIAL,
        description="How to handle failures"
    )
    callback_url: Optional[str] = Field(
        default=None,
        description="Webhook URL for completion notification",
        max_length=2048
    )
    priority: int = Field(
        default=5,
        description="Job priority (1-10)",
        ge=1,
        le=10
    )


class CreateJobResponse(APIResponse):
    """
    Response model for job creation.

    Attributes:
        job_id: Unique job identifier
        status: Current job status
        status_url: URL to check job status
        estimated_completion: Estimated completion time (if available)
    """

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "api_revision": "2026-01-30",
                "job_id": "job_f47ac10b-58cc-4372-a567-0e02b2c3d479",
                "status": "PENDING",
                "status_url": "/v1/jobs/job_f47ac10b-58cc-4372-a567-0e02b2c3d479",
                "estimated_completion": "2026-01-30T12:30:00Z"
            }
        }
    )

    job_id: str = Field(
        ...,
        description="Unique job identifier"
    )
    status: JobStatus = Field(
        ...,
        description="Current job status"
    )
    status_url: str = Field(
        ...,
        description="URL to check job status"
    )
    estimated_completion: Optional[datetime] = Field(
        default=None,
        description="Estimated completion time"
    )


class JobProgress(BaseModel):
    """
    Job progress information.

    Attributes:
        processed: Items processed so far
        total: Total items to process
        percent_complete: Completion percentage
        current_rate: Current processing rate (items/second)
    """

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "processed": 50000,
                "total": 100000,
                "percent_complete": 50.0,
                "current_rate": 5000.0
            }
        }
    )

    processed: int = Field(
        ...,
        description="Items processed so far",
        ge=0
    )
    total: int = Field(
        ...,
        description="Total items to process",
        ge=0
    )
    percent_complete: float = Field(
        ...,
        description="Completion percentage",
        ge=0.0,
        le=100.0
    )
    current_rate: Optional[float] = Field(
        default=None,
        description="Current processing rate (items/second)"
    )


class JobStatusResponse(APIResponse):
    """
    Response model for job status inquiry.

    Attributes:
        job_id: Unique job identifier
        status: Current job status
        progress: Progress information (if processing)
        summary: Final summary (if completed)
        result_url: URL to download results (if completed)
        error: Error details (if failed)
        created_at: Job creation timestamp
        started_at: Processing start timestamp
        completed_at: Completion timestamp
    """

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "api_revision": "2026-01-30",
                "job_id": "job_f47ac10b-58cc-4372-a567-0e02b2c3d479",
                "status": "PROCESSING",
                "progress": {
                    "processed": 50000,
                    "total": 100000,
                    "percent_complete": 50.0,
                    "current_rate": 5000.0
                },
                "created_at": "2026-01-30T12:00:00Z",
                "started_at": "2026-01-30T12:01:00Z"
            }
        }
    )

    job_id: str = Field(
        ...,
        description="Unique job identifier"
    )
    status: JobStatus = Field(
        ...,
        description="Current job status"
    )
    progress: Optional[JobProgress] = Field(
        default=None,
        description="Progress information"
    )
    summary: Optional[BatchSummary] = Field(
        default=None,
        description="Final summary (if completed)"
    )
    result_url: Optional[str] = Field(
        default=None,
        description="URL to download results"
    )
    error: Optional[ErrorDetail] = Field(
        default=None,
        description="Error details (if failed)"
    )
    created_at: datetime = Field(
        ...,
        description="Job creation timestamp"
    )
    started_at: Optional[datetime] = Field(
        default=None,
        description="Processing start timestamp"
    )
    completed_at: Optional[datetime] = Field(
        default=None,
        description="Completion timestamp"
    )


# ==============================================================================
# Vocabularies
# ==============================================================================


class VocabularyInfo(BaseModel):
    """
    Information about a normalization vocabulary.

    Attributes:
        id: Unique vocabulary identifier
        name: Human-readable name
        description: Vocabulary description
        version: Version string
        entry_count: Number of entries in vocabulary
        last_updated: Last update timestamp
        categories: Supported categories
    """

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "id": "ghg_units",
                "name": "GHG Emission Units",
                "description": "Standard units for greenhouse gas emissions",
                "version": "2026.1",
                "entry_count": 245,
                "last_updated": "2026-01-15T00:00:00Z",
                "categories": ["emissions", "energy", "transport"]
            }
        }
    )

    id: str = Field(
        ...,
        description="Unique vocabulary identifier"
    )
    name: str = Field(
        ...,
        description="Human-readable name"
    )
    description: Optional[str] = Field(
        default=None,
        description="Vocabulary description"
    )
    version: str = Field(
        ...,
        description="Version string"
    )
    entry_count: int = Field(
        ...,
        description="Number of entries",
        ge=0
    )
    last_updated: datetime = Field(
        ...,
        description="Last update timestamp"
    )
    categories: list[str] = Field(
        default_factory=list,
        description="Supported categories"
    )


class VocabulariesResponse(APIResponse):
    """
    Response model for vocabulary listing.

    Attributes:
        vocabularies: List of available vocabularies
        total: Total number of vocabularies
    """

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "api_revision": "2026-01-30",
                "vocabularies": [
                    {
                        "id": "ghg_units",
                        "name": "GHG Emission Units",
                        "version": "2026.1",
                        "entry_count": 245,
                        "last_updated": "2026-01-15T00:00:00Z",
                        "categories": ["emissions", "energy"]
                    }
                ],
                "total": 5
            }
        }
    )

    vocabularies: list[VocabularyInfo] = Field(
        ...,
        description="List of available vocabularies"
    )
    total: int = Field(
        ...,
        description="Total number of vocabularies",
        ge=0
    )


# ==============================================================================
# Health Check
# ==============================================================================


class HealthStatus(str, Enum):
    """
    Service health status.

    Attributes:
        HEALTHY: All systems operational
        DEGRADED: Some systems impaired but service available
        UNHEALTHY: Service unavailable
    """

    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"


class DependencyHealth(BaseModel):
    """
    Health status of a service dependency.

    Attributes:
        name: Dependency name
        status: Health status
        latency_ms: Response latency in milliseconds
        message: Additional status message
    """

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "name": "redis",
                "status": "healthy",
                "latency_ms": 2,
                "message": "Connected"
            }
        }
    )

    name: str = Field(
        ...,
        description="Dependency name"
    )
    status: HealthStatus = Field(
        ...,
        description="Health status"
    )
    latency_ms: Optional[int] = Field(
        default=None,
        description="Response latency in milliseconds"
    )
    message: Optional[str] = Field(
        default=None,
        description="Additional status message"
    )


class HealthResponse(APIResponse):
    """
    Response model for health check endpoint.

    Provides comprehensive health information for monitoring and
    load balancer integration.

    Attributes:
        status: Overall health status
        version: Service version
        uptime_seconds: Service uptime
        dependencies: Health of service dependencies
        timestamp: Health check timestamp
    """

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "api_revision": "2026-01-30",
                "status": "healthy",
                "version": "1.0.0",
                "uptime_seconds": 86400,
                "dependencies": [
                    {"name": "redis", "status": "healthy", "latency_ms": 2},
                    {"name": "vocabulary_service", "status": "healthy", "latency_ms": 15}
                ],
                "timestamp": "2026-01-30T12:00:00Z"
            }
        }
    )

    status: HealthStatus = Field(
        ...,
        description="Overall health status"
    )
    version: str = Field(
        ...,
        description="Service version"
    )
    uptime_seconds: int = Field(
        ...,
        description="Service uptime in seconds",
        ge=0
    )
    dependencies: list[DependencyHealth] = Field(
        default_factory=list,
        description="Health of service dependencies"
    )
    timestamp: datetime = Field(
        ...,
        description="Health check timestamp"
    )
