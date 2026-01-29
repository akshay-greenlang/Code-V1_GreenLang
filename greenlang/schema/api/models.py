# -*- coding: utf-8 -*-
"""
API Request/Response Models for GL-FOUND-X-002 (GreenLang Schema Compiler & Validator).

This module provides Pydantic v2 models for the FastAPI REST API, including:
- ValidateRequest/ValidateResponse: Single payload validation
- BatchValidateRequest/BatchValidateResponse: Batch validation
- CompileRequest/CompileResponse: Schema compilation
- HealthResponse: Health check endpoint
- MetricsResponse: Prometheus-compatible metrics
- SchemaVersionsResponse: Schema version listing

All models include OpenAPI examples and comprehensive validation.

Example:
    >>> from greenlang.schema.api.models import ValidateRequest, ValidateResponse
    >>> request = ValidateRequest(
    ...     schema_ref=SchemaRef(schema_id="emissions/activity", version="1.3.0"),
    ...     payload={"energy": 100, "unit": "kWh"}
    ... )

Author: GreenLang Framework Team
Version: 1.0.0
GL-FOUND-X-002: Schema Compiler & Validator - Task 5.3
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field, field_validator

from greenlang.schema.models.schema_ref import SchemaRef
from greenlang.schema.models.config import ValidationOptions, ValidationProfile
from greenlang.schema.models.finding import Finding
from greenlang.schema.models.report import ValidationSummary
from greenlang.schema.models.patch import FixSuggestion


# =============================================================================
# ERROR CODES
# =============================================================================


class GreenLangSchemaErrorCodes:
    """GreenLang Schema API error code constants."""
    # Validation errors (GLSCHEMA-API-VAL-*)
    PARAM_MISSING = "GLSCHEMA-API-VAL-001"
    PARAM_INVALID = "GLSCHEMA-API-VAL-002"
    PAYLOAD_INVALID = "GLSCHEMA-API-VAL-003"
    PAYLOAD_TOO_LARGE = "GLSCHEMA-API-VAL-004"
    BATCH_TOO_LARGE = "GLSCHEMA-API-VAL-005"

    # Resource errors (GLSCHEMA-API-RES-*)
    SCHEMA_NOT_FOUND = "GLSCHEMA-API-RES-001"
    VERSION_NOT_FOUND = "GLSCHEMA-API-RES-002"

    # Operation errors (GLSCHEMA-API-OPS-*)
    COMPILATION_FAILED = "GLSCHEMA-API-OPS-001"
    VALIDATION_FAILED = "GLSCHEMA-API-OPS-002"
    TIMEOUT = "GLSCHEMA-API-OPS-003"

    # Auth errors (GLSCHEMA-API-AUTH-*)
    UNAUTHORIZED = "GLSCHEMA-API-AUTH-001"
    RATE_LIMITED = "GLSCHEMA-API-AUTH-002"

    # System errors (GLSCHEMA-API-SYS-*)
    INTERNAL_ERROR = "GLSCHEMA-API-SYS-001"
    SERVICE_UNAVAILABLE = "GLSCHEMA-API-SYS-002"


# =============================================================================
# ERROR RESPONSE MODELS
# =============================================================================


class ErrorDetail(BaseModel):
    """Detail about a specific error."""

    code: str = Field(
        ...,
        description="Error code (e.g., 'GLSCHEMA-API-VAL-001')"
    )

    message: str = Field(
        ...,
        description="Human-readable error message"
    )

    field: Optional[str] = Field(
        default=None,
        description="Field that caused the error (if applicable)"
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "code": "GLSCHEMA-API-VAL-001",
                    "message": "Schema reference is required",
                    "field": "schema_ref"
                }
            ]
        }
    }


class ErrorResponse(BaseModel):
    """Standardized error response."""

    error: str = Field(
        ...,
        description="Error type (e.g., 'validation_error', 'not_found')"
    )

    message: str = Field(
        ...,
        description="Human-readable error message"
    )

    details: List[ErrorDetail] = Field(
        default_factory=list,
        description="Detailed error information"
    )

    trace_id: Optional[str] = Field(
        default=None,
        description="Request trace ID for debugging"
    )

    timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="Error timestamp"
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "error": "validation_error",
                    "message": "Invalid request payload",
                    "details": [
                        {
                            "code": "GLSCHEMA-API-VAL-001",
                            "message": "Schema reference is required",
                            "field": "schema_ref"
                        }
                    ],
                    "trace_id": "abc123",
                    "timestamp": "2026-01-28T10:30:00Z"
                }
            ]
        }
    }


# =============================================================================
# VALIDATE REQUEST/RESPONSE
# =============================================================================


class ValidateRequest(BaseModel):
    """
    Request to validate a single payload against a schema.

    Attributes:
        schema_ref: Reference to the schema to validate against.
        payload: The payload to validate (as dict or YAML/JSON string).
        options: Optional validation options to override defaults.

    Example:
        >>> request = ValidateRequest(
        ...     schema_ref=SchemaRef(schema_id="emissions/activity", version="1.3.0"),
        ...     payload={"energy": {"value": 100, "unit": "kWh"}}
        ... )
    """

    schema_ref: SchemaRef = Field(
        ...,
        description="Reference to the schema to validate against"
    )

    payload: Union[Dict[str, Any], str] = Field(
        ...,
        description="Payload to validate (dict or YAML/JSON string)"
    )

    options: Optional[ValidationOptions] = Field(
        default=None,
        description="Validation options to override defaults"
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "schema_ref": {
                        "schema_id": "emissions/activity",
                        "version": "1.3.0"
                    },
                    "payload": {
                        "activity_type": "electricity",
                        "energy": {"value": 100, "unit": "kWh"},
                        "location": "US"
                    },
                    "options": {
                        "profile": "standard",
                        "normalize": True
                    }
                }
            ]
        }
    }


class ValidateResponse(BaseModel):
    """
    Response from single payload validation.

    Attributes:
        valid: Whether the payload passed validation (no errors).
        schema_ref: The schema reference used for validation.
        schema_hash: SHA-256 hash of the compiled schema.
        summary: Summary counts of findings.
        findings: List of all validation findings.
        normalized_payload: Normalized payload (if normalization enabled).
        fix_suggestions: List of fix suggestions (if enabled).
        timings_ms: Performance timing for each phase.

    Example:
        >>> response.valid
        True
        >>> response.summary.error_count
        0
    """

    valid: bool = Field(
        ...,
        description="Whether the payload passed validation"
    )

    schema_ref: SchemaRef = Field(
        ...,
        description="Schema reference used for validation"
    )

    schema_hash: str = Field(
        ...,
        min_length=64,
        max_length=64,
        description="SHA-256 hash of the compiled schema"
    )

    summary: ValidationSummary = Field(
        ...,
        description="Summary of validation findings"
    )

    findings: List[Finding] = Field(
        default_factory=list,
        description="List of validation findings"
    )

    normalized_payload: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Normalized payload (if normalization enabled)"
    )

    fix_suggestions: Optional[List[FixSuggestion]] = Field(
        default=None,
        description="List of fix suggestions (if enabled)"
    )

    timings_ms: Dict[str, float] = Field(
        default_factory=dict,
        description="Performance timing in milliseconds"
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "valid": True,
                    "schema_ref": {
                        "schema_id": "emissions/activity",
                        "version": "1.3.0"
                    },
                    "schema_hash": "a" * 64,
                    "summary": {
                        "valid": True,
                        "error_count": 0,
                        "warning_count": 0,
                        "info_count": 0
                    },
                    "findings": [],
                    "normalized_payload": {
                        "activity_type": "electricity",
                        "energy": {"value": 100, "unit": "kWh"}
                    },
                    "timings_ms": {
                        "parse_ms": 2.5,
                        "compile_ms": 0.1,
                        "validate_ms": 15.3,
                        "total_ms": 18.0
                    }
                }
            ]
        }
    }


# =============================================================================
# BATCH VALIDATE REQUEST/RESPONSE
# =============================================================================


class BatchValidateRequest(BaseModel):
    """
    Request to validate multiple payloads against the same schema.

    Attributes:
        schema_ref: Reference to the schema to validate against.
        payloads: List of payloads to validate.
        options: Optional validation options to override defaults.

    Example:
        >>> request = BatchValidateRequest(
        ...     schema_ref=SchemaRef(schema_id="emissions/activity", version="1.3.0"),
        ...     payloads=[{"energy": 100}, {"energy": 200}]
        ... )
    """

    schema_ref: SchemaRef = Field(
        ...,
        description="Reference to the schema to validate against"
    )

    payloads: List[Union[Dict[str, Any], str]] = Field(
        ...,
        min_length=1,
        max_length=1000,
        description="List of payloads to validate"
    )

    options: Optional[ValidationOptions] = Field(
        default=None,
        description="Validation options to override defaults"
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "schema_ref": {
                        "schema_id": "emissions/activity",
                        "version": "1.3.0"
                    },
                    "payloads": [
                        {"activity_type": "electricity", "energy": {"value": 100, "unit": "kWh"}},
                        {"activity_type": "gas", "energy": {"value": 50, "unit": "kWh"}}
                    ],
                    "options": {
                        "profile": "standard",
                        "normalize": True
                    }
                }
            ]
        }
    }


class BatchItemResult(BaseModel):
    """
    Result for a single item in batch validation.

    Attributes:
        index: Zero-based index of the item in the batch.
        id: Optional identifier for the item.
        valid: Whether this item passed validation.
        findings: List of findings for this item.
        normalized_payload: Normalized payload (if enabled).

    Example:
        >>> result.index
        0
        >>> result.valid
        True
    """

    index: int = Field(
        ...,
        ge=0,
        description="Zero-based index in the batch"
    )

    id: Optional[str] = Field(
        default=None,
        max_length=256,
        description="Optional identifier for the item"
    )

    valid: bool = Field(
        ...,
        description="Whether this item passed validation"
    )

    findings: List[Finding] = Field(
        default_factory=list,
        description="List of findings for this item"
    )

    normalized_payload: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Normalized payload (if enabled)"
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "index": 0,
                    "id": "record-001",
                    "valid": True,
                    "findings": [],
                    "normalized_payload": {"energy": {"value": 100, "unit": "kWh"}}
                }
            ]
        }
    }


class BatchSummary(BaseModel):
    """
    Summary statistics for batch validation.

    Attributes:
        total_items: Total number of items in the batch.
        valid_count: Number of items that passed validation.
        error_count: Number of items with errors.
        warning_count: Number of items with warnings.
        total_findings: Total number of findings across all items.
        processing_time_ms: Total processing time in milliseconds.

    Example:
        >>> summary.success_rate()
        95.0
    """

    total_items: int = Field(
        ...,
        ge=0,
        description="Total number of items in the batch"
    )

    valid_count: int = Field(
        default=0,
        ge=0,
        description="Number of items that passed validation"
    )

    error_count: int = Field(
        default=0,
        ge=0,
        description="Number of items with errors"
    )

    warning_count: int = Field(
        default=0,
        ge=0,
        description="Number of items with warnings"
    )

    total_findings: int = Field(
        default=0,
        ge=0,
        description="Total findings across all items"
    )

    processing_time_ms: float = Field(
        default=0.0,
        ge=0,
        description="Total processing time in milliseconds"
    )

    def success_rate(self) -> float:
        """Calculate success rate as percentage."""
        if self.total_items == 0:
            return 100.0
        return (self.valid_count / self.total_items) * 100.0

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "total_items": 100,
                    "valid_count": 95,
                    "error_count": 5,
                    "warning_count": 10,
                    "total_findings": 15,
                    "processing_time_ms": 250.5
                }
            ]
        }
    }


class BatchValidateResponse(BaseModel):
    """
    Response from batch validation.

    Attributes:
        schema_ref: The schema reference used for validation.
        schema_hash: SHA-256 hash of the compiled schema.
        summary: Aggregate summary of all items.
        results: Individual results for each item.

    Example:
        >>> response.summary.success_rate()
        95.0
    """

    schema_ref: SchemaRef = Field(
        ...,
        description="Schema reference used for validation"
    )

    schema_hash: str = Field(
        ...,
        min_length=64,
        max_length=64,
        description="SHA-256 hash of the compiled schema"
    )

    summary: BatchSummary = Field(
        ...,
        description="Aggregate summary of all items"
    )

    results: List[BatchItemResult] = Field(
        default_factory=list,
        description="Individual results for each item"
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "schema_ref": {
                        "schema_id": "emissions/activity",
                        "version": "1.3.0"
                    },
                    "schema_hash": "a" * 64,
                    "summary": {
                        "total_items": 2,
                        "valid_count": 2,
                        "error_count": 0,
                        "warning_count": 0,
                        "total_findings": 0,
                        "processing_time_ms": 35.0
                    },
                    "results": [
                        {"index": 0, "valid": True, "findings": []},
                        {"index": 1, "valid": True, "findings": []}
                    ]
                }
            ]
        }
    }


# =============================================================================
# COMPILE REQUEST/RESPONSE
# =============================================================================


class CompileRequest(BaseModel):
    """
    Request to compile a schema to intermediate representation (IR).

    Attributes:
        schema_source: Schema source as dict or YAML/JSON string.
        schema_ref: Optional schema reference (for registration).
        include_ir: Whether to include the compiled IR in response.

    Example:
        >>> request = CompileRequest(
        ...     schema_source={"type": "object", "properties": {...}},
        ...     schema_ref=SchemaRef(schema_id="test/schema", version="1.0.0")
        ... )
    """

    schema_source: Union[Dict[str, Any], str] = Field(
        ...,
        description="Schema source (dict or YAML/JSON string)"
    )

    schema_ref: Optional[SchemaRef] = Field(
        default=None,
        description="Optional schema reference for registration"
    )

    include_ir: bool = Field(
        default=False,
        description="Whether to include compiled IR in response"
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "schema_source": {
                        "$schema": "https://json-schema.org/draft/2020-12/schema",
                        "type": "object",
                        "properties": {
                            "energy": {
                                "type": "object",
                                "properties": {
                                    "value": {"type": "number"},
                                    "unit": {"type": "string"}
                                }
                            }
                        }
                    },
                    "schema_ref": {
                        "schema_id": "test/energy",
                        "version": "1.0.0"
                    },
                    "include_ir": False
                }
            ]
        }
    }


class CompileResponse(BaseModel):
    """
    Response from schema compilation.

    Attributes:
        schema_hash: SHA-256 hash of the compiled schema.
        compiled_at: Timestamp when compilation completed.
        warnings: List of compilation warnings.
        ir: Compiled intermediate representation (if requested).
        properties_count: Number of properties in schema.
        rules_count: Number of rules in schema.

    Example:
        >>> response.schema_hash
        'abc123...'
    """

    schema_hash: str = Field(
        ...,
        min_length=64,
        max_length=64,
        description="SHA-256 hash of the compiled schema"
    )

    compiled_at: datetime = Field(
        ...,
        description="Compilation timestamp"
    )

    warnings: List[str] = Field(
        default_factory=list,
        description="Compilation warnings"
    )

    ir: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Compiled IR (if requested)"
    )

    properties_count: int = Field(
        default=0,
        ge=0,
        description="Number of properties in schema"
    )

    rules_count: int = Field(
        default=0,
        ge=0,
        description="Number of rules in schema"
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "schema_hash": "a" * 64,
                    "compiled_at": "2026-01-28T10:30:00Z",
                    "warnings": [],
                    "properties_count": 5,
                    "rules_count": 2
                }
            ]
        }
    }


# =============================================================================
# SCHEMA VERSION MODELS
# =============================================================================


class SchemaVersionInfo(BaseModel):
    """
    Information about a specific schema version.

    Attributes:
        version: The version string.
        created_at: When this version was created.
        deprecated: Whether this version is deprecated.
        deprecated_message: Deprecation message (if deprecated).

    Example:
        >>> info.version
        '1.3.0'
    """

    version: str = Field(
        ...,
        description="Version string"
    )

    created_at: Optional[datetime] = Field(
        default=None,
        description="Version creation timestamp"
    )

    deprecated: bool = Field(
        default=False,
        description="Whether version is deprecated"
    )

    deprecated_message: Optional[str] = Field(
        default=None,
        description="Deprecation message"
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "version": "1.3.0",
                    "created_at": "2026-01-01T00:00:00Z",
                    "deprecated": False
                },
                {
                    "version": "1.2.0",
                    "created_at": "2025-06-01T00:00:00Z",
                    "deprecated": True,
                    "deprecated_message": "Use version 1.3.0 or later"
                }
            ]
        }
    }


class SchemaVersionsResponse(BaseModel):
    """
    Response listing schema versions.

    Attributes:
        schema_id: The schema identifier.
        versions: List of available versions.
        latest: The latest (recommended) version.

    Example:
        >>> response.latest
        '1.3.0'
    """

    schema_id: str = Field(
        ...,
        description="Schema identifier"
    )

    versions: List[SchemaVersionInfo] = Field(
        default_factory=list,
        description="Available versions"
    )

    latest: Optional[str] = Field(
        default=None,
        description="Latest (recommended) version"
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "schema_id": "emissions/activity",
                    "versions": [
                        {"version": "1.3.0", "deprecated": False},
                        {"version": "1.2.0", "deprecated": True}
                    ],
                    "latest": "1.3.0"
                }
            ]
        }
    }


class SchemaDetailResponse(BaseModel):
    """
    Response with full schema details.

    Attributes:
        schema_id: The schema identifier.
        version: The schema version.
        schema_hash: SHA-256 hash of the schema.
        content: The full schema content.
        created_at: When this version was created.
        deprecated: Whether this version is deprecated.

    Example:
        >>> response.content
        {'type': 'object', ...}
    """

    schema_id: str = Field(
        ...,
        description="Schema identifier"
    )

    version: str = Field(
        ...,
        description="Schema version"
    )

    schema_hash: str = Field(
        ...,
        min_length=64,
        max_length=64,
        description="SHA-256 hash of the schema"
    )

    content: Dict[str, Any] = Field(
        ...,
        description="Full schema content"
    )

    created_at: Optional[datetime] = Field(
        default=None,
        description="Version creation timestamp"
    )

    deprecated: bool = Field(
        default=False,
        description="Whether version is deprecated"
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "schema_id": "emissions/activity",
                    "version": "1.3.0",
                    "schema_hash": "a" * 64,
                    "content": {
                        "$schema": "https://json-schema.org/draft/2020-12/schema",
                        "type": "object",
                        "properties": {}
                    },
                    "created_at": "2026-01-01T00:00:00Z",
                    "deprecated": False
                }
            ]
        }
    }


# =============================================================================
# HEALTH CHECK MODELS
# =============================================================================


class ComponentHealth(BaseModel):
    """
    Health status of a single component.

    Attributes:
        name: Component name.
        status: Component status ('healthy', 'degraded', 'unhealthy').
        message: Optional status message.
        latency_ms: Optional latency measurement.

    Example:
        >>> component.status
        'healthy'
    """

    name: str = Field(
        ...,
        description="Component name"
    )

    status: str = Field(
        ...,
        description="Component status"
    )

    message: Optional[str] = Field(
        default=None,
        description="Status message"
    )

    latency_ms: Optional[float] = Field(
        default=None,
        ge=0,
        description="Latency in milliseconds"
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "name": "schema_cache",
                    "status": "healthy",
                    "latency_ms": 0.5
                }
            ]
        }
    }


class HealthResponse(BaseModel):
    """
    Health check response.

    Attributes:
        status: Overall status ('healthy', 'degraded', 'unhealthy').
        version: Service version.
        cache_size: Number of cached schemas.
        uptime_seconds: Service uptime in seconds.
        components: Health of individual components.

    Example:
        >>> response.status
        'healthy'
    """

    status: str = Field(
        ...,
        description="Overall health status"
    )

    version: str = Field(
        ...,
        description="Service version"
    )

    cache_size: int = Field(
        default=0,
        ge=0,
        description="Number of cached schemas"
    )

    uptime_seconds: float = Field(
        default=0.0,
        ge=0,
        description="Service uptime in seconds"
    )

    components: List[ComponentHealth] = Field(
        default_factory=list,
        description="Component health details"
    )

    timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="Health check timestamp"
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "status": "healthy",
                    "version": "1.0.0",
                    "cache_size": 50,
                    "uptime_seconds": 3600.0,
                    "components": [
                        {"name": "schema_cache", "status": "healthy"},
                        {"name": "registry", "status": "healthy"}
                    ],
                    "timestamp": "2026-01-28T10:30:00Z"
                }
            ]
        }
    }


# =============================================================================
# METRICS MODELS
# =============================================================================


class MetricsResponse(BaseModel):
    """
    Prometheus-compatible metrics response.

    Attributes:
        validations_total: Total validation requests.
        validations_success: Successful validations.
        validations_failed: Failed validations.
        batch_validations_total: Total batch validation requests.
        cache_hits: Schema cache hits.
        cache_misses: Schema cache misses.
        cache_size: Current cache size.
        avg_validation_time_ms: Average validation time.
        p95_validation_time_ms: 95th percentile validation time.

    Example:
        >>> response.cache_hit_rate()
        0.95
    """

    validations_total: int = Field(
        default=0,
        ge=0,
        description="Total validation requests"
    )

    validations_success: int = Field(
        default=0,
        ge=0,
        description="Successful validations"
    )

    validations_failed: int = Field(
        default=0,
        ge=0,
        description="Failed validations"
    )

    batch_validations_total: int = Field(
        default=0,
        ge=0,
        description="Total batch validation requests"
    )

    batch_items_total: int = Field(
        default=0,
        ge=0,
        description="Total items validated in batches"
    )

    cache_hits: int = Field(
        default=0,
        ge=0,
        description="Schema cache hits"
    )

    cache_misses: int = Field(
        default=0,
        ge=0,
        description="Schema cache misses"
    )

    cache_size: int = Field(
        default=0,
        ge=0,
        description="Current cache size"
    )

    avg_validation_time_ms: float = Field(
        default=0.0,
        ge=0,
        description="Average validation time in milliseconds"
    )

    p95_validation_time_ms: float = Field(
        default=0.0,
        ge=0,
        description="95th percentile validation time"
    )

    uptime_seconds: float = Field(
        default=0.0,
        ge=0,
        description="Service uptime in seconds"
    )

    def cache_hit_rate(self) -> float:
        """Calculate cache hit rate (0.0 to 1.0)."""
        total = self.cache_hits + self.cache_misses
        if total == 0:
            return 0.0
        return self.cache_hits / total

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "validations_total": 10000,
                    "validations_success": 9500,
                    "validations_failed": 500,
                    "batch_validations_total": 100,
                    "batch_items_total": 5000,
                    "cache_hits": 9000,
                    "cache_misses": 1000,
                    "cache_size": 50,
                    "avg_validation_time_ms": 15.5,
                    "p95_validation_time_ms": 25.0,
                    "uptime_seconds": 86400.0
                }
            ]
        }
    }


# =============================================================================
# MODULE EXPORTS
# =============================================================================


__all__ = [
    # Error codes
    "GreenLangSchemaErrorCodes",

    # Error response
    "ErrorDetail",
    "ErrorResponse",

    # Validate
    "ValidateRequest",
    "ValidateResponse",

    # Batch validate
    "BatchValidateRequest",
    "BatchItemResult",
    "BatchSummary",
    "BatchValidateResponse",

    # Compile
    "CompileRequest",
    "CompileResponse",

    # Schema versions
    "SchemaVersionInfo",
    "SchemaVersionsResponse",
    "SchemaDetailResponse",

    # Health
    "ComponentHealth",
    "HealthResponse",

    # Metrics
    "MetricsResponse",
]
