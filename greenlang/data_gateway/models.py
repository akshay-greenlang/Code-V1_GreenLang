# -*- coding: utf-8 -*-
"""
API Gateway Agent Service Data Models - AGENT-DATA-004: Data Gateway

Pydantic v2 data models for the API Gateway Agent SDK. Defines all
enumerations, core data models, and request wrappers required for the
unified data query interface that routes queries to backend data agents.

The API Gateway Agent sits behind Kong API Gateway and provides a single
query interface across all GreenLang data sources (PDF extractor, Excel
normalizer, ERP connector, EUDR traceability, SCADA/BMS, fleet
telematics, utility tariff, supplier portal, and more).

Models:
    - Enumerations: DataSourceType, QueryOperation, AggregationType,
        SortOrder, SourceStatus, CacheStrategy, QueryStatus,
        ConflictResolution
    - Core models: DataSource, SchemaField, SchemaDefinition, QueryFilter,
        QuerySort, QueryAggregation, QueryPlan, QueryResult, CacheEntry,
        SourceHealthCheck, SchemaMapping, QueryTemplate, DataCatalogEntry,
        GatewayStatistics
    - Request models: ExecuteQueryRequest, BatchQueryRequest,
        RegisterSourceRequest, CreateTemplateRequest,
        TranslateSchemaRequest, CacheInvalidateRequest

Author: GreenLang Platform Team
Date: February 2026
PRD: AGENT-DATA-004 API Gateway Agent
Status: Production Ready
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _utcnow() -> datetime:
    """Return current UTC datetime with microseconds zeroed."""
    return datetime.now(timezone.utc).replace(microsecond=0)


# =============================================================================
# Enumerations
# =============================================================================


class DataSourceType(str, Enum):
    """Types of data sources that can be registered with the gateway.

    Each value corresponds to a backend GreenLang data agent or external
    data integration endpoint that the gateway can route queries to.
    """

    PDF_EXTRACTOR = "pdf_extractor"
    EXCEL_NORMALIZER = "excel_normalizer"
    ERP_CONNECTOR = "erp_connector"
    EUDR_TRACEABILITY = "eudr_traceability"
    SCADA_BMS = "scada_bms"
    FLEET_TELEMATICS = "fleet_telematics"
    UTILITY_TARIFF = "utility_tariff"
    SUPPLIER_PORTAL = "supplier_portal"
    EVENT_PROCESSOR = "event_processor"
    DOCUMENT_CLASSIFIER = "document_classifier"
    OCR_AGENT = "ocr_agent"
    EMAIL_PROCESSOR = "email_processor"
    IOT_METER = "iot_meter"
    EMISSION_FACTOR = "emission_factor"
    CUSTOM = "custom"


class QueryOperation(str, Enum):
    """Supported query operations that can be applied to data sources.

    Operations define the type of data retrieval or transformation
    to perform against one or more registered backend data sources.
    """

    SELECT = "select"
    FILTER = "filter"
    AGGREGATE = "aggregate"
    JOIN = "join"
    TRANSFORM = "transform"
    COUNT = "count"
    DISTINCT = "distinct"
    SEARCH = "search"


class AggregationType(str, Enum):
    """Supported aggregation functions for query results.

    Aggregations can be applied to numeric or categorical fields
    to compute summary statistics across query result sets.
    """

    SUM = "sum"
    AVG = "avg"
    MIN = "min"
    MAX = "max"
    COUNT = "count"
    GROUP_BY = "group_by"
    PERCENTILE = "percentile"


class SortOrder(str, Enum):
    """Sort direction for query result ordering."""

    ASC = "asc"
    DESC = "desc"


class SourceStatus(str, Enum):
    """Health status of a registered data source.

    Tracks the operational state of each backend data agent
    as determined by periodic health checks.
    """

    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"
    MAINTENANCE = "maintenance"


class CacheStrategy(str, Enum):
    """Caching strategies for query results.

    Determines how query results are cached and when cached
    entries are invalidated or refreshed.
    """

    ALWAYS = "always"
    NEVER = "never"
    TTL_BASED = "ttl_based"
    EVENT_DRIVEN = "event_driven"


class QueryStatus(str, Enum):
    """Lifecycle status of a query execution.

    Tracks the current state of a query from submission
    through execution to completion or failure.
    """

    PENDING = "pending"
    EXECUTING = "executing"
    COMPLETED = "completed"
    FAILED = "failed"
    TIMEOUT = "timeout"
    CANCELLED = "cancelled"


class ConflictResolution(str, Enum):
    """Strategies for resolving data conflicts from multiple sources.

    When the same logical record is returned by multiple backend
    data sources with differing values, this strategy determines
    which value takes precedence.
    """

    LATEST_WINS = "latest_wins"
    SOURCE_PRIORITY = "source_priority"
    MERGE = "merge"
    ERROR = "error"


# =============================================================================
# Core Data Models
# =============================================================================


class DataSource(BaseModel):
    """Registered backend data source for the gateway.

    Represents a backend GreenLang data agent or external data endpoint
    that has been registered with the API Gateway for query routing,
    health monitoring, and capability discovery.

    Attributes:
        source_id: Unique identifier for this data source.
        name: Human-readable name of the data source.
        source_type: Type of data source (maps to a backend agent).
        endpoint_url: Base URL for the data source API.
        api_version: API version string for the data source.
        status: Current health status of the data source.
        capabilities: List of capability tags for this source.
        supported_operations: Query operations this source supports.
        health_check_url: URL for health check probes.
        last_health_check: Timestamp of the most recent health check.
        response_time_ms: Most recent response time in milliseconds.
        metadata: Additional key-value metadata for the source.
    """

    source_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique identifier for this data source",
    )
    name: str = Field(
        ..., description="Human-readable name of the data source",
    )
    source_type: DataSourceType = Field(
        ..., description="Type of data source (maps to a backend agent)",
    )
    endpoint_url: str = Field(
        ..., description="Base URL for the data source API",
    )
    api_version: str = Field(
        default="v1",
        description="API version string for the data source",
    )
    status: SourceStatus = Field(
        default=SourceStatus.UNKNOWN,
        description="Current health status of the data source",
    )
    capabilities: List[str] = Field(
        default_factory=list,
        description="List of capability tags for this source",
    )
    supported_operations: List[QueryOperation] = Field(
        default_factory=list,
        description="Query operations this source supports",
    )
    health_check_url: str = Field(
        default="",
        description="URL for health check probes",
    )
    last_health_check: Optional[datetime] = Field(
        None,
        description="Timestamp of the most recent health check",
    )
    response_time_ms: Optional[float] = Field(
        None, ge=0.0,
        description="Most recent response time in milliseconds",
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional key-value metadata for the source",
    )

    model_config = ConfigDict(from_attributes=True)

    @field_validator("name")
    @classmethod
    def validate_name(cls, v: str) -> str:
        """Validate name is non-empty."""
        if not v or not v.strip():
            raise ValueError("name must be non-empty")
        return v

    @field_validator("endpoint_url")
    @classmethod
    def validate_endpoint_url(cls, v: str) -> str:
        """Validate endpoint_url is non-empty."""
        if not v or not v.strip():
            raise ValueError("endpoint_url must be non-empty")
        return v


class SchemaField(BaseModel):
    """Definition of a single field within a schema.

    Describes the name, type, constraints, and source mapping for
    a field in a data source schema definition.

    Attributes:
        field_name: Name of the field in the schema.
        field_type: Data type of the field (e.g. string, integer, float).
        required: Whether this field is required in query results.
        description: Human-readable description of the field.
        source_field: Original field name in the source system.
        default_value: Default value if the field is not present.
    """

    field_name: str = Field(
        ..., description="Name of the field in the schema",
    )
    field_type: str = Field(
        ..., description="Data type of the field (e.g. string, integer, float)",
    )
    required: bool = Field(
        default=False,
        description="Whether this field is required in query results",
    )
    description: str = Field(
        default="",
        description="Human-readable description of the field",
    )
    source_field: str = Field(
        default="",
        description="Original field name in the source system",
    )
    default_value: Any = Field(
        default=None,
        description="Default value if the field is not present",
    )

    model_config = ConfigDict(from_attributes=True)

    @field_validator("field_name")
    @classmethod
    def validate_field_name(cls, v: str) -> str:
        """Validate field_name is non-empty."""
        if not v or not v.strip():
            raise ValueError("field_name must be non-empty")
        return v

    @field_validator("field_type")
    @classmethod
    def validate_field_type(cls, v: str) -> str:
        """Validate field_type is non-empty."""
        if not v or not v.strip():
            raise ValueError("field_type must be non-empty")
        return v


class SchemaDefinition(BaseModel):
    """Schema definition for a data source.

    Describes the structure and versioning of data produced by a
    backend data source, used for schema translation and validation.

    Attributes:
        schema_id: Unique identifier for this schema definition.
        schema_name: Human-readable name of the schema.
        version: Schema version string (e.g. 1.0.0).
        source_type: Data source type this schema applies to.
        fields: List of field definitions in this schema.
        created_at: Timestamp when the schema was created.
        updated_at: Timestamp of the most recent schema update.
    """

    schema_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique identifier for this schema definition",
    )
    schema_name: str = Field(
        ..., description="Human-readable name of the schema",
    )
    version: str = Field(
        ..., description="Schema version string (e.g. 1.0.0)",
    )
    source_type: DataSourceType = Field(
        ..., description="Data source type this schema applies to",
    )
    fields: List[SchemaField] = Field(
        default_factory=list,
        description="List of field definitions in this schema",
    )
    created_at: datetime = Field(
        default_factory=_utcnow,
        description="Timestamp when the schema was created",
    )
    updated_at: Optional[datetime] = Field(
        None,
        description="Timestamp of the most recent schema update",
    )

    model_config = ConfigDict(from_attributes=True)

    @field_validator("schema_name")
    @classmethod
    def validate_schema_name(cls, v: str) -> str:
        """Validate schema_name is non-empty."""
        if not v or not v.strip():
            raise ValueError("schema_name must be non-empty")
        return v

    @field_validator("version")
    @classmethod
    def validate_version(cls, v: str) -> str:
        """Validate version is non-empty."""
        if not v or not v.strip():
            raise ValueError("version must be non-empty")
        return v


class QueryFilter(BaseModel):
    """Filter condition for a query.

    Defines a single predicate that is applied to filter query results
    from one or more data sources.

    Attributes:
        field: Field name to filter on.
        operator: Comparison operator to apply.
        value: Value to compare against.
    """

    field: str = Field(
        ..., description="Field name to filter on",
    )
    operator: str = Field(
        ..., description=(
            "Comparison operator: eq, ne, gt, gte, lt, lte, in, not_in, "
            "contains, starts_with, ends_with, between, is_null, is_not_null"
        ),
    )
    value: Any = Field(
        ..., description="Value to compare against",
    )

    model_config = ConfigDict(from_attributes=True)

    @field_validator("field")
    @classmethod
    def validate_field(cls, v: str) -> str:
        """Validate field is non-empty."""
        if not v or not v.strip():
            raise ValueError("field must be non-empty")
        return v

    @field_validator("operator")
    @classmethod
    def validate_operator(cls, v: str) -> str:
        """Validate operator is one of the supported values."""
        valid_ops = {
            "eq", "ne", "gt", "gte", "lt", "lte", "in", "not_in",
            "contains", "starts_with", "ends_with", "between",
            "is_null", "is_not_null",
        }
        if v not in valid_ops:
            raise ValueError(
                f"operator must be one of {sorted(valid_ops)}, got '{v}'"
            )
        return v


class QuerySort(BaseModel):
    """Sort specification for query results.

    Defines how query results should be ordered by a given field.

    Attributes:
        field: Field name to sort by.
        order: Sort direction (ascending or descending).
    """

    field: str = Field(
        ..., description="Field name to sort by",
    )
    order: SortOrder = Field(
        default=SortOrder.ASC,
        description="Sort direction (ascending or descending)",
    )

    model_config = ConfigDict(from_attributes=True)

    @field_validator("field")
    @classmethod
    def validate_field(cls, v: str) -> str:
        """Validate field is non-empty."""
        if not v or not v.strip():
            raise ValueError("field must be non-empty")
        return v


class QueryAggregation(BaseModel):
    """Aggregation specification for query results.

    Defines an aggregation function to compute over query results,
    optionally grouped by one or more fields.

    Attributes:
        operation: Aggregation function to apply.
        field: Field name to aggregate on.
        alias: Optional alias for the aggregation result column.
        group_by: Fields to group results by before aggregating.
    """

    operation: AggregationType = Field(
        ..., description="Aggregation function to apply",
    )
    field: str = Field(
        ..., description="Field name to aggregate on",
    )
    alias: str = Field(
        default="",
        description="Optional alias for the aggregation result column",
    )
    group_by: List[str] = Field(
        default_factory=list,
        description="Fields to group results by before aggregating",
    )

    model_config = ConfigDict(from_attributes=True)

    @field_validator("field")
    @classmethod
    def validate_field(cls, v: str) -> str:
        """Validate field is non-empty."""
        if not v or not v.strip():
            raise ValueError("field must be non-empty")
        return v


class QueryPlan(BaseModel):
    """Execution plan for a gateway query.

    Represents the fully resolved query plan that the gateway will
    execute, including source routing, operations, filters, sorts,
    aggregations, and pagination parameters.

    Attributes:
        query_id: Unique identifier for this query plan.
        sources: List of source IDs to query.
        operations: List of query operations to perform.
        filters: List of filter conditions to apply.
        sorts: List of sort specifications for result ordering.
        aggregations: List of aggregation specifications.
        fields: List of field names to include in results.
        limit: Maximum number of result rows to return.
        offset: Number of result rows to skip (for pagination).
        timeout_seconds: Optional query-specific timeout override.
        complexity_score: Computed complexity score for this query.
    """

    query_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique identifier for this query plan",
    )
    sources: List[str] = Field(
        default_factory=list,
        description="List of source IDs to query",
    )
    operations: List[QueryOperation] = Field(
        default_factory=list,
        description="List of query operations to perform",
    )
    filters: List[QueryFilter] = Field(
        default_factory=list,
        description="List of filter conditions to apply",
    )
    sorts: List[QuerySort] = Field(
        default_factory=list,
        description="List of sort specifications for result ordering",
    )
    aggregations: List[QueryAggregation] = Field(
        default_factory=list,
        description="List of aggregation specifications",
    )
    fields: List[str] = Field(
        default_factory=list,
        description="List of field names to include in results",
    )
    limit: int = Field(
        default=100, ge=0,
        description="Maximum number of result rows to return",
    )
    offset: int = Field(
        default=0, ge=0,
        description="Number of result rows to skip (for pagination)",
    )
    timeout_seconds: Optional[int] = Field(
        None, ge=1,
        description="Optional query-specific timeout override",
    )
    complexity_score: float = Field(
        default=0.0, ge=0.0,
        description="Computed complexity score for this query",
    )

    model_config = ConfigDict(from_attributes=True)


class QueryResult(BaseModel):
    """Result of a gateway query execution.

    Contains the data rows, metadata, performance metrics, and
    provenance information from executing a query plan.

    Attributes:
        query_id: Query identifier that produced this result.
        status: Execution status of the query.
        data: List of result row dictionaries.
        total_count: Total number of matching rows (before pagination).
        sources_queried: List of source IDs that were queried.
        execution_time_ms: Total execution time in milliseconds.
        cache_hit: Whether the result was served from cache.
        errors: List of error messages from query execution.
        warnings: List of warning messages from query execution.
        metadata: Additional result metadata.
        provenance_hash: SHA-256 provenance chain hash for audit trail.
    """

    query_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Query identifier that produced this result",
    )
    status: QueryStatus = Field(
        default=QueryStatus.PENDING,
        description="Execution status of the query",
    )
    data: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="List of result row dictionaries",
    )
    total_count: int = Field(
        default=0, ge=0,
        description="Total number of matching rows (before pagination)",
    )
    sources_queried: List[str] = Field(
        default_factory=list,
        description="List of source IDs that were queried",
    )
    execution_time_ms: float = Field(
        default=0.0, ge=0.0,
        description="Total execution time in milliseconds",
    )
    cache_hit: bool = Field(
        default=False,
        description="Whether the result was served from cache",
    )
    errors: List[str] = Field(
        default_factory=list,
        description="List of error messages from query execution",
    )
    warnings: List[str] = Field(
        default_factory=list,
        description="List of warning messages from query execution",
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional result metadata",
    )
    provenance_hash: str = Field(
        default="",
        description="SHA-256 provenance chain hash for audit trail",
    )

    model_config = ConfigDict(from_attributes=True)


class CacheEntry(BaseModel):
    """Cached query result entry.

    Represents a single cached query result with metadata for
    TTL management, hit tracking, and size accounting.

    Attributes:
        cache_key: Unique cache key for this entry.
        query_hash: SHA-256 hash of the query that produced this result.
        source_id: Data source ID associated with this cached result.
        result_count: Number of result rows in this cached entry.
        size_bytes: Approximate size of the cached data in bytes.
        created_at: Timestamp when the cache entry was created.
        expires_at: Timestamp when the cache entry expires.
        hit_count: Number of times this cache entry has been served.
        last_accessed: Timestamp of the most recent cache hit.
    """

    cache_key: str = Field(
        ..., description="Unique cache key for this entry",
    )
    query_hash: str = Field(
        ..., description="SHA-256 hash of the query that produced this result",
    )
    source_id: str = Field(
        ..., description="Data source ID associated with this cached result",
    )
    result_count: int = Field(
        default=0, ge=0,
        description="Number of result rows in this cached entry",
    )
    size_bytes: int = Field(
        default=0, ge=0,
        description="Approximate size of the cached data in bytes",
    )
    created_at: datetime = Field(
        default_factory=_utcnow,
        description="Timestamp when the cache entry was created",
    )
    expires_at: datetime = Field(
        ..., description="Timestamp when the cache entry expires",
    )
    hit_count: int = Field(
        default=0, ge=0,
        description="Number of times this cache entry has been served",
    )
    last_accessed: Optional[datetime] = Field(
        None,
        description="Timestamp of the most recent cache hit",
    )

    model_config = ConfigDict(from_attributes=True)

    @field_validator("cache_key")
    @classmethod
    def validate_cache_key(cls, v: str) -> str:
        """Validate cache_key is non-empty."""
        if not v or not v.strip():
            raise ValueError("cache_key must be non-empty")
        return v

    @field_validator("query_hash")
    @classmethod
    def validate_query_hash(cls, v: str) -> str:
        """Validate query_hash is non-empty."""
        if not v or not v.strip():
            raise ValueError("query_hash must be non-empty")
        return v

    @field_validator("source_id")
    @classmethod
    def validate_source_id(cls, v: str) -> str:
        """Validate source_id is non-empty."""
        if not v or not v.strip():
            raise ValueError("source_id must be non-empty")
        return v


class SourceHealthCheck(BaseModel):
    """Health check result for a registered data source.

    Records the outcome of a single health probe against a
    backend data source, used for status tracking and alerting.

    Attributes:
        source_id: Data source that was checked.
        status: Health status determined by the check.
        response_time_ms: Response time of the health check in ms.
        checked_at: Timestamp when the health check was performed.
        error_message: Error message if the check failed.
        consecutive_failures: Number of consecutive failed checks.
    """

    source_id: str = Field(
        ..., description="Data source that was checked",
    )
    status: SourceStatus = Field(
        ..., description="Health status determined by the check",
    )
    response_time_ms: float = Field(
        ..., ge=0.0,
        description="Response time of the health check in milliseconds",
    )
    checked_at: datetime = Field(
        default_factory=_utcnow,
        description="Timestamp when the health check was performed",
    )
    error_message: str = Field(
        default="",
        description="Error message if the check failed",
    )
    consecutive_failures: int = Field(
        default=0, ge=0,
        description="Number of consecutive failed checks",
    )

    model_config = ConfigDict(from_attributes=True)

    @field_validator("source_id")
    @classmethod
    def validate_source_id(cls, v: str) -> str:
        """Validate source_id is non-empty."""
        if not v or not v.strip():
            raise ValueError("source_id must be non-empty")
        return v


class SchemaMapping(BaseModel):
    """Field-level mapping between source and target schemas.

    Defines how a single field is translated from one data source
    schema to another, including type conversion and transformation.

    Attributes:
        source_field: Field name in the source schema.
        target_field: Field name in the target schema.
        source_type: Data type in the source schema.
        target_type: Data type in the target schema.
        transform: Transformation expression to apply during mapping.
        description: Human-readable description of the mapping.
    """

    source_field: str = Field(
        ..., description="Field name in the source schema",
    )
    target_field: str = Field(
        ..., description="Field name in the target schema",
    )
    source_type: str = Field(
        ..., description="Data type in the source schema",
    )
    target_type: str = Field(
        ..., description="Data type in the target schema",
    )
    transform: str = Field(
        default="",
        description="Transformation expression to apply during mapping",
    )
    description: str = Field(
        default="",
        description="Human-readable description of the mapping",
    )

    model_config = ConfigDict(from_attributes=True)

    @field_validator("source_field")
    @classmethod
    def validate_source_field(cls, v: str) -> str:
        """Validate source_field is non-empty."""
        if not v or not v.strip():
            raise ValueError("source_field must be non-empty")
        return v

    @field_validator("target_field")
    @classmethod
    def validate_target_field(cls, v: str) -> str:
        """Validate target_field is non-empty."""
        if not v or not v.strip():
            raise ValueError("target_field must be non-empty")
        return v

    @field_validator("source_type")
    @classmethod
    def validate_source_type(cls, v: str) -> str:
        """Validate source_type is non-empty."""
        if not v or not v.strip():
            raise ValueError("source_type must be non-empty")
        return v

    @field_validator("target_type")
    @classmethod
    def validate_target_type(cls, v: str) -> str:
        """Validate target_type is non-empty."""
        if not v or not v.strip():
            raise ValueError("target_type must be non-empty")
        return v


class QueryTemplate(BaseModel):
    """Reusable query template with parameterization.

    Allows users to save and reuse common query patterns with
    configurable parameters for repeated execution.

    Attributes:
        template_id: Unique identifier for this query template.
        name: Human-readable name of the template.
        description: Description of the template purpose.
        query_plan: Serialized query plan dictionary.
        parameters: Default parameter values for the template.
        created_by: User or agent that created the template.
        created_at: Timestamp when the template was created.
        usage_count: Number of times this template has been executed.
    """

    template_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique identifier for this query template",
    )
    name: str = Field(
        ..., description="Human-readable name of the template",
    )
    description: str = Field(
        default="",
        description="Description of the template purpose",
    )
    query_plan: Dict[str, Any] = Field(
        default_factory=dict,
        description="Serialized query plan dictionary",
    )
    parameters: Dict[str, Any] = Field(
        default_factory=dict,
        description="Default parameter values for the template",
    )
    created_by: str = Field(
        default="",
        description="User or agent that created the template",
    )
    created_at: datetime = Field(
        default_factory=_utcnow,
        description="Timestamp when the template was created",
    )
    usage_count: int = Field(
        default=0, ge=0,
        description="Number of times this template has been executed",
    )

    model_config = ConfigDict(from_attributes=True)

    @field_validator("name")
    @classmethod
    def validate_name(cls, v: str) -> str:
        """Validate name is non-empty."""
        if not v or not v.strip():
            raise ValueError("name must be non-empty")
        return v


class DataCatalogEntry(BaseModel):
    """Catalog entry for a data source in the data catalog.

    Provides discovery metadata for a data source including domain,
    tags, available fields, and sample queries for self-service.

    Attributes:
        catalog_id: Unique identifier for this catalog entry.
        source_id: Data source this catalog entry describes.
        source_type: Type of data source.
        domain: Business domain this data source belongs to.
        name: Human-readable name of the catalog entry.
        description: Description of the data source for discovery.
        tags: Tags for search and classification.
        fields_available: List of field names available from this source.
        sample_queries: Example queries demonstrating usage.
        last_updated: Timestamp of the most recent catalog update.
    """

    catalog_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique identifier for this catalog entry",
    )
    source_id: str = Field(
        ..., description="Data source this catalog entry describes",
    )
    source_type: DataSourceType = Field(
        ..., description="Type of data source",
    )
    domain: str = Field(
        ..., description="Business domain this data source belongs to",
    )
    name: str = Field(
        ..., description="Human-readable name of the catalog entry",
    )
    description: str = Field(
        default="",
        description="Description of the data source for discovery",
    )
    tags: List[str] = Field(
        default_factory=list,
        description="Tags for search and classification",
    )
    fields_available: List[str] = Field(
        default_factory=list,
        description="List of field names available from this source",
    )
    sample_queries: List[str] = Field(
        default_factory=list,
        description="Example queries demonstrating usage",
    )
    last_updated: Optional[datetime] = Field(
        None,
        description="Timestamp of the most recent catalog update",
    )

    model_config = ConfigDict(from_attributes=True)

    @field_validator("source_id")
    @classmethod
    def validate_source_id(cls, v: str) -> str:
        """Validate source_id is non-empty."""
        if not v or not v.strip():
            raise ValueError("source_id must be non-empty")
        return v

    @field_validator("domain")
    @classmethod
    def validate_domain(cls, v: str) -> str:
        """Validate domain is non-empty."""
        if not v or not v.strip():
            raise ValueError("domain must be non-empty")
        return v

    @field_validator("name")
    @classmethod
    def validate_name(cls, v: str) -> str:
        """Validate name is non-empty."""
        if not v or not v.strip():
            raise ValueError("name must be non-empty")
        return v


class GatewayStatistics(BaseModel):
    """Aggregated statistics for the API Gateway service.

    Provides high-level operational metrics for monitoring the
    overall health, performance, and activity of the gateway.

    Attributes:
        total_queries: Total number of queries processed.
        cache_hit_rate: Percentage of queries served from cache.
        avg_response_time_ms: Average query response time in ms.
        active_sources: Total number of registered data sources.
        healthy_sources: Number of sources in healthy status.
        degraded_sources: Number of sources in degraded status.
        unhealthy_sources: Number of sources in unhealthy status.
        total_cache_entries: Total number of cached entries.
        cache_size_bytes: Total size of cached data in bytes.
        queries_by_source: Query count breakdown by source ID.
        errors_by_source: Error count breakdown by source ID.
    """

    total_queries: int = Field(
        default=0, ge=0,
        description="Total number of queries processed",
    )
    cache_hit_rate: float = Field(
        default=0.0, ge=0.0, le=100.0,
        description="Percentage of queries served from cache",
    )
    avg_response_time_ms: float = Field(
        default=0.0, ge=0.0,
        description="Average query response time in milliseconds",
    )
    active_sources: int = Field(
        default=0, ge=0,
        description="Total number of registered data sources",
    )
    healthy_sources: int = Field(
        default=0, ge=0,
        description="Number of sources in healthy status",
    )
    degraded_sources: int = Field(
        default=0, ge=0,
        description="Number of sources in degraded status",
    )
    unhealthy_sources: int = Field(
        default=0, ge=0,
        description="Number of sources in unhealthy status",
    )
    total_cache_entries: int = Field(
        default=0, ge=0,
        description="Total number of cached entries",
    )
    cache_size_bytes: int = Field(
        default=0, ge=0,
        description="Total size of cached data in bytes",
    )
    queries_by_source: Dict[str, int] = Field(
        default_factory=dict,
        description="Query count breakdown by source ID",
    )
    errors_by_source: Dict[str, int] = Field(
        default_factory=dict,
        description="Error count breakdown by source ID",
    )

    model_config = ConfigDict(from_attributes=True)


# =============================================================================
# Request Models
# =============================================================================


class ExecuteQueryRequest(BaseModel):
    """Request body for executing a query through the gateway.

    Attributes:
        sources: List of data source IDs to query.
        operations: List of query operation names to perform.
        filters: List of filter condition dictionaries.
        sorts: List of sort specification dictionaries.
        aggregations: List of aggregation specification dictionaries.
        fields: List of field names to include in results.
        limit: Maximum number of result rows to return.
        offset: Number of result rows to skip (for pagination).
        use_cache: Whether to use cached results if available.
        timeout_seconds: Optional query-specific timeout override.
    """

    sources: List[str] = Field(
        ..., description="List of data source IDs to query",
    )
    operations: List[str] = Field(
        default_factory=lambda: ["select"],
        description="List of query operation names to perform",
    )
    filters: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="List of filter condition dictionaries",
    )
    sorts: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="List of sort specification dictionaries",
    )
    aggregations: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="List of aggregation specification dictionaries",
    )
    fields: List[str] = Field(
        default_factory=list,
        description="List of field names to include in results",
    )
    limit: int = Field(
        default=100, ge=0,
        description="Maximum number of result rows to return",
    )
    offset: int = Field(
        default=0, ge=0,
        description="Number of result rows to skip (for pagination)",
    )
    use_cache: bool = Field(
        default=True,
        description="Whether to use cached results if available",
    )
    timeout_seconds: Optional[int] = Field(
        None, ge=1,
        description="Optional query-specific timeout override",
    )

    model_config = ConfigDict(extra="forbid")

    @field_validator("sources")
    @classmethod
    def validate_sources(cls, v: List[str]) -> List[str]:
        """Validate that at least one source is specified."""
        if not v:
            raise ValueError("sources must contain at least one source ID")
        return v


class BatchQueryRequest(BaseModel):
    """Request body for executing multiple queries in a batch.

    Attributes:
        queries: List of individual query requests to execute.
        parallel: Whether to execute queries in parallel.
    """

    queries: List[ExecuteQueryRequest] = Field(
        ..., description="List of individual query requests to execute",
    )
    parallel: bool = Field(
        default=True,
        description="Whether to execute queries in parallel",
    )

    model_config = ConfigDict(extra="forbid")

    @field_validator("queries")
    @classmethod
    def validate_queries(cls, v: List[ExecuteQueryRequest]) -> List[ExecuteQueryRequest]:
        """Validate that at least one query is provided."""
        if not v:
            raise ValueError("queries must contain at least one query")
        return v


class RegisterSourceRequest(BaseModel):
    """Request body for registering a new data source with the gateway.

    Attributes:
        name: Human-readable name of the data source.
        source_type: Type of data source to register.
        endpoint_url: Base URL for the data source API.
        api_version: API version string for the data source.
        health_check_url: URL for health check probes.
        capabilities: List of capability tags for this source.
        metadata: Additional key-value metadata for the source.
    """

    name: str = Field(
        ..., description="Human-readable name of the data source",
    )
    source_type: str = Field(
        ..., description="Type of data source to register",
    )
    endpoint_url: str = Field(
        ..., description="Base URL for the data source API",
    )
    api_version: str = Field(
        default="v1",
        description="API version string for the data source",
    )
    health_check_url: str = Field(
        default="",
        description="URL for health check probes",
    )
    capabilities: List[str] = Field(
        default_factory=list,
        description="List of capability tags for this source",
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional key-value metadata for the source",
    )

    model_config = ConfigDict(extra="forbid")

    @field_validator("name")
    @classmethod
    def validate_name(cls, v: str) -> str:
        """Validate name is non-empty."""
        if not v or not v.strip():
            raise ValueError("name must be non-empty")
        return v

    @field_validator("source_type")
    @classmethod
    def validate_source_type(cls, v: str) -> str:
        """Validate source_type is non-empty."""
        if not v or not v.strip():
            raise ValueError("source_type must be non-empty")
        return v

    @field_validator("endpoint_url")
    @classmethod
    def validate_endpoint_url(cls, v: str) -> str:
        """Validate endpoint_url is non-empty."""
        if not v or not v.strip():
            raise ValueError("endpoint_url must be non-empty")
        return v


class CreateTemplateRequest(BaseModel):
    """Request body for creating a reusable query template.

    Attributes:
        name: Human-readable name of the template.
        description: Description of the template purpose.
        query: Query request to save as a template.
        parameters: Default parameter values for the template.
    """

    name: str = Field(
        ..., description="Human-readable name of the template",
    )
    description: str = Field(
        default="",
        description="Description of the template purpose",
    )
    query: ExecuteQueryRequest = Field(
        ..., description="Query request to save as a template",
    )
    parameters: Dict[str, Any] = Field(
        default_factory=dict,
        description="Default parameter values for the template",
    )

    model_config = ConfigDict(extra="forbid")

    @field_validator("name")
    @classmethod
    def validate_name(cls, v: str) -> str:
        """Validate name is non-empty."""
        if not v or not v.strip():
            raise ValueError("name must be non-empty")
        return v


class TranslateSchemaRequest(BaseModel):
    """Request body for translating fields between source schemas.

    Attributes:
        source_type: Source schema type to translate from.
        target_type: Target schema type to translate to.
        fields: Field name-to-value mapping to translate.
    """

    source_type: str = Field(
        ..., description="Source schema type to translate from",
    )
    target_type: str = Field(
        ..., description="Target schema type to translate to",
    )
    fields: Dict[str, Any] = Field(
        ..., description="Field name-to-value mapping to translate",
    )

    model_config = ConfigDict(extra="forbid")

    @field_validator("source_type")
    @classmethod
    def validate_source_type(cls, v: str) -> str:
        """Validate source_type is non-empty."""
        if not v or not v.strip():
            raise ValueError("source_type must be non-empty")
        return v

    @field_validator("target_type")
    @classmethod
    def validate_target_type(cls, v: str) -> str:
        """Validate target_type is non-empty."""
        if not v or not v.strip():
            raise ValueError("target_type must be non-empty")
        return v


class CacheInvalidateRequest(BaseModel):
    """Request body for invalidating cached query results.

    Attributes:
        source_id: Optional source ID to invalidate cache entries for.
        query_hash: Optional query hash to invalidate a specific entry.
        invalidate_all: Whether to invalidate all cache entries.
    """

    source_id: Optional[str] = Field(
        None,
        description="Optional source ID to invalidate cache entries for",
    )
    query_hash: Optional[str] = Field(
        None,
        description="Optional query hash to invalidate a specific entry",
    )
    invalidate_all: bool = Field(
        default=False,
        description="Whether to invalidate all cache entries",
    )

    model_config = ConfigDict(extra="forbid")


__all__ = [
    # Enumerations
    "DataSourceType",
    "QueryOperation",
    "AggregationType",
    "SortOrder",
    "SourceStatus",
    "CacheStrategy",
    "QueryStatus",
    "ConflictResolution",
    # Core data models
    "DataSource",
    "SchemaField",
    "SchemaDefinition",
    "QueryFilter",
    "QuerySort",
    "QueryAggregation",
    "QueryPlan",
    "QueryResult",
    "CacheEntry",
    "SourceHealthCheck",
    "SchemaMapping",
    "QueryTemplate",
    "DataCatalogEntry",
    "GatewayStatistics",
    # Request models
    "ExecuteQueryRequest",
    "BatchQueryRequest",
    "RegisterSourceRequest",
    "CreateTemplateRequest",
    "TranslateSchemaRequest",
    "CacheInvalidateRequest",
]
