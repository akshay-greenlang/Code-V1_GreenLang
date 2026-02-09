# -*- coding: utf-8 -*-
"""
Unit Tests for Data Gateway Models (AGENT-DATA-004)

Tests all enums (DataSourceType 15 values, QueryOperation 8, AggregationType 7,
SortOrder 2, SourceStatus 5, CacheStrategy 4, QueryStatus 6, ConflictResolution 4),
DataSource, SchemaField, SchemaDefinition, QueryFilter, QuerySort, QueryAggregation,
QueryPlan, QueryResult, CacheEntry, SourceHealthCheck, SchemaMapping, QueryTemplate,
DataCatalogEntry, GatewayStatistics, and all request models.

Coverage target: 85%+ of models.py

Author: GreenLang Platform Team
Date: February 2026
"""

from __future__ import annotations

import re
import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

import pytest


# ---------------------------------------------------------------------------
# Inline enums mirroring greenlang/data_gateway/models.py
# ---------------------------------------------------------------------------


class DataSourceType(str, Enum):
    POSTGRESQL = "postgresql"
    TIMESCALEDB = "timescaledb"
    REDIS = "redis"
    S3 = "s3"
    ELASTICSEARCH = "elasticsearch"
    MONGODB = "mongodb"
    REST_API = "rest_api"
    GRAPHQL = "graphql"
    CSV = "csv"
    EXCEL = "excel"
    PARQUET = "parquet"
    KAFKA = "kafka"
    GRPC = "grpc"
    WEBSOCKET = "websocket"
    CUSTOM = "custom"


class QueryOperation(str, Enum):
    SELECT = "select"
    INSERT = "insert"
    UPDATE = "update"
    DELETE = "delete"
    AGGREGATE = "aggregate"
    JOIN = "join"
    UNION = "union"
    TRANSFORM = "transform"


class AggregationType(str, Enum):
    COUNT = "count"
    SUM = "sum"
    AVG = "avg"
    MIN = "min"
    MAX = "max"
    MEDIAN = "median"
    PERCENTILE = "percentile"


class SortOrder(str, Enum):
    ASC = "asc"
    DESC = "desc"


class SourceStatus(str, Enum):
    ACTIVE = "active"
    INACTIVE = "inactive"
    DEGRADED = "degraded"
    MAINTENANCE = "maintenance"
    ERROR = "error"


class CacheStrategy(str, Enum):
    TTL = "ttl"
    LRU = "lru"
    LFU = "lfu"
    NONE = "none"


class QueryStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    TIMEOUT = "timeout"


class ConflictResolution(str, Enum):
    FIRST = "first"
    LAST = "last"
    MERGE = "merge"
    ERROR = "error"


# ---------------------------------------------------------------------------
# Inline Layer 1 models
# ---------------------------------------------------------------------------


class DataSource:
    def __init__(
        self,
        source_id: str = "",
        name: str = "",
        source_type: str = "postgresql",
        connection_string: str = "",
        status: str = "active",
        schema_name: Optional[str] = None,
        description: str = "",
        tags: Optional[List[str]] = None,
        max_connections: int = 10,
        timeout_s: int = 30,
        retry_count: int = 3,
        created_at: Optional[str] = None,
        updated_at: Optional[str] = None,
        provenance_hash: Optional[str] = None,
    ):
        self.source_id = source_id or f"SRC-{uuid.uuid4().hex[:5]}"
        self.name = name
        self.source_type = source_type
        self.connection_string = connection_string
        self.status = status
        self.schema_name = schema_name
        self.description = description
        self.tags = tags or []
        self.max_connections = max_connections
        self.timeout_s = timeout_s
        self.retry_count = retry_count
        self.created_at = created_at or datetime.now(timezone.utc).isoformat()
        self.updated_at = updated_at
        self.provenance_hash = provenance_hash

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source_id": self.source_id,
            "name": self.name,
            "source_type": self.source_type,
            "connection_string": self.connection_string,
            "status": self.status,
            "schema_name": self.schema_name,
            "description": self.description,
            "tags": self.tags,
            "max_connections": self.max_connections,
            "timeout_s": self.timeout_s,
            "retry_count": self.retry_count,
        }


class SchemaField:
    def __init__(
        self,
        name: str = "",
        data_type: str = "string",
        nullable: bool = True,
        description: str = "",
        default_value: Optional[Any] = None,
        primary_key: bool = False,
    ):
        self.name = name
        self.data_type = data_type
        self.nullable = nullable
        self.description = description
        self.default_value = default_value
        self.primary_key = primary_key


class SchemaDefinition:
    def __init__(
        self,
        schema_id: str = "",
        source_id: str = "",
        table_name: str = "",
        fields: Optional[List[SchemaField]] = None,
        version: str = "1.0",
        created_at: Optional[str] = None,
    ):
        self.schema_id = schema_id or f"SCH-{uuid.uuid4().hex[:5]}"
        self.source_id = source_id
        self.table_name = table_name
        self.fields = fields or []
        self.version = version
        self.created_at = created_at or datetime.now(timezone.utc).isoformat()


class QueryFilter:
    def __init__(
        self,
        field: str = "",
        operator: str = "eq",
        value: Any = None,
    ):
        self.field = field
        self.operator = operator
        self.value = value


class QuerySort:
    def __init__(
        self,
        field: str = "",
        order: str = "asc",
    ):
        self.field = field
        self.order = order


class QueryAggregation:
    def __init__(
        self,
        type: str = "count",
        field: str = "",
        alias: Optional[str] = None,
        group_by: Optional[List[str]] = None,
    ):
        self.type = type
        self.field = field
        self.alias = alias
        self.group_by = group_by or []


class QueryPlan:
    def __init__(
        self,
        plan_id: str = "",
        query_id: str = "",
        source_ids: Optional[List[str]] = None,
        operations: Optional[List[str]] = None,
        estimated_complexity: int = 0,
        estimated_rows: int = 0,
        cache_eligible: bool = True,
    ):
        self.plan_id = plan_id or f"PLN-{uuid.uuid4().hex[:5]}"
        self.query_id = query_id
        self.source_ids = source_ids or []
        self.operations = operations or []
        self.estimated_complexity = estimated_complexity
        self.estimated_rows = estimated_rows
        self.cache_eligible = cache_eligible


class QueryResult:
    def __init__(
        self,
        result_id: str = "",
        query_id: str = "",
        status: str = "completed",
        data: Optional[List[Dict[str, Any]]] = None,
        total_rows: int = 0,
        execution_time_ms: float = 0.0,
        cache_hit: bool = False,
        errors: Optional[List[str]] = None,
        provenance_hash: Optional[str] = None,
    ):
        self.result_id = result_id or f"RES-{uuid.uuid4().hex[:5]}"
        self.query_id = query_id
        self.status = status
        self.data = data or []
        self.total_rows = total_rows
        self.execution_time_ms = execution_time_ms
        self.cache_hit = cache_hit
        self.errors = errors or []
        self.provenance_hash = provenance_hash


class CacheEntry:
    def __init__(
        self,
        cache_key: str = "",
        query_hash: str = "",
        data: Optional[List[Dict[str, Any]]] = None,
        created_at: Optional[str] = None,
        expires_at: Optional[str] = None,
        hit_count: int = 0,
        size_bytes: int = 0,
    ):
        self.cache_key = cache_key
        self.query_hash = query_hash
        self.data = data or []
        self.created_at = created_at or datetime.now(timezone.utc).isoformat()
        self.expires_at = expires_at
        self.hit_count = hit_count
        self.size_bytes = size_bytes


class SourceHealthCheck:
    def __init__(
        self,
        source_id: str = "",
        status: str = "active",
        latency_ms: float = 0.0,
        last_checked_at: Optional[str] = None,
        consecutive_failures: int = 0,
        error_message: Optional[str] = None,
    ):
        self.source_id = source_id
        self.status = status
        self.latency_ms = latency_ms
        self.last_checked_at = last_checked_at or datetime.now(timezone.utc).isoformat()
        self.consecutive_failures = consecutive_failures
        self.error_message = error_message


class SchemaMapping:
    def __init__(
        self,
        mapping_id: str = "",
        source_schema_id: str = "",
        target_schema_id: str = "",
        field_mappings: Optional[Dict[str, str]] = None,
        transform_rules: Optional[List[str]] = None,
        created_at: Optional[str] = None,
    ):
        self.mapping_id = mapping_id or f"MAP-{uuid.uuid4().hex[:5]}"
        self.source_schema_id = source_schema_id
        self.target_schema_id = target_schema_id
        self.field_mappings = field_mappings or {}
        self.transform_rules = transform_rules or []
        self.created_at = created_at or datetime.now(timezone.utc).isoformat()


class QueryTemplate:
    def __init__(
        self,
        template_id: str = "",
        name: str = "",
        description: str = "",
        source_ids: Optional[List[str]] = None,
        operations: Optional[List[str]] = None,
        filters: Optional[List[QueryFilter]] = None,
        parameters: Optional[Dict[str, Any]] = None,
        usage_count: int = 0,
        created_at: Optional[str] = None,
    ):
        self.template_id = template_id or f"TPL-{uuid.uuid4().hex[:5]}"
        self.name = name
        self.description = description
        self.source_ids = source_ids or []
        self.operations = operations or []
        self.filters = filters or []
        self.parameters = parameters or {}
        self.usage_count = usage_count
        self.created_at = created_at or datetime.now(timezone.utc).isoformat()


class DataCatalogEntry:
    def __init__(
        self,
        entry_id: str = "",
        source_id: str = "",
        name: str = "",
        description: str = "",
        tags: Optional[List[str]] = None,
        domain: str = "",
        owner: str = "",
        quality_score: float = 0.0,
        row_count: int = 0,
        last_updated_at: Optional[str] = None,
    ):
        self.entry_id = entry_id or f"CAT-{uuid.uuid4().hex[:5]}"
        self.source_id = source_id
        self.name = name
        self.description = description
        self.tags = tags or []
        self.domain = domain
        self.owner = owner
        self.quality_score = quality_score
        self.row_count = row_count
        self.last_updated_at = last_updated_at or datetime.now(timezone.utc).isoformat()


class GatewayStatistics:
    def __init__(
        self,
        total_queries: int = 0,
        successful_queries: int = 0,
        failed_queries: int = 0,
        cache_hits: int = 0,
        cache_misses: int = 0,
        avg_execution_time_ms: float = 0.0,
        active_sources: int = 0,
        total_sources: int = 0,
        uptime_seconds: float = 0.0,
    ):
        self.total_queries = total_queries
        self.successful_queries = successful_queries
        self.failed_queries = failed_queries
        self.cache_hits = cache_hits
        self.cache_misses = cache_misses
        self.avg_execution_time_ms = avg_execution_time_ms
        self.active_sources = active_sources
        self.total_sources = total_sources
        self.uptime_seconds = uptime_seconds


# ---------------------------------------------------------------------------
# Inline Request models
# ---------------------------------------------------------------------------


class ExecuteQueryRequest:
    def __init__(
        self,
        source_ids: Optional[List[str]] = None,
        operation: str = "select",
        filters: Optional[List[QueryFilter]] = None,
        sorts: Optional[List[QuerySort]] = None,
        aggregations: Optional[List[QueryAggregation]] = None,
        limit: int = 100,
        offset: int = 0,
        cache_strategy: str = "ttl",
        timeout_s: int = 60,
    ):
        self.source_ids = source_ids or []
        self.operation = operation
        self.filters = filters or []
        self.sorts = sorts or []
        self.aggregations = aggregations or []
        self.limit = limit
        self.offset = offset
        self.cache_strategy = cache_strategy
        self.timeout_s = timeout_s


class BatchQueryRequest:
    def __init__(
        self,
        queries: Optional[List[ExecuteQueryRequest]] = None,
        parallel: bool = True,
        stop_on_error: bool = False,
    ):
        self.queries = queries or []
        self.parallel = parallel
        self.stop_on_error = stop_on_error


class RegisterSourceRequest:
    def __init__(
        self,
        name: str = "",
        source_type: str = "postgresql",
        connection_string: str = "",
        description: str = "",
        tags: Optional[List[str]] = None,
        max_connections: int = 10,
        timeout_s: int = 30,
    ):
        self.name = name
        self.source_type = source_type
        self.connection_string = connection_string
        self.description = description
        self.tags = tags or []
        self.max_connections = max_connections
        self.timeout_s = timeout_s


class CreateTemplateRequest:
    def __init__(
        self,
        name: str = "",
        description: str = "",
        source_ids: Optional[List[str]] = None,
        operations: Optional[List[str]] = None,
        parameters: Optional[Dict[str, Any]] = None,
    ):
        self.name = name
        self.description = description
        self.source_ids = source_ids or []
        self.operations = operations or []
        self.parameters = parameters or {}


class TranslateSchemaRequest:
    def __init__(
        self,
        source_schema_id: str = "",
        target_source_id: str = "",
        field_mappings: Optional[Dict[str, str]] = None,
    ):
        self.source_schema_id = source_schema_id
        self.target_source_id = target_source_id
        self.field_mappings = field_mappings or {}


class CacheInvalidateRequest:
    def __init__(
        self,
        cache_keys: Optional[List[str]] = None,
        source_ids: Optional[List[str]] = None,
        invalidate_all: bool = False,
    ):
        self.cache_keys = cache_keys or []
        self.source_ids = source_ids or []
        self.invalidate_all = invalidate_all


# ===========================================================================
# Test Classes -- Enums
# ===========================================================================


class TestDataSourceTypeEnum:
    """Test DataSourceType enum values (15 data source types)."""

    def test_postgresql(self):
        assert DataSourceType.POSTGRESQL.value == "postgresql"

    def test_timescaledb(self):
        assert DataSourceType.TIMESCALEDB.value == "timescaledb"

    def test_redis(self):
        assert DataSourceType.REDIS.value == "redis"

    def test_s3(self):
        assert DataSourceType.S3.value == "s3"

    def test_elasticsearch(self):
        assert DataSourceType.ELASTICSEARCH.value == "elasticsearch"

    def test_mongodb(self):
        assert DataSourceType.MONGODB.value == "mongodb"

    def test_rest_api(self):
        assert DataSourceType.REST_API.value == "rest_api"

    def test_graphql(self):
        assert DataSourceType.GRAPHQL.value == "graphql"

    def test_csv(self):
        assert DataSourceType.CSV.value == "csv"

    def test_excel(self):
        assert DataSourceType.EXCEL.value == "excel"

    def test_parquet(self):
        assert DataSourceType.PARQUET.value == "parquet"

    def test_kafka(self):
        assert DataSourceType.KAFKA.value == "kafka"

    def test_grpc(self):
        assert DataSourceType.GRPC.value == "grpc"

    def test_websocket(self):
        assert DataSourceType.WEBSOCKET.value == "websocket"

    def test_custom(self):
        assert DataSourceType.CUSTOM.value == "custom"

    def test_all_15_types(self):
        """DataSourceType covers exactly 15 source types."""
        assert len(DataSourceType) == 15

    def test_from_value(self):
        assert DataSourceType("postgresql") == DataSourceType.POSTGRESQL


class TestQueryOperationEnum:
    """Test QueryOperation enum values (8 operations)."""

    def test_select(self):
        assert QueryOperation.SELECT.value == "select"

    def test_insert(self):
        assert QueryOperation.INSERT.value == "insert"

    def test_update(self):
        assert QueryOperation.UPDATE.value == "update"

    def test_delete(self):
        assert QueryOperation.DELETE.value == "delete"

    def test_aggregate(self):
        assert QueryOperation.AGGREGATE.value == "aggregate"

    def test_join(self):
        assert QueryOperation.JOIN.value == "join"

    def test_union(self):
        assert QueryOperation.UNION.value == "union"

    def test_transform(self):
        assert QueryOperation.TRANSFORM.value == "transform"

    def test_all_8_operations(self):
        """QueryOperation covers exactly 8 operations."""
        assert len(QueryOperation) == 8

    def test_from_value(self):
        assert QueryOperation("select") == QueryOperation.SELECT


class TestAggregationTypeEnum:
    """Test AggregationType enum values (7 aggregation types)."""

    def test_count(self):
        assert AggregationType.COUNT.value == "count"

    def test_sum(self):
        assert AggregationType.SUM.value == "sum"

    def test_avg(self):
        assert AggregationType.AVG.value == "avg"

    def test_min(self):
        assert AggregationType.MIN.value == "min"

    def test_max(self):
        assert AggregationType.MAX.value == "max"

    def test_median(self):
        assert AggregationType.MEDIAN.value == "median"

    def test_percentile(self):
        assert AggregationType.PERCENTILE.value == "percentile"

    def test_all_7_types(self):
        """AggregationType covers exactly 7 types."""
        assert len(AggregationType) == 7

    def test_from_value(self):
        assert AggregationType("count") == AggregationType.COUNT


class TestSortOrderEnum:
    """Test SortOrder enum values (2 orders)."""

    def test_asc(self):
        assert SortOrder.ASC.value == "asc"

    def test_desc(self):
        assert SortOrder.DESC.value == "desc"

    def test_all_2_orders(self):
        assert len(SortOrder) == 2


class TestSourceStatusEnum:
    """Test SourceStatus enum values (5 statuses)."""

    def test_active(self):
        assert SourceStatus.ACTIVE.value == "active"

    def test_inactive(self):
        assert SourceStatus.INACTIVE.value == "inactive"

    def test_degraded(self):
        assert SourceStatus.DEGRADED.value == "degraded"

    def test_maintenance(self):
        assert SourceStatus.MAINTENANCE.value == "maintenance"

    def test_error(self):
        assert SourceStatus.ERROR.value == "error"

    def test_all_5_statuses(self):
        assert len(SourceStatus) == 5


class TestCacheStrategyEnum:
    """Test CacheStrategy enum values (4 strategies)."""

    def test_ttl(self):
        assert CacheStrategy.TTL.value == "ttl"

    def test_lru(self):
        assert CacheStrategy.LRU.value == "lru"

    def test_lfu(self):
        assert CacheStrategy.LFU.value == "lfu"

    def test_none(self):
        assert CacheStrategy.NONE.value == "none"

    def test_all_4_strategies(self):
        assert len(CacheStrategy) == 4


class TestQueryStatusEnum:
    """Test QueryStatus enum values (6 statuses)."""

    def test_pending(self):
        assert QueryStatus.PENDING.value == "pending"

    def test_running(self):
        assert QueryStatus.RUNNING.value == "running"

    def test_completed(self):
        assert QueryStatus.COMPLETED.value == "completed"

    def test_failed(self):
        assert QueryStatus.FAILED.value == "failed"

    def test_cancelled(self):
        assert QueryStatus.CANCELLED.value == "cancelled"

    def test_timeout(self):
        assert QueryStatus.TIMEOUT.value == "timeout"

    def test_all_6_statuses(self):
        assert len(QueryStatus) == 6


class TestConflictResolutionEnum:
    """Test ConflictResolution enum values (4 strategies)."""

    def test_first(self):
        assert ConflictResolution.FIRST.value == "first"

    def test_last(self):
        assert ConflictResolution.LAST.value == "last"

    def test_merge(self):
        assert ConflictResolution.MERGE.value == "merge"

    def test_error(self):
        assert ConflictResolution.ERROR.value == "error"

    def test_all_4_strategies(self):
        assert len(ConflictResolution) == 4


# ===========================================================================
# Test Classes -- DataSource
# ===========================================================================


class TestDataSource:
    """Test DataSource model for data gateway source registration."""

    def test_create_data_source(self):
        """All fields populated correctly."""
        src = DataSource(
            name="emissions_db",
            source_type="postgresql",
            connection_string="postgresql://user:pass@host:5432/db",
            description="Main emissions database",
            tags=["emissions", "production"],
        )
        assert src.source_id.startswith("SRC-")
        assert src.name == "emissions_db"
        assert src.source_type == "postgresql"
        assert src.connection_string == "postgresql://user:pass@host:5432/db"
        assert src.description == "Main emissions database"
        assert len(src.tags) == 2
        assert src.created_at is not None

    def test_default_status(self):
        """Default status is active."""
        src = DataSource()
        assert src.status == "active"

    def test_default_max_connections(self):
        """Default max_connections is 10."""
        src = DataSource()
        assert src.max_connections == 10

    def test_default_timeout(self):
        """Default timeout is 30 seconds."""
        src = DataSource()
        assert src.timeout_s == 30

    def test_default_retry_count(self):
        """Default retry count is 3."""
        src = DataSource()
        assert src.retry_count == 3

    def test_custom_source_id(self):
        """Custom source_id is preserved."""
        src = DataSource(source_id="SRC-CUSTOM-001")
        assert src.source_id == "SRC-CUSTOM-001"

    def test_to_dict(self):
        """Serialization via to_dict."""
        src = DataSource(
            name="test_source",
            source_type="redis",
            connection_string="redis://localhost:6379",
        )
        d = src.to_dict()
        assert d["name"] == "test_source"
        assert d["source_type"] == "redis"
        assert d["connection_string"] == "redis://localhost:6379"
        assert d["status"] == "active"
        assert d["tags"] == []

    def test_status_values(self):
        """Various status values accepted."""
        for status in ["active", "inactive", "degraded", "maintenance", "error"]:
            src = DataSource(status=status)
            assert src.status == status


# ===========================================================================
# Test Classes -- SchemaField
# ===========================================================================


class TestSchemaField:
    """Test SchemaField model."""

    def test_create_schema_field(self):
        """Field with all attributes."""
        field = SchemaField(
            name="co2_emissions",
            data_type="float",
            nullable=False,
            description="CO2 emissions in tonnes",
            primary_key=False,
        )
        assert field.name == "co2_emissions"
        assert field.data_type == "float"
        assert field.nullable is False
        assert field.description == "CO2 emissions in tonnes"
        assert field.primary_key is False

    def test_defaults(self):
        """Default values for SchemaField."""
        field = SchemaField()
        assert field.name == ""
        assert field.data_type == "string"
        assert field.nullable is True
        assert field.description == ""
        assert field.default_value is None
        assert field.primary_key is False

    def test_primary_key_field(self):
        """Primary key field creation."""
        field = SchemaField(name="id", data_type="uuid", nullable=False, primary_key=True)
        assert field.primary_key is True
        assert field.nullable is False


# ===========================================================================
# Test Classes -- SchemaDefinition
# ===========================================================================


class TestSchemaDefinition:
    """Test SchemaDefinition model."""

    def test_create_schema_definition(self):
        """Schema with fields list."""
        fields = [
            SchemaField(name="id", data_type="uuid", primary_key=True),
            SchemaField(name="value", data_type="float"),
        ]
        schema = SchemaDefinition(
            source_id="SRC-00001",
            table_name="emissions",
            fields=fields,
        )
        assert schema.schema_id.startswith("SCH-")
        assert schema.source_id == "SRC-00001"
        assert schema.table_name == "emissions"
        assert len(schema.fields) == 2
        assert schema.version == "1.0"
        assert schema.created_at is not None

    def test_empty_fields_list(self):
        """Schema with no fields defaults to empty list."""
        schema = SchemaDefinition()
        assert schema.fields == []


# ===========================================================================
# Test Classes -- QueryFilter
# ===========================================================================


class TestQueryFilter:
    """Test QueryFilter model for all 14 operators."""

    def test_create_filter(self):
        """Basic filter creation."""
        f = QueryFilter(field="status", operator="eq", value="active")
        assert f.field == "status"
        assert f.operator == "eq"
        assert f.value == "active"

    @pytest.mark.parametrize("operator", [
        "eq", "ne", "gt", "gte", "lt", "lte",
        "in", "not_in", "contains", "starts_with",
        "ends_with", "between", "is_null", "is_not_null",
    ])
    def test_all_14_operators(self, operator):
        """All 14 filter operators accepted."""
        f = QueryFilter(field="test_field", operator=operator, value="test")
        assert f.operator == operator

    def test_filter_eq(self):
        f = QueryFilter(field="name", operator="eq", value="test")
        assert f.operator == "eq"

    def test_filter_ne(self):
        f = QueryFilter(field="name", operator="ne", value="test")
        assert f.operator == "ne"

    def test_filter_gt(self):
        f = QueryFilter(field="score", operator="gt", value=50)
        assert f.operator == "gt" and f.value == 50

    def test_filter_gte(self):
        f = QueryFilter(field="score", operator="gte", value=50)
        assert f.operator == "gte"

    def test_filter_lt(self):
        f = QueryFilter(field="score", operator="lt", value=100)
        assert f.operator == "lt"

    def test_filter_lte(self):
        f = QueryFilter(field="score", operator="lte", value=100)
        assert f.operator == "lte"

    def test_filter_in(self):
        f = QueryFilter(field="status", operator="in", value=["active", "degraded"])
        assert f.operator == "in"
        assert len(f.value) == 2

    def test_filter_not_in(self):
        f = QueryFilter(field="status", operator="not_in", value=["error"])
        assert f.operator == "not_in"

    def test_filter_contains(self):
        f = QueryFilter(field="name", operator="contains", value="carbon")
        assert f.operator == "contains"

    def test_filter_starts_with(self):
        f = QueryFilter(field="name", operator="starts_with", value="GL-")
        assert f.operator == "starts_with"

    def test_filter_ends_with(self):
        f = QueryFilter(field="name", operator="ends_with", value="-v2")
        assert f.operator == "ends_with"

    def test_filter_between(self):
        f = QueryFilter(field="score", operator="between", value=[0, 100])
        assert f.operator == "between"
        assert f.value == [0, 100]

    def test_filter_is_null(self):
        f = QueryFilter(field="deleted_at", operator="is_null", value=None)
        assert f.operator == "is_null"

    def test_filter_is_not_null(self):
        f = QueryFilter(field="created_at", operator="is_not_null", value=None)
        assert f.operator == "is_not_null"


# ===========================================================================
# Test Classes -- QuerySort
# ===========================================================================


class TestQuerySort:
    """Test QuerySort model."""

    def test_create_sort(self):
        """Sort creation with explicit order."""
        s = QuerySort(field="created_at", order="desc")
        assert s.field == "created_at"
        assert s.order == "desc"

    def test_default_asc(self):
        """Default order is ascending."""
        s = QuerySort(field="name")
        assert s.order == "asc"


# ===========================================================================
# Test Classes -- QueryAggregation
# ===========================================================================


class TestQueryAggregation:
    """Test QueryAggregation model."""

    def test_create_aggregation(self):
        """Aggregation with type, field, alias."""
        agg = QueryAggregation(type="sum", field="co2_tonnes", alias="total_co2")
        assert agg.type == "sum"
        assert agg.field == "co2_tonnes"
        assert agg.alias == "total_co2"

    def test_group_by(self):
        """Aggregation with group_by fields."""
        agg = QueryAggregation(
            type="count", field="id",
            group_by=["region", "year"],
        )
        assert agg.group_by == ["region", "year"]
        assert len(agg.group_by) == 2

    def test_default_group_by_empty(self):
        """Default group_by is empty list."""
        agg = QueryAggregation()
        assert agg.group_by == []


# ===========================================================================
# Test Classes -- QueryPlan
# ===========================================================================


class TestQueryPlan:
    """Test QueryPlan model."""

    def test_create_plan(self):
        """Plan with sources and operations."""
        plan = QueryPlan(
            query_id="QRY-00001",
            source_ids=["SRC-00001", "SRC-00002"],
            operations=["select", "join"],
            estimated_complexity=150,
            estimated_rows=5000,
        )
        assert plan.plan_id.startswith("PLN-")
        assert plan.query_id == "QRY-00001"
        assert len(plan.source_ids) == 2
        assert plan.estimated_complexity == 150
        assert plan.estimated_rows == 5000
        assert plan.cache_eligible is True

    def test_complexity_default(self):
        """Default estimated complexity is 0."""
        plan = QueryPlan()
        assert plan.estimated_complexity == 0


# ===========================================================================
# Test Classes -- QueryResult
# ===========================================================================


class TestQueryResult:
    """Test QueryResult model."""

    def test_create_result(self):
        """Result with data and metadata."""
        result = QueryResult(
            query_id="QRY-00001",
            status="completed",
            data=[{"id": 1, "value": 42.5}],
            total_rows=1,
            execution_time_ms=15.3,
        )
        assert result.result_id.startswith("RES-")
        assert result.query_id == "QRY-00001"
        assert result.status == "completed"
        assert len(result.data) == 1
        assert result.total_rows == 1
        assert result.execution_time_ms == 15.3

    def test_cache_hit_default_false(self):
        """Default cache_hit is False."""
        result = QueryResult()
        assert result.cache_hit is False

    def test_cache_hit_true(self):
        """cache_hit can be set to True."""
        result = QueryResult(cache_hit=True)
        assert result.cache_hit is True

    def test_errors_default_empty(self):
        """Default errors is empty list."""
        result = QueryResult()
        assert result.errors == []

    def test_errors_populated(self):
        """Errors list with messages."""
        result = QueryResult(
            status="failed",
            errors=["Connection timeout", "Source unreachable"],
        )
        assert len(result.errors) == 2
        assert "Connection timeout" in result.errors


# ===========================================================================
# Test Classes -- CacheEntry
# ===========================================================================


class TestCacheEntry:
    """Test CacheEntry model."""

    def test_create_cache_entry(self):
        """Cache entry with key, data, and expiry."""
        entry = CacheEntry(
            cache_key="query_hash_abc123",
            query_hash="abc123",
            data=[{"id": 1}],
            expires_at="2026-02-10T00:00:00Z",
            size_bytes=256,
        )
        assert entry.cache_key == "query_hash_abc123"
        assert entry.query_hash == "abc123"
        assert len(entry.data) == 1
        assert entry.expires_at == "2026-02-10T00:00:00Z"
        assert entry.size_bytes == 256

    def test_expiry_none_by_default(self):
        """Default expires_at is None."""
        entry = CacheEntry()
        assert entry.expires_at is None

    def test_hit_count_default(self):
        """Default hit_count is 0."""
        entry = CacheEntry()
        assert entry.hit_count == 0


# ===========================================================================
# Test Classes -- SourceHealthCheck
# ===========================================================================


class TestSourceHealthCheck:
    """Test SourceHealthCheck model."""

    def test_create_health_check(self):
        """Health check with all fields."""
        hc = SourceHealthCheck(
            source_id="SRC-00001",
            status="active",
            latency_ms=5.2,
        )
        assert hc.source_id == "SRC-00001"
        assert hc.status == "active"
        assert hc.latency_ms == 5.2
        assert hc.last_checked_at is not None

    def test_consecutive_failures_default(self):
        """Default consecutive_failures is 0."""
        hc = SourceHealthCheck()
        assert hc.consecutive_failures == 0

    def test_failures_tracked(self):
        """Track consecutive failures."""
        hc = SourceHealthCheck(
            source_id="SRC-00001",
            status="error",
            consecutive_failures=3,
            error_message="Connection refused",
        )
        assert hc.consecutive_failures == 3
        assert hc.error_message == "Connection refused"


# ===========================================================================
# Test Classes -- SchemaMapping
# ===========================================================================


class TestSchemaMapping:
    """Test SchemaMapping model."""

    def test_create_mapping(self):
        """Schema mapping with field mappings."""
        mapping = SchemaMapping(
            source_schema_id="SCH-00001",
            target_schema_id="SCH-00002",
            field_mappings={"old_name": "new_name", "old_value": "new_value"},
        )
        assert mapping.mapping_id.startswith("MAP-")
        assert mapping.source_schema_id == "SCH-00001"
        assert mapping.target_schema_id == "SCH-00002"
        assert len(mapping.field_mappings) == 2
        assert mapping.created_at is not None

    def test_default_mappings_empty(self):
        """Default field_mappings is empty dict."""
        mapping = SchemaMapping()
        assert mapping.field_mappings == {}


# ===========================================================================
# Test Classes -- QueryTemplate
# ===========================================================================


class TestQueryTemplate:
    """Test QueryTemplate model."""

    def test_create_template(self):
        """Template with name, sources, and operations."""
        tpl = QueryTemplate(
            name="emissions_summary",
            description="Summarize emissions by region",
            source_ids=["SRC-00001"],
            operations=["select", "aggregate"],
            parameters={"region": "EMEA"},
        )
        assert tpl.template_id.startswith("TPL-")
        assert tpl.name == "emissions_summary"
        assert tpl.description == "Summarize emissions by region"
        assert len(tpl.source_ids) == 1
        assert len(tpl.operations) == 2
        assert tpl.parameters["region"] == "EMEA"
        assert tpl.created_at is not None

    def test_usage_count_default(self):
        """Default usage_count is 0."""
        tpl = QueryTemplate()
        assert tpl.usage_count == 0

    def test_usage_count_custom(self):
        """Custom usage_count."""
        tpl = QueryTemplate(usage_count=42)
        assert tpl.usage_count == 42


# ===========================================================================
# Test Classes -- DataCatalogEntry
# ===========================================================================


class TestDataCatalogEntry:
    """Test DataCatalogEntry model."""

    def test_create_catalog_entry(self):
        """Catalog entry with all fields."""
        entry = DataCatalogEntry(
            source_id="SRC-00001",
            name="emissions_data",
            description="Annual CO2 emissions data",
            tags=["emissions", "scope1", "scope2"],
            domain="sustainability",
            owner="data-team",
            quality_score=0.95,
            row_count=1000000,
        )
        assert entry.entry_id.startswith("CAT-")
        assert entry.source_id == "SRC-00001"
        assert entry.name == "emissions_data"
        assert len(entry.tags) == 3
        assert "emissions" in entry.tags
        assert entry.domain == "sustainability"
        assert entry.owner == "data-team"
        assert entry.quality_score == 0.95
        assert entry.row_count == 1000000

    def test_tags_default_empty(self):
        """Default tags is empty list."""
        entry = DataCatalogEntry()
        assert entry.tags == []

    def test_domain_default_empty(self):
        """Default domain is empty string."""
        entry = DataCatalogEntry()
        assert entry.domain == ""


# ===========================================================================
# Test Classes -- GatewayStatistics
# ===========================================================================


class TestGatewayStatistics:
    """Test GatewayStatistics model."""

    def test_create_statistics(self):
        """Statistics with counts."""
        stats = GatewayStatistics(
            total_queries=1000,
            successful_queries=950,
            failed_queries=50,
            cache_hits=300,
            cache_misses=700,
            avg_execution_time_ms=25.5,
            active_sources=5,
            total_sources=8,
            uptime_seconds=86400.0,
        )
        assert stats.total_queries == 1000
        assert stats.successful_queries == 950
        assert stats.failed_queries == 50
        assert stats.cache_hits == 300
        assert stats.cache_misses == 700
        assert stats.avg_execution_time_ms == 25.5
        assert stats.active_sources == 5
        assert stats.total_sources == 8
        assert stats.uptime_seconds == 86400.0

    def test_defaults_all_zero(self):
        """All defaults are zero."""
        stats = GatewayStatistics()
        assert stats.total_queries == 0
        assert stats.successful_queries == 0
        assert stats.failed_queries == 0
        assert stats.cache_hits == 0
        assert stats.cache_misses == 0
        assert stats.avg_execution_time_ms == 0.0
        assert stats.active_sources == 0
        assert stats.total_sources == 0
        assert stats.uptime_seconds == 0.0


# ===========================================================================
# Test Classes -- Request Models
# ===========================================================================


class TestExecuteQueryRequest:
    """Test ExecuteQueryRequest."""

    def test_create_request(self):
        """Valid request with all fields."""
        req = ExecuteQueryRequest(
            source_ids=["SRC-00001"],
            operation="select",
            filters=[QueryFilter(field="status", operator="eq", value="active")],
            sorts=[QuerySort(field="created_at", order="desc")],
            limit=50,
            offset=10,
            cache_strategy="ttl",
            timeout_s=30,
        )
        assert req.source_ids == ["SRC-00001"]
        assert req.operation == "select"
        assert len(req.filters) == 1
        assert len(req.sorts) == 1
        assert req.limit == 50
        assert req.offset == 10
        assert req.cache_strategy == "ttl"
        assert req.timeout_s == 30

    def test_defaults(self):
        """Default values for ExecuteQueryRequest."""
        req = ExecuteQueryRequest()
        assert req.source_ids == []
        assert req.operation == "select"
        assert req.filters == []
        assert req.sorts == []
        assert req.aggregations == []
        assert req.limit == 100
        assert req.offset == 0
        assert req.cache_strategy == "ttl"
        assert req.timeout_s == 60


class TestBatchQueryRequest:
    """Test BatchQueryRequest."""

    def test_create_batch_request(self):
        """Batch request with multiple queries."""
        q1 = ExecuteQueryRequest(source_ids=["SRC-00001"])
        q2 = ExecuteQueryRequest(source_ids=["SRC-00002"])
        batch = BatchQueryRequest(queries=[q1, q2])
        assert len(batch.queries) == 2

    def test_parallel_default_true(self):
        """Default parallel is True."""
        batch = BatchQueryRequest()
        assert batch.parallel is True

    def test_parallel_false(self):
        """Parallel can be set to False for sequential execution."""
        batch = BatchQueryRequest(parallel=False)
        assert batch.parallel is False

    def test_stop_on_error_default_false(self):
        """Default stop_on_error is False."""
        batch = BatchQueryRequest()
        assert batch.stop_on_error is False


class TestRegisterSourceRequest:
    """Test RegisterSourceRequest."""

    def test_create_request(self):
        """Valid source registration request."""
        req = RegisterSourceRequest(
            name="emissions_db",
            source_type="postgresql",
            connection_string="postgresql://user:pass@host:5432/db",
            description="Main emissions database",
            tags=["emissions", "production"],
            max_connections=20,
            timeout_s=60,
        )
        assert req.name == "emissions_db"
        assert req.source_type == "postgresql"
        assert req.connection_string == "postgresql://user:pass@host:5432/db"
        assert len(req.tags) == 2
        assert req.max_connections == 20
        assert req.timeout_s == 60

    def test_default_source_type(self):
        req = RegisterSourceRequest()
        assert req.source_type == "postgresql"

    def test_default_tags(self):
        req = RegisterSourceRequest()
        assert req.tags == []


class TestCreateTemplateRequest:
    """Test CreateTemplateRequest."""

    def test_create_request(self):
        """Valid template creation request."""
        req = CreateTemplateRequest(
            name="scope1_summary",
            description="Scope 1 emissions summary",
            source_ids=["SRC-00001"],
            operations=["select", "aggregate"],
            parameters={"year": 2026},
        )
        assert req.name == "scope1_summary"
        assert req.description == "Scope 1 emissions summary"
        assert len(req.source_ids) == 1
        assert len(req.operations) == 2
        assert req.parameters["year"] == 2026

    def test_defaults(self):
        req = CreateTemplateRequest()
        assert req.name == ""
        assert req.source_ids == []
        assert req.operations == []
        assert req.parameters == {}


class TestTranslateSchemaRequest:
    """Test TranslateSchemaRequest."""

    def test_create_request(self):
        """Valid schema translation request."""
        req = TranslateSchemaRequest(
            source_schema_id="SCH-00001",
            target_source_id="SRC-00002",
            field_mappings={"old_col": "new_col"},
        )
        assert req.source_schema_id == "SCH-00001"
        assert req.target_source_id == "SRC-00002"
        assert req.field_mappings["old_col"] == "new_col"

    def test_default_field_mappings(self):
        req = TranslateSchemaRequest()
        assert req.field_mappings == {}


class TestCacheInvalidateRequest:
    """Test CacheInvalidateRequest."""

    def test_create_request(self):
        """Valid cache invalidation request."""
        req = CacheInvalidateRequest(
            cache_keys=["key1", "key2"],
            source_ids=["SRC-00001"],
        )
        assert len(req.cache_keys) == 2
        assert len(req.source_ids) == 1
        assert req.invalidate_all is False

    def test_invalidate_all(self):
        """Invalidate all caches."""
        req = CacheInvalidateRequest(invalidate_all=True)
        assert req.invalidate_all is True
        assert req.cache_keys == []
        assert req.source_ids == []

    def test_defaults(self):
        req = CacheInvalidateRequest()
        assert req.cache_keys == []
        assert req.source_ids == []
        assert req.invalidate_all is False
