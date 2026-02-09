# -*- coding: utf-8 -*-
"""
GL-DATA-GW-001: GreenLang API Gateway Agent Service SDK
========================================================

This package provides a unified data query interface that sits behind
Kong API Gateway and routes queries to backend data agents. It supports:

- 15 data source types (PDF extractor, Excel normalizer, ERP connector,
  EUDR traceability, SCADA/BMS, fleet telematics, utility tariff,
  supplier portal, event processor, document classifier, OCR agent,
  email processor, IoT meter, emission factor, custom)
- Query parsing with complexity scoring and validation
- Intelligent query routing to backend data agents
- Connection management with circuit breaker and retry
- Response aggregation with conflict resolution strategies
- Schema translation between heterogeneous data sources
- Multi-layer caching with TTL, event-driven invalidation, and warming
- Data catalog for source discovery and self-service
- Batch query processing with parallel workers
- SHA-256 provenance chain tracking for complete audit trails
- 12 Prometheus metrics for observability
- FastAPI REST API with 20 endpoints
- Thread-safe configuration with GL_DATA_GATEWAY_ env prefix

Key Components:
    - config: DataGatewayConfig with GL_DATA_GATEWAY_ env prefix
    - models: Pydantic v2 models for all data structures
    - query_parser: Query parsing, validation, and complexity scoring engine
    - query_router: Intelligent query routing to backend sources engine
    - connection_manager: Connection pool, circuit breaker, retry engine
    - response_aggregator: Multi-source response merging engine
    - schema_translator: Cross-source schema translation engine
    - cache_manager: Multi-layer cache with TTL and warming engine
    - data_catalog: Source discovery and metadata catalog engine
    - provenance: SHA-256 chain-hashed audit trails
    - metrics: 12 Prometheus metrics
    - api: FastAPI HTTP service
    - setup: DataGatewayService facade

Example:
    >>> from greenlang.data_gateway import DataGatewayService
    >>> service = DataGatewayService()
    >>> result = service.execute_query(request)
    >>> print(result.status)
    completed

Agent ID: GL-DATA-GW-001
Agent Name: API Gateway Agent
"""

__version__ = "1.0.0"
__agent_id__ = "GL-DATA-GW-001"
__agent_name__ = "API Gateway Agent"

# SDK availability flag
DATA_GATEWAY_SDK_AVAILABLE = True

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
from greenlang.data_gateway.config import (
    DataGatewayConfig,
    get_config,
    set_config,
    reset_config,
)

# ---------------------------------------------------------------------------
# Models (enums, data models, request models)
# ---------------------------------------------------------------------------
from greenlang.data_gateway.models import (
    # Enumerations
    DataSourceType,
    QueryOperation,
    AggregationType,
    SortOrder,
    SourceStatus,
    CacheStrategy,
    QueryStatus,
    ConflictResolution,
    # Core data models
    DataSource,
    SchemaField,
    SchemaDefinition,
    QueryFilter,
    QuerySort,
    QueryAggregation,
    QueryPlan,
    QueryResult,
    CacheEntry,
    SourceHealthCheck,
    SchemaMapping,
    QueryTemplate,
    DataCatalogEntry,
    GatewayStatistics,
    # Request models
    ExecuteQueryRequest,
    BatchQueryRequest,
    RegisterSourceRequest,
    CreateTemplateRequest,
    TranslateSchemaRequest,
    CacheInvalidateRequest,
)

# ---------------------------------------------------------------------------
# Core engines
# ---------------------------------------------------------------------------
from greenlang.data_gateway.query_parser import QueryParserEngine
from greenlang.data_gateway.query_router import QueryRouterEngine
from greenlang.data_gateway.connection_manager import ConnectionManagerEngine
from greenlang.data_gateway.response_aggregator import ResponseAggregatorEngine
from greenlang.data_gateway.schema_translator import SchemaTranslatorEngine
from greenlang.data_gateway.cache_manager import CacheManagerEngine
from greenlang.data_gateway.data_catalog import DataCatalogEngine
from greenlang.data_gateway.provenance import ProvenanceTracker

# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
from greenlang.data_gateway.metrics import (
    PROMETHEUS_AVAILABLE,
    # Metric objects
    gateway_queries_total,
    gateway_query_duration_seconds,
    gateway_cache_hits_total,
    gateway_cache_misses_total,
    gateway_source_requests_total,
    gateway_source_errors_total,
    gateway_batch_queries_total,
    gateway_schema_translations_total,
    gateway_circuit_breaker_trips_total,
    gateway_retry_attempts_total,
    gateway_active_sources,
    gateway_cache_size_bytes,
    # Helper functions
    record_query,
    record_cache_hit,
    record_cache_miss,
    record_source_request,
    record_source_error,
    record_batch_query,
    record_schema_translation,
    record_circuit_breaker_trip,
    record_retry_attempt,
    update_active_sources,
    update_cache_size,
)

# ---------------------------------------------------------------------------
# Service setup facade
# ---------------------------------------------------------------------------
from greenlang.data_gateway.setup import (
    DataGatewayService,
    configure_data_gateway,
    get_data_gateway,
    get_router,
)

__all__ = [
    # Version
    "__version__",
    "__agent_id__",
    "__agent_name__",
    # SDK flag
    "DATA_GATEWAY_SDK_AVAILABLE",
    # Configuration
    "DataGatewayConfig",
    "get_config",
    "set_config",
    "reset_config",
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
    # Core engines
    "QueryParserEngine",
    "QueryRouterEngine",
    "ConnectionManagerEngine",
    "ResponseAggregatorEngine",
    "SchemaTranslatorEngine",
    "CacheManagerEngine",
    "DataCatalogEngine",
    "ProvenanceTracker",
    # Metric objects
    "PROMETHEUS_AVAILABLE",
    "gateway_queries_total",
    "gateway_query_duration_seconds",
    "gateway_cache_hits_total",
    "gateway_cache_misses_total",
    "gateway_source_requests_total",
    "gateway_source_errors_total",
    "gateway_batch_queries_total",
    "gateway_schema_translations_total",
    "gateway_circuit_breaker_trips_total",
    "gateway_retry_attempts_total",
    "gateway_active_sources",
    "gateway_cache_size_bytes",
    # Metric helper functions
    "record_query",
    "record_cache_hit",
    "record_cache_miss",
    "record_source_request",
    "record_source_error",
    "record_batch_query",
    "record_schema_translation",
    "record_circuit_breaker_trip",
    "record_retry_attempt",
    "update_active_sources",
    "update_cache_size",
    # Service setup facade
    "DataGatewayService",
    "configure_data_gateway",
    "get_data_gateway",
    "get_router",
]
