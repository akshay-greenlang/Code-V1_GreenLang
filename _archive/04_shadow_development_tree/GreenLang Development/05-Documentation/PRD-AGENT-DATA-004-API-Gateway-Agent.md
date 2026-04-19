# PRD: AGENT-DATA-004 - API Gateway Agent

## 1. Overview

| Field | Value |
|-------|-------|
| **PRD ID** | AGENT-DATA-004 |
| **Agent ID** | GL-DATA-GW-001 |
| **Component** | API Gateway Agent (Unified Data Query Interface, Query Routing, Schema Translation, Response Aggregation, Caching, Source Health Monitoring) |
| **Category** | Data Intake Agent |
| **Priority** | P0 - Critical (unified data access layer for all downstream agents and external consumers) |
| **Status** | New Build Required |
| **Author** | GreenLang Platform Team |
| **Date** | February 2026 |

## 2. Problem Statement

GreenLang Climate OS operates 13+ data intake agents (PDF Extractor, Excel/CSV Normalizer,
ERP Connector, EUDR Traceability, SCADA/BMS, Fleet Telematics, etc.) each with independent
REST APIs. Without a unified data gateway agent:

- **No unified query interface**: Consumers must know each agent's specific API contract
- **No cross-source queries**: Cannot query multiple data sources in a single request
- **No response aggregation**: Cannot combine data from multiple agents into unified results
- **No query routing**: No intelligent routing of queries to the correct backend agent
- **No schema translation**: Each agent has different request/response schemas
- **No centralized caching**: Each agent handles its own caching independently
- **No source health monitoring**: No unified view of all data source availability
- **No query lineage**: No cross-agent provenance tracking for data queries
- **No rate limiting coordination**: No coordinated rate limiting across data agents
- **No query templating**: No reusable query templates for common data patterns
- **No batch multi-source queries**: Cannot efficiently query 5+ sources in parallel
- **No data catalog**: No unified catalog of available data sources and their capabilities

## 3. Existing Implementation

### 3.1 Layer 1: Foundation Code
**File**: `greenlang/agents/data/__init__.py`
- References GL-DATA-X-008 as "API Gateway Agent" in agent catalog
- No actual implementation exists for the API Gateway Agent

### 3.2 Infrastructure Foundation (INFRA-006)
**Kong API Gateway** (Infrastructure layer, ALREADY BUILT):
- Kong OSS 3.6 DB-less mode with KIC 3.x
- JWT/OAuth2 authentication via Keycloak
- Redis-backed rate limiting (3 tiers: Free/Standard/Enterprise)
- Request/response transformation
- Circuit breaking, health checks
- Prometheus metrics exposure
- `gl-tenant-isolation` custom Kong plugin

**Important Distinction**: Kong (INFRA-006) handles **network-level** gateway concerns
(auth, rate limiting, TLS, traffic routing). The GL-DATA-GW-001 agent handles
**application-level** data gateway concerns (query parsing, schema translation,
response aggregation, caching, provenance tracking).

### 3.3 Existing Data Agents (Backend Sources)
| Agent ID | Name | Status |
|----------|------|--------|
| GL-DATA-X-001 | PDF & Invoice Extractor | BUILT (AGENT-DATA-001) |
| GL-DATA-X-002 | Excel/CSV Normalizer | BUILT (AGENT-DATA-002) |
| GL-DATA-X-003 | ERP/Finance Connector | BUILT (AGENT-DATA-003) |
| GL-DATA-EUDR-001 | EUDR Traceability Connector | BUILT (AGENT-DATA-005) |

### 3.4 Layer 1 Tests
None found.

## 4. Identified Gaps

### Gap 1: No Data Gateway SDK Package
No `greenlang/data_gateway/` package providing a unified query interface.

### Gap 2: No Query Parser Engine
No engine to parse REST/GraphQL queries into an internal query representation.

### Gap 3: No Query Router
No intelligent routing of parsed queries to the correct backend data agents.

### Gap 4: No Connection Manager
No centralized management of connections to all data agents with health checking.

### Gap 5: No Response Aggregator
No engine to aggregate, merge, and normalize responses from multiple sources.

### Gap 6: No Schema Translator
No translation layer between unified GreenLang schema and agent-specific schemas.

### Gap 7: No Cache Manager
No Redis-backed intelligent caching with TTL, invalidation, and cache warming.

### Gap 8: No Prometheus Metrics
No 12-metric pattern for gateway-specific monitoring.

### Gap 9: No Service Setup Facade
No `configure_data_gateway(app)` / `get_data_gateway(app)` pattern.

### Gap 10: No REST API Router
No `greenlang/data_gateway/api/router.py` with 20+ FastAPI endpoints.

### Gap 11: No K8s Deployment Manifests
No `deployment/kubernetes/data-gateway-service/` manifests.

### Gap 12: No Database Migration
No migration for query registry, data catalog, and audit logging.

### Gap 13: No Monitoring
No Grafana dashboard or alert rules for gateway operations.

### Gap 14: No CI/CD Pipeline
No `.github/workflows/data-gateway-ci.yml`.

## 5. Architecture (Final State)

### 5.1 SDK Package Structure

```
greenlang/data_gateway/
├── __init__.py              # Public API, agent metadata (GL-DATA-GW-001)
├── config.py                # DataGatewayConfig with GL_DATA_GATEWAY_ env prefix
├── models.py                # Pydantic v2 models for all data structures
├── query_parser.py          # QueryParserEngine - REST/GraphQL query parsing
├── query_router.py          # QueryRouterEngine - intelligent query routing
├── connection_manager.py    # ConnectionManagerEngine - agent connections & health
├── response_aggregator.py   # ResponseAggregatorEngine - multi-source response merge
├── schema_translator.py     # SchemaTranslatorEngine - unified schema translation
├── cache_manager.py         # CacheManagerEngine - Redis-backed intelligent caching
├── provenance.py            # ProvenanceTracker - SHA-256 chain-hashed audit trails
├── metrics.py               # 12 Prometheus metrics
├── setup.py                 # DataGatewayService facade
└── api/
    ├── __init__.py
    └── router.py            # FastAPI HTTP service with 20+ endpoints
```

### 5.2 Seven Core Engines

#### Engine 1: QueryParserEngine
- Parse REST JSON queries into internal QueryPlan representation
- Support filter, sort, pagination, field selection, aggregation operators
- Validate query syntax and semantics against registered schemas
- Support query templating with parameter substitution
- Parse GraphQL-style field selection for efficient data retrieval
- Query complexity scoring to prevent expensive queries

#### Engine 2: QueryRouterEngine
- Route parsed queries to the correct backend data agent(s)
- Support single-source and multi-source query plans
- Intelligent source selection based on data type and availability
- Parallel execution of independent sub-queries
- Sequential execution for dependent sub-queries
- Circuit breaker per backend source (5 failures → open for 60s)

#### Engine 3: ConnectionManagerEngine
- Manage HTTP/gRPC connections to all data agents
- Connection pooling with configurable pool size per agent
- Health checking (periodic ping + response time tracking)
- Automatic retry with exponential backoff (3 retries, 1s/2s/4s)
- Connection warm-up on startup
- Source capability registry (what queries each source supports)
- Source status: HEALTHY, DEGRADED, UNHEALTHY, UNKNOWN

#### Engine 4: ResponseAggregatorEngine
- Aggregate responses from multiple backend agents into unified result
- Field-level merge with conflict resolution (latest-wins, source-priority)
- Response normalization to common GreenLang schema
- Pagination across aggregated results
- Response size limiting and truncation
- Error handling: partial results with degraded flag

#### Engine 5: SchemaTranslatorEngine
- Translate between unified GreenLang query schema and agent-specific schemas
- Maintain schema registry with version tracking
- Support schema migration and backwards compatibility
- Field mapping tables per backend agent
- Type coercion and unit normalization
- Validate input/output against registered schemas

#### Engine 6: CacheManagerEngine
- Redis-backed query result caching
- Content-addressable caching (SHA-256 of normalized query)
- Configurable TTL per data source (default 300s, configurable 0-3600s)
- Cache invalidation: manual, TTL-based, event-driven
- Cache warming for frequently used queries
- Cache hit/miss ratio tracking
- Memory-bounded with LRU eviction

#### Engine 7: DataCatalogEngine
- Registry of all available data sources and their capabilities
- Source metadata: agent ID, name, version, endpoints, supported operations
- Data domain classification (emissions, supply chain, financial, geospatial)
- Schema discovery: what fields each source provides
- Dependency mapping between sources
- Source search by capability or data domain

### 5.3 Database Schema

**Schema**: `data_gateway_service`

| Table | Purpose | Type |
|-------|---------|------|
| `data_sources` | Registry of backend data agents | Regular |
| `source_schemas` | Schema definitions per source | Regular |
| `query_templates` | Reusable query templates | Regular |
| `query_log` | Audit log of all queries | Hypertable |
| `cache_entries` | Cache metadata and statistics | Regular |
| `source_health_checks` | Health check history | Hypertable |
| `schema_mappings` | Field mapping between schemas | Regular |
| `aggregation_rules` | Rules for multi-source aggregation | Regular |
| `query_metrics` | Per-query performance metrics | Hypertable |
| `data_catalog` | Unified data catalog entries | Regular |

### 5.4 Prometheus Metrics (12)

| # | Metric | Type | Labels |
|---|--------|------|--------|
| 1 | `gl_data_gateway_queries_total` | Counter | `source`, `operation`, `status` |
| 2 | `gl_data_gateway_query_duration_seconds` | Histogram | `source`, `operation` |
| 3 | `gl_data_gateway_cache_hits_total` | Counter | `source` |
| 4 | `gl_data_gateway_cache_misses_total` | Counter | `source` |
| 5 | `gl_data_gateway_routing_decisions_total` | Counter | `source`, `strategy` |
| 6 | `gl_data_gateway_aggregation_operations_total` | Counter | `sources_count`, `status` |
| 7 | `gl_data_gateway_schema_translations_total` | Counter | `source`, `direction` |
| 8 | `gl_data_gateway_active_queries` | Gauge | - |
| 9 | `gl_data_gateway_source_health` | Gauge | `source`, `status` |
| 10 | `gl_data_gateway_connection_pool_size` | Gauge | `source` |
| 11 | `gl_data_gateway_processing_errors_total` | Counter | `source`, `error_type` |
| 12 | `gl_data_gateway_response_size_bytes` | Histogram | `source` |

### 5.5 REST API Endpoints (20)

| # | Method | Path | Description |
|---|--------|------|-------------|
| 1 | POST | `/v1/gateway/query` | Execute unified data query |
| 2 | POST | `/v1/gateway/query/batch` | Execute batch multi-source query |
| 3 | GET | `/v1/gateway/query/{query_id}` | Get query result by ID |
| 4 | GET | `/v1/gateway/query/{query_id}/lineage` | Get query provenance chain |
| 5 | GET | `/v1/gateway/sources` | List all registered data sources |
| 6 | GET | `/v1/gateway/sources/{source_id}` | Get source details |
| 7 | POST | `/v1/gateway/sources/{source_id}/test` | Test source connectivity |
| 8 | GET | `/v1/gateway/sources/{source_id}/schema` | Get source schema |
| 9 | GET | `/v1/gateway/catalog` | Browse unified data catalog |
| 10 | GET | `/v1/gateway/catalog/search` | Search data catalog |
| 11 | GET | `/v1/gateway/schemas` | List registered schemas |
| 12 | POST | `/v1/gateway/schemas/translate` | Translate between schemas |
| 13 | GET | `/v1/gateway/templates` | List query templates |
| 14 | POST | `/v1/gateway/templates` | Create query template |
| 15 | POST | `/v1/gateway/templates/{template_id}/execute` | Execute query from template |
| 16 | GET | `/v1/gateway/cache/stats` | Get cache statistics |
| 17 | DELETE | `/v1/gateway/cache` | Invalidate cache entries |
| 18 | GET | `/v1/gateway/health` | Service health check |
| 19 | GET | `/v1/gateway/health/sources` | All source health statuses |
| 20 | GET | `/v1/gateway/statistics` | Service-wide statistics |

### 5.6 Configuration

**Environment Variable Prefix**: `GL_DATA_GATEWAY_`

| Variable | Default | Description |
|----------|---------|-------------|
| `GL_DATA_GATEWAY_DATABASE_URL` | `""` | PostgreSQL connection string |
| `GL_DATA_GATEWAY_REDIS_URL` | `""` | Redis connection string |
| `GL_DATA_GATEWAY_LOG_LEVEL` | `"INFO"` | Logging level |
| `GL_DATA_GATEWAY_CACHE_DEFAULT_TTL` | `300` | Default cache TTL in seconds |
| `GL_DATA_GATEWAY_CACHE_MAX_SIZE_MB` | `256` | Maximum cache size in MB |
| `GL_DATA_GATEWAY_MAX_QUERY_COMPLEXITY` | `100` | Maximum query complexity score |
| `GL_DATA_GATEWAY_MAX_SOURCES_PER_QUERY` | `10` | Maximum sources in a single query |
| `GL_DATA_GATEWAY_QUERY_TIMEOUT_SECONDS` | `30` | Query execution timeout |
| `GL_DATA_GATEWAY_CONNECTION_POOL_MIN` | `2` | Minimum connections per source |
| `GL_DATA_GATEWAY_CONNECTION_POOL_MAX` | `10` | Maximum connections per source |
| `GL_DATA_GATEWAY_HEALTH_CHECK_INTERVAL` | `60` | Health check interval in seconds |
| `GL_DATA_GATEWAY_CIRCUIT_BREAKER_THRESHOLD` | `5` | Failures before circuit opens |
| `GL_DATA_GATEWAY_CIRCUIT_BREAKER_TIMEOUT` | `60` | Circuit breaker timeout in seconds |
| `GL_DATA_GATEWAY_RETRY_MAX_ATTEMPTS` | `3` | Maximum retry attempts |
| `GL_DATA_GATEWAY_RETRY_BASE_DELAY` | `1.0` | Base retry delay in seconds |
| `GL_DATA_GATEWAY_BATCH_MAX_SIZE` | `50` | Maximum queries in a batch |
| `GL_DATA_GATEWAY_BATCH_WORKER_COUNT` | `4` | Parallel batch workers |
| `GL_DATA_GATEWAY_POOL_MIN_SIZE` | `2` | DB pool minimum connections |
| `GL_DATA_GATEWAY_POOL_MAX_SIZE` | `10` | DB pool maximum connections |
| `GL_DATA_GATEWAY_RETENTION_DAYS` | `90` | Query log retention in days |

## 6. Completion Plan

### Phase 1: SDK Core (Config + Models + 7 Engines)
1. Build `config.py` - DataGatewayConfig with GL_DATA_GATEWAY_ env prefix
2. Build `models.py` - All Pydantic v2 models (enums, data models, request models)
3. Build `query_parser.py` - QueryParserEngine
4. Build `query_router.py` - QueryRouterEngine
5. Build `connection_manager.py` - ConnectionManagerEngine
6. Build `response_aggregator.py` - ResponseAggregatorEngine
7. Build `schema_translator.py` - SchemaTranslatorEngine
8. Build `cache_manager.py` - CacheManagerEngine
9. Build `data_catalog.py` - DataCatalogEngine
10. Build `provenance.py` - ProvenanceTracker
11. Build `metrics.py` - 12 Prometheus metrics
12. Build `setup.py` - DataGatewayService facade
13. Build `api/router.py` - 20 FastAPI endpoints
14. Build `__init__.py` - Package exports

### Phase 2: Infrastructure
15. Build V035 database migration
16. Build K8s deployment manifests (8 files)
17. Build CI/CD pipeline
18. Build Grafana dashboard
19. Build alert rules

### Phase 3: Testing
20. Build unit tests (600+ tests across 13 test files)

## 7. Success Criteria

- [ ] All 7 engines implemented with deterministic behavior
- [ ] 20 REST API endpoints operational
- [ ] 12 Prometheus metrics instrumented and exposed
- [ ] SHA-256 provenance chain on all query operations
- [ ] Cache hit rate tracking with configurable TTL
- [ ] Source health monitoring with circuit breaker
- [ ] Schema translation between unified and agent-specific formats
- [ ] Multi-source query aggregation with parallel execution
- [ ] Query complexity scoring and limiting
- [ ] V035 database migration with 10 tables
- [ ] K8s manifests with full security hardening
- [ ] CI/CD pipeline with 8+ stages
- [ ] Grafana dashboard with 20+ panels
- [ ] 15+ PrometheusRule alert rules
- [ ] 600+ tests passing
- [ ] Integration with Kong (INFRA-006) for network-level concerns

## 8. Integration Points

### Upstream Dependencies
- AGENT-FOUND-002 Schema Compiler (schema validation)
- AGENT-FOUND-003 Unit Normalizer (unit conversion in schema translation)
- AGENT-FOUND-006 Access & Policy Guard (authorization)
- AGENT-FOUND-007 Agent Registry (source discovery)
- AGENT-FOUND-010 Observability Agent (metrics/tracing)

### Downstream Consumers
- All MRV calculation agents (data retrieval)
- All reporting agents (data for compliance reports)
- Admin dashboard (data exploration)
- External API clients via Kong

### Infrastructure Integration
- Kong API Gateway (INFRA-006) - network routing, auth, rate limiting
- PostgreSQL + TimescaleDB (INFRA-002) - persistent storage
- Redis (INFRA-003) - caching, connection pooling
- Prometheus (OBS-001) - metrics collection
- Grafana (OBS-002) - dashboards
- Kubernetes (INFRA-001) - deployment & scaling

## 9. Risk Assessment

| Risk | Impact | Mitigation |
|------|--------|------------|
| High query latency from aggregation | Performance degradation | Parallel sub-query execution, aggressive caching |
| Backend agent unavailability | Partial data loss | Circuit breaker, graceful degradation, partial results |
| Schema drift between agents | Translation errors | Version-pinned schema registry, automated testing |
| Cache staleness | Stale data served | Configurable TTL, event-driven invalidation |
| Query complexity explosion | Resource exhaustion | Complexity scoring, query limits, timeout enforcement |
