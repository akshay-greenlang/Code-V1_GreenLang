# Data Gateway (API Gateway Agent) API Reference

**Agent:** AGENT-DATA-004 (GL-DATA-GW-001)
**Prefix:** `/v1/gateway`
**Source:** `greenlang/agents/data/data_gateway/api/router.py`
**Status:** Production Ready

## Overview

The Data Gateway agent provides a unified query layer across all registered data sources. It supports multi-source queries, schema translation, result caching, a data catalog, query templates, and provenance tracking for every query execution.

---

## Endpoint Summary

| # | Method | Path | Summary | Auth |
|---|--------|------|---------|------|
| 1 | POST | `/query` | Execute unified data query | Yes |
| 2 | POST | `/query/batch` | Execute batch multi-source query | Yes |
| 3 | GET | `/query/{query_id}` | Get query result by ID | Yes |
| 4 | GET | `/query/{query_id}/lineage` | Get query provenance chain | Yes |
| 5 | GET | `/sources` | List registered data sources | Yes |
| 6 | GET | `/sources/{source_id}` | Get source details | Yes |
| 7 | POST | `/sources/{source_id}/test` | Test source connectivity | Yes |
| 8 | GET | `/sources/{source_id}/schema` | Get source schema | Yes |
| 9 | GET | `/catalog` | Browse unified data catalog | Yes |
| 10 | GET | `/catalog/search` | Search data catalog | Yes |
| 11 | GET | `/schemas` | List registered schemas | Yes |
| 12 | POST | `/schemas/translate` | Translate between schemas | Yes |
| 13 | GET | `/templates` | List query templates | Yes |
| 14 | POST | `/templates` | Create query template | Yes |
| 15 | POST | `/templates/{template_id}/execute` | Execute from template | Yes |
| 16 | GET | `/cache/stats` | Cache statistics | Yes |
| 17 | DELETE | `/cache` | Invalidate cache | Yes |
| 18 | GET | `/health` | Health check | No |
| 19 | GET | `/health/sources` | Source health statuses | Yes |
| 20 | GET | `/statistics` | Service statistics | Yes |

---

## Key Endpoints

### 1. Execute Unified Data Query

```http
POST /v1/gateway/query
Content-Type: application/json
Authorization: Bearer {token}
```

**Request Body:**

```json
{
  "sources": ["postgresql_emissions", "redis_cache"],
  "filters": {
    "facility_id": "FAC-001",
    "reporting_year": 2025
  },
  "sort": [{"field": "total_co2e", "order": "desc"}],
  "aggregations": {"total_co2e": "sum"},
  "limit": 100,
  "offset": 0
}
```

**Response:**

```json
{
  "query_id": "qry_abc123",
  "source_id": "postgresql_emissions",
  "data": [
    {"facility_id": "FAC-001", "total_co2e": 1500.25, "year": 2025}
  ],
  "total_count": 1,
  "row_count": 1,
  "errors": [],
  "execution_time_ms": 45.2
}
```

### 4. Get Query Provenance Chain

```http
GET /v1/gateway/query/{query_id}/lineage
```

**Response:**

```json
{
  "query_id": "qry_abc123",
  "is_valid": true,
  "chain_length": 3,
  "chain": [
    {"step": "query_parse", "hash": "sha256:..."},
    {"step": "source_execution", "hash": "sha256:..."},
    {"step": "result_merge", "hash": "sha256:..."}
  ]
}
```

### 12. Translate Between Schemas

```http
POST /v1/gateway/schemas/translate
```

**Request Body:**

```json
{
  "data": {"emissions_mt": 1500, "year": 2025},
  "source_type": "epa_ghgrp",
  "target_type": "greenlang_canonical"
}
```

### 17. Invalidate Cache

```http
DELETE /v1/gateway/cache?source_id=postgresql_emissions&invalidate_all=false
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `source_id` | string | Invalidate entries for this source |
| `query_hash` | string | Invalidate specific query hash |
| `invalidate_all` | boolean | Invalidate all entries (default: false) |
