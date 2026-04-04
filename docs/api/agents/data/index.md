# GreenLang Data Agents API Reference

## Overview

The GreenLang Data Agent layer provides 20 specialized agents for data intake, normalization, quality management, and geospatial connectivity. Each agent exposes a FastAPI REST API with 20 endpoints, mounted under a unique prefix.

**Base URL:** `https://api.greenlang.io/api/v1/{agent-prefix}`

**Authentication:** JWT Bearer token (OAuth2 with RS256)

**Rate Limits:**
- 100 requests per minute (authenticated)
- 10 requests per minute (unauthenticated)

---

## Agent Registry

### Intake Agents (001-007)

| # | Agent | API Prefix | Endpoints | Description |
|---|-------|------------|-----------|-------------|
| 001 | [PDF & Invoice Extractor](./pdf_extractor.md) | `/api/v1/pdf-extractor` | 20 | Document ingestion, OCR, field extraction, invoice/manifest/utility bill processing |
| 002 | [Excel & CSV Normalizer](./excel_normalizer.md) | `/api/v1/excel-normalizer` | 20 | File upload, column mapping, data type detection, schema validation, normalization |
| 003 | [ERP/Finance Connector](./erp_connector.md) | `/api/v1/erp-connector` | 20 | SAP/Oracle/NetSuite/Dynamics connections, spend sync, PO sync, Scope 3 calculation |
| 004 | [API Gateway (Data Gateway)](./data_gateway.md) | `/v1/gateway` | 20 | Unified query execution, source management, schema translation, caching, catalog |
| 005 | [EUDR Traceability Connector](./eudr_traceability.md) | `/v1/eudr` | 20 | Plot management, chain of custody, DDS generation, risk assessment, commodity classification |
| 006 | [GIS/Mapping Connector](./gis_mapping.md) | `/v1/gis` | 20 | Format parsing, CRS transform, spatial analysis, land cover, geocoding, layer management |
| 007 | [Deforestation Satellite Connector](./satellite_connector.md) | `/v1/deforestation` | 20 | Satellite imagery, vegetation indices, change detection, EUDR baseline, monitoring |

### Quality Agents (008-019)

| # | Agent | API Prefix | Endpoints | Description |
|---|-------|------------|-----------|-------------|
| 008 | [Supplier Questionnaire Processor](./questionnaire_processor.md) | `/api/v1/questionnaires` | 20 | Template management, distribution, response collection, validation, scoring, analytics |
| 009 | [Spend Data Categorizer](./spend_categorizer.md) | `/api/v1/spend-categorizer` | 20 | Taxonomy classification, Scope 3 mapping, emission calculation from spend data |
| 010 | [Data Quality Profiler](./data_quality_profiler.md) | `/api/v1/data-quality` | 20 | Dataset profiling, quality assessment, anomaly detection, rule management |
| 011 | [Duplicate Detection](./duplicate_detection.md) | `/api/v1/dedup` | 20 | Record deduplication: fingerprinting, blocking, comparison, clustering, merging |
| 012 | [Missing Value Imputer](./missing_value_imputer.md) | `/api/v1/imputer` | 20 | Missingness analysis, multi-strategy imputation, validation, rules, templates, pipeline |
| 013 | [Outlier Detection](./outlier_detection.md) | `/api/v1/outlier` | 20 | Multi-method detection, classification, treatment, thresholds, feedback, impact analysis |
| 014 | [Time Series Gap Filler](./time_series_gap_filler.md) | `/api/v1/gap-filler` | 20 | Gap detection, frequency analysis, fill strategies, validation, correlations, calendars |
| 015 | [Cross-Source Reconciliation](./cross_source_reconciliation.md) | `/api/v1/reconciliation` | 20 | Source registration, record matching, discrepancy detection, resolution, golden records |
| 016 | [Data Freshness Monitor](./data_freshness_monitor.md) | `/api/v1/freshness` | 20 | Dataset registration, SLA definitions, freshness checks, breach management, predictions |
| 017 | [Schema Migration](./schema_migration.md) | `/api/v1/schema-migration` | 20 | Schema registration, versioning, change detection, compatibility, migration, rollback |
| 018 | [Data Lineage Tracker](./data_lineage_tracker.md) | `/api/v1/data-lineage` | 20 | Asset registration, transformation tracking, lineage graph, impact analysis |
| 019 | [Validation Rule Engine](./validation_rule_engine.md) | `/api/v1/validation-rules` | 20 | Rule management, rule sets, evaluation, conflict detection, compliance packs |

### Geospatial Agents (020)

| # | Agent | API Prefix | Endpoints | Description |
|---|-------|------------|-----------|-------------|
| 020 | [Climate Hazard Connector](./climate_hazard_connector.md) | `/api/v1/climate-hazard` | 20 | Climate risk indexing, scenario projection, exposure assessment, vulnerability scoring |

---

## Common Patterns

### Authentication

All endpoints (except `/health`) require a valid JWT Bearer token:

```http
Authorization: Bearer eyJhbGciOiJSUzI1NiIs...
```

### Pagination

List endpoints support offset-based pagination:

```http
GET /api/v1/{agent}/items?limit=50&offset=0
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `limit` | integer | 50 | Items per page (max: 200-1000 depending on agent) |
| `offset` | integer | 0 | Number of items to skip |

### Error Responses

All agents return consistent error responses:

| Status Code | Description |
|-------------|-------------|
| 400 | Bad Request - Invalid input data or missing required fields |
| 404 | Not Found - Resource does not exist |
| 429 | Too Many Requests - Rate limit exceeded |
| 503 | Service Unavailable - Agent service not initialized |
| 500 | Internal Server Error |

### Health Check

Every agent exposes a health endpoint at `/health` (no authentication required):

```json
{
  "status": "healthy",
  "service": "{agent-name}",
  "version": "1.0.0"
}
```

### Provenance

Most data records carry a `provenance_hash` (SHA-256) for audit trail integrity verification. This hash covers all input fields and calculation parameters, ensuring deterministic reproducibility.

---

## Quick Start

```python
import requests

# 1. Authenticate
token_response = requests.post(
    "https://api.greenlang.io/api/v1/auth/token",
    data={
        "grant_type": "client_credentials",
        "client_id": "your_client_id",
        "client_secret": "your_client_secret"
    }
)
access_token = token_response.json()["access_token"]
headers = {"Authorization": f"Bearer {access_token}"}

# 2. Upload a CSV for normalization
upload_response = requests.post(
    "https://api.greenlang.io/api/v1/excel-normalizer/v1/files/upload",
    headers=headers,
    json={
        "file_name": "emissions_data.csv",
        "file_content_base64": "<base64-encoded-content>",
        "file_format": "csv"
    }
)
file_id = upload_response.json()["file_id"]

# 3. Check data quality
quality_response = requests.post(
    "https://api.greenlang.io/api/v1/data-quality/v1/assess",
    headers=headers,
    json={
        "data": [{"facility": "HQ", "emissions_kg": 1500.0}],
        "dataset_name": "facility_emissions",
        "dimensions": ["completeness", "validity", "accuracy"]
    }
)
print(quality_response.json())
```

---

## Source Files

All router implementations are located at:
`greenlang/agents/data/{agent_name}/api/router.py`

Each agent provides exactly 20 REST API endpoints and follows the GreenLang agent architecture pattern with 7 computation engines per agent.
