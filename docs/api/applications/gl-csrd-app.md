# GL-CSRD-APP -- CSRD/ESRS Digital Reporting Platform API Reference

**Source:** `applications/GL-CSRD-APP/CSRD-Reporting-Platform/api/server.py`
**Title:** CSRD/ESRS Digital Reporting Platform
**Version:** 1.0.0
**Description:** Zero-hallucination EU sustainability reporting API

---

## Overview

The GL-CSRD-APP provides a REST API for CSRD/ESRS (Corporate Sustainability Reporting Directive / European Sustainability Reporting Standards) compliance. It supports a 6-agent pipeline covering data intake, metric calculation (975 ESRS metrics), double materiality assessment, compliance auditing, XBRL/iXBRL report generation, and multi-framework aggregation.

**Authentication:** Rate limiting via `slowapi`. CORS restricted to configured origins.
**Docs:** `/docs` (Swagger UI), `/redoc` (ReDoc), `/openapi.json`

---

## Endpoints

### System

| Method | Path | Summary | Auth |
|--------|------|---------|------|
| GET | `/` | Root -- API information | None |
| GET | `/health` | Basic health check | None |
| GET | `/ready` | Readiness check (DB, Redis, Weaviate) | None |
| GET | `/metrics` | Prometheus-compatible metrics | None |

### Pipeline Execution

| Method | Path | Summary | Auth |
|--------|------|---------|------|
| POST | `/api/v1/pipeline/run` | Execute full CSRD/ESRS pipeline | Rate limited (10/min) |
| GET | `/api/v1/pipeline/status/{job_id}` | Get pipeline job status | None |
| GET | `/api/v1/pipeline/jobs` | List all pipeline jobs | None |

### Validation

| Method | Path | Summary | Auth |
|--------|------|---------|------|
| POST | `/api/v1/validate` | Validate input data against ESRS schema | Rate limited (60/min) |

### Calculation

| Method | Path | Summary | Auth |
|--------|------|---------|------|
| POST | `/api/v1/calculate/{metric_id}` | Calculate a specific ESRS metric | None |

### Reporting

| Method | Path | Summary | Auth |
|--------|------|---------|------|
| POST | `/api/v1/report/generate` | Generate CSRD report (XBRL, JSON, PDF, Excel) | None |

### Materiality

| Method | Path | Summary | Auth |
|--------|------|---------|------|
| POST | `/api/v1/materiality/assess` | Double materiality assessment | None |

---

## Endpoint Details

### POST /api/v1/pipeline/run

Execute the complete CSRD/ESRS 6-agent reporting pipeline asynchronously.

**Rate Limit:** 10 requests/minute

**Request Body:**

```json
{
  "company_name": "GreenLang GmbH",
  "lei_code": "529900T8BM49AURSDO55",
  "reporting_year": 2025,
  "input_file": "/data/esg_data_2025.csv",
  "output_format": "xbrl",
  "enable_ai": true
}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `company_name` | string | Yes | Legal entity name |
| `lei_code` | string | No | Legal Entity Identifier |
| `reporting_year` | integer | Yes | Reporting year |
| `input_file` | string | Yes | Path to input data file |
| `output_format` | string | No | Output format: `xbrl` (default), `json`, `pdf` |
| `enable_ai` | boolean | No | Enable AI-powered features (default: true) |

**Response (200):**

```json
{
  "job_id": "f47ac10b-58cc-4372-a567-0e02b2c3d479",
  "status": "queued",
  "message": "Pipeline execution queued successfully",
  "started_at": "2026-04-04T12:00:00Z"
}
```

**Error (400):** Input file not found.

### GET /api/v1/pipeline/status/{job_id}

**Path Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `job_id` | string | UUID returned by pipeline/run |

**Response (200):**

```json
{
  "job_id": "f47ac10b-58cc-4372-a567-0e02b2c3d479",
  "status": "completed",
  "request": { "company_name": "GreenLang GmbH", "reporting_year": 2025 },
  "started_at": "2026-04-04T12:00:00Z",
  "completed_at": "2026-04-04T12:05:00Z",
  "progress": 100,
  "result": {
    "pipeline_id": "CSRD-2025-001",
    "status": "success",
    "compliance_status": "compliant",
    "data_quality_score": 94.5,
    "warnings_count": 3,
    "errors_count": 0,
    "output_dir": "/output/api_jobs/f47ac10b..."
  }
}
```

**Error (404):** Job ID not found.

### GET /api/v1/pipeline/jobs

**Query Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `limit` | integer | 100 | Maximum jobs to return |
| `status_filter` | string | None | Filter by status: `queued`, `running`, `completed`, `failed` |

**Response (200):**

```json
{
  "total": 25,
  "filtered": 10,
  "jobs": [...]
}
```

### POST /api/v1/validate

Validate input data against CSRD/ESRS schema without running the full pipeline.

**Rate Limit:** 60 requests/minute

**Request Body:**

```json
{
  "input_file": "/data/esg_data_2025.csv",
  "schema_version": "1.0"
}
```

**Response (200):**

```json
{
  "is_valid": true,
  "errors": [],
  "warnings": ["Missing optional field: biodiversity_impact"],
  "data_quality_score": 92.0
}
```

### POST /api/v1/calculate/{metric_id}

Calculate a single ESRS metric.

**Path Parameters:** `metric_id` (string) -- ESRS metric identifier

**Request Body:** JSON dictionary with metric-specific input data.

**Response (200):**

```json
{
  "metric_id": "E1-6",
  "value": 1250.5,
  "unit": "tCO2e",
  "calculation_method": "GHG Protocol",
  "data_quality": "high"
}
```

### POST /api/v1/report/generate

**Query Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `company_name` | string | Required | Company name |
| `reporting_year` | integer | Required | Year |
| `format` | string | `xbrl` | Output format: `xbrl`, `json`, `pdf`, `excel` |

**Response (200):**

```json
{
  "report_id": "REP-2025-A1B2C3D4",
  "status": "generated",
  "format": "xbrl",
  "download_url": "/api/v1/report/download/xbrl/REP-2025-A1B2C3D4"
}
```

### POST /api/v1/materiality/assess

**Query Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `company_name` | string | Required | Company name |
| `sector` | string | Required | Industry sector |
| `enable_ai` | boolean | true | Use RAG for stakeholder analysis |

**Response (200):**

```json
{
  "assessment_id": "MAT-2026-A1B2C3D4",
  "material_topics": [
    "Climate change mitigation",
    "Energy efficiency",
    "Water management"
  ],
  "methodology": "ESRS 2 IRO-1",
  "ai_enabled": true
}
```

---

## Request/Response Models

| Model | Description |
|-------|-------------|
| `HealthResponse` | `status`, `version`, `timestamp`, `uptime_seconds` |
| `ReadinessResponse` | `status`, `database`, `redis`, `weaviate` |
| `PipelineRequest` | `company_name`, `lei_code`, `reporting_year`, `input_file`, `output_format`, `enable_ai` |
| `PipelineResponse` | `job_id`, `status`, `message`, `started_at` |
| `ValidationRequest` | `input_file`, `schema_version` |
| `ValidationResponse` | `is_valid`, `errors`, `warnings`, `data_quality_score` |

---

## Source File

`applications/GL-CSRD-APP/CSRD-Reporting-Platform/api/server.py`
