# PACK-030: API Documentation

**Pack:** PACK-030 Net Zero Reporting Pack
**Version:** 1.0.0
**Last Updated:** 2026-03-20
**Base URL:** `http://localhost:8030/api/v1`

---

## Table of Contents

1. [Authentication](#authentication)
2. [Health & Metrics](#health--metrics)
3. [Report Generation APIs](#report-generation-apis)
4. [Framework-Specific APIs](#framework-specific-apis)
5. [Workflow APIs](#workflow-apis)
6. [Dashboard APIs](#dashboard-apis)
7. [Data Aggregation APIs](#data-aggregation-apis)
8. [Narrative APIs](#narrative-apis)
9. [Validation APIs](#validation-apis)
10. [Assurance APIs](#assurance-apis)
11. [Configuration APIs](#configuration-apis)
12. [Error Handling](#error-handling)
13. [Rate Limiting](#rate-limiting)
14. [Pagination](#pagination)

---

## 1. Authentication

All API endpoints require authentication via JWT Bearer tokens.

### Obtain Token

```http
POST /auth/token
Content-Type: application/json

{
  "username": "your-username",
  "password": "your-password"
}
```

**Response:**

```json
{
  "access_token": "eyJhbGciOiJSUzI1NiIs...",
  "token_type": "bearer",
  "expires_in": 3600
}
```

### Use Token

```http
Authorization: Bearer eyJhbGciOiJSUzI1NiIs...
```

### Required Permissions

| Permission | Description |
|-----------|-------------|
| `report:read` | View reports |
| `report:write` | Generate and edit reports |
| `report:approve` | Approve reports for publication |
| `report:publish` | Publish reports externally |
| `dashboard:read` | View dashboards |
| `config:read` | View configuration |
| `config:write` | Modify configuration |
| `assurance:read` | View assurance evidence |
| `assurance:export` | Export evidence bundles |
| `admin:manage` | Administrative functions |

---

## 2. Health & Metrics

### Health Check

```http
GET /health
```

**Response (200 OK):**

```json
{
  "status": "healthy",
  "version": "1.0.0",
  "pack_id": "PACK-030-net-zero-reporting",
  "uptime_seconds": 86400,
  "database": {"status": "connected", "latency_ms": 2},
  "redis": {"status": "connected", "latency_ms": 1},
  "packs": {
    "PACK-021": {"status": "available", "latency_ms": 15},
    "PACK-022": {"status": "available", "latency_ms": 12},
    "PACK-028": {"status": "available", "latency_ms": 18},
    "PACK-029": {"status": "available", "latency_ms": 14}
  },
  "applications": {
    "GL-SBTi-APP": {"status": "available", "latency_ms": 25},
    "GL-CDP-APP": {"status": "available", "latency_ms": 22},
    "GL-TCFD-APP": {"status": "available", "latency_ms": 20},
    "GL-GHG-APP": {"status": "available", "latency_ms": 18}
  }
}
```

### Prometheus Metrics

```http
GET /metrics
```

Returns Prometheus-compatible metrics for monitoring.

### OpenAPI Documentation

```http
GET /docs      # Swagger UI
GET /redoc     # ReDoc UI
GET /openapi.json  # OpenAPI JSON schema
```

---

## 3. Report Generation APIs

### Generate Report

```http
POST /api/v1/reports/generate
Content-Type: application/json
Authorization: Bearer {token}

{
  "organization_id": "uuid",
  "framework": "SBTi",
  "reporting_period": {
    "start": "2025-01-01",
    "end": "2025-12-31"
  },
  "output_formats": ["PDF", "JSON"],
  "language": "en",
  "branding": {
    "logo_path": "/path/to/logo.png",
    "primary_color": "#1E3A8A"
  }
}
```

**Response (202 Accepted):**

```json
{
  "report_id": "uuid",
  "status": "generating",
  "framework": "SBTi",
  "estimated_time_seconds": 5,
  "status_url": "/api/v1/reports/{report_id}/status"
}
```

### Get Report Status

```http
GET /api/v1/reports/{report_id}/status
```

**Response (200 OK):**

```json
{
  "report_id": "uuid",
  "status": "completed",
  "framework": "SBTi",
  "generation_time_seconds": 2.1,
  "quality_score": 98.5,
  "provenance_hash": "sha256:abc123...",
  "download_urls": {
    "PDF": "/api/v1/reports/{report_id}/download?format=pdf",
    "JSON": "/api/v1/reports/{report_id}/download?format=json"
  }
}
```

### Download Report

```http
GET /api/v1/reports/{report_id}/download?format=pdf
```

**Response:** Binary file download with appropriate Content-Type header.

### List Reports

```http
GET /api/v1/reports?organization_id={org_id}&framework={framework}&status={status}&page=1&per_page=20
```

**Response (200 OK):**

```json
{
  "reports": [
    {
      "report_id": "uuid",
      "framework": "SBTi",
      "status": "completed",
      "reporting_period": {"start": "2025-01-01", "end": "2025-12-31"},
      "created_at": "2026-03-20T10:00:00Z",
      "quality_score": 98.5
    }
  ],
  "pagination": {
    "page": 1,
    "per_page": 20,
    "total": 42,
    "total_pages": 3
  }
}
```

### Delete Report

```http
DELETE /api/v1/reports/{report_id}
```

**Response (204 No Content)**

---

## 4. Framework-Specific APIs

### Generate Multi-Framework Report

```http
POST /api/v1/reports/multi-framework
Content-Type: application/json

{
  "organization_id": "uuid",
  "reporting_period": {"start": "2025-01-01", "end": "2025-12-31"},
  "frameworks": ["SBTi", "CDP", "TCFD", "GRI", "ISSB", "SEC", "CSRD"],
  "language": "en",
  "parallel": true
}
```

**Response (202 Accepted):**

```json
{
  "batch_id": "uuid",
  "status": "generating",
  "frameworks": {
    "SBTi": {"report_id": "uuid1", "status": "generating"},
    "CDP": {"report_id": "uuid2", "status": "generating"},
    "TCFD": {"report_id": "uuid3", "status": "generating"},
    "GRI": {"report_id": "uuid4", "status": "generating"},
    "ISSB": {"report_id": "uuid5", "status": "generating"},
    "SEC": {"report_id": "uuid6", "status": "generating"},
    "CSRD": {"report_id": "uuid7", "status": "generating"}
  },
  "estimated_time_seconds": 10,
  "status_url": "/api/v1/reports/batch/{batch_id}/status"
}
```

### Get Framework Coverage

```http
GET /api/v1/frameworks/coverage?organization_id={org_id}&reporting_year=2025
```

**Response (200 OK):**

```json
{
  "organization_id": "uuid",
  "reporting_year": 2025,
  "frameworks": {
    "SBTi": {"coverage_pct": 95.0, "metrics_provided": 19, "metrics_required": 20},
    "CDP": {"coverage_pct": 88.0, "metrics_provided": 264, "metrics_required": 300},
    "TCFD": {"coverage_pct": 100.0, "metrics_provided": 11, "metrics_required": 11},
    "GRI": {"coverage_pct": 92.0, "metrics_provided": 23, "metrics_required": 25},
    "ISSB": {"coverage_pct": 85.0, "metrics_provided": 17, "metrics_required": 20},
    "SEC": {"coverage_pct": 100.0, "metrics_provided": 8, "metrics_required": 8},
    "CSRD": {"coverage_pct": 90.0, "metrics_provided": 36, "metrics_required": 40}
  }
}
```

### Get Framework Deadlines

```http
GET /api/v1/frameworks/deadlines?reporting_year=2026
```

**Response (200 OK):**

```json
{
  "deadlines": [
    {
      "framework": "SEC",
      "deadline_date": "2026-03-31",
      "days_remaining": 11,
      "status": "urgent"
    },
    {
      "framework": "CSRD",
      "deadline_date": "2026-05-31",
      "days_remaining": 72,
      "status": "approaching"
    },
    {
      "framework": "CDP",
      "deadline_date": "2026-07-31",
      "days_remaining": 133,
      "status": "on_track"
    }
  ]
}
```

---

## 5. Workflow APIs

### Execute Workflow

```http
POST /api/v1/workflows/{workflow_name}/execute
Content-Type: application/json

{
  "organization_id": "uuid",
  "parameters": {
    "reporting_period": {"start": "2025-01-01", "end": "2025-12-31"},
    "language": "en"
  }
}
```

**Available Workflows:**

| Endpoint | Description |
|----------|-------------|
| `/workflows/sbti-progress/execute` | SBTi progress report |
| `/workflows/cdp-questionnaire/execute` | CDP questionnaire |
| `/workflows/tcfd-disclosure/execute` | TCFD 4-pillar report |
| `/workflows/gri-305/execute` | GRI 305 disclosure |
| `/workflows/issb-ifrs-s2/execute` | ISSB IFRS S2 |
| `/workflows/sec-climate/execute` | SEC climate disclosure |
| `/workflows/csrd-esrs-e1/execute` | CSRD ESRS E1 |
| `/workflows/multi-framework/execute` | All 7 frameworks |

### Get Workflow Status

```http
GET /api/v1/workflows/{workflow_id}/status
```

---

## 6. Dashboard APIs

### Get Executive Dashboard Data

```http
GET /api/v1/dashboards/executive?organization_id={org_id}
```

### Get Stakeholder View

```http
GET /api/v1/dashboards/stakeholder/{view_type}?organization_id={org_id}
```

**View Types:** `investor`, `regulator`, `customer`, `employee`

### Get Framework Heatmap

```http
GET /api/v1/dashboards/heatmap?organization_id={org_id}&reporting_year=2025
```

---

## 7. Data Aggregation APIs

### Trigger Data Aggregation

```http
POST /api/v1/data/aggregate
Content-Type: application/json

{
  "organization_id": "uuid",
  "reporting_period": {"start": "2025-01-01", "end": "2025-12-31"},
  "sources": ["PACK-021", "PACK-022", "PACK-028", "PACK-029"]
}
```

### Get Aggregation Status

```http
GET /api/v1/data/aggregation/{aggregation_id}/status
```

### Get Data Lineage

```http
GET /api/v1/data/lineage/{report_id}?metric_name={metric_name}
```

### Get Reconciliation Report

```http
GET /api/v1/data/reconciliation/{aggregation_id}
```

---

## 8. Narrative APIs

### Generate Narrative

```http
POST /api/v1/narratives/generate
Content-Type: application/json

{
  "section_type": "governance",
  "framework": "TCFD",
  "language": "en",
  "source_data_id": "uuid"
}
```

### Validate Narrative Consistency

```http
POST /api/v1/narratives/validate-consistency
Content-Type: application/json

{
  "report_ids": ["uuid1", "uuid2", "uuid3"]
}
```

### Translate Narrative

```http
POST /api/v1/narratives/translate
Content-Type: application/json

{
  "narrative_id": "uuid",
  "target_language": "de"
}
```

---

## 9. Validation APIs

### Validate Report

```http
POST /api/v1/validation/report
Content-Type: application/json

{
  "report_id": "uuid",
  "framework": "SBTi",
  "checks": ["schema", "completeness", "consistency"]
}
```

**Response (200 OK):**

```json
{
  "report_id": "uuid",
  "framework": "SBTi",
  "quality_score": 98.5,
  "completeness_score": 95.0,
  "consistency_score": 99.0,
  "errors": [],
  "warnings": [
    {
      "field": "scope3_category_15",
      "message": "Category 15 (Investments) not reported",
      "severity": "low"
    }
  ]
}
```

### Cross-Framework Validation

```http
POST /api/v1/validation/cross-framework
Content-Type: application/json

{
  "report_ids": ["uuid1", "uuid2", "uuid3"]
}
```

---

## 10. Assurance APIs

### Generate Evidence Bundle

```http
POST /api/v1/assurance/evidence-bundle
Content-Type: application/json

{
  "report_ids": ["uuid1", "uuid2"],
  "audit_scope": "full",
  "include_lineage_diagrams": true,
  "include_methodology_docs": true,
  "include_control_matrix": true
}
```

### Download Evidence Bundle

```http
GET /api/v1/assurance/evidence-bundle/{bundle_id}/download
```

### Get Audit Trail

```http
GET /api/v1/assurance/audit-trail?report_id={report_id}&start_date={date}&end_date={date}
```

---

## 11. Configuration APIs

### Get Current Configuration

```http
GET /api/v1/config
```

### Update Configuration

```http
PUT /api/v1/config
Content-Type: application/json

{
  "frameworks": ["SBTi", "CDP", "TCFD"],
  "languages": ["en", "de"],
  "assurance_enabled": true
}
```

### Load Preset

```http
POST /api/v1/config/preset
Content-Type: application/json

{
  "preset_name": "cdp_alist"
}
```

---

## 12. Error Handling

All errors follow the standard GreenLang error format:

```json
{
  "error": {
    "code": "VALIDATION_ERROR",
    "message": "Report validation failed",
    "details": [
      {
        "field": "scope1_total",
        "message": "Scope 1 total is required for SBTi framework"
      }
    ],
    "request_id": "uuid",
    "timestamp": "2026-03-20T10:00:00Z"
  }
}
```

### Error Codes

| Code | HTTP Status | Description |
|------|------------|-------------|
| `AUTHENTICATION_ERROR` | 401 | Invalid or expired token |
| `AUTHORIZATION_ERROR` | 403 | Insufficient permissions |
| `NOT_FOUND` | 404 | Resource not found |
| `VALIDATION_ERROR` | 422 | Input validation failed |
| `RATE_LIMIT_EXCEEDED` | 429 | Rate limit exceeded |
| `DATA_SOURCE_ERROR` | 502 | Upstream data source unavailable |
| `GENERATION_ERROR` | 500 | Report generation failed |
| `TIMEOUT_ERROR` | 504 | Operation timed out |

---

## 13. Rate Limiting

| Endpoint Category | Rate Limit |
|-------------------|-----------|
| Health/metrics | Unlimited |
| Report read | 100 requests/second |
| Report generation | 60 requests/minute |
| Dashboard | 100 requests/second |
| Data aggregation | 30 requests/minute |
| Narrative generation | 30 requests/minute |
| Evidence bundle | 10 requests/minute |

Rate limit headers are included in every response:

```http
X-RateLimit-Limit: 100
X-RateLimit-Remaining: 95
X-RateLimit-Reset: 1679299200
```

---

## 14. Pagination

All list endpoints support cursor-based pagination:

```http
GET /api/v1/reports?page=2&per_page=20&sort=created_at&order=desc
```

**Parameters:**

| Parameter | Default | Max | Description |
|-----------|---------|-----|-------------|
| `page` | 1 | - | Page number |
| `per_page` | 20 | 100 | Items per page |
| `sort` | `created_at` | - | Sort field |
| `order` | `desc` | - | Sort order (asc/desc) |

---

*Built with GreenLang Platform - Zero-Hallucination Climate Intelligence*
