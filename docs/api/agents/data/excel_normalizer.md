# Excel & CSV Normalizer API Reference

**Agent:** AGENT-DATA-002
**Prefix:** `/api/v1/excel-normalizer`
**Source:** `greenlang/agents/data/excel_normalizer/api/router.py`
**Status:** Production Ready

## Overview

The Excel & CSV Normalizer agent handles file upload, column mapping to canonical GreenLang schemas, data type detection, schema validation, transform pipelines, and data quality scoring. Supports xlsx, xls, csv, and tsv formats.

---

## Endpoint Summary

| # | Method | Path | Summary | Auth |
|---|--------|------|---------|------|
| 1 | POST | `/v1/files/upload` | Upload single file | Yes |
| 2 | POST | `/v1/files/batch` | Batch upload files | Yes |
| 3 | GET | `/v1/files` | List files | Yes |
| 4 | GET | `/v1/files/{file_id}` | Get file details | Yes |
| 5 | GET | `/v1/files/{file_id}/sheets` | Get file sheets | Yes |
| 6 | GET | `/v1/files/{file_id}/preview` | Preview file data | Yes |
| 7 | POST | `/v1/files/{file_id}/reprocess` | Reprocess file | Yes |
| 8 | POST | `/v1/normalize` | Normalize inline data | Yes |
| 9 | POST | `/v1/columns/map` | Map columns to canonical schema | Yes |
| 10 | GET | `/v1/columns/canonical` | List canonical fields | Yes |
| 11 | POST | `/v1/columns/detect-types` | Detect column data types | Yes |
| 12 | POST | `/v1/validate` | Validate data against schema | Yes |
| 13 | POST | `/v1/transform` | Apply transform pipeline | Yes |
| 14 | GET | `/v1/quality/{file_id}` | Get quality score | Yes |
| 15 | POST | `/v1/templates` | Create mapping template | Yes |
| 16 | GET | `/v1/templates` | List templates | Yes |
| 17 | GET | `/v1/templates/{template_id}` | Get template details | Yes |
| 18 | GET | `/v1/jobs` | List normalization jobs | Yes |
| 19 | GET | `/v1/statistics` | Get statistics | Yes |
| 20 | GET | `/health` | Health check | No |

---

## Key Endpoints

### 1. Upload Single File

```http
POST /api/v1/excel-normalizer/v1/files/upload
Content-Type: application/json
Authorization: Bearer {token}
```

**Request Body:**

```json
{
  "file_name": "emissions_report_Q1.xlsx",
  "file_content_base64": "<base64-encoded-content>",
  "file_format": "xlsx",
  "template_id": "tpl_ghg_inventory",
  "tenant_id": "tenant_abc"
}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `file_name` | string | Yes | Original filename including extension |
| `file_content_base64` | string | Yes | Base64-encoded file content |
| `file_format` | string | Optional | Format hint: `xlsx`, `xls`, `csv`, `tsv` (auto-detected if omitted) |
| `template_id` | string | Optional | Mapping template ID to apply |
| `tenant_id` | string | Optional | Tenant identifier (default: `"default"`) |

### 9. Map Columns to Canonical Schema

```http
POST /api/v1/excel-normalizer/v1/columns/map
```

**Request Body:**

```json
{
  "headers": ["Facility Name", "CO2 Emissions (tonnes)", "Year", "Scope"],
  "strategy": "fuzzy",
  "template_id": null
}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `headers` | string[] | Yes | Source column headers to map |
| `strategy` | string | Optional | Mapping strategy: `exact`, `synonym`, `fuzzy`, `pattern`, `manual` |
| `template_id` | string | Optional | Template for pre-defined mappings |

**Response:**

```json
{
  "mappings": {
    "Facility Name": "facility_name",
    "CO2 Emissions (tonnes)": "co2_emissions_tonnes",
    "Year": "reporting_year",
    "Scope": "ghg_scope"
  },
  "confidence_scores": {
    "Facility Name": 0.98,
    "CO2 Emissions (tonnes)": 0.91,
    "Year": 0.95,
    "Scope": 0.88
  },
  "unmapped": []
}
```

### 14. Get Quality Score

```http
GET /api/v1/excel-normalizer/v1/quality/{file_id}
```

**Response:**

```json
{
  "file_id": "file_abc123",
  "quality_score": 94.2,
  "completeness": 97.5,
  "accuracy": 92.1,
  "consistency": 93.0,
  "provenance_hash": "sha256:def456..."
}
```

---

## Error Responses

| Status | Condition |
|--------|-----------|
| 400 | Invalid input data, unsupported format, or failed validation |
| 404 | File or template not found |
| 503 | Excel normalizer service not configured |
