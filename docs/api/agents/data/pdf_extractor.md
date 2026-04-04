# PDF & Invoice Extractor API Reference

**Agent:** AGENT-DATA-001
**Prefix:** `/api/v1/pdf-extractor`
**Source:** `greenlang/agents/data/pdf_extractor/api/router.py`
**Status:** Production Ready

## Overview

The PDF & Invoice Extractor agent provides document ingestion, OCR-based field extraction, and specialized processing for invoices, shipping manifests, and utility bills. It supports multiple OCR engines (Tesseract, AWS Textract, Azure, Google) and template-based extraction patterns.

---

## Endpoint Summary

| # | Method | Path | Summary | Auth |
|---|--------|------|---------|------|
| 1 | POST | `/v1/documents/ingest` | Ingest single document | Yes |
| 2 | POST | `/v1/documents/batch` | Batch ingest documents | Yes |
| 3 | GET | `/v1/documents` | List documents | Yes |
| 4 | GET | `/v1/documents/{document_id}` | Get document details | Yes |
| 5 | GET | `/v1/documents/{document_id}/fields` | Get extracted fields | Yes |
| 6 | GET | `/v1/documents/{document_id}/pages` | Get document pages | Yes |
| 7 | POST | `/v1/documents/{document_id}/reprocess` | Reprocess document | Yes |
| 8 | POST | `/v1/documents/classify` | Classify document type | Yes |
| 9 | POST | `/v1/invoices/extract` | Extract invoice data | Yes |
| 10 | GET | `/v1/invoices/{document_id}` | Get invoice result | Yes |
| 11 | POST | `/v1/manifests/extract` | Extract manifest data | Yes |
| 12 | GET | `/v1/manifests/{document_id}` | Get manifest result | Yes |
| 13 | POST | `/v1/utility-bills/extract` | Extract utility bill data | Yes |
| 14 | GET | `/v1/utility-bills/{document_id}` | Get utility bill result | Yes |
| 15 | POST | `/v1/templates` | Create extraction template | Yes |
| 16 | GET | `/v1/templates` | List templates | Yes |
| 17 | POST | `/v1/validate/{document_id}` | Run validation rules | Yes |
| 18 | GET | `/v1/jobs` | List extraction jobs | Yes |
| 19 | GET | `/v1/statistics` | Get service statistics | Yes |
| 20 | GET | `/health` | Health check | No |

---

## Endpoints

### 1. Ingest Single Document

```http
POST /api/v1/pdf-extractor/v1/documents/ingest
Content-Type: application/json
Authorization: Bearer {token}
```

**Request Body:**

```json
{
  "file_path": "/uploads/invoice_2026_Q1.pdf",
  "file_base64": null,
  "file_content": null,
  "document_type": "invoice",
  "ocr_engine": "tesseract",
  "confidence_threshold": 0.85,
  "tenant_id": "tenant_abc"
}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `file_path` | string | Optional* | Server-side file path to ingest |
| `file_content` | string | Optional* | Raw text content of the document |
| `file_base64` | string | Optional* | Base64-encoded file content |
| `document_type` | string | Optional | Type hint: `invoice`, `manifest`, `utility_bill`, `receipt`, `other` |
| `ocr_engine` | string | Optional | OCR engine: `tesseract`, `textract`, `azure`, `google` |
| `confidence_threshold` | float | Optional | Minimum confidence (0.0-1.0) |
| `tenant_id` | string | Optional | Tenant identifier (default: `"default"`) |

*At least one of `file_path`, `file_content`, or `file_base64` must be provided.

**Response (200):**

```json
{
  "document_id": "doc_abc123",
  "document_type": "invoice",
  "status": "processed",
  "page_count": 3,
  "extracted_fields": {
    "vendor_name": "Acme Corp",
    "invoice_number": "INV-2026-001",
    "total_amount": 15420.50,
    "currency": "USD"
  },
  "confidence": 0.92,
  "provenance_hash": "sha256:abc123..."
}
```

### 2. Batch Ingest Documents

```http
POST /api/v1/pdf-extractor/v1/documents/batch
```

**Request Body:**

```json
{
  "documents": [
    {
      "file_path": "/uploads/inv_001.pdf",
      "document_type": "invoice"
    },
    {
      "file_base64": "<base64-content>",
      "document_type": "utility_bill"
    }
  ],
  "ocr_engine": "textract",
  "confidence_threshold": 0.8,
  "tenant_id": "tenant_abc"
}
```

### 3. List Documents

```http
GET /api/v1/pdf-extractor/v1/documents?limit=50&offset=0&document_type=invoice&tenant_id=tenant_abc
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `limit` | integer | 50 | Items per page (1-200) |
| `offset` | integer | 0 | Skip count |
| `tenant_id` | string | null | Filter by tenant |
| `document_type` | string | null | Filter by type |

### 9. Extract Invoice Data

```http
POST /api/v1/pdf-extractor/v1/invoices/extract
```

**Request Body:**

```json
{
  "document_id": "doc_abc123",
  "confidence_threshold": 0.85,
  "template": "standard_invoice_v1"
}
```

**Response:**

```json
{
  "document_id": "doc_abc123",
  "vendor_name": "Acme Corp",
  "invoice_number": "INV-2026-001",
  "invoice_date": "2026-01-15",
  "due_date": "2026-02-15",
  "line_items": [
    {
      "description": "Natural Gas Supply - January 2026",
      "quantity": 15000,
      "unit": "therms",
      "unit_price": 0.95,
      "total": 14250.00
    }
  ],
  "subtotal": 14250.00,
  "tax": 1170.50,
  "total_amount": 15420.50,
  "currency": "USD",
  "confidence": 0.92
}
```

### 15. Create Extraction Template

```http
POST /api/v1/pdf-extractor/v1/templates
```

**Request Body:**

```json
{
  "name": "utility_bill_template_v2",
  "template_type": "utility_bill",
  "field_patterns": {
    "account_number": "Account\\s*#?:?\\s*(\\d{8,12})",
    "billing_period": "(\\d{2}/\\d{2}/\\d{4})\\s*-\\s*(\\d{2}/\\d{2}/\\d{4})",
    "total_kwh": "Total\\s+Usage:?\\s*([\\d,]+\\.?\\d*)\\s*kWh"
  },
  "validation_rules": {
    "total_kwh_positive": "total_kwh > 0"
  },
  "description": "Template for standard US utility bills"
}
```

### 20. Health Check

```http
GET /api/v1/pdf-extractor/health
```

**Response:**

```json
{
  "status": "healthy",
  "service": "pdf-extractor"
}
```

---

## Error Responses

```json
{
  "detail": "Document doc_xyz not found"
}
```

| Status | Condition |
|--------|-----------|
| 400 | Invalid input (missing file source, unsupported format) |
| 404 | Document, template, or job not found |
| 503 | PDF extractor service not configured |
