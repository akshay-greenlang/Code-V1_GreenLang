# AGENT-EUDR-012: Document Authentication API

**Agent ID:** `GL-EUDR-DAV-012`
**Prefix:** `/v1/eudr-dav`
**Version:** 1.0.0
**PRD:** PRD-AGENT-EUDR-012
**Regulation:** EU 2023/1115 (EUDR) -- Document verification per Articles 9 and 12

## Purpose

The Document Authentication agent verifies the authenticity and integrity of
supply chain documents required for EUDR compliance. It classifies document
types, verifies digital and wet signatures, computes and validates content
hashes, validates certificates (phytosanitary, origin, organic), extracts
metadata, detects document fraud using ML models, cross-references documents
against external registries, and generates authentication reports.

---

## Endpoint Summary

| Method | Path | Summary | Auth |
|--------|------|---------|------|
| POST | `/classify` | Classify document type | JWT |
| POST | `/classify/batch` | Batch classify documents | JWT |
| GET | `/classify/{doc_id}` | Get classification result | JWT |
| GET | `/classify/types` | List supported document types | JWT |
| POST | `/classify/train` | Submit training sample | JWT |
| POST | `/signatures/verify` | Verify document signature | JWT |
| GET | `/signatures/{verification_id}` | Get signature result | JWT |
| POST | `/signatures/batch` | Batch verify signatures | JWT |
| GET | `/signatures/algorithms` | List supported algorithms | JWT |
| POST | `/hash/compute` | Compute content hash | JWT |
| POST | `/hash/verify` | Verify content hash | JWT |
| GET | `/hash/{hash_id}` | Get hash record | JWT |
| GET | `/hash/history/{doc_id}` | Get hash history | JWT |
| POST | `/certificates/validate` | Validate certificate | JWT |
| GET | `/certificates` | List validated certificates | JWT |
| GET | `/certificates/{cert_id}` | Get certificate details | JWT |
| POST | `/certificates/chain` | Validate certificate chain | JWT |
| POST | `/metadata/extract` | Extract document metadata | JWT |
| GET | `/metadata/{extraction_id}` | Get extraction result | JWT |
| POST | `/metadata/batch` | Batch extract metadata | JWT |
| POST | `/fraud/detect` | Run fraud detection | JWT |
| GET | `/fraud/{detection_id}` | Get fraud detection result | JWT |
| POST | `/fraud/batch` | Batch fraud detection | JWT |
| GET | `/fraud/patterns` | List known fraud patterns | JWT |
| POST | `/fraud/report` | Report suspected fraud | JWT |
| POST | `/crossref/verify` | Cross-reference verification | JWT |
| GET | `/crossref/{verification_id}` | Get cross-ref result | JWT |
| POST | `/crossref/batch` | Batch cross-reference check | JWT |
| GET | `/crossref/registries` | List available registries | JWT |
| POST | `/reports/generate` | Generate authentication report | JWT |
| GET | `/reports` | List reports | JWT |
| GET | `/reports/{report_id}` | Get report details | JWT |
| GET | `/reports/{report_id}/download` | Download report | JWT |
| POST | `/reports/dashboard` | Get dashboard data | JWT |
| POST | `/batch` | Submit batch job | JWT |
| DELETE | `/batch/{job_id}` | Cancel batch job | JWT |
| GET | `/health` | Health check | None |

**Total: 37 endpoints**

---

## Endpoints

### POST /v1/eudr-dav/classify

Classify a document against EUDR-relevant document types using an ML
classification model. Supported types include certificates of origin,
phytosanitary certificates, export permits, organic certificates, FLEGT
licenses, bills of lading, and customs declarations.

**Request:**

```json
{
  "document_url": "https://storage.greenlang.io/docs/cert-origin-001.pdf",
  "filename": "cert-origin-001.pdf",
  "mime_type": "application/pdf",
  "operator_id": "OP-2024-001",
  "expected_type": "certificate_of_origin"
}
```

**Response (200 OK):**

```json
{
  "doc_id": "doc_001",
  "predicted_type": "certificate_of_origin",
  "confidence": 0.96,
  "alternative_types": [
    {"type": "export_permit", "confidence": 0.03}
  ],
  "expected_match": true,
  "metadata_extracted": true,
  "classified_at": "2026-04-04T10:00:00Z"
}
```

---

### POST /v1/eudr-dav/fraud/detect

Run fraud detection analysis on a document using pattern matching, metadata
consistency checks, and anomaly detection.

**Request:**

```json
{
  "document_url": "https://storage.greenlang.io/docs/cert-origin-001.pdf",
  "document_type": "certificate_of_origin",
  "issuing_authority": "Ghana Customs",
  "issue_date": "2026-01-10",
  "check_types": ["metadata_consistency", "template_matching", "anomaly_detection"]
}
```

**Response (200 OK):**

```json
{
  "detection_id": "fraud_001",
  "document_authentic": true,
  "fraud_risk_score": 0.08,
  "risk_level": "low",
  "checks_performed": [
    {"check": "metadata_consistency", "passed": true, "details": "All metadata fields consistent"},
    {"check": "template_matching", "passed": true, "details": "Matches Ghana Customs template v3"},
    {"check": "anomaly_detection", "passed": true, "details": "No anomalies detected"}
  ],
  "warnings": [],
  "detected_at": "2026-04-04T10:10:00Z"
}
```

---

### POST /v1/eudr-dav/batch

Submit an asynchronous batch processing job. Supported types:
`classify_batch`, `verify_signatures_batch`, `detect_fraud_batch`,
`crossref_batch`, `report_generation`.

**Request:**

```json
{
  "job_type": "classify_batch",
  "priority": 5,
  "parameters": {
    "document_urls": [
      "https://storage.greenlang.io/docs/cert-001.pdf",
      "https://storage.greenlang.io/docs/cert-002.pdf"
    ]
  },
  "callback_url": "https://client.example.com/webhooks/dav"
}
```

**Response (202 Accepted):**

```json
{
  "job_id": "job_dav_001",
  "job_type": "classify_batch",
  "status": "queued",
  "priority": 5,
  "progress_percent": 0.0,
  "submitted_at": "2026-04-04T10:15:00Z",
  "provenance_hash": "sha256:a1b2c3d4..."
}
```

---

## Error Responses

| Status | Error Code | Description |
|--------|------------|-------------|
| 400 | `invalid_document` | Document cannot be processed |
| 404 | `document_not_found` | Document ID not found |
| 409 | `job_not_cancellable` | Batch job already completed |
| 422 | `unsupported_format` | Document format not supported |
