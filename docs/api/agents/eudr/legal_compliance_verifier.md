# AGENT-EUDR-023: Legal Compliance Verifier API

**Agent ID:** `GL-EUDR-LCV-023`
**Prefix:** `/v1/eudr-lcv`
**Version:** 1.0.0
**PRD:** AGENT-EUDR-023
**Regulation:** EU 2023/1115 (EUDR) -- Legal compliance per Articles 3(b) and 10

## Purpose

The Legal Compliance Verifier agent determines whether commodities in a
supply chain were produced in compliance with the relevant laws of the country
of production, as required by EUDR Article 3(b). It maintains a registry of
legal frameworks by country and commodity, verifies supporting documents,
validates certifications for EUDR equivalence, detects red flags indicating
potential non-compliance, performs compliance assessments, ingests audit
findings, and generates compliance reports.

---

## Endpoint Summary

| Method | Path | Summary | Auth |
|--------|------|---------|------|
| POST | `/frameworks` | Register legal framework | JWT |
| GET | `/frameworks` | List legal frameworks | JWT |
| GET | `/frameworks/{framework_id}` | Get framework details | JWT |
| PUT | `/frameworks/{framework_id}` | Update framework | JWT |
| POST | `/frameworks/search` | Search frameworks | JWT |
| POST | `/documents/verify` | Verify legal document | JWT |
| GET | `/documents` | List verified documents | JWT |
| GET | `/documents/{doc_id}` | Get document details | JWT |
| POST | `/documents/validity-check` | Check document validity | JWT |
| GET | `/documents/expiring` | List expiring documents | JWT |
| POST | `/certifications/validate` | Validate certification | JWT |
| GET | `/certifications` | List certifications | JWT |
| GET | `/certifications/{cert_id}` | Get certification details | JWT |
| POST | `/certifications/eudr-equivalence` | Check EUDR equivalence | JWT |
| POST | `/red-flags/detect` | Detect red flags | JWT |
| GET | `/red-flags` | List detected red flags | JWT |
| GET | `/red-flags/{flag_id}` | Get red flag details | JWT |
| PUT | `/red-flags/{flag_id}/suppress` | Suppress false positive | JWT |
| POST | `/compliance/assess` | Run compliance assessment | JWT |
| POST | `/compliance/check-category` | Check specific category | JWT |
| GET | `/compliance` | List assessments | JWT |
| GET | `/compliance/{assessment_id}` | Get assessment details | JWT |
| GET | `/compliance/{assessment_id}/history` | Get assessment history | JWT |
| POST | `/audits/ingest` | Ingest audit findings | JWT |
| GET | `/audits` | List audit findings | JWT |
| GET | `/audits/{audit_id}/findings` | Get audit findings | JWT |
| PUT | `/audits/{audit_id}/corrective-actions` | Update corrective actions | JWT |
| POST | `/reports/generate` | Generate compliance report | JWT |
| GET | `/reports` | List reports | JWT |
| GET | `/reports/{report_id}/download` | Download report | JWT |
| POST | `/reports/schedule` | Schedule periodic report | JWT |
| POST | `/batch/assess` | Batch compliance assessment | JWT |
| POST | `/batch/verify` | Batch document verification | JWT |
| GET | `/batch/{job_id}/status` | Get batch job status | JWT |
| GET | `/health` | Health check | None |

**Total: 35 endpoints + health**

**RBAC Permissions (20):**

| Permission | Operations |
|------------|------------|
| `eudr-lcv:framework:create` | Register, update frameworks |
| `eudr-lcv:framework:read` | List, get, search frameworks |
| `eudr-lcv:document:create` | Verify, validity-check documents |
| `eudr-lcv:document:read` | List, get, expiring documents |
| `eudr-lcv:certification:create` | Validate, check equivalence |
| `eudr-lcv:certification:read` | List, get certifications |
| `eudr-lcv:red-flag:create` | Detect red flags |
| `eudr-lcv:red-flag:read` | List, get red flags |
| `eudr-lcv:red-flag:update` | Suppress false positives |
| `eudr-lcv:compliance:create` | Assess, check-category |
| `eudr-lcv:compliance:read` | List, get, history |
| `eudr-lcv:audit:create` | Ingest findings |
| `eudr-lcv:audit:read` | List, get findings |
| `eudr-lcv:audit:update` | Corrective actions |
| `eudr-lcv:report:create` | Generate, schedule reports |
| `eudr-lcv:report:read` | List, download reports |
| `eudr-lcv:batch:create` | Batch operations |
| `eudr-lcv:batch:read` | Batch status |

**Rate Limiting:**

| Tier | Limit | Endpoints |
|------|-------|-----------|
| Standard | 100/min | GET list, GET detail, POST search |
| Write | 30/min | POST register, PUT update, POST verify |
| Heavy | 10/min | POST detect, POST assess, batch operations |
| Export | 5/min | POST reports/generate |

---

## Endpoints

### POST /v1/eudr-lcv/compliance/assess

Run a comprehensive legal compliance assessment for an operator and
commodity, evaluating all relevant country-of-production laws.

**Request:**

```json
{
  "operator_id": "OP-2024-001",
  "commodity": "cocoa",
  "country_code": "GH",
  "evidence": {
    "documents": ["doc_001", "doc_002"],
    "certifications": ["cert_001"],
    "audit_findings": []
  },
  "assessment_depth": "full"
}
```

**Response (200 OK):**

```json
{
  "assessment_id": "asmt_001",
  "operator_id": "OP-2024-001",
  "commodity": "cocoa",
  "country_code": "GH",
  "overall_status": "compliant",
  "categories": [
    {
      "category": "land_rights",
      "status": "compliant",
      "framework": "Ghana Lands Commission Act 2008",
      "evidence_count": 2
    },
    {
      "category": "environmental_protection",
      "status": "compliant",
      "framework": "Ghana Environmental Protection Agency Act 1994",
      "evidence_count": 1
    },
    {
      "category": "labor_rights",
      "status": "compliant",
      "framework": "Ghana Labour Act 2003",
      "evidence_count": 1
    }
  ],
  "red_flags": [],
  "confidence": 0.88,
  "assessed_at": "2026-04-04T10:00:00Z"
}
```

---

### POST /v1/eudr-lcv/red-flags/detect

Scan supply chain data for red flags indicating potential legal
non-compliance, such as document inconsistencies, sanctions exposure,
or known fraud indicators.

**Request:**

```json
{
  "operator_id": "OP-2024-001",
  "supplier_ids": ["sup-001", "sup-002"],
  "commodity": "cocoa",
  "check_types": ["sanctions", "document_inconsistency", "fraud_indicator", "labor_risk"]
}
```

**Response (200 OK):**

```json
{
  "detection_id": "rf_001",
  "total_flags": 1,
  "red_flags": [
    {
      "flag_id": "flag_001",
      "flag_type": "document_inconsistency",
      "severity": "medium",
      "supplier_id": "sup-002",
      "description": "Certificate of origin date precedes company registration date",
      "recommendation": "Request updated certificate from supplier",
      "auto_suppressible": false
    }
  ],
  "clean_suppliers": ["sup-001"],
  "detected_at": "2026-04-04T10:10:00Z"
}
```

---

## Error Responses

| Status | Error Code | Description |
|--------|------------|-------------|
| 400 | `invalid_assessment` | Assessment parameters are invalid |
| 404 | `framework_not_found` | Legal framework ID not found |
| 422 | `unsupported_country` | No legal framework data for country |
| 429 | `rate_limit_exceeded` | Request rate exceeded |
