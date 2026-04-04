# AGENT-FOUND-005: Citations & Evidence API Reference

**Agent ID:** AGENT-FOUND-005
**Service:** Citations & Evidence
**Status:** Production Ready
**Base Path:** `/api/v1/citations`
**Tag:** `citations`
**Source:** `greenlang/agents/foundation/citations/api/router.py`

The Citations & Evidence agent manages regulatory citations, evidence packages,
verification records, and supports export/import in multiple formats (JSON,
BibTeX, CSL).

---

## Endpoint Summary

| # | Method | Path | Summary | Status Codes |
|---|--------|------|---------|--------------|
| 1 | GET | `/health` | Health check | 200 |
| 2 | GET | `/metrics` | Metrics summary | 200, 503 |
| 3 | POST | `/` | Create citation | 200, 400, 503 |
| 4 | GET | `/` | List citations | 200, 503 |
| 5 | GET | `/{citation_id}` | Get citation | 200, 404, 503 |
| 6 | PUT | `/{citation_id}` | Update citation | 200, 400, 503 |
| 7 | DELETE | `/{citation_id}` | Delete citation | 200, 400, 404, 503 |
| 8 | GET | `/{citation_id}/versions` | Get version history | 200, 404, 503 |
| 9 | POST | `/{citation_id}/verify` | Verify citation | 200, 404, 503 |
| 10 | POST | `/verify/batch` | Batch verify citations | 200, 503 |
| 11 | GET | `/{citation_id}/verification-history` | Get verification history | 200, 503 |
| 12 | POST | `/packages` | Create evidence package | 200, 400, 503 |
| 13 | GET | `/packages/list` | List evidence packages | 200, 503 |
| 14 | GET | `/packages/{package_id}` | Get evidence package | 200, 404, 503 |
| 15 | POST | `/packages/{package_id}/items` | Add evidence item | 200, 400, 503 |
| 16 | POST | `/packages/{package_id}/citations` | Add citation to package | 200, 400, 503 |
| 17 | POST | `/packages/{package_id}/finalize` | Finalize package | 200, 400, 503 |
| 18 | DELETE | `/packages/{package_id}` | Delete package | 200, 400, 404, 503 |
| 19 | POST | `/export` | Export citations | 200, 400, 503 |
| 20 | POST | `/import` | Import citations | 200, 400, 503 |

---

## Detailed Endpoints

### POST / -- Create Citation

**Request Body:**

```json
{
  "citation_type": "regulation",
  "source_authority": "European Commission",
  "metadata": {
    "title": "EU CBAM Regulation 2023/956",
    "publication_date": "2023-05-17",
    "document_id": "EU-2023-956",
    "url": "https://eur-lex.europa.eu/..."
  },
  "effective_date": "2023-10-01",
  "user_id": "compliance@company.com",
  "change_reason": "Initial regulation entry",
  "expiration_date": null,
  "supersedes": null,
  "regulatory_frameworks": ["CBAM", "EU-ETS"],
  "abstract": "Regulation establishing a carbon border adjustment mechanism",
  "key_values": {
    "transitional_period_end": "2025-12-31",
    "definitive_period_start": "2026-01-01"
  },
  "notes": "Covers cement, iron/steel, aluminium, fertilizers, electricity, hydrogen"
}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `citation_type` | string | Yes | Type: `regulation`, `standard`, `guidance`, `publication`, `dataset` |
| `source_authority` | string | Yes | Issuing authority |
| `metadata` | object | Yes | Citation metadata (title, URL, etc.) |
| `effective_date` | string | Yes | Effective date (`YYYY-MM-DD`) |
| `user_id` | string | No | Creator (default: `system`) |
| `change_reason` | string | No | Reason for entry |
| `citation_id` | string | No | Optional pre-assigned ID |
| `expiration_date` | string | No | Expiration date |
| `supersedes` | string | No | ID of superseded citation |
| `regulatory_frameworks` | array | No | Applicable frameworks |
| `abstract` | string | No | Abstract text |
| `key_values` | object | No | Key values extracted from citation |
| `notes` | string | No | Notes |

**Response (200):**

```json
{
  "citation_id": "cit_abc123",
  "citation_type": "regulation",
  "source_authority": "European Commission",
  "effective_date": "2023-10-01",
  "verification_status": "unverified",
  "version": 1,
  "provenance_hash": "sha256:..."
}
```

---

### GET / -- List Citations

**Query Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `citation_type` | string | Filter by type |
| `source_authority` | string | Filter by authority |
| `verification_status` | string | Filter by status: `unverified`, `verified`, `expired` |
| `search` | string | Search in metadata |

---

### POST /{citation_id}/verify -- Verify Citation

Verify a citation's validity, checking source accessibility, date validity, and
integrity.

**Query Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `user_id` | string | User performing verification (default: `system`) |

**Response (200):**

```json
{
  "citation_id": "cit_abc123",
  "verification_status": "verified",
  "verified_by": "compliance@company.com",
  "verified_at": "2026-04-04T10:00:00Z",
  "checks": {
    "source_accessible": true,
    "date_valid": true,
    "not_superseded": true,
    "hash_matches": true
  },
  "provenance_hash": "sha256:..."
}
```

---

### POST /packages -- Create Evidence Package

Create an evidence package to bundle citations, calculation results, and
supporting evidence for audit or regulatory submission.

**Request Body:**

```json
{
  "name": "CBAM Q4-2025 Evidence Package",
  "description": "Evidence package for CBAM quarterly report",
  "user_id": "compliance@company.com",
  "calculation_context": {
    "reporting_period": "2025-Q4",
    "installation_id": "INST-001"
  },
  "calculation_result": {
    "total_emissions_tCO2e": 1250.5,
    "scope": "embedded"
  },
  "regulatory_frameworks": ["CBAM"],
  "compliance_notes": "All data verified against supplier declarations"
}
```

**Response (200):**

```json
{
  "package_id": "pkg_xyz789",
  "name": "CBAM Q4-2025 Evidence Package",
  "status": "draft",
  "citation_count": 0,
  "evidence_item_count": 0,
  "created_at": "2026-04-04T10:00:00Z"
}
```

---

### POST /packages/{package_id}/items -- Add Evidence Item

**Request Body:**

```json
{
  "evidence_type": "calculation_output",
  "description": "GHG emission calculation results",
  "data": {
    "total_emissions": 1250.5,
    "methodology": "direct_measurement"
  },
  "citation_ids": ["cit_abc123"],
  "source_system": "greenlang",
  "source_agent": "ghg-scope1-stationary"
}
```

---

### POST /packages/{package_id}/finalize -- Finalize Package

Seal an evidence package, computing a tamper-evident hash. Once finalized, the
package is immutable.

**Response (200):**

```json
{
  "package_id": "pkg_xyz789",
  "package_hash": "sha256:fedcba...",
  "finalized": true
}
```

---

### POST /export -- Export Citations

**Request Body:**

```json
{
  "citation_ids": ["cit_abc123", "cit_def456"],
  "format": "bibtex"
}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `citation_ids` | array | No | Specific IDs (null = export all) |
| `format` | string | No | Format: `json` (default), `bibtex`, `csl` |

**Response (200):**

```json
{
  "format": "bibtex",
  "content": "@article{cit_abc123, ...}",
  "count": 2
}
```

---

### POST /import -- Import Citations

**Request Body:**

```json
{
  "content": "[{\"citation_type\": \"regulation\", ...}]",
  "user_id": "admin@company.com"
}
```

**Response (200):**

```json
{
  "imported_count": 5,
  "citation_ids": ["cit_001", "cit_002", "cit_003", "cit_004", "cit_005"]
}
```
