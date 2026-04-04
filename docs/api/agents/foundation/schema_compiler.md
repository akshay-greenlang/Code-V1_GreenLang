# AGENT-FOUND-002: Schema Compiler & Validator API Reference

**Agent ID:** AGENT-FOUND-002
**Service:** Schema Compiler & Validator
**Status:** Production Ready
**Base Path:** `/v1/schema` (validation/compilation), root (system endpoints)
**Source:** `greenlang/agents/foundation/schema/api/routes.py`

The Schema Compiler & Validator provides endpoints for payload validation against
GreenLang schemas, batch validation, schema compilation to intermediate
representation (IR), and schema registry lookup.

---

## Endpoint Summary

| Method | Path | Summary | Status Codes |
|--------|------|---------|--------------|
| POST | `/v1/schema/validate` | Validate a single payload | 200, 400, 404, 429, 500 |
| POST | `/v1/schema/validate/batch` | Validate multiple payloads | 200, 400, 404, 413, 429, 500 |
| POST | `/v1/schema/compile` | Compile a schema to IR | 200, 400, 429, 500 |
| GET | `/v1/schema/{schema_id}/versions` | List schema versions | 200, 404, 500 |
| GET | `/v1/schema/{schema_id}/{version}` | Get schema details | 200, 404, 500 |
| GET | `/health` | Health check | 200, 503 |
| GET | `/metrics` | Get service metrics (JSON) | 200 |
| GET | `/metrics/prometheus` | Prometheus metrics (text) | 200 |

---

## Authentication

All validation and compilation endpoints require an API key and are subject to
rate limiting. System endpoints (`/health`, `/metrics`) do not require
authentication.

```
X-API-Key: <api_key>
```

---

## Detailed Endpoints

### POST /v1/schema/validate -- Validate Single Payload

Validate a single payload against a GreenLang schema through multiple phases:

1. **Parse** -- Parse YAML/JSON payload
2. **Compile** -- Compile schema to IR (cached)
3. **Structural** -- Validate types, required fields
4. **Constraints** -- Validate ranges, patterns, enums
5. **Units** -- Validate unit dimensions
6. **Rules** -- Evaluate cross-field rules
7. **Lint** -- Check for typos, deprecated fields

**Request Body:**

```json
{
  "schema_ref": "emissions/activity@1.3.0",
  "payload": {
    "activity_type": "stationary_combustion",
    "fuel_type": "natural_gas",
    "quantity": 1500.0,
    "unit": "m3",
    "period": "2025-Q4"
  },
  "options": {
    "strict": true,
    "include_suggestions": true
  }
}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `schema_ref` | string | Yes | Schema reference (`schema_id@version`) |
| `payload` | object | Yes | Data payload to validate |
| `options` | object | No | Validation options (strict mode, etc.) |

**Response (200):**

```json
{
  "valid": true,
  "schema_ref": "emissions/activity@1.3.0",
  "schema_hash": "sha256:a1b2c3d4...",
  "summary": {
    "error_count": 0,
    "warning_count": 1,
    "info_count": 2
  },
  "findings": [
    {
      "severity": "warning",
      "path": "$.unit",
      "message": "Consider using SI unit 'cubic_meter' instead of 'm3'",
      "code": "GLSCHEMA-LINT-001"
    }
  ],
  "normalized_payload": {
    "activity_type": "stationary_combustion",
    "fuel_type": "natural_gas",
    "quantity": 1500.0,
    "unit": "cubic_meter",
    "period": "2025-Q4"
  },
  "fix_suggestions": [],
  "timings_ms": {
    "parse": 0.5,
    "compile": 1.2,
    "structural": 0.8,
    "constraints": 0.3,
    "total": 3.1
  }
}
```

---

### POST /v1/schema/validate/batch -- Validate Multiple Payloads

Validate multiple payloads against the same schema in a single request. The
schema is compiled once and reused for all payloads.

**Request Body:**

```json
{
  "schema_ref": "emissions/activity@1.3.0",
  "payloads": [
    { "activity_type": "stationary_combustion", "fuel_type": "natural_gas", "quantity": 1500.0, "unit": "m3" },
    { "activity_type": "mobile_combustion", "fuel_type": "diesel", "quantity": 200.0, "unit": "liters" },
    { "activity_type": "invalid_type", "fuel_type": null }
  ],
  "options": {}
}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `schema_ref` | string | Yes | Schema reference |
| `payloads` | array | Yes | Array of payloads to validate (max: 1000) |
| `options` | object | No | Validation options |

**Response (200):**

```json
{
  "schema_ref": "emissions/activity@1.3.0",
  "schema_hash": "sha256:a1b2c3d4...",
  "summary": {
    "total_items": 3,
    "valid_count": 2,
    "error_count": 1,
    "warning_count": 0,
    "total_findings": 3,
    "processing_time_ms": 12.5
  },
  "results": [
    { "index": 0, "id": null, "valid": true, "findings": [], "normalized_payload": { ... } },
    { "index": 1, "id": null, "valid": true, "findings": [], "normalized_payload": { ... } },
    { "index": 2, "id": null, "valid": false, "findings": [ { "severity": "error", "path": "$.activity_type", "message": "Invalid enum value" } ] }
  ]
}
```

**Error (413) -- Batch Too Large:**

```json
{
  "error": "batch_too_large",
  "message": "Batch size 1500 exceeds maximum 1000",
  "details": [
    { "code": "GLSCHEMA-API-BATCH-001", "message": "Maximum batch size is 1000", "field": "payloads" }
  ]
}
```

---

### POST /v1/schema/compile -- Compile Schema

Compile a schema to intermediate representation (IR), computing the schema hash,
property definitions, constraint metadata, and rule bindings.

**Request Body:**

```json
{
  "schema_source": {
    "type": "object",
    "properties": {
      "fuel_type": { "type": "string", "enum": ["natural_gas", "diesel", "coal"] },
      "quantity": { "type": "number", "minimum": 0 }
    },
    "required": ["fuel_type", "quantity"]
  },
  "schema_ref": {
    "schema_id": "custom/fuel_input",
    "version": "1.0.0"
  },
  "include_ir": true
}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `schema_source` | object or string | Yes | Schema definition (JSON object or YAML string) |
| `schema_ref` | object | No | Schema reference for caching (schema_id + version) |
| `include_ir` | boolean | No | Include compiled IR in response (default: false) |

**Response (200):**

```json
{
  "schema_hash": "sha256:e7f8g9h0...",
  "compiled_at": "2026-04-04T10:00:00Z",
  "warnings": [],
  "properties_count": 2,
  "rules_count": 0,
  "ir": {
    "schema_id": "custom/fuel_input",
    "version": "1.0.0",
    "schema_hash": "sha256:e7f8g9h0...",
    "properties": ["fuel_type", "quantity"],
    "required_paths": ["fuel_type", "quantity"]
  }
}
```

---

### GET /v1/schema/{schema_id}/versions -- List Schema Versions

Get all available versions of a schema from the registry.

**Path Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `schema_id` | string | Schema identifier (e.g., `emissions/activity`) |

**Response (200):**

```json
{
  "schema_id": "emissions/activity",
  "versions": [
    { "version": "1.0.0", "created_at": "2025-06-01T00:00:00Z", "deprecated": true, "deprecated_message": "Use v1.3.0" },
    { "version": "1.2.0", "created_at": "2025-09-15T00:00:00Z", "deprecated": false },
    { "version": "1.3.0", "created_at": "2026-01-10T00:00:00Z", "deprecated": false }
  ],
  "latest": "1.3.0"
}
```

---

### GET /v1/schema/{schema_id}/{version} -- Get Schema Details

Get the full schema definition for a specific version.

**Response (200):**

```json
{
  "schema_id": "emissions/activity",
  "version": "1.3.0",
  "schema_hash": "sha256:a1b2c3d4...",
  "content": {
    "type": "object",
    "properties": { ... }
  },
  "created_at": "2026-01-10T00:00:00Z",
  "deprecated": false
}
```

---

### GET /health -- Health Check

**Response (200):**

```json
{
  "status": "healthy",
  "version": "1.0.0",
  "cache_size": 42,
  "uptime_seconds": 86400.5,
  "components": [
    { "name": "validator", "status": "healthy", "message": "Validator is operational" },
    { "name": "registry", "status": "healthy", "message": "Registry is configured" }
  ],
  "timestamp": "2026-04-04T12:00:00Z"
}
```

---

### GET /metrics -- Service Metrics (JSON)

**Response (200):**

```json
{
  "validations_total": 15230,
  "validations_success": 14985,
  "validations_failed": 245,
  "batch_validations_total": 320,
  "batch_items_total": 48000,
  "cache_hits": 14800,
  "cache_misses": 430,
  "cache_size": 42,
  "avg_validation_time_ms": 3.2,
  "p95_validation_time_ms": 8.5,
  "uptime_seconds": 86400.5
}
```

### GET /metrics/prometheus -- Prometheus Metrics

Returns metrics in Prometheus text exposition format (`text/plain; version=0.0.4`).

---

## Error Codes

| Code | Description |
|------|-------------|
| `GLSCHEMA-API-VAL-002` | Invalid parameter |
| `GLSCHEMA-API-SCH-001` | Schema not found |
| `GLSCHEMA-API-VER-001` | Version not found |
| `GLSCHEMA-API-CMP-001` | Compilation failed |
| `GLSCHEMA-API-BATCH-001` | Batch too large |
| `GLSCHEMA-API-SYS-001` | Internal error |
