# Schema Compiler and Validator API Reference (AGENT-FOUND-002)

## Overview

The Schema Compiler and Validator Service provides REST endpoints for validating payloads against GreenLang schemas, compiling schemas to intermediate representation (IR), and querying the schema registry. Supports multi-phase validation (structural, constraints, units, rules, lint), batch processing, and Prometheus-compatible metrics.

**Router Prefix:** `/v1/schema` (schema operations), root-level (system endpoints)
**Tags:** `Schema Validation`, `System`
**Source:** `greenlang/agents/foundation/schema/api/routes.py`

---

## Endpoint Summary

### Validation

| Method | Path | Summary | Auth Required |
|--------|------|---------|---------------|
| POST | `/v1/schema/validate` | Validate single payload | Yes |
| POST | `/v1/schema/validate/batch` | Validate multiple payloads | Yes |

### Compilation

| Method | Path | Summary | Auth Required |
|--------|------|---------|---------------|
| POST | `/v1/schema/compile` | Compile schema to IR | Yes |

### Schema Registry

| Method | Path | Summary | Auth Required |
|--------|------|---------|---------------|
| GET | `/v1/schema/{schema_id}/versions` | List schema versions | Yes |
| GET | `/v1/schema/{schema_id}/{version}` | Get schema details | Yes |

### System

| Method | Path | Summary | Auth Required |
|--------|------|---------|---------------|
| GET | `/health` | Health check | No |
| GET | `/metrics` | Service metrics (JSON) | No |
| GET | `/metrics/prometheus` | Prometheus text metrics | No |

---

## Validation Endpoints

### POST /v1/schema/validate

Validate a single payload against a GreenLang schema. The validation runs through seven phases:

1. **Parse** - Parse YAML/JSON payload
2. **Compile** - Compile schema to IR (cached)
3. **Structural** - Validate types and required fields
4. **Constraints** - Validate ranges, patterns, enums
5. **Units** - Validate unit dimensions and conversions
6. **Rules** - Evaluate cross-field business rules
7. **Lint** - Check for typos and deprecated fields

**Request Body:**

```json
{
  "schema_ref": {
    "schema_id": "emissions/activity",
    "version": "1.3.0"
  },
  "payload": {
    "activity_type": "stationary_combustion",
    "fuel_type": "natural_gas",
    "quantity": 1500.0,
    "unit": "m3",
    "reporting_period": "2025-Q4"
  },
  "options": {
    "strict_mode": true,
    "skip_lint": false,
    "normalize_units": true
  }
}
```

**Request Fields:**

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `schema_ref` | object | Yes | Schema reference (schema_id + version) |
| `payload` | object | Yes | Payload data to validate |
| `options` | object | No | Validation options (strict_mode, skip_lint, normalize_units) |

**Response (200 OK):**

```json
{
  "valid": true,
  "schema_ref": {
    "schema_id": "emissions/activity",
    "version": "1.3.0"
  },
  "schema_hash": "sha256:abc123...",
  "summary": {
    "total_findings": 1,
    "error_count": 0,
    "warning_count": 1,
    "info_count": 0
  },
  "findings": [
    {
      "severity": "warning",
      "phase": "lint",
      "path": "$.reporting_period",
      "message": "Consider using ISO 8601 date range format",
      "rule": "GLSCHEMA-LINT-008",
      "suggestion": "Use '2025-10-01/2025-12-31' instead of '2025-Q4'"
    }
  ],
  "normalized_payload": {
    "activity_type": "stationary_combustion",
    "fuel_type": "natural_gas",
    "quantity": 1500.0,
    "unit": "m3",
    "reporting_period": "2025-Q4"
  },
  "fix_suggestions": [],
  "timings_ms": {
    "parse": 0.5,
    "compile": 2.1,
    "structural": 1.2,
    "constraints": 0.8,
    "units": 0.4,
    "rules": 1.5,
    "lint": 0.9,
    "total": 7.4
  }
}
```

**Error Responses:**

| Status Code | Error Code | Description |
|-------------|------------|-------------|
| 400 | GLSCHEMA-API-002 | Invalid request parameters |
| 404 | GLSCHEMA-API-003 | Schema not found |
| 429 | GLSCHEMA-API-004 | Rate limit exceeded |

---

### POST /v1/schema/validate/batch

Validate multiple payloads against the same schema. The schema is compiled once and reused for all payloads, providing better performance than individual requests.

**Request Body:**

```json
{
  "schema_ref": {
    "schema_id": "emissions/activity",
    "version": "1.3.0"
  },
  "payloads": [
    {"activity_type": "stationary_combustion", "fuel_type": "natural_gas", "quantity": 1500.0, "unit": "m3"},
    {"activity_type": "mobile_combustion", "fuel_type": "diesel", "quantity": 500.0, "unit": "liters"},
    {"activity_type": "invalid_type", "fuel_type": "coal", "quantity": -100, "unit": "kg"}
  ],
  "options": {
    "strict_mode": true
  }
}
```

**Response (200 OK):**

```json
{
  "schema_ref": {
    "schema_id": "emissions/activity",
    "version": "1.3.0"
  },
  "schema_hash": "sha256:abc123...",
  "summary": {
    "total_items": 3,
    "valid_count": 2,
    "error_count": 1,
    "warning_count": 0,
    "total_findings": 2,
    "processing_time_ms": 15.8
  },
  "results": [
    {"index": 0, "id": null, "valid": true, "findings": [], "normalized_payload": {...}},
    {"index": 1, "id": null, "valid": true, "findings": [], "normalized_payload": {...}},
    {"index": 2, "id": null, "valid": false, "findings": [
      {"severity": "error", "phase": "constraints", "path": "$.activity_type", "message": "Value 'invalid_type' not in allowed enum values"},
      {"severity": "error", "phase": "constraints", "path": "$.quantity", "message": "Value must be >= 0"}
    ], "normalized_payload": null}
  ]
}
```

**Maximum batch size:** 1000 items (configurable). Returns 413 if exceeded.

---

## Compilation Endpoint

### POST /v1/schema/compile

Compile a schema to intermediate representation (IR). The compiled IR is cached for subsequent validation requests.

**Request Body:**

```json
{
  "schema_source": {
    "type": "object",
    "properties": {
      "activity_type": {"type": "string", "enum": ["stationary_combustion", "mobile_combustion"]},
      "quantity": {"type": "number", "minimum": 0},
      "unit": {"type": "string", "gl_unit_dimension": "volume"}
    },
    "required": ["activity_type", "quantity", "unit"]
  },
  "schema_ref": {
    "schema_id": "emissions/activity",
    "version": "1.4.0"
  },
  "include_ir": true
}
```

**Response (200 OK):**

```json
{
  "schema_hash": "sha256:def456...",
  "compiled_at": "2026-04-04T12:00:00Z",
  "warnings": [],
  "properties_count": 3,
  "rules_count": 0,
  "ir": {
    "schema_id": "emissions/activity",
    "version": "1.4.0",
    "schema_hash": "sha256:def456...",
    "properties": ["activity_type", "quantity", "unit"],
    "required_paths": ["activity_type", "quantity", "unit"]
  }
}
```

---

## Schema Registry Endpoints

### GET /v1/schema/{schema_id}/versions

List all available versions of a schema from the registry. The `schema_id` supports nested paths (e.g., `emissions/activity`).

**Response (200 OK):**

```json
{
  "schema_id": "emissions/activity",
  "versions": [
    {"version": "1.0.0", "created_at": "2025-06-01T00:00:00Z", "deprecated": true, "deprecated_message": "Use 1.3.0"},
    {"version": "1.1.0", "created_at": "2025-09-01T00:00:00Z", "deprecated": true, "deprecated_message": "Use 1.3.0"},
    {"version": "1.2.0", "created_at": "2025-12-01T00:00:00Z", "deprecated": false, "deprecated_message": null},
    {"version": "1.3.0", "created_at": "2026-03-01T00:00:00Z", "deprecated": false, "deprecated_message": null}
  ],
  "latest": "1.3.0"
}
```

---

### GET /v1/schema/{schema_id}/{version}

Get the full schema definition for a specific version.

**Response (200 OK):**

```json
{
  "schema_id": "emissions/activity",
  "version": "1.3.0",
  "schema_hash": "sha256:abc123...",
  "content": {
    "type": "object",
    "properties": {
      "activity_type": {"type": "string", "enum": ["stationary_combustion", "mobile_combustion", "process"]},
      "quantity": {"type": "number", "minimum": 0},
      "unit": {"type": "string", "gl_unit_dimension": "volume"}
    },
    "required": ["activity_type", "quantity", "unit"]
  },
  "created_at": "2026-03-01T00:00:00Z",
  "deprecated": false
}
```

---

## System Endpoints

### GET /health

Health check returning component-level status for the validator, compiler, and schema registry.

**Response (200 OK):**

```json
{
  "status": "healthy",
  "version": "1.0.0",
  "cache_size": 42,
  "uptime_seconds": 86400.5,
  "components": [
    {"name": "validator", "status": "healthy", "message": "Validator is operational"},
    {"name": "registry", "status": "healthy", "message": "Registry is configured"}
  ],
  "timestamp": "2026-04-04T12:00:00Z"
}
```

### GET /metrics

Service metrics in JSON format.

**Response (200 OK):**

```json
{
  "validations_total": 15420,
  "validations_success": 14800,
  "validations_failed": 620,
  "batch_validations_total": 350,
  "batch_items_total": 42000,
  "cache_hits": 14500,
  "cache_misses": 920,
  "cache_size": 42,
  "avg_validation_time_ms": 7.2,
  "p95_validation_time_ms": 15.8,
  "uptime_seconds": 86400.5
}
```

### GET /metrics/prometheus

Prometheus text exposition format metrics for scraping by Prometheus.

---

## Error Code Reference

| Code | Description |
|------|-------------|
| GLSCHEMA-API-001 | Batch size exceeds maximum |
| GLSCHEMA-API-002 | Invalid parameter value |
| GLSCHEMA-API-003 | Schema not found in registry |
| GLSCHEMA-API-004 | Rate limit exceeded |
| GLSCHEMA-API-005 | Schema version not found |
| GLSCHEMA-API-006 | Compilation failed |
| GLSCHEMA-API-099 | Internal server error |

## Authentication

Endpoints require an `X-API-Key` header. Distributed tracing is supported via `X-Trace-ID` header propagation.
