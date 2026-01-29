# GL-FOUND-X-002: REST API Reference

## Overview

The Schema Compiler & Validator REST API provides HTTP endpoints for validating payloads, compiling schemas, and managing schema versions. All endpoints follow RESTful conventions with JSON request/response bodies.

**Base URL:** `https://api.greenlang.io/v1/schema`

---

## Authentication

### API Key Authentication

```http
X-API-Key: your_api_key_here
```

### Bearer Token (OAuth2)

```http
Authorization: Bearer your_access_token
```

### Obtain Access Token

```http
POST /api/v1/auth/token
Content-Type: application/x-www-form-urlencoded

grant_type=client_credentials&
client_id=your_client_id&
client_secret=your_client_secret
```

**Response:**

```json
{
  "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "token_type": "Bearer",
  "expires_in": 3600
}
```

---

## Common Headers

| Header | Description | Required |
|--------|-------------|----------|
| `Content-Type` | `application/json` | Yes |
| `Authorization` | Bearer token or API key | Conditional |
| `X-API-Key` | API key (alternative to Bearer) | Conditional |
| `X-Trace-ID` | Request trace ID for debugging | No |

---

## Rate Limiting

- **Authenticated requests:** 1000 requests per minute
- **Unauthenticated requests:** 10 requests per minute

Rate limit headers are included in all responses:

```http
X-RateLimit-Limit: 1000
X-RateLimit-Remaining: 995
X-RateLimit-Reset: 1609459200
```

When rate limited, the API returns:

```http
HTTP/1.1 429 Too Many Requests
Retry-After: 42

{
  "error": "rate_limit_exceeded",
  "message": "Too many requests",
  "retry_after": 42,
  "trace_id": "abc123"
}
```

---

## Endpoints

### POST /v1/schema/validate

Validate a single payload against a GreenLang schema.

**Request:**

```http
POST /v1/schema/validate
Content-Type: application/json
Authorization: Bearer {access_token}
```

```json
{
  "payload": {
    "emissions": [
      {
        "fuel_type": "Natural Gas",
        "quantity": 1000,
        "unit": "therms",
        "co2e_emissions_kg": 5300.0,
        "scope": 1
      }
    ],
    "organization_id": "ORG-001"
  },
  "schema_ref": {
    "schema_id": "gl-emissions-input",
    "version": "1.0.0"
  },
  "options": {
    "profile": "standard",
    "normalize": true,
    "emit_patches": true,
    "max_errors": 100
  }
}
```

**Request Parameters:**

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `payload` | object/string | Yes | Data payload (object or YAML/JSON string) |
| `schema_ref` | object | Yes | Schema reference with ID and version |
| `schema_ref.schema_id` | string | Yes | Schema identifier (e.g., `gl-emissions-input`) |
| `schema_ref.version` | string | Yes | Schema version (e.g., `1.0.0`) |
| `options` | object | No | Validation options |
| `options.profile` | string | No | Validation profile: `strict`, `standard`, `permissive` |
| `options.normalize` | boolean | No | Return normalized payload (default: false) |
| `options.emit_patches` | boolean | No | Generate fix suggestions (default: true) |
| `options.max_errors` | integer | No | Maximum errors to report (default: 100) |

**Response (200 OK):**

```json
{
  "valid": true,
  "schema_ref": {
    "schema_id": "gl-emissions-input",
    "version": "1.0.0"
  },
  "schema_hash": "a1b2c3d4e5f6789012345678901234567890abcdef1234567890abcdef123456",
  "summary": {
    "valid": true,
    "error_count": 0,
    "warning_count": 0,
    "info_count": 0
  },
  "findings": [],
  "normalized_payload": {
    "emissions": [
      {
        "fuel_type": "Natural Gas",
        "quantity": 1000.0,
        "unit": "therms",
        "co2e_emissions_kg": 5300.0,
        "scope": 1
      }
    ],
    "organization_id": "ORG-001"
  },
  "fix_suggestions": [],
  "timings_ms": {
    "total": 8.5,
    "parse": 1.2,
    "compile": 0.5,
    "validate": 6.3,
    "normalize": 0.5
  }
}
```

**Response with Errors (200 OK, valid=false):**

```json
{
  "valid": false,
  "schema_ref": {
    "schema_id": "gl-emissions-input",
    "version": "1.0.0"
  },
  "schema_hash": "a1b2c3d4...",
  "summary": {
    "valid": false,
    "error_count": 2,
    "warning_count": 1,
    "info_count": 0
  },
  "findings": [
    {
      "code": "GLSCHEMA-E100",
      "severity": "error",
      "path": "emissions.0.quantity",
      "message": "Type mismatch: expected 'number', got 'string'",
      "expected": "number",
      "actual": "string",
      "value": "1000"
    },
    {
      "code": "GLSCHEMA-E101",
      "severity": "error",
      "path": "emissions.0.scope",
      "message": "Invalid enum value: 4 is not one of [1, 2, 3]",
      "expected": [1, 2, 3],
      "actual": 4
    },
    {
      "code": "GLSCHEMA-W300",
      "severity": "warning",
      "path": "organization_id",
      "message": "Field 'organization_id' is recommended but missing"
    }
  ],
  "fix_suggestions": [
    {
      "finding_index": 0,
      "patch_level": "safe",
      "description": "Convert string to number",
      "code_snippet": "data[\"emissions\"][0][\"quantity\"] = 1000",
      "auto_fixable": true
    },
    {
      "finding_index": 1,
      "patch_level": "needs_review",
      "description": "Change scope to valid value",
      "code_snippet": "data[\"emissions\"][0][\"scope\"] = 1",
      "auto_fixable": false
    }
  ],
  "timings_ms": {
    "total": 12.3
  }
}
```

**Error Responses:**

| Status | Description |
|--------|-------------|
| 400 | Invalid request (malformed JSON, missing required fields) |
| 404 | Schema not found |
| 429 | Rate limit exceeded |
| 500 | Internal server error |

---

### POST /v1/schema/validate/batch

Validate multiple payloads against the same schema in a single request.

**Request:**

```http
POST /v1/schema/validate/batch
Content-Type: application/json
Authorization: Bearer {access_token}
```

```json
{
  "payloads": [
    {
      "id": "record-001",
      "data": {
        "emissions": [{"fuel_type": "Gas", "co2e_emissions_kg": 100}]
      }
    },
    {
      "id": "record-002",
      "data": {
        "emissions": [{"fuel_type": "Electric", "co2e_emissions_kg": 50}]
      }
    }
  ],
  "schema_ref": {
    "schema_id": "gl-emissions-input",
    "version": "1.0.0"
  },
  "options": {
    "profile": "standard",
    "normalize": false
  }
}
```

**Request Parameters:**

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `payloads` | array | Yes | Array of payload objects |
| `payloads[].id` | string | No | Optional identifier for the payload |
| `payloads[].data` | object | Yes | The payload data to validate |
| `schema_ref` | object | Yes | Schema reference |
| `options` | object | No | Validation options (applied to all payloads) |

**Limits:**
- Maximum batch size: 1000 items
- Maximum total request size: 10 MB

**Response (200 OK):**

```json
{
  "schema_ref": {
    "schema_id": "gl-emissions-input",
    "version": "1.0.0"
  },
  "schema_hash": "a1b2c3d4...",
  "summary": {
    "total_items": 2,
    "valid_count": 2,
    "error_count": 0,
    "warning_count": 0,
    "total_findings": 0,
    "processing_time_ms": 15.5
  },
  "results": [
    {
      "index": 0,
      "id": "record-001",
      "valid": true,
      "findings": []
    },
    {
      "index": 1,
      "id": "record-002",
      "valid": true,
      "findings": []
    }
  ]
}
```

**Error Responses:**

| Status | Description |
|--------|-------------|
| 400 | Invalid request |
| 404 | Schema not found |
| 413 | Batch too large (exceeds 1000 items) |
| 429 | Rate limit exceeded |
| 500 | Internal server error |

---

### POST /v1/schema/compile

Compile a schema to intermediate representation (IR).

**Request:**

```http
POST /v1/schema/compile
Content-Type: application/json
Authorization: Bearer {access_token}
```

```json
{
  "schema_source": {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "type": "object",
    "properties": {
      "name": {"type": "string"},
      "value": {"type": "number", "minimum": 0}
    },
    "required": ["name", "value"]
  },
  "schema_ref": {
    "schema_id": "custom-schema",
    "version": "1.0.0"
  },
  "include_ir": true
}
```

**Request Parameters:**

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `schema_source` | object/string | Yes | Schema definition (object or YAML/JSON string) |
| `schema_ref` | object | No | Optional schema reference for identification |
| `include_ir` | boolean | No | Include full IR in response (default: false) |

**Response (200 OK):**

```json
{
  "schema_hash": "b2c3d4e5f6789012345678901234567890abcdef1234567890abcdef1234567",
  "compiled_at": "2026-01-29T10:30:00Z",
  "warnings": [],
  "properties_count": 2,
  "rules_count": 1,
  "ir": {
    "schema_id": "custom-schema",
    "version": "1.0.0",
    "schema_hash": "b2c3d4e5...",
    "properties": ["name", "value"],
    "required_paths": ["name", "value"]
  }
}
```

---

### GET /v1/schema/{schema_id}/versions

List all available versions of a schema.

**Request:**

```http
GET /v1/schema/emissions/activity/versions
Authorization: Bearer {access_token}
```

**Response (200 OK):**

```json
{
  "schema_id": "emissions/activity",
  "versions": [
    {
      "version": "1.0.0",
      "created_at": "2025-01-15T00:00:00Z",
      "deprecated": true,
      "deprecated_message": "Use version 1.3.0 or later"
    },
    {
      "version": "1.2.0",
      "created_at": "2025-06-01T00:00:00Z",
      "deprecated": true,
      "deprecated_message": "Use version 1.3.0 or later"
    },
    {
      "version": "1.3.0",
      "created_at": "2025-10-01T00:00:00Z",
      "deprecated": false
    }
  ],
  "latest": "1.3.0"
}
```

---

### GET /v1/schema/{schema_id}/{version}

Get the full schema definition for a specific version.

**Request:**

```http
GET /v1/schema/emissions/activity/1.3.0
Authorization: Bearer {access_token}
```

**Response (200 OK):**

```json
{
  "schema_id": "emissions/activity",
  "version": "1.3.0",
  "schema_hash": "c3d4e5f6...",
  "content": {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "type": "object",
    "properties": {
      "emissions": {
        "type": "array",
        "items": {
          "type": "object",
          "properties": {
            "fuel_type": {"type": "string"},
            "quantity": {"type": "number", "minimum": 0},
            "unit": {"type": "string"},
            "co2e_emissions_kg": {"type": "number"},
            "scope": {"type": "integer", "enum": [1, 2, 3]}
          },
          "required": ["fuel_type", "co2e_emissions_kg"]
        }
      },
      "organization_id": {"type": "string"}
    },
    "required": ["emissions"]
  },
  "created_at": "2025-10-01T00:00:00Z",
  "deprecated": false
}
```

---

### GET /health

Health check endpoint for load balancers and monitoring.

**Request:**

```http
GET /health
```

**Response (200 OK):**

```json
{
  "status": "healthy",
  "version": "1.0.0",
  "cache_size": 42,
  "uptime_seconds": 86400.5,
  "components": [
    {
      "name": "validator",
      "status": "healthy",
      "message": "Validator is operational"
    },
    {
      "name": "registry",
      "status": "healthy",
      "message": "Registry is configured"
    }
  ],
  "timestamp": "2026-01-29T10:30:00Z"
}
```

---

### GET /metrics

Get service metrics for monitoring.

**Request:**

```http
GET /metrics
```

**Response (200 OK):**

```json
{
  "validations_total": 1250000,
  "validations_success": 1200000,
  "validations_failed": 50000,
  "batch_validations_total": 5000,
  "batch_items_total": 500000,
  "cache_hits": 1150000,
  "cache_misses": 100000,
  "cache_size": 42,
  "avg_validation_time_ms": 8.5,
  "p95_validation_time_ms": 25.0,
  "uptime_seconds": 86400.5
}
```

---

### GET /metrics/prometheus

Get metrics in Prometheus text format.

**Request:**

```http
GET /metrics/prometheus
```

**Response (200 OK):**

```text
# HELP glschema_validations_total Total number of validation requests
# TYPE glschema_validations_total counter
glschema_validations_total 1250000

# HELP glschema_validations_success Number of successful validations
# TYPE glschema_validations_success counter
glschema_validations_success 1200000

# HELP glschema_validations_failed Number of failed validations
# TYPE glschema_validations_failed counter
glschema_validations_failed 50000

# HELP glschema_cache_hits Schema cache hits
# TYPE glschema_cache_hits counter
glschema_cache_hits 1150000

# HELP glschema_validation_time_avg_ms Average validation time in milliseconds
# TYPE glschema_validation_time_avg_ms gauge
glschema_validation_time_avg_ms 8.5

# HELP glschema_validation_time_p95_ms 95th percentile validation time
# TYPE glschema_validation_time_p95_ms gauge
glschema_validation_time_p95_ms 25.0
```

---

## Error Response Format

All error responses follow this format:

```json
{
  "error": "error_type",
  "message": "Human-readable error message",
  "details": [
    {
      "code": "GLSCHEMA-API-E001",
      "message": "Specific error detail",
      "field": "affected_field"
    }
  ],
  "trace_id": "abc123-def456",
  "timestamp": "2026-01-29T10:30:00Z"
}
```

---

## Code Examples

### Python (requests)

```python
import requests

API_BASE = "https://api.greenlang.io/v1/schema"
ACCESS_TOKEN = "your_access_token"

headers = {
    "Authorization": f"Bearer {ACCESS_TOKEN}",
    "Content-Type": "application/json"
}

# Validate a payload
response = requests.post(
    f"{API_BASE}/validate",
    headers=headers,
    json={
        "payload": {
            "emissions": [
                {"fuel_type": "Gas", "co2e_emissions_kg": 100}
            ]
        },
        "schema_ref": {
            "schema_id": "gl-emissions-input",
            "version": "1.0.0"
        }
    }
)

result = response.json()
print(f"Valid: {result['valid']}")
print(f"Errors: {result['summary']['error_count']}")
```

### JavaScript (Node.js)

```javascript
const axios = require('axios');

const API_BASE = 'https://api.greenlang.io/v1/schema';
const ACCESS_TOKEN = 'your_access_token';

async function validatePayload(payload, schemaId, version) {
  const response = await axios.post(
    `${API_BASE}/validate`,
    {
      payload: payload,
      schema_ref: {
        schema_id: schemaId,
        version: version
      }
    },
    {
      headers: {
        'Authorization': `Bearer ${ACCESS_TOKEN}`,
        'Content-Type': 'application/json'
      }
    }
  );

  return response.data;
}

// Usage
const result = await validatePayload(
  { emissions: [{ fuel_type: 'Gas', co2e_emissions_kg: 100 }] },
  'gl-emissions-input',
  '1.0.0'
);

console.log(`Valid: ${result.valid}`);
```

### cURL

```bash
# Single validation
curl -X POST "https://api.greenlang.io/v1/schema/validate" \
  -H "Authorization: Bearer $ACCESS_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "payload": {
      "emissions": [{"fuel_type": "Gas", "co2e_emissions_kg": 100}]
    },
    "schema_ref": {
      "schema_id": "gl-emissions-input",
      "version": "1.0.0"
    }
  }'

# Batch validation
curl -X POST "https://api.greenlang.io/v1/schema/validate/batch" \
  -H "Authorization: Bearer $ACCESS_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "payloads": [
      {"id": "1", "data": {"emissions": [{"fuel_type": "Gas", "co2e_emissions_kg": 100}]}},
      {"id": "2", "data": {"emissions": [{"fuel_type": "Electric", "co2e_emissions_kg": 50}]}}
    ],
    "schema_ref": {"schema_id": "gl-emissions-input", "version": "1.0.0"}
  }'

# Health check
curl "https://api.greenlang.io/health"
```

---

## OpenAPI Specification

The full OpenAPI 3.0 specification is available at:

```
https://api.greenlang.io/openapi.json
https://api.greenlang.io/docs  # Swagger UI
https://api.greenlang.io/redoc # ReDoc
```

---

## See Also

- [CLI Reference](cli.md)
- [Python SDK Guide](sdk.md)
- [Error Codes Reference](error_codes.md)
