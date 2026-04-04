# Schema Migration Agent API Reference

**Agent:** AGENT-DATA-017 (GL-DATA-X-020)
**Prefix:** `/api/v1/schema-migration`
**Source:** `greenlang/agents/data/schema_migration/api/router.py`
**Status:** Production Ready

## Overview

The Schema Migration Agent provides 20 REST API endpoints for managing schema evolution across GreenLang data pipelines. Capabilities include schema registration with namespace and type classification, automatic SemVer versioning, field-level change detection between versions, backward/forward/full compatibility checking, migration plan generation with ordered transformation steps, plan execution with dry-run support, checkpoint-based rollback, and full pipeline orchestration.

Supported schema types: `json_schema`, `avro`, `protobuf`.

---

## Endpoint Summary

| # | Method | Path | Summary | Auth |
|---|--------|------|---------|------|
| 1 | POST | `/schemas` | Register a new schema | Yes |
| 2 | GET | `/schemas` | List registered schemas | Yes |
| 3 | GET | `/schemas/{schema_id}` | Get schema details | Yes |
| 4 | PUT | `/schemas/{schema_id}` | Update schema metadata | Yes |
| 5 | DELETE | `/schemas/{schema_id}` | Deregister schema (soft delete) | Yes |
| 6 | POST | `/versions` | Create a new schema version | Yes |
| 7 | GET | `/versions` | List schema versions | Yes |
| 8 | GET | `/versions/{version_id}` | Get version details | Yes |
| 9 | POST | `/changes/detect` | Detect changes between versions | Yes |
| 10 | GET | `/changes` | List detected changes | Yes |
| 11 | POST | `/compatibility/check` | Check compatibility between versions | Yes |
| 12 | GET | `/compatibility` | List compatibility check results | Yes |
| 13 | POST | `/plans` | Generate migration plan | Yes |
| 14 | GET | `/plans/{plan_id}` | Get migration plan details | Yes |
| 15 | POST | `/execute` | Execute migration plan | Yes |
| 16 | GET | `/executions/{execution_id}` | Get execution status | Yes |
| 17 | POST | `/rollback/{execution_id}` | Rollback migration execution | Yes |
| 18 | POST | `/pipeline` | Run full migration pipeline | Yes |
| 19 | GET | `/health` | Health check | No |
| 20 | GET | `/stats` | Service statistics | Yes |

---

## Key Endpoints

### 1. Register Schema

Register a new schema definition in the migration registry.

```http
POST /api/v1/schema-migration/schemas
```

**Request Body:**

```json
{
  "namespace": "greenlang.emissions.scope1",
  "name": "Stationary Combustion Record",
  "schema_type": "json_schema",
  "definition": {
    "type": "object",
    "properties": {
      "facility_id": {"type": "string"},
      "fuel_type": {"type": "string"},
      "quantity": {"type": "number"},
      "unit": {"type": "string"},
      "emission_factor": {"type": "number"},
      "emissions_tco2e": {"type": "number"}
    },
    "required": ["facility_id", "fuel_type", "quantity", "unit"]
  },
  "owner": "mrv-team",
  "tags": ["scope1", "stationary", "ghg"],
  "description": "Schema for stationary combustion emission records"
}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `namespace` | string | Yes | Dot/hyphen-separated namespace |
| `name` | string | Yes | Human-readable schema name |
| `schema_type` | string | No | `json_schema`, `avro`, `protobuf` (default: `json_schema`) |
| `definition` | object | Yes | Schema definition as JSON |
| `owner` | string | No | Responsible team or service |
| `tags` | array | No | Discovery and filtering tags |
| `description` | string | No | Human-readable description |

**Response (201):** Complete schema record with generated `schema_id`, initial `active` status, and creation timestamp.

**Status Codes:** `201` Created | `400` Validation error | `500` Server error

---

### 6. Create Schema Version

Create a new version of an existing schema. The version string is automatically determined by SemVer rules based on the severity of detected changes.

```http
POST /api/v1/schema-migration/versions
```

**Request Body:**

```json
{
  "schema_id": "sch_abc123",
  "definition": {
    "type": "object",
    "properties": {
      "facility_id": {"type": "string"},
      "fuel_type": {"type": "string"},
      "quantity": {"type": "number"},
      "unit": {"type": "string"},
      "emission_factor": {"type": "number"},
      "emissions_tco2e": {"type": "number"},
      "reporting_period": {"type": "string"}
    },
    "required": ["facility_id", "fuel_type", "quantity", "unit"]
  },
  "changelog_note": "Added optional reporting_period field for temporal context"
}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `schema_id` | string | Yes | Parent schema definition ID |
| `definition` | object | Yes | New schema definition at this version |
| `changelog_note` | string | No | Description of what changed |

**Response (201):** Version record with auto-assigned SemVer string, definition snapshot, and provenance data.

---

### 9. Detect Changes Between Versions

Detect structural changes between two schema versions at the field level.

```http
POST /api/v1/schema-migration/changes/detect
```

**Request Body:**

```json
{
  "source_version_id": "ver_001",
  "target_version_id": "ver_002"
}
```

**Response (201):** List of changes including field additions, removals, renames, type changes, and constraint modifications, each with severity classification.

---

### 11. Check Compatibility

Check backward, forward, or full compatibility between two schema versions.

```http
POST /api/v1/schema-migration/compatibility/check
```

**Request Body:**

```json
{
  "source_version_id": "ver_001",
  "target_version_id": "ver_002",
  "level": "backward"
}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `source_version_id` | string | Yes | Source (older) schema version ID |
| `target_version_id` | string | Yes | Target (newer) schema version ID |
| `level` | string | No | `backward`, `forward`, `full` (default: `backward`) |

**Response (201):** Compatibility result with determination (backward, forward, full, breaking), list of issues, and recommendations.

---

### 13. Generate Migration Plan

Generate an ordered migration plan with transformation steps for migrating records from one schema to another.

```http
POST /api/v1/schema-migration/plans
```

**Request Body:**

```json
{
  "source_schema_id": "sch_abc123",
  "target_schema_id": "sch_abc123",
  "source_version": "1.0.0",
  "target_version": "2.0.0"
}
```

**Response (201):** Migration plan with ordered transformation steps, estimated effort, affected record counts, and status.

---

### 15. Execute Migration Plan

Execute a validated migration plan. Supports dry-run mode for pre-execution validation.

```http
POST /api/v1/schema-migration/execute
```

**Request Body:**

```json
{
  "plan_id": "plan_abc123",
  "dry_run": true
}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `plan_id` | string | Yes | Migration plan to execute |
| `dry_run` | boolean | No | Validate without committing changes (default: `false`) |

**Response (201):** Execution record with step progress, record counts, checkpoint data, and error details (if any).

---

### 17. Rollback Migration

Rollback a migration execution fully or partially to a specified checkpoint.

```http
POST /api/v1/schema-migration/rollback/{execution_id}
```

**Request Body:**

```json
{
  "to_checkpoint": 3
}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `to_checkpoint` | integer | No | Checkpoint step number to rollback to (full rollback if omitted) |

**Response (201):** Rollback record with reverted records count and status.

**Status Codes:** `201` Success | `400` Validation error or execution not rollbackable

---

### 18. Run Full Pipeline

Run the complete end-to-end migration pipeline: detect changes, check compatibility, generate plan, optional dry-run, and execute.

```http
POST /api/v1/schema-migration/pipeline
```

**Request Body:**

```json
{
  "source_schema_id": "sch_abc123",
  "target_schema_id": "sch_abc123",
  "source_version": "1.0.0",
  "target_version": "2.0.0",
  "skip_compatibility": false,
  "skip_dry_run": false
}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `source_schema_id` | string | Yes | Source schema definition ID |
| `target_schema_id` | string | Yes | Target schema definition ID |
| `source_version` | string | No | Source SemVer string |
| `target_version` | string | No | Target SemVer string |
| `skip_compatibility` | boolean | No | Skip compatibility analysis (default: `false`) |
| `skip_dry_run` | boolean | No | Skip dry-run validation (default: `false`) |

**Response (201):** Aggregated pipeline result with outputs from each stage (changes, compatibility, plan, execution).

---

## Filtering and Pagination

List endpoints support filtering and pagination via query parameters:

**Schemas:** `namespace`, `schema_type`, `status`, `owner`, `tag`
**Versions:** `schema_id`, `version_range` (SemVer range, e.g., `>=1.0.0 <2.0.0`), `deprecated`
**Changes:** `schema_id`, `severity` (`breaking`, `non_breaking`, `cosmetic`), `change_type` (`added`, `removed`, `renamed`, `retyped`)
**Compatibility:** `schema_id`, `result` (`backward`, `forward`, `full`, `breaking`)

All list endpoints accept `limit` (1-1000, default 50) and `offset` (default 0).

---

## Error Responses

All error responses follow a standard format:

```json
{
  "detail": "Descriptive error message"
}
```

| Status Code | Description |
|-------------|-------------|
| 400 | Bad Request -- validation error, no update fields provided, or invalid parameters |
| 404 | Not Found -- schema, version, plan, or execution not found |
| 500 | Internal Server Error |
| 503 | Service Unavailable -- schema migration service not configured |
