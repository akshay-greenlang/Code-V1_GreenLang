# CBAM Pack MVP -- Enterprise Shell API Reference

**Source:** `cbam-pack-mvp/src/cbam_pack/web/app.py`
**Version:** Matches `cbam_pack.__version__`
**Title:** GreenLang CBAM Pack
**Factory function:** `create_app() -> FastAPI`

The CBAM Pack MVP serves as the unified **GreenLang Enterprise Shell**, exposing a single FastAPI application that provides workspace pages, multi-app pipeline execution, run management, governance inspection, and admin tooling.

---

## Table of Contents

1. [Workspace Pages (HTML)](#workspace-pages)
2. [Run Management API](#run-management-api)
3. [App Pipeline Execution](#app-pipeline-execution)
4. [CBAM-Specific Processing](#cbam-specific-processing)
5. [Governance API](#governance-api)
6. [Admin API](#admin-api)
7. [Shell Chrome API](#shell-chrome-api)
8. [Telemetry and Streaming](#telemetry-and-streaming)
9. [Artifact Download](#artifact-download)
10. [Health](#health)

---

## Authentication

All `/api/` endpoints enforce API-key authentication when the `CBAM_API_KEY` environment variable is set.

**Header:** `x-api-key: <value of CBAM_API_KEY>`

When `CBAM_API_KEY` is not set, endpoints are accessible without authentication (development mode).

**Rate limiting:** 60 requests per minute per client IP on all `/api/` endpoints.

**Upload limit:** 10 MB maximum per file upload.

---

## Workspace Pages

HTML pages served by the enterprise shell. If the React frontend build (`frontend/dist/`) is available, the compiled SPA is served; otherwise, server-rendered fallback HTML is returned.

| Method | Path | Summary |
|--------|------|---------|
| GET | `/` | Home page |
| GET | `/apps` | Multi-app shell home |
| GET | `/apps/cbam` | CBAM workspace |
| GET | `/apps/csrd` | CSRD workspace |
| GET | `/apps/vcci` | VCCI workspace |
| GET | `/apps/eudr` | EUDR workspace |
| GET | `/apps/ghg` | GHG workspace |
| GET | `/apps/iso14064` | ISO 14064 workspace |
| GET | `/apps/sb253` | SB 253 workspace |
| GET | `/apps/taxonomy` | Taxonomy workspace |
| GET | `/runs` | Run Center page |
| GET | `/governance` | Governance Center page |
| GET | `/admin` | Admin Center page |
| GET | `/ui.js` | Shared workspace UI script |

All workspace pages return `text/html` responses.

---

## Run Management API

### List Runs

```
GET /api/v1/runs
```

List all run records across apps. Supports filtering and search.

**Auth:** API Key (when configured)

**Query Parameters:**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `app_id` | string | No | Filter by app ID (case-insensitive). Example: `cbam`, `csrd` |
| `status` | string | No | Filter by `run_state` or `status_chip`. Values: `completed`, `failed`, `blocked`, `partial_success`, `PASS`, `WARN`, `FAIL` |
| `since_ts` | float | No | Minimum `created_at_ts` (Unix seconds) |
| `until_ts` | float | No | Maximum `created_at_ts` (Unix seconds) |
| `q` | string | No | Substring match on `run_id` or `app_id` |

**Response (200):**

```json
{
  "runs": [
    {
      "run_id": "a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4",
      "created_at_ts": 1712160000.0,
      "app_id": "cbam",
      "status": "completed",
      "execution_mode": "native",
      "success": true,
      "artifacts": ["cbam_report.xml", "provenance.json"],
      "can_export": true,
      "run_state": "completed",
      "lifecycle_phase": "completed",
      "status_chip": "PASS",
      "error_envelope": null
    }
  ]
}
```

**Response Fields:**

| Field | Type | Description |
|-------|------|-------------|
| `run_id` | string | 32-character hex UUID |
| `created_at_ts` | float | Unix timestamp of run creation |
| `app_id` | string | Application identifier (cbam, csrd, vcci, eudr, ghg, iso14064, sb253, taxonomy) |
| `status` | string | Raw status: `completed`, `failed` |
| `execution_mode` | string | `native`, `fallback`, or `unknown` |
| `success` | boolean | Whether the pipeline succeeded |
| `artifacts` | array[string] | List of output filenames |
| `can_export` | boolean | Whether artifacts can be downloaded |
| `run_state` | string | Derived lifecycle state: `completed`, `partial_success`, `failed`, `blocked` |
| `lifecycle_phase` | string | Always `completed` for finished runs |
| `status_chip` | string | UI chip label: `PASS`, `WARN`, `FAIL` |
| `error_envelope` | object or null | Structured error details (see below) |

**Error Envelope (when present):**

```json
{
  "title": "Run encountered errors",
  "message": "Primary error message",
  "details": ["Additional detail 1", "Additional detail 2"]
}
```

---

### Stream Runs (SSE)

```
GET /api/v1/stream/runs
```

Server-Sent Events endpoint for live run count updates.

**Auth:** API Key (when configured)

**Response:** `text/event-stream`

Each event (every 2 seconds):
```
data: {"status": "live", "runs": 5, "timestamp": "2026-04-04T12:00:00Z"}
```

---

## App Pipeline Execution

All pipeline execution endpoints accept a file upload and return a standardized run response.

### CBAM Run

```
POST /api/v1/apps/cbam/run
Content-Type: multipart/form-data
```

**Request Body:**

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `input_file` | file | Yes | CSV or Excel shipment data file (max 10 MB) |

**Response (200):**

```json
{
  "run_id": "a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4",
  "app_id": "cbam",
  "success": true,
  "status": "completed",
  "run_state": "completed",
  "lifecycle_phase": "completed",
  "status_chip": "PASS",
  "execution_mode": "native",
  "artifacts": ["cbam_report.xml", "provenance.json", "gap_report.md"],
  "can_export": true,
  "warnings": [],
  "errors": [],
  "summary": {
    "total_shipments": 150,
    "total_emissions_tco2": 12345.67,
    "xsd_valid": true,
    "policy_status": "PASS"
  },
  "error_envelope": null
}
```

### CSRD Run

```
POST /api/v1/apps/csrd/run
Content-Type: multipart/form-data
```

Executes the CSRD/ESRS reporting pipeline. Requires `greenlang.v1.backends.run_csrd_backend` to be available.

**Request Body:** `input_file` (file upload, max 10 MB)

**Response:** Same structure as CBAM Run. Summary sourced from `esrs_report.json`.

### VCCI Run

```
POST /api/v1/apps/vcci/run
Content-Type: multipart/form-data
```

Executes the VCCI Scope 3 pipeline. Requires `greenlang.v1.backends.run_vcci_backend`.

**Request Body:** `input_file` (file upload, max 10 MB)

**Response:** Same structure. Summary sourced from `scope3_inventory.json`.

### EUDR Run

```
POST /api/v1/apps/eudr/run
Content-Type: multipart/form-data
```

Executes the EUDR compliance pipeline via V2 profile backend.

**Request Body:** `input_file` (file upload, max 10 MB)

**Response:** Same structure. Summary sourced from `due_diligence_statement.json`.

### GHG Run

```
POST /api/v1/apps/ghg/run
Content-Type: multipart/form-data
```

Executes the GHG corporate inventory pipeline.

**Request Body:** `input_file` (file upload, max 10 MB)

**Response:** Same structure. Summary sourced from `ghg_inventory.json`.

### ISO 14064 Run

```
POST /api/v1/apps/iso14064/run
Content-Type: multipart/form-data
```

**Request Body:** `input_file` (file upload, max 10 MB)

**Response:** Same structure. Summary sourced from `iso14064_verification_report.json`.

### SB 253 Run

```
POST /api/v1/apps/sb253/run
Content-Type: multipart/form-data
```

**Request Body:** `input_file` (file upload, max 10 MB)

**Response:** Same structure. Summary sourced from `sb253_disclosure.json`.

### Taxonomy Run

```
POST /api/v1/apps/taxonomy/run
Content-Type: multipart/form-data
```

**Request Body:** `input_file` (file upload, max 10 MB)

**Response:** Same structure. Summary sourced from `taxonomy_alignment.json`.

### Demo Runs

Each application also supports a demo run endpoint that uses bundled sample input data. No file upload is required.

| Method | Path | Description |
|--------|------|-------------|
| POST | `/api/v1/apps/cbam/demo-run` | CBAM demo pipeline |
| POST | `/api/v1/apps/csrd/demo-run` | CSRD demo pipeline |
| POST | `/api/v1/apps/vcci/demo-run` | VCCI demo pipeline |
| POST | `/api/v1/apps/eudr/demo-run` | EUDR demo pipeline |
| POST | `/api/v1/apps/ghg/demo-run` | GHG demo pipeline |
| POST | `/api/v1/apps/iso14064/demo-run` | ISO 14064 demo pipeline |
| POST | `/api/v1/apps/sb253/demo-run` | SB 253 demo pipeline |
| POST | `/api/v1/apps/taxonomy/demo-run` | Taxonomy demo pipeline |

**Response:** Same structure as the full run endpoints.

**Error (404):** Returned when demo input data is not found.

**Error (503):** Returned when the backend integration is not available.

---

## CBAM-Specific Processing

### Process Shipment Data (Legacy)

```
POST /api/process
Content-Type: multipart/form-data
```

Original CBAM processing endpoint. Accepts shipment data, runs the 3-agent pipeline (Intake, Calculator, Reporter), and returns full results.

**Request Body:**

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `input_file` | file | Yes | CSV shipment data |
| `config_yaml` | string | No | Optional YAML configuration override |

**Response (200):** Run result with XSD validation status, policy PASS/WARN/FAIL status, row-level drilldown, gap report, and evidence folder metadata.

### Preview Config

```
POST /api/preview-config
Content-Type: application/json
```

Preview a CBAM pipeline configuration without executing.

**Request Body:**

```json
{
  "config_yaml": "reporting_period: Q4-2025\nregion: EU"
}
```

**Response (200):** Parsed and validated configuration object.

---

## Artifact Download

### Download Single Artifact

```
GET /api/v1/runs/{run_id}/artifacts/{artifact_path}
```

Download a specific output artifact from a completed run.

**Path Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `run_id` | string | 32-char hex run ID |
| `artifact_path` | string | Relative path to the artifact file |

**Response:** File download with appropriate MIME type.

**Error (404):** Run or artifact not found.

### Download Artifact Bundle

```
GET /api/v1/runs/{run_id}/bundle
```

Download all artifacts from a run as a ZIP archive.

**Response:** `application/zip` file containing all output artifacts.

### Download Session Output (Legacy)

```
GET /api/download/{session_id}
```

Download all session output as a ZIP.

```
GET /api/download/{session_id}/{filename}
```

Download a specific file from a session.

---

## Governance API

### List Pack Tiers

```
GET /api/v1/governance/pack-tiers
```

Returns the pack tier registry from `greenlang/ecosystem/packs/v2_tier_registry.yaml`.

**Response (200):**

```json
{
  "packs": [
    {
      "pack_slug": "cbam-importer",
      "app_id": "cbam",
      "tier": "pilot",
      "owner_team": "team-cbam",
      "promotion_status": "promoted"
    }
  ]
}
```

### List Agents

```
GET /api/v1/governance/agents
```

Returns the agent registry from `greenlang/agents/v2_agent_registry.yaml`.

**Response (200):**

```json
{
  "agents": [
    {
      "agent_id": "gl_eudr_001",
      "owner_team": "team-eudr",
      "state": "active",
      "current_version": "1.0.0",
      "replacement_agent_id": null
    }
  ]
}
```

### List Policy Bundles

```
GET /api/v1/governance/policy-bundles
```

Returns OPA/Rego policy bundles from `greenlang/governance/policy/bundles/`.

**Response (200):**

```json
{
  "bundles": [
    {"bundle": "cbam_export_policy.rego", "bytes": 2048},
    {"bundle": "eudr_compliance_policy.rego", "bytes": 3120}
  ]
}
```

---

## Admin API

### Release Train Evidence

```
GET /api/v1/admin/release-train
```

Returns release train test evidence from `docs/v2/RELEASE_TRAIN_LOCAL_EVIDENCE.json`.

**Response (200):**

```json
{
  "available": true,
  "evidence": { "cycles": [...] },
  "cycle_summary": [
    {
      "cycle": "2026-Q1",
      "all_passed": true,
      "executed_at_utc": "2026-03-31T10:00:00Z"
    }
  ]
}
```

### List Connectors

```
GET /api/v1/admin/connectors
```

Returns the connector registry from `applications/connectors/v2_connector_registry.yaml`.

**Response (200):**

```json
{
  "registry_version": "2.0.0",
  "connectors": [
    {
      "connector_id": "sap_erp_v2",
      "app_id": "ghg",
      "owner_team": "team-data",
      "support_channel": "#data-connectors",
      "read_timeout_ms": 5000,
      "circuit_open_s": 30,
      "operational_status": "ok",
      "incident_summary": null,
      "slo_target_availability_pct": 99.5,
      "runbook_url": "https://wiki.greenlang.io/runbooks/sap-erp"
    }
  ]
}
```

### Connector Health Probes

```
GET /api/v1/admin/connectors/health
```

Runs lightweight health probes against registered connectors and caches results for shell chrome merge.

**Response (200):**

```json
{
  "updated_at_utc": "2026-04-04T12:00:00Z",
  "probes": [
    {
      "connector_id": "sap_erp_v2",
      "app_id": "ghg",
      "ok": true,
      "latency_ms": 2,
      "checked_at_utc": "2026-04-04T12:00:00Z",
      "registry_status": "ok"
    }
  ]
}
```

---

## Shell Chrome API

### Chrome Context

```
GET /api/v1/shell/chrome-context
```

Aggregated compliance rail and connector incident signals for the enterprise shell chrome (sidebar rail and incident banner).

**Response (200):**

```json
{
  "compliance_rail": {
    "managed_pack_count": 8,
    "policy_bundle_count": 12,
    "deprecated_agent_count": 2
  },
  "connector_incidents": [
    {
      "connector_id": "sap_erp_v2",
      "app_id": "ghg",
      "severity": "warning",
      "message": "Connector marked degraded in v2_connector_registry.yaml"
    }
  ],
  "connector_probe_meta": {
    "probe_count": 5,
    "last_refresh_utc": "2026-04-04T12:00:00Z"
  }
}
```

---

## Telemetry and Streaming

### Client Error Telemetry

```
POST /api/telemetry/client-error
Content-Type: application/json
```

Non-blocking endpoint that captures frontend error telemetry. Stores the last 250 error reports in memory.

**Request Body:** Any JSON object describing the error.

**Response (200):**

```json
{"ok": true}
```

---

## Health

### Health Check

```
GET /health
```

Basic liveness check.

**Response (200):**

```json
{
  "status": "healthy",
  "version": "1.0.0",
  "service": "GreenLang CBAM Pack"
}
```

---

## Error Responses

All API errors return JSON with sanitized messages (local filesystem paths are redacted):

```json
{
  "detail": "Input file exceeds upload size limit"
}
```

| Status | Condition |
|--------|-----------|
| 400 | Invalid upload filename, bad request body |
| 401 | Missing or incorrect `x-api-key` header |
| 404 | Run ID not found, demo input not found, registry file missing |
| 413 | File upload exceeds 10 MB |
| 429 | Rate limit exceeded (60/minute per IP) |
| 500 | Pipeline processing failure, registry parse error |
| 503 | Backend integration unavailable |

---

## Source File

`cbam-pack-mvp/src/cbam_pack/web/app.py`
