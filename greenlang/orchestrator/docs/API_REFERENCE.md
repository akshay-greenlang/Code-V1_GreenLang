# GL-FOUND-X-001 GreenLang Orchestrator - API Reference

**Version:** 2.0.0
**Base URL:** `https://api.greenlang.io/v1/orchestrator`
**Last Updated:** 2026-01-27

---

## Overview

The GreenLang Orchestrator API provides programmatic access to submit, monitor, and manage pipeline executions. It supports the GLIP v1 (GreenLang Integration Protocol) for standardized agent execution with full provenance tracking.

**Key Features:**
- RESTful API design
- JWT Bearer token authentication (OAuth2)
- Hash-chained audit trail for all operations
- Real-time status updates via webhooks
- Comprehensive error responses with suggested fixes

---

## Authentication

All API endpoints require authentication using JWT Bearer tokens.

### Obtain Access Token

```http
POST /api/v1/auth/token
Content-Type: application/x-www-form-urlencoded
```

**Request Body:**

```
grant_type=client_credentials&
client_id=your_client_id&
client_secret=your_client_secret&
scope=orchestrator:read orchestrator:write
```

**Response (200 OK):**

```json
{
  "access_token": "eyJhbGciOiJSUzI1NiIsInR5cCI6IkpXVCJ9...",
  "token_type": "Bearer",
  "expires_in": 3600,
  "scope": "orchestrator:read orchestrator:write"
}
```

**Example (Python):**

```python
import requests

response = requests.post(
    "https://api.greenlang.io/v1/auth/token",
    data={
        "grant_type": "client_credentials",
        "client_id": "your_client_id",
        "client_secret": "your_client_secret",
        "scope": "orchestrator:read orchestrator:write"
    }
)

access_token = response.json()["access_token"]
headers = {"Authorization": f"Bearer {access_token}"}
```

### Required Scopes

| Scope | Description |
|-------|-------------|
| `orchestrator:read` | Read run status, artifacts, events |
| `orchestrator:write` | Submit runs, cancel runs |
| `orchestrator:admin` | Administrative operations |

---

## Rate Limiting

| Client Type | Requests/Minute | Burst |
|-------------|-----------------|-------|
| Authenticated | 100 | 150 |
| Unauthenticated | 10 | 15 |

**Rate Limit Headers:**

```http
X-RateLimit-Limit: 100
X-RateLimit-Remaining: 95
X-RateLimit-Reset: 1706371200
```

**Rate Limit Exceeded (429):**

```json
{
  "error": "rate_limit_exceeded",
  "message": "Rate limit exceeded. Try again in 42 seconds.",
  "retry_after": 42
}
```

---

## API Endpoints

### Runs

#### Submit Pipeline Run

Submit a new pipeline for execution.

```http
POST /api/v1/runs
Content-Type: application/json
Authorization: Bearer {access_token}
```

**Request Body:**

```json
{
  "pipeline_id": "carbon-emissions-calc",
  "pipeline_yaml": "base64-encoded-yaml-or-null",
  "pipeline_url": "s3://pipelines/carbon-emissions-calc/v1.0.0.yaml",
  "tenant_id": "acme-corp",
  "namespace": "production",
  "inputs": {
    "data_source": "s3://acme-data/emissions/2025-q4.csv",
    "reporting_period": "2025-Q4",
    "emission_factors_version": "ipcc-2024"
  },
  "config": {
    "trace_id": "abc123-def456",
    "dry_run": false,
    "skip_policy_check": false,
    "prefer_glip": true
  },
  "labels": {
    "environment": "production",
    "team": "sustainability"
  },
  "resources": {
    "default_cpu_limit": "2",
    "default_memory_limit": "4Gi",
    "default_timeout_seconds": 3600
  },
  "callbacks": {
    "on_complete": "https://acme.com/webhooks/greenlang",
    "on_failure": "https://acme.com/webhooks/greenlang-failure"
  }
}
```

**Request Parameters:**

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `pipeline_id` | string | Yes | Pipeline identifier |
| `pipeline_yaml` | string | No | Base64-encoded pipeline YAML |
| `pipeline_url` | string | No | URL to pipeline YAML (S3, HTTPS) |
| `tenant_id` | string | Yes | Tenant identifier |
| `namespace` | string | No | Execution namespace (default: "default") |
| `inputs` | object | No | Pipeline input parameters |
| `config.trace_id` | string | No | Distributed trace ID (auto-generated if not provided) |
| `config.dry_run` | boolean | No | Validate without executing (default: false) |
| `config.skip_policy_check` | boolean | No | Skip policy enforcement (dev only) |
| `config.prefer_glip` | boolean | No | Prefer GLIP v1 execution (default: true) |
| `labels` | object | No | Metadata labels |
| `resources` | object | No | Resource overrides |
| `callbacks.on_complete` | string | No | Webhook URL for completion |
| `callbacks.on_failure` | string | No | Webhook URL for failure |

*Note: Either `pipeline_yaml` or `pipeline_url` must be provided.*

**Response (202 Accepted):**

```json
{
  "run_id": "run_abc123def456",
  "pipeline_id": "carbon-emissions-calc",
  "status": "PENDING",
  "created_at": "2026-01-27T10:30:00Z",
  "plan_id": "plan_789xyz",
  "trace_id": "abc123-def456",
  "steps_total": 5,
  "estimated_duration_seconds": 1800,
  "links": {
    "self": "/api/v1/runs/run_abc123def456",
    "steps": "/api/v1/runs/run_abc123def456/steps",
    "events": "/api/v1/runs/run_abc123def456/events",
    "artifacts": "/api/v1/runs/run_abc123def456/artifacts",
    "cancel": "/api/v1/runs/run_abc123def456:cancel"
  }
}
```

**Error Responses:**

| Status | Code | Description |
|--------|------|-------------|
| 400 | `GL-E-YAML-INVALID` | Invalid pipeline YAML |
| 400 | `GL-E-DAG-CYCLE` | Circular dependency in pipeline |
| 400 | `GL-E-PARAM-MISSING` | Required parameter missing |
| 401 | `unauthorized` | Invalid or missing authentication |
| 403 | `GL-E-OPA-DENY` | Policy denied execution |
| 409 | `duplicate_run` | Duplicate idempotency key |
| 429 | `rate_limit_exceeded` | Rate limit exceeded |

**Example (Python):**

```python
import requests
import base64

# Load pipeline YAML
with open("pipeline.yaml", "r") as f:
    pipeline_yaml = base64.b64encode(f.read().encode()).decode()

response = requests.post(
    "https://api.greenlang.io/v1/orchestrator/runs",
    headers={"Authorization": f"Bearer {access_token}"},
    json={
        "pipeline_id": "carbon-emissions-calc",
        "pipeline_yaml": pipeline_yaml,
        "tenant_id": "acme-corp",
        "inputs": {
            "data_source": "s3://acme-data/emissions/2025-q4.csv"
        }
    }
)

run = response.json()
print(f"Run ID: {run['run_id']}")
print(f"Status: {run['status']}")
```

**Example (cURL):**

```bash
curl -X POST "https://api.greenlang.io/v1/orchestrator/runs" \
  -H "Authorization: Bearer $ACCESS_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "pipeline_id": "carbon-emissions-calc",
    "pipeline_url": "s3://pipelines/carbon-emissions-calc/v1.0.0.yaml",
    "tenant_id": "acme-corp",
    "inputs": {
      "data_source": "s3://acme-data/emissions/2025-q4.csv"
    }
  }'
```

---

#### Get Run Status

Retrieve the current status of a pipeline run.

```http
GET /api/v1/runs/{run_id}
Authorization: Bearer {access_token}
```

**Path Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `run_id` | string | Run identifier |

**Response (200 OK):**

```json
{
  "run_id": "run_abc123def456",
  "pipeline_id": "carbon-emissions-calc",
  "tenant_id": "acme-corp",
  "namespace": "production",
  "status": "RUNNING",
  "started_at": "2026-01-27T10:30:05Z",
  "completed_at": null,
  "duration_ms": null,
  "plan_id": "plan_789xyz",
  "trace_id": "abc123-def456",
  "progress": {
    "steps_total": 5,
    "steps_completed": 2,
    "steps_running": 1,
    "steps_pending": 2,
    "steps_failed": 0,
    "percent_complete": 40
  },
  "current_step": {
    "step_id": "step_calculate_scope1",
    "agent_id": "GL-MRV-X-001",
    "status": "RUNNING",
    "started_at": "2026-01-27T10:35:00Z"
  },
  "metadata": {
    "glip_version": "1.0.0",
    "input_hash": "sha256:abc123...",
    "labels": {
      "environment": "production",
      "team": "sustainability"
    }
  },
  "links": {
    "self": "/api/v1/runs/run_abc123def456",
    "steps": "/api/v1/runs/run_abc123def456/steps",
    "events": "/api/v1/runs/run_abc123def456/events",
    "artifacts": "/api/v1/runs/run_abc123def456/artifacts",
    "cancel": "/api/v1/runs/run_abc123def456:cancel"
  }
}
```

**Run Status Values:**

| Status | Description |
|--------|-------------|
| `PENDING` | Run submitted, awaiting scheduling |
| `RUNNING` | One or more steps currently executing |
| `SUCCESS` | All steps completed successfully |
| `FAILED` | One or more steps failed |
| `CANCELED` | Run was canceled by user |
| `TIMEOUT` | Run exceeded maximum duration |

---

#### List Runs

List pipeline runs with optional filtering.

```http
GET /api/v1/runs?status=RUNNING&pipeline_id=carbon-emissions-calc&limit=50
Authorization: Bearer {access_token}
```

**Query Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `status` | string | - | Filter by status |
| `pipeline_id` | string | - | Filter by pipeline ID |
| `tenant_id` | string | - | Filter by tenant |
| `namespace` | string | - | Filter by namespace |
| `created_after` | datetime | - | Filter by creation time |
| `created_before` | datetime | - | Filter by creation time |
| `page` | integer | 1 | Page number |
| `limit` | integer | 20 | Items per page (max: 100) |
| `sort` | string | `-created_at` | Sort field (prefix `-` for descending) |

**Response (200 OK):**

```json
{
  "items": [
    {
      "run_id": "run_abc123def456",
      "pipeline_id": "carbon-emissions-calc",
      "status": "RUNNING",
      "created_at": "2026-01-27T10:30:00Z",
      "progress": {
        "steps_total": 5,
        "steps_completed": 2,
        "percent_complete": 40
      }
    }
  ],
  "pagination": {
    "page": 1,
    "limit": 20,
    "total_items": 150,
    "total_pages": 8,
    "has_next": true,
    "has_prev": false
  }
}
```

---

#### Cancel Run

Cancel a running pipeline execution.

```http
POST /api/v1/runs/{run_id}:cancel
Authorization: Bearer {access_token}
Content-Type: application/json
```

**Request Body:**

```json
{
  "reason": "User requested cancellation",
  "force": false
}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `reason` | string | No | Cancellation reason (for audit) |
| `force` | boolean | No | Force immediate termination (default: false) |

**Response (200 OK):**

```json
{
  "run_id": "run_abc123def456",
  "status": "CANCELED",
  "canceled_at": "2026-01-27T10:45:00Z",
  "canceled_by": "user@acme.com",
  "reason": "User requested cancellation",
  "steps_canceled": 2,
  "steps_completed": 3
}
```

---

### Steps

#### Get Run Steps

Retrieve status of all steps in a run.

```http
GET /api/v1/runs/{run_id}/steps
Authorization: Bearer {access_token}
```

**Response (200 OK):**

```json
{
  "run_id": "run_abc123def456",
  "steps": [
    {
      "step_id": "step_data_ingestion",
      "agent_id": "GL-INGEST-X-001",
      "agent_version": "1.2.0",
      "status": "SUCCESS",
      "execution_mode": "glip_v1",
      "started_at": "2026-01-27T10:30:10Z",
      "completed_at": "2026-01-27T10:32:00Z",
      "duration_ms": 110000,
      "exit_code": 0,
      "dependencies": [],
      "artifacts": {
        "output": {
          "uri": "s3://artifacts/run_abc123/steps/step_data_ingestion/result.json",
          "checksum": "sha256:def456..."
        }
      }
    },
    {
      "step_id": "step_calculate_scope1",
      "agent_id": "GL-MRV-X-001",
      "agent_version": "2.0.0",
      "status": "RUNNING",
      "execution_mode": "glip_v1",
      "started_at": "2026-01-27T10:35:00Z",
      "completed_at": null,
      "duration_ms": null,
      "dependencies": ["step_data_ingestion"],
      "resource_usage": {
        "memory_current": "1.2Gi",
        "memory_limit": "4Gi",
        "cpu_current": "0.8"
      }
    },
    {
      "step_id": "step_calculate_scope2",
      "agent_id": "GL-MRV-X-002",
      "status": "PENDING",
      "execution_mode": "glip_v1",
      "dependencies": ["step_data_ingestion"]
    }
  ]
}
```

**Step Status Values:**

| Status | Description |
|--------|-------------|
| `PENDING` | Waiting for dependencies |
| `SCHEDULED` | Dependencies met, awaiting execution |
| `RUNNING` | Currently executing |
| `SUCCESS` | Completed successfully |
| `FAILED` | Execution failed |
| `SKIPPED` | Skipped due to upstream failure |
| `CANCELED` | Canceled by user |
| `TIMEOUT` | Exceeded timeout |

---

#### Get Step Details

Retrieve detailed information about a specific step.

```http
GET /api/v1/runs/{run_id}/steps/{step_id}
Authorization: Bearer {access_token}
```

**Response (200 OK):**

```json
{
  "run_id": "run_abc123def456",
  "step_id": "step_calculate_scope1",
  "agent_id": "GL-MRV-X-001",
  "agent_version": "2.0.0",
  "container_image": "ghcr.io/greenlang/agent-mrv:2.0.0",
  "status": "SUCCESS",
  "execution_mode": "glip_v1",
  "idempotency_key": "idem_xyz789",
  "started_at": "2026-01-27T10:35:00Z",
  "completed_at": "2026-01-27T10:40:00Z",
  "duration_ms": 300000,
  "exit_code": 0,
  "dependencies": ["step_data_ingestion"],
  "inputs": {
    "upstream_data": {
      "uri": "s3://artifacts/run_abc123/steps/step_data_ingestion/result.json",
      "checksum": "sha256:def456..."
    }
  },
  "outputs": {
    "scope1_emissions": 1234.56,
    "unit": "tCO2e",
    "methodology": "GHG Protocol"
  },
  "artifacts": {
    "result": {
      "uri": "s3://artifacts/run_abc123/steps/step_calculate_scope1/result.json",
      "checksum": "sha256:ghi789...",
      "size_bytes": 4096
    },
    "detailed_report": {
      "uri": "s3://artifacts/run_abc123/steps/step_calculate_scope1/artifacts/report.pdf",
      "checksum": "sha256:jkl012...",
      "size_bytes": 1048576,
      "media_type": "application/pdf"
    }
  },
  "metadata": {
    "peak_memory_bytes": 2147483648,
    "cpu_time_ms": 180000,
    "result_checksum": "sha256:ghi789...",
    "input_context_hash": "sha256:mno345..."
  },
  "resource_profile": {
    "cpu_request": "500m",
    "cpu_limit": "2",
    "memory_request": "1Gi",
    "memory_limit": "4Gi"
  },
  "k8s_metadata": {
    "job_name": "gl-step_calculate_scope1-a1b2c3d4",
    "pod_name": "gl-step_calculate_scope1-a1b2c3d4-xyz",
    "namespace": "greenlang-production",
    "node": "node-pool-1-abc123"
  }
}
```

---

#### Get Step Logs

Retrieve logs from a step's execution.

```http
GET /api/v1/runs/{run_id}/steps/{step_id}/logs
Authorization: Bearer {access_token}
```

**Query Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `tail` | integer | - | Number of lines to return from end |
| `since` | datetime | - | Return logs since timestamp |
| `follow` | boolean | false | Stream logs (SSE) |

**Response (200 OK):**

```json
{
  "run_id": "run_abc123def456",
  "step_id": "step_calculate_scope1",
  "logs": [
    {
      "timestamp": "2026-01-27T10:35:01Z",
      "level": "INFO",
      "message": "Agent GL-MRV-X-001 v2.0.0 starting..."
    },
    {
      "timestamp": "2026-01-27T10:35:02Z",
      "level": "INFO",
      "message": "Reading input from GL_INPUT_URI: s3://artifacts/run_abc123/steps/step_calculate_scope1/input.json"
    },
    {
      "timestamp": "2026-01-27T10:35:05Z",
      "level": "INFO",
      "message": "Processing 1000 emission records..."
    },
    {
      "timestamp": "2026-01-27T10:39:55Z",
      "level": "INFO",
      "message": "Calculation complete. Total: 1234.56 tCO2e"
    },
    {
      "timestamp": "2026-01-27T10:40:00Z",
      "level": "INFO",
      "message": "Writing result to GL_OUTPUT_URI"
    }
  ],
  "truncated": false
}
```

---

### Artifacts

#### Get Artifact Manifest

Retrieve the manifest of all artifacts for a run.

```http
GET /api/v1/runs/{run_id}/artifacts
Authorization: Bearer {access_token}
```

**Query Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `step_id` | string | - | Filter by step ID |

**Response (200 OK):**

```json
{
  "run_id": "run_abc123def456",
  "manifest_checksum": "sha256:abc123...",
  "total_size_bytes": 52428800,
  "artifact_count": 8,
  "artifacts": [
    {
      "artifact_id": "art_001",
      "step_id": "step_data_ingestion",
      "name": "result.json",
      "artifact_type": "RESULT",
      "uri": "s3://artifacts/run_abc123/steps/step_data_ingestion/result.json",
      "checksum": "sha256:def456...",
      "size_bytes": 4096,
      "media_type": "application/json",
      "created_at": "2026-01-27T10:32:00Z"
    },
    {
      "artifact_id": "art_002",
      "step_id": "step_calculate_scope1",
      "name": "report.pdf",
      "artifact_type": "DATA",
      "uri": "s3://artifacts/run_abc123/steps/step_calculate_scope1/artifacts/report.pdf",
      "checksum": "sha256:ghi789...",
      "size_bytes": 1048576,
      "media_type": "application/pdf",
      "created_at": "2026-01-27T10:40:00Z"
    }
  ]
}
```

**Artifact Types:**

| Type | Description |
|------|-------------|
| `INPUT_CONTEXT` | GLIP v1 input.json |
| `RESULT` | GLIP v1 result.json |
| `METADATA` | GLIP v1 step_metadata.json |
| `DATA` | Custom data artifacts |
| `ERROR` | Error information |

---

#### Get Artifact Download URL

Generate a pre-signed URL for artifact download.

```http
GET /api/v1/runs/{run_id}/artifacts/{artifact_id}/download
Authorization: Bearer {access_token}
```

**Query Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `expires_in` | integer | 3600 | URL expiration in seconds (max: 86400) |

**Response (200 OK):**

```json
{
  "artifact_id": "art_002",
  "download_url": "https://greenlang-artifacts.s3.amazonaws.com/run_abc123/steps/step_calculate_scope1/artifacts/report.pdf?X-Amz-Algorithm=...",
  "expires_at": "2026-01-27T11:40:00Z",
  "checksum": "sha256:ghi789...",
  "size_bytes": 1048576,
  "media_type": "application/pdf"
}
```

---

### Events (Audit Trail)

#### Get Run Events

Retrieve the hash-chained audit trail for a run.

```http
GET /api/v1/runs/{run_id}/events
Authorization: Bearer {access_token}
```

**Query Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `event_type` | string | - | Filter by event type |
| `step_id` | string | - | Filter by step ID |
| `since` | datetime | - | Events after timestamp |
| `limit` | integer | 100 | Max events to return |

**Response (200 OK):**

```json
{
  "run_id": "run_abc123def456",
  "chain_valid": true,
  "events": [
    {
      "event_id": "evt_001",
      "run_id": "run_abc123def456",
      "step_id": null,
      "event_type": "RUN_SUBMITTED",
      "timestamp": "2026-01-27T10:30:00Z",
      "payload": {
        "pipeline_id": "carbon-emissions-calc",
        "tenant_id": "acme-corp",
        "user_id": "user@acme.com"
      },
      "prev_event_hash": "genesis",
      "event_hash": "sha256:abc123..."
    },
    {
      "event_id": "evt_002",
      "run_id": "run_abc123def456",
      "step_id": null,
      "event_type": "PLAN_COMPILED",
      "timestamp": "2026-01-27T10:30:01Z",
      "payload": {
        "plan_id": "plan_789xyz",
        "step_count": 5
      },
      "prev_event_hash": "sha256:abc123...",
      "event_hash": "sha256:def456..."
    },
    {
      "event_id": "evt_003",
      "run_id": "run_abc123def456",
      "step_id": null,
      "event_type": "POLICY_EVALUATED",
      "timestamp": "2026-01-27T10:30:02Z",
      "payload": {
        "evaluation_point": "pre_run",
        "allowed": true,
        "policies_evaluated": ["yaml_rules", "opa"]
      },
      "prev_event_hash": "sha256:def456...",
      "event_hash": "sha256:ghi789..."
    },
    {
      "event_id": "evt_004",
      "run_id": "run_abc123def456",
      "step_id": "step_data_ingestion",
      "event_type": "STEP_STARTED",
      "timestamp": "2026-01-27T10:30:10Z",
      "payload": {
        "agent_id": "GL-INGEST-X-001",
        "execution_mode": "glip_v1"
      },
      "prev_event_hash": "sha256:ghi789...",
      "event_hash": "sha256:jkl012..."
    }
  ]
}
```

**Event Types:**

| Event Type | Description |
|------------|-------------|
| `RUN_SUBMITTED` | Run created and queued |
| `PLAN_COMPILED` | DAG compiled from pipeline |
| `POLICY_EVALUATED` | Governance policies checked |
| `RUN_STARTED` | Execution began |
| `STEP_READY` | Step dependencies satisfied |
| `STEP_STARTED` | Step execution began |
| `STEP_RETRIED` | Step retried after failure |
| `STEP_SUCCEEDED` | Step completed successfully |
| `STEP_FAILED` | Step failed (terminal) |
| `ARTIFACT_WRITTEN` | Output artifact stored |
| `RUN_SUCCEEDED` | All steps completed |
| `RUN_FAILED` | Run failed (terminal) |
| `RUN_CANCELED` | Run canceled by user |

---

#### Export Audit Package

Export the complete audit trail with integrity verification.

```http
GET /api/v1/runs/{run_id}/audit-package
Authorization: Bearer {access_token}
```

**Query Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `include_signature` | boolean | false | Include cryptographic signature |
| `format` | string | json | Output format (json, yaml) |

**Response (200 OK):**

```json
{
  "run_id": "run_abc123def456",
  "events": [...],
  "chain_valid": true,
  "exported_at": "2026-01-27T11:00:00Z",
  "package_hash": "sha256:xyz789...",
  "signature": null,
  "metadata": {
    "event_count": 15,
    "store_type": "postgresql",
    "hash_algorithm": "sha256"
  }
}
```

---

### Health & Metrics

#### Health Check

```http
GET /health/ready
```

**Response (200 OK):**

```json
{
  "status": "healthy",
  "version": "2.0.0",
  "checks": {
    "database": "ok",
    "redis": "ok",
    "s3": "ok",
    "k8s": "ok",
    "opa": "ok"
  },
  "timestamp": "2026-01-27T10:00:00Z"
}
```

#### Liveness Check

```http
GET /health/live
```

**Response (200 OK):**

```json
{
  "status": "alive",
  "timestamp": "2026-01-27T10:00:00Z"
}
```

#### Prometheus Metrics

```http
GET /metrics
```

Returns Prometheus-formatted metrics.

---

## Webhooks

### Configure Webhook

Register a webhook for run events.

```http
POST /api/v1/webhooks
Content-Type: application/json
Authorization: Bearer {access_token}
```

**Request Body:**

```json
{
  "url": "https://your-app.com/webhooks/greenlang",
  "events": ["run.completed", "run.failed", "step.failed"],
  "secret": "your_webhook_secret",
  "active": true,
  "filters": {
    "pipeline_id": "carbon-emissions-calc",
    "namespace": "production"
  }
}
```

**Response (201 Created):**

```json
{
  "webhook_id": "wh_abc123",
  "url": "https://your-app.com/webhooks/greenlang",
  "events": ["run.completed", "run.failed", "step.failed"],
  "active": true,
  "created_at": "2026-01-27T10:00:00Z"
}
```

### Webhook Payload

**Example (`run.completed`):**

```json
{
  "event": "run.completed",
  "timestamp": "2026-01-27T10:45:00Z",
  "data": {
    "run_id": "run_abc123def456",
    "pipeline_id": "carbon-emissions-calc",
    "status": "SUCCESS",
    "duration_ms": 900000,
    "steps_succeeded": 5,
    "steps_failed": 0,
    "final_outputs": {
      "total_emissions": 1234.56,
      "unit": "tCO2e"
    },
    "artifact_manifest_url": "https://api.greenlang.io/v1/orchestrator/runs/run_abc123def456/artifacts"
  },
  "signature": "sha256=abc123..."
}
```

### Verifying Webhook Signatures

```python
import hmac
import hashlib

def verify_webhook_signature(payload: bytes, signature: str, secret: str) -> bool:
    expected = "sha256=" + hmac.new(
        secret.encode(),
        payload,
        hashlib.sha256
    ).hexdigest()
    return hmac.compare_digest(expected, signature)
```

---

## Error Responses

All errors follow a consistent format:

```json
{
  "error": {
    "code": "GL-E-PARAM-MISSING",
    "class": "USER_CONFIG",
    "message": "Required parameter 'data_source' is missing for step 'data_ingestion'",
    "retry_policy": "NO_RETRY",
    "timestamp": "2026-01-27T10:30:00Z"
  },
  "context": {
    "run_id": "run_abc123def456",
    "step_id": "step_data_ingestion",
    "trace_id": "abc123-def456"
  },
  "details": {
    "param": "data_source",
    "step_id": "data_ingestion"
  },
  "suggested_fixes": [
    {
      "type": "PARAM_CHANGE",
      "field": "steps.data_ingestion.params.data_source",
      "description": "Add the required parameter",
      "doc_link": "https://docs.greenlang.io/pipelines/parameters"
    }
  ],
  "links": {
    "documentation": "https://docs.greenlang.io/errors/GL-E-PARAM-MISSING"
  }
}
```

**Error Classes:**

| Class | Description | Retry |
|-------|-------------|-------|
| `TRANSIENT` | Temporary failure (network, 5xx) | Yes (exponential backoff) |
| `RESOURCE` | Resource exhaustion (OOM, quota) | Yes (bounded) |
| `USER_CONFIG` | Invalid configuration | No (fix required) |
| `POLICY_DENIED` | Policy violation | No (approval required) |
| `AGENT_BUG` | Agent code error | No (code fix required) |
| `INFRASTRUCTURE` | Platform failure | Depends |

---

## SDK Libraries

Official SDKs are available:

**Python:**

```bash
pip install greenlang-sdk
```

```python
from greenlang import OrchestratorClient

client = OrchestratorClient(
    client_id="your_client_id",
    client_secret="your_client_secret",
    base_url="https://api.greenlang.io/v1"
)

# Submit a run
run = client.runs.submit(
    pipeline_id="carbon-emissions-calc",
    pipeline_url="s3://pipelines/carbon-emissions-calc/v1.0.0.yaml",
    tenant_id="acme-corp",
    inputs={
        "data_source": "s3://acme-data/emissions/2025-q4.csv"
    }
)

# Wait for completion
result = client.runs.wait(run.run_id, timeout=3600)

# Download artifacts
client.artifacts.download(run.run_id, "step_calculate_scope1", "report.pdf", "/tmp/report.pdf")
```

**JavaScript/TypeScript:**

```bash
npm install @greenlang/sdk
```

```typescript
import { OrchestratorClient } from '@greenlang/sdk';

const client = new OrchestratorClient({
  clientId: 'your_client_id',
  clientSecret: 'your_client_secret',
  baseUrl: 'https://api.greenlang.io/v1'
});

// Submit a run
const run = await client.runs.submit({
  pipelineId: 'carbon-emissions-calc',
  pipelineUrl: 's3://pipelines/carbon-emissions-calc/v1.0.0.yaml',
  tenantId: 'acme-corp',
  inputs: {
    dataSource: 's3://acme-data/emissions/2025-q4.csv'
  }
});

// Get status
const status = await client.runs.get(run.runId);
console.log(`Status: ${status.status}`);
```

**Go:**

```bash
go get github.com/greenlang/sdk-go
```

```go
package main

import (
    "context"
    "github.com/greenlang/sdk-go/orchestrator"
)

func main() {
    client := orchestrator.NewClient(
        orchestrator.WithClientCredentials("client_id", "client_secret"),
        orchestrator.WithBaseURL("https://api.greenlang.io/v1"),
    )

    run, err := client.Runs.Submit(context.Background(), &orchestrator.SubmitRequest{
        PipelineID:  "carbon-emissions-calc",
        PipelineURL: "s3://pipelines/carbon-emissions-calc/v1.0.0.yaml",
        TenantID:    "acme-corp",
        Inputs: map[string]interface{}{
            "data_source": "s3://acme-data/emissions/2025-q4.csv",
        },
    })
}
```

---

## OpenAPI Specification

The complete OpenAPI 3.0 specification is available at:

```
https://api.greenlang.io/v1/orchestrator/openapi.json
https://api.greenlang.io/v1/orchestrator/openapi.yaml
```

Interactive documentation (Swagger UI):

```
https://api.greenlang.io/v1/orchestrator/docs
```

---

## Support

- **Documentation:** https://docs.greenlang.io/orchestrator
- **API Status:** https://status.greenlang.io
- **Support:** support@greenlang.io
- **Community:** https://community.greenlang.io
- **GitHub Issues:** https://github.com/greenlang/orchestrator/issues

---

## Changelog

| Version | Date | Changes |
|---------|------|---------|
| 2.0.0 | 2026-01-27 | GLIP v1 support, hash-chained audit trail |
| 1.5.0 | 2025-10-15 | Added webhook support |
| 1.0.0 | 2025-06-01 | Initial release |
