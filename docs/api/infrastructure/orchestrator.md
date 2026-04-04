# Orchestrator Control Plane API Reference (AGENT-FOUND-001)

## Overview

The GreenLang Orchestrator Control Plane provides REST endpoints for pipeline management, run operations, agent registry, approval workflows, health monitoring, and metrics. This is the central coordination API for all GreenLang DAG-based pipeline execution.

**Router Prefix:** (versioned under `/v1` or root-level)
**Tags:** `Pipeline Management`, `Run Operations`, `Agent Registry`, `Approvals`, `Quota Management`
**Source:** `greenlang/agents/foundation/orchestrator/api/routes.py`, `approval_routes.py`

---

## Endpoint Summary

### Pipeline Management

| Method | Path | Summary | Auth Required |
|--------|------|---------|---------------|
| POST | `/pipelines` | Register a pipeline | Yes |
| GET | `/pipelines` | List pipelines | Yes |
| GET | `/pipelines/{pipeline_id}` | Get pipeline details | Yes |
| DELETE | `/pipelines/{pipeline_id}` | Delete pipeline | Yes |

### Run Operations

| Method | Path | Summary | Auth Required |
|--------|------|---------|---------------|
| POST | `/runs` | Submit a pipeline run | Yes |
| GET | `/runs` | List runs | Yes |
| GET | `/runs/{run_id}` | Get run details | Yes |
| POST | `/runs/{run_id}/cancel` | Cancel a run | Yes |
| GET | `/runs/{run_id}/logs` | Get run logs | Yes |
| GET | `/runs/{run_id}/audit` | Get run audit trail | Yes |
| POST | `/runs/{run_id}/retry` | Retry a failed run (FR-074) | Yes |
| GET | `/runs/{run_id}/checkpoints` | Get checkpoint status (FR-074) | Yes |
| DELETE | `/runs/{run_id}/checkpoints` | Clear checkpoints (FR-074) | Yes |

### Approvals (FR-043)

| Method | Path | Summary | Auth Required |
|--------|------|---------|---------------|
| POST | `/approvals/{approval_id}/decide` | Submit approval decision | Yes |
| GET | `/approvals/{approval_id}` | Get approval status | Yes |
| GET | `/approvals/{approval_id}/verify` | Verify attestation signature | Yes |
| GET | `/runs/{run_id}/approvals` | List pending approvals for run | Yes |
| POST | `/runs/{run_id}/steps/{step_id}/approve` | Submit step approval | Yes |

### System

| Method | Path | Summary | Auth Required |
|--------|------|---------|---------------|
| GET | `/health` | Health check | No |
| GET | `/metrics` | Service metrics | No |
| GET | `/ready` | Readiness probe | No |

---

## Pipeline Management

### POST /pipelines

Register a new pipeline definition. The pipeline specification (DAG of steps) is validated and stored.

**Request Body:**

```json
{
  "pipeline_id": "cbam-intake-pipeline",
  "version": "1.2.0",
  "description": "CBAM data intake and validation pipeline",
  "steps": [
    {
      "step_id": "validate",
      "agent": "data-quality-profiler",
      "config": {"strict_mode": true},
      "depends_on": []
    },
    {
      "step_id": "normalize",
      "agent": "excel-csv-normalizer",
      "config": {},
      "depends_on": ["validate"]
    },
    {
      "step_id": "calculate",
      "agent": "cbam-calculation-engine",
      "config": {},
      "depends_on": ["normalize"]
    }
  ],
  "namespace": "production",
  "tags": {"app": "cbam", "team": "platform"}
}
```

**Response (201 Created):**

```json
{
  "pipeline_id": "cbam-intake-pipeline",
  "version": "1.2.0",
  "status": "registered",
  "steps_count": 3,
  "registered_at": "2026-04-04T12:00:00Z",
  "schema_hash": "sha256:abc123..."
}
```

**Error Responses:**

| Status Code | Error Code | Description |
|-------------|------------|-------------|
| 400 | GL-E-VAL-003 | Invalid pipeline definition |
| 409 | GL-E-OPS-001 | Pipeline already exists |

---

### POST /runs

Submit a new pipeline run. The orchestrator creates a DAG execution plan and begins processing.

**Request Body:**

```json
{
  "pipeline_id": "cbam-intake-pipeline",
  "inputs": {
    "file_url": "https://storage.greenlang.io/uploads/cbam-data.csv",
    "format": "CSV",
    "tenant_id": "t-acme-corp"
  },
  "priority": "normal",
  "namespace": "production",
  "idempotency_key": "run-20260404-001"
}
```

**Response (202 Accepted):**

```json
{
  "run_id": "run-uuid-001",
  "pipeline_id": "cbam-intake-pipeline",
  "status": "pending",
  "steps": [
    {"step_id": "validate", "status": "pending"},
    {"step_id": "normalize", "status": "pending"},
    {"step_id": "calculate", "status": "pending"}
  ],
  "submitted_at": "2026-04-04T12:00:00Z",
  "trace_id": "trace-uuid-001"
}
```

---

### GET /runs/{run_id}

Get detailed run status including step-level progress, timings, and outputs.

**Response (200 OK):**

```json
{
  "run_id": "run-uuid-001",
  "pipeline_id": "cbam-intake-pipeline",
  "status": "running",
  "steps": [
    {
      "step_id": "validate",
      "status": "succeeded",
      "started_at": "2026-04-04T12:00:01Z",
      "completed_at": "2026-04-04T12:00:15Z",
      "duration_ms": 14000,
      "output": {"records_valid": 985, "records_invalid": 15}
    },
    {
      "step_id": "normalize",
      "status": "running",
      "started_at": "2026-04-04T12:00:16Z",
      "completed_at": null,
      "duration_ms": null,
      "output": null
    },
    {
      "step_id": "calculate",
      "status": "pending",
      "started_at": null,
      "completed_at": null,
      "duration_ms": null,
      "output": null
    }
  ],
  "submitted_at": "2026-04-04T12:00:00Z",
  "started_at": "2026-04-04T12:00:01Z",
  "completed_at": null,
  "inputs": {"file_url": "...", "format": "CSV"},
  "trace_id": "trace-uuid-001"
}
```

---

### POST /runs/{run_id}/cancel

Cancel a running pipeline. Steps that have not yet started will be skipped.

**Request Body:**

```json
{
  "reason": "Data file contains incorrect column mapping, will resubmit.",
  "force": false
}
```

**Error Responses:**

| Status Code | Error Code | Description |
|-------------|------------|-------------|
| 400 | GL-E-OPS-002 | Run is not in a cancelable state |

---

### POST /runs/{run_id}/retry (FR-074)

Retry a failed run from the last checkpoint. Supports schema compatibility checks and idempotency validation.

**Request Body:**

```json
{
  "from_step": null,
  "force": false,
  "skip_non_idempotent_check": false
}
```

**Response (200 OK):**

```json
{
  "run_id": "run-uuid-001",
  "retry_run_id": "run-uuid-002",
  "checkpoint_step": "normalize",
  "steps_skipped": ["validate"],
  "steps_to_retry": ["normalize", "calculate"],
  "warnings": [],
  "message": "Retry initiated from checkpoint at step 'normalize'"
}
```

---

## Approval Endpoints (FR-043)

### POST /approvals/{approval_id}/decide

Submit an approval decision with Ed25519 cryptographic signature. The approval attestation is stored with a verifiable signature chain.

**Request Body:**

```json
{
  "decision": "approved",
  "reason": "Data validation passed manual review. Emission factors verified against EU reference data.",
  "approver_name": "Jane Smith",
  "approver_role": "compliance-officer",
  "signature": "base64-encoded-ed25519-private-key",
  "public_key": "base64-encoded-ed25519-public-key"
}
```

**Response (200 OK):**

```json
{
  "request_id": "apr-uuid-001",
  "status": "approved",
  "attestation": {
    "approver_id": "user-jane",
    "approver_name": "Jane Smith",
    "approver_role": "compliance-officer",
    "decision": "approved",
    "reason": "Data validation passed manual review...",
    "timestamp": "2026-04-04T14:00:00Z",
    "signature": "base64-truncated-signature...",
    "attestation_hash": "sha256:def456...",
    "signature_valid": true
  },
  "message": "Approval approved successfully recorded"
}
```

**Error Responses:**

| Status Code | Error Code | Description |
|-------------|------------|-------------|
| 404 | GL-E-APR-001 | Approval not found |
| 409 | GL-E-APR-002 | Approval already decided |
| 409 | GL-E-APR-003 | Approval expired |

---

### GET /approvals/{approval_id}/verify

Cryptographically verify the Ed25519 signature on an approval attestation.

**Response (200 OK):**

```json
{
  "approval_id": "apr-uuid-001",
  "signature_valid": true,
  "verified_at": "2026-04-04T14:05:00Z"
}
```

---

## System Endpoints

### GET /health

Health check for load balancers and monitoring. Returns component-level status.

**Response (200 OK):**

```json
{
  "status": "healthy",
  "version": "1.0.0",
  "components": [
    {"name": "orchestrator", "status": "healthy", "message": "Orchestrator is operational"},
    {"name": "event_store", "status": "healthy", "message": "Event store connected"},
    {"name": "policy_engine", "status": "healthy", "message": "Policy engine loaded"}
  ],
  "uptime_seconds": 86400.5,
  "timestamp": "2026-04-04T12:00:00Z"
}
```

---

## Error Code Reference

| Code | Description |
|------|-------------|
| GL-E-VAL-001 | Required parameter missing |
| GL-E-VAL-002 | Invalid parameter value |
| GL-E-VAL-003 | Invalid payload structure |
| GL-E-RES-001 | Pipeline not found |
| GL-E-RES-002 | Run not found |
| GL-E-RES-003 | Agent not found |
| GL-E-RES-005 | Checkpoint not found (FR-074) |
| GL-E-OPS-001 | Pipeline already exists |
| GL-E-OPS-002 | Run not in cancelable state |
| GL-E-OPS-003 | Policy violation |
| GL-E-OPS-004 | Run not in retryable state (FR-074) |
| GL-E-OPS-005 | Maximum retries exceeded (FR-074) |
| GL-E-AUTH-001 | Unauthorized (missing/invalid API key) |
| GL-E-AUTH-002 | Forbidden (insufficient permissions) |
| GL-E-SYS-001 | Internal server error |

## Run Status Values

```
pending -> running -> succeeded
                   \-> failed -> retrying (FR-074)
                   \-> cancelled
```

## Authentication

All endpoints require an `X-API-Key` header. Distributed tracing is supported via `X-Trace-ID` and `X-Correlation-ID` headers.
