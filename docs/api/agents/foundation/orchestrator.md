# AGENT-FOUND-001: DAG Orchestrator API Reference

**Agent ID:** AGENT-FOUND-001
**Service:** DAG Orchestrator / Control Plane
**Status:** Production Ready
**Source:** `greenlang/agents/foundation/orchestrator/api/`

The Orchestrator exposes three groups of endpoints across three routers:

1. **DAG Router** -- DAG CRUD, execution, checkpointing, monitoring (`/api/v1/orchestrator`)
2. **Control Plane Router** -- Pipeline management, run operations, agents (`/pipelines`, `/runs`, `/agents`)
3. **Approval Router** -- Signed approvals and attestations (`/approvals`, `/runs/{run_id}/...`)

---

## 1. DAG Router

**Base Path:** `/api/v1/orchestrator`
**Tag:** `orchestrator`
**Source file:** `dag_router.py`

### Endpoint Summary

| Method | Path | Summary | Status Codes |
|--------|------|---------|--------------|
| POST | `/dags` | Create DAG workflow | 201, 400, 503 |
| GET | `/dags` | List DAG workflows | 200, 503 |
| GET | `/dags/{dag_id}` | Get DAG workflow | 200, 404, 503 |
| PUT | `/dags/{dag_id}` | Update DAG workflow | 200, 400, 404, 503 |
| DELETE | `/dags/{dag_id}` | Delete DAG workflow | 200, 404, 503 |
| POST | `/dags/{dag_id}/validate` | Validate DAG | 200, 404, 503 |
| POST | `/dags/{dag_id}/execute` | Execute DAG | 202, 400, 404, 503 |
| GET | `/executions` | List executions | 200, 503 |
| GET | `/executions/{execution_id}` | Get execution details | 200, 404, 503 |
| GET | `/executions/{execution_id}/trace` | Get execution trace | 200, 404, 503 |
| POST | `/executions/{execution_id}/cancel` | Cancel execution | 200, 404, 503 |
| POST | `/executions/{execution_id}/resume` | Resume from checkpoint | 202, 400, 404, 503 |
| GET | `/executions/{execution_id}/provenance` | Get provenance chain | 200, 404, 503 |
| GET | `/checkpoints/{execution_id}` | Get checkpoints | 200, 503 |
| DELETE | `/checkpoints/{execution_id}` | Delete checkpoints | 200, 503 |
| GET | `/metrics` | Get orchestrator metrics | 200, 503 |
| POST | `/dags/import` | Import DAG from YAML | 201, 400, 503 |
| GET | `/dags/{dag_id}/export` | Export DAG to YAML | 200, 404, 503 |
| GET | `/dags/{dag_id}/visualize` | Get Mermaid visualization | 200, 404, 503 |
| GET | `/health` | Health check | 200 |

### Detailed Endpoints

#### POST /dags -- Create DAG Workflow

Create a new DAG workflow definition.

**Request Body:**

```json
{
  "name": "emissions-pipeline",
  "description": "GHG emissions calculation pipeline",
  "version": "1.0.0",
  "nodes": {
    "intake": {
      "agent_id": "data-intake",
      "depends_on": []
    },
    "calculate": {
      "agent_id": "ghg-calculator",
      "depends_on": ["intake"]
    }
  },
  "default_retry_policy": {
    "max_retries": 3,
    "backoff_seconds": 10
  },
  "default_timeout_policy": {
    "timeout_seconds": 300
  },
  "on_failure": "fail_fast",
  "max_parallel_nodes": 10,
  "metadata": {}
}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `name` | string | Yes | Workflow name |
| `description` | string | No | Workflow description (default: `""`) |
| `version` | string | No | Version string (default: `"1.0.0"`) |
| `nodes` | object | Yes | Node definitions keyed by node ID |
| `default_retry_policy` | object | No | DAG-level default retry policy |
| `default_timeout_policy` | object | No | DAG-level default timeout policy |
| `on_failure` | string | No | Failure strategy: `fail_fast` (default) |
| `max_parallel_nodes` | integer | No | Max parallel nodes, 1-500 (default: 10) |
| `metadata` | object | No | Arbitrary metadata |

**Response (201):**

```json
{
  "dag": {
    "dag_id": "dag_abc123",
    "name": "emissions-pipeline",
    "version": "1.0.0",
    "nodes": { ... },
    "created_at": "2026-04-04T10:00:00Z"
  },
  "message": "DAG created"
}
```

---

#### POST /dags/{dag_id}/execute -- Execute DAG

Start execution of a DAG workflow. Returns immediately with an execution ID.

**Request Body:**

```json
{
  "input_data": {
    "file_url": "s3://bucket/data.csv",
    "reporting_period": "2025-Q4"
  },
  "execution_options": {
    "priority": "high",
    "dry_run": false
  }
}
```

**Response (202):**

```json
{
  "execution_id": "exec_xyz789",
  "dag_id": "dag_abc123",
  "status": "running",
  "message": "Execution started"
}
```

---

#### GET /executions -- List Executions

List DAG executions with optional filters.

**Query Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `dag_id` | string | Filter by DAG ID |
| `status` | string | Filter by status |

**Response (200):**

```json
{
  "executions": [
    {
      "execution_id": "exec_xyz789",
      "dag_id": "dag_abc123",
      "status": "completed",
      "started_at": "2026-04-04T10:00:00Z",
      "completed_at": "2026-04-04T10:05:00Z"
    }
  ],
  "count": 1
}
```

---

#### POST /executions/{execution_id}/resume -- Resume From Checkpoint

Resume a failed execution from its last checkpoint.

**Request Body:**

```json
{
  "agent_registry": {
    "data-intake": "http://intake-service:8080"
  }
}
```

**Response (202):**

```json
{
  "execution_id": "exec_xyz789",
  "status": "running",
  "message": "Execution resumed"
}
```

---

#### GET /dags/{dag_id}/visualize -- Mermaid Visualization

Returns a Mermaid diagram string representing the DAG.

**Response (200):**

```json
{
  "dag_id": "dag_abc123",
  "mermaid": "graph TD\n    intake[intake\\ndata-intake]\n    calculate[calculate\\nghg-calculator]\n    intake --> calculate"
}
```

---

## 2. Control Plane Router

**Base Path:** Multiple prefixes (`/pipelines`, `/runs`, `/agents`)
**Source file:** `routes.py`
**Authentication:** `X-API-Key` header required

### Pipeline Management (`/pipelines`)

| Method | Path | Summary | Status Codes |
|--------|------|---------|--------------|
| POST | `/pipelines` | Register a pipeline | 201, 400, 409, 401 |
| GET | `/pipelines` | List pipelines | 200, 401 |
| GET | `/pipelines/{pipeline_id}` | Get pipeline details | 200, 404, 401 |
| DELETE | `/pipelines/{pipeline_id}` | Delete pipeline | 204, 404, 401 |

#### POST /pipelines -- Register Pipeline

**Request Body:**

```json
{
  "api_version": "v1",
  "kind": "Pipeline",
  "metadata": {
    "name": "cbam-reporting",
    "namespace": "compliance",
    "version": "1.0.0",
    "description": "CBAM quarterly reporting pipeline",
    "owner": "sustainability-team",
    "team": "compliance",
    "labels": { "region": "eu" },
    "tags": ["cbam", "quarterly"]
  },
  "spec": {
    "steps": [
      {
        "id": "intake",
        "agent_id": "data-intake",
        "depends_on": []
      },
      {
        "id": "calculate",
        "agent_id": "cbam-calculator",
        "depends_on": ["intake"]
      }
    ]
  }
}
```

**Response (201):**

```json
{
  "pipeline_id": "pipe_abc123",
  "name": "cbam-reporting",
  "namespace": "compliance",
  "version": "1.0.0",
  "step_count": 2,
  "content_hash": "sha256:a1b2c3...",
  "created_at": "2026-04-04T10:00:00Z",
  "updated_at": "2026-04-04T10:00:00Z"
}
```

---

#### GET /pipelines -- List Pipelines

**Query Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `namespace` | string | Filter by namespace |
| `owner` | string | Filter by owner |
| `team` | string | Filter by team |
| `label` | string | Filter by label (`key=value`) |
| `search` | string | Search in name/description |
| `offset` | integer | Pagination offset (default: 0) |
| `limit` | integer | Pagination limit (default: 100, max: 1000) |

---

### Run Operations (`/runs`)

| Method | Path | Summary | Status Codes |
|--------|------|---------|--------------|
| POST | `/runs` | Submit a run | 202, 400, 404, 403, 401 |
| GET | `/runs` | List runs | 200, 401 |
| GET | `/runs/{run_id}` | Get run details | 200, 404, 401 |
| POST | `/runs/{run_id}/cancel` | Cancel a run | 200, 404, 409, 401 |
| GET | `/runs/{run_id}/logs` | Get run logs | 200, 404, 401 |
| GET | `/runs/{run_id}/audit` | Get run audit trail | 200, 404, 401 |

#### POST /runs -- Submit a Run

**Request Body:**

```json
{
  "pipeline_id": "pipe_abc123",
  "tenant_id": "tenant_001",
  "user_id": "user@example.com",
  "parameters": {
    "reporting_period": "2025-Q4",
    "region": "EU"
  },
  "labels": { "priority": "high" },
  "priority": 5,
  "dry_run": false,
  "skip_policy_check": false,
  "timeout_seconds": 3600
}
```

**Response (202):**

```json
{
  "run_id": "run_xyz789",
  "pipeline_id": "pipe_abc123",
  "pipeline_name": "cbam-reporting",
  "tenant_id": "tenant_001",
  "status": "pending",
  "created_at": "2026-04-04T10:00:00Z",
  "progress_percent": 0.0,
  "steps_total": 2,
  "steps_completed": 0,
  "trace_id": "tr-abc123"
}
```

---

## 3. Approval Router

**Base Path:** `/approvals` and `/runs/{run_id}/...`
**Source file:** `approval_routes.py`
**Authentication:** API key required via `X-API-Key` header

### Endpoint Summary

| Method | Path | Summary | Status Codes |
|--------|------|---------|--------------|
| POST | `/approvals/{approval_id}/decide` | Submit approval decision | 200, 400, 404, 409 |
| GET | `/approvals/{approval_id}` | Get approval status | 200, 404 |
| GET | `/approvals/{approval_id}/verify` | Verify attestation signature | 200, 404 |
| GET | `/runs/{run_id}/approvals` | List pending approvals for run | 200, 404 |
| POST | `/runs/{run_id}/steps/{step_id}/approve` | Submit step approval | 200, 400, 404, 409 |

#### POST /approvals/{approval_id}/decide

Submit an approval decision with Ed25519 cryptographic signature.

**Request Body:**

```json
{
  "decision": "approved",
  "reason": "Data quality check passed",
  "approver_name": "Jane Smith",
  "approver_role": "Compliance Officer",
  "signature": "<base64-encoded-ed25519-private-key>",
  "public_key": "<base64-encoded-ed25519-public-key>"
}
```

**Response (200):**

```json
{
  "request_id": "apr_abc123",
  "status": "approved",
  "attestation": {
    "approver_id": "user_456",
    "approver_name": "Jane Smith",
    "approver_role": "Compliance Officer",
    "decision": "approved",
    "reason": "Data quality check passed",
    "timestamp": "2026-04-04T10:30:00Z",
    "signature": "c2lnbmF0dXJlX2hlcmVfYWJj...",
    "attestation_hash": "sha256:d4e5f6...",
    "signature_valid": true
  },
  "message": "Approval approved successfully recorded"
}
```

---

## Common Error Responses

**400 Bad Request:**

```json
{
  "error": "validation_error",
  "message": "File format not supported",
  "details": [
    {
      "code": "GL-E-VAL-003",
      "message": "File format not supported"
    }
  ],
  "trace_id": "tr-abc123",
  "timestamp": "2026-04-04T12:00:00Z"
}
```

**404 Not Found:**

```json
{
  "error": "not_found",
  "message": "DAG not found",
  "details": [
    {
      "code": "GL-E-RES-001",
      "message": "Pipeline with ID 'pipe_xxx' does not exist"
    }
  ],
  "trace_id": "tr-abc123"
}
```

**503 Service Unavailable:**

```json
{
  "detail": "DAG Orchestrator not configured"
}
```
